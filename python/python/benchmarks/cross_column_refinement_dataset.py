# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Read-only refinement comparison on an existing dataset and fixed queries.

The NPZ query file contains Float32 arrays ``full`` and ``coarse`` and UInt64
``rowids`` identifying the in-corpus query rows. Recall excludes each query row
before candidate selection. A separate unfiltered workload measures latency only.
No dataset or index is created or changed. All indices must cover the snapshot.

Run from python/ with an optimized extension already installed:
    uv run --no-sync python python/benchmarks/cross_column_refinement_dataset.py \
        --uri /path/to/data.lance --version 4 --queries queries.npz \
        --full-column full_vector --coarse-column short_vector \
        --full-index full_flat --coarse-index short_flat --output results.json

Results are relative to full-column exact cosine search, not relevance labels.
The JSON output includes row IDs and should be treated as dataset-derived data.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


def configurations(pairs):
    paths = [
        {"name": "full_exact", "kind": "exact", "space": "full"},
        {"name": "coarse_exact", "kind": "exact", "space": "coarse"},
    ]
    for space in ("full", "coarse"):
        paths.extend(
            {
                "name": f"{space}_ann_p{probes}",
                "kind": "ann",
                "space": space,
                "nprobes": probes,
            }
            for probes in sorted({probes for probes, _ in pairs})
        )
    for probes, factor in pairs:
        paths.extend(
            {
                "name": f"{kind}_p{probes}_f{factor}",
                "kind": kind,
                "space": "coarse",
                "nprobes": probes,
                "factor": factor,
            }
            for kind in ("native", "numpy")
        )
    return paths


def execute(ds, queries, segments, args, config, qi, exclude_self):
    space = config["space"]
    kind = config["kind"]
    nearest = {
        "column": args.full_column if space == "full" else args.coarse_column,
        "q": queries[space][qi],
        "k": args.k,
        "metric": "cosine" if space == "full" else "l2",
        "use_index": kind != "exact",
        "query_parallelism": 1,
    }
    if "nprobes" in config:
        nearest["nprobes"] = config["nprobes"]
    if kind == "native":
        nearest.update(
            refine_factor=config["factor"],
            refine_column=args.full_column,
            refine_q=queries["full"][qi],
            refine_metric="cosine",
        )
    elif kind == "numpy":
        nearest["k"] *= config["factor"]
    table = ds.scanner(
        columns=[args.full_column, "_distance"] if kind == "numpy" else ["_distance"],
        with_row_id=True,
        nearest=nearest,
        index_segments=segments[space] if kind != "exact" else None,
        filter=f"_rowid != {int(queries['rowids'][qi])}" if exclude_self else None,
        prefilter=True,
    ).to_table()
    if kind != "numpy":
        return table
    array = table[args.full_column].combine_chunks()
    vectors = array.values.to_numpy().reshape(len(array), array.type.list_size)
    query = queries["full"][qi]
    distances = 1 - (vectors * query).sum(axis=1) / (
        np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    )
    rowids = table["_rowid"].to_numpy()
    order = np.lexsort((rowids, distances))[: args.k]
    return pa.table({"_rowid": rowids[order], "_distance": distances[order]})


def summarize(records, truth, k):
    summary = {}
    for name, samples in records.items():
        elapsed = [s["elapsed_ms"] for s in samples]
        row = {
            "n": len(samples),
            "p50_ms": float(np.median(elapsed)),
            "p95_ms": float(np.percentile(elapsed, 95)),
            "mean_ms": float(np.mean(elapsed)),
            "sequential_qps": 1000 / float(np.mean(elapsed)),
        }
        if truth is not None:
            recalls = np.array(
                [len(set(s["rowids"]) & set(truth[s["query"]])) / k for s in samples]
            )
            rng = np.random.default_rng(91)
            boot = recalls[rng.integers(0, len(recalls), (2000, len(recalls)))].mean(
                axis=1
            )
            row.update(
                recall_at_k=float(recalls.mean()),
                recall_95ci=np.quantile(boot, [0.025, 0.975]).tolist(),
                top1_agreement=float(
                    np.mean([s["rowids"][0] == truth[s["query"]][0] for s in samples])
                ),
            )
        summary[name] = row
    return summary


def run(args):
    pairs = [tuple(map(int, pair.split(":"))) for pair in args.pairs.split(",")]
    if any(len(pair) != 2 or min(pair) < 1 for pair in pairs):
        raise ValueError("pairs must be positive nprobes:factor pairs")
    configs = configurations(pairs)
    with np.load(args.queries, allow_pickle=False) as archive:
        queries = {key: archive[key] for key in ("full", "coarse", "rowids")}
    count = len(queries["rowids"])
    if not count or any(len(queries[key]) != count for key in ("full", "coarse")):
        raise ValueError("Query arrays must have matching, nonzero lengths")
    for space in ("full", "coarse"):
        if queries[space].ndim != 2 or queries[space].dtype != np.float32:
            raise ValueError("Vector queries must be 2D Float32 arrays")
    ds = lance.dataset(
        args.uri,
        version=args.version,
        index_cache_size_bytes=args.cache_gib * 1024**3,
        metadata_cache_size_bytes=128 * 1024**2,
    )
    index_names = {"full": args.full_index, "coarse": args.coarse_index}
    described = {index.name: index for index in ds.describe_indices()}
    segments = {
        space: [segment.uuid for segment in described[name].segments]
        for space, name in index_names.items()
    }
    index_bytes = {}
    for space, name in index_names.items():
        statistics = ds.index_statistics(name)
        if statistics["num_unindexed_rows"] != 0:
            raise ValueError(f"Index {name} does not cover the complete snapshot")
        local_files = Path(args.uri) / "_indices"
        if local_files.is_dir():
            index_bytes[space] = sum(
                file.stat().st_size
                for segment in segments[space]
                for file in (local_files / segment).rglob("*")
                if file.is_file()
            )
        ds.prewarm_index(name)
    extension = Path(lance.__file__).parent / "lance.abi3.so"
    output = {
        "parameters": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "snapshot": {"version": ds.version, "rows": ds.count_rows()},
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "lance": lance.__version__,
            "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest()
            if extension.exists()
            else None,
        },
        "configurations": configs,
        "index_segments": segments,
        "index_bytes": index_bytes,
        "workloads": {},
    }
    for exclude_self in (True, False):
        mode = "self_excluded" if exclude_self else "unfiltered_latency_only"
        # Warm all indexed paths on the same fixed queries. Exact scans warm once.
        for config in configs:
            warm_count = min(count, 20) if config["kind"] != "exact" else 1
            for qi in range(warm_count):
                execute(ds, queries, segments, args, config, qi, exclude_self)
        records = {config["name"]: [] for config in configs}
        rng = np.random.default_rng(args.seed)
        for position, qi in enumerate(rng.permutation(count)):
            current = {}
            for ci in rng.permutation(len(configs)):
                config = configs[ci]
                started = time.perf_counter()
                result = execute(ds, queries, segments, args, config, qi, exclude_self)
                elapsed = (time.perf_counter() - started) * 1000
                ids = result["_rowid"].to_pylist()
                distances = result["_distance"].to_pylist()
                assert len(ids) == args.k
                if exclude_self:
                    assert int(queries["rowids"][qi]) not in ids
                sample = {
                    "query": int(qi),
                    "elapsed_ms": elapsed,
                    "rowids": ids,
                    "distances": distances,
                }
                records[config["name"]].append(sample)
                current[config["name"]] = sample
            for probes, factor in pairs:
                native = current[f"native_p{probes}_f{factor}"]
                numpy = current[f"numpy_p{probes}_f{factor}"]
                np.testing.assert_array_equal(native["rowids"], numpy["rowids"])
                np.testing.assert_allclose(
                    native["distances"], numpy["distances"], atol=2e-6
                )
            if position % 10 == 0:
                print(f"{mode}: {position + 1}/{count} queries", flush=True)
        truth = (
            {s["query"]: s["rowids"] for s in records["full_exact"]}
            if exclude_self
            else None
        )
        output["workloads"][mode] = {
            "summary": summarize(records, truth, args.k),
            "observations": records,
            "native_matches_numpy": True,
        }
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output["workloads"][mode]["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--full-column", required=True)
    parser.add_argument("--coarse-column", required=True)
    parser.add_argument("--full-index", required=True)
    parser.add_argument("--coarse-index", required=True)
    parser.add_argument("--pairs", default="4:2,16:2,16:10,64:50")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--cache-gib", type=int, default=8)
    parser.add_argument("--seed", type=int, default=734)
    parser.add_argument("--build-label", default="unspecified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.k, args.cache_gib) < 1:
        parser.error("k and cache-gib must be positive")
    run(args)


if __name__ == "__main__":
    main()
