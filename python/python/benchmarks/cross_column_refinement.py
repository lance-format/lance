# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Compare cross-column refinement with Python reranking and full-vector indices.

Run from python/ after the initial make install:
    uv run maturin develop --uv --profile release-with-debug
    uv run python python/benchmarks/cross_column_refinement.py \
        --build-label release-with-debug --output results.json

The synthetic vectors have decreasing coordinate variance. The coarse column
contains their first coordinates; no embedding model, PCA fit, or private data is
required. Queries are held out of the dataset. Results describe this synthetic
workload, not semantic relevance or production recall.
"""

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


def run(args, path):
    rng = np.random.default_rng(args.seed)
    full = rng.standard_normal(
        (args.rows + args.queries, args.dimensions), dtype=np.float32
    )
    full *= np.exp(-np.arange(args.dimensions) / (args.dimensions / 4))
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    queries = full[args.rows :].copy()
    full = full[: args.rows]
    short = np.ascontiguousarray(full[:, : args.coarse_dimensions])
    table = pa.table(
        {
            "full_vector": pa.FixedSizeListArray.from_arrays(
                pa.array(full.reshape(-1)), args.dimensions
            ),
            "short_vector": pa.FixedSizeListArray.from_arrays(
                pa.array(short.reshape(-1)), args.coarse_dimensions
            ),
        }
    )
    ds = lance.write_dataset(table, path, max_rows_per_file=2048)
    specs = {
        "full_flat": ("full_vector", "IVF_FLAT", "cosine", {}),
        "short_flat": ("short_vector", "IVF_FLAT", "l2", {}),
        "full_pq": (
            "full_vector",
            "IVF_PQ",
            "cosine",
            {
                "num_sub_vectors": args.dimensions // 16,
                "num_bits": 4,
            },
        ),
    }
    build_seconds = {}
    for name, (column, kind, metric, options) in specs.items():
        started = time.perf_counter()
        ds.create_index(
            column,
            kind,
            name=name,
            metric=metric,
            num_partitions=args.partitions,
            **options,
        )
        build_seconds[name] = time.perf_counter() - started
    ds = lance.dataset(path, index_cache_size_bytes=1024**3)
    segments = {
        index.name: [segment.uuid for segment in index.segments]
        for index in ds.describe_indices()
    }
    index_bytes = {
        name: sum(
            f.stat().st_size
            for segment in ids
            for f in (path / "_indices" / segment).rglob("*")
            if f.is_file()
        )
        for name, ids in segments.items()
    }

    def search(query, *, short_space=False, index=None, factor=None, cross=False):
        nearest = {
            "column": "short_vector" if short_space else "full_vector",
            "q": query[: args.coarse_dimensions] if short_space else query,
            "metric": "l2" if short_space else "cosine",
            "k": args.k,
            "nprobes": args.nprobes,
            "use_index": index is not None,
            "query_parallelism": 1,
        }
        if factor is not None:
            nearest["refine_factor"] = factor
        if cross:
            nearest.update(
                refine_column="full_vector", refine_q=query, refine_metric="cosine"
            )
        return ds.scanner(
            columns=["_distance"],
            with_row_id=True,
            nearest=nearest,
            index_segments=segments[index] if index else None,
        ).to_table()

    def python_refine(query):
        # Projection asks Lance to fetch final vectors only for coarse TopM.
        # This avoids an unnecessary second Python-to-Lance query round trip.
        candidates = ds.scanner(
            columns=["full_vector", "_distance"],
            with_row_id=True,
            index_segments=segments["short_flat"],
            nearest={
                "column": "short_vector",
                "q": query[: args.coarse_dimensions],
                "metric": "l2",
                "k": args.k * args.factor,
                "nprobes": args.nprobes,
                "query_parallelism": 1,
            },
        ).to_table()
        vectors = candidates["full_vector"].combine_chunks().values.to_numpy()
        vectors = vectors.reshape(-1, args.dimensions)
        distances = 1 - (vectors * query).sum(axis=1) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
        )
        row_ids = candidates["_rowid"].to_numpy()
        order = np.lexsort((row_ids, distances))[: args.k]
        return pa.table({"_rowid": row_ids[order], "_distance": distances[order]})

    paths = {
        "full_exact": lambda q: search(q),
        "short_exact": lambda q: search(q, short_space=True),
        "full_ivf_flat": lambda q: search(q, index="full_flat"),
        "short_ivf_flat": lambda q: search(q, short_space=True, index="short_flat"),
        "native_refine": lambda q: search(
            q, short_space=True, index="short_flat", factor=args.factor, cross=True
        ),
        "python_refine": python_refine,
        "full_ivf_pq": lambda q: search(q, index="full_pq"),
        "full_ivf_pq_refine": lambda q: search(q, index="full_pq", factor=args.factor),
    }
    ground_truth = [set(search(q)["_rowid"].to_pylist()) for q in queries]
    # Warm each path on the query set, then interleave timed paths in random order.
    for query in queries:
        native = paths["native_refine"](query)
        python = python_refine(query)
        assert native["_rowid"].to_pylist() == python["_rowid"].to_pylist()
        np.testing.assert_allclose(native["_distance"], python["_distance"], atol=2e-6)
        for name, execute in paths.items():
            if name not in {"native_refine", "python_refine"}:
                execute(query)
    observations = {name: [] for name in paths}
    for qi in rng.permutation(len(queries)):
        for name in rng.permutation(list(paths)):
            started = time.perf_counter()
            result = paths[name](queries[qi])
            elapsed_ms = (time.perf_counter() - started) * 1000
            ids = result["_rowid"].to_pylist()
            observations[name].append(
                {
                    "query": int(qi),
                    "elapsed_ms": elapsed_ms,
                    "recall": len(set(ids) & ground_truth[qi]) / args.k,
                }
            )
    summary = {}
    for name, samples in observations.items():
        latency = [sample["elapsed_ms"] for sample in samples]
        summary[name] = {
            "recall_at_k": float(np.mean([sample["recall"] for sample in samples])),
            "p50_ms": float(np.median(latency)),
            "p95_ms": float(np.percentile(latency, 95)),
            "sequential_qps": 1000 / float(np.mean(latency)),
        }
    return {
        "parameters": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "lance": lance.__version__,
        },
        "workload": "local, held-out synthetic queries, warmed paths, no scalar filter",
        "timing": (
            "query planning through Arrow results; "
            "excludes input generation and index construction"
        ),
        "native_matches_python": True,
        "index_configuration": specs,
        "index_bytes": index_bytes,
        "index_build_seconds": build_seconds,
        "summary": summary,
        "observations": observations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dimensions", type=int, default=1024)
    parser.add_argument("--coarse-dimensions", type=int, default=128)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--nprobes", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=734)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tmpdir", type=Path)
    parser.add_argument("--build-label", default="unspecified")
    args = parser.parse_args()
    if (
        min(
            args.rows,
            args.queries,
            args.dimensions,
            args.coarse_dimensions,
            args.partitions,
            args.nprobes,
            args.k,
            args.factor,
        )
        < 1
        or args.dimensions % 16
        or args.coarse_dimensions >= args.dimensions
        or args.nprobes > args.partitions
        or args.rows < max(256, args.k)
    ):
        parser.error(
            "Require positive counts, full dimensions divisible by 16, "
            "coarse dimensions < full dimensions, nprobes <= partitions, "
            "and rows >= max(256, k)"
        )
    with tempfile.TemporaryDirectory(
        prefix="lance-refinement-", dir=args.tmpdir
    ) as tmp:
        result = run(args, Path(tmp) / "vectors.lance")
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
