# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Build frozen-membership RQ5 indices and audit paired Auto-probe recall."""

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from common import DATASETS, emit, matrix, positions, row_ids_by_position, save, summary
from lance.dataset import VectorIndexReader
from lance.lance import indices
from measure import configure_override, run_query
from prepare_large import sha256

K_VALUES = [1, 10, 100, 200, 500, 1000]
ARMS = ["flat_auto", "rq_auto", "rq_all", "rq_legacy"]


def prepare(source, root, name):
    previous = source / name
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prepared.json").exists():
        raise FileExistsError(f"Preserve the completed RQ5 preparation: {out}")
    build = json.loads((previous / "build.json").read_text())
    source_manifest = json.loads((previous / "prepared.json").read_text())
    for filename in ["centroids.npy", "membership.npy", "row-ids.npy", "truth-ids.npy"]:
        assert sha256(previous / filename) == source_manifest["sha256"][filename], (
            filename
        )
    assert sha256(previous / "split.npz") == source_manifest["split_sha256"]
    assert sha256(previous / "build.json") == source_manifest["build_sha256"]
    original = lance.dataset(
        source / "data" / name / "base.lance", version=build["version"]
    )
    assert original.stats.index_stats(build["index_name"])["index_type"] == "IVF_FLAT"
    centroids = np.load(previous / "centroids.npy")
    membership = np.load(previous / "membership.npy", mmap_mode="r")
    lookup = np.load(previous / "row-ids.npy", mmap_mode="r")
    assert len(lookup) == len(membership) == build["rows"]
    assert np.array_equal(
        matrix(original.centroids(index_name=build["index_name"])), centroids
    )
    clone_path = out / "rq5.lance"
    assert not clone_path.exists(), "Do not overwrite an earlier RQ5 index"
    clone = original.shallow_clone(clone_path, build["version"])
    assert np.array_equal(row_ids_by_position(clone), lookup)
    assignments = out / "assignments.lance"
    lance.write_dataset(
        pa.table(
            {
                "row_id": pa.array(lookup, type=pa.uint64()),
                "partition": pa.array(membership, type=pa.uint32()),
            }
        ),
        assignments,
        max_rows_per_file=build["rows"],
    )
    started = time.monotonic()
    last = [0.0]

    def progress(event):
        now = time.monotonic()
        if event.event != "progress" or now - last[0] >= 30:
            emit("rq5_build_progress", dataset=name, progress=dataclasses.asdict(event))
            last[0] = now

    model = indices.build_rq_model(dimension=centroids.shape[1], num_bits=5)
    (out / "rabitq-model.json").write_text(model + "\n")
    clone.create_index(
        "vector",
        "IVF_RQ",
        name=build["index_name"],
        replace=True,
        metric=DATASETS[name],
        num_bits=5,
        rabitq_model=model,
        num_partitions=build["partitions"],
        ivf_centroids=centroids,
        precomputed_partition_dataset=str(assignments),
        progress_callback=progress,
    )
    stats = clone.stats.index_stats(build["index_name"])
    assert stats["index_type"] == "IVF_RQ"
    assert stats["num_segments"] == 1
    assert (
        stats["num_indexed_rows"] == build["rows"] and stats["num_unindexed_rows"] == 0
    )
    assert stats["indices"][0]["sub_index"]["num_bits"] == 5
    assert stats["indices"][0]["sub_index"]["packed"] is True
    for segment in stats["indices"]:
        # The complete centroid matrix is already verified and hashed separately.
        segment.pop("centroids", None)
    assert np.array_equal(
        matrix(clone.centroids(index_name=build["index_name"])), centroids
    )
    observed = np.full(build["rows"], -1, dtype=np.int32)
    reader = VectorIndexReader(clone, build["index_name"])
    for partition in range(reader.num_partitions()):
        ids = reader.read_partition(partition)["_rowid"].to_numpy()
        rows = positions(ids, lookup)
        assert len(np.unique(rows)) == len(rows)
        assert np.all(observed[rows] == -1)
        observed[rows] = partition
        if partition % 1024 == 0:
            emit("rq5_membership_audit", dataset=name, partition=partition)
    assert np.array_equal(observed, membership)
    rq_indices = clone.list_indices()
    assert len(rq_indices) == 1
    index_dir = clone_path / "_indices" / rq_indices[0]["uuid"]
    index_hashes = {
        str(path.relative_to(clone_path)): sha256(path)
        for path in sorted(index_dir.rglob("*"))
        if path.is_file()
    }
    assert index_hashes
    save(
        out / "prepared.json",
        {
            "source": str(source.resolve()),
            "corpus": name,
            "metric": DATASETS[name],
            "source_build": build,
            "rq_version": clone.version,
            "rq_indices": rq_indices,
            "rq_stats": stats,
            "rabitq_model_sha256": sha256(out / "rabitq-model.json"),
            "index_sha256": index_hashes,
            "input_sha256": {
                filename: sha256(previous / filename)
                for filename in [
                    "build.json",
                    "prepared.json",
                    "centroids.npy",
                    "membership.npy",
                    "row-ids.npy",
                    "split.npz",
                ]
            },
            "membership_equal": True,
            "centroids_equal": True,
            "build_seconds": time.monotonic() - started,
            "binary_sha256": sha256(Path(lance.__file__).parent / "lance.abi3.so"),
        },
    )
    emit("rq5_prepared", dataset=name, seconds=time.monotonic() - started)


def inputs(root, name):
    out = root / name
    prepared = json.loads((out / "prepared.json").read_text())
    assert sha256(out / "rabitq-model.json") == prepared["rabitq_model_sha256"]
    source = Path(prepared["source"])
    previous = source / name
    for filename, digest in prepared["input_sha256"].items():
        assert sha256(previous / filename) == digest, filename
    build = prepared["source_build"]
    rq = lance.dataset(
        out / "rq5.lance",
        version=prepared["rq_version"],
        index_cache_size_bytes=32 * 1024**3,
    )
    queries = matrix(
        lance.dataset(source / "data" / name / "queries.lance").to_table()["vector"]
    )
    lookup = np.load(previous / "row-ids.npy", mmap_mode="r")
    return prepared, previous, build, rq, queries, lookup


def baseline_worker(root, name):
    _, _, build, rq, queries, lookup = inputs(root, name)
    configure_override(None)
    rq.prewarm_index(build["index_name"])
    print(
        json.dumps({"sha256": sha256(Path(lance.__file__).parent / "lance.abi3.so")}),
        flush=True,
    )
    for line in sys.stdin:
        query_id, k = json.loads(line)
        found, _, stats = run_query(
            rq,
            queries[query_id],
            k,
            DATASETS[name],
            "auto",
            build["partitions"],
            lookup,
            None,
        )
        print(
            json.dumps(
                {
                    "ids": found.tolist(),
                    "partitions": stats.all_counts["partitions_searched"],
                    "comparisons": stats.index_comparisons,
                }
            ),
            flush=True,
        )


def measure(root, name, args):
    prepared, previous, build, rq, queries, lookup = inputs(root, name)
    source = Path(prepared["source"])
    flat = lance.dataset(
        source / "data" / name / "base.lance",
        version=build["version"],
        index_cache_size_bytes=200 * 1024**3,
    )
    split = np.load(previous / "split.npz")
    query_ids = split[args.split][: args.queries]
    truth = np.load(previous / "truth-ids.npy", mmap_mode="r")
    out = root / name / ("native" if not args.label else f"native-{args.label}")
    out.mkdir(parents=True, exist_ok=True)
    assert not list(out.glob("*.csv")), "Preserve earlier measurements"
    configure_override(None)
    binary_sha = sha256(Path(lance.__file__).parent / "lance.abi3.so")
    assert binary_sha == (root / "candidate-binary.sha256").read_text().split()[0]
    assert binary_sha == prepared["binary_sha256"]
    for dataset in [flat, rq]:
        dataset.prewarm_index(build["index_name"])
    emit(
        "rq5_prewarmed", dataset=name, queries=len(query_ids), binary_sha256=binary_sha
    )

    # Verify that bounded Auto reproduces the actual pre-change RQ Auto path.
    environment = {**os.environ, "PYTHONPATH": str(args.baseline_runtime.resolve())}
    process = subprocess.Popen(
        [sys.executable, __file__, "baseline-worker", str(root), name],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        baseline_sha = json.loads(process.stdout.readline())["sha256"]
        assert baseline_sha == (root / "baseline-binary.sha256").read_text().split()[0]
        checks = []
        for k in args.k:
            for query_id in split["calibration"][:2]:
                process.stdin.write(json.dumps([int(query_id), k]) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                assert line, f"Baseline exited: {process.poll()}"
                reference = json.loads(line)
                found, _, stats = run_query(
                    rq,
                    queries[query_id],
                    k,
                    DATASETS[name],
                    "legacy",
                    build["partitions"],
                    lookup,
                    None,
                )
                assert reference["ids"] == found.tolist()
                assert (
                    reference["partitions"] == stats.all_counts["partitions_searched"]
                )
                assert reference["comparisons"] == stats.index_comparisons
                checks.append({"query_id": int(query_id), "k": k})
        save(
            out / "baseline-equivalence.json",
            {"binary_sha256": baseline_sha, "checks": checks},
        )
    finally:
        process.stdin.close()
        assert process.wait(timeout=60) == 0

    fields = [
        "ordinal",
        "query_id",
        "k",
        "arm",
        "recall",
        "returned",
        "partitions",
        "comparisons",
        "bytes_read",
        "diagnostic_latency_ms",
    ]
    for k in args.k:
        arrays = {
            arm: np.lib.format.open_memmap(
                out / f"ids-{k}-{arm}.npy",
                mode="w+",
                dtype=np.int64,
                shape=(len(query_ids), k),
            )
            for arm in ARMS
        }

        def query(item):
            ordinal, query_id = item
            rows = []
            arms = ARMS[ordinal % len(ARMS) :] + ARMS[: ordinal % len(ARMS)]
            for arm in arms:
                policy = (
                    f"fixed{build['partitions']}"
                    if arm == "rq_all"
                    else ("legacy" if arm == "rq_legacy" else "auto")
                )
                found, elapsed, stats = run_query(
                    flat if arm == "flat_auto" else rq,
                    queries[query_id],
                    k,
                    DATASETS[name],
                    policy,
                    build["partitions"],
                    lookup,
                    None,
                )
                assert len(found) == k
                arrays[arm][ordinal] = found
                count = stats.all_counts["partitions_searched"]
                if arm == "rq_all":
                    assert count == build["partitions"]
                rows.append(
                    {
                        "ordinal": ordinal,
                        "query_id": int(query_id),
                        "k": k,
                        "arm": arm,
                        "recall": len(np.intersect1d(found, truth[query_id, :k])) / k,
                        "returned": len(found),
                        "partitions": count,
                        "comparisons": stats.index_comparisons,
                        "bytes_read": stats.bytes_read,
                        "diagnostic_latency_ms": elapsed,
                    }
                )
            counts = {row["arm"]: row["partitions"] for row in rows}
            assert counts["rq_auto"] == counts["flat_auto"], (name, query_id, k, counts)
            return rows

        with (
            (out / f"queries-{k}.csv").open("w") as stream,
            ThreadPoolExecutor(args.workers) as pool,
        ):
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for ordinal, rows in enumerate(pool.map(query, enumerate(query_ids))):
                writer.writerows(rows)
                if (ordinal + 1) % 32 == 0 or ordinal + 1 == len(query_ids):
                    stream.flush()
                    emit("rq5_measured", dataset=name, k=k, completed=ordinal + 1)
        for array in arrays.values():
            array.flush()
    save(
        out / "complete.json",
        {
            "query_ids": query_ids.tolist(),
            "k": args.k,
            "split": args.split,
            "candidate_sha256": binary_sha,
            "baseline_sha256": baseline_sha,
            "workers": args.workers,
            "approx_mode": "normal",
            "refine_factor": None,
            "prepared_sha256": sha256(root / name / "prepared.json"),
        },
    )


def audit(root, names, label):
    results = []
    for name in names:
        prepared = json.loads((root / name / "prepared.json").read_text())
        previous = Path(prepared["source"]) / name
        truth = np.load(previous / "truth-ids.npy", mmap_mode="r")
        directory = root / name / ("native" if not label else f"native-{label}")
        complete = json.loads((directory / "complete.json").read_text())
        assert complete["prepared_sha256"] == sha256(root / name / "prepared.json")
        assert (
            complete["candidate_sha256"]
            == (root / "candidate-binary.sha256").read_text().split()[0]
        )
        ids = complete["query_ids"]
        split = np.load(previous / "split.npz")
        assert np.array_equal(ids, split[complete["split"]][: len(ids)])
        assert not set(split["calibration"].tolist()) & set(
            split["evaluation"].tolist()
        )
        for k in complete["k"]:
            rows = list(csv.DictReader((directory / f"queries-{k}.csv").open()))
            assert len(rows) == len(ids) * len(ARMS)
            arrays = {
                arm: np.load(directory / f"ids-{k}-{arm}.npy", mmap_mode="r")
                for arm in ARMS
            }
            recalls = {}
            for arm, returned in arrays.items():
                assert returned.shape == (len(ids), k)
                arm_rows = [row for row in rows if row["arm"] == arm]
                assert [int(row["query_id"]) for row in arm_rows] == ids
                values = []
                for ordinal, query_id in enumerate(ids):
                    found = returned[ordinal]
                    assert len(np.unique(found)) == k
                    assert np.all(
                        (found >= 0) & (found < prepared["source_build"]["rows"])
                    )
                    value = (
                        len(set(found.tolist()) & set(truth[query_id, :k].tolist())) / k
                    )
                    assert abs(value - float(arm_rows[ordinal]["recall"])) < 1e-12
                    assert int(arm_rows[ordinal]["returned"]) == k
                    assert int(arm_rows[ordinal]["bytes_read"]) == 0
                    if arm == "rq_all":
                        assert (
                            int(arm_rows[ordinal]["partitions"])
                            == prepared["source_build"]["partitions"]
                        )
                    values.append(value)
                recalls[arm] = np.asarray(values)
                results.append(
                    {
                        "corpus": name,
                        "metric": DATASETS[name],
                        "k": k,
                        "arm": arm,
                        "queries": len(ids),
                        "recall": summary(values),
                        "minimum_recall": min(values),
                        "partitions": summary(
                            [int(row["partitions"]) for row in arm_rows]
                        ),
                        "comparisons": summary(
                            [int(row["comparisons"]) for row in arm_rows]
                        ),
                    }
                )
            overlap = [
                len(set(a.tolist()) & set(b.tolist())) / k
                for a, b in zip(arrays["rq_auto"], arrays["rq_all"])
            ]
            paired = {
                "rq_auto_minus_flat_auto_pp": summary(
                    100 * (recalls["rq_auto"] - recalls["flat_auto"])
                ),
                "rq_auto_minus_rq_all_pp": summary(
                    100 * (recalls["rq_auto"] - recalls["rq_all"])
                ),
                "rq_auto_overlap_rq_all": summary(overlap),
            }
            for result in results[-len(ARMS) :]:
                result["paired"] = paired
    save(
        root
        / ("audited-results.json" if not label else f"audited-results-{label}.json"),
        results,
    )
    emit("rq5_audit_complete", groups=len(results), all_integrity_checks_passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "measure", "baseline-worker", "audit"]
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("name", choices=[*DATASETS, "all"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--baseline-runtime", type=Path)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k", type=int, nargs="+", default=K_VALUES)
    parser.add_argument(
        "--split", choices=["calibration", "evaluation"], default="evaluation"
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    names = list(DATASETS) if args.name == "all" else [args.name]
    if args.command == "audit":
        audit(args.root, names, args.label)
    else:
        for name in names:
            if args.command == "prepare":
                assert args.source
                prepare(args.source, args.root, name)
            elif args.command == "measure":
                assert args.baseline_runtime
                measure(args.root, name, args)
            else:
                baseline_worker(args.root, name)
