# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Measure frozen large-k profiles and preserve returned IDs in binary arrays."""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
from common import DATASETS, emit, matrix, predict, save
from measure import (
    BaselineWorker,
    configure_override,
    fixed_budgets,
    profile_tuple,
    run_query,
)
from prepare_large import K_VALUES, sha256


def measure(root, name, args):
    out = root / name
    result = out / ("native" + (f"-{args.label}" if args.label else ""))
    result.mkdir(exist_ok=True)
    assert not list(result.glob("*.csv")), (
        "Preserve earlier measurements; use a new label"
    )
    calibration_path = root / "calibration-frozen.json"
    assert sha256(calibration_path) == sha256(root / "calibration.json")
    calibration = json.loads(calibration_path.read_text())
    metric = DATASETS[name]
    build = json.loads((out / "build.json").read_text())
    dataset = lance.dataset(
        root / "data" / name / "base.lance",
        version=build["version"],
        index_cache_size_bytes=200 * 1024**3,
    )
    queries = matrix(
        lance.dataset(root / "data" / name / "queries.lance").to_table()["vector"]
    )
    lookup = np.load(out / "row-ids.npy", mmap_mode="r")
    truth = np.load(out / "truth-ids.npy", mmap_mode="r")
    distances = np.load(out / "distances.npy", mmap_mode="r")
    ranks = np.load(out / "ranks.npy", mmap_mode="r")
    partition_rows = np.load(out / "scanned_rows.npy", mmap_mode="r")
    ids = np.load(out / "split.npz")[args.split]
    if args.limit:
        ids = ids[: args.limit]
    configure_override(None)
    binary = Path(lance.__file__).parent / "lance.abi3.so"
    candidate_sha = sha256(binary)
    assert candidate_sha == (root / "candidate-binary.sha256").read_text().split()[0]
    dataset.prewarm_index(build["index_name"])
    emit("prewarmed", dataset=name, candidate_sha256=candidate_sha)
    baseline = BaselineWorker(root, name, args.baseline_runtime)
    fields = [
        "ordinal",
        "query_id",
        "k",
        "policy",
        "phase",
        "latency_ms",
        "recall",
        "returned",
        "partitions",
        "comparisons",
        "bytes_read",
        "predicted_initial_partitions",
        "predicted_initial_rows",
        "route_recall",
    ]
    try:
        for k in args.k:
            shared = profile_tuple(calibration["profiles"][metric][str(k)])
            tuned = calibration["per_corpus"][name][str(k)]
            policies = [
                "auto",
                *[f"fixed{b}" for b in fixed_budgets(calibration, name, k)],
                "tuned",
                "legacy",
            ]
            budget = {
                "auto": predict(metric, distances, shared),
                "tuned": predict(metric, distances, profile_tuple(tuned)),
            }
            budget["legacy"] = np.maximum(
                (distances <= distances[:, :1] * np.float32(81.0)).sum(axis=1), 1
            )
            arrays = {
                policy: np.lib.format.open_memmap(
                    result / f"ids-{k}-{policy}.npy",
                    mode="w+",
                    dtype=np.int64,
                    shape=(
                        min(len(ids), args.legacy_queries)
                        if policy == "legacy"
                        else len(ids),
                        k,
                    ),
                )
                for policy in policies
            }

            def query(ordinal, query_id, active, phase):
                rows = []
                offset = ordinal % len(active)
                for policy in active[offset:] + active[:offset]:
                    if phase != "recall":
                        configure_override(tuned if policy == "tuned" else None)
                    if policy == "legacy":
                        found, elapsed, stats = baseline.query(query_id, k, None)
                    else:
                        found, elapsed, stats = run_query(
                            dataset,
                            queries[query_id],
                            k,
                            metric,
                            policy,
                            build["partitions"],
                            lookup,
                            None,
                        )
                    assert len(found) <= k
                    if not policy.startswith("fixed"):
                        assert len(found) == k, (name, k, policy, query_id, len(found))
                    initial = (
                        min(int(policy.removeprefix("fixed")), build["partitions"])
                        if policy.startswith("fixed")
                        else int(budget[policy][query_id])
                    )
                    if phase != "warmup":
                        arrays[policy][ordinal, : len(found)] = found
                        arrays[policy][ordinal, len(found) :] = -1
                    rows.append(
                        {
                            "ordinal": ordinal,
                            "query_id": int(query_id),
                            "k": k,
                            "policy": policy,
                            "phase": phase,
                            "latency_ms": elapsed,
                            "recall": len(
                                np.intersect1d(
                                    found, truth[query_id, :k], assume_unique=True
                                )
                            )
                            / k,
                            "returned": len(found),
                            "partitions": stats.all_counts["partitions_searched"],
                            "comparisons": stats.index_comparisons,
                            "bytes_read": stats.bytes_read,
                            "predicted_initial_partitions": initial,
                            "predicted_initial_rows": int(
                                partition_rows[query_id, initial - 1]
                            ),
                            "route_recall": float(
                                np.mean(ranks[query_id, :k] <= initial)
                            ),
                        }
                    )
                return rows

            with (result / f"records-{k}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for policy in policies:
                    query(0, ids[0], [policy], "warmup")
                timing_count = min(args.timing_queries, len(ids))
                for ordinal, query_id in enumerate(ids[:timing_count]):
                    active = [
                        p
                        for p in policies
                        if p != "legacy" or ordinal < args.legacy_queries
                    ]
                    writer.writerows(query(ordinal, query_id, active, "timing"))
                    if ordinal % 64 == 0:
                        stream.flush()
                        emit(
                            "timing",
                            dataset=name,
                            k=k,
                            completed=ordinal + 1,
                            total=timing_count,
                        )
                remaining = list(enumerate(ids[timing_count:], start=timing_count))
                for active in [
                    [p for p in policies if p not in ("tuned", "legacy")],
                    ["tuned"],
                ]:
                    configure_override(tuned if active == ["tuned"] else None)
                    with ThreadPoolExecutor(args.recall_workers) as pool:
                        for ordinal, rows in enumerate(
                            pool.map(
                                lambda item: query(item[0], item[1], active, "recall"),
                                remaining,
                            )
                        ):
                            writer.writerows(rows)
                            if ordinal % 256 == 0:
                                stream.flush()
                                emit(
                                    "recall",
                                    dataset=name,
                                    k=k,
                                    policies=active,
                                    completed=timing_count + ordinal + 1,
                                    total=len(ids),
                                )
                configure_override(None)
            for array in arrays.values():
                array.flush()
            arrays.clear()
            emit("measured", dataset=name, k=k)
    finally:
        configure_override(None)
        baseline.close()
    save(
        result / "identity.json",
        {
            "dataset": name,
            "k": args.k,
            "split": args.split,
            "query_count": len(ids),
            "timing_queries": args.timing_queries,
            "legacy_queries": args.legacy_queries,
            "candidate_binary_sha256": candidate_sha,
            "baseline_binary_sha256": baseline.binary_sha256,
            "calibration_sha256": sha256(calibration_path),
            "prepared_sha256": sha256(out / "prepared.json"),
            "split_sha256": sha256(out / "split.npz"),
            "harness_sha256": {
                filename: sha256(Path(__file__).with_name(filename))
                for filename in [
                    "measure_large.py",
                    "measure.py",
                    "common.py",
                    "prepare_large.py",
                ]
            },
            "files_sha256": {
                path.name: sha256(path)
                for path in sorted(result.iterdir())
                if path.suffix in (".npy", ".csv")
            },
        },
    )
    emit("complete", dataset=name, output=str(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--baseline-runtime", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", choices=K_VALUES, default=K_VALUES)
    parser.add_argument("--timing-queries", type=int, default=512)
    parser.add_argument("--legacy-queries", type=int, default=32)
    parser.add_argument("--recall-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--split", choices=["calibration", "evaluation"], default="evaluation"
    )
    args = parser.parse_args()
    assert args.legacy_queries <= args.timing_queries
    measure(args.root, args.dataset, args)
