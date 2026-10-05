# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Measure frozen probing policies with a warm cache and serial paired queries."""

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
from calibrate import summary
from prepare import emit, matrix, positions, save


def predict(scores, profile):
    scores = scores.astype(np.float64)
    scale = (
        np.abs(scores[:, 0])
        if profile["mode"] == "relative_score"
        else scores.std(axis=1)
    )
    counts = ((scores[:, :1] - scores) <= profile["margin"] * scale[:, None]).sum(
        axis=1
    )
    return np.clip(counts, profile["floor"], min(profile["cap"], scores.shape[1]))


def configure_override(profile):
    for variable, field in [
        ("LANCE_AUTO_PROBE_MARGIN", "margin"),
        ("LANCE_AUTO_MIN_INITIAL_NPROBES", "floor"),
        ("LANCE_AUTO_MAX_INITIAL_NPROBES", "cap"),
    ]:
        if profile is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = str(profile[field])


def run_query(dataset, vector, k, policy, partitions, fragment_offsets):
    nearest = {
        "column": "vector",
        "q": vector,
        "k": k,
        "metric": "dot",
        "query_parallelism": 1,
    }
    if policy.startswith("fixed"):
        nearest["nprobes"] = int(policy.removeprefix("fixed"))
    elif policy == "legacy":
        # Explicit upper bound keeps the exact main legacy policy on the candidate
        # binary. The full partition count leaves its actual budget unchanged.
        nearest["maximum_nprobes"] = partitions
    captured = []
    start = time.perf_counter_ns()
    table = dataset.scanner(
        columns=["_distance"],
        with_row_id=True,
        nearest=nearest,
        scan_stats_callback=captured.append,
    ).to_table()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    assert len(table) == k
    assert len(captured) == 1
    stats = captured[0]
    ids = positions(table["_rowid"].to_numpy(), fragment_offsets)
    assert len(np.unique(ids)) == k
    return ids, elapsed, stats


def measure(root, name, limit, native, timing_queries, legacy_queries, recall_workers):
    directory = root / "data" / name
    out = root / name
    calibration = json.loads((root / "calibration.json").read_text())
    build = json.loads((out / "build.json").read_text())
    dataset = lance.dataset(
        directory / "base.lance",
        version=build["version"],
        index_cache_size_bytes=128 * 1024**3,
    )
    queries = matrix(lance.dataset(directory / "queries.lance").to_table()["vector"])
    truth = matrix(
        lance.dataset(directory / "ground_truth.lance").to_table()["neighbor_ids"]
    )
    ids = np.load(out / "split.npz")["evaluation"]
    if limit:
        ids = ids[:limit]
    timing_ids = set(ids[:timing_queries].tolist())
    routes = np.load(out / "routes.npz")
    scores = routes["scores"]
    scanned_rows = routes["scanned_rows"]
    ranks = routes["ranks"]
    fragment_offsets = np.load(out / "fragment-offsets.npy")
    started = time.monotonic()
    dataset.prewarm_index("dot_flat")
    emit("prewarmed", dataset=name, seconds=time.monotonic() - started)
    records = []
    results = {}
    suffix = "native" if native else "baseline-audit"
    output = out / f"measure-{suffix}.csv"
    with output.open("w") as stream:
        fields = [
            "query_id",
            "k",
            "policy",
            "latency_ms",
            "recall",
            "partitions",
            "comparisons",
            "bytes_read",
            "predicted_partitions",
            "predicted_rows",
            "route_recall",
            "neighbor_ids",
            "phase",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for k in [1, 10, 100]:
            profile = calibration["selected"][str(k)]
            tuned = calibration["profiles"][str(k)][calibration["selected_mode"]][name]
            configure_override(None)
            fixed = calibration["fixed"][name][str(k)]
            meeting = next(
                index for index, row in enumerate(fixed) if row["recall"] >= 0.95
            )
            budgets = [
                row["nprobes"] for row in fixed[max(0, meeting - 1) : meeting + 2]
            ]
            policies = (
                ["auto", "tuned", "legacy", *[f"fixed{value}" for value in budgets]]
                if native
                else ["auto", "legacy"]
            )
            predicted = predict(scores, profile)
            tuned_predicted = predict(scores, {**tuned, "mode": profile["mode"]})
            legacy_distances = np.float32(1) - scores
            legacy = np.maximum(
                1,
                (
                    legacy_distances
                    <= legacy_distances[:, :1] * {1: 0.6, 10: 7.0, 100: 81.0}[k]
                ).sum(axis=1),
            )
            for policy in policies:
                configure_override(tuned if policy == "tuned" else None)
                run_query(
                    dataset,
                    queries[ids[0]],
                    k,
                    policy,
                    build["partitions"],
                    fragment_offsets,
                )

            def query_records(item, tuned_only=False):
                ordinal, query_id = item
                # Rotate order to balance systematic drift between policies.
                active = (
                    policies
                    if ordinal < legacy_queries or not native
                    else [p for p in policies if p != "legacy"]
                )
                if tuned_only:
                    active = ["tuned"]
                elif ordinal >= timing_queries:
                    active = [p for p in active if p != "tuned"]
                shift = ordinal % len(active)
                baseline_pair = []
                query_rows = []
                for policy in active[shift:] + active[:shift]:
                    if ordinal < timing_queries:
                        configure_override(tuned if policy == "tuned" else None)
                    actual_ids, elapsed, stats = run_query(
                        dataset,
                        queries[query_id],
                        k,
                        policy,
                        build["partitions"],
                        fragment_offsets,
                    )
                    expected = (
                        int(policy.removeprefix("fixed"))
                        if policy.startswith("fixed")
                        else int(
                            tuned_predicted[query_id]
                            if policy == "tuned"
                            else predicted[query_id]
                            if native and policy == "auto"
                            else legacy[query_id]
                        )
                    )
                    recall = len(np.intersect1d(actual_ids, truth[query_id, :k])) / k
                    scanned = stats.all_counts["partitions_searched"]
                    if not native:
                        baseline_pair.append(
                            (actual_ids.tolist(), scanned, stats.index_comparisons)
                        )
                    if ordinal == 0:
                        save(
                            out / f"metrics-{suffix}-{k}-{policy}.json",
                            stats.all_counts,
                        )
                    row = {
                        "query_id": int(query_id),
                        "k": k,
                        "policy": policy,
                        "latency_ms": elapsed,
                        "recall": recall,
                        "partitions": scanned,
                        "comparisons": stats.index_comparisons,
                        "bytes_read": stats.bytes_read,
                        "predicted_partitions": expected,
                        "predicted_rows": int(scanned_rows[query_id, expected - 1]),
                        "route_recall": float(np.mean(ranks[query_id, :k] <= expected)),
                        "neighbor_ids": json.dumps(actual_ids.tolist()),
                        "phase": "timing" if query_id in timing_ids else "recall",
                    }
                    query_rows.append(row)
                if not native:
                    assert baseline_pair[0] == baseline_pair[1], baseline_pair
                return query_rows

            def record(ordinal, query_rows):
                writer.writerows(query_rows)
                records.extend(query_rows)
                if ordinal % 100 == 0:
                    stream.flush()
                    emit(
                        "measuring",
                        dataset=name,
                        k=k,
                        completed=ordinal,
                        total=len(ids),
                    )

            serial_count = min(timing_queries, len(ids))
            for ordinal in range(serial_count):
                record(ordinal, query_records((ordinal, ids[ordinal])))
            # Only correctness is collected concurrently. These latencies are
            # excluded from every timing summary, and all futures finish before
            # the next serial timing phase starts.
            configure_override(None)
            with ThreadPoolExecutor(max_workers=recall_workers) as executor:
                remaining = enumerate(ids[serial_count:], start=serial_count)
                for ordinal, query_rows in enumerate(
                    executor.map(query_records, remaining), start=serial_count
                ):
                    record(ordinal, query_rows)
            if native:
                # Environment overrides are process-wide. Run the tuned recall
                # phase separately with constant settings, never mutate them
                # while concurrent queries are in flight.
                configure_override(tuned)
                with ThreadPoolExecutor(max_workers=recall_workers) as executor:
                    remaining = enumerate(ids[serial_count:], start=serial_count)
                    for ordinal, query_rows in enumerate(
                        executor.map(lambda item: query_records(item, True), remaining),
                        start=serial_count,
                    ):
                        record(ordinal, query_rows)
                configure_override(None)
            results[str(k)] = {}
            for policy in policies:
                rows = [
                    row for row in records if row["k"] == k and row["policy"] == policy
                ]
                results[str(k)][policy] = {
                    "queries": len(rows),
                    "recall": float(np.mean([r["recall"] for r in rows])),
                    "timing_queries": sum(r["query_id"] in timing_ids for r in rows),
                    **{
                        field: summary(
                            [r[field] for r in rows if r["query_id"] in timing_ids]
                        )
                        for field in ["latency_ms", "partitions", "comparisons"]
                    },
                    "bytes_read": sum(r["bytes_read"] for r in rows),
                    "partition_prediction_mismatches": sum(
                        r["partitions"] != r["predicted_partitions"] for r in rows
                    ),
                    "recall_prediction_max_error": max(
                        abs(r["recall"] - r["route_recall"]) for r in rows
                    ),
                    "row_prediction_mismatches": sum(
                        r["comparisons"] != r["predicted_rows"] for r in rows
                    ),
                }
            save(out / f"measure-{suffix}.json", results)
            emit("measured", dataset=name, k=k, result=results[str(k)])
    library = Path(lance.__file__).parent / "lance.abi3.so"
    save(
        out / f"identity-{suffix}.json",
        {
            "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "calibration_sha256": hashlib.sha256(
                (root / "calibration.json").read_bytes()
            ).hexdigest(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "threads": {
                key: os.environ.get(key)
                for key in [
                    "LANCE_CPU_THREADS",
                    "RAYON_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                ]
            },
            "lance_version": lance.__version__,
            "query_count": len(ids),
            "timing_queries": min(timing_queries, len(ids)),
            "legacy_queries": min(legacy_queries, len(ids)),
            "recall_workers": recall_workers,
            "native": native,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--timing-queries", type=int, default=512)
    parser.add_argument("--legacy-queries", type=int, default=32)
    parser.add_argument("--recall-workers", type=int, default=8)
    args = parser.parse_args()
    measure(
        args.root,
        args.dataset,
        args.limit,
        args.native,
        args.timing_queries,
        args.legacy_queries,
        args.recall_workers,
    )
