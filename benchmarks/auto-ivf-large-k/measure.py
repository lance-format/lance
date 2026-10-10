# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Measure frozen probing policies on held-out queries with a warm index cache.

Timed queries run serially with a rotating policy order. Remaining held-out
queries only contribute recall and scan counts and run with several workers.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import lance
import numpy as np
from common import (
    ANCHORS,
    DATASETS,
    EVALUATION_TARGET,
    EXISTING_K100,
    FILTER,
    MEASURED_K,
    emit,
    matrix,
    positions,
    predict,
    save,
    summary,
)

OVERRIDES = [
    ("LANCE_AUTO_PROBE_MARGIN", "margin"),
    ("LANCE_AUTO_MIN_INITIAL_NPROBES", "floor"),
    ("LANCE_AUTO_MAX_INITIAL_NPROBES", "cap"),
]


def configure_override(profile):
    for variable, field in OVERRIDES:
        if profile is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = str(profile[field])


def profile_tuple(profile):
    return (profile["margin"], profile["floor"], profile["cap"])


def auto_profile(calibration, metric, k):
    if k <= 100:
        return EXISTING_K100[metric]
    anchor = next((a for a in ANCHORS if k <= a), ANCHORS[-1])
    return profile_tuple(calibration["profiles"][metric][str(anchor)])


def tuned_profile(calibration, name, k):
    if k <= 100:
        return None
    anchor = next((a for a in ANCHORS if k <= a), ANCHORS[-1])
    return calibration["per_corpus"][name][str(anchor)]


def fixed_budgets(calibration, name, k):
    """fixed20 plus the calibration fixed budgets bracketing 95% recall."""
    rows = calibration["fixed"][name][str(k)]
    meeting = next(
        (i for i, row in enumerate(rows) if row["recall"] >= EVALUATION_TARGET),
        len(rows) - 1,
    )
    budgets = {20} | {row["nprobes"] for row in rows[max(0, meeting - 1) : meeting + 2]}
    return sorted(budgets)


def run_query(
    dataset, vector, k, metric, policy, partitions, lookup, flt, *, search_effort=None
):
    nearest = {
        "column": "vector",
        "q": vector,
        "k": k,
        "metric": metric,
        "query_parallelism": 1,
    }
    if search_effort is not None:
        nearest["search_effort"] = search_effort
    if policy.startswith("fixed"):
        nearest["nprobes"] = int(policy.removeprefix("fixed"))
    elif policy == "legacy":
        # An explicit full upper bound reproduces main's k > 100 heuristic on the
        # candidate binary without limiting its actual budget.
        nearest["maximum_nprobes"] = partitions
    captured = []
    start = time.perf_counter_ns()
    table = dataset.scanner(
        columns=["_distance"],
        with_row_id=True,
        nearest=nearest,
        filter=flt,
        prefilter=flt is not None,
        scan_stats_callback=captured.append,
    ).to_table()
    elapsed = (time.perf_counter_ns() - start) / 1e6
    assert len(captured) == 1
    ids = positions(table["_rowid"].to_numpy(), lookup)
    assert len(np.unique(ids)) == len(ids)
    return ids, elapsed, captured[0]


class BaselineWorker:
    """Keep the frozen baseline binary warm in a separate, serial query process."""

    def __init__(self, root, name, runtime):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(runtime.resolve())
        for variable, _ in OVERRIDES:
            environment.pop(variable, None)
        self.process = subprocess.Popen(
            [sys.executable, __file__, str(root), name, "--baseline-worker"],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.lock = threading.Lock()
        ready = self.read()
        self.binary_sha256 = ready["binary_sha256"]
        expected = (root / "baseline-binary.sha256").read_text().split()[0]
        assert self.binary_sha256 == expected, (ready, expected)

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Baseline worker exited with {self.process.poll()}")
        return json.loads(line)

    def query(self, query_id, k, flt):
        with self.lock:
            self.process.stdin.write(json.dumps([int(query_id), k, flt]) + "\n")
            self.process.stdin.flush()
            result = self.read()
        stats = SimpleNamespace(
            all_counts={"partitions_searched": result["partitions"]},
            index_comparisons=result["comparisons"],
            bytes_read=result["bytes_read"],
        )
        return np.asarray(result["ids"], dtype=np.int64), result["latency_ms"], stats

    def close(self):
        self.process.stdin.close()
        assert self.process.wait(timeout=60) == 0


def baseline_worker(root, name):
    """JSON-lines protocol; timings exclude IPC and include only the native query."""
    out = root / name
    directory = root / "data" / name
    build = json.loads((out / "build.json").read_text())
    dataset = lance.dataset(
        directory / "base.lance",
        version=build["version"],
        index_cache_size_bytes=200 * 1024**3,
    )
    queries = matrix(lance.dataset(directory / "queries.lance").to_table()["vector"])
    lookup = np.load(out / "row-ids.npy")
    dataset.prewarm_index(build["index_name"])
    library = Path(lance.__file__).parent / "lance.abi3.so"
    print(
        json.dumps({"binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest()}),
        flush=True,
    )
    for line in sys.stdin:
        query_id, k, flt = json.loads(line)
        found, elapsed, stats = run_query(
            dataset,
            queries[query_id],
            k,
            DATASETS[name],
            "auto",
            build["partitions"],
            lookup,
            flt,
        )
        print(
            json.dumps(
                {
                    "ids": found.tolist(),
                    "latency_ms": elapsed,
                    "partitions": stats.all_counts["partitions_searched"],
                    "comparisons": stats.index_comparisons,
                    "bytes_read": stats.bytes_read,
                }
            ),
            flush=True,
        )


def measure(root, name, args):
    metric = DATASETS[name]
    directory = root / "data" / name
    out = root / name
    calibration = json.loads((root / "calibration.json").read_text())
    build = json.loads((out / "build.json").read_text())
    dataset = lance.dataset(
        directory / "base.lance",
        version=build["version"],
        index_cache_size_bytes=200 * 1024**3,
    )
    queries = matrix(lance.dataset(directory / "queries.lance").to_table()["vector"])
    truth = np.load(out / "truth.npz")["ids"]
    ids = np.load(out / "split.npz")[args.split]
    if args.limit:
        ids = ids[: args.limit]
    routes = np.load(out / "routes.npz")
    distances = routes["distances"]
    scanned_rows = routes["scanned_rows"]
    ranks = routes["ranks"]
    lookup = np.load(out / "row-ids.npy")
    partitions = build["partitions"]
    started = time.monotonic()
    dataset.prewarm_index(build["index_name"])
    emit("prewarmed", dataset=name, seconds=time.monotonic() - started)
    baseline = (
        BaselineWorker(root, name, args.baseline_runtime)
        if args.native and args.baseline_runtime
        else None
    )
    suffix = "native" if args.native else "baseline-audit"
    if args.label:
        suffix += f"-{args.label}"
    results = {}
    records = []
    stream = (out / f"measure-{suffix}.csv").open("w")
    fields = [
        "query_id",
        "k",
        "policy",
        "filtered",
        "latency_ms",
        "recall",
        "returned",
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
    for k in args.k:
        auto = auto_profile(calibration, metric, k)
        tuned = tuned_profile(calibration, name, k)
        predicted = {"auto": predict(metric, distances, auto)}
        if tuned is not None:
            predicted["tuned"] = predict(metric, distances, profile_tuple(tuned))
        legacy_budget = np.maximum(
            (distances <= distances[:, :1] * np.float32(81.0)).sum(axis=1), 1
        )
        if args.native:
            policies = [
                "auto",
                *[f"fixed{b}" for b in fixed_budgets(calibration, name, k)],
            ]
            if k > 100:
                policies += ["tuned", "legacy"]
        else:
            # Main's default Auto must equal the candidate's bounded legacy arm.
            policies = ["auto", "legacy"] if k > 100 else ["auto"]
        timing_count = min(args.timing_queries, len(ids))

        def expected_budget(policy, query_id):
            if policy.startswith("fixed"):
                return min(int(policy.removeprefix("fixed")), partitions)
            if policy == "legacy" or (not args.native and k > 100):
                return int(legacy_budget[query_id])
            return int(predicted[policy][query_id])

        def one(ordinal, query_id, active, phase):
            rows = []
            shift = ordinal % len(active)
            for policy in active[shift:] + active[:shift]:
                if phase != "recall":
                    # Overrides are process-wide; only change them between
                    # serial queries.
                    configure_override(tuned if policy == "tuned" else None)
                if policy == "legacy" and baseline is not None:
                    found, elapsed, stats = baseline.query(query_id, k, None)
                else:
                    found, elapsed, stats = run_query(
                        dataset,
                        queries[query_id],
                        k,
                        metric,
                        policy,
                        partitions,
                        lookup,
                        None,
                    )
                if not policy.startswith("fixed"):
                    assert len(found) == k, (name, k, policy, query_id, len(found))
                else:
                    assert len(found) <= k, (name, k, policy, query_id, len(found))
                budget = expected_budget(policy, query_id)
                rows.append(
                    {
                        "query_id": int(query_id),
                        "k": k,
                        "policy": policy,
                        "filtered": False,
                        "latency_ms": elapsed,
                        "recall": len(np.intersect1d(found, truth[query_id, :k])) / k,
                        "returned": len(found),
                        "partitions": stats.all_counts["partitions_searched"],
                        "comparisons": stats.index_comparisons,
                        "bytes_read": stats.bytes_read,
                        "predicted_partitions": budget,
                        "predicted_rows": int(scanned_rows[query_id, budget - 1]),
                        "route_recall": float(np.mean(ranks[query_id, :k] <= budget)),
                        "neighbor_ids": json.dumps(found.tolist()),
                        "phase": phase,
                    }
                )
            if not args.native and k > 100 and set(active) == {"auto", "legacy"}:
                assert rows[0]["neighbor_ids"] == rows[1]["neighbor_ids"], rows
                assert rows[0]["partitions"] == rows[1]["partitions"], rows
                assert rows[0]["comparisons"] == rows[1]["comparisons"], rows
            return rows

        def record(rows):
            writer.writerows(rows)
            records.extend(rows)

        for policy in policies:
            one(0, ids[0], [policy], "warmup")
        for ordinal in range(timing_count):
            active = [
                p for p in policies if p != "legacy" or ordinal < args.legacy_queries
            ]
            record(one(ordinal, ids[ordinal], active, "timing"))
            if ordinal % 100 == 0:
                stream.flush()
                emit("timing", dataset=name, k=k, completed=ordinal)
        # Only correctness is collected concurrently; these latencies are
        # excluded. The tuned arm runs in its own phase with constant overrides.
        remaining = list(enumerate(ids[timing_count:], start=timing_count))
        recall_phases = [[p for p in policies if p not in ("legacy", "tuned")]]
        if "tuned" in policies:
            recall_phases.append(["tuned"])
        for active in recall_phases:
            configure_override(tuned if active == ["tuned"] else None)
            with ThreadPoolExecutor(args.recall_workers) as pool:
                for rows in pool.map(
                    lambda item: one(item[0], item[1], active, "recall"), remaining
                ):
                    record(rows)
        configure_override(None)
        stream.flush()
        results[str(k)] = summarize(
            [r for r in records if r["k"] == k and not r["filtered"]], policies
        )
        save(out / f"measure-{suffix}.json", results)
        emit("measured", dataset=name, k=k, result=results[str(k)])

    if args.native and args.filtered:
        results["filtered"] = measure_filtered(
            dataset,
            name,
            metric,
            queries,
            ids,
            partitions,
            lookup,
            writer,
            records,
            args,
            baseline,
        )
        save(out / f"measure-{suffix}.json", results)
    stream.close()
    if baseline is not None:
        baseline.close()
    library = Path(lance.__file__).parent / "lance.abi3.so"
    save(
        out / f"identity-{suffix}.json",
        {
            "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "baseline_binary_sha256": baseline.binary_sha256 if baseline else None,
            "harness_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [Path(__file__), Path(__file__).with_name("common.py")]
            },
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
            "split": args.split,
            "split_sha256": hashlib.sha256(
                (out / "split.npz").read_bytes()
            ).hexdigest(),
            "timing_queries": min(args.timing_queries, len(ids)),
            "legacy_queries": min(args.legacy_queries, len(ids)),
            "recall_workers": args.recall_workers,
            "k": args.k,
            "native": args.native,
            "filtered_queries": min(args.filtered_queries, len(ids))
            if args.filtered
            else 0,
            "dataset_version": build["version"],
            "index_segments": build["indices"],
            "prepared_sha256": hashlib.sha256(
                (out / "prepared.json").read_bytes()
            ).hexdigest(),
        },
    )


def measure_filtered(
    dataset,
    name,
    metric,
    queries,
    ids,
    partitions,
    lookup,
    writer,
    records,
    args,
    baseline,
):
    """Selective prefilter: exercises late probing beyond the initial budget."""
    out = args.root / name
    filtered_truth = np.load(out / "filtered-truth.npz")
    truth = filtered_truth["ids"]
    matching = set(filtered_truth["positions"].tolist())
    results = {}
    for k in [101, 1000]:
        policies = ["auto", "legacy", "fixed20"]

        def one(item):
            ordinal, query_id = item
            rows = []
            for policy in policies:
                if policy == "legacy" and baseline is not None:
                    found, elapsed, stats = baseline.query(query_id, k, FILTER)
                else:
                    found, elapsed, stats = run_query(
                        dataset,
                        queries[query_id],
                        k,
                        metric,
                        policy,
                        partitions,
                        lookup,
                        FILTER,
                    )
                assert matching.issuperset(found.tolist())
                rows.append(
                    {
                        "query_id": int(query_id),
                        "k": k,
                        "policy": policy,
                        "filtered": True,
                        "latency_ms": elapsed,
                        "recall": len(np.intersect1d(found, truth[query_id, :k])) / k,
                        "returned": len(found),
                        "partitions": stats.all_counts["partitions_searched"],
                        "comparisons": stats.index_comparisons,
                        "bytes_read": stats.bytes_read,
                        "predicted_partitions": -1,
                        "predicted_rows": -1,
                        "route_recall": -1,
                        "neighbor_ids": json.dumps(found.tolist()),
                        "phase": "recall",
                    }
                )
            return rows

        subset = ids[: args.filtered_queries]
        with ThreadPoolExecutor(args.recall_workers) as pool:
            for rows in pool.map(one, enumerate(subset)):
                writer.writerows(rows)
                records.extend(rows)
        results[str(k)] = {
            policy: {
                "queries": len(rows),
                "recall": float(np.mean([r["recall"] for r in rows])),
                "returned": summary([r["returned"] for r in rows]),
                "partitions": summary([r["partitions"] for r in rows]),
                "comparisons": summary([r["comparisons"] for r in rows]),
            }
            for policy in policies
            for rows in [
                [
                    r
                    for r in records
                    if r["filtered"] and r["k"] == k and r["policy"] == policy
                ]
            ]
        }
        emit("measured_filtered", dataset=name, k=k, result=results[str(k)])
    return results


def summarize(rows, policies):
    result = {}
    for policy in policies:
        selected = [r for r in rows if r["policy"] == policy]
        timed = [r for r in selected if r["phase"] == "timing"]
        result[policy] = {
            "queries": len(selected),
            "timing_queries": len(timed),
            "recall": float(np.mean([r["recall"] for r in selected])),
            "returned_min": int(min(r["returned"] for r in selected)),
            "latency_ms": summary([r["latency_ms"] for r in timed]),
            "partitions": summary([r["partitions"] for r in selected]),
            "comparisons": summary([r["comparisons"] for r in selected]),
            "bytes_read": int(sum(r["bytes_read"] for r in selected)),
            "partition_prediction_mismatches": sum(
                r["partitions"] != r["predicted_partitions"] for r in selected
            ),
            "row_prediction_mismatches": sum(
                r["comparisons"] != r["predicted_rows"] for r in selected
            ),
            "recall_prediction_max_error": max(
                abs(r["recall"] - r["route_recall"]) for r in selected
            ),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", default=MEASURED_K)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timing-queries", type=int, default=512)
    parser.add_argument("--legacy-queries", type=int, default=32)
    parser.add_argument("--recall-workers", type=int, default=8)
    parser.add_argument("--filtered", action="store_true")
    parser.add_argument("--filtered-queries", type=int, default=256)
    parser.add_argument("--baseline-runtime", type=Path)
    parser.add_argument(
        "--baseline-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--split", choices=["calibration", "evaluation"], default="evaluation"
    )
    args = parser.parse_args()
    if args.baseline_worker:
        baseline_worker(args.root, args.dataset)
    else:
        measure(args.root, args.dataset, args)
