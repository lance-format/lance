# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Measure a frozen constant fallback against the original fallback binary."""

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import lance
import numpy as np
from calibrate_fallback import signed_counts
from common import DATASETS, emit, gap_counts, matrix, save
from measure import OVERRIDES, configure_override, run_query
from prepare_large import sha256

NATIVE_K = [1, 10, 100, 1000, 10000, 100000]


def dataset_inputs(root, name):
    calibration = json.loads((root / "calibration-frozen.json").read_text())
    assert calibration["calibration_all_k"] is True
    assert calibration["calibration_k_range"] == [1, 100000]
    assert sha256(root / "calibration.json") == sha256(root / "calibration-frozen.json")
    source = Path(calibration["source"])
    directory = source / name
    assert sha256(directory / "prepared.json") == calibration["prepared_sha256"][name]
    assert sha256(directory / "split.npz") == calibration["split_sha256"][name]
    build = json.loads((directory / "build.json").read_text())
    dataset = lance.dataset(
        source / "data" / name / "base.lance",
        version=build["version"],
        index_cache_size_bytes=200 * 1024**3,
    )
    queries = matrix(
        lance.dataset(source / "data" / name / "queries.lance").to_table()["vector"]
    )
    lookup = np.load(directory / "row-ids.npy", mmap_mode="r")
    return calibration, directory, build, dataset, queries, lookup


def baseline_worker(root, name):
    _, _, build, dataset, queries, lookup = dataset_inputs(root, name)
    configure_override(None)
    dataset.prewarm_index(build["index_name"])
    library = Path(lance.__file__).parent / "lance.abi3.so"
    print(json.dumps({"binary_sha256": sha256(library)}), flush=True)
    for line in sys.stdin:
        query_id, k = json.loads(line)
        found, elapsed, stats = run_query(
            dataset,
            queries[query_id],
            k,
            DATASETS[name],
            "legacy",
            build["partitions"],
            lookup,
            None,
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


class OriginalFallback:
    """Invoke the original binary with an explicit full maximum at every k."""

    def __init__(self, root, name, runtime):
        environment = {**os.environ, "PYTHONPATH": str(runtime.resolve())}
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
        self.binary_sha256 = self.read()["binary_sha256"]
        assert (
            self.binary_sha256
            == (root / "baseline-binary.sha256").read_text().split()[0]
        )

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Fallback baseline exited with {self.process.poll()}")
        return json.loads(line)

    def query(self, query_id, k):
        self.process.stdin.write(json.dumps([int(query_id), k]) + "\n")
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


def measure(root, name, args):
    calibration, directory, build, dataset, queries, lookup = dataset_inputs(root, name)
    metric = DATASETS[name]
    profile = calibration["profiles"][metric]["gap"]
    assert profile["feasible"]
    result = root / name / ("native" + (f"-{args.label}" if args.label else ""))
    result.mkdir(parents=True, exist_ok=True)
    assert not list(result.glob("*.csv")), "Preserve earlier measurements"
    truth = np.load(directory / "truth-ids.npy", mmap_mode="r")
    distances = np.load(directory / "distances.npy", mmap_mode="r")
    ranks = np.load(directory / "ranks.npy", mmap_mode="r")
    scanned_rows = np.load(directory / "scanned_rows.npy", mmap_mode="r")
    ids = np.load(directory / "split.npz")[args.split][: args.query_count]
    candidate_budget = np.maximum(
        gap_counts(metric, distances, profile["threshold"]), 1
    )
    configure_override(None)
    candidate_sha = sha256(Path(lance.__file__).parent / "lance.abi3.so")
    assert candidate_sha == (root / "candidate-binary.sha256").read_text().split()[0]
    dataset.prewarm_index(build["index_name"])
    emit("fallback_prewarmed", dataset=name, candidate_sha256=candidate_sha)
    baseline = OriginalFallback(root, name, args.baseline_runtime)
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
        "predicted_final_partitions",
        "predicted_final_rows",
        "route_recall",
    ]
    try:
        for k in args.k:
            factor = 0.6 if k == 1 else 7.0 if k <= 10 else 81.0
            baseline_budget = signed_counts(distances, factor)
            arrays = {
                policy: np.lib.format.open_memmap(
                    result / f"ids-{k}-{policy}.npy",
                    mode="w+",
                    dtype=np.int64,
                    shape=(
                        len(ids)
                        if policy == "candidate"
                        else min(len(ids), args.baseline_queries),
                        k,
                    ),
                )
                for policy in ["candidate", "baseline"]
            }

            def query(ordinal, query_id, policies, phase):
                rows = []
                offset = ordinal % len(policies)
                for policy in policies[offset:] + policies[:offset]:
                    if policy == "baseline":
                        found, elapsed, stats = baseline.query(query_id, k)
                        initial = int(baseline_budget[query_id])
                    else:
                        found, elapsed, stats = run_query(
                            dataset,
                            queries[query_id],
                            k,
                            metric,
                            "legacy",
                            build["partitions"],
                            lookup,
                            None,
                        )
                        initial = int(candidate_budget[query_id])
                    assert len(found) == k, (name, k, policy, query_id, len(found))
                    fill = (
                        int(np.searchsorted(scanned_rows[query_id], k, side="left")) + 1
                    )
                    final = max(initial, fill)
                    assert final <= build["partitions"]
                    if phase != "warmup":
                        arrays[policy][ordinal] = found
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
                            "predicted_final_partitions": final,
                            "predicted_final_rows": int(
                                scanned_rows[query_id, final - 1]
                            ),
                            "route_recall": float(
                                np.mean(ranks[query_id, :k] <= final)
                            ),
                        }
                    )
                return rows

            with (result / f"records-{k}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for policy in ["candidate", "baseline"]:
                    query(0, ids[0], [policy], "warmup")
                timing_count = min(args.timing_queries, len(ids))
                for ordinal, query_id in enumerate(ids[:timing_count]):
                    policies = ["candidate"]
                    if ordinal < args.baseline_queries:
                        policies.append("baseline")
                    writer.writerows(query(ordinal, query_id, policies, "timing"))
                    if ordinal % 32 == 0:
                        stream.flush()
                        emit(
                            "fallback_timing",
                            dataset=name,
                            k=k,
                            completed=ordinal + 1,
                            total=timing_count,
                        )
                remaining = list(enumerate(ids[timing_count:], start=timing_count))
                with ThreadPoolExecutor(args.recall_workers) as pool:
                    for ordinal, rows in enumerate(
                        pool.map(
                            lambda item: query(
                                item[0], item[1], ["candidate"], "recall"
                            ),
                            remaining,
                        )
                    ):
                        writer.writerows(rows)
                        if ordinal % 128 == 0:
                            stream.flush()
                            emit(
                                "fallback_recall",
                                dataset=name,
                                k=k,
                                completed=timing_count + ordinal + 1,
                                total=len(ids),
                            )
            for array in arrays.values():
                array.flush()
            arrays.clear()
            emit("fallback_measured", dataset=name, k=k)
    finally:
        baseline.close()
    save(
        result / "identity.json",
        {
            "dataset": name,
            "k": args.k,
            "split": args.split,
            "query_count": len(ids),
            "timing_queries": args.timing_queries,
            "baseline_queries": args.baseline_queries,
            "candidate_binary_sha256": candidate_sha,
            "baseline_binary_sha256": baseline.binary_sha256,
            "calibration_sha256": sha256(root / "calibration-frozen.json"),
            "prepared_sha256": sha256(directory / "prepared.json"),
            "split_sha256": sha256(directory / "split.npz"),
            "build_sha256": sha256(directory / "build.json"),
            "harness_sha256": {
                filename: sha256(Path(__file__).with_name(filename))
                for filename in [
                    "measure_fallback.py",
                    "calibrate_fallback.py",
                    "measure.py",
                    "common.py",
                    "prepare_large.py",
                    "prepare.py",
                    "build_large_patch.py",
                ]
            },
            "files_sha256": {
                path.name: sha256(path)
                for path in sorted(result.iterdir())
                if path.suffix in (".npy", ".csv")
            },
        },
    )
    emit("fallback_complete", dataset=name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--baseline-runtime", type=Path)
    parser.add_argument("--baseline-worker", action="store_true")
    parser.add_argument("--k", type=int, nargs="+", choices=NATIVE_K, default=NATIVE_K)
    parser.add_argument("--query-count", type=int, default=512)
    parser.add_argument("--timing-queries", type=int, default=128)
    parser.add_argument("--baseline-queries", type=int, default=32)
    parser.add_argument("--recall-workers", type=int, default=8)
    parser.add_argument(
        "--split", choices=["calibration", "evaluation"], default="evaluation"
    )
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    if args.baseline_worker:
        baseline_worker(args.root, args.dataset)
    else:
        assert args.baseline_runtime is not None
        assert 0 < args.baseline_queries <= args.timing_queries <= args.query_count
        measure(args.root, args.dataset, args)
