# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Paired IVF_RQ5 effort measurements; see SEARCH_EFFORT_PROTOCOL.md."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
from common import DATASETS as CALIBRATION_DATASETS
from common import emit, save, summary
from measure import configure_override, run_query
from prepare_large import sha256
from rq5 import inputs

EFFORTS = {"e0": 0.0, "e025": 0.25, "e05": 0.5, "e075": 0.75, "e1": 1.0}
DATASETS = {**CALIBRATION_DATASETS, "coyo-ve-qwen3vl-2048": "cosine"}
ARMS = [*EFFORTS, "baseline"]
K_VALUES = [1, 10, 100, 1000, 10000, 100000]
CODE_FILES = [
    "rust/lance-index/src/vector.rs",
    "rust/lance/src/io/exec/knn.rs",
    "rust/lance/src/io/exec/knn/adaptive_probe.rs",
    "rust/lance/src/io/exec/ann_proto.rs",
    "rust/lance/src/dataset/scanner.rs",
    "python/src/dataset.rs",
    "python/python/lance/dataset.py",
    "protos/ann.proto",
    "benchmarks/auto-ivf-large-k/search_effort.py",
    "benchmarks/auto-ivf-large-k/measure.py",
    "benchmarks/auto-ivf-large-k/SEARCH_EFFORT_PROTOCOL.md",
    "benchmarks/auto-ivf-large-k/COYO_SEARCH_EFFORT_PROTOCOL.md",
    "benchmarks/auto-ivf-large-k/prepare_coyo.py",
    "benchmarks/auto-ivf-large-k/prepare.py",
    "benchmarks/auto-ivf-large-k/prepare_large.py",
    "benchmarks/auto-ivf-large-k/common.py",
    "benchmarks/auto-ivf-large-k/rq5.py",
]
FIELDS = [
    "ordinal",
    "query_id",
    "k",
    "arm",
    "phase",
    "recall",
    "returned",
    "partitions",
    "comparisons",
    "bytes_read",
    "latency_ms",
]


def native_sha():
    return sha256(Path(lance.__file__).parent / "lance.abi3.so")


def native_query(rq, vector, k, metric, build, lookup, effort=None):
    found, elapsed, stats = run_query(
        rq,
        vector,
        k,
        metric,
        "auto",
        build["partitions"],
        lookup,
        None,
        search_effort=effort,
    )
    return found, {
        "latency_ms": elapsed,
        "partitions": stats.all_counts["partitions_searched"],
        "comparisons": stats.index_comparisons,
        "bytes_read": stats.bytes_read,
    }


def worker(args, name):
    _, _, build, rq, queries, lookup = inputs(args.indices, name)
    configure_override(None)
    rq.prewarm_index(build["index_name"])
    print(json.dumps({"sha256": native_sha()}), flush=True)
    for line in sys.stdin:
        query_id, k = json.loads(line)
        found, record = native_query(
            rq, queries[query_id], k, DATASETS[name], build, lookup
        )
        print(json.dumps({**record, "ids": found.tolist()}), flush=True)


class Baseline:
    def __init__(self, args, name):
        environment = {**os.environ, "PYTHONPATH": str(args.baseline_runtime.resolve())}
        self.process = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "worker",
                str(args.root),
                name,
                "--indices",
                str(args.indices),
            ],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.lock = threading.Lock()
        self.sha256 = self.read()["sha256"]
        assert (
            self.sha256 == (args.root / "baseline-binary.sha256").read_text().split()[0]
        )

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Baseline worker exited: {self.process.poll()}")
        return json.loads(line)

    def query(self, query_id, k):
        with self.lock:
            self.process.stdin.write(json.dumps([int(query_id), k]) + "\n")
            self.process.stdin.flush()
            record = self.read()
        return np.asarray(record.pop("ids"), dtype=np.int64), record

    def close(self):
        self.process.stdin.close()
        assert self.process.wait(timeout=60) == 0


def identity(args, name, prepared, previous, queries):
    index_dir = args.indices / name / "rq5.lance"
    for filename, digest in prepared["index_sha256"].items():
        assert sha256(index_dir / filename) == digest, filename
    source = json.loads((previous / "prepared.json").read_text())
    truth_sha = sha256(previous / "truth-ids.npy")
    assert truth_sha == source["sha256"]["truth-ids.npy"]
    repo = Path(__file__).resolve().parents[2]
    return {
        "prepared_sha256": sha256(args.indices / name / "prepared.json"),
        "truth_sha256": truth_sha,
        "query_array_sha256": hashlib.sha256(queries.tobytes()).hexdigest(),
        "binary_sha256": native_sha(),
        "index_sha256": prepared["index_sha256"],
        "source_sha256": {filename: sha256(repo / filename) for filename in CODE_FILES},
    }


def measure(args, name):
    expected_threads = {
        "LANCE_CPU_THREADS": "16",
        "RAYON_NUM_THREADS": "16",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    assert {key: os.environ.get(key) for key in expected_threads} == expected_threads
    assert sorted(os.sched_getaffinity(0)) == list(range(16))
    prepared, previous, build, rq, queries, lookup = inputs(args.indices, name)
    split = np.load(previous / "split.npz")
    query_ids = split[args.split][: args.queries]
    assert len(query_ids) == args.queries and args.timing <= len(query_ids)
    assert not set(split["calibration"]) & set(split["evaluation"])
    truth = np.load(previous / "truth-ids.npy", mmap_mode="r")
    out = args.root / args.label / name
    out.mkdir(parents=True, exist_ok=False)
    before = identity(args, name, prepared, previous, queries)
    assert (
        before["binary_sha256"]
        == (args.root / "candidate-binary.sha256").read_text().split()[0]
    )
    save(out / "identity-before.json", before)
    configure_override(None)
    rq.prewarm_index(build["index_name"])
    baseline = Baseline(args, name)
    emit("effort_prewarmed", corpus=name, queries=len(query_ids), timing=args.timing)
    try:
        # Omitted effort and explicit default use identical candidate paths.
        default_checks = []
        for query_id in split["calibration"][:2]:
            for k in args.k:
                omitted, omitted_stats = native_query(
                    rq,
                    queries[query_id],
                    k,
                    DATASETS[name],
                    build,
                    lookup,
                )
                explicit, explicit_stats = native_query(
                    rq,
                    queries[query_id],
                    k,
                    DATASETS[name],
                    build,
                    lookup,
                    0.5,
                )
                assert np.array_equal(omitted, explicit)
                for field in ["partitions", "comparisons", "bytes_read"]:
                    assert omitted_stats[field] == explicit_stats[field]
                default_checks.append({"query_id": int(query_id), "k": k})
        save(out / "omitted-default-checks.json", default_checks)
        for k in args.k:
            assert k <= truth.shape[1]
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
                records = []
                arms = ARMS[ordinal % len(ARMS) :] + ARMS[: ordinal % len(ARMS)]
                for arm in arms:
                    if arm == "baseline":
                        found, record = baseline.query(query_id, k)
                    else:
                        found, record = native_query(
                            rq,
                            queries[query_id],
                            k,
                            DATASETS[name],
                            build,
                            lookup,
                            EFFORTS[arm],
                        )
                    assert len(found) == k and record["bytes_read"] == 0
                    arrays[arm][ordinal] = found
                    if arm == "e1":
                        assert record["partitions"] == build["partitions"]
                    records.append(
                        {
                            "ordinal": ordinal,
                            "query_id": int(query_id),
                            "k": k,
                            "arm": arm,
                            "phase": "serial" if ordinal < args.timing else "recall",
                            "recall": len(np.intersect1d(found, truth[query_id, :k]))
                            / k,
                            "returned": len(found),
                            **record,
                        }
                    )
                assert np.array_equal(
                    arrays["e05"][ordinal], arrays["baseline"][ordinal]
                )
                by_arm = {record["arm"]: record for record in records}
                for field in ["partitions", "comparisons"]:
                    assert by_arm["e05"][field] == by_arm["baseline"][field], (
                        name,
                        query_id,
                        k,
                        field,
                    )
                return records

            with (out / f"queries-{k}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                # No other query, worker, hashing or validation runs while timing.
                for ordinal in range(args.timing):
                    writer.writerows(query((ordinal, query_ids[ordinal])))
                    if (ordinal + 1) % 16 == 0 or ordinal + 1 == args.timing:
                        stream.flush()
                        emit("effort_serial", corpus=name, k=k, completed=ordinal + 1)
                with ThreadPoolExecutor(args.workers) as pool:
                    remaining = enumerate(query_ids[args.timing :], start=args.timing)
                    for ordinal, records in enumerate(
                        pool.map(query, remaining), start=args.timing
                    ):
                        writer.writerows(records)
                        if (ordinal + 1) % 16 == 0 or ordinal + 1 == len(query_ids):
                            stream.flush()
                            emit(
                                "effort_recall", corpus=name, k=k, completed=ordinal + 1
                            )
            for array in arrays.values():
                array.flush()
            arrays.clear()
    finally:
        baseline.close()
    # inputs() also rechecks the frozen model and source membership/centroids.
    after_prepared, after_previous, _, _, after_queries, _ = inputs(args.indices, name)
    after = identity(args, name, after_prepared, after_previous, after_queries)
    assert after == before
    save(out / "identity-after.json", after)
    save(
        out / "complete.json",
        {
            "query_ids": query_ids.tolist(),
            "k": args.k,
            "split": args.split,
            "timing": args.timing,
            "workers": args.workers,
            "baseline_sha256": baseline.sha256,
            "candidate_sha256": native_sha(),
            "partitions": build["partitions"],
            "rows": build["rows"],
            "approx_mode": "normal",
            "refine_factor": None,
            "threads": expected_threads,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    )


def audit(args, names):
    results = []
    for name in names:
        out = args.root / args.label / name
        complete = json.loads((out / "complete.json").read_text())
        before = json.loads((out / "identity-before.json").read_text())
        assert before == json.loads((out / "identity-after.json").read_text())
        prepared = json.loads((args.indices / name / "prepared.json").read_text())
        assert (
            sha256(args.indices / name / "prepared.json") == before["prepared_sha256"]
        )
        previous = Path(prepared["source"]) / name
        split = np.load(previous / "split.npz")
        query_ids = complete["query_ids"]
        if args.label == "native":
            assert complete["split"] == "evaluation"
            assert len(query_ids) == 128 and complete["timing"] == 64
            assert complete["k"] == K_VALUES and complete["workers"] == 8
        assert complete["partitions"] == prepared["source_build"]["partitions"]
        assert complete["rows"] == prepared["source_build"]["rows"]
        assert complete["cpu_affinity"] == list(range(16))
        assert np.array_equal(query_ids, split[complete["split"]][: len(query_ids)])
        assert len(set(query_ids)) == len(query_ids)
        assert not set(split["calibration"]) & set(split["evaluation"])
        assert (
            complete["baseline_sha256"]
            == (args.root / "baseline-binary.sha256").read_text().split()[0]
        )
        assert complete["candidate_sha256"] == before["binary_sha256"]
        assert (
            complete["candidate_sha256"]
            == (args.root / "candidate-binary.sha256").read_text().split()[0]
        )
        truth = np.load(previous / "truth-ids.npy", mmap_mode="r")
        for k in complete["k"]:
            with (out / f"queries-{k}.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == len(query_ids) * len(ARMS)
            arrays = {
                arm: np.load(out / f"ids-{k}-{arm}.npy", mmap_mode="r") for arm in ARMS
            }
            groups = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
            assert np.array_equal(arrays["e05"], arrays["baseline"])
            for arm, found in arrays.items():
                assert found.shape == (len(query_ids), k)
                records = groups[arm]
                assert [int(row["query_id"]) for row in records] == query_ids
                assert [int(row["ordinal"]) for row in records] == list(
                    range(len(query_ids))
                )
                recalls = []
                for ordinal, (query_id, record) in enumerate(zip(query_ids, records)):
                    ids = found[ordinal]
                    assert len(set(ids.tolist())) == k
                    assert np.all((ids >= 0) & (ids < complete["rows"]))
                    recall = (
                        len(set(ids.tolist()) & set(truth[query_id, :k].tolist())) / k
                    )
                    assert abs(recall - float(record["recall"])) < 1e-12
                    assert int(record["returned"]) == k and int(record["k"]) == k
                    assert int(record["bytes_read"]) == 0
                    assert (
                        math.isfinite(float(record["latency_ms"]))
                        and float(record["latency_ms"]) > 0
                    )
                    assert record["phase"] == (
                        "serial" if ordinal < complete["timing"] else "recall"
                    )
                    assert 1 <= int(record["partitions"]) <= complete["partitions"]
                    if arm == "e1":
                        assert int(record["partitions"]) == complete["partitions"]
                    if arm == "e05":
                        for field in ["partitions", "comparisons"]:
                            assert record[field] == groups["baseline"][ordinal][field]
                    recalls.append(recall)
                serial = [
                    float(row["latency_ms"])
                    for row in records
                    if row["phase"] == "serial"
                ]
                assert len(serial) == complete["timing"]
                results.append(
                    {
                        "corpus": name,
                        "metric": DATASETS[name],
                        "k": k,
                        "arm": arm,
                        "effort": EFFORTS.get(arm),
                        "queries": len(query_ids),
                        "timed_queries": len(serial),
                        "recall": summary(recalls),
                        "minimum_recall": min(recalls),
                        "latency_ms": summary(serial),
                        "partitions": summary(
                            [int(row["partitions"]) for row in records]
                        ),
                        "comparisons": summary(
                            [int(row["comparisons"]) for row in records]
                        ),
                    }
                )
    save(args.root / args.label / "audited-results.json", results)
    emit("effort_audit_complete", groups=len(results), all_integrity_checks_passed=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["measure", "worker", "audit"])
    parser.add_argument("root", type=Path)
    parser.add_argument("name", choices=[*DATASETS, "all"])
    parser.add_argument("--indices", required=True, type=Path)
    parser.add_argument("--baseline-runtime", type=Path)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--timing", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k", type=int, nargs="+", default=K_VALUES)
    parser.add_argument(
        "--split", choices=["calibration", "evaluation"], default="evaluation"
    )
    parser.add_argument("--label", default="native")
    args = parser.parse_args()
    # Keep the original five-corpus calibration campaign reproducible.
    names = list(CALIBRATION_DATASETS) if args.name == "all" else [args.name]
    if args.command == "audit":
        audit(args, names)
    else:
        for name in names:
            if args.command == "measure":
                assert args.baseline_runtime
                measure(args, name)
            else:
                worker(args, name)
