# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Compare two native binaries on identical S3 indices and held-out queries."""

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def worker(args):
    sys.path.insert(0, str(args.runtime))
    import lance

    binary = Path(importlib.import_module("lance.lance").__file__)
    native_hash = hashlib.file_digest(binary.open("rb"), "sha256").hexdigest()
    with np.load(args.queries) as arrays:
        queries = {
            name: arrays[name] for name in ("vectors", "truth", "query_ids", "split")
        }
    datasets = json.loads(args.datasets.read_text())
    current, dataset = None, None
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "setup":
            current = datasets[request["index"]]
            dataset = lance.dataset(
                current["uri"],
                version=current["version"],
                index_cache_size_bytes=48 * 1024**3,
            )
            assert dataset.count_rows() == current["rows"]
            if request["cache"] == "warm":
                dataset.prewarm_index(current["index_name"])
            print(
                json.dumps(
                    {
                        "ready": True,
                        "native_sha256": native_hash,
                        "native_path": str(binary),
                    }
                ),
                flush=True,
            )
            continue
        position, k = request["position"], request["k"]
        truth = queries["truth"][position]
        if request["filtered"]:
            truth = truth[truth % 10 == 0]
        assert len(truth) >= k, (
            "exhaustive oracle depth is insufficient for this filter"
        )
        truth = truth[:k]
        nearest = {
            "column": "vector",
            "q": queries["vectors"][position],
            "k": k,
            "query_parallelism": 1,
            "search_effort": request["effort"],
        }
        if request["refine"]:
            nearest["refine_factor"] = request["refine"]
        captured = []
        before_bytes, before_iops = lance.bytes_read_counter(), lance.iops_counter()
        start_cpu, start = time.process_time_ns(), time.perf_counter_ns()
        if request["cache"] == "cold":
            dataset = lance.dataset(
                current["uri"], version=current["version"], index_cache_size_bytes=0
            )
        table = dataset.scanner(
            columns=["_distance"],
            with_row_id=True,
            nearest=nearest,
            index_segments=[current["index_uuid"]],
            filter="_rowid % 10 = 0" if request["filtered"] else None,
            prefilter=True,
            scan_stats_callback=captured.append,
        ).to_table()
        elapsed, cpu = (
            time.perf_counter_ns() - start,
            time.process_time_ns() - start_cpu,
        )
        ids = table["_rowid"].to_numpy()
        assert len(ids) == k and len(np.unique(ids)) == k
        if request["filtered"]:
            assert np.all(ids % 10 == 0)
        assert len(captured) == 1
        stats = captured[0]
        print(
            json.dumps(
                {
                    "latency_ms": elapsed / 1e6,
                    "cpu_ms": cpu / 1e6,
                    "recall": len(np.intersect1d(ids, truth)) / k,
                    "ids": ids.tolist(),
                    "partitions": stats.all_counts["partitions_searched"],
                    "comparisons": stats.index_comparisons,
                    "bytes_read": lance.bytes_read_counter() - before_bytes,
                    "iops": lance.iops_counter() - before_iops,
                    "scan_requests": stats.requests,
                    "scan_bytes_read": stats.bytes_read,
                }
            ),
            flush=True,
        )


def exchange(process, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    response = process.stdout.readline()
    if not response:
        raise RuntimeError(f"native worker exited: {process.poll()}")
    return json.loads(response)


def measure(args):
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.queries) as arrays:
        queries = {name: arrays[name] for name in ("query_ids", "split")}
    evaluation = np.flatnonzero(queries["split"] == 1)[:136]
    assert len(evaluation) == 136
    # The last eight queries are warmups; no timed query is used for warmup.
    measured, warmup = evaluation[:128], evaluation[128:]
    processes, identities = {}, {}
    try:
        for label, runtime in [
            ("baseline", args.baseline),
            ("candidate", args.candidate),
        ]:
            command = [
                sys.executable,
                __file__,
                "worker",
                "--runtime",
                str(runtime),
                "--queries",
                str(args.queries),
                "--datasets",
                str(args.datasets),
            ]
            processes[label] = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        scenarios = [
            ("flat", 10, 0, False),
            ("rq", 10, 0, False),
            ("rq", 10, 2, False),
            ("rq", 1000, 2, False),
            ("rq", 10, 2, True),
        ]
        arms = [(binary, effort) for binary in processes for effort in [0.1, 0.25, 0.5]]
        metadata = {
            "baseline_head": args.baseline_head,
            "candidate_head": args.candidate_head,
            "queries_sha256": hashlib.file_digest(
                args.queries.open("rb"), "sha256"
            ).hexdigest(),
            "datasets": json.loads(args.datasets.read_text()),
            "evaluation_ids": queries["query_ids"][measured].tolist(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "thread_environment": {
                key: os.environ.get(key)
                for key in [
                    "LANCE_CPU_THREADS",
                    "RAYON_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                ]
            },
        }
        count = 0
        with (args.output / "queries.jsonl").open("x") as stream:
            for cache in ["warm", "cold"]:
                selected = (
                    scenarios if cache == "warm" else [scenarios[0], scenarios[2]]
                )
                for index, k, refine, filtered in selected:
                    for binary, process in processes.items():
                        identities[binary] = exchange(
                            process, {"op": "setup", "index": index, "cache": cache}
                        )
                    assert (
                        identities["baseline"]["native_sha256"]
                        != identities["candidate"]["native_sha256"]
                    ), "both workers loaded the same native binary"
                    common = {
                        "op": "query",
                        "index": index,
                        "k": k,
                        "refine": refine,
                        "filtered": filtered,
                        "cache": cache,
                    }
                    for position in warmup if cache == "warm" else []:
                        for binary, effort in arms:
                            exchange(
                                processes[binary],
                                {**common, "position": int(position), "effort": effort},
                            )
                    schedule = (
                        [
                            (int(p), repeat, True)
                            for repeat in range(3)
                            for p in measured[:64]
                        ]
                        + [(int(p), 0, False) for p in measured[64:]]
                        if cache == "warm"
                        else [(int(p), 0, True) for p in measured[:16]]
                    )
                    for offset, (position, repeat, timed) in enumerate(schedule):
                        ordered = (
                            arms[offset % len(arms) :] + arms[: offset % len(arms)]
                        )
                        default_ids = {}
                        for binary, effort in ordered:
                            request = {**common, "position": position, "effort": effort}
                            result = exchange(processes[binary], request)
                            if effort == 0.5:
                                default_ids[binary] = result["ids"]
                            record = {
                                **request,
                                "binary": binary,
                                "repeat": repeat,
                                "timed": timed,
                                "query_id": int(queries["query_ids"][position]),
                                **result,
                            }
                            stream.write(json.dumps(record) + "\n")
                            count += 1
                        assert default_ids["baseline"] == default_ids["candidate"], (
                            "default changed"
                        )
                        if offset % 16 == 0:
                            stream.flush()
                            print(
                                json.dumps(
                                    {"completed": count, **common, "position": offset}
                                ),
                                flush=True,
                            )
        metadata["binaries"] = identities
        (args.output / "identity.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    finally:
        for process in processes.values():
            process.stdin.close()
            process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("measure")
    run.add_argument("--baseline", type=Path, required=True)
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--baseline-head", required=True)
    run.add_argument("--candidate-head", required=True)
    run.add_argument("--output", type=Path, required=True)
    child = commands.add_parser("worker")
    child.add_argument("--runtime", type=Path, required=True)
    for command in [run, child]:
        command.add_argument("--queries", type=Path, required=True)
        command.add_argument("--datasets", type=Path, required=True)
    args = parser.parse_args()
    worker(args) if args.command == "worker" else measure(args)
