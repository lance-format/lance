# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Compare two native binaries on identical S3 indices and held-out queries."""

import argparse
import hashlib
import importlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil


def worker(args):
    sys.path.insert(0, str(args.runtime))
    import lance

    binary = Path(importlib.import_module("lance.lance").__file__)
    native_hash = hashlib.file_digest(binary.open("rb"), "sha256").hexdigest()
    with np.load(args.queries) as arrays:
        queries = {name: arrays[name] for name in ("vectors", "query_ids", "split")}
        queries["truth"] = (
            np.load(args.truth, mmap_mode="r") if args.truth else arrays["truth"]
        )
    datasets = json.loads(args.datasets.read_text())
    current, dataset, active_cache = None, None, None
    ids_output = args.ids.open("xb")
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "setup":
            setup_start = time.monotonic()
            selected = datasets[request["index"]]
            reused = selected == current and request["cache"] == active_cache
            if not reused:
                current, active_cache = selected, request["cache"]
                dataset = lance.dataset(
                    current["uri"],
                    version=current["version"],
                    index_cache_size_bytes=args.index_cache_gib * 1024**3
                    if active_cache == "warm"
                    else 0,
                )
                assert dataset.count_rows() == current["rows"]
                if active_cache == "warm":
                    dataset.prewarm_index(current["index_name"])
            print(
                json.dumps(
                    {
                        "ready": True,
                        "setup_reused": reused,
                        "native_sha256": native_hash,
                        "native_path": str(binary),
                        "setup_seconds": time.monotonic() - setup_start,
                        "rss_bytes": psutil.Process().memory_info().rss,
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
            "metric": current["metric"],
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
        id_bytes = ids.astype("<u8", copy=False).tobytes()
        ids_offset = ids_output.tell()
        ids_output.write(id_bytes)
        print(
            json.dumps(
                {
                    "latency_ms": elapsed / 1e6,
                    "cpu_ms": cpu / 1e6,
                    "recall": len(np.intersect1d(ids, truth)) / k,
                    "ids_offset": ids_offset,
                    "ids_sha256": hashlib.sha256(id_bytes).hexdigest(),
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
    ids_output.close()


def exchange(process, request):
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    start = time.monotonic()
    while not select.select([process.stdout], [], [], 30)[0]:
        print(
            json.dumps(
                {
                    "event": "native_request_running",
                    "pid": process.pid,
                    "elapsed_seconds": time.monotonic() - start,
                    **request,
                }
            ),
            flush=True,
        )
    response = process.stdout.readline()
    if not response:
        raise RuntimeError(f"native worker exited: {process.poll()}")
    return json.loads(response)


def measure(args):
    start = time.monotonic()
    if (
        not 0 < args.timing_queries <= args.recall_queries
        or args.recall_queries < 16
        or args.repeats < 1
    ):
        raise ValueError(
            "Require recall_queries >= 16, 0 < timing_queries <= recall_queries "
            "and repeats >= 1"
        )
    if args.index_cache_gib <= 0:
        raise ValueError(
            f"index_cache_gib must be positive, got {args.index_cache_gib}"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.queries) as arrays:
        queries = {name: arrays[name] for name in ("query_ids", "split")}
    evaluation = np.flatnonzero(queries["split"] == 1)[: args.recall_queries + 8]
    assert len(evaluation) == args.recall_queries + 8
    # The last eight queries are warmups; no timed query is used for warmup.
    measured, warmup = (
        evaluation[: args.recall_queries],
        evaluation[args.recall_queries :],
    )
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
                "--ids",
                str(args.output / f"{label}.ids"),
                "--index-cache-gib",
                str(args.index_cache_gib),
            ]
            if args.truth:
                command.extend(["--truth", str(args.truth)])
            processes[label] = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            print(
                json.dumps(
                    {
                        "event": "worker_started",
                        "binary": label,
                        "pid": processes[label].pid,
                    }
                ),
                flush=True,
            )
        scenarios = [
            ("flat", 10, 0, False),
            ("rq", 10, 0, False),
            ("rq", 10, 2, False),
            ("rq", 1000, 2, False),
            ("rq", 10, 2, True),
        ]
        plan = (
            json.loads(args.plan.read_text())
            if args.plan
            else {"warm": scenarios, "cold": [scenarios[0], scenarios[2]]}
        )
        assert set(plan) == {"warm", "cold"}
        arms = [(binary, effort) for binary in processes for effort in [0.1, 0.25, 0.5]]
        metadata = {
            "baseline_head": args.baseline_head,
            "candidate_head": args.candidate_head,
            "queries_sha256": hashlib.file_digest(
                args.queries.open("rb"), "sha256"
            ).hexdigest(),
            "datasets": json.loads(args.datasets.read_text()),
            "plan": plan,
            "truth_sha256": (
                hashlib.file_digest(args.truth.open("rb"), "sha256").hexdigest()
                if args.truth
                else None
            ),
            "index_cache_bytes_per_worker": args.index_cache_gib * 1024**3,
            "setups": [],
            "id_format": "little-endian uint64; ids_offset is a byte offset",
            "evaluation_ids": queries["query_ids"][measured].tolist(),
            "timing_queries": args.timing_queries,
            "timing_repeats": args.repeats,
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
                selected = plan[cache]
                for index, k, refine, filtered in selected:
                    for binary, process in processes.items():
                        identities[binary] = exchange(
                            process, {"op": "setup", "index": index, "cache": cache}
                        )
                        setup = {
                            "binary": binary,
                            "index": index,
                            "cache": cache,
                            "k": k,
                            **identities[binary],
                        }
                        metadata["setups"].append(setup)
                        print(json.dumps({"event": "setup", **setup}), flush=True)
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
                            for repeat in range(args.repeats)
                            for p in measured[: args.timing_queries]
                        ]
                        + [(int(p), 0, False) for p in measured[args.timing_queries :]]
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
                                default_ids[binary] = result["ids_sha256"]
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
        for process in processes.values():
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("native worker failed after query execution")
        metadata["ids_sha256"] = {
            label: hashlib.file_digest(
                (args.output / f"{label}.ids").open("rb"), "sha256"
            ).hexdigest()
            for label in processes
        }
        metadata["full_job_wall_seconds"] = time.monotonic() - start
        (args.output / "identity.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    finally:
        for process in processes.values():
            if not process.stdin.closed:
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
    run.add_argument("--plan", type=Path)
    run.add_argument("--recall-queries", type=int, default=128)
    run.add_argument("--timing-queries", type=int, default=64)
    run.add_argument("--repeats", type=int, default=3)
    child = commands.add_parser("worker")
    child.add_argument("--runtime", type=Path, required=True)
    child.add_argument("--ids", type=Path, required=True)
    for command in [run, child]:
        command.add_argument("--queries", type=Path, required=True)
        command.add_argument("--datasets", type=Path, required=True)
        command.add_argument("--truth", type=Path)
        command.add_argument("--index-cache-gib", type=int, default=48)
    args = parser.parse_args()
    worker(args) if args.command == "worker" else measure(args)
