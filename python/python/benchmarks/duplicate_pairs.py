# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Profile index-only duplicate pairs against immutable local or S3 fixtures.

Use `prepare` once on a new, dedicated prefix, then run each measured invocation
in a fresh process. `run` never writes to the source dataset. Its trace records
source-store I/O separately from process disk I/O and total network traffic.
"""

import argparse
import hashlib
import json
import os
import resource
import threading
import time
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.vector import find_duplicate_pairs_in_partition


def credentials(path):
    if path:
        value = json.loads(Path(path).read_text())
        for key, name in [
            ("AccessKeyId", "AWS_ACCESS_KEY_ID"),
            ("SecretAccessKey", "AWS_SECRET_ACCESS_KEY"),
            ("SessionToken", "AWS_SESSION_TOKEN"),
        ]:
            os.environ[name] = value[key]
    os.environ.setdefault("AWS_REGION", "us-east-1")


def prepare(args):
    cases = []
    for rows in [6408, 17783]:
        rng = np.random.default_rng(9535)
        vectors = rng.normal(size=(rows, 1536)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        # Known duplicate groups give the sparse threshold meaningful output.
        for start in range(0, rows, 32):
            vectors[start : min(start + 16, rows)] = vectors[start]
        table = pa.table(
            {
                "id": pa.array(np.arange(rows, dtype=np.uint64)),
                "latents": pa.FixedSizeListArray.from_arrays(
                    pa.array(vectors.reshape(-1)), 1536
                ),
            }
        )
        for bits in [5, 8]:
            uri = f"{args.prefix.rstrip('/')}/rq{bits}-{rows}.lance"
            # Refuse to overwrite an existing table. This command is only for
            # creating dedicated synthetic fixtures, never production inputs.
            ds = lance.write_dataset(
                table,
                uri,
                mode="create",
                max_rows_per_file=4096,
                max_rows_per_group=4096,
            )
            ds = ds.create_index(
                "latents",
                "IVF_RQ",
                metric="cosine",
                num_bits=bits,
                num_partitions=1,
                ivf_centroids=np.full((1, 1536), 1 / np.sqrt(1536), dtype=np.float32),
            )
            segment = str(ds.describe_indices()[0].segments[0].uuid)
            cases.append(
                dict(
                    uri=uri,
                    version=ds.version,
                    segment_id=segment,
                    partition_id=0,
                    column="latents",
                    rows=rows,
                    dimension=1536,
                    num_bits=bits,
                    metric="cosine",
                    seed=9535,
                )
            )
            Path(args.output).write_text(json.dumps(cases, indent=2))


# Any threshold at or above this value must emit every pair of the partition.
ALL_PAIRS_THRESHOLD = 1e9


def pair_hash_sum(batch):
    a = batch.column(0).to_numpy(zero_copy_only=True)
    b = batch.column(1).to_numpy(zero_copy_only=True)
    x = np.minimum(a, b) * np.uint64(0x9E3779B97F4A7C15) ^ np.maximum(a, b)
    x ^= x >> np.uint64(33)
    x *= np.uint64(0xFF51AFD7ED558CCD)
    x ^= x >> np.uint64(33)
    return x.sum(dtype=np.uint64)


def process_snapshot():
    stats = resource.getrusage(resource.RUSAGE_SELF)
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value, _ = line.split()
            status[key.rstrip(":")] = int(value) * 1024
    io = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":")
        io[key] = int(value)
    received = sent = 0
    for device in Path("/sys/class/net").iterdir():
        if device.name == "lo":
            continue
        received += int((device / "statistics/rx_bytes").read_text())
        sent += int((device / "statistics/tx_bytes").read_text())
    return dict(
        cpu_s=stats.ru_utime + stats.ru_stime,
        rss_bytes=status["VmRSS"],
        peak_rss_bytes=status["VmHWM"],
        disk_read_bytes=io["read_bytes"],
        disk_write_bytes=io["write_bytes"],
        process_rchar=io["rchar"],
        process_wchar=io["wchar"],
        host_rx_bytes=received,
        host_tx_bytes=sent,
    )


def source_snapshot(dataset):
    stats = dataset.io_stats_snapshot()
    return dict(
        source_read_bytes=stats.read_bytes,
        source_read_iops=stats.read_iops,
        source_write_iops=stats.write_iops,
        source_written_bytes=stats.written_bytes,
    )


def run(args):
    case = json.loads(Path(args.config).read_text())[args.case]
    open_start = time.perf_counter()
    dataset = lance.dataset(case["uri"], version=case["version"])
    open_s = time.perf_counter() - open_start
    dataset.io_stats_incremental()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trace = []
    stop = threading.Event()
    start = time.perf_counter()

    def sample():
        trace.append(
            dict(
                t_s=time.perf_counter() - start,
                **process_snapshot(),
                **source_snapshot(dataset),
            )
        )

    def monitor():
        while not stop.wait(0.1):
            sample()

    sample()
    observer = threading.Thread(target=monitor, daemon=True)
    observer.start()
    hashes = [hashlib.sha256() for _ in range(3)]
    # Order-independent multiset hash of unordered ID pairs, so implementations
    # with different (but deterministic) emission orders can be compared.
    pair_set_hash = 0
    count = 0
    batches = 0
    kwargs = {}
    if args.memory_limit is not None:
        kwargs["memory_limit"] = args.memory_limit
    if args.max_concurrency is not None:
        kwargs["max_concurrency"] = args.max_concurrency
    try:
        with find_duplicate_pairs_in_partition(
            dataset,
            case["column"],
            case["segment_id"],
            case["partition_id"],
            args.threshold,
            **kwargs,
        ) as reader:
            for batch in reader:
                count += batch.num_rows
                batches += 1
                for column, digest in zip(batch.columns, hashes):
                    digest.update(column.to_numpy(zero_copy_only=True).tobytes())
                pair_set_hash = (pair_set_hash + int(pair_hash_sum(batch))) % 2**64
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        observer.join()
        sample()
    final = source_snapshot(dataset)
    assert final["source_write_iops"] == 0
    assert final["source_written_bytes"] == 0
    possible_pairs = case["rows"] * (case["rows"] - 1) // 2
    if args.threshold >= ALL_PAIRS_THRESHOLD:
        assert count == possible_pairs, (count, possible_pairs)
    source_windows = [
        (a, b)
        for a, b in zip(trace, trace[1:])
        if b["source_read_bytes"] > a["source_read_bytes"]
    ]
    result = dict(
        label=args.label,
        case=case,
        threshold=args.threshold,
        memory_limit=args.memory_limit,
        max_concurrency=args.max_concurrency,
        package_file=lance.__file__,
        dataset_open_s=open_s,
        elapsed_s=elapsed,
        possible_pairs=possible_pairs,
        output_rows=count,
        output_batches=batches,
        output_sha256=[h.hexdigest() for h in hashes],
        pair_set_hash=f"{pair_set_hash:016x}",
        pairs_per_s=possible_pairs / elapsed,
        cpu_s=trace[-1]["cpu_s"] - trace[0]["cpu_s"],
        cpu_cores=(trace[-1]["cpu_s"] - trace[0]["cpu_s"]) / elapsed,
        start_rss_bytes=trace[0]["rss_bytes"],
        peak_rss_bytes=max(point["peak_rss_bytes"] for point in trace),
        **final,
    )
    for key in [
        "disk_read_bytes",
        "disk_write_bytes",
        "process_rchar",
        "process_wchar",
        "host_rx_bytes",
        "host_tx_bytes",
    ]:
        result[key] = trace[-1][key] - trace[0][key]
    result["source_last_read_observed_s"] = (
        source_windows[-1][1]["t_s"] if source_windows else 0
    )
    result["source_peak_sample_bytes_per_s"] = (
        max(
            ((b["source_read_bytes"] - a["source_read_bytes"]) / (b["t_s"] - a["t_s"]))
            for a, b in source_windows
        )
        if source_windows
        else 0
    )
    output.with_suffix(".trace.json").write_text(json.dumps(trace))
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--credentials", help="Private AWS credential JSON; never logged"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("prepare")
    create.add_argument("--prefix", required=True)
    create.add_argument("--output", required=True)
    measure = commands.add_parser("run")
    measure.add_argument("--config", required=True)
    measure.add_argument("--case", type=int, required=True)
    measure.add_argument("--threshold", type=float, required=True)
    measure.add_argument("--label", required=True)
    measure.add_argument("--output", required=True)
    measure.add_argument("--memory-limit", type=int)
    measure.add_argument("--max-concurrency", type=int)
    args = parser.parse_args()
    credentials(args.credentials)
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
