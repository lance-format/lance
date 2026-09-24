# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Read-only S3 bandwidth reference for duplicate_pairs.py (Linux).

Repeated range GETs measure reachable throughput, not a hardware-rated ceiling
or Lance scan throughput. Run on the benchmark host with no other workloads.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from duplicate_pairs import credentials, process_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=10)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    credentials(args.credentials)
    case = json.loads(Path(args.config).read_text())[-1]
    uri = urlparse(case["uri"])
    if uri.scheme != "s3":
        parser.error("the reference requires an S3 fixture")
    client = boto3.client("s3", config=Config(max_pool_connections=16))
    objects = client.list_objects_v2(
        Bucket=uri.netloc, Prefix=uri.path.lstrip("/") + "/data/"
    )["Contents"]
    source = max(objects, key=lambda value: value["Size"])
    size = min(8 * 1024 * 1024, source["Size"])
    results = []
    for concurrency in [1, 8, 16]:
        before = process_snapshot()
        start = time.perf_counter()
        deadline = start + args.seconds

        def read():
            read_bytes = requests = 0
            while time.perf_counter() < deadline:
                response = client.get_object(
                    Bucket=uri.netloc,
                    Key=source["Key"],
                    Range=f"bytes=0-{size - 1}",
                    IfMatch=source["ETag"],
                )
                with response["Body"] as body:
                    while chunk := body.read(1024 * 1024):
                        read_bytes += len(chunk)
                requests += 1
            return read_bytes, requests

        with ThreadPoolExecutor(max_workers=concurrency) as workers:
            counts = list(workers.map(lambda _: read(), range(concurrency)))
        elapsed = time.perf_counter() - start
        after = process_snapshot()
        read_bytes = sum(item[0] for item in counts)
        results.append(
            dict(
                concurrency=concurrency,
                elapsed_s=elapsed,
                read_bytes=read_bytes,
                bytes_per_s=read_bytes / elapsed,
                requests=sum(item[1] for item in counts),
                cpu_cores=(after["cpu_s"] - before["cpu_s"]) / elapsed,
                host_rx_bytes=after["host_rx_bytes"] - before["host_rx_bytes"],
                peak_rss_bytes=after["peak_rss_bytes"],
            )
        )
    Path(args.output).write_text(
        json.dumps(
            dict(
                source_uri=f"s3://{uri.netloc}/{source['Key']}",
                etag=source["ETag"],
                range_bytes=size,
                measurements=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
