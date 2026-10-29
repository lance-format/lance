# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Qualify an independent worker and compare eager/lazy I/O without timing claims.

Run from python/: uv run python python/benchmarks/blob_reference_conformance.py
The temporary datasets and native reservations are released on completion.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import lance
import pyarrow as pa

FIXTURE = (
    Path(__file__).resolve().parents[3] / "test_data/blob_reference_v1/conformance.json"
)


def capture_dataset(path, values):
    table = pa.Table.from_arrays(
        [lance.blob_array(values)],
        schema=pa.schema(
            [
                lance.blob_field(
                    "blob", inline_size_threshold=0, dedicated_size_threshold=1
                )
            ]
        ),
    )
    dataset = lance.write_dataset(
        table,
        path,
        data_storage_version="2.2",
        max_rows_per_file=3,
    )
    source = lance.BlobSourceContext.from_dataset(
        dataset, "conformance/table", "worker"
    )
    retention = lance.SnapshotRetention.acquire(
        dataset, source.snapshot_identity(), str(uuid.uuid4())
    )
    try:
        before_scan = dataset.io_stats_snapshot()
        scanned = (
            dataset.to_table(with_row_address=True).combine_chunks().to_batches()[0]
        )
        after_scan = dataset.io_stats_snapshot()
        with lance.BlobRefCapture(
            dataset, source, dataset.lance_schema.field("blob").id()
        ) as capture:
            refs = capture.capture(
                source,
                scanned.schema.field("blob"),
                scanned["_rowaddr"],
                scanned["blob"],
            )
        after_capture = dataset.io_stats_snapshot()
        producer_io = {
            "descriptor_scan_requests": after_scan.read_iops - before_scan.read_iops,
            "descriptor_scan_bytes": after_scan.read_bytes - before_scan.read_bytes,
            "capture_requests": after_capture.read_iops - after_scan.read_iops,
            "capture_bytes": after_capture.read_bytes - after_scan.read_bytes,
        }
        assert producer_io["capture_requests"] == 0
        return dataset, retention, refs, producer_io
    except BaseException:
        retention.release(dataset)
        raise


def digest(values):
    result = hashlib.sha256()
    for value in values:
        if value is None:
            result.update(b"null\0")
        else:
            result.update(len(value).to_bytes(8, "little"))
            result.update(value)
    return result.hexdigest()


def worker(reference_path, job_path):
    required = {
        "read_whole",
        "read_range",
        "streaming",
        "cancellation",
        "reader_metrics_v1",
    }
    missing = required.difference(lance.BlobSourceContext.supported_operations())
    if missing:
        raise RuntimeError(f"Worker lacks required blob operations: {sorted(missing)}")
    refs = lance.BlobRefBatch.decode(Path(reference_path).read_bytes())
    source = refs.contexts[0]
    dataset = lance.dataset(source.table_uri, version=source.version)
    output = {}
    for job in json.loads(Path(job_path).read_text()):
        requests = job["requests"]
        with lance.BlobRefReader(
            [(source, dataset)],
            io_buffer_size_bytes=8 << 20,
            materialization_readahead_bytes=32 << 20,
        ) as reader:
            selected = {}
            with reader.read(
                refs,
                [request["reference_index"] for request in requests],
                [int(request["request_index"]) for request in requests],
                [
                    tuple(request["range"]) if request["range"] is not None else None
                    for request in requests
                ],
            ) as stream:
                for batch in stream:
                    index = batch["request_index"][0].as_py()
                    if index in job["output_indices"]:
                        value = batch["data"][0].as_py()
                        if value is not None and job.get("output_range"):
                            offset, length = job["output_range"]
                            value = value[offset : offset + length]
                        selected[index] = value
                    del batch
            output_values = (
                [str(len(refs)).encode()]
                if job.get("workload") == "metadata"
                else [selected[index] for index in job["output_indices"]]
            )
            actual = digest(output_values)
            assert actual == job["expected_hash"], job["name"]
            for bounds in job.get("invalid_ranges", []):
                assert_invalid_range(reader, refs, bounds)
            output[job["name"]] = {
                "output_hash": actual,
                "output_rows": len(output_values),
                **reader.metrics,
            }
    return output


def assert_invalid_range(reader, refs, bounds):
    try:
        reader.read(refs, [2], [0], [(int(bounds["offset"]), int(bounds["length"]))])
    except ValueError:
        return
    raise AssertionError(f"Accepted invalid range: {bounds}")


def independent_worker(directory, refs, jobs):
    reference_path = directory / "refs.bin"
    reference_path.write_bytes(refs.encode())
    job_path = directory / "jobs.json"
    job_path.write_text(json.dumps(jobs))
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", str(reference_path), str(job_path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(completed.stdout)


def qualify(directory):
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["protocol_version"] == lance.BlobSourceContext.protocol_version()
    values = [
        value.encode() if value is not None else None for value in fixture["payloads"]
    ]
    dataset, retention, refs, producer_io = capture_dataset(
        directory / "fixture.lance", values
    )
    try:
        requests = fixture["requests"]
        result = independent_worker(
            directory,
            refs,
            [
                {
                    "name": "conformance",
                    "requests": requests,
                    "output_indices": [
                        int(request["request_index"]) for request in requests
                    ],
                    "expected_hash": digest(
                        [
                            request["expected"].encode()
                            if request["expected"] is not None
                            else None
                            for request in requests
                        ]
                    ),
                    "invalid_ranges": fixture["invalid_ranges"],
                }
            ],
        )
        assert result["conformance"]["duplicate_demands"] == 1
        return {**result["conformance"], "producer_io": producer_io}
    finally:
        retention.release(dataset)


def compare(directory, rows, payload_bytes, selected_rows):
    values = [bytes([row % 251]) * payload_bytes for row in range(rows)]
    dataset, retention, refs, producer_io = capture_dataset(
        directory / "compare.lance", values
    )
    try:
        jobs = []
        for workload in ["metadata", "top_k", "range"]:
            output_indices = (
                [] if workload == "metadata" else list(range(selected_rows))
            )
            expected = (
                [str(rows).encode()]
                if workload == "metadata"
                else values[:selected_rows]
            )
            bounds = [7, min(16, payload_bytes - 7)] if workload == "range" else None
            if bounds:
                expected = [value[bounds[0] : sum(bounds)] for value in expected]
            for strategy in ["eager", "lazy"]:
                demands = list(range(rows)) if strategy == "eager" else output_indices
                jobs.append(
                    {
                        "name": f"{strategy}_{workload}",
                        "workload": workload,
                        "requests": [
                            {
                                "reference_index": index,
                                "request_index": str(index),
                                "range": bounds if strategy == "lazy" else None,
                            }
                            for index in demands
                        ],
                        "output_indices": output_indices,
                        "output_range": bounds if strategy == "eager" else None,
                        "expected_hash": digest(expected),
                    }
                )
        report = independent_worker(directory, refs, jobs)
        assert report["lazy_metadata"]["payload_requests"] == 0
        assert report["lazy_top_k"]["payload_requests"] == selected_rows
        assert (
            report["lazy_range"]["payload_bytes"]
            < report["eager_range"]["payload_bytes"]
        )
        for workload in ["metadata", "top_k", "range"]:
            assert (
                report[f"eager_{workload}"]["output_hash"]
                == report[f"lazy_{workload}"]["output_hash"]
            )
        ipc = pa.BufferOutputStream()
        payload_batch = pa.record_batch(
            [pa.array(values, type=pa.large_binary())], names=["blob"]
        )
        with pa.ipc.new_stream(ipc, payload_batch.schema) as writer:
            writer.write_batch(payload_batch)
        report["candidate_exchange"] = {
            "eager_payload_arrow_ipc_bytes": ipc.getvalue().size,
            "lazy_reference_envelope_bytes": len(refs.encode()),
        }
        report["producer_io"] = producer_io
        return report
    finally:
        retention.release(dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("REFERENCES", "JOBS"))
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--payload-bytes", type=int, default=65536)
    parser.add_argument("--selected-rows", type=int, default=3)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker), sort_keys=True))
        return
    if not 0 < args.selected_rows < args.rows or args.payload_bytes < 8:
        parser.error("require 0 < selected-rows < rows and payload-bytes >= 8")
    with tempfile.TemporaryDirectory(prefix="lance-blob-conformance-") as temporary:
        directory = Path(temporary)
        print(
            json.dumps(
                {
                    "conformance": qualify(directory),
                    "comparison": compare(
                        directory, args.rows, args.payload_bytes, args.selected_rows
                    ),
                    "scope": (
                        "Payload submissions and native reservations. Scan I/O, "
                        "Python copies, retries and engine exchange are separate. "
                        "Candidate sizes include one reference context envelope "
                        "or one payload Arrow stream. No latency measurement."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
