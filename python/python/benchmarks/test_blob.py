# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from itertools import count
from time import process_time

import pyarrow as pa
import pytest
from lance.file import LanceFileSession

# Many small blobs isolate per-row Python overhead; fewer large blobs show how
# the bulk path behaves once payload copying accounts for more of the CPU time.
WORKLOADS = [
    pytest.param(50_000, 256, id="50000x256b"),
    pytest.param(2_000, 64 * 1024, id="2000x64kib"),
]


def _packed_blob_benchmark(benchmark, tmpdir_factory, row_count, payload_size, mode):
    payload = b"x" * payload_size
    payloads = pa.repeat(payload, row_count)
    files = LanceFileSession(str(tmpdir_factory.mktemp("packed_blob_writer")))
    file_number = count()

    if mode == "scalar_preconverted":
        python_payloads = payloads.to_pylist()

        def write():
            writer = files.open_packed_blob_writer(
                f"scalar-{next(file_number)}.lance", 1
            )
            for value in python_payloads:
                writer.write_blob(value)
            return writer.finish()

    elif mode == "scalar_from_arrow":

        def write():
            writer = files.open_packed_blob_writer(
                f"scalar-arrow-{next(file_number)}.lance", 1
            )
            for value in payloads.to_pylist():
                writer.write_blob(value)
            return writer.finish()

    elif mode == "bulk":

        def write():
            writer = files.open_packed_blob_writer(f"bulk-{next(file_number)}.lance", 1)
            writer.write_blobs(payloads)
            return writer.finish_array("blob")

    else:
        raise ValueError(f"Unknown benchmark mode: {mode}")

    result = benchmark.pedantic(write, iterations=1, rounds=5)
    assert len(result) == row_count


@pytest.mark.benchmark(group="packed_blob_writer_cpu", timer=process_time)
@pytest.mark.parametrize("row_count,payload_size", WORKLOADS)
@pytest.mark.parametrize(
    "mode",
    ["scalar_preconverted", "scalar_from_arrow", "bulk"],
)
def test_packed_blob_writer(benchmark, tmpdir_factory, row_count, payload_size, mode):
    _packed_blob_benchmark(
        benchmark,
        tmpdir_factory,
        row_count,
        payload_size,
        mode,
    )
