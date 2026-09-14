# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import random
from pathlib import Path

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pytest

NUM_ROWS = 10_000
READER_BRIDGE_ROWS = 100_000


@pytest.mark.parametrize(
    "array_factory",
    [
        lambda: pa.array(range(NUM_ROWS), type=pa.int32()),
        lambda: pc.random(NUM_ROWS),
        lambda: pa.array(
            [random.choice(["hello", "world", "today"]) for _ in range(NUM_ROWS)],
            type=pa.string(),
        ),
        lambda: pa.array(
            [random.choice(["hello", "world", "today"]) for _ in range(NUM_ROWS)],
            type=pa.dictionary(pa.int8(), pa.string()),
        ),
        lambda: pa.FixedSizeListArray.from_arrays(
            pc.random(NUM_ROWS * 128).cast(pa.float32()), 128
        ),
    ],
    ids=["i32", "f64", "string", "dictionary", "vector"],
)
@pytest.mark.benchmark(group="scan_single_column")
def test_scan_integer(tmp_path: Path, benchmark, array_factory):
    values = array_factory()
    table = pa.table({"values": values})
    dataset = lance.write_dataset(table, tmp_path)

    result = benchmark(
        dataset.to_table,
    )

    assert result.num_rows == NUM_ROWS


@pytest.fixture(scope="module")
def sample_dataset(tmpdir_factory):
    tmp_path = Path(tmpdir_factory.mktemp("data"))
    table = pa.table(
        {
            "i": pa.array(range(NUM_ROWS), type=pa.int32()),
            "f": pc.random(NUM_ROWS).cast(pa.float32()),
            "s": pa.array(
                [random.choice(["hello", "world", "today"]) for _ in range(NUM_ROWS)],
                type=pa.string(),
            ),
            "fsl": pa.FixedSizeListArray.from_arrays(
                pc.random(NUM_ROWS * 128).cast(pa.float32()), 128
            ),
            "blob": pa.array(
                [
                    random.choice(
                        [
                            random.randbytes(100 * 1024),
                            random.randbytes(100 * 1024),
                            random.randbytes(100 * 1024),
                        ]
                    )
                    for _ in range(NUM_ROWS)
                ],
                type=pa.binary(),
            ),
        }
    )

    return lance.write_dataset(table, tmp_path)


@pytest.fixture(scope="module")
def reader_bridge_dataset(tmpdir_factory):
    tmp_path = Path(tmpdir_factory.mktemp("reader_bridge"))
    table = pa.table({"value": pa.array(range(READER_BRIDGE_ROWS), type=pa.int32())})
    return lance.write_dataset(table, tmp_path)


@pytest.mark.parametrize("batch_size", [64, 1024, 8192, 65536])
@pytest.mark.parametrize(
    "columns",
    [pytest.param([], id="zero_columns"), pytest.param(["value"], id="i32")],
)
@pytest.mark.benchmark(group="scan_reader_bridge")
def test_scan_reader_bridge(benchmark, reader_bridge_dataset, columns, batch_size):
    scanner = reader_bridge_dataset.scanner(columns=columns, batch_size=batch_size)

    def consume_reader():
        return sum(batch.num_rows for batch in scanner.to_reader())

    assert benchmark(consume_reader) == READER_BRIDGE_ROWS


@pytest.mark.benchmark(group="scan_table")
def test_scan_table_full(benchmark, sample_dataset):
    result = benchmark(
        sample_dataset.to_table,
    )

    assert result.num_rows == NUM_ROWS


@pytest.mark.benchmark(group="scan_table")
def test_scan_table_project(benchmark, sample_dataset):
    result = benchmark(sample_dataset.to_table, columns=["i", "f"])

    assert result.schema.names == ["i", "f"]
    assert result.num_rows == NUM_ROWS


@pytest.mark.parametrize("keep_percent", [0.1, 0.5, 0.9])
@pytest.mark.benchmark(group="scan_table")
def test_scan_table_filter_project(benchmark, sample_dataset, keep_percent):
    result = benchmark(
        sample_dataset.to_table,
        filter=f"f < {keep_percent}",
        columns=["i", "blob"],
    )

    assert result.schema.names == ["i", "blob"]


@pytest.mark.parametrize("keep_percent", [0.1, 0.5, 0.9])
@pytest.mark.benchmark(group="scan_table")
def test_scan_table_filter_full(benchmark, sample_dataset, keep_percent):
    result = benchmark(
        sample_dataset.to_table,
        filter=f"f < {keep_percent}",
    )

    assert result.schema.names == ["i", "f", "s", "fsl", "blob"]


@pytest.mark.benchmark(group="filter_table")
def test_filter_for_range(benchmark, sample_dataset):
    result = benchmark(
        sample_dataset.to_table,
        filter="i > 1000 and i < 5000",
    )

    assert result.schema.names == ["i", "f", "s", "fsl", "blob"]


@pytest.mark.benchmark(group="filter_table")
def test_filter_for_row(benchmark, sample_dataset):
    result = benchmark(
        sample_dataset.to_table,
        filter="i = 4200",
    )

    assert result.num_rows == 1
    assert result.schema.names == ["i", "f", "s", "fsl", "blob"]


@pytest.mark.benchmark(group="filter_table")
def test_filter_multiple(benchmark, sample_dataset):
    result = benchmark(
        sample_dataset.to_table,
        filter="i > 1000 and i < 5000 and s in ('hello', 'world')",
    )

    assert result.num_rows > 1
    assert result.schema.names == ["i", "f", "s", "fsl", "blob"]
