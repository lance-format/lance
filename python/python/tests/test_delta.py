# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa
import pytest
from lance import write_dataset


def test_delta_get_inserted_rows():
    # Create initial dataset (version 1)
    table1 = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "val": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )
    ds = write_dataset(table1, "memory://delta_api_test", enable_stable_row_ids=True)

    # Append more rows to create version 2
    table2 = pa.table(
        {
            "id": pa.array([4, 5], type=pa.int32()),
            "val": pa.array(["d", "e"], type=pa.string()),
        }
    )
    ds.insert(table2)

    # Build delta compared to v1 and fetch inserted rows
    delta = ds.delta(compared_against=1)
    print(delta.list_transactions())
    reader = delta.get_inserted_rows()

    # Sum rows from all batches
    total_rows = 0
    for batch in reader:
        total_rows += batch.num_rows

    assert total_rows == 2


def test_delta_get_updated_rows():
    # Create initial dataset (version 1)
    table1 = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "val": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )
    ds = write_dataset(
        table1, "memory://delta_api_test_update", enable_stable_row_ids=True
    )

    # Update an existing row to create version 2
    update_stats = ds.update({"val": "'b_updated'"}, where="id = 2")
    assert update_stats["num_rows_updated"] == 1

    # Build delta compared to v1 and fetch updated rows
    delta = ds.delta(compared_against=1)

    # Ensure the transaction is an Update (not an Append/Delete)
    txs = delta.list_transactions()
    assert len(txs) == 1
    assert type(txs[0].operation).__name__ == "Update"

    reader = delta.get_updated_rows()

    # Collect updated rows and validate contents
    total_rows = 0
    for batch in reader:
        total_rows += batch.num_rows

    assert total_rows == 1

    # Ensure no inserted rows are present in this diff
    inserted_reader = delta.get_inserted_rows()
    total_inserted = 0
    for batch in inserted_reader:
        total_inserted += batch.num_rows
    assert total_inserted == 0


def test_delta_with_explicit_version_range():
    # Create initial dataset (version 1)
    table1 = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "val": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )
    ds = write_dataset(
        table1, "memory://delta_version_range_test", enable_stable_row_ids=True
    )

    # Append more rows to create version 2
    table2 = pa.table(
        {
            "id": pa.array([4, 5], type=pa.int32()),
            "val": pa.array(["d", "e"], type=pa.string()),
        }
    )
    ds.insert(table2)

    # Use explicit version range instead of compared_against
    delta = ds.delta(begin_version=1, end_version=2)
    reader = delta.get_inserted_rows()

    total_rows = 0
    for batch in reader:
        total_rows += batch.num_rows

    assert total_rows == 2

    # Version 0 is the empty snapshot: (0, 1] holds the first write, (0, 0] nothing.
    inserted = ds.delta(begin_version=0, end_version=1).get_inserted_rows()
    assert sorted(inserted.read_all().column("id").to_pylist()) == [1, 2, 3]
    empty = ds.delta(begin_version=0, end_version=0).get_inserted_rows()
    assert empty.read_all().num_rows == 0


def test_delta_validation_errors():
    table = pa.table({"id": pa.array([1, 2, 3], type=pa.int32())})
    ds = write_dataset(table, "memory://delta_validation_test")

    # Error: no parameters specified
    with pytest.raises(ValueError, match="Must specify either"):
        ds.delta()

    # Error: only begin_version specified
    with pytest.raises(
        ValueError,
        match="Invalid user input: Must specify both with_begin_version "
        "and with_end_version",
    ):
        ds.delta(begin_version=1)

    # Error: only end_version specified
    with pytest.raises(
        ValueError,
        match="Invalid user input: Must specify both with_begin_version "
        "and with_end_version",
    ):
        ds.delta(end_version=2)


def test_delta_get_deleted_row_ids():
    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4], type=pa.int32()),
            "val": pa.array(["a", "b", "c", "d"], type=pa.string()),
        }
    )
    ds = write_dataset(
        table, "memory://delta_api_test_delete", enable_stable_row_ids=True
    )
    row_ids = ds.to_table(columns=[], with_row_id=True).column("_rowid").to_pylist()

    ds.delete("id in (2, 3)")

    delta = ds.delta(compared_against=1)
    reader = delta.get_deleted_row_ids()

    deleted = []
    for batch in reader:
        assert batch.schema.names == ["_rowid"]
        deleted.extend(batch.column("_rowid").to_pylist())

    assert sorted(deleted) == sorted([row_ids[1], row_ids[2]])


def test_delta_get_upserted_rows():
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "val": pa.array(["a", "b", "c"], type=pa.string()),
        }
    )
    ds = write_dataset(
        table, "memory://delta_api_test_upsert", enable_stable_row_ids=True
    )
    ds.update({"val": "'b_updated'"}, where="id = 2")
    ds.insert(
        pa.table(
            {
                "id": pa.array([4], type=pa.int32()),
                "val": pa.array(["d"], type=pa.string()),
            }
        )
    )

    rows = ds.delta(begin_version=1, end_version=3).get_upserted_rows().read_all()
    assert sorted(rows.column("id").to_pylist()) == [2, 4]


def test_delta_with_dates():
    from datetime import datetime, timedelta, timezone

    table = pa.table({"id": pa.array([1, 2], type=pa.int32())})
    ds = write_dataset(
        table, "memory://delta_api_test_dates", enable_stable_row_ids=True
    )
    # versions() returns a naive local datetime. Attach the local zone before
    # converting to UTC; labeling that wall time as UTC shifts the window.
    created = ds.versions()[0]["timestamp"]
    local_tz = datetime.now().astimezone().tzinfo
    created_utc = created.replace(tzinfo=local_tz).astimezone(timezone.utc)
    begin = created_utc + timedelta(microseconds=1)
    ds.insert(pa.table({"id": pa.array([3], type=pa.int32())}))

    end = datetime.now(timezone.utc) + timedelta(hours=1)
    inserted = ds.delta(begin_date=begin, end_date=end).get_inserted_rows().read_all()
    assert inserted.column("id").to_pylist() == [3]

    # Naive datetimes are treated as UTC, so drop the zone from a UTC value.
    naive_begin = begin.replace(tzinfo=None)
    naive_end = end.replace(tzinfo=None)
    inserted = (
        ds.delta(begin_date=naive_begin, end_date=naive_end)
        .get_inserted_rows()
        .read_all()
    )
    assert inserted.column("id").to_pylist() == [3]
