# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import time
from datetime import datetime, timedelta, timezone, tzinfo

import lance
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

    # Error: compared_against combined with a version range
    with pytest.raises(
        ValueError,
        match="Invalid user input: Cannot combine compared_against_version",
    ):
        ds.delta(compared_against=1, begin_version=1, end_version=1)


@pytest.mark.parametrize("method", ["get_inserted_rows", "get_updated_rows"])
@pytest.mark.parametrize("begin_version", [0, 1, 3])
def test_delta_row_changes_require_stable_row_ids(method, begin_version):
    ds = write_dataset(pa.table({"id": [1, 2]}), "memory://", max_rows_per_file=1)
    ds.insert(pa.table({"id": [3]}))
    ds.update({"id": "4"}, where="id = 1")
    delta = ds.delta(begin_version=begin_version, end_version=3)

    # Transaction history is available without row-level version tracking.
    assert len(delta.list_transactions()) == 3 - begin_version
    with pytest.raises(ValueError, match="require stable row ids, version"):
        getattr(delta, method)()


@pytest.mark.parametrize(
    "column", ["_row_created_at_version", "_row_last_updated_at_version"]
)
@pytest.mark.parametrize(
    "read,error_type",
    [
        ("projection", ValueError),
        ("expression", ValueError),
        ("filter", ValueError),
        ("take", OSError),
    ],
)
def test_row_version_columns_require_stable_row_ids(column, read, error_type):
    ds = write_dataset(pa.table({"id": [1, 2]}), "memory://", max_rows_per_file=1)
    ds.insert(pa.table({"id": [3]}))

    with pytest.raises(error_type, match="requires stable row ids"):
        if read == "projection":
            ds.to_table(columns=["id", column])
        elif read == "expression":
            ds.to_table(columns={"version": f"{column} + 1"})
        elif read == "filter":
            ds.to_table(filter=f"{column} > 1")
        else:
            ds.take([2], columns=[column])


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

    # Aware values name instants. The same window in another zone still
    # selects the write that happened inside it.
    shifted_begin = begin.astimezone(timezone(timedelta(hours=8)))
    shifted_end = end.astimezone(timezone(timedelta(hours=-5)))
    inserted = (
        ds.delta(begin_date=shifted_begin, end_date=shifted_end)
        .get_inserted_rows()
        .read_all()
    )
    assert inserted.column("id").to_pylist() == [3]

    # Naive datetimes are local wall times, matching versions() and asof.
    # Dropping the zone from a UTC value would name a different instant
    # wherever the machine is not on UTC.
    naive_begin = begin.astimezone().replace(tzinfo=None)
    naive_end = end.astimezone().replace(tzinfo=None)
    inserted = (
        ds.delta(begin_date=naive_begin, end_date=naive_end)
        .get_inserted_rows()
        .read_all()
    )
    assert inserted.column("id").to_pylist() == [3]


def test_date_delta_after_external_append(tmp_path):
    uri = tmp_path / "data.lance"
    original = write_dataset(pa.table({"id": [1]}), uri, enable_stable_row_ids=True)
    begin = datetime.now(timezone.utc)
    writer = lance.dataset(uri)
    writer.insert(pa.table({"id": [2]}))
    assert original.version == 1
    assert writer.version == 2

    delta = original.delta(begin_date=begin)
    assert len(delta.list_transactions()) == 1
    observed = delta.get_upserted_rows().read_all()["id"].to_pylist()
    assert observed == [2]


class _NoOffsetTzInfo(tzinfo):
    """A tzinfo without a UTC offset; Python treats such datetimes as naive."""

    def utcoffset(self, dt):
        return None

    def dst(self, dt):
        return None

    def tzname(self, dt):
        return None


@pytest.fixture
def asia_shanghai_tz():
    # time.tzset is Unix-only, matching the guard in test_dataset.py.
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is Unix-only")
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Shanghai"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="time.tzset is Unix-only")
@pytest.mark.usefixtures("asia_shanghai_tz")
def test_delta_naive_version_timestamp_is_local_wall_time(tmp_path):
    """A naive version timestamp passed back as begin_date stays local.

    versions() and get_version() return naive local wall times. In
    Asia/Shanghai, 12:00 is 04:00Z. Labeling that naive value as UTC would
    move an exclusive begin to 12:00Z and skip commits over the next eight
    hours.
    """
    noon = datetime(2026, 6, 1, 12, 0)
    assert noon.utcoffset() is None
    assert noon.astimezone(timezone.utc) == datetime(
        2026, 6, 1, 4, 0, tzinfo=timezone.utc
    )

    ds = write_dataset(
        pa.table({"id": pa.array([1], type=pa.int32())}),
        tmp_path / "delta.lance",
        enable_stable_row_ids=True,
    )
    created = ds.get_version()["timestamp"]
    assert created.tzinfo is None
    assert created == ds.versions()[0]["timestamp"]
    # The same eight-hour shift as 12:00 local versus 12:00Z.
    assert created.astimezone(timezone.utc) == created.replace(
        tzinfo=timezone.utc
    ) - timedelta(hours=8)

    ds.insert(pa.table({"id": pa.array([2], type=pa.int32())}))

    # The commit timestamp, unchanged, is an exclusive begin. The following
    # write stays inside the window.
    inserted = ds.delta(begin_date=created).get_inserted_rows().read_all()
    assert inserted.column("id").to_pylist() == [2]

    # Aware UTC names the same instant.
    inserted = (
        ds.delta(begin_date=created.astimezone(timezone.utc))
        .get_inserted_rows()
        .read_all()
    )
    assert inserted.column("id").to_pylist() == [2]

    # A tzinfo with no offset is naive too, matching asof.
    unmarked = created.replace(tzinfo=_NoOffsetTzInfo())
    assert unmarked.tzinfo is not None
    assert unmarked.utcoffset() is None
    inserted = ds.delta(begin_date=unmarked).get_inserted_rows().read_all()
    assert inserted.column("id").to_pylist() == [2]


def _wait_until_after(stamp: datetime) -> None:
    """Block until the local clock moves past ``stamp``.

    An inclusive end keeps every commit in that microsecond. The commit that
    exercises the upper bound has to land in a later one.
    """
    deadline = time.monotonic() + 2.0
    while datetime.now() <= stamp:
        if time.monotonic() > deadline:
            break
        time.sleep(0.0002)


def test_delta_exposed_version_timestamps_round_trip(tmp_path):
    """get_version() timestamps are the exclusive begin and inclusive end.

    Create ids 1 and 9, append 2, delete 9, then a later append. (1, 3] is
    inserted id 2 and deleted row id 1. A membership check on 2 would still
    pass if version 1 leaked, and a begin that resolved to version 0 would
    report no deletions.
    """
    uri = tmp_path / "delta.lance"
    ds = write_dataset(pa.table({"id": [1, 9]}), uri, enable_stable_row_ids=True)
    ds.insert(pa.table({"id": [2]}))
    live = ds.to_table(columns=["id"], with_row_id=True)
    row_id_of_9 = dict(zip(live["id"].to_pylist(), live["_rowid"].to_pylist()))[9]
    assert row_id_of_9 == 1
    ds.delete("id = 9")
    end_commit = ds.get_version()["timestamp"]
    _wait_until_after(end_commit)
    ds.insert(pa.table({"id": [3]}))
    assert ds.version == 4

    begin = lance.dataset(uri, version=1).get_version()["timestamp"]
    end = lance.dataset(uri, version=3).get_version()["timestamp"]
    reported = {v["version"]: v["timestamp"] for v in ds.versions()}
    assert begin == reported[1]
    assert end == reported[3]
    assert begin.tzinfo is None
    assert end.tzinfo is None

    delta = ds.delta(begin_date=begin, end_date=end)
    assert len(delta.list_transactions()) == 2
    inserted = delta.get_inserted_rows().read_all()["id"].to_pylist()
    deleted = delta.get_deleted_row_ids().read_all()["_rowid"].to_pylist()
    assert inserted == [2]
    assert deleted == [1]

    by_version = ds.delta(begin_version=1, end_version=3)
    assert by_version.get_inserted_rows().read_all()["id"].to_pylist() == inserted
    assert by_version.get_deleted_row_ids().read_all()["_rowid"].to_pylist() == deleted
