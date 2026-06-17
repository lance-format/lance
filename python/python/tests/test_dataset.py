# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import base64
import contextlib
import os
import pickle
import platform
import random
import re
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List
from unittest import mock

import lance
import lance.fragment
import numpy as np
import pandas as pd
import pandas.testing as tm
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import pytest
from helper import ProgressForTest
from lance._dataset.sharded_batch_iterator import ShardedBatchIterator
from lance.commit import CommitConflictError
from lance.dataset import LANCE_COMMIT_MESSAGE_KEY, AutoCleanupConfig
from lance.debug import format_fragment
from lance.file import stable_version
from lance.schema import LanceSchema
from lance.util import validate_vector_index

# Various valid inputs for write_dataset
input_schema = pa.schema([pa.field("a", pa.float64()), pa.field("b", pa.int64())])
input_data = [
    # (schema, data)
    (None, pa.table({"a": [1.0, 2.0], "b": [20, 30]})),
    (None, pa.record_batch([[1.0, 2.0], [20, 30]], names=["a", "b"])),
    (None, pd.DataFrame({"a": [1.0, 2.0], "b": [20, 30]})),
    (None, pl.DataFrame({"a": [1.0, 2.0], "b": [20, 30]})),
    (
        input_schema,
        [pa.record_batch([pa.array([1.0, 2.0]), pa.array([20, 30])], names=["a", "b"])],
    ),
    # Can provide an iterator with a schema that is different but cast-able
    (
        input_schema,
        iter(
            pa.table(
                {
                    "a": [1.0, 2.0],
                    "b": pa.array([20, 30], pa.int32()),
                }
            ).to_batches()
        ),
    ),
]


@pytest.mark.parametrize("schema,data", input_data, ids=type)
def test_input_data(tmp_path: Path, schema, data):
    base_dir = tmp_path / "test"
    dataset = lance.write_dataset(data, base_dir, schema=schema)
    assert dataset.to_table() == input_data[0][1]


def test_roundtrip_types(tmp_path: Path):
    table = pa.table(
        {
            "dict": pa.array(["a", "b", "a"], pa.dictionary(pa.int8(), pa.string())),
            # PyArrow doesn't support creating large_string dictionaries easily.
            "large_dict": pa.DictionaryArray.from_arrays(
                pa.array([0, 1, 1], pa.int8()),
                pa.array(["foo", "bar"], pa.large_string()),
            ),
            "list": pa.array(
                [["a", "b"], ["c", "d"], ["e", "f"]], pa.list_(pa.string())
            ),
            "large_list": pa.array(
                [["a", "b"], ["c", "d"], ["e", "f"]], pa.large_list(pa.string())
            ),
        }
    )

    # TODO: V2 does not currently handle large_dict
    dataset = lance.write_dataset(table, tmp_path, data_storage_version="legacy")
    assert dataset.schema == table.schema
    assert dataset.to_table() == table


def test_dataset_overwrite(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    table2 = pa.Table.from_pylist([{"s": "one"}, {"s": "two"}])
    lance.write_dataset(table2, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)
    assert dataset.to_table() == table2

    ds_v1 = lance.dataset(base_dir, version=1)
    assert ds_v1.to_table() == table1


def test_truncate_table(tmp_path: Path):
    base_dir = tmp_path / "truncate"
    table = pa.table(
        {
            "i": pa.array([1, 2, 3], pa.int32()),
            "dict": pa.DictionaryArray.from_arrays(
                pa.array([0, 1, 2], pa.uint16()), pa.array(["a", "b", "c"])
            ),
        }
    )
    ds = lance.write_dataset(table, base_dir, data_storage_version="stable")
    assert ds.count_rows() == 3

    ds.truncate_table()
    assert ds.count_rows() == 0
    assert ds.schema == table.schema


def test_dataset_append(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    # verify append works even if no dataset existed at the uri
    lance.write_dataset(table, base_dir, mode="append")
    dataset = lance.dataset(base_dir)
    assert dataset.to_table() == table

    # verify appending batches with a different schema doesn't work
    table2 = pa.Table.from_pydict({"COLUMN-C": [1, 2, 3], "colB": [4, 5, 6]})
    with pytest.raises(OSError):
        lance.write_dataset(table2, dataset, mode="append")

    # But we can append subschemas
    table3 = pa.Table.from_pydict({"colA": [4, 5, 6]})
    dataset.insert(table3)  # Append is default
    assert dataset.to_table() == pa.table(
        {"colA": [1, 2, 3, 4, 5, 6], "colB": [4, 5, 6, None, None, None]}
    )


def test_dataset_from_record_batch_iterable(tmp_path: Path):
    base_dir = tmp_path / "test"

    test_pylist = [{"colA": "Alice", "colB": 20}, {"colA": "Blob", "colB": 30}]

    # split into two batches
    batches = [
        pa.RecordBatch.from_pylist([test_pylist[0]]),
        pa.RecordBatch.from_pylist([test_pylist[1]]),
    ]

    # define schema
    schema = pa.schema(
        [
            pa.field("colA", pa.string()),
            pa.field("colB", pa.int64()),
        ]
    )

    # write dataset with iterator
    lance.write_dataset(iter(batches), base_dir, schema)
    dataset = lance.dataset(base_dir)

    # After combined into one batch, make sure it is the same as original pylist
    assert list(dataset.to_batches())[0].to_pylist() == test_pylist

    # write dataset with list
    lance.write_dataset(batches, base_dir, schema, mode="overwrite")

    # After combined into one batch, make sure it is the same as original pylist
    assert list(dataset.to_batches())[0].to_pylist() == test_pylist


def test_to_batches_with_partial_last_batch(tmp_path: Path):
    base_dir = tmp_path / "test_batches"
    row_count_per_file = 32
    batch_size = 5

    # Generate 3 batches of 32 rows each (96 total)
    pylist = [{"colA": f"Row{i}", "colB": i} for i in range(row_count_per_file * 3)]
    batches = [
        pa.RecordBatch.from_pylist(
            pylist[i * row_count_per_file : (i + 1) * row_count_per_file]
        )
        for i in range(3)
    ]

    # Write dataset
    schema = pa.schema([pa.field("colA", pa.string()), pa.field("colB", pa.int64())])
    lance.write_dataset(batches, base_dir, schema, max_rows_per_file=row_count_per_file)
    dataset = lance.dataset(base_dir)

    # Check batch sizes
    # strict_batch_size = True, batch_size = 5(batch_size < row_count_per_file),
    all_batches = list(
        dataset.to_batches(batch_size=batch_size, strict_batch_size=True)
    )
    print(all_batches)
    assert sum(b.num_rows for b in all_batches) == row_count_per_file * 3  # Total rows
    assert all(b.num_rows == batch_size for b in all_batches[:-1])  # Full batches
    assert all_batches[-1].num_rows == 1  # Final partial batch
    # Verify data integrity
    combined = [row for batch in all_batches for row in batch.to_pylist()]
    assert combined == pylist

    # strict_batch_size = True, batch_size = 5*10 (batch_size > row_count_per_file),
    large_batch_size = 10 * batch_size
    all_batches = list(
        dataset.to_batches(batch_size=large_batch_size, strict_batch_size=True)
    )
    assert sum(b.num_rows for b in all_batches) == row_count_per_file * 3  # Total rows
    assert all(b.num_rows == large_batch_size for b in all_batches[:-1])  # Full batches
    assert all_batches[-1].num_rows == 46  # Final partial batch

    # strict_batch_size = False
    # fragment 32 rows --> [5,5,5,5,5,5,2]
    all_batches = list(
        dataset.to_batches(batch_size=batch_size, strict_batch_size=False)
    )
    assert sum(b.num_rows for b in all_batches) == row_count_per_file * 3  # Total rows
    partial_batches = [b for b in all_batches if b.num_rows < batch_size]
    partial_batch_count = 3
    assert len(partial_batches) == partial_batch_count
    assert all(b.num_rows == 2 for b in partial_batches[:-1])

    # strict_batch_size = False, batch_size = 5*10 (batch_size > row_count_per_file),
    all_batches = list(
        dataset.to_batches(batch_size=large_batch_size, strict_batch_size=False)
    )
    assert sum(b.num_rows for b in all_batches) == row_count_per_file * 3  # Total rows
    assert all(b.num_rows == 32 for b in all_batches)  # Full batches
    # 32 rows, 1 row per file
    ds = lance.write_dataset(
        pa.table({"a": range(32)}), base_dir, max_rows_per_file=1, mode="overwrite"
    )
    # 1 row per batch if strict_batch_size is False (regardless of batch_size)
    for batch in ds.to_batches():
        assert batch.num_rows == 1
    for batch in ds.to_batches(batch_size=8):
        assert batch.num_rows == 1

    # We should get 8 rows per batch if strict_batch_size is True
    for batch in ds.to_batches(batch_size=8, strict_batch_size=True):
        assert batch.num_rows == 8


def test_schema_metadata(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("a", pa.int64(), metadata={b"thisis": "a"}),
            pa.field("b", pa.int64(), metadata={b"thisis": "b"}),
        ],
        metadata={b"foo": b"bar", b"baz": b"qux"},
    )
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)}, schema=schema)
    ds = lance.write_dataset(table, tmp_path)
    # Original schema
    assert ds.schema.metadata == {b"foo": b"bar", b"baz": b"qux"}
    assert ds.schema.field("a").metadata == {b"thisis": b"a"}
    assert ds.schema.field("b").metadata == {b"thisis": b"b"}

    # Replace schema metadata (using new unified API)
    ds.update_schema_metadata({"foo": "baz"}, replace=True)
    assert ds.schema.metadata == {b"foo": b"baz"}
    assert ds.schema.field("a").metadata == {b"thisis": b"a"}
    assert ds.schema.field("b").metadata == {b"thisis": b"b"}

    # Replace field metadata (using new unified API)
    # Use field path instead of field ID
    ds.update_field_metadata({"a": {"thisis": "c"}}, replace=True)
    assert ds.schema.field("a").metadata == {b"thisis": b"c"}
    assert ds.schema.field("b").metadata == {b"thisis": b"b"}

    # Overwrite overwrites metadata
    ds = lance.write_dataset(table, tmp_path, mode="overwrite")
    assert ds.schema.metadata == {b"foo": b"bar", b"baz": b"qux"}
    assert ds.schema.field("a").metadata == {b"thisis": b"a"}
    assert ds.schema.field("b").metadata == {b"thisis": b"b"}


def test_versions(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    assert len(lance.dataset(base_dir).versions()) == 1

    table2 = pa.Table.from_pylist([{"s": "one"}, {"s": "two"}])
    time.sleep(1)
    lance.write_dataset(table2, base_dir, mode="overwrite")

    assert len(lance.dataset(base_dir).versions()) == 2

    v1, v2 = lance.dataset(base_dir).versions()
    assert v1["version"] == 1
    assert v2["version"] == 2
    assert isinstance(v1["timestamp"], datetime)
    assert isinstance(v2["timestamp"], datetime)
    assert v1["timestamp"] < v2["timestamp"]
    assert isinstance(v1["metadata"], dict)
    assert isinstance(v2["metadata"], dict)


def test_version_id(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    original_ds = lance.write_dataset(table1, base_dir)

    assert original_ds.version == 1
    assert original_ds.latest_version == 1

    table2 = pa.Table.from_pylist([{"s": "one"}, {"s": "two"}])
    time.sleep(1)
    updated_ds = lance.write_dataset(table2, base_dir, mode="overwrite")

    assert original_ds.version == 1
    assert original_ds.latest_version == 2

    assert updated_ds.version == 2
    assert updated_ds.latest_version == 2

    historical_ds = updated_ds.checkout_version(1)
    assert historical_ds.version == 1
    assert historical_ds.latest_version == 2
    assert historical_ds.checkout_version(historical_ds.latest_version).version == 2


def test_checkout(tmp_path: Path):
    tab = pa.table({"a": range(3)})
    ds1 = lance.write_dataset(tab, tmp_path)
    ds1.delete("a = 1")

    ds2 = ds1.checkout_version(1)

    assert ds2.version == 1
    assert ds2.to_table() == tab

    assert ds1.version == 2
    assert ds1.to_table() == pa.table({"a": [0, 2]})

    ds1.delete("a = 2")
    assert ds1.count_rows() == 1

    assert ds2.checkout_version(ds2.latest_version).version == ds1.version


def test_asof_checkout(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    assert len(lance.dataset(base_dir).versions()) == 1
    time.sleep(0.1)
    ts_1 = datetime.now()
    time.sleep(0.1)

    lance.write_dataset(table, base_dir, mode="append")
    assert len(lance.dataset(base_dir).versions()) == 2
    time.sleep(0.1)
    ts_2 = datetime.now()
    time.sleep(0.1)

    lance.write_dataset(table, base_dir, mode="append")
    assert len(lance.dataset(base_dir).versions()) == 3
    time.sleep(0.1)
    ts_3 = datetime.now()

    # check that only the first batch is present
    ds = lance.dataset(base_dir, asof=ts_1)
    assert ds.version == 1
    assert len(ds.to_table()) == 3

    # check that the first and second batch are present
    ds = lance.dataset(base_dir, asof=ts_2)
    assert ds.version == 2
    assert len(ds.to_table()) == 6

    # check that all batches are present
    ds = lance.dataset(base_dir, asof=ts_3)
    assert ds.version == 3
    assert len(ds.to_table()) == 9


def test_enable_stable_row_ids(tmp_path: Path):
    table = pa.Table.from_pylist(
        [{"name": "Alice", "age": 20}, {"name": "Bob", "age": 30}]
    )
    lance.write_dataset(table, tmp_path, enable_stable_row_ids=True)
    ds = lance.write_dataset(table, tmp_path, enable_stable_row_ids=True, mode="append")
    table_before = ds.scanner(with_row_id=True, with_row_address=True).to_table()
    assert len(table_before) == 4
    assert table_before["_rowid"][0].as_py() == 0
    assert table_before["_rowid"][1].as_py() == 1
    assert table_before["_rowid"][2].as_py() == 2
    assert table_before["_rowid"][3].as_py() == 3

    assert table_before["_rowaddr"][0].as_py() == 0
    assert table_before["_rowaddr"][1].as_py() == 1
    assert table_before["_rowaddr"][2].as_py() == (1 << 32) + 0
    assert table_before["_rowaddr"][3].as_py() == (1 << 32) + 1

    ds.optimize.compact_files()

    table_after = ds.scanner(with_row_id=True, with_row_address=True).to_table()
    assert len(table_after) == 4
    assert table_after["_rowid"][0].as_py() == 0
    assert table_after["_rowid"][1].as_py() == 1
    assert table_after["_rowid"][2].as_py() == 2
    assert table_after["_rowid"][3].as_py() == 3

    assert table_after["_rowaddr"][0].as_py() == (2 << 32) + 0
    assert table_after["_rowaddr"][1].as_py() == (2 << 32) + 1
    assert table_after["_rowaddr"][2].as_py() == (2 << 32) + 2
    assert table_after["_rowaddr"][3].as_py() == (2 << 32) + 3


@pytest.mark.parametrize("enable_stable_row_ids", [True, False])
@pytest.mark.parametrize(
    "rows",
    [[{"a": 1}, {"a": 2}], []],
    ids=["non_empty", "empty"],
)
def test_has_stable_row_ids_property(tmp_path: Path, enable_stable_row_ids: bool, rows):
    schema = pa.schema([pa.field("a", pa.int64())])
    table = pa.Table.from_pylist(rows, schema=schema)

    path = tmp_path / f"stable_row_ids_{enable_stable_row_ids}_{len(rows)}"
    lance.write_dataset(
        table,
        path,
        enable_stable_row_ids=enable_stable_row_ids,
    )
    ds = lance.dataset(path)

    assert ds.count_rows() == len(rows)
    assert len(ds.get_fragments()) == (0 if len(rows) == 0 else 1)
    assert ds.has_stable_row_ids is enable_stable_row_ids


def _list_manifests(versions_dir):
    # Ignore the version hint file, which is not a manifest.
    return [
        name
        for name in os.listdir(versions_dir)
        if not name.startswith("latest_version_hint")
    ]


def test_v2_manifest_paths(tmp_path: Path):
    lance.write_dataset(
        pa.table({"a": range(100)}), tmp_path, enable_v2_manifest_paths=True
    )
    manifest_path = _list_manifests(tmp_path / "_versions")
    assert len(manifest_path) == 1
    assert re.match(r"\d{20}\.manifest", manifest_path[0])


def test_default_v2_manifest_paths(tmp_path: Path):
    lance.write_dataset(pa.table({"a": range(100)}), tmp_path)
    manifest_path = _list_manifests(tmp_path / "_versions")
    assert len(manifest_path) == 1
    assert re.match(r"\d{20}\.manifest", manifest_path[0])


def test_v2_manifest_paths_migration(tmp_path: Path):
    # Create a dataset with v1 manifest paths
    lance.write_dataset(
        pa.table({"a": range(100)}), tmp_path, enable_v2_manifest_paths=False
    )
    manifest_path = _list_manifests(tmp_path / "_versions")
    assert manifest_path == ["1.manifest"]

    # Migrate to v2 manifest paths
    lance.dataset(tmp_path).migrate_manifest_paths_v2()
    manifest_path = _list_manifests(tmp_path / "_versions")
    assert len(manifest_path) == 1
    assert re.match(r"\d{20}\.manifest", manifest_path[0])


def test_tag(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    ds = lance.write_dataset(table, base_dir, mode="append")

    assert len(ds.tags.list()) == 0

    with pytest.raises(ValueError):
        ds.tags.create("tag1", 3)

    with pytest.raises(ValueError):
        ds.tags.delete("tag1")

    ds.tags.create("tag1", 1)
    ds.tags.replace_metadata("tag1", {"description": "first tag"})
    tag1_meta = ds.tags.list()["tag1"]
    assert tag1_meta["created_at"] is not None
    assert isinstance(tag1_meta["created_at"], datetime)
    assert tag1_meta["updated_at"] is not None
    assert isinstance(tag1_meta["updated_at"], datetime)
    assert tag1_meta["created_at"] == tag1_meta["updated_at"]
    assert len(ds.tags.list()) == 1
    assert ds.tags.list()["tag1"]["metadata"] == {"description": "first tag"}

    with pytest.raises(ValueError):
        ds.tags.create("tag1", 1)

    ds.tags.delete("tag1")

    ds.tags.create("tag1", 1)
    ds.tags.create("tag2", 1)

    assert len(ds.tags.list()) == 2

    with pytest.raises(OSError):
        ds.checkout_version("tag3")

    assert ds.checkout_version("tag1").version == 1

    ds = lance.dataset(base_dir, "tag1")
    assert ds.version == 1

    with pytest.raises(ValueError):
        lance.dataset(base_dir, "missing-tag")

    # test tag update
    with pytest.raises(
        ValueError, match="Version not found error: version main:3 does not exist"
    ):
        ds.tags.update("tag1", 3)

    with pytest.raises(
        ValueError, match="Ref not found error: tag tag3 does not exist"
    ):
        ds.tags.update("tag3", 1)

    tag1_meta = ds.tags.list()["tag1"]
    tag1_created_at = tag1_meta["created_at"]
    tag1_updated_at = tag1_meta["updated_at"]
    assert tag1_created_at is not None
    assert tag1_updated_at is not None
    ds.tags.replace_metadata("tag1", {"description": "updated tag"})
    ds = lance.dataset(base_dir, "tag1")
    assert ds.version == 1
    replaced_tag1_meta = ds.tags.list()["tag1"]
    assert replaced_tag1_meta["metadata"] == {"description": "updated tag"}
    assert replaced_tag1_meta["updated_at"] == tag1_updated_at

    ds.tags.replace_metadata("tag1", {"owner": "ml-team"})
    replaced_again_tag1_meta = ds.tags.list()["tag1"]
    assert replaced_again_tag1_meta["metadata"] == {"owner": "ml-team"}
    assert replaced_again_tag1_meta["updated_at"] == tag1_updated_at

    ds.tags.update("tag1", 2)
    updated_tag1_meta = ds.tags.list()["tag1"]
    assert updated_tag1_meta["created_at"] == tag1_created_at
    assert updated_tag1_meta["updated_at"] is not None
    assert updated_tag1_meta["updated_at"] >= tag1_updated_at
    ds = lance.dataset(base_dir, "tag1")
    assert ds.version == 2
    assert ds.tags.list()["tag1"]["metadata"] == {"owner": "ml-team"}

    ds.tags.replace_metadata("tag1", {})
    ds = lance.dataset(base_dir, "tag1")
    assert ds.version == 2
    assert ds.tags.list()["tag1"]["metadata"] == {}

    ds.tags.update("tag1", 1)
    ds = lance.dataset(base_dir, "tag1")
    assert ds.version == 1
    assert ds.tags.list()["tag1"]["metadata"] == {}

    version = ds.tags.get_version("tag1")
    assert version == 1

    ds.create_branch("branch", "tag1")
    ds.tags.create("tag3", ("branch", None))
    target_tag = ds.tags.list().get("tag3")
    assert ds.tags.get_version("tag3") == 1
    assert len(ds.tags.list()) == 3
    assert target_tag is not None
    assert target_tag["version"] == 1
    assert target_tag["branch"] == "branch"

    ds.tags.update("tag3", (None, 2))
    target_tag = ds.tags.list()["tag3"]
    assert ds.tags.get_version("tag3") == 2
    assert target_tag is not None
    assert target_tag["version"] == 2
    assert target_tag["branch"] is None

    ds.create_branch("branch2", 2)
    ds.tags.update("tag3", ("branch2", 2))
    target_tag = ds.tags.list()["tag3"]
    assert ds.tags.get_version("tag3") == 2
    assert target_tag is not None
    assert target_tag["version"] == 2
    assert target_tag["branch"] == "branch2"

    ds.tags.delete("tag3")
    assert len(ds.tags.list()) == 2


def test_tag_order(tmp_path: Path):
    table = pa.Table.from_pydict({"colA": [1, 2, 3], "colB": [4, 5, 6]})
    base_dir = tmp_path / "test"

    for i in range(3):
        mode = "append" if i > 0 else "create"
        ds = lance.write_dataset(table, base_dir, mode=mode)

    expected_tags = {"tag3": 3, "tag2": 2, "tag1": 1}
    for name, version in expected_tags.items():
        ds.tags.create(name, version)

    tags_asc = ds.tags.list_ordered(order="asc")
    assert len(tags_asc) == 3
    first_tag = tags_asc[0][1]
    assert first_tag["created_at"] is not None
    assert isinstance(first_tag["created_at"], datetime)
    assert first_tag["updated_at"] is not None
    assert isinstance(first_tag["updated_at"], datetime)
    tag_names_asc = [t[0] for t in tags_asc]
    assert tag_names_asc == sorted(expected_tags.keys()), (
        f"Unexpected ascending order: {tag_names_asc}"
    )

    # Test descending order (default)
    tags_desc = ds.tags.list_ordered(order="desc")
    assert len(tags_desc) == 3
    tag_names_desc = [t[0] for t in tags_desc]
    assert tag_names_desc == list(expected_tags.keys()), (
        f"Unexpected descending order: {tag_names_desc}"
    )

    # Test without parameter (should default to descending)
    tags_default = ds.tags.list_ordered()
    assert tags_default == tags_desc, "Default order should match descending order"


def test_sample(tmp_path: Path):
    table1 = pa.Table.from_pydict({"x": [0, 10, 20, 30, 40, 50], "y": range(6)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    sampled = dataset.sample(3)

    assert sampled.num_rows == 3
    assert sampled.num_columns == 2

    sampled = dataset.sample(4, columns=["x"])

    assert sampled.num_rows == 4
    assert sampled.num_columns == 1

    for row in sampled.column(0).chunk(0):
        assert row.as_py() % 10 == 0


def test_take(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0, 1])

    assert isinstance(table2, pa.Table)
    assert table2 == table1


def test_take_rowid_rowaddr(tmp_path: Path):
    sample_size = 10
    table1 = pa.table({"a": range(1000), "b": range(1000)})
    base_dir = tmp_path / "test_take_rowid_rowaddr"
    lance.write_dataset(
        table1, base_dir, enable_stable_row_ids=False, max_rows_per_file=50
    )
    dataset = lance.dataset(base_dir)
    total_rows = len(dataset)
    sampled_indices = random.sample(range(total_rows), min(sample_size, total_rows))

    sample_dataset = dataset.take(sampled_indices, columns=["_rowid"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = dataset.take(sampled_indices, columns=["_rowid", "_rowid"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = dataset.take([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowid"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = dataset.take(sampled_indices, columns=["_rowaddr"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = dataset.take(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowaddr"]
    )
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 1

    sample_dataset = dataset.take(sampled_indices, columns=["_rowaddr", "_rowid"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = dataset.take(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowaddr", "_rowid"]
    )
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = dataset.take(sampled_indices, columns=["_rowid", "_rowaddr"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = dataset.take(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowid", "_rowaddr"]
    )
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = dataset.take(sampled_indices, columns=["a", "_rowid", "_rowaddr"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 3

    sample_dataset = dataset.take(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["a", "_rowid", "_rowaddr"]
    )
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 3

    sample_dataset = dataset.take(sampled_indices, columns=["_rowid", "_rowaddr", "b"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 3

    sample_dataset = dataset.take(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["_rowid", "_rowaddr", "b"]
    )
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 3

    sample_dataset = dataset.take(sampled_indices, columns=["a", "b"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2

    sample_dataset = dataset.take([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], columns=["a", "b"])
    assert sample_dataset.num_rows == 10
    assert sample_dataset.num_columns == 2


@pytest.mark.parametrize(
    "column_name",
    [
        "_rowid",
        "_rowaddr",
        "_rowoffset",
        "_row_created_at_version",
        "_row_last_updated_at_version",
    ],
)
def test_take_system_columns_values(tmp_path: Path, column_name: str):
    """Test that system columns return correct values in take."""
    table = pa.table({"a": range(100), "b": range(100, 200)})
    base_dir = tmp_path / "test_take_system_columns_values"
    # Use max_rows_per_file to create multiple fragments
    lance.write_dataset(table, base_dir, max_rows_per_file=25)
    dataset = lance.dataset(base_dir)

    indices = [0, 5, 10, 50, 99]
    result = dataset.take(indices, columns=[column_name, "a"])
    assert result.num_rows == len(indices)
    assert result.schema.names == [column_name, "a"]

    col_values = result.column(column_name).to_pylist()
    a_values = result.column("a").to_pylist()

    # Verify column type is UInt64
    assert result.column(column_name).type == pa.uint64()

    # Verify data column values
    assert a_values == indices

    # Verify system column values based on column type
    if column_name == "_rowid":
        # Without stable row IDs, _rowid equals _rowaddr (not the index).
        # Row address = (fragment_id << 32) | row_offset_within_fragment
        # With max_rows_per_file=25: frag0=0-24, frag1=25-49, frag2=50-74, frag3=75-99
        expected_rowids = [
            (0 << 32) | 0,  # index 0: fragment 0, offset 0
            (0 << 32) | 5,  # index 5: fragment 0, offset 5
            (0 << 32) | 10,  # index 10: fragment 0, offset 10
            (2 << 32) | 0,  # index 50: fragment 2, offset 0
            (3 << 32) | 24,  # index 99: fragment 3, offset 24
        ]
        assert col_values == expected_rowids
    elif column_name in ("_row_created_at_version", "_row_last_updated_at_version"):
        # All rows created/updated at version 1
        assert col_values == [1] * len(indices)
    # _rowaddr and _rowoffset values depend on fragment layout


def test_take_system_columns_column_ordering(tmp_path: Path):
    """Test that column ordering is preserved when using system columns."""
    table = pa.table({"a": range(50), "b": range(50, 100)})
    base_dir = tmp_path / "test_take_column_ordering"
    lance.write_dataset(table, base_dir)
    dataset = lance.dataset(base_dir)

    indices = [0, 1, 2]

    # Test different orderings with all system columns
    result = dataset.take(indices, columns=["_rowid", "a", "_rowaddr"])
    assert result.schema.names == ["_rowid", "a", "_rowaddr"]

    result = dataset.take(indices, columns=["a", "_rowaddr", "_rowid"])
    assert result.schema.names == ["a", "_rowaddr", "_rowid"]

    result = dataset.take(indices, columns=["_rowaddr", "_rowid", "b", "a"])
    assert result.schema.names == ["_rowaddr", "_rowid", "b", "a"]

    # Test with version columns
    result = dataset.take(
        indices,
        columns=[
            "_row_created_at_version",
            "a",
            "_row_last_updated_at_version",
            "_rowid",
        ],
    )
    assert result.schema.names == [
        "_row_created_at_version",
        "a",
        "_row_last_updated_at_version",
        "_rowid",
    ]

    # Test with all system columns in mixed order
    result = dataset.take(
        indices,
        columns=[
            "_rowoffset",
            "_row_last_updated_at_version",
            "b",
            "_rowaddr",
            "_row_created_at_version",
            "a",
            "_rowid",
        ],
    )
    assert result.schema.names == [
        "_rowoffset",
        "_row_last_updated_at_version",
        "b",
        "_rowaddr",
        "_row_created_at_version",
        "a",
        "_rowid",
    ]


def test_take_version_system_columns(tmp_path: Path):
    """Test _row_created_at_version and _row_last_updated_at_version columns."""
    table = pa.table({"a": range(50)})
    base_dir = tmp_path / "test_take_version_columns"
    lance.write_dataset(table, base_dir, enable_stable_row_ids=True)
    dataset = lance.dataset(base_dir)

    # Initial version is 1
    initial_version = dataset.version

    indices = [0, 10, 25]
    result = dataset.take(
        indices,
        columns=["a", "_row_created_at_version", "_row_last_updated_at_version"],
    )

    assert result.num_rows == 3
    created_at = result.column("_row_created_at_version").to_pylist()
    updated_at = result.column("_row_last_updated_at_version").to_pylist()

    # All rows were created and last updated at the initial version
    assert created_at == [initial_version] * 3
    assert updated_at == [initial_version] * 3

    # Now update some rows by overwriting
    table2 = pa.table({"a": range(50, 100)})
    lance.write_dataset(table2, base_dir, mode="append")
    dataset = lance.dataset(base_dir)

    # New rows should have version 2
    result = dataset.take([50, 60], columns=["_row_created_at_version"])
    created_at = result.column("_row_created_at_version").to_pylist()
    assert created_at == [dataset.version] * 2


@pytest.mark.parametrize("indices", [[], [1, 1], [1, 1, 20, 20, 21], [21, 0, 21, 1, 0]])
def test_take_duplicate_index(tmp_path: Path, indices: List[int]):
    table = pa.table({"x": range(24)})
    dataset = lance.write_dataset(
        table, tmp_path, max_rows_per_group=3, max_rows_per_file=9
    )
    expected = table.take(pa.array(indices, pa.int64()))

    assert dataset.take(indices) == expected


def test_take_with_columns(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0], columns=["b"])

    assert table2 == pa.Table.from_pylist([{"b": 2}])


def test_take_with_projection(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.take([0], columns={"a2": "a*2", "bup": "UPPER(b)"})

    assert table2 == pa.Table.from_pylist([{"a2": 2, "bup": "X"}])

    table3 = dataset._take_rows([0], columns={"a2": "a*2", "bup": "UPPER(b)"})
    assert table3 == table2


def test_filter(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    actual_tab = dataset.to_table(columns=["a"], filter=(pa.compute.field("b") > 50))
    assert actual_tab == pa.Table.from_pydict({"a": range(51, 100)})


def test_filter_meta_columns(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    ds = lance.write_dataset(table, base_dir)

    rowids = ds.to_table(with_row_id=True, columns=[])
    some_row_id = random.sample(rowids.column(0).to_pylist(), 1)[0]
    filtered = ds.to_table(filter=f"_rowid = {some_row_id}", with_row_id=True)

    assert len(filtered) == 1

    rowaddrs = ds.to_table(with_row_address=True, columns=[])
    some_row_addr = random.sample(rowaddrs.column(0).to_pylist(), 1)[0]
    filtered = ds.to_table(filter=f"_rowaddr = {some_row_addr}", with_row_address=True)

    assert len(filtered) == 1


@pytest.mark.parametrize("data_storage_version", ["legacy", "stable"])
def test_limit_offset(tmp_path: Path, data_storage_version: str):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(
        table, base_dir, data_storage_version=data_storage_version, max_rows_per_file=10
    )
    dataset = lance.dataset(base_dir)

    # test just limit
    assert dataset.to_table(limit=10) == table.slice(0, 10)
    assert dataset.to_table(limit=100) == table.slice(0, 100)

    # test just offset
    assert dataset.to_table(offset=0) == table.slice(0, 100)
    assert dataset.to_table(offset=10) == table.slice(10, 90)

    # test both
    assert dataset.to_table(offset=10, limit=10) == table.slice(10, 10)

    # Slicing in the middle of fragments
    assert dataset.to_table(offset=5, limit=20) == table.slice(5, 20)

    # Slicing within a single fragment
    assert dataset.to_table(offset=5, limit=3) == table.slice(5, 3)

    # Skipping entire fragments
    assert dataset.to_table(offset=50, limit=25) == table.slice(50, 25)

    # Limit past the end
    assert dataset.to_table(limit=101) == table.slice(0, 100)

    # Limit with offset past the end
    assert dataset.to_table(offset=50, limit=51) == table.slice(50, 50)

    # Offset past the end
    assert dataset.to_table(offset=100) == table.slice(100, 0)  # Empty table
    assert dataset.to_table(offset=101) == table.slice(100, 0)  # Empty table

    # Offset with limit past the end
    assert dataset.to_table(offset=100, limit=1) == table.slice(100, 0)  # Empty table
    assert dataset.to_table(offset=101, limit=1) == table.slice(100, 0)  # Empty table

    # Invalid limit / offset
    with pytest.raises(ValueError, match="Offset must be non-negative"):
        assert dataset.to_table(offset=-1, limit=10) == table.slice(50, 50)
    with pytest.raises(ValueError, match="Limit must be non-negative"):
        assert dataset.to_table(offset=10, limit=-1) == table.slice(50, 50)

    full_ds_version = dataset.version
    dataset.delete("a % 2 = 0")
    filt_table = table.filter((pa.compute.bit_wise_and(pa.compute.field("a"), 1)) != 0)
    assert (
        dataset.to_table(offset=10).combine_chunks()
        == filt_table.slice(10).combine_chunks()
    )

    dataset = dataset.checkout_version(full_ds_version)
    dataset.restore()
    dataset.delete("a > 2 AND a < 7")
    filt_table = table.slice(7, 1)

    assert dataset.to_table(offset=3, limit=1) == filt_table


def test_relative_paths(tmp_path: Path):
    # relative paths get coerced to the full absolute path
    current_dir = os.getcwd()
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    rel_uri = "test.lance"
    try:
        os.chdir(tmp_path)
        lance.write_dataset(table, rel_uri)

        # relative path works in the current dir
        ds = lance.dataset(rel_uri)
        assert ds.to_table() == table
    finally:
        os.chdir(current_dir)

    # relative path doesn't work in the context of a different dir
    with pytest.raises(ValueError):
        ds = lance.dataset(rel_uri)

    # relative path gets resolved to the right absolute path
    ds = lance.dataset(tmp_path / rel_uri)
    assert ds.to_table() == table


@pytest.mark.skipif(
    (platform.system() == "Windows"),
    reason="mocking user's home folder it not working",
)
def test_tilde_paths(tmp_path: Path):
    # tilde paths get resolved to the right absolute path
    tilde_uri = "~/test.lance"
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})

    with mock.patch.dict(
        os.environ, {"HOME": str(tmp_path), "USERPROFILE": str(tmp_path)}
    ):
        # NOTE: the resolution logic is a bit finicky
        # link 1 - https://docs.rs/dirs/4.0.0/dirs/fn.home_dir.html
        # link 2 - https://docs.python.org/3/library/os.path.html#os.path.expanduser
        expected_abs_path = os.path.expanduser(tilde_uri)
        assert expected_abs_path == os.fspath(tmp_path / "test.lance")

        lance.write_dataset(table, tilde_uri)
        # works in the current context
        ds = lance.dataset(tilde_uri)
        assert ds.to_table() == table

    # works with the resolved absolute path
    ds = lance.dataset(expected_abs_path)
    assert ds.to_table() == table


def test_to_batches(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    batches = dataset.to_batches(batch_readahead=20)
    assert pa.Table.from_batches(batches) == table

    unordered_batches = dataset.to_batches(batch_readahead=20, scan_in_order=False)
    sorted_batches = pa.Table.from_batches(unordered_batches).sort_by("a")
    assert sorted_batches == table


def test_list_from_parquet(tmp_path: Path):
    # This is a regression for GH-1482, the parquet reader creates
    # list fields with the name 'element' instead of 'item'.  We should
    # ignore that
    tab = pa.Table.from_pydict(
        {"x": pa.array([[1, 2], [3, 4]], pa.list_(pa.float32(), 2))}
    )
    pq.write_table(tab, tmp_path / "foo.parquet")
    tab = pq.read_table(tmp_path / "foo.parquet")
    lance.write_dataset(tab, tmp_path / "foo.lance")


def test_pickle(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    pickled = pickle.dumps(dataset)
    unpickled = pickle.loads(pickled)
    assert dataset.to_table() == unpickled.to_table()


def test_nested_projection(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "a": range(100),
            "b": range(100),
            "struct": [{"x": counter, "y": counter % 2 == 0} for counter in range(100)],
        }
    )
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)

    projected = dataset.to_table(columns=["struct.x"])
    assert projected == pa.Table.from_pydict({"struct.x": range(100)})

    projected = dataset.to_table(columns=["struct.y"])
    assert projected == pa.Table.from_pydict(
        {"struct.y": [i % 2 == 0 for i in range(100)]}
    )


def test_nested_projection_list(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "a": range(100),
            "b": range(100),
            "list_struct": [
                [{"x": counter, "y": counter % 2 == 0}] for counter in range(100)
            ],
        }
    )
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)

    projected = dataset.to_table(columns={"list_struct": "list_struct[1]['x']"})
    assert projected == pa.Table.from_pydict({"list_struct": range(100)})

    # FIXME: sqlparser seems to ignore the .y part, but I can't create a simple
    # reproducible example for sqlparser. Possibly an issue in our dialect.
    # projected = dataset.to_table(
    #   columns={"list_struct": "array_element(list_struct, 1).y"})
    # assert projected == pa.Table.from_pydict(
    #     {"list_struct": [i % 2 == 0 for i in range(100)]}
    # )


def test_polar_scan(tmp_path: Path):
    some_structs = [{"x": counter, "y": counter} for counter in range(100)]
    table = pa.Table.from_pydict(
        {
            "a": range(100),
            "b": range(100),
            "struct": some_structs,
        }
    )
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    polars_df = pl.scan_pyarrow_dataset(dataset)
    df = dataset.to_table().to_pandas()
    tm.assert_frame_equal(polars_df.collect().to_pandas(), df)

    # Note, this doesn't verify that the filter is actually pushed down.
    # It only checks that, if the filter is pushed down, we interpret it
    # correctly.
    def check_pushdown_filt(pl_filt, sql_filt):
        polars_df = pl.scan_pyarrow_dataset(dataset).filter(pl_filt)
        df = dataset.to_table(filter=sql_filt).to_pandas()
        tm.assert_frame_equal(polars_df.collect().to_pandas(), df)

    # These three should push down (but we don't verify)
    check_pushdown_filt(pl.col("a") > 50, "a > 50")
    check_pushdown_filt(~(pl.col("a") > 50), "a <= 50")
    check_pushdown_filt(pl.col("a").is_in([50, 51, 52]), "a IN (50, 51, 52)")
    # At the current moment it seems polars cannot pushdown this
    # kind of filter
    check_pushdown_filt((pl.col("a") + 3) < 100, "(a + 3) < 100")

    # I can't seem to get struct["x"] to work in Lance but maybe there is
    # a way.  For now, let's compare it directly to the pyarrow compute version

    # Doesn't yet work today :( due to upstream issue (datafusion's substrait parser
    # doesn't yet handle nested refs)
    # if pa.cpp_version_info.major >= 14:
    #     polars_df = pl.scan_pyarrow_dataset(dataset).filter(pl.col("struct.x") < 10)
    #     df = dataset.to_table(filter=pc.field("struct", "x") < 10).to_pandas()
    #     tm.assert_frame_equal(polars_df.collect().to_pandas(), df)


def test_count_fragments(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    count_fragments = dataset.stats.dataset_stats()["num_fragments"]
    assert count_fragments == 1


def test_count_rows(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    count_rows = dataset.count_rows()
    assert count_rows == 100

    assert dataset.count_rows(filter="a < 50") == 50


def test_count_rows_via_scanner(tmp_path: Path):
    ds = lance.write_dataset(pa.table({"a": range(100), "b": range(100)}), tmp_path)

    assert ds.scanner(filter="a < 50", columns=[], with_row_id=True).count_rows() == 50
    assert (
        ds.scanner(filter="a < 50", columns=["a"], with_row_id=True).count_rows() == 50
    )
    assert ds.scanner(with_row_id=True).count_rows() == 100
    assert ds.scanner(columns=[]).count_rows() == 100
    assert ds.scanner().count_rows() == 100


def test_select_none(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    ds = lance.write_dataset(table, base_dir)

    assert "projection=[a]" in ds.scanner(
        columns=[], filter="a < 50", with_row_id=True
    ).explain_plan(True)


def test_analyze_filtered_scan(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    ds = lance.write_dataset(table, base_dir)
    plan = ds.scanner(columns=[], filter="a < 50", with_row_id=True).analyze_plan()
    assert re.search(
        r"^\s*LanceRead:.*output_rows=50.*rows_scanned=100.*$", plan, re.MULTILINE
    )


def test_analyze_index_scan(tmp_path: Path):
    table = pa.table({"filter": range(100)})
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_scalar_index("filter", "BTREE")
    plan = dataset.scanner(filter="filter = 10").analyze_plan()
    assert re.search(
        r"^\s*LanceRead:.*output_rows=1.*rows_scanned=1.*$", plan, re.MULTILINE
    )


def test_analyze_scan(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(table, tmp_path)
    plan = dataset.scanner().analyze_plan()
    assert re.search(
        r"^\s*LanceRead:.*bytes_read=\d+.*iops=\d+.*requests=\d+.*$", plan, re.MULTILINE
    )


def test_analyze_take(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_scalar_index("a", "BTREE")
    plan = dataset.scanner(filter="a = 50").analyze_plan()
    assert re.search(
        r"^\s*LanceRead:.*bytes_read=\d+.*iops=\d+.*requests=\d+.*$", plan, re.MULTILINE
    )


def test_analyze_vector_search(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "id": [i for i in range(10)],
            "vector": pa.array(
                [[1.0, 1.0] for _ in range(10)], pa.list_(pa.float32(), 2)
            ),
        }
    )
    dataset = lance.write_dataset(table, tmp_path / "dataset", mode="create")
    dataset.delete("id = 0")
    plan = dataset.scanner(
        nearest={"column": "vector", "k": 10, "q": [1.0, 1.0]}
    ).analyze_plan()
    assert "KNNVectorDistance:" in plan
    assert "metric=l2" in plan
    assert "output_rows=10" in plan


def test_get_fragments(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragment = dataset.get_fragments()[0]
    assert fragment.count_rows() == 100
    assert fragment.physical_rows == 100
    assert fragment.num_deletions == 0

    assert fragment.metadata.id == 0

    head = fragment.head(10)
    tm.assert_frame_equal(head.to_pandas(), table.to_pandas()[0:10])

    assert fragment.to_table() == table

    taken = fragment.take([18, 20, 33, 53])
    assert taken == pa.Table.from_pydict({"a": [18, 20, 33, 53], "b": [18, 20, 33, 53]})


def test_pickle_fragment(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)

    dataset = lance.dataset(base_dir)
    fragment = dataset.get_fragments()[0]
    pickled = pickle.dumps(fragment)
    unpickled = pickle.loads(pickled)

    assert fragment.to_table() == unpickled.to_table()


def test_cleanup_old_versions(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    time.sleep(0.1)
    moment = datetime.now()
    lance.write_dataset(table, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)

    # These calls do nothing, but make sure we can call the method ok

    # Ok, defaults to two weeks ago
    stats = dataset.cleanup_old_versions()
    assert stats.bytes_removed == 0
    assert stats.old_versions == 0

    # Ok, can accept timedelta
    dataset.cleanup_old_versions(older_than=timedelta(days=14))

    # print(tmp_path)
    # for root, dirnames, filenames in os.walk(tmp_path):
    #     for filename in filenames:
    #         print(root + "/" + filename)

    # Now this call will actually delete the old version
    stats = dataset.cleanup_old_versions(older_than=(datetime.now() - moment))
    assert stats.bytes_removed > 0
    assert stats.old_versions == 1


def test_cleanup_error_when_tagged_old_versions(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="overwrite")
    time.sleep(0.1)
    moment = datetime.now()
    lance.write_dataset(table, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)
    dataset.tags.create("old-tag", 1)
    dataset.tags.create("another-old-tag", 2)

    with pytest.raises(OSError):
        dataset.cleanup_old_versions(older_than=(datetime.now() - moment))
    assert len(dataset.versions()) == 3

    dataset.tags.delete("old-tag")
    with pytest.raises(OSError):
        dataset.cleanup_old_versions(older_than=(datetime.now() - moment))
    assert len(dataset.versions()) == 3

    dataset.tags.delete("another-old-tag")
    stats = dataset.cleanup_old_versions(older_than=(datetime.now() - moment))
    assert stats.bytes_removed > 0
    assert stats.old_versions == 2
    assert len(dataset.versions()) == 1


def test_cleanup_around_tagged_old_versions(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="overwrite")
    time.sleep(0.1)
    moment = datetime.now()
    lance.write_dataset(table, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)
    dataset.tags.create("old-tag", 1)
    dataset.tags.create("another-old-tag", 2)
    dataset.tags.create("tag-latest", 3)

    stats = dataset.cleanup_old_versions(
        older_than=(datetime.now() - moment), error_if_tagged_old_versions=False
    )
    assert stats.bytes_removed == 0
    assert stats.old_versions == 0

    dataset.tags.delete("old-tag")
    stats = dataset.cleanup_old_versions(
        older_than=(datetime.now() - moment), error_if_tagged_old_versions=False
    )
    assert stats.bytes_removed > 0
    assert stats.old_versions == 1

    dataset.tags.delete("another-old-tag")
    stats = dataset.cleanup_old_versions(
        older_than=(datetime.now() - moment), error_if_tagged_old_versions=False
    )
    assert stats.bytes_removed > 0
    assert stats.old_versions == 1


def test_cleanup_with_retain_versions(tmp_path: Path):
    base_dir = tmp_path / "cleanup_policy"
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    lance.write_dataset(table, base_dir, mode="create")
    time.sleep(0.05)
    lance.write_dataset(table, base_dir, mode="overwrite")
    time.sleep(0.05)
    lance.write_dataset(table, base_dir, mode="overwrite")
    time.sleep(0.05)
    ds = lance.write_dataset(table, base_dir, mode="append")

    assert len(ds.versions()) == 4
    stats = ds.cleanup_old_versions(retain_versions=3)
    assert stats.old_versions == 1
    assert stats.data_files_removed == 1
    assert stats.transaction_files_removed == 1
    assert stats.index_files_removed == 0
    assert stats.deletion_files_removed == 0
    assert len(ds.versions()) == 3
    assert ds.count_rows() == len(ds.to_table())


def test_cleanup_with_older_than_and_retain_versions(tmp_path: Path):
    base_dir = tmp_path / "cleanup_policy"
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    lance.write_dataset(table, base_dir, mode="create")
    time.sleep(0.05)
    lance.write_dataset(table, base_dir, mode="overwrite")
    time.sleep(0.05)
    lance.write_dataset(table, base_dir, mode="overwrite")
    moment = datetime.now()
    time.sleep(0.05)
    ds = lance.write_dataset(table, base_dir, mode="append")

    stats = ds.cleanup_old_versions(
        older_than=datetime.now() - moment, retain_versions=2
    )
    assert stats.old_versions == 2
    assert len(ds.versions()) == 2
    assert ds.count_rows() == len(ds.to_table())


def test_auto_cleanup(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    auto_cleanup_options = AutoCleanupConfig(
        interval=1,
        older_than_seconds=1,
    )
    lance.write_dataset(
        table, base_dir, auto_cleanup_options=auto_cleanup_options, mode="create"
    )
    lance.write_dataset(table, base_dir, mode="append")
    lance.write_dataset(table, base_dir, mode="append")
    lance.write_dataset(table, base_dir, mode="append")

    time.sleep(5)

    # trigger cleanup
    lance.write_dataset(table, base_dir, mode="append")
    dataset = lance.dataset(base_dir)
    assert len(dataset.versions()) == 2


def test_config_update_auto_cleanup(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    ds = lance.write_dataset(table, base_dir, mode="create")
    ds.update_config(
        {"lance.auto_cleanup.interval": "1", "lance.auto_cleanup.older_than": "1ms"}
    )
    lance.write_dataset(table, base_dir, mode="append")
    lance.write_dataset(table, base_dir, mode="append")
    lance.write_dataset(table, base_dir, mode="append")

    time.sleep(5)

    # trigger cleanup
    lance.write_dataset(table, base_dir, mode="append")
    dataset = lance.dataset(base_dir)
    assert len(dataset.versions()) == 2


def test_access_config(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    ds = lance.write_dataset(table, base_dir, mode="create")
    ds.update_config({"test_key": "test_value"})
    config_value = ds.config()["test_key"]
    assert config_value == "test_value"
    assert 1 == len(ds.config())

    ds.delete_config_keys(["test_key"])
    assert 0 == len(ds.config())


def test_auto_cleanup_invalid(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    auto_cleanup_options = AutoCleanupConfig(
        interval=1,
        older_than_seconds=1,
    )
    lance.write_dataset(table, base_dir, mode="create")
    lance.write_dataset(
        table, base_dir, auto_cleanup_options=auto_cleanup_options, mode="append"
    )
    lance.write_dataset(
        table, base_dir, auto_cleanup_options=auto_cleanup_options, mode="append"
    )

    time.sleep(3)

    lance.write_dataset(
        table, base_dir, auto_cleanup_options=auto_cleanup_options, mode="append"
    )
    dataset = lance.dataset(base_dir)
    assert len(dataset.versions()) == 4


def test_enable_disable_auto_cleanup(tmp_path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    ds = lance.write_dataset(table, base_dir, mode="create")
    auto_cleanup_options = AutoCleanupConfig(
        interval=1,
        older_than_seconds=1,
    )
    # enable auto cleanup
    ds.optimize.enable_auto_cleanup(auto_cleanup_options)
    lance.write_dataset(table, base_dir, mode="append")
    lance.write_dataset(table, base_dir, mode="append")
    lance.write_dataset(table, base_dir, mode="append")

    time.sleep(5)

    # trigger cleanup
    lance.write_dataset(table, base_dir, mode="append")
    assert len(ds.versions()) == 2

    # this is a transactional commit, so will increase a version
    ds.optimize.disable_auto_cleanup()

    lance.write_dataset(table, base_dir, mode="append")
    lance.write_dataset(table, base_dir, mode="append")
    lance.write_dataset(table, base_dir, mode="append")

    time.sleep(5)

    # wait to see if cleanup would be trigger
    lance.write_dataset(table, base_dir, mode="append")

    assert len(ds.versions()) == 7


def test_cleanup_with_rate_limit(tmp_path):
    """Test that cleanup_old_versions works with delete_rate_limit parameter."""
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir, mode="create")
    lance.write_dataset(table, base_dir, mode="overwrite")
    lance.write_dataset(table, base_dir, mode="overwrite")
    lance.write_dataset(table, base_dir, mode="overwrite")

    dataset = lance.dataset(base_dir)
    latest_version_timestamp = dataset.versions()[-1]["timestamp"]
    now = (
        datetime.now(latest_version_timestamp.tzinfo)
        if latest_version_timestamp.tzinfo is not None
        else datetime.now()
    )

    start = time.time_ns()
    # Cleanup with a rate limit should still remove old versions correctly
    stats = dataset.cleanup_old_versions(
        older_than=(now - latest_version_timestamp), delete_rate_limit=1
    )
    finished = time.time_ns()

    assert stats.old_versions == 3
    assert stats.bytes_removed > 0
    assert (finished - start) >= 2_000_000_000  # 2s


def test_create_from_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    fragment = lance.fragment.LanceFragment.create(base_dir, table)

    operation = lance.LanceOperation.Overwrite(table.schema, [fragment])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    tbl = dataset.to_table()
    assert tbl == table


def test_strict_overwrite(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    fragment = lance.fragment.LanceFragment.create(base_dir, table)
    operation = lance.LanceOperation.Overwrite(table.schema, [fragment])
    dataset_v1 = lance.LanceDataset.commit(base_dir, operation, max_retries=0)
    lance.LanceDataset.commit(
        base_dir, operation, read_version=dataset_v1.version, max_retries=0
    )
    with pytest.raises(
        OSError, match=f"Commit conflict for version {dataset_v1.version + 1}"
    ):
        lance.LanceDataset.commit(
            base_dir, operation, read_version=dataset_v1.version, max_retries=0
        )


def test_commit_timeout(tmp_path: Path):
    from datetime import timedelta

    table = pa.Table.from_pydict({"a": range(10)})
    base_dir = tmp_path / "timeout"
    dataset = lance.write_dataset(table, base_dir)

    fragment = lance.fragment.LanceFragment.create(base_dir, table)
    append = lance.LanceOperation.Append([fragment])

    # A zero duration reaches Rust and is rejected as invalid input.
    with pytest.raises(OSError, match="non-zero"):
        lance.LanceDataset.commit(
            dataset, append, read_version=1, commit_timeout=timedelta(0)
        )

    # A negative duration is rejected by PyO3's timedelta -> Duration conversion.
    with pytest.raises(ValueError):
        lance.LanceDataset.commit(
            dataset, append, read_version=1, commit_timeout=timedelta(seconds=-1)
        )

    # None disables the timeout.
    dataset_no_timeout = lance.LanceDataset.commit(
        dataset, append, read_version=1, commit_timeout=None
    )
    assert dataset_no_timeout.version == dataset.version + 1

    # Explicit positive timeout works.
    fragment2 = lance.fragment.LanceFragment.create(base_dir, table)
    append2 = lance.LanceOperation.Append([fragment2])
    dataset_with_timeout = lance.LanceDataset.commit(
        dataset_no_timeout,
        append2,
        read_version=dataset_no_timeout.version,
        commit_timeout=timedelta(minutes=1),
    )
    assert dataset_with_timeout.version == dataset_no_timeout.version + 1

    # Timeout *firing* behavior is covered by the Rust test
    # `test_commit_timeout_triggers`, which uses a throttled store for a
    # reliable trigger; reproducing it from Python without exposing
    # throttling would be flaky on fast runners.


def test_append_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    fragment = lance.fragment.LanceFragment.create(base_dir, table)
    append = lance.LanceOperation.Append([fragment])

    with pytest.raises(ValueError):
        # Must specify read version
        dataset = lance.LanceDataset.commit(dataset, append)

    dataset = lance.LanceDataset.commit(dataset, append, read_version=1)

    tbl = dataset.to_table()

    expected = pa.Table.from_pydict(
        {
            "a": list(range(100)) + list(range(100)),
            "b": list(range(100)) + list(range(100)),
        }
    )
    assert tbl == expected


def test_commit_batch_append():
    data1 = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(data1, "memory://test")

    data2 = pa.Table.from_pydict({"a": range(100, 200), "b": range(100, 200)})
    fragments2 = lance.fragment.write_fragments(data2, dataset)
    op2 = lance.LanceOperation.Append(fragments2)
    txn2 = lance.Transaction(1, op2)
    data3 = pa.Table.from_pydict({"a": range(200, 300), "b": range(200, 300)})
    fragments3 = lance.fragment.write_fragments(data3, dataset)
    op3 = lance.LanceOperation.Append(fragments3)
    txn3 = lance.Transaction(2, op3)

    result = lance.LanceDataset.commit_batch(dataset, [txn2, txn3])
    dataset = result["dataset"]
    assert dataset.version == 2
    assert dataset.checkout_version(1).version == 1
    assert len(dataset.get_fragments()) == 3
    assert dataset.to_table() == pa.concat_tables([data1, data2, data3])
    merged_txn = result["merged"]
    assert isinstance(merged_txn, lance.Transaction)
    assert isinstance(merged_txn.operation, lance.LanceOperation.Append)
    assert merged_txn.operation.fragments == fragments2 + fragments3


def test_delete_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="append")

    half_table = pa.Table.from_pydict({"a": range(50), "b": range(50)})

    fragments = lance.dataset(base_dir).get_fragments()

    updated_fragment = fragments[0].delete("a >= 50")
    delete = lance.LanceOperation.Delete(
        [updated_fragment], [fragments[1].fragment_id], "hello"
    )

    dataset = lance.LanceDataset.commit(base_dir, delete, read_version=2)

    tbl = dataset.to_table()
    assert tbl == half_table


def test_count_deleted_rows(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)
    dataset.delete("a >= 50")

    assert dataset.stats.dataset_stats()["num_deleted_rows"] == 50


def test_restore_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)
    lance.write_dataset(table, base_dir, mode="append")

    restore = lance.LanceOperation.Restore(1)
    dataset = lance.LanceDataset.commit(base_dir, restore)

    tbl = dataset.to_table()
    assert tbl == table


def test_merge_insert_with_commit():
    table = pa.table({"id": range(10), "updated": [False] * 10})
    dataset = lance.write_dataset(table, "memory://test")

    updates = pa.Table.from_pylist([{"id": 1, "updated": True}])
    transaction, stats = (
        dataset.merge_insert(on="id")
        .when_matched_update_all()
        .execute_uncommitted(updates)
    )

    assert isinstance(stats, dict)
    assert stats["num_updated_rows"] == 1
    assert stats["num_inserted_rows"] == 0
    assert stats["num_deleted_rows"] == 0

    assert isinstance(transaction, lance.Transaction)
    assert isinstance(transaction.operation, lance.LanceOperation.Update)

    dataset = lance.LanceDataset.commit(dataset, transaction)
    assert dataset.to_table().sort_by("id") == pa.table(
        {"id": range(10), "updated": [False] + [True] + [False] * 8}
    )


def test_merge_with_commit(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"

    lance.write_dataset(table, base_dir)

    fragment = lance.dataset(base_dir).get_fragments()[0]
    merged = fragment.merge_columns(
        lambda _: pa.RecordBatch.from_pydict({"c": range(100)})
    )[0]

    expected = pa.Table.from_pydict({"a": range(100), "b": range(100), "c": range(100)})

    # PyArrow schema is deprecated, but should still work for now.
    with pytest.deprecated_call():
        merge = lance.LanceOperation.Merge([merged], expected.schema)
        dataset = lance.LanceDataset.commit(base_dir, merge, read_version=1)

    tbl = dataset.to_table()

    assert tbl == expected


def test_merge_with_schema_holes(tmp_path: Path):
    # Create table with 3 cols
    table = pa.table({"a": range(10)})
    dataset = lance.write_dataset(table, tmp_path)
    dataset.add_columns({"b": "a + 1"})
    dataset.add_columns({"c": "a + 2"})
    # Delete the middle column to create a hole in the field ids
    dataset.drop_columns(["b"])

    fragment = dataset.get_fragments()[0]
    merged, schema = fragment.merge_columns(
        lambda _: pa.RecordBatch.from_pydict({"d": range(10, 20)})
    )

    merge = lance.LanceOperation.Merge([merged], schema)
    dataset = lance.LanceDataset.commit(tmp_path, merge, read_version=dataset.version)

    dataset.validate()

    tbl = dataset.to_table()
    expected = pa.table(
        {
            "a": range(10),
            "c": range(2, 12),
            "d": range(10, 20),
        }
    )
    assert tbl == expected


def test_merge_columns_rowid(tmp_path: Path):
    base_ds = lance.write_dataset(
        pa.table({"a": range(10), "b": range(10)}), tmp_path / "base"
    )

    merged_frags = []
    schema = None
    for frag in base_ds.get_fragments():

        def copy_row_id(batch: pa.RecordBatch) -> pa.RecordBatch:
            return pa.record_batch(
                {
                    "row_id_copy": batch["_rowid"],
                }
            )

        merged, new_schema = frag.merge_columns(copy_row_id, columns=["_rowid"])
        merged_frags.append(merged)
        schema = new_schema

    merged_ds = lance.LanceDataset.commit(
        base_ds.uri,
        lance.LanceOperation.Merge(merged_frags, schema),
        read_version=base_ds.version,
    )

    assert merged_ds.to_table() == pa.table(
        {
            "a": range(10),
            "b": range(10),
            "row_id_copy": pa.array(range(10), pa.uint64()),
        }
    )


def test_merge_separate_dataset(tmp_path: Path):
    base_ds = lance.write_dataset(
        pa.table({"a": range(10), "b": range(10)}), tmp_path / "base"
    )

    def create_view():
        for batch in base_ds.to_batches(columns=["a"], with_row_id=True):
            new_batch = pa.record_batch(
                {
                    "original_row_id": batch["_rowid"],
                    "double_a": pc.multiply(batch["a"], 2),
                }
            )
            yield new_batch

    schema = list(create_view())[0].schema

    view_ds = lance.write_dataset(create_view(), tmp_path / "view", schema=schema)
    view_ds.create_scalar_index("original_row_id", "BTREE")

    merged_frags = []
    schema = None
    for frag in base_ds.get_fragments():
        # TODO: It would be nice to use frag.merge_columns with a UDF, but if the UDF
        # reads from another dataset, we will get a deadlock.
        table = frag.to_table(with_row_id=True)
        min_id = table["_rowid"][0]
        max_id = table["_rowid"][-1]
        view_table = view_ds.to_table(
            columns=["double_a"],
            filter=f"original_row_id >= {min_id} and original_row_id <= {max_id}",
        )
        if view_table.num_rows != table.num_rows:
            raise ValueError(
                f"Expected {table.num_rows} rows, got {view_table.num_rows}"
            )

        merged, new_schema = frag.merge_columns(view_table)
        merged_frags.append(merged)
        schema = new_schema

    merged_ds = lance.LanceDataset.commit(
        base_ds.uri,
        lance.LanceOperation.Merge(merged_frags, schema),
        read_version=base_ds.version,
    )

    assert merged_ds.to_table() == pa.table(
        {"a": range(10), "b": range(10), "double_a": range(0, 20, 2)}
    )


def test_merge_search(tmp_path: Path):
    left_table = pa.Table.from_pydict({"id": [1, 2, 3], "left": ["a", "b", "c"]})
    right_table = pa.Table.from_pydict({"id": [1, 2, 3], "right": ["A", "B", "C"]})

    left_ds = lance.write_dataset(left_table, tmp_path / "left")

    right_ds = lance.write_dataset(right_table, tmp_path / "right")
    left_ds.merge(right_ds, "id")

    full = left_ds.to_table()
    full_filtered = left_ds.to_table(filter="id < 3")

    partial = left_ds.to_table(columns=["left"])

    assert full.column("left") == partial.column("left")

    partial = left_ds.to_table(columns=["left"], filter="id < 3")

    assert full_filtered.column("left") == partial.column("left")


def test_data_files(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    fragment = lance.fragment.LanceFragment.create(base_dir, table)

    data_files = fragment.files
    assert len(data_files) == 1
    # Filename should be in format: 24-bit binary prefix + 26 hex chars
    with pytest.warns(DeprecationWarning):
        path = data_files[0].path()
    filename_without_ext = os.path.splitext(path)[0]

    # Should be 50 characters: 24 binary + 26 hex
    assert len(filename_without_ext) == 50
    # First 24 should be binary (0s and 1s)
    assert all(c in "01" for c in filename_without_ext[:24])
    # Last 26 should be hex
    assert all(c in "0123456789abcdef" for c in filename_without_ext[24:])

    assert fragment.deletion_file is None


def test_deletion_file(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    base_dir = tmp_path / "test"
    dataset = lance.write_dataset(table, base_dir)
    fragment = dataset.get_fragment(0)

    assert fragment.deletion_file() is None

    new_fragment = fragment.delete("a < 10")

    # Old fragment unchanged
    assert fragment.deletion_file() is None

    # New fragment has deletion file
    assert new_fragment.deletion_file is not None
    assert re.match(
        "_deletions/0-1-[0-9]{1,32}.arrow", new_fragment.deletion_file.path(0)
    )
    operation = lance.LanceOperation.Overwrite(table.schema, [new_fragment])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    assert dataset.count_rows() == 90


def test_commit_fragments_via_scanner(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": range(100)})
    parquet_dir = tmp_path / "parquet.parquet"
    pq.write_to_dataset(table, parquet_dir)

    base_dir = tmp_path / "test"
    scanner = pa.dataset.dataset(parquet_dir).scanner()
    fragment_metadata = lance.fragment.LanceFragment.create(base_dir, scanner)

    # Pickle-able
    pickled = pickle.dumps(fragment_metadata)
    unpickled = pickle.loads(pickled)
    assert fragment_metadata == unpickled
    with pytest.warns(DeprecationWarning):
        path = fragment_metadata.files[0].path()
        assert path == unpickled.files[0].path()

    operation = lance.LanceOperation.Overwrite(table.schema, [fragment_metadata])
    dataset = lance.LanceDataset.commit(base_dir, operation)
    assert dataset.schema == table.schema

    tbl = dataset.to_table()
    assert tbl == table


def test_load_scanner_from_fragments(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100)})
    for _ in range(3):
        lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")
    fragments = list(dataset.get_fragments())
    assert len(fragments) == 3

    scanner = dataset.scanner(fragments=fragments[0:2])
    assert scanner.to_table().num_rows == 2 * 100

    # Accepts an iterator
    scanner = dataset.scanner(fragments=iter(fragments[0:2]), scan_in_order=False)
    assert scanner.to_table().num_rows == 2 * 100


def test_merge_data_legacy(tmp_path: Path):
    tab = pa.table({"a": range(100), "b": range(100)})
    lance.write_dataset(
        tab, tmp_path / "dataset", mode="append", data_storage_version="legacy"
    )

    dataset = lance.dataset(tmp_path / "dataset")

    # rejects partial data for non-nullable types
    new_tab = pa.table({"a": range(40), "c": range(40)})
    with pytest.raises(OSError, match=r"Join produced null values for type: Int64"):
        dataset.merge(new_tab, "a")


def test_merge_data(tmp_path: Path):
    tab = pa.table({"a": range(100)})
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    # accepts partial data for nullable types
    new_tab = pa.table({"a": range(40), "b": range(40)})
    dataset.merge(new_tab, "a")
    assert dataset.version == 2
    assert dataset.to_table() == pa.table(
        {
            "a": range(100),
            "b": pa.array(list(range(40)) + [None] * 60),
        }
    )

    # accepts a full merge
    new_tab = pa.table({"a": range(100), "c": range(100)})
    dataset.merge(new_tab, "a")
    assert dataset.version == 3
    assert dataset.to_table() == pa.table(
        {
            "a": range(100),
            "b": pa.array(list(range(40)) + [None] * 60),
            "c": range(100),
        }
    )

    # accepts a partial for string
    new_tab = pa.table({"a2": range(5), "d": ["a", "b", "c", "d", "e"]})
    dataset.merge(new_tab, left_on="a", right_on="a2")
    assert dataset.version == 4
    expected = pa.table(
        {
            "a": range(100),
            "b": pa.array(list(range(40)) + [None] * 60),
            "c": range(100),
            "d": ["a", "b", "c", "d", "e"] + [None] * 95,
        }
    )
    assert dataset.to_table() == expected
    # Verify we can also load from fresh instance
    dataset = lance.dataset(tmp_path / "dataset")
    assert dataset.to_table() == expected


def test_merge_from_dataset(tmp_path: Path):
    tab1 = pa.table({"a": range(100), "b": range(100)})
    ds1 = lance.write_dataset(tab1, tmp_path / "dataset1", mode="append")

    tab2 = pa.table({"a": range(100), "c": range(100)})
    ds2 = lance.write_dataset(tab2, tmp_path / "dataset2", mode="append")

    ds1.merge(ds2.to_batches(), "a", schema=ds2.schema)
    assert ds1.version == 2
    assert ds1.to_table() == pa.table(
        {
            "a": range(100),
            "b": range(100),
            "c": range(100),
        }
    )


def test_delete_data(tmp_path: Path):
    # We pass schema explicitly since we want b to be non-nullable.
    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.int64(), nullable=False),
        ]
    )
    tab = pa.table({"a": range(100), "b": range(100)}, schema=schema)
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    dataset.delete("a < 10")
    dataset.delete("b in (98, 99)")
    assert dataset.version == 3
    assert dataset.to_table() == pa.table(
        {"a": range(10, 98), "b": range(10, 98)}, schema=schema
    )

    dataset.delete(pa_ds.field("a") < 20)
    assert dataset.version == 4
    assert dataset.to_table() == pa.table(
        {"a": range(20, 98), "b": range(20, 98)}, schema=schema
    )

    # These sorts of filters were previously used as a work-around for not
    # supporting "WHERE true". But now we need to make sure they still work
    # even with presence of expression simplification passes.
    old_version = dataset.version
    dataset.delete("b IS NOT NULL")
    assert dataset.count_rows() == 0

    dataset = lance.dataset(tmp_path / "dataset", version=old_version)
    dataset.restore()
    assert dataset.count_rows() > 0
    dataset.delete("true")
    assert dataset.count_rows() == 0


def check_merge_stats(merge_dict, expected):
    assert (
        merge_dict["num_inserted_rows"],
        merge_dict["num_updated_rows"],
        merge_dict["num_deleted_rows"],
    ) == expected


def test_merge_insert(tmp_path: Path):
    nrows = 1000
    # Create a schema with some metadata to regress an issue where the metadata
    # caused schema comparison problems in merge_insert.
    schema = pa.schema(
        [
            pa.field("a", pa.int64()),
            pa.field("b", pa.int64()),
            pa.field("c", pa.int64()),
        ],
        metadata={"foo": "bar"},
    )
    table = pa.Table.from_pydict(
        {
            "a": range(nrows),
            "b": [1 for _ in range(nrows)],
            "c": [x % 2 for x in range(nrows)],
        },
        schema=schema,
    )
    dataset = lance.write_dataset(
        table, tmp_path / "dataset", mode="create", max_rows_per_file=100
    )
    version = dataset.version

    new_table = pa.Table.from_pydict(
        {
            "a": range(300, 300 + nrows),
            "b": [2 for _ in range(nrows)],
            "c": [0 for _ in range(nrows)],
        },
        schema=schema,
    )

    is_new = pc.field("b") == 2

    merge_dict = (
        dataset.merge_insert("a")
        .when_not_matched_insert_all()
        .retry_timeout(timedelta(seconds=5))
        .conflict_retries(0)
        .execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 300
    check_merge_stats(merge_dict, (300, 0, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = dataset.merge_insert("a").when_matched_update_all().execute(new_table)
    table = dataset.to_table()
    assert table.num_rows == 1000
    assert table.filter(is_new).num_rows == 700
    check_merge_stats(merge_dict, (0, 700, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = (
        dataset.merge_insert("a")
        .when_not_matched_insert_all()
        .when_matched_update_all()
        .execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 1000
    check_merge_stats(merge_dict, (300, 700, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = (
        dataset.merge_insert("a")
        .when_not_matched_insert_all()
        .when_matched_update_all("target.c == source.c")
        .execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 650
    check_merge_stats(merge_dict, (300, 350, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = (
        dataset.merge_insert("a").when_not_matched_by_source_delete().execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 700
    assert table.filter(is_new).num_rows == 0
    check_merge_stats(merge_dict, (0, 0, 300))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    merge_dict = (
        dataset.merge_insert("a")
        .when_not_matched_by_source_delete("a < 100")
        .when_not_matched_insert_all()
        .execute(new_table)
    )

    table = dataset.to_table()
    assert table.num_rows == 1200
    assert table.filter(is_new).num_rows == 300
    check_merge_stats(merge_dict, (300, 0, 100))

    # If the user doesn't specify anything then the merge_insert is
    # a no-op and the operation fails
    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()
    with pytest.raises(ValueError):
        merge_dict = dataset.merge_insert("a").execute(new_table)
        check_merge_stats(merge_dict, (None, None, None))


def test_merge_insert_subcols(tmp_path: Path):
    initial_data = pa.table(
        {
            "a": range(10),
            "b": range(10),
            "c": range(10, 20),
        }
    )
    # Split across two fragments
    dataset = lance.write_dataset(
        initial_data, tmp_path / "dataset", max_rows_per_file=5
    )

    new_values = pa.table(
        {
            "a": range(3, 5),
            "b": range(20, 22),
        }
    )
    (dataset.merge_insert("a").when_matched_update_all().execute(new_values))

    # Partial-schema upserts run through the v2 `FullSchemaMergeInsertExec`
    # path (see Rust issue #6442). We only assert the semantic outcome here
    # — the exact fragment / file layout is an implementation detail of the
    # v1 RewriteColumns optimization that no longer applies. Column `c`
    # (not in the source) must retain its original values for the updated
    # rows.
    expected = pa.table(
        {
            "a": range(10),
            "b": [0, 1, 2, 20, 21, 5, 6, 7, 8, 9],
            "c": range(10, 20),
        }
    )
    assert dataset.to_table().sort_by("a") == expected

    new_values = pa.table(
        {
            "a": range(9, 12),
            "b": range(30, 33),
        }
    )
    (
        dataset.merge_insert("a")
        .when_not_matched_insert_all()
        .when_matched_update_all()
        .execute(new_values)
    )

    assert dataset.count_rows() == 12
    # Newly inserted rows (keys 10, 11) get NULL for column `c` because
    # the source did not provide a value for it and `c` is nullable.
    expected = pa.table(
        {
            "a": range(0, 12),
            "b": [0, 1, 2, 20, 21, 5, 6, 7, 8, 30, 31, 32],
            "c": list(range(10, 20)) + [None] * 2,
        }
    )
    assert dataset.to_table().sort_by("a") == expected


def test_merge_insert_defaults_to_pk_when_on_omitted(tmp_path):
    base_dir = tmp_path / "merge_insert_pk_default"

    schema = pa.schema(
        [
            pa.field(
                "id",
                pa.int32(),
                nullable=False,
                metadata={b"lance-schema:unenforced-primary-key": b"true"},
            ),
            pa.field("value", pa.int32(), nullable=False),
        ]
    )

    base_table = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]}, schema=schema)
    dataset = lance.write_dataset(base_table, base_dir)

    new_table = pa.table({"id": [2, 3, 4], "value": [200, 300, 400]}, schema=schema)

    builder = dataset.merge_insert()
    builder = builder.when_matched_update_all().when_not_matched_insert_all()
    stats = builder.execute(new_table)

    assert stats["num_inserted_rows"] == 1
    assert stats["num_updated_rows"] == 2
    assert stats["num_deleted_rows"] == 0

    result = dataset.to_table().sort_by("id")
    assert result.to_pydict() == {"id": [1, 2, 3, 4], "value": [10, 200, 300, 400]}


def test_merge_insert_raises_without_pk_and_on_omitted(tmp_path):
    base_dir = tmp_path / "merge_insert_no_pk"

    table = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]})
    dataset = lance.write_dataset(table, base_dir)

    with pytest.raises(ValueError) as excinfo:
        dataset.merge_insert()

    msg = str(excinfo.value)
    assert "join keys" in msg or "primary key" in msg


def test_flat_vector_search_with_delete(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "id": [i for i in range(10)],
            "vector": pa.array(
                [[1.0, 1.0] for _ in range(10)], pa.list_(pa.float32(), 2)
            ),
        }
    )
    dataset = lance.write_dataset(table, tmp_path / "dataset", mode="create")
    dataset.delete("id = 0")
    assert (
        dataset.scanner(nearest={"column": "vector", "k": 10, "q": [1.0, 1.0]})
        .to_table()
        .num_rows
        == 9
    )


def test_null_reader_with_deletes(tmp_path: Path):
    full_schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("other", pa.int64()),
        ]
    )
    ds = lance.write_dataset([], tmp_path, schema=full_schema, mode="create")
    ds.insert(pa.table({"id": [1, 2, 3, 4, 5]}))
    ds.delete("id in (1, 2)")
    ds.to_table()


def test_merge_insert_conditional_upsert_example(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "txNumber": [1, 1, 2, 2, 3],
            "vector": pa.array([[1.0, 1.0] for _ in range(5)], pa.list_(pa.float32())),
        }
    )
    dataset = lance.write_dataset(table, tmp_path / "dataset", mode="create")

    new_table = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "txNumber": [1, 2, 1, 2, 5],
            "vector": pa.array([[2.0, 2.0] for _ in range(5)], pa.list_(pa.float32())),
        }
    )

    merge_dict = (
        dataset.merge_insert("id")
        .when_matched_update_all("target.txNumber < source.txNumber")
        .execute(new_table)
    )

    table = dataset.to_table()

    expected = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "txNumber": [1, 2, 2, 2, 5],
            "vector": pa.array(
                [[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
                pa.list_(pa.float32()),
            ),
        }
    )

    assert table.sort_by("id") == expected
    check_merge_stats(merge_dict, (0, 2, 0))

    # No matches

    new_table = pa.Table.from_pydict(
        {
            "id": [1, 2, 3, 4, 5],
            "txNumber": [0, 0, 0, 0, 0],
            "vector": pa.array([[2.0, 2.0] for _ in range(5)], pa.list_(pa.float32())),
        }
    )

    merge_dict = (
        dataset.merge_insert("id")
        .when_matched_update_all("target.txNumber < source.txNumber")
        .execute(new_table)
    )

    check_merge_stats(merge_dict, (0, 0, 0))


def test_merge_insert_source_is_dataset(tmp_path: Path):
    nrows = 1000
    table = pa.Table.from_pydict({"a": range(nrows), "b": [1 for _ in range(nrows)]})
    dataset = lance.write_dataset(
        table, tmp_path / "dataset", mode="create", max_rows_per_file=100
    )
    version = dataset.version

    new_table = pa.Table.from_pydict(
        {
            "a": range(300, 300 + nrows),
            "b": [2 for _ in range(nrows)],
        }
    )
    new_dataset = lance.write_dataset(
        new_table, tmp_path / "dataset2", mode="create", max_rows_per_file=80
    )

    is_new = pc.field("b") == 2

    merge_dict = (
        dataset.merge_insert("a").when_not_matched_insert_all().execute(new_dataset)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 300
    check_merge_stats(merge_dict, (300, 0, 0))

    dataset = lance.dataset(tmp_path / "dataset", version=version)
    dataset.restore()

    reader = new_dataset.to_batches()

    merge_dict = (
        dataset.merge_insert("a")
        .when_not_matched_insert_all()
        .execute(reader, schema=new_dataset.schema)
    )
    table = dataset.to_table()
    assert table.num_rows == 1300
    assert table.filter(is_new).num_rows == 300
    check_merge_stats(merge_dict, (300, 0, 0))


def test_merge_insert_multiple_keys(tmp_path: Path):
    nrows = 1000
    # a - [0, 1, 2, ..., 999]
    # b - [1, 1, 1, ..., 1]
    # c - [0, 1, 0, ..., 1]
    table = pa.Table.from_pydict(
        {
            "a": range(nrows),
            "b": [1 for _ in range(nrows)],
            "c": [i % 2 for i in range(nrows)],
        }
    )
    dataset = lance.write_dataset(
        table, tmp_path / "dataset", mode="create", max_rows_per_file=100
    )

    # a - [300, 301, 302, ..., 1299]
    # b - [2, 2, 2, ..., 2]
    # c - [0, 0, 0, ..., 0]
    new_table = pa.Table.from_pydict(
        {
            "a": range(300, 300 + nrows),
            "b": [2 for _ in range(nrows)],
            "c": [0 for _ in range(nrows)],
        }
    )

    is_new = pc.field("b") == 2

    merge_dict = (
        dataset.merge_insert(["a", "c"]).when_matched_update_all().execute(new_table)
    )
    table = dataset.to_table()
    assert table.num_rows == 1000
    assert table.filter(is_new).num_rows == 350
    check_merge_stats(merge_dict, (0, 350, 0))


def test_merge_insert_vector_column(tmp_path: Path):
    table = pa.Table.from_pydict(
        {
            "vec": pa.array([[1, 2, 3], [4, 5, 6]], pa.list_(pa.float32(), 3)),
            "key": [1, 2],
        }
    )

    new_table = pa.Table.from_pydict(
        {
            "vec": pa.array([[7, 8, 9], [10, 11, 12]], pa.list_(pa.float32(), 3)),
            "key": [2, 3],
        }
    )

    dataset = lance.write_dataset(
        table, tmp_path / "dataset", mode="create", max_rows_per_file=100
    )

    merge_dict = (
        dataset.merge_insert(["key"])
        .when_not_matched_insert_all()
        .when_matched_update_all()
        .execute(new_table)
    )
    expected = pa.Table.from_pydict(
        {
            "vec": pa.array(
                [[1, 2, 3], [7, 8, 9], [10, 11, 12]], pa.list_(pa.float32(), 3)
            ),
            "key": [1, 2, 3],
        }
    )

    assert dataset.to_table().sort_by("key") == expected
    check_merge_stats(merge_dict, (1, 1, 0))


def test_merge_insert_when_matched_fail(tmp_path: Path):
    data = pa.table({"id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
    ds = lance.write_dataset(data, tmp_path / "dataset")
    version = ds.version

    # No matching rows should succeed
    new_data = pa.table({"id": [6, 7, 8], "val": [60, 70, 80]})
    result = (
        ds.merge_insert("id")
        .when_matched_fail()
        .when_not_matched_insert_all()
        .execute(new_data)
    )
    assert result["num_inserted_rows"] == 3
    assert result["num_updated_rows"] == 0
    assert result["num_deleted_rows"] == 0

    # Matching rows should fail
    ds = lance.dataset(tmp_path / "dataset", version=version)
    ds.restore()
    new_data = pa.table({"id": [1, 2, 9], "val": [100, 200, 900]})
    with pytest.raises(Exception):
        ds.merge_insert("id").when_matched_fail().when_not_matched_insert_all().execute(
            new_data
        )

    # Test with execute_uncommitted
    # This should raise an exception because there are matching rows
    ds = lance.dataset(tmp_path / "dataset", version=version)
    ds.restore()
    with pytest.raises(Exception):
        transaction, _ = (
            ds.merge_insert("id")
            .when_matched_fail()
            .when_not_matched_insert_all()
            .execute_uncommitted(new_data)
        )

    # Verify that the data remains unchanged after failed operation
    ds = lance.dataset(tmp_path / "dataset", version=version)
    ds.restore()
    unchanged_ds = lance.dataset(tmp_path / "dataset")
    unchanged_data = unchanged_ds.to_table().sort_by("id")
    expected = pa.table({"id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
    assert unchanged_data == expected


def test_merge_insert_when_matched_delete(tmp_path: Path):
    """Test when_matched_delete functionality for merge insert."""
    # Create initial dataset with ids 1-6
    data = pa.table({"id": [1, 2, 3, 4, 5, 6], "val": [10, 20, 30, 40, 50, 60]})
    ds = lance.write_dataset(data, tmp_path / "dataset")
    version = ds.version

    # Test 1: Basic when_matched_delete - delete matched rows only
    # Source has ids 4, 5, 6 (match) and 7, 8, 9 (no match)
    # Only matched rows should be deleted, unmatched rows are ignored
    delete_keys = pa.table({"id": [4, 5, 6, 7, 8, 9], "val": [0, 0, 0, 0, 0, 0]})
    result = ds.merge_insert("id").when_matched_delete().execute(delete_keys)

    assert result["num_deleted_rows"] == 3
    assert result["num_inserted_rows"] == 0
    assert result["num_updated_rows"] == 0

    # Verify only ids 1, 2, 3 remain
    remaining = ds.to_table().sort_by("id")
    expected = pa.table({"id": [1, 2, 3], "val": [10, 20, 30]})
    assert remaining == expected

    # Test 2: when_matched_delete with ID-only source
    # Source contains only the key column
    ds = lance.dataset(tmp_path / "dataset", version=version)
    ds.restore()

    id_only_source = pa.table({"id": [2, 4, 6]})  # Delete even ids
    result = ds.merge_insert("id").when_matched_delete().execute(id_only_source)

    assert result["num_deleted_rows"] == 3
    assert result["num_inserted_rows"] == 0
    assert result["num_updated_rows"] == 0

    # Verify only odd ids remain
    remaining = ds.to_table().sort_by("id")
    expected = pa.table({"id": [1, 3, 5], "val": [10, 30, 50]})
    assert remaining == expected

    # Test 3: when_matched_delete combined with when_not_matched_insert_all
    # Delete existing rows that match, insert new rows that don't match
    ds = lance.dataset(tmp_path / "dataset", version=version)
    ds.restore()

    new_data = pa.table(
        {"id": [4, 5, 6, 7, 8, 9], "val": [400, 500, 600, 700, 800, 900]}
    )
    result = (
        ds.merge_insert("id")
        .when_matched_delete()
        .when_not_matched_insert_all()
        .execute(new_data)
    )

    # Should delete 3 (ids 4, 5, 6) and insert 3 (ids 7, 8, 9)
    assert result["num_deleted_rows"] == 3
    assert result["num_inserted_rows"] == 3
    assert result["num_updated_rows"] == 0

    # Verify: ids 1, 2, 3 (original), 7, 8, 9 (new inserts)
    remaining = ds.to_table().sort_by("id")
    expected = pa.table({"id": [1, 2, 3, 7, 8, 9], "val": [10, 20, 30, 700, 800, 900]})
    assert remaining == expected

    # Test 4: when_matched_delete with no matches (should be a no-op delete)
    ds = lance.dataset(tmp_path / "dataset", version=version)
    ds.restore()

    non_matching = pa.table({"id": [100, 200, 300], "val": [0, 0, 0]})
    result = ds.merge_insert("id").when_matched_delete().execute(non_matching)

    assert result["num_deleted_rows"] == 0
    assert result["num_inserted_rows"] == 0
    assert result["num_updated_rows"] == 0

    # Data should be unchanged
    remaining = ds.to_table().sort_by("id")
    expected = pa.table({"id": [1, 2, 3, 4, 5, 6], "val": [10, 20, 30, 40, 50, 60]})
    assert remaining == expected


def test_merge_insert_large():
    # Doing subcolumns update with merge insert triggers this error.
    # Data needs to be large enough to make DataFusion create multiple batches
    # when outputting join results.
    # https://github.com/lancedb/lance/issues/3406
    # This test is in Python because for whatever reason, the error doesn't
    # reproduce in the equivalent Rust test.
    dims = 32
    nrows = 20_000
    data = pa.table({"id": range(nrows), "num": [str(i) for i in range(nrows)]})

    ds = lance.write_dataset(data, "memory://")

    ds.add_columns({"vector": f"arrow_cast(NULL, 'FixedSizeList({dims}, Float32)')"})

    batch_size = 10_000
    other_columns = pa.table(
        {
            "id": range(batch_size),
            "vector": pa.FixedSizeListArray.from_arrays(
                pc.random(batch_size * dims).cast(pa.float32()), dims
            ),
        }
    )

    (
        ds.merge_insert(on="id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(other_columns)
    )


def test_merge_insert_large_rows():
    # Verify that merge_insert succeeds when individual batches exceed the
    # DataFusion memory pool size (100 MiB by default).  DataFusion's sort
    # cannot spill a single batch that is larger than the pool, so the
    # HardCapBatchSizeExec node must rechunk before the sort.
    #
    # The sort path (update_fragments) is only used for partial/subcolumn
    # updates, so the source must have a subset of the target columns.
    #
    # One batch of ~400 MiB which exceeds the 300 MiB memory pool.
    # Without HardCapBatchSizeExec rechunking it before the sort, the sort
    # would fail because it cannot handle a single batch larger than the pool.
    os.environ["LANCE_MEM_POOL_SIZE"] = str(1024 * 1024 * 1024)
    rows_per_batch = 2000
    num_batches = 1
    nrows = rows_per_batch * num_batches
    # ~200 KiB per row → ~400 MiB per batch
    blob_size = 200 * 1024

    # Create dataset with id, blob, and an extra column so the source is partial.
    full_schema = pa.schema(
        [("id", pa.int64()), ("blob", pa.large_binary()), ("extra", pa.int32())]
    )

    def make_full_batches():
        for i in range(num_batches):
            start = i * rows_per_batch
            ids = pa.array(range(start, start + rows_per_batch), type=pa.int64())
            blobs = pa.array(
                [b"x" * blob_size] * rows_per_batch, type=pa.large_binary()
            )
            extras = pa.array(range(start, start + rows_per_batch), type=pa.int32())
            yield pa.record_batch([ids, blobs, extras], schema=full_schema)

    reader = pa.RecordBatchReader.from_batches(full_schema, make_full_batches())
    ds = lance.write_dataset(reader, "memory://")

    # Partial update: only update the blob column (omit extra).
    # This triggers the update_fragments path which uses the sort.
    partial_schema = pa.schema([("id", pa.int64()), ("blob", pa.large_binary())])

    def make_partial_batches():
        for i in range(num_batches):
            start = i * rows_per_batch
            ids = pa.array(range(start, start + rows_per_batch), type=pa.int64())
            blobs = pa.array(
                [b"y" * blob_size] * rows_per_batch, type=pa.large_binary()
            )
            yield pa.record_batch([ids, blobs], schema=partial_schema)

    new_reader = pa.RecordBatchReader.from_batches(
        partial_schema, make_partial_batches()
    )

    stats = ds.merge_insert(on="id").when_matched_update_all().execute(new_reader)

    assert stats["num_updated_rows"] == nrows
    assert stats["num_inserted_rows"] == 0

    result = ds.to_table()
    assert result.num_rows == nrows
    # Spot-check that the blobs were updated and extra column preserved.
    assert result.column("blob")[0].as_py() == b"y" * blob_size
    assert result.column("extra")[0].as_py() == 0

    del os.environ["LANCE_MEM_POOL_SIZE"]


def test_merge_insert_empty_index():
    # Reported in https://github.com/lancedb/lancedb/issues/2285
    empty_table = pa.table({"id": pa.array([], type=pa.float64())})
    empty_ds = lance.write_dataset(empty_table, "memory://")

    empty_ds.create_scalar_index("id", "BTREE")

    df = pa.table({"id": [1.0, 2.0, 3.0]})

    empty_ds.merge_insert("id").when_not_matched_insert_all().execute(df)


def test_merge_insert_explain_analyze_plan():
    """Test that explain_plan and analyze_plan work on merge insert operations."""
    # Create a simple test dataset
    data = pa.table(
        [pa.array([1, 2, 3, 4]), pa.array(["a", "b", "c", "d"])], names=["id", "value"]
    )

    dataset = lance.write_dataset(data, "memory://test-merge-explain")

    # Create merge insert builder
    builder = (
        dataset.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
    )

    # Test explain_plan with default schema (None)
    plan = builder.explain_plan()  # Uses dataset schema by default
    assert isinstance(plan, str)
    assert len(plan) > 0
    assert "MergeInsert" in plan

    # Test with explicit schema (must match dataset schema exactly)
    source_schema = pa.schema(
        [pa.field("id", pa.int64()), pa.field("value", pa.string())]
    )
    plan_explicit = builder.explain_plan(source_schema, verbose=False)
    assert isinstance(plan_explicit, str)
    assert len(plan_explicit) > 0
    assert "MergeInsert" in plan_explicit

    # Test verbose mode with default schema
    plan_verbose = builder.explain_plan(verbose=True)
    assert isinstance(plan_verbose, str)
    assert len(plan_verbose) > 0
    assert "MergeInsert" in plan_verbose

    # Create test source data for analyze_plan (must match dataset schema exactly)
    source_data = pa.table(
        [
            pa.array([1, 5]),  # 1 matches existing, 5 is new
            pa.array(["updated_a", "e"]),
        ],
        names=["id", "value"],
    )

    # Test analyze_plan
    analysis = builder.analyze_plan(source_data)
    assert isinstance(analysis, str)
    assert len(analysis) > 0
    assert "MergeInsert" in analysis
    assert "metrics" in analysis
    # Check for new write metrics
    assert "bytes_written" in analysis
    assert "num_files_written" in analysis


def test_merge_insert_use_index():
    """Test that use_index parameter controls whether indices are used."""
    data = pa.table({"id": range(100), "value": [i * 10 for i in range(100)]})
    dataset = lance.write_dataset(data, "memory://test-merge-use-index")
    dataset.create_scalar_index("id", "BTREE")

    source_data = pa.table({"id": [1, 2, 101], "value": [999, 999, 999]})

    # Test 1: use_index=False should allow explain_plan to succeed
    builder_no_index = (
        dataset.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .use_index(False)  # Force not using index
    )

    # With use_index=False, explain_plan should succeed even with an index present
    plan = builder_no_index.explain_plan()
    assert isinstance(plan, str)
    assert "MergeInsert" in plan
    # Should use hash join, not index scan
    assert "HashJoinExec" in plan or "Join" in plan

    # Test 2: use_index=True (default) should fail explain_plan with index present
    builder_with_index = (
        dataset.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .use_index(True)  # Explicitly set to use index (though it's the default)
    )

    # With use_index=True and an index present, explain_plan should fail
    with pytest.raises(Exception) as exc_info:
        builder_with_index.explain_plan()
    assert "does not support explain_plan" in str(exc_info.value)

    # Test 3: Verify actual execution works with no index
    result = builder_no_index.execute(source_data)
    assert result["num_updated_rows"] == 2
    assert result["num_inserted_rows"] == 1

    # Verify the data was updated correctly
    updated_table = dataset.to_table()
    values = updated_table.column("value").to_pylist()
    updated_count = sum(1 for v in values if v == 999)
    assert updated_count == 3


def test_add_null_columns(tmp_path: Path):
    data = pa.table({"id": [1, 2, 4]})
    ds = lance.write_dataset(data, tmp_path)
    fragments = ds.get_fragments()
    assert len(fragments) == 1
    assert len(fragments[0].data_files()) == 1

    ds.add_columns(pa.field("f1", pa.float32()))
    fragments = ds.get_fragments()
    assert len(fragments) == 1
    assert len(fragments[0].data_files()) == 1
    assert ds.schema == pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("f1", pa.float32()),
        ]
    )

    ds.add_columns(
        [pa.field("v2", pa.list_(pa.float32(), 32)), pa.field("v3", pa.int32())]
    )
    fragments = ds.get_fragments()
    assert len(fragments) == 1
    assert len(fragments[0].data_files()) == 1
    assert ds.schema == pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("f1", pa.float32()),
            pa.field("v2", pa.list_(pa.float32(), 32)),
            pa.field("v3", pa.int32()),
        ]
    )

    ds.add_columns(
        pa.schema([pa.field("s6", pa.struct([("a", pa.int32()), ("b", pa.bool_())]))])
    )
    fragments = ds.get_fragments()
    assert len(fragments) == 1
    assert len(fragments[0].data_files()) == 1
    assert ds.schema == pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("f1", pa.float32()),
            pa.field("v2", pa.list_(pa.float32(), 32)),
            pa.field("v3", pa.int32()),
            pa.field("s6", pa.struct([("a", pa.int32()), ("b", pa.bool_())])),
        ]
    )


def test_merge_insert_permissive_nullability(tmp_path):
    """
    Reported in https://github.com/lancedb/lance/issues/4518
    Tests that merge_insert works when the source schema is nullable
    but the target is not, as long as no nulls are present.
    """
    target_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("value", pa.int64(), nullable=False),
        ]
    )
    initial_data = pa.table(
        {"id": [1, 2, 3], "value": [10, 20, 30]}, schema=target_schema
    )

    uri = tmp_path / "dataset"
    ds = lance.write_dataset(initial_data, uri)

    source_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=True),
            pa.field("value", pa.int64(), nullable=True),
        ]
    )

    new_data = pa.table(
        {"id": [2, 4, 5], "value": [200, 400, 500]}, schema=source_schema
    )

    # Execute merge_insert, which should now succeed.
    stats = (
        ds.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(new_data)
    )

    # Verify the results.
    assert stats["num_updated_rows"] == 1
    assert stats["num_inserted_rows"] == 2

    expected_data = pa.table(
        {"id": [1, 2, 3, 4, 5], "value": [10, 200, 30, 400, 500]}, schema=target_schema
    )

    result_table = ds.to_table()
    assert result_table.sort_by("id").equals(expected_data.sort_by("id"))


def test_add_null_columns_with_conflict_names(tmp_path: Path):
    data = pa.table({"id": [1, 2, 4]})
    ds = lance.write_dataset(data, tmp_path)
    fragments = ds.get_fragments()
    assert len(fragments) == 1
    assert len(fragments[0].data_files()) == 1

    with pytest.raises(
        Exception, match=".*Type conflicts between id\\(Int64\\) and id\\(Float32\\).*"
    ):
        ds.add_columns(pa.field("id", pa.float32()))

    with pytest.raises(
        Exception, match=".*Type conflicts between id\\(Int64\\) and id\\(Float32\\).*"
    ):
        ds.add_columns([pa.field("id", pa.float32()), pa.field("good", pa.int32())])


def check_update_stats(update_dict, expected):
    assert (update_dict["num_rows_updated"],) == expected


def test_update_dataset(tmp_path: Path):
    nrows = 100
    vecs = pa.FixedSizeListArray.from_arrays(
        pa.array(range(2 * nrows), type=pa.float32()), 2
    )
    tab = pa.table({"a": range(nrows), "b": range(nrows), "vec": vecs})
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    update_dict = dataset.update(dict(b="b + 1"))
    expected = pa.table({"a": range(100), "b": range(1, 101)})
    assert dataset.to_table(columns=["a", "b"]) == expected
    check_update_stats(update_dict, (100,))

    update_dict = dataset.update(dict(a="a * 2"), where="a < 50")
    expected = pa.table(
        {
            "a": [x * 2 if x < 50 else x for x in range(100)],
            "b": range(1, 101),
        }
    )
    assert dataset.to_table(columns=["a", "b"]).sort_by("b") == expected
    check_update_stats(update_dict, (50,))

    update_dict = dataset.update(dict(vec="[42.0, 43.0]"))
    expected = pa.table(
        {
            "b": range(1, 101),
            "vec": pa.array(
                [[42.0, 43.0] for _ in range(100)], pa.list_(pa.float32(), 2)
            ),
        }
    )
    assert dataset.to_table(columns=["b", "vec"]).sort_by("b") == expected
    check_update_stats(update_dict, (100,))


def test_update_dataset_scanner_after_stable_row_id_update(tmp_path: Path):
    dataset = lance.write_dataset(
        pa.table(
            {
                "name_1": pa.array(["1", "4", "7"]),
                "name_2": pa.array(["2", "5", "8"]),
                "name_3": pa.array(["3", "6", "9"]),
            }
        ),
        tmp_path / "dataset",
        enable_stable_row_ids=True,
    )

    update_dict = dataset.update(updates=dict(name_3="'xxxx'"), where="name_1 = '7'")
    check_update_stats(update_dict, (1,))

    expected = pa.table(
        {
            "name_1": pa.array(["1", "4", "7"]),
            "name_2": pa.array(["2", "5", "8"]),
            "name_3": pa.array(["3", "6", "xxxx"]),
        }
    )
    actual = dataset.to_table().sort_by("name_1")
    assert actual == expected

    scanner_table = dataset.scanner(limit=10).to_table().sort_by("name_1")
    assert scanner_table == expected
    assert scanner_table == actual


def test_update_by_rowid(tmp_path: Path):
    nrows = 30
    table = pa.table(
        {
            "id": pa.array(range(nrows), pa.int64()),
            "name": pa.array(["foo"] * nrows, pa.string()),
        }
    )
    dataset = lance.write_dataset(
        table,
        tmp_path / "dataset",
        max_rows_per_file=10,
        enable_stable_row_ids=True,
    )

    orig = dataset.to_table(columns=["id"], with_row_id=True)
    target_idx = 5
    target_row_id = orig.column("_rowid")[target_idx].as_py()
    target_id = orig.column("id")[target_idx].as_py()

    update_dict = dataset.update(
        updates=dict(name="'updated'"),
        where=f"_rowid = {target_row_id}",
    )
    check_update_stats(update_dict, (1,))

    actual = dataset.to_table().sort_by("id")
    for row in actual.to_pylist():
        if row["id"] == target_id:
            assert row["name"] == "updated"
        else:
            assert row["name"] == "foo"


def test_update_by_rowid_in_list(tmp_path: Path):
    nrows = 30
    table = pa.table(
        {
            "id": pa.array(range(nrows), pa.int64()),
            "name": pa.array(["foo"] * nrows, pa.string()),
        }
    )
    dataset = lance.write_dataset(
        table,
        tmp_path / "dataset",
        max_rows_per_file=10,
        enable_stable_row_ids=True,
    )

    orig = dataset.to_table(columns=["id"], with_row_id=True)
    target_indices = [3, 7, 15]
    target_row_ids = [orig.column("_rowid")[i].as_py() for i in target_indices]
    target_ids = {orig.column("id")[i].as_py() for i in target_indices}

    in_list = ", ".join(str(rid) for rid in target_row_ids)
    update_dict = dataset.update(
        updates=dict(name="'updated'"),
        where=f"_rowid IN ({in_list})",
    )
    check_update_stats(update_dict, (3,))

    actual = dataset.to_table().sort_by("id")
    for row in actual.to_pylist():
        if row["id"] in target_ids:
            assert row["name"] == "updated"
        else:
            assert row["name"] == "foo"


def test_update_dataset_all_types(tmp_path: Path):
    table = pa.table(
        {
            "int32": pa.array([1], pa.int32()),
            "int64": pa.array([1], pa.int64()),
            "uint32": pa.array([1], pa.uint32()),
            "string": pa.array(["foo"], pa.string()),
            "large_string": pa.array(["foo"], pa.large_string()),
            "float32": pa.array([1.0], pa.float32()),
            "float64": pa.array([1.0], pa.float64()),
            "bool": pa.array([True], pa.bool_()),
            "date32": pa.array([date(2021, 1, 1)], pa.date32()),
            "timestamp_ns": pa.array([datetime(2021, 1, 1)], pa.timestamp("ns")),
            "timestamp_ms": pa.array([datetime(2021, 1, 1)], pa.timestamp("ms")),
            "vec_f32": pa.array([[1.0, 2.0]], pa.list_(pa.float32(), 2)),
            "vec_f64": pa.array([[1.0, 2.0]], pa.list_(pa.float64(), 2)),
        }
    )

    dataset = lance.write_dataset(table, tmp_path)

    # One update with all matching types
    update_dict = dataset.update(
        dict(
            int32="2",
            int64="2",
            uint32="2",
            string="'bar'",
            large_string="'bar'",
            float32="2.0",
            float64="2.0",
            bool="false",
            date32="DATE '2021-01-02'",
            timestamp_ns='TIMESTAMP "2021-01-02 00:00:00"',
            timestamp_ms='TIMESTAMP "2021-01-02 00:00:00"',
            vec_f32="[3.0, 4.0]",
            vec_f64="[3.0, 4.0]",
        )
    )
    expected = pa.table(
        {
            "int32": pa.array([2], pa.int32()),
            "int64": pa.array([2], pa.int64()),
            "uint32": pa.array([2], pa.uint32()),
            "string": pa.array(["bar"], pa.string()),
            "large_string": pa.array(["bar"], pa.large_string()),
            "float32": pa.array([2.0], pa.float32()),
            "float64": pa.array([2.0], pa.float64()),
            "bool": pa.array([False], pa.bool_()),
            "date32": pa.array([date(2021, 1, 2)], pa.date32()),
            "timestamp_ns": pa.array([datetime(2021, 1, 2)], pa.timestamp("ns")),
            "timestamp_ms": pa.array([datetime(2021, 1, 2)], pa.timestamp("ms")),
            "vec_f32": pa.array([[3.0, 4.0]], pa.list_(pa.float32(), 2)),
            "vec_f64": pa.array([[3.0, 4.0]], pa.list_(pa.float64(), 2)),
        }
    )
    assert dataset.to_table() == expected
    check_update_stats(update_dict, (1,))


def test_update_with_binary_field(tmp_path: Path):
    # Create a lance dataset with binary fields
    table = pa.Table.from_pydict(
        {
            "a": [f"str-{i}" for i in range(100)],
            "b": [b"bin-{i}" for i in range(100)],
            "c": list(range(100)),
        }
    )
    dataset = lance.write_dataset(table, tmp_path)

    # Update binary field
    update_dict = dataset.update({"b": "X'616263'"}, where="c < 2")

    ds = lance.dataset(tmp_path)
    assert ds.scanner(filter="c < 2").to_table().column(
        "b"
    ).combine_chunks() == pa.array([b"abc", b"abc"])
    check_update_stats(update_dict, (2,))


def test_create_update_empty_dataset(tmp_path: Path, provide_pandas: bool):
    base_dir = tmp_path / "dataset"

    fields = [
        ("a", pa.string()),
        ("b", pa.int64()),
        ("c", pa.float64()),
    ]
    df = pd.DataFrame(columns=[name for name, _ in fields])
    tab = pa.Table.from_pandas(df, schema=pa.schema(fields))
    dataset = lance.write_dataset(tab, base_dir)

    assert dataset.count_rows() == 0
    expected_schema = pa.schema(
        [
            pa.field("a", pa.string()),
            pa.field("b", pa.int64()),
            pa.field("c", pa.float64()),
        ]
    )
    assert dataset.schema == expected_schema
    assert dataset.to_table() == pa.table(
        {"a": [], "b": [], "c": []}, schema=expected_schema
    )

    tab2 = pa.table({"a": ["foo"], "b": [1], "c": [2.0]})
    dataset = lance.write_dataset(tab2, base_dir, mode="append")

    assert dataset.count_rows() == 1
    assert dataset.to_table() == pa.table(
        {"a": ["foo"], "b": [1], "c": [2.0]}, schema=expected_schema
    )


def test_update_with_retry_parameters(tmp_path: Path):
    """Test that update accepts conflict_retries and retry_timeout parameters"""
    nrows = 10
    tab = pa.table({"a": range(nrows), "b": range(nrows)})
    lance.write_dataset(tab, tmp_path / "dataset", mode="append")

    dataset = lance.dataset(tmp_path / "dataset")

    # Test with custom conflict_retries and retry_timeout
    update_dict = dataset.update(
        updates=dict(b="b + 1"), conflict_retries=5, retry_timeout=timedelta(seconds=60)
    )

    expected = pa.table({"a": range(10), "b": range(1, 11)})
    assert dataset.to_table(columns=["a", "b"]) == expected
    check_update_stats(update_dict, (10,))


def test_scan_with_batch_size(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(10000), "b": range(10000)})
    dataset = lance.write_dataset(df, base_dir)

    batches = dataset.scanner(batch_size=16, scan_in_order=True).to_batches()

    for idx, batch in enumerate(batches):
        assert batch.num_rows == 16
        df = batch.to_pandas()
        assert df["a"].iloc[0] == idx * 16

    os.environ["LANCE_DEFAULT_BATCH_SIZE"] = "12"
    batches = dataset.scanner(scan_in_order=True).to_batches()
    for batch in batches:
        # The last batch in each file has 4 rows
        assert batch.num_rows == 12 or batch.num_rows == 4

    del os.environ["LANCE_DEFAULT_BATCH_SIZE"]
    batches = dataset.scanner(scan_in_order=True).to_batches()
    for batch in batches:
        assert batch.num_rows != 12


def test_dictionaries(tmp_path: Path):
    data = pa.table(
        {
            "id": pa.array([1, 2, 3]),
            "dict": pa.array(
                ["foo", "bar", "baz"], pa.dictionary(pa.int32(), pa.string())
            ),
        }
    )
    ds = lance.write_dataset(data, tmp_path)
    assert ds.schema == pa.schema(
        {"id": pa.int64(), "dict": pa.dictionary(pa.int32(), pa.string())}
    )
    assert ds.to_table() == data

    # Can insert data with new values
    new_data = pa.table(
        {
            "id": [4, 5, 6],
            "dict": pa.array(
                ["qux", "quux", "corge"], pa.dictionary(pa.int32(), pa.string())
            ),
        }
    )
    ds.insert(new_data)
    table = ds.to_table().combine_chunks()
    assert table == pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "dict": pa.array(
                ["foo", "bar", "baz", "qux", "quux", "corge"],
                pa.dictionary(pa.int32(), pa.string()),
            ),
        }
    )

    dict_arr = table.column("dict").chunk(0)
    assert dict_arr.type == pa.dictionary(pa.int32(), pa.string())
    assert dict_arr.to_pylist() == ["foo", "bar", "baz", "qux", "quux", "corge"]

    assert dict_arr.dictionary.to_pylist() == [
        "foo",
        "bar",
        "baz",
        "qux",
        "quux",
        "corge",
    ]

    # Can merge insert data that has even more values
    new_data = pa.table(
        {
            "id": [1, 7],
            "dict": pa.array(
                ["grault", "garply"], pa.dictionary(pa.int32(), pa.string())
            ),
        }
    )
    ds.merge_insert(
        "id"
    ).when_matched_update_all().when_not_matched_insert_all().execute(new_data)
    table = ds.to_table().combine_chunks().sort_by("id")
    assert table.column("id").to_pylist() == [1, 2, 3, 4, 5, 6, 7]
    assert table.column("dict").to_pylist() == [
        "grault",
        "bar",
        "baz",
        "qux",
        "quux",
        "corge",
        "garply",
    ]


@pytest.mark.slow
def test_io_buffer_size(tmp_path: Path):
    # These cases regress deadlock issues that happen when the
    # batch size was very large (in bytes) so that batches are
    # bigger than the I/O buffer size
    #
    # The test is slow (it needs to generate enough data to cover a variety
    # of cases) but it is essential to run if any changes are made to the
    # 2.0 scheduling priority / decoding strategy
    #
    # In this test we create 4 pages of data, 2 for each column.  We
    # then set the I/O buffer size to 5000 bytes so that we only read
    # 1 page at a time and set the batch size to 2M rows so that we
    # read in all 4 pages for a single batch.
    #
    # The scheduler will schedule in this order: C0P0, C1P0, C0P1, C1P1
    #
    # The deadlock would happen because the decoder was decoding in this
    # order: C0P0, C0P1, C1P0, C1P1.
    #
    # The decoder will wait for C0P1 but the scheduler will only have scheduled
    # C1P0 and so deadlock would happen.
    #
    # The fix was to change the decoder to decode in the same order as the
    # scheduler.
    base_dir = tmp_path / "dataset"

    def datagen():
        for i in range(2):
            yield pa.record_batch(
                [
                    pa.array(range(1024 * 1024), pa.uint64()),
                    pa.array(range(1024 * 1024), pa.uint64()),
                ],
                names=["a", "b"],
            )

    schema = pa.schema({"a": pa.uint64(), "b": pa.uint64()})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        max_rows_per_file=2 * 1024 * 1024,
        mode="overwrite",
    )

    dataset.scanner(batch_size=2 * 1024 * 1024, io_buffer_size=5000).to_table()

    # This test is similar but the first column is a list column.  The I/O to grab
    # the list items will span multiple requests and it is important that all of
    # those requests share the priority of the parent list

    def datagen():
        for i in range(2):
            yield pa.record_batch(
                [
                    pa.array([[0]] * 1024 * 1024, pa.list_(pa.uint64())),
                    pa.array(range(1024 * 1024), pa.uint64()),
                ],
                names=["a", "b"],
            )

    schema = pa.schema({"a": pa.list_(pa.uint64()), "b": pa.uint64()})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        max_rows_per_file=2 * 1024 * 1024,
        mode="overwrite",
    )

    dataset.scanner(batch_size=2 * 1024 * 1024, io_buffer_size=5000).to_table()

    # Another scenario.  Each list item is a page in itself and we have two list
    # columns

    def datagen():
        for i in range(16):
            yield pa.record_batch(
                [
                    pa.array([[0] * 5 * 1024 * 1024], pa.list_(pa.uint64())),
                    pa.array([[0] * 5 * 1024 * 1024], pa.list_(pa.uint64())),
                ],
                names=["a", "b"],
            )

    schema = pa.schema({"a": pa.list_(pa.uint64()), "b": pa.list_(pa.uint64())})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        mode="overwrite",
    )

    dataset.scanner(batch_size=16, io_buffer_size=5000).to_table()

    # Same scenario as above except now it's a list<list<int>>

    def datagen():
        for i in range(16):
            yield pa.record_batch(
                [
                    pa.array(
                        [[[0] * 5 * 1024 * 1024]], pa.list_(pa.list_(pa.uint64()))
                    ),
                    pa.array(
                        [[[0] * 5 * 1024 * 1024]], pa.list_(pa.list_(pa.uint64()))
                    ),
                ],
                names=["a", "b"],
            )

    schema = pa.schema(
        {"a": pa.list_(pa.list_(pa.uint64())), "b": pa.list_(pa.list_(pa.uint64()))}
    )

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        mode="overwrite",
    )

    dataset.scanner(batch_size=16, io_buffer_size=5000).to_table()

    # Next we consider the case where the column is a struct column and we want to
    # make sure we don't decode too deeply into the struct child

    def datagen():
        for i in range(2):
            yield pa.record_batch(
                [
                    pa.array(
                        [{"foo": i} for i in range(1024 * 1024)],
                        pa.struct([pa.field("foo", pa.uint64())]),
                    ),
                    pa.array(range(1024 * 1024), pa.uint64()),
                ],
                names=["a", "b"],
            )

    schema = pa.schema(
        {"a": pa.struct([pa.field("foo", pa.uint64())]), "b": pa.uint64()}
    )

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        max_rows_per_file=2 * 1024 * 1024,
        mode="overwrite",
    )

    dataset.scanner(batch_size=2 * 1024 * 1024, io_buffer_size=5000).to_table()

    # This reproduces another issue we saw where there are a bunch of empty lists and
    # lance was calculating the page priority incorrectly.

    fsl_type = pa.list_(pa.uint64(), 32 * 1024 * 1024)
    list_type = pa.list_(fsl_type)

    def datagen():
        # Each item is 32
        values = pa.array(range(32 * 1024 * 1024 * 7), pa.uint64())
        fsls = pa.FixedSizeListArray.from_arrays(values, 32 * 1024 * 1024)
        # 3 items, 5 empties, 2 items
        offsets = pa.array([0, 1, 2, 3, 4, 4, 5, 6, 7], pa.int32())
        lists = pa.ListArray.from_arrays(offsets, fsls)

        values2 = pa.array(range(32 * 1024 * 1024 * 8), pa.uint64())
        fsls2 = pa.FixedSizeListArray.from_arrays(values2, 32 * 1024 * 1024)
        offsets2 = pa.array([0, 1, 2, 3, 4, 5, 6, 7, 8], pa.int32())
        lists2 = pa.ListArray.from_arrays(offsets2, fsls2)

        yield pa.record_batch(
            [
                lists,
                lists2,
            ],
            names=["a", "b"],
        )

    schema = pa.schema({"a": list_type, "b": list_type})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        mode="overwrite",
    )

    dataset.scanner(batch_size=10, io_buffer_size=5000).to_table()

    # We need to make sure that different data files within a fragment are
    # given the same priority.  This is because we scan both files at the same
    # time in order.

    def datagen():
        for i in range(2):
            buf = pa.allocate_buffer(8 * 1024 * 1024)
            arr = pa.FixedSizeBinaryArray.from_buffers(
                pa.binary(1024 * 1024), 8, [None, buf]
            )
            yield pa.record_batch(
                [
                    arr,
                    pa.array(range(8), pa.uint64()),
                ],
                names=["a", "b"],
            )

    schema = pa.schema({"a": pa.binary(1024 * 1024), "b": pa.uint64()})

    dataset = lance.write_dataset(
        datagen(),
        base_dir,
        schema=schema,
        data_storage_version="stable",
        max_rows_per_file=2 * 1024 * 1024,
        mode="overwrite",
    )

    dataset.add_columns({"c": "b"})

    dataset.scanner(batch_size=1, io_buffer_size=5000).to_table()


def test_scan_no_columns(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(100)})
    dataset = lance.write_dataset(df, base_dir)

    # columns=[] can be used to get just the row ids
    batches = dataset.scanner(columns=[], with_row_id=True).to_batches()

    expected_schema = pa.schema([pa.field("_rowid", pa.uint64())])
    for batch in batches:
        assert batch.schema == expected_schema

    # Can specify nothing at all to get empty batches (with correct counts)
    batches = list(dataset.scanner(columns=[], batch_size=10).to_batches())
    assert len(batches) == 10
    for batch in batches:
        assert batch.schema.names == []
        assert batch.num_rows == 10

    # also test with deleted data to make sure deleted ids not included
    dataset.delete("a = 5")
    num_rows = 0
    for batch in dataset.scanner(columns=[], with_row_id=True).to_batches():
        num_rows += batch.num_rows

    assert num_rows == 99


def test_scan_prefilter(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    vecs = pa.array(
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]], type=pa.list_(pa.float32(), 2)
    )
    df = pa.Table.from_pydict(
        {
            "index": [1, 2, 3, 4, 5, 6],
            "type": ["a", "a", "a", "b", "b", "b"],
            "vecs": vecs,
        }
    )
    dataset = lance.write_dataset(df, base_dir)
    query = pa.array([1, 1], pa.float32())
    args = {
        "columns": ["index"],
        "filter": "type = 'b'",
        "nearest": {"column": "vecs", "q": pa.array(query), "k": 2, "use_index": False},
    }

    # With post-filter no results are returned because all
    # the closest results don't match the filter
    assert dataset.scanner(**args).to_table().num_rows == 0

    args["prefilter"] = True
    table = dataset.scanner(**args).to_table()

    expected = pa.Table.from_pydict({"index": [4, 5]})

    assert table.column("index") == expected.column("index")

    table = dataset.to_table(**args)
    assert table.column("index") == expected.column("index")

    table = pa.Table.from_batches(dataset.to_batches(**args))
    assert table.column("index") == expected.column("index")


def test_scan_count_rows(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(42), "b": range(42)})
    dataset = lance.write_dataset(df, base_dir)

    assert dataset.scanner(columns=[], with_row_id=True).count_rows() == 42
    assert dataset.count_rows(filter="a < 10") == 10
    assert dataset.count_rows(filter=pa_ds.field("a") < 20) == 20


def test_with_row_offset(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(100)})
    dataset = lance.write_dataset(df, base_dir, max_rows_per_file=25)

    tbl = dataset.scanner(columns=["_rowoffset"]).to_table()
    tbl = tbl.combine_chunks()
    assert tbl == pa.table({"_rowoffset": pa.array(range(100), pa.uint64())})


def test_system_columns(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(100)})
    dataset = lance.write_dataset(df, base_dir, max_rows_per_file=25)

    # Projection order is respected when system columns are specified via projection
    tbl = dataset.scanner(columns=["_rowid", "_rowaddr", "_rowoffset"]).to_table()
    assert tbl.schema.names == ["_rowid", "_rowaddr", "_rowoffset"]
    tbl = dataset.scanner(columns=["_rowoffset", "_rowid", "_rowaddr"]).to_table()
    assert tbl.schema.names == ["_rowoffset", "_rowid", "_rowaddr"]

    # If specified using with_row_id or with_row_address, the system columns are
    # added to the end of the schema
    tbl = dataset.scanner(with_row_id=True).to_table()
    assert tbl.schema.names == ["a", "_rowid"]
    tbl = dataset.scanner(with_row_address=True).to_table()
    assert tbl.schema.names == ["a", "_rowaddr"]
    tbl = dataset.scanner(with_row_address=True, with_row_id=True).to_table()
    assert tbl.schema.names == ["a", "_rowid", "_rowaddr"]


def test_scoring_columns(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pa.table(
        {
            "vec": pa.array([[i, i] for i in range(100)], pa.list_(pa.float32(), 2)),
        }
    )
    dataset = lance.write_dataset(df, base_dir)

    nearest = {
        "column": "vec",
        "q": pa.array([1, 1], pa.float32()),
        "k": 10,
        "use_index": False,
    }

    # Common behavior, no projection means we add distance column at the end
    print(dataset.scanner(nearest=nearest).explain_plan())
    tbl = dataset.scanner(nearest=nearest).to_table()
    assert tbl.schema.names == ["vec", "_distance"]

    # If _rowid or _rowaddr specified via legacy means, they go after distance
    tbl = dataset.scanner(nearest=nearest, with_row_id=True).to_table()
    assert tbl.schema.names == ["vec", "_distance", "_rowid"]

    tbl = dataset.scanner(nearest=nearest, with_row_address=True).to_table()
    assert tbl.schema.names == ["vec", "_distance", "_rowaddr"]

    tbl = dataset.scanner(
        nearest=nearest, with_row_address=True, with_row_id=True
    ).to_table()
    assert tbl.schema.names == ["vec", "_distance", "_rowid", "_rowaddr"]

    # If _rowid or _rowaddr are specified via projection, they stay in the specified
    # order; _distance must be explicitly included
    tbl = dataset.scanner(
        columns=["_rowid", "vec", "_distance"], nearest=nearest
    ).to_table()
    assert tbl.schema.names == ["_rowid", "vec", "_distance"]
    tbl = dataset.scanner(
        columns=["vec", "_rowaddr", "_distance"],
        nearest=nearest,
    ).to_table()
    assert tbl.schema.names == ["vec", "_rowaddr", "_distance"]

    # With an explicit projection, _distance is NOT added automatically
    tbl = dataset.scanner(columns=["vec"], nearest=nearest).to_table()
    assert tbl.schema.names == ["vec"]

    # disable_scoring_autoprojection=False raises an error
    with pytest.raises(Exception, match="no longer supported"):
        dataset.scanner(
            columns=["vec"], nearest=nearest, disable_scoring_autoprojection=False
        ).to_table()


def test_scanner_schemas(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(50), "s": [f"s-{i}" for i in range(50)]})

    dataset = lance.write_dataset(df, base_dir)

    scanner = dataset.scanner(columns=["a"])
    assert scanner.dataset_schema == dataset.schema
    assert scanner.projected_schema == pa.schema([pa.field("a", pa.int64())])


def test_scan_deleted_rows(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    ds = lance.write_dataset(df, base_dir, max_rows_per_file=25)
    ds.create_scalar_index("b", "BTREE")
    ds.delete("a < 30")

    assert ds.count_rows() == 70

    assert ds.scanner(with_row_id=True).to_table().num_rows == 70
    with_deleted = ds.scanner(with_row_id=True, include_deleted_rows=True).to_table()

    assert with_deleted.num_rows == 75

    assert with_deleted.slice(0, 5) == pa.table(
        {
            "a": range(25, 30),
            "b": range(25, 30),
            "_rowid": pa.array([None] * 5, pa.uint64()),
        }
    )

    assert (
        ds.scanner(with_row_id=True, include_deleted_rows=True, filter="a < 32")
        .to_table()
        .num_rows
        == 7
    )

    assert (
        ds.scanner(include_deleted_rows=True, with_row_id=True, filter="b < 30")
        .to_table()
        .num_rows
        == 5
    )

    with pytest.raises(ValueError, match="with_row_id is false"):
        ds.scanner(include_deleted_rows=True, filter="a < 30").to_table()


def test_custom_commit_lock(tmp_path: Path):
    called_lock = False
    called_release = False

    @contextlib.contextmanager
    def commit_lock(version: int):
        nonlocal called_lock
        nonlocal called_release
        called_lock = True
        assert version == 1
        yield
        called_release = True

    lance.write_dataset(
        pa.table({"a": range(100)}), tmp_path / "test1", commit_lock=commit_lock
    )
    assert called_lock
    assert called_release

    @contextlib.contextmanager
    def commit_lock(_version: int):
        try:
            yield
        finally:
            raise Exception("hello world!")

    with pytest.raises(Exception, match="hello world!"):
        lance.write_dataset(
            pa.table({"a": range(100)}), tmp_path / "test2", commit_lock=commit_lock
        )

    @contextlib.contextmanager
    def commit_lock(_version: int):
        raise CommitConflictError()

    with pytest.raises(Exception, match="CommitConflictError"):
        lance.write_dataset(
            pa.table({"a": range(100)}), tmp_path / "test3", commit_lock=commit_lock
        )

    dataset = lance.dataset(tmp_path / "test1", commit_lock=commit_lock)
    with pytest.raises(Exception, match="CommitConflictError"):
        dataset.delete("a < 10")


def test_dataset_restore(tmp_path: Path):
    data = pa.table({"a": range(100)})
    dataset = lance.write_dataset(data, tmp_path)
    assert dataset.version == 1
    assert dataset.count_rows() == 100

    dataset.delete("a >= 50")
    assert dataset.count_rows() == 50
    assert dataset.version == 2

    dataset = lance.dataset(tmp_path, version=1)
    assert dataset.version == 1

    dataset.restore()
    assert dataset.version == 3
    assert dataset.count_rows() == 100


def test_mixed_mode_overwrite(tmp_path: Path):
    data = pa.table({"a": range(100)})
    dataset = lance.write_dataset(data, tmp_path, data_storage_version="legacy")

    assert dataset.data_storage_version == "0.1"

    dataset = lance.write_dataset(data, tmp_path, mode="overwrite")

    assert dataset.data_storage_version == "0.1"


def test_roundtrip_reader(tmp_path: Path):
    # Can roundtrip a reader
    data = pa.table({"a": range(100)})
    dataset = lance.write_dataset(data, tmp_path / "test1")
    reader = dataset.to_batches()
    lance.write_dataset(reader, tmp_path / "test2", schema=dataset.schema)

    # Can roundtrip a whole dataset
    lance.write_dataset(dataset, tmp_path / "test3")


def test_metadata(tmp_path: Path):
    data = pa.table({"a": range(100)})
    data = data.replace_schema_metadata({"foo": pickle.dumps("foo")})
    with pytest.raises(ValueError):
        lance.write_dataset(data, tmp_path)

    data = data.replace_schema_metadata({"foo": base64.b64encode(pickle.dumps("foo"))})
    lance.write_dataset(data, tmp_path)


def test_default_scan_options(tmp_path: Path):
    data = pa.table({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(data, tmp_path)
    assert dataset.schema.names == ["a", "b"]

    dataset = lance.dataset(tmp_path)
    assert dataset.schema.names == ["a", "b"]

    dataset = lance.dataset(
        tmp_path,
        default_scan_options={
            "with_row_id": True,
        },
    )
    assert dataset.schema.names == ["a", "b", "_rowid"]

    dataset = lance.dataset(tmp_path, default_scan_options={"with_row_address": True})
    assert dataset.schema.names == ["a", "b", "_rowaddr"]


def test_scan_with_row_ids(tmp_path: Path):
    """Test expose physical row ids in the scanner."""
    data = pa.table({"a": range(1000)})
    ds = lance.write_dataset(data, tmp_path, max_rows_per_file=250)

    tbl = ds.scanner(filter="a % 10 == 0 AND a < 500", with_row_id=True).to_table()
    assert "_rowid" in tbl.column_names
    row_ids = tbl["_rowid"].to_pylist()
    assert row_ids == list(range(0, 250, 10)) + list(range(2**32, 2**32 + 250, 10))

    tbl2 = ds._take_rows(row_ids)
    assert tbl2["a"] == tbl["a"]


@pytest.mark.cuda
def test_random_dataset_recall_accelerated(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    from lance.dependencies import torch

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a",
        "IVF_PQ",
        num_partitions=2,
        num_sub_vectors=32,
        accelerator=torch.device("cpu"),
    )
    validate_vector_index(dataset, "a", pass_threshold=0.5)


@pytest.mark.cuda
def test_random_dataset_recall_accelerated_one_pass(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    from lance.dependencies import torch

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a",
        "IVF_PQ",
        num_partitions=2,
        num_sub_vectors=32,
        accelerator=torch.device("cpu"),
        one_pass_ivfpq=True,
    )
    validate_vector_index(dataset, "a", pass_threshold=0.5)


@pytest.mark.cuda
def test_count_index_rows_accelerated(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    # assert we return None for index name that doesn't exist
    index_name = "a_idx"
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_unindexed_rows"]
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_indexed_rows"]

    from lance.dependencies import torch

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a",
        "IVF_PQ",
        name=index_name,
        num_partitions=2,
        num_sub_vectors=1,
        accelerator=torch.device("cpu"),
    )
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 0
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512

    # append some data
    new_table = pa.Table.from_pydict(
        {"a": [[float(i) for i in range(32)] for _ in range(512)]}, schema=schema
    )
    dataset = lance.write_dataset(new_table, base_dir, mode="append")

    # assert rows added since index was created are uncounted
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 512
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512


@pytest.mark.cuda
def test_count_index_rows_accelerated_one_pass(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    # assert we return None for index name that doesn't exist
    index_name = "a_idx"
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_unindexed_rows"]
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_indexed_rows"]

    from lance.dependencies import torch

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a",
        "IVF_PQ",
        name=index_name,
        num_partitions=2,
        num_sub_vectors=1,
        accelerator=torch.device("cpu"),
        one_pass_ivfpq=True,
    )
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 0
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512

    # append some data
    new_table = pa.Table.from_pydict(
        {"a": [[float(i) for i in range(32)] for _ in range(512)]}, schema=schema
    )
    dataset = lance.write_dataset(new_table, base_dir, mode="append")

    # assert rows added since index was created are uncounted
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 512
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512


def test_count_index_rows(tmp_path: Path):
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(512 * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    # assert we return None for index name that doesn't exist
    index_name = "a_idx"
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_unindexed_rows"]
    with pytest.raises(KeyError):
        dataset.stats.index_stats(index_name)["num_indexed_rows"]

    # create index and assert no rows are uncounted
    dataset.create_index(
        "a", "IVF_PQ", name=index_name, num_partitions=2, num_sub_vectors=1
    )
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 0
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512

    # append some data
    new_table = pa.Table.from_pydict(
        {"a": [[float(i) for i in range(32)] for _ in range(512)]}, schema=schema
    )
    dataset = lance.write_dataset(new_table, base_dir, mode="append")

    # assert rows added since index was created are uncounted
    assert dataset.stats.index_stats(index_name)["num_unindexed_rows"] == 512
    assert dataset.stats.index_stats(index_name)["num_indexed_rows"] == 512


def test_dataset_progress(tmp_path: Path):
    data = pa.table({"a": range(10)})
    progress = ProgressForTest()
    ds = lance.write_dataset(data, tmp_path, max_rows_per_file=5, progress=progress)
    assert len(ds.get_fragments()) == 2
    assert progress.begin_called == 2
    assert progress.complete_called == 2


def test_tensor_type(tmp_path: Path):
    arr = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
    storage = pa.array(arr, pa.list_(pa.float32(), 4))
    tensor_type = pa.fixed_shape_tensor(pa.float32(), [4])
    ext_arr = pa.ExtensionArray.from_storage(tensor_type, storage)
    data = pa.table({"tensor": ext_arr})

    ds = lance.write_dataset(data, tmp_path)

    query_arr = [[10, 20, 30, 40]]
    storage = pa.array(query_arr, pa.list_(pa.float32(), 4))
    ext_arr = pa.ExtensionArray.from_storage(tensor_type, storage)
    ext_scalar = ext_arr[0]

    results = ds.to_table(
        nearest={
            "column": "tensor",
            "k": 1,
            "q": ext_scalar,
        }
    )
    assert results.num_rows == 1


def test_sharded_iterator_fragments(tmp_path: Path):
    arr = pa.array(range(1000))
    tbl = pa.table({"a": arr})
    # Write about 10 files
    lance.write_dataset(tbl, tmp_path, max_rows_per_group=20, max_rows_per_file=100)

    shard_datast = ShardedBatchIterator(tmp_path, 1, 2, columns=["a"])
    batches = pa.concat_arrays([b["a"] for b in shard_datast])
    assert batches == pa.array(
        list(range(100, 200))
        + list(range(300, 400))
        + list(range(500, 600))
        + list(range(700, 800))
        + list(range(900, 1000))
    )


def test_sharded_iterator_batches(tmp_path: Path):
    arr = pa.array(range(1000))
    tbl = pa.table({"a": arr})
    # Write about 10 files
    ds = lance.write_dataset(tbl, tmp_path, max_rows_per_file=100)

    RANK = 1
    WORLD_SIZE = 2
    BATCH_SIZE = 15
    shard_datast = ShardedBatchIterator(
        ds,
        RANK,
        WORLD_SIZE,
        columns=["a"],
        batch_size=BATCH_SIZE,
        granularity="batch",
    )
    batches = pa.concat_arrays([b["a"] for b in shard_datast])
    assert batches == pa.array(
        [
            j
            for i in range(RANK * BATCH_SIZE, 1000, WORLD_SIZE * BATCH_SIZE)
            for j in range(i, i + BATCH_SIZE)
        ]
    )


def test_sharded_iterator_non_full_batch(tmp_path: Path):
    arr = pa.array(range(1186))
    tbl = pa.table({"a": arr})

    ds = lance.write_dataset(tbl, tmp_path)
    shard_datast = ShardedBatchIterator(
        ds,
        1,
        2,
        columns=["a"],
        batch_size=100,
        granularity="batch",
    )
    batches = pa.concat_arrays([b["a"] for b in shard_datast])

    # Can read partial batches
    assert len(set(range(1100, 1186)) - set(batches.to_pylist())) == 0


def test_dynamic_projection(tmp_path: Path):
    table1 = pa.Table.from_pylist([{"a": 1, "b": 2}, {"a": 10, "b": 20}])
    base_dir = tmp_path / "test"
    lance.write_dataset(table1, base_dir)

    dataset = lance.dataset(base_dir)
    table2 = dataset.to_table(
        columns={
            "bool": "a > 5",
        }
    )

    expected = pa.Table.from_pylist([{"bool": False}, {"bool": True}])
    assert expected == table2


def test_migrate_manifest(tmp_path: Path):
    from lance.lance import manifest_needs_migration

    table = pa.table({"x": [1, 2, 3]})
    ds = lance.write_dataset(table, tmp_path)
    # We shouldn't need a migration for a brand new dataset.
    assert not manifest_needs_migration(ds)


def test_legacy_dataset(tmp_path: Path):
    table = pa.table({"a": range(100), "b": range(100)})
    dataset = lance.write_dataset(table, tmp_path, data_storage_version="stable")
    batches = list(dataset.to_batches())
    assert len(batches) == 1
    assert pa.Table.from_batches(batches) == table
    fragment = list(dataset.get_fragments())[0]
    assert "major_version: 2" in format_fragment(fragment.metadata, dataset)
    assert dataset.data_storage_version == stable_version()

    # Append will write v2 if dataset was originally created with v2
    dataset = lance.write_dataset(table, tmp_path, mode="append")

    assert len(dataset.get_fragments()) == 2

    fragment = list(dataset.get_fragments())[1]
    assert "major_version: 2" in format_fragment(fragment.metadata, dataset)

    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="legacy", mode="overwrite"
    )
    assert dataset.data_storage_version == "0.1"

    fragment = list(dataset.get_fragments())[0]
    assert "major_version: 2" not in format_fragment(fragment.metadata, dataset)

    # Append will write v1 if dataset was originally created with v1
    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="stable", mode="append"
    )

    fragment = list(dataset.get_fragments())[1]
    assert "major_version: 2" not in format_fragment(fragment.metadata, dataset)

    # Writing an empty table with v2 will put dataset in "v2 mode"
    dataset = lance.write_dataset(
        [],
        tmp_path,
        schema=table.schema,
        data_storage_version="stable",
        mode="overwrite",
    )

    assert len(dataset.get_fragments()) == 0

    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="legacy", mode="append"
    )

    fragment = list(dataset.get_fragments())[0]
    assert "major_version: 2" in format_fragment(fragment.metadata, dataset)

    # Writing an empty table with v1 will put dataset in "v1 mode"
    dataset = lance.write_dataset(
        [],
        tmp_path,
        schema=table.schema,
        data_storage_version="legacy",
        mode="overwrite",
    )

    assert len(dataset.get_fragments()) == 0

    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="stable", mode="append"
    )

    fragment = list(dataset.get_fragments())[0]
    assert "major_version: 2" not in format_fragment(fragment.metadata, dataset)


def test_late_materialization_param(tmp_path: Path):
    table = pa.table(
        {
            "filter": np.arange(4),
            "values": pa.array([b"abcd", b"efgh", b"ijkl", b"mnop"]),
        }
    )
    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="stable", max_rows_per_file=10000
    )
    filt = "filter % 2 == 0"

    assert "(values)" in dataset.scanner(
        filter=filt, late_materialization=None
    ).explain_plan(True)
    assert ", values" in dataset.scanner(
        filter=filt, late_materialization=False
    ).explain_plan(True)
    assert "(values)" in dataset.scanner(
        filter=filt, late_materialization=True
    ).explain_plan(True)
    assert "(values)" in dataset.scanner(
        filter=filt, late_materialization=["values"]
    ).explain_plan(True)
    assert ", values" in dataset.scanner(
        filter=filt, late_materialization=["filter"]
    ).explain_plan(True)

    # These tests just make sure we can pass in the parameter.  There's no great
    # way to know if late materialization happened or not.  That will have to be
    # for benchmarks
    expected = dataset.to_table(filter=filt)
    assert dataset.to_table(filter=filt, late_materialization=None) == expected
    assert dataset.to_table(filter=filt, late_materialization=False) == expected
    assert dataset.to_table(filter=filt, late_materialization=True) == expected
    assert dataset.to_table(filter=filt, late_materialization=["values"]) == expected

    expected = list(dataset.to_batches(filter=filt))
    assert list(dataset.to_batches(filter=filt, late_materialization=None)) == expected
    assert list(dataset.to_batches(filter=filt, late_materialization=False)) == expected
    assert list(dataset.to_batches(filter=filt, late_materialization=True)) == expected
    assert (
        list(dataset.to_batches(filter=filt, late_materialization=["values"]))
        == expected
    )


def test_late_materialization_batch_size(tmp_path: Path):
    table = pa.table({"filter": np.arange(32 * 32), "values": np.arange(32 * 32)})
    dataset = lance.write_dataset(
        table, tmp_path, data_storage_version="stable", max_rows_per_file=10000
    )
    for batch in dataset.to_batches(
        columns=["values"],
        filter="filter % 2 == 0",
        batch_size=32,
        late_materialization=True,
    ):
        assert batch.num_rows == 32


def test_batch_size_bytes_chopping_fallback(tmp_path: Path):
    # Write rows with large strings (~10KiB each). The file reader's byte-size
    # estimation is based on on-disk size which, for strings, can differ
    # significantly from in-memory size.  The rechunk fallback should still
    # split oversized batches so no single batch exceeds the target by too much.
    num_rows = 200
    row_text = "x" * 10 * 1024  # 10 KiB per row
    table = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "text": pa.array([row_text] * num_rows, type=pa.large_string()),
        }
    )
    dataset = lance.write_dataset(table, tmp_path, data_storage_version="stable")

    # Target ~50 KiB per batch with ~10 KiB per row → expect ~5 rows per batch.
    # Without the rechunk fallback all 200 rows would arrive in a single batch.
    target_bytes = 50 * 1024
    batches = list(
        dataset.to_batches(
            filter="id >= 0",
            batch_size_bytes=target_bytes,
            late_materialization=False,
        )
    )

    total_rows = sum(b.num_rows for b in batches)
    assert total_rows == num_rows

    # We should get many batches, not one giant batch.
    assert len(batches) > 1, f"expected many batches, got {len(batches)}"
    # Each batch should have a small number of rows (target ~5, allow up to 10).
    for batch in batches:
        assert batch.num_rows <= 10, (
            f"batch has {batch.num_rows} rows, expected at most ~10 "
            f"for a {target_bytes} byte target with ~10KiB rows"
        )


def test_use_scalar_index(tmp_path: Path):
    table = pa.table({"filter": range(100)})
    dataset = lance.write_dataset(table, tmp_path)
    dataset.create_scalar_index("filter", "BTREE")

    assert "ScalarIndexQuery: query=[filter = 10]@filter_idx" in dataset.scanner(
        filter="filter = 10"
    ).explain_plan(True)

    assert "ScalarIndexQuery: query=[filter = 10]@filter_idx" in dataset.scanner(
        filter="filter = 10", use_scalar_index=True
    ).explain_plan(True)

    assert "ScalarIndexQuery" not in dataset.scanner(
        filter="filter = 10", use_scalar_index=False
    ).explain_plan(True)


def test_fast_search_scalar_index_skips_unindexed_fragments(tmp_path: Path):
    table = pa.table({"filter": range(100)})
    dataset = lance.write_dataset(table, tmp_path, max_rows_per_file=100)
    dataset.create_scalar_index("filter", "BTREE")
    dataset = lance.write_dataset(
        pa.table({"filter": range(100, 110)}), tmp_path, mode="append"
    )

    normal = dataset.to_table(filter="filter >= 95")
    fast = dataset.to_table(filter="filter >= 95", fast_search=True)

    assert normal.num_rows == 15
    assert fast.num_rows == 5
    assert sorted(fast.column("filter").to_pylist()) == list(range(95, 100))


EXPECTED_DEFAULT_STORAGE_VERSION = stable_version()
EXPECTED_MAJOR_VERSION = int(stable_version().split(".")[0])
EXPECTED_MINOR_VERSION = int(stable_version().split(".")[1])


def test_stats(tmp_path: Path):
    table = pa.table({"x": [1, 2, 3, 4], "y": ["foo", "bar", "baz", "qux"]})
    dataset = lance.write_dataset(table, tmp_path)
    stats = dataset.stats.dataset_stats()

    assert stats["num_deleted_rows"] == 0
    assert stats["num_fragments"] == 1
    assert stats["num_small_files"] == 1

    data_stats = dataset.stats.data_stats()

    assert data_stats.fields[0].id == 0
    assert data_stats.fields[0].bytes_on_disk > 0
    assert data_stats.fields[1].id == 1
    assert data_stats.fields[1].bytes_on_disk > 0

    dataset.add_columns({"z": "y"})

    dataset.insert(pa.table({"x": [5], "z": ["quux"]}))

    data_stats = dataset.stats.data_stats()

    assert data_stats.fields[0].id == 0
    assert data_stats.fields[0].bytes_on_disk > 0
    assert data_stats.fields[1].id == 1
    assert data_stats.fields[1].bytes_on_disk > 0
    assert data_stats.fields[2].id == 2
    assert data_stats.fields[2].bytes_on_disk > 0


def test_default_storage_version(tmp_path: Path):
    table = pa.table({"x": [0]})
    dataset = lance.write_dataset(table, tmp_path)
    assert dataset.data_storage_version == EXPECTED_DEFAULT_STORAGE_VERSION

    frag = lance.LanceFragment.create(dataset.uri, table)
    sample_file = frag.files[0]
    assert sample_file.file_major_version == EXPECTED_MAJOR_VERSION
    assert sample_file.file_minor_version == EXPECTED_MINOR_VERSION

    from lance.fragment import write_fragments

    frags = write_fragments(table, dataset.uri)
    frag = frags[0]
    sample_file = frag.files[0]
    assert sample_file.file_major_version == EXPECTED_MAJOR_VERSION
    assert sample_file.file_minor_version == EXPECTED_MINOR_VERSION


def test_no_detached_v1(tmp_path: Path):
    table = pa.table({"x": [0]})
    dataset = lance.write_dataset(table, tmp_path, enable_v2_manifest_paths=False)

    # Make a detached append
    table = pa.table({"x": [1]})
    frag = lance.LanceFragment.create(dataset.uri, table)
    op = lance.LanceOperation.Append([frag])
    with pytest.raises(OSError, match="v1 manifest paths"):
        dataset.commit(dataset.uri, op, read_version=dataset.version, detached=True)


def test_detached_commits(tmp_path: Path):
    table = pa.table({"x": [0]})
    dataset = lance.write_dataset(table, tmp_path, enable_v2_manifest_paths=True)

    # Make a detached append
    table = pa.table({"x": [1]})
    frag = lance.LanceFragment.create(dataset.uri, table)
    op = lance.LanceOperation.Append([frag])
    detached = dataset.commit(
        dataset.uri, op, read_version=dataset.version, detached=True
    )
    assert (detached.version & 0x8000000000000000) != 0

    assert detached.to_table() == pa.table({"x": [0, 1]})
    # Detached commit should not show up in the dataset
    dataset = lance.dataset(tmp_path)
    assert dataset.to_table() == pa.table({"x": [0]})

    # We can make more commits to dataset and they don't affect attached
    table = pa.table({"x": [2]})
    dataset = lance.write_dataset(table, tmp_path, mode="append")
    assert dataset.to_table() == pa.table({"x": [0, 2]})

    # We can check out the detached commit
    detached = dataset.checkout_version(detached.version)
    assert detached.to_table() == pa.table({"x": [0, 1]})

    # Detached commit can use detached commit as read version
    table = pa.table({"x": [3]})
    frag = lance.LanceFragment.create(detached.uri, table)
    op = lance.LanceOperation.Append([frag])

    detached2 = dataset.commit(
        dataset.uri, op, read_version=detached.version, detached=True
    )

    assert detached2.to_table() == pa.table({"x": [0, 1, 3]})


def test_dataset_drop(tmp_path: Path):
    table = pa.table({"x": [0]})
    lance.write_dataset(table, tmp_path)
    assert Path(tmp_path).exists()
    lance.LanceDataset.drop(tmp_path)
    assert not Path(tmp_path).exists()
    lance.LanceDataset.drop(tmp_path, ignore_not_found=True)
    with pytest.raises(OSError):
        lance.LanceDataset.drop(tmp_path, ignore_not_found=False)
    with pytest.raises(OSError):
        lance.LanceDataset.drop(tmp_path)


def test_dataset_schema(tmp_path: Path):
    table = pa.table({"x": [0]})
    ds = lance.write_dataset(table, str(tmp_path))  # noqa: F841
    ds._default_scan_options = {"with_row_id": True}
    assert ds.schema == ds.to_table().schema


def test_data_replacement(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100), "b": [str(i) for i in range(100)]})
    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    # Note: by using a string column here we are making sure the new file has a
    # different size than the old file (this regresses a bug).
    table = pa.Table.from_pydict(
        {"a": range(100, 200), "b": [str(i) for i in range(100, 200)]}
    )
    fragment = lance.fragment.LanceFragment.create(base_dir, table)
    data_file = fragment.files[0]
    data_replacement = lance.LanceOperation.DataReplacement(
        [lance.LanceOperation.DataReplacementGroup(0, data_file)]
    )
    dataset = lance.LanceDataset.commit(dataset, data_replacement, read_version=1)

    tbl = dataset.to_table()

    expected = pa.Table.from_pydict(
        {
            "a": list(range(100, 200)),
            "b": list([str(i) for i in range(100, 200)]),
        }
    )
    assert tbl == expected


def test_schema_project_drop_column(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100, 200), "b": range(300, 400)})
    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    schema = pa.Table.from_pydict({"a": range(1)}).schema
    lance_schema = LanceSchema.from_pyarrow(schema)

    project = lance.LanceOperation.Project(lance_schema)
    dataset = lance.LanceDataset.commit(dataset, project, read_version=1)

    tbl = dataset.to_table()

    expected = pa.Table.from_pydict(
        {
            "a": list(range(100, 200)),
        }
    )
    assert tbl == expected


def test_schema_project_rename_column(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100, 200), "b": range(300, 400)})
    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    schema = pa.Table.from_pydict({"c": range(1), "d": range(1)}).schema
    lance_schema = LanceSchema.from_pyarrow(schema)

    project = lance.LanceOperation.Project(lance_schema)
    dataset = lance.LanceDataset.commit(dataset, project, read_version=1)

    tbl = dataset.to_table()

    expected = pa.Table.from_pydict(
        {
            "c": list(range(100, 200)),
            "d": list(range(300, 400)),
        }
    )
    assert tbl == expected


def test_schema_project_swap_column(tmp_path: Path):
    table = pa.Table.from_pydict({"a": range(100, 200), "b": range(300, 400)})
    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    schema = pa.Table.from_pydict({"b": range(1), "a": range(1)}).schema
    lance_schema = LanceSchema.from_pyarrow(schema)

    project = lance.LanceOperation.Project(lance_schema)
    dataset = lance.LanceDataset.commit(dataset, project, read_version=1)

    tbl = dataset.to_table()

    expected = pa.Table.from_pydict(
        {
            "b": list(range(100, 200)),
            "a": list(range(300, 400)),
        }
    )
    assert tbl == expected


def test_empty_structs(tmp_path):
    schema = pa.schema([pa.field("id", pa.int32()), pa.field("empties", pa.struct([]))])
    table = pa.table({"id": [0, 1, 2], "empties": [{}] * 3}, schema=schema)
    ds = lance.write_dataset(table, tmp_path)
    res = ds.take([2, 0, 1])
    assert res.num_rows == 3
    assert res == table.take([2, 0, 1])


def test_create_table_from_pylist(tmp_path):
    data = [
        {"foo": 1, "bar": "one"},
        {"foo": 3, "bar": "three"},
    ]
    ds = lance.write_dataset(data, tmp_path)

    assert ds.to_table() == pa.Table.from_pylist(data)


def test_create_table_from_pydict(tmp_path):
    dat = {
        "foo": [1, 3],
        "bar": ["one", "three"],
    }
    ds = lance.write_dataset(dat, tmp_path)
    assert ds.to_table() == pa.Table.from_pydict(dat)


def test_metadata_cache_size(tmp_path):
    lance.write_dataset(pa.table({"id": [1, 2, 3]}), tmp_path / "test")

    ds = lance.dataset(tmp_path / "test")
    ds.to_table()  # Populate cache
    default_size = ds.session().size_bytes()
    assert default_size > 0

    ds = lance.dataset(tmp_path / "test", metadata_cache_size_bytes=0)
    ds.to_table()  # Attempt to populate cache (should be limited by 0 size)
    zero_cache_size = ds.session().size_bytes()

    # With zero cache size, session should be smaller than default
    # (it won't be exactly 0 due to struct overhead)
    assert zero_cache_size < default_size


def test_dataset_sql(tmp_path: Path):
    table = pa.table({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    ds = lance.write_dataset(table, tmp_path / "test")

    query = ds.sql("SELECT * FROM test WHERE id > 1").table_name("test").build()

    result = query.to_batch_records()
    expected = pa.table({"id": [2, 3], "value": ["b", "c"]})
    assert pa.Table.from_batches(result) == expected

    stream_result = (
        ds.sql("SELECT value FROM test WHERE id = 1")
        .table_name("test")
        .build()
        .to_stream_reader()
    )
    batches = list(stream_result)
    assert len(batches) == 1
    assert batches[0].to_pydict() == {"value": ["a"]}

    complex_query = ds.sql("""
                           SELECT id as user_id, UPPER(value) as val
                           FROM dataset
                           WHERE id BETWEEN 1 AND 3
                           """).build()
    complex_result = complex_query.to_batch_records()
    expected_complex = pa.table({"user_id": [1, 2, 3], "val": ["A", "B", "C"]})
    assert pa.Table.from_batches(complex_result) == expected_complex


def test_file_reader_options(tmp_path: Path):
    """Test cache_repetition_index and validate_on_decode options"""
    # Create a dataset with large repetitive strings to test cache_repetition_index
    # Using large strings to ensure repetition index is used
    large_string = "x" * 1000
    table = pa.table(
        {
            "id": range(10000),
            "text": [large_string] * 10000,  # Highly repetitive column
            "unique": [f"unique_{i}" for i in range(10000)],  # Non-repetitive column
        }
    )
    lance.write_dataset(table, tmp_path / "test")

    # Test cache_repetition_index reduces I/O operations
    # First read without cache
    dataset_no_cache = lance.dataset(
        tmp_path / "test", read_params={"cache_repetition_index": False}
    )
    iops_before = lance.iops_counter()
    result1 = dataset_no_cache.scanner(columns=["text"]).to_table()
    iops_without_cache = lance.iops_counter() - iops_before
    assert result1.num_rows == 10000

    # Second read with cache enabled
    dataset_with_cache = lance.dataset(
        tmp_path / "test", read_params={"cache_repetition_index": True}
    )
    iops_before = lance.iops_counter()
    result2 = dataset_with_cache.scanner(columns=["text"]).to_table()
    iops_with_cache = lance.iops_counter() - iops_before
    assert result2.num_rows == 10000

    # With cache, we should see fewer I/O operations for repetitive data
    # The difference might be small for small datasets, but should be measurable
    assert iops_with_cache <= iops_without_cache

    # Test validate_on_decode option
    # For now just verify it doesn't break normal operation
    dataset_validate = lance.dataset(
        tmp_path / "test", read_params={"validate_on_decode": True}
    )
    result3 = dataset_validate.to_table()
    assert result3.num_rows == 10000

    # Test both options together
    dataset_both = lance.dataset(
        tmp_path / "test",
        read_params={"cache_repetition_index": True, "validate_on_decode": False},
    )
    result4 = dataset_both.to_table()
    assert result4.num_rows == 10000

    # Test that scanner inherits options from dataset
    dataset_inherit = lance.dataset(
        tmp_path / "test",
        read_params={"cache_repetition_index": True, "validate_on_decode": True},
    )
    scanner = dataset_inherit.scanner()
    result5 = scanner.to_table()
    assert result5.num_rows == 10000

    # Verify the scanner is using the same options by checking I/O pattern
    iops_before = lance.iops_counter()
    scanner2 = dataset_inherit.scanner(columns=["text"])
    result6 = scanner2.to_table()
    assert result6.num_rows == 10000
    iops_scanner = lance.iops_counter() - iops_before

    # Scanner with inherited cache option should have similar I/O pattern as
    # direct dataset read
    assert iops_scanner <= iops_without_cache


def test_read_transaction_properties(tmp_path):
    """Test retrieving properties from transactions at different versions."""
    # Create schema and data for the dataset
    schema = pa.schema([pa.field("id", pa.int32()), pa.field("value", pa.string())])

    # First batch with properties
    batch1 = pa.RecordBatch.from_arrays(
        [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])], schema=schema
    )

    # Create the first version with properties
    properties1 = {
        LANCE_COMMIT_MESSAGE_KEY: "First commit",
        "custom_prop": "custom_value",
    }

    dataset = lance.write_dataset(batch1, tmp_path, transaction_properties=properties1)
    mytrans = dataset.read_transaction(1)
    print(mytrans)

    # Test retrieving properties from the first version
    transaction = dataset.read_transaction(1)
    props = transaction.transaction_properties
    assert props.get(LANCE_COMMIT_MESSAGE_KEY) == "First commit"
    assert props.get("custom_prop") == "custom_value"

    # Create a second batch with different properties
    batch2 = pa.RecordBatch.from_arrays(
        [pa.array([4, 5]), pa.array(["d", "e"])], schema=schema
    )

    # Add the second batch with different properties
    properties2 = {
        LANCE_COMMIT_MESSAGE_KEY: "Second commit",
        "another_prop": "another_value",
    }

    dataset = lance.write_dataset(
        batch2, tmp_path, mode="append", transaction_properties=properties2
    )

    # Test retrieving properties from the second version
    transaction = dataset.read_transaction(2)
    props = transaction.transaction_properties
    assert props.get(LANCE_COMMIT_MESSAGE_KEY) == "Second commit"
    assert props.get("another_prop") == "another_value"

    # Test retrieving properties from the first version again
    # to ensure old versions' properties are still accessible
    transaction = dataset.read_transaction(1)
    props = transaction.transaction_properties
    assert props.get(LANCE_COMMIT_MESSAGE_KEY) == "First commit"
    assert props.get("custom_prop") == "custom_value"


def test_get_properties_with_no_properties(tmp_path):
    """Test retrieving properties when none were set."""
    # Create a test dataset without properties

    # Create schema and data for the dataset
    schema = pa.schema([pa.field("id", pa.int32())])
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], schema=schema)

    # Create the dataset without properties
    dataset = lance.write_dataset(batch, tmp_path)

    # Test retrieving properties - should return None
    transaction = dataset.read_transaction(1)
    assert transaction.transaction_properties == {}


def test_commit_message_and_get_properties(tmp_path):
    """Test accessing commit messages via get_transactions."""
    table = pa.table({"a": [1]})

    # 1. Test case: Commit with a message
    dataset = lance.write_dataset(table, tmp_path, commit_message="first commit")
    transactions = dataset.get_transactions()
    assert len(transactions) == 1
    assert (
        transactions[0].transaction_properties.get(LANCE_COMMIT_MESSAGE_KEY)
        == "first commit"
    )
    # 2. Test case: Commit without a message
    lance.write_dataset(table, tmp_path, mode="append")
    dataset = lance.dataset(tmp_path)
    transactions = dataset.get_transactions()
    # Transactions are listed in reverse chronological order
    assert len(transactions) == 2
    # The latest transaction has no message,
    # so the key should be missing or properties is None
    assert (
        transactions[0].transaction_properties == {}
        or LANCE_COMMIT_MESSAGE_KEY not in transactions[0].transaction_properties
    )
    # The first transaction should still have the message
    assert (
        transactions[1].transaction_properties.get(LANCE_COMMIT_MESSAGE_KEY)
        == "first commit"
    )
    # 3. Test case: Transaction with no properties at all
    # A delete operation creates a new version that may have no properties.
    dataset.delete("a > 100")  # A no-op delete
    transactions = dataset.get_transactions()
    assert len(transactions) == 3
    # The latest transaction from delete should have no properties.
    assert transactions[0].transaction_properties == {}

    # 4. Test case: Commit using the commit method instead of write_dataset
    frags = lance.fragment.write_fragments(pa.table({"a": [5]}), dataset.uri)
    dataset = lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Append(frags),
        read_version=dataset.version,
        commit_message="Use Dataset.commit",
    )

    transactions = dataset.get_transactions()
    assert len(transactions) == 4
    assert (
        transactions[0].transaction_properties.get(LANCE_COMMIT_MESSAGE_KEY)
        == "Use Dataset.commit"
    )


def test_commit_with_stable_row_ids(tmp_path: Path):
    """Test that commit() with enable_stable_row_ids creates stable row IDs."""
    base_uri = str(tmp_path)
    table = pa.table({"a": range(10)})

    # Create dataset via commit with Overwrite and enable_stable_row_ids
    fragments = lance.fragment.write_fragments(table, base_uri)
    operation = lance.LanceOperation.Overwrite(table.schema, fragments)
    ds = lance.LanceDataset.commit(
        base_uri,
        operation,
        enable_stable_row_ids=True,
    )

    # Append more data
    table2 = pa.table({"a": range(10, 20)})
    fragments2 = lance.fragment.write_fragments(table2, base_uri)
    ds = lance.LanceDataset.commit(
        base_uri,
        lance.LanceOperation.Append(fragments2),
        read_version=ds.version,
    )

    # Verify row IDs are sequential (stable row IDs assign monotonic IDs)
    result = ds.scanner(with_row_id=True).to_table()
    assert len(result) == 20
    row_ids = [result["_rowid"][i].as_py() for i in range(20)]
    assert row_ids == list(range(20))


def test_table_metadata_updates(tmp_path: Path):
    """Test table metadata incremental updates and full replacement."""
    arr = pa.array([1, 2, 3])
    tbl = pa.table({"a": arr})
    ds = lance.write_dataset(tbl, tmp_path)

    # Test incremental updates
    ds.update_metadata({"key1": "value1", "key2": "value2"})
    metadata = ds.metadata
    assert metadata["key1"] == "value1"
    assert metadata["key2"] == "value2"

    # Test updating existing key
    ds.update_metadata({"key1": "updated_value1", "key3": "value3"})
    metadata = ds.metadata
    assert metadata["key1"] == "updated_value1"
    assert metadata["key2"] == "value2"  # Should remain
    assert metadata["key3"] == "value3"

    # Test deletion with None values
    ds.update_metadata({"key2": None, "key4": "value4"})
    metadata = ds.metadata
    assert metadata["key1"] == "updated_value1"
    assert "key2" not in metadata  # Should be deleted
    assert metadata["key3"] == "value3"
    assert metadata["key4"] == "value4"

    # Test full replacement
    ds.update_metadata({"new_key": "new_value"}, replace=True)
    metadata = ds.metadata
    assert metadata == {"new_key": "new_value"}  # All previous keys gone


def test_config_updates(tmp_path: Path):
    """Test config incremental updates and full replacement."""
    arr = pa.array([1, 2, 3])
    tbl = pa.table({"a": arr})
    ds = lance.write_dataset(tbl, tmp_path)

    original_config = ds.config()

    # Test incremental updates
    ds.update_config({"custom_key1": "value1", "custom_key2": "value2"})
    config = ds.config()
    assert config["custom_key1"] == "value1"
    assert config["custom_key2"] == "value2"
    # Original config should still be there
    for key, value in original_config.items():
        assert config[key] == value

    # Test updating existing custom key
    ds.update_config({"custom_key1": "updated_value1", "custom_key3": "value3"})
    config = ds.config()
    assert config["custom_key1"] == "updated_value1"
    assert config["custom_key2"] == "value2"
    assert config["custom_key3"] == "value3"

    # Test deletion with None values
    ds.update_config({"custom_key2": None, "custom_key4": "value4"})
    config = ds.config()
    assert config["custom_key1"] == "updated_value1"
    assert "custom_key2" not in config
    assert config["custom_key3"] == "value3"
    assert config["custom_key4"] == "value4"

    # Test full replacement (should preserve original config plus new values)
    ds.update_config({"only_custom": "only_value"}, replace=True)
    config = ds.config()
    assert config == {"only_custom": "only_value"}


def test_schema_metadata_updates(tmp_path: Path):
    """Test schema metadata incremental updates and full replacement."""
    arr = pa.array([1, 2, 3])
    tbl = pa.table({"a": arr})
    ds = lance.write_dataset(tbl, tmp_path)

    # Test incremental updates
    ds.update_schema_metadata({"schema_key1": "value1", "schema_key2": "value2"})
    schema_metadata = ds.schema_metadata
    assert schema_metadata["schema_key1"] == "value1"
    assert schema_metadata["schema_key2"] == "value2"

    # Test updating existing key
    ds.update_schema_metadata(
        {"schema_key1": "updated_value1", "schema_key3": "value3"}
    )
    schema_metadata = ds.schema_metadata
    assert schema_metadata["schema_key1"] == "updated_value1"
    assert schema_metadata["schema_key2"] == "value2"
    assert schema_metadata["schema_key3"] == "value3"

    # Test deletion with None values
    ds.update_schema_metadata({"schema_key2": None, "schema_key4": "value4"})
    schema_metadata = ds.schema_metadata
    assert schema_metadata["schema_key1"] == "updated_value1"
    assert "schema_key2" not in schema_metadata
    assert schema_metadata["schema_key3"] == "value3"
    assert schema_metadata["schema_key4"] == "value4"

    # Test full replacement
    ds.update_schema_metadata({"only_schema": "only_value"}, replace=True)
    schema_metadata = ds.schema_metadata
    assert schema_metadata == {"only_schema": "only_value"}


def test_field_metadata_updates(tmp_path: Path):
    """Test field metadata updates using field paths."""
    arr = pa.array([1, 2, 3])
    tbl = pa.table({"a": arr})
    ds = lance.write_dataset(tbl, tmp_path)

    # Test incremental updates using field path
    ds.update_field_metadata({"a": {"field_key1": "value1", "field_key2": "value2"}})

    # Get field metadata for verification
    field_a = None
    for f in ds.lance_schema.fields():
        if f.name() == "a":
            field_a = f
            break
    field_metadata = field_a.metadata
    assert field_metadata["field_key1"] == "value1"
    assert field_metadata["field_key2"] == "value2"

    # Test updating existing key
    ds.update_field_metadata(
        {"a": {"field_key1": "updated_value1", "field_key3": "value3"}}
    )
    field_a = None
    for f in ds.lance_schema.fields():
        if f.name() == "a":
            field_a = f
            break
    field_metadata = field_a.metadata
    assert field_metadata["field_key1"] == "updated_value1"
    assert field_metadata["field_key2"] == "value2"  # Should remain
    assert field_metadata["field_key3"] == "value3"

    # Test deletion with None values
    ds.update_field_metadata({"a": {"field_key2": None, "field_key4": "value4"}})
    field_a = None
    for f in ds.lance_schema.fields():
        if f.name() == "a":
            field_a = f
            break
    field_metadata = field_a.metadata
    assert field_metadata["field_key1"] == "updated_value1"
    assert "field_key2" not in field_metadata  # Should be deleted
    assert field_metadata["field_key3"] == "value3"
    assert field_metadata["field_key4"] == "value4"

    # Test full replacement
    ds.update_field_metadata({"a": {"only_field": "only_value"}}, replace=True)
    field_a = None
    for f in ds.lance_schema.fields():
        if f.name() == "a":
            field_a = f
            break
    field_metadata = field_a.metadata
    assert field_metadata == {"only_field": "only_value"}


def test_field_metadata_multiple_fields(tmp_path: Path):
    """Test field metadata updates on multiple fields at once."""
    arr1 = pa.array([1, 2, 3])
    arr2 = pa.array(["a", "b", "c"])
    tbl = pa.table({"num": arr1, "str": arr2})
    ds = lance.write_dataset(tbl, tmp_path)

    # Update metadata for both fields
    ds.update_field_metadata(
        {
            "num": {"type": "numeric", "unit": "count"},
            "str": {"type": "string", "encoding": "utf8"},
        }
    )

    # Verify both fields got updated
    num_field = None
    str_field = None
    for f in ds.lance_schema.fields():
        if f.name() == "num":
            num_field = f
        elif f.name() == "str":
            str_field = f

    assert num_field.metadata["type"] == "numeric"
    assert num_field.metadata["unit"] == "count"
    assert str_field.metadata["type"] == "string"
    assert str_field.metadata["encoding"] == "utf8"

    # Update one field while leaving the other
    ds.update_field_metadata({"num": {"unit": "items", "description": "Item count"}})

    num_field = None
    str_field = None
    for f in ds.lance_schema.fields():
        if f.name() == "num":
            num_field = f
        elif f.name() == "str":
            str_field = f

    # num field should be updated
    assert num_field.metadata["type"] == "numeric"  # preserved
    assert num_field.metadata["unit"] == "items"  # updated
    assert num_field.metadata["description"] == "Item count"  # added

    # str field should be unchanged
    assert str_field.metadata["type"] == "string"
    assert str_field.metadata["encoding"] == "utf8"


def test_metadata_apis_return_post_image(tmp_path: Path):
    """Test that metadata update methods return the post-update state."""
    arr = pa.array([1, 2, 3])
    tbl = pa.table({"a": arr})
    ds = lance.write_dataset(tbl, tmp_path)

    # Test table metadata
    result = ds.update_metadata({"key1": "value1", "key2": "value2"})
    assert result["key1"] == "value1"
    assert result["key2"] == "value2"

    result = ds.update_metadata({"key1": "updated", "key3": "value3"})
    assert result["key1"] == "updated"
    assert result["key2"] == "value2"
    assert result["key3"] == "value3"

    # Test config
    original_config = ds.config()
    result = ds.update_config({"custom": "value"})
    expected = {**original_config, "custom": "value"}
    assert result == expected

    # Test schema metadata
    result = ds.update_schema_metadata({"schema_key": "schema_value"})
    assert result["schema_key"] == "schema_value"


def test_update_config_transaction(tmp_path: Path):
    """Test UpdateConfig operation using transaction with new UpdateMap pattern."""
    # Create initial dataset
    arr = pa.array([1, 2, 3])
    tbl = pa.table({"a": arr, "b": [10, 20, 30]})
    schema = pa.schema(
        [
            pa.field("a", pa.int64(), metadata={b"field_a": b"original"}),
            pa.field("b", pa.int64(), metadata={b"field_b": b"original"}),
        ],
        metadata={b"schema_key": b"original"},
    )
    tbl = tbl.replace_schema_metadata(schema.metadata)
    ds = lance.write_dataset(tbl, tmp_path, schema=schema)

    # Test 1: Update configuration using UpdateConfig transaction
    config_update_map = lance.LanceOperation.UpdateMap(
        updates={"key1": "value1", "key2": "value2"}, replace=False
    )

    update_config_op = lance.LanceOperation.UpdateConfig(
        config_updates=config_update_map,
        table_metadata_updates=None,
        schema_metadata_updates=None,
        field_metadata_updates=None,
    )

    transaction = lance.Transaction(
        read_version=ds.version, operation=update_config_op, uuid=str(uuid.uuid4())
    )

    # Commit transaction
    ds_v2 = lance.LanceDataset.commit(tmp_path, transaction)
    assert ds_v2.version == 2
    config = ds_v2.config()
    assert config["key1"] == "value1"
    assert config["key2"] == "value2"

    # Test 2: Delete configuration key with null value
    config_delete_map = lance.LanceOperation.UpdateMap(
        updates={"key1": None},  # None means delete
        replace=False,
    )

    update_config_op2 = lance.LanceOperation.UpdateConfig(
        config_updates=config_delete_map,
        table_metadata_updates=None,
        schema_metadata_updates=None,
        field_metadata_updates=None,
    )

    transaction2 = lance.Transaction(
        read_version=ds_v2.version, operation=update_config_op2, uuid=str(uuid.uuid4())
    )

    ds_v3 = lance.LanceDataset.commit(tmp_path, transaction2)
    assert ds_v3.version == 3
    config = ds_v3.config()
    assert "key1" not in config
    assert config["key2"] == "value2"

    # Test 3: Update schema metadata
    schema_update_map = lance.LanceOperation.UpdateMap(
        updates={"schema_new": "schema_value", "schema_key": "updated"}, replace=False
    )

    update_config_op3 = lance.LanceOperation.UpdateConfig(
        config_updates=None,
        table_metadata_updates=None,
        schema_metadata_updates=schema_update_map,
        field_metadata_updates=None,
    )

    transaction3 = lance.Transaction(
        read_version=ds_v3.version, operation=update_config_op3, uuid=str(uuid.uuid4())
    )

    ds_v4 = lance.LanceDataset.commit(tmp_path, transaction3)
    assert ds_v4.version == 4
    schema_metadata = ds_v4.schema_metadata
    assert schema_metadata["schema_new"] == "schema_value"
    assert schema_metadata["schema_key"] == "updated"

    # Test 4: Update field metadata for multiple fields
    field_a_update_map = lance.LanceOperation.UpdateMap(
        updates={"field_a": "updated_a", "new_key_a": "new_value_a"}, replace=False
    )

    field_b_update_map = lance.LanceOperation.UpdateMap(
        updates={"field_b": "updated_b", "new_key_b": "new_value_b"}, replace=False
    )

    update_config_op4 = lance.LanceOperation.UpdateConfig(
        config_updates=None,
        table_metadata_updates=None,
        schema_metadata_updates=None,
        field_metadata_updates={
            0: field_a_update_map,  # field "a"
            1: field_b_update_map,  # field "b"
        },
    )

    transaction4 = lance.Transaction(
        read_version=ds_v4.version, operation=update_config_op4, uuid=str(uuid.uuid4())
    )

    ds_v5 = lance.LanceDataset.commit(tmp_path, transaction4)
    assert ds_v5.version == 5

    # Verify field metadata updates
    field_a = None
    field_b = None
    for field in ds_v5.lance_schema.fields():
        if field.name() == "a":
            field_a = field
        elif field.name() == "b":
            field_b = field

    assert field_a is not None
    assert field_b is not None
    assert field_a.metadata["field_a"] == "updated_a"
    assert field_a.metadata["new_key_a"] == "new_value_a"
    assert field_b.metadata["field_b"] == "updated_b"
    assert field_b.metadata["new_key_b"] == "new_value_b"

    # Test 5: Replace mode - completely replace metadata
    table_replace_map = lance.LanceOperation.UpdateMap(
        updates={"only_key": "only_value"},
        replace=True,  # This will replace all existing table metadata
    )

    # First add some table metadata to replace
    ds_v5.update_metadata(
        {"existing_key": "existing_value", "another_key": "another_value"}
    )
    # Reload dataset to get the updated version
    ds_v6 = lance.dataset(tmp_path)

    update_config_op5 = lance.LanceOperation.UpdateConfig(
        config_updates=None,
        table_metadata_updates=table_replace_map,
        schema_metadata_updates=None,
        field_metadata_updates=None,
    )

    transaction5 = lance.Transaction(
        read_version=ds_v6.version,  # Use the actual latest version
        operation=update_config_op5,
        uuid=str(uuid.uuid4()),
    )

    ds_v7 = lance.LanceDataset.commit(tmp_path, transaction5)
    table_metadata = ds_v7.metadata
    assert table_metadata == {
        "only_key": "only_value"
    }  # All previous metadata should be gone


def test_shallow_clone(tmp_path: Path):
    """Shallow clone a filesystem dataset by version number and by tag.

    Arrange:
      - Create a source dataset at a filesystem path with two versions
        (create v1, then overwrite to v2).
      - Create a tag "v1" pointing to version 1.
    Act:
      - Shallow clone by version=2 (numeric) to one destination.
      - Shallow clone by version="v1" (tag) to another destination.
    Assert:
      - Re-open cloned datasets and verify their tables equal the source
        version they were cloned from (schema and record count).

    This test uses pathlib paths and tmp_path for cross-platform compatibility
    and should not skip on Windows.
    """
    # Prepare source dataset with two versions
    src_dir = tmp_path / "shallow_src"
    table_v1 = pa.table({"a": [1, 2, 3], "b": [10, 20, 30]})
    lance.write_dataset(table_v1, src_dir, mode="create")

    table_v2 = pa.table({"a": [4, 5, 6], "b": [40, 50, 60]})
    ds = lance.write_dataset(table_v2, src_dir, mode="overwrite")

    # Create a tag pointing to version 1
    ds.tags.create("v1", 1)

    # Clone by numeric version (v2) and assert equality
    clone_v2_dir = tmp_path / "clone_v2"
    ds_clone_v2 = ds.shallow_clone(clone_v2_dir, 2)
    assert ds_clone_v2.to_table() == table_v2
    assert lance.dataset(clone_v2_dir).to_table() == table_v2

    # Clone by tag (v1) and assert equality
    clone_v1_tag_dir = tmp_path / "clone_v1_tag"
    ds_clone_v1_tag = ds.shallow_clone(clone_v1_tag_dir, "v1")
    assert ds_clone_v1_tag.to_table() == table_v1
    assert lance.dataset(clone_v1_tag_dir).to_table() == table_v1

    table_v3 = pa.table({"a": [7, 8, 9], "b": [40, 50, 60]})
    branch = ds.create_branch("branch", 2)
    lance.write_dataset(table_v3, branch.uri, mode="overwrite")
    clone_branch_v3 = tmp_path / "clone_branch_v3"
    cloned_by_branch = branch.shallow_clone(clone_branch_v3, 3)
    assert cloned_by_branch.to_table() == table_v3
    assert lance.dataset(clone_branch_v3).to_table() == table_v3


def test_branches(tmp_path: Path):
    # Step 1: create branch1 from main → append to branch1 → create branch2 from tag
    base_dir = tmp_path / "test_branches"
    main_table = pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
    ds_main = lance.write_dataset(main_table, base_dir)

    branch1 = ds_main.create_branch("branch1")
    ds_main.branches.replace_metadata("branch1", {"description": "branch one"})
    assert branch1.version == 1
    # The dataset returned by create_branch must be fully constructed
    assert branch1.checkout_version(("main", None)).version == 1
    branch1_append = pa.Table.from_pydict({"a": [7, 8], "b": [9, 10]})
    branch1 = lance.write_dataset(branch1_append, branch1, mode="append")
    assert branch1.version == 2

    expected_branch1 = pa.Table.from_pydict(
        {"a": [1, 2, 3, 7, 8], "b": [4, 5, 6, 9, 10]}
    )
    assert branch1.to_table().combine_chunks() == expected_branch1.combine_chunks()

    # Step 2:
    # tag latest of branch1 → create branch2 from that tag
    # test create tag on the main branch by different ways
    # test create branch from the main branch by specifying "main"
    branch1.tags.create("branch1_latest", ("branch1", None))
    branch1.tags.create("main_latest", (None, None))
    branch1.tags.create("main_latest2", ("main", None))
    branch1.create_branch("branch_from_main", ("main", None))
    ordered_tags = dict(branch1.tags.list_ordered())
    branches_with_main = branch1.branches.list()
    assert branch1.tags.list()["branch1_latest"]["branch"] == "branch1"
    assert branch1.tags.list()["main_latest"]["branch"] is None
    assert branch1.tags.list()["main_latest2"]["branch"] is None
    assert ordered_tags["branch1_latest"]["branch"] == "branch1"
    assert ordered_tags["main_latest"]["branch"] is None
    assert branches_with_main["branch_from_main"]["parent_branch"] is None
    assert branches_with_main["branch_from_main"]["branch_identifier"][0][0] == 1
    assert isinstance(
        branches_with_main["branch_from_main"]["branch_identifier"][0][1], str
    )
    assert branches_with_main["branch_from_main"]["parent_version"] == 1
    assert branch1.checkout_version("main_latest").latest_version == 1
    assert branch1.checkout_version("main_latest2").latest_version == 1
    assert branch1.checkout_version(("branch_from_main", None)).latest_version == 1
    branch2 = branch1.create_branch("branch2", "branch1_latest")
    assert branch2.version == 2

    # Step 3: append more data to branch2 → verify contains branch1 data + new
    branch2_append = pa.Table.from_pydict({"a": [11], "b": [12]})
    branch2 = lance.write_dataset(branch2_append, branch2, mode="append")
    assert branch2.version == 3

    expected_branch2 = pa.Table.from_pydict(
        {"a": [1, 2, 3, 7, 8, 11], "b": [4, 5, 6, 9, 10, 12]}
    )
    assert branch2.to_table().combine_chunks() == expected_branch2.combine_chunks()

    # Step 4: validate branches.list() and metadata
    # delete branch1; validate list after deletion
    branches = ds_main.branches.list()
    assert "branch1" in branches
    assert "branch2" in branches

    b1_meta = branches["branch1"]
    assert isinstance(b1_meta["parent_version"], int)
    assert b1_meta["manifest_size"] > 0
    assert b1_meta["branch_identifier"][0][0] == b1_meta["parent_version"]
    assert isinstance(b1_meta["branch_identifier"][0][1], str)
    assert len(b1_meta["branch_identifier"][0][1]) > 0
    assert "create_at" in b1_meta
    assert b1_meta["metadata"] == {"description": "branch one"}
    ordered_branches = dict(ds_main.branches.list_ordered())
    assert ordered_branches["branch1"]["metadata"] == {"description": "branch one"}
    assert "metadata" in ordered_branches["branch2"]

    try:
        ds_main.checkout_version("branch_not_exists")
        assert False, "Expected OSError was not raised"
    except OSError as e:
        if "does not exist" not in str(e):
            raise

    ds_main.branches.delete("branch2")
    branches_after = ds_main.branches.list()
    assert "branch2" not in branches_after
    assert "branch1" in branches_after

    branch1 = ds_main.checkout_version(("branch1", None))
    assert branch1.version == 2
    assert branch1.to_table().combine_chunks() == expected_branch1.combine_chunks()
    branch1 = ds_main.checkout_version(("branch1", 1))
    assert branch1.version == 1
    assert branch1.to_table().combine_chunks() == main_table.combine_chunks()
    branch1.checkout_latest()
    assert branch1.version == 2
    assert branch1.to_table().combine_chunks() == expected_branch1.combine_chunks()


def test_default_scan_options_nearest(tmp_path: Path) -> None:
    dim = 4
    num_rows = 10

    values = []
    for i in range(num_rows):
        values.extend(float(i) for _ in range(dim))
    value_array = pa.array(values, type=pa.float32())
    vector_array = pa.FixedSizeListArray.from_arrays(value_array, dim)
    table = pa.Table.from_pydict({"vector": vector_array, "id": list(range(num_rows))})

    base_dir = tmp_path / "nearest_default_scan_options"
    lance.write_dataset(table, base_dir)

    query_vec = [0.0] * dim
    default_scan_options = {
        "nearest": {
            "column": "vector",
            "q": query_vec,
            "k": 5,
        },
    }

    ds = lance.dataset(base_dir, default_scan_options=default_scan_options)
    result = ds.to_table()

    assert result.num_rows == 5

    assert "_distance" in result.column_names
    distances = result["_distance"].to_pylist()
    assert distances == sorted(distances)


def test_tracked_files(tmp_path):
    table = pa.table({"x": [1, 2, 3]})
    ds = lance.write_dataset(table, tmp_path / "ds")
    ds.delete("x = 2")  # adds a deletion file

    reader = ds.tracked_files()
    assert isinstance(reader, pa.RecordBatchReader)

    result = reader.read_all()
    assert result.schema.field("version").type == pa.int64()
    assert result.num_rows >= 2  # at least manifest + data file

    types = set(result.column("type").to_pylist())
    assert "manifest" in types
    assert "data file" in types
    assert "deletion file" in types


def test_all_files(tmp_path):
    table = pa.table({"x": [1, 2, 3]})
    ds = lance.write_dataset(table, tmp_path / "ds")

    reader = ds.all_files()
    assert isinstance(reader, pa.RecordBatchReader)

    result = reader.read_all()
    assert result.schema.field("size_bytes").type == pa.int64()
    assert result.num_rows >= 2  # at least manifest + data file
    assert all(s > 0 for s in result.column("size_bytes").to_pylist())
