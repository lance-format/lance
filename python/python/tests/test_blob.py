# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import gc
import importlib
import io
import json
import queue
import random
import shutil
import subprocess
import sys
import tarfile
import textwrap
import threading
import uuid
import zipfile
from datetime import timedelta
from pathlib import Path

import lance
import pandas as pd
import pyarrow as pa
import pytest
from lance import Blob, BlobColumn, BlobFile, DatasetBasePath
from lance.blob import (
    BlobRefBatch,
    BlobRefCapture,
    BlobRefReader,
    BlobSourceContext,
    BlobType,
    NestedBlobRefBatch,
)
from lance.file import LanceFileSession
from lance.fragment import write_fragments

lance_dataset_module = importlib.import_module("lance.dataset")

LEGACY_BLOB_STORAGE_VERSION = "2.1"


@pytest.mark.parametrize("complete", [False, True], ids=["minimal", "complete"])
@pytest.mark.parametrize(
    "parent",
    ["video", "meta.video", "meta.`video.dot`"],
    ids=["top-level", "nested", "quoted"],
)
@pytest.mark.parametrize(
    "child, value",
    [
        ("kind", 0),
        ("position", 0),
        ("size", 5),
        ("blob_id", 1),
        ("blob_uri", "external.bin"),
    ],
)
def test_blob_v2_descriptor_subfield_queries(tmp_path, complete, parent, child, value):
    external = tmp_path / "external.bin"
    external.write_bytes(b"world!")
    blobs = lance.blob_array([b"hello", None, external.as_uri(), b"again"])
    if not complete:
        storage_type = pa.struct(list(blobs.type.storage_type)[:2])
        blob_type = BlobType.__arrow_ext_deserialize__(storage_type, b"")
        storage = pa.StructArray.from_arrays(
            [blobs.storage.field("data"), blobs.storage.field("uri")],
            fields=list(storage_type),
            mask=blobs.is_null(),
        )
        blobs = pa.ExtensionArray.from_storage(blob_type, storage)
    if parent != "video":
        leaf_name = "video.dot" if "`" in parent else "video"
        data = pa.table(
            {
                "id": [0, 1, 2, 3],
                "meta": pa.StructArray.from_arrays([blobs], names=[leaf_name]),
            }
        )
    else:
        data = pa.table({"id": [0, 1, 2, 3], "video": blobs})
    ds = lance.write_dataset(
        data,
        tmp_path / "dataset",
        data_storage_version="2.2",
        initial_bases=[DatasetBasePath(tmp_path.as_uri(), name="external", id=1)],
        max_rows_per_file=3,
        max_rows_per_group=3,
    )
    assert len(ds.get_fragments()) == 2
    descriptors = ds.to_table(columns=[parent])[parent].to_pylist()
    expected = [None if row is None else row[child] for row in descriptors]
    column = f"{parent}.{child}"
    for columns in [[column], [parent, column]]:
        result = ds.to_table(columns=columns)
        assert result.column_names == columns
        assert result[column].to_pylist() == expected

    literal = f"'{value}'" if isinstance(value, str) else str(value)
    expected_ids = [i for i, item in enumerate(expected) if item == value]
    for columns in [["id"], ["id", parent]]:
        result = ds.to_table(columns=columns, filter=f"{column} = {literal}")
        assert result.column_names == columns
        assert result["id"].to_pylist() == expected_ids
        if parent in columns:
            assert result[parent].to_pylist() == [descriptors[i] for i in expected_ids]
        result = ds.to_table(columns=columns, filter=f"{column} IS NULL")
        assert result.column_names == columns
        assert result["id"].to_pylist() == [
            i for i, item in enumerate(expected) if item is None
        ]

    for index, (predicate, deleted_ids) in enumerate(
        [
            (f"{column} = {literal}", expected_ids),
            (f"{column} IS NULL", [1]),
            (f"{column} IS NOT NULL", [0, 2, 3]),
        ]
    ):
        assert ds.to_table(columns=["id"], filter=predicate)["id"].to_pylist() == (
            deleted_ids
        )
        path = shutil.copytree(tmp_path / "dataset", tmp_path / f"delete_{index}")
        deleted = lance.dataset(path)
        deleted.delete(predicate)
        remaining_ids = [i for i in range(4) if i not in deleted_ids]
        result = deleted.to_table(columns=["id", parent])
        assert result["id"].to_pylist() == remaining_ids
        assert result[parent].to_pylist() == [descriptors[i] for i in remaining_ids]


def _blob_row_ids(dataset):
    return dataset.to_table(columns=[], with_row_id=True).column("_rowid").to_pylist()


def _blob_row_addresses(dataset):
    return (
        dataset.to_table(columns=["idx"], with_row_address=True)
        .column("_rowaddr")
        .to_pylist()
    )


def _commit_blob_fragments(dataset_uri, schema, fragments, initial_bases=None):
    operation = lance.LanceOperation.Overwrite(
        schema,
        fragments,
        initial_bases=initial_bases,
    )
    return lance.LanceDataset.commit(dataset_uri, operation)


def _external_blob_table(blob_path, payload=b"hello"):
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(payload)
    return pa.table({"blob": lance.blob_array([blob_path.as_uri()])})


def _blob_sidecar_path(data_dir, data_file_key, blob_id):
    sidecar_name = f"{int(f'{blob_id:032b}'[::-1], 2):032b}.blob"
    return data_dir / data_file_key / sidecar_name


def _complete_blob_table(ids, values):
    schema = pa.schema([pa.field("id", pa.int64()), lance.blob_field("blob")])
    return pa.Table.from_arrays(
        [pa.array(ids, type=pa.int64()), lance.blob_array(values)],
        schema=schema,
    )


def _complete_blob_storage_table(data, uri, position, size):
    storage = pa.StructArray.from_arrays(
        [
            pa.array([data], type=pa.large_binary()),
            pa.array([uri], type=pa.utf8()),
            pa.array([position], type=pa.uint64()),
            pa.array([size], type=pa.uint64()),
        ],
        names=["data", "uri", "position", "size"],
    )
    blobs = pa.ExtensionArray.from_storage(BlobType(), storage)
    return pa.Table.from_arrays([blobs], schema=pa.schema([lance.blob_field("blob")]))


def _assert_complete_blob_schema(dataset):
    blob_type = dataset.schema.field("blob").type
    assert isinstance(blob_type, pa.ExtensionType)
    assert blob_type.extension_name == "lance.blob.v2"
    assert [field.name for field in blob_type.storage_type] == [
        "data",
        "uri",
        "position",
        "size",
    ]


def _assert_blob_storage_type_equal(actual, expected):
    assert isinstance(actual, pa.StructType)
    assert isinstance(expected, pa.StructType)
    assert len(actual) == len(expected)
    for actual_field, expected_field in zip(actual, expected):
        assert actual_field.equals(expected_field, check_metadata=True)


def _inline_blob_array(storage_type, value):
    arrays = [
        pa.array([value], type=pa.large_binary()),
        pa.array([None], type=pa.utf8()),
    ]
    if len(storage_type) == 4:
        arrays.extend(
            [
                pa.array([None], type=pa.uint64()),
                pa.array([None], type=pa.uint64()),
            ]
        )
    storage = pa.StructArray.from_arrays(arrays, fields=list(storage_type))
    blob_type = BlobType.__arrow_ext_deserialize__(storage_type, b"")
    return pa.ExtensionArray.from_storage(blob_type, storage)


def _add_columns_blob_v2_values(tmp_path):
    external_base = tmp_path / "external_base"
    external_blob = external_base / "external_blob.bin"
    external_blob.parent.mkdir(parents=True, exist_ok=True)
    external_blob.write_bytes(b"external")

    payloads = [
        b"inline",
        b"p" * (64 * 1024 + 1024),
        b"d" * (4 * 1024 * 1024 + 1024),
        b"external",
    ]
    values = [payloads[0], payloads[1], payloads[2], external_blob.as_uri()]
    initial_bases = [DatasetBasePath(external_base.as_uri(), name="external", id=1)]
    return values, payloads, initial_bases


def _assert_blob_v2_add_columns_result(dataset, column, payloads):
    desc = dataset.to_table(columns=[column]).column(column).chunk(0)

    assert desc.field("kind").to_pylist() == [0, 1, 2, 3]
    assert desc.field("blob_id").to_pylist()[3] == 1
    assert desc.field("blob_uri").to_pylist()[3] == "external_blob.bin"

    blobs = dataset.take_blobs(column, indices=range(len(payloads)))
    assert [blob.readall() for blob in blobs] == payloads


def _dataset_file_set(dataset_path):
    return {
        path.relative_to(dataset_path)
        for path in dataset_path.rglob("*")
        if path.is_file()
    }


def _write_two_fragment_blob_v2_seed_dataset(tmp_path, name):
    values, payloads, initial_bases = _add_columns_blob_v2_values(tmp_path)
    dataset_path = tmp_path / name
    ds = lance.write_dataset(
        pa.table({"id": range(8)}),
        dataset_path,
        data_storage_version="2.2",
        initial_bases=initial_bases,
        max_rows_per_file=4,
        max_rows_per_group=4,
    )
    return ds, dataset_path, values, payloads


def _out_of_order_blob_selection(dataset_with_blobs, selection_kind):
    addresses = _blob_row_addresses(dataset_with_blobs)
    expected = [(addresses[4], b"quux"), (addresses[0], b"foo")]

    if selection_kind == "ids":
        return [
            _blob_row_ids(dataset_with_blobs)[4],
            _blob_row_ids(dataset_with_blobs)[0],
        ], expected
    if selection_kind == "addresses":
        return [addresses[4], addresses[0]], expected
    return [4, 0], expected


def test_blob_read_from_binary():
    values = [b"foo", b"bar", b"baz"]
    data = pa.table(
        {
            "bin": pa.array(values, type=pa.binary()),
            "largebin": pa.array(values, type=pa.large_binary()),
        }
    )

    for col_name in ["bin", "largebin"]:
        blobs = BlobColumn(data.column(col_name))
        for i, f in enumerate(blobs):
            assert f.read() in values[i]


def test_blob_reject_invalid_col():
    values = pa.array([1, 2, 3])
    with pytest.raises(ValueError, match="Expected a binary array"):
        BlobColumn(values)


def test_blob_descriptions(tmp_path):
    values = pa.array([b"foo", b"bar", b"baz"], pa.large_binary())
    table = pa.table(
        [values],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                )
            ]
        ),
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds",
        data_storage_version=LEGACY_BLOB_STORAGE_VERSION,
    )
    # These positions may be surprising but lance pads buffers to 64-byte boundaries
    expected_positions = pa.array([0, 64, 128], pa.uint64())
    expected_sizes = pa.array([3, 3, 3], pa.uint64())
    descriptions = ds.to_table().column("blobs").chunk(0)

    assert descriptions.field(0) == expected_positions
    assert descriptions.field(1) == expected_sizes


def test_scan_blob_as_binary(tmp_path):
    values = [b"foo", b"bar", b"baz"]
    arr = pa.array(values, pa.large_binary())
    table = pa.table(
        [arr],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                )
            ]
        ),
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds",
        data_storage_version=LEGACY_BLOB_STORAGE_VERSION,
    )

    tbl = ds.scanner(columns=["blobs"], blob_handling="all_binary").to_table()
    assert tbl.column("blobs").to_pylist() == values


def test_sql_blob_as_binary(tmp_path):
    values = [b"foo", b"bar", b"baz"]
    table = pa.table(
        [pa.array(values, pa.large_binary())],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                )
            ]
        ),
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds",
        data_storage_version=LEGACY_BLOB_STORAGE_VERSION,
    )

    batches = (
        ds.sql("SELECT blobs FROM dataset")
        .blob_handling("all_binary")
        .build()
        .to_batch_records()
    )
    assert pa.Table.from_batches(batches).column("blobs").to_pylist() == values


def test_v2_0_blob_descriptor_projection_and_reads(tmp_path):
    values = [b"abc", b"defgh", b"ijklmnop"]
    blob_field = pa.field(
        "blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
    )
    schema = pa.schema([pa.field("id", pa.int32()), blob_field])
    table = pa.table(
        {"id": [0, 1, 2], "blob": values},
        schema=schema,
    )
    ds = lance.write_dataset(table, tmp_path / "test_ds", data_storage_version="2.0")

    for blob_handling in [None, "all_descriptions", "blobs_descriptions"]:
        kwargs = {} if blob_handling is None else {"blob_handling": blob_handling}
        descriptions = ds.scanner(columns=["blob"], **kwargs).to_table().column("blob")
        chunk = descriptions.chunk(0)
        assert chunk.type == pa.struct(
            [
                pa.field("position", pa.uint64()),
                pa.field("size", pa.uint64()),
            ]
        )
        assert chunk.field("size").to_pylist() == [3, 5, 8]

    tbl = ds.scanner(columns=["blob"], blob_handling="all_binary").to_table()
    assert tbl.column("blob").to_pylist() == values
    assert [blob.read() for blob in ds.take_blobs("blob", indices=[0, 1, 2])] == values
    assert ds.read_blobs("blob", indices=[0, 1, 2]) == [
        (0, b"abc"),
        (1, b"defgh"),
        (2, b"ijklmnop"),
    ]


def test_fragment_scan_blob_as_binary(tmp_path):
    values = [b"foo", b"bar", b"baz"]
    arr = pa.array(values, pa.large_binary())
    table = pa.table(
        [arr],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                )
            ]
        ),
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds",
        data_storage_version=LEGACY_BLOB_STORAGE_VERSION,
    )

    fragment = ds.get_fragments()[0]

    tbl = fragment.scanner(columns=["blobs"], blob_handling="all_binary").to_table()
    assert tbl.column("blobs").to_pylist() == values

    tbl = fragment.to_table(columns=["blobs"], blob_handling="all_binary")
    assert tbl.column("blobs").to_pylist() == values


@pytest.fixture
def dataset_with_blobs(tmp_path):
    values = pa.array([b"foo", b"bar", b"baz"], pa.large_binary())
    idx = pa.array([0, 1, 2], pa.uint64())
    table = pa.table(
        [values, idx],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                ),
                pa.field("idx", pa.uint64()),
            ]
        ),
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds",
        data_storage_version=LEGACY_BLOB_STORAGE_VERSION,
    )

    values = pa.array([b"qux", b"quux", b"corge"], pa.large_binary())
    idx = pa.array([3, 4, 5], pa.uint64())
    table = pa.table(
        [values, idx],
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                ),
                pa.field("idx", pa.uint64()),
            ]
        ),
    )
    ds.insert(table)
    return ds


@pytest.fixture
def dataset_with_mixed_blob_v2(tmp_path):
    external_blob = tmp_path / "external.bin"
    external_blob.write_bytes(b"external")
    payloads = [b"in", b"packed!!", b"dedicated payload", b"external", None, b""]
    values = [
        payloads[0],
        payloads[1],
        payloads[2],
        external_blob.as_uri(),
        None,
        payloads[5],
    ]
    schema = pa.schema(
        [
            lance.blob_field(
                "blobs",
                inline_size_threshold=4,
                dedicated_size_threshold=12,
            ),
            pa.field("idx", pa.uint64()),
        ]
    )
    table = pa.table(
        {"blobs": lance.blob_array(values), "idx": range(len(values))}, schema=schema
    )
    dataset = lance.write_dataset(
        table,
        tmp_path / "mixed_blob_v2",
        data_storage_version="2.2",
        allow_external_blob_outside_bases=True,
        max_rows_per_file=2,
        max_rows_per_group=2,
    )
    assert len(dataset.get_fragments()) == 3
    return dataset, payloads


def test_blob_files(dataset_with_blobs):
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)

    for expected in [b"foo", b"bar", b"baz"]:
        with blobs.pop(0) as f:
            assert f.read() == expected


def test_blob_files_close_no_shutdown_panic(tmp_path):
    script = textwrap.dedent(
        f"""
        import pyarrow as pa
        import lance

        table = pa.table(
            [pa.array([b"foo", b"bar"], pa.large_binary())],
            schema=pa.schema(
                [
                    pa.field(
                        "blob",
                        pa.large_binary(),
                        metadata={{"lance-encoding:blob": "true"}},
                    )
                ]
            ),
        )
        ds = lance.write_dataset(
            table,
            {str(tmp_path / "ds")!r},
            data_storage_version={LEGACY_BLOB_STORAGE_VERSION!r},
        )
        row_ids = ds.to_table(columns=[], with_row_id=True).column("_rowid").to_pylist()
        blobs = ds.take_blobs("blob", ids=row_ids)
        for blob in blobs:
            blob.close()
        print("done")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "interpreter_lifecycle.rs" not in result.stderr
    assert "The Python interpreter is not initialized" not in result.stderr


def test_blob_files_by_address(dataset_with_blobs):
    addresses = (
        dataset_with_blobs.to_table(columns=[], with_row_address=True)
        .column("_rowaddr")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs("blobs", addresses=addresses)

    for expected in [b"foo", b"bar", b"baz"]:
        with blobs.pop(0) as f:
            assert f.read() == expected


def test_blob_files_by_address_with_stable_row_ids(tmp_path):
    table = pa.table(
        {
            "blobs": pa.array([b"foo"], pa.large_binary()),
            "idx": pa.array([0], pa.uint64()),
        },
        schema=pa.schema(
            [
                pa.field(
                    "blobs", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                ),
                pa.field("idx", pa.uint64()),
            ]
        ),
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds",
        data_storage_version=LEGACY_BLOB_STORAGE_VERSION,
        enable_stable_row_ids=True,
    )

    ds.insert(
        pa.table(
            {
                "blobs": pa.array([b"bar"], pa.large_binary()),
                "idx": pa.array([1], pa.uint64()),
            },
            schema=table.schema,
        )
    )

    t = ds.to_table(columns=["idx"], with_row_address=True)
    row_idx = t.column("idx").to_pylist().index(1)
    addr = t.column("_rowaddr").to_pylist()[row_idx]

    blobs = ds.take_blobs("blobs", addresses=[addr])
    assert len(blobs) == 1
    with blobs[0] as f:
        assert f.read() == b"bar"


def test_blob_by_indices(tmp_path, dataset_with_blobs):
    indices = [0, 4]
    blobs = dataset_with_blobs.take_blobs("blobs", indices=indices)

    blobs2 = dataset_with_blobs.take_blobs("blobs", ids=[0, (1 << 32) + 1])
    assert len(blobs) == len(blobs2)
    for b1, b2 in zip(blobs, blobs2):
        with b1 as f1, b2 as f2:
            assert f1.read() == f2.read()


@pytest.mark.parametrize(
    ("selection_kind", "selection_values", "expected"),
    [
        ("ids", [0, (1 << 32) + 1], [(0, b"foo"), ((1 << 32) + 1, b"quux")]),
        ("addresses", [0, (1 << 32) + 1], [(0, b"foo"), ((1 << 32) + 1, b"quux")]),
        ("indices", [0, 4], [(0, b"foo"), ((1 << 32) + 1, b"quux")]),
    ],
)
def test_read_blobs(dataset_with_blobs, selection_kind, selection_values, expected):
    kwargs = {selection_kind: selection_values}

    blobs = dataset_with_blobs.read_blobs(
        "blobs",
        **kwargs,
        io_buffer_size=1024,
        preserve_order=True,
    )

    assert blobs == expected


def test_read_blobs_requires_single_selector(dataset_with_blobs):
    with pytest.raises(
        ValueError, match="Exactly one of ids, indices, or addresses must be specified"
    ):
        dataset_with_blobs.read_blobs("blobs", ids=[0], indices=[0])


def test_read_blobs_requires_selector(dataset_with_blobs):
    with pytest.raises(
        ValueError, match="Exactly one of ids, indices, or addresses must be specified"
    ):
        dataset_with_blobs.read_blobs("blobs")


def test_read_blobs_rejects_non_blob_column(dataset_with_blobs):
    with pytest.raises(ValueError, match="not a blob column"):
        dataset_with_blobs.read_blobs("idx", indices=[0])


@pytest.mark.parametrize(
    ("selection_kind", "selection_values", "expected"),
    [
        (
            "ids",
            pa.array([0, (1 << 32) + 1], type=pa.uint64()),
            [(0, b"foo"), ((1 << 32) + 1, b"quux")],
        ),
        (
            "addresses",
            pa.array([0, (1 << 32) + 1], type=pa.uint64()),
            [(0, b"foo"), ((1 << 32) + 1, b"quux")],
        ),
        (
            "indices",
            pa.array([0, 4], type=pa.uint64()),
            [(0, b"foo"), ((1 << 32) + 1, b"quux")],
        ),
    ],
)
def test_read_blobs_accepts_arrow_array_selectors(
    dataset_with_blobs, selection_kind, selection_values, expected
):
    kwargs = {selection_kind: selection_values}

    blobs = dataset_with_blobs.read_blobs("blobs", **kwargs)

    assert blobs == expected


@pytest.mark.parametrize(
    ("selection_kind", "selection_values"),
    [
        ("ids", []),
        ("addresses", []),
        ("indices", []),
        ("ids", pa.array([], type=pa.uint64())),
        ("addresses", pa.array([], type=pa.uint64())),
        ("indices", pa.array([], type=pa.uint64())),
    ],
)
def test_read_blobs_accepts_empty_selection(
    dataset_with_blobs, selection_kind, selection_values
):
    kwargs = {selection_kind: selection_values}

    assert dataset_with_blobs.read_blobs("blobs", **kwargs) == []


@pytest.mark.parametrize(
    ("planner_kwargs", "error_message"),
    [
        ({"io_buffer_size": 0}, "io_buffer_size must be greater than 0"),
    ],
)
def test_read_blobs_rejects_invalid_planner_options(
    dataset_with_blobs, planner_kwargs, error_message
):
    with pytest.raises(ValueError, match=error_message):
        dataset_with_blobs.read_blobs("blobs", indices=[0], **planner_kwargs)


@pytest.mark.parametrize("selection_kind", ["ids", "addresses", "indices"])
def test_read_blobs_preserves_input_order(dataset_with_blobs, selection_kind):
    selection_values, expected = _out_of_order_blob_selection(
        dataset_with_blobs, selection_kind
    )
    kwargs = {selection_kind: selection_values}

    blobs = dataset_with_blobs.read_blobs("blobs", **kwargs, preserve_order=True)

    assert blobs == expected


@pytest.mark.parametrize("selection_kind", ["ids", "addresses", "indices"])
def test_read_blobs_without_preserve_order_returns_same_rows(
    dataset_with_blobs, selection_kind
):
    selection_values, expected = _out_of_order_blob_selection(
        dataset_with_blobs, selection_kind
    )
    kwargs = {selection_kind: selection_values}

    blobs = dataset_with_blobs.read_blobs("blobs", **kwargs, preserve_order=False)

    assert sorted(blobs) == sorted(expected)


@pytest.mark.parametrize("selection_kind", ["ids", "addresses", "indices"])
@pytest.mark.parametrize(
    "selected_indices",
    [
        [0, 5],
        [0, 5, 4],
        [5],
        [0, 5, 1],
        [4, 4, 5],
    ],
)
def test_blob_selection_apis_preserve_nulls_and_empty_values(
    dataset_with_mixed_blob_v2, selection_kind, selected_indices
):
    dataset, payloads = dataset_with_mixed_blob_v2
    if selection_kind == "ids":
        all_selectors = _blob_row_ids(dataset)
    elif selection_kind == "addresses":
        all_selectors = _blob_row_addresses(dataset)
    else:
        all_selectors = list(range(len(payloads)))
    selectors = [all_selectors[index] for index in selected_indices]
    kwargs = {selection_kind: selectors}
    expected = [payloads[index] for index in selected_indices]

    blob_files = dataset.take_blobs("blobs", **kwargs)
    assert [None if blob is None else blob.readall() for blob in blob_files] == expected

    blobs = dataset.read_blobs("blobs", **kwargs, preserve_order=True)
    assert [data for _, data in blobs] == expected


@pytest.mark.parametrize("selection_kind", ["ids", "addresses", "indices"])
def test_read_blob_ranges_mixed_sources_preserves_request_identity(
    dataset_with_mixed_blob_v2, selection_kind
):
    dataset, payloads = dataset_with_mixed_blob_v2
    selected_indices = [3, 1, 1, 0, 2, 4, 5]
    ranges = [(1, 3), (2, 4), (0, 4), (1, 1), (5, 0), (0, 0), (0, 0)]
    addresses = _blob_row_addresses(dataset)
    if selection_kind == "ids":
        all_selectors = _blob_row_ids(dataset)
    elif selection_kind == "addresses":
        all_selectors = addresses
    else:
        all_selectors = list(range(len(payloads)))
    selectors = [all_selectors[index] for index in selected_indices]
    requests = [
        (row, offset, length) for row, (offset, length) in zip(selectors, ranges)
    ]

    results = dataset.read_blob_ranges(
        "blobs",
        requests,
        selector=selection_kind,
        io_buffer_size=1024,
        preserve_order=True,
    )

    expected = []
    for request_index, (row_index, (offset, length)) in enumerate(
        zip(selected_indices, ranges)
    ):
        payload = payloads[row_index]
        expected.append(
            (
                request_index,
                addresses[row_index],
                None if payload is None else payload[offset : offset + length],
            )
        )
    assert results == expected


def test_blob_selection_apis_preserve_blob_v1_nulls_and_empty_values(tmp_path):
    schema = pa.schema(
        [
            pa.field(
                "blobs",
                pa.large_binary(),
                metadata={"lance-encoding:blob": "true"},
            ),
            pa.field("idx", pa.uint64()),
        ]
    )
    dataset = lance.write_dataset(
        pa.table({"blobs": [None, b"", b"abc"], "idx": range(3)}, schema=schema),
        tmp_path / "blob_v1_nulls",
        data_storage_version="2.0",
    )
    addresses = _blob_row_addresses(dataset)

    blob_files = dataset.take_blobs("blobs", indices=[0, 1, 2])
    assert [None if blob is None else blob.readall() for blob in blob_files] == [
        None,
        b"",
        b"abc",
    ]

    blobs = dataset.read_blobs("blobs", indices=[0, 1, 2])
    assert [data for _, data in blobs] == [None, b"", b"abc"]

    results = dataset.read_blob_ranges(
        "blobs",
        [(0, 0, 0), (0, 0, 1), (1, 0, 0), (2, 1, 1)],
        selector="indices",
    )

    assert results == [
        (0, addresses[0], None),
        (1, addresses[0], None),
        (2, addresses[1], b""),
        (3, addresses[2], b"b"),
    ]


def test_read_blob_ranges_without_preserve_order_keeps_request_identity(
    dataset_with_mixed_blob_v2,
):
    dataset, _ = dataset_with_mixed_blob_v2
    results = dataset.read_blob_ranges(
        "blobs",
        [(3, 0, 1), (4, 0, 1), (1, 1, 2), (0, 0, 2)],
        selector="indices",
        preserve_order=False,
    )

    assert sorted(results) == sorted(
        [
            (0, _blob_row_addresses(dataset)[3], b"e"),
            (1, _blob_row_addresses(dataset)[4], None),
            (2, _blob_row_addresses(dataset)[1], b"ac"),
            (3, _blob_row_addresses(dataset)[0], b"in"),
        ]
    )


@pytest.mark.parametrize(
    ("requests", "message"),
    [
        ([(0, 2**64 - 1, 2)], "offset \\+ length overflowed"),
        ([(0, 0, 4)], "exceeds blob size"),
    ],
)
def test_read_blob_ranges_rejects_invalid_ranges(
    dataset_with_mixed_blob_v2, requests, message
):
    dataset, _ = dataset_with_mixed_blob_v2

    with pytest.raises(ValueError, match=message):
        dataset.read_blob_ranges("blobs", requests, selector="indices")


def test_read_blob_ranges_rejects_invalid_selector(dataset_with_mixed_blob_v2):
    dataset, _ = dataset_with_mixed_blob_v2

    with pytest.raises(ValueError, match="selector must be one of"):
        dataset.read_blob_ranges("blobs", [], selector="offsets")


def test_read_blob_ranges_accepts_empty_batch(dataset_with_mixed_blob_v2):
    dataset, _ = dataset_with_mixed_blob_v2

    assert dataset.read_blob_ranges("blobs", [], selector="indices") == []


def test_blob_file_seek(tmp_path, dataset_with_blobs):
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)
    with blobs[1] as f:
        assert f.seek(1) == 1
        assert f.read(1) == b"a"
        assert f.seek(-1, io.SEEK_CUR) == 1
        assert f.seek(-1, io.SEEK_END) == 2


@pytest.mark.parametrize(
    "whence",
    [
        pytest.param(io.SEEK_SET, id="set"),
        pytest.param(io.SEEK_CUR, id="cur"),
        pytest.param(io.SEEK_END, id="end"),
    ],
)
def test_blob_file_negative_seek_raises_value_error(dataset_with_blobs, whence):
    blob = dataset_with_blobs.take_blobs("blobs", indices=[1])[0]
    offset = -(blob.size() + 1) if whence == io.SEEK_END else -1
    with pytest.raises(ValueError, match="negative seek value -1"):
        blob.seek(offset, whence)
    assert blob.tell() == 0


def test_blob_file_negative_seek_does_not_move_cursor(dataset_with_blobs):
    blob = dataset_with_blobs.take_blobs("blobs", indices=[1])[0]
    blob.seek(2)
    with pytest.raises(ValueError, match="negative seek value -1"):
        blob.seek(-1)
    assert blob.tell() == 2


def test_blob_file_read_past_eof_leaves_cursor(dataset_with_blobs):
    with dataset_with_blobs.take_blobs("blobs", indices=[1])[0] as blob:
        past_eof = blob.size() + 1
        assert blob.seek(past_eof) == past_eof
        assert blob.read() == b""
        assert blob.tell() == past_eof


@pytest.mark.parametrize(
    "ranges",
    [
        pytest.param([], id="empty_list"),
        pytest.param([(0, 0), (4, 0)], id="empty_ranges"),
        pytest.param([(2, 1), (0, 1), (1, 1)], id="non_monotonic"),
        pytest.param([(1, 2), (0, 4), (1, 2), (0, 0), (2, 2)], id="dup_overlap"),
    ],
)
def test_blob_file_read_ranges_matches_read_range(dataset_with_blobs, ranges):
    row_ids = _blob_row_ids(dataset_with_blobs)
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)
    with blobs[4] as f:
        expected = [f.read_range(offset, length) for offset, length in ranges]
        assert f.read_ranges(ranges) == expected


def test_blob_file_read_ranges_preserves_input_order(dataset_with_blobs):
    row_ids = _blob_row_ids(dataset_with_blobs)
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)
    with blobs[1] as f:
        assert f.read_ranges([(2, 1), (0, 1), (1, 2)]) == [b"r", b"b", b"ar"]


def test_blob_file_read_ranges_does_not_change_cursor(dataset_with_blobs):
    row_ids = _blob_row_ids(dataset_with_blobs)
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)
    with blobs[1] as f:
        assert f.tell() == 0
        assert f.read_ranges([(0, 3)]) == [b"bar"]
        assert f.tell() == 0
        f.seek(2)
        assert f.read_ranges([(2, 1), (0, 3)]) == [b"r", b"bar"]
        assert f.tell() == 2
        assert f.read(1) == b"r"


@pytest.mark.parametrize(
    ("ranges", "message"),
    [
        pytest.param([(2**64 - 1, 2)], "offset \\+ length overflowed", id="overflow"),
        pytest.param([(0, 1), (1, 100)], "exceeds blob size", id="out_of_bounds"),
    ],
)
def test_blob_file_read_ranges_rejects_invalid_ranges(
    dataset_with_blobs, ranges, message
):
    row_ids = _blob_row_ids(dataset_with_blobs)
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)
    with blobs[0] as f:
        with pytest.raises(ValueError, match=message):
            f.read_ranges(ranges)
        assert f.tell() == 0


def test_null_blobs(tmp_path):
    table = pa.table(
        {
            "id": range(100),
            "blob": pa.array([None] * 100, pa.large_binary()),
        },
        schema=pa.schema(
            [
                pa.field("id", pa.uint64()),
                pa.field(
                    "blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                ),
            ]
        ),
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds",
        data_storage_version=LEGACY_BLOB_STORAGE_VERSION,
    )

    blobs = ds.take_blobs("blob", ids=range(100))
    assert blobs == [None] * 100

    ds.insert(pa.table({"id": pa.array(range(100, 200), pa.uint64())}))

    ds.add_columns(
        pa.field(
            "more_blob",
            pa.large_binary(),
            metadata={"lance-encoding:blob": "true"},
        )
    )

    for blob_col in ["blob", "more_blob"]:
        blobs = ds.take_blobs(blob_col, indices=range(100, 200))
        assert blobs == [None] * 100

        blobs = ds.to_table(columns=[blob_col])
        for blob in blobs.column(blob_col):
            py_blob = blob.as_py()
            # When we write blobs to a file we store the position as 1 and size as 0
            # to avoid needing a validity buffer.
            assert py_blob is None or py_blob == {
                "position": 1,
                "size": 0,
            }


def test_blob_file_read_middle(tmp_path, dataset_with_blobs):
    # This regresses an issue where we were not setting the cursor
    # correctly after a call to `read` when the blob was not the
    # first thing in the file.
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    blobs = dataset_with_blobs.take_blobs("blobs", ids=row_ids)
    with blobs[1] as f:
        assert f.read(1) == b"b"
        assert f.read(1) == b"a"
        assert f.read(1) == b"r"


def test_blob_file_buffer_size_configures_inner():
    inner = _CountingBlobInner(b"x")
    blob = BlobFile(inner, buffer_size=64 * 1024)

    assert isinstance(blob, io.BufferedIOBase)
    assert inner.buffer_size == 64 * 1024


def test_blob_file_buffer_size_zero_configures_inner():
    inner = _CountingBlobInner(b"x")
    BlobFile(inner, buffer_size=0)
    assert inner.buffer_size == 0


def test_blob_file_read_range_does_not_issue_sequential_read():
    payload = bytes(range(256))
    inner = _CountingBlobInner(payload)
    blob = BlobFile(inner, buffer_size=64)

    assert blob.read_range(1, 3) == payload[1:4]
    assert blob.tell() == 0
    assert inner.read_sizes == []
    assert blob.read(2) == payload[:2]


def test_blob_file_read_range_does_not_move_cursor():
    payload = bytes(range(256)) * 4
    inner = _CountingBlobInner(payload)
    blob = BlobFile(inner, buffer_size=64)

    assert blob.read(8) == payload[:8]
    assert blob.tell() == 8
    assert blob.read_range(20, 5) == payload[20:25]
    assert blob.tell() == 8
    assert blob.read(4) == payload[8:12]


@pytest.mark.parametrize("buffer_size", [32, 0])
def test_blob_file_text_wrapper_readline(buffer_size):
    payload = b"hello\nworld\n"
    blob = BlobFile(_CountingBlobInner(payload), buffer_size=buffer_size)
    wrapper = io.TextIOWrapper(blob, encoding="utf-8")

    assert wrapper.readline() == "hello\n"
    assert wrapper.readline() == "world\n"


def test_take_blobs_sequential_small_reads_reuse_prefetch(tmp_path):
    payload = bytes(range(256)) * 160
    ds = lance.write_dataset(
        _complete_blob_table([0], [payload]),
        tmp_path / "seq_prefetch",
        data_storage_version="2.2",
    )
    blob = ds.take_blobs("blob", indices=[0], buffer_size=32 * 1024)[0]

    assert _read_in_8k_chunks(blob) == payload
    assert blob.inner._range_submission_count() == 2


def test_take_blobs_buffer_size_zero_fetches_each_sequential_read(tmp_path):
    payload = b"x" * (24 * 1024)
    ds = lance.write_dataset(
        _complete_blob_table([0], [payload]),
        tmp_path / "seq_unbuffered",
        data_storage_version="2.2",
    )
    blob = ds.take_blobs("blob", indices=[0], buffer_size=0)[0]

    assert _read_in_8k_chunks(blob) == payload
    assert blob.inner._range_submission_count() == 3


def test_take_blobs_seek_inside_prefetch_does_not_refetch(tmp_path):
    payload = bytes((i * 31) % 256 for i in range(40 * 1024))
    ds = lance.write_dataset(
        _complete_blob_table([0], [payload]),
        tmp_path / "seek_inside",
        data_storage_version="2.2",
    )
    blob = ds.take_blobs("blob", indices=[0], buffer_size=32 * 1024)[0]

    assert blob.read(100) == payload[:100]
    after_fill = blob.inner._range_submission_count()
    blob.seek(1000)
    assert blob.read(100) == payload[1000:1100]
    assert blob.inner._range_submission_count() == after_fill


def test_take_blobs_seek_outside_prefetch_issues_new_range_submission(tmp_path):
    payload = random.Random(0).randbytes(40 * 1024)
    ds = lance.write_dataset(
        _complete_blob_table([0], [payload]),
        tmp_path / "seek_outside",
        data_storage_version="2.2",
    )
    blob = ds.take_blobs("blob", indices=[0], buffer_size=32 * 1024)[0]

    assert blob.read(100) == payload[:100]
    after_fill = blob.inner._range_submission_count()
    blob.seek(33 * 1024)
    assert blob.read(100) == payload[33 * 1024 : 33 * 1024 + 100]
    assert blob.inner._range_submission_count() == after_fill + 1


def test_take_blobs_read_fills_across_buffer_boundary(tmp_path):
    payload = bytes(range(40))
    ds = lance.write_dataset(
        _complete_blob_table([0], [payload]),
        tmp_path / "read_boundary",
        data_storage_version="2.2",
    )
    blob = ds.take_blobs("blob", indices=[0], buffer_size=16)[0]
    assert blob.read(4) == payload[:4]
    assert blob.read(20) == payload[4:24]
    assert blob.tell() == 24

    blob = ds.take_blobs("blob", indices=[0], buffer_size=16)[0]
    assert blob.read(4) == payload[:4]
    buf = bytearray(20)
    assert blob.readinto(buf) == 20
    assert bytes(buf) == payload[4:24]

    blob = ds.take_blobs("blob", indices=[0], buffer_size=16)[0]
    assert blob.read(4) == payload[:4]
    assert blob.read1(20) == payload[4:24]
    assert blob.tell() == 24


def test_take_blobs_text_wrapper_readline(tmp_path):
    payload = b"hello\nworld\n"
    ds = lance.write_dataset(
        _complete_blob_table([0], [payload]),
        tmp_path / "text_wrapper",
        data_storage_version="2.2",
    )
    blob = ds.take_blobs("blob", indices=[0])[0]
    wrapper = io.TextIOWrapper(blob, encoding="utf-8")

    assert wrapper.readline() == "hello\n"
    assert wrapper.readline() == "world\n"


def test_blob_file_readinto_readonly_does_not_advance_cursor():
    inner = _CountingBlobInner(b"abcdef")
    blob = BlobFile(inner, buffer_size=0)

    with pytest.raises(TypeError, match="read-write"):
        blob.readinto(b"xx")
    assert blob.tell() == 0
    assert inner.read_sizes == []


def test_blob_file_empty_read_after_close_raises():
    blob = BlobFile(_CountingBlobInner(b"abcdef"), buffer_size=0)
    blob.close()

    with pytest.raises(ValueError, match="read of closed file"):
        blob.read(0)
    with pytest.raises(ValueError, match="readinto of closed file"):
        blob.readinto(bytearray())


@pytest.mark.parametrize(
    ("buffer_size", "match"),
    [
        pytest.param(-1, "non-negative", id="negative"),
        pytest.param(True, "must be an int", id="bool"),
        pytest.param(1.5, "must be an int", id="float"),
    ],
)
def test_blob_file_rejects_invalid_buffer_size(buffer_size, match):
    with pytest.raises((TypeError, ValueError), match=match):
        BlobFile(_CountingBlobInner(b"x"), buffer_size=buffer_size)


def test_take_blobs_buffer_size_zero_reads_payload(dataset_with_blobs):
    blob = dataset_with_blobs.take_blobs("blobs", indices=[1], buffer_size=0)[0]
    assert isinstance(blob, BlobFile)
    assert blob.read() == b"bar"


def test_take_blobs_rejects_invalid_buffer_size_for_empty_selection(dataset_with_blobs):
    with pytest.raises(ValueError, match="non-negative"):
        dataset_with_blobs.take_blobs("blobs", indices=[], buffer_size=-1)


def test_blob_file_zipfile_reads_payload(tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("hello.txt", "hello blob")
    payload = archive.getvalue()

    table = pa.table(
        {"blob": lance.blob_array([payload])},
        schema=pa.schema([lance.blob_field("blob")]),
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "zip_blob",
        data_storage_version="2.2",
    )
    blob = ds.take_blobs("blob", indices=[0])[0]
    with zipfile.ZipFile(blob) as zf:
        assert zf.read("hello.txt") == b"hello blob"


def test_take_deleted_blob(tmp_path, dataset_with_blobs):
    row_ids = (
        dataset_with_blobs.to_table(columns=[], with_row_id=True)
        .column("_rowid")
        .to_pylist()
    )
    dataset_with_blobs.delete("idx = 1")

    with pytest.raises(
        NotImplementedError,
        match="A take operation that includes row addresses must not target deleted",
    ):
        dataset_with_blobs.take_blobs("blobs", ids=row_ids)


def test_scan_blob(tmp_path, dataset_with_blobs):
    ds = dataset_with_blobs.scanner(filter="idx = 2").to_table()
    assert ds.num_rows == 1


def test_blob_extension_write_inline(tmp_path):
    table = pa.table({"blob": lance.blob_array([b"foo", b"bar"])})
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds_v2",
        data_storage_version="2.2",
    )

    desc = ds.to_table(columns=["blob"]).column("blob").chunk(0)
    assert pa.types.is_struct(desc.type)

    blobs = ds.take_blobs("blob", indices=[0, 1])
    with blobs[0] as f:
        assert f.read() == b"foo"


def test_complete_blob_schema_survives_create(tmp_path):
    dataset_path = tmp_path / "complete_blob_create"
    ds = lance.write_dataset(
        _complete_blob_table([0], [b"created"]),
        dataset_path,
        data_storage_version="2.2",
    )
    _assert_complete_blob_schema(ds)
    assert ds.to_table(blob_handling="all_binary")["blob"].to_pylist() == [b"created"]


def test_complete_blob_schema_survives_append(tmp_path):
    dataset_path = tmp_path / "complete_blob_append"
    lance.write_dataset(
        _complete_blob_table([0], [b"initial"]),
        dataset_path,
        data_storage_version="2.2",
    )

    lance.write_dataset(
        _complete_blob_table([1], [b"appended"]), dataset_path, mode="append"
    )
    ds = lance.dataset(dataset_path)
    _assert_complete_blob_schema(ds)
    result = ds.to_table(blob_handling="all_binary").sort_by("id")
    assert result["id"].to_pylist() == [0, 1]
    assert result["blob"].to_pylist() == [b"initial", b"appended"]


def test_complete_blob_schema_survives_merge_insert(tmp_path):
    dataset_path = tmp_path / "complete_blob_merge_insert"
    ds = lance.write_dataset(
        _complete_blob_table([0, 1], [b"zero", b"initial"]),
        dataset_path,
        data_storage_version="2.2",
    )

    (
        ds.merge_insert("id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(_complete_blob_table([1, 2], [b"updated", b"inserted"]))
    )
    ds = lance.dataset(dataset_path)
    _assert_complete_blob_schema(ds)

    result = ds.to_table(blob_handling="all_binary").sort_by("id")
    assert result["id"].to_pylist() == [0, 1, 2]
    assert result["blob"].to_pylist() == [b"zero", b"updated", b"inserted"]


@pytest.mark.parametrize(
    ("use_uri", "position", "size"),
    [
        pytest.param(True, 3, None, id="missing-size"),
        pytest.param(False, 3, 2, id="range-without-uri"),
    ],
)
def test_complete_blob_rows_reject_invalid_ranges(tmp_path, use_uri, position, size):
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789")
    storage = pa.StructArray.from_arrays(
        [
            pa.array([None], type=pa.large_binary()),
            pa.array([source.as_uri() if use_uri else None], type=pa.utf8()),
            pa.array([position], type=pa.uint64()),
            pa.array([size], type=pa.uint64()),
        ],
        names=["data", "uri", "position", "size"],
    )
    blobs = pa.ExtensionArray.from_storage(BlobType(), storage)
    table = pa.Table.from_arrays([blobs], schema=pa.schema([lance.blob_field("blob")]))

    with pytest.raises(OSError, match="position|size|uri"):
        lance.write_dataset(
            table,
            tmp_path / "dataset",
            data_storage_version="2.2",
            allow_external_blob_outside_bases=True,
        )


@pytest.mark.parametrize("mode", ["reference", "ingest"])
def test_complete_blob_rows_reject_zero_size_range(tmp_path, mode):
    source = tmp_path / "source.bin"
    source.write_bytes(b"0123456789")
    table = _complete_blob_storage_table(None, source.as_uri(), 3, 0)

    with pytest.raises(OSError, match="greater than zero"):
        lance.write_dataset(
            table,
            tmp_path / "dataset",
            data_storage_version="2.2",
            external_blob_mode=mode,
            allow_external_blob_outside_bases=mode == "reference",
        )


@pytest.mark.parametrize(
    ("data", "use_uri"),
    [
        pytest.param(b"small", True, id="both-small"),
        pytest.param(b"x" * 70_000, True, id="both-packed"),
        pytest.param(None, False, id="neither"),
    ],
)
def test_complete_blob_rows_reject_invalid_representation(tmp_path, data, use_uri):
    source = tmp_path / "source.bin"
    source.write_bytes(b"external")
    table = _complete_blob_storage_table(
        data, source.as_uri() if use_uri else None, None, None
    )

    with pytest.raises(OSError, match="data|uri"):
        lance.write_dataset(
            table,
            tmp_path / "dataset",
            data_storage_version="2.2",
            allow_external_blob_outside_bases=True,
        )


@pytest.mark.parametrize("mode", ["reference", "ingest"])
def test_empty_external_object_without_range(tmp_path, mode):
    source = tmp_path / "empty.bin"
    source.write_bytes(b"")

    dataset = lance.write_dataset(
        pa.table({"blob": lance.blob_array([source.as_uri()])}),
        tmp_path / "dataset",
        data_storage_version="2.2",
        external_blob_mode=mode,
        allow_external_blob_outside_bases=mode == "reference",
    )

    with dataset.take_blobs("blob", indices=[0])[0] as blob:
        assert blob.read() == b""


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({}, "must set `data` or `uri`", id="neither"),
        pytest.param(
            {"data": b"data", "uri": "file:///source.bin"},
            "both data and uri",
            id="both",
        ),
        pytest.param(
            {"uri": "file:///source.bin", "position": 3, "size": 0},
            "greater than zero",
            id="zero-size",
        ),
    ],
)
def test_blob_rejects_invalid_logical_value(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Blob(**kwargs)


@pytest.mark.parametrize(
    "storage_type",
    [
        pytest.param(
            pa.struct(
                [
                    pa.field("data", pa.large_binary()),
                    pa.field("uri", pa.utf8()),
                ]
            ),
            id="minimal",
        ),
        pytest.param(BlobType().storage_type, id="complete-nullable-range"),
        pytest.param(
            pa.struct(
                [
                    pa.field(
                        "data",
                        pa.large_binary(),
                        metadata={b"source": b"preserved"},
                    ),
                    pa.field("uri", pa.utf8()),
                    pa.field("position", pa.uint64(), nullable=False),
                    pa.field("size", pa.uint64(), nullable=False),
                ]
            ),
            id="complete-required-range-with-metadata",
        ),
    ],
)
def test_blob_type_preserves_storage_type_through_arrow_ipc(storage_type):
    blob_type = BlobType.__arrow_ext_deserialize__(storage_type, b"")
    schema = pa.schema([pa.field("blob", blob_type)])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, schema):
        pass

    restored_type = pa.ipc.open_stream(sink.getvalue()).schema.field("blob").type

    assert isinstance(restored_type, BlobType)
    _assert_blob_storage_type_equal(restored_type.storage_type, storage_type)


@pytest.mark.parametrize(
    "storage_type",
    [
        pytest.param(pa.int32(), id="not-struct"),
        pytest.param(
            pa.struct(
                [
                    pa.field("data", pa.large_binary()),
                    pa.field("uri", pa.utf8()),
                    pa.field("position", pa.uint64()),
                ]
            ),
            id="incomplete-range-shape",
        ),
        pytest.param(
            pa.struct(
                [
                    pa.field("data", pa.large_binary(), nullable=False),
                    pa.field("uri", pa.utf8()),
                ]
            ),
            id="required-data",
        ),
        pytest.param(
            pa.struct(
                [
                    pa.field("data", pa.large_binary()),
                    pa.field("uri", pa.utf8()),
                    pa.field("position", pa.uint64()),
                    pa.field("size", pa.int64()),
                ]
            ),
            id="wrong-range-type",
        ),
    ],
)
def test_blob_type_rejects_invalid_storage_type(storage_type):
    with pytest.raises(TypeError, match="BlobType storage"):
        BlobType.__arrow_ext_deserialize__(storage_type, b"")


@pytest.mark.parametrize(
    ("initial_storage_type", "append_storage_type"),
    [
        pytest.param(
            pa.struct(
                [
                    pa.field("data", pa.large_binary()),
                    pa.field("uri", pa.utf8()),
                ]
            ),
            BlobType().storage_type,
            id="complete-to-minimal",
        ),
        pytest.param(
            BlobType().storage_type,
            pa.struct(
                [
                    pa.field("data", pa.large_binary()),
                    pa.field("uri", pa.utf8()),
                ]
            ),
            id="minimal-to-complete",
        ),
    ],
)
def test_blob_v2_append_accepts_mixed_logical_shapes(
    tmp_path, initial_storage_type, append_storage_type
):
    dataset_path = tmp_path / "mixed_blob_shapes"
    initial = pa.Table.from_arrays(
        [_inline_blob_array(initial_storage_type, b"initial")], names=["blob"]
    )
    append = pa.Table.from_arrays(
        [_inline_blob_array(append_storage_type, b"appended")], names=["blob"]
    )

    lance.write_dataset(initial, dataset_path, data_storage_version="2.2")
    lance.write_dataset(append, dataset_path, mode="append")
    dataset = lance.dataset(dataset_path)

    _assert_blob_storage_type_equal(
        dataset.schema.field("blob").type.storage_type,
        initial_storage_type,
    )
    blobs = dataset.take_blobs("blob", indices=[0, 1])
    assert [blob.readall() for blob in blobs] == [b"initial", b"appended"]


def test_blob_field_threshold_metadata():
    field = lance.blob_field(
        "blob",
        inline_size_threshold=16 * 1024,
        dedicated_size_threshold=2 * 1024 * 1024,
        pack_file_size_threshold=512 * 1024 * 1024,
    )

    assert field.metadata[b"lance-encoding:blob-inline-size-threshold"] == b"16384"
    assert field.metadata[b"lance-encoding:blob-dedicated-size-threshold"] == b"2097152"
    assert (
        field.metadata[b"lance-encoding:blob-pack-file-size-threshold"] == b"536870912"
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        pytest.param(
            {"inline_size_threshold": -1},
            ValueError,
            "inline_size_threshold must be non-negative",
            id="negative_inline",
        ),
        pytest.param(
            {"dedicated_size_threshold": 0},
            ValueError,
            "dedicated_size_threshold must be positive",
            id="zero_dedicated",
        ),
        pytest.param(
            {"dedicated_size_threshold": -1},
            ValueError,
            "dedicated_size_threshold must be positive",
            id="negative_dedicated",
        ),
        pytest.param(
            {"inline_size_threshold": True},
            TypeError,
            "inline_size_threshold must be an int",
            id="bool_inline",
        ),
        pytest.param(
            {"dedicated_size_threshold": True},
            TypeError,
            "dedicated_size_threshold must be an int",
            id="bool_dedicated",
        ),
        pytest.param(
            {"inline_size_threshold": 1.5},
            TypeError,
            "inline_size_threshold must be an int",
            id="float_inline",
        ),
        pytest.param(
            {"inline_size_threshold": 2**100},
            OverflowError,
            "inline_size_threshold must fit in a Rust usize",
            id="overflow_inline",
        ),
        pytest.param(
            {"dedicated_size_threshold": 2**100},
            OverflowError,
            "dedicated_size_threshold must fit in a Rust usize",
            id="overflow_dedicated",
        ),
        pytest.param(
            {"pack_file_size_threshold": 0},
            ValueError,
            "pack_file_size_threshold must be positive",
            id="zero_pack_file",
        ),
        pytest.param(
            {"pack_file_size_threshold": -1},
            ValueError,
            "pack_file_size_threshold must be positive",
            id="negative_pack_file",
        ),
        pytest.param(
            {"pack_file_size_threshold": True},
            TypeError,
            "pack_file_size_threshold must be an int",
            id="bool_pack_file",
        ),
        pytest.param(
            {"pack_file_size_threshold": 2**100},
            OverflowError,
            "pack_file_size_threshold must fit in a Rust usize",
            id="overflow_pack_file",
        ),
    ],
)
def test_blob_field_rejects_invalid_thresholds(kwargs, error, message):
    with pytest.raises(error, match=message):
        lance.blob_field("blob", **kwargs)


def test_blob_extension_inline_threshold_per_column(tmp_path):
    payload = b"x" * 2048
    schema = pa.schema(
        [
            lance.blob_field("inline_blob", inline_size_threshold=4096),
            lance.blob_field("packed_blob", inline_size_threshold=1024),
        ]
    )
    table = pa.table(
        {
            "inline_blob": lance.blob_array([payload]),
            "packed_blob": lance.blob_array([payload]),
        },
        schema=schema,
    )
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds_v2_inline_threshold_per_column",
        data_storage_version="2.2",
    )

    desc = ds.to_table(columns=["inline_blob", "packed_blob"])
    assert desc.column("inline_blob").chunk(0).field("kind").to_pylist() == [0]
    assert desc.column("packed_blob").chunk(0).field("kind").to_pylist() == [1]


def test_blob_extension_threshold_metadata_persists_after_reopen(tmp_path):
    dataset_path = tmp_path / "test_ds_v2_threshold_metadata_persists"
    schema = pa.schema([lance.blob_field("blob", inline_size_threshold=1024)])
    table = pa.table({"blob": lance.blob_array([b"x"])}, schema=schema)

    lance.write_dataset(table, dataset_path, data_storage_version="2.2")
    reopened = lance.dataset(dataset_path)

    assert (
        reopened.schema.field("blob").metadata[
            b"lance-encoding:blob-inline-size-threshold"
        ]
        == b"1024"
    )


def test_blob_extension_append_rejects_explicit_threshold_mismatch(tmp_path):
    dataset_path = tmp_path / "test_ds_v2_append_threshold_mismatch"
    initial_schema = pa.schema([lance.blob_field("blob", inline_size_threshold=4096)])
    initial = pa.table(
        {"blob": lance.blob_array([b"x" * 2048])},
        schema=initial_schema,
    )
    lance.write_dataset(initial, dataset_path, data_storage_version="2.2")

    append_schema = pa.schema([lance.blob_field("blob", inline_size_threshold=1024)])
    append = pa.table(
        {"blob": lance.blob_array([b"x" * 2048])},
        schema=append_schema,
    )

    with pytest.raises(
        OSError, match="Cannot append data with blob threshold metadata"
    ):
        lance.write_dataset(append, dataset_path, mode="append")


def test_blob_extension_pack_file_threshold_metadata_persists_after_reopen(
    tmp_path: Path,
):
    dataset_path = tmp_path / "test_ds_v2_pack_file_threshold_persists"
    threshold = 512 * 1024 * 1024
    schema = pa.schema([lance.blob_field("blob", pack_file_size_threshold=threshold)])
    table = pa.table({"blob": lance.blob_array([b"x"])}, schema=schema)

    lance.write_dataset(table, dataset_path, data_storage_version="2.2")
    reopened = lance.dataset(dataset_path)

    assert (
        reopened.schema.field("blob").metadata[
            b"lance-encoding:blob-pack-file-size-threshold"
        ]
        == str(threshold).encode()
    )


def test_blob_extension_append_rejects_pack_file_threshold_mismatch(tmp_path: Path):
    dataset_path = tmp_path / "test_ds_v2_append_pack_file_mismatch"
    initial_schema = pa.schema(
        [lance.blob_field("blob", pack_file_size_threshold=512 * 1024 * 1024)]
    )
    initial = pa.table(
        {"blob": lance.blob_array([b"x" * 2048])},
        schema=initial_schema,
    )
    lance.write_dataset(initial, dataset_path, data_storage_version="2.2")

    append_schema = pa.schema(
        [lance.blob_field("blob", pack_file_size_threshold=256 * 1024 * 1024)]
    )
    append = pa.table(
        {"blob": lance.blob_array([b"x" * 2048])},
        schema=append_schema,
    )

    with pytest.raises(
        OSError, match="Cannot append data with blob threshold metadata"
    ):
        lance.write_dataset(append, dataset_path, mode="append")


def test_blob_extension_dedicated_threshold_precedes_inline_threshold(tmp_path):
    payload = b"x" * 2048
    schema = pa.schema(
        [
            lance.blob_field(
                "blob",
                inline_size_threshold=4096,
                dedicated_size_threshold=1024,
            )
        ]
    )
    table = pa.table({"blob": lance.blob_array([payload])}, schema=schema)
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds_v2_dedicated_precedes_inline",
        data_storage_version="2.2",
    )

    desc = ds.to_table(columns=["blob"]).column("blob").chunk(0)
    assert desc.field("kind").to_pylist() == [2]


def test_blob_extension_write_external(tmp_path):
    blob_path = tmp_path / "external_blob.bin"
    blob_path.write_bytes(b"hello")
    uri = blob_path.as_uri()

    table = pa.table({"blob": lance.blob_array([uri])})
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds_v2_external",
        data_storage_version="2.2",
        allow_external_blob_outside_bases=True,
    )

    blob = ds.take_blobs("blob", indices=[0])[0]
    assert blob.size() == 5
    with blob as f:
        assert f.read() == b"hello"


@pytest.mark.parametrize(
    ("position", "size"),
    [
        pytest.param(None, None, id="explicit_none"),
        pytest.param(1, 3, id="slice"),
    ],
)
def test_blob_from_uri_accepts_optional_slice_metadata(position, size):
    blob = Blob.from_uri("file:///tmp/blob.bin", position=position, size=size)

    assert blob.uri == "file:///tmp/blob.bin"
    assert blob.position == position
    assert blob.size == size


def test_blob_extension_write_external_ingest(tmp_path):
    blob_path = tmp_path / "external_blob.bin"
    blob_path.write_bytes(b"hello")
    uri = blob_path.as_uri()

    table = pa.table({"blob": lance.blob_array([uri])})
    ds = lance.write_dataset(
        table,
        tmp_path / "test_ds_v2_external_ingest",
        data_storage_version="2.2",
        external_blob_mode="ingest",
    )

    blob_path.unlink()

    blob = ds.take_blobs("blob", indices=[0])[0]
    assert blob.size() == 5
    with blob as f:
        assert f.read() == b"hello"


def test_blob_extension_write_external_ingest_rejects_reference_only_options(tmp_path):
    blob_path = tmp_path / "external_blob.bin"
    blob_path.write_bytes(b"hello")
    uri = blob_path.as_uri()
    message = (
        "allow_external_blob_outside_bases only applies when "
        'external_blob_mode="reference"'
    )

    table = pa.table({"blob": lance.blob_array([uri])})
    with pytest.raises(OSError, match=message):
        lance.write_dataset(
            table,
            tmp_path / "test_ds_v2_external_ingest_invalid",
            data_storage_version="2.2",
            external_blob_mode="ingest",
            allow_external_blob_outside_bases=True,
        )


def test_blob_extension_add_columns_record_batch_reader_all_kinds(tmp_path):
    values, payloads, initial_bases = _add_columns_blob_v2_values(tmp_path)
    ds = lance.write_dataset(
        pa.table({"id": range(4)}),
        tmp_path / "test_add_columns_reader_blob_v2",
        data_storage_version="2.2",
        initial_bases=initial_bases,
    )

    ds.add_columns(pa.table({"blob": lance.blob_array(values)}).to_reader())

    _assert_blob_v2_add_columns_result(ds, "blob", payloads)


@pytest.mark.parametrize(
    "failure_mode",
    [
        pytest.param("raises_after_first_fragment", id="reader_raises_mid_stream"),
        pytest.param("wrong_schema", id="reader_yields_wrong_schema"),
        pytest.param("too_many_rows", id="reader_produces_too_many_rows"),
    ],
)
def test_blob_extension_add_columns_record_batch_reader_failure_cleans_files(
    tmp_path,
    failure_mode,
):
    ds, dataset_path, values, payloads = _write_two_fragment_blob_v2_seed_dataset(
        tmp_path,
        f"test_add_columns_reader_blob_v2_fail_cleanup_{failure_mode}",
    )
    external_blob_path = tmp_path / "external_base" / "external_blob.bin"
    files_before = _dataset_file_set(dataset_path)

    schema = pa.schema([lance.blob_field("blob")])
    first_fragment_batch = pa.record_batch([lance.blob_array(values)], schema=schema)
    second_fragment_batch = pa.record_batch([lance.blob_array(values)], schema=schema)

    if failure_mode == "raises_after_first_fragment":
        match = "reader failed after first fragment"

        def failing_reader():
            yield first_fragment_batch
            raise RuntimeError("reader failed after first fragment")

    elif failure_mode == "wrong_schema":
        match = "field names"

        def failing_reader():
            yield first_fragment_batch
            yield pa.record_batch([pa.array(range(4))], ["not_blob"])

    else:
        match = "Stream produced more values than expected for dataset"

        def failing_reader():
            yield first_fragment_batch
            yield second_fragment_batch
            yield pa.record_batch([lance.blob_array([payloads[0]])], schema=schema)

    with pytest.raises(OSError, match=match):
        ds.add_columns(failing_reader(), reader_schema=schema)

    assert ds.version == 1
    files_after = _dataset_file_set(dataset_path)
    assert files_before <= files_after
    assert files_after == files_before
    ds.cleanup_old_versions(delete_unverified=True)
    assert _dataset_file_set(dataset_path) == files_before
    assert external_blob_path.exists()


def test_blob_extension_add_columns_batch_udf_failure_cleans_files(tmp_path):
    ds, dataset_path, values, _ = _write_two_fragment_blob_v2_seed_dataset(
        tmp_path,
        "test_add_columns_udf_blob_v2_fail_cleanup",
    )
    external_blob_path = tmp_path / "external_base" / "external_blob.bin"
    files_before = _dataset_file_set(dataset_path)
    call_count = 0

    @lance.batch_udf(output_schema=pa.schema([lance.blob_field("blob")]))
    def fail_on_second_fragment(batch):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("udf failed after first fragment")
        blob_values = [values[row.as_py() % len(values)] for row in batch["id"]]
        return pa.record_batch(
            [lance.blob_array(blob_values)],
            ["blob"],
        )

    with pytest.raises(OSError, match="udf failed after first fragment"):
        ds.add_columns(fail_on_second_fragment, read_columns=["id"], batch_size=4)

    assert call_count == 2
    assert ds.version == 1
    files_after = _dataset_file_set(dataset_path)
    assert files_before <= files_after
    assert files_after == files_before
    ds.cleanup_old_versions(delete_unverified=True)
    assert _dataset_file_set(dataset_path) == files_before
    assert external_blob_path.exists()


def test_blob_extension_add_columns_batch_udf_all_kinds(tmp_path):
    values, payloads, initial_bases = _add_columns_blob_v2_values(tmp_path)
    ds = lance.write_dataset(
        pa.table({"id": range(4)}),
        tmp_path / "test_add_columns_udf_blob_v2",
        data_storage_version="2.2",
        initial_bases=initial_bases,
    )

    @lance.batch_udf(output_schema=pa.schema([lance.blob_field("blob")]))
    def make_blob_column(batch):
        return pa.record_batch(
            [lance.blob_array([values[row.as_py()] for row in batch["id"]])],
            ["blob"],
        )

    ds.add_columns(make_blob_column, read_columns=["id"])

    _assert_blob_v2_add_columns_result(ds, "blob", payloads)


def test_blob_extension_add_columns_all_nulls_blob_v2(tmp_path):
    ds = lance.write_dataset(
        pa.table({"id": range(4)}),
        tmp_path / "test_add_columns_all_nulls_blob_v2",
        data_storage_version="2.2",
    )

    ds.add_columns(lance.blob_field("blob"))

    assert ds.to_table(columns=["blob"]).column("blob").to_pylist() == [None] * 4
    assert ds.take_blobs("blob", indices=range(4)) == [None] * 4
    assert [data for _, data in ds.read_blobs("blob", indices=range(4))] == [None] * 4
    ranges = ds.read_blob_ranges(
        "blob",
        [(index, 0, 1) for index in range(4)],
        selector="indices",
    )
    assert [data for _, _, data in ranges] == [None] * 4


@pytest.mark.parametrize("version", ["2.2", "2.3"])
@pytest.mark.parametrize("nested", [False, True], ids=["top_level", "struct"])
@pytest.mark.parametrize(
    "prepared", [False, True], ids=["logical_upload", "prepared_upload"]
)
def test_prepared_managed_blob_sharing_lifecycle(tmp_path, version, nested, prepared):
    dataset_uri = tmp_path / "shared.lance"

    def table(ids, field, values):
        if nested:
            values = pa.StructArray.from_arrays([values], fields=[field])
            field = pa.field(
                "value", values.type, metadata={b"example:type": b"opaque"}
            )
        return pa.Table.from_arrays(
            [pa.array(ids, type=pa.int64()), values],
            schema=pa.schema(
                [pa.field("id", pa.int64()), field],
                metadata={"lance.blob.managed": "true"},
            ),
        )

    if prepared:
        writer = LanceFileSession(dataset_uri).open_managed_blob_writer()
        assert isinstance(writer, lance.ManagedBlobWriter)
        writer.write_blobs(pa.array([b"shared", b"", None, b"unused"]))
        shared, empty, null, _unused = writer.finish()
        with pytest.raises(ValueError, match="ManagedBlobWriter is already finished"):
            writer.write_blob(b"late")
        builder = lance.BlobDescriptorArrayBuilder("blob")
        builder.append(shared)
        initial = table([0], builder.field, builder.finish())
    else:
        initial = table(
            [0],
            lance.blob_field("blob", inline_size_threshold=0),
            lance.blob_array([b"shared"]),
        )
    ds = lance.write_dataset(initial, dataset_uri, data_storage_version=version)
    blob_column = "value.blob" if nested else "blob"
    row = ds.to_table(blob_handling="all_descriptions").to_pylist()[0]
    descriptor = (row["value"] if nested else row)["blob"]
    assert descriptor["kind"] == 4
    payload_files = set(dataset_uri.glob("_blobs/*.blob"))
    assert len(payload_files) == 1
    payload_contents = b"sharedunused" if prepared else b"shared"
    assert next(iter(payload_files)).read_bytes() == payload_contents

    builder = lance.BlobDescriptorArrayBuilder("blob")
    if prepared:
        builder.extend([shared, shared])
    else:
        for _ in range(2):
            builder.append_managed(
                descriptor["blob_uri"], descriptor["position"], descriptor["size"]
            )
    builder.append_managed(descriptor["blob_uri"], descriptor["position"] + 1, 3)
    if prepared:
        builder.extend([null, empty])
    else:
        builder.append_null()
        builder.append_inline(b"")
    ds = lance.write_dataset(
        table([1, 2, 3, 4, 5], builder.field, builder.finish()),
        dataset_uri,
        mode="append",
        max_rows_per_file=1,
    )
    assert len(ds.get_fragments()) == 6
    assert [data for _, data in ds.read_blobs(blob_column, indices=range(6))] == [
        b"shared",
        b"shared",
        b"shared",
        b"har",
        None,
        b"",
    ]
    assert set(dataset_uri.glob("_blobs/*.blob")) == payload_files

    ds.optimize.compact_files()
    assert len(ds.get_fragments()) == 1
    ds.delete("id = 0")
    ds.cleanup_old_versions(retain_versions=1, delete_unverified=True)
    assert [data for _, data in ds.read_blobs(blob_column, indices=range(5))] == [
        b"shared",
        b"shared",
        b"har",
        None,
        b"",
    ]
    assert set(dataset_uri.glob("_blobs/*.blob")) == payload_files
    if nested:
        assert ds.schema.field("value").metadata == {b"example:type": b"opaque"}
    assert next(iter(payload_files)).read_bytes() == payload_contents

    protected_version = ds.version
    ds.tags.create("shared", protected_version)
    ds.delete("id < 4 or id = 5")
    ds.cleanup_old_versions(
        retain_versions=1, delete_unverified=True, error_if_tagged_old_versions=False
    )
    protected = lance.dataset(dataset_uri, version=protected_version)
    assert protected.take_blobs(blob_column, indices=[0])[0].readall() == b"shared"
    assert set(dataset_uri.glob("_blobs/*.blob")) == payload_files

    ds.tags.delete("shared")
    ds.cleanup_old_versions(retain_versions=1, delete_unverified=True)
    assert not list(dataset_uri.glob("_blobs/*.blob"))
    assert ds.take_blobs(blob_column, indices=[0]) == [None]


@pytest.mark.parametrize("base_id", [None, 7, 2**32 - 1])
def test_managed_blob_writer_finish_column_and_commit_validation(tmp_path, base_id):
    dataset_uri = tmp_path / "prepared.lance"
    payload_base = dataset_uri if base_id is None else dataset_uri / "payloads"
    writer = LanceFileSession(payload_base).open_managed_blob_writer(base_id=base_id)
    writer.write_blobs(pa.array([b"abc", b"", None]))
    field, values = writer.finish_column("blob")
    assert field.name == "blob"
    assert field.metadata[b"ARROW:extension:name"] == b"lance.blob.v2"
    with pytest.raises(ValueError, match="ManagedBlobWriter is already finished"):
        writer.finish_column("blob")
    assert values.field("kind").to_pylist() == [4 if base_id is None else 5] * 2 + [
        None
    ]
    table = pa.Table.from_arrays([values], schema=pa.schema([field]))
    with pytest.raises(
        OSError, match="requires schema metadata lance.blob.managed=true"
    ):
        lance.write_dataset(table, dataset_uri, data_storage_version="2.2")
    table = table.replace_schema_metadata({"lance.blob.managed": "true"})
    if base_id is not None:
        with pytest.raises(OSError, match=f"unknown base_id {base_id}"):
            lance.write_dataset(table, dataset_uri, data_storage_version="2.2")
    table = table.take(pa.array([0, 0, 0, 1, 2]))
    ds = lance.write_dataset(
        table,
        dataset_uri,
        data_storage_version="2.2",
        initial_bases=(
            [DatasetBasePath(payload_base.as_uri(), id=base_id)]
            if base_id is not None
            else None
        ),
        target_all_bases=True if base_id is not None else None,
        max_rows_per_file=1,
    )
    assert len(ds.get_fragments()) == 5
    if base_id is not None:
        assert list(dataset_uri.glob("data/*.lance"))
        assert list(payload_base.glob("*.lance"))
    assert [data for _, data in ds.read_blobs("blob", indices=range(5))] == [
        b"abc",
        b"abc",
        b"abc",
        b"",
        None,
    ]
    assert [path.read_bytes() for path in payload_base.glob("_blobs/*.blob")] == [
        b"abc"
    ]


@pytest.mark.parametrize("base_id", [None, 7])
def test_blob_descriptor_array_builder_managed(base_id):
    builder = lance.BlobDescriptorArrayBuilder("blob")
    builder.append_managed("_blobs/existing.blob", 10, 0, base_id=base_id)
    assert builder.finish().to_pylist() == [
        {
            "kind": 4 if base_id is None else 5,
            "data": None,
            "uri": "_blobs/existing.blob",
            "blob_id": base_id,
            "blob_size": 0,
            "position": 10,
        }
    ]
    with pytest.raises(ValueError, match="already finished"):
        builder.append_managed("_blobs/existing.blob", 0, 1)


@pytest.mark.parametrize(
    "uri,offset,size,message",
    [
        ("../outside.blob", 0, 1, "canonical relative object path"),
        ("s3://bucket/object", 0, 1, "canonical relative object path"),
        ("_blobs/existing.blob", 2**64 - 1, 1, "overflows u64"),
    ],
)
def test_blob_descriptor_array_builder_rejects_invalid_managed(
    uri, offset, size, message
):
    builder = lance.BlobDescriptorArrayBuilder("blob")
    with pytest.raises(ValueError, match=message):
        builder.append_managed(uri, offset, size)
    assert len(builder.finish()) == 0


def test_blob_descriptor_array_builder_writes_prepared_packed_blob_for_data_replacement(
    tmp_path,
):
    dataset_uri = tmp_path / "test_blob_descriptor_array_builder_data_replacement"
    logical_schema = pa.schema(
        [
            pa.field("id", pa.uint32(), nullable=False),
            lance.blob_field("blob"),
        ]
    )
    initial = pa.table(
        [
            pa.array([0], type=pa.uint32()),
            lance.blob_array([b"initial"]),
        ],
        schema=logical_schema,
    )
    ds = lance.write_dataset(initial, dataset_uri, data_storage_version="2.2")

    file_id = str(uuid.uuid4())
    data_file_name = f"{file_id}.lance"
    blob_id = 1
    blob_path = _blob_sidecar_path(dataset_uri / "data", file_id, blob_id)

    files = LanceFileSession(dataset_uri / "data")
    blob_writer = lance.BlobDescriptorArrayBuilder("blob")
    packed = files.open_packed_blob_writer(data_file_name, blob_id)
    assert packed.path.endswith(blob_path.relative_to(dataset_uri).as_posix())
    packed.write_blob(b"replacement")
    blob_writer.extend(packed.finish())

    physical_schema = pa.schema(
        [
            pa.field("id", pa.uint32(), nullable=False),
            blob_writer.field,
        ]
    )
    replacement = pa.record_batch(
        [
            pa.array([1], type=pa.uint32()),
            blob_writer.finish(),
        ],
        schema=physical_schema,
    )

    with files.open_writer(
        data_file_name, schema=physical_schema, version="2.2"
    ) as file_writer:
        file_writer.write_batch(replacement)

    data_file = lance.fragment.DataFile.create(ds, data_file_name)
    operation = lance.LanceOperation.DataReplacement(
        [lance.LanceOperation.DataReplacementGroup(0, data_file)]
    )
    ds = lance.LanceDataset.commit(ds, operation, read_version=ds.version)

    assert ds.to_table(columns=["id"]).column("id").to_pylist() == [1]
    blobs = ds.take_blobs("blob", indices=[0])
    assert len(blobs) == 1
    assert blobs[0].readall() == b"replacement"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"payload", id="bytes"),
        pytest.param(bytearray(b"payload"), id="bytearray"),
        pytest.param(memoryview(b"payload"), id="memoryview"),
        pytest.param(list(b"payload"), id="integer_sequence"),
    ],
)
def test_packed_blob_writer_scalar_buffer_inputs(tmp_path, payload):
    file_id = str(uuid.uuid4())
    blob_id = 7
    files = LanceFileSession(tmp_path)
    packed = files.open_packed_blob_writer(f"{file_id}.lance", blob_id)

    packed.write_blob(payload)
    descriptors = packed.finish()

    assert [repr(descriptor) for descriptor in descriptors] == [
        "Packed { blob_id: 7, offset: 0, size: 7 }"
    ]
    assert _blob_sidecar_path(tmp_path, file_id, blob_id).read_bytes() == b"payload"


@pytest.mark.parametrize(
    "array_type", [pa.binary(), pa.large_binary(), pa.binary_view()]
)
@pytest.mark.parametrize("as_chunked", [False, True], ids=["array", "chunked_array"])
@pytest.mark.parametrize(
    "values,slice_offset,slice_length,expected_values,expected_data",
    [
        pytest.param(
            [b"prefix", b"a", None, b"", b"bc", b"suffix"],
            1,
            4,
            [b"a", None, b"", b"bc"],
            b"abc",
            id="interleaved_null",
        ),
        pytest.param(
            [b"prefix", b"a", b"", b"bc", b"suffix"],
            1,
            3,
            [b"a", b"", b"bc"],
            b"abc",
            id="all_valid",
        ),
        pytest.param(
            [b"prefix", None, None, b"suffix"],
            1,
            2,
            [None, None],
            b"",
            id="all_null",
        ),
        pytest.param(
            [b"prefix", b"suffix"],
            1,
            0,
            [],
            b"",
            id="empty",
        ),
    ],
)
def test_packed_blob_writer_bulk_arrow_array(
    tmp_path,
    array_type,
    as_chunked,
    values,
    slice_offset,
    slice_length,
    expected_values,
    expected_data,
):
    file_id = str(uuid.uuid4())
    data_file_name = f"{file_id}.lance"
    blob_id = 7
    payloads = pa.array(values, type=array_type).slice(slice_offset, slice_length)
    if as_chunked:
        split_at = max(1, len(payloads) // 2)
        payloads = pa.chunked_array(
            [payloads.slice(0, split_at), payloads.slice(split_at)]
        )

    files = LanceFileSession(tmp_path)
    packed = files.open_packed_blob_writer(data_file_name, blob_id)
    with pytest.raises(ValueError, match="available after finish_array"):
        packed.field
    packed.write_blobs(payloads)
    descriptors = packed.finish_array("image_bytes")
    descriptor_field = packed.field

    expected_descriptors = []
    position = 0
    for value in expected_values:
        if value is None:
            expected_descriptors.append(None)
        else:
            expected_descriptors.append(
                {
                    "kind": 1,
                    "data": None,
                    "uri": None,
                    "blob_id": blob_id,
                    "blob_size": len(value),
                    "position": position,
                }
            )
            position += len(value)

    assert descriptors.to_pylist() == expected_descriptors
    assert descriptor_field == lance.BlobDescriptorArrayBuilder("image_bytes").field
    assert descriptor_field.metadata[b"ARROW:extension:name"] == b"lance.blob.v2"
    assert pa.record_batch(
        [descriptors], schema=pa.schema([descriptor_field])
    ).num_rows == len(expected_values)
    assert _blob_sidecar_path(tmp_path, file_id, blob_id).read_bytes() == expected_data


@pytest.mark.parametrize(
    "array_type,offset_type",
    [
        pytest.param(pa.binary(), pa.int32(), id="binary"),
        pytest.param(pa.large_binary(), pa.int64(), id="large_binary"),
    ],
)
def test_packed_blob_writer_bulk_excludes_physical_null_bytes(
    tmp_path, array_type, offset_type
):
    offsets = pa.array([0, 1, 5, 5, 7], type=offset_type).buffers()[1]
    payloads = pa.Array.from_buffers(
        array_type,
        4,
        [
            pa.py_buffer(bytes([0b00001101])),
            offsets,
            pa.py_buffer(b"aJUNKbc"),
        ],
    )
    file_id = str(uuid.uuid4())
    blob_id = 7
    files = LanceFileSession(tmp_path)
    packed = files.open_packed_blob_writer(f"{file_id}.lance", blob_id)

    packed.write_blobs(payloads)
    descriptors = packed.finish_array("image_bytes")

    assert descriptors.to_pylist() == [
        {
            "kind": 1,
            "data": None,
            "uri": None,
            "blob_id": blob_id,
            "blob_size": 1,
            "position": 0,
        },
        None,
        {
            "kind": 1,
            "data": None,
            "uri": None,
            "blob_id": blob_id,
            "blob_size": 0,
            "position": 1,
        },
        {
            "kind": 1,
            "data": None,
            "uri": None,
            "blob_id": blob_id,
            "blob_size": 2,
            "position": 1,
        },
    ]
    assert _blob_sidecar_path(tmp_path, file_id, blob_id).read_bytes() == b"abc"


@pytest.mark.parametrize(
    "array_type,offset_type",
    [
        pytest.param(pa.binary(), pa.int32(), id="binary"),
        pytest.param(pa.large_binary(), pa.int64(), id="large_binary"),
    ],
)
@pytest.mark.parametrize("as_chunked", [False, True], ids=["array", "chunked_array"])
def test_packed_blob_writer_bulk_rejects_non_monotonic_offsets(
    tmp_path, array_type, offset_type, as_chunked
):
    offsets = pa.array([0, 2, 1, 2], type=offset_type).buffers()[1]
    malformed = pa.Array.from_buffers(
        array_type,
        3,
        [None, offsets, pa.py_buffer(b"ab")],
    )
    payloads = malformed
    expected_context = "Packed blob payload array"
    if as_chunked:
        payloads = pa.chunked_array([pa.array([b"valid"], type=array_type), malformed])
        expected_context = "Packed blob payload chunk 1"

    file_id = str(uuid.uuid4())
    blob_id = 7
    files = LanceFileSession(tmp_path)
    packed = files.open_packed_blob_writer(f"{file_id}.lance", blob_id)

    with pytest.raises(ValueError, match="invalid Arrow data") as error:
        packed.write_blobs(payloads)
    assert expected_context in str(error.value)
    assert "non-monotonic offset" in str(error.value)

    packed.write_blob(b"still usable")
    descriptors = packed.finish_array("blob")
    assert len(descriptors) == 1
    assert (
        _blob_sidecar_path(tmp_path, file_id, blob_id).read_bytes() == b"still usable"
    )


def test_packed_blob_writer_mixed_calls_preserve_legacy_finish_alignment(tmp_path):
    file_id = str(uuid.uuid4())
    blob_id = 7
    files = LanceFileSession(tmp_path)
    packed = files.open_packed_blob_writer(f"{file_id}.lance", blob_id)

    packed.write_blob(b"s")
    packed.write_blobs(pa.array([b"a", None, b""], type=pa.binary()))
    packed.write_blobs(pa.array([None, b"bc"], type=pa.large_binary()))
    descriptors = packed.finish()

    assert [repr(descriptor) for descriptor in descriptors] == [
        "Packed { blob_id: 7, offset: 0, size: 1 }",
        "Packed { blob_id: 7, offset: 1, size: 1 }",
        "Null",
        "Packed { blob_id: 7, offset: 2, size: 0 }",
        "Null",
        "Packed { blob_id: 7, offset: 2, size: 2 }",
    ]
    assert _blob_sidecar_path(tmp_path, file_id, blob_id).read_bytes() == b"sabc"


@pytest.mark.parametrize(
    "payloads",
    [
        pytest.param(pa.array([1, 2], type=pa.int32()), id="array"),
        pytest.param(pa.chunked_array([[1], [2]], type=pa.int32()), id="chunked_array"),
        pytest.param([b"not an Arrow array"], id="python_list"),
    ],
)
def test_packed_blob_writer_bulk_rejects_non_binary_array(tmp_path, payloads):
    files = LanceFileSession(tmp_path)
    packed = files.open_packed_blob_writer("data-file.lance", 1)

    with pytest.raises(ValueError, match="Binary") as error:
        packed.write_blobs(payloads)
    if isinstance(payloads, pa.Array):
        assert "chunk" not in str(error.value)

    packed.write_blob(b"still usable")
    assert len(packed.finish_array("blob")) == 1


@pytest.mark.parametrize(
    "open_writer",
    [
        pytest.param("open_packed_blob_writer", id="packed"),
        pytest.param("open_dedicated_blob_writer", id="dedicated"),
    ],
)
def test_failed_blob_writer_traceback_can_be_released_on_another_thread(
    tmp_path, monkeypatch, open_writer
):
    failures = queue.Queue()

    def fail_with_live_writer():
        files = LanceFileSession(tmp_path)
        writer = getattr(files, open_writer)("data-file.lance", 1)
        assert writer.blob_id == 1
        try:
            raise OSError("simulated object-store write failure")
        except OSError as error:
            # The traceback retains this frame and its local writer, matching a
            # writer-thread failure handed to an owning thread for propagation.
            failures.put(error)

    writer_thread = threading.Thread(target=fail_with_live_writer)
    writer_thread.start()
    writer_thread.join(timeout=10)
    assert not writer_thread.is_alive()

    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    error = failures.get_nowait()
    assert str(error) == "simulated object-store write failure"
    del error
    gc.collect()

    assert unraisable == []


def test_blob_extension_write_fragments_external_denied_by_default(tmp_path):
    blob_path = tmp_path / "external_blob.bin"

    table = _external_blob_table(blob_path)
    with pytest.raises(OSError, match="outside registered external bases"):
        write_fragments(
            table,
            tmp_path / "test_fragments_v2_external_denied",
            data_storage_version="2.2",
            external_blob_mode="reference",
        )


def test_blob_extension_write_fragments_external_outside_base_allowed(tmp_path):
    blob_path = tmp_path / "external_blob.bin"
    dataset_uri = tmp_path / "test_fragments_v2_external_allowed"

    table = _external_blob_table(blob_path)
    fragments = write_fragments(
        table,
        dataset_uri,
        data_storage_version="2.2",
        external_blob_mode="reference",
        allow_external_blob_outside_bases=True,
    )
    ds = _commit_blob_fragments(dataset_uri, table.schema, fragments)

    blob = ds.take_blobs("blob", indices=[0])[0]
    assert blob.size() == 5
    with blob as f:
        assert f.read() == b"hello"


def test_blob_extension_write_fragments_transaction_external_outside_base_allowed(
    tmp_path,
):
    blob_path = tmp_path / "external_blob.bin"
    dataset_uri = tmp_path / "test_fragments_transaction_v2_external_allowed"

    table = _external_blob_table(blob_path)
    transaction = write_fragments(
        table,
        dataset_uri,
        mode="create",
        return_transaction=True,
        data_storage_version="2.2",
        external_blob_mode="reference",
        allow_external_blob_outside_bases=True,
    )
    ds = lance.LanceDataset.commit(dataset_uri, transaction)

    blob = ds.take_blobs("blob", indices=[0])[0]
    assert blob.size() == 5
    with blob as f:
        assert f.read() == b"hello"


def test_blob_extension_write_fragments_external_registered_base(tmp_path):
    external_base = tmp_path / "external_base"
    blob_path = external_base / "external_blob.bin"
    dataset_uri = tmp_path / "test_fragments_v2_external_registered_base"

    table = _external_blob_table(blob_path)
    initial_bases = [DatasetBasePath(external_base.as_uri(), name="external", id=1)]
    fragments = write_fragments(
        table,
        dataset_uri,
        mode="create",
        data_storage_version="2.2",
        external_blob_mode="reference",
        initial_bases=initial_bases,
    )
    ds = _commit_blob_fragments(
        dataset_uri,
        table.schema,
        fragments,
        initial_bases=initial_bases,
    )

    blob = ds.take_blobs("blob", indices=[0])[0]
    assert blob.size() == 5
    with blob as f:
        assert f.read() == b"hello"


def test_blob_extension_write_fragments_external_ingest(tmp_path):
    blob_path = tmp_path / "external_blob.bin"
    dataset_uri = tmp_path / "test_fragments_v2_external_ingest"

    table = _external_blob_table(blob_path)
    fragments = write_fragments(
        table,
        dataset_uri,
        data_storage_version="2.2",
        external_blob_mode="ingest",
    )
    ds = _commit_blob_fragments(dataset_uri, table.schema, fragments)

    blob_path.unlink()

    blob = ds.take_blobs("blob", indices=[0])[0]
    assert blob.size() == 5
    with blob as f:
        assert f.read() == b"hello"


def test_blob_extension_write_fragments_transaction_external_ingest(tmp_path):
    blob_path = tmp_path / "external_blob.bin"
    dataset_uri = tmp_path / "test_fragments_transaction_v2_external_ingest"

    table = _external_blob_table(blob_path)
    transaction = write_fragments(
        table,
        dataset_uri,
        mode="create",
        return_transaction=True,
        data_storage_version="2.2",
        external_blob_mode="ingest",
    )
    ds = lance.LanceDataset.commit(dataset_uri, transaction)

    blob_path.unlink()

    blob = ds.take_blobs("blob", indices=[0])[0]
    assert blob.size() == 5
    with blob as f:
        assert f.read() == b"hello"


def test_blob_extension_write_fragments_external_ingest_rejects_reference_only_options(
    tmp_path,
):
    blob_path = tmp_path / "external_blob.bin"
    message = (
        "allow_external_blob_outside_bases only applies when "
        'external_blob_mode="reference"'
    )

    table = _external_blob_table(blob_path)
    with pytest.raises(OSError, match=message):
        write_fragments(
            table,
            tmp_path / "test_fragments_v2_external_ingest_invalid",
            data_storage_version="2.2",
            external_blob_mode="ingest",
            allow_external_blob_outside_bases=True,
        )


def test_blob_extension_write_external_slice(tmp_path):
    tar_path = tmp_path / "container.tar"
    names = ["a.bin", "b.bin", "c.bin"]
    payloads = [b"alpha", b"bravo", b"charlie"]

    # Build a tar container with three distinct binary entries.
    with tarfile.open(tar_path, "w") as tf:
        for name, data in zip(names, payloads):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    # Re-open the tar to obtain offsets and sizes for each member.
    positions: list[int] = []
    sizes: list[int] = []
    with tarfile.open(tar_path, "r") as tf:
        for name in names:
            member = tf.getmember(name)
            positions.append(member.offset_data)
            sizes.append(member.size)

    uri = tar_path.as_uri()

    blob_values = [
        Blob.from_uri(uri, position, size) for position, size in zip(positions, sizes)
    ]

    table = pa.table({"blob": lance.blob_array(blob_values)})

    ds = lance.write_dataset(
        table,
        tmp_path / "ds",
        data_storage_version="2.2",
        allow_external_blob_outside_bases=True,
    )

    blobs = ds.take_blobs("blob", indices=[0, 1, 2])
    assert len(blobs) == len(payloads)

    for expected, blob_file in zip(payloads, blobs):
        assert blob_file.size() == len(expected)
        with blob_file as f:
            assert f.read() == expected

    assert ds.read_blobs("blob", indices=[0, 1, 2]) == [
        (0, b"alpha"),
        (1, b"bravo"),
        (2, b"charlie"),
    ]


def test_blob_extension_write_external_slice_ingest(tmp_path):
    tar_path = tmp_path / "container.tar"
    names = ["a.bin", "b.bin", "c.bin"]
    payloads = [b"alpha", b"bravo", b"charlie"]

    with tarfile.open(tar_path, "w") as tf:
        for name, data in zip(names, payloads):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))

    positions: list[int] = []
    sizes: list[int] = []
    with tarfile.open(tar_path, "r") as tf:
        for name in names:
            member = tf.getmember(name)
            positions.append(member.offset_data)
            sizes.append(member.size)

    uri = tar_path.as_uri()
    blob_values = [
        Blob.from_uri(uri, position, size) for position, size in zip(positions, sizes)
    ]
    table = pa.table({"blob": lance.blob_array(blob_values)})

    ds = lance.write_dataset(
        table,
        tmp_path / "ds_ingest",
        data_storage_version="2.2",
        external_blob_mode="ingest",
    )

    tar_path.unlink()

    blobs = ds.take_blobs("blob", indices=[0, 1, 2])
    assert len(blobs) == len(payloads)

    for expected, blob_file in zip(payloads, blobs):
        assert blob_file.size() == len(expected)
        with blob_file as f:
            assert f.read() == expected


@pytest.mark.parametrize(
    ("payload", "is_dataset_root"),
    [
        (b"inline", True),
        (b"p" * (64 * 1024 + 1024), True),
        (b"d" * (4 * 1024 * 1024 + 1024), True),
        (b"x" * (64 * 1024 + 1024), False),
    ],
    ids=["inline", "packed", "dedicated", "packed_data_only_base"],
)
def test_blob_extension_take_blobs_multi_base(payload, is_dataset_root, tmp_path):
    base_path = tmp_path / "blob_base"
    base_path.mkdir(parents=True, exist_ok=True)
    table = pa.table({"blob": lance.blob_array([payload])})

    ds = lance.write_dataset(
        table,
        tmp_path / "primary_ds",
        mode="create",
        data_storage_version="2.2",
        initial_bases=[
            DatasetBasePath(
                str(base_path), name="blob_base", is_dataset_root=is_dataset_root
            )
        ],
        target_bases=["blob_base"],
    )

    fragments = list(ds.get_fragments())
    assert len(fragments) == 1
    data_file = fragments[0].data_files()[0]
    assert data_file.base_id is not None

    blobs = ds.take_blobs("blob", indices=[0])
    assert len(blobs) == 1
    with blobs[0] as f:
        assert f.read() == payload

    assert ds.read_blobs("blob", indices=[0]) == [(0, payload)]


@pytest.fixture
def dataset_for_pandas_blob_tests(tmp_path):
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "blob": pa.array([b"hello", None, b"world"], pa.large_binary()),
            "bin": pa.array([b"x", b"y", b"z"], pa.large_binary()),
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field(
                    "blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
                ),
                pa.field("bin", pa.large_binary()),
            ]
        ),
    )
    return lance.write_dataset(
        table,
        tmp_path / "blob_pandas_ds",
        data_storage_version=LEGACY_BLOB_STORAGE_VERSION,
    )


@pytest.fixture
def dataset_for_pandas_no_blob_tests(tmp_path):
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], pa.int64()),
            "name": pa.array(["one", "two", "three"], pa.string()),
            "bin": pa.array([b"x", b"y", b"z"], pa.large_binary()),
        }
    )
    return lance.write_dataset(table, tmp_path / "no_blob_pandas_ds")


def _lazy_blob_values(dataset):
    values = dataset.scanner(columns=["blob"]).to_pandas(blob_mode="lazy")["blob"]
    return [None if value is None else value.readall() for value in values]


@pytest.mark.parametrize(
    "values, has_sidecar",
    [
        pytest.param([b"", b"payload", None], False, id="leading_empty"),
        pytest.param([b"", b"", b""], False, id="all_empty"),
        pytest.param(
            [b"payload", b"", None, b"tail", b""],
            False,
            id="payload_empty_null",
        ),
        pytest.param(
            [b"", b"p" * (64 * 1024 + 1024), None, b""],
            True,
            id="sidecar_backed",
        ),
    ],
)
def test_blob_v2_lazy_preserves_empty_and_null(tmp_path, values, has_sidecar):
    dataset_path = tmp_path / "blob_v2_lazy_empty"
    schema = pa.schema([lance.blob_field("blob")])
    dataset = lance.write_dataset(
        pa.Table.from_arrays([lance.blob_array(values)], schema=schema),
        dataset_path,
        data_storage_version="2.2",
    )

    descriptions = dataset.to_table(columns=["blob"]).column("blob").to_pylist()
    assert [description is None for description in descriptions] == [
        value is None for value in values
    ]
    if values[0] == b"":
        assert descriptions[0]["position"] == 0
        assert descriptions[0]["size"] == 0
    if has_sidecar:
        assert any(
            description is not None and description["kind"] == 1
            for description in descriptions
        )
        assert any(path.suffix == ".blob" for path in _dataset_file_set(dataset_path))

    assert _lazy_blob_values(dataset) == values


def test_blob_v1_lazy_preserves_empty_and_null_sentinels(tmp_path):
    values = [b"", None, b"payload"]
    schema = pa.schema(
        [
            pa.field(
                "blob",
                pa.large_binary(),
                metadata={"lance-encoding:blob": "true"},
            )
        ]
    )
    dataset = lance.write_dataset(
        pa.Table.from_arrays(
            [pa.array(values, type=pa.large_binary())],
            schema=schema,
        ),
        tmp_path / "blob_v1_lazy_empty",
        data_storage_version="2.0",
    )

    descriptions = dataset.to_table(columns=["blob"]).column("blob").to_pylist()
    assert descriptions[0] == {"position": 0, "size": 0}
    assert descriptions[1] == {"position": 1, "size": 0}
    assert _lazy_blob_values(dataset) == values


@pytest.mark.parametrize("source", ["dataset", "scanner", "fragment"])
def test_to_pandas_without_blobs_matches_arrow_with_kwargs(
    dataset_for_pandas_no_blob_tests,
    source,
):
    kwargs = {"types_mapper": pd.ArrowDtype}
    ds = dataset_for_pandas_no_blob_tests

    if source == "dataset":
        actual = ds.to_pandas(**kwargs)
        expected = ds.to_table().to_pandas(**kwargs)
    elif source == "scanner":
        scanner = ds.scanner(
            columns=["id", "name"],
            filter="id >= 2",
            limit=2,
            offset=0,
            batch_size=1,
        )
        actual = scanner.to_pandas(**kwargs)
        expected = scanner.to_table().to_pandas(**kwargs)
    else:
        fragment = ds.get_fragments()[0]
        actual = fragment.to_pandas(
            columns=["id", "name"],
            filter="id >= 2",
            limit=2,
            offset=0,
            batch_size=1,
            **kwargs,
        )
        expected = fragment.to_table(
            columns=["id", "name"],
            filter="id >= 2",
            limit=2,
            offset=0,
        ).to_pandas(**kwargs)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.dtypes["id"] == pd.ArrowDtype(pa.int64())


def test_dataset_to_pandas_blob_lazy(dataset_for_pandas_blob_tests):
    df = dataset_for_pandas_blob_tests.to_pandas()

    assert list(df.columns) == ["id", "blob", "bin"]
    assert isinstance(df.iloc[0]["blob"], BlobFile)
    assert df.iloc[1]["blob"] is None
    assert isinstance(df.iloc[2]["blob"], BlobFile)
    assert df["bin"].tolist() == [b"x", b"y", b"z"]
    assert [df.iloc[0]["blob"].readall(), df.iloc[2]["blob"].readall()] == [
        b"hello",
        b"world",
    ]


def test_dataset_to_pandas_blob_bytes(dataset_for_pandas_blob_tests):
    df = dataset_for_pandas_blob_tests.to_pandas(blob_mode="bytes")

    assert list(df.columns) == ["id", "blob", "bin"]
    assert df["blob"].tolist() == [b"hello", None, b"world"]
    assert df["bin"].tolist() == [b"x", b"y", b"z"]


def test_dataset_to_pandas_blob_descriptions(dataset_for_pandas_blob_tests):
    descriptions_df = dataset_for_pandas_blob_tests.to_pandas(blob_mode="descriptions")
    table_df = dataset_for_pandas_blob_tests.to_table().to_pandas()

    assert descriptions_df.equals(table_df)


def test_scanner_to_pandas_blob_alias(dataset_for_pandas_blob_tests):
    df = dataset_for_pandas_blob_tests.scanner(
        columns={"video": "blob", "id": "id"}
    ).to_pandas()

    assert list(df.columns) == ["video", "id"]
    assert isinstance(df.iloc[0]["video"], BlobFile)
    assert df.iloc[1]["video"] is None
    assert df.iloc[2]["video"].readall() == b"world"


def test_scanner_to_pandas_blob_filter_limit_order(dataset_for_pandas_blob_tests):
    df = dataset_for_pandas_blob_tests.scanner(
        columns=["id", "blob"],
        filter="id > 1",
        limit=1,
        order_by=["id"],
    ).to_pandas(blob_mode="bytes")

    assert list(df.columns) == ["id", "blob"]
    assert df["id"].tolist() == [2]
    assert df["blob"].tolist() == [None]


@pytest.mark.parametrize("source", ["dataset", "scanner", "fragment"])
def test_to_pandas_blob_scan_parameters(dataset_for_pandas_blob_tests, source):
    ds = dataset_for_pandas_blob_tests
    scan_kwargs = {
        "columns": ["id", "blob"],
        "filter": "id > 1",
        "limit": 1,
        "offset": 1,
        "batch_size": 1,
    }

    if source == "dataset":
        df = ds.to_pandas(**scan_kwargs, blob_mode="bytes")
    elif source == "scanner":
        df = ds.scanner(**scan_kwargs).to_pandas(blob_mode="bytes")
    else:
        df = ds.get_fragments()[0].to_pandas(**scan_kwargs, blob_mode="bytes")

    assert list(df.columns) == ["id", "blob"]
    assert df["id"].tolist() == [3]
    assert df["blob"].tolist() == [b"world"]


def test_scanner_to_pandas_blob_empty_result(dataset_for_pandas_blob_tests):
    df = dataset_for_pandas_blob_tests.scanner(
        columns=["id", "blob"], filter="id > 10"
    ).to_pandas()

    assert list(df.columns) == ["id", "blob"]
    assert df.empty


def test_fragment_to_pandas_blob(dataset_for_pandas_blob_tests):
    fragment = dataset_for_pandas_blob_tests.get_fragments()[0]
    df = fragment.to_pandas(columns=["id", "blob"], batch_size=1, blob_mode="bytes")

    assert list(df.columns) == ["id", "blob"]
    assert df["blob"].tolist() == [b"hello", None, b"world"]


def test_dataset_to_pandas_invalid_blob_mode(dataset_for_pandas_blob_tests):
    with pytest.raises(ValueError, match="blob_mode must be one of"):
        dataset_for_pandas_blob_tests.to_pandas(blob_mode="inline")


@pytest.fixture
def dataset_with_nested_blobs(tmp_path):
    blob_field = pa.field(
        "blob", pa.large_binary(), metadata={"lance-encoding:blob": "true"}
    )
    info_type = pa.struct([pa.field("name", pa.string()), blob_field])
    info_array = pa.array(
        [
            {"name": "a", "blob": b"foo"},
            {"name": "b", "blob": None},
            {"name": "c", "blob": b"baz"},
        ],
        type=info_type,
    )
    table = pa.table(
        [pa.array([1, 2, 3], pa.int64()), info_array],
        schema=pa.schema([pa.field("id", pa.int64()), pa.field("info", info_type)]),
    )
    return lance.write_dataset(
        table,
        tmp_path / "nested_blob_ds",
        data_storage_version=LEGACY_BLOB_STORAGE_VERSION,
    )


def test_to_pandas_returns_blob_file_handles_for_nested_fields(
    dataset_with_nested_blobs,
):
    df = dataset_with_nested_blobs.to_pandas()
    row0, row1, row2 = df["info"].tolist()

    assert row0["blob"].readall() == b"foo"
    assert row1["blob"] is None
    assert row2["blob"].readall() == b"baz"


def test_to_pandas_reads_nested_blob_bytes_directly(dataset_with_nested_blobs):
    rows = dataset_with_nested_blobs.to_pandas(blob_mode="bytes")["info"].tolist()

    assert [r["blob"] for r in rows] == [b"foo", None, b"baz"]


def test_to_pandas_returns_descriptors_for_nested_fields(dataset_with_nested_blobs):
    descriptions_df = dataset_with_nested_blobs.to_pandas(blob_mode="descriptions")
    table_df = dataset_with_nested_blobs.to_table().to_pandas()

    assert descriptions_df.equals(table_df)


def test_take_blobs_resolves_nested_field_path(dataset_with_nested_blobs):
    blobs = dataset_with_nested_blobs.take_blobs("info.blob", indices=[0, 2])

    with blobs[0] as f:
        assert f.read() == b"foo"
    with blobs[1] as f:
        assert f.read() == b"baz"


def test_read_blobs_resolves_nested_field_path(dataset_with_nested_blobs):
    results = dataset_with_nested_blobs.read_blobs("info.blob", indices=[0, 2])

    assert [data for _, data in results] == [b"foo", b"baz"]


def test_write_nested_blob_v2_and_take_by_field_path(tmp_path):
    packed = b"x" * (70 * 1024)
    blob_field = lance.blob_field("blob")
    info_fields = [pa.field("name", pa.string()), blob_field]
    info_type = pa.struct(info_fields)
    info_array = pa.StructArray.from_arrays(
        [pa.array(["a", "b", "c"]), lance.blob_array([b"foo", packed, None])],
        fields=info_fields,
    )
    table = pa.table(
        [info_array],
        schema=pa.schema([pa.field("info", info_type)]),
    )

    dataset = lance.write_dataset(
        table,
        tmp_path / "nested_blob_v2",
        data_storage_version="2.2",
    )

    desc = dataset.to_table(columns=["info.blob"]).column("info.blob").chunk(0)
    assert desc.field("kind").to_pylist()[:2] == [0, 1]

    blobs = dataset.take_blobs("info.blob", indices=[0, 1])
    with blobs[0] as f:
        assert f.read() == b"foo"
    with blobs[1] as f:
        assert f.read() == packed

    assert dataset.take_blobs("info.blob", indices=[2]) == [None]


def test_write_list_struct_nested_blob_v2(tmp_path):
    payload_fields = [
        lance.blob_field("blob"),
        pa.field("format", pa.string()),
    ]
    payload_values = pa.StructArray.from_arrays(
        [
            lance.blob_array([b"first", b"second", None]),
            pa.array(["bin", "bin", "bin"]),
        ],
        fields=payload_fields,
    )
    media_fields = [
        pa.field("type", pa.string()),
        pa.field("payload", payload_values.type),
    ]
    media_values = pa.StructArray.from_arrays(
        [pa.array(["image", "image", "image"]), payload_values],
        fields=media_fields,
    )
    media = pa.ListArray.from_arrays(pa.array([0, 2, 3]), media_values)
    table = pa.table({"media": media})

    dataset = lance.write_dataset(
        table,
        tmp_path / "list_struct_nested_blob_v2",
        data_storage_version="2.2",
    )

    media = (
        dataset.scanner(columns=["media"], blob_handling="all_binary")
        .to_table()
        .column("media")
        .combine_chunks()
    )
    blobs = media.values.field("payload").field("blob")
    assert blobs.to_pylist() == [b"first", b"second", None]


def test_to_pandas_returns_blob_files_for_projected_nested_fields(
    dataset_with_nested_blobs,
):
    images = (
        dataset_with_nested_blobs.scanner(columns=["info.blob"])
        .to_pandas()["info.blob"]
        .tolist()
    )

    assert images[0].readall() == b"foo"
    assert images[1] is None
    assert images[2].readall() == b"baz"


def test_to_pandas_reads_bytes_for_projected_nested_fields(dataset_with_nested_blobs):
    df = dataset_with_nested_blobs.scanner(columns=["info.blob"]).to_pandas(
        blob_mode="bytes"
    )

    assert df["info.blob"].tolist() == [b"foo", None, b"baz"]


def test_to_pandas_returns_blob_files_when_nested_field_is_aliased(
    dataset_with_nested_blobs,
):
    images = (
        dataset_with_nested_blobs.scanner(columns={"my_img": "info.blob"})
        .to_pandas()["my_img"]
        .tolist()
    )

    assert images[0].readall() == b"foo"
    assert images[1] is None
    assert images[2].readall() == b"baz"


@pytest.mark.parametrize("as_chunked", [False, True], ids=["array", "chunked_array"])
def test_packed_blob_writer_bulk_binary_view(tmp_path, as_chunked):
    file_id = str(uuid.uuid4())
    blob_id = 7
    values = [b"hello", None, b"", b"world"]
    payloads = pa.array(values, type=pa.binary_view())
    if as_chunked:
        payloads = pa.chunked_array([payloads.slice(0, 2), payloads.slice(2)])

    files = LanceFileSession(tmp_path)
    packed = files.open_packed_blob_writer(f"{file_id}.lance", blob_id)
    packed.write_blobs(payloads)
    descriptors = packed.finish_array("image_bytes")

    expected_descriptors = []
    position = 0
    for value in values:
        if value is None:
            expected_descriptors.append(None)
        else:
            expected_descriptors.append(
                {
                    "kind": 1,
                    "data": None,
                    "uri": None,
                    "blob_id": blob_id,
                    "blob_size": len(value),
                    "position": position,
                }
            )
            position += len(value)

    assert descriptors.to_pylist() == expected_descriptors
    assert _blob_sidecar_path(tmp_path, file_id, blob_id).read_bytes() == b"helloworld"


@pytest.mark.parametrize("as_chunked", [False, True], ids=["array", "chunked_array"])
def test_packed_blob_writer_bulk_fixed_size_binary(tmp_path, as_chunked):
    file_id = str(uuid.uuid4())
    blob_id = 7
    values = [b"word", None, b"test"]
    payloads = pa.array(values, type=pa.binary(4))
    if as_chunked:
        payloads = pa.chunked_array([payloads.slice(0, 2), payloads.slice(2)])

    files = LanceFileSession(tmp_path)
    packed = files.open_packed_blob_writer(f"{file_id}.lance", blob_id)
    packed.write_blobs(payloads)
    descriptors = packed.finish_array("image_bytes")

    expected_descriptors = []
    position = 0
    for value in values:
        if value is None:
            expected_descriptors.append(None)
        else:
            expected_descriptors.append(
                {
                    "kind": 1,
                    "data": None,
                    "uri": None,
                    "blob_id": blob_id,
                    "blob_size": len(value),
                    "position": position,
                }
            )
            position += len(value)

    assert descriptors.to_pylist() == expected_descriptors
    assert _blob_sidecar_path(tmp_path, file_id, blob_id).read_bytes() == b"wordtest"


class _CountingBlobInner:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._pos = 0
        self._closed = False
        self.buffer_size: int | None = None
        self.read_sizes: list[int] = []

    def set_buffer_size(self, buffer_size: int) -> None:
        self.buffer_size = buffer_size

    def close(self) -> None:
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed

    def seek(self, position: int) -> None:
        self._pos = position

    def tell(self) -> int:
        return self._pos

    def size(self) -> int:
        return len(self._payload)

    def readall(self) -> bytes:
        data = self._payload[self._pos :]
        if data:
            self.read_sizes.append(len(data))
        self._pos = len(self._payload)
        return data

    def read_range(self, offset: int, length: int) -> bytes:
        return self._payload[offset : offset + length]

    def read_ranges(self, ranges: list[tuple[int, int]]) -> list[bytes]:
        return [self.read_range(offset, length) for offset, length in ranges]

    def read_into(self, b: bytearray) -> int:
        if self._pos >= len(self._payload):
            return 0
        bytes_read = min(len(b), len(self._payload) - self._pos)
        self.read_sizes.append(bytes_read)
        b[:bytes_read] = self._payload[self._pos : self._pos + bytes_read]
        self._pos += bytes_read
        return bytes_read


def _read_in_8k_chunks(blob: BlobFile) -> bytes:
    chunks = []
    while True:
        chunk = blob.read(8192)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.fixture
def portable_blob_references(tmp_path):
    fixture = json.loads(
        (
            Path(__file__).parents[3] / "test_data/blob_reference_v1/conformance.json"
        ).read_text()
    )
    values = [
        value.encode() if value is not None else None for value in fixture["payloads"]
    ]
    table = pa.Table.from_arrays(
        [lance.blob_array(values)],
        schema=pa.schema(
            [
                lance.blob_field(
                    "blob", inline_size_threshold=4, dedicated_size_threshold=8
                )
            ]
        ),
    )
    dataset = lance.write_dataset(
        table,
        tmp_path / "portable.lance",
        max_rows_per_file=3,
        enable_stable_row_ids=True,
        data_storage_version="2.2",
    )
    source = BlobSourceContext.from_dataset(dataset, "catalog/portable", "workers")
    retention = lance.SnapshotRetention.acquire(
        dataset, source.snapshot_identity(), str(uuid.uuid4())
    )
    try:
        batch = (
            dataset.to_table(
                columns=["blob"],
                with_row_address=True,
                blob_handling="all_descriptions",
            )
            .combine_chunks()
            .to_batches()[0]
        )
        with BlobRefCapture(
            dataset, source, field_id=dataset.lance_schema.field("blob").id()
        ) as capture:
            references = capture.capture(
                source,
                batch.schema.field("blob"),
                batch["_rowaddr"],
                batch["blob"],
            )
        capture.close()
        yield dataset, source, retention, references
    finally:
        retention.release(dataset)


def test_portable_nested_capture_null_kind(tmp_path):
    dataset = lance.write_dataset(
        pa.table({"blob": lance.blob_array([b"a", None])}),
        tmp_path / "null-kind.lance",
        data_storage_version="2.2",
    )
    scanned = dataset.to_table(with_row_address=True).combine_chunks().to_batches()[0]
    descriptors = scanned["blob"]
    # Arrow operators may retain valid structs while making their children nullable.
    descriptors = pa.StructArray.from_arrays(
        [pa.array([None, None], type=pa.uint8())]
        + [descriptors.field(i) for i in range(1, len(descriptors.type))],
        fields=[field.with_nullable(True) for field in descriptors.type],
    )
    field = scanned.schema.field("blob").with_type(descriptors.type)
    source = BlobSourceContext.from_dataset(dataset, "catalog/null-kind", "workers")
    with BlobRefCapture(
        dataset, source, dataset.lance_schema.field("blob").id()
    ) as capture:
        before = dataset.io_stats_snapshot()
        nested = capture.capture_nested(source, field, scanned["_rowaddr"], descriptors)
        assert dataset.io_stats_snapshot().read_iops == before.read_iops
    assert nested.references.array.to_pylist() == [None, None]
    assert nested.indices.column(0).to_pylist() == [None, None]


@pytest.mark.parametrize("offset", [None, 0, 1])
@pytest.mark.parametrize("sort_column", ["id", "blob"])
def test_blob_zero_limit_sort(tmp_path, offset, sort_column):
    dataset = lance.write_dataset(
        pa.table({"id": [1, 0], "blob": lance.blob_array([b"a", b"b"])}),
        tmp_path / "zero-limit.lance",
        data_storage_version="2.2",
    )
    before = dataset.io_stats_snapshot()
    result = dataset.to_table(
        columns=["id"],
        order_by=[sort_column],
        blob_handling="all_binary",
        limit=0,
        offset=offset,
    )
    assert result.schema == pa.schema([pa.field("id", pa.int64())])
    assert result.num_rows == 0
    assert dataset.io_stats_snapshot().read_iops == before.read_iops


def test_portable_blob_reference_requests(portable_blob_references):
    dataset, source, _, references = portable_blob_references
    assert BlobSourceContext.protocol_version() == 1
    assert BlobSourceContext.supported_capabilities() == [1, 2]
    assert "snapshot_retention_v1" in BlobSourceContext.supported_operations()
    assert source.table_identity == "catalog/portable"
    assert source.access_context_key == "workers"
    snapshot = lance.SnapshotIdentity.from_dataset(dataset, "catalog/portable")
    assert source.snapshot_identity().encode() == snapshot.encode()
    assert source.branch is None
    assert BlobSourceContext.decode(source.encode()).encode() == source.encode()
    references = BlobRefBatch.decode(references.encode())
    selected = references.array.take(pa.array([2, 0, 1, 3, 4, 2]))
    selected = BlobRefBatch.from_arrow(references.contexts, references.field, selected)
    selected = selected.remap_contexts([BlobSourceContext.decode(source.encode())])
    assert len(selected) == 6
    worker = lance.dataset(dataset.uri, version=source.version)
    with BlobRefReader(
        [(source, worker)],
        io_buffer_size_bytes=1024,
        materialization_readahead_bytes=1024,
    ) as reader:
        with reader.read(
            selected,
            list(range(6)),
            [(1 << 64) - 1, 0, 1, 2, 3, 4],
            [None, None, (1, 3), None, None, (2, 4)],
        ) as stream:
            results = {}
            for batch in stream:
                results[batch["request_index"][0].as_py()] = batch["data"][0].as_py()
                del batch
        assert results == {
            (1 << 64) - 1: b"abcdefghij",
            0: b"abc",
            1: b"bcd",
            2: None,
            3: b"",
            4: b"cdef",
        }
        assert reader.reserved_bytes == 0
        assert reader.io_reserved_bytes == 0
        metrics = reader.metrics
        assert metrics["descriptor_requests"] == 0
        assert metrics["metadata_requests"] == 0
        assert metrics["requested_logical_bytes"] == 20
        assert metrics["returned_logical_bytes"] == 20
        assert metrics["payload_bytes"] > 0
        assert metrics["peak_reserved_bytes"] > 0


def _nested_reference_scan(tmp_path, large_list):
    fields = [
        lance.blob_field(
            "blob.dot", inline_size_threshold=0, dedicated_size_threshold=1
        ),
        lance.blob_field(
            "sibling", inline_size_threshold=0, dedicated_size_threshold=1
        ),
    ]
    items = pa.StructArray.from_arrays(
        [
            lance.blob_array(
                [
                    b"first",
                    None,
                    b"",
                    b"hidden-child",
                    b"hidden-parent",
                    b"second",
                    None,
                ]
            ),
            lance.blob_array([b"unrelated"] * 7),
        ],
        fields=fields,
        mask=pa.array([False, False, False, True, False, False, False]),
    )
    list_type = pa.LargeListArray if large_list else pa.ListArray
    lists = list_type.from_arrays(
        pa.array(
            [0, 3, 3, 3, 4, 5, 6, 7], type=pa.int64() if large_list else pa.int32()
        ),
        items,
        mask=pa.array([False, False, True, False, False, False, False]),
    )
    root = pa.StructArray.from_arrays(
        [lists],
        names=["clips.dot"],
        mask=pa.array([False, False, False, False, True, False, False]),
    )
    dataset = lance.write_dataset(
        pa.table({"media": root}),
        tmp_path / "nested.lance",
        max_rows_per_file=3,
        data_storage_version="2.2",
    )
    source = BlobSourceContext.from_dataset(dataset, "catalog/nested", "workers")
    scanned = dataset.to_table(with_row_address=True).combine_chunks().to_batches()[0]
    scanned = scanned.take(pa.array([5, 0, 4, 3, 1, 2, 6, 0]))
    leaf = dataset.lance_schema.fields()[0].children()[0].children()[0].children()[0]
    return dataset, source, leaf.id(), scanned


@pytest.mark.parametrize("large_list", [False, True], ids=["list", "large-list"])
def test_portable_nested_capture(tmp_path, large_list):
    dataset, source, field_id, scanned = _nested_reference_scan(tmp_path, large_list)
    with BlobRefCapture(dataset, source, field_id) as capture:
        before = dataset.io_stats_snapshot()
        nested = capture.capture_nested(
            source, scanned.schema.field("media"), scanned["_rowaddr"], scanned["media"]
        )
        assert dataset.io_stats_snapshot().read_iops == before.read_iops
        with pytest.raises(ValueError, match="capture_nested"):
            capture.capture(
                source,
                scanned.schema.field("media"),
                scanned["_rowaddr"],
                scanned["media"],
            )
    nested = NestedBlobRefBatch.decode(nested.encode())
    nested = NestedBlobRefBatch.from_arrow(
        nested.references.remap_contexts([source]), nested.indices
    )
    with pytest.raises(ValueError, match="exceeds reference count"):
        NestedBlobRefBatch.from_arrow(
            nested.references,
            pa.record_batch(
                [pa.array([(1 << 64) - 1], type=pa.uint64())], names=["invalid"]
            ),
        )
    indices = nested.indices.column(0).to_pylist()
    assert indices[2] is None
    assert indices[4] == {"clips.dot": []}
    assert indices[5] == {"clips.dot": None}
    assert indices[3] == {"clips.dot": [None]}
    assert indices[6] == {"clips.dot": [{"blob.dot": None}]}
    refs = nested.references
    selected = [
        indices[0]["clips.dot"][0]["blob.dot"],
        indices[1]["clips.dot"][0]["blob.dot"],
        indices[1]["clips.dot"][2]["blob.dot"],
        indices[7]["clips.dot"][0]["blob.dot"],
    ]
    with BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=1024,
        materialization_readahead_bytes=1024,
    ) as reader:
        with reader.read(refs, selected, list(range(4))) as stream:
            results = {}
            for batch in stream:
                results[batch["request_index"][0].as_py()] = batch["data"][0].as_py()
                del batch
        assert results == {0: b"second", 1: b"first", 2: b"", 3: b"first"}
        assert reader.metrics["payload_requests"] == 2
        assert reader.metrics["payload_bytes"] == 11
        assert reader.metrics["duplicate_demands"] == 1


def test_portable_shared_conformance_fixture(portable_blob_references):
    dataset, source, _, refs = portable_blob_references
    fixture = json.loads(
        (
            Path(__file__).parents[3] / "test_data/blob_reference_v1/conformance.json"
        ).read_text()
    )
    requests = fixture["requests"]
    assert fixture["protocol_version"] == BlobSourceContext.protocol_version()
    with BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=1024,
        materialization_readahead_bytes=1024,
    ) as reader:
        with reader.read(
            refs,
            [request["reference_index"] for request in requests],
            [int(request["request_index"]) for request in requests],
            [
                tuple(request["range"]) if request["range"] is not None else None
                for request in requests
            ],
        ) as stream:
            results = {}
            for batch in stream:
                results[batch["request_index"][0].as_py()] = batch["data"][0].as_py()
                del batch
        assert results == {
            int(request["request_index"]): request["expected"].encode()
            if request["expected"] is not None
            else None
            for request in requests
        }
        for bounds in fixture["invalid_ranges"]:
            with pytest.raises(ValueError, match="range"):
                reader.read(
                    refs, [2], [0], [(int(bounds["offset"]), int(bounds["length"]))]
                )
        assert reader.metrics["duplicate_demands"] == 1


@pytest.mark.parametrize("large_list", [False, True], ids=["list", "large-list"])
@pytest.mark.parametrize("start,length", [(0, 0), (1, 2), (4, 0), (7, 1), (8, 0)])
def test_portable_nested_capture_slices(tmp_path, large_list, start, length):
    dataset, source, field_id, scanned = _nested_reference_scan(tmp_path, large_list)
    expected = [
        {"clips.dot": [{"blob.dot": b"second"}]},
        {"clips.dot": [{"blob.dot": b"first"}, {"blob.dot": None}, {"blob.dot": b""}]},
        None,
        {"clips.dot": [None]},
        {"clips.dot": []},
        {"clips.dot": None},
        {"clips.dot": [{"blob.dot": None}]},
        {"clips.dot": [{"blob.dot": b"first"}, {"blob.dot": None}, {"blob.dot": b""}]},
    ][start : start + length]
    scanned = scanned.slice(start, length)
    with BlobRefCapture(dataset, source, field_id) as capture:
        before = dataset.io_stats_snapshot()
        nested = capture.capture_nested(
            source, scanned.schema.field("media"), scanned["_rowaddr"], scanned["media"]
        )
        assert dataset.io_stats_snapshot().read_iops == before.read_iops
    nested = NestedBlobRefBatch.decode(nested.encode())
    refs = nested.references
    with BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=1024,
        materialization_readahead_bytes=1024,
    ) as reader:
        values = {}
        with reader.read(
            refs, list(range(len(refs))), list(range(len(refs)))
        ) as stream:
            for batch in stream:
                values[batch["request_index"][0].as_py()] = batch["data"][0].as_py()
                del batch
    actual = nested.indices.column(0).to_pylist()
    for root in actual:
        if root is not None and root["clips.dot"] is not None:
            for item in root["clips.dot"]:
                if item is not None:
                    index = item["blob.dot"]
                    item["blob.dot"] = None if index is None else values[index]
    assert actual == expected
    lists = nested.indices.column(0).field("clips.dot")
    assert lists.offsets[0].as_py() == 0


@pytest.mark.parametrize(
    "array,message",
    [
        (pa.array([0], type=pa.int32()), "Unsupported nested blob index layout"),
        (pa.array([6], type=pa.uint64()), "exceeds reference count"),
        (pa.array([[6]], type=pa.list_(pa.uint64())), "exceeds reference count"),
        (
            pa.array(
                [{"a": 0, "b": 0}],
                type=pa.struct([("a", pa.uint64()), ("b", pa.uint64())]),
            ),
            "Unsupported nested blob index layout",
        ),
        (
            pa.array([[0]], type=pa.list_(pa.uint64(), 1)),
            "Unsupported nested blob index layout",
        ),
    ],
)
def test_portable_nested_mapping_validation(portable_blob_references, array, message):
    _, _, _, refs = portable_blob_references
    with pytest.raises(ValueError, match=message):
        NestedBlobRefBatch.from_arrow(refs, pa.record_batch([array], names=["indices"]))


def test_portable_nested_null_mapping_without_references(portable_blob_references):
    _, _, _, refs = portable_blob_references
    refs = BlobRefBatch.from_arrow(refs.contexts, refs.field, refs.array.slice(0, 0))
    indices = pa.record_batch(
        [pa.array([None, None], type=pa.uint64())], names=["indices"]
    )
    nested = NestedBlobRefBatch.decode(
        NestedBlobRefBatch.from_arrow(refs, indices).encode()
    )
    assert len(nested.references) == 0
    assert nested.indices == indices


@pytest.mark.parametrize(
    "offset,length,expected",
    [(0, 0, b""), (0, 10, b"abcdefghij"), (9, 1, b"j"), (10, 0, b"")],
)
def test_portable_range_boundaries(portable_blob_references, offset, length, expected):
    dataset, source, _, refs = portable_blob_references
    with BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=32,
        materialization_readahead_bytes=64,
    ) as reader:
        with reader.read(refs, [2], [0], [(offset, length)]) as stream:
            batch = next(stream)
            assert batch["data"].to_pylist() == [expected]
            del batch
            assert next(stream, None) is None
        assert reader.metrics["payload_bytes"] == length
        assert reader.metrics["metadata_requests"] == 0
        assert reader.reserved_bytes == 0


@pytest.mark.parametrize("offset,length", [(10, 1), (11, 0), ((1 << 64) - 1, 1)])
def test_portable_invalid_last_demand_is_atomic(
    portable_blob_references, offset, length
):
    dataset, source, _, refs = portable_blob_references
    with BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=32,
        materialization_readahead_bytes=64,
    ) as reader:
        with pytest.raises(ValueError, match="range"):
            reader.read(refs, [0, 2], [0, 1], [None, (offset, length)])
        assert reader.metrics["payload_requests"] == 0
        assert reader.metrics["metadata_requests"] == 0
        assert reader.reserved_bytes == 0


@pytest.mark.parametrize("cancel", [False, True], ids=["close", "cancel"])
def test_portable_unpolled_stream_termination(portable_blob_references, cancel):
    dataset, source, _, refs = portable_blob_references
    with BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=1,
        materialization_readahead_bytes=1,
    ) as reader:
        stream = reader.read(refs, [0], [0])
        if cancel:
            stream.cancel()
            with pytest.raises(OSError, match="cancel"):
                next(stream)
        else:
            stream.close()
        assert next(stream, None) is None
        stream.close()
        assert reader.metrics["payload_requests"] == 0
        assert reader.metrics["metadata_requests"] == 0
        assert reader.reserved_bytes == 0


def test_portable_factory_close_preserves_stream_and_slice(portable_blob_references):
    dataset, source, _, refs = portable_blob_references
    reader = BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=4,
        materialization_readahead_bytes=8,
    )
    stream = reader.read(refs, [0, 1], [0, 1])
    reader.close()
    with pytest.raises(ValueError, match="closed"):
        reader.read(refs, [0], [0])
    first = next(stream)
    retained = first["data"].slice(0, 1)
    del first
    assert retained.to_pylist() == [b"abc"]
    assert reader.reserved_bytes > 0
    del retained
    second = next(stream)
    retained = second["data"].slice(0, 1)
    del second
    stream.close()
    assert retained.to_pylist() == [b"abcde"]
    assert reader.reserved_bytes > 0
    del retained
    assert reader.reserved_bytes == 0
    assert next(stream, None) is None


def test_portable_error_after_yield_preserves_output(tmp_path):
    healthy = tmp_path / "healthy.bin"
    missing = tmp_path / "missing.bin"
    healthy.write_bytes(b"good")
    missing.write_bytes(b"gone")
    dataset = lance.write_dataset(
        pa.table(
            {
                "blob": lance.blob_array(
                    [
                        Blob.from_uri(path.as_uri(), position=0, size=4)
                        for path in [healthy, missing]
                    ]
                )
            }
        ),
        tmp_path / "partial-error.lance",
        data_storage_version="2.2",
        allow_external_blob_outside_bases=True,
    )
    source = BlobSourceContext.from_dataset(
        dataset,
        "catalog/partial-error",
        "workers",
        assume_immutable_external=True,
    )
    scanned = dataset.to_table(with_row_address=True).combine_chunks().to_batches()[0]
    with BlobRefCapture(
        dataset, source, dataset.lance_schema.field("blob").id()
    ) as capture:
        refs = capture.capture(
            source, scanned.schema.field("blob"), scanned["_rowaddr"], scanned["blob"]
        )
    # Deliberately violate the external lifetime promise to exercise the error path.
    missing.unlink()
    with BlobRefReader(
        [(source, dataset)], io_buffer_size_bytes=4, materialization_readahead_bytes=16
    ) as reader:
        with reader.read(refs, [0, 1], [70, 71]) as stream:
            first = next(stream)
            with pytest.raises(OSError, match="request 71.*payload read failed"):
                next(stream)
            assert next(stream, None) is None
        assert first["data"].to_pylist() == [b"good"]
        assert reader.reserved_bytes > 0
        del first
        with reader.read(refs, [0], [72]) as stream:
            result = next(stream)
            assert result["data"].to_pylist() == [b"good"]
            del result
        assert reader.reserved_bytes == 0


@pytest.mark.parametrize("large_list", [False, True])
def test_portable_nested_combine_sources(tmp_path, large_list):
    captures = []
    bindings = []
    for name in ["first", "second"]:
        dataset, source, field_id, scanned = _nested_reference_scan(
            tmp_path / name, large_list
        )
        bindings.append((source, dataset))
        scanned = scanned.slice(0 if name == "first" else 1, 1)
        with BlobRefCapture(dataset, source, field_id) as capture:
            captures.append(
                capture.capture_nested(
                    source,
                    scanned.schema.field("media"),
                    scanned["_rowaddr"],
                    scanned["media"],
                )
            )
    contexts = [source for source, _ in bindings]
    arrays = [nested.references.remap_contexts(contexts).array for nested in captures]
    refs = BlobRefBatch.from_arrow(
        contexts, captures[0].references.field, pa.concat_arrays(arrays)
    )
    mappings = []
    offset = 0
    for nested in captures:
        rows = nested.indices.column(0).to_pylist()
        for row in rows:
            for item in row["clips.dot"]:
                if item is not None and item["blob.dot"] is not None:
                    item["blob.dot"] += offset
        mappings.extend(rows)
        offset += len(nested.references)
    indices = pa.record_batch(
        [pa.array(mappings, type=captures[0].indices.column(0).type)],
        schema=captures[0].indices.schema,
    ).take(pa.array([1, 0, 1]))
    nested = NestedBlobRefBatch.decode(
        NestedBlobRefBatch.from_arrow(refs, indices).encode()
    )
    selected = [
        row["clips.dot"][0]["blob.dot"] for row in nested.indices.column(0).to_pylist()
    ]
    assert selected == [1, 0, 1]
    assert [
        row["source_context"] for row in refs.array.to_pylist() if row is not None
    ] == [0, 1, 1]
    with BlobRefReader(
        bindings, io_buffer_size_bytes=1024, materialization_readahead_bytes=1024
    ) as reader:
        with reader.read(nested.references, selected, [0, 1, 2]) as stream:
            values = {}
            for batch in stream:
                values[batch["request_index"][0].as_py()] = batch["data"][0].as_py()
                del batch
        assert values == {0: b"first", 1: b"second", 2: b"first"}
        assert reader.metrics["payload_requests"] == 2
    # Independent slot zero must not silently select the first source.
    with pytest.raises(ValueError, match="does not match source_context"):
        BlobRefBatch.from_arrow(
            contexts,
            refs.field,
            pa.concat_arrays([item.references.array for item in captures]),
        )


@pytest.mark.parametrize("indexed", [False, True])
def test_blob_scalar_refinement_with_unindexed_tail(tmp_path, indexed):
    schema = pa.schema([pa.field("id", pa.int64()), lance.blob_field("blob")])
    dataset = lance.write_dataset(
        pa.Table.from_arrays(
            [pa.array([0, 1, 2, 3]), lance.blob_array([b"c", None, b"b", b"a"])],
            schema=schema,
        ),
        tmp_path / "refine.lance",
        data_storage_version="2.2",
        max_rows_per_file=2,
        enable_stable_row_ids=True,
    )
    if indexed:
        dataset.create_scalar_index("id", "BTREE")
    dataset = lance.write_dataset(
        pa.Table.from_arrays(
            [pa.array([4, 5]), lance.blob_array([b"", b"aa"])], schema=schema
        ),
        dataset.uri,
        mode="append",
        max_rows_per_file=2,
    )
    dataset.delete("id = 2")
    result = dataset.to_table(
        columns=["id"],
        filter="id >= 1 AND blob IS NOT NULL",
        order_by=["blob", "id"],
        blob_handling="all_binary",
        limit=2,
        offset=1,
    )
    assert result.to_pydict() == {"id": [3, 5]}


def test_portable_metadata_filter_top_k_capture(portable_blob_references):
    dataset, source, _, _ = portable_blob_references
    scanned = (
        dataset.to_table(
            columns=["blob"],
            filter="blob.size > 4",
            order_by=["blob.size"],
            limit=1,
            with_row_address=True,
            blob_handling="all_descriptions",
        )
        .combine_chunks()
        .to_batches()[0]
    )
    with BlobRefCapture(
        dataset, source, dataset.lance_schema.field("blob").id()
    ) as capture:
        before = dataset.io_stats_snapshot()
        refs = capture.capture(
            source, scanned.schema.field("blob"), scanned["_rowaddr"], scanned["blob"]
        )
        assert dataset.io_stats_snapshot().read_iops == before.read_iops
    with BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=1024,
        materialization_readahead_bytes=1024,
    ) as reader:
        with reader.read(refs, [0], [42]) as stream:
            result = next(stream)
            assert result["request_index"][0].as_py() == 42
            assert result["data"][0].as_py() == b"abcde"
            del result
            assert next(stream, None) is None
        assert reader.metrics["requested_logical_bytes"] == 5
        assert reader.metrics["payload_requests"] == 1


@pytest.mark.parametrize("with_row_address", [False, True])
def test_blob_scan_orders_payload_bytes(tmp_path, with_row_address):
    dataset = lance.write_dataset(
        pa.table(
            {"id": [0, 1, 2, 3], "blob": lance.blob_array([b"c", b"a", None, b"b"])}
        ),
        tmp_path / "sort.lance",
        data_storage_version="2.2",
        max_rows_per_file=2,
    )
    result = dataset.to_table(
        columns=["id"],
        order_by=["blob"],
        blob_handling="all_binary",
        with_row_address=with_row_address,
    )
    assert result["id"].to_pylist() == [1, 3, 0, 2]
    assert result.column_names == (["id", "_rowaddr"] if with_row_address else ["id"])
    with pytest.raises(ValueError, match="requires binary blob handling"):
        dataset.to_table(
            columns=["id"], order_by=["blob"], blob_handling="all_descriptions"
        )
    assert dataset.to_table(columns=["id"], order_by=["blob.size", "id"])[
        "id"
    ].to_pylist() == [0, 1, 3, 2]


@pytest.mark.parametrize("prefilter", [False, True], ids=["postfilter", "prefilter"])
@pytest.mark.parametrize("indexed", [False, True], ids=["flat", "indexed"])
def test_blob_vector_filter_preserves_binary_output(tmp_path, prefilter, indexed):
    vectors = pa.FixedSizeListArray.from_arrays(
        pa.array(
            [float(row) for row in range(32) for _ in range(2)], type=pa.float32()
        ),
        2,
    )
    payloads = [None if row % 3 == 1 else bytes([row]) for row in range(32)]
    dataset = lance.write_dataset(
        pa.table({"vector": vectors, "blob": lance.blob_array(payloads)}),
        tmp_path / "vectors.lance",
        data_storage_version="2.2",
        max_rows_per_file=16,
    )
    if indexed:
        dataset.create_index("vector", "IVF_FLAT", num_partitions=2)
    scanner = dataset.scanner(
        columns=["blob"],
        filter="blob IS NOT NULL",
        prefilter=prefilter,
        nearest={"column": "vector", "q": [0.0, 0.0], "k": 32, "nprobes": 2},
        blob_handling="all_binary",
    )
    if indexed:
        assert "ANNSubIndex" in scanner.explain_plan()
    result = scanner.to_table()
    assert result["blob"].type == pa.large_binary()
    assert sorted(result["blob"].to_pylist()) == sorted(
        value for value in payloads if value is not None
    )


@pytest.mark.parametrize(
    "indices, request_indices, ranges, message",
    [
        ([0], [], None, "counts differ"),
        ([99], [0], None, "index"),
        ([0, 0], [1, 1], None, "Duplicate"),
        ([0], [0], [(2, 2)], "range"),
        ([0], [0], [((1 << 64) - 1, 1)], "overflow"),
    ],
)
def test_portable_blob_reference_validation(
    portable_blob_references, indices, request_indices, ranges, message
):
    dataset, source, _, references = portable_blob_references
    with BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=1024,
        materialization_readahead_bytes=1024,
    ) as reader:
        with pytest.raises(ValueError, match=message):
            reader.read(references, indices, request_indices, ranges)
        assert reader.reserved_bytes == 0
    wire = bytearray(source.encode())
    wire[1] = 99
    with pytest.raises(ValueError, match="protocol version"):
        BlobSourceContext.decode(bytes(wire))


def test_portable_blob_reference_cancel_and_buffer_ownership(portable_blob_references):
    dataset, source, _, references = portable_blob_references
    with BlobRefReader(
        [(source, dataset)],
        io_buffer_size_bytes=1,
        materialization_readahead_bytes=1,
    ) as reader:
        stream = reader.read(references, [0, 1], [0, 1])
        first = next(stream)
        array = first["data"]
        del first
        assert array[0].as_py() == b"abc"
        assert reader.reserved_bytes >= 3
        stream.cancel()
        with pytest.raises(OSError, match="cancel"):
            next(stream)
        stream.close()
        stream.close()
        assert array[0].as_py() == b"abc"
        assert reader.reserved_bytes >= 3
        del array
        assert reader.reserved_bytes == 0
        assert next(stream, None) is None


def test_portable_blob_references_in_separate_process(
    portable_blob_references, tmp_path
):
    dataset, source, retention, references = portable_blob_references
    reference_path = tmp_path / "refs.bin"
    reference_path.write_bytes(references.encode())
    token_path = tmp_path / "retention.bin"
    token_path.write_bytes(retention.encode())
    dataset.delete("true")
    dataset.cleanup_old_versions(older_than=timedelta(0))
    script = textwrap.dedent("""
        import sys
        from pathlib import Path
        import lance
        refs = lance.BlobRefBatch.decode(Path(sys.argv[1]).read_bytes())
        source = refs.contexts[0]
        retention = lance.SnapshotRetention.decode(Path(sys.argv[2]).read_bytes())
        assert retention.snapshot.encode() == source.snapshot_identity().encode()
        worker = lance.dataset(source.table_uri, version=source.version)
        with lance.BlobRefReader([(source, worker)], io_buffer_size_bytes=1024,
                                 materialization_readahead_bytes=1024) as reader:
            with reader.read(refs, [2, 0, 3, 4], [(1 << 64) - 1, 1, 2, 3],
                             [(2, 4), None, None, None]) as stream:
                values = {}
                for batch in stream:
                    values[batch['request_index'][0].as_py()] = batch['data'][0].as_py()
                    del batch
            assert values == {(1 << 64) - 1: b'cdef', 1: b'abc', 2: None, 3: b''}
            assert reader.reserved_bytes == 0
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script, str(reference_path), str(token_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    retention.release(dataset)
    retention.release(dataset)
    stats = dataset.cleanup_old_versions(older_than=timedelta(0))
    assert stats.data_files_removed >= 2
