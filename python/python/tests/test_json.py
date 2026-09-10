# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
import tempfile
from pathlib import Path
from typing import Union

import lance
import pyarrow as pa
import pytest


def check_json_type(ds: Union[lance.LanceDataset, pa.Table], col_name: str):
    # TODO: In the future it should be possible to verify
    # the logical type of a column.

    schema = ds.schema
    field = schema.field(col_name)
    assert field.type == pa.json_()


def check_nested_json_type(ds: Union[lance.LanceDataset, pa.Table]):
    media_field = ds.schema.field("media")
    extra_field = media_field.type.value_type.field("extra")
    assert extra_field.type == pa.json_()


def nested_extra_value(table: pa.Table):
    return table.to_pylist()[0]["media"][0]["extra"]


def test_json_basic_write_read():
    """Test basic JSON type write and read functionality."""

    # Create test data with JSON strings
    json_data = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "San Francisco"},
        {"name": "Charlie", "age": 35, "city": "Chicago"},
        None,  # Test null handling
        {"nested": {"key": "value", "list": [1, 2, 3]}},
    ]

    json_strings = [json.dumps(d) if d is not None else None for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())
    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "data": json_arr,
        }
    )

    # Write to Lance dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "json_test.lance"

        # Write the dataset
        lance.write_dataset(table, dataset_path)

        # Read back the dataset
        dataset = lance.dataset(dataset_path)

        # Verify logical schema exposed to users
        logical_schema = dataset.schema
        assert len(logical_schema) == 2
        assert logical_schema.field("id").type == pa.int32()
        check_json_type(dataset, "data")

        # Read data back
        result_table = dataset.to_table()

        # Check that data is returned as Arrow JSON for Python
        check_json_type(result_table, "data")

        # Verify data
        assert result_table.num_rows == 5
        assert result_table.column("id").to_pylist() == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("version", ["2.1", "2.2", "2.3"])
def test_nested_json_large_binary_cast_roundtrip(tmp_path: Path, version: str):
    storage_schema = pa.schema(
        [
            pa.field(
                "media",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("uri", pa.utf8(), nullable=False),
                            pa.field("extra", pa.large_binary()),
                        ]
                    )
                ),
            )
        ]
    )
    extension_schema = pa.schema(
        [
            pa.field(
                "media",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("uri", pa.utf8(), nullable=False),
                            pa.field("extra", pa.json_()),
                        ]
                    )
                ),
            )
        ]
    )
    table = pa.Table.from_pylist(
        [
            {
                "media": [
                    {
                        "uri": "s3://bucket/a.mp4",
                        "extra": b'{"codec": "h264"}',
                    }
                ]
            }
        ],
        schema=storage_schema,
    ).cast(extension_schema)

    dataset_path = tmp_path / f"nested_json_{version.replace('.', '_')}.lance"
    lance.write_dataset(table, dataset_path, data_storage_version=version)
    dataset = lance.dataset(dataset_path)

    check_nested_json_type(dataset)

    scanned = dataset.to_table()
    check_nested_json_type(scanned)
    assert json.loads(nested_extra_value(scanned)) == {"codec": "h264"}

    taken = dataset.take([0])
    check_nested_json_type(taken)
    assert json.loads(nested_extra_value(taken)) == {"codec": "h264"}

    empty = dataset.take([])
    assert empty.num_rows == 0
    check_nested_json_type(empty)


def test_json_with_other_types():
    """Test JSON type alongside other data types."""

    # Create mixed type data
    json_data = [
        {"product": "laptop", "specs": {"cpu": "i7", "ram": 16}},
        {"product": "phone", "specs": {"screen": "6.1", "battery": 4000}},
    ]

    json_strings = [json.dumps(d) for d in json_data]

    # Create JSON array using PyArrow's JSON type
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int64()),
            "name": pa.array(["Product A", "Product B"], type=pa.string()),
            "price": pa.array([999.99, 599.99], type=pa.float64()),
            "metadata": json_arr,
            "in_stock": pa.array([True, False], type=pa.bool_()),
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "mixed_types.lance"

        # Write and read the dataset
        lance.write_dataset(table, dataset_path)
        dataset = lance.dataset(dataset_path)

        # Verify all fields are preserved
        assert len(dataset.schema) == 5

        result = dataset.to_table()
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [1, 2]
        assert result.column("name").to_pylist() == ["Product A", "Product B"]
        assert result.column("in_stock").to_pylist() == [True, False]


def test_json_null_handling():
    """Test handling of null JSON values."""

    # Mix of valid JSON and nulls
    json_strings = [
        json.dumps({"key": "value1"}),
        None,
        json.dumps({"key": "value2"}),
        None,
        json.dumps({"key": "value3"}),
    ]

    # Create JSON array with nulls using PyArrow's JSON type
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table({"id": pa.array(range(5)), "optional_data": json_arr})

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "null_test.lance"
        lance.write_dataset(table, dataset_path)
        dataset = lance.dataset(dataset_path)

        result = dataset.to_table()
        assert result.num_rows == 5

        # Check null mask
        data_column = result.column("optional_data")
        assert data_column.null_count == 2
        assert data_column.is_null().to_pylist() == [False, True, False, True, False]


def test_json_batch_operations():
    """Test batch operations with JSON data."""

    # Create multiple batches
    batch_size = 1000
    num_batches = 5

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "batch_test.lance"

        for batch_num in range(num_batches):
            json_data = [
                json.dumps({"batch": batch_num, "item": i}) for i in range(batch_size)
            ]

            # Create JSON array using PyArrow's JSON type
            json_arr = pa.array(json_data, type=pa.json_())

            table = pa.table(
                {
                    "id": pa.array(
                        range(batch_num * batch_size, (batch_num + 1) * batch_size)
                    ),
                    "batch_data": json_arr,
                }
            )

            if batch_num == 0:
                lance.write_dataset(table, dataset_path)
            else:
                lance.write_dataset(table, dataset_path, mode="append")

        # Verify all batches were written
        dataset = lance.dataset(dataset_path)
        assert dataset.count_rows() == batch_size * num_batches

        # Test batch reading
        batches = list(dataset.to_batches(batch_size=batch_size))
        assert len(batches) == num_batches

        for batch in batches:
            assert batch.num_rows == batch_size


def test_json_path_queries():
    """Test JSON path queries using json_extract and json_exists."""
    # Create test data with JSON columns
    json_data = [
        {"user": {"name": "Alice", "age": 30}, "tags": ["python", "ml"]},
        {"user": {"name": "Bob", "age": 25}, "tags": ["rust", "db"]},
        {"user": {"name": "Charlie"}, "tags": []},
        None,
    ]

    json_strings = [json.dumps(d) if d is not None else None for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())

    # Create a Lance dataset with JSON data
    table = pa.table(
        {
            "id": [1, 2, 3, 4],
            "data": json_arr,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "json_test.lance"
        lance.write_dataset(table, ds_path)
        dataset = lance.dataset(ds_path)

        # Test json_extract
        result = dataset.to_table(
            filter="json_extract(data, '$.user.name') = '\"Alice\"'"
        )
        sql = (
            dataset.sql(
                "SELECT * FROM dataset WHERE "
                "json_extract(data, '$.user.name') = '\"Alice\"'"
            )
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Test json_exists
        result = dataset.to_table(filter="json_exists(data, '$.user.age')")
        assert result.num_rows == 2  # Alice and Bob have age field

        # Test json_array_contains
        result = dataset.to_table(
            filter="json_array_contains(data, '$.tags', 'python')"
        )
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1


def test_json_get_functions():
    """Test json_get_* functions for type-safe value extraction."""
    # Create test data with various JSON types
    json_data = [
        {"name": "Alice", "age": 30, "active": True, "score": 95.5},
        {"name": "Bob", "age": 25, "active": False, "score": 87.3},
        {"name": "Charlie", "age": "35", "active": "true", "score": "92"},
        {"name": "David"},  # Missing fields
    ]

    json_strings = [json.dumps(d) for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table(
        {
            "id": [1, 2, 3, 4],
            "data": json_arr,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "json_get_test.lance"
        lance.write_dataset(table, ds_path)
        dataset = lance.dataset(ds_path)

        # Test json_get_string
        result = dataset.to_table(filter="json_get_string(data, 'name') = 'Alice'")
        sql = (
            dataset.sql(
                "SELECT * FROM dataset WHERE json_get_string(data, 'name') = 'Alice'"
            )
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Test json_get_int with type coercion
        result = dataset.to_table(filter="json_get_int(data, 'age') > 28")
        sql = (
            dataset.sql("SELECT * FROM dataset WHERE json_get_int(data, 'age') > 28")
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 2  # Alice (30) and Charlie ("35" -> 35)

        # Test json_get_bool with type coercion
        result = dataset.to_table(filter="json_get_bool(data, 'active') = true")
        sql = (
            dataset.sql(
                "SELECT * FROM dataset WHERE json_get_bool(data, 'active') = true"
            )
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 2  # Alice (true) and Charlie ("true" -> true)

        # Test json_get_float
        result = dataset.to_table(filter="json_get_float(data, 'score') > 90")
        sql = (
            dataset.sql(
                "SELECT * FROM dataset WHERE json_get_float(data, 'score') > 90"
            )
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 2  # Alice (95.5) and Charlie ("92" -> 92.0)


def test_nested_json_access():
    """Test accessing nested JSON structures."""
    json_data = [
        {"user": {"profile": {"name": "Alice", "settings": {"theme": "dark"}}}},
        {"user": {"profile": {"name": "Bob", "settings": {"theme": "light"}}}},
    ]

    json_strings = [json.dumps(d) for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table(
        {
            "id": [1, 2],
            "data": json_arr,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "nested_json_test.lance"
        lance.write_dataset(table, ds_path)
        dataset = lance.dataset(ds_path)

        # Access nested fields using json_get recursively
        # First get user, then profile, then name
        result = dataset.to_table(
            filter="""
                json_get_string(
                    json_get(
                        json_get(data, 'user'),
                        'profile'),
                    'name')
                = 'Alice'"""
        )
        sql = (
            dataset.sql(
                "SELECT * FROM dataset WHERE "
                "json_get_string("
                "json_get(json_get(data, 'user'), 'profile'), "
                "'name') = 'Alice'"
            )
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Or use JSONPath for deep access
        result = dataset.to_table(
            filter="json_extract(data, '$.user.profile.settings.theme') = '\"dark\"'"
        )
        sql = (
            dataset.sql(
                "SELECT * FROM dataset WHERE "
                "json_extract(data, '$.user.profile.settings.theme') = '\"dark\"'"
            )
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1


def test_json_array_operations():
    """Test JSON array operations."""
    json_data = [
        {"items": ["apple", "banana", "orange"], "counts": [1, 2, 3, 4, 5]},
        {"items": ["grape", "melon"], "counts": [10, 20]},
        {"items": [], "counts": []},
    ]

    json_strings = [json.dumps(d) for d in json_data]
    json_arr = pa.array(json_strings, type=pa.json_())

    table = pa.table(
        {
            "id": [1, 2, 3],
            "data": json_arr,
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ds_path = Path(tmpdir) / "array_json_test.lance"
        lance.write_dataset(table, ds_path)
        dataset = lance.dataset(ds_path)

        # Test array contains
        result = dataset.to_table(
            filter="json_array_contains(data, '$.items', 'apple')"
        )
        sql = (
            dataset.sql(
                "SELECT * FROM dataset WHERE "
                "json_array_contains(data, '$.items', 'apple')"
            )
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Test array length
        result = dataset.to_table(filter="json_array_length(data, '$.counts') > 3")
        sql = (
            dataset.sql(
                "SELECT * FROM dataset WHERE json_array_length(data, '$.counts') > 3"
            )
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 1

        # Test empty array
        result = dataset.to_table(filter="json_array_length(data, '$.items') = 0")
        sql = (
            dataset.sql(
                "SELECT * FROM dataset WHERE json_array_length(data, '$.items') = 0"
            )
            .build()
            .to_batch_records()
        )
        sql_result = pa.Table.from_batches(sql)
        assert result == sql_result
        assert result.num_rows == 1
        assert result["id"][0].as_py() == 3


def test_json_filter_append_missing_json_cast(tmp_path: Path):
    """Ensure appending via dataset.schema keeps JSON columns valid."""

    dataset_path = tmp_path / "json_append_issue.lance"

    initial_table = pa.table(
        {
            "article_metadata": pa.array(
                [json.dumps({"article_journal": "Cell"})], type=pa.json_()
            ),
            "article_journal": pa.array(["Cell"], type=pa.string()),
        }
    )

    lance.write_dataset(initial_table, dataset_path)
    dataset = lance.dataset(dataset_path)
    schema = dataset.schema
    check_json_type(dataset, "article_metadata")

    append_table = pa.table(
        {
            "article_metadata": pa.array(
                [
                    json.dumps({"article_journal": "PLoS One"}),
                    json.dumps({"article_journal": "Nature"}),
                ],
                type=pa.json_(),
            ),
            "article_journal": pa.array(["PLoS One", "Nature"], type=pa.string()),
        }
    )

    append_cast = append_table.cast(schema)
    first_value = append_cast.column("article_metadata").to_pylist()[0]
    assert isinstance(first_value, str)

    lance.write_dataset(append_cast, dataset_path, mode="append")
    dataset = lance.dataset(dataset_path)
    assert dataset.count_rows() == 3

    result = dataset.to_table(
        filter="json_get(article_metadata, 'article_journal') IS NOT NULL"
    )
    sql = (
        dataset.sql(
            "SELECT * FROM dataset WHERE "
            "json_get(article_metadata, 'article_journal') IS NOT NULL"
        )
        .build()
        .to_batch_records()
    )
    sql_result = pa.Table.from_batches(sql)

    assert result == sql_result
    assert result.num_rows == 3
    assert result.column("article_journal").to_pylist() == [
        "Cell",
        "PLoS One",
        "Nature",
    ]


def test_json_with_compaction(tmp_path: Path):
    """Test that JSON data survives compaction across fragments."""

    dataset_path = tmp_path / "json_compaction.lance"

    # Write first fragment
    table1 = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "data": pa.array(
                [
                    json.dumps({"name": "Alice", "score": 10}),
                    json.dumps({"name": "Bob", "score": 20}),
                    json.dumps({"name": "Charlie", "score": 30}),
                ],
                type=pa.json_(),
            ),
        }
    )
    lance.write_dataset(table1, dataset_path)

    # Write second fragment
    table2 = pa.table(
        {
            "id": pa.array([4, 5], type=pa.int32()),
            "data": pa.array(
                [
                    json.dumps({"name": "David", "score": 40}),
                    json.dumps({"name": "Eve", "score": 50}),
                ],
                type=pa.json_(),
            ),
        }
    )
    lance.write_dataset(table2, dataset_path, mode="append")

    dataset = lance.dataset(dataset_path)
    assert len(dataset.get_fragments()) == 2

    # Run compaction
    dataset.optimize.compact_files()
    dataset = lance.dataset(dataset_path)
    assert len(dataset.get_fragments()) == 1

    # Verify data is intact
    result = dataset.to_table()
    assert result.num_rows == 5
    assert result.column("id").to_pylist() == [1, 2, 3, 4, 5]

    # Verify JSON type is preserved
    check_json_type(dataset, "data")

    # Verify JSON functions still work after compaction
    result = dataset.to_table(filter="json_get_string(data, 'name') = 'Alice'")
    assert result.num_rows == 1
    assert result["id"][0].as_py() == 1

    result = dataset.to_table(filter="json_get_int(data, 'score') > 25")
    assert result.num_rows == 3
    assert result["id"].to_pylist() == [3, 4, 5]


def test_json_limit_offset_batch_transfer_preserves_extension_metadata(tmp_path: Path):
    """Ensure JSON extension metadata survives limit/offset scans.

    This covers recreating a table by reading a source dataset in chunks and
    appending each chunk into a new dataset.
    """

    source_path = tmp_path / "json_source.lance"
    dest_path = tmp_path / "json_dest.lance"

    num_rows = 25
    batch_size = 10

    table = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int32()),
            "meta": pa.array(
                [json.dumps({"i": i}) for i in range(num_rows)], type=pa.json_()
            ),
        }
    )

    lance.write_dataset(table, source_path)
    source = lance.dataset(source_path)

    first_batch = source.to_table(limit=batch_size)
    meta_field = first_batch.schema.field("meta")
    assert (
        str(meta_field.type) == "extension<arrow.json>" or meta_field.type == pa.utf8()
    )

    lance.write_dataset(first_batch, dest_path, mode="overwrite")

    offset = batch_size
    while True:
        batch = source.to_table(limit=batch_size, offset=offset)
        if batch.num_rows == 0:
            break

        assert batch.schema == first_batch.schema
        meta_field = batch.schema.field("meta")
        assert (
            str(meta_field.type) == "extension<arrow.json>"
            or meta_field.type == pa.utf8()
        )

        lance.write_dataset(batch, dest_path, mode="append")
        offset += batch_size

    dest = lance.dataset(dest_path)
    assert dest.count_rows() == num_rows

    # Ensure JSON functions still recognize the column as JSON.
    assert dest.to_table(filter="json_get(meta, 'i') IS NOT NULL").num_rows == num_rows


def test_json_append(tmp_path: Path):
    """Test appending JSON data to an existing dataset."""

    dataset_path = tmp_path / "json_append.lance"

    # Write initial data
    table1 = pa.table(
        {
            "id": pa.array([1, 2], type=pa.int32()),
            "data": pa.array(
                [
                    json.dumps({"color": "red", "count": 1}),
                    json.dumps({"color": "blue", "count": 2}),
                ],
                type=pa.json_(),
            ),
        }
    )
    lance.write_dataset(table1, dataset_path)

    # Append more data
    table2 = pa.table(
        {
            "id": pa.array([3, 4, 5], type=pa.int32()),
            "data": pa.array(
                [
                    json.dumps({"color": "green", "count": 3}),
                    json.dumps({"color": "yellow", "count": 4}),
                    None,
                ],
                type=pa.json_(),
            ),
        }
    )
    lance.write_dataset(table2, dataset_path, mode="append")

    dataset = lance.dataset(dataset_path)
    assert dataset.count_rows() == 5

    # Verify JSON type is preserved
    check_json_type(dataset, "data")

    # Verify all data is readable
    result = dataset.to_table()
    assert result.column("id").to_pylist() == [1, 2, 3, 4, 5]

    # Verify null handling
    data_col = result.column("data")
    assert data_col.null_count == 1
    assert data_col.is_null().to_pylist() == [False, False, False, False, True]

    # Verify JSON functions work across both fragments
    result = dataset.to_table(filter="json_get_string(data, 'color') = 'green'")
    assert result.num_rows == 1
    assert result["id"][0].as_py() == 3

    result = dataset.to_table(filter="json_get_int(data, 'count') >= 2")
    assert result.num_rows == 3
    assert result["id"].to_pylist() == [2, 3, 4]


def test_json_add_columns(tmp_path: Path):
    """Test adding a JSON column to an existing dataset via add_columns."""

    dataset_path = tmp_path / "json_add_col.lance"

    # Create a dataset without a JSON column
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "name": pa.array(["Alice", "Bob", "Charlie"], type=pa.string()),
        }
    )
    dataset = lance.write_dataset(table, dataset_path)

    # Add a JSON column using a record batch reader
    names = table.column("name").to_pylist()
    json_values = [json.dumps({"greeting": f"hello {n}"}) for n in names]
    new_col = pa.record_batch([pa.array(json_values, type=pa.json_())], ["metadata"])
    reader_schema = pa.schema([pa.field("metadata", pa.json_())])

    dataset.add_columns(iter([new_col]), reader_schema=reader_schema)
    dataset = lance.dataset(dataset_path)

    # Verify the new column exists and has the right type
    assert dataset.schema.names == ["id", "name", "metadata"]
    check_json_type(dataset, "metadata")

    # Verify data round-trips
    result = dataset.to_table()
    assert result.num_rows == 3
    metadata_values = result.column("metadata").to_pylist()
    for name, val in zip(names, metadata_values):
        assert json.loads(val) == {"greeting": f"hello {name}"}

    result = dataset.to_table(
        filter="json_get_string(metadata, 'greeting') = 'hello Alice'"
    )
    assert result.num_rows == 1
    assert result["id"][0].as_py() == 1


def test_json_merge_insert(tmp_path: Path):
    """Test merge_insert with JSON data."""

    dataset_path = tmp_path / "json_merge_insert.lance"

    # Create initial dataset
    table = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int32()),
            "data": pa.array(
                [
                    json.dumps({"name": "Alice", "score": 10}),
                    json.dumps({"name": "Bob", "score": 20}),
                    json.dumps({"name": "Charlie", "score": 30}),
                ],
                type=pa.json_(),
            ),
        }
    )
    lance.write_dataset(table, dataset_path)

    # Merge insert: update id=2, insert id=4
    new_data = pa.table(
        {
            "id": pa.array([2, 4], type=pa.int32()),
            "data": pa.array(
                [
                    json.dumps({"name": "Bob", "score": 99}),
                    json.dumps({"name": "David", "score": 40}),
                ],
                type=pa.json_(),
            ),
        }
    )

    dataset = lance.dataset(dataset_path)
    dataset.merge_insert(
        "id"
    ).when_matched_update_all().when_not_matched_insert_all().execute(new_data)
    dataset = lance.dataset(dataset_path)

    # Verify row count
    assert dataset.count_rows() == 4

    # Verify JSON type preserved
    check_json_type(dataset, "data")

    # Verify data is readable
    result = dataset.to_table()
    assert sorted(result.column("id").to_pylist()) == [1, 2, 3, 4]

    result = dataset.to_table(filter="json_get_int(data, 'score') >= 35")
    assert result.num_rows == 2


def test_json_fragment_session_take(tmp_path: Path):
    """FragmentSession.take must return JSON columns as pa.json_(), not raw JSONB."""

    dataset_path = tmp_path / "json_fragment_session.lance"

    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int32()),
            "data": pa.array(
                [json.dumps({"name": "row", "score": v}) for v in range(5)],
                type=pa.json_(),
            ),
        }
    )
    lance.write_dataset(table, dataset_path)

    dataset = lance.dataset(dataset_path)
    fragment = dataset.get_fragments()[0]
    session = fragment.open_session(columns=["id", "data"])
    result = session.take([0, 2, 4])

    # JSON column must be the logical pa.json_() type, not raw LargeBinary JSONB.
    assert result.schema.field("data").type == pa.json_()
    assert result.column("id").to_pylist() == [1, 3, 5]
    assert [json.loads(v) for v in result.column("data").to_pylist()] == [
        {"name": "row", "score": 0},
        {"name": "row", "score": 2},
        {"name": "row", "score": 4},
    ]
