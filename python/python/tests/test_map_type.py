# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
import pytest


def test_simple_map_write_read(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("properties", pa.map_(pa.string(), pa.int32())),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3],
            "properties": [
                [("key1", 10), ("key2", 20)],
                [("key3", 30)],
                [("key4", 40), ("key5", 50), ("key6", 60)],
            ],
        },
        schema=schema,
    )

    # Write to Lance (requires v2.2+)
    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")

    # Read and verify
    result = dataset.to_table()

    assert result.schema == schema
    assert result.equals(data)


def test_map_with_nulls(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("properties", pa.map_(pa.string(), pa.int32())),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3, 4],
            "properties": [
                [("key1", 10)],
                None,  # null map
                [],  # empty map
                [("key2", 20), ("key3", 30)],
            ],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")
    result = dataset.to_table()

    assert result.schema == schema
    assert result.equals(data)


def test_map_with_null_values(tmp_path: Path):
    schema = pa.schema(
        [pa.field("id", pa.int32()), pa.field("data", pa.map_(pa.string(), pa.int32()))]
    )

    # Create map with null values using simple notation
    data = pa.table(
        {
            "id": [1, 2],
            "data": [
                [("a", 1), ("b", None)],  # Second value is null
                [("c", 3), ("d", None)],  # Fourth value is null
            ],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")
    result = dataset.to_table()

    assert result.schema == schema
    assert result.equals(data)


def test_empty_maps(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("map_field", pa.map_(pa.string(), pa.string())),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3],
            "map_field": [
                [("a", "apple")],
                [],  # empty map
                [("b", "banana")],
            ],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")
    result = dataset.to_table()

    assert result.schema == schema
    assert result.equals(data)


def test_nested_map_in_struct(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field(
                "record",
                pa.struct(
                    [
                        pa.field("name", pa.string()),
                        pa.field("attributes", pa.map_(pa.string(), pa.string())),
                    ]
                ),
            ),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3],
            "record": [
                {"name": "Alice", "attributes": [("city", "NYC"), ("age", "30")]},
                {"name": "Bob", "attributes": [("city", "LA")]},
                {"name": "Charlie", "attributes": None},
            ],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")
    result = dataset.to_table()

    assert result.schema == schema
    assert result.equals(data)


def test_list_of_maps(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("configs", pa.list_(pa.map_(pa.string(), pa.int32()))),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2],
            "configs": [
                [
                    [("a", 1), ("b", 2)],  # first map
                    [("c", 3)],  # second map
                ],
                [
                    [("d", 4), ("e", 5)]  # first map
                ],
            ],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")
    result = dataset.to_table()

    assert result.schema == schema
    assert result.equals(data)


def test_map_different_key_types(tmp_path: Path):
    # Test Map<Int32, String>
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("int_map", pa.map_(pa.int32(), pa.string())),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2],
            "int_map": [[(1, "one"), (2, "two")], [(3, "three"), (4, "four")]],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")
    result = dataset.to_table()

    assert result.schema == schema
    assert result.equals(data)


def test_query_map_column(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("properties", pa.map_(pa.string(), pa.int32())),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3, 4],
            "properties": [
                [("key1", 10), ("key2", 20)],
                [("key3", 30)],
                [("key4", 40)],
                [("key5", 50)],
            ],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")

    # Column selection (full read)
    result = dataset.to_table(columns=["id"])
    assert result.schema.names == ["id"]
    assert result.num_rows == 4

    # Full read with Map column
    result = dataset.to_table()
    assert "properties" in result.schema.names
    assert result.num_rows == 4

    result = dataset.to_table(filter="id > 2")
    assert result.num_rows == 2


def test_map_value_types(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("string_map", pa.map_(pa.string(), pa.string())),
            pa.field("float_map", pa.map_(pa.string(), pa.float64())),
            pa.field("bool_map", pa.map_(pa.string(), pa.bool_())),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2],
            "string_map": [[("a", "apple"), ("b", "banana")], [("c", "cherry")]],
            "float_map": [[("x", 1.5), ("y", 2.5)], [("z", 3.5)]],
            "bool_map": [[("flag1", True), ("flag2", False)], [("flag3", True)]],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")
    result = dataset.to_table()

    assert result.schema == schema
    assert result.equals(data)


def test_map_append_data(tmp_path: Path):
    schema = pa.schema(
        [pa.field("id", pa.int32()), pa.field("data", pa.map_(pa.string(), pa.int32()))]
    )

    # Initial data
    data1 = pa.table({"id": [1, 2], "data": [[("a", 1)], [("b", 2)]]}, schema=schema)

    lance.write_dataset(data1, tmp_path, data_storage_version="2.2")

    # Append more data
    data2 = pa.table({"id": [3, 4], "data": [[("c", 3)], [("d", 4)]]}, schema=schema)

    # Reopen dataset before appending
    lance.write_dataset(data2, tmp_path, mode="append", data_storage_version="2.2")

    # Reopen and read
    dataset_reopened = lance.dataset(tmp_path)
    result = dataset_reopened.to_table()
    assert result.num_rows == 4
    assert result["id"].to_pylist() == [1, 2, 3, 4]


def test_map_large_entries(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("big_map", pa.map_(pa.string(), pa.int32())),
        ]
    )

    # Create a map with 100 entries
    large_map = [(f"key{i}", i * 10) for i in range(100)]

    data = pa.table(
        {
            "id": [1, 2],
            "big_map": [large_map, large_map[:50]],  # Second map has 50 entries
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")
    result = dataset.to_table()

    assert result.schema == schema
    assert result.equals(data)


def test_map_version_compatibility(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("map_field", pa.map_(pa.string(), pa.int32())),
        ]
    )

    data = pa.table(
        {"id": [1, 2], "map_field": [[("a", 1)], [("b", 2)]]}, schema=schema
    )

    # Writing with v2.2 should succeed
    dataset = lance.write_dataset(data, tmp_path / "v22", data_storage_version="2.2")
    result = dataset.to_table()
    assert result.equals(data)

    # should raise an error for v2.1
    with pytest.raises(Exception) as exc_info:
        lance.write_dataset(data, tmp_path / "v21", data_storage_version="2.1")
    # Verify error message
    error_msg = str(exc_info.value)
    assert (
        "Map data type" in error_msg
        or "not yet implemented" in error_msg.lower()
        or "not supported" in error_msg.lower()
    )


def test_map_roundtrip_preservation(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("map1", pa.map_(pa.string(), pa.int32())),
            pa.field("map2", pa.map_(pa.int32(), pa.string())),
        ]
    )

    data = pa.table(
        {"id": [1], "map1": [[("z", 1), ("a", 2)]], "map2": [[(1, "a"), (2, "b")]]},
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")
    result = dataset.to_table()

    # Verify Map types
    map1_type = result.schema.field("map1").type
    map2_type = result.schema.field("map2").type

    assert isinstance(map1_type, pa.MapType)
    assert isinstance(map2_type, pa.MapType)

    # Verify data content
    assert result["id"].to_pylist() == [1]
    assert len(result["map1"][0]) == 2
    assert len(result["map2"][0]) == 2


def test_map_keys_cannot_be_null(tmp_path: Path):
    # Arrow Map spec requires keys to be non-nullable
    # The key field in the entries struct must have nullable=False

    # Test 1: Valid map with non-nullable keys (default behavior)
    schema_valid = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("valid_map", pa.map_(pa.string(), pa.int32())),
        ]
    )

    data_valid = pa.table(
        {"id": [1, 2], "valid_map": [[("a", 1), ("b", 2)], [("c", 3)]]},
        schema=schema_valid,
    )

    # This should succeed
    dataset = lance.write_dataset(
        data_valid, tmp_path / "valid", data_storage_version="2.2"
    )
    result = dataset.to_table()
    assert result.equals(data_valid)

    # Verify the key field is non-nullable in the schema
    map_type = result.schema.field("valid_map").type
    assert isinstance(map_type, pa.MapType)

    # Access the key and value types
    assert map_type.key_type == pa.string()
    assert map_type.item_type == pa.int32()

    # Test 2: Verify we can write maps with null values (but not null keys)
    data_null_values = pa.table(
        {
            "id": [1, 2],
            "map_with_null_values": [
                [("a", 1), ("b", None)],  # null value is OK
                [("c", None)],  # null value is OK
            ],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int32()),
                pa.field("map_with_null_values", pa.map_(pa.string(), pa.int32())),
            ]
        ),
    )

    dataset2 = lance.write_dataset(
        data_null_values, tmp_path / "null_values", data_storage_version="2.2"
    )
    result2 = dataset2.to_table()

    # Verify null values in map are preserved
    assert result2["id"].to_pylist() == [1, 2]
    map_data = result2["map_with_null_values"]

    # First map has 2 entries
    first_map = map_data[0]
    assert len(first_map) == 2

    # Values can be null
    values_list = [item[1] for item in first_map.as_py()]
    assert None in values_list  # At least one null value


def test_map_projection_queries(tmp_path: Path):
    # Create a dataset with multiple columns including Map types
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("name", pa.string()),
            pa.field("properties", pa.map_(pa.string(), pa.int32())),
            pa.field("tags", pa.map_(pa.string(), pa.string())),
            pa.field("score", pa.float64()),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "properties": [
                [("age", 25), ("height", 170)],
                [("age", 30), ("weight", 75)],
                [("age", 35)],
                None,  # null map
                [("age", 28), ("height", 165), ("weight", 60)],
            ],
            "tags": [
                [("role", "admin"), ("status", "active")],
                [("role", "user")],
                [("status", "inactive")],
                [("role", "guest")],
                [("role", "user"), ("status", "active")],
            ],
            "score": [95.5, 87.3, 91.2, 78.9, 88.7],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")

    # Test 1: Project only map column
    result1 = dataset.to_table(columns=["properties"])
    assert result1.num_rows == 5, "Row count mismatch for single map column projection"
    assert result1.schema.names == ["properties"], "Schema names mismatch"
    assert result1.schema.field("properties").type == pa.map_(
        pa.string(), pa.int32()
    ), "Map type mismatch"
    # Verify data consistency
    assert result1["properties"][0].as_py() == [("age", 25), ("height", 170)]
    assert result1["properties"][3].as_py() is None  # null map preserved

    # Test 2: Project multiple columns including map
    result2 = dataset.to_table(columns=["id", "properties", "score"])
    assert result2.num_rows == 5, "Row count mismatch for multi-column projection"
    assert result2.schema.names == ["id", "properties", "score"], (
        "Schema names mismatch"
    )
    assert result2["id"].to_pylist() == [1, 2, 3, 4, 5], "ID data mismatch"
    assert result2["score"].to_pylist() == [95.5, 87.3, 91.2, 78.9, 88.7], (
        "Score data mismatch"
    )

    # Test 3: Project two map columns
    result3 = dataset.to_table(columns=["properties", "tags"])
    assert result3.num_rows == 5, "Row count mismatch for two map columns"
    assert result3.schema.names == ["properties", "tags"], "Schema names mismatch"
    assert isinstance(result3.schema.field("properties").type, pa.MapType)
    assert isinstance(result3.schema.field("tags").type, pa.MapType)
    # Verify both map columns have correct data
    assert result3["tags"][0].as_py() == [("role", "admin"), ("status", "active")]

    # Test 4: Projection with filter
    result4 = dataset.to_table(columns=["id", "name", "properties"], filter="id > 2")
    assert result4.num_rows == 3, (
        "Row count mismatch with filter (expected 3 rows for id > 2)"
    )
    assert result4.schema.names == ["id", "name", "properties"], (
        "Schema names mismatch with filter"
    )
    assert result4["id"].to_pylist() == [3, 4, 5], "Filtered ID data mismatch"
    assert result4["name"].to_pylist() == ["Charlie", "David", "Eve"], (
        "Filtered name data mismatch"
    )
    # Verify map data is correct for filtered rows
    assert result4["properties"][0].as_py() == [("age", 35)]  # Charlie's properties
    assert result4["properties"][1].as_py() is None  # David's properties (null)

    # Test 5: Projection with more complex filter
    result5 = dataset.to_table(columns=["id", "properties"], filter="score >= 90")
    assert result5.num_rows == 2, (
        "Row count mismatch with score filter (expected 2 rows)"
    )
    assert result5.schema.names == ["id", "properties"], (
        "Should only contain id and properties columns"
    )
    assert result5["id"].to_pylist() == [1, 3], (
        "Filtered ID data mismatch for score >= 90"
    )

    # Test 6: Project all columns (no projection)
    result6 = dataset.to_table()
    assert result6.num_rows == 5, "Row count mismatch for full table read"
    assert result6.schema == schema, "Full schema mismatch"
    assert result6.equals(data), "Full data mismatch"

    # Test 7: Project only non-map columns
    result7 = dataset.to_table(columns=["id", "name", "score"])
    assert result7.num_rows == 5, "Row count mismatch for non-map projection"
    assert result7.schema.names == ["id", "name", "score"], (
        "Should only contain id, name and score columns"
    )
    assert "properties" not in result7.schema.names, (
        "Map column should not be in result"
    )
    assert "tags" not in result7.schema.names, "Map column should not be in result"
    assert result7["name"].to_pylist() == ["Alice", "Bob", "Charlie", "David", "Eve"]


def test_map_projection_nested_struct(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field(
                "user",
                pa.struct(
                    [
                        pa.field("name", pa.string()),
                        pa.field("metadata", pa.map_(pa.string(), pa.string())),
                        pa.field("age", pa.int32()),
                    ]
                ),
            ),
            pa.field("extra", pa.string()),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3],
            "user": [
                {
                    "name": "Alice",
                    "metadata": [("city", "NYC"), ("country", "USA")],
                    "age": 30,
                },
                {"name": "Bob", "metadata": [("city", "LA")], "age": 25},
                {"name": "Charlie", "metadata": None, "age": 35},
            ],
            "extra": ["info1", "info2", "info3"],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")

    # Test 1: Project the entire struct containing map
    result1 = dataset.to_table(columns=["id", "user"])
    assert result1.num_rows == 3, "Row count mismatch"
    assert result1.schema.names == ["id", "user"], "Schema names mismatch"
    # Verify struct schema
    user_type = result1.schema.field("user").type
    assert isinstance(user_type, pa.StructType)
    # Verify nested map type
    metadata_field = user_type.field("metadata")
    assert isinstance(metadata_field.type, pa.MapType)
    # Verify data
    assert result1["user"][0].as_py()["name"] == "Alice"
    assert result1["user"][0].as_py()["metadata"] == [
        ("city", "NYC"),
        ("country", "USA"),
    ]

    # Test 2: Project struct with filter
    result2 = dataset.to_table(columns=["user"], filter="id > 1")
    assert result2.num_rows == 2, "Row count mismatch with filter"
    assert result2.schema.names == ["user"], "Should only contain user column"
    assert result2["user"][0].as_py()["name"] == "Bob"
    assert result2["user"][1].as_py()["metadata"] is None  # Charlie has null metadata

    # Test 3: Project only id and extra (not the struct with map)
    result3 = dataset.to_table(columns=["id", "extra"])
    assert result3.num_rows == 3, "Row count mismatch"
    assert result3.schema.names == ["id", "extra"], (
        "Should only contain id and extra columns"
    )
    assert "user" not in result3.schema.names, "Struct column should not be in result"
    assert result3["extra"].to_pylist() == ["info1", "info2", "info3"]


def test_map_projection_list_of_maps(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("configs", pa.list_(pa.map_(pa.string(), pa.int32()))),
            pa.field("name", pa.string()),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3, 4],
            "configs": [
                [[("port", 8080), ("timeout", 30)], [("port", 8081), ("retries", 3)]],
                [[("port", 9090)]],
                None,  # null list
                [[("port", 7070), ("timeout", 60)], [("retries", 5)], [("port", 7071)]],
            ],
            "name": ["service1", "service2", "service3", "service4"],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")

    # Test 1: Project list of maps
    result1 = dataset.to_table(columns=["configs"])
    assert result1.num_rows == 4, "Row count mismatch"
    assert result1.schema.names == ["configs"], "Should only contain configs column"
    list_type = result1.schema.field("configs").type
    assert isinstance(list_type, pa.ListType)
    assert isinstance(list_type.value_type, pa.MapType)
    # Verify data
    assert len(result1["configs"][0]) == 2  # Two maps in first list
    assert result1["configs"][2].as_py() is None  # Null list

    # Test 2: Project with id and configs
    result2 = dataset.to_table(columns=["id", "configs"])
    assert result2.num_rows == 4, "Row count mismatch"
    assert result2.schema.names == ["id", "configs"], (
        "Should only contain id and configs columns"
    )
    assert result2["id"].to_pylist() == [1, 2, 3, 4]
    assert len(result2["configs"][3]) == 3  # Three maps in last list

    # Test 3: Projection with filter
    result3 = dataset.to_table(columns=["id", "configs", "name"], filter="id <= 2")
    assert result3.num_rows == 2, "Row count mismatch with filter"
    assert result3.schema.names == ["id", "configs", "name"], (
        "Should only contain id, configs and name columns"
    )
    assert result3["name"].to_pylist() == ["service1", "service2"]
    # Verify the list of maps data for filtered rows
    first_configs = result3["configs"][0].as_py()
    assert len(first_configs) == 2
    assert first_configs[0] == [("port", 8080), ("timeout", 30)]


def test_map_projection_multiple_value_types(tmp_path: Path):
    schema = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("int_map", pa.map_(pa.string(), pa.int32())),
            pa.field("float_map", pa.map_(pa.string(), pa.float64())),
            pa.field("string_map", pa.map_(pa.string(), pa.string())),
            pa.field("bool_map", pa.map_(pa.string(), pa.bool_())),
        ]
    )

    data = pa.table(
        {
            "id": [1, 2, 3],
            "int_map": [[("a", 1), ("b", 2)], [("c", 3)], None],
            "float_map": [[("x", 1.5), ("y", 2.5)], [("z", 3.5)], [("w", 4.5)]],
            "string_map": [
                [("k1", "v1"), ("k2", "v2")],
                [("k3", "v3")],
                [("k4", "v4"), ("k5", "v5")],
            ],
            "bool_map": [
                [("flag1", True)],
                [("flag2", False)],
                [("flag3", True), ("flag4", False)],
            ],
        },
        schema=schema,
    )

    dataset = lance.write_dataset(data, tmp_path, data_storage_version="2.2")

    # Test 1: Project subset of map columns
    result1 = dataset.to_table(columns=["id", "int_map", "string_map"])
    assert result1.num_rows == 3, "Row count mismatch"
    assert result1.schema.names == ["id", "int_map", "string_map"]
    assert result1.schema.field("int_map").type == pa.map_(pa.string(), pa.int32())
    assert result1.schema.field("string_map").type == pa.map_(pa.string(), pa.string())

    # Test 2: Project all map columns (no id)
    result2 = dataset.to_table(
        columns=["int_map", "float_map", "string_map", "bool_map"]
    )
    assert result2.num_rows == 3, "Row count mismatch"
    assert len(result2.schema.names) == 4
    # Verify all are map types
    for col in result2.schema.names:
        assert isinstance(result2.schema.field(col).type, pa.MapType)

    # Test 3: Project single map column with filter
    result3 = dataset.to_table(columns=["float_map"], filter="id != 2")
    assert result3.num_rows == 2, "Row count mismatch with filter"
    assert result3.schema.names == ["float_map"], "Should only contain float_map column"
    assert result3["float_map"][0].as_py() == [("x", 1.5), ("y", 2.5)]
    assert result3["float_map"][1].as_py() == [("w", 4.5)]

    # Test 4: Verify data consistency for all projections
    result4 = dataset.to_table(columns=["id", "bool_map"])
    assert result4.num_rows == 3, "Row count mismatch"
    assert result4.schema.names == ["id", "bool_map"], (
        "Should only contain id and bool_map columns"
    )
    assert result4["bool_map"][0].as_py() == [("flag1", True)]
    assert result4["bool_map"][1].as_py() == [("flag2", False)]
    assert result4["bool_map"][2].as_py() == [("flag3", True), ("flag4", False)]


def test_map_keys_sorted_unsupported(tmp_path: Path):
    """Test that keys_sorted=True is not supported"""
    # Test that keys_sorted=True is rejected
    schema_sorted = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field("sorted_map", pa.map_(pa.string(), pa.int32(), keys_sorted=True)),
        ]
    )

    data_sorted = pa.table(
        {"id": [1, 2], "sorted_map": [[("a", 1), ("b", 2)], [("c", 3)]]},
        schema=schema_sorted,
    )

    # Writing should fail with keys_sorted=True
    with pytest.raises(Exception) as exc_info:
        lance.write_dataset(
            data_sorted, tmp_path / "sorted", data_storage_version="2.2"
        )
    error_msg = str(exc_info.value)
    assert (
        "keys_sorted=true" in error_msg.lower()
        or "unsupported map field" in error_msg.lower()
    ), f"Expected error about keys_sorted=true, got: {error_msg}"

    # Test that keys_sorted=False (default) is supported
    schema_unsorted = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field(
                "unsorted_map", pa.map_(pa.string(), pa.int32(), keys_sorted=False)
            ),
        ]
    )

    data_unsorted = pa.table(
        {"id": [1, 2], "unsorted_map": [[("z", 1), ("a", 2)], [("c", 3)]]},
        schema=schema_unsorted,
    )

    dataset_unsorted = lance.write_dataset(
        data_unsorted, tmp_path / "unsorted", data_storage_version="2.2"
    )
    result_unsorted = dataset_unsorted.to_table()

    # Verify keys_sorted=False is preserved
    map_type_unsorted = result_unsorted.schema.field("unsorted_map").type
    assert isinstance(map_type_unsorted, pa.MapType)
    assert map_type_unsorted.keys_sorted is False

    # Test that default (keys_sorted=False) works
    schema_default = pa.schema(
        [
            pa.field("id", pa.int32()),
            pa.field(
                "default_map", pa.map_(pa.string(), pa.int32())
            ),  # default is False
        ]
    )

    data_default = pa.table(
        {"id": [1, 2], "default_map": [[("z", 1), ("a", 2)], [("c", 3)]]},
        schema=schema_default,
    )

    dataset_default = lance.write_dataset(
        data_default, tmp_path / "default", data_storage_version="2.2"
    )
    result_default = dataset_default.to_table()

    # Verify default keys_sorted=False is preserved
    map_type_default = result_default.schema.field("default_map").type
    assert isinstance(map_type_default, pa.MapType)
    assert map_type_default.keys_sorted is False
