# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Tests for column name handling with mixed case and special characters.

These tests verify that Lance properly handles column names that:
1. Use mixed case (e.g., "userId", "OrderId") - common in TypeScript/JavaScript
2. Contain special characters (e.g., "user-id", "order:id")

See: https://github.com/lancedb/lance/issues/3424
"""

from pathlib import Path

import lance
import pyarrow as pa
import pytest
from lance.dataset import ColumnOrdering


class TestMixedCaseColumnNames:
    """
    Test that mixed-case column names work without requiring backtick quoting.

    Users coming from TypeScript/JavaScript commonly use camelCase column names.
    These should work in filter expressions, order by, scalar indices, etc.
    without requiring backtick escaping.
    """

    @pytest.fixture
    def mixed_case_table(self):
        """Create a table with mixed-case column names."""
        return pa.table(
            {
                "userId": range(100),
                "OrderId": range(100, 200),
                "itemName": [f"item_{i}" for i in range(100)],
            }
        )

    @pytest.fixture
    def mixed_case_dataset(self, tmp_path: Path, mixed_case_table):
        """Create a dataset with mixed-case column names."""
        return lance.write_dataset(mixed_case_table, tmp_path / "mixed_case")

    def test_create_table_with_mixed_case(self, mixed_case_dataset):
        """Verify table creation with mixed-case columns works."""
        # Table creation preserves column names - this works
        assert "userId" in [f.name for f in mixed_case_dataset.schema]
        assert "OrderId" in [f.name for f in mixed_case_dataset.schema]
        assert "itemName" in [f.name for f in mixed_case_dataset.schema]

    def test_filter_with_mixed_case(self, mixed_case_dataset):
        """Filter expressions should work with mixed-case column names."""
        # This should work without backticks
        result = mixed_case_dataset.to_table(filter="userId > 50")
        assert result.num_rows == 49

        # Also test with the other mixed-case columns
        result = mixed_case_dataset.to_table(filter="OrderId >= 150")
        assert result.num_rows == 50

        result = mixed_case_dataset.to_table(filter="itemName = 'item_25'")
        assert result.num_rows == 1

    def test_order_by_with_mixed_case(self, mixed_case_dataset):
        """Order by works with mixed-case column names when using proper API."""
        # order_by takes a list of column names or ColumnOrdering objects
        # This does NOT go through SQL parsing, so it preserves case
        ordering = ColumnOrdering("userId", ascending=False)
        scanner = mixed_case_dataset.scanner(order_by=[ordering])
        result = scanner.to_table()
        assert result.num_rows == 100
        assert result["userId"][0].as_py() == 99

        # Also test ordering by OrderId
        ordering = ColumnOrdering("OrderId", ascending=True)
        scanner = mixed_case_dataset.scanner(order_by=[ordering])
        result = scanner.to_table()
        assert result["OrderId"][0].as_py() == 100

    def test_scalar_index_with_mixed_case(self, mixed_case_dataset):
        """Scalar index creation should work with mixed-case column names."""
        mixed_case_dataset.create_scalar_index("userId", index_type="BTREE")

        indices = mixed_case_dataset.describe_indices()
        assert len(indices) == 1
        assert indices[0].field_names == ["userId"]
        assert indices[0].name == "userId_idx"

        # Query using the indexed column
        result = mixed_case_dataset.to_table(filter="userId = 50")
        assert result.num_rows == 1

        # Verify the index is actually used in the query plan
        plan = mixed_case_dataset.scanner(filter="userId = 50").explain_plan()
        assert "ScalarIndexQuery" in plan

        stats = mixed_case_dataset.stats.index_stats("userId_idx")
        assert stats["index_type"] == "BTree"

    def test_alter_column_with_mixed_case(self, mixed_case_dataset):
        """Altering columns works with mixed-case column names."""
        # alter_columns uses direct schema lookup, not SQL parsing
        mixed_case_dataset.alter_columns({"path": "userId", "name": "user_id"})

        assert "user_id" in [f.name for f in mixed_case_dataset.schema]
        assert "userId" not in [f.name for f in mixed_case_dataset.schema]

    def test_drop_column_with_mixed_case(self, tmp_path: Path, mixed_case_table):
        """Dropping columns works with mixed-case column names."""
        # drop_columns uses direct schema lookup, not SQL parsing
        dataset = lance.write_dataset(mixed_case_table, tmp_path / "drop_test")

        dataset.drop_columns(["OrderId"])

        assert "OrderId" not in [f.name for f in dataset.schema]
        assert "userId" in [f.name for f in dataset.schema]

    def test_merge_insert_with_mixed_case_key(self, tmp_path: Path, mixed_case_table):
        """Merge insert should work with mixed-case column as the key."""
        dataset = lance.write_dataset(mixed_case_table, tmp_path / "merge_test")

        new_data = pa.table(
            {
                "userId": range(50, 150),
                "OrderId": range(1000, 1100),
                "itemName": [f"new_item_{i}" for i in range(100)],
            }
        )

        dataset.merge_insert(
            "userId"
        ).when_matched_update_all().when_not_matched_insert_all().execute(new_data)

        result = dataset.to_table()
        assert result.num_rows == 150


class TestCaseOnlyDifferentColumnNames:
    """
    Test that columns differing only in case can both be resolved correctly.

    This tests the edge case where two column names are identical except for
    casing (e.g., "camelCase" and "CamelCase"). The case-insensitive lookup
    should still find the exact match when one exists.
    """

    @pytest.fixture
    def case_variant_table(self):
        """Create a table with columns that differ only in case.

        Values are deliberately non-correlated to ensure tests catch
        incorrect column resolution:
        - camelCase: 0, 1, 2, ... (ascending)
        - CamelCase: 99, 98, 97, ... (descending)
        - CAMELCASE: 50, 51, 52, ..., 99, 0, 1, ... (rotated)
        """
        return pa.table(
            {
                "camelCase": list(range(100)),
                "CamelCase": list(range(99, -1, -1)),  # reversed
                "CAMELCASE": list(range(50, 100)) + list(range(50)),  # rotated
            }
        )

    @pytest.fixture
    def case_variant_dataset(self, tmp_path: Path, case_variant_table):
        """Create a dataset with columns that differ only in case."""
        return lance.write_dataset(case_variant_table, tmp_path / "case_variant")

    def test_create_table_preserves_all_cases(self, case_variant_dataset):
        """Verify all case variants are preserved as distinct columns."""
        column_names = [f.name for f in case_variant_dataset.schema]
        assert "camelCase" in column_names
        assert "CamelCase" in column_names
        assert "CAMELCASE" in column_names

    def test_filter_resolves_exact_case_match(self, case_variant_dataset):
        """Filter expressions resolve to exact case match when available."""
        # camelCase has values 0-99 ascending, so camelCase < 10 matches rows 0-9
        result = case_variant_dataset.to_table(filter="camelCase < 10")
        assert result.num_rows == 10
        # Verify we got the right rows by checking other column values
        # Row 0 has: camelCase=0, CamelCase=99, CAMELCASE=50
        assert result["CamelCase"][0].as_py() == 99

        # CamelCase has values 99-0 descending, so CamelCase < 10 matches rows 90-99
        result = case_variant_dataset.to_table(filter="CamelCase < 10")
        assert result.num_rows == 10
        # These rows have camelCase values 90-99
        camel_values = sorted([v.as_py() for v in result["camelCase"]])
        assert camel_values == list(range(90, 100))

        # CAMELCASE has values 50-99,0-49 (rotated), so CAMELCASE < 10
        # matches rows 50-59 (which have CAMELCASE values 0-9)
        result = case_variant_dataset.to_table(filter="CAMELCASE < 10")
        assert result.num_rows == 10
        # These rows have camelCase values 50-59
        camel_values = sorted([v.as_py() for v in result["camelCase"]])
        assert camel_values == list(range(50, 60))

    def test_scalar_index_on_each_case_variant(self, tmp_path, case_variant_table):
        """Scalar index can be created on each case variant independently."""
        # Create separate datasets for each test to avoid index conflicts
        ds1 = lance.write_dataset(case_variant_table, tmp_path / "ds1")
        ds1.create_scalar_index("camelCase", index_type="BTREE")
        assert ds1.describe_indices()[0].field_names == ["camelCase"]

        # Query camelCase=50 should return row 50 (where CamelCase=49, CAMELCASE=0)
        result = ds1.to_table(filter="camelCase = 50")
        assert result.num_rows == 1
        assert result["camelCase"][0].as_py() == 50
        assert result["CamelCase"][0].as_py() == 49  # 99 - 50
        assert result["CAMELCASE"][0].as_py() == 0  # (50 + 50) % 100

        plan = ds1.scanner(filter="camelCase = 50").explain_plan()
        assert "ScalarIndexQuery" in plan

        # Test CamelCase index
        ds2 = lance.write_dataset(case_variant_table, tmp_path / "ds2")
        ds2.create_scalar_index("CamelCase", index_type="BTREE")
        assert ds2.describe_indices()[0].field_names == ["CamelCase"]

        # Query CamelCase=50 should return row 49 (where camelCase=49, CAMELCASE=99)
        result = ds2.to_table(filter="CamelCase = 50")
        assert result.num_rows == 1
        assert result["CamelCase"][0].as_py() == 50
        assert result["camelCase"][0].as_py() == 49  # row 49
        assert result["CAMELCASE"][0].as_py() == 99  # (49 + 50) % 100

        plan = ds2.scanner(filter="CamelCase = 50").explain_plan()
        assert "ScalarIndexQuery" in plan

        # Test CAMELCASE index
        ds3 = lance.write_dataset(case_variant_table, tmp_path / "ds3")
        ds3.create_scalar_index("CAMELCASE", index_type="BTREE")
        assert ds3.describe_indices()[0].field_names == ["CAMELCASE"]

        # Query CAMELCASE=50 should return row 0 (where camelCase=0, CamelCase=99)
        result = ds3.to_table(filter="CAMELCASE = 50")
        assert result.num_rows == 1
        assert result["CAMELCASE"][0].as_py() == 50
        assert result["camelCase"][0].as_py() == 0  # row 0
        assert result["CamelCase"][0].as_py() == 99  # 99 - 0

        plan = ds3.scanner(filter="CAMELCASE = 50").explain_plan()
        assert "ScalarIndexQuery" in plan

    def test_order_by_each_case_variant(self, case_variant_dataset):
        """Order by works with each case variant independently.

        With our test data:
        - camelCase: 0-99 ascending (row 99 has max value 99)
        - CamelCase: 99-0 descending (row 0 has max value 99)
        - CAMELCASE: 50-99,0-49 rotated (row 49 has max value 99)

        Ordering by each column DESC should put a different row first.
        """
        # Order by camelCase DESC: row 99 comes first
        ordering = ColumnOrdering("camelCase", ascending=False)
        result = case_variant_dataset.scanner(order_by=[ordering]).to_table()
        assert result["camelCase"][0].as_py() == 99
        assert result["CamelCase"][0].as_py() == 0  # row 99 has CamelCase=0
        assert result["CAMELCASE"][0].as_py() == 49  # row 99 has CAMELCASE=49

        # Order by CamelCase DESC: row 0 comes first
        ordering = ColumnOrdering("CamelCase", ascending=False)
        result = case_variant_dataset.scanner(order_by=[ordering]).to_table()
        assert result["CamelCase"][0].as_py() == 99
        assert result["camelCase"][0].as_py() == 0  # row 0 has camelCase=0
        assert result["CAMELCASE"][0].as_py() == 50  # row 0 has CAMELCASE=50

        # Order by CAMELCASE DESC: row 49 comes first
        ordering = ColumnOrdering("CAMELCASE", ascending=False)
        result = case_variant_dataset.scanner(order_by=[ordering]).to_table()
        assert result["CAMELCASE"][0].as_py() == 99
        assert result["camelCase"][0].as_py() == 49  # row 49 has camelCase=49
        assert result["CamelCase"][0].as_py() == 50  # row 49 has CamelCase=50


class TestSpecialCharacterColumnNames:
    """
    Test that column names with special characters work properly.

    Users may have column names with dashes, colons, or other special
    characters. These should work in filter expressions, order by,
    scalar indices, etc.

    Note: Column names with `.` are NOT allowed at the top level since `.` is
    used for nested field paths. This test uses `-` and `:` instead.
    """

    @pytest.fixture
    def special_char_table(self):
        """Create a table with special character column names."""
        return pa.table(
            {
                "user-id": range(100),
                "order:id": range(100, 200),
                "item_name": [f"item_{i}" for i in range(100)],
            }
        )

    @pytest.fixture
    def special_char_dataset(self, tmp_path: Path, special_char_table):
        """Create a dataset with special character column names."""
        return lance.write_dataset(special_char_table, tmp_path / "special_char")

    def test_create_table_with_special_chars(self, special_char_dataset):
        """Verify table creation with special character columns works."""
        # Table creation preserves column names - this works
        assert "user-id" in [f.name for f in special_char_dataset.schema]
        assert "order:id" in [f.name for f in special_char_dataset.schema]
        assert "item_name" in [f.name for f in special_char_dataset.schema]

    def test_filter_with_special_chars_using_backticks(self, special_char_dataset):
        """Filter expressions work with special char columns when using backticks."""
        # Backticks work for escaping special characters in SQL
        result = special_char_dataset.to_table(filter="`user-id` > 50")
        assert result.num_rows == 49

        result = special_char_dataset.to_table(filter="`order:id` >= 150")
        assert result.num_rows == 50

        # Regular column for comparison
        result = special_char_dataset.to_table(filter="item_name = 'item_25'")
        assert result.num_rows == 1

    def test_order_by_with_special_chars(self, special_char_dataset):
        """Order by works with special character column names."""
        # order_by uses column name directly, not SQL parsing
        ordering = ColumnOrdering("user-id", ascending=False)
        scanner = special_char_dataset.scanner(order_by=[ordering])
        result = scanner.to_table()
        assert result.num_rows == 100
        assert result["user-id"][0].as_py() == 99

        ordering = ColumnOrdering("order:id", ascending=True)
        scanner = special_char_dataset.scanner(order_by=[ordering])
        result = scanner.to_table()
        assert result["order:id"][0].as_py() == 100

    def test_scalar_index_with_special_chars(self, special_char_dataset):
        """Scalar index creation works with special character column names."""
        # Column name is used directly without SQL parsing
        special_char_dataset.create_scalar_index("user-id", index_type="BTREE")

        indices = special_char_dataset.describe_indices()
        assert len(indices) == 1
        assert indices[0].field_names == ["user-id"]
        assert indices[0].name == "user-id_idx"

        # Query using the indexed column (requires backticks in filter)
        result = special_char_dataset.to_table(filter="`user-id` = 50")
        assert result.num_rows == 1

        # Verify the index is actually used in the query plan
        plan = special_char_dataset.scanner(filter="`user-id` = 50").explain_plan()
        assert "ScalarIndexQuery" in plan

        stats = special_char_dataset.stats.index_stats("user-id_idx")
        assert stats["index_type"] == "BTree"

    def test_alter_column_with_special_chars(self, special_char_dataset):
        """Altering columns works with special character column names."""
        # alter_columns uses direct schema lookup
        special_char_dataset.alter_columns({"path": "user-id", "name": "user_id"})

        assert "user_id" in [f.name for f in special_char_dataset.schema]
        assert "user-id" not in [f.name for f in special_char_dataset.schema]

    def test_drop_column_with_special_chars(self, tmp_path: Path, special_char_table):
        """Dropping columns works with special character column names."""
        # drop_columns uses direct schema lookup
        dataset = lance.write_dataset(special_char_table, tmp_path / "drop_test")

        dataset.drop_columns(["order:id"])

        assert "order:id" not in [f.name for f in dataset.schema]
        assert "user-id" in [f.name for f in dataset.schema]

    def test_merge_insert_with_special_char_key(
        self, tmp_path: Path, special_char_table
    ):
        """Merge insert should work with special character column as the key."""
        dataset = lance.write_dataset(special_char_table, tmp_path / "merge_test")

        new_data = pa.table(
            {
                "user-id": range(50, 150),
                "order:id": range(1000, 1100),
                "item_name": [f"new_item_{i}" for i in range(100)],
            }
        )

        dataset.merge_insert(
            "user-id"
        ).when_matched_update_all().when_not_matched_insert_all().execute(new_data)

        result = dataset.to_table()
        assert result.num_rows == 150


class TestNestedFieldColumnNames:
    """
    Test that column names with mixed case and special characters work
    properly within nested (struct) fields.

    This tests nested field paths like:
    - MetaData.userId (mixed case in both parent and nested field)
    - `meta-data`.`user-id` (special chars in both parent and nested field)
    """

    @pytest.fixture
    def nested_mixed_case_table(self):
        """Create a table with mixed-case column names at all levels."""
        return pa.table(
            {
                "rowId": range(100),
                "MetaData": [{"userId": i, "itemCount": i * 10} for i in range(100)],
            }
        )

    @pytest.fixture
    def nested_mixed_case_dataset(self, tmp_path: Path, nested_mixed_case_table):
        """Create a dataset with mixed-case nested column names."""
        return lance.write_dataset(
            nested_mixed_case_table, tmp_path / "nested_mixed_case"
        )

    def test_create_table_with_nested_mixed_case(self, nested_mixed_case_dataset):
        """Verify table creation with nested mixed-case columns preserves names."""
        schema = nested_mixed_case_dataset.schema
        assert "rowId" in [f.name for f in schema]
        assert "MetaData" in [f.name for f in schema]
        metadata_field = schema.field("MetaData")
        nested_names = [f.name for f in metadata_field.type]
        assert "userId" in nested_names
        assert "itemCount" in nested_names

    def test_filter_with_nested_mixed_case(self, nested_mixed_case_dataset):
        """Filter expressions should work with mixed-case column names at all levels."""
        # Test top-level mixed case
        result = nested_mixed_case_dataset.to_table(filter="rowId > 50")
        assert result.num_rows == 49

        # Test nested mixed case (parent and child both mixed case)
        result = nested_mixed_case_dataset.to_table(filter="MetaData.userId > 50")
        assert result.num_rows == 49

        result = nested_mixed_case_dataset.to_table(filter="MetaData.itemCount >= 500")
        assert result.num_rows == 50

    def test_scalar_index_with_nested_mixed_case(self, nested_mixed_case_dataset):
        """Scalar index creation should work with mixed-case nested column names."""
        nested_mixed_case_dataset.create_scalar_index(
            "MetaData.userId", index_type="BTREE"
        )

        indices = nested_mixed_case_dataset.describe_indices()
        assert len(indices) == 1
        assert indices[0].name == "MetaData.userId_idx"
        assert indices[0].field_names == ["MetaData.userId"]

        # Query using the indexed column
        result = nested_mixed_case_dataset.to_table(filter="MetaData.userId = 50")
        assert result.num_rows == 1

        # Verify the index is actually used in the query plan
        plan = nested_mixed_case_dataset.scanner(
            filter="MetaData.userId = 50"
        ).explain_plan()
        assert "ScalarIndexQuery" in plan

        stats = nested_mixed_case_dataset.stats.index_stats("MetaData.userId_idx")
        assert stats["index_type"] == "BTree"

    def test_scalar_index_on_top_level_mixed_case(self, nested_mixed_case_dataset):
        """Scalar index on top-level mixed-case column works."""
        nested_mixed_case_dataset.create_scalar_index("rowId", index_type="BTREE")

        indices = nested_mixed_case_dataset.describe_indices()
        assert len(indices) == 1
        assert indices[0].name == "rowId_idx"
        assert indices[0].field_names == ["rowId"]

        result = nested_mixed_case_dataset.to_table(filter="rowId = 50")
        assert result.num_rows == 1

        plan = nested_mixed_case_dataset.scanner(filter="rowId = 50").explain_plan()
        assert "ScalarIndexQuery" in plan

        stats = nested_mixed_case_dataset.stats.index_stats("rowId_idx")
        assert stats["index_type"] == "BTree"

    def test_scalar_index_with_lowercased_nested_path(self, nested_mixed_case_dataset):
        """Scalar index creation should work even when path is lowercased.

        This tests the case-insensitive resolution for nested field paths.
        The schema has "MetaData.userId" but we pass "metadata.userid" (lowercased).
        It should still resolve and create the index with the correct case.
        """
        # Schema has: MetaData.userId (mixed case)
        # Pass lowercased path - should still resolve and create index
        nested_mixed_case_dataset.create_scalar_index(
            "metadata.userid", index_type="BTREE"
        )

        indices = nested_mixed_case_dataset.describe_indices()
        assert len(indices) == 1
        # Should store with correct case from schema
        assert indices[0].name == "MetaData.userId_idx"
        assert indices[0].field_names == ["MetaData.userId"]

        # Query should also work with correct case
        result = nested_mixed_case_dataset.to_table(filter="MetaData.userId = 50")
        assert result.num_rows == 1

        plan = nested_mixed_case_dataset.scanner(
            filter="MetaData.userId = 50"
        ).explain_plan()
        assert "ScalarIndexQuery" in plan

    @pytest.fixture
    def nested_special_char_table(self):
        """Create a table with special character column names at all levels."""
        return pa.table(
            {
                "row-id": range(100),
                "meta-data": [{"user-id": i, "item:count": i * 10} for i in range(100)],
            }
        )

    @pytest.fixture
    def nested_special_char_dataset(self, tmp_path: Path, nested_special_char_table):
        """Create a dataset with special character nested column names."""
        return lance.write_dataset(
            nested_special_char_table, tmp_path / "nested_special_char"
        )

    def test_create_table_with_nested_special_chars(self, nested_special_char_dataset):
        """Verify table creation with nested special char columns preserves names."""
        schema = nested_special_char_dataset.schema
        assert "row-id" in [f.name for f in schema]
        assert "meta-data" in [f.name for f in schema]
        metadata_field = schema.field("meta-data")
        nested_names = [f.name for f in metadata_field.type]
        assert "user-id" in nested_names
        assert "item:count" in nested_names

    def test_filter_with_nested_special_chars(self, nested_special_char_dataset):
        """Filter expressions work with special char columns at all levels."""
        # Test top-level special char column
        result = nested_special_char_dataset.to_table(filter="`row-id` > 50")
        assert result.num_rows == 49

        # Both the parent and child need backticks when they contain special chars
        result = nested_special_char_dataset.to_table(
            filter="`meta-data`.`user-id` > 50"
        )
        assert result.num_rows == 49

        result = nested_special_char_dataset.to_table(
            filter="`meta-data`.`item:count` >= 500"
        )
        assert result.num_rows == 50

    def test_scalar_index_with_nested_special_chars(self, nested_special_char_dataset):
        """Scalar index creation should work with special char nested column names."""
        # Use backtick syntax for nested field path with special chars
        nested_special_char_dataset.create_scalar_index(
            "`meta-data`.`user-id`", index_type="BTREE"
        )

        indices = nested_special_char_dataset.describe_indices()
        assert len(indices) == 1
        assert indices[0].field_names == ["meta-data.user-id"]
        assert indices[0].name == "meta-data.user-id_idx"

        # Query using the indexed column (backticks required in filter)
        result = nested_special_char_dataset.to_table(
            filter="`meta-data`.`user-id` = 50"
        )
        assert result.num_rows == 1

        # Verify the index is actually used in the query plan
        plan = nested_special_char_dataset.scanner(
            filter="`meta-data`.`user-id` = 50"
        ).explain_plan()
        assert "ScalarIndexQuery" in plan

        stats = nested_special_char_dataset.stats.index_stats("meta-data.user-id_idx")
        assert stats["index_type"] == "BTree"

    def test_scalar_index_on_top_level_special_chars(self, nested_special_char_dataset):
        """Scalar index on top-level special char column works."""
        nested_special_char_dataset.create_scalar_index("`row-id`", index_type="BTREE")

        indices = nested_special_char_dataset.describe_indices()
        assert len(indices) == 1
        assert indices[0].field_names == ["row-id"]

        result = nested_special_char_dataset.to_table(filter="`row-id` = 50")
        assert result.num_rows == 1

        plan = nested_special_char_dataset.scanner(
            filter="`row-id` = 50"
        ).explain_plan()
        assert "ScalarIndexQuery" in plan
