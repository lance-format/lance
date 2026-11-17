# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors


"""Tests for sort by"""

from pathlib import Path

import lance
import pyarrow as pa
from lance._dataset.scanner import ColumnOrdering


def test_one_column_order_by(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = pa.table(
        {"int_col": [2, 1, 3, None, 4], "str_col": ["a", None, "d", "b", "c"]}
    )
    dataset = lance.write_dataset(data, base_dir)

    ## asc null_last
    ordering = ColumnOrdering("int_col")
    true_value = pa.table(
        {"int_col": [1, 2, 3, 4, None], "str_col": [None, "a", "d", "c", "b"]}
    )
    assert dataset.scanner(order_by=[ordering]).to_table() == true_value

    ## asc null_first
    ordering = ColumnOrdering("int_col", nulls_first=True)
    true_value = pa.table(
        {
            "int_col": [None, 1, 2, 3, 4],
            "str_col": ["b", None, "a", "d", "c"],
        }
    )
    assert dataset.scanner(order_by=[ordering]).to_table() == true_value

    ## desc null_last
    ordering = ColumnOrdering("int_col", ascending=False)
    true_value = pa.table(
        {
            "int_col": [4, 3, 2, 1, None],
            "str_col": ["c", "d", "a", None, "b"],
        }
    )
    assert dataset.scanner(order_by=[ordering]).to_table() == true_value

    ## desc null_first
    ordering = ColumnOrdering("int_col", ascending=False, nulls_first=True)
    true_value = pa.table(
        {
            "int_col": [None, 4, 3, 2, 1],
            "str_col": ["b", "c", "d", "a", None],
        }
    )
    assert dataset.scanner(order_by=[ordering]).to_table() == true_value


def test_two_column_order_by(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = pa.table(
        {
            "int_col": [2, 1, 3, 3, None, 4, 3],
            "str_col": ["a", None, "d", None, "b", "c", "e"],
        }
    )
    dataset = lance.write_dataset(data, base_dir)

    ## int column asc null_last and str col desc null
    ordering1 = ColumnOrdering("int_col")
    ordering2 = ColumnOrdering("str_col", ascending=False, nulls_first=True)

    true_value = pa.table(
        {
            "int_col": [1, 2, 3, 3, 3, 4, None],
            "str_col": [None, "a", None, "e", "d", "c", "b"],
        }
    )
    assert dataset.scanner(order_by=[ordering1, ordering2]).to_table() == true_value

    ## Can also pass just column name for default behavior
    assert dataset.to_table(order_by=["int_col", ordering2]) == true_value


def test_all_order_by_support_functions(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = pa.table(
        {"int_col": [2, 1, 3, None, 4], "str_col": ["a", None, "d", "b", "c"]}
    )
    dataset = lance.write_dataset(data, base_dir)

    ## asc null_last
    ordering = ColumnOrdering("int_col")
    true_value = pa.table(
        {"int_col": [1, 2, 3, 4, None], "str_col": [None, "a", "d", "c", "b"]}
    )

    assert dataset.to_table(order_by=[ordering]) == true_value
    assert pa.Table.from_batches(dataset.to_batches(order_by=[ordering])) == true_value
    assert dataset.scanner(order_by=[ordering]).to_table() == true_value

    fragment = dataset.get_fragment(0)
    assert fragment.to_table(order_by=[ordering]) == true_value
    assert pa.Table.from_batches(fragment.to_batches(order_by=[ordering])) == true_value
    assert fragment.scanner(order_by=[ordering]).to_table() == true_value
