# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors


"""Tests for sort by"""

from pathlib import Path

import lance
import pyarrow as pa
import pytest
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


@pytest.mark.parametrize(
    "orderings, indices",
    [
        pytest.param(["value"], [2, 3, 0, 1], id="column-name"),
        pytest.param(
            [ColumnOrdering("value", ascending=False, nulls_first=True)],
            [1, 0, 3, 2],
            id="ordering-object",
        ),
        pytest.param(
            ["group", ColumnOrdering("value", ascending=False)],
            [0, 3, 1, 2],
            id="mixed-orderings",
        ),
    ],
)
def test_default_scan_options_order_by(tmp_path: Path, orderings, indices):
    data = pa.table({"value": [3, None, 1, 2], "group": [0, 0, 1, 0]})
    dataset = lance.write_dataset(data, tmp_path, max_rows_per_file=2)
    assert len(dataset.get_fragments()) == 2
    expected = data.take(indices)
    assert dataset.scanner(order_by=orderings).to_table() == expected

    ordered = lance.dataset(tmp_path, default_scan_options={"order_by": orderings})
    assert ordered.scanner().to_table() == expected
    assert ordered.to_table() == expected
    assert pa.Table.from_batches(ordered.to_batches()) == expected

    assert ordered.scanner(order_by=["value"]).to_table() == data.take([2, 3, 0, 1])
    assert ordered.scanner(order_by=[]).to_table() == data
    assert ordered.to_table() == expected


@pytest.mark.parametrize("use_defaults", [False, True], ids=["explicit", "default"])
@pytest.mark.parametrize("orderings", ["xy", [1], ["xy", None]])
def test_order_by_invalid_orderings(tmp_path: Path, use_defaults, orderings):
    data = pa.table({"x": [2, 1], "y": [1, 2], "xy": [1, 2]})
    dataset = lance.write_dataset(data, tmp_path)
    with pytest.raises(TypeError, match="orderings must"):
        if use_defaults:
            lance.dataset(
                tmp_path, default_scan_options={"order_by": orderings}
            ).to_table()
        else:
            dataset.scanner(order_by=orderings)
