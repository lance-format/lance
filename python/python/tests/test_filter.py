# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors


"""Tests for predicate pushdown"""

import random
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import lance
import numpy as np
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from lance.vector import vec_to_table


def create_table(nrows=100):
    intcol = pa.array(range(nrows))
    floatcol = pa.array(np.arange(nrows) * 2 / 3, type=pa.float32())
    arr = np.arange(nrows) < nrows / 2
    structcol = pa.StructArray.from_arrays(
        [
            pa.array(arr, type=pa.bool_()),
            pa.array([date(2021, 1, 1) + timedelta(days=i) for i in range(nrows)]),
            pa.array([datetime(2021, 1, 1) + timedelta(hours=i) for i in range(nrows)]),
        ],
        names=["bool", "date", "dt"],
    )
    random.seed(42)

    def gen_str(n):
        return "".join(random.choices("abc", k=n))

    string_col = pa.array([gen_str(2) for _ in range(nrows)])

    decimal_col = pa.array([Decimal(f"{str(i)}.000") for i in range(nrows)])

    tbl = pa.Table.from_arrays(
        [intcol, floatcol, structcol, string_col, decimal_col],
        names=["int", "float", "rec", "str", "decimal"],
    )
    return tbl


@pytest.fixture()
def dataset(tmp_path: Path):
    tbl = create_table()
    yield lance.write_dataset(tbl, tmp_path)


def test_simple_predicates(dataset):
    predicates = [
        pc.field("int") >= 50,
        pc.field("int") == 50,
        pc.field("int") != 50,
        pc.field("float") < 30.0,
        pc.field("float") > 30.0,
        pc.field("float") <= 30.0,
        pc.field("float") >= 30.0,
        pc.field("str") != "aa",
        pc.field("str") == "aa",
        (pc.field("int") >= 50) & (pc.field("int") < 200),
        pc.invert(pc.field("int") >= 50),
        pc.is_null(pc.field("int")),
        pc.field("int") + 3 >= 50,
        pc.is_valid(pc.field("int")),
    ]
    # test simple
    for expr in predicates:
        assert dataset.to_table(filter=expr) == dataset.to_table().filter(expr)


def test_sql_predicates(dataset):
    # Predicate and expected number of rows
    predicates_nrows = [
        ("int >= 50", 50),
        ("int = 50", 1),
        ("int != 50", 99),
        ("int BETWEEN 50 AND 60", 11),
        ("float < 30.0", 45),
        ("str = 'aa'", 16),
        ("str in ('aa', 'bb')", 26),
        ("rec.bool", 50),
        ("rec.bool is true", 50),
        ("rec.bool is not true", 50),
        ("rec.bool is false", 50),
        ("rec.bool is not false", 50),
        ("rec.date = cast('2021-01-01' as date)", 1),
        ("rec.dt = cast('2021-01-01 00:00:00' as timestamp(6))", 1),
        ("rec.dt = cast('2021-01-01 00:00:00' as timestamp)", 1),
        ("rec.dt = cast('2021-01-01 00:00:00' as datetime(6))", 1),
        ("rec.dt = cast('2021-01-01 00:00:00' as datetime)", 1),
        ("rec.dt = TIMESTAMP '2021-01-01 00:00:00'", 1),
        ("rec.dt = TIMESTAMP(6) '2021-01-01 00:00:00'", 1),
        ("rec.date = DATE '2021-01-01'", 1),
        ("rec.date >= cast('2021-01-31' as date)", 70),
        ("cast(rec.date as string) = '2021-01-01'", 1),
        ("decimal = DECIMAL(5,3) '12.000'", 1),
        ("decimal >= DECIMAL(5,3) '50.000'", 50),
    ]

    for expr, expected_num_rows in predicates_nrows:
        assert dataset.to_table(filter=expr).num_rows == expected_num_rows


@pytest.mark.parametrize("unit", ["s", "ms", "us"])
@pytest.mark.parametrize("timezone", [None, "UTC", "America/New_York"])
def test_timestamp_pyarrow_predicates(tmp_path: Path, unit: str, timezone: str | None):
    # PyArrow filters reach Lance as Substrait, where the timestamp literal used to be
    # decoded in the wrong unit.
    tz = ZoneInfo(timezone) if timezone else None
    start = datetime(2021, 1, 1, tzinfo=tz)
    ts_type = pa.timestamp(unit, timezone)
    table = pa.table(
        {"ts": pa.array([start + timedelta(hours=i) for i in range(100)], ts_type)}
    )
    dataset = lance.write_dataset(table, tmp_path / f"{unit}_{timezone}")

    cutoff = pa.scalar(start + timedelta(hours=50), ts_type)
    for expr in [
        pc.field("ts") > cutoff,
        pc.field("ts") < cutoff,
        pc.field("ts") == cutoff,
    ]:
        assert dataset.to_table(filter=expr) == table.filter(expr)


def test_sql_current_date(tmp_path: Path):
    table = pa.table(
        {"date": pa.array([date(2020, 1, 1), date(2020, 1, 2)], type=pa.date32())}
    )
    dataset = lance.write_dataset(table, tmp_path / "current_date")

    filtered = dataset.to_table(filter="date <= current_date()")
    assert filtered.equals(dataset.to_table())


def test_illegal_predicates(dataset):
    bad_parse = [
        "str BETWEEN 10 AND 20",
        "str > 10",
        "str AN",
        "🥞",
    ]
    for expr in bad_parse:
        with pytest.raises(ValueError, match="Invalid user input: *"):
            dataset.to_table(filter=expr)
    with pytest.raises(ValueError, match="No field named foo"):
        dataset.to_table(filter="foo = 7")
    with pytest.raises(ValueError, match="does not return a boolean"):
        dataset.to_table(filter="int")


def test_compound(dataset):
    predicates = [
        pc.field("int") >= 50,
        pc.field("float") < 90.0,
        pc.field("str") == "aa",
    ]
    # test compound
    for expr in predicates:
        for other_expr in predicates:
            compound = expr & other_expr
            assert dataset.to_table(filter=compound) == dataset.to_table().filter(
                compound
            )
            compound = expr | other_expr
            assert dataset.to_table(filter=compound) == dataset.to_table().filter(
                compound
            )


def test_match(tmp_path: Path, provide_pandas: bool):
    array = pa.array(["aaa", "bbb", "abc", "bca", "cab", "cba"])
    table = pa.Table.from_arrays([array], names=["str"])
    dataset = lance.write_dataset(table, tmp_path / "test_match")

    result = dataset.to_table(filter="str LIKE 'a%'").to_pandas()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"str": ["aaa", "abc"]}))

    result = dataset.to_table(filter="str NOT LIKE 'a%'").to_pandas()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"str": ["bbb", "bca", "cab", "cba"]})
    )

    result = dataset.to_table(filter="regexp_match(str, 'c.+')").to_pandas()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"str": ["bca", "cab", "cba"]}))


def test_escaped_name(tmp_path: Path, provide_pandas: bool):
    table = pa.table({"silly :name": pa.array([0, 1, 2])})
    dataset = lance.write_dataset(table, tmp_path / "test_escaped_name")

    dataset = lance.dataset(tmp_path / "test_escaped_name")
    result = dataset.to_table(filter="`silly :name` > 1").to_pandas()
    pd.testing.assert_frame_equal(result, pd.DataFrame({"silly :name": [2]}))

    # nested case
    table = pa.table({"outer field": pa.array([{"inner field": i} for i in range(3)])})
    dataset = lance.write_dataset(table, tmp_path / "test_escaped_name_nested")

    dataset = lance.dataset(tmp_path / "test_escaped_name_nested")
    result = dataset.to_table(filter="`outer field`.`inner field` > 1").to_pandas()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"outer field": [{"inner field": 2}]})
    )

    # test uppercase name
    table = pa.table({"ALLCAPSNAME": pa.array([0, 1]), "other": pa.array([2, 3])})
    _ = lance.write_dataset(table, tmp_path / "test_uppercase_name")

    dataset = lance.dataset(tmp_path / "test_uppercase_name")
    result = dataset.to_table(filter="`ALLCAPSNAME` > 0").to_pandas()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame([{"ALLCAPSNAME": 1, "other": 3}])
    )

    table = pa.table(
        {"Nested with Space": pa.array([{"Inner With Caps": i} for i in range(3)])}
    )
    _ = lance.write_dataset(table, tmp_path / "test_escaped_name_nested_and_capped")

    dataset = lance.dataset(tmp_path / "test_escaped_name_nested_and_capped")
    result = dataset.to_table(
        filter="`Nested with Space`.`Inner With Caps` > 1"
    ).to_pandas()
    pd.testing.assert_frame_equal(
        result, pd.DataFrame({"Nested with Space": [{"Inner With Caps": 2}]})
    )


def test_functions(tmp_path: Path):
    # Ensure that we can use complex functions
    table = pa.table(
        {"genres": [["action", "comedy"], ["anime", "drama"], ["adventure"]]}
    )
    expected = table.slice(1, 2)
    dataset = lance.write_dataset(table, tmp_path / "test_neg_expr")
    assert (
        dataset.to_table(filter="array_has_any(genres, Array['anime', 'adventure'])")
        == expected
    )

    expected = table.slice(0, 1)
    assert dataset.to_table(filter="array_contains(genres, 'comedy')") == expected


def test_negative_expressions(tmp_path: Path):
    table = pa.table({"x": [-1, 0, 1, 1], "y": [1, 2, 3, 4]})
    dataset = lance.write_dataset(table, tmp_path / "test_neg_expr")
    filters_expected = [
        ("x = -1", [-1]),
        ("x > -1", [0, 1, 1]),
        ("x = 1 * -1", [-1]),
        ("x <= 2 + -2 ", [-1, 0]),
        ("x = y - 2", [-1, 0, 1]),
    ]
    for filter, expected in filters_expected:
        assert dataset.scanner(filter=filter).to_table()["x"].to_pylist() == expected


def create_table_for_duckdb(nvec=10000, ndim=768):
    mat = np.random.randn(nvec, ndim)
    price = (np.random.rand(nvec) + 1) * 100

    def gen_str(n):
        return "".join(random.choices("abc"))

    meta = np.array([gen_str(1) for _ in range(nvec)])
    tbl = (
        vec_to_table(data=mat)
        .append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
        .append_column("id", pa.array(range(nvec)))
    )
    return tbl


def test_datatypes(tmp_path):
    table = pa.table(
        {
            "binary": pa.array([b"abc", None], type=pa.binary()),
            "largebin": pa.array([b"abc", None], type=pa.large_binary()),
        }
    )
    dataset = lance.write_dataset(table, tmp_path)

    for filter, expected_matches in [
        ("binary = X'616263'", 1),
        ("binary is NULL", 1),
        ("largebin = X'616263'", 1),
        ("largebin is NULL", 1),
    ]:
        assert dataset.count_rows(filter=filter) == expected_matches


def test_duckdb(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    tbl = create_table_for_duckdb()
    ds = lance.write_dataset(tbl, str(tmp_path))  # noqa: F841

    actual = duckdb.query("SELECT id, meta, price FROM ds WHERE id==1000").to_df()
    expected = duckdb.query("SELECT id, meta, price FROM ds").to_df()
    expected = expected[expected.id == 1000].reset_index(drop=True)
    tm.assert_frame_equal(actual, expected)

    actual = duckdb.query("SELECT id, meta, price FROM ds WHERE id=1000").to_df()
    expected = duckdb.query("SELECT id, meta, price FROM ds").to_df()
    expected = expected[expected.id == 1000].reset_index(drop=True)
    tm.assert_frame_equal(actual, expected)

    actual = duckdb.query(
        "SELECT id, meta, price FROM ds WHERE price>20.0 and price<=90"
    ).to_df()
    expected = duckdb.query("SELECT id, meta, price FROM ds").to_df()
    expected = expected[(expected.price > 20.0) & (expected.price <= 90)].reset_index(
        drop=True
    )
    tm.assert_frame_equal(actual, expected, check_dtype=False)

    actual = duckdb.query("SELECT id, meta, price FROM ds WHERE meta=='aa'").to_df()
    expected = duckdb.query("SELECT id, meta, price FROM ds").to_df()
    expected = expected[expected.meta == "aa"].reset_index(drop=True)
    tm.assert_frame_equal(actual, expected, check_dtype=False)


def test_struct_field_order(tmp_path):
    """
    This test regresses some old behavior where the order of struct fields would get
    messed up due to late materialization and we would get {y,x} instead of {x,y}
    """
    data = pa.table({"struct": [{"x": i, "y": i} for i in range(10)]})
    dataset = lance.write_dataset(data, tmp_path)

    for late_materialization in [True, False]:
        result = dataset.to_table(
            filter="struct.y > 5", late_materialization=late_materialization
        )
        expected = pa.table({"struct": [{"x": i, "y": i} for i in range(6, 10)]})
        assert result == expected


def test_filter_on_column_beside_struct_with_extension_type(tmp_path):
    tensor_type = pa.fixed_shape_tensor(pa.float32(), (3,))
    tensor_arr = pa.ExtensionArray.from_storage(
        tensor_type,
        pa.FixedSizeListArray.from_arrays(pa.array([1.0, 2.0, 3.0], pa.float32()), 3),
    )
    struct_arr = pa.StructArray.from_arrays([tensor_arr], names=["vec"])

    arrow_table = pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "checkpoint": pa.array([None], pa.int64()),
            "items": struct_arr,
        }
    )
    ds = lance.write_dataset(arrow_table, tmp_path)

    expr = pc.field("checkpoint").is_null() | (pc.field("checkpoint") == 0)
    result = ds.to_table(filter=expr)
    assert result["id"].to_pylist() == [1]


def test_filter_on_column_beside_root_extension_type(tmp_path):
    """Filtering should work when the schema has a top-level extension type column.

    fixed_shape_tensor at the root level cannot be converted to a substrait type,
    so it must also be replaced with a placeholder.
    """
    tensor_type = pa.fixed_shape_tensor(pa.float32(), (3,))
    tensor_arr = pa.ExtensionArray.from_storage(
        tensor_type,
        pa.FixedSizeListArray.from_arrays(pa.array([1.0, 2.0, 3.0], pa.float32()), 3),
    )
    arrow_table = pa.table(
        {
            "id": pa.array([1], pa.int64()),
            "checkpoint": pa.array([None], pa.int64()),
            "vec": tensor_arr,
        }
    )
    ds = lance.write_dataset(arrow_table, tmp_path)

    expr = pc.field("checkpoint").is_null() | (pc.field("checkpoint") == 0)
    result = ds.to_table(filter=expr)
    assert result["id"].to_pylist() == [1]


@pytest.mark.skip(
    reason="requires a release build; see "
    "https://github.com/lance-format/lance/pull/4190"
)
def test_filter_depth_limit():
    column_name = "a_very_long_column_name"
    ds = lance.write_dataset(pa.table({column_name: [1, 2]}), "memory://")
    ds.create_scalar_index(column_name, "BTREE")

    filter = " AND ".join([f"{column_name} = {i}" for i in range(500)])
    ds.to_table(filter=filter)
    with pytest.raises(ValueError, match="the filter expression is too long"):
        filter = " AND ".join([f"{column_name} = {i}" for i in range(501)])
        ds.to_table(filter=filter)
