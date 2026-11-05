# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import duckdb
import lance
import pyarrow as pa
import pytest


@pytest.mark.skip(reason="Unsigned integers not available for pushdown yet")
def test_duckdb_filter_on_rowid(tmp_path):
    tab = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    ds = lance.write_dataset(tab, str(tmp_path))
    ds = lance.dataset(str(tmp_path), default_scan_options={"with_row_id": True})
    row_ids = ds.scanner(columns=[], with_row_id=True).to_table().column(0).to_pylist()
    expected = tab.slice(1, 1)
    actual = duckdb.query(
        f"SELECT * FROM ds WHERE _rowid = {row_ids[1]}"
    ).to_arrow_table()

    assert actual.to_pydict() == expected.to_pydict()


def test_duckdb_pushdown_extension_types(tmp_path):
    # large_binary is reported by pyarrow as a substrait extension type.  Datafusion
    # does not currently handle these extension types.  This should be ok as long
    # as the filter isn't accessing the column with the extension type.
    #
    # Lance works around this by removing any columns with extension types from the
    # schema it gives to duckdb.
    tab = pa.table(
        {
            "filterme": [1, 2, 3],
            "largebin": pa.array([b"123", b"456", b"789"], pa.large_binary()),
            "othercol": [4, 5, 6],
        }
    )
    ds = lance.write_dataset(tab, str(tmp_path))  # noqa: F841
    expected = tab.slice(1, 1)
    actual = duckdb.query("SELECT * FROM ds WHERE filterme = 2").to_arrow_table()
    assert actual.to_pydict() == expected.to_pydict()

    expected = tab.slice(0, 1)
    actual = duckdb.query("SELECT * FROM ds WHERE othercol = 4").to_arrow_table()
    assert actual.to_pydict() == expected.to_pydict()

    # Not the best error message but hopefully this is short lived until datafusion
    # supports substrait extension types.
    with pytest.raises(
        # Older versions of duckdb use duckdb.InvalidInputException and newer versions
        # use duckdb.duckdb.Error but that part isn't really important so just check for
        # Exception
        Exception,
        match="referenced a field that is not yet supported by Substrait conversion",
    ):
        duckdb.query("SELECT * FROM ds WHERE largebin = '456'").fetchall()

    # Unclear if all of these result in pushdown or not but they shouldn't error if
    # they do.
    for filt in [
        "filterme != 2 AND filterme != 1",
        "filterme in (1)",
        "filterme IS NULL",
        "filterme IS NOT NULL",
        "filterme < 2",
    ]:
        expected = duckdb.query(f"SELECT * FROM tab WHERE {filt}").to_arrow_table()
        actual = duckdb.query(f"SELECT * FROM ds WHERE {filt}").to_arrow_table()
        assert actual == expected


def test_duckdb_pushdown_struct_field(tmp_path):
    # DuckDB pushes a filter on a struct field down as a nested field reference.
    tab = pa.table(
        {
            "id": [1, 2, 3],
            "meta": [{"x": 1, "y": "a"}, {"x": 6, "y": "b"}, {"x": 9, "y": "c"}],
        }
    )
    ds = lance.write_dataset(tab, str(tmp_path))  # noqa: F841
    actual = duckdb.query("SELECT id FROM ds WHERE meta.x > 5 ORDER BY id").fetchall()
    assert actual == [(2,), (3,)]
    actual = duckdb.query("SELECT id FROM ds WHERE meta.y = 'a'").fetchall()
    assert actual == [(1,)]
