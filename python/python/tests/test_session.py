# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import pyarrow as pa
import pytest


def test_cache_size_bytes(
    tmp_path: Path,
):
    data = pa.table({"a": range(1000)})
    lance.write_dataset(data, tmp_path, max_rows_per_file=250)

    ds = lance.dataset(tmp_path)

    initial_size = ds.session().size_bytes()

    ds.scanner().to_table()

    after_scan_size = ds.session().size_bytes()

    assert after_scan_size > initial_size


def test_share_session(tmp_path: Path):
    data = pa.table({"a": range(1000)})
    ds1 = lance.write_dataset(data, tmp_path, max_rows_per_file=250)

    assert ds1.to_table() == data

    ds2 = lance.dataset(tmp_path, session=ds1.session())

    assert ds1.session().is_same_as(ds2.session())

    assert ds1.session().size_bytes() == ds2.session().size_bytes()

    assert ds1.to_table() == ds2.to_table()


def test_fragment_write_with_session(tmp_path: Path):
    from lance.fragment import LanceFragment, write_fragments

    data = pa.table({"a": range(10), "b": [str(i) for i in range(10)]})
    ds = lance.write_dataset(data, tmp_path)
    # Drop a column so the surviving field id is non-trivial (!= 0). Appends
    # that infer the schema must pick up this field id from the dataset.
    ds.drop_columns(["a"])
    field_id = ds.lance_schema.field_case_insensitive("b").id()
    assert field_id != 0

    session = ds.session()
    size_before = session.size_bytes()

    append_data = pa.table({"b": ["x", "y"]})
    fragments = write_fragments(
        append_data, str(tmp_path), mode="append", session=session
    )
    assert len(fragments) == 1
    assert fragments[0].files[0].fields == [field_id]

    fragment = LanceFragment.create(
        str(tmp_path), append_data, mode="append", session=session
    )
    assert fragment.files[0].fields == [field_id]

    # The manifest loads for schema inference went through the shared session.
    assert session.size_bytes() > size_before

    # A LanceDataset destination always uses its own session; a different
    # explicit session is rejected.
    with pytest.raises(ValueError, match="not the destination dataset's own session"):
        write_fragments(append_data, ds, mode="append", session=lance.Session())


def test_cache_backend_uri_config():
    session = lance.Session(index_cache_backend="moka://?capacity=1048576")

    assert session.index_cache_size_bytes() == 0


def test_cache_backend_dict_config():
    session = lance.Session(
        index_cache_backend={
            "kind": "MOKA",
            "options": {"capacity": "1048576"},
        },
    )

    assert session.index_cache_size_bytes() == 0


def test_cache_backend_rejects_size_and_backend():
    with pytest.raises(
        ValueError,
        match="index_cache_size_bytes and index_cache_backend are mutually exclusive",
    ):
        lance.Session(
            index_cache_size_bytes=1024,
            index_cache_backend="moka://?capacity=1048576",
        )


def test_cache_backend_rejects_unknown_dict_key():
    with pytest.raises(ValueError, match="unknown dict key"):
        lance.Session(
            index_cache_backend={
                "kind": "moka",
                "capacity": "1048576",
            },
        )


def test_cache_backend_rejects_moka_without_capacity():
    with pytest.raises(ValueError, match="capacity is required"):
        lance.Session(index_cache_backend="moka://")


def test_cache_backend_rejects_moka_empty_capacity():
    with pytest.raises(ValueError, match="capacity must not be empty"):
        lance.Session(index_cache_backend="moka://?capacity=")
