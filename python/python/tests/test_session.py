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


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("by_type", [False, True])
def test_cache_diagnostics_reports_both_cache_tiers(refresh: bool, by_type: bool):
    session = lance.Session(
        index_cache_size_bytes=0,
        metadata_cache_size_bytes=2048,
    )

    diagnostics = session.cache_diagnostics(refresh=refresh, by_type=by_type)

    assert diagnostics.keys() == {"index", "metadata"}
    for cache in diagnostics.values():
        expected_keys = {"activity", "backend", "utilization"}
        if by_type:
            expected_keys.add("by_type")
        assert cache.keys() == expected_keys
        assert cache["activity"]["loads_in_flight"] == 0
        assert cache["activity"]["warm"] == {
            "attempts": 0,
            "hits": 0,
            "loads_started": 0,
            "loads_succeeded": 0,
            "loads_failed": 0,
            "loads_cancelled": 0,
            "load_bytes": 0,
            "errors": 0,
        }
        assert cache["backend"]["kind"] == "quick"
        assert cache["backend"]["pool_id"] is not None
        if by_type:
            assert cache["by_type"].keys() == {
                "activity",
                "occupancy",
                "type_label_overflow_events",
            }
            assert cache["by_type"]["activity"] == {}
            assert cache["by_type"]["occupancy"] == {
                "types": {},
                "untagged_size_bytes": 0,
                "untagged_num_entries": 0,
            }
            assert cache["by_type"]["type_label_overflow_events"] == 0

    assert diagnostics["index"]["backend"]["capacity_bytes"] == 0
    assert diagnostics["index"]["backend"]["enabled"] is False
    assert diagnostics["index"]["utilization"] is None
    assert diagnostics["metadata"]["backend"]["capacity_bytes"] == 2048
    assert diagnostics["metadata"]["backend"]["enabled"] is True
    assert diagnostics["metadata"]["utilization"] == 0.0

    # Existing APIs keep their return types and semantics.
    assert isinstance(session.size_bytes(), int)
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
