# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from lance.file import stable_version


def prep_dataset(tmp_path: Path, version, name: str):
    dataset_dir = (
        Path(__file__).parent.parent.parent.parent / "test_data" / version / name
    )
    shutil.copytree(dataset_dir, tmp_path / version / name)
    return lance.dataset(tmp_path / version / name)


def test_add_data_storage_version(tmp_path: Path):
    """
    In version 0.15 and below we did not have a data storage version.  We had
    writer flags that were used to determine if we should use the old or new
    storage format.  In version 0.16 we added a data storage version to the
    manifest that should be correctly populated from existing files and/or the
    writer flags
    """
    tab = pa.table({"x": range(1024)})

    def check_dataset(dataset_name: str, expected_version: str):
        ds = prep_dataset(tmp_path, "v0.15.0", dataset_name)
        assert ds.data_storage_version == expected_version

        lance.write_dataset(tab, ds.uri, mode="append")
        assert ds.data_storage_version == expected_version

    check_dataset("v1_no_files.lance", "0.1")
    check_dataset("v2_no_files.lance", stable_version())
    check_dataset("v1_with_files.lance", "0.1")
    check_dataset("v2_with_files.lance", "2.0")


def test_fix_data_storage_version(tmp_path: Path):
    """
    In versions above 0.15 and below 0.17 we wrote the data storage version but
    we might have written it incorrectly.  In version 0.17 we fixed the inference
    rules.  However, this means a dataset could exist that has v2 files and a
    data storage version of 0.1.  Or, worse, it is even possible to have created
    a dataset with a mix of 0.1 and 2.0 files.  The former, we can fix automatically
    and the latter we can at least detect and advise the user rollback their dataset.
    """
    ds = prep_dataset(tmp_path, "v0.16.0", "wrong_data_version_can_fix.lance")
    assert ds.data_storage_version == "0.1"

    ds.delete("false")
    assert ds.data_storage_version == "2.0"

    ds = prep_dataset(tmp_path, "v0.16.0", "wrong_data_version_no_fix.lance")
    assert ds.data_storage_version == "0.1"

    with pytest.raises(
        OSError, match="The dataset contains a mixture of file versions"
    ):
        ds.delete("false")


def test_scan_batches_rejects_duplicate_manifest_fragment_ids(tmp_path: Path):
    ds = prep_dataset(tmp_path, "v0.16.0", "wrong_data_version_no_fix.lance")
    ds = lance.dataset(ds.uri, version=2)
    scanner = ds.scanner(columns=["foo"], batch_size=1)
    assert scanner.to_table()["foo"].to_pylist() == [1, 2, 1, 3]

    with pytest.raises(ValueError, match=r"duplicate fragment ID 1\b.*version 2\b"):
        next(scanner.scan_batches())


def test_old_btree_bitmap_indices(tmp_path: Path):
    """
    In versions below 0.21.0 we used the legacy file format for btree and bitmap
    indices.  In version 0.21.0 we switched to the new format.  This test ensures
    that we can still read the old indices.
    """
    ds = prep_dataset(tmp_path, "v0.20.0", "old_btree_bitmap_indices.lance")

    def query(filt: str):
        table = ds.to_table(filter=filt)
        assert table == pa.table({"bitmap": [3, 4], "btree": [3, 4]})

        explain = ds.scanner(filter=filt).explain_plan()
        assert "ScalarIndexQuery" in explain or "MaterializeIndex" in explain

    query("bitmap > 2")
    query("btree > 2")


def test_index_no_details(tmp_path: Path):
    """
    In versions below 0.19.3 we did not write index details to the index metadata.
    This test ensures that we can still read the old indices.
    """
    ds = prep_dataset(tmp_path, "v0.18.2", "index_no_details.lance")
    assert ds.to_table(filter="id > 2").num_rows == 97
