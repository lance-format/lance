# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Scalar index compatibility tests for Lance.

Tests that scalar indices (BTREE, BITMAP, LABEL_LIST, NGRAM, ZONEMAP,
BLOOMFILTER, JSON, FTS) created with one version of Lance can be read
and written by other versions.
"""

import os
import shutil
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from packaging.version import Version

from .compat_decorator import (
    UpgradeDowngradeTest,
    compat_test,
)
from .util import safe_data_storage_version


@compat_test(min_version="0.30.0")
class BTreeIndex(UpgradeDowngradeTest):
    """Test BTREE scalar index compatibility (introduced in 0.20.0).

    Started fully working in 0.30.0 with various fixes.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with BTREE index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "btree": pa.array(range(1000)),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_scalar_index("btree", "BTREE")

    def check_read(self):
        """Verify BTREE index can be queried."""
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="btree == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        # Verify index is used -- but only when we can be sure this build
        # recognizes whatever format the index is actually in. BTREE moved to
        # storing row addresses and bumped its on-disk format version (see
        # BTreeRowAddressDomainIndex below); the current build understands
        # both the old and new format, but an older build only understands
        # the old one. In the upgrade/downgrade round trip this same method
        # runs under the older build *after* the current build's check_write
        # has already rebuilt the index in the newer format, so the older
        # build can no longer see it here and must fall back to a full scan
        # -- still correct, just not re-asserted as "indexed" in that case.
        #
        # Checked via _running_in_old_venv, not a lance.__version__ /
        # compat_version comparison: the current build's own version is not a
        # fixed marker distinct from every pinned release (see that flag's
        # docstring on UpgradeDowngradeTest).
        if not self._running_in_old_venv:
            explain = ds.scanner(filter="btree == 7").explain_plan()
            assert "ScalarIndexQuery" in explain or "MaterializeIndex" in explain

    def check_write(self):
        """Verify can insert data and optimize BTREE index."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "btree": pa.array([1000]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()

        # Verify new data is queryable
        table = ds.to_table(filter="btree == 1000")
        assert table.num_rows >= 1


@compat_test(min_version="0.39.0")
class BTreeRowAddressDomainIndex(UpgradeDowngradeTest):
    """Test BTREE forward/backward compatibility across the row-address-domain
    format change.

    min_version is 0.39.0, not the usual 0.36.0, because this test's whole
    point is an old build correctly ignoring an index format version newer
    than it understands -- and that mechanism itself was only introduced in
    0.39.0 (#4906). Before that, an old build doesn't inspect index_version
    at all and will happily misuse a too-new BTREE index.

    BTREE was changed to store physical row addresses (``_rowaddr``) instead
    of row ids, which bumped its on-disk format version. This dataset enables
    stable row ids, so the two domains genuinely differ and updating a
    pre-change segment must rebuild it rather than merge new data into it (a
    merge would otherwise silently combine row ids and row addresses in the
    same column). This test checks the full story:

    - An old-format (row-id domain) BTREE index is loaded and used correctly
      by the current build.
    - Updating it (inserting rows, then optimizing) with the current build
      keeps both old and new rows queryable.
    - Readers released before 14.0.0b16 must ignore the updated index and
      answer correctly via a full scan. Readers from 14.0.0b16 onward must
      use it. Both must still be able to write to the dataset afterwards.
    """

    def __init__(self, path: Path):
        self.path = path

    def _supports_row_address_index(self) -> bool:
        # beta.16 is the first released reader for the row-address format.
        # The current checkout's version string may still predate that release,
        # so distinguish it by the harness role, not by lance.__version__.
        return not self._running_in_old_venv or Version(self.compat_version) >= Version(
            "14.0.0b16"
        )

    def create(self):
        """Create a stable-row-id dataset with a BTREE index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "btree": pa.array(range(1000)),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
            enable_stable_row_ids=True,
        )
        dataset.create_scalar_index("btree", "BTREE")

    def _assert_queryable(self, expect_index_used: bool):
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="btree == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        explain = ds.scanner(filter="btree == 7").explain_plan()
        used_index = "ScalarIndexQuery" in explain or "MaterializeIndex" in explain
        if expect_index_used:
            assert used_index, "expected the BTREE index to be used"
        else:
            assert not used_index, (
                "an older build must not use a BTREE index in a format it "
                "does not understand -- it should fall back to a full scan"
            )

    def check_read(self):
        """Use supported index formats and safely ignore too-new ones."""
        self._assert_queryable(expect_index_used=self._supports_row_address_index())

    def check_write(self):
        """Insert a row and update the index, then verify old and new rows
        both stay correct -- whether or not this build can even see the
        index."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "btree": pa.array([1000]),
            }
        )
        ds.insert(data)
        # For a build that cannot see this index at all, there is nothing
        # registered to update, so this is a safe no-op rather than an
        # error. For the current build updating a legacy row-id-domain
        # segment, this must rebuild (not merge) the index -- see the class
        # docstring.
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()

        ds = lance.dataset(self.path)
        table = ds.to_table(filter="btree == 7")
        assert table.num_rows == 1
        # `check_write` runs more than once across the upgrade/downgrade
        # round trip, each time inserting another `btree == 1000` row, so
        # this can't assert an exact count the way the `== 7` row (written
        # once, in `create`) can.
        table = ds.to_table(filter="btree == 1000")
        assert table.num_rows >= 1

        # Preserve the post-compaction result checks above, then restore
        # physical-address coverage before testing the next reader's format
        # support. Older readers cannot rebuild this index format.
        if self._supports_row_address_index():
            ds.optimize.optimize_indices()


@pytest.fixture
def btree_row_address_compat_case(tmp_path):
    path = tmp_path / "data.lance"
    dataset = lance.write_dataset(
        pa.table({"idx": range(16), "btree": range(16)}),
        path,
        max_rows_per_file=8,
        data_storage_version="2.1",
        enable_stable_row_ids=True,
    )
    dataset.create_scalar_index("btree", "BTREE")
    return BTreeRowAddressDomainIndex(path)


@pytest.mark.parametrize(
    "version,running_in_old_venv,expect_index_used",
    [
        pytest.param("14.0.0b15", False, True, id="current-before-release"),
        pytest.param("14.0.0b16", False, True, id="current-same-version"),
        pytest.param("13.0.0", True, False, id="older-stable"),
        pytest.param("14.0.0b15", True, False, id="last-unsupported-beta"),
        pytest.param("14.0.0b16", True, True, id="first-supported-beta"),
        pytest.param("14.0.0-beta.16", True, True, id="cargo-beta-spelling"),
        pytest.param("14.0.0", True, True, id="stable-release"),
    ],
)
def test_btree_row_address_reader_capability(
    btree_row_address_compat_case,
    monkeypatch,
    version,
    running_in_old_venv,
    expect_index_used,
):
    case = btree_row_address_compat_case
    monkeypatch.setattr(lance, "__version__", version)
    case.compat_version = version
    case._running_in_old_venv = running_in_old_venv
    dataset = lance.dataset(case.path)
    if not expect_index_used:
        dataset.drop_index("btree_idx")
    case.check_read()

    # The opposite plan must still fail: neither reader capability may be
    # treated as permission to silently stop checking index usage.
    if expect_index_used:
        dataset.drop_index("btree_idx")
        message = "expected the BTREE index to be used"
    else:
        dataset.create_scalar_index("btree", "BTREE")
        message = "an older build must not use a BTREE index"
    with pytest.raises(AssertionError, match=message):
        case.check_read()


@pytest.mark.parametrize("running_in_old_venv", [False, True])
def test_btree_row_address_maintenance_restores_coverage(
    btree_row_address_compat_case, running_in_old_venv
):
    case = btree_row_address_compat_case
    case.compat_version = "14.0.0b16"
    case._running_in_old_venv = running_in_old_venv
    case.check_write()
    case.check_read()
    assert lance.dataset(case.path).count_rows("btree == 1000") == 1


@compat_test(min_version="0.30.0")
class BTreeRowIdDomainIndex(UpgradeDowngradeTest):
    """Test that a BTREE index on a dataset *without* stable row ids stays
    usable -- not just readable -- by an older build, no matter which build
    wrote it.

    BTREE moved to storing physical row addresses (``_rowaddr``) instead of
    row ids, bumping its on-disk format version to 1 (see
    BTreeRowAddressDomainIndex above). But a row id and a row address are the
    same value on a dataset that does not enable stable row ids, so the
    current build keeps writing the old format (index_version 0) there --
    on creation, and again on every rebuild during optimize -- specifically
    so a user who never turns on stable row ids sees no
    forward-compatibility impact from that migration.

    Unlike BTreeIndex above, this asserts the index is genuinely *used*
    (not answered via a full-scan fallback) in every direction: right after
    the current build creates it, and after the current build has rebuilt
    it with optimize_indices() -- the scenario that would have forced a
    fallback if the rebuild had silently bumped the format version.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create a dataset without stable row ids and a BTREE index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "btree": pa.array(range(1000)),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_scalar_index("btree", "BTREE")

    def _assert_queryable_and_indexed(self):
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="btree == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        explain = ds.scanner(filter="btree == 7").explain_plan()
        assert "ScalarIndexQuery" in explain or "MaterializeIndex" in explain, (
            "a BTREE index on a dataset without stable row ids must stay "
            "usable by every build, not fall back to a full scan"
        )

    def check_read(self):
        """The index must be used, whichever build wrote it."""
        self._assert_queryable_and_indexed()

    def check_write(self):
        """Insert a row and rebuild the index, then verify it is still
        queryable -- and still actually used, including by an older build
        reading what the current build just rebuilt."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "btree": pa.array([1000]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()

        ds = lance.dataset(self.path)
        table = ds.to_table(filter="btree == 7")
        assert table.num_rows == 1
        # `check_write` runs more than once across the upgrade/downgrade
        # round trip, each time inserting another `btree == 1000` row, so
        # this can't assert an exact count the way the `== 7` row (written
        # once, in `create`) can.
        table = ds.to_table(filter="btree == 1000")
        assert table.num_rows >= 1
        self._assert_queryable_and_indexed()


@compat_test(min_version="0.22.0")
class BitmapLabelListIndex(UpgradeDowngradeTest):
    """Test BITMAP and LABEL_LIST scalar index compatibility (introduced in 0.20.0).

    Started fully working in 0.22.0 with fixes to LABEL_LIST index.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with BITMAP and LABEL_LIST indices."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "bitmap": pa.array(range(1000)),
                "label_list": pa.array([[f"label{i}"] for i in range(1000)]),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_scalar_index("bitmap", "BITMAP")
        dataset.create_scalar_index("label_list", "LABEL_LIST")

    def check_read(self):
        """Verify BITMAP and LABEL_LIST indices can be queried."""
        ds = lance.dataset(self.path)

        # Test BITMAP index
        table = ds.to_table(filter="bitmap == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        # Test LABEL_LIST index
        table = ds.to_table(filter="array_has_any(label_list, ['label7'])")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

    def check_write(self):
        """Verify can insert data and optimize indices."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "bitmap": pa.array([1000]),
                "label_list": pa.array([["label1000"]]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="0.39.0")
class BitmapRowAddressDomainIndex(UpgradeDowngradeTest):
    """Test BITMAP forward/backward compatibility across the row-address-domain
    format change.

    min_version is 0.39.0, not BitmapLabelListIndex's 0.22.0, because this
    test's whole point is an old build correctly ignoring an index format
    version newer than it understands -- and that mechanism itself was only
    introduced in 0.39.0 (#4906). Before that, an old build doesn't inspect
    index_version at all and will happily misuse a too-new BITMAP index.

    BITMAP was changed to store physical row addresses (``_rowaddr``)
    instead of row ids, which bumped its on-disk format version. This
    dataset enables stable row ids, so the two domains genuinely differ and
    updating a pre-change segment must rebuild it rather than merge new
    data into it (a merge would otherwise silently combine row ids and row
    addresses in the same column). This test checks the full story:

    - An old-format (row-id domain) BITMAP index is loaded and used
      correctly by the current build.
    - Updating it (inserting rows, then optimizing) with the current build
      keeps both old and new rows queryable.
    - Readers released before 15.0.0b2 must ignore the updated index and
      answer correctly via a full scan. Readers from 15.0.0b2 onward must
      use it. Both must still be able to write to the dataset afterwards.
    """

    def __init__(self, path: Path):
        self.path = path

    def _supports_row_address_index(self) -> bool:
        # 15.0.0b2 is the first release containing the row-address format.
        return not self._running_in_old_venv or Version(self.compat_version) >= Version(
            "15.0.0b2"
        )

    def create(self):
        """Create a stable-row-id dataset with a BITMAP index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "bitmap": pa.array(range(1000)),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
            enable_stable_row_ids=True,
        )
        dataset.create_scalar_index("bitmap", "BITMAP")

    def _assert_queryable(self, expect_index_used: bool):
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="bitmap == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        explain = ds.scanner(filter="bitmap == 7").explain_plan()
        used_index = "ScalarIndexQuery" in explain or "MaterializeIndex" in explain
        if expect_index_used:
            assert used_index, "expected the BITMAP index to be used"
        else:
            assert not used_index, (
                "an older build must not use a BITMAP index in a format it "
                "does not understand -- it should fall back to a full scan"
            )

    def check_read(self):
        """Use supported index formats and safely ignore too-new ones."""
        self._assert_queryable(expect_index_used=self._supports_row_address_index())

    def check_write(self):
        """Insert a row and update the index, then verify old and new rows
        both stay correct -- whether or not this build can even see the
        index."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "bitmap": pa.array([1000]),
            }
        )
        ds.insert(data)
        # For a build that cannot see this index at all, there is nothing
        # registered to update, so this is a safe no-op rather than an
        # error. For the current build updating a legacy row-id-domain
        # segment, this must rebuild (not merge) the index -- see the class
        # docstring.
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()

        ds = lance.dataset(self.path)
        table = ds.to_table(filter="bitmap == 7")
        assert table.num_rows == 1
        # `check_write` runs more than once across the upgrade/downgrade
        # round trip, each time inserting another `bitmap == 1000` row, so
        # this can't assert an exact count the way the `== 7` row (written
        # once, in `create`) can.
        table = ds.to_table(filter="bitmap == 1000")
        assert table.num_rows >= 1


@compat_test(min_version="0.22.0")
class BitmapRowIdDomainIndex(UpgradeDowngradeTest):
    """Test that a BITMAP index on a dataset *without* stable row ids stays
    usable -- not just readable -- by an older build, no matter which build
    wrote it.

    BITMAP moved to storing physical row addresses (``_rowaddr``) instead of
    row ids, bumping its on-disk format version to 1 (see
    BitmapRowAddressDomainIndex above). But a row id and a row address are
    the same value on a dataset that does not enable stable row ids, so the
    current build keeps writing the old format (index_version 0) there --
    on creation, and again on every rebuild during optimize -- specifically
    so a user who never turns on stable row ids sees no
    forward-compatibility impact from that migration.

    Unlike BitmapLabelListIndex above, this asserts the index is genuinely
    *used* (not answered via a full-scan fallback) in every direction: right
    after the current build creates it, and after the current build has
    rebuilt it with optimize_indices() -- the scenario that would have
    forced a fallback if the rebuild had silently bumped the format
    version.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create a dataset without stable row ids and a BITMAP index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "bitmap": pa.array(range(1000)),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_scalar_index("bitmap", "BITMAP")

    def _assert_queryable_and_indexed(self):
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="bitmap == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        explain = ds.scanner(filter="bitmap == 7").explain_plan()
        assert "ScalarIndexQuery" in explain or "MaterializeIndex" in explain, (
            "a BITMAP index on a dataset without stable row ids must stay "
            "usable by every build, not fall back to a full scan"
        )

    def check_read(self):
        """The index must be used, whichever build wrote it."""
        self._assert_queryable_and_indexed()

    def check_write(self):
        """Insert a row and rebuild the index, then verify it is still
        queryable -- and still actually used, including by an older build
        reading what the current build just rebuilt."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "bitmap": pa.array([1000]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()

        ds = lance.dataset(self.path)
        table = ds.to_table(filter="bitmap == 7")
        assert table.num_rows == 1
        # `check_write` runs more than once across the upgrade/downgrade
        # round trip, each time inserting another `bitmap == 1000` row, so
        # this can't assert an exact count the way the `== 7` row (written
        # once, in `create`) can.
        table = ds.to_table(filter="bitmap == 1000")
        assert table.num_rows >= 1
        self._assert_queryable_and_indexed()


@compat_test(min_version="0.39.0")
class LabelListRowAddressDomainIndex(UpgradeDowngradeTest):
    """Test LABEL_LIST forward/backward compatibility across the
    row-address-domain format change.

    min_version is 0.39.0, not BitmapLabelListIndex's 0.22.0, because this
    test's whole point is an old build correctly ignoring an index format
    version newer than it understands -- and that mechanism itself was only
    introduced in 0.39.0 (#4906). Before that, an old build doesn't inspect
    index_version at all and will happily misuse a too-new LABEL_LIST index.

    LABEL_LIST was changed to store physical row addresses (``_rowaddr``)
    instead of row ids, which bumped its on-disk format version. This
    dataset enables stable row ids, so the two domains genuinely differ and
    updating a pre-change segment must rebuild it rather than merge new
    data into it (a merge would otherwise silently combine row ids and row
    addresses in the same column). This test checks the full story:

    - An old-format (row-id domain) LABEL_LIST index is loaded and used
      correctly by the current build.
    - Updating it (inserting rows, then optimizing) with the current build
      keeps both old and new rows queryable.
    - Readers released before 15.0.0b2 must ignore the updated index and
      answer correctly via a full scan. Readers from 15.0.0b2 onward must
      use it. Both must still be able to write to the dataset afterwards.
    """

    def __init__(self, path: Path):
        self.path = path

    def _supports_row_address_index(self) -> bool:
        # 15.0.0b2 is the first release containing the row-address format.
        return not self._running_in_old_venv or Version(self.compat_version) >= Version(
            "15.0.0b2"
        )

    def create(self):
        """Create a stable-row-id dataset with a LABEL_LIST index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "label_list": pa.array([[f"label{i}"] for i in range(1000)]),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
            enable_stable_row_ids=True,
        )
        dataset.create_scalar_index("label_list", "LABEL_LIST")

    def _assert_queryable(self, expect_index_used: bool):
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="array_has_any(label_list, ['label7'])")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        explain = ds.scanner(
            filter="array_has_any(label_list, ['label7'])"
        ).explain_plan()
        used_index = "ScalarIndexQuery" in explain or "MaterializeIndex" in explain
        if expect_index_used:
            assert used_index, "expected the LABEL_LIST index to be used"
        else:
            assert not used_index, (
                "an older build must not use a LABEL_LIST index in a format "
                "it does not understand -- it should fall back to a full scan"
            )

    def check_read(self):
        """Use supported index formats and safely ignore too-new ones."""
        self._assert_queryable(expect_index_used=self._supports_row_address_index())

    def check_write(self):
        """Insert a row and update the index, then verify old and new rows
        both stay correct -- whether or not this build can even see the
        index."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "label_list": pa.array([["label1000"]]),
            }
        )
        ds.insert(data)
        # For a build that cannot see this index at all, there is nothing
        # registered to update, so this is a safe no-op rather than an
        # error. For the current build updating a legacy row-id-domain
        # segment, this must rebuild (not merge) the index -- see the class
        # docstring.
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()

        ds = lance.dataset(self.path)
        table = ds.to_table(filter="array_has_any(label_list, ['label7'])")
        assert table.num_rows == 1
        # `check_write` runs more than once across the upgrade/downgrade
        # round trip, each time inserting another row, so this can't assert
        # an exact count the way the `label7` row (written once, in
        # `create`) can.
        table = ds.to_table(filter="array_has_any(label_list, ['label1000'])")
        assert table.num_rows >= 1


@compat_test(min_version="0.22.0")
class LabelListRowIdDomainIndex(UpgradeDowngradeTest):
    """Test that a LABEL_LIST index on a dataset *without* stable row ids
    stays usable -- not just readable -- by an older build, no matter which
    build wrote it.

    LABEL_LIST moved to storing physical row addresses (``_rowaddr``)
    instead of row ids, bumping its on-disk format version to 2 (see
    LabelListRowAddressDomainIndex above). But a row id and a row address
    are the same value on a dataset that does not enable stable row ids, so
    the current build keeps writing the old format (index_version 1) there
    -- on creation, and again on every rebuild during optimize --
    specifically so a user who never turns on stable row ids sees no
    forward-compatibility impact from that migration.

    Unlike BitmapLabelListIndex above, this asserts the index is genuinely
    *used* (not answered via a full-scan fallback) in every direction: right
    after the current build creates it, and after the current build has
    rebuilt it with optimize_indices() -- the scenario that would have
    forced a fallback if the rebuild had silently bumped the format
    version.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create a dataset without stable row ids and a LABEL_LIST index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "label_list": pa.array([[f"label{i}"] for i in range(1000)]),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_scalar_index("label_list", "LABEL_LIST")

    def _assert_queryable_and_indexed(self):
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="array_has_any(label_list, ['label7'])")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        explain = ds.scanner(
            filter="array_has_any(label_list, ['label7'])"
        ).explain_plan()
        assert "ScalarIndexQuery" in explain or "MaterializeIndex" in explain, (
            "a LABEL_LIST index on a dataset without stable row ids must "
            "stay usable by every build, not fall back to a full scan"
        )

    def check_read(self):
        """The index must be used, whichever build wrote it."""
        self._assert_queryable_and_indexed()

    def check_write(self):
        """Insert a row and rebuild the index, then verify it is still
        queryable -- and still actually used, including by an older build
        reading what the current build just rebuilt."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "label_list": pa.array([["label1000"]]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()

        ds = lance.dataset(self.path)
        table = ds.to_table(filter="array_has_any(label_list, ['label7'])")
        assert table.num_rows == 1
        # `check_write` runs more than once across the upgrade/downgrade
        # round trip, each time inserting another row, so this can't assert
        # an exact count the way the `label7` row (written once, in
        # `create`) can.
        table = ds.to_table(filter="array_has_any(label_list, ['label1000'])")
        assert table.num_rows >= 1
        self._assert_queryable_and_indexed()


@compat_test(min_version="0.36.0")
class NgramIndex(UpgradeDowngradeTest):
    """Test NGRAM index compatibility (introduced in 0.36.0)."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with NGRAM index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "ngram": pa.array([f"word{i}" for i in range(1000)]),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_scalar_index("ngram", "NGRAM")

    def check_read(self):
        """Verify NGRAM index can be queried."""
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="contains(ngram, 'word7')")
        # word7, word70-79, word700-799 = 111 results
        assert table.num_rows == 111

        # Verify index is used
        explain = ds.scanner(filter="contains(ngram, 'word7')").explain_plan()
        assert "ScalarIndexQuery" in explain

    def check_write(self):
        """Verify can insert data and optimize NGRAM index."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "ngram": pa.array(["word1000"]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="0.36.0")
class ZonemapBloomfilterIndex(UpgradeDowngradeTest):
    """Test ZONEMAP and BLOOMFILTER index compatibility (introduced in 0.36.0)."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with ZONEMAP and BLOOMFILTER indices.

        The zonemap column contains nulls at rows 0 and 500 so that IS NULL
        queries can be verified across version boundaries.
        """
        shutil.rmtree(self.path, ignore_errors=True)
        zonemap_values = [None if i in (0, 500) else i for i in range(1000)]
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "zonemap": pa.array(zonemap_values, type=pa.int64()),
                "bloomfilter": pa.array(range(1000)),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_scalar_index("zonemap", "ZONEMAP")
        dataset.create_scalar_index("bloomfilter", "BLOOMFILTER")

    def check_read(self):
        """Verify ZONEMAP and BLOOMFILTER indices can be queried."""
        ds = lance.dataset(self.path)

        # Test ZONEMAP equality
        table = ds.to_table(filter="zonemap == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        # Test ZONEMAP IS NULL — two nulls were inserted at rows 0 and 500.
        # Older versions without a null bitmap fall back to a zone scan, which
        # is still correct; newer versions may return an exact result.
        table = ds.to_table(filter="zonemap IS NULL")
        if 1000 in table.column("idx").to_pylist():
            # After write, there are 3 NULLs
            assert table.num_rows == 3
        else:
            # Before write, there are 2 NULLs
            assert table.num_rows == 2

        # Test BLOOMFILTER
        table = ds.to_table(filter="bloomfilter == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

    def check_write(self):
        """Verify can insert data and optimize indices."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "zonemap": pa.array([None], type=pa.int64()),
                "bloomfilter": pa.array([1000]),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()

        # IS NULL must still return results after the index is updated and
        # files are compacted.  The newly inserted null must be found
        # regardless of which version handles the seed-based index update.
        table = ds.to_table(filter="zonemap IS NULL")
        assert table.num_rows >= 1

    def skip_downgrade(self, version: str) -> bool:
        # In 0.X the zonemap index did not properly handle NULL in filters
        return version.startswith("0.")


@compat_test(min_version="0.36.0")
class JsonIndex(UpgradeDowngradeTest):
    """Test JSON index compatibility (introduced in 0.36.0)."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with JSON index."""
        from lance.indices import IndexConfig

        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "json": pa.array([f'{{"val": {i}}}' for i in range(1000)], pa.json_()),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_scalar_index(
            "json",
            IndexConfig(
                index_type="json",
                parameters={"target_index_type": "btree", "path": "val"},
            ),
        )

    def check_read(self):
        """Verify JSON index can be queried."""
        ds = lance.dataset(self.path)
        table = ds.to_table(filter="json_get_int(json, 'val') == 7")
        assert table.num_rows == 1
        assert table.column("idx").to_pylist() == [7]

        # Verify index is used
        explain = ds.scanner(filter="json_get_int(json, 'val') == 7").explain_plan()
        assert "ScalarIndexQuery" in explain

    def check_write(self):
        """Verify can insert data with JSON index."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "json": pa.array(['{"val": 1000}'], pa.json_()),
            }
        )
        ds.insert(data)
        ds.optimize.compact_files()


@compat_test(min_version="0.36.0")
class FtsIndex(UpgradeDowngradeTest):
    """Test FTS (full-text search) index compatibility (introduced in 0.36.0)."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with FTS index."""
        shutil.rmtree(self.path, ignore_errors=True)
        data = pa.table(
            {
                "idx": pa.array(range(1000)),
                "text": pa.array(
                    [f"document with words {i} and more text" for i in range(1000)]
                ),
            }
        )
        dataset = lance.write_dataset(
            data,
            self.path,
            max_rows_per_file=100,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        kwargs = {"with_position": True}
        # Downgrade reads use older wheels, so current-created FTS indexes must
        # stay on the legacy posting block layout.
        if os.environ.get("LANCE_COMPAT_FTS_LEGACY_BLOCK_SIZE") == "1":
            kwargs["block_size"] = 128
        dataset.create_scalar_index("text", "INVERTED", format_version=1, **kwargs)

    def check_read(self):
        """Verify FTS index can be queried."""
        ds = lance.dataset(self.path)
        match_table = ds.to_table(
            full_text_query={"query": "words 7", "columns": ["text"]}
        )
        assert match_table.num_rows > 0
        assert 7 in match_table.column("idx").to_pylist()

    def check_write(self):
        """Verify can insert data with FTS index."""
        # Dataset::load_manifest does not do retain_supported_indices
        # so this can only work with no cache
        session = lance.Session(index_cache_size_bytes=0, metadata_cache_size_bytes=0)
        ds = lance.dataset(self.path, session=session)
        data = pa.table(
            {
                "idx": pa.array([1000]),
                "text": pa.array(["new document to index"]),
            }
        )
        ds.insert(data)
        ds.optimize.compact_files()

    def skip_downgrade(self, version: str) -> bool:
        return version.startswith("0.")

    def current_env(self, method_name: str) -> dict[str, str]:
        if method_name == "create":
            return {
                "LANCE_COMPAT_FTS_LEGACY_BLOCK_SIZE": "1",
                "LANCE_FTS_FORMAT_VERSION": "1",
            }
        if method_name == "check_write":
            return {"LANCE_FTS_FORMAT_VERSION": "2"}
        return {}

    def compat_env(self, version: str, method_name: str) -> dict[str, str]:
        if method_name in {"create", "check_write"}:
            return {"LANCE_FTS_FORMAT_VERSION": "1"}
        return {}
