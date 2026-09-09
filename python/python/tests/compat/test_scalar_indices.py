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

        # Verify index is used
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
