# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Vector index compatibility tests for Lance.

Tests that vector indices (IVF_PQ, etc.) created with one version of Lance
can be read and written by other versions.
"""

import os
import shutil
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .compat_decorator import (
    UpgradeDowngradeTest,
    compat_test,
)
from .util import safe_data_storage_version


@compat_test(min_version="0.29.1.beta2")
class PqVectorIndex(UpgradeDowngradeTest):
    """Test PQ (Product Quantization) vector index compatibility."""

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with PQ vector index."""
        shutil.rmtree(self.path, ignore_errors=True)
        ndims = 32
        nvecs = 512

        data = pa.table(
            {
                "id": pa.array(range(nvecs)),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(ndims * nvecs).cast(pa.float32()), ndims
                ),
            }
        )

        dataset = lance.write_dataset(
            data,
            self.path,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_index(
            "vec",
            "IVF_PQ",
            num_partitions=1,
            num_sub_vectors=4,
        )

    def check_read(self):
        """Verify PQ index can be queried."""
        ds = lance.dataset(self.path)
        # Query with random vector
        q = pc.random(32).cast(pa.float32())
        result = ds.to_table(
            nearest={
                "q": q,
                "k": 4,
                "column": "vec",
            }
        )
        assert result.num_rows == 4

        if hasattr(ds, "describe_indices"):
            indices = ds.describe_indices()
            assert len(indices) >= 1
            name = indices[0].name
        elif self.compat_version >= "0.39.0":
            indices = ds.list_indices()
            assert len(indices) >= 1
            name = indices[0]["name"]
            stats = ds.stats.index_stats(name)
            assert stats["num_indexed_rows"] > 0

    def check_write(self):
        """Verify can insert vectors and rebuild index."""
        ds = lance.dataset(self.path)
        # Add new vectors
        data = pa.table(
            {
                "id": pa.array([1000]),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(32).cast(pa.float32()), 32
                ),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="0.39.0")
class HnswPqVectorIndex(UpgradeDowngradeTest):
    """Test IVF_HNSW_PQ vector index compatibility.

    Note: Only tests versions >= 0.39.0 because earlier versions don't support
    remapping for IVF_HNSW_PQ indices, which is required for optimize operations.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with IVF_HNSW_PQ vector index."""
        shutil.rmtree(self.path, ignore_errors=True)
        ndims = 32
        nvecs = 512

        data = pa.table(
            {
                "id": pa.array(range(nvecs)),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(ndims * nvecs).cast(pa.float32()), ndims
                ),
            }
        )

        dataset = lance.write_dataset(
            data,
            self.path,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_index(
            "vec",
            "IVF_HNSW_PQ",
            num_partitions=4,
            num_sub_vectors=4,
        )

    def check_read(self):
        """Verify IVF_HNSW_PQ index can be queried."""
        ds = lance.dataset(self.path)
        # Query with random vector
        q = pc.random(32).cast(pa.float32())
        result = ds.to_table(
            nearest={
                "q": q,
                "k": 4,
                "column": "vec",
            }
        )
        assert result.num_rows == 4

        if hasattr(ds, "describe_indices"):
            indices = ds.describe_indices()
            assert len(indices) >= 1
            name = indices[0].name
        else:
            indices = ds.list_indices()
            assert len(indices) >= 1
            name = indices[0]["name"]

        stats = ds.stats.index_stats(name)
        assert stats["num_indexed_rows"] > 0

    def check_write(self):
        """Verify can insert vectors and rebuild index."""
        ds = lance.dataset(self.path)
        # Add new vectors
        data = pa.table(
            {
                "id": pa.array([1000]),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(32).cast(pa.float32()), 32
                ),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="0.39.0")
class HnswSqVectorIndex(UpgradeDowngradeTest):
    """Test IVF_HNSW_SQ vector index compatibility.

    Note: Only tests versions >= 0.39.0 because earlier versions don't support
    remapping for IVF_HNSW_SQ indices, which is required for optimize operations.
    """

    def __init__(self, path: Path):
        self.path = path

    def create(self):
        """Create dataset with IVF_HNSW_SQ vector index."""
        shutil.rmtree(self.path, ignore_errors=True)
        ndims = 32
        nvecs = 512

        data = pa.table(
            {
                "id": pa.array(range(nvecs)),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(ndims * nvecs).cast(pa.float32()), ndims
                ),
            }
        )

        dataset = lance.write_dataset(
            data,
            self.path,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_index(
            "vec",
            "IVF_HNSW_SQ",
            num_partitions=4,
            num_sub_vectors=4,
        )

    def check_read(self):
        """Verify IVF_HNSW_SQ index can be queried."""
        ds = lance.dataset(self.path)
        # Query with random vector
        q = pc.random(32).cast(pa.float32())
        result = ds.to_table(
            nearest={
                "q": q,
                "k": 4,
                "column": "vec",
            }
        )
        assert result.num_rows == 4

        if hasattr(ds, "describe_indices"):
            indices = ds.describe_indices()
            assert len(indices) >= 1
            name = indices[0].name
        else:
            indices = ds.list_indices()
            assert len(indices) >= 1
            name = indices[0]["name"]

        stats = ds.stats.index_stats(name)
        assert stats["num_indexed_rows"] > 0

    def check_write(self):
        """Verify can insert vectors and rebuild index."""
        ds = lance.dataset(self.path)
        # Add new vectors
        data = pa.table(
            {
                "id": pa.array([1000]),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(32).cast(pa.float32()), 32
                ),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()


@compat_test(min_version="4.0.0-beta.8")
class IvfRqVectorIndex(UpgradeDowngradeTest):
    """Test IVF_RQ vector index compatibility. V2 was introduced in v4.0.0-beta.8"""

    def __init__(self, path: Path):
        self.path = path

    def current_env(self, method_name: str):
        if method_name == "check_read":
            return {"LANCE_COMPAT_CURRENT_RUNTIME": "1"}
        return {}

    def skip_read_after_current_write(self, version: str) -> bool:
        # Newly written IVF_RQ indexes carry raw-query estimator metadata and
        # split-code schema that older runtimes cannot query or optimize safely.
        # The upgrade_downgrade variant still covers old 1-bit residual-query
        # indexes being read and rewritten by the current runtime.
        return True

    def skip_write_after_current_write(self, version: str) -> bool:
        return True

    def create(self):
        """Create dataset with IVF_RQ vector index."""
        shutil.rmtree(self.path, ignore_errors=True)
        ndims = 32
        nvecs = 512

        data = pa.table(
            {
                "id": pa.array(range(nvecs)),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(ndims * nvecs).cast(pa.float32()), ndims
                ),
            }
        )

        dataset = lance.write_dataset(
            data,
            self.path,
            data_storage_version=safe_data_storage_version(self.compat_version),
        )
        dataset.create_index(
            "vec",
            "IVF_RQ",
            num_partitions=4,
            num_bits=1,
        )

    def check_read(self):
        """Verify vector query can run (indexed or brute-force fallback)."""
        ds = lance.dataset(self.path)
        q = np.random.random(32).astype(np.float32)
        result = ds.to_table(
            nearest={
                "q": q,
                "k": 4,
                "column": "vec",
            }
        )
        assert result.num_rows == 4

        if hasattr(ds, "describe_indices"):
            indices = ds.describe_indices()
            assert len(indices) >= 1
            name = indices[0].name
        else:
            indices = ds.list_indices()
            assert len(indices) >= 1
            name = indices[0].name

        stats = ds.stats.index_stats(name)
        assert stats["num_indexed_rows"] > 0
        if os.environ.get("LANCE_COMPAT_CURRENT_RUNTIME") == "1":
            # Old 1-bit IVF_RQ indexes do not have split ex-code columns.
            # The successful query above verifies the current reader does not
            # require them.
            sub_index = stats["indices"][0]["sub_index"]
            assert sub_index["num_bits"] == 1

    def check_write(self):
        """Verify can insert vectors and run optimize workflows."""
        ds = lance.dataset(self.path)
        data = pa.table(
            {
                "id": pa.array([1000]),
                "vec": pa.FixedSizeListArray.from_arrays(
                    pc.random(32).cast(pa.float32()), 32
                ),
            }
        )
        ds.insert(data)
        ds.optimize.optimize_indices()
        ds.optimize.compact_files()
