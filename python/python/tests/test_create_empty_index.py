# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for creating empty indices with train=False."""

import lance
import pyarrow as pa
import pyarrow.compute as pc


def test_create_empty_scalar_index():
    data = pa.table({"id": range(100)})
    dataset = lance.write_dataset(data, "memory://")

    # Passing train=False to create an empty index
    dataset.create_scalar_index("id", "BTREE", train=False)

    # Verify index exists and has correct stats
    indices = dataset.describe_indices()
    assert len(indices) == 1
    assert indices[0].index_type == "BTree"
    stats = dataset.stats.index_stats(indices[0].name)
    assert stats["num_indexed_rows"] == 0
    assert stats["num_unindexed_rows"] == dataset.count_rows()


def test_create_empty_vector_index():
    dim = 32
    values = pc.random(100 * dim).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, dim)
    data = pa.table({"vector": vectors})
    dataset = lance.write_dataset(data, "memory://")

    dataset.create_index(
        "vector", "IVF_PQ", num_partitions=10, num_sub_vectors=8, train=False
    )

    # Same shape the scalar case reports: listed, covering nothing yet.
    indices = dataset.describe_indices()
    assert len(indices) == 1
    stats = dataset.stats.index_stats(indices[0].name)
    assert stats["num_indexed_rows"] == 0
    assert stats["num_unindexed_rows"] == dataset.count_rows()


def test_create_vector_index_below_the_row_floor():
    """A table too small to train the quantizer takes the index anyway."""
    dim = 32
    values = pc.random(100 * dim).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, dim)
    data = pa.table({"vector": vectors})
    dataset = lance.write_dataset(data, "memory://")

    # 100 vectors cannot train a 256-code codebook, and train defaults to True.
    dataset.create_index("vector", "IVF_PQ", num_partitions=10, num_sub_vectors=8)

    indices = dataset.describe_indices()
    assert len(indices) == 1
    stats = dataset.stats.index_stats(indices[0].name)
    assert stats["num_indexed_rows"] == 0
    assert stats["num_unindexed_rows"] == dataset.count_rows()
