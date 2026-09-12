# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for creating empty indices with train=False."""

import pytest

import lance
import pyarrow as pa
import pyarrow.compute as pc


@pytest.mark.parametrize("index_type", ["BTREE", "BITMAP"])
def test_create_empty_scalar_index(index_type):
    data = pa.table({"id": range(100)})
    dataset = lance.write_dataset(data, "memory://")

    dataset.create_scalar_index("id", index_type, train=False)

    indices = dataset.describe_indices()
    assert len(indices) == 1
    stats = dataset.stats.index_stats(indices[0].name)
    assert stats["num_indexed_rows"] == 0
    assert stats["num_unindexed_rows"] == dataset.count_rows()
    assert len(stats["indices"]) == 1
    assert stats["indices"][0]["num_rows"] == 0
    assert stats["indices"][0]["index_type"].upper().startswith(index_type[:4])


def test_create_empty_fts_index():
    data = pa.table({"text": ["hello world", "foo bar", "lance db"]})
    dataset = lance.write_dataset(data, "memory://")

    dataset.create_scalar_index("text", "FTS", train=False)

    indices = dataset.describe_indices()
    assert len(indices) == 1
    stats = dataset.stats.index_stats(indices[0].name)
    assert stats["num_indexed_rows"] == 0
    assert stats["num_unindexed_rows"] == dataset.count_rows()
    assert len(stats["indices"]) == 1
    assert stats["indices"][0]["num_rows"] == 0


def test_create_empty_vector_index():
    dim = 32
    values = pc.random(100 * dim).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, dim)
    data = pa.table({"vector": vectors})
    dataset = lance.write_dataset(data, "memory://")

    dataset.create_index(
        "vector", "IVF_PQ", num_partitions=10, num_sub_vectors=8, train=False
    )

    indices = dataset.describe_indices()
    assert len(indices) == 1
    stats = dataset.stats.index_stats(indices[0].name)
    assert stats["num_indexed_rows"] == 0
    assert stats["num_unindexed_rows"] == dataset.count_rows()
    assert len(stats["indices"]) == 1
    assert stats["indices"][0]["num_rows"] == 0
    assert stats["indices"][0]["index_type"] == "Vector"


def test_create_empty_index_with_name():
    data = pa.table({"id": range(100)})
    dataset = lance.write_dataset(data, "memory://")

    dataset.create_scalar_index("id", "BTREE", name="my_custom_idx", train=False)

    indices = dataset.describe_indices()
    assert len(indices) == 1
    assert indices[0].name == "my_custom_idx"
    stats = dataset.stats.index_stats("my_custom_idx")
    assert stats["num_indexed_rows"] == 0
    assert stats["num_unindexed_rows"] == dataset.count_rows()


def test_create_multiple_empty_indices():
    dim = 32
    values = pc.random(50 * dim).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, dim)
    data = pa.table({"id": range(50), "text": ["a"] * 50, "vector": vectors})
    dataset = lance.write_dataset(data, "memory://")

    dataset.create_scalar_index("id", "BTREE", train=False)
    dataset.create_scalar_index("text", "FTS", train=False)
    dataset.create_index(
        "vector", "IVF_PQ", num_partitions=5, num_sub_vectors=8, train=False
    )

    indices = dataset.describe_indices()
    assert len(indices) == 3
    for idx in indices:
        stats = dataset.stats.index_stats(idx.name)
        assert stats["num_indexed_rows"] == 0
        assert stats["num_unindexed_rows"] == dataset.count_rows()
        assert len(stats["indices"]) == 1
        assert stats["indices"][0]["num_rows"] == 0
