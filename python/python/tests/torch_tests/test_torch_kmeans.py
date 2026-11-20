# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

from lance.torch import preferred_device  # noqa: E402
from lance.torch.distance import dot_distance, l2_distance  # noqa: E402
from lance.torch.kmeans import KMeans  # noqa: E402
from lance.util import validate_vector_index  # noqa: E402
from lance.vector import train_ivf_centroids_on_accelerator  # noqa: E402


def test_kmeans():
    arr = np.array(range(128)).reshape(-1, 8).astype(np.float32)
    # These duplicate centroids reproduce the empty clusters that made this test
    # depend on random initialization before empty-cluster recovery was fixed.
    centroids = torch.from_numpy(arr[[5, 5, 13, 13]])
    kmeans = KMeans(4, centroids=centroids, device="cpu")
    kmeans.fit(arr)

    cluster_ids = kmeans.transform(arr)
    _, cnts = torch.unique(cluster_ids, return_counts=True)
    assert (cnts >= 1).all() and (cnts <= 8).all()
    assert len(cnts) == 4  # all cluster has data


def test_kmeans_dot_updates_centroids_for_large_norm_vectors():
    # For vectors with norm > 1 the dot distance (1 - x.y) sums to a negative
    # value, which must not be treated as converged before the first update.
    centers = np.array([[10, 0], [0, 10], [-10, -10]], dtype=np.float32)
    noise = np.random.default_rng(0).normal(scale=0.5, size=(300, 2))
    data = np.repeat(centers, 100, axis=0) + noise.astype(np.float32)
    # One initial centroid per blob, so no cluster empties out.
    initial = torch.from_numpy(data[[0, 100, 200]].copy())

    kmeans = KMeans(3, metric="dot", centroids=initial, device="cpu")
    kmeans.fit(data)

    blob_means = torch.from_numpy(data.reshape(3, 100, 2).mean(axis=1))
    assert torch.allclose(kmeans.centroids, blob_means, atol=1e-4)


@pytest.mark.parametrize(
    "metric,expected",
    [("L2", "l2"), ("euclidean", "l2"), ("COSINE", "cosine"), ("Dot", "dot")],
)
def test_kmeans_normalizes_metric(metric, expected):
    kmeans = KMeans(4, metric=metric, device="cpu")
    assert kmeans.metric == expected
    assert kmeans.dist_func is (dot_distance if expected == "dot" else l2_distance)


def test_torch_kmeans_accept_torch_device(tmp_path: Path):
    values = pa.array(np.array(range(128)).astype(np.float32))
    arr = pa.FixedSizeListArray.from_arrays(values, 8)
    tbl = pa.Table.from_arrays([arr], ["vector"])
    ds = lance.write_dataset(tbl, tmp_path)
    # Not raising exception if pass a `torch.device()` directly
    train_ivf_centroids_on_accelerator(
        ds,
        "vector",
        2,
        metric_type="L2",
        accelerator=torch.device("cpu"),
    )


def test_torch_kmeans_nans(tmp_path: Path):
    kmeans = KMeans(4, centroids=torch.rand(4, 8), device="cpu")
    values = np.arange(64).astype(np.float32)
    np.put(values, range(8, 16), np.nan)
    values = pa.array(values)
    fsl = pa.FixedSizeListArray.from_arrays(values, 8)

    part_ids = kmeans.transform(fsl)
    for idx, part_id in enumerate(part_ids):
        if idx == 1:
            assert part_ids[1] == -1
        else:
            assert part_ids[idx] != -1


def test_kmeans_transform_sliced_fixed_size_list():
    data = np.arange(80, dtype=np.float32).reshape(10, 8)
    fsl = pa.FixedSizeListArray.from_arrays(pa.array(data.reshape(-1)), 8)
    kmeans = KMeans(3, centroids=torch.from_numpy(data[[0, 4, 8]]), device="cpu")

    expected = kmeans.transform(torch.from_numpy(data[4:7]))
    assert len(expected) == 3
    assert torch.equal(kmeans.transform(fsl.slice(4, 3)), expected)


def test_kmeans_random_init_is_seeded():
    data = torch.arange(64 * 4, dtype=torch.float32).reshape(64, 4)

    def init(seed):
        kmeans = KMeans(16, seed=seed, device="cpu")
        kmeans._random_init(data)
        return kmeans.centroids

    first, second = init(7), init(7)
    assert torch.equal(first, second)
    assert not torch.equal(first, init(8))


@pytest.mark.gpu
def test_create_ivf_pq_float64_on_accelerator(tmp_path: Path):
    values = pa.array(np.random.default_rng(0).standard_normal(512 * 16))
    arr = pa.FixedSizeListArray.from_arrays(values, 16)
    ds = lance.write_dataset(pa.Table.from_arrays([arr], ["vector"]), tmp_path)

    ds = ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=4,
        accelerator=preferred_device().type,
    )

    validate_vector_index(ds, "vector", sample_size=16)
