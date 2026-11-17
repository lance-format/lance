# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import numpy as np
import pyarrow as pa
import pytest


def test_invalid_inputs():
    kmeans = lance.util.KMeans(32)
    data = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.random.randn(1000, 128, 8).astype(np.float32)
    )
    with pytest.raises(ValueError, match="must be a 1-D array"):
        kmeans.fit(data)

    data = pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.random.randn(1000, 128).astype(np.float64)
    )
    with pytest.raises(ValueError, match="Array must be float32 type, got: double"):
        kmeans.fit(data)


def test_kmeans_dot():
    kmeans = lance.util.KMeans(32, metric_type="dot")
    data = np.random.randn(1000, 128).astype(np.float32)
    kmeans.fit(data)
    norms = np.linalg.norm(kmeans.centroids.to_numpy_ndarray(), axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_kmeans_cosine():
    # Cosine trains and predicts over unit-length vectors: fit normalizes the
    # training rows and predict normalizes the queries. This also used to panic
    # rather than train at this size.
    kmeans = lance.util.KMeans(8, metric_type="cosine")
    data = np.random.randn(1000, 128).astype(np.float32)
    kmeans.fit(data)

    centroids = np.stack(kmeans.centroids.to_numpy(zero_copy_only=False))
    assert centroids.shape == (8, 128)
    # A cosine centroid is the mean of unit-length training rows, so its norm
    # cannot exceed 1.
    assert np.all(np.linalg.norm(centroids, axis=1) <= 1.0)

    clusters = kmeans.predict(data).to_numpy(zero_copy_only=False)
    assert clusters.shape == (1000,)
    assert clusters.max() < 8


def test_kmeans_cosine_predict_normalizes_queries():
    # Centroids are model state, not input, so predict normalizes only the
    # query. These two centroids have the norms a cosine fit can leave behind:
    # a loose cluster averages to a short mean, a tight one stays near unit
    # length. The query sits 0.41 away from the decision boundary in squared
    # L2, so the expected labels do not depend on rounding.
    centroids = np.zeros((2, 4), dtype=np.float32)
    centroids[0, 0] = 0.3
    centroids[1, 1] = 1.0
    values = pa.FixedSizeListArray.from_arrays(
        pa.array(centroids.reshape(-1), type=pa.float32()), list_size=4
    )
    kmeans = lance.util.KMeans(2, metric_type="cosine", centroids=values)
    np.testing.assert_array_equal(
        np.stack(kmeans.centroids.to_numpy(zero_copy_only=False)), centroids
    )

    # The direction 0.86/0.51 is nearest to the short centroid once the query
    # is normalized; an unnormalized query is nearest to the long one. The raw
    # query is deliberately not unit length, which is what the normalization
    # under test has to fix.
    query = np.array([[0.8613, 0.5084, 0.0, 0.0]], dtype=np.float32) * np.float32(3.7)
    expected = np.array([0], dtype=np.uint32)
    np.testing.assert_array_equal(
        kmeans.predict(query).to_numpy(zero_copy_only=False), expected
    )
    np.testing.assert_array_equal(
        kmeans.predict(query * np.float32(7.0)).to_numpy(zero_copy_only=False),
        expected,
    )


@pytest.mark.parametrize("metric_type", ["l2", "cosine"])
def test_precomputed_kmeans(metric_type):
    data = np.random.randn(1000, 128).astype(np.float32)
    kmeans = lance.util.KMeans(8, metric_type=metric_type)
    kmeans.fit(data)
    original_clusters = kmeans.predict(data)

    values = np.stack(kmeans.centroids.to_numpy(zero_copy_only=False)).flatten()
    centroids = pa.FixedSizeListArray.from_arrays(values, list_size=128)

    # Initialize a new KMeans with precomputed centroids.
    new_kmeans = lance.util.KMeans(8, metric_type=metric_type, centroids=centroids)
    new_clusters = new_kmeans.predict(data)

    # Verify the predictions are the same for both KMeans instances.
    assert original_clusters == new_clusters
