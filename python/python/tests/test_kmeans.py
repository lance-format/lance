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


def test_precomputed_kmeans():
    data = np.random.randn(1000, 128).astype(np.float32)
    kmeans = lance.util.KMeans(8, metric_type="l2")
    kmeans.fit(data)
    original_clusters = kmeans.predict(data)

    values = np.stack(kmeans.centroids.to_numpy(zero_copy_only=False)).flatten()
    centroids = pa.FixedSizeListArray.from_arrays(values, list_size=128)

    # Initialize a new KMeans with precomputed centroids.
    new_kmeans = lance.util.KMeans(8, metric_type="l2", centroids=centroids)
    new_clusters = new_kmeans.predict(data)

    # Verify the predictions are the same for both KMeans instances.
    assert original_clusters == new_clusters
