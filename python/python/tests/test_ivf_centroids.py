# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import lance
import numpy as np
import pyarrow as pa


def test_ivf_centroids_exposed(tmp_path):
    """Verify that centroids for an IVF-based index are exposed via both"""

    # A partition is trained only when the data can give it a 256-code
    # codebook's worth of vectors, so the fixture covers every partition.
    dim, parts = 4, 8
    rows = parts * 256
    vecs = pa.array(
        np.random.randn(rows, dim).tolist(),
        type=pa.list_(pa.float32(), dim),
    )
    ds = lance.write_dataset(
        pa.Table.from_pylist([{"vector": v} for v in vecs]),
        tmp_path / "ds",
    )

    # build small IVF_PQ index so centroids exist
    ds.create_index(
        "vector", index_type="IVF_PQ", num_partitions=parts, num_sub_vectors=2
    )

    # arrow view via explicit index name
    arrow_centroids = ds.centroids(index_name="vector_idx")
    assert arrow_centroids is not None
    assert len(arrow_centroids) == parts
    assert arrow_centroids.type.list_size == dim
