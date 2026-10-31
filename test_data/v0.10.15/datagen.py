# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil

import lance
import numpy as np
import pyarrow as pa

# To generate the test file, we should be running this version of lance.
assert lance.__version__ == "0.10.15"

name = "non_divisible_pq"
dimension = 64
num_sub_vectors = 14
sub_vector_dimension = dimension // num_sub_vectors

shutil.rmtree(name, ignore_errors=True)

vector = np.arange(1, dimension + 1, dtype=np.float32)
data = pa.table(
    {
        "id": pa.array([0]),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vector), dimension),
    }
)
dataset = lance.write_dataset(data, name)

ivf_centroids = np.zeros((1, dimension), dtype=np.float32)
persisted_prefix = vector[: num_sub_vectors * sub_vector_dimension].reshape(
    num_sub_vectors, sub_vector_dimension
)
pq_codebook = np.repeat(persisted_prefix[:, np.newaxis, :], 256, axis=1)
dataset.create_index(
    "vector",
    "IVF_PQ",
    metric="l2",
    num_partitions=1,
    ivf_centroids=ivf_centroids,
    num_sub_vectors=num_sub_vectors,
    pq_codebook=pq_codebook,
)
