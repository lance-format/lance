# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import glob
import shutil
import struct

import lance
import numpy as np
import pyarrow as pa

# To generate the test file, we should be running this version of lance.
# 0.16.x was the last release whose IVF_HNSW_PQ builder wrote the legacy
# index file, a v1 Lance file with format version (0, 2). Current releases
# refuse these files and ask for a rebuild.
assert lance.__version__ == "0.16.1"

name = "legacy_ivf_hnsw_pq"
num_rows = 256
dimension = 8

shutil.rmtree(name, ignore_errors=True)

vectors = np.random.default_rng(42).normal(size=(num_rows, dimension))
vectors = vectors.astype(np.float32)
# Unit vectors make Dot rank neighbors the same as L2.
vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
data = pa.table(
    {
        "i": pa.array(np.arange(num_rows, dtype=np.int32)),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), dimension
        ),
    }
)
dataset = lance.write_dataset(data, name, max_rows_per_file=num_rows // 2)
for index_name, metric in [("l2_idx", "l2"), ("dot_idx", "dot")]:
    dataset.create_index(
        "vector",
        index_type="IVF_HNSW_PQ",
        name=index_name,
        metric=metric,
        num_partitions=2,
        num_sub_vectors=2,
    )

dataset = lance.dataset(name)
assert len(dataset.get_fragments()) == 2
assert sorted(index["name"] for index in dataset.list_indices()) == [
    "dot_idx",
    "l2_idx",
]
for path in glob.glob(f"{name}/_indices/*/*.idx"):
    with open(path, "rb") as f:
        contents = f.read()
    assert struct.unpack("<HH", contents[-8:-4]) == (0, 2), path
    if path.endswith("auxiliary.idx"):
        # The PQ storage metadata names the bit width `num_bits` and has no
        # `transposed` key.
        assert b'"num_bits":' in contents, path
        assert b"transposed" not in contents, path
