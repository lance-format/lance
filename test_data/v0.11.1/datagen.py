# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import glob
import shutil
import struct

import lance
import numpy as np
import pyarrow as pa

# To generate the test file, we should be running this version of lance.
# Releases before 0.12 wrote the PQ storage metadata of IVF_HNSW_PQ without a
# `codebook_tensor` key, besides naming the bit width `num_bits`.
assert lance.__version__ == "0.11.1"

name = "legacy_ivf_hnsw_pq"
num_rows = 256
dimension = 8

shutil.rmtree(name, ignore_errors=True)

vectors = np.random.default_rng(42).normal(size=(num_rows, dimension))
vectors = vectors.astype(np.float32)
data = pa.table(
    {
        "i": pa.array(np.arange(num_rows, dtype=np.int32)),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), dimension
        ),
    }
)
dataset = lance.write_dataset(data, name, max_rows_per_file=num_rows // 2)
# This release cannot build an IVF_HNSW_PQ index with the Dot metric.
dataset.create_index(
    "vector",
    index_type="IVF_HNSW_PQ",
    name="l2_idx",
    metric="l2",
    num_partitions=2,
    num_sub_vectors=2,
)

dataset = lance.dataset(name)
assert len(dataset.get_fragments()) == 2
assert [index["name"] for index in dataset.list_indices()] == ["l2_idx"]
for path in glob.glob(f"{name}/_indices/*/*.idx"):
    with open(path, "rb") as f:
        contents = f.read()
    assert struct.unpack("<HH", contents[-8:-4]) == (0, 2), path
    if path.endswith("auxiliary.idx"):
        assert b'"num_bits":' in contents, path
        assert b"codebook_tensor" not in contents, path
