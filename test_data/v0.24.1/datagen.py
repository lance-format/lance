# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import glob
import json
import os
import shutil
import struct

import lance
import numpy as np
import pyarrow as pa
from lance.file import LanceFileReader

# To generate the test file, we should be running this version of lance.
# Releases from 0.20 to 0.24 wrote the PQ storage metadata of IVF_HNSW_PQ with
# a `num_bits` key, and recorded the index details as an empty
# `lance.table.VectorIndexDetails`, which current releases fill in by reading
# that metadata.
assert lance.__version__ == "0.24.1"

name = "ivf_hnsw_pq"
num_rows = 256
dimension = 8

shutil.rmtree(name, ignore_errors=True)

vectors = np.random.default_rng(42).random((num_rows, dimension), dtype=np.float32)
data = pa.table(
    {
        "i": pa.array(np.arange(num_rows, dtype=np.int32)),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), dimension
        ),
    }
)
dataset = lance.write_dataset(data, name)
dataset.create_index(
    "vector",
    index_type="IVF_HNSW_PQ",
    metric="l2",
    num_partitions=2,
    num_sub_vectors=2,
)

dataset = lance.dataset(name)
assert [index["name"] for index in dataset.list_indices()] == ["vector_idx"]
(aux,) = glob.glob(f"{name}/_indices/*/auxiliary.idx")
with open(aux, "rb") as f:
    assert struct.unpack("<HH", f.read()[-8:-4]) == (0, 3)
reader = LanceFileReader(os.path.abspath(aux))
storage = json.loads(reader.metadata().schema.metadata[b"storage_metadata"])
pq_metadata = json.loads(storage[0])
assert "num_bits" in pq_metadata and pq_metadata["transposed"], pq_metadata
(manifest,) = glob.glob(f"{name}/_versions/2.manifest")
with open(manifest, "rb") as f:
    assert b"lance.table.VectorIndexDetails" in f.read()
