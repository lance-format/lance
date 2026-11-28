# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import glob
import json
import os
import shutil
import struct
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.file import LanceFileReader

# To generate the test file, we should be running this version of lance.
# Releases from 0.20 to 0.24 wrote the PQ storage metadata of IVF_HNSW_PQ with
# a `num_bits` key, and recorded the index details as an empty
# `lance.table.VectorIndexDetails`, which current releases fill in by reading
# that metadata. Before 0.25.0, `create_index("IVF_PQ")` wrote the legacy (v1)
# index file.
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
dataset = lance.write_dataset(data, name, max_rows_per_file=num_rows // 2)
dataset.create_index(
    "vector",
    index_type="IVF_HNSW_PQ",
    metric="l2",
    num_partitions=2,
    num_sub_vectors=2,
)

dataset = lance.dataset(name)
assert len(dataset.get_fragments()) == 2
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

name = "legacy_ivf_pq_4bit"
num_rows = 512
dimension = 16

shutil.rmtree(name, ignore_errors=True)

vectors = np.random.default_rng(42).random((num_rows, dimension), dtype=np.float32)
data = pa.table(
    {
        "id": pa.array(np.arange(num_rows, dtype=np.int64)),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), dimension
        ),
    }
)
dataset = lance.write_dataset(data, name, max_rows_per_file=num_rows // 4)
dataset.create_index(
    "vector",
    index_type="IVF_PQ",
    metric="l2",
    num_partitions=2,
    num_sub_vectors=4,
    num_bits=4,
)

stats = dataset.stats.index_stats("vector_idx")
assert len(dataset.get_fragments()) == 4
assert stats["indices"][0]["sub_index"]["nbits"] == 4

# An append-only optimize adds a delta segment, written at the packed width,
# next to the 0.19.2 segment, which it does not read.
name = "legacy_ivf_pq_4bit_unpacked_appended"
source = Path(__file__).parent.parent / "v0.19.2" / "legacy_ivf_pq_4bit_unpacked"
num_appended = 64

shutil.rmtree(name, ignore_errors=True)
shutil.copytree(source, name)

vectors = np.random.default_rng(7).random((num_appended, dimension), dtype=np.float32)
data = pa.table(
    {
        "id": pa.array(np.arange(num_rows, num_rows + num_appended, dtype=np.int64)),
        "vector": pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), dimension
        ),
    }
)
dataset = lance.write_dataset(data, name, mode="append")
dataset.optimize.optimize_indices(num_indices_to_merge=0)

dataset = lance.dataset(name)
assert len(dataset.get_fragments()) == 5
assert [index["fragment_ids"] for index in dataset.list_indices()] == [
    {0, 1, 2, 3},
    {4},
]
