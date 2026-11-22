# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import shutil

import lance
import numpy as np
import pyarrow as pa

# To generate the test file, we should be running this version of lance.
# Before 0.25.0, `create_index("IVF_PQ")` wrote the legacy (v1) index file.
assert lance.__version__ == "0.24.1"

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
