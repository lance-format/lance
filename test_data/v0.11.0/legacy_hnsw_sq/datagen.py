# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate the legacy IVF_HNSW_SQ fixture with a released writer."""

import shutil
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

# To generate the test file, we should be running this version of lance.
assert lance.__version__ == "0.11.0"

path = Path(__file__).parent / "legacy_hnsw_sq"
shutil.rmtree(path, ignore_errors=True)

dim = 8
base = np.random.default_rng(20261120).random((256, dim), dtype=np.float32)
vector = pa.FixedSizeListArray.from_arrays(
    pa.array(base.reshape(-1), type=pa.float32()), dim
)
table = pa.table({"id": pa.array(range(256), type=pa.int64()), "vector": vector})
dataset = lance.write_dataset(table, path)
dataset.create_index("vector", index_type="IVF_HNSW_SQ", metric="l2", num_partitions=2)

# Leave one fragment unindexed so optimize has a delta to process.
appended = np.random.default_rng(20261121).random((64, dim), dtype=np.float32)
vector = pa.FixedSizeListArray.from_arrays(
    pa.array(appended.reshape(-1), type=pa.float32()), dim
)
table = pa.table({"id": pa.array(range(256, 320), type=pa.int64()), "vector": vector})
lance.write_dataset(table, path, mode="append")
