# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Run with each released writer to cover the historical PQ metadata layouts."""

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa

assert lance.__version__ in {"0.10.18", "0.16.1", "0.18.2", "0.24.1"}

path = Path(__file__).parent / lance.__version__
vectors = np.random.default_rng(0).standard_normal((512, 8)).astype(np.float32)
table = pa.table(
    {
        "id": pa.array(range(512), type=pa.int32()),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), 8),
    }
)
dataset = lance.write_dataset(table, path, max_rows_per_file=256)
dataset.create_index(
    "vector",
    "IVF_HNSW_PQ",
    num_partitions=2,
    num_sub_vectors=2,
    metric="l2",
)
assert len(dataset.get_fragments()) == 2
