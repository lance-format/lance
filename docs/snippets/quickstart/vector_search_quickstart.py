# Runnable source for docs/src/quickstart/vector-search.md.
# ruff: noqa: E402, B018, I001

# --8<-- [start:imports]
import random
import shutil

import lance
import numpy as np
import pandas as pd
import pyarrow as pa
# --8<-- [end:imports]

# --8<-- [start:create-vectors]
from lance.vector import vec_to_table

uri = "/tmp/vec_data.lance"
shutil.rmtree(uri, ignore_errors=True)

num_rows, dim, num_clusters = 1024, 16, 8
row_ids = np.arange(num_rows, dtype=np.int32)
centers = np.eye(num_clusters, dim, dtype=np.float32) * 16.0
in_cluster = (row_ids // num_clusters)[:, None]
dims = np.arange(dim, dtype=np.int32)[None, :]
offsets = (((in_cluster >> (dims % 7)) & 1) * 0.25 + in_cluster * 0.125).astype(
    np.float32
)
vectors = centers[row_ids % num_clusters] + offsets

table = vec_to_table(dict(enumerate(vectors)))
dataset = lance.write_dataset(table, uri, mode="overwrite")
dataset.count_rows()
# --8<-- [end:create-vectors]

# --8<-- [start:sample-queries]
random.seed(42)
samples = dataset.sample(4, columns=["vector"]).to_pandas()["vector"]
len(samples)
# --8<-- [end:sample-queries]

# --8<-- [start:flat-search]
dataset.to_table(
    columns=["id", "_distance"],
    nearest={"column": "vector", "q": samples.iloc[0], "k": 5},
).to_pandas()
# --8<-- [end:flat-search]

# --8<-- [start:build-index]
dataset.create_index(
    "vector",
    index_type="IVF_RQ",  # recommended for most use cases
    num_partitions=8,
    replace=True,
)
dataset = lance.dataset(uri)
[(idx.name, idx.index_type) for idx in dataset.describe_indices()]
# --8<-- [end:build-index]

# --8<-- [start:ann-search]
dataset.to_table(
    columns=["id", "_distance"],
    nearest={
        "column": "vector",
        "q": samples.iloc[0],
        "k": 5,
        "nprobes": 8,
        "refine_factor": 5,
    },
).to_pandas()
# --8<-- [end:ann-search]

# --8<-- [start:tune-search]
dataset.to_table(
    columns=["id", "_distance"],
    nearest={
        "column": "vector",
        "q": samples.iloc[0],
        "k": 5,
        "nprobes": 4,
        "refine_factor": 5,
    },
).to_pandas()
# --8<-- [end:tune-search]

# --8<-- [start:batch-search]
query_batch = np.stack(samples.to_list())
(
    dataset.to_table(
        columns=["id", "_distance"],
        nearest={
            "column": "vector",
            "q": query_batch,
            "k": 2,
            "nprobes": 4,
            "refine_factor": 5,
        },
    )
    .to_pandas()
    .sort_values(["query_index", "_distance", "id"])
    .reset_index(drop=True)
)
# --8<-- [end:batch-search]

# --8<-- [start:add-columns-sql]
dataset.add_columns(
    {
        "item_id": "id + 1000000",
        "bucket": "id % 8",
    }
)
# --8<-- [end:add-columns-sql]


# --8<-- [start:add-columns-udf]
@lance.batch_udf()
def compute_revenue(batch: pa.RecordBatch) -> pd.DataFrame:
    ids = batch["id"].to_numpy()
    return pd.DataFrame({"revenue": (ids % 10) * 100.0 + 500.0})


dataset.add_columns(compute_revenue)
# --8<-- [end:add-columns-udf]

# --8<-- [start:search-with-metadata]
dataset.to_table(
    columns=["id", "item_id", "bucket", "revenue", "_distance"],
    nearest={
        "column": "vector",
        "q": samples.iloc[0],
        "k": 5,
        "nprobes": 8,
        "refine_factor": 5,
    },
).to_pandas()
# --8<-- [end:search-with-metadata]
