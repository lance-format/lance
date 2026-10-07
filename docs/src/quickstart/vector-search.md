---
title: Vector Search
description: High-performance vector search with ANN indices, including IVF_PQ, IVF_HNSW_PQ, and IVF_HNSW_SQ
---

# Vector Indexing and Vector Search With Lance

Lance provides high-performance vector search capabilities with ANN (Approximate Nearest Neighbor) indices.

By the end of this tutorial, you'll be able to build and use ANN indices to dramatically speed up vector search operations while maintaining high accuracy. You'll also learn how to tune search parameters for optimal performance and combine vector search with metadata queries in a single operation.

## Install the Python SDK

```bash
pip install pylance
```

## Set Up Your Environment

First, import the necessary libraries:

```python
import shutil
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import duckdb
```

## Prepare Your Vector Embeddings

For this tutorial, download and prepare the SIFT 1M dataset for vector search experiments.

- Download `ANN_SIFT1M` from: http://corpus-texmex.irisa.fr/
- Direct link: `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`

You can just use `wget`:

```bash
rm -rf sift* vec_data.lance
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

## Convert Your Data to Lance Format

Then, convert the raw vector data into Lance format for efficient storage and querying.

```python
from lance.vector import vec_to_table
import struct

uri = "vec_data.lance"

with open("sift/sift_base.fvecs", mode="rb") as fobj:
    buf = fobj.read()
    data = np.array(struct.unpack("<128000000f", buf[4 : 4 + 4 * 1000000 * 128])).reshape((1000000, 128))
    dd = dict(zip(range(1000000), data))

table = vec_to_table(dd)
lance.write_dataset(table, uri, max_rows_per_group=8192, max_rows_per_file=1024*1024)
```

Now you can load the dataset:

```python
uri = "vec_data.lance"
sift1m = lance.dataset(uri)
```

## Search Without an Index

You'll perform vector search without an index to see the baseline performance, then compare it with indexed search.

First, let's sample some query vectors:

```python
import duckdb
# Make sure DuckDB v0.7+ is installed
samples = duckdb.query("SELECT vector FROM sift1m USING SAMPLE 100").to_df().vector
```

```
0     [29.0, 10.0, 1.0, 50.0, 7.0, 89.0, 95.0, 51.0,...
1     [7.0, 5.0, 39.0, 49.0, 17.0, 12.0, 83.0, 117.0...
2     [0.0, 0.0, 0.0, 10.0, 12.0, 31.0, 6.0, 0.0, 0....
3     [0.0, 2.0, 9.0, 1.793662034335766e-43, 30.0, 1...
4     [54.0, 112.0, 16.0, 0.0, 0.0, 7.0, 112.0, 44.0...
                            ...
95    [1.793662034335766e-43, 33.0, 47.0, 28.0, 0.0,...
96    [1.0, 4.0, 2.0, 32.0, 3.0, 7.0, 119.0, 116.0, ...
97    [17.0, 46.0, 12.0, 0.0, 0.0, 3.0, 23.0, 58.0, ...
98    [0.0, 11.0, 30.0, 14.0, 34.0, 7.0, 0.0, 0.0, 1...
99    [20.0, 8.0, 121.0, 98.0, 37.0, 77.0, 9.0, 18.0...
Name: vector, Length: 100, dtype: object
```

Now, perform nearest neighbor search without an index:

```python
import time

start = time.time()
tbl = sift1m.to_table(columns=["id"], nearest={"column": "vector", "q": samples[0], "k": 10})
end = time.time()

print(f"Time(sec): {end-start}")
print(tbl.to_pandas())
```

Expected output:
```
Time(sec): 0.10735273361206055
       id                                             vector    score
0  144678  [29.0, 10.0, 1.0, 50.0, 7.0, 89.0, 95.0, 51.0,...      0.0
1  575538  [2.0, 0.0, 1.0, 42.0, 3.0, 38.0, 152.0, 27.0, ...  76908.0
2  241428  [11.0, 0.0, 2.0, 118.0, 11.0, 108.0, 116.0, 21...  92877.0
...
```

Without the index, the search will scan throughout the whole dataset to compute the distance between each data point. For practical real-time performance with, you will get much better performance with an ANN index.

## Build the Search Index

If you build an ANN index - you can dramatically speed up vector search operations while maintaining high accuracy. In this example, we will build the `IVF_PQ` index: 

```python
sift1m.create_index(
    "vector",
    index_type="IVF_PQ", # specify the IVF_PQ index type
    num_partitions=256,  # IVF
    num_sub_vectors=16,  # PQ
)
```

The sample response should look like this:

```
Building vector index: IVF256,PQ16
CPU times: user 2min 23s, sys: 2.77 s, total: 2min 26s
Wall time: 22.7 s
Sample 65536 out of 1000000 to train kmeans of 128 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
Sample 65536 out of 1000000 to train kmeans of 8 dim, 256 clusters
```

!!! warning "Index Creation Performance"
    If you're trying this on your own data, make sure your vector (dimensions / num_sub_vectors) % 8 == 0, or else index creation will take much longer than expected due to SIMD misalignment.

## Vector Search with the ANN Index

You can now perform the same search operation using your newly created index and see the dramatic performance improvement.

```python
sift1m = lance.dataset(uri)

import time

tot = 0
for q in samples:
    start = time.time()
    tbl = sift1m.to_table(nearest={"column": "vector", "q": q, "k": 10})
    end = time.time()
    tot += (end - start)

print(f"Avg(sec): {tot / len(samples)}")
print(tbl.to_pandas())
```

Expected output:
```
Avg(sec): 0.0009334301948547364
       id                                             vector         score
0  378825  [20.0, 8.0, 121.0, 98.0, 37.0, 77.0, 9.0, 18.0...  16560.197266
1  143787  [11.0, 24.0, 122.0, 122.0, 53.0, 4.0, 0.0, 3.0...  61714.941406
2  356895  [0.0, 14.0, 67.0, 122.0, 83.0, 23.0, 1.0, 0.0,...  64147.218750
3  535431  [9.0, 22.0, 118.0, 118.0, 4.0, 5.0, 4.0, 4.0, ...  69092.593750
4  308778  [1.0, 7.0, 48.0, 123.0, 73.0, 36.0, 8.0, 4.0, ...  69131.812500
5  222477  [14.0, 73.0, 39.0, 4.0, 16.0, 94.0, 19.0, 8.0,...  69244.195312
6  672558  [2.0, 1.0, 0.0, 11.0, 36.0, 23.0, 7.0, 10.0, 0...  70264.828125
7  365538  [54.0, 43.0, 97.0, 59.0, 34.0, 17.0, 10.0, 15....  70273.710938
8  659787  [10.0, 9.0, 23.0, 121.0, 38.0, 26.0, 38.0, 9.0...  70374.703125
9  603930  [32.0, 32.0, 122.0, 122.0, 70.0, 4.0, 15.0, 12...  70583.375000
```

!!! note "Performance Note"
    Your actual numbers will vary by your storage. These numbers are from local disk on an M2 MacBook Air. If you're querying S3 directly, HDD, or network drives, performance will be slower.

## Tune the Search Parameters

You need to adjust search parameters to balance between speed and accuracy, finding the optimal settings for your use case.

The latency vs recall is tunable via:
- **nprobes**: how many IVF partitions to search
- **refine_factor**: determines how many vectors are retrieved during re-ranking

```python
sift1m.to_table(
    nearest={
        "column": "vector",
        "q": samples[0],
        "k": 10,
        "nprobes": 10,
        "refine_factor": 5,
    }
).to_pandas()
```

**Parameter Explanation:**
- `q` => sample vector
- `k` => how many neighbors to return
- `nprobes` => how many partitions (in the coarse quantizer) to probe
- `refine_factor` => controls "re-ranking". If k=10 and refine_factor=5 then retrieve 50 nearest neighbors by ANN and re-sort using actual distances then return top 10. This improves recall without sacrificing performance too much

!!! note "Memory Usage"
    The latencies above include file I/O as Lance currently doesn't hold anything in memory. Along with index building speed, creating a purely in-memory version of the dataset would make the biggest impact on performance.

## Combine Features and Vectors

You can add metadata columns to your vector dataset and query both vectors and features together in a single operation.

In real-life situations, users have other feature or metadata columns that need to be stored and fetched together. If you're managing data and the index separately, you have to do a bunch of annoying plumbing to put stuff together. 

With Lance, you can add columns directly to the dataset using `add_columns()`. For basic use cases, you can use SQL:

```python
sift1m.add_columns(
    {
        "item_id": "id + 1000000",
        "revenue": "random() * 1000 + 5000",
    }
)
```
For more complex columns, you can provide a Python function to generate the new column data:
```python
@lance.batch_udf()
def add_columns_func(batch: pa.Table) -> pd.DataFrame:
    """Add item_id and revenue columns to a batch of data.

    Args:
        batch: PyArrow Table containing the original data

    Returns:
        Pandas DataFrame with added item_id and revenue columns
    """
    item_ids: np.ndarray = np.arange(batch.num_rows)
    revenue: np.ndarray = (np.random.randn(batch.num_rows) + 5) * 1000
    return pd.DataFrame({"item_id": item_ids, "revenue": revenue})


sift1m.add_columns(add_columns_func)
```
You can then query both vectors and metadata together:

```python
# Get vectors and metadata together
result = sift1m.to_table(
    columns=["item_id", "revenue"],
    nearest={"column": "vector", "q": samples[0], "k": 10}
)
print(result.to_pandas())
```

## Next Steps

Check out **[Full-text Search](../quickstart/full-text-search.md)**, where we show how to create and query a BM25 index for keyword-based search in Lance.
