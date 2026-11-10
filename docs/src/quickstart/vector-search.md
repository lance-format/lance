---
title: Vector Search
description: High-performance vector search with the recommended IVF_RQ ANN index, plus IVF_PQ and HNSW alternatives
---

# Vector Indexing and Vector Search With Lance

Lance provides high-performance vector search capabilities with ANN (Approximate Nearest Neighbor) indices.

By the end of this tutorial, you'll be able to build and use ANN indices to dramatically speed up vector search operations while maintaining high accuracy. You'll also learn how to tune search parameters for optimal performance and combine vector search with metadata queries in a single operation.

## Install the Python SDK

```bash
--8<-- "snippets/quickstart/pylance-install.sh"
```

## Set Up Your Environment

First, import the necessary libraries:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:imports"
```

## Prepare Your Vector Embeddings

Generate a synthetic clustered vector dataset and write it to Lance format with `vec_to_table`:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:create-vectors"
```

```output
--8<-- "snippets/quickstart/outputs/vector_search_quickstart.py.create-vectors.txt"
```

## Search Without an Index

You'll perform an exact vector search without an index first, then compare it with indexed search.

First, let's sample some query vectors:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:sample-queries"
```

```output
--8<-- "snippets/quickstart/outputs/vector_search_quickstart.py.sample-queries.txt"
```

Now, perform nearest neighbor search without an index:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:flat-search"
```

```output
--8<-- "snippets/quickstart/outputs/vector_search_quickstart.py.flat-search.txt"
```

Without the index, the search will scan throughout the whole dataset to compute the distance between each data point. For practical real-time performance on large datasets, you will get much better performance with an ANN index.

## Build the Search Index

If you build an ANN index, you can dramatically speed up vector search operations while maintaining high accuracy. We recommend starting with the `IVF_RQ` index for most workloads:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:build-index"
```

```output
--8<-- "snippets/quickstart/outputs/vector_search_quickstart.py.build-index.txt"
```

The default 5-bit RaBitQ configuration provides compact storage and multi-bit distance estimates. Set `num_bits=1` to reduce index size and build I/O, then measure recall because searches cannot use the multi-bit distance estimate.

## Vector Search with the ANN Index

You can now perform the same search operation using your newly created index:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:ann-search"
```

```output
--8<-- "snippets/quickstart/outputs/vector_search_quickstart.py.ann-search.txt"
```

!!! note "Performance Note"
    On million-vector datasets, an `IVF_RQ` index typically reduces search latency from hundreds of milliseconds (full scan) to single-digit milliseconds on local SSDs. Latency varies by storage backend (local NVMe vs. object storage) and partition parameters.

## Tune the Search Parameters

You need to adjust search parameters to balance between speed and accuracy, finding the optimal settings for your use case.

The latency vs recall is tunable via:
- **nprobes**: how many IVF partitions to search
- **refine_factor**: determines how many vectors are retrieved during re-ranking

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:tune-search"
```

```output
--8<-- "snippets/quickstart/outputs/vector_search_quickstart.py.tune-search.txt"
```

**Parameter Explanation:**
- `q` => sample vector
- `k` => how many neighbors to return
- `nprobes` => how many partitions (in the coarse quantizer) to probe
- `refine_factor` => controls "re-ranking". If k=5 and refine_factor=5 then retrieve 25 nearest neighbors by ANN and re-sort using actual distances then return top 5. This improves recall without sacrificing performance too much

Pass a two-dimensional query array to refine several queries together:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:batch-search"
```

```output
--8<-- "snippets/quickstart/outputs/vector_search_quickstart.py.batch-search.txt"
```

Results include a `query_index` column identifying each query. With fixed,
positive `nprobes` and fully indexed IVF_FLAT/PQ/SQ/RQ data, Lance shares index
partition scans and reads duplicate refinement candidates once per input batch.
Each query still ranks only its own candidates. Vector reads are split into
chunks targeting 32 MiB of decoded vector data; candidate metadata and per-query
top-k results consume additional memory. Adaptive probes, HNSW, incomplete index
coverage, stale vector overlays, and external row masks use per-query execution.

!!! note "Memory Usage"
    The latencies above include file I/O as Lance currently doesn't hold anything in memory. Along with index building speed, creating a purely in-memory version of the dataset would make the biggest impact on performance.

## Combine Features and Vectors

You can add metadata columns to your vector dataset and query both vectors and features together in a single operation.

In real-life situations, users have other feature or metadata columns that need to be stored and fetched together. If you're managing data and the index separately, you have to do a bunch of annoying plumbing to put stuff together. 

With Lance, you can add columns directly to the dataset using `add_columns()`. For basic use cases, you can use SQL:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:add-columns-sql"
```

For more complex columns, you can provide a Python function to generate the new column data:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:add-columns-udf"
```

You can then query both vectors and metadata together:

```python
--8<-- "snippets/quickstart/vector_search_quickstart.py:search-with-metadata"
```

```output
--8<-- "snippets/quickstart/outputs/vector_search_quickstart.py.search-with-metadata.txt"
```

## Refine Using Another Vector Column

If a dataset stores two vector representations of each row, you can generate
candidates with one column and rerank them with the other. For example, index a
128-dimensional projection and compute final distances with the original
1024-dimensional vectors. Both columns must be `FixedSizeList<Float32>`.

Given a dataset `ds` with `short_vector` and `full_vector` columns, and a query
represented in both spaces:

```python
results = ds.to_table(
    columns=["_distance"],
    with_row_id=True,
    nearest={
        "column": "short_vector",
        "q": short_query,
        "metric": "l2",
        "k": 10,
        "nprobes": 4,
        "refine_factor": 2,
        "refine_column": "full_vector",
        "refine_q": full_query,
        "refine_metric": "cosine",
    },
)
```

Lance selects up to 20 coarse candidates, reads their `full_vector` values from
the same dataset snapshot, computes cosine distances, and returns the best 10.
The refinement column does not need an index. The final `_distance` and
`distance_range` refer to the refinement metric. Supply all three new parameters
together and set a positive `refine_factor`; omitting them preserves ordinary
same-column refinement. Batched queries must have matching query counts.

For a fully indexed coarse search, quantized candidates are selected using the
index's approximate distances. Cross-column refinement replaces the ordinary
same-column exact refinement step. If appended or updated coarse vectors must be
merged with index results, Lance's existing merge path may read and rescore coarse
vectors before selecting the final coarse TopM.
With `use_index=False`, Lance selects exact coarse TopM before refinement, which
is still different from a full scan in the final vector space.

Scalar prefilters restrict candidates before selection. Postfilters run after
the final TopK and may return fewer than `k` rows. Vector and full-text query
filters cannot be combined with cross-column refinement. Null or invalid
refinement vectors can also reduce the result count.

Produce both query representations before calling Lance. For a PCA projection,
apply the same fitted transform and preprocessing used for the stored short
vectors. Increasing the candidate budget can improve recall, but increases
random reads of the refinement column, especially on remote storage.

## Next Steps

Continue with the **[Vector Search User Guide](../guide/vector_search.md)** for index selection, filtering, recall tuning, and index maintenance. You can also explore **[Full-text Search](../quickstart/full-text-search.md)** to create and query a BM25 index for keyword-based search in Lance.
