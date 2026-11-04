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

## Next Steps

Continue with the **[Vector Search User Guide](../guide/vector_search.md)** for index selection, filtering, recall tuning, and index maintenance. You can also explore **[Full-text Search](../quickstart/full-text-search.md)** to create and query a BM25 index for keyword-based search in Lance.
