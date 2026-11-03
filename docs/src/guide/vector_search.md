---
title: Vector Search
description: Choose vector indexes, run nearest-neighbor queries, and tune recall, latency, filtering, and index maintenance.
---

# Vector Search

Lance supports exact nearest-neighbor search without an index and approximate nearest-neighbor (ANN) search with vector indexes. Vector columns should use an Arrow fixed-size list or fixed-shape tensor type with a consistent dimension.

This guide covers Python's `LanceDataset` API. For a runnable end-to-end example, see the [Vector Search Quickstart](../quickstart/vector-search.md).

## Search without an Index

Pass `nearest` to `to_table` or `scanner` to retrieve the `k` nearest rows. Lance adds a `_distance` column to the result:

```python
result = dataset.to_table(
    columns=["id", "category"],
    nearest={
        "column": "vector",
        "q": query_vector,
        "k": 10,
    },
)
```

Without a usable vector index, Lance computes distances against the vectors in the search scope. This is exact and is useful as a correctness baseline. To force exact search even when an index exists, set `nearest["use_index"] = False`.

Supported distance metrics include:

- `l2` (also `euclidean`): squared Euclidean distance; smaller is closer.
- `cosine`: cosine distance; smaller is closer.
- `dot`: dot-product distance; smaller is closer.

Use the same metric for index creation and querying. Dot-product search may require vector normalization when the intended ranking is cosine-like.

## Choose an Index Type

All `IVF_*` indexes first partition vectors with an inverted-file (IVF) coarse quantizer. At query time, Lance searches selected partitions and then ranks candidates using the index's storage format.

| Index family | Candidate storage | Strengths | Tradeoffs |
|---|---|---|---|
| `IVF_FLAT` | Original vectors | Exact distances within searched partitions; no vector quantizer training | Larger index and more candidate I/O |
| `IVF_PQ` | Product-quantized vectors | Small index and low candidate I/O | Lower recall without enough probes/refinement; requires `num_sub_vectors` |
| `IVF_SQ` | Scalar-quantized vectors | Simple compression with more information per dimension than typical PQ | More storage than PQ |
| `IVF_RQ` | RaBitQ-quantized vectors | Compact quantization with multi-bit distance estimates | Tune `num_bits`; higher bits improve estimates but increase build work and storage |
| `IVF_HNSW_FLAT` | Original vectors plus HNSW graph | Graph navigation within IVF partitions; exact candidate distances | Larger graph and storage overhead |
| `IVF_HNSW_SQ` | Scalar-quantized vectors plus HNSW graph | Graph search with compressed candidate storage | More build parameters and graph overhead |
| `IVF_HNSW_PQ` | Product-quantized vectors plus HNSW graph | Graph search with small candidate storage | Quantization and graph tuning both affect recall |

**Start with `IVF_RQ` for most vector-search workloads.** It provides compact storage and multi-bit distance estimates without requiring PQ sub-vector configuration. Use `IVF_PQ` when its smaller index footprint or GPU-accelerated training is important, and `IVF_HNSW_*` when graph-based search fits your latency and recall requirements. Validate the choice against measured recall and latency on representative data rather than selecting by index size alone.

See [Performance Guide](performance.md#vector-index) for index-size formulas and training memory estimates.

## Create a Vector Index

Create an `IVF_RQ` index with the default 5-bit RaBitQ configuration:

```python
dataset.create_index(
    "vector",
    index_type="IVF_RQ",
    metric="l2",
    num_partitions=256,
)
```

The default `num_bits=5` is a good starting point for recall. Use `num_bits=1` when minimizing index size and build I/O is more important, then measure recall because searches cannot use the multi-bit distance estimate.

Create an `IVF_PQ` or `IVF_HNSW_SQ` index when those alternatives better fit your workload:

```python
dataset.create_index(
    "vector",
    index_type="IVF_PQ",
    metric="l2",
    num_partitions=256,
    num_sub_vectors=16,
)

dataset.create_index(
    "vector",
    index_type="IVF_HNSW_SQ",
    metric="l2",
    num_partitions=256,
    m=20,
    ef_construction=100,
)
```

Important creation parameters:

- `num_partitions`: number of IVF partitions. More partitions reduce work per probe but require more probes for comparable recall.
- `num_sub_vectors`: number of PQ sub-vectors. Required for PQ variants.
- `num_bits`: quantization bit width where supported. RQ defaults to 5 bits; PQ supports 4 or 8 bits.
- `ivf_centroids`, `pq_codebook`: optional pre-trained model artifacts.
- `m`, `max_level`, `ef_construction`: HNSW graph connectivity and build-time search parameters.
- `accelerator="cuda"` or `"mps"`: optional GPU acceleration for IVF_PQ training when the corresponding support is installed.

The default index name is `<column>_idx`. Vector index creation uses `replace=False` by default, so pass `replace=True` explicitly to replace an existing index with the same name.

## Tune Search Parameters

Use `nprobes` and `refine_factor` to trade recall for query work:

```python
result = dataset.to_table(
    nearest={
        "column": "vector",
        "q": query_vector,
        "k": 10,
        "nprobes": 20,
        "refine_factor": 5,
    }
)
```

- `nprobes`: number of IVF partitions to search. Higher values usually improve recall and increase latency.
- `refine_factor`: retrieve up to `k * refine_factor` candidates, re-rank them using more accurate distances, and return the top `k`.
- `minimum_nprobes`, `maximum_nprobes`: bounds for adaptive probing.
- `ef`: HNSW search breadth for HNSW-based indexes.

Measure recall against exact search:

```python
exact = dataset.to_table(
    nearest={"column": "vector", "q": query_vector, "k": 10, "use_index": False}
)
approx = dataset.to_table(
    nearest={"column": "vector", "q": query_vector, "k": 10, "nprobes": 20}
)
recall = len(set(exact["id"].to_pylist()) & set(approx["id"].to_pylist())) / 10
```

Use a representative set of query vectors and report average recall, not just a single query. A recall target alone is not sufficient: compare latency under the same data, filter, cache, and storage conditions.

## Filter Vector Search

The `prefilter` scanner option controls when SQL filtering happens relative to top-k selection:

```python
# Filter candidates before selecting top-k
result = dataset.scanner(
    nearest={"column": "vector", "q": query_vector, "k": 10},
    filter="category = 'science'",
    prefilter=True,
).to_table()
```

- `prefilter=False` (default): select top-k first and then apply the SQL filter. Rejected rows are not replaced, so fewer than `k` rows may be returned.
- `prefilter=True`: restrict candidates to rows matching the SQL filter before selecting top-k. This is usually the desired behavior for metadata-constrained retrieval, although it can be more expensive and does not make an approximate index exact.

A [scalar index](scalar_indexing.md) on the metadata column can accelerate prefiltering. Deletions and explicit fragment scopes restrict candidates regardless of the prefilter setting.

## Combine Full-Text and Vector Search

Use `filter` with `search_filter` and `expr_filter` to combine full-text and vector search in one scanner:

```python
from lance.query import MatchQuery

result = dataset.scanner(
    nearest={"column": "vector", "q": query_vector, "k": 10},
    filter={
        "search_filter": MatchQuery("machine learning", "text"),
        "expr_filter": "category = 'science'",
    },
).to_table()
```

Build the required inverted index on the text column and vector index on the embedding column. See [Full-Text Search](full_text_search.md) for text-index configuration and query types.

## Index Maintenance

Appending rows does not automatically add them to existing vector indexes. By default, Lance combines index results with exact search over unindexed data, so new rows remain visible while unindexed search cost grows.

Incrementally add new data to an index:

```python
dataset.optimize.optimize_indices(index_names=["vector_idx"])
```

By default, this assigns new vectors to the existing IVF partitions and quantizer models without retraining. If the new data has a substantially different distribution, retrain the index:

```python
dataset.optimize.optimize_indices(index_names=["vector_idx"], retrain=True)
```

Retraining is more expensive but can restore recall when existing centroids or codebooks no longer represent the data well.

Use `fast_search=True` only when it is acceptable to skip recently appended unindexed data. Inspect index coverage before choosing it:

```python
stats = dataset.stats.index_stats("vector_idx")
print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
```

For distributed index builds, see [Distributed Indexing](distributed_indexing.md). For general index inspection, replacement, and compaction, see [Indexing and Search](indexing.md#index-lifecycle).
