---
title: Vector Search
description: Choose vector indexes, run nearest-neighbor queries, and tune recall, latency, filtering, and index maintenance.
---

# Vector Search

Lance supports exact nearest-neighbor search without an index and approximate nearest-neighbor (ANN) search with vector indexes. Vector columns should use an Arrow fixed-size list or fixed-shape tensor type with a consistent dimension.

This guide covers the Python (`pylance`), Rust (`lance`), and Java (`org.lance:lance-core`) SDKs. For runnable end-to-end examples, see the [Vector Search Quickstart](../quickstart/vector-search.md), [Rust Quickstart](../quickstart/rust.md), and [Java Quickstart](../quickstart/java.md).

## Search without an Index

Configure a nearest-neighbor scan to retrieve the `k` nearest rows. Include `_distance` in projected columns when specifying an explicit column list:

=== "Python"

    ```python
    result = dataset.to_table(
        columns=["id", "category", "_distance"],
        nearest={
            "column": "vector",
            "q": query_vector,
            "k": 10,
        },
    )
    ```

=== "Rust"

    ```rust
    use arrow_array::{Float32Array, RecordBatch};
    use futures::TryStreamExt;

    let query = Float32Array::from(query_vector);
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .project(&["id", "category", "_distance"])?
        .nearest("vector", &query, 10)?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    import org.apache.arrow.vector.ipc.ArrowReader;
    import org.lance.ipc.LanceScanner;
    import org.lance.ipc.Query;
    import org.lance.ipc.ScanOptions;
    import java.util.Arrays;

    ScanOptions options =
        new ScanOptions.Builder()
            .columns(Arrays.asList("id", "category", "_distance"))
            .nearest(
                new Query.Builder()
                    .setColumn("vector")
                    .setKey(queryVector)
                    .setK(10)
                    .build())
            .build();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

Without a usable vector index, Lance computes distances against the vectors in the search scope. This is exact and is useful as a correctness baseline. To force exact search even when an index exists, set `nearest["use_index"] = False` (Python), `.use_index(false)` (Rust), or `.setUseIndex(false)` (Java).

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

=== "Python"

    ```python
    dataset.create_index(
        "vector",
        index_type="IVF_RQ",
        metric="l2",
        num_partitions=256,
    )
    ```

=== "Rust"

    ```rust
    use lance::index::vector::VectorIndexParams;
    use lance::index::DatasetIndexExt;
    use lance_index::IndexType;
    use lance_linalg::distance::MetricType;

    let params = VectorIndexParams::ivf_rq(256, 5, MetricType::L2);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, true)
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.index.DistanceType;
    import org.lance.index.IndexOptions;
    import org.lance.index.IndexParams;
    import org.lance.index.IndexType;
    import org.lance.index.vector.VectorIndexParams;
    import java.util.Collections;

    VectorIndexParams vecParams =
        VectorIndexParams.ivfRq(256, (byte) 5, DistanceType.L2);
    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("vector"),
                IndexType.IVF_RQ,
                IndexParams.builder().setVectorIndexParams(vecParams).build())
            .replace(true)
            .build());
    ```

The default `num_bits=5` is a good starting point for recall. Use `num_bits=1` when minimizing index size and build I/O is more important, then measure recall because searches cannot use the multi-bit distance estimate.

Create an `IVF_PQ` or `IVF_HNSW_SQ` index when those alternatives better fit your workload:

=== "Python"

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

=== "Rust"

    ```rust
    use lance_index::vector::{
        hnsw::builder::HnswBuildParams, ivf::IvfBuildParams, sq::builder::SQBuildParams,
    };

    let pq_params = VectorIndexParams::ivf_pq(256, 8, 16, MetricType::L2, 50);
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &pq_params, true)
        .await?;

    let hnsw_params = VectorIndexParams::with_ivf_hnsw_sq_params(
        MetricType::L2,
        IvfBuildParams::new(256),
        HnswBuildParams::default().num_edges(20).ef_construction(100),
        SQBuildParams::default(),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &hnsw_params, true)
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.index.vector.HnswBuildParams;
    import org.lance.index.vector.IvfBuildParams;
    import org.lance.index.vector.SQBuildParams;

    VectorIndexParams pqParams =
        VectorIndexParams.ivfPq(256, 8, 16, DistanceType.L2, 50);
    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("vector"),
                IndexType.IVF_PQ,
                IndexParams.builder().setVectorIndexParams(pqParams).build())
            .replace(true)
            .build());

    VectorIndexParams hnswParams =
        VectorIndexParams.withIvfHnswSqParams(
            DistanceType.L2,
            new IvfBuildParams.Builder().setNumPartitions(256).build(),
            new HnswBuildParams.Builder().setM(20).setEfConstruction(100).build(),
            new SQBuildParams.Builder().build());
    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("vector"),
                IndexType.IVF_HNSW_SQ,
                IndexParams.builder().setVectorIndexParams(hnswParams).build())
            .replace(true)
            .build());
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

=== "Python"

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

=== "Rust"

    ```rust
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .nearest("vector", &query, 10)?
        .nprobes(20)
        .refine(5)
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    ScanOptions options =
        new ScanOptions.Builder()
            .nearest(
                new Query.Builder()
                    .setColumn("vector")
                    .setKey(queryVector)
                    .setK(10)
                    .setNprobes(20)
                    .setRefineFactor(5)
                    .build())
            .build();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
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

=== "Python"

    ```python
    # Filter candidates before selecting top-k
    result = dataset.scanner(
        nearest={"column": "vector", "q": query_vector, "k": 10},
        filter="category = 'science'",
        prefilter=True,
    ).to_table()
    ```

=== "Rust"

    ```rust
    // Filter candidates before selecting top-k
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .filter("category = 'science'")?
        .prefilter(true)
        .nearest("vector", &query, 10)?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    // Filter candidates before selecting top-k
    ScanOptions options =
        new ScanOptions.Builder()
            .filter("category = 'science'")
            .prefilter(true)
            .nearest(
                new Query.Builder()
                    .setColumn("vector")
                    .setKey(queryVector)
                    .setK(10)
                    .build())
            .build();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
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

=== "Python"

    ```python
    dataset.optimize.optimize_indices(index_names=["vector_idx"])
    ```

=== "Rust"

    ```rust
    use lance_index::optimize::OptimizeOptions;

    dataset
        .optimize_indices(
            &OptimizeOptions::new().index_names(vec!["vector_idx".to_string()]),
        )
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.index.OptimizeOptions;
    import java.util.Collections;

    dataset.optimizeIndices(
        OptimizeOptions.builder()
            .indexNames(Collections.singletonList("vector_idx"))
            .build());
    ```

By default, this assigns new vectors to the existing IVF partitions and quantizer models without retraining. If the new data has a substantially different distribution, retrain the index:

=== "Python"

    ```python
    dataset.optimize.optimize_indices(index_names=["vector_idx"], retrain=True)
    ```

=== "Rust"

    ```rust
    dataset
        .optimize_indices(
            &OptimizeOptions::retrain().index_names(vec!["vector_idx".to_string()]),
        )
        .await?;
    ```

=== "Java"

    ```java
    dataset.optimizeIndices(
        OptimizeOptions.builder()
            .indexNames(Collections.singletonList("vector_idx"))
            .retrain(true)
            .build());
    ```

Retraining is more expensive but can restore recall when existing centroids or codebooks no longer represent the data well.

Use `fast_search=True` only when it is acceptable to skip recently appended unindexed data. Inspect index coverage before choosing it:

```python
stats = dataset.stats.index_stats("vector_idx")
print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
```

For distributed index builds, see [Distributed Indexing](distributed_indexing.md). For general index inspection, replacement, and compaction, see [Indexing and Search](indexing.md#index-lifecycle).
