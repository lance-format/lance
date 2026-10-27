# Vamana Vector Index

The `IVF_VAMANA_FLAT`, `IVF_VAMANA_SQ` and `IVF_VAMANA_PQ` vector indices split a vector column into IVF partitions, like Lance's other IVF indices, and keep a Vamana graph in each partition. Vamana is the single-layer proximity graph of DiskANN: every vector is a vertex with at most `max_degree` out-edges, and a query walks the graph greedily from an entry point towards its nearest neighbors. The three types differ in the codes the walk measures:

| Index type | Codes the walk measures | Re-scored by the index |
|---|---|---|
| `IVF_VAMANA_FLAT` | the full vectors | no, the distances are already exact |
| `IVF_VAMANA_SQ` | 8-bit scalar codes, one byte per dimension | yes |
| `IVF_VAMANA_PQ` | product codes, one byte per sub-vector | yes |

Over SQ and PQ codes the index re-scores its best candidates itself: it keeps the `2 * k` nearest by code distance, reads their vectors from the dataset and returns the `k` nearest by exact distance. A query therefore returns exact distances, as it would with `refine_factor=2`, without setting a refine factor.

A query can also end each walk early, as soon as its nearest candidates stand clear of the rest, with `stop_margin` (see [Stop the Walk Early](#stop-the-walk-early)).

## When to Choose It

Start with `IVF_VAMANA_SQ`, the type the numbers below were measured on. On four datasets of about a million vectors each (SIFT, GloVe-200, Cohere and GIST), with every index tuned to the same recall per dataset (from 0.85 to 0.99) and the index and the data in the page cache, it compared with Lance's `IVF_HNSW_SQ` and `IVF_RQ` as follows:

- Latency, one query at a time: 1.43-1.82x faster at top-10 than the faster of the two, and 1.05-1.76x faster at top-100.
- Throughput, 24 queries at once on 12 logical cores in short bursts: 1.8-2.2x as many top-10 queries a second as the faster of the two, and 1.2-2.8x as many top-100 queries.
- Index size: 9-30% smaller than `IVF_HNSW_SQ`, which stores the same SQ codes next to a larger graph.
- Build time: 1.15-1.63x that of `IVF_HNSW_SQ`, at equal or lower peak memory.

These numbers come from one machine; measure on your own data and queries before relying on them.

Choose another index type when you need:

- the `dot` or `hamming` distance, RaBitQ codes, SQ codes of other than 8 bits, or a multivector column;
- the fastest build, or no graph to hold in memory: `IVF_PQ` and `IVF_RQ` keep none;
- the legacy index file version, or indices maintained by MemWAL.

Of the three types, `IVF_VAMANA_FLAT` keeps the full vectors in the index and needs no re-score, and `IVF_VAMANA_PQ` keeps the smallest codes.

## Create an Index

=== "Python"

    ```python
    import lance

    ds = lance.dataset("/tmp/vectors.lance")
    ds.create_index(
        "vector",
        index_type="IVF_VAMANA_SQ",
        metric="cosine",
        max_degree=64,
        search_list_size=100,
        alpha=1.2,
    )
    ```

=== "Rust"

    ```rust
    use lance::index::{DatasetIndexExt, vector::VectorIndexParams};
    use lance_index::IndexType;
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::sq::builder::SQBuildParams;
    use lance_index::vector::vamana::VamanaBuildParams;
    use lance_linalg::distance::DistanceType;

    let vamana = VamanaBuildParams::default()
        .with_max_degree(64)
        .with_search_list_size(100)
        .with_alpha(1.2);
    let params = VectorIndexParams::with_ivf_vamana_sq_params(
        DistanceType::Cosine,
        IvfBuildParams::default(),
        vamana,
        SQBuildParams::default(),
    );
    dataset
        .create_index(&["vector"], IndexType::Vector, None, &params, false)
        .await?;
    ```

=== "Java"

    ```java
    VamanaBuildParams vamana =
        new VamanaBuildParams.Builder()
            .setMaxDegree(64)
            .setSearchListSize(100)
            .setAlpha(1.2f)
            .build();
    IvfBuildParams ivf = new IvfBuildParams.Builder().setNumPartitions(1).build();
    VectorIndexParams params =
        VectorIndexParams.withIvfVamanaSqParams(
            DistanceType.Cosine, ivf, vamana, new SQBuildParams.Builder().build());
    dataset.createIndex(
        IndexOptions.builder(
                List.of("vector"),
                IndexType.IVF_VAMANA_SQ,
                IndexParams.builder().setVectorIndexParams(params).build())
            .build());
    ```

For `IVF_VAMANA_FLAT`, pass `index_type="IVF_VAMANA_FLAT"` in Python, or use `VectorIndexParams::ivf_vamana` in Rust and `VectorIndexParams.ivfVamana` in Java. For `IVF_VAMANA_PQ`, pass `index_type="IVF_VAMANA_PQ"` with `num_sub_vectors` in Python, or use `with_ivf_vamana_pq_params` in Rust and `withIvfVamanaPqParams` in Java. Python accepts the type name in any letter case.

| Parameter | Default | Allowed values | What it sets |
|---|---|---|---|
| `metric` | `l2` | `l2`, `cosine` | The distance. |
| `num_partitions`, `target_partition_size` | a target of 1,048,576 rows per partition; 32 partitions in Java | as for the other IVF indices | How many IVF partitions the vectors are split into. Java sets a number of partitions, never a target size. |
| `max_degree` | 64 | 1 to 1024 | The most out-edges a vertex keeps. |
| `search_list_size` | 100 | 1 to 2^32 - 1; to 2^31 - 1 in Java | How many candidates each search keeps while the graph is built. Queries do not use it. |
| `alpha` | 1.2 | finite, at least 1.0 | The pruning slack of the build's second pass: a larger value keeps more long edges. |
| `num_bits` (SQ) | 8 | 8 | The bits of a scalar code. Rust and Java refuse other values; Python's `num_bits` sets only product codes, so its SQ codes always have 8 bits. |
| `num_sub_vectors`, `num_bits` (PQ) | 16 sub-vectors of 8 bits; Python requires `num_sub_vectors` | as for `IVF_PQ` | The shape of a product code. |

The other IVF and quantizer parameters (sample rates, given centroids or a given codebook) work as for `IVF_HNSW_*`. The defaults of `max_degree` and `search_list_size` were the best pair measured on SIFT1M against Lance's HNSW at matched memory, so change them only after measuring on your own data. Changing a graph parameter takes a rebuild (`create_index` with `replace=True`): `optimize_indices` keeps the graph parameters the index was built with, and a merge of index segments refuses segments built with different ones.

A partition of at least 100,000 vectors builds its graph on all cores; smaller partitions build on one thread each, several partitions at a time. The graph is the same whatever the number of threads.

## Search

Vamana indices take Lance's usual vector query options, plus `stop_margin`:

```python
ds.to_table(
    nearest={
        "column": "vector",
        "q": q,
        "k": 10,
        "ef": 2048,
        "stop_margin": 0.05,
    }
)
```

Without `refine_factor`, `IVF_VAMANA_SQ` and `IVF_VAMANA_PQ` re-score a shortlist of the `2 * k` best candidates and return `k` rows with exact distances, and `IVF_VAMANA_FLAT` returns the `k` nearest of its walks directly. With `refine_factor`, the index returns `k * refine_factor` candidates and Lance's refine re-scores them instead, so set it only when you want more candidates re-scored than the shortlist holds.

`ef` is the length of each walk's candidate list. It defaults to 1.5 times the shortlist, which is `2 * k`, or `k * refine_factor` when the query sets one, and it is never shorter than the shortlist: at `k=10` a walk keeps 30 candidates by default. An `ef` below `k` (below `k * refine_factor` with a refine factor) is refused.

`nprobes`, filters, `prefilter`, `distance_range` and deleted rows work as for the other IVF indices, except that the partitions are searched one after another unless the query sets `query_parallelism`. The lower bound of a distance range is inclusive and the upper bound exclusive. `approx_mode` is ignored.

### Stop the Walk Early

Without a stop rule, a walk ends once it has expanded every candidate in its list of `ef`. With `stop_margin` set to `m`, a walk stops as soon as none of its unexpanded candidates is among its `k` nearest or nearer than `1 + m` times the length to its `k`-th nearest. A query whose neighbors stand clear of everything else stops soon after finding them, and one crowded by near-equals walks further. At `m = 0` a walk expands its `k` nearest and stops; a larger margin walks further and finds more of the true neighbors.

`ef` stays the most candidates a walk keeps and becomes a cap: set it to a length no walk reaches, so that the margin decides where each walk stops. The measurements above used 2048 at top-10 and 4096 at top-100.

Pick the margin for each `k` by measuring recall on a sample of queries with exact answers. The measurements above reached their recall with 0.0375-0.055 at top-10 and 0.005-0.025 at top-100, so 0.05 and 0.02 are reasonable starting points. A search without the index (`"use_index": False`) gives the exact answers. Give it the index's metric: a search without the index ranks by `l2` unless the query names a metric.

```python
def recall(ds, queries, k, metric, **search):
    found = 0
    for q in queries:
        nearest = {"column": "vector", "q": q, "k": k, "metric": metric}
        exact = ds.to_table(
            columns=["id", "_distance"],
            nearest={**nearest, "use_index": False},
        )
        approx = ds.to_table(
            columns=["id", "_distance"],
            nearest={**nearest, **search},
        )
        found += len(set(exact["id"].to_pylist()) & set(approx["id"].to_pylist()))
    return found / (k * len(queries))


for margin in [0.0, 0.01, 0.02, 0.05, 0.1]:
    print(margin, recall(ds, queries, 10, "cosine", ef=2048, stop_margin=margin))
```

Under a filter or a distance range, the first walk with a margin is a trial. It answers only if it collected the rows it was asked for with at least half of its list allowed, or, in a distance-range query, if its list already reaches past the upper bound. Otherwise the partition is searched as it would be without the margin, which can walk a wider list or score the allowed rows directly: the answer stays correct and costs more. Other index types ignore `stop_margin`, and leaving it unset walks each list to its end.

The margin is a finite value of at least zero and belongs to a vector query. In Rust, call `stop_margin` after `nearest` (calling `nearest` again clears it); in Java, set it on the query builder:

=== "Rust"

    ```rust
    let mut scanner = dataset.scan();
    scanner.nearest("vector", &query, 10)?.ef(2048).stop_margin(0.05)?;
    ```

=== "Java"

    ```java
    Query query =
        new Query.Builder()
            .setColumn("vector")
            .setKey(vector)
            .setK(10)
            .setEf(2048)
            .setStopMargin(0.05f)
            .build();
    ```

The directory namespace builds Vamana indices with the default parameters only, and cannot set `stop_margin`.

## Memory

A loaded partition keeps its codes, each with an 8-byte row id (one byte per dimension for SQ, one byte per sub-vector for PQ, the full vector for FLAT, which adds a 4-byte norm under cosine), and its whole graph: another 8 bytes of row id and 4 bytes of offset per vertex, plus 4 bytes per edge. The graph thus takes at most `12 + 4 * max_degree` bytes per vertex, 268 bytes at the default degree of 64. Pruning leaves most vertices below the full degree: for the four datasets above, about a million vectors each, the graph took 90-196 MB on disk. Each thread that walks a partition also keeps one byte per vertex for its visit marks, sized to the largest partition it has walked.

Loaded partitions live in the session's index cache (`index_cache_size_bytes` of `lance.dataset`, or a `Session`). Size it for the loaded form of the partitions your queries touch, not for the size of the index files: as with Lance's other vector indices, a partition the cache cannot hold is loaded again by every query that searches it, which shows only as slow queries.

An `optimize_indices` merge that extends existing graphs holds the old graph of each partition it extends, with no byte budget: about 0.27 GiB for a partition of 1,048,576 rows, times the partitions merged at once (about 2.7 GiB for 10). This is an estimate, not a measurement. A fresh build with `create_index` peaked at the same or lower memory than `IVF_HNSW_SQ` in the measurements above.

## Keep the Index Current

- Appends: rows added after the index was built are searched by brute force until `optimize_indices` adds them to the index.
- Deletes: deleted rows are never returned. Compaction then drops them from the graphs, consolidating each affected partition or building it again. With stable row ids, compaction leaves them in the graphs until the next `optimize_indices` merge: `IVF_VAMANA_SQ` and `IVF_VAMANA_PQ` drop them at the re-score, where they still take places in the shortlist, and `IVF_VAMANA_FLAT`, as Lance's `IVF_FLAT` and `IVF_HNSW_FLAT` do, can return fewer than `k` rows. With a deferred remap, the affected partitions are scored directly, correct but slower, until the next `optimize_indices` merge or remap restores the graph search.
- Distributed builds: the shards of an index can be merged into one segment or committed side by side, as for the other IVF indices. Shards merged together must share the index type, the distance and the graph parameters.
- `describe_indices` and `index_statistics` report the type and the graph parameters.

## Limitations

- Distances: `l2` and `cosine` only. Python's own check lets `dot` and `hamming` through, and Lance then refuses them with an error.
- Vectors: one fixed-size list of 16-, 32- or 64-bit floats per row; multivector columns are not supported.
- Codes: full vectors, 8-bit SQ or PQ; no RaBitQ.
- Index file version V3 only, Lance's default for vector indices.
- Partitions: at most 2^32 - 1 vectors and 2^31 - 1 edges each, about 33.6 million vectors at a full degree of 64.
- Builds: a partition under 100,000 vectors builds its graph on one thread, and the rows `optimize_indices` inserts into an existing graph go in one at a time. The batched build runs on the rayon thread pool, which ignores `LANCE_CPU_THREADS`, as Lance's HNSW build does.
- Search: no batch or prepared partition search; `approx_mode` is ignored; the beam width (4 vertices expanded per step) and the prefetch distance are fixed.
- Filters: a filter that allows only rows far from the query leaves the walk short of allowed rows, and the partition's allowed rows are then scored directly: correct, but slower.
- Re-score reads: a query reads `2 * k` vectors for each index segment it searches, 2,000 at `k=1000`, and `2 * k` for each partition when the partitions are searched in parallel, or one by one by a filtered query that probes beyond its first partitions. On object storage that is at least one ranged request per data file per query, which has not been measured.
- The fastest re-score reads rows at their byte offsets, in place. That needs data files of Lance format 2.1 or 2.2 whose pages store 32-bit float vectors full-zip and uncompressed (Lance writes vectors of 64 or more dimensions that way); local files on 64-bit Linux, on ext2-4, xfs, btrfs, tmpfs or overlayfs, that the process already holds open (the first re-score of a file reads it through the scheduler and opens it for the next); and `LANCE_VAMANA_READ_IN_PLACE` unset or true, which a process reads once. Other rows are read through Lance's I/O scheduler or `take`, with the same answers, only slower. Setting `LANCE_VAMANA_READ_IN_PLACE=false` sends the reads at byte offsets through the scheduler.
- Statistics: the per-query distance counter leaves out the re-score; rows the re-score reads through `take` are missing from the query's I/O statistics, and in-place reads from the scheduler's process-wide counters.
- MemWAL does not maintain Vamana indices.
- A Lance build without these index types hides an `IVF_VAMANA_*` index, keeps it in the manifests it writes, and answers vector queries on the column without it. Without stable row ids or a deferred remap it cannot remap the index, so its compaction leaves the fragments the index covers as they are.
