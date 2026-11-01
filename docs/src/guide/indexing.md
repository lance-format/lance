---
title: Indexing and Search
description: Choose and maintain scalar, vector, and full-text indexes for Lance datasets.
---

# Indexing and Search

Lance indexes accelerate queries without changing the logical contents of a dataset. They are stored alongside the dataset and participate in its transactional versioning.

Use these guides to choose an index and keep it useful as data changes:

- [Scalar Indexes](scalar_indexing.md): accelerate SQL filters on scalar values and arrays with B-tree, bitmap, label-list, zone-map, and other indexes.
- [Full-Text Search](full_text_search.md): index text and run BM25-ranked keyword, phrase, fuzzy, and multi-column queries.
- [Vector Search](vector_search.md): choose an approximate nearest-neighbor index and tune latency, recall, storage, and filtering.

For logical row identity across updates and compaction, see [Stable Row IDs and Lineage](row_ids.md).

## Index Lifecycle

### Create an Index

Create scalar and full-text indexes with `LanceDataset.create_scalar_index`, and vector indexes with `LanceDataset.create_index`:

```python
# A B-tree index on an ID column
dataset.create_scalar_index("id", "BTREE")

# A BM25 full-text index on a text column
dataset.create_scalar_index("text", "INVERTED", with_position=True)

# An IVF_RQ vector index on a fixed-size vector column
dataset.create_index(
    "vector",
    index_type="IVF_RQ",
    num_partitions=256,
)
```

The default index name is `<column>_idx`. Pass `name=...` to choose a different name, which is useful when one column needs indexes for different query patterns.

### Inspect Indexes and Index Coverage

Use `describe_indices()` for index names, types, indexed columns, segment UUIDs, and indexed row counts. Use `stats.index_stats(name)` for index-specific details and unindexed fragment/row counts:

```python
for index in dataset.describe_indices():
    print(index.name, index.index_type, index.num_rows_indexed)

stats = dataset.stats.index_stats("text_idx")
print(stats["num_indexed_rows"], stats["num_unindexed_rows"])
```

To verify that a query uses a scalar index, inspect the execution plan. Scalar-index acceleration appears as `ScalarIndexQuery` or `MaterializeIndex`:

```python
print(dataset.scanner(filter="id >= 100").explain_plan())
```

### Maintain Indexes after Appends

Appending data does not automatically add it to existing indexes. By default, searches combine indexed results with unindexed scans, so newly appended rows remain visible. As unindexed data grows, latency can increase.

Call `optimize_indices()` periodically to index new data. This does not retrain vector models by default:

```python
dataset.insert(new_rows)
dataset.optimize.optimize_indices(index_names=["text_idx", "vector_idx"])
```

Omit `index_names` to optimize all indexes. For vector indexes, use `retrain=True` when the data distribution has changed enough that existing IVF centroids or quantizer models no longer provide good recall:

```python
dataset.optimize.optimize_indices(index_names=["vector_idx"], retrain=True)
```

Retraining rebuilds the selected indexes and can be expensive. See [Index Maintenance](vector_search.md#index-maintenance) for the vector-specific tradeoff.

`fast_search=True` limits searches to indexed coverage and can skip recently appended unindexed data. Use it only when lower latency is more important than seeing every newly appended row.

### Compaction and Rebuilding

Physical rewrites can invalidate row addresses used by indexes. Run compaction before rebuilding or optimizing indexes, and let Lance perform the index remap rather than preserving stale row addresses. Stable row IDs preserve logical row identity across updates and compaction, but they do not by themselves eliminate index maintenance.

See [Table Maintenance](read_and_write.md#table-maintenance) and [Performance Guide](performance.md) for compaction, remapping, memory, and cache details.

### Drop or Replace an Index

Drop an index by name:

```python
dataset.drop_index("id_idx")
```

To change tokenizer settings, index type, distance metric, or training parameters, create the index again with `replace=True` and the same name. For scalar indexes, `replace=True` is the default; for vector indexes, pass it explicitly.
