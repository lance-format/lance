---
title: Scalar Indexes
description: Accelerate SQL filters with B-tree, bitmap, label-list, zone-map, n-gram, bloom-filter, and FM indexes.
---

# Scalar Indexes

Scalar indexes accelerate filter pushdown on columns such as IDs, categories, timestamps, text, and arrays of labels. Lance applies the index to produce candidate rows and then evaluates the full filter, so index choice does not change query results.

This page covers exact scalar indexes and substring-oriented indexes. For BM25 text search, see [Full-Text Search](full_text_search.md). For JSON paths, see [JSON Support](json.md#json-indexing).

## Choose an Index Type

| Index type | Typical column or workload | Accelerates | Main limitation |
|---|---|---|---|
| `BTREE` | High-cardinality scalar values, such as IDs and timestamps | Equality, ranges, `IN`, null predicates | Larger than lightweight skipping structures |
| `BITMAP` | Low-cardinality values with many rows per value, such as categories | Equality, small ranges, set membership | Many unique values increase index size and build memory |
| `LABEL_LIST` | List columns whose elements have low cardinality, such as tags | `array_has`, `array_contains`, `array_has_all`, `array_has_any` | Not intended for arbitrary high-cardinality arrays |
| `ZONEMAP` | Approximately sorted columns, such as append-time timestamps | Range predicates by skipping zones | Little benefit when values are randomly distributed |
| `NGRAM` | Substring search in text | `contains(column, 'substring')` | May return candidates that need rechecking; use `INVERTED` for BM25 search |
| `BLOOMFILTER` | Membership-oriented equality checks | Equality and inequality filters | Inexact and limited predicate support |
| `FM` | Exact substring search, including one-character patterns | `contains(column, 'substring')` | Specialized index; use scalar indexes for ordinary comparisons |

B-tree is the usual default for scalar columns. Choose bitmap for genuinely low-cardinality columns, and zone maps when data is sorted or nearly sorted. See [Performance Guide](performance.md#indexes) for resource tradeoffs.

## Create Indexes

Create a B-tree index on a high-cardinality ID and a bitmap index on a low-cardinality category:

```python
import lance
import pyarrow as pa

uri = "scalar_indexes.lance"
table = pa.table(
    {
        "id": pa.array(range(1000), pa.int64()),
        "category": pa.array(["tech", "life"] * 500),
        "created_at": pa.array(range(1000), pa.timestamp("s")),
        "tags": pa.array([["python", "ml"], ["rust"]] * 500),
    }
)
dataset = lance.write_dataset(table, uri)

dataset.create_scalar_index("id", "BTREE")
dataset.create_scalar_index("category", "BITMAP")
```

By default, the index name is `<column>_idx`, and an existing index with the same name is replaced. Pass `name=...` and `replace=False` when you want to preserve an existing index.

For approximately sorted data, a zone map can provide cheap range pruning:

```python
dataset.create_scalar_index("created_at", "ZONEMAP")
```

For tag arrays, use a label-list index:

```python
dataset.create_scalar_index("tags", "LABEL_LIST")
```

For substring and full-text workloads, create the index that matches the query semantics rather than using a B-tree:

```python
# Substring filters using contains(text, '...')
dataset.create_scalar_index("text", "NGRAM")

# BM25-ranked full_text_query searches
dataset.create_scalar_index("text", "INVERTED", with_position=True)
```

## Query with Scalar Indexes

No special query API is required. Lance selects usable indexes automatically from SQL filters:

```python
result = dataset.to_table(
    columns=["id", "category"],
    filter="category = 'tech' AND id >= 100",
    limit=10,
)
```

Array membership filters can use a label-list index:

```python
result = dataset.to_table(filter="array_has_any(tags, ['python', 'rust'])")
```

Scalar indexes support common equality, comparison, range, set-membership, and null predicates. Multiple indexed predicates can be combined with `AND` or `OR`. A predicate such as `id = 1 OR unindexed_column = 2` may not be able to use the index on `id`, because the unindexed branch must also be evaluated.

Set `use_scalar_index=False` on a scanner to disable scalar-index acceleration for a specific query.

## Verify Index Usage

Inspect the query plan to confirm that a filter uses an index. `ScalarIndexQuery` or `MaterializeIndex` indicates scalar-index acceleration:

```python
print(dataset.scanner(filter="id >= 100").explain_plan())
```

Inspect index coverage and segment metadata with `describe_indices()`:

```python
for index in dataset.describe_indices():
    print(index.name, index.index_type, index.field_names)
    print(index.num_rows_indexed)
```

For a concise check of unindexed data, use index statistics:

```python
stats = dataset.stats.index_stats("id_idx")
print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
```

## Maintain Indexes

Appends do not automatically add rows to existing scalar indexes. Queries still include new rows by scanning unindexed fragments, but this becomes more expensive as unindexed coverage grows. Periodically run:

```python
dataset.optimize.optimize_indices(index_names=["id_idx", "category_idx"])
```

Omit `index_names` to optimize every index. Use `fast_search=True` only when it is acceptable to skip recently appended unindexed data in exchange for lower latency.

For more on updates, compaction, and replacement, see [Indexing and Search](indexing.md#index-lifecycle).

## Nulls and Empty Collections

Scalar indexes account for null values so SQL three-valued logic remains correct. A filter that evaluates to null is not the same as a filter that evaluates to true. Test queries on data containing null scalar values, null list elements, empty lists, and all-null columns rather than assuming that an index treats them as ordinary values.

To exclude nulls explicitly, add the predicate to the query:

```python
result = dataset.to_table(filter="category IS NOT NULL AND category = 'tech'")
```
