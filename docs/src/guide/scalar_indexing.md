---
title: Scalar Indexes
description: Accelerate SQL filters with B-tree, bitmap, label-list, zone-map, n-gram, bloom-filter, and FM indexes.
---

# Scalar Indexes

Scalar indexes accelerate filter pushdown on columns such as IDs, categories, timestamps, text, and arrays of labels. Lance applies the index to produce candidate rows and then evaluates the full filter, so index choice does not change query results.

This page covers exact scalar indexes and substring-oriented indexes. For BM25 text search, see [Full-Text Search](full_text_search.md). For JSON paths, see [JSON Support](json.md#json-indexing).

The workflows below cover the Python (`pylance`), Rust (`lance`), and Java (`org.lance:lance-core`) SDKs.

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

=== "Python"

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

=== "Rust"

    ```rust
    use lance::index::DatasetIndexExt;
    use lance::Dataset;
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_index::IndexType;

    let mut dataset = Dataset::open("scalar_indexes.lance").await?;

    dataset
        .create_index(
            &["id"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            true,
        )
        .await?;

    dataset
        .create_index(
            &["category"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
            true,
        )
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.Dataset;
    import org.lance.index.IndexOptions;
    import org.lance.index.IndexParams;
    import org.lance.index.IndexType;
    import org.lance.index.scalar.BTreeIndexParams;
    import org.lance.index.scalar.BitmapIndexParams;
    import java.util.Collections;

    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("id"),
                IndexType.BTREE,
                IndexParams.builder()
                    .setScalarIndexParams(BTreeIndexParams.builder().build())
                    .build())
            .replace(true)
            .build());

    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("category"),
                IndexType.BITMAP,
                IndexParams.builder()
                    .setScalarIndexParams(BitmapIndexParams.builder().build())
                    .build())
            .replace(true)
            .build());
    ```

By default, the index name is `<column>_idx`. In Python `create_scalar_index` replaces an existing index with the same name by default (`replace=True`); in Rust and Java, pass `replace = true` / `.replace(true)` explicitly when replacing an existing index.

For approximately sorted data, a zone map can provide cheap range pruning:

=== "Python"

    ```python
    dataset.create_scalar_index("created_at", "ZONEMAP")
    ```

=== "Rust"

    ```rust
    dataset
        .create_index(
            &["created_at"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
            true,
        )
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.index.scalar.ZoneMapIndexParams;

    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("created_at"),
                IndexType.ZONEMAP,
                IndexParams.builder()
                    .setScalarIndexParams(ZoneMapIndexParams.builder().build())
                    .build())
            .replace(true)
            .build());
    ```

For tag arrays, use a label-list index:

=== "Python"

    ```python
    dataset.create_scalar_index("tags", "LABEL_LIST")
    ```

=== "Rust"

    ```rust
    dataset
        .create_index(
            &["tags"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList),
            true,
        )
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.index.scalar.LabelListIndexParams;

    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("tags"),
                IndexType.LABEL_LIST,
                IndexParams.builder()
                    .setScalarIndexParams(LabelListIndexParams.builder().build())
                    .build())
            .replace(true)
            .build());
    ```

For substring and full-text workloads, create the index that matches the query semantics rather than using a B-tree:

=== "Python"

    ```python
    # Substring filters using contains(text, '...')
    dataset.create_scalar_index("text", "NGRAM", name="text_ngram_idx")

    # BM25-ranked full_text_query searches
    dataset.create_scalar_index("text", "INVERTED", name="text_fts_idx", with_position=True)
    ```

=== "Rust"

    ```rust
    use lance_index::scalar::InvertedIndexParams;

    // Substring filters using contains(text, '...')
    dataset
        .create_index(
            &["text"],
            IndexType::Scalar,
            Some("text_ngram_idx".to_string()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::NGram),
            true,
        )
        .await?;

    // BM25-ranked full-text searches
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("text_fts_idx".to_string()),
            &InvertedIndexParams::default().with_position(true),
            true,
        )
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.index.scalar.InvertedIndexParams;
    import org.lance.index.scalar.NGramIndexParams;

    // Substring filters using contains(text, '...')
    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("text"),
                IndexType.NGRAM,
                IndexParams.builder()
                    .setScalarIndexParams(NGramIndexParams.builder().build())
                    .build())
            .withIndexName("text_ngram_idx")
            .replace(true)
            .build());

    // BM25-ranked full-text searches
    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("text"),
                IndexType.INVERTED,
                IndexParams.builder()
                    .setScalarIndexParams(
                        InvertedIndexParams.builder().withPosition(true).build())
                    .build())
            .withIndexName("text_fts_idx")
            .replace(true)
            .build());
    ```

## Query with Scalar Indexes

No special query API is required. Lance selects usable indexes automatically from SQL filters:

=== "Python"

    ```python
    result = dataset.to_table(
        columns=["id", "category"],
        filter="category = 'tech' AND id >= 100",
        limit=10,
    )
    ```

=== "Rust"

    ```rust
    use arrow_array::RecordBatch;
    use futures::TryStreamExt;

    let batches: Vec<RecordBatch> = dataset
        .scan()
        .project(&["id", "category"])?
        .filter("category = 'tech' AND id >= 100")?
        .limit(Some(10), None)?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    import org.apache.arrow.vector.ipc.ArrowReader;
    import org.lance.ipc.LanceScanner;
    import org.lance.ipc.ScanOptions;
    import java.util.Arrays;

    ScanOptions options =
        new ScanOptions.Builder()
            .columns(Arrays.asList("id", "category"))
            .filter("category = 'tech' AND id >= 100")
            .limit(10)
            .build();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

Array membership filters can use a label-list index:

=== "Python"

    ```python
    result = dataset.to_table(filter="array_has_any(tags, ['python', 'rust'])")
    ```

=== "Rust"

    ```rust
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .filter("array_has_any(tags, ['python', 'rust'])")?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    ScanOptions options =
        new ScanOptions.Builder()
            .filter("array_has_any(tags, ['python', 'rust'])")
            .build();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

Scalar indexes support common equality, comparison, range, set-membership, and null predicates. Multiple indexed predicates can be combined with `AND` or `OR`. A predicate such as `id = 1 OR unindexed_column = 2` may not be able to use the index on `id`, because the unindexed branch must also be evaluated. An `OR` whose branches each include an indexed predicate, such as `(id = 1 AND unindexed_column = 2) OR id = 3`, still uses the index: the index finds the candidate rows and the whole filter is applied to them.

Set `use_scalar_index=False` (Python), `.use_scalar_index(false)` (Rust), or `.useScalarIndex(false)` (Java) on a scanner to disable scalar-index acceleration for a specific query.

## Verify Index Usage

Inspect the query plan to confirm that a filter uses an index. `ScalarIndexQuery` or `MaterializeIndex` indicates scalar-index acceleration:

=== "Python"

    ```python
    print(dataset.scanner(filter="id >= 100").explain_plan())
    ```

=== "Rust"

    ```rust
    let plan = dataset.scan().filter("id >= 100")?.explain_plan(false).await?;
    println!("{plan}");
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

=== "Python"

    ```python
    dataset.optimize.optimize_indices(index_names=["id_idx", "category_idx"])
    ```

=== "Rust"

    ```rust
    use lance_index::optimize::OptimizeOptions;

    dataset
        .optimize_indices(&OptimizeOptions::new().index_names(vec![
            "id_idx".to_string(),
            "category_idx".to_string(),
        ]))
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.index.OptimizeOptions;
    import java.util.Arrays;

    dataset.optimizeIndices(
        OptimizeOptions.builder()
            .indexNames(Arrays.asList("id_idx", "category_idx"))
            .build());
    ```

Omit `index_names` to optimize every index. Use `fast_search=True` only when it is acceptable to skip recently appended unindexed data in exchange for lower latency.

For more on updates, compaction, and replacement, see [Indexing and Search](indexing.md#index-lifecycle).

## Nulls and Empty Collections

Scalar indexes account for null values so SQL three-valued logic remains correct. A filter that evaluates to null is not the same as a filter that evaluates to true. Test queries on data containing null scalar values, null list elements, empty lists, and all-null columns rather than assuming that an index treats them as ordinary values.

To exclude nulls explicitly, add the predicate to the query:

```python
result = dataset.to_table(filter="category IS NOT NULL AND category = 'tech'")
```
