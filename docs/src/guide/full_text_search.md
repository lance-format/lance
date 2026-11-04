---
title: Full-Text Search
description: Build inverted indexes and run BM25-ranked keyword, phrase, fuzzy, and multi-column text searches.
---

# Full-Text Search

Lance full-text search uses an inverted index and BM25 relevance scoring. It supports token matching, boolean queries, exact phrases, fuzzy matching, and searches across multiple text columns.

For ordinary substring filters, use an [N-gram or FM scalar index](scalar_indexing.md#choose-an-index-type). Those indexes accelerate `contains(...)` filters but do not provide BM25 ranking.

This guide covers the Python (`pylance`), Rust (`lance`), and Java (`org.lance:lance-core`) SDKs.

## Create a Full-Text Index

Create an `INVERTED` index on a text column:

=== "Python"

    ```python
    import lance
    import pyarrow as pa

    uri = "full_text_search.lance"
    table = pa.table(
        {
            "id": [1, 2, 3],
            "title": ["Lance guide", "Cooking ramen", "Evening train"],
            "body": [
                "A columnar format for multimodal data",
                "Simmer the ramen broth for three hours",
                "The train leaves for Boston in the evening",
            ],
            "category": ["docs", "food", "travel"],
        }
    )
    dataset = lance.write_dataset(table, uri)
    dataset.create_scalar_index("body", "INVERTED")
    ```

=== "Rust"

    ```rust
    use lance::index::DatasetIndexExt;
    use lance::Dataset;
    use lance_index::scalar::InvertedIndexParams;
    use lance_index::IndexType;

    let mut dataset = Dataset::open("full_text_search.lance").await?;
    dataset
        .create_index(
            &["body"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default(),
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
    import org.lance.index.scalar.InvertedIndexParams;
    import java.util.Collections;

    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("body"),
                IndexType.INVERTED,
                IndexParams.builder()
                    .setScalarIndexParams(InvertedIndexParams.builder().build())
                    .build())
            .replace(true)
            .build());
    ```

The default index name is `<column>_idx`. Use a custom `name=...` when a column needs more than one index, such as both an inverted index and an N-gram index.

Use `with_position=True` when phrase queries are required. Positions increase index size but do not affect non-phrase query performance:

=== "Python"

    ```python
    dataset.create_scalar_index("body", "INVERTED", with_position=True)
    ```

=== "Rust"

    ```rust
    dataset
        .create_index(
            &["body"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().with_position(true),
            true,
        )
        .await?;
    ```

=== "Java"

    ```java
    dataset.createIndex(
        IndexOptions.builder(
                Collections.singletonList("body"),
                IndexType.INVERTED,
                IndexParams.builder()
                    .setScalarIndexParams(
                        InvertedIndexParams.builder().withPosition(true).build())
                    .build())
            .replace(true)
            .build());
    ```

Common analyzer options include:

- `base_tokenizer`: controls tokenization, such as `simple`, `whitespace`, `raw`, `icu`, or language-specific tokenizers.
- `lower_case`, `stem`, `remove_stop_words`, `ascii_folding`: control case normalization, stemming, stop-word removal, and accent folding.
- `language`: selects the language for stemming and default stop words.
- `max_token_length`: an exclusive UTF-8 byte limit; tokens whose encoded length is at least the limit are removed.
- `analyzer="code"`: enables the code-analysis profile and its identifier/operator options.

See [Tokenizers](tokenizer.md) for language models, code analysis, and token inspection with `lance.tokenize`.

## Run Keyword Searches

Results are ranked by BM25, with higher `_score` values indicating stronger matches. In Python and Rust, passing a query string searches all indexed text columns:

=== "Python"

    ```python
    result = dataset.to_table(full_text_query="train Boston")
    ```

=== "Rust"

    ```rust
    use arrow_array::RecordBatch;
    use futures::TryStreamExt;
    use lance_index::scalar::FullTextSearchQuery;

    let batches: Vec<RecordBatch> = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new("train Boston".to_string()))?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

A string query uses the dataset's searchable text configuration. To select a column and control matching explicitly, use `MatchQuery`:

=== "Python"

    ```python
    from lance.query import FullTextOperator, MatchQuery

    # Both terms must be present
    query = MatchQuery("train Boston", "body", operator=FullTextOperator.AND)
    result = dataset.to_table(full_text_query=query)

    # Either term may be present
    query = MatchQuery("ramen broth", "body", operator=FullTextOperator.OR)
    result = dataset.to_table(full_text_query=query)
    ```

=== "Rust"

    ```rust
    use lance_index::scalar::inverted::query::{MatchQuery, Operator};

    // Both terms must be present
    let query = MatchQuery::new("train Boston".to_string())
        .with_column(Some("body".to_string()))
        .with_operator(Operator::And);
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(query.into()))?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    import org.apache.arrow.vector.ipc.ArrowReader;
    import org.lance.ipc.FullTextQuery;
    import org.lance.ipc.LanceScanner;
    import org.lance.ipc.ScanOptions;
    import java.util.Optional;

    // Both terms must be present
    ScanOptions options =
        new ScanOptions.Builder()
            .fullTextQuery(
                FullTextQuery.match(
                    "train Boston",
                    "body",
                    1.0f,
                    Optional.empty(),
                    50,
                    FullTextQuery.Operator.AND,
                    0))
            .build();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

Combine metadata filtering in the same scan:

=== "Python"

    ```python
    result = dataset.to_table(
        full_text_query=MatchQuery("train", "body"),
        filter="category = 'travel'",
    )
    ```

=== "Rust"

    ```rust
    let query = MatchQuery::new("train".to_string()).with_column(Some("body".to_string()));
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(query.into()))?
        .filter("category = 'travel'")?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    ScanOptions options =
        new ScanOptions.Builder()
            .fullTextQuery(FullTextQuery.match("train", "body"))
            .filter("category = 'travel'")
            .build();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

## Phrase Queries

Phrase queries require an index built with `with_position=True`:

=== "Python"

    ```python
    from lance.query import PhraseQuery

    result = dataset.to_table(full_text_query=PhraseQuery("ramen broth", "body"))
    ```

=== "Rust"

    ```rust
    use lance_index::scalar::inverted::query::PhraseQuery;

    let phrase = PhraseQuery::new("ramen broth".to_string())
        .with_column(Some("body".to_string()));
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(phrase.into()))?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    ScanOptions options =
        new ScanOptions.Builder()
            .fullTextQuery(FullTextQuery.phrase("ramen broth", "body"))
            .build();
    try (LanceScanner scanner = dataset.newScan(options);
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

Stop words affect phrase positions. If phrases must contain stop words, build the index with `remove_stop_words=False`; otherwise those words are absent from the indexed positions and the exact phrase may not match.

## Boolean and Fuzzy Queries

Use `BooleanQuery` and `Occur` when some clauses are required and others are forbidden:

=== "Python"

    ```python
    from lance.query import BooleanQuery, MatchQuery, Occur

    query = BooleanQuery(
        [
            (Occur.MUST, MatchQuery("train", "body")),
            (Occur.MUST_NOT, MatchQuery("Boston", "body")),
        ]
    )
    result = dataset.to_table(full_text_query=query)
    ```

=== "Rust"

    ```rust
    use lance_index::scalar::inverted::query::{BooleanQuery, FtsQuery, Occur};

    let boolean_query = BooleanQuery::new([
        (
            Occur::Must,
            FtsQuery::Match(
                MatchQuery::new("train".to_string()).with_column(Some("body".to_string())),
            ),
        ),
        (
            Occur::MustNot,
            FtsQuery::Match(
                MatchQuery::new("Boston".to_string()).with_column(Some("body".to_string())),
            ),
        ),
    ]);
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(boolean_query.into()))?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    import java.util.Arrays;

    FullTextQuery booleanQuery =
        FullTextQuery.booleanQuery(
            Arrays.asList(
                new FullTextQuery.BooleanClause(
                    FullTextQuery.Occur.MUST, FullTextQuery.match("train", "body")),
                new FullTextQuery.BooleanClause(
                    FullTextQuery.Occur.MUST_NOT, FullTextQuery.match("Boston", "body"))));
    try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().fullTextQuery(booleanQuery).build());
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

`MatchQuery` also supports fuzzy matching with Levenshtein edit distance:

=== "Python"

    ```python
    query = MatchQuery(
        "ramen",  # matches "ramen" in the document body
        "body",
        fuzziness=1,
        max_expansions=50,
    )
    result = dataset.to_table(full_text_query=query)
    ```

=== "Rust"

    ```rust
    let fuzzy = MatchQuery::new("ramen".to_string())
        .with_column(Some("body".to_string()))
        .with_fuzziness(Some(1))
        .with_max_expansions(50);
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(fuzzy.into()))?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    FullTextQuery fuzzy =
        FullTextQuery.match(
            "ramen", "body", 1.0f, Optional.of(1), 50, FullTextQuery.Operator.OR, 0);
    try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().fullTextQuery(fuzzy).build());
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

Set `fuzziness=None` for automatic fuzziness based on term length. Use `prefix_length` to require an exact prefix before edits are allowed.

## Search Multiple Columns

Use `MultiMatchQuery` when the best score from any column should win:

=== "Python"

    ```python
    from lance.query import MultiMatchQuery

    query = MultiMatchQuery(
        "columnar guide",
        ["title", "body"],
        boosts=[2.0, 1.0],  # title matches count more
    )
    result = dataset.to_table(full_text_query=query)
    ```

=== "Rust"

    ```rust
    use lance_index::scalar::inverted::query::MultiMatchQuery;

    let multi = MultiMatchQuery::try_new(
        "columnar guide".to_string(),
        vec!["title".to_string(), "body".to_string()],
    )?
    .try_with_boosts(vec![2.0, 1.0])?;
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(multi.into()))?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    FullTextQuery multi =
        FullTextQuery.multiMatch(
            "columnar guide",
            Arrays.asList("title", "body"),
            Arrays.asList(2.0f, 1.0f),
            FullTextQuery.Operator.OR);
    try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().fullTextQuery(multi).build());
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

Use `CombinedFieldsQuery` (BM25F) when terms may be spread across columns and their corpus statistics should be combined:

=== "Python"

    ```python
    from lance.query import CombinedFieldsQuery, FullTextOperator

    query = CombinedFieldsQuery(
        "Lance multimodal",
        ["title", "body"],
        operator=FullTextOperator.AND,
    )
    result = dataset.to_table(full_text_query=query)
    ```

=== "Rust"

    ```rust
    use lance_index::scalar::inverted::query::CombinedFieldsQuery;

    let combined = CombinedFieldsQuery::try_new(
        "Lance multimodal".to_string(),
        vec!["title".to_string(), "body".to_string()],
    )?
    .with_operator(Operator::And);
    let batches: Vec<RecordBatch> = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new_query(combined.into()))?
        .try_into_stream()
        .await?
        .try_collect()
        .await?;
    ```

=== "Java"

    ```java
    FullTextQuery combined =
        FullTextQuery.combinedFields(
            "Lance multimodal",
            Arrays.asList("title", "body"),
            null,
            FullTextQuery.Operator.AND);
    try (LanceScanner scanner =
            dataset.newScan(new ScanOptions.Builder().fullTextQuery(combined).build());
        ArrowReader reader = scanner.scanBatches()) {
      while (reader.loadNextBatch()) {
        // process reader.getVectorSchemaRoot()
      }
    }
    ```

Columns in a combined-fields query must use the same tokenizer and index configuration.

## Maintain Full-Text Indexes

Appended rows remain searchable: Lance combines indexed search with unindexed scans. As unindexed data grows, periodically add new fragments to the index:

=== "Python"

    ```python
    dataset.insert(new_rows)
    dataset.optimize.optimize_indices(index_names=["body_idx"])

    stats = dataset.stats.index_stats("body_idx")
    print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
    ```

=== "Rust"

    ```rust
    use lance_index::optimize::OptimizeOptions;

    dataset.append(new_rows_reader, None).await?;
    dataset
        .optimize_indices(
            &OptimizeOptions::new().index_names(vec!["body_idx".to_string()]),
        )
        .await?;
    ```

=== "Java"

    ```java
    import org.lance.index.OptimizeOptions;
    import java.util.Collections;

    dataset.optimizeIndices(
        OptimizeOptions.builder()
            .indexNames(Collections.singletonList("body_idx"))
            .build());
    ```

After changing tokenizer or analyzer settings, rebuild the index with `create_scalar_index(..., replace=True)` so the entire dataset uses the new configuration. `optimize_indices()` adds new data to existing indexes; it does not change their analyzer settings.

For large builds, `num_workers` and `memory_limit` control build parallelism and memory. See [Performance Guide](performance.md#indexes) and the [FTS format specification](../format/index/scalar/fts.md) for resource and compatibility details.

## Full-Text and Vector Search

A scanner can combine full-text relevance and vector similarity. See [Combine Full-Text and Vector Search](vector_search.md#combine-full-text-and-vector-search) for the query form and filtering behavior.

For a runnable tutorial, see the [Full-Text Search Quickstart](../quickstart/full-text-search.md).
