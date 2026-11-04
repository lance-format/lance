---
title: Full-Text Search
description: Full-text search (FTS) with inverted BM25 indexes and N-gram search in Lance
---

# Full-Text Search in Lance

Lance provides powerful full-text search (FTS) capabilities using an inverted index. This tutorial guides you through building and using FTS indexes to dramatically speed up text search operations while maintaining high accuracy.

By the end of this tutorial, you'll be able to build and use an FTS index, understand performance differences between indexed and non-indexed searches, and learn how to tune search parameters for optimal performance.

## Install the Python SDK

First, install the required dependencies:

```bash
--8<-- "snippets/quickstart/pylance-install.sh"
```

## Set Up Your Environment

Import the necessary libraries for working with Lance datasets:

```python
--8<-- "snippets/quickstart/fts_quickstart.py:imports"
```

## Prepare Your Text Data

In this quickstart, we'll create a simple dataset with text documents:

```python
--8<-- "snippets/quickstart/fts_quickstart.py:prepare-data"
```

This creates a Lance dataset with three text documents containing overlapping keywords that we'll use to demonstrate different search scenarios.

## Explore Your Dataset Schema

Let's examine the structure of our dataset:

```python
--8<-- "snippets/quickstart/fts_quickstart.py:schema"
```

This prints the PyArrow schema of the dataset:

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.schema.txt"
```

## Build the Full-Text Search Index

Full-text search is created with an inverted scalar index on your text column. Choose the `INVERTED` index type when calling `create_scalar_index` on your Lance dataset. Lance uses the BM25 ranking algorithm for relevance scoring. Results are automatically ranked by relevance, with higher scores indicating better matches.

```python
--8<-- "snippets/quickstart/fts_quickstart.py:build-index"
```

The index creation process builds an efficient lookup structure that maps words to the documents containing them. This enables high-performance keyword-based search, even on large datasets.

!!! warning "Index Creation Time"
    Index creation time depends on the size of your text data. For large datasets, this process may take several minutes, but the performance benefits at query time are substantial.

## Advanced Index Configuration

You can customize the index creation with various parameters to optimize for your specific use case:

```python
--8<-- "snippets/quickstart/fts_quickstart.py:advanced-index"
```

### Tokenizer Options

- **simple**: Splits tokens on whitespace and punctuation
- **whitespace**: Splits tokens only on whitespace
- **raw**: No tokenization (useful for exact matching)
- **code**: Splits code-like text into identifiers and operators for source-code search. Use `analyzer="code"`, which changes the tokenizer defaults. See the [code analyzer](../guide/tokenizer.md#code-analyzer) section for its flags and its FTS format v3 requirement.

Lance also supports multilingual tokenization:

- **icu**: Unicode word segmentation with built-in ICU dictionaries
- **jieba/default**: Chinese text tokenization using Jieba
- **lindera/ipadic**: Japanese text tokenization using Lindera with IPAdic dictionary
- **lindera/ko-dic**: Korean text tokenization using Lindera with Ko-dic dictionary
- **lindera/unidic**: Japanese text tokenization using Lindera with UniDic dictionary

### Language Processing Features

- **stemming**: Reduces words to their root form (e.g., "running" → "run")
- **stop words**: Removes common words like "the", "and", "is"
- **ascii folding**: Converts accented characters to ASCII (e.g., "é" → "e")

## Search With FTS Queries

Now you can run FTS queries using your inverted index:

```python
--8<-- "snippets/quickstart/fts_quickstart.py:basic-search"
```

This query returns documents that contain either "umbrella" or "train" (or both). The search is case-insensitive and uses the inverted index for fast retrieval.

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.basic-search.txt"
```

## Combining Full-Text Search with Metadata

It can be useful to combine FTS with metadata filtering in a single query to find more relevant results.
You can do this by passing a filter expression to the `filter` parameter.

```python
--8<-- "snippets/quickstart/fts_quickstart.py:metadata-filter"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.metadata-filter.txt"
```

## Advanced Search Features

### Boolean Search Operators

You can use boolean search operators by constructing a structured query object.

#### All terms: `AND`

```python
--8<-- "snippets/quickstart/fts_quickstart.py:boolean-and"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.boolean-and.txt"
```

#### Any terms: `OR`

```python
--8<-- "snippets/quickstart/fts_quickstart.py:boolean-or"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.boolean-or.txt"
```

#### Mix `AND`/`OR` queries via operators

You can mix `AND`/`OR` queries using operators in Python:

```python
--8<-- "snippets/quickstart/fts_quickstart.py:boolean-mix"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.boolean-mix.txt"
```

To combine `OR` queries via operators, use the pattern `q1 | q2`.

Every query combined with `AND` becomes a scoring `MUST` clause: all clauses must match,
and every matching clause contributes to the final `_score`.

#### Exclude terms: `NOT`

Queries that exclude specific keywords are explicitly written using `BooleanQuery`/`Occur`
as shown below.

```python
--8<-- "snippets/quickstart/fts_quickstart.py:boolean-not"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.boolean-not.txt"
```

### Phrase Search

For exact phrase matching, ensure you enable `with_position=True` during index creation (disabled by default) and wrap the phrase in double quotes (or pass a `PhraseQuery`):

```python
--8<-- "snippets/quickstart/fts_quickstart.py:phrase-search"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.phrase-search.txt"
```

!!! warning "Stop Words Are Removed by Default"
    Common words like "to", "the", etc. are categorized as stop words and are removed by default when creating the index. If you want to search exact phrases that include stop words, set `remove_stop_words=False` when creating the index.

### Substring matches with N-gram indexing

`NGRAM` is a type of scalar index for **substring / pattern-style** searches over text. It is a good alternative to wildcard-style queries like `term*` / `*term` (which are not parsed by `full_text_query` in Lance).

The N-gram index creates a bitmap for each N-gram in the string. By default, Lance uses trigrams. This index can be used to speed up queries using the `contains` function in filters.

```python
--8<-- "snippets/quickstart/fts_quickstart.py:ngram-search"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.ngram-search.txt"
```

You can explain the query plan to confirm the N-gram index's usage as shown below:

```python
--8<-- "snippets/quickstart/fts_quickstart.py:ngram-explain"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.ngram-explain.txt"
```

### Fuzzy Search

Fuzzy search is supported for FTS `MatchQuery` on `INVERTED` indexes. Pass a positive integer `fuzziness` (typically `1` or `2`) to specify the maximum Levenshtein edit distance used to match terms with typos or slight variations:

```python
--8<-- "snippets/quickstart/fts_quickstart.py:fuzzy-explicit"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.fuzzy-explicit.txt"
```

To enforce exact prefixes during fuzzy matching, set `prefix_length`.
This means the first `N` characters must match exactly before fuzzy edits are allowed on the rest of the term.
For example, with `prefix_length=2`, `"rammen"` can match terms starting with `"ra"` (like `"ramen"`), but not terms starting with other prefixes.

```python
--8<-- "snippets/quickstart/fts_quickstart.py:fuzzy-prefix"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.fuzzy-prefix.txt"
```

### Searching Multiple Columns

When a query should span several text columns (for example `title` and `body`),
Lance offers two strategies. They differ in how they combine the matches from
each column.

#### `best_fields` with `MultiMatchQuery`

`MultiMatchQuery` runs the query against each column independently and keeps the
score of the best matching column for each row. Each column is scored against
its own corpus statistics, and an optional per-column `boosts` multiplier is
applied before taking the maximum.

```python
--8<-- "snippets/quickstart/fts_quickstart.py:multi-match"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.multi-match.txt"
```

This works well when the columns are interchangeable and a strong match in one
column is enough to make a row relevant.

#### `combined_fields` (BM25F) with `CombinedFieldsQuery`

`CombinedFieldsQuery` instead treats the target columns as one virtual field and
combines their statistics into a single BM25F score. This makes scores
comparable across columns, and a query can match with terms spread over
different columns.

```python
--8<-- "snippets/quickstart/fts_quickstart.py:combined-fields"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.combined-fields.txt"
```

Prefer `combined_fields` in these cases:

- A term is rare in one column but common in another. Because document
  frequencies are combined, a chance hit in a short column does not outrank a
  better match elsewhere. `best_fields` cannot do this.
- The query terms are spread over several columns. With `operator=AND`,
  `"john smith"` matches a row whose `first_name` is `"john"` and whose
  `last_name` is `"smith"`, even though neither column contains both terms.
  `best_fields` would need a single column to contain the whole query.

Per-column `boosts` set how much each column contributes to the combined term
frequency. Each weight must be at least `1`, and fractional weights are allowed.
For example, a title boost of `2` counts every title term twice.

!!! note "Shared tokenizer required"

    All columns in a `CombinedFieldsQuery` must use the same tokenizer and index
    configuration, because BM25F only makes sense when the columns are tokenized
    the same way. Mixing configurations raises an error that lists the columns
    involved.

!!! note "Indexes and unindexed data"

    At least one target column needs an FTS index. Columns without an index, and
    rows added since the last index build, are read from the data and scored
    together with the indexed rows. Scanning is slower than reading an index, so
    optimize the indexes after adding a lot of data.

    BM25F adds up each column's contribution for a whole row, so a column indexed
    only with `document_granularity="list_element"` is rejected. Its per-element
    documents have no counterpart in the other columns, and the result has no
    `_doc_index` column. A column that has both a row index and a list-element
    index uses the row index.

## Performance Tips

### Index Maintenance

When you append new rows after creating an `INVERTED` index, Lance still returns those rows in `full_text_query` results. It searches indexed fragments using the FTS index, scans unindexed fragments with flat search, and then merges the results.

To keep FTS latency low as new data arrives, periodically add unindexed fragments into the existing FTS index by calling `ds.optimize.optimize_indices()`:

```python
--8<-- "snippets/quickstart/fts_quickstart.py:index-maintenance"
```

```output
--8<-- "snippets/quickstart/outputs/fts_quickstart.py.index-maintenance.txt"
```

!!! info
    If you used a custom index name, replace `"text_idx"` with your index name.
    If you did not set `name=...` when creating the FTS index on column `"text"`, the default index name is `"text_idx"`.

If you changed tokenizer settings (such as `with_position`, `base_tokenizer`, stop words, or stemming), rebuild the index with `create_scalar_index(..., replace=True)` so the full dataset is indexed with the new configuration.

### Index Configuration Best Practices

- Enable `with_position` when you need phrase queries, because it stores word positions within documents. For simple term searches, disabling this option can save considerable storage space without impacting performance.

- Keep `lower_case=True` enabled for most applications to ensure case-insensitive search behavior. This provides a better user experience and matches common search expectations, though you can disable it if case sensitivity is important for your use case.

- Enable stemming (`stem=True`) when you want better recall by matching word variations (e.g., "running" matches "run"). Disable stemming if you need exact term matching or if your domain requires precise terminology.

- Consider enabling `remove_stop_words=True` for cleaner search results, especially in content-heavy applications. This removes common words like "the", "and", and "is" from the index, reducing noise and improving relevance. Keep stop words if they carry important meaning in your domain.

### Query Optimization

Using specific, targeted search terms often yields better performance than broad, generic queries. More specific terms reduce the number of potential matches and allow the index to work more efficiently. Consider analyzing your most common search patterns and optimizing your index configuration accordingly.

Combining full-text search with metadata filters can significantly reduce the search space and improve performance. Use structured data filters to narrow down results before applying text search, or vice versa. This approach is particularly effective for large datasets where you can eliminate many irrelevant documents early in the query process.

### Further Reading

For advanced usage instructions with different tokenizers and more technical details on the index training process, including information about the expected memory and disk usage, visit the [full-text index](../format/index/scalar/fts.md) specification.

## Next Steps

Continue with the **[Full-Text Search User Guide](../guide/full_text_search.md)** for index configuration, query types, and index maintenance.
