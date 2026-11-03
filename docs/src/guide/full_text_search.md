---
title: Full-Text Search
description: Build inverted indexes and run BM25-ranked keyword, phrase, fuzzy, and multi-column text searches.
---

# Full-Text Search

Lance full-text search uses an inverted index and BM25 relevance scoring. It supports token matching, boolean queries, exact phrases, fuzzy matching, and searches across multiple text columns.

For ordinary substring filters, use an [N-gram or FM scalar index](scalar_indexing.md#choose-an-index-type). Those indexes accelerate `contains(...)` filters but do not provide BM25 ranking.

## Create a Full-Text Index

Create an `INVERTED` index on a text column with `create_scalar_index`:

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

The default index name is `<column>_idx`. Use a custom `name=...` when a column needs more than one index, such as both an inverted index and an N-gram index.

Use `with_position=True` when phrase queries are required. Positions increase index size but do not affect non-phrase query performance:

```python
dataset.create_scalar_index("body", "INVERTED", with_position=True)
```

Common analyzer options include:

- `base_tokenizer`: controls tokenization, such as `simple`, `whitespace`, `raw`, `icu`, or language-specific tokenizers.
- `lower_case`, `stem`, `remove_stop_words`, `ascii_folding`: control case normalization, stemming, stop-word removal, and accent folding.
- `language`: selects the language for stemming and default stop words.
- `max_token_length`: drops tokens longer than the configured limit.
- `analyzer="code"`: enables the code-analysis profile and its identifier/operator options.

See [Tokenizers](tokenizer.md) for language models, code analysis, and token inspection with `lance.tokenize`.

## Run Keyword Searches

Pass a string to `full_text_query`. Results are ranked by BM25, with higher `_score` values indicating stronger matches:

```python
result = dataset.to_table(full_text_query="train Boston")
```

A string query uses the dataset's searchable text configuration. To select a column and control matching explicitly, use `MatchQuery`:

```python
from lance.query import FullTextOperator, MatchQuery

# Both terms must be present
query = MatchQuery("train Boston", "body", operator=FullTextOperator.AND)
result = dataset.to_table(full_text_query=query)

# Either term may be present
query = MatchQuery("ramen broth", "body", operator=FullTextOperator.OR)
result = dataset.to_table(full_text_query=query)
```

Combine metadata filtering in the same scan:

```python
result = dataset.to_table(
    full_text_query=MatchQuery("train", "body"),
    filter="category = 'travel'",
)
```

## Phrase Queries

Phrase queries require an index built with `with_position=True`:

```python
from lance.query import PhraseQuery

result = dataset.to_table(full_text_query=PhraseQuery("for Boston", "body"))
```

Stop words affect phrase positions. If phrases must contain stop words, build the index with `remove_stop_words=False`; otherwise those words are absent from the indexed positions and the exact phrase may not match.

## Boolean and Fuzzy Queries

Use `BooleanQuery` and `Occur` when some clauses are required and others are forbidden:

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

`MatchQuery` also supports fuzzy matching with Levenshtein edit distance:

```python
query = MatchQuery(
    "ramen",  # matches "ramen" in the document body
    "body",
    fuzziness=1,
    max_expansions=50,
)
result = dataset.to_table(full_text_query=query)
```

Set `fuzziness=None` for automatic fuzziness based on term length. Use `prefix_length` to require an exact prefix before edits are allowed.

## Search Multiple Columns

Use `MultiMatchQuery` when the best score from any column should win:

```python
from lance.query import MultiMatchQuery

query = MultiMatchQuery(
    "columnar guide",
    ["title", "body"],
    boosts=[2.0, 1.0],  # title matches count more
)
result = dataset.to_table(full_text_query=query)
```

Use `CombinedFieldsQuery` (BM25F) when terms may be spread across columns and their corpus statistics should be combined:

```python
from lance.query import CombinedFieldsQuery, FullTextOperator

query = CombinedFieldsQuery(
    "Lance multimodal",
    ["title", "body"],
    operator=FullTextOperator.AND,
)
result = dataset.to_table(full_text_query=query)
```

Columns in a combined-fields query must use the same tokenizer and index configuration.

## Maintain Full-Text Indexes

Appended rows remain searchable: Lance combines indexed search with unindexed scans. As unindexed data grows, periodically add new fragments to the index:

```python
dataset.insert(new_rows)
dataset.optimize.optimize_indices(index_names=["body_idx"])

stats = dataset.stats.index_stats("body_idx")
print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
```

After changing tokenizer or analyzer settings, rebuild the index with `create_scalar_index(..., replace=True)` so the entire dataset uses the new configuration. `optimize_indices()` adds new data to existing indexes; it does not change their analyzer settings.

For large builds, `num_workers` and `memory_limit` control build parallelism and memory. See [Performance Guide](performance.md#indexes) and the [FTS format specification](../format/index/scalar/fts.md) for resource and compatibility details.

## Full-Text and Vector Search

A scanner can combine full-text relevance and vector similarity. See [Combine Full-Text and Vector Search](vector_search.md#combine-full-text-and-vector-search) for the query form and filtering behavior.

For a runnable tutorial, see the [Full-Text Search Quickstart](../quickstart/full-text-search.md).
