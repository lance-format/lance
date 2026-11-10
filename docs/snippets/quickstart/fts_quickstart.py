# Runnable source for docs/src/quickstart/full-text-search.md.
# ruff: noqa: E402, B018, I001
import shutil

for path in ("/tmp/fts.lance", "/tmp/fts_metadata.lance", "/tmp/fts_multicol.lance"):
    shutil.rmtree(path, ignore_errors=True)

# --8<-- [start:imports]
import lance
import pyarrow as pa
# --8<-- [end:imports]

# --8<-- [start:prepare-data]
table = pa.table(
    {
        "id": [1, 2, 3],
        "text": [
            "I left my umbrella on the evening train to Boston",
            "This ramen recipe simmers the broth for three hours with dried mushrooms.",
            "This train is scheduled to leave for Edinburgh at 9:30 in the morning",
        ],
    }
)

ds = lance.write_dataset(table, "/tmp/fts.lance", mode="overwrite")
# --8<-- [end:prepare-data]

# --8<-- [start:schema]
ds = lance.dataset("/tmp/fts.lance")
print(ds.schema)
# --8<-- [end:schema]

# --8<-- [start:build-index]
ds.create_scalar_index(
    column="text",
    index_type="INVERTED",
)
# --8<-- [end:build-index]

# --8<-- [start:advanced-index]
ds.create_scalar_index(
    column="text",
    index_type="INVERTED",
    name="text_idx",  # Optional index name (defaults to "text_idx")
    replace=True,  # Replace existing index with the same name
    with_position=False,  # Set True to enable phrase queries
    base_tokenizer="simple",  # "simple", "whitespace", "raw", "icu", or "ngram"
    language="English",  # Language for stemming and stop words
    max_token_length=40,  # Drop tokens of 40 UTF-8 bytes or more
    lower_case=True,  # Lowercase text before tokenization
    stem=True,  # Stem tokens (language-dependent)
    remove_stop_words=True,  # Remove stop words (language-dependent)
    custom_stop_words=None,  # Optional additional stop words
    ascii_folding=True,  # Fold accents to ASCII (e.g., "é" -> "e")
)
# --8<-- [end:advanced-index]

# --8<-- [start:basic-search]
ds = lance.dataset("/tmp/fts.lance")
ds.to_table(full_text_query="umbrella train").to_pandas()
# --8<-- [end:basic-search]

# --8<-- [start:metadata-filter]
meta_table = pa.table(
    {
        "id": [1, 2, 3],
        "text": [
            "I left my umbrella on the morning train to Boston",
            "This ramen recipe simmers the broth for three hours with dried mushrooms.",
            "This train is scheduled to leave for Edinburgh at 9:30 AM",
        ],
        "category": ["travel", "food", "travel"],
    }
)

meta_ds = lance.write_dataset(meta_table, "/tmp/fts_metadata.lance", mode="overwrite")
meta_ds.create_scalar_index(column="text", index_type="INVERTED")
meta_ds = lance.dataset("/tmp/fts_metadata.lance")

meta_ds.to_table(
    full_text_query="three",
    filter="category = 'food'",
).to_pandas()
# --8<-- [end:metadata-filter]

# --8<-- [start:boolean-and]
from lance.query import FullTextOperator, MatchQuery

and_query = MatchQuery("umbrella train boston", "text", operator=FullTextOperator.AND)
ds.to_table(full_text_query=and_query).to_pandas()
# --8<-- [end:boolean-and]

# --8<-- [start:boolean-or]
or_query = MatchQuery("morning evening", "text", operator=FullTextOperator.OR)
ds.to_table(full_text_query=or_query).to_pandas()
# --8<-- [end:boolean-or]

# --8<-- [start:boolean-mix]
q1 = MatchQuery("morning evening", "text", operator=FullTextOperator.OR)
q2 = MatchQuery("train", "text")
ds.to_table(full_text_query=(q1 & q2)).to_pandas()
# --8<-- [end:boolean-mix]

# --8<-- [start:boolean-not]
from lance.query import BooleanQuery, Occur

q = BooleanQuery(
    [
        (Occur.MUST, MatchQuery("umbrella", "text")),
        (Occur.MUST_NOT, MatchQuery("train", "text")),
    ]
)
ds.to_table(full_text_query=q).to_pandas()
# --8<-- [end:boolean-not]

# --8<-- [start:phrase-search]
ds.create_scalar_index(
    "text",
    "INVERTED",
    with_position=True,
    remove_stop_words=False,
    replace=True,
)
ds = lance.dataset("/tmp/fts.lance")
ds.to_table(full_text_query='"train to boston"').to_pandas()
# --8<-- [end:phrase-search]

# --8<-- [start:ngram-search]
ds.create_scalar_index(column="text", index_type="NGRAM", name="text_ngram")
ds = lance.dataset("/tmp/fts.lance")
ds.to_table(filter="contains(text, 'ramen')").to_pandas()
# --8<-- [end:ngram-search]

# --8<-- [start:ngram-explain]
print(ds.scanner(filter="contains(text, 'train')").explain_plan())
# --8<-- [end:ngram-explain]

# --8<-- [start:fuzzy-explicit]
ds.to_table(
    full_text_query=MatchQuery(
        "rammen",  # Misspelled 'ramen'
        "text",
        fuzziness=1,
        max_expansions=50,
    )
).to_pandas()
# --8<-- [end:fuzzy-explicit]

# --8<-- [start:fuzzy-prefix]
ds.to_table(
    full_text_query=MatchQuery(
        "rammen",
        "text",
        fuzziness=1,
        prefix_length=2,  # "ra" must match exactly
    )
).to_pandas()
# --8<-- [end:fuzzy-prefix]

# --8<-- [start:multi-match]
from lance.query import MultiMatchQuery

multi_table = pa.table(
    {
        "id": [1, 2],
        "title": ["Albino elephant sighting", "Safari packing guide"],
        "body": [
            "Rangers photographed a rare calf near the river.",
            "Visitors hoping to spot an albino elephant should pack binoculars.",
        ],
        "first_name": ["john", "jane"],
        "last_name": ["smith", "doe"],
    }
)
multi_ds = lance.write_dataset(multi_table, "/tmp/fts_multicol.lance", mode="overwrite")
for col in ("title", "body", "first_name", "last_name"):
    multi_ds.create_scalar_index(col, "INVERTED")
multi_ds = lance.dataset("/tmp/fts_multicol.lance")

multi_ds.to_table(
    columns=["id", "title", "_score"],
    full_text_query=MultiMatchQuery(
        "albino elephant",
        ["title", "body"],
        boosts=[2.0, 1.0],  # weight title matches twice as much
    ),
).to_pandas()
# --8<-- [end:multi-match]

# --8<-- [start:combined-fields]
from lance.query import CombinedFieldsQuery

multi_ds.to_table(
    columns=["id", "first_name", "last_name", "_score"],
    full_text_query=CombinedFieldsQuery(
        "john smith",
        ["first_name", "last_name"],
        boosts=[1.0, 1.0],
        operator=FullTextOperator.AND,
    ),
).to_pandas()
# --8<-- [end:combined-fields]

# --8<-- [start:index-maintenance]
new_rows = pa.table(
    {
        "id": [4],
        "text": ["The next train leaves at noon"],
    }
)
ds.insert(new_rows)

ds.optimize.optimize_indices(index_names=["text_idx"])
ds = lance.dataset("/tmp/fts.lance")

stats = ds.stats.index_stats("text_idx")
print(stats["num_unindexed_rows"], stats["num_indexed_rows"])
# --8<-- [end:index-maintenance]
