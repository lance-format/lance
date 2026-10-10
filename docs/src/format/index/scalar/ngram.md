# N-gram Index

N-gram indices break text into overlapping sequences (trigrams) for efficient substring matching.
They provide fast text search by indexing all 3-character sequences in the text after
applying ASCII folding and lowercasing.

## Index Details

```protobuf
%%% proto.message.NGramIndexDetails %%%
```

## Storage Layout

The N-gram index stores tokenized text as trigrams with their posting lists:

1. `ngram_postings.lance` - Trigram tokens and their posting lists

### File Schema

| Column         | Type   | Nullable | Description                                       |
|----------------|--------|----------|---------------------------------------------------|
| `tokens`       | UInt32 | true     | Hashed trigram token                              |
| `posting_list` | Binary | false    | Compressed bitmap of the physical row addresses (`fragment_id << 32 \| offset`) containing the token. A segment persisted before format version 1 (`IndexMetadata::index_version`) stores row ids here instead. |

## Accelerated Queries

The N-gram index provides inexact results for the following query types:

| Query Type     | Description              | Operation                                             | Result Type |
|----------------|--------------------------|-------------------------------------------------------|-------------|
| **contains**   | Substring search in text | Finds all trigrams in query, intersects posting lists | AtMost      |
| **regexp_like** / **regexp_match** | Regular-expression match | Derives a necessary trigram condition from the pattern (AND of intersections, OR of unions), then rechecks the true regex | AtMost |
| **LIKE** (infix) | Wildcard match such as `%foo%bar%` | Uses the literal segments of the pattern as a trigram condition, then rechecks the LIKE | AtMost |

Patterns from which no trigram can be derived - for example `a.b`, `.*`,
case-insensitive matches, or literal runs shorter than three characters - fall
back to rechecking every row. This is always correct, just not accelerated.
