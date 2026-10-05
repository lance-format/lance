# Full Text Search Index

The full text search (FTS) index (a.k.a. inverted index) provides efficient text search by mapping terms to the documents containing them.
It's designed for high-performance text search with support for various scoring algorithms and phrase queries.

## Index Details

```protobuf
%%% proto.message.InvertedIndexDetails %%%
```

## Storage Layout

The FTS index consists of multiple files storing the token dictionary, document information, and posting lists:

1. `tokens.lance` - Token dictionary mapping tokens to token IDs
2. `docs.lance` - Document metadata including token counts
3. `invert.lance` - Compressed posting lists for each token
4. `metadata.lance` - Index metadata and configuration

An FTS index may contain multiple partitions. Each partition has its own set of token, document, and posting list files, prefixed with the partition ID (e.g. `part_0_tokens.lance`, `part_0_docs.lance`, `part_0_invert.lance`). The `metadata.lance` file lists all partition IDs in the index. At query time, every partition must be searched and the results combined to produce the final ranked output. Fewer partitions generally means better query performance, since each partition requires its own token dictionary lookup and posting list scan. The number of partitions is controlled by the training configuration -- specifically `LANCE_FTS_TARGET_SIZE` determines how large each merged partition can grow (see [Training Process](#training-process) for details).

### Token Dictionary File Schema

| Column      | Type   | Nullable | Description                     |
|-------------|--------|----------|---------------------------------|
| `_token`    | Utf8   | false    | The token string                |
| `_token_id` | UInt32 | false    | Unique identifier for the token |

### Document File Schema

| Column        | Type   | Nullable | Description                      |
|---------------|--------|----------|----------------------------------|
| `_rowid`      | UInt64 | false    | Document row ID                  |
| `_num_tokens` | UInt32 | false    | Number of tokens in the document |

Partitioned `docs.lance` files may include the optional schema metadata key
`total_tokens`. Its decimal `UInt64` value is the sum of `_num_tokens` in that
file. `_num_tokens` remains the canonical per-document data. Readers use the
metadata value to construct exact corpus statistics without scanning the column;
when the key is absent, they compute the sum from `_num_tokens`. Writers produce
the key from the same document table in the same file commit. A present value
that cannot be parsed, or that differs when `_num_tokens` is subsequently
loaded, is file corruption.

### FTS List File Schema

| Column                 | Type                    | Nullable | Description                                                      |
|------------------------|-------------------------|----------|------------------------------------------------------------------|
| `_posting`             | List<LargeBinary>       | false    | Compressed posting lists (delta-encoded row IDs and frequencies) |
| `_max_score`           | Float32                 | false    | Maximum score for the token (for query optimization)             |
| `_length`              | UInt32                  | false    | Number of documents containing the token                         |
| `_compressed_position` | List<List<LargeBinary>> | true     | Optional compressed position lists for phrase queries            |

The posting-list file schema metadata includes `posting_block_size`, the number of documents encoded per compressed posting block. Older indexes that do not have this metadata use the legacy block size `128`.

### Metadata File Schema

The metadata file contains JSON-serialized configuration and partition information:

| Key          | Type          | Description                                              |
|--------------|---------------|----------------------------------------------------------|
| `partitions` | Array<UInt64> | List of partition IDs for distributed index organization |
| `params`     | JSON Object   | Serialized InvertedIndexParams with tokenizer config     |

#### InvertedIndexParams Structure

| Field               | Type    | Default   | Description                                                    |
|---------------------|---------|-----------|----------------------------------------------------------------|
| `base_tokenizer`    | String  | "simple"  | Base tokenizer type (see Tokenizers section)                   |
| `language`          | String  | "English" | Language for stemming and stop words                           |
| `with_position`     | Boolean | false     | Store term positions for phrase queries (increases index size) |
| `max_token_length`  | UInt32? | None      | Maximum token length (tokens longer than this are removed)     |
| `lower_case`        | Boolean | true      | Convert tokens to lowercase                                    |
| `stem`              | Boolean | false     | Apply language-specific stemming                               |
| `remove_stop_words` | Boolean | false     | Remove common stop words for the specified language            |
| `ascii_folding`     | Boolean | true      | Convert accented characters to ASCII equivalents               |
| `min_gram`          | UInt32  | 2         | Minimum n-gram length (only for ngram tokenizer)               |
| `max_gram`          | UInt32  | 15        | Maximum n-gram length (only for ngram tokenizer)               |
| `prefix_only`       | Boolean | false     | Generate only prefix n-grams (only for ngram tokenizer)        |
| `block_size`        | UInt32  | 128       | Documents per compressed posting block. Must be 128 or 256. Missing values from older indexes read as 128. `256` is experimental and may introduce breaking changes. |

## Tokenizers

The full text search index supports multiple tokenizer types for different text processing needs:

### Base Tokenizers

| Tokenizer      | Description                                                               | Use Case               |
|----------------|---------------------------------------------------------------------------|------------------------|
| **simple**     | Splits on whitespace and punctuation, removes non-alphanumeric characters | General text (default) |
| **whitespace** | Splits only on whitespace characters                                      | Preserve punctuation   |
| **raw**        | No tokenization, treats entire text as single token                       | Exact matching         |
| **ngram**      | Breaks text into overlapping character sequences                          | Substring/fuzzy search |
| **icu**        | ICU dictionary-based Unicode word segmentation                            | Mixed-language text    |
| **icu/split**  | ICU segmentation with simple-style delimiter splitting                    | Mixed-language identifiers |
| **jieba/***    | Chinese text tokenizer with word segmentation                             | Chinese text           |
| **lindera/***  | Japanese text tokenizer with morphological analysis                       | Japanese text          |

#### ICU Tokenizer (Mixed-language text)

The ICU tokenizer uses Unicode word boundary rules and dictionary-based segmentation for complex scripts. It is useful for mixed-language text where the default `simple` tokenizer would keep an unspaced CJK span as one large token.

By default, Lance preserves ICU word segments as returned by ICU. Use `base_tokenizer: "icu/split"` to split ICU word segments again on non-alphanumeric delimiters such as underscores and punctuation. For example, `hello_world こんにちは世界` is tokenized as `hello`, `world`, `こんにちは`, and `世界`.

- **Models**: Uses compiled ICU4X segmenter data bundled with Lance
- **Usage**: Specify as `icu`, or `icu/split` to split punctuation-delimited identifiers
- **Features**:
  - Unicode-aware word boundary detection
  - Dictionary-based segmentation for Chinese, Japanese, Khmer, Lao, Myanmar, and Thai
  - No external language model download required

#### Jieba Tokenizer (Chinese)

Jieba is a popular Chinese text segmentation library that uses a dictionary-based approach with statistical methods for word segmentation.

- **Configuration**: Uses a `config.json` file in the model directory
- **Models**: Must be downloaded and placed in the Lance home directory under `jieba/`
- **Usage**: Specify as `jieba/<model_name>` or just `jieba` for the default model
- **Config Structure**:
  ```json
  {
    "main": "path/to/main/dictionary",
    "users": ["path/to/user/dict1", "path/to/user/dict2"]
  }
  ```
- **Features**:
  - Accurate word segmentation for Simplified and Traditional Chinese
  - Support for custom user dictionaries
  - Multiple segmentation modes (precise, full, search engine)

#### Lindera Tokenizer (Japanese)

Lindera is a morphological analysis tokenizer specifically designed for Japanese text. It provides proper word segmentation for Japanese, which doesn't use spaces between words.

- **Configuration**: Uses a `config.yml` file in the model directory
- **Models**: Must be downloaded and placed in the Lance home directory under `lindera/`
- **Usage**: Specify as `lindera/<model_name>` where `<model_name>` is the subdirectory containing the model files
- **Features**:
  - Morphological analysis with part-of-speech tagging
  - Dictionary-based tokenization
  - Support for custom user dictionaries

### Token Filters

Token filters are applied in sequence after the base tokenizer:

| Filter           | Description                                 | Configuration                   |
|------------------|---------------------------------------------|---------------------------------|
| **RemoveLong**   | Removes tokens exceeding max_token_length   | `max_token_length`              |
| **LowerCase**    | Converts tokens to lowercase                | `lower_case` (default: true)    |
| **Stemmer**      | Reduces words to their root form            | `stem`, `language`              |
| **StopWords**    | Removes common words like "the", "is", "at" | `remove_stop_words`, `language` |
| **AsciiFolding** | Converts accented characters to ASCII       | `ascii_folding` (default: true) |

### Supported Languages

For stemming and stop word removal, the following languages are supported:
Arabic, Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Tamil, Turkish

## Document Type
Lance supports 2 kinds of documents: text and json. Different document types have different tokenization rules, and
parse tokens in different format.

### Text Type
Text type includes text and list of text. Tokens are generated by base_tokenizer.

The example below shows how text document is parsed into tokens. 
```text
Tom lives in San Francisco.
```

The tokens are below.
```text
Tom
lives
in
San
Francisco
```

### Json Type
Json is a nested structure, lance breaks down json document into tokens in triplet format `path,type,value`. The valid
types are: str, number, bool, null.

In scenarios where the triplet value is a str, the text value will be further tokenized using the base_tokenizer,
resulting in multiple triplet tokens.

During querying, the Json Tokenizer uses the triplet format instead of the json format, which simplifies the query
syntax.

The example below shows how the json document is tokenized. Assume we have the following json document:
```json
{
  "name": "Lance",
  "legal.age": 30,
  "address": {
    "city": "San Francisco",
    "zip:us": 94102
  }
}
```

After parsing, the document will be tokenized into the following tokens:
```
name,str,Lance
legal.age,number,30
address.city,str,San
address.city,str,Francisco
address.zip:us,number,94102
```

Then we do full text search in triplet format. To search for "San Francisco," we can search with one of the triplets
below:
```
address.city:San Francisco
address.city:San
address.city:Francisco
```

## Training Process

Building an FTS index is a multi-phase pipeline: the source column is scanned, documents are tokenized in parallel, intermediate results are spilled to part files on disk, and the part files are merged into final output partitions.

### Phase 1: Tokenization

The input column is read as a stream of record batches and dispatched to a pool of tokenizer worker tasks. Each worker tokenizes documents independently, accumulating tokens, posting lists, and document metadata in memory.

When a worker's accumulated data reaches the partition size limit or the document count hits `u32::MAX`, it flushes the data to disk as a set of part files (`part_<id>_tokens.lance`, `part_<id>_invert.lance`, `part_<id>_docs.lance`). A single worker may produce multiple part files if it processes enough data.

### Phase 2: Merge

After all workers finish, the part files are merged into output partitions. Part files are streamed with bounded buffering so that not all data needs to be loaded into memory at once. For each part file, the token dictionaries are unified, document sets are concatenated, and posting lists are rewritten with adjusted IDs.

When a merged partition reaches the target size, it is written to the destination store and a new one is started. After all part files are consumed the final partition is flushed, and a `metadata.lance` file is written listing the partition IDs and index parameters.

### Configuration

| Environment Variable       | Default                          | Description                                                                                                           |
|----------------------------|----------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `LANCE_FTS_NUM_SHARDS`     | Number of compute-intensive CPUs | Number of parallel tokenizer worker tasks. Higher values increase indexing throughput but use more memory.             |
| `LANCE_FTS_PARTITION_SIZE` | 256 (MiB)                        | Maximum uncompressed size of a worker's in-memory buffer before it is spilled to a part file.                         |
| `LANCE_FTS_TARGET_SIZE`    | 4096 (MiB)                       | Target uncompressed size for merged output partitions. Fewer, larger partitions improve query performance.             |

### Memory and Performance Considerations

Memory usage is primarily determined by two factors:

- **`LANCE_FTS_NUM_SHARDS`** -- Each worker holds an independent in-memory buffer. Peak memory is roughly `NUM_SHARDS * PARTITION_SIZE` plus the overhead of token dictionaries and posting list structures.
- **`LANCE_FTS_PARTITION_SIZE`** -- Larger values reduce the number of part files and make the merge phase cheaper. Smaller values reduce per-worker memory at the cost of more part files.

Merge phase memory is bounded by the streaming approach: part files are loaded one at a time with a small concurrency buffer. The merged partition's in-memory size is bounded by `LANCE_FTS_TARGET_SIZE`.

Building an FTS index requires temporary disk space to store the part files generated during tokenization. The amount of temporary space depends heavily on whether position information is enabled. An index with `with_position: true` stores the position of every token occurrence in every document, which can easily require 10x the size of the original column or more in temporary disk space. An index without positions tends to be smaller than the original column and will typically need less than 2x the size of the column in total disk space.

Performance tips:

- Larger `LANCE_FTS_TARGET_SIZE` produces fewer output partitions, which is beneficial for query performance because queries must scan every partition's token dictionary. When memory allows, prefer fewer, larger partitions.
- `with_position: true` significantly increases index size because term positions are stored for every occurrence. Only enable it when phrase queries are needed.
- The ngram tokenizer generates many more tokens per document than word-level tokenizers, so expect larger index sizes and higher memory usage.

### Distributed Training

The FTS index supports distributed training where different worker nodes each index a subset of the data and the results are assembled afterward.

1. Each distributed worker is assigned a **fragment mask** (`(fragment_id as u64) << 32`) that is OR'd into the partition IDs it generates, ensuring globally unique IDs across workers.
2. Workers set `skip_merge: true` so they write their part files directly without running the merge phase.
3. Instead of a single `metadata.lance`, each worker writes per-partition metadata files named `part_<id>_metadata.lance`.
4. After all workers finish, a coordinator merges the metadata files: it collects all partition IDs, remaps them to a sequential range starting from 0 (renaming the corresponding data files), and writes the final unified `metadata.lance`.

This allows each worker to operate independently during the tokenization phase. Only the final metadata merge requires a single-node step, and it is lightweight since it only renames files and writes a small metadata file.

## Accelerated Queries

Lance SDKs provide dedicated full text search APIs to leverage the FTS index capabilities. 
These APIs support complex query types beyond simple token matching, 
enabling sophisticated text search operations.
Here are the query types enabled by the FTS index:

| Query Type          | Description                                                                              | Example Usage                                        | Result Type |
|---------------------|------------------------------------------------------------------------------------------|------------------------------------------------------|-------------|
| **contains_tokens** | Basic token-based search (UDF) with BM25 scoring and automatic result ranking            | SQL: `contains_tokens(column, 'search terms')`       | AtMost      |
| **match**           | Match query with configurable AND/OR operators and relevance scoring                     | `{"match": {"query": "text", "operator": "and/or"}}` | AtMost      |
| **phrase**          | Exact phrase matching with position information (requires `with_position: true`)         | `{"phrase": {"query": "exact phrase"}}`              | AtMost      |
| **boolean**         | Complex boolean queries with must/should/must_not clauses for sophisticated search logic | `{"boolean": {"must": [...], "should": [...]}}`      | AtMost      |
| **multi_match**     | Search across multiple fields simultaneously with unified scoring                        | `{"multi_match": [{"field1": "query"}, ...]}`        | AtMost      |
| **boost**           | Boost relevance scores for specific terms or queries by a configurable factor            | `{"boost": {"query": {...}, "factor": 2.0}}`         | AtMost      |
