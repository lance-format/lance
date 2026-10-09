# MinHash LSH Index (Near-Duplicate Search)

!!! warning "Experimental"

    This index is an experimental format feature. Its details message and
    its files may change incompatibly without a separate vote; after such a
    change the index must be rebuilt, the data itself is never rewritten.
    The feature is removed if its stabilization vote does not pass. Design
    discussion: [lance#8820](https://github.com/lance-format/lance/discussions/8820).

The MinHash LSH index finds near duplicates of a text: rows whose content is
essentially the same as the query with a few edits, such as reposts with a
different footer, mirror pages, or copy-pasted records with small changes. It
scores rows by the **estimated Jaccard similarity** of their token shingle
sets, a literal measure of overlap that needs no model. Unlike the Full-Text
Search index, which ranks rows by term relevance, and unlike a vector index,
which finds rows that mean the same thing, this index answers "which rows are
copies of this one".

The index is approximate: candidates are found with locality sensitive hashing
(LSH) over MinHash signatures and ranked by comparing signatures, so the
reported similarity is an estimate, and rows whose similarity is below the
index's threshold are usually not found.

## High-Level Architecture

Every indexed text is turned into a fixed-length **signature** of `num_hashes`
values, and the signature is split into `num_bands` **bands** whose hashes are
the lookup keys. Two rows become candidates for each other when they share at
least one band key; candidates are then ranked by the fraction of equal
signature values.

```
              text
               |  tokenize, shingle, hash
               v
   +-----------------------------------------+
   |   signature: num_hashes 16-bit values   |   1 - (fraction of equal values)
   +-----------------------------------------+   = Jaccard distance
       |  split into num_bands bands, hash each
       v
   band keys ------------> bands table   (candidate lookup: rows sharing a key)
   signature ------------> signature table  (ranking the candidates)
```

The index uses Lance's **Segmented Index** architecture: each segment covers a
disjoint group of fragments and holds its own bands table and signature table.
Scores depend only on the two signatures being compared, never on corpus
statistics, so results from different segments merge exactly.

```
                     +----------------------------------------+
                     |            Lance Dataset               |
                     |   (Disjoint groups of Fragments 0..N)  |
                     +----------------------------------------+
                                         |
                                         v
                     +----------------------------------------+
                     |            Segmented Index             |
                     |  +-----------+ +-----------+ +-------+ |
                     |  | Segment 1 | | Segment 2 | | ...   | |
                     |  | bands +   | | bands +   | |       | |
                     |  | signatures| | signatures| |       | |
                     |  +-----------+ +-----------+ +-------+ |
                     +----------------------------------------+
```

## Index Details

```protobuf
%%% proto.message.MinHashLshIndexDetails %%%
```

| Parameter           | Default                                                          | Effect                                                                                                                                                                                                                     |
|:--------------------|:-----------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `num_hashes`        | 128                                                              | Signature length, 1 to 4096 and a multiple of `num_bands`. More hashes make the similarity estimate more precise (error about `1 / sqrt(num_hashes)`) and the signature table larger (`8 + 2 * num_hashes` bytes per row). |
| `num_bands`         | 16                                                               | Number of bands, 1 to 256. With `num_hashes` it sets the similarity threshold `(1 / num_bands) ^ (num_bands / num_hashes)`. The bands table costs `12 * num_bands` bytes per row.                                         |
| `shingle_size`      | 3                                                                | Tokens per shingle. Shorter shingles tolerate more edits but let unrelated texts that share common phrases look alike; 5 suits long documents, 2 very short texts.                                                         |
| `tokenizer`         | full text search default, without stemming and stop-word removal | The tokenizer, recorded as the Full-Text Search index records it. Stemming and stop-word removal are off by default because merging different words inflates similarity.                                                  |
| `signature_version` | 0                                                                | Version of the signature procedure, including its hash seed. Managed by Lance, not a user parameter.                                                                                                                       |

Reference points: the probability that a row with true Jaccard similarity `J`
becomes a candidate is `1 - (1 - J^r)^b` with `b = num_bands` and
`r = num_hashes / num_bands`.

| `num_hashes` / `num_bands` | Threshold | Found at J = 0.9 | Found at J = 0.7 | Bytes per row |
|:---------------------------|:----------|:-----------------|:-----------------|:--------------|
| 64 / 8                     | 0.77      | 99%              | 38%              | 232           |
| 128 / 16 (default)         | 0.71      | 99.99%           | 61%              | 456           |
| 96 / 16                    | 0.63      | 100%             | 87%              | 392           |

Bands of fewer than six values admit a noticeable share of unrelated rows into
the candidate set on tables of billions of rows and are better left to offline
use.

A query is tokenized from the details alone, so the details must identify the
tokenizer completely: settings the Full-Text Search details cannot record
(custom stop words, document-level text extraction) and tokenizers that load
dictionaries from the deployment (`jieba/*`, `lindera/*`) are rejected. For
CJK text, `ngram` or `icu` gives deterministic shingles.

## Signature Generation

The build side and the query side run the same procedure, so a query is
comparable with every stored row. The exact procedure, its constants and
known-answer vectors are in [Version 0 Reference](#version-0-reference):

1. Tokenize the text with the configured tokenizer.
2. Form shingles of `shingle_size` consecutive tokens. A text with fewer
   tokens than `shingle_size` forms one shingle of all its tokens; a text with
   no tokens (NULL, empty, whitespace only) has no signature, is not indexed,
   and never appears in results.
3. Hash every shingle to 64 bits with a fixed seed.
4. Apply `num_hashes` fixed permutations and keep the minimum of each over
   all shingles of the text.
5. Store each minimum as a 16-bit value. A minimum shrinks as the text grows,
   so its raw high bits would agree between any two long texts; the stored
   value is a multiply-shift hash of the minimum, which keeps the chance that
   two different minima collide negligible against the `1 / num_hashes`
   resolution of the estimate.

Band keys are hashes of the `num_hashes / num_bands` consecutive values of a
band, prefixed with the band number so that the keys of one band are
contiguous.

The estimated Jaccard similarity of two signatures is the fraction of
positions whose values are equal; the reported `_distance` is
`1 - estimated similarity`. Every constant of the procedure is fixed by
`signature_version`, and every segment of an index must carry identical
details, so signatures written by different segments or at query time are
always comparable. Changing a parameter requires rebuilding the index.

## Storage Layout

Each segment consists of two Lance files:

- `signatures.lance` — one row per indexed document: its `_rowid` and its
  signature as a fixed-size list of `num_hashes` 16-bit values. The row number
  is the segment-local document id, so a segment holds at most `2^32`
  documents; larger tables use several segments. Rows are fixed width and
  stored without compression, so one document's signature is a single ranged
  read at a known offset.
- `bands.lance` — one row per (band key, document id) pair, sorted by band key
  and then document id. A bucket is the run of rows sharing a key. The file is
  divided into logical pages of a fixed number of rows recorded in the file
  (4096 as written), and a page table holding the largest key of every page
  is stored in a global buffer of the file and kept in memory when the index
  is open, so the pages of any bucket are known without reading the file.

Both files repeat the serialized index details and the file format version as
schema metadata; a segment is opened only when both files match the details of
the index. The file schemas, metadata keys and the page table encoding are in
[Version 0 Reference](#version-0-reference).

## Segments, Appends and Merging

- **Building**: `create_index` reads the text column, signs every row and
  writes one segment. Distributed builds create one uncommitted segment per
  worker over disjoint fragment sets and commit them together; the commit
  rejects segments whose details differ.
- **Unindexed appends**: fragments appended after the build are not covered
  by any segment. `optimize_indices` adds a segment over them. Until then a
  query still searches those rows, signing them on the fly with the index
  parameters, unless the scan asks for indexed rows only (`fast_search`).
- **Segment merging**: merging segments rebuilds one segment from the stored
  signatures without tokenizing the text again — the signatures of the
  surviving rows are concatenated, band keys are recomputed and the bands
  table is rewritten. A merge whose result would exceed `2^32` documents is
  rejected and the segments stay separate.
- **Deletes and compaction**: deleted rows are filtered at query time; after a
  compaction the stored row ids are remapped like those of every other scalar
  index.

## Query Evaluation

When a search is submitted (`nearest = MinHashQuery(text, column)`):

1. The query text is signed with the parameters recorded in the index, and its
   band keys are computed.
2. In every segment, each band key is looked up through the page table, which
   yields the pages of the key's bucket; the pages of all buckets are fetched
   through the page cache, the missing ones with bounded scattered reads. The
   document ids in the buckets form the candidate set.
3. The candidates' signatures are read — from memory when resident, otherwise
   with scattered reads, or with a sequential scan when the candidates cover a
   large share of the segment — and each candidate's Jaccard distance to the
   query is computed. Rows removed by filters or deletions are skipped.
4. Each segment keeps its `limit` closest rows; rows not covered by any
   segment are scored the same way by the query engine.
5. The per-segment results are merged by distance, and the `limit` closest
   rows are returned with their `_distance`.

## Compatibility

- A reader that does not know this index opens a dataset that has one
  normally: the index is listed under a name derived from the type URL of
  its details and is not used by queries. Creating the index sets no reader
  or writer feature flag.
- A writer that does not create the index changes nothing for readers that
  know it.
- Dropping the feature is dropping the index: its files live only under the
  index's own `_indices/{uuid}/` directory, and no data file, manifest field
  or other index is rewritten.

## Version 0 Reference

This section is normative for `signature_version = 0` and
`minhash_lsh_index_version = 0`: an independent implementation must reproduce
the signatures, band keys and files defined here exactly. All arithmetic is on
unsigned 64-bit integers modulo `2^64`, and every multi-byte value is
little-endian.

### Signature Procedure

1. **Tokenize** the text with the configured tokenizer. A token is the UTF-8
   bytes of the token text the tokenizer emits, in emission order.
2. **Shingle**: a shingle is `shingle_size` consecutive tokens joined with the
   single byte `0x1F`; a text of `n` tokens has `n - shingle_size + 1`
   shingles, one starting at every token position. A text with fewer tokens
   than `shingle_size` but at least one token forms one shingle of all its
   tokens. A text with no tokens (NULL, empty, whitespace only, or every token
   discarded by the tokenizer) has no signature.
3. **Hash** every shingle to 64 bits: `x = XXH64(shingle bytes, seed = 42)`.
4. **Coefficients** come from a SplitMix64 generator whose state `s` starts at
   42. One draw is:

    ```
    s    = s + 0x9E3779B97F4A7C15
    z    = s
    z    = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z    = (z ^ (z >> 27)) * 0x94D049BB133111EB
    draw = z ^ (z >> 31)
    ```

    The draws are consumed in this order: for `i = 0 .. num_hashes - 1`,
    `a_i = draw() | 1` and then `b_i = draw()`; after all pairs, for
    `i = 0 .. num_hashes - 1`, `c_i = draw() | 1`. The coefficients depend on
    `num_hashes` only.
5. **Permute and take the minima**: `m_i = min over all shingles x of
   (a_i * x + b_i)`, comparing the full 64-bit values.
6. **Compress** each minimum to 16 bits: `signature[i] = (c_i * m_i) >> 48`.
7. **Band keys**: the signature is split into `num_bands` bands of
   `r = num_hashes / num_bands` consecutive values; band `j` holds
   `signature[j * r .. (j + 1) * r)` and its key is

    ```
    band_key_j = (j << 56) | (XXH64(band bytes, seed = 42) & 0x00FFFFFFFFFFFFFF)
    ```

    where the band bytes are the band's `r` values, each written as 2
    little-endian bytes, in signature order.

The estimated Jaccard similarity of two signatures is the number of positions
`i` with equal `signature[i]` divided by `num_hashes`; `_distance` is
`1 - estimated similarity`, and rows at equal distance are ordered by row id.

### Known-Answer Vectors

An implementation must reproduce these values, computed with
`num_hashes = 4`, `num_bands = 2`, `shingle_size = 2`, the default tokenizer
and `signature_version = 0`.

Coefficients:

| `i` | `a_i`                | `b_i`                | `c_i`                |
|:----|:---------------------|:---------------------|:---------------------|
| 0   | `0xbdd732262feb6e95` | `0x28efe333b266f103` | `0x5705b8770b3d7dd5` |
| 1   | `0x47526757130f9f53` | `0x581ce1ff0e4ae394` | `0x9e54d738297f77af` |
| 2   | `0x09bc585a244823f3` | `0xde4431fa3c80db06` | `0x3474724a775b19bf` |
| 3   | `0x37e9671c45376d5d` | `0xccf635ee9e9e2fa4` | `0x7e348a0e451650bf` |

Text `"The quick brown fox"` tokenizes to `the`, `quick`, `brown`, `fox`:

| Shingle bytes          | `XXH64`              |
|:-----------------------|:---------------------|
| `the` `0x1F` `quick`   | `0x4c0b0d36203ce55f` |
| `quick` `0x1F` `brown` | `0x09c7ae895d82b184` |
| `brown` `0x1F` `fox`   | `0x68e7f4b1be7b4c1f` |

| `i` | `m_i`                | `signature[i]` |
|:----|:---------------------|:---------------|
| 0   | `0x0959767d77eafad7` | `0xb00f`       |
| 1   | `0x464e4457275cd2a1` | `0x59d6`       |
| 2   | `0x50f41804abaa5973` | `0x511a`       |
| 3   | `0x8944bd01067b09e7` | `0xcd2c`       |

Band keys: `0x00f30f3501d458ea` (band 0), `0x01aec5df28d04915` (band 1).

Text `"Fox"` has one token, fewer than `shingle_size`, so its only shingle is
`fox` with `XXH64` `0x07c0130ab04388d5`; the minima are `0x2bb8ade505081afc`,
`0x0a273c6ff9a78ba3`, `0x802f20573838dc35`, `0x1ef46a8d372c9605`, the
signature is `0x2970 0xd6db 0x65f8 0xadfc`, and the band keys are
`0x003cf6e803312eca` and `0x017c5504fbc83f36`.

The texts `""` and `"  \n"` have no tokens and therefore no signature.

### Signature File Schema

`signatures.lance` holds one row per indexed document. Row `i` is document
`i` of the segment; rows are written in document id order and never
reordered.

```python
pa.schema(
    [
        pa.field("_rowid", pa.uint64(), nullable=False),
        pa.field(
            "signature",
            pa.list_(pa.field("item", pa.uint16(), nullable=False), num_hashes),
            nullable=False,
            metadata={
                "lance-encoding:structural-encoding": "fullzip",
                "lance-encoding:compression": "none",
            },
        ),
    ],
    metadata={
        "minhash_lsh_details": "<hexadecimal serialized MinHashLshIndexDetails>",
        "minhash_lsh_index_version": "0",
    },
)
```

`_rowid` is the row id of the indexed row as the dataset hands it to the
index: the row address, or the stable row id when the dataset has stable row
ids enabled, like every scalar index. `signature` is the signature of the
[Signature Procedure](#signature-procedure); its values are never null (the
Lance schema does not record the nullability of a fixed-size list item, so a
reader treats a null value as corruption of the file). The field metadata of
`signature` selects the full-zip structural encoding without compression, so
every row occupies the same `2 * num_hashes` bytes and a reader fetches the
signature of document `i` by row number as one ranged read.

### Bands File Schema

`bands.lance` holds one row per (band key, document id) pair, sorted by
`band_key` and then `doc_id`, ascending.

```python
pa.schema(
    [
        pa.field(
            "band_key",
            pa.uint64(),
            nullable=False,
            metadata={"lance-encoding:compression": "none"},
        ),
        pa.field(
            "doc_id",
            pa.uint32(),
            nullable=False,
            metadata={"lance-encoding:compression": "none"},
        ),
    ],
    metadata={
        "minhash_lsh_details": "<hexadecimal serialized MinHashLshIndexDetails>",
        "minhash_lsh_index_version": "0",
        "minhash_lsh_num_docs": "<decimal document count>",
        "minhash_lsh_page_rows": "4096",
        "minhash_lsh_page_table_buffer": "<decimal global buffer index>",
    },
)
```

`band_key` is the key of step 7 of the [Signature Procedure](#signature-procedure);
`doc_id` is the segment-local document id, the row number in
`signatures.lance`. Both columns are stored without compression, so any run
of rows is one ranged read per column. The file is divided into logical pages
of `minhash_lsh_page_rows` rows: page `p` holds rows
`[p * page_rows, (p + 1) * page_rows)`, and the last page may be shorter. A
bucket may span any number of pages.

### Schema Metadata

| Key                             | File    | Value                                                                                                                                                                                                          |
|:--------------------------------|:--------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `minhash_lsh_details`           | both    | Lower-case hexadecimal encoding of the serialized `MinHashLshIndexDetails` the segment was built from. A reader decodes it and requires it to describe, field by field, the same details as the index metadata. |
| `minhash_lsh_index_version`     | both    | Decimal file layout version; `0` for the layout described here. Independent of `signature_version`.                                                                                                           |
| `minhash_lsh_num_docs`          | `bands` | Decimal number of documents of the segment; equals the row count of `signatures.lance`.                                                                                                                       |
| `minhash_lsh_page_rows`         | `bands` | Decimal number of rows per logical page; positive. Writers use `4096`.                                                                                                                                        |
| `minhash_lsh_page_table_buffer` | `bands` | Decimal index of the global buffer of `bands.lance` that holds the page table.                                                                                                                                |

### Page Table

The page table is a global buffer of `bands.lance` holding
`ceil(num_rows / page_rows)` little-endian UInt64 values; entry `p` is the
largest `band_key` of page `p`, that is the key of its last row. It is read
when the segment is opened and kept in memory while the index is open. The
bucket of key `K` lies in the pages from the first page whose entry is `>= K`
through the first page whose entry is `> K` (or the last page when there is
none); the pages strictly between hold nothing but that bucket. Both
positions are binary searches over the table, so a bucket's page range is
known without reading the file.

### Opening a Segment

A reader opens a segment in this order and treats a failed check as
corruption of the named file, except where noted:

1. Parse the index details and check the ranges of
   [Index Details](#index-details).
2. Open both files. In each, the schema must match its definition above
   exactly in field names, types, nullability and list size, which must equal
   `num_hashes`; `minhash_lsh_details` must be present, decode, validate and
   equal the index details field by field; and `minhash_lsh_index_version`
   must be present. A version greater than the one the reader implements is
   rejected as unsupported, not as corruption.
3. In `bands.lance`, `minhash_lsh_page_rows` must be positive and
   `minhash_lsh_num_docs` must equal the row count of `signatures.lance`.
4. Read the page table buffer; its length must be a multiple of 8 and its
   entry count must be `ceil(num_rows / page_rows)` for the row count of
   `bands.lance`.

While answering queries, a page that does not decode to non-null `UInt64`
and `UInt32` columns, and a `doc_id` that is not below
`minhash_lsh_num_docs`, are corruption of `bands.lance`; a signature batch
with a null value is corruption of `signatures.lance`.

The fragments a segment covers are recorded in the index metadata of the
dataset, never derived from the stored `_rowid` values (which do not identify
fragments once stable row ids are enabled).
