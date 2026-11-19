# BTree Index

The BTree index is a two-level structure that provides efficient range queries and sorted access. 
It strikes a balance between an expensive memory structure containing all values 
and an expensive disk structure that can't be efficiently searched.

The upper layers of the BTree are designed to be cached in memory and stored in a 
BTree structure (`page_lookup.lance`), while the leaves are searched using sub-indices 
(`page_data.lance`, currently just a flat file). 
This design enables efficient memory usage - for example, with 1 billion values, 
the index can store 256K leaves of size 4K each, requiring only a few MiB of memory 
(depending on data type) for the BTree metadata while narrowing any search to just 4K values.

## Index Details

```protobuf
%%% proto.message.BTreeIndexDetails %%%
```

Index version 2 is a superset of version 1. A version 2 segment without
`key_layout` has exactly the version 1 layout and meaning. A segment sets
`key_layout` when its pages store encoded keys instead of raw values. See
[Key Layout](#key-layout). Under the independent covering-field contract, a
segment at any index version may also store the values of the columns it
declares in `covering_fields`. See [Carried Columns](#carried-columns).

## Storage Layout

The BTree index consists of two files:

1. `page_lookup.lance` - The BTree structure mapping value ranges to page numbers
2. `page_data.lance` - The actual sub-indices (flat file) containing sorted values and row addresses

### Page Lookup File Schema (BTree Structure)

| Column       | Type       | Nullable | Description                                              |
|--------------|------------|----------|----------------------------------------------------------|
| `min`        | {DataType} | true     | Minimum value in the page (forms BTree keys)             |
| `max`        | {DataType} | true     | Maximum value in the page (for range pruning)            |
| `null_count` | UInt32     | false    | Number of null values in the page                        |
| `page_idx`   | UInt32     | false    | Page number pointing to the sub-index in page_data.lance |

### Schema Metadata

| Key | Type | Description |
|-----|------|-------------|
| `batch_size` | String | Number of rows per page (default: "4096") |

### Page Data File Schema (Sub-indices)

| Column   | Type       | Nullable | Description                                       |
|----------|------------|----------|---------------------------------------------------|
| `values` | {DataType} | true     | Sorted values from the indexed column (flat file) |
| `ids`    | UInt64     | false    | Physical row address (`fragment_id << 32 \| offset`) corresponding to each value. A segment persisted before format version 1 (`IndexMetadata::index_version`) stores a row id here instead. |

A segment that carries columns stores them after `ids`. See
[Carried Page Layout](#carried-page-layout). A reader may serve the keys of a
segment from `values`. See [Serving Keys](#serving-keys).

## Stable Row ID Reverse Lookup

An ordinary BTree may index the logical `_rowid` column of a table, given the following rules:

* Stable Row IDs **must** be enabled on the table.
* The [key fields](#key-fields) **must** be the single field ID `-3`. A
  segment that declares no covering fields therefore has `fields` equal to
  `[-3]`.
* `_rowid` **must** be represented as a non-nullable `UInt64` field.
* The index **must** use the address-domain format and be at least version 1.
* The index **must not** set `key_layout`.

The segment may carry columns as [Carried Columns](#carried-columns) describes.
`covering_fields` never names `-3`, because every covering field is a
top-level field of the dataset schema (see
[Loading an index](../index.md#loading-an-index)). A reader serves the
`_rowid` key from `values` as [Serving Keys](#serving-keys) describes, with the
non-nullable `UInt64` above as its storage type.

The logical field need not appear in the user schema. Its inline and spilled
representations are defined in
[Row ID and Lineage](../../table/row_id_lineage.md#inline-and-spilled-storage).

## Accelerated Queries

The BTree index provides exact results for the following query types:

| Query Type | Description               | Operation                                                                   |
|------------|---------------------------|-----------------------------------------------------------------------------|
| **Equals** | `column = value`          | BTree lookup to find relevant pages, then search within sub-indices         |
| **Range**  | `column BETWEEN a AND b`  | BTree traversal for pages overlapping the range, then search each sub-index |
| **IsIn**   | `column IN (v1, v2, ...)` | Multiple BTree lookups, union results from all matching sub-indices         |
| **IsNull** | `column IS NULL`          | Returns rows from all pages where null_count > 0                            |

## Key Layout

!!! warning "Experimental"

    Key layouts are unstable. The `key_layout` message, the `arrow_row`
    encoding, the `lexicographic` ordering, and the page layout in this
    section may change incompatibly without a separate vote. After such a
    change, an index with a `key_layout` must be rebuilt. The table data is
    never rewritten. The feature is removed if its stabilization vote does not
    pass.

A BTree index with a `key_layout` has one or more keys and stores them in the
[encoded page layout](#encoded-page-layout). With two or more keys it is a
composite index. With one key it is a single-column index whose direction and
null placement are set explicitly.

The index sorts rows lexicographically by their tuple of key values: compare
the first key, and if the values are equal, compare the second key, and so on.
Each comparison uses that key's direction and null placement.

Each page key is a single byte string that encodes the whole tuple, such that
comparing two encoded keys as bytes gives the same result as comparing their
tuples. The page lookup above therefore works unchanged on encoded keys.

Because nulls are encoded as part of the tuple, the null handling in
[Accelerated Queries](#accelerated-queries) does not apply to an index with a
`key_layout`.

### Key Layout Metadata

A segment with `key_layout` sets these `IndexMetadata` fields:

- `index_version` is 2.
- The [key fields](#key-fields) list the key columns in key order. There is
  at least one. Each names a field of the dataset schema, and no field appears
  twice among them.
- `index_details` is a `BTreeIndexDetails` whose `key_layout` describes the keys.

Under either
[covering-field contract](../index.md#covering-field-contracts), the key fields
are the first entries of `fields` (see [Key Fields](#key-fields)).

`key_layout.keys[i]` gives the direction and the null placement of the key
column `fields[i]`. The dataset schema gives the type of each key column, so
`key_layout` does not repeat it. Each key sets its direction and its null
placement explicitly. The null placement does not follow from the direction,
so an ascending or a descending key may put its nulls first or last. In the
protobuf, a false `descending` means ascending and a false `nulls_first` means
nulls last.

`key_layout.encoding` sets how a tuple becomes bytes, and `key_layout.ordering`
sets how those bytes order tuples. Each holds a `oneof` with one message per
kind, and the parameters of a kind are fields of its message. This page defines
the `arrow_row` encoding with `version` 1, and the `lexicographic` ordering,
which has no parameters.

```protobuf
%%% proto.message.BTreeKeyLayout %%%
```

### Encoded Page Layout

An index with a `key_layout` has the same two files and the same schema
metadata as an index without one. Only the type of the value columns changes.
`page_data.lance` has one row for each indexed row:

```python
pa.schema([
    pa.field("values", pa.binary(), nullable=False),
    pa.field("ids", pa.uint64(), nullable=False),
])
```

`values` holds the encoded key tuple of the row. It is never null, even when
some keys in the tuple are null. `ids` holds the row address, as in version 1.
Rows sort by `values` compared as unsigned byte strings, and rows with equal
`values` sort by `ids` in ascending order. Two byte strings compare at their
first differing byte, so the one-byte string `0xFF` sorts after the two-byte
string `0x0000`. A byte string sorts before every longer byte string that
starts with it, so `0x01` sorts before `0x0100`.

`page_lookup.lance` has one row for each page:

```python
pa.schema([
    pa.field("min", pa.binary(), nullable=False),
    pa.field("max", pa.binary(), nullable=False),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("page_idx", pa.uint32(), nullable=False),
])
```

`min` and `max` are the first and the last `values` of the page. `null_count`
is always 0, because `values` is never null. The column stays so that
`page_lookup.lance` has the same columns with or without a `key_layout`.

### Encoding `arrow_row` Version 1

`arrow_row` version 1 encodes a tuple as the encodings of its keys,
concatenated in key order, with no header, separator, or length. A reader splits
an encoded tuple into keys from left to right, because the type and direction
of each key determine where its encoding ends.

The encoding admits these key types: Boolean, Int8, Int16, Int32, Int64,
UInt8, UInt16, UInt32, UInt64, Float32, Float64, Date32, Date64, Timestamp with
any unit and with or without a time zone, Utf8, LargeUtf8, Binary, and
LargeBinary. The encoding admits no other type, including nested types such as
Struct and List and the Decimal types, and a later encoding version may admit
them.

The encoding of a key depends on its type, on `descending`, and on
`nulls_first`. The null sentinel of a key is the byte `0x00` when `nulls_first`
is true and the byte `0xFF` when it is false. Every non-null encoding below
starts with a byte strictly between `0x00` and `0xFF`. A fixed-width value
starts with `0x01` in either direction. A variable-width value starts with
`0x01` or `0x02`, or with their inverses `0xFE` or `0xFD` when `descending` is
true. The first byte alone therefore puts a null before or after every non-null
value of the key, however many bytes the values take.

#### Fixed-Width Types

A key of a fixed-width type encodes in 1 + W bytes:

- A non-null value is the byte `0x01`, then the W value bytes from the table
  below. When `descending` is true, every value byte is inverted, but the
  `0x01` is not.
- A null is the null sentinel, then W bytes of `0x00`. These bytes are never
  inverted.

| Type                          | W          | Value bytes                                                                                         |
|-------------------------------|------------|-----------------------------------------------------------------------------------------------------|
| Boolean                       | 1          | `0x00` for false and `0x01` for true                                                                |
| UInt8, UInt16, UInt32, UInt64 | 1, 2, 4, 8 | The value, big-endian                                                                               |
| Int8, Int16, Int32, Int64     | 1, 2, 4, 8 | The two's-complement value, big-endian, with the most significant bit inverted                      |
| Float32, Float64              | 4, 8       | The IEEE 754 bit pattern, big-endian. If the sign bit is 1, every bit is inverted. Otherwise only the sign bit is inverted. |
| Date32                        | 4          | The days since the UNIX epoch, encoded as an Int32                                                  |
| Date64                        | 8          | The milliseconds since the UNIX epoch, encoded as an Int64                                          |
| Timestamp                     | 8          | The stored value, encoded as an Int64. The unit and the time zone do not change the bytes.          |

#### Variable-Width Types

A Utf8 or LargeUtf8 key encodes the UTF-8 bytes of its value. A Binary or
LargeBinary key encodes the bytes of its value. A type and its large variant
give the same bytes.

- A null is the null sentinel alone.
- An empty value is the byte `0x01`.
- Any other value is the byte `0x02`, then the value split into blocks. The
  first four blocks hold 8 bytes each, and every later block holds 32 bytes. The
  last block is padded with `0x00` to its full size. Each block is followed by
  one byte, which is `0xFF` if more of the value follows the block, and
  otherwise the number of value bytes in the block.

When `descending` is true, every byte of a non-null encoding is inverted. That
includes the leading `0x01` or `0x02`, the padding, and the byte after each
block. A null is never inverted.

### Ordering `lexicographic`

`lexicographic` compares two tuples one key at a time, in key order.
The first key whose values differ decides the order. Two values of one key
compare as follows:

1. If exactly one value is null, the null sorts first when `nulls_first` is
   true and last when it is false. Two nulls are equal.
2. Otherwise the values compare in the order of their type, reversed when
   `descending` is true.

The order of each type is:

- Integers, dates, and timestamps order by their stored value. The time zone
  of a timestamp does not change its order.
- Boolean orders false before true.
- Float32 and Float64 order by the IEEE 754 `totalOrder` predicate. Negative
  NaNs sort first, then negative infinity, the negative numbers, -0.0, +0.0, the
  positive numbers, positive infinity, and the positive NaNs. NaNs with
  different bit patterns are different values.
- Strings and binary values order by their bytes, compared as unsigned byte
  strings.

Under `arrow_row` version 1, comparing two encoded tuples as unsigned byte
strings gives the same order as `lexicographic` gives the tuples. Two
tuples are equal exactly when their encodings are equal. A reader therefore
orders and searches pages by comparing bytes, and never decodes a key to do so.

### Encoding Examples

Each row below gives the `arrow_row` version 1 encoding of one value, in
hexadecimal. Spaces separate the leading byte, the value bytes, and the byte
after each block. They are not part of the encoding.

| Key type          | Direction and nulls     | Value                         | Encoding                                     |
|-------------------|-------------------------|-------------------------------|----------------------------------------------|
| Boolean           | ascending, nulls first  | null                          | `00 00`                                      |
| Boolean           | ascending, nulls first  | false                         | `01 00`                                      |
| Boolean           | ascending, nulls first  | true                          | `01 01`                                      |
| Boolean           | descending, nulls last  | null                          | `ff 00`                                      |
| Boolean           | descending, nulls last  | false                         | `01 ff`                                      |
| Boolean           | descending, nulls last  | true                          | `01 fe`                                      |
| Int8              | ascending, nulls first  | -2                            | `01 7e`                                      |
| Int16             | ascending, nulls first  | -2                            | `01 7ffe`                                    |
| Int32             | ascending, nulls first  | null                          | `00 00000000`                                |
| Int32             | ascending, nulls first  | MIN                           | `01 00000000`                                |
| Int32             | ascending, nulls first  | -2                            | `01 7ffffffe`                                |
| Int32             | ascending, nulls first  | MAX                           | `01 ffffffff`                                |
| Int32             | ascending, nulls last   | null                          | `ff 00000000`                                |
| Int32             | descending, nulls first | -2                            | `01 80000001`                                |
| Int32             | descending, nulls last  | null                          | `ff 00000000`                                |
| Int64             | ascending, nulls first  | -2                            | `01 7ffffffffffffffe`                        |
| UInt8             | ascending, nulls first  | 128                           | `01 80`                                      |
| UInt16            | ascending, nulls first  | 32768                         | `01 8000`                                    |
| UInt32            | ascending, nulls first  | 2147483648                    | `01 80000000`                                |
| UInt64            | ascending, nulls first  | 9223372036854775808           | `01 8000000000000000`                        |
| Float32           | ascending, nulls first  | -MAX (bits `ff7fffff`)        | `01 00800000`                                |
| Float32           | ascending, nulls first  | -0.0 (bits `80000000`)        | `01 7fffffff`                                |
| Float32           | ascending, nulls first  | +0.0 (bits `00000000`)        | `01 80000000`                                |
| Float32           | ascending, nulls first  | 1.5 (bits `3fc00000`)         | `01 bfc00000`                                |
| Float32           | ascending, nulls first  | -NaN (bits `ffc00000`)        | `01 003fffff`                                |
| Float32           | ascending, nulls first  | +NaN (bits `7fc00000`)        | `01 ffc00000`                                |
| Float64           | ascending, nulls first  | -inf (bits `fff0000000000000`) | `01 000fffffffffffff`                       |
| Float64           | ascending, nulls first  | 1.5 (bits `3ff8000000000000`) | `01 bff8000000000000`                        |
| Float64           | ascending, nulls first  | +NaN (bits `7ff8000000000001`) | `01 fff8000000000001`                       |
| Date32            | ascending, nulls first  | 0                             | `01 80000000`                                |
| Date64            | ascending, nulls first  | 0                             | `01 8000000000000000`                        |
| Timestamp(ns)     | ascending, nulls first  | MIN                           | `01 0000000000000000`                        |
| Timestamp(ns, UTC) | ascending, nulls first | 0                             | `01 8000000000000000`                        |
| Utf8              | ascending, nulls first  | null                          | `00`                                         |
| Utf8              | ascending, nulls first  | `""`                          | `01`                                         |
| Utf8              | ascending, nulls first  | `"ab"`                        | `02 6162000000000000 02`                     |
| Utf8              | ascending, nulls first  | `"\u{10FFFF}"`                | `02 f48fbfbf00000000 04`                     |
| Utf8              | ascending, nulls first  | `"0123456789abcdefghijklmnopqrstuv"` | `02 3031323334353637 ff 3839616263646566 ff 6768696a6b6c6d6e ff 6f70717273747576 08` |
| Utf8              | ascending, nulls first  | `"0123456789abcdefghijklmnopqrstuvw"` | `02 3031323334353637 ff 3839616263646566 ff 6768696a6b6c6d6e ff 6f70717273747576 ff 7700000000000000000000000000000000000000000000000000000000000000 01` |
| Utf8              | ascending, nulls last   | null                          | `ff`                                         |
| Utf8              | descending, nulls first | null                          | `00`                                         |
| Utf8              | descending, nulls first | `""`                          | `fe`                                         |
| Utf8              | descending, nulls first | `"ab"`                        | `fd 9e9dffffffffffff fd`                     |
| Utf8              | descending, nulls first | `"012345678"`                 | `fd cfcecdcccbcac9c8 00 c7ffffffffffffff fe` |
| Utf8              | descending, nulls last  | null                          | `ff`                                         |
| LargeUtf8         | ascending, nulls first  | `"ab"`                        | `02 6162000000000000 02`                     |
| Binary            | ascending, nulls first  | `[]`                          | `01`                                         |
| Binary            | ascending, nulls first  | `[00 ff]`                     | `02 00ff000000000000 02`                     |
| Binary            | ascending, nulls first  | `[ff]`                        | `02 ff00000000000000 01`                     |
| LargeBinary       | ascending, nulls first  | `[00 ff]`                     | `02 00ff000000000000 02`                     |

A tuple concatenates the encodings of its keys. With an Int32 key that is
ascending with nulls first, followed by a Utf8 key that is descending with nulls
last:

| Tuple          | Encoding                              |
|----------------|---------------------------------------|
| `(-1, "ab")`   | `01 7fffffff` `fd 9e9dffffffffffff fd` |
| `(null, null)` | `00 00000000` `ff`                    |

### Key Layout Reader Rules

A reader applies these rules from the manifest alone, before it opens an index
file. It picks the page layout of a BTree segment from whether `key_layout` is
present, never from the number of entries in `fields`. It treats a segment with
a `key_layout` as unusable when any of these is true:

- `index_version` is below 2.
- The [key fields](#key-fields) are empty, name a field twice, or name a field
  that is not in the dataset schema.
- `key_layout.keys` and the key fields have different lengths.
- The reader does not support the `encoding` or the `ordering`, including their
  parameters such as the `arrow_row` `version`. A missing `encoding` or
  `ordering`, or one whose `oneof` is unset, is not supported.
- The encoding does not admit the type of a key column.
- The manifest's writer feature flags lack `FLAG_BTREE_KEY_LAYOUT`.

An engine skips an unusable index and scans the fragments that the index
covers, as it does for any index segment it cannot use (see
[Index Compatibility](../index.md#index-compatibility)). The
[Key Layout Writer Rules](#key-layout-writer-rules) say how a writer maintains
an unusable index.

### Key Layout Writer Rules

These rules apply to every writer that supports key layouts. Released versions
do not support key layouts.
[Compatibility With Released Versions](#compatibility-with-released-versions)
explains how the index version and `FLAG_BTREE_KEY_LAYOUT` fence them.

By default, a writer builds a single-column index without a `key_layout`.

A writer must set `FLAG_BTREE_KEY_LAYOUT` (`2^20`) in the writer
[feature flags](../../table/versioning.md#current-feature-flags) of every
manifest it publishes, including by clone and restore, exactly when the
manifest's index section has a segment with a `key_layout`. It must not set the
bit in the reader feature flags.

A writer must not carry a segment with a `key_layout` out of a manifest whose
writer feature flags lack the bit. This rule covers the parent manifest that a
commit builds on and the version that a restore, clone, or branch copies. The
writer must drop such a segment before it computes the bit.

A writer may keep a segment with a `key_layout` that the
[Key Layout Reader Rules](#key-layout-reader-rules) make unusable to it. It must
then maintain the segment as follows:

- It must not add rows or fragments to the segment.
- It must remove fragments from the segment's `fragment_bitmap` when an update
  requires it (see
  [Creating and Updating Index Segments](../index.md#creating-and-updating-index-segments)).
- Its compaction may rewrite the fragments that the segment covers. The commit
  that rewrites them must remove them from the segment's `fragment_bitmap`, as
  an update does. Readers then scan the new fragments (see
  [Compaction and remapping](../index.md#compaction-and-remapping)).
- When compaction defers the remap to a
  [fragment reuse index](../system/frag_reuse.md), the commit must still remove
  the rewritten fragments from the segment's `fragment_bitmap`. The segment
  then covers none of the fragments that the reuse index maps them to, so
  readers scan those fragments.

### Encoding Stability

The bytes of `arrow_row` version 1 are the bytes that the `RowConverter` of the
Arrow Rust crate `arrow-row` 58.4.0 writes for the admitted types. Arrow does
not promise to keep those bytes across releases, so this page, not the crate,
defines version 1. If a later `arrow-row` release writes different bytes for an
admitted type, Lance keeps writing version 1 bytes for that type by vendoring
the version 1 encoder. An Arrow upgrade never changes version 1. A different
byte layout is a new encoding version, `arrow_row` version 2.

### Compatibility With Released Versions

Index version 2 fences released readers. Releases from v10.0.0 on do not use a
BTree index of version 2, so they skip an index with a `key_layout` and answer
queries from the table data.

`FLAG_BTREE_KEY_LAYOUT` fences released writers. Without it, an update by
v12.0.0 or v13.0.0 on a table with stable row IDs would add the fragment that
receives the rewritten rows to the index's `fragment_bitmap` without indexing
those rows, because these releases do not check the index version before they
update. Releases v12.0.0 through v14.0.0 refuse to write a table that sets the
bit. Releases v10.0.0 and v11.0.0 check writer feature flags only when they
insert, and their restore, shallow clone, and branch creation clear the bit but
keep the index. For that reason, a reader does not use an index with a
`key_layout` from a manifest that lacks the bit, and a writer drops such an
index (see [Key Layout Reader Rules](#key-layout-reader-rules) and
[Key Layout Writer Rules](#key-layout-writer-rules)).

## Carried Columns

!!! warning "Experimental"

    Carried columns are unstable. The carried page layout and the rules in
    this section may change incompatibly without a separate vote. After such a
    change, a segment with carried columns must be rebuilt or must withdraw
    them. The table data is never rewritten. The feature is removed if its
    stabilization vote does not pass.

A BTree segment may store the values of the columns that its `IndexMetadata`
declares in `covering_fields`, so that a query that projects only those
columns and the segment's keys can read them from the index instead of taking
them from the table.
Any BTree segment may carry columns, at any index version. That includes a
single-column segment without a `key_layout`, and a single-column or composite
segment with one. Only the independent
[covering-field contract](../index.md#covering-field-contracts) supports
carried columns. These rules apply
[Serving carried columns](../index.md#serving-carried-columns) to the BTree.
For a BTree segment, the schema of `page_data.lance` is the storage schema that
those rules call authoritative.

### Key Fields

The manifest's covering-field contract decides which entries of `fields` are
the keys of a segment, as
[Covering-field contracts](../index.md#covering-field-contracts) defines it:

- Under the independent contract, every entry of `fields` is a key.
- Under the legacy contract, the keys are
  `fields[..len(fields) - len(covering_fields)]`.

The keys are in key order. A segment without a `key_layout` has exactly one
key. A segment with a `key_layout` has one key for each entry of
`key_layout.keys`.

### Carried Page Layout

A writer stores carried columns in a BTree segment only when the manifest that
it commits the segment to sets `FLAG_COVERED_INDEX_METADATA` and
`FLAG_INDEPENDENT_COVERING_FIELDS` in its reader and writer feature flags. If a
commit retries against a manifest that lacks them, the writer either
activates the independent contract in the manifest that it builds, as
[Activating independent declarations](../index.md#activating-independent-declarations)
describes, or does not commit the segment. It stores carried columns only in
`page_data.lance`, never in `part_*_page_data.lance` page files.

A segment that carries columns adds one column for each carried field to
`page_data.lance`, after `values` and `ids`. Without a `key_layout`, the schema
is:

```python
pa.schema(
    [
        pa.field("values", value_type),
        pa.field("ids", pa.uint64(), nullable=False),
        *[
            pa.field(field.name, field.type, nullable=field.nullable)
            for field in carried_fields
        ],
    ],
    metadata={"covering_field_ids": "<field id>,<field id>,..."},
)
```

With a `key_layout`, the schema is:

```python
pa.schema(
    [
        pa.field("values", pa.binary(), nullable=False),
        pa.field("ids", pa.uint64(), nullable=False),
        *[
            pa.field(field.name, field.type, nullable=field.nullable)
            for field in carried_fields
        ],
    ],
    metadata={"covering_field_ids": "<field id>,<field id>,..."},
)
```

`values` and `ids` are as in the [raw](#page-data-file-schema-sub-indices) or
the [encoded](#encoded-page-layout) page layout, and `value_type` is the
`{DataType}` of the raw layout. `carried_fields` lists the field of each
carried column, in column order, as the dataset schema has it when the writer
writes the file. Carried columns change neither `page_lookup.lance` nor the
rows of `page_data.lance`, their order, or the pages that they form. They do
not change the Lance file version of `page_data.lance` either, and a reader
accepts every file version that it accepts for the same page file without
carried columns. Each page that `page_lookup.lance` references holds 1 to
`batch_size` rows. An empty index has no rows in either file. Its
`page_data.lance` may still have carried columns, and `covering_field_ids` then
lists them as in any other file.

- Each carried column holds a field that `covering_fields` declares, which is
  a top-level field as [Loading an index](../index.md#loading-an-index)
  requires. The writer gives the column the name, the Arrow type, and the
  nullability that the field has in the dataset schema when the writer writes
  the file. A reader ignores the column's top-level nullability when it binds
  the column (see [Carried Column Reader Rules](#carried-column-reader-rules)),
  so a later nullability change of the field keeps the binding. The
  nullability of nested children is part of the storage type.
- Row `i` of a carried column holds the field's value for the row that
  `ids[i]` identifies. A null value is a null in the table. A writer never
  writes a null for a value that it does not have.
- A writer does not store a [blob](../../table/schema.md#blob-type) field, or a
  field that contains one, as a carried column. The data files of a blob
  field hold descriptors instead of its values (see
  [Blob v2](../../table/blob.md)).
- A writer does not store a field named `values` or `ids` as a carried column.
- `covering_field_ids` lists the field id of each carried column in the order
  of the columns in the schema, separated by single commas. Each id is a
  signed 32-bit integer in canonical decimal form: `0`, or a digit from `1` to
  `9` followed by any digits, with a leading `-` for a negative id. It has no
  `+`, no leading zero, and no whitespace. The key is present exactly when the
  file has at least one carried column.

The file may hold fewer carried columns than `covering_fields` declares. A
declared field without a carried column is a legal state, and a reader takes its
values from the table.

### Serving Keys

`values` holds every key of a segment for every row, in the segment's layout:
the raw value without a `key_layout`, and the encoded tuple with one. A reader
may therefore serve a key field of any BTree segment from `values`, whether or
not `covering_fields` declares it. That holds under either
[covering-field contract](../index.md#covering-field-contracts) and for a
segment without covering fields. The
[Carried Column Reader Rules](#carried-column-reader-rules) name the segments
that serve no value at all.

The reader serves a field in `covering_fields` that is not a key from its
carried column when one binds (see
[Carried Column Reader Rules](#carried-column-reader-rules)), and takes it from
the table otherwise. A key that `covering_fields` also declares may have its
own carried column too. That copy is optional. The reader serves the key from
the copy when the segment holds one that binds, and from `values` otherwise.
Under the legacy contract no carried column binds. A segment that omits or
withdraws the copy, or that cannot store it because the key is named `values`
or `ids`, still serves the key from `values`.

A reader serves a key from `values` as follows:

- Without a `key_layout`, row `i` of `values` holds the key's value for the row
  that `ids[i]` identifies, nulls included. The value is exactly the one that
  the table stores, so a float keeps its bit pattern. The reader serves the key
  only when the [storage type](#carried-column-reader-rules) of `values` equals
  the storage type of the key field in the dataset schema, and otherwise takes
  the key from the table.
- With a `key_layout`, the reader splits the encoded tuple of row `i` into its
  keys from left to right, then inverts
  [Encoding `arrow_row` Version 1](#encoding-arrow_row-version-1) for the key's
  type, `descending`, and `nulls_first`. The inverse gives back each null and
  the original bits of each value, NaN payloads and `-0.0` included, in the
  type that the key field has in the dataset schema.

A key served from `values` covers the same rows that a carried column of the
segment would, the rows that remain after
[Handling deleted and invalidated rows](../index.md#handling-deleted-and-invalidated-rows).

### Carried Column Reader Rules

A reader applies these rules together with the rules of the segment's page
layout, including the [Key Layout Reader Rules](#key-layout-reader-rules). From
the manifest alone, it treats a BTree segment as unusable when either of these
is true:

- Under the legacy contract, `covering_fields` is not a suffix of `fields`
  that leaves at least one key.
- The segment has no `key_layout` and does not have exactly one key.

Under the legacy contract, a reader takes the [key fields](#key-fields) from
the suffix rule and applies the key-count rules above: one key without a
`key_layout`, and `len(key_layout.keys)` keys with one. It serves no carried
column, serves the keys from `values` as [Serving Keys](#serving-keys)
describes, and takes every other projected column from the table. The segment
still answers key queries. This holds whatever `page_data.lance` stores, for
example after a writer converts the manifest back to the legacy contract.

Under the independent contract, a reader finds the carried columns of
`page_data.lance` by exclusion: every column other than `values` and `ids` is a
carried column. It binds the carried column at position `i` among them to the
field id at position `i` of `covering_field_ids`. This physical order binds the
columns, and the order of `covering_fields` does not. Binding ignores the
column name. The reader serves a carried column only when its storage type
equals the storage type of the bound field in the dataset schema. It serves the
values under the current dataset field, with that field's current name,
metadata, and extension type.

Two fields have the same storage type when their Arrow data types are equal.
The comparison covers the type and its parameters, such as bit widths, units,
time zones, list sizes, and decimal precision and scale. For a nested type it
also covers each child's name, storage type, and nullability, in order. For a
dictionary it covers the key type and the value type. Field metadata at any
level, the name and the nullability of the top-level field, and the
dictionary's `ordered` flag are not part of the storage type. A metadata or
nullability change keeps the field id and rewrites no data (see
[Schema Evolution](../../table/schema.md#schema-evolution)), so the stored
values still equal what a take returns. A type change creates a replacement
field with a new id, so a carried column of the old field never binds to it.
Neither a carried column nor `values` binds to a reused id, because
[Removed and replaced fields](../index.md#removed-and-replaced-fields) removes
the segment in the manifest that removes the field.

A usable segment serves no value, neither a key nor a carried column, when its
pages are in `part_*_page_data.lance` files instead of `page_data.lance`, or
when a column name appears more than once in `page_data.lance`. It still
answers key queries.

A usable segment serves no carried column when any of these is true:

- `covering_fields` names a field twice, or names a field that is not a
  top-level field of the dataset schema.
- The file has carried columns and no `covering_field_ids`, or has
  `covering_field_ids` and no carried columns.
- `covering_field_ids` does not follow the form above, names an id twice, or
  has a different number of ids than the file has carried columns.
- A bound id is not in `covering_fields`, or is not a top-level field of the
  dataset schema, or the column's storage type differs from the field's.
- A field that `covering_fields` declares is a blob field, or contains one, in
  the dataset schema.

Such a segment still answers key queries and serves its keys as
[Serving Keys](#serving-keys) describes, and the reader takes every other
projected column from the table. A reader serves carried values only for the
rows that the segment returns once
[Handling deleted and invalidated rows](../index.md#handling-deleted-and-invalidated-rows)
has filtered them, and it checks each segment on its own, as
[Serving carried columns](../index.md#serving-carried-columns) requires.

### Maintaining Carried Columns

A writer that rewrites a segment either keeps each carried value in the row it
belongs to, or withdraws the column. It never keeps a carried column whose
values it has not moved with their rows.

- Index optimization takes the carried value of each row that it keeps from
  the source segment's column that binds to the same field, and the value of
  each new row from the table version that it indexes. It writes a carried
  column only if it has a value for every row of the new page data file.
- Remapping changes `ids` and leaves each carried value in its row. A row that
  remapping removes takes its carried values with it.
- A writer never fills one carried column from source columns that bind to
  different field ids, even when their names and types match.
- A writer that withdraws a column writes the new page data file without it
  and without its id in `covering_field_ids`. The field may stay in
  `covering_fields`. A file without carried columns has no
  `covering_field_ids` key.

### Carried Column Compatibility

Carried columns add no index version and no feature flag.
`FLAG_INDEPENDENT_COVERING_FIELDS` (`2^13`) in the reader and the writer
feature flags fences them. A writer stores carried columns only under the
independent contract, and an implementation that does not support that
contract rejects the table (see
[Current Feature Flags](../../table/versioning.md#current-feature-flags)).
Serving keys from `values` changes no file, so it adds no index version and no
feature flag.
