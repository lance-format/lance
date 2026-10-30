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
[Key Layout](#key-layout).

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

## Stable Row ID Reverse Lookup

An ordinary BTree may index the logical `_rowid` column of a table, given the following rules:

* Stable Row IDs **must** be enabled on the table.
* `IndexMetadata.fields` **must** be a single field ID `-3`.
* `_rowid` **must** be represented as a non-nullable `UInt64` field.
* The index **must** use the address-domain format and be at least version 1.
* The index **must not** set `key_layout`.

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

The index orders rows by the tuple of their key values, compared one key at a
time. Each key keeps its own order inside the tuple order, once its direction
and null placement are applied. When two tuples have equal values for every
earlier key, they compare as their values of the next key compare. Each page
key is one byte string that encodes the whole tuple, and byte order equals
tuple order, so the page lookup above finds pages by comparing encoded keys as
bytes. Nulls are part of the encoded tuple, so the null handling in
[Accelerated Queries](#accelerated-queries) does not apply to an index with a
`key_layout`.

### Key Layout Metadata

A segment with `key_layout` sets these `IndexMetadata` fields:

- `index_version` is 2.
- `fields` lists the key columns in key order. It has at least one entry.
  Each entry names a field of the dataset schema, and no field appears twice.
- `index_details` is a `BTreeIndexDetails` whose `key_layout` describes the keys.

This version defines no covering columns for an index with a `key_layout`, so
`fields` holds exactly its keys under either
[covering-field contract](../index.md#covering-field-contracts). A later format
change may add covering columns.

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
- `fields` is empty, names a field twice, or names a field that is not in the
  dataset schema.
- `covering_fields` is not empty.
- `key_layout.keys` and `fields` have different lengths.
- The reader does not support the `encoding` or the `ordering`, including their
  parameters such as the `arrow_row` `version`. A missing `encoding` or
  `ordering`, or one whose `oneof` is unset, is not supported.
- The encoding does not admit the type of a key column.
- The manifest's writer feature flags lack `FLAG_BTREE_KEY_LAYOUT`.

An engine skips an unusable index and scans the fragments that the index
covers, as it does for any index segment it cannot use (see
[Index Compatibility](../index.md#index-compatibility)). If the manifest lacks
`FLAG_BTREE_KEY_LAYOUT`, a writer that supports key layouts drops the
index (see
[Compatibility With Released Versions](#compatibility-with-released-versions)).
Otherwise the engine does not add rows or fragments to the index, and its
compaction does not rewrite the fragments that the index covers. It still
removes fragments from the index's `fragment_bitmap` when an update requires it
(see
[Creating and Updating Index Segments](../index.md#creating-and-updating-index-segments)).

### Encoding Stability

The bytes of `arrow_row` version 1 are the bytes that the `RowConverter` of the
Arrow Rust crate `arrow-row` 58.4.0 writes for the admitted types. Arrow does
not promise to keep those bytes across releases, so this page, not the crate,
defines version 1. If a later `arrow-row` release writes different bytes for an
admitted type, Lance keeps writing version 1 bytes for that type by vendoring
the version 1 encoder. An Arrow upgrade never changes version 1. A different
byte layout is a new encoding version, `arrow_row` version 2.

### Compatibility With Released Versions

Index version 2 keeps released readers from using an index with a
`key_layout`, but nothing in the index keeps v12.0.0 and v13.0.0 writers from
updating a table with stable row IDs. Their update adds the fragment that
receives the rewritten rows to the index's `fragment_bitmap`, although the
index holds no row address in that fragment. A table with such an index
therefore sets `FLAG_BTREE_KEY_LAYOUT` (`2^19`) in its writer
[feature flags](../../table/versioning.md#current-feature-flags), and not in its
reader feature flags. The index version fences readers from v10.0.0 on, and the
flag fences writers from v12.0.0 on.

Releases v12.0.0 and v13.0.0 read BTree indices up to version 0, and release
v14.0.0 reads them up to version 1, so these releases ignore an index with a
`key_layout` and answer queries from the table data. Releases v10.0.0 and
v11.0.0 also ignore it when they read. In all five releases, equality filters
on key columns return the same rows as a full scan.

Releases v12.0.0 through v14.0.0 commit no change to a table that sets
`FLAG_BTREE_KEY_LAYOUT`. Their insert, update, delete, index creation, column
addition, restore, shallow clone, and branch creation fail with an
unsupported-feature error. Their index optimization and compaction either fail
the same way or commit nothing.

Releases v10.0.0 and v11.0.0 check writer feature flags only when they insert.
Their update, delete, index creation, column addition, and compaction commit
anyway. Each of these commits removes the index with a `key_layout` from the
manifest and clears `FLAG_BTREE_KEY_LAYOUT`. Their index optimization does the
same or commits nothing. Their restore, shallow clone, and branch creation
clear the bit but keep the index. On a table with stable row IDs, a later
update by v12.0.0 or v13.0.0 can then add fragments to the index's
`fragment_bitmap`, and a later commit, clone, or restore by a writer that
supports `FLAG_ROW_ADDR_INDICES` (`2^18`) but not key layouts sets
`FLAG_ROW_ADDR_INDICES` again from `index_version >= 1` without repairing or
dropping the segment. Their reads stay correct throughout.

A reader therefore does not use an index with a `key_layout` from a manifest
whose writer feature flags lack `FLAG_BTREE_KEY_LAYOUT` (see
[Key Layout Reader Rules](#key-layout-reader-rules)).

A writer that supports key layouts never carries an index with a `key_layout`
out of a manifest whose writer feature flags lack the bit. That rule covers the
manifest that a commit builds on and the version that a restore, clone, or
branch copies. The writer drops such an index before it computes the bit.
Every manifest it publishes, including by clone and restore, sets
`FLAG_BTREE_KEY_LAYOUT` exactly when its index section has a segment with a
`key_layout`. The bit covers only segments with a `key_layout`.
