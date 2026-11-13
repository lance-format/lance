# List Element Index

!!! warning "Experimental"

    This index is an experimental format feature. Its details message and
    its files may change incompatibly without a separate vote; after such a
    change the index must be rebuilt, the data itself is never rewritten.
    The feature is removed if its stabilization vote does not pass. Design
    discussion: [#10948](https://github.com/lance-format/lance/discussions/10948).

The list element index indexes fields inside the elements of a `list<struct>`
column. It holds one entry per element. The entry's key is a tuple of chosen
fields of that element, its *keys*, and its posting is the row address of the
row whose list holds the element. Because every key tuple comes from a single
element, a search for a tuple finds the rows that have one element matching on
every key at once, not rows where different elements match different keys.

Take a column `detections` of type
`list<struct<label: string, confidence: float, area: float>>`, indexed with the
keys `label` and `confidence`. A row whose list holds `{car, 0.95, 120}` and
`{person, 0.30, 80}` has the entries `(car, 0.95)` and `(person, 0.30)`, both
posting the row's address.

The index is a wrapper, like the [JSON index](json.md). It encodes each key
tuple as one byte string whose byte order is the tuple order, and stores those
byte strings in an ordinary [BTree](btree.md) over a `Binary` column, the
*target index*, which does all of the storage and searching.

## Index Details

```protobuf
%%% proto.message.ListElementIndexDetails %%%
```

`IndexMetadata.fields` holds the single `list<struct>` field the index is built
on, which may be a `List` or a `LargeList` of a `Struct`.

`key_field_ids` identifies the key fields, in key order, by their field IDs
(`file.Field.id`), following the [field-reference
contract](../../table/schema.md#using-field-ids). Each is a direct child of the
element struct, there is at least one key, and no ID appears twice. `keys` holds
the names those fields had when the segment was written. The names are for
display only: a rename does not update them, and a reader must not use them to
find the key fields.

`key_types` holds the [logical type](../../table/schema.md) of each key field
when the index was trained, in the same order as `key_field_ids`. A key may
have one of these logical types:

| Logical types | Arrow types |
|---|---|
| `bool` | Boolean |
| `int8`, `int16`, `int32`, `int64` | Int8, Int16, Int32, Int64 |
| `uint8`, `uint16`, `uint32`, `uint64` | UInt8, UInt16, UInt32, UInt64 |
| `float`, `double` | Float32, Float64 |
| `date32:day`, `date64:ms` | Date32, Date64 |
| `timestamp:{unit}:{tz}` | Timestamp with any unit and time zone |
| `string`, `large_string` | Utf8 (or Utf8View), LargeUtf8 |
| `binary`, `large_binary` | Binary (or BinaryView), LargeBinary |

Before using the index, a reader resolves each ID in `key_field_ids` against
the current schema. The field must exist, must be a direct child of the element
struct of the column in `IndexMetadata.fields`, and must have the logical type
recorded in `key_types`. The reader then refers to each key field by its
current name, so the index follows a rename, including one that gives a key's
old name to another field. If any key field fails these checks, because it was
dropped, moved or changed type, the reader must not use the index and evaluates
the filter on the data instead.

`key_encoding` is `LIST_ELEMENT_KEY_ENCODING_ARROW_ROW_V1`, the encoding
defined in [Key Encoding](#key-encoding). A reader must not use an index whose
`key_encoding` it does not know, including
`LIST_ELEMENT_KEY_ENCODING_UNSPECIFIED`.

`target_details` holds the `BTreeIndexDetails` of the target index, and
`target_index_version` holds the target's own format version, the
`index_version` it would have in `IndexMetadata` as a standalone BTree. It is 1
or greater, so the target stores row addresses. `IndexMetadata.index_version`
holds the version of the list element index itself, which is 0.

Every segment of one logical index has the same `key_field_ids`, `key_types`
and `key_encoding`. Their `keys` may differ, if they were written on either
side of a rename.

## Storage Layout

The list element index writes no files of its own. Its files are exactly the
files of its target BTree, written with the BTree's names and schemas into the
same index directory, with `Binary` as the BTree's `{DataType}`. See
[BTree](btree.md) for the files.

The target's files therefore have these schemas:

```python
# page_data.lance: one row per element, sorted by `values`, nulls first
pa.schema([
    pa.field("values", pa.binary(), nullable=True),
    pa.field("ids", pa.uint64(), nullable=False),
])

# page_lookup.lance: one row per page
pa.schema([
    pa.field("min", pa.binary(), nullable=True),
    pa.field("max", pa.binary(), nullable=True),
    pa.field("null_count", pa.uint32(), nullable=False),
    pa.field("page_idx", pa.uint32(), nullable=False),
])
```

Each element of the indexed column gives one row of the target's
`page_data.lance`:

- `values` holds the element's key tuple, encoded as in
  [Key Encoding](#key-encoding). A null element encodes as a tuple whose keys are
  all null; its `values` is not null.
- `ids` holds the row address of the row that holds the element. A row with
  several elements has one row in `page_data.lance` for each element.

A row whose list is null has one row in `page_data.lance`, whose `values` is
null. A row whose list is empty has none.

To read the index, decode `ListElementIndexDetails`, resolve and check the key
fields and `key_encoding` as above, and then read the target exactly as a
standalone BTree, using `target_details` as its details and
`target_index_version` as its version.

## Key Encoding

`LIST_ELEMENT_KEY_ENCODING_ARROW_ROW_V1` encodes a tuple as the encodings of its
keys, concatenated in key order, with no header, separator or length. Every key
is encoded ascending with nulls first. Byte strings compare as unsigned bytes,
from the first byte, and a byte string sorts before every longer byte string
that starts with it. Under this comparison, the encoded tuples sort as the
tuples do: by the first key, then by the next key among equal first keys, and
so on, with a null key before every value of that key.

The encoding is the row format of the Arrow Rust implementation (`arrow-row`),
with every sort field ascending and nulls first. A key of a fixed-width type
encodes in 1 + W bytes:

- A non-null value is the byte `0x01`, then W value bytes from the table below.
- A null is the byte `0x00`, then W bytes of `0x00`.

| Type | W | Value bytes |
|---|---|---|
| Boolean | 1 | `0x00` for false and `0x01` for true |
| UInt8, UInt16, UInt32, UInt64 | 1, 2, 4, 8 | The value, big-endian |
| Int8, Int16, Int32, Int64 | 1, 2, 4, 8 | The two's-complement value, big-endian, with the most significant bit inverted |
| Float32, Float64 | 4, 8 | The IEEE 754 bit pattern, big-endian. If the sign bit is 1, every bit is inverted. Otherwise only the sign bit is inverted. |
| Date32 | 4 | The days since the UNIX epoch, encoded as an Int32 |
| Date64 | 8 | The milliseconds since the UNIX epoch, encoded as an Int64 |
| Timestamp | 8 | The stored value, encoded as an Int64. The unit and the time zone do not change the bytes. |

A key of a string or binary type encodes the UTF-8 bytes or the bytes of its
value, and a type and its large or view variant give the same bytes:

- A null is the byte `0x00` alone.
- An empty value is the byte `0x01`.
- Any other value is the byte `0x02`, then the value split into blocks. The
  first four blocks hold 8 bytes each, and every later block holds 32 bytes. The
  last block is padded with `0x00` to its full size. Each block is followed by
  one byte, which is `0xFF` if more of the value follows the block, and
  otherwise the number of value bytes in the block.

For example, with the keys `label: string` and `confidence: float`, the tuple
`("car", 0.5)` encodes as `02 63 61 72 00 00 00 00 00 03 01 BF 00 00 00`: the
string `car` in one 8-byte block with its length `03`, then `0x01` and the bit
pattern `3F000000` of 0.5 with its sign bit inverted.

## Accelerated Queries

The index accelerates the filter function
`any_element(list_column, 'predicate')`, which is TRUE for a row when the
predicate is TRUE on at least one element of its list, NULL when the list is
null or no element makes it TRUE but it is UNKNOWN on some element, and FALSE
otherwise. A null element makes every field UNKNOWN. The predicate refers to the
element's fields by their current names.

A reader may answer such a call from the index when its predicate is a
conjunction that fixes the first key with `=`, `IN` or `IS NULL`. Further
conjuncts may fix the following keys in order, and the last fixed key may be
followed by one range on the next key. Each combination of fixed values and the
range becomes one range of encoded tuples, searched in the target as a range of
`Binary` values: the tuples whose encoding starts with the encoded fixed values
and whose next key lies in the range. The answer is the union of the row
addresses found.

When some conjunct is not answered this way, for example a condition on a field
that is not a key, the index returns a superset of the rows on which the call
is TRUE, and the reader rechecks it by evaluating the call on them.

`NOT any_element(...)` also needs the rows on which the call is NULL, since
those must be excluded. They are the rows whose list is null, found as the null
`values` of the target, and the rows with an element on which the predicate is
UNKNOWN, unless another element of the row makes it TRUE. A conjunction is
UNKNOWN on an element when no conjunct is FALSE and at least one is UNKNOWN.

The index determines these rows exactly only when every conjunct is answered
from the keys as above and is UNKNOWN exactly when its key is null: `=`, `IN`
with non-null values, or a range with non-null bounds. `IS NULL` is never
UNKNOWN. Under that condition, the predicate is UNKNOWN on an element exactly
when a non-empty set of the keys that `=`, `IN` or the range constrain is null
and every other constrained key matches. With k such keys there are up to
2^k − 1 such sets, each crossed with the `IN` values of the keys it leaves
non-null; each combination is one range of encoded tuples, with a null in the
positions of the set. A reader may stop enumerating combinations past a limit
of its choosing.

Otherwise the index does not determine the NULL rows. That is the case when a
conjunct is not answered from the keys, when a conjunct holds a NULL literal
(`label IN ('car', NULL)` is UNKNOWN for `dog`, although `label` is not null),
when it uses `IS NOT NULL`, `IS [NOT] TRUE`, `IS [NOT] FALSE`,
`IS [NOT] UNKNOWN` or `IS [NOT] DISTINCT FROM`, or when the reader stopped
enumerating. A reader must then not answer `NOT any_element(...)` from the
index; it evaluates the call on the data instead.
