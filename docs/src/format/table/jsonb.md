# JSONB Format Specification

This specification defines Lance's `json` logical type, its Arrow representation,
and the complete binary JSONB encoding maintained by `lance-jsonb`. The encoding
originated in Databend JSONB. Its container layout, scalar encodings, and historical
read encodings are defined here so that readers and writers can implement the
persisted format independently.

The binary format and text conversion have different domains. Binary values can
contain decimals and extension types that the text parser does not produce. The
text-conversion section describes current Lance behavior, including precision
loss and the distinction between representational limits and enforced input limits.

## Logical Type

A `json` field contains one complete value per row. Ordinary JSON values are null,
boolean, number, UTF-8 string, array, and object. Any of these can be the root.
Arrays preserve order and can contain different value types. Objects associate
unique UTF-8 keys with values. Object key order is not part of the logical value;
writers sort keys by unsigned lexicographic order of their decoded UTF-8 bytes,
with a shorter prefix preceding a longer key. No Unicode normalization or case
folding is applied.

Numbers have signed-integer, unsigned-integer, binary64, and fixed-width decimal
representations. The binary format also supports binary strings, dates,
timestamps, timestamps with UTC offsets, and intervals. These extension values
can appear at the root or inside arrays and objects. A timestamp extension and
a JSON string containing a timestamp are different values.

An Arrow null is represented by the enclosing column's validity information. The
bytes of a null slot have no JSON meaning. A JSON null is a valid column value
with a JSONB null encoding. Nulls inside JSON arrays or objects do not introduce
Arrow child validity bitmaps. A missing object key has no entry and is distinct
from a key containing JSON null. Empty strings, arrays, and objects are distinct
non-null values.

The enclosing field follows the [schema and field-ID rules](schema.md). Keys and
elements inside a JSONB value do not receive Lance field IDs. Changing them changes
the value, not the table schema.

## Arrow Representation and File Placement

The persisted Lance field has `logical_type = "json"`. Its physical Arrow field
uses `LargeBinary` and the `lance.json` extension:

```python
pa.schema([
    pa.field(
        "value",
        pa.large_binary(),
        nullable=True,
        metadata={b"ARROW:extension:name": b"lance.json"},
    ),
])
```

The field name and nullability are application-defined. Each non-null binary slot
contains exactly one complete JSONB container. Arrow supplies the slot boundaries
through 64-bit offsets. On disk, the column uses the ordinary
[variable-width encodings](../file/encoding.md); page compression and structural
encoding surround the JSONB value. Arrow buffers need not be copied verbatim.
The `json` type does not request external blob storage.

There is no JSONB-specific magic, format-version field, checksum, or outer length
prefix. The table schema identifies the value encoding, and the enclosing column
identifies its byte extent. PostgreSQL and SQLite formats also called JSONB have
different layouts.

Text input uses the `arrow.json` extension with either of these storage types:

```python
pa.schema([
    pa.field(
        "value",
        pa.string(),
        nullable=True,
        metadata={b"ARROW:extension:name": b"arrow.json"},
    ),
])

pa.schema([
    pa.field(
        "value",
        pa.large_string(),
        nullable=True,
        metadata={b"ARROW:extension:name": b"arrow.json"},
    ),
])
```

Text-to-storage conversion replaces the storage type and extension with
`LargeBinary` and `lance.json`. Normal dataset scan output converts them to
`Utf8` and `arrow.json`, including when the input used `LargeUtf8`. Internal
consumers can also request decoded `LargeUtf8` arrays. These choices affect Arrow
buffer limits, not JSONB bytes.

Conversion preserves field names, nullability, and other field metadata. It
recurses through structs, lists, large lists, fixed-size lists, and map entries;
each enclosing Arrow structure retains its own offsets and validity. An Arrow
list of JSON fields is distinct from an array inside a single JSONB value.
Reconstructing an Arrow field from `logical_type = "json"` restores `lance.json`.

An unmarked string or binary field is not implicitly JSON. Already encoded
`lance.json` input passes through the storage adapter without text parsing,
re-encoding, or complete binary validation.

## Binary Encoding

### Conventions

All multibyte integers, container headers, entry descriptors, and floating-point
bit patterns use big-endian byte order. Signed integers use two's complement.
Offsets below are zero-based byte offsets from the start of the containing
container. Lengths count bytes, not characters. Fields and payloads are packed
without alignment, padding, or terminators.

A document is a tree of containers. Each container consists of a 4-byte header,
a descriptor table, and concatenated payloads. A descriptor is called a
**JEntry**. Its type determines how to interpret its payload.

### Container Header

For a 32-bit header `H`, `H & 0xE0000000` identifies the container type and
`H & 0x1FFFFFFF` gives its count:

| Container | Type bits | Count | Number of JEntries |
|---|---|---|---|
| Scalar | `0x20000000` | Must be zero | One |
| Object | `0x40000000` | Number of key/value pairs, `N` | `2N` |
| Array | `0x80000000` | Number of elements, `N` | `N` |

The scalar header is exactly `0x20000000`. Other combinations of the three type
bits are undefined. The object count is the number of pairs, not the size of its
descriptor table.

### JEntry

Each JEntry is a 32-bit word with this layout:

```text
bit 31        bits 30:28 bits 27:0
+-------------+---------+-------------------------------------+
| reserved: 0 | type: 3 | payload length in bytes: 28          |
+-------------+---------+-------------------------------------+
```

The type is `J & 0x70000000`; the length is `J & 0x0FFFFFFF`. The high bit must
be zero. Although that bit was reserved for an offset mode, Lance implements
lengths only. A reader computes payload offsets by summing preceding lengths.

| Type | Type bits | Payload |
|---|---|---|
| Null | `0x00000000` | Empty; length must be zero |
| String | `0x10000000` | Exactly `length` UTF-8 bytes |
| Number | `0x20000000` | Numeric tag and numeric data |
| False | `0x30000000` | Empty; length must be zero |
| True | `0x40000000` | Empty; length must be zero |
| Container | `0x50000000` | Complete nested container, including its header and table |
| Extension | `0x60000000` | Extension tag and extension data |

Type bits `0x70000000` are undefined. Number and extension lengths include their
one-byte subtype tag. A string has no subtype tag, quote, terminator, additional
length field, or automatically inserted byte-order mark. Its payload is the
decoded string; escape sequences from JSON text are not stored. Embedded zero
bytes are allowed as the UTF-8 encoding of U+0000.

### Scalar Container

```text
Scalar container
|-- header: 4 bytes, 0x20000000
|-- JEntry: 4 bytes
`-- payload: JEntry.length bytes
```

The single JEntry is null, false, true, string, number, or extension. Its payload
begins at offset 8. The total size is `8 + length`. Arrays and objects use their
own container headers directly rather than a scalar wrapper.

### Array Container

```text
Array container
|-- header: 4 bytes, 0x80000000 | N
|-- JEntry[0] ... JEntry[N-1]: 4N bytes
`-- payload[0] ... payload[N-1]
```

JEntry `i` is at offset `4 + 4i`. The payload area starts at `P = 4 + 4N`.
For lengths `L[0] ... L[N-1]`, element `i` occupies
`[P + sum(L[0:i]), P + sum(L[0:i+1]))`. Empty payloads consume no bytes but
still have descriptors. The total size is `P + sum(L)`.

Scalar elements use their scalar JEntry and payload directly, without an
8-byte scalar-container wrapper. Nested arrays and objects use a container
JEntry whose length covers the entire nested container. The nested container's
offsets are relative to its own first byte. An empty array consists only of
the header `0x80000000`.

### Object Container

```text
Object container
|-- header: 4 bytes, 0x40000000 | N
|-- key JEntry[0] ... key JEntry[N-1]: 4N bytes
|-- value JEntry[0] ... value JEntry[N-1]: 4N bytes
|-- key payload[0] ... key payload[N-1]
`-- value payload[0] ... value payload[N-1]
```

Keys and values are in matching key-sorted order. Every key descriptor is a string
JEntry. Key `i` has its descriptor at `4 + 4i`; value `i` has its descriptor at
`4 + 4N + 4i`. All keys precede all values in both the descriptor and payload
areas; key/value pairs are not interleaved.

Let `K[i]` and `V[i]` be key and value payload lengths. The key area starts at
`P = 4 + 8N`, and the value area starts at `Q = P + sum(K)`. Key `i` begins at
`P + sum(K[0:i])`; value `i` begins at `Q + sum(V[0:i])`. The total size is
`Q + sum(V)`. Empty keys are permitted; duplicate decoded keys are not. An empty
object consists only of the header `0x40000000`.

### Reader Navigation and Writer Construction

A reader first obtains a complete document slice from the enclosing column. It
reads the header and descriptor table, computes the payload boundaries above,
then dispatches by JEntry type. Reading array element `i` requires summing the
preceding element lengths. Object lookup finds a key in the key area and uses
the corresponding value descriptor and prefix sum in the value area. There is
no persisted offset index or object hash table.

A container payload is recursively decoded within its own byte extent. The
writer emits container entries for nested arrays and objects. Some existing
reader paths also accept a scalar container behind a container entry; this is
not a layout emitted by the normal writer or a requirement for new writers.
To expose an ordinary scalar element as a standalone document, prepend the scalar
header to its JEntry and payload. A nested array or object is already a complete
document and can be copied directly.

A writer orders object keys, writes the appropriate header, reserves its complete
descriptor table, and appends payloads in table order. It records each payload's
actual byte length in the corresponding descriptor. Child containers are
complete before their parent descriptor length is finalized. Null and boolean
values need no payload allocation. A writer can equivalently compute sizes before
emitting bytes; the resulting layout is the same.

## Numeric Payloads

A numeric payload starts with a one-byte tag. The JEntry length distinguishes
variable-width forms; there is no separate numeric width field.

| Tag | Representation | Total payload length | Bytes after the tag |
|---|---|---|---|
| `0x00` | Integer zero | 1 | None |
| `0x10` | NaN | 1 | None |
| `0x20` | Positive infinity | 1 | None |
| `0x30` | Negative infinity | 1 | None |
| `0x40` | Signed integer | 2, 3, 5, or 9 | 1-, 2-, 4-, or 8-byte signed integer |
| `0x50` | Unsigned integer | 2, 3, 5, or 9 | 1-, 2-, 4-, or 8-byte unsigned integer |
| `0x60` | Binary64 | 9 | 8-byte IEEE 754 bit pattern |
| `0x70` | Decimal | 10, 18, or 34 | Signed coefficient followed by one unsigned scale byte |

Other tags are undefined. Historical decimal lengths are defined separately below.

### Integers

Signed integers cover `[-2^63, 2^63 - 1]`; unsigned integers cover
`[0, 2^64 - 1]`. For a nonzero value, the writer chooses the smallest width from
1, 2, 4, and 8 bytes that contains the value in the chosen signedness. There are
no 3-, 5-, 6-, or 7-byte integer bodies. A reader sign-extends a signed body and
zero-extends an unsigned body. Wider permitted widths still identify the same
number, so minimal width is a writer choice, not a precondition for decoding.

Both signed and unsigned integer zero serialize as the single byte `00`, decoded
as unsigned zero. Positive values can use either signed or unsigned tags when
supplied as already typed numbers. The text parser normally selects unsigned
integers for nonnegative integer input.

### Floating Point

Finite binary64 values use tag `60` and their big-endian IEEE 754 bits. Positive
and negative floating-point zero keep distinct sign bits and remain distinct
from the integer-zero representation. New serialization maps every NaN to tag
`10`, discarding its payload and sign, and maps infinities to tags `20` and `30`.
A decoder of tag `60` interprets its bits as binary64, including non-finite bit
patterns accepted by existing readers.

Binary64 has 53 bits of significand precision. Its largest finite magnitude is
approximately `1.7976931348623157e308`, and its smallest positive subnormal
magnitude is approximately `4.9406564584124654e-324`.

### Decimals

A decimal has the exact mathematical value `coefficient * 10^(-scale)`. The
coefficient is a two's-complement integer in 8, 16, or 32 bytes, and scale is an
unsigned byte in `[0, 255]`. The three layouts are:

```text
Decimal64:  70 | coefficient: i64  | scale: u8
Decimal128: 70 | coefficient: i128 | scale: u8
Decimal256: 70 | coefficient: i256 | scale: u8
```

All coefficient bytes are big-endian, including the 256-bit form. The JEntry
length selects the width. There is no current precision field. The encoding
does not impose a smaller decimal-digit range than the signed coefficient width.
The writer preserves the selected decimal width and scale, even for zero or an
integral value; it does not remove trailing decimal zeros or convert decimals
to integer or binary64 payloads. The text parser does not construct decimal
values, but already encoded decimals remain readable and can be copied.

## Extension Payloads

An extension payload starts with a one-byte tag. Every integer field below is
big-endian and signed unless specified otherwise. Length includes the tag.

| Tag | Type | Length | Fields after the tag |
|---|---|---|---|
| `0x00` | Binary | `1 + B` | `B` uninterpreted bytes; `B` can be zero |
| `0x10` | Date | 5 | `days: i32` |
| `0x20` | Timestamp | 9 | `microseconds: i64` |
| `0x30` | Timestamp with offset | 13 | `microseconds: i64`, `offset_seconds: i32` |
| `0x40` | Interval | 17 | `months: i32`, `days: i32`, `microseconds: i64` |

Other tags are undefined. Binary data has no UTF-8 requirement. Dates count days
from 1970-01-01. Timestamps count microseconds from 1970-01-01 00:00:00 UTC; a
plain timestamp stores no separate timezone. An offset timestamp stores the UTC
instant and a fixed offset in seconds east of UTC. Local time is the stored
instant plus that offset. It stores no timezone name or daylight-saving rules.

Intervals retain months, days, and microseconds as independent signed components.
There is no normalization between components: a month is not a fixed number of
days, and microseconds are not carried into the day field. Components can have
different signs. Extension subtype tags have meaning only under an extension
JEntry, independently of identically numbered numeric subtype tags.

## Historical Read Encodings

Readers preserve these existing layouts. Newly serialized typed values use the
current layouts above; copying an existing binary value can retain its original
bytes.

| JEntry type | Subtype | Length | Historical layout |
|---|---|---|---|
| Number | `0x70` | 19 | `70`, coefficient `i128`, precision `u8`, scale `u8` |
| Number | `0x70` | 35 | `70`, coefficient `i256`, precision `u8`, scale `u8` |
| Extension | `0x30` | 10 | `30`, timestamp microseconds `i64`, offset hours `i8` |

Historical decimal readers skip the precision byte and use the final byte as
scale. The current 18- and 34-byte layouts omit that precision byte. There is no
corresponding historical Decimal64 form. Historical timestamp readers sign-extend
the hour byte and multiply it by 3,600 to obtain the offset in seconds.

These forms are selected by the JEntry length. No document-wide version marker
is needed to distinguish them, and current and historical values can coexist in
one array or object.

## Conformance Vectors

Each hexadecimal sequence below is a complete document, including its container
header and descriptors. Decimal and historical extension values are identified
by their binary representation rather than by text-parser input.

| Value or representation | Complete JSONB bytes in hexadecimal |
|---|---|
| JSON null | `2000000000000000` |
| Integer zero | `200000002000000100` |
| Binary64 negative zero | `2000000020000009608000000000000000` |
| String `"a"` | `200000001000000161` |
| Array `[1,2,3]` | `80000003200000022000000220000002500150025003` |
| Object `{"b":1,"a":2}`, stored in key order | `4000000210000001100000012000000220000002616250025001` |
| Decimal64, coefficient -12345, scale 2 | `200000002000000a70ffffffffffffcfc702` |
| Historical offset timestamp, epoch microseconds 0, offset +8 hours | `200000006000000a30000000000000000008` |

## Limits and Well-Formedness

### Representational Limits

The count field has 29 bits, so an array can name at most `2^29 - 1` elements and
an object at most `2^29 - 1` pairs. Every JEntry payload is at most `2^28 - 1`
bytes (268,435,455 bytes). This includes an entire child container, its descriptor
table, and all descendants when stored as a nested value.

Consequently, a string or key can contain at most 268,435,455 UTF-8 bytes, and a
binary extension can contain at most 268,435,454 data bytes after its tag. A
nested container must satisfy both its own count bound and its parent's byte
length bound. All descendants must fit inside that enclosing extent.

The root has no enclosing JEntry. A root scalar is at most `8 + (2^28 - 1)`
bytes. A root array or object has no single 28-bit total-length field; its total
size is the header and descriptor table plus the sum of its payload lengths.
The per-entry bound must not be presented as a universal 256 MiB document limit.

Arrow's 64-bit binary offsets do not enlarge these fields. Normal scan output
uses signed 32-bit string offsets, so its cumulative text-buffer extent must fit
within `2^31 - 1` bytes. Text escaping can expand the data. Column encodings,
batch sizes, available memory, and recursion resources impose additional limits.
There is no encoded nesting-depth field or separate guaranteed maximum input
document size or depth.

### Validation Rules

A well-formed document satisfies the following checks. Arithmetic for sizes and
prefix sums must be checked before allocation or slicing:

1. The document contains a valid 4-byte header. Its descriptor table fits in the
   supplied extent. A scalar has a zero count and exactly one scalar descriptor.
2. Each descriptor has a zero reserved bit and a defined type. Object keys use
   string descriptors. Null and boolean lengths are zero.
3. Summed payload lengths exactly exhaust the container's extent. Every child
   lies inside its parent's extent, with no padding or trailing bytes.
4. Strings and keys contain valid UTF-8. Object keys are unique and sorted as
   defined by the logical type.
5. Number and extension tags have one of the defined lengths. Their fields are
   decoded with the specified width, signedness, and byte order. Child containers
   satisfy the same structural checks recursively.

Length overflow, unknown tags, unsupported offset mode, truncated tables,
out-of-bounds payloads, invalid UTF-8, and inconsistent subtype lengths do not
define additional valid encodings.

These are format validity rules, not a claim that every current entry point
enforces them. The current encoder does not uniformly preflight count and length
overflow, and native `lance.json` ingestion does not perform a full validation
pass. Some readers assume valid input and can fail outside a structured error
path on malformed bytes. The text-rendering helper also falls back to interpreting
some invalid binary inputs as text. Neither pass-through storage nor successful
rendering establishes conformance. Binary producers must supply well-formed
values; the representation bounds are not a guarantee of successful ingestion
or consistent rejection at each boundary.

## Text Conversion

### Parsing and Numeric Precision

The current parser accepts JSON with extensions including single-quoted strings,
unquoted object keys, case-insensitive null and boolean literals, NaN and
Infinity, leading plus signs and zeros, omitted digits around a decimal point,
and hexadecimal numbers. Empty or whitespace-only text becomes JSON null.
Omitted array elements become JSON null, including an extra null for a trailing
comma. Duplicate decoded object keys are rejected. After one value, the parser
skips its whitespace and escape extensions and rejects any remaining input.
This describes the current parser, not a general JSON5 conformance claim.

For ordinary decimal notation without an exponent or fractional digits, values
in `[0, 2^64 - 1]` become unsigned integers, and negative values in
`[-2^63, -1]` become signed integers. Integer `-0` becomes integer zero. A
decimal point with no following digits, such as `1.`, does not itself require
floating-point storage. Fractional digits, exponent notation, and decimal
integers outside those ranges fall back to binary64. Floating `-0.0` retains
its sign. Hexadecimal integers use the same signed/unsigned ranges after parsing
their magnitude; the current parser bounds hexadecimal components by UInt128
and can reject values beyond that intermediate range.

The text writer's integer representations therefore cover the continuous range
`[-2^63, 2^64 - 1]` exactly. It is not an accepted-input limit. Other input can be
rounded to binary64, overflow to infinity, or underflow to zero. Arbitrary-precision
decimal parsing and lossless numeric round-trip validation are not enabled.
Whitespace, source key order, escape spelling, and numeric spelling are not
preserved.

Arrow JSON input and SQL JSONB literals use this text encoder. JSON text assigned
through the dataset update path is encoded before storage. The separate binary
input path preserves already encoded values.

### Rendering

Normal Arrow JSON output is compact JSON text. Keys are emitted in stored order;
strings are quoted and escaped. JSON null renders as the non-null text `null`,
while an Arrow null remains an Arrow null. Signed and unsigned integer values
render with their exact decimal digits. Binary64 renders from its stored value;
NaN and both infinities render as JSON `null`.

The current renderer converts a decimal coefficient to binary64 and divides it
by binary64 `10^scale` before rendering. Thus exact decimal bytes do not guarantee
exact decimal text output. Extension values render as JSON strings: binary data
as uppercase hexadecimal without a prefix, dates as calendar dates, timestamps
with six fractional-second digits, offset timestamps with their fixed UTC offset,
and intervals as month/day/time components. Calendar and timezone rendering has
a narrower domain than the full integer fields in the binary layout; the current
formatter can clamp extreme timestamps or fail for out-of-range calendar values
or offsets.

A text round trip can therefore lose a numeric representation, non-finite value,
decimal precision, or extension type. Preserving binary values avoids these text
conversions. Byte equality is not general numeric or JSON equality: signed and
unsigned representations, integer and floating zero, and decimal scales can
differ. Lexicographic comparison of encoded bytes is not a JSON value ordering.

## Compatibility and Evolution

Lance owns this persisted encoding and its compatibility obligations. The
`lance-jsonb` package version is not stored in a payload. Refactoring or upgrading
the codec must preserve the interpretation of existing values and keep new
output readable by the released readers covered by Lance's format contract.
The historical read encodings remain readable without being extended into new
write formats.

Undefined header combinations, entry types, subtype tags, and the reserved
JEntry bit are not silently available for new writers. There is no payload-level
negotiation that would let an old reader distinguish a new interpretation.
An incompatible change requires an explicit format selection and reader/writer
compatibility mechanism under the [file-format versioning rules](../file/versioning.md)
and [table-format versioning rules](versioning.md), including how existing data
and supported readers are handled.

Text parsing, rounding, and rendering are also observable behavior. A change to
those conversions must be evaluated separately from byte-layout compatibility.
Increasing parser precision cannot recover digits already lost in stored
binary64 values.

[JSON indexes](../index/index.md) are redundant structures over decoded values.
Their inferred key types, query coercions, and index versions do not redefine
the base column's payload or numerical precision. SQL accessor and JSONPath
evaluation rules are separate from this storage encoding.
