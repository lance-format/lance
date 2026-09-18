# JSONB Format Specification

## Scope

This specification describes the `json` logical type and the binary JSON (JSONB)
representation stored by Lance. It defines the mapping from a table field to an
Arrow array, the placement of each non-null array value, and the current conversion
between text and binary values. Lance adopts the upstream Databend JSONB encoding;
the payload's internal layout is defined by that upstream encoding, while this
specification defines its use within Lance.

Text conversion behavior is described separately from the adopted encoding
because accepted text, numeric precision, and representable binary values are
different boundaries.

## Logical Storage

### Values

A JSON field contains one complete value per row. Its logical values are null,
boolean, number, UTF-8 string, array, or object. A top-level value can be any of
these types. Arrays preserve element order and can contain values of different
types. Objects associate unique UTF-8 string keys with values of any type.

Object key order is not part of the logical value. The current codec orders keys
lexicographically by their decoded UTF-8 bytes, without Unicode normalization or
case folding. The text parser rejects duplicate decoded keys. Source key order,
whitespace, quote style, and escape spelling are not preserved.

The JSON value inside a field has no independently assigned Lance field IDs.
Changing an object's keys or an array's element types changes the field's value,
not the table schema. The enclosing field still follows the usual
[schema and field-ID rules](schema.md).

### Nullability

An Arrow null and a JSON null are distinct:

- An Arrow null is represented by the enclosing column's validity information.
  Its binary value has no JSON meaning.
- A JSON null is a valid, non-null column value containing a JSONB null scalar.
- A null inside a JSON array or object is a JSONB null entry. It does not introduce
  a child Arrow validity bitmap.
- A missing object key has no entry. It is distinct from a present key whose
  value is JSON null.

Empty strings, arrays, and objects are valid non-null values and have distinct
representations.

### Schema and Arrow Mapping

The persisted Lance field has `logical_type = "json"`. Its physical Arrow field
is defined by this schema (the field name and nullability are application-defined):

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

The text-facing Arrow field uses the `arrow.json` extension. The writer recognizes
both of the following storage types:

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

Text-to-storage conversion replaces the storage type with `large_binary` and the
extension name with `lance.json`. The normal dataset scan converts it back to
`arrow.json` with `string` storage, including when the original input used
`large_string`. Conversion preserves field names, nullability, and other field
metadata. Reconstructing an Arrow field from the persisted `json` logical type
restores the `lance.json` extension name.

These conversions also apply to JSON fields nested inside structs, lists, large
lists, fixed-size lists, and map entries. The enclosing Arrow structure retains
its own offsets and validity information. A JSON array inside one JSON value is
different from an Arrow list whose child field has the JSON extension.

A binary or string field without the corresponding extension metadata is not
implicitly a JSON field. Already encoded `lance.json` values pass through the
storage conversion without being parsed as text or re-encoded.

### Numbers

The normal text writer produces three numeric representations:

- Signed integers: inclusive range −9,223,372,036,854,775,808 through
  9,223,372,036,854,775,807 (−2^63 through 2^63−1).
- Unsigned integers: inclusive range 0 through 18,446,744,073,709,551,615 (2^64−1).
- IEEE 754 binary64 floating-point values, including signed zero and the
  non-finite values admitted by the current extended parser.

Nonnegative integer tokens fitting UInt64 are normally encoded as unsigned
integers; negative integer tokens fitting Int64 are encoded as signed integers.
Already encoded input can also contain positive signed integers. The upstream
codec determines the physical representation of these numbers.

The maximum exactly stored integer range is therefore −2^63 through 2^64−1.
This is **not an input rejection limit in the current implementation**. Decimal
fractions, exponent notation, and integer tokens outside that range can fall back
to Float64. They can lose precision. Float64 has 53 bits of significand precision,
a largest finite magnitude of approximately 1.7976931348623157 × 10^308, and a
smallest positive subnormal magnitude of approximately 4.9406564584124654 × 10^−324.
Overflow can produce infinity and underflow can produce zero.

The writer does not enable arbitrary-precision decimal parsing and does not
currently validate lossless decimal round trips. JSON text is consequently not
preserved as an arbitrary-precision number. Numeric spelling, including trailing
fractional zeros and exponent notation, is not part of the logical value.

## Physical Storage

### Placement in Lance Files

Each non-null physical Arrow value is one contiguous JSONB byte sequence. The
outer `large_binary` array supplies 64-bit offsets and column validity in memory.
On disk, it uses the ordinary variable-width column encodings described in the
[encoding specification](../file/encoding.md); Arrow buffers need not be copied
verbatim into a file. Page compression and structural encoding are outside the
JSONB payload.

Lance stores the upstream payload directly in the binary value, without adding a
JSONB-specific header, version marker, or checksum. The table field's logical type
identifies how to interpret that value. The `json` type does not itself request
external blob storage.

### Upstream Encoding

The adopted payload representation is the encoding provided by the `databend`
feature of `jsonb` 0.5.6. Its upstream description is the
[Encoding format section](https://github.com/databendlabs/jsonb/blob/c39b24aaea2c27edf6e596874d3a08b8e7806e60/README.md#encoding-format)
at the revision corresponding to that release. Other formats named JSONB,
including PostgreSQL and SQLite representations, are not interchangeable with
these payloads.

Container headers, entry tags, numeric encodings, byte order, and navigation
within a payload belong to the upstream encoding. Lance readers first decode the
enclosing column value, then interpret its bytes using a compatible JSONB
decoder. Writers serialize each logical value with a compatible encoder before
passing the resulting bytes to the column writer.

The upstream description outlines the encoding but is not a complete specification
of every numeric or extension representation recognized by the codec. This
reference identifies the adopted encoding; it does not assert a compatibility
guarantee for every future upstream release. Additional upstream decoder
capabilities, such as decimal or non-JSON extension values, do not by themselves
extend Lance's text ingestion contract.

### Payload and Array Limits

The adopted codec's size constraints apply to each JSONB payload. The outer
`large_binary` array's 64-bit offsets do not relax those constraints. Normal scan
output uses 32-bit-offset Arrow strings and is also subject to their buffer-size
limits. Lance does not currently expose a separate JSON document-size or
nesting-depth limit through this format, or guarantee ingestion up to every
representational limit of the upstream encoding.

## Text Conversion and Existing Behavior

### Parsing

The current writer invokes the codec's extended parser, not its strict
[RFC 8259](https://www.rfc-editor.org/rfc/rfc8259.html) parser. It accepts standard
JSON together with extensions including single-quoted strings, unquoted object
keys, case-insensitive null and boolean literals, NaN and Infinity, leading plus
signs and zeros, omitted integer or fractional digits around a decimal point,
and hexadecimal numbers. Empty text becomes JSON null; omitted array elements
become JSON null. Duplicate decoded object keys are rejected.

Arrow JSON conversion and SQL JSONB literals use the same text encoder. A JSON
string assigned through the dataset update path is encoded before storage;
already encoded binary input retains the separate pass-through behavior described
above.

Successful parsing produces the logical tree before serialization. Numbers that
fit the integer representations retain their integer values. Other numeric input
falls back to Float64 under the configuration described above. An integer's
exact-storage range must therefore not be presented as a maximum accepted text
value, and successful ingestion does not establish numeric precision preservation.

These are current implementation behaviors. Tightening accepted syntax or adding
a lossless-number requirement would change the text ingestion contract and is
not established by adopting the upstream encoding.

### Serialization to Arrow JSON

The dataset adapter renders stored values as compact JSON text. Object keys are
emitted in stored order, strings are quoted and escaped, and numbers are rendered
from their stored representations. JSON null produces the non-null string
`null`; an Arrow null remains an Arrow null.

Integer digits survive when the stored representation is an integer. Float64
rendering reflects any precision already lost during parsing. The current
serializer renders non-finite numbers as JSON `null`, even though their binary
representations are distinct from JSONB null. Reading such values as text and
writing them back therefore need not preserve the original binary value.

### Validation

The text input path reports parser failures. Already encoded `lance.json` input
is not validated by the text-to-binary adapter. The current text-output helper
also has permissive fallback behavior for invalid binary input, so the ability
to store or print arbitrary bytes is not evidence that they conform to this
encoding. A binary input must contain a well-formed value in the adopted upstream
representation; the adapter's pass-through behavior does not establish validity.

## Compatibility

Lance remains responsible for compatibility of the JSONB values persisted in its
tables. Updating the `jsonb` dependency does not redefine those values or select
a new payload format. Compatible codec updates can be adopted after verifying
that existing values retain their interpretation and new output remains readable
by the readers covered by Lance's compatibility contract.

The upstream release cited above identifies the current encoding reference; its
crate version is not a version field in stored data. Since Lance adds no payload
version marker, an incompatible encoding cannot be introduced merely by upgrading
the dependency. Such a change requires an explicit format evolution mechanism
under the [file-format versioning rules](../file/versioning.md) and
[table-format versioning rules](versioning.md), covering existing data and
supported readers.

Text parsing and rendering also affect compatibility. A codec update that changes
accepted input, numeric rounding, or null conversion must be evaluated against
the Lance-facing behavior described here. Changing parser precision does not
recover digits already lost in stored Float64 values.

JSON indexes are redundant structures governed by the
[index specifications](../index/index.md). Their inferred key types, query
coercions, and index versions do not alter a field's JSONB payload or define its
logical numeric precision.
