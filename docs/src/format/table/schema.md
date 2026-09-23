# Schema Format Specification

## Overview

The schema describes the structure of a Lance table, including all fields, their data types, and metadata.
Schemas use a logical type system where data types are represented as strings that map to Apache Arrow data types.
Each field in the schema has a unique identifier (field ID) that enables robust schema evolution and version tracking.

The meaning of a field's `logical_type` depends on whether the table sets the `FLAG_SEMANTIC_TYPES` [feature flag](versioning.md#feature-flags).
In a table with the flag, `logical_type` names a [semantic type](#semantic-types), each data file records the physical layout it encodes, and an optional output encoding selects the Arrow layout that reads return.
In a table without the flag (a legacy table) and in every data file schema, each `logical_type` string names exactly one Arrow type, as listed in [Data Types](#data-types) and the [Type Conversion Reference](#type-conversion-reference).

## Data Types

Lance supports a comprehensive set of data types that map to Apache Arrow types.
Data types are represented as strings in the schema and can be grouped into several categories.
This section lists the Arrow-mapped strings. Legacy tables and data file schemas use them with the exact Arrow types listed here.
A table that sets `FLAG_SEMANTIC_TYPES` interprets the same strings as described in [Semantic Types](#semantic-types).

### Primitive Types

| Logical Type | Arrow Type | Description |
|---|---|---|
| `null` | `Null` | Null type (no values) |
| `bool` | `Boolean` | Boolean (true/false) |
| `int8` | `Int8` | Signed 8-bit integer |
| `uint8` | `UInt8` | Unsigned 8-bit integer |
| `int16` | `Int16` | Signed 16-bit integer |
| `uint16` | `UInt16` | Unsigned 16-bit integer |
| `int32` | `Int32` | Signed 32-bit integer |
| `uint32` | `UInt32` | Unsigned 32-bit integer |
| `int64` | `Int64` | Signed 64-bit integer |
| `uint64` | `UInt64` | Unsigned 64-bit integer |

### Floating Point Types

| Logical Type | Arrow Type | Description |
|---|---|---|
| `halffloat` | `Float16` | IEEE 754 half-precision floating point (16-bit) |
| `float` | `Float32` | IEEE 754 single-precision floating point (32-bit) |
| `double` | `Float64` | IEEE 754 double-precision floating point (64-bit) |

### String and Binary Types

| Logical Type | Arrow Type | Description |
|---|---|---|
| `string` | `Utf8` | Variable-length UTF-8 encoded string |
| `binary` | `Binary` | Variable-length binary data |
| `large_string` | `LargeUtf8` | Variable-length UTF-8 string (supports large offsets) |
| `large_binary` | `LargeBinary` | Variable-length binary data (supports large offsets) |

### Decimal Types

Decimal types support arbitrary-precision numeric values. The format is: `decimal:<bit_width>:<precision>:<scale>`

| Logical Type | Arrow Type | Precision | Example |
|---|---|---|---|
| `decimal:128:P:S` | `Decimal128` | Up to 38 digits | `decimal:128:10:2` (10 total digits, 2 after decimal) |
| `decimal:256:P:S` | `Decimal256` | Up to 76 digits | `decimal:256:20:5` |

- **Precision (P)**: Total number of digits (1-38 for Decimal128, up to 76 for Decimal256)
- **Scale (S)**: Number of digits after the decimal point (0 ≤ S ≤ P)

In tables that set `FLAG_SEMANTIC_TYPES`, the canonical name of a decimal type is `decimal:<precision>:<scale>`, and the width is an output encoding (see [Semantic Types](#semantic-types)).

### Date and Time Types

| Logical Type | Arrow Type | Description |
|---|---|---|
| `date32:day` | `Date32` | Date (days since epoch) |
| `date64:ms` | `Date64` | Date (milliseconds since epoch) |
| `time32:s` | `Time32` | Time (seconds since midnight) |
| `time32:ms` | `Time32` | Time (milliseconds since midnight) |
| `time64:us` | `Time64` | Time (microseconds since midnight) |
| `time64:ns` | `Time64` | Time (nanoseconds since midnight) |
| `duration:s` | `Duration` | Duration (seconds) |
| `duration:ms` | `Duration` | Duration (milliseconds) |
| `duration:us` | `Duration` | Duration (microseconds) |
| `duration:ns` | `Duration` | Duration (nanoseconds) |

### Timestamp Types

Timestamp types represent a point in time and may include timezone information.
Format: `timestamp:<unit>:<timezone>`

- **Unit**: `s` (seconds), `ms` (milliseconds), `us` (microseconds), `ns` (nanoseconds)
- **Timezone**: IANA timezone string (e.g., `UTC`, `America/New_York`) or `-` for no timezone

Examples:
- `timestamp:us:UTC` - Microsecond precision timestamp in UTC
- `timestamp:ms:America/New_York` - Millisecond precision timestamp in America/New_York timezone
- `timestamp:ns:-` - Nanosecond precision timestamp with no timezone

### Complex Types

#### Struct Type

A struct is a container for named fields with heterogeneous types.

| Logical Type | Arrow Type | Description |
|---|---|---|
| `struct` | `Struct` | Composite type containing multiple named fields |

Struct fields are represented as child fields in the schema.

Example schema with a struct:
```protobuf
Field {
    name: "address"
    type: "struct"
    children: [
        Field { name: "street", type: "string" },
        Field { name: "city", type: "string" },
        Field { name: "zip", type: "int32" }
    ]
}
```

#### List Types

Lists represent variable-length arrays of a single type.

| Logical Type | Arrow Type | Description |
|---|---|---|
| `list` | `List` | Variable-length list of values |
| `list.struct` | `List(Struct)` | Variable-length list of struct values |
| `large_list` | `LargeList` | Variable-length list (supports large offsets) |
| `large_list.struct` | `LargeList(Struct)` | Variable-length list of struct values (large offsets) |

The element type is specified as a child field.

#### Fixed-Size List Types

Fixed-size lists have a predetermined size known at schema definition time.
Format: `fixed_size_list:<element_type>:<size>`

| Logical Type | Description | Example |
|---|---|---|
| `fixed_size_list:float:128` | Fixed-size list of 128 floats | Vector embeddings (128-dimensional) |
| `fixed_size_list:int32:10` | Fixed-size list of 10 integers | |

Special extension types:
- `fixed_size_list:lance.bfloat16:256` - Fixed-size list of bfloat16 values

#### Fixed-Size Binary Type

Fixed-size binary data with a predetermined size in bytes.
Format: `fixed_size_binary:<size>`

| Logical Type | Description | Example |
|---|---|---|
| `fixed_size_binary:16` | Fixed-size binary of 16 bytes | MD5 hash |
| `fixed_size_binary:32` | Fixed-size binary of 32 bytes | SHA-256 hash |

#### Dictionary Type

Dictionary-encoded data with separate keys and values.
Format: `dict:<value_type>:<key_type>:<ordered>`

- **Value type**: The type of dictionary values
- **Key type**: The type used for dictionary indices (typically int8, int16, or int32)
- **Ordered**: Boolean indicating if dictionary values are sorted (currently not fully supported)

Example: `dict:string:int16:false` - Dictionary-encoded strings with int16 keys

#### Map Type

Key-value pairs stored in a structured format.

| Logical Type | Arrow Type | Description |
|---|---|---|
| `map` | `Map` | Key-value pairs (currently supports unordered keys only) |

Maps have key and value types specified as child fields.

### Extension Types

Lance supports custom extension types that provide semantic meaning on top of Arrow types.

#### Blob Type

Represents large binary data stored externally.

| Logical Type | Description |
|---|---|
| `blob` | Large binary data with external storage reference |
| `json` | JSON-encoded data stored as binary |

Blob types are stored as large binary data with metadata describing storage location.

#### BFloat16 Type

Brain float (bfloat16) is a 16-bit floating point format optimized for ML.
Used within fixed-size lists: `fixed_size_list:lance.bfloat16:SIZE`

## Semantic Types

This section applies to tables that set `FLAG_SEMANTIC_TYPES`.
In these tables, `logical_type` names a **semantic type**: the value domain and query semantics that every engine must preserve.
Arrow layout choices that do not change values, such as 32- or 64-bit offsets, view layouts, dictionary encoding, and decimal width, are not part of the semantic type.

### Admission Rule

A semantic type is distinct from another exactly when it changes the value domain, the precision, or the computation semantics.
Layout and encoding choices never create a new semantic type.
By this rule `int32` and `int64` are distinct because the width bounds the values, `float` and `double` are distinct because the width sets the arithmetic precision, and `string` and `large_string` are one type.

### Type Classes

Every semantic type belongs to one of three classes.

**Unchanged types** keep the `logical_type` strings listed in [Data Types](#data-types), and each has exactly one Arrow representation:
`null`, `bool`, `int8` through `int64`, `uint8` through `uint64`, `halffloat`, `float`, `double`, `date32:day`, `date64:ms`, `time32:*`, `time64:*`, `duration:*`, `timestamp:*`, `struct`, `map`, `fixed_size_list:<element>:<size>` (including `lance.bfloat16` elements), and `fixed_size_binary:<size>`.

**Representation-only types** have several Arrow layouts that hold the same values.
A writer encodes any listed physical layout unchanged, and different data files of one column may use different layouts.

| Semantic type | Parameters | Physical layouts in data files | Additional accepted input | Output encodings (default first) |
|---|---|---|---|---|
| `string` | none | `Utf8`, `LargeUtf8`, `Dictionary<K, Utf8>`, `Dictionary<K, LargeUtf8>` | `Utf8View` | `utf8`, `large_utf8`, `utf8_view`, `dictionary:<key>:<value>` |
| `binary` | none | `Binary`, `LargeBinary`, `Dictionary<K, Binary>`, `Dictionary<K, LargeBinary>` | `BinaryView` | `binary`, `large_binary`, `binary_view`, `dictionary:<key>:<value>` |
| `list` | child field | `List`, `LargeList` | none | `list`, `large_list` |
| `decimal:<p>:<s>` | precision `p` (1 to 76), scale `s` (at most `p`) | `Decimal128(p, s)`, `Decimal256(p, s)` | none | `decimal128`, `decimal256` when `p` ≤ 38; only `decimal256` when `p` > 38 |

`K` is any integer Arrow type.
Dictionary encoding is a layout of `string` or `binary` values, not a type; values of other semantic types are not dictionary encoded in these tables.
A view input is written as a non-view layout that holds every value; the writer chooses which one.
The elements of a `list` are described by its child field, which has its own semantic type.

**Value-transforming types** convert input values to a different stored representation, and reads convert them back.
Only these types need conversion between input, storage, and output.

| Semantic type | Accepted input | Physical layout | Output encodings (default first) |
|---|---|---|---|
| `json` | `arrow.json` extension over `Utf8`, `LargeUtf8`, or `Utf8View`, validated and converted to JSONB; `lance.json` extension over `LargeBinary`, unchanged | `LargeBinary` with the `lance.json` extension (JSONB) | `arrow.json` (JSON text), `lance.json` (JSONB) |
| `blob` | as defined for blob columns | Blob v1: `LargeBinary` with `lance-encoding:blob`. Blob v2: `struct` with the `lance.blob.v2` extension | the blob read modes |

This specification only classifies `blob`.
Its representations and read modes are unchanged.

### Output Encoding

The optional field metadata entry `lance-schema:output-encoding` names the Arrow layout that a read returns for the field.
Its value must be one of the output encodings of the field's semantic type:

| Value | Arrow layout returned | Semantic type |
|---|---|---|
| `utf8`, `large_utf8`, `utf8_view` | `Utf8`, `LargeUtf8`, `Utf8View` | `string` |
| `binary`, `large_binary`, `binary_view` | `Binary`, `LargeBinary`, `BinaryView` | `binary` |
| `dictionary:<key>:<value>` | `Dictionary<key, value>`. `<key>` is one of `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`. `<value>` is `utf8` or `large_utf8` for `string`, and `binary` or `large_binary` for `binary` | `string`, `binary` |
| `list`, `large_list` | `List`, `LargeList` of the child field's output layout | `list` |
| `decimal128` | `Decimal128(p, s)`; valid only when `p` ≤ 38 | `decimal:<p>:<s>` |
| `decimal256` | `Decimal256(p, s)` | `decimal:<p>:<s>` |
| `arrow.json`, `lance.json` | JSON text as `Utf8` with the `arrow.json` extension; JSONB as `LargeBinary` with the `lance.json` extension | `json` |

For nested types, each child field carries its own entry.
A reader picks the output layout of a field in this order:

1. a per-request override from the caller;
2. the field's `lance-schema:output-encoding` entry;
3. the default output encoding of the semantic type, which is the first one listed for the type.

When a writer creates a column (table creation, add column, or alter type), it records the Arrow layout of the input as the field's output encoding if that layout differs from the default.
A `LargeUtf8` input to a new `string` column therefore reads back as `LargeUtf8`.
Appends never change the entry.
Changing the entry is a metadata-only schema update.

The output encoding is advisory.
It never affects which values are stored, schema compatibility, or correctness.
A writer rejects a schema update that sets a value that is not valid for the field's semantic type and parameters.
A reader that does not recognize a value, or cannot produce that layout, uses the default output encoding instead.

### Extension Semantic Types

An extension semantic type is a field of a core semantic type that carries `ARROW:extension:name` and, optionally, `ARROW:extension:metadata`.
The core type is its storage type.
For example, a field with `logical_type` `fixed_size_list:float:4` and `ARROW:extension:name` `example.bbox` is a bounding box stored as four floats.
A field carries one extension name, so an extension cannot be layered on a type that already uses one (`json`, Blob v2, or `lance.bfloat16` elements).

- **Value preservation:** an extension type annotates values and does not transform them. Stored values are exactly the values of the storage type, so a reader that does not recognize the extension returns correct data of the storage type.
- **Metadata preservation:** implementations must keep the extension name and metadata unchanged through schema evolution, compaction, and any other rewrite, including for extensions they do not recognize.
- **Reserved prefix:** extension names starting with `lance.` are reserved for Lance. A Lance-owned extension may transform values only when a data storage version or a feature flag gates it; for example, `lance.blob.v2` requires data storage version 2.2. Other extensions should use a project prefix such as `lancedb.` or a reverse-DNS name.
- **No flag:** adding an extension semantic type requires no feature flag and no change to this specification.

### Schema Compatibility

Two fields are compatible for append, merge, and update when their semantic types and semantic parameters are equal.
Output encodings, physical layouts, and dictionary key types are not compared.
Field IDs, nullability, and nested structure follow the existing rules, and the children of nested fields are compared field by field.
As a result:

- An append that supplies any accepted input for a column's semantic type succeeds.
- An append that supplies `Decimal256(10, 2)` to a `decimal:10:2` column succeeds.
- An append that supplies `Decimal128(12, 2)` to a `decimal:10:2` column fails, because the value domain differs. Changing the precision is a type change, not a compatible append.

### Data File Schemas

A data file schema records the exact physical layout encoded in that file, using the Arrow-mapped strings from [Data Types](#data-types), such as `large_string`, `dict:string:int16:false`, or `decimal:128:10:2`.
These strings keep their meaning inside data files, and output encodings do not apply to data file schemas.
Different data files of one column may record different physical layouts of the column's semantic type.

The table schema is the only source of semantics.
Readers resolve semantics through field IDs in the table schema and never infer them from the physical layout of a data file.
The encodings, footer, page metadata, and global buffers of data files are unchanged.

Writers and readers keep these invariants:

- **Writer boundary:** every array handed to the encoder has the exact Arrow type and extension metadata recorded for its field in the data file schema. Representation-only inputs satisfy this by passing through unchanged, except that view inputs are converted to a non-view layout. Value-transforming inputs satisfy it after conversion, applied recursively through `struct`, `list`, and `map` children.
- **Lossless conversion:** conversions between an accepted input, a physical layout, and an output layout are lossless. An operation that would lose data fails instead of truncating it.
- **Metadata only:** output encodings and extension metadata never require rewriting data files.

### Failure Semantics

| Condition | Result |
|---|---|
| The input Arrow type is not an accepted input for the field's semantic type | The write fails before any data file is written. The error names the field path, the semantic type, and the input type. |
| A `json` input value is not valid JSON | The write fails. The error names the field path and the row. |
| An array reaches the encoder in a layout different from the one recorded in the data file schema | Internal invariant violation. The write fails with an error naming the field path, the expected layout and extension, and the actual ones. |
| The requested output layout cannot hold a value, for example a `utf8` output whose values exceed the 32-bit offset limit | The read fails with an error naming the field, the value size, and the requested output encoding. The caller can request `large_utf8` or `utf8_view` instead. Readers may emit smaller batches so that values fit, but must never truncate. |
| A writer sets `lance-schema:output-encoding` to a value that is not valid for the field's semantic type or parameters, for example `decimal128` for precision 40 | The schema update fails. |
| A reader encounters an unrecognized or unsupported `lance-schema:output-encoding` value | The reader uses the default output encoding of the semantic type. |

### Legacy Aliases

Legacy tables name layouts directly in `logical_type`.
Each such string is an alias for a canonical semantic type plus the output encoding that reproduces its Arrow type:

| Legacy `logical_type` | Canonical type | Implied output encoding |
|---|---|---|
| `string`, `binary`, `list` | same | none |
| `list.struct` | `list`, whose child field is a `struct` | none |
| `large_string` | `string` | `large_utf8` |
| `large_binary` | `binary` | `large_binary` |
| `large_list`, `large_list.struct` | `list` | `large_list` |
| `dict:<value>:<key>:false` with a `string`, `large_string`, `binary`, or `large_binary` value | the canonical type of `<value>` | `dictionary:<key>:<value layout>`, where the value layout is the output encoding `<value>` implies or the default |
| `decimal:128:<p>:<s>` | `decimal:<p>:<s>` | none, because `p` ≤ 38 and the default is already `decimal128` |
| `decimal:256:<p>:<s>` | `decimal:<p>:<s>` | `decimal256` when `p` ≤ 38; none otherwise |

Writers of tables that set `FLAG_SEMANTIC_TYPES` write only canonical names in the table schema.
Readers of these tables still interpret a legacy alias as its canonical type with the implied output encoding; a `lance-schema:output-encoding` entry on the same field takes precedence over the implied one.

## Field IDs

Field IDs are unique integer identifiers assigned to each field in a schema.
They are essential for robust schema evolution, as they allow fields to be renamed or reordered without breaking references.

### Field ID Assignment

**Initial assignment (depth-first order):**
When a table is created, field IDs are assigned to all fields in depth-first order, starting from 0.

Nested fields are linked via the `parent_id` field in the protobuf message. For example, if field "c" (id: 2) is a struct containing fields "x", "y", "z", those child fields will have `parent_id: 2`. Top-level fields have `parent_id: -1`.

Example with nested structure:
```
Field order: a, b, c.x, c.y, c.z, d

Assigned IDs with parent relationships:
- a: 0 (parent_id: -1)
- b: 1 (parent_id: -1)
- c: 2 (parent_id: -1, struct type)
- c.x: 3 (parent_id: 2)
- c.y: 4 (parent_id: 2)
- c.z: 5 (parent_id: 2)
- d: 6 (parent_id: -1)
```

Note: A `parent_id` of -1 indicates a top-level field. For nested fields, `parent_id` references the ID of the parent field. Child fields reference their parent via `parent_id` rather than being stored as separate "children" arrays in the protobuf message (though the Rust in-memory representation maintains a children vector for convenience).

**New field assignment (incremental):**
When fields are added later (e.g., through schema evolution), they receive the next available ID
incrementally. This preserves the history of field additions.

### Field ID Properties

- **Immutable**: Once assigned, a field's ID never changes
- **Unique**: Each field within a table has a unique ID
- **Stable**: IDs are preserved across schema evolution operations
- **Sparse**: Field IDs may not form a contiguous sequence after schema evolution

### Using Field IDs

When referencing fields internally within the format, use the field ids rather than field names or positions.

## Field Metadata

Fields can carry additional metadata as key-value pairs to configure encoding, primary key behavior, and other properties.

### Primary Key Metadata

Primary key configuration is handled by two protobuf fields in the Field message:
- **unenforced_primary_key** (bool): Whether this field is part of the primary key
- **unenforced_primary_key_position** (uint32): Position in primary key ordering (1-based for ordered, 0 for unordered)

For detailed discussion on primary key configuration, see [Unenforced Primary Key](index.md#unenforced-primary-key) in the table format overview.

### Clustering Key Metadata

Clustering key configuration uses a single protobuf field in the Field message:
- **unenforced_clustering_key_position** (uint32): 1-based position in clustering key ordering. 0 means not a clustering key field.

Clustering keys hint at the physical ordering of data within a table. Unlike primary keys,
clustering key fields may be nullable. This metadata enables query engines to perform
optimizations such as storage-partitioned joins.

### Encoding Metadata

Column encoding configurations are specified with the `lance-encoding:` prefix.
See [File Format Encoding Specification](../file/encoding.md) for complete details on available encodings.

### Output Encoding Metadata

In tables that set `FLAG_SEMANTIC_TYPES`, the `lance-schema:output-encoding` entry names the Arrow layout that reads return for the field.
See [Output Encoding](#output-encoding) for its values and rules.
Legacy tables and data file schemas do not interpret this entry.

### Arrow Extension Type Metadata

Custom Arrow extension types may have metadata under the `ARROW:extension:` namespace
(e.g., `ARROW:extension:name`).

## Schema Protobuf Definition

The schema is serialized using protobuf messages. Key messages include:

### Field Message

```protobuf
%%% proto.message.lance.file.Field %%%
```

The Field message contains:
- **id**: Unique field identifier (int32)
- **name**: Field name (string)
- **type**: Field type enum (PARENT, REPEATED, or LEAF)
- **logical_type**: Logical type string representation (string) - e.g., "int64", "struct", "list"
- **nullable**: Whether the field can be null (bool)
- **parent_id**: Parent field ID for nested fields; -1 for top-level fields (int32)
- **metadata**: Key-value pairs for additional configuration (map<string, bytes>)
- **unenforced_primary_key**: Whether this field is part of the primary key (bool)
- **unenforced_primary_key_position**: Position in primary key ordering (uint32, 0 = unordered)

### Schema Message

The complete schema is represented as a collection of top-level fields plus metadata.

## Schema Evolution

Field IDs enable efficient schema evolution:

- **Add Column**: Assign a new field ID and add to schema
- **Drop Column**: Remove field from schema; its ID may be reused in some systems
- **Rename Column**: Change field name; ID remains the same
- **Reorder Columns**: Change field order in schema; IDs remain the same
- **Type Evolution**: Data type can be changed. This might require rewriting the column in the data, depending on how the type was changed.

The use of field IDs ensures that data files can be correctly interpreted even as the schema changes over time.

## Example Schemas

The examples below use a simplified representation of the field structure. In the actual protobuf format, `type` refers to the field type enum (PARENT/REPEATED/LEAF) and `logical_type` contains the data type string representation.

### Simple Table

```
Field {
    id: 0
    name: "id"
    logical_type: "int64"
    nullable: false
    parent_id: -1
}
Field {
    id: 1
    name: "name"
    logical_type: "string"
    nullable: true
    parent_id: -1
}
Field {
    id: 2
    name: "created_at"
    logical_type: "timestamp:us:UTC"
    nullable: true
    parent_id: -1
}
```

### Nested Structure

```
Field {
    id: 0
    name: "id"
    logical_type: "int64"
    nullable: false
    parent_id: -1  // Top-level field
}
Field {
    id: 1
    name: "user"
    logical_type: "struct"
    nullable: true
    parent_id: -1  // Top-level field
}
Field {
    id: 2
    name: "name"
    logical_type: "string"
    nullable: true
    parent_id: 1  // Nested under "user" struct (id: 1)
}
Field {
    id: 3
    name: "email"
    logical_type: "string"
    nullable: true
    parent_id: 1  // Nested under "user" struct (id: 1)
}
Field {
    id: 4
    name: "tags"
    logical_type: "list"
    nullable: true
    parent_id: -1  // Top-level field
}
Field {
    id: 5
    name: "item"
    logical_type: "string"
    nullable: true
    parent_id: 4  // Nested under "tags" list (id: 4)
}
```

### With Vector Embeddings

```
Field {
    id: 0
    name: "id"
    logical_type: "int64"
    nullable: false
    parent_id: -1  // Top-level field
    unenforced_primary_key: true
    unenforced_primary_key_position: 1  // Ordered position in primary key
}
Field {
    id: 1
    name: "text"
    logical_type: "string"
    nullable: true
    parent_id: -1  // Top-level field
}
Field {
    id: 2
    name: "embedding"
    logical_type: "fixed_size_list:lance.bfloat16:384"
    nullable: true
    parent_id: -1  // Top-level field
}
```

## Type Conversion Reference

When converting between logical types and Arrow types, Lance uses the following mappings.
Legacy tables and data file schemas use them in both directions.
Tables that set `FLAG_SEMANTIC_TYPES` interpret these strings as described in [Legacy Aliases](#legacy-aliases).

| Arrow Type | Logical Type Format |
|---|---|
| `Arrow::Null` | `null` |
| `Arrow::Boolean` | `bool` |
| `Arrow::Int8` to `Int64` | `int8`, `int16`, `int32`, `int64` |
| `Arrow::UInt8` to `UInt64` | `uint8`, `uint16`, `uint32`, `uint64` |
| `Arrow::Float16` | `halffloat` |
| `Arrow::Float32` | `float` |
| `Arrow::Float64` | `double` |
| `Arrow::Utf8` | `string` |
| `Arrow::LargeUtf8` | `large_string` |
| `Arrow::Binary` | `binary` |
| `Arrow::LargeBinary` | `large_binary` |
| `Arrow::Decimal128(p, s)` | `decimal:128:p:s` |
| `Arrow::Decimal256(p, s)` | `decimal:256:p:s` |
| `Arrow::Date32` | `date32:day` |
| `Arrow::Date64` | `date64:ms` |
| `Arrow::Time32(TimeUnit)` | `time32:s`, `time32:ms` |
| `Arrow::Time64(TimeUnit)` | `time64:us`, `time64:ns` |
| `Arrow::Timestamp(unit, tz)` | `timestamp:unit:tz` |
| `Arrow::Duration(unit)` | `duration:s`, `duration:ms`, `duration:us`, `duration:ns` |
| `Arrow::Struct` | `struct` |
| `Arrow::List(Element)` | `list` or `list.struct` if element is Struct |
| `Arrow::LargeList(Element)` | `large_list` or `large_list.struct` |
| `Arrow::FixedSizeList(Element, Size)` | `fixed_size_list:type:size` |
| `Arrow::FixedSizeBinary(Size)` | `fixed_size_binary:size` |
| `Arrow::Dictionary(KeyType, ValueType)` | `dict:value_type:key_type:false` |
| `Arrow::Map` | `map` |
