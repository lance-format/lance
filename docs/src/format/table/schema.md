# Schema Format Specification

## Overview

The schema describes the structure of a Lance table, including all fields, their data types, and metadata.
Schemas use a logical type system where data types are represented as strings that map to Apache Arrow data types.
Each field in the schema has a unique identifier (field ID) that enables robust schema evolution and version tracking.

!!! note

    Logical types are currently being simplified through discussion [#5864](https://github.com/lance-format/lance/discussions/5864).
    Proposed changes include consolidating encoding-specific variants (e.g., `large_string` and `string`, `large_binary` and `binary`)
    into single logical types with runtime optimization. Additionally, [#5817](https://github.com/lance-format/lance/discussions/5817) proposes adding
    `string_view` and `binary_view` types. This document describes the current implementation.

## Data Types

Lance supports a comprehensive set of data types that map to Apache Arrow types.
Data types are represented as strings in the schema and can be grouped into several categories.

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
| `string` | `Utf8`, `Utf8View` | Variable-length UTF-8 encoded string |
| `binary` | `Binary`, `BinaryView` | Variable-length binary data |
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

When converting between logical types and Arrow types, Lance uses the following mappings:

| Arrow Type | Logical Type Format |
|---|---|
| `Arrow::Null` | `null` |
| `Arrow::Boolean` | `bool` |
| `Arrow::Int8` to `Int64` | `int8`, `int16`, `int32`, `int64` |
| `Arrow::UInt8` to `UInt64` | `uint8`, `uint16`, `uint32`, `uint64` |
| `Arrow::Float16` | `halffloat` |
| `Arrow::Float32` | `float` |
| `Arrow::Float64` | `double` |
| `Arrow::Utf8`, `Arrow::Utf8View` | `string` |
| `Arrow::LargeUtf8` | `large_string` |
| `Arrow::Binary`, `Arrow::BinaryView` | `binary` |
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
