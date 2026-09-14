# Blob v2

A Blob v2 value is a nullable byte sequence. Its bytes may reside in a Lance data
file, a Lance-managed sidecar object, or an externally managed object. This
specification documents the implemented logical input contract, stored descriptor,
object resolution, and ownership rules. API examples and configuration belong in the
[Blob Columns guide](../../guide/blob.md).

## Logical Input

A Blob v2 field carries the Arrow field metadata
`ARROW:extension:name = "lance.blob.v2"`. The metadata belongs to the blob field
itself, including when that field is a child of a struct or a list element.

Writers accept the following two logical schemas. `blob` is an arbitrary column
name; the child names, order, and data types are part of the input contract.

```python
import pyarrow as pa

minimal_input_schema = pa.schema([
    pa.field(
        "blob",
        pa.struct([
            pa.field("data", pa.large_binary(), nullable=True),
            pa.field("uri", pa.utf8(), nullable=True),
        ]),
        nullable=True,
        metadata={"ARROW:extension:name": "lance.blob.v2"},
    ),
])

complete_input_schema = pa.schema([
    pa.field(
        "blob",
        pa.struct([
            pa.field("data", pa.large_binary(), nullable=True),
            pa.field("uri", pa.utf8(), nullable=True),
            pa.field("position", pa.uint64(), nullable=True),
            pa.field("size", pa.uint64(), nullable=True),
        ]),
        nullable=True,
        metadata={"ARROW:extension:name": "lance.blob.v2"},
    ),
])
```

The outer field may instead be non-nullable. `data` and `uri` must be declared
nullable. In the complete schema, each of `position` and `size` may instead be
declared non-nullable; the per-value rules below still apply. Additional children,
reordered children, and substitutions such as `Binary` for `LargeBinary` or
`LargeUtf8` for `Utf8` are not accepted logical shapes. Child metadata does not
change shape recognition.

### Values and Ranges

For each non-null blob struct:

- Exactly one of `data` and `uri` must be non-null. Both present and both absent
  are invalid, including when `data` is an empty byte sequence.
- `data` supplies the value's bytes. An empty byte sequence is a valid empty
  blob, distinct from a null blob.
- `uri` supplies a non-empty external object locator. Without a range it denotes
  the complete object, which may itself be empty.
- `position` and `size` must both be present or both be null. A range requires
  `uri`, must have `size > 0`, and must not overflow `UInt64` when computing
  `position + size`. It selects the half-open byte interval
  `[position, position + size)` in the external object.

In the minimal shape, range fields are absent and have the same meaning as null
range fields in the complete shape. These logical range fields never select a
location in Lance-managed storage.

A null blob is represented by the outer struct's validity. Children beneath a
null struct are not values and must not be interpreted as bytes, URIs, or ranges.
The outer field's nullability determines whether a null blob is allowed.

### Nesting and Schema Preservation

Blob fields may appear at the top level or beneath `Struct`, `List`, and
`LargeList` fields, including combinations of these containers. Container
validity, list element order, and list boundaries retain their Arrow meanings.
A null list, an empty list, a list containing a null blob, and a list containing
an empty blob are distinct values. Payload materialization replaces the blob
leaf with `LargeBinary` while preserving the surrounding logical structure.

A plain binary field or a list of binary values does not become Blob v2 merely
because its values are large. It needs an explicitly marked blob field with an
accepted logical shape. Other container types are outside this logical input
contract.

The table schema retains the accepted logical shape, child metadata, and
nullability across create, append, and merge-insert writes. It is separate from
the data file's descriptor schema. Descriptor scans expose the stored
representation described below; they do not redefine the table's logical schema.

## Stored Descriptor

Each blob value occupies one descriptor in a packed struct column. The canonical
descriptor schema is:

```python
import pyarrow as pa

descriptor_schema = pa.schema([
    pa.field(
        "blob",
        pa.struct([
            pa.field("kind", pa.uint8(), nullable=False),
            pa.field("position", pa.uint64(), nullable=False),
            pa.field("size", pa.uint64(), nullable=False),
            pa.field("blob_id", pa.uint32(), nullable=False),
            pa.field("blob_uri", pa.utf8(), nullable=False),
        ]),
        nullable=True,
        metadata={
            "lance-encoding:blob": "true",
            "lance-encoding:packed": "true",
        },
    ),
])
```

The outer name and nullability follow the blob field. The file descriptor records
this physical shape, while the table manifest records the logical shape. The
blob field maps to one physical column; its logical input children are not
separate payload columns.

Field names, order, data types, and the numeric `kind` values are durable format
contracts. Struct packing uses the file version's
[structural encoding](../file/encoding.md#struct-packing); it does not prescribe
a fixed byte width or byte offset for each descriptor child. The packed struct
encoding and the `Packed` payload storage kind are independent concepts.

All positions and sizes are byte counts. Positions are relative to the beginning
of the selected object, with an exclusive end of `position + size`. The meaning
of `blob_id` depends on `kind`.

### Inline: `kind = 0`

The bytes occupy an out-of-line buffer in the owning `.lance` data file.
`position` is an absolute offset in that file and `size` is the payload length.
Writers set `blob_id = 0` and `blob_uri = ""`.

An Inline value with `size = 0` is empty regardless of its position. A valid
all-zero descriptor is therefore an empty blob, not a null marker. Payload bytes
are addressed directly; descriptor page compression does not change their byte
offsets or introduce payload decompression.

### Packed: `kind = 1`

`blob_id` identifies a managed sidecar object in the owning data file's namespace.
`position` and `size` select a byte interval in that object. Writers set
`blob_uri = ""`.

A packed sidecar contains raw payload bytes. It has no Lance header, footer,
per-value length prefix, or separate offset index. Value boundaries come from
the descriptors. Multiple values can refer to the same object, and a zero-length
interval denotes a valid empty value. Readers must use the recorded intervals
without inferring boundaries or payload order from row order.

### Dedicated: `kind = 2`

`blob_id` identifies a managed sidecar containing a complete raw payload.
Writers set `position = 0`, `size` to the complete object length, and
`blob_uri = ""`. The object has no Lance header or footer. Readers read `size`
bytes from offset zero.

### External: `kind = 3`

`blob_uri` and `blob_id` identify an externally managed object:

- `blob_id = 0`: `blob_uri` is an absolute URI, interpreted by its object store.
- `blob_id > 0`: `blob_id` is a manifest base path **ID**, not an array index or
  a managed sidecar ID. Look up the entry with that ID in `manifest.base_paths`
  and append `blob_uri` as an object path relative to that base. Use that base's
  object store and credentials. Do not add a `data/` prefix.

For `size > 0`, read the recorded interval starting at `position`. A stored
`size = 0` is a sentinel: resolve the object's complete length and use that as
the read length, retaining the recorded `position`. It does not mean an explicit
empty range or "the bytes remaining after position". Writers representing a
complete external object set both `position = 0` and `size = 0`.

This sentinel belongs to the stored format. The logical input restriction
`size > 0` for explicit ranges must not be used to reject these descriptors.

### Null Descriptors

Current writers encode blob nullness in the outer descriptor struct validity,
using the file's repetition and definition levels. Child values beneath a null
descriptor are ignored. NULL must never be inferred from `position`, `size`,
`blob_id`, or `blob_uri` alone.

Released descriptors also represent nullness with a nullable `kind` child.
Readers must recognize a null blob when either the outer struct or `kind` is
null, and must accept released descriptor schemas with nullable children.
For a non-null blob, the children used by its storage kind must contain values.
Unrecognized numeric kinds must produce an error.

## Managed Sidecar Resolution

Inline, Packed, and Dedicated descriptors need the identity of their owning
data file. A descriptor copied out of a scan does not contain that identity.
Locate the row's fragment, then the `DataFile` supplying the blob field through
the fragment's field-to-file mapping.

Resolve that data file's object store and data directory using its own
`DataFile.base_id`, following the [storage layout specification](layout.md#file-metadata-base-references).
Without a base ID, the directory is `<dataset_root>/data`. A base representing
a dataset root also adds `data/`; a non-root base directly names the data
directory. A blob column may reside in a different base from other columns in
the same row.

Let `data_file_key` be the final component of `DataFile.path` with its `.lance`
suffix removed. The resolved layout is:

```text
<data_directory>/
    <DataFile.path>                         # Lance data file
    <data_file_key>/
        <encoded_blob_id>.blob              # Packed or Dedicated raw object
```

For a managed sidecar, `blob_id` is a nonzero unsigned 32-bit integer. Encode its
file name by reversing all 32 bits and writing the result as exactly 32 binary
digits, including leading zeros, followed by `.blob`. Equivalently, emit the
original ID's bits from least significant to most significant. Thus ID 1 names
`10000000000000000000000000000000.blob`, and ID 2 names
`01000000000000000000000000000000.blob`.

Managed IDs identify objects within one data file namespace, shared by all blob
columns in that file. A writer must not assign the same ID to different objects
in that namespace. IDs are neither row IDs nor globally unique blob identities;
different data files may reuse the same numeric ID.

## Writer Policy and External Ownership

For supplied bytes, the writer chooses Inline, Packed, or Dedicated placement.
Column thresholds and sidecar rollover limits control this choice; they are not
needed to interpret existing descriptors. Readers must use `kind` and the
recorded location even if a value's length would lead the reader's own writer to
choose another kind.

For a logical `uri`, reference mode records an External descriptor without
copying its bytes. The dataset writer matches registered non-dataset-root bases
by object store and path components, preferring the longest matching base path.
It stores the remaining path and the matching base ID. When explicitly allowed
to reference objects outside those bases, it can instead store an absolute URI
with `blob_id = 0`. Otherwise an unmatched URI is a write error.

Ingest mode reads the complete external object or the requested range and writes
those bytes into managed storage. The resulting managed descriptor's position
is relative to its destination object, not the source URI. Reference and ingest
are writer choices; there is no additional persisted ingest storage kind.

External descriptors contain no content hash or object version that freezes
the referenced bytes. The external owner is responsible for preserving the
object and referenced range for as long as readers need them. Table snapshots
preserve the reference, not a copy of externally mutable content. Storage
credentials are runtime configuration, not descriptor fields.

## Snapshot Lifetime and Rewrites

Managed payloads and their descriptors must be complete before publishing a
manifest that makes them visible. Staging may leave uncommitted objects;
successful payload uploads alone do not create a visible table version.
Visibility is established by a committed manifest. Publication and cleanup follow
the table's [transaction](transaction.md) and snapshot lifetime rules.

Managed sidecars are associated with their owning data file rather than listed
individually in the manifest. Preserve them while that file is needed by any
retained snapshot. Deleting a row does not authorize deleting a sidecar that
other rows or snapshots can still read. Cleanup of table-managed data does not
transfer ownership of External objects to Lance.

Rewriting rows into another data file must preserve blob bytes, nullness, and
external reference semantics. Inline positions must address the destination
data file. Packed and Dedicated descriptors must resolve in the destination
file's namespace, with their payload objects made available there. Copying a
descriptor alone cannot relocate its payload. Placement may change during a
rewrite without changing the logical value.

External references can be preserved during compaction without ingesting their
payloads. If manifest base IDs change, remap External `blob_id` values so they
continue to identify the same object. Managed `blob_id` values remain local to
their data file namespace and do not use this base-ID mapping.

## Validation and Compatibility

Writers must reject logical shapes and values outside the input contract, with
errors identifying the field and invalid value or combination. Reference writes
need not read an external object to establish its existence or length. A
successful reference write therefore does not guarantee the referenced bytes
are available; failures may be reported when materializing the payload.

Readers must resolve the selected object and range using the rules above. An
unknown base ID, unrecognized storage kind, arithmetic overflow, unavailable
object, or truncated requested payload is an error, not a null or empty value.
Partial reads within a blob use offsets relative to that blob, which are added
to its resolved object position with checked arithmetic.

Blob v2 requires Lance file format 2.2 or a later format retaining this
representation. Blob v2's name is independent of the Lance library version.
The earlier two-child `position`/`size` blob layout remains a separate read
compatibility surface; see [Blob Page Layout](../file/encoding.md#blob-page-layout).
Its null encoding must not be applied to Blob v2.

The descriptor schema, kind values, and sidecar naming in this specification
are present in released Lance 11.0.0. That writer was more permissive about
ambiguous `data`/`uri` values and external range inputs than the current logical
input contract. Tightened writer validation does not change how already stored
descriptors are read, including nullable descriptor children and the External
zero-size sentinel.

The stable representation must preserve backward and forward compatibility.
Changes to the enclosing file version and experimental encodings remain subject
to the [file format versioning rules](../file/versioning.md).
