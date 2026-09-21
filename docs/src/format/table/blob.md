# Blob v2

A Blob v2 value is a nullable byte sequence. Its bytes may reside in a Lance data
file, an independently addressed Lance-managed object, or an externally managed
object. This specification defines the logical input contract, stored descriptor,
object resolution, and snapshot ownership rules. Packed and Dedicated sidecars
remain readable under their existing contracts. API examples and configuration
belong in the [Blob Columns guide](../../guide/blob.md).

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

This legacy kind identifies a Lance-owned sidecar in the owning data file's
namespace. `blob_id` selects the sidecar, and `position` and `size` select a byte
interval within it. `blob_uri = ""`. Multiple values can refer to the same object,
and a zero-length interval denotes an empty value.

### Dedicated: `kind = 2`

This legacy kind uses `blob_id` to identify a Lance-owned sidecar containing a
complete raw payload. `position = 0`, `size` is the complete object length, and
`blob_uri = ""`. Readers read `size` bytes from offset zero.

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

### Managed: `kind = 4`

`blob_id` is an explicit base path **ID** in the snapshot's `manifest.base_paths`.
Every `UInt32` value, including zero, is a possible ID; no value means an implicit
dataset root. The ID must have a binding in that snapshot.

`blob_uri` is a non-empty canonical object path relative to the bound base.
It must not contain a URI scheme delimiter (`://`), backslashes, a leading or
trailing slash, empty path components, or `.` or `..` components. Append it
directly to the base path and use that base's object store and credentials.
Do not add `data/`, `_blobs/`, or any other prefix during resolution, regardless
of `BasePath.is_dataset_root`.

`position` and `size` select the exact byte interval in the object. Their sum
must not overflow `UInt64`. `size = 0` means an empty blob; it never requests an
object-length lookup. Multiple descriptors may reference different intervals in
the same object. A Managed descriptor and its base binding fully identify the
payload without the identity, path, or continued existence of the data file
that originally produced it.

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

## Managed Object Layout

New Managed payload objects use independent names:

```text
<base_path>/
    _blobs/
        <uuid>.blob
```

These objects contain raw bytes with no Lance header, footer, per-value length
prefix, or separate offset index. A writer may pack several values into one
object or dedicate an object to one value. Descriptors supply all value
boundaries; readers must not infer boundaries or payload order from row order.

An existing Packed or Dedicated sidecar may be adopted as a Managed object
without moving or copying it. Its descriptor records the sidecar's complete
path relative to its explicit base, including `data/` when that base is a
dataset root. Such paths remain valid after the original `.lance` data file is
removed. Readers must not require Managed paths to start with `_blobs/` or
reconstruct them from the current data file's name.

Base IDs belong to the snapshot's manifest namespace. A writer must publish each
required binding with the descriptors that use it. The default dataset root also
needs an explicit binding. Reusing a binding for that root requires both its
path and `is_dataset_root = true` to match; an entry with the same path but
`is_dataset_root = false` has different file-routing semantics.

An existing ID must not be rebound to a different path or root interpretation
while descriptors or other file metadata still use that binding. When moving
descriptors to another base-ID namespace, preserve their resolved locations and
remap IDs as needed. A concurrent binding conflict must fail the commit rather
than silently redirect already written descriptors.

## Legacy Sidecar Resolution

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

For a Packed or Dedicated sidecar, `blob_id` is a nonzero unsigned 32-bit integer.
Encode its file name by reversing all 32 bits and writing the result as exactly 32 binary
digits, including leading zeros, followed by `.blob`. Equivalently, emit the
original ID's bits from least significant to most significant. Thus ID 1 names
`10000000000000000000000000000000.blob`, and ID 2 names
`01000000000000000000000000000000.blob`.

These IDs identify objects within one data file namespace, shared by all blob
columns in that file. One ID must not identify different objects in that
namespace. They are neither manifest base IDs nor globally unique blob
identities; different data files may reuse the same numeric ID. Both sidecar
kinds contain raw payload bytes with the same absence of framing as Managed
objects.

## Writer Policy and External Ownership

For supplied bytes, current table writers choose Inline or Managed placement.
Column thresholds and object rollover limits control inlining, packing, and
dedicating an object to one value. Both packed and dedicated independent objects
use `kind = 4`; these placement choices do not produce the legacy kinds 1 and 2.
Readers must use the recorded kind, location, and range regardless of their own
writer's placement policy.

For a logical `uri`, reference mode records an External descriptor without
copying its bytes. The dataset writer matches registered non-dataset-root bases
by object store and path components, preferring the longest matching base path.
It stores the remaining path and the matching base ID. When explicitly allowed
to reference objects outside those bases, it can instead store an absolute URI
with `blob_id = 0`. Otherwise an unmatched URI is a write error.

Ingest mode reads the complete external object or the requested range and writes
those bytes using Inline or Managed placement. The resulting descriptor's
position is relative to its destination object, not the source URI. Reference
and ingest are writer choices; there is no additional persisted ingest kind.

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
Published Lance-owned payload objects must remain unchanged while protected
snapshots reference them; updating a value must not overwrite a shared object.

| Storage kind | Reads | Compaction | Garbage collection |
|---|---|---|---|
| Inline | Resolve the owning data file and read its recorded interval. | Copy the bytes and record their destination location. | Payload lifetime follows the data file. |
| Packed / Dedicated | Resolve the sidecar through the owning data file's base, stem, and local ID. | Adopt the original object with a Managed descriptor; payload bytes stay in place. | Retain sidecars for protected legacy data files, and retain any object referenced by Managed descriptors independently of that file. |
| External | Resolve the absolute URI or explicit base reference. | Preserve the external reference without ingesting bytes. | Lance does not own or delete the referenced object. |
| Managed | Resolve the explicit base ID and relative path; read the recorded interval. | Preserve the object and range while rewriting the descriptor. | Retain the whole object while any protected snapshot references it. |

### Compaction

Rewriting rows must preserve bytes, nullness, nesting, and external reference
semantics. Managed compaction rewrites descriptors without reading or copying
their payloads. The output data file may use another storage route: the payload
does not have to exist under that destination route. Its explicit binding and
path continue to identify the original object.

For legacy sidecars, compaction first resolves the original object using the
rules above, then records that location as a Managed descriptor. Packed retains
its position and size; Dedicated uses position zero and its recorded size.
The new descriptor's `blob_id` is a manifest base ID, replacing the old local
sidecar ID. Copying an unchanged legacy descriptor into another data file would
change its address and is not a valid rewrite. Inline bytes still need copying
because they reside inside the data file being replaced.

### Garbage Collection

A protected snapshot is one retained by the table's cleanup policy, including
the latest version, retained historical versions, tags, and branch references.
Managed object liveness is the union of references from live rows in all
protected snapshots. Follow each snapshot's schema, descriptor files, deletion
state, and base bindings, including nested blob fields. Read descriptors to
discover references; payload reads are unnecessary. The manifest does not need
a separate per-object inventory or a reference to the original producing data
file.

Retain an entire object if any referenced range is live, even when other ranges
in that object belong only to deleted rows. Compaction does not reclaim those
unused ranges. Deleting a row or the original data file does not authorize
deleting an object still referenced elsewhere. Legacy sidecar retention by data
file and Managed retention by explicit reference must both be honored during
the transition.

Cleanup may delete an unreferenced object only within its own storage namespace
and under the table's rules for expired versions, uncommitted objects, and
in-progress writes. A base reference into another dataset does not give the
referencing dataset authority to delete that dataset's objects. Cleanup must
establish the references needed to protect retained snapshots before deleting
objects; it must stop if a read failure or invalid descriptor prevents this.
Missing source manifests do not prove that descendant branches have released
their references.

### Clones

A shallow clone preserves object locations and base bindings. It relies on the
source snapshot's existing retention contract, such as a retained source tag;
creating the clone does not independently extend the source objects' lifetime.
Its cleanup must not delete objects owned by the source dataset.

A deep clone copies Lance-owned payloads into an independent destination
namespace and rewrites Managed references to destination bindings and paths,
preserving their byte ranges. It must remain readable after the source dataset
is removed. External references remain externally owned in either clone mode;
cloning does not ingest or freeze their contents.

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

### Table Capability

Managed descriptors use the same five-field schema in Lance file formats 2.2
and 2.3. They require the table capability `FLAG_MANAGED_BLOBS = 2048` (`1 << 11`)
in **both** `reader_feature_flags` and `writer_feature_flags`. A manifest with
only one bit set is invalid. Clients without this capability must reject access
to flagged snapshots; interpreting kind 4 as a legacy kind is not permitted.

Publishing new data files containing Blob v2 fields activates both bits, even
if the new values are all Inline or null. The commit must atomically publish
the data-file references, capability bits, and required base bindings after the
payloads are complete. Metadata-only updates and deletion-vector changes to an
unflagged table do not activate the capability.

Once activated, both bits must remain set in subsequent versions, including
restore of a snapshot written before activation. Restoring old data does not
authorize clients without Managed support to perform maintenance. See
[Format Versioning](versioning.md) for the table feature-flag rules.

All maintenance after activation requires a Managed-aware client. Older
released clients may bypass capability checks when running cleanup from an
unflagged historical snapshot or a cached handle. Such cleanup can delete
adopted sidecars after their original data files disappear. Capability checks
on normal snapshot opens do not make those old maintenance paths safe.

### Released Formats

Blob v2 requires file format 2.2 or a later format retaining this representation;
its name is independent of the Lance library version. Lance 11.0.0 contains the
five-field descriptor and kinds 0 through 3. Their schema, null handling,
sidecar naming, and External zero-size sentinel retain their existing meanings.
Managed adds kind 4 under the paired table capability rather than changing
those released meanings. Tightened logical input validation must not be applied
retroactively to stored descriptors.

The earlier two-child `position`/`size` blob layout remains a separate read
compatibility surface; see [Blob Page Layout](../file/encoding.md#blob-page-layout).
Its null encoding must not be applied to Blob v2. Stable encodings retain their
compatibility guarantees, and changes to the enclosing file version remain
subject to the [file format versioning rules](../file/versioning.md).
