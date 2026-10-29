# Blob v2

Blob v2 represents nullable binary values such as images, audio, and video.
A column stores a small descriptor per value, allowing scans without fetching
payloads and subsequent reads of whole values or byte ranges. Payloads may reside
inside a Lance data file, in Lance-owned objects, or in externally owned objects.
Independently addressed Managed objects can survive data-file rewrites without
copying their bytes.

Writers accept bytes or a URI. A URI can be recorded as an External reference or
ingested into Lance-owned storage. Lance retains its payloads while any
protected snapshot (a table version retained by the cleanup policy) references
them; see [Snapshot Lifetime](#snapshot-lifetime). External owners must preserve
their objects. Blob v2 requires file format 2.2 or later. API examples and
configuration belong in the
[Blob Columns guide](../../guide/blob.md).

## Logical Input Values

For each non-null blob struct:

- Exactly one of `data` and `uri` must be non-null. Empty `data` is valid.
- `uri` must be non-empty. Without a range, it denotes the whole external object.
- `position` and `size` must both be null or both be non-null. When supplied,
  they require `uri`, must have `size > 0`, and select
  `[position, position + size)` without overflowing `UInt64`.

Absent range children mean the same as null range children. Input ranges select
external bytes; they never address Lance-owned storage.
Nullness comes from outer struct validity, and children beneath a null struct
are ignored. Writers must reject invalid shapes and value combinations with
errors identifying the field and offending input.

## Schema

The **table manifest** retains the logical struct shape supplied when the column
is created: either the minimal or complete schema below, including child metadata
and nullability. Legacy blob-marked binary inputs are converted to a Blob v2
logical struct by dataset writers targeting file format 2.2 or later; see
[Version Compatibility](../../guide/blob.md#version-compatibility).
The **data file** stores the five-child descriptor schema, regardless of the
logical shape. Writers convert logical `data`, `uri`, `position`, and `size`
values into that descriptor; those input children are not stored alongside it.
The descriptor itself is a packed struct occupying one physical column.

### Table Schema

Both shapes are logical input schemas. Use the minimal shape for bytes or
whole-object URIs; use the complete shape to supply byte ranges within external
objects. Writers may append either shape to an existing Blob v2 column, retaining
the table's existing logical shape. Supplying a complete input to a column created
with the minimal shape does not require rewriting existing payloads or changing
the manifest schema.

The minimal schema is:

```text
blob: struct (nullable)
    <metadata>: {"ARROW:extension:name": "lance.blob.v2"}
    data: large_binary (nullable)
    uri:        string (nullable)
```

The complete schema is:

```text
blob: struct (nullable)
    <metadata>: {"ARROW:extension:name": "lance.blob.v2"}
    data:     large_binary (nullable)
    uri:            string (nullable)
    position:          u64 (nullable)
    size:              u64 (nullable)
```

A logical blob field:

- Must carry `ARROW:extension:name = "lance.blob.v2"` on the blob field itself,
  including when nested.
- Must have exactly the children, types, and order of one of these schemas.
  `data` and `uri` must be nullable. Child metadata does not affect recognition.
- May declare both `position` and `size`, or neither; either field may
  be nullable or non-nullable.
- May have any outer name and be nullable or non-nullable.
- May appear beneath `Struct`, `List`, and `LargeList`, including combinations.

Other child types, extra children, and reordered children are invalid.
Container validity, list boundaries, and element order keep their Arrow
meanings. Materialization replaces the blob field with `LargeBinary`, preserving
that structure. A null list, empty list, null blob, and empty blob remain distinct.

### Data File Schema

```text
blob: struct (nullable)
    <metadata>: {
        "lance-encoding:blob": "true",
        "lance-encoding:packed": "true"
    }
    kind:         u8 (non-nullable)
    position:    u64 (non-nullable)
    size:        u64 (non-nullable)
    blob_id:     u32 (non-nullable)
    blob_uri: string (non-nullable)
```

The outer name and nullability follow the blob field. It maps to one physical
column; see [Blob Identity Namespace](schema.md#blob-identity-namespace) for
field-ID mapping. Descriptor scans expose this physical representation without
changing the table schema.

Names, order, types, and numeric kinds are format contracts.
[Struct packing](../file/encoding.md#struct-packing) determines child encoding,
not fixed byte offsets. Packed struct encoding is independent of the `Packed`
payload kind.

## Descriptor Kinds

Positions and sizes count bytes from the beginning of the selected object.
The exclusive end is `position + size`; arithmetic must not overflow.

Packed and Dedicated remain the default out-of-line storage kinds without the
[Managed opt-in](#capability-and-compatibility). They are not deprecated by this
specification.

| Kind | Object | `position` | `size` | `blob_id` | `blob_uri` |
|---|---|---|---|---|---|
| 0 Inline | Owning data file | Absolute file offset | Length | 0 | `""` |
| 1 Packed | Owning file's sidecar | Offset in sidecar | Length | Sidecar ID | `""` |
| 2 Dedicated | Owning file's sidecar | 0 | Object length | Sidecar ID | `""` |
| 3 External | Externally owned | Offset | Length; 0 requests object-length lookup | 0 for absolute URI; otherwise base ID | Absolute URI or relative path |
| 4 Managed | Lance-owned, implicit base | Offset | Length; 0 is empty | Unused; writers set 0 | Relative path |
| 5 ManagedWithBase | Lance-owned, explicit base | Offset | Length; 0 is empty | Base ID, including 0 | Relative path |

Inline bytes occupy an out-of-line buffer inside the data file; descriptor page
compression does not change payload offsets or require payload decompression.
An Inline descriptor with `size = 0`, including an all-zero descriptor, is empty.

### External Resolution

With `blob_id = 0`, resolve `blob_uri` as an absolute URI through the reader's
object-store registry, using the dataset's runtime storage options and credential
configuration. Anonymous access is not assumed; credentials are not stored in
the descriptor. Otherwise, look up that **ID**, not an array index, in
`manifest.base_paths`, use its object store and credentials, and append
`blob_uri` directly. Do not add `data/`.

For `size > 0`, read the recorded interval. A writer that knows a non-empty
object's length `N` may use `position = 0, size = N`, avoiding the length lookup.
Without an explicit input range, writers store `position = 0, size = 0`.
The zero-size sentinel discovers the complete object length at read time and
uses it as the read length, retaining the recorded position. It means neither
an explicit empty range nor the bytes remaining after position.

Recording a length fixes the requested interval, not the object's contents.
External descriptors store no ETag, content hash, or object version to detect
mutation. Owners should use immutable object locations when reproducible reads
are required. Reference writes need not check existence or length, so missing
objects and truncated ranges may fail only when read.

### Managed Resolution

Both Managed kinds use a non-empty canonical relative `blob_uri`: no `://`,
backslashes, leading or trailing slash, empty components, or `.` or `..`
components.

- **Kind 4:** locate the row's fragment and the `DataFile` supplying the blob
  field. Use that file's `base_id` binding, or the current dataset root if absent.
- **Kind 5:** use the descriptor's `blob_id` binding in `manifest.base_paths`.
  Every `UInt32` value, including zero, is a possible explicit ID.

Append `blob_uri` directly to the resolved base, using its object store and
credentials. Do not infer any prefix from `BasePath.is_dataset_root`.
The default dataset root needs no registration. Explicit IDs must already have
bindings, which must not be rebound while referenced. Moving descriptors between
files or base-ID namespaces must preserve their resolved object locations.

`size` is exact. Materializing a zero-length Managed value returns empty bytes
without reading the payload or discovering its length; it does not establish
that the object exists. Path, base, and range validation still apply. This
differs from the released External sentinel. Empty byte inputs can use Inline,
but empty Managed ranges are also valid stored values.

### Null Descriptors

Writers must encode nullness in the outer descriptor struct validity, using
the file's repetition and definition levels. Readers must also accept released
schemas with nullable children and recognize a null `kind` as a null blob.
Children beneath a null blob are ignored; nullness must not be inferred from
any other child. Non-null blobs require values in the children used by their kind.
Unknown kinds or bases and invalid ranges are errors. Non-empty reads of missing
or truncated payloads must fail rather than return null or an empty value.

## Object Layout

### Managed Object Layout

New Managed objects use names independent of data files:

```text
<base_path>/
    _blobs/
        <uuid>.blob
```

`_blobs/` is inside the selected table storage base. Objects contain raw bytes,
without framing or an offset index, and may hold one or many values. Descriptors
supply their boundaries. Readers must not infer boundaries from row order or
require the `_blobs/` prefix: an adopted sidecar keeps its existing path under
`data/`, including after its original data file is removed.

### Sidecar Resolution

Inline, Packed, and Dedicated need the owning data file's identity. Locate it
through the row's fragment and field-to-file mapping. Its `DataFile.base_id`
determines the data directory: `<dataset_root>/data` without a base, `<base>/data`
for a dataset-root base, or `<base>` for a non-root base.

Let `data_file_key` be the final component of `DataFile.path` without `.lance`:

```text
<data_directory>/
    <DataFile.path>
    <data_file_key>/
        <encoded_blob_id>.blob
```

Sidecar IDs are nonzero `UInt32` values scoped to the data file, shared by its
blob columns. Encode the ID as 32 binary digits, least significant bit first,
then append `.blob`: ID 1 is `10000000000000000000000000000000.blob`, and ID 2
is `01000000000000000000000000000000.blob`. Different data files may reuse an
ID. Both sidecar kinds, like Managed objects, contain raw bytes without framing.

## Writer Policy

Without the [Managed opt-in](#capability-and-compatibility), byte inputs use
Inline, Packed, or Dedicated according to placement thresholds. With the opt-in,
out-of-line payloads use Managed kinds instead; packing multiple values and
dedicating an object to one value do not require different Managed kinds.
Readers always follow the recorded descriptor, independent of placement policy.

In **reference mode**, URI input becomes External without copying bytes.
Writers match registered non-dataset-root bases by object store and path
components, preferring the longest matching base. They record the base ID and
remaining relative path. Unmatched URIs require explicit permission to reference
outside those bases; otherwise writing fails.

In **ingest mode**, writers copy the source object or selected range into
Lance-owned storage using the byte-input policy. Destination offsets address
the copied bytes; there is no persisted ingest kind. Credentials remain runtime
configuration.

## Snapshot Lifetime

Payloads and descriptor files must be complete before the
[manifest commit](transaction.md) publishes their references. Uploading an
object alone does not make it live. Published Lance-owned payloads must remain
immutable while protected snapshots reference them.

When reusing an existing object, the source snapshot must remain protected from
cleanup until the destination commit completes. Establish retention, for example
through a source tag, before cleanup can retire it. An open handle or an
existence check does not pin the snapshot. Compaction, reference appends, and
shallow clones rely on this retention rule; manifest publication alone provides
no additional lease for the preceding work.

### Rewrites

Rewrites must preserve bytes, nullness, nesting, and reference semantics.
With Managed storage enabled, compaction behaves as follows:

| Input kind | Output descriptor | Payload bytes |
|---|---|---|
| 0 Inline | Inline, pointing into the new data file | Copied |
| 1 Packed | Managed, pointing to the original sidecar and range | Not read |
| 2 Dedicated | Managed, pointing to the original sidecar, offset 0 and recorded size | Not read |
| 3 External | Same resolved reference and range | Not read |
| 4 Managed | Same object and range, preserving the source file's base | Not read |
| 5 ManagedWithBase | Same object, base, and range | Not read |

Packed and Dedicated cannot be copied unchanged into a different data-file
namespace. Resolve the sidecar first, then record its relative path, including
`data/` when appropriate. For any Managed rewrite, use kind 5 when retaining
an explicit source base; kind 4 requires the destination file to resolve to
the same implicit base. Ordinary row compaction does not repack Managed objects
or reclaim their unused ranges. Without the opt-in, compaction uses byte-input
placement for Lance-owned payloads.

### Garbage Collection

Protected snapshots include the latest version, retained history, tags, and
branch references under the table's cleanup policy. Managed liveness is the
union of references from their live rows. Cleanup follows each snapshot's
schema, descriptor files, deletion state, and base bindings, including nested
fields. It needs no payload reads or per-object manifest inventory.

Inline lifetime follows its data file. Packed and Dedicated sidecars remain
protected by their owning data files, and adopted sidecars also remain
protected by Managed references. Any live range retains the whole object;
removing the producing data file or deleting other ranges does not free it.
External objects are never deleted by Lance.

Cleanup may delete unreferenced objects only within its own storage namespace
and under the rules for expired versions and uncommitted objects. References
must be established before deletion; failures reading required manifests or
descriptors must stop cleanup. Referencing another dataset's base does not grant
deletion authority, and a missing source manifest does not prove descendant
branches have released their references.

### Clones

A shallow clone preserves source locations, including kind 4's implicit base
through `DataFile.base_id`. It relies on continued source retention, such as a
source tag, and does not independently extend object lifetime.

A deep clone copies Lance-owned payloads into destination storage. Local Managed
objects keep their relative paths, allowing their descriptor files to be copied
unchanged. Registered-base objects are copied and their references rebound to
the destination. The clone must survive removal of the source dataset. External
references remain externally owned in both clone modes.

## Capability and Compatibility

Schema metadata `lance.blob.managed = "true"` opts writes and compaction into
Managed storage. Absent metadata or `"false"` selects the existing placement
policy; other values are invalid. Prepared Managed references require the opt-in.
Committing `"true"` activates `FLAG_MANAGED_BLOBS = 16384` (`1 << 14`) in both
`reader_feature_flags` and `writer_feature_flags`, even without new blob files.
A manifest with only one bit set is invalid.

Both bits remain set in subsequent versions, including after removing the
opt-in, dropping blob fields, or restoring an older snapshot. Activation does
not rewrite existing files: flagged snapshots can contain all six kinds, and
supporting readers must understand Packed and Dedicated as well as Managed.
Clients without the capability must reject flagged snapshots.

Before enabling Managed storage, upgrade or stop **all** maintenance clients.
Older clients can run cleanup through cached handles or unflagged historical
snapshots and delete adopted sidecars still referenced by newer snapshots.
The flag cannot revoke those handles or protect those maintenance paths.
All maintenance after activation therefore requires Managed support.

Managed kinds use the same descriptor schema in file formats 2.2 and 2.3.
They preserve the released meanings of kinds 0–3, including nullable `kind`
and the External sentinel; logical input validation must not be applied
retroactively to descriptors. The earlier two-child blob layout retains its
separate [Blob Page Layout](../file/encoding.md#blob-page-layout) read contract.
Legacy marked binary inputs are adapted as described in the
[Blob Columns guide](../../guide/blob.md); ordinary unmarked binary columns
do not become Blob v2. See [Format Versioning](versioning.md) for capability
rules and [File Format Versioning](../file/versioning.md) for stability policy.
