# Blob reference protocol

This proposal defines an unreleased execution wire contract and its Arrow
representation. `protos/blob_reference.proto` and the generated
`lance::blob_reference_pb` types define transport only. Capture, payload reading,
worker bindings, budgets, cancellation, and their behavioral tests belong to
implementation follow-ups. No persisted descriptor or Lance file format changes.

A **blob** is one logical binary value. An **object** is a physical container,
such as a data file or packed blob file; several blobs can occupy ranges of one
object. Engines transport locators opaquely. Version 1 supports resolved raw
object ranges, not arbitrary encoded objects or unresolved Blob descriptors.

## Source contexts and locators

Every envelope and locator requires `protocol_version = 1`. Unknown versions,
origins, and required capabilities must fail; they must never fall back to user
bytes or a latest-snapshot open. The protocol version is independent of the Lance
file version. New semantics require a negotiated capability or a new version.

`SourceContext` contains:

| Field | Meaning |
| --- | --- |
| `snapshot` | A validated exact `SnapshotRef`, with its selected verification mechanism. |
| `access_context_key` | A nonempty portable key for an authenticated worker binding, including credentials, configuration, and per-base routing. A driver-local pointer is insufficient. |
| `required_capabilities` | Sorted, unique capability IDs: 1 is raw object-range transport; 2 is immutable external-object support. |

The context contains no manifest, base registry, worker handle, or credentials.
The capability list must be exactly `[1]` or `[1, 2]`. Missing capability 1,
duplicates, other ordering, and unsupported IDs are errors. Locator origin must
be `LANCE` or `EXTERNAL_IMMUTABLE`; `UNSPECIFIED` is invalid.
Its source binding fingerprint is BLAKE3 of the validated encoded context.
Encoding uses ascending protobuf field numbers, minimal varints, omitted implicit
defaults, a sorted packed capability list, and optional-field presence. Neither
the context nor its nested snapshot contains protobuf maps. The v1 fingerprint
covers fields 1, 2, 10, and 11; fields 3 through 9 are reserved. Future extensions
must preserve this fingerprint or change the protocol version.

`ObjectLocator` contains an absolute canonical object URI, physical byte offset,
optional exact length, origin, optional registered base ID, and source binding
fingerprint. The fingerprint differs from the manifest digest and the
storage-binding-independent `SnapshotRef::snapshot_id()`.

Lance-owned objects require a known length. An unknown length is allowed only for
a whole immutable external object at offset zero. A present zero length denotes
a valid empty blob. A present base ID, including zero, must exist in the bound
source manifest. Absent base ID selects primary storage for Lance-owned objects,
or the absolute URI's storage binding for external objects.

References require trusted producers, or authorization of every object URI before
worker credentials are used. A fingerprint checks associations; it does not
authenticate locator contents or grant access to an arbitrary URI.

## Arrow representation

The Arrow extension name is `lance.blob.ref`, with extension metadata
`{"version":1}`. Its storage type is:

```text
nullable Struct
+-- source_context: UInt32, non-null
+-- locator: LargeBinary, non-null
+-- length: UInt64, nullable
```

`source_context` is a batch-local slot in the accompanying context table.
`locator` is the opaque encoded locator. `length` is the exact logical blob
length when known and must agree with the locator. Null means unknown; zero
means a known empty value. Slots and lengths use their full unsigned ranges.

Parent validity represents SQL null. Child values under a null parent are ignored
and need no context. A valid parent requires valid non-null slot and locator
children and a matching source fingerprint. Null and valid empty values must
remain distinct. Plain BINARY, magic-prefixed bytes, and untagged descriptor
structs are not reference columns. The extension field must survive transport.

Row order, duplicate references, validity, empty values, and unknown lengths must
be preserved. Reordering consumers carry request indices separately. Equality is
source identity plus object location/range, not content equality. Slot numbers
from independent batches cannot be compared before remapping.

## Envelopes and nested values

`ReferenceBatch` contains a context dictionary and one Arrow IPC stream with
exactly one batch and one reference-extension column. All contexts and visible
non-null references must be validated before payload I/O. All-null and empty
arrays may have an empty dictionary. Missing or malformed references are errors,
never substituted nulls or omitted rows.

When combining batches, intern contexts by full binding fingerprint and remap
slots without changing locator bytes, row order, or validity. Missing contexts
and duplicate entries in the target dictionary are errors; unused contexts are
allowed. Engines may instead register encoded contexts once and transport the
same extension arrays through their existing Arrow exchange.

`NestedReferenceBatch` contains an encoded `ReferenceBatch` and an Arrow IPC
batch with one indices column. It preserves the selected Struct/List/LargeList
path, offsets, and nulls; unrelated struct fields are omitted. Its non-null
`UInt64` leaves index the reference array. Null leaves denote SQL null. Empty
lists remain distinct from null lists. Only visible list child ranges participate
in validation; unrelated backing values must not change a slice's meaning.
Runtime support for nested capture is advertised separately from message decoding.

Logical demand ranges are relative to the beginning of a blob, not its backing
object. Arithmetic overflow and ranges outside a known length are errors. An
empty range at the end is valid. Unknown lengths must be resolved before bounds
can be accepted; logical ranges must also fit physical address arithmetic.

## Authority and lifetime

Workers authenticate bindings and validate exact snapshots and registered bases
before payload I/O. Engines must not reconstruct sidecar paths, reinterpret
unsupported encoded objects as raw ranges, or bypass the source binding by using
default storage credentials for a locator URI.

This transport acquires no retention. Applications keep source snapshots and
immutable external objects available through all workers, retries, outstanding
reads, and deferred consumers, and coordinate cleanup and branch deletion.
External objects require an explicit application immutability guarantee and
capability 2. Matching codecs are required while this protocol is unreleased.

The schema remains under the existing format-change review and vote process.
