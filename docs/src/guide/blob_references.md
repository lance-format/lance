# Portable blob references

The Rust `lance::blob::reference` module defines an execution-wire protocol for
moving lazy blob values between processes. It carries source contexts and nullable
reference arrays through exchange, spill, and retries. It does not change stored
Blob v2 descriptors or any persisted Lance format.

Protocol version 1 supplies transport, descriptor capture, validation, context
remapping, and planned reference reads through Rust, Java/JNI, and Python.
Explicit snapshot retention is available as a separate SDK operation. Serializing
a reference does not authenticate a worker or retain source files.

## Source contexts

A `BlobSourceContext` identifies one exact snapshot and its storage authority:

```text
Source context
+-- protocol version and required capabilities
+-- canonical table-incarnation identity
+-- absolute table root URI
+-- resolved branch and numeric version
+-- exact manifest path, original protobuf image, optional commit-store identity
+-- key to an authenticated worker storage context
```

`BlobSourceContext::from_dataset` captures an already resolved `Dataset`. It reads
manifest metadata and verifies that it still matches the loaded snapshot. It does
not fetch blob payloads, resolve a tag again, or substitute the latest version.
Open the source using an absolute URI or local path; relative locations are rejected.
The manifest image includes registered base mappings, including shallow-clone
source bases. Its original bytes are preserved across transport. Re-encoding the
manifest protobuf would change map-entry order and must not replace that image.

The caller supplies a canonical `table_identity` that distinguishes table
incarnations and storage/catalog authorities. A URI or catalog name alone is
insufficient when tables can be recreated or different storage endpoints use the
same URI. The snapshot additionally binds the branch, version, manifest path, and
manifest contents. When available, the commit handler's authoritative record
identity is included. An ETag is not used as a table-incarnation or content identity.

The caller also supplies an `access_context_key`. Every participating worker must
resolve it through an authenticated context with the correct storage settings,
credentials, commit service, and per-base routing. It cannot name only a driver
cache entry or native handle. Credentials may be renewed behind the same binding
key. Credentials are not embedded in individual references.

```rust
use lance::{Dataset, Result};
use lance::blob::reference::BlobSourceContext;

async fn capture_context(dataset: &Dataset) -> Result<Vec<u8>> {
    let context = BlobSourceContext::from_dataset(
        dataset,
        "catalog/table-incarnation-id",
        "query/authenticated-storage-binding",
    ).await?;
    Ok(context.encode())
}
```

Contexts carry protocol version 1 and a sorted required-capability list:

| Capability | Meaning |
| --- | --- |
| 1 | Transport of snapshot-bound object-range locators |
| 2 | Explicit application guarantee that referenced external objects are immutable |

Capability 1 is required. Capability 2 is optional. Missing, duplicate, reordered,
or unsupported capability entries are rejected. These are transport capabilities;
they do not advertise a payload reader or retention service.

## Capturing scan descriptors

`BlobRefCapture` converts an existing top-level Blob V2 descriptor column into a
`BlobRefBatch`. Create a session for the exact scan snapshot and Lance field ID,
then reuse it across batches. Preserve the source context alongside the scan
output; capture rejects a different snapshot or storage binding. Raw descriptor
arrays do not carry provenance, so callers must maintain that association when
reordering or combining scan output.

```rust
use std::sync::Arc;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use lance::{Dataset, Result};
use lance::blob::reference::{BlobRefCapture, BlobRefBatch, BlobSourceContext};
use lance_core::datatypes::BlobHandling;
use lance_core::utils::address::RowAddress;

async fn capture(dataset: Arc<Dataset>) -> Result<BlobRefBatch> {
    let source = BlobSourceContext::from_dataset(
        &dataset, "catalog/table-incarnation", "query/storage-binding",
    ).await?;
    let field_id = dataset.schema().field_id("image")?;
    let mut capture = BlobRefCapture::try_new(dataset.clone(), source.clone(), field_id)?;
    let batch = dataset.scan().project(&["image"])?
        .blob_handling(BlobHandling::BlobsDescriptions)
        .with_row_address().try_into_batch().await?;
    let addresses = batch["_rowaddr"].as_primitive::<UInt64Type>()
        .values().iter().copied().map(RowAddress::from).collect::<Vec<_>>();
    capture.capture(
        &source, batch.schema().field_with_name("image")?,
        &addresses, batch["image"].as_struct(),
    ).await
}
```

Use the original scan field before aliasing, and physical `_rowaddr` values from
the same scan. Capture validates field identity, descriptor layout and child
validity, cardinality, physical address bounds, and range overflow. Null blobs,
known empty blobs, duplicates, and input ordering are preserved. Nested containers
require a row-to-parent mapping and are outside this capture API.

Capture reuses Lance's descriptor resolver for Inline, Packed, Dedicated, Managed,
and External storage. Inline references point into the original data file; other
Lance-owned references point to the resolved sidecar or managed object. Registered
bases, including shallow-clone source bases, are resolved from the source snapshot.
The locator preserves the selected base ID so workers can apply its exact storage
binding even when multiple bases have the same URI.

Capture does not reread descriptors, read payloads, or HEAD objects. Fragment,
object store, and base resolution metadata are cached within the session. External
absolute URIs are captured without constructing storage clients. Whole-object
lengths remain unknown; a later reader must resolve and cache that metadata.
External references require the source context's explicit immutability
assumption. The application must retain source snapshots through all deferred
reads and retries, as described below.

## Reading references

`BlobRefReader` accepts full-value and blob-local range demands. On each worker,
resolve every context's `access_context_key` to authenticated storage settings and
open its exact branch and version. Pair that snapshot with its transported context
using `BlobRefBinding::try_new`. Supply bindings in the same slot order as the
reference context table. Construction checks the loaded manifest; each request
also checks its locator fingerprint, registered base, and bound storage root.

```rust
use futures::TryStreamExt;
use lance::Result;
use lance::blob::reference::{
    BlobMaterializationContext, BlobReadRange, BlobRef, BlobRefBinding,
    BlobRefDemand, BlobRefReader, BlobRefRequest,
};

async fn fetch(bindings: Vec<BlobRefBinding>, reference: Option<BlobRef>) -> Result<()> {
    let budget = BlobMaterializationContext::try_new(1 << 20, 8 << 20)?;
    let reader = BlobRefReader::try_new(bindings, budget)?;
    let mut results = reader.read([
        BlobRefRequest::new(7, reference.clone(), BlobRefDemand::Whole),
        BlobRefRequest::new(19, reference, BlobRefDemand::Range(BlobReadRange::new(0, 16))),
    ])?;
    while let Some(result) = results.try_next().await? {
        // Route by result.request_index and consume result.data here.
        // Drop the result before requesting more than the budget can retain.
        let _ = result.request_index;
    }
    Ok(())
}
```

Request indices are engine supplied and must be unique within an operation.
Completion order is unspecified; repeated references still produce separate
indexed results. SQL nulls return `None`, while empty values and empty ranges
return a non-null empty buffer. Nulls do not require a source binding or payload
bounds, but all range arithmetic must be valid. Non-null ranges must fit the
logical blob, including an empty range at its end.

Reader sessions cache object resolution and unknown external lengths across
operations. Known lengths require no HEAD request. An unknown external length
requires one successful metadata lookup per object and binding; a later demand
uses the cached size. Store resolution and metadata lookups run concurrently over
distinct objects, bounded by the reader's I/O parallelism. Payload fetching never
takes rows or rereads descriptors.
Registered storage bases use their bound routing; standalone external URIs use
the authenticated snapshot's registry and storage parameters.

The existing Lance planner groups compatible backing objects, merges overlapping
and nearby physical ranges, and shares immutable byte buffers for duplicate and
overlapping demands. Equal URIs with distinct object-store bindings remain
separate. The I/O scheduler is shared through `BlobMaterializationContext`.

Both context limits must be nonzero. Reference reads sharing a context share an
aggregate I/O admission budget and a materialization budget. Each bounded batch
charges its coalesced physical spans plus its logical result lengths, counting
duplicates separately. This also allows an adapter to construct one copied output
per request while retaining the corresponding returned buffers. Planning metadata
is proportional to request count; session metadata is proportional to the distinct
objects touched. These limits govern payload bytes, not that metadata.

One request whose charge exceeds a limit is admitted alone when that budget is
idle. A limit therefore does not impose a hard cap on a single oversized value.
Applications requiring a hard maximum must split large values into range demands.

Returned `Bytes`, their clones, and their slices retain the batch's materialization
reservation until the last owner is dropped. An adapter copying results into Arrow
arrays must retain those owners alongside the arrays. The batch reservation
already includes one logical copy per request. Requesting an additional reservation
from the same budget while holding the results can block indefinitely if the batch
fills the budget.
Arbitrary additional copies are outside the reader's accounting. Use
`reserved_bytes()` and `io_reserved_bytes()` to observe outstanding reservations.
Retaining all results while collecting a multi-batch stream can fill the budget
and prevent the next batch from proceeding; consume and release results as the
stream advances.

Call `stream.cancel()` or obtain `stream.cancellation_token()` for another task.
Cancellation interrupts metadata, admission, and read waits. Dropping a stream
also discards pending outputs and readers. Backend I/O already submitted may
finish before its cancelled response is discarded; its reader retains both
reservations until that happens. Previously yielded buffers remain valid and
keep their reservations. A read error terminates the operation and marks all
remaining demands incomplete; it never substitutes null values for failures.

## Reference values

A reference column uses the Arrow extension name `lance.blob.ref` and extension
metadata `{"version":1}`. Its storage type is:

```text
nullable Struct
+-- source_context: UInt32, non-null
+-- locator: LargeBinary, non-null
+-- length: UInt64, nullable
```

`source_context` is a slot in the context table accompanying the batch. `locator`
is a versioned, opaque Lance locator. `length` is the exact logical byte length
when known. A null length means unknown; zero is a valid known length.

The parent validity bit represents SQL null. Child values under a null parent are
ignored and require no context. A valid empty blob is a non-null parent with
`length = 0` and a valid locator. It must survive transport as distinct from SQL
null. Plain BINARY values, magic-prefixed payload bytes, and untagged descriptor
structs are not reference columns. Engines must preserve the extension field.

The original request position is the array row index. Serialization and remapping
preserve row order, duplicate references, nulls, empty values, and unknown lengths.
Engines that reorder requests or complete reads out of order must carry the
original request index and correlate results explicitly. Missing or malformed
references are errors, never null values or omitted rows.

Reference equality compares source identity and object location/range, not blob
contents. Distinct references may contain identical bytes. Identical locator bytes
with different context slots cannot be compared across batches until those slots
have been remapped.

## Locator ownership and ranges

The wire messages are defined in `protos/blob_reference.proto`. Engines transport
locators without constructing object paths or interpreting stored Blob-kind values.
Version 1 encodes an absolute object URI, physical offset, optional exact length,
origin, optional registered base ID, and source fingerprint. An absent base ID
selects primary storage for a Lance-owned object or the absolute URI's storage
binding for an external object. A present base ID must exist in the source
manifest; zero is a valid registered ID. A Lance-owned object must have a known length.
An external whole-object reference may have an unknown length only at object
offset zero. External subranges require an explicit length.
Workers must resolve primary and registered-base storage through the authenticated
source binding. Opening the locator URI with default storage settings is not equivalent.

The source fingerprint is BLAKE3 of the validated `SourceContext` encoded by Lance.
Encoding uses ascending protobuf field numbers, minimal varints, omitted implicit
defaults, and the sorted, packed capability list. Optional fields preserve presence.
The original manifest image is an opaque byte field in that encoding. There are no
protobuf maps in `SourceContext` itself. Engines should forward the encoded context
or use Lance's codec rather than reconstructing its fingerprint.
The v1 fingerprint covers the fields numbered 1 through 10 in this contract.
Future extensions must preserve that fingerprint or change the protocol version.

Each locator embeds this fingerprint. Validation compares it with the context at
its slot before any payload I/O. A slot with another version, branch, table
incarnation, authority, or access binding is rejected even when its numeric value
is in range. Fingerprints detect inconsistent associations; they do not authenticate
untrusted producers. Workers still authorize the storage binding.

Byte-range requests use `(offset, length)` relative to the beginning of the logical
blob, not the backing object. `BlobRef::validate_range` rejects arithmetic overflow
and ranges beyond a known length. An empty range at the end of a known-length blob
is valid. For unknown lengths, the reader must determine the remaining bounds
before reading. Logical ranges must also fit within physical address arithmetic.

### Relationship to the Object Layer

The [Object Layer proposal](https://github.com/lance-format/lance/discussions/7174)
separates a logical blob object from its representation and physical reads. This
transport sits outside that resolution boundary: SQL engines carry opaque locators,
while Lance interprets them and plans reads.

Version 1's object-range capability describes raw byte ranges only. It is not a
general reference to an encoded object. Capture must resolve the descriptor's
placement semantics before constructing a locator; a descriptor's `position` must
not be assumed to be a physical offset for future object-backed packs.

Delta, chunked, or other encoded objects need an appropriate locator and explicit
capability or protocol version before capture and reading can support them.
Unsupported capabilities and protocol versions are rejected during transport
validation. Engines must not decode locators themselves or reinterpret unsupported
objects as raw byte ranges. This execution protocol does not change the persisted
descriptor layout or require future Object Layer backends to use raw ranges.

## Combining batches

For nested capture, worker reconstruction, conformance inputs, and measured I/O
qualification, see [Engine blob integration](blob_reference_integration.md).

Two independently produced batches can both use slot zero for unrelated sources.
Intern contexts by their full source fingerprint, then call
`BlobRefBatch::remap_contexts` with the shared context table. The method updates
slots without changing locator bytes or row validity. Missing contexts and duplicate
entries in the target context table are errors. Extra unused contexts are allowed.

```rust
use lance::Result;
use lance::blob::reference::{BlobRefBatch, BlobSourceContext};

fn transport(
    batch: &BlobRefBatch,
    shared_contexts: Vec<BlobSourceContext>,
) -> Result<BlobRefBatch> {
    let remapped = batch.remap_contexts(shared_contexts)?;
    let wire = remapped.encode()?;
    BlobRefBatch::decode(&wire)
}
```

`BlobRefBatch::encode` writes one protobuf envelope containing the contexts and one
Arrow IPC stream with exactly one batch and one extension column. The decoder
validates the envelope, contexts, Arrow field, locator versions, lengths, and source
associations. An all-null or empty batch may have an empty context table.

The envelope is a self-contained convenience format. For a streaming exchange,
engines can send each encoded context once during query setup, then carry extension
arrays in their existing Arrow transport. Workers validate those arrays with
`BlobRefBatch::try_from_arrow` and the registered context table. This avoids sending
the manifest images again for every batch.

Unknown protocol versions, locator origins, and required capabilities fail clearly.
They must never trigger interpretation as user bytes, selector fallback, or opening
the latest snapshot. Protocol versions are independent of the Lance file version.
New required semantics need a negotiated capability or a new protocol version;
unknown protobuf fields alone cannot establish support.

## External objects and lifetime

An exact dataset snapshot fixes the external reference, not the external object's
contents. Version 1 accepts external references only when the producer explicitly
calls `with_external_immutability_assumption` before issuing them. The application
guarantees that the addressed contents remain unchanged for the entire lifetime of
the references. This changes the context fingerprint and enables capability 2.

This mode does not record an object version or enforce an ETag/conditional read.
Mutable external objects require a future capability that records an immutable
object identity and enforces it in the reader. A backend without that enforcement
can provide only the application's immutability guarantee.

Source files, sidecars, Managed objects, and applicable external bases must remain
available until every deferred read, worker retry, and lazy consumer has finished.
An open `Dataset`, a serialized manifest, or a reference value is not a GC root.
The application must retain the exact source snapshots and coordinate cleanup.
External objects remain subject to their own ownership and immutability contract.
Cross-query persistence or export needs an explicit retention agreement extending
beyond the original query. Encoding a reference or retention token never acquires
or extends protection by itself.

## Query lifetime and cleanup

Use general [snapshot retention](snapshot_retention.md) before issuing references.
`SnapshotIdentity` identifies an exact dataset snapshot independently of blob
capabilities and worker routing. `SnapshotRetention` protects that snapshot with
the existing durable query tags until explicit release. Obtain the identity using
`BlobSourceContext::snapshot_identity()` in Rust, `source.snapshot_identity()` in
Python, or `source.getSnapshotIdentity()` in Java. Ordinary scans use the same API
without creating a blob context.

Acquisition, release, and cleanup coordinate with a table-root maintenance lock.
Acquisition and release share the lock; cleanup holds it exclusively. Internal
query tags protect their snapshots without triggering cleanup's user-tag error.
Maintenance owners have a fixed expiry of 120 hours by default, configurable using
`lance.snapshot_retention.lock_expiration_hours`. Each owner records its expiry;
other callers honor it regardless of their own configuration. Query tags do not
expire. Fixed lock expiry requires bounded operation/storage-request durations and
clock agreement; it does not fence a paused owner or cancel submitted requests.
See [maintenance expiration](snapshot_retention.md#maintenance-lock-expiration)
for the complete deployment contract.

The coordinator must persist acquisition intent, the active token, and release
intent. Release only after all workers, retries, outstanding reads, and lazy
consumers stop. Follow the [interruption recovery procedure](snapshot_retention.md#recovery-after-interruption).
External object lifetime and foreign dataset owners still require separate
agreements.

## Python and Java/JNI bindings

Python exposes `BlobSourceContext`, `BlobRefCapture`, `BlobRefBatch`,
`BlobRefReader`, and `BlobRefReadStream` from `lance.blob`
and `lance`. Java exposes the corresponding operations through
`org.lance.BlobReferences`, with `SourceContext`, `Capture`, `Batch`, `Reader`,
and `ReadStream` types. Both use the general `SnapshotRetention` API for query
lifetime protection.

Query the protocol version, source capabilities, and supported SDK operations
before scheduling workers. Python provides
`BlobSourceContext.protocol_version()`, `supported_capabilities()`, and
`supported_operations()`; Java provides `getProtocolVersion()`,
`getSupportedCapabilities()`, and `getSupportedOperations()` on `BlobReferences`.
Require `capture_top_level`, `read_whole`, `read_range`, `streaming`, and
`cancellation`, plus `snapshot_retention_v1` on retention and maintenance clients.
Reject executors missing required operations. Unknown wire versions/capabilities
also fail during decode, before any payload request.

Both bindings use the Rust capture validator and payload planner. Source contexts,
reference batches, and retention tokens have opaque protobuf encodings. A reference
batch also exposes its Arrow extension field and array for engine operators; keep
its source table and remap context slots when combining independently captured
arrays. Workers reopen the exact branch/version with their own storage binding.
Native driver pointers and credentials are not part of the transport.

This example shows local API use. A distributed coordinator must also persist the
lifecycle state described in [Recovery after interruption](snapshot_retention.md#recovery-after-interruption)
before issuing references to workers.

```python
import uuid
from lance.blob import (
    BlobRefCapture, BlobRefReader, BlobSourceContext,
)
from lance.retention import SnapshotRetention

def read_values(dataset, worker_dataset, consume):
    source = BlobSourceContext.from_dataset(dataset, "catalog/table", "workers")
    retention = SnapshotRetention.acquire(
        dataset, source.snapshot_identity(), str(uuid.uuid4()),
    )
    try:
        field_id = dataset.lance_schema.field("blob").id()
        with BlobRefCapture(dataset, source, field_id) as capture:
            with BlobRefReader(
                [(source, worker_dataset)],
                io_buffer_size_bytes=8 << 20,
                materialization_readahead_bytes=32 << 20,
            ) as reader:
                scanner = dataset.scanner(
                    columns=["blob"], with_row_address=True,
                    blob_handling="all_descriptions",
                )
                for scanned in scanner.to_batches():
                    refs = capture.capture(
                        source, scanned.schema.field("blob"),
                        scanned["_rowaddr"], scanned["blob"],
                    )
                    # Correlate completed requests with the engine's output rows.
                    indices = list(range(len(refs)))
                    with reader.read(refs, indices, indices) as stream:
                        for result in stream:
                            consume(result)
                            del result  # Release before waiting for another batch.
    finally:
        # Wait for any other participating workers/retries before this call.
        retention.release(dataset)
```

A read stream yields one completed request per Arrow batch, with
`request_index: uint64` and nullable `data: large_binary`. Completion order is
unspecified; use the request index to place results. Null and valid empty bytes
remain distinct. Python consumes the stream as an iterator/context manager.
Java calls `ReadStream.next(allocator)` until it returns null and must close every
returned `VectorSchemaRoot`, preferably with try-with-resources.
For range demands, Python passes a `ranges` list with `(offset, length)` tuples
or `None` for whole values. Java uses `Request.range(...)` or `Request.whole(...)`.
Invalid ranges fail; neither binding clamps them to the blob length.

The Arrow payload buffers retain their native `Bytes` owners and materialization
reservations without copying. Close/drop consumed batches before waiting for work
that needs the same budget. A retained slice or array keeps its reservation even
after the stream is cancelled or closed. Additional copies made by an engine are
outside this native budget and need separate accounting.

Both factories require finite, nonzero `io_buffer_size_bytes` and
`materialization_readahead_bytes` limits (camel case in Java). One oversized demand
may run alone when idle; use chunked ranges if an engine requires a per-value cap.
Streams from one reader share these limits. Closing the reader factory prevents
new submissions; existing streams and returned arrays keep their owners and must
be closed/released separately. Cancellation can interrupt a blocked `next` or a
budget wait. Pending requests terminate with an error instead of becoming nulls.

Java represents unsigned indices, versions, offsets, lengths, and limits as `long`
bit patterns; use `Long.parseUnsignedLong`/`Long.toUnsignedString` as needed.
Python accepts integers in `0..=2**64-1`. Neither binding truncates high bits.

| Capability | Rust | Java/JNI | Python |
| --- | --- | --- | --- |
| Context/ref transport and Arrow remapping | Yes | Yes | Yes |
| Top-level scanned Blob V2 capture | Yes | Yes | Yes |
| Planned whole/range reads and cancellation | Yes | Yes | Yes |
| Streaming results retaining native budgets | Yes | Yes | Yes |
| Explicit-release owned-snapshot retention | Yes | Yes | Yes |
| Fixed maintenance-lock expiration | Configurable; default 120 hours | Same table setting | Same table setting |
| Automatic query-tag expiration/renewal | No | No | No |
| Foreign dataset owner retention | Separate agreement | Separate agreement | Separate agreement |
| Struct/List/LargeList reference capture | Yes | Yes | Yes |
| FixedSizeList/Map/union reference capture | No | No | No |
| Reader I/O, demand, and reservation metrics | Yes | Yes | Yes |
