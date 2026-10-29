# Engine blob integration

Lance supplies portable locators, source validation, payload planning, and snapshot
retention. The engine decides which expressions need bytes and where to fetch
them. Reference identity is not SQL content equality or ordering. Preserve the
Arrow extension field, validity, source table, and nested element mapping until
the expression needs the payload.

## Coordinator and worker handoff

1. Check the worker's reference protocol and supported operations before assigning
   work. Require `capture_top_level`, `read_whole`, `read_range`, `streaming`, and
   `cancellation`. Require `capture_nested` for nested capture and
   `reader_metrics_v1` for the metrics below. Retention and maintenance clients
   must support `snapshot_retention_v1`.
2. Resolve the exact branch/version. Capture a `BlobSourceContext` with a stable
   table-incarnation identity and an access key that each worker can resolve to
   authenticated storage. Credentials are never embedded in reference rows.
3. Acquire `SnapshotRetention` before issuing references. Persist acquisition
   intent and the resulting token in the coordinator's recovery journal.
4. Scan descriptors with physical `_rowaddr` and capture from those arrays. Keep
   the scan's original field IDs and source association through engine operators.
   Capture performs no descriptor re-take or payload read.
5. Transport encoded contexts once per source and the reference extension arrays
   through the engine's Arrow exchange/spill. Remap source slots when combining
   independently produced batches. The self-contained `BlobRefBatch.encode()`
   envelope is convenient for small handoffs and conformance tests.
6. Each worker independently resolves storage credentials and opens the recorded
   snapshot. Bind the decoded contexts in their transported slot order. Submit
   only demanded references with engine request indices and whole/range demands.
   Use completion indices to restore engine order.
7. Consume and release each Arrow result before requesting more work that needs
   the same budget. On cancellation/failure, close streams and stop participating
   workers and retries. Release retention only after all readers and lazy
   consumers have stopped. Use the durable journal for interrupted operations.

See the [Python/Java examples](blob_references.md#python-and-javajni-bindings)
and the [snapshot retention deployment contract](snapshot_retention.md). Query
tags require explicit release. Only maintenance owners expire; their configurable
default is 120 hours. Expiration does not provide storage-enforced fencing against
arbitrary process pauses or outstanding storage requests.

The [Blob V2 storage specification](../format/table/blob.md) describes persisted
descriptors and ownership. The independently versioned execution wire is
`protos/blob_reference.proto`. Adding the nested capture envelope does not change
descriptor storage or the version-1 object locator. Unsupported versions and
required capabilities fail before payload I/O; never reinterpret them as user
binary data or reopen the latest snapshot.

## Descriptor scans, predicates, and ordering

Descriptor handling returns physical metadata views, such as `blob.kind` and
`blob.size`. These are metadata predicates; `size` is not a content comparison.
Binary handling returns logical payload bytes. The scanner preserves that choice
through filter refinement, vector search, and sorting, including sort columns
that are absent from the final output.

Ordering by a Blob V2 column's contents requires binary handling. Descriptor
handling rejects that ordering instead of comparing descriptor structs. To sort
metadata without fetching payloads, name the descriptor subfield explicitly, for
example `ORDER BY blob.size`. The engine must also materialize payloads before
SQL equality/content ordering of portable references; the opaque locator has no
SQL value-ordering meaning.

Use descriptor handling for descriptor subfield predicates/projections. A direct
`all_binary` scan cannot evaluate descriptor subfields such as `blob.size`.
For metadata predicates with payload output, scan/filter/order descriptors first,
capture the selected references, then fetch their bytes with the reference reader.
This also keeps payload demand after metadata selection.

## Nested capture and element identity

A row address identifies a row, but one row may contain several blob values.
Create a capture session for the selected **leaf field ID**. Call `capture_nested`
(`captureNested` in Java) with the original **top-level** scan field/array and one
physical address per root row. Capture returns `NestedBlobRefBatch`
(`BlobReferences.NestedBatch` in Java):

- `references`: the flat nullable reference batch consumed by the existing reader.
- `indices`: one Arrow column retaining only the selected container path, with
  uint64 leaves indexing `references`. It preserves row order, struct validity,
  list offsets, and element ordinals. Other struct children are omitted.

Mapping fields use plain Arrow containers and uint64 leaves. Original value
extension metadata is omitted because the mapping has a different value type.

For example, a selected path becomes:

```text
media: struct
+-- clips: list
    +-- item: struct
        +-- payload: uint64 -> references[index]
```

Null structs, null list elements, null lists, empty lists, and lists containing
only null blobs remain distinct. Hidden descendants of null containers produce
null references and cannot trigger payload reads. Sliced lists have normalized
offsets. Reordered and duplicated root rows have independent aligned mappings;
duplicate payload requests can still share physical I/O.

```python
from lance.blob import BlobRefCapture, NestedBlobRefBatch

def capture_clips(dataset, source, scanned, leaf_field_id):
    with BlobRefCapture(dataset, source, leaf_field_id) as capture:
        nested = capture.capture_nested(
            source, scanned.schema.field("media"),
            scanned["_rowaddr"], scanned["media"],
        )
    return NestedBlobRefBatch.decode(nested.encode())

def select_clip(nested, row, ordinal):
    # The mapping is ordinary Arrow data; an engine can UNNEST it directly.
    mapping = nested.indices.column(0)[row].as_py()
    return mapping["clips"][ordinal]["payload"]
```

Submit the resulting non-null index to `reader.read(nested.references, ...)`;
null mappings remain SQL null without a payload demand. Capture before UNNEST
discards the source path/ordinal, then carry the mapping through reconstruction.
`encode()` transports both references and the Arrow mapping. Engines can instead
transport them separately in their existing Arrow protocol. Remap contexts on
the flat reference batch; element indices address reference rows, not context slots.
Use `NestedBlobRefBatch.from_arrow(references, indices)` in Python or
`NestedBatch.fromArrow(...)` in Java to rebuild a validated envelope after these
operations. Rust provides `NestedBlobRefBatch::try_new`.

Field IDs select the leaf without parsing quoted field names. The supplied
scan fields must retain their original names and metadata; aliases cannot prove
provenance. A session cannot verify that arbitrary caller-modified descriptor
bytes came from the claimed scan.

Supported containers are Struct, List, and LargeList, including repeated list
levels and structs between lists. FixedSizeList, Map, unions, encoded-object
locators, and legacy blob leaves are unsupported. Indexed row-selector reads are
a separate API: portable capture consumes already scanned descriptors and does
not use a row selector to infer a nested element.

## Reader metrics

Rust exposes `BlobRefReader::metrics()`, Python `reader.metrics`, and Java
`reader.getMetrics()`. Binding dictionaries/maps use the same snake_case names.
Counters accumulate across a reader's streams; Rust clones share them. Snapshots
during concurrent work are not atomic. There is no reset that can erase another
stream's counters.

| Metric | Meaning |
| --- | --- |
| `descriptor_requests` | Always zero for the reference reader; descriptors were supplied to capture. |
| `metadata_requests` | Attempts to resolve unknown external object lengths; cache hits do not count. |
| `payload_requests` | Physical ranges submitted by the scheduler after coalescing/splitting. |
| `payload_bytes` | Physical bytes submitted, including gaps read during coalescing. |
| `requested_logical_bytes` | Resolved demand bytes in successfully planned operations; duplicates count separately. |
| `returned_logical_bytes` | Logical bytes yielded; duplicates count separately. |
| `coalesced_bytes` | Planned physical footprint of admitted batches, not bytes saved. |
| `duplicate_demands` | Exact repeated nonempty physical ranges after the first in each operation. Overlap alone does not count. |
| `reserved_bytes`, `peak_reserved_bytes` | Current/peak native materialization charge, including physical buffers and logical output. |
| `io_reserved_bytes`, `peak_io_reserved_bytes` | Current/peak physical I/O charge retained by outstanding native reads. |

Physical counts include submitted work even if it later fails or is cancelled.
They do not count backend retries or promise that every byte was successfully
received. An unpolled operation has no resolved logical demand count. Metadata
attempts can be retried after failure. Returned bytes can be smaller than requested
bytes after cancellation.

Reservation peaks cover the **shared materialization context**, including other
readers using it and result buffers held by consumers. Peaks persist after
release. An oversized value can run alone and exceed a configured limit; use
chunked ranges for a per-value cap. Engine/Python copies and transport buffers
require separate memory accounting. Reservation charges are not process RSS
measurements. Descriptor/reference arrays and object metadata caches are also
outside these payload budgets.

Descriptor scans, source-context creation, and provider discovery occur outside
these reader counters. Use scanner metrics or isolated dataset I/O snapshots for
that phase. Do not subtract global I/O counters to attribute one query while
other queries use the same store. The engine supplies actual exchange/spill
bytes, operator placement, and any end-to-end latency measurements.

## Runnable conformance and I/O qualification

The shared `test_data/blob_reference_v1/conformance.json` fixture covers whole
and range reads, duplicates, SQL null versus valid empty bytes, empty ranges,
unsigned request indices, and range overflow/bounds failures. Rust, Python, and
Java/JNI run the same demands through their native reader bindings. Released
Lance 11.0.0 sidecar fixtures separately check descriptor compatibility. Nested
tests cover multiple fragments, quoted child names, null parents/elements/lists,
empty/all-null lists, reordered/duplicated rows, and selective payload reads.

From `python/`, run:

```shell
uv run python python/benchmarks/blob_reference_conformance.py
```

The producer captures and retains a real dataset. An independent process decodes
the encoded references, opens the recorded snapshot, validates results, and
reports counters. A second dataset uses separate Dedicated payload objects to
compare metadata-only, top-K, and range demands with an eager baseline that reads
all payloads before applying the same selection. The runner checks identical
logical output and verifies that metadata-only work submits no payload requests,
top-K requests only selected objects, and ranges avoid full-object reads.

JSON output includes physical requests/bytes, logical demand/output, native
reservation peaks, and candidate exchange sizes for a payload Arrow stream versus
a self-contained reference envelope. The latter includes its context once; a
real engine can amortize contexts across batches. These are serialized transport
sizes, not measured network traffic. The default comparison uses 16 values of
64 KiB and selects three. `--rows`, `--payload-bytes`, and `--selected-rows` allow
larger runs. No latency benefit is claimed. Inline payloads can share read blocks
with descriptors, so request-separation assertions use separate sidecar objects
and do not demand impossible zero over-read for every storage layout.

### Example measured result

A local filesystem run with the defaults above produced the following counters.
The eager baseline reads all 16 payloads before selection; the lazy cases move
payload demand after selection. Each eager/lazy pair produced the same output
hash. Metadata work returns a row count; range work selects 16 bytes at offset 7
from each of three values.

| Workload | Payload requests | Payload bytes submitted | Peak native reservation charge |
| --- | ---: | ---: | ---: |
| Eager baseline, each workload | 16 | 1,048,576 | 2,097,152 |
| Lazy metadata | 0 | 0 | 0 |
| Lazy top-K, three whole values | 3 | 196,608 | 393,216 |
| Lazy ranges, three 16-byte demands | 3 | 48 | 96 |

The shared descriptor scan read 4,327 bytes in six requests; reference capture
added zero requests and zero bytes. These scan counters come from isolated
producer dataset snapshots and are separate from the worker reader counters.
Candidate exchange encoding used 1,049,000 bytes for the payload Arrow stream
and 5,695 bytes for the reference envelope, including one source context. Envelope
size depends on the source manifest and paths. These results describe this fixture
and demand placement; the runner does not measure an engine's network traffic,
process RSS, or end-to-end latency.
