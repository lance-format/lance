# Fragment Reuse Index

The Fragment Reuse Index is an internal index used to optimize fragment operations 
during compaction and dataset updates.

When data modifications happen against a Lance table,
it could trigger compaction and index optimization at the same time to improve data layout and index coverage.
By default, compaction will remap all indices at the same time to prevent read regression.
This means both compaction and index optimization could modify the same index and cause one process to fail.
Typically, the compaction would fail because it has to modify all indices and takes longer,
resulting in table layout degrading over time.

Fragment Reuse Index allows a compaction to defer the index remap process.
Suppose a compaction removes fragments A and B and produces C.
At query runtime, it reuses the old fragments A and B by 
updating the row addresses related to A and B in the index to the latest ones in C.
Because indices are typically cached in memory after initial load,
the in-memory index is up to date after the fragment reuse application process.

## Index Details

```protobuf
%%% proto.message.FragmentReuseIndexDetails %%%
```

## Expected Use Pattern

Fragment Reuse Index should be created if the user defers index remap in compaction.
The index accumulates a new **reuse version** every time a compaction is executed.

As long as all the scalar and vector indices are created after the specific reuse version,
the indices are all caught up and the specific reuse version can be trimmed.

## Impacts

### Conflict Resolution

The presence of the Fragment Reuse Index changes how Lance detects conflicts between concurrent
operations. Operations that would normally conflict with compaction (such as index building) can
proceed without conflict when the FRI is in use. For full details on how conflict detection is
affected, see [conflict resolution](../../table/transaction.md#conflict-resolution).

### Index Load Cost

When the FRI is present, indices must be remapped at load time. Each time an index is loaded into
the cache, the FRI is applied to translate old row addresses to current ones. This adds a small
cost to index loading but does not affect query performance once the index is cached.

### FRI Growth and Cleanup

The FRI grows with each compaction. Every compaction that defers index remapping adds a new reuse
version to the index. Over time, this can accumulate and increase the cost of index loading since
more address translations must be applied.

Once all scalar and vector indices have been rebuilt past a given reuse version, that version is no
longer needed and can be trimmed. Users should schedule a periodic process to trim stale reuse
versions and keep the FRI size under control.

## FRI index versions

`IndexMetadata.name` remains `__lance_frag_reuse`, and `index_details` retains
its `FragmentReuseIndexDetails` protobuf type.

- `index_version = 0`: the existing ordered-compaction history in field 1 of
  `InlineContent` (`legacy_versions`, previously named `versions`).
- `index_version = 1`: the first format containing tagged `transitions` in
  field 2 of `InlineContent`. A single FRI history can retain both legacy
  versions and tagged transitions. Publishing tagged transitions sets this
  version to 1; replacing the history's UUID does not change its format version.

Version 1 readers must inspect `index_version` before interpreting FRI payloads.
For version 0 they read legacy history; for version 1 they combine legacy groups
and tagged mappings. A version outside their supported set is unsupported and
must not be decoded as either known format. Existing V1 readers do not reliably
enforce this check, so first publishing version 1 also requires the paired
manifest capability flags. This proposal does not enable version 1 in the
library before the implementation supplies its compatibility behavior.

Adding a mapping kind within the version-1 common contract uses the
unknown-encoding rules below; it does not automatically increment `index_version`
or allocate a manifest flag. Changing the common contract requires a separate
compatibility decision.

## Unified history and immutable mappings

A dataset has one FRI entry named `__lance_frag_reuse`. Its `InlineContent`
contains `legacy_versions` and tagged `transitions`. Existing compaction can
continue appending legacy versions; new rewrites append tagged transitions.
Each legacy group is interpreted as an ordered-compaction node in memory.

The outer `inline` and `external` choices keep their existing meanings. An
external `details.binpb` contains the complete serialized `InlineContent`, not
the per-row label payload. Updating the history replaces the FRI index UUID and
serializes its retained metadata; immutable mapping files are not rewritten.

A transition contains ordered source and destination digests and exactly one
mapping. Sources include physical deleted positions. Destination counts describe
creation-time rows. Mapping field numbers 5 and above are exclusively alternatives
with wire type 2. Future alternatives use new field numbers. Readers inspect
raw transition fields to distinguish unknown alternatives from a missing mapping;
a generated `oneof` accessor alone is insufficient. Conflicting or repeated
alternatives are invalid. Source/commit versions are not stored per transition.

Dependencies follow fragment lineage, not list or version order. Reject duplicate
producers, duplicate consumers, and cycles. All mapping kinds are value-preserving
rewrites, consume whole source fragments, and describe their complete source and
destination sets in common metadata. Those sets remain meaningful without decoding
the mapping payload. Fragment IDs must fit the physical row-address domain and
must not be reused within a lineage.

`StablePartition.map_id` is a UUID independent of the history index UUID. Its
immutable artifact is `<base>/_fri/<map_id>/row_map.lance`. An absent `base_id`
selects this dataset's base; a present value resolves through `Manifest.base_paths`.
The complete file length is `map_size_bytes`. The history's ordinary external
metadata remains under `_indices/<FRI UUID>/details.binpb`.

`Transaction.Rewrite.append_fri_transitions` expresses a semantic append delta,
committed atomically with its rewrite groups. Sources and destinations must match
the rewrite's ordered fragment digests. On rebase, validate source snapshot identity,
reserved destination IDs, and the combined lineage, then append to the latest ledger.
Disjoint valid rewrites retain both deltas; stale complete-ledger replacement is
not allowed. The outer transaction supplies its read version and commit identity.
Pruning is recomputed against the latest index provenance and complete mapping
chain, not against an earlier independently prepared history snapshot.

## Stable-partition encoding

`StablePartition` references the complete immutable `row_map.lance` file.
The file schema is:

```python
pa.schema([pa.field("label", pa.uint16(), nullable=True)])
```

There is one label per physical source row, in `Transition.sources` list order and
ascending offset within each source. NULL means the row is deleted and not
written. A non-null label is the zero-based index into `Transition.destinations`. Each
destination receives rows in this same source order. Sorting rows within a
destination is not representable by this encoding. There are 1 through 65,536
destinations, initially without deleted rows.

Schema metadata `lance:stable_partition:counts_buffer_index` contains the decimal
ID of the counts global buffer. Its integers are unsigned little-endian. The
28-byte header contains, in order: four magic bytes `LSPC`, a uint32 version
(1), a uint32 representation (0 for dense), uint32 destination count, uint32
logical block length in rows, and uint64 physical source row count. The logical
block length is positive; writers normally use 65,536. Block boundaries are
independent of physical Lance pages.

The header is followed by a row-major matrix of uint32 counts with
`ceil(total_rows / block_rows)` rows and one column per destination. Cell `(b,d)`
is the cumulative count of label `d` through the end of block `b`. There is no
extra initial zero row. Before block zero all counts are zero. The last matrix
row supplies destination totals. The matrix must have exactly the declared
length, nondecreasing counts per destination, and a sum of per-block count
increments no larger than that block's physical row count. Each matrix cell must
agree with the cumulative count of its label in the label column. Destination totals must equal the corresponding
`Transition.destinations[].physical_rows`; total labels must equal the sum of
source physical rows, and NULL counts within each source must match that
source digest's `num_deleted_rows`.

To translate a source address, sum the physical lengths of preceding sources
and add its offset to obtain position `g`. Read the label at `g`; NULL translates
to deleted. Otherwise destination offset is the preceding block's cumulative
count for that label (zero for block zero), plus the count of equal labels
strictly before `g` in the same block. A range translation seeds per-destination
counters from the preceding block, scans any prefix before the requested range,
and increments the selected counter for each non-null label. Readers fetch the
counts through the global-buffer reference and read labels by logical row range.
Unknown counts versions or representations are unsupported, never decoded using
version-1 assumptions; malformed known layouts are corruption errors.

## Capability and unknown encodings

The first commit publishing the extensible representation sets
`FLAG_FRAGMENT_REUSE_INDEX` (512) in both manifest flag fields. Both bits persist
in subsequent versions. A client must implement the following rules before
advertising support. Existing V1 clients must reject the required capability;
otherwise a legacy decoder can ignore the new `transitions` field and load
only `legacy_versions`, while a legacy writer can discard tagged transitions
when replacing the history.

A reader uses known encodings to translate addresses and derive queryable index
coverage. An unknown encoding does not provide queryable coverage through that
mapping. Its affected destinations and all dependent downstream destinations
must be scanned unless independently covered by usable indices with current
addresses. Knowing the destination bitmap alone never permits skipping a scan.
Malformed known encodings are errors, not an excuse to silently omit rows.

A writer preserves an unknown mapping's original serialized transition bytes.
It must retain unknown fields when rewriting the enclosing history; a protobuf
decode/re-encode that drops them is not preservation. It protects all fragments
and index segments depending on that mapping, including transitive dependencies,
from rewriting, merging, and removal. Background maintenance skips protected work;
explicit mutations touching it return unsupported. Unrelated appends remain permitted.

All mapping artifacts reside under `_fri/` in the dataset's declared storage
bases. When an unknown mapping is retained, cleanup conservatively retains these
mapping subtrees because it cannot enumerate that mapping's file references.
This permits future encodings to carry opaque references without losing artifacts.
Known mappings are removed only when no index or remaining mapping chain needs
them. Files are reclaimed only after all retained dataset versions release their
references; manifests containing unknown mappings retain the conservative roots.

Adding a mapping within this common dependency, coverage, and preservation
contract does not allocate another manifest capability bit or require a new
index version. Changing that common contract requires a separate compatibility
decision. Unknown index versions must not be interpreted as version 1.
