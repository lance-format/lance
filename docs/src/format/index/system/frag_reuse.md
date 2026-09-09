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

## FRI Index Versions

`IndexMetadata.index_version = 0` retains the existing compaction format and
read/write behavior: `InlineContent.legacy_versions` (field 1) is the only
content. Version 1 adds `InlineContent.transitions` at field 2; field 1 keeps
its original field number and wire representation. A version-1 reader lifts
each legacy group into an ordered-compaction transition, so both histories
form one graph.

A transition records ordered source digests at field 1 (scan order; row counts
include deleted rows) and ordered destination digests at field 2 (mapping
order; creation-time counts with zero deletions). Exactly one mapping is
required: ordered compaction at field 3 or stable partition at field 4. Field
numbers outside the declared mapping alternatives carry ordinary metadata and
do not identify mappings; such fields must be registered in the protobuf
definition. New mapping types require a new FRI `index_version`. Readers that
do not support a version must reject it and request an upgrade before decoding
its content. `FLAG_FRAGMENT_REUSE_INDEX` (see
[feature flags](../../table/versioning.md)) fences readers and writers that
predate version negotiation.

The retained history must satisfy: each fragment has at most one producing
transition and at most one consuming transition; lineage is acyclic; within a
transition, total surviving source rows equal total destination rows; across
transitions, a fragment's physical row count is unchanged while its deletion
count may only increase. Transition order is derived from fragment
dependencies, not serialization order.

## Stable Partition Mapping

A stable partition rewrite processes its source fragments sequentially in the
recorded order and assigns every surviving row a label, the index of the
destination fragment that receives it. Rows with equal labels keep their
relative source order. Deleted source rows carry no label. Per-destination
ordering beyond source order is not representable.

The mapping payload is one immutable Lance file,
`_fri/<map_id>/stable_partition.lance`, resolved against the dataset base when
`StablePartition.base_id` is unset and against the identified
`Manifest.base_paths` entry otherwise. `map_id` is a UUID independent of the
FRI index UUID, so the file survives FRI metadata rewrites.
`map_size_bytes` records the exact file size. Operations that relocate the
dataset root are unsupported until they copy the mapping files or rewrite
these references.

### Row Map File Schema

```python
import pyarrow as pa

row_map_schema = pa.schema(
    [pa.field("label", pa.uint16(), nullable=True)],
    metadata={
        # Index of the Lance file global buffer holding the counts matrix.
        b"lance:stable_partition:counts_buffer_index": b"<integer>",
    },
)
```

The file holds one row per physical source row, ordered by the transition's
source fragments in their recorded order and by physical row offset within
each fragment. A non-null value is the row's destination label and must be
less than the number of destinations; null means the row was deleted at
rewrite time.

The counts matrix lives in the global buffer named by the schema metadata key.
Its little-endian layout is: magic `LSPC` (4 bytes), version `u32 = 1`,
representation `u32` (0 = dense grid; other values are reserved and must be
rejected), `num_destinations u32`, `block_rows u32`, `total_rows u64`,
followed by `num_blocks x num_destinations` `u32` values in block-major order,
where `num_blocks = ceil(total_rows / block_rows)`. Grid row `b` holds, for
each destination `d`, the number of rows labeled `d` in blocks `0..=b`; the
final grid row therefore equals every destination's physical row count. Null
rows are not counted, so a block's deleted count is its length minus the sum
of its per-label deltas.

### Reader Navigation

To translate source row `r`: the containing block is `r / block_rows`; the
destination offset is the previous block's cumulative count for the row's
label (zero for the first block) plus the label's rank among rows from the
block start up to `r`, read from the label column. To translate a run of rows,
initialize per-destination counters from the cumulative counts at the
preceding block boundary and sweep the label column forward, assigning
`offset = counter[label]` and incrementing that counter for each surviving
row. Readers can validate a file against its transition: per-destination
totals must equal the destination digests' physical row counts, and
`total_rows` must equal the sum of source digests' physical row counts.

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
