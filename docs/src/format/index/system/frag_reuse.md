# Fragment Reuse Index

The Fragment Reuse Index (FRI) is an internal index that keeps existing indices
usable while fragments are compacted or reclustered. It records how old physical
row addresses map to new addresses, without changing surviving row values.

## Use Case 1: Compact Fragments

When data modifications happen against a Lance table, they can trigger compaction
and index optimization at the same time to improve data layout and index coverage.
By default, compaction remaps all indices to prevent read regression.
This means both compaction and index optimization can modify the same index and
cause one process to fail. Typically, compaction fails because it has to modify
all indices and takes longer, resulting in table layout degrading over time.

Fragment Reuse Index allows compaction to defer the index remap process.
Suppose a compaction removes fragments A and B and produces C. At query runtime,
the existing indices are reused by translating their addresses in A and B to
addresses in C. Because indices are typically cached in memory after initial
load, the translated index can be reused by subsequent queries.

## Use Case 2: Recluster Fragments

Fragments are often organized by data arrival order. Queries filtering by fields
such as day or user UUID may therefore need to search many fragments or index
segments. An external engine, such as Spark, can reorganize rows into destination
fragments grouped by those fields.

A **stable partition** assigns source rows to destination fragments while
preserving their relative source order within each destination. This lets FRI
reuse existing indices after reclustering. Sorting or arbitrarily shuffling rows
within a destination requires a different mapping; it is not represented by the
stable-partition format defined here.

## Index Details

FRI uses one system-index entry named `__lance_frag_reuse`. Its
`IndexMetadata.index_details` contains `FragmentReuseIndexDetails`:

```protobuf
%%% proto.message.FragmentReuseIndexDetails %%%
```

The outer `InlineContent` / `ExternalFile` choice applies to the entire history.
Small histories are stored inline; larger histories store the serialized
`InlineContent` in `_indices/<FRI UUID>/details.binpb`. Updating the history
replaces the FRI metadata. Stable-partition row maps are separate immutable files
and are not rewritten with that history.

### FRI Index Versions

`IndexMetadata.index_version` identifies the format required to interpret FRI;
it is not a dataset version or the sequence number of a rewrite.

- **Version 0** retains the existing compaction format and read/write behavior.
  `InlineContent.legacy_versions` keeps the original field number, 1, and wire
  representation of `versions`.
- **Version 1** adds `InlineContent.transitions` at field 2. It supports both
  ordered compaction and stable partition. Each legacy group can be interpreted
  as an ordered-compaction transition, forming one history with the new records.

Adding a transition or a new mapping type does not itself require increasing
`index_version`. Readers may skip unsupported mappings and fall back to scanning
where a complete translation path is unavailable. Writers must reject operations
that require interpreting or maintaining unsupported mappings; operations that
carry the existing history unchanged need not interpret them. Changes to the
shared metadata contract that require reader upgrades must increase `index_version`.

### Shared Transition Metadata

Each transition records ordered `sources` and `destinations`, plus exactly one
mapping. A fragment digest contains its ID, physical row count, and deleted row
count. Source counts describe the rewrite input; destination counts describe the
newly written fragments, with zero deletions.

Sources define scan order; destinations define output order for compaction and
label order for stable partition. The surviving source row count must equal the
total destination row count. Each fragment has at most one producer and one
consumer in the retained history, and the graph must be acyclic. A fragment's
physical row count stays constant across transitions; its deletion count may
increase. Translation follows these dependencies, not the serialized list order.

## Mappings

### Ordered Compaction

Ordered compaction concatenates surviving rows in source-fragment order, with
ascending physical row offsets within each source, then splits that stream into
the ordered destination fragments.

`changed_row_addrs` stores a serialized RoaringTreemap of surviving source
addresses. A physical address uses the upper 32 bits for the fragment ID and the
lower 32 bits for the row offset. A valid source row absent from the bitmap was
deleted.

To translate a surviving row, count the surviving rows in preceding source
fragments and the surviving rows before its offset in its own fragment. Their
sum is its zero-based position in the output stream. Find the destination whose
cumulative physical-row range contains that position and subtract the start of
that range to obtain the destination offset.

The legacy `Group` and the new `OrderedCompaction` mapping use this same ordering
and bitmap representation. No per-row destination labels are required.

### Stable Partition

A stable partition processes source fragments sequentially in their recorded
order and assigns each surviving row a label: the zero-based position of its
destination in `destinations`. Rows with the same label retain their relative
source order. Deleted source rows carry a null label.

The mapping payload is one immutable Lance file,
`_fri/<map_id>/stable_partition.lance`. `map_id` is a UUID independent of the FRI
index UUID, so the file survives FRI metadata rewrites. `map_size_bytes` records
its exact size. An unset `base_id` selects the dataset base; otherwise it selects
the corresponding `Manifest.base_paths` entry. Relocating a dataset requires
copying these files or updating their references.

#### Row Map File Schema

The row map is a Lance file with one nullable `uint16` label per physical source
row. Rows follow source-fragment order, then physical row offset within each
fragment, including deleted rows.

```python
import pyarrow as pa

row_map_schema = pa.schema([pa.field("label", pa.uint16(), nullable=True)])
```

A label tells us which destination receives the row: `0` means
`destinations[0]`, `1` means `destinations[1]`, and `null` means the row was
deleted at rewrite time. It stores the destination's position in the list, not
its fragment ID or row offset. The number of destinations must be between 1
and 65,536, and each non-null label must be smaller than that number.

#### Counts Matrix

Labels tell us the destination fragment. To find the row offset inside that
fragment, we count earlier source rows with the same label.

To avoid reading all earlier labels, source rows are divided into blocks.
For each block and destination, the counts matrix stores the cumulative number
of rows sent to that destination through the end of the block. Null labels are
not counted. `block_rows` must be positive; the current writer uses 65,536.
Only the final block may be shorter.

The matrix is stored in a Lance global buffer. The schema metadata key
`lance:stable_partition:counts_buffer_index` contains its buffer index as a
decimal string. Readers locate the buffer through the normal Lance file metadata.

The buffer starts with a 28-byte header, in this order:

- Magic: four bytes, `LSPC`.
- Version: `u32`, value 1.
- Representation: `u32`, value 0 for the dense grid.
- Number of destinations: `u32`.
- Rows per block: `u32`.
- Total physical source rows: `u64`.

All integers are unsigned and little-endian. The header is followed by
`ceil(total_rows / block_rows) * num_destinations` cumulative `u32` counts,
ordered by block, then destination. No extra bytes are allowed. Unsupported
versions or representations must be rejected.

Counts must never decrease, and a block cannot contribute more live rows than
its length. Final destination counts must match the destination digests. The
label-file row count and header total must match the sum of source physical row
counts. An empty file has no grid rows and zero destination totals.

#### Address Translation

For source fragment `s` and row offset `o`:

1. Find its label-file position: the physical row counts of all preceding source
   fragments, plus `o`.
2. Read the block containing that position. A null label means the row was deleted;
   otherwise label `d` selects `destinations[d]`.
3. The destination row offset is the count for `d` before this block, plus the
   number of occurrences of `d` strictly before this row within the block.
   The count before the first block is zero.

For batch translation, read each requested block once. Initialize destination
counters from the preceding counts row, then scan the block in order. Each
non-null label takes the current counter as its offset and increments it.
Opening FRI history does not require reading labels; translation only needs the
counts matrix and the requested label blocks.

## Expected Use Pattern

When indexing or index remapping cannot keep up with compaction or reclustering,
FRI allows fragment rewrites to proceed while retaining existing indices.
Each rewrite that defers index remapping records its mapping: a reuse version
in FRI index version 0, or transitions in index version 1. A transition must
reference only fragments that are already committed. The rewrite and its
mapping may be published in separate commits: until the mapping lands, the
destination fragments are not covered by any index and are served by scanning,
so correctness never depends on the mapping being present. Atomic composition
of a rewrite with its transition is deferred to a future composite transaction
mechanism.

Once all dependent indices have caught up, the corresponding history can be
trimmed. Cleanup must retain intermediate transitions still needed to translate
old addresses. External mapping files can be deleted only when no retained
dataset version references them.

## Impacts

### Conflict Resolution

Deferring index remapping avoids replacing existing indices during a fragment
rewrite, reducing conflicts with concurrent index building or optimization.
FRI does not remove conflicts between overlapping rewrites. See
[conflict resolution](../../table/transaction.md#conflict-resolution).

### Index Load Cost

Loading affected indices requires translating their stored row addresses.
Ordered compaction uses its bitmap and fragment layouts; stable partition also
reads the required row-map blocks. Translated indices can be cached and reused.
Longer mapping chains add translation work; trimming unused history reduces it.

### Reader and Writer Compatibility

The first commit publishing FRI index version 1 sets
`FLAG_FRAGMENT_REUSE_INDEX` (1024) in both manifest flag fields. Subsequent
manifests retain both bits. FRI index version 0 does not require this flag.

Tables using stable row IDs do not support tagged histories; writers must not
publish `index_version >= 1` on them.

The reader flag prevents older clients from partially interpreting the new
history. The writer flag prevents them from dropping mappings when rewriting
FRI metadata. Clients that do not support the corresponding flag must reject
the read or write and require an upgrade. See
[feature flags](../../table/versioning.md).
