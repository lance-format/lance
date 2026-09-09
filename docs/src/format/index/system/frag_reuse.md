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

`IndexMetadata.index_version = 0` retains the existing compaction format and
read/write behavior. Version 1 adds `InlineContent.transitions` at field 2;
field 1 retains the original compaction versions without changing their wire
representation. Each legacy group becomes an ordered-compaction transition
when read as part of a version-1 history.

A transition records ordered sources at field 1 and ordered destinations at
field 2. Exactly one mapping is required: ordered compaction at field 3 or
stable partition at field 4. Additional fields may carry ordinary metadata;
field numbers outside the known mapping alternatives do not identify mappings.
New mapping types require a new FRI `index_version`. Readers that do not support
that version must reject it and request an upgrade before decoding its content.

The outer inline/external content choice is unchanged. External content stores
serialized `InlineContent`, while stable-partition references inside that content
identify separate immutable row-map files. A history has one producer and at most
one consumer for each produced fragment; root source fragments need not have a
producer in the retained history. Lineage must be acyclic. Transition order is
derived from fragment dependencies, not the serialization order.

Source digests record physical and deleted row counts at rewrite time.
Destination digests record creation-time counts with zero deleted rows.
Within each transition, total surviving source rows equal total destination rows.
Across transitions, a fragment's physical row count is unchanged; its deletion
count may increase. These metadata checks do not validate external row-map data.
