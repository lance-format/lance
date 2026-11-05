# Durable index row references

## Overview

Index files are immutable once written, while Lance assigns fragment IDs and new stable row IDs when it builds a manifest. If an index is written before those IDs are final, a concurrent append can claim the expected IDs while the prepared index file still contains the earlier ones. Updating `fragment_bitmap` alone does not change the references stored inside the file.

`IndexMetadata` has an optional `row_binding` field that maps the references stored in an index segment to their committed IDs. A commit retry updates the binding in metadata and leaves the index files unchanged.

Whether a segment carries a binding depends on what its files store:

- An index written with committed identities has no binding.
- An index built with local references keeps its binding even if no concurrent writer commits first.
- A compaction or index rewrite drops the binding once the replacement files store committed identities.

Publishing an append and its index together is one use of this contract, and any writer that prepares index files before commit follows the same rules.

## Row references

**Physical row addresses** pack a fragment ID and a row offset into a 64 bit integer:

```text
row_address = (fragment_id << 32) | row_offset
```

Compaction can move a row to a new address, and the [Fragment Reuse Index](system/frag_reuse.md) translates that address at read time without modifying the index file.

**Stable row IDs** track logical rows across compaction. Once committed, a row's stable ID never changes, though IDs assigned to staged rows can shift if another writer advances `next_row_id` before commit. See [row ID assignment](../table/row_id_lineage.md#row-id-assignment).

An index segment can store either kind of reference. Some indexes store physical addresses even when stable row IDs are enabled. A BTree on `_rowid` stores stable row IDs as keys and physical addresses as values, so a single segment can need both bindings.

## Row binding

`row_binding` is field 14 on `IndexMetadata`. It has separate fields for physical addresses and stable row IDs, and each field uses a `oneof` so future encodings can be added alongside the initial ones.

### Fragment array

For physical addresses, the index file stores fragment ordinals starting at `0`. Each ordinal is a position in that segment's array of committed fragment IDs:

```text
Index file:            (0, 5)   (1, 8)   (2, 3)
Committed fragments:  [42, 47, 53]
```

Here, `(1, 8)` resolves to `(47, 8)` with the row offset unchanged.

The writer fixes the ordinal order when it builds the index. A commit retry can replace the committed fragment IDs in the array, but it must not reorder the positions. Each segment has its own array, so two segments can both use ordinal `0`.

`fragment_bitmap` records the committed fragments the segment currently covers, while the array records how to decode references inside the index files. Removing a fragment from `fragment_bitmap` must not remove or reorder array entries. Every fragment listed in `fragment_bitmap` must have a valid array entry.

### Stable row ID spans

Indexes such as NGram store stable row IDs directly, so a fragment ID mapping does not change their references.

For a contiguous run of newly assigned row IDs, one span maps the starting ID stored in the index to the starting ID assigned at commit:

```text
stored_base    = 10
committed_base = 15
count          = 10

Stored:     10..19
Committed:  15..24
```

Stored row ID `12` resolves to `17`. A segment can hold several spans, including spans with different offsets.

Caller supplied row IDs that are already final need no span when every stored row ID in the segment is already committed. Multiple spans (including spans of length one) can represent a noncontiguous caller supplied prefix alongside an assigned tail that shifted on rebase. A writer that cannot encode its stored row IDs as spans must reject the commit.

When a stable row ID binding is present, every stored row ID in the segment must match exactly one span, and any unmapped ID is an error.

### Row IDs stored as search keys

In most indexes, search keys are column values and the binding only translates the row references returned by the index.

A BTree on `_rowid` (field `-3`) is different because it stores stable row IDs as search keys and physical addresses as values. To read a bound `_rowid` BTree, the reader must translate query predicates into the stored row ID space before searching, and then resolve the returned physical addresses. Range queries across multiple spans may require separate lookups for each span. Writers must reject indexes with provisional `_rowid` keys until that predicate translation is supported. See [Stable Row ID Reverse Lookup](scalar/btree.md#stable-row-id-reverse-lookup).

## Format rules

* **Build:** When writing an index with local references, the writer records the fragment array order and any provisional row ID spans required to decode the files.
* **Commit:** Manifest construction assigns the committed IDs and writes `row_binding` with the data and the index in one commit. A retry updates `row_binding` without rewriting the index files.
* **Read:** Readers translate local references before using them as dataset identities. Physical addresses resolve through the fragment array before Fragment Reuse Index translation, and stable row IDs resolve through the row ID spans. A reader must never treat a stable row ID as a physical address.
* **Coverage:** `fragment_bitmap` stores committed fragment IDs for query planning. An ordinal in an index file is never treated as a committed fragment ID.
* **Maintenance:** An operation that keeps index files keeps their `row_binding`. An operation that rewrites index files resolves the old references and writes either committed identities (with no binding) or a new binding. Compaction must preserve the binding, rewrite the index with committed identities, or safely drop the index, and if none of these is supported it must reject the operation before commit.
* **Validation:** If `row_binding` is absent, the index files already store committed identities. If `row_binding` is present, an empty message, an unknown `oneof` variant, an out of bounds ordinal, an invalid span, or spans that overlap in the stored or committed ID space is an error. Readers must reject any binding they cannot interpret rather than fall back to the raw stored number.
* **Compatibility:** A dataset with a bound index segment sets both a reader feature flag and a writer feature flag so implementations that do not recognize the flag refuse the dataset. Clones and other metadata copies that keep the index files must keep the binding and both flags. Writers must not emit field 14 until the feature flag bit is assigned. See [feature flags](../table/versioning.md#feature-flags).

## Protobuf

```protobuf
%%% proto.message.IndexRowBinding %%%
```

```protobuf
%%% proto.message.FragmentBinding %%%
```

```protobuf
%%% proto.message.FragmentArray %%%
```

```protobuf
%%% proto.message.StableRowBinding %%%
```

```protobuf
%%% proto.message.RowIdSpans %%%
```

```protobuf
%%% proto.message.RowIdSpan %%%
```

## See also

* [Index format](index.md)
* [Row addresses and stable row IDs](../table/row_id_lineage.md)
* [Fragment Reuse Index](system/frag_reuse.md)
* [Feature flags](../table/versioning.md#feature-flags)
* [Transaction V2](https://github.com/lance-format/lance/pull/7954)
* [Append and index in one commit](https://github.com/lance-format/lance/issues/10386)
* [Staged transactions](https://github.com/lance-format/lance/pull/10722)
