# Indices in Lance

Lance treats indices as independent, redundant data structures layered on top of table row identifiers. This keeps the file format free of built-in search structures and lets index formats evolve independently from the table layout.

Lance supports three main categories of indices to accelerate data access: scalar
indices, vector indices, and system indices.

**Scalar indices** accelerate queries on scalar data types such as integers, timestamps,
and strings. This includes primary skipping structures such as [zone maps](scalar/zonemap.md)
as well as secondary structures such as [B-trees](scalar/btree.md), [bitmap indices](scalar/bitmap.md),
and [full-text search indices](scalar/fts.md). They typically accept predicates such as equality,
range, set-membership, or token matches and return matching row identifiers.

<figure markdown="span">
  ![](./scalar_index.drawio.svg)
</figure>

**[Vector indices](./vector/index.md)** are specialized for approximate nearest neighbor search on
high-dimensional embeddings. Examples include IVF-based layouts and HNSW graphs. Instead of scalar
predicates, vector indices receive a query vector and return row identifiers plus distance scores.

**System indices** are auxiliary structures that support internal table maintenance and row-identifier
resolution. They are not queried directly by end users. Examples include the [Fragment Reuse Index](system/frag_reuse.md),
which supports efficient remapping after compaction.

## Design

Lance indices are designed with the following design choices in mind:

1. **Indices are loaded on demand**: A dataset can be loaded and read without loading any indices.
   Indices are only loaded when a query can benefit from them.
   This design minimizes memory usage and speeds up dataset opening time.
2. **Indices can be loaded progressively**: indices are designed so that only the necessary parts
   are loaded into memory during query execution. For example, when querying a B-tree index,
   it loads a small page table to figure out which pages of the index to load for the given query,
   and then only loads those pages to perform the indexed search. This amortizes the cost of
   cold index queries, since each query only needs to load a small portion of the index.
3. **Indices can be coalesced to larger units than fragments.** Indices are much smaller than
   data files, so it is efficient to coalesce index segments to cover multiple fragments.
   This reduces the number of index files that need to be opened during query execution and
   then number of unique index data structures that need to be queried.
4. **Index files are immutable once written, similar to data files.** They can be modified only
   by creating new files. This means they can be safely cached in memory or on disk without
   worrying about consistency issues.

## Basic Concepts

An index in Lance is defined over a specific column (or multiple columns) of a dataset.
It is identified by its name.

An index is made up of multiple **index segments**, identified by their unique UUIDs.
Each segment is an independent, self-contained index covering a subset of the data.

Each index segment covers a disjoint subset of fragments in the dataset. The segments must cover
all rows in the fragments they cover, with one exception: if a fragment has delete markers at the time
of index creation, the index segment is allowed to not contain the deleted rows. The fragments an index
covers are those recorded in the `fragment_bitmap` field.

Index segments together **do not** need to cover all fragments. This means an index isn't required to
be fully up-to-date. When this happens, engines can split their queries into indexed and unindexed
subplans and merge the results.

<figure markdown="span">
  ![](./starter-example.drawio.svg)
  <figcaption>Abstract layout of a typical dataset, with three fragments and two indices.
  </figcaption>
</figure>

Consider the example dataset in the figure above:

- The dataset contains three fragments with ids 0, 1, 2. Fragment 1 has 10 deleted rows, indicated
  by the deletion file.
- There is an index called "id_idx", which has two segments: one covering fragments 0 and another covering
  fragment 1. Fragment 2 is not covered by the index. Queries using this index will need to query both
  segments and then scan fragment 2 directly. Additionally, when querying the segment covering fragment 1,
  the engine will need to filter out the 10 deleted rows.
- There is another index called "vec_idx", which has a single segment covering all three fragments.
  Because it covers all fragments, queries using this index do not need to scan any fragments directly.
  They do, however, need to filter out the 10 deleted rows from fragment 1.

## Segment Coverage and Freshness

Two `IndexMetadata` fields describe what a segment covers and how fresh it is.

`fragment_bitmap` records the fragments the segment covers.

- It may contain ids of fragments that have since been deleted.
- When the dataset has a [Fragment Reuse Index](system/frag_reuse.md) (FRI), it may record a
  compacted fragment by its id before compaction or by its id after compaction. Writers may persist
  either form, and readers must accept both.
- Readers normalize the bitmap by mapping it through the FRI. Fragment ids are never reused, so
  mapping an already-mapped bitmap has no effect.
- For indices whose results are stable row ids, it records the fragments whose rows, identified by
  stable row id, are represented in the segment.

`dataset_version` is the segment's watermark. The segment reflects the data of every covered
fragment as of this dataset version. For indices whose results are row addresses, that includes
row addresses. A consequence is that the bitmap does not determine which address space the index
data is in; `dataset_version` does, as follows:

- An [overlay](../table/data_overlay_file.md) with `committed_version > dataset_version` is not
  reflected in the segment.
- An FRI version entry whose own `dataset_version` is greater than or equal to the segment's
  `dataset_version` has not been applied to the segment's data. The segment's row addresses must
  still be translated through it. An FRI version entry's `dataset_version` is the version that
  immediately precedes the commit of the rewrite it records.

Let B be the version whose data was read into a segment. Let W be the smallest of:

- `committed_version - 1` of each overlay committed after B on an indexed field of a covered
  fragment, and
- the `dataset_version` of each FRI version entry added after B whose old or new fragments
  intersect the segment's `fragment_bitmap`.

If there are none, W is the current dataset version. A segment's `dataset_version` must never
exceed W: a larger value hides changes the segment does not reflect, and lets a writer discard FRI
entries the segment still needs. Any value at or below B is valid. A lower value may cause more
rows to be treated as stale, but never yields wrong results.

The **indexed fields** of a segment are the union of its `fields` and `covering_fields`, including
all of their nested children.

The **live coverage** of a segment is its normalized `fragment_bitmap` intersected with the
fragments of the current dataset version. Under a tagged FRI history (FRI index version 1), live
coverage is derived for the index as a whole, from the union of the bitmaps of all segments that
share its name. Segments of an index must have disjoint `fragment_bitmap`s as stored.

## Index Storage

The content of each index is stored at the `_indices/{UUID}` directory under the [base path](../table/layout.md#base-path-system).
We call this location the **index directory**.
The actual content stored in the index directory depends on the index type. These can be
arbitrary files defined by the index implementation. However, often they are made up of
Lance files containing the index data structures. This allows reuse of the existing Lance
file format code for reading and writing index data.

## Creating and Updating Index Segments

Index segments are created and updated through a transactional process:

1. **Build the index data**: Read the relevant column data from the fragments to be indexed
   and construct the index data structures. Write these to files in a new `_indices/{UUID}`
   directory, where `{UUID}` is a newly generated unique identifier.

2. **Prepare the metadata**: Create an `IndexMetadata` message with:
   - `uuid`: The newly generated UUID
   - `name`: The index name (must match existing segments if adding to an existing index)
   - `fields`: The columns the index is keyed on, in key order.
   - `covering_fields`: An independent declaration of the columns whose values the index carries,
     letting a query that only projects those columns be answered without a fragment take. A field
     may appear in both lists when the index carries a value it is also keyed on. Empty for an index
     that carries no extra columns. Declaring a column here does not by itself make it servable --
     see [Serving carried columns](#serving-carried-columns).
   - `fragment_bitmap`: The set of fragment IDs covered by this segment
   - `dataset_version`: The dataset version the index data was read from. See
     [Segment Coverage and Freshness](#segment-coverage-and-freshness).
   - `index_details`: Index-specific configuration and parameters
   - `version`: The format version of this index type
   - See the full protobuf definition in [table.proto](https://github.com/lance-format/lance/blob/main/protos/table.proto).

3. **Commit the transaction**: Write a new manifest that includes the new index segment
   in its `IndexSection`. This is done atomically using the same transaction mechanism
   as data writes.

When updating a column in place (without deleting the row), the engine must remove the
affected fragment IDs from the `fragment_bitmap` field of any index segment whose `fields`
or `covering_fields` include that column. This marks those fragments as needing re-indexing
without invalidating the entire segment and prevents invalid data from being read from the
index.

### Maintaining `fragment_bitmap` and `dataset_version`

A writer that produces a segment, by building, merging, or remapping, chooses its
`dataset_version` in one of two ways:

- **Keep a conservative version.** Use a value no greater than the B of every source segment the
  new segment's data came from.
- **Stamp its read version R.** Before doing so, the writer must catch up every covered fragment
  from each source's B through R:
    - Each overlay on an indexed field is either folded into the index data or its fragment is
      removed from the `fragment_bitmap`. Folding must be idempotent: applying an overlay the
      index already reflects must leave the index unchanged.
    - Each FRI version entry is applied to the index data.

A writer must not change a segment's `dataset_version` when it rebases or retries a commit. The
stamp must stay the version the segment's data was read from.

Other operations update existing segments as follows:

- **Delete** leaves `fragment_bitmap` and `dataset_version` unchanged. Deleted fragments are
  filtered at read time.
- **In-place updates** of an indexed field remove the modified fragments from the
  `fragment_bitmap`, as described above.
- **Updates that move rows to new fragments** may add the new fragments to the `fragment_bitmap`
  of a segment whose results are stable row ids, if no indexed field changed. `dataset_version`
  is unchanged.
- **Compaction** is described in [Compaction and remapping](#compaction-and-remapping).
- **Overwrite** removes all index segments.

A new segment may be committed on a dataset version newer than its `dataset_version`, for example
when it was built separately and committed later. The manifest at that `dataset_version` need not
be an ancestor of the version the commit is built on, for example after a restore. Before
committing such a segment, the writer must remove from its `fragment_bitmap`
every fragment whose indexed fields, overlays, or lineage differ between the manifest at
`dataset_version` and the version the commit is built on.

A segment whose live coverage is empty may be removed if another segment of the same index still
has non-empty live coverage. If every segment of an index has empty live coverage, the writer
must keep the one with the earliest `created_at`, so that the index definition is not lost.
Segments without `created_at` sort after those with one, and ties are broken by the lowest
`dataset_version`.
Under a tagged FRI history, a segment must not be removed based on its own derived coverage.

## Index Compatibility

Before using an index segment, engines must verify they support it:

1. **Check the index type**: The `index_details` field contains a protobuf `Any` message
   whose type URL identifies the index type (e.g., B-tree, IVF, HNSW). If the engine
   does not recognize the type, it should skip this index segment.

2. **Check the version**: The `version` field in `IndexMetadata` indicates the format
   version of the index segment. If the engine does not support this version, it should
   skip this index segment. This allows index formats to evolve over time while
   maintaining backwards compatibility.

When an engine cannot use an index segment, it should fall back to scanning the
fragments that would have been covered by that segment.

### Covering-field contracts

A manifest that uses independent key and covering declarations must set
`FLAG_COVERED_INDEX_METADATA` and `FLAG_INDEPENDENT_COVERING_FIELDS` in both its reader
and writer feature flags. In this contract, `fields` contains only the key fields and
`covering_fields` separately contains the carried fields. The lists have no positional
relationship and may overlap. The index dependency set is their union.

For backwards compatibility, a manifest with `FLAG_COVERED_INDEX_METADATA` but without
`FLAG_INDEPENDENT_COVERING_FIELDS` uses the legacy contract: the carried columns form the
trailing suffix of `fields` in emission order, so `covering_fields` equals
`fields[len(fields) - len(covering_fields)..]`. A column the index is both keyed on and
carries appears twice, once as a key and once in the carried suffix: `fields = [1, 1]` with
`covering_fields = [1]` is keyed on `1` and carries `1`. Readers must select the contract
from the manifest flags before interpreting any `IndexMetadata`.

### Activating independent declarations

Setting `FLAG_INDEPENDENT_COVERING_FIELDS` reinterprets every `IndexMetadata` in the
manifest's index section, so a writer must not set it while any entry remains in legacy
form. A legacy `fields = [vector, payload]` with `covering_fields = [payload]` reads
under the new contract as keyed on both columns, and a query on `payload` would then be
planned against an index that is really keyed on `vector`.

Activation is therefore a single atomic manifest change. A writer that sets the flag must
normalize every entry of the index section in that same manifest to the key prefix:
replace `fields` with `fields[..len(fields) - len(covering_fields)]` and leave
`covering_fields` unchanged. For example, `fields = [1, 1]` with `covering_fields = [1]`
becomes `fields = [1]` with `covering_fields = [1]`. A
writer that cannot normalize an entry must not set the flag; it may instead set the flag
only on a manifest in which no entry declares covering fields at all. No committed
manifest may set the flag while holding a legacy-form entry.

A commit that loses a conflict and retries must normalize the index section of the
manifest it is retrying against rather than the one it first read, so an entry another
writer committed in between is never carried forward unnormalized.

Derived manifests retain the flag. Every manifest built from one that sets it keeps the
bit set, unless a writer converts every entry back to the legacy contract in one atomic
change of its own. Because `FLAG_COVERED_INDEX_METADATA` is recomputed from covering
declarations on every commit, a writer that sets `FLAG_INDEPENDENT_COVERING_FIELDS` must
also keep `FLAG_COVERED_INDEX_METADATA` set in both feature words whenever the independent
bit is set, including manifests with no covering entries and manifests that drop the last
covering index. The two bits are retained together until an atomic reverse conversion
clears them together.

### Serving carried columns

`IndexMetadata.covering_fields` records the columns an index segment *declares* it
carries. It does not establish that the segment's storage holds their values.

**The segment's storage schema is authoritative.** Before answering a query from a
carried column, an engine must confirm that column is present and bound to the declared
logical field in the storage it opened, and fall back to a take against the base table
when it cannot. A segment whose metadata identifies a column its storage does not hold is
a legal state, not corruption: a maintenance operation that cannot carry the payload
through a rebuild is permitted to withdraw it and leave the `covering_fields` listed in
the metadata.

!!! note "Capability varies by segment"

    Whether a segment's storage holds a declared column depends on the index type, on the
    writer that produced the segment, and on what later maintenance did to it, so one
    logical index may hold values for some of its segments and not others. An engine
    therefore verifies each selected segment rather than inferring capability from the
    index type, the writer version, or the `covering_fields` metadata alone, and serves
    from the base table every column it cannot verify.

## Loading an index

When loading an index:

1. Get the offset to the index section from the `index_section` field in the [manifest](../table/index.md#manifest).
2. Read the index section from the manifest file. This is a protobuf message of type `IndexSection`, which
   contains a list of `IndexMetadata` messages, each describing an index segment.
3. Read the index files from the `_indices/{UUID}` directory under the dataset directory,
   where `{UUID}` is the UUID of the index segment.

!!! tip "Optimizing manifest loading"

    When the manifest file is small, you can read and cache the index section eagerly. This avoids
    an extra file read when loading indices.

The `IndexMetadata` message contains important information about the index segment:

- `uuid`: the unique identifier of the index segment.
- `fields`: the columns the index is keyed on, in key order.
- `covering_fields`: the independently declared columns whose values the index carries alongside
  its own data. A field may also occur in `fields`. Empty for an index that carries no extra
  columns. Every id in `covering_fields` names a top-level field. Covering a struct column
  carries the whole struct, its children included, as one column. This declaration is not
  authoritative for what the segment can serve -- see
  [Serving carried columns](#serving-carried-columns) and the legacy interpretation under
  [Covering-field contracts](#covering-field-contracts).
- `fragment_bitmap`: the set of fragment IDs covered by this index segment.
- `dataset_version`: the segment's watermark. The manifest copy is authoritative over any copy an
  index type stores in its own files. See
  [Segment Coverage and Freshness](#segment-coverage-and-freshness).
- `index_details`: a protobuf `Any` message that contains index-specific details, such as index type,
  parameters, and storage format. This allows different index types to store their own metadata.

<details>
  <summary>Full protobuf definitions</summary>

There are both part of the `table.proto` file in the Lance source code.

```protobuf
%%% proto.message.IndexSection %%%

%%% proto.message.IndexMetadata %%%
```

</details>

## Handling deleted and invalidated rows

Since index segments are immutable, they may contain references to rows that have been deleted
or updated. These should be filtered out during query execution.

<figure markdown="span">
  ![](./indices-fragment handling.drawio.svg)
  <figcaption>Representation of index segment covering fragments that have deleted rows,
  completely deleted fragments, and updated fragments.
  </figcaption>
</figure>

Deletion files and fragment ids describe the current dataset version. When an FRI applies to the
segment, its row addresses must be translated through the FRI before they are filtered.

There are four situations to consider:

1. **A fragment has some deleted rows.** A few of the rows in the fragment have been marked
   as deleted, but some of the rows are still present. The row addresses from the deletion
   file should be used to filter out results from the index.
2. **A fragment has been completely deleted.** This can be detected by checking if a
   fragment ID present in the fragment bitmap is missing from the dataset.
   Any row addresses from this fragment should be filtered out.
3. **A fragment has had one of the index's columns updated in place.** This cannot be detected
   just by examining metadata. To prevent reading invalid data, the engine should filter out any
   row addresses that are not in the index's current `fragment_bitmap`.
   The column need not be one the index is keyed on: every column in the union of `fields` and
   `covering_fields` counts. A carried column can be updated while the keyed column is untouched,
   and a segment left covering that fragment would answer from an obsolete carried value.
4. **A fragment has an updated value in an [overlay file](../table/data_overlay_file.md).**
   This can be detected by checking if any of the fragments in the index's `fragment_bitmap`
   have overlay files. For each overlay whose `committed_version` is greater than the index
   segment's `dataset_version`, the overlay carries updated values not reflected in the index,
   so its covered rows must be excluded from index results. Excluded rows are re-evaluated
   against their current (overlaid) values on the flat path — dropping them without
   re-evaluation would silently lose rows that match under the new value. Exclusion is
   field-aware: only overlays covering a column in the union of the index's `fields` and
   `covering_fields` matter. Restricting this to the keyed columns would leave a fragment
   covered after an overlay updated a carried one, and the index would then serve a stale
   carried value.
   You may exclude just the affected rows or the whole fragment; the latter is simpler and
   safer but re-evaluates more rows than necessary.
   See [Data Overlay Files](../table/data_overlay_file.md#index-integration)
   for the exclusion set, re-evaluation, and correctness invariant.

## Compaction and remapping

When fragments are compacted, the row addresses of the rows in the fragments change.
This means that any index segments referencing those fragments will no longer point
to existing row addresses. There are three ways to handle this:

<figure markdown="span">
![](./indices-compaction.drawio.svg)
</figure>

1. Do nothing and let the index segment not cover those fragments anymore. This approach is
   simple and valid, but it means compaction can immediately make an index out-of-date. This
   is the worst options for query performance.

2. Immediately rewrite the index segments with the row addresses remapped. This approach
   ensures the index is kept up-to-date, but it incurs significant write amplification
   during compaction.

3. Create a [Fragment Reuse Index](system/frag_reuse.md) that maps old row addresses to new
   row addresses. This allows readers to remap the row addresses in memory upon reading
   the index segments. This approach adds some IO and computation overhead during query
   execution, but avoids write amplification during compaction.

With option 2, the writer replaces the compacted fragment ids in the `fragment_bitmap` with their
new ids and leaves `dataset_version` unchanged.

With option 3, the segment's data and `dataset_version` are unchanged. The `fragment_bitmap` may
keep the old fragment ids, or a later commit may persist it already mapped through the FRI; see
[Segment Coverage and Freshness](#segment-coverage-and-freshness). A writer may later remap the
segment's data through the FRI. It may then stamp its read version, following
[Maintaining `fragment_bitmap` and `dataset_version`](#maintaining-fragment_bitmap-and-dataset_version).

If compaction materializes an overlay on an indexed field that is newer than a segment's
`dataset_version`, the overlay's values enter the base data and no overlay remains to exclude
those rows. The writer must therefore do one of the following:

- fold the overlay's values into the segment's data, or
- remove from the segment's `fragment_bitmap` the compacted fragment, by both its old and new ids.

## Stable Row ID for Index

Indices can optionally use stable row IDs instead of row addresses. A stable row ID is a
logical identifier that remains constant even when rows are moved during compaction.

**Benefits:**

- No remapping needed after compaction
- Updates only invalidate the index if data in the union of its `fields` and
  `covering_fields` changes

**Tradeoffs:**

- Requires an additional lookup to translate stable row IDs to physical row addresses
  at query time

This feature is currently experimental. Performance evaluation is ongoing to determine
when the tradeoff is worthwhile.
