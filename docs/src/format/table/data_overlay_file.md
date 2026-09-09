# Data Overlay Files

!!! warning "Experimental"

    This feature is currently experimental and not yet supported in any library.

<!-- TODO: When overlay file support is implemented, update this note to state
     the released version that first supports the feature (and drop the
     "experimental" framing once it is stable). -->

!!! note "Overlay files require feature flag 64 (data overlay files)"

    A reader or writer that does not understand overlay files must refuse a
    dataset that uses them. Silently ignoring an overlay would return stale base
    values, which is a correctness bug rather than a degraded experience.

Overlay files supply new values for a subset of `(row offset, field)` cells
within a fragment **without rewriting the fragment's base data files**. They make
updates cheap when only a small fraction of rows and/or columns change: instead
of rewriting whole columns or moving rows to a new fragment, a writer appends a
small file carrying just the changed cells.

This is Lance's third mechanism for changing data in place, alongside
[deletion files](index.md#deletion-files) (which remove rows) and
[data evolution](index.md#data-evolution) (which adds or rewrites whole columns).
An overlay changes individual cells.

## Concepts

### Coverage and resolution

Each overlay declares which cells it provides through a **coverage** bitmap (or,
for sparse overlays, one bitmap per field). The bitmaps index **physical row
offsets**. They include deleted rows and are stable even as deletion vectors change.

To resolve a cell `(offset, field)` on read, walk the fragment's overlays from
**newest to oldest**. The first overlay that covers `(offset, field)` wins; its
value is used. If no overlay covers the cell, the value falls through to the base
data file (or is `NULL` if no base data file holds that field).

Precedence among overlays is determined by:

1. `committed_version` — higher wins (see [Versioning](#versioning-and-ordering)).
2. Position in `DataFragment.overlays` as a tiebreaker — a later entry is newer.

A covered offset whose value is `NULL` overrides the cell **to** `NULL`. This is
distinct from an offset that is simply absent from the bitmap, which falls
through to the base. Coverage, not value-nullness, decides whether an overlay
applies.

### Interaction with deletions

Deletions take precedence over overlays. If a row offset is marked deleted in the
fragment's deletion file, any overlay value for that offset is dead and is
ignored, regardless of commit order.

### Physical layout

An overlay's data file stores **one value column per field**, in the order of
`data_file.fields`. It does **not** store a row-offset key column. The position of
a covered offset's value within its column is the **rank** of that offset in the
field's coverage bitmap — the number of set bits below it. Resolving a cell is a
rank lookup plus one value fetch, with no separate offset column to store or
search.

Because different fields may cover different offset sets, the value columns of a
single sparse overlay may have **different lengths**. The Lance file format
permits columns of differing item counts within one file, so a sparse overlay is
representable as a single file. (See [Writer support](#writer-support) for the
current implementation status.)

### Dense vs. sparse overlays

A single overlay is one of two shapes:

- **Dense (rectangular).** One `shared_offset_bitmap` applies to every field. Every
  covered offset has a value for every field. This is the common case for a plain
  `UPDATE`, where one `SET` list is applied to one set of rows.
- **Sparse.** A `FieldCoverage` carries one bitmap per field, used when different
  fields cover different offset sets — for example a `MERGE` with multiple
  `WHEN MATCHED` branches, where different rows update different columns. A dense
  overlay would have to widen to the bounding rectangle and fill the untouched
  cells with their current values (post-images), which for wide columns such as
  embeddings means re-storing data that did not change. A sparse overlay stores
  exactly the changed cells.

## Protobuf

<details>
<summary>DataOverlayFile protobuf message</summary>

```protobuf
%%% proto.message.DataOverlayFile %%%
```

</details>

<details>
<summary>FieldCoverage protobuf message</summary>

```protobuf
%%% proto.message.FieldCoverage %%%
```

</details>

## Versioning and ordering

Overlays reuse the dataset version as their ordering clock rather than
introducing a separate generation counter.

`committed_version` is the dataset version at which an overlay **became
effective** — the version of the commit that introduced it, **not** the version
it was read from. It is stamped at commit time and re-stamped if the commit is
retried, in the same way as the created-at / last-updated-at version sequences.

This single value drives every ordering decision:

- **Overlay vs. overlay** (read precedence): higher `committed_version` wins.
- **Overlay vs. index** (query correctness): an index records the
  `dataset_version` it was built from. An index whose `dataset_version >=
  committed_version` already incorporates the overlay. An overlay whose
  `committed_version > index.dataset_version` is newer than the index and its
  cells must be excluded from index results and re-evaluated.
- **Scheduler signal**: the gap between an overlay's `committed_version` and an
  index's `dataset_version`, or between an overlay and the base, is a staleness
  measure the compaction scheduler can use.

!!! note "Why effective version, not read version"

    Suppose an overlay reads version 5 and commits at version 6, while an index
    is built reading version 5 (before the overlay) and commits at version 7 with
    `dataset_version = 5`. If the overlay stored its *read* version (5), the test
    `5 > 5` is false, the row would not be excluded, and the index — which never
    saw the overlay — would return a stale result. Storing the *effective*
    version (6) makes `6 > 5` true, the cell is excluded and re-evaluated, and the
    result is correct.

## Index integration

Building an index over a fragment that has overlays does **not** require dropping
the fragment from the index's coverage. The fragment stays indexed, and the query
path reconciles overlays at query time using an **exclusion set**.

The exclusion set for an index on field `F` is the union of the coverage bitmaps,
restricted to field `F`, of every overlay whose `committed_version >
index.dataset_version`. The exclusion is **field-aware**: an overlay that touches
only unrelated columns does not exclude anything from the index on `F`.

`F` here ranges over every field in the index's `fields`, not only the ones it is
keyed on. An index that carries columns it is not keyed on (see
[`covering_fields`](../index/index.md#serving-carried-columns)) depends on those
columns too: an overlay updating a merely-carried column leaves the keyed value
correct while making the carried value stale, so it must exclude those rows just
the same.

The query then proceeds as:

1. Run the index search as usual, producing candidate rows.
2. Remove any candidate in the exclusion set. (Its indexed value may be stale.)
3. **Re-evaluate** the excluded rows against their current values — the same flat
   path already used for the unindexed tail of fragments. For a scalar predicate
   this re-applies the filter; for a vector query it re-scores the row's current
   vector. Rows that still match are added back to the result.

Step 3 is what makes exclusion correct rather than merely safe: removing a row
from index candidates without re-evaluating it would silently drop a row that
should match under its new value.

Exclusion is always *sufficient* because a write changes a cell only by adding an
overlay, and that overlay's `committed_version` — the version of the commit that
adds it — necessarily exceeds the `dataset_version` of any pre-existing index. So
every cell a write changes is guaranteed to fall in that index's exclusion set.
Compaction may remove an overlay only if no index still relies on it for exclusion
(see [Compaction](#compaction)).

## Compaction

Overlays accumulate read cost — every overlay is a bitmap to test, a possible
file to open, and additional work to interleave values. Compaction bounds that cost in two modes:

- **Overlay → overlay.** Merge several overlays into fewer, computing the
  post-image per `(offset, field)` by walking the merged overlays newest-first.
  The merged overlay takes the **maximum** `committed_version` of its inputs, so
  the exclusion semantics are preserved. The merged overlays must be **contiguous
  in `committed_version`** — with overlays at v10, v30, and v50 you cannot merge
  just v10 and v50, because stamping the result v50 would incorrectly promote
  v10's values above the intervening v30 for any cell v30 also covers. Indexes can
  still be re-used, but they may now need to exclude more rows. This is cheap to
  write and does not touch the base.
- **Overlay → base.** Fold overlays into a fresh base data file, computing the
  post-image for every covered cell, then clear the overlays. The base is
  complete, so every post-image is well defined. Overlay offsets are physical, so
  they cannot survive a rewrite that reorders rows; folding therefore materializes
  values rather than carrying overlays forward.

!!! warning "Folding an indexed field must update its index"

    An overlay→base fold removes the overlay, which removes the exclusion signal
    that kept an index correct. Folding an overlay that covers an indexed field
    `F` is therefore equivalent to a column rewrite of `F` and must, in the same
    commit, either rebuild the index to a `dataset_version` at least the folded
    overlay's `committed_version`, or remove the fragment from the index's
    coverage so the rows fall to the flat path. Otherwise the index would serve
    stale values with no overlay to exclude them. This is the same rule that
    already governs rewriting a column that an index is built on.

When a fragment with overlays is compacted by a row-rewriting operation
(`RewriteRows`, which produces new fragments with new row addresses), the
overlays are folded into the new base as part of the rewrite, and existing
[fragment-reuse remapping](row_id_lineage.md) handles the row-address changes as
it does today.

## Row lineage

An overlay write updates the `last_updated_at_version` of every covered row, so
change-data-feed and time-travel queries observe the update. Because overlays are
addressed by physical offset, they do **not** require stable row IDs to be
enabled; lineage updates apply only when those features are on.

## Worked example

The following example illustrates how overlays function across their lifecycle, to make the rules above concrete.

A table `users` with stable row IDs enabled and these fields:

| field id | name      | type                    |
|----------|-----------|-------------------------|
| 1        | id        | `int32` (primary key)   |
| 2        | name      | `utf8`                  |
| 3        | age       | `int32`                 |
| 4        | embedding | `fixed_size_list<f32,4>`|

Created at version 1 as a single fragment `0` with one base data file
`data/file0.lance` holding all four columns. `physical_rows = 4`:

| offset | id | name  | age | embedding        |
|--------|----|-------|-----|------------------|
| 0      | 1  | Alice | 30  | …                |
| 1      | 2  | Bob   | 25  | …                |
| 2      | 3  | Carol | 40  | …                |
| 3      | 4  | Dave  | 22  | …                |

A BTree scalar index on `age` is built at version 1, covering fragment `0`
(`dataset_version = 1`).

### Step 1 — write an overlay

```sql
UPDATE users SET age = age + 1 WHERE id IN (2, 4);   -- Bob (offset 1), Dave (offset 3)
```

This touches one field (`age`) for two rows, so the writer emits a dense overlay —
one shared bitmap covering both offsets — and commits it as version 2. Fragment
`0` gains:

```text
DataOverlayFile {
  data_file: { path: "data/overlay-<uuid>.lance", fields: [3], column_indices: [0] }
  coverage:  shared_offset_bitmap = {1, 3}
  committed_version: 2
}
```

The overlay file stores a single `age` column with two values, `[26, 23]`, at
ranks `{1,3}.rank(1) = 0` and `{1,3}.rank(3) = 1`. `last_updated_at_version` is
set to 2 for offsets 1 and 3.

### Step 2 — read

`SELECT id, age FROM users` reads base ages `[30, 25, 40, 22]`. For `age`
(field 3), the overlay covers offsets 1 and 3, so `age[1]` is replaced with the
overlay value at rank `{1,3}.rank(1) = 0` → `26`, and `age[3]` with the value at
rank `{1,3}.rank(3) = 1` → `23`. Result ages: `[30, 26, 40, 23]`.

### Step 3 — index query

```sql
SELECT * FROM users WHERE age = 26;
```

The `age` index was built at `dataset_version = 1`; the overlay's
`committed_version` is 2. Since `2 > 1`, the overlay's coverage for `age`, `{1, 3}`,
is the exclusion set for this query.

- The index (built at v1) holds Bob's *old* `age = 25`, so a lookup for `26`
  returns nothing from the index.
- The whole exclusion set is re-evaluated on the flat path, not just the rows the
  index returned. Offset 1's current `age` (26, via the overlay) matches, so Bob
  is returned; offset 3's current `age` (23) does not match and is dropped.

The mirror case `WHERE age = 25` shows exclusion preventing a stale hit: the index
returns offset 1 (stale `25`), but offset 1 is excluded, re-evaluated to `26`, and
correctly dropped.

### Step 4 — a second, non-rectangular write

```sql
MERGE INTO users USING staged ON users.id = staged.id
WHEN MATCHED AND staged.kind = 'rename'  THEN UPDATE SET name = staged.name        -- Carol(2), Dave(3)
WHEN MATCHED AND staged.kind = 'embed'   THEN UPDATE SET embedding = staged.embedding -- Bob(1)
```

`name` is updated for offsets `{2, 3}` and `embedding` for offset `{1}` — different
fields over different rows. This is a sparse overlay, committed as version 3:

```text
DataOverlayFile {
  data_file: { path: "data/overlay-<uuid2>.lance", fields: [2, 4], column_indices: [0, 1] }
  coverage:  field_coverage { offset_bitmaps: [ {2,3},   {1} ] }
  //                                     name (field 2)  ^    ^ embedding (field 4)
  committed_version: 3
}
```

The file's `name` column has **two** values (`["Caroline", "David"]`, at
ranks 0 and 1 of `{2,3}`) and its `embedding` column has **one** value (at rank 0
of `{1}`) — columns of different lengths in one file.

### Step 5 — read after the second write

`SELECT name, age, embedding FROM users` resolves each field independently,
newest overlay first:

- `name`: the v3 overlay covers `{2,3}` → `["Alice", "Bob", "Caroline", "David"]`.
- `age`: the v3 overlay does not cover `age`; the v2 overlay still applies at
  offsets 1 and 3 → `[30, 26, 40, 23]`.
- `embedding`: the v3 overlay covers `{1}` → Bob's vector is the new one, others
  from base.

Overlays from different versions coexist and apply per field.

### Step 6 — compaction (overlay → base)

The scheduler folds both overlays into fragment `0` at version 4, computing
post-images for `age`, `name`, and `embedding`, and writing a new base data file
`data/file1.lance` with those columns. In the old file, fields 2, 3, and 4 are
marked with a tombstone (`-2`); field 1 (`id`) remains. The fragment's `overlays` list is
cleared. Row addresses are preserved (a column rewrite, not a row rewrite), so
stable row IDs and the deletion vector are untouched.

Because the fold removed the overlay that was excluding offsets 1 and 3 from the
`age` index, the commit must drop fragment `0` from its coverage so `age` queries
fall to the flat path.

## Guidance

!!! note "This section is a stub."

    The following are implementation considerations, not part of the on-disk
    specification.

### When to overlay vs. rewrite a column vs. move rows

<!-- TODO: Replace the rough heuristic below with concrete thresholds once we
     have benchmarked the crossover points between overlays, column rewrites,
     and row moves. -->

*(To be expanded.)* The choice between appending an overlay, rewriting a full
column (data evolution), and moving updated rows to a new fragment depends on the
fraction of rows changed, the fraction of columns changed, column width, the
presence of indexes on the changed columns, and the accumulated overlay read
cost. Roughly: few rows changed favors overlays; most rows in a few columns
favors a column rewrite; most columns changed favors moving rows to a new
fragment.

### Writer support

<!-- TODO: Fill in as writer implementation progresses, including the status of
     single-file sparse overlays (independent-length columns). -->

*(To be expanded.)* Dense (rectangular) overlays write with the existing
equal-length file writer today. Sparse overlays stored as a **single** file
require the writer to emit columns of independent lengths, which the current v2
writer does not yet do (it advances all columns from one global row counter).
Until that support lands, a writer can express a sparse update as multiple dense
overlays in one transaction.

### Scheduling compaction

<!-- TODO: Fill in with a concrete cost/benefit policy once compaction is
     implemented and benchmarked. -->

*(To be expanded.)* The overlay→overlay and overlay→base modes have very
different costs; a cost/benefit scheduler decides when each is worthwhile, using
the version gap as a staleness signal.

## Related specifications

- [Table format overview](index.md)
- [Transactions: DataOverlay operation](transaction.md#dataoverlay) — write path
  and conflict semantics
- [Row ID & Lineage](row_id_lineage.md)
- [Index Formats: handling overlay rows](../index/index.md#handling-deleted-and-invalidated-rows)
- [Format Versioning](versioning.md)
