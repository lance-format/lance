## Health indicators

Use this file to explain the report printed by `scripts/doctor.py` and to decide which operation the
table needs.

Every kind of write leaves some debt in the table. The report shows how much has piled up, and each
kind of debt is paid back by one operation.

## What the debts are

| Debt | Caused by | What it costs | Fields |
| --- | --- | --- | --- |
| Too many small fragments | Frequent small appends | Opening the table and every commit read and write the whole manifest. Scans need more requests | `num_fragments`, `num_small_fragments`, `compaction_tasks` |
| Deleted rows still stored | `delete`, `update`, `merge_insert` | Reads filter them out. They take up storage | `num_deleted_rows`, `compaction_tasks` |
| Index does not cover some rows | Writes after the index was created. For some index types also compaction, see `references/operations.md` | The uncovered rows are scanned without the index | `indices[].num_unindexed_rows` |
| Index has several segments | Incremental index updates that add a segment instead of merging | Each query searches every segment | `indices[].num_segments` |
| Old versions | Every commit adds a version | Storage | `num_versions`, `tags` |
| No automatic cleanup | — | Versions grow without limit | `auto_cleanup` |

## Which operation is needed

| Operation | Needed when | Job options |
| --- | --- | --- |
| `compact` | `compaction_tasks` is above 0. This covers both small fragments and deleted rows: a fragment whose share of deleted rows is above the threshold (10% by default) is counted too. Whether it is worth its cost shows in the preview, see "Is a compaction worth it" in `references/operations.md` | None needed |
| `optimize_indices` | An index has `num_unindexed_rows` above 0 | None needed |
| `optimize_indices` to merge segments | An index has `num_segments` above 1 | `num_indices_to_merge` set to that number of segments. Without it, an index that already covers every row is left as it is |
| `cleanup` | The user wants to reclaim storage. Tell them `num_versions` and `tags`, and let them give the retention | `older_than_seconds` or `retain_versions`, from the user |

When more than one is needed, update indices before compacting; `references/operations.md` explains
why. Nothing is needed when none of these holds. `auto_cleanup` being empty is only reported: the
user can turn automatic cleanup on with `ds.optimize.enable_auto_cleanup(...)`.

How many bytes a cleanup would free depends on which versions are kept, so the report does not
guess. Preview a `cleanup` job to get the number; `references/operations.md` says what the preview
shows.

## Fields

| Field | Meaning |
| --- | --- |
| `table`, `pylance_version`, `version` | What was checked, with which pylance, at which table version |
| `num_rows` | Live rows |
| `num_fragments` | Fragments in this version |
| `num_deleted_rows` | Rows marked deleted but still stored. The share of deleted rows is `num_deleted_rows / (num_rows + num_deleted_rows)` |
| `small_fragment_threshold_rows` | A fragment with fewer rows counts as small. Taken from the table config `lance.compaction.target_rows_per_fragment`, else 1048576 |
| `num_small_fragments` | Fragments below that threshold |
| `compaction_tasks` | How many tasks a compaction would run. See below |
| `compaction_budget` | Per-run budgets found in the table config. Shown for information only |
| `num_versions` | Versions that still exist |
| `tags` | Tag name to version. Tagged versions block a cleanup unless the user allows it |
| `auto_cleanup` | The `lance.auto_cleanup.*` keys of the table config. Empty means automatic cleanup is off |
| `indices[]` | Per index: `name`, `index_type`, `num_segments`, `num_rows_indexed`, `num_unindexed_rows` |
| `errors` | Sections that could not be computed, with the reason. The other sections are still valid. An older pylance cannot compute some sections cheaply; the check then leaves them out and says so here instead of falling back to a slow method |

## Reading `compaction_tasks`

A task rewrites one group of neighboring small fragments, or one fragment with many deleted rows,
into fewer fragments. The number of tasks is therefore not the number of fragments a compaction
removes: 2000 small neighbors can make up a single task.

The number comes from Lance's own compaction planner, the same code that runs a compaction. It is
planned with the table's own targets, such as `target_rows_per_fragment`, and without any per-run
budget.

| Value | Meaning |
| --- | --- |
| Greater than 0 | A compaction would change the table now |
| 0 | A compaction would do nothing now |

`compaction_tasks` being 0 does not mean there are no small fragments. If `num_small_fragments` is
still high, the remaining small fragments cannot be merged now. For example, a small fragment
between two large ones has no neighbor to merge with. Fragments are also only merged with neighbors
covered by the same index segments, so a fragment no index covers, or one under another segment,
stays apart. In that case `indices[].num_unindexed_rows` is above 0 or `indices[].num_segments`
above 1, and the fragments can be merged after `optimize_indices`.

Do not use `num_small_fragments` or `num_deleted_rows` alone to decide whether to compact.

## When a section is slow or missing

The check reads only the manifest of the latest version, so its cost grows with the size of that
manifest, not with the amount of data. On tables written by old versions of Lance, fragments may
lack row counts. Then `num_rows` and `compaction_tasks` read every data file to count the rows, so
the check gets slow on such a table with many fragments; the `indices` section fails with a message
that asks for a write with a current version.

The time it took to open the table and the time each section took are logged on stderr; the report
on stdout is not affected. For a table in an object store most of the time goes into opening it,
which downloads that manifest.
