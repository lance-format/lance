## Operations

Use this file when writing a job file or explaining what a job did.

## The three operations

| `operation` | What it does | pylance call | Can it be undone | Versions |
| --- | --- | --- | --- | --- |
| `compact` | Merges small fragments and drops deleted rows | `ds.optimize.compact_files(...)` | Yes, until the old files are deleted, see below | Usually adds 2, the first of which changes no data. A compaction that is stopped half way may leave that one. Adds none if there is nothing to do |
| `optimize_indices` | Adds new rows to existing indices. With `num_indices_to_merge`, also merges index segments | `ds.optimize.optimize_indices(...)` | Yes, in the same way | Adds 1 or none |
| `cleanup` | Deletes old versions, and files that nothing refers to once they are 7 days old; younger ones may belong to a write in progress | `ds.cleanup_old_versions(...)` | **No** | Adds none |

To undo a compaction or an index update, restore the version before it:
`lance.dataset(uri, version=v).restore()`. This works only while the old files exist: a cleanup job
deletes them, and so does automatic cleanup if the table has it on (see `auto_cleanup` in the health
report).

Run only the operations the health report shows are needed; `references/health-indicators.md` maps
each indicator to its operation. When more than one is needed, run them in this order:
`optimize_indices`, `compact`, `cleanup`. Cleanup goes last because the files it deletes are what
makes the first two reversible. Propose a cleanup only if the user wants to reclaim storage.

Indices go first because compaction only merges neighboring fragments that are covered by the same
index segments. Fragments no index covers yet are merged apart from the indexed ones. For example,
with 32 indexed and 8 new small fragments, updating the index and then compacting gives 1 fragment
in two jobs. Compacting first gives 2 fragments, and reaching 1 then takes an index update and a
second compaction, which rewrites all rows again.

The same holds for segments: fragments covered by different segments of an index are not merged
either. If the new rows went into a segment of their own, merge the segments first by setting
`num_indices_to_merge` to the index's `num_segments`.

Some index types cannot follow rows into the fragments a compaction writes: `ZoneMap`,
`BloomFilter`, `RTree` and `Fm` (the `index_type` in the health report). After a compaction, such an
index no longer covers the rewritten rows, and its `num_unindexed_rows` rises by that many. Queries
still return the right rows, but scan the rewritten ones without that index until the next
`optimize_indices`. So on a table with such an index, every compaction is followed by an index
update. Limit that update to the indices that were left behind, with `index_names`: the other
indices need none, and an update they do not need may add a segment to them. Keep the order above
anyway: updating first still saves a second compaction. Tell the user before a compaction, because
on a large table that update indexes all the rewritten rows again.

## Job file

```json
{"table": "s3://bucket/path/t.lance", "operation": "cleanup", "options": {"retain_versions": 10}}
```

`table`, `operation` and `options` are the only fields. Anything else is rejected.

## Options

Write only the options the user asked for, the ones the user agreed to after a preview, and the ones
the chosen operation needs, such as `num_indices_to_merge` to merge segments (see
`references/health-indicators.md`). The user confirms the whole job file before it runs. An option
that is left out takes the value from the table config, or Lance's default if the table config does
not set it.

| `operation` | Option | Type | Meaning |
| --- | --- | --- | --- |
| `compact` | `target_rows_per_fragment` | integer | Rows a fragment should have. Default 1048576 |
| `compact` | `max_bytes_per_file` | integer | Upper size of a data file |
| `compact` | `materialize_deletions` | boolean | Also rewrite fragments that are not small but have many deleted rows, to drop those rows. Default true. Small fragments that are merged lose their deleted rows either way |
| `compact` | `materialize_deletions_threshold` | number | Share of deleted rows above which a fragment is rewritten. Default 0.1 |
| `optimize_indices` | `index_names` | list of strings | Indices to update. Default all |
| `optimize_indices` | `num_indices_to_merge` | integer | How many segments to merge. Needed to merge the segments of an index that already covers every row; set it to the index's `num_segments`. An index with fewer segments keeps the segments it has |
| `optimize_indices` | `retrain` | boolean | Vector indices only: rebuild the index from all rows. Expensive. Has no effect on scalar indices |
| `cleanup` | `older_than_seconds` | number | Only delete versions older than this |
| `cleanup` | `retain_versions` | integer, at least 1 | Keep this many of the latest versions |
| `cleanup` | `error_if_tagged_old_versions` | boolean | Fail if a tagged version would be deleted. Default true. When false, tagged versions are kept and the other old versions are deleted |

A `cleanup` job needs `older_than_seconds` or `retain_versions`. When both are given, a version is
deleted only if it matches both.

Settings that only change how fast a job runs or how much it uses, such as threads or batch sizes,
are not job options. They belong to the execution backend.

## What a preview shows

| `operation` | Preview on this machine |
| --- | --- |
| `compact` | `tasks`: tasks the run would execute. `tasks_without_budget`: tasks if no per-run budget applied. `budgets`: budgets in effect and where they come from. `fragments_rewritten`, `rows_rewritten`, `bytes_rewritten`: what those tasks read and write again; `bytes_rewritten` is null if the table does not record file sizes. `deleted_rows_dropped`: deleted rows they drop. This preview reads the metadata of every fragment in the plan, so on a table with many fragments it takes longer than the health check |
| `optimize_indices` | Per index: segments and rows not covered |
| `cleanup` | Versions, files and bytes that would be deleted. Fails like the job would, for example with `invalid_input` when a tagged version is in the way. With a pylance that has no `explain_cleanup_old_versions`, the preview says which versions the retention options select instead: `old_versions`, the oldest and the newest of them, `versions_kept` and `tagged_versions_kept`; `bytes_removed` is null then. This preview reads the manifest of every version, so it takes longer the more versions the table has |

## Is a compaction worth it

A compaction rewrites every row of the fragments it takes, so it costs `bytes_rewritten` of reads
and writes. It gains fewer fragments, and less storage when `deleted_rows_dropped` is large. Each
task leaves at least one fragment, so it removes at most `fragments_rewritten - tasks` fragments. It
removes few when the fragments it takes already hold close to `target_rows_per_fragment` rows each,
that is when `rows_rewritten / fragments_rewritten` is near that target.

When proposing a compaction, show the user `bytes_rewritten` and how many fragments it removes at
most. If the fragments are near the target, say that the compaction gains little and let the user
decide. If the job would run on this machine, also say that this machine reads and writes all those
bytes.

## Per-run budgets for compaction

A budget limits how much one compaction run takes on, so that a large table can be compacted in
several runs. Each run commits on its own, and an interrupted series can continue.

| Where the budget is set | Keys |
| --- | --- |
| Table config | `lance.compaction.max_source_fragments`, `lance.compaction.max_source_rows`, `lance.compaction.max_source_bytes` |
| Environment variable, for jobs on this machine | `LANCE_MAINTENANCE_LOCAL_MAX_SOURCE_FRAGMENTS` |

If both set a fragment budget, the smaller one applies.

With a budget, `succeeded` means this run is done, not that the table is fully compacted. Check the
health again and look at `compaction_tasks`.

A budget smaller than the first compaction task would compact nothing. On this machine such a job
ends as `failed` with code `invalid_input`, and the message names the budget and where it comes
from. Raise the budget and submit again. Another backend may instead report `succeeded` without
changing the table.

## Jobs on this machine

The default execution backend runs the job in a background process. It works on Linux and macOS.

Each job has a directory under `$LANCE_MAINTENANCE_HOME`, by default `~/.lance-maintenance`:

| File | Content |
| --- | --- |
| `job.json` | The job as submitted, and the settings of this machine it runs with: the fragment budget, the path of the storage options file and the directory `submit` ran in. Never the content of that file |
| `state.json` | State, result or error, and the machine and process that run the job |
| `log.txt` | Log of the background process: when the job started, when and how it ended, and whatever Lance printed in between |

For a table in an object store, the background process reads the storage options file when it
starts; see `references/object-store.md`.

A job uses as much of the machine as Lance takes by default. The background process inherits the
limits of the `submit` command, so to leave room for other work, run `submit` under a limit of the
operating system, for example `nice -n 19 taskset -c 0-7 python scripts/maintain.py submit job.json`
on Linux.

The directory may be on a file system that several machines share. Only the machine that runs a job
can tell whether its process is still there, so the status carries `host`, the name of that machine.
From any other machine, `status` shows the state as it was last recorded, and `cancel` fails with
`unsupported` and a message that names the machine: run it there.

A job that ends as `failed` with code `internal` and a message about the background process ended
without reporting a result: it was stopped from outside, for example by the system or by a sandbox
that ends child processes, or it crashed. Read `log.txt` and check the health again before deciding
what to do.
