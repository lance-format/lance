---
name: lance-table-maintenance
description: Guide Code Agents to check the health of a Lance table and to maintain it. Use when a user asks whether a Lance table is healthy, why a table got slow or large, how many small fragments or old versions it has, whether an index covers all rows, or asks to compact a table (compact_files), update indices (optimize_indices), or clean up old versions (cleanup_old_versions), on this machine or on another execution backend such as their own cluster.
---

# Lance Table Maintenance

## Scope

Use this skill to:

- Check the health of a Lance table without changing it
- Compact a table, update its indices, or clean up its old versions
- Run such a job on this machine (the default) or hand it to another execution backend

Do not use this skill for:

- Writing, reading, or creating indices: use the `lance-user-guide` skill
- Contributing to Lance itself

## Before you start

All script paths below are relative to this skill's directory.

The scripts need pylance. Confirm it is installed:

```bash
python -c "import lance; print(lance.__version__)"
```

Run every script with this same Python. If the user runs jobs on another execution backend, that
backend's package must be importable from it too. When `python` is not that interpreter, use its
full path in all the commands below.

### Tables in an object store

The health check, the previews on this machine and the jobs on this machine open the table from this
machine. For a table in an object store, this machine therefore needs the settings of that store:
credentials, and for an S3-compatible store also the endpoint and the region. Ask the user which of
these applies:

| The user says | What to do |
| --- | --- |
| This machine already has access, for example through environment variables or a role of the machine | Nothing. Lance finds the settings on its own |
| Anything else | The user writes the settings into a storage options file and tells you its path. Start every command with `LANCE_MAINTENANCE_STORAGE_OPTIONS_FILE=<path>` |

```bash
LANCE_MAINTENANCE_STORAGE_OPTIONS_FILE=<path> python scripts/doctor.py <table-uri>
```

The file holds credentials, so you only handle its path. `references/object-store.md` describes the
file, what to tell a user who has to write one, and what to check when the table cannot be reached.

## Workflow

Follow these steps in order.

1. **Ask** for the table URI, for how this machine reaches the table if it is in an object store
   (see "Tables in an object store" above), and whether other writers are active on the table. If
   they are, tell the user a job may end with `conflict` and need to be submitted again.
2. **Check the health** of the table and explain the report using `references/health-indicators.md`:

   ```bash
   python scripts/doctor.py <table-uri>
   ```

3. **Pick one operation** that the report shows is needed, using `references/health-indicators.md`.
   Skip the ones that are not needed; if none is, stop here. If more than one is needed, take them
   in this order: optimize indices, then compact, then clean up. Old versions are not a reason on
   their own: tell the user how many there are, and propose a cleanup only if they want to reclaim
   storage. Before a compaction, check whether the table has an index that a compaction leaves
   behind, such as `ZoneMap` (see `references/operations.md`), and tell the user it will need an
   index update afterwards.
4. **Write the job file**, for example `job.json`. See `references/operations.md` for the options of
   each operation and which ones to write.

   ```json
   {"table": "s3://bucket/path/t.lance", "operation": "compact", "options": {}}
   ```

5. **Choose the execution backend** using the table below.
6. **Preview** the job and show the result:

   ```bash
   python scripts/maintain.py preview job.json
   ```

   This preview reads the table from this machine and shows what the job would change, for example
   how many compaction tasks there are, or how many versions and bytes a cleanup would delete. If
   another backend was chosen, also run it with `--backend <backend>`: that shows how the backend
   would run the job. Skip a preview that fails with `unsupported`; if both do, say that no preview
   is available and go on. A backend that cannot run the operation at all also answers the preview
   with `unsupported`, and its message says so: then tell the user and ask where to run the job
   instead. For a compaction, also tell the user what it costs and gains, as described under "Is a
   compaction worth it" in `references/operations.md`.

   A backend's preview may contain `options`: the options it recommends. Show them. If the user
   agrees, write them into the job file and preview again.
7. **Confirm.** Show the user the job file, the value of `--backend` and the previews exactly as
   they are, and wait for a yes.
8. **Submit**, then tell the user the backend and the job id from the output:

   ```bash
   python scripts/maintain.py submit job.json [--backend <backend>]
   ```

9. **Poll** until `state` is `succeeded`, `failed` or `canceled`. Wait 5, 10, 30, then 60 seconds
   between calls. After about 10 minutes stop polling and tell the user they can ask for the status
   later with the backend and the job id.

   ```bash
   python scripts/maintain.py status <job-id> [--backend <backend>]
   ```

10. **Check the health again** and compare it with the first report. If it still shows work to do,
    for example `compaction_tasks` above 0 after an index update, go back to step 3. If the job
    `succeeded` but did not change the fields it was meant to change (see
    `references/health-indicators.md`), do not submit it again: show both reports to the user and
    stop.

To stop a job:

```bash
python scripts/maintain.py cancel <job-id> [--backend <backend>]
```

### Choosing the execution backend

| The user | What to do |
| --- | --- |
| Does not mention another place to run the job | Use the default. Do not pass `--backend` |
| Gives a full class path such as `my_pkg.ray_backend.Backend` | Pass it as is: `--backend my_pkg.ray_backend.Backend` |
| Gives only a name, not a class path | Run `python scripts/maintain.py backends`. If the name is listed, pass it. If not, ask the user for the full class path |

## Safety rules

- The health check is read-only. Anything that changes the table needs the user's confirmation after
  they have seen the job file and the previews.
- Never choose or switch the execution backend on your own. If the chosen backend fails, report it.
  Do not fall back to this machine.
- Run one job per table at a time.
- Cleanup cannot be undone. It needs `older_than_seconds` or `retain_versions`. If the user gave
  neither, ask.
- Do not set `error_if_tagged_old_versions` to `false` without asking.
- Never open, print or write a storage options file. Never put a credential in a job file, on a
  command line, or in the conversation.
- If the outcome of `submit` is unknown, for example after a timeout, do not submit again. The job
  may have been accepted.
- After a job ends as `failed` or `canceled`, check the health again first. The table may have
  changed.

## Reading the output

Every call to `maintain.py` prints one JSON object.

| Output | Meaning |
| --- | --- |
| `{"job_id": ..., "state": ..., "backend": ...}` | The job's status. The exit code is 0 even when `state` is `failed`. A backend may add fields of its own, such as its native state name |
| `{"error": {"code": ..., "message": ...}}` | The call itself failed. The exit code is not 0 |

The same codes appear in `error` of a job that ended as `failed`. What to do for each:

| Code | What to do |
| --- | --- |
| `unsupported` | For a preview, skip it (see step 6). For an operation, tell the user this backend cannot do it and ask where to run it instead |
| `invalid_input` | Fix the job file, the job id or the backend name, then confirm with the user again. For a cleanup blocked by tagged versions, ask the user whether to delete the tags or to set `error_if_tagged_old_versions` to `false` |
| `table_not_found` | Ask the user to check the table URI. For a table in an object store, also see `references/object-store.md` |
| `job_not_found` | Ask the user to check the job id and the backend. Do not submit again |
| `permission_denied` | Ask the user to check the credentials; for a job on this machine see `references/object-store.md`. Do not retry |
| `conflict` | Another writer changed the table. Check the health again, then submit once more |
| `internal`, or any other code | Show the message to the user. Do not retry |

## Bundled resources

- What each field of the health report means: `references/health-indicators.md`
- Operations, their options and what can be undone: `references/operations.md`
- Reaching a table in an object store from this machine: `references/object-store.md`
- How to write an execution backend: `references/backend-interface.md`
- Health check: `scripts/doctor.py`
- Entry point for jobs: `scripts/maintain.py`
- Check an execution backend against the interface: `scripts/conformance.py`
