## Execution backend interface

Use this file to write an execution backend, or to understand what `scripts/maintain.py` does with
one.

An execution backend decides where a maintenance job runs. This skill ships one, `local`, which runs
the job on the user's machine. Anyone can add another, for example one that submits the job to their
own Ray or Spark cluster.

A backend is a class with three methods. `scripts/maintain.py` does everything else: it validates
the job, loads the backend, calls it, checks what comes back and prints the result.

## Entry point

```bash
python scripts/maintain.py backends
python scripts/maintain.py preview <job.json> [--backend B]
python scripts/maintain.py submit  <job.json> [--backend B]
python scripts/maintain.py status  <job_id>   [--backend B]
python scripts/maintain.py cancel  <job_id>   [--backend B]
```

`backends` lists the backends in `scripts/backends/`, for example `{"backends": ["local"]}`. Every
`.py` file there except `__init__.py` and `base.py` counts as a backend, so a backend that needs
helper files keeps them in a subdirectory.

`--backend` defaults to `local`. The entry point only looks at whether the value contains a dot:

| Value of `--backend` | Where the class is found | Example |
| --- | --- | --- |
| Without a dot | The class `Backend` in `scripts/backends/<value>.py` | `local` |
| With a dot | Module and class name, split at the last dot and imported from the current Python environment | `my_pkg.ray_backend.Backend` |

A backend given by class path must be importable from the Python that runs `maintain.py`: installed
there, or on `PYTHONPATH`.

The entry point then checks that the class has `submit`, `status` and `cancel`. A backend that
cannot be found or lacks a method is an error. The entry point never falls back to `local`.

## The interface

```python
class Backend(Protocol):
    def submit(self, job: dict) -> dict: ...      # returns a job status
    def status(self, job_id: str) -> dict: ...    # returns a job status
    def cancel(self, job_id: str) -> dict: ...    # returns a job status
    # optional: def preview(self, job: dict) -> dict
```

- The constructor takes no arguments. A backend reads its own settings from environment variables or
  its own configuration file.
- To report an error, raise an exception. If the exception has a `code` attribute that is a string,
  it becomes the error code. Otherwise the code is `internal`.
- A backend outside this skill does not need to import anything from it. Matching method names is
  enough.
- A job does not say how to reach the table's object store, and never carries credentials. A backend
  uses the access of the environment it runs the job in, or settings of its own.

## Job

The entry point validates the job and passes it on, with a local table path made absolute.

```json
{"table": "s3://bucket/path/t.lance", "operation": "compact", "options": {"target_rows_per_fragment": 1048576}}
```

| `operation` | Options, named as in pylance except `older_than_seconds`, which is pylance's `older_than` in seconds |
| --- | --- |
| `compact` | `target_rows_per_fragment`, `max_bytes_per_file`, `materialize_deletions`, `materialize_deletions_threshold` |
| `optimize_indices` | `index_names`, `num_indices_to_merge`, `retrain` |
| `cleanup` | `older_than_seconds`, `retain_versions`, `error_if_tagged_old_versions` |

A job carries only options that decide the result, not options that decide how it is computed. How
many machines, how much memory, how many threads and when to run are up to the backend and never
appear in a job.

The meaning and default of each option are in `references/operations.md`. When translating them for
another engine, keep these rules:

- An option that is left out takes the value from the table config, or Lance's default if the table
  config does not set it.
- When a `cleanup` job has both `older_than_seconds` and `retain_versions`, a version is deleted
  only if it matches both.
- `retain_versions` counts the versions that exist, tagged ones included: Lance keeps the newest n
  of them and deletes older ones. Versions can be missing in between, so this is not the same as
  keeping the version numbers above `latest - n`.
- `error_if_tagged_old_versions` defaults to true: a cleanup that would delete a tagged version
  fails.

A job describes what to do, not how. It does not contain a compaction plan: a plan grows with the
number of fragments, so it gets large for a large table, and its format is not stable across Lance
versions. Each backend plans on its own side.

## Job status

This is what the three methods return.

```json
{"job_id": "20260928T101500Z-3fa9c1d2", "state": "running"}
{"job_id": "...", "state": "succeeded", "result": {"fragments_removed": 900, "fragments_added": 2}}
{"job_id": "...", "state": "failed", "error": {"code": "conflict", "message": "..."}}
```

| Field | Meaning |
| --- | --- |
| `job_id` | Made up by the backend in `submit`. Letters, digits, `.`, `_` and `-`, at most 128 characters, not `.` or `..`. A backend that submits to a cluster can use the id the cluster returns |
| `state` | One of `queued`, `running`, `succeeded`, `failed`, `canceled` |
| `result` | Free form, for information. Whether the job had an effect is decided by checking the table again |
| `error` | `code` and `message`, when the state is `failed` |

A status may carry fields of its own, such as the native state name of the cluster; the entry point
passes them through. The entry point adds `"backend": "<value of --backend>"` to what it prints.

## Preview

`preview` returns a JSON object. One key has an agreed meaning: `options`, the options the backend
recommends. Everything else is free form, for example an estimated duration.

A backend recommends options here and nowhere else. They must use the option names and types a job
accepts, because the caller writes them into the job as they are once the user agrees, and the entry
point rejects anything else.

## Errors

When a call produces no job status, the entry point prints an error and exits with a code other
than 0.

```json
{"error": {"code": "invalid_input", "message": "unknown option: retain_version"}}
```

| Code | Meaning |
| --- | --- |
| `unsupported` | The backend does not support this call or operation |
| `invalid_input` | Something is wrong with the job, the job id or the backend name |
| `table_not_found` | The table does not exist or cannot be reached from where the job runs |
| `job_not_found` | No job with this id |
| `permission_denied` | Not allowed |
| `conflict` | Another writer changed the table |
| `internal` | Anything else |

A backend translates the errors of its environment into one of these and puts the original text into
`message`, with the cause first: a message that starts with a long stack trace hides it. It may use
a code of its own. The entry point prints it as is, and callers treat a code they do not know like
`internal`.

## What the entry point guarantees

A backend can rely on these and does not have to repeat them.

| Rule | What it prevents |
| --- | --- |
| A job with an unknown field, an unknown option or a value of the wrong type is rejected with `invalid_input` | A misspelled `retain_version` is ignored and a cleanup that cannot be undone runs with the wrong retention |
| A `cleanup` job has `older_than_seconds` or `retain_versions`, and `retain_versions` is at least 1 | Without either, Lance's bindings behave differently: Python keeps 14 days, Rust and Java delete every version except the latest and tagged ones. The same job would delete different data on different backends |
| A local `table` path is made absolute. A `table` with a user name or password in it is rejected | Secrets written to disk |
| A job id passed to `status` or `cancel` follows the rules for `job_id` | Quoting problems and path traversal |
| What the backend returns has a valid `state` and `job_id`. Otherwise the error code is `internal` | A backend that returns something else |
| The exit code is 0 whenever there is a job status, also for `failed` | Mixing up "the job failed" and "the call failed" |
| Stdout is exactly one JSON object. What the backend prints goes to stderr | Output that cannot be parsed |
| A backend without `preview` gives `unsupported` | The need for a separate way to ask what a backend can do |

## What a backend must do

| Rule | What it prevents |
| --- | --- |
| Return from every method within about 60 seconds. Never wait for the job to finish | A caller's timeout stops `submit` halfway |
| Use the table config or Lance's default for an option that is left out. Never fill in a value in `submit` | Running with values the user did not confirm |
| Always work on the latest version of the table | Doing all the work and then failing to commit |
| `status` and `cancel` work from any process, given the `job_id` and the same `--backend` as `submit` | A job that cannot be found later |
| `cancel` is a request. It returns the current status. For a job that has ended it returns that status | Canceling on a cluster takes time |
| `status` and `cancel` only answer for jobs this backend started. Any other id is `job_not_found` | An id that belongs to somebody else's job on the same cluster, and `cancel` stops that job |
| The state describes the job, not the table. After `failed` or `canceled` the table may have changed | A process stopped after the commit and before the state was written |
| Raise an error with code `unsupported` for an operation it does not support | — |
| Keep secrets out of everything the methods return or raise | `message` and `result` are shown to the user as they are |
| If a retention option has to be turned into a version number or a point in time before the job runs, round toward keeping more | Deleting a version that was committed while the job waited |

## Example

This backend submits the job to a Ray cluster of your own. It only shows how the interface is used
and is not part of this skill.

```python
import json
import os

from ray.job_submission import JobStatus, JobSubmissionClient

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = {
    JobStatus.PENDING: "queued",
    JobStatus.RUNNING: "running",
    JobStatus.SUCCEEDED: "succeeded",
    JobStatus.FAILED: "failed",
    JobStatus.STOPPED: "canceled",
}


class JobError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Backend:
    def __init__(self):
        self.client = JobSubmissionClient(os.environ["RAY_ADDRESS"])

    def submit(self, job):
        job_id = self.client.submit_job(
            entrypoint="python run_job.py",  # runs on the cluster, written by you
            runtime_env={"working_dir": HERE, "env_vars": {"LANCE_JOB": json.dumps(job)}},
        )
        return {"job_id": job_id, "state": "queued"}

    def status(self, job_id):
        try:
            state = self.client.get_job_status(job_id)
        except RuntimeError as e:  # Ray reports "Job <id> does not exist."
            if "does not exist" in str(e):
                raise JobError("job_not_found", str(e)) from e
            raise
        return {"job_id": job_id, "state": STATE[state]}

    def cancel(self, job_id):
        self.client.stop_job(job_id)
        return self.status(job_id)
```

`run_job.py` reads the job on the cluster and runs it with a distributed engine, for example
`compact_files` from lance-ray. How many workers it uses is decided there or in the cluster
configuration, not in the job.

## Checking a backend

`scripts/conformance.py` drives a backend through the entry point and prints one `PASS`, `FAIL` or
`SKIP` line per check. It exits with 1 if any check fails.

```bash
python scripts/conformance.py
python scripts/conformance.py --backend my_pkg.ray_backend.Backend --table <uri>
```

Without `--table` it creates a temporary table on this machine and also checks that each job changed
the table. With `--table` it uses a table the backend can reach and only checks the job states.
**That table is modified**: it is compacted, its indices are updated and all but its latest version
are deleted. Use a table made for this check, without tags: a tag blocks the cleanup.

| Check | Expected |
| --- | --- |
| `status` for a job id that does not exist | `job_not_found` |
| `preview` | A JSON object, or `unsupported`, which counts as skipped |
| A `compact`, an `optimize_indices` and a `cleanup` job | Each is found by its id and ends as `succeeded`. `unsupported` counts as skipped |
| `status` and `cancel` for a job that has ended | The state does not change |
| `cancel` right after `submit` | The job reaches a final state |
| Every call | Returns within 60 seconds and prints one JSON object |
