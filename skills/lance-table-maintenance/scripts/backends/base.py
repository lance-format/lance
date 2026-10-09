"""Definitions shared by the entry point and the execution backends."""

from __future__ import annotations

from typing import Protocol

STATES = ("queued", "running", "succeeded", "failed", "canceled")
TERMINAL_STATES = ("succeeded", "failed", "canceled")

ERROR_CODES = (
    "unsupported",
    "invalid_input",
    "table_not_found",
    "job_not_found",
    "permission_denied",
    "conflict",
    "internal",
)

# Options a job may carry, per operation. Only options that change what the
# table looks like afterwards belong here. Names follow pylance.
OPTIONS = {
    "compact": {
        "target_rows_per_fragment": int,
        "max_bytes_per_file": int,
        "materialize_deletions": bool,
        "materialize_deletions_threshold": float,
    },
    "optimize_indices": {
        "index_names": list,
        "num_indices_to_merge": int,
        "retrain": bool,
    },
    "cleanup": {
        "older_than_seconds": float,
        "retain_versions": int,
        "error_if_tagged_old_versions": bool,
    },
}

# Table config keys that start with this prefix set compaction options.
CONFIG_PREFIX = "lance.compaction."

# Per-run budgets limit how much one run does. They do not say whether the
# table needs compaction, so planning for a health check switches off the ones
# the table config sets. Only those: older pylance rejects the keys it lacks.
BUDGET_KEYS = ("max_source_fragments", "max_source_rows", "max_source_bytes")


class BackendError(Exception):
    """An error with one of the agreed error codes."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Backend(Protocol):
    """What an execution backend provides.

    ``preview(job) -> dict`` is optional. A backend outside this directory
    does not need to import anything from here; matching method names is
    enough.
    """

    def submit(self, job: dict) -> dict: ...

    def status(self, job_id: str) -> dict: ...

    def cancel(self, job_id: str) -> dict: ...
