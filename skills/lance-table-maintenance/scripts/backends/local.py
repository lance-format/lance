"""Default execution backend: runs the job on this machine.

``submit`` starts a background process and returns at once. The background
process records its progress in a state file, which ``status`` reads and
``cancel`` finishes off. Linux and macOS only.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

from .base import (
    BUDGET_KEYS,
    CONFIG_PREFIX,
    TERMINAL_STATES,
    BackendError,
)

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SOURCE_FRAGMENTS_ENV = "LANCE_MAINTENANCE_LOCAL_MAX_SOURCE_FRAGMENTS"
STORAGE_OPTIONS_ENV = "LANCE_MAINTENANCE_STORAGE_OPTIONS_FILE"
# The name of an option or environment variable that holds a credential
# contains one of these.
CREDENTIAL_WORDS = ("key", "secret", "token", "password")
START_TIMEOUT = 10
STOP_TIMEOUT = 10

log = logging.getLogger("lance_maintenance")


def log_to_stderr() -> None:
    """Send this skill's log to stderr, each line with its time. Stdout stays
    free for the result. Other libraries' loggers are left as they are."""
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S%z"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)


def home() -> str:
    path = os.environ.get("LANCE_MAINTENANCE_HOME", "~/.lance-maintenance")
    return os.path.abspath(os.path.expanduser(path))


def read_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_json(path: str, value: dict) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, default=str)
    os.replace(tmp, path)


def write_state(job_dir: str, state: dict) -> dict:
    """Write the state unless a final state is already recorded."""
    path = os.path.join(job_dir, "state.json")
    current = read_json(path)
    if current and current["state"] in TERMINAL_STATES:
        return current
    write_json(path, state)
    return state


def started_at(pid: int) -> str | None:
    """Start time of a live process, or None if it cannot be read.

    Only ever compared for equality: a process id alone does not identify a
    process because the system reuses ids.
    """
    if os.path.isdir("/proc/self"):
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
                # The second field is the process name and may contain spaces.
                fields = f.read().rsplit(")", 1)[1].split()
        except OSError:
            return None
        return None if fields[0] == "Z" else fields[19]
    command = ["ps", "-p", str(pid), "-o", "lstart="]
    # ps prints the time in the locale and time zone of whoever asks.
    env = {**os.environ, "LC_ALL": "C", "TZ": "UTC"}
    try:
        out = subprocess.run(command, capture_output=True, text=True, env=env)
    except FileNotFoundError:
        return None
    return out.stdout.strip() or None


def is_alive(pid: int, started: str | None) -> bool:
    return started is not None and started_at(pid) == started


def wait_until_gone(state: dict, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while is_alive(state["pid"], state["started_at"]):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def public(state: dict) -> dict:
    shown = ("job_id", "state", "host", "result", "error")
    return {k: state[k] for k in shown if k in state}


def storage_options_file() -> str | None:
    """This machine's own setting: the file that says how to reach the object
    store of the table."""
    path = os.environ.get(STORAGE_OPTIONS_ENV)
    return os.path.abspath(os.path.expanduser(path)) if path else None


def storage_options(path: str | None) -> dict | None:
    """Lance storage options read from a JSON file. Without a file Lance finds
    the settings on its own, for example in environment variables."""
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            options = json.load(f)
    # Neither error quotes the content of the file, which holds credentials.
    except (OSError, ValueError) as e:
        message = f"cannot read the storage options in {path}: {e}"
        raise BackendError("invalid_input", message) from e
    if not isinstance(options, dict) or not all(
        isinstance(v, str) for v in options.values()
    ):
        message = f"{path} must hold a JSON object whose values are strings"
        raise BackendError("invalid_input", message)
    return options


def redact(text: str, options: dict | None) -> str:
    """Best effort at keeping credentials out of what is shown to the user: an
    object store repeats the access key id in its error responses."""
    for name, value in {**os.environ, **(options or {})}.items():
        # Replacing a short value would garble the text.
        if len(value) >= 8 and any(w in name.lower() for w in CREDENTIAL_WORDS):
            text = text.replace(value, "***")
    return text


def classify(exc: BaseException) -> str:
    """Best effort: pylance raises plain errors, so only the message tells."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    message = str(exc).lower()
    if message.startswith("dataset at path") and "was not found" in message:
        return "table_not_found"
    # An object store that is reached without its endpoint or region.
    if re.search(r"bucket '[^']*' not found", message):
        return "table_not_found"
    if "commit conflict" in message or "too many concurrent writers" in message:
        return "conflict"
    denied = (
        "permission denied",
        "access denied",
        "403 forbidden",
        "401 unauthorized",
        "failed to get aws credentials",
    )
    if any(text in message for text in denied):
        return "permission_denied"
    if message.startswith("cleanup error"):
        return "invalid_input"
    return "internal"


def fragment_limit() -> int | None:
    """This backend's own setting: most fragments one compaction run may take."""
    limit = os.environ.get(MAX_SOURCE_FRAGMENTS_ENV)
    if limit is None:
        return None
    if not limit.isdigit() or int(limit) < 1:
        raise BackendError(
            "invalid_input", f"{MAX_SOURCE_FRAGMENTS_ENV} must be a positive integer"
        )
    return int(limit)


def budgets(ds, max_source_fragments: int | None) -> dict:
    """Per-run budgets in effect, as {key: (value, where it comes from)}."""
    config = ds.config() if hasattr(ds, "config") else {}
    found = {}
    for key in BUDGET_KEYS:
        if CONFIG_PREFIX + key in config:
            found[key] = (int(config[CONFIG_PREFIX + key]), "table config")
    key = "max_source_fragments"
    if max_source_fragments is not None:
        if key not in found or max_source_fragments < found[key][0]:
            found[key] = (max_source_fragments, MAX_SOURCE_FRAGMENTS_ENV)
    return found


def plan(ds, options: dict, max_source_fragments: int | None) -> dict:
    from lance.optimize import Compaction

    found = budgets(ds, max_source_fragments)
    effective = {**options, **{k: v for k, (v, _) in found.items()}}
    compaction = Compaction.plan(ds, effective)
    tasks = compaction.num_tasks()
    unbudgeted = tasks
    if found:
        # Switches off only the budgets in effect; see BUDGET_KEYS.
        unbudgeted = Compaction.plan(
            ds, {**options, **dict.fromkeys(found)}
        ).num_tasks()
    return {
        "options": effective,
        "plan": compaction,
        "tasks": tasks,
        "tasks_without_budget": unbudgeted,
        "budgets": {k: {"value": v, "source": s} for k, (v, s) in found.items()},
    }


def rewrite(compaction) -> dict:
    """What the planned tasks rewrite. Reads the metadata of each fragment in
    the plan, which takes longer than planning on a wide table."""
    fragments = rows = deleted = 0
    size = 0
    for task in compaction.tasks:
        for fragment in task.fragments:
            dropped = getattr(fragment.deletion_file, "num_deleted_rows", None) or 0
            fragments += 1
            rows += fragment.physical_rows - dropped
            deleted += dropped
            for data_file in fragment.files:
                # None when the table does not record file sizes.
                file_size = getattr(data_file, "file_size_bytes", None)
                size = None if size is None or file_size is None else size + file_size
    return {
        "fragments_rewritten": fragments,
        "rows_rewritten": rows,
        "deleted_rows_dropped": deleted,
        "bytes_rewritten": size,
    }


def check_index_names(ds, options: dict) -> None:
    # optimize_indices silently does nothing for a name it does not know.
    if hasattr(ds, "describe_indices"):
        known = {index.name for index in ds.describe_indices()}
    else:
        known = {index["name"] for index in ds.list_indices()}
    unknown = [n for n in options.get("index_names") or [] if n not in known]
    if unknown:
        raise BackendError("invalid_input", f"unknown index: {', '.join(unknown)}")


def cleanup_arguments(options: dict) -> dict:
    arguments = {k: v for k, v in options.items() if k != "older_than_seconds"}
    if "older_than_seconds" in options:
        try:
            older_than = timedelta(seconds=options["older_than_seconds"])
        except OverflowError as e:
            raise BackendError(
                "invalid_input", "older_than_seconds is too large"
            ) from e
        arguments["older_than"] = older_than
    return arguments


def selected_versions(ds, options: dict) -> dict:
    """What a cleanup would delete, worked out from the retention options
    alone, for a pylance that cannot explain a cleanup. It says which
    versions go, not how many files and bytes that frees."""
    versions = sorted(ds.versions(), key=lambda v: v["version"])
    old = versions[:-1]  # The latest version always stays.
    if "retain_versions" in options:
        keep = options["retain_versions"]
        first_kept = versions[max(len(versions) - keep, 0)]["version"]
        old = [v for v in old if v["version"] < first_kept]
    if "older_than_seconds" in options:
        # pylance gives the time of a version in local time, without a zone.
        cutoff = datetime.now() - cleanup_arguments(options)["older_than"]
        old = [v for v in old if v["timestamp"] < cutoff]
    tagged = {tag["version"]: name for name, tag in ds.tags.list().items()}
    kept = {tagged[v["version"]]: v["version"] for v in old if v["version"] in tagged}
    if kept and options.get("error_if_tagged_old_versions", True):
        message = (
            f"{len(kept)} tagged version(s) would be deleted: {json.dumps(kept)}. "
            "Delete the tags, or set error_if_tagged_old_versions to false to keep "
            "those versions"
        )
        raise BackendError("invalid_input", message)
    deleted = [v["version"] for v in old if v["version"] not in tagged]
    return {
        "old_versions": len(deleted),
        "oldest_version_deleted": deleted[0] if deleted else None,
        "newest_version_deleted": deleted[-1] if deleted else None,
        "versions_kept": len(versions) - len(deleted),
        "tagged_versions_kept": kept,
        "bytes_removed": None,
    }


def fields(value) -> dict:
    names = [n for n in dir(value) if not n.startswith("_")]
    return {n: getattr(value, n) for n in names if not callable(getattr(value, n))}


def run_operation(
    job: dict, storage: dict | None, max_source_fragments: int | None
) -> dict:
    import lance

    ds = lance.dataset(job["table"], storage_options=storage)
    options = job["options"]
    if job["operation"] == "compact":
        planned = plan(ds, options, max_source_fragments)
        if planned["tasks"] == 0 and planned["tasks_without_budget"] > 0:
            raise BackendError(
                "invalid_input",
                "the per-run budget is smaller than the first compaction task, "
                f"so nothing would be compacted: {planned['budgets']}",
            )
        return fields(ds.optimize.compact_files(**planned["options"]))
    if job["operation"] == "optimize_indices":
        check_index_names(ds, options)
        ds.optimize.optimize_indices(**options)
        return {}
    return fields(ds.cleanup_old_versions(**cleanup_arguments(options)))


def preview_operation(
    job: dict, storage: dict | None, max_source_fragments: int | None
) -> dict:
    import lance

    ds = lance.dataset(job["table"], storage_options=storage)
    options = job["options"]
    if job["operation"] == "compact":
        planned = plan(ds, options, max_source_fragments)
        shown = ("tasks", "tasks_without_budget", "budgets")
        return {**{k: planned[k] for k in shown}, **rewrite(planned["plan"])}
    if job["operation"] == "optimize_indices":
        if not hasattr(ds, "describe_indices"):
            raise BackendError("unsupported", "this pylance cannot describe indices")
        check_index_names(ds, options)
        rows = ds.count_rows()
        names = options.get("index_names")
        return {
            "indices": [
                {
                    "name": index.name,
                    "num_segments": len(index.segments),
                    "num_unindexed_rows": rows - index.num_rows_indexed,
                }
                for index in ds.describe_indices()
                # Lance's own indices do not cover rows.
                if not index.name.startswith("__lance_")
                and (names is None or index.name in names)
            ]
        }
    if not hasattr(ds, "explain_cleanup_old_versions"):
        return selected_versions(ds, options)
    explained = ds.explain_cleanup_old_versions(**cleanup_arguments(options))
    return fields(explained.stats)


class Backend:
    def submit(self, job: dict) -> dict:
        limit = fragment_limit()
        storage_file = storage_options_file()
        # Reports a file that cannot be used now instead of in a failed job.
        storage_options(storage_file)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = f"{stamp}-{secrets.token_hex(4)}"
        job_dir = os.path.join(home(), job_id)
        os.makedirs(job_dir, mode=0o700)
        # The path only: the credentials stay in the user's file.
        record = {
            "job": job,
            "max_source_fragments": limit,
            "storage_options_file": storage_file,
            "cwd": os.getcwd(),
        }
        write_json(os.path.join(job_dir, "job.json"), record)

        log_path = os.path.join(job_dir, "log.txt")
        with open(log_path, "ab") as log:
            worker = subprocess.Popen(
                [sys.executable, "-m", "backends.local", job_dir],
                # Makes `backends` resolve to this skill's own package.
                cwd=SCRIPTS_DIR,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
            )
        state_path = os.path.join(job_dir, "state.json")
        deadline = time.monotonic() + START_TIMEOUT
        while time.monotonic() < deadline and worker.poll() is None:
            if os.path.exists(state_path):
                break
            time.sleep(0.05)
        state = read_json(state_path)
        if state is None:
            worker.kill()
            worker.wait()
            with open(log_path, encoding="utf-8", errors="replace") as log:
                tail = log.read()[-2000:]
            message = f"the background process did not start. Log: {tail}"
            error = {"code": "internal", "message": message}
            state = {"job_id": job_id, "state": "failed", "error": error}
            state = write_state(job_dir, state)
        return public(state)

    def status(self, job_id: str) -> dict:
        job_dir, state = self._load(job_id)
        if state is None:
            return {"job_id": job_id, "state": "queued"}
        # Only the machine that runs the job can tell whether it still does.
        if state["state"] in TERMINAL_STATES or state["host"] != socket.gethostname():
            return public(state)
        if not is_alive(state["pid"], state["started_at"]):
            # The process is gone. It may have finished since we read the file.
            state = read_json(os.path.join(job_dir, "state.json"))
            if state["state"] not in TERMINAL_STATES:
                message = (
                    "the background process exited without reporting a result. "
                    f"See {os.path.join(job_dir, 'log.txt')} and check the table again."
                )
                failed = {"code": "internal", "message": message}
                state = {**state, "state": "failed", "error": failed}
                state = write_state(job_dir, state)
        return public(state)

    def cancel(self, job_id: str) -> dict:
        job_dir, state = self._load(job_id)
        if state is None:
            return {"job_id": job_id, "state": "queued"}
        if state["state"] in TERMINAL_STATES:
            return public(state)
        if state["host"] != socket.gethostname():
            message = f"the job runs on {state['host']}. Cancel it there."
            raise BackendError("unsupported", message)
        if not is_alive(state["pid"], state["started_at"]):
            # It ended on its own. Reports how, as status does.
            return self.status(job_id)
        # The process may exit on its own between the check and the signal.
        with contextlib.suppress(ProcessLookupError):
            os.kill(state["pid"], signal.SIGTERM)
            if not wait_until_gone(state, STOP_TIMEOUT):
                os.kill(state["pid"], signal.SIGKILL)
                wait_until_gone(state, STOP_TIMEOUT)
        # Keeps a final state the process managed to write before it stopped.
        return public(write_state(job_dir, {**state, "state": "canceled"}))

    def preview(self, job: dict) -> dict:
        storage = None
        try:
            storage = storage_options(storage_options_file())
            return preview_operation(job, storage, fragment_limit())
        except Exception as e:
            raise BackendError(classify(e), redact(str(e), storage)) from e

    def _load(self, job_id: str):
        job_dir = os.path.join(home(), job_id)
        if not os.path.isdir(job_dir):
            raise BackendError("job_not_found", f"no such job: {job_id}")
        return job_dir, read_json(os.path.join(job_dir, "state.json"))


def run_worker(job_dir: str) -> None:
    # Stderr of this process is the job's log.txt.
    log_to_stderr()
    pid = os.getpid()
    state = {
        "job_id": os.path.basename(job_dir),
        "state": "running",
        "host": socket.gethostname(),
        "pid": pid,
        "started_at": started_at(pid),
    }
    if state["started_at"] is None:
        # Without it a running job cannot be told from one that died.
        message = "cannot read the start time of a process: needs /proc or ps"
        error = {"code": "unsupported", "message": message}
        write_state(job_dir, {**state, "state": "failed", "error": error})
        log.error("job %s failed: %s", state["job_id"], error)
        return
    # Written before lance is imported, so it is there within milliseconds.
    write_state(job_dir, state)
    record = read_json(os.path.join(job_dir, "job.json"))
    job = record["job"]
    log.info("job %s: %s on %s", state["job_id"], job["operation"], job["table"])
    started = time.monotonic()
    storage = None
    try:
        # Relative paths in the storage settings mean what they meant to submit.
        os.chdir(record["cwd"])
        storage = storage_options(record["storage_options_file"])
        result = run_operation(job, storage, record["max_source_fragments"])
        final = {"state": "succeeded", "result": result}
    except Exception as e:
        error = {"code": classify(e), "message": redact(str(e), storage)}
        final = {"state": "failed", "error": error}
    took = time.monotonic() - started
    level = logging.ERROR if final["state"] == "failed" else logging.INFO
    outcome = final.get("error") or final["result"]
    log.log(
        level,
        "job %s %s after %.1fs: %s",
        state["job_id"],
        final["state"],
        took,
        outcome,
    )
    write_state(job_dir, {**state, **final})


if __name__ == "__main__":
    run_worker(sys.argv[1])
