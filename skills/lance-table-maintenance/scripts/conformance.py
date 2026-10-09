#!/usr/bin/env python3
"""Check an execution backend against the interface.

Drives the backend through maintain.py, one new process per call, and
prints one PASS, FAIL or SKIP line per check. The table is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

from backends.base import TERMINAL_STATES

MAINTAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maintain.py")
CALL_LIMIT = 60
JOBS = (
    ("compact", {}),
    ("optimize_indices", {}),
    ("cleanup", {"retain_versions": 1}),
)


class Checker:
    def __init__(self, backend: str, table: str, timeout: int, workdir: str):
        self.backend = backend
        self.table = table
        self.timeout = timeout
        self.workdir = workdir
        self.failed = False
        self.slowest = 0.0

    def report(self, outcome: str, name: str, detail="") -> None:
        self.failed = self.failed or outcome == "FAIL"
        print(f"{outcome} {name}: {detail}")

    def call(self, command: str, target: str):
        """Returns (exit code, the JSON object printed, or None if it is not one)."""
        args = [sys.executable, MAINTAIN, command, target, "--backend", self.backend]
        started = time.monotonic()
        done = subprocess.run(args, capture_output=True, text=True)
        self.slowest = max(self.slowest, time.monotonic() - started)
        try:
            output = json.loads(done.stdout)
        except ValueError:
            output = None
        if not isinstance(output, dict):
            self.report("FAIL", f"{command} prints one JSON object", done.stdout[:200])
            output = {}
        return done.returncode, output

    def job_file(self, operation: str, options: dict) -> str:
        path = os.path.join(self.workdir, f"{operation}.json")
        job = {"table": self.table, "operation": operation, "options": options}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(job, f)
        return path

    def wait(self, job_id: str) -> dict:
        deadline = time.monotonic() + self.timeout
        while True:
            _, status = self.call("status", job_id)
            if status.get("state") in TERMINAL_STATES or time.monotonic() > deadline:
                return status
            time.sleep(1)

    def run(self, own_table: bool) -> None:
        code, output = self.call("status", "no-such-job-0000")
        found = output.get("error", {}).get("code")
        outcome = "PASS" if code != 0 and found == "job_not_found" else "FAIL"
        self.report(outcome, "unknown job id is job_not_found", found)

        code, output = self.call("preview", self.job_file("compact", {}))
        if code == 0:
            self.report("PASS", "preview", output)
        elif output.get("error", {}).get("code") == "unsupported":
            self.report("SKIP", "preview", "not supported")
        else:
            self.report("FAIL", "preview", output)

        finished = None
        for operation, options in JOBS:
            code, output = self.call("submit", self.job_file(operation, options))
            if output.get("error", {}).get("code") == "unsupported":
                self.report("SKIP", f"{operation} job", "not supported")
                continue
            if code != 0:
                self.report("FAIL", f"{operation} job", output)
                continue
            _, seen = self.call("status", output["job_id"])
            same = seen.get("job_id") == output["job_id"]
            outcome = "PASS" if same else "FAIL"
            self.report(outcome, f"{operation} job is found by its id", seen)
            status = self.wait(output["job_id"])
            outcome = "PASS" if status.get("state") == "succeeded" else "FAIL"
            self.report(outcome, f"{operation} job succeeds", status)
            finished = finished or (outcome == "PASS" and status)
            if own_table and outcome == "PASS":
                self.check_effect(operation)

        if finished:
            _, again = self.call("status", finished["job_id"])
            code, canceled = self.call("cancel", finished["job_id"])
            stable = again.get("state") == canceled.get("state") == finished["state"]
            outcome = "PASS" if code == 0 and stable else "FAIL"
            self.report(outcome, "a final state does not change", canceled)

        if own_table:
            append(self.table)
        code, output = self.call("submit", self.job_file("compact", {}))
        if code == 0:
            self.call("cancel", output["job_id"])
            state = self.wait(output["job_id"]).get("state")
            outcome = "PASS" if state in TERMINAL_STATES else "FAIL"
            self.report(outcome, "a canceled job reaches a final state", state)
        else:
            self.report("FAIL", "a canceled job reaches a final state", output)

        outcome = "PASS" if self.slowest < CALL_LIMIT else "FAIL"
        detail = f"slowest took {self.slowest:.1f}s"
        self.report(outcome, f"every call returns within {CALL_LIMIT}s", detail)

    def check_effect(self, operation: str) -> None:
        import lance
        from lance.optimize import Compaction

        ds = lance.dataset(self.table)
        if operation == "compact":
            left = Compaction.plan(ds, {}).num_tasks()
            outcome, detail = ("PASS" if left == 0 else "FAIL"), f"{left} tasks left"
        elif operation == "optimize_indices":
            rows = ds.count_rows()
            indices = ds.describe_indices()
            # Lance's own indices do not cover rows.
            indices = [i for i in indices if not i.name.startswith("__lance_")]
            left = sum(rows - i.num_rows_indexed for i in indices)
            outcome, detail = ("PASS" if left == 0 else "FAIL"), f"{left} rows left"
        else:
            left = len(ds.versions())
            outcome, detail = ("PASS" if left == 1 else "FAIL"), f"{left} versions"
        self.report(outcome, f"{operation} changed the table", detail)


def append(uri: str) -> None:
    import lance
    import pyarrow as pa

    table = pa.table({"id": pa.array(range(2000), pa.int64())})
    lance.write_dataset(table, uri, mode="append", max_rows_per_file=20)


def create_table(uri: str) -> None:
    import lance
    import pyarrow as pa

    table = pa.table({"id": pa.array(range(2000), pa.int64())})
    ds = lance.write_dataset(table, uri, max_rows_per_file=20)
    ds.create_scalar_index("id", "BTREE")
    append(uri)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="local")
    parser.add_argument(
        "--table",
        help="Table the backend can reach. It is modified. Default: a temporary table",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per job")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        os.environ["LANCE_MAINTENANCE_HOME"] = os.path.join(workdir, "state")
        table = args.table or os.path.join(workdir, "table.lance")
        if not args.table:
            create_table(table)
        checker = Checker(args.backend, table, args.timeout, workdir)
        checker.run(own_table=not args.table)
    sys.exit(1 if checker.failed else 0)


if __name__ == "__main__":
    main()
