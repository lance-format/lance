# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""The table maintenance skill in skills/, driven the way an agent drives it:
through its scripts, on a table that needs every operation."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import lance
import pyarrow as pa
import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "skills/lance-table-maintenance/scripts"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the default backend runs on Linux and macOS"
)


def run(env, script, *args):
    """Runs a script of the skill. Returns (exit code, the JSON object it printed)."""
    command = [sys.executable, str(SCRIPTS / script), *args]
    done = subprocess.run(command, capture_output=True, text=True, env=env)
    return done.returncode, json.loads(done.stdout)


def job(tmp_path, table, operation, options):
    path = tmp_path / f"{operation}.json"
    path.write_text(
        json.dumps({"table": table, "operation": operation, "options": options})
    )
    return str(path)


def wait(env, job_id):
    for _ in range(300):
        _, status = run(env, "maintain.py", "status", job_id)
        if status["state"] in ("succeeded", "failed", "canceled"):
            return status
        time.sleep(0.2)
    raise TimeoutError(job_id)


@pytest.fixture
def env(tmp_path):
    return {**os.environ, "LANCE_MAINTENANCE_HOME": str(tmp_path / "state")}


@pytest.fixture
def table(tmp_path):
    """40 small fragments, 8 of them not covered by the index, deleted rows, a tag."""
    uri = str(tmp_path / "t.lance")

    def rows(start, n):
        return pa.table({"id": pa.array(range(start, start + n), pa.int64())})

    ds = lance.write_dataset(rows(0, 3200), uri, max_rows_per_file=100)
    ds.tags.create("keep", ds.version)
    ds.create_scalar_index("id", "BTREE")
    lance.write_dataset(rows(3200, 800), uri, mode="append", max_rows_per_file=100)
    lance.dataset(uri).delete("id % 7 = 0")
    return uri


def test_doctor_report_matches_the_table(env, table, tmp_path):
    ds = lance.dataset(table)
    files = sorted(Path(table).rglob("*"))
    code, report = run(env, "doctor.py", table)
    assert code == 0 and report["errors"] == {}
    assert report["num_rows"] == ds.count_rows()
    assert report["num_fragments"] == len(ds.get_fragments()) == 40
    assert report["num_deleted_rows"] == ds.stats.dataset_stats()["num_deleted_rows"]
    assert report["compaction_tasks"] > 0
    assert report["tags"] == {"keep": 1}
    (index,) = report["indices"]
    assert index["num_unindexed_rows"] == ds.count_rows("id >= 3200")
    assert sorted(Path(table).rglob("*")) == files, "the health check is read-only"

    code, out = run(env, "doctor.py", str(tmp_path / "nope.lance"))
    assert code != 0 and out["error"]["code"] == "table_not_found"


def test_jobs_in_the_recommended_order_leave_the_table_healthy(env, table, tmp_path):
    ids = sorted(lance.dataset(table).to_table()["id"].to_pylist())

    _, preview = run(env, "maintain.py", "preview", job(tmp_path, table, "compact", {}))
    assert preview["tasks"] > 0 and preview["fragments_rewritten"] == 40
    assert preview["deleted_rows_dropped"] == 4000 - len(ids)
    blocked = job(tmp_path, table, "cleanup", {"retain_versions": 1})
    code, out = run(env, "maintain.py", "preview", blocked)
    assert code != 0 and out["error"]["code"] == "invalid_input", "a tag is in the way"

    jobs = [
        ("optimize_indices", {}),
        ("compact", {}),
        ("cleanup", {"retain_versions": 1, "error_if_tagged_old_versions": False}),
    ]
    for operation, options in jobs:
        _, submitted = run(
            env, "maintain.py", "submit", job(tmp_path, table, operation, options)
        )
        assert wait(env, submitted["job_id"])["state"] == "succeeded", operation

    _, report = run(env, "doctor.py", table)
    assert report["num_fragments"] == 1 and report["num_deleted_rows"] == 0
    assert (
        report["compaction_tasks"] == 0
        and report["indices"][0]["num_unindexed_rows"] == 0
    )
    ds = lance.dataset(table)
    assert [v["version"] for v in ds.versions()] == [1, ds.version]
    assert sorted(ds.to_table()["id"].to_pylist()) == ids


@pytest.mark.parametrize(
    "bad",
    [
        {"operation": "compact", "options": {"retain_version": 1}},
        {"operation": "cleanup", "options": {}},
        {
            "table": "s3://key:secret@bucket/t.lance",
            "operation": "compact",
            "options": {},
        },
    ],
    ids=[
        "misspelled option",
        "cleanup without retention",
        "credentials in the table URI",
    ],
)
def test_maintain_rejects_a_job_it_cannot_run_safely(env, table, tmp_path, bad):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({"table": table, **bad}))
    code, out = run(env, "maintain.py", "submit", str(path))
    assert code != 0 and out["error"]["code"] == "invalid_input"
    assert not (tmp_path / "state").exists(), "no job was started"


def test_local_backend_passes_the_conformance_check(env):
    done = subprocess.run(
        [sys.executable, str(SCRIPTS / "conformance.py")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "FAIL" not in done.stdout
