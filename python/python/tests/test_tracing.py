# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import glob
import os
import subprocess
import sys
import uuid

import pytest


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="test fails in CI on Windows but passes locally on Windows",
)
def test_tracing():
    trace_files_before = set(glob.glob("trace-*.json"))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from lance.tracing import trace_to_chrome; trace_to_chrome()",
        ],
        check=True,
        env={
            "LANCE_LOG": "debug",
        },
    )
    trace_files_after = set(glob.glob("trace-*.json"))
    assert len(trace_files_before) + 1 == len(trace_files_after)

    new_trace_file = next(iter(trace_files_after - trace_files_before))

    os.remove(new_trace_file)

    some_uuid = uuid.uuid4()
    trace_name = f"{some_uuid}.json"

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from lance.tracing import trace_to_chrome;"
            + f"trace_to_chrome(file='{trace_name}')",
        ],
        check=True,
        env={
            "LANCE_LOG": "debug",
        },
    )

    assert os.path.exists(trace_name)
    os.remove(trace_name)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="test fails in CI on Windows but passes locally on Windows",
)
def test_tracing_callback(tmp_path):
    script = tmp_path / "script.py"
    script.write_text(
        """import lance
import pyarrow as pa
import time

from lance.tracing import capture_trace_events

def wait_until(condition, timeout=1, interval=0.01):
    start = time.time()
    while time.time() - start < timeout:
        if condition():
            return
        time.sleep(interval)
    raise TimeoutError(f"Condition not met after {timeout} seconds")

events = []
def callback(evt):
    events.append(evt)

capture_trace_events(callback)

lance.write_dataset(pa.table({"x": range(100)}), "memory://test")
wait_until(lambda: len(events) == 6)

assert events[0].target == "lance::dataset_events"
assert events[0].args["event"] == "loading"
assert events[0].args["uri"] == "memory://test"

assert events[1].target == "lance::dataset_events"
assert events[1].args["event"] == "writing"
assert events[1].args["uri"] == "memory://test"
assert events[1].args["mode"] == "Create"

assert events[2].target == "lance::file_audit"
assert events[2].args["mode"] == "create"
assert events[2].args["type"] == "data"

assert events[3].target == "lance::dataset_events"
assert events[3].args["event"] == "loading"
assert events[3].args["uri"] == "memory://test"

assert events[4].target == "lance::file_audit"
assert events[4].args["mode"] == "create"
assert events[4].args["type"] == "manifest"

assert events[5].target == "lance::dataset_events"
assert events[5].args["event"] == "committed"
assert events[5].args["uri"] == "memory://test"
assert events[5].args["read_version"] == "0"
assert events[5].args["committed_version"] == "1"
assert events[5].args["detached"] == "false"
assert events[5].args["operation"] == "Overwrite"

for e in events:
    assert e.args["timestamp"] is not None
    assert e.args["client_version"] == lance.__version__
"""
    )
    subprocess.run(
        [sys.executable, script],
        capture_output=True,
        check=True,
        env={
            "LANCE_LOG": "debug",
        },
    )


def test_tracing_callback_twice(tmp_path):
    # A second registration must raise a catchable error. Panicking would poison the
    # subscriber lock and every later operation in the process would panic too.
    script = tmp_path / "script.py"
    script.write_text(
        """import sys

import lance
import pyarrow as pa

from lance.tracing import capture_trace_events, shutdown_tracing

capture_trace_events(lambda evt: None)
try:
    capture_trace_events(lambda evt: None)
except RuntimeError as err:
    assert "already registered" in str(err)
else:
    raise AssertionError("the second capture_trace_events call did not raise")

path = sys.argv[1]
lance.write_dataset(pa.table({"x": range(10)}), path)
assert lance.dataset(path).count_rows() == 10

# Once the first callback is shut down, another one can be registered.
shutdown_tracing()
capture_trace_events(lambda evt: None)
assert lance.dataset(path).count_rows() == 10
"""
    )
    subprocess.run(
        [sys.executable, script, str(tmp_path / "data")],
        capture_output=True,
        check=True,
        env={
            "LANCE_LOG": "debug",
        },
    )
