# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import subprocess

import pytest

from . import venv_manager
from .venv_manager import _lance_namespace_dependency, _pip_install


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.0.1", "lance-namespace<0.7"),
        ("5.0.0", "lance-namespace<0.7"),
        ("6.0.0", "lance-namespace>=0.7.2,<0.8"),
        ("7.2.0", "lance-namespace>=0.8.0,<0.9"),
        ("11.0.0", "lance-namespace>=0.8.0,<0.9"),
        ("12.0.0", "lance-namespace>=0.11.1,<0.12"),
        ("13.0.0", "lance-namespace>=0.11.1,<0.12"),
        ("14.0.0", "lance-namespace>=0.13.0,<0.14"),
    ],
)
def test_lance_namespace_dependency(version: str, expected: str):
    assert _lance_namespace_dependency(version) == expected


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(venv_manager.time, "sleep", sleeps.append)
    return sleeps


def test_pip_install_failure_includes_pip_output(monkeypatch, no_sleep):
    calls: list[list[str]] = []

    def failing_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(
            2, cmd, output="resolving deps", stderr="No matching distribution found"
        )

    monkeypatch.setattr(venv_manager.subprocess, "run", failing_run)
    with pytest.raises(RuntimeError) as exc_info:
        _pip_install("python", ["pylance==0.38.0"])
    # The error must surface pip's captured output, which CalledProcessError hides.
    message = str(exc_info.value)
    assert "No matching distribution found" in message
    assert "resolving deps" in message
    assert "pylance==0.38.0" in message
    assert len(calls) == venv_manager._PIP_INSTALL_ATTEMPTS
    assert len(no_sleep) == venv_manager._PIP_INSTALL_ATTEMPTS - 1


def test_pip_install_retries_transient_failure(monkeypatch, no_sleep):
    calls: list[list[str]] = []

    def flaky_run(cmd, **kwargs):
        calls.append(cmd)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                2, cmd, output="", stderr="ReadTimeoutError: pypi.fury.io"
            )

    monkeypatch.setattr(venv_manager.subprocess, "run", flaky_run)
    _pip_install("python", ["pylance==0.38.0"])
    assert len(calls) == 2
    assert len(no_sleep) == 1


def test_pip_install_succeeds_first_try_without_sleeping(monkeypatch, no_sleep):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        venv_manager.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd)
    )
    _pip_install("python", ["pytest"])
    assert calls == [["python", "-m", "pip", "install", "--quiet", "pytest"]]
    assert no_sleep == []


class _FakeRunner:
    """Stands in for the runner subprocess; `returncode` None means alive.

    Records the stderr file it was given and the calls made on it.
    """

    instances: list["_FakeRunner"] = []

    def __init__(self, *args, stderr=None, **kwargs):
        self.stdin = open(os.devnull, "wb")
        self.stderr_file = stderr
        self.returncode = None
        self.calls: list[str] = []
        self.wait_times_out = False
        _FakeRunner.instances.append(self)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")

    def wait(self, timeout=None):
        self.calls.append("wait")
        if self.wait_times_out and timeout is not None:
            raise subprocess.TimeoutExpired("runner", timeout)
        self.returncode = -9 if "kill" in self.calls else -15
        return self.returncode

    def kill(self):
        self.calls.append("kill")


@pytest.fixture
def executor(tmp_path):
    _FakeRunner.instances.clear()
    venv_path = tmp_path / "venv"
    venv_path.mkdir()
    return venv_manager.VenvExecutor("0.0.0", venv_path)


@pytest.mark.parametrize("wait_times_out", [False, True])
def test_runner_stderr_handle_is_closed_in_the_parent(
    monkeypatch, executor, wait_times_out
):
    monkeypatch.setattr(venv_manager.subprocess, "Popen", _FakeRunner)
    executor._ensure_subprocess()
    runner = _FakeRunner.instances[-1]
    # The child holds its own copy; the parent's handle closes right away.
    assert runner.stderr_file.closed

    runner.wait_times_out = wait_times_out
    executor.cleanup()
    expected = ["terminate", "wait"] + (["kill", "wait"] if wait_times_out else [])
    assert runner.calls == expected
    assert not executor.venv_path.exists()

    # A second cleanup is a no-op.
    executor.cleanup()


def test_runner_stderr_handle_is_closed_when_startup_fails(monkeypatch, executor):
    handed_to_popen = []

    def failing_popen(*args, stderr=None, **kwargs):
        handed_to_popen.append(stderr)
        raise FileNotFoundError("python")

    monkeypatch.setattr(venv_manager.subprocess, "Popen", failing_popen)
    with pytest.raises(FileNotFoundError):
        executor._ensure_subprocess()
    assert handed_to_popen[0].closed
    executor.cleanup()
    assert not executor.venv_path.exists()
