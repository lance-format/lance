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
    """Stands in for the runner subprocess; `returncode` None means alive."""

    def __init__(self, *args, **kwargs):
        self.stdin = open(os.devnull, "wb")
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.fixture
def fake_runner(monkeypatch, tmp_path):
    monkeypatch.setattr(venv_manager.subprocess, "Popen", _FakeRunner)
    venv_path = tmp_path / "venv"
    venv_path.mkdir()
    return venv_manager.VenvExecutor("0.0.0", venv_path)


def test_cleanup_closes_runner_stderr_handle(fake_runner):
    fake_runner._ensure_subprocess()
    handle = fake_runner._stderr_file
    assert not handle.closed

    fake_runner.cleanup()
    assert handle.closed
    assert fake_runner._stderr_file is None
    assert not fake_runner.venv_path.exists()

    # A second cleanup is a no-op.
    fake_runner.cleanup()


def test_restarting_a_crashed_runner_closes_its_old_stderr_handle(fake_runner):
    fake_runner._ensure_subprocess()
    old_handle = fake_runner._stderr_file
    fake_runner._subprocess.returncode = 1  # the runner crashed

    fake_runner._ensure_subprocess()
    assert old_handle.closed
    assert not fake_runner._stderr_file.closed

    fake_runner.cleanup()
