# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import subprocess

import pytest

from . import venv_manager
from .venv_manager import _lance_namespace_dependency, _pip_install


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.0.1", "lance-namespace<0.7"),
        ("4.0.0b1", "lance-namespace<0.7"),
        ("6.0.0b5", "lance-namespace>=0.7.2,<0.8"),
        ("6.0.0", "lance-namespace>=0.7.2,<0.8"),
        ("7.2.0b5", "lance-namespace>=0.8.0,<0.9"),
        ("7.2.0", "lance-namespace>=0.8.0,<0.9"),
        ("12.0.0b5", "lance-namespace>=0.8.0,<0.9"),
        ("12.0.0b6", "lance-namespace>=0.11.1,<0.12"),
        ("12.0.0", "lance-namespace>=0.11.1,<0.12"),
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
