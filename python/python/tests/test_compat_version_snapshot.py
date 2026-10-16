# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Coverage for the version snapshot the compat tests are parametrized over."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from compat import compat_decorator

TESTS_DIR = Path(__file__).resolve().parent

# Returns an extra release on gw1 only, so a worker that resolves the list itself
# parametrizes differently from the controller and from its sibling.
STUB_PLUGIN = """
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from packaging.version import Version

from compat import compat_decorator


def _fetch_stable_versions():
    extra = ["8.0.0"] if os.environ.get("PYTEST_XDIST_WORKER") == "gw1" else []
    return sorted(Version(v) for v in ["6.0.0", "7.0.0", *extra])


def _fetch_last_beta_release():
    return "7.1.0b1"


compat_decorator._fetch_stable_versions = _fetch_stable_versions
compat_decorator._fetch_last_beta_release = _fetch_last_beta_release
"""

# Not run with --run-compat: the generated cases stay collected but skipped, which
# is what makes their ids part of the collection xdist compares between workers.
INNER_TEST = """
import os

from compat import compat_decorator
from compat.compat_decorator import UpgradeDowngradeTest, compat_test

SEEDED_AT_IMPORT = compat_decorator._SNAPSHOT is not None


@compat_test()
class Sample(UpgradeDowngradeTest):
    def __init__(self, path):
        self.path = path


def test_snapshot_arrived_before_collection():
    assert os.environ["PYTEST_XDIST_WORKER"]
    assert SEEDED_AT_IMPORT, "worker resolved the version list itself"
"""


def test_workers_share_the_controller_version_snapshot(tmp_path):
    """Every xdist worker parametrizes on the releases the controller resolved.

    The compat suite discovers pylance releases over the network, so a worker
    left to query for them itself can collect a different parameter set than its
    siblings and abort the run. This pins both halves of the fix: that the
    snapshot reaches the worker at all, and that it arrives before collection
    imports any test module.
    """
    root = tmp_path / "inner"
    (root / "compat").mkdir(parents=True)
    shutil.copy(TESTS_DIR / "conftest.py", root / "conftest.py")
    shutil.copy(
        TESTS_DIR / "compat" / "compat_decorator.py",
        root / "compat" / "compat_decorator.py",
    )
    (root / "compat" / "__init__.py").touch()
    (root / "stubnet.py").write_text(STUB_PLUGIN)
    (root / "test_snapshot.py").write_text(INNER_TEST)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "stubnet",
            "-n",
            "2",
            "--dist",
            "loadgroup",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr

    assert "Different tests were collected" not in output, output
    assert "8.0.0" not in output, output
    assert result.returncode == 0, output


def _stub_pip_index(monkeypatch, available: str) -> None:
    """Stand in for the `pip index versions` call, checking how it asks.

    Pre-releases are only listed when `--pre` and the fury.io indices are
    passed, so a stub that ignored its arguments would keep passing if the
    command lost them.
    """

    class PipIndexResult:
        returncode = 0
        stdout = f"pylance (14.0.0)\nAvailable versions: {available}\n"

    def fake_run(args, **kwargs):
        assert "--pre" in args, args
        assert any("fury.io" in arg for arg in args), args
        return PipIndexResult()

    monkeypatch.setattr(compat_decorator.subprocess, "run", fake_run)


@pytest.mark.parametrize(
    ("available", "build", "expected"),
    [
        # The beta published from main carries the version the branch builds.
        ("14.0.0b16, 14.0.0b15, 14.0.0b14", "14.0.0b16", "14.0.0b15"),
        # The automation can be a release ahead of the branch.
        ("14.0.0b17, 14.0.0b16, 14.0.0b15", "14.0.0b16", "14.0.0b15"),
        # Ordering is by version, not by the order pip printed them.
        ("14.0.0b14, 14.0.0b15, 13.0.0", "14.0.0b16", "14.0.0b15"),
        # A stable release is not a pre-release, however recent it is.
        ("14.0.0b16, 13.5.0, 13.0.0b1", "14.0.0b16", "13.0.0b1"),
        # A version the index offers that does not parse is skipped.
        ("14.0.0b16, not-a-version, 14.0.0b15", "14.0.0b16", "14.0.0b15"),
        # Nothing older to pin leaves the slot empty rather than guessing.
        ("14.0.0b16", "14.0.0b16", None),
    ],
)
def test_beta_slot_takes_the_newest_release_older_than_the_build(
    monkeypatch, available, build, expected
):
    """A compat test's older build has to be older than the build under test.

    The beta slot is resolved from an index that carries the release just cut
    from main, and that one has the same version as every branch built since.
    Pinning it makes the "older build" the build under test.
    """
    _stub_pip_index(monkeypatch, available)
    assert compat_decorator._fetch_last_beta_release(build) == expected


def test_beta_slot_measures_against_the_installed_build(monkeypatch):
    """The default bound is the installed build, in the spelling it reports.

    `lance.__version__` carries Cargo's spelling of a pre-release, so the
    published wheel's normalized version has to compare equal to it rather
    than older.
    """
    _stub_pip_index(monkeypatch, "14.0.0b16, 14.0.0b15")
    monkeypatch.setattr(compat_decorator, "_build_version", lambda: "14.0.0-beta.16")
    assert compat_decorator._fetch_last_beta_release() == "14.0.0b15"


def test_beta_slot_stays_empty_without_a_build_to_measure_against(monkeypatch):
    """A host that only drives provisioned venvs has no pylance of its own."""
    monkeypatch.setattr(compat_decorator, "_build_version", lambda: None)
    assert compat_decorator._fetch_last_beta_release() is None


def test_build_version_is_unavailable_rather_than_fatal(monkeypatch, capsys):
    """An uninstalled pylance says so, instead of reading as a failed query."""
    monkeypatch.setitem(sys.modules, "lance", None)
    assert compat_decorator._build_version() is None
    assert "build's own version" in capsys.readouterr().err
