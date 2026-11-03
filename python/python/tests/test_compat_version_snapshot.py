# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Coverage for the version snapshot the compat tests are parametrized over."""

import json
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
compat_decorator._fetch_stable_versions = _fetch_stable_versions
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


def test_stable_release_discovery_excludes_prereleases(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            releases = {
                "14.0.0": [],
                "15.0.0b8": [],
                "15.0.0rc1": [],
                "15.0.0.dev1": [],
                "not-a-version": [],
            }
            return json.dumps({"releases": releases}).encode()

    monkeypatch.setattr(
        compat_decorator.urllib.request, "urlopen", lambda *args, **kwargs: Response()
    )

    assert [str(v) for v in compat_decorator._fetch_stable_versions()] == ["14.0.0"]


def test_version_snapshot_rejects_prereleases():
    compat_decorator.use_version_snapshot({"stable": ["14.0.0", "15.0.0b8"]})
    try:
        with pytest.raises(ValueError, match="only stable releases"):
            compat_decorator.pylance_stable_versions()
    finally:
        compat_decorator.use_version_snapshot({"stable": []})


def test_compat_versions_select_recent_stable_majors():
    compat_decorator.use_version_snapshot(
        {"stable": ["0.39.0", "12.0.0", "13.0.0", "14.0.0", "14.0.2"]}
    )
    assert compat_decorator.compat_versions() == ["14.0.2", "13.0.0", "12.0.0"]


@pytest.mark.parametrize("version", ["15.0.0b8", "15.0.0-beta.8", "15.0.0rc1"])
def test_compat_test_rejects_prerelease_minimum(version):
    with pytest.raises(ValueError, match="only stable releases"):
        compat_decorator.compat_test(min_version=version)
