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
from packaging.version import Version

TESTS_DIR = Path(__file__).resolve().parent


@pytest.fixture
def snapshot_releases(monkeypatch):
    """Serve the given stable releases and restore the cached snapshot after."""

    def use(*releases):
        monkeypatch.setattr(compat_decorator, "_SNAPSHOT", {"stable": list(releases)})

    return use


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


def _source_version():
    return Version("9.0.0")


compat_decorator._fetch_stable_versions = _fetch_stable_versions
compat_decorator._source_version = _source_version
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


def test_version_snapshot_rejects_prereleases(snapshot_releases):
    snapshot_releases("14.0.0", "15.0.0b8")
    with pytest.raises(ValueError, match="only stable releases"):
        compat_decorator.pylance_stable_versions()


def test_compat_versions_select_recent_stable_majors(snapshot_releases, monkeypatch):
    # Pin the checkout version so the selection under test does not depend on
    # which branch this file runs from.
    monkeypatch.setattr(compat_decorator, "_source_version", lambda: Version("15.0.0"))
    snapshot_releases("0.39.0", "12.0.0", "13.0.0", "14.0.0", "14.0.2")
    assert compat_decorator.compat_versions() == ["14.0.2", "13.0.0", "12.0.0"]


@pytest.mark.parametrize("version", ["15.0.0b8", "15.0.0-beta.8", "15.0.0rc1"])
def test_compat_test_rejects_prerelease_minimum(version):
    with pytest.raises(ValueError, match="only stable releases"):
        compat_decorator.compat_test(min_version=version)


def test_compat_versions_skip_releases_newer_than_the_source(snapshot_releases):
    newer = f"{compat_decorator._source_version().major + 1}.0.0"
    snapshot_releases("13.0.0", newer)
    assert compat_decorator.compat_versions() == ["13.0.0"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # A pre-release checkout predates the release it leads to, so a stable
        # build of that release may write data this checkout cannot read.
        ("15.0.0-beta.25", ["14.0.2"]),
        ("15.0.0-rc.1", ["14.0.2"]),
        # A checkout at a stable version does test against that release.
        ("15.0.0", ["15.0.0", "14.0.2"]),
    ],
)
def test_compat_versions_bound_releases_by_checkout_version(
    source, expected, snapshot_releases, monkeypatch
):
    monkeypatch.setattr(compat_decorator, "_source_version", lambda: Version(source))
    snapshot_releases("14.0.2", "15.0.0")
    assert compat_decorator.compat_versions() == expected
