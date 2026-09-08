# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Coverage for the version snapshot the compat tests are parametrized over."""

import shutil
import subprocess
import sys
from pathlib import Path

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
