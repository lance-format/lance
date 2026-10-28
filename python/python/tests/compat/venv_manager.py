# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Virtual environment management for compatibility testing.

Manages creation and execution of test code in isolated virtual environments
with specific Lance versions installed.
"""

import contextlib
import glob
import os
import pickle
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import pytest
from packaging.version import InvalidVersion, Version

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


@contextlib.contextmanager
def _venv_lock(lock_path: Path):
    """Hold an exclusive lock so parallel workers don't race creating the same venv."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)


_PIP_INSTALL_ATTEMPTS = 3
_PIP_RETRY_BACKOFF_SECONDS = 2.0


def _pip_install(python: Union[str, Path], args: list[str]) -> None:
    """Run ``pip install`` with output captured, retrying transient failures.

    pip installs hit the network (PyPI / fury.io) and a single flaky download
    should not fail a whole test run, so retry a bounded number of times with a
    short backoff. On final failure raise an error that includes pip's
    stdout/stderr, which a bare CalledProcessError from ``capture_output=True``
    would hide from the test log.
    """
    cmd = [str(python), "-m", "pip", "install", "--quiet", *args]
    for attempt in range(1, _PIP_INSTALL_ATTEMPTS + 1):
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as exc:  # noqa: PERF203
            if attempt == _PIP_INSTALL_ATTEMPTS:
                raise RuntimeError(
                    f"pip install failed after {attempt} attempts "
                    f"(exit status {exc.returncode}): {cmd}\n"
                    f"stdout:\n{exc.stdout}\n"
                    f"stderr:\n{exc.stderr}"
                ) from exc
            time.sleep(_PIP_RETRY_BACKOFF_SECONDS * attempt)


NAMESPACE_0_6_DEPENDENCY = "lance-namespace<0.7"
NAMESPACE_0_7_DEPENDENCY = "lance-namespace>=0.7.2,<0.8"
NAMESPACE_0_8_DEPENDENCY = "lance-namespace>=0.8.0,<0.9"
NAMESPACE_0_11_DEPENDENCY = "lance-namespace>=0.11.1,<0.12"


def _lance_namespace_dependency(pylance_version: str) -> str:
    # 12.0.0b5 is the last release published while pylance still pinned
    # lance-namespace <0.9; releases cut after that carry the 0.11 range.
    if Version(pylance_version) > Version("12.0.0b5"):
        return NAMESPACE_0_11_DEPENDENCY
    if Version(pylance_version) >= Version("7.2.0b5"):
        return NAMESPACE_0_8_DEPENDENCY
    if Version(pylance_version) >= Version("6.0.0b0"):
        return NAMESPACE_0_7_DEPENDENCY
    return NAMESPACE_0_6_DEPENDENCY


def _is_release_version(ref: str) -> bool:
    """A ref is treated as a published release (install a wheel) if it parses as a
    version; anything else (commit sha, branch, tag) is built from source."""
    try:
        Version(ref)
        return True
    except InvalidVersion:
        return False


def _prebuilt_wheel_for(ref: str) -> Optional[str]:
    """A prebuilt wheel to install for `ref` instead of building it from source.

    When CI has already built a ref (e.g. the PR head, built once by the Python build
    job), COMPAT_PREBUILT_REF names that ref and COMPAT_PREBUILT_WHEEL points at the
    wheel (a path or glob). Lets the PR workflow reuse that wheel rather than rebuilding
    the reader. Returns None when no prebuilt wheel applies to `ref`.
    """
    if os.environ.get("COMPAT_PREBUILT_REF") != ref:
        return None
    pattern = os.environ.get("COMPAT_PREBUILT_WHEEL")
    if not pattern:
        return None
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"COMPAT_PREBUILT_WHEEL={pattern!r} matched no wheel for ref {ref!r}"
        )
    return matches[0]


def _repo_root() -> Path:
    """Lance source checkout holding this test file (used to build refs from source)."""
    # .../python/python/tests/compat/venv_manager.py -> repo root is parents[4]
    return Path(__file__).resolve().parents[4]


def _safe(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", ref)


# Explicit skip whitelist for known (lance_version, python_version) incompatibilities.
# Each entry is (max_lance_version_inclusive, min_python_version_inclusive).
# A test is skipped when the tested lance version <= max_lance AND the current
# Python >= min_python.  Add entries here only after verifying the incompatibility
# is a runtime/ABI issue rather than a format regression — anything NOT listed
# will surface as a test failure so new problems are visible.
_COMPAT_SKIP: list[tuple[str, tuple[int, int]]] = [
    # lance 0.22.x abi3 wheel was built with old PyO3 (<0.23) that crashes on
    # Python 3.14+ due to removed internal CPython APIs.
    ("0.22.0", (3, 14)),
]


def _skip_reason(lance_version: str) -> Optional[str]:
    """Return a skip reason if this lance/Python combo is whitelisted, else None."""
    try:
        ver = Version(lance_version)
    except InvalidVersion:
        return None
    py = sys.version_info[:2]
    for max_lance, min_python in _COMPAT_SKIP:
        if ver <= Version(max_lance) and py >= min_python:
            py_str = f"{py[0]}.{py[1]}"
            return (
                f"Lance {lance_version} + Python {py_str}: whitelisted skip "
                f"(see _COMPAT_SKIP in venv_manager.py)"
            )
    return None


class VenvExecutor:
    """Manages a virtual environment with a specific Lance version."""

    def __init__(self, version: str, venv_path: Path, persistent: bool = False):
        """
        Initialize a VenvExecutor.

        Parameters
        ----------
        version : str
            Lance version to install (e.g., "0.30.0")
        venv_path : Path
            Directory where virtual environment will be created
        persistent : bool
            If True, venv is persistent and validated before use
        """
        self.version = version
        self.venv_path = Path(venv_path)
        self.persistent = persistent
        self._created = False
        self._subprocess: Optional[subprocess.Popen] = None
        self._stderr_path: Optional[Path] = None
        self._stderr_file = None

    @property
    def python_path(self) -> Path:
        if sys.platform == "win32":
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"

    @property
    def _marker_path(self) -> Path:
        return self.venv_path / ".compat_ref"

    @staticmethod
    def _python_version_tag() -> str:
        return f"{sys.version_info.major}.{sys.version_info.minor}"

    def _validate_venv(self) -> bool:
        """A cached venv is reusable if it exists, its recorded ref matches, and it was
        built with the same Python major.minor as the current interpreter.

        The marker file format is two lines: `<lance_ref>\\n<python_major.minor>`.
        Old single-line markers (no Python version) are treated as stale so the venv
        is rebuilt — this handles cached venvs from a different Python installation."""
        if not self.python_path.exists():
            return False
        try:
            lines = self._marker_path.read_text().strip().splitlines()
        except OSError:
            return False
        if not lines or lines[0] != self.version:
            return False
        # Require a Python version line; single-line markers are stale.
        if len(lines) < 2 or lines[1] != self._python_version_tag():
            return False
        return True

    def create(self):
        """Create the virtual environment and install the specified Lance version."""
        if self._created:
            return
        if self.persistent and self._validate_venv():
            self._created = True
            return

        # Lock so parallel workers don't build the same venv at once; re-check in the
        # lock since another worker may have just finished it.
        with _venv_lock(self.venv_path.parent / f".lock_{_safe(self.version)}"):
            if not self._validate_venv():
                if self.venv_path.exists():
                    shutil.rmtree(self.venv_path)  # drop any partial build
                subprocess.run(
                    [sys.executable, "-m", "venv", str(self.venv_path)],
                    check=True,
                    capture_output=True,
                )
                # Prefer a wheel CI already built for this ref; else a published
                # release installs its wheel; else build the ref (commit/branch/tag)
                # from source -- so two arbitrary refs can be compared and only the
                # ones without a wheel pay a build.
                prebuilt = _prebuilt_wheel_for(self.version)
                if prebuilt is not None:
                    self._install_wheel(prebuilt)
                elif _is_release_version(self.version):
                    self._install_release_wheel()
                else:
                    self._build_from_source()
                self._marker_path.write_text(
                    f"{self.version}\n{self._python_version_tag()}"
                )
        self._created = True

    def _install_wheel(self, wheel: str):
        _pip_install(self.python_path, [wheel, "pytest"])

    def _install_release_wheel(self):
        _pip_install(
            self.python_path,
            [
                "--pre",
                "--extra-index-url",
                "https://pypi.fury.io/lance-format/",
                "--extra-index-url",
                "https://pypi.fury.io/lancedb/",
                f"pylance=={self.version}",
                # Older Lance wheels (e.g. 2.0.1, 4.0.0b1) import
                # CreateEmptyTableRequest from lance_namespace, which was
                # removed in lance-namespace 0.7.0. Pin to <0.7 so old wheels
                # resolve a compatible transitive dep.
                _lance_namespace_dependency(self.version),
                "pytest",
            ],
        )

    def _build_from_source(self):
        """Build a wheel for an arbitrary git ref via a worktree + maturin, then install
        it. The worktree/build is cached by ref so it is paid at most once."""
        py = str(self.python_path)
        src = self.venv_path.parent / f"src_{_safe(self.version)}"
        if not src.exists():
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(_repo_root()),
                    "worktree",
                    "add",
                    "--detach",
                    str(src),
                    self.version,
                ],
                check=True,
                capture_output=True,
            )
        _pip_install(py, ["maturin", "pytest", "pyarrow"])
        wheels = src / "target" / "compat-wheels"
        subprocess.run(
            [
                py,
                "-m",
                "maturin",
                "build",
                "--release",
                "--interpreter",
                py,
                "-m",
                str(src / "python" / "Cargo.toml"),
                "--out",
                str(wheels),
            ],
            check=True,
            capture_output=True,
        )
        wheel = next(wheels.glob("pylance-*.whl"))
        _pip_install(py, [str(wheel)])

    def _ensure_subprocess(self):
        """Ensure the persistent subprocess is running."""
        if self._subprocess is not None and self._subprocess.poll() is None:
            # Subprocess is already running
            return

        # Start persistent subprocess
        runner_script = Path(__file__).parent / "venv_runner.py"

        # Set PYTHONPATH so the subprocess can import compat test modules.
        # pytest adds the `tests/` directory (the first ancestor without __init__.py)
        # to sys.path, so test modules are imported as `compat.<module>`.
        env = os.environ.copy()
        tests_dir = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(tests_dir)
        env.setdefault("RUST_BACKTRACE", "full")

        # Capture stderr to a file so a Rust panic (which crashes the runner) can be
        # surfaced in the error instead of an opaque "broken pipe".
        self._stderr_path = self.venv_path / ".runner_stderr.log"
        self._stderr_file = open(self._stderr_path, "w")
        self._subprocess = subprocess.Popen(
            [str(self.python_path), "-u", str(runner_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            env=env,
        )

    def _last_panic(self) -> str:
        """Pull the panic message from the runner's captured stderr, if any."""
        try:
            text = self._stderr_path.read_text()
        except (OSError, AttributeError):
            return ""
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "panicked at" in line:
                # Compact the long path to just "builder.rs:962:57"
                loc = line.split("panicked at", 1)[1].strip().rstrip(":")
                loc = loc.rsplit("/", 1)[-1]
                msg = lines[i + 1].strip() if i + 1 < len(lines) else ""
                return f"panic at {loc}: {msg}" if msg else f"panic at {loc}"
        tail = [line.strip() for line in lines if line.strip()]
        return tail[-1] if tail else ""

    def _send_message(self, obj: Any):
        """Send a length-prefixed pickled message to subprocess."""
        data = pickle.dumps(obj)
        length = struct.pack(">I", len(data))
        self._subprocess.stdin.write(length)
        self._subprocess.stdin.write(data)
        self._subprocess.stdin.flush()

    def _receive_message(self) -> Any:
        """Receive a length-prefixed pickled message from subprocess."""
        # Short reads mean the subprocess closed stdout (usually a crash); raise
        # EOFError so the caller can surface the panic from captured stderr.
        length_bytes = self._subprocess.stdout.read(4)
        if len(length_bytes) < 4:
            raise EOFError("subprocess closed stdout before sending a message length")

        length = struct.unpack(">I", length_bytes)[0]

        # Read message data
        data = self._subprocess.stdout.read(length)
        if len(data) < length:
            raise EOFError(
                f"incomplete message: expected {length} bytes, got {len(data)}"
            )

        return pickle.loads(data)

    def execute_method(
        self,
        obj: Any,
        method_name: str,
        env_overrides: Optional[dict[str, str]] = None,
    ) -> Any:
        """
        Execute a method on a pickled object in the virtual environment.

        Uses a persistent subprocess to avoid repeatedly importing Lance and
        its dependencies.

        Parameters
        ----------
        obj : Any
            Object to pickle and send to venv. Must be picklable.
        method_name : str
            Name of the method to call on the object

        Returns
        -------
        Any
            Return value from the method call

        Raises
        ------
        Exception
            Re-raises any exception that occurred in the venv
        """
        if not self._created:
            raise RuntimeError("Virtual environment not created. Call create() first.")

        reason = _skip_reason(self.version)
        if reason:
            pytest.skip(reason)

        # Ensure subprocess is running
        self._ensure_subprocess()
        try:
            # Send request: (obj, method_name, env_overrides)
            self._send_message((obj, method_name, env_overrides or {}))

            # Receive response
            response = self._receive_message()

            if response["success"]:
                return response["result"]
            else:
                # Error occurred in subprocess
                error_msg = (
                    f"Error in venv (Lance {self.version}) calling {method_name}:\n"
                    f"{response['exception_type']}: {response['exception_msg']}\n"
                    f"\nTraceback from venv:\n{response['traceback']}"
                )
                raise RuntimeError(error_msg)

        except (BrokenPipeError, EOFError, struct.error) as e:
            # Subprocess died (usually a Rust panic); flush it, then surface that.
            returncode = "unknown"
            if self._subprocess is not None:
                try:
                    self._subprocess.wait(timeout=2)
                    returncode = str(self._subprocess.returncode)
                except Exception:
                    pass
            panic = self._last_panic()
            detail = panic or f"subprocess communication failed: {e}"
            raise RuntimeError(f"Lance {self.version} (exit={returncode}): {detail}")

    def cleanup(self):
        """Remove the virtual environment directory and terminate subprocess."""
        # Terminate the persistent subprocess
        if self._subprocess is not None:
            try:
                self._subprocess.stdin.close()
                self._subprocess.terminate()
                self._subprocess.wait(timeout=5)
            except Exception:
                # Force kill if graceful termination fails
                self._subprocess.kill()
            finally:
                self._subprocess = None

        # Remove venv directory
        if self.venv_path.exists():
            shutil.rmtree(self.venv_path)
        self._created = False


class VenvFactory:
    """Factory for creating and managing VenvExecutor instances."""

    def __init__(self, base_path: Path, persistent: bool = False):
        """
        Initialize the factory.

        Parameters
        ----------
        base_path : Path
            Base directory for creating virtual environments
        persistent : bool
            If True, venvs are not cleaned up and can be reused across sessions
        """
        self.base_path = Path(base_path)
        self.persistent = persistent
        self.venvs: dict[str, VenvExecutor] = {}

    def get_venv(self, version: str) -> VenvExecutor:
        """
        Get or create a VenvExecutor for the specified version.

        Parameters
        ----------
        version : str
            Lance version

        Returns
        -------
        VenvExecutor
            Executor for the specified version
        """
        if version not in self.venvs:
            venv_path = self.base_path / f"venv_{_safe(version)}"
            executor = VenvExecutor(version, venv_path, persistent=self.persistent)
            executor.create()
            self.venvs[version] = executor
        return self.venvs[version]

    def cleanup_all(self):
        """Clean up all created virtual environments (skips persistent venvs)."""
        if not self.persistent:
            for venv in self.venvs.values():
                venv.cleanup()
        self.venvs.clear()
