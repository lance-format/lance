# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""numpy is an optional dependency: importing lance must not require it."""

import subprocess
import sys
import textwrap


def _run_without_numpy(code: str) -> subprocess.CompletedProcess:
    # Hide numpy from both `import` and `find_spec`, then run the snippet in a
    # fresh interpreter.
    guard = textwrap.dedent(
        """
        import sys

        class _BlockNumpy:
            def find_spec(self, name, path=None, target=None):
                if name == "numpy" or name.startswith("numpy."):
                    raise ModuleNotFoundError("No module named 'numpy'")
                return None

        sys.meta_path.insert(0, _BlockNumpy())
        """
    )
    return subprocess.run(
        [sys.executable, "-c", guard + code],
        capture_output=True,
        text=True,
    )


def test_import_lance_without_numpy():
    result = _run_without_numpy(
        textwrap.dedent(
            """
            import lance  # noqa: F401
            from lance.dependencies import _NUMPY_AVAILABLE
            assert _NUMPY_AVAILABLE is False
            print("ok")
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_numpy_path_errors_clearly_without_numpy():
    # A numpy-only path should raise a helpful error, not crash at import.
    result = _run_without_numpy(
        textwrap.dedent(
            """
            from lance.dependencies import numpy as np
            try:
                np.array([1, 2, 3])
            except ModuleNotFoundError as e:
                assert "numpy" in str(e)
                print("ok")
            else:
                raise AssertionError("expected ModuleNotFoundError")
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
