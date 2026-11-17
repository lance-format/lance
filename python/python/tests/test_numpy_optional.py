# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""numpy is an optional dependency: lance must import and work without it."""

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


def test_round_trip_without_numpy(tmp_path):
    # Write, read, take and vector search must all work with numpy blocked.
    uri = str(tmp_path / "ds.lance")
    result = _run_without_numpy(
        textwrap.dedent(
            f"""
            import pyarrow as pa
            import lance

            vec = pa.FixedSizeListArray.from_arrays(
                pa.array([0.0, 0.0, 1.0, 1.0, 5.0, 5.0], type=pa.float32()), 2
            )
            table = pa.table({{"id": [0, 1, 2], "vec": vec}})
            lance.write_dataset(table, {uri!r})
            ds = lance.dataset({uri!r})

            assert ds.to_table().num_rows == 3
            assert ds.take([0, 2]).column("id").to_pylist() == [0, 2]
            assert ds.scanner().take([1]).column("id").to_pylist() == [1]

            hits = ds.to_table(nearest={{"column": "vec", "q": [1.0, 1.0], "k": 1}})
            assert hits.column("id").to_pylist() == [1]
            print("ok")
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
