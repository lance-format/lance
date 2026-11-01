# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from compat import compat_decorator
from conftest import pytest_configure


@pytest.mark.parametrize("cache_root", [None, "custom_cache"])
def test_serial_torch_inductor_cache(monkeypatch, tmp_path, cache_root):
    if cache_root is None:
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    else:
        cache_root = str(tmp_path / cache_root)
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", cache_root)

    pytest_configure(Mock(spec=["addinivalue_line"]))

    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") == cache_root


@pytest.mark.parametrize("cache_root", [None, "", "custom_cache"])
def test_workers_have_separate_torch_inductor_caches(monkeypatch, tmp_path, cache_root):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(compat_decorator, "_SNAPSHOT", None)
    expected_root = tmp_path / (cache_root or "lance_torchinductor")
    worker_caches = []
    for worker_id in ("gw0", "gw1"):
        # Each worker inherits the controller's environment, not its sibling's.
        if cache_root is None:
            monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        else:
            monkeypatch.setenv(
                "TORCHINDUCTOR_CACHE_DIR",
                str(tmp_path / cache_root) if cache_root else "",
            )
        config = Mock(
            spec=["addinivalue_line", "workerinput"],
            workerinput={
                "workerid": worker_id,
                "lance_compat_versions": {"stable": [], "beta": None},
            },
        )

        pytest_configure(config)

        cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        assert cache_dir is not None
        cache = Path(cache_dir)
        assert cache.parent == expected_root
        worker_caches.append(cache)

    assert worker_caches[0] != worker_caches[1]
