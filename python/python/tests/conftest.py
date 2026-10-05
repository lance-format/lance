# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import sys
from typing import Optional

import pytest


class ProgressRecorder:
    """Reusable progress callback recorder for index build tests."""

    def __init__(
        self,
        fail_after: Optional[int] = None,
        fail_on_tag: Optional[str] = None,
    ):
        self.events = []
        self.fail_after = fail_after
        self.fail_on_tag = fail_on_tag

    def __call__(self, event):
        self.events.append(event)
        event_tag = f"{event.event}:{event.stage}"
        if self.fail_on_tag is not None and event_tag == self.fail_on_tag:
            raise RuntimeError("progress callback failure")
        if self.fail_after is not None and len(self.events) >= self.fail_after:
            raise RuntimeError("progress callback failure")


def progress_event_tags(events):
    return [f"{event.event}:{event.stage}" for event in events]


def stage_progress_values(events, stage):
    return [
        event.completed
        for event in events
        if event.event == "progress"
        and event.stage == stage
        and event.completed is not None
    ]


@pytest.fixture(params=(True, False))
def provide_pandas(request, monkeypatch):
    if not request.param:
        monkeypatch.setitem(sys.modules, "pd", None)
    return request.param


def disable_items_with_mark(items, mark, reason):
    skipper = pytest.mark.skip(reason=reason)
    for item in items:
        if mark in item.keywords:
            item.add_marker(skipper)


# These are initialization hooks and must have an exact name for pytest to pick them up
# https://docs.pytest.org/en/7.1.x/reference/reference.html


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires S3 buckets to be setup with access)",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--run-forward",
        action="store_true",
        default=False,
        help="Run forward compatibility tests (requires files to be generated already)",
    )
    parser.addoption(
        "--run-compat",
        action="store_true",
        default=False,
        help="Run upgrade/downgrade compatibility tests (creates virtual environments)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "forward: mark tests that require forward compatibility datagen files",
    )
    config.addinivalue_line(
        "markers", "integration: mark test that requires object storage integration"
    )
    config.addinivalue_line(
        "markers", "slow: mark tests that require large CPU or RAM resources"
    )
    config.addinivalue_line(
        "markers",
        "compat: mark tests that run upgrade/downgrade compatibility checks",
    )

    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        from compat.compat_decorator import use_version_snapshot

        use_version_snapshot(workerinput["lance_compat_versions"])


def pytest_configure_node(node):
    """Resolve the compat version list once, on the xdist controller.

    The compat tests are parametrized over the pylance releases published to
    PyPI and fury.io, and collecting `python/tests` imports them whether or not
    --run-compat was passed. Left to itself every worker queries for that list
    while collecting, so a request that times out or a release that lands
    mid-run gives one worker a different parameter set, and xdist aborts the
    whole run with "Different tests were collected".
    """
    from compat.compat_decorator import version_snapshot

    node.workerinput["lance_compat_versions"] = version_snapshot()


# tryfirst because xdist reads xdist_group off each item to build its scheduling
# groups before ordinary pytest_collection_modifyitems hooks run; a mark added
# later is silently ignored rather than rejected.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # The lindera fixture unzips a dictionary into the checked-out models tree and
    # removes it again on teardown, and the tokenizer configs name that path
    # relative to the repo, so it cannot be relocated per worker. Pinning these
    # tests to one xdist worker keeps a teardown in one worker from deleting the
    # dictionary another is still reading. Without -n it changes nothing.
    for item in items:
        if "lindera_ipadic" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.xdist_group("lindera"))
    if not config.getoption("--run-integration"):
        disable_items_with_mark(items, "integration", "--run-integration not specified")
    if not config.getoption("--run-slow"):
        disable_items_with_mark(items, "slow", "--run-slow not specified")
    if not config.getoption("--run-forward"):
        disable_items_with_mark(items, "forward", "--run-forward not specified")
    if not config.getoption("--run-compat"):
        disable_items_with_mark(items, "compat", "--run-compat not specified")
    try:
        import torch

        # torch.cuda.is_available will return True on some CI machines even though any
        # attempt to use CUDA will then fail.  torch.cuda.device_count seems to be more
        # reliable
        if (
            torch.backends.cuda.is_built()
            and not torch.cuda.is_available
            or torch.cuda.device_count() <= 0
        ):
            disable_items_with_mark(
                items, "cuda", "torch is installed but cuda is not available"
            )
            if (
                not torch.backends.mps.is_available()
                or not torch.backends.mps.is_built()
            ):
                disable_items_with_mark(
                    items, "gpu", "torch is installed but no gpu is available"
                )
    except ImportError as err:
        reason = f"torch not installed ({err})"
        disable_items_with_mark(items, "torch", reason)
        disable_items_with_mark(items, "cuda", reason)
        disable_items_with_mark(items, "gpu", reason)
