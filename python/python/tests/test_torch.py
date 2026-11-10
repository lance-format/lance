# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import contextlib
import multiprocessing
import os
import pickle
import signal
from functools import partial
from unittest.mock import patch

import lance
import numpy as np
import psutil
import pyarrow as pa
import pytest

torch = pytest.importorskip("torch")

import lance.torch.async_dataset as async_module  # noqa: E402
from lance.torch.async_dataset import AsyncDataset, async_dataset  # noqa: E402
from lance.torch.data import SafeLanceDataset, get_safe_loader  # noqa: E402


@pytest.fixture(scope="module")
def temp_lance_dataset(tmp_path_factory):
    """Create temporary Lance dataset for testing"""
    test_dir = tmp_path_factory.mktemp("lance_data")
    dataset_path = test_dir / "test_dataset.lance"

    # Generate test data with batch_size aligned sample count
    num_samples = 96  # 16 samples/batch * 6 batches
    data = pa.table(
        {
            "id": range(num_samples),
            "embedding": [
                np.random.rand(128).astype(np.float32).tobytes()
                for _ in range(num_samples)
            ],
        }
    )

    lance.write_dataset(data, dataset_path)
    yield str(dataset_path)


def test_dataset_initialization(temp_lance_dataset):
    """Verify dataset basic functionality"""
    ds = SafeLanceDataset(temp_lance_dataset)

    # Validate metadata
    assert len(ds) == 96, "Sample count should match configured size"

    # Validate single sample format
    sample = ds[0]
    assert isinstance(sample, dict), "Sample should be dictionary type"
    assert {"id", "embedding"}.issubset(sample.keys()), "Missing required fields"


def test_multiprocess_loading(temp_lance_dataset, capsys):
    """Verify multi-worker data loading"""
    dataset = SafeLanceDataset(temp_lance_dataset)
    loader = get_safe_loader(
        dataset,
        num_workers=2,
        batch_size=16,
        drop_last=False,  # Ensure full batches
    )

    total_samples = 0
    for batch in loader:
        assert batch["id"].shape == (16,), "Batch dimension mismatch"
        total_samples += batch["id"].shape[0]

    # Validate complete dataset loading
    assert total_samples == 96, "Should load all samples"


def _creator_failure():
    raise ValueError("creator exploded")


def _abrupt_exit():
    os._exit(17)


def _empty_dataset():
    return []


class _FailureDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        yield torch.tensor([17])
        raise ValueError("iteration exploded")


class _EpochDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        yield None
        yield {"epoch": self.epoch, "pid": os.getpid(), "tensor": torch.tensor([1, 2])}


class _FullQueueDataset(torch.utils.data.IterableDataset):
    def __init__(self, ready, cancelled, finish_cleanup):
        self.ready = ready
        self.cancelled = cancelled
        self.finish_cleanup = finish_cleanup

    def __iter__(self):
        try:
            # The first item fills both the queue slot and its feeder's pipe.
            yield bytes(1024 * 1024)
            self.ready.set()
            yield 2
        finally:
            self.cancelled.set()
            self.finish_cleanup.wait()


class _LargeItemDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        yield bytes(16 * 1024 * 1024)


class _BlockedDataset(torch.utils.data.IterableDataset):
    def __init__(self, ready, release, ignore_termination=False):
        self.ready = ready
        self.release = release
        self.ignore_termination = ignore_termination

    def __iter__(self):
        if self.ignore_termination:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        self.ready.set()
        self.release.wait()
        yield 1


def _async_lifecycle_scenario(scenario, start_method):
    multiprocessing.set_start_method(start_method, force=True)
    # Exercise escalation without spending a production grace period per case.
    async_module._SHUTDOWN_TIMEOUT_SECONDS = (
        0.05 if scenario in ("blocked_user_code", "ignored_termination") else 0.5
    )
    async_module._QUEUE_POLL_SECONDS = 0.01
    if scenario == "unused_context":
        with async_dataset(_creator_failure) as dataset:
            assert not dataset.started
        assert not dataset.started
        dataset.close()
    elif scenario == "start_failure":
        dataset = AsyncDataset(lambda: [])
        with pytest.raises((AttributeError, pickle.PicklingError), match="pickle"):
            next(iter(dataset))
        assert not dataset.started
        dataset.close()
    elif scenario in ("creator_failure", "iteration_failure", "abrupt_exit"):
        creator, message = {
            "creator_failure": (_creator_failure, "ValueError: creator exploded"),
            "iteration_failure": (_FailureDataset, "ValueError: iteration exploded"),
            "abrupt_exit": (_abrupt_exit, "exited unexpectedly with exit code 17"),
        }[scenario]
        with async_dataset(creator, queue_size=1) as dataset:
            batches = iter(dataset)
            if scenario == "iteration_failure":
                assert torch.equal(next(batches), torch.tensor([17]))
            with pytest.raises(RuntimeError, match=message) as error:
                next(batches)
            if scenario != "abrupt_exit":
                assert "Traceback" in str(error.value)
            dataset.close()
    elif scenario in ("early_exit", "iterator_close", "repeated_close"):
        with async_dataset(_EpochDataset, queue_size=1) as dataset:
            batches = iter(dataset)
            assert next(batches) is None
            if scenario == "iterator_close":
                batches.close()
                with pytest.raises(RuntimeError, match="AsyncDataset is closed"):
                    next(iter(dataset))
            elif scenario == "repeated_close":
                dataset.close()
                dataset.close()
            # Leave the epoch partially consumed when the context exits.
        batches.close()
    elif scenario in ("partial_transfer", "large_transfer"):
        with async_dataset(_LargeItemDataset, queue_size=1) as dataset:
            if scenario == "partial_transfer":
                dataset._start()
                # With no reader draining it, this frame cannot fit in the pipe.
                assert dataset.queue._reader.poll(5), "Worker never began the transfer"
                dataset.worker.kill()
                dataset.worker.join(timeout=5)
                assert not dataset.worker.is_alive()
                message = f"exit code {dataset.worker.exitcode}"
                with pytest.raises(RuntimeError, match=message):
                    next(iter(dataset))
            else:
                batches = iter(dataset)
                assert next(batches) == bytes(16 * 1024 * 1024)
                with pytest.raises(StopIteration):
                    next(batches)
    elif scenario in ("blocked_producer", "blocked_user_code", "ignored_termination"):
        ready = multiprocessing.Event()
        if scenario == "blocked_producer":
            cancelled = multiprocessing.Event()
            finish_cleanup = multiprocessing.Event()
            creator = partial(_FullQueueDataset, ready, cancelled, finish_cleanup)
        else:
            creator = partial(
                _BlockedDataset,
                ready,
                multiprocessing.Event(),
                scenario == "ignored_termination",
            )
        dataset = AsyncDataset(creator, queue_size=1)
        try:
            dataset._start()
            assert ready.wait(5), "Worker never reached the blocking operation"
            if scenario == "blocked_producer":
                # Prove cancellation while cleanup is deliberately held open.
                # Process teardown may exceed close()'s grace period under load.
                dataset.shutdown.value = True
                assert cancelled.wait(5), (
                    "Full queue prevented cooperative cancellation"
                )
                assert dataset.worker.is_alive()
                finish_cleanup.set()
                dataset.worker.join(timeout=5)
                assert not dataset.worker.is_alive(), "Worker did not finish cleanup"
                assert dataset.worker.exitcode == 0
            with (
                patch.object(
                    dataset.worker, "terminate", wraps=dataset.worker.terminate
                ) as terminate,
                patch.object(dataset.worker, "kill", wraps=dataset.worker.kill) as kill,
            ):
                dataset.close()
                assert terminate.call_count == int(scenario != "blocked_producer")
                assert kill.call_count == int(scenario == "ignored_termination")
        finally:
            if scenario == "blocked_producer":
                finish_cleanup.set()
            dataset.close()
        dataset.close()
    elif scenario in ("repeated_epochs", "empty_epochs"):
        creator = _EpochDataset if scenario == "repeated_epochs" else _empty_dataset
        with async_dataset(creator, queue_size=1) as dataset:
            for epoch in range(1, 4):
                batches = list(dataset)
                if scenario == "empty_epochs":
                    assert batches == []
                else:
                    assert batches[0] is None
                    assert len(batches) == 2
                    assert batches[1]["epoch"] == epoch
                    assert batches[1]["pid"] == dataset.worker.pid
                    assert torch.equal(batches[1]["tensor"], torch.tensor([1, 2]))
    else:
        raise AssertionError(f"Unknown lifecycle scenario: {scenario}")
    assert dataset.shutdown.value, "Closing AsyncDataset did not signal cancellation"
    assert not multiprocessing.active_children(), "AsyncDataset leaked a worker"


@pytest.mark.parametrize(
    "scenario,start_method",
    [
        (scenario, "spawn")
        for scenario in (
            "unused_context",
            "start_failure",
            "creator_failure",
            "iteration_failure",
            "abrupt_exit",
            "partial_transfer",
            "large_transfer",
            "early_exit",
            "iterator_close",
            "repeated_close",
            "blocked_producer",
            "blocked_user_code",
            "repeated_epochs",
            "empty_epochs",
        )
    ]
    + ([("ignored_termination", "spawn")] if os.name == "posix" else [])
    + (
        [
            (scenario, "forkserver")
            for scenario in ("repeated_epochs", "partial_transfer", "large_transfer")
        ]
        if "forkserver" in multiprocessing.get_all_start_methods()
        else []
    ),
)
def test_async_dataset_lifecycle(scenario, start_method):
    # A supervisor puts a deadline around the consumer as well as its worker.
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_async_lifecycle_scenario, args=(scenario, start_method)
    )
    process.start()
    try:
        process.join(timeout=15)
        assert not process.is_alive(), (
            f"AsyncDataset hung in {scenario} ({start_method})"
        )
        assert process.exitcode == 0, (
            f"AsyncDataset failed in {scenario} ({start_method})"
        )
    finally:
        if process.is_alive():
            for child in psutil.Process(process.pid).children(recursive=True):
                with contextlib.suppress(psutil.NoSuchProcess):
                    child.kill()
            process.kill()
            process.join(timeout=5)
        process.close()
