# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import contextlib
import time
import traceback
from enum import Enum
from multiprocessing import Process, Queue, Value
from multiprocessing.reduction import ForkingPickler
from queue import Empty, Full
from typing import Callable, Iterable

from torch.utils.data import IterableDataset

from lance.log import LOGGER

_QUEUE_POLL_SECONDS = 0.1
_SHUTDOWN_TIMEOUT_SECONDS = 1.0


class _Message(Enum):
    DATA = "data"
    EPOCH_END = "epoch_end"
    ERROR = "error"
    SHUTDOWN = "shutdown"


def _send(queue: Queue, shutdown: Value, message: tuple[_Message, object]) -> bool:
    if shutdown.value:
        return False
    if message[0] is _Message.DATA:
        # Queue's feeder thread cannot propagate serialization failures to us.
        message = (_Message.DATA, bytes(ForkingPickler.dumps(message[1])))
    while not shutdown.value:
        try:
            queue.put(message, timeout=_QUEUE_POLL_SECONDS)
            return True
        except Full:  # noqa: PERF203 - Timed puts must retry until cancelled.
            continue
    return False


def _worker_ep(
    dataset_creator: Callable[[], IterableDataset],
    queue: Queue,
    shutdown: Value,
) -> None:
    try:
        if shutdown.value:
            return
        dataset = dataset_creator()
        while not shutdown.value:
            # Reuse the same dataset across epochs to preserve caches and state.
            for item in dataset:
                if not _send(queue, shutdown, (_Message.DATA, item)):
                    return
            if not _send(queue, shutdown, (_Message.EPOCH_END, None)):
                return
    except BaseException:
        if _send(queue, shutdown, (_Message.ERROR, traceback.format_exc())):
            # Tensor IPC handles remain valid only while their producer lives.
            # Keep it alive until the consumer receives the error and closes us.
            while not shutdown.value:
                time.sleep(_QUEUE_POLL_SECONDS)
    finally:
        if shutdown.value:
            try:
                queue.put_nowait((_Message.SHUTDOWN, None))
            except Full:
                pass
            # Cancellation discards prefetched data. Waiting for the feeder to
            # flush a full pipe would otherwise prevent the process from exiting.
            queue.cancel_join_thread()
        queue.close()


class AsyncDataset(IterableDataset):
    """Prefetch epochs in one persistent worker process.

    Worker exceptions and unexpected exits raise :class:`RuntimeError` in the
    consumer, including errors serializing a yielded item. After abandoning an
    epoch, the next iterator discards its remaining items through the epoch-end
    marker before yielding a fresh epoch from the same worker. Close a retained
    iterator before starting another; overlapping iterators are not supported.
    Use :meth:`close` or :func:`async_dataset` to stop prefetching when finished.
    Closing an unused dataset never starts a worker, and repeated calls to
    :meth:`close` are harmless.

    Parameters
    ----------
    dataset_creator : Callable[[], IterableDataset]
        Called once in the worker to create an iterable reused across epochs.
        Must support multiprocessing serialization with spawn or forkserver.
    queue_size : int, default 4
        Maximum number of queued data or control messages; zero is unbounded.
    """

    def __init__(
        self,
        dataset_creator: Callable[[], IterableDataset],
        *,
        queue_size: int = 4,
    ):
        self.dataset_creator = dataset_creator
        self.queue = Queue(maxsize=queue_size)
        self.started = False
        self._closed = False
        self._iterating = False
        self._epoch_pending = False

        # Only the parent writes this byte. A process lock could be left held by
        # a killed worker and prevent even the bounded shutdown from starting.
        self.shutdown = Value("b", False, lock=False)

    def _start(self):
        if self._closed:
            raise RuntimeError("AsyncDataset is closed")
        if self.started:
            return
        self.worker = Process(
            target=_worker_ep,
            args=(self.dataset_creator, self.queue, self.shutdown),
        )
        try:
            self.worker.start()
        except BaseException:
            self.worker.close()
            self.close()
            raise
        self.started = True
        # Only the worker writes. Keeping a writer open in the consumer prevents
        # EOF from interrupting a partial-frame read after the worker dies.
        self.queue._writer.close()

    def _receive(self) -> tuple[_Message, object]:
        while not self._closed:
            try:
                kind, payload = self.queue.get(timeout=_QUEUE_POLL_SECONDS)
                if kind is _Message.DATA:
                    payload = ForkingPickler.loads(payload)
            except Empty:
                if not self.worker.is_alive():
                    raise RuntimeError(
                        "AsyncDataset worker exited unexpectedly "
                        f"with exit code {self.worker.exitcode}"
                    ) from None
                continue
            except (EOFError, OSError) as error:
                self.worker.join(timeout=_QUEUE_POLL_SECONDS)
                exit_code = self.worker.exitcode
                status = (
                    f"exited unexpectedly with exit code {exit_code}"
                    if exit_code is not None
                    else "connection failed while still running"
                )
                raise RuntimeError(
                    f"AsyncDataset worker {status} during a transfer"
                ) from error
            if kind is _Message.ERROR:
                raise RuntimeError(f"AsyncDataset worker failed:\n{payload}")
            if kind is _Message.SHUTDOWN:
                raise RuntimeError(
                    "AsyncDataset worker shut down before epoch completion"
                )
            if kind not in (_Message.DATA, _Message.EPOCH_END):
                raise RuntimeError(f"Unexpected AsyncDataset message: {kind!r}")
            return kind, payload
        raise RuntimeError("AsyncDataset is closed")

    def __iter__(self):
        self._start()
        if self._iterating:
            raise RuntimeError(
                "An AsyncDataset iterator is already active; close it before "
                "starting another epoch"
            )
        self._iterating = True
        try:
            while self._epoch_pending and not self._closed:
                kind, _ = self._receive()
                if kind is _Message.EPOCH_END:
                    self._epoch_pending = False
            if self._closed:
                return
            self._epoch_pending = True
            while not self._closed:
                kind, payload = self._receive()
                if kind is _Message.EPOCH_END:
                    self._epoch_pending = False
                    return
                yield payload
        except GeneratorExit:
            raise
        except BaseException:
            self.close()
            raise
        finally:
            self._iterating = False

    def close(self):
        """Release queues and stop the worker, with bounded waits for shutdown.

        Prefetched items are discarded. A worker stuck in user code is terminated
        after a grace period, then killed if it still does not exit.
        """
        if self._closed:
            return
        self._closed = True
        self.shutdown.value = True

        try:
            if self.started:
                self.worker.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
                if self.worker.is_alive():
                    LOGGER.warning("AsyncDataset worker did not stop; terminating it")
                    self.worker.terminate()
                    self.worker.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
                if self.worker.is_alive():
                    LOGGER.warning(
                        "AsyncDataset worker ignored termination; killing it"
                    )
                    self.worker.kill()
                    self.worker.join(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
                if self.worker.is_alive():
                    raise RuntimeError(
                        "AsyncDataset worker did not exit after being killed"
                    )
                self.worker.close()
        finally:
            self.queue.close()
            self.queue.join_thread()
            # A consumer has no feeder thread to release its pipe handles.
            self.queue = None


@contextlib.contextmanager
def async_dataset(
    dataset_creator: Callable[[], IterableDataset],
    *,
    queue_size: int = 4,
) -> Iterable[AsyncDataset]:
    """Prefetch a dataset and release its worker when the context exits.

    Examples
    --------
    .. code-block:: python

        from torch.utils.data import IterableDataset
        from lance.torch.async_dataset import async_dataset

        class Rows(IterableDataset):
            def __iter__(self):
                yield from range(3)

        if __name__ == "__main__":
            with async_dataset(Rows) as dataset:
                assert list(dataset) == [0, 1, 2]
                assert next(iter(dataset)) == 0
                assert list(dataset) == [0, 1, 2]
    """
    ds = AsyncDataset(dataset_creator, queue_size=queue_size)
    try:
        yield ds
    finally:
        ds.close()
