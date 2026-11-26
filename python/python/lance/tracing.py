# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import atexit
from typing import Callable, Optional

from .lance import (
    TraceEvent,
    shutdown_tracing,
)
from .lance import (
    capture_trace_events as lance_capture_trace_events,
)
from .lance import (
    trace_to_chrome as lance_trace_to_chrome,
)


def trace_to_chrome(*, file: Optional[str] = None):
    """
    Begins tracing lance events to a chrome trace file.

    The trace file can be opened with chrome://tracing or with the Perfetto UI.

    The file will be finished (and closed) when the python process exits.

    The trace level is controlled by the `LANCE_LOG` environment variable.

    Parameters
    ----------
    file: str, optional
        The file to write the trace to. If None, then a file in the current
        directory will be created named ./trace-{unix epoch in micros}.json
    """
    guard = lance_trace_to_chrome(file)
    atexit.register(lambda: guard.finish_tracing())


def capture_trace_events(callback: Callable[[TraceEvent], None]):
    """
    Capture trace events and call the given callback with each event.

    When trace events occur they will be placed on a queue and a dedicated thread
    will call the callback with each event.  This prevents the callback from blocking
    the operation of the program.  This also means the callback may not be called
    immediately when the event occurs and so this method should be used for reporting
    and not synchronization or timing.

    Parameters
    ----------
    callback: Callable[[TraceEvent], None]
        The callback to call with each trace event.

    Raises
    ------
    RuntimeError
        If a callback is already registered. Call ``shutdown_tracing()`` before
        registering another one.
    """
    lance_capture_trace_events(callback)
    atexit.register(lambda: shutdown_tracing())
