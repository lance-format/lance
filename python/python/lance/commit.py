# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from contextlib import AbstractContextManager
from typing import Callable

CommitLock = Callable[[int], AbstractContextManager]


class CommitConflictError(OSError):
    """A commit conflicted with a concurrent transaction.

    Subclasses :class:`OSError` so existing ``except OSError`` handlers keep
    working. ``retryable`` is ``True`` when the transaction was preempted and
    can be retried against the newer version, ``False`` when the conflict is
    incompatible and retrying will not help.
    """

    def __init__(self, message: str = "", retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable
