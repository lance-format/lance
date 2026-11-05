# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset cleanup retention policies."""

from __future__ import annotations

import json
from abc import ABC
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence


class CleanupRule(ABC):
    """A cleanup retention rule: KeepOne samples a band; RemoveAll expires it."""


@dataclass(frozen=True)
class KeepOne(CleanupRule):
    """Thin history older than a full-history window, keeping one commit per interval.

    Use in ``cleanup_old_versions(policy=CleanupPolicy(rules=[...]))``.
    ``per_interval`` must be positive. ``older_than`` defaults to two days and
    must be at least two days, measured from the hinted commit timestamp (or
    the dataset snapshot without a hint). Validation occurs during cleanup.
    Cleanup's separate ``older_than`` argument expires all older versions,
    including sampled commits. For example::

        dataset.cleanup_old_versions(
            older_than=timedelta(days=7),
            keep_one=KeepOne(timedelta(days=1), timedelta(days=2)),
        )
    """

    per_interval: timedelta
    older_than: timedelta = timedelta(days=2)


@dataclass(frozen=True)
class RemoveAll(CleanupRule):
    """Expire all versions older than a wall-clock age, including sampled commits.

    Use as the final rule in a ``CleanupPolicy``. For example,
    ``RemoveAll(older_than=timedelta(days=30))`` expires thirty-day-old history.
    Current versions, tags and branch dependencies remain protected.
    """

    older_than: timedelta


@dataclass(frozen=True)
class CleanupPolicy:
    """Ordered CleanupRule entries, with an optional final RemoveAll rule.

    By default, contains only ``RemoveAll(timedelta(days=14))``, preserving
    the existing two-week expiration behavior. For example::

        dataset.cleanup_old_versions(policy=CleanupPolicy(rules=[
            KeepOne(timedelta(hours=1), timedelta(days=2)),
            KeepOne(timedelta(days=1), timedelta(days=7)),
            RemoveAll(timedelta(days=30)),
        ]))
    """

    rules: Sequence[CleanupRule] = (RemoveAll(timedelta(days=14)),)


def _cleanup_duration_string(value: timedelta) -> str:
    seconds = value.days * 86_400 + value.seconds
    return f"{seconds}s {value.microseconds}us"


def _cleanup_policy_to_json(policy: CleanupPolicy) -> str:
    if not isinstance(policy, CleanupPolicy):
        raise TypeError(f"policy must be a CleanupPolicy, got {type(policy)}")
    rules = []
    for rule in policy.rules:
        if isinstance(rule, KeepOne):
            rules.append(
                {
                    "type": "keep_one",
                    "per_interval": _cleanup_duration_string(rule.per_interval),
                    "older_than": _cleanup_duration_string(rule.older_than),
                }
            )
        elif isinstance(rule, RemoveAll):
            rules.append(
                {
                    "type": "remove_all",
                    "older_than": _cleanup_duration_string(rule.older_than),
                }
            )
        else:
            raise TypeError(
                f"policy entries must be KeepOne or RemoveAll, got {type(rule)}"
            )
    return json.dumps({"rules": rules})
