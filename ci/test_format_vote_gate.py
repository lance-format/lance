"""Unit tests for the format-spec vote gate logic.

Run with: pytest ci/test_format_vote_gate.py
"""

from datetime import datetime, timedelta, timezone

import pytest

from format_vote_gate import (
    PERIOD_HOURS,
    decide_verdict,
    tally_reviews,
    vote_opened_at,
    weekday_deadline,
)

HEAD = "sha_head"
PMC = {"alice", "bob", "carol", "dave"}


def is_pmc(login):
    return login is not None and login.lower() in PMC


def review(login, state, commit_id=HEAD):
    return {"login": login, "state": state, "commit_id": commit_id}


def test_counts_distinct_pmc_approvals_on_head_commit():
    approvals, stale, vetoes = tally_reviews(
        [
            review("alice", "APPROVED"),
            review("bob", "APPROVED"),
            review("carol", "APPROVED"),
        ],
        HEAD,
        "author",
        is_pmc,
    )
    assert sorted(approvals) == ["alice", "bob", "carol"]
    assert stale == []
    assert vetoes == []


def test_only_latest_review_per_member_counts():
    # Alice approved, then later requested changes -> she is a veto, not approval.
    approvals, _, vetoes = tally_reviews(
        [review("alice", "APPROVED"), review("alice", "CHANGES_REQUESTED")],
        HEAD,
        "author",
        is_pmc,
    )
    assert approvals == []
    assert vetoes == ["alice"]


def test_approvals_on_earlier_commit_are_stale():
    approvals, stale, _ = tally_reviews(
        [review("alice", "APPROVED", "old_sha"), review("bob", "APPROVED")],
        HEAD,
        "author",
        is_pmc,
    )
    assert approvals == ["bob"]
    assert stale == ["alice"]


def test_ignores_author_non_pmc_and_dismissed():
    approvals, _, vetoes = tally_reviews(
        [
            review("author", "APPROVED"),  # PR author, even if PMC, never counts
            review("eve", "APPROVED"),  # not on the PMC
            review("dave", "DISMISSED"),  # withdrawn
            review("carol", "COMMENTED"),  # a comment is not a vote
        ],
        HEAD,
        "author",
        is_pmc,
    )
    assert approvals == []
    assert vetoes == []


@pytest.mark.parametrize(
    ("veto_count", "approval_count", "period_elapsed", "expected"),
    [
        (1, 5, True, "veto"),  # veto wins even with enough approvals + elapsed
        (0, 2, True, "insufficient"),
        (0, 3, False, "waiting_period"),
        (0, 3, True, "pass"),
    ],
)
def test_decide_verdict_priority(veto_count, approval_count, period_elapsed, expected):
    assert decide_verdict(veto_count, approval_count, period_elapsed, 3) == expected


def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


# 2026-08-03 is a Monday, so this week runs Mon 03 .. Sun 09 August.
@pytest.mark.parametrize(
    ("opened", "expected"),
    [
        # Fully inside a work week: a plain 72-hour offset.
        ("2026-08-03T09:00", "2026-08-06T09:00"),
        # Opened Friday afternoon: 7h accrue before Saturday, the remaining 65h
        # resume Monday 00:00 and land Wednesday afternoon.
        ("2026-08-07T17:00", "2026-08-12T17:00"),
        # Opened during a weekend: the clock only starts on Monday.
        ("2026-08-08T12:00", "2026-08-13T00:00"),
        # Opened the instant a weekend ends.
        ("2026-08-10T00:00", "2026-08-13T00:00"),
        # The deadline itself lands exactly on the weekend boundary.
        ("2026-08-05T00:00", "2026-08-08T00:00"),
    ],
)
def test_weekday_deadline_excludes_weekends(opened, expected):
    assert weekday_deadline(utc(opened), PERIOD_HOURS) == utc(expected)


def test_weekday_deadline_spans_multiple_weekends():
    # A period longer than one work week has to skip more than one weekend.
    assert weekday_deadline(utc("2026-08-03T00:00"), 24 * 6) == utc("2026-08-11T00:00")


def test_weekday_deadline_converts_to_weekend_tz():
    # Late Friday in a UTC+X zone is already Saturday in UTC, so the clock waits.
    friday_evening_tokyo = utc("2026-08-08T01:00").astimezone(
        timezone(timedelta(hours=9))
    )
    assert weekday_deadline(friday_evening_tokyo, PERIOD_HOURS) == utc(
        "2026-08-13T00:00"
    )


def test_vote_opens_at_the_later_of_label_and_ready():
    labeled, ready = utc("2026-08-03T09:00"), utc("2026-08-04T09:00")
    assert vote_opened_at(labeled, ready) == ready
    assert vote_opened_at(ready, labeled) == ready


@pytest.mark.parametrize(
    ("labeled", "ready"),
    [
        (None, utc("2026-08-03T09:00")),  # not a format change (yet)
        (utc("2026-08-03T09:00"), None),  # still a draft
        (None, None),
    ],
)
def test_vote_does_not_open_until_both_conditions_hold(labeled, ready):
    assert vote_opened_at(labeled, ready) is None
