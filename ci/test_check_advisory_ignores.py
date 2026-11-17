"""Unit tests for the deny.toml advisory ignore check.

Run with: pytest ci/test_check_advisory_ignores.py
"""

from datetime import date

import pytest

from check_advisory_ignores import check_config, main, parse_reason

TODAY = date(2026, 10, 21)


def deny_toml(tmp_path, entries):
    """Write a deny.toml ignoring `entries`, or with no [advisories] if None."""
    path = tmp_path / "deny.toml"
    if entries is None:
        lines = ["[bans]", 'multiple-versions = "warn"']
    else:
        lines = ["[advisories]", "ignore = ["]
        lines += [f"    {entry}," for entry in entries]
        lines.append("]")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        pytest.param("no fix: unmaintained", None, id="no_fix"),
        pytest.param("expires 2026-10-29: too young", date(2026, 10, 29), id="expires"),
        pytest.param("expires 2026-10-29: a: b", date(2026, 10, 29), id="colon_in_why"),
    ],
)
def test_parse_reason_accepts_a_marker(reason, expected):
    assert parse_reason(reason) == expected


@pytest.mark.parametrize(
    ("reason", "problem"),
    [
        pytest.param("", "has no reason", id="empty"),
        pytest.param("until 48 hours old", "does not start with", id="no_marker"),
        pytest.param("No fix: unmaintained", "does not start with", id="marker_case"),
        pytest.param("expires: 2026-10-29", "does not start with", id="colon_first"),
        pytest.param("expires 2026-13-01: x", "`2026-13-01`", id="month_13"),
        pytest.param("expires 2026-10-1: x", "`2026-10-1`", id="short_day"),
        pytest.param("expires 20261029: x", "`20261029`", id="basic_iso"),
        pytest.param("expires next week: x", "`next week`", id="words"),
        pytest.param("expires 2026-10-29 x: y", "`2026-10-29 x`", id="text_in_date"),
        pytest.param("no fix:", "says nothing after", id="no_fix_without_why"),
        pytest.param("expires 2026-10-29: ", "says nothing", id="expires_without_why"),
    ],
)
def test_parse_reason_rejects(reason, problem):
    with pytest.raises(ValueError, match=problem):
        parse_reason(reason)


@pytest.mark.parametrize(
    ("entries", "passes", "printed"),
    [
        pytest.param(
            ['{ id = "RUSTSEC-2026-0001", reason = "expires 2026-10-28: young" }'],
            True,
            ["ignores RUSTSEC-2026-0001: expires 2026-10-28: young"],
            id="expires_later",
        ),
        pytest.param(
            ['{ id = "RUSTSEC-2026-0001", reason = "expires 2026-10-21: young" }'],
            True,
            ["ignores RUSTSEC-2026-0001"],
            id="expires_today",
        ),
        pytest.param(
            ['{ id = "RUSTSEC-2026-0001", reason = "expires 2026-10-20: young" }'],
            False,
            [
                "RUSTSEC-2026-0001 has expired\n",
                "reason: expires 2026-10-20: young\n",
                "advisory: https://rustsec.org/advisories/RUSTSEC-2026-0001\n",
            ],
            id="expired_yesterday",
        ),
        pytest.param(
            ['{ id = "RUSTSEC-2026-0001", reason = "no fix: unmaintained" }'],
            True,
            ["ignores RUSTSEC-2026-0001: no fix: unmaintained"],
            id="no_fix",
        ),
        pytest.param(
            ['"RUSTSEC-2026-0001"'],
            False,
            ["RUSTSEC-2026-0001 has no reason"],
            id="bare_id",
        ),
        pytest.param(
            ['{ id = "RUSTSEC-2026-0001" }'],
            False,
            ["RUSTSEC-2026-0001 has no reason"],
            id="no_reason",
        ),
        pytest.param(
            ['{ crate = "demo@0.1.0", reason = "yanked upstream" }'],
            False,
            ["demo@0.1.0 has a reason that does not start with"],
            id="yanked_crate",
        ),
        pytest.param(
            [
                '{ id = "RUSTSEC-2026-0001", reason = "no fix: unmaintained" }',
                '{ id = "RUSTSEC-2026-0002", reason = "until 48 hours old" }',
                '{ id = "RUSTSEC-2026-0003", reason = "expires 2026-10-01: young" }',
            ],
            False,
            [
                "ignores RUSTSEC-2026-0001",
                "RUSTSEC-2026-0002 has a reason that does not start with",
                "RUSTSEC-2026-0003 has expired",
            ],
            id="reports_every_entry",
        ),
        pytest.param([], True, [], id="no_ignores"),
        pytest.param(None, True, [], id="no_advisories_table"),
    ],
)
def test_check_config(tmp_path, capsys, entries, passes, printed):
    assert check_config(deny_toml(tmp_path, entries), TODAY) is passes
    out = capsys.readouterr().out
    for line in printed:
        assert line in out


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        pytest.param("expires 2999-01-01: young", 0, id="future"),
        pytest.param("expires 2000-01-01: young", 1, id="past"),
    ],
)
def test_main_tells_how_to_fix_a_failure(tmp_path, capsys, reason, code):
    path = deny_toml(tmp_path, [f'{{ id = "RUSTSEC-2026-0001", reason = "{reason}" }}'])
    assert main([str(path)]) == code
    err = capsys.readouterr().err
    if code:
        assert "remove the ignore, or move the date" in err
        assert "cargo deny check advisories" in err
    else:
        assert err == ""
