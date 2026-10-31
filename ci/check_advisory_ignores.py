#!/usr/bin/env python3
"""Fail if an advisory ignore in deny.toml has outlived its reason.

We ignore a RustSec advisory when its fix cannot land yet, most often because
the fixed version is younger than the 48 hours ci/check_dependency_age.py
requires. Nothing used to bring us back once the fix could land: #9985 ignored
four rustls advisories until the fix was 48 hours old, and rustls then stayed
on the vulnerable version for a week. So every ignore's `reason` must start
with one of two markers:

    expires YYYY-MM-DD: <why>   the ignore should be gone after this day (UTC)
    no fix: <why>               there is nothing to upgrade to

Once an `expires` day has passed, this check fails until someone upgrades the
crate and removes the ignore, or moves the date and says why it still holds.
An ignore that no longer matches any crate fails cargo-deny itself, through
`unused-ignored-advisory = "deny"`.
"""

import argparse
import re
import sys
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path

DEFAULT_CONFIG = "deny.toml"
NO_FIX = "no fix:"
EXPIRES = re.compile(r"expires ([^:]*):")
# date.fromisoformat also takes forms like 20261029, so pin the layout first.
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
RUSTSEC_ID = re.compile(r"RUSTSEC-\d{4}-\d{4}")


def parse_date(text):
    if DATE.fullmatch(text):
        try:
            return date.fromisoformat(text)
        except ValueError:
            pass
    raise ValueError(f"has `{text}` where a YYYY-MM-DD date belongs")


def parse_reason(reason):
    """Return the day an ignore expires, or None for an ignore with no fix.

    Raises ValueError, saying what is wrong, if the reason has no marker, has a
    malformed date, or says nothing after its marker.
    """
    if not reason.strip():
        raise ValueError("has no reason")
    if reason.startswith(NO_FIX):
        expires, why = None, reason[len(NO_FIX) :]
    else:
        match = EXPIRES.match(reason)
        if match is None:
            raise ValueError(
                f"has a reason that does not start with `expires YYYY-MM-DD:` or `{NO_FIX}`"
            )
        expires, why = parse_date(match[1]), reason[match.end() :]
    if not why.strip():
        raise ValueError("has a reason that says nothing after its marker")
    return expires


def load_ignores(path):
    config = tomllib.loads(Path(path).read_text())
    return config.get("advisories", {}).get("ignore", [])


def check_config(path, today):
    """Print every ignore in a cargo-deny config and return whether all are valid."""
    passed = True
    for entry in load_ignores(path):
        # A bare string is an advisory id or crate spec with no reason at all.
        if isinstance(entry, str):
            entry = {"id": entry}
        name = entry.get("id") or entry.get("crate") or str(entry)
        reason = entry.get("reason", "")
        try:
            expires = parse_reason(reason)
        except ValueError as error:
            problem = str(error)
        else:
            if expires is None or expires >= today:
                print(f"{path} ignores {name}: {reason}")
                continue
            problem = "has expired"
        passed = False
        print(f"{path}: {name} {problem}")
        if reason:
            print(f"    reason: {reason}")
        if RUSTSEC_ID.fullmatch(name):
            print(f"    advisory: https://rustsec.org/advisories/{name}")
    return passed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    if check_config(args.config, datetime.now(timezone.utc).date()):
        return 0

    # Keep the findings above the advice when CI interleaves the two streams.
    sys.stdout.flush()
    print(
        "\nEvery [advisories] ignore needs a reason that starts with "
        "`expires YYYY-MM-DD:` or `no fix:`. Use `expires` about a week out "
        "for a fix that cannot land yet, and `no fix` when there is nothing to "
        "upgrade to. Once an ignore expires, upgrade the crate and remove the "
        "ignore, or move the date and say why it still holds. To see the fixed "
        "versions and what pulls in the affected crate, remove the ignore and "
        "run `cargo deny check advisories`, or read the advisory.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
