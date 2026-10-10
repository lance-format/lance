#!/usr/bin/env python3
"""Report ordering-relevant dependency changes between two Cargo.lock files.

Lance inherits its comparison semantics from Arrow Rust (sort kernels,
comparators, the row encoding), from DataFusion (ScalarValue, SortExec and the
merge cursors) and from `half` (f16::total_cmp). A version change in any of
them is the moment to run the fixed ordering fixtures in
rust/lance-compat/tests/arrow_ordering.rs against the new code, so this
script answers "did one of those crates change?" for the CI job that does so.

Usage: arrow_upgrade.py BASE_LOCK HEAD_LOCK

Prints one line per changed crate. When run under GitHub Actions it also
writes `changed=true|false` to $GITHUB_OUTPUT and a review table, with links
to the upstream release notes, to $GITHUB_STEP_SUMMARY. The exit code is 0
whether or not anything changed; a parse failure exits non-zero.
"""

import os
import sys
import tomllib

ARROW_RELEASES = "https://github.com/apache/arrow-rs/releases/tag/"
DATAFUSION_RELEASES = "https://github.com/apache/datafusion/releases/tag/"
HALF_RELEASES = "https://github.com/VoidStarKat/half-rs/releases/tag/v"

FIXTURE_PATH = "rust/lance-compat/tests/arrow_ordering.rs"


def is_ordering_crate(name):
    """Whether a crate's version decides how Lance compares values.

    Lance's own `lance-arrow-*` crates wrap Arrow but do not define the order,
    so the prefix match is anchored at the start of the name.
    """
    return name in ("arrow", "datafusion", "half") or name.startswith(
        ("arrow-", "datafusion-")
    )


CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"


def ordering_crates(lock_text):
    """Map each ordering crate in a Cargo.lock to the set of (version, source) it resolves to.

    A lockfile can hold several versions of one crate when two dependencies
    disagree, so the value is a set rather than a single entry. The source is
    part of the identity because a git or patched dependency can change code
    without changing its version: `arrow-ord 58.4.0` at one commit and at
    another are different code. A crates.io source is recorded as `None`.
    """
    versions = {}
    for package in tomllib.loads(lock_text).get("package", []):
        name = package["name"]
        if is_ordering_crate(name):
            source = package.get("source")
            if source == CRATES_IO:
                source = None
            versions.setdefault(name, set()).add((package["version"], source))
    return versions


def diff(base, head):
    """Crates whose resolved version set differs, as (name, base_versions, head_versions).

    Sorted by name so the output is stable across runs.
    """
    changes = []
    for name in sorted(set(base) | set(head)):
        before = base.get(name, set())
        after = head.get(name, set())
        if before != after:
            changes.append((name, before, after))
    return changes


def release_notes_url(name, version):
    if name.startswith("arrow"):
        return ARROW_RELEASES + version
    if name.startswith("datafusion"):
        return DATAFUSION_RELEASES + version
    return HALF_RELEASES + version


def format_versions(versions):
    """Render a set of (version, source) entries; crates.io entries show the version alone."""
    if not versions:
        return "(absent)"
    return ", ".join(
        version if source is None else f"{version} from {source}"
        for version, source in sorted(
            versions, key=lambda entry: (entry[0], entry[1] or "")
        )
    )


def summary_markdown(changes):
    lines = [
        "## Arrow ordering compatibility",
        "",
        "Ordering-relevant crates changed in `Cargo.lock`. The `Arrow ordering",
        f"compatibility` job runs the fixed fixtures in `{FIXTURE_PATH}` against",
        "the new versions, first the base branch's copy where one exists and then",
        "this PR's; its step logs hold the outcome.",
        "",
        "| Crate | Base | Head | Release notes |",
        "|---|---|---|---|",
    ]
    for name, before, after in changes:
        links = " ".join(
            f"[{version}]({release_notes_url(name, version)})"
            for version in sorted({version for version, _ in after - before})
        )
        lines.append(
            f"| `{name}` | {format_versions(before)} | {format_versions(after)} | {links} |"
        )
    lines += [
        "",
        "Before merging, record in the PR description:",
        "",
        "1. Any sorting, comparison or row-encoding change in the release notes above,",
        "   and whether it is an upstream contract change, an upstream bug fix, or a",
        "   change in how Lance calls the API.",
        "2. Which Lance behaviours depend on it (see the table at the top of the fixture file).",
        "3. Whether persisted btree pages, zone map or bloom filter extrema need",
        "   compatibility handling.",
        "",
        "If the fixtures failed, that is a finding, not an expectation to refresh.",
        "Once reviewed, apply the `arrow-ordering-reviewed` label to let the job pass",
        "with the updated fixtures.",
    ]
    return "\n".join(lines) + "\n"


def main(argv):
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[1], "rb") as f:
        base = ordering_crates(f.read().decode())
    with open(argv[2], "rb") as f:
        head = ordering_crates(f.read().decode())
    changes = diff(base, head)

    for name, before, after in changes:
        print(f"{name}: {format_versions(before)} -> {format_versions(after)}")
    if not changes:
        print("no ordering-relevant dependency changes")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as f:
            f.write(f"changed={'true' if changes else 'false'}\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary and changes:
        with open(summary, "a") as f:
            f.write(summary_markdown(changes))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
