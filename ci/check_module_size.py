#!/usr/bin/env python3
"""Flag Rust modules that a change grows past reviewable sizes.

GitHub stops syntax highlighting very large files in PR diffs, which makes
reviewing changes to them painful. This compares each `.rs` file changed
between a base and head commit and reports files the change makes larger:

* above the warn threshold (lines): a warning suggesting the module be split.
* above the fail threshold (bytes): an error; the module must be split before
  it grows. Files in the exceptions list are downgraded to a warning.

Changes that shrink a module are never flagged, so a PR that splits an
oversized module always passes. Entries in the exceptions list that are no
longer over the fail threshold are errors, so the list only shrinks.

Results are emitted as GitHub workflow annotations, plus an optional markdown
summary. Only git objects are read.
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass

WARN_LINES = 3000
FAIL_BYTES = 400_000
EXCEPTIONS_FILE = "ci/module_size_exceptions.txt"


@dataclass(frozen=True)
class Size:
    lines: int
    bytes: int


@dataclass(frozen=True)
class FileChange:
    path: str
    before: Size
    after: Size


@dataclass(frozen=True)
class Finding:
    change: FileChange
    failed: bool
    excepted: bool = False


def git(*args):
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def read_size(commit, path):
    """Return the size of `path` at `commit`, or None if it does not exist."""
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{commit}:{path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return Size(result.stdout.count(b"\n"), len(result.stdout))


def changed_rust_files(base, head):
    """Return a FileChange for each `.rs` file added, modified or renamed."""
    merge_base = git("merge-base", base, head).strip()
    output = git(
        "diff",
        "--name-status",
        "-z",
        "--find-renames",
        "--diff-filter=AMR",
        merge_base,
        head,
    )
    fields = output.split("\0")
    changes = []
    i = 0
    while i < len(fields) - 1:
        status = fields[i]
        if status.startswith("R"):
            old_path, new_path = fields[i + 1], fields[i + 2]
            i += 3
        else:
            old_path = new_path = fields[i + 1]
            i += 2
        if not new_path.endswith(".rs"):
            continue
        before = Size(0, 0) if status == "A" else read_size(merge_base, old_path)
        changes.append(FileChange(new_path, before, read_size(head, new_path)))
    return changes


def parse_exceptions(text):
    """Return {path: line_number} for each entry, skipping blanks and comments."""
    exceptions = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.split("#", 1)[0].strip()
        if line:
            exceptions[line] = number
    return exceptions


def evaluate(changes, exceptions=(), warn_lines=WARN_LINES, fail_bytes=FAIL_BYTES):
    findings = []
    for change in changes:
        if change.after.bytes <= change.before.bytes:
            continue
        if change.after.bytes > fail_bytes:
            excepted = change.path in exceptions
            findings.append(Finding(change, failed=not excepted, excepted=excepted))
        elif change.after.lines > warn_lines:
            findings.append(Finding(change, failed=False))
    return sorted(findings, key=lambda f: (not f.failed, -f.change.after.bytes))


def stale_exceptions(exceptions, sizes, fail_bytes=FAIL_BYTES):
    """Return exception paths that are missing or no longer over the limit."""
    return [
        path
        for path in exceptions
        if sizes.get(path) is None or sizes[path].bytes <= fail_bytes
    ]


def annotation(finding, warn_lines=WARN_LINES, fail_bytes=FAIL_BYTES):
    change = finding.change
    growth = (
        f"{change.path} grows from {change.before.lines:,} to "
        f"{change.after.lines:,} lines ({change.before.bytes / 1000:,.0f} KB to "
        f"{change.after.bytes / 1000:,.0f} KB)."
    )
    if finding.failed:
        return (
            "error",
            "Module too large",
            (
                f"{growth} Modules over {fail_bytes / 1000:,.0f} KB are not syntax "
                "highlighted in GitHub diffs. Split this module in a separate PR "
                "before making this change."
            ),
        )
    if finding.excepted:
        return (
            "warning",
            "Module too large (exception)",
            (
                f"{growth} This module is in {EXCEPTIONS_FILE} but is over "
                f"{fail_bytes / 1000:,.0f} KB; please split it before adding to it."
            ),
        )
    return (
        "warning",
        "Large module",
        (
            f"{growth} Consider splitting modules over {warn_lines:,} lines before "
            "adding to them."
        ),
    )


def render_summary(findings, stale):
    lines = ["### Rust module size check", ""]
    if findings:
        lines += [
            "| | Module | Lines before | Lines after | KB before | KB after |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for finding in findings:
            icon = ":x:" if finding.failed else ":warning:"
            change = finding.change
            lines.append(
                f"| {icon} | `{change.path}` | {change.before.lines:,} | "
                f"{change.after.lines:,} | {change.before.bytes / 1000:,.0f} | "
                f"{change.after.bytes / 1000:,.0f} |"
            )
        lines.append("")
    for path in stale:
        lines.append(f":x: Remove `{path}` from `{EXCEPTIONS_FILE}`.")
    if not findings and not stale:
        lines.append(":white_check_mark: No modules grew past the size limits.")
    return "\n".join(lines) + "\n"


def escape(value):
    """Escape a workflow command value (https://github.com/actions/toolkit/issues/193)."""
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="base commit the change is compared against")
    parser.add_argument("head", nargs="?", default="HEAD", help="head commit")
    parser.add_argument("--warn-lines", type=int, default=WARN_LINES)
    parser.add_argument("--fail-bytes", type=int, default=FAIL_BYTES)
    parser.add_argument("--summary", help="append a markdown summary to this file")
    args = parser.parse_args(argv)

    # Read the list from the same tree as the module sizes, not the working
    # tree, so the two cannot disagree when the checkout differs from `head`.
    exceptions = parse_exceptions(git("show", f"{args.head}:{EXCEPTIONS_FILE}"))

    changes = changed_rust_files(args.base, args.head)
    findings = evaluate(changes, exceptions, args.warn_lines, args.fail_bytes)
    for finding in findings:
        level, title, message = annotation(finding, args.warn_lines, args.fail_bytes)
        print(f"::{level} file={finding.change.path},title={title}::{escape(message)}")

    sizes = {path: read_size(args.head, path) for path in exceptions}
    stale = stale_exceptions(exceptions, sizes, args.fail_bytes)
    for path in stale:
        print(
            f"::error file={EXCEPTIONS_FILE},line={exceptions[path]},"
            f"title=Stale module size exception::{path} is missing or no longer "
            f"over {args.fail_bytes / 1000:,.0f} KB; remove it from the list."
        )

    if args.summary:
        with open(args.summary, "a") as f:
            f.write(render_summary(findings, stale))

    return 1 if stale or any(f.failed for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
