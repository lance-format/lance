#!/usr/bin/env python3
"""Collect the raw signals a Lance stable-release readiness review needs.

This script only gathers facts; it makes no go/no-go judgement. See the
`lance-stable-release-prep` SKILL.md for how to read each section.

Requires `git` and an authenticated `gh`. Standard library only.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# Files that hold a written-by-default version. A bump here changes what new
# datasets and new indices look like on disk, which is what the three-month
# reader rule governs.
CONSTANT_PATHS = [
    "rust/lance-file/src",
    "rust/lance-index/src",
    "rust/lance-geo/src",
    "rust/lance-table/src",
]
DEFAULT_SELECTOR_FILES = [
    "rust/lance-file/src/version.rs",
    "rust/lance-index/src/registry.rs",
    "rust/lance-index/src/lib.rs",
]

CONSTANT_RE = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?const\s+"
    r"([A-Z][A-Z0-9_]*)\s*:\s*[^=]+=\s*([^;]+);"
)
# `VERSIONS` (plural) is a count, not a version; `*_KEY` names hold metadata keys.
VERSION_NAME_RE = re.compile(r"VERSION(?!S)|CURRENT")
NOT_A_VERSION_RE = re.compile(r"_KEY$")
PR_NUMBER_RE = re.compile(r"\(#(\d+)\)\s*$")

STABLE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def run(args, check=True):
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(f"command failed: {' '.join(args)}\n{result.stderr.strip()}")
    return result.stdout


def heading(text):
    print(f"\n## {text}\n")


def detect_repo():
    url = run(["git", "remote", "get-url", "origin"], check=False).strip()
    match = re.search(r"github\.com[:/]+([^/]+/[^/.]+)", url)
    return match.group(1) if match else "lance-format/lance"


def stable_tags():
    """Every `vX.Y.Z` tag with its commit date, oldest first."""
    raw = run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:short)\t%(creatordate:iso-strict)",
            "refs/tags/v*",
        ]
    )
    tags = []
    for line in raw.splitlines():
        name, _, date = line.partition("\t")
        match = STABLE_TAG_RE.match(name)
        if match:
            tags.append((name, tuple(int(p) for p in match.groups()), date))
    tags.sort(key=lambda entry: entry[2])
    return tags


def tag_date(tag):
    return run(["git", "log", "-1", "--format=%cI", tag]).strip()


def days_since(iso):
    when = datetime.fromisoformat(iso)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - when).days


def workspace_version():
    with open("Cargo.toml", encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r'^version\s*=\s*"([^"]+)"', line)
            if match:
                return match.group(1)
    return "unknown"


def section_release(baseline, baseline_date, head):
    heading("Release under preparation")
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    print(f"- Workspace version (`Cargo.toml`): `{workspace_version()}`")
    print(f"- Branch: `{branch}`")
    print(f"- Head: `{run(['git', 'rev-parse', '--short', head]).strip()}`")
    print(f"- Baseline (last stable release): `{baseline}`")
    print(
        f"- Baseline released: {baseline_date} ({days_since(baseline_date)} days ago)"
    )
    recent = stable_tags()[-6:]
    print("\nRecent stable releases (oldest first):\n")
    for name, _, date in recent:
        print(f"  - `{name}` — {date[:10]}")


def gh_json(args):
    raw = run(["gh", *args])
    return json.loads(raw) if raw.strip() else []


def section_issues(repo, baseline, baseline_date, issue_limit):
    heading(f"Issues opened since {baseline} ({baseline_date[:10]})")
    issues = gh_json(
        [
            "issue",
            "list",
            "--repo",
            repo,
            "--state",
            "all",
            "--search",
            f"created:>={baseline_date[:10]}",
            "--limit",
            str(issue_limit),
            "--json",
            "number,title,state,createdAt,labels,url,comments",
        ]
    )
    if len(issues) >= issue_limit:
        print(
            f"> WARNING: hit the --issue-limit of {issue_limit}. Raise it; the\n"
            "> list below is truncated.\n"
        )

    buckets = defaultdict(list)
    for issue in issues:
        names = {label["name"] for label in issue["labels"]}
        if "critical-fix" in names:
            bucket = "critical-fix (label already applied)"
        elif "bug" in names and issue["state"] == "OPEN":
            bucket = "bug, still open (triage each one)"
        elif "bug" in names:
            bucket = "bug, closed (confirm the fix is on the release branch)"
        elif names & {"performance"}:
            bucket = "performance (a severe regression can still block)"
        else:
            bucket = (
                "other (feature / docs / ci / unlabeled — skim for mislabeled bugs)"
            )
        buckets[bucket].append((issue, sorted(names)))

    order = [
        "critical-fix (label already applied)",
        "bug, still open (triage each one)",
        "bug, closed (confirm the fix is on the release branch)",
        "performance (a severe regression can still block)",
        "other (feature / docs / ci / unlabeled — skim for mislabeled bugs)",
    ]
    print(f"{len(issues)} issue(s) total.\n")
    for bucket in order:
        entries = buckets.get(bucket, [])
        print(f"### {bucket} — {len(entries)}\n")
        if not entries:
            print("_none_\n")
            continue
        for issue, names in entries:
            print(
                f"- #{issue['number']} [{issue['state']}] {issue['title']}\n"
                f"  - labels: {', '.join(names) or 'none'}"
                f" | opened {issue['createdAt'][:10]}"
                f" | {len(issue['comments'])} comment(s) | {issue['url']}"
            )
        print()


def prs_in_range(baseline, head):
    """PR numbers merged in `baseline..head`, read from commit subjects."""
    subjects = run(["git", "log", "--format=%s", f"{baseline}..{head}"]).splitlines()
    numbers = {}
    for subject in subjects:
        match = PR_NUMBER_RE.search(subject)
        if match:
            numbers[int(match.group(1))] = subject
    return numbers


def range_floor(baseline, head):
    """Earliest merge date a PR in `baseline..head` can have.

    The baseline tag is often a patch cut from a release branch weeks *after* that
    branch diverged from main, so its own date would hide everything merged to main
    between the branch point and the patch. Floor on the merge-base instead, with two
    days of slack for rebased committer dates.
    """
    base = run(["git", "merge-base", baseline, head]).strip()
    when = datetime.fromisoformat(
        run(["git", "log", "-1", "--format=%cI", base]).strip()
    )
    return base, (when - timedelta(days=2)).date().isoformat()


def section_breaking(repo, baseline, head):
    heading(f"Labeled changes in {baseline}..{head}")
    base_sha, floor = range_floor(baseline, head)
    in_range = prs_in_range(baseline, head)
    print(f"{len(in_range)} commit(s) in range carry a PR number.")
    print(f"Branch point: `{base_sha[:9]}`. Label search floored at {floor}.\n")

    for label in ("breaking-change", "format-change"):
        labeled = gh_json(
            [
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "merged",
                "--search",
                f"label:{label} merged:>={floor}",
                "--limit",
                "300",
                "--json",
                "number,title,url,mergedAt",
            ]
        )
        hits = [pr for pr in labeled if pr["number"] in in_range]
        print(f"### `{label}` PRs in range — {len(hits)}\n")
        if not hits:
            print("_none_\n")
            continue
        for pr in hits:
            print(
                f"- #{pr['number']} {pr['title']}\n"
                f"  - merged {pr['mergedAt'][:10]} | {pr['url']}"
            )
        print()
        off_range = [pr for pr in labeled if pr["number"] not in in_range]
        if off_range:
            print(
                f"_{len(off_range)} other `{label}` PR(s) merged since {floor} are\n"
                "not in this range (merged to main but not cherry-picked, or merged\n"
                "before the branch point)._\n"
            )


def constants_at(rev):
    """name -> value for every version-shaped const at `rev`, keyed by file."""
    raw = run(
        [
            "git",
            "grep",
            "-n",
            "-E",
            r"^\s*(pub(\([^)]*\))?\s+)?const\s+[A-Z]",
            rev,
            "--",
            *CONSTANT_PATHS,
        ],
        check=False,
    )
    found = {}
    for line in raw.splitlines():
        # format: <rev>:<path>:<lineno>:<text>
        parts = line.split(":", 3)
        if len(parts) < 4:
            continue
        _, path, _, text = parts
        match = CONSTANT_RE.match(text)
        if not match:
            continue
        name, value = match.group(1), match.group(2).strip()
        if not VERSION_NAME_RE.search(name) or NOT_A_VERSION_RE.search(name):
            continue
        found[f"{path}::{name}"] = value
    return found


def section_defaults(baseline, head):
    heading(f"Written-by-default versions: {baseline} vs {head}")
    before = constants_at(baseline)
    after = constants_at(head)

    changed = sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    )
    if not changed:
        print(
            "No version constant changed. Still read the diffs below before\n"
            "concluding.\n"
        )
    else:
        print(f"{len(changed)} version constant(s) changed:\n")
        for key in changed:
            path, _, name = key.partition("::")
            print(
                f"- `{name}` in `{path}`: "
                f"`{before.get(key, '(absent)')}` -> `{after.get(key, '(absent)')}`"
            )
        print()
        print(
            "> A constant defined as an alias (e.g. `BTREE_INDEX_VERSION = "
            "BTREE_ROW_ADDR_DOMAIN_VERSION`) only moves when the constant it names\n"
            "> moves, "
            "so both appear here. Read the code before calling it two separate bumps.\n"
        )

    print("### Diff of the default-selecting files\n")
    diff = run(
        ["git", "diff", f"{baseline}..{head}", "--stat", "--", *DEFAULT_SELECTOR_FILES],
        check=False,
    ).strip()
    print("```")
    print(diff or "(no changes)")
    print("```\n")
    if diff:
        print(
            "Read the full diff for each file above before judging whether a\n"
            "*default* moved "
            "(as opposed to a new non-default version simply becoming available):\n"
        )
        for path in DEFAULT_SELECTOR_FILES:
            print(f"    git diff {baseline}..{head} -- {path}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        help="Stable tag to compare against. Default: newest `vX.Y.Z` tag.",
    )
    parser.add_argument(
        "--head", default="HEAD", help="Release candidate ref. Default: HEAD."
    )
    parser.add_argument(
        "--repo", help="owner/name. Default: inferred from the `origin` remote."
    )
    parser.add_argument(
        "--issue-limit",
        type=int,
        default=400,
        help="Maximum issues to fetch. Default: 400.",
    )
    parser.add_argument(
        "--section",
        action="append",
        choices=["release", "issues", "breaking", "defaults"],
        help="Run only these sections. Repeatable. Default: all.",
    )
    args = parser.parse_args()

    tags = stable_tags()
    if not tags:
        sys.exit("no `vX.Y.Z` tags found; fetch tags with `git fetch --tags`")
    baseline = args.baseline or tags[-1][0]
    baseline_date = tag_date(baseline)
    repo = args.repo or detect_repo()
    sections = args.section or ["release", "issues", "breaking", "defaults"]

    print(f"# Stable release signals for `{repo}`")

    if "release" in sections:
        section_release(baseline, baseline_date, args.head)
    if "issues" in sections:
        section_issues(repo, baseline, baseline_date, args.issue_limit)
    if "breaking" in sections:
        section_breaking(repo, baseline, args.head)
    if "defaults" in sections:
        section_defaults(baseline, args.head)


if __name__ == "__main__":
    main()
