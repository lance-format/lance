"""
Check whether there are any breaking changes in the PRs between the base and head commits.
If there are, assert that we have incremented the minor version.

Can also be used as a library to detect breaking changes without version validation.
"""

import argparse
import os
import sys
from packaging.version import parse

from github import Github


def detect_breaking_changes(repo, base, head):
    """
    Detect if there are any breaking changes between base and head commits.

    Args:
        repo: GitHub repository object
        base: Base commit/tag
        head: Head commit/tag

    Returns:
        bool: True if breaking changes found, False otherwise
    """
    commits = repo.compare(base, head).commits
    prs = (pr for commit in commits for pr in commit.get_pulls())

    for pr in prs:
        if any(label.name == "breaking-change" for label in pr.labels):
            print(f"Breaking change in PR: {pr.html_url}")
            return True

    print("No breaking changes found.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="Base commit/tag for comparison")
    parser.add_argument("head", help="Head commit/tag for comparison")
    parser.add_argument(
        "last_stable_version", nargs="?", help="Last stable version (for validation)"
    )
    parser.add_argument(
        "current_version", nargs="?", help="Current version (for validation)"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Only detect breaking changes, don't validate version",
    )
    args = parser.parse_args()

    repo = Github(os.environ["GITHUB_TOKEN"]).get_repo(os.environ["GITHUB_REPOSITORY"])

    has_breaking_changes = detect_breaking_changes(repo, args.base, args.head)

    if args.detect_only:
        # Exit with 1 if breaking changes found, 0 if not
        sys.exit(1 if has_breaking_changes else 0)

    # Original behavior: validate version bump if breaking changes found
    if not has_breaking_changes:
        sys.exit(0)

    # Breaking changes found, validate version was bumped appropriately
    if not args.last_stable_version or not args.current_version:
        print("Error: last_stable_version and current_version required for validation")
        sys.exit(1)

    last_stable_version = parse(args.last_stable_version)
    current_version = parse(args.current_version)
    if current_version.minor <= last_stable_version.minor:
        print("Minor version is not greater than the last stable version.")
        sys.exit(1)
