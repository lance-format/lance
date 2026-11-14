#!/usr/bin/env python3
"""Gate for rebuilding one missing beta wheel from an already published tag.

A beta release publishes seven wheels, all built from a single release commit.
When one of them never reaches the Gemfury beta channel -- for example the
Windows ARM64 job hits its timeout while linking the extension -- the tag, the
GitHub pre-release and the six already published wheels must stay untouched.
The only safe recovery is to rebuild that one wheel from the exact commit the
published tag resolves to and upload it to the same channel.

This script is the read-only gate in front of that build and upload. It rejects
any target that is not an existing beta tag, whose GitHub release is not public
and marked as a pre-release, that does not resolve to the expected commit, or
whose wheel is already published. With ``--wheel-dir`` it instead checks a
locally built wheel, so only the expected artifact can be uploaded.

Run with: python ci/check_beta_wheel_recovery.py --tag v1.2.3-beta.4 --sha <sha>
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from packaging.version import Version

# The beta channel the release workflow uploads to, see python/AGENTS.md.
FURY_INDEX_URL = "https://pypi.fury.io/lance-format/pylance/"
WHEEL_DIST = "pylance"
# pyo3 is built with the abi3-py310 feature, so every wheel of a release shares
# this tag. They must match the wheel the release workflow would have produced.
WHEEL_PYTHON_TAG = "cp310"
WHEEL_ABI_TAG = "abi3"
WHEEL_PLATFORM_TAG = "win_arm64"
BETA_TAG = re.compile(r"^v(?P<release>\d+\.\d+\.\d+)-beta\.(?P<number>\d+)$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
# Wheel file names appear both in the href and as the link text of a simple
# index, so only accept a match that is followed by an HTML or fragment
# delimiter instead of, say, a longer file name.
WHEEL_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*\.whl(?=[\"'<#])")


class RecoveryError(Exception):
    """The requested recovery target is not safe to build or publish."""


def beta_version(tag: str) -> str:
    """Convert a beta tag to the version its wheels use, e.g. 15.0.0b20."""
    match = BETA_TAG.fullmatch(tag)
    if match is None:
        raise RecoveryError(
            f"tag {tag!r} is not a beta release tag; expected vMAJOR.MINOR.PATCH-beta.N"
        )
    # The release build writes the tag into python/Cargo.toml and lets maturin
    # and packaging normalize it, so normalize it the same way.
    return str(Version(f"{match['release']}-beta.{match['number']}"))


def wheel_name(tag: str) -> str:
    version = beta_version(tag)
    return (
        f"{WHEEL_DIST}-{version}"
        f"-{WHEEL_PYTHON_TAG}-{WHEEL_ABI_TAG}-{WHEEL_PLATFORM_TAG}.whl"
    )


def check_sha(sha: str) -> None:
    if not COMMIT_SHA.fullmatch(sha):
        raise RecoveryError(
            f"sha {sha!r} is not a full 40 character lowercase commit SHA"
        )


def check_release(release: dict, tag: str) -> None:
    if release.get("tag_name") != tag:
        raise RecoveryError(
            f"GitHub release for {tag!r} reports tag_name {release.get('tag_name')!r}"
        )
    if release.get("draft"):
        raise RecoveryError(f"GitHub release for {tag} is still a draft")
    if not release.get("prerelease"):
        raise RecoveryError(
            f"GitHub release for {tag} is not a pre-release, so it is not a beta release"
        )
    if not release.get("published_at"):
        raise RecoveryError(f"GitHub release for {tag} has no publication time")


def check_tag_commit(ls_remote: str, tag: str, sha: str) -> None:
    """Check that the tag on origin points at `sha`, dereferencing annotated tags."""
    refs = {}
    for line in ls_remote.splitlines():
        line_sha, _, ref = line.partition("\t")
        refs[ref] = line_sha
    # An annotated tag is listed twice: the tag object and its `<ref>^{}` commit.
    # A lightweight tag is listed once and already points at the commit.
    actual = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
    if actual is None:
        raise RecoveryError(f"tag {tag} does not exist on origin")
    if actual != sha:
        raise RecoveryError(f"tag {tag} resolves to {actual}, not the expected {sha}")


def check_wheel_absent(index_html: str, name: str, index_url: str) -> None:
    if name in WHEEL_FILE.findall(index_html):
        raise RecoveryError(
            f"{name} is already published on {index_url}; refusing to overwrite it"
        )


def check_built_wheel(wheel_dir: str, tag: str) -> Path:
    """Check that exactly the expected wheel was built and nothing else."""
    wheels = sorted(Path(wheel_dir).glob("*.whl"))
    if not wheels:
        raise RecoveryError(f"no wheel was built in {wheel_dir}")
    if len(wheels) > 1:
        found = [wheel.name for wheel in wheels]
        raise RecoveryError(f"expected exactly one wheel in {wheel_dir}, found {found}")
    expected = wheel_name(tag)
    if wheels[0].name != expected:
        raise RecoveryError(f"built {wheels[0].name}, expected {expected}")
    return wheels[0]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def github_release(repository: str, tag: str) -> dict:
    url = (
        "https://api.github.com/repos/"
        f"{repository}/releases/tags/{urllib.parse.quote(tag, safe='')}"
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "lance-beta-wheel-recovery",
        },
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise RecoveryError(
                f"{tag} has no GitHub release; publish the release before "
                "recovering a wheel for it"
            ) from error
        raise RecoveryError(
            f"could not read the GitHub release for {tag}: HTTP {error.code}"
        ) from error
    except urllib.error.URLError as error:
        raise RecoveryError(
            f"could not read the GitHub release for {tag}: {error.reason}"
        ) from error


def fury_index(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode(errors="replace")
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        raise RecoveryError(f"could not read {url}: {error}") from error


def main(
    argv=None,
    *,
    run_git=git,
    get_release=github_release,
    get_index=fury_index,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", required=True, help="existing beta tag, e.g. v1.2.3-beta.4"
    )
    parser.add_argument("--sha", help="full commit SHA the tag must resolve to")
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="owner/repo that holds the release (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--index-url",
        default=FURY_INDEX_URL,
        help="public simple index of the beta channel",
    )
    parser.add_argument(
        "--wheel-dir",
        help="check a locally built wheel in this directory instead of a "
        "published target",
    )
    args = parser.parse_args(argv)

    try:
        if args.wheel_dir:
            print(f"verified {check_built_wheel(args.wheel_dir, args.tag)}")
            return 0
        if not args.sha or not args.repository:
            parser.error(
                "--sha and --repository are required unless --wheel-dir is set"
            )
        check_sha(args.sha)
        name = wheel_name(args.tag)
        check_release(get_release(args.repository, args.tag), args.tag)
        # Ask for the peeled ref too, so annotated tags are compared by commit.
        ls_remote = run_git(
            "ls-remote",
            "--tags",
            "origin",
            f"refs/tags/{args.tag}",
            f"refs/tags/{args.tag}^{{}}",
        )
        check_tag_commit(ls_remote, args.tag, args.sha)
        check_wheel_absent(get_index(args.index_url), name, args.index_url)
    except RecoveryError as error:
        print(f"::error::{error}")
        return 1
    except subprocess.CalledProcessError as error:
        print(f"::error::git failed ({error.returncode}): {error.stderr.strip()}")
        return 1
    print(f"recovery target verified: {name} from {args.sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
