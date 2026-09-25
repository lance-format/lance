#!/usr/bin/env python3
"""Fail if any crates.io dependency in a Cargo.lock is younger than a minimum age.

A freshly published version is the window in which a compromised crate is most
likely to still be live, so we refuse to ship one until it has had time to be
noticed and yanked. Cargo's own `registry.global-min-publish-age` only constrains
resolution: versions already written into Cargo.lock are grandfathered in and are
never re-checked, so it cannot answer "is what we committed too new?".

Publish timestamps come from the `pubtime` field of the sparse index. Cargo keeps
the index lines it has downloaded in `$CARGO_HOME/registry/index/*/.cache`, which
lets a full scan run without any network calls; anything not found there is
fetched from index.crates.io, and anything the index has not published yet is
looked up through the crates.io API, which is authoritative and never lags.

The same floor is enforced on developer machines by Aikido endpoint protection,
which filters the index it serves them. A security fix that has to land sooner
therefore needs an Aikido allowlist entry as well as one in
ci/dependency-age-allowlist.toml; the entry here only unblocks CI.
"""

import argparse
import json
import os
import sys
import time
import tomllib
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_LOCKFILES = ("Cargo.lock", "python/Cargo.lock", "java/lance-jni/Cargo.lock")
DEFAULT_ALLOWLIST = "ci/dependency-age-allowlist.toml"
DEFAULT_MIN_AGE_HOURS = 48
CRATES_IO_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
INDEX_URL = "https://index.crates.io"
API_URL = "https://crates.io/api/v1/crates"
USER_AGENT = "lance-ci-dependency-age (https://github.com/lance-format/lance)"
FETCH_THREADS = 16
# The API is rate limited and is only ever asked about the few versions the
# index CDN has not caught up with, so it gets a much smaller pool.
API_THREADS = 4
FETCH_TIMEOUT_SECONDS = 30
FETCH_ATTEMPTS = 4

# The cache file is a cargo internal: a format byte, an index-format u32, an
# `etag: ...` line, then a NUL-separated run of alternating version and index
# JSON. Only the JSON matters here, so rather than validating the framing we
# pick out the chunks that parse, and let the caller fall back to HTTP if a
# cargo release ever changes the layout underneath us.
CACHE_JSON_PREFIX = b"{"


def index_path(name):
    """Return the sparse-index path prefix cargo uses for a crate name."""
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def parse_index_entries(chunks):
    """Map version -> pubtime (or None) from an iterable of index JSON chunks."""
    entries = {}
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith(CACHE_JSON_PREFIX):
            continue
        entry = json.loads(chunk)
        entries[entry["vers"]] = entry.get("pubtime")
    return entries


def parse_cache_file(data):
    try:
        return parse_index_entries(data.split(b"\x00"))
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
        return {}


def parse_index_response(data):
    return parse_index_entries(data.splitlines())


def parse_api_response(data):
    """Map version -> publish time from a crates.io `/versions` response."""
    versions = json.loads(data)["versions"]
    return {version["num"]: version.get("created_at") for version in versions}


def crates_io_packages(lock_text):
    """Return (name, version) for every crates.io package in a lockfile."""
    lock = tomllib.loads(lock_text)
    return [
        (package["name"], package["version"])
        for package in lock.get("package", [])
        if package.get("source") == CRATES_IO_SOURCE
    ]


def load_allowlist(path):
    """Map (crate, version) -> reason for the exemptions in an allowlist file."""
    if not Path(path).is_file():
        return {}
    entries = tomllib.loads(Path(path).read_text()).get("allow", [])
    allowed = {}
    for entry in entries:
        missing = {"crate", "version", "reason"} - entry.keys()
        if missing:
            raise SystemExit(
                f"{path}: allow entry {entry} is missing {sorted(missing)}"
            )
        allowed[(entry["crate"], entry["version"])] = entry["reason"]
    return allowed


def cache_dirs(cargo_home):
    return sorted(Path(cargo_home).glob("registry/index/*/.cache"))


def pubtimes_from_cache(names, cargo_home):
    dirs = cache_dirs(cargo_home)
    found = {}
    for name in names:
        for directory in dirs:
            path = directory / index_path(name)
            if not path.is_file():
                continue
            entries = parse_cache_file(path.read_bytes())
            if entries:
                found[name] = entries
                break
    return found


def http_get(url):
    """Return the body, or None for a 404. Retries transient failures."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(FETCH_ATTEMPTS):
        try:
            with urllib.request.urlopen(
                request, timeout=FETCH_TIMEOUT_SECONDS
            ) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == FETCH_ATTEMPTS - 1:
                raise
        except OSError:
            if attempt == FETCH_ATTEMPTS - 1:
                raise
        time.sleep(2**attempt)


def fetch_index_pubtimes(name):
    body = http_get(f"{INDEX_URL}/{index_path(name)}")
    return name, {} if body is None else parse_index_response(body)


def fetch_api_pubtimes(name):
    body = http_get(f"{API_URL}/{name}/versions")
    return name, {} if body is None else parse_api_response(body)


def pubtimes_from_index(names):
    return fetch_all(fetch_index_pubtimes, names, FETCH_THREADS)


def pubtimes_from_api(names):
    return fetch_all(fetch_api_pubtimes, names, API_THREADS)


def fetch_all(fetch, names, threads):
    if not names:
        return {}
    with ThreadPoolExecutor(threads) as pool:
        return dict(pool.map(fetch, names))


def resolve_pubtimes(packages, cargo_home):
    """Sort packages into dated, undated, and not-in-the-index.

    Returns ({(name, version): pubtime}, [undated], [absent]).
    """
    known = pubtimes_from_cache({name for name, _ in packages}, cargo_home)
    for lookup in (pubtimes_from_index, pubtimes_from_api):
        stale = sorted(
            {name for name, version in packages if version not in known.get(name, {})}
        )
        for name, entries in lookup(stale).items():
            known[name] = {**known.get(name, {}), **entries}

    dated, undated, absent = {}, [], []
    for name, version in packages:
        entries = known.get(name, {})
        if version not in entries:
            absent.append((name, version))
        elif entries[version] is None:
            undated.append((name, version))
        else:
            dated[(name, version)] = entries[version]
    return dated, undated, absent


def too_new(dated, cutoff):
    """Return (name, version, published) for versions published after the cutoff."""
    violations = []
    for (name, version), pubtime in dated.items():
        published = datetime.fromisoformat(pubtime.replace("Z", "+00:00"))
        if published > cutoff:
            violations.append((name, version, published))
    return violations


def check_lockfile(path, cutoff, cargo_home, allowed):
    """Print this lockfile's findings and return whether it passed."""
    packages = crates_io_packages(Path(path).read_text())
    dated, undated, absent = resolve_pubtimes(packages, cargo_home)

    if undated:
        # The index carries no pubtime for a few dozen versions predating the
        # field. Those are years old, so treating them as passing is safe.
        print(f"{path}: {len(undated)} of {len(packages)} versions predate `pubtime`")

    violations = [
        violation
        for violation in too_new(dated, cutoff)
        if (violation[0], violation[1]) not in allowed
    ]
    absent = [package for package in absent if package not in allowed]

    for name, version, published in sorted(violations):
        short_by = (published - cutoff).total_seconds() / 3600
        print(
            f"{path}: {name} {version} is too new — published {published.isoformat()}, "
            f"{short_by:.1f}h short of the minimum age"
        )
    for name, version in sorted(absent):
        print(f"{path}: {name} {version} is not in the index, so it cannot be dated")

    return not violations and not absent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lockfiles", nargs="*", default=list(DEFAULT_LOCKFILES))
    parser.add_argument("--min-age-hours", type=int, default=DEFAULT_MIN_AGE_HOURS)
    parser.add_argument("--allowlist", default=DEFAULT_ALLOWLIST)
    parser.add_argument(
        "--cargo-home", default=os.environ.get("CARGO_HOME", Path.home() / ".cargo")
    )
    args = parser.parse_args(argv)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=args.min_age_hours)
    allowed = load_allowlist(args.allowlist)
    for (name, version), reason in sorted(allowed.items()):
        print(f"{args.allowlist} exempts {name} {version}: {reason}")

    passed = [
        check_lockfile(path, cutoff, args.cargo_home, allowed)
        for path in args.lockfiles
    ]
    if all(passed):
        return 0

    print(
        f"\nEvery crates.io dependency must be at least {args.min_age_hours}h old. "
        "Wait for these versions to age, or pin the previous version. Dependabot's "
        "cooldown should normally keep this from firing.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
