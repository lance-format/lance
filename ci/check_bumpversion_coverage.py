#!/usr/bin/env python3
"""Fail if `.bumpversion.toml` does not cover every pinned workspace crate.

The release bump rewrites the workspace version and each pinned in-workspace
dependency of the root `Cargo.toml`, but bump-my-version has no notion of the
dependency table: every crate needs its own `[[tool.bumpversion.files]]` block,
listed by hand. A new crate without one bumps `[workspace.package] version`
while leaving its own `=` pin behind, so the bumped manifest cannot resolve:

    error: failed to select a version for the requirement
    `lance-table-runtime = "=15.0.0-beta.32"`
    candidate versions found which didn't match: 15.0.0-beta.33

That only surfaces when publish-beta runs, long after the PR that added the
crate. This reports the same inconsistency on the PR, and also flags blocks
whose search string no longer matches the manifest, so the list cannot rot in
the other direction either.

Results are emitted as GitHub workflow annotations. Only the two files are read.
"""

import argparse
import sys
import tomllib

CARGO_FILE = "Cargo.toml"
BUMPVERSION_FILE = ".bumpversion.toml"


class _Passthrough(dict):
    """Leave placeholders this check does not substitute (`{major}`) alone."""

    def __missing__(self, key):
        return "{" + key + "}"


def render(template, **values):
    """Expand a bump-my-version search/replace template."""
    return template.format_map(_Passthrough(values))


def workspace_version(cargo_text):
    return tomllib.loads(cargo_text)["workspace"]["package"]["version"]


def pinned_deps(cargo_text):
    """Return the in-workspace dependencies pinned to an exact version.

    A `path` plus a `=` requirement is what makes a dependency part of the
    release bump: path-only dependencies are not published and caret
    requirements do not name the version being bumped.
    """
    table = tomllib.loads(cargo_text)["workspace"]["dependencies"]
    return {
        name: spec["version"].lstrip("=")
        for name, spec in table.items()
        if isinstance(spec, dict)
        and "path" in spec
        and str(spec.get("version", "")).startswith("=")
    }


def cargo_entries(bump_text):
    """Return the `search` template of each block targeting the root manifest."""
    config = tomllib.loads(bump_text)["tool"]["bumpversion"]
    return [
        entry["search"]
        for entry in config.get("files", [])
        if entry.get("filename") == CARGO_FILE and "search" in entry
    ]


def bumpversion_version(bump_text):
    return tomllib.loads(bump_text)["tool"]["bumpversion"]["current_version"]


def expected_search(name, version):
    """The manifest text a block for `name` has to match."""
    return f'{name} = {{ version = "={version}"'


def block_for(name):
    return (
        "[[tool.bumpversion.files]]\n"
        f'filename = "{CARGO_FILE}"\n'
        f"search = '{name} = {{{{ version = \"={{current_version}}\"'\n"
        f"replace = '{name} = {{{{ version = \"={{new_version}}\"'\n"
    )


def evaluate(cargo_text, bump_text):
    """Return a (level, title, message) annotation for each problem found."""
    problems = []
    declared = bumpversion_version(bump_text)
    actual = workspace_version(cargo_text)
    if declared != actual:
        problems.append(
            (
                "error",
                "Stale bumpversion current_version",
                f"{BUMPVERSION_FILE} declares current_version {declared}, but "
                f"[workspace.package] version is {actual}. The bump searches for "
                "the declared version and would not match the manifest.",
            )
        )

    deps = pinned_deps(cargo_text)
    searches = cargo_entries(bump_text)
    covered = {render(search, current_version=declared) for search in searches}
    for name, version in sorted(deps.items()):
        if expected_search(name, version) not in covered:
            problems.append(
                (
                    "error",
                    "Crate missing from bumpversion config",
                    f"`{name}` is pinned to ={version} in {CARGO_FILE} but no "
                    f"{BUMPVERSION_FILE} block updates it, so the release bump "
                    "would leave the pin behind and the bumped manifest would not "
                    f"resolve. Add:\n{block_for(name)}",
                )
            )

    for search in searches:
        rendered = render(search, current_version=declared)
        if rendered not in cargo_text:
            problems.append(
                (
                    "error",
                    "Unused bumpversion search",
                    f"No line of {CARGO_FILE} contains `{rendered}`. The block "
                    "is for a crate that was renamed or removed, or the "
                    "dependency was reformatted; update or delete it.",
                )
            )
    return problems


def escape(value):
    """Escape a workflow command value (https://github.com/actions/toolkit/issues/193)."""
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo", default=CARGO_FILE, help="root workspace manifest")
    parser.add_argument(
        "--bumpversion", default=BUMPVERSION_FILE, help="bump-my-version config"
    )
    args = parser.parse_args(argv)

    with open(args.cargo) as f:
        cargo_text = f.read()
    with open(args.bumpversion) as f:
        bump_text = f.read()

    problems = evaluate(cargo_text, bump_text)
    for level, title, message in problems:
        print(f"::{level} file={args.bumpversion},title={title}::{escape(message)}")
    if not problems:
        print(
            f"All crates pinned in {args.cargo} are covered by {args.bumpversion}.",
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
