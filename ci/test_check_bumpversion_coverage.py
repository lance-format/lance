"""Unit tests for the bumpversion crate coverage check.

Run with: pytest ci/test_check_bumpversion_coverage.py
"""

from pathlib import Path

import pytest

from check_bumpversion_coverage import (
    block_for,
    evaluate,
    pinned_deps,
    render,
    workspace_version,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

VERSION = "15.0.0-beta.32"


def cargo(*deps, version=VERSION):
    """A root manifest pinning `deps` the way the real workspace does."""
    lines = [
        "[workspace.package]",
        f'version = "{version}"',
        "",
        "[workspace.dependencies]",
    ]
    lines += [
        f'{name} = {{ version = "={version}", path = "./rust/{name}" }}'
        for name in deps
    ]
    return "\n".join(lines) + "\n"


def bumpversion(*deps, version=VERSION):
    """A config with one block per crate in `deps`, plus the version block."""
    blocks = [
        "[tool.bumpversion]",
        f'current_version = "{version}"',
        "",
        "[[tool.bumpversion.files]]",
        'filename = "Cargo.toml"',
        "search = 'version = \"{current_version}\"'",
        "replace = 'version = \"{new_version}\"'",
        "",
    ]
    return "\n".join(blocks) + "\n" + "\n".join(block_for(name) for name in deps)


def titles(cargo_text, bump_text):
    return [title for _, title, _ in evaluate(cargo_text, bump_text)]


def test_repo_config_covers_every_pinned_crate():
    """The real files must pass; this is what the check exists to enforce."""
    cargo_text = (REPO_ROOT / "Cargo.toml").read_text()
    bump_text = (REPO_ROOT / ".bumpversion.toml").read_text()
    assert evaluate(cargo_text, bump_text) == []


def test_accepts_a_fully_covered_workspace():
    covered = bumpversion("lance", "lance-core")
    assert evaluate(cargo("lance", "lance-core"), covered) == []


def test_flags_a_crate_with_no_block():
    """The failure that broke publish-beta: a new crate nobody wired up."""
    cargo_text = cargo("lance", "lance-table-runtime")
    problems = evaluate(cargo_text, bumpversion("lance"))
    assert [title for _, title, _ in problems] == [
        "Crate missing from bumpversion config"
    ]
    assert "lance-table-runtime" in problems[0][2]


def test_missing_crate_message_contains_the_block_to_add():
    (_, _, message) = evaluate(cargo("lance-geo"), bumpversion())[0]
    assert block_for("lance-geo") in message


def test_flags_a_block_with_no_matching_dependency():
    """A renamed or removed crate leaves a block that can never match."""
    assert titles(cargo("lance"), bumpversion("lance", "lance-gone")) == [
        "Unused bumpversion search"
    ]


def test_flags_a_reformatted_dependency_line():
    """`version` must stay the first key, or the search no longer matches."""
    cargo_text = (
        "[workspace.package]\n"
        f'version = "{VERSION}"\n\n'
        "[workspace.dependencies]\n"
        f'lance-io = {{ default-features = false, version = "={VERSION}", '
        'path = "./rust/lance-io" }\n'
    )
    assert titles(cargo_text, bumpversion("lance-io")) == ["Unused bumpversion search"]


def test_flags_a_current_version_that_has_drifted_from_the_manifest():
    cargo_text = cargo("lance", version="15.0.0-beta.33")
    problems = evaluate(cargo_text, bumpversion("lance", version=VERSION))
    assert problems[0][1] == "Stale bumpversion current_version"
    assert "15.0.0-beta.33" in problems[0][2]


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param('{ path = "./rust/lance-tools" }', id="path_only"),
        pytest.param('{ version = "0.4", path = "./rust/fsst" }', id="caret_version"),
        pytest.param('"1.2"', id="plain_version"),
    ],
)
def test_ignores_dependencies_that_are_not_pinned_to_the_workspace_version(spec):
    """Only `path` + `=` dependencies name the version the release bumps."""
    cargo_text = (
        "[workspace.package]\n"
        f'version = "{VERSION}"\n\n'
        "[workspace.dependencies]\n"
        f"some-dep = {spec}\n"
    )
    assert pinned_deps(cargo_text) == {}
    assert evaluate(cargo_text, bumpversion()) == []


def test_pinned_deps_strips_the_exact_requirement_marker():
    assert pinned_deps(cargo("lance-file")) == {"lance-file": VERSION}


def test_workspace_version_reads_the_package_table():
    assert workspace_version(cargo("lance", version="16.1.2")) == "16.1.2"


def test_render_substitutes_the_version_and_unescapes_braces():
    template = 'lance = {{ version = "={current_version}"'
    assert render(template, current_version=VERSION) == (
        f'lance = {{ version = "={VERSION}"'
    )


def test_render_leaves_unrelated_placeholders_alone():
    assert render("{major}.{minor}", current_version=VERSION) == "{major}.{minor}"
