"""Unit tests for the Arrow/DataFusion upgrade detector.

Run with: pytest ci/test_arrow_upgrade.py
"""

import pytest
from arrow_upgrade import (
    diff,
    is_ordering_crate,
    main,
    ordering_crates,
    release_notes_url,
    summary_markdown,
)


CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"
ARROW_GIT = "git+https://github.com/apache/arrow-rs?branch=main#"


def lockfile(*packages):
    """Render a minimal Cargo.lock holding (name, version) or (name, version, source) entries."""
    blocks = []
    for entry in packages:
        name, version = entry[0], entry[1]
        source = entry[2] if len(entry) > 2 else CRATES_IO
        blocks.append(
            f'[[package]]\nname = "{name}"\nversion = "{version}"\nsource = "{source}"\n'
        )
    return "version = 4\n\n" + "\n".join(blocks)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("arrow", True),
        ("arrow-ord", True),
        ("arrow-row", True),
        ("datafusion", True),
        ("datafusion-physical-plan", True),
        ("half", True),
        ("lance-arrow", False),
        ("lance-arrow-scalar", False),
        ("lance-datafusion", False),
        ("arrow2", False),
        ("halfbrown", False),
    ],
)
def test_is_ordering_crate(name, expected):
    assert is_ordering_crate(name) is expected


def test_ordering_crates_keeps_every_resolved_version():
    text = lockfile(
        ("arrow-ord", "58.4.0"), ("arrow-ord", "57.0.0"), ("serde", "1.0.0")
    )
    assert ordering_crates(text) == {"arrow-ord": {("58.4.0", None), ("57.0.0", None)}}


def test_diff_reports_bumps_additions_and_removals_sorted():
    base = ordering_crates(
        lockfile(("arrow-ord", "58.4.0"), ("datafusion", "54.1.0"), ("half", "2.4.1"))
    )
    head = ordering_crates(
        lockfile(("arrow-ord", "59.0.0"), ("arrow-row", "59.0.0"), ("half", "2.4.1"))
    )
    assert diff(base, head) == [
        ("arrow-ord", {("58.4.0", None)}, {("59.0.0", None)}),
        ("arrow-row", set(), {("59.0.0", None)}),
        ("datafusion", {("54.1.0", None)}, set()),
    ]


def test_diff_sees_a_git_revision_change_at_the_same_version():
    base = ordering_crates(lockfile(("arrow-ord", "58.4.0", ARROW_GIT + "aaa111")))
    head = ordering_crates(lockfile(("arrow-ord", "58.4.0", ARROW_GIT + "bbb222")))
    assert diff(base, head) == [
        (
            "arrow-ord",
            {("58.4.0", ARROW_GIT + "aaa111")},
            {("58.4.0", ARROW_GIT + "bbb222")},
        ),
    ]


def test_diff_sees_a_move_between_crates_io_and_git():
    base = ordering_crates(lockfile(("arrow-ord", "58.4.0")))
    head = ordering_crates(lockfile(("arrow-ord", "58.4.0", ARROW_GIT + "aaa111")))
    assert diff(base, head) == [
        ("arrow-ord", {("58.4.0", None)}, {("58.4.0", ARROW_GIT + "aaa111")}),
    ]


def test_diff_ignores_unrelated_bumps():
    base = ordering_crates(lockfile(("arrow-ord", "58.4.0"), ("tokio", "1.40.0")))
    head = ordering_crates(lockfile(("arrow-ord", "58.4.0"), ("tokio", "1.41.0")))
    assert diff(base, head) == []


@pytest.mark.parametrize(
    ("name", "version", "expected"),
    [
        (
            "arrow-row",
            "59.0.0",
            "https://github.com/apache/arrow-rs/releases/tag/59.0.0",
        ),
        (
            "datafusion-common",
            "55.0.0",
            "https://github.com/apache/datafusion/releases/tag/55.0.0",
        ),
        ("half", "2.5.0", "https://github.com/VoidStarKat/half-rs/releases/tag/v2.5.0"),
    ],
)
def test_release_notes_url(name, version, expected):
    assert release_notes_url(name, version) == expected


def test_summary_links_only_the_new_versions():
    markdown = summary_markdown(
        [("arrow-ord", {("58.4.0", None)}, {("58.4.0", None), ("59.0.0", None)})]
    )
    assert (
        "| `arrow-ord` | 58.4.0 | 58.4.0, 59.0.0 | [59.0.0](https://github.com/apache/arrow-rs/releases/tag/59.0.0) |"
        in markdown
    )
    assert "arrow-ordering-reviewed" in markdown


def test_summary_shows_the_git_source():
    markdown = summary_markdown(
        [("arrow-ord", {("58.4.0", None)}, {("58.4.0", ARROW_GIT + "aaa111")})]
    )
    assert f"| `arrow-ord` | 58.4.0 | 58.4.0 from {ARROW_GIT}aaa111 |" in markdown


def test_main_writes_github_outputs(tmp_path, monkeypatch, capsys):
    base = tmp_path / "base.lock"
    head = tmp_path / "head.lock"
    base.write_text(lockfile(("arrow-ord", "58.4.0")))
    head.write_text(lockfile(("arrow-ord", "59.0.0")))
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert main(["arrow_upgrade.py", str(base), str(head)]) == 0

    assert capsys.readouterr().out == "arrow-ord: 58.4.0 -> 59.0.0\n"
    assert output.read_text() == "changed=true\n"
    assert "| `arrow-ord` | 58.4.0 | 59.0.0 |" in summary.read_text()


def test_main_reports_no_change_without_a_summary(tmp_path, monkeypatch, capsys):
    lock = tmp_path / "lock"
    lock.write_text(lockfile(("arrow-ord", "58.4.0")))
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert main(["arrow_upgrade.py", str(lock), str(lock)]) == 0

    assert "no ordering-relevant dependency changes" in capsys.readouterr().out
    assert output.read_text() == "changed=false\n"
    assert not summary.exists()
