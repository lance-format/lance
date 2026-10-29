"""Unit tests for the Rust module size check.

Run with: pytest ci/test_check_module_size.py
"""

import subprocess

import pytest
from check_module_size import (
    EXCEPTIONS_FILE,
    FileChange,
    Size,
    changed_rust_files,
    evaluate,
    main,
    parse_exceptions,
    stale_exceptions,
)

SMALL = Size(100, 4_000)
WARN = Size(3_500, 140_000)
LARGE = Size(10_000, 410_000)


@pytest.mark.parametrize(
    "before, after, excepted, expected",
    [
        pytest.param(SMALL, Size(200, 8_000), False, None, id="small"),
        pytest.param(Size(2_990, 100_000), Size(3_000, 101_000), False, None, id="at_warn_limit"),
        pytest.param(Size(2_990, 100_000), Size(3_001, 101_000), False, "warn", id="crosses_warn"),
        pytest.param(WARN, Size(3_510, 140_500), False, "warn", id="grows_above_warn"),
        pytest.param(Size(7_000, 399_000), Size(7_100, 400_001), False, "fail", id="crosses_fail"),
        pytest.param(LARGE, Size(10_000, 410_100), False, "fail", id="longer_lines_only"),
        pytest.param(LARGE, Size(10_100, 414_000), True, "excepted", id="excepted"),
        pytest.param(Size(500, 390_000), Size(600, 401_000), False, "fail", id="few_long_lines"),
        pytest.param(LARGE, LARGE, False, None, id="unchanged_size"),
        pytest.param(LARGE, Size(9_000, 380_000), False, None, id="shrinks"),
        pytest.param(Size(0, 0), LARGE, False, "fail", id="new_file"),
    ],
)  # fmt: skip
def test_evaluate(before, after, excepted, expected):
    exceptions = {"a.rs": 1} if excepted else {}
    findings = evaluate([FileChange("a.rs", before, after)], exceptions)
    kinds = [
        "fail" if f.failed else "excepted" if f.excepted else "warn" for f in findings
    ]
    assert kinds == ([] if expected is None else [expected])


def test_parse_exceptions():
    text = "# header\n\nrust/a.rs\nrust/b.rs  # being split\n"
    assert parse_exceptions(text) == {"rust/a.rs": 3, "rust/b.rs": 4}


def test_stale_exceptions():
    exceptions = {"large.rs": 1, "split.rs": 2, "deleted.rs": 3}
    sizes = {"large.rs": LARGE, "split.rs": WARN, "deleted.rs": None}
    assert stale_exceptions(exceptions, sizes) == ["split.rs", "deleted.rs"]


def test_changed_rust_files(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", "-C", tmp_path, *args], check=True, capture_output=True)

    def commit(files):
        for name, lines in files.items():
            (tmp_path / name).write_text("x\n" * lines)
        git("add", "-A")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")

    git("init", "-q")
    commit({"grown.rs": 10, "renamed.rs": 20, "same.rs": 5, "notes.md": 1})
    git("branch", "base")
    git("mv", "renamed.rs", "moved.rs")
    commit({"grown.rs": 15, "moved.rs": 21, "added.rs": 3, "notes.md": 50})

    monkeypatch.chdir(tmp_path)
    changes = changed_rust_files("base", "HEAD")

    assert sorted(changes, key=lambda c: c.path) == [
        FileChange("added.rs", Size(0, 0), Size(3, 6)),
        FileChange("grown.rs", Size(10, 20), Size(15, 30)),
        FileChange("moved.rs", Size(20, 40), Size(21, 42)),
    ]


def numbered(start, stop):
    """Distinct 8-byte lines, so git can merge edits to different regions."""
    return "".join(f"line {i:02}\n" for i in range(start, stop))


@pytest.mark.parametrize(
    "base_files, main_files, pr_files",
    [
        # main grows big.rs past the limit and lists it after the PR branched.
        pytest.param(
            {"big.rs": numbered(0, 10), "other.rs": "", EXCEPTIONS_FILE: ""},
            {"big.rs": numbered(0, 14), EXCEPTIONS_FILE: "big.rs\n"},
            {"other.rs": numbered(0, 1)},
            id="main_adds_exception",
        ),
        # main splits big.rs and unlists it while the PR appends to it.
        pytest.param(
            {"big.rs": numbered(0, 15), EXCEPTIONS_FILE: "big.rs\n"},
            {"big.rs": numbered(10, 15), EXCEPTIONS_FILE: ""},
            {"big.rs": numbered(0, 16)},
            id="main_removes_exception",
        ),
    ],
)
@pytest.mark.parametrize("head", ["HEAD", "HEAD^2"])
def test_pr_diverged_from_main(
    tmp_path, monkeypatch, base_files, main_files, pr_files, head
):
    def git(*args):
        subprocess.run(["git", "-C", tmp_path, *args], check=True, capture_output=True)

    def commit(files):
        for name, content in files.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        git("add", "-A")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c")

    git("init", "-q", "-b", "main")
    commit(base_files)
    git("switch", "-q", "-c", "pr")
    commit(pr_files)
    git("switch", "-q", "main")
    commit(main_files)
    # Mirror the workflow's checkout: the test merge commit of the PR into main.
    git("-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff", "pr")

    monkeypatch.chdir(tmp_path)
    assert main(["HEAD^1", head, "--fail-bytes", "100"]) == 0
