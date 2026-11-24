"""Unit tests for the crates.io dependency age check.

Run with: pytest ci/test_check_dependency_age.py
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from check_dependency_age import (
    check_lockfile,
    crates_io_packages,
    index_path,
    load_allowlist,
    parse_api_response,
    parse_cache_file,
    parse_index_response,
    too_new,
)

NOW = datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)
CUTOFF = NOW - timedelta(hours=48)


def cache_blob(entries):
    """Build a cargo index cache file the way cargo lays one out."""
    out = b"\x03\x02\x00\x00\x00" + b'etag: "abc"' + b"\x00"
    for version, pubtime in entries.items():
        entry = {"name": "demo", "vers": version, "deps": [], "yanked": False}
        if pubtime is not None:
            entry["pubtime"] = pubtime
        out += version.encode() + b"\x00" + json.dumps(entry).encode() + b"\x00"
    return out


def lockfile(packages):
    blocks = ["version = 4\n"]
    for name, version, source in packages:
        block = f'[[package]]\nname = "{name}"\nversion = "{version}"\n'
        if source is not None:
            block += f'source = "{source}"\n'
        blocks.append(block)
    return "\n".join(blocks)


CRATES_IO = "registry+https://github.com/rust-lang/crates.io-index"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param("a", "1/a", id="one_char"),
        pytest.param("ab", "2/ab", id="two_chars"),
        pytest.param("abc", "3/a/abc", id="three_chars"),
        pytest.param("serde", "se/rd/serde", id="long"),
        pytest.param("Inflector", "in/fl/inflector", id="uppercase_is_folded"),
    ],
)
def test_index_path(name, expected):
    assert index_path(name) == expected


def test_parse_cache_file_reads_pubtimes():
    blob = cache_blob({"1.0.0": "2020-01-01T00:00:00Z", "1.1.0": None})
    assert parse_cache_file(blob) == {
        "1.0.0": "2020-01-01T00:00:00Z",
        "1.1.0": None,
    }


def test_parse_cache_file_tolerates_an_unknown_layout():
    """A cargo change to the cache format must degrade to the HTTP fallback."""
    assert parse_cache_file(b"\x09\x99{not json at all}\x00") == {}


def test_parse_index_response_reads_newline_delimited_json():
    body = b'{"vers":"1.0.0","pubtime":"2020-01-01T00:00:00Z"}\n{"vers":"2.0.0"}\n'
    assert parse_index_response(body) == {
        "1.0.0": "2020-01-01T00:00:00Z",
        "2.0.0": None,
    }


def test_parse_api_response_reads_created_at():
    body = b'{"versions":[{"num":"1.0.0","created_at":"2020-01-01T00:00:00.123456Z"}]}'
    assert parse_api_response(body) == {"1.0.0": "2020-01-01T00:00:00.123456Z"}


def test_crates_io_packages_ignores_path_and_git_packages():
    text = lockfile(
        [
            ("serde", "1.0.0", CRATES_IO),
            ("lance", "0.1.0", None),
            ("forked", "0.2.0", "git+https://example.com/forked#abc123"),
        ]
    )
    assert crates_io_packages(text) == [("serde", "1.0.0")]


@pytest.mark.parametrize(
    ("pubtime", "expected"),
    [
        pytest.param("2026-09-10T00:00:00Z", [], id="old"),
        pytest.param("2026-09-14T11:59:00Z", [], id="just_past_cutoff"),
        pytest.param("2026-09-14T12:01:00Z", [("serde", "1.0.0")], id="just_too_new"),
        pytest.param("2026-09-16T11:00:00Z", [("serde", "1.0.0")], id="an_hour_old"),
    ],
)
def test_too_new_uses_the_cutoff_as_the_boundary(pubtime, expected):
    violations = too_new({("serde", "1.0.0"): pubtime}, CUTOFF)
    assert [(name, version) for name, version, _ in violations] == expected


def cargo_home_with(tmp_path, crates):
    home = tmp_path / "cargo"
    for name, entries in crates.items():
        path = (
            home / "registry/index/index.crates.io-1949cf8c/.cache" / index_path(name)
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(cache_blob(entries))
    return home


def test_load_allowlist_is_empty_when_the_file_is_absent(tmp_path):
    assert load_allowlist(tmp_path / "nope.toml") == {}


def test_load_allowlist_reads_entries(tmp_path):
    path = tmp_path / "allow.toml"
    path.write_text(
        '[[allow]]\ncrate = "serde"\nversion = "1.0.0"\nreason = "RUSTSEC-0"\n'
    )
    assert load_allowlist(path) == {("serde", "1.0.0"): "RUSTSEC-0"}


def test_load_allowlist_rejects_an_entry_without_a_reason(tmp_path):
    """An exemption with no rationale is not reviewable, so refuse to run."""
    path = tmp_path / "allow.toml"
    path.write_text('[[allow]]\ncrate = "serde"\nversion = "1.0.0"\n')
    with pytest.raises(SystemExit, match="reason"):
        load_allowlist(path)


def test_check_lockfile_passes_when_every_version_is_old_enough(tmp_path, capsys):
    lock = tmp_path / "Cargo.lock"
    lock.write_text(lockfile([("serde", "1.0.0", CRATES_IO)]))
    home = cargo_home_with(tmp_path, {"serde": {"1.0.0": "2026-09-01T00:00:00Z"}})

    assert check_lockfile(lock, CUTOFF, home, {}) is True
    assert capsys.readouterr().out == ""


def test_check_lockfile_reports_a_too_new_version(tmp_path, capsys):
    lock = tmp_path / "Cargo.lock"
    lock.write_text(lockfile([("serde", "1.0.0", CRATES_IO)]))
    home = cargo_home_with(tmp_path, {"serde": {"1.0.0": "2026-09-16T00:00:00Z"}})

    assert check_lockfile(lock, CUTOFF, home, {}) is False
    assert "serde 1.0.0 is too new" in capsys.readouterr().out


def test_check_lockfile_accepts_versions_predating_pubtime(tmp_path, capsys):
    lock = tmp_path / "Cargo.lock"
    lock.write_text(lockfile([("serde", "1.0.0", CRATES_IO)]))
    home = cargo_home_with(tmp_path, {"serde": {"1.0.0": None}})

    assert check_lockfile(lock, CUTOFF, home, {}) is True
    assert "1 of 1 versions predate `pubtime`" in capsys.readouterr().out


def test_check_lockfile_fails_when_the_index_does_not_list_the_version(
    tmp_path, capsys, monkeypatch
):
    """An undatable version is a hole in the check, so it must not pass silently."""
    monkeypatch.setattr("check_dependency_age.pubtimes_from_index", lambda names: {})
    monkeypatch.setattr("check_dependency_age.pubtimes_from_api", lambda names: {})
    lock = tmp_path / "Cargo.lock"
    lock.write_text(lockfile([("serde", "9.9.9", CRATES_IO)]))
    home = cargo_home_with(tmp_path, {"serde": {"1.0.0": "2026-09-01T00:00:00Z"}})

    assert check_lockfile(lock, CUTOFF, home, {}) is False
    assert "serde 9.9.9 is not in the index" in capsys.readouterr().out


def test_check_lockfile_honours_an_allowlisted_version(tmp_path, capsys):
    lock = tmp_path / "Cargo.lock"
    lock.write_text(lockfile([("serde", "1.0.0", CRATES_IO)]))
    home = cargo_home_with(tmp_path, {"serde": {"1.0.0": "2026-09-16T00:00:00Z"}})
    allowed = {("serde", "1.0.0"): "RUSTSEC-0"}

    assert check_lockfile(lock, CUTOFF, home, allowed) is True
    assert "too new" not in capsys.readouterr().out


def test_check_lockfile_allowlist_also_covers_an_undatable_version(
    tmp_path, capsys, monkeypatch
):
    """Aikido hides allowlisted-but-young versions from the index it serves."""
    monkeypatch.setattr("check_dependency_age.pubtimes_from_index", lambda names: {})
    monkeypatch.setattr("check_dependency_age.pubtimes_from_api", lambda names: {})
    lock = tmp_path / "Cargo.lock"
    lock.write_text(lockfile([("serde", "9.9.9", CRATES_IO)]))
    home = cargo_home_with(tmp_path, {"serde": {"1.0.0": "2026-09-01T00:00:00Z"}})

    assert check_lockfile(lock, CUTOFF, home, {("serde", "9.9.9"): "RUSTSEC-0"}) is True
    assert capsys.readouterr().out == ""
