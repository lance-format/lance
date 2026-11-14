"""Unit tests for the beta wheel recovery gate.

Run with: pytest ci/test_check_beta_wheel_recovery.py
"""

import pytest
from check_beta_wheel_recovery import (
    RecoveryError,
    check_built_wheel,
    check_release,
    check_sha,
    check_tag_commit,
    check_wheel_absent,
    main,
    wheel_name,
)

TAG = "v15.0.0-beta.20"
SHA = "86fff5f40f299a2ff6abb0c10210451f27e605eb"
WHEEL = "pylance-15.0.0b20-cp310-abi3-win_arm64.whl"
INDEX_URL = "https://pypi.fury.io/lance-format/pylance/"

RELEASE = {
    "tag_name": TAG,
    "draft": False,
    "prerelease": True,
    "published_at": "2026-11-13T15:36:12Z",
}
LS_REMOTE = (
    "ef769e505b77d53dc7ee1d4bbaeb582eeb4a5baa\trefs/tags/v15.0.0-beta.20\n"
    "86fff5f40f299a2ff6abb0c10210451f27e605eb\trefs/tags/v15.0.0-beta.20^{}\n"
)
INDEX_WITH_OTHER_PLATFORMS = (
    '<a href="https://pypi.fury.io/lance-format/-/ver_1/'
    'pylance-15.0.0b20-cp310-abi3-win_amd64.whl#sha256=00" data-requires-python='
    '"&gt;=3.10">pylance-15.0.0b20-cp310-abi3-win_amd64.whl</a><br/>\n'
    '<a href="https://pypi.fury.io/lance-format/-/ver_2/'
    'pylance-15.0.0b20-cp310-abi3-macosx_11_0_arm64.whl#sha256=01">'
    "pylance-15.0.0b20-cp310-abi3-macosx_11_0_arm64.whl</a><br/>\n"
)


@pytest.mark.parametrize(
    "tag, expected",
    [
        pytest.param(TAG, WHEEL, id="current_beta"),
        pytest.param(
            "v2.3.4-beta.1", "pylance-2.3.4b1-cp310-abi3-win_arm64.whl", id="other_beta"
        ),
    ],
)
def test_wheel_name_uses_the_pep_440_version(tag, expected):
    assert wheel_name(tag) == expected


@pytest.mark.parametrize(
    "tag",
    [
        pytest.param("15.0.0-beta.20", id="missing_v_prefix"),
        pytest.param("v15.0.0", id="stable_release"),
        pytest.param("v15.0.0rc1", id="release_candidate"),
        pytest.param("v15.0.0-rc.1", id="release_candidate_with_hyphen"),
        pytest.param("v15.0.0-beta", id="missing_beta_number"),
        pytest.param("main", id="branch"),
        pytest.param("v15.0.0-beta.20; echo", id="not_a_tag"),
    ],
)
def test_wheel_name_rejects_anything_but_a_beta_tag(tag):
    with pytest.raises(RecoveryError, match="not a beta release tag"):
        wheel_name(tag)


@pytest.mark.parametrize(
    "sha",
    [
        pytest.param(SHA, id="full_sha"),
        pytest.param("0" * 40, id="full_zero_sha"),
    ],
)
def test_check_sha_accepts_full_shas(sha):
    check_sha(sha)


@pytest.mark.parametrize(
    "sha",
    [
        pytest.param("86fff5f40", id="abbreviated"),
        pytest.param(SHA.upper(), id="uppercase"),
        pytest.param(SHA + "0", id="too_long"),
        pytest.param("main", id="branch"),
        pytest.param("", id="empty"),
    ],
)
def test_check_sha_rejects_everything_else(sha):
    with pytest.raises(RecoveryError, match="not a full 40 character"):
        check_sha(sha)


def test_check_release_accepts_a_published_prerelease():
    check_release(dict(RELEASE), TAG)


@pytest.mark.parametrize(
    "release, message",
    [
        pytest.param({**RELEASE, "draft": True}, "still a draft", id="draft"),
        pytest.param(
            {**RELEASE, "prerelease": False}, "is not a pre-release", id="not_beta"
        ),
        pytest.param(
            {**RELEASE, "published_at": None}, "no publication time", id="unpublished"
        ),
        pytest.param(
            {**RELEASE, "tag_name": "v15.0.0-beta.19"},
            "reports tag_name",
            id="mismatch",
        ),
    ],
)
def test_check_release_rejects_unpublished_or_mismatched_releases(release, message):
    with pytest.raises(RecoveryError, match=message):
        check_release(release, TAG)


def test_check_tag_commit_prefers_the_peeled_commit():
    check_tag_commit(LS_REMOTE, TAG, SHA)


def test_check_tag_commit_accepts_a_lightweight_tag():
    ls_remote = f"{SHA}\trefs/tags/v15.0.0-beta.20\n"
    check_tag_commit(ls_remote, TAG, SHA)


def test_check_tag_commit_rejects_the_tag_object_sha():
    ls_remote = LS_REMOTE.splitlines()[0] + "\n"
    with pytest.raises(RecoveryError, match="resolves to ef769e505b77"):
        check_tag_commit(ls_remote, TAG, SHA)


def test_check_tag_commit_rejects_another_commit():
    with pytest.raises(RecoveryError, match="not the expected"):
        check_tag_commit(LS_REMOTE, TAG, "0" * 40)


def test_check_tag_commit_rejects_a_missing_tag():
    with pytest.raises(RecoveryError, match="does not exist on origin"):
        check_tag_commit("", TAG, SHA)


def test_check_wheel_absent_accepts_a_missing_platform():
    check_wheel_absent(INDEX_WITH_OTHER_PLATFORMS, WHEEL, INDEX_URL)


def test_check_wheel_absent_rejects_an_already_published_wheel():
    index_html = (
        INDEX_WITH_OTHER_PLATFORMS
        + f'<a href="https://pypi.fury.io/lance-format/-/ver_3/{WHEEL}#sha256=02">'
        + f"{WHEEL}</a><br/>\n"
    )
    with pytest.raises(RecoveryError, match="already published"):
        check_wheel_absent(index_html, WHEEL, INDEX_URL)


def _touch(wheel_dir, *names):
    for name in names:
        (wheel_dir / name).touch()


def test_check_built_wheel_accepts_the_expected_wheel(tmp_path):
    _touch(tmp_path, WHEEL)
    assert check_built_wheel(str(tmp_path), TAG) == tmp_path / WHEEL


@pytest.mark.parametrize(
    "names, message",
    [
        pytest.param([], "no wheel was built", id="nothing_built"),
        pytest.param(
            [WHEEL, "pylance-15.0.0b20-cp310-abi3-win_amd64.whl"],
            "expected exactly one wheel",
            id="debug_and_release",
        ),
        pytest.param(
            ["pylance-15.0.0b20-cp310-abi3-win_amd64.whl"],
            "built pylance-15.0.0b20-cp310-abi3-win_amd64.whl",
            id="wrong_platform",
        ),
        pytest.param(
            ["pylance-15.0.0b19-cp310-abi3-win_arm64.whl"],
            "built pylance-15.0.0b19-cp310-abi3-win_arm64.whl",
            id="wrong_version",
        ),
    ],
)
def test_check_built_wheel_rejects_anything_else(tmp_path, names, message):
    _touch(tmp_path, *names)
    with pytest.raises(RecoveryError, match=message):
        check_built_wheel(str(tmp_path), TAG)


def test_main_verifies_a_missing_wheel(capsys):
    requests = []

    def get_release(repository, tag):
        requests.append(("release", repository, tag))
        return dict(RELEASE)

    def run_git(*args):
        requests.append(("git", *args))
        return LS_REMOTE

    def get_index(url):
        requests.append(("index", url))
        return INDEX_WITH_OTHER_PLATFORMS

    assert (
        main(
            ["--tag", TAG, "--sha", SHA, "--repository", "lance-format/lance"],
            run_git=run_git,
            get_release=get_release,
            get_index=get_index,
        )
        == 0
    )
    assert requests == [
        ("release", "lance-format/lance", TAG),
        (
            "git",
            "ls-remote",
            "--tags",
            "origin",
            f"refs/tags/{TAG}",
            f"refs/tags/{TAG}^{{}}",
        ),
        ("index", INDEX_URL),
    ]
    assert f"recovery target verified: {WHEEL}" in capsys.readouterr().out


def test_main_stops_before_the_index_when_the_tag_points_elsewhere(capsys):
    def run_git(*args):
        return LS_REMOTE.replace(SHA, "0" * 40)

    def get_index(url):
        raise AssertionError("the published index must not be read")

    assert (
        main(
            ["--tag", TAG, "--sha", SHA, "--repository", "lance-format/lance"],
            run_git=run_git,
            get_release=lambda repository, tag: dict(RELEASE),
            get_index=get_index,
        )
        == 1
    )
    assert "::error::tag v15.0.0-beta.20 resolves to" in capsys.readouterr().out


def test_main_rejects_a_bad_tag_before_any_request():
    def fail(*args, **kwargs):
        raise AssertionError("invalid input must not reach the network")

    args = ["--tag", "v15.0.0-rc.1", "--sha", SHA, "--repository", "lance-format/lance"]
    assert main(args, run_git=fail, get_release=fail, get_index=fail) == 1


def test_main_checks_a_local_wheel_without_reading_the_network(tmp_path, capsys):
    _touch(tmp_path, WHEEL)

    def fail(*args, **kwargs):
        raise AssertionError("--wheel-dir must not read the network")

    args = ["--tag", TAG, "--wheel-dir", str(tmp_path)]
    assert main(args, run_git=fail, get_release=fail, get_index=fail) == 0
    assert f"verified {tmp_path / WHEEL}" in capsys.readouterr().out
