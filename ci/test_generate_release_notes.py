"""Unit tests for ci/generate_release_notes.py.

Run with: pytest ci/test_generate_release_notes.py
"""

import json
import subprocess
from pathlib import Path

import pytest
from generate_release_notes import (
    MAX_LENGTH_ENV_VAR,
    REPO_URL,
    Category,
    ChangelogConfig,
    GitHubAPIError,
    PullRequest,
    categorize_pr,
    extract_pr_number,
    fetch_pr_batch,
    fetch_prs_details,
    format_pr_entry,
    format_truncation_notice,
    generate_release_notes,
    load_config,
    main,
    render_release_notes,
)

ROOT = Path(__file__).parent.parent
RELEASE_CONFIG_PATH = str(ROOT / ".github" / "release.yml")


@pytest.fixture(autouse=True)
def _forbid_real_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError(
            f"unit tests must not invoke real subprocess.run: {args!r}"
        )

    monkeypatch.setattr("generate_release_notes.subprocess.run", _unexpected)


def _completed(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["gh", "api", "graphql", "--input", "-"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _pr_node(
    number: int,
    title: str | None = None,
    author: dict | None = None,
    labels: list[str] | None = None,
    *,
    author_missing: bool = False,
) -> dict:
    if author_missing:
        author_value = None
    elif author is not None:
        author_value = author
    else:
        author_value = {"__typename": "User", "login": f"user{number}"}

    return {
        "number": number,
        "title": title if title is not None else f"feat: change {number}",
        "author": author_value,
        "labels": {"nodes": [{"name": name} for name in (labels or [])]},
    }


def _graphql_response(
    repo_nodes: dict[str, dict | None],
    errors: list[dict] | None = None,
    *,
    returncode: int | None = None,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    body: dict = {"data": {"repository": repo_nodes}}
    if errors is not None:
        body["errors"] = errors
    if returncode is None:
        returncode = 1 if errors else 0
    return _completed(stdout=json.dumps(body), stderr=stderr, returncode=returncode)


def test_fetch_prs_details_batches_deduplicates_and_parses_authors():
    calls: list[tuple[str, dict[str, str]]] = []

    def runner(
        query: str, variables: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        calls.append((query, variables))
        if len(calls) == 1:
            assert "pr_10:" in query and "pr_20:" in query and "pr_30:" not in query
            return _graphql_response(
                {
                    "pr_10": _pr_node(
                        10,
                        title="fix: user pr",
                        author={"__typename": "User", "login": "alice"},
                        labels=["bug", "A-rust"],
                    ),
                    "pr_20": _pr_node(
                        20,
                        title="chore: bot pr",
                        author={"__typename": "Bot", "login": "lance-gatefixer"},
                        labels=["chore"],
                    ),
                }
            )
        assert "pr_30:" in query and "pr_40:" in query
        return _graphql_response(
            {
                "pr_30": _pr_node(
                    30,
                    title="docs: deleted author",
                    author_missing=True,
                    labels=["documentation"],
                ),
                "pr_40": _pr_node(
                    40,
                    title="feat: pre-suffixed bot",
                    author={"__typename": "Bot", "login": "dependabot[bot]"},
                    labels=["enhancement"],
                ),
            }
        )

    prs = fetch_prs_details(
        [40, 10, 20, 10, 30],
        batch_size=2,
        repo="org/repo",
        runner=runner,
        sleep_fn=lambda _: (_ for _ in ()).throw(AssertionError("unexpected sleep")),
    )

    assert len(calls) == 2
    assert all(variables == {"owner": "org", "name": "repo"} for _, variables in calls)
    assert prs == {
        10: PullRequest(10, "fix: user pr", "alice", ["bug", "A-rust"]),
        20: PullRequest(20, "chore: bot pr", "lance-gatefixer[bot]", ["chore"]),
        30: PullRequest(30, "docs: deleted author", "ghost", ["documentation"]),
        40: PullRequest(
            40, "feat: pre-suffixed bot", "dependabot[bot]", ["enhancement"]
        ),
    }


def test_fetch_prs_details_empty_input_makes_no_api_calls():
    def fail_runner(query: str, variables: dict[str, str]):
        raise AssertionError("GraphQL runner must not be called for empty PR list")

    assert fetch_prs_details([], runner=fail_runner) == {}


@pytest.mark.parametrize(
    ("error_type", "message"),
    [
        pytest.param(
            "NOT_FOUND",
            "Could not resolve to a PullRequest with the number of 999.",
            id="not_found_issue_or_missing_pr",
        ),
        pytest.param(
            "FORBIDDEN",
            "Resource not accessible by integration",
            id="forbidden_single_pr",
        ),
    ],
)
def test_fetch_pr_batch_skips_missing_or_inaccessible_pr_without_retrying(
    error_type: str,
    message: str,
    capsys: pytest.CaptureFixture[str],
):
    sleeps: list[float] = []
    calls = 0

    def runner(
        query: str, variables: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return _graphql_response(
            {
                "pr_100": _pr_node(100, title="feat: valid", labels=["enhancement"]),
                "pr_999": None,
                "pr_101": _pr_node(101, title="fix: valid", labels=["bug"]),
            },
            errors=[
                {
                    "type": error_type,
                    "path": ["repository", "pr_999"],
                    "message": message,
                }
            ],
            returncode=1,
            stderr=f"gh: {message}",
        )

    prs = fetch_pr_batch(
        [100, 999, 101],
        runner=runner,
        sleep_fn=sleeps.append,
    )

    assert calls == 1
    assert sleeps == []
    assert set(prs.keys()) == {100, 101}
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Skipping PR #999" in captured.err


def test_fetch_pr_batch_retries_when_not_found_cooccurs_with_transient_error():
    sleeps: list[float] = []
    responses = [
        _graphql_response(
            {
                "pr_100": None,
                "pr_999": None,
            },
            errors=[
                {
                    "type": "NOT_FOUND",
                    "path": ["repository", "pr_999"],
                    "message": "Could not resolve to a PullRequest with the number of 999.",
                },
                {
                    "type": "SERVICE_UNAVAILABLE",
                    "path": ["repository", "pr_100"],
                    "message": "Service temporarily unavailable",
                },
            ],
            returncode=1,
        ),
        _graphql_response(
            {
                "pr_100": _pr_node(100, title="feat: recovered"),
                "pr_999": None,
            },
            errors=[
                {
                    "type": "NOT_FOUND",
                    "path": ["repository", "pr_999"],
                    "message": "Could not resolve to a PullRequest with the number of 999.",
                }
            ],
            returncode=1,
        ),
    ]

    def runner(
        query: str, variables: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return responses.pop(0)

    prs = fetch_pr_batch(
        [100, 999],
        initial_backoff=2.0,
        max_backoff=10.0,
        runner=runner,
        sleep_fn=sleeps.append,
    )

    assert sleeps == [2.0]
    assert list(prs.keys()) == [100]
    assert prs[100].title == "feat: recovered"


def test_fetch_pr_batch_retries_and_fails_on_incomplete_alias_response():
    sleeps: list[float] = []

    def runner(
        query: str, variables: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        # pr_101 is omitted entirely rather than explicitly null
        return _graphql_response({"pr_100": _pr_node(100)}, returncode=0)

    with pytest.raises(GitHubAPIError, match="missing aliases: pr_101"):
        fetch_pr_batch(
            [100, 101],
            max_retries=2,
            initial_backoff=1.0,
            max_backoff=10.0,
            runner=runner,
            sleep_fn=sleeps.append,
        )

    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(
            _completed(
                stdout=json.dumps(
                    {
                        "errors": [
                            {
                                "type": "RATE_LIMITED",
                                "message": "API rate limit exceeded",
                            }
                        ]
                    }
                ),
                stderr="gh: API rate limit exceeded",
                returncode=1,
            ),
            id="graphql_rate_limited",
        ),
        pytest.param(
            _completed(
                stdout=json.dumps(
                    {
                        "message": "You have exceeded a secondary rate limit. Please wait a few minutes before you try again."
                    }
                ),
                stderr="gh: You have exceeded a secondary rate limit (HTTP 403)",
                returncode=1,
            ),
            id="http_403_secondary_rate_limit",
        ),
        pytest.param(
            _completed(
                stdout="",
                stderr="gh: Too Many Requests (HTTP 429)",
                returncode=1,
            ),
            id="http_429",
        ),
        pytest.param(
            _completed(
                stdout="<html><body><h1>502 Bad Gateway</h1></body></html>",
                stderr="gh: Bad Gateway (HTTP 502)",
                returncode=1,
            ),
            id="http_502_html",
        ),
        pytest.param(
            _completed(
                stdout="",
                stderr="error connecting to api.github.com: read tcp: connection reset by peer",
                returncode=1,
            ),
            id="network_connection_reset",
        ),
        pytest.param(
            _graphql_response(
                {"pr_100": None},
                errors=[
                    {
                        "type": "FORBIDDEN",
                        "path": ["repository", "pr_100"],
                        "message": "You have exceeded a secondary rate limit",
                    }
                ],
                returncode=1,
            ),
            id="forbidden_with_rate_limit_message",
        ),
        pytest.param(
            subprocess.TimeoutExpired(cmd=["gh", "api", "graphql"], timeout=60),
            id="subprocess_timeout",
        ),
    ],
)
def test_fetch_pr_batch_retries_transient_errors_with_exponential_backoff(failure):
    sleeps: list[float] = []
    attempts = 0

    def runner(
        query: str, variables: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            if isinstance(failure, Exception):
                raise failure
            return failure
        return _graphql_response({"pr_100": _pr_node(100, title="fix: after retries")})

    prs = fetch_pr_batch(
        [100],
        max_retries=4,
        initial_backoff=2.0,
        max_backoff=5.0,
        runner=runner,
        sleep_fn=sleeps.append,
    )

    assert attempts == 4
    assert sleeps == [2.0, 4.0, 5.0]
    assert prs[100].title == "fix: after retries"


def test_fetch_pr_batch_raises_after_exhausting_retries():
    sleeps: list[float] = []

    def runner(
        query: str, variables: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return _completed(
            stderr="gh: Service Unavailable (HTTP 503)",
            returncode=1,
        )

    with pytest.raises(GitHubAPIError, match="after 3 attempt\\(s\\).*HTTP 503"):
        fetch_pr_batch(
            [10, 11],
            max_retries=2,
            initial_backoff=1.0,
            max_backoff=10.0,
            runner=runner,
            sleep_fn=sleeps.append,
        )

    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(
            _completed(stderr="gh: Bad credentials (HTTP 401)", returncode=1),
            id="http_401_bad_credentials",
        ),
        pytest.param(
            _completed(
                stdout=json.dumps(
                    {
                        "errors": [
                            {
                                "type": "NOT_FOUND",
                                "path": ["repository"],
                                "message": "Could not resolve to a Repository with the name 'org/missing'.",
                            }
                        ]
                    }
                ),
                stderr="gh: Could not resolve to a Repository",
                returncode=1,
            ),
            id="missing_repository",
        ),
        pytest.param(
            _completed(
                stdout=json.dumps(
                    {
                        "errors": [
                            {
                                "type": "VALIDATION_ERROR",
                                "message": "Field 'bad' doesn't exist on type 'PullRequest'",
                            }
                        ]
                    }
                ),
                returncode=1,
            ),
            id="graphql_validation_error",
        ),
    ],
)
def test_fetch_pr_batch_fails_fast_on_non_retryable_errors(failure):
    sleeps: list[float] = []
    calls = 0

    def runner(
        query: str, variables: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return failure

    with pytest.raises(GitHubAPIError, match="after 1 attempt\\(s\\)"):
        fetch_pr_batch(
            [100],
            max_retries=3,
            initial_backoff=2.0,
            runner=runner,
            sleep_fn=sleeps.append,
        )

    assert calls == 1
    assert sleeps == []


def test_categorization_sorting_and_markdown_format_with_repo_config():
    config = load_config(RELEASE_CONFIG_PATH)
    prs = [
        PullRequest(300, "chore: bump deps", "bot[bot]", ["chore"]),
        PullRequest(301, "ci: tweak job", "alice", ["ci", "bug"]),
        PullRequest(205, "feat!: drop v1", "bob", ["breaking-change", "bug"]),
        PullRequest(201, "feat!: earlier breaking", "alice", ["breaking-change"]),
        PullRequest(210, "fix: crash", "carol", ["bug"]),
        PullRequest(220, "refactor: clean up", "dave", ["A-rust"]),
    ]

    categorized: dict[str, list[PullRequest]] = {
        cat.title: [] for cat in config.categories
    }
    for pr in prs:
        category = categorize_pr(pr, config)
        if category is not None:
            categorized[category].append(pr)

    output = render_release_notes("v14.0.0", "v15.0.0-rc.1", config, categorized)

    expected = "\n".join(
        [
            "<!-- Release notes generated using configuration in .github/release.yml at v15.0.0-rc.1 -->",
            "",
            "## What's Changed",
            "### Breaking Changes 🛠",
            f"* feat!: earlier breaking by @alice in {REPO_URL}/pull/201",
            f"* feat!: drop v1 by @bob in {REPO_URL}/pull/205",
            "### Bug Fixes 🐛",
            f"* fix: crash by @carol in {REPO_URL}/pull/210",
            "### Other Changes",
            f"* refactor: clean up by @dave in {REPO_URL}/pull/220",
            f"\n**Full Changelog**: {REPO_URL}/compare/v14.0.0...v15.0.0-rc.1",
        ]
    )
    assert output == expected


def _sample_config_and_prs() -> tuple[ChangelogConfig, dict[str, list[PullRequest]]]:
    config = ChangelogConfig(
        exclude_labels=["ci", "chore"],
        categories=[
            Category("Breaking Changes", ["breaking-change"]),
            Category("Bug Fixes", ["bug"]),
            Category("Other Changes", ["*"]),
        ],
    )
    categorized = {
        "Breaking Changes": [
            PullRequest(1, "feat!: breaking change one", "alice", ["breaking-change"]),
            PullRequest(2, "feat!: breaking change two", "bob", ["breaking-change"]),
        ],
        "Bug Fixes": [
            PullRequest(3, "fix: bug fix one", "carol", ["bug"]),
            PullRequest(4, "fix: bug fix two", "dave", ["bug"]),
        ],
        "Other Changes": [
            PullRequest(5, "refactor: other change one", "eve", []),
            PullRequest(6, "refactor: other change two", "frank", []),
        ],
    }
    return config, categorized


def test_render_release_notes_exact_max_length_does_not_truncate():
    config, categorized = _sample_config_and_prs()
    full = render_release_notes("v14.0.0", "v15.0.0", config, categorized)

    exact = render_release_notes(
        "v14.0.0", "v15.0.0", config, categorized, max_length=len(full)
    )
    assert exact == full
    assert "omitted" not in exact


def test_render_release_notes_truncates_cleanly_at_entry_boundaries(
    capsys: pytest.CaptureFixture[str],
):
    config, categorized = _sample_config_and_prs()
    full = render_release_notes("v14.0.0", "v15.0.0", config, categorized)
    footer = f"**Full Changelog**: {REPO_URL}/compare/v14.0.0...v15.0.0"

    # Sweep tight budgets from one char below full down to where only 1 PR fits.
    all_entries = {format_pr_entry(pr) for prs in categorized.values() for pr in prs}
    for budget in (len(full) - 1, len(full) - 150, len(full) - 300, len(full) - 450):
        truncated = render_release_notes(
            "v14.0.0", "v15.0.0", config, categorized, max_length=budget
        )
        assert len(truncated) <= budget
        assert truncated.endswith(footer)
        assert "omitted to stay within GitHub's release notes length limit" in truncated

        # Every emitted bullet line must be an intact, un-sliced PR entry.
        bullet_lines = [
            line for line in truncated.splitlines() if line.startswith("* ")
        ]
        assert len(bullet_lines) < 6
        assert all(line in all_entries for line in bullet_lines)

        # Omitted count must equal total PRs minus emitted entries.
        expected_omitted = 6 - len(bullet_lines)
        expected_notice = format_truncation_notice(expected_omitted).strip()
        assert expected_notice in truncated

        # No category heading may be orphaned without at least one entry after it.
        lines = truncated.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("### "):
                assert idx + 1 < len(lines) and lines[idx + 1].startswith("* ")

    assert "Warning: Release notes exceeded" in capsys.readouterr().err


def test_render_release_notes_avoids_orphan_heading_and_preserves_priority_order():
    config, categorized = _sample_config_and_prs()
    footer = f"\n**Full Changelog**: {REPO_URL}/compare/v14.0.0...v15.0.0"
    header = [
        "<!-- Release notes generated using configuration in .github/release.yml at v15.0.0 -->",
        "",
        "## What's Changed",
        "### Breaking Changes",
        format_pr_entry(categorized["Breaking Changes"][0]),
        format_pr_entry(categorized["Breaking Changes"][1]),
    ]
    notice_4 = format_truncation_notice(4)
    # Budget fits Breaking Changes + "### Bug Fixes" heading alone, but NOT its first PR entry.
    exact_with_two_prs = len("\n".join(header + [notice_4, footer]))
    budget = exact_with_two_prs + len("\n### Bug Fixes") + 10

    truncated = render_release_notes(
        "v14.0.0", "v15.0.0", config, categorized, max_length=budget
    )

    assert len(truncated) <= budget
    assert "### Breaking Changes" in truncated
    assert "### Bug Fixes" not in truncated
    assert "### Other Changes" not in truncated
    assert "4 more PRs omitted" in truncated
    assert truncated.endswith(footer.lstrip("\n"))


def test_render_release_notes_singular_omitted_pr_notice():
    config = ChangelogConfig(
        exclude_labels=[],
        categories=[Category("Other Changes", ["*"])],
    )
    categorized = {
        "Other Changes": [
            PullRequest(1, "short one", "alice", []),
            PullRequest(2, "x" * 200, "bob", []),
        ]
    }
    full = render_release_notes("v1.0.0", "v1.0.1", config, categorized)
    budget = len(full) - 20

    truncated = render_release_notes(
        "v1.0.0", "v1.0.1", config, categorized, max_length=budget
    )
    assert len(truncated) <= budget
    assert "1 more PR omitted" in truncated
    assert "1 more PRs" not in truncated


def test_render_release_notes_zero_prs_fit_still_preserves_notice_and_footer():
    config, categorized = _sample_config_and_prs()
    footer = f"**Full Changelog**: {REPO_URL}/compare/v14.0.0...v15.0.0"
    notice = format_truncation_notice(6).lstrip("\n")
    min_required = len(f"{notice}\n\n{footer}")

    truncated = render_release_notes(
        "v14.0.0", "v15.0.0", config, categorized, max_length=min_required
    )
    assert len(truncated) <= min_required
    assert "6 more PRs omitted" in truncated
    assert truncated.endswith(footer)

    with pytest.raises(ValueError, match="too small"):
        render_release_notes(
            "v14.0.0", "v15.0.0", config, categorized, max_length=min_required - 1
        )


def test_render_release_notes_rejects_non_positive_max_length():
    config, categorized = _sample_config_and_prs()
    with pytest.raises(ValueError, match="max_length must be > 0"):
        render_release_notes("v14.0.0", "v15.0.0", config, categorized, max_length=0)


def test_main_respects_env_and_cli_max_length_and_propagates_api_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        "generate_release_notes.get_commits_between_tags",
        lambda prev, curr: [
            "feat: first (#10)",
            "fix: second (#11)",
            "chore: release beta",
        ],
    )
    monkeypatch.setattr(
        "generate_release_notes.run_graphql_query",
        lambda query, variables: _graphql_response(
            {
                "pr_10": _pr_node(10, title="feat: first", labels=["enhancement"]),
                "pr_11": _pr_node(11, title="fix: second", labels=["bug"]),
            }
        ),
    )

    # Env var forces truncation; stdout stays pure Markdown and warnings go to stderr.
    monkeypatch.setenv(MAX_LENGTH_ENV_VAR, "320")
    rc = main(["--config", RELEASE_CONFIG_PATH, "v14.0.0", "v15.0.0"])
    captured = capsys.readouterr()
    assert rc == 0
    assert len(captured.out.rstrip("\n")) <= 320
    assert "**Full Changelog**:" in captured.out
    assert "Warning: Release notes exceeded 320 characters" in captured.err

    # CLI --max-length overrides env var.
    rc = main(
        ["--config", RELEASE_CONFIG_PATH, "--max-length", "10000", "v14.0.0", "v15.0.0"]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "omitted" not in captured.out
    assert "feat: first" in captured.out and "fix: second" in captured.out

    # Invalid env var fails cleanly with non-zero exit code.
    monkeypatch.setenv(MAX_LENGTH_ENV_VAR, "not-a-number")
    rc = main(["--config", RELEASE_CONFIG_PATH, "v14.0.0", "v15.0.0"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert MAX_LENGTH_ENV_VAR in captured.err

    # Exhausted GraphQL failure exits non-zero rather than emitting partial release notes.
    monkeypatch.delenv(MAX_LENGTH_ENV_VAR, raising=False)
    monkeypatch.setattr(
        "generate_release_notes.run_graphql_query",
        lambda query, variables: _completed(
            stderr="gh: Bad credentials (HTTP 401)", returncode=1
        ),
    )
    rc = main(["--config", RELEASE_CONFIG_PATH, "v14.0.0", "v15.0.0"])
    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert "Error: Failed to fetch PR metadata" in captured.err


def test_generate_release_notes_end_to_end_skips_missing_and_excludes_labels(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(
        "generate_release_notes.get_commits_between_tags",
        lambda prev, curr: [
            "feat: keep me (#50)",
            "chore: skip me (#51)",
            "fix: references issue (#52)",
        ],
    )

    def runner(
        query: str, variables: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        return _graphql_response(
            {
                "pr_50": _pr_node(50, title="feat: keep me", labels=["enhancement"]),
                "pr_51": _pr_node(51, title="chore: skip me", labels=["chore"]),
                "pr_52": None,
            },
            errors=[
                {
                    "type": "NOT_FOUND",
                    "path": ["repository", "pr_52"],
                    "message": "Could not resolve to a PullRequest with the number of 52.",
                }
            ],
            returncode=1,
        )

    notes = generate_release_notes(
        "v14.0.0",
        "v15.0.0",
        config_path=RELEASE_CONFIG_PATH,
        runner=runner,
    )

    assert "feat: keep me" in notes
    assert "chore: skip me" not in notes
    assert "#52" not in notes
    assert f"**Full Changelog**: {REPO_URL}/compare/v14.0.0...v15.0.0" in notes
    assert "Skipping PR #52" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        pytest.param("fix: something (#1234)", 1234, id="standard_squash_merge"),
        pytest.param("chore: release beta version 15.0.0-beta.25", None, id="no_pr"),
    ],
)
def test_extract_pr_number(message: str, expected: int | None):
    assert extract_pr_number(message) == expected
