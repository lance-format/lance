#!/usr/bin/env python3
"""
Usage: python ci/generate_release_notes.py <previous_tag> <current_tag>

Generates release notes by comparing two git tags.

This uses the configuration in .github/release.yml to format the release notes.

Format for line is:

* <Title> by @<Author> in <PR Link>

Example output:

* fix: dir namespace cloud storage path removes one subdir level by @jackye1995 in https://github.com/lance-format/lance/pull/5495
* fix: panic unwrap on None in decoder.rs by @camilesing in https://github.com/lance-format/lance/pull/5424
* fix: ensure trailing slash is normalized in rest adapter by @jackye1995 in https://github.com/lance-format/lance/pull/5500

**Full Changelog**: https://github.com/lance-format/lance/compare/v1.0.0...v1.0.1
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import yaml

REPO = "lance-format/lance"
REPO_URL = f"https://github.com/{REPO}"

# Keep a safety buffer below GitHub's 125,000-character Release body hard limit.
DEFAULT_MAX_LENGTH = 120_000
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 5
DEFAULT_INITIAL_BACKOFF_SECONDS = 4.0
DEFAULT_MAX_BACKOFF_SECONDS = 60.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
MAX_LENGTH_ENV_VAR = "RELEASE_NOTES_MAX_LENGTH"

RETRYABLE_GRAPHQL_ERROR_TYPES = frozenset(
    {
        "RATE_LIMITED",
        "SERVICE_UNAVAILABLE",
        "INTERNAL",
    }
)
NON_RETRYABLE_GRAPHQL_ERROR_TYPES = frozenset(
    {
        "PARSE_ERROR",
        "VALIDATION_ERROR",
    }
)
NON_RETRYABLE_PATTERNS = (
    re.compile(r"\bbad credentials\b", re.IGNORECASE),
    re.compile(r"\bhttp\s+401\b", re.IGNORECASE),
    re.compile(r"\bgh auth login\b", re.IGNORECASE),
    re.compile(r"\bcould not resolve to a repository\b", re.IGNORECASE),
)
RETRYABLE_PATTERNS = (
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"abuse detection", re.IGNORECASE),
    re.compile(r"\b(?:http|status)\s*[: ]?\s*429\b", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"\b(?:http|status)\s*[: ]?\s*5\d{2}\b", re.IGNORECASE),
    re.compile(r"bad gateway", re.IGNORECASE),
    re.compile(r"service unavailable", re.IGNORECASE),
    re.compile(r"gateway timeout", re.IGNORECASE),
    re.compile(r"internal server error", re.IGNORECASE),
    re.compile(r"something went wrong", re.IGNORECASE),
    re.compile(r"temporarily unavailable", re.IGNORECASE),
    re.compile(r"try again", re.IGNORECASE),
    re.compile(r"timed?\s*out|timeout", re.IGNORECASE),
    re.compile(r"connection reset", re.IGNORECASE),
    re.compile(r"connection refused", re.IGNORECASE),
    re.compile(r"broken pipe", re.IGNORECASE),
    re.compile(r"unexpected eof|\beof\b", re.IGNORECASE),
    re.compile(r"tls handshake", re.IGNORECASE),
    re.compile(r"dial tcp", re.IGNORECASE),
    re.compile(r"no such host", re.IGNORECASE),
    re.compile(r"temporary failure", re.IGNORECASE),
    re.compile(r"error connecting to", re.IGNORECASE),
    re.compile(r"context deadline exceeded", re.IGNORECASE),
    re.compile(r"stream error", re.IGNORECASE),
    re.compile(r"server closed idle connection", re.IGNORECASE),
)

GraphQLRunner = Callable[[str, dict[str, str]], subprocess.CompletedProcess[str]]


class GitHubAPIError(RuntimeError):
    """Raised when GitHub GraphQL metadata lookup fails after exhausting retries."""


@dataclass
class Category:
    title: str
    labels: list[str]


@dataclass
class ChangelogConfig:
    exclude_labels: list[str]
    categories: list[Category]


@dataclass
class PullRequest:
    number: int
    title: str
    author: str
    labels: list[str]


def parse_repo(repo: str) -> tuple[str, str]:
    """Split an 'owner/name' repository string into its owner and name."""
    owner, sep, name = repo.partition("/")
    if not sep or not owner or not name or "/" in name:
        raise ValueError(f"Invalid repository {repo!r}; expected 'owner/name'")
    return owner, name


def load_config(config_path: str = ".github/release.yml") -> ChangelogConfig:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    changelog = config.get("changelog", {})
    exclude_labels = changelog.get("exclude", {}).get("labels", [])

    categories = []
    for cat in changelog.get("categories", []):
        categories.append(Category(title=cat["title"], labels=cat["labels"]))

    return ChangelogConfig(exclude_labels=exclude_labels, categories=categories)


def get_commits_between_tags(previous_tag: str, current_tag: str) -> list[str]:
    """Get commit subject lines between two git refs."""
    result = subprocess.run(
        ["git", "log", f"{previous_tag}..{current_tag}", "--format=%s"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def extract_pr_number(commit_message: str) -> int | None:
    """Extract PR number from commit message like 'fix: something (#1234)'."""
    match = re.search(r"\(#(\d+)\)", commit_message)
    if match:
        return int(match.group(1))
    return None


def build_pr_batch_query(pr_numbers: Sequence[int]) -> str:
    """Build a GitHub GraphQL query to fetch metadata for a batch of PR numbers."""
    if not pr_numbers:
        raise ValueError("pr_numbers must not be empty")

    selections = []
    for pr_num in pr_numbers:
        if isinstance(pr_num, bool) or not isinstance(pr_num, int) or pr_num <= 0:
            raise ValueError(f"Invalid PR number: {pr_num!r}")
        selections.append(
            f"    pr_{pr_num}: pullRequest(number: {pr_num}) {{\n"
            f"      number\n"
            f"      title\n"
            f"      author {{ __typename login }}\n"
            f"      labels(first: 100) {{ nodes {{ name }} }}\n"
            f"    }}"
        )

    fields = "\n".join(selections)
    return (
        "query($owner: String!, $name: String!) {\n"
        "  repository(owner: $owner, name: $name) {\n"
        f"{fields}\n"
        "  }\n"
        "}"
    )


def run_graphql_query(
    query: str,
    variables: dict[str, str],
    timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    """Execute a GitHub GraphQL query via the gh CLI."""
    payload = json.dumps({"query": query, "variables": variables})
    return subprocess.run(
        ["gh", "api", "graphql", "--input", "-"],
        input=payload,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _is_skippable_pr_error(error: Any, expected_aliases: set[str]) -> bool:
    """Return True if a GraphQL error only indicates a single missing/inaccessible PR."""
    if not isinstance(error, dict):
        return False
    error_type = str(error.get("type") or "").upper()
    if error_type not in {"NOT_FOUND", "FORBIDDEN"}:
        return False
    message = str(error.get("message") or "")
    if _matches_any(message, RETRYABLE_PATTERNS):
        return False
    path = error.get("path")
    return (
        isinstance(path, list)
        and len(path) == 2
        and path[0] == "repository"
        and path[1] in expected_aliases
    )


def _is_retryable_failure(
    payload: dict[str, Any] | None,
    stderr: str,
    raw_stdout: str,
) -> bool:
    """Return True when a GraphQL failure looks like a rate limit or transient error."""
    diagnostic_texts: list[str] = []
    if stderr:
        diagnostic_texts.append(stderr)

    error_types: set[str] = set()
    if payload is not None:
        top_message = payload.get("message")
        if isinstance(top_message, str) and top_message:
            diagnostic_texts.append(top_message)
        errors = payload.get("errors")
        if isinstance(errors, list):
            for err in errors:
                if not isinstance(err, dict):
                    continue
                err_type = str(err.get("type") or "").upper()
                if err_type:
                    error_types.add(err_type)
                err_msg = err.get("message")
                if isinstance(err_msg, str) and err_msg:
                    diagnostic_texts.append(err_msg)
    elif raw_stdout:
        # Only inspect raw stdout when it could not be parsed as JSON (e.g. HTML 502 pages),
        # so PR titles inside GraphQL JSON responses cannot trigger false matches.
        diagnostic_texts.append(raw_stdout)

    if error_types & NON_RETRYABLE_GRAPHQL_ERROR_TYPES:
        return False

    combined = "\n".join(diagnostic_texts)
    if _matches_any(combined, NON_RETRYABLE_PATTERNS):
        return False

    if error_types & RETRYABLE_GRAPHQL_ERROR_TYPES:
        return True

    return _matches_any(combined, RETRYABLE_PATTERNS)


def _format_api_error_detail(
    payload: dict[str, Any] | None,
    stderr: str,
    returncode: int,
) -> str:
    """Build a concise diagnostic string for a failed GraphQL response."""
    parts: list[str] = []
    if payload is not None:
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            error_summaries = []
            for err in errors:
                if isinstance(err, dict):
                    err_type = err.get("type")
                    err_msg = err.get("message") or str(err)
                    if err_type:
                        error_summaries.append(f"{err_type}: {err_msg}")
                    else:
                        error_summaries.append(str(err_msg))
                else:
                    error_summaries.append(str(err))
            parts.append("; ".join(error_summaries))
        top_message = payload.get("message")
        if isinstance(top_message, str) and top_message:
            parts.append(top_message)

    stderr_text = stderr.strip()
    if stderr_text and stderr_text not in parts:
        parts.append(stderr_text)

    if not parts:
        parts.append(f"gh api graphql exited with status {returncode}")

    return " | ".join(parts)


def _parse_pr_node(pr_num: int, node: Any) -> PullRequest:
    if not isinstance(node, dict) or "title" not in node:
        raise GitHubAPIError(
            f"Malformed GraphQL payload for PR #{pr_num}: expected object with 'title'"
        )

    author_info = node.get("author")
    if isinstance(author_info, dict) and author_info.get("login"):
        author = str(author_info["login"])
        if author_info.get("__typename") == "Bot" and not author.endswith("[bot]"):
            author = f"{author}[bot]"
    else:
        author = "ghost"

    labels_conn = node.get("labels") or {}
    label_nodes = labels_conn.get("nodes") if isinstance(labels_conn, dict) else []
    labels = [
        str(label["name"])
        for label in (label_nodes or [])
        if isinstance(label, dict) and label.get("name")
    ]

    return PullRequest(
        number=int(node.get("number", pr_num)),
        title=str(node["title"]),
        author=author,
        labels=labels,
    )


def _format_batch_label(pr_numbers: Sequence[int]) -> str:
    if len(pr_numbers) == 1:
        return f"#{pr_numbers[0]}"
    return f"#{pr_numbers[0]}..#{pr_numbers[-1]} ({len(pr_numbers)} PRs)"


def fetch_pr_batch(
    pr_numbers: Sequence[int],
    *,
    repo: str = REPO,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_backoff: float = DEFAULT_MAX_BACKOFF_SECONDS,
    runner: GraphQLRunner | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[int, PullRequest]:
    """Fetch metadata for one batch of PRs via GraphQL with retry on transient errors."""
    if not pr_numbers:
        return {}
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    if initial_backoff < 0 or max_backoff < 0:
        raise ValueError(
            f"backoff durations must be >= 0, got initial={initial_backoff}, max={max_backoff}"
        )

    if runner is None:
        runner = run_graphql_query
    if sleep_fn is None:
        sleep_fn = time.sleep

    owner, name = parse_repo(repo)
    query = build_pr_batch_query(pr_numbers)
    variables = {"owner": owner, "name": name}
    expected_aliases = {f"pr_{pr_num}" for pr_num in pr_numbers}
    batch_label = _format_batch_label(pr_numbers)

    for attempt in range(max_retries + 1):
        try:
            result = runner(query, variables)
        except subprocess.TimeoutExpired as exc:
            detail = f"request timed out after {exc.timeout}s"
            if attempt < max_retries:
                delay = min(initial_backoff * (2**attempt), max_backoff)
                print(
                    f"Warning: GitHub GraphQL request failed for batch {batch_label} "
                    f"(attempt {attempt + 1}/{max_retries + 1}): {detail}. "
                    f"Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                sleep_fn(delay)
                continue
            raise GitHubAPIError(
                f"Failed to fetch PR metadata for batch {batch_label} "
                f"after {attempt + 1} attempt(s): {detail}"
            ) from exc

        payload: dict[str, Any] | None = None
        if result.stdout and result.stdout.strip():
            try:
                parsed = json.loads(result.stdout)
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = None

        data = payload.get("data") if payload is not None else None
        repo_data = data.get("repository") if isinstance(data, dict) else None
        raw_errors = payload.get("errors") if payload is not None else None
        errors = raw_errors if isinstance(raw_errors, list) else []

        # gh api graphql exits non-zero whenever GraphQL returns an `errors` list,
        # even if all errors are per-PR NOT_FOUND/FORBIDDEN alongside valid PR data.
        has_only_skippable_errors = bool(errors) and all(
            _is_skippable_pr_error(err, expected_aliases) for err in errors
        )
        can_use_repo_data = isinstance(repo_data, dict) and (
            (result.returncode == 0 and not errors) or has_only_skippable_errors
        )

        if can_use_repo_data:
            missing_aliases = [
                alias for alias in expected_aliases if alias not in repo_data
            ]
            if not missing_aliases:
                error_by_alias: dict[str, str] = {}
                for err in errors:
                    path = err.get("path")
                    if isinstance(path, list) and len(path) == 2:
                        error_by_alias[str(path[1])] = str(
                            err.get("message") or err.get("type") or "not found"
                        )

                prs: dict[int, PullRequest] = {}
                for pr_num in pr_numbers:
                    alias = f"pr_{pr_num}"
                    node = repo_data[alias]
                    if node is None:
                        reason = error_by_alias.get(alias)
                        if reason:
                            print(
                                f"Warning: Skipping PR #{pr_num}: {reason}",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                f"Warning: Could not fetch PR #{pr_num}",
                                file=sys.stderr,
                            )
                        continue
                    prs[pr_num] = _parse_pr_node(pr_num, node)
                return prs

            detail = "incomplete GraphQL response; missing aliases: " + ", ".join(
                sorted(missing_aliases)
            )
            retryable = True
        else:
            detail = _format_api_error_detail(payload, result.stderr, result.returncode)
            retryable = _is_retryable_failure(payload, result.stderr, result.stdout)

        if retryable and attempt < max_retries:
            delay = min(initial_backoff * (2**attempt), max_backoff)
            print(
                f"Warning: GitHub GraphQL request failed for batch {batch_label} "
                f"(attempt {attempt + 1}/{max_retries + 1}): {detail}. "
                f"Retrying in {delay:.1f}s...",
                file=sys.stderr,
            )
            sleep_fn(delay)
            continue

        raise GitHubAPIError(
            f"Failed to fetch PR metadata for batch {batch_label} "
            f"after {attempt + 1} attempt(s): {detail}"
        )

    raise GitHubAPIError(f"Failed to fetch PR metadata for batch {batch_label}")


def fetch_prs_details(
    pr_numbers: Iterable[int],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    repo: str = REPO,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_backoff: float = DEFAULT_MAX_BACKOFF_SECONDS,
    runner: GraphQLRunner | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> dict[int, PullRequest]:
    """Fetch metadata for all given PR numbers in GraphQL batches."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    sorted_prs = sorted(set(pr_numbers))
    if not sorted_prs:
        return {}

    fetched: dict[int, PullRequest] = {}
    for start in range(0, len(sorted_prs), batch_size):
        batch = sorted_prs[start : start + batch_size]
        fetched.update(
            fetch_pr_batch(
                batch,
                repo=repo,
                max_retries=max_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                runner=runner,
                sleep_fn=sleep_fn,
            )
        )
    return fetched


def categorize_pr(pr: PullRequest, config: ChangelogConfig) -> str | None:
    """Return category title for a PR, or None if excluded."""
    for label in pr.labels:
        if label in config.exclude_labels:
            return None

    for category in config.categories:
        if "*" in category.labels:
            return category.title
        for label in pr.labels:
            if label in category.labels:
                return category.title

    return None


def format_pr_entry(pr: PullRequest) -> str:
    """Format a single PR entry."""
    return f"* {pr.title} by @{pr.author} in {REPO_URL}/pull/{pr.number}"


def format_truncation_notice(omitted_count: int) -> str:
    """Format the notice emitted when release notes are truncated to fit max_length."""
    noun = "PR" if omitted_count == 1 else "PRs"
    return (
        f"\n_... and {omitted_count} more {noun} omitted to stay within "
        "GitHub's release notes length limit. See the full changelog below._"
    )


def render_release_notes(
    previous_tag: str,
    current_tag: str,
    config: ChangelogConfig,
    categorized: dict[str, list[PullRequest]],
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> str:
    """Render categorized PRs into Markdown, truncating cleanly at entry boundaries."""
    if max_length <= 0:
        raise ValueError(f"max_length must be > 0, got {max_length}")

    header_lines = [
        f"<!-- Release notes generated using configuration in .github/release.yml at {current_tag} -->",
        "",
        "## What's Changed",
    ]
    footer_line = (
        f"\n**Full Changelog**: {REPO_URL}/compare/{previous_tag}...{current_tag}"
    )

    ordered_sections: list[tuple[str, list[str]]] = []
    total_prs = 0
    for category in config.categories:
        prs = categorized.get(category.title, [])
        if prs:
            entries = [
                format_pr_entry(pr) for pr in sorted(prs, key=lambda p: p.number)
            ]
            ordered_sections.append((f"### {category.title}", entries))
            total_prs += len(entries)

    full_lines = list(header_lines)
    for heading, entries in ordered_sections:
        full_lines.append(heading)
        full_lines.extend(entries)
    full_lines.append(footer_line)

    full_output = "\n".join(full_lines)
    if len(full_output) <= max_length:
        return full_output

    if total_prs == 0:
        for candidate_lines in (
            ["## What's Changed", footer_line],
            [footer_line.lstrip("\n")],
        ):
            candidate = "\n".join(candidate_lines)
            if len(candidate) <= max_length:
                return candidate
        min_required = len(footer_line.lstrip("\n"))
        raise ValueError(
            f"max_length ({max_length}) is too small to fit the Full Changelog footer "
            f"(minimum required: {min_required} characters)"
        )

    kept_lines = list(header_lines)
    kept_len = len("\n".join(kept_lines))
    footer_overhead = 1 + len(footer_line)

    included_count = 0
    stop = False
    for heading, entries in ordered_sections:
        heading_emitted = False
        for entry in entries:
            candidate_omitted = total_prs - (included_count + 1)
            # We already know the untruncated output exceeds max_length, so at least
            # one PR must remain omitted to stay within max_length.
            if candidate_omitted == 0:
                stop = True
                break

            addition = [entry] if heading_emitted else [heading, entry]
            addition_len = sum(1 + len(line) for line in addition)
            notice = format_truncation_notice(candidate_omitted)
            candidate_total = (
                kept_len + addition_len + 1 + len(notice) + footer_overhead
            )

            if candidate_total <= max_length:
                kept_lines.extend(addition)
                kept_len += addition_len
                heading_emitted = True
                included_count += 1
            else:
                stop = True
                break
        if stop:
            break

    omitted_count = total_prs - included_count
    notice = format_truncation_notice(omitted_count)

    if included_count > 0:
        truncated_output = "\n".join(kept_lines + [notice, footer_line])
    else:
        truncated_output = ""
        for candidate_lines in (
            header_lines + [notice, footer_line],
            ["## What's Changed", notice, footer_line],
            [notice.lstrip("\n"), footer_line],
        ):
            candidate = "\n".join(candidate_lines)
            if len(candidate) <= max_length:
                truncated_output = candidate
                break
        if not truncated_output:
            min_required = len("\n".join([notice.lstrip("\n"), footer_line]))
            raise ValueError(
                f"max_length ({max_length}) is too small to fit the truncation notice "
                f"and Full Changelog link (minimum required: {min_required} characters)"
            )

    print(
        f"Warning: Release notes exceeded {max_length} characters ({len(full_output)} chars); "
        f"omitted {omitted_count} of {total_prs} PRs (final length: {len(truncated_output)} chars).",
        file=sys.stderr,
    )
    return truncated_output


def generate_release_notes(
    previous_tag: str,
    current_tag: str,
    *,
    config_path: str = ".github/release.yml",
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF_SECONDS,
    max_backoff: float = DEFAULT_MAX_BACKOFF_SECONDS,
    runner: GraphQLRunner | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> str:
    config = load_config(config_path)
    commits = get_commits_between_tags(previous_tag, current_tag)

    pr_numbers = set()
    for commit in commits:
        pr_num = extract_pr_number(commit)
        if pr_num is not None:
            pr_numbers.add(pr_num)

    fetched_prs = fetch_prs_details(
        pr_numbers,
        batch_size=batch_size,
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        runner=runner,
        sleep_fn=sleep_fn,
    )

    categorized: dict[str, list[PullRequest]] = {
        cat.title: [] for cat in config.categories
    }
    for pr_num in sorted(fetched_prs):
        pr = fetched_prs[pr_num]
        category = categorize_pr(pr, config)
        if category:
            categorized[category].append(pr)

    return render_release_notes(
        previous_tag,
        current_tag,
        config,
        categorized,
        max_length=max_length,
    )


def resolve_default_max_length() -> int:
    raw = os.environ.get(MAX_LENGTH_ENV_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_LENGTH
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{MAX_LENGTH_ENV_VAR} must be a positive integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ValueError(f"{MAX_LENGTH_ENV_VAR} must be > 0, got {value}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate release notes by comparing two git tags."
    )
    parser.add_argument("previous_tag", help="Starting git tag or ref (exclusive)")
    parser.add_argument("current_tag", help="Ending git tag or ref (inclusive)")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            f"Maximum output character length (default: ${MAX_LENGTH_ENV_VAR} "
            f"or {DEFAULT_MAX_LENGTH})"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of PRs per GraphQL query batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--config",
        default=".github/release.yml",
        help="Path to release notes configuration YAML (default: .github/release.yml)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        max_length = (
            args.max_length
            if args.max_length is not None
            else resolve_default_max_length()
        )
        notes = generate_release_notes(
            args.previous_tag,
            args.current_tag,
            config_path=args.config,
            max_length=max_length,
            batch_size=args.batch_size,
        )
    except (
        GitHubAPIError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
