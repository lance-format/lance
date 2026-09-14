"""Format-specification vote gate (see `.github/workflows/format-vote-gate.yml`).

Structurally enforces the PMC vote required for Lance format-specification
changes (https://lance.org/community/voting/). The `format-change` label is
applied by the path labeler (`.github/labeler-area.yml`); this script reads it
and publishes the `format-spec-vote` commit status, which blocks merging until:

  * 3 PMC members have approved the PR (excluding the author), counted only on
    the head commit so new pushes invalidate stale approvals;
  * no PMC member has an outstanding "Request changes" review (a veto); and
  * the 72-hour voting period has elapsed. The clock starts once the PR is both
    labeled and out of draft, and pauses over weekends.

A PMC member can waive a trivial edit by applying the `format-waived` label.
Non-format PRs get a passing status immediately and are otherwise left alone.
Drafts get a blocking status but no comment: the vote has not opened yet.

The vote-counting and deadline rules are pure functions (`tally_reviews`,
`decide_verdict`, `vote_opened_at`, `weekday_deadline`) unit tested in
`test_format_vote_gate.py`; `main` wires them to the GitHub API.
"""

import json
import os
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

STATUS_CONTEXT = "format-spec-vote"
FORMAT_LABEL = "format-change"
WAIVED_LABEL = "format-waived"
COMMENT_MARKER = "<!-- format-spec-vote-status -->"
REQUIRED_APPROVALS = 3
PERIOD_HOURS = 72
VOTING_URL = "https://lance.org/community/voting/"
WORKFLOW_FILE = "format-vote-gate.yml"
# Keep in sync with the `cron` in the workflow; only used in the comment text.
SWEEP_MINUTES = 15

# The weekend boundary is fixed in UTC rather than a local zone: it has no DST
# transitions to reason about, and no PMC member's timezone gets to define when
# everyone else's clock pauses. Deadlines are *displayed* in UTC and Pacific.
WEEKEND_TZ = timezone.utc
DISPLAY_TZ = ZoneInfo("America/Los_Angeles")

# datetime.weekday() numbers Monday 0 .. Sunday 6, so the weekend is >= 5.
_SATURDAY = 5

# Review states that express a stance; COMMENTED/PENDING are ignored.
_STANCE_STATES = ("APPROVED", "CHANGES_REQUESTED", "DISMISSED")

TimelineFacts = namedtuple("TimelineFacts", "labeled_at waived ready_at")


def tally_reviews(reviews, head_sha, author, is_pmc):
    """Tally PMC votes from a PR's reviews.

    `reviews` is an ordered list of dicts with `login`, `state`, `commit_id`.
    A member's stance is their most recent stance review. Approvals only count
    on the head commit; earlier ones are stale. A "changes requested" review is
    a veto regardless of commit. The PR author never counts.
    """
    latest = {}
    for review in reviews:
        login = review["login"]
        if not login or not is_pmc(login) or login == author:
            continue
        if review["state"] not in _STANCE_STATES:
            continue
        latest[login.lower()] = review

    approvals, stale_approvals, vetoes = [], [], []
    for review in latest.values():
        if review["state"] == "APPROVED":
            target = approvals if review["commit_id"] == head_sha else stale_approvals
            target.append(review["login"])
        elif review["state"] == "CHANGES_REQUESTED":
            vetoes.append(review["login"])
    return approvals, stale_approvals, vetoes


def decide_verdict(veto_count, approval_count, period_elapsed, required):
    """Return the blocking condition (if any), in priority order."""
    if veto_count > 0:
        return "veto"
    if approval_count < required:
        return "insufficient"
    if not period_elapsed:
        return "waiting_period"
    return "pass"


def vote_opened_at(labeled_at, ready_at):
    """When the voting period starts, or None if it hasn't.

    A vote opens only once the proposal is both identified as a format change
    and offered for review, so the clock starts at the later of the two. A draft
    is still being drafted; time spent there shouldn't count toward the period.
    """
    if labeled_at is None or ready_at is None:
        return None
    return max(labeled_at, ready_at)


def _start_of_day(dt):
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


# Advance to Monday 00:00 if `dt` lands on a weekend; otherwise leave it alone.
def _skip_weekend(dt):
    while dt.weekday() >= _SATURDAY:
        dt = _start_of_day(dt) + timedelta(days=1)
    return dt


# Saturday 00:00 following `dt`, which must already be a weekday.
def _next_weekend(dt):
    return _start_of_day(dt) + timedelta(days=_SATURDAY - dt.weekday())


def weekday_deadline(start, hours):
    """When `hours` of non-weekend time have elapsed after `start`.

    Weekends don't count toward the voting period, so a proposal opened on a
    Friday afternoon doesn't burn most of its period while nobody is reading it.
    Both `start` and the result are aware datetimes; the arithmetic happens in
    `WEEKEND_TZ`, which decides where each weekend begins and ends.
    """
    cursor = _skip_weekend(start.astimezone(WEEKEND_TZ))
    remaining = timedelta(hours=hours)
    while True:
        until_weekend = _next_weekend(cursor) - cursor
        if remaining <= until_weekend:
            return cursor + remaining
        remaining -= until_weekend
        cursor = _skip_weekend(_next_weekend(cursor))


def _fmt_list(logins):
    return ", ".join(f"@{login}" for login in logins) if logins else "none"


# Renders as `Wed 2026-08-05 17:00 UTC (10:00 PDT)` — the PMC spans both zones.
# The Pacific weekday is spelled out only when the deadline falls on a different
# day there, which is the case that actually trips people up.
def _fmt_deadline(dt):
    local = dt.astimezone(DISPLAY_TZ)
    local_day = "" if local.date() == dt.date() else f"{local:%a }"
    return f"{dt:%a %Y-%m-%d %H:%M} UTC ({local_day}{local:%H:%M %Z})"


def _as_utc(dt):
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _build_comment(headline, approval_cell, vetoes, period_cell, rerun_url):
    return "\n".join(
        [
            COMMENT_MARKER,
            "> [!IMPORTANT]",
            "> ## Format specification vote",
            "",
            "This PR modifies the Lance format specification, so it requires "
            f"**{REQUIRED_APPROVALS} binding +1 votes from PMC members** "
            "(excluding the proposer) and a minimum "
            f"**{PERIOD_HOURS}-hour** voting period, weekends excluded, before "
            "it can merge. "
            "Vote by approving this PR (+1) or requesting changes (−1, a veto). "
            f"See the [voting process]({VOTING_URL}).",
            "",
            f"**Status: {headline}**",
            "",
            "| | |",
            "|---|---|",
            f"| Approvals (this commit) | {approval_cell} |",
            f"| Vetoes | {_fmt_list(vetoes)} |",
            f"| Voting period | {period_cell} |",
            "",
            "<sub>Updated automatically by the format-spec vote gate, which "
            f"re-checks every {SWEEP_MINUTES} minutes — just voted? "
            f"[Re-check now]({rerun_url}) (press Run workflow; leave the input "
            "blank to re-check every open format PR). A PMC member may apply "
            f"the `{WAIVED_LABEL}` label to waive the vote for a trivial edit "
            "(typo, wording, formatting).</sub>",
        ]
    )


def _load_pmc(workspace):
    import yaml

    roster_path = os.path.join(workspace, "docs", "src", "community", "pmc.yaml")
    with open(roster_path) as handle:
        roster = yaml.safe_load(handle)
    return {member["handle"].lower() for member in roster["members"]}


class Gate:
    def __init__(self, repo, pmc, run_url, rerun_url):
        self.repo = repo
        self.pmc = pmc
        self.run_url = run_url
        self.rerun_url = rerun_url

    def is_pmc(self, login):
        return login is not None and login.lower() in self.pmc

    def set_status(self, sha, state, description):
        self.repo.get_commit(sha).create_status(
            state=state,
            context=STATUS_CONTEXT,
            description=description[:140],
            target_url=self.run_url,
        )

    def upsert_comment(self, issue, body):
        for comment in issue.get_comments():
            if COMMENT_MARKER in (comment.body or ""):
                if comment.body != body:
                    comment.edit(body)
                return
        issue.create_comment(body)

    def timeline_facts(self, issue):
        """Read the vote-clock inputs off the PR timeline in one pass."""
        labeled_at = None
        waived_by_pmc = False
        ready_at = None
        for event in issue.get_events():
            actor = event.actor.login if event.actor else None
            # A PR converted back to draft and re-opened for review restarts the
            # clock, so the *last* ready_for_review wins.
            if event.event == "ready_for_review":
                ready_at = _as_utc(event.created_at)
                continue
            if event.event != "labeled" or event.label is None:
                continue
            if event.label.name == FORMAT_LABEL and labeled_at is None:
                labeled_at = _as_utc(event.created_at)
            elif event.label.name == WAIVED_LABEL and self.is_pmc(actor):
                waived_by_pmc = True
        return TimelineFacts(labeled_at, waived_by_pmc, ready_at)

    def evaluate(self, number):
        pr = self.repo.get_pull(number)
        if pr.state != "open":
            print(f"PR #{number} is {pr.state}; skipping.")
            return
        head_sha = pr.head.sha
        labels = {label.name for label in pr.labels}

        # Non-format PRs get a passing status and are otherwise left alone.
        if FORMAT_LABEL not in labels:
            self.set_status(
                head_sha, "success", "No format-spec change; vote not required."
            )
            print(f"PR #{number}: not a format change.")
            return

        issue = self.repo.get_issue(number)
        facts = self.timeline_facts(issue)

        if WAIVED_LABEL in labels and facts.waived:
            self.set_status(
                head_sha, "success", "Format-spec vote waived by a PMC member."
            )
            print(f"PR #{number}: vote waived.")
            return

        # Stay quiet on drafts: the proposal isn't up for a vote yet, so there is
        # nothing for the PMC to act on and no deadline to announce.
        if pr.draft:
            self.set_status(
                head_sha,
                "failure",
                "Draft; voting period starts when marked ready for review.",
            )
            print(f"PR #{number}: draft, vote not open.")
            return

        reviews = [
            {
                "login": review.user.login if review.user else None,
                "state": review.state,
                "commit_id": review.commit_id,
            }
            for review in pr.get_reviews()
        ]
        approvals, stale, vetoes = tally_reviews(
            reviews, head_sha, pr.user.login, self.is_pmc
        )

        now = datetime.now(timezone.utc)
        # `pr.created_at` covers a PR opened ready for review, which never emits a
        # ready_for_review event.
        opened_at = vote_opened_at(
            facts.labeled_at, facts.ready_at or _as_utc(pr.created_at)
        )
        period_ends = weekday_deadline(opened_at or now, PERIOD_HOURS)
        period_elapsed = now >= period_ends
        verdict = decide_verdict(
            len(vetoes), len(approvals), period_elapsed, REQUIRED_APPROVALS
        )

        deadline = _fmt_deadline(period_ends)
        if verdict == "veto":
            state, summary = "failure", f"Vetoed by {len(vetoes)} PMC member(s)."
            headline = f"❌ Blocked — vetoed by {_fmt_list(vetoes)}"
        elif verdict == "insufficient":
            state = "failure"
            summary = (
                f"{len(approvals)}/{REQUIRED_APPROVALS} PMC approvals on this commit."
            )
            headline = f"❌ Blocked — {len(approvals)} of {REQUIRED_APPROVALS} required approvals"
        elif verdict == "waiting_period":
            state, summary = "failure", f"Approved; voting period ends {deadline}."
            headline = (
                f"⏳ Approvals met ({len(approvals)}/{REQUIRED_APPROVALS}); "
                f"voting period ends {deadline}"
            )
        else:
            state = "success"
            summary = f"Passed — {len(approvals)} PMC approvals, period elapsed."
            headline = f"✅ Vote passed — {len(approvals)} PMC approvals, voting period elapsed"

        period_cell = (
            f"elapsed — ended {deadline}" if period_elapsed else f"ends {deadline}"
        )
        approval_cell = (
            f"{_fmt_list(approvals)} ({len(approvals)}/{REQUIRED_APPROVALS})"
        )
        if stale:
            approval_cell += f" — stale, re-approve needed: {_fmt_list(stale)}"

        self.set_status(head_sha, state, summary)
        self.upsert_comment(
            issue,
            _build_comment(
                headline, approval_cell, vetoes, period_cell, self.rerun_url
            ),
        )
        print(f"PR #{number}: {summary}")


def main():
    from github import Github

    workspace = os.environ["GITHUB_WORKSPACE"]
    token = os.environ["GITHUB_TOKEN"]
    repo_name = os.environ["GITHUB_REPOSITORY"]
    event_name = os.environ["GITHUB_EVENT_NAME"]
    actions_url = f"{os.environ['GITHUB_SERVER_URL']}/{repo_name}/actions"
    run_url = f"{actions_url}/runs/{os.environ['GITHUB_RUN_ID']}"
    rerun_url = f"{actions_url}/workflows/{WORKFLOW_FILE}"

    repo = Github(token).get_repo(repo_name)
    gate = Gate(repo, _load_pmc(workspace), run_url, rerun_url)
    print(f"Re-check on demand: {rerun_url}")

    if event_name in ("schedule", "workflow_dispatch"):
        # Neither trigger carries PR context. A manual run may name one PR;
        # otherwise sweep every open format-change PR.
        requested = os.environ.get("GATE_PR", "").strip()
        if requested:
            gate.evaluate(int(requested))
            return
        pulls = [
            pr
            for pr in repo.get_pulls(state="open")
            if any(label.name == FORMAT_LABEL for label in pr.labels)
        ]
        print(f"Sweep: {len(pulls)} open {FORMAT_LABEL} PR(s).")
        for pr in pulls:
            try:
                gate.evaluate(pr.number)
            except Exception as err:  # noqa: BLE001 - keep sweeping other PRs
                print(f"PR #{pr.number}: {err}")
    else:
        with open(os.environ["GITHUB_EVENT_PATH"]) as handle:
            event = json.load(handle)
        gate.evaluate(event["pull_request"]["number"])


if __name__ == "__main__":
    main()
