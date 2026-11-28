---
name: lance-stable-release-prep
description: Audit a Lance stable release for blockers before it is cut, and report findings to a maintainer. Use when a Lance maintainer asks to prepare, vet, or sanity-check a stable release (or RC) — whether the release is ready, whether any critical bugs have been reported since the last stable release, whether breaking changes in the release are opt-in, or whether a changed default (default file format version, default index version) is safe to ship.
---

# Lance Stable Release Prep

## Scope

Use this skill to audit a Lance stable release before it is cut and hand a maintainer the
findings. It answers three questions:

1. Have any **critical bugs** been reported since the last stable release?
2. Are all **breaking changes opt-in**?
3. Has any **written-by-default version** moved, and if so, have readers been able to read it
   for at least three months?

Do not use this skill for:

- Actually cutting the release. The mechanics live in [`release_process.md`](../../release_process.md)
  and the `create-release-branch.yml` / `create-rc.yml` / `approve-rc.yml` workflows.
- Writing release notes (`ci/generate_release_notes.py` does that).
- Beta or preview releases. They carry no stability guarantee, so none of these gates apply.

## Output

**There is no artifact and no file to write.** Produce a report in your reply to the maintainer
who invoked the skill. You do not decide whether to release; you give them what they need to
decide. Do not open issues, add labels, push, or comment on PRs unless they ask.

## Before you start

Confirm you can reach both data sources, from the repository root:

```bash
git fetch --tags --quiet && gh auth status
```

Without tags the baseline cannot be found; without `gh` the issue and label queries fail. If
either is unavailable, say so and stop rather than reporting a partial audit as a clean one.

## Step 1 — Collect the signals

```bash
python3 skills/lance-stable-release-prep/scripts/collect_release_signals.py
```

The script gathers facts only; every judgement below is yours. It prints four sections:
the release under preparation, issues opened since the baseline, PRs in range labeled
`breaking-change` or `format-change`, and version constants that moved. It takes a few minutes,
mostly on the issue query. Useful flags:

- `--baseline vX.Y.Z` — compare against a different stable tag. The default is the newest
  `vX.Y.Z` tag. **When that newest tag is a patch** (e.g. `v14.0.3`), rerun with the GA of that
  line (`v14.0.0`) to widen the issue window across the whole stable line users are running.
- `--head <ref>` — audit a release branch or RC tag instead of `HEAD`.
- `--section issues` (repeatable) — rerun one section without paying for the others.

Note the baseline the script chose and its date. Everything below is relative to it.

## Step 2 — Triage the issues for critical bugs

A **critical bug** is one that would make shipping this release worse than not shipping it:
data loss or silent corruption, a crash or panic on a supported path, a security
vulnerability, a severe performance regression, or an unintended compatibility break.
The repository's own `critical-fix` label is defined as "bugs that cause crashes, security
vulnerabilities, or incorrect data" — use that as the bar.

Work through the script's buckets:

- **`critical-fix` labeled.** Already judged critical by someone. For each, establish whether
  the fix is *in this release*: a closed issue is not enough, because the fix may have landed on
  `main` after the release branch was cut. Find the PR that closed it and check
  `git merge-base --is-ancestor <sha> <head>`.
- **`bug`, still open.** The bucket that needs real judgement. Read each one. Reach for
  `gh issue view <n> --comments` whenever the title is not decisive. Classify each as blocking,
  non-blocking, or needs-a-maintainer-decision, and say which.
- **`bug`, closed.** Verify the fix is actually in range, same ancestry check as above. A fix
  that only exists on `main` does not count toward this release.
- **`performance`.** Only a severe regression blocks. A missing optimization does not.
- **Other / unlabeled.** Skim titles for bugs filed without the `bug` label, especially
  anything reported against a recent beta.

Two signals raise an issue's weight: it is a **regression** from the last stable release
(users who upgrade get worse), and it has **no workaround**. Say so explicitly when either
holds. A long-standing bug that has survived several stable releases is rarely a blocker for
this one — note it and move on.

## Step 3 — Confirm every breaking change is opt-in

**Opt-in** means existing callers and existing datasets keep their current behavior after
upgrading, and only get the new behavior when they ask for it — a new parameter, a new mode,
a new explicit version, a config value. A breaking change that takes effect on upgrade with no
action by the user is **not** opt-in, and is the finding this step exists to surface.

For each `breaking-change` PR the script lists, read the diff and decide which it is:

```bash
gh pr view <n> --json title,body,labels
gh pr diff <n>
```

Judge against the repository's own rules in [`AGENTS.md`](../../AGENTS.md) — public API
signatures are never broken, deprecation goes through `#[deprecated]`/`@deprecated` plus a new
method, and parameter names stay consistent across Rust, Python, and Java. Classify each PR as:

- **Opt-in** — new API, new parameter defaulting to the old behavior, new non-default version.
- **Source-breaking only** — a Rust signature, trait, or crate reorganization that breaks
  compilation but not stored data or runtime behavior. Record it; it belongs in the release
  notes and, for a major version, a migration guide. It is not usually a blocker.
- **Behavior- or data-breaking on upgrade** — the default changed, stored data is read or
  written differently, or an old call now errors. **Report each of these individually**: what
  breaks, for whom, and whether an escape hatch exists.

`format-change` PRs touch the persisted format and require a PMC vote
(`.github/workflows/format-vote-gate.yml`). Check that the `format-spec-vote` status passed on
each, or that `format-waived` was applied. An unvoted format change is a blocker on process
grounds regardless of its technical merit.

Finally, cross-check the version: `ci/check_breaking_changes.py` requires a minor bump when
breaking changes are present, and `docs/src/community/release.md` requires a **major** bump for
changes that are not backwards compatible. If you found a behavior- or data-breaking change,
confirm the target version reflects it.

## Step 4 — Check changed defaults against the three-month reader rule

**The rule: a default may only move to a version that readers have been able to read for at
least three months.** Writing a version that a user's existing readers cannot open is a
compatibility break that no opt-out can undo after the data is written.

The script lists version constants that moved between the baseline and head. For each, first
decide whether a *default* actually moved, because a new version merely becoming available is
fine:

- **File format.** The default is the `#[default]` variant of `LanceFileVersion` in
  [`rust/lance-file/src/version.rs`](../../rust/lance-file/src/version.rs), together with
  `stable_file_version()` (what the `stable` alias resolves to) and `next_file_version()`.
  Read the diff of that file. Note that a version still marked unstable must never become the
  default at all — the three-month clock does not start until it is stable.
- **Index versions.** The written version per scalar index is the `max_version` column of
  `BUILTIN_SCALAR_INDICES` in
  [`rust/lance-index/src/registry.rs`](../../rust/lance-index/src/registry.rs), which points at
  per-index constants (`BTREE_ROW_ADDR_DOMAIN_VERSION`, `INVERTED_INDEX_VERSION_V3`,
  `ZoneMapIndexVersion::CURRENT`, …). Vector indices use `VECTOR_INDEX_VERSION` and its
  siblings in [`rust/lance-index/src/lib.rs`](../../rust/lance-index/src/lib.rs).

For each default that did move, date the reader support:

```bash
# 1. Find the commit that added READ support for the new version (decode path, not just
#    the enum variant or the constant). Confirm by reading the commit.
git log --reverse --format='%h %cI %s' -S'<decoder symbol>' -- <path>

# 2. Earliest stable release containing it.
git tag --contains <sha> | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | head -1

# 3. When that release shipped.
git log -1 --format=%cI <tag>
```

`git tag --contains` only sees the commit's own SHA, so it misses a cherry-pick onto a release
branch, which carries a different SHA. If read support was backported, the earliest release with
it may be older than this reports — check the release branch with
`git log --oneline --grep '#<pr>' release/vX.Y` before concluding.

Reader support counts from the date that **stable release** shipped, not from the merge date —
users install releases, not commits. If the earliest stable release containing read support
shipped **less than three months** before today, the default change **fails the rule**: report
it as a blocker and say which date it would clear on.

Pin the *decode* path, not the name. A commit that adds an enum variant or bumps a constant
usually predates working read support, and taking it as the start date overstates the coverage.
When you cannot pin a single commit confidently, say so and give the range — do not guess in the
permissive direction.

Cross-check against the "Minimal Lance Version" table in
[`docs/src/format/file/versioning.md`](../../docs/src/format/file/versioning.md). Treat it as
corroboration, not proof: entries there have been left as `None` or gone stale. Where the table
and the git history disagree, trust the history and flag the table as needing an update.

For an index version bump, also check whether the compat suite actually covers reading it under
older readers (`python/python/tests/compat/`, the `compat-sequence` job in `nightly_run.yml`, or
an on-demand `compat-pair.yml` run). Untested forward compatibility is a finding of its own.

## Step 5 — Report

Give the maintainer a short report, in this shape:

1. **Scope.** Target version, baseline tag and its release date, head ref.
2. **Critical bugs.** Each candidate blocker: issue number, one-line summary, whether it is a
   regression, whether a workaround exists, whether a fix is already in range. Then the
   non-blocking bugs as a bare list of numbers.
3. **Breaking changes.** Count by classification, then every behavior- or data-breaking change
   spelled out. Note any `format-change` PR missing its vote.
4. **Default changes.** For each: old value, new value, earliest stable release with read
   support, that release's date, months elapsed, and pass or fail.
5. **Open questions.** Anything you could not resolve, and what would resolve it.

State clearly where you exercised judgement versus where you are reporting a fact, and name
what you could not check. Do not round a "probably fine" up to a clean bill of health — a
maintainer acting on an overconfident audit is the failure mode this skill exists to prevent.
