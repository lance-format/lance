# Lance Community Voting Process

Lance uses a consensus-based voting process for decision-making.

## Expressing Votes

Votes are expressed as the following:

- **+1**: Yes
- **0**: Abstain
- **-1**: No

When voting, it is recommended that voters indicate whether their vote is binding or not (e.g., `+1 (non-binding)`, `-1 (binding)`)
to ease the counting of binding votes.

In addition to the vote, voters can also express their justification as part of the comment.
**-1** votes must include justification to allow meaningful discussion.
Any **-1** vote not accompanied by justification is considered invalid.

For votes conducted on GitHub Discussions,
each vote should be cast as an independent comment instead of as a reply within a comment.
This ensures that people can discuss the vote as replies to that specific comment if needed
(e.g., to discuss **-1** vetoes or address concerns).

For votes conducted on a pull request, cast **+1** by approving the PR and **-1** by
requesting changes. These votes are counted automatically, so a **+1** written only as
a comment does not count.

## Binding Votes

Only votes from the binding voters are counted for each decision,
but other people in the community are also encouraged to cast non-binding votes.
Binding voters should consider any concern from non-binding voters during the vote process.

## Vetoes

A **-1** binding vote is considered a veto for all decision types. Vetoes:

- Stop the proposal until the concerns are resolved
- Cannot be overruled
- Trigger consensus gathering to address concerns

## Voting Requirements

| Decision Type                                                                 | +1 Votes Required                            | Binding Voters                 | Location                              | Minimum Period |
|-------------------------------------------------------------------------------|----------------------------------------------|--------------------------------|---------------------------------------|----------------|
| Governance process and structure modifications                                | 3                                            | PMC                            | Private Mailing List                  | 1 week         |
| Changes in maintainers and PMC rosters                                        | 3 (excluding the people proposed for change) | PMC                            | Private Mailing List                  | 1 week         |
| Incubating subproject graduation to subproject                                | 3                                            | PMC                            | GitHub Discussions                    | 3 days         |
| Subproject management                                                         | 1                                            | PMC                            | GitHub Discussions                    | N/A            |
| Release a new stable major version of the core project                            | 3                                            | PMC                            | GitHub Discussions                    | 3 days         |
| Release a new stable minor version of the core project                            | 3                                            | PMC                            | GitHub Discussions                    | 3 days         |
| Release a new stable patch version of the core project                            | 3                                            | PMC                            | GitHub Discussions                    | N/A            |
| Lance Format Specification modifications                                      | 3 (excluding proposer)                       | PMC                            | GitHub PR (see [below](#lance-format-specification-changes)) | 72 hours, excluding weekends |
| Code modifications in the core project (except changes to format specifications)  | 1 (excluding proposer)                       | Maintainers with write access  | GitHub PR                             | N/A            |
| Release a new stable version of subprojects                                   | 1                                            | PMC                            | GitHub Discussions                    | N/A            |
| Code modifications in subprojects                                             | 1 (excluding proposer)                       | Contributors with write access | GitHub PR                             | N/A            |

## Lance Format Specification Changes

The pull request *is* the proposal. Open a PR with the specification change,
and the PMC votes on it there — there is no separate design document or
discussion thread to write first, and the requirement is enforced structurally
in CI rather than by convention.

### Proposing a Change

Keep a format-specification PR to the specification itself: the protobuf
definitions and the spec documentation, plus the minimum library changes needed
to keep the build green (for example, matching a renamed generated field).
Implement the behavior behind the change in follow-up PRs.

This is not just a tidiness preference. The vote is on the format — a durable
compatibility contract that outlives any one implementation — and PMC members
should be able to read the whole of what they are voting on. A PR that also
carries the reader, writer, and test changes buries the contract in
implementation detail, and it drags an ordinary code review through a 72-hour
voting period it does not need.

Discussion happens as review comments on the PR, so reviewers can respond to
specific lines of the specification. Open the PR as a draft while it is still
taking shape; the voting period starts when you mark it ready for review.

### How the Vote is Counted

A PR counts as a format-specification change when it modifies the protobuf
definitions (`protos/**/*.proto`) or the spec documentation (`docs/src/format/**`);
such PRs are labeled `format-change` automatically. The
[format spec vote gate](https://github.com/lance-format/lance/blob/main/.github/workflows/format-vote-gate.yml)
blocks merging a `format-change` PR until all of the following hold:

- **Three binding +1 votes.** Three PMC members have approved the PR, excluding
  the proposer. Cast +1 by approving the PR. Only approvals on the latest commit
  count — pushing new commits invalidates earlier approvals, since the proposal
  has changed.
- **No veto.** No PMC member has an outstanding "Request changes" review. A `-1`
  binding vote (cast by requesting changes) is a veto and blocks the merge until
  withdrawn.
- **Minimum voting period.** At least 72 hours have elapsed since the vote
  opened. Weekends do not count toward the 72 hours, so a proposal opened on a
  Friday afternoon still gets three working days of attention. The voting period
  opens once the PR is both labeled `format-change` and marked ready for
  review, whichever comes last. Weekends are delimited in UTC; the gate comments
  on the PR with the exact closing time in both UTC and Pacific Time.

The gate is the `format-spec-vote` required status check on protected branches.
The PMC roster used to count votes is read from
[`docs/src/community/pmc.yaml`](./pmc.md). It re-evaluates on a 15-minute
schedule, so the tally comment and the status check trail a review by a few
minutes; the comment links to a "Run workflow" page for anyone who would rather
re-check immediately.

For a trivial edit that does not change the format — a typo, wording, or
formatting fix — a PMC member may apply the `format-waived` label to waive the
vote.
