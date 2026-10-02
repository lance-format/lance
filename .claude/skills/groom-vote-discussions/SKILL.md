---
name: groom-vote-discussions
description: >-
  Groom the Vote discussion category on lance-format/lance: classify every open thread's
  state against the community voting rules, propose closing threads whose vote already
  passed (linked PR merged), and produce a Discord-ready report for threads still needing
  attention. Use when the user asks to groom/triage/review vote threads, check on open
  votes, or prepare the weekly vote report. Manually triggered — not scheduled.
---

# Groom Vote Discussions

Reviews every open discussion in the [Vote category](https://github.com/lance-format/lance/discussions/categories/vote)
of `lance-format/lance`, classifies it against the rules in
[lance.org/community/voting](https://lance.org/community/voting/), and produces:

1. A Discord-ready report of threads still needing attention (states 1/2/3/5 below).
2. A confirmed batch of closures for threads whose vote already passed and merged (state 4).

Discussions have no REST API and no "linked PRs" sidebar like Issues do — everything here
uses `gh api graphql`, and PR links are found by regex over thread body/comments.

## Step 1 — Fetch open Vote threads

```shell
gh api graphql -f query='
query {
  repository(owner: "lance-format", name: "lance") {
    discussionCategories(first: 20) { nodes { id name slug } }
  }
}'
```

Find the `id` for the category with `slug: "vote"`, then:

```shell
gh api graphql -f query='
query {
  repository(owner: "lance-format", name: "lance") {
    discussions(first: 50, categoryId: "<VOTE_CATEGORY_ID>", states: OPEN) {
      nodes {
        id
        number
        title
        body
        url
        createdAt
        comments(first: 50) {
          totalCount
          nodes {
            id
            body
            createdAt
            author { login }
          }
        }
      }
    }
  }
}'
```

Paginate with `after:` if `totalCount`/comments exceed page size (unlikely at current volume).

## Step 2 — Get the PMC roster (binding voters)

Binding votes only count from PMC members (`docs/src/community/pmc.md` §Roster — a GitHub
handle column). Prefer `docs/src/community/pmc.yaml` if it exists (introduced by PR #7399 as
the new source of truth); fall back to parsing the markdown table otherwise.

```shell
# Try yaml first (post #7399)
gh api repos/lance-format/lance/contents/docs/src/community/pmc.yaml 2>/dev/null \
  | jq -r .content | base64 -d

# Fallback: markdown roster table
gh api repos/lance-format/lance/contents/docs/src/community/pmc.md \
  | jq -r .content | base64 -d
```

Extract the "GitHub Handle" column into a set of binding-voter logins.

## Step 3 — Per-thread classification

For each thread, do the following in order.

### 3a. Find a linked PR

Regex over `body` + every comment `body` for `github\.com/lance-format/lance/pull/(\d+)` or
bare `#(\d+)`. If multiple matches, prefer one explicitly near words like "PR", "implementation",
"tracking".

### 3b. If a PR was found, check its state

```shell
gh pr view <PR_NUMBER> --repo lance-format/lance --json state,mergedAt,labels,reviews
```

- **If `state == MERGED`** → this is **State 4: Passed but stale**, regardless of anything
  else below. Skip straight to Step 4 (closure).
- **If the PR has label `format-change`** (post-PR #7399 world — format-spec votes move to PR
  review, gated by `ci/format_vote_gate.py`): tally binding votes from `reviews` instead of
  discussion comments — `APPROVED` review by a PMC login = binding +1, `CHANGES_REQUESTED` by a
  PMC login = binding veto (only the latest review per author counts; a newer review supersedes
  an older one by the same author). Still check the discussion thread's own comments too, in case
  people are voting in both places during the transition — report both tallies if they differ.
  Otherwise, fall through to 3c to tally from discussion comments.

### 3c. Tally votes from discussion comments

Per the voting rules, a vote must be **its own top-level comment**, not a reply. For each
top-level comment matching `^\s*[+-]?[01]\b` at the start (i.e. `+1`, `-1`, `0`):

- Look up the author's login against the PMC set → **binding** or **non-binding**.
- For a `-1`: valid only if the comment includes reasoning beyond the vote marker itself
  (more than just "-1" or "-1." with nothing else). An unjustified `-1` doesn't count as a
  veto — flag it separately (State 5).
- Count distinct binding `+1`, binding `-1` (valid only), and non-binding votes of each kind.
  If the same binding author votes more than once, use their most recent comment.

### 3d. Classify (if not already State 4)

- **State 2 — Blocked by veto**: at least one valid binding `-1` exists. Takes priority over
  1/3/5.
- **State 5 — Invalid veto present**: a `-1` exists (binding or not) that lacks justification.
  Can co-occur with state 1 or 3 below — report it as an addendum, not a replacement.
- **State 1 — Within voting period**: `createdAt` is less than 7 days ago, and not state 2.
- **State 3 — Blocked by insufficient votes**: `createdAt` is 7+ days ago, not state 2, and
  binding `+1` count doesn't clearly clear even the loosest quorum (varies 1–3 by decision
  type per the voting doc — report the exact binding `+1` count and let the reader judge
  against the specific quorum for that decision type; do not assert pass/fail here).

A thread with no linked PR at all is classified the same way via 3c/3d, and the report should
note the absence of a PR link as a caveat (harder to eventually detect state 4 for it).

## Step 4 — Propose closures (State 4)

For every State 4 thread, draft (do not yet post) a closing comment:

```
This vote passed and the corresponding PR (#<N>) has been merged. Closing this thread —
thanks everyone for voting!
```

List all proposed State 4 closures to the user and **ask for confirmation** before executing
any of them — closing a discussion is visible to the community. Once confirmed, for each:

```shell
gh api graphql -f query='
mutation($id: ID!, $body: String!) {
  addDiscussionComment(input: {discussionId: $id, body: $body}) { comment { id } }
}' -f id="<DISCUSSION_NODE_ID>" -f body="<closing message>"

gh api graphql -f query='
mutation($id: ID!) {
  closeDiscussion(input: {discussionId: $id, reason: RESOLVED}) { discussion { id } }
}' -f id="<DISCUSSION_NODE_ID>"
```

## Step 5 — Discord report

Everything not closed in Step 4 goes into a Discord-ready report. Discord renders tables
poorly, so use headers and bullets, not markdown tables. Group by state, most-actionable first
(veto > insufficient votes > within period), and call out state-5 flags inline under whichever
thread has them.

Template:

```markdown
**Lance Vote Thread Report — <Mon DD>**

🚫 **Blocked by veto**
- [#<N> <Title>](<url>) — vetoed by @<login> (binding): "<short quote of their reasoning>"

⏳ **Blocked — insufficient votes** (open <D> days)
- [#<N> <Title>](<url>) — <B> binding +1, <NB> non-binding +1, 0 valid -1. Needs more binding votes to clear quorum.
  - ⚠️ has an unjustified -1 from @<login> — needs justification or should be disregarded.

🗳️ **Within voting period** (opened <D> days ago, closes <date>)
- [#<N> <Title>](<url>) — <B> binding +1 so far.

_Closed this pass (vote passed, PR merged):_ #<N>, #<N>
```

Worked example, from the live Vote category checked in this session (2026-07-06):

```markdown
**Lance Vote Thread Report — Jul 6**

⏳ **Blocked — insufficient votes** (open 82 days)
- [#6529 Add `invalidated_fragments` to `IndexMetadata`](https://github.com/lance-format/lance/discussions/6529) — 0 binding +1 found in 5 comments. Needs binding PMC votes to proceed.

🗳️ **Within voting period**
- [#7548 Add details to zone map index](https://github.com/lance-format/lance/discussions/7548) — opened Jul 1, 0 comments yet.
- [#7447 Add Data Overlay Files as experimental feature](https://github.com/lance-format/lance/discussions/7447) — opened Jun 24, 0 comments yet.
- [#7418 WAL shard manifest Status field](https://github.com/lance-format/lance/discussions/7418) — opened Jun 23, 2 comments, tally pending.

_No threads closed this pass._
```

Deliver the report in the conversation for the user to copy into Discord — do not post it
anywhere automatically.

## Important guidelines

- Use `gh` CLI / `gh api graphql` for all GitHub interaction — Discussions have no REST
  endpoint, GraphQL is mandatory.
- Never execute `closeDiscussion` without explicit user confirmation of the specific thread
  list first.
- Don't assert a vote has definitively failed — insufficient votes may still be within the
  informal grace period maintainers allow; report data, let a human decide.
- Keep the report tight — attention is the most valuable resource. One line per thread, no
  restating the full voting rules in the report itself.
