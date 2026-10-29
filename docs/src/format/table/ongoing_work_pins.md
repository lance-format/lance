# Ongoing Work Pins

An ongoing-work pin adds a retention requirement for a fixed source snapshot
and the applicable fragment reuse index (FRI) history while a job runs. Pins do
not establish the freshness or publication validity of a job's output. Physical
retention has the [existing maintenance concurrency limits](#maintenance-concurrency-limits)
described below; confirmation does not fence deletion already in flight.

## Persisted state

`Manifest.ongoing_work_pins` (field 27) contains `OngoingWorkPin` records. Only
the current attached manifest's list is authoritative. Missing or empty means
no active pins. Historical copies of the list do not retain ended pins. Order
is insignificant; duplicate IDs or incomplete records are invalid for pin
mutation and maintenance.

Each record contains `id`, `source_version`, `source_transaction_uuid`,
`created_at`, `expires_at`, and optional `job_label`. Presence and absence
semantics are defined in `protos/table.proto`. The pin ID is the registering
transaction's `uuid`; the source version is that transaction's `read_version`.

`expires_at` is the UTC time after which cleanup may expire the pin. Passing
that time does not automatically remove it. Renewal changes only `expires_at`;
the source identity, creation time, and label stay fixed. No generation counter
is needed because every extension strictly increases `expires_at`.

The first registration sets writer feature bit 19, `FLAG_ONGOING_WORK_PINS`,
in the same commit as the pin. A nonempty list without the bit is invalid.
The bit is sticky on subsequent versions, including when the list becomes
empty, and introduces no reader requirement solely for pins. Ordinary reads
may ignore the pin list. Pin-aware maintenance must validate it.

## Identity

IDs are nonempty UTF-8 strings compared byte-for-byte. Do not case-fold,
normalize, or replace a stored ID with a different textual representation.
This feature imposes no new canonical UUID spelling. Supported writers must
generate fresh transaction IDs for distinct logical transactions, including
Create, Clone, and Restore and after branch-name or table-path reuse. A retry
preserves its original ID. A committed ID must not identify a different
transaction or a second successful publication. This is a writer requirement,
not a persistent registry of all completed transactions.

The exact source is `(source_version, source_transaction_uuid)`. Read the UUID
from the source manifest's inline transaction, or from its `transaction_file`
name. For the filename carrier, require a single relative filename ending in
`.txn`, split at the first hyphen, validate the nonempty decimal read-version
prefix, and retain all remaining nonempty text before `.txn` as the ID. The
prefix is the transaction's read version, not necessarily the source manifest
version. The sidecar's contents need not still exist. When both carriers are
present, their IDs must agree exactly. Failure to read or consistently identify
the source rejects acquisition; never fall back to version alone.

If an active source manifest is missing, unreadable, or its identity disagrees
with the record, report a protection/integrity failure and halt the affected
destructive cleanup. Also refuse any FRI-trimming or fragment-rewrite commit
that cannot establish the pinned fragment set required by its retention checks.
The failure is not an implicit release. Explicit Release by an unambiguous ID
remains allowed without reading that source, so an operator can end the pin.
Readers do not need to validate pins to read unrelated data.

## Time

Times use `google.protobuf.Timestamp` in UTC: seconds in
`[-62135596800, 253402300799]` and nanoseconds in `[0, 999999999]`. Reject invalid
values; do not clamp or truncate them. Compare normalized seconds and then
nanoseconds, preserving exact precision. Initial `expires_at` must be later
than `created_at`.

The format stores absolute times. It does not prescribe an API default duration
or renewal schedule. Fix `created_at` and `expires_at` before the first
registration attempt and preserve both on retries. If a client derives
`expires_at` from a duration, use checked arithmetic and reject overflow.
`created_at` is the registration request's time; do not recompute it at commit.
Each renewal likewise fixes one requested `expires_at` before its first attempt.

Participating clients use reasonably synchronized clocks. Consumers choose a
renewal margin; no table setting or absolute lifetime cap is required. A pin
remains active after `expires_at` until a committed ReleasePin or ExpirePins
removes it from the authoritative list.

## Current state and acquisition

Resolve the authoritative current attached head as specified in
[Format Versioning](versioning.md#current-state-for-pin-decisions). A cached
Dataset or a successful write to a previously reclaimed slot is insufficient.

For a registration attempt with ID P, source V, and source transaction S:

1. If current membership already contains P with the same `(V, S)`, recover
   that pin's current `expires_at` and label without committing or resetting them.
   A different source under P is an error.
2. Otherwise the requested snapshot must still be current: its version and
   source transaction UUID must equal `(V, S)`. If the branch has moved on or
   the snapshot has been replaced, registration fails. Compare those two values
   directly; intervening transaction files are not needed for this check.
3. Construct the next manifest from that current source, add the pin, and set
   the sticky capability. Publish only at V+1 through the existing atomic
   version-slot protocol. A conflict cannot rebase this acquisition onto another
   source. After a definite conflict, a caller that has not started work may
   explicitly acquire the new head with a new ID.
4. Before acknowledging either a fresh acquisition or recovery, confirm the
   current named-branch reference still has the acquired branch ID, the
   authoritative head contains P with `(V, S)`, and the source manifest has S.
   The pin may have been carried into a newer head. A historical transaction
   matching P alone is insufficient. If these facts cannot be established,
   return an indeterminate outcome or failure, never acknowledge registration.

These confirmation reads establish observed current state, not an atomic
handoff against physical deletion. Branch deletion must perform the
[active-pin check](branch_tag.md#deletion-with-ongoing-work-pins), but that check
is not atomic with registration or subsequent physical deletion.
Main has no branch-reference ID that distinguishes table-root recreation;
whole-table deletion must be coordinated by the owning service or catalog.
No permanent table identity is introduced here.

Jobs begin work only after confirmation. Jobs that need recovery after a restart
must save their pin ID and, until registration is confirmed, the original
request needed to retry it. Recovery returns the current pin, whose `expires_at`
may already have passed; the consumer chooses whether to renew before continuing.
The pin remains active until its removal commits. Ended registration outcomes
are not retained indefinitely.

Concurrent commits, including other pin operations, can repeatedly move the
head before registration commits. Callers should bound their registration
attempts and retain the original request for each attempt whose outcome remains
unknown. A definitively rejected attempt does not require recovery state.

## Renewal, release, and expiration

`RenewPin` requires the target ID to exist. If the current `expires_at` already
equals or exceeds the fixed request, return the current pin without a commit.
Otherwise commit the requested later `expires_at` while preserving every other
field. Re-evaluate after conflicts. A concurrent Release or ExpirePins that
wins first makes renewal fail; renewal never creates a missing pin.

`ReleasePin` removes only its target ID. Absence is a successful no-op. It does
not stop the worker or cancel an unconfirmed registration still in flight.
Different jobs have independent IDs even when their source files are identical.

`ExpirePins` contains the inspection time `observed_at` and candidates
`(pin_id, expected_expires_at)`. All timestamps must be valid, candidate IDs
unique, and each `expected_expires_at` at or before `observed_at`.

Before committing expiration, remove a candidate only if its ID is still active
and its current `expires_at` exactly equals `expected_expires_at`. If renewal
changed that value, skip the candidate. Also skip absent IDs and preserve all
other pins. Retries use the same `observed_at` and candidates; cleanup may make
a new inspection later. If no candidate qualifies, make no commit.

This orders renewal against expiration: a prior extension invalidates the old
expiration decision; prior expiration prevents renewal. Renewal and other pin
commits count as ordinary versions and cleanup intervals. No-op operations do
not create versions. Expiration commits precede planning physical reclamation
of the resources they release. Cleanup's maintenance commits must not recursively
trigger the same cleanup.

## Retention and data operations

Resolve current pin membership when planning retention. Retain each active
source manifest and the resources that normal snapshot retention preserves for
that manifest, including base data, overlay files, FRI metadata, and referenced
external mapping files. Reuse the existing retention machinery: an active pin
adds a retained snapshot. Cleanup may still remove resources that no active pin
or other retention rule requires. Report when pins prevented removal. Removing a
pin does not remove protection supplied by another pin, snapshot, or consumer.

Compaction must record the FRI row movements active pins need. Before committing
its result, compaction must recheck the active pins, including pins registered
after planning began. If its result lacks required movement records, it must
not commit; the caller can replan.

History removal must retain the union of requirements from current pins and
existing consumers, except for translation paths discarded by Restore or
Overwrite as described below. An index catching up does not end an active pin's
requirement. An error in an existing consumer check does not permit dropping
pin-required history. If pin requirements cannot be established, refuse the trim.

For legacy-only FRI (`index_version = 0`), a source at version V retains
applicable current-chain records whose `dataset_version` is V or later. When
publishing a new legacy record, a pin-aware writer must stamp its
`dataset_version` with the version immediately preceding the publishing
manifest, and restamp that new record on retry. This makes the retention
boundary inclusive of V; it does not change stamps on existing records.

For FRI `index_version = 1`, including histories that also contain legacy
records, determine pin requirements from fragment references rather than
version stamps. Initialize a set with the fragment IDs in all active pinned
source snapshots. Retain each current-history record with a source fragment in
that set, and add its destination fragment IDs to the set. Repeat until no
additional records are selected. If a selected record is a legacy version,
retain all its groups and add every group's destination IDs. Retain whole
records and the downstream records they require. Add these pin requirements
to the existing consumers' requirements; do not replace the existing consumer
checks.

Retain the external mapping files referenced by required history, including
records added after registration. Physical cleanup may remove a mapping file
only when no retained snapshot or required history references it, subject to
the existing maintenance concurrency limits. Replacing FRI metadata does not
by itself make its mapping files eligible for deletion.

Pin retention does not change overlay/index correctness requirements or make
an unsupported overlay or rewrite operation valid.

Restore and Overwrite carry the current active list, fixed sources, timestamps,
and sticky bit. They never import the target snapshot's historical pin list.
Apply FRI retention to the resulting current history. Neither operation is
required to reconstruct discarded translation paths. Within that history,
retain movement records needed to connect a pinned snapshot to the current
table layout. Records needed only for a discarded intermediate layout may be
removed if no other consumer requires them. Pins keep source files available,
but do not guarantee that a job can translate or publish its results after a
restore or overwrite; retention does not supply a missing mapping.

New branches and clones have empty pin lists and preserve the sticky bit when
inherited. Detached snapshots cannot register or mutate authoritative pins.

Other ordinary commits preserve the current list. All operations that rewrite
or trim FRI revalidate the applicable retention requirements at commit, subject
to the Restore and Overwrite rules above. Detailed directional rules are in
[Transactions](transaction.md#ongoing-work-pin-transactions).

Pins do not protect newly written, uncommitted output. Jobs need a separate
staging and publication strategy for those files. Before releasing a pin, a job
must publish its output and confirm that the result retains the files and FRI
history it needs. Publication and release are separate transactions; the four
pin operations do not add a combined publication-and-release transaction.

## Compatibility and implementation boundary

Bit 19 is reserved and unsupported until this contract is implemented. Ordinary
writers that do not support it must refuse flagged manifests. Readers may still
read data if they support the other reader flags. A flag check gates attempts
that reach it; it does not cancel destructive operations already in flight.

The minimum upstream Rust cleanup guard boundary identified for this proposal
is `v14.0.0-beta.4`, containing commit
`84d1681e07af565837f2d97e72ff1b8b040cd15c`. That release checks writer flags while
inspecting manifests; it rejects an unknown pin bit rather than honoring pins.
The preceding `v14.0.0-beta.3` lacks that check. Rust-based Python/JNI clients
must be assessed by their embedded core revision. This boundary is not a
pin-capable client release, and other cleanup implementations require their own
verification. Activation requires pin-aware maintenance clients, exclusion of
older or unverified destructive clients, and completion or cancellation of
incompatible operations already in flight. Existing branch deleters that never
inspect capability flags are not made safe by this bit.

## Maintenance concurrency limits

This feature does not add atomic coordination between protection acquisition,
restore, and destructive maintenance. Existing tags also use separate
acquisition and retention checks. A tag created after cleanup's inspection can
succeed while that sweep still deletes its old source files. An earlier sweep
can likewise delete files adopted by a concurrent restore, and a delayed branch
deletion can damage a replacement after overlapping recovery and name reuse.

Current-version-only acquisition avoids selecting an arbitrary old version
already eligible for cleanup. A sweep retains the head it inspected, and the
usual grace period protects newly written, unverified files while they remain
recent. This narrows exposure compared with old-version tags, but is not a
general concurrency guarantee: the grace period can elapse or be bypassed by
`delete_unverified`, and restore can reintroduce older files. The old-version
tag case is not evidence that every current-version pin has the same exposure.

The pin checks do not cancel an earlier deletion plan or prevent a destructive
operation from running after confirmation. A successful registration therefore
does not guarantee safety against these overlapping operations. Callers must
not treat the pin as a lock against destructive maintenance.

These are existing maintenance and branch-coordination gaps. Their fixes are
outside this proposal; no deletion marker, permanent name retirement, new
branch layout, or reference journal is specified. Branch names remain reusable.
This limitation does not relax the required active-pin checks, the retention
of sources selected by those checks, or the transactional FRI retention rules.
