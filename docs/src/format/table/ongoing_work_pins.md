# Ongoing Work Pins

An ongoing-work pin retains a fixed source version, intervening committed
history needed to validate publication, and applicable fragment reuse index
(FRI) history while a job runs. This revision preserves the intervening history
by retaining complete versions. Pins do not establish the freshness or
publication validity of a job's output.

A successful registration cannot stop an earlier cleanup or branch-deletion
operation from deleting files it already selected. See
[Cleanup and deletion races](#cleanup-and-deletion-races) for examples and
guidance.

## Persisted state

`Manifest.ongoing_work_pins` (field 27) contains `OngoingWorkPin` records. Only
the current attached manifest's list is authoritative. Missing or empty means
no active pins. Historical copies do not keep removed pins active. Order
is insignificant; duplicate IDs or incomplete records are invalid for pin
mutation and maintenance.

Each record contains `id`, `source_version`, `created_at`, `expires_at`, and
optional `job_label`. Presence and absence semantics are defined in
`protos/table.proto`. The pin ID is the registering
transaction's `uuid`; the source version is that transaction's `read_version`.

`expires_at` is the UTC time after which cleanup may expire the pin. Passing
that time does not automatically remove it. Renewal changes only `expires_at`;
the source version, creation time, and label stay fixed. No generation counter
is needed because every extension strictly increases `expires_at`.

The first registration sets writer feature bit 20, `FLAG_ONGOING_WORK_PINS`,
in the same commit as the pin. A nonempty list without the bit is invalid.
The bit is sticky on subsequent versions, including when the list becomes
empty, and introduces no reader requirement solely for pins. Ordinary reads
may ignore the pin list. Pin-aware maintenance must validate it.

## Identity

Pin IDs are nonempty UTF-8 strings compared byte-for-byte. Do not case-fold,
normalize, or replace a stored ID with a different textual representation.
This feature imposes no new canonical UUID spelling. Distinct logical
registrations must use fresh transaction IDs, including after branch-name or
table-path reuse. A retry preserves its original ID. A pin ID must not be reused
for another registration, even after the original pin ends. No persistent
registry of ended pins is required.

The source is the attached `source_version` in the pin's table and branch.
Requests and handles must remain bound to that table location and, for named
branches, the acquired branch identifier. Within that table and branch, the
version number identifies the source. Registration does not require the source
transaction's UUID or contents.

Requests and handles must not be reused after the whole table is deleted and
recreated, even at the same path. The owning service or catalog must coordinate
that lifecycle. Pins do not introduce a table-incarnation identifier or detect
every replacement of a table at the same location.

If an active source manifest is missing, unreadable, or does not match the
requested version, report a protection/integrity failure and halt the affected
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

Validate the request's table location and named-branch identity before
submitting or recovering a registration. For an attempt with pin ID P and
source version V:

1. If current membership already contains P with source version V, recover
   that pin's current `expires_at` and label without committing or resetting them.
   A different source under P is an error.
2. Otherwise the authoritative current version must equal V. If the branch
   has moved on, registration fails. Compare the version numbers directly;
   intervening transaction files are not needed for this check.
3. Construct the next manifest from that current source, add the pin, and set
   the sticky capability. Publish only at V+1 through the existing atomic
   version-slot protocol. A conflict cannot rebase this acquisition onto another
   source. After a definite conflict, a caller that has not started work may
   explicitly acquire the new head with a new ID.
4. Before acknowledging either a fresh acquisition or recovery, confirm the
   current named-branch reference still has the acquired branch ID, the
   authoritative head contains P with source version V, and that source
   manifest can still be read and validated as version V.
   The pin may have been carried into a newer head. A historical transaction
   matching P alone is insufficient. If these facts cannot be established,
   return an indeterminate outcome or failure, never acknowledge registration.

These reads confirm that the pin and its source are present at that time.
They do not cancel deletion decisions already in progress. Branch deletion
must perform the [active-pin check](branch_tag.md#deletion-with-ongoing-work-pins),
but a pin registered after that check does not stop the deletion.
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

Resolve current pin membership and the authoritative attached head H when
planning retention. If any pins are active, let F be their minimum
`source_version`. Retain every attached version in the same branch from F
through H, inclusive. Detached versions are not part of this window. Preserve
the manifest and any commit-store records needed to discover and read each
version through the backend's authoritative protocol.

Reuse normal version retention for every manifest in this window, including
its base data, deletion files, overlays, indices, FRI metadata, referenced
transaction sidecars, and external mapping files. Retain shared physical files
once. These remain ordinary readable versions; retaining their manifests
while deleting their referenced data does not satisfy this contract.

Each attached commit whose result contains active pins must persist its own
transaction, either inline in its manifest or in the referenced transaction
sidecar, before publishing the manifest. This includes registration and commits
carrying pins through Restore or Overwrite. Keep the transaction for every
version after F through H so publication can inspect the committed changes in
order. A transaction filename alone is not this evidence. The transaction that
created F is not required solely for pin registration or this retention rule.

If a manifest in the required window, or a transaction required by the preceding
rule, is missing or unreadable, report a protection/integrity failure and halt
the affected destructive cleanup. Do not silently skip the missing version or
reclaim other evidence in its place. This does not release any pin; explicit
release by an unambiguous ID remains available.

Renewal never moves a pin's source version. Release or committed expiration may
advance the minimum source version; removing the last pin removes this window
requirement. Other pins, tags, retained versions, and consumers still retain their
resources. Cleanup may remove other eligible resources and must report retention
caused by pins. Passing `expires_at` alone does not advance the boundary.

This window can retain intermediate data and index generations that publication
does not itself read. Its storage cost grows with rewritten resources and the
age of the oldest active source. No new pin field or metadata-only history
representation is introduced. Replacing complete version retention with a
smaller evidence representation requires a separately specified compatible
discovery and retention protocol.

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

For FRI version 1 (`index_version = 1`), including histories that also contain
legacy records, determine pin requirements from fragment references rather
than version stamps. For each active pin, load and validate the manifest at
its `source_version` in the same table and branch. Take all fragment IDs listed
in that manifest; no row-data scan is needed to obtain this set. Initialize a
set with these IDs from all active pins. Retain each current-history record
with a source fragment in that set, and add its destination fragment IDs to
the set. Repeat until no additional records are selected. If a selected record
is a legacy version,
retain all its groups and add every group's destination IDs. Retain whole
records and the downstream records they require. Add these pin requirements
to the existing consumers' requirements; do not replace the existing consumer
checks.

Retain the external mapping files referenced by required history, including
records added after registration. Physical cleanup may remove a mapping file
only when no retained version or required history references it, subject to
the [cleanup and deletion races](#cleanup-and-deletion-races). Replacing FRI
metadata does not by itself make its mapping files eligible for deletion.

Pin retention does not change overlay/index correctness requirements or make
an unsupported overlay or rewrite operation valid.

Attached Restore and Overwrite carry the current active list, fixed sources,
timestamps, and sticky bit. They never import the target version's historical
pin list.
Neither operation shortens the retained version window.
Apply FRI retention to the resulting current history. Neither operation is
required to reconstruct discarded translation paths. Within that history,
retain movement records needed to connect a pinned version to the current
table layout. Records needed only for a discarded intermediate layout may be
removed if no other consumer requires them. Pins keep source files available,
but do not guarantee that a job can translate or publish its results after a
restore or overwrite; retention does not supply a missing mapping.
Removing a record from the current FRI does not reclaim files still referenced
by versions in the retained window.

New branches and clones have empty pin lists and preserve the sticky bit when
inherited. Detached versions must serialize an empty pin list and preserve the
sticky bit when inherited. Creating a detached version does not change the
branch's current pin list. Detached versions cannot own active pins, and none
of the four pin operations may be used in a detached commit.

Other attached commits preserve the current list. All operations that rewrite
or trim FRI revalidate the applicable retention requirements at commit, subject
to the Restore and Overwrite rules above. Detailed directional rules are in
[Transactions](transaction.md#ongoing-work-pin-transactions).

Pins do not protect newly written, uncommitted output. Jobs need a separate
staging and publication strategy for those files. Before releasing a pin, a job
must publish its output and confirm that the result retains the files and FRI
history it needs. Publication and release are separate transactions; the four
pin operations do not add a combined publication-and-release transaction.

## Compatibility and implementation boundary

Bit 20 is reserved and unsupported until this contract is implemented. Ordinary
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

## Cleanup and deletion races

Cleanup checks which versions and files are still needed, then deletes the
eligible files. Registering a pin changes what later checks must retain. It
does not cancel a deletion decision already made by an earlier operation.

Pins only register against the current version. A cleanup run retains the
current version it inspected. Cleanup also leaves recent files alone when it
cannot verify that they belong to an old version being removed. This grace
period gives writers time to commit their files. It ends as files age, and
`delete_unverified` bypasses it.

Restore can bring old files back into use after cleanup has selected them for
deletion. For example:

1. Cleanup decides that an old version's file A can be deleted.
2. A user restores that version. The new current version references file A.
3. A job registers a pin on the restored version. Registration succeeds while
   file A still exists.
4. Cleanup deletes file A using its earlier decision. The job now needs a
   missing file, even though its pin is active.

Users can call restore at any time. The restore API does not wait for cleanup,
and pins do not block restore. Applications that control both operations can
avoid this race by waiting for running cleanup to finish before restoring,
and keeping new cleanup runs paused until the job's pin is confirmed. The
application must arrange that ordering; pins do not enforce it. Cleanup can
then resume while the job runs.

Running cleanup less often does not guarantee that it will avoid a restore.
Keeping recent versions for longer also does not protect old files already
selected for deletion. Jobs should register before reading and preserve normal
cleanup grace protections, but neither measure fixes this restore race.

Branch deletion has a similar gap: it can check the current manifest, find no
pins, and proceed after a job registers one. Delayed deletion can also damage
a replacement branch when recovery and name reuse overlap the deletion.
Likewise, a tag added after cleanup's inspection may fail to protect an old
version. Pins avoid selecting arbitrary old versions, but inherit the restore
and branch-deletion races described above.

Fixing these existing races is outside this proposal. Branch names remain
reusable; pins add no deletion marker, permanent name retirement, new branch
layout, or reference journal. Cleanup and branch deletion must still check
active pins and retain the resources those checks require. Compaction and
FRI trimming must still validate pin requirements before committing.
