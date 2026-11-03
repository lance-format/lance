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
transaction's `uuid` parsed as a UUID. The source version is the version that
the successful registration commit creates, not its initial `read_version`.

`expires_at` is the UTC time when the pin becomes eligible for removal by the
next attached commit. Passing that time does not edit a manifest or authorize
physical deletion. Renewal changes only `expires_at`;
the source version, creation time, and label stay fixed. No generation counter
is needed because every extension strictly increases `expires_at`.

Registration sets writer feature bit 20, `FLAG_ONGOING_WORK_PINS`,
in the same commit as the pin. A transaction using `release_pin_id` also sets
the bit. A nonempty list without the bit is invalid.
The bit is sticky on subsequent versions, including when the list becomes
empty, and introduces no reader requirement solely for pins. Ordinary reads
may ignore the pin list. Pin-aware maintenance must validate it.

## Identity

Pin IDs and references use the existing `UUID` message with exactly 16 bytes.
Missing values or any other byte length are invalid. Compare the UUID bytes.
A registration's textual `Transaction.uuid` must parse as that UUID; ordinary
transactions keep their existing UUID representation. Distinct logical
registrations must use fresh transaction IDs, including after branch-name or
table-path reuse. A retry preserves its original ID. A pin ID must not be reused
for another registration, even after the original pin ends. No persistent
registry of ended pins is required.

The source is the attached `source_version` in the pin's table and branch.
Requests and handles must remain bound to that table location and, for named
branches, the acquired [branch identifier](branch_tag.md#branch-identity).
Use the existing identifier and legacy fallback defined there; their recreation
limits also apply to pins. Within that table and branch, the
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
continues to retain resources while its record remains in the authoritative
list, including after `expires_at`. Consumers must renew before the deadline;
physical cleanup waits for committed removal.

## Current state and acquisition

Resolve the authoritative current attached head as specified in
[Format Versioning](versioning.md#current-state-for-pin-decisions). A cached
Dataset or a successful write to a previously reclaimed slot is insufficient.

Validate the request's table location and named-branch identity before
submitting or recovering a registration with pin ID P:

1. If current membership already contains P, recover its stored source version,
   current `expires_at`, and label without committing or resetting them.
2. Otherwise resolve the current attached version H. Reject a fresh request
   whose fixed `expires_at` has already passed. The request does not choose a
   source version; the job has not started planning or reading yet.
3. Construct the next manifest from H, add P with `source_version = H+1`, and
   set the sticky capability. Publish at H+1 through the existing atomic
   version-slot protocol. On conflict, repeat against the newer head with the
   same ID and request times. Recompute the source for each uncommitted attempt;
   once registration succeeds, its stored source never moves.
4. Before acknowledging either a fresh acquisition or recovery, confirm the
   current named-branch reference still has the acquired branch ID, the
   authoritative head contains P with its stored source version V, and that
   source manifest can still be read and validated as version V. Return V.
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
may already have passed. Recovering a record does not extend its deadline.
A consumer that missed renewal must acquire a new pin before starting new work.
The old record still retains resources until its removal commits. Do not retry
registration after a known release or expiration; ended outcomes are not
retained indefinitely.

Concurrent commits, including other pin operations, can repeatedly move the
head before registration commits. Callers should bound their registration
attempts and retain the original request for each attempt whose outcome remains
unknown. A definitively rejected attempt does not require recovery state.

## Renewal, release, and expiration

`RenewPin` requires the target ID to exist and its current deadline to be in
the future when the attempt is validated. Check this again after conflicts.
If the current `expires_at` already equals or exceeds the fixed request,
return the current pin without a commit.
Otherwise commit the requested later `expires_at` while preserving every other
field. A concurrent release or committed expiration that wins first makes
renewal fail; renewal never creates a missing pin.

`ReleasePin` explicitly removes its target ID. Absence is a successful no-op. It does
not stop the worker or cancel an unconfirmed registration still in flight.
Different jobs have independent IDs even when their source files are identical.
A release that commits also applies the ordinary expiration rule to other pins.

Expiration happens during ordinary attached commits; there is no separate
expiration operation. For each commit attempt, sample the current UTC time and
derive pin state from that attempt's predecessor manifest. Omit records whose
`expires_at` is at or before that time. Preserve every field of the remaining
records except for explicit pin changes requested by the transaction. A fresh
registration must have a future deadline; renewal must pass the check above.
After a conflict, rebuild from the new predecessor and recheck expiration.
Never reuse an old removal list without checking renewed records.

The removal becomes effective only when that manifest becomes an authoritative
attached version. A renewed deadline that is still in the future survives a
retry; removal that commits first prevents renewal from recreating the pin.
A local time check is not an atomic storage deadline: a validated write can
finish later. The committed ordering, not response timing, protects against
delayed writes and lost responses.

No-op pin requests create no version solely to remove other expired pins.
Real commits use ordinary version counting and cleanup intervals. Cleanup does
not create an expiration commit: on a quiet table, expired records and their
retention requirements remain until another attached commit or explicit
release. Physical deletion must wait until removal is committed and current
retention has been resolved again.

## Publication and release

An ordinary attached transaction may set `Transaction.release_pin_id` (field 5)
to release one pin in the same commit as its operation. The field uses the
16-byte `UUID` type. It names the existing pin; the publication transaction has
its own UUID. Absence requests no explicit release. A missing target is a no-op
for the release, not for the accompanying operation. The field is invalid on
RegisterPin, RenewPin, ReleasePin, Clone, and detached commits.

Validate the operation and its publication evidence before releasing the pin.
Then publish the operation's result and the pin's removal in the same manifest.
Failure without publication leaves both unapplied. A conflict re-evaluates both
against the new predecessor. Pin release never bypasses data conflicts,
publication validation, or the retention requirements of other pins and consumers.

Persist this transaction inline or in its referenced sidecar before publishing
the manifest, even when it releases the last pin or the target is already absent.
If neither representation can be stored, reject before publication. This
preserves evidence for commit recovery; it does not extend retention after the
pin ends. If later cleanup removes the evidence, recovery may be indeterminate.

After a lost response, resolve the publication transaction's outcome using its
UUID and normal commit recovery. An absent pin alone does not prove that the
results were published: another commit may have expired or released it. Keep
standalone ReleasePin for cancellation, abandonment, and callers that release
after separately confirming publication. This uses the current transaction
format and does not depend on Transaction V2.

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

Each attached commit whose result contains active pins, or which sets
`release_pin_id`, must persist its own transaction, either inline in its
manifest or in the referenced transaction sidecar, before publishing the
manifest. This includes registration, commits carrying pins through Restore or
Overwrite, and publication that releases the last pin. Keep the transaction
for every version after F through H so publication can inspect the committed changes in
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

Compaction must record the FRI row movements retained pins need. Before committing
its result, compaction must recheck the current pins, including pins registered
after planning began, and apply only the removals published in that same commit.
If its result lacks required movement records for the remaining pins, it must
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
its `source_version` in the same table and branch. Resolve the source version's
complete fragment metadata using its declared layout and take all fragment IDs.
A flat layout uses `Manifest.fragments`; a [fragment-tree layout](fragment_metadata.md)
requires reading its referenced metadata. Its empty inline fragment list does
not mean an empty source. Reject acquisition for an unsupported layout; if
required metadata cannot be resolved later, refuse the affected cleanup,
rewrite, or trim. No row-data scan is needed to obtain this set. Initialize a
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

Attached Restore and Overwrite start from the current pin list, apply the same
expiration and explicit-release rules, and preserve the remaining fixed sources,
timestamps, and sticky bit. They never import the target version's historical
pin list. Restoring data does not itself end a pin or shorten its retention window.
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
of the three pin operations or `release_pin_id` may be used in a detached commit.

Other attached commits preserve the current list except for the expiration and
explicit-release rules above. All operations that rewrite
or trim FRI revalidate the applicable retention requirements at commit, subject
to the Restore and Overwrite rules above. Detailed directional rules are in
[Transactions](transaction.md#ongoing-work-pin-transactions).

Pins do not protect newly written, uncommitted output. Jobs need a separate
staging and publication strategy for those files. A job may release its pin in
the transaction that publishes its output, provided the resulting version
retains the files and FRI history it needs. If using a separate release, confirm
publication first.

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

Pins register the new version their registration commit creates. A cleanup run retains the
current version it inspected. Cleanup also leaves recent files alone when it
cannot verify that they belong to an old version being removed. This grace
period gives writers time to commit their files. It ends as files age, and
`delete_unverified` bypasses it.

Restore can bring old files back into use after cleanup has selected them for
deletion. For example:

1. Cleanup decides that an old version's file A can be deleted.
2. A user restores that version. The new current version references file A.
3. A job registers a pin. Its registration version still references file A,
   and registration succeeds while file A exists.
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
