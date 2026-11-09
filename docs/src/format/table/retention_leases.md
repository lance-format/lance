# Retention Leases

A writer can add a **retention lease** to protect work in progress. While the lease
remains in the current manifest:

1. File cleanup **must retain** every version from the lease source through
   the current head, including the files and transaction history those versions need.
2. Fragment Reuse Index (FRI) cleanup **must retain** the movement records needed
   to connect the lease source fragments to the current layout.

This lets compaction and cleanup run while an index or another result is being
built. Unlike a tag, which retains a snapshot indefinitely, a lease can be renewed
and expires if its writer stops renewing. The next attached commit removes an
expired lease; until then, cleanup still honors it. The writer can explicitly
release a lease when its work finishes or is abandoned.

Leases retain resources. The consumer remains responsible for validating and
publishing its result and protecting uncommitted output. Leases cannot cancel
[deletion already in progress](#cleanup-and-deletion-races).

## Persisted state

`Manifest.retention_leases` (field 27) stores the leases. Only the current attached
manifest's list is authoritative; historical lists cannot revive ended leases.
Missing or empty means no leases, and list order has no meaning.

<details>
<summary>Lease protobuf message</summary>

```protobuf
%%% proto.message.lance.table.RetentionLease %%%
```

</details>

Writers and maintenance **must reject** incomplete records, invalid field
values, and duplicate IDs. A corrupt list blocks lease mutations, including
release, and affected destructive maintenance; it requires administrator repair.
A valid lease can still be released if its source is unreadable.

Registration and transactions carrying `release_lease_id` **must set** writer
flag 20 (`FLAG_RETENTION_LEASES`). A nonempty list without it is invalid.
Once set, the flag **must remain set**, including when all leases end and through
restore or clone. Leases require no additional reader flag; readers may ignore the list.

## Identity and time

A lease's ID is the UUID of its registration transaction. Distinct registrations
**must use distinct IDs**, even after a lease ends or a table or branch is recreated.
Retries keep the same ID. The lease belongs to one table and branch; its
`source_version` is the version created by its successful registration commit.

Requests and handles **must stay bound** to that table and branch, including
the existing [branch identifier](branch_tag.md#branch-identity) for named branches.
The identifier's legacy recreation limits still apply. The owning service or
catalog must prevent reuse across whole-table deletion and recreation at the
same path; leases cannot detect that replacement.

`created_at` and `expires_at` are absolute UTC times. Writers choose both before
the first registration attempt and preserve them on retries. A renewal likewise
fixes its requested `expires_at` once. Times must be valid protobuf timestamps,
compared at full precision, and expiration must be later than creation.

Clients use reasonably synchronized clocks and renew before expiration. The
format specifies no default duration, renewal schedule, or maximum lifetime.

## Registration and recovery

Lease decisions use the [authoritative current head](versioning.md#current-state-for-lease-decisions).
To register lease P:

1. Validate the target table and branch identity. If P is already listed,
   recover its stored source, expiration, and label without changing them.
2. For a fresh submission, require an unexpired request. Resolve head H, build
   version H+1 from it, and add P with `source_version = H+1`.
3. Publish atomically. A definitively lost attempt may retry against a newer
   head with the same ID and times. Recompute the source on each uncommitted
   attempt; once registration succeeds, the source stays fixed.
4. Before confirming acquisition, verify the named-branch identity, current
   unexpired membership with its stored source V, and a readable source manifest
   matching V. Return V. The job **must wait for confirmation** before planning
   or reading its source.

After an uncertain response, recovery **must not submit the registration again**.
A current unexpired record can be confirmed using step 4. A listed expired record,
or evidence of earlier registration followed by current absence, means the lease
has ended for starting work. Otherwise report an indeterminate outcome. Missing
historical evidence or an absent current ID does not prove the attempt failed.

A timeout followed by an absent read-back is indeterminate. Do not delete its
transaction evidence as failed-attempt cleanup: the manifest may still arrive.
A local clock before expiration or a foreign transaction in a reclaimed slot
also cannot prove that earlier registration never succeeded and ended.
Clients needing restart recovery must save the original request and target
identity before submission and bound their retry attempts.

Before explicit release, the owner or operator **must stop** the job's
registration and retry loop. Release does not cancel storage writes already in
flight or fence a worker that continues submitting. Starting a new registration
requires a fresh ID and its newly confirmed source.

## Renewal, release, and expiration

Every attached commit **must derive lease state from its attempt's predecessor**.
Sample current UTC time, remove records with `expires_at` at or before that time,
apply explicit lease changes, and preserve all remaining fields. After conflicts,
repeat against the new predecessor; an old removal decision must not overwrite
a renewal. Restore and overwrite use current membership, never a historical list.

- **RenewLease** requires a listed, unexpired target at each attempt's validation.
  Change only `expires_at`, and only to a later value. An already satisfied
  request is a no-op; an absent or expired target is an error.
- **ReleaseLease** removes its target ID. An absent ID is a successful no-op.
  Different jobs' IDs are independent, even when their source files overlap.

Commit order settles renewal versus expiration. If renewal extends the deadline
first, a competing commit must honor it. If removal commits first, renewal
cannot recreate the lease. A validated write may finish after its sampled deadline;
wall-clock passage alone does not remove a stored record.

No-op lease requests create no version just to expire other leases. Real mutations
use ordinary version counting. Cleanup creates no expiration commit: on an idle
table, protection remains until another attached commit or explicit release.

## Publication and release

An ordinary attached transaction may set `Transaction.release_lease_id` (field 5)
to remove one lease atomically with its operation. The publication transaction has
its own UUID. An absent field requests no release; an absent target makes only
the release a no-op. The field is invalid on lease operations, Clone, and detached commits.

Apply the operation's existing validation and conflict rules. Its result and
release **must commit together or neither commits**; a conflict rechecks both
against the new predecessor. Intermediate publications keep their lease unless
explicit release or expiration removes it.

Once removal commits, the lease's retention guarantee ends. Publication validation
and recovery belong to the consumer and publication machinery. An absent lease
is not proof that results were published. A caller using standalone release
must separately confirm publication first, or abandon the work.

## Retention and data operations

### Retained versions

Let F be the oldest source among the current leases and H the current head.
Cleanup **must retain every attached version from F through H**, including
expired leases still listed. Detached versions are outside this window.

Keep each retained manifest, its required commit-store discovery records, and
all referenced resources: data, deletion files, overlays, indices, FRI metadata,
mapping files, and transaction sidecars. These must remain readable versions,
not manifests whose data has been deleted. Shared files are retained once.

Every attached commit whose result contains leases, or which carries
`release_lease_id`, **must publish transaction evidence with its manifest**, inline
or in an already stored referenced sidecar. This includes release of the last
lease or an already absent lease. Reject the commit if neither representation is available. Retain
transaction evidence after F through H; the transaction that created F is not
required solely by this retention rule.

Cleanup must validate required manifests and FRI metadata and establish the
presence of required transactions. For inline evidence, check the offset and
length-prefixed section bounds without decoding its payload; prefer inline
when both representations exist. For sidecars, use a complete authoritative
listing or an existence check. An age-filtered or partial listing, or an earlier
cached read, is insufficient. Present but malformed payloads may be discovered
later when a consumer reads them; cleanup does not audit every transaction.

Missing required versions or carriers, invalid metadata, or unresolved reads
**must stop affected destructive cleanup** and report a protection failure.
Unresolvable source fragments must also stop affected rewrites and FRI trimming.
These failures do not release the lease; release by a valid ID remains available.

Cleanup preview and execution must report each branch's listed lease count,
expired count awaiting removal, and oldest protected source. Byte attribution
among overlapping requirements is unnecessary. After committed removal,
re-resolve current retention before reclaiming files; other leases, tags, retained
versions, and consumers may still need them. This full-version policy can retain
intermediate data generations as well as metadata.

### Required FRI history

Resolve each lease source's complete fragment metadata and use all its fragment
IDs. Read a fragment tree when the source declares one; an empty inline fragment
list is not empty coverage. No row-data scan is needed. Reject acquisition for
unsupported layouts and refuse affected maintenance if required metadata cannot
be resolved.

Retain the union of lease and existing-consumer requirements. An index catching
up or a consumer-check error **must not discard** lease-required history.

- **Legacy-only FRI (`index_version = 0`):** retain applicable current-chain
  records stamped at or after the oldest lease source, including that boundary.
  Stamp new records with their successful publishing version minus one;
  restamp on retry and preserve existing stamps.
- **FRI version 1:** start with the lease source fragment IDs. Select each
  current-history record whose source intersects the set, add its destination
  IDs, and repeat until no more records are selected. Keep whole selected
  records. If one is a legacy record, include every group's destinations and
  their downstream records. Version 1 transitions have no dataset-version stamp.

Compaction **must record** the movements required by surviving leases, even when
no published index needs them. Every rewrite or trim **must recheck** current
requirements before committing, including after conflicts. Exclude only leases
whose removal that same commit will publish. Reject a result lacking required
mappings and replan; planning before registration is not an exemption.

Mapping files remain protected while any retained version or required history
references them, including files added after registration. Replacing FRI metadata
does not release those files. Existing overlay and index correctness rules still apply.

### Restore, branches, and detached versions

Restore and overwrite preserve surviving leases and their full retention windows.
Apply FRI retention to the resulting current history; these operations need not
reconstruct discarded translation paths. Retention does not guarantee that a
job can still translate or publish after these operations. Removing a current
FRI record does not reclaim mappings referenced by retained versions.

New branches and clones start with empty lease lists and preserve an inherited
capability flag. Detached versions likewise serialize an empty list and inherited
flag, without changing branch membership. They cannot contain lease operations or
`release_lease_id`. Normal and force branch deletion must follow the
[lease checks](branch_tag.md#deletion-with-retention-leases).

## Compatibility and implementation boundary

Bit 20 remains reserved and unsupported until lease maintenance is implemented.
Writers without support must reject it. Activation requires lease-aware maintenance
clients and completion or cancellation of incompatible destructive work already
in flight; a flag cannot revoke old handles or stop an operation that never checks it.

The minimum identified Rust cleanup flag-check boundary is `v14.0.0-beta.4`
(commit `84d1681e07af565837f2d97e72ff1b8b040cd15c`), not `beta.3`. That check
rejects an unknown lease bit; it does not implement leases. Assess Python/JNI by
their embedded core revision and verify other maintenance clients separately.

## Cleanup and deletion races

Registration does not cancel deletion already approved by an earlier inspection.
For example, cleanup can select an old file, a user can restore a version that
references it, and a job can then lease that source before cleanup deletes the file.
Applications coordinating restore and cleanup can finish running cleanup first
and pause new cleanup until lease confirmation; leases do not enforce that ordering.

Branch deletion has the same inspection/deletion gap. It can miss a later lease,
and delayed deletion after recovery and branch-name reuse can damage a replacement
branch. Leases inherit these existing races and do not change branch-name reuse.
Jobs should register before reading and preserve ordinary cleanup grace protections.
