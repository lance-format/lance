# Retention Leases

A **retention lease** keeps a version and everything needed to reach the current
head from it. While the lease is listed in the current manifest:

1. File cleanup **must retain** every version from the leased version through
   the current head, including the files and transactions those versions need.
2. Fragment Reuse Index (FRI) cleanup **must retain** the transitions needed
   to connect the leased version's fragments to the current layout.

This lets compaction and cleanup run while a holder builds an index or another
result from the leased version. Unlike a tag, which retains a snapshot
indefinitely, a lease can be renewed and expires if its holder stops renewing.
The next attached commit removes an expired lease; until then, cleanup still
honors it. Committing the transaction the lease was acquired for releases it; the holder
can also explicitly release a lease when its work is abandoned.

Leases retain resources. The holder remains responsible for validating and
publishing its result and protecting uncommitted output. Leases cannot cancel
[deletion already in progress](#cleanup-and-deletion-races).

## Persisted state

`Manifest.retention_leases` (field 27) stores the lease list. Only the current
attached manifest's list is authoritative; historical lists cannot revive
removed leases. Missing or empty means no leases, and list order has no meaning.
A lease is **listed** while its record is in the current list, including after
it expires; listed leases retain resources.

<details>
<summary>Lease protobuf message</summary>

```protobuf
%%% proto.message.lance.table.RetentionLease %%%
```

</details>

Writers and maintenance **must reject** incomplete records, invalid field
values, and duplicate IDs. A corrupt list blocks lease mutations, including
release, and affected destructive maintenance; it requires administrator repair.
A valid lease can still be released if its leased version is unreadable.

Lease operations **must set** writer
flag 20 (`FLAG_RETENTION_LEASES`). A nonempty list without it is invalid.
Once set, the flag **must remain set**, including when all leases are removed and
through restore or clone. Leases require no additional reader flag; readers may
ignore the list.

## Identity and time

A lease's ID is the UUID of the transaction its holder intends to publish.
Acquiring a lease declares that intent: retain the leased version until a
transaction with that UUID commits, the lease is released, or it expires.
The holder generates a fresh UUID for each acquisition; it **must differ** from
the acquisition transaction's own UUID. Retries keep the same ID. The lease
belongs to one table and branch; its `version` is the version created by its
successful acquisition commit.

`created_at` and `expires_at` are absolute UTC times. Holders choose both before
the first acquisition attempt and preserve them on retries. A renewal likewise
fixes its requested `expires_at` once. Times must be valid protobuf timestamps,
compared at full precision, and expiration must be later than creation.

Holders use reasonably synchronized clocks and renew before expiration. The
format specifies no default duration, renewal schedule, or maximum lifetime.

## Acquisition and recovery

Lease decisions use the [authoritative current head](versioning.md#current-state-for-lease-decisions).
To acquire lease P:

1. If P is already listed, recover its stored version, expiration, and label without changing them.
2. For a fresh submission, require an unexpired request. Resolve head H, build
   version H+1 from it, and add P with `version = H+1`.
3. Publish atomically. A definitively lost attempt may retry against a newer
   head with the same ID and times. Recompute the version on each uncommitted
   attempt; once acquisition succeeds, the version stays fixed.
4. Before confirming acquisition, verify current unexpired membership with its stored version V, and a readable manifest for
   V. Return V. The holder **must wait for confirmation** before planning or
   reading V.

The leased version is the one the acquisition commit creates, not one the
holder chooses. An older version's files or FRI transitions could already be
removed or approved for deletion; the newly created head cannot be.

After an uncertain response, recovery **must not submit the acquisition again**.
A current unexpired record can be confirmed using step 4. A listed expired record,
or evidence of earlier acquisition followed by current absence, means the lease
can no longer be used to start work. Otherwise report an indeterminate outcome.
Missing historical evidence or an absent current ID does not prove the attempt
failed.

A timeout followed by an absent read-back is indeterminate. Do not delete its
transaction file as failed-attempt cleanup: the manifest may still arrive.
A local clock before expiration or a foreign transaction in a reclaimed slot
also cannot prove that earlier acquisition never succeeded and was removed.
Holders needing restart recovery must save the original request and target
identity before submission and bound their retry attempts.

Before explicit release, the holder or operator **must stop** the holder's
acquisition and retry loop. Release does not cancel storage writes already in
flight or fence a holder that continues submitting. Starting a new acquisition
requires a fresh ID and its newly confirmed version.

## Renewal, release, and expiration

Every attached commit **must derive lease state from its attempt's predecessor**.
Sample current UTC time, remove records with `expires_at` at or before that time,
apply explicit lease changes, and preserve all remaining fields. After conflicts,
repeat against the new predecessor; an old removal decision must not overwrite
a renewal. Restore and overwrite use the current list, never a historical list.

- **RenewLease** requires a listed, unexpired target at each attempt's validation.
  Change only `expires_at`, and only to a later value. An already satisfied
  request is a no-op; an absent or expired target is an error.
- **ReleaseLease** removes its target ID. An absent ID is a successful no-op.
  Different leases are independent, even when their retained files overlap.

Commit order settles renewal versus expiration. If renewal extends the deadline
first, a competing commit must honor it. If removal commits first, renewal
cannot recreate the lease. A validated write may finish after its sampled deadline;
wall-clock passage alone does not remove a stored record.

No-op lease requests create no version just to expire other leases. Real mutations
use ordinary version counting. Cleanup creates no expiration commit: on an idle
table, protection remains until another attached commit or explicit release.

## Publication and release

An ordinary attached commit whose `Transaction.uuid`, parsed as a UUID, equals a listed lease ID
removes that lease atomically with its operation. Lease operations, Clone, and
detached commits never release a lease this way. If no listed lease matches,
the commit proceeds unchanged.

Apply the operation's existing validation and conflict rules. Its result and
release **must commit together or neither commits**; a retry keeps its UUID and
rechecks both against the new predecessor. Publications with other UUIDs keep
the lease unless explicit release or expiration removes it.

Once removal commits, the lease's retention guarantee ends. Publication validation
and recovery belong to the holder and publication machinery. A retained
transaction with the lease's UUID shows that the holder's result was published;
an absent lease alone does not.

## Retention and data operations

### Retained versions

Let F be the oldest leased version among the listed leases and H the current
head. Cleanup **must retain every attached version from F through H**, including
for expired leases still listed. Detached versions are outside this window.

Keep each retained manifest, its required commit-store discovery records, and
all referenced resources: data, deletion files, overlays, indices, FRI metadata,
mapping files, and transaction files. These must remain readable versions,
not manifests whose data has been deleted. Shared files are retained once.

Every attached commit whose result lists leases, or which releases a lease,
**must record its [transaction](transaction.md)** with its manifest. This
includes release of the last lease.
Reject the commit if the transaction cannot be recorded. Retain the transactions
of versions after F through H; the transaction that created F is not required
solely by this retention rule.

Cleanup must validate required manifests and FRI metadata and establish that
each required transaction is present. A transaction file's presence requires a
complete authoritative listing or an existence check; an age-filtered or partial
listing, or an earlier cached read, is insufficient. Cleanup does not decode or
audit every transaction.

Missing required versions or transactions, invalid metadata, or unresolved reads
**must stop affected destructive cleanup** and report a protection failure.
An unresolvable leased version must also stop affected rewrites and FRI trimming.
These failures do not release the lease; release by a valid ID remains available.

Cleanup preview and execution must report each branch's listed lease count,
expired count awaiting removal, and oldest leased version. Byte attribution
among overlapping requirements is unnecessary. After committed removal,
re-resolve current retention before reclaiming files; other leases, tags, retained
versions, and FRI consumers may still need them. This full-version policy can
retain intermediate data generations as well as metadata.

### Required transitions

Resolve each leased version's complete fragment metadata and use all its fragment
IDs. Read a fragment tree when the version declares one; an empty inline fragment
list is not empty coverage. No row-data scan is needed. Reject acquisition for
unsupported layouts and refuse affected maintenance if required metadata cannot
be resolved.

Retain the union of the transitions that leases and existing FRI consumers
require. An index catching up or a consumer-check error **must not discard**
transitions a lease requires.

- **Legacy-only FRI (`index_version = 0`):** retain applicable current-chain
  reuse versions stamped at or after the oldest leased version, including that
  boundary. Stamp new reuse versions with their successful publishing version
  minus one; restamp on retry and preserve existing stamps.
- **FRI version 1:** start with the leased versions' fragment IDs. Select each
  current-history transition whose sources intersect the set, add its
  destination IDs, and repeat until no more transitions are selected. Keep whole
  selected transitions. If one is a legacy reuse version, include every group's
  destinations and their downstream transitions. Version 1 transitions have no
  dataset-version stamp.

Compaction **must record** the transitions required by listed leases, even when
no published index needs them. Every rewrite or trim **must recheck** current
requirements before committing, including after conflicts. Exclude only leases
whose removal that same commit will publish. Reject a result lacking required
transitions and replan; planning before acquisition is not an exemption.

Mapping files remain protected while any retained version or required transition
references them, including files added after acquisition. Replacing FRI metadata
does not release those files. Existing overlay and index correctness rules still apply.

### Restore, branches, and detached versions

Restore and overwrite preserve listed leases and their full retention windows.
Apply FRI retention to the resulting current history; these operations need not
reconstruct discarded transitions. Retention does not guarantee that a holder
can still translate or publish after these operations. Removing a current
FRI transition does not reclaim mappings referenced by retained versions.

New branches and clones start with empty lease lists and preserve an inherited
capability flag. Detached versions likewise serialize an empty list and inherited
flag, without changing branch membership. They cannot contain lease operations
or release leases. Branch deletion does not check leases; deleting a branch
ends its leases.

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

Acquisition does not cancel deletion already approved by an earlier inspection.
For example, cleanup can select an old file, a user can restore a version that
references it, and a holder can then lease that version before cleanup deletes
the file. Applications coordinating restore and cleanup can finish running
cleanup first and pause new cleanup until lease confirmation; leases do not
enforce that ordering.

Holders should acquire before reading and preserve ordinary cleanup grace protections.
