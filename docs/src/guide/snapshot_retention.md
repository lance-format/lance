# Snapshot retention

`SnapshotIdentity` and `SnapshotRetention` provide explicit snapshot protection for
ordinary scans and deferred blob reads. They share Lance's existing tag-based GC
roots. Opening a Dataset, creating a scanner, or serializing an identity does not
protect files from cleanup.

`SnapshotIdentity` captures the table incarnation, canonical table root, branch,
exact version, original manifest bytes, and authoritative commit identity when
available. It does not contain blob capabilities, worker routing keys, or storage
credentials. Workers resolve authenticated storage independently.

## Acquire and release

Acquire before issuing work that depends on the snapshot. Use a globally unique
`retention_id` for each query lifetime. Acquisition with the same identity and ID
is idempotent. Never reuse an ID after release or share it between independent
queries: either owner could release their shared protection.

```python
import uuid
from lance.retention import SnapshotIdentity, SnapshotRetention

snapshot = SnapshotIdentity.from_dataset(dataset, "catalog/table-incarnation")
retention = SnapshotRetention.acquire(dataset, snapshot, str(uuid.uuid4()))
try:
    result = dataset.to_table()
finally:
    # First stop all distributed workers, retries, outstanding reads, and lazy consumers.
    retention.release(dataset)
```

Rust exposes the same types at `lance::SnapshotIdentity` and
`lance::SnapshotRetention`. Java exposes `org.lance.SnapshotIdentity` and
`org.lance.SnapshotRetention`.

```rust
use lance::{Dataset, Result, SnapshotIdentity, SnapshotRetention};

async fn query(dataset: &Dataset, query_id: &str) -> Result<()> {
    let snapshot = SnapshotIdentity::from_dataset(dataset, "catalog/table-incarnation").await?;
    let retention = SnapshotRetention::acquire(dataset, snapshot, query_id).await?;
    // Run scans and wait for all consumers and retries to stop.
    retention.release(dataset).await
}
```

The durable tag is named `lance-query-<digest>`, derived from the generic snapshot
identity and retention ID. Cleanup preserves the protected snapshot and its
referenced files. Configure cleanup to clean around old tagged versions rather
than erroring on them; Python uses `error_if_tagged_old_versions=False`.

`release(dataset)` is idempotent and accepts an authenticated handle to a later
snapshot of the same table. Encoding, decoding, or dropping a token does not
release protection. **Query tags have no automatic expiration or renewal.**

Blob source contexts expose `snapshot_identity()` in Rust/Python and
`getSnapshotIdentity()` in Java. Different worker routing keys and external-blob
capabilities do not change that snapshot identity. The existing
`BlobSnapshotRetention` and Java `BlobReferences.Retention` APIs remain convenience
adapters to the same general retention implementation.

## Maintenance lock expiration

Acquisition, release, and cleanup coordinate through a table-root maintenance
lock across branches and processes. The lock is held only for the individual
operation; cleanup holds it for its entire run. Workers reading retained files
do not hold the maintenance lock.

Maintenance owners have a **fixed lifetime of 120 hours** by default. Set the
positive integer table configuration `lance.snapshot_retention.lock_expiration_hours`
to change the lifetime of new owners. Zero, negative, fractional, malformed, and
overflowing values are rejected before an owner is created.

```python
dataset.update_config({"lance.snapshot_retention.lock_expiration_hours": "24"})
```

```rust
use lance::{Dataset, Result};

async fn configure(dataset: &mut Dataset) -> Result<()> {
    dataset.update_config([
        ("lance.snapshot_retention.lock_expiration_hours", "24"),
    ]).await?;
    Ok(())
}
```

Java callers can set the same key through `UpdateConfig` and the Dataset
transaction builder. Settings are read from the Dataset handle's loaded manifest;
reopen older handles to use a newly committed configuration. A contender's setting
never shortens an existing owner's recorded lifetime.

Each attempt creates its own object under
`_refs/retention-v1/maintenance/<owner-UUID>`. Its JSON contains an immutable
`expires_at` timestamp and a `ticket`. Ticket zero announces that the owner is
choosing an ordering number. Published `(ticket, UUID)` pairs establish priority.
Contenders return `RefConflict`; retry the whole operation with backoff.

Later attempts ignore expired owners and try to remove their objects, including
owners abandoned while choosing a ticket. Removal uses cleanup's shared delete
limits when initiated by cleanup. A delayed release deletes only its own UUID,
so it cannot remove a successor's owner object. There are no heartbeats or renewal.
Snapshot tags and retirement markers are unaffected by owner expiration.

!!! warning

    Fixed expiration is a bounded-operation deployment contract, not
    storage-enforced fencing. Configure the lifetime to exceed the duration of
    every participating operation, process pause, retry, and submitted storage
    request. Client clocks must agree sufficiently for that contract. Owners
    check their deadline before protected writes and deletes and after awaited
    work; an expired owner returns an error. These checks cannot close the gap
    between a check and a storage operation or cancel a request already submitted.
    A paused owner resuming after takeover can violate retention if those bounds
    are not enforced by the deployment. Deployments requiring recovery across
    arbitrary pauses or partitions need stronger fencing or atomic pin/retirement
    coordination. Increasing the timeout reduces exposure but does not supply
    that stronger guarantee.

## Cleanup coordination and supported deployments

Under the maintenance lock, acquisition revalidates the exact manifest and commit
identity before creating its durable tag. Cleanup reads protected tags and marks
expiring manifests before deleting payloads. If cleanup wins, acquisition fails.
A retirement marker prevents later acquisition when partial cleanup has deleted
payloads but left their manifest on storage. Markers are cleared only after the
corresponding manifest has been deleted.

Every cleanup client must participate in the current protocol, including automatic
and branch cleanup. Stop or exclude older clients and already-open older handles
before enabling this guarantee. A capability check cannot revoke an older client's
permission to delete files. Forced table/branch deletion, manual tag modification,
and deletion outside Lance are excluded.

Storage must provide atomic conditional creation and consistent reads/listing.
Tencent COS retention is unsupported; ordinary cleanup remains available there.
Acquisition assumes payloads are initially available and does not verify every
object or repair earlier deletion or corruption.

Protection covers files and objects owned by this table. Snapshots with foreign
dataset roots, including shallow clones, need separate owner retention and are
rejected. External objects and registered bases owned elsewhere retain their own
immutability/lifetime contracts.

`SnapshotMaintenanceLock` is available as a Rust hook for maintenance coordinators.
Call `check_valid()` before protected work and bound all submitted operations.
Do not call acquisition, release, or cleanup while already holding the hook: those
operations acquire the same lock themselves.

## Recovery after interruption

The query coordinator supplies a durable recovery journal; Lance does not track
distributed workers or write that journal on the application's behalf.

| Durable phase | Recorded state | Next operation |
| --- | --- | --- |
| Acquiring | Globally unique query ID and encoded exact `SnapshotIdentity` | Acquire with that identity and ID. |
| Active | Serialized token, tag name, and participating workers/retries | Issue work. |
| Releasing | Token and confirmation that all consumers have stopped | Release the token. |
| Released | Acknowledged release and completed query ID | Complete the record and never reuse that ID. |

Persist each phase before starting the next operation. Acquisition intent covers a
crash after tag creation but before its response or token is saved. If only intent
is available, reopen the exact recorded snapshot and replay acquisition with the
same identity and ID, then release. If a token exists, decode and release using an
authenticated handle to the same table; a later snapshot is sufficient. Recover a
lost release response by retrying release, never by reacquiring the query.

Stop assignments and retries and drain all workers, lazy consumers, outstanding
reads, and submitted storage work before release. Orphaned query tags continue to
protect files until explicitly released, even after maintenance owners expire.
Monitor tags without active recovery records and repeated maintenance conflicts.

An abandoned maintenance owner stops blocking at its recorded expiration. Manual
removal before expiry still requires proof that the operation and its submitted
storage work have stopped. If ownership cannot be established, quiesce participating
coordinators and maintenance clients for the table and drain requests first.
Never delete the whole `_refs` directory.

Retirement markers are separate from owner objects. Never remove one to make a
partly deleted snapshot acquirable. Finish deleting its expired manifest under
maintenance coordination before clearing the marker. A marker whose manifest is
already absent may be removed after maintenance has stopped.
