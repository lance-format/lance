# Retention Leases

A long job that reads one version of a dataset, such as an index build that
runs for hours, competes with routine maintenance. Compaction rewrites the
fragments the job is reading, and cleanup deletes the old versions and files
the job needs to publish its result. Without coordination, the job either
blocks maintenance or fails when it commits.

A **retention lease** removes that trade-off. While a lease is held, cleanup
keeps the leased version and every later one, and compaction records how it
moved rows so the job's result can be translated onto the current fragments
when it commits. Maintenance keeps running; the job publishes as if nothing
had changed.

A lease ends when the holder publishes its result, releases the lease, or stops
renewing it. See the [Retention Leases specification](../format/table/retention_leases.md)
for the exact rules.

!!! warning "Upgrade every writer first"

    While any lease is held, Lance versions without lease support refuse to
    write to the dataset, so that they cannot delete what a lease retains.
    Upgrade every writer, including scheduled cleanup and compaction jobs,
    before using leases. Once no lease is held, older writers work again.

## Building an index under a lease

For an index built in one call, ask the build to manage a lease. It acquires
the lease, builds on the version the lease retains, renews the lease in the
background, and publishes the index for it. If the build fails, the lease is
released.

=== "Python"

    ```python
    import lance
    from datetime import timedelta

    dataset = lance.dataset("s3://bucket/table.lance")
    dataset.create_scalar_index(
        "category",
        "BTREE",
        retention_lease=lance.LeaseOptions(ttl=timedelta(minutes=30)),
    )
    ```

=== "Rust"

    ```rust
    use lance::dataset::LeaseOptions;
    use lance::index::DatasetIndexExt;
    use lance_index::{IndexType, scalar::ScalarIndexParams};

    let params = ScalarIndexParams::default();
    dataset
        .create_index_builder(&["category"], IndexType::BTree, &params)
        .retention_lease(LeaseOptions::default())
        .await?;
    ```

The time to live (TTL) bounds how long cleanup keeps retaining history if the
job dies without releasing its lease. The build renews the lease well before
it expires, so a TTL of minutes is enough for a build of any length.

## Distributed index builds

In a [distributed build](distributed_indexing.md), the coordinator holds the
lease. It acquires the lease, keeps it alive with a `LeaseKeeper` while the
workers run, and publishes the workers' segments for the lease. The lease can
be pickled or serialized, so the coordinator ships it to the workers, which
build on the leased version.

=== "Python"

    ```python
    import lance

    dataset = lance.dataset(uri)
    lease, _ = dataset.acquire_retention_lease(label="nightly vector index")

    # On each worker, with the lease shipped from the coordinator:
    def build_segment(uri, lease, fragment_ids):
        leased = lance.dataset(uri).checkout_lease(lease)
        return leased.create_index_uncommitted(
            "vector", "IVF_PQ", name="vector_idx", fragment_ids=fragment_ids
        )

    with lance.LeaseKeeper(dataset, lease) as keeper:
        segments = run_workers(build_segment, uri, lease)  # your scheduler
        if keeper.lost_reason is not None:
            print(f"lease lost, commit may fail: {keeper.lost_reason}")
        dataset.commit_existing_index_segments(
            "vector_idx", "vector", segments, lease=lease
        )
    ```

=== "Rust"

    ```rust
    use lance::dataset::{LeaseKeeper, LeaseOptions};
    use lance::index::DatasetIndexExt;

    let (lease, _) = dataset.acquire_retention_lease(LeaseOptions::default()).await?;
    let keeper = LeaseKeeper::start(&dataset, lease.clone());
    // Workers build segments on `dataset.checkout_lease(&lease)`.
    let segments = run_workers(&lease).await?;
    dataset
        .commit_existing_index_segments_for_lease("vector_idx", "vector", segments, &lease)
        .await?;
    ```

The commit validates the segments against the leased version and rebases them
over everything committed since, including compactions. Publishing releases the
lease. To abandon the work instead, call `release()` on the keeper.

## Other long jobs

Any job can publish its result for a lease. Commit the result with
`Transaction.for_lease(lease, operation)` in Python, or
`TransactionBuilder::for_lease(&lease, operation)` in Rust. The transaction
reads the leased version and releases the lease when it commits.

## When a lease is lost

The keeper never stops the job. If a renewal fails temporarily, it retries
before the lease expires. If the lease expires or is released by someone else,
the keeper stops renewing and reports why (`lost_reason` in Python,
`LeaseStatus::Lost` in Rust); a managed build logs this when it publishes.

Publishing for a lost lease still commits if nothing it needs was removed.
If cleanup already deleted the files or versions the job relied on, the commit
fails with a clear error and the job must be rerun. A new lease cannot rescue
the job, because it would retain a newer version than the one the job read.

## Effects on maintenance

- **Cleanup** keeps every version from the oldest leased version through the
  latest, along with recently written files that nothing references yet, such
  as staged index segments. When leases make cleanup keep versions its policy
  would have removed, it logs a warning that lists the leases responsible.
- **Compaction** records how rows moved in a
  [fragment reuse index](../format/index/system/frag_reuse.md) while a lease is
  held, and defers remapping existing indices to it. Queries keep using those
  indices through the fragment reuse index until the next index optimization.
  If the dataset had no fragment reuse index, compaction creates one. That
  sets a feature flag that stays after the lease ends, so Lance versions
  without fragment reuse support can no longer read or write the dataset.
- **Restore and overwrite** end every lease, since a job built on an older
  version cannot be published over them.
