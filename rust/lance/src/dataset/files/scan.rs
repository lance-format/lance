// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The shared manifest walk behind [`Dataset::tracked_files`] and
//! [`Dataset::referenced_files`].
//!
//! Both need the same thing: every present manifest, read with bounded memory
//! and bounded parallelism, together with the index metadata stored alongside
//! it. They differ only in what they build from it, so the walk lives here and
//! each caller materializes its own result.
//!
//! ```text
//! Lister ──► tx_locations ──► Reader ──► tx_manifest ──► caller's stream
//! ```
//!
//! The reader keeps several manifests in flight but stops launching reads once
//! the estimated in-flight size reaches [`MANIFEST_MEMORY_BUDGET`]. The budget
//! is charged for as long as the consumer holds a [`ScannedManifest`], so
//! dropping each one after use is what keeps the pipeline moving.
//!
//! The bound is on the reader's prefetch, not on what a consumer chooses to
//! retain: one read is always allowed when nothing is in flight, so a consumer
//! that holds every manifest degrades the walk to serial reads rather than
//! stopping it. That escape hatch is what keeps a manifest larger than the whole
//! budget from deadlocking the walk.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::stream::{BoxStream, FuturesUnordered};
use futures::{Future, StreamExt};
use lance_core::Result;
use lance_table::format::{IndexMetadata, Manifest};
use lance_table::io::commit::ManifestLocation;
use lance_table::io::manifest::{read_manifest, read_manifest_indexes};
use object_store::path::Path;

use super::strip_prefix;
use crate::Dataset;

/// Memory budget for in-flight manifests (estimated in-memory size).
const MANIFEST_MEMORY_BUDGET: usize = 1024 * 1024 * 1024; // 1 GB
/// Estimated ratio of in-memory size to on-disk size for manifests. Found
/// empirically; manifests are protobuf with significant decompression and
/// allocator overhead once parsed.
const MANIFEST_DECOMPRESSION_RATIO: usize = 4;

// A `ManifestLocation` is ~100 bytes, so a 50k-slot mpsc channel costs ~5 MB
// in the worst case. That's enough headroom for the lister to run well ahead
// of the reader on datasets with hundreds of thousands of manifests, while
// still bounding memory.
const MAX_BUFFERED_LOCATIONS: usize = 50_000;

/// Releases a manifest's share of the memory budget and wakes the reader.
///
/// Held by [`ScannedManifest`] so the budget is returned when the caller drops
/// it, whether or not the caller remembers to.
struct MemoryPermit {
    bytes: usize,
    inflight: Arc<AtomicUsize>,
    notify: Arc<tokio::sync::Notify>,
}

impl Drop for MemoryPermit {
    fn drop(&mut self) {
        self.inflight.fetch_sub(self.bytes, Ordering::AcqRel);
        self.notify.notify_one();
    }
}

/// One manifest produced by [`scan_manifests`], with what was read alongside it.
pub struct ScannedManifest {
    pub manifest: Arc<Manifest>,
    /// The manifest's own path, relative to the dataset root.
    pub manifest_path: String,
    /// Index metadata from this manifest's index section. Empty when it has none.
    pub indexes: Vec<IndexMetadata>,
    // Order matters: dropping the permit last means the budget is returned only
    // after `manifest` is freed.
    _permit: MemoryPermit,
}

/// A running manifest walk.
pub struct ManifestScan {
    /// Manifests in completion order, which is not version order.
    pub stream: BoxStream<'static, Result<ScannedManifest>>,
    /// Number of manifests the walk will yield. Set once listing finishes, so a
    /// consumer reading it mid-walk may still see `None`.
    pub total: Arc<std::sync::OnceLock<usize>>,
    /// Estimated in-memory bytes the reader has read and the consumer has not
    /// yet dropped. Test-only: production consumers rely on the budget
    /// implicitly, by dropping each manifest after use, so keeping this in
    /// release builds would be a field nothing reads.
    #[cfg(test)]
    inflight_bytes: Arc<AtomicUsize>,
}

#[cfg(test)]
impl ManifestScan {
    /// Estimated in-memory bytes currently charged against the budget.
    fn inflight_bytes(&self) -> usize {
        self.inflight_bytes.load(Ordering::Acquire)
    }

    /// The budget counter itself, for assertions that outlive `stream`.
    fn inflight_handle(&self) -> Arc<AtomicUsize> {
        self.inflight_bytes.clone()
    }
}

/// Walk every present manifest of `dataset`.
///
/// `min_version`, when set, skips manifests older than that version. Note that
/// this makes the result an incomplete view of what the dataset references, so
/// a caller building a deletion predicate must leave it unset.
pub fn scan_manifests(dataset: &Dataset, min_version: Option<u64>) -> ManifestScan {
    let base = dataset.base.clone();
    let object_store = dataset.object_store.clone();
    let commit_handler = dataset.commit_handler.clone();

    let (tx_manifest, rx_manifest) = tokio::sync::mpsc::channel::<Result<ScannedManifest>>(2);
    let (tx_locations, rx_locations) =
        tokio::sync::mpsc::channel::<ManifestLocation>(MAX_BUFFERED_LOCATIONS);

    let inflight_mem = Arc::new(AtomicUsize::new(0));
    let mem_notify = Arc::new(tokio::sync::Notify::new());
    let total: Arc<std::sync::OnceLock<usize>> = Arc::new(std::sync::OnceLock::new());

    spawn_lister(
        commit_handler,
        object_store.clone(),
        base.clone(),
        min_version,
        tx_locations,
        total.clone(),
        tx_manifest.clone(),
    );
    spawn_reader(
        object_store,
        base,
        rx_locations,
        tx_manifest,
        &inflight_mem,
        mem_notify,
    );

    ManifestScan {
        stream: tokio_stream::wrappers::ReceiverStream::new(rx_manifest).boxed(),
        total,
        #[cfg(test)]
        inflight_bytes: inflight_mem,
    }
}

/// Lists manifest locations, applies `min_version`, and records the total.
///
/// Locations are small, so they are buffered generously to let the lister run
/// ahead of the reader.
fn spawn_lister(
    commit_handler: Arc<dyn lance_table::io::commit::CommitHandler>,
    object_store: Arc<lance_io::object_store::ObjectStore>,
    base: Path,
    min_version: Option<u64>,
    tx_locations: tokio::sync::mpsc::Sender<ManifestLocation>,
    total: Arc<std::sync::OnceLock<usize>>,
    tx_err: tokio::sync::mpsc::Sender<Result<ScannedManifest>>,
) {
    tokio::spawn(async move {
        let result: Result<()> = async {
            let mut locations = commit_handler.list_manifest_locations(&base, &object_store, false);
            let mut count = 0usize;
            while let Some(location) = locations.next().await {
                let location = location?;
                if let Some(min_version) = min_version
                    && location.version < min_version
                {
                    continue;
                }
                count += 1;
                if tx_locations.send(location).await.is_err() {
                    // The consumer went away; stop listing.
                    return Ok(());
                }
            }
            let _ = total.set(count);
            Ok(())
        }
        .await;
        if let Err(error) = result {
            let _ = tx_err.send(Err(error)).await;
        }
    });
}

/// Reads manifests with memory-aware parallelism.
///
/// Read failures travel as `Err` items in the stream rather than ending the
/// walk, so this task itself is infallible.
fn spawn_reader(
    object_store: Arc<lance_io::object_store::ObjectStore>,
    base: Path,
    mut rx_locations: tokio::sync::mpsc::Receiver<ManifestLocation>,
    tx_manifest: tokio::sync::mpsc::Sender<Result<ScannedManifest>>,
    inflight_mem: &Arc<AtomicUsize>,
    mem_notify: Arc<tokio::sync::Notify>,
) {
    let inflight_mem = inflight_mem.clone();
    tokio::spawn(async move {
        let max_parallelism = object_store.io_parallelism();
        type ScanResult = Result<ScannedManifest>;
        let mut in_flight: FuturesUnordered<
            std::pin::Pin<Box<dyn Future<Output = ScanResult> + Send>>,
        > = FuturesUnordered::new();
        let mut locations_exhausted = false;

        loop {
            // Always allow one read even when over budget, or a single
            // manifest larger than the budget would deadlock the walk.
            let can_launch = !locations_exhausted
                && in_flight.len() < max_parallelism
                && (in_flight.is_empty()
                    || inflight_mem.load(Ordering::Acquire) < MANIFEST_MEMORY_BUDGET);

            if in_flight.is_empty() && !can_launch {
                break;
            }

            tokio::select! {
                biased;
                // Always drain completed reads first.
                Some(scanned) = in_flight.next(), if !in_flight.is_empty() => {
                    // The consumer went away; stop reading.
                    if tx_manifest.send(scanned).await.is_err() {
                        return;
                    }
                }
                location = rx_locations.recv(), if can_launch => {
                    match location {
                        Some(location) => {
                            let estimated = location.size.unwrap_or(0) as usize
                                * MANIFEST_DECOMPRESSION_RATIO;
                            inflight_mem.fetch_add(estimated, Ordering::AcqRel);
                            let permit = MemoryPermit {
                                bytes: estimated,
                                inflight: inflight_mem.clone(),
                                notify: mem_notify.clone(),
                            };

                            let object_store = object_store.clone();
                            let base = base.clone();
                            in_flight.push(Box::pin(async move {
                                let manifest = read_manifest(
                                    &object_store,
                                    &location.path,
                                    location.size,
                                )
                                .await?;
                                let indexes = read_manifest_indexes(
                                    &object_store,
                                    &location,
                                    &manifest,
                                )
                                .await?;
                                Ok(ScannedManifest {
                                    manifest: Arc::new(manifest),
                                    manifest_path: strip_prefix(&location.path, &base)?
                                        .to_string(),
                                    indexes,
                                    _permit: permit,
                                })
                            }));
                        }
                        None => locations_exhausted = true,
                    }
                }
                // Wake up when a consumer frees budget by dropping a manifest.
                _ = mem_notify.notified(), if !can_launch && !in_flight.is_empty() => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};

    fn simple_batch() -> impl arrow_array::RecordBatchReader {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        RecordBatchIterator::new(vec![Ok(batch)], schema)
    }

    async fn dataset_with_three_versions(uri: &str) -> Dataset {
        let mut dataset = Dataset::write(simple_batch(), uri, None).await.unwrap();
        dataset.append(simple_batch(), None).await.unwrap();
        dataset.append(simple_batch(), None).await.unwrap();
        dataset
    }

    /// The budget must return to zero once the consumer drops every manifest.
    /// A leaked permit would leave it charged and eventually stall the reader.
    #[tokio::test]
    async fn budget_returns_to_zero_after_consuming() {
        let dataset = dataset_with_three_versions("memory://scan_budget_zero").await;
        let mut scan = scan_manifests(&dataset, None);

        let mut seen = 0usize;
        while let Some(scanned) = scan.stream.next().await {
            scanned.unwrap();
            seen += 1;
        }

        assert_eq!(seen, 3, "expected every present manifest");
        assert_eq!(
            scan.inflight_bytes(),
            0,
            "dropping every ScannedManifest must return the whole budget"
        );
    }

    /// Holding manifests keeps the budget charged, which is the signal the
    /// reader throttles on. It does not stop the walk: one read is always
    /// allowed when nothing is in flight, so a hoarding consumer gets serial
    /// reads rather than a stall.
    #[tokio::test]
    async fn holding_manifests_keeps_budget_charged() {
        let dataset = dataset_with_three_versions("memory://scan_budget_held").await;
        let mut scan = scan_manifests(&dataset, None);

        let mut held = Vec::new();
        while let Some(scanned) = scan.stream.next().await {
            held.push(scanned.unwrap());
        }
        assert!(
            scan.inflight_bytes() > 0,
            "held manifests must still be charged against the budget"
        );

        drop(held);
        assert_eq!(
            scan.inflight_bytes(),
            0,
            "the budget must come back when the consumer lets go"
        );
    }

    /// `min_version` really does skip manifests, which is why a keep-set must
    /// leave it unset.
    #[tokio::test]
    async fn min_version_skips_older_manifests() {
        let dataset = dataset_with_three_versions("memory://scan_min_version").await;

        let mut versions = Vec::new();
        let ManifestScan { mut stream, .. } = scan_manifests(&dataset, Some(3));
        while let Some(scanned) = stream.next().await {
            versions.push(scanned.unwrap().manifest.version);
        }

        assert_eq!(versions, vec![3], "min_version must drop versions 1 and 2");
    }

    /// Dropping the stream early must not leave the reader running: the closed
    /// channel is what tells it to stop. Observed through the budget returning
    /// to zero, which happens only once every in-flight permit is released.
    #[tokio::test]
    async fn dropping_the_stream_releases_every_permit() {
        let dataset = dataset_with_three_versions("memory://scan_drop_early").await;
        let scan = scan_manifests(&dataset, None);
        // Keep the budget handle after the stream goes away.
        let inflight = scan.inflight_handle();
        let mut stream = scan.stream;

        // Hold the first manifest so the budget is provably charged. Without
        // this the assertion below could pass on a walk that never charged
        // anything.
        let first = stream.next().await.expect("at least one manifest").unwrap();
        assert!(
            inflight.load(Ordering::Acquire) > 0,
            "holding a manifest must charge the budget"
        );

        // Drop the stream while the walk may still have reads in flight, then
        // release our own manifest.
        drop(stream);
        drop(first);

        // The reader unwinds asynchronously, so poll rather than assume it has
        // already observed the closed channel. Ten seconds is far longer than
        // this needs locally and is only here so a loaded machine reports a
        // real failure instead of a flake.
        let mut released = false;
        for _ in 0..1000 {
            if inflight.load(Ordering::Acquire) == 0 {
                released = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            released,
            "dropping the stream must release every in-flight permit; \
             timed out waiting for the reader to unwind"
        );
    }

    /// A manifest that cannot be read surfaces as an `Err` item in the stream
    /// rather than being skipped. A skipped manifest would make the walk
    /// silently incomplete, which for a deletion predicate means authorizing the
    /// deletion of files only that manifest still references. The reader keeps
    /// going after a failure; it is the consumer that decides whether to stop.
    #[tokio::test]
    async fn read_failure_surfaces_as_an_error_item() {
        use crate::dataset::builder::DatasetBuilder;
        use crate::dataset::{ObjectStoreParams, ReadParams};
        use crate::utils::test::FailingProxyStore;

        // A real store, not `memory://`: the failing proxy wraps the store, and
        // re-opening with a wrapper changes the registry cache key, so an
        // in-memory reopen would land on a fresh empty store and lose the three
        // versions this test needs.
        let dir = tempfile::tempdir().unwrap();
        let uri = dir.path().to_str().unwrap();
        drop(dataset_with_three_versions(uri).await);

        // Install the proxy at open time but arm it only afterwards: opening
        // reads the latest manifest itself, so failing that read would break the
        // open rather than the walk under test.
        let failing = Arc::new(FailingProxyStore::new());
        let dataset = DatasetBuilder::from_uri(uri)
            .with_read_params(ReadParams {
                store_options: Some(ObjectStoreParams {
                    object_store_wrapper: Some(failing.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .load()
            .await
            .unwrap();
        failing.fail_when("get_opts", "_versions", "injected manifest read failure");

        let mut scan = scan_manifests(&dataset, None);
        let mut errors = 0usize;
        let mut successes = 0usize;
        while let Some(scanned) = scan.stream.next().await {
            match scanned {
                Ok(_) => successes += 1,
                Err(_) => errors += 1,
            }
        }

        // One Err per manifest, not one for the whole walk: that is what pins
        // the reader continuing after a failure. A listing failure would give a
        // single Err instead, so this also proves the failure came from the
        // reads rather than from listing.
        assert_eq!(
            successes, 0,
            "no manifest read can succeed while every `_versions` read fails"
        );
        assert_eq!(
            errors, 3,
            "each of the three manifests must surface its own read error"
        );
        assert_eq!(
            scan.inflight_bytes(),
            0,
            "a failed read must return its share of the budget"
        );
    }

    /// The total is the number of manifests the walk will yield, available once
    /// listing finishes.
    #[tokio::test]
    async fn total_counts_every_yielded_manifest() {
        let dataset = dataset_with_three_versions("memory://scan_total").await;
        let ManifestScan {
            mut stream, total, ..
        } = scan_manifests(&dataset, None);

        let mut seen = 0usize;
        while let Some(scanned) = stream.next().await {
            scanned.unwrap();
            seen += 1;
        }

        assert_eq!(total.get().copied(), Some(seen));
    }
}
