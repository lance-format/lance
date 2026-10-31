// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark for dataset cleanup.
//!
//! Two scenarios:
//!
//! - `orphans`: a dataset with `N` recent, unreferenced data files. Cleanup
//!   with `delete_unverified` removes them.
//! - `versions`: a dataset with `V` versions, each with one data file.
//!   Cleanup removes every version but the latest.
//!
//! The benchmark also counts object-store delete requests. This lets a
//! one-request-per-file implementation be compared with a bulk-delete
//! implementation without changing the benchmark: `delete_paths` is the number
//! of files cleanup deletes, while `delete_requests` is the number of
//! underlying requests. On a one-request-per-file implementation the two are
//! equal; on a bulk-delete implementation `delete_requests` is much smaller.
//! The counts are printed once per scenario.
//!
//! ## Configuration
//!
//! - `CLEANUP_BENCH_ORPHANS`: number of orphan files (default: 2000).
//! - `CLEANUP_BENCH_VERSIONS`: number of dataset versions (default: 200).
//! - `CLEANUP_BENCH_SAMPLE_SIZE`: criterion sample size (default: 10).
//! - `CLEANUP_BENCH_SKIP_ORPHANS`: skip the orphan scenario when truthy.
//! - `CLEANUP_BENCH_SKIP_VERSIONS`: skip the version scenario when truthy.
//!
//! Run with:
//!
//! ```bash
//! cargo bench --bench cleanup
//! ```

#![allow(clippy::print_stderr)]

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{ArrowError, DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use bytes::Bytes;
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::stream::{BoxStream, StreamExt};
use lance::dataset::cleanup::{CleanupPolicy, RemovalStats};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta,
    ObjectStore as OSObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    RenameOptions, Result as OSResult,
};
use tempfile::TempDir;
use tokio::runtime::Runtime;

/// Delete requests observed at the raw object-store layer.
#[derive(Debug, Default)]
struct DeleteCounters {
    delete_requests: AtomicU64,
    delete_paths: AtomicU64,
}

impl DeleteCounters {
    fn reset(&self) {
        self.delete_requests.store(0, Ordering::Relaxed);
        self.delete_paths.store(0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> DeleteCountSnapshot {
        DeleteCountSnapshot {
            delete_requests: self.delete_requests.load(Ordering::Relaxed),
            delete_paths: self.delete_paths.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DeleteCountSnapshot {
    delete_requests: u64,
    delete_paths: u64,
}

/// A pass-through object store that counts how cleanup deletes files.
///
/// Every delete eventually goes through `delete_stream` at this layer, so the
/// counter sees one request per single-file delete and one request per bulk
/// delete. The wrapper does not change which paths are deleted, so the same
/// benchmark works for both implementations.
#[derive(Debug)]
struct CountingStore {
    inner: Arc<dyn OSObjectStore>,
    counters: Arc<DeleteCounters>,
}

impl std::fmt::Display for CountingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CountingStore({})", self.inner)
    }
}

#[async_trait]
impl OSObjectStore for CountingStore {
    async fn put_opts(
        &self,
        location: &Path,
        bytes: PutPayload,
        opts: PutOptions,
    ) -> OSResult<PutResult> {
        self.inner.put_opts(location, bytes, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOptions,
    ) -> OSResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OSResult<GetResult> {
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<u64>]) -> OSResult<Vec<Bytes>> {
        self.inner.get_ranges(location, ranges).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, OSResult<Path>>,
    ) -> BoxStream<'static, OSResult<Path>> {
        self.counters
            .delete_requests
            .fetch_add(1, Ordering::Relaxed);
        let counters = Arc::clone(&self.counters);
        let counted = locations
            .inspect(move |location| {
                if location.is_ok() {
                    counters.delete_paths.fetch_add(1, Ordering::Relaxed);
                }
            })
            .boxed();
        self.inner.delete_stream(counted)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    fn list_with_offset(
        &self,
        prefix: Option<&Path>,
        offset: &Path,
    ) -> BoxStream<'static, OSResult<ObjectMeta>> {
        self.inner.list_with_offset(prefix, offset)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OSResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(&self, from: &Path, to: &Path, options: CopyOptions) -> OSResult<()> {
        self.inner.copy_opts(from, to, options).await
    }

    async fn rename_opts(&self, from: &Path, to: &Path, options: RenameOptions) -> OSResult<()> {
        self.inner.rename_opts(from, to, options).await
    }
}

#[derive(Debug, Clone)]
struct CountingStoreWrapper {
    counters: Arc<DeleteCounters>,
}

impl WrappingObjectStore for CountingStoreWrapper {
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn OSObjectStore>,
    ) -> Arc<dyn OSObjectStore> {
        Arc::new(CountingStore {
            inner: original,
            counters: Arc::clone(&self.counters),
        })
    }

    // The wrapper only observes deletes, so listings can keep their pushdown.
    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        Some(original)
    }
}

fn one_row_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let values = Int64Array::from(vec![0]);
    RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap()
}

fn batch_reader() -> RecordBatchIterator<Vec<Result<RecordBatch, ArrowError>>> {
    let batch = one_row_batch();
    let schema = batch.schema();
    RecordBatchIterator::new(vec![Ok(batch)], schema)
}

/// Creates a dataset with `versions` versions through the wrapped store.
///
/// Each version overwrites the table so that every version has its own data
/// file. Appending would keep every earlier data file referenced by the latest
/// manifest, leaving cleanup nothing to delete besides manifests. The
/// `file-object-store://` scheme keeps writes on the object-store path so that
/// the delete counter observes cleanup's requests.
async fn write_dataset(dir: &TempDir, counters: Arc<DeleteCounters>, versions: usize) -> Dataset {
    let uri = format!("file-object-store://{}", dir.path().join("db").display());
    let write_params = || WriteParams {
        mode: WriteMode::Overwrite,
        store_params: Some(ObjectStoreParams {
            object_store_wrapper: Some(Arc::new(CountingStoreWrapper {
                counters: Arc::clone(&counters),
            })),
            ..Default::default()
        }),
        skip_auto_cleanup: true,
        ..Default::default()
    };
    let mut dataset = Dataset::write(batch_reader(), &uri, Some(write_params()))
        .await
        .unwrap();

    for _ in 1..versions {
        dataset = Dataset::write(batch_reader(), &uri, Some(write_params()))
            .await
            .unwrap();
    }
    dataset
}

async fn setup_versions(versions: usize) -> (TempDir, Dataset, Arc<DeleteCounters>) {
    let dir = TempDir::new().unwrap();
    let counters = Arc::new(DeleteCounters::default());
    let dataset = write_dataset(&dir, Arc::clone(&counters), versions.max(1)).await;
    (dir, dataset, counters)
}

async fn setup_orphans(orphans: usize) -> (TempDir, Dataset, Arc<DeleteCounters>) {
    let dir = TempDir::new().unwrap();
    let counters = Arc::new(DeleteCounters::default());
    let dataset = write_dataset(&dir, Arc::clone(&counters), 1).await;
    let data_dir = dataset.data_dir();
    let store = dataset.object_store(None).await.unwrap();

    let writes = futures::stream::iter(0..orphans).map(move |i| {
        let store = Arc::clone(&store);
        let path = data_dir.clone().join(format!("orphan-{i:08}.lance"));
        async move {
            store.put(&path, &[0u8; 64]).await.unwrap();
        }
    });
    writes.buffer_unordered(64).collect::<Vec<_>>().await;

    (dir, dataset, counters)
}

fn versions_policy(dataset: &Dataset) -> CleanupPolicy {
    CleanupPolicy {
        before_version: Some(dataset.version().version),
        // The benchmark has no concurrent writers, so recent data files from
        // expired versions can be removed along with their manifests.
        delete_unverified: true,
        error_if_tagged_old_versions: false,
        ..Default::default()
    }
}

fn orphans_policy() -> CleanupPolicy {
    CleanupPolicy {
        delete_unverified: true,
        ..Default::default()
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
}

fn report_once(
    reported: &AtomicBool,
    scenario: &str,
    size: usize,
    stats: &RemovalStats,
    counts: DeleteCountSnapshot,
) {
    if !reported.swap(true, Ordering::Relaxed) {
        eprintln!(
            "cleanup {scenario} (n={size}): delete_requests={} \
             delete_paths={} | old_versions={} data_files_removed={} \
             transaction_files_removed={} index_files_removed={} \
             failed_deletes={}",
            counts.delete_requests,
            counts.delete_paths,
            stats.old_versions,
            stats.data_files_removed,
            stats.transaction_files_removed,
            stats.index_files_removed,
            stats.failed_deletes,
        );
    }
}

fn bench_cleanup(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let sample_size = env_usize("CLEANUP_BENCH_SAMPLE_SIZE", 10).max(10);
    let mut group = c.benchmark_group("cleanup");
    group.sample_size(sample_size);

    if !env_flag("CLEANUP_BENCH_SKIP_ORPHANS") {
        let orphans = env_usize("CLEANUP_BENCH_ORPHANS", 2000).max(1);
        let reported = AtomicBool::new(false);
        group.bench_function(BenchmarkId::new("orphans", orphans), |b| {
            b.iter_batched(
                || runtime.block_on(setup_orphans(orphans)),
                |(dir, dataset, counters)| {
                    counters.reset();
                    let stats = runtime
                        .block_on(dataset.cleanup(orphans_policy()).execute())
                        .unwrap();
                    let counts = counters.snapshot();
                    report_once(&reported, "orphans", orphans, &stats, counts);
                    // Return the fixture so Criterion drops it after the timed
                    // region; otherwise recursive temp-dir removal is charged
                    // to cleanup.
                    (dir, dataset, stats, counts)
                },
                BatchSize::PerIteration,
            );
        });
    }

    if !env_flag("CLEANUP_BENCH_SKIP_VERSIONS") {
        let versions = env_usize("CLEANUP_BENCH_VERSIONS", 200).max(2);
        let reported = AtomicBool::new(false);
        group.bench_function(BenchmarkId::new("versions", versions), |b| {
            b.iter_batched(
                || runtime.block_on(setup_versions(versions)),
                |(dir, dataset, counters)| {
                    counters.reset();
                    let stats = runtime
                        .block_on(dataset.cleanup(versions_policy(&dataset)).execute())
                        .unwrap();
                    let counts = counters.snapshot();
                    report_once(&reported, "versions", versions, &stats, counts);
                    (dir, dataset, stats, counts)
                },
                BatchSize::PerIteration,
            );
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_cleanup
}
criterion_main!(benches);
