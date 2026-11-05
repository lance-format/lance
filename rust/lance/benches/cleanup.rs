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
//! The benchmark also counts calls made through the wrapped object store. This
//! lets a one-request-per-file implementation be compared with a bulk-delete
//! implementation without changing the benchmark: `delete_paths` is the number
//! of files cleanup deletes, while `delete_requests` is the number of
//! `delete_stream` calls the wrapper observes. On a one-request-per-file
//! implementation the two are equal; on a bulk-delete implementation
//! `delete_requests` is much smaller. These are not physical cloud HTTP
//! requests: provider retries, request splitting, and network latency are not
//! represented. The counts are printed once per scenario.
//!
//! ## Configuration
//!
//! - `CLEANUP_BENCH_ORPHANS`: number of orphan files (default: 2000, minimum: 1).
//! - `CLEANUP_BENCH_VERSIONS`: number of dataset versions (default: 200, minimum: 2).
//! - `CLEANUP_BENCH_SAMPLE_SIZE`: criterion sample size (default: 10, minimum: 10).
//! - `CLEANUP_BENCH_SKIP_ORPHANS`: set to 1/true/yes to skip the orphan scenario.
//! - `CLEANUP_BENCH_SKIP_VERSIONS`: set to 1/true/yes to skip the version scenario.
//!
//! Malformed or out-of-range values are rejected instead of silently falling
//! back to a default or resizing the requested workload. Skip flags also accept
//! 0/false/no to run the scenario; other values are rejected.
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

/// Delete work observed at the wrapped object-store layer.
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
/// counter sees one call per single-file delete and one call per bulk
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

/// Builds a `file-object-store` URI without letting a Windows drive letter
/// become the URL authority.
///
/// This follows the local URI construction used by the cleanup tests: normalize
/// separators and ensure the path starts with `/` after the scheme.
fn file_object_store_uri(path: &std::path::Path) -> String {
    let path = path.display().to_string().replace('\\', "/");
    let path_prefix = if path.starts_with('/') { "" } else { "/" };
    format!("file-object-store://{path_prefix}{path}")
}

/// Creates a dataset with `versions` versions through the wrapped store.
///
/// Each version overwrites the table so that every version has its own data
/// file. Appending would keep every earlier data file referenced by the latest
/// manifest, leaving cleanup nothing to delete besides manifests. The
/// `file-object-store://` scheme keeps writes on the object-store path so that
/// the delete counter observes cleanup's requests.
async fn write_dataset(dir: &TempDir, counters: Arc<DeleteCounters>, versions: usize) -> Dataset {
    let uri = file_object_store_uri(&dir.path().join("db"));
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
    let dataset = write_dataset(&dir, Arc::clone(&counters), versions).await;
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

fn env_usize(name: &str, default: usize, minimum: usize) -> usize {
    let value = match std::env::var(name) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{name} must be an integer, got {value:?}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("failed to read {name}: {error}"),
    };
    assert!(
        value >= minimum,
        "{name} must be at least {minimum}, got {value}"
    );
    value
}

fn env_flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.as_str() {
            "1" | "true" | "TRUE" | "yes" | "YES" => true,
            "0" | "false" | "FALSE" | "no" | "NO" => false,
            _ => panic!("{name} must be one of 1, true, yes, 0, false, no; got {value:?}"),
        },
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => panic!("failed to read {name}: {error}"),
    }
}

fn report_once(
    reported: &AtomicBool,
    scenario: &str,
    size: usize,
    stats: &RemovalStats,
    counts: DeleteCountSnapshot,
) {
    if !reported.swap(true, Ordering::Relaxed) {
        assert_eq!(
            stats.failed_deletes, 0,
            "{scenario} cleanup reported failed deletes"
        );
        assert_eq!(
            counts.delete_paths,
            stats.old_versions
                + stats.data_files_removed
                + stats.transaction_files_removed
                + stats.index_files_removed
                + stats.deletion_files_removed,
            "wrapped-store delete paths must match the reported removals"
        );
        match scenario {
            "orphans" => assert_eq!(
                stats.data_files_removed, size as u64,
                "orphans cleanup must remove every orphan data file"
            ),
            "versions" => {
                assert_eq!(
                    stats.old_versions,
                    size as u64 - 1,
                    "versions cleanup must remove every version but the latest"
                );
                assert_eq!(
                    stats.data_files_removed,
                    size as u64 - 1,
                    "versions cleanup must remove one data file per expired version"
                );
            }
            _ => {}
        }
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
    let sample_size = env_usize("CLEANUP_BENCH_SAMPLE_SIZE", 10, 10);
    let mut group = c.benchmark_group("cleanup");
    group.sample_size(sample_size);

    if !env_flag("CLEANUP_BENCH_SKIP_ORPHANS") {
        let orphans = env_usize("CLEANUP_BENCH_ORPHANS", 2000, 1);
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
        let versions = env_usize("CLEANUP_BENCH_VERSIONS", 200, 2);
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
