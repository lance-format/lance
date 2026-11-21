// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! End-to-end costs of the stable row id feature: cold index loading, take by
//! row id, prefilter mask construction, append commits and compaction, with
//! stable row ids on and off at matched scale.
//!
//! ```text
//! cargo bench -p lance --profile release-with-debug --bench stable_row_ids
//! ```
//!
//! Every dataset-level benchmark reports the latest manifest size, because an
//! inline `RowIdMeta` is part of the manifest every reader decodes and every
//! writer rewrites.
//!
//! ## Configuration
//!
//! - `BENCH_COLD_FRAGMENTS` / `BENCH_COLD_ROWS_PER_FRAGMENT`: the largest
//!   cold-index case (default 1000 x 1000); two smaller cases are always run.
//! - `BENCH_TAKE_FRAGMENTS` / `BENCH_TAKE_ROWS_PER_FRAGMENT`: take fixture.
//! - `BENCH_PREFILTER_FRAGMENTS` / `BENCH_PREFILTER_ROWS_PER_FRAGMENT`:
//!   prefilter fixture.
//! - `BENCH_APPEND_BASE_FRAGMENTS` / `BENCH_APPEND_ROWS_PER_FRAGMENT`: append
//!   fixture and the rows in each appended fragment.
//! - `BENCH_COMPACT_FRAGMENTS` / `BENCH_COMPACT_ROWS_PER_FRAGMENT`: compaction
//!   fixture. The target doubles the input fragment size so every group
//!   rewrites.
//!
//! A malformed value stops the benchmark instead of silently measuring a
//! smaller workload.

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema as ArrowSchema};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::rowids::get_row_id_index;
use lance::dataset::transaction::{Operation, TransactionBuilder};
use lance::dataset::{Dataset, ProjectionRequest, WriteDestination, WriteMode, WriteParams};
use lance::index::prefilter::DatasetPreFilter;
use lance::session::Session;
use lance_core::ROW_ID;
use lance_core::utils::tempfile::TempStrDir;
use lance_io::object_store::ObjectStoreRegistry;
use lance_table::format::ManifestBuildConfig;
use roaring::RoaringBitmap;
use tokio::runtime::Runtime;

const DEFAULT_FRAGMENTS: usize = 100;
const DEFAULT_ROWS_PER_FRAGMENT: usize = 10_000;
const DEFAULT_DELETE_PERCENT: usize = 10;

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .map(|value| value.parse().expect("benchmark size must be an integer"))
}

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn batch(start: i64, len: usize) -> RecordBatch {
    let ids = Int64Array::from_iter_values(start..(start + len as i64));
    let values = Int64Array::from_iter_values((start..(start + len as i64)).map(|id| id * 3));
    RecordBatch::try_new(schema(), vec![Arc::new(ids), Arc::new(values)]).unwrap()
}

fn batches(fragments: usize, rows_per_fragment: usize) -> Vec<RecordBatch> {
    (0..fragments)
        .map(|fragment| batch((fragment * rows_per_fragment) as i64, rows_per_fragment))
        .collect()
}

/// A fresh session with no caches: every measurement pays the real decode
/// rather than reading back what the previous one memoized.
fn cold_session() -> Arc<Session> {
    Arc::new(Session::new(0, 0, Arc::new(ObjectStoreRegistry::default())))
}

async fn open_cold(uri: &str) -> Dataset {
    DatasetBuilder::from_uri(uri)
        .with_session(cold_session())
        .load()
        .await
        .unwrap()
}

async fn write_fixture(
    label: &str,
    uri: &str,
    stable_row_ids: bool,
    fragments: usize,
    rows_per_fragment: usize,
    delete_percent: usize,
) -> Dataset {
    let data = batches(fragments, rows_per_fragment);
    let reader = RecordBatchIterator::new(data.into_iter().map(Ok), schema());
    let mut dataset = Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            mode: WriteMode::Create,
            max_rows_per_file: rows_per_fragment,
            max_rows_per_group: rows_per_fragment,
            enable_stable_row_ids: stable_row_ids,
            skip_auto_cleanup: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), fragments);
    assert_eq!(dataset.manifest.uses_stable_row_ids(), stable_row_ids);
    if delete_percent > 0 {
        dataset
            .delete(&format!("id % 100 < {delete_percent}"))
            .await
            .unwrap();
    }
    report_manifest(label, stable_row_ids, &dataset);
    dataset
}

fn inline_row_id_bytes(dataset: &Dataset) -> u64 {
    dataset
        .manifest
        .fragments
        .iter()
        .filter_map(|fragment| match &fragment.row_id_meta {
            Some(lance_table::format::RowIdMeta::Inline(bytes)) => Some(bytes.len() as u64),
            _ => None,
        })
        .sum()
}

fn report_manifest(label: &str, stable_row_ids: bool, dataset: &Dataset) {
    if label.is_empty() {
        return;
    }
    println!(
        "{label}: stable_row_ids={stable_row_ids} fragments={} rows={} manifest_bytes={} inline_row_id_bytes={}",
        dataset.manifest.fragments.len(),
        dataset
            .manifest
            .fragments
            .iter()
            .filter_map(|f| f.physical_rows)
            .sum::<usize>(),
        dataset.manifest_location().size.unwrap_or(0),
        inline_row_id_bytes(dataset)
    );
}

fn report_manifest_delta(label: &str, before: u64, dataset: &Dataset) {
    let after = dataset.manifest_location().size.unwrap_or(0);
    println!(
        "{label}: manifest_bytes={before}->{after} delta={} inline_row_id_bytes={}",
        after.saturating_sub(before),
        inline_row_id_bytes(dataset)
    );
}

/// Row ids of evenly spaced live rows, read through the public scan path so the
/// stable and address domains are both represented by real values.
async fn sample_row_ids(dataset: &Dataset, count: usize) -> Vec<u64> {
    let batch = dataset.scan().with_row_id().try_into_batch().await.unwrap();
    let ids = batch
        .column_by_name(ROW_ID)
        .unwrap()
        .as_primitive::<UInt64Type>();
    let rows = ids.len();
    (0..count)
        .map(|index| ids.value(index * rows / count.max(1)))
        .collect()
}

fn bench_cold_index_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let cases = [
        (10, 100_000),
        (100, 10_000),
        (
            env_usize("BENCH_COLD_FRAGMENTS").unwrap_or(1_000),
            env_usize("BENCH_COLD_ROWS_PER_FRAGMENT").unwrap_or(1_000),
        ),
    ];
    let mut group = c.benchmark_group("stable_row_id_cold_index");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(5));
    for (fragments, rows_per_fragment) in cases {
        let dir = TempStrDir::default();
        let label = format!("cold_index/{fragments}x{rows_per_fragment}");
        let dataset = rt.block_on(write_fixture(
            &label,
            dir.as_str(),
            true,
            fragments,
            rows_per_fragment,
            0,
        ));
        let first_id = rt.block_on(sample_row_ids(&dataset, 1))[0];
        let name = format!("fragments={fragments}/rows_per_fragment={rows_per_fragment}");
        group.bench_function(&name, |b| {
            b.to_async(&rt).iter(|| async {
                let dataset = open_cold(dir.as_str()).await;
                let index = get_row_id_index(&dataset).await.unwrap().unwrap();
                std::hint::black_box(index.get(first_id).unwrap());
            });
        });
    }
    group.finish();
}

fn bench_take(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let fragments = env_usize("BENCH_TAKE_FRAGMENTS").unwrap_or(DEFAULT_FRAGMENTS);
    let rows_per_fragment =
        env_usize("BENCH_TAKE_ROWS_PER_FRAGMENT").unwrap_or(DEFAULT_ROWS_PER_FRAGMENT);
    let mut group = c.benchmark_group("stable_row_id_take");
    group.sample_size(10);
    for stable_row_ids in [false, true] {
        let dir = TempStrDir::default();
        let label = format!("take/{}", if stable_row_ids { "stable" } else { "address" });
        let dataset = rt.block_on(write_fixture(
            &label,
            dir.as_str(),
            stable_row_ids,
            fragments,
            rows_per_fragment,
            0,
        ));
        let projection = ProjectionRequest::from_columns(["value"], dataset.schema());
        for count in [1, 100, 10_000] {
            let rows = rt.block_on(dataset.count_rows(None)).unwrap();
            let count = count.min(rows);
            let ids = rt.block_on(sample_row_ids(&dataset, count));
            group.throughput(Throughput::Elements(count as u64));
            group.bench_with_input(
                BenchmarkId::new(if stable_row_ids { "stable" } else { "address" }, count),
                &ids,
                |b, ids| {
                    let dataset = dataset.clone();
                    let projection = projection.clone();
                    b.to_async(&rt).iter(|| {
                        let dataset = dataset.clone();
                        let projection = projection.clone();
                        async move {
                            std::hint::black_box(dataset.take_rows(ids, projection).await.unwrap());
                        }
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_prefilter(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let fragments = env_usize("BENCH_PREFILTER_FRAGMENTS").unwrap_or(100);
    let rows_per_fragment = env_usize("BENCH_PREFILTER_ROWS_PER_FRAGMENT").unwrap_or(10_000);
    let mut group = c.benchmark_group("stable_row_id_prefilter");
    group.sample_size(10);
    for stable_row_ids in [false, true] {
        let dir = TempStrDir::default();
        let label = format!(
            "prefilter/{}",
            if stable_row_ids { "stable" } else { "address" }
        );
        rt.block_on(write_fixture(
            &label,
            dir.as_str(),
            stable_row_ids,
            fragments,
            rows_per_fragment,
            DEFAULT_DELETE_PERCENT,
        ));
        let fragment_ids = (0..fragments as u32).collect::<RoaringBitmap>();
        // A zero-capacity metadata cache keeps every sample cold: the mask is
        // built from freshly loaded deletion vectors and row id sequences
        // instead of being served from the per-version mask cache.
        group.bench_function(if stable_row_ids { "stable" } else { "address" }, |b| {
            b.to_async(&rt).iter(|| async {
                let dataset = open_cold(dir.as_str()).await;
                let mask =
                    DatasetPreFilter::create_deletion_mask(Arc::new(dataset), fragment_ids.clone())
                        .expect("the fixture has deletions");
                std::hint::black_box(mask.await.unwrap());
            });
        });
    }
    group.finish();
}

fn bench_append(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let base_fragments = env_usize("BENCH_APPEND_BASE_FRAGMENTS").unwrap_or(10);
    let rows_per_fragment = env_usize("BENCH_APPEND_ROWS_PER_FRAGMENT").unwrap_or(10_000);
    let mut group = c.benchmark_group("stable_row_id_append");
    group.sample_size(10);
    for stable_row_ids in [false, true] {
        for fragments_per_commit in [1, 10] {
            let data = batches(fragments_per_commit, rows_per_fragment);
            let label = format!(
                "append/{}/fragments_per_commit={fragments_per_commit}",
                if stable_row_ids { "stable" } else { "address" }
            );
            group.bench_function(
                BenchmarkId::new(
                    if stable_row_ids { "stable" } else { "address" },
                    fragments_per_commit,
                ),
                |b| {
                    let data = data.clone();
                    let label = label.clone();
                    b.to_async(&rt).iter_custom(move |iterations| {
                        let data = data.clone();
                        let label = label.clone();
                        async move {
                            let mut elapsed = Duration::ZERO;
                            let mut reported = false;
                            for _ in 0..iterations {
                                // A fresh base per sample, built outside the
                                // timer: appending repeatedly to one table would
                                // measure a manifest that keeps growing.
                                let dir = TempStrDir::default();
                                let base = write_fixture(
                                    "",
                                    dir.as_str(),
                                    stable_row_ids,
                                    base_fragments,
                                    rows_per_fragment,
                                    0,
                                )
                                .await;
                                let before = base.manifest_location().size.unwrap_or(0);
                                let reader = RecordBatchIterator::new(
                                    data.clone().into_iter().map(Ok),
                                    schema(),
                                );
                                let started = Instant::now();
                                let next = Dataset::write(
                                    reader,
                                    WriteDestination::Dataset(Arc::new(base)),
                                    Some(WriteParams {
                                        mode: WriteMode::Append,
                                        max_rows_per_file: rows_per_fragment,
                                        enable_stable_row_ids: stable_row_ids,
                                        skip_auto_cleanup: true,
                                        ..Default::default()
                                    }),
                                )
                                .await
                                .unwrap();
                                elapsed += started.elapsed();
                                if !reported {
                                    report_manifest_delta(&label, before, &next);
                                    reported = true;
                                }
                                std::hint::black_box((next, dir));
                            }
                            elapsed
                        }
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_compaction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let fragments = env_usize("BENCH_COMPACT_FRAGMENTS").unwrap_or(20);
    let rows_per_fragment = env_usize("BENCH_COMPACT_ROWS_PER_FRAGMENT").unwrap_or(10_000);
    let mut group = c.benchmark_group("stable_row_id_compact");
    group.sample_size(10);
    for stable_row_ids in [false, true] {
        for delete_percent in [0, DEFAULT_DELETE_PERCENT, 50] {
            let label = format!(
                "compact/{}/deleted={delete_percent}%",
                if stable_row_ids { "stable" } else { "address" }
            );
            group.bench_function(
                BenchmarkId::new(
                    if stable_row_ids { "stable" } else { "address" },
                    delete_percent,
                ),
                |b| {
                    let label = label.clone();
                    b.to_async(&rt).iter_custom(move |iterations| {
                        let label = label.clone();
                        async move {
                            let mut elapsed = Duration::ZERO;
                            let mut reported = false;
                            for _ in 0..iterations {
                                // A fresh fixture per sample, built outside the
                                // timer: the first compaction rewrites every
                                // group, and a second one on the same table would
                                // have no work.
                                let dir = TempStrDir::default();
                                let mut dataset = write_fixture(
                                    "",
                                    dir.as_str(),
                                    stable_row_ids,
                                    fragments,
                                    rows_per_fragment,
                                    delete_percent,
                                )
                                .await;
                                let before = dataset.manifest_location().size.unwrap_or(0);
                                let options = CompactionOptions {
                                    target_rows_per_fragment: rows_per_fragment * 2,
                                    materialize_deletions: true,
                                    materialize_deletions_threshold: 0.0,
                                    ..Default::default()
                                };
                                let started = Instant::now();
                                std::hint::black_box(
                                    compact_files(&mut dataset, options, None).await.unwrap(),
                                );
                                elapsed += started.elapsed();
                                if !reported {
                                    report_manifest_delta(&label, before, &dataset);
                                    reported = true;
                                }
                            }
                            elapsed
                        }
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_assign_row_ids(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let rows_per_fragment =
        env_usize("BENCH_ASSIGN_ROWS_PER_FRAGMENT").unwrap_or(DEFAULT_ROWS_PER_FRAGMENT);
    let mut group = c.benchmark_group("stable_row_id_assign");
    group.sample_size(10);
    for fragments in [1_usize, 10, 100] {
        let dir = TempStrDir::default();
        let dataset = rt.block_on(write_fixture(
            "",
            dir.as_str(),
            true,
            1,
            rows_per_fragment,
            0,
        ));
        let source_fragment = dataset.manifest.fragments[0].clone();
        let template = (0..fragments)
            .map(|_| {
                let mut fragment = source_fragment.clone();
                // An append hands the writer fragments without lineage; the
                // commit assigns it fresh ids and row id sequences.
                fragment.id = 0;
                fragment.row_id_meta = None;
                fragment.created_at_version_meta = None;
                fragment.last_updated_at_version_meta = None;
                fragment.deletion_file = None;
                fragment
            })
            .collect::<Vec<_>>();
        group.bench_with_input(
            BenchmarkId::from_parameter(fragments),
            &template,
            |b, template| {
                b.iter(|| {
                    let transaction = TransactionBuilder::new(
                        dataset.version().version,
                        Operation::Append {
                            fragments: template.clone(),
                        },
                    )
                    .build();
                    // A commit assigns fresh ids and serializes one row id
                    // sequence per appended fragment.
                    let (manifest, _) = transaction
                        .build_manifest(
                            Some(dataset.manifest.as_ref()),
                            vec![],
                            "txn",
                            &ManifestBuildConfig {
                                auto_set_feature_flags: false,
                                timestamp_nanos: 0,
                                use_stable_row_ids: true,
                                use_legacy_format: None,
                                storage_format: None,
                                disable_transaction_file: true,
                                migration_next_row_id: None,
                                activate_non_reusable_field_ids: false,
                                clustering_definitions: None,
                                spilled_row_lineage: Default::default(),
                            },
                        )
                        .unwrap();
                    std::hint::black_box(manifest);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cold_index_load,
    bench_take,
    bench_prefilter,
    bench_append,
    bench_compaction,
    bench_assign_row_ids
);
criterion_main!(benches);
