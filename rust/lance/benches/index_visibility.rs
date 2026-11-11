// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Planning and search costs for clean, deleted, sparse, dense, and segmented
//! scalar visibility, in both address and stable-ID datasets.
//!
//! ```text
//! LANCE_ENABLE_UNSTABLE_DATA_OVERLAY_FILES=1 cargo bench -p lance \
//!   --profile release-with-debug --bench index_visibility
//! ```

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{
    ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, UInt64Array, record_batch,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use lance::Dataset;
use lance::dataset::scanner::Scanner;
use lance::dataset::transaction::Operation;
use lance::dataset::{WriteDestination, WriteParams};
use lance::index::{CreateIndexBuilder, DatasetIndexExt};
use lance_core::utils::address::RowAddress;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;

const ROWS: usize = 100_000;
const ROWS_PER_FRAGMENT: usize = 25_000;

struct Fixture {
    _directory: TempStrDir,
    dataset: Dataset,
    expected_rows: usize,
}

impl Fixture {
    async fn new(
        stable_row_ids: bool,
        changed_rows: usize,
        segmented: bool,
        deleted: bool,
    ) -> Self {
        let directory = TempStrDir::default();
        let batch = record_batch!(
            ("id", Int32, (0..ROWS as i32).collect::<Vec<_>>()),
            ("age", Int32, (0..ROWS as i32).collect::<Vec<_>>())
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            directory.as_str(),
            Some(WriteParams {
                max_rows_per_file: ROWS_PER_FRAGMENT,
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let params = ScalarIndexParams::default();
        if segmented {
            let mut segments = Vec::with_capacity(ROWS / ROWS_PER_FRAGMENT);
            for fragment in 0..(ROWS / ROWS_PER_FRAGMENT) as u32 {
                segments.push(
                    CreateIndexBuilder::new(&mut dataset, &["age"], IndexType::BTree, &params)
                        .name("age_idx".into())
                        .fragments(vec![fragment])
                        .execute_uncommitted()
                        .await
                        .unwrap(),
                );
            }
            dataset
                .commit_existing_index_segments("age_idx", "age", segments)
                .await
                .unwrap();
        } else {
            dataset
                .create_index(&["age"], IndexType::BTree, None, &params, true)
                .await
                .unwrap();
        }
        if changed_rows > 0 {
            // Use fragment 1: stable IDs differ from physical addresses here.
            let field = dataset.schema().field("age").unwrap().id;
            let projection = dataset.schema().project_by_ids(&[field], true);
            let offsets = (0..changed_rows)
                .map(|offset| u64::from(RowAddress::new_from_parts(1, offset as u32)))
                .collect::<Vec<_>>();
            let patch = RecordBatch::try_from_iter(vec![
                ("_rowaddr", Arc::new(UInt64Array::from(offsets)) as ArrayRef),
                (
                    "age",
                    Arc::new(Int32Array::from(vec![-1; changed_rows])) as ArrayRef,
                ),
            ])
            .unwrap();
            let mut writer = dataset
                .get_fragment(1)
                .unwrap()
                .write_overlay(&projection)
                .await
                .unwrap();
            writer.write_batch(&patch).await.unwrap();
            let group = writer.finish().await.unwrap().unwrap();
            let version = dataset.version().version;
            dataset = Dataset::commit(
                WriteDestination::Dataset(Arc::new(dataset)),
                Operation::DataOverlay {
                    groups: vec![group],
                },
                Some(version),
                None,
                None,
                Arc::new(Default::default()),
                false,
            )
            .await
            .unwrap();
        }
        if deleted {
            dataset.delete("id = 25000").await.unwrap();
        }
        Self {
            _directory: directory,
            dataset,
            expected_rows: changed_rows + 1,
        }
    }

    fn scan(&self) -> Scanner {
        let mut scan = self.dataset.scan();
        scan.filter("age = 42 OR age = -1")
            .unwrap()
            .project(&["id"])
            .unwrap();
        scan
    }
}

#[expect(
    clippy::print_stderr,
    reason = "Report untimed I/O measurements alongside Criterion results."
)]
fn bench_visibility(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("index_visibility");
    for stable in [false, true] {
        for (name, changed, segmented, deleted) in [
            ("clean", 0, false, false),
            ("deleted", 0, false, true),
            ("sparse", 1, false, false),
            ("dense", 4096, false, false),
            ("segments", 1, true, false),
        ] {
            let fixture = runtime.block_on(Fixture::new(stable, changed, segmented, deleted));
            let store = runtime
                .block_on(fixture.dataset.object_store(None))
                .unwrap();
            let label = format!("{}/{name}", if stable { "stable" } else { "address" });
            runtime.block_on(async {
                assert_eq!(fixture.scan().try_into_batch().await.unwrap().num_rows(), fixture.expected_rows);
                store.io_stats_incremental();
                fixture.scan().create_plan().await.unwrap();
                let planning = store.io_stats_incremental();
                fixture.scan().try_into_batch().await.unwrap();
                let search = store.io_stats_incremental();
                // One warm measurement per fixture, separate from Criterion's timings.
                eprintln!("{label}: planning_reads={} planning_bytes={} search_reads={} search_bytes={} replay_rows={changed}",
                    planning.read_iops, planning.read_bytes, search.read_iops, search.read_bytes);
            });
            group.bench_with_input(BenchmarkId::new("plan", &label), &fixture, |b, fixture| {
                b.to_async(&runtime).iter(|| async {
                    fixture.scan().create_plan().await.unwrap();
                    store.io_stats_incremental();
                });
            });
            group.bench_with_input(
                BenchmarkId::new("search", &label),
                &fixture,
                |b, fixture| {
                    b.to_async(&runtime).iter(|| async {
                        let batch = fixture.scan().try_into_batch().await.unwrap();
                        assert_eq!(batch.num_rows(), fixture.expected_rows);
                        // Drain request records so their memory does not grow between iterations.
                        store.io_stats_incremental();
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(1))
        .sample_size(20);
    targets = bench_visibility
);
criterion_main!(benches);
