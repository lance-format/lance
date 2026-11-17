// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compare CDF fragment pruning with equivalent full scans.
//!
//! Run with `cargo bench -p lance --bench cdf --profile release-with-debug`.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::types::Int32Type;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::TryStreamExt;
use lance::Dataset;
use lance::dataset::{UpdateBuilder, WriteParams};
use lance_core::utils::tempfile::TempStrDir;
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION, WILDCARD};
use lance_datagen::{BatchCount, RowCount, array, gen_batch};

// Small fragments isolate planning and file-open overhead.
const ROWS_PER_FRAGMENT: usize = 16;

#[derive(Clone, Copy, Debug)]
enum Query {
    Inserted,
    Updated,
    Upserted,
    AllInserted,
}

#[derive(Clone, Copy, Debug)]
enum ScanPath {
    Full,
    Pruned,
}

struct Fixture {
    _dir: TempStrDir,
    dataset: Dataset,
}

impl Fixture {
    async fn new(fragments: u32) -> Self {
        let dir = TempStrDir::default();
        let reader = gen_batch()
            .col("key", array::step::<Int32Type>())
            .col("value", array::fill_utf8("initial".into()))
            .into_reader_rows(
                RowCount::from(ROWS_PER_FRAGMENT as u64),
                BatchCount::from(fragments),
            );
        let mut dataset = Dataset::write(
            reader,
            dir.as_str(),
            Some(WriteParams {
                max_rows_per_file: ROWS_PER_FRAGMENT,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), fragments as usize);

        let reader = gen_batch()
            .col(
                "key",
                array::step_custom::<Int32Type>(fragments as i32 * ROWS_PER_FRAGMENT as i32, 1),
            )
            .col("value", array::fill_utf8("inserted".into()))
            .into_reader_rows(
                RowCount::from(ROWS_PER_FRAGMENT as u64),
                BatchCount::from(1),
            );
        dataset.append(reader, None).await.unwrap();
        let result = UpdateBuilder::new(Arc::new(dataset))
            .update_where("key = 0")
            .unwrap()
            .set("value", "'updated'")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap();
        Self {
            _dir: dir,
            dataset: result.new_dataset.as_ref().clone(),
        }
    }
}

async fn scan(dataset: &Dataset, query: Query, path: ScanPath) -> usize {
    let begin = if matches!(query, Query::AllInserted) {
        0
    } else {
        1
    };
    let mut stream = match path {
        ScanPath::Full => {
            let mut scanner = dataset.scan();
            scanner
                .project(&[
                    WILDCARD,
                    ROW_ID,
                    ROW_CREATED_AT_VERSION,
                    ROW_LAST_UPDATED_AT_VERSION,
                ])
                .unwrap();
            let inserted =
                format!("_row_created_at_version > {begin} AND _row_created_at_version <= 3");
            let updated = "_row_created_at_version <= 1 AND _row_last_updated_at_version > 1 AND _row_last_updated_at_version <= 3";
            let filter = match query {
                Query::Inserted | Query::AllInserted => inserted,
                Query::Updated => updated.to_string(),
                Query::Upserted => format!("({inserted}) OR ({updated})"),
            };
            scanner.filter(&filter).unwrap();
            scanner.try_into_stream().await.unwrap()
        }
        ScanPath::Pruned => {
            let delta = dataset
                .delta()
                .with_begin_version(begin)
                .with_end_version(3)
                .build()
                .unwrap();
            match query {
                Query::Inserted | Query::AllInserted => delta.get_inserted_rows().await.unwrap(),
                Query::Updated => delta.get_updated_rows().await.unwrap(),
                Query::Upserted => delta.get_upserted_rows().await.unwrap(),
            }
        }
    };
    let mut rows = 0;
    while let Some(batch) = stream.try_next().await.unwrap() {
        rows += batch.num_rows();
    }
    // Dev-dependencies enable lance-io's `test-util`, which keeps a record of every
    // I/O request until drained. Left to grow, that log slows each later scan, so
    // sequential Criterion runs would misattribute the drift to whichever path ran last.
    dataset
        .object_store(None)
        .await
        .unwrap()
        .io_stats_incremental();
    rows
}

fn bench_cdf(c: &mut Criterion) {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("cdf");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    for fragments in [100, 1_000, 20_000] {
        let fixture = runtime.block_on(Fixture::new(fragments));
        for query in [
            Query::Inserted,
            Query::Updated,
            Query::Upserted,
            Query::AllInserted,
        ] {
            let expected = match query {
                Query::Inserted => ROWS_PER_FRAGMENT,
                Query::Updated => 1,
                Query::Upserted => ROWS_PER_FRAGMENT + 1,
                Query::AllInserted => (fragments as usize + 1) * ROWS_PER_FRAGMENT,
            };
            for path in [ScanPath::Full, ScanPath::Pruned] {
                assert_eq!(
                    runtime.block_on(scan(&fixture.dataset, query, path)),
                    expected
                );
                group.bench_function(
                    BenchmarkId::new(format!("{query:?}/{path:?}"), fragments),
                    |b| b.iter(|| runtime.block_on(scan(&fixture.dataset, query, path))),
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_cdf);
criterion_main!(benches);
