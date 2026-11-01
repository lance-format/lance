// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{hint::black_box, sync::Arc, time::Duration};

use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use futures::{FutureExt, StreamExt, TryStreamExt, future, stream};
use lance_io::ReadBatchParams;
use lance_table::{
    rowids::{
        segment::U64Segment,
        version::{RowDatasetVersionRun, RowDatasetVersionSequence},
    },
    utils::stream::{
        ReadBatchTask, ReadBatchTaskStream, RowIdAndDeletesConfig, apply_row_id_and_deletes,
        wrap_with_row_id_and_delete,
    },
};

fn make_version_sequence(total_rows: u64, run_length: u64) -> RowDatasetVersionSequence {
    let runs = (0..total_rows)
        .step_by(run_length as usize)
        .enumerate()
        .map(|(run_index, start)| RowDatasetVersionRun {
            span: U64Segment::Range(start..(start + run_length).min(total_rows)),
            version: run_index as u64 + 1,
        })
        .collect();
    RowDatasetVersionSequence { runs }
}

fn make_batch(batch_size: u64) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::UInt64,
            false,
        )])),
        vec![Arc::new(UInt64Array::from(vec![0; batch_size as usize]))],
    )
    .unwrap()
}

fn make_config(total_rows: u64, sequence: Arc<RowDatasetVersionSequence>) -> RowIdAndDeletesConfig {
    RowIdAndDeletesConfig {
        params: ReadBatchParams::RangeFull,
        with_row_id: false,
        with_row_addr: false,
        with_row_last_updated_at_version: true,
        with_row_created_at_version: false,
        deletion_vector: None,
        row_id_sequence: None,
        last_updated_at_sequence: Some(sequence),
        created_at_sequence: None,
        make_deletions_null: false,
        total_num_rows: total_rows as u32,
    }
}

fn make_tasks(batch: RecordBatch, total_rows: u64, batch_size: u64) -> ReadBatchTaskStream {
    let tasks = (0..total_rows)
        .step_by(batch_size as usize)
        .map(move |offset| {
            let num_rows = batch_size.min(total_rows - offset) as usize;
            let batch = if num_rows == batch.num_rows() {
                batch.clone()
            } else {
                batch.slice(0, num_rows)
            };
            ReadBatchTask {
                task: future::ready(Ok(batch)).boxed(),
                num_rows: num_rows as u32,
            }
        });
    stream::iter(tasks).boxed()
}

fn bench_apply_versions(c: &mut Criterion) {
    let total_rows = std::env::var("BENCH_VERSION_ROWS")
        .map(|value| value.parse().unwrap())
        .unwrap_or(100_000_u64);
    let batch_size = std::env::var("BENCH_VERSION_BATCH_SIZE")
        .map(|value| value.parse().unwrap())
        .unwrap_or(1_024_u64)
        .min(total_rows);
    let batch = make_batch(batch_size);

    let mut group = c.benchmark_group("apply_version_columns");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for run_length in [total_rows, 1_024, 32, 1] {
        let sequence = Arc::new(make_version_sequence(total_rows, run_length));
        let run_count = sequence.runs.len();
        let config = make_config(total_rows, sequence);
        group.bench_with_input(BenchmarkId::new("runs", run_count), &run_count, |b, _| {
            let mut batch_offset = 0_u32;
            b.iter(|| {
                let result =
                    apply_row_id_and_deletes(batch.clone(), batch_offset, 0, &config).unwrap();
                black_box(result);
                batch_offset += batch_size as u32;
                if u64::from(batch_offset) + batch_size > total_rows {
                    batch_offset = 0;
                }
            });
        });
    }
    group.finish();
}

fn bench_stream_versions(c: &mut Criterion) {
    let total_rows = std::env::var("BENCH_VERSION_ROWS")
        .map(|value| value.parse().unwrap())
        .unwrap_or(100_000_u64);
    let batch_size = std::env::var("BENCH_VERSION_BATCH_SIZE")
        .map(|value| value.parse().unwrap())
        .unwrap_or(1_024_u64)
        .min(total_rows);
    let batch = make_batch(batch_size);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("stream_version_columns");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for run_length in [total_rows, 1_024, 32, 1] {
        let sequence = Arc::new(make_version_sequence(total_rows, run_length));
        let run_count = sequence.runs.len();
        group.bench_with_input(BenchmarkId::new("runs", run_count), &run_count, |b, _| {
            b.iter_batched(
                || {
                    (
                        make_tasks(batch.clone(), total_rows, batch_size),
                        make_config(total_rows, sequence.clone()),
                    )
                },
                |(tasks, config)| {
                    let batches = runtime
                        .block_on(
                            wrap_with_row_id_and_delete(tasks, 0, config)
                                .buffered(8)
                                .try_collect::<Vec<_>>(),
                        )
                        .unwrap();
                    black_box(batches);
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_apply_versions, bench_stream_versions);
criterion_main!(benches);
