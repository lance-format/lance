// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compare the existing in-memory reverse index with an ordinary `_rowid` BTree.
//!
//! Run with `cargo bench -p lance --profile release-with-debug --bench row_id_lookup`.
//! `ROW_ID_LOOKUP_ROWS` sets the row count (default 100,000, at least 16).
//! `ROW_ID_LOOKUP_COVERAGE` selects full (default), none, tail (10% inline tail),
//! scan90/scan50 (90%/50% indexed with a spilled, scanned remainder), compacted
//! (index then compact), optimized (index, compact, optimize), or deleted.
//! Fragmented scenarios require a row count divisible by 20.
//! `ROW_ID_LOOKUP_LAYOUT` selects all (default), append, or shuffled.
//! `ROW_ID_LOOKUP_KEY_BATCHES` rotates this many deterministic batches (default 16).
//! Append-only and globally shuffled tables use the same keys and local store.
//! Cold means a fresh Lance session (OS caches are not flushed); dataset open is
//! outside the timed region. Warm means the reverse index / BTree pages have
//! already been read. Each result is checked against the other implementation.
//! The report also includes BTree bytes/row and the reverse index's retained
//! deep size; these are different resources, not a memory savings comparison.
//!
//! `ROW_ID_LOOKUP_CAPTURE_ROWS` instead benchmarks preparing a predicate delete
//! of this many rows from a real dataset twice that size. Input is streamed in
//! fixed batches. Timing includes scanning, capture resolution and deletion-file
//! writing, but excludes dataset creation and transaction publication. Run each
//! revision in a fresh process to compare whole-process peak RSS externally.

#![allow(clippy::print_stdout)]

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{RecordBatchIterator, UInt64Array, record_batch};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::optimize::{CompactionOptions, compact_files};
use lance::dataset::rowids::{
    INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, SPILL_ROW_LINEAGE_CONFIG_KEY, get_row_id_index,
    row_ids_to_row_addrs,
};
use lance::dataset::{CommitBuilder, DeleteBuilder, InsertBuilder, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance_core::ROW_ID;
use lance_core::deepsize::DeepSizeOf;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::{IndexType, scalar::ScalarIndexParams};
use lance_select::RowSetOps;
use lance_table::format::{ROW_ID_FIELD_ID, RowIdMeta};
use lance_table::rowids::{RowIdSequence, write_row_ids};
use lance_table::transaction::{Operation, RewriteGroup, TransactionBuilder};
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::SliceRandom};
use roaring::RoaringBitmap;
use tokio::runtime::Runtime;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Full,
    None,
    InlineTail,
    Scan90,
    Scan50,
    Compacted,
    Optimized,
    Deleted,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::None => "none",
            Self::InlineTail => "tail",
            Self::Scan90 => "scan90",
            Self::Scan50 => "scan50",
            Self::Compacted => "compacted",
            Self::Optimized => "optimized",
            Self::Deleted => "deleted",
        }
    }
}

async fn build_table(uri: &str, total_rows: usize, shuffled: bool, scenario: Scenario) -> Dataset {
    let rows = if scenario == Scenario::InlineTail {
        total_rows * 9 / 10
    } else {
        total_rows
    };
    let fragmented = matches!(
        scenario,
        Scenario::Scan90 | Scenario::Scan50 | Scenario::Compacted | Scenario::Optimized
    );
    if fragmented {
        assert!(
            total_rows >= 20 && total_rows.is_multiple_of(20),
            "fragmented scenarios need at least 20 rows and a multiple of 20, got {total_rows}"
        );
    }
    let batch = record_batch!(("key", UInt64, (0..rows as u64).collect::<Vec<_>>())).unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let params = WriteParams {
        enable_stable_row_ids: true,
        max_rows_per_file: rows.div_ceil(if fragmented { 20 } else { 8 }),
        ..Default::default()
    };
    let mut dataset = Dataset::write(reader, uri, Some(params.clone()))
        .await
        .unwrap();
    if shuffled {
        let mut ids = (0..rows as u64).collect::<Vec<_>>();
        ids.shuffle(&mut SmallRng::seed_from_u64(42));
        let batch = record_batch!(("key", UInt64, ids.clone())).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let dataset_ref = Arc::new(dataset);
        let staged = InsertBuilder::new(dataset_ref.clone())
            .with_params(&WriteParams {
                mode: WriteMode::Append,
                ..params
            })
            .execute_uncommitted_stream(reader)
            .await
            .unwrap();
        let Operation::Append { mut fragments } = staged.operation else {
            panic!("expected an uncommitted append");
        };
        let mut offset = 0;
        for fragment in &mut fragments {
            let len = fragment.physical_rows.unwrap();
            fragment.row_id_meta = Some(RowIdMeta::Inline(
                write_row_ids(&RowIdSequence::from(&ids[offset..offset + len])).into(),
            ));
            offset += len;
        }
        let transaction = TransactionBuilder::new(
            dataset_ref.version().version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: dataset_ref.fragments().as_ref().clone(),
                    new_fragments: fragments,
                }],
                rewritten_indices: vec![],
                frag_reuse_index: None,
            },
        )
        .build();
        dataset = CommitBuilder::new(dataset_ref)
            .execute(transaction)
            .await
            .unwrap();
    }
    if shuffled || fragmented {
        dataset
            .update_config([
                (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: if fragmented { rows / 10 } else { rows },
                defer_index_remap: false,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(
            dataset
                .fragments()
                .iter()
                .all(|fragment| fragment.row_id_meta == Some(RowIdMeta::Column)),
            "the scan scenarios must read spilled row IDs, not compact inline ranges"
        );
    }
    if fragmented {
        assert_eq!(dataset.fragments().len(), 10);
    }
    if scenario != Scenario::None {
        let fragments = if matches!(scenario, Scenario::Scan90 | Scenario::Scan50) {
            let indexed_fragments = if scenario == Scenario::Scan90 { 9 } else { 5 };
            Some(
                dataset.fragments()[..indexed_fragments]
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect(),
            )
        } else {
            None
        };
        let params = ScalarIndexParams::default();
        let mut builder = dataset.create_index_builder(&[ROW_ID], IndexType::BTree, &params);
        if let Some(fragments) = fragments {
            builder = builder.fragments(fragments);
        }
        builder.await.unwrap();
    }
    if matches!(scenario, Scenario::Compacted | Scenario::Optimized) {
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: total_rows,
                defer_index_remap: false,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 1);
        if scenario == Scenario::Optimized {
            dataset.optimize_indices(&Default::default()).await.unwrap();
        }
    }
    if scenario == Scenario::Deleted {
        dataset.delete("key % 10 = 0").await.unwrap();
    }
    if rows < total_rows {
        dataset
            .update_config([(SPILL_ROW_LINEAGE_CONFIG_KEY, "false")])
            .await
            .unwrap();
        let batch = record_batch!((
            "key",
            UInt64,
            (rows as u64..total_rows as u64).collect::<Vec<_>>()
        ))
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        dataset
            .append(
                reader,
                Some(WriteParams {
                    max_rows_per_file: total_rows - rows,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
    }
    let indices = dataset.load_indices().await.unwrap();
    let covered: RoaringBitmap = indices
        .iter()
        .filter(|index| index.keyed_field() == Some(ROW_ID_FIELD_ID))
        .filter_map(|index| index.fragment_bitmap.as_ref())
        .flat_map(|fragments| fragments.iter())
        .collect();
    let indexed_rows = dataset
        .fragments()
        .iter()
        .filter(|fragment| covered.contains(fragment.id as u32))
        .map(|fragment| fragment.physical_rows.unwrap())
        .sum::<usize>();
    let expected_indexed_rows = match scenario {
        Scenario::None => 0,
        Scenario::InlineTail | Scenario::Scan90 => total_rows * 9 / 10,
        Scenario::Scan50 => total_rows / 2,
        _ => total_rows,
    };
    assert_eq!(indexed_rows, expected_indexed_rows, "{scenario:?}");
    let bytes = indices
        .iter()
        .filter(|index| index.keyed_field() == Some(ROW_ID_FIELD_ID))
        .filter_map(|index| index.files.as_ref())
        .flatten()
        .map(|file| file.size_bytes)
        .sum::<u64>();
    let reverse = get_row_id_index(&dataset).await.unwrap().unwrap();
    println!(
        "layout={} coverage={} rows={total_rows} indexed_rows={indexed_rows} fragments={} btree_bytes={bytes} btree_bytes_per_row={:.3} reverse_index_deep_size_bytes={}",
        if shuffled { "shuffled" } else { "append" },
        scenario.name(),
        dataset.fragments().len(),
        bytes as f64 / total_rows as f64,
        reverse.deep_size_of()
    );
    dataset
}

fn bench_capture(criterion: &mut Criterion, rows: u64) {
    assert!(
        rows > 0 && rows <= u32::MAX as u64 / 2,
        "capture rows must be in 1..=u32::MAX/2, got {rows}"
    );
    let runtime = Runtime::new().unwrap();
    let directory = TempStrDir::default();
    let total_rows = rows * 2;
    let schema = record_batch!(("i", UInt64, [0])).unwrap().schema();
    let batches = (0..total_rows).step_by(8192).map(move |start| {
        record_batch!((
            "i",
            UInt64,
            (start..(start + 8192).min(total_rows)).collect::<Vec<_>>()
        ))
    });
    let input = RecordBatchIterator::new(batches, schema);
    let dataset = Arc::new(
        runtime
            .block_on(Dataset::write(
                input,
                directory.as_ref(),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: total_rows as usize,
                    max_rows_per_group: 8192,
                    ..Default::default()
                }),
            ))
            .unwrap(),
    );
    assert_eq!(dataset.fragments().len(), 1);
    let deletion = runtime
        .block_on(DeleteBuilder::new(dataset.clone(), "i % 2 = 0").execute_uncommitted())
        .unwrap();
    assert_eq!(deletion.num_deleted_rows, rows);
    let affected = deletion.affected_rows.as_ref().unwrap();
    assert_eq!(affected.len(), Some(rows));
    assert!(affected.contains(0));
    assert!(affected.contains(total_rows - 2));
    assert!(!affected.contains(1));
    drop(deletion);
    let mut group = criterion.benchmark_group("row_id_capture/delete");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.bench_function(BenchmarkId::from_parameter(rows), |bench| {
        bench.iter(|| {
            std::hint::black_box(
                runtime
                    .block_on(
                        DeleteBuilder::new(dataset.clone(), "i % 2 = 0").execute_uncommitted(),
                    )
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn bench_row_id_lookup(criterion: &mut Criterion) {
    if let Ok(rows) = std::env::var("ROW_ID_LOOKUP_CAPTURE_ROWS") {
        bench_capture(
            criterion,
            rows.parse()
                .expect("ROW_ID_LOOKUP_CAPTURE_ROWS must be an integer"),
        );
        return;
    }
    let rows = std::env::var("ROW_ID_LOOKUP_ROWS").map_or(100_000, |value| {
        value
            .parse::<usize>()
            .expect("ROW_ID_LOOKUP_ROWS must be an integer")
    });
    assert!(rows >= 16, "ROW_ID_LOOKUP_ROWS must be at least 16");
    let scenario = match std::env::var("ROW_ID_LOOKUP_COVERAGE")
        .as_deref()
        .unwrap_or("full")
    {
        "full" => Scenario::Full,
        "none" => Scenario::None,
        "tail" => Scenario::InlineTail,
        "scan90" => Scenario::Scan90,
        "scan50" => Scenario::Scan50,
        "compacted" => Scenario::Compacted,
        "optimized" => Scenario::Optimized,
        "deleted" => Scenario::Deleted,
        other => panic!("invalid ROW_ID_LOOKUP_COVERAGE: {other}"),
    };
    let layouts: &[bool] = match std::env::var("ROW_ID_LOOKUP_LAYOUT")
        .as_deref()
        .unwrap_or("all")
    {
        "all" => &[false, true],
        "append" => &[false],
        "shuffled" => &[true],
        other => panic!("invalid ROW_ID_LOOKUP_LAYOUT: {other}"),
    };
    let batches = std::env::var("ROW_ID_LOOKUP_KEY_BATCHES")
        .map_or(16, |value| value.parse::<usize>().unwrap());
    assert!(batches > 0, "ROW_ID_LOOKUP_KEY_BATCHES must be positive");
    let runtime = Runtime::new().unwrap();
    for &shuffled in layouts {
        let directory = TempStrDir::default();
        let dataset = runtime.block_on(build_table(&directory, rows, shuffled, scenario));
        let layout = if shuffled { "shuffled" } else { "append" };
        let mut group =
            criterion.benchmark_group(format!("row_id_lookup/{layout}/{}", scenario.name()));
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(1));
        group.measurement_time(Duration::from_secs(2));
        for key_count in [1, 1000] {
            let mut random = SmallRng::seed_from_u64(7);
            let ids = (0..batches)
                .map(|_| {
                    (0..key_count)
                        .map(|_| random.random_range(0..rows as u64))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let keys = ids
                .iter()
                .map(|ids| UInt64Array::from(ids.clone()))
                .collect::<Vec<_>>();
            let reverse = runtime
                .block_on(get_row_id_index(&dataset))
                .unwrap()
                .unwrap();
            for (ids, keys) in ids.iter().zip(&keys) {
                assert_eq!(
                    runtime
                        .block_on(row_ids_to_row_addrs(&dataset, keys))
                        .unwrap(),
                    reverse.get_many(ids).unwrap()
                );
            }
            group.bench_function(BenchmarkId::new("reverse_warm", key_count), |bench| {
                let mut batch = 0;
                bench.iter(|| {
                    let ids = &ids[batch % batches];
                    batch += 1;
                    std::hint::black_box(runtime.block_on(async {
                        get_row_id_index(&dataset)
                            .await
                            .unwrap()
                            .unwrap()
                            .get_many(ids)
                            .unwrap()
                    }))
                });
            });
            group.bench_function(BenchmarkId::new("lookup_warm", key_count), |bench| {
                let mut batch = 0;
                bench.iter(|| {
                    let keys = &keys[batch % batches];
                    batch += 1;
                    std::hint::black_box(
                        runtime
                            .block_on(row_ids_to_row_addrs(&dataset, keys))
                            .unwrap(),
                    )
                });
            });
            for btree in [false, true] {
                group.bench_function(
                    BenchmarkId::new(
                        if btree { "lookup_cold" } else { "reverse_cold" },
                        key_count,
                    ),
                    |bench| {
                        let mut batch = 0;
                        bench.iter_batched(
                            || {
                                let dataset = runtime
                                    .block_on(DatasetBuilder::from_uri(&directory).load())
                                    .unwrap();
                                let key_batch = batch % batches;
                                batch += 1;
                                (dataset, key_batch)
                            },
                            |(cold, key_batch)| {
                                runtime.block_on(async {
                                    let actual = if btree {
                                        row_ids_to_row_addrs(&cold, &keys[key_batch]).await.unwrap()
                                    } else {
                                        get_row_id_index(&cold)
                                            .await
                                            .unwrap()
                                            .unwrap()
                                            .get_many(&ids[key_batch])
                                            .unwrap()
                                    };
                                    std::hint::black_box(actual)
                                })
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches, bench_row_id_lookup);
criterion_main!(benches);
