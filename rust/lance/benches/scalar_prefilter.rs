// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Run with `cargo bench -p lance --profile release-with-debug --bench scalar_prefilter`.
//! `LANCE_PREFILTER_BENCH_METRICS` optionally names a JSONL file for untimed I/O and mask-size diagnostics.

use arrow_array::{Array, cast::AsArray, types::UInt64Type};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::{
    execution::TaskContext,
    physical_plan::{ExecutionPlan, execute_stream},
};
use futures::TryStreamExt;
use lance::{
    Dataset,
    index::DatasetIndexExt,
    io::exec::{PreFilterSource, fts::MatchQueryExec},
};
use lance_core::{ROW_ID, deepsize::DeepSizeOf};
use lance_index::{
    IndexType,
    scalar::{FullTextSearchQuery, ScalarIndexParams},
};
use lance_io::{object_store::ObjectStoreParams, utils::tracking_store::IOTracker};
use lance_select::{IndexExprResult, RowAddrMask, RowAddrTreeMap};
#[cfg(target_os = "linux")]
use lance_testing::pprof::{Output, PProfProfiler};
use std::{
    alloc::System,
    collections::HashMap,
    env,
    io::Write,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant},
};
use tracking_allocator::{AllocationGroupId, AllocationRegistry, AllocationTracker, Allocator};

mod fts_common;
use fts_common::{INDEXED_FRAGMENT_COUNT, ROWS_PER_FRAGMENT, build_segmented_fts_dataset};

#[global_allocator]
static ALLOCATOR: Allocator<System> = Allocator::system();

#[derive(Default)]
struct PrefilterAllocations {
    live: HashMap<usize, usize>,
    bytes: usize,
    peak: usize,
}

static ALLOCATIONS: LazyLock<Mutex<PrefilterAllocations>> = LazyLock::new(Mutex::default);

struct PrefilterAllocationTracker;

impl AllocationTracker for PrefilterAllocationTracker {
    fn allocated(&self, address: usize, _: usize, bytes: usize, _: AllocationGroupId) {
        let mut allocations = ALLOCATIONS.lock().unwrap();
        allocations.live.insert(address, bytes);
        allocations.bytes = allocations.bytes.checked_add(bytes).unwrap();
        allocations.peak = allocations.peak.max(allocations.bytes);
    }

    fn deallocated(
        &self,
        address: usize,
        _: usize,
        _: usize,
        _: AllocationGroupId,
        _: AllocationGroupId,
    ) {
        let mut allocations = ALLOCATIONS.lock().unwrap();
        // Allocations made before the diagnostic window are not part of its live set.
        if let Some(bytes) = allocations.live.remove(&address) {
            allocations.bytes -= bytes;
        }
    }
}

fn find_scalar_prefilter(plan: &Arc<dyn ExecutionPlan>) -> Option<(Arc<dyn ExecutionPlan>, bool)> {
    if let Some(scorer) = plan.downcast_ref::<MatchQueryExec>() {
        return match scorer.prefilter_source() {
            PreFilterSource::ScalarIndexQuery(source) => Some((source.clone(), true)),
            PreFilterSource::FilteredRowIds(source) => Some((source.clone(), false)),
            PreFilterSource::None => None,
        };
    }
    plan.children().into_iter().find_map(find_scalar_prefilter)
}

async fn prepare_scalar_prefilter(
    dataset: &Dataset,
    selected: &[usize],
    predicate: &str,
) -> (Arc<dyn ExecutionPlan>, bool) {
    let mut scanner = dataset.scan();
    scanner.with_fragments(
        selected
            .iter()
            .map(|id| dataset.fragments()[*id].clone())
            .collect(),
    );
    scanner.prefilter(true).filter(predicate).unwrap();
    scanner
        .full_text_search(
            FullTextSearchQuery::new("shared".to_string())
                .with_column("text".to_string())
                .unwrap(),
        )
        .unwrap();
    find_scalar_prefilter(&scanner.create_plan().await.unwrap()).unwrap()
}

async fn execute_scalar_prefilter(
    source: Arc<dyn ExecutionPlan>,
    is_mask: bool,
) -> (RowAddrMask, usize) {
    let mut stream = execute_stream(source, Arc::new(TaskContext::default())).unwrap();
    let mut selection = RowAddrTreeMap::new();
    let mut mask = RowAddrMask::allow_nothing();
    let mut materialized_rows = 0;
    let mut mask_batches = 0;
    while let Some(batch) = stream.try_next().await.unwrap() {
        if is_mask {
            let (result, _) = IndexExprResult::deserialize(&batch).unwrap();
            assert!(result.is_exact());
            mask = result.upper;
            mask_batches += 1;
        } else {
            let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>();
            materialized_rows += row_ids.len() - row_ids.null_count();
            selection.extend(row_ids.iter().flatten());
        }
    }
    if is_mask {
        assert_eq!(mask_batches, 1);
        (mask, 0)
    } else {
        (RowAddrMask::from_allowed(selection), materialized_rows)
    }
}

fn prefilter_count(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    let own = plan.metrics().map_or(0, |metrics| {
        metrics
            .iter()
            .filter(|metric| metric.value().name() == name)
            .map(|metric| metric.value().as_usize())
            .sum::<usize>()
    });
    own + plan
        .children()
        .into_iter()
        .map(|child| prefilter_count(child, name))
        .sum::<usize>()
}

fn bench_hybrid_scalar_prefilter(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let has_diagnostics = env::var_os("LANCE_PREFILTER_BENCH_METRICS").is_some();
    if has_diagnostics {
        LazyLock::force(&ALLOCATIONS);
        AllocationRegistry::set_global_tracker(PrefilterAllocationTracker).unwrap();
    }
    let mut group = c.benchmark_group("fts_scalar_prefilter");
    for covered in [INDEXED_FRAGMENT_COUNT, INDEXED_FRAGMENT_COUNT / 2] {
        let mut fixture = rt.block_on(build_segmented_fts_dataset(1));
        rt.block_on(async {
            let segment = fixture
                .dataset
                .create_index_builder(&["id"], IndexType::BTree, &ScalarIndexParams::default())
                .name("id_idx".to_string())
                .fragments((0..covered as u32).collect())
                .execute_uncommitted()
                .await
                .unwrap();
            fixture
                .dataset
                .commit_existing_index_segments("id_idx", "id", vec![segment])
                .await
                .unwrap();
        });
        for selected_count in [1, INDEXED_FRAGMENT_COUNT] {
            let selected = if selected_count == 1 {
                vec![INDEXED_FRAGMENT_COUNT / 2 - 1]
            } else {
                (0..INDEXED_FRAGMENT_COUNT).collect::<Vec<_>>()
            };
            for modulus in [2, 32] {
                // Center the range on the coverage boundary so both branches produce matches.
                let rows = INDEXED_FRAGMENT_COUNT * ROWS_PER_FRAGMENT;
                let start = rows / 2 - rows / modulus / 2;
                let predicate = format!("id >= {start} AND id < {}", start + rows / modulus);
                let expected = rt.block_on(async {
                    let mut oracle = fixture.dataset.scan();
                    oracle.with_fragments(
                        selected
                            .iter()
                            .map(|id| fixture.dataset.fragments()[*id].clone())
                            .collect(),
                    );
                    oracle
                        .use_scalar_index(false)
                        .with_row_id()
                        .filter(&predicate)
                        .unwrap();
                    oracle.project(&[ROW_ID]).unwrap();
                    let batch = oracle.try_into_batch().await.unwrap();
                    RowAddrMask::from_allowed(
                        batch[ROW_ID]
                            .as_primitive::<UInt64Type>()
                            .values()
                            .iter()
                            .copied()
                            .collect(),
                    )
                });
                for is_cold in [false, true] {
                    let tracker = Arc::new(IOTracker::default());
                    let uri = fixture.dataset.uri().to_owned();
                    let open = || {
                        lance::dataset::builder::DatasetBuilder::from_uri(&uri)
                            .with_store_params(ObjectStoreParams {
                                object_store_wrapper: Some(tracker.clone()),
                                ..Default::default()
                            })
                            .load()
                    };
                    let dataset = rt.block_on(open()).unwrap();
                    if !is_cold {
                        rt.block_on(async {
                            let (source, is_mask) =
                                prepare_scalar_prefilter(&dataset, &selected, &predicate).await;
                            execute_scalar_prefilter(source, is_mask).await;
                        });
                    }
                    let (source, is_mask) =
                        rt.block_on(prepare_scalar_prefilter(&dataset, &selected, &predicate));
                    tracker.incremental_stats();
                    if has_diagnostics {
                        *ALLOCATIONS.lock().unwrap() = PrefilterAllocations::default();
                        AllocationRegistry::enable_tracking();
                    }
                    let (actual, raw_rows) =
                        rt.block_on(execute_scalar_prefilter(source.clone(), is_mask));
                    AllocationRegistry::disable_tracking();
                    let peak_bytes = ALLOCATIONS.lock().unwrap().peak;
                    assert_eq!(actual, expected);
                    let stats = tracker.incremental_stats();
                    let mut covered_reads = 0;
                    let mut missing_reads = 0;
                    for fragment in dataset.fragments().iter() {
                        let reads = stats
                            .requests
                            .iter()
                            .filter(|request| {
                                fragment
                                    .files
                                    .iter()
                                    .any(|file| request.path.as_ref().ends_with(&file.path))
                            })
                            .count();
                        if fragment.id < covered as u64 {
                            covered_reads += reads;
                        } else {
                            missing_reads += reads;
                        }
                    }
                    if is_mask {
                        assert_eq!(covered_reads, 0);
                    }
                    if let Some(path) = env::var_os("LANCE_PREFILTER_BENCH_METRICS") {
                        // Measure individual executions separately from Criterion's
                        // sample averages, with allocation tracking disabled.
                        let mut latencies = rt.block_on(async {
                            let mut latencies = Vec::with_capacity(100);
                            for _ in 0..100 {
                                let iteration_dataset = if is_cold {
                                    open().await.unwrap()
                                } else {
                                    dataset.clone()
                                };
                                let (source, is_mask) = prepare_scalar_prefilter(
                                    &iteration_dataset,
                                    &selected,
                                    &predicate,
                                )
                                .await;
                                tracker.incremental_stats();
                                let start = Instant::now();
                                std::hint::black_box(
                                    execute_scalar_prefilter(source, is_mask).await,
                                );
                                latencies.push(start.elapsed().as_nanos());
                            }
                            latencies
                        });
                        latencies.sort_unstable();
                        let mut file = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(path)
                            .unwrap();
                        writeln!(file, "{}", serde_json::json!({
                            "covered": covered, "selected": selected_count, "modulus": modulus,
                            "cold": is_cold, "source": source.name(),
                            "covered_data_reads": covered_reads, "missing_data_reads": missing_reads,
                            "read_bytes": stats.read_bytes, "scalar_segments_opened": prefilter_count(&source, "indices_loaded"),
                            "row_ids_materialized": if is_mask { prefilter_count(&source, "prefilter_input_rows") } else { raw_rows },
                            "selection_bytes": actual.deep_size_of(),
                            "peak_prefilter_allocated_bytes": peak_bytes,
                            "latency_samples": latencies.len(),
                            "latency_p50_ns": latencies[49], "latency_p95_ns": latencies[94],
                        })).unwrap();
                    }
                    group.bench_function(
                        format!(
                            "covered={covered}/selected={selected_count}/match=1-{modulus}/cold={is_cold}"
                        ),
                        |b| {
                            b.iter_custom(|iterations| rt.block_on(async {
                                let mut elapsed = Duration::ZERO;
                                for _ in 0..iterations {
                                    let iteration_dataset = if is_cold { open().await.unwrap() }
                                        else { dataset.clone() };
                                    let (source, is_mask) = prepare_scalar_prefilter(
                                        &iteration_dataset, &selected, &predicate,
                                    ).await;
                                    tracker.incremental_stats();
                                    let start = Instant::now();
                                    std::hint::black_box(execute_scalar_prefilter(source, is_mask).await);
                                    elapsed += start.elapsed();
                                }
                                elapsed
                            }));
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench_hybrid_scalar_prefilter
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_hybrid_scalar_prefilter
);

criterion_main!(benches);
