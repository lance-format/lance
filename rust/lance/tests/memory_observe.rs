// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `LANCE_MEMORY_OBSERVE` reports the memory of index builds, compaction,
//! `optimize_indices`, MemWAL flushes, writes, `merge_insert`, updates and
//! deletes as trace events and, with the `metrics` feature, through the
//! `metrics` facade.
//!
//! This file must stay a single-test integration binary: Lance reads the
//! switch once per process, and the tracing subscriber and the metrics
//! recorder are process-global, so a sibling test could read the switch
//! before the `set_var` below or report into the same subscriber or recorder.
#![cfg(target_os = "linux")]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow_array::types::{Float32Type, Int64Type};
use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{DataType, Field, Schema};
use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::fragment::FileFragment;
use lance::dataset::index::DatasetIndexRemapperOptions;
use lance::dataset::mem_wal::scanner::SsTableWarmer;
use lance::dataset::mem_wal::{DatasetMemWalExt, ShardWriterConfig};
use lance::dataset::optimize::{
    CompactionOptions, commit_compaction, compact_files, plan_compaction,
};
use lance::dataset::{
    DeleteBuilder, InsertBuilder, MergeInsertBuilder, MergeInsertJob, NewColumnTransform,
    UpdateBuilder, WhenMatched, WhenNotMatched, WriteMode, WriteParams,
};
use lance::index::optimize::{commit_optimize_indices, plan_optimize_indices};
use lance::index::vector::VectorIndexParams;
use lance::index::{CreateIndexBuilder, DatasetIndexExt};
#[cfg(feature = "metrics")]
use lance::memory::{
    METRIC_OPERATION_PEAK, METRIC_RSS, METRIC_RSS_PEAK, METRIC_STAGE_PEAK, refresh_metrics,
};
use lance_core::utils::testing::{ProxyObjectStore, ProxyObjectStorePolicy};
use lance_core::utils::tracing::TRACE_MEMORY;
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_io::object_store::{ObjectStoreParams, WrappingObjectStore};
use lance_linalg::distance::MetricType;
#[cfg(feature = "metrics")]
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use tracing_subscriber::Layer;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::layer::SubscriberExt;
use uuid::Uuid;

#[cfg(feature = "metrics")]
type Recorded = Vec<(String, Vec<(String, String)>, DebugValue)>;

/// The labels of an operation: `index_type` only for operations that build
/// one index.
#[cfg(feature = "metrics")]
fn labels<'a>(
    operation: &'a str,
    index_type: Option<&'a str>,
    stage: Option<&'a str>,
) -> Vec<(&'static str, &'a str)> {
    let mut labels = vec![("operation", operation)];
    labels.extend(index_type.map(|index_type| ("index_type", index_type)));
    labels.extend(stage.map(|stage| ("stage", stage)));
    labels
}

#[cfg(feature = "metrics")]
fn histogram(recorded: &Recorded, name: &str, labels: &[(&str, &str)]) -> Vec<f64> {
    recorded
        .iter()
        .filter(|(metric, metric_labels, _)| {
            metric == name
                && metric_labels.len() == labels.len()
                && labels
                    .iter()
                    .all(|(k, v)| metric_labels.iter().any(|(mk, mv)| mk == k && mv == v))
        })
        .flat_map(|(_, _, value)| match value {
            DebugValue::Histogram(values) => values.iter().map(|value| value.0).collect(),
            _ => Vec::new(),
        })
        .collect()
}

#[cfg(feature = "metrics")]
fn gauge(recorded: &Recorded, name: &str) -> Option<f64> {
    recorded.iter().find_map(|(metric, _, value)| match value {
        DebugValue::Gauge(value) if metric == name => Some(value.0),
        _ => None,
    })
}

/// Stages the workload below reports, as `(operation, index_type, stage)`:
/// each completes at least once and as often as it starts. The stages of the
/// index builders that maintenance runs report into the maintenance
/// operation, which builds no single index; together with
/// `MAINTENANCE_STAGES` they are every stage a maintenance operation reports.
const STAGES: &[(&str, Option<&str>, &str)] = &[
    ("create_index", Some("ivf_flat"), "train_ivf"),
    ("create_index", Some("ivf_flat"), "train_quantizer"),
    ("create_index", Some("ivf_flat"), "shuffle"),
    ("create_index", Some("ivf_flat"), "merge_partitions"),
    ("create_index", Some("btree"), "load_data"),
    ("create_index", Some("bitmap"), "load_data"),
    ("create_index", Some("fm"), "load_data"),
    ("create_index", Some("inverted"), "tokenize_docs"),
    ("create_index", Some("inverted"), "copy_partitions"),
    ("create_index", Some("inverted"), "write_metadata"),
    (
        "merge_index_metadata",
        Some("inverted"),
        "read_partition_metadata",
    ),
    (
        "merge_index_metadata",
        Some("inverted"),
        "remap_partition_files",
    ),
    (
        "merge_index_metadata",
        Some("inverted"),
        "write_merged_metadata",
    ),
    ("optimize_indices", None, "load_data"),
    ("optimize_indices", None, "tokenize_docs"),
    ("optimize_indices", None, "copy_partitions"),
    ("optimize_indices", None, "write_metadata"),
    ("optimize_indices", None, "train_ivf"),
    ("optimize_indices", None, "train_quantizer"),
    ("optimize_indices", None, "shuffle"),
    ("optimize_indices", None, "merge_partitions"),
    ("optimize_indices", None, "sort_values"),
    ("optimize_indices", None, "write_btree_index"),
    ("mem_wal_flush", None, "load_data"),
    ("mem_wal_flush", None, "sort_values"),
    ("mem_wal_flush", None, "write_btree_index"),
];

/// Observed operations below, as `(operation, index_type, started,
/// completed)`. The inverted index is built once whole and once as one
/// segment per fragment, which the merge then joins. The FM-Index builds one
/// segment per fragment and commits them together. One BTREE and one
/// FM-Index build, one compaction, one `optimize_indices`, the MemWAL flush
/// of a fenced writer, the first attempt of a retried MemWAL flush, and one
/// write, `merge_insert`, update and delete fail, so they start and never
/// complete. The untrained BTREE adds nothing, and neither do the builds
/// inside `optimize_indices` and the MemWAL flushes, the writes of the
/// generations that MemWAL flushes commit, nor the commits of
/// `commit_compaction`'s callers. Every dataset the workload creates or
/// appends to is a write of its own.
const OPERATIONS: &[(&str, Option<&str>, usize, usize)] = &[
    ("create_index", Some("ivf_flat"), 1, 1),
    ("create_index", Some("btree"), 5, 4),
    ("create_index", Some("bitmap"), 1, 1),
    ("create_index", Some("inverted"), 4, 4),
    ("merge_index_metadata", Some("inverted"), 1, 1),
    ("create_index", Some("fm"), 2, 1),
    ("compaction", None, 6, 5),
    ("optimize_indices", None, 12, 11),
    ("mem_wal_flush", None, 9, 7),
    ("write", None, 21, 19),
    ("merge_insert", None, 4, 3),
    ("update", None, 2, 1),
    ("delete", None, 3, 2),
];

/// The stages of the maintenance operations below, as `(operation, stage,
/// started, completed)`, counted over every operation of that kind. A
/// compaction rewrites its files in one stage however many tasks it runs, and
/// an `optimize_indices` with nothing to do still runs its stages around an
/// empty plan. A warm that fails starts its stage and never completes it,
/// while its flush still commits. A flush attempt that fails to write its
/// generation starts `write_fragments` and never completes it.
const MAINTENANCE_STAGES: &[(&str, &str, usize, usize)] = &[
    ("compaction", "rewrite_files", 4, 4),
    ("compaction", "reserve_fragments", 4, 4),
    ("compaction", "remap_indices", 4, 4),
    ("compaction", "commit", 4, 3),
    ("optimize_indices", "plan", 7, 7),
    ("optimize_indices", "build_segments", 8, 8),
    ("optimize_indices", "merge_segments", 7, 7),
    ("optimize_indices", "commit", 5, 4),
    ("mem_wal_flush", "write_fragments", 8, 7),
    ("mem_wal_flush", "build_indices", 7, 7),
    ("mem_wal_flush", "finalize", 7, 7),
    ("mem_wal_flush", "warm", 2, 1),
    ("mem_wal_flush", "commit", 7, 7),
];

/// The stages of the writes below, as `(operation, stage, started,
/// completed)`, counted over every operation of that kind. A write that
/// commits nothing has no `commit` stage, while `merge_insert`, update and
/// delete run their stages once per attempt, and a commit that fails never
/// completes its stage.
const WRITE_STAGES: &[(&str, &str, usize, usize)] = &[
    ("write", "write_fragments", 21, 21),
    ("write", "commit", 17, 15),
    ("merge_insert", "execute", 5, 5),
    ("merge_insert", "commit", 4, 2),
    ("update", "execute", 2, 2),
    ("update", "commit", 2, 1),
    ("delete", "execute", 3, 3),
    ("delete", "commit", 2, 1),
];

#[test]
fn test_observed_operations_report_memory_peaks() {
    // SAFETY: this is the only test in the binary and no other thread runs
    // yet: the runtime that runs Lance is built below.
    unsafe { std::env::set_var("LANCE_MEMORY_OBSERVE", "1") };
    let events = MemoryEvents::default();
    let memory_only = Targets::new().with_target(TRACE_MEMORY, tracing::Level::INFO);
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(events.clone().with_filter(memory_only)),
    )
    .unwrap();
    #[cfg(feature = "metrics")]
    let snapshotter = {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        recorder.install().unwrap();
        snapshotter
    };
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            // A retried flush waits before its second attempt, so it runs
            // alongside the rest of the workload.
            let retried = tokio::spawn(run_retried_flush());
            build_indices().await;
            run_nested_scalar_builds(&events).await;
            Box::pin(run_maintenance()).await;
            Box::pin(run_writes()).await;
            retried.await.unwrap();

            check_events(&events.0.lock().unwrap());
            #[cfg(feature = "metrics")]
            check_metrics(&snapshotter);

            for mixed_plan in [false, true] {
                for conflicting_repack in [false, true] {
                    Box::pin(check_repack_events(&events, mixed_plan, conflicting_repack)).await;
                }
            }
        });
}

#[cfg(feature = "metrics")]
fn check_metrics(snapshotter: &Snapshotter) {
    refresh_metrics();

    let recorded: Recorded = snapshotter
        .snapshot()
        .into_vec()
        .into_iter()
        .map(|(key, _unit, _description, value)| {
            let key = key.key();
            let labels = key
                .labels()
                .map(|label| (label.key().to_string(), label.value().to_string()))
                .collect();
            (key.name().to_string(), labels, value)
        })
        .collect();

    for &(operation, index_type, stage) in STAGES {
        let peaks = histogram(
            &recorded,
            METRIC_STAGE_PEAK,
            &labels(operation, index_type, Some(stage)),
        );
        assert!(
            !peaks.is_empty() && peaks.iter().all(|peak| *peak > 0.0),
            "{operation} {index_type:?} {stage}: {peaks:?} in {recorded:?}"
        );
    }
    for &(operation, stage, _, completed) in MAINTENANCE_STAGES.iter().chain(WRITE_STAGES) {
        let peaks = histogram(
            &recorded,
            METRIC_STAGE_PEAK,
            &labels(operation, None, Some(stage)),
        );
        assert_eq!(peaks.len(), completed, "{operation} {stage}: {recorded:?}");
    }
    for &(operation, index_type, _, completed) in OPERATIONS {
        let peaks = histogram(
            &recorded,
            METRIC_OPERATION_PEAK,
            &labels(operation, index_type, None),
        );
        assert_eq!(
            peaks.len(),
            completed,
            "{operation} {index_type:?}: {recorded:?}"
        );
        assert!(
            peaks.iter().all(|peak| *peak > 0.0),
            "{operation} {index_type:?}: {peaks:?}"
        );
    }

    let rss = gauge(&recorded, METRIC_RSS).expect("RSS gauge is set");
    let peak_rss = gauge(&recorded, METRIC_RSS_PEAK).expect("peak RSS gauge is set");
    assert!(rss > 0.0 && peak_rss >= rss, "rss {rss}, peak {peak_rss}");
}

/// The fields of every memory event, with strings unquoted.
#[derive(Clone, Default)]
struct MemoryEvents(Arc<Mutex<Vec<BTreeMap<String, String>>>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for MemoryEvents {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        if event.metadata().target() == TRACE_MEMORY {
            let mut fields = EventFields::default();
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }
    }
}

#[derive(Default)]
struct EventFields(BTreeMap<String, String>);

impl tracing::field::Visit for EventFields {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().to_string(), value.to_string());
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0
            .insert(field.name().to_string(), format!("{value:?}"));
    }
}

fn check_events(events: &[BTreeMap<String, String>]) {
    let count = |event: &str, operation: &str, index_type: Option<&str>, stage: Option<&str>| {
        events
            .iter()
            .filter(|fields| {
                fields.get("event").is_some_and(|value| value == event)
                    && fields
                        .get("operation")
                        .is_some_and(|value| value == operation)
                    && fields.get("index_type").map(String::as_str) == index_type
                    && (stage.is_none() || fields.get("stage").map(String::as_str) == stage)
            })
            .count()
    };
    for &(operation, index_type, stage) in STAGES {
        let completed = count("stage_complete", operation, index_type, Some(stage));
        assert!(
            completed > 0 && count("stage_start", operation, index_type, Some(stage)) == completed,
            "{operation} {index_type:?} {stage}: {events:?}"
        );
    }
    for &(operation, stage, started, completed) in MAINTENANCE_STAGES.iter().chain(WRITE_STAGES) {
        for (event, stages) in [("stage_start", started), ("stage_complete", completed)] {
            assert_eq!(
                count(event, operation, None, Some(stage)),
                stages,
                "{event} {operation} {stage}: {events:?}"
            );
        }
    }
    for &(operation, index_type, started, completed) in OPERATIONS {
        for (event, operations) in [
            ("operation_start", started),
            ("operation_complete", completed),
        ] {
            assert_eq!(
                count(event, operation, index_type, None),
                operations,
                "{event} {operation} {index_type:?}: {events:?}"
            );
        }
    }
    // Nothing else reports: no operation the lists leave out, and no stage of
    // a maintenance operation or a write they leave out.
    let started: usize = OPERATIONS.iter().map(|&(_, _, started, _)| started).sum();
    assert_eq!(
        events
            .iter()
            .filter(|fields| fields["event"] == "operation_start")
            .count(),
        started,
        "{events:?}"
    );
    for fields in events
        .iter()
        .filter(|fields| fields["event"] == "stage_start" && !fields.contains_key("index_type"))
    {
        let (operation, stage) = (fields["operation"].as_str(), fields["stage"].as_str());
        assert!(
            MAINTENANCE_STAGES
                .iter()
                .chain(WRITE_STAGES)
                .any(|&(listed, listed_stage, _, _)| (listed, listed_stage) == (operation, stage))
                || STAGES.iter().any(|&(listed, index_type, listed_stage)| {
                    (listed, index_type, listed_stage) == (operation, None, stage)
                }),
            "an unlisted stage: {fields:?}"
        );
    }
    for fields in events {
        let positive = |name: &str| {
            fields
                .get(name)
                .and_then(|value| value.parse::<u64>().ok())
                .is_some_and(|value| value > 0)
        };
        assert_eq!(
            fields.get("pid"),
            Some(&std::process::id().to_string()),
            "{fields:?}"
        );
        assert!(positive("rss_bytes"), "{fields:?}");
        let completion = fields["event"].ends_with("_complete");
        assert_eq!(fields.contains_key("peak_bytes"), completion, "{fields:?}");
        assert!(!completion || positive("peak_bytes"), "{fields:?}");
        let stage_event = fields["event"].starts_with("stage_");
        assert_eq!(fields.contains_key("stage"), stage_event, "{fields:?}");
    }

    // Each id starts once, completes at most once, and reports its stages in
    // between, and every event of an id keeps the labels of its start. Only
    // the operations that fail stay open.
    let operation_id = |fields: &BTreeMap<String, String>| {
        fields
            .get("operation_id")
            .cloned()
            .unwrap_or_else(|| panic!("no operation_id: {fields:?}"))
    };
    let labels_of = |fields: &BTreeMap<String, String>| {
        (
            fields["operation"].clone(),
            fields.get("index_type").cloned(),
        )
    };
    let mut operations = BTreeMap::new();
    let mut completed = BTreeSet::new();
    for fields in events {
        let id = operation_id(fields);
        match fields["event"].as_str() {
            "operation_start" => {
                assert!(
                    operations.insert(id.clone(), labels_of(fields)).is_none(),
                    "{fields:?}"
                );
            }
            "operation_complete" => assert!(completed.insert(id.clone()), "{fields:?}"),
            _ => assert!(
                operations.contains_key(&id) && !completed.contains(&id),
                "a stage outside its operation: {fields:?}"
            ),
        }
        assert_eq!(operations.get(&id), Some(&labels_of(fields)), "{fields:?}");
    }
    let failed: usize = OPERATIONS
        .iter()
        .map(|&(_, _, started, completed)| started - completed)
        .sum();
    assert_eq!(operations.len() - completed.len(), failed, "{events:?}");

    // In this workload each maintenance operation starts each stage at most once.
    // A retried flush reports the stages of each attempt under its own operation ID.
    let mut started_stages = BTreeSet::new();
    for fields in events.iter().filter(|fields| {
        fields["event"] == "stage_start"
            && MAINTENANCE_STAGES.iter().any(|&(operation, stage, _, _)| {
                fields["operation"] == operation && fields["stage"] == stage
            })
    }) {
        assert!(
            started_stages.insert((operation_id(fields), fields["stage"].clone())),
            "a stage started twice in one operation: {fields:?}"
        );
    }
}

async fn check_repack_events(events: &MemoryEvents, mixed_plan: bool, conflicting_repack: bool) {
    let data = gen_batch()
        .col("id", array::step::<Int64Type>())
        .into_reader_rows(RowCount::from(4), BatchCount::from(2));
    let mut dataset = Dataset::write(
        data,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 4,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("extra".into(), "id + 1".into())]),
            None,
            None,
        )
        .await
        .unwrap();
    if mixed_plan {
        // Only the first fragment needs a rewrite. The second still needs a repack.
        dataset.delete("id < 2").await.unwrap();
    }
    if conflicting_repack {
        let mut latest = dataset.clone();
        latest.delete("id >= 4").await.unwrap();
    }
    let options = CompactionOptions {
        target_rows_per_fragment: 3,
        max_data_files_per_fragment: Some(1),
        ..Default::default()
    };
    let before = events.0.lock().unwrap().len();
    let result = compact_files(&mut dataset, options, None).await;
    if conflicting_repack {
        let error = result.unwrap_err();
        assert!(
            matches!(error, lance::Error::RetryableCommitConflict { .. })
                && error.to_string().contains(
                    "DataReplacement transaction was preempted by concurrent transaction Delete"
                ),
            "{error}"
        );
    } else {
        assert_eq!(result.unwrap().files_added, 2);
    }

    let recorded = events.0.lock().unwrap();
    let events = &recorded[before..];
    let operation_id = &events[0]["operation_id"];
    assert!(
        events
            .iter()
            .all(|event| event["operation"] == "compaction"
                && &event["operation_id"] == operation_id),
        "{events:?}"
    );
    let stages: Vec<_> = events
        .iter()
        .map(|event| {
            (
                event["event"].as_str(),
                event.get("stage").map(String::as_str),
            )
        })
        .filter(|(_, stage)| {
            !(mixed_plan && matches!(stage, Some("remap_indices" | "reserve_fragments")))
        })
        .collect();
    let execution = if mixed_plan {
        "rewrite_files"
    } else {
        "repack_files"
    };
    let mut expected = vec![
        ("operation_start", None),
        ("stage_start", Some(execution)),
        ("stage_complete", Some(execution)),
    ];
    if mixed_plan {
        expected.extend([
            ("stage_start", Some("commit")),
            ("stage_complete", Some("commit")),
        ]);
    }
    expected.push(("stage_start", Some("commit")));
    if !conflicting_repack {
        expected.extend([
            ("stage_complete", Some("commit")),
            ("operation_complete", None),
        ]);
    }
    assert_eq!(stages, expected);
}

async fn build_indices() {
    let data = gen_batch()
        .col("vec", array::rand_vec::<Float32Type>(Dimension::from(8)))
        .col("id", array::step::<Int64Type>())
        .col("text", array::random_sentence(2, 6, false))
        .col("category", array::cycle::<Int64Type>(vec![1, 2, 3]))
        .into_reader_rows(RowCount::from(128), BatchCount::from(2));
    let write_params = WriteParams {
        max_rows_per_file: 128,
        ..Default::default()
    };
    let mut dataset = Dataset::write(data, "memory://", Some(write_params))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);

    dataset
        .create_index(
            &["vec"],
            IndexType::IvfFlat,
            None,
            &VectorIndexParams::ivf_flat(2, MetricType::L2),
            false,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            None,
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();
    // A build whose commit fails starts and never completes: the restore
    // committed after the version the stale handle builds on is incompatible
    // with its index. The multi-segment FM-Index commits on a path of its own.
    let mut stale = dataset.clone();
    dataset.restore().await.unwrap();
    let btree_params = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    let fm_params = ScalarIndexParams {
        index_type: "fm".to_string(),
        params: Some(r#"{"num_segments": 2}"#.to_string()),
    };
    for (column, index_type, name, params) in [
        (
            "category",
            IndexType::BTree,
            "category_stale",
            &btree_params,
        ),
        ("text", IndexType::Fm, "text_fm_stale", &fm_params),
    ] {
        let error = stale
            .create_index(&[column], index_type, Some(name.to_string()), params, false)
            .await
            .unwrap_err();
        assert!(
            matches!(error, lance::Error::IncompatibleTransaction { .. })
                && error
                    .to_string()
                    .contains("incompatible with concurrent transaction Restore"),
            "{error:?}"
        );
    }
    // The generic scalar type trains the plugin its parameters name.
    dataset
        .create_index(
            &["category"],
            IndexType::Scalar,
            None,
            &ScalarIndexParams::new("bitmap".to_string()),
            false,
        )
        .await
        .unwrap();
    CreateIndexBuilder::new(
        &mut dataset,
        &["id"],
        IndexType::BTree,
        &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
    )
    .name("id_untrained".to_string())
    .train(false)
    .execute_uncommitted()
    .await
    .unwrap();
    let inverted_params = InvertedIndexParams::default();
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &inverted_params,
            false,
        )
        .await
        .unwrap();

    // The distributed workflow: one uncommitted segment per fragment, then a
    // merge of their metadata.
    let segment_uuid = Uuid::new_v4();
    let fragment_ids: Vec<u32> = dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .collect();
    for fragment_id in fragment_ids {
        CreateIndexBuilder::new(
            &mut dataset,
            &["text"],
            IndexType::Inverted,
            &inverted_params,
        )
        .name("text_segments".to_string())
        .fragments(vec![fragment_id])
        .index_uuid(segment_uuid)
        .execute_uncommitted()
        .await
        .unwrap();
    }
    dataset
        .merge_index_metadata(
            &segment_uuid,
            IndexType::Inverted,
            None,
            Arc::new(NoopIndexBuildProgress),
        )
        .await
        .unwrap();

    // Retraining rebuilds the vector index as one step of another operation.
    dataset
        .optimize_indices(&OptimizeOptions::retrain())
        .await
        .unwrap();

    // A multi-segment FM-Index builds one segment per fragment and commits
    // them together.
    dataset
        .create_index(
            &["text"],
            IndexType::Fm,
            Some("text_fm".to_string()),
            &fm_params,
            false,
        )
        .await
        .unwrap();
    assert_eq!(
        dataset.load_indices_by_name("text_fm").await.unwrap().len(),
        2
    );
}

/// Scalar and FTS rebuilds must keep the progress of their optimize operation,
/// both when appending a segment directly and in a split task with no old segments.
async fn run_nested_scalar_builds(events: &MemoryEvents) {
    let data = || {
        gen_batch()
            .col("id", array::step::<Int64Type>())
            .col("text", array::random_sentence(2, 3, false))
            .into_reader_rows(RowCount::from(4), BatchCount::from(2))
    };
    let params = WriteParams {
        max_rows_per_file: 4,
        ..Default::default()
    };
    let mut dataset = Dataset::write(data(), "memory://nested-scalar", Some(params.clone()))
        .await
        .unwrap();
    dataset
        .create_index(
            &["id"],
            IndexType::BTree,
            Some("id_idx".to_string()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            false,
        )
        .await
        .unwrap();
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("text_idx".to_string()),
            &InvertedIndexParams::default(),
            false,
        )
        .await
        .unwrap();

    let check_stages = |before: usize, stages: &[&str]| {
        let events = events.0.lock().unwrap();
        let operation_events = events[before..]
            .iter()
            .filter(|event| event["operation"] == "optimize_indices")
            .collect::<Vec<_>>();
        let starts = operation_events
            .iter()
            .filter(|event| event["event"] == "operation_start")
            .collect::<Vec<_>>();
        assert_eq!(starts.len(), 1, "{operation_events:?}");
        let operation_id = &starts[0]["operation_id"];
        assert!(operation_events.iter().all(|event| {
            &event["operation_id"] == operation_id && !event.contains_key("index_type")
        }));
        for stage in stages {
            for phase in ["stage_start", "stage_complete"] {
                assert_eq!(
                    operation_events
                        .iter()
                        .filter(|event| event["event"] == phase
                            && event
                                .get("stage")
                                .is_some_and(|value| value.as_str() == *stage))
                        .count(),
                    1,
                    "{stage} {phase}: {operation_events:?}"
                );
            }
        }
    };
    let append_params = WriteParams {
        mode: WriteMode::Append,
        ..params
    };
    dataset
        .append(data(), Some(append_params.clone()))
        .await
        .unwrap();
    for (name, stages) in [
        (
            "id_idx",
            &["load_data", "sort_values", "write_btree_index"][..],
        ),
        (
            "text_idx",
            &[
                "load_data",
                "tokenize_docs",
                "copy_partitions",
                "write_metadata",
            ][..],
        ),
    ] {
        let before = events.0.lock().unwrap().len();
        dataset
            .optimize_indices(&OptimizeOptions::append().index_names(vec![name.to_string()]))
            .await
            .unwrap();
        check_stages(before, stages);
    }

    dataset.append(data(), Some(append_params)).await.unwrap();
    let options = OptimizeOptions::append()
        .index_names(vec!["text_idx".to_string()])
        .max_rows_per_task(Some(4));
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!((plan.tasks.len(), plan.merges.len()), (2, 1));
    assert!(plan.tasks.iter().all(|task| task.old_segments.is_empty()));
    let mut built = Vec::new();
    for task in &plan.tasks {
        let before = events.0.lock().unwrap().len();
        built.extend(task.execute(&dataset).await.unwrap());
        check_stages(
            before,
            &[
                "load_data",
                "tokenize_docs",
                "copy_partitions",
                "write_metadata",
            ],
        );
    }
    let mut merged = Vec::new();
    for merge in &plan.merges {
        merged.extend(merge.execute(&dataset, &built).await.unwrap());
    }
    commit_optimize_indices(&mut dataset, merged, &options)
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 6);
    assert_eq!(dataset.count_rows(None).await.unwrap(), 24);
}

/// Two fragments of 64 rows, each a fragment of its own.
fn small_fragments() -> impl arrow_array::RecordBatchReader + Send + 'static {
    gen_batch()
        .col("id", array::step::<Int64Type>())
        .into_reader_rows(RowCount::from(64), BatchCount::from(2))
}

fn small_fragment_params(mode: WriteMode) -> Option<WriteParams> {
    Some(WriteParams {
        max_rows_per_file: 64,
        mode,
        ..Default::default()
    })
}

/// Warms nothing, or fails, which a MemWAL flush logs before it commits.
#[derive(Debug)]
struct TestWarmer {
    fail: bool,
}

#[async_trait::async_trait]
impl SsTableWarmer for TestWarmer {
    async fn warm(&self, _path: &str) -> lance::Result<()> {
        if self.fail {
            Err(lance::Error::io("a simulated warm failure"))
        } else {
            Ok(())
        }
    }
}

/// Compaction and `optimize_indices`, each also run the way a distributed job
/// runs it and once with a commit that fails, and MemWAL flushes, with and
/// without a maintained index, with a warm that succeeds and one that fails,
/// while a writer replays its WAL on open, and once on a fenced writer.
async fn run_maintenance() {
    let btree = ScalarIndexParams::for_builtin(BuiltinIndexType::BTree);
    // Two pairs of small fragments around one too large for the compaction
    // below, which then runs two tasks: they rewrite side by side in one stage,
    // and the compaction remaps the BTREE over the new fragments. A second
    // compaction finds nothing to do and completes without a stage.
    let mut dataset = Dataset::write(
        small_fragments(),
        "memory://maintenance",
        small_fragment_params(WriteMode::Create),
    )
    .await
    .unwrap();
    let mut large = small_fragment_params(WriteMode::Append).unwrap();
    large.max_rows_per_file = 128;
    dataset
        .append(small_fragments(), Some(large))
        .await
        .unwrap();
    dataset
        .append(small_fragments(), small_fragment_params(WriteMode::Append))
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 5);
    dataset
        .create_index(&["id"], IndexType::BTree, None, &btree, false)
        .await
        .unwrap();
    let two_tasks = CompactionOptions {
        target_rows_per_fragment: 100,
        ..Default::default()
    };
    assert_eq!(
        plan_compaction(&dataset, &two_tasks)
            .await
            .unwrap()
            .num_tasks(),
        2
    );
    compact_files(&mut dataset, two_tasks.clone(), None)
        .await
        .unwrap();
    assert_eq!(dataset.get_fragments().len(), 3);
    let nothing = compact_files(&mut dataset, two_tasks.clone(), None)
        .await
        .unwrap();
    assert_eq!(nothing.fragments_removed, 0);

    // Distributed: every task is an operation of its own, and so is the commit.
    // Only the new fragments are below the target, so they make one task.
    dataset
        .append(small_fragments(), small_fragment_params(WriteMode::Append))
        .await
        .unwrap();
    let plan = plan_compaction(&dataset, &two_tasks).await.unwrap();
    assert_eq!(plan.num_tasks(), 1);
    let mut rewritten = Vec::new();
    for task in plan.compaction_tasks() {
        rewritten.push(task.execute(&dataset).await.unwrap());
    }
    commit_compaction(
        &mut dataset,
        rewritten,
        Arc::new(DatasetIndexRemapperOptions::default()),
        &two_tasks,
    )
    .await
    .unwrap();

    // Two compactions of the same fragments: the second rewrites and remaps
    // them too, and then its commit fails. (Against a restore it would fail
    // earlier, in the `reserve_fragments` stage.)
    dataset
        .append(small_fragments(), small_fragment_params(WriteMode::Append))
        .await
        .unwrap();
    let mut stale = dataset.clone();
    compact_files(&mut dataset, CompactionOptions::default(), None)
        .await
        .unwrap();
    let error = compact_files(&mut stale, CompactionOptions::default(), None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, lance::Error::RetryableCommitConflict { .. }),
        "{error:?}"
    );

    // A distributed optimize plans, builds and merges the new fragments, each
    // an operation of its own, and its commit is not observed. Then nothing is
    // left to do: an optimize commits nothing, and still runs its stages
    // around the empty plan.
    dataset
        .append(small_fragments(), small_fragment_params(WriteMode::Append))
        .await
        .unwrap();
    let options = OptimizeOptions::default();
    let plan = plan_optimize_indices(&dataset, &options).await.unwrap();
    assert_eq!((plan.tasks.len(), plan.merges.len()), (1, 1));
    let mut built = Vec::new();
    for task in &plan.tasks {
        built.extend(task.execute(&dataset).await.unwrap());
    }
    let mut merged = Vec::new();
    for merge in &plan.merges {
        merged.extend(merge.execute(&dataset, &built).await.unwrap());
    }
    commit_optimize_indices(&mut dataset, merged, &options)
        .await
        .unwrap();
    let version = dataset.version().version;
    dataset.optimize_indices(&options).await.unwrap();
    assert_eq!(dataset.version().version, version);

    // An optimize whose commit fails, on a handle a restore has made stale.
    dataset
        .append(small_fragments(), small_fragment_params(WriteMode::Append))
        .await
        .unwrap();
    let mut stale = dataset.clone();
    dataset.restore().await.unwrap();
    let error = stale.optimize_indices(&options).await.unwrap_err();
    assert!(
        matches!(error, lance::Error::IncompatibleTransaction { .. })
            && error
                .to_string()
                .contains("incompatible with concurrent transaction Restore"),
        "{error:?}"
    );

    // A MemWAL writer that maintains a BTREE flushes its memtable, the index
    // included, when it closes, and warms the new generation before the
    // commit. A peer that claims the shard replays the WAL the next writer
    // leaves unflushed and fences that writer, whose flush on close then fails.
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut base = Dataset::write(small_fragments(), uri, None).await.unwrap();
    base.create_index(
        &["id"],
        IndexType::BTree,
        Some("id_idx".to_string()),
        &btree,
        false,
    )
    .await
    .unwrap();
    base.initialize_mem_wal()
        .unsharded()
        .maintained_indexes(["id_idx"])
        .execute()
        .await
        .unwrap();
    let config = ShardWriterConfig {
        durable_write: true,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        ..Default::default()
    };
    let shard_id = Uuid::new_v4();
    let batch = small_fragments().next().unwrap().unwrap();
    let warmer: Arc<dyn SsTableWarmer> = Arc::new(TestWarmer { fail: false });
    let warmed = ShardWriterConfig {
        warmer: Some(warmer),
        ..config.clone()
    };
    let writer = base.mem_wal_writer(shard_id, warmed).await.unwrap();
    writer.put(vec![batch.clone()]).await.unwrap();
    writer.close().await.unwrap();

    // Without a maintained index the flush takes the plain path. Its warm
    // fails, and the flush commits all the same.
    let plain_dir = tempfile::tempdir().unwrap();
    let mut plain = Dataset::write(small_fragments(), plain_dir.path().to_str().unwrap(), None)
        .await
        .unwrap();
    plain
        .initialize_mem_wal()
        .unsharded()
        .execute()
        .await
        .unwrap();
    let warmer: Arc<dyn SsTableWarmer> = Arc::new(TestWarmer { fail: true });
    let cold = ShardWriterConfig {
        warmer: Some(warmer),
        ..config.clone()
    };
    let writer = plain.mem_wal_writer(Uuid::new_v4(), cold).await.unwrap();
    writer.put(vec![batch.clone()]).await.unwrap();
    writer.close().await.unwrap();

    // A WAL-only writer leaves five entries that name no generation. A writer
    // whose memtable has room for two batches replays them when it opens: it
    // flushes twice, each time before an entry that no longer fits, and
    // flushes the fifth on close.
    let replayed_shard = Uuid::new_v4();
    let wal_only = ShardWriterConfig {
        enable_memtable: false,
        ..config.clone()
    };
    let writer = plain
        .mem_wal_writer(replayed_shard, wal_only)
        .await
        .unwrap();
    for _ in 0..5 {
        writer.put(vec![batch.clone()]).await.unwrap();
    }
    writer.close().await.unwrap();
    let two_batches = ShardWriterConfig {
        max_memtable_batches: 2,
        ..config.clone()
    };
    let writer = plain
        .mem_wal_writer(replayed_shard, two_batches)
        .await
        .unwrap();
    let manifest = writer.manifest().await.unwrap().unwrap();
    assert_eq!(manifest.sstables.len(), 2, "{manifest:?}");
    writer.close().await.unwrap();

    let fenced = base.mem_wal_writer(shard_id, config.clone()).await.unwrap();
    fenced.put(vec![batch]).await.unwrap();
    let peer = base.mem_wal_writer(shard_id, config).await.unwrap();
    let error = fenced.close().await.unwrap_err();
    assert!(error.fence_reason().is_some(), "{error:?}");
    peer.close().await.unwrap();
}

/// Fails the first write under a MemWAL generation that reaches the wrapped
/// store, and lets later ones through. Data files on a local disk are written
/// around the store, so the write that fails is one the generation's commit
/// makes.
#[derive(Debug)]
struct FailingGenerationWrapper(Arc<Mutex<ProxyObjectStorePolicy>>);

impl FailingGenerationWrapper {
    fn new() -> Self {
        let generation_writes = AtomicUsize::new(0);
        let mut policy = ProxyObjectStorePolicy::new();
        policy.set_before_policy(
            "fail_first_generation_write",
            Arc::new(move |method: &str, path: &object_store::path::Path| {
                // Below the MemWAL directory, so the temporary directory's
                // own name cannot match.
                let in_generation = path
                    .as_ref()
                    .rsplit_once("_mem_wal/")
                    .is_some_and(|(_, below)| below.contains("_gen_"));
                if method.starts_with("put")
                    && in_generation
                    && generation_writes.fetch_add(1, Ordering::SeqCst) == 0
                {
                    return Err(lance::Error::io("a simulated storage failure"));
                }
                Ok(())
            }),
        );
        Self(Arc::new(Mutex::new(policy)))
    }
}

impl WrappingObjectStore for FailingGenerationWrapper {
    fn wrap(
        &self,
        _store_prefix: &str,
        original: Arc<dyn object_store::ObjectStore>,
    ) -> Arc<dyn object_store::ObjectStore> {
        Arc::new(ProxyObjectStore::new(original, self.0.clone()))
    }

    // Only writes fail, so listings keep their pushdown.
    fn wrap_paginated(
        &self,
        _store_prefix: &str,
        original: Arc<dyn object_store::list::PaginatedListStore>,
    ) -> Option<Arc<dyn object_store::list::PaginatedListStore>> {
        Some(original)
    }
}

/// A MemWAL flush whose first attempt fails to write its generation and whose
/// retry lands: each attempt is an operation of its own.
async fn run_retried_flush() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut dataset = Dataset::write(small_fragments(), uri, None).await.unwrap();
    dataset
        .initialize_mem_wal()
        .unsharded()
        .execute()
        .await
        .unwrap();
    let failing = DatasetBuilder::from_uri(uri)
        .with_store_params(ObjectStoreParams {
            object_store_wrapper: Some(Arc::new(FailingGenerationWrapper::new())),
            ..Default::default()
        })
        .load()
        .await
        .unwrap();
    let config = ShardWriterConfig {
        durable_write: true,
        max_wal_flush_interval: Some(Duration::from_millis(10)),
        ..Default::default()
    };
    let writer = failing
        .mem_wal_writer(Uuid::new_v4(), config)
        .await
        .unwrap();
    let batch = small_fragments().next().unwrap().unwrap();
    writer.put(vec![batch]).await.unwrap();
    let flushed = writer.force_freeze_active().await.unwrap();
    // A flush that keeps failing retries for minutes rather than failing.
    tokio::time::timeout(Duration::from_secs(30), flushed.wait())
        .await
        .expect("the retried flush lands")
        .unwrap();
    assert_eq!(writer.stats().memtable_flush_failures, 1);
    writer.close().await.unwrap();
}

/// Rows keyed on `id`, the primary key of the table, each holding `value`.
fn keyed_rows(ids: Range<i64>, value: i64) -> RecordBatch {
    let id = Field::new("id", DataType::Int64, false).with_metadata(HashMap::from([(
        "lance-schema:unenforced-primary-key".to_string(),
        "true".to_string(),
    )]));
    let schema = Arc::new(Schema::new(vec![
        id,
        Field::new("value", DataType::Int64, false),
    ]));
    let values = Int64Array::from(vec![value; ids.clone().count()]);
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from_iter_values(ids)),
            Arc::new(values),
        ],
    )
    .unwrap()
}

/// A `merge_insert` on `id` that updates the rows it matches and inserts the
/// others.
fn upsert(dataset: Arc<Dataset>) -> MergeInsertJob {
    MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .when_not_matched(WhenNotMatched::InsertAll)
        .try_build()
        .unwrap()
}

/// Writes, `merge_insert`, updates and deletes: writes, fragments, a
/// `merge_insert` and a delete that commit nothing, two `merge_insert`s of
/// one new key that race, and every kind once more with a commit that fails,
/// the write both from batches and from a stream.
async fn run_writes() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let dataset = InsertBuilder::new(uri)
        .execute(vec![keyed_rows(0..64, 0)])
        .await
        .unwrap();
    let dataset = Arc::new(dataset);
    let append = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    let schema = keyed_rows(0..1, 0).schema();
    let source = |rows: RecordBatch| RecordBatchIterator::new([Ok(rows)], schema.clone());

    let insert = InsertBuilder::new(dataset.clone()).with_params(&append);
    insert
        .execute_uncommitted(vec![keyed_rows(64..128, 0)])
        .await
        .unwrap();
    insert
        .execute_uncommitted_stream(source(keyed_rows(64..128, 0)))
        .await
        .unwrap();
    FileFragment::create(
        uri,
        100,
        source(keyed_rows(64..128, 0)),
        Some(append.clone()),
    )
    .await
    .unwrap();
    FileFragment::create_fragments(uri, source(keyed_rows(64..128, 0)), Some(append.clone()))
        .await
        .unwrap();
    upsert(dataset.clone())
        .execute_uncommitted_batches(vec![keyed_rows(0..8, 1)])
        .await
        .unwrap();
    DeleteBuilder::new(dataset.clone(), "id < 8")
        .execute_uncommitted()
        .await
        .unwrap();

    let dataset = UpdateBuilder::new(dataset)
        .update_where("id < 4")
        .unwrap()
        .set("value", "1")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap()
        .new_dataset;
    let dataset = DeleteBuilder::new(dataset, "id >= 60")
        .execute()
        .await
        .unwrap()
        .new_dataset;

    // Both insert the same new key, which the primary key makes a conflict:
    // the one that commits second retries, and its retry updates the row the
    // first inserted.
    upsert(dataset.clone())
        .execute_reader(source(keyed_rows(1000..1001, 1)))
        .await
        .unwrap();
    let (dataset, stats) = upsert(dataset)
        .execute_batches(vec![keyed_rows(1000..1001, 2)])
        .await
        .unwrap();
    assert_eq!(stats.num_attempts, 2);

    // A restore makes the handle stale, so every commit on it fails, and none
    // of them retries.
    let stale = dataset.clone();
    let mut latest = dataset.as_ref().clone();
    latest.restore().await.unwrap();
    let mut stale_append = stale.as_ref().clone();
    let errors = [
        InsertBuilder::new(stale.clone())
            .with_params(&append)
            .execute(vec![keyed_rows(2000..2001, 0)])
            .await
            .unwrap_err(),
        stale_append
            .append(source(keyed_rows(2001..2002, 0)), Some(append.clone()))
            .await
            .unwrap_err(),
        upsert(stale.clone())
            .execute_batches(vec![keyed_rows(10..11, 3)])
            .await
            .unwrap_err(),
        UpdateBuilder::new(stale.clone())
            .update_where("id = 11")
            .unwrap()
            .set("value", "3")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap_err(),
        DeleteBuilder::new(stale, "id = 12")
            .execute()
            .await
            .unwrap_err(),
    ];
    for error in errors {
        assert!(
            matches!(error, lance::Error::IncompatibleTransaction { .. })
                && error
                    .to_string()
                    .contains("incompatible with concurrent transaction Restore"),
            "{error:?}"
        );
    }
}
