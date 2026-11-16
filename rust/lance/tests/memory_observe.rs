// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `LANCE_MEMORY_OBSERVE` reports the memory of index builds as trace events
//! and, with the `metrics` feature, through the `metrics` facade.
//!
//! This file must stay a single-test integration binary: Lance reads the
//! switch once per process, and the tracing subscriber and the metrics
//! recorder are process-global, so a sibling test could read the switch
//! before the `set_var` below or report into the same subscriber or recorder.
#![cfg(target_os = "linux")]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use arrow_array::types::{Float32Type, Int64Type};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::vector::VectorIndexParams;
use lance::index::{CreateIndexBuilder, DatasetIndexExt};
#[cfg(feature = "metrics")]
use lance::memory::{
    METRIC_OPERATION_PEAK, METRIC_RSS, METRIC_RSS_PEAK, METRIC_STAGE_PEAK, refresh_metrics,
};
use lance_core::utils::tracing::TRACE_MEMORY;
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_linalg::distance::MetricType;
#[cfg(feature = "metrics")]
use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};
use tracing_subscriber::Layer;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::layer::SubscriberExt;
use uuid::Uuid;

#[cfg(feature = "metrics")]
type Recorded = Vec<(String, Vec<(String, String)>, DebugValue)>;

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

/// Stages the builds below report, as `(operation, index_type, stage)`.
const STAGES: &[(&str, &str, &str)] = &[
    ("create_index", "ivf_flat", "train_ivf"),
    ("create_index", "ivf_flat", "train_quantizer"),
    ("create_index", "ivf_flat", "shuffle"),
    ("create_index", "ivf_flat", "merge_partitions"),
    ("create_index", "btree", "load_data"),
    ("create_index", "bitmap", "load_data"),
    ("create_index", "fm", "load_data"),
    ("create_index", "inverted", "tokenize_docs"),
    ("create_index", "inverted", "write_metadata"),
    (
        "merge_index_metadata",
        "inverted",
        "read_partition_metadata",
    ),
    ("merge_index_metadata", "inverted", "remap_partition_files"),
    ("merge_index_metadata", "inverted", "write_merged_metadata"),
];

/// Observed builds below, as `(operation, index_type, started, completed)`.
/// The inverted index is built once whole and once as one segment per
/// fragment, which the merge then joins. The FM-Index builds one segment per
/// fragment and commits them together. One BTREE and one FM-Index build fail
/// at their commit, so they start and never complete. The untrained BTREE and
/// the retraining inside `optimize_indices` add nothing.
const BUILDS: &[(&str, &str, usize, usize)] = &[
    ("create_index", "ivf_flat", 1, 1),
    ("create_index", "btree", 2, 1),
    ("create_index", "bitmap", 1, 1),
    ("create_index", "inverted", 3, 3),
    ("merge_index_metadata", "inverted", 1, 1),
    ("create_index", "fm", 2, 1),
];

#[test]
fn test_index_builds_report_memory_peaks() {
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
        .block_on(build_indices());

    check_events(&events.0.lock().unwrap());
    #[cfg(feature = "metrics")]
    check_metrics(&snapshotter);
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
            &[
                ("operation", operation),
                ("index_type", index_type),
                ("stage", stage),
            ],
        );
        assert!(
            !peaks.is_empty() && peaks.iter().all(|peak| *peak > 0.0),
            "{operation} {index_type} {stage}: {peaks:?} in {recorded:?}"
        );
    }
    for &(operation, index_type, _, completed) in BUILDS {
        let peaks = histogram(
            &recorded,
            METRIC_OPERATION_PEAK,
            &[("operation", operation), ("index_type", index_type)],
        );
        assert_eq!(
            peaks.len(),
            completed,
            "{operation} {index_type}: {recorded:?}"
        );
        assert!(
            peaks.iter().all(|peak| *peak > 0.0),
            "{operation} {index_type}: {peaks:?}"
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
    let count = |event: &str, labels: &[(&str, &str)]| {
        events
            .iter()
            .filter(|fields| {
                fields.get("event").is_some_and(|value| value == event)
                    && labels
                        .iter()
                        .all(|(name, label)| fields.get(*name).is_some_and(|value| value == label))
            })
            .count()
    };
    for &(operation, index_type, stage) in STAGES {
        let labels = [
            ("operation", operation),
            ("index_type", index_type),
            ("stage", stage),
        ];
        let completed = count("stage_complete", &labels);
        assert!(
            completed > 0 && count("stage_start", &labels) == completed,
            "{operation} {index_type} {stage}: {events:?}"
        );
    }
    for &(operation, index_type, started, completed) in BUILDS {
        let labels = [("operation", operation), ("index_type", index_type)];
        for (event, builds) in [
            ("operation_start", started),
            ("operation_complete", completed),
        ] {
            assert_eq!(
                count(event, &labels),
                builds,
                "{event} {operation} {index_type}: {events:?}"
            );
        }
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

    // Each id starts once and completes at most once, and every event of an id
    // keeps the labels of its start. Only the builds that fail stay open.
    let operation_id = |fields: &BTreeMap<String, String>| {
        fields
            .get("operation_id")
            .cloned()
            .unwrap_or_else(|| panic!("no operation_id: {fields:?}"))
    };
    let mut operations = BTreeMap::new();
    for fields in events
        .iter()
        .filter(|fields| fields["event"] == "operation_start")
    {
        let labels = (&fields["operation"], &fields["index_type"]);
        assert!(
            operations.insert(operation_id(fields), labels).is_none(),
            "{fields:?}"
        );
    }
    let mut completed = BTreeSet::new();
    for fields in events {
        let id = operation_id(fields);
        assert_eq!(
            operations.get(&id),
            Some(&(&fields["operation"], &fields["index_type"])),
            "{fields:?}"
        );
        if fields["event"] == "operation_complete" {
            assert!(completed.insert(id), "{fields:?}");
        }
    }
    let failed: usize = BUILDS
        .iter()
        .map(|&(_, _, started, completed)| started - completed)
        .sum();
    assert_eq!(operations.len() - completed.len(), failed, "{events:?}");
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
