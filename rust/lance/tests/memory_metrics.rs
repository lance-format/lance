// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `LANCE_MEMORY_OBSERVE` reports the memory of index builds through the
//! `metrics` facade.
//!
//! This file must stay a single-test integration binary: Lance reads the
//! switch once per process and the recorder is process-global, so a sibling
//! test could read the switch before the `set_var` below or record into the
//! same recorder.
#![cfg(target_os = "linux")]

use std::sync::Arc;

use arrow_array::types::{Float32Type, Int64Type};
use lance::Dataset;
use lance::dataset::WriteParams;
use lance::index::vector::VectorIndexParams;
use lance::index::{CreateIndexBuilder, DatasetIndexExt};
use lance::memory::{
    METRIC_OPERATION_PEAK, METRIC_RSS, METRIC_RSS_PEAK, METRIC_STAGE_PEAK, refresh_metrics,
};
use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::progress::NoopIndexBuildProgress;
use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_linalg::distance::MetricType;
use metrics_util::debugging::{DebugValue, DebuggingRecorder};
use uuid::Uuid;

type Recorded = Vec<(String, Vec<(String, String)>, DebugValue)>;

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

fn gauge(recorded: &Recorded, name: &str) -> Option<f64> {
    recorded.iter().find_map(|(metric, _, value)| match value {
        DebugValue::Gauge(value) if metric == name => Some(value.0),
        _ => None,
    })
}

#[test]
fn test_index_builds_report_memory_peaks() {
    // SAFETY: this is the only test in the binary and no other thread runs
    // yet: the runtime that runs Lance is built below.
    unsafe { std::env::set_var("LANCE_MEMORY_OBSERVE", "1") };
    let recorder = DebuggingRecorder::new();
    let snapshotter = recorder.snapshotter();
    recorder.install().unwrap();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(build_indices());
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

    for (operation, index_type, stage) in [
        ("create_index", "ivf_flat", "train_ivf"),
        ("create_index", "ivf_flat", "train_quantizer"),
        ("create_index", "ivf_flat", "shuffle"),
        ("create_index", "ivf_flat", "merge_partitions"),
        ("create_index", "btree", "load_data"),
        ("create_index", "bitmap", "load_data"),
        ("create_index", "inverted", "tokenize_docs"),
        ("create_index", "inverted", "write_metadata"),
        (
            "merge_index_metadata",
            "inverted",
            "read_partition_metadata",
        ),
        ("merge_index_metadata", "inverted", "remap_partition_files"),
        ("merge_index_metadata", "inverted", "write_merged_metadata"),
    ] {
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
    // The inverted index is built once whole and once as one segment per
    // fragment, which the merge then joins. The untrained BTREE and the
    // retraining inside `optimize_indices` add nothing.
    for (operation, index_type, builds) in [
        ("create_index", "ivf_flat", 1),
        ("create_index", "btree", 1),
        ("create_index", "bitmap", 1),
        ("create_index", "inverted", 3),
        ("merge_index_metadata", "inverted", 1),
    ] {
        let peaks = histogram(
            &recorded,
            METRIC_OPERATION_PEAK,
            &[("operation", operation), ("index_type", index_type)],
        );
        assert_eq!(
            peaks.len(),
            builds,
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
}
