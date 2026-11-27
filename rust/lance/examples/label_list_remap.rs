// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build a deterministic single-segment fixture, then run on a fresh copy:
//! ```text
//! cargo run --profile release-with-debug -p lance --example label_list_remap -- build BASE STRIDE
//! cp -R BASE COPY
//! cargo run --profile release-with-debug -p lance --example label_list_remap -- run COPY STRIDE FRAGMENTS
//! ```
//! STRIDE=1 gives common labels, 128 gives sparse labels. Rewrite 1 or 64 fragments.
//! Timings separate the remap commit from rewriting data and from warm indexed counts.

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{
    Array, ListArray, RecordBatch, RecordBatchIterator, UInt32Array, types::UInt32Type,
};
use arrow_schema::{DataType, Field, Schema};
use lance::{
    Dataset, Result,
    dataset::{
        WriteParams,
        index::DatasetIndexRemapperOptions,
        optimize::{CompactionOptions, CompactionTask, TaskData, commit_compaction},
    },
    index::DatasetIndexExt,
};
use lance_index::{IndexType, scalar::ScalarIndexParams};

const FRAGMENTS: usize = 64;
const ROWS_PER_FRAGMENT: usize = 4096;

fn labels(id: usize, stride: usize) -> Vec<Option<u32>> {
    if !id.is_multiple_of(stride) {
        return vec![];
    }
    (0..16)
        .map(|i| Some(((id / stride + i) % 32) as u32))
        .collect()
}

async fn build(uri: &str, stride: usize) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new(
            "labels",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            true,
        ),
    ]));
    let batches = (0..FRAGMENTS)
        .map(|f| {
            let ids = f * ROWS_PER_FRAGMENT..(f + 1) * ROWS_PER_FRAGMENT;
            let values = ListArray::from_iter_primitive::<UInt32Type, _, _>(
                ids.clone().map(|id| Some(labels(id, stride))),
            );
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(UInt32Array::from_iter_values(ids.map(|id| id as u32))),
                    Arc::new(values),
                ],
            )
        })
        .collect::<Vec<_>>();
    let mut ds = Dataset::write(
        RecordBatchIterator::new(batches, schema),
        uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT,
            max_rows_per_group: ROWS_PER_FRAGMENT,
            ..Default::default()
        }),
    )
    .await?;
    ds.create_index(
        &["labels"],
        IndexType::LabelList,
        Some("labels_idx".into()),
        &ScalarIndexParams::default(),
        false,
    )
    .await?;
    Ok(())
}

async fn run(uri: &str, stride: usize, rewritten: usize) -> Result<()> {
    let mut ds = Dataset::open(uri).await?;
    assert_eq!(ds.fragments().len(), FRAGMENTS);
    let original = ds.load_indices_by_name("labels_idx").await?;
    assert_eq!(original.len(), 1);
    let options = CompactionOptions {
        target_rows_per_fragment: ROWS_PER_FRAGMENT * 2,
        ..Default::default()
    };
    let task = CompactionTask {
        task: TaskData::rewrite_fragments(ds.fragments()[..rewritten].to_vec()),
        read_version: ds.manifest().version,
        options: options.clone(),
    };
    let result = task.execute(&ds).await?;
    let start = Instant::now();
    let metrics = commit_compaction(
        &mut ds,
        vec![result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        &options,
    )
    .await?;
    let remap_ms = start.elapsed().as_secs_f64() * 1000.;
    assert_eq!(metrics.fragments_removed as usize, rewritten);
    let ds = Dataset::open(uri).await?;
    let segments = ds.load_indices_by_name("labels_idx").await?;
    assert_eq!(segments.len(), 1);
    assert_ne!(segments[0].uuid, original[0].uuid);
    let predicate = "array_has_any(labels, [7])";
    let mut scanner = ds.scan();
    scanner.project(&["id"])?.filter(predicate)?;
    let plan = scanner.explain_plan(false).await?;
    assert!(
        plan.contains("ScalarIndexQuery") && plan.contains("LabelList"),
        "{plan}"
    );
    let mut times = vec![];
    for i in 0..41 {
        let start = Instant::now();
        let count = ds.count_rows(Some(predicate.to_owned())).await?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        assert_eq!(count, FRAGMENTS * ROWS_PER_FRAGMENT / stride / 2);
        if i > 0 {
            times.push(elapsed);
        }
    }
    let batch = scanner.try_into_batch().await?;
    let mut actual = batch["id"]
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap()
        .values()
        .to_vec();
    actual.sort_unstable();
    let expected = (0..FRAGMENTS * ROWS_PER_FRAGMENT)
        .step_by(stride)
        .filter(|&id| labels(id, stride).contains(&Some(7)))
        .map(|id| id as u32)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert_eq!(
        ds.count_rows(Some("array_has_any(labels, [32])".into()))
            .await?,
        0
    );
    times.sort_by(f64::total_cmp);
    println!(
        "stride={stride} rewritten={rewritten} remap_commit_ms={remap_ms:.3} search_p50_ms={:.3} search_p95_ms={:.3} segments={} matches={}",
        times[19],
        times[37],
        segments.len(),
        actual.len()
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 4 {
        return Err(lance_core::Error::invalid_input(
            "expected build URI STRIDE or run URI STRIDE FRAGMENTS",
        ));
    }
    let stride = args[3].parse().expect("positive stride");
    assert!(matches!(stride, 1 | 128));
    match args[1].as_str() {
        "build" => build(&args[2], stride).await,
        "run" if args.len() == 5 => {
            run(
                &args[2],
                stride,
                args[4].parse().expect("number of fragments"),
            )
            .await
        }
        _ => panic!("expected build URI STRIDE or run URI STRIDE REWRITTEN"),
    }
}
