// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ordinary overlay scan diagnostics. Prepare an immutable fixture once, then
//! run the same executable at both source revisions against that fixture.
//! `verify` checks every value/order without collecting the result or timing it.
//! `run` measures complete consumption; use only on a managed benchmark VM.
//!
//! Commands: `prepare ROOT ROWS OVERLAYS VERSION`, or
//! `verify|run ROOT full|ranges BATCH_ROWS`. VERSION is `2.2` or `2.3`.

#![allow(clippy::print_stdout)]

use std::error::Error;
use std::hint::black_box;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use futures::{StreamExt, TryStreamExt};
use lance::Dataset;
use lance::dataset::fragment::FragReadConfig;
use lance::dataset::transaction::Operation;
use lance::dataset::{WriteDestination, WriteParams};
use lance_core::ROW_ADDR;
use lance_core::utils::address::RowAddress;
use lance_file::version::LanceFileVersion;
use serde::{Deserialize, Serialize};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[derive(Serialize, Deserialize)]
struct Fixture {
    rows: u32,
    overlays: u32,
    version: String,
}

async fn prepare(root: &Path, fixture: Fixture) -> Result<()> {
    if fixture.rows == 0 || fixture.overlays > 32 {
        return Err("rows must be positive; overlays must be in 0..=32".into());
    }
    let version = match fixture.version.as_str() {
        "2.2" => LanceFileVersion::V2_2,
        "2.3" => LanceFileVersion::V2_3,
        _ => return Err("version must be 2.2 or 2.3".into()),
    };
    // Never overwrite a fixture used by another source/repetition.
    std::fs::create_dir(root)?;
    let uri = root.join("data.lance").to_string_lossy().into_owned();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("val", DataType::Int64, false),
    ]));
    let rows = fixture.rows;
    let batch_schema = schema.clone();
    let batches = (0..rows).step_by(8192).map(move |start| {
        let end = start.saturating_add(8192).min(rows);
        RecordBatch::try_new(
            batch_schema.clone(),
            vec![
                Arc::new(UInt32Array::from_iter_values(start..end)),
                Arc::new(Int64Array::from_iter_values((start..end).map(i64::from))),
            ],
        )
    });
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(batches, schema.clone()),
        uri.as_str(),
        Some(WriteParams {
            max_rows_per_file: fixture.rows as usize,
            max_rows_per_group: 8192,
            data_storage_version: Some(version),
            ..Default::default()
        }),
    )
    .await?;
    if dataset.get_fragments().len() != 1 {
        return Err("fixture must contain exactly one fragment".into());
    }
    let value_id = dataset
        .schema()
        .fields
        .iter()
        .find(|field| field.name == "val")
        .ok_or("fixture has no val field")?
        .id;
    let projection = dataset.schema().project_by_ids(&[value_id], true);
    for layer in 0..fixture.overlays {
        let fragment = dataset.get_fragment(0).ok_or("fixture has no fragment 0")?;
        let mut writer = fragment.write_overlay(&projection).await?;
        // Sparse disjoint coverage plus row zero in every layer tests newest-wins.
        let offsets: Vec<u32> = std::iter::once(0)
            .chain((layer..fixture.rows).step_by(1024).filter(|row| *row != 0))
            .collect();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(ROW_ADDR, DataType::UInt64, false),
                Field::new("val", DataType::Int64, false),
            ])),
            vec![
                Arc::new(UInt64Array::from_iter_values(
                    offsets
                        .iter()
                        .map(|row| u64::from(RowAddress::new_from_parts(0, *row))),
                )),
                Arc::new(Int64Array::from_iter_values(offsets.iter().map(|row| {
                    i64::from(*row) + (i64::from(layer) + 1) * 1_000_000_000
                }))),
            ],
        )?;
        writer.write_batch(&batch).await?;
        let group = writer.finish().await?.ok_or("empty overlay fixture")?;
        let read_version = dataset.version().version;
        dataset = Dataset::commit(
            WriteDestination::Dataset(Arc::new(dataset)),
            Operation::DataOverlay {
                groups: vec![group],
            },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await?;
    }
    std::fs::write(root.join("fixture.json"), serde_json::to_vec(&fixture)?)?;
    println!(
        "{}",
        serde_json::json!({"prepared": fixture.rows, "overlays": fixture.overlays, "version": fixture.version})
    );
    Ok(())
}

async fn consume(root: &Path, scenario: &str, batch_rows: u32, verify: bool) -> Result<()> {
    if batch_rows == 0 || !matches!(scenario, "full" | "ranges") {
        return Err("batch rows must be positive; scenario must be full or ranges".into());
    }
    let fixture: Fixture = serde_json::from_slice(&std::fs::read(root.join("fixture.json"))?)?;
    let ranges = scenario == "ranges";
    let mut expected = (0..fixture.rows).filter(|row| !ranges || row % 4096 < 32);
    let expected_count = if ranges {
        u64::from(fixture.rows / 4096) * 32 + u64::from((fixture.rows % 4096).min(32))
    } else {
        u64::from(fixture.rows)
    };
    let start = (!verify).then(Instant::now);
    let uri = root.join("data.lance").to_string_lossy().into_owned();
    let dataset = Dataset::open(&uri).await?;
    let fragment = dataset.get_fragment(0).ok_or("fixture has no fragment 0")?;
    let reader = fragment
        .open(dataset.schema(), FragReadConfig::default())
        .await?;
    let mut stream = if ranges {
        let selection = (0..fixture.rows)
            .step_by(4096)
            .map(|row| u64::from(row)..(u64::from(row) + 32).min(u64::from(fixture.rows)))
            .collect::<Vec<_>>();
        reader
            .read_ranges(selection.into(), batch_rows)
            .await?
            .buffered(2)
            .boxed()
    } else {
        reader.read_all(batch_rows).await?.buffered(2).boxed()
    };
    let mut first_batch_us = None;
    let mut rows = 0_u64;
    let mut batches = 0_u64;
    while let Some(batch) = stream.try_next().await? {
        if batches == 0 {
            first_batch_us = start.map(|start| start.elapsed().as_micros());
        }
        rows += batch.num_rows() as u64;
        batches += 1;
        if verify {
            let ids = batch["id"]
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or("wrong id type")?;
            let values = batch["val"]
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or("wrong value type")?;
            if ids.null_count() != 0 || values.null_count() != 0 {
                return Err("unexpected null in fixture result".into());
            }
            for i in 0..batch.num_rows() {
                let row = expected.next().ok_or("extra result row")?;
                let layer = if row == 0 {
                    fixture.overlays
                } else if row % 1024 < fixture.overlays {
                    row % 1024 + 1
                } else {
                    0
                };
                let value = i64::from(row) + i64::from(layer) * 1_000_000_000;
                if ids.value(i) != row || values.value(i) != value {
                    return Err(format!("wrong output at physical row {row}").into());
                }
            }
        }
        black_box(batch);
    }
    if rows != expected_count || (verify && expected.next().is_some()) {
        return Err(format!("expected {expected_count} rows, received {rows}").into());
    }
    println!(
        "{}",
        serde_json::json!({
            "scenario": scenario, "rows": rows, "batches": batches,
            "overlays": fixture.overlays, "version": fixture.version,
            "batch_rows": batch_rows, "batch_concurrency": 2,
            "verified": verify, "first_batch_us": first_batch_us,
            "full_drain_us": start.map(|start| start.elapsed().as_micros()),
        })
    );
    Ok(())
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("prepare") if args.len() == 6 => {
            prepare(
                Path::new(&args[2]),
                Fixture {
                    rows: args[3].parse()?,
                    overlays: args[4].parse()?,
                    version: args[5].clone(),
                },
            )
            .await
        }
        Some(mode @ ("verify" | "run")) if args.len() == 5 => {
            consume(
                Path::new(&args[2]),
                &args[3],
                args[4].parse()?,
                mode == "verify",
            )
            .await
        }
        _ => Err(
            "usage: prepare ROOT ROWS OVERLAYS VERSION | verify|run ROOT full|ranges BATCH_ROWS"
                .into(),
        ),
    }
}
