// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Full FTS maintenance after deleting a quarter of the documents and appending
//! replacements. Compare the same command on the base and patched revisions:
//!
//! cargo bench -p lance --no-default-features --bench fts_update --profile release-no-lto
//!
//! FTS_BENCH_ROWS (default 50000) and FTS_BENCH_REPEATS (default 3) control size.
//! FTS_BENCH_RARE_TERMS (default 8) controls unique terms per generated document.
//! FTS_BENCH_POSITIONS=true enables positional postings (default false).
//! FTS_BENCH_CORPUS optionally names a Lance dataset with a Utf8 `text` column;
//! it is read only. Otherwise deterministic documents mix common and rare terms.
//! Every iteration restores the same seed snapshot in a temporary dataset.
//! Timings include the complete optimize call and its commit, not just remap.

// The benchmark reports its measurements to the person running it.
#![allow(clippy::print_stdout)]

use std::{env, sync::Arc, time::Instant};

use arrow_array::{
    Array, ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray,
};
use lance::{
    Dataset,
    dataset::{WriteParams, optimize::compact_files},
    index::DatasetIndexExt,
};
use lance_index::{
    IndexType,
    optimize::OptimizeOptions,
    scalar::{FullTextSearchQuery, InvertedIndexParams},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn setting(name: &str, default: usize) -> Result<usize> {
    Ok(env::var(name)
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(default))
}

fn word(mut id: usize) -> String {
    let mut result = String::from("token");
    for _ in 0..5 {
        result.push((b'a' + (id % 26) as u8) as char);
        id /= 26;
    }
    result
}

async fn input(rows: usize) -> Result<Vec<String>> {
    if let Ok(path) = env::var("FTS_BENCH_CORPUS") {
        let dataset = Dataset::open(&path).await?;
        let batch = dataset
            .scan()
            .project(&["text"])?
            .limit(Some(rows as i64), None)?
            .try_into_batch()
            .await?;
        let text = batch["text"]
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("text must be Utf8")?;
        println!(
            "corpus_version={} rows={}",
            dataset.version().version,
            text.len()
        );
        return Ok(text
            .iter()
            .map(|text| text.unwrap_or_default().to_owned())
            .collect());
    }
    let rare_terms = setting("FTS_BENCH_RARE_TERMS", 8)?;
    Ok((0..rows)
        .map(|row| {
            let mut text = String::from("class service request return result common common");
            for term in 0..rare_terms {
                text.push(' ');
                text.push_str(&word(row * rare_terms + term));
            }
            text
        })
        .collect())
}

fn batch(text: &[String], start_id: usize, fresh: bool) -> RecordBatch {
    let ids: Vec<_> = (start_id..start_id + text.len())
        .map(|id| id as i32)
        .collect();
    let text: Vec<_> = ids
        .iter()
        .zip(text)
        .map(|(&id, text)| {
            let marker = if fresh {
                "freshmarker"
            } else if id % 4 == 0 {
                "deletedmarker"
            } else {
                "keptmarker"
            };
            format!("sharedmarker {marker} {text}")
        })
        .collect();
    RecordBatch::try_from_iter([
        ("id", Arc::new(Int32Array::from(ids)) as ArrayRef),
        ("text", Arc::new(StringArray::from(text)) as ArrayRef),
    ])
    .unwrap()
}

fn reader(batch: RecordBatch) -> impl RecordBatchReader + Send {
    let schema = batch.schema();
    RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema)
}

async fn check(dataset: &Dataset, term: &str, expected: usize) -> Result<()> {
    let actual = dataset
        .scan()
        .full_text_search(FullTextSearchQuery::new(term.to_owned()))?
        .count_rows()
        .await?;
    assert_eq!(actual as usize, expected, "query {term}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let text = input(setting("FTS_BENCH_ROWS", 50_000)?).await?;
    let rows = text.len();
    if rows < 100 {
        return Err("need at least 100 rows".into());
    }
    let repeats = setting("FTS_BENCH_REPEATS", 3)?;
    let dir = tempfile::tempdir()?;
    let params = WriteParams {
        max_rows_per_file: 5_000,
        ..Default::default()
    };
    let mut dataset = Dataset::write(
        reader(batch(&text, 0, false)),
        dir.path().to_str().unwrap(),
        Some(params.clone()),
    )
    .await?;
    let with_positions = env::var("FTS_BENCH_POSITIONS")
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(false);
    let index_params = InvertedIndexParams::code()
        .split_identifiers(true)
        .preserve_original(true)
        .language("Russian")?
        .stem(true)
        .with_position(with_positions);
    dataset
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("text_idx".into()),
            &index_params,
            false,
        )
        .await?;
    let seed = dataset.version().version;
    let deleted = rows.div_ceil(4);
    let added = rows / 4;
    let replacement = batch(&text[..added], rows, true);
    println!(
        "positions={with_positions} rows={rows} deleted={deleted} added={added} text_bytes={} repeats={repeats}",
        text.iter().map(String::len).sum::<usize>()
    );
    for compact in [false, true] {
        for repeat in 0..repeats {
            dataset = dataset.checkout_version(seed).await?;
            dataset.restore().await?;
            let total = Instant::now();
            dataset.delete("id % 4 = 0").await?;
            dataset
                .append(reader(replacement.clone()), Some(params.clone()))
                .await?;
            let started = Instant::now();
            if compact {
                compact_files(&mut dataset, Default::default(), None).await?;
            }
            let compact_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            dataset
                .optimize_indices(&OptimizeOptions::default())
                .await?;
            let index_ms = started.elapsed().as_secs_f64() * 1000.0;
            let total_ms = total.elapsed().as_secs_f64() * 1000.0;
            check(&dataset, "sharedmarker", rows - deleted + added).await?;
            check(&dataset, "deletedmarker", 0).await?;
            check(&dataset, "freshmarker", added).await?;
            check(&dataset, "keptmarker", rows - deleted).await?;
            println!(
                "compact={compact} run={} compact_ms={compact_ms:.3} index_ms={index_ms:.3} total_ms={total_ms:.3} verified=true",
                repeat + 1
            );
        }
    }
    Ok(())
}
