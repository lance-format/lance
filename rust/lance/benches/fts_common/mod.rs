// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StringArray};
use lance::{Dataset, dataset::WriteParams, index::DatasetIndexExt};
use lance_index::{IndexType, scalar::inverted::tokenizer::InvertedIndexParams};
use std::sync::Arc;
use tempfile::TempDir;

const INDEX_NAME: &str = "segmented_fts";
pub const INDEXED_FRAGMENT_COUNT: usize = 12;
const UNINDEXED_FRAGMENT_COUNT: usize = 1;
pub const ROWS_PER_FRAGMENT: usize = 64;

pub struct BenchDataset {
    pub _data_dir: TempDir,
    pub dataset: Dataset,
}

fn create_fragment_batch(fragment_id: usize) -> RecordBatch {
    let start = (fragment_id * ROWS_PER_FRAGMENT) as i32;
    let ids = Arc::new(Int32Array::from_iter_values(
        start..start + ROWS_PER_FRAGMENT as i32,
    ));
    let texts = Arc::new(StringArray::from_iter_values((0..ROWS_PER_FRAGMENT).map(
        |row| {
            let term = match (fragment_id + row) % 4 {
                0 => "alpha",
                1 => "beta",
                2 => "gamma",
                _ => "delta",
            };
            format!("shared {term} fragment-{fragment_id} row-{row}")
        },
    )));
    RecordBatch::try_from_iter(vec![("id", ids as ArrayRef), ("text", texts as ArrayRef)]).unwrap()
}

pub fn grouped_fragment_ids(total_fragments: usize, segment_count: usize) -> Vec<Vec<u32>> {
    let fragments_per_segment = total_fragments / segment_count;
    (0..segment_count)
        .map(|segment_idx| {
            let start = segment_idx * fragments_per_segment;
            let end = start + fragments_per_segment;
            (start..end).map(|fragment_id| fragment_id as u32).collect()
        })
        .collect()
}

pub async fn build_segmented_fts_dataset(segment_count: usize) -> BenchDataset {
    let tmpdir = TempDir::new().unwrap();
    let uri = format!("file://{}", tmpdir.path().display());
    let batches = RecordBatchIterator::new(
        (0..(INDEXED_FRAGMENT_COUNT + UNINDEXED_FRAGMENT_COUNT))
            .map(|fragment_id| Ok(create_fragment_batch(fragment_id)))
            .collect::<Vec<_>>(),
        create_fragment_batch(0).schema(),
    );
    let mut dataset = Dataset::write(
        batches,
        &uri,
        Some(WriteParams {
            max_rows_per_file: ROWS_PER_FRAGMENT,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    assert_eq!(
        dataset.get_fragments().len(),
        INDEXED_FRAGMENT_COUNT + UNINDEXED_FRAGMENT_COUNT
    );

    let params = InvertedIndexParams::default();
    let mut staged_segments = Vec::with_capacity(segment_count);
    for fragment_ids in grouped_fragment_ids(INDEXED_FRAGMENT_COUNT, segment_count) {
        let segment = dataset
            .create_index_builder(&["text"], IndexType::Inverted, &params)
            .name(INDEX_NAME.to_string())
            .fragments(fragment_ids)
            .execute_uncommitted()
            .await
            .unwrap();
        staged_segments.push(segment);
    }
    dataset
        .commit_existing_index_segments(INDEX_NAME, "text", staged_segments)
        .await
        .unwrap();

    BenchDataset {
        _data_dir: tmpdir,
        dataset,
    }
}
