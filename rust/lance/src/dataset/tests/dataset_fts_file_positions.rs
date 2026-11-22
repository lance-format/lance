// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;
use std::sync::atomic::Ordering;

use arrow_array::builder::{ListBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Int32Type, UInt32Type};
use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator};
use lance_core::utils::tempfile::TempStrDir;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_index::scalar::inverted::query::PhraseQuery;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_index::{FtsPrewarmOptions, IndexType, PrewarmOptions};

use crate::Dataset;
use crate::dataset::builder::DatasetBuilder;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::{UpdateBuilder, WriteParams};
use crate::index::DatasetIndexExt;
use crate::session::Session;
use crate::utils::test::serializing_cache::{FileReadState, SerializingCacheBackend};

async fn warm(
    path: &str,
    batch: RecordBatch,
    granularity: DocumentGranularity,
) -> (Dataset, Dataset, Arc<FileReadState>) {
    let mut baseline = Dataset::write(
        RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
        path,
        Some(WriteParams {
            max_rows_per_file: 2,
            max_rows_per_group: 2,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    baseline
        .create_index(
            &["text"],
            IndexType::Inverted,
            Some("fts".into()),
            &InvertedIndexParams::default()
                .with_position(true)
                .document_granularity(granularity),
            true,
        )
        .await
        .unwrap();

    let reads = Arc::new(FileReadState::default());
    let backend = SerializingCacheBackend::new().with_file_reader(reads.clone());
    let session =
        Session::with_index_cache_backend(Arc::new(backend), 32 << 20, Default::default());
    let dataset = DatasetBuilder::from_uri(path)
        .with_session(Arc::new(session))
        .load()
        .await
        .unwrap();
    dataset
        .prewarm_index_with_options(
            "fts",
            &PrewarmOptions::Fts(FtsPrewarmOptions::new().with_position(true)),
        )
        .await
        .unwrap();
    (baseline, dataset, reads)
}

async fn phrase(dataset: &Dataset, granularity: DocumentGranularity) -> Vec<(i32, Vec<u32>, u32)> {
    let q = PhraseQuery::new("orange grape".into())
        .with_column(Some("text".into()))
        .with_slop(1)
        .with_document_granularity(granularity);
    let batch = dataset
        .scan()
        .filter("id < 4")
        .unwrap()
        .full_text_search(FullTextSearchQuery::new_query(q.into()))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch["id"].as_primitive::<Int32Type>();
    let scores = batch["_score"].as_primitive::<Float32Type>();
    let coords = batch
        .column_by_name("_doc_index")
        .map(|c| c.as_list::<i32>());
    let mut hits: Vec<_> = (0..batch.num_rows())
        .map(|i| {
            let coord = coords
                .map(|c| c.value(i).as_primitive::<UInt32Type>().values().to_vec())
                .unwrap_or_default();
            (ids.value(i), coord, scores.value(i).to_bits())
        })
        .collect();
    hits.sort_unstable();
    hits
}

#[tokio::test]
async fn file_positions_follow_mutations_and_index_replacement() {
    let tmp = TempStrDir::default();
    let batch = arrow_array::record_batch!(
        ("id", Int32, [0, 1, 2, 3, 4]),
        (
            "text",
            Utf8,
            [
                "orange ripe grape",
                "orange grape",
                "plum",
                "orange ripe ripe grape",
                "orange grape"
            ]
        )
    )
    .unwrap();
    let g = DocumentGranularity::Row;
    let (baseline, mut ds, reads) = warm(tmp.as_str(), batch, g).await;
    let before = reads.block_reads.load(Ordering::Relaxed);
    let expected = phrase(&baseline, g).await;
    assert_eq!(expected.iter().map(|h| h.0).collect::<Vec<_>>(), [0, 1]);
    assert_eq!(phrase(&ds, g).await, expected);
    assert!(reads.block_reads.load(Ordering::Relaxed) > before);

    reads.fail_payload.store(true, Ordering::Relaxed);
    let before_fault = reads.block_reads.load(Ordering::Relaxed);
    assert_eq!(phrase(&ds, g).await, expected);
    assert!(reads.block_reads.load(Ordering::Relaxed) > before_fault);
    reads.fail_payload.store(false, Ordering::Relaxed);

    let old_snapshot = ds.clone();
    ds.delete("id = 0").await.unwrap();
    ds = UpdateBuilder::new(Arc::new(ds))
        .update_where("id = 2")
        .unwrap()
        .set("text", "'orange grape'")
        .unwrap()
        .build()
        .unwrap()
        .execute()
        .await
        .unwrap()
        .new_dataset
        .as_ref()
        .clone();
    let assert_current_snapshot = async |snapshot: &Dataset| {
        let control = Dataset::open(tmp.as_str()).await.unwrap();
        let expected = phrase(&control, g).await;
        assert_eq!(expected.iter().map(|h| h.0).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(phrase(snapshot, g).await, expected);
    };
    assert_current_snapshot(&ds).await;
    // A new segment starts its term IDs at zero. A cache keyed only by term
    // could accidentally reuse a positions directory from the first segment.
    let delta =
        arrow_array::record_batch!(("id", Int32, [5]), ("text", Utf8, ["apricot peach"])).unwrap();
    ds.append(
        RecordBatchIterator::new([Ok(delta.clone())], delta.schema()),
        None,
    )
    .await
    .unwrap();
    ds.optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let segments = ds.load_indices_by_name("fts").await.unwrap();
    assert_eq!(segments.len(), 2);
    ds.optimize_indices(&OptimizeOptions::merge(2))
        .await
        .unwrap();
    let merged = ds.load_indices_by_name("fts").await.unwrap();
    assert_eq!(merged.len(), 1);
    assert!(segments.iter().all(|index| index.uuid != merged[0].uuid));
    ds.prewarm_index_with_options(
        "fts",
        &PrewarmOptions::Fts(FtsPrewarmOptions::new().with_position(true)),
    )
    .await
    .unwrap();
    assert_current_snapshot(&ds).await;
    let compacted = compact_files(&mut ds, CompactionOptions::default(), None)
        .await
        .unwrap();
    assert!(compacted.fragments_removed > 0);
    assert_current_snapshot(&ds).await;
    assert_eq!(phrase(&old_snapshot, g).await, phrase(&baseline, g).await);
}

#[tokio::test]
async fn file_positions_preserve_list_element_coordinates() {
    let mut lists = ListBuilder::new(StringBuilder::new());
    for words in [
        vec!["orange", "grape", "orange ripe grape"],
        vec!["grape", "orange grape"],
        vec!["orange ripe ripe grape"],
        vec![],
        vec!["orange grape"],
    ] {
        for word in words {
            lists.values().append_value(word);
        }
        lists.append(true);
    }
    let batch = RecordBatch::try_from_iter([
        (
            "id",
            Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4])) as ArrayRef,
        ),
        ("text", Arc::new(lists.finish()) as ArrayRef),
    ])
    .unwrap();
    let tmp = TempStrDir::default();
    let g = DocumentGranularity::ListElement;
    let (control, ds, reads) = warm(tmp.as_str(), batch, g).await;
    let before = reads.block_reads.load(Ordering::Relaxed);
    let hits = phrase(&ds, g).await;
    assert_eq!(
        hits.iter()
            .map(|(id, coord, _)| (*id, coord.as_slice()))
            .collect::<Vec<_>>(),
        [(0, &[2][..]), (1, &[1][..])]
    );
    assert_eq!(hits, phrase(&control, g).await);
    assert!(reads.block_reads.load(Ordering::Relaxed) > before);
}
