// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{
    Dataset,
    blob::{BlobArrayBuilder, BlobFile, blob_field},
    dataset::WriteParams,
};
use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use lance_core::utils::tempfile::TempStrDir;
use lance_file::version::LanceFileVersion;
use rstest::rstest;
use std::sync::Arc;

#[tokio::test]
async fn default_sequential_buffer_uses_512_kib_windows() {
    let payload = vec![0xABu8; 1024 * 1024];
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();

    let got = read_up_to_chunks(blob, 8192).await;
    assert_eq!(got.as_ref(), payload.as_slice());
    assert_eq!(blob.range_submission_count(), 2);
}

#[tokio::test]
async fn sequential_small_reads_reuse_prefetch() {
    let payload = vec![0xABu8; 40 * 1024];
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    blob.set_buffer_size(32 * 1024).await.unwrap();

    let got = read_up_to_chunks(blob, 8192).await;
    assert_eq!(got.as_ref(), payload.as_slice());
    assert_eq!(blob.range_submission_count(), 2);
}

#[rstest]
#[case::window_covers_the_rest(4 * 1024 * 1024, 0, 1)]
#[case::small_tail_is_appended(64 * 1024, 32 * 1024, 2)]
#[case::large_tail_is_read_from_cursor(16 * 1024, 95 * 1024, 2)]
#[tokio::test]
async fn read_after_read_up_to_reuses_prefetch(
    #[case] buffer_size: usize,
    #[case] fetched: u64,
    #[case] submissions: usize,
) {
    let payload: Vec<u8> = (0..96 * 1024).map(|i| (i % 251) as u8).collect();
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    blob.set_buffer_size(buffer_size).await.unwrap();

    blob.read_up_to(1024).await.unwrap();
    let _ = dataset.table_state.object_store.io_stats_incremental();
    let rest = blob.read().await.unwrap();
    assert_eq!(rest.as_ref(), &payload[1024..]);
    assert_eq!(
        dataset
            .table_state
            .object_store
            .io_stats_incremental()
            .read_bytes,
        fetched
    );
    assert_eq!(blob.range_submission_count(), submissions);
    assert_eq!(blob.tell().await.unwrap(), payload.len() as u64);
}

#[tokio::test]
async fn zero_buffer_size_fetches_each_sequential_read() {
    let payload = vec![0xCDu8; 16 * 1024];
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    blob.set_buffer_size(0).await.unwrap();

    let got = read_up_to_chunks(blob, 8192).await;
    assert_eq!(got.as_ref(), payload.as_slice());
    assert_eq!(blob.range_submission_count(), 2);
}

#[tokio::test]
async fn read_range_does_not_use_or_advance_prefetch() {
    let payload: Vec<u8> = (0..32).collect();
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    blob.set_buffer_size(16).await.unwrap();

    let first = blob.read_up_to(4).await.unwrap();
    assert_eq!(first.as_ref(), &payload[..4]);
    let after_fill = blob.range_submission_count();
    let ranged = blob.read_range(10..14).await.unwrap();
    assert_eq!(ranged.as_ref(), &payload[10..14]);
    assert_eq!(blob.tell().await.unwrap(), 4);
    let second = blob.read_up_to(4).await.unwrap();
    assert_eq!(second.as_ref(), &payload[4..8]);
    assert_eq!(blob.range_submission_count(), after_fill + 1);
}

#[tokio::test]
async fn seek_inside_prefetch_does_not_refetch() {
    let payload: Vec<u8> = (0..40).map(|i| i as u8).collect();
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    blob.set_buffer_size(16).await.unwrap();

    let _ = blob.read_up_to(4).await.unwrap();
    let after_fill = blob.range_submission_count();
    blob.seek(8).await.unwrap();
    let jumped = blob.read_up_to(4).await.unwrap();
    assert_eq!(jumped.as_ref(), &payload[8..12]);
    assert_eq!(blob.range_submission_count(), after_fill);
}

#[tokio::test]
async fn seek_outside_prefetch_drops_it() {
    let payload: Vec<u8> = (0..40).map(|i| i as u8).collect();
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    blob.set_buffer_size(16).await.unwrap();

    let _ = blob.read_up_to(4).await.unwrap();
    let after_fill = blob.range_submission_count();
    blob.seek(20).await.unwrap();
    let jumped = blob.read_up_to(4).await.unwrap();
    assert_eq!(jumped.as_ref(), &payload[20..24]);
    assert_eq!(blob.range_submission_count(), after_fill + 1);
}

#[tokio::test]
async fn read_past_eof_leaves_the_cursor() {
    let payload = b"abcdef";
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    let past_eof = blob.size() + 1;

    blob.seek(past_eof).await.unwrap();
    assert!(blob.read().await.unwrap().is_empty());
    assert_eq!(blob.tell().await.unwrap(), past_eof);
}

#[tokio::test]
async fn changing_buffer_size_drops_unused_prefetch() {
    let payload: Vec<u8> = (0..40).map(|i| i as u8).collect();
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    blob.set_buffer_size(16).await.unwrap();

    let _ = blob.read_up_to(4).await.unwrap();
    let after_fill = blob.range_submission_count();
    blob.set_buffer_size(8).await.unwrap();
    let next = blob.read_up_to(4).await.unwrap();
    assert_eq!(next.as_ref(), &payload[4..8]);
    assert_eq!(blob.range_submission_count(), after_fill + 1);
}

#[tokio::test]
async fn read_up_to_fills_across_prefetch_boundary() {
    let payload: Vec<u8> = (0..40).map(|i| i as u8).collect();
    let (_dir, dataset) = write_blob_v2_dataset(&[payload.as_slice()]).await;
    let blobs = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    blob.set_buffer_size(16).await.unwrap();

    let first = blob.read_up_to(4).await.unwrap();
    assert_eq!(first.as_ref(), &payload[..4]);
    let after_fill = blob.range_submission_count();
    let second = blob.read_up_to(20).await.unwrap();
    assert_eq!(second.as_ref(), &payload[4..24]);
    assert_eq!(blob.tell().await.unwrap(), 24);
    assert_eq!(blob.range_submission_count(), after_fill + 1);
}

async fn write_blob_v2_dataset(payloads: &[&[u8]]) -> (TempStrDir, Arc<Dataset>) {
    let test_dir = TempStrDir::default();
    let mut blob_builder = BlobArrayBuilder::new(payloads.len());
    for payload in payloads {
        blob_builder.push_bytes(*payload).unwrap();
    }
    let blob_array: ArrayRef = blob_builder.finish().unwrap();
    let ids: ArrayRef = Arc::new(UInt32Array::from(
        (0..payloads.len() as u32).collect::<Vec<_>>(),
    ));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        blob_field("blob", true),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![ids, blob_array]).unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)], schema),
            &test_dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    (test_dir, dataset)
}

async fn read_up_to_chunks(blob: &BlobFile, chunk: usize) -> Bytes {
    let mut out = Vec::new();
    loop {
        let piece = blob.read_up_to(chunk).await.unwrap();
        if piece.is_empty() {
            break;
        }
        out.extend_from_slice(&piece);
    }
    Bytes::from(out)
}
