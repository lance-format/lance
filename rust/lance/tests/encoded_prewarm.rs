// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance::{Dataset, dataset::WriteParams};
use lance_core::Error;
use lance_datagen::{BatchCount, RowCount, array, gen_batch};
use lance_file::writer::FileWriterOptions;
use roaring::RoaringBitmap;

#[rstest::rstest]
#[case::single_fragment(65536)]
#[case::multiple_fragments(32768)]
#[tokio::test]
async fn encoded_prewarm_real_file_narrows_projection_and_physical_pages(
    #[case] max_rows_per_file: usize,
) {
    let batch = gen_batch()
        .col("id", array::step::<arrow_array::types::Int64Type>())
        .col("payload", array::step::<arrow_array::types::Int64Type>())
        .into_reader_rows(RowCount::from(1024), BatchCount::from(64));
    let dataset = Dataset::write(
        batch,
        "memory://",
        Some(WriteParams {
            max_rows_per_file,
            file_writer_options: Some(FileWriterOptions {
                data_cache_bytes: Some(4096),
                max_page_bytes: Some(4096),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    let fragments = dataset.get_fragments();
    assert_eq!(fragments.len(), 65536 / max_rows_per_file);
    let projection = dataset.schema().project(&["payload"]).unwrap();
    for fragment in &fragments {
        let all = fragment.plan_data_prewarm(&projection, None).await.unwrap();
        let selected = fragment
            .plan_data_prewarm(&projection, Some(&RoaringBitmap::from_iter([0])))
            .await
            .unwrap();
        assert!(all.num_pages() > 1);
        assert!(selected.num_pages() < all.num_pages());
        assert!(selected.num_bytes() < all.num_bytes());
        assert!(selected.fallback_reasons().next().is_none());
        let selected = std::sync::Arc::new(selected);
        let background = selected.clone();
        tokio::spawn(async move { background.execute(2).await })
            .await
            .unwrap()
            .unwrap();
        let empty = fragment
            .plan_data_prewarm(&projection, Some(&RoaringBitmap::new()))
            .await
            .unwrap();
        assert_eq!(empty.num_bytes(), 0);
        assert!(matches!(
            selected.execute(0).await.unwrap_err(),
            Error::InvalidInput { .. }
        ));
    }
}
