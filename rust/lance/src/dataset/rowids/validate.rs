// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod tests {
    use super::super::load_row_id_sequence;
    use crate::{Dataset, Error};
    use lance_table::format::RowDatasetVersionMeta;
    use std::sync::Arc;

    // Shared with the row id tests next door, which cover the same operations.
    use super::super::test::{compact, delete};

    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::{
        MergeInsertBuilder, UpdateBuilder, WhenMatched, WhenNotMatched, WriteMode, WriteParams,
    };
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::RecordBatchIterator;
    use arrow_array::types::Int32Type;
    use arrow_schema::Schema as ArrowSchema;
    use lance_table::format::{Fragment, RowIdMeta, pb};
    use lance_table::rowids::version::RowDatasetVersionSequence;
    use lance_table::rowids::{RowIdSequence, write_row_ids};
    use prost::Message;
    use roaring::RoaringTreemap;
    use rstest::rstest;

    fn encode_segments(segments: Vec<pb::u64_segment::Segment>) -> Vec<u8> {
        pb::RowIdSequence {
            segments: segments
                .into_iter()
                .map(|segment| pb::U64Segment {
                    segment: Some(segment),
                })
                .collect(),
        }
        .encode_to_vec()
    }

    fn range_segment(start: u64, end: u64) -> pb::u64_segment::Segment {
        pb::u64_segment::Segment::Range(pb::u64_segment::Range { start, end })
    }

    fn empty_encoded_array() -> pb::EncodedU64Array {
        pb::EncodedU64Array {
            array: Some(pb::encoded_u64_array::Array::U64Array(
                pb::encoded_u64_array::U64Array { values: Vec::new() },
            )),
        }
    }

    fn empty_array_segment() -> pb::u64_segment::Segment {
        pb::u64_segment::Segment::Array(empty_encoded_array())
    }

    fn empty_sorted_array_segment() -> pb::u64_segment::Segment {
        pb::u64_segment::Segment::SortedArray(empty_encoded_array())
    }

    fn empty_range_with_holes_segment() -> pb::u64_segment::Segment {
        pb::u64_segment::Segment::RangeWithHoles(pb::u64_segment::RangeWithHoles {
            start: 0,
            end: 0,
            holes: Some(empty_encoded_array()),
        })
    }

    fn empty_range_with_bitmap_segment() -> pb::u64_segment::Segment {
        pb::u64_segment::Segment::RangeWithBitmap(pb::u64_segment::RangeWithBitmap {
            start: 0,
            end: 0,
            bitmap: Vec::new(),
        })
    }

    /// Two `Range` segments that each fit a `usize` but whose lengths sum past `u64`.
    /// Decoding accepts them, so `validate()` must report corruption rather than
    /// overflowing while measuring the sequence.
    #[tokio::test]
    async fn test_validate_rejects_row_id_count_overflow() {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = validation_fixture(&temp_dir).await;
        let encoded = encode_segments(vec![range_segment(0, u64::MAX), range_segment(0, u64::MAX)]);
        edit_fragments(&mut dataset, |fragments| {
            fragments[1].row_id_meta = Some(RowIdMeta::Inline(encoded.into()));
        });

        assert_invalid(&dataset, "total length exceeding u64::MAX").await;
    }

    /// An empty segment carries no minimum or maximum, and decoding accepts an empty
    /// encoding of every variant. Neither the cardinality check nor the `next_row_id`
    /// bound may unwind on one.
    #[rstest]
    #[case::array(empty_array_segment())]
    #[case::sorted_array(empty_sorted_array_segment())]
    #[case::range_with_holes(empty_range_with_holes_segment())]
    #[case::range_with_bitmap(empty_range_with_bitmap_segment())]
    #[tokio::test]
    async fn test_validate_tolerates_empty_segment(#[case] empty: pb::u64_segment::Segment) {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = validation_fixture(&temp_dir).await;
        // Ten ids across two segments matches the fragment's ten physical rows, and the
        // ids stay clear of fragment 0's, so this dataset is well-formed.
        let encoded = encode_segments(vec![range_segment(10, 20), empty]);
        edit_fragments(&mut dataset, |fragments| {
            fragments[1].row_id_meta = Some(RowIdMeta::Inline(encoded.into()));
        });

        dataset.validate().await.unwrap();
    }

    /// Write a two-fragment dataset with stable row ids to `uri`, then reopen it with a
    /// fresh session so the row id sequence cache is empty. Tests that doctor a
    /// fragment's `row_id_meta` need that: the cache is keyed by fragment id alone, so a
    /// sequence loaded before the mutation would shadow the doctored one.
    async fn validation_fixture(uri: &str) -> Dataset {
        lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_dataset_with_params(
                uri,
                FragmentCount::from(2),
                FragmentRowCount::from(10),
                Some(WriteParams {
                    max_rows_per_file: 10,
                    enable_stable_row_ids: true,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        DatasetBuilder::from_uri(uri).load().await.unwrap()
    }

    fn edit_fragments(dataset: &mut Dataset, edit: impl FnOnce(&mut Vec<Fragment>)) {
        let mut manifest = dataset.manifest.as_ref().clone();
        let mut fragments = manifest.fragments.as_ref().clone();
        edit(&mut fragments);
        manifest.fragments = Arc::new(fragments);
        dataset.manifest = Arc::new(manifest);
    }

    async fn assert_invalid(dataset: &Dataset, expected_message: &str) {
        let err = dataset.validate().await.unwrap_err();
        assert!(
            matches!(err, Error::CorruptFile { .. }),
            "expected a corrupt file error, got {err:?}"
        );
        let message = err.to_string();
        assert!(
            message.contains(expected_message),
            "expected {expected_message:?} in {message:?}"
        );
    }

    #[tokio::test]
    async fn test_validate_rejects_missing_row_id_meta() {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = validation_fixture(&temp_dir).await;
        edit_fragments(&mut dataset, |fragments| fragments[1].row_id_meta = None);

        assert_invalid(&dataset, "Fragment 1 has no row id metadata").await;
    }

    #[tokio::test]
    async fn test_validate_rejects_row_id_sequence_length_mismatch() {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = validation_fixture(&temp_dir).await;
        edit_fragments(&mut dataset, |fragments| {
            let short = RowIdSequence::from(&[100u64, 101, 102][..]);
            fragments[1].row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&short).into()));
        });

        assert_invalid(&dataset, "Fragment 1 has 3 row ids, but 10 physical rows").await;
    }

    #[tokio::test]
    async fn test_validate_rejects_duplicate_live_row_ids() {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = validation_fixture(&temp_dir).await;
        edit_fragments(&mut dataset, |fragments| {
            fragments[1].row_id_meta = fragments[0].row_id_meta.clone();
        });

        assert_invalid(
            &dataset,
            "Row id 0 is live in both fragment 0 at offset 0 and fragment 1 at offset 0",
        )
        .await;
    }

    #[tokio::test]
    async fn test_validate_rejects_reused_next_row_id() {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = validation_fixture(&temp_dir).await;
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.next_row_id = 19;
        dataset.manifest = Arc::new(manifest);

        assert_invalid(
            &dataset,
            "will hand out row id 19 next, but row id 19 is already in use",
        )
        .await;
    }

    #[tokio::test]
    async fn test_validate_rejects_misaligned_version_sequence() {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let mut dataset = validation_fixture(&temp_dir).await;
        edit_fragments(&mut dataset, |fragments| {
            let too_long = RowDatasetVersionSequence::from_uniform_row_count(11, 1);
            fragments[1].created_at_version_meta =
                Some(RowDatasetVersionMeta::from_sequence(&too_long).unwrap());
        });

        assert_invalid(
            &dataset,
            "Fragment 1 has 11 created_at versions, but 10 physical rows",
        )
        .await;
    }

    /// Number of sequence entries whose row id already appeared in an earlier fragment,
    /// tombstoned slots included.
    async fn count_repeated_row_ids(dataset: &Dataset) -> usize {
        let mut seen = RoaringTreemap::new();
        let mut repeated = 0;
        for fragment in dataset.get_fragments() {
            let sequence = load_row_id_sequence(dataset, fragment.metadata())
                .await
                .unwrap();
            repeated += sequence.iter().filter(|id| !seen.insert(*id)).count();
        }
        repeated
    }

    /// A tombstoned slot keeps its row id, so after an update the same id lives in the
    /// rewritten fragment and lingers, deleted, in the original. Only live ids must be
    /// unique — this asserts the checks agree with that.
    #[tokio::test]
    async fn test_validate_across_write_operations() {
        let temp_dir = lance_core::utils::tempfile::TempStrDir::default();
        let dataset = validation_fixture(&temp_dir).await;
        dataset.validate().await.unwrap();

        let batch = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step_custom::<Int32Type>(20, 1))
            .into_batch_rows(lance_datagen::RowCount::from(10))
            .unwrap();
        let arrow_schema = Arc::new(ArrowSchema::from(dataset.schema()));
        let reader = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema);
        let mut dataset = Dataset::write(
            reader,
            &temp_dir,
            Some(WriteParams {
                mode: WriteMode::Append,
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.validate().await.unwrap();

        delete(&mut dataset, "i = 4 or i = 12").await;
        dataset.validate().await.unwrap();

        let dataset = UpdateBuilder::new(Arc::new(dataset))
            .update_where("i >= 15")
            .unwrap()
            .set("i", "i + 1000")
            .unwrap()
            .build()
            .unwrap()
            .execute()
            .await
            .unwrap()
            .new_dataset;
        dataset.validate().await.unwrap();
        assert!(
            count_repeated_row_ids(&dataset).await > 0,
            "update should leave rewritten row ids behind in tombstoned slots"
        );

        let merge_source = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step_custom::<Int32Type>(8, 1))
            .into_batch_rows(lance_datagen::RowCount::from(6))
            .unwrap();
        let schema = merge_source.schema();
        let merge_job = MergeInsertBuilder::try_new(dataset.clone(), vec!["i".to_string()])
            .unwrap()
            .when_matched(WhenMatched::UpdateAll)
            .when_not_matched(WhenNotMatched::InsertAll)
            .try_build()
            .unwrap();
        let source = lance_datafusion::utils::reader_to_stream(Box::new(RecordBatchIterator::new(
            [Ok(merge_source)],
            schema,
        )));
        let (dataset, _stats) = merge_job.execute(source).await.unwrap();
        dataset.validate().await.unwrap();

        let mut dataset = dataset.as_ref().clone();
        compact(&mut dataset, 20).await;
        dataset.validate().await.unwrap();
    }
}
