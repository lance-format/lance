// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub use lance_table_runtime::overlay::{OverlayWriter, WriteOverlayError};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dataset;
    use arrow_array::{Array, RecordBatch, cast::AsArray};
    use arrow_schema::{DataType, Schema as ArrowSchema};
    use lance_core::{Error, ROW_ADDR, datatypes::Schema, utils::address::RowAddress};
    use lance_table::format::overlay::OverlayCoverage;
    use lance_table::transaction::DataOverlayGroup;
    use roaring::RoaringBitmap;
    use std::sync::Arc;

    use arrow_array::types::Int32Type;
    use arrow_array::{
        ArrayRef, Int32Array, RecordBatchIterator, StructArray, UInt64Array, record_batch,
    };
    use arrow_buffer::NullBuffer;
    use arrow_schema::Field as ArrowField;
    use arrow_schema::Fields;
    use lance_file::version::LanceFileVersion;
    use rstest::rstest;
    use tempfile::{TempDir, tempdir};

    use crate::dataset::transaction::Operation;
    use crate::dataset::{DATA_DIR, WriteDestination, WriteParams};

    /// Two six-row fragments of `id` (field 0) = 0..12, `val` (field 1) = id * 10,
    /// and `tag` (field 2) = id * 100. Backed by a temp dir so tests can count the
    /// files left in `data/`.
    async fn test_dataset() -> (TempDir, Arc<Dataset>) {
        test_dataset_with(None).await
    }

    async fn test_dataset_with(
        data_storage_version: Option<LanceFileVersion>,
    ) -> (TempDir, Arc<Dataset>) {
        let test_dir = tempdir().unwrap();
        let uri = test_dir.path().to_str().unwrap().to_string();
        let batch = record_batch!(
            ("id", Int32, (0..12).collect::<Vec<_>>()),
            ("val", Int32, (0..12).map(|v| v * 10).collect::<Vec<_>>()),
            ("tag", Int32, (0..12).map(|v| v * 100).collect::<Vec<_>>())
        )
        .unwrap();
        let schema = batch.schema();
        let params = WriteParams {
            max_rows_per_file: 6,
            max_rows_per_group: 6,
            data_storage_version,
            ..Default::default()
        };
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = Dataset::write(reader, &uri, Some(params)).await.unwrap();
        (test_dir, Arc::new(dataset))
    }

    /// The overlay schema for `val` alone, or for `val` and `tag`.
    fn overlay_schema(dataset: &Dataset, ids: &[i32]) -> Schema {
        dataset.schema().project_by_ids(ids, true)
    }

    fn addrs(fragment_id: u32, offsets: &[u32]) -> Vec<u64> {
        offsets
            .iter()
            .map(|&offset| u64::from(RowAddress::new_from_parts(fragment_id, offset)))
            .collect()
    }

    fn int32(values: Vec<Option<i32>>) -> ArrayRef {
        Arc::new(Int32Array::from(values))
    }

    /// `_rowaddr` for `offsets` of `fragment_id`, alongside the given columns.
    fn batch(fragment_id: u32, offsets: &[u32], columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let addrs: ArrayRef = Arc::new(UInt64Array::from(addrs(fragment_id, offsets)));
        let columns = std::iter::once((ROW_ADDR, addrs, true))
            .chain(columns.into_iter().map(|(name, array)| (name, array, true)));
        RecordBatch::try_from_iter_with_nullable(columns).unwrap()
    }

    fn val_batch(offsets: &[u32], values: Vec<Option<i32>>) -> RecordBatch {
        batch(0, offsets, vec![("val", int32(values))])
    }

    fn tag_batch(offsets: &[u32], values: Vec<Option<i32>>) -> RecordBatch {
        batch(0, offsets, vec![("tag", int32(values))])
    }

    async fn open_writer(dataset: &Arc<Dataset>, ids: &[i32]) -> OverlayWriter {
        open_writer_on(dataset, 0, ids).await
    }

    async fn open_writer_on(
        dataset: &Arc<Dataset>,
        fragment_id: usize,
        ids: &[i32],
    ) -> OverlayWriter {
        let schema = overlay_schema(dataset, ids);
        dataset
            .get_fragment(fragment_id)
            .unwrap()
            .write_overlay(&schema)
            .await
            .unwrap()
    }

    async fn commit(dataset: Arc<Dataset>, group: DataOverlayGroup) -> Dataset {
        commit_groups(dataset, vec![group]).await
    }

    async fn commit_groups(dataset: Arc<Dataset>, groups: Vec<DataOverlayGroup>) -> Dataset {
        let read_version = dataset.version().version;
        Dataset::commit(
            WriteDestination::Dataset(dataset),
            Operation::DataOverlay { groups },
            Some(read_version),
            None,
            None,
            Arc::new(Default::default()),
            false,
        )
        .await
        .unwrap()
    }

    /// `id -> (val, tag)`, nulls included, ordered by `id`.
    async fn scanned(dataset: &Dataset) -> Vec<(i32, Option<i32>, Option<i32>)> {
        let batch = dataset
            .scan()
            .project(&["id", "val", "tag"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let column = |name: &str| batch[name].as_primitive::<Int32Type>().clone();
        let (ids, vals, tags) = (column("id"), column("val"), column("tag"));
        let mut rows = (0..batch.num_rows())
            .map(|i| {
                (
                    ids.value(i),
                    vals.is_valid(i).then(|| vals.value(i)),
                    tags.is_valid(i).then(|| tags.value(i)),
                )
            })
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| row.0);
        rows
    }

    /// Every entry in `data/`, in-progress temp files included: on local storage
    /// a file being written lives there under a temporary name until it is
    /// finished, so this catches a leak whether or not the writer got as far as
    /// publishing the `.lance` path.
    fn staged_file_count(test_dir: &TempDir) -> usize {
        std::fs::read_dir(test_dir.path().join(DATA_DIR))
            .unwrap()
            .count()
    }

    fn shared_coverage(group: &DataOverlayGroup) -> &RoaringBitmap {
        match &group.overlays[0].coverage {
            OverlayCoverage::Shared(bitmap) => bitmap,
            other => panic!("expected shared coverage, got {other:?}"),
        }
    }

    /// The coverage the read path will resolve for `field_id`, looked up the way
    /// `plan_overlays` looks it up: by the field's position in `data_file.fields`.
    fn coverage_of(group: &DataOverlayGroup, field_id: i32) -> RoaringBitmap {
        let overlay = &group.overlays[0];
        let position = overlay
            .data_file
            .fields
            .iter()
            .position(|&id| id == field_id)
            .unwrap_or_else(|| {
                panic!(
                    "field {field_id} is not in the overlay's fields {:?}",
                    overlay.data_file.fields
                )
            });
        (*overlay.coverage_for_field(position).unwrap()).clone()
    }

    fn field_id(dataset: &Dataset, name: &str) -> i32 {
        dataset.schema().field(name).unwrap().id
    }

    /// Every V2 file version the dataset can be written at. 2.2 and 2.3 route
    /// through the blob writer, which refuses `write_column` when it has a
    /// sidecar to prepare; a blob-free schema must not get one, or no overlay
    /// can be written at those versions at all.
    #[rstest]
    #[case::v2_0(LanceFileVersion::V2_0)]
    #[case::v2_1(LanceFileVersion::V2_1)]
    #[case::v2_2(LanceFileVersion::V2_2)]
    #[case::v2_3(LanceFileVersion::V2_3)]
    #[tokio::test]
    async fn test_overlay_overrides_covered_cells(#[case] version: LanceFileVersion) {
        let (_test_dir, dataset) = test_dataset_with(Some(version)).await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[1, 3], vec![Some(-1), Some(-3)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        assert_eq!(group.fragment_id, 0);
        assert_eq!(*shared_coverage(&group), RoaringBitmap::from_iter([1, 3]));

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[1], (1, Some(-1), Some(100)));
        assert_eq!(rows[3], (3, Some(-3), Some(300)));
        // Uncovered rows keep their base values, in this fragment and the next.
        assert_eq!(rows[2], (2, Some(20), Some(200)));
        assert_eq!(rows[7], (7, Some(70), Some(700)));
    }

    #[tokio::test]
    async fn test_covering_a_cell_with_null_overrides_it() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[2], vec![None]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        let dataset = commit(dataset, group).await;
        assert_eq!(scanned(&dataset).await[2], (2, None, Some(200)));
    }

    #[tokio::test]
    async fn test_batches_accumulate_coverage() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[0, 1], vec![Some(-1), Some(-2)]))
            .await
            .unwrap();
        writer
            .write_batch(&val_batch(&[4], vec![Some(-5)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();
        assert_eq!(
            *shared_coverage(&group),
            RoaringBitmap::from_iter([0, 1, 4])
        );

        let dataset = commit(dataset, group).await;
        let vals = scanned(&dataset)
            .await
            .into_iter()
            .map(|(_, val, _)| val)
            .collect::<Vec<_>>();
        assert_eq!(
            vals[..6],
            [Some(-1), Some(-2), Some(20), Some(30), Some(-5), Some(50)]
        );
    }

    #[tokio::test]
    async fn test_fields_covered_alike_share_one_bitmap() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1, 2]).await;
        let both = batch(
            0,
            &[0, 2],
            vec![
                ("val", int32(vec![Some(-1), Some(-3)])),
                ("tag", int32(vec![Some(-10), Some(-30)])),
            ],
        );
        writer.write_batch(&both).await.unwrap();
        let group = writer.finish().await.unwrap().unwrap();
        assert_eq!(*shared_coverage(&group), RoaringBitmap::from_iter([0, 2]));

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[0], (0, Some(-1), Some(-10)));
        assert_eq!(rows[2], (2, Some(-3), Some(-30)));
        assert_eq!(rows[1], (1, Some(10), Some(100)));
    }

    #[tokio::test]
    async fn test_fields_covered_differently_get_their_own_bitmaps() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1, 2]).await;
        writer
            .write_batch(&val_batch(&[0, 1], vec![Some(-1), Some(-2)]))
            .await
            .unwrap();
        writer
            .write_batch(&tag_batch(&[5], vec![Some(-50)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        assert!(matches!(
            group.overlays[0].coverage,
            OverlayCoverage::PerField(_)
        ));
        assert_eq!(
            coverage_of(&group, field_id(&dataset, "val")),
            RoaringBitmap::from_iter([0, 1])
        );
        assert_eq!(
            coverage_of(&group, field_id(&dataset, "tag")),
            RoaringBitmap::from_iter([5])
        );

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[0], (0, Some(-1), Some(0)));
        assert_eq!(rows[1], (1, Some(-2), Some(100)));
        assert_eq!(rows[5], (5, Some(50), Some(-50)));
    }

    #[tokio::test]
    async fn test_declared_field_may_go_unwritten() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1, 2]).await;
        writer
            .write_batch(&val_batch(&[3], vec![Some(-4)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        // `tag` still gets a column in the value file, covering nothing -- the
        // shape resolution has to tolerate an empty bitmap rather than a missing
        // field.
        assert_eq!(
            coverage_of(&group, field_id(&dataset, "val")),
            RoaringBitmap::from_iter([3])
        );
        assert!(coverage_of(&group, field_id(&dataset, "tag")).is_empty());

        let dataset = commit(dataset, group).await;
        assert_eq!(scanned(&dataset).await[3], (3, Some(-4), Some(300)));
    }

    #[tokio::test]
    async fn test_nothing_staged_leaves_nothing_behind() {
        let (test_dir, dataset) = test_dataset().await;
        let before = staged_file_count(&test_dir);
        let writer = open_writer(&dataset, &[1]).await;
        assert!(writer.finish().await.unwrap().is_none());
        assert_eq!(staged_file_count(&test_dir), before);
    }

    #[tokio::test]
    async fn test_abort_leaves_nothing_behind() {
        let (test_dir, dataset) = test_dataset().await;
        let before = staged_file_count(&test_dir);
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[0], vec![Some(-1)]))
            .await
            .unwrap();
        writer.abort().await;
        assert_eq!(staged_file_count(&test_dir), before);
    }

    /// Batches an overlay must refuse. Each case writes its batches in turn and
    /// expects the last to be rejected, so a case can first set up the state --
    /// accumulated coverage, a per-field cursor -- that makes it wrong.
    #[rstest]
    #[case::no_row_addr(
        vec![record_batch!(("val", Int32, vec![Some(-1)])).unwrap()],
        |error| matches!(error, WriteOverlayError::MissingRowAddr { .. }),
        "has no '_rowaddr' column"
    )]
    #[case::row_addr_of_the_wrong_type(
        vec![record_batch!(
            (ROW_ADDR, Int32, vec![Some(0)]),
            ("val", Int32, vec![Some(-1)])
        ).unwrap()],
        |error| matches!(error, WriteOverlayError::RowAddrType { .. }),
        "has a '_rowaddr' of type Int32, expected UInt64"
    )]
    #[case::null_row_addr(
        vec![record_batch!(
            (ROW_ADDR, UInt64, vec![Some(addrs(0, &[0])[0]), None]),
            ("val", Int32, vec![Some(-1), Some(-2)])
        ).unwrap()],
        |error| matches!(error, WriteOverlayError::NullRowAddr { offset_in_batch: 1, .. }),
        "null '_rowaddr' at batch offset 1"
    )]
    #[case::row_addr_from_another_fragment(
        vec![batch(1, &[0], vec![("val", int32(vec![Some(-1)]))])],
        |error| matches!(error, WriteOverlayError::ForeignRowAddr { .. }),
        "overlay for fragment 0 received a row address for fragment 1"
    )]
    #[case::offset_past_the_end_of_the_fragment(
        vec![val_batch(&[6], vec![Some(-1)])],
        |error| matches!(
            error,
            WriteOverlayError::OffsetOutOfRange { offset_in_frag: 6, physical_rows: 6, .. }
        ),
        "received offset 6, but the fragment has 6 physical rows"
    )]
    #[case::offsets_descending_within_a_batch(
        vec![val_batch(&[3, 1], vec![Some(-1), Some(-2)])],
        |error| matches!(
            error,
            WriteOverlayError::OffsetNotAscending { previous: 3, offset_in_frag: 1, .. }
        ),
        "received offset 1 for '_rowaddr' after offset 3"
    )]
    #[case::offset_repeated_across_batches(
        vec![
            val_batch(&[1, 3], vec![Some(-1), Some(-2)]),
            val_batch(&[3], vec![Some(-3)]),
        ],
        |error| matches!(
            error,
            WriteOverlayError::OffsetNotAscending { previous: 3, offset_in_frag: 3, .. }
        ),
        "received offset 3 for 'val' after offset 3"
    )]
    #[case::undeclared_column(
        vec![tag_batch(&[0], vec![Some(-1)])],
        |error| matches!(error, WriteOverlayError::UndeclaredColumn { .. }),
        "has column 'tag', which the overlay was not opened for"
    )]
    #[case::duplicate_column(
        vec![record_batch!(
            (ROW_ADDR, UInt64, addrs(0, &[0])),
            ("val", Int32, vec![Some(-1)]),
            ("val", Int32, vec![Some(-2)])
        ).unwrap()],
        |error| matches!(error, WriteOverlayError::DuplicateColumn { .. }),
        "has column 'val' twice"
    )]
    #[case::value_of_the_wrong_type(
        vec![record_batch!(
            (ROW_ADDR, UInt64, addrs(0, &[0])),
            ("val", Utf8, vec![Some("not an int")])
        ).unwrap()],
        |error| matches!(error, WriteOverlayError::BatchSchemaMismatch { .. }),
        "val"
    )]
    #[case::duplicate_row_addr(
        vec![record_batch!(
            (ROW_ADDR, UInt64, addrs(0, &[0])),
            (ROW_ADDR, UInt64, addrs(0, &[1])),
            ("val", Int32, vec![Some(-1)])
        ).unwrap()],
        |error| matches!(error, WriteOverlayError::DuplicateColumn { .. }),
        "has column '_rowaddr' twice"
    )]
    #[tokio::test]
    async fn test_bad_batch_is_rejected(
        #[case] batches: Vec<RecordBatch>,
        #[case] expected: fn(WriteOverlayError) -> bool,
        #[case] message: &str,
    ) {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        let (last, accepted) = batches.split_last().unwrap();
        for batch in accepted {
            writer.write_batch(batch).await.unwrap();
        }

        let error = writer.write_batch(last).await.unwrap_err();
        let display = error.to_string();
        assert!(display.contains(message), "{display}");
        assert!(expected(error), "unexpected variant: {display}");
    }

    #[tokio::test]
    async fn test_undeclared_field_id_is_rejected_at_open() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut schema = overlay_schema(&dataset, &[1]);
        schema.fields[0].id = 99;
        let Err(error) = dataset
            .get_fragment(0)
            .unwrap()
            .write_overlay(&schema)
            .await
        else {
            panic!("expected opening the overlay to fail");
        };
        assert!(matches!(error, WriteOverlayError::SchemaMismatch { .. }));
        assert!(
            error
                .to_string()
                .contains("names field id 99 ('val') that the dataset schema does not define"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn test_legacy_file_format_is_rejected() {
        let (_test_dir, dataset) = test_dataset_with(Some(LanceFileVersion::Legacy)).await;
        let schema = overlay_schema(&dataset, &[1]);
        let Err(error) = dataset
            .get_fragment(0)
            .unwrap()
            .write_overlay(&schema)
            .await
        else {
            panic!("expected opening the overlay to fail");
        };
        assert!(matches!(error, WriteOverlayError::LegacyFileFormat { .. }));
        assert!(
            error
                .to_string()
                .contains("the dataset uses the legacy file format"),
            "{error}"
        );
        assert!(matches!(Error::from(error), Error::NotSupported { .. }));
    }

    #[tokio::test]
    async fn test_offsets_are_local_to_the_fragment() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer_on(&dataset, 1, &[1]).await;
        let values = batch(1, &[0, 2], vec![("val", int32(vec![Some(-1), Some(-3)]))]);
        writer.write_batch(&values).await.unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        assert_eq!(group.fragment_id, 1);
        assert_eq!(*shared_coverage(&group), RoaringBitmap::from_iter([0, 2]));

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        // Fragment 1 holds ids 6..12, so its offsets 0 and 2 are ids 6 and 8 --
        // not the dataset's rows 0 and 2, which belong to fragment 0.
        assert_eq!(rows[6], (6, Some(-1), Some(600)));
        assert_eq!(rows[8], (8, Some(-3), Some(800)));
        assert_eq!(rows[0], (0, Some(0), Some(0)));
        assert_eq!(rows[2], (2, Some(20), Some(200)));
    }

    #[tokio::test]
    async fn test_one_commit_can_overlay_several_fragments() {
        let (_test_dir, dataset) = test_dataset().await;
        // Both open at once, as a scan spanning the fragments would leave them.
        let mut first = open_writer_on(&dataset, 0, &[1]).await;
        let mut second = open_writer_on(&dataset, 1, &[1]).await;
        first
            .write_batch(&val_batch(&[1], vec![Some(-1)]))
            .await
            .unwrap();
        second
            .write_batch(&batch(1, &[1], vec![("val", int32(vec![Some(-2)]))]))
            .await
            .unwrap();
        let groups = vec![
            first.finish().await.unwrap().unwrap(),
            second.finish().await.unwrap().unwrap(),
        ];

        let dataset = commit_groups(dataset, groups).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[1], (1, Some(-1), Some(100)));
        assert_eq!(rows[7], (7, Some(-2), Some(700)));
    }

    #[tokio::test]
    async fn test_offsets_are_physical_so_deletions_do_not_shift_them() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut dataset = Arc::try_unwrap(dataset).unwrap();
        dataset.delete("id = 1").await.unwrap();
        let dataset = Arc::new(dataset);

        let mut writer = open_writer(&dataset, &[1]).await;
        // Physical offsets, so the deleted row still occupies offset 1: id 2 is
        // at offset 2, not at the offset 1 it would hold if deletions closed up.
        // Offset 1 itself stays addressable -- it is in range, and its value is
        // simply unreachable while the row is deleted.
        writer
            .write_batch(&val_batch(&[1, 2, 5], vec![Some(-2), Some(-3), Some(-6)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();
        assert_eq!(
            *shared_coverage(&group),
            RoaringBitmap::from_iter([1, 2, 5])
        );

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert!(!rows.iter().any(|row| row.0 == 1), "id 1 is deleted");
        assert_eq!(rows[1], (2, Some(-3), Some(200)));
        assert_eq!(rows[4], (5, Some(-6), Some(500)));
        assert_eq!(rows[0], (0, Some(0), Some(0)));
    }

    #[tokio::test]
    async fn test_range_is_bounded_by_physical_rows_not_live_rows() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut dataset = Arc::try_unwrap(dataset).unwrap();
        dataset.delete("id < 3").await.unwrap();
        let dataset = Arc::new(dataset);

        let mut writer = open_writer(&dataset, &[1]).await;
        // Three of six rows are deleted, but the last physical offset is still 5.
        writer
            .write_batch(&val_batch(&[5], vec![Some(-6)]))
            .await
            .unwrap();
        let error = writer
            .write_batch(&val_batch(&[6], vec![Some(-7)]))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            WriteOverlayError::OffsetOutOfRange {
                offset_in_frag: 6,
                physical_rows: 6,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn test_ascension_is_tracked_per_field() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1, 2]).await;
        writer
            .write_batch(&val_batch(&[5], vec![Some(-6)]))
            .await
            .unwrap();
        // A lower offset for a different field is legal: each field has its own
        // cursor. A single shared cursor would reject this.
        writer
            .write_batch(&tag_batch(&[0], vec![Some(-10)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[5], (5, Some(-6), Some(500)));
        assert_eq!(rows[0], (0, Some(0), Some(-10)));
    }

    #[tokio::test]
    async fn test_newest_overlay_wins() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[2], vec![Some(-1)]))
            .await
            .unwrap();
        let dataset = Arc::new(commit(dataset, writer.finish().await.unwrap().unwrap()).await);
        assert_eq!(scanned(&dataset).await[2], (2, Some(-1), Some(200)));

        // Overlaying an already-overlaid fragment stacks a second file on it.
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[2], vec![Some(-2)]))
            .await
            .unwrap();
        let dataset = commit(dataset, writer.finish().await.unwrap().unwrap()).await;
        assert_eq!(scanned(&dataset).await[2], (2, Some(-2), Some(200)));
    }

    #[tokio::test]
    async fn test_empty_batch_is_a_no_op() {
        let (_test_dir, dataset) = test_dataset().await;
        let mut writer = open_writer(&dataset, &[1]).await;
        writer
            .write_batch(&val_batch(&[3], vec![Some(-4)]))
            .await
            .unwrap();
        // Neither an empty batch nor one carrying no values disturbs the
        // coverage or the per-field cursor already accumulated.
        writer.write_batch(&val_batch(&[], vec![])).await.unwrap();
        writer.write_batch(&batch(0, &[4], vec![])).await.unwrap();
        writer
            .write_batch(&val_batch(&[5], vec![Some(-6)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();
        assert_eq!(*shared_coverage(&group), RoaringBitmap::from_iter([3, 5]));

        let dataset = commit(dataset, group).await;
        let rows = scanned(&dataset).await;
        assert_eq!(rows[3], (3, Some(-4), Some(300)));
        assert_eq!(rows[4], (4, Some(40), Some(400)));
        assert_eq!(rows[5], (5, Some(-6), Some(500)));
    }

    fn nested_array_with_validity(
        a: Vec<i32>,
        b: Vec<i32>,
        nulls: Option<NullBuffer>,
        children_nullable: bool,
    ) -> ArrayRef {
        let field = |name| Arc::new(ArrowField::new(name, DataType::Int32, children_nullable));
        Arc::new(StructArray::new(
            vec![field("a"), field("b")].into(),
            vec![
                Arc::new(Int32Array::from(a)) as ArrayRef,
                Arc::new(Int32Array::from(b)) as ArrayRef,
            ],
            nulls,
        ))
    }

    fn nested_array(a: Vec<i32>, b: Vec<i32>) -> ArrayRef {
        nested_array_with_validity(a, b, None, true)
    }

    /// One four-row fragment of `val` = 0..4 and `nested: struct<a: i32, b: i32>`
    /// with `a` = 0..4 and `b` = a * 10.
    ///
    /// `nested` is non-nullable because these tests do not exercise struct
    /// validity replacement.
    async fn nested_dataset() -> (TempDir, Arc<Dataset>) {
        nested_dataset_with_nullable_struct(false, true, None).await
    }

    async fn nested_dataset_with_nullable_struct(
        nullable: bool,
        children_nullable: bool,
        data_storage_version: Option<LanceFileVersion>,
    ) -> (TempDir, Arc<Dataset>) {
        let test_dir = tempdir().unwrap();
        let uri = test_dir.path().to_str().unwrap().to_string();
        let base = RecordBatch::try_from_iter_with_nullable(vec![
            ("val", int32((0..4).map(Some).collect()), true),
            (
                "nested",
                nested_array_with_validity(
                    (0..4).collect(),
                    (0..4).map(|v| v * 10).collect(),
                    nullable.then(|| NullBuffer::from(vec![false, true, true, true])),
                    children_nullable,
                ),
                nullable,
            ),
        ])
        .unwrap();
        let arrow_schema = base.schema();
        let reader = RecordBatchIterator::new(vec![Ok(base)], arrow_schema);
        let dataset = Arc::new(
            Dataset::write(
                reader,
                &uri,
                Some(WriteParams {
                    data_storage_version,
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        (test_dir, dataset)
    }

    /// An overlay replaces a struct's validity, filling in row 0 (null in the
    /// base) and nulling row 1. With required children, the null cell has to
    /// mask the merged child values before the struct is rebuilt, or arrow
    /// rejects it ("Found unmasked nulls for non-nullable StructArray field").
    #[rstest]
    #[tokio::test]
    async fn test_nullable_struct_validity_is_replaced(
        #[values(LanceFileVersion::V2_1, LanceFileVersion::V2_2, LanceFileVersion::V2_3)]
        version: LanceFileVersion,
        #[values(true, false)] children_nullable: bool,
    ) {
        let (_test_dir, dataset) =
            nested_dataset_with_nullable_struct(true, children_nullable, Some(version)).await;
        let nested_id = field_id(&dataset, "nested");
        let mut writer = open_writer_on(&dataset, 0, &[nested_id]).await;
        writer
            .write_batch(&batch(
                0,
                &[0, 1],
                vec![(
                    "nested",
                    nested_array_with_validity(
                        vec![-7, -8],
                        vec![-70, -80],
                        Some(NullBuffer::from(vec![true, false])),
                        children_nullable,
                    ),
                )],
            ))
            .await
            .unwrap();
        let dataset = commit(dataset, writer.finish().await.unwrap().unwrap()).await;

        let batch = dataset
            .scan()
            .project(&["nested"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let nested = batch["nested"].as_struct();
        assert!(nested.is_valid(0));
        assert!(nested.is_null(1));
        assert!(nested.is_valid(2));
        assert!(nested.is_valid(3));
        let a = nested.column(0).as_primitive::<Int32Type>();
        let b = nested.column(1).as_primitive::<Int32Type>();
        assert_eq!((a.value(0), b.value(0)), (-7, -70));
        assert_eq!((a.value(2), b.value(2)), (2, 20));
    }

    /// `outer: struct<middle: struct<a: i32>>` where `outer` is nullable, `a` is
    /// required, and `middle` is required unless `middle_nullable` is set.
    fn double_nested_array(
        a: Vec<i32>,
        outer_nulls: Option<NullBuffer>,
        middle_nulls: Option<NullBuffer>,
        middle_nullable: bool,
    ) -> ArrayRef {
        let middle_fields =
            Fields::from(vec![Arc::new(ArrowField::new("a", DataType::Int32, false))]);
        let middle: ArrayRef = Arc::new(
            StructArray::try_new(
                middle_fields.clone(),
                vec![Arc::new(Int32Array::from(a)) as ArrayRef],
                middle_nulls,
            )
            .unwrap(),
        );
        let outer_fields = Fields::from(vec![Arc::new(ArrowField::new(
            "middle",
            DataType::Struct(middle_fields),
            middle_nullable,
        ))]);
        Arc::new(StructArray::try_new(outer_fields, vec![middle], outer_nulls).unwrap())
    }

    /// Two four-row fragments of `id` and `outer: struct<middle: struct<a: i32>>`.
    /// Row 0 of fragment 0 has `outer` NULL, so `middle` and `a` are NULL under it;
    /// every other row is valid. Fragment 1 holds values 100.. so an overlay leaking
    /// across fragments is caught.
    async fn double_nested_dataset(
        middle_nullable: bool,
        data_storage_version: LanceFileVersion,
    ) -> (TempDir, Arc<Dataset>) {
        let test_dir = tempdir().unwrap();
        let uri = test_dir.path().to_str().unwrap().to_string();
        let mut a = (0..4).collect::<Vec<_>>();
        a.extend(100..104);
        let valid = || NullBuffer::from((0..8).map(|row| row != 0).collect::<Vec<_>>());
        let base = RecordBatch::try_from_iter_with_nullable(vec![
            ("id", int32((0..8).map(Some).collect()), true),
            (
                "outer",
                double_nested_array(a, Some(valid()), Some(valid()), middle_nullable),
                true,
            ),
        ])
        .unwrap();
        let arrow_schema = base.schema();
        let reader = RecordBatchIterator::new(vec![Ok(base)], arrow_schema);
        let dataset = Arc::new(
            Dataset::write(
                reader,
                &uri,
                Some(WriteParams {
                    max_rows_per_file: 4,
                    max_rows_per_group: 4,
                    data_storage_version: Some(data_storage_version),
                    ..Default::default()
                }),
            )
            .await
            .unwrap(),
        );
        (test_dir, dataset)
    }

    /// Every row and validity bit the overlay in
    /// `test_required_inner_struct_validity_is_replaced` can affect, read through
    /// whichever handle is passed in.
    async fn assert_double_nested_overlay_rows(dataset: &Dataset) {
        let batch = dataset
            .scan()
            .project(&["id", "outer"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let ids = batch["id"].as_primitive::<Int32Type>();
        assert_eq!(
            (0..8).map(|row| ids.value(row)).collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        let outer = batch["outer"].as_struct();
        assert!(outer.is_valid(0));
        assert!(outer.is_null(1));
        assert!(outer.is_valid(2));
        assert!(outer.is_valid(3));
        let middle = outer.column(0).as_struct();
        assert!(middle.is_valid(0));
        assert!(middle.is_null(1));
        assert!(middle.is_valid(2));
        assert!(middle.is_valid(3));
        let a = middle.column(0).as_primitive::<Int32Type>();
        assert_eq!(a.value(0), -7);
        assert_eq!(a.value(2), 2);
        assert_eq!(a.value(3), 3);
        // Fragment 1 is not covered by the overlay: values and validity stay unchanged.
        for row in 4..8 {
            assert!(outer.is_valid(row));
            assert!(middle.is_valid(row));
            assert!(a.is_valid(row));
        }
        assert_eq!(a.value(4), 100);
        assert_eq!(a.value(5), 101);
        assert_eq!(a.value(6), 102);
        assert_eq!(a.value(7), 103);
    }

    /// An overlay replaces the validity of a struct whose required child is itself a
    /// struct. Restoring the outer struct before the inner one would build an illegal
    /// intermediate: the required `middle` still carries the base null on the row the
    /// overlay makes valid, which arrow rejects. `middle_nullable = true` is the
    /// control, since the required-child check does not apply to it.
    #[rstest]
    #[tokio::test]
    async fn test_required_inner_struct_validity_is_replaced(
        #[values(LanceFileVersion::V2_1, LanceFileVersion::V2_2, LanceFileVersion::V2_3)]
        version: LanceFileVersion,
        #[values(false, true)] middle_nullable: bool,
    ) {
        let (_test_dir, dataset) = double_nested_dataset(middle_nullable, version).await;
        let outer_id = field_id(&dataset, "outer");
        let mut writer = open_writer_on(&dataset, 0, &[outer_id]).await;
        writer
            .write_batch(&batch(
                0,
                &[0, 1],
                vec![(
                    "outer",
                    double_nested_array(
                        vec![-7, 0],
                        Some(NullBuffer::from(vec![true, false])),
                        Some(NullBuffer::from(vec![true, false])),
                        middle_nullable,
                    ),
                )],
            ))
            .await
            .unwrap();
        let committed = commit(dataset, writer.finish().await.unwrap().unwrap()).await;
        // The commit handle may still serve readers from write-side state. Reopen the
        // published version so the persisted struct columns go through the normal read
        // path as well, and assert both handles against the same oracle.
        let reopened = Dataset::open(committed.uri()).await.unwrap();
        for dataset in [&committed, &reopened] {
            assert_double_nested_overlay_rows(dataset).await;
        }
    }

    #[tokio::test]
    async fn test_struct_column_shares_its_coverage_with_its_children() {
        let (_test_dir, dataset) = nested_dataset().await;

        let val_id = field_id(&dataset, "val");
        let nested_id = field_id(&dataset, "nested");
        let child_ids = dataset
            .schema()
            .field("nested")
            .unwrap()
            .children
            .iter()
            .map(|child| child.id)
            .collect::<Vec<_>>();
        assert_eq!(child_ids.len(), 2);

        let mut writer = open_writer_on(&dataset, 0, &[val_id, nested_id]).await;
        writer
            .write_batch(&batch(0, &[0], vec![("val", int32(vec![Some(-1)]))]))
            .await
            .unwrap();
        writer
            .write_batch(&batch(
                0,
                &[2],
                vec![("nested", nested_array(vec![-7], vec![-70]))],
            ))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        // The value file lists leaves, so `nested` contributes two entries
        // rather than one: sparse coverage is longer than the overlay's two
        // declared fields, and a child's bitmap has to be found through its
        // parent.
        let overlay = &group.overlays[0];
        assert_eq!(
            overlay.data_file.fields.len(),
            1 + child_ids.len(),
            "expected leaf fields in {:?}",
            overlay.data_file.fields
        );
        assert!(matches!(overlay.coverage, OverlayCoverage::PerField(_)));
        assert_eq!(coverage_of(&group, val_id), RoaringBitmap::from_iter([0]));
        for id in child_ids {
            assert_eq!(
                coverage_of(&group, id),
                RoaringBitmap::from_iter([2]),
                "field {id} should carry the struct's coverage"
            );
        }

        let dataset = commit(dataset, group).await;
        let read_back = dataset
            .scan()
            .project(&["val", "nested"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let vals = read_back["val"].as_primitive::<Int32Type>();
        assert_eq!((vals.value(0), vals.value(2)), (-1, 2));
        let nested = read_back["nested"].as_struct();
        let child = |i: usize| {
            nested
                .column(i)
                .as_primitive::<Int32Type>()
                .values()
                .to_vec()
        };
        assert_eq!((child(0)[2], child(1)[2]), (-7, -70));
        assert_eq!((child(0)[0], child(1)[0]), (0, 0));
    }

    /// A struct array of the given `(name, values)` children, in the order given.
    fn struct_of(children: Vec<(&str, Vec<i32>)>) -> ArrayRef {
        Arc::new(StructArray::from(
            children
                .into_iter()
                .map(|(name, values)| {
                    (
                        Arc::new(ArrowField::new(name, DataType::Int32, true)),
                        Arc::new(Int32Array::from(values)) as ArrayRef,
                    )
                })
                .collect::<Vec<_>>(),
        ))
    }

    /// Projection descends by name and downcasts to the *target's* type, so
    /// every one of these reached an Arrow cast and panicked before the batch
    /// was admitted against the whole field tree.
    #[rstest]
    #[case::struct_arrived_as_a_primitive(
        vec![("nested", int32(vec![Some(99)]))],
        "nested"
    )]
    #[case::struct_child_of_the_wrong_type(
        vec![("nested", Arc::new(StructArray::from(vec![
            (
                Arc::new(ArrowField::new("a", DataType::Utf8, true)),
                Arc::new(arrow_array::StringArray::from(vec![Some("x")])) as ArrayRef,
            ),
            (
                Arc::new(ArrowField::new("b", DataType::Int32, true)),
                Arc::new(Int32Array::from(vec![1])) as ArrayRef,
            ),
        ])) as ArrayRef)],
        "a"
    )]
    #[case::struct_missing_a_child(
        vec![("nested", struct_of(vec![("a", vec![1])]))],
        "b"
    )]
    #[case::struct_with_an_extra_child(
        vec![("nested", struct_of(vec![("a", vec![1]), ("b", vec![2]), ("c", vec![3])]))],
        "c"
    )]
    #[tokio::test]
    async fn test_malformed_nested_batch_is_rejected(
        #[case] columns: Vec<(&str, ArrayRef)>,
        #[case] message: &str,
    ) {
        let (_test_dir, dataset) = nested_dataset().await;
        let nested_id = field_id(&dataset, "nested");
        let mut writer = open_writer_on(&dataset, 0, &[nested_id]).await;

        let error = writer
            .write_batch(&batch(0, &[0], columns))
            .await
            .unwrap_err();
        let display = error.to_string();
        assert!(display.contains(message), "{display}");
        assert!(
            matches!(error, WriteOverlayError::BatchSchemaMismatch { .. }),
            "unexpected variant: {display}"
        );
        writer.abort().await;
    }

    /// Duplicate siblings are invisible to a name-set comparison, and projection
    /// would silently keep whichever came first.
    #[tokio::test]
    async fn test_duplicate_struct_child_is_rejected() {
        let (_test_dir, dataset) = nested_dataset().await;
        let nested_id = field_id(&dataset, "nested");
        let mut writer = open_writer_on(&dataset, 0, &[nested_id]).await;

        let duplicated = struct_of(vec![("a", vec![1]), ("a", vec![2]), ("b", vec![3])]);
        let error = writer
            .write_batch(&batch(0, &[0], vec![("nested", duplicated)]))
            .await
            .unwrap_err();
        let display = error.to_string();
        assert!(display.contains("'nested.a' twice"), "{display}");
        assert!(
            matches!(error, WriteOverlayError::DuplicateColumn { .. }),
            "unexpected variant: {display}"
        );
        writer.abort().await;
    }

    /// The admission pass compares the tree, not the child order: a struct whose
    /// children arrive reversed is projected back into the manifest's order
    /// rather than landing under the wrong field ids.
    #[tokio::test]
    async fn test_struct_children_may_arrive_in_any_order() {
        let (_test_dir, dataset) = nested_dataset().await;
        let nested_id = field_id(&dataset, "nested");
        let mut writer = open_writer_on(&dataset, 0, &[nested_id]).await;

        let reversed = struct_of(vec![("b", vec![-70]), ("a", vec![-7])]);
        writer
            .write_batch(&batch(0, &[2], vec![("nested", reversed)]))
            .await
            .unwrap();
        let group = writer.finish().await.unwrap().unwrap();

        let dataset = commit(dataset, group).await;
        let read_back = dataset
            .scan()
            .project(&["nested"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let nested = read_back["nested"].as_struct();
        let child = |i: usize| nested.column(i).as_primitive::<Int32Type>().value(2);
        assert_eq!((child(0), child(1)), (-7, -70));
    }

    #[tokio::test]
    async fn test_blob_field_is_rejected() {
        let test_dir = tempdir().unwrap();
        let uri = test_dir.path().to_str().unwrap().to_string();
        let field = |name, data_type| Arc::new(ArrowField::new(name, data_type, true));
        let descriptor: ArrayRef = Arc::new(StructArray::from(vec![
            (
                field("data", DataType::LargeBinary),
                Arc::new(arrow_array::LargeBinaryArray::from(vec![Some(
                    b"hello".as_slice(),
                )])) as ArrayRef,
            ),
            (
                field("uri", DataType::Utf8),
                Arc::new(arrow_array::StringArray::from(vec![None::<&str>])) as ArrayRef,
            ),
            (
                field("position", DataType::UInt64),
                Arc::new(UInt64Array::from(vec![None::<u64>])) as ArrayRef,
            ),
            (
                field("size", DataType::UInt64),
                Arc::new(UInt64Array::from(vec![None::<u64>])) as ArrayRef,
            ),
        ]));
        let blob = ArrowField::new("blob", descriptor.data_type().clone(), true).with_metadata(
            [(
                lance_arrow::ARROW_EXT_NAME_KEY.to_string(),
                lance_arrow::BLOB_V2_EXT_NAME.to_string(),
            )]
            .into_iter()
            .collect(),
        );
        let arrow_schema = Arc::new(ArrowSchema::new(vec![blob]));
        let base = RecordBatch::try_new(arrow_schema.clone(), vec![descriptor]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(base)], arrow_schema);
        // Blob v2 columns need a 2.2 or newer file.
        let params = WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        };
        let dataset = Arc::new(Dataset::write(reader, &uri, Some(params)).await.unwrap());

        let schema = overlay_schema(&dataset, &[field_id(&dataset, "blob")]);
        let Err(error) = dataset
            .get_fragment(0)
            .unwrap()
            .write_overlay(&schema)
            .await
        else {
            panic!("expected opening the overlay to fail");
        };
        assert!(matches!(error, WriteOverlayError::BlobField { .. }));
        assert!(
            error.to_string().contains("blob columns are not supported"),
            "{error}"
        );
        assert!(matches!(Error::from(error), Error::NotSupported { .. }));
    }
}
