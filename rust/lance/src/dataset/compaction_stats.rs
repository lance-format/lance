// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-fragment column-layout statistics.
//!
//! An `add_columns` call that writes data files writes one per fragment, so a
//! wide fragment can end up split across many small files. These stats, from
//! [`Dataset::column_layout_stats`], make that layout visible. The default
//! compaction planner considers repacking a fragment's columns when its
//! `live_file_count` is above
//! [`max_data_files_per_fragment`](crate::dataset::optimize::CompactionOptions::max_data_files_per_fragment);
//! a custom
//! [`CompactionPlanner`](crate::dataset::optimize::CompactionPlanner) can use
//! the rest.

use super::Dataset;

pub use lance_table_runtime::optimize::FragmentColumnLayoutStats;

impl Dataset {
    /// Per-fragment column-layout stats, in manifest fragment order.
    ///
    /// Reads only fragment metadata, never a data file.
    ///
    /// This is an experimental API. It may change at any time.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::Dataset;
    /// # fn example(dataset: &Dataset) {
    /// for stats in dataset.column_layout_stats() {
    ///     println!(
    ///         "fragment {}: {} live files",
    ///         stats.fragment_id, stats.live_file_count
    ///     );
    /// }
    /// # }
    /// ```
    pub fn column_layout_stats(&self) -> Vec<FragmentColumnLayoutStats> {
        self.table_view().to_table().column_layout_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{NewColumnTransform, WriteParams};
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, RecordBatchIterator, StructArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_table::format::overlay::TOMBSTONE_FIELD_ID;
    use std::sync::Arc;

    #[tokio::test]
    async fn column_layout_stats_counts_files_per_fragment() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..8))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        // Two fragments (max_rows_per_file = 4), one data file each to start.
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let stats = dataset.column_layout_stats();
        assert_eq!(stats.len(), 2);
        assert!(
            stats
                .iter()
                .all(|s| s.live_file_count == 1 && s.overlay_count == 0)
        );

        // add_columns appends a second data file per fragment.
        dataset
            .add_columns(
                NewColumnTransform::SqlExpressions(vec![("b".into(), "a + 1".into())]),
                None,
                None,
            )
            .await
            .unwrap();

        let stats = dataset.column_layout_stats();
        assert_eq!(stats.len(), 2);
        assert!(
            stats.iter().all(|s| s.live_file_count == 2),
            "each fragment should now have 2 data files: {stats:?}"
        );
        for s in &stats {
            assert_eq!(s.fields_per_file, vec![1, 1]);
            assert!(s.file_sizes.iter().all(Option::is_some), "{s:?}");
            assert_eq!(s.dead_slot_ratio, 0.0);
        }
    }

    /// A dropped column leaves its field id in the file that held it, which
    /// counts as a dead slot.
    #[tokio::test]
    async fn column_layout_stats_counts_dropped_columns_as_dead_slots() {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..4)),
                Arc::new(Int32Array::from_iter_values(4..8)),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        dataset.drop_columns(&["b"]).await.unwrap();

        let stats = dataset.column_layout_stats();
        assert_eq!(stats[0].live_file_count, 1);
        assert_eq!(stats[0].fields_per_file, vec![1]);
        assert_eq!(stats[0].dead_slot_ratio, 0.5);
    }

    /// Only files holding a schema column count, and an overlay is not one of
    /// the fragment's files. The extra files and the overlay are added to the
    /// manifest by hand: the commit paths that leave such files behind (an
    /// in-place column update after a drop, a full horizontal rewrite on a
    /// table whose lineage was spilled into a data file) are not what this
    /// test is about.
    #[tokio::test]
    async fn column_layout_stats_skips_files_without_schema_columns() {
        use lance_file::version::ConcreteFileVersion;
        use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
        use lance_table::format::{
            DataFile, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
            ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, RowIdMeta,
        };
        use lance_table::rowids::version::RowDatasetVersionMeta;
        use roaring::RoaringBitmap;

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..4))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let dropped_id = dataset.schema().max_field_id().unwrap() + 1;
        let a_id = dataset.schema().field("a").unwrap().id;
        let mut manifest = dataset.manifest.as_ref().clone();
        let mut fragments = manifest.fragments.as_ref().clone();
        let file = |fields: Vec<i32>| {
            let indices = (0..fields.len() as i32).collect();
            DataFile::new(
                "extra.lance",
                fields,
                indices,
                ConcreteFileVersion::V2_0,
                None,
                None,
            )
        };
        fragments[0].files.push(file(vec![TOMBSTONE_FIELD_ID]));
        fragments[0].files.push(file(vec![dropped_id]));
        fragments[0].files.push(file(vec![
            TOMBSTONE_FIELD_ID,
            ROW_ID_FIELD_ID,
            ROW_CREATED_AT_VERSION_FIELD_ID,
            ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
        ]));
        fragments[0].row_id_meta = Some(RowIdMeta::Column);
        fragments[0].created_at_version_meta = Some(RowDatasetVersionMeta::Column);
        fragments[0].last_updated_at_version_meta = Some(RowDatasetVersionMeta::Column);
        // An overlay holding a schema column is counted on its own, not as one
        // of the fragment's files.
        fragments[0].overlays.push(DataOverlayFile {
            data_file: file(vec![a_id]),
            coverage: OverlayCoverage::Shared(Arc::new(RoaringBitmap::from_iter([0u32]))),
            committed_version: dataset.version().version,
        });
        manifest.fragments = Arc::new(fragments);
        dataset.manifest = Arc::new(manifest);

        let stats = dataset.column_layout_stats();
        assert_eq!(stats[0].live_file_count, 1, "{stats:?}");
        assert_eq!(stats[0].fields_per_file, vec![1, 0, 0, 0]);
        assert!(stats[0].file_sizes[0].is_some());
        assert_eq!(stats[0].file_sizes[1..], [None, None, None]);
        // `a`, two tombstones and the dropped id; the lineage slots count on
        // neither side.
        assert_eq!(stats[0].dead_slot_ratio, 0.75);
        assert_eq!(stats[0].overlay_count, 1);
    }

    /// A v2 file lists a struct's child fields, which are schema fields too.
    #[tokio::test]
    async fn column_layout_stats_counts_nested_fields_as_live() {
        let struct_fields = vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Int32, false),
        ];
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("s", DataType::Struct(struct_fields.clone().into()), false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from_iter_values(0..4)),
                Arc::new(StructArray::new(
                    struct_fields.into(),
                    vec![
                        Arc::new(Int32Array::from_iter_values(4..8)) as ArrayRef,
                        Arc::new(Int32Array::from_iter_values(8..12)),
                    ],
                    None,
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let fields_in_file = dataset.manifest.fragments[0].files[0].fields.len();
        assert!(fields_in_file >= 3, "expected the child ids in the file");
        let stats = dataset.column_layout_stats();
        assert_eq!(stats[0].live_file_count, 1);
        assert_eq!(stats[0].fields_per_file, vec![fields_in_file]);
        assert_eq!(stats[0].dead_slot_ratio, 0.0);
    }

    /// `Schema::validate` accepts a top-level field carrying the reserved id of
    /// its lineage name, so the schema ids can include one. It still counts on
    /// neither side of the ratio.
    #[tokio::test]
    async fn column_layout_stats_ignores_a_lineage_id_in_the_schema() {
        use lance_core::{ROW_ID, ROW_ID_FIELD_ID, datatypes::Field as LanceField};
        use lance_file::version::ConcreteFileVersion;
        use lance_table::format::{DataFile, RowIdMeta};

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from_iter_values(0..4))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();

        let mut manifest = dataset.manifest.as_ref().clone();
        let mut row_id = LanceField::new_arrow(ROW_ID, DataType::UInt64, false).unwrap();
        row_id.id = ROW_ID_FIELD_ID;
        manifest.schema.fields.push(row_id);
        manifest.schema.validate().unwrap();
        let mut fragments = manifest.fragments.as_ref().clone();
        fragments[0].files.push(DataFile::new(
            "lineage.lance",
            vec![ROW_ID_FIELD_ID],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        ));
        fragments[0].row_id_meta = Some(RowIdMeta::Column);
        manifest.fragments = Arc::new(fragments);
        dataset.manifest = Arc::new(manifest);

        let stats = dataset.column_layout_stats();
        assert_eq!(stats[0].dead_slot_ratio, 0.0, "{stats:?}");
    }
}
