// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-fragment column-layout statistics.
//!
//! A compaction planner keys on how many data files each fragment carries: a
//! fragment split across many small per-column files (typically from repeated
//! `add_columns` backfills) is the signal horizontal compaction targets.
//! Exposing these stats via [`Dataset::column_layout_stats`] keeps the planning
//! decision observable instead of buried in a black-box heuristic.

use std::collections::HashSet;

use lance_table::format::overlay::TOMBSTONE_FIELD_ID;

use super::Dataset;

/// Column-layout statistics for a single fragment.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentColumnLayoutStats {
    /// The fragment these stats describe.
    pub fragment_id: u64,
    /// Number of data files holding at least one column of the dataset schema.
    /// A large value on a wide dataset is what horizontal compaction collapses
    /// back into fewer files. A file left holding only tombstones or dropped
    /// columns, or kept only for the fragment's spilled row lineage, holds no
    /// column a read touches and is not counted.
    pub live_file_count: usize,
    /// Recorded size in bytes of each data file, in the fragment's file order.
    /// `None` when the manifest has no size for the file.
    pub file_sizes: Vec<Option<u64>>,
    /// Number of fields of the dataset schema each data file holds, in the
    /// fragment's file order. A file counted in `live_file_count` holds at
    /// least one.
    pub fields_per_file: Vec<usize>,
    /// The share of the fragment's field slots that hold no live data: slots
    /// tombstoned by a column update, or left by a dropped column.
    /// The reserved ids of spilled row lineage are not counted on either side.
    /// A fragment rewrite reclaims these slots.
    pub tombstoned_field_ratio: f64,
    /// Number of overlay files attached to the fragment.
    pub overlay_count: usize,
}

impl Dataset {
    /// Per-fragment column-layout stats, in manifest fragment order.
    ///
    /// This is the planning input for horizontal compaction: it reads only
    /// fragment metadata (no data files), so it is cheap to call.
    pub fn column_layout_stats(&self) -> Vec<FragmentColumnLayoutStats> {
        let schema_ids: HashSet<i32> = self
            .schema()
            .fields_pre_order()
            .map(|field| field.id)
            .collect();
        self.manifest
            .fragments
            .iter()
            .map(|fragment| {
                let fields_per_file: Vec<usize> = fragment
                    .files
                    .iter()
                    .map(|file| {
                        file.fields
                            .iter()
                            .filter(|id| schema_ids.contains(id))
                            .count()
                    })
                    .collect();
                // Field slots of user columns, live or dead; the other negative
                // ids are spilled row lineage and are left out.
                let user_slots = fragment
                    .files
                    .iter()
                    .flat_map(|file| file.fields.iter())
                    .filter(|id| **id >= 0 || **id == TOMBSTONE_FIELD_ID)
                    .count();
                let live_slots: usize = fields_per_file.iter().sum();
                FragmentColumnLayoutStats {
                    fragment_id: fragment.id,
                    live_file_count: fields_per_file.iter().filter(|count| **count > 0).count(),
                    file_sizes: fragment
                        .files
                        .iter()
                        .map(|file| file.file_size_bytes.get().map(|size| size.get()))
                        .collect(),
                    fields_per_file,
                    tombstoned_field_ratio: if user_slots == 0 {
                        0.0
                    } else {
                        (user_slots - live_slots) as f64 / user_slots as f64
                    },
                    overlay_count: fragment.overlays.len(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{NewColumnTransform, WriteParams};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
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
            assert_eq!(s.tombstoned_field_ratio, 0.0);
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
        assert_eq!(stats[0].tombstoned_field_ratio, 0.5);
    }

    /// Only files holding a schema column count. The extra files are added to
    /// the manifest by hand: the commit paths that leave such files behind (an
    /// in-place column update after a drop, a full horizontal rewrite on a
    /// table whose lineage was spilled into a data file) are not what this
    /// test is about.
    #[tokio::test]
    async fn column_layout_stats_skips_files_without_schema_columns() {
        use lance_file::version::ConcreteFileVersion;
        use lance_table::format::{DataFile, ROW_ID_FIELD_ID, RowIdMeta};

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
        fragments[0]
            .files
            .push(file(vec![TOMBSTONE_FIELD_ID, ROW_ID_FIELD_ID]));
        fragments[0].row_id_meta = Some(RowIdMeta::Column);
        manifest.fragments = Arc::new(fragments);
        dataset.manifest = Arc::new(manifest);

        let stats = dataset.column_layout_stats();
        assert_eq!(stats[0].live_file_count, 1, "{stats:?}");
        assert_eq!(stats[0].fields_per_file, vec![1, 0, 0, 0]);
        assert!(stats[0].file_sizes[0].is_some());
        assert_eq!(stats[0].file_sizes[1..], [None, None, None]);
        // `a`, two tombstones and the dropped id; the row id slot is lineage.
        assert_eq!(stats[0].tombstoned_field_ratio, 0.75);
    }
}
