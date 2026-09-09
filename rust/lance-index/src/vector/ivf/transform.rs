// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Transform of a Vector Input with partition IDs.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::Float32Array;
use arrow_array::{
    Array, FixedSizeListArray, RecordBatch, UInt32Array, cast::AsArray, types::UInt32Type,
};
use lance_table::utils::LanceIteratorExtension;
use tracing::instrument;

use lance_arrow::RecordBatchExt;
use lance_core::Result;
use lance_linalg::distance::DistanceType;

use crate::vector::kmeans::compute_partitions_arrow_array;
use crate::vector::transform::Transformer;
use crate::vector::utils::SimpleIndex;
use crate::vector::{CENTROID_DIST_COLUMN, CENTROID_DIST_FIELD, LOSS_METADATA_KEY, PART_ID_FIELD};

use super::PART_ID_COLUMN;

/// PartitionTransformer
///
/// It computes the partition ID for each row from the input batch,
/// and adds the partition ID as a new column to the batch,
/// and adds the loss as a metadata to the batch.
///
/// If the partition ID ("__ivf_part_id") column is already present in the Batch,
/// this transform is a Noop.
///
#[derive(Debug)]
pub struct PartitionTransformer {
    centroids: FixedSizeListArray,
    distance_type: DistanceType,
    input_column: String,
    output_column: String,
    with_distance: bool,
    index: Option<SimpleIndex>,
}

impl PartitionTransformer {
    pub fn new(
        centroids: FixedSizeListArray,
        distance_type: DistanceType,
        input_column: impl AsRef<str>,
    ) -> Self {
        let index = SimpleIndex::may_train_index(
            centroids.values().clone(),
            centroids.value_length() as usize,
            distance_type,
        )
        .unwrap();

        Self {
            centroids,
            distance_type,
            input_column: input_column.as_ref().to_owned(),
            output_column: PART_ID_COLUMN.to_owned(),
            with_distance: false,
            index,
        }
    }

    pub fn with_distance(mut self, with_distance: bool) -> Self {
        self.with_distance = with_distance;
        self
    }
}
impl Transformer for PartitionTransformer {
    #[instrument(name = "PartitionTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        if !(batch.column_by_name(&self.output_column).is_none()
            || self.with_distance && batch.column_by_name(CENTROID_DIST_COLUMN).is_none())
        {
            // If the output columns are already present, we don't need to compute it again.
            return Ok(batch.clone());
        }

        // clear the columns if any of them is present
        let batch = batch
            .drop_column(PART_ID_COLUMN)?
            .drop_column(CENTROID_DIST_COLUMN)?;

        let arr = batch.column_by_name(&self.input_column).ok_or_else(|| {
            lance_core::Error::index(format!(
                "PartitionTransformer: column {} not found in the RecordBatch",
                self.input_column
            ))
        })?;

        let fsl = arr.as_fixed_size_list_opt().ok_or_else(|| {
            lance_core::Error::index(format!(
                "PartitionTransformer: column {} is not a FixedSizeListArray: {}",
                self.input_column,
                arr.data_type(),
            ))
        })?;

        let (part_ids, dists) = match &self.index {
            Some(index) => fsl
                .iter()
                .map(|vec| match vec {
                    Some(v) => {
                        let (id, dist) = index.search(v).unwrap();
                        (Some(id), Some(dist))
                    }
                    None => (None, None),
                })
                .unzip(),
            None => compute_partitions_arrow_array(&self.centroids, fsl, self.distance_type)?,
        };
        let loss = dists
            .iter()
            .map(|d| d.unwrap_or_default() as f64)
            .sum::<f64>();
        let part_ids = UInt32Array::from(part_ids);
        let mut batch = batch.try_with_column(PART_ID_FIELD.clone(), Arc::new(part_ids))?;
        if self.with_distance {
            let dists = Float32Array::from(dists);
            batch = batch.try_with_column(CENTROID_DIST_FIELD.clone(), Arc::new(dists))?;
        }
        Ok(batch.add_metadata(LOSS_METADATA_KEY.to_owned(), loss.to_string())?)
    }
}

#[derive(Debug)]
pub(super) struct PartitionFilter {
    /// The partition column name.
    column: String,
    /// The partition range to filter.
    partition_range: Range<u32>,
}

impl PartitionFilter {
    pub fn new(column: impl AsRef<str>, partition_range: Range<u32>) -> Self {
        Self {
            column: column.as_ref().to_owned(),
            partition_range,
        }
    }

    fn filter_row_ids(&self, partition_ids: &[u32]) -> Vec<u32> {
        partition_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &part_id)| {
                if self.partition_range.contains(&part_id) {
                    Some(idx as u32)
                } else {
                    None
                }
            })
            // in most cases, no partition will be filtered out.
            .exact_size(partition_ids.len())
            .collect()
    }
}

impl Transformer for PartitionFilter {
    #[instrument(name = "PartitionFilter::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        // TODO: use datafusion execute?
        let arr = batch.column_by_name(&self.column).ok_or_else(|| {
            lance_core::Error::index(format!(
                "PartitionFilter: column {} not found in the RecordBatch",
                self.column
            ))
        })?;
        let part_ids = arr.as_primitive::<UInt32Type>();
        let indices = UInt32Array::from(self.filter_row_ids(part_ids.values()));
        Ok(batch.take(&indices)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Int32Array, cast::AsArray};
    use arrow_schema::{DataType, Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;

    const VECTOR_COLUMN: &str = "v";

    /// Two centroids far enough apart that assignment is unambiguous regardless
    /// of accumulation precision.
    fn centroids() -> FixedSizeListArray {
        FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0, 0.0, 100.0, 100.0]), 2)
            .unwrap()
    }

    fn transformer() -> PartitionTransformer {
        PartitionTransformer::new(centroids(), DistanceType::L2, VECTOR_COLUMN)
    }

    fn vector_batch(vectors: Vec<Vec<f32>>) -> RecordBatch {
        let fsl = FixedSizeListArray::from_iter_primitive::<arrow_array::types::Float32Type, _, _>(
            vectors.into_iter().map(|v| Some(v.into_iter().map(Some))),
            2,
        );
        let schema = Schema::new(vec![Field::new(
            VECTOR_COLUMN,
            fsl.data_type().clone(),
            true,
        )]);
        RecordBatch::try_new(schema.into(), vec![Arc::new(fsl)]).unwrap()
    }

    fn part_ids_of(batch: &RecordBatch) -> Vec<Option<u32>> {
        batch
            .column_by_name(PART_ID_COLUMN)
            .unwrap()
            .as_primitive::<UInt32Type>()
            .iter()
            .collect()
    }

    fn loss_of(batch: &RecordBatch) -> f64 {
        batch
            .schema_ref()
            .metadata()
            .get(LOSS_METADATA_KEY)
            .expect("loss metadata should be attached")
            .parse()
            .unwrap()
    }

    /// A vector assigned to the wrong partition is never searched in the right
    /// one, so this is the assertion the whole file exists for.
    #[test]
    fn test_assigns_the_nearest_centroid() {
        let batch = vector_batch(vec![vec![1.0, -1.0], vec![99.0, 101.0], vec![2.0, 2.0]]);

        let output = transformer().transform(&batch).unwrap();

        assert_eq!(part_ids_of(&output), vec![Some(0), Some(1), Some(0)]);
        // The vector column survives untouched next to the new partition column.
        assert_eq!(
            output.column_by_name(VECTOR_COLUMN).unwrap(),
            batch.column_by_name(VECTOR_COLUMN).unwrap()
        );
    }

    /// Loss is the accumulated centroid distance and is read back by the
    /// shuffler, so it has to be attached even when nothing is off-centroid.
    #[test]
    fn test_loss_is_zero_when_vectors_sit_on_centroids() {
        let batch = vector_batch(vec![vec![0.0, 0.0], vec![100.0, 100.0]]);

        let output = transformer().transform(&batch).unwrap();

        assert_eq!(loss_of(&output), 0.0);
    }

    #[test]
    fn test_loss_accumulates_across_rows() {
        let batch = vector_batch(vec![vec![3.0, 4.0], vec![100.0, 100.0]]);

        let loss = loss_of(&transformer().transform(&batch).unwrap());

        assert!(
            loss > 0.0,
            "loss should reflect the off-centroid row: {loss}"
        );
    }

    #[test]
    fn test_centroid_distance_is_opt_in() {
        let batch = vector_batch(vec![vec![1.0, 1.0]]);

        let without = transformer().transform(&batch).unwrap();
        assert!(without.column_by_name(CENTROID_DIST_COLUMN).is_none());

        let with = transformer().with_distance(true).transform(&batch).unwrap();
        let dists = with
            .column_by_name(CENTROID_DIST_COLUMN)
            .expect("distance column requested")
            .as_primitive::<arrow_array::types::Float32Type>();
        assert_eq!(dists.len(), 1);
        assert!(dists.value(0) > 0.0);
    }

    /// Recomputing over an already-assigned batch would waste the work and could
    /// disagree with the ids the caller already wrote.
    #[test]
    fn test_is_noop_when_partitions_already_present() {
        let assigned = transformer()
            .transform(&vector_batch(vec![vec![1.0, 1.0]]))
            .unwrap();

        let output = transformer().transform(&assigned).unwrap();

        assert_eq!(output, assigned);
    }

    /// Partitions present but distances missing is the one case that still has to
    /// recompute, otherwise `with_distance` silently returns a batch without the
    /// column it promised.
    #[test]
    fn test_recomputes_when_distance_requested_but_absent() {
        let assigned = transformer()
            .transform(&vector_batch(vec![vec![1.0, 1.0]]))
            .unwrap();
        assert!(assigned.column_by_name(CENTROID_DIST_COLUMN).is_none());

        let output = transformer()
            .with_distance(true)
            .transform(&assigned)
            .unwrap();

        assert!(output.column_by_name(CENTROID_DIST_COLUMN).is_some());
        assert_eq!(part_ids_of(&output), vec![Some(0)]);
    }

    #[test]
    fn test_reports_missing_vector_column() {
        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new("other", DataType::Int32, false)]).into(),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();

        let message = transformer().transform(&batch).unwrap_err().to_string();

        assert!(message.contains(VECTOR_COLUMN), "{message}");
        assert!(message.contains("not found"), "{message}");
    }

    #[test]
    fn test_reports_non_vector_column() {
        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new(VECTOR_COLUMN, DataType::Int32, false)]).into(),
            vec![Arc::new(Int32Array::from(vec![1]))],
        )
        .unwrap();

        let message = transformer().transform(&batch).unwrap_err().to_string();

        assert!(message.contains("is not a FixedSizeListArray"), "{message}");
        assert!(message.contains("Int32"), "{message}");
    }

    fn partitioned_batch(part_ids: Vec<u32>, tags: Vec<i32>) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new(PART_ID_COLUMN, DataType::UInt32, false),
            Field::new("tag", DataType::Int32, false),
        ]);
        RecordBatch::try_new(
            schema.into(),
            vec![
                Arc::new(UInt32Array::from(part_ids)),
                Arc::new(Int32Array::from(tags)),
            ],
        )
        .unwrap()
    }

    /// A sharded build only writes the partitions in its own range. Keeping a row
    /// outside the range would put it in the wrong shard's file; dropping one
    /// inside it loses the vector from the index entirely.
    #[test]
    fn test_partition_filter_keeps_only_the_requested_range() {
        let batch = partitioned_batch(vec![0, 3, 1, 7, 2], vec![10, 13, 11, 17, 12]);

        let output = PartitionFilter::new(PART_ID_COLUMN, 1..3)
            .transform(&batch)
            .unwrap();

        let kept: Vec<u32> = output
            .column_by_name(PART_ID_COLUMN)
            .unwrap()
            .as_primitive::<UInt32Type>()
            .values()
            .to_vec();
        assert_eq!(kept, vec![1, 2]);
        // The filter has to carry every other column along with it.
        let tags: Vec<i32> = output
            .column_by_name("tag")
            .unwrap()
            .as_primitive::<arrow_array::types::Int32Type>()
            .values()
            .to_vec();
        assert_eq!(tags, vec![11, 12]);
    }

    #[test]
    fn test_partition_filter_can_keep_nothing() {
        let batch = partitioned_batch(vec![0, 5], vec![1, 2]);

        let output = PartitionFilter::new(PART_ID_COLUMN, 9..10)
            .transform(&batch)
            .unwrap();

        assert_eq!(output.num_rows(), 0);
        assert_eq!(output.schema(), batch.schema());
    }

    #[test]
    fn test_partition_filter_reports_missing_column() {
        let batch = partitioned_batch(vec![0], vec![1]);

        let message = PartitionFilter::new("absent", 0..1)
            .transform(&batch)
            .unwrap_err()
            .to_string();

        assert!(message.contains("absent"), "{message}");
    }
}
