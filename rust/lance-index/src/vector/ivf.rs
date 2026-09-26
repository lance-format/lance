// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF - Inverted File Index

use std::ops::Range;
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, UInt32Array};

pub use builder::IvfBuildParams;
use lance_core::Result;
use lance_linalg::distance::{DistanceType, MetricType};
use tracing::instrument;

use crate::vector::bq::builder::RabitQuantizer;
use crate::vector::bq::transform::RQTransformer;
use crate::vector::ivf::transform::PartitionTransformer;
use crate::vector::kmeans::{compute_partitions_arrow_array, kmeans_find_partitions_arrow_array};
use crate::vector::{pq::ProductQuantizer, transform::Transformer};

use super::flat::transform::FlatTransformer;
use super::pq::transform::PQTransformer;
use super::quantizer::Quantization;
use super::residual::ResidualTransform;
use super::sq::ScalarQuantizer;
use super::sq::transform::SQTransformer;
use super::transform::KeepFiniteVectors;
use super::{PART_ID_COLUMN, PQ_CODE_COLUMN, SQ_CODE_COLUMN};
use super::{quantizer::Quantizer, residual::compute_residual};

pub mod builder;
pub mod shuffler;
pub mod storage;
mod transform;

/// Create an IVF from the flatten centroids.
///
/// Parameters
/// ----------
/// - *centroids*: a flatten floating number array of centroids.
/// - *dimension*: dimension of the vector.
/// - *metric_type*: metric type to compute pair-wise vector distance.
/// - *transforms*: a list of transforms to apply to the vector column.
/// - *range*: only covers a range of partitions. Default is None
pub fn new_ivf_transformer(
    centroids: FixedSizeListArray,
    metric_type: DistanceType,
    transforms: Vec<Arc<dyn Transformer>>,
) -> IvfTransformer {
    IvfTransformer::new(centroids, metric_type, transforms)
}

pub fn new_ivf_transformer_with_quantizer(
    centroids: FixedSizeListArray,
    metric_type: MetricType,
    vector_column: &str,
    quantizer: Quantizer,
    range: Option<Range<u32>>,
) -> Result<IvfTransformer> {
    match quantizer {
        Quantizer::Flat(_) | Quantizer::FlatBin(_) => Ok(IvfTransformer::new_flat(
            centroids,
            metric_type,
            vector_column,
            range,
        )),
        Quantizer::Product(pq) => Ok(IvfTransformer::with_pq(
            centroids,
            metric_type,
            vector_column,
            pq,
            range,
        )),
        Quantizer::Scalar(sq) => Ok(IvfTransformer::with_sq(
            centroids,
            metric_type,
            vector_column,
            sq,
            range,
        )),
        Quantizer::Rabit(rq) => {
            IvfTransformer::with_rq(centroids, metric_type, vector_column, rq, range)
        }
    }
}

/// IVF - IVF file partition
///
#[derive(Debug)]
pub struct IvfTransformer {
    /// Centroids of a cluster algorithm, to run IVF.
    ///
    /// It is a 2-D `(num_partitions * dimension)` of floating array.
    centroids: FixedSizeListArray,

    /// Transform applied to each partition.
    transforms: Vec<Arc<dyn Transformer>>,

    /// Metric type to compute pair-wise vector distance.
    distance_type: DistanceType,
}

impl IvfTransformer {
    /// Create a new Ivf model.
    pub fn new(
        centroids: FixedSizeListArray,
        metric_type: MetricType,
        transforms: Vec<Arc<dyn Transformer>>,
    ) -> Self {
        Self {
            centroids,
            distance_type: metric_type,
            transforms,
        }
    }

    pub fn new_partition_transformer(
        centroids: FixedSizeListArray,
        distance_type: DistanceType,
        vector_column: &str,
    ) -> Self {
        let mut transforms: Vec<Arc<dyn Transformer>> =
            vec![Arc::new(super::transform::Flatten::new(vector_column))];

        let distance_type = if distance_type == MetricType::Cosine {
            transforms.push(Arc::new(super::transform::NormalizeTransformer::new(
                vector_column,
            )));
            MetricType::L2
        } else {
            distance_type
        };
        transforms.push(Arc::new(KeepFiniteVectors::new(vector_column)));

        let partition_transform = Arc::new(PartitionTransformer::new(
            centroids.clone(),
            distance_type,
            vector_column,
        ));
        transforms.push(partition_transform);
        Self::new(centroids, distance_type, transforms)
    }

    pub fn new_flat(
        centroids: FixedSizeListArray,
        distance_type: DistanceType,
        vector_column: &str,
        range: Option<Range<u32>>,
    ) -> Self {
        let mut transforms: Vec<Arc<dyn Transformer>> =
            vec![Arc::new(super::transform::Flatten::new(vector_column))];

        let dt = if distance_type == DistanceType::Cosine {
            transforms.push(Arc::new(super::transform::NormalizeTransformer::new(
                vector_column,
            )));
            MetricType::L2
        } else {
            distance_type
        };
        transforms.push(Arc::new(KeepFiniteVectors::new(vector_column)));

        let ivf_transform = Arc::new(PartitionTransformer::new(
            centroids.clone(),
            dt,
            vector_column,
        ));
        transforms.push(ivf_transform);

        if let Some(range) = range {
            transforms.push(Arc::new(transform::PartitionFilter::new(
                PART_ID_COLUMN,
                range,
            )));
        }

        transforms.push(Arc::new(FlatTransformer::new(vector_column)));

        // Keep the converted metric, like the sibling constructors: the chain
        // above normalized for cosine and assigns with L2, so the transformer's
        // own `find_partitions` and `compute_partitions` have to measure the
        // same way.
        Self::new(centroids, dt, transforms)
    }

    /// Create a IVF_PQ struct.
    pub fn with_pq(
        centroids: FixedSizeListArray,
        distance_type: DistanceType,
        vector_column: &str,
        pq: ProductQuantizer,
        range: Option<Range<u32>>,
    ) -> Self {
        let mut transforms: Vec<Arc<dyn Transformer>> =
            vec![Arc::new(super::transform::Flatten::new(vector_column))];

        let distance_type = if distance_type == MetricType::Cosine {
            transforms.push(Arc::new(super::transform::NormalizeTransformer::new(
                vector_column,
            )));
            MetricType::L2
        } else {
            distance_type
        };
        transforms.push(Arc::new(KeepFiniteVectors::new(vector_column)));

        let partition_transform = Arc::new(PartitionTransformer::new(
            centroids.clone(),
            distance_type,
            vector_column,
        ));
        transforms.push(partition_transform);

        if let Some(range) = range {
            transforms.push(Arc::new(transform::PartitionFilter::new(
                PART_ID_COLUMN,
                range,
            )));
        }

        if ProductQuantizer::use_residual(distance_type) {
            transforms.push(Arc::new(ResidualTransform::new(
                centroids.clone(),
                PART_ID_COLUMN,
                vector_column,
            )));
        }
        transforms.push(Arc::new(PQTransformer::new(
            pq,
            vector_column,
            PQ_CODE_COLUMN,
        )));

        Self::new(centroids, distance_type, transforms)
    }

    fn with_sq(
        centroids: FixedSizeListArray,
        metric_type: MetricType,
        vector_column: &str,
        sq: ScalarQuantizer,
        range: Option<Range<u32>>,
    ) -> Self {
        let mut transforms: Vec<Arc<dyn Transformer>> =
            vec![Arc::new(super::transform::Flatten::new(vector_column))];

        let distance_type = if metric_type == MetricType::Cosine {
            transforms.push(Arc::new(super::transform::NormalizeTransformer::new(
                vector_column,
            )));
            MetricType::L2
        } else {
            metric_type
        };
        transforms.push(Arc::new(KeepFiniteVectors::new(vector_column)));

        let partition_transformer = Arc::new(PartitionTransformer::new(
            centroids.clone(),
            distance_type,
            vector_column,
        ));
        transforms.push(partition_transformer);

        if let Some(range) = range {
            transforms.push(Arc::new(transform::PartitionFilter::new(
                PART_ID_COLUMN,
                range,
            )));
        }

        transforms.push(Arc::new(SQTransformer::new(
            sq,
            vector_column.to_owned(),
            SQ_CODE_COLUMN.to_owned(),
        )));

        Self::new(centroids, distance_type, transforms)
    }

    fn with_rq(
        centroids: FixedSizeListArray,
        distance_type: DistanceType,
        vector_column: &str,
        rq: RabitQuantizer,
        range: Option<Range<u32>>,
    ) -> Result<Self> {
        let mut transforms: Vec<Arc<dyn Transformer>> =
            vec![Arc::new(super::transform::Flatten::new(vector_column))];

        let distance_type = if distance_type == MetricType::Cosine {
            transforms.push(Arc::new(super::transform::NormalizeTransformer::new(
                vector_column,
            )));
            MetricType::L2
        } else {
            distance_type
        };
        transforms.push(Arc::new(KeepFiniteVectors::new(vector_column)));

        let partition_transform = Arc::new(
            PartitionTransformer::new(centroids.clone(), distance_type, vector_column)
                .with_distance(true),
        );
        transforms.push(partition_transform);

        if let Some(range) = range {
            transforms.push(Arc::new(transform::PartitionFilter::new(
                PART_ID_COLUMN,
                range,
            )));
        }

        transforms.push(Arc::new(ResidualTransform::new(
            centroids.clone(),
            PART_ID_COLUMN,
            vector_column,
        )));

        transforms.push(Arc::new(RQTransformer::new(
            rq,
            distance_type,
            centroids.clone(),
            vector_column,
        )?));

        Ok(Self::new(centroids, distance_type, transforms))
    }

    #[inline]
    pub fn compute_residual(&self, data: &FixedSizeListArray) -> Result<FixedSizeListArray> {
        compute_residual(&self.centroids, data, Some(self.distance_type), None)
    }

    #[inline]
    pub fn compute_partitions(&self, data: &FixedSizeListArray) -> Result<UInt32Array> {
        Ok(
            compute_partitions_arrow_array(&self.centroids, data, self.distance_type)
                .map(|(part_ids, _)| part_ids.into())?,
        )
    }

    pub fn find_partitions(
        &self,
        query: &dyn Array,
        nprobes: usize,
    ) -> Result<(UInt32Array, Float32Array)> {
        Ok(kmeans_find_partitions_arrow_array(
            &self.centroids,
            query,
            nprobes,
            self.distance_type,
        )?)
    }
}

impl Transformer for IvfTransformer {
    #[instrument(name = "IvfTransformer::transform", level = "debug", skip_all)]
    fn transform(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let mut batch = batch.clone();
        for transform in self.transforms.as_slice() {
            batch = transform.transform(&batch)?;
        }
        Ok(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float32Array;
    use lance_arrow::FixedSizeListArrayExt;

    /// `new_flat` normalizes the vectors for cosine and assigns partitions with
    /// L2, like its sibling constructors, so the metric it stores has to be the
    /// converted one too. Otherwise the transformer's own `find_partitions` and
    /// `compute_partitions` measure with cosine while the chain measured with
    /// L2, and the distances it hands back are on a different scale than the
    /// ones it wrote into the batch.
    #[test]
    fn test_new_flat_keeps_the_converted_metric() {
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![1.0f32, 0.0, 0.0, 1.0]),
            2,
        )
        .unwrap();
        let query = Float32Array::from(vec![1.0f32, 0.0]);

        let cosine =
            IvfTransformer::new_flat(centroids.clone(), DistanceType::Cosine, "vector", None);
        let l2 = IvfTransformer::new_flat(centroids, DistanceType::L2, "vector", None);

        let (cosine_parts, cosine_dists) = cosine.find_partitions(&query, 2).unwrap();
        let (l2_parts, l2_dists) = l2.find_partitions(&query, 2).unwrap();

        assert_eq!(cosine_parts.values(), l2_parts.values());
        assert_eq!(
            cosine_dists.values(),
            l2_dists.values(),
            "the cosine flat transformer should report the L2 distances it assigns with"
        );
    }
}
