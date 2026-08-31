// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::OnceLock};

use arrow_array::{Array, ArrayRef, FixedSizeListArray, Float32Array, UInt32Array};
use itertools::Itertools;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use lance_file::versions::v1::{
    reader::FileReader as V1FileReader, writer::FileWriter as V1FileWriter,
};
use lance_io::{traits::WriteExt, utils::read_message};
use lance_linalg::distance::DistanceType;
use lance_table::io::manifest::ManifestDescribing;
use log::debug;
use serde::{Deserialize, Serialize};

use crate::pb::Ivf as PbIvf;
use crate::vector::utils::SimpleIndex;

pub const IVF_METADATA_KEY: &str = "lance:ivf";
pub const IVF_PARTITION_KEY: &str = "lance:ivf:partition";

/// Ivf Model
#[derive(Debug, Clone, PartialEq)]
pub struct IvfModel {
    /// Centroids of each partition.
    ///
    /// It is a 2-D `(num_partitions * dimension)` of vector array.
    pub centroids: Option<FixedSizeListArray>,

    /// Offset of each partition in the file.
    pub offsets: Vec<usize>,

    /// Number of vectors in each partition.
    pub lengths: Vec<u32>,

    /// Kmeans loss
    pub loss: Option<f64>,
}

#[derive(Debug)]
struct CachedCentroidIndex {
    distance_type: DistanceType,
    index: Option<SimpleIndex>,
}

/// The existing build-time centroid HNSW benchmark crosses over near one million scalar values.
/// Keep exact routing below that point until the query-time benchmark justifies a new threshold.
const MIN_CENTROID_INDEX_VALUES: usize = 1_000_000;

/// Session-local cache for an optional HNSW graph over IVF centroids.
///
/// The cache is deliberately owned by the loaded vector index instead of [`IvfModel`], so it is
/// never serialized and does not change the IVF file format.
#[derive(Debug, Default)]
pub struct CentroidIndexCache {
    index: OnceLock<std::result::Result<CachedCentroidIndex, String>>,
}

impl DeepSizeOf for CentroidIndexCache {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.index
            .get()
            .and_then(|result| result.as_ref().ok())
            .and_then(|cached| cached.index.as_ref())
            .map(|index| index.deep_size_of_children(context))
            .unwrap_or_default()
    }
}

impl DeepSizeOf for IvfModel {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.centroids
            .as_ref()
            .map(|centroids| (centroids as &dyn arrow_array::Array).deep_size_of_children(context))
            .unwrap_or_default()
            + self.lengths.deep_size_of_children(context)
            + self.offsets.deep_size_of_children(context)
    }
}

impl IvfModel {
    pub fn empty() -> Self {
        Self {
            centroids: None,
            offsets: vec![],
            lengths: vec![],
            loss: None,
        }
    }

    pub fn new(centroids: FixedSizeListArray, loss: Option<f64>) -> Self {
        Self {
            centroids: Some(centroids),
            offsets: vec![],
            lengths: vec![],
            loss,
        }
    }

    pub fn centroid(&self, partition: usize) -> Option<ArrayRef> {
        self.centroids.as_ref().map(|c| c.value(partition))
    }

    /// Ivf model dimension.
    pub fn dimension(&self) -> usize {
        self.centroids
            .as_ref()
            .map(|c| c.value_length() as usize)
            .unwrap_or(0)
    }

    /// Number of IVF partitions.
    pub fn num_partitions(&self) -> usize {
        self.centroids
            .as_ref()
            .map(|c| c.len())
            .unwrap_or_else(|| self.offsets.len())
    }

    pub fn partition_size(&self, part: usize) -> usize {
        self.lengths.get(part).copied().unwrap_or_default() as usize
    }

    pub fn num_rows(&self) -> u64 {
        self.lengths.iter().map(|x| *x as u64).sum()
    }

    pub fn loss(&self) -> Option<f64> {
        self.loss
    }

    /// Use the query vector to find `nprobes` closest partitions.
    pub fn find_partitions(
        &self,
        query: &dyn Array,
        nprobes: usize,
        distance_type: DistanceType,
    ) -> Result<(UInt32Array, Float32Array)> {
        let internal = crate::vector::ivf::new_ivf_transformer(
            self.centroids.clone().unwrap(),
            distance_type,
            vec![],
        );
        internal.find_partitions(query, nprobes)
    }

    /// Find IVF partitions, optionally using a lazily cached HNSW graph over centroids.
    ///
    /// Exact scanning remains the default. When `centroid_ef` is set, it must be at least
    /// `nprobes`. Small centroid sets, unsupported types, and a cache built for a different
    /// distance metric fall back to exact scanning.
    pub fn find_partitions_with_centroid_index(
        &self,
        query: &dyn Array,
        nprobes: usize,
        distance_type: DistanceType,
        centroid_ef: Option<usize>,
        centroid_index: &CentroidIndexCache,
    ) -> Result<(UInt32Array, Float32Array)> {
        let Some(centroid_ef) = centroid_ef else {
            return self.find_partitions(query, nprobes, distance_type);
        };
        if centroid_ef < nprobes {
            return Err(Error::invalid_input(format!(
                "centroid_ef must be >= maximum_nprobes, got centroid_ef={centroid_ef}, maximum_nprobes={nprobes}"
            )));
        }

        let Some(centroids) = self.centroids.as_ref() else {
            return self.find_partitions(query, nprobes, distance_type);
        };
        if centroids.values().len() < MIN_CENTROID_INDEX_VALUES {
            return self.find_partitions(query, nprobes, distance_type);
        }

        self.find_partitions_with_hnsw(query, nprobes, distance_type, centroid_ef, centroid_index)
    }

    fn find_partitions_with_hnsw(
        &self,
        query: &dyn Array,
        nprobes: usize,
        distance_type: DistanceType,
        centroid_ef: usize,
        centroid_index: &CentroidIndexCache,
    ) -> Result<(UInt32Array, Float32Array)> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| Error::index("IVF centroids are not available for centroid routing"))?;
        let cached = centroid_index.index.get_or_init(|| {
            SimpleIndex::try_new_centroid_index(
                centroids.values().clone(),
                centroids.value_length() as usize,
                distance_type,
            )
            .map(|index| CachedCentroidIndex {
                distance_type,
                index,
            })
            .map_err(|error| error.to_string())
        });
        let cached = cached.as_ref().map_err(|message| {
            Error::index(format!("failed to build centroid HNSW index: {message}"))
        })?;
        let Some(index) = cached
            .index
            .as_ref()
            .filter(|_| cached.distance_type == distance_type)
        else {
            return self.find_partitions(query, nprobes, distance_type);
        };
        let result = index.search(query.slice(0, query.len()), nprobes, centroid_ef)?;
        let expected = nprobes.min(centroids.len());
        if result.0.len() < expected {
            return self.find_partitions(query, nprobes, distance_type);
        }
        Ok(result)
    }

    /// Add the offset and length of one partition.
    pub fn add_partition(&mut self, len: u32) {
        self.offsets.push(
            self.offsets.last().cloned().unwrap_or_default()
                + self.lengths.last().cloned().unwrap_or_default() as usize,
        );
        self.lengths.push(len);
    }

    /// Add the offset and length of one partition with the given offset.
    /// this is used for old index format of IVF_PQ.
    pub fn add_partition_with_offset(&mut self, offset: usize, len: u32) {
        self.offsets.push(offset);
        self.lengths.push(len);
    }

    /// Get a reference to all centroids as a [`FixedSizeListArray`].
    ///
    /// Returns `None` if the model does not contain centroids
    pub fn centroids_array(&self) -> Option<&FixedSizeListArray> {
        self.centroids.as_ref()
    }

    pub fn row_range(&self, partition: usize) -> Range<usize> {
        let start = self.offsets[partition];
        let end = start + self.lengths[partition] as usize;
        start..end
    }

    pub async fn load(reader: &V1FileReader) -> Result<Self> {
        let schema = reader.schema();
        let meta_str = schema
            .metadata
            .get(IVF_METADATA_KEY)
            .ok_or(Error::index(format!(
                "{} not found during search",
                IVF_METADATA_KEY
            )))?;
        let ivf_metadata: IvfMetadata = serde_json::from_str(meta_str)
            .map_err(|e| Error::index(format!("Failed to parse IVF metadata: {}", e)))?;

        let pb: PbIvf = read_message(
            reader.object_reader.as_ref(),
            ivf_metadata.pb_position as usize,
        )
        .await?;
        Self::try_from(pb)
    }

    /// Write the IVF metadata to the lance file.
    pub async fn write(&self, writer: &mut V1FileWriter<ManifestDescribing>) -> Result<()> {
        let pb = PbIvf::try_from(self)?;
        let pos = writer.object_writer.write_protobuf(&pb).await?;
        let ivf_metadata = IvfMetadata { pb_position: pos };
        writer.add_metadata(IVF_METADATA_KEY, &serde_json::to_string(&ivf_metadata)?);
        Ok(())
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<&IvfModel> for PbIvf {
    type Error = Error;

    fn try_from(ivf: &IvfModel) -> Result<Self> {
        let lengths = ivf.lengths.clone();

        Ok(Self {
            centroids: vec![], // Deprecated
            lengths,
            offsets: ivf.offsets.iter().map(|x| *x as u64).collect(),
            centroids_tensor: ivf.centroids.as_ref().map(|c| c.try_into()).transpose()?,
            loss: ivf.loss,
        })
    }
}

/// Convert IvfModel to protobuf.
impl TryFrom<PbIvf> for IvfModel {
    type Error = Error;

    fn try_from(proto: PbIvf) -> Result<Self> {
        let centroids = if let Some(tensor) = proto.centroids_tensor.as_ref() {
            // For new index format and IVFIndex
            debug!("Ivf: loading IVF centroids from index format v2");
            Some(FixedSizeListArray::try_from(tensor)?)
        } else if !proto.centroids.is_empty() {
            // For backward-compatibility
            debug!("Ivf: loading IVF centroids from index format v1");
            let f32_centroids = Float32Array::from(proto.centroids.clone());
            let dimension = f32_centroids.len() / proto.lengths.len();
            Some(FixedSizeListArray::try_new_from_values(
                f32_centroids,
                dimension as i32,
            )?)
        } else {
            // We also use IvfModel to track the offsets/lengths of sub-index like HNSW
            // which does not have centroids.
            None
        };
        // We are not using offsets from the protobuf, which was the file offset in the
        // v1 index format. It will be deprecated soon.
        //
        // This new offset uses the row offset in the lance file.
        let offsets = match proto.offsets.len() {
            0 => proto
                .lengths
                .iter()
                .scan(0_usize, |state, &x| {
                    let old = *state;
                    *state += x as usize;
                    Some(old)
                })
                .collect_vec(),
            _ => proto.offsets.iter().map(|x| *x as usize).collect(),
        };
        assert_eq!(offsets.len(), proto.lengths.len());
        Ok(Self {
            centroids,
            offsets,
            lengths: proto.lengths,
            loss: proto.loss,
        })
    }
}

/// The IVF metadata stored in the Lance Schema
#[derive(Serialize, Deserialize, Debug)]
struct IvfMetadata {
    // The file position to store the protobuf binary of IVF metadata.
    pb_position: usize,
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use arrow_array::{Float32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance_core::datatypes::Schema;
    use lance_io::object_store::ObjectStore;
    use lance_table::format::SelfDescribingFileReader;
    use object_store::path::Path;

    use crate::pb;

    use super::*;

    #[test]
    fn test_ivf_find_rows() {
        let mut ivf = IvfModel::empty();
        ivf.add_partition(20);
        ivf.add_partition(50);

        assert_eq!(ivf.row_range(0), 0..20);
        assert_eq!(ivf.row_range(1), 20..70);
    }

    #[test]
    fn test_centroid_ef_must_cover_maximum_nprobes() {
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..16).map(|value| value as f32).collect::<Vec<_>>()),
            2,
        )
        .unwrap();
        let ivf = IvfModel::new(centroids, None);
        let query = Float32Array::from(vec![0.0, 1.0]);
        let error = ivf
            .find_partitions_with_centroid_index(
                &query,
                4,
                DistanceType::L2,
                Some(3),
                &CentroidIndexCache::default(),
            )
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(
            "centroid_ef must be >= maximum_nprobes, got centroid_ef=3, maximum_nprobes=4"
        ));
    }

    #[test]
    fn test_small_centroid_set_falls_back_to_exact_routing() {
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from((0..32).map(|value| value as f32).collect::<Vec<_>>()),
            2,
        )
        .unwrap();
        let ivf = IvfModel::new(centroids, None);
        let query = Float32Array::from(vec![9.0, 10.0]);
        let cache = CentroidIndexCache::default();

        for nprobes in [1, 3, 8] {
            let exact = ivf
                .find_partitions(&query, nprobes, DistanceType::L2)
                .unwrap();
            let routed = ivf
                .find_partitions_with_centroid_index(
                    &query,
                    nprobes,
                    DistanceType::L2,
                    Some(nprobes),
                    &cache,
                )
                .unwrap();
            assert_eq!(routed, exact);
        }
        assert!(cache.index.get().is_none());
    }

    #[test]
    fn test_centroid_hnsw_routing_recall_and_cache() {
        const NUM_CENTROIDS: usize = 128;
        const DIMENSION: usize = 8;
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(
                (0..NUM_CENTROIDS * DIMENSION)
                    .map(|i| ((i * 17 + i / DIMENSION * 13) % 101) as f32 / 100.0)
                    .collect::<Vec<_>>(),
            ),
            DIMENSION as i32,
        )
        .unwrap();
        let ivf = IvfModel::new(centroids, None);
        // Query an existing centroid so the nprobes=1 assertion is stable even though
        // parallel HNSW construction may choose different, equally valid graph edges.
        let query = ivf.centroid(42).unwrap();
        let cache = CentroidIndexCache::default();

        for nprobes in [1, 4, 16] {
            let (expected, _) = ivf
                .find_partitions(query.as_ref(), nprobes, DistanceType::L2)
                .unwrap();
            let (actual, distances) = ivf
                .find_partitions_with_hnsw(
                    query.as_ref(),
                    nprobes,
                    DistanceType::L2,
                    NUM_CENTROIDS,
                    &cache,
                )
                .unwrap();
            let expected = expected.values().iter().copied().collect::<HashSet<_>>();
            let matches = actual
                .values()
                .iter()
                .filter(|id| expected.contains(id))
                .count();
            let recall = matches as f32 / nprobes as f32;
            assert!(recall >= 0.5, "nprobes={nprobes}, recall={recall}");
            assert!(distances.values().windows(2).all(|pair| pair[0] <= pair[1]));
        }
        assert!(cache.index.get().is_some());
    }

    #[tokio::test]
    async fn test_write_and_load() {
        let mut ivf = IvfModel::empty();
        ivf.add_partition(20);
        ivf.add_partition(50);

        let object_store = ObjectStore::memory();
        let path = Path::from("/foo");
        let arrow_schema = ArrowSchema::new(vec![Field::new("a", DataType::Float32, true)]);
        let schema = Schema::try_from(&arrow_schema).unwrap();

        {
            let mut writer =
                V1FileWriter::try_new(&object_store, &path, schema.clone(), &Default::default())
                    .await
                    .unwrap();
            // Write some dummy data
            let batch = RecordBatch::try_new(
                Arc::new(arrow_schema),
                vec![Arc::new(Float32Array::from(vec![Some(1.0)]))],
            )
            .unwrap();
            writer.write(&[batch]).await.unwrap();
            ivf.write(&mut writer).await.unwrap();
            writer.finish().await.unwrap();
        }

        let reader = V1FileReader::try_new_self_described(&object_store, &path, None)
            .await
            .unwrap();
        assert!(reader.schema().metadata.contains_key(IVF_METADATA_KEY));

        let ivf2 = IvfModel::load(&reader).await.unwrap();
        assert_eq!(ivf, ivf2);
        assert_eq!(ivf2.num_partitions(), 2);
    }

    #[test]
    fn test_load_v1_format_ivf() {
        // in v1 format, the centroids are stored as a flat array in field `centroids`.
        let pb_ivf = pb::Ivf {
            centroids: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            lengths: vec![2, 2],
            offsets: vec![0, 2],
            centroids_tensor: None,
            loss: None,
        };

        let ivf = IvfModel::try_from(pb_ivf).unwrap();
        assert_eq!(ivf.num_partitions(), 2);
        assert_eq!(ivf.dimension(), 3);
        assert_eq!(ivf.centroids.as_ref().unwrap().len(), 2);
        assert_eq!(ivf.centroids.as_ref().unwrap().value_length(), 3);
    }

    #[test]
    fn test_centroids_array_getter() {
        use arrow_array::Float32Array;
        // two centroids, dim = 2
        let values = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let centroids = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        let ivf = IvfModel::new(centroids.clone(), None);
        let out = ivf.centroids_array().unwrap();

        // Validate that the returned array has expected structure
        assert_eq!(out.len(), centroids.len());
        assert_eq!(out.value_length(), centroids.value_length());

        // Validate centroid accessor returns correct values for the first partition
        let first = ivf.centroid(0).unwrap();
        let first_vals = first.as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(first_vals.len(), 2);
        assert_eq!(first_vals.value(0), 1.0);
        assert_eq!(first_vals.value(1), 2.0);
    }
}
