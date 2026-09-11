// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type, Int8Type, UInt8Type};
use arrow_array::{Array, FixedSizeListArray};
use lance_index::pb::VectorIndexDetails;
use lance_index::vector::ivf::storage::IvfModel;
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;
use sha2::{Digest, Sha256};

use crate::{Error, Result};

const FINGERPRINT_SIZE: usize = 32;

/// Content identity of the state used to route queries to IVF partitions.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CoarseQuantizerFingerprint([u8; FINGERPRINT_SIZE]);

impl CoarseQuantizerFingerprint {
    /// Compute an identity from the final IVF model that will be written.
    ///
    /// Unsupported or incomplete models return `None`, which makes readers use
    /// per-segment routing.
    pub fn from_model(ivf: &IvfModel, metric: DistanceType) -> Option<Self> {
        let centroids = ivf.centroids_array()?;
        if centroids.null_count() != 0 || centroids.values().null_count() != 0 {
            return None;
        }

        let mut hasher = Sha256::new();
        hasher.update(b"lance.coarse_quantizer.v1");
        hasher.update([metric_tag(metric)]);
        hasher.update((centroids.len() as u64).to_le_bytes());
        hasher.update((centroids.value_length() as u64).to_le_bytes());
        hash_centroids(&mut hasher, centroids)?;
        Some(Self(hasher.finalize().into()))
    }

    /// Decode a validated identity from persisted index metadata.
    pub fn from_metadata(index: &IndexMetadata) -> Option<Self> {
        let details = index
            .index_details
            .as_ref()?
            .to_msg::<VectorIndexDetails>()
            .ok()?;
        let bytes: [u8; FINGERPRINT_SIZE] =
            details.coarse_quantizer_fingerprint?.try_into().ok()?;
        Some(Self(bytes))
    }

    pub fn to_bytes(self) -> Vec<u8> {
        self.0.to_vec()
    }
}

/// Replace the persisted identity while preserving all other vector details.
pub fn with_fingerprint(
    details: prost_types::Any,
    fingerprint: Option<CoarseQuantizerFingerprint>,
) -> Result<prost_types::Any> {
    let mut details = details.to_msg::<VectorIndexDetails>().map_err(|error| {
        Error::index(format!(
            "Failed to decode VectorIndexDetails while setting coarse quantizer fingerprint: {error}"
        ))
    })?;
    details.coarse_quantizer_fingerprint = fingerprint.map(|value| value.to_bytes());
    prost_types::Any::from_msg(&details).map_err(|error| {
        Error::index(format!(
            "Failed to encode VectorIndexDetails with coarse quantizer fingerprint: {error}"
        ))
    })
}

/// Return the common identity only when every segment has the same valid value.
pub fn common_fingerprint(indices: &[IndexMetadata]) -> Option<CoarseQuantizerFingerprint> {
    let mut indices = indices.iter();
    let expected = CoarseQuantizerFingerprint::from_metadata(indices.next()?)?;
    indices
        .all(|index| CoarseQuantizerFingerprint::from_metadata(index) == Some(expected))
        .then_some(expected)
}

fn metric_tag(metric: DistanceType) -> u8 {
    match metric {
        DistanceType::L2 => 1,
        DistanceType::Cosine => 2,
        DistanceType::Dot => 3,
        DistanceType::Hamming => 4,
    }
}

fn hash_centroids(hasher: &mut Sha256, centroids: &FixedSizeListArray) -> Option<()> {
    let values = centroids.values();
    match values.data_type() {
        arrow_schema::DataType::Float16 => {
            hasher.update([1]);
            for value in values.as_primitive::<Float16Type>().values() {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        arrow_schema::DataType::Float32 => {
            hasher.update([2]);
            for value in values.as_primitive::<Float32Type>().values() {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        arrow_schema::DataType::Float64 => {
            hasher.update([3]);
            for value in values.as_primitive::<Float64Type>().values() {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
        arrow_schema::DataType::UInt8 => {
            hasher.update([4]);
            hasher.update(values.as_primitive::<UInt8Type>().values().as_ref());
        }
        arrow_schema::DataType::Int8 => {
            hasher.update([5]);
            for value in values.as_primitive::<Int8Type>().values() {
                hasher.update(value.to_le_bytes());
            }
        }
        _ => return None,
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Float32Array, Float64Array};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::pb::VectorIndexDetails;
    use uuid::Uuid;

    use super::*;

    fn model_f32(values: Vec<f32>, dimension: i32) -> IvfModel {
        IvfModel::new(
            FixedSizeListArray::try_new_from_values(Float32Array::from(values), dimension).unwrap(),
            None,
        )
    }

    #[test]
    fn fingerprint_describes_final_routing_state() {
        let first = model_f32(vec![1.0, 2.0, 3.0, 4.0], 2);
        let same = model_f32(vec![1.0, 2.0, 3.0, 4.0], 2);
        let reordered = model_f32(vec![3.0, 4.0, 1.0, 2.0], 2);
        let reshaped = model_f32(vec![1.0, 2.0, 3.0, 4.0], 1);
        let f64_model = IvfModel::new(
            FixedSizeListArray::try_new_from_values(
                Float64Array::from(vec![1.0, 2.0, 3.0, 4.0]),
                2,
            )
            .unwrap(),
            None,
        );

        let fingerprint = CoarseQuantizerFingerprint::from_model(&first, DistanceType::L2).unwrap();
        assert_eq!(
            fingerprint,
            CoarseQuantizerFingerprint::from_model(&same, DistanceType::L2).unwrap()
        );
        assert_ne!(
            fingerprint,
            CoarseQuantizerFingerprint::from_model(&reordered, DistanceType::L2).unwrap()
        );
        assert_ne!(
            fingerprint,
            CoarseQuantizerFingerprint::from_model(&reshaped, DistanceType::L2).unwrap()
        );
        assert_ne!(
            fingerprint,
            CoarseQuantizerFingerprint::from_model(&first, DistanceType::Dot).unwrap()
        );
        assert_ne!(
            fingerprint,
            CoarseQuantizerFingerprint::from_model(&f64_model, DistanceType::L2).unwrap()
        );
    }

    #[test]
    fn metadata_decoder_fails_closed() {
        fn metadata(fingerprint: Option<Vec<u8>>) -> IndexMetadata {
            let details = VectorIndexDetails {
                coarse_quantizer_fingerprint: fingerprint,
                ..Default::default()
            };
            IndexMetadata {
                uuid: Uuid::new_v4(),
                name: "vector_idx".to_string(),
                fields: vec![0],
                covering_fields: vec![],
                dataset_version: 1,
                fragment_bitmap: None,
                index_details: Some(Arc::new(prost_types::Any::from_msg(&details).unwrap())),
                index_version: 1,
                created_at: None,
                base_id: None,
                files: None,
            }
        }

        assert!(CoarseQuantizerFingerprint::from_metadata(&metadata(None)).is_none());
        assert!(CoarseQuantizerFingerprint::from_metadata(&metadata(Some(vec![1; 31]))).is_none());
        assert!(CoarseQuantizerFingerprint::from_metadata(&metadata(Some(vec![1; 32]))).is_some());

        let mut malformed = metadata(None);
        malformed.index_details = Some(Arc::new(prost_types::Any {
            type_url: "type.googleapis.com/lance.index.VectorIndexDetails".to_string(),
            value: vec![0xff],
        }));
        assert!(CoarseQuantizerFingerprint::from_metadata(&malformed).is_none());
    }
}
