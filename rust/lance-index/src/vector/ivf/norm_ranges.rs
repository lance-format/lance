// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Descriptive norm statistics for current Float32 IVF_FLAT storage.

use arrow_array::{Array, FixedSizeListArray, cast::AsArray, types::Float32Type};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

/// Schema metadata on the small `index.idx` file, never the auxiliary file.
pub const NORM_RANGES_METADATA_KEY: &str = "lance:ivf:norm_ranges";

/// L2 norm extrema computed with ordinary f64 arithmetic over stored vectors.
///
/// These are descriptive statistics, not certified Float32 search bounds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, DeepSizeOf)]
pub struct VectorNormRange {
    /// Smallest stored vector norm, including zero vectors.
    pub min: f64,
    /// Largest stored vector norm.
    pub max: f64,
}

impl VectorNormRange {
    /// Compute complete extrema for a Float32 partition in its storage space.
    ///
    /// Empty partitions, unsupported types, nulls, and nonfinite vectors return
    /// `None`; omitting invalid rows would incorrectly describe a partial range.
    /// The caller must dispatch substantial computation off async worker threads.
    pub fn from_vectors(vectors: &FixedSizeListArray) -> Option<Self> {
        if vectors.is_empty() || vectors.null_count() != 0 || vectors.value_length() <= 0 {
            return None;
        }
        let values = vectors.values().as_primitive_opt::<Float32Type>()?;
        if values.null_count() != 0 {
            return None;
        }
        let mut range = Self {
            min: f64::INFINITY,
            max: 0.0,
        };
        for row in 0..vectors.len() {
            let start = vectors.value_offset(row) as usize;
            let end = start + vectors.value_length() as usize;
            let mut squared = 0.0;
            for &value in &values.values()[start..end] {
                if !value.is_finite() {
                    return None;
                }
                let value = f64::from(value);
                squared += value * value;
            }
            let norm = squared.sqrt();
            range.min = range.min.min(norm);
            range.max = range.max.max(norm);
        }
        Some(range)
    }
}

/// Versioned metadata bound to the index's metric, dimension and posting layout.
///
/// Each range describes actual stored vectors, including the Float32
/// normalization already applied by cosine writers. A null range means empty
/// or unavailable; a missing metadata key means an index without statistics.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, DeepSizeOf)]
pub struct IvfNormRanges {
    version: u32,
    dimension: usize,
    metric: String,
    partition_lengths: Vec<u32>,
    ranges: Vec<Option<VectorNormRange>>,
}

impl IvfNormRanges {
    /// Bind computed statistics to the actual auxiliary-file partition rows.
    pub fn new(
        dimension: usize,
        metric: DistanceType,
        partition_lengths: Vec<u32>,
        ranges: Vec<Option<VectorNormRange>>,
    ) -> Result<Self> {
        let value = Self {
            version: 1,
            dimension,
            metric: metric.to_string(),
            partition_lengths,
            ranges,
        };
        value.validate(dimension, metric, &value.partition_lengths)?;
        Ok(value)
    }

    /// Parse optional schema metadata and validate bindings for known versions.
    ///
    /// Unknown versions disable these optional statistics without blocking index
    /// reads, and extra fields are ignored. Missing or malformed versions and
    /// corrupt version-1 data remain errors rather than silently losing coverage.
    pub fn from_json(
        value: Option<&str>,
        dimension: usize,
        metric: DistanceType,
        partition_lengths: &[u32],
    ) -> Result<Option<Self>> {
        let Some(value) = value else {
            return Ok(None);
        };
        #[derive(Deserialize)]
        struct VersionHeader {
            version: u64,
        }
        let invalid = |error| {
            Error::corrupt_file_named(
                NORM_RANGES_METADATA_KEY,
                format!("invalid norm range JSON: {error}"),
            )
        };
        // A future version need not retain the version-1 shape. Decode only its
        // version before interpreting any binding or range fields.
        let header: VersionHeader = serde_json::from_str(value).map_err(invalid)?;
        if header.version != 1 {
            return Ok(None);
        }
        let parsed: Self = serde_json::from_str(value).map_err(invalid)?;
        parsed.validate(dimension, metric, partition_lengths)?;
        Ok(Some(parsed))
    }

    /// Validate against independently loaded index and auxiliary metadata.
    pub fn validate(
        &self,
        dimension: usize,
        metric: DistanceType,
        partition_lengths: &[u32],
    ) -> Result<()> {
        let invalid = |message| Error::corrupt_file_named(NORM_RANGES_METADATA_KEY, message);
        if self.version != 1 {
            return Err(invalid(format!(
                "unsupported norm range version {}",
                self.version
            )));
        }
        if dimension == 0
            || self.dimension != dimension
            || self.metric != metric.to_string()
            || metric == DistanceType::Hamming
        {
            return Err(invalid(format!(
                "norm range binding dimension={}, metric={} differs from dimension={dimension}, metric={metric}",
                self.dimension, self.metric
            )));
        }
        if self.partition_lengths != partition_lengths
            || self.ranges.len() != partition_lengths.len()
        {
            return Err(invalid(format!(
                "norm range posting layout differs: {} ranges, {} recorded lengths, {} actual partitions",
                self.ranges.len(),
                self.partition_lengths.len(),
                partition_lengths.len()
            )));
        }
        for (partition, (range, &length)) in self.ranges.iter().zip(partition_lengths).enumerate() {
            if let Some(range) = range
                && (length == 0
                    || !range.min.is_finite()
                    || !range.max.is_finite()
                    || range.min < 0.0
                    || range.min > range.max)
            {
                return Err(invalid(format!(
                    "invalid norm range for partition {partition}, rows={length}: {range:?}"
                )));
            }
        }
        Ok(())
    }

    /// Return a partition's range, or `None` for empty/unavailable/out-of-range.
    pub fn get(&self, partition: usize) -> Option<VectorNormRange> {
        self.ranges.get(partition).copied().flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float32Array;
    use lance_arrow::FixedSizeListArrayExt;
    use rstest::rstest;

    #[test]
    fn test_full_extrema_and_sliced_vectors() {
        let values = Float32Array::from(vec![3.0, 4.0, 0.0, 0.0, -5.0, 12.0]);
        let vectors = FixedSizeListArray::try_new_from_values(values, 2).unwrap();
        assert_eq!(
            VectorNormRange::from_vectors(&vectors),
            Some(VectorNormRange {
                min: 0.0,
                max: 13.0
            })
        );
        assert_eq!(
            VectorNormRange::from_vectors(&vectors.slice(1, 2)),
            Some(VectorNormRange {
                min: 0.0,
                max: 13.0
            })
        );
        assert_eq!(VectorNormRange::from_vectors(&vectors.slice(0, 0)), None);
        let huge =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![f32::MAX; 2]), 2)
                .unwrap();
        assert!(
            VectorNormRange::from_vectors(&huge)
                .unwrap()
                .max
                .is_finite()
        );
    }

    #[rstest]
    #[case(None)]
    #[case(Some(f32::NAN))]
    #[case(Some(f32::INFINITY))]
    fn test_incomplete_vectors_are_unavailable(#[case] value: Option<f32>) {
        let vectors =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![Some(1.0), value]), 2)
                .unwrap();
        assert_eq!(VectorNormRange::from_vectors(&vectors), None);
    }

    #[test]
    fn test_roundtrip_missing_and_unavailable() {
        let metadata = IvfNormRanges::new(
            2,
            DistanceType::L2,
            vec![2, 0, 1],
            vec![Some(VectorNormRange { min: 0.0, max: 5.0 }), None, None],
        )
        .unwrap();
        let json = serde_json::to_string(&metadata).unwrap();
        assert_eq!(
            IvfNormRanges::from_json(Some(&json), 2, DistanceType::L2, &[2, 0, 1]).unwrap(),
            Some(metadata)
        );
        assert_eq!(
            IvfNormRanges::from_json(None, 2, DistanceType::L2, &[2]).unwrap(),
            None
        );
    }

    #[rstest]
    #[case::malformed_version("version", serde_json::json!("1"))]
    #[case::metric("metric", serde_json::json!("dot"))]
    #[case::dimension("dimension", serde_json::json!(3))]
    #[case::lengths("partition_lengths", serde_json::json!([3]))]
    #[case::range_count("ranges", serde_json::json!([]))]
    #[case::negative("ranges", serde_json::json!([{ "min": -1, "max": 2 }]))]
    #[case::reversed("ranges", serde_json::json!([{ "min": 3, "max": 2 }]))]
    fn test_invalid_bindings(#[case] field: &str, #[case] value: serde_json::Value) {
        let mut metadata = serde_json::json!({"version":1,"dimension":2,"metric":"l2","partition_lengths":[2],"ranges":[{"min":0,"max":5}]});
        metadata[field] = value;
        let error =
            IvfNormRanges::from_json(Some(&metadata.to_string()), 2, DistanceType::L2, &[2])
                .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("norm range"));
    }

    #[test]
    fn test_extra_version1_fields_are_ignored() {
        let json = r#"{"version":1,"dimension":2,"metric":"l2","partition_lengths":[2],
            "future_note":"ignored","ranges":[{"min":0,"max":5,"extra":true}]}"#;
        let parsed = IvfNormRanges::from_json(Some(json), 2, DistanceType::L2, &[2])
            .unwrap()
            .unwrap();
        assert_eq!(parsed.get(0), Some(VectorNormRange { min: 0.0, max: 5.0 }));
    }

    #[test]
    fn test_unknown_version_with_different_shape_disables_statistics() {
        let json = r#"{"version":2,"dimension":"future encoding","ranges":{"new":"shape"}}"#;
        assert_eq!(
            IvfNormRanges::from_json(Some(json), 2, DistanceType::L2, &[2]).unwrap(),
            None
        );
    }

    #[rstest]
    #[case::missing(r#"{"ranges":[]}"#)]
    #[case::null(r#"{"version":null}"#)]
    #[case::fractional(r#"{"version":1.5}"#)]
    #[case::incomplete_version1(r#"{"version":1}"#)]
    fn test_invalid_version_or_known_shape_remains_corrupt(#[case] json: &str) {
        let error = IvfNormRanges::from_json(Some(json), 2, DistanceType::L2, &[2]).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("invalid norm range JSON"));
    }
}
