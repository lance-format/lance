// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Centroid-based selection of the initial IVF probe budget.
//!
//! Auto probing is an empirical heuristic, not a recall guarantee. The initial
//! budget does not limit subsequent probing when filters leave fewer than k rows.
//! `LANCE_AUTO_PROBE_MARGIN` overrides the nonnegative relative distance margin;
//! `LANCE_AUTO_MIN_INITIAL_NPROBES` and `LANCE_AUTO_MAX_INITIAL_NPROBES` override
//! the positive learned floor and cap. Each override replaces only that profile
//! field; the resulting floor must not exceed the resulting cap. Override both
//! bounds when an individual change would conflict with the other profile bound.
//! The caller minimum takes precedence over the learned cap, while the caller
//! maximum and available candidate count limit the final initial budget.
//! Explicit fixed nprobes bypasses both the heuristic and these overrides.
//! Only ordinary Float32 IVF_FLAT queries using L2 or cosine with k <= 100
//! use this profile. Dot, Hamming, other index types, and explicitly bounded
//! Auto queries retain their existing heuristic and ignore these overrides.
//! Dot profiles are deferred until their cost is validated across datasets.
//! No extra index statistics or file-format changes are needed.

use std::env;

use arrow_array::{Array, cast::AsArray};
use arrow_schema::DataType;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use lance_index::vector::{
    Query, VectorIndex, quantizer::QuantizationType, v3::subindex::SubIndexType,
};
use lance_linalg::distance::DistanceType;

const MARGIN_ENV: &str = "LANCE_AUTO_PROBE_MARGIN";
const MIN_INITIAL_NPROBES_ENV: &str = "LANCE_AUTO_MIN_INITIAL_NPROBES";
const MAX_INITIAL_NPROBES_ENV: &str = "LANCE_AUTO_MAX_INITIAL_NPROBES";

/// Select the probing behavior once, before interpreting experimental overrides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum AutoProbePolicy {
    Fixed,
    Legacy,
    Adaptive(AutoProbeConfig),
}

impl AutoProbePolicy {
    pub(super) fn from_env(
        query: &Query,
        index: &dyn VectorIndex,
        vector_type: &DataType,
    ) -> DataFusionResult<Self> {
        Self::select_with_config(query, index, vector_type, AutoProbeConfig::from_env)
    }

    pub(super) fn select_with_config(
        query: &Query,
        index: &dyn VectorIndex,
        vector_type: &DataType,
        read_config: impl FnOnce(&Query, DistanceType) -> DataFusionResult<Option<AutoProbeConfig>>,
    ) -> DataFusionResult<Self> {
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return Ok(Self::Fixed);
        }
        // Legacy IVF indices do not expose sub-index metadata. Keep their
        // original probing without calling those unsupported methods.
        if !index.supports_prepared_partition_search() {
            return Ok(Self::Legacy);
        }
        if query.maximum_nprobes.is_some()
            || query.key.data_type() != &DataType::Float32
            || query.key.null_count() != 0
            || !matches!(vector_type, DataType::FixedSizeList(item, dimension)
                if item.data_type() == &DataType::Float32 && *dimension as usize == query.key.len())
            || !matches!(
                index.sub_index_type(),
                (SubIndexType::Flat, QuantizationType::Flat)
            )
            || !matches!(index.metric_type(), DistanceType::L2 | DistanceType::Cosine)
            || query.k > 100
            || query.refine_factor.is_some_and(|factor| factor > 1)
            || !query
                .key
                .as_primitive::<arrow_array::types::Float32Type>()
                .values()
                .iter()
                .all(|x| x.is_finite())
        {
            return Ok(Self::Legacy);
        }
        Ok(read_config(query, index.metric_type())?.map_or(Self::Fixed, Self::Adaptive))
    }

    pub(super) fn apply(self, query: &mut Query, distances: &[f32], metric: DistanceType) {
        match self {
            Self::Fixed => {}
            Self::Legacy => apply_legacy_probes(query, distances),
            Self::Adaptive(config) => config.apply(query, distances, metric),
        }
    }
}

/// Preserve the pre-experiment f32 heuristic, including signed distances and
/// overflow behavior. Available partitions are clipped by the search operators.
fn apply_legacy_probes(query: &mut Query, distances: &[f32]) {
    let selected = distances.first().map_or(0, |nearest| {
        let factor = match query.k {
            ..=1 => 0.6,
            2..=10 => 7.0,
            _ => 81.0,
        };
        let threshold = *nearest * factor;
        distances.partition_point(|distance| *distance <= threshold)
    });
    query.minimum_nprobes = query.minimum_nprobes.max(selected);
    if let Some(maximum) = query.maximum_nprobes {
        query.minimum_nprobes = query.minimum_nprobes.min(maximum);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct AutoProbeConfig {
    pub(super) min_initial_nprobes: usize,
    pub(super) margin: f32,
    pub(super) max_initial_nprobes: Option<usize>,
}

impl Default for AutoProbeConfig {
    fn default() -> Self {
        Self {
            min_initial_nprobes: 1,
            margin: 0.0,
            max_initial_nprobes: None,
        }
    }
}

impl AutoProbeConfig {
    pub(super) fn from_env(query: &Query, metric: DistanceType) -> DataFusionResult<Option<Self>> {
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return Ok(None);
        }
        if !matches!(metric, DistanceType::L2 | DistanceType::Cosine) {
            return Ok(Some(Self::default()));
        }
        fn read_override(name: &str) -> DataFusionResult<Option<String>> {
            match env::var(name) {
                Ok(value) => Ok(Some(value)),
                Err(env::VarError::NotPresent) => Ok(None),
                Err(error) => Err(DataFusionError::Execution(format!(
                    "invalid {name}: {error}"
                ))),
            }
        }
        let margin = read_override(MARGIN_ENV)?;
        let minimum = read_override(MIN_INITIAL_NPROBES_ENV)?;
        let maximum = read_override(MAX_INITIAL_NPROBES_ENV)?;
        Self::parse(
            query,
            metric,
            margin.as_deref(),
            minimum.as_deref(),
            maximum.as_deref(),
        )
    }

    fn parse(
        query: &Query,
        metric: DistanceType,
        margin: Option<&str>,
        minimum: Option<&str>,
        maximum: Option<&str>,
    ) -> DataFusionResult<Option<Self>> {
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return Ok(None);
        }
        if !matches!(metric, DistanceType::L2 | DistanceType::Cosine) {
            return Ok(Some(Self::default()));
        }
        let bucket = match query.k {
            ..=1 => 0,
            2..=10 => 1,
            _ => 2,
        };
        // Profiles depend only on metric and k; every index uses the same values.
        let (default_margin, default_minimum, cap) = match metric {
            DistanceType::L2 => (
                [0.2175, 0.265, 0.33][bucket],
                [5, 6, 11][bucket],
                [19, 24, 38][bucket],
            ),
            DistanceType::Cosine => (
                [0.235, 0.2875, 0.38][bucket],
                [3, 8, 7][bucket],
                [50, 77, 106][bucket],
            ),
            _ => return Ok(Some(Self::default())),
        };
        let config = Self {
            min_initial_nprobes: default_minimum,
            margin: default_margin,
            max_initial_nprobes: Some(cap),
        };
        config.with_overrides(margin, minimum, maximum).map(Some)
    }

    fn with_overrides(
        mut self,
        margin: Option<&str>,
        minimum: Option<&str>,
        maximum: Option<&str>,
    ) -> DataFusionResult<Self> {
        fn positive_override(name: &str, value: Option<&str>) -> DataFusionResult<Option<usize>> {
            value
                .map(|value| {
                    value
                        .parse::<usize>()
                        .ok()
                        .filter(|value| *value > 0)
                        .ok_or_else(|| {
                            DataFusionError::Execution(format!(
                                "invalid {name} value {value:?}: expected a positive integer"
                            ))
                        })
                })
                .transpose()
        }
        if let Some(value) = margin {
            self.margin = value
                .parse::<f32>()
                .ok()
                .filter(|margin| margin.is_finite() && *margin >= 0.0)
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "invalid {MARGIN_ENV} value {value:?}: expected a finite number >= 0"
                    ))
                })?;
        }
        if let Some(minimum) = positive_override(MIN_INITIAL_NPROBES_ENV, minimum)? {
            self.min_initial_nprobes = minimum;
        }
        if let Some(maximum) = positive_override(MAX_INITIAL_NPROBES_ENV, maximum)? {
            self.max_initial_nprobes = Some(maximum);
        }
        if let Some(maximum) = self.max_initial_nprobes
            && self.min_initial_nprobes > maximum
        {
            return Err(DataFusionError::Execution(format!(
                "invalid Auto probe interval: {MIN_INITIAL_NPROBES_ENV} effective minimum {} exceeds {MAX_INITIAL_NPROBES_ENV} effective maximum {maximum}",
                self.min_initial_nprobes
            )));
        }
        Ok(self)
    }

    /// Select an initial prefix of sorted centroid distances, honoring caller bounds.
    ///
    /// L2 and normalized-cosine routing use squared L2 distances. At zero
    /// distance only best-distance ties qualify; the caller minimum still applies.
    /// f64 arithmetic avoids overflow when subtracting finite f32 distances or
    /// applying a large finite margin.
    pub(super) fn apply(self, query: &mut Query, distances: &[f32], metric: DistanceType) {
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return;
        }
        if !matches!(metric, DistanceType::L2 | DistanceType::Cosine) {
            apply_legacy_probes(query, distances);
            return;
        }
        let selected = match distances.first().copied() {
            Some(nearest) if nearest.is_finite() => {
                let nearest = f64::from(nearest);
                let allowed_gap = f64::from(self.margin) * nearest;
                distances.partition_point(|distance| {
                    distance.is_finite() && f64::from(*distance) - nearest <= allowed_gap
                })
            }
            // No finite nearest distance is available to estimate a relative
            // gap. Leave the candidate budget unconstrained by distance pruning.
            Some(_) => distances.len(),
            None => 0,
        };
        let selected = selected
            .max(self.min_initial_nprobes)
            .min(self.max_initial_nprobes.unwrap_or(distances.len()));
        query.minimum_nprobes = query
            .minimum_nprobes
            .max(selected)
            .min(query.maximum_nprobes.unwrap_or(distances.len()))
            .min(distances.len());
        // Keep maximum_nprobes unchanged: late search needs the remaining
        // candidates when filters or deletions exhaust the initial budget.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float32Array;
    use rstest::rstest;
    use std::sync::Arc;

    fn query() -> Query {
        Query {
            column: "vector".to_string(),
            key: Arc::new(Float32Array::from(vec![1.0, 0.0])),
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: None,
            use_index: true,
            query_parallelism: 0,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
        }
    }

    #[rstest]
    #[case::l2_top1(DistanceType::L2, 1, 0.2175, 5, 19)]
    #[case::l2_top10_lower_boundary(DistanceType::L2, 2, 0.265, 6, 24)]
    #[case::l2_top10_upper_boundary(DistanceType::L2, 10, 0.265, 6, 24)]
    #[case::l2_top100_lower_boundary(DistanceType::L2, 11, 0.33, 11, 38)]
    #[case::l2_top100(DistanceType::L2, 100, 0.33, 11, 38)]
    #[case::cosine_top1(DistanceType::Cosine, 1, 0.235, 3, 50)]
    #[case::cosine_top10_lower_boundary(DistanceType::Cosine, 2, 0.2875, 8, 77)]
    #[case::cosine_top10_upper_boundary(DistanceType::Cosine, 10, 0.2875, 8, 77)]
    #[case::cosine_top100_lower_boundary(DistanceType::Cosine, 11, 0.38, 7, 106)]
    #[case::cosine_top100(DistanceType::Cosine, 100, 0.38, 7, 106)]
    fn test_auto_probe_metric_profiles(
        #[case] metric: DistanceType,
        #[case] k: usize,
        #[case] margin: f32,
        #[case] minimum: usize,
        #[case] maximum: usize,
    ) {
        let mut query = query();
        query.k = k;
        let config = AutoProbeConfig::parse(&query, metric, None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            config,
            AutoProbeConfig {
                min_initial_nprobes: minimum,
                margin,
                max_initial_nprobes: Some(maximum),
            }
        );

        let mut distances = vec![f32::MAX; maximum + 1];
        distances[0] = 1.0;
        config.apply(&mut query, &distances, metric);
        assert_eq!(query.minimum_nprobes, minimum);

        distances.fill(1.0);
        config.apply(&mut query, &distances, metric);
        assert_eq!(query.minimum_nprobes, maximum);
        assert_eq!(query.maximum_nprobes, None);
    }

    #[rstest]
    #[case::top1_positive(1, &[4.0, 4.0, 5.0], 1, None, 1)]
    #[case::top1_zero_ties(1, &[0.0, 0.0, 1.0], 1, None, 2)]
    #[case::top10_positive(10, &[1.0, 7.0, 8.0], 1, None, 2)]
    #[case::top10_zero_ties(10, &[0.0, 0.0, 1.0], 1, None, 2)]
    #[case::top100_positive(100, &[1.0, 81.0, 82.0], 1, None, 2)]
    #[case::above_learned_cap(10, &[1.0, 1.0, 1.0, 1.0, 1.0], 1, None, 5)]
    #[case::maximum(10, &[1.0, 1.0, 1.0], 1, Some(2), 2)]
    #[case::minimum(1, &[1.0, 1.0], 4, None, 4)]
    fn test_uncalibrated_metrics_preserve_probe_budget(
        #[values(DistanceType::Hamming, DistanceType::Dot)] metric: DistanceType,
        #[case] k: usize,
        #[case] distances: &[f32],
        #[case] minimum: usize,
        #[case] maximum: Option<usize>,
        #[case] expected: usize,
    ) {
        let mut query = query();
        query.k = k;
        query.minimum_nprobes = minimum;
        query.maximum_nprobes = maximum;
        // Calibrated Auto overrides must neither reject nor change these metrics.
        assert_eq!(
            AutoProbeConfig::parse(&query, metric, Some("invalid"), Some("0"), Some("0")).unwrap(),
            Some(AutoProbeConfig::default()),
        );
        AutoProbeConfig {
            min_initial_nprobes: 4,
            margin: 80.0,
            max_initial_nprobes: Some(4),
        }
        .apply(&mut query, distances, metric);
        assert_eq!(query.minimum_nprobes, expected);
        assert_eq!(query.maximum_nprobes, maximum);
    }

    #[rstest]
    #[case::l2(DistanceType::L2, &[4.0, 6.0, 6.25], 0.5, 2)]
    #[case::cosine(DistanceType::Cosine, &[0.25, 0.5, 0.75], 1.0, 2)]
    #[case::large_l2_gap(DistanceType::L2, &[f32::MAX / 2.0, f32::MAX], 1.0, 2)]
    #[case::zero_l2(DistanceType::L2, &[0.0, 0.0, 0.25], 80.0, 2)]
    #[case::zero_margin(DistanceType::L2, &[1.0, 1.0, 2.0], 0.0, 2)]
    #[case::empty(DistanceType::L2, &[], 1.0, 0)]
    #[case::nonfinite_nearest(DistanceType::L2, &[f32::INFINITY, f32::INFINITY], 1.0, 2)]
    #[case::nan_nearest(DistanceType::L2, &[f32::NAN], 1.0, 1)]
    #[case::nonfinite_tail(DistanceType::L2, &[1.0, 2.0, f32::INFINITY], 1.0, 2)]
    fn test_auto_probe_gap(
        #[case] metric: DistanceType,
        #[case] distances: &[f32],
        #[case] margin: f32,
        #[case] expected: usize,
    ) {
        let mut query = query();
        AutoProbeConfig {
            min_initial_nprobes: 1,
            margin,
            max_initial_nprobes: None,
        }
        .apply(&mut query, distances, metric);
        assert_eq!(query.minimum_nprobes, expected);
        assert_eq!(query.maximum_nprobes, None);
    }

    #[rstest]
    #[case::caller_minimum(1, 10.0, 4, None, Some(2), 4)]
    #[case::caller_maximum(1, 10.0, 1, Some(3), None, 3)]
    #[case::initial_cap(1, 10.0, 1, None, Some(2), 2)]
    #[case::fixed(4, 10.0, 3, Some(3), Some(4), 3)]
    #[case::minimum_exceeds_candidates(1, 10.0, 10, None, None, 5)]
    #[case::learned_floor(4, 0.0, 1, None, None, 4)]
    #[case::learned_floor_limited_by_caller(4, 0.0, 1, Some(2), None, 2)]
    #[case::learned_floor_exceeds_candidates(8, 0.0, 1, None, None, 5)]
    #[case::selected_above_floor(2, 10.0, 1, None, Some(4), 4)]
    #[case::learned_floor_equals_cap(3, 0.0, 1, None, Some(3), 3)]
    fn test_auto_probe_bounds(
        #[case] learned_minimum: usize,
        #[case] margin: f32,
        #[case] minimum: usize,
        #[case] maximum: Option<usize>,
        #[case] cap: Option<usize>,
        #[case] expected: usize,
    ) {
        let mut query = query();
        query.minimum_nprobes = minimum;
        query.maximum_nprobes = maximum;
        AutoProbeConfig {
            min_initial_nprobes: learned_minimum,
            margin,
            max_initial_nprobes: cap,
        }
        .apply(&mut query, &[1.0, 2.0, 3.0, 4.0, 5.0], DistanceType::L2);
        assert_eq!(query.minimum_nprobes, expected);
        assert_eq!(query.maximum_nprobes, maximum);
    }

    #[rstest]
    fn test_auto_probe_learned_floor_matches_caller_floor(
        #[values(4, 16, 32)] minimum: usize,
        #[values(0.0, 5.0, 30.0)] margin: f32,
    ) {
        let distances = (1..=64).map(|distance| distance as f32).collect::<Vec<_>>();
        let mut learned = query();
        let mut explicit = query();
        explicit.minimum_nprobes = minimum;
        AutoProbeConfig {
            min_initial_nprobes: minimum,
            margin,
            max_initial_nprobes: Some(48),
        }
        .apply(&mut learned, &distances, DistanceType::L2);
        AutoProbeConfig {
            min_initial_nprobes: 1,
            margin,
            max_initial_nprobes: Some(48),
        }
        .apply(&mut explicit, &distances, DistanceType::L2);
        assert_eq!(learned.minimum_nprobes, explicit.minimum_nprobes);
        assert!(learned.minimum_nprobes >= minimum);
        assert_eq!(learned.maximum_nprobes, None);
    }

    #[rstest]
    #[case::invalid_margin(Some("no"), None, None, MARGIN_ENV)]
    #[case::negative_margin(Some("-1"), None, None, MARGIN_ENV)]
    #[case::nan_margin(Some("NaN"), None, None, MARGIN_ENV)]
    #[case::infinite_margin(Some("inf"), None, None, MARGIN_ENV)]
    #[case::zero_floor(None, Some("0"), None, MIN_INITIAL_NPROBES_ENV)]
    #[case::negative_floor(None, Some("-1"), None, MIN_INITIAL_NPROBES_ENV)]
    #[case::invalid_floor(None, Some("no"), None, MIN_INITIAL_NPROBES_ENV)]
    #[case::nan_floor(None, Some("NaN"), None, MIN_INITIAL_NPROBES_ENV)]
    #[case::infinite_floor(None, Some("inf"), None, MIN_INITIAL_NPROBES_ENV)]
    #[case::zero_cap(None, None, Some("0"), MAX_INITIAL_NPROBES_ENV)]
    #[case::negative_cap(None, None, Some("-1"), MAX_INITIAL_NPROBES_ENV)]
    #[case::invalid_cap(None, None, Some("no"), MAX_INITIAL_NPROBES_ENV)]
    #[case::nan_cap(None, None, Some("NaN"), MAX_INITIAL_NPROBES_ENV)]
    #[case::infinite_cap(None, None, Some("inf"), MAX_INITIAL_NPROBES_ENV)]
    fn test_auto_probe_invalid_overrides(
        #[values(DistanceType::L2, DistanceType::Cosine)] metric: DistanceType,
        #[case] margin: Option<&str>,
        #[case] minimum: Option<&str>,
        #[case] maximum: Option<&str>,
        #[case] name: &str,
    ) {
        let error = AutoProbeConfig::parse(&query(), metric, margin, minimum, maximum).unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(error.to_string().contains(name));
        let mut fixed = query();
        fixed.maximum_nprobes = Some(fixed.minimum_nprobes);
        assert_eq!(
            AutoProbeConfig::parse(&fixed, metric, margin, minimum, maximum).unwrap(),
            None
        );
    }

    #[rstest]
    #[case::minimum_above_default_cap(Some("9"), None)]
    #[case::maximum_below_default_floor(None, Some("2"))]
    #[case::inverted_overrides(Some("8"), Some("4"))]
    fn test_auto_probe_inverted_interval(
        #[case] minimum: Option<&str>,
        #[case] maximum: Option<&str>,
    ) {
        let config = AutoProbeConfig {
            min_initial_nprobes: 4,
            margin: 0.5,
            max_initial_nprobes: Some(8),
        };
        let error = config.with_overrides(None, minimum, maximum).unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        let message = error.to_string();
        assert!(message.contains(MIN_INITIAL_NPROBES_ENV));
        assert!(message.contains(MAX_INITIAL_NPROBES_ENV));
        assert!(message.contains("exceeds"));
    }

    #[test]
    fn test_auto_probe_partial_overrides_preserve_profile_fields() {
        let config = AutoProbeConfig {
            min_initial_nprobes: 4,
            margin: 0.5,
            max_initial_nprobes: Some(8),
        };
        let floor_only = config.with_overrides(None, Some("6"), None).unwrap();
        assert_eq!(floor_only.min_initial_nprobes, 6);
        assert_eq!(floor_only.margin, config.margin);
        assert_eq!(floor_only.max_initial_nprobes, config.max_initial_nprobes);
        let cap_only = config.with_overrides(None, None, Some("6")).unwrap();
        assert_eq!(cap_only.min_initial_nprobes, config.min_initial_nprobes);
        assert_eq!(cap_only.margin, config.margin);
        assert_eq!(cap_only.max_initial_nprobes, Some(6));
        let both = config.with_overrides(None, Some("1"), Some("2")).unwrap();
        assert_eq!(both.min_initial_nprobes, 1);
        assert_eq!(both.max_initial_nprobes, Some(2));
    }

    #[test]
    fn test_auto_probe_valid_overrides() {
        let config = AutoProbeConfig::parse(
            &query(),
            DistanceType::Cosine,
            Some("0"),
            Some("4"),
            Some("7"),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            config,
            AutoProbeConfig {
                min_initial_nprobes: 4,
                margin: 0.0,
                max_initial_nprobes: Some(7)
            }
        );
    }

    #[rstest]
    fn test_auto_probe_margin_monotonicity(
        #[values(DistanceType::L2, DistanceType::Cosine)] metric: DistanceType,
    ) {
        let distances = &[1.0, 2.0, 3.0, 4.0];
        let mut previous = 0;
        for margin in [0.0, 0.25, 0.5, 1.0, 6.0, 80.0] {
            let mut query = query();
            AutoProbeConfig {
                min_initial_nprobes: 1,
                margin,
                max_initial_nprobes: None,
            }
            .apply(&mut query, distances, metric);
            assert!(query.minimum_nprobes >= previous);
            previous = query.minimum_nprobes;
        }
    }

    #[rstest]
    #[case::hamming(DistanceType::Hamming, &[1.0, 2.0, 8.0])]
    #[case::dot_positive_inner_product(DistanceType::Dot, &[-3.0, -1.0, 0.0, 1.0])]
    #[case::dot_negative_inner_product(DistanceType::Dot, &[3.0, 4.0, 5.0, 6.0])]
    #[case::dot_zero_inner_product(DistanceType::Dot, &[1.0, 1.0, 2.0, 3.0])]
    fn test_uncalibrated_metrics_preserve_original_auto(
        #[case] metric: DistanceType,
        #[case] distances: &[f32],
        #[values(1, 10, 100)] k: usize,
    ) {
        let mut actual = query();
        actual.k = k;
        let mut expected = actual.clone();
        apply_legacy_probes(&mut expected, distances);
        let config = AutoProbeConfig::parse(&actual, metric, Some("invalid"), Some("0"), Some("0"))
            .unwrap()
            .unwrap();
        config.apply(&mut actual, distances, metric);
        assert_eq!(actual.minimum_nprobes, expected.minimum_nprobes);
        assert_eq!(actual.maximum_nprobes, expected.maximum_nprobes);
    }
}
