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
//! Queries using L2, cosine, or dot with k <= 100000 use these profiles across IVF
//! index types, vector types, and refinement factors. The policy selects partitions
//! from centroid distances independently of query values, quantization, or the
//! search within each partition.
//! Explicit maximum bounds limit adaptive probing without disabling it.
//! Larger k and Hamming retain their existing heuristic and ignore these overrides.
//! Dot uses the magnitude of the best centroid inner product to scale its gap,
//! rather than the signed `1 - dot` distance. Corpus and query norms are preserved.
//! No extra index statistics or file-format changes are needed.

use std::env;

use datafusion::error::{DataFusionError, Result as DataFusionResult};
use lance_index::vector::{Query, VectorIndex};
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
    pub(super) fn from_env(query: &Query, index: &dyn VectorIndex) -> DataFusionResult<Self> {
        Self::select_with_config(query, index, AutoProbeConfig::from_env)
    }

    pub(super) fn select_with_config(
        query: &Query,
        index: &dyn VectorIndex,
        read_config: impl FnOnce(&Query, DistanceType) -> DataFusionResult<Option<AutoProbeConfig>>,
    ) -> DataFusionResult<Self> {
        query.validate_search_effort()?;
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return Ok(Self::Fixed);
        }
        if !matches!(
            index.metric_type(),
            DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot
        ) || query.k > 100_000
        {
            return Ok(Self::Legacy);
        }
        Ok(read_config(query, index.metric_type())?.map_or(Self::Fixed, Self::Adaptive))
    }

    pub(super) fn apply(self, query: &mut Query, distances: &[f32], metric: DistanceType) {
        let caller_minimum = query.minimum_nprobes;
        match self {
            Self::Fixed => {}
            Self::Legacy => apply_legacy_probes(query, distances),
            Self::Adaptive(config) => config.apply(query, distances, metric),
        }
        // Preserve the original budget exactly, including legacy rounding and
        // clipping behavior, when the new option is omitted or set to default.
        if query.search_effort == 0.5 {
            return;
        }
        let upper = distances
            .len()
            .min(query.maximum_nprobes.unwrap_or(distances.len()));
        if upper == 0 {
            query.minimum_nprobes = 0;
            return;
        }
        let lower = caller_minimum.max(1).min(upper);
        let auto = query.minimum_nprobes.clamp(lower, upper);
        let effort = query.search_effort;
        // Round-off near 0.5 can move the geometric budget past Auto. Bound
        // each half separately so increasing effort preserves the midpoint.
        query.minimum_nprobes = if effort == 0.0 {
            lower
        } else if effort == 1.0 {
            upper
        } else if effort < 0.5 {
            let budget = lower as f64 * (auto as f64 / lower as f64).powf(2.0 * effort);
            (budget.ceil() as usize).clamp(lower, auto)
        } else {
            let budget = auto as f64 * (upper as f64 / auto as f64).powf(2.0 * effort - 1.0);
            (budget.ceil() as usize).clamp(auto, upper)
        };
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
        if !matches!(
            metric,
            DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot
        ) {
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
        if !matches!(
            metric,
            DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot
        ) {
            return Ok(Some(Self::default()));
        }
        let bucket = match query.k {
            ..=1 => 0,
            2..=10 => 1,
            11..=100 => 2,
            101..=200 => 3,
            201..=500 => 4,
            501..=1000 => 5,
            1001..=10_000 => 6,
            _ => 7,
        };
        // Profiles depend only on metric and k; every index uses the same values.
        // Larger-k buckets are calibrated at their upper endpoints on separate
        // queries from the native evaluation (benchmarks/auto-ivf-large-k).
        // Each bucket uses its measured upper-endpoint profile; measurements do
        // not establish recall guarantees for every intermediate k. The policy
        // gate retains legacy probing above the largest calibrated endpoint.
        let (default_margin, default_minimum, cap) = match metric {
            DistanceType::L2 => (
                [0.2175, 0.265, 0.33, 0.375, 0.3725, 0.4175, 0.57, 0.75][bucket],
                [5, 6, 11, 15, 22, 29, 98, 463][bucket],
                [19, 24, 38, 56, 81, 93, 238, 704][bucket],
            ),
            DistanceType::Cosine => (
                [0.235, 0.2875, 0.38, 0.415, 0.45, 0.47, 0.6475, 0.58][bucket],
                [3, 8, 7, 18, 28, 34, 104, 738][bucket],
                [50, 77, 106, 156, 192, 248, 447, 958][bucket],
            ),
            // Calibrated on unnormalized Wiki-Cohere and DPR vectors. An initial
            // cap controls overscanning without limiting later filtered search.
            DistanceType::Dot => (
                [0.14, 0.055, 0.0625, 0.0675, 0.07, 0.0725, 0.0825, 0.21][bucket],
                [56, 144, 200, 211, 244, 279, 517, 1009][bucket],
                [112, 432, 768, 833, 971, 1092, 1824, 2528][bucket],
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
    /// Dot instead scales the gap by `abs(1 - nearest)`, the magnitude of the
    /// best inner product. Its budget is insensitive to positive query scaling
    /// apart from floating-point rounding, including when distances change sign.
    /// f64 arithmetic avoids overflow when subtracting finite f32 distances or
    /// applying a large finite margin.
    pub(super) fn apply(self, query: &mut Query, distances: &[f32], metric: DistanceType) {
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return;
        }
        if !matches!(
            metric,
            DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot
        ) {
            apply_legacy_probes(query, distances);
            return;
        }
        let selected = match distances.first().copied() {
            Some(nearest) if nearest.is_finite() => {
                let nearest = f64::from(nearest);
                let scale = if metric == DistanceType::Dot {
                    (1.0 - nearest).abs()
                } else {
                    nearest
                };
                let allowed_gap = f64::from(self.margin) * scale;
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
            search_effort: 0.5,
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
    #[case::minimum(0.0, 1)]
    #[case::below_auto(0.25, 4)]
    #[case::auto(0.5, 16)]
    #[case::above_auto(0.75, 32)]
    #[case::all(1.0, 64)]
    fn test_search_effort_interpolation(#[case] effort: f64, #[case] expected: usize) {
        let mut query = query();
        query.search_effort = effort;
        AutoProbePolicy::Adaptive(AutoProbeConfig {
            min_initial_nprobes: 16,
            margin: 0.0,
            max_initial_nprobes: Some(16),
        })
        .apply(&mut query, &[1.0; 64], DistanceType::L2);
        // The learned floor/cap must not be reapplied after interpolation.
        assert_eq!(query.minimum_nprobes, expected);
        assert_eq!(query.maximum_nprobes, None);
    }

    #[rstest]
    fn test_search_effort_bounds_and_monotonicity(
        #[values(0, 1, 7, 11, 64, 100)] caller_minimum: usize,
        #[values(None, Some(0), Some(16), Some(128))] maximum: Option<usize>,
        #[values(0, 1, 64)] available: usize,
    ) {
        let config = AutoProbeConfig {
            min_initial_nprobes: 25,
            margin: 0.0,
            max_initial_nprobes: Some(25),
        };
        let distances = vec![1.0; available];
        let upper = available.min(maximum.unwrap_or(available));
        let lower = caller_minimum.max(1).min(upper);
        let mut previous = lower;
        let mut efforts = (0..=100)
            .map(|step| f64::from(step) / 100.0)
            .collect::<Vec<_>>();
        efforts.extend([
            f64::from_bits(0.5_f64.to_bits() - 1),
            f64::from_bits(0.5_f64.to_bits() + 1),
        ]);
        efforts.sort_by(f64::total_cmp);
        for (step, effort) in efforts.into_iter().enumerate() {
            let mut query = query();
            query.minimum_nprobes = caller_minimum;
            query.maximum_nprobes = maximum;
            query.search_effort = effort;
            // Equal caller bounds select fixed probing, covered separately.
            if maximum == Some(caller_minimum) {
                continue;
            }
            AutoProbePolicy::Adaptive(config).apply(&mut query, &distances, DistanceType::L2);
            assert!((lower..=upper).contains(&query.minimum_nprobes));
            assert!(
                query.minimum_nprobes >= previous,
                "search_effort={effort}, budget={} must be at least previous budget={previous}",
                query.minimum_nprobes,
            );
            assert_eq!(query.maximum_nprobes, maximum);
            previous = query.minimum_nprobes;
            if step == 0 {
                assert_eq!(query.minimum_nprobes, lower);
            }
        }
        if maximum != Some(caller_minimum) {
            assert_eq!(previous, upper);
        }
    }

    #[rstest]
    fn test_search_effort_default_preserves_policy(
        #[values(0, 1, 100)] minimum: usize,
        #[values(None, Some(8))] maximum: Option<usize>,
        #[values(0, 16)] available: usize,
        #[values(AutoProbePolicy::Fixed, AutoProbePolicy::Legacy)] policy: AutoProbePolicy,
    ) {
        let mut actual = query();
        actual.minimum_nprobes = minimum;
        actual.maximum_nprobes = maximum;
        let mut expected = actual.clone();
        let distances = vec![1.0; available];
        if policy == AutoProbePolicy::Legacy {
            apply_legacy_probes(&mut expected, &distances);
        }
        policy.apply(&mut actual, &distances, DistanceType::Hamming);
        assert_eq!(actual.minimum_nprobes, expected.minimum_nprobes);
        assert_eq!(actual.maximum_nprobes, expected.maximum_nprobes);
    }

    #[rstest]
    #[case::negative(-0.01)]
    #[case::too_large(1.01)]
    #[case::nan(f64::NAN)]
    #[case::positive_infinity(f64::INFINITY)]
    #[case::negative_infinity(f64::NEG_INFINITY)]
    fn test_search_effort_invalid(#[case] effort: f64) {
        let mut query = query();
        query.search_effort = effort;
        let error = query.validate_search_effort().unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("search_effort must be finite and in [0, 1]")
        );
    }

    #[rstest]
    fn test_search_effort_equal_explicit_bounds(#[values(0.0, 0.25, 0.5, 0.75, 1.0)] effort: f64) {
        let mut query = query();
        query.minimum_nprobes = 8;
        query.maximum_nprobes = Some(8);
        query.search_effort = effort;
        query.validate_search_effort().unwrap();
        AutoProbePolicy::Fixed.apply(&mut query, &[1.0; 16], DistanceType::L2);
        assert_eq!(query.minimum_nprobes, 8);
        assert_eq!(query.maximum_nprobes, Some(8));
    }

    #[rstest]
    #[case::l2_top1(DistanceType::L2, 1, 0.2175, 5, 19)]
    #[case::l2_top10_lower_boundary(DistanceType::L2, 2, 0.265, 6, 24)]
    #[case::l2_top10_upper_boundary(DistanceType::L2, 10, 0.265, 6, 24)]
    #[case::l2_top100_lower_boundary(DistanceType::L2, 11, 0.33, 11, 38)]
    #[case::l2_top100(DistanceType::L2, 100, 0.33, 11, 38)]
    #[case::l2_top200_lower_boundary(DistanceType::L2, 101, 0.375, 15, 56)]
    #[case::l2_top200(DistanceType::L2, 200, 0.375, 15, 56)]
    #[case::l2_top500_lower_boundary(DistanceType::L2, 201, 0.3725, 22, 81)]
    #[case::l2_top500(DistanceType::L2, 500, 0.3725, 22, 81)]
    #[case::l2_top1000_lower_boundary(DistanceType::L2, 501, 0.4175, 29, 93)]
    #[case::l2_top1000(DistanceType::L2, 1000, 0.4175, 29, 93)]
    #[case::l2_top10000_lower_boundary(DistanceType::L2, 1001, 0.57, 98, 238)]
    #[case::l2_top10000(DistanceType::L2, 10_000, 0.57, 98, 238)]
    #[case::l2_top100000_lower_boundary(DistanceType::L2, 10_001, 0.75, 463, 704)]
    #[case::l2_top100000(DistanceType::L2, 100_000, 0.75, 463, 704)]
    #[case::cosine_top1(DistanceType::Cosine, 1, 0.235, 3, 50)]
    #[case::cosine_top10_lower_boundary(DistanceType::Cosine, 2, 0.2875, 8, 77)]
    #[case::cosine_top10_upper_boundary(DistanceType::Cosine, 10, 0.2875, 8, 77)]
    #[case::cosine_top100_lower_boundary(DistanceType::Cosine, 11, 0.38, 7, 106)]
    #[case::cosine_top100(DistanceType::Cosine, 100, 0.38, 7, 106)]
    #[case::cosine_top200_lower_boundary(DistanceType::Cosine, 101, 0.415, 18, 156)]
    #[case::cosine_top200(DistanceType::Cosine, 200, 0.415, 18, 156)]
    #[case::cosine_top500_lower_boundary(DistanceType::Cosine, 201, 0.45, 28, 192)]
    #[case::cosine_top500(DistanceType::Cosine, 500, 0.45, 28, 192)]
    #[case::cosine_top1000_lower_boundary(DistanceType::Cosine, 501, 0.47, 34, 248)]
    #[case::cosine_top1000(DistanceType::Cosine, 1000, 0.47, 34, 248)]
    #[case::cosine_top10000_lower_boundary(DistanceType::Cosine, 1001, 0.6475, 104, 447)]
    #[case::cosine_top10000(DistanceType::Cosine, 10_000, 0.6475, 104, 447)]
    #[case::cosine_top100000_lower_boundary(DistanceType::Cosine, 10_001, 0.58, 738, 958)]
    #[case::cosine_top100000(DistanceType::Cosine, 100_000, 0.58, 738, 958)]
    #[case::dot_top1(DistanceType::Dot, 1, 0.14, 56, 112)]
    #[case::dot_top10_lower_boundary(DistanceType::Dot, 2, 0.055, 144, 432)]
    #[case::dot_top10_upper_boundary(DistanceType::Dot, 10, 0.055, 144, 432)]
    #[case::dot_top100_lower_boundary(DistanceType::Dot, 11, 0.0625, 200, 768)]
    #[case::dot_top100(DistanceType::Dot, 100, 0.0625, 200, 768)]
    #[case::dot_top200_lower_boundary(DistanceType::Dot, 101, 0.0675, 211, 833)]
    #[case::dot_top200(DistanceType::Dot, 200, 0.0675, 211, 833)]
    #[case::dot_top500_lower_boundary(DistanceType::Dot, 201, 0.07, 244, 971)]
    #[case::dot_top500(DistanceType::Dot, 500, 0.07, 244, 971)]
    #[case::dot_top1000_lower_boundary(DistanceType::Dot, 501, 0.0725, 279, 1092)]
    #[case::dot_top1000(DistanceType::Dot, 1000, 0.0725, 279, 1092)]
    #[case::dot_top10000_lower_boundary(DistanceType::Dot, 1001, 0.0825, 517, 1824)]
    #[case::dot_top10000(DistanceType::Dot, 10_000, 0.0825, 517, 1824)]
    #[case::dot_top100000_lower_boundary(DistanceType::Dot, 10_001, 0.21, 1009, 2528)]
    #[case::dot_top100000(DistanceType::Dot, 100_000, 0.21, 1009, 2528)]
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
        #[values(DistanceType::Hamming)] metric: DistanceType,
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
    #[case::dot_positive_inner_product(DistanceType::Dot, &[-3.0, -1.0, 0.0], 0.5, 2)]
    #[case::dot_negative_inner_product(DistanceType::Dot, &[3.0, 4.0, 5.0], 0.5, 2)]
    #[case::dot_zero_inner_product(DistanceType::Dot, &[1.0, 1.0, 2.0], 80.0, 2)]
    #[case::dot_zero_distance(DistanceType::Dot, &[0.0, 0.5, 0.75], 0.5, 2)]
    #[case::dot_large_gap(DistanceType::Dot, &[-f32::MAX, 0.0, f32::MAX], 1.0, 2)]
    #[case::dot_nonfinite_tail(DistanceType::Dot, &[-3.0, -1.0, f32::INFINITY], 0.5, 2)]
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
    fn test_dot_gap_preserves_positive_query_scaling(
        #[values(0.125, 1.0, 8.0)] scale: f32,
        #[values(1, 10, 100, 101, 200, 500, 1000)] k: usize,
    ) {
        let distances = [4.0, 2.0, 1.0, -1.0].map(|inner_product| 1.0 - scale * inner_product);
        let mut query = query();
        query.k = k;
        AutoProbeConfig {
            min_initial_nprobes: 1,
            margin: 0.5,
            max_initial_nprobes: None,
        }
        .apply(&mut query, &distances, DistanceType::Dot);
        assert_eq!(query.minimum_nprobes, 2);
        assert_eq!(query.maximum_nprobes, None);
    }

    #[rstest]
    #[case::positive_inner_product(&[-3.0, -1.0, 0.0, 1.0], 1, 1)]
    #[case::positive_inner_product_top10(&[-3.0, -1.0, 0.0, 1.0], 10, 1)]
    #[case::negative_inner_product(&[3.0, 4.0, 5.0, 6.0], 10, 4)]
    #[case::zero_inner_product(&[1.0, 1.0, 2.0, 3.0], 100, 4)]
    fn test_legacy_dot_policy_preserves_signed_distances(
        #[case] distances: &[f32],
        #[case] k: usize,
        #[case] expected: usize,
    ) {
        let mut query = query();
        query.k = k;
        AutoProbePolicy::Legacy.apply(&mut query, distances, DistanceType::Dot);
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
        #[values(DistanceType::L2, DistanceType::Dot)] metric: DistanceType,
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
        .apply(&mut query, &[2.0, 3.0, 4.0, 5.0, 6.0], metric);
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
        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)] metric: DistanceType,
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
        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)] metric: DistanceType,
    ) {
        let distances = &[2.0, 3.0, 4.0, 5.0];
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
