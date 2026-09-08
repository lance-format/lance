// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Centroid-based selection of the initial IVF probe budget.
//!
//! Auto probing is an empirical heuristic, not a recall guarantee. The initial
//! budget does not limit subsequent probing when filters leave fewer than k rows.
//! `LANCE_AUTO_PROBE_MARGIN` overrides the nonnegative relative distance margin;
//! `LANCE_AUTO_MAX_INITIAL_NPROBES` overrides the positive initial budget cap.
//! Explicit fixed nprobes bypasses both the heuristic and these overrides.

use std::env;

use datafusion::error::{DataFusionError, Result as DataFusionResult};
use lance_index::vector::Query;
use lance_linalg::distance::DistanceType;

const MARGIN_ENV: &str = "LANCE_AUTO_PROBE_MARGIN";
const MAX_INITIAL_NPROBES_ENV: &str = "LANCE_AUTO_MAX_INITIAL_NPROBES";

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct AutoProbeConfig {
    pub(super) margin: f32,
    pub(super) max_initial_nprobes: Option<usize>,
}

impl AutoProbeConfig {
    pub(super) fn from_env(query: &Query, metric: DistanceType) -> DataFusionResult<Option<Self>> {
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return Ok(None);
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
        let maximum = read_override(MAX_INITIAL_NPROBES_ENV)?;
        Self::parse(query, metric, margin.as_deref(), maximum.as_deref())
    }

    fn parse(
        query: &Query,
        metric: DistanceType,
        margin: Option<&str>,
        maximum: Option<&str>,
    ) -> DataFusionResult<Option<Self>> {
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return Ok(None);
        }
        let bucket = match query.k {
            ..=1 => 0,
            2..=10 => 1,
            _ => 2,
        };
        // TODO: Finalize these profiles with independent metric-specific calibration.
        // Cosine starts from the measured LAION candidates; other metrics retain
        // the previous coarse k buckets without an implicit initial cap.
        let mut config = if metric == DistanceType::Cosine {
            Self {
                margin: [0.3, 1.0, 1.0][bucket],
                max_initial_nprobes: Some([20, 25, 47][bucket]),
            }
        } else {
            Self {
                margin: [0.0, 6.0, 80.0][bucket],
                ..Self::default()
            }
        };
        if let Some(value) = margin {
            config.margin = value
                .parse::<f32>()
                .ok()
                .filter(|margin| margin.is_finite() && *margin >= 0.0)
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "invalid {MARGIN_ENV} value {value:?}: expected a finite number >= 0"
                    ))
                })?;
        }
        if let Some(value) = maximum {
            let maximum = value
                .parse::<usize>()
                .ok()
                .filter(|maximum| *maximum > 0)
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "invalid {MAX_INITIAL_NPROBES_ENV} value {value:?}: expected a positive integer"
                    ))
                })?;
            config.max_initial_nprobes = Some(maximum);
        }
        Ok(Some(config))
    }

    /// Select an initial prefix of sorted centroid distances, honoring caller bounds.
    ///
    /// L2 and normalized-cosine routing use squared L2 distances. Dot routing uses
    /// `1 - dot(query, centroid)`, so its scale is the magnitude of the best dot
    /// product, not the shifted distance. At zero scale only best-distance ties
    /// qualify; the caller minimum still applies. f64 arithmetic avoids overflow
    /// when subtracting finite f32 distances or applying a large finite margin.
    pub(super) fn apply(self, query: &mut Query, distances: &[f32], metric: DistanceType) {
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return;
        }
        let selected = match distances.first().copied() {
            Some(nearest) if nearest.is_finite() => {
                let nearest = f64::from(nearest);
                let scale = match metric {
                    DistanceType::Dot => (1.0 - nearest).abs(),
                    _ => nearest,
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
        let selected = selected.min(self.max_initial_nprobes.unwrap_or(distances.len()));
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
    #[case::l2(DistanceType::L2, &[4.0, 6.0, 6.25], 0.5, 2)]
    #[case::cosine(DistanceType::Cosine, &[0.25, 0.5, 0.75], 1.0, 2)]
    #[case::negative_dot(DistanceType::Dot, &[-3.0, -1.0, 0.0], 0.5, 2)]
    #[case::zero_dot_distance(DistanceType::Dot, &[0.0, 0.5, 1.0], 0.5, 2)]
    #[case::negative_similarity(DistanceType::Dot, &[3.0, 4.0, 5.0], 0.5, 2)]
    #[case::zero_similarity(DistanceType::Dot, &[1.0, 1.0, 1.25], 80.0, 2)]
    #[case::zero_l2(DistanceType::L2, &[0.0, 0.0, 0.25], 80.0, 2)]
    #[case::zero_margin(DistanceType::L2, &[1.0, 1.0, 2.0], 0.0, 2)]
    #[case::empty(DistanceType::L2, &[], 1.0, 0)]
    #[case::single(DistanceType::Dot, &[-3.0], 1.0, 1)]
    #[case::nonfinite_nearest(DistanceType::L2, &[f32::INFINITY, f32::INFINITY], 1.0, 2)]
    #[case::nan_nearest(DistanceType::L2, &[f32::NAN], 1.0, 1)]
    #[case::nonfinite_tail(DistanceType::L2, &[1.0, 2.0, f32::INFINITY], 1.0, 2)]
    #[case::large_gap(DistanceType::Dot, &[-f32::MAX, f32::MAX], 2.0, 2)]
    fn test_auto_probe_gap(
        #[case] metric: DistanceType,
        #[case] distances: &[f32],
        #[case] margin: f32,
        #[case] expected: usize,
    ) {
        let mut query = query();
        AutoProbeConfig {
            margin,
            max_initial_nprobes: None,
        }
        .apply(&mut query, distances, metric);
        assert_eq!(query.minimum_nprobes, expected);
        assert_eq!(query.maximum_nprobes, None);
    }

    #[rstest]
    #[case::caller_minimum(4, None, Some(2), 4)]
    #[case::caller_maximum(1, Some(3), None, 3)]
    #[case::initial_cap(1, None, Some(2), 2)]
    #[case::fixed(3, Some(3), Some(1), 3)]
    #[case::minimum_exceeds_candidates(10, None, None, 5)]
    fn test_auto_probe_bounds(
        #[case] minimum: usize,
        #[case] maximum: Option<usize>,
        #[case] cap: Option<usize>,
        #[case] expected: usize,
    ) {
        let mut query = query();
        query.minimum_nprobes = minimum;
        query.maximum_nprobes = maximum;
        AutoProbeConfig {
            margin: 10.0,
            max_initial_nprobes: cap,
        }
        .apply(&mut query, &[1.0, 2.0, 3.0, 4.0, 5.0], DistanceType::L2);
        assert_eq!(query.minimum_nprobes, expected);
        assert_eq!(query.maximum_nprobes, maximum);
    }

    #[rstest]
    #[case::invalid_margin(Some("no"), None, MARGIN_ENV)]
    #[case::negative_margin(Some("-1"), None, MARGIN_ENV)]
    #[case::nan_margin(Some("NaN"), None, MARGIN_ENV)]
    #[case::infinite_margin(Some("inf"), None, MARGIN_ENV)]
    #[case::zero_cap(None, Some("0"), MAX_INITIAL_NPROBES_ENV)]
    #[case::negative_cap(None, Some("-1"), MAX_INITIAL_NPROBES_ENV)]
    #[case::invalid_cap(None, Some("no"), MAX_INITIAL_NPROBES_ENV)]
    fn test_auto_probe_invalid_overrides(
        #[case] margin: Option<&str>,
        #[case] maximum: Option<&str>,
        #[case] name: &str,
    ) {
        let error =
            AutoProbeConfig::parse(&query(), DistanceType::Dot, margin, maximum).unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(error.to_string().contains(name));
        let mut fixed = query();
        fixed.maximum_nprobes = Some(fixed.minimum_nprobes);
        assert_eq!(
            AutoProbeConfig::parse(&fixed, DistanceType::Dot, margin, maximum).unwrap(),
            None
        );
    }

    #[test]
    fn test_auto_probe_valid_overrides() {
        let config = AutoProbeConfig::parse(&query(), DistanceType::Cosine, Some("0"), Some("7"))
            .unwrap()
            .unwrap();
        assert_eq!(
            config,
            AutoProbeConfig {
                margin: 0.0,
                max_initial_nprobes: Some(7)
            }
        );
    }

    #[rstest]
    fn test_auto_probe_margin_monotonicity(
        #[values(DistanceType::L2, DistanceType::Dot)] metric: DistanceType,
    ) {
        let distances: &[f32] = if metric == DistanceType::Dot {
            &[-3.0, -2.0, 0.0, 2.0]
        } else {
            &[1.0, 2.0, 3.0, 4.0]
        };
        let mut previous = 0;
        for margin in [0.0, 0.25, 0.5, 1.0, 6.0, 80.0] {
            let mut query = query();
            AutoProbeConfig {
                margin,
                max_initial_nprobes: None,
            }
            .apply(&mut query, distances, metric);
            assert!(query.minimum_nprobes >= previous);
            previous = query.minimum_nprobes;
        }
    }

    #[rstest]
    #[case::positive(&[4.0, 2.0, 1.0])]
    #[case::negative(&[-2.0, -3.0, -4.0])]
    #[case::zero(&[0.0, 0.0, -1.0])]
    fn test_auto_probe_dot_query_scaling(#[case] similarities: &[f32]) {
        let mut expected = None;
        for scale in [0.5, 1.0, 4.0] {
            let distances = similarities
                .iter()
                .map(|similarity| 1.0 - scale * similarity)
                .collect::<Vec<_>>();
            let mut query = query();
            AutoProbeConfig {
                margin: 0.5,
                max_initial_nprobes: None,
            }
            .apply(&mut query, &distances, DistanceType::Dot);
            assert_eq!(
                *expected.get_or_insert(query.minimum_nprobes),
                query.minimum_nprobes
            );
        }
    }
}
