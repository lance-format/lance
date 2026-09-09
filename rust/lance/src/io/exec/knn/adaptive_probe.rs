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
//! Only ordinary Float32 FLAT vectors with complete norm statistics use the
//! learned profile and completion. Other capabilities, old indexes without
//! statistics, and explicitly bounded Auto queries retain the pre-experiment
//! f32 heuristic and ignore these overrides. Hamming is included in this fallback.

use std::env;

use arrow_array::{Array, UInt32Array, cast::AsArray};
use arrow_schema::DataType;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use lance_index::vector::{
    Query, VectorIndex, VectorNormRange, quantizer::QuantizationType, v3::subindex::SubIndexType,
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
    NormAware(AutoProbeConfig),
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
        if query.maximum_nprobes.is_some()
            || query.key.data_type() != &DataType::Float32
            || query.key.null_count() != 0
            || !matches!(vector_type, DataType::FixedSizeList(item, dimension)
                if item.data_type() == &DataType::Float32 && *dimension as usize == query.key.len())
            || !matches!(
                index.sub_index_type(),
                (SubIndexType::Flat, QuantizationType::Flat)
            )
            || index.metric_type() == DistanceType::Hamming
            || !query
                .key
                .as_primitive::<arrow_array::types::Float32Type>()
                .values()
                .iter()
                .all(|x| x.is_finite())
        {
            return Ok(Self::Legacy);
        }
        let mut has_vectors = false;
        for part in 0..index.total_partitions() {
            if index.partition_size(part) == 0 {
                continue;
            }
            has_vectors = true;
            let Some(range) = index.partition_norm_range(part) else {
                return Ok(Self::Legacy);
            };
            if !range.min.is_finite()
                || !range.max.is_finite()
                || range.min < 0.0
                || range.min > range.max
            {
                return Ok(Self::Legacy);
            }
        }
        if !has_vectors {
            return Ok(Self::Legacy);
        }
        Ok(read_config(query, index.metric_type())?.map_or(Self::Fixed, Self::NormAware))
    }

    pub(super) fn apply(self, query: &mut Query, distances: &[f32], metric: DistanceType) {
        match self {
            Self::Fixed => {}
            Self::Legacy => apply_legacy_probes(query, distances),
            Self::NormAware(config) => config.apply(query, distances, metric),
        }
    }

    pub(super) fn allows_completion(self) -> bool {
        matches!(self, Self::NormAware(_))
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

#[derive(Debug)]
struct NormCompletionCandidate {
    position: usize,
    lower_bound: f64,
    rows: usize,
}

/// A fixed eligible set, consumed in radial-bound order with fresh kth feedback.
#[derive(Debug)]
pub(super) struct NormCompletionPlan {
    candidates: Vec<NormCompletionCandidate>,
    cursor: usize,
    remaining_rows: usize,
}

impl NormCompletionPlan {
    /// Plan one complete batch at the current threshold. Never skip an expensive
    /// partition to consume a smaller later one, or split a partition's rows.
    pub(super) fn next_batch(&mut self, kth_distance: f32) -> Vec<usize> {
        let mut positions = Vec::with_capacity(8);
        while positions.len() < 8 && self.cursor < self.candidates.len() {
            let candidate = &self.candidates[self.cursor];
            if !kth_distance.is_finite()
                || candidate.lower_bound > f64::from(kth_distance)
                || candidate.rows > self.remaining_rows
            {
                self.cursor = self.candidates.len();
                break;
            }
            self.remaining_rows -= candidate.rows;
            positions.push(candidate.position);
            self.cursor += 1;
        }
        positions
    }
}

/// Supplement an Auto prefix when norm ranges identify an affordable remainder.
///
/// These descriptive f64 ranges do not certify native f32 pruning. The rule only
/// adds candidates to exact FLAT search; it never replaces the initial results.
/// Eligibility requires the whole qualifying remainder to fit the initial
/// prefix's nominal rows and to exclude at least one nonempty partition. Actual
/// extra work is limited to one tenth of those rows per segment, in batches of
/// at most eight partitions. Explicit probe limits bypass completion.
pub(super) fn radial_completion_plan(
    query: &Query,
    index: &dyn VectorIndex,
    partitions: &UInt32Array,
    kth_distance: f32,
) -> Option<NormCompletionPlan> {
    if query.maximum_nprobes.is_some()
        || !kth_distance.is_finite()
        || query.key.data_type() != &DataType::Float32
        || query.key.null_count() != 0
        || partitions
            .values()
            .iter()
            .all(|&part| index.partition_norm_range(part as usize).is_none())
    {
        return None;
    }
    if !matches!(
        index.sub_index_type(),
        (SubIndexType::Flat, QuantizationType::Flat)
    ) || partitions.len() != index.ivf_model().num_partitions()
    {
        return None;
    }
    let norm = query
        .key
        .as_primitive::<arrow_array::types::Float32Type>()
        .values()
        .iter()
        .map(|&value| f64::from(value).powi(2))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() {
        return None;
    }
    let initial = query.minimum_nprobes.min(partitions.len());
    let initial_rows = partitions.values()[..initial]
        .iter()
        .try_fold(0usize, |rows, &part| {
            rows.checked_add(index.partition_size(part as usize))
        })?;
    affordable_norm_completion(
        norm,
        index.metric_type(),
        kth_distance,
        initial_rows,
        (initial..partitions.len()).map(|position| {
            let part = partitions.value(position) as usize;
            (
                position,
                index.partition_size(part),
                index.partition_norm_range(part),
            )
        }),
    )
}

fn affordable_norm_completion(
    norm: f64,
    metric: DistanceType,
    kth_distance: f32,
    initial_rows: usize,
    candidates: impl Iterator<Item = (usize, usize, Option<VectorNormRange>)>,
) -> Option<NormCompletionPlan> {
    let mut extra_rows = 0usize;
    let mut selected = Vec::new();
    let mut nonempty = 0usize;
    for (position, rows, range) in candidates {
        if rows == 0 {
            continue;
        }
        nonempty += 1;
        let range = range?;
        let lower_bound = match metric {
            DistanceType::L2 | DistanceType::Cosine => {
                let gap = (range.min - norm).max(norm - range.max).max(0.0);
                let squared = gap * gap;
                if metric == DistanceType::Cosine {
                    squared * 0.5
                } else {
                    squared
                }
            }
            DistanceType::Dot => 1.0 - norm * range.max,
            DistanceType::Hamming => return None,
        };
        if lower_bound <= f64::from(kth_distance) {
            extra_rows = extra_rows.checked_add(rows)?;
            if extra_rows > initial_rows {
                // Preserve the original whole-remainder eligibility gate even
                // though the progressive execution budget is smaller.
                return None;
            }
            selected.push(NormCompletionCandidate {
                position,
                lower_bound,
                rows,
            });
        }
    }
    if selected.len() == nonempty {
        return None;
    }
    selected.sort_unstable_by(|a, b| {
        a.lower_bound
            .total_cmp(&b.lower_bound)
            .then(a.position.cmp(&b.position))
    });
    Some(NormCompletionPlan {
        candidates: selected,
        cursor: 0,
        remaining_rows: initial_rows / 10,
    })
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
        if metric == DistanceType::Hamming {
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
        if metric == DistanceType::Hamming {
            return Ok(Some(Self::default()));
        }
        let bucket = match query.k {
            ..=1 => 0,
            2..=10 => 1,
            _ => 2,
        };
        // Profiles depend only on metric and k; every index uses the same values.
        let config = if metric == DistanceType::L2 {
            Self {
                min_initial_nprobes: 6,
                margin: [0.39, 0.42, 0.45][bucket],
                max_initial_nprobes: Some([450, 432, 491][bucket]),
            }
        } else if metric == DistanceType::Cosine {
            Self {
                min_initial_nprobes: 1,
                margin: [0.35, 0.39, 0.41][bucket],
                max_initial_nprobes: Some([482, 553, 588][bucket]),
            }
        } else {
            Self {
                min_initial_nprobes: [1, 23, 1][bucket],
                margin: [0.13, 0.1375, 0.1425][bucket],
                max_initial_nprobes: Some([615, 683, 740][bucket]),
            }
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
    /// L2 and normalized-cosine routing use squared L2 distances. Dot routing uses
    /// `1 - dot(query, centroid)`, so its scale is the magnitude of the best dot
    /// product, not the shifted distance. At zero scale only best-distance ties
    /// qualify; the learned and caller minimums still apply. f64 arithmetic avoids
    /// overflow when subtracting finite f32 distances or applying a large finite margin.
    pub(super) fn apply(self, query: &mut Query, distances: &[f32], metric: DistanceType) {
        if query.maximum_nprobes == Some(query.minimum_nprobes) {
            return;
        }
        if metric == DistanceType::Hamming {
            apply_legacy_probes(query, distances);
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

    #[rstest]
    #[case::top1_positive_ties(1, vec![10.0, 10.0, 11.0], 1, None, 1)]
    #[case::top10_boundary(10, vec![1.0, 7.0, 8.0], 1, None, 2)]
    #[case::top100_boundary(100, vec![1.0, 81.0, 82.0], 1, None, 2)]
    #[case::zero_ties(1, vec![0.0, 0.0, 1.0], 1, None, 2)]
    #[case::negative_top1(1, vec![-10.0, -7.0, -6.0, -5.0], 1, None, 3)]
    #[case::negative_top10(10, vec![-2.0, -1.0, 0.0], 1, None, 1)]
    #[case::overflow(10, vec![f32::MAX, f32::INFINITY], 1, None, 2)]
    #[case::empty_preserves_min(1, vec![], 4, None, 4)]
    #[case::available_clipped_by_operator(1, vec![1.0, 2.0], 8, None, 8)]
    #[case::caller_max(100, vec![1.0, 2.0, 3.0], 1, Some(2), 2)]
    fn test_legacy_auto_matches_pre_experiment_arithmetic(
        #[case] k: usize,
        #[case] distances: Vec<f32>,
        #[case] minimum: usize,
        #[case] maximum: Option<usize>,
        #[case] expected: usize,
    ) {
        let mut query = query();
        query.k = k;
        query.minimum_nprobes = minimum;
        query.maximum_nprobes = maximum;
        AutoProbePolicy::Legacy.apply(&mut query, &distances, DistanceType::Dot);
        assert_eq!(query.minimum_nprobes, expected);
        assert_eq!(query.maximum_nprobes, maximum);
    }

    #[rstest]
    #[case::l2(DistanceType::L2, 4.0, 10.0)]
    #[case::cosine(DistanceType::Cosine, 2.0, 10.0)]
    #[case::dot(DistanceType::Dot, -2.0, 0.0)]
    fn test_norm_completion_keeps_boundary_candidates(
        #[case] metric: DistanceType,
        #[case] threshold: f32,
        #[case] pruned_norm: f64,
    ) {
        let range = Some(VectorNormRange { min: 3.0, max: 3.0 });
        let pruned = Some(VectorNormRange {
            min: pruned_norm,
            max: pruned_norm,
        });
        let mut plan = affordable_norm_completion(
            1.0,
            metric,
            threshold,
            70,
            [(5, 0, None), (6, 7, range), (7, 1, pruned)].into_iter(),
        )
        .unwrap();
        assert_eq!(plan.next_batch(threshold), vec![6]);
        assert!(plan.next_batch(threshold).is_empty());
    }

    #[test]
    fn test_norm_completion_requires_whole_remainder_and_pruning_information() {
        let range = Some(VectorNormRange { min: 1.0, max: 2.0 });
        for candidates in [
            vec![(1, 6, range), (2, 5, range)],
            vec![(1, 6, range), (2, 5, None)],
            vec![(1, 1, range), (2, 0, None)],
        ] {
            assert!(
                affordable_norm_completion(0.0, DistanceType::L2, 4.0, 10, candidates.into_iter())
                    .is_none()
            );
        }
        let mut plan = affordable_norm_completion(
            0.0,
            DistanceType::L2,
            0.5,
            10,
            [(1, 6, range), (2, 5, range)].into_iter(),
        )
        .unwrap();
        assert!(plan.next_batch(0.5).is_empty());
    }

    #[test]
    fn test_norm_completion_budget_stops_without_skipping_whole_partitions() {
        let range = |norm| {
            Some(VectorNormRange {
                min: norm,
                max: norm,
            })
        };
        let mut plan = affordable_norm_completion(
            0.0,
            DistanceType::L2,
            10.0,
            1000,
            [
                (0, 60, range(1.0)),
                (1, 50, range(2.0)),
                (2, 1, range(3.0)),
                (3, 1, range(100.0)),
            ]
            .into_iter(),
        )
        .unwrap();
        assert_eq!(plan.next_batch(10.0), vec![0]);
        assert!(plan.next_batch(10.0).is_empty());
        assert_eq!(plan.remaining_rows, 40);
        let mut rounded = affordable_norm_completion(
            0.0,
            DistanceType::L2,
            10.0,
            19,
            [(0, 2, range(1.0)), (1, 1, range(100.0))].into_iter(),
        )
        .unwrap();
        assert!(rounded.next_batch(10.0).is_empty()); // floor(19/10)=1
    }

    #[test]
    fn test_norm_completion_batch_limit_stable_ties_and_updated_kth() {
        let range = |norm| {
            Some(VectorNormRange {
                min: norm,
                max: norm,
            })
        };
        let mut candidates = (0..9)
            .rev()
            .map(|position| (position, 1, range(1.0)))
            .collect::<Vec<_>>();
        candidates.extend([(9, 1, range(2.0)), (10, 1, range(100.0))]);
        let mut plan =
            affordable_norm_completion(0.0, DistanceType::L2, 10.0, 1000, candidates.into_iter())
                .unwrap();
        assert_eq!(plan.next_batch(10.0), (0..8).collect::<Vec<_>>());
        assert_eq!(plan.next_batch(1.0), vec![8]);
        assert!(plan.next_batch(1.0).is_empty());
    }

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
    #[case::l2_top1(DistanceType::L2, 1, 0.39, 6, 450)]
    #[case::l2_top10_lower_boundary(DistanceType::L2, 2, 0.42, 6, 432)]
    #[case::l2_top10_upper_boundary(DistanceType::L2, 10, 0.42, 6, 432)]
    #[case::l2_top100_lower_boundary(DistanceType::L2, 11, 0.45, 6, 491)]
    #[case::l2_top100(DistanceType::L2, 100, 0.45, 6, 491)]
    #[case::cosine_top1(DistanceType::Cosine, 1, 0.35, 1, 482)]
    #[case::cosine_top10_lower_boundary(DistanceType::Cosine, 2, 0.39, 1, 553)]
    #[case::cosine_top10_upper_boundary(DistanceType::Cosine, 10, 0.39, 1, 553)]
    #[case::cosine_top100_lower_boundary(DistanceType::Cosine, 11, 0.41, 1, 588)]
    #[case::cosine_top100(DistanceType::Cosine, 100, 0.41, 1, 588)]
    #[case::dot_top1(DistanceType::Dot, 1, 0.13, 1, 615)]
    #[case::dot_top10_lower_boundary(DistanceType::Dot, 2, 0.1375, 23, 683)]
    #[case::dot_top10_upper_boundary(DistanceType::Dot, 10, 0.1375, 23, 683)]
    #[case::dot_top100_lower_boundary(DistanceType::Dot, 11, 0.1425, 1, 740)]
    #[case::dot_top100(DistanceType::Dot, 100, 0.1425, 1, 740)]
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

        let mut distances = vec![2.0; maximum + 1];
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
    fn test_hamming_preserves_probe_budget(
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
        // Real-valued Auto overrides must neither reject nor change Hamming.
        assert_eq!(
            AutoProbeConfig::parse(
                &query,
                DistanceType::Hamming,
                Some("invalid"),
                Some("0"),
                Some("0")
            )
            .unwrap(),
            Some(AutoProbeConfig::default()),
        );
        AutoProbeConfig {
            min_initial_nprobes: 4,
            margin: 80.0,
            max_initial_nprobes: Some(4),
        }
        .apply(&mut query, distances, DistanceType::Hamming);
        assert_eq!(query.minimum_nprobes, expected);
        assert_eq!(query.maximum_nprobes, maximum);
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
        #[case] margin: Option<&str>,
        #[case] minimum: Option<&str>,
        #[case] maximum: Option<&str>,
        #[case] name: &str,
    ) {
        let error = AutoProbeConfig::parse(&query(), DistanceType::Dot, margin, minimum, maximum)
            .unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(error.to_string().contains(name));
        let mut fixed = query();
        fixed.maximum_nprobes = Some(fixed.minimum_nprobes);
        assert_eq!(
            AutoProbeConfig::parse(&fixed, DistanceType::Dot, margin, minimum, maximum).unwrap(),
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
                min_initial_nprobes: 1,
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
