// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental adaptive IVF probe policies used by the KNN execution path.
//!
//! `LANCE_EXPERIMENTAL_PROBE_POLICY` selects `legacy` (default), `spann`,
//! `quake` (source-faithful control), or `quake-corrected` (retain finite negative
//! cosine distances in the raw global top-k; clamp only the geometric radius).
//! `LANCE_EXPERIMENTAL_PROBE_MIN_PARTITIONS` and `_MAX_PARTITIONS` are positive
//! scan-count bounds for experimental policies. They do not alter Quake's
//! fraction-selected candidate list or the probability denominator. The floor
//! must fit the available candidates and cap; the cap can end before k results.
//! Fixed nprobes and legacy ignore all experimental overrides.
//!
//! `LANCE_EXPERIMENTAL_QUAKE_ZERO_RADIUS_STOP=true` separately enables stopping
//! when the clamped kth radius is exactly zero, after k results and the floor.
//! No new positive-distance epsilon is introduced; the source APS profile keeps
//! its original numerical thresholds. Zero is an exact mathematical lower bound
//! for squared L2, but cosine rounding may yield smaller raw values elsewhere:
//! the cosine shortcut is empirical, with no IEEE raw-ranking or tie-ID guarantee.
//! `LANCE_EXPERIMENTAL_QUAKE_TRACE=true` emits per-partition calibration records
//! through the `lance::quake_calibration` tracing target outside CPU callbacks.
//! Enable an info-level subscriber for calibration; disable tracing in timings.

use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use arrow::datatypes::Float32Type;
use arrow_array::{Array, Float32Array, RecordBatch, UInt32Array, cast::AsArray};
use arrow_schema::DataType;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use lance_index::vector::{
    DIST_COL, PartitionSearchControl, Query, VectorIndex, quantizer::QuantizationType,
    v3::subindex::SubIndexType,
};
use lance_linalg::distance::DistanceType;

const EXPERIMENTAL_PROBE_POLICY_ENV: &str = "LANCE_EXPERIMENTAL_PROBE_POLICY";
const EXPERIMENTAL_SPANN_RATIO_ENV: &str = "LANCE_EXPERIMENTAL_SPANN_RATIO";
const EXPERIMENTAL_QUAKE_RECALL_TARGET_ENV: &str = "LANCE_EXPERIMENTAL_QUAKE_RECALL_TARGET";
const EXPERIMENTAL_QUAKE_INITIAL_FRACTION_ENV: &str = "LANCE_EXPERIMENTAL_QUAKE_INITIAL_FRACTION";
const EXPERIMENTAL_MIN_PARTITIONS_ENV: &str = "LANCE_EXPERIMENTAL_PROBE_MIN_PARTITIONS";
const EXPERIMENTAL_MAX_PARTITIONS_ENV: &str = "LANCE_EXPERIMENTAL_PROBE_MAX_PARTITIONS";
const EXPERIMENTAL_QUAKE_TRACE_ENV: &str = "LANCE_EXPERIMENTAL_QUAKE_TRACE";
const EXPERIMENTAL_ZERO_RADIUS_STOP_ENV: &str = "LANCE_EXPERIMENTAL_QUAKE_ZERO_RADIUS_STOP";

/// Limits actual scanned partitions, independently of Quake's candidate universe.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct ProbeLimits {
    minimum: Option<usize>,
    maximum: Option<usize>,
}

impl ProbeLimits {
    pub(super) fn apply(self, query: &mut Query, candidate_count: usize) -> DataFusionResult<()> {
        if self
            .minimum
            .is_some_and(|minimum| minimum > candidate_count)
        {
            return Err(DataFusionError::Execution(format!(
                "experimental scan minimum {:?} exceeds available candidate count {candidate_count}",
                self.minimum
            )));
        }
        if let Some(minimum) = self.minimum {
            query.minimum_nprobes = query.minimum_nprobes.max(minimum);
        }
        if let Some(maximum) = self.maximum {
            query.maximum_nprobes = Some(query.maximum_nprobes.unwrap_or(maximum).min(maximum));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct QuakeOptions {
    pub(super) limits: ProbeLimits,
    corrected_cosine: bool,
    zero_radius_stop: bool,
    trace: bool,
}

fn parse_probe_limits(
    minimum: Option<&str>,
    maximum: Option<&str>,
    query: &Query,
) -> DataFusionResult<ProbeLimits> {
    fn positive(name: &str, value: Option<&str>) -> DataFusionResult<Option<usize>> {
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
    let limits = ProbeLimits {
        minimum: positive(EXPERIMENTAL_MIN_PARTITIONS_ENV, minimum)?,
        maximum: positive(EXPERIMENTAL_MAX_PARTITIONS_ENV, maximum)?,
    };
    let minimum = limits
        .minimum
        .unwrap_or(query.minimum_nprobes)
        .max(query.minimum_nprobes);
    let maximum = limits
        .maximum
        .into_iter()
        .chain(query.maximum_nprobes)
        .min();
    if maximum.is_some_and(|maximum| minimum > maximum) {
        return Err(DataFusionError::Execution(format!(
            "experimental scan minimum {minimum} exceeds scan maximum {maximum:?}"
        )));
    }
    Ok(limits)
}

fn parse_bool(name: &str, value: Option<&str>) -> DataFusionResult<bool> {
    match value.unwrap_or("false") {
        "false" => Ok(false),
        "true" => Ok(true),
        value => Err(DataFusionError::Execution(format!(
            "invalid {name} value {value:?}: expected true or false"
        ))),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum ExperimentalProbePolicy {
    Legacy,
    Spann {
        ratio: f32,
        limits: ProbeLimits,
    },
    Quake {
        recall_target: f32,
        initial_fraction: f32,
        options: QuakeOptions,
    },
}

fn is_fixed_probe_count(query: &Query) -> bool {
    query.maximum_nprobes == Some(query.minimum_nprobes)
}

fn parse_experimental_probe_policy(
    policy: Option<&str>,
    spann_ratio: Option<&str>,
    quake_recall_target: Option<&str>,
    quake_initial_fraction: Option<&str>,
    k: usize,
) -> DataFusionResult<ExperimentalProbePolicy> {
    fn parse_f32(name: &str, value: Option<&str>, default: f32) -> DataFusionResult<f32> {
        let value = match value {
            Some(value) => value.parse::<f32>().map_err(|error| {
                DataFusionError::Execution(format!(
                    "invalid {name} value {value:?}: expected a finite number ({error})"
                ))
            })?,
            None => default,
        };
        if !value.is_finite() {
            return Err(DataFusionError::Execution(format!(
                "invalid {name} value {value}: expected a finite number"
            )));
        }
        Ok(value)
    }

    match policy.unwrap_or("legacy") {
        "legacy" => Ok(ExperimentalProbePolicy::Legacy),
        "spann" => {
            // The SPANN paper's calibrated (1 + epsilon) is 1.6 for top-1 and
            // 8 for top-10. Larger k uses the configurable top-k default rather
            // than the legacy heuristic's uncalibrated 81x jump.
            let default_ratio = if k == 1 { 1.6 } else { 8.0 };
            let ratio = parse_f32(EXPERIMENTAL_SPANN_RATIO_ENV, spann_ratio, default_ratio)?;
            if ratio < 1.0 {
                return Err(DataFusionError::Execution(format!(
                    "invalid {EXPERIMENTAL_SPANN_RATIO_ENV} value {ratio}: expected ratio >= 1"
                )));
            }
            Ok(ExperimentalProbePolicy::Spann {
                ratio,
                limits: ProbeLimits::default(),
            })
        }
        "quake" | "quake-corrected" => {
            let recall_target = parse_f32(
                EXPERIMENTAL_QUAKE_RECALL_TARGET_ENV,
                quake_recall_target,
                0.9,
            )?;
            if !(0.0..=1.0).contains(&recall_target) || recall_target == 0.0 {
                return Err(DataFusionError::Execution(format!(
                    "invalid {EXPERIMENTAL_QUAKE_RECALL_TARGET_ENV} value {recall_target}: expected 0 < target <= 1"
                )));
            }
            let initial_fraction = parse_f32(
                EXPERIMENTAL_QUAKE_INITIAL_FRACTION_ENV,
                quake_initial_fraction,
                0.1,
            )?;
            if !(0.0..=1.0).contains(&initial_fraction) || initial_fraction == 0.0 {
                return Err(DataFusionError::Execution(format!(
                    "invalid {EXPERIMENTAL_QUAKE_INITIAL_FRACTION_ENV} value {initial_fraction}: expected 0 < fraction <= 1"
                )));
            }
            Ok(ExperimentalProbePolicy::Quake {
                recall_target,
                initial_fraction,
                options: QuakeOptions {
                    corrected_cosine: policy == Some("quake-corrected"),
                    ..Default::default()
                },
            })
        }
        value => Err(DataFusionError::Execution(format!(
            "invalid {EXPERIMENTAL_PROBE_POLICY_ENV} value {value:?}: expected legacy, spann, quake, or quake-corrected"
        ))),
    }
}

pub(super) fn experimental_probe_policy(
    query: &Query,
) -> DataFusionResult<ExperimentalProbePolicy> {
    if is_fixed_probe_count(query) {
        return Ok(ExperimentalProbePolicy::Legacy);
    }
    let policy = env::var(EXPERIMENTAL_PROBE_POLICY_ENV).ok();
    let spann_ratio = env::var(EXPERIMENTAL_SPANN_RATIO_ENV).ok();
    let quake_recall_target = env::var(EXPERIMENTAL_QUAKE_RECALL_TARGET_ENV).ok();
    let quake_initial_fraction = env::var(EXPERIMENTAL_QUAKE_INITIAL_FRACTION_ENV).ok();
    let mut policy = parse_experimental_probe_policy(
        policy.as_deref(),
        spann_ratio.as_deref(),
        quake_recall_target.as_deref(),
        quake_initial_fraction.as_deref(),
        query.k,
    )?;
    if policy != ExperimentalProbePolicy::Legacy {
        let minimum = env::var(EXPERIMENTAL_MIN_PARTITIONS_ENV).ok();
        let maximum = env::var(EXPERIMENTAL_MAX_PARTITIONS_ENV).ok();
        let limits = parse_probe_limits(minimum.as_deref(), maximum.as_deref(), query)?;
        match &mut policy {
            ExperimentalProbePolicy::Spann { limits: target, .. } => *target = limits,
            ExperimentalProbePolicy::Quake { options, .. } => {
                options.limits = limits;
                let zero_radius_stop = env::var(EXPERIMENTAL_ZERO_RADIUS_STOP_ENV).ok();
                options.zero_radius_stop = parse_bool(
                    EXPERIMENTAL_ZERO_RADIUS_STOP_ENV,
                    zero_radius_stop.as_deref(),
                )?;
                let trace = env::var(EXPERIMENTAL_QUAKE_TRACE_ENV).ok();
                options.trace = parse_bool(EXPERIMENTAL_QUAKE_TRACE_ENV, trace.as_deref())?;
            }
            ExperimentalProbePolicy::Legacy => {}
        }
    }
    Ok(policy)
}

pub(super) fn quake_candidate_count(
    query: &Query,
    total_partitions: usize,
    initial_fraction: f32,
) -> usize {
    let maximum = query
        .maximum_nprobes
        .unwrap_or(total_partitions)
        .min(total_partitions);
    let initial = ((total_partitions as f32) * initial_fraction) as usize;
    initial
        .max(1)
        .max(query.minimum_nprobes.min(maximum))
        .min(maximum)
}

pub(super) fn spann_nprobes(dists: &[f32], ratio: f32) -> usize {
    let Some(&nearest) = dists.first() else {
        return 0;
    };
    let threshold = nearest * ratio;
    dists.partition_point(|distance| *distance <= threshold)
}

const QUAKE_BETA_TABLE_POINTS: usize = 1001;
const QUAKE_RECOMPUTE_THRESHOLD: f32 = 0.001;
const QUAKE_EPSILON: f32 = 1.0e-9;

static QUAKE_BETA_TABLE: LazyLock<Mutex<Option<(usize, Arc<[f64]>)>>> =
    LazyLock::new(|| Mutex::new(None));

fn regularized_incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(b, a, 1.0 - x);
    }

    let log_beta = libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b);
    let front = (x.ln() * a + (1.0 - x).ln() * b - log_beta).exp() / a;
    let mut fraction = 1.0;
    let mut c = 1.0;
    let mut d = 0.0;
    for i in 0..=200 {
        let m = i / 2;
        let numerator = if i == 0 {
            1.0
        } else if i % 2 == 0 {
            let m = m as f64;
            (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        } else {
            let m = m as f64;
            -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        };
        d = 1.0 + numerator * d;
        if d.abs() < 1.0e-30 {
            d = 1.0e-30;
        }
        d = 1.0 / d;
        c = 1.0 + numerator / c;
        if c.abs() < 1.0e-30 {
            c = 1.0e-30;
        }
        let change = c * d;
        fraction *= change;
        if (1.0 - change).abs() < 1.0e-8 {
            return front * (fraction - 1.0);
        }
    }
    f64::NAN
}

fn quake_beta_table(dimension: usize) -> DataFusionResult<Arc<[f64]>> {
    let mut cached = QUAKE_BETA_TABLE.lock().unwrap();
    if let Some((cached_dimension, table)) = cached.as_ref() {
        if *cached_dimension != dimension {
            return Err(DataFusionError::Execution(format!(
                "Quake probing initialized its process-wide beta table for dimension {cached_dimension}, cannot use dimension {dimension}"
            )));
        }
        return Ok(table.clone());
    }
    let a = (dimension as f64 + 1.0) / 2.0;
    let b = 0.5;
    let table: Arc<[f64]> = (0..QUAKE_BETA_TABLE_POINTS)
        .map(|idx| {
            if idx == 0 {
                0.0
            } else if idx + 1 == QUAKE_BETA_TABLE_POINTS {
                1.0
            } else {
                regularized_incomplete_beta(a, b, idx as f64 / (QUAKE_BETA_TABLE_POINTS - 1) as f64)
            }
        })
        .collect::<Vec<_>>()
        .into();
    *cached = Some((dimension, table.clone()));
    Ok(table)
}

fn quake_beta_lookup(table: &[f64], x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let scaled = x * (QUAKE_BETA_TABLE_POINTS - 1) as f64;
    let lower = (scaled as usize).min(QUAKE_BETA_TABLE_POINTS - 2);
    let fraction = scaled - lower as f64;
    table[lower] + fraction * (table[lower + 1] - table[lower])
}

fn quake_cap_probability(radius: f32, boundary_distance: f32, beta_table: &[f64]) -> f32 {
    let boundary_distance = f64::from(boundary_distance.max(0.0));
    let radius = f64::from(radius);
    if boundary_distance >= radius {
        return 0.0;
    }
    let ratio = boundary_distance / radius;
    let x = (1.0 - ratio * ratio).max(0.0).sqrt();
    (0.5 * quake_beta_lookup(beta_table, x)).clamp(0.0, 0.5) as f32
}

fn squared_l2_distance(distance: f32, metric_type: DistanceType) -> f32 {
    match metric_type {
        DistanceType::L2 => distance,
        // Lance cosine distance is 1 - cos(theta); normalized L2 squared is
        // 2 * (1 - cos(theta)).
        DistanceType::Cosine => 2.0 * distance,
        _ => f32::NAN,
    }
}

fn quake_recall_profile(
    boundary_distances: &[f32],
    radius: f32,
    partition_sizes: &[usize],
    beta_table: &[f64],
) -> Vec<f32> {
    let count = boundary_distances.len();
    if count <= 1 {
        return vec![1.0; count];
    }
    if radius <= QUAKE_EPSILON {
        let mut probabilities = vec![0.0; count];
        probabilities[0] = 1.0;
        return probabilities;
    }

    let mut normalized_caps = Vec::with_capacity(count);
    normalized_caps.push(0.0);
    normalized_caps.extend(
        boundary_distances[1..]
            .iter()
            .map(|&distance| quake_cap_probability(radius, distance, beta_table)),
    );
    let cap_sum: f32 = normalized_caps[1..].iter().sum();
    if cap_sum > QUAKE_EPSILON {
        for cap in &mut normalized_caps[1..] {
            *cap /= cap_sum;
        }
    } else {
        normalized_caps[1..].fill(0.0);
    }

    let root_probability = normalized_caps[1..]
        .iter()
        .fold(1.0, |probability, cap| probability * (1.0 - cap))
        .clamp(0.0, 1.0);
    let neighbor_mass = (1.0 - root_probability).clamp(0.0, 1.0);
    let neighbor_sum: f32 = normalized_caps[1..].iter().sum();
    let mut probabilities = vec![0.0; count];
    probabilities[0] = root_probability;
    if neighbor_mass > QUAKE_EPSILON && neighbor_sum > QUAKE_EPSILON {
        let scale = neighbor_mass / neighbor_sum;
        for (probability, cap) in probabilities[1..].iter_mut().zip(&normalized_caps[1..]) {
            *probability = (*cap * scale).max(0.0);
        }
        let current_sum: f32 = probabilities[1..].iter().sum();
        if current_sum > QUAKE_EPSILON {
            let final_scale = neighbor_mass / current_sum;
            if final_scale.is_finite() {
                for probability in &mut probabilities[1..] {
                    *probability = (*probability * final_scale).max(0.0);
                }
            }
        }
    }

    // Match Quake's current density weighting exactly: zero-size partitions
    // retain their geometric probability instead of being multiplied by zero.
    for (probability, &partition_size) in probabilities.iter_mut().zip(partition_sizes) {
        if partition_size > 0 {
            *probability *= partition_size as f32;
        }
    }
    let sum: f32 = probabilities.iter().sum();
    if sum > QUAKE_EPSILON {
        for probability in &mut probabilities {
            *probability /= sum;
        }
    } else {
        probabilities.fill(0.0);
    }
    probabilities
}

fn quake_boundary_distances(
    query: &Float32Array,
    centroids: &arrow_array::FixedSizeListArray,
    partitions: &UInt32Array,
) -> DataFusionResult<Vec<f32>> {
    // Faithful concise port of Quake geometry.h at 0651c88c5c71343191ef06ee4ed292cb4a01e1aa:
    // https://github.com/marius-team/quake/blob/0651c88c5c71343191ef06ee4ed292cb4a01e1aa/src/cpp/include/geometry.h
    let dimension = query.len();
    if partitions.is_empty() {
        return Err(DataFusionError::Execution(
            "Quake probing requires at least one IVF partition".to_string(),
        ));
    }
    let first_partition = partitions.value(0) as usize;
    let first = centroids.value(first_partition);
    if first.data_type() != &DataType::Float32 || first.len() != dimension {
        return Err(DataFusionError::Execution(format!(
            "Quake probing requires Float32 centroids with query dimension {dimension}, got {:?} with length {}",
            first.data_type(),
            first.len()
        )));
    }
    let first = first.as_primitive::<Float32Type>();
    let first_norm_squared: f32 = first.values().iter().map(|value| value * value).sum();
    let mut distances = Vec::with_capacity(partitions.len());
    distances.push(0.0);
    for partition in partitions.values().iter().skip(1) {
        let centroid = centroids.value(*partition as usize);
        if centroid.data_type() != &DataType::Float32 || centroid.len() != dimension {
            return Err(DataFusionError::Execution(format!(
                "Quake probing requires Float32 centroids with query dimension {dimension}, got {:?} with length {} for partition {partition}",
                centroid.data_type(),
                centroid.len()
            )));
        }
        let centroid = centroid.as_primitive::<Float32Type>();
        let mut vector_norm_squared = 0.0;
        let mut centroid_norm_squared = 0.0;
        let mut query_dot_vector = 0.0;
        for ((&current, &origin), &query_value) in centroid
            .values()
            .iter()
            .zip(first.values())
            .zip(query.values())
        {
            let vector = current - origin;
            vector_norm_squared += vector * vector;
            centroid_norm_squared += current * current;
            query_dot_vector += query_value * vector;
        }
        let boundary = (query_dot_vector - 0.5 * (centroid_norm_squared - first_norm_squared))
            .abs()
            / (vector_norm_squared.sqrt() + 1.0e-12);
        distances.push(boundary);
    }
    Ok(distances)
}

#[derive(Debug)]
struct QuakeTrace {
    partition_id: u32,
    partitions_searched: usize,
    partition_vectors: usize,
    scanned_vectors: usize,
    raw_global_kth: Option<f32>,
    policy_raw_kth: Option<f32>,
    geometry_radius: Option<f32>,
    reason: &'static str,
}

struct QuakeSearchState {
    best_raw_distances: Vec<f32>,
    partitions_searched: usize,
    last_radius: Option<f32>,
    probabilities: Vec<f32>,
    traces: Vec<QuakeTrace>,
    trace_raw_distances: Vec<f32>,
}

pub(super) struct QuakePartitionSearchControl {
    k: usize,
    minimum_nprobes: usize,
    maximum_nprobes: usize,
    partition_ids: Vec<u32>,
    recall_target: f32,
    metric_type: DistanceType,
    options: QuakeOptions,
    boundary_distances: Vec<f32>,
    partition_sizes: Vec<usize>,
    beta_table: Arc<[f64]>,
    state: Mutex<QuakeSearchState>,
    should_stop: AtomicBool,
    callback_failed: AtomicBool,
}

impl QuakePartitionSearchControl {
    pub(super) fn try_new(
        query: &Query,
        index: &dyn VectorIndex,
        partitions: &UInt32Array,
        minimum_nprobes: usize,
        recall_target: f32,
        options: QuakeOptions,
    ) -> DataFusionResult<Self> {
        let centroids = index.ivf_model().centroids_array().ok_or_else(|| {
            DataFusionError::Execution("Quake probing requires IVF centroids".to_string())
        })?;
        if query.key.data_type() != &DataType::Float32 {
            return Err(DataFusionError::Execution(format!(
                "Quake probing requires a Float32 query, got {:?}",
                query.key.data_type()
            )));
        }
        let query_values = query.key.as_primitive::<Float32Type>();
        if query_values.null_count() > 0 {
            return Err(DataFusionError::Execution(
                "Quake probing does not support null query values".to_string(),
            ));
        }
        let dimension = query_values.len();
        if dimension == 0 || centroids.value_length() as usize != dimension {
            return Err(DataFusionError::Execution(format!(
                "Quake probing query dimension {dimension} does not match centroid dimension {}",
                centroids.value_length()
            )));
        }
        let boundary_distances = quake_boundary_distances(query_values, centroids, partitions)?;
        let partition_sizes = partitions
            .values()
            .iter()
            .map(|partition| index.partition_size(*partition as usize))
            .collect();
        Ok(Self {
            k: query.k,
            minimum_nprobes,
            maximum_nprobes: query
                .maximum_nprobes
                .unwrap_or(partitions.len())
                .min(partitions.len()),
            partition_ids: if options.trace {
                partitions.values().to_vec()
            } else {
                Vec::new()
            },
            recall_target,
            metric_type: index.metric_type(),
            options,
            boundary_distances,
            partition_sizes,
            beta_table: quake_beta_table(dimension)?,
            state: Mutex::new(QuakeSearchState {
                best_raw_distances: Vec::with_capacity(query.k),
                partitions_searched: 0,
                last_radius: None,
                probabilities: Vec::new(),
                traces: Vec::new(),
                trace_raw_distances: Vec::new(),
            }),
            should_stop: AtomicBool::new(false),
            callback_failed: AtomicBool::new(false),
        })
    }

    /// Emit only outside spawn_cpu: logging may perform I/O or take locks.
    pub(super) fn emit_trace(&self) {
        if !self.options.trace {
            return;
        }
        let Ok(mut state) = self.state.try_lock() else {
            return;
        };
        let traces = std::mem::take(&mut state.traces);
        drop(state);
        for trace in traces {
            tracing::info!(
                target: "lance::quake_calibration",
                partition_id = trace.partition_id,
                partitions_searched = trace.partitions_searched,
                partition_vectors = trace.partition_vectors,
                scanned_vectors = trace.scanned_vectors,
                raw_global_kth = ?trace.raw_global_kth,
                policy_raw_kth = ?trace.policy_raw_kth,
                geometry_radius = ?trace.geometry_radius,
                stop_reason = trace.reason,
                "experimental Quake calibration"
            );
        }
    }

    pub(super) fn callback_failed(&self) -> bool {
        self.callback_failed.load(Ordering::Relaxed)
    }
}

impl PartitionSearchControl for QuakePartitionSearchControl {
    fn should_stop(&self) -> bool {
        self.should_stop.load(Ordering::Relaxed)
    }

    fn record_batch(&self, batch: &RecordBatch) {
        let Some(distances) = batch.column_by_name(DIST_COL) else {
            return;
        };
        let distances = distances.as_primitive::<Float32Type>();
        // The experimental policy requires query_parallelism=1, so callbacks are
        // sequential. Use try_lock because this callback runs inside spawn_cpu and
        // must never park a compute worker waiting for synchronization.
        let Ok(mut state) = self.state.try_lock() else {
            self.callback_failed.store(true, Ordering::Relaxed);
            self.should_stop.store(true, Ordering::Relaxed);
            return;
        };
        state.partitions_searched += 1;
        state.best_raw_distances.extend(
            distances
                .iter()
                .flatten()
                .filter(|distance| distance.is_finite())
                .filter(|distance| {
                    let squared = squared_l2_distance(*distance, self.metric_type);
                    (self.options.corrected_cosine && self.metric_type == DistanceType::Cosine)
                        || (squared.is_finite() && squared >= 0.0)
                }),
        );
        if state.best_raw_distances.len() >= self.k {
            state.best_raw_distances.sort_unstable_by(f32::total_cmp);
            state.best_raw_distances.truncate(self.k);
        }
        let reason = (|| {
            if state.partitions_searched < self.minimum_nprobes {
                return "minimum_floor";
            }
            if state.best_raw_distances.len() < self.k {
                return "insufficient_top_k";
            }
            // Keep the original distances in the top-k: a rounded negative cosine
            // distance still ranks ahead of zero. Clamp only the geometric radius.
            let squared_radius =
                squared_l2_distance(state.best_raw_distances[self.k - 1], self.metric_type);
            let radius = squared_radius.max(0.0).sqrt();
            if !radius.is_finite() {
                return "nonfinite_radius";
            }
            if radius == 0.0 {
                // Exact squared L2 has lower bound zero, so k zero-distance results
                // are distance-optimal (ties may choose different row IDs). Cosine
                // kernels can round below zero; this optional shortcut is empirical
                // for cosine and does not guarantee the globally smallest raw scores.
                // Deliberately no epsilon: a positive radius still uses Quake APS.
                if self.options.zero_radius_stop {
                    self.should_stop.store(true, Ordering::Relaxed);
                    return "zero_radius";
                }
                return "zero_radius_continue";
            }
            let should_recompute = state.last_radius.is_none_or(|previous| {
                (radius - previous).abs() / radius > QUAKE_RECOMPUTE_THRESHOLD
            });
            if should_recompute {
                state.probabilities = quake_recall_profile(
                    &self.boundary_distances,
                    radius,
                    &self.partition_sizes,
                    &self.beta_table,
                );
                state.last_radius = Some(radius);
            }
            let recall_estimate: f32 = state
                .probabilities
                .iter()
                .take(state.partitions_searched)
                .sum();
            if recall_estimate >= self.recall_target {
                self.should_stop.store(true, Ordering::Relaxed);
                return "recall_target";
            }
            "continue"
        })();
        if self.options.trace {
            // Calibration also records the unfiltered finite raw kth, so the
            // source-faithful policy's negative-cosine omission is observable.
            state.trace_raw_distances.extend(
                distances
                    .iter()
                    .flatten()
                    .filter(|distance| distance.is_finite()),
            );
            state.trace_raw_distances.sort_unstable_by(f32::total_cmp);
            state.trace_raw_distances.truncate(self.k);
            let searched = state.partitions_searched;
            let raw_global_kth = (state.trace_raw_distances.len() >= self.k)
                .then(|| state.trace_raw_distances[self.k - 1]);
            let policy_raw_kth = (state.best_raw_distances.len() >= self.k)
                .then(|| state.best_raw_distances[self.k - 1]);
            let reason = if !self.should_stop() && searched >= self.maximum_nprobes {
                if searched < self.boundary_distances.len() {
                    "scan_cap"
                } else {
                    "candidate_exhausted"
                }
            } else {
                reason
            };
            state.traces.push(QuakeTrace {
                partition_id: self.partition_ids[searched - 1],
                partitions_searched: searched,
                partition_vectors: self.partition_sizes[searched - 1],
                scanned_vectors: self.partition_sizes.iter().take(searched).sum(),
                raw_global_kth,
                policy_raw_kth,
                geometry_radius: policy_raw_kth.map(|distance| {
                    squared_l2_distance(distance, self.metric_type)
                        .max(0.0)
                        .sqrt()
                }),
                reason,
            });
        }
    }
}

pub(super) fn validate_experimental_probe_plan(
    policy: ExperimentalProbePolicy,
    query: &Query,
    num_indices: usize,
    has_prefilter: bool,
    has_overlay_block: bool,
    has_external_mask: bool,
) -> DataFusionResult<()> {
    if policy == ExperimentalProbePolicy::Legacy {
        return Ok(());
    }
    if num_indices != 1 {
        return Err(DataFusionError::Execution(format!(
            "experimental IVF probing requires exactly one index segment, got {num_indices}"
        )));
    }
    if has_prefilter || has_overlay_block || has_external_mask {
        return Err(DataFusionError::Execution(
            "experimental IVF probing does not support filters, overlays, or external masks"
                .to_string(),
        ));
    }
    if query.lower_bound.is_some() || query.upper_bound.is_some() {
        return Err(DataFusionError::Execution(
            "experimental IVF probing does not support distance bounds".to_string(),
        ));
    }
    if query.query_parallelism != 1 {
        return Err(DataFusionError::Execution(format!(
            "experimental IVF probing requires query_parallelism=1, got {}",
            query.query_parallelism
        )));
    }
    Ok(())
}

pub(super) fn validate_experimental_probe_index(
    policy: ExperimentalProbePolicy,
    query: &Query,
    index: &dyn VectorIndex,
) -> DataFusionResult<()> {
    if policy == ExperimentalProbePolicy::Legacy {
        return Ok(());
    }
    if !matches!(
        index.sub_index_type(),
        (SubIndexType::Flat, QuantizationType::Flat)
    ) {
        return Err(DataFusionError::Execution(format!(
            "experimental IVF probing requires IVF_FLAT, got {}_{}",
            index.sub_index_type().0,
            index.sub_index_type().1
        )));
    }
    if !matches!(index.metric_type(), DistanceType::L2 | DistanceType::Cosine) {
        return Err(DataFusionError::Execution(format!(
            "experimental IVF probing requires L2 or cosine distance, got {}",
            index.metric_type()
        )));
    }
    if query.key.data_type() != &DataType::Float32 {
        return Err(DataFusionError::Execution(format!(
            "experimental IVF probing requires a Float32 query, got {:?}",
            query.key.data_type()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{ArrayRef, FixedSizeListArray};
    use lance_arrow::FixedSizeListArrayExt;
    use lance_index::vector::DEFAULT_QUERY_PARALLELISM;

    const GOLDEN_DIMENSION: usize = 768;

    fn base_query() -> Query {
        Query {
            column: "vec".to_string(),
            key: Arc::new(Float32Array::from(vec![0.0; GOLDEN_DIMENSION])) as ArrayRef,
            k: 10,
            lower_bound: None,
            upper_bound: None,
            minimum_nprobes: 1,
            maximum_nprobes: None,
            ef: None,
            refine_factor: None,
            metric_type: Some(DistanceType::L2),
            use_index: true,
            query_parallelism: DEFAULT_QUERY_PARALLELISM,
            dist_q_c: 0.0,
            approx_mode: Default::default(),
        }
    }

    fn test_control(metric_type: DistanceType) -> QuakePartitionSearchControl {
        QuakePartitionSearchControl {
            k: 3,
            minimum_nprobes: 1,
            maximum_nprobes: 2,
            partition_ids: vec![0, 1],
            recall_target: 0.9,
            metric_type,
            options: QuakeOptions::default(),
            boundary_distances: vec![0.0, 0.0],
            partition_sizes: vec![10, 10],
            beta_table: quake_beta_table(GOLDEN_DIMENSION).unwrap(),
            state: Mutex::new(QuakeSearchState {
                best_raw_distances: Vec::with_capacity(3),
                partitions_searched: 0,
                last_radius: None,
                probabilities: Vec::new(),
                traces: Vec::new(),
                trace_raw_distances: Vec::new(),
            }),
            should_stop: AtomicBool::new(false),
            callback_failed: AtomicBool::new(false),
        }
    }

    fn distance_batch(distances: Vec<f32>) -> RecordBatch {
        RecordBatch::try_from_iter(vec![(
            DIST_COL,
            Arc::new(Float32Array::from(distances)) as ArrayRef,
        )])
        .unwrap()
    }

    #[test]
    fn test_parse_policy_and_probe_counts() {
        assert_eq!(
            parse_experimental_probe_policy(None, None, None, None, 1).unwrap(),
            ExperimentalProbePolicy::Legacy
        );
        assert_eq!(
            parse_experimental_probe_policy(Some("spann"), None, None, None, 1).unwrap(),
            ExperimentalProbePolicy::Spann {
                ratio: 1.6,
                limits: ProbeLimits::default()
            }
        );
        assert_eq!(
            parse_experimental_probe_policy(Some("spann"), None, None, None, 100).unwrap(),
            ExperimentalProbePolicy::Spann {
                ratio: 8.0,
                limits: ProbeLimits::default()
            }
        );
        assert_eq!(
            parse_experimental_probe_policy(Some("quake"), None, Some("0.95"), Some("1"), 10)
                .unwrap(),
            ExperimentalProbePolicy::Quake {
                recall_target: 0.95,
                initial_fraction: 1.0,
                options: QuakeOptions::default(),
            }
        );
        assert!(
            parse_experimental_probe_policy(Some("quake"), None, Some("0"), None, 10)
                .unwrap_err()
                .to_string()
                .contains(EXPERIMENTAL_QUAKE_RECALL_TARGET_ENV)
        );

        let mut query = base_query();
        query.minimum_nprobes = 4;
        query.maximum_nprobes = Some(4);
        assert!(is_fixed_probe_count(&query));
        query.maximum_nprobes = Some(80);
        assert_eq!(quake_candidate_count(&query, 100, 0.1), 10);
        assert_eq!(quake_candidate_count(&query, 100, 1.0), 80);
        query.minimum_nprobes = 20;
        assert_eq!(quake_candidate_count(&query, 100, 0.1), 20);

        assert_eq!(spann_nprobes(&[], 1.6), 0);
        assert_eq!(spann_nprobes(&[4.0, 6.4, 6.41, 20.0], 1.6), 2);
        assert_eq!(spann_nprobes(&[0.0, 0.0, 0.1], 8.0), 2);
    }

    #[test]
    fn test_quake_dimension_768_geometry_goldens() {
        let table = quake_beta_table(GOLDEN_DIMENSION).unwrap();
        assert_eq!(quake_beta_lookup(&table, 0.0), 0.0);
        assert_eq!(quake_beta_lookup(&table, 1.0), 1.0);
        let cap = quake_cap_probability(0.5, 0.01, &table);
        assert!((cap - 0.438_050_03).abs() < 1.0e-6, "cap={cap}");
        assert_eq!(quake_cap_probability(0.1, 0.1, &table), 0.0);

        let mut centroid_values = vec![0.0; GOLDEN_DIMENSION * 3];
        centroid_values[GOLDEN_DIMENSION] = 2.0;
        let centroids = FixedSizeListArray::try_new_from_values(
            Float32Array::from(centroid_values),
            GOLDEN_DIMENSION as i32,
        )
        .unwrap();
        let query = Float32Array::from(vec![0.0; GOLDEN_DIMENSION]);
        let partitions = UInt32Array::from(vec![0, 1, 2]);
        let boundaries = quake_boundary_distances(&query, &centroids, &partitions).unwrap();
        assert_eq!(boundaries, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_quake_dimension_768_recall_profile_goldens() {
        let table = quake_beta_table(GOLDEN_DIMENSION).unwrap();
        let probabilities = quake_recall_profile(
            &[0.0, 0.01, 0.03, 0.1, 0.3],
            0.5,
            &[5, 50, 500, 5000, 50_000],
            &table,
        );
        let expected = [
            0.006_892_448_3,
            0.258_885_86,
            0.731_983_3,
            0.002_238_391_2,
            1.034_540_6e-36,
        ];
        for (actual, expected) in probabilities.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2.0e-7,
                "{actual} != {expected}"
            );
        }
        assert_eq!(quake_recall_profile(&[0.0], 0.1, &[], &table), vec![1.0]);
        assert_eq!(
            quake_recall_profile(&[0.0, 0.1], 0.0, &[5, 50], &table),
            vec![1.0, 0.0]
        );
        assert_eq!(squared_l2_distance(0.5, DistanceType::Cosine), 1.0);
        assert_eq!(squared_l2_distance(0.5, DistanceType::L2), 0.5);
    }

    #[test]
    fn test_quake_control_uses_global_kth_distance_and_waits_for_k() {
        let control = test_control(DistanceType::L2);
        control.record_batch(&distance_batch(vec![0.09, 0.01]));
        assert!(!control.should_stop());
        assert_eq!(control.state.lock().unwrap().last_radius, None);

        // IVF_FLAT returns heap order, so this makes the exactly-k input
        // deliberately unsorted. Its kth distance is 0.09, not the final 0.04.
        control.record_batch(&distance_batch(vec![0.04]));
        {
            let state = control.state.lock().unwrap();
            assert!((state.last_radius.unwrap() - 0.3).abs() < 1.0e-6);
            assert_eq!(state.partitions_searched, 2);
        }
        assert!(control.should_stop());
    }

    #[test]
    fn test_quake_control_recomputes_when_global_kth_shrinks() {
        let control = test_control(DistanceType::L2);
        control.record_batch(&distance_batch(vec![0.09, 0.01, 0.04]));
        assert!(!control.should_stop());
        assert!((control.state.lock().unwrap().last_radius.unwrap() - 0.3).abs() < 1.0e-6);

        control.record_batch(&distance_batch(vec![0.0025]));
        let state = control.state.lock().unwrap();
        assert!((state.last_radius.unwrap() - 0.2).abs() < 1.0e-6);
        drop(state);
        assert!(control.should_stop());
    }

    #[test]
    fn test_quake_control_converts_cosine_distance_to_normalized_l2_radius() {
        let control = test_control(DistanceType::Cosine);
        control.record_batch(&distance_batch(vec![0.045, 0.005, 0.02]));
        let radius = control.state.lock().unwrap().last_radius.unwrap();
        assert!((radius - 0.3).abs() < 1.0e-6, "radius={radius}");
    }
    #[rstest::rstest]
    #[case::l2_zero(DistanceType::L2, false, vec![0.0, 0.0, 0.0], true)]
    #[case::cosine_zero(DistanceType::Cosine, true, vec![0.0, 0.0, 0.0], true)]
    #[case::cosine_negative(DistanceType::Cosine, true, vec![-1e-7, -1e-8, -1e-9], true)]
    #[case::cosine_mixed(DistanceType::Cosine, true, vec![-1e-7, 0.0, 0.0], true)]
    #[case::cosine_positive(DistanceType::Cosine, true, vec![-1e-7, 0.0, 1e-10], false)]
    #[case::l2_positive(DistanceType::L2, false, vec![0.0, 0.0, 1e-10], false)]
    #[case::l2_negative_rejected(DistanceType::L2, true, vec![-1e-7, 0.0, 0.0], false)]
    fn test_exact_zero_stop_and_raw_ranking(
        #[case] metric: DistanceType,
        #[case] corrected_cosine: bool,
        #[case] distances: Vec<f32>,
        #[case] expected_stop: bool,
    ) {
        let mut control = test_control(metric);
        control.options.corrected_cosine = corrected_cosine;
        control.options.zero_radius_stop = true;
        control.record_batch(&distance_batch(distances.clone()));
        assert_eq!(control.should_stop(), expected_stop);
        if corrected_cosine && metric == DistanceType::Cosine {
            assert_eq!(control.state.lock().unwrap().best_raw_distances, distances);
        }
    }

    #[rstest::rstest]
    #[case::source(false, 1)]
    #[case::corrected(true, 3)]
    fn test_zero_stop_is_opt_in_and_negative_cosine_is_selectable(
        #[case] corrected_cosine: bool,
        #[case] expected_count: usize,
    ) {
        let mut control = test_control(DistanceType::Cosine);
        control.options.corrected_cosine = corrected_cosine;
        control.record_batch(&distance_batch(vec![
            f32::NAN,
            -1e-7,
            -1e-8,
            0.0,
            f32::INFINITY,
        ]));
        let state = control.state.lock().unwrap();
        assert_eq!(state.best_raw_distances.len(), expected_count);
        assert!(!control.should_stop());
    }

    #[test]
    fn test_zero_stop_respects_floor_and_waits_for_k() {
        let mut control = test_control(DistanceType::L2);
        control.minimum_nprobes = 4;
        control.options.zero_radius_stop = true;
        for _ in 0..3 {
            control.record_batch(&distance_batch(vec![0.0]));
            assert!(!control.should_stop());
        }
        control.record_batch(&distance_batch(vec![0.0]));
        assert!(control.should_stop());
        assert_eq!(
            control.state.lock().unwrap().best_raw_distances,
            vec![0.0; 3]
        );
    }

    #[rstest::rstest]
    #[case::floor_four_cap_twenty("4", "20")]
    #[case::floor_eight_cap_thirty_two("8", "32")]
    #[case::floor_eight_cap_sixty_four("8", "64")]
    fn test_scan_limits_do_not_change_candidate_universe(
        #[case] minimum: &str,
        #[case] maximum: &str,
    ) {
        let mut query = base_query();
        let candidate_count = quake_candidate_count(&query, 1024, 0.1);
        assert_eq!(candidate_count, 102);
        let limits = parse_probe_limits(Some(minimum), Some(maximum), &query).unwrap();
        // Ranking receives the original query. Limits are applied only after
        // its complete candidate list has been produced.
        limits.apply(&mut query, candidate_count).unwrap();
        assert_eq!(query.minimum_nprobes, minimum.parse::<usize>().unwrap());
        assert_eq!(
            query.maximum_nprobes,
            Some(maximum.parse::<usize>().unwrap())
        );
        let mut control = test_control(DistanceType::L2);
        control.boundary_distances = vec![0.0; candidate_count];
        control.partition_sizes = vec![10; candidate_count];
        control.record_batch(&distance_batch(vec![0.01, 0.04, 0.09]));
        assert_eq!(
            control.state.lock().unwrap().probabilities.len(),
            candidate_count
        );
    }

    #[rstest::rstest]
    #[case::zero(Some("0"), None)]
    #[case::negative(None, Some("-1"))]
    #[case::inverted(Some("8"), Some("4"))]
    fn test_scan_limits_reject_invalid_values(
        #[case] minimum: Option<&str>,
        #[case] maximum: Option<&str>,
    ) {
        let error = parse_probe_limits(minimum, maximum, &base_query()).unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(
            error.to_string().contains("experimental")
                || error.to_string().contains("LANCE_EXPERIMENTAL")
        );
    }

    #[test]
    fn test_parse_corrected_and_zero_stop() {
        let policy =
            parse_experimental_probe_policy(Some("quake-corrected"), None, None, None, 10).unwrap();
        assert!(matches!(
            policy,
            ExperimentalProbePolicy::Quake {
                options: QuakeOptions {
                    corrected_cosine: true,
                    zero_radius_stop: false,
                    ..
                },
                ..
            }
        ));
        assert!(parse_bool(EXPERIMENTAL_ZERO_RADIUS_STOP_ENV, Some("true")).unwrap());
        assert!(!parse_bool(EXPERIMENTAL_ZERO_RADIUS_STOP_ENV, None).unwrap());
        let error = parse_bool(EXPERIMENTAL_ZERO_RADIUS_STOP_ENV, Some("1")).unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(
            error
                .to_string()
                .contains(EXPERIMENTAL_ZERO_RADIUS_STOP_ENV)
        );
    }
    #[test]
    fn test_trace_distinguishes_cap_zero_and_tiny_positive_radius() {
        let mut control = test_control(DistanceType::Cosine);
        control.options.trace = true;
        control.options.corrected_cosine = true;
        control.options.zero_radius_stop = true;
        control.maximum_nprobes = 1;
        control.record_batch(&distance_batch(vec![0.01, 0.02, 0.03]));
        let state = control.state.lock().unwrap();
        let trace = &state.traces[0];
        assert_eq!(trace.reason, "scan_cap");
        assert_eq!(trace.partition_id, 0);
        assert_eq!(trace.scanned_vectors, 10);
        assert_eq!(trace.raw_global_kth, Some(0.03));
        assert_eq!(state.probabilities.len(), 2);
        drop(state);

        let mut control = test_control(DistanceType::Cosine);
        control.options.trace = true;
        control.options.zero_radius_stop = true;
        control.record_batch(&distance_batch(vec![0.0, 0.0, f32::MIN_POSITIVE]));
        let state = control.state.lock().unwrap();
        // The source APS epsilon can still stop on its probability estimate;
        // this must never be classified as the explicit exact-zero shortcut.
        assert_ne!(state.traces[0].reason, "zero_radius");
        assert!(state.traces[0].geometry_radius.unwrap() > 0.0);
    }

    #[test]
    fn test_floor_rejects_too_few_candidates() {
        let mut query = base_query();
        let limits = parse_probe_limits(Some("8"), None, &query).unwrap();
        let error = limits.apply(&mut query, 4).unwrap_err();
        assert!(matches!(error, DataFusionError::Execution(_)));
        assert!(error.to_string().contains("candidate count 4"));
    }
    #[test]
    fn test_trace_exposes_source_negative_cosine_omission() {
        let mut control = test_control(DistanceType::Cosine);
        control.options.trace = true;
        control.record_batch(&distance_batch(vec![-1e-7, -1e-8, 0.0, 0.01, 0.02]));
        let state = control.state.lock().unwrap();
        let trace = &state.traces[0];
        assert_eq!(trace.raw_global_kth, Some(0.0));
        assert_eq!(trace.policy_raw_kth, Some(0.02));
        assert_eq!(trace.geometry_radius, Some(0.2));
    }
}
