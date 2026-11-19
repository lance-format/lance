// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! e-means: mini-batch k-means with exponential-moving-average (EMA) centroid
//! updates.
//!
//! This is a direct implementation of Algorithm 1 of Bessa et al., "e-means:
//! outperforming k-means (in one-tenth the epochs)", arXiv:2611.09501:
//!
//! ```text
//! init:   c_j <- k distinct random training vectors
//!         N_j <- 1e-10,  S_j <- N_j * c_j
//! for each of T epochs, for each batch B of a fresh shuffle:
//!   assign: a(x) = argmin_j ||x - c_j||^2           for x in B
//!   batch:  n_j = |{x : a(x) = j}|,  s_j = sum_{a(x) = j} x
//!   EMA:    N_j <- (1 - alpha_t) N_j + alpha_t n_j
//!           S_j <- (1 - alpha_t) S_j + alpha_t s_j,   c_j = S_j / N_j
//!   every 5 batches: split a donor into each cluster with N_j < tau * mean(N)
//! finalize: c_j <- mean of the vectors assigned to j during the last epoch
//! ```
//!
//! `alpha_t` follows a cosine schedule from a peak step down to a tenth of it.
//! The batch size and the peak step default to the paper's scaling laws.
//!
//! Only the L2 distance is supported: e-means minimizes the mean squared error.
//! For cosine similarity, normalize the vectors to unit length first.

use std::f64::consts::PI;
use std::sync::Arc;

use arrow_array::types::{ArrowPrimitiveType, Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, FixedSizeListArray, PrimitiveArray, cast::AsArray};
use arrow_schema::DataType;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use num_traits::Float;
use rand::prelude::*;
use rand::seq::index::sample;
use rayon::prelude::*;

use super::KMeans;
use super::profile::{self, Phase, Work};
use assign::{Assignment, NearestCentroid};

pub(super) mod assign;

/// Count that every cluster starts with. It makes `S_j / N_j` return the
/// initial centroid exactly while giving it almost no weight against the
/// first batch.
const INITIAL_COUNT: f64 = 1e-10;
/// Dead clusters are revived once every this many batches.
const REVIVAL_PERIOD: usize = 5;
/// The cosine schedule ends at this fraction of the peak step.
const STEP_FLOOR_FRACTION: f64 = 0.1;
/// Relative perturbation that splits a donor centroid in two. The paper uses
/// the machine epsilon of bf16, the lowest precision it supports.
const SPLIT_PERTURBATION: f64 = 1.0 / 128.0;
/// Above this many training vectors the paper divides the batch law by 8, as
/// smaller batches gave the same throughput.
const LARGE_DATASET_ROWS: usize = 2_000_000;
/// Minimum vector-centroid coordinate pairs a parallel assignment task scores,
/// about 20 µs of work, so per-task scheduling overhead stays negligible even
/// for the 16-centroid nodes of hierarchical training.
const MIN_TASK_DISTANCE_TERMS: usize = 1 << 16;
/// Upper bound on the parallel tasks one batch is split into, per pass.
const MAX_BATCH_TASKS: usize = 64;
/// Minimum vector components a parallel summation task adds, about 5 µs of
/// work, so small batches are summed by few tasks.
const MIN_TASK_SUM_TERMS: usize = 1 << 14;

/// How many vectors each e-means mini-batch holds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchSize {
    /// The paper's batch law (Equation 7) times `scale`, rounded to the
    /// nearest power of two and capped at the number of training vectors.
    /// `scale` 1 is the law itself; it must be positive and finite. A scale
    /// rather than a count suits hierarchical training, where nodes hold
    /// different numbers of vectors.
    Law { scale: f64 },
    /// This many vectors, in `[1, n]` for `n` training vectors.
    Fixed(usize),
}

impl Default for BatchSize {
    fn default() -> Self {
        Self::Law { scale: 1.0 }
    }
}

/// Parameters of [`train_emeans`].
#[derive(Debug, Clone)]
pub struct EMeansParams {
    /// Passes over the training vectors. The paper reaches the loss of 25
    /// Lloyd iterations in about 2 epochs.
    pub epochs: u32,
    /// Vectors per mini-batch. Defaults to the paper's batch law.
    pub batch_size: BatchSize,
    /// Peak EMA step `alpha_0 = 1 - decay`, in `(0, 1]`. `None` uses the
    /// paper's step law.
    pub peak_step: Option<f64>,
    /// A cluster whose EMA count falls below this fraction of the mean count is
    /// revived by splitting a donor. Must be in `[0, 1)`; `0` disables revival.
    pub revival_threshold: f64,
    /// Seed for initialization, shuffling and donor selection. `None` uses OS
    /// randomness.
    pub seed: Option<u64>,
}

impl Default for EMeansParams {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: BatchSize::default(),
            peak_step: None,
            revival_threshold: 0.1,
            seed: None,
        }
    }
}

/// Train `k` L2 centroids over `data` with e-means.
///
/// Returns a [`KMeans`] whose `loss` is the sum of squared distances measured
/// while assigning the last epoch, as Lloyd's `loss` is measured while
/// assigning its last iteration.
///
/// ```
/// # use arrow_array::{FixedSizeListArray, Float32Array};
/// # use lance_arrow::FixedSizeListArrayExt;
/// # use lance_index::vector::kmeans::emeans::{EMeansParams, train_emeans};
/// # fn test() -> lance_core::Result<()> {
/// let values = Float32Array::from_iter_values((0..4_000).map(|i| (i % 97) as f32));
/// let data = FixedSizeListArray::try_new_from_values(values, 4)?;
/// let params = EMeansParams {
///     seed: Some(42),
///     ..Default::default()
/// };
/// let model = train_emeans(&data, 16, &params)?;
/// assert_eq!(model.centroids.len(), 16 * 4);
/// # Ok(())
/// # }
/// ```
pub fn train_emeans(data: &FixedSizeListArray, k: usize, params: &EMeansParams) -> Result<KMeans> {
    let n = data.len();
    let dimension = data.value_length() as usize;
    if k == 0 {
        return Err(Error::invalid_input("e-means: k must be positive, got 0"));
    }
    if n < k {
        return Err(Error::invalid_input(format!(
            "e-means: need at least k={k} training vectors, got n={n}"
        )));
    }
    if dimension == 0 {
        return Err(Error::invalid_input(
            "e-means: vectors must have a positive dimension, got 0",
        ));
    }
    if params.epochs == 0 {
        return Err(Error::invalid_input(
            "e-means: epochs must be positive, got 0",
        ));
    }
    if !(0.0..1.0).contains(&params.revival_threshold) {
        return Err(Error::invalid_input(format!(
            "e-means: revival_threshold must be in [0, 1), got {}",
            params.revival_threshold
        )));
    }
    let batch_size = match params.batch_size {
        BatchSize::Fixed(batch_size) if batch_size == 0 || batch_size > n => {
            return Err(Error::invalid_input(format!(
                "e-means: batch_size must be in [1, n={n}], got {batch_size}"
            )));
        }
        BatchSize::Fixed(batch_size) => batch_size,
        BatchSize::Law { scale } if !(scale > 0.0 && scale.is_finite()) => {
            return Err(Error::invalid_input(format!(
                "e-means: the batch law scale must be positive and finite, got {scale}"
            )));
        }
        BatchSize::Law { scale } => batch_size_law(n, k, params.epochs, scale),
    };
    let peak_step = match params.peak_step {
        Some(step) if !(step > 0.0 && step <= 1.0) => {
            return Err(Error::invalid_input(format!(
                "e-means: peak_step must be in (0, 1], got {step}"
            )));
        }
        Some(step) => step,
        None => peak_step_law(n, k, batch_size, params.epochs),
    };

    let trainer = Trainer {
        dimension,
        k,
        batch_size,
        peak_step,
        params,
    };
    match data.value_type() {
        DataType::Float16 => trainer.train::<Float16Type>(data),
        DataType::Float32 => trainer.train::<Float32Type>(data),
        DataType::Float64 => trainer.train::<Float64Type>(data),
        value_type => Err(Error::invalid_input(format!(
            "e-means: vectors must be Float16, Float32 or Float64, got {value_type}"
        ))),
    }
}

/// The paper's batch law (Equation 7, Table 2) times `scale`, rounded to the
/// nearest power of two and capped at `n`.
fn batch_size_law(n: usize, k: usize, epochs: u32, scale: f64) -> usize {
    let coefficient = if n > LARGE_DATASET_ROWS {
        0.0044
    } else {
        0.035
    };
    let batch_size = scale
        * coefficient
        * (k as f64).powf(0.13)
        * (n as f64).powf(0.88)
        * f64::from(epochs).powf(0.81);
    let rounded = 2f64.powi(batch_size.max(1.0).log2().round() as i32) as usize;
    rounded.min(n)
}

/// The paper's peak-step law (Equation 6, Table 2), capped at 1.
fn peak_step_law(n: usize, k: usize, batch_size: usize, epochs: u32) -> f64 {
    let step = 0.824
        * (k as f64).powf(-0.46)
        * (batch_size as f64).powf(0.90)
        * (n as f64).powf(-0.45)
        * f64::from(epochs).powf(-0.11);
    step.min(1.0)
}

/// The EMA step of update `t` out of `total`, cosine-annealed from `peak` on
/// the first update to `STEP_FLOOR_FRACTION * peak` on the last (Equation 5).
fn step_at(peak: f64, t: usize, total: usize) -> f64 {
    let cosine = 0.5 * (1.0 + (PI * t as f64 / total as f64).cos());
    peak * (STEP_FLOOR_FRACTION + (1.0 - STEP_FLOOR_FRACTION) * cosine)
}

struct Trainer<'a> {
    dimension: usize,
    k: usize,
    batch_size: usize,
    peak_step: f64,
    params: &'a EMeansParams,
}

impl Trainer<'_> {
    fn train<T: ArrowPrimitiveType>(&self, data: &FixedSizeListArray) -> Result<KMeans>
    where
        T::Native: NearestCentroid,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let Self {
            dimension,
            k,
            batch_size,
            peak_step,
            params,
        } = *self;
        let values = data.values().as_primitive::<T>().values();
        let n = data.len();
        let mut rng = match params.seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        };

        let init_busy = profile::work(Work::Other);
        let mut centroids: Vec<T::Native> = sample(&mut rng, n, k)
            .into_iter()
            .flat_map(|row| &values[row * dimension..(row + 1) * dimension])
            .copied()
            .collect();
        let mut counts = vec![INITIAL_COUNT; k];
        let mut sums: Vec<f64> = centroids
            .iter()
            .map(|value| to_f64(*value) * INITIAL_COUNT)
            .collect();

        let batches_per_epoch = n / batch_size;
        let total_updates = params.epochs as usize * batches_per_epoch;
        let mut order: Vec<usize> = (0..n).collect();
        // Every vector of the last epoch, summed per cluster, for the
        // last-epoch-mean finalization.
        let mut last_epoch_stats = BatchStats::zeros(k, dimension);
        let mut t = 0;
        drop(init_busy);
        for epoch in 0..params.epochs {
            let is_last_epoch = epoch + 1 == params.epochs;
            let shuffle_busy = profile::work(Work::Gather);
            order.shuffle(&mut rng);
            drop(shuffle_busy);
            // The tail that does not fill a batch is skipped; the next shuffle
            // puts other vectors there.
            for batch in order.chunks_exact(batch_size) {
                let assign_phase = profile::phase(Phase::EMeansAssign);
                let batch_stats = assign_batch(values, dimension, &centroids, k, batch);
                drop(assign_phase);

                let _update_phase = profile::phase(Phase::EMeansUpdate);
                let _update_busy = profile::work(Work::Update);
                let step = step_at(peak_step, t, total_updates);
                let decay = 1.0 - step;
                for (count, batch_count) in counts.iter_mut().zip(&batch_stats.counts) {
                    *count = decay * *count + step * batch_count;
                }
                for (sum, batch_sum) in sums.iter_mut().zip(&batch_stats.sums) {
                    *sum = decay * *sum + step * batch_sum;
                }
                if is_last_epoch {
                    last_epoch_stats.merge(&batch_stats);
                }
                t += 1;
                if params.revival_threshold > 0.0 && t % REVIVAL_PERIOD == 0 {
                    revive_dead_clusters(
                        &mut counts,
                        &mut sums,
                        dimension,
                        params.revival_threshold,
                        &mut rng,
                    );
                }
                read_centroids(&counts, &sums, dimension, &mut centroids);
            }
        }

        // Last-epoch-mean finalization: the last epoch's counts and sums are
        // exactly those of its recorded assignments. A cluster that caught no
        // vector in the last epoch keeps its EMA centroid.
        let _finalize_phase = profile::phase(Phase::EMeansFinalize);
        let _finalize_busy = profile::work(Work::Update);
        read_centroids(
            &last_epoch_stats.counts,
            &last_epoch_stats.sums,
            dimension,
            &mut centroids,
        );

        Ok(KMeans {
            centroids: Arc::new(PrimitiveArray::<T>::from(centroids)),
            dimension,
            distance_type: DistanceType::L2,
            loss: last_epoch_stats.loss,
        })
    }
}

/// What one batch contributes to every cluster: `n_j`, `s_j` and the squared
/// distances of its vectors to their centroids.
struct BatchStats {
    counts: Vec<f64>,
    sums: Vec<f64>,
    loss: f64,
}

impl BatchStats {
    fn zeros(k: usize, dimension: usize) -> Self {
        Self {
            counts: vec![0.0; k],
            sums: vec![0.0; k * dimension],
            loss: 0.0,
        }
    }

    fn merge(&mut self, other: &Self) {
        for (count, other) in self.counts.iter_mut().zip(&other.counts) {
            *count += other;
        }
        for (sum, other) in self.sums.iter_mut().zip(&other.sums) {
            *sum += other;
        }
        self.loss += other.loss;
    }
}

/// Rows of a batch per parallel assignment task: enough that a task scores at
/// least `MIN_TASK_DISTANCE_TERMS` vector-centroid coordinate pairs, so the
/// scheduling cost of a task stays small next to its work, and few enough that
/// a batch yields at most `MAX_BATCH_TASKS` tasks.
fn rows_per_task(batch_len: usize, k: usize, dimension: usize) -> usize {
    let min_rows = MIN_TASK_DISTANCE_TERMS.div_ceil(k * dimension);
    min_rows.max(batch_len.div_ceil(MAX_BATCH_TASKS)).max(1)
}

/// Clusters whose sums one parallel summation task owns: enough that a task
/// adds about `MIN_TASK_SUM_TERMS` components, and at most `MAX_BATCH_TASKS`
/// tasks per batch.
fn clusters_per_task(batch_len: usize, k: usize, dimension: usize) -> usize {
    let tasks = (batch_len * dimension / MIN_TASK_SUM_TERMS).clamp(1, MAX_BATCH_TASKS.min(k));
    k.div_ceil(tasks)
}

/// Assigns every vector of `batch` (row ids into `values`) to its nearest
/// centroid and returns the batch's per-cluster statistics.
///
/// Both passes run on the rayon pool, so they nest under the sibling sub-trees
/// of hierarchical training and idle threads steal tasks from either. The
/// assignment pass splits the batch into tasks of [`rows_per_task`] rows. The
/// summation pass splits the clusters into ranges of [`clusters_per_task`]:
/// each task scans the assignments and adds the vectors that joined its
/// clusters straight into the batch's table, so every component is added once
/// and no per-task table is zeroed or merged. Sums are taken in batch order,
/// independent of the thread count. A vector whose distances to every centroid
/// are NaN joins no cluster.
fn assign_batch<N: NearestCentroid>(
    values: &[N],
    dimension: usize,
    centroids: &[N],
    k: usize,
    batch: &[usize],
) -> BatchStats {
    let prepare_busy = profile::work(Work::Assign);
    let prepared = N::prepare_centroids(centroids, dimension);
    drop(prepare_busy);
    let assignments: Vec<Assignment> = batch
        .par_chunks(rows_per_task(batch.len(), k, dimension))
        .flat_map_iter(|rows| {
            let _busy = profile::work(Work::Assign);
            let mut assignments = Vec::with_capacity(rows.len());
            N::assign_rows(
                values,
                dimension,
                centroids,
                &prepared,
                rows,
                &mut assignments,
            );
            assignments
        })
        .collect();

    let mut stats = BatchStats::zeros(k, dimension);
    let counts_busy = profile::work(Work::Update);
    for (cluster, distance) in assignments.iter().flatten() {
        stats.counts[*cluster as usize] += 1.0;
        stats.loss += f64::from(*distance);
    }
    drop(counts_busy);
    let clusters_per_task = clusters_per_task(batch.len(), k, dimension);
    stats
        .sums
        .par_chunks_mut(clusters_per_task * dimension)
        .enumerate()
        .for_each(|(task, sums)| {
            let _busy = profile::work(Work::Update);
            let first_cluster = task * clusters_per_task;
            for (&row, assignment) in batch.iter().zip(&assignments) {
                let Some((cluster, _)) = *assignment else {
                    continue;
                };
                let Some(local) = (cluster as usize).checked_sub(first_cluster) else {
                    continue;
                };
                let Some(sum) = sums.get_mut(local * dimension..(local + 1) * dimension) else {
                    continue;
                };
                for (sum, value) in sum
                    .iter_mut()
                    .zip(&values[row * dimension..(row + 1) * dimension])
                {
                    *sum += to_f64(*value);
                }
            }
        });
    stats
}

fn to_f64<N: Float>(value: N) -> f64 {
    value.to_f64().unwrap_or(f64::NAN)
}

fn from_f64<N: Float>(value: f64) -> N {
    num_traits::cast(value).unwrap_or_else(N::nan)
}

/// Read every centroid off as `S_j / N_j` (Equation 4). A cluster whose count
/// decayed to exactly zero (a full step on a batch that missed it) keeps its
/// previous centroid rather than becoming `0 / 0`.
fn read_centroids<N: Float>(counts: &[f64], sums: &[f64], dimension: usize, centroids: &mut [N]) {
    for (cluster, &count) in counts.iter().enumerate() {
        if count <= 0.0 {
            continue;
        }
        let range = cluster * dimension..(cluster + 1) * dimension;
        for (centroid, sum) in centroids[range.clone()].iter_mut().zip(&sums[range]) {
            *centroid = from_f64(sum / count);
        }
    }
}

/// Revive every cluster whose count fell below `threshold` times the mean
/// count by splitting a donor (Appendix H.1).
///
/// Donors are drawn without replacement among the live clusters with
/// probability proportional to their count. The donor centroid `c` becomes
/// `c * [1 - d, 1 + d, ...]` and the dead one `c * [1 + d, 1 - d, ...]`, so the
/// donor's vectors are split by a hyperplane through `c`; each keeps half of
/// the donor's count. Dead clusters beyond the number of live ones stay dead.
fn revive_dead_clusters(
    counts: &mut [f64],
    sums: &mut [f64],
    dimension: usize,
    threshold: f64,
    rng: &mut impl Rng,
) {
    let floor = threshold * counts.iter().sum::<f64>() / counts.len() as f64;
    let (dead, live): (Vec<usize>, Vec<usize>) =
        (0..counts.len()).partition(|&cluster| counts[cluster] < floor);
    if dead.is_empty() {
        return;
    }
    let Ok(donors) = live.choose_multiple_weighted(rng, dead.len(), |&cluster| counts[cluster])
    else {
        return;
    };
    let donors: Vec<usize> = donors.copied().collect();
    for (&dead, &donor) in dead.iter().zip(&donors) {
        let count = counts[donor] / 2.0;
        for d in 0..dimension {
            let centroid = sums[donor * dimension + d] / counts[donor];
            let (dead_scale, donor_scale) = if d % 2 == 0 {
                (1.0 + SPLIT_PERTURBATION, 1.0 - SPLIT_PERTURBATION)
            } else {
                (1.0 - SPLIT_PERTURBATION, 1.0 + SPLIT_PERTURBATION)
            };
            sums[dead * dimension + d] = centroid * dead_scale * count;
            sums[donor * dimension + d] = centroid * donor_scale * count;
        }
        counts[dead] = count;
        counts[donor] = count;
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Float16Array, Float32Array, Int32Array};
    use half::f16;
    use lance_arrow::FixedSizeListArrayExt;
    use rand_distr::{Distribution, Normal};
    use rstest::rstest;

    use super::*;
    use crate::vector::kmeans::KMeansParams;

    /// `clusters` unit-variance Gaussian blobs of `rows_per_cluster` vectors
    /// each, centered uniformly in `[-spread, spread)^dimension`.
    fn blobs(
        clusters: usize,
        rows_per_cluster: usize,
        dimension: usize,
        spread: f32,
    ) -> FixedSizeListArray {
        let mut rng = SmallRng::seed_from_u64(7);
        let noise = Normal::new(0.0f32, 1.0).unwrap();
        let centers: Vec<f32> = (0..clusters * dimension)
            .map(|_| rng.random_range(-spread..spread))
            .collect();
        let values: Vec<f32> = (0..clusters * rows_per_cluster)
            .flat_map(|row| {
                let center = row % clusters;
                (0..dimension)
                    .map(|d| centers[center * dimension + d] + noise.sample(&mut rng))
                    .collect::<Vec<_>>()
            })
            .collect();
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), dimension as i32)
            .unwrap()
    }

    fn seeded() -> EMeansParams {
        EMeansParams {
            seed: Some(42),
            ..Default::default()
        }
    }

    #[test]
    fn test_emeans_matches_lloyd_loss() {
        // Many overlapping blobs, so the loss landscape resembles embeddings
        // rather than a few separated blobs where any method can merge two.
        let data = blobs(256, 32, 16, 3.0);
        let emeans = train_emeans(&data, 32, &seeded()).unwrap();
        let lloyd_params = KMeansParams {
            seed: Some(42),
            ..Default::default()
        };
        let lloyd = KMeans::new_with_params(&data, 32, &lloyd_params).unwrap();

        let emeans_loss = emeans.compute_loss(&data).unwrap();
        let lloyd_loss = lloyd.compute_loss(&data).unwrap();
        assert!(
            emeans_loss <= lloyd_loss * 1.01,
            "e-means loss {emeans_loss} exceeds Lloyd loss {lloyd_loss}"
        );
    }

    #[test]
    fn test_emeans_loss_is_last_epoch_assignment_loss() {
        let data = blobs(8, 128, 8, 50.0);
        let model = train_emeans(&data, 8, &seeded()).unwrap();
        // The last epoch was assigned against moving centroids, so its loss is
        // close to, but not exactly, the loss of the final centroids.
        let final_loss = model.compute_loss(&data).unwrap();
        assert!(model.loss > 0.0);
        assert!((model.loss - final_loss).abs() / final_loss < 0.1);
    }

    #[test]
    fn test_emeans_seed_is_reproducible() {
        let data = blobs(8, 128, 8, 50.0);
        let first = train_emeans(&data, 8, &seeded()).unwrap();
        let second = train_emeans(&data, 8, &seeded()).unwrap();
        assert_eq!(
            first.centroids.as_primitive::<Float32Type>().values(),
            second.centroids.as_primitive::<Float32Type>().values()
        );
    }

    #[test]
    fn test_emeans_trains_float16() {
        let data = blobs(8, 128, 8, 50.0);
        let values: Vec<f16> = data
            .values()
            .as_primitive::<Float32Type>()
            .values()
            .iter()
            .map(|value| f16::from_f32(*value))
            .collect();
        let data = FixedSizeListArray::try_new_from_values(Float16Array::from(values), 8).unwrap();
        let model = train_emeans(&data, 8, &seeded()).unwrap();
        assert_eq!(model.centroids.data_type(), &DataType::Float16);
        assert_eq!(model.centroids.len(), 8 * 8);
        assert!(
            model
                .centroids
                .as_primitive::<Float16Type>()
                .values()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn test_emeans_full_step_full_batch_keeps_centroids_finite() {
        // A full step on a full batch zeroes the count of every cluster the
        // batch missed; their centroids must not become 0 / 0.
        let data = blobs(4, 64, 4, 50.0);
        let params = EMeansParams {
            batch_size: BatchSize::Fixed(data.len()),
            peak_step: Some(1.0),
            revival_threshold: 0.0,
            ..seeded()
        };
        let model = train_emeans(&data, 32, &params).unwrap();
        assert!(
            model
                .centroids
                .as_primitive::<Float32Type>()
                .values()
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[rstest]
    #[case::zero_k(0, EMeansParams::default())]
    #[case::k_above_n(1_000, EMeansParams::default())]
    #[case::zero_epochs(4, EMeansParams { epochs: 0, ..Default::default() })]
    #[case::zero_batch(4, EMeansParams { batch_size: BatchSize::Fixed(0), ..Default::default() })]
    #[case::batch_above_n(4, EMeansParams { batch_size: BatchSize::Fixed(1_000), ..Default::default() })]
    #[case::zero_batch_scale(4, EMeansParams { batch_size: BatchSize::Law { scale: 0.0 }, ..Default::default() })]
    #[case::nan_batch_scale(4, EMeansParams { batch_size: BatchSize::Law { scale: f64::NAN }, ..Default::default() })]
    #[case::zero_step(4, EMeansParams { peak_step: Some(0.0), ..Default::default() })]
    #[case::step_above_one(4, EMeansParams { peak_step: Some(1.5), ..Default::default() })]
    #[case::nan_step(4, EMeansParams { peak_step: Some(f64::NAN), ..Default::default() })]
    #[case::threshold_one(4, EMeansParams { revival_threshold: 1.0, ..Default::default() })]
    #[case::negative_threshold(4, EMeansParams { revival_threshold: -0.1, ..Default::default() })]
    fn test_emeans_rejects_invalid_input(#[case] k: usize, #[case] params: EMeansParams) {
        let data = blobs(4, 16, 4, 50.0);
        let err = train_emeans(&data, k, &params).unwrap_err();
        assert!(err.to_string().contains("e-means"), "{err}");
    }

    #[test]
    fn test_emeans_rejects_integer_vectors() {
        let data =
            FixedSizeListArray::try_new_from_values(Int32Array::from(vec![1; 64]), 4).unwrap();
        let err = train_emeans(&data, 2, &EMeansParams::default()).unwrap_err();
        assert!(err.to_string().contains("Int32"), "{err}");
    }

    #[test]
    fn test_revive_dead_clusters_splits_donor() {
        // Cluster 1 is dead (count 0 < 0.1 * mean); cluster 0 is the only donor.
        let mut counts = vec![10.0, 0.0];
        let mut sums = vec![20.0, 40.0, 0.0, 0.0];
        let mut rng = SmallRng::seed_from_u64(0);
        revive_dead_clusters(&mut counts, &mut sums, 2, 0.1, &mut rng);

        assert_eq!(counts, vec![5.0, 5.0]);
        let donor = [sums[0] / counts[0], sums[1] / counts[1]];
        let revived = [sums[2] / counts[1], sums[3] / counts[1]];
        let d = SPLIT_PERTURBATION;
        assert!((donor[0] - 2.0 * (1.0 - d)).abs() < 1e-12);
        assert!((donor[1] - 4.0 * (1.0 + d)).abs() < 1e-12);
        assert!((revived[0] - 2.0 * (1.0 + d)).abs() < 1e-12);
        assert!((revived[1] - 4.0 * (1.0 - d)).abs() < 1e-12);
    }

    #[test]
    fn test_revive_dead_clusters_ignores_live_clusters() {
        let mut counts = vec![10.0, 2.0, 5.0];
        let mut sums = vec![1.0, 2.0, 3.0];
        let mut rng = SmallRng::seed_from_u64(0);
        revive_dead_clusters(&mut counts, &mut sums, 1, 0.1, &mut rng);
        assert_eq!(counts, vec![10.0, 2.0, 5.0]);
        assert_eq!(sums, vec![1.0, 2.0, 3.0]);
    }

    /// Every summation task adds exactly the vectors of its clusters, whatever
    /// the split: the batch statistics match a sequential pass.
    #[rstest]
    #[case::one_sum_task(256, 16, 8)]
    #[case::one_cluster_per_task(256, 16, 1_024)]
    #[case::cluster_ranges(4_096, 1_000, 64)]
    fn test_assign_batch_matches_sequential_sums(
        #[case] batch_len: usize,
        #[case] k: usize,
        #[case] dimension: usize,
    ) {
        let mut rng = SmallRng::seed_from_u64(5);
        let mut values: Vec<f32> = (0..2 * batch_len * dimension)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        // A NaN vector joins no cluster.
        values[..dimension].fill(f32::NAN);
        let centroids: Vec<f32> = values[batch_len * dimension..][..k * dimension].to_vec();
        let mut batch: Vec<usize> = (1..2 * batch_len).collect();
        batch.shuffle(&mut rng);
        batch.truncate(batch_len - 1);
        batch.push(0);

        let stats = assign_batch(&values, dimension, &centroids, k, &batch);

        let norms = f32::prepare_centroids(&centroids, dimension);
        let mut assignments = Vec::new();
        f32::assign_rows(
            &values,
            dimension,
            &centroids,
            &norms,
            &batch,
            &mut assignments,
        );
        let mut expected = BatchStats::zeros(k, dimension);
        for (&row, assignment) in batch.iter().zip(&assignments) {
            let Some((cluster, distance)) = *assignment else {
                continue;
            };
            let cluster = cluster as usize;
            expected.counts[cluster] += 1.0;
            expected.loss += f64::from(distance);
            for d in 0..dimension {
                expected.sums[cluster * dimension + d] += f64::from(values[row * dimension + d]);
            }
        }
        assert_eq!(stats.counts, expected.counts);
        assert_eq!(stats.sums, expected.sums);
        assert_eq!(stats.loss, expected.loss);
        assert_eq!(stats.counts.iter().sum::<f64>(), (batch_len - 1) as f64);
    }

    #[rstest]
    #[case::small(100_000, 1_024, 3)]
    #[case::large(10_000_000, 65_536, 2)]
    #[case::capped(1_000, 16, 25)]
    fn test_batch_size_law_is_power_of_two_within_n(
        #[case] n: usize,
        #[case] k: usize,
        #[case] epochs: u32,
    ) {
        let batch_size = batch_size_law(n, k, epochs, 1.0);
        assert!(batch_size <= n);
        assert!(batch_size.is_power_of_two() || batch_size == n);
    }

    #[test]
    fn test_batch_size_law_scale_moves_by_powers_of_two() {
        let law = batch_size_law(8_192, 16, 3, 1.0);
        assert_eq!(batch_size_law(8_192, 16, 3, 4.0), 4 * law);
        assert_eq!(batch_size_law(8_192, 16, 3, 0.25), law / 4);
        // A scale that would overshoot the training vectors is capped there.
        assert_eq!(batch_size_law(8_192, 16, 3, 1e6), 8_192);
    }

    #[test]
    fn test_peak_step_law_is_capped_at_one() {
        assert_eq!(peak_step_law(1_000, 16, 1_000, 1), 1.0);
        let step = peak_step_law(1_000_000, 1_024, 1_024, 4);
        assert!(step > 0.0 && step < 1.0);
    }

    #[test]
    fn test_step_schedule_anneals_to_tenth_of_peak() {
        assert!((step_at(0.5, 0, 100) - 0.5).abs() < 1e-12);
        assert!((step_at(0.5, 100, 100) - 0.05).abs() < 1e-12);
        assert!(step_at(0.5, 30, 100) > step_at(0.5, 60, 100));
    }
}
