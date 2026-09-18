// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! KMeans implementation for Apache Arrow Arrays.
//!
//! Support ``l2``, ``cosine`` and ``dot`` distances, see [DistanceType].
//!
//! ``Cosine`` distance are calculated by normalizing the vectors to unit length,
//! and run ``l2`` distance on the unit vectors.
//!

use core::f32;
use std::cmp::Reverse;
use std::ops::{AddAssign, DivAssign};
use std::sync::Arc;
use std::vec;
use std::{collections::HashMap, ops::MulAssign};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, PrimitiveArray, UInt32Array,
    cast::AsArray,
    types::{ArrowPrimitiveType, Float16Type, Float32Type, Float64Type, UInt8Type},
};
use arrow_array::{ArrowNumericType, UInt8Array};
use arrow_ord::sort::sort_to_indices;
use arrow_schema::{ArrowError, DataType};
use bitvec::prelude::*;
use half::f16;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_linalg::distance::dot_f16::{
    PackedCentroidsF16, amx_fp16_available, amx_fp16_supported, dot_f16_batch_16,
};
use lance_linalg::distance::hamming::{hamming, hamming_distance_batch};
use lance_linalg::distance::{DistanceType, Normalize, dot_distance_batch};
use lance_linalg::kernels::{argmin_value_float, argmin_value_float_with_bias};
use log::{info, warn};
use num_traits::One;
use num_traits::{AsPrimitive, Float, FromPrimitive, Num, Zero};
use rand::prelude::*;
use rayon::prelude::*;
use {
    lance_linalg::distance::{
        Dot,
        l2::{L2, l2_distance_batch},
    },
    lance_linalg::kernels::argmin_value,
};

use crate::vector::utils::SimpleIndex;
use lance_core::{Error, Result};

/// KMean initialization method.
#[derive(Debug, Clone, PartialEq)]
pub enum KMeanInit {
    Random,
    Incremental(Arc<FixedSizeListArray>),
}

/// KMean Training Parameters
#[derive(Clone)]
pub struct KMeansParams {
    /// Max number of iterations.
    pub max_iters: u32,

    /// When the difference of mean distance to the centroids is less than this `tolerance`
    /// threshold, stop the training.
    pub tolerance: f64,

    /// Run kmeans multiple times and pick the best (balanced) one.
    pub redos: usize,

    /// Init methods.
    pub init: KMeanInit,

    /// The metric to calculate distance.
    pub distance_type: DistanceType,

    /// Balance factor for the kmeans clustering.
    /// Higher value means more balanced clustering.
    ///
    /// Setting this value to 0 means no balance factor,
    /// which is the same as normal kmeans clustering.
    pub balance_factor: f32,

    /// The number of sub-clusters each node of the hierarchy is split into.
    ///
    /// Default is 16. A larger fan-out makes a shallower tree with fewer
    /// sub-tree boundaries, which improves the clustering a little at a higher
    /// training cost; a smaller one is faster but loses recall.
    /// Hierarchical k-means is enabled only if hierarchical_k > 1 and k > 256.
    pub hierarchical_k: usize,

    /// Number of global Lloyd iterations run over the whole training sample after
    /// the hierarchical tree is built, with every centroid competing for every
    /// vector. The tree fits each centroid only against its own sub-tree's
    /// vectors, so this pass repairs the boundaries between sub-trees. Each
    /// iteration costs about as much as assigning the training sample to the
    /// centroids once. Off by default (`0`); two iterations typically recover
    /// most of the gap to flat k-means in WCSS and recall. Non-hierarchical
    /// training already iterates globally and ignores it.
    pub refine_iters: u32,

    /// Seed for centroid initialization. `None` draws a seed from the OS, so two
    /// trainings of the same data may pick different initial centroids.
    pub seed: Option<u64>,

    /// Optional sync callback for iteration progress: (current_iteration, max_iterations).
    pub on_progress: Option<Arc<dyn Fn(u32, u32) + Send + Sync>>,
}

impl std::fmt::Debug for KMeansParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KMeansParams")
            .field("max_iters", &self.max_iters)
            .field("tolerance", &self.tolerance)
            .field("redos", &self.redos)
            .field("init", &self.init)
            .field("distance_type", &self.distance_type)
            .field("balance_factor", &self.balance_factor)
            .field("hierarchical_k", &self.hierarchical_k)
            .field("refine_iters", &self.refine_iters)
            .field("seed", &self.seed)
            .field("on_progress", &self.on_progress.as_ref().map(|_| "..."))
            .finish()
    }
}

impl Default for KMeansParams {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tolerance: 1e-4,
            redos: 1,
            init: KMeanInit::Random,
            distance_type: DistanceType::L2,
            balance_factor: 0.0,
            hierarchical_k: 16,
            refine_iters: 0,
            seed: None,
            on_progress: None,
        }
    }
}

impl KMeansParams {
    pub fn new(
        centroids: Option<Arc<FixedSizeListArray>>,
        max_iters: u32,
        redos: usize,
        distance_type: DistanceType,
    ) -> Self {
        let init = match centroids {
            Some(centroids) => KMeanInit::Incremental(centroids),
            None => KMeanInit::Random,
        };
        Self {
            max_iters,
            redos,
            distance_type,
            init,
            ..Default::default()
        }
    }

    /// Set the balance factor for the kmeans clustering.
    ///
    /// Higher value means more balanced clustering.
    /// Setting this value to 0 means no balance factor,
    /// which is the same as normal kmeans clustering.
    pub fn with_balance_factor(mut self, balance_factor: f32) -> Self {
        self.balance_factor = balance_factor;
        self
    }

    pub fn with_on_progress(mut self, cb: Arc<dyn Fn(u32, u32) + Send + Sync>) -> Self {
        self.on_progress = Some(cb);
        self
    }

    /// Set the number of sub-clusters each node of the hierarchy is split into.
    /// See [`KMeansParams::hierarchical_k`].
    pub fn with_hierarchical_k(mut self, hierarchical_k: usize) -> Self {
        self.hierarchical_k = hierarchical_k;
        self
    }

    /// Set the number of global refinement iterations run after hierarchical
    /// training. See [`KMeansParams::refine_iters`].
    pub fn with_refine_iters(mut self, refine_iters: u32) -> Self {
        self.refine_iters = refine_iters;
        self
    }

    /// Seed centroid initialization, making training reproducible.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn rng(&self) -> SmallRng {
        match self.seed {
            Some(seed) => SmallRng::seed_from_u64(seed),
            None => SmallRng::from_os_rng(),
        }
    }

    /// The same parameters with a seed derived for one sub-problem, so sibling
    /// sub-trees of a hierarchical training do not share initial centroids.
    /// Unseeded parameters stay unseeded.
    fn derive(&self, salt: u64) -> Self {
        let mut params = self.clone();
        params.seed = self
            .seed
            .map(|seed| seed ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17));
        params
    }
}

/// Randomly initialize kmeans centroids.
///
///
fn kmeans_random_init<T: ArrowPrimitiveType>(
    data: &[T::Native],
    dimension: usize,
    k: usize,
    mut rng: impl Rng,
    distance_type: DistanceType,
) -> KMeans {
    assert!(data.len() >= k * dimension);
    let chosen = (0..data.len() / dimension).choose_multiple(&mut rng, k);
    let centroids = PrimitiveArray::<T>::from_iter_values(
        chosen
            .iter()
            .flat_map(|&i| data[i * dimension..(i + 1) * dimension].iter())
            .copied(),
    );
    KMeans {
        centroids: Arc::new(centroids),
        dimension,
        distance_type,
        loss: f64::MAX,
    }
}

/// Give every empty cluster half of the currently largest cluster. After the
/// split each of the two has approximately half of the vectors.
///
/// Draining the largest cluster first keeps the result independent of a random
/// draw, so seeded trainings stay reproducible.
fn split_clusters<T: Float + MulAssign>(cnts: &mut [usize], centroids: &mut [T], dim: usize) {
    let eps = T::from(1.0 / 1024.0).unwrap();
    for i in 0..cnts.len() {
        if cnts[i] == 0 {
            let Some(j) = (0..cnts.len()).max_by_key(|&j| cnts[j]) else {
                break;
            };
            if cnts[j] < 2 {
                // Nothing left to split.
                break;
            }

            cnts[i] = cnts[j] / 2;
            cnts[j] -= cnts[i];
            for k in 0..dim {
                if k % 2 == 0 {
                    centroids[i * dim + k] = centroids[j * dim + k] * (T::one() + eps);
                    centroids[j * dim + k] *= T::one() - eps;
                } else {
                    centroids[i * dim + k] = centroids[j * dim + k] * (T::one() - eps);
                    centroids[j * dim + k] *= T::one() + eps;
                }
            }
        }
    }
}

// compute the cluster sizes and return adjusted balance factor
fn compute_cluster_sizes(
    membership: &[Option<u32>],
    radius: &[f32],
    losses: &[f64],
    cluster_sizes: &mut [usize],
) -> f32 {
    cluster_sizes.fill(0);
    let mut max_cluster_id = 0;
    let mut max_cluster_size = 0;
    membership.iter().for_each(|cluster_id| {
        if let Some(cluster_id) = cluster_id {
            let cluster_id = *cluster_id as usize;
            cluster_sizes[cluster_id] += 1;
            if cluster_sizes[cluster_id] > max_cluster_size {
                max_cluster_size = cluster_sizes[cluster_id];
                max_cluster_id = cluster_id;
            }
        }
    });

    (radius[max_cluster_id] - losses[max_cluster_id] as f32 / cluster_sizes[max_cluster_id] as f32)
        / membership.len() as f32
}

fn compute_balance_loss(cluster_sizes: &[usize], n: usize, balance_factor: f32) -> f32 {
    let size_loss = cluster_sizes.iter().map(|size| size.pow(2)).sum::<usize>() as f32;
    balance_factor * (size_loss - n.pow(2) as f32 / cluster_sizes.len() as f32)
}

pub trait KMeansAlgo<T: Num> {
    /// Recompute the membership of each vector.
    ///
    /// Parameters:
    ///
    /// - *data*: a `N * dimension` floating array. Not necessarily normalized.
    ///
    /// Returns:
    /// - *membership*: the membership of each vector.
    /// - *cluster_radius*: the radius of each cluster.
    /// - *losses*: the losses of each cluster.
    fn compute_membership_and_loss(
        centroids: &[T],
        data: &[T],
        dimension: usize,
        distance_type: DistanceType,
        balance_factor: f32,
        cluster_sizes: Option<&[usize]>,
        index: Option<&SimpleIndex>,
    ) -> (Vec<Option<u32>>, Vec<f32>, Vec<f64>) {
        let (membership, dists) = Self::compute_membership_and_dist(
            centroids,
            data,
            dimension,
            distance_type,
            balance_factor,
            cluster_sizes,
            index,
        );

        let k = centroids.len() / dimension;
        let mut cluster_radius = vec![0.0; k];
        let mut losses = vec![0.0; k];
        for (cluster_id, dist) in membership.iter().zip(dists.iter()) {
            if let (Some(cluster_id), Some(dist)) = (cluster_id, dist) {
                let cluster_id = *cluster_id as usize;
                cluster_radius[cluster_id] = cluster_radius[cluster_id].max(*dist);
                losses[cluster_id] += *dist as f64;
            }
        }

        (membership, cluster_radius, losses)
    }

    fn compute_membership_and_dist(
        centroids: &[T],
        data: &[T],
        dimension: usize,
        distance_type: DistanceType,
        balance_factor: f32,
        cluster_sizes: Option<&[usize]>,
        index: Option<&SimpleIndex>,
    ) -> (Vec<Option<u32>>, Vec<Option<f32>>);

    /// Construct a new KMeans model.
    fn to_kmeans(
        data: &[T],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        cluster_sizes: &mut [usize],
        distance_type: DistanceType,
        loss: f64,
    ) -> KMeans;
}

/// Reads a `T::Native` slice as `f16` when — and only when — that is what it is.
///
/// The default body answers `None`, so every element type opts out until it
/// says otherwise, and [`Float16Type`] is the one that overrides it with the
/// identity. That keeps "is this f16?" a compile-time property of `T` for the
/// dot-distance kernel below, rather than a `DataType` comparison paired with a
/// transmute whose correctness the compiler cannot check.
pub(crate) trait MaybeF16: ArrowNumericType {
    fn as_f16_slice(_values: &[Self::Native]) -> Option<&[f16]> {
        None
    }
}

impl MaybeF16 for Float16Type {
    fn as_f16_slice(values: &[f16]) -> Option<&[f16]> {
        Some(values)
    }
}
impl MaybeF16 for Float32Type {}
impl MaybeF16 for Float64Type {}

/// Per-thread score-buffer budget for [`dot_membership_amx_f16`], in f32
/// values: 256 KB, sized to stay within a typical private L2 alongside the
/// vectors and packed centroids a block streams past.
const AMX_DOT_SCRATCH_F32: usize = 64 * 1024;

/// Assigns each row of `data` (row-major `[_, dimension]`) to its nearest
/// centroid under dot distance using the AMX-FP16 GEMM, scoring 32 vectors
/// against every centroid per tile pass instead of one vector at a time.
///
/// `None` — the kernel is unavailable on this build or host, or the shape does
/// not suit it — means the caller must run its own per-vector path. The output
/// is otherwise identical in content and order to that path: `(centroid,
/// distance)` per row, `None` for a row whose distances are all NaN.
///
/// Answers only "can this shape run here": the `LANCE_DISABLE_AMX` kill switch
/// is checked by the caller, so the accelerated path stays directly testable
/// while production traffic honours an operator who turned it off.
fn dot_membership_amx_f16(
    centroids: &[f16],
    data: &[f16],
    dimension: usize,
    balance_factor: f32,
    cluster_sizes: Option<&[usize]>,
) -> Option<Vec<Option<(u32, f32)>>> {
    let k = centroids.len() / dimension;
    // Under one full 32-wide k-pass the GEMM degenerates to the kernel's scalar
    // cleanup, so `dimension` still has to clear 32.
    //
    // `k` only has to clear 16, not the kernel's 32-centroid block width. The
    // hierarchical k-means splitter caps every sub-clustering at
    // `hierarchical_k` (16 by default), so a 32-centroid floor kept the GEMM out
    // of every one of those sub-clusterings -- inside `train_ivf` it only ever
    // ran for the closing assignment against the full centroid set, and during
    // shuffle. A 16-centroid block half-fills the kernel's 32-wide pass with the
    // zero padding `PackedCentroidsF16` already appends, and that half-filled
    // pass still beats scoring one vector at a time, so the sub-clusterings are
    // worth admitting even though they waste half the columns.
    if dimension < 32 || k < 16 {
        return None;
    }
    let packed = PackedCentroidsF16::new(centroids, k, dimension)?;
    let n_padded = packed.num_centroids_padded();
    // Rows per block: as many as the scratch budget buys, rounded down to the
    // kernel's 32-row granularity, and capped so a large input still splits
    // into enough blocks to spread across threads. Very large `k` blows the
    // budget on a single row, hence the lower clamp back to one tile pass.
    let block_rows = ((AMX_DOT_SCRATCH_F32 / n_padded) & !31).clamp(32, 512);
    // Precomputed once, not per row. The bias depends only on the centroid, so
    // rebuilding it inside the loop would repeat `k` multiplications for every
    // one of the `n` vectors -- `n * k` of them across the call, against `k` here.
    let biases: Option<Vec<f32>> = cluster_sizes.map(|sizes| {
        sizes
            .iter()
            .map(|size| balance_factor * *size as f32)
            .collect()
    });
    let biases = || biases.as_deref().map(|b| b.iter().copied());

    Some(
        data.par_chunks(block_rows * dimension)
            .map_init(
                || vec![0f32; block_rows * n_padded],
                |scores, block| {
                    let rows = block.len() / dimension;
                    let tiled = rows - rows % 32;
                    let mut assignments = Vec::with_capacity(rows);

                    packed.score(block, tiled, dimension, scores, n_padded);
                    for row in 0..tiled {
                        // Only the first `k` columns. The rest score the zero
                        // centroids padding `n` up to the kernel's block size,
                        // at distance exactly 1.0 — which beats every real
                        // centroid whose dot product happens to be negative.
                        let dots = &scores[row * n_padded..row * n_padded + k];
                        assignments.push(argmin_value_float_with_bias(
                            dots.iter().map(|dot| 1.0 - dot),
                            biases(),
                        ));
                    }
                    // Rows past the last whole tile pass keep the per-vector path.
                    for vector in block[tiled * dimension..].chunks(dimension) {
                        assignments.push(argmin_value_float_with_bias(
                            dot_distance_batch(vector, centroids, dimension),
                            biases(),
                        ));
                    }
                    assignments
                },
            )
            .flatten_iter()
            .collect(),
    )
}

pub struct KMeansAlgoFloat<T: ArrowNumericType>
where
    T::Native: Float + Num,
{
    phantom_data: std::marker::PhantomData<T>,
}

impl<T: ArrowNumericType + MaybeF16> KMeansAlgo<T::Native> for KMeansAlgoFloat<T>
where
    T::Native: Float + Dot + L2 + MulAssign + DivAssign + AddAssign + FromPrimitive + Sync,
    PrimitiveArray<T>: From<Vec<T::Native>>,
{
    fn compute_membership_and_dist(
        centroids: &[T::Native],
        data: &[T::Native],
        dimension: usize,
        distance_type: DistanceType,
        balance_factor: f32,
        cluster_sizes: Option<&[usize]>,
        index: Option<&SimpleIndex>,
    ) -> (Vec<Option<u32>>, Vec<Option<f32>>) {
        let cluster_and_dists = match index {
            Some(index) => data
                .par_chunks(dimension)
                .map(|vec| {
                    let query = PrimitiveArray::<T>::from_iter_values(vec.iter().copied());
                    // unable to use balance_factor here because index.search returns the closest centroid
                    index
                        .search(Arc::new(query))
                        .map(|(id, dist)| Some((id, dist)))
                        .unwrap()
                })
                .collect::<Vec<_>>(),
            None => match distance_type {
                DistanceType::L2 => data
                    .par_chunks(dimension)
                    .map(|vec| {
                        argmin_value_float_with_bias(
                            l2_distance_batch(vec, centroids, dimension),
                            cluster_sizes
                                .map(|size| size.iter().map(|size| balance_factor * *size as f32)),
                        )
                    })
                    .collect::<Vec<_>>(),
                DistanceType::Dot => T::as_f16_slice(centroids)
                    .zip(T::as_f16_slice(data))
                    // The kill switch is enforced here rather than inside the
                    // kernel wrapper: this is the one place production work is
                    // routed onto the GEMM, and `prefers_flat_amx_assignment`
                    // reads the same flag, so the two stay in lockstep.
                    .filter(|_| amx_fp16_available())
                    .and_then(|(centroids, data)| {
                        dot_membership_amx_f16(
                            centroids,
                            data,
                            dimension,
                            balance_factor,
                            cluster_sizes,
                        )
                    })
                    .unwrap_or_else(|| {
                        data.par_chunks(dimension)
                            .map(|vec| {
                                argmin_value_float_with_bias(
                                    dot_distance_batch(vec, centroids, dimension),
                                    cluster_sizes.map(|size| {
                                        size.iter().map(|size| balance_factor * *size as f32)
                                    }),
                                )
                            })
                            .collect::<Vec<_>>()
                    }),
                _ => {
                    panic!(
                        "KMeans::find_partitions: {} is not supported",
                        distance_type
                    );
                }
            },
        };

        cluster_and_dists.into_iter().map(Option::unzip).unzip()
    }

    fn to_kmeans(
        data: &[T::Native],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        cluster_sizes: &mut [usize],
        distance_type: DistanceType,
        loss: f64,
    ) -> KMeans {
        let mut centroids = vec![T::Native::zero(); k * dimension];
        let threads = get_num_compute_intensive_cpus();

        if let Some(sums) = Self::sum_clusters_over_data(data, dimension, k, membership, threads) {
            centroids
                .par_chunks_mut(dimension)
                .zip(sums.par_chunks(dimension))
                .zip(cluster_sizes.par_iter())
                .for_each(|((centroid, sums), &cnt)| {
                    // An empty cluster keeps the all-zero centroid the
                    // partitioned path would leave it, for `split_clusters`
                    // below to replace.
                    if cnt == 0 {
                        return;
                    }
                    let norm = 1.0 / cnt as f32;
                    centroid.iter_mut().zip(sums).for_each(|(c, s)| {
                        *c = T::Native::from_f32(s * norm).unwrap_or_else(T::Native::zero);
                    });
                });
        } else {
            Self::sum_clusters_over_centroids(
                data,
                dimension,
                k,
                membership,
                &mut centroids,
                threads,
            );

            centroids
                .par_chunks_mut(dimension)
                .zip(cluster_sizes.par_iter())
                .for_each(|(centroid, &cnt)| {
                    if cnt > 0 {
                        let norm = T::Native::one() / T::Native::from_usize(cnt).unwrap();
                        centroid.iter_mut().for_each(|v| *v *= norm);
                    }
                });
        }

        let empty_clusters = cluster_sizes.iter().filter(|&cnt| *cnt == 0).count();
        if empty_clusters as f32 / k as f32 > 0.1 {
            if data.len() / dimension < k * 256 {
                warn!(
                    "KMeans: more than 10% of clusters are empty: {} of {}.\nHelp: this could mean your dataset \
                is too small to have a meaningful index ({} < {}) or has many duplicate vectors.",
                    empty_clusters,
                    k,
                    data.len() / dimension,
                    k * 256
                );
            } else {
                warn!(
                    "KMeans: more than 10% of clusters are empty: {} of {}.\nHelp: this could mean your dataset \
                has many duplicate vectors.",
                    empty_clusters, k
                );
            }
        }

        split_clusters(cluster_sizes, &mut centroids, dimension);

        KMeans {
            centroids: Arc::new(PrimitiveArray::<T>::from(centroids)),
            dimension,
            distance_type,
            loss,
        }
    }
}

/// How much memory the data-parallel centroid update may hold in per-block
/// accumulators at once.
///
/// The strategy trades memory for work: each block sums into a private
/// `k * dimension` buffer, and all of them are live at once while they are
/// folded together, so the footprint is `blocks * k * dimension * 4` bytes.
/// A budget rather than a per-block cap because what has to stay bounded is the
/// footprint of a training round, and the block count is the only free variable
/// in that product.
const CENTROID_SUM_BUDGET_BYTES: usize = 512 << 20;

/// Fewest rows worth giving a block an accumulator of its own.
///
/// Below this the zeroing and the reduction cost more than the summing they
/// parallelize, and the reduction is `k * dimension` per block however few rows
/// the block covers.
const MIN_ROWS_PER_SUM_BLOCK: usize = 1024;

/// Accumulator floats the data-partitioned sum may fold for every membership
/// entry the in-place update would otherwise visit, when that update is itself
/// parallel.
///
/// With `k` centroids spread over `owners` threads the in-place path reads every
/// vector exactly once, and the work it repeats is the walk over the membership
/// array that each owner makes to find its own rows: `n` entries per owner, on
/// all owners at once, so `n` visits of wall time. The data-partitioned path
/// skips that walk but zeroes and then folds `blocks * k * dimension` floats of
/// accumulator, the fold serially, so per row it trades
/// `blocks * k * dimension / n` folded floats for one membership visit.
///
/// Measured on a 64-thread host with both strategies called directly on f32:
/// at 2 folded floats per row (a PQ sub-vector, `dimension` 8, `k` 256) the
/// data path is 1.9-3.6x faster, at 4 (`dimension` 16) 1.9x, at 8 1.24x, and
/// at 16 (`dimension` 64 or 128, `k` 256) it is slower at 0.78-0.87x; the
/// shapes of `benches/kmeans_recompute.rs` with `k * dimension` in the hundreds
/// of thousands or more are 0.13-0.45x. The crossover is near 11; 4 sits on the
/// winning side of it with room to spare.
const FOLD_FLOATS_PER_MEMBERSHIP_VISIT: usize = 4;

/// Accumulator floats the data-partitioned sum may fold for every scalar
/// half-precision add the in-place update would make per owner.
///
/// An f16 column has a second cost on the in-place path that the f32 column
/// does not: `*c += *v` on `f16` is a scalar convert-add-convert on every
/// element an owner accumulates, `n * dimension / owners` of them in wall time,
/// where the data-partitioned path reads the same elements once into an f32
/// accumulator at a fraction of the cost. That saving is proportional to
/// `dimension / owners`, so the allowance for f16 grows with it rather than
/// being a second constant: the same 64 folded floats per row that are 2-3.3x
/// faster at `dimension` 1024 (`k` 64) are break-even at `dimension` 512
/// (`k` 128), and at `dimension` 128 (`k` 256) 32 per row is already 0.96x.
/// Measured against the crossover on the same host -- break-even at 32, 64,
/// 128 and 256 per row for `dimension` 128, 512, 1024 and 2048 respectively,
/// all with 64 threads -- the per-element saving is worth about 8 folded
/// floats; 6 sits on the winning side.
const FOLD_FLOATS_PER_HALF_ADD: usize = 6;

impl<T: ArrowNumericType> KMeansAlgoFloat<T>
where
    T::Native: Float + AddAssign,
{
    /// Sums each cluster's member vectors by splitting the *data*, giving every
    /// block a private f32 accumulator and reducing them at the end.
    ///
    /// `None` when this strategy does not apply, and the caller must use
    /// [`sum_clusters_over_centroids`](Self::sum_clusters_over_centroids)
    /// instead.
    ///
    /// Walking the membership once is what this is for. Partitioning the
    /// *centroids* instead — the only way to write centroids in place without
    /// conflicting — makes each of the `p` blocks walk the whole membership
    /// array to find the rows falling in its own centroid range, so that path
    /// costs `p * n` membership visits against `n` here. Worse, it cannot use
    /// more blocks than there are centroids, and hierarchical k-means
    /// sub-clusters at `hierarchical_k` (16 by default) while a build host has
    /// far more cores than that, so the centroid-partitioned update ran
    /// single-threaded for the whole of `train_ivf` — it was the dominant cost
    /// of training, several times the distance computation the AMX GEMM
    /// accelerates.
    ///
    /// Where the centroid-partitioned path *is* parallel — `k` at least the
    /// thread count — the membership walk (and, for f16, its scalar adds) is
    /// all this path saves, and it pays for the saving with
    /// `blocks * k * dimension` floats of accumulator to zero and fold; see
    /// [`FOLD_FLOATS_PER_MEMBERSHIP_VISIT`] and [`FOLD_FLOATS_PER_HALF_ADD`]
    /// for where that trade stops paying and this path hands the shape back.
    ///
    /// Every block accumulates in f32 whatever `T::Native` is, and that is why
    /// an f16 column takes this path even when there is only one block to give
    /// it. The private accumulators are fresh memory either way, so widening them
    /// to f32 costs one extra copy of `k * dimension` per block and nothing in
    /// the read of the data. What it buys is a guard against f16's 11 significant
    /// bits: a running f16 total past 2048 can no longer represent an increment
    /// of 1, so a cluster of a few thousand rows sums to a fraction of its true
    /// total and the mean comes out wrong rather than slow. `train_kmeans` caps
    /// its input at `k * 512` rows precisely so the in-place f16 update never
    /// reaches that regime, and this path keeps the same property without
    /// leaning on the cap. The centroid is a mean, so an f32 total divided by
    /// the count lands back in f16 range whatever the cluster size.
    ///
    /// `available_threads` is a parameter rather than a read of
    /// [`get_num_compute_intensive_cpus`] so the block arithmetic — and with it
    /// which strategy a shape lands on — is reproducible in a test on any host.
    /// It is the right bound on the block count and not just a convenient one:
    /// past one block per thread there is no parallelism left to buy, only more
    /// partial sums to fold, and that fold is `k * dimension` per block.
    ///
    /// The sums are therefore a function of the core count, and two hosts with
    /// different core counts will differ in the last bits of a centroid. Within
    /// a host the result is reproducible — the partial sums are folded in block
    /// order, not in completion order — which is what debugging a training run
    /// needs. Reproducibility *across* hosts is not something this path could
    /// offer anyway: `train_kmeans` seeds its initial centroids from
    /// `SmallRng::from_os_rng`.
    fn sum_clusters_over_data(
        data: &[T::Native],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        available_threads: usize,
    ) -> Option<Vec<f32>> {
        // An f64 column has more mantissa than the accumulator would keep, so it
        // stays on the path that accumulates in its own element type.
        if T::DATA_TYPE == DataType::Float64 {
            return None;
        }
        let width = k.checked_mul(dimension)?;
        let per_block_bytes = width.checked_mul(size_of::<f32>())?.max(1);
        // Checked before the block count and not folded into it: the count is
        // clamped up to 1, so a budget that cannot pay for even a single
        // accumulator would otherwise still allocate one.
        if per_block_bytes > CENTROID_SUM_BUDGET_BYTES {
            return None;
        }
        let n = data.len() / dimension;
        debug_assert_eq!(membership.len(), n);
        // `par_chunks` rejects a zero chunk size, which is what an empty input
        // would compute below. No rows means every sum is zero.
        if n == 0 {
            return Some(vec![0f32; width]);
        }

        let blocks = available_threads
            .min(CENTROID_SUM_BUDGET_BYTES / per_block_bytes)
            .min(n.div_ceil(MIN_ROWS_PER_SUM_BLOCK))
            .max(1);
        // A single block buys no parallelism, so for a type whose own arithmetic
        // is exact enough the in-place path is the cheaper way to the same
        // answer. f16 is the exception above: it takes the wider accumulator on
        // any core count.
        if blocks < 2 && T::DATA_TYPE != DataType::Float16 {
            return None;
        }
        // Against an in-place update that is itself parallel, this path only
        // pays off while the accumulators it folds stay small next to the work
        // it saves that update: the membership walk, and for f16 the scalar
        // adds. A single f16 block is exempt for the same reason as above: it is
        // here for the accumulator, and it has no partial sums to fold. An
        // overflow in the arithmetic hands the shape back too, which is the
        // safe way round.
        let owners = Self::centroid_owner_blocks(k, available_threads);
        if blocks >= 2 && owners > 1 {
            let mut allowance = n.checked_mul(FOLD_FLOATS_PER_MEMBERSHIP_VISIT)?;
            if T::DATA_TYPE == DataType::Float16 {
                let half_adds_per_owner = n.checked_mul(dimension)? / owners;
                allowance = allowance
                    .checked_add(half_adds_per_owner.checked_mul(FOLD_FLOATS_PER_HALF_ADD)?)?;
            }
            if blocks.checked_mul(width)? > allowance {
                return None;
            }
        }
        let rows_per_block = n.div_ceil(blocks);

        let partials: Vec<Vec<f32>> = data
            .par_chunks(rows_per_block * dimension)
            .zip(membership.par_chunks(rows_per_block))
            .map(|(rows, ids)| {
                let mut sums = vec![0f32; width];
                rows.chunks_exact(dimension)
                    .zip(ids)
                    .filter_map(|(vector, id)| id.map(|id| (vector, id as usize)))
                    // A membership out of range would index past the end of the
                    // accumulator. The centroid-partitioned path drops those
                    // rows silently by construction, so this one does too.
                    .filter(|&(_, cluster_id)| cluster_id < k)
                    .for_each(|(vector, cluster_id)| {
                        let sums = &mut sums[cluster_id * dimension..(cluster_id + 1) * dimension];
                        // Fully qualified: importing `ToPrimitive` here would
                        // shadow `half::f16`'s inherent `to_f32` -- which
                        // returns `f32`, not `Option<f32>` -- for every caller
                        // in this module.
                        sums.iter_mut().zip(vector).for_each(|(s, v)| {
                            *s += num_traits::ToPrimitive::to_f32(v).unwrap_or_default()
                        });
                    });
                sums
            })
            .collect();

        // Folded in block order rather than reduced pairwise. `reduce` would
        // join the partial sums in whatever order the work stealing produced,
        // so a centroid's last bits would vary between two runs on the same
        // host.
        //
        // Sequentially, and not striped across threads by output position:
        // striping measured 75% slower than this at the shape that matters,
        // and this is within 3% of the unordered `reduce` it replaces. The fold
        // is `blocks * k * dimension` additions over memory that is read
        // straight through, while a stripe narrow enough to give every thread
        // one reaches into all `blocks` allocations for a few hundred bytes
        // each. Hierarchical k-means sub-clusters at `hierarchical_k`, so the
        // accumulator here is 16 centroids wide -- kilobytes, not megabytes,
        // and already cheap to walk once per block.
        let mut partials = partials.into_iter();
        let mut sums = partials.next().unwrap_or_else(|| vec![0f32; width]);
        for partial in partials {
            sums.iter_mut().zip(&partial).for_each(|(s, p)| *s += *p);
        }
        Some(sums)
    }

    /// How many blocks [`sum_clusters_over_centroids`](Self::sum_clusters_over_centroids)
    /// splits the centroids into: one per thread, or a single one when there is
    /// not a centroid per thread to give, or fewer than 16 in all.
    ///
    /// Every block needs at least one centroid of its own, and below 16 the
    /// membership walk this repeats per block costs more than the parallelism
    /// returns. Shared with the data-partitioned path so that its routing rule
    /// can tell a parallel in-place update from a serial one by the same test
    /// the update itself makes.
    fn centroid_owner_blocks(k: usize, available_threads: usize) -> usize {
        let num_cpus = available_threads.max(1);
        if k < num_cpus || k < 16 { 1 } else { num_cpus }
    }

    /// Sums each cluster's member vectors by splitting the *centroids*, each
    /// block scanning the whole dataset for the rows assigned to the centroids
    /// it owns.
    ///
    /// The fallback for the cases
    /// [`sum_clusters_over_data`](Self::sum_clusters_over_data) declines: an f64
    /// column, a `k * dimension` too large to hold an accumulator for, a non-f16
    /// column with only one block's worth of rows, and — when this path runs
    /// parallel — a shape whose accumulators would cost more to fold than the
    /// membership walks they save. All four are cases where accumulating in
    /// place is the cheaper trade.
    ///
    /// `available_threads` is a parameter for the same reason it is one there:
    /// so a test can pin the partitioning without depending on the host.
    fn sum_clusters_over_centroids(
        data: &[T::Native],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        centroids: &mut [T::Native],
        available_threads: usize,
    ) {
        let num_cpus = Self::centroid_owner_blocks(k, available_threads);
        let chunk_size = k / num_cpus;

        centroids
            .par_chunks_mut(dimension * chunk_size)
            .enumerate()
            .with_max_len(1)
            .for_each(|(i, centroids)| {
                let start = i * chunk_size;
                let end = ((i + 1) * chunk_size).min(k);
                data.chunks(dimension)
                    .zip(membership.iter())
                    .filter_map(|(vector, cluster_id)| {
                        cluster_id.map(|cluster_id| (vector, cluster_id as usize))
                    })
                    .for_each(|(vector, cluster_id)| {
                        if start <= cluster_id && cluster_id < end {
                            let local_id = cluster_id - start;
                            let centroid =
                                &mut centroids[local_id * dimension..(local_id + 1) * dimension];
                            centroid.iter_mut().zip(vector).for_each(|(c, v)| *c += *v);
                        }
                    });
            });
    }
}

struct KModeAlgo {}

impl KMeansAlgo<u8> for KModeAlgo {
    fn compute_membership_and_dist(
        centroids: &[u8],
        data: &[u8],
        dimension: usize,
        distance_type: DistanceType,
        balance_factor: f32,
        cluster_sizes: Option<&[usize]>,
        _: Option<&SimpleIndex>,
    ) -> (Vec<Option<u32>>, Vec<Option<f32>>) {
        assert_eq!(distance_type, DistanceType::Hamming);
        let cluster_and_dists = data
            .par_chunks(dimension)
            .map(|vec| {
                argmin_value(
                    centroids
                        .chunks_exact(dimension)
                        .enumerate()
                        .map(|(id, c)| {
                            hamming(vec, c)
                                + balance_factor
                                    * cluster_sizes.map(|sizes| sizes[id] as f32).unwrap_or(0.0)
                        }),
                )
            })
            .collect::<Vec<_>>();
        cluster_and_dists.into_iter().map(Option::unzip).unzip()
    }

    fn to_kmeans(
        data: &[u8],
        dimension: usize,
        k: usize,
        membership: &[Option<u32>],
        _cluster_sizes: &mut [usize],
        distance_type: DistanceType,
        loss: f64,
    ) -> KMeans {
        assert_eq!(distance_type, DistanceType::Hamming);

        let mut clusters = HashMap::<u32, Vec<usize>>::new();
        membership.iter().enumerate().for_each(|(i, part_id)| {
            if let Some(part_id) = part_id {
                clusters.entry(*part_id).or_default().push(i);
            }
        });
        let centroids = (0..k as u32)
            .into_par_iter()
            .flat_map(|part_id| {
                if let Some(vecs) = clusters.get(&part_id) {
                    let mut ones = vec![0_u32; dimension * 8];
                    let cnt = vecs.len() as u32;
                    vecs.iter().for_each(|&i| {
                        let vec = &data[i * dimension..(i + 1) * dimension];
                        ones.iter_mut()
                            .zip(vec.view_bits::<Lsb0>())
                            .for_each(|(c, v)| {
                                if *v.as_ref() {
                                    *c += 1;
                                }
                            });
                    });

                    let bits = ones.iter().map(|&c| c * 2 > cnt).collect::<BitVec<u8>>();
                    bits.as_raw_slice()
                        .iter()
                        .copied()
                        .map(Some)
                        .collect::<Vec<_>>()
                } else {
                    vec![None; dimension]
                }
            })
            .collect::<Vec<_>>();

        KMeans {
            centroids: Arc::new(UInt8Array::from(centroids)),
            dimension,
            distance_type,
            loss,
        }
    }
}

/// Cluster id assignment for each vector in a batch.
pub type KMeansMembership = Vec<Option<u32>>;

/// Distance from each vector to its assigned centroid.
pub type KMeansDistances = Vec<Option<f32>>;

/// Maximum assignment distance per centroid.
pub type KMeansClusterRadii = Vec<f32>;

/// Sum of assignment distances per centroid.
pub type KMeansClusterLosses = Vec<f64>;

/// Batch assignment results with per-centroid radii and losses.
pub type KMeansMembershipAndLoss = (KMeansMembership, KMeansClusterRadii, KMeansClusterLosses);

/// Batch assignment results with per-vector distances.
pub type KMeansMembershipAndDistances = (KMeansMembership, KMeansDistances);

/// Rows gathered per centroid to train one node of the hierarchy; the same
/// prefix `train_kmeans` keeps for itself.
const TRAINING_ROWS_PER_CENTROID: usize = 512;
/// Bytes of vectors gathered at a time when assigning a node's rows to its
/// sub-clusters. One such chunk is held per worker thread, so this is sized in
/// bytes rather than rows: a row-count budget would hold a multiple of it on
/// high-dimensional data.
const MEMBERSHIP_CHUNK_BYTES: usize = 4 * 1024 * 1024;

/// One leaf of the proportional hierarchy: a centroid and the training vectors
/// it was fitted on.
struct Leaf<N> {
    centroid: Vec<N>,
    indices: Vec<usize>,
}

/// Apportion `quota` centroids over sub-clusters of the given sizes so that each
/// sub-cluster's share follows its share of the vectors (largest-remainder
/// rounding). A sub-cluster never gets more centroids than it has vectors, and
/// one whose share rounds to zero gets none.
fn allocate_quotas(sizes: &[usize], quota: usize) -> Vec<usize> {
    let total: usize = sizes.iter().sum();
    debug_assert!(
        total > 0,
        "cannot apportion centroids over empty sub-clusters"
    );
    let mut quotas: Vec<usize> = sizes
        .iter()
        .map(|&size| (quota * size / total).min(size))
        .collect();
    let mut remaining = quota.saturating_sub(quotas.iter().sum::<usize>());
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&i| (Reverse(quota * sizes[i] % total), Reverse(sizes[i])));
    // Hand out the remainder by fractional share, going around again while some
    // sub-cluster still has room for another centroid.
    while remaining > 0 {
        let mut handed = 0;
        for &i in &order {
            if remaining == 0 {
                break;
            }
            if quotas[i] < sizes[i] {
                quotas[i] += 1;
                remaining -= 1;
                handed += 1;
            }
        }
        if handed == 0 {
            break;
        }
    }
    quotas
}

/// KMeans implementation for Apache Arrow Arrays.
#[derive(Debug, Clone)]
pub struct KMeans {
    /// Flattened array of centroids.
    ///
    /// dimension * k of floating number.
    pub centroids: ArrayRef,

    /// The dimension of each vector.
    pub dimension: usize,

    /// How to calculate distance between two vectors.
    pub distance_type: DistanceType,

    /// The loss of the last training.
    pub loss: f64,
}

impl KMeans {
    fn empty(dimension: usize, distance_type: DistanceType) -> Self {
        Self {
            centroids: arrow_array::array::new_empty_array(&DataType::Float32),
            dimension,
            distance_type,
            loss: f64::MAX,
        }
    }

    /// Create a [`KMeans`] with existing centroids.
    /// It is useful for continuing training.
    pub fn with_centroids(
        centroids: ArrayRef,
        dimension: usize,
        distance_type: DistanceType,
        loss: f64,
    ) -> Self {
        assert!(matches!(
            centroids.data_type(),
            DataType::Float16 | DataType::Float32 | DataType::Float64 | DataType::UInt8
        ));
        Self {
            centroids,
            dimension,
            distance_type,
            loss,
        }
    }

    /// Initialize a [`KMeans`] with random centroids.
    ///
    /// Parameters
    /// - *data*: training data. provided to do samplings.
    /// - *k*: the number of clusters.
    /// - *distance_type*: the distance type to calculate distance.
    /// - *rng*: random generator.
    fn init_random<T: ArrowPrimitiveType>(
        data: &[T::Native],
        dimension: usize,
        k: usize,
        rng: impl Rng,
        distance_type: DistanceType,
    ) -> Self {
        kmeans_random_init::<T>(data, dimension, k, rng, distance_type)
    }

    /// Train a KMeans model on data with `k` clusters.
    pub fn new(data: &FixedSizeListArray, k: usize, max_iters: u32) -> arrow::error::Result<Self> {
        let params = KMeansParams {
            max_iters,
            distance_type: DistanceType::L2,
            ..Default::default()
        };
        Self::new_with_params(data, k, &params)
    }

    /// Assign a batch of vectors to these centroids and return membership, radius, and loss.
    pub fn compute_membership_and_loss(
        &self,
        data: &FixedSizeListArray,
    ) -> arrow::error::Result<KMeansMembershipAndLoss> {
        let (membership, distances) = self.compute_membership_and_distances(data)?;
        let k = self.centroids.len() / self.dimension;
        let mut cluster_radius: Vec<f32> = vec![0.0_f32; k];
        let mut losses = vec![0.0; k];
        for (cluster_id, dist) in membership.iter().zip(distances.iter()) {
            if let (Some(cluster_id), Some(dist)) = (cluster_id, dist) {
                let cluster_id = *cluster_id as usize;
                cluster_radius[cluster_id] = cluster_radius[cluster_id].max(*dist);
                losses[cluster_id] += *dist as f64;
            }
        }
        Ok((membership, cluster_radius, losses))
    }

    /// Assign a batch of vectors to these centroids and return per-vector distances.
    pub fn compute_membership_and_distances(
        &self,
        data: &FixedSizeListArray,
    ) -> arrow::error::Result<KMeansMembershipAndDistances> {
        if data.value_length() as usize != self.dimension {
            return Err(ArrowError::InvalidArgumentError(format!(
                "KMeans: data dimension {} does not match centroid dimension {}",
                data.value_length(),
                self.dimension
            )));
        }

        let index = SimpleIndex::may_train_index(
            self.centroids.clone(),
            self.dimension,
            self.distance_type,
        )
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
        match (
            data.value_type(),
            self.centroids.data_type(),
            self.distance_type,
        ) {
            (DataType::Float16, DataType::Float16, _) => {
                let data_values = data.values().as_primitive::<Float16Type>().values();
                let centroids = self.centroids.as_primitive::<Float16Type>().values();
                Ok(KMeansAlgoFloat::<Float16Type>::compute_membership_and_dist(
                    centroids,
                    data_values,
                    self.dimension,
                    self.distance_type,
                    0.0,
                    None,
                    index.as_ref(),
                ))
            }
            (DataType::Float32, DataType::Float32, _) => {
                let data_values = data.values().as_primitive::<Float32Type>().values();
                let centroids = self.centroids.as_primitive::<Float32Type>().values();
                Ok(KMeansAlgoFloat::<Float32Type>::compute_membership_and_dist(
                    centroids,
                    data_values,
                    self.dimension,
                    self.distance_type,
                    0.0,
                    None,
                    index.as_ref(),
                ))
            }
            (DataType::Float64, DataType::Float64, _) => {
                let data_values = data.values().as_primitive::<Float64Type>().values();
                let centroids = self.centroids.as_primitive::<Float64Type>().values();
                Ok(KMeansAlgoFloat::<Float64Type>::compute_membership_and_dist(
                    centroids,
                    data_values,
                    self.dimension,
                    self.distance_type,
                    0.0,
                    None,
                    index.as_ref(),
                ))
            }
            (DataType::UInt8, DataType::UInt8, DistanceType::Hamming) => {
                let data_values = data.values().as_primitive::<UInt8Type>().values();
                let centroids = self.centroids.as_primitive::<UInt8Type>().values();
                Ok(KModeAlgo::compute_membership_and_dist(
                    centroids,
                    data_values,
                    self.dimension,
                    self.distance_type,
                    0.0,
                    None,
                    index.as_ref(),
                ))
            }
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "KMeans: can not compute membership for data type {} with centroid type {} and distance type {}",
                data.value_type(),
                self.centroids.data_type(),
                self.distance_type
            ))),
        }
    }

    /// Compute the kmeans loss for a batch of vectors against these centroids.
    pub fn compute_loss(&self, data: &FixedSizeListArray) -> arrow::error::Result<f64> {
        let (_, _, losses) = self.compute_membership_and_loss(data)?;
        Ok(losses.iter().sum())
    }

    fn train_kmeans<T: ArrowNumericType, Algo: KMeansAlgo<T::Native>>(
        data: &FixedSizeListArray,
        k: usize,
        params: &KMeansParams,
    ) -> arrow::error::Result<Self>
    where
        T::Native: Num,
    {
        // the data is `num_partitions * sample_rate` vectors,
        // but here `k` may be not `num_partitions` in the case of hierarchical kmeans,
        // so we need to sample the sampled data again here.
        // we have to limit the number of data to avoid division underflow,
        // the threshold 512 is chosen because the minimal normal f16 value will be 0 if divided by 1024.
        let data = if data.len() >= k * 512 {
            data.slice(0, k * 512)
        } else {
            data.clone()
        };

        let n = data.len();
        let dimension = data.value_length() as usize;

        let data =
            data.values()
                .as_primitive_opt::<T>()
                .ok_or(ArrowError::InvalidArgumentError(format!(
                    "KMeans: data must be {}, got: {}",
                    T::DATA_TYPE,
                    data.value_type()
                )))?;

        let mut best_kmeans = Self::empty(dimension, params.distance_type);
        let mut cluster_sizes = vec![0; k];
        let mut adjusted_balance_factor = f32::MAX;

        let mut rng = params.rng();
        for redo in 1..=params.redos {
            let mut kmeans: Self = match &params.init {
                KMeanInit::Random => Self::init_random::<T>(
                    data.values(),
                    dimension,
                    k,
                    &mut rng,
                    params.distance_type,
                ),
                KMeanInit::Incremental(centroids) => Self::with_centroids(
                    centroids.values().clone(),
                    dimension,
                    params.distance_type,
                    f64::MAX,
                ),
            };

            let mut loss = f64::MAX;
            for i in 1..=params.max_iters {
                if let Some(cb) = &params.on_progress {
                    cb(i, params.max_iters);
                }
                if i % 10 == 0 {
                    info!(
                        "KMeans training: iteration {} / {}, redo={}",
                        i, params.max_iters, redo
                    );
                };

                let index = SimpleIndex::may_train_index(
                    kmeans.centroids.clone(),
                    kmeans.dimension,
                    kmeans.distance_type,
                )?;

                let balance_factor = adjusted_balance_factor.min(params.balance_factor);
                let (membership, radius, losses) = Algo::compute_membership_and_loss(
                    kmeans.centroids.as_primitive::<T>().values(),
                    data.values(),
                    dimension,
                    params.distance_type,
                    balance_factor,
                    Some(&cluster_sizes),
                    index.as_ref(),
                );

                adjusted_balance_factor =
                    compute_cluster_sizes(&membership, &radius, &losses, &mut cluster_sizes);
                let balance_loss = compute_balance_loss(&cluster_sizes, n, balance_factor);
                let last_loss = losses.iter().sum::<f64>() + balance_loss as f64;

                kmeans = Algo::to_kmeans(
                    data.values(),
                    dimension,
                    k,
                    &membership,
                    &mut cluster_sizes,
                    params.distance_type,
                    last_loss,
                );
                if (loss - last_loss).abs() < params.tolerance * last_loss {
                    info!(
                        "KMeans training: converged at iteration {} / {}, redo={}, loss={}, last_loss={}, loss_diff={}",
                        i,
                        params.max_iters,
                        redo,
                        loss,
                        last_loss,
                        (loss - last_loss).abs() / last_loss
                    );
                    break;
                }
                loss = last_loss;
            }
            if kmeans.loss < best_kmeans.loss {
                best_kmeans = kmeans;
            }
        }

        Ok(best_kmeans)
    }

    /// Helper function to create a FixedSizeListArray from indices
    /// Nearest centroid of every row in `indices`, computed over gathered chunks
    /// so that no copy of the whole node is ever held.
    fn membership_of_rows<T: ArrowNumericType, Algo: KMeansAlgo<T::Native>>(
        centroids: &[T::Native],
        data_values: &[T::Native],
        dimension: usize,
        indices: &[usize],
        distance_type: DistanceType,
    ) -> Vec<Option<u32>>
    where
        T::Native: Num,
    {
        let chunk_rows = (MEMBERSHIP_CHUNK_BYTES / (dimension * size_of::<T::Native>())).max(1);
        indices
            .par_chunks(chunk_rows)
            .flat_map_iter(|chunk| {
                let mut rows = Vec::with_capacity(chunk.len() * dimension);
                for &idx in chunk {
                    rows.extend_from_slice(&data_values[idx * dimension..(idx + 1) * dimension]);
                }
                let (membership, _) = Algo::compute_membership_and_dist(
                    centroids,
                    &rows,
                    dimension,
                    distance_type,
                    0.0,
                    None,
                    None,
                );
                membership
            })
            .collect()
    }

    fn create_array_from_indices<T: ArrowNumericType>(
        indices: &[usize],
        data_values: &[T::Native],
        dimension: usize,
    ) -> arrow::error::Result<FixedSizeListArray>
    where
        T::Native: Clone,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let mut subset_data = Vec::with_capacity(indices.len() * dimension);
        for &idx in indices {
            let start = idx * dimension;
            let end = start + dimension;
            subset_data.extend_from_slice(&data_values[start..end]);
        }
        let array = PrimitiveArray::<T>::from(subset_data);
        FixedSizeListArray::try_new_from_values(array, dimension as i32)
    }

    /// Train a hierarchical KMeans model when k > 256.
    ///
    /// Grows the centroid tree with size-proportional quotas
    /// ([`Self::train_hierarchical_proportional`]), then optionally refines every
    /// centroid against the whole sample ([`KMeansParams::refine_iters`]).
    fn train_hierarchical_kmeans<T: ArrowNumericType, Algo: KMeansAlgo<T::Native>>(
        data: &FixedSizeListArray,
        target_k: usize,
        params: &KMeansParams,
    ) -> arrow::error::Result<Self>
    where
        T::Native: Num,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let tree = Self::train_hierarchical_proportional::<T, Algo>(data, target_k, params)?;
        if params.refine_iters == 0 {
            return Ok(tree);
        }
        info!(
            "Hierarchical clustering: refining {} centroids for {} iterations",
            target_k, params.refine_iters
        );
        let centroids =
            FixedSizeListArray::try_new_from_values(tree.centroids.clone(), tree.dimension as i32)?;
        let refine_params = KMeansParams {
            init: KMeanInit::Incremental(Arc::new(centroids)),
            max_iters: params.refine_iters,
            redos: 1,
            ..params.clone()
        };
        Self::train_kmeans::<T, Algo>(data, target_k, &refine_params)
    }

    /// Hierarchical training that gives every sub-cluster a centroid quota
    /// proportional to its share of the vectors and grows the sub-trees in
    /// parallel.
    ///
    /// Every node holds `n` vectors and owes `quota` centroids. It runs one small
    /// k-means with `min(hierarchical_k, quota)` centroids over its vectors and
    /// hands each sub-cluster `quota * n_i / n` of the quota, so a leaf ends up
    /// with roughly `n / k` training vectors however skewed the data is. The
    /// recursion bottoms out where a node's quota fits one k-means run; there the
    /// sub-clusters become the leaves, each centroid the mean of its vectors.
    fn train_hierarchical_proportional<T: ArrowNumericType, Algo: KMeansAlgo<T::Native>>(
        data: &FixedSizeListArray,
        target_k: usize,
        params: &KMeansParams,
    ) -> arrow::error::Result<Self>
    where
        T::Native: Num,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let n = data.len();
        let dimension = data.value_length() as usize;
        let data_values = data
            .values()
            .as_primitive_opt::<T>()
            .ok_or(ArrowError::InvalidArgumentError(format!(
                "KMeans: data must be {}, got: {}",
                T::DATA_TYPE,
                data.value_type()
            )))?
            .values();
        info!(
            "Hierarchical clustering (proportional): branching {}, target k={}",
            params.hierarchical_k.max(2),
            target_k
        );

        let indices: Vec<usize> = (0..n).collect();
        let mut leaves = Self::grow_proportional_subtree::<T, Algo>(
            data_values,
            dimension,
            indices,
            target_k,
            params,
            1,
        )?;
        Self::fill_missing_leaves::<T, Algo>(
            data_values,
            dimension,
            &mut leaves,
            target_k,
            params,
        )?;

        let centroids: Vec<T::Native> = leaves.into_iter().flat_map(|leaf| leaf.centroid).collect();
        Ok(Self {
            centroids: Arc::new(PrimitiveArray::<T>::from(centroids)),
            dimension,
            distance_type: params.distance_type,
            loss: 0.0, // Loss is not meaningful for hierarchical clustering
        })
    }

    /// Grow the sub-tree under one node of the proportional hierarchy.
    ///
    /// `indices` are the node's training vectors and `quota` the number of leaf
    /// centroids it owes. `salt` derives per-node seeds from a seeded training,
    /// so sibling sub-trees do not repeat each other's initialization.
    fn grow_proportional_subtree<T: ArrowNumericType, Algo: KMeansAlgo<T::Native>>(
        data_values: &[T::Native],
        dimension: usize,
        indices: Vec<usize>,
        quota: usize,
        params: &KMeansParams,
        salt: u64,
    ) -> arrow::error::Result<Vec<Leaf<T::Native>>>
    where
        T::Native: Num,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let n = indices.len();
        if quota <= 1 || n <= 1 {
            return Ok(vec![Self::leaf_from_indices::<T, Algo>(
                data_values,
                dimension,
                indices,
                params.distance_type,
            )?]);
        }

        let branching = params.hierarchical_k.max(2);
        let k = branching.min(quota).min(n);
        // Train on a prefix of the node's rows, which is a uniform subsample
        // because the sample is in random order (`train_kmeans` would take the
        // same prefix itself). Gathering only the prefix, and assigning the rest
        // in chunks, keeps memory at the sample plus one chunk instead of a copy
        // of every level of the tree.
        let training_rows = indices.len().min(k * TRAINING_ROWS_PER_CENTROID);
        let training = Self::create_array_from_indices::<T>(
            &indices[..training_rows],
            data_values,
            dimension,
        )?;
        let kmeans = Self::train_kmeans::<T, Algo>(&training, k, &params.derive(salt))?;
        drop(training);
        let membership = Self::membership_of_rows::<T, Algo>(
            kmeans.centroids.as_primitive::<T>().values(),
            data_values,
            dimension,
            &indices,
            params.distance_type,
        );
        let mut children: Vec<(usize, Vec<usize>)> =
            (0..k).map(|centroid| (centroid, Vec::new())).collect();
        for (local, cluster) in membership.iter().enumerate() {
            if let Some(cluster) = cluster {
                children[*cluster as usize].1.push(indices[local]);
            }
        }
        children.retain(|(_, child)| !child.is_empty());

        if children.len() <= 1 {
            // Every vector landed in the same sub-cluster (duplicates): the node
            // cannot be split. It becomes one leaf; the centroids it still owes
            // are filled from other leaves afterwards.
            return Ok(vec![Self::leaf_from_indices::<T, Algo>(
                data_values,
                dimension,
                indices,
                params.distance_type,
            )?]);
        }

        if quota <= k {
            // Bottom of the tree: the sub-clusters are the leaves.
            return children
                .into_iter()
                .map(|(_, child)| {
                    Self::leaf_from_indices::<T, Algo>(
                        data_values,
                        dimension,
                        child,
                        params.distance_type,
                    )
                })
                .collect();
        }

        let sizes: Vec<usize> = children.iter().map(|(_, child)| child.len()).collect();
        let quotas = allocate_quotas(&sizes, quota);
        let (children, quotas) = Self::merge_unfunded_children::<T, Algo>(
            data_values,
            dimension,
            children,
            quotas,
            kmeans.centroids.as_primitive::<T>().values(),
            params.distance_type,
        )?;
        if children.len() < 2 {
            // The merge handed every vector to one sub-cluster (say 999 duplicates
            // and one outlier with a quota of 300), which would be this very node
            // again: recursing could never make progress. Stop here as a leaf;
            // `fill_missing_leaves` tries the remaining splits with a bounded
            // loop and reports the shortfall.
            let indices = children.into_iter().flatten().collect();
            return Ok(vec![Self::leaf_from_indices::<T, Algo>(
                data_values,
                dimension,
                indices,
                params.distance_type,
            )?]);
        }
        debug_assert!(
            children.iter().all(|child| child.len() < n),
            "every sub-cluster must be smaller than its parent for the recursion to end"
        );

        let leaves = children
            .into_par_iter()
            .zip(quotas.into_par_iter())
            .enumerate()
            .map(|(child_id, (child, child_quota))| {
                Self::grow_proportional_subtree::<T, Algo>(
                    data_values,
                    dimension,
                    child,
                    child_quota,
                    params,
                    salt.wrapping_mul(branching as u64 + 1)
                        .wrapping_add(child_id as u64 + 1),
                )
            })
            .collect::<arrow::error::Result<Vec<_>>>()?;
        Ok(leaves.into_iter().flatten().collect())
    }

    /// Hand the vectors of sub-clusters whose quota rounded to zero to the
    /// nearest sibling that did earn centroids, and drop those sub-clusters.
    fn merge_unfunded_children<T: ArrowNumericType, Algo: KMeansAlgo<T::Native>>(
        data_values: &[T::Native],
        dimension: usize,
        children: Vec<(usize, Vec<usize>)>,
        quotas: Vec<usize>,
        centroids: &[T::Native],
        distance_type: DistanceType,
    ) -> arrow::error::Result<(Vec<Vec<usize>>, Vec<usize>)>
    where
        T::Native: Num,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let (mut funded, unfunded): (Vec<_>, Vec<_>) = children
            .into_iter()
            .zip(quotas)
            .partition(|(_, quota)| *quota > 0);
        if !unfunded.is_empty() {
            let funded_centroids: Vec<T::Native> = funded
                .iter()
                .flat_map(|((centroid, _), _)| {
                    centroids[centroid * dimension..(centroid + 1) * dimension]
                        .iter()
                        .copied()
                })
                .collect();
            let orphans: Vec<usize> = unfunded
                .into_iter()
                .flat_map(|((_, child), _)| child)
                .collect();
            let orphan_data =
                Self::create_array_from_indices::<T>(&orphans, data_values, dimension)?;
            let (membership, _) = Algo::compute_membership_and_dist(
                &funded_centroids,
                orphan_data.values().as_primitive::<T>().values(),
                dimension,
                distance_type,
                0.0,
                None,
                None,
            );
            for (orphan, cluster) in orphans.into_iter().zip(membership) {
                if let Some(cluster) = cluster {
                    funded[cluster as usize].0.1.push(orphan);
                }
            }
        }
        Ok(funded
            .into_iter()
            .map(|((_, child), quota)| (child, quota))
            .unzip())
    }

    /// One leaf from the given vectors: its centroid is their mean (or mode).
    fn leaf_from_indices<T: ArrowNumericType, Algo: KMeansAlgo<T::Native>>(
        data_values: &[T::Native],
        dimension: usize,
        indices: Vec<usize>,
        distance_type: DistanceType,
    ) -> arrow::error::Result<Leaf<T::Native>>
    where
        T::Native: Num,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let sub_data = Self::create_array_from_indices::<T>(&indices, data_values, dimension)?;
        let membership = vec![Some(0); indices.len()];
        let mut sizes = vec![indices.len()];
        let kmeans = Algo::to_kmeans(
            sub_data.values().as_primitive::<T>().values(),
            dimension,
            1,
            &membership,
            &mut sizes,
            distance_type,
            0.0,
        );
        Ok(Leaf {
            centroid: kmeans.centroids.as_primitive::<T>().values().to_vec(),
            indices,
        })
    }

    /// Split the largest leaves in two until `target_k` leaves exist, for the
    /// rare tree that came up short because nodes of duplicate vectors could not
    /// be split.
    fn fill_missing_leaves<T: ArrowNumericType, Algo: KMeansAlgo<T::Native>>(
        data_values: &[T::Native],
        dimension: usize,
        leaves: &mut Vec<Leaf<T::Native>>,
        target_k: usize,
        params: &KMeansParams,
    ) -> arrow::error::Result<()>
    where
        T::Native: Num,
        PrimitiveArray<T>: From<Vec<T::Native>>,
    {
        let n = data_values.len() / dimension;
        let mut unsplittable = vec![false; leaves.len()];
        let mut salt = u64::MAX / 2;
        while leaves.len() < target_k {
            let largest = (0..leaves.len())
                .filter(|&i| !unsplittable[i] && leaves[i].indices.len() >= 2)
                .max_by_key(|&i| leaves[i].indices.len());
            let Some(largest) = largest else {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Cannot create {target_k} IVF partitions: k-means could only form {} non-empty \
                     clusters from {n} training vectors. The dataset is likely too small or has too \
                     many (near-)duplicate vectors for this many partitions. Reduce num_partitions to \
                     <= {} or provide more diverse data.",
                    leaves.len(),
                    leaves.len()
                )));
            };
            salt += 1;
            let indices = std::mem::take(&mut leaves[largest].indices);
            let mut halves = Self::grow_proportional_subtree::<T, Algo>(
                data_values,
                dimension,
                indices.clone(),
                2,
                params,
                salt,
            )?;
            if halves.len() < 2 {
                leaves[largest].indices = indices;
                unsplittable[largest] = true;
                continue;
            }
            leaves[largest] = halves.pop().unwrap();
            leaves.push(halves.pop().unwrap());
            unsplittable.push(false);
        }
        Ok(())
    }

    /// Train a [`KMeans`] model with full parameters.
    ///
    /// If the DistanceType is `Cosine`, the input vectors will be normalized with each iteration.
    pub fn new_with_params(
        data: &FixedSizeListArray,
        k: usize,
        params: &KMeansParams,
    ) -> arrow::error::Result<Self> {
        let n = data.len();
        if n < k {
            return Err(ArrowError::InvalidArgumentError(format!(
                "KMeans: training does not have sufficient data points: n({}) is smaller than k({})",
                n, k
            )));
        }

        // use hierarchical clustering if k > 256 and hierarchical_k > 1
        // we set 256 as the threshold because:
        // 1. PQ would run kmeans with k=256, in that case we don't want to use hierarchical clustering for accuracy
        // 2. kmeans with k=256 is small enough that we don't need to use hierarchical clustering for efficiency
        if k > 256 && params.hierarchical_k > 1 {
            log::debug!("Using hierarchical clustering for k={}", k);
            return match (data.value_type(), params.distance_type) {
                (DataType::Float16, _) => Self::train_hierarchical_kmeans::<
                    Float16Type,
                    KMeansAlgoFloat<Float16Type>,
                >(data, k, params),
                (DataType::Float32, _) => Self::train_hierarchical_kmeans::<
                    Float32Type,
                    KMeansAlgoFloat<Float32Type>,
                >(data, k, params),
                (DataType::Float64, _) => Self::train_hierarchical_kmeans::<
                    Float64Type,
                    KMeansAlgoFloat<Float64Type>,
                >(data, k, params),
                (DataType::UInt8, DistanceType::Hamming) => {
                    Self::train_hierarchical_kmeans::<UInt8Type, KModeAlgo>(data, k, params)
                }
                _ => Err(ArrowError::InvalidArgumentError(format!(
                    "KMeans: can not train data type {} with distance type: {}",
                    data.value_type(),
                    params.distance_type
                ))),
            };
        }

        match (data.value_type(), params.distance_type) {
            (DataType::Float16, _) => {
                Self::train_kmeans::<Float16Type, KMeansAlgoFloat<Float16Type>>(data, k, params)
            }

            (DataType::Float32, _) => {
                Self::train_kmeans::<Float32Type, KMeansAlgoFloat<Float32Type>>(data, k, params)
            }
            (DataType::Float64, _) => {
                Self::train_kmeans::<Float64Type, KMeansAlgoFloat<Float64Type>>(data, k, params)
            }
            (DataType::UInt8, DistanceType::Hamming) => {
                Self::train_kmeans::<UInt8Type, KModeAlgo>(data, k, params)
            }
            _ => Err(ArrowError::InvalidArgumentError(format!(
                "KMeans: can not train data type {} with distance type: {}",
                data.value_type(),
                params.distance_type
            ))),
        }
    }
}

pub fn kmeans_find_partitions_arrow_array(
    centroids: &FixedSizeListArray,
    query: &dyn Array,
    nprobes: usize,
    distance_type: DistanceType,
) -> arrow::error::Result<(UInt32Array, Float32Array)> {
    if centroids.value_length() as usize != query.len() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "Centroids and vectors have different dimensions: {} != {}",
            centroids.value_length(),
            query.len()
        )));
    }

    match (centroids.value_type(), query.data_type()) {
        (DataType::Float16, DataType::Float16) => {
            let centroids = centroids.values().as_primitive::<Float16Type>().values();
            let query = query.as_primitive::<Float16Type>().values();
            if distance_type == DistanceType::Dot
                && amx_fp16_available()
                && let Some(dists) = dot_f16_partitions_amx(centroids, query)
            {
                return smallest_nprobes(dists, nprobes);
            }
            Ok(kmeans_find_partitions(
                centroids,
                query,
                nprobes,
                distance_type,
            )?)
        }
        (DataType::Float32, DataType::Float32) => Ok(kmeans_find_partitions(
            centroids.values().as_primitive::<Float32Type>().values(),
            query.as_primitive::<Float32Type>().values(),
            nprobes,
            distance_type,
        )?),
        (DataType::Float64, DataType::Float64) => Ok(kmeans_find_partitions(
            centroids.values().as_primitive::<Float64Type>().values(),
            query.as_primitive::<Float64Type>().values(),
            nprobes,
            distance_type,
        )?),
        (DataType::UInt8, DataType::UInt8) => Ok(kmeans_find_partitions_binary(
            centroids.values().as_primitive::<UInt8Type>().values(),
            query.as_primitive::<UInt8Type>().values(),
            nprobes,
            distance_type,
        )?),
        _ => Err(ArrowError::InvalidArgumentError(format!(
            "Centroids and vectors have different types: {} != {}",
            centroids.value_type(),
            query.data_type()
        ))),
    }
}

/// KMeans finds N nearest partitions.
///
/// Parameters:
/// The `nprobes` smallest distances and the partitions they belong to.
fn smallest_nprobes(
    dists: Vec<f32>,
    nprobes: usize,
) -> arrow::error::Result<(UInt32Array, Float32Array)> {
    // TODO: use heap to just keep nprobes smallest values.
    let dists_arr = Float32Array::from(dists);
    let indices = sort_to_indices(&dists_arr, None, Some(nprobes))?;
    let dists = arrow::compute::take(&dists_arr, &indices, None)?
        .as_primitive::<Float32Type>()
        .clone();
    Ok((indices, dists))
}

/// `Dot` distances from `query` to every centroid, through the AMX-FP16 kernel,
/// or `None` when this build/CPU/shape cannot use it.
///
/// Partition selection is one query against every centroid, so on paper it needs
/// well under 1% of this machine's arithmetic. It measured at 33% of a saturated
/// IVF_HNSW_SQ query because `dot_f16_avx512` carries no vector instruction at
/// all under GCC 13.4 -- disassembly shows 30 `vcvtsh2ss` / 15 `vmulss` /
/// 14 `vaddss` and zero `zmm` operands, since GCC has no packed `_Float16` ->
/// `float` widening pattern. The scalar loop, not the work, is the cost.
///
/// Sixteen centroids at a time rather than the `M x N` GEMM: the GEMM steps its
/// centroid loop by 32 and would spend 31 of every 32 output columns on padding
/// for a single query (16 MAC/cycle), while this shape wastes 15 of 16 and
/// reaches 32 MAC/cycle. Those rates count tile work only; each call also pays
/// one LDTILECFG plus one TILERELEASE, which at these shapes is the larger term.
/// Beating either needs several queries scored together, which the per-query
/// search API does not offer.
fn dot_f16_partitions_amx(centroids: &[f16], query: &[f16]) -> Option<Vec<f32>> {
    let dim = query.len();
    // Below one full 32-wide k-pass the kernel is all scalar cleanup, so a dim
    // that short would run at a loss. Support, not the `LANCE_DISABLE_AMX` kill
    // switch: the caller has already decided to use AMX, and this only declines
    // shapes the kernel cannot pay for.
    if dim < 32 || !amx_fp16_supported() {
        return None;
    }
    debug_assert_eq!(centroids.len() % dim, 0);

    let mut dists = vec![0f32; centroids.len() / dim];
    let row = |i: usize| &centroids[i * dim..(i + 1) * dim];
    for (g, out) in dists.chunks_mut(16).enumerate() {
        let base = g * 16;
        // `dot_f16_batch_16` requires 16 slices of the query's length even when
        // only `len` of them are scored, so the tail repeats a valid row; those
        // lanes are computed and discarded.
        let mut group: [&[f16]; 16] = [row(base); 16];
        for (i, slot) in group.iter_mut().enumerate().take(out.len()) {
            *slot = row(base + i);
        }
        // The kernel returns raw dot products; `Dot` distance is `1 - dot`, the
        // same convention `dot_distance_batch` applies.
        let dots = dot_f16_batch_16(query, &group, out.len());
        for (d, dot) in out.iter_mut().zip(dots.iter()) {
            *d = 1.0 - *dot;
        }
    }
    Some(dists)
}

/// - *centroids*: a `k * dimension` floating array.
/// - *query*: a `dimension` floating array.
/// - *nprobes*: the number of partitions to find.
/// - *distance_type*: the distance type to calculate distance.
///
/// This function allows to conduct kmeans search without constructing
/// `Arrow Array` or `Vec<Float>` types.
///
pub fn kmeans_find_partitions<T: Float + L2 + Dot>(
    centroids: &[T],
    query: &[T],
    nprobes: usize,
    distance_type: DistanceType,
) -> arrow::error::Result<(UInt32Array, Float32Array)> {
    let dists: Vec<f32> = match distance_type {
        DistanceType::L2 => l2_distance_batch(query, centroids, query.len()).collect(),
        DistanceType::Dot => dot_distance_batch(query, centroids, query.len()).collect(),
        _ => {
            panic!(
                "KMeans::find_partitions: {} is not supported",
                distance_type
            );
        }
    };

    smallest_nprobes(dists, nprobes)
}

pub fn kmeans_find_partitions_binary(
    centroids: &[u8],
    query: &[u8],
    nprobes: usize,
    distance_type: DistanceType,
) -> arrow::error::Result<(UInt32Array, Float32Array)> {
    let dists: Vec<f32> = match distance_type {
        DistanceType::Hamming => hamming_distance_batch(query, centroids, query.len()).collect(),
        _ => {
            panic!(
                "KMeans::find_partitions: {} is not supported",
                distance_type
            );
        }
    };

    // TODO: use heap to just keep nprobes smallest values.
    let dists_arr = Float32Array::from(dists);
    let indices = sort_to_indices(&dists_arr, None, Some(nprobes))?;
    let dists = arrow::compute::take(&dists_arr, &indices, None)?
        .as_primitive::<Float32Type>()
        .clone();
    Ok((indices, dists))
}

/// Compute partitions from Arrow FixedSizeListArray.
#[allow(clippy::type_complexity)]
pub fn compute_partitions_arrow_array(
    centroids: &FixedSizeListArray,
    vectors: &FixedSizeListArray,
    distance_type: DistanceType,
) -> arrow::error::Result<(Vec<Option<u32>>, Vec<Option<f32>>)> {
    if centroids.value_length() != vectors.value_length() {
        return Err(ArrowError::InvalidArgumentError(
            "Centroids and vectors have different dimensions".to_string(),
        ));
    }
    match (centroids.value_type(), vectors.value_type()) {
        (DataType::Float16, DataType::Float16) => Ok(compute_partitions_with_dists::<
            Float16Type,
            KMeansAlgoFloat<Float16Type>,
        >(
            centroids.values().as_primitive(),
            vectors.values().as_primitive(),
            centroids.value_length(),
            distance_type,
        )),
        (DataType::Float32, DataType::Float32) => Ok(compute_partitions_with_dists::<
            Float32Type,
            KMeansAlgoFloat<Float32Type>,
        >(
            centroids.values().as_primitive(),
            vectors.values().as_primitive(),
            centroids.value_length(),
            distance_type,
        )),
        (DataType::Float32, DataType::Int8) => Ok(compute_partitions_with_dists::<
            Float32Type,
            KMeansAlgoFloat<Float32Type>,
        >(
            centroids.values().as_primitive(),
            vectors.convert_to_floating_point()?.values().as_primitive(),
            centroids.value_length(),
            distance_type,
        )),
        (DataType::Float64, DataType::Float64) => Ok(compute_partitions_with_dists::<
            Float64Type,
            KMeansAlgoFloat<Float64Type>,
        >(
            centroids.values().as_primitive(),
            vectors.values().as_primitive(),
            centroids.value_length(),
            distance_type,
        )),
        (DataType::UInt8, DataType::UInt8) => {
            Ok(compute_partitions_with_dists::<UInt8Type, KModeAlgo>(
                centroids.values().as_primitive(),
                vectors.values().as_primitive(),
                centroids.value_length(),
                distance_type,
            ))
        }
        _ => Err(ArrowError::InvalidArgumentError(
            "Centroids and vectors have incompatible types".to_string(),
        )),
    }
}

/// Compute partition ID of each vector in the KMeans.
///
/// If returns `None`, means the vector is not valid, i.e., all `NaN`.
pub fn compute_partitions<T: ArrowNumericType, K: KMeansAlgo<T::Native>>(
    centroids: &PrimitiveArray<T>,
    vectors: &PrimitiveArray<T>,
    dimension: impl AsPrimitive<usize>,
    distance_type: DistanceType,
) -> (Vec<Option<u32>>, f64)
where
    T::Native: Num,
{
    let dimension = dimension.as_();
    let (membership, _, losses) = K::compute_membership_and_loss(
        centroids.values(),
        vectors.values(),
        dimension,
        distance_type,
        0.0,
        None,
        None,
    );
    (membership, losses.iter().sum::<f64>())
}

/// compute the partition id and the distance to the centroid for each vector,
/// NOTE the distance is squared distance for L2
pub fn compute_partitions_with_dists<T: ArrowNumericType, K: KMeansAlgo<T::Native>>(
    centroids: &PrimitiveArray<T>,
    vectors: &PrimitiveArray<T>,
    dimension: impl AsPrimitive<usize>,
    distance_type: DistanceType,
) -> (Vec<Option<u32>>, Vec<Option<f32>>)
where
    T::Native: Num,
{
    let dimension = dimension.as_();
    K::compute_membership_and_dist(
        centroids.values(),
        vectors.values(),
        dimension,
        distance_type,
        0.0,
        None,
        None,
    )
}

/// Train KMeans model and returns the centroids of each cluster.
///
/// Parameters
/// ----------
/// - *centroids*: initial centroids, use the random initialization if None
/// - *array*: a flatten floating number array of vectors
/// - *dimension*: dimension of the vector
/// - *k*: number of clusters
/// - *max_iterations*: maximum number of iterations
/// - *redos*: number of times to redo the k-means clustering
/// - *distance_type*: distance type to compute pair-wise vector distance
/// - *sample_rate*: sample rate to select the data for training
#[allow(clippy::too_many_arguments)]
pub fn train_kmeans<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    mut params: KMeansParams,
    dimension: usize,
    k: usize,
    sample_rate: usize,
) -> Result<KMeans>
where
    T::Native: Dot + L2 + Normalize,
    PrimitiveArray<T>: From<Vec<T::Native>>,
{
    let num_rows = array.len() / dimension;
    if num_rows < k {
        return Err(Error::unprocessable(format!(
            "KMeans cannot train {k} centroids with {num_rows} vectors; choose a smaller K (< {num_rows})"
        )));
    }

    // Only sample sample_rate * num_clusters. See Faiss
    let data = if num_rows > sample_rate * k {
        log::info!(
            "Sample {} out of {} to train kmeans of {} dim, {} clusters",
            sample_rate * k,
            array.len() / dimension,
            dimension,
            k,
        );
        let sample_size = sample_rate * k;
        array.slice(0, sample_size * dimension)
    } else {
        array.clone()
    };

    let data = FixedSizeListArray::try_new_from_values(data, dimension as i32)?;

    params.balance_factor /= data.len() as f32;
    let model = KMeans::new_with_params(&data, k, &params)?;
    Ok(model)
}

#[inline]
pub fn compute_partition<T: Float + L2 + Dot>(
    centroids: &[T],
    vector: &[T],
    distance_type: DistanceType,
) -> Option<u32> {
    match distance_type {
        DistanceType::L2 => {
            argmin_value_float(l2_distance_batch(vector, centroids, vector.len())).map(|(c, _)| c)
        }
        DistanceType::Dot => {
            argmin_value_float(dot_distance_batch(vector, centroids, vector.len())).map(|(c, _)| c)
        }
        _ => {
            panic!(
                "KMeans::compute_partition: distance type {} is not supported",
                distance_type
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::iter::repeat_n;

    use arrow_array::Float16Array;
    use arrow_array::types::Float16Type;
    use half::f16;
    use lance_arrow::*;
    use lance_testing::datagen::generate_random_array;

    use super::*;
    use lance_linalg::distance::dot_f16::amx_fp16_supported;
    use lance_linalg::distance::l2;
    use lance_linalg::kernels::argmin;

    /// The AMX partition path must pick the same partitions as the scalar one.
    /// Exact equality on the distances is not required -- the kernel accumulates
    /// in a different order -- but the chosen partition ids must match, since a
    /// different choice silently changes which vectors a query can ever see.
    #[test]
    fn test_amx_find_partitions_matches_scalar() {
        if !amx_fp16_supported() {
            return;
        }
        // (dim, k): a production shape, one with a partial 16-group tail, and one
        // whose dimension is not a multiple of the kernel's 32-wide k-pass.
        for (dim, k) in [(768usize, 10_000usize), (768, 37), (133, 100)] {
            let mut st = 0x9E37u64;
            let mut next = || {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                f16::from_f32(((st >> 33) as f32 / (1u64 << 31) as f32) - 0.5)
            };
            let centroids: Vec<f16> = (0..k * dim).map(|_| next()).collect();
            let query: Vec<f16> = (0..dim).map(|_| next()).collect();

            let amx = dot_f16_partitions_amx(&centroids, &query)
                .expect("the AMX path declined a shape it should accept");
            let scalar: Vec<f32> = dot_distance_batch(&query[..], &centroids[..], dim).collect();
            assert_eq!(amx.len(), scalar.len(), "dim={dim} k={k}");

            for nprobes in [1usize, 8, 32] {
                let (amx_idx, _) = smallest_nprobes(amx.clone(), nprobes).unwrap();
                let (scalar_idx, _) = smallest_nprobes(scalar.clone(), nprobes).unwrap();
                assert_eq!(
                    amx_idx.values(),
                    scalar_idx.values(),
                    "dim={dim} k={k} nprobes={nprobes} picked different partitions"
                );
            }
        }
    }

    #[test]
    fn test_train_with_small_dataset() {
        let data = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
        let data = FixedSizeListArray::try_new_from_values(data, 2).unwrap();
        match KMeans::new(&data, 128, 5) {
            Ok(_) => panic!("Should fail to train KMeans"),
            Err(e) => {
                assert!(e.to_string().contains("smaller than"));
            }
        }
    }

    #[test]
    fn test_compute_partitions() {
        const DIM: usize = 256;
        let centroids = generate_random_array(DIM * 18);
        let data = generate_random_array(DIM * 20);

        let expected = data
            .values()
            .chunks(DIM)
            .map(|row| {
                argmin(
                    centroids
                        .values()
                        .chunks(DIM)
                        .map(|centroid| l2(row, centroid)),
                )
            })
            .collect::<Vec<_>>();
        let (actual, _) = compute_partitions::<Float32Type, KMeansAlgoFloat<Float32Type>>(
            &centroids,
            &data,
            DIM,
            DistanceType::L2,
        );
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_random_init_advances_rng() {
        let values = Float32Array::from_iter_values((0..64).map(|value| value as f32));
        let mut rng = SmallRng::seed_from_u64(42);
        let first =
            KMeans::init_random::<Float32Type>(values.values(), 1, 8, &mut rng, DistanceType::L2);
        let second =
            KMeans::init_random::<Float32Type>(values.values(), 1, 8, &mut rng, DistanceType::L2);

        assert_ne!(
            first.centroids.as_primitive::<Float32Type>().values(),
            second.centroids.as_primitive::<Float32Type>().values(),
        );
    }

    #[tokio::test]
    async fn test_compute_membership_and_loss() {
        const DIM: usize = 256;
        let centroids = generate_random_array(DIM * 18);
        let data = generate_random_array(DIM * 20);

        let (membership, _, losses) = KMeansAlgoFloat::<Float32Type>::compute_membership_and_loss(
            centroids.as_slice(),
            data.values(),
            DIM,
            DistanceType::L2,
            0.0,
            None,
            None,
        );
        let loss = losses.iter().sum::<f64>();
        assert!(loss > 0.0, "loss is not zero: {}", loss);
        membership.iter().for_each(|cd| {
            assert!(cd.is_some());
        });
    }

    #[tokio::test]
    async fn test_l2_with_nans() {
        const DIM: usize = 8;
        const K: usize = 32;
        const NUM_CENTROIDS: usize = 16 * 2048;
        let centroids = generate_random_array(DIM * NUM_CENTROIDS);
        let values = Float32Array::from_iter_values(repeat_n(f32::NAN, DIM * K));

        compute_partitions::<Float32Type, KMeansAlgoFloat<Float32Type>>(
            &centroids,
            &values,
            DIM,
            DistanceType::L2,
        )
        .0
        .iter()
        .for_each(|cd| {
            assert!(cd.is_none());
        });
    }

    #[tokio::test]
    async fn test_train_l2_kmeans_with_nans() {
        const DIM: usize = 8;
        const K: usize = 32;
        const NUM_CENTROIDS: usize = 16 * 2048;
        let centroids = generate_random_array(DIM * NUM_CENTROIDS);
        let values = repeat_n(f32::NAN, DIM * K).collect::<Vec<_>>();

        let (membership, _, _) = KMeansAlgoFloat::<Float32Type>::compute_membership_and_loss(
            centroids.as_slice(),
            &values,
            DIM,
            DistanceType::L2,
            0.0,
            None,
            None,
        );

        membership.iter().for_each(|cd| assert!(cd.is_none()));
    }

    #[tokio::test]
    async fn test_train_kmode() {
        const DIM: usize = 16;
        const K: usize = 32;
        const NUM_VALUES: usize = 256 * K;

        let mut rng = SmallRng::from_os_rng();
        let values =
            UInt8Array::from_iter_values((0..NUM_VALUES * DIM).map(|_| rng.random_range(0..255)));

        let fsl = FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap();

        let params = KMeansParams {
            distance_type: DistanceType::Hamming,
            ..Default::default()
        };
        let kmeans = KMeans::new_with_params(&fsl, K, &params).unwrap();
        assert_eq!(kmeans.centroids.len(), K * DIM);
        assert_eq!(kmeans.dimension, DIM);
        assert_eq!(kmeans.centroids.data_type(), &DataType::UInt8);
    }

    #[tokio::test]
    async fn test_hierarchical_kmeans() {
        const DIM: usize = 64;
        const K: usize = 257; // Greater than 256 to trigger hierarchical clustering
        const NUM_VALUES: usize = 1024 * K;

        let values = generate_random_array(NUM_VALUES * DIM);
        let fsl = FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap();

        let params = KMeansParams {
            max_iters: 10,
            hierarchical_k: 16,
            ..Default::default()
        };

        let kmeans = KMeans::new_with_params(&fsl, K, &params).unwrap();

        // Verify that we have the correct number of clusters
        assert_eq!(kmeans.centroids.len(), K * DIM);
        assert_eq!(kmeans.dimension, DIM);
        assert_eq!(kmeans.centroids.data_type(), &DataType::Float32);

        // Verify that all centroids are valid (not NaN)
        let centroids = kmeans.centroids.as_primitive::<Float32Type>().values();
        for val in centroids {
            assert!(!val.is_nan(), "Centroid should not contain NaN values");
        }
    }

    #[tokio::test]
    async fn test_hierarchical_kmeans_too_few_distinct_vectors_errors() {
        // Regression test for https://github.com/lance-format/lance/issues/7867
        //
        // With a small number of distinct vectors repeated many times (heavy
        // near-duplication) and dot distance, hierarchical k-means cannot form
        // `target_k` non-empty clusters no matter how it splits: every split of a
        // cluster of identical vectors is either ineffective (`all_same`) or
        // immediately hits the "<= 1 point" floor. This used to trip
        // `debug_assert_eq!(heap.len(), target_k)` (panic in debug builds) or
        // silently return a half-empty centroid set (release builds). It should
        // now return a clear error instead.
        const DIM: usize = 8;
        const NUM_DISTINCT: usize = 5;
        const REPEATS: usize = 200;
        const TARGET_K: usize = 300; // > 256 to trigger hierarchical clustering

        let base_vectors = generate_random_array(NUM_DISTINCT * DIM);
        let mut values = Vec::with_capacity(NUM_DISTINCT * REPEATS * DIM);
        for _ in 0..REPEATS {
            values.extend_from_slice(base_vectors.values());
        }
        let values = Float32Array::from(values);
        let fsl = FixedSizeListArray::try_new_from_values(values, DIM as i32).unwrap();

        let params = KMeansParams {
            max_iters: 10,
            hierarchical_k: 16,
            distance_type: DistanceType::Dot,
            ..Default::default()
        };

        let err = KMeans::new_with_params(&fsl, TARGET_K, &params)
            .expect_err("training should fail rather than panic or silently under-produce");
        let msg = err.to_string();
        assert!(
            msg.contains("Cannot create") && msg.contains(&TARGET_K.to_string()),
            "unexpected error message: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // AMX-FP16 dot-distance assignment
    // -----------------------------------------------------------------------

    /// Relative tolerance between the AMX and per-vector distances. Both
    /// accumulate f32-widened products and differ only in summation order, so
    /// this is far looser than what they actually differ by (~1e-4) and far
    /// tighter than fp16's own representational error.
    const AMX_REL_TOL: f32 = 5e-3;

    /// A vector this much nearer its best centroid than its runner-up cannot
    /// change hands on summation order alone. Closer ties are allowed to
    /// disagree — that is fp16 arithmetic, not a bug.
    const AMX_TIE_GAP: f32 = 1e-2;

    fn random_f16(count: usize, rng: &mut SmallRng) -> Vec<f16> {
        (0..count)
            .map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0)))
            .collect()
    }

    /// Assert the AMX dot path engages for this input and assigns every vector
    /// where the per-vector path does.
    fn assert_dot_paths_agree(
        centroids: &[f16],
        data: &[f16],
        dimension: usize,
        balance_factor: f32,
        cluster_sizes: Option<&[usize]>,
        ctx: &str,
    ) {
        let k = centroids.len() / dimension;
        // The AMX path's own output, not `compute_membership_and_dist`'s: that
        // entry point falls back to the per-vector path whenever this one
        // declines, so going through it would silently degrade this into a
        // scalar-against-scalar comparison on any host or build that lacks the
        // kernel, and prove nothing about it.
        let amx = dot_membership_amx_f16(centroids, data, dimension, balance_factor, cluster_sizes)
            .unwrap_or_else(|| {
                panic!("{ctx}: the AMX path declined this shape, so agreeing proves nothing")
            });

        for (i, vector) in data.chunks(dimension).enumerate() {
            let row = dot_distance_batch(vector, centroids, dimension).collect::<Vec<_>>();
            let want = argmin_value_float_with_bias(
                row.iter().copied(),
                cluster_sizes.map(|sizes| sizes.iter().map(|size| balance_factor * *size as f32)),
            );
            let got = amx[i];
            let (Some((want_id, _)), Some((got_id, got_dist))) = (want, got) else {
                assert_eq!(
                    want.is_none(),
                    got.is_none(),
                    "{ctx}: row {i} is assigned by one path only: {want:?} vs {got:?}"
                );
                continue;
            };

            assert!(
                (got_id as usize) < k,
                "{ctx}: row {i} landed on centroid {got_id}, outside the {k} real ones"
            );
            // Check the reported distance against the reported centroid's own
            // rather than against the winner's: on a near-tie the paths may
            // pick different centroids, and then only this identity has to hold.
            let want_dist = row[got_id as usize];
            assert!(
                (got_dist - want_dist).abs() <= AMX_REL_TOL * want_dist.abs() + 1e-3,
                "{ctx}: row {i} centroid {got_id} distance {got_dist}, want {want_dist}"
            );

            let mut biased = row
                .iter()
                .enumerate()
                .map(|(j, dist)| {
                    dist + cluster_sizes.map_or(0.0, |sizes| balance_factor * sizes[j] as f32)
                })
                .collect::<Vec<_>>();
            biased.sort_by(f32::total_cmp);
            if biased[1] - biased[0] > AMX_TIE_GAP {
                assert_eq!(
                    got_id, want_id,
                    "{ctx}: row {i} is not a tie ({} vs {}) but the paths disagree",
                    biased[0], biased[1]
                );
            }
        }
    }

    /// The two paths across the shapes that exercise each boundary: `k` on and
    /// off the kernel's 32-centroid block (so with and without zero padding,
    /// including the half-filled 16 the hierarchical splitter trains at), `dim`
    /// with and without the kernel's scalar tail, and row counts on and off the
    /// 32-row tile pass (so with and without trailing fallback rows).
    #[test]
    fn test_dot_amx_matches_per_vector_path() {
        if !amx_fp16_supported() {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(0xD07);
        for k in [16usize, 32, 64, 100] {
            for dimension in [32usize, 64, 768] {
                for n in [64usize, 100, 1000] {
                    let centroids = random_f16(k * dimension, &mut rng);
                    let data = random_f16(n * dimension, &mut rng);
                    assert_dot_paths_agree(
                        &centroids,
                        &data,
                        dimension,
                        0.0,
                        None,
                        &format!("k={k} dim={dimension} n={n}"),
                    );
                }
            }
        }
    }

    /// The padding columns must be unreachable by the argmin.
    ///
    /// `k` is not a multiple of 32, so the GEMM's `n` block is filled out with
    /// zero centroids, which score a dot product of 0 — distance exactly 1.0.
    /// Here every real dot product is negative, so every real distance exceeds
    /// 1.0 and a reduction over the padded row width would hand *every* vector
    /// a cluster id past the end of the centroid set.
    ///
    /// `k = 16` is the worst case the training path actually runs at: half the
    /// block is padding, so half of every scored row is a column that must
    /// never win.
    #[test]
    fn test_dot_amx_padding_columns_never_win() {
        if !amx_fp16_supported() {
            return;
        }
        const DIM: usize = 64;
        const N: usize = 128;

        let mut rng = SmallRng::seed_from_u64(0xBAD5);
        for k in [100usize, 16] {
            let negate = |v: &f16| f16::from_f32(-v.to_f32().abs() - 0.1);
            let centroids = random_f16(k * DIM, &mut rng)
                .iter()
                .map(negate)
                .collect::<Vec<_>>();
            let data = random_f16(N * DIM, &mut rng)
                .iter()
                .map(|v| f16::from_f32(v.to_f32().abs() + 0.1))
                .collect::<Vec<_>>();

            for vector in data.chunks(DIM) {
                assert!(
                    dot_distance_batch(vector, &centroids, DIM).all(|dist| dist > 1.0),
                    "premise broken: a real centroid is nearer than the zero padding"
                );
            }
            assert_dot_paths_agree(&centroids, &data, DIM, 0.0, None, &format!("padding k={k}"));
        }
    }

    /// Shapes the kernel is not worth entering for must decline before touching
    /// it, so this needs no AMX host to run.
    ///
    /// The `k` bound is 16 -- half the kernel's 32-centroid block -- and
    /// `prefers_flat_amx_assignment` in `utils.rs` mirrors it; the two have to
    /// move together or a build could take the exact-assignment route with no
    /// GEMM under it.
    #[test]
    fn test_dot_amx_declines_shapes_below_half_a_block() {
        let mut rng = SmallRng::seed_from_u64(0x5A11);
        for (k, dimension) in [(15usize, 64usize), (64, 31), (15, 31)] {
            let centroids = random_f16(k * dimension, &mut rng);
            let data = random_f16(64 * dimension, &mut rng);
            assert!(
                dot_membership_amx_f16(&centroids, &data, dimension, 0.0, None).is_none(),
                "k={k} dim={dimension}"
            );
        }
    }

    /// The bias path. `argmin_value_float_with_bias` minimizes `distance +
    /// bias` but reports the unbiased distance, so both halves of that have to
    /// survive the AMX path; the balance factor is sized to actually move
    /// assignments, which the test asserts rather than assumes.
    #[test]
    fn test_dot_amx_with_balance_bias() {
        if !amx_fp16_supported() {
            return;
        }
        const K: usize = 64;
        const DIM: usize = 128;
        const N: usize = 256;
        const BALANCE_FACTOR: f32 = 0.02;

        let mut rng = SmallRng::seed_from_u64(0xB1A5);
        let centroids = random_f16(K * DIM, &mut rng);
        let data = random_f16(N * DIM, &mut rng);
        let cluster_sizes = (0..K).map(|id| id * 4).collect::<Vec<_>>();

        assert_dot_paths_agree(
            &centroids,
            &data,
            DIM,
            BALANCE_FACTOR,
            Some(&cluster_sizes),
            "bias",
        );

        let assign = |balance_factor, sizes| {
            KMeansAlgoFloat::<Float16Type>::compute_membership_and_dist(
                &centroids,
                &data,
                DIM,
                DistanceType::Dot,
                balance_factor,
                sizes,
                None,
            )
            .0
        };
        assert_ne!(
            assign(BALANCE_FACTOR, Some(cluster_sizes.as_slice())),
            assign(0.0, None),
            "the balance factor is too small to move any assignment"
        );
    }

    /// A row of NaNs has no nearest centroid — `distance + bias < min` is false
    /// for every centroid — and the AMX path has to reach the same `None` as
    /// the per-vector one instead of defaulting to cluster 0. Covered in both
    /// the tiled rows and the trailing rows that fall back per vector.
    #[test]
    fn test_dot_amx_all_nan_row_is_unassigned() {
        if !amx_fp16_supported() {
            return;
        }
        const K: usize = 64;
        const DIM: usize = 64;
        const N: usize = 100; // 3 full tile passes, then 4 fallback rows
        const NAN_ROWS: [usize; 2] = [7, 98];

        let mut rng = SmallRng::seed_from_u64(0x4A4);
        let centroids = random_f16(K * DIM, &mut rng);
        let mut data = random_f16(N * DIM, &mut rng);
        for row in NAN_ROWS {
            data[row * DIM..(row + 1) * DIM].fill(f16::NAN);
        }

        assert_dot_paths_agree(&centroids, &data, DIM, 0.0, None, "nan");

        let (membership, _) = KMeansAlgoFloat::<Float16Type>::compute_membership_and_dist(
            &centroids,
            &data,
            DIM,
            DistanceType::Dot,
            0.0,
            None,
            None,
        );
        for (row, cluster_id) in membership.iter().enumerate() {
            assert_eq!(
                cluster_id.is_none(),
                NAN_ROWS.contains(&row),
                "row {row} membership {cluster_id:?}"
            );
        }
    }

    /// Wall-clock throughput of the dot-distance assignment the AMX path above
    /// accelerates, swept over `(threads, dim, k)`.
    ///
    /// The path is picked inside `compute_membership_and_dist` from run-time
    /// capability and the data's shape, so there is nothing to toggle per
    /// iteration: run this same binary twice — once as-is for the AMX path, once
    /// with `LANCE_DISABLE_AMX=1` for the per-vector path — and divide. The
    /// header line reports which path the process took, so the two outputs
    /// cannot be confused.
    ///
    /// Each point runs for a wall-clock budget rather than a fixed pass count, so
    /// a 1-thread point and an all-core point take comparable time and every
    /// point averages over enough work to be stable.
    ///
    /// `#[ignore]` -- run:
    ///   cargo test -p lance-index --release \
    ///     kmeans_dot_f16_membership_bench -- --ignored --nocapture
    /// Tune with `BENCH_N`, `BENCH_DIMS` / `BENCH_KS` / `BENCH_THREADS`
    /// (comma-separated; threads default `<ncpu>,32,1`) and `BENCH_SECONDS` (the
    /// wall-clock budget each measured point gets).
    #[test]
    #[ignore]
    #[allow(clippy::print_stderr)]
    fn kmeans_dot_f16_membership_bench() {
        use std::time::{Duration, Instant};

        let env_usize = |key: &str, default: usize| -> usize {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(default)
        };
        let env_list = |key: &str, default: &[usize]| -> Vec<usize> {
            std::env::var(key)
                .ok()
                .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
                .unwrap_or_else(|| default.to_vec())
        };

        let n = env_usize("BENCH_N", 65_536);
        let dims = env_list("BENCH_DIMS", &[128, 768, 1536]);
        let ks = env_list("BENCH_KS", &[32, 64, 128, 256, 1024, 4096]);
        let ncpu = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let thread_counts = env_list("BENCH_THREADS", &[ncpu, 32, 1]);
        let budget = Duration::from_secs_f64(
            std::env::var("BENCH_SECONDS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3.0),
        );

        eprintln!(
            "[kmeans_dot_f16_bench] n={n} ncpu={ncpu} budget={:.1}s amx_fp16_available={}",
            budget.as_secs_f64(),
            amx_fp16_available(),
        );

        let mut rng = SmallRng::seed_from_u64(0x9E37);
        for &dimension in &dims {
            // Random data *and* random centroids: with degenerate inputs every
            // vector would reduce to the same centroid and the argmin's branches
            // and the score buffer's access pattern would both be unrealistic.
            let data = random_f16(n * dimension, &mut rng);
            for &k in &ks {
                if k >= n {
                    eprintln!(
                        "[kmeans_dot_f16_bench]   dim={dimension} k={k}: skipped, k must be < n={n}"
                    );
                    continue;
                }
                let centroids = random_f16(k * dimension, &mut rng);
                for &nthreads in &thread_counts {
                    if nthreads == 0 || nthreads > ncpu {
                        eprintln!(
                            "[kmeans_dot_f16_bench]   dim={dimension} k={k} threads={nthreads}: skipped, not in 1..={ncpu}"
                        );
                        continue;
                    }
                    // A private pool so the sweep sets the width exactly, without
                    // reconfiguring (or being limited by) the global one.
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(nthreads)
                        .build()
                        .unwrap();
                    let run_pass = || {
                        pool.install(|| {
                            KMeansAlgoFloat::<Float16Type>::compute_membership_and_dist(
                                &centroids,
                                &data,
                                dimension,
                                DistanceType::Dot,
                                0.0,
                                None,
                                None,
                            )
                        })
                    };
                    let warm = run_pass(); // page-in and thread spin-up, untimed
                    std::hint::black_box(&warm);
                    drop(warm);

                    let t0 = Instant::now();
                    let mut passes = 0usize;
                    while t0.elapsed() < budget {
                        let assigned = run_pass();
                        std::hint::black_box(&assigned);
                        passes += 1;
                    }
                    let elapsed = t0.elapsed().as_secs_f64();
                    let vectors = passes * n;
                    let vec_per_s = vectors as f64 / elapsed;
                    eprintln!(
                        "[kmeans_dot_f16_bench]   dim={dimension:>5} k={k:>5} threads={nthreads:>4} passes={passes:>6} vec_per_s={vec_per_s:>12.0} us_per_vec={:>9.4} Gpair_per_s={:>8.2}",
                        1e6 / vec_per_s,
                        vec_per_s * k as f64 / 1e9,
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_float16_underflow_fix() {
        // This test verifies the fix for float16 division underflow
        // When training k-means on many float16 vectors with small k,
        // without limiting the data size, dividing centroids by count
        // can underflow to 0,
        // The fix limits data to k * 512 to prevent this
        const DIM: usize = 2;
        const K: usize = 2;
        const NUM_VALUES: usize = K * 65536; // Many vectors to trigger the issue

        let f32_values = generate_random_array(NUM_VALUES * DIM);
        let f16_values = Float16Array::from_iter_values(
            f32_values.values().iter().map(|&v| half::f16::from_f32(v)),
        );
        let fsl = FixedSizeListArray::try_new_from_values(f16_values, DIM as i32).unwrap();

        let params = KMeansParams {
            max_iters: 10,
            ..Default::default()
        };

        let kmeans = KMeans::new_with_params(&fsl, K, &params).unwrap();

        // Verify that we have the correct number of clusters
        assert_eq!(kmeans.centroids.len(), K * DIM);
        assert_eq!(kmeans.dimension, DIM);
        assert_eq!(kmeans.centroids.data_type(), &DataType::Float16);

        // Verify that all centroids are valid (not zero or NaN)
        // Without the fix, they would all be zero due to underflow
        let centroids = kmeans.centroids.as_primitive::<Float16Type>().values();
        for &val in centroids {
            assert!(!val.is_nan(), "Centroid should not contain NaN values");
            assert!(val != f16::ZERO);
        }
    }

    #[test]
    fn test_hierarchical_kmeans_one_outlier_among_duplicates_errors() {
        // 999 identical rows plus one outlier with a quota of 300: the outlier's
        // sub-cluster earns no centroid and is merged back, which used to hand the
        // whole node to itself again and recurse until the stack overflowed. It
        // must end in the bounded "cannot create" error instead.
        let mut values = vec![1.0_f32; 999];
        values.push(100.0);
        let data = FixedSizeListArray::try_new_from_values(Float32Array::from(values), 1).unwrap();
        let params = KMeansParams {
            max_iters: 10,
            seed: Some(42),
            ..Default::default()
        };
        let err = KMeans::new_with_params(&data, 300, &params)
            .expect_err("duplicates cannot be split into 300 partitions");
        let msg = err.to_string();
        assert!(
            msg.contains("Cannot create") && msg.contains("300"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn test_allocate_quotas_follows_sizes() {
        // Exact shares.
        assert_eq!(allocate_quotas(&[100, 100, 100, 100], 8), vec![2, 2, 2, 2]);
        assert_eq!(allocate_quotas(&[700, 200, 100], 10), vec![7, 2, 1]);
        // Equal fractional parts: the remainder still adds up to the quota.
        let quotas = allocate_quotas(&[1000, 1000, 1000], 10);
        assert_eq!(quotas.iter().sum::<usize>(), 10);
        assert!(quotas.iter().all(|&q| (3..=4).contains(&q)));
        // Sub-clusters whose share rounds to zero get nothing; the remainder
        // goes to the largest fractional share.
        assert_eq!(allocate_quotas(&[1, 1, 1000], 20), vec![0, 0, 20]);
        assert_eq!(allocate_quotas(&[1, 10_000], 10), vec![0, 10]);
        // A sub-cluster never gets more centroids than vectors.
        assert_eq!(allocate_quotas(&[2, 2], 10), vec![2, 2]);
    }

    #[test]
    fn test_split_clusters_takes_half_of_the_largest() {
        let mut cnts = vec![0, 10, 4];
        let mut centroids = vec![0.0f32, 0.0, 1.0, 2.0, 3.0, 4.0];
        split_clusters(&mut cnts, &mut centroids, 2);
        assert_eq!(cnts, vec![5, 5, 4]);
        // The empty cluster's centroid is a perturbed copy of the largest one.
        assert!((centroids[0] - 1.0).abs() < 0.01 && centroids[0] != centroids[2]);
        assert!((centroids[1] - 2.0).abs() < 0.01 && centroids[1] != centroids[3]);
    }

    fn skewed_training_data(rows: usize, dim: usize) -> FixedSizeListArray {
        // 80% of the vectors sit in one dense blob near the origin, the rest
        // spread over the unit cube, so a fixed fan-out would starve the blob
        // of centroids.
        let mut values = generate_random_array(rows * dim).values().to_vec();
        for value in values.iter_mut().take(rows * 8 / 10 * dim) {
            *value *= 0.05;
        }
        FixedSizeListArray::try_new_from_values(Float32Array::from(values), dim as i32).unwrap()
    }

    fn assigned_sizes(kmeans: &KMeans, data: &FixedSizeListArray) -> Vec<usize> {
        let (membership, _) =
            compute_partitions_with_dists::<Float32Type, KMeansAlgoFloat<Float32Type>>(
                kmeans.centroids.as_primitive(),
                data.values().as_primitive(),
                kmeans.dimension,
                kmeans.distance_type,
            );
        let mut sizes = vec![0; kmeans.centroids.len() / kmeans.dimension];
        for cluster in membership.into_iter().flatten() {
            sizes[cluster as usize] += 1;
        }
        sizes
    }

    #[test]
    fn test_hierarchical_kmeans_is_seeded_and_balanced() {
        const DIM: usize = 16;
        const K: usize = 300;
        let data = skewed_training_data(K * 32, DIM);
        let params = KMeansParams {
            max_iters: 10,
            hierarchical_k: 16,
            seed: Some(42),
            ..Default::default()
        };

        let kmeans = KMeans::new_with_params(&data, K, &params).unwrap();
        assert_eq!(kmeans.centroids.len(), K * DIM);
        let again = KMeans::new_with_params(&data, K, &params).unwrap();
        assert_eq!(
            kmeans.centroids.as_primitive::<Float32Type>().values(),
            again.centroids.as_primitive::<Float32Type>().values(),
            "seeded training must be reproducible"
        );

        let sizes = assigned_sizes(&kmeans, &data);
        let mean = data.len() as f64 / K as f64;
        let max = *sizes.iter().max().unwrap() as f64;
        assert_eq!(
            sizes.iter().filter(|&&s| s == 0).count(),
            0,
            "no empty partition"
        );
        assert!(max <= 4.0 * mean, "largest partition {max} vs mean {mean}");
    }

    #[test]
    fn test_hierarchical_refinement_lowers_loss() {
        const DIM: usize = 16;
        const K: usize = 300;
        let data = skewed_training_data(K * 16, DIM);
        let loss_of = |refine_iters: u32| {
            let params = KMeansParams {
                max_iters: 5,
                hierarchical_k: 8,
                refine_iters,
                seed: Some(7),
                ..Default::default()
            };
            KMeans::new_with_params(&data, K, &params)
                .unwrap()
                .compute_loss(&data)
                .unwrap()
        };
        let unrefined = loss_of(0);
        let refined = loss_of(3);
        assert!(
            refined <= unrefined * 1.001,
            "refinement raised the loss from {unrefined} to {refined}"
        );
    }

    #[test]
    fn test_flat_kmeans_seed_is_reproducible() {
        const DIM: usize = 8;
        let data = generate_random_array(4096 * DIM);
        let data = FixedSizeListArray::try_new_from_values(data, DIM as i32).unwrap();
        let params = KMeansParams {
            max_iters: 5,
            seed: Some(3),
            ..Default::default()
        };
        let first = KMeans::new_with_params(&data, 16, &params).unwrap();
        let second = KMeans::new_with_params(&data, 16, &params).unwrap();
        assert_eq!(
            first.centroids.as_primitive::<Float32Type>().values(),
            second.centroids.as_primitive::<Float32Type>().values()
        );
    }

    /// The two centroid-summing strategies must total the same clusters.
    ///
    /// They disagree structurally -- one splits the data and reduces private
    /// accumulators, the other splits the centroids and accumulates in place --
    /// so only one of them runs for any given call and nothing else compares
    /// them. Exact equality is not required: summing in a different order moves
    /// the last bits, and the data-parallel path deliberately accumulates in f32
    /// rather than in the element type.
    ///
    /// Unassigned rows are in the fixture because both paths have to skip them,
    /// by different code: a `filter_map` on one side, a range test on the other.
    ///
    /// The thread count is passed rather than probed so this pins the same
    /// partitioning on every host. Probing would make the assertion vacuous on a
    /// small CI container -- an f32 column with one block's worth of threads
    /// takes the in-place path, and the call under test would return `None`.
    #[test]
    fn test_centroid_sums_agree_across_strategies() {
        const N: usize = 8192;
        const DIM: usize = 16;
        const K: usize = 8;
        const THREADS: usize = 4;

        let mut st = 0x9E37u64;
        let mut next = || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let data: Vec<f32> = (0..N * DIM).map(|_| next()).collect();
        let membership: Vec<Option<u32>> = (0..N)
            .map(|i| (i % 17 != 0).then_some((i % K) as u32))
            .collect();

        let over_data = KMeansAlgoFloat::<Float32Type>::sum_clusters_over_data(
            &data,
            DIM,
            K,
            &membership,
            THREADS,
        )
        .expect("an f32 column with several blocks of rows is this path's own case");
        let mut over_centroids = vec![0f32; K * DIM];
        KMeansAlgoFloat::<Float32Type>::sum_clusters_over_centroids(
            &data,
            DIM,
            K,
            &membership,
            &mut over_centroids,
            THREADS,
        );

        for (i, (over_data, over_centroids)) in over_data.iter().zip(&over_centroids).enumerate() {
            let tolerance = 1e-3 * over_data.abs().max(1.0);
            assert!(
                (over_data - over_centroids).abs() <= tolerance,
                "sums disagree at {i}: {over_data} (over data) vs {over_centroids} (over centroids)"
            );
        }
    }

    /// A cluster whose running total outgrows f16 must still produce the mean of
    /// its members -- on any number of cores.
    ///
    /// f16 carries 11 significant bits, so a total past 2048 can no longer
    /// represent an increment of 1: summing 4096 ones in f16 stops dead at 2048
    /// and the mean comes out at half its true value. The in-place path this
    /// compares against relies on `train_kmeans` capping its input at `k * 512`
    /// rows to stay clear of that (see `test_float16_underflow_fix`); the
    /// data-partitioned path accumulates in f32 and must not need the cap. The
    /// single-block case is asserted explicitly below because it is what a CI
    /// container gets, and the guard has to hold there too.
    #[test]
    fn test_centroid_sum_survives_a_cluster_f16_cannot_total() {
        const N: usize = 4096;
        const DIM: usize = 4;
        const K: usize = 1;

        let data = vec![f16::ONE; N * DIM];
        let membership: Vec<Option<u32>> = vec![Some(0); N];
        let mut cluster_sizes = vec![N];

        let kmeans = <KMeansAlgoFloat<Float16Type> as KMeansAlgo<f16>>::to_kmeans(
            &data,
            DIM,
            K,
            &membership,
            &mut cluster_sizes,
            DistanceType::L2,
            0.0,
        );
        let centroids = kmeans.centroids.as_primitive::<Float16Type>().values();
        assert_eq!(
            centroids,
            [f16::ONE; DIM].as_slice(),
            "the mean of {N} copies of 1.0 is 1.0"
        );

        // The single-core case, which is the one a small CI host runs and the
        // one an earlier revision of this code got wrong by handing f16 back to
        // the in-place path whenever there was no parallelism to be had.
        let single_block =
            KMeansAlgoFloat::<Float16Type>::sum_clusters_over_data(&data, DIM, K, &membership, 1)
                .expect("an f16 column takes this path for its accumulator, not for its threads");
        assert_eq!(
            single_block,
            vec![N as f32; DIM],
            "one block must still total {N} rows of 1.0 exactly"
        );

        // The premise: f16 really cannot hold this total, so the assertions
        // above are testing the accumulator and not something the element type
        // would have got right anyway.
        let mut in_element_type = vec![f16::ZERO; K * DIM];
        KMeansAlgoFloat::<Float16Type>::sum_clusters_over_centroids(
            &data,
            DIM,
            K,
            &membership,
            &mut in_element_type,
            1,
        );
        assert!(
            in_element_type[0].to_f32() < N as f32,
            "an f16 accumulator was expected to lose increments past 2048, but it totalled {}",
            in_element_type[0]
        );
    }

    /// Which shapes the data-parallel path hands back to the in-place one, and
    /// which it keeps.
    ///
    /// Every one of these is a quiet failure if the routing is wrong: an f64
    /// column would be summed at two thirds of its mantissa, an oversized `k`
    /// would allocate past the budget meant to bound it, and an f16 column sent
    /// to the in-place path would lose increments (see the test above).
    #[test]
    fn test_centroid_sum_over_data_routes_shapes_it_cannot_serve() {
        const DIM: usize = 8;
        const K: usize = 4;
        const N: usize = 8192;

        let membership: Vec<Option<u32>> = vec![Some(0); N];

        let f64_data = vec![1.0f64; N * DIM];
        assert!(
            KMeansAlgoFloat::<Float64Type>::sum_clusters_over_data(
                &f64_data,
                DIM,
                K,
                &membership,
                8
            )
            .is_none(),
            "an f64 column must keep the accumulator that holds its mantissa"
        );

        // One accumulator past the budget: `k * dimension * 4` here is 800 MB
        // against a 512 MB allowance. Typed f16 deliberately -- for every other
        // element type the `blocks < 2` rule below would decline this shape on
        // its own (the budget division floors to zero and the count clamps up to
        // one), so an f32 fixture here would still pass with the budget check
        // deleted. f16 is exempt from that rule, which leaves the budget as the
        // only thing standing between this call and an 800 MB allocation.
        // Nothing of that size is allocated -- the check happens before the
        // buffer does.
        let tiny = vec![f16::ONE; N];
        assert!(
            KMeansAlgoFloat::<Float16Type>::sum_clusters_over_data(
                &tiny,
                1,
                200_000_000,
                &membership,
                8
            )
            .is_none(),
            "an accumulator larger than the whole budget must not be allocated"
        );

        let f32_rows = vec![1.0f32; MIN_ROWS_PER_SUM_BLOCK * DIM];
        let f32_membership: Vec<Option<u32>> = vec![Some(0); MIN_ROWS_PER_SUM_BLOCK];
        assert!(
            KMeansAlgoFloat::<Float32Type>::sum_clusters_over_data(
                &f32_rows,
                DIM,
                K,
                &f32_membership,
                8
            )
            .is_none(),
            "one block's worth of f32 rows buys no parallelism and needs no wider accumulator"
        );

        // The same shape in f16 is kept, because there the accumulator is the
        // point.
        let f16_rows = vec![f16::ONE; MIN_ROWS_PER_SUM_BLOCK * DIM];
        assert!(
            KMeansAlgoFloat::<Float16Type>::sum_clusters_over_data(
                &f16_rows,
                DIM,
                K,
                &f32_membership,
                8
            )
            .is_some(),
            "an f16 column needs the wider accumulator whether or not it needs the threads"
        );
    }

    /// Against an in-place update that is itself parallel, the data-parallel path
    /// must keep only the shapes whose accumulators are small next to the work
    /// it saves, and must keep every shape where the in-place update is serial.
    ///
    /// The failure this guards is a slowdown rather than a wrong answer: both
    /// strategies sum to the same centroids, so nothing else would notice a
    /// shape landing on the slower one. Each case fixes `blocks` by construction
    /// -- `n / MIN_ROWS_PER_SUM_BLOCK` at or below the thread count -- so it is
    /// the allowance arithmetic that is being tested, not the host.
    #[test]
    fn test_centroid_sum_over_data_yields_to_a_parallel_in_place_update() {
        const THREADS: usize = 8;
        let rows = |n: usize, dim: usize| vec![1.0f32; n * dim];
        let rows_f16 = |n: usize, dim: usize| vec![f16::ONE; n * dim];
        let membership = |n: usize, k: usize| -> Vec<Option<u32>> {
            (0..n).map(|i| Some((i % k) as u32)).collect()
        };

        // k >= threads, so the in-place update runs 8 owners in parallel. 4
        // blocks of 256 x 64 fold 65536 floats against 4096 rows: 16 per row,
        // past the f32 allowance of 4. f16 also saves 64 / 8 scalar adds per
        // row, worth 6 folded floats each, and keeps the shape.
        let (n, dim, k) = (4096, 64, 256);
        assert!(
            KMeansAlgoFloat::<Float32Type>::sum_clusters_over_data(
                &rows(n, dim),
                dim,
                k,
                &membership(n, k),
                THREADS
            )
            .is_none(),
            "f32 must hand back a shape whose fold outweighs the membership walk it saves"
        );
        assert!(
            KMeansAlgoFloat::<Float16Type>::sum_clusters_over_data(
                &rows_f16(n, dim),
                dim,
                k,
                &membership(n, k),
                THREADS
            )
            .is_some(),
            "f16 also saves the in-place path's scalar adds, so it keeps this shape"
        );

        // Same parallel in-place update, but a PQ sub-vector: 8 blocks of
        // 256 x 8 fold 16384 floats against 8192 rows, 2 per row.
        let (n, dim, k) = (8192, 8, 256);
        assert!(
            KMeansAlgoFloat::<Float32Type>::sum_clusters_over_data(
                &rows(n, dim),
                dim,
                k,
                &membership(n, k),
                THREADS
            )
            .is_some(),
            "small accumulators against a long membership are this path's win even at k >= threads"
        );

        // Past the f16 allowance too: 8 blocks of 4096 x 8 fold 32 floats per
        // row, and at `dimension` 8 over 8 owners the scalar adds saved are
        // worth only 6 more.
        let (n, dim, k) = (8192, 8, 4096);
        assert!(
            KMeansAlgoFloat::<Float16Type>::sum_clusters_over_data(
                &rows_f16(n, dim),
                dim,
                k,
                &membership(n, k),
                THREADS
            )
            .is_none(),
            "f16 must hand back a shape whose fold outweighs both the walk and the adds it saves"
        );

        // k < threads: the in-place update is serial here, so the allowance
        // must not apply -- 8 blocks of 4 x 1024 fold 32768 floats against 8192
        // rows, which the rule above would refuse.
        let (n, dim, k) = (8192, 1024, 4);
        assert!(
            KMeansAlgoFloat::<Float32Type>::sum_clusters_over_data(
                &rows(n, dim),
                dim,
                k,
                &membership(n, k),
                THREADS
            )
            .is_some(),
            "with fewer centroids than threads the in-place update is serial and this path always wins"
        );

        // k >= threads but k < 16: the in-place update is serial for the other
        // reason, and the same must hold.
        let (n, dim, k) = (8192, 1024, 12);
        assert_eq!(
            KMeansAlgoFloat::<Float32Type>::centroid_owner_blocks(k, THREADS),
            1
        );
        assert!(
            KMeansAlgoFloat::<Float32Type>::sum_clusters_over_data(
                &rows(n, dim),
                dim,
                k,
                &membership(n, k),
                THREADS
            )
            .is_some(),
            "below 16 centroids the in-place update is serial whatever the thread count"
        );
    }

    /// A membership id past the last centroid must be dropped, not indexed with.
    ///
    /// The in-place path drops it as a side effect of testing the id against the
    /// centroid range it owns; the data-parallel path indexes its accumulator
    /// directly and would panic on the same input, so it filters explicitly.
    #[test]
    fn test_centroid_sums_ignore_out_of_range_membership() {
        const N: usize = 4096;
        const DIM: usize = 4;
        const K: usize = 2;
        const THREADS: usize = 4;

        let data = vec![1.0f32; N * DIM];
        let membership: Vec<Option<u32>> = (0..N)
            .map(|i| Some(if i % 2 == 0 { 0 } else { K as u32 + 7 }))
            .collect();

        let over_data = KMeansAlgoFloat::<Float32Type>::sum_clusters_over_data(
            &data,
            DIM,
            K,
            &membership,
            THREADS,
        )
        .expect("an f32 column with several blocks of rows is this path's own case");
        let mut over_centroids = vec![0f32; K * DIM];
        KMeansAlgoFloat::<Float32Type>::sum_clusters_over_centroids(
            &data,
            DIM,
            K,
            &membership,
            &mut over_centroids,
            THREADS,
        );

        assert_eq!(
            over_data[0],
            (N / 2) as f32,
            "only the in-range half of the rows should have been summed"
        );
        assert_eq!(over_data, over_centroids);
    }

    /// An empty input must total to zeros rather than panicking.
    ///
    /// `par_chunks` rejects a zero chunk size, which is what the block
    /// arithmetic computes for an input with no rows at all.
    #[test]
    fn test_centroid_sum_over_data_handles_empty_input() {
        let sums = KMeansAlgoFloat::<Float16Type>::sum_clusters_over_data(&[], 8, 4, &[], 8)
            .expect("an empty f16 column still takes the wider accumulator");
        assert_eq!(sums, vec![0f32; 32]);
    }
}
