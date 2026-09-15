// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Hamming sufficient statistics and weighted k-modes for streaming IVF training.

use std::{
    cmp::Ordering,
    collections::{BTreeSet, BinaryHeap},
};

use arrow_array::{Array, FixedSizeListArray, UInt8Array, cast::AsArray};
use arrow_schema::DataType;
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_index::vector::kmeans::KMeans;
use lance_linalg::distance::DistanceType;
use rand::{Rng, SeedableRng, rngs::SmallRng};

use super::{KMeansProgressCallback, KMeansStepOptions, train_ivf_kmeans_step_arrow_array_no_loss};

mod counter;

use counter::CompactCounters;
#[cfg(test)]
use counter::{CounterWidth, add_vector_bits, mode_from_counts, summary_cost};

#[derive(Clone, Debug)]
pub(super) struct HammingCoreset {
    dimension: usize,
    counts: Vec<u64>,
    ones: CompactCounters,
    representatives: BTreeSet<Vec<u8>>,
    representative_capacity: usize,
}

impl HammingCoreset {
    pub(super) fn new(dimension: usize, capacity: usize) -> Self {
        Self {
            dimension,
            counts: Vec::with_capacity(capacity),
            ones: CompactCounters::with_capacity(
                capacity.saturating_mul(dimension).saturating_mul(8),
            ),
            representatives: BTreeSet::new(),
            representative_capacity: capacity,
        }
    }

    pub(super) fn len(&self) -> usize {
        self.counts.len()
    }

    fn bit_dimension(&self) -> usize {
        self.dimension * 8
    }

    fn summary_start(&self, index: usize) -> usize {
        index * self.bit_dimension()
    }

    fn insert_representative(&mut self, vector: &[u8]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::invalid_input(format!(
                "Hamming representative dimension {}, expected {}",
                vector.len(),
                self.dimension
            )));
        }
        if self.representative_capacity == 0 {
            return Ok(());
        }
        self.representatives.insert(vector.to_vec());
        if self.representatives.len() > self.representative_capacity {
            let _ = self.representatives.pop_last();
        }
        Ok(())
    }

    fn extend_representatives(
        &mut self,
        representatives: impl IntoIterator<Item = Vec<u8>>,
    ) -> Result<()> {
        for representative in representatives {
            self.insert_representative(&representative)?;
        }
        Ok(())
    }

    fn validate_distinct_representatives(&self, target_k: usize) -> Result<()> {
        if self.representative_capacity < target_k {
            return Err(Error::invalid_input(format!(
                "Hamming coreset representative capacity {} is smaller than the requested {target_k} partitions",
                self.representative_capacity
            )));
        }
        if self.representatives.len() < target_k {
            return Err(Error::invalid_input(format!(
                "weighted Hamming training requires at least {target_k} distinct training vectors, but sampled only {}",
                self.representatives.len()
            )));
        }
        Ok(())
    }

    #[cfg(test)]
    fn push(&mut self, count: u64, ones: &[u64]) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if ones.len() != self.bit_dimension() {
            return Err(Error::invalid_input(format!(
                "Hamming coreset summary has {} bit counters, expected {}",
                ones.len(),
                self.bit_dimension()
            )));
        }
        if let Some((bit, value)) = ones
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value > count)
        {
            return Err(Error::invalid_input(format!(
                "Hamming coreset one-count at bit {bit} is {value}, greater than row count {count}"
            )));
        }
        self.insert_representative(&mode_from_counts(count, ones, self.dimension))?;
        self.counts.push(count);
        self.ones.extend_u64(ones, count)?;
        Ok(())
    }

    fn push_compact(&mut self, count: u64, ones: &CompactCounters, start: usize) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let bit_dimension = self.bit_dimension();
        let end = start.checked_add(bit_dimension).ok_or_else(|| {
            Error::invalid_input("Hamming summary counter range overflow while appending")
        })?;
        if end > ones.len() {
            return Err(Error::invalid_input(format!(
                "Hamming summary counter range {start}..{end} exceeds length {}",
                ones.len()
            )));
        }
        if let Some((bit, value)) = (start..end)
            .map(|index| ones.value(index))
            .enumerate()
            .find(|(_, value)| *value > count)
        {
            return Err(Error::invalid_input(format!(
                "Hamming coreset one-count at bit {bit} is {value}, greater than row count {count}"
            )));
        }
        self.counts.push(count);
        self.ones.extend_from(ones, start, bit_dimension, count)?;
        Ok(())
    }

    pub(super) fn append(&mut self, other: Self) -> Result<()> {
        if self.dimension != other.dimension {
            return Err(Error::invalid_input(format!(
                "cannot append Hamming coreset dimension {} to dimension {}",
                other.dimension, self.dimension
            )));
        }
        let Self {
            counts,
            ones,
            representatives,
            ..
        } = other;
        self.extend_representatives(representatives)?;
        self.counts.extend(counts);
        self.ones.append(ones)?;
        Ok(())
    }

    pub(super) fn reduce_to_budget(&mut self, budget: usize) -> Result<()> {
        if self.len() <= budget {
            return Ok(());
        }
        if budget == 0 {
            return Err(Error::invalid_input(
                "Hamming coreset budget must be greater than zero",
            ));
        }

        let total_count = self.counts.iter().try_fold(0_u64, |total, count| {
            total.checked_add(*count).ok_or_else(|| {
                Error::invalid_input("Hamming coreset row count overflow during reduction")
            })
        })?;
        if total_count == 0 {
            let mut reduced = Self::new(self.dimension, budget);
            reduced.extend_representatives(self.representatives.iter().cloned())?;
            *self = reduced;
            return Ok(());
        }

        let bit_dimension = self.bit_dimension();
        let mut weighted_sums = vec![0.0; bit_dimension];
        let mut weighted_square_sums = vec![0.0; bit_dimension];
        for row in 0..self.len() {
            let count = self.counts[row] as f64;
            let summary_start = self.summary_start(row);
            for bit in 0..bit_dimension {
                let fraction = self.ones.value(summary_start + bit) as f64 / count;
                weighted_sums[bit] += count * fraction;
                weighted_square_sums[bit] += count * fraction * fraction;
            }
        }
        let total_count = total_count as f64;
        let variances = weighted_sums
            .into_iter()
            .zip(weighted_square_sums)
            .map(|(weighted_sum, weighted_square_sum)| {
                let mean = weighted_sum / total_count;
                weighted_square_sum / total_count - mean * mean
            })
            .collect::<Vec<_>>();
        let split_bit = variances
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
            .map(|(bit, _)| bit)
            .unwrap_or(0);

        let mut indices = (0..self.len()).collect::<Vec<_>>();
        indices.sort_unstable_by(|left, right| {
            let left_count = self.counts[*left] as u128;
            let right_count = self.counts[*right] as u128;
            let left_ones = self.ones.value(self.summary_start(*left) + split_bit) as u128;
            let right_ones = self.ones.value(self.summary_start(*right) + split_bit) as u128;
            (left_ones * right_count)
                .cmp(&(right_ones * left_count))
                .then_with(|| left.cmp(right))
        });

        let mut reduced = Self::new(self.dimension, budget);
        for group_index in 0..budget {
            let group_start = group_index * indices.len() / budget;
            let group_end = (group_index + 1) * indices.len() / budget;
            if group_start == group_end {
                continue;
            }
            let group_indices = &indices[group_start..group_end];
            let count = group_indices.iter().try_fold(0_u64, |count, index| {
                count.checked_add(self.counts[*index]).ok_or_else(|| {
                    Error::invalid_input("Hamming coreset row count overflow during merge")
                })
            })?;
            let mut ones = CompactCounters::zeros(bit_dimension);
            for &index in group_indices {
                ones.add_from(
                    0,
                    &self.ones,
                    self.summary_start(index),
                    bit_dimension,
                    count,
                )?;
            }
            reduced.push_compact(count, &ones, 0)?;
        }
        reduced.extend_representatives(self.representatives.iter().cloned())?;
        *self = reduced;
        Ok(())
    }

    fn subset(&self, indices: &[usize]) -> Result<Self> {
        let mut subset = Self::new(self.dimension, indices.len());
        for &index in indices {
            subset.push_compact(self.counts[index], &self.ones, self.summary_start(index))?;
        }
        Ok(subset)
    }

    fn mode(&self, index: usize) -> Vec<u8> {
        self.ones.mode(
            self.summary_start(index),
            self.counts[index],
            self.dimension,
        )
    }

    fn cost(&self, index: usize, centroid: &[u8]) -> Result<u64> {
        self.ones.cost(
            self.summary_start(index),
            self.counts[index],
            centroid,
            self.dimension,
        )
    }
}

#[derive(Debug)]
pub(super) struct HammingAccumulator {
    dimension: usize,
    counts: Vec<u64>,
    ones: CompactCounters,
}

impl HammingAccumulator {
    pub(super) fn try_new(num_clusters: usize, dimension: usize) -> Result<Self> {
        let bit_dimension = dimension.checked_mul(8).ok_or_else(|| {
            Error::invalid_input(format!(
                "Hamming accumulator bit dimension overflow for dimension {dimension}"
            ))
        })?;
        let num_counters = num_clusters.checked_mul(bit_dimension).ok_or_else(|| {
            Error::invalid_input(format!(
                "Hamming accumulator size overflow for {num_clusters} clusters and dimension {dimension}"
            ))
        })?;
        Ok(Self {
            dimension,
            counts: vec![0; num_clusters],
            ones: CompactCounters::zeros(num_counters),
        })
    }

    fn bit_dimension(&self) -> usize {
        self.dimension * 8
    }

    fn add_vector(&mut self, cluster_id: usize, vector: &[u8]) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(Error::invalid_input(format!(
                "Hamming accumulator vector dimension {}, expected {}",
                vector.len(),
                self.dimension
            )));
        }
        let count = self
            .counts
            .get(cluster_id)
            .copied()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Hamming accumulator cluster {cluster_id} is outside 0..{}",
                    self.counts.len()
                ))
            })?
            .checked_add(1)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Hamming accumulator row count overflow for cluster {cluster_id}"
                ))
            })?;
        self.ones
            .add_vector(cluster_id * self.bit_dimension(), vector, count)?;
        self.counts[cluster_id] = count;
        Ok(())
    }

    fn add_summary(
        &mut self,
        cluster_id: usize,
        coreset: &HammingCoreset,
        summary_index: usize,
    ) -> Result<()> {
        if self.dimension != coreset.dimension {
            return Err(Error::invalid_input(format!(
                "Hamming accumulator dimension {} does not match coreset dimension {}",
                self.dimension, coreset.dimension
            )));
        }
        let summary_count = coreset.counts.get(summary_index).copied().ok_or_else(|| {
            Error::invalid_input(format!(
                "Hamming coreset summary {summary_index} is outside 0..{}",
                coreset.len()
            ))
        })?;
        let count = self
            .counts
            .get(cluster_id)
            .copied()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Hamming accumulator cluster {cluster_id} is outside 0..{}",
                    self.counts.len()
                ))
            })?
            .checked_add(summary_count)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "Hamming accumulator row count overflow for cluster {cluster_id}"
                ))
            })?;
        self.ones.add_from(
            cluster_id * self.bit_dimension(),
            &coreset.ones,
            coreset.summary_start(summary_index),
            self.bit_dimension(),
            count,
        )?;
        self.counts[cluster_id] = count;
        Ok(())
    }

    fn count(&self, cluster_id: usize) -> u64 {
        self.counts[cluster_id]
    }

    fn mode(&self, cluster_id: usize) -> Vec<u8> {
        self.ones.mode(
            cluster_id * self.bit_dimension(),
            self.counts[cluster_id],
            self.dimension,
        )
    }

    pub(super) fn total_count(&self) -> Result<u64> {
        self.counts.iter().try_fold(0_u64, |total, count| {
            total
                .checked_add(*count)
                .ok_or_else(|| Error::invalid_input("Hamming accumulator total row count overflow"))
        })
    }
}

pub(super) fn append_local_coreset(
    coreset: &mut HammingCoreset,
    data: &FixedSizeListArray,
    local_k: usize,
    max_iters: usize,
    on_progress: KMeansProgressCallback,
) -> Result<()> {
    if data.value_type() != DataType::UInt8 {
        return Err(Error::invalid_input(format!(
            "streaming Hamming k-modes requires UInt8 vectors, got {}",
            data.value_type()
        )));
    }
    let dimension = data.value_length() as usize;
    let local_k = local_k.max(1);
    let sample_rate = data.len().div_ceil(local_k).max(1);
    let kmeans = train_ivf_kmeans_step_arrow_array_no_loss(
        None,
        data,
        KMeansStepOptions {
            dimension,
            metric_type: DistanceType::Hamming,
            num_partitions: local_k,
            sample_rate,
            max_iters,
            disable_hierarchical: false,
            on_progress,
        },
    )?;
    let centroids = FixedSizeListArray::try_new_from_values(kmeans.centroids, dimension as i32)?;
    let assignment_model = KMeans::with_centroids(
        centroids.values().clone(),
        dimension,
        DistanceType::Hamming,
        f64::MAX,
    );
    let (membership, _) = assignment_model.compute_membership_and_distances(data)?;
    let values = data
        .values()
        .as_primitive::<arrow_array::types::UInt8Type>()
        .values();
    let mut summaries = HammingAccumulator::try_new(centroids.len(), dimension)?;
    for (row_index, member) in membership.into_iter().enumerate() {
        let Some(cluster_id) = member.map(|value| value as usize) else {
            continue;
        };
        let vector = &values[row_index * dimension..(row_index + 1) * dimension];
        coreset.insert_representative(vector)?;
        summaries.add_vector(cluster_id, vector)?;
    }
    for cluster_id in 0..centroids.len() {
        coreset.push_compact(
            summaries.count(cluster_id),
            &summaries.ones,
            cluster_id * summaries.bit_dimension(),
        )?;
    }
    Ok(())
}

#[derive(Debug)]
struct WeightedKModesResult {
    centroids: Vec<u8>,
    membership: Vec<u32>,
    cluster_counts: Vec<u64>,
    cluster_losses: Vec<f64>,
    loss: f64,
}

fn initialize_centroids(coreset: &HammingCoreset, k: usize) -> Result<Vec<u8>> {
    let mut rng = SmallRng::seed_from_u64(0x1f17_5eed);
    let mut selected = vec![false; coreset.len()];
    let total_count = coreset.counts.iter().try_fold(0_u64, |total, count| {
        total
            .checked_add(*count)
            .ok_or_else(|| Error::invalid_input("Hamming initialization row count overflow"))
    })?;
    let first = if total_count > 0 {
        let mut threshold = rng.random_range(0..total_count);
        let mut row_index = 0;
        for (index, count) in coreset.counts.iter().copied().enumerate() {
            if threshold < count {
                row_index = index;
                break;
            }
            threshold -= count;
        }
        row_index
    } else {
        0
    };
    selected[first] = true;
    let mut centroids = coreset.mode(first);
    let mut min_costs = vec![u64::MAX; coreset.len()];

    while centroids.len() / coreset.dimension < k {
        let last = &centroids[centroids.len() - coreset.dimension..];
        for row_index in 0..coreset.len() {
            if selected[row_index] {
                min_costs[row_index] = 0;
            } else {
                min_costs[row_index] = min_costs[row_index].min(coreset.cost(row_index, last)?);
            }
        }
        let total_cost = min_costs.iter().try_fold(0_u64, |total, cost| {
            total
                .checked_add(*cost)
                .ok_or_else(|| Error::invalid_input("Hamming initialization distance sum overflow"))
        })?;
        let next = if total_cost > 0 {
            let mut threshold = rng.random_range(0..total_cost);
            let mut choice = None;
            for (index, cost) in min_costs.iter().copied().enumerate() {
                if selected[index] {
                    continue;
                }
                if threshold < cost {
                    choice = Some(index);
                    break;
                }
                threshold -= cost;
            }
            choice
        } else {
            None
        }
        .or_else(|| (0..coreset.len()).find(|index| !selected[*index]));
        let Some(next) = next else {
            break;
        };
        selected[next] = true;
        centroids.extend(coreset.mode(next));
    }
    while centroids.len() / coreset.dimension < k {
        let row_index = (centroids.len() / coreset.dimension) * coreset.len() / k;
        centroids.extend(coreset.mode(row_index));
    }
    Ok(centroids)
}

fn assign_summaries(coreset: &HammingCoreset, centroids: &[u8]) -> Result<WeightedKModesResult> {
    let k = centroids.len() / coreset.dimension;
    let mut membership = Vec::with_capacity(coreset.len());
    let mut accumulator = HammingAccumulator::try_new(k, coreset.dimension)?;
    let mut cluster_losses = vec![0.0_f64; k];

    for row_index in 0..coreset.len() {
        let mut best_assignment = None;
        for cluster_id in 0..k {
            let centroid =
                &centroids[cluster_id * coreset.dimension..(cluster_id + 1) * coreset.dimension];
            let cost = coreset.cost(row_index, centroid)?;
            if best_assignment
                .as_ref()
                .is_none_or(|(_, best_cost)| cost < *best_cost)
            {
                best_assignment = Some((cluster_id, cost));
            }
        }
        let (cluster_id, cost) = best_assignment
            .ok_or_else(|| Error::index("weighted Hamming training has no centroids"))?;
        membership.push(cluster_id as u32);
        accumulator.add_summary(cluster_id, coreset, row_index)?;
        cluster_losses[cluster_id] += cost as f64;
    }

    let mut next_centroids = Vec::with_capacity(centroids.len());
    for cluster_id in 0..k {
        if accumulator.count(cluster_id) == 0 {
            next_centroids.extend_from_slice(
                &centroids[cluster_id * coreset.dimension..(cluster_id + 1) * coreset.dimension],
            );
        } else {
            next_centroids.extend(accumulator.mode(cluster_id));
        }
    }
    let loss = cluster_losses.iter().sum();
    Ok(WeightedKModesResult {
        centroids: next_centroids,
        membership,
        cluster_counts: accumulator.counts,
        cluster_losses,
        loss,
    })
}

fn has_converged(previous_loss: f64, loss: f64) -> bool {
    if loss == 0.0 {
        previous_loss == 0.0
    } else {
        (previous_loss - loss).abs() < 1e-4 * loss.abs()
    }
}

fn train_weighted_kmodes(
    coreset: &HammingCoreset,
    k: usize,
    max_iters: usize,
    on_progress: KMeansProgressCallback,
) -> Result<WeightedKModesResult> {
    if coreset.len() < k {
        return Err(Error::invalid_input(format!(
            "weighted Hamming training requires at least {k} summaries, got {}",
            coreset.len()
        )));
    }
    let mut centroids = initialize_centroids(coreset, k)?;
    let mut previous_loss = f64::MAX;
    let max_iters = max_iters.max(1);
    for iteration in 1..=max_iters {
        on_progress(iteration as u32, max_iters as u32);
        let mut result = assign_summaries(coreset, &centroids)?;
        let converged = has_converged(previous_loss, result.loss);
        previous_loss = result.loss;
        if converged || iteration == max_iters {
            return Ok(result);
        }
        centroids = std::mem::take(&mut result.centroids);
    }
    unreachable!("weighted Hamming training runs at least one iteration")
}

#[derive(Clone, Debug)]
struct WeightedCluster {
    id: usize,
    indices: Vec<usize>,
    centroid: Vec<u8>,
    count: u64,
    loss: f64,
    finalized: bool,
}

impl Eq for WeightedCluster {}

impl PartialEq for WeightedCluster {
    fn eq(&self, other: &Self) -> bool {
        self.finalized == other.finalized && self.loss == other.loss && self.count == other.count
    }
}

impl Ord for WeightedCluster {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.finalized, other.finalized) {
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            _ => self
                .loss
                .partial_cmp(&other.loss)
                .unwrap_or(Ordering::Equal)
                .then_with(|| self.count.cmp(&other.count)),
        }
    }
}

impl PartialOrd for WeightedCluster {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn complete_distinct_centroids(
    candidates: impl IntoIterator<Item = Vec<u8>>,
    fallbacks: impl IntoIterator<Item = Vec<u8>>,
    target_k: usize,
    dimension: usize,
) -> Result<Vec<u8>> {
    if target_k == 0 {
        return Ok(Vec::new());
    }
    let mut distinct = BTreeSet::new();
    let mut values = Vec::with_capacity(target_k.saturating_mul(dimension));
    for centroid in candidates.into_iter().chain(fallbacks) {
        if centroid.len() != dimension {
            return Err(Error::invalid_input(format!(
                "Hamming centroid dimension {}, expected {dimension}",
                centroid.len()
            )));
        }
        if distinct.insert(centroid.clone()) {
            values.extend(centroid);
            if distinct.len() == target_k {
                return Ok(values);
            }
        }
    }
    Err(Error::index(format!(
        "Hamming centroid completion produced only {} distinct centroids for {target_k} partitions",
        distinct.len()
    )))
}

pub(super) fn train_hierarchical(
    coreset: &HammingCoreset,
    target_k: usize,
    max_iters: usize,
    on_progress: KMeansProgressCallback,
) -> Result<FixedSizeListArray> {
    if coreset.len() == 0 {
        return Err(Error::index("empty Hamming coreset"));
    }
    coreset.validate_distinct_representatives(target_k)?;
    let initial_k = 16_usize.min(target_k).min(coreset.len()).max(1);
    let initial = train_weighted_kmodes(coreset, initial_k, max_iters, on_progress.clone())?;
    let mut heap = BinaryHeap::new();
    let mut next_cluster_id = 0;
    for cluster_id in 0..initial_k {
        let indices = initial
            .membership
            .iter()
            .enumerate()
            .filter_map(|(index, member)| (*member as usize == cluster_id).then_some(index))
            .collect::<Vec<_>>();
        if !indices.is_empty() {
            heap.push(WeightedCluster {
                id: next_cluster_id,
                indices,
                centroid: initial.centroids
                    [cluster_id * coreset.dimension..(cluster_id + 1) * coreset.dimension]
                    .to_vec(),
                count: initial.cluster_counts[cluster_id],
                loss: initial.cluster_losses[cluster_id],
                finalized: false,
            });
            next_cluster_id += 1;
        }
    }

    while heap.len() < target_k {
        let mut cluster = heap
            .pop()
            .ok_or_else(|| Error::index("No weighted Hamming cluster can be further split"))?;
        if cluster.finalized {
            heap.push(cluster);
            break;
        }
        if cluster.indices.len() <= 1 {
            cluster.finalized = true;
            heap.push(cluster);
            continue;
        }
        let remaining_k = target_k - heap.len();
        let cluster_k = if cluster.indices.len() <= 16 {
            2.min(remaining_k).min(cluster.indices.len())
        } else {
            (cluster.indices.len() / 16).min(remaining_k).clamp(2, 16)
        };
        let subset = coreset.subset(&cluster.indices)?;
        let split =
            train_weighted_kmodes(&subset, cluster_k, max_iters.min(20), on_progress.clone())?;
        let mut assignments = vec![Vec::new(); cluster_k];
        for (local_index, member) in split.membership.iter().copied().enumerate() {
            assignments[member as usize].push(cluster.indices[local_index]);
        }
        let nonempty = assignments
            .iter()
            .filter(|indices| !indices.is_empty())
            .count();
        if nonempty <= 1 {
            cluster.finalized = true;
            heap.push(cluster);
            continue;
        }
        for (child_id, child_indices) in assignments.into_iter().enumerate() {
            if child_indices.is_empty() {
                continue;
            }
            heap.push(WeightedCluster {
                id: next_cluster_id,
                indices: child_indices,
                centroid: split.centroids
                    [child_id * coreset.dimension..(child_id + 1) * coreset.dimension]
                    .to_vec(),
                count: split.cluster_counts[child_id],
                loss: split.cluster_losses[child_id],
                finalized: false,
            });
            next_cluster_id += 1;
        }
    }

    let mut clusters = heap.into_vec();
    clusters.sort_by_key(|cluster| cluster.id);
    let values = complete_distinct_centroids(
        clusters.into_iter().map(|cluster| cluster.centroid),
        coreset.representatives.iter().cloned(),
        target_k,
        coreset.dimension,
    )?;
    Ok(FixedSizeListArray::try_new_from_values(
        UInt8Array::from(values),
        coreset.dimension as i32,
    )?)
}

pub(super) fn refine_weighted(
    coreset: &HammingCoreset,
    initial_centroids: &FixedSizeListArray,
    max_iters: usize,
    on_progress: KMeansProgressCallback,
) -> Result<FixedSizeListArray> {
    let target_k = initial_centroids.len();
    let mut centroids = initial_centroids
        .values()
        .as_primitive::<arrow_array::types::UInt8Type>()
        .values()
        .to_vec();
    let mut previous_loss = f64::MAX;
    for iteration in 1..=max_iters.max(1) {
        on_progress(iteration as u32, max_iters.max(1) as u32);
        let result = assign_summaries(coreset, &centroids)?;
        let converged = has_converged(previous_loss, result.loss);
        previous_loss = result.loss;
        centroids = result.centroids;
        if converged {
            break;
        }
    }
    let centroids = complete_distinct_centroids(
        centroids
            .chunks_exact(coreset.dimension)
            .map(|centroid| centroid.to_vec()),
        coreset.representatives.iter().cloned(),
        target_k,
        coreset.dimension,
    )?;
    Ok(FixedSizeListArray::try_new_from_values(
        UInt8Array::from(centroids),
        coreset.dimension as i32,
    )?)
}

pub(super) fn accumulate_raw_assignments(
    data: &FixedSizeListArray,
    centroids: &FixedSizeListArray,
    accumulator: &mut HammingAccumulator,
) -> Result<f64> {
    let dimension = data.value_length() as usize;
    if accumulator.dimension != dimension || accumulator.counts.len() != centroids.len() {
        return Err(Error::invalid_input(format!(
            "Hamming refinement accumulator has {} clusters of dimension {}, expected {} clusters of dimension {dimension}",
            accumulator.counts.len(),
            accumulator.dimension,
            centroids.len()
        )));
    }
    let model = KMeans::with_centroids(
        centroids.values().clone(),
        dimension,
        DistanceType::Hamming,
        f64::MAX,
    );
    let (membership, distances) = model.compute_membership_and_distances(data)?;
    let values = data
        .values()
        .as_primitive::<arrow_array::types::UInt8Type>()
        .values();
    let mut loss = 0.0;
    for row_index in 0..data.len() {
        let (Some(cluster_id), Some(distance)) = (membership[row_index], distances[row_index])
        else {
            continue;
        };
        let cluster_id = cluster_id as usize;
        accumulator.add_vector(
            cluster_id,
            &values[row_index * dimension..(row_index + 1) * dimension],
        )?;
        loss += distance as f64;
    }
    Ok(loss)
}

pub(super) fn update_raw_centroids(
    centroids: &FixedSizeListArray,
    accumulator: &HammingAccumulator,
) -> Result<FixedSizeListArray> {
    let dimension = centroids.value_length() as usize;
    if accumulator.dimension != dimension || accumulator.counts.len() != centroids.len() {
        return Err(Error::invalid_input(format!(
            "Hamming refinement accumulator has {} clusters of dimension {}, expected {} clusters of dimension {dimension}",
            accumulator.counts.len(),
            accumulator.dimension,
            centroids.len()
        )));
    }
    let current = centroids
        .values()
        .as_primitive::<arrow_array::types::UInt8Type>()
        .values();
    let mut candidates = Vec::with_capacity(centroids.len());
    for cluster_id in 0..centroids.len() {
        if accumulator.count(cluster_id) == 0 {
            candidates.push(current[cluster_id * dimension..(cluster_id + 1) * dimension].to_vec());
        } else {
            candidates.push(accumulator.mode(cluster_id));
        }
    }
    let next = complete_distinct_centroids(
        candidates,
        current
            .chunks_exact(dimension)
            .map(|centroid| centroid.to_vec()),
        centroids.len(),
        dimension,
    )?;
    Ok(FixedSizeListArray::try_new_from_values(
        UInt8Array::from(next),
        dimension as i32,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::u8_max(u8::MAX as u64, CounterWidth::U8)]
    #[case::u16_min(u8::MAX as u64 + 1, CounterWidth::U16)]
    #[case::u16_max(u16::MAX as u64, CounterWidth::U16)]
    #[case::u32_min(u16::MAX as u64 + 1, CounterWidth::U32)]
    #[case::u32_max(u32::MAX as u64, CounterWidth::U32)]
    #[case::u64_min(u32::MAX as u64 + 1, CounterWidth::U64)]
    fn compact_counter_selects_narrowest_exact_width(
        #[case] count: u64,
        #[case] expected_width: CounterWidth,
    ) {
        let mut counters = CompactCounters::with_capacity(1);
        counters.extend_u64(&[count], count).unwrap();
        assert_eq!(counters.width(), expected_width);
        assert_eq!(counters.to_u64_vec(), vec![count]);
    }

    #[test]
    fn compact_counter_promotion_preserves_existing_values() {
        let mut counters = CompactCounters::with_capacity(4);
        counters
            .extend_u64(&[1, u8::MAX as u64], u8::MAX as u64)
            .unwrap();
        counters.ensure_width_for(u8::MAX as u64 + 1);
        counters
            .extend_u64(&[u16::MAX as u64], u16::MAX as u64)
            .unwrap();
        counters.ensure_width_for(u16::MAX as u64 + 1);
        counters
            .extend_u64(&[u32::MAX as u64], u32::MAX as u64)
            .unwrap();
        counters.ensure_width_for(u32::MAX as u64 + 1);

        assert_eq!(counters.width(), CounterWidth::U64);
        assert_eq!(
            counters.to_u64_vec(),
            vec![1, u8::MAX as u64, u16::MAX as u64, u32::MAX as u64]
        );
    }

    #[test]
    fn summary_cost_and_mode_preserve_raw_hamming_statistics() {
        let mut ones = vec![0_u64; 8];
        for value in [0b0000_0001, 0b0000_0011, 0b0000_0010] {
            add_vector_bits(&mut ones, &[value]).unwrap();
        }
        assert_eq!(mode_from_counts(3, &ones, 1), vec![0b0000_0011]);
        assert_eq!(summary_cost(3, &ones, &[0], 1).unwrap(), 4);
        assert_eq!(summary_cost(3, &ones, &[0b11], 1).unwrap(), 2);
    }

    #[test]
    fn summary_cost_matches_bruteforce_for_multiple_candidates() {
        let vectors = [[0b0000_0000], [0b0000_0011], [0b0000_1111], [0b1111_1111]];
        let mut ones = vec![0_u64; 8];
        for vector in vectors {
            add_vector_bits(&mut ones, &vector).unwrap();
        }
        for centroid in [[0_u8], [0b0000_0011], [0b1111_1111]] {
            let expected = vectors
                .iter()
                .map(|vector| (vector[0] ^ centroid[0]).count_ones() as u64)
                .sum::<u64>();
            assert_eq!(summary_cost(4, &ones, &centroid, 1).unwrap(), expected);
        }
    }

    #[test]
    fn hamming_summary_rejects_invalid_dimensions_and_counts() {
        let mut coreset = HammingCoreset::new(2, 2);
        assert!(coreset.push(1, &[1; 8]).is_err());
        assert!(coreset.push(1, &[2; 16]).is_err());
        let mut accumulator = HammingAccumulator::try_new(1, 2).unwrap();
        assert!(accumulator.add_vector(0, &[1]).is_err());
        assert!(accumulator.add_vector(1, &[1, 2]).is_err());
    }

    #[test]
    fn hamming_reduction_is_deterministic_for_equal_bit_fractions() {
        let mut first = HammingCoreset::new(1, 6);
        let mut second = HammingCoreset::new(1, 6);
        for coreset in [&mut first, &mut second] {
            coreset.push(1, &[0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
            coreset.push(1, &[0, 1, 0, 0, 0, 0, 0, 0]).unwrap();
            coreset.push(1, &[0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
            coreset.push(1, &[0, 1, 0, 0, 0, 0, 0, 0]).unwrap();
            coreset.push(1, &[0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
            coreset.push(1, &[1, 1, 0, 0, 0, 0, 0, 0]).unwrap();
        }
        first.reduce_to_budget(2).unwrap();
        second.reduce_to_budget(2).unwrap();
        assert_eq!(first.counts, second.counts);
        assert_eq!(first.ones, second.ones);
        assert_eq!(first.counts, vec![3, 3]);
        assert_eq!(
            first.ones.to_u64_vec(),
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn hamming_empty_and_duplicate_summaries_are_safe() {
        let mut coreset = HammingCoreset::new(1, 3);
        coreset.push(3, &[3, 3, 0, 0, 0, 0, 0, 0]).unwrap();
        coreset.push(3, &[3, 3, 0, 0, 0, 0, 0, 0]).unwrap();
        assert!(coreset.reduce_to_budget(0).is_err());
        assert_eq!(coreset.mode(0), vec![3]);
        assert_eq!(coreset.cost(0, &[3]).unwrap(), 0);
    }

    #[test]
    fn majority_ties_resolve_to_zero() {
        let mut ones = vec![0_u64; 8];
        add_vector_bits(&mut ones, &[0b1010_1010]).unwrap();
        assert_eq!(mode_from_counts(2, &ones, 1), vec![0]);
    }

    #[test]
    fn coreset_merge_is_associative() {
        let mut coreset = HammingCoreset::new(1, 3);
        coreset.push(2, &[1, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        coreset.push(3, &[2, 1, 0, 0, 0, 0, 0, 0]).unwrap();
        coreset.push(1, &[1, 1, 0, 0, 0, 0, 0, 0]).unwrap();
        coreset.reduce_to_budget(1).unwrap();
        assert_eq!(coreset.counts, vec![6]);
        assert_eq!(coreset.ones.width(), CounterWidth::U8);
        assert_eq!(coreset.ones.to_u64_vec(), vec![4, 2, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn coreset_append_promotes_mixed_widths_without_losing_counts() {
        let mut left = HammingCoreset::new(1, 2);
        left.push(2, &[1, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        let mut right = HammingCoreset::new(1, 1);
        right.push(300, &[299, 1, 0, 0, 0, 0, 0, 0]).unwrap();

        left.append(right).unwrap();

        assert_eq!(left.counts, vec![2, 300]);
        assert_eq!(left.ones.width(), CounterWidth::U16);
        assert_eq!(
            left.ones.to_u64_vec(),
            vec![1, 0, 0, 0, 0, 0, 0, 0, 299, 1, 0, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn compact_accumulator_matches_u64_reference_across_promotion() {
        const DIMENSION: usize = 2;
        const CLUSTERS: usize = 3;
        let mut rng = SmallRng::seed_from_u64(0xacca_5510);
        let mut accumulator = HammingAccumulator::try_new(CLUSTERS, DIMENSION).unwrap();
        let mut reference_counts = vec![0_u64; CLUSTERS];
        let mut reference_ones = vec![0_u64; CLUSTERS * DIMENSION * 8];

        for _ in 0..1_000 {
            let cluster_id = rng.random_range(0..CLUSTERS);
            let vector = [rng.random::<u8>(), rng.random::<u8>()];
            accumulator.add_vector(cluster_id, &vector).unwrap();
            reference_counts[cluster_id] += 1;
            add_vector_bits(
                &mut reference_ones[cluster_id * DIMENSION * 8..(cluster_id + 1) * DIMENSION * 8],
                &vector,
            )
            .unwrap();
        }

        assert_eq!(accumulator.ones.width(), CounterWidth::U16);
        assert_eq!(accumulator.counts, reference_counts);
        assert_eq!(accumulator.ones.to_u64_vec(), reference_ones);
        for (cluster_id, count) in reference_counts.into_iter().enumerate() {
            assert_eq!(
                accumulator.mode(cluster_id),
                mode_from_counts(
                    count,
                    &reference_ones[cluster_id * DIMENSION * 8..(cluster_id + 1) * DIMENSION * 8],
                    DIMENSION,
                )
            );
        }
    }

    #[test]
    fn raw_centroid_update_promotes_and_preserves_empty_centroids() {
        let centroids =
            FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![0_u8, 0b0101_0101]), 1)
                .unwrap();
        let mut accumulator = HammingAccumulator::try_new(2, 1).unwrap();
        for _ in 0..256 {
            accumulator.add_vector(0, &[u8::MAX]).unwrap();
        }

        assert_eq!(accumulator.ones.width(), CounterWidth::U16);
        let updated = update_raw_centroids(&centroids, &accumulator).unwrap();
        assert_eq!(
            updated
                .values()
                .as_primitive::<arrow_array::types::UInt8Type>(),
            &UInt8Array::from(vec![u8::MAX, 0b0101_0101])
        );
    }

    #[test]
    fn raw_centroid_update_preserves_distinct_centroids() {
        let centroids =
            FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![0_u8, 1, 2]), 1).unwrap();
        let mut accumulator = HammingAccumulator::try_new(3, 1).unwrap();
        accumulator.add_vector(0, &[0]).unwrap();
        accumulator.add_vector(1, &[0]).unwrap();

        let updated = update_raw_centroids(&centroids, &accumulator).unwrap();
        let values = updated
            .values()
            .as_primitive::<arrow_array::types::UInt8Type>()
            .values();

        assert_eq!(values.iter().copied().collect::<BTreeSet<_>>().len(), 3);
    }

    #[test]
    fn local_coreset_clamps_zero_cluster_count() {
        let data =
            FixedSizeListArray::try_new_from_values(UInt8Array::from(vec![0_u8, u8::MAX]), 1)
                .unwrap();
        let mut coreset = HammingCoreset::new(1, 1);

        append_local_coreset(&mut coreset, &data, 0, 2, std::sync::Arc::new(|_, _| {})).unwrap();

        assert_eq!(coreset.counts, vec![2]);
    }

    #[test]
    fn hierarchical_training_rejects_unsplittable_summaries() {
        let mut coreset = HammingCoreset::new(1, 2);
        coreset.push(1, &[0; 8]).unwrap();
        coreset.push(1, &[0; 8]).unwrap();

        let error = train_hierarchical(&coreset, 2, 2, std::sync::Arc::new(|_, _| {})).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("requires at least 2 distinct training vectors, but sampled only 1")
        );
    }

    #[test]
    fn hierarchical_training_completes_from_distinct_representatives() {
        let mut coreset = HammingCoreset::new(1, 4);
        coreset.push(4, &[2, 2, 0, 0, 0, 0, 0, 0]).unwrap();
        for representative in 1_u8..4 {
            coreset.insert_representative(&[representative]).unwrap();
        }

        let centroids = train_hierarchical(&coreset, 4, 2, std::sync::Arc::new(|_, _| {})).unwrap();
        let values = centroids
            .values()
            .as_primitive::<arrow_array::types::UInt8Type>()
            .values();

        assert_eq!(values, &[0, 1, 2, 3]);
    }

    #[test]
    fn hierarchical_training_keeps_splitting_after_a_high_loss_singleton() {
        let mut coreset = HammingCoreset::new(1, 18);
        coreset.push(1_000_000, &[500_001; 8]).unwrap();
        for value in 0_u8..17 {
            let mut ones = vec![0_u64; 8];
            add_vector_bits(&mut ones, &[value]).unwrap();
            coreset.push(1, &ones).unwrap();
        }

        let centroids =
            train_hierarchical(&coreset, 17, 5, std::sync::Arc::new(|_, _| {})).unwrap();
        let values = centroids
            .values()
            .as_primitive::<arrow_array::types::UInt8Type>()
            .values();

        assert_eq!(values.iter().filter(|&&value| value == u8::MAX).count(), 1);
    }
}
