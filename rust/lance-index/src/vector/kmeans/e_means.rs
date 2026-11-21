// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Native e-means: <https://arxiv.org/abs/2611.09501>, Algorithm 1.
//! The fitted recipes and initial mass follow FAISS 1.16.0's reference
//! implementation. Accumulation uses f64 to avoid overflowing FP32 batch sums.

use arrow_array::{Array, FixedSizeListArray, Float32Array, cast::AsArray, types::Float32Type};
use arrow_schema::{ArrowError, DataType};
use lance_arrow::FixedSizeListArrayExt;
use rand::prelude::*;

use super::{KMeanInit, KMeans, KMeansAlgo, KMeansAlgoFloat, KMeansParams};
use lance_linalg::distance::DistanceType;

/// Mini-batch clustering with exponential moving sums and counts.
///
/// Supports L2 over floating-point vectors; normalize cosine inputs first.
/// Coordinates are trained in FP32 and returned in the input type. Input and
/// output must remain finite in these representations. The sample is resident
/// in memory; assignment scratch is bounded independently of the batch size.
///
/// Batch size and initial step are derived from the FAISS 1.16.0 scaling laws
/// unless overridden. Empty clusters are revived every five updates (disabled
/// at dimension <= 2). Final centroids are the unweighted means of assignments
/// made during the last pass, without an additional assignment pass.
///
/// ```
/// use lance_index::vector::kmeans::{EMeansParams, KMeansParams};
/// let params = KMeansParams::default()
///     .with_e_means(EMeansParams { epochs: 2.5, ..Default::default() })
///     .with_seed(42);
/// ```
#[derive(Debug, Clone)]
pub struct EMeansParams {
    /// Passes over the sample, including a possible fractional pass. Default 5.
    /// Replaces `KMeansParams::max_iters` and its convergence tolerance.
    pub epochs: f64,
    /// Rows per update. `None` selects the quality law at dimension <= 16,
    /// and the throughput law otherwise. A supplied value must be positive.
    pub batch_size: Option<usize>,
    /// Initial EMA step in (0, 1]. `None` uses the fitted step law. The step
    /// decays on a cosine schedule towards one tenth of its initial value.
    pub alpha0: Option<f64>,
}

impl Default for EMeansParams {
    fn default() -> Self {
        Self {
            epochs: 5.0,
            batch_size: None,
            alpha0: None,
        }
    }
}

fn invalid(message: impl Into<String>) -> ArrowError {
    ArrowError::InvalidArgumentError(message.into())
}

fn round_power_of_two(value: f64) -> usize {
    if value <= 1.0 {
        return 1;
    }
    2.0_f64.powf(value.log2().round()) as usize
}

impl EMeansParams {
    /// Validate the epoch budget and explicit overrides before reading data.
    pub fn validate(&self) -> Result<(), ArrowError> {
        if !self.epochs.is_finite() || self.epochs <= 0.0 || self.epochs > u32::MAX as f64 {
            return Err(invalid(format!(
                "e-means epochs must be finite and in (0, {}], got {}",
                u32::MAX,
                self.epochs
            )));
        }
        if self.batch_size == Some(0) {
            return Err(invalid("e-means batch_size must be greater than zero"));
        }
        if self
            .alpha0
            .is_some_and(|a| !a.is_finite() || a <= 0.0 || a > 1.0)
        {
            return Err(invalid(format!(
                "e-means alpha0 must be finite and in (0, 1], got {:?}",
                self.alpha0
            )));
        }
        Ok(())
    }

    fn recipe(&self, n: usize, k: usize, d: usize) -> (usize, f64) {
        let batch = self
            .batch_size
            .unwrap_or_else(|| {
                let coefficient = if d <= 16 { 0.00439175 } else { 0.035134 };
                let raw = coefficient
                    * (k as f64).powf(0.134749)
                    * (n as f64).powf(0.876473)
                    * self.epochs.powf(0.813344);
                let cap = if d <= 16 {
                    n / 2
                } else {
                    round_power_of_two(n as f64 / 2.0)
                };
                round_power_of_two(raw).min(cap).max(1)
            })
            .min(n);
        let alpha = self.alpha0.unwrap_or_else(|| {
            (0.8244
                * (k as f64).powf(-0.4617)
                * (batch as f64).powf(0.9003)
                * (n as f64).powf(-0.4498)
                * self.epochs.powf(-0.1057))
            .clamp(1e-4, 1.0)
        });
        (batch, alpha)
    }
}

/// Weighted donor sampling without replacement, followed by symmetric splits.
fn revive(centroids: &mut [f32], mass: &mut [f64], d: usize, rng: &mut SmallRng) {
    if d <= 2 {
        return;
    }
    let threshold = 0.1 * mass.iter().sum::<f64>() / mass.len() as f64;
    let unused: Vec<_> = (0..mass.len()).filter(|&i| mass[i] < threshold).collect();
    if unused.is_empty() {
        return;
    }
    let mut donors: Vec<_> = (0..mass.len())
        .filter(|&i| mass[i] >= threshold && mass[i] > 0.0)
        .map(|i| {
            let u = rng.random::<f64>().max(f64::MIN_POSITIVE);
            (i, mass[i].ln() - (-u.ln()).ln())
        })
        .collect();
    donors.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    for (child, (donor, _)) in unused.into_iter().zip(donors) {
        mass[donor] *= 0.5;
        mass[child] = mass[donor];
        let mut identical = true;
        for j in 0..d {
            let c = centroids[donor * d + j];
            let offset = if j % 2 == 0 { 1e-4 } else { -1e-4 };
            // Saturate a perturbation at the representable endpoint. Means
            // stay in the input's convex hull, but donor perturbations may not.
            let left =
                (c as f64 * (1.0 - offset)).clamp(-(f32::MAX as f64), f32::MAX as f64) as f32;
            let right =
                (c as f64 * (1.0 + offset)).clamp(-(f32::MAX as f64), f32::MAX as f64) as f32;
            centroids[donor * d + j] = left;
            centroids[child * d + j] = right;
            identical &= left == right;
        }
        if identical {
            let value = centroids[child * d];
            centroids[child * d] = if value.next_up().is_finite() {
                value.next_up()
            } else {
                value.next_down()
            };
        }
    }
}

fn update_centroids(
    centroids: &mut [f32],
    mass: &mut [f64],
    sums: &[f64],
    counts: &[usize],
    d: usize,
    alpha: f64,
) {
    for i in 0..counts.len() {
        // At alpha=1 retain mass for untouched clusters, as in the reference.
        if alpha != 1.0 {
            mass[i] *= 1.0 - alpha;
        }
        if counts[i] == 0 {
            continue;
        }
        let retained = if alpha == 1.0 { 0.0 } else { mass[i] };
        mass[i] = retained + alpha * counts[i] as f64;
        let weight = alpha * counts[i] as f64 / mass[i];
        for j in 0..d {
            let mean = sums[i * d + j] / counts[i] as f64;
            let old = centroids[i * d + j] as f64;
            centroids[i * d + j] = (old + weight * (mean - old)) as f32;
        }
    }
}

pub(super) fn train(
    data: &FixedSizeListArray,
    k: usize,
    params: &KMeansParams,
    options: &EMeansParams,
) -> Result<KMeans, ArrowError> {
    options.validate()?;
    let n = data.len();
    let d = data.value_length() as usize;
    if d == 0 || k == 0 || k > n || k > u32::MAX as usize {
        return Err(invalid(format!(
            "e-means requires dimension > 0 and 0 < k <= min(n, u32::MAX), got d={d}, k={k}, n={n}"
        )));
    }
    if params.distance_type != DistanceType::L2 || params.balance_factor != 0.0 || params.redos == 0
    {
        return Err(invalid(
            "e-means requires L2, balance_factor=0 and redos>0; normalize cosine input before training",
        ));
    }
    if !matches!(
        data.value_type(),
        DataType::Float16 | DataType::Float32 | DataType::Float64
    ) || data.null_count() > 0
        || data.values().null_count() > 0
    {
        return Err(invalid("e-means requires non-null floating-point vectors"));
    }
    let values = arrow::compute::cast(data.values(), &DataType::Float32)?;
    let values = values.as_primitive::<Float32Type>();
    if values.values().iter().any(|x| !x.is_finite()) {
        return Err(invalid("e-means training vectors must be finite in FP32"));
    }
    let (batch_size, alpha0) = options.recipe(n, k, d);
    let whole = options.epochs.floor() as usize;
    let remainder = ((options.epochs - whole as f64) * n as f64).round_ties_even() as usize;
    // Keep the final pass complete whenever the budget is at least one pass.
    let mut epoch_rows = vec![n; whole.saturating_sub(1)];
    if remainder > 0 {
        epoch_rows.push(remainder);
    }
    if whole > 0 {
        epoch_rows.push(n);
    }
    if epoch_rows.is_empty() {
        return Err(invalid("e-means epoch budget rounds to zero training rows"));
    }
    let total_steps = epoch_rows.iter().try_fold(0usize, |steps, &rows| {
        steps
            .checked_add(rows.div_ceil(batch_size))
            .ok_or_else(|| invalid("e-means update count overflows usize"))
    })?;
    let mut rng = params.rng();
    let mut best: Option<KMeans> = None;
    for _ in 0..params.redos {
        let mut centroids = match &params.init {
            KMeanInit::Random => (0..n)
                .choose_multiple(&mut rng, k)
                .into_iter()
                .flat_map(|i| values.values()[i * d..(i + 1) * d].iter().copied())
                .collect::<Vec<_>>(),
            KMeanInit::Incremental(initial) => {
                if initial.len() != k
                    || initial.value_length() as usize != d
                    || initial.null_count() > 0
                    || initial.values().null_count() > 0
                {
                    return Err(invalid(format!(
                        "e-means initial centroids must have shape {k}x{d} without nulls"
                    )));
                }
                let initial = arrow::compute::cast(initial.values(), &DataType::Float32)?;
                let initial = initial.as_primitive::<Float32Type>();
                if initial.values().iter().any(|x| !x.is_finite()) {
                    return Err(invalid("e-means initial centroids must be finite in FP32"));
                }
                initial.values().to_vec()
            }
        };
        let mut mass = vec![2.0_f64.powi(-16); k];
        let mut order: Vec<_> = (0..n).collect();
        let mut sums = vec![0.0_f64; k * d];
        let mut counts = vec![0usize; k];
        let mut closing_sums = vec![0.0_f64; k * d];
        let mut closing_counts = vec![0usize; k];
        let mut chunk = Vec::with_capacity(batch_size.min(4096) * d);
        let mut step = 0;
        for (epoch, &rows) in epoch_rows.iter().enumerate() {
            order.shuffle(&mut rng);
            let closing = epoch + 1 == epoch_rows.len();
            for batch in order[..rows].chunks(batch_size) {
                sums.fill(0.0);
                counts.fill(0);
                for indices in batch.chunks(4096) {
                    chunk.clear();
                    for &i in indices {
                        chunk.extend_from_slice(&values.values()[i * d..(i + 1) * d]);
                    }
                    let (assignments, _) =
                        KMeansAlgoFloat::<Float32Type>::compute_membership_and_dist(
                            &centroids,
                            &chunk,
                            d,
                            DistanceType::L2,
                            0.0,
                            None,
                            None,
                        );
                    for (vector, assignment) in chunk.chunks_exact(d).zip(assignments) {
                        let cluster = assignment.ok_or_else(|| {
                            invalid(
                                "e-means could not assign a finite vector; distances overflowed",
                            )
                        })? as usize;
                        counts[cluster] += 1;
                        for j in 0..d {
                            sums[cluster * d + j] += vector[j] as f64;
                        }
                    }
                }
                let alpha = alpha0
                    * (0.1
                        + 0.9
                            * 0.5
                            * (1.0
                                + (std::f64::consts::PI * step as f64 / total_steps as f64).cos()));
                update_centroids(&mut centroids, &mut mass, &sums, &counts, d, alpha);
                if closing {
                    for i in 0..k {
                        for j in 0..d {
                            closing_sums[i * d + j] += sums[i * d + j];
                        }
                        closing_counts[i] += counts[i];
                    }
                }
                step += 1;
                if step % 5 == 0 {
                    revive(&mut centroids, &mut mass, d, &mut rng);
                }
            }
            if let Some(callback) = &params.on_progress {
                callback(epoch as u32 + 1, epoch_rows.len() as u32);
            }
        }
        for i in 0..k {
            if closing_counts[i] > 0 {
                for j in 0..d {
                    centroids[i * d + j] =
                        (closing_sums[i * d + j] / closing_counts[i] as f64) as f32;
                }
            }
        }
        let centroids = arrow::compute::cast(&Float32Array::from(centroids), &data.value_type())?;
        let centroid_rows = FixedSizeListArray::try_new_from_values(centroids.clone(), d as i32)?;
        if crate::vector::utils::is_finite(&centroid_rows).true_count() != k {
            return Err(invalid(
                "e-means centroids are not finite in the output type",
            ));
        }
        let mut model = KMeans {
            centroids,
            dimension: d,
            distance_type: DistanceType::L2,
            loss: 0.0,
        };
        model.loss = model.compute_loss(data)?;
        if !model.loss.is_finite() {
            return Err(invalid("e-means final loss is not finite"));
        }
        if best.as_ref().is_none_or(|best| model.loss < best.loss) {
            best = Some(model);
        }
    }
    best.ok_or_else(|| invalid("e-means requires at least one redo"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float16Array;
    use half::f16;
    use rstest::rstest;
    use std::sync::{Arc, Mutex};

    #[rstest]
    #[case(100_000, 256, 128, 5.0, 8192, 1.0)]
    #[case(100_000, 256, 8, 5.0, 1024, 0.15543381717430582)]
    #[case(1024, 16, 8, 2.5, 8, 0.059862179242384884)]
    fn recipe_matches_faiss_1_16(
        #[case] n: usize,
        #[case] k: usize,
        #[case] d: usize,
        #[case] epochs: f64,
        #[case] batch: usize,
        #[case] alpha: f64,
    ) {
        let actual = EMeansParams {
            epochs,
            ..Default::default()
        }
        .recipe(n, k, d);
        assert_eq!(actual.0, batch);
        assert!((actual.1 - alpha).abs() < 1e-12);
    }

    #[test]
    fn updates_match_faiss_1_16() {
        // Captured from NumpyEmeans(1, 2, nepochs=2, batch_size=3,
        // alpha0=.5).begin(6, [[0], [10]]) followed by batches
        // [[1], [3], [12]] and [[2], [8], [14]]. Compare both state variables;
        // averaging batch centroids instead of sums/counts fails this fixture.
        let mut centroids = vec![0.0, 10.0];
        let mut mass = vec![2.0_f64.powi(-16); 2];
        update_centroids(&mut centroids, &mut mass, &[4.0, 12.0], &[2, 1], 1, 0.5);
        assert!((centroids[0] - 1.9999847).abs() < 1e-6);
        assert!((centroids[1] - 11.999969).abs() < 1e-6);
        let alpha = 0.5 * (0.1 + 0.9 * 0.5 * (1.0 + (std::f64::consts::PI / 4.0).cos()));
        update_centroids(&mut centroids, &mut mass, &[2.0, 22.0], &[1, 2], 1, alpha);
        assert!((centroids[0] - 1.9999914).abs() < 1e-6);
        assert!((centroids[1] - 11.245793).abs() < 2e-6);
        assert!((mass[0] - 1.0000042915344238).abs() < 1e-7);
        assert!((mass[1] - 1.1511528491973877).abs() < 1e-7);
        update_centroids(&mut centroids, &mut mass, &[0.0, 5.0], &[0, 1], 1, 1.0);
        assert!(mass[0] > 0.0);
        assert_eq!(centroids[1], 5.0);
    }

    #[rstest]
    #[case(1.0, 1)]
    #[case(2.5, 3)]
    fn final_mean_includes_remainder_batch(#[case] epochs: f64, #[case] passes: usize) {
        let data = FixedSizeListArray::try_new_from_values(
            Float32Array::from(vec![0.0, 1.0, 2.0, 3.0, 14.0]),
            1,
        )
        .unwrap();
        let progress = Arc::new(Mutex::new(Vec::new()));
        let captured = progress.clone();
        let params = KMeansParams::default()
            .with_seed(42)
            .with_e_means(EMeansParams {
                epochs,
                batch_size: Some(2),
                ..Default::default()
            })
            .with_on_progress(Arc::new(move |step, total| {
                captured.lock().unwrap().push((step, total))
            }));
        let model = KMeans::new_with_params(&data, 1, &params).unwrap();
        assert_eq!(model.centroids.as_primitive::<Float32Type>().value(0), 4.0);
        assert_eq!(model.loss, 130.0);
        assert_eq!(progress.lock().unwrap().len(), passes);
    }

    #[test]
    fn float16_accumulation_does_not_overflow() {
        let data = FixedSizeListArray::try_new_from_values(
            Float16Array::from(vec![f16::from_f32(1000.0); 256]),
            2,
        )
        .unwrap();
        let params = KMeansParams::default()
            .with_seed(42)
            .with_e_means(EMeansParams {
                batch_size: Some(128),
                ..Default::default()
            });
        let model = KMeans::new_with_params(&data, 1, &params).unwrap();
        assert_eq!(model.centroids.data_type(), &DataType::Float16);
        assert_eq!(model.loss, 0.0);
    }

    #[test]
    fn seeded_training_reduces_loss() {
        let values: Vec<_> = (0..256)
            .flat_map(|i| (0..4).map(move |j| (i % 4) as f32 * 10.0 + ((i + j) % 7) as f32 * 0.01))
            .collect();
        let data = FixedSizeListArray::try_new_from_values(Float32Array::from(values), 4).unwrap();
        let params = KMeansParams::default()
            .with_seed(7)
            .with_e_means(EMeansParams::default());
        let first = KMeans::new_with_params(&data, 4, &params).unwrap();
        let second = KMeans::new_with_params(&data, 4, &params).unwrap();
        assert_eq!(first.centroids.as_ref(), second.centroids.as_ref());
        assert!(first.loss < 1.0, "{}", first.loss);
    }

    #[test]
    fn revival_conserves_mass_and_separates_zero_donors() {
        let mut centroids = vec![0.0; 9];
        let mut mass = vec![8.0, 0.0, 0.0];
        revive(
            &mut centroids,
            &mut mass,
            3,
            &mut SmallRng::seed_from_u64(42),
        );
        assert_eq!(mass.iter().sum::<f64>(), 8.0);
        assert_eq!(mass, vec![4.0, 4.0, 0.0]);
        assert_ne!(centroids[0], centroids[3]);
    }

    #[rstest]
    #[case(EMeansParams { epochs: f64::NAN, ..Default::default() }, "epochs")]
    #[case(EMeansParams { epochs: 0.0, ..Default::default() }, "epochs")]
    #[case(EMeansParams { batch_size: Some(0), ..Default::default() }, "batch_size")]
    #[case(EMeansParams { alpha0: Some(1.1), ..Default::default() }, "alpha0")]
    #[case(EMeansParams { epochs: 0.01, ..Default::default() }, "zero training rows")]
    fn invalid_parameters(#[case] options: EMeansParams, #[case] message: &str) {
        let data =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0, 2.0]), 1).unwrap();
        let error =
            KMeans::new_with_params(&data, 1, &KMeansParams::default().with_e_means(options))
                .unwrap_err();
        assert!(matches!(error, ArrowError::InvalidArgumentError(_)));
        assert!(error.to_string().contains(message));
    }
}
