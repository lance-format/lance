// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building blocks of WARP multivector search.
//!
//! WARP (<https://arxiv.org/abs/2501.17788>) serves the XTR objective from an IVF
//! index. For query tokens `q_1..q_n` and centroids `c_1..c_K` it computes
//! `Γ[k][i] = <c_k, q_i>` once, then reads off both the partitions each token
//! probes and the similarity `m_i` imputed for documents that token never
//! observes ("WARPselect").

use std::cmp::Ordering;

use arrow::compute::cast;
use arrow_array::{Array, FixedSizeListArray, cast::AsArray, types::Float32Type};
use arrow_schema::DataType;
use lance_core::{Error, Result};
use lance_linalg::distance::dot_f32;

/// Default number of partitions WARP probes per query token.
///
/// The paper reports recall saturating between 16 and 32.
pub const DEFAULT_WARP_NPROBE: usize = 32;

/// The walk sorts centroids lazily in chunks that double from this size, since
/// it usually stops after a small prefix of the K centroids.
const INITIAL_WALK_CHUNK: usize = 64;

/// Highest similarity first; ties go to the lower partition id so selection is
/// deterministic.
fn by_similarity_desc(similarities: &[f32]) -> impl Fn(&u32, &u32) -> Ordering + '_ {
    |&a, &b| {
        similarities[b as usize]
            .total_cmp(&similarities[a as usize])
            .then(a.cmp(&b))
    }
}

/// Compute `Γ[k] = <c_k, q>` for one query token against every centroid.
///
/// Centroids of any float type are compared in f32.
///
/// ```
/// # use arrow_array::{FixedSizeListArray, Float32Array};
/// # use lance_arrow::FixedSizeListArrayExt;
/// # use lance_index::vector::warp::centroid_similarities;
/// let centroids =
///     FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0, 0.0, 0.0, 1.0]), 2)
///         .unwrap();
/// let similarities = centroid_similarities(&[0.6, 0.8], &centroids).unwrap();
/// assert_eq!(similarities, vec![0.6, 0.8]);
/// ```
pub fn centroid_similarities(
    query_token: &[f32],
    centroids: &FixedSizeListArray,
) -> Result<Vec<f32>> {
    let dim = centroids.value_length() as usize;
    if query_token.len() != dim {
        return Err(Error::invalid_input(format!(
            "WARP query token has dimension {}, but the IVF centroids have dimension {dim}",
            query_token.len()
        )));
    }
    let values = cast(centroids.values().as_ref(), &DataType::Float32)?;
    let values = values.as_primitive::<Float32Type>().values();
    let start = centroids.offset() * dim;
    Ok(values[start..start + centroids.len() * dim]
        .chunks_exact(dim)
        .map(|centroid| dot_f32(query_token, centroid))
        .collect())
}

/// Select the `nprobe` partitions most similar to a query token (`P_i`), best first.
///
/// Selects every partition when `nprobe` exceeds the partition count, matching
/// how IVF search treats an oversized probe count.
///
/// ```
/// # use lance_index::vector::warp::probe_partitions;
/// assert_eq!(probe_partitions(&[0.1, 0.9, 0.5], 2), vec![1, 2]);
/// ```
pub fn probe_partitions(similarities: &[f32], nprobe: usize) -> Vec<u32> {
    let mut order = (0..similarities.len() as u32).collect::<Vec<_>>();
    let nprobe = nprobe.min(order.len());
    let cmp = by_similarity_desc(similarities);
    if nprobe < order.len() && nprobe > 0 {
        order.select_nth_unstable_by(nprobe - 1, &cmp);
    }
    order.truncate(nprobe);
    order.sort_unstable_by(&cmp);
    order
}

/// Estimate the similarity a query token assigns to tokens it does not observe
/// (`m_i`), by WARPselect.
///
/// Pretending every token sits at its partition's centroid, walk partitions
/// from most to least similar, adding whole partition sizes rather than
/// individual tokens. `m_i` is the similarity of the partition at which the
/// accumulated population first reaches `imputation_rank` (`t'`), i.e. an
/// estimate of the similarity of the `t'`-th most similar token. If the whole
/// index holds fewer than `t'` tokens, it is the least similar partition's
/// similarity.
///
/// `similarities[k]` and `partition_sizes[k]` describe the same partition. To
/// estimate over several index segments, concatenate their partitions.
///
/// ```
/// # use lance_index::vector::warp::missing_similarity;
/// // Walk order: 0.9 (5 tokens), 0.7 (5 tokens) reaches 8 tokens.
/// let m = missing_similarity(&[0.7, 0.9, 0.2], &[5, 5, 100], 8).unwrap();
/// assert_eq!(m, 0.7);
/// ```
pub fn missing_similarity(
    similarities: &[f32],
    partition_sizes: &[u32],
    imputation_rank: usize,
) -> Result<f32> {
    if similarities.len() != partition_sizes.len() {
        return Err(Error::invalid_input(format!(
            "WARPselect got {} centroid similarities but {} partition sizes",
            similarities.len(),
            partition_sizes.len()
        )));
    }
    if similarities.is_empty() {
        return Err(Error::invalid_input(
            "WARPselect needs at least one partition",
        ));
    }
    if imputation_rank == 0 {
        return Err(Error::invalid_input(
            "WARP imputation_rank must be at least 1, got 0",
        ));
    }

    let cmp = by_similarity_desc(similarities);
    let mut order = (0..similarities.len() as u32).collect::<Vec<_>>();
    let mut population = 0_u64;
    let mut walked = 0;
    while walked < order.len() {
        // Everything in `order[walked..]` is no more similar than what was
        // already walked, so the next chunk only needs a partial selection.
        let end = (walked * 2).max(INITIAL_WALK_CHUNK).min(order.len());
        let rest = &mut order[walked..];
        let chunk_len = end - walked;
        if chunk_len < rest.len() {
            rest.select_nth_unstable_by(chunk_len - 1, &cmp);
        }
        rest[..chunk_len].sort_unstable_by(&cmp);
        for &partition in &rest[..chunk_len] {
            population += partition_sizes[partition as usize] as u64;
            if population >= imputation_rank as u64 {
                return Ok(similarities[partition as usize]);
            }
        }
        walked = end;
    }
    Ok(similarities[*order.last().expect("non-empty") as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float32Array;
    use lance_arrow::FixedSizeListArrayExt;
    use rstest::rstest;

    #[rstest]
    #[case::fewer_than_all(vec![0.1, 0.9, 0.5, 0.7], 2, vec![1, 3])]
    #[case::all(vec![0.1, 0.9, 0.5], 3, vec![1, 2, 0])]
    #[case::oversized(vec![0.1, 0.9], 5, vec![1, 0])]
    #[case::zero(vec![0.1, 0.9], 0, vec![])]
    #[case::ties_prefer_lower_id(vec![0.5, 0.5, 0.5], 2, vec![0, 1])]
    fn test_probe_partitions(
        #[case] similarities: Vec<f32>,
        #[case] nprobe: usize,
        #[case] expected: Vec<u32>,
    ) {
        assert_eq!(probe_partitions(&similarities, nprobe), expected);
    }

    #[rstest]
    // Walk: 0.9 (4) -> 0.7 (3) -> 0.5 (10) -> 0.1 (2).
    #[case::inside_first(1, 0.9)]
    #[case::exactly_first(4, 0.9)]
    #[case::skips_into_second(5, 0.7)]
    #[case::exactly_second(7, 0.7)]
    #[case::inside_third(8, 0.5)]
    #[case::last(18, 0.1)]
    #[case::exceeds_total(1000, 0.1)]
    fn test_missing_similarity_walks_partition_chunks(
        #[case] imputation_rank: usize,
        #[case] expected: f32,
    ) {
        let similarities = [0.5, 0.1, 0.9, 0.7];
        let sizes = [10, 2, 4, 3];
        assert_eq!(
            missing_similarity(&similarities, &sizes, imputation_rank).unwrap(),
            expected
        );
    }

    #[test]
    fn test_missing_similarity_skips_empty_partitions() {
        let similarities = [0.9, 0.8, 0.3];
        let sizes = [0, 0, 1];
        assert_eq!(missing_similarity(&similarities, &sizes, 1).unwrap(), 0.3);
    }

    /// Spans several lazily sorted chunks and compares against a full sort.
    #[test]
    fn test_missing_similarity_matches_full_walk() {
        let num_partitions = 1000;
        let similarities = (0..num_partitions)
            .map(|k| ((k * 7919) % num_partitions) as f32 / num_partitions as f32)
            .collect::<Vec<_>>();
        let sizes = (0..num_partitions)
            .map(|k| (k % 5) as u32)
            .collect::<Vec<_>>();

        let mut order = (0..num_partitions as u32).collect::<Vec<_>>();
        order.sort_by(by_similarity_desc(&similarities));
        for imputation_rank in [1, 63, 64, 65, 500, 1500, 1999, 2000, 5000] {
            let mut population = 0;
            let expected = order
                .iter()
                .find(|&&k| {
                    population += sizes[k as usize] as usize;
                    population >= imputation_rank
                })
                .map(|&k| similarities[k as usize])
                .unwrap_or(similarities[*order.last().unwrap() as usize]);
            assert_eq!(
                missing_similarity(&similarities, &sizes, imputation_rank).unwrap(),
                expected,
                "imputation_rank={imputation_rank}"
            );
        }
    }

    #[rstest]
    #[case::length_mismatch(vec![0.1, 0.2], vec![1], 1)]
    #[case::empty(vec![], vec![], 1)]
    #[case::zero_rank(vec![0.1], vec![1], 0)]
    fn test_missing_similarity_rejects_invalid_input(
        #[case] similarities: Vec<f32>,
        #[case] sizes: Vec<u32>,
        #[case] imputation_rank: usize,
    ) {
        let err = missing_similarity(&similarities, &sizes, imputation_rank).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }

    #[test]
    fn test_centroid_similarities_sliced_and_float64() {
        let centroids = FixedSizeListArray::try_new_from_values(
            arrow_array::Float64Array::from(vec![9.0, 9.0, 1.0, 2.0, 3.0, 4.0]),
            2,
        )
        .unwrap()
        .slice(1, 2);
        assert_eq!(
            centroid_similarities(&[1.0, 1.0], &centroids).unwrap(),
            vec![3.0, 7.0]
        );

        let centroids =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![1.0, 2.0]), 2).unwrap();
        let err = centroid_similarities(&[1.0], &centroids).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }
}
