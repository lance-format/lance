// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building blocks of WARP multivector search.
//!
//! WARP (<https://arxiv.org/abs/2501.17788>) serves the XTR objective from an IVF
//! index. For query tokens `q_1..q_n` and centroids `c_1..c_K` it computes
//! `Γ[k][i] = <c_k, q_i>` once, then reads off both the partitions each token
//! probes and the similarity `m_i` imputed for documents that token never
//! observes ("WARPselect"). After the probed partitions are scored into
//! [`Stride`]s, [`merge_strides`] reduces each query token's strides and
//! [`top_k_documents`] sums the tokens into document scores.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{Array, FixedSizeListArray, cast::AsArray, types::Float32Type};
use arrow_schema::DataType;
use lance_core::{Error, Result};
use lance_linalg::distance::dot_f32;

/// Default number of partitions WARP probes per query token.
///
/// The paper reports recall saturating between 16 and 32.
pub const DEFAULT_WARP_NPROBE: usize = 32;

/// Up to this many partitions, the default `t'` grows with the corpus.
const MAX_CORPUS_SCALED_PARTITIONS: usize = 1 << 16;

/// The default `t'` for an index of `num_tokens` token embeddings in
/// `num_partitions` partitions, searched for the top `k` documents.
///
/// This follows the [reference implementation]: up to 65,536 partitions, `t'`
/// is `floor(sqrt(8 * num_tokens) / 1000) * 1000`, as the paper recommends
/// scaling it with the square root of the corpus size. Beyond that it is
/// 100,000 when `k` exceeds 100 and 50,000 otherwise. Below 125,000 tokens it
/// is 0, so [`missing_similarity`] imputes a similarity of 0.
///
/// [reference implementation]: https://github.com/jlscheerer/xtr-warp/blob/cca97613e6f969ac89f259946b976f8c5a6f1399/warp/engine/search/index_storage.py#L103-L109
///
/// ```
/// # use lance_index::vector::warp::default_imputation_rank;
/// assert_eq!(default_imputation_rank(2_000_000, 4096, 10), 4000);
/// assert_eq!(default_imputation_rank(100_000, 256, 10), 0);
/// assert_eq!(default_imputation_rank(2_000_000, 1 << 17, 1000), 100_000);
/// ```
pub fn default_imputation_rank(num_tokens: u64, num_partitions: usize, k: usize) -> usize {
    if num_partitions <= MAX_CORPUS_SCALED_PARTITIONS {
        ((8.0 * num_tokens as f64).sqrt() / 1000.0) as usize * 1000
    } else if k > 100 {
        100_000
    } else {
        50_000
    }
}

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
/// similarity. A `t'` of 0 imputes 0, as the reference implementation does.
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
        return Ok(0.0);
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

/// Scored rows of one query token `q_i`: either one probed partition's tokens
/// (a stride), or all of a token's strides merged by [`merge_strides`] (`f_i`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stride {
    /// Rows with at least one scored token, strictly ascending. Shared by the
    /// strides of query tokens that probe the same partition.
    pub row_ids: Arc<[u64]>,
    /// Per row, the highest `<t̂, q_i>` over its scored tokens.
    pub similarities: Vec<f32>,
}

impl Stride {
    fn check(&self) -> Result<()> {
        if self.row_ids.len() != self.similarities.len() {
            return Err(Error::invalid_input(format!(
                "WARP stride has {} row ids but {} similarities",
                self.row_ids.len(),
                self.similarities.len()
            )));
        }
        Ok(())
    }

    /// The positions of the rows within `rows`.
    fn positions(&self, rows: &impl RangeBounds<u64>) -> std::ops::Range<usize> {
        let start = match rows.start_bound() {
            Bound::Included(&start) => self.row_ids.partition_point(|&row| row < start),
            Bound::Excluded(&start) => self.row_ids.partition_point(|&row| row <= start),
            Bound::Unbounded => 0,
        };
        let end = match rows.end_bound() {
            Bound::Included(&end) => self.row_ids.partition_point(|&row| row <= end),
            Bound::Excluded(&end) => self.row_ids.partition_point(|&row| row < end),
            Bound::Unbounded => self.row_ids.len(),
        };
        start..end.max(start)
    }
}

/// Merge one query token's strides into `f_i`: each row's highest similarity
/// over all the strides that contain it.
///
/// ```
/// # use lance_index::vector::warp::{Stride, merge_strides};
/// let stride = |row_ids: Vec<u64>, similarities: Vec<f32>| Stride {
///     row_ids: row_ids.into(),
///     similarities,
/// };
/// let merged = merge_strides(&[stride(vec![1, 2], vec![0.5, 0.25]), stride(vec![2], vec![0.75])]);
/// assert_eq!(merged.unwrap(), stride(vec![1, 2], vec![0.5, 0.75]));
/// ```
pub fn merge_strides(strides: &[Stride]) -> Result<Stride> {
    for stride in strides {
        stride.check()?;
    }
    let strides = strides
        .iter()
        .filter(|stride| !stride.row_ids.is_empty())
        .collect::<Vec<_>>();
    match strides.as_slice() {
        [] => return Ok(Stride::default()),
        [stride] => return Ok((*stride).clone()),
        _ => {}
    }

    // A balanced tree of linear two-way merges (Section 5.3) is cheaper than
    // a heap over every stride: each level is one sequential pass.
    let mut level = strides
        .chunks(2)
        .map(|pair| match pair {
            [a, b] => merge_two(a, b),
            [a] => (*a).clone(),
            _ => unreachable!("chunks of two"),
        })
        .collect::<Vec<_>>();
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| match pair {
                [a, b] => merge_two(a, b),
                [a] => a.clone(),
                _ => unreachable!("chunks of two"),
            })
            .collect();
    }
    Ok(level.pop().unwrap_or_default())
}

/// Merge two strides, keeping the higher similarity of a row in both.
fn merge_two(a: &Stride, b: &Stride) -> Stride {
    let capacity = a.row_ids.len() + b.row_ids.len();
    let mut row_ids = Vec::with_capacity(capacity);
    let mut similarities = Vec::with_capacity(capacity);
    let (mut i, mut j) = (0, 0);
    while i < a.row_ids.len() && j < b.row_ids.len() {
        match a.row_ids[i].cmp(&b.row_ids[j]) {
            Ordering::Less => {
                row_ids.push(a.row_ids[i]);
                similarities.push(a.similarities[i]);
                i += 1;
            }
            Ordering::Greater => {
                row_ids.push(b.row_ids[j]);
                similarities.push(b.similarities[j]);
                j += 1;
            }
            Ordering::Equal => {
                row_ids.push(a.row_ids[i]);
                similarities.push(a.similarities[i].max(b.similarities[j]));
                i += 1;
                j += 1;
            }
        }
    }
    row_ids.extend_from_slice(&a.row_ids[i..]);
    similarities.extend_from_slice(&a.similarities[i..]);
    row_ids.extend_from_slice(&b.row_ids[j..]);
    similarities.extend_from_slice(&b.similarities[j..]);
    Stride {
        row_ids: row_ids.into(),
        similarities,
    }
}

/// Split the row id space into at most `max_ranges` disjoint ranges holding
/// similar numbers of the `token_scores` rows, so [`top_k_documents`] can run
/// on each range in parallel. The ranges cover every row id, in order.
pub fn split_row_ids(token_scores: &[Stride], max_ranges: usize) -> Vec<(Bound<u64>, Bound<u64>)> {
    // Quantiles of a sample are as good as exact ones for balancing work.
    const SAMPLES_PER_RANGE: usize = 32;
    let num_rows = token_scores
        .iter()
        .map(|scores| scores.row_ids.len())
        .sum::<usize>();
    let num_ranges = max_ranges.clamp(1, num_rows.max(1));
    let step = (num_rows / (num_ranges * SAMPLES_PER_RANGE)).max(1);
    let mut sample = token_scores
        .iter()
        .flat_map(|scores| scores.row_ids.iter().step_by(step).copied())
        .collect::<Vec<_>>();
    sample.sort_unstable();
    let mut boundaries = (1..num_ranges)
        .map(|range| sample[range * sample.len() / num_ranges])
        .collect::<Vec<_>>();
    boundaries.dedup();

    let mut start = Bound::Unbounded;
    let mut ranges = Vec::with_capacity(boundaries.len() + 1);
    for boundary in boundaries {
        ranges.push((start, Bound::Excluded(boundary)));
        start = Bound::Included(boundary);
    }
    ranges.push((start, Bound::Unbounded));
    ranges
}

/// A scored document; greater means ranked higher.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    score: f32,
    row_id: u64,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Ties go to the lower row id so the top k is deterministic.
        self.score
            .total_cmp(&other.score)
            .then(other.row_id.cmp(&self.row_id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

/// The `k` best documents with row ids in `rows`, best first, as
/// `(row_id, score)`.
///
/// `token_scores[i]` is query token `i`'s merged scores `f_i` (see
/// [`merge_strides`]) and `missing_similarities[i]` its `m_i`. A document `D`
/// some token observes scores `Σ_i f_i(D)`, with `f_i(D) = m_i` when token `i`
/// does not observe it. The token lists are merged by row id, so no
/// candidate × token score matrix is built, and each score adds its terms in
/// ascending token order. Section 5.3's `Σ_i m_i + Σ_{i observes D} (f_i(D) - m_i)`
/// would cancel in floating point: a document every token observes would not
/// score exactly the sum of its similarities, and a zero lower distance bound
/// could reject an exact match. Documents no token observes are never candidates, and
/// neither are documents whose score fails `is_eligible`: a distance range
/// must be applied here, before selection, or the top k can crowd out every
/// document in range.
///
/// ```
/// # use lance_index::vector::warp::{Stride, top_k_documents};
/// let stride = |row_ids: Vec<u64>, similarities: Vec<f32>| Stride {
///     row_ids: row_ids.into(),
///     similarities,
/// };
/// // Token 0 observes rows 1 and 2, token 1 only row 2.
/// let token_scores = [stride(vec![1, 2], vec![0.5, 0.75]), stride(vec![2], vec![0.5])];
/// // Row 1 misses token 1, which imputes m_1 = 0.125.
/// let top = top_k_documents(&token_scores, &[0.0, 0.125], .., |_| true, 2).unwrap();
/// assert_eq!(top, vec![(2, 1.25), (1, 0.625)]);
/// ```
pub fn top_k_documents(
    token_scores: &[Stride],
    missing_similarities: &[f32],
    rows: impl RangeBounds<u64>,
    is_eligible: impl Fn(f32) -> bool,
    k: usize,
) -> Result<Vec<(u64, f32)>> {
    if token_scores.len() != missing_similarities.len() {
        return Err(Error::invalid_input(format!(
            "WARP got scores for {} query tokens but {} missing similarities",
            token_scores.len(),
            missing_similarities.len()
        )));
    }
    for scores in token_scores {
        scores.check()?;
    }
    if k == 0 {
        return Ok(Vec::new());
    }

    let positions = token_scores
        .iter()
        .map(|scores| scores.positions(&rows))
        .collect::<Vec<_>>();
    let mut cursors = positions
        .iter()
        .map(|range| range.start)
        .collect::<Vec<_>>();
    // Min-heap of each token's next row id; equal rows pop in token order.
    let mut frontier = positions
        .iter()
        .enumerate()
        .filter(|(_, range)| !range.is_empty())
        .map(|(token, range)| Reverse((token_scores[token].row_ids[range.start], token)))
        .collect::<BinaryHeap<_>>();
    // Min-heap holding the best k candidates seen so far.
    let mut top = BinaryHeap::with_capacity(k + 1);

    while let Some(&Reverse((row_id, _))) = frontier.peek() {
        // Tokens pop in ascending order, so the tokens skipped since the last
        // observed one are those that do not observe this row.
        let mut score = 0.0_f32;
        let mut next_token = 0;
        while let Some(&Reverse((next_row_id, token))) = frontier.peek() {
            if next_row_id != row_id {
                break;
            }
            frontier.pop();
            score = missing_similarities[next_token..token]
                .iter()
                .fold(score, |score, missing| score + missing);
            score += token_scores[token].similarities[cursors[token]];
            next_token = token + 1;
            cursors[token] += 1;
            if cursors[token] < positions[token].end {
                frontier.push(Reverse((
                    token_scores[token].row_ids[cursors[token]],
                    token,
                )));
            }
        }
        score = missing_similarities[next_token..]
            .iter()
            .fold(score, |score, missing| score + missing);
        if !is_eligible(score) {
            continue;
        }
        top.push(Reverse(Candidate { score, row_id }));
        if top.len() > k {
            top.pop();
        }
    }

    Ok(top
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse(candidate)| (candidate.row_id, candidate.score))
        .collect())
}

/// Combine the [`top_k_documents`] of disjoint row ranges into the overall
/// `k` best, best first, with the same tie-breaking.
pub fn merge_top_k(tops: impl IntoIterator<Item = Vec<(u64, f32)>>, k: usize) -> Vec<(u64, f32)> {
    let mut merged = tops.into_iter().flatten().collect::<Vec<_>>();
    merged.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    merged.truncate(k);
    merged
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
    #[case::zero(0, 0.0)]
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

    #[rstest]
    #[case::below_first_step(124_999, 256, 10, 0)]
    #[case::first_step(125_000, 256, 10, 1000)]
    #[case::truncates(2_000_000, 4096, 10, 4000)]
    #[case::last_scaled_partition_count(2_000_000, 1 << 16, 10, 4000)]
    #[case::fixed_small_k(2_000_000, (1 << 16) + 1, 100, 50_000)]
    #[case::fixed_large_k(2_000_000, (1 << 16) + 1, 101, 100_000)]
    fn test_default_imputation_rank(
        #[case] num_tokens: u64,
        #[case] num_partitions: usize,
        #[case] k: usize,
        #[case] expected: usize,
    ) {
        assert_eq!(
            default_imputation_rank(num_tokens, num_partitions, k),
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

    fn stride(row_ids: Vec<u64>, similarities: Vec<f32>) -> Stride {
        Stride {
            row_ids: row_ids.into(),
            similarities,
        }
    }

    /// Scores every observed document by the definition and sorts them all.
    fn brute_force_top_k(
        strides: &[Vec<Stride>],
        missing_similarities: &[f32],
        k: usize,
    ) -> Vec<(u64, f32)> {
        let mut rows = strides
            .iter()
            .flatten()
            .flat_map(|s| s.row_ids.iter().copied())
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows.dedup();
        let mut scored = rows
            .into_iter()
            .map(|row_id| {
                let score = strides
                    .iter()
                    .zip(missing_similarities)
                    .map(|(token_strides, &missing)| {
                        token_strides
                            .iter()
                            .filter_map(|s| {
                                s.row_ids
                                    .binary_search(&row_id)
                                    .ok()
                                    .map(|position| s.similarities[position])
                            })
                            .reduce(f32::max)
                            .unwrap_or(missing)
                    })
                    .sum::<f32>();
                (row_id, score)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    /// Four query tokens with three strides each over 40 rows; integer-valued
    /// similarities keep every summation order exact.
    fn sample_strides() -> Vec<Vec<Stride>> {
        (0..4_u64)
            .map(|token| {
                (0..3)
                    .map(|partition| {
                        let row_ids = (0..40_u64)
                            .filter(|row| (row * 7 + token * 3 + partition) % 5 == 0)
                            .collect::<Vec<_>>();
                        let similarities = row_ids
                            .iter()
                            .map(|row| ((row * 13 + token * 5 + partition) % 11) as f32)
                            .collect();
                        stride(row_ids, similarities)
                    })
                    .collect()
            })
            .collect()
    }

    #[rstest]
    #[case::fewer_than_k(10)]
    #[case::exactly_k(30)]
    #[case::truncated(5)]
    #[case::one(1)]
    fn test_top_k_documents_matches_brute_force(#[case] k: usize) {
        let strides = sample_strides();
        let missing = [1.0, 2.0, 0.0, 3.0];
        let token_scores = strides
            .iter()
            .map(|token_strides| merge_strides(token_strides).unwrap())
            .collect::<Vec<_>>();
        let expected = brute_force_top_k(&strides, &missing, k);
        assert_eq!(
            top_k_documents(&token_scores, &missing, .., |_| true, k).unwrap(),
            expected
        );
        // Splitting by row range and merging the parts gives the same top k.
        for max_ranges in [1, 2, 3, 7, 100] {
            let tops = split_row_ids(&token_scores, max_ranges)
                .into_iter()
                .map(|rows| top_k_documents(&token_scores, &missing, rows, |_| true, k).unwrap());
            assert_eq!(merge_top_k(tops, k), expected, "max_ranges={max_ranges}");
        }
    }

    #[rstest]
    #[case::one_range(1)]
    #[case::several(4)]
    #[case::more_than_rows(1000)]
    fn test_split_row_ids_partitions_rows(#[case] max_ranges: usize) {
        let token_scores = sample_strides()
            .iter()
            .map(|token_strides| merge_strides(token_strides).unwrap())
            .collect::<Vec<_>>();
        let ranges = split_row_ids(&token_scores, max_ranges);
        assert!(!ranges.is_empty() && ranges.len() <= max_ranges);
        for row in token_scores.iter().flat_map(|s| s.row_ids.iter()) {
            assert_eq!(
                ranges.iter().filter(|range| range.contains(row)).count(),
                1,
                "row {row}"
            );
        }
        assert_eq!(
            split_row_ids(&[], max_ranges),
            vec![(Bound::Unbounded, Bound::Unbounded)]
        );
    }

    #[rstest]
    #[case::no_strides(vec![], stride(vec![], vec![]))]
    #[case::single(vec![stride(vec![3], vec![0.5])], stride(vec![3], vec![0.5]))]
    #[case::keeps_max(
        vec![stride(vec![1, 4], vec![0.5, 0.25]), stride(vec![], vec![]), stride(vec![4, 6], vec![0.75, 0.0])],
        stride(vec![1, 4, 6], vec![0.5, 0.75, 0.0])
    )]
    fn test_merge_strides(#[case] strides: Vec<Stride>, #[case] expected: Stride) {
        assert_eq!(merge_strides(&strides).unwrap(), expected);
    }

    #[rstest]
    #[case::zero_k(vec![stride(vec![1], vec![0.5])], vec![0.0], 0)]
    #[case::no_rows(vec![stride(vec![], vec![]), stride(vec![], vec![])], vec![0.1, 0.2], 3)]
    fn test_top_k_documents_empty(
        #[case] token_scores: Vec<Stride>,
        #[case] missing: Vec<f32>,
        #[case] k: usize,
    ) {
        assert!(
            top_k_documents(&token_scores, &missing, .., |_| true, k)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn test_top_k_documents_skips_ineligible_scores() {
        let token_scores = [stride(vec![1, 2, 3, 4], vec![0.9, 0.8, 0.3, 0.2])];
        // Only scores at most 0.5 compete, so the two best rows do not crowd
        // them out of a top 2.
        let top = top_k_documents(&token_scores, &[0.0], .., |score| score <= 0.5, 2).unwrap();
        assert_eq!(top, vec![(3, 0.3), (4, 0.2)]);
    }

    #[test]
    fn test_top_k_documents_scores_without_cancellation() {
        // Row 7 is observed by every token, row 8 by every token but the last.
        let num_tokens = 32;
        let token_scores = (0..num_tokens)
            .map(|token| {
                if token + 1 < num_tokens {
                    stride(vec![7, 8], vec![1.0, 0.9])
                } else {
                    stride(vec![7], vec![1.0])
                }
            })
            .collect::<Vec<_>>();
        let missing = vec![0.3; num_tokens];
        let top = top_k_documents(&token_scores, &missing, .., |_| true, 2).unwrap();
        // An exact match scores exactly one per token, i.e. distance zero.
        assert_eq!(top[0], (7, num_tokens as f32));
        let expected = (0..num_tokens - 1).fold(0.0_f32, |score, _| score + 0.9) + 0.3;
        assert_eq!(top[1], (8, expected));
    }

    #[test]
    fn test_top_k_documents_ties_prefer_lower_row_id() {
        let token_scores = [stride(vec![3, 5, 9], vec![1.0, 1.0, 1.0])];
        let top = top_k_documents(&token_scores, &[0.0], .., |_| true, 2).unwrap();
        assert_eq!(top, vec![(3, 1.0), (5, 1.0)]);
        assert_eq!(
            merge_top_k([vec![(9, 1.0)], vec![(3, 1.0), (5, 0.5)]], 2),
            vec![(3, 1.0), (9, 1.0)]
        );
    }

    #[rstest]
    #[case::token_count_mismatch(vec![stride(vec![], vec![])], vec![0.1, 0.2])]
    #[case::ragged_stride(vec![stride(vec![1, 2], vec![0.5])], vec![0.1])]
    fn test_top_k_documents_rejects_invalid_input(
        #[case] token_scores: Vec<Stride>,
        #[case] missing: Vec<f32>,
    ) {
        let err = top_k_documents(&token_scores, &missing, .., |_| true, 3).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }

    #[test]
    fn test_merge_strides_rejects_ragged_stride() {
        let err = merge_strides(&[stride(vec![1, 2], vec![0.5])]).unwrap_err();
        assert!(matches!(err, Error::InvalidInput { .. }), "{err}");
    }
}
