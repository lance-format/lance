// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::InvertedPartition;
use std::collections::HashMap;
use std::sync::Arc;

// the Scorer trait is used to calculate the score of a token in a document
// in general, the score is calculated as:
// sum over all query_weight(query_token) * doc_weight(freq, doc_tokens)
pub trait Scorer: Send + Sync {
    fn query_weight(&self, token: &str) -> f32;
    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32;

    /// Finite upper bound for every non-negative value returned by
    /// [`Self::doc_weight`]. Returning `None` disables score-independent
    /// pruning where the posting format has no stored impact bounds.
    fn doc_weight_upper_bound(&self) -> Option<f32> {
        None
    }

    /// Stable identity for the corpus-level inputs used by [`Self::doc_weight`].
    ///
    /// Implementations should return `Some` only when equal keys guarantee the
    /// same document weight for every `(freq, doc_tokens)` pair. Scorers without
    /// such an identity keep impact bounds in the query-local cache only.
    fn doc_weight_cache_key(&self) -> Option<u64> {
        None
    }

    /// The doc-length-dependent BM25 denominator addend, when `doc_weight`
    /// factors as `(K1 + 1) * freq / (freq + addend)`; `None` for scorers
    /// without that shape. Scoring hot loops use this to bake a per-norm-code
    /// addend cache (Lucene's norm cache), which is bit-identical to calling
    /// `doc_weight` because both paths evaluate the same expressions.
    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        let _ = doc_tokens;
        None
    }
}

impl<T: Scorer + ?Sized> Scorer for Arc<T> {
    fn query_weight(&self, token: &str) -> f32 {
        self.as_ref().query_weight(token)
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        self.as_ref().doc_weight(freq, doc_tokens)
    }

    fn doc_weight_upper_bound(&self) -> Option<f32> {
        self.as_ref().doc_weight_upper_bound()
    }

    fn doc_weight_cache_key(&self) -> Option<u64> {
        self.as_ref().doc_weight_cache_key()
    }

    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        self.as_ref().doc_norm(doc_tokens)
    }
}

/// The frequency-dependent half of the BM25 doc weight; `doc_norm` is the
/// doc-length addend (from [`Scorer::doc_norm`] or a per-norm-code cache).
#[inline]
pub(super) fn bm25_doc_weight_with_norm(freq: u32, doc_norm: f32) -> f32 {
    let freq = freq as f32;
    (K1 + 1.0) * freq / (freq + doc_norm)
}

// BM25 parameters
pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

// The f32 multiply/add/divide sequence in `bm25_doc_weight_with_norm` can
// round one ULP above the mathematical K1 + 1 limit.  Keep two ULPs of room so
// every scorer-independent pruning bound remains conservative after its final
// multiplication by the query weight.
pub(super) const BM25_DOC_WEIGHT_UPPER_BOUND: f32 = f32::from_bits((K1 + 1.0).to_bits() + 2);

#[inline]
fn bm25_doc_norm(doc_tokens: u32, avg_doc_length: f32) -> f32 {
    let doc_tokens = doc_tokens as f32;
    K1 * (1.0 - B + B * doc_tokens / avg_doc_length)
}

#[derive(Debug, Clone)]
pub struct MemBM25Scorer {
    pub total_tokens: u64,
    pub num_docs: usize,
    pub token_docs: HashMap<String, usize>,
}

impl MemBM25Scorer {
    pub fn new(total_tokens: u64, num_docs: usize, token_docs: HashMap<String, usize>) -> Self {
        Self {
            total_tokens,
            num_docs,
            token_docs,
        }
    }

    /// Incremental update bm25 scorer with one new document.
    ///
    /// # Arguments
    /// * `tokens` - The tokens of the new document that are also in the query
    /// * `num_tokens` - The total number of tokens in the document
    pub fn update(&mut self, doc_token_count: &HashMap<String, usize>, num_tokens: u64) {
        self.total_tokens += num_tokens;
        self.num_docs += 1;
        for (token, count) in doc_token_count {
            if let Some(old_count) = self.token_docs.get_mut(token) {
                *old_count += *count;
            } else {
                // This shouldn't happen because `tokens` should only contain tokens that are in the query
                // and we should have already initialized this with query tokens.  Still, log a warning just in case.
                log::warn!("Token {} not found in token_docs", token);
            }
        }
    }

    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    pub fn avg_doc_length(&self) -> f32 {
        self.total_tokens as f32 / self.num_docs as f32
    }

    pub fn num_docs_containing_token(&self, token: &str) -> usize {
        match self.token_docs.get(token) {
            Some(nq) => *nq,
            None => 0,
        }
    }
}

impl Scorer for MemBM25Scorer {
    fn query_weight(&self, token: &str) -> f32 {
        let token_docs = self.num_docs_containing_token(token);
        if token_docs == 0 {
            return 0.0;
        }
        idf(token_docs, self.num_docs)
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        let doc_norm = bm25_doc_norm(doc_tokens, self.avg_doc_length());
        bm25_doc_weight_with_norm(freq, doc_norm)
    }

    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        Some(bm25_doc_norm(doc_tokens, self.avg_doc_length()))
    }

    fn doc_weight_upper_bound(&self) -> Option<f32> {
        Some(BM25_DOC_WEIGHT_UPPER_BOUND)
    }

    fn doc_weight_cache_key(&self) -> Option<u64> {
        Some(u64::from(self.avg_doc_length().to_bits()))
    }
}

pub struct IndexBM25Scorer<'a> {
    partitions: Vec<&'a InvertedPartition>,
    num_docs: usize,
    avg_doc_length: f32,
}

impl<'a> IndexBM25Scorer<'a> {
    /// Sync constructor.  Query setup populates immutable partition stats
    /// before entering the CPU-only WAND executor.
    pub fn new(partitions: impl Iterator<Item = &'a InvertedPartition>) -> Self {
        let partitions = partitions.collect::<Vec<_>>();
        let stats = partitions
            .iter()
            .map(|partition| {
                partition.docs.cached_stats().expect(
                    "IndexBM25Scorer::new requires partition stats to be loaded before WAND",
                )
            })
            .collect::<Vec<_>>();
        let num_docs = stats.iter().map(|stats| stats.num_docs).sum();
        let total_tokens: u64 = stats.iter().map(|stats| stats.total_tokens).sum();
        let avgdl = total_tokens as f32 / num_docs as f32;
        Self {
            partitions,
            num_docs,
            avg_doc_length: avgdl,
        }
    }

    pub fn num_docs_containing_token(&self, token: &str) -> usize {
        self.partitions
            .iter()
            .map(|part| {
                if let Some(token_id) = part.tokens.get(token) {
                    part.inverted_list.posting_len(token_id)
                } else {
                    0
                }
            })
            .sum()
    }
}

impl Scorer for IndexBM25Scorer<'_> {
    fn query_weight(&self, token: &str) -> f32 {
        let token_docs = self.num_docs_containing_token(token);
        if token_docs == 0 {
            return 0.0;
        }
        idf(token_docs, self.num_docs)
    }

    fn doc_weight(&self, freq: u32, doc_tokens: u32) -> f32 {
        let doc_norm = bm25_doc_norm(doc_tokens, self.avg_doc_length);
        bm25_doc_weight_with_norm(freq, doc_norm)
    }

    fn doc_norm(&self, doc_tokens: u32) -> Option<f32> {
        Some(bm25_doc_norm(doc_tokens, self.avg_doc_length))
    }

    fn doc_weight_upper_bound(&self) -> Option<f32> {
        Some(BM25_DOC_WEIGHT_UPPER_BOUND)
    }

    fn doc_weight_cache_key(&self) -> Option<u64> {
        Some(u64::from(self.avg_doc_length.to_bits()))
    }
}

#[inline]
pub fn idf(token_docs: usize, num_docs: usize) -> f32 {
    let num_docs = num_docs as f32;
    ((num_docs - token_docs as f32 + 0.5) / (token_docs as f32 + 0.5) + 1.0).ln()
}

/// BM25F scorer over a virtual field formed by combining several columns
/// (Lucene `CombinedFieldQuery` / Elasticsearch `combined_fields`).
///
/// Where [`MemBM25Scorer`] / `IndexBM25Scorer` score one column against that
/// column's own statistics, this scorer holds statistics blended across the
/// target columns per the BM25F rules:
/// - `doc_count` = `max_f docCount_f`
/// - `avg_doc_length` = `sumTotalTermFreq' / doc_count`, where
///   `sumTotalTermFreq' = Σ_f w_f · sumTotalTermFreq_f`
/// - `doc_freq(t)` = `max_f docFreq_f(t)`
///
/// The caller supplies the blended term frequency `tf' = Σ_f w_f · tf_f(t, d)`
/// and blended document length `dl' = Σ_f w_f · dl_f(d)` per candidate document;
/// this type turns those into an IDF weight and a BM25 document weight. Lance's
/// `(k1 + 1)` numerator is kept (a constant factor vs. Lucene that preserves
/// ranking).
#[derive(Debug, Clone)]
pub struct CombinedFieldsBM25Scorer {
    doc_count: usize,
    avg_doc_length: f32,
    doc_freq: HashMap<String, usize>,
}

impl CombinedFieldsBM25Scorer {
    pub fn new(doc_count: usize, avg_doc_length: f32, doc_freq: HashMap<String, usize>) -> Self {
        Self {
            doc_count,
            avg_doc_length,
            doc_freq,
        }
    }

    pub fn doc_count(&self) -> usize {
        self.doc_count
    }

    pub fn avg_doc_length(&self) -> f32 {
        self.avg_doc_length
    }

    /// Blended IDF for `term` over the virtual field. Returns `0.0` for a term
    /// absent from every target column (`docFreq' == 0`), and for the inconsistent
    /// statistics (`docFreq' > docCount'`) that drive `idf` negative or, once
    /// `docFreq'` is large enough for the ratio to round to `-1`, to `-inf`. A
    /// `-inf` weight times a zero document weight is `NaN`; see [`Self::doc_weight`]
    /// for why no score may be non-finite.
    pub fn query_weight(&self, term: &str) -> f32 {
        match self.doc_freq.get(term).copied() {
            Some(token_docs) if token_docs > 0 => {
                let idf = idf(token_docs, self.doc_count);
                if idf.is_finite() && idf > 0.0 {
                    idf
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// BM25 term contribution for a document, given the blended term frequency
    /// `tf'` and blended document length `dl'`.
    ///
    /// The result is always finite and within `[0, BM25_DOC_WEIGHT_UPPER_BOUND]`,
    /// whatever reaches it. BM25 saturates the `tf'` factor at `K1 + 1` in exact
    /// arithmetic, but the f32 evaluation can land above it: once `doc_norm` is
    /// small relative to `ulp(tf')` the rounded `tf' + doc_norm` collapses back to
    /// `tf'`, and `fl(fl((K1 + 1) · tf') / tf')` then rounds up. Large per-column
    /// boosts can also push `tf'`, `dl'`, or `avgdl'` to infinity, and `Inf / Inf`
    /// is `NaN`; one `NaN` contribution would poison a whole query's scores. The
    /// clamp makes the ceiling hold by construction rather than by trusting the
    /// query-level validation of the boosts.
    pub fn doc_weight(&self, tf_prime: f32, dl_prime: f32) -> f32 {
        if self.avg_doc_length <= 0.0 {
            return 0.0;
        }
        let doc_norm = K1 * (1.0 - B + B * dl_prime / self.avg_doc_length);
        let weight = (K1 + 1.0) * tf_prime / (tf_prime + doc_norm);
        if weight.is_nan() {
            // `Inf / Inf`, or a `NaN` that came in through `tf'` / `dl'` /
            // `avgdl'`. There is no meaningful weight to report, and `clamp`
            // would propagate the `NaN`.
            return 0.0;
        }
        weight.clamp(0.0, BM25_DOC_WEIGHT_UPPER_BOUND)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_doc_weight_upper_bound_covers_f32_rounding() {
        let scorer = MemBM25Scorer::new(6_242_289_027, 2, HashMap::new());
        let doc_weight = scorer.doc_weight(3_926_982_873, 4_078_552_115);

        assert_eq!(doc_weight.to_bits(), 0x400c_ccce);
        assert!(doc_weight > K1 + 1.0);
        assert!(scorer.doc_weight_upper_bound().unwrap() >= doc_weight);
    }

    #[test]
    fn test_combined_scorer_query_weight_matches_blended_idf() {
        let doc_freq = HashMap::from([("rare".to_string(), 2), ("common".to_string(), 800)]);
        let scorer = CombinedFieldsBM25Scorer::new(1000, 12.0, doc_freq);

        // IDF uses the blended docFreq' over the blended docCount'.
        assert_eq!(scorer.query_weight("rare"), idf(2, 1000));
        assert_eq!(scorer.query_weight("common"), idf(800, 1000));
        // A rarer term outweighs a common one.
        assert!(scorer.query_weight("rare") > scorer.query_weight("common"));
        // A term absent from every field contributes nothing.
        assert_eq!(scorer.query_weight("missing"), 0.0);
    }

    /// No corpus statistics and no blended `(tf', dl')` may produce a non-finite
    /// term score. A `NaN` would order arbitrarily in the top-k heap, and a weight
    /// above [`BM25_DOC_WEIGHT_UPPER_BOUND`] would break any pruning bound derived
    /// from that ceiling.
    #[test]
    fn test_combined_scorer_stays_finite_for_extreme_statistics() {
        // `docFreq' > docCount'` drives `idf` negative and then to `-inf`.
        let doc_freq = HashMap::from([
            ("ok".to_string(), 1),
            ("over".to_string(), 2_000),
            ("way_over".to_string(), usize::MAX),
        ]);
        let blends = [
            (1.0f32, 8.0f32),
            (0.0, 0.0),
            (f32::MAX, f32::MAX),
            (f32::INFINITY, f32::INFINITY),
            (f32::INFINITY, 8.0),
            (8.0, f32::INFINITY),
            (f32::NAN, f32::NAN),
            (-1.0, -1.0),
        ];
        for avg_doc_length in [10.0f32, 0.0, f32::MAX, f32::INFINITY, f32::NAN, -1.0] {
            let scorer = CombinedFieldsBM25Scorer::new(1000, avg_doc_length, doc_freq.clone());
            for term in ["ok", "over", "way_over", "missing"] {
                let query_weight = scorer.query_weight(term);
                assert!(
                    query_weight.is_finite() && query_weight >= 0.0,
                    "query_weight({term}) = {query_weight:e}"
                );
                for (tf_prime, dl_prime) in blends {
                    let score = query_weight * scorer.doc_weight(tf_prime, dl_prime);
                    assert!(
                        score.is_finite() && score >= 0.0,
                        "score for {term} at tf'={tf_prime:e} dl'={dl_prime:e} \
                         avgdl'={avg_doc_length:e} was {score:e}"
                    );
                }
            }
        }
        // The unclamped reference: `idf` really does leave the usable range for
        // these statistics, so the clamp in `query_weight` is important. In the
        // `-inf` case multiplied by a zero `doc_weight` it would yield `NaN`.
        assert!(idf(2_000, 1000) < 0.0);
        assert_eq!(idf(usize::MAX, 1000), f32::NEG_INFINITY);
    }

    #[test]
    fn test_combined_scorer_doc_weight_saturates_and_penalizes_length() {
        let scorer = CombinedFieldsBM25Scorer::new(1000, 10.0, HashMap::new());

        // Matches the BM25 formula (with Lance's (k1 + 1) numerator).
        let expected = {
            let doc_norm = K1 * (1.0 - B + B * 20.0 / 10.0);
            (K1 + 1.0) * 3.0 / (3.0 + doc_norm)
        };
        assert!((scorer.doc_weight(3.0, 20.0) - expected).abs() < 1e-6);
        // More term frequency scores higher, saturating below (k1 + 1).
        assert!(scorer.doc_weight(5.0, 20.0) > scorer.doc_weight(1.0, 20.0));
        assert!(scorer.doc_weight(1000.0, 20.0) < K1 + 1.0);
        // A longer document is penalized for the same term frequency.
        assert!(scorer.doc_weight(3.0, 40.0) < scorer.doc_weight(3.0, 5.0));
        // A degenerate (empty) corpus scores zero rather than dividing by zero.
        let empty = CombinedFieldsBM25Scorer::new(0, 0.0, HashMap::new());
        assert_eq!(empty.doc_weight(3.0, 20.0), 0.0);
    }
}
