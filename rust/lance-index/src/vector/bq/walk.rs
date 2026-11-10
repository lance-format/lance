// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The 1-bit graph walk of `IVF_HNSW_RQ`: it scores the query against the sign
//! bits only, with the query quantized to 4 bits so that AND + popcount over bit
//! planes replaces the float inner product. `VectorStore::rerank` rescores.

use crate::vector::utils::do_prefetch;

/// Bits per dim of the walk's query.
const QUERY_BITS: usize = 4;
/// From this many `u64` words (512 dims) on, each query plane is stored whole,
/// which the compiler vectorizes; below it, each word keeps its planes together.
const PLANE_MAJOR_WORDS: usize = 8;

/// Packs each row of row-major `_rabit_codes` into little-endian `u64` words,
/// dim `d` at bit `d % 64` of word `d / 64`.
pub fn sign_words(codes: &[u8], code_len: usize) -> Vec<u64> {
    let mut words = Vec::with_capacity(codes.len() / code_len * code_len.div_ceil(8));
    words.extend(
        codes
            .chunks_exact(code_len)
            .flat_map(|row| row.chunks(8))
            .map(|bytes| {
                let mut word = [0u8; 8];
                word[..bytes.len()].copy_from_slice(bytes);
                u64::from_le_bytes(word)
            }),
    );
    words
}

/// The walk over a partition's [`sign_words`], with the rotated query quantized
/// uniformly over its range to 4-bit codes that reconstruct as `delta * code + vl`.
pub struct OneBitWalk<'a> {
    signs: &'a [u64],
    words: usize,
    planes: Vec<u64>,
    delta: f32,
    vl: f32,
    ip_error_bound: f32,
}

impl<'a> OneBitWalk<'a> {
    pub fn new(signs: &'a [u64], rotated: &[f32]) -> Self {
        let (lo, hi) = rotated
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let max_code = ((1 << QUERY_BITS) - 1) as f32;
        let (vl, delta) = if hi > lo {
            (lo, (hi - lo) / max_code)
        } else {
            (lo.min(0.0).max(hi), 0.0)
        };
        let (mut codes, mut sum_error, mut error_sq) =
            (Vec::with_capacity(rotated.len()), 0.0, 0.0);
        for &v in rotated {
            let code = if delta > 0.0 {
                ((v - vl) / delta).round().clamp(0.0, max_code)
            } else {
                0.0
            };
            let error = delta * code + vl - v;
            sum_error += error;
            error_sq += error * error;
            codes.push(code as u8);
        }
        Self {
            signs,
            words: rotated.len().div_ceil(64),
            planes: transpose(&codes),
            delta,
            vl,
            // A sign row `b` sums the reconstruction error `err` over its set
            // bits, `(sum(err) + <2b - 1, err>) / 2`, and padded bits are never
            // set; Cauchy-Schwarz bounds `|<2b - 1, err>|` by `sqrt(dim) |err|`.
            ip_error_bound: 0.5
                * (f32::abs(sum_error) + (rotated.len() as f32).sqrt() * f32::sqrt(error_sq)),
        }
    }

    /// The reconstructed query summed over the dims whose sign bit is set in row `id`.
    #[inline(always)]
    pub fn ip(&self, id: usize) -> f32 {
        let (ip, count) = ip_counts(self.row(id), &self.planes);
        self.delta * ip as f32 + self.vl * count as f32
    }

    /// Bounds how far [`Self::ip`] lies from the exact inner product with any row.
    pub fn ip_error_bound(&self) -> f32 {
        self.ip_error_bound
    }

    #[inline(always)]
    pub fn prefetch(&self, id: usize) {
        do_prefetch(self.row(id).as_ptr_range());
    }

    #[inline(always)]
    fn row(&self, id: usize) -> &'a [u64] {
        &self.signs[id * self.words..(id + 1) * self.words]
    }
}

/// Transposes 4-bit query codes into bit planes, dim `d` at bit `d % 64` of word
/// `d / 64`: plane by plane from [`PLANE_MAJOR_WORDS`] words on, else word by word.
fn transpose(codes: &[u8]) -> Vec<u64> {
    let words = codes.len().div_ceil(64);
    let mut planes = vec![0u64; words * QUERY_BITS];
    for (d, &code) in codes.iter().enumerate() {
        for j in 0..QUERY_BITS {
            let at = if words >= PLANE_MAJOR_WORDS {
                j * words + d / 64
            } else {
                d / 64 * QUERY_BITS + j
            };
            planes[at] |= u64::from((code >> j) & 1) << (d % 64);
        }
    }
    planes
}

/// The sum of the query codes over the set bits of `signs`, and their count.
#[inline(always)]
fn ip_counts(signs: &[u64], planes: &[u64]) -> (u64, u64) {
    let count = signs.iter().map(|x| u64::from(x.count_ones())).sum();
    let mut ip = 0u64;
    if signs.len() >= PLANE_MAJOR_WORDS {
        for (j, plane) in planes.chunks_exact(signs.len()).enumerate() {
            let sum: u64 = signs
                .iter()
                .zip(plane)
                .map(|(x, q)| u64::from((x & q).count_ones()))
                .sum();
            ip += sum << j;
        }
    } else {
        for (&x, p) in signs.iter().zip(planes.as_chunks::<QUERY_BITS>().0) {
            ip += u64::from((x & p[0]).count_ones())
                + (u64::from((x & p[1]).count_ones()) << 1)
                + (u64::from((x & p[2]).count_ones()) << 2)
                + (u64::from((x & p[3]).count_ones()) << 3);
        }
    }
    (ip, count)
}

#[cfg(test)]
mod tests {
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    use super::*;

    /// Both plane layouts sum the query codes over the set sign bits, and the
    /// walk stays within its error bound of the exact inner product. 8 and 200
    /// dims end in a partial word; 512 and 960 store the planes whole.
    #[test]
    fn test_walk_inner_products() {
        let mut rng = SmallRng::seed_from_u64(7);
        for dim in [8usize, 64, 200, 512, 960] {
            let codes: Vec<u8> = (0..4 * dim / 8).map(|_| rng.random()).collect();
            let signs = sign_words(&codes, dim / 8);
            let is_set = |id: usize, d: usize| (codes[id * dim / 8 + d / 8] >> (d % 8)) & 1 == 1;
            let query: Vec<u8> = (0..dim).map(|_| rng.random_range(0..16)).collect();
            let rotated: Vec<f32> = (0..dim).map(|_| rng.random_range(-3.0..3.0)).collect();
            let walk = OneBitWalk::new(&signs, &rotated);
            // Each dim reconstructs within half a step, which caps the bound.
            assert!(walk.ip_error_bound() <= 0.5 * dim as f32 * walk.delta + 1e-4);
            for id in 0..4 {
                let row = &signs[id * walk.words..(id + 1) * walk.words];
                let set = (0..dim).filter(|&d| is_set(id, d));
                let want = (
                    set.clone().map(|d| query[d] as u64).sum(),
                    set.clone().count() as u64,
                );
                assert_eq!(ip_counts(row, &transpose(&query)), want, "{dim}/{id}");
                let exact: f32 = set.map(|d| rotated[d]).sum();
                let error = (walk.ip(id) - exact).abs();
                assert!(error <= walk.ip_error_bound() + 1e-3, "{dim}/{id}: {error}");
            }
        }
    }
}
