// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::arch::x86_64::*;

use lance_linalg::simd::dist_table::{BATCH_SIZE, PERM0};

pub(super) const SIGN_BYTES_512: usize = 64;
const TILE_COLS: usize = 16;
const BATCH_BYTES: usize = BATCH_SIZE * SIGN_BYTES_512;

/// Decode complete FastScan batches into 512-dimensional row-major signs.
///
/// # Safety
/// The CPU must support AVX2. Both slices must hold equally many full batches.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn unpack_512_blocks(packed: &[u8], unpacked: &mut [u8]) {
    debug_assert_eq!(packed.len(), unpacked.len());
    debug_assert!(packed.len().is_multiple_of(BATCH_BYTES));
    let blocks = packed.as_chunks::<BATCH_BYTES>().0;
    let output = unpacked.as_chunks_mut::<BATCH_BYTES>().0;
    for (packed, out) in blocks.iter().zip(output) {
        let out = out.as_chunks_mut::<SIGN_BYTES_512>().0;
        let cols = packed.as_chunks::<BATCH_SIZE>().0;
        for (tile_id, tile) in cols.as_chunks::<TILE_COLS>().0.iter().enumerate() {
            let columns = tile.each_ref().map(|col| {
                // SAFETY: each packed column consists of two 16-byte halves.
                let (lo, hi) = unsafe {
                    (
                        _mm_loadu_si128(col.as_ptr().cast()),
                        _mm_loadu_si128(col.as_ptr().add(16).cast()),
                    )
                };
                let low_nibble = _mm_set1_epi8(0x0f);
                let high_nibble = _mm_set1_epi8(0xf0u8 as i8);
                let first_rows = _mm_or_si128(
                    _mm_and_si128(lo, low_nibble),
                    _mm_and_si128(_mm_slli_epi16::<4>(hi), high_nibble),
                );
                let last_rows = _mm_or_si128(
                    _mm_and_si128(_mm_srli_epi16::<4>(lo), low_nibble),
                    _mm_and_si128(hi, high_nibble),
                );
                _mm256_set_m128i(last_rows, first_rows)
            });

            // Each 128-bit lane holds a column of a 16x16 tile from one half
            // of the batch. The unpacks transpose both tiles at once.
            macro_rules! transpose_stage {
                ($vectors:ident, $half:literal, $low:ident, $high:ident) => {
                    std::array::from_fn::<_, TILE_COLS, _>(|i| {
                        let group = i / (2 * $half) * (2 * $half);
                        let start = group + i % (2 * $half) / 2;
                        let left = $vectors[start];
                        let right = $vectors[start + $half];
                        if i % 2 == 0 {
                            $low(left, right)
                        } else {
                            $high(left, right)
                        }
                    })
                };
            }
            let pairs = transpose_stage!(columns, 1, _mm256_unpacklo_epi8, _mm256_unpackhi_epi8);
            let quads = transpose_stage!(pairs, 2, _mm256_unpacklo_epi16, _mm256_unpackhi_epi16);
            let octets = transpose_stage!(quads, 4, _mm256_unpacklo_epi32, _mm256_unpackhi_epi32);
            let rows = transpose_stage!(octets, 8, _mm256_unpacklo_epi64, _mm256_unpackhi_epi64);
            let offset = tile_id * TILE_COLS;
            // Apply the permutation once per output row, after transposing.
            for (row, position) in rows.into_iter().zip(PERM0) {
                // SAFETY: all four tile offsets leave room for 16 bytes in
                // each 64-byte output row.
                unsafe {
                    _mm_storeu_si128(
                        out[position][offset..].as_mut_ptr().cast(),
                        _mm256_castsi256_si128(row),
                    );
                    _mm_storeu_si128(
                        out[position + 16][offset..].as_mut_ptr().cast(),
                        _mm256_extracti128_si256::<1>(row),
                    );
                }
            }
        }
    }
}
