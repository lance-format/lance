// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact distance-table scans over column-major PQ8 codes.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use lance_core::assume_eq;

const NUM_CENTROIDS: usize = 256;
// Four code columns, four LUTs and this distance block occupy 20 KiB.
// Keeping that working set in L1 avoids streaming the full distance array
// once per sub-vector, without changing the persisted code layout.
#[cfg(target_arch = "x86_64")]
const BLOCK_ROWS: usize = 2048;
// Checking all LUT entries costs more than gathers save for short columns.
#[cfg(target_arch = "x86_64")]
const MIN_GATHER_ROWS: usize = 512;

type ScanFn = unsafe fn(&[u8], &[[f32; NUM_CENTROIDS]], &mut [f32]);

lance_core::dispatch! {
    static SCAN: Dispatch<ScanFn> = {
        #[cfg(target_arch = "x86_64")]
        { &[SCAN_AVX512, SCAN_PORTABLE] }
        #[cfg(not(target_arch = "x86_64"))]
        { &[SCAN_PORTABLE] }
    };
}

pub(super) fn compute_pq8_distance_into(
    distance_table: &[f32],
    num_sub_vectors: usize,
    codes: &[u8],
    distances: &mut Vec<f32>,
) {
    if codes.is_empty() {
        distances.clear();
        return;
    }
    assert!(num_sub_vectors > 0, "PQ8 requires at least one sub-vector");
    assert!(
        codes.len().is_multiple_of(num_sub_vectors),
        "PQ8 code columns have unequal lengths"
    );
    let tables = &distance_table.as_chunks::<NUM_CENTROIDS>().0[..num_sub_vectors];
    distances.resize(codes.len() / num_sub_vectors, 0.0);
    SCAN.get().call(codes, tables, distances);
}

// Every candidate checks its own bounds: Candidate::kernel() is also used
// directly by tests, without passing through compute_pq8_distance_into.
fn check_code_bounds(codes: &[u8], num_sub_vectors: usize, num_rows: usize) {
    assert!(
        num_rows == 0 || num_sub_vectors <= codes.len() / num_rows,
        "PQ8 codes are shorter than the requested rows and sub-vectors"
    );
}

lance_core::target_kernel! {
    #[target_features()]
    #[candidate(SCAN_PORTABLE: ScanFn)]
    fn scan_portable(codes: &[u8], tables: &[[f32; NUM_CENTROIDS]], distances: &mut [f32]) {
        check_code_bounds(codes, tables.len(), distances.len());
        distances.fill(0.0);
        if distances.is_empty() {
            return;
        }
        // Keep the existing column-at-a-time loop for hosts without AVX-512,
        // short partitions and exceptional LUT entries.
        for (table, indices) in tables.iter().zip(codes.chunks_exact(distances.len())) {
            assume_eq!(indices.len(), distances.len());
            indices.iter().zip(distances.iter_mut()).for_each(|(&index, sum)| {
                *sum += table[index as usize];
            });
        }
    }
}

#[cfg(target_arch = "x86_64")]
lance_core::target_kernel! {
    #[target_features("avx512f")]
    #[candidate(SCAN_AVX512: ScanFn)]
    fn scan_avx512(codes: &[u8], tables: &[[f32; NUM_CENTROIDS]], distances: &mut [f32]) {
        check_code_bounds(codes, tables.len(), distances.len());
        if distances.len() < MIN_GATHER_ROWS || tables.len() <= 1 || !all_finite(tables) {
            scan_portable(codes, tables, distances);
            return;
        }
        // SAFETY: the dispatch checks AVX-512F. The code bounds above and
        // fixed-size LUTs cover every load; each u8 gather index is < 256.
        unsafe { scan_blocks_avx512(codes, tables, distances) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
fn all_finite(tables: &[[f32; NUM_CENTROIDS]]) -> bool {
    // Exclude infinities and all NaN payloads before selecting SIMD arithmetic.
    // SAFETY: each table is a multiple of 16 floats and AVX-512F is enabled.
    unsafe {
        let exponent = _mm512_set1_epi32(0x7f800000);
        for chunk in tables.as_flattened().as_chunks::<16>().0 {
            let bits = _mm512_castps_si512(_mm512_loadu_ps(chunk.as_ptr()));
            if _mm512_cmpeq_epi32_mask(_mm512_and_si512(bits, exponent), exponent) != 0 {
                return false;
            }
        }
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn lookup16(table: &[f32; NUM_CENTROIDS], codes: *const u8) -> __m512 {
    // SAFETY: caller supplies 16 readable codes. Zero extension bounds every
    // gather offset to this table's 256 entries; neither input needs alignment.
    unsafe {
        _mm512_i32gather_ps::<4>(
            _mm512_cvtepu8_epi32(_mm_loadu_si128(codes.cast())),
            table.as_ptr(),
        )
    }
}

/// The caller must enable AVX-512F and provide at least tables.len() complete
/// code columns of distances.len() bytes each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scan_blocks_avx512(codes: &[u8], tables: &[[f32; NUM_CENTROIDS]], distances: &mut [f32]) {
    let num_rows = distances.len();
    for start in (0..num_rows).step_by(BLOCK_ROWS) {
        let end = num_rows.min(start + BLOCK_ROWS);
        let vector_end = start + (end - start) / 16 * 16;
        let mut sub = 0;
        // SAFETY: only complete 16-row vectors are loaded and stored. The
        // checked code shape bounds sub * num_rows + row, and LUT indices
        // cannot exceed 255. Scalar code handles the final partial vector.
        unsafe {
            while sub + 4 <= tables.len() {
                let c0 = codes.as_ptr().add(sub * num_rows);
                let c1 = c0.add(num_rows);
                let c2 = c1.add(num_rows);
                let c3 = c2.add(num_rows);
                for row in (start..vector_end).step_by(16) {
                    let dst = distances.as_mut_ptr().add(row);
                    let mut sum = if sub == 0 {
                        _mm512_setzero_ps()
                    } else {
                        _mm512_loadu_ps(dst)
                    };
                    // Four ordered additions reuse one output load/store.
                    // Reassociation would change scores at the TopK boundary.
                    sum = _mm512_add_ps(sum, lookup16(&tables[sub], c0.add(row)));
                    sum = _mm512_add_ps(sum, lookup16(&tables[sub + 1], c1.add(row)));
                    sum = _mm512_add_ps(sum, lookup16(&tables[sub + 2], c2.add(row)));
                    sum = _mm512_add_ps(sum, lookup16(&tables[sub + 3], c3.add(row)));
                    _mm512_storeu_ps(dst, sum);
                }
                sub += 4;
            }
            for (sub, table) in tables.iter().enumerate().skip(sub) {
                let column = codes.as_ptr().add(sub * num_rows);
                for row in (start..vector_end).step_by(16) {
                    let dst = distances.as_mut_ptr().add(row);
                    let sum = if sub == 0 {
                        _mm512_setzero_ps()
                    } else {
                        _mm512_loadu_ps(dst)
                    };
                    _mm512_storeu_ps(dst, _mm512_add_ps(sum, lookup16(table, column.add(row))));
                }
            }
        }
        for (row, distance) in distances.iter_mut().enumerate().take(end).skip(vector_end) {
            *distance = tables.iter().enumerate().fold(0.0f32, |sum, (sub, table)| {
                sum + table[codes[sub * num_rows + row] as usize]
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use rstest::rstest;

    #[rstest]
    fn test_candidates_match_row_reference(
        #[values(0, 1, 15, 16, 31, 32, 511, 512, 513, 2047, 2048, 2049, 4099)] rows: usize,
        #[values(1, 2, 3, 4, 5, 7, 16, 96, 192)] sub_vectors: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(42);
        let tables = (0..sub_vectors)
            .map(|_| std::array::from_fn(|_| rng.random_range(-10.0f32..10.0)))
            .collect::<Vec<[f32; NUM_CENTROIDS]>>();
        let codes = (0..rows * sub_vectors)
            .map(|_| rng.random())
            .collect::<Vec<u8>>();
        let expected = (0..rows)
            .map(|row| {
                tables
                    .iter()
                    .enumerate()
                    .fold(0.0f32, |sum, (sub, table)| {
                        sum + table[codes[sub * rows + row] as usize]
                    })
                    .to_bits()
            })
            .collect::<Vec<_>>();
        for candidate in SCAN.candidates() {
            if let Some(kernel) = candidate.kernel() {
                let mut actual = vec![f32::NAN; rows];
                kernel.call(&codes, &tables, &mut actual);
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected,
                    "{}",
                    candidate.name()
                );
            }
        }
        // Exercise the production entry point with reused and oversized scratch.
        let mut actual = vec![f32::NAN; rows + 17];
        for _ in 0..2 {
            compute_pq8_distance_into(tables.as_flattened(), sub_vectors, &codes, &mut actual);
            assert_eq!(
                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[rstest]
    fn test_unaligned_inputs_and_output(
        #[values(0, 1, 3, 15)] code_offset: usize,
        #[values(0, 1, 7)] float_offset: usize,
    ) {
        let (rows, sub_vectors) = (2051, 7);
        let table = (0..float_offset + sub_vectors * 256)
            .map(|i| (i % 513) as f32 * 0.017 - 4.0)
            .collect::<Vec<_>>();
        let tables = table[float_offset..].as_chunks::<256>().0;
        let codes = (0..code_offset + rows * sub_vectors)
            .map(|i| (i * 127 + i / 13) as u8)
            .collect::<Vec<_>>();
        let codes = &codes[code_offset..];
        let mut expected = vec![0.0; rows];
        scan_portable(codes, tables, &mut expected);
        for candidate in SCAN.candidates() {
            if let Some(kernel) = candidate.kernel() {
                let mut actual = vec![123.0; float_offset + rows + 16];
                kernel.call(
                    codes,
                    tables,
                    &mut actual[float_offset..float_offset + rows],
                );
                assert_eq!(actual[float_offset..float_offset + rows], expected);
                assert!(actual[..float_offset].iter().all(|&x| x == 123.0));
                assert!(actual[float_offset + rows..].iter().all(|&x| x == 123.0));
            }
        }
    }

    #[rstest]
    #[case::finite(&[0.0, -0.0, f32::MAX, -f32::MAX, f32::MIN_POSITIVE, -f32::MIN_POSITIVE, f32::from_bits(1), -f32::from_bits(1)])]
    #[case::nonfinite(&[0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::from_bits(0x7fc01234), f32::from_bits(0xffc05678)])]
    fn test_exceptional_values(#[case] values: &[f32]) {
        let (rows, sub_vectors) = (2051, 7);
        let tables = (0..sub_vectors)
            .map(|sub| std::array::from_fn(|i| values[(i + sub) % values.len()]))
            .collect::<Vec<[f32; 256]>>();
        let codes = (0..rows * sub_vectors)
            .map(|i| (i * 127) as u8)
            .collect::<Vec<_>>();
        let mut expected = vec![0.0; rows];
        scan_portable(&codes, &tables, &mut expected);
        for candidate in SCAN.candidates() {
            if let Some(kernel) = candidate.kernel() {
                let mut actual = vec![f32::NAN; rows];
                kernel.call(&codes, &tables, &mut actual);
                for (a, b) in actual.iter().zip(&expected) {
                    assert_eq!(a.to_bits(), b.to_bits(), "{}", candidate.name());
                }
            }
        }
    }
}
