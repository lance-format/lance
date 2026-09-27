// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

#[allow(unused_imports)]
use lance_core::utils::cpu::{SIMD_SUPPORT, SimdSupport};

pub const PERM0: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
pub const PERM0_INVERSE: [usize; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15];
pub const BATCH_SIZE: usize = 32;
/// A row sums `2 * code_len` table entries into a `u16`, so a full-range
/// (`0..=u8::MAX`) table fits only up to here: `2 * 128 * u8::MAX == 65280`.
/// Callers that cannot cap their table must use [`sum_4bit_dist_table_u32`].
pub const SAFE_U16_CODE_LEN: usize = 128;

/// Which kernel a `dist_table` entry point prefers on this host. An entry point
/// with no arm for the chosen backend falls through to scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DistTableBackend {
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

/// Picks the kernel from a tier and two feature bits passed in, reading neither
/// `SIMD_SUPPORT` nor the CPU, so the consequences of the tier ladder are
/// testable without a host of each kind.
///
/// `avx512_kernel` says whether the calling entry point has an AVX-512 kernel to
/// offer: `sum_4bit_dist_table_uninit` has one behind
/// `kernel_support = "avx512_dist_table"`, `sum_4bit_dist_table_transposed` and
/// `filter_4bit_dist_table_transposed` have a Rust one on every x86_64 build
/// with no `kernel_support` cfg, the exact `f32` entry points pass `true` on
/// every build since only an x86_64 tier can select it, and the hacc entry
/// point has none.
fn dist_table_backend(
    support: SimdSupport,
    avx512_kernel: bool,
    has_avx512bw: bool,
    has_avx2: bool,
) -> DistTableBackend {
    match support {
        SimdSupport::Avx512 | SimdSupport::Avx512FP16 if avx512_kernel && has_avx512bw => {
            DistTableBackend::Avx512
        }
        // `SIMD_SUPPORT` is a single exclusive tier, so an AVX-512 host reports
        // `Avx512` or `Avx512FP16` and never `Avx2`. Naming only `Avx2` here
        // would send an AVX-512 host that missed the arm above down the scalar
        // path while it has AVX2.
        SimdSupport::Avx512 | SimdSupport::Avx512FP16 | SimdSupport::Avx2 if has_avx2 => {
            DistTableBackend::Avx2
        }
        SimdSupport::Neon => DistTableBackend::Neon,
        _ => DistTableBackend::Scalar,
    }
}

/// The two feature bits [`dist_table_backend`] needs, or `false` off x86_64.
#[inline]
fn x86_dist_table_features() -> (bool, bool) {
    #[cfg(target_arch = "x86_64")]
    {
        (
            std::arch::is_x86_feature_detected!("avx512bw"),
            std::arch::is_x86_feature_detected!("avx2"),
        )
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        (false, false)
    }
}

// This function is used to sum the distance table for 4-bit codes.
// the distance table is a 2D array, that dist_table[i][j] is the distance between the i-th subvector and the code j,
// the distance table is stored as a flat array for better cache locality and SIMD instruction usage.
//
// The codes are organized in the order of PERM0:
// +----------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
// | address  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
// | (bytes)  |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
// +----------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
// | bits 0..3|  0 |  8 |  1 |  9 |  2 | 10 |  3 | 11 |  4 | 12 |  5 | 13 |  6 | 14 |  7 | 15 |
// | bits 4..7| 16 | 24 | 17 | 25 | 18 | 26 | 19 | 27 | 20 | 28 | 21 | 29 | 22 | 30 | 23 | 31 |
// +----------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
// so that we can use SIMD instruction (especially _mm256_shuffle_epi8) to do the summation.
// The accumulator is `u16`: see [`SAFE_U16_CODE_LEN`] for what that costs the caller.
#[inline]
pub fn sum_4bit_dist_table(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
) {
    assert!(n.is_multiple_of(BATCH_SIZE));
    assert!(dists.len() >= n);
    assert!(codes.len() >= n * code_len);
    assert!(dist_table.len() >= BATCH_SIZE * code_len);
    // A `u16` slice is also a valid `MaybeUninit<u16>` slice. The dispatched
    // kernels overwrite every output slot.
    let dists = unsafe {
        std::slice::from_raw_parts_mut(dists.as_mut_ptr().cast::<MaybeUninit<u16>>(), dists.len())
    };
    unsafe { sum_4bit_dist_table_uninit(n, code_len, codes, dist_table, dists) };
}

/// Sum a 4-bit distance table into potentially uninitialized output storage.
///
/// Every element in `dists[..n]` is initialized before this function returns.
///
/// # Safety
///
/// `n` must be a multiple of [`BATCH_SIZE`], `codes` must contain at least
/// `n * code_len` bytes, `dist_table` must contain at least
/// `BATCH_SIZE * code_len` bytes, and `dists` must contain at least `n` slots.
#[inline]
pub unsafe fn sum_4bit_dist_table_uninit(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [MaybeUninit<u16>],
) {
    debug_assert!(n.is_multiple_of(BATCH_SIZE));
    debug_assert!(dists.len() >= n);
    debug_assert!(codes.len() >= n * code_len);
    debug_assert!(dist_table.len() >= BATCH_SIZE * code_len);

    let (has_avx512bw, has_avx2) = x86_dist_table_features();
    match dist_table_backend(
        *SIMD_SUPPORT,
        cfg!(all(
            kernel_support = "avx512_dist_table",
            target_arch = "x86_64"
        )),
        has_avx512bw,
        has_avx2,
    ) {
        #[cfg(all(kernel_support = "avx512_dist_table", target_arch = "x86_64"))]
        DistTableBackend::Avx512 => {
            for i in (0..n).step_by(BATCH_SIZE) {
                let codes = &codes[i * code_len..(i + BATCH_SIZE) * code_len];
                unsafe {
                    sum_4bit_dist_table_32bytes_batch_avx512(
                        codes.as_ptr(),
                        codes.len(),
                        dist_table.as_ptr(),
                        dists[i..i + BATCH_SIZE].as_mut_ptr().cast::<u16>(),
                    )
                }
            }
        }
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx2 => unsafe {
            for i in (0..n).step_by(BATCH_SIZE) {
                sum_dist_table_32bytes_batch_avx2(
                    &codes[i * code_len..(i + BATCH_SIZE) * code_len],
                    dist_table,
                    &mut dists[i..i + BATCH_SIZE],
                )
            }
        },
        #[cfg(target_arch = "aarch64")]
        DistTableBackend::Neon => unsafe {
            for i in (0..n).step_by(BATCH_SIZE) {
                sum_dist_table_32bytes_batch_neon(
                    &codes[i * code_len..(i + BATCH_SIZE) * code_len],
                    dist_table,
                    &mut dists[i..i + BATCH_SIZE],
                )
            }
        },
        // `Scalar`.
        _ => {
            dists[..n].fill(MaybeUninit::new(0));
            // Every slot was initialized immediately above.
            let dists =
                unsafe { std::slice::from_raw_parts_mut(dists.as_mut_ptr().cast::<u16>(), n) };
            sum_4bit_dist_table_scalar(code_len, &codes[..n * code_len], dist_table, dists);
        }
    }
}

#[inline]
#[allow(unused)]
pub fn sum_4bit_dist_table_scalar(
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
) {
    let num_full_vectors = codes.len() / (BATCH_SIZE * code_len) * BATCH_SIZE;
    dists[..num_full_vectors].fill(0);

    for (vec_block_idx, blocks) in codes.chunks_exact(BATCH_SIZE * code_len).enumerate() {
        for (sub_vec_idx, block) in blocks.chunks_exact(BATCH_SIZE).enumerate() {
            let current_dist_table = &dist_table[sub_vec_idx * 2 * 16..(sub_vec_idx * 2 + 1) * 16];
            let next_dist_table =
                &dist_table[(sub_vec_idx * 2 + 1) * 16..(sub_vec_idx * 2 + 2) * 16];

            for j in 0..16 {
                let low_current_code = (block[j] & 0x0F) as usize;
                let high_current_code = (block[j] >> 4) as usize;
                let low_next_code = (block[j + 16] & 0x0F) as usize;
                let high_next_code = (block[j + 16] >> 4) as usize;

                let lower_id = vec_block_idx * BATCH_SIZE + PERM0[j];
                let higher_id = vec_block_idx * BATCH_SIZE + PERM0[j] + 16;
                dists[lower_id] = dists[lower_id]
                    .saturating_add(current_dist_table[low_current_code] as u16)
                    .saturating_add(next_dist_table[low_next_code] as u16);
                dists[higher_id] = dists[higher_id]
                    .saturating_add(current_dist_table[high_current_code] as u16)
                    .saturating_add(next_dist_table[high_next_code] as u16);
            }
        }
    }
}

/// [`sum_4bit_dist_table`] with an accumulator wide enough for any `code_len`:
/// the codes are summed in chunks of [`SAFE_U16_CODE_LEN`] through the same
/// `u16` kernels, and each chunk is widened into the output.
#[inline]
pub fn sum_4bit_dist_table_u32(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u32],
) {
    assert!(n.is_multiple_of(BATCH_SIZE));
    assert!(dists.len() >= n);
    assert!(codes.len() >= n * code_len);
    assert!(dist_table.len() >= BATCH_SIZE * code_len);
    // A `u32` slice is also a valid `MaybeUninit<u32>` slice. The chunk loop
    // overwrites every output slot.
    let dists = unsafe {
        std::slice::from_raw_parts_mut(dists.as_mut_ptr().cast::<MaybeUninit<u32>>(), dists.len())
    };
    unsafe { sum_4bit_dist_table_u32_uninit(n, code_len, codes, dist_table, dists) };
}

/// Sum a 4-bit distance table into potentially uninitialized `u32` output.
///
/// Every element in `dists[..n]` is initialized before this function returns.
///
/// # Safety
///
/// `n` must be a multiple of [`BATCH_SIZE`], `codes` must contain at least
/// `n * code_len` bytes, `dist_table` must contain at least
/// `BATCH_SIZE * code_len` bytes, and `dists` must contain at least `n` slots.
#[inline]
pub unsafe fn sum_4bit_dist_table_u32_uninit(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [MaybeUninit<u32>],
) {
    debug_assert!(n.is_multiple_of(BATCH_SIZE));
    debug_assert!(dists.len() >= n);
    debug_assert!(codes.len() >= n * code_len);
    debug_assert!(dist_table.len() >= BATCH_SIZE * code_len);

    if code_len == 0 {
        dists[..n].fill(MaybeUninit::new(0));
        return;
    }

    for i in (0..n).step_by(BATCH_SIZE) {
        let batch_codes = &codes[i * code_len..(i + BATCH_SIZE) * code_len];
        let mut sums = [0u32; BATCH_SIZE];
        for chunk_start in (0..code_len).step_by(SAFE_U16_CODE_LEN) {
            let chunk_end = (chunk_start + SAFE_U16_CODE_LEN).min(code_len);
            // Bytes are grouped by sub-vector, so one range slices both arrays.
            let bytes = chunk_start * BATCH_SIZE..chunk_end * BATCH_SIZE;
            let mut chunk_dists = [MaybeUninit::<u16>::uninit(); BATCH_SIZE];
            unsafe {
                sum_4bit_dist_table_uninit(
                    BATCH_SIZE,
                    chunk_end - chunk_start,
                    &batch_codes[bytes.clone()],
                    &dist_table[bytes],
                    &mut chunk_dists,
                );
            }
            // The dispatched kernel initialized every chunk output slot.
            let chunk_dists = unsafe {
                std::slice::from_raw_parts(chunk_dists.as_ptr().cast::<u16>(), BATCH_SIZE)
            };
            sums.iter_mut()
                .zip(chunk_dists.iter())
                .for_each(|(sum, chunk_dist)| *sum += *chunk_dist as u32);
        }
        dists[i..i + BATCH_SIZE]
            .iter_mut()
            .zip(sums.iter())
            .for_each(|(dist, sum)| {
                dist.write(*sum);
            });
    }
}

#[inline]
#[allow(unused)]
pub fn sum_4bit_dist_table_u16(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u16],
    dists: &mut [u32],
) {
    debug_assert!(n.is_multiple_of(BATCH_SIZE));
    debug_assert!(dists.len() >= n);
    debug_assert!(codes.len() >= n * code_len);
    sum_4bit_dist_table_u16_scalar(
        code_len,
        &codes[..n * code_len],
        dist_table,
        &mut dists[..n],
    );
}

#[inline]
pub fn transfer_4bit_dist_table_u16(dist_table: &[u16], hacc_dist_table: &mut Vec<u8>) {
    debug_assert!(dist_table.len().is_multiple_of(32));

    let num_tables = dist_table.len() / 16;
    hacc_dist_table.clear();
    hacc_dist_table.resize(dist_table.len() * 2, 0);

    for table_idx in 0..num_tables {
        let table = &dist_table[table_idx * 16..(table_idx + 1) * 16];
        let low_offset = (table_idx / 2) * 64 + (table_idx % 2) * 16;
        let high_offset = low_offset + 32;
        for (code, value) in table.iter().enumerate() {
            hacc_dist_table[low_offset + code] = *value as u8;
            hacc_dist_table[high_offset + code] = (value >> 8) as u8;
        }
    }
}

#[inline]
pub fn sum_4bit_hacc_dist_table(
    n: usize,
    code_len: usize,
    codes: &[u8],
    hacc_dist_table: &[u8],
    dists: &mut [u32],
) {
    assert!(n.is_multiple_of(BATCH_SIZE));
    assert!(dists.len() >= n);
    assert!(codes.len() >= n * code_len);
    assert!(hacc_dist_table.len() >= code_len * 64);
    // A `u32` slice is also a valid `MaybeUninit<u32>` slice. The dispatched
    // kernels overwrite every output slot.
    let dists = unsafe {
        std::slice::from_raw_parts_mut(dists.as_mut_ptr().cast::<MaybeUninit<u32>>(), dists.len())
    };
    unsafe { sum_4bit_hacc_dist_table_uninit(n, code_len, codes, hacc_dist_table, dists) };
}

/// Sum a high-accuracy 4-bit distance table into uninitialized output storage.
///
/// Every element in `dists[..n]` is initialized before this function returns.
///
/// # Safety
///
/// `n` must be a multiple of [`BATCH_SIZE`], `codes` must contain at least
/// `n * code_len` bytes, `hacc_dist_table` must contain at least
/// `code_len * 64` bytes, and `dists` must contain at least `n` slots.
#[inline]
pub unsafe fn sum_4bit_hacc_dist_table_uninit(
    n: usize,
    code_len: usize,
    codes: &[u8],
    hacc_dist_table: &[u8],
    dists: &mut [MaybeUninit<u32>],
) {
    debug_assert!(n.is_multiple_of(BATCH_SIZE));
    debug_assert!(dists.len() >= n);
    debug_assert!(codes.len() >= n * code_len);
    debug_assert!(hacc_dist_table.len() >= code_len * 64);

    // `false` for the AVX-512 kernel: this entry point has none, so an AVX-512
    // host with AVX2 takes AVX2 here. It has no NEON kernel either, and an
    // aarch64 tier is `Neon` or `None`, so the selector returns `Neon` or
    // `Scalar` and both land on the scalar `_` arm.
    let (has_avx512bw, has_avx2) = x86_dist_table_features();
    match dist_table_backend(*SIMD_SUPPORT, false, has_avx512bw, has_avx2) {
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx2 => {
            sum_4bit_hacc_dist_table_avx2(n, code_len, codes, hacc_dist_table, dists);
        }
        _ => {
            dists[..n].fill(MaybeUninit::new(0));
            // Every slot was initialized immediately above.
            let dists =
                unsafe { std::slice::from_raw_parts_mut(dists.as_mut_ptr().cast::<u32>(), n) };
            sum_4bit_hacc_dist_table_scalar(
                code_len,
                &codes[..n * code_len],
                hacc_dist_table,
                dists,
            );
        }
    }
}

#[inline]
#[allow(unused)]
pub fn sum_4bit_hacc_dist_table_scalar(
    code_len: usize,
    codes: &[u8],
    hacc_dist_table: &[u8],
    dists: &mut [u32],
) {
    let num_full_vectors = codes.len() / (BATCH_SIZE * code_len) * BATCH_SIZE;
    dists[..num_full_vectors].fill(0);

    for (vec_block_idx, blocks) in codes.chunks_exact(BATCH_SIZE * code_len).enumerate() {
        for (sub_vec_idx, block) in blocks.chunks_exact(BATCH_SIZE).enumerate() {
            let table_offset = sub_vec_idx * 64;
            let current_low = &hacc_dist_table[table_offset..table_offset + 16];
            let next_low = &hacc_dist_table[table_offset + 16..table_offset + 32];
            let current_high = &hacc_dist_table[table_offset + 32..table_offset + 48];
            let next_high = &hacc_dist_table[table_offset + 48..table_offset + 64];

            for j in 0..16 {
                let low_current_code = (block[j] & 0x0F) as usize;
                let high_current_code = (block[j] >> 4) as usize;
                let low_next_code = (block[j + 16] & 0x0F) as usize;
                let high_next_code = (block[j + 16] >> 4) as usize;

                let lower_id = vec_block_idx * BATCH_SIZE + PERM0[j];
                let higher_id = lower_id + 16;
                dists[lower_id] += ((current_high[low_current_code] as u32) << 8)
                    + current_low[low_current_code] as u32
                    + ((next_high[low_next_code] as u32) << 8)
                    + next_low[low_next_code] as u32;
                dists[higher_id] += ((current_high[high_current_code] as u32) << 8)
                    + current_low[high_current_code] as u32
                    + ((next_high[high_next_code] as u32) << 8)
                    + next_low[high_next_code] as u32;
            }
        }
    }
}

#[inline]
#[allow(unused)]
pub fn sum_4bit_dist_table_u16_scalar(
    code_len: usize,
    codes: &[u8],
    dist_table: &[u16],
    dists: &mut [u32],
) {
    let num_full_vectors = codes.len() / (BATCH_SIZE * code_len) * BATCH_SIZE;
    dists[..num_full_vectors].fill(0);

    for (vec_block_idx, blocks) in codes.chunks_exact(BATCH_SIZE * code_len).enumerate() {
        for (sub_vec_idx, block) in blocks.chunks_exact(BATCH_SIZE).enumerate() {
            let current_dist_table = &dist_table[sub_vec_idx * 2 * 16..(sub_vec_idx * 2 + 1) * 16];
            let next_dist_table =
                &dist_table[(sub_vec_idx * 2 + 1) * 16..(sub_vec_idx * 2 + 2) * 16];

            for j in 0..16 {
                let low_current_code = (block[j] & 0x0F) as usize;
                let high_current_code = (block[j] >> 4) as usize;
                let low_next_code = (block[j + 16] & 0x0F) as usize;
                let high_next_code = (block[j + 16] >> 4) as usize;

                let lower_id = vec_block_idx * BATCH_SIZE + PERM0[j];
                let higher_id = lower_id + 16;
                dists[lower_id] += current_dist_table[low_current_code] as u32
                    + next_dist_table[low_next_code] as u32;
                dists[higher_id] += current_dist_table[high_current_code] as u32
                    + next_dist_table[high_next_code] as u32;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn sum_4bit_hacc_dist_table_avx2(
    n: usize,
    code_len: usize,
    codes: &[u8],
    hacc_dist_table: &[u8],
    dists: &mut [MaybeUninit<u32>],
) {
    for i in (0..n).step_by(BATCH_SIZE) {
        let batch_codes = &codes[i * code_len..(i + BATCH_SIZE) * code_len];
        let batch_dists = &mut dists[i..i + BATCH_SIZE];

        if code_len == 0 {
            batch_dists.fill(MaybeUninit::new(0));
            continue;
        }

        for code_start in (0..code_len).step_by(SAFE_U16_CODE_LEN) {
            let code_end = (code_start + SAFE_U16_CODE_LEN).min(code_len);
            let code_range = code_start * BATCH_SIZE..code_end * BATCH_SIZE;
            let table_range = code_start * 64..code_end * 64;
            if code_start == 0 {
                unsafe {
                    sum_hacc_dist_table_32bytes_batch_avx2(
                        &batch_codes[code_range],
                        &hacc_dist_table[table_range],
                        batch_dists,
                    );
                }
            } else {
                let mut chunk_dists = [MaybeUninit::<u32>::uninit(); BATCH_SIZE];
                unsafe {
                    sum_hacc_dist_table_32bytes_batch_avx2(
                        &batch_codes[code_range],
                        &hacc_dist_table[table_range],
                        &mut chunk_dists,
                    );
                }
                // The kernel above initializes every temporary output slot.
                let chunk_dists = unsafe {
                    std::slice::from_raw_parts(chunk_dists.as_ptr().cast::<u32>(), BATCH_SIZE)
                };
                // The first code chunk initialized every output slot.
                let batch_dists = unsafe {
                    std::slice::from_raw_parts_mut(
                        batch_dists.as_mut_ptr().cast::<u32>(),
                        BATCH_SIZE,
                    )
                };
                batch_dists
                    .iter_mut()
                    .zip(chunk_dists.iter())
                    .for_each(|(dist, chunk_dist)| *dist += *chunk_dist);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unused)]
unsafe fn sum_hacc_dist_table_32bytes_batch_avx2(
    codes: &[u8],
    hacc_dist_table: &[u8],
    dists: &mut [MaybeUninit<u32>],
) {
    let low_mask = _mm256_set1_epi8(0x0f);
    let mut low_accu0 = _mm256_setzero_si256();
    let mut low_accu1 = _mm256_setzero_si256();
    let mut low_accu2 = _mm256_setzero_si256();
    let mut low_accu3 = _mm256_setzero_si256();
    let mut high_accu0 = _mm256_setzero_si256();
    let mut high_accu1 = _mm256_setzero_si256();
    let mut high_accu2 = _mm256_setzero_si256();
    let mut high_accu3 = _mm256_setzero_si256();

    for code_offset in (0..codes.len()).step_by(BATCH_SIZE) {
        let table_offset = code_offset * 2;
        let c = _mm256_loadu_si256(codes.as_ptr().add(code_offset) as *const __m256i);
        let lo = _mm256_and_si256(c, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        let low_lut =
            _mm256_loadu_si256(hacc_dist_table.as_ptr().add(table_offset) as *const __m256i);
        let low_res_lo = _mm256_shuffle_epi8(low_lut, lo);
        let low_res_hi = _mm256_shuffle_epi8(low_lut, hi);
        low_accu0 = _mm256_add_epi16(low_accu0, low_res_lo);
        low_accu1 = _mm256_add_epi16(low_accu1, _mm256_srli_epi16(low_res_lo, 8));
        low_accu2 = _mm256_add_epi16(low_accu2, low_res_hi);
        low_accu3 = _mm256_add_epi16(low_accu3, _mm256_srli_epi16(low_res_hi, 8));

        let high_lut =
            _mm256_loadu_si256(hacc_dist_table.as_ptr().add(table_offset + 32) as *const __m256i);
        let high_res_lo = _mm256_shuffle_epi8(high_lut, lo);
        let high_res_hi = _mm256_shuffle_epi8(high_lut, hi);
        high_accu0 = _mm256_add_epi16(high_accu0, high_res_lo);
        high_accu1 = _mm256_add_epi16(high_accu1, _mm256_srli_epi16(high_res_lo, 8));
        high_accu2 = _mm256_add_epi16(high_accu2, high_res_hi);
        high_accu3 = _mm256_add_epi16(high_accu3, _mm256_srli_epi16(high_res_hi, 8));
    }

    low_accu0 = _mm256_sub_epi16(low_accu0, _mm256_slli_epi16(low_accu1, 8));
    let low_dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(low_accu0, low_accu1, 0x21),
        _mm256_blend_epi32(low_accu0, low_accu1, 0xF0),
    );
    low_accu2 = _mm256_sub_epi16(low_accu2, _mm256_slli_epi16(low_accu3, 8));
    let low_dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(low_accu2, low_accu3, 0x21),
        _mm256_blend_epi32(low_accu2, low_accu3, 0xF0),
    );

    high_accu0 = _mm256_sub_epi16(high_accu0, _mm256_slli_epi16(high_accu1, 8));
    let high_dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(high_accu0, high_accu1, 0x21),
        _mm256_blend_epi32(high_accu0, high_accu1, 0xF0),
    );
    high_accu2 = _mm256_sub_epi16(high_accu2, _mm256_slli_epi16(high_accu3, 8));
    let high_dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(high_accu2, high_accu3, 0x21),
        _mm256_blend_epi32(high_accu2, high_accu3, 0xF0),
    );

    let low0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(low_dis0));
    let low1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(low_dis0, 1));
    let high0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(high_dis0));
    let high1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(high_dis0, 1));
    let res0 = _mm256_add_epi32(low0, _mm256_slli_epi32(high0, 8));
    let res1 = _mm256_add_epi32(low1, _mm256_slli_epi32(high1, 8));
    _mm256_storeu_si256(dists.as_mut_ptr() as *mut __m256i, res0);
    _mm256_storeu_si256(dists.as_mut_ptr().add(8) as *mut __m256i, res1);

    let low2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(low_dis1));
    let low3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(low_dis1, 1));
    let high2 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(high_dis1));
    let high3 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(high_dis1, 1));
    let res2 = _mm256_add_epi32(low2, _mm256_slli_epi32(high2, 8));
    let res3 = _mm256_add_epi32(low3, _mm256_slli_epi32(high3, 8));
    _mm256_storeu_si256(dists.as_mut_ptr().add(16) as *mut __m256i, res2);
    _mm256_storeu_si256(dists.as_mut_ptr().add(24) as *mut __m256i, res3);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
#[allow(unused)]
unsafe fn sum_dist_table_32bytes_batch_avx2(
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [MaybeUninit<u16>],
) {
    let mut c = _mm256_undefined_si256();
    let mut lo = _mm256_undefined_si256();
    let mut hi = _mm256_undefined_si256();
    let mut lut_vec = _mm256_undefined_si256();
    let mut res_lo = _mm256_undefined_si256();
    let mut res_hi = _mm256_undefined_si256();

    let mut accu0 = _mm256_setzero_si256();
    let mut accu1 = _mm256_setzero_si256();
    let mut accu2 = _mm256_setzero_si256();
    let mut accu3 = _mm256_setzero_si256();
    let low_mask = _mm256_set1_epi8(0x0f);

    for i in (0..codes.len()).step_by(64) {
        // load 32 * 2 codes (we pack 2 codes into 1 byte)
        c = _mm256_loadu_si256(codes.as_ptr().add(i) as *const __m256i);
        lut_vec = _mm256_loadu_si256(dist_table.as_ptr().add(i) as *const __m256i);

        // split the first 4 bits and the second 4 bits
        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        // lookup the lut
        res_lo = _mm256_shuffle_epi8(lut_vec, lo);
        res_hi = _mm256_shuffle_epi8(lut_vec, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));

        if i + 32 >= codes.len() {
            continue;
        }

        // load the left 32 bytes of codes and lut
        c = _mm256_loadu_si256(codes.as_ptr().add(i + 32) as *const __m256i);
        lut_vec = _mm256_loadu_si256(dist_table.as_ptr().add(i + 32) as *const __m256i);

        lo = _mm256_and_si256(c, low_mask);
        hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        res_lo = _mm256_shuffle_epi8(lut_vec, lo);
        res_hi = _mm256_shuffle_epi8(lut_vec, hi);

        accu0 = _mm256_add_epi16(accu0, res_lo);
        accu1 = _mm256_add_epi16(accu1, _mm256_srli_epi16(res_lo, 8));
        accu2 = _mm256_add_epi16(accu2, res_hi);
        accu3 = _mm256_add_epi16(accu3, _mm256_srli_epi16(res_hi, 8));
    }

    // merge the low 4 bits
    accu0 = _mm256_sub_epi16(accu0, _mm256_slli_epi16(accu1, 8));
    let dis0 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu0, accu1, 0x21),
        _mm256_blend_epi32(accu0, accu1, 0xF0),
    );
    _mm256_storeu_si256(dists.as_mut_ptr() as *mut __m256i, dis0);

    // merge the high 4 bits
    accu2 = _mm256_sub_epi16(accu2, _mm256_slli_epi16(accu3, 8));
    let dis1 = _mm256_add_epi16(
        _mm256_permute2f128_si256(accu2, accu3, 0x21),
        _mm256_blend_epi32(accu2, accu3, 0xF0),
    );

    _mm256_storeu_si256(dists.as_mut_ptr().add(16) as *mut __m256i, dis1);
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn sum_dist_table_32bytes_batch_neon(
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [MaybeUninit<u16>],
) {
    let low_mask = vdupq_n_u8(0x0f);

    // 8 accumulators: 4 per 128-bit "lane" (lo = bytes 0..16, hi = bytes 16..32 of each block)
    let mut accu0_lo = vdupq_n_u16(0);
    let mut accu1_lo = vdupq_n_u16(0);
    let mut accu2_lo = vdupq_n_u16(0);
    let mut accu3_lo = vdupq_n_u16(0);
    let mut accu0_hi = vdupq_n_u16(0);
    let mut accu1_hi = vdupq_n_u16(0);
    let mut accu2_hi = vdupq_n_u16(0);
    let mut accu3_hi = vdupq_n_u16(0);

    let codes_ptr = codes.as_ptr();
    let dt_ptr = dist_table.as_ptr();

    for i in (0..codes.len()).step_by(32) {
        // Process lo lane: bytes [i..i+16]
        let c_lo = vld1q_u8(codes_ptr.add(i));
        let lut_lo = vld1q_u8(dt_ptr.add(i));

        let lo_lo = vandq_u8(c_lo, low_mask);
        let hi_lo = vshrq_n_u8::<4>(c_lo);

        let res_lo_lo = vqtbl1q_u8(lut_lo, lo_lo);
        let res_hi_lo = vqtbl1q_u8(lut_lo, hi_lo);

        accu0_lo = vaddq_u16(accu0_lo, vreinterpretq_u16_u8(res_lo_lo));
        accu1_lo = vaddq_u16(accu1_lo, vshrq_n_u16::<8>(vreinterpretq_u16_u8(res_lo_lo)));
        accu2_lo = vaddq_u16(accu2_lo, vreinterpretq_u16_u8(res_hi_lo));
        accu3_lo = vaddq_u16(accu3_lo, vshrq_n_u16::<8>(vreinterpretq_u16_u8(res_hi_lo)));

        // Process hi lane: bytes [i+16..i+32]
        let c_hi = vld1q_u8(codes_ptr.add(i + 16));
        let lut_hi = vld1q_u8(dt_ptr.add(i + 16));

        let lo_hi = vandq_u8(c_hi, low_mask);
        let hi_hi = vshrq_n_u8::<4>(c_hi);

        let res_lo_hi = vqtbl1q_u8(lut_hi, lo_hi);
        let res_hi_hi = vqtbl1q_u8(lut_hi, hi_hi);

        accu0_hi = vaddq_u16(accu0_hi, vreinterpretq_u16_u8(res_lo_hi));
        accu1_hi = vaddq_u16(accu1_hi, vshrq_n_u16::<8>(vreinterpretq_u16_u8(res_lo_hi)));
        accu2_hi = vaddq_u16(accu2_hi, vreinterpretq_u16_u8(res_hi_hi));
        accu3_hi = vaddq_u16(accu3_hi, vshrq_n_u16::<8>(vreinterpretq_u16_u8(res_hi_hi)));
    }

    // Merge: clean even bytes by subtracting the odd-byte bleed
    accu0_lo = vsubq_u16(accu0_lo, vshlq_n_u16::<8>(accu1_lo));
    accu0_hi = vsubq_u16(accu0_hi, vshlq_n_u16::<8>(accu1_hi));

    // Cross-lane merge: add lo and hi lane accumulators
    // This is the NEON equivalent of AVX2's permute2f128 + blend + add
    let dis0_even = vaddq_u16(accu0_lo, accu0_hi);
    let dis0_odd = vaddq_u16(accu1_lo, accu1_hi);
    vst1q_u16(dists.as_mut_ptr().cast::<u16>(), dis0_even);
    vst1q_u16(dists.as_mut_ptr().add(8).cast::<u16>(), dis0_odd);

    // Same for hi-nibble accumulators (vectors 16..31)
    accu2_lo = vsubq_u16(accu2_lo, vshlq_n_u16::<8>(accu3_lo));
    accu2_hi = vsubq_u16(accu2_hi, vshlq_n_u16::<8>(accu3_hi));

    let dis1_even = vaddq_u16(accu2_lo, accu2_hi);
    let dis1_odd = vaddq_u16(accu3_lo, accu3_hi);
    vst1q_u16(dists.as_mut_ptr().add(16).cast::<u16>(), dis1_even);
    vst1q_u16(dists.as_mut_ptr().add(24).cast::<u16>(), dis1_odd);
}

/// Sum a quantized 4-bit PQ distance table over column-major codes.
///
/// `codes` holds `code_len` columns of `n` bytes: `codes[c * n + row]` packs
/// the codes of table `2 * c` (low nibble) and table `2 * c + 1` (high nibble)
/// for `row`, which is the layout PQ storage keeps its 4-bit codes in.
/// `dist_table` holds the `2 * code_len` tables of 16 entries each.
///
/// Sums use wrapping `u16` arithmetic, so the caller must scale the table such
/// that no row's sum exceeds `u16::MAX`.
pub fn sum_4bit_dist_table_transposed(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
) {
    // Checked products: a wrapped one would let short slices past the
    // unchecked pointer arithmetic of the kernels.
    assert!(
        n.checked_mul(code_len)
            .is_some_and(|needed| codes.len() >= needed),
        "codes has {} bytes, fewer than n * code_len = {n} * {code_len}",
        codes.len()
    );
    assert!(
        code_len
            .checked_mul(32)
            .is_some_and(|needed| dist_table.len() >= needed),
        "dist_table has {} entries, fewer than 32 * code_len = 32 * {code_len}",
        dist_table.len()
    );
    assert!(dists.len() >= n);
    // The SIMD kernels offset the code pointer by the first row before the
    // column loop, which with no columns would point past empty `codes`.
    if code_len == 0 {
        dists[..n].fill(0);
        return;
    }

    let (has_avx512bw, has_avx2) = x86_dist_table_features();
    let simd_len = match dist_table_backend(
        *SIMD_SUPPORT,
        cfg!(target_arch = "x86_64"),
        has_avx512bw,
        has_avx2,
    ) {
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx512 => {
            // The asserts above bound every full-width access; the tail is masked.
            unsafe { sum_transposed_avx512(n, code_len, codes, dist_table, dists) };
            n
        }
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx2 => sum_transposed_in_batches(n, |start| {
            // The asserts above bound every load and store of the batch.
            unsafe { sum_transposed_batch_avx2(start, n, code_len, codes, dist_table, dists) }
        }),
        #[cfg(target_arch = "aarch64")]
        DistTableBackend::Neon => sum_transposed_in_batches(n, |start| {
            // The asserts above bound every load and store of the batch.
            unsafe { sum_transposed_batch_neon(start, n, code_len, codes, dist_table, dists) }
        }),
        _ => 0,
    };
    sum_4bit_dist_table_transposed_scalar(simd_len, n, code_len, codes, dist_table, dists);
}

/// Rows summed per SIMD batch by [`sum_4bit_dist_table_transposed`], and rows
/// per batch reported by [`filter_4bit_dist_table_transposed`].
pub const TRANSPOSED_BATCH_SIZE: usize = 64;

/// Calls `batch(start)` on 64-row windows that cover all `n` rows and returns
/// how many leading rows they cover: `n`, or 0 when `n < 64`.
///
/// When `n` is not a multiple of 64 the last window starts at `n - 64`. It
/// overlaps rows an earlier window already stored, but a batch stores sums that
/// depend only on the codes, so it rewrites them with identical values. That is
/// cheaper than summing up to 63 rows in scalar code.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn sum_transposed_in_batches(n: usize, mut batch: impl FnMut(usize)) -> usize {
    if n < TRANSPOSED_BATCH_SIZE {
        return 0;
    }
    let aligned_len = n - n % TRANSPOSED_BATCH_SIZE;
    for start in (0..aligned_len).step_by(TRANSPOSED_BATCH_SIZE) {
        batch(start);
    }
    if aligned_len < n {
        batch(n - TRANSPOSED_BATCH_SIZE);
    }
    n
}

/// Scalar [`sum_4bit_dist_table_transposed`] for rows `start..n`.
fn sum_4bit_dist_table_transposed_scalar(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
) {
    let dists = &mut dists[start..n];
    dists.fill(0);
    for (column, tables) in dist_table.chunks_exact(32).take(code_len).enumerate() {
        let (low_table, high_table) = tables.split_at(16);
        let column = &codes[column * n + start..column * n + n];
        for (dist, &code) in dists.iter_mut().zip(column) {
            *dist = dist
                .wrapping_add(low_table[(code & 0x0f) as usize] as u16)
                .wrapping_add(high_table[(code >> 4) as usize] as u16);
        }
    }
}

/// Finds the rows of [`sum_4bit_dist_table_transposed`] whose sum is at most a
/// threshold, without storing every sum.
///
/// Rows are visited in batches of 64 in ascending order. For each batch with at
/// least one such row, `on_batch(start, mask, sums)` is called: bit `i` of
/// `mask` is set when row `start + i` has a sum at most the threshold, and
/// `sums[i]` holds that sum. Other entries of `sums` are unspecified. Every row
/// is reported at most once and no bit is set for a row at or past `n`, though
/// the last batch may start before a multiple of 64. The value `on_batch`
/// returns is the threshold for the batches after it, so a caller can tighten
/// it as candidates arrive.
///
/// ```
/// use lance_linalg::simd::dist_table::filter_4bit_dist_table_transposed;
///
/// // One column: row 0 selects low entry 1 (sum 1), row 1 high entry 1 (sum 5).
/// let codes = [0x01, 0x10];
/// let mut dist_table = [0u8; 32];
/// dist_table[1] = 1;
/// dist_table[16 + 1] = 5;
/// let mut rows = Vec::new();
/// filter_4bit_dist_table_transposed(2, 1, &codes, &dist_table, 3, |start, mut mask, sums| {
///     while mask != 0 {
///         let i = mask.trailing_zeros() as usize;
///         rows.push((start + i, sums[i]));
///         mask &= mask - 1;
///     }
///     3
/// });
/// assert_eq!(rows, [(0, 1)]);
/// ```
///
/// # Panics
///
/// If `codes` holds fewer than `n * code_len` bytes or `dist_table` fewer than
/// `32 * code_len` entries.
pub fn filter_4bit_dist_table_transposed(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    threshold: u16,
    mut on_batch: impl FnMut(usize, u64, &[u16; TRANSPOSED_BATCH_SIZE]) -> u16,
) {
    let (has_avx512bw, has_avx2) = x86_dist_table_features();
    let backend = dist_table_backend(
        *SIMD_SUPPORT,
        cfg!(target_arch = "x86_64"),
        has_avx512bw,
        has_avx2,
    );
    // `Avx512` needs `avx512bw`, which implies the `avx512f` its kernel also
    // uses, and `Avx2` needs `avx2`.
    unsafe {
        filter_transposed_with_backend(
            backend,
            n,
            code_len,
            codes,
            dist_table,
            threshold,
            &mut on_batch,
        )
    };
}

/// [`filter_4bit_dist_table_transposed`] on `backend`; a backend without a
/// kernel here, or fewer than 64 rows for the AVX2 and NEON batches, runs the
/// scalar path.
///
/// # Safety
///
/// The host must support `avx512f` and `avx512bw` when `backend` is `Avx512`
/// and `avx2` when it is `Avx2`.
unsafe fn filter_transposed_with_backend(
    backend: DistTableBackend,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    threshold: u16,
    on_batch: &mut impl FnMut(usize, u64, &[u16; TRANSPOSED_BATCH_SIZE]) -> u16,
) {
    // Checked products: a wrapped one would let short slices past the
    // unchecked pointer arithmetic of the kernels.
    assert!(
        n.checked_mul(code_len)
            .is_some_and(|needed| codes.len() >= needed),
        "codes has {} bytes, fewer than n * code_len = {n} * {code_len}",
        codes.len()
    );
    assert!(
        code_len
            .checked_mul(32)
            .is_some_and(|needed| dist_table.len() >= needed),
        "dist_table has {} entries, fewer than 32 * code_len = 32 * {code_len}",
        dist_table.len()
    );
    match backend {
        // The SIMD kernels offset the code pointer by the first row before the
        // column loop, which with no columns would point past empty `codes`.
        // Every row then sums to 0, which the scalar path reports.
        _ if code_len == 0 => {
            filter_transposed_scalar(n, code_len, codes, dist_table, threshold, on_batch)
        }
        // The asserts above bound every full-width access; the tail is masked.
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx512 => {
            filter_transposed_avx512(n, code_len, codes, dist_table, threshold, on_batch)
        }
        // The asserts above bound every load of a batch within `0..n`.
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx2 if n >= TRANSPOSED_BATCH_SIZE => {
            filter_transposed_in_batches(n, threshold, on_batch, |start, threshold, sums| {
                filter_transposed_batch_avx2(start, n, code_len, codes, dist_table, threshold, sums)
            })
        }
        #[cfg(target_arch = "aarch64")]
        DistTableBackend::Neon if n >= TRANSPOSED_BATCH_SIZE => {
            filter_transposed_in_batches(n, threshold, on_batch, |start, threshold, sums| {
                filter_transposed_batch_neon(start, n, code_len, codes, dist_table, threshold, sums)
            })
        }
        _ => filter_transposed_scalar(n, code_len, codes, dist_table, threshold, on_batch),
    }
}

/// Drives `batch(start, threshold, sums)` over 64-row windows covering all `n`
/// rows, `n >= 64`, as [`sum_transposed_in_batches`] does. The last window
/// overlaps rows an earlier one already reported, so their bits are cleared
/// before `on_batch` sees them.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn filter_transposed_in_batches(
    n: usize,
    mut threshold: u16,
    on_batch: &mut impl FnMut(usize, u64, &[u16; TRANSPOSED_BATCH_SIZE]) -> u16,
    mut batch: impl FnMut(usize, u16, &mut [u16; TRANSPOSED_BATCH_SIZE]) -> u64,
) {
    debug_assert!(n >= TRANSPOSED_BATCH_SIZE);
    let mut sums = [0u16; TRANSPOSED_BATCH_SIZE];
    let aligned_len = n - n % TRANSPOSED_BATCH_SIZE;
    for start in (0..aligned_len).step_by(TRANSPOSED_BATCH_SIZE) {
        let mask = batch(start, threshold, &mut sums);
        if mask != 0 {
            threshold = on_batch(start, mask, &sums);
        }
    }
    if aligned_len < n {
        let start = n - TRANSPOSED_BATCH_SIZE;
        let mask = batch(start, threshold, &mut sums) & (u64::MAX << (aligned_len - start));
        if mask != 0 {
            on_batch(start, mask, &sums);
        }
    }
}

/// Scalar [`filter_4bit_dist_table_transposed`].
fn filter_transposed_scalar(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    mut threshold: u16,
    on_batch: &mut impl FnMut(usize, u64, &[u16; TRANSPOSED_BATCH_SIZE]) -> u16,
) {
    let mut sums = [0u16; TRANSPOSED_BATCH_SIZE];
    for start in (0..n).step_by(TRANSPOSED_BATCH_SIZE) {
        let mask = filter_transposed_batch_scalar(
            start, n, code_len, codes, dist_table, threshold, &mut sums,
        );
        if mask != 0 {
            threshold = on_batch(start, mask, &sums);
        }
    }
}

/// Sums rows `start..min(start + 64, n)` into `sums` and returns the mask of
/// those at most `threshold`.
fn filter_transposed_batch_scalar(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    threshold: u16,
    sums: &mut [u16; TRANSPOSED_BATCH_SIZE],
) -> u64 {
    let end = n.min(start + TRANSPOSED_BATCH_SIZE);
    let sums = &mut sums[..end - start];
    sums.fill(0);
    for (column, tables) in dist_table.chunks_exact(32).take(code_len).enumerate() {
        let (low_table, high_table) = tables.split_at(16);
        let column = &codes[column * n + start..column * n + end];
        for (sum, &code) in sums.iter_mut().zip(column) {
            *sum = sum
                .wrapping_add(low_table[(code & 0x0f) as usize] as u16)
                .wrapping_add(high_table[(code >> 4) as usize] as u16);
        }
    }
    sums.iter().enumerate().fold(0, |mask, (i, &sum)| {
        mask | (u64::from(sum <= threshold) << i)
    })
}

/// Stores the sums of rows `start..start + 64` to `dists`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_transposed_batch_avx2(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
) {
    let rows = transposed_batch_sums_avx2(start, n, code_len, codes, dist_table);
    let out = dists.as_mut_ptr().add(start) as *mut __m256i;
    for (i, row) in rows.into_iter().enumerate() {
        _mm256_storeu_si256(out.add(i), row);
    }
}

/// Sums rows `start..start + 64` into four vectors of 16 rows each, in row
/// order. Each `u16` lane first accumulates a pair of adjacent byte lookups
/// (`even + 256 * odd`) plus a separate `odd` sum; the even sum is recovered at
/// the end, as in faiss' fast-scan kernels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn transposed_batch_sums_avx2(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
) -> [__m256i; 4] {
    let low_mask = _mm256_set1_epi8(0x0f);
    let mut pairs = [_mm256_setzero_si256(); 2];
    let mut odds = [_mm256_setzero_si256(); 2];
    let codes = codes.as_ptr().add(start);
    for column in 0..code_len {
        let tables = dist_table.as_ptr().add(column * 32);
        let low_table = _mm256_broadcastsi128_si256(_mm_loadu_si128(tables as *const __m128i));
        let high_table =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(tables.add(16) as *const __m128i));
        let column = codes.add(column * n);
        for half in 0..2 {
            let code = _mm256_loadu_si256(column.add(half * 32) as *const __m256i);
            let low = _mm256_shuffle_epi8(low_table, _mm256_and_si256(code, low_mask));
            let high = _mm256_shuffle_epi8(
                high_table,
                _mm256_and_si256(_mm256_srli_epi16(code, 4), low_mask),
            );
            // A carry out of the even byte of `low + high` is exact `u16`
            // arithmetic, so subtracting `256 * odds` still recovers the even sum.
            pairs[half] = _mm256_add_epi16(pairs[half], _mm256_add_epi16(low, high));
            odds[half] = _mm256_add_epi16(
                odds[half],
                _mm256_add_epi16(_mm256_srli_epi16(low, 8), _mm256_srli_epi16(high, 8)),
            );
        }
    }
    let mut rows = [_mm256_setzero_si256(); 4];
    for half in 0..2 {
        let evens = _mm256_sub_epi16(pairs[half], _mm256_slli_epi16(odds[half], 8));
        // Within each 128-bit lane, even rows are in `evens` and odd rows in
        // `odds`; interleave them back into row order.
        let first = _mm256_unpacklo_epi16(evens, odds[half]);
        let second = _mm256_unpackhi_epi16(evens, odds[half]);
        rows[2 * half] = _mm256_permute2x128_si256(first, second, 0x20);
        rows[2 * half + 1] = _mm256_permute2x128_si256(first, second, 0x31);
    }
    rows
}

/// Sums rows `start..start + 64` and returns the mask of those at most
/// `threshold`, writing their sums to `sums` only when the mask is non-zero.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_transposed_batch_avx2(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    threshold: u16,
    sums: &mut [u16; TRANSPOSED_BATCH_SIZE],
) -> u64 {
    let rows = transposed_batch_sums_avx2(start, n, code_len, codes, dist_table);
    let limit = _mm256_set1_epi16(threshold as i16);
    let mut at_most = [_mm256_setzero_si256(); 4];
    for (flags, &row) in at_most.iter_mut().zip(rows.iter()) {
        // AVX2 has no unsigned `u16` compare, but `min(row, limit) == row`
        // holds exactly when `row <= limit`.
        *flags = _mm256_cmpeq_epi16(_mm256_min_epu16(row, limit), row);
    }
    // Packing the all-ones or zero flags to bytes is exact, but it works per
    // 128-bit lane and leaves the quadwords in row order 0, 2, 1, 3.
    let low = _mm256_permute4x64_epi64(_mm256_packs_epi16(at_most[0], at_most[1]), 0xD8);
    let high = _mm256_permute4x64_epi64(_mm256_packs_epi16(at_most[2], at_most[3]), 0xD8);
    let mask = u64::from(_mm256_movemask_epi8(low) as u32)
        | (u64::from(_mm256_movemask_epi8(high) as u32) << 32);
    if mask != 0 {
        let out = sums.as_mut_ptr() as *mut __m256i;
        for (i, row) in rows.into_iter().enumerate() {
            _mm256_storeu_si256(out.add(i), row);
        }
    }
    mask
}

/// Sums all `n` rows of [`sum_4bit_dist_table_transposed`]. The last partial
/// window uses masked loads and stores, so no row is left for scalar code.
///
/// # Safety
///
/// The host must support `avx512f` and `avx512bw`, and the slices must satisfy
/// the bounds `sum_4bit_dist_table_transposed` asserts.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn sum_transposed_avx512(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
) {
    let mut start = 0;
    while start + 2 * TRANSPOSED_BATCH_SIZE <= n {
        sum_transposed_block_avx512::<2, false>(start, n, code_len, codes, dist_table, dists, 0);
        start += 2 * TRANSPOSED_BATCH_SIZE;
    }
    if start + TRANSPOSED_BATCH_SIZE <= n {
        sum_transposed_block_avx512::<1, false>(start, n, code_len, codes, dist_table, dists, 0);
        start += TRANSPOSED_BATCH_SIZE;
    }
    if start < n {
        let row_mask = (1u64 << (n - start)) - 1;
        sum_transposed_block_avx512::<1, true>(
            start, n, code_len, codes, dist_table, dists, row_mask,
        );
    }
}

/// Stores the sums of rows `start..start + 64 * REGS`, or only of the rows set
/// in `row_mask` when `MASKED`, to `dists`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn sum_transposed_block_avx512<const REGS: usize, const MASKED: bool>(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
    row_mask: u64,
) {
    let rows = transposed_block_sums_avx512::<REGS, MASKED>(
        start, n, code_len, codes, dist_table, row_mask,
    );
    for (reg, [first, second]) in rows.into_iter().enumerate() {
        let out = dists.as_mut_ptr().add(start + reg * 64);
        if MASKED {
            _mm512_mask_storeu_epi16(out.cast(), row_mask as u32, first);
            // With 32 or fewer tail rows `out + 32` can point past `dists`, and
            // forming that pointer is UB even when the store mask is empty.
            let high_mask = (row_mask >> 32) as u32;
            if high_mask != 0 {
                _mm512_mask_storeu_epi16(out.add(32).cast(), high_mask, second);
            }
        } else {
            _mm512_storeu_si512(out.cast(), first);
            _mm512_storeu_si512(out.add(32).cast(), second);
        }
    }
}

/// Sums rows `start..start + 64 * REGS`, or only the rows set in `row_mask`
/// when `MASKED`, into two vectors of 32 rows each per 64 rows, in row order;
/// masked-off rows sum as if their codes were zero. The even/odd `u16` split is
/// the one [`transposed_batch_sums_avx2`] uses, on 64 rows per register.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn transposed_block_sums_avx512<const REGS: usize, const MASKED: bool>(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    row_mask: u64,
) -> [[__m512i; 2]; REGS] {
    const { assert!(!MASKED || REGS == 1) };
    let low_mask = _mm512_set1_epi8(0x0f);
    let mut pairs = [_mm512_setzero_si512(); REGS];
    let mut odds = [_mm512_setzero_si512(); REGS];
    let codes = codes.as_ptr().add(start);
    for column in 0..code_len {
        let tables = dist_table.as_ptr().add(column * 32);
        let low_table = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.cast()));
        let high_table = _mm512_broadcast_i32x4(_mm_loadu_si128(tables.add(16).cast()));
        let column = codes.add(column * n);
        for reg in 0..REGS {
            // A zeroing masked load costs an extra uop on Intel, so only the
            // tail pays for it. Masked-off bytes are never read.
            let code = if MASKED {
                _mm512_maskz_loadu_epi8(row_mask, column.cast())
            } else {
                _mm512_loadu_si512(column.add(reg * 64).cast())
            };
            let low = _mm512_shuffle_epi8(low_table, _mm512_and_si512(code, low_mask));
            let high = _mm512_shuffle_epi8(
                high_table,
                _mm512_and_si512(_mm512_srli_epi16::<4>(code), low_mask),
            );
            // A carry out of the even byte of `low + high` is exact `u16`
            // arithmetic, so subtracting `256 * odds` still recovers the even sum.
            pairs[reg] = _mm512_add_epi16(pairs[reg], _mm512_add_epi16(low, high));
            odds[reg] = _mm512_add_epi16(
                odds[reg],
                _mm512_add_epi16(_mm512_srli_epi16::<8>(low), _mm512_srli_epi16::<8>(high)),
            );
        }
    }
    // Unpacking works within 128-bit lanes: lane `l` of `lo` holds rows
    // `16l..16l + 8` and lane `l` of `hi` rows `16l + 8..16l + 16`. These qword
    // indices (8 and up pick from `hi`) gather lanes 0-1 into `first` and lanes
    // 2-3 into `second`, both in row order.
    let first_half = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
    let second_half = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
    let mut rows = [[_mm512_setzero_si512(); 2]; REGS];
    for (reg, rows) in rows.iter_mut().enumerate() {
        let evens = _mm512_sub_epi16(pairs[reg], _mm512_slli_epi16::<8>(odds[reg]));
        let lo = _mm512_unpacklo_epi16(evens, odds[reg]);
        let hi = _mm512_unpackhi_epi16(evens, odds[reg]);
        *rows = [
            _mm512_permutex2var_epi64(lo, first_half, hi),
            _mm512_permutex2var_epi64(lo, second_half, hi),
        ];
    }
    rows
}

/// [`filter_4bit_dist_table_transposed`] over all `n` rows, two 64-row batches
/// per pass; the last partial batch is masked, so no row is left for scalar
/// code.
///
/// # Safety
///
/// The host must support `avx512f` and `avx512bw`, and the slices must satisfy
/// the bounds `filter_4bit_dist_table_transposed` asserts.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn filter_transposed_avx512(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    mut threshold: u16,
    on_batch: &mut impl FnMut(usize, u64, &[u16; TRANSPOSED_BATCH_SIZE]) -> u16,
) {
    let mut sums = [0u16; TRANSPOSED_BATCH_SIZE];
    let mut start = 0;
    while start + 2 * TRANSPOSED_BATCH_SIZE <= n {
        let rows =
            transposed_block_sums_avx512::<2, false>(start, n, code_len, codes, dist_table, 0);
        for (reg, [first, second]) in rows.into_iter().enumerate() {
            // The second batch is compared only after the first one's callback,
            // against the threshold that callback returned.
            threshold = emit_rows_at_most_avx512(
                start + reg * TRANSPOSED_BATCH_SIZE,
                first,
                second,
                u64::MAX,
                threshold,
                &mut sums,
                on_batch,
            );
        }
        start += 2 * TRANSPOSED_BATCH_SIZE;
    }
    if start + TRANSPOSED_BATCH_SIZE <= n {
        let [[first, second]] =
            transposed_block_sums_avx512::<1, false>(start, n, code_len, codes, dist_table, 0);
        threshold = emit_rows_at_most_avx512(
            start,
            first,
            second,
            u64::MAX,
            threshold,
            &mut sums,
            on_batch,
        );
        start += TRANSPOSED_BATCH_SIZE;
    }
    if start < n {
        let row_mask = (1u64 << (n - start)) - 1;
        let [[first, second]] = transposed_block_sums_avx512::<1, true>(
            start, n, code_len, codes, dist_table, row_mask,
        );
        emit_rows_at_most_avx512(
            start, first, second, row_mask, threshold, &mut sums, on_batch,
        );
    }
}

/// Calls `on_batch` for the rows of the batch at `start` that are in
/// `row_mask` and at most `threshold`, if any, and returns the threshold for
/// the next batch.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn emit_rows_at_most_avx512(
    start: usize,
    first: __m512i,
    second: __m512i,
    row_mask: u64,
    threshold: u16,
    sums: &mut [u16; TRANSPOSED_BATCH_SIZE],
    on_batch: &mut impl FnMut(usize, u64, &[u16; TRANSPOSED_BATCH_SIZE]) -> u16,
) -> u16 {
    let limit = _mm512_set1_epi16(threshold as i16);
    let mask = (u64::from(_mm512_cmple_epu16_mask(first, limit))
        | (u64::from(_mm512_cmple_epu16_mask(second, limit)) << 32))
        & row_mask;
    if mask == 0 {
        return threshold;
    }
    _mm512_storeu_si512(sums.as_mut_ptr().cast(), first);
    _mm512_storeu_si512(sums.as_mut_ptr().add(32).cast(), second);
    on_batch(start, mask, sums)
}

/// Stores the sums of rows `start..start + 64` to `dists`.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn sum_transposed_batch_neon(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    dists: &mut [u16],
) {
    let rows = transposed_batch_sums_neon(start, n, code_len, codes, dist_table);
    let out = dists.as_mut_ptr().add(start);
    for (i, row) in rows.into_iter().enumerate() {
        vst1q_u16(out.add(i * 8), row);
    }
}

/// Sums rows `start..start + 64` with widening adds into eight vectors of 8
/// rows each, in row order.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn transposed_batch_sums_neon(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
) -> [uint16x8_t; 8] {
    let low_mask = vdupq_n_u8(0x0f);
    let mut sums = [vdupq_n_u16(0); 8];
    let codes = codes.as_ptr().add(start);
    for column in 0..code_len {
        let tables = dist_table.as_ptr().add(column * 32);
        let low_table = vld1q_u8(tables);
        let high_table = vld1q_u8(tables.add(16));
        let column = codes.add(column * n);
        for quarter in 0..4 {
            let code = vld1q_u8(column.add(quarter * 16));
            let low = vqtbl1q_u8(low_table, vandq_u8(code, low_mask));
            let high = vqtbl1q_u8(high_table, vshrq_n_u8::<4>(code));
            let first = &mut sums[quarter * 2];
            *first = vaddw_u8(*first, vget_low_u8(low));
            *first = vaddw_u8(*first, vget_low_u8(high));
            let second = &mut sums[quarter * 2 + 1];
            *second = vaddw_high_u8(*second, low);
            *second = vaddw_high_u8(*second, high);
        }
    }
    sums
}

/// Sums rows `start..start + 64` and returns the mask of those at most
/// `threshold`, writing their sums to `sums` only when the mask is non-zero.
#[cfg(target_arch = "aarch64")]
unsafe fn filter_transposed_batch_neon(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[u8],
    threshold: u16,
    sums: &mut [u16; TRANSPOSED_BATCH_SIZE],
) -> u64 {
    let rows = transposed_batch_sums_neon(start, n, code_len, codes, dist_table);
    let limit = vdupq_n_u16(threshold);
    let mut at_most = [vdupq_n_u16(0); 8];
    let mut any = vdupq_n_u16(0);
    for (flags, &row) in at_most.iter_mut().zip(rows.iter()) {
        *flags = vcleq_u16(row, limit);
        any = vorrq_u16(any, *flags);
    }
    // Most batches have no row at or below the threshold once it is tight, so
    // they skip building the mask.
    if vmaxvq_u16(any) == 0 {
        return 0;
    }
    let bits = [1u8, 2, 4, 8, 16, 32, 64, 128];
    let bits = vld1_u8(bits.as_ptr());
    let mut mask = 0u64;
    for (i, (&flags, &row)) in at_most.iter().zip(rows.iter()).enumerate() {
        vst1q_u16(sums.as_mut_ptr().add(i * 8), row);
        let byte = vaddv_u8(vand_u8(vmovn_u16(flags), bits));
        mask |= u64::from(byte) << (i * 8);
    }
    mask
}

/// Exact `f32` sum of one row of a 4-bit PQ distance table over column-major
/// codes: the reference every exact `f32` kernel below reproduces.
///
/// `codes` and `dist_table` have the layout [`sum_4bit_dist_table_transposed`]
/// describes, with `f32` entries. Starting from `-0.0`, each column adds the
/// pair `low + high` its code byte selects, as one value and in column order:
/// `(0..code_len).fold(-0.0, |acc, c| acc + (low(c) + high(c)))`. Rust never
/// contracts or reassociates these adds.
///
/// ```
/// use lance_linalg::simd::dist_table::sum_4bit_dist_table_f32_row;
///
/// // One column: row 0 selects low entry 1, row 1 selects high entry 2.
/// let codes = [0x01, 0x20];
/// let mut dist_table = [0.0f32; 32];
/// dist_table[1] = 1.5;
/// dist_table[16 + 2] = 2.5;
/// assert_eq!(sum_4bit_dist_table_f32_row(2, 1, &codes, &dist_table, 1), 2.5);
/// ```
///
/// # Panics
///
/// If `row >= n`, `codes` holds fewer than `n * code_len` bytes or
/// `dist_table` fewer than `32 * code_len` entries.
// Inlined into per-row callers such as `PQDistCalculator::distance`, which the
// workspace builds without LTO.
#[inline]
pub fn sum_4bit_dist_table_f32_row(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    row: usize,
) -> f32 {
    assert!(row < n, "row {row} is out of range for {n} rows");
    assert_f32_table_bounds(n, code_len, codes, dist_table);
    f32_row_sum(n, codes, f32_tables(dist_table, code_len), row)
}

/// Exact `f32` sums of all `n` rows of a 4-bit PQ distance table over
/// column-major codes, writing row `r` to `dists[r]`.
///
/// Every backend performs, per row, the two IEEE-754 additions per column of
/// [`sum_4bit_dist_table_f32_row`] in the same order, from the same `-0.0`
/// start, so each sum is bit-identical to it. SIMD lanes each hold one row,
/// table lookups are bit copies and vector adds (`vaddps`, NEON `fadd`) round
/// like scalar ones (`addss`, `fadd`) under the same MXCSR or FPCR. The one
/// exception is NaN, which only non-finite tables produce: both paths then
/// return NaN, but not necessarily with the same sign or payload.
///
/// ```
/// use lance_linalg::simd::dist_table::sum_4bit_dist_table_f32_transposed;
///
/// // One column: row 0 selects low entry 1, row 1 selects high entry 2.
/// let codes = [0x01, 0x20];
/// let mut dist_table = [0.0f32; 32];
/// dist_table[1] = 1.5;
/// dist_table[16 + 2] = 2.5;
/// let mut dists = [0.0f32; 2];
/// sum_4bit_dist_table_f32_transposed(2, 1, &codes, &dist_table, &mut dists);
/// assert_eq!(dists, [1.5, 2.5]);
/// ```
///
/// # Panics
///
/// If `codes` holds fewer than `n * code_len` bytes, `dist_table` fewer than
/// `32 * code_len` entries or `dists` fewer than `n` slots.
pub fn sum_4bit_dist_table_f32_transposed(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    dists: &mut [f32],
) {
    let (has_avx512bw, has_avx2) = x86_dist_table_features();
    let backend = dist_table_backend(*SIMD_SUPPORT, true, has_avx512bw, has_avx2);
    // `Avx512` needs `avx512bw`, which implies the `avx512f` its kernel uses,
    // and `Avx2` needs `avx2`.
    unsafe { sum_f32_transposed_with_backend(backend, n, code_len, codes, dist_table, dists) };
}

/// Exact `f32` sums of the selected `rows`, writing the sum of `rows[i]` to
/// `dists[i]`; rows may repeat and come in any order.
///
/// Sums are bit-identical to [`sum_4bit_dist_table_f32_row`], for the reasons
/// [`sum_4bit_dist_table_f32_transposed`] gives. The SIMD backends score 16
/// rows at a time, so a caller rescoring candidates in order loses little by
/// passing 16 per call.
///
/// ```
/// use lance_linalg::simd::dist_table::sum_4bit_dist_table_f32_rows;
///
/// let codes = [0x01, 0x20];
/// let mut dist_table = [0.0f32; 32];
/// dist_table[1] = 1.5;
/// dist_table[16 + 2] = 2.5;
/// let mut dists = [0.0f32; 3];
/// sum_4bit_dist_table_f32_rows(2, 1, &codes, &dist_table, &[1, 0, 1], &mut dists);
/// assert_eq!(dists, [2.5, 1.5, 2.5]);
/// ```
///
/// # Panics
///
/// If any row is `>= n`, `codes` holds fewer than `n * code_len` bytes,
/// `dist_table` fewer than `32 * code_len` entries or `dists` fewer than
/// `rows.len()` slots.
pub fn sum_4bit_dist_table_f32_rows(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    rows: &[u32],
    dists: &mut [f32],
) {
    let (has_avx512bw, has_avx2) = x86_dist_table_features();
    let backend = dist_table_backend(*SIMD_SUPPORT, true, has_avx512bw, has_avx2);
    // Same backend contract as `sum_4bit_dist_table_f32_transposed`.
    unsafe { sum_f32_rows_with_backend(backend, n, code_len, codes, dist_table, rows, dists) };
}

/// Rows per `f32` lane group: one `zmm` register on AVX-512, two `ymm` on AVX2,
/// four `q` registers on NEON.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const F32_GROUP_ROWS: usize = 16;
/// Rows per block of the scalar pair-table path, and the fewest rows it takes:
/// building the 256 pairs of a column costs about as much as summing 256 rows.
const F32_PAIR_TABLE_BLOCK_ROWS: usize = 256;

/// Bounds the unchecked pointer arithmetic of the SIMD kernels, so the products
/// are checked: a wrapped one would let short slices pass.
#[inline]
fn assert_f32_table_bounds(n: usize, code_len: usize, codes: &[u8], dist_table: &[f32]) {
    let codes_needed = n.checked_mul(code_len);
    assert!(
        codes_needed.is_some_and(|needed| codes.len() >= needed),
        "codes has {} bytes, fewer than n * code_len = {n} * {code_len}",
        codes.len()
    );
    let table_needed = code_len.checked_mul(32);
    assert!(
        table_needed.is_some_and(|needed| dist_table.len() >= needed),
        "dist_table has {} entries, fewer than 32 * code_len = 32 * {code_len}",
        dist_table.len()
    );
}

/// One 32-entry table per code column: 16 entries for the low nibble, then 16
/// for the high one.
#[inline]
fn f32_tables(dist_table: &[f32], code_len: usize) -> &[[f32; 32]] {
    dist_table[..32 * code_len].as_chunks::<32>().0
}

#[inline]
fn f32_row_sum(n: usize, codes: &[u8], tables: &[[f32; 32]], row: usize) -> f32 {
    // An explicit fold rather than `Iterator::sum` pins the `-0.0` start.
    tables
        .iter()
        .enumerate()
        .fold(-0.0f32, |acc, (column, table)| {
            let code = codes[column * n + row];
            acc + (table[(code & 0x0f) as usize] + table[16 + (code >> 4) as usize])
        })
}

/// [`sum_4bit_dist_table_f32_transposed`] on `backend`; any other backend than
/// `Avx512`, `Avx2` and `Neon` runs the scalar path.
///
/// # Safety
///
/// The host must support `avx512f` when `backend` is `Avx512` and `avx2` when
/// it is `Avx2`.
unsafe fn sum_f32_transposed_with_backend(
    backend: DistTableBackend,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    dists: &mut [f32],
) {
    assert_f32_table_bounds(n, code_len, codes, dist_table);
    assert!(
        dists.len() >= n,
        "dists has {} slots, fewer than n = {n}",
        dists.len()
    );
    match backend {
        // The SIMD kernels offset the code pointer by the first row before the
        // column loop, which with no columns would point past empty `codes`.
        // Every sum is then `-0.0`, which the scalar path returns.
        _ if code_len == 0 => {
            sum_f32_transposed_scalar(n, codes, f32_tables(dist_table, code_len), dists)
        }
        // The asserts above bound every load and store of the kernels, which
        // need at least one full lane group.
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx512 if n >= F32_GROUP_ROWS => {
            sum_f32_transposed_avx512(n, code_len, codes, dist_table, dists)
        }
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx2 if n >= F32_GROUP_ROWS / 2 => {
            sum_f32_transposed_avx2(n, code_len, codes, dist_table, dists)
        }
        #[cfg(target_arch = "aarch64")]
        DistTableBackend::Neon if n >= F32_GROUP_ROWS => {
            sum_f32_transposed_neon(n, code_len, codes, dist_table, dists)
        }
        _ => sum_f32_transposed_scalar(n, codes, f32_tables(dist_table, code_len), dists),
    }
}

/// [`sum_4bit_dist_table_f32_rows`] on `backend`.
///
/// # Safety
///
/// As for [`sum_f32_transposed_with_backend`].
unsafe fn sum_f32_rows_with_backend(
    backend: DistTableBackend,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    rows: &[u32],
    dists: &mut [f32],
) {
    assert_f32_table_bounds(n, code_len, codes, dist_table);
    assert!(
        dists.len() >= rows.len(),
        "dists has {} slots, fewer than the {} rows",
        dists.len(),
        rows.len()
    );
    let max_row = rows.iter().max();
    assert!(
        max_row.is_none_or(|&row| (row as usize) < n),
        "row {max_row:?} is out of range for {n} rows"
    );
    let dists = &mut dists[..rows.len()];
    match backend {
        // The asserts above keep every gathered code byte, table load and store
        // in bounds.
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx512 => {
            sum_f32_rows_avx512(n, code_len, codes, dist_table, rows, dists)
        }
        #[cfg(target_arch = "x86_64")]
        DistTableBackend::Avx2 => sum_f32_rows_avx2(n, code_len, codes, dist_table, rows, dists),
        // A NEON kernel measured no faster than scalar on Graviton3: per
        // 16 rows it gathers the code bytes and deinterleaves every column's
        // tables, which costs more than its lookups save.
        _ => sum_f32_rows_scalar(n, codes, f32_tables(dist_table, code_len), rows, dists),
    }
}

/// Scalar [`sum_4bit_dist_table_f32_transposed`].
fn sum_f32_transposed_scalar(n: usize, codes: &[u8], tables: &[[f32; 32]], dists: &mut [f32]) {
    let dists = &mut dists[..n];
    if n < F32_PAIR_TABLE_BLOCK_ROWS {
        for (row, dist) in dists.iter_mut().enumerate() {
            *dist = f32_row_sum(n, codes, tables, row);
        }
        return;
    }
    // Every byte of a column selects one of 256 pair sums `low + high`. The
    // precomputed pair is the very value the per-row sum adds, so sums stay
    // bit-identical while each row needs one lookup per byte instead of two.
    let pair_tables = tables
        .iter()
        .map(|table| {
            std::array::from_fn::<f32, 256, _>(|byte| table[byte & 0x0f] + table[16 + (byte >> 4)])
        })
        .collect::<Vec<_>>();
    dists.fill(-0.0);
    for block_start in (0..n).step_by(F32_PAIR_TABLE_BLOCK_ROWS) {
        let block_end = (block_start + F32_PAIR_TABLE_BLOCK_ROWS).min(n);
        // A block's sums stay in L1 while every code column is added to them.
        let block = &mut dists[block_start..block_end];
        for (column, pair_table) in pair_tables.iter().enumerate() {
            let codes = &codes[column * n + block_start..column * n + block_end];
            for (dist, &code) in block.iter_mut().zip(codes) {
                *dist += pair_table[code as usize];
            }
        }
    }
}

/// Scalar [`sum_4bit_dist_table_f32_rows`]. Eight rows at a time overlap their
/// otherwise serial add chains.
fn sum_f32_rows_scalar(
    n: usize,
    codes: &[u8],
    tables: &[[f32; 32]],
    rows: &[u32],
    dists: &mut [f32],
) {
    const LANES: usize = 8;
    let score = |rows: &[u32], dists: &mut [f32]| {
        let mut acc = [-0.0f32; LANES];
        for (column, table) in tables.iter().enumerate() {
            let codes = &codes[column * n..(column + 1) * n];
            for (acc, &row) in acc.iter_mut().zip(rows) {
                let code = codes[row as usize];
                *acc += table[(code & 0x0f) as usize] + table[16 + (code >> 4) as usize];
            }
        }
        for (dist, acc) in dists.iter_mut().zip(acc) {
            *dist = acc;
        }
    };
    let (row_groups, row_tail) = rows.as_chunks::<LANES>();
    let (dist_groups, dist_tail) = dists.as_chunks_mut::<LANES>();
    for (rows, dists) in row_groups.iter().zip(dist_groups) {
        score(rows, dists);
    }
    if !row_tail.is_empty() {
        score(row_tail, dist_tail);
    }
}

/// The code bytes of `rows` in the column starting at `column`, row `i` in
/// byte `i`. Packing them in general-purpose registers avoids the failed
/// store-to-load forwarding of 16 byte stores read back as one vector.
///
/// # Safety
///
/// `column.add(row)` must be readable for every row.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn gather_code_bytes(column: *const u8, rows: &[usize; F32_GROUP_ROWS]) -> __m128i {
    let pack = |rows: &[usize]| {
        rows.iter()
            .rev()
            .fold(0u64, |word, &row| (word << 8) | *column.add(row) as u64)
    };
    _mm_set_epi64x(pack(&rows[8..]) as i64, pack(&rows[..8]) as i64)
}

/// `rows` as offsets into a column, padded to a full group with `rows[0]`, a
/// valid row whose extra sums are discarded.
#[cfg(target_arch = "x86_64")]
fn padded_group_rows(rows: &[u32]) -> [usize; F32_GROUP_ROWS] {
    let mut group = [rows[0] as usize; F32_GROUP_ROWS];
    for (slot, &row) in group.iter_mut().zip(rows) {
        *slot = row as usize;
    }
    group
}

/// `low[code & 15] + high[code >> 4]` per lane of 32-bit codes below 256.
/// `vpermps` reads only index bits 3:0, so the low nibble needs no mask and
/// the shifted code is exactly the high nibble.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
fn lookup_pair_avx512(code: __m512i, low_table: __m512, high_table: __m512) -> __m512 {
    _mm512_add_ps(
        _mm512_permutexvar_ps(code, low_table),
        _mm512_permutexvar_ps(_mm512_srli_epi32::<4>(code), high_table),
    )
}

/// Sums all `n` rows in 256-row blocks, then 16-row groups. A last partial
/// group is summed as the 16 rows ending at `n`, rewriting some sums of the
/// previous group with the same values.
///
/// # Safety
///
/// The host must support `avx512f`, `n >= 16`, and the slices must satisfy
/// the bounds `sum_f32_transposed_with_backend` asserts.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_f32_transposed_avx512(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    dists: &mut [f32],
) {
    const BLOCK_GROUPS: usize = 16;
    debug_assert!(n >= F32_GROUP_ROWS);
    let mut start = 0;
    while start + BLOCK_GROUPS * F32_GROUP_ROWS <= n {
        sum_f32_transposed_block_avx512::<BLOCK_GROUPS>(
            start, n, code_len, codes, dist_table, dists,
        );
        start += BLOCK_GROUPS * F32_GROUP_ROWS;
    }
    while start + F32_GROUP_ROWS <= n {
        sum_f32_transposed_block_avx512::<1>(start, n, code_len, codes, dist_table, dists);
        start += F32_GROUP_ROWS;
    }
    if start < n {
        sum_f32_transposed_block_avx512::<1>(
            n - F32_GROUP_ROWS,
            n,
            code_len,
            codes,
            dist_table,
            dists,
        );
    }
}

/// Sums rows `start..start + 16 * GROUPS`, one register of 16 rows per group,
/// each an independent add chain.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn sum_f32_transposed_block_avx512<const GROUPS: usize>(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    dists: &mut [f32],
) {
    let mut acc = [_mm512_set1_ps(-0.0); GROUPS];
    let codes = codes.as_ptr().add(start);
    for column in 0..code_len {
        let tables = dist_table.as_ptr().add(column * 32);
        let low_table = _mm512_loadu_ps(tables);
        let high_table = _mm512_loadu_ps(tables.add(16));
        let column = codes.add(column * n);
        for (group, acc) in acc.iter_mut().enumerate() {
            let code =
                _mm512_cvtepu8_epi32(_mm_loadu_si128(column.add(group * F32_GROUP_ROWS).cast()));
            *acc = _mm512_add_ps(*acc, lookup_pair_avx512(code, low_table, high_table));
        }
    }
    let out = dists.as_mut_ptr().add(start);
    for (group, acc) in acc.iter().enumerate() {
        _mm512_storeu_ps(out.add(group * F32_GROUP_ROWS), *acc);
    }
}

/// Sums `rows` 16 at a time, gathering each column's code bytes first.
///
/// # Safety
///
/// The host must support `avx512f`, and the slices must satisfy the bounds
/// `sum_f32_rows_with_backend` asserts, with `dists.len() == rows.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_f32_rows_avx512(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    rows: &[u32],
    dists: &mut [f32],
) {
    for (rows, dists) in rows
        .chunks(F32_GROUP_ROWS)
        .zip(dists.chunks_mut(F32_GROUP_ROWS))
    {
        let rows = padded_group_rows(rows);
        let mut acc = _mm512_set1_ps(-0.0);
        for column in 0..code_len {
            let tables = dist_table.as_ptr().add(column * 32);
            let code =
                _mm512_cvtepu8_epi32(gather_code_bytes(codes.as_ptr().add(column * n), &rows));
            acc = _mm512_add_ps(
                acc,
                lookup_pair_avx512(
                    code,
                    _mm512_loadu_ps(tables),
                    _mm512_loadu_ps(tables.add(16)),
                ),
            );
        }
        let mut sums = [0.0f32; F32_GROUP_ROWS];
        _mm512_storeu_ps(sums.as_mut_ptr(), acc);
        dists.copy_from_slice(&sums[..dists.len()]);
    }
}

/// `low[code & 15] + high[code >> 4]` per lane of 32-bit codes below 256,
/// with each 16-entry table split into two registers. `vpermps` reads index
/// bits 2:0 and `blendv` picks the upper half by the sign bit, so bit 3 of the
/// code (bit 7 for the high nibble) is shifted into it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn lookup_pair_avx2(code: __m256i, low_table: [__m256; 2], high_table: [__m256; 2]) -> __m256 {
    let low = _mm256_blendv_ps(
        _mm256_permutevar8x32_ps(low_table[0], code),
        _mm256_permutevar8x32_ps(low_table[1], code),
        _mm256_castsi256_ps(_mm256_slli_epi32::<28>(code)),
    );
    let high_code = _mm256_srli_epi32::<4>(code);
    let high = _mm256_blendv_ps(
        _mm256_permutevar8x32_ps(high_table[0], high_code),
        _mm256_permutevar8x32_ps(high_table[1], high_code),
        _mm256_castsi256_ps(_mm256_slli_epi32::<24>(code)),
    );
    _mm256_add_ps(low, high)
}

/// The two halves of each 16-entry table of `column`, low table first.
///
/// # Safety
///
/// `dist_table.add(32 * column)` must be readable for 32 entries.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_tables_avx2(dist_table: *const f32, column: usize) -> ([__m256; 2], [__m256; 2]) {
    let tables = dist_table.add(column * 32);
    (
        [_mm256_loadu_ps(tables), _mm256_loadu_ps(tables.add(8))],
        [
            _mm256_loadu_ps(tables.add(16)),
            _mm256_loadu_ps(tables.add(24)),
        ],
    )
}

/// Sums all `n` rows in 32-row blocks, then 8-row groups, ending with the 8
/// rows that end at `n` like [`sum_f32_transposed_avx512`].
///
/// # Safety
///
/// The host must support `avx2`, `n >= 8`, and the slices must satisfy the
/// bounds `sum_f32_transposed_with_backend` asserts.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_f32_transposed_avx2(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    dists: &mut [f32],
) {
    const BLOCK_GROUPS: usize = 4;
    const GROUP_ROWS: usize = F32_GROUP_ROWS / 2;
    debug_assert!(n >= GROUP_ROWS);
    let mut start = 0;
    while start + BLOCK_GROUPS * GROUP_ROWS <= n {
        sum_f32_transposed_block_avx2::<BLOCK_GROUPS>(start, n, code_len, codes, dist_table, dists);
        start += BLOCK_GROUPS * GROUP_ROWS;
    }
    while start + GROUP_ROWS <= n {
        sum_f32_transposed_block_avx2::<1>(start, n, code_len, codes, dist_table, dists);
        start += GROUP_ROWS;
    }
    if start < n {
        sum_f32_transposed_block_avx2::<1>(n - GROUP_ROWS, n, code_len, codes, dist_table, dists);
    }
}

/// Sums rows `start..start + 8 * GROUPS`, one register of 8 rows per group.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sum_f32_transposed_block_avx2<const GROUPS: usize>(
    start: usize,
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    dists: &mut [f32],
) {
    const GROUP_ROWS: usize = F32_GROUP_ROWS / 2;
    let mut acc = [_mm256_set1_ps(-0.0); GROUPS];
    let codes = codes.as_ptr().add(start);
    for column in 0..code_len {
        let (low_table, high_table) = load_tables_avx2(dist_table.as_ptr(), column);
        let column = codes.add(column * n);
        for (group, acc) in acc.iter_mut().enumerate() {
            let code = _mm256_cvtepu8_epi32(_mm_loadl_epi64(column.add(group * GROUP_ROWS).cast()));
            *acc = _mm256_add_ps(*acc, lookup_pair_avx2(code, low_table, high_table));
        }
    }
    let out = dists.as_mut_ptr().add(start);
    for (group, acc) in acc.iter().enumerate() {
        _mm256_storeu_ps(out.add(group * GROUP_ROWS), *acc);
    }
}

/// Sums `rows` 16 at a time in two 8-row registers, gathering each column's
/// code bytes first.
///
/// # Safety
///
/// As for [`sum_f32_rows_avx512`], with `avx2` in place of `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_f32_rows_avx2(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    rows: &[u32],
    dists: &mut [f32],
) {
    for (rows, dists) in rows
        .chunks(F32_GROUP_ROWS)
        .zip(dists.chunks_mut(F32_GROUP_ROWS))
    {
        let rows = padded_group_rows(rows);
        let mut acc = [_mm256_set1_ps(-0.0); 2];
        for column in 0..code_len {
            let (low_table, high_table) = load_tables_avx2(dist_table.as_ptr(), column);
            let code = gather_code_bytes(codes.as_ptr().add(column * n), &rows);
            let codes = [
                _mm256_cvtepu8_epi32(code),
                _mm256_cvtepu8_epi32(_mm_unpackhi_epi64(code, code)),
            ];
            for (acc, code) in acc.iter_mut().zip(codes) {
                *acc = _mm256_add_ps(*acc, lookup_pair_avx2(code, low_table, high_table));
            }
        }
        let mut sums = [0.0f32; F32_GROUP_ROWS];
        _mm256_storeu_ps(sums.as_mut_ptr(), acc[0]);
        _mm256_storeu_ps(sums.as_mut_ptr().add(F32_GROUP_ROWS / 2), acc[1]);
        dists.copy_from_slice(&sums[..dists.len()]);
    }
}

/// The entries of a 16-entry table that 16 nibbles select, as four vectors of
/// 4 rows in row order. `planes` is the table as `vld4q_u8` loads it: plane `b`
/// holds byte `b` of every little-endian entry, so one `tbl` per plane looks up
/// that byte for all 16 nibbles, and two rounds of zips reassemble the four
/// bytes of each row into one lane, bit for bit.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn lookup_neon(planes: uint8x16x4_t, nibbles: uint8x16_t) -> [float32x4_t; 4] {
    let byte0 = vqtbl1q_u8(planes.0, nibbles);
    let byte1 = vqtbl1q_u8(planes.1, nibbles);
    let byte2 = vqtbl1q_u8(planes.2, nibbles);
    let byte3 = vqtbl1q_u8(planes.3, nibbles);
    // Bytes 0 and 1, then 2 and 3, of rows 0..8 and 8..16 as `u16` lanes.
    let low_halves = [
        vreinterpretq_u16_u8(vzip1q_u8(byte0, byte1)),
        vreinterpretq_u16_u8(vzip2q_u8(byte0, byte1)),
    ];
    let high_halves = [
        vreinterpretq_u16_u8(vzip1q_u8(byte2, byte3)),
        vreinterpretq_u16_u8(vzip2q_u8(byte2, byte3)),
    ];
    [
        vreinterpretq_f32_u16(vzip1q_u16(low_halves[0], high_halves[0])),
        vreinterpretq_f32_u16(vzip2q_u16(low_halves[0], high_halves[0])),
        vreinterpretq_f32_u16(vzip1q_u16(low_halves[1], high_halves[1])),
        vreinterpretq_f32_u16(vzip2q_u16(low_halves[1], high_halves[1])),
    ]
}

/// `low[code & 15] + high[code >> 4]` for 16 code bytes, as four vectors of 4
/// rows in row order.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
fn lookup_pair_neon(
    code: uint8x16_t,
    low_table: uint8x16x4_t,
    high_table: uint8x16x4_t,
) -> [float32x4_t; 4] {
    let low = lookup_neon(low_table, vandq_u8(code, vdupq_n_u8(0x0f)));
    let high = lookup_neon(high_table, vshrq_n_u8::<4>(code));
    [
        vaddq_f32(low[0], high[0]),
        vaddq_f32(low[1], high[1]),
        vaddq_f32(low[2], high[2]),
        vaddq_f32(low[3], high[3]),
    ]
}

/// Sums all `n` rows in 32-row blocks, then 16-row groups, ending with the 16
/// rows that end at `n` like the x86 kernels.
///
/// # Safety
///
/// `n >= 16`, and the slices must satisfy the bounds
/// `sum_f32_transposed_with_backend` asserts.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_f32_transposed_neon(
    n: usize,
    code_len: usize,
    codes: &[u8],
    dist_table: &[f32],
    dists: &mut [f32],
) {
    // Two groups hold 8 accumulators next to the 8 table planes and the
    // lookup temporaries, within the 32 vector registers.
    const BLOCK_GROUPS: usize = 2;
    debug_assert!(n >= F32_GROUP_ROWS);
    // Deinterleaving once here rather than per block and column: on Graviton3
    // a `vld4q_u8` costs about as much as the lookups of a 16-row group.
    let planes: Vec<[uint8x16x4_t; 2]> = (0..code_len)
        .map(|column| {
            let tables = dist_table.as_ptr().add(column * 32);
            [vld4q_u8(tables.cast()), vld4q_u8(tables.add(16).cast())]
        })
        .collect();
    let mut start = 0;
    while start + BLOCK_GROUPS * F32_GROUP_ROWS <= n {
        sum_f32_transposed_block_neon::<BLOCK_GROUPS>(start, n, codes, &planes, dists);
        start += BLOCK_GROUPS * F32_GROUP_ROWS;
    }
    while start + F32_GROUP_ROWS <= n {
        sum_f32_transposed_block_neon::<1>(start, n, codes, &planes, dists);
        start += F32_GROUP_ROWS;
    }
    if start < n {
        sum_f32_transposed_block_neon::<1>(n - F32_GROUP_ROWS, n, codes, &planes, dists);
    }
}

/// Sums rows `start..start + 16 * GROUPS` over the columns whose low and high
/// table `planes` hold, four registers of 4 rows per group, each an
/// independent add chain.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sum_f32_transposed_block_neon<const GROUPS: usize>(
    start: usize,
    n: usize,
    codes: &[u8],
    planes: &[[uint8x16x4_t; 2]],
    dists: &mut [f32],
) {
    let mut acc = [[vdupq_n_f32(-0.0); 4]; GROUPS];
    let codes = codes.as_ptr().add(start);
    for (column, &[low_table, high_table]) in planes.iter().enumerate() {
        let column = codes.add(column * n);
        for (group, acc) in acc.iter_mut().enumerate() {
            let code = vld1q_u8(column.add(group * F32_GROUP_ROWS));
            let pairs = lookup_pair_neon(code, low_table, high_table);
            for (acc, pair) in acc.iter_mut().zip(pairs) {
                *acc = vaddq_f32(*acc, pair);
            }
        }
    }
    let out = dists.as_mut_ptr().add(start);
    for (group, acc) in acc.iter().enumerate() {
        for (quarter, acc) in acc.iter().enumerate() {
            vst1q_f32(out.add(group * F32_GROUP_ROWS + quarter * 4), *acc);
        }
    }
}

// We implement the AVX512 version in C because AVX512 is not stable yet in Rust,
// implement it in Rust once we upgrade rust to 1.89.0.
unsafe extern "C" {
    #[cfg(all(kernel_support = "avx512_dist_table", target_arch = "x86_64"))]
    pub fn sum_4bit_dist_table_32bytes_batch_avx512(
        codes: *const u8,
        code_length: usize,
        dist_table: *const u8,
        dists: *mut u16,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `avx512_kernel` false covers every caller that reaches it: the hacc entry
    /// point, which has no AVX-512 kernel at all,
    /// `sum_4bit_dist_table_transposed` on every non-x86_64 build, and
    /// `sum_4bit_dist_table_uninit` wherever its
    /// `cfg!(all(kernel_support = "avx512_dist_table", target_arch = "x86_64"))`
    /// is false, which is every non-x86_64 build as well as an x86_64 one that
    /// compiled no AVX-512 C.
    #[rstest::rstest]
    // An AVX-512 host whose entry point has an AVX-512 kernel and `avx512bw`
    // present takes it.
    #[case::avx512_ready(SimdSupport::Avx512, true, true, true, DistTableBackend::Avx512)]
    #[case::avx512fp16_ready(SimdSupport::Avx512FP16, true, true, true, DistTableBackend::Avx512)]
    // The two ways an AVX-512 host with AVX2 misses the AVX-512 arm. Both must
    // reach AVX2 rather than scalar.
    #[case::avx512_no_bw(SimdSupport::Avx512, true, false, true, DistTableBackend::Avx2)]
    #[case::avx512_kernel_absent(SimdSupport::Avx512, false, true, true, DistTableBackend::Avx2)]
    #[case::avx512fp16_no_bw(SimdSupport::Avx512FP16, true, false, true, DistTableBackend::Avx2)]
    #[case::avx512fp16_kernel_absent(
        SimdSupport::Avx512FP16,
        false,
        true,
        true,
        DistTableBackend::Avx2
    )]
    #[case::avx2_tier(SimdSupport::Avx2, true, false, true, DistTableBackend::Avx2)]
    // The tier alone must not authorize the AVX2 kernel: it is
    // `#[target_feature(enable = "avx2")]`. No host produces this pair, since
    // `cpu.rs` selects `Avx2` only when the same `avx2` detection returned true,
    // so this row pins the selector's contract rather than a real configuration.
    #[case::avx2_tier_without_the_feature(
        SimdSupport::Avx2,
        false,
        false,
        false,
        DistTableBackend::Scalar
    )]
    // Without AVX2 there is nothing to fall back to; this row pins the selector.
    #[case::avx512_no_avx2(SimdSupport::Avx512, false, false, false, DistTableBackend::Scalar)]
    #[case::avx_fma(SimdSupport::AvxFma, false, false, false, DistTableBackend::Scalar)]
    #[case::avx(SimdSupport::Avx, false, false, false, DistTableBackend::Scalar)]
    // An `Avx` tier that reports AVX2. `cpu.rs` notes that every shipping AVX2
    // part has FMA, so no shipping part produces this pair, though a masked or
    // feature-forced build can: the tier alone leaves it on scalar, even though
    // the kernel needs no FMA.
    #[case::avx_tier_with_avx2(SimdSupport::Avx, false, false, true, DistTableBackend::Scalar)]
    // `Sse` is a variant the x86 ladder never produces, so this row pins the
    // selector rather than a real host.
    #[case::sse(SimdSupport::Sse, false, false, false, DistTableBackend::Scalar)]
    #[case::none(SimdSupport::None, false, false, false, DistTableBackend::Scalar)]
    #[case::neon(SimdSupport::Neon, false, false, false, DistTableBackend::Neon)]
    // loongarch64 has LSX and LASX kernels elsewhere in this crate but none for
    // this table, so both tiers belong on the scalar route.
    #[case::lsx(SimdSupport::Lsx, false, false, false, DistTableBackend::Scalar)]
    #[case::lasx(SimdSupport::Lasx, false, false, false, DistTableBackend::Scalar)]
    // The exact `f32` entry points pass `avx512_kernel = true` on every build,
    // so a non-AVX-512 tier must ignore it and pick what it would without it.
    #[case::f32_entry_neon(SimdSupport::Neon, true, false, false, DistTableBackend::Neon)]
    #[case::f32_entry_avx2(SimdSupport::Avx2, true, false, true, DistTableBackend::Avx2)]
    #[case::f32_entry_avx_fma(SimdSupport::AvxFma, true, false, false, DistTableBackend::Scalar)]
    #[case::f32_entry_lasx(SimdSupport::Lasx, true, false, false, DistTableBackend::Scalar)]
    #[case::f32_entry_none(SimdSupport::None, true, false, false, DistTableBackend::Scalar)]
    fn dist_table_backend_follows_the_tier_ladder(
        #[case] support: SimdSupport,
        #[case] avx512_kernel: bool,
        #[case] has_avx512bw: bool,
        #[case] has_avx2: bool,
        #[case] expected: DistTableBackend,
    ) {
        assert_eq!(
            dist_table_backend(support, avx512_kernel, has_avx512bw, has_avx2),
            expected
        );
    }

    #[test]
    fn test_perm0_inverse_matches_perm0() {
        for (idx, &value) in PERM0.iter().enumerate() {
            assert_eq!(PERM0_INVERSE[value], idx);
        }
    }

    #[test]
    fn test_sum_4bit_dist_table_basic() {
        // we have 32 vectors
        let n = 32;

        // each code is 2 bytes (16 dim), so code_len = 2
        let code_len = 2;

        let codes = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, // codes[0..8]
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, // codes[8..16]
            0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, // codes[16..24]
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, // codes[24..32]
        ];
        let codes = codes.repeat(n * code_len / codes.len());

        let mut dist_table = vec![0u8; 16 * 4];
        for (i, dist) in dist_table.iter_mut().enumerate() {
            *dist = (i % 16 + 1) as u8;
        }

        // Test the function
        let mut dists = vec![0u16; n];
        sum_4bit_dist_table(n, code_len, &codes, &dist_table, &mut dists);

        // Compare with reference implementation
        let mut expected_dists = vec![0u16; n];
        sum_4bit_dist_table_scalar(code_len, &codes, &dist_table, &mut expected_dists);

        assert_eq!(dists, expected_dists);
        // the vector 1's code is the low 4bits of codes[PERM0_INVERSE[1]] = codes[2],
        // the first 4 bits are the low 4 bits of codes[2], so it's 0x6,
        // the second 4 bits are the low 4 bits of codes[2 + 16], so it's 0xb,
        // the third 4 bits are the same as the first 4 bits, so it's 0x6,
        // the fourth 4 bits are the same as the second 4 bits, so it's 0xb,

        // so the distance is 2 * (dist_table[0x6] + dist_table[0xb + 16]) = 2*(7 + 12) = 38
        assert_eq!(dists[1], 38);
    }

    #[test]
    fn test_sum_4bit_dist_table_overwrites_output() {
        let n = BATCH_SIZE;
        let code_len = 16;
        let codes = vec![0x12; n * code_len];
        let dist_table = vec![1u8; BATCH_SIZE * code_len];

        let mut expected = vec![u16::MAX; n];
        sum_4bit_dist_table_scalar(code_len, &codes, &dist_table, &mut expected);

        let mut actual = vec![u16::MAX; n];
        sum_4bit_dist_table(n, code_len, &codes, &dist_table, &mut actual);

        assert_eq!(actual, expected);
        assert!(actual.iter().all(|dist| *dist != u16::MAX));
    }

    #[test]
    fn test_sum_4bit_dist_table_u16_basic() {
        let n = BATCH_SIZE;
        let code_len = 2;
        let codes = [
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66,
            0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x12, 0x34, 0x56, 0x78,
            0x9a, 0xbc, 0xde, 0xf0,
        ];
        let codes = codes.repeat(n * code_len / codes.len());
        let dist_table: Vec<u16> = (0..16 * 4).map(|idx| (idx % 16 + 1) as u16).collect();

        let mut dists = vec![0u32; n];
        sum_4bit_dist_table_u16(n, code_len, &codes, &dist_table, &mut dists);

        assert_eq!(dists[1], 38);
    }

    #[test]
    fn test_transfer_4bit_dist_table_u16_layout() {
        let dist_table: Vec<u16> = (0..32).map(|idx| 0x1200 + idx as u16).collect();
        let mut hacc_dist_table = Vec::new();
        transfer_4bit_dist_table_u16(&dist_table, &mut hacc_dist_table);

        assert_eq!(hacc_dist_table.len(), 64);
        for code in 0..16 {
            assert_eq!(hacc_dist_table[code], dist_table[code] as u8);
            assert_eq!(hacc_dist_table[16 + code], dist_table[16 + code] as u8);
            assert_eq!(hacc_dist_table[32 + code], (dist_table[code] >> 8) as u8);
            assert_eq!(
                hacc_dist_table[48 + code],
                (dist_table[16 + code] >> 8) as u8
            );
        }
    }

    #[test]
    fn test_sum_4bit_dist_table_u16_matches_reference_multi_batch() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);

        for code_len in [1, 3, 16, 191, 192, 1024] {
            let n = BATCH_SIZE * 4;
            let codes: Vec<u8> = (0..n * code_len).map(|_| rng.random::<u8>()).collect();
            let dist_table: Vec<u16> = (0..BATCH_SIZE * code_len)
                .map(|_| rng.random::<u16>())
                .collect();

            let mut expected = vec![0u32; n];
            sum_4bit_dist_table_u16_scalar(code_len, &codes, &dist_table, &mut expected);

            let mut actual = vec![u32::MAX; n];
            sum_4bit_dist_table_u16(n, code_len, &codes, &dist_table, &mut actual);

            assert_eq!(
                actual,
                expected,
                "u16 dist-table mismatch for code_len={} (DIM={})",
                code_len,
                code_len * 8,
            );
        }
    }

    #[test]
    fn test_sum_4bit_hacc_dist_table_matches_u16_reference_multi_batch() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(101);

        for code_len in [1, 3, 16, 191, 192, 1024] {
            let n = BATCH_SIZE * 4;
            let codes: Vec<u8> = (0..n * code_len).map(|_| rng.random::<u8>()).collect();
            let dist_table: Vec<u16> = (0..BATCH_SIZE * code_len)
                .map(|_| rng.random::<u16>())
                .collect();

            let mut hacc_dist_table = Vec::new();
            transfer_4bit_dist_table_u16(&dist_table, &mut hacc_dist_table);

            let mut expected = vec![0u32; n];
            sum_4bit_dist_table_u16_scalar(code_len, &codes, &dist_table, &mut expected);

            let mut actual = vec![u32::MAX; n];
            sum_4bit_hacc_dist_table(n, code_len, &codes, &hacc_dist_table, &mut actual);

            assert_eq!(
                actual,
                expected,
                "hacc dist-table mismatch for code_len={} (DIM={})",
                code_len,
                code_len * 8,
            );
        }
    }

    /// SIMD against scalar over the range the `u16` entry point is contracted
    /// for; longer codes belong to [`sum_4bit_dist_table_u32`].
    #[test]
    fn test_simd_matches_scalar_varied_dimensions() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // code_len = dim / 8 for 1-bit quantization; 128 is DIM=1024.
        for code_len in [1, 2, 3, 16, 95, 96, 128] {
            let n = BATCH_SIZE;

            let codes: Vec<u8> = (0..n * code_len).map(|_| rng.random::<u8>()).collect();
            // `quantize_dist_table_into` spans the whole u8 range, so must this.
            let dist_table: Vec<u8> = (0..BATCH_SIZE * code_len)
                .map(|_| rng.random::<u8>())
                .collect();

            let mut expected = vec![0u16; n];
            sum_4bit_dist_table_scalar(code_len, &codes, &dist_table, &mut expected);

            let mut actual = vec![0u16; n];
            sum_4bit_dist_table(n, code_len, &codes, &dist_table, &mut actual);

            assert_eq!(
                actual,
                expected,
                "SIMD and scalar mismatch for code_len={} (DIM={})",
                code_len,
                code_len * 8,
            );
        }
    }

    /// Test with multiple batches to verify accumulation across batch boundaries.
    #[test]
    fn test_simd_matches_scalar_multi_batch() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);

        for code_len in [1, 3, 16, 95, 96, 128] {
            let n = BATCH_SIZE * 10; // 320 vectors = 10 batches

            let codes: Vec<u8> = (0..n * code_len).map(|_| rng.random::<u8>()).collect();
            let dist_table: Vec<u8> = (0..BATCH_SIZE * code_len)
                .map(|_| rng.random::<u8>())
                .collect();

            let mut expected = vec![0u16; n];
            sum_4bit_dist_table_scalar(code_len, &codes, &dist_table, &mut expected);

            let mut actual = vec![0u16; n];
            sum_4bit_dist_table(n, code_len, &codes, &dist_table, &mut actual);

            assert_eq!(
                actual,
                expected,
                "SIMD and scalar mismatch for multi-batch code_len={} (DIM={}, n={})",
                code_len,
                code_len * 8,
                n,
            );
        }
    }
    /// Exact reference: the `u16`-table scalar kernel already sums into `u32`.
    fn reference_u32_sums(n: usize, code_len: usize, codes: &[u8], dist_table: &[u8]) -> Vec<u32> {
        let dist_table: Vec<u16> = dist_table.iter().map(|value| *value as u16).collect();
        let mut expected = vec![0u32; n];
        sum_4bit_dist_table_u16_scalar(
            code_len,
            &codes[..n * code_len],
            &dist_table,
            &mut expected,
        );
        expected
    }

    #[test]
    fn test_safe_u16_code_len_is_the_overflow_bound() {
        assert!(2 * SAFE_U16_CODE_LEN * u8::MAX as usize <= u16::MAX as usize);
        assert!(2 * (SAFE_U16_CODE_LEN + 1) * u8::MAX as usize > u16::MAX as usize);
    }

    #[test]
    fn test_sum_4bit_dist_table_u32_matches_reference_full_range() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7157);

        for code_len in [1, 3, 16, 128, 129, 191, 192, 300, 512, 1024] {
            let n = BATCH_SIZE * 4;
            let codes: Vec<u8> = (0..n * code_len).map(|_| rng.random::<u8>()).collect();
            let dist_table: Vec<u8> = (0..BATCH_SIZE * code_len)
                .map(|_| rng.random::<u8>())
                .collect();

            let expected = reference_u32_sums(n, code_len, &codes, &dist_table);

            let mut actual = vec![u32::MAX; n];
            sum_4bit_dist_table_u32(n, code_len, &codes, &dist_table, &mut actual);

            assert_eq!(
                actual,
                expected,
                "u32 dist-table mismatch for code_len={} (DIM={})",
                code_len,
                code_len * 8,
            );
        }
    }

    /// https://github.com/lance-format/lance/issues/7157: at DIM=4096 a
    /// full-range table sums to four times what a `u16` holds.
    #[test]
    fn test_sum_4bit_dist_table_u32_stays_exact_past_u16_range() {
        let code_len = 512;
        let n = BATCH_SIZE;
        let codes = vec![0u8; n * code_len];
        let dist_table = vec![u8::MAX; BATCH_SIZE * code_len];
        let expected = 2 * code_len as u32 * u8::MAX as u32;
        assert!(expected > u16::MAX as u32);

        let mut wide = vec![0u32; n];
        sum_4bit_dist_table_u32(n, code_len, &codes, &dist_table, &mut wide);
        assert!(wide.iter().all(|dist| *dist == expected));
    }

    /// The ex-code FastScan LUT runs codes far longer than
    /// [`SAFE_U16_CODE_LEN`] through the `u16` kernels, staying in range by
    /// capping the table instead. Model that cap so those lengths keep coverage.
    #[test]
    fn test_simd_matches_scalar_capped_table_long_code_len() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(8192);

        for code_len in [191, 192, 512, 1024, 8192] {
            let n = BATCH_SIZE;
            let max_val = (u16::MAX as usize / (2 * code_len)).min(u8::MAX as usize) as u8;
            assert!(2 * code_len * max_val as usize <= u16::MAX as usize);

            let codes: Vec<u8> = (0..n * code_len).map(|_| rng.random::<u8>()).collect();
            let dist_table: Vec<u8> = (0..BATCH_SIZE * code_len)
                .map(|_| rng.random_range(0..=max_val))
                .collect();

            let mut expected = vec![0u16; n];
            sum_4bit_dist_table_scalar(code_len, &codes, &dist_table, &mut expected);

            let mut actual = vec![0u16; n];
            sum_4bit_dist_table(n, code_len, &codes, &dist_table, &mut actual);

            assert_eq!(
                actual,
                expected,
                "SIMD and scalar mismatch for capped code_len={} (DIM={})",
                code_len,
                code_len * 8,
            );
        }
    }

    #[test]
    fn test_sum_4bit_dist_table_u32_zero_code_len() {
        let n = BATCH_SIZE;
        let mut dists = vec![u32::MAX; n];
        sum_4bit_dist_table_u32(n, 0, &[], &[], &mut dists);
        assert!(dists.iter().all(|dist| *dist == 0));
    }
    /// Whether this host can run `backend`'s column-major kernels.
    fn transposed_backend_available(backend: DistTableBackend) -> bool {
        match backend {
            #[cfg(target_arch = "x86_64")]
            DistTableBackend::Avx512 => is_x86_feature_detected!("avx512bw"),
            #[cfg(target_arch = "x86_64")]
            DistTableBackend::Avx2 => is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "aarch64")]
            DistTableBackend::Neon => true,
            DistTableBackend::Scalar => true,
            _ => false,
        }
    }

    const TRANSPOSED_BACKENDS: [DistTableBackend; 4] = [
        DistTableBackend::Scalar,
        DistTableBackend::Avx2,
        DistTableBackend::Avx512,
        DistTableBackend::Neon,
    ];

    /// Random column-major codes, a full-range `u8` table and an `f32` table.
    fn transposed_case(n: usize, code_len: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64((n * 1000 + code_len) as u64);
        let codes = (0..n * code_len).map(|_| rng.random::<u8>()).collect();
        let u8_table = (0..code_len * 32).map(|_| rng.random::<u8>()).collect();
        let f32_table = (0..code_len * 32)
            .map(|_| rng.random_range(-3.0f32..5.0))
            .collect();
        (codes, u8_table, f32_table)
    }

    /// Every backend of the `u16` store and filter kernels sums like the
    /// reference, and the filter reports exactly the rows at most its threshold.
    #[rstest::rstest]
    fn test_sum_4bit_dist_table_transposed_matches_reference(
        #[values(0, 65, 1000)] n: usize,
        #[values(0, 3, 48)] code_len: usize,
    ) {
        let (codes, table, _) = transposed_case(n, code_len);
        let expected = (0..n)
            .map(|row| {
                (0..code_len)
                    .map(|column| {
                        let code = codes[column * n + row];
                        table[column * 32 + (code & 0x0f) as usize] as u16
                            + table[column * 32 + 16 + (code >> 4) as usize] as u16
                    })
                    .sum::<u16>()
            })
            .collect::<Vec<_>>();
        let mut actual = vec![u16::MAX; n];
        sum_4bit_dist_table_transposed(n, code_len, &codes, &table, &mut actual);
        assert_eq!(actual, expected);

        let mut sorted = expected.clone();
        sorted.sort_unstable();
        let threshold = sorted.get(n / 2).copied().unwrap_or(0);
        let wanted = expected
            .iter()
            .enumerate()
            .filter(|(_, sum)| **sum <= threshold)
            .map(|(row, sum)| (row, *sum))
            .collect::<Vec<_>>();
        for backend in TRANSPOSED_BACKENDS {
            if !transposed_backend_available(backend) {
                continue;
            }
            let mut found = Vec::new();
            // The host supports `backend`, checked just above.
            unsafe {
                filter_transposed_with_backend(
                    backend,
                    n,
                    code_len,
                    &codes,
                    &table,
                    threshold,
                    &mut |start, mut mask, sums: &[u16; TRANSPOSED_BATCH_SIZE]| {
                        while mask != 0 {
                            let i = mask.trailing_zeros() as usize;
                            found.push((start + i, sums[i]));
                            mask &= mask - 1;
                        }
                        threshold
                    },
                )
            };
            assert_eq!(found, wanted, "{backend:?}");
        }
    }

    /// Every backend of the exact `f32` kernels matches the per-row fold bit
    /// for bit, over all rows and over selected rows.
    #[rstest::rstest]
    fn test_sum_4bit_dist_table_f32_matches_fold(
        #[values(0, 17, 1000)] n: usize,
        #[values(0, 3, 48)] code_len: usize,
    ) {
        let (codes, _, table) = transposed_case(n, code_len);
        let fold = |row: usize| {
            (0..code_len)
                .map(|column| {
                    let code = codes[column * n + row];
                    table[column * 32 + (code & 0x0f) as usize]
                        + table[column * 32 + 16 + (code >> 4) as usize]
                })
                .fold(-0.0f32, |acc, pair| acc + pair)
                .to_bits()
        };
        let expected = (0..n).map(fold).collect::<Vec<_>>();
        let rows = (0..n as u32).rev().step_by(3).collect::<Vec<_>>();
        let expected_rows = rows
            .iter()
            .map(|&row| fold(row as usize))
            .collect::<Vec<_>>();
        let bits = |dists: &[f32]| dists.iter().map(|d| d.to_bits()).collect::<Vec<_>>();
        for backend in TRANSPOSED_BACKENDS {
            if !transposed_backend_available(backend) {
                continue;
            }
            let mut dists = vec![f32::NAN; n];
            let mut row_dists = vec![f32::NAN; rows.len()];
            // The host supports `backend`, checked just above.
            unsafe {
                sum_f32_transposed_with_backend(backend, n, code_len, &codes, &table, &mut dists);
                sum_f32_rows_with_backend(
                    backend,
                    n,
                    code_len,
                    &codes,
                    &table,
                    &rows,
                    &mut row_dists,
                );
            }
            assert_eq!(bits(&dists), expected, "{backend:?}");
            assert_eq!(bits(&row_dists), expected_rows, "{backend:?}");
        }
    }
}
