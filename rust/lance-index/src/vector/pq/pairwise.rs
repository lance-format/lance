// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Symmetric PQ code-to-code scoring through a codeword distance table.
//!
//! Staged codes are column-major per batch (all rows' first code byte, then
//! the second, ...), like the on-disk transposed layout, so one sub-vector's
//! codes for a candidate range are contiguous.

use std::{ops::Range, sync::Arc};

use arrow::compute::cast;
use arrow::datatypes::{Float16Type, Float32Type, Float64Type, UInt8Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float64Array, RecordBatch, UInt8Array, cast::AsArray,
};
use arrow_schema::{DataType, Field};
use lance_arrow::FixedSizeListArrayExt;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;

use super::{ProductQuantizer, storage::build_pairwise_distance_table};
use crate::vector::PQ_CODE_COLUMN;
use crate::vector::pairwise::{
    NORM_COLUMN, PairKernel, PairScorer, column_values, l2_to_cosine, list_values, norm_field,
};

pub struct PQPairScorer {
    num_bits: u32,
    num_sub_vectors: usize,
    code_bytes: usize,
    metric: DistanceType,
    /// Codeword distances `[sub][code_a][code_b]`: squared L2 for l2 and
    /// cosine (residuals share the partition centroid, which cancels), or
    /// per-sub-vector dot distance for dot.
    table: CodewordTable,
    /// Cosine only: `‖c_sub + codeword‖²` per `[sub][code]`, where `c` is the
    /// partition centroid that residual codes are relative to.
    squared_norms: Vec<f64>,
    /// Resolved once per scorer; it also fixes the table layout.
    backend: LookupBackend,
}

/// The codeword distance table in the layout its lookup backend reads.
enum CodewordTable {
    /// `[sub][code_a][code_b]` f32 entries.
    Rows(Vec<f32>),
    /// 8-bit only: `[sub][code_a][k][code_b]` holding byte `k` of each
    /// little-endian f32 entry, so byte shuffles can look up 64 entries at once.
    #[cfg(target_arch = "x86_64")]
    BytePlanes(Vec<u8>),
}

impl CodewordTable {
    fn new(rows: Vec<f32>, backend: LookupBackend) -> Self {
        match backend {
            #[cfg(target_arch = "x86_64")]
            LookupBackend::Avx512Vbmi => Self::BytePlanes(
                rows.chunks_exact(256)
                    .flat_map(|row| {
                        (0..4).flat_map(move |k| row.iter().map(move |v| v.to_le_bytes()[k]))
                    })
                    .collect(),
            ),
            _ => Self::Rows(rows),
        }
    }

    fn rows(&self) -> &[f32] {
        match self {
            Self::Rows(rows) => rows,
            #[cfg(target_arch = "x86_64")]
            Self::BytePlanes(_) => unreachable!("the lookup backend reads byte planes"),
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn byte_planes(&self) -> &[u8] {
        match self {
            Self::BytePlanes(planes) => planes,
            Self::Rows(_) => unreachable!("the lookup backend reads f32 rows"),
        }
    }
}

impl PQPairScorer {
    pub(crate) fn new(
        pq: &ProductQuantizer,
        centroid: &ArrayRef,
        metric: DistanceType,
    ) -> Result<Self> {
        Self::with_backend(pq, centroid, metric, LookupBackend::best(pq.num_bits))
    }

    fn with_backend(
        pq: &ProductQuantizer,
        centroid: &ArrayRef,
        metric: DistanceType,
        backend: LookupBackend,
    ) -> Result<Self> {
        debug_assert!(LookupBackend::supported(pq.num_bits).contains(&backend));
        let num_sub_vectors = pq.num_sub_vectors;
        let code_bytes = match pq.num_bits {
            8 => num_sub_vectors,
            4 if num_sub_vectors.is_multiple_of(2) => num_sub_vectors / 2,
            bits => {
                return Err(Error::not_supported(format!(
                    "PQ pair scoring requires 4 or 8 bits (with an even sub-vector count for 4), got bits={bits} num_sub_vectors={num_sub_vectors}"
                )));
            }
        };
        let table_metric = match metric {
            DistanceType::L2 | DistanceType::Cosine => DistanceType::L2,
            DistanceType::Dot => DistanceType::Dot,
            other => {
                return Err(Error::not_supported(format!(
                    "PQ pair scoring with {other} distance"
                )));
            }
        };
        let values = pq.codebook.values();
        macro_rules! table {
            ($ty:ty) => {
                build_pairwise_distance_table(
                    values.as_primitive::<$ty>().values(),
                    pq.num_bits,
                    num_sub_vectors,
                    pq.dimension,
                    table_metric,
                )
            };
        }
        let table = match values.data_type() {
            DataType::Float16 => table!(Float16Type),
            DataType::Float32 => table!(Float32Type),
            DataType::Float64 => table!(Float64Type),
            other => {
                return Err(Error::not_supported(format!(
                    "PQ pair scoring codebook type {other}"
                )));
            }
        };
        let squared_norms = if metric == DistanceType::Cosine {
            let codebook = cast(values, &DataType::Float64)?;
            let centroid = cast(centroid, &DataType::Float64)?;
            codeword_squared_norms(
                codebook.as_primitive::<Float64Type>().values(),
                centroid.as_primitive::<Float64Type>().values(),
                pq,
            )?
        } else {
            Vec::new()
        };
        Ok(Self {
            num_bits: pq.num_bits,
            num_sub_vectors,
            code_bytes,
            metric,
            table: CodewordTable::new(table, backend),
            squared_norms,
            backend,
        })
    }

    fn num_centroids(&self) -> usize {
        1 << self.num_bits
    }
}

fn codeword_squared_norms(
    codebook: &[f64],
    centroid: &[f64],
    pq: &ProductQuantizer,
) -> Result<Vec<f64>> {
    let num_centroids = 1usize << pq.num_bits;
    let width = pq.dimension / pq.num_sub_vectors;
    if centroid.len() != pq.dimension || codebook.len() != num_centroids * pq.dimension {
        return Err(Error::invalid_input(format!(
            "PQ pair scoring expects a {}-d centroid and {num_centroids}x{} codebook, got {} and {} values",
            pq.dimension,
            pq.dimension,
            centroid.len(),
            codebook.len()
        )));
    }
    Ok(codebook
        .chunks_exact(width)
        .enumerate()
        .map(|(i, codeword)| {
            let sub = i / num_centroids;
            codeword
                .iter()
                .zip(&centroid[sub * width..(sub + 1) * width])
                .map(|(w, c)| (w + c).powi(2))
                .sum()
        })
        .collect())
}

impl PairScorer for PQPairScorer {
    type Kernel<'a> = PQKernel<'a>;

    fn row_bytes(&self) -> usize {
        self.code_bytes
            + if self.metric == DistanceType::Cosine {
                8
            } else {
                0
            }
    }

    /// `source` codes must be column-major over all of its rows.
    fn stage(&self, source: &RecordBatch, rows: Range<usize>) -> Result<Vec<(Field, ArrayRef)>> {
        let values = list_values::<UInt8Type>(source, PQ_CODE_COLUMN)?;
        let stride = source.num_rows();
        if values.len() != stride * self.code_bytes {
            return Err(Error::invalid_input(format!(
                "PQ source has {} code bytes for {stride} rows of {} bytes",
                values.len(),
                self.code_bytes
            )));
        }
        let mut codes = Vec::with_capacity(rows.len() * self.code_bytes);
        for byte in 0..self.code_bytes {
            let start = byte * stride;
            codes.extend_from_slice(&values[start + rows.start..start + rows.end]);
        }
        let mut columns = Vec::with_capacity(2);
        if self.metric == DistanceType::Cosine {
            let norms = self.staged_norms(&codes, rows.len());
            columns.push((
                norm_field(),
                Arc::new(Float64Array::from(norms)) as ArrayRef,
            ));
        }
        let codes = FixedSizeListArray::try_new_from_values(
            UInt8Array::from(codes),
            self.code_bytes as i32,
        )?;
        columns.insert(
            0,
            (
                Field::new(PQ_CODE_COLUMN, codes.data_type().clone(), false),
                Arc::new(codes) as ArrayRef,
            ),
        );
        Ok(columns)
    }

    fn kernel<'a>(
        &'a self,
        anchor: &'a RecordBatch,
        candidates: &'a RecordBatch,
    ) -> Result<Self::Kernel<'a>> {
        let norms = |batch: &'a RecordBatch| -> Result<&'a [f64]> {
            if self.metric == DistanceType::Cosine {
                column_values::<Float64Type>(batch, NORM_COLUMN)
            } else {
                Ok(&[])
            }
        };
        Ok(PQKernel {
            scorer: self,
            anchor: list_values::<UInt8Type>(anchor, PQ_CODE_COLUMN)?,
            anchor_stride: anchor.num_rows(),
            candidates: list_values::<UInt8Type>(candidates, PQ_CODE_COLUMN)?,
            candidate_stride: candidates.num_rows(),
            anchor_norms: norms(anchor)?,
            candidate_norms: norms(candidates)?,
        })
    }
}

impl PQPairScorer {
    /// `‖x̂‖` per row of column-major codes, summing sub-vectors in order.
    fn staged_norms(&self, codes: &[u8], rows: usize) -> Vec<f64> {
        let num_centroids = self.num_centroids();
        let mut squared = vec![0.0f64; rows];
        for (byte, codes) in codes.chunks_exact(rows.max(1)).enumerate() {
            for (sum, &code) in squared.iter_mut().zip(codes) {
                if self.num_bits == 4 {
                    let low = (2 * byte) * num_centroids + usize::from(code & 0x0f);
                    let high = (2 * byte + 1) * num_centroids + usize::from(code >> 4);
                    *sum += self.squared_norms[low];
                    *sum += self.squared_norms[high];
                } else {
                    *sum += self.squared_norms[byte * num_centroids + usize::from(code)];
                }
            }
        }
        squared.into_iter().map(f64::sqrt).collect()
    }
}

pub struct PQKernel<'a> {
    scorer: &'a PQPairScorer,
    anchor: &'a [u8],
    anchor_stride: usize,
    candidates: &'a [u8],
    candidate_stride: usize,
    anchor_norms: &'a [f64],
    candidate_norms: &'a [f64],
}

impl<'a> PQKernel<'a> {
    fn anchor_code(&self, byte: usize, anchor_row: usize) -> u8 {
        self.anchor[byte * self.anchor_stride + anchor_row]
    }

    fn candidate_codes(&self, byte: usize, candidates: &Range<usize>) -> &'a [u8] {
        let start = byte * self.candidate_stride;
        &self.candidates[start + candidates.start..start + candidates.end]
    }

    /// Distances from the anchor's codeword `code` of sub-vector `sub` to all
    /// `N` codewords of that sub-vector.
    fn table_row<const N: usize>(&self, sub: usize, code: u8) -> &'a [f32; N] {
        let start = (sub * N + usize::from(code)) * N;
        self.scorer.table.rows()[start..start + N]
            .try_into()
            .expect("a table row holds one entry per codeword")
    }
}

// The scalar backends apply one code byte per pass: the compiler vectorizes
// each pass over the candidates, which is faster than grouping bytes per
// candidate on both x86 and aarch64.

fn lookup8_scalar(
    kernel: &PQKernel<'_>,
    anchor_row: usize,
    candidates: Range<usize>,
    out: &mut [f32],
) {
    out.fill(0.0);
    for byte in 0..kernel.scorer.code_bytes {
        let row: &[f32; 256] = kernel.table_row(byte, kernel.anchor_code(byte, anchor_row));
        for (out, &code) in out
            .iter_mut()
            .zip(kernel.candidate_codes(byte, &candidates))
        {
            *out += row[usize::from(code)];
        }
    }
}

fn lookup4_scalar(
    kernel: &PQKernel<'_>,
    anchor_row: usize,
    candidates: Range<usize>,
    out: &mut [f32],
) {
    out.fill(0.0);
    for byte in 0..kernel.scorer.code_bytes {
        let code = kernel.anchor_code(byte, anchor_row);
        let low: &[f32; 16] = kernel.table_row(2 * byte, code & 0x0f);
        let high: &[f32; 16] = kernel.table_row(2 * byte + 1, code >> 4);
        for (out, &code) in out
            .iter_mut()
            .zip(kernel.candidate_codes(byte, &candidates))
        {
            *out += low[usize::from(code & 0x0f)];
            *out += high[usize::from(code >> 4)];
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;
    use std::ops::Range;

    use super::PQKernel;

    const LANES: usize = 16;

    /// 4-bit code bytes whose table rows are applied per pass over the
    /// candidates: their 16 rows stay in registers, and each partial sum is
    /// loaded and stored once per group rather than once per sub-vector.
    const LOOKUP_GROUP: usize = 8;

    /// Runs `$body` with the const `$group` = [`LOOKUP_GROUP`] for each full
    /// group of code bytes starting at `$first`, then with `$group` = 1 per
    /// remaining byte. Each candidate's sum still adds sub-vectors in order
    /// starting from zero, so every distance is independent of the grouping,
    /// the SIMD width and the candidate range.
    macro_rules! for_byte_groups {
        ($code_bytes:expr, |$first:ident, $group:ident| $body:block) => {{
            let code_bytes: usize = $code_bytes;
            let mut $first = 0;
            while $first + LOOKUP_GROUP <= code_bytes {
                const $group: usize = LOOKUP_GROUP;
                $body
                $first += LOOKUP_GROUP;
            }
            while $first < code_bytes {
                const $group: usize = 1;
                $body
                $first += 1;
            }
        }};
    }

    /// 4-bit `(low nibble, high nibble)` table rows and candidate codes of the
    /// `G` code bytes from `first`.
    #[allow(clippy::type_complexity)]
    fn rows4<'a, const G: usize>(
        kernel: &PQKernel<'a>,
        first: usize,
        anchor_row: usize,
        candidates: &Range<usize>,
    ) -> ([(&'a [f32; 16], &'a [f32; 16]); G], [&'a [u8]; G]) {
        (
            std::array::from_fn(|g| {
                let byte = first + g;
                let code = kernel.anchor_code(byte, anchor_row);
                (
                    kernel.table_row(2 * byte, code & 0x0f),
                    kernel.table_row(2 * byte + 1, code >> 4),
                )
            }),
            std::array::from_fn(|g| kernel.candidate_codes(first + g, candidates)),
        )
    }

    /// Adds the entries of one 4-bit byte group, low nibble first, in byte order,
    /// to `out[start..]`.
    fn add_entries4<const G: usize>(
        rows: &[(&[f32; 16], &[f32; 16]); G],
        codes: &[&[u8]; G],
        out: &mut [f32],
        start: usize,
    ) {
        let codes = codes.map(|codes| &codes[..out.len()]);
        for (c, out) in out.iter_mut().enumerate().skip(start) {
            for ((low, high), codes) in rows.iter().zip(&codes) {
                let code = codes[c];
                *out += low[usize::from(code & 0x0f)];
                *out += high[usize::from(code >> 4)];
            }
        }
    }

    /// Codes of candidates `c..c + 16`, widened to u32 lanes.
    #[target_feature(enable = "avx512f")]
    fn load_codes(codes: &[u8], c: usize) -> __m512i {
        let codes = &codes[c..c + LANES];
        // SAFETY: `codes` holds exactly 16 bytes.
        _mm512_cvtepu8_epi32(unsafe { _mm_loadu_si128(codes.as_ptr().cast()) })
    }

    /// Looks 16 candidates' nibbles up per instruction in 16-float rows held
    /// in registers.
    #[target_feature(enable = "avx512f")]
    pub(super) fn lookup4_avx512(
        kernel: &PQKernel<'_>,
        anchor_row: usize,
        candidates: Range<usize>,
        out: &mut [f32],
    ) {
        out.fill(0.0);
        let n = out.len();
        for_byte_groups!(kernel.scorer.code_bytes, |first, G| {
            let (rows, codes) = rows4::<G>(kernel, first, anchor_row, &candidates);
            let codes = codes.map(|codes| &codes[..n]);
            // SAFETY: each row holds exactly 16 floats.
            let tables = rows.map(|(low, high)| unsafe {
                (
                    _mm512_loadu_ps(low.as_ptr()),
                    _mm512_loadu_ps(high.as_ptr()),
                )
            });
            let mut c = 0;
            while c + LANES <= n {
                let out = &mut out[c..c + LANES];
                // SAFETY: `out` holds exactly 16 floats.
                let mut sum = unsafe { _mm512_loadu_ps(out.as_ptr()) };
                for ((low, high), codes) in tables.iter().zip(&codes) {
                    // `vpermps` reads only the low 4 bits of each index, so
                    // the low nibble needs no mask.
                    let code = load_codes(codes, c);
                    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(code, *low));
                    let code = _mm512_srli_epi32::<4>(code);
                    sum = _mm512_add_ps(sum, _mm512_permutexvar_ps(code, *high));
                }
                // SAFETY: as above.
                unsafe { _mm512_storeu_ps(out.as_mut_ptr(), sum) };
                c += LANES;
            }
            add_entries4(&rows, &codes, out, c);
        });
    }

    /// Entries of 8 nibbles (the low 4 bits of each u32 lane; higher bits are
    /// ignored) from a 16-float row held as two 8-float halves.
    #[target_feature(enable = "avx2")]
    fn lookup16_avx2(nibbles: __m256i, (lower, upper): (__m256, __m256)) -> __m256 {
        let from_lower = _mm256_permutevar8x32_ps(lower, nibbles);
        let from_upper = _mm256_permutevar8x32_ps(upper, nibbles);
        // Bit 3 of each nibble, shifted into the sign bit, picks the half.
        let upper = _mm256_castsi256_ps(_mm256_slli_epi32::<28>(nibbles));
        _mm256_blendv_ps(from_lower, from_upper, upper)
    }

    /// The AVX2 variant of [`lookup4_avx512`]: 8 candidates per step, with
    /// each 16-float row split across two registers.
    #[target_feature(enable = "avx2")]
    pub(super) fn lookup4_avx2(
        kernel: &PQKernel<'_>,
        anchor_row: usize,
        candidates: Range<usize>,
        out: &mut [f32],
    ) {
        const LANES: usize = 8;
        out.fill(0.0);
        let n = out.len();
        for_byte_groups!(kernel.scorer.code_bytes, |first, G| {
            let (rows, codes) = rows4::<G>(kernel, first, anchor_row, &candidates);
            let codes = codes.map(|codes| &codes[..n]);
            let halves = rows.map(|(low, high)| {
                // SAFETY: each row holds exactly 16 floats.
                unsafe {
                    (
                        (
                            _mm256_loadu_ps(low.as_ptr()),
                            _mm256_loadu_ps(low[8..].as_ptr()),
                        ),
                        (
                            _mm256_loadu_ps(high.as_ptr()),
                            _mm256_loadu_ps(high[8..].as_ptr()),
                        ),
                    )
                }
            });
            let mut c = 0;
            while c + LANES <= n {
                let out = &mut out[c..c + LANES];
                // SAFETY: `out` holds exactly 8 floats.
                let mut sum = unsafe { _mm256_loadu_ps(out.as_ptr()) };
                for ((low, high), codes) in halves.iter().zip(&codes) {
                    let code = &codes[c..c + LANES];
                    // SAFETY: `code` holds exactly 8 bytes.
                    let code = unsafe { _mm_loadl_epi64(code.as_ptr().cast()) };
                    let code = _mm256_cvtepu8_epi32(code);
                    sum = _mm256_add_ps(sum, lookup16_avx2(code, *low));
                    let code = _mm256_srli_epi32::<4>(code);
                    sum = _mm256_add_ps(sum, lookup16_avx2(code, *high));
                }
                // SAFETY: as above.
                unsafe { _mm256_storeu_ps(out.as_mut_ptr(), sum) };
                c += LANES;
            }
            add_entries4(&rows, &codes, out, c);
        });
    }

    /// Byte `p` of the reordered codes holds candidate
    /// `16 * (p / 4 % 4) + 4 * (p / 16) + p % 4`, so the byte-to-f32 unpacks in
    /// [`lookup64`], which interleave within 128-bit lanes, leave candidates
    /// `16 * r..16 * (r + 1)` in output register `r`.
    const UNPACK_ORDER: [u8; 64] = {
        let mut order = [0u8; 64];
        let mut p = 0;
        while p < 64 {
            order[p] = (16 * (p / 4 % 4) + 4 * (p / 16) + p % 4) as u8;
            p += 1;
        }
        order
    };

    /// Table entries of 64 candidate codes from one 256-entry row stored as 4
    /// byte planes of 4 registers each.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    fn lookup64(planes: &[__m512i; 16], codes: __m512i, order: __m512i) -> [__m512; 4] {
        let codes = _mm512_permutexvar_epi8(order, codes);
        // Two-source byte shuffles cover codes below 128; bit 7 picks the
        // upper half.
        let upper = _mm512_movepi8_mask(codes);
        let bytes: [__m512i; 4] = std::array::from_fn(|k| {
            let low = _mm512_permutex2var_epi8(planes[4 * k], codes, planes[4 * k + 1]);
            let high = _mm512_permutex2var_epi8(planes[4 * k + 2], codes, planes[4 * k + 3]);
            _mm512_mask_blend_epi8(upper, low, high)
        });
        let low01 = _mm512_unpacklo_epi8(bytes[0], bytes[1]);
        let high01 = _mm512_unpackhi_epi8(bytes[0], bytes[1]);
        let low23 = _mm512_unpacklo_epi8(bytes[2], bytes[3]);
        let high23 = _mm512_unpackhi_epi8(bytes[2], bytes[3]);
        [
            _mm512_castsi512_ps(_mm512_unpacklo_epi16(low01, low23)),
            _mm512_castsi512_ps(_mm512_unpackhi_epi16(low01, low23)),
            _mm512_castsi512_ps(_mm512_unpacklo_epi16(high01, high23)),
            _mm512_castsi512_ps(_mm512_unpackhi_epi16(high01, high23)),
        ]
    }

    /// Looks 64 candidates' codes up per step with byte shuffles over the
    /// anchor's table row, held in registers as byte planes. One code byte
    /// per pass keeps the 16 row registers live across all candidates.
    #[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
    pub(super) fn lookup8_avx512_vbmi(
        kernel: &PQKernel<'_>,
        anchor_row: usize,
        candidates: Range<usize>,
        out: &mut [f32],
    ) {
        const BLOCK: usize = 64;
        out.fill(0.0);
        let n = out.len();
        let planes = kernel.scorer.table.byte_planes();
        // SAFETY: `UNPACK_ORDER` holds exactly 64 bytes.
        let order = unsafe { _mm512_loadu_si512(UNPACK_ORDER.as_ptr().cast()) };
        let code_bytes = kernel.scorer.code_bytes;
        let row_start =
            |byte: usize| (byte * 256 + usize::from(kernel.anchor_code(byte, anchor_row))) * 1024;
        for byte in 0..code_bytes {
            let start = row_start(byte);
            let row: &[u8; 1024] = planes[start..start + 1024]
                .try_into()
                .expect("a byte-plane row holds 1024 bytes");
            // An anchor's rows are spread over the whole table, which exceeds
            // L2, so fetch the next row while this one is applied.
            if byte + 1 < code_bytes {
                let next = row_start(byte + 1);
                for line in planes[next..next + 1024].chunks_exact(64) {
                    // `_mm_prefetch` is safe in newer toolchains but unsafe at
                    // the 1.91 MSRV, so allow both.
                    // SAFETY: prefetching an in-bounds address has no side effects.
                    #[allow(unused_unsafe)]
                    unsafe {
                        _mm_prefetch::<_MM_HINT_T0>(line.as_ptr().cast())
                    };
                }
            }
            // SAFETY: each register loads 64 of the row's 1024 bytes.
            let row: [__m512i; 16] = std::array::from_fn(|i| unsafe {
                _mm512_loadu_si512(row[BLOCK * i..].as_ptr().cast())
            });
            let codes = &kernel.candidate_codes(byte, &candidates)[..n];
            let mut c = 0;
            while c + BLOCK <= n {
                // SAFETY: the slice holds exactly 64 bytes.
                let block = unsafe { _mm512_loadu_si512(codes[c..c + BLOCK].as_ptr().cast()) };
                let entries = lookup64(&row, block, order);
                for (out, entry) in out[c..c + BLOCK].chunks_exact_mut(LANES).zip(entries) {
                    // SAFETY: `out` holds exactly 16 floats.
                    unsafe {
                        let sum = _mm512_add_ps(_mm512_loadu_ps(out.as_ptr()), entry);
                        _mm512_storeu_ps(out.as_mut_ptr(), sum);
                    }
                }
                c += BLOCK;
            }
            if c < n {
                let mut block = [0u8; BLOCK];
                block[..n - c].copy_from_slice(&codes[c..]);
                // SAFETY: `block` holds exactly 64 bytes.
                let block = unsafe { _mm512_loadu_si512(block.as_ptr().cast()) };
                let mut entries = [0.0f32; BLOCK];
                for (values, entry) in entries
                    .chunks_exact_mut(LANES)
                    .zip(lookup64(&row, block, order))
                {
                    // SAFETY: `values` holds exactly 16 floats.
                    unsafe { _mm512_storeu_ps(values.as_mut_ptr(), entry) };
                }
                for (out, entry) in out[c..].iter_mut().zip(entries) {
                    *out += entry;
                }
            }
        }
    }
}

/// How [`PQKernel`] looks codeword distances up; resolved once per scorer
/// from the host's CPU features. Every backend sums each candidate's
/// sub-vectors in order starting from zero, without fused operations, so all
/// backends give bit-identical distances for any candidate range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LookupBackend {
    Scalar,
    /// 4-bit only: nibble permutes over 16-float rows in registers.
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// 4-bit only: the 256-bit variant of `Avx512`.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// 8-bit only: byte shuffles over byte-plane rows in registers.
    #[cfg(target_arch = "x86_64")]
    Avx512Vbmi,
}

impl LookupBackend {
    /// Backends this host supports for `num_bits` codes, fastest first.
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
    fn supported(num_bits: u32) -> Vec<Self> {
        let mut backends = Vec::with_capacity(3);
        #[cfg(target_arch = "x86_64")]
        if num_bits == 4 {
            if is_x86_feature_detected!("avx512f") {
                backends.push(Self::Avx512);
            }
            if is_x86_feature_detected!("avx2") {
                backends.push(Self::Avx2);
            }
        } else if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
        {
            backends.push(Self::Avx512Vbmi);
        }
        backends.push(Self::Scalar);
        backends
    }

    fn best(num_bits: u32) -> Self {
        Self::supported(num_bits)[0]
    }
}

impl PairKernel for PQKernel<'_> {
    fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        let scorer = self.scorer;
        assert_eq!(out.len(), candidates.len());
        let range = candidates.clone();
        match scorer.backend {
            LookupBackend::Scalar if scorer.num_bits == 8 => {
                lookup8_scalar(self, anchor_row, range, out)
            }
            LookupBackend::Scalar => lookup4_scalar(self, anchor_row, range, out),
            // SAFETY: `LookupBackend::supported` detected the CPU features of
            // each x86 backend, for this code width.
            #[cfg(target_arch = "x86_64")]
            LookupBackend::Avx512 => unsafe { x86::lookup4_avx512(self, anchor_row, range, out) },
            #[cfg(target_arch = "x86_64")]
            LookupBackend::Avx2 => unsafe { x86::lookup4_avx2(self, anchor_row, range, out) },
            #[cfg(target_arch = "x86_64")]
            LookupBackend::Avx512Vbmi => unsafe {
                x86::lookup8_avx512_vbmi(self, anchor_row, range, out)
            },
        }
        match scorer.metric {
            DistanceType::Dot => {
                // Each sub-vector contributed `1 - dot`; keep a single `1 -`.
                let offset = scorer.num_sub_vectors as f32 - 1.0;
                out.iter_mut().for_each(|distance| *distance -= offset);
            }
            DistanceType::Cosine => l2_to_cosine(
                out,
                self.anchor_norms[anchor_row],
                &self.candidate_norms[candidates],
            ),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::pq::storage::transpose;
    use arrow_array::{Float32Array, RecordBatch, UInt64Array};
    use arrow_schema::Schema;
    use lance_core::ROW_ID;
    use rstest::rstest;

    /// Dimension 6 with 3 sub-vectors of width 2; codewords are small
    /// integers so the f64 oracle and the f32 table agree closely.
    fn quantizer(num_bits: u32, metric: DistanceType) -> ProductQuantizer {
        let num_sub_vectors = if num_bits == 4 { 4 } else { 3 };
        let dimension = num_sub_vectors * 2;
        let num_centroids = 1usize << num_bits;
        let codebook: Vec<f32> = (0..num_sub_vectors * num_centroids * 2)
            .map(|i| ((i * 7 + 3) % 11) as f32 - 5.0)
            .collect();
        ProductQuantizer::new(
            num_sub_vectors,
            num_bits,
            dimension,
            FixedSizeListArray::try_new_from_values(Float32Array::from(codebook), dimension as i32)
                .unwrap(),
            metric,
        )
    }

    #[rstest]
    fn test_pq_distances(
        #[values(4, 8)] num_bits: u32,
        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)] metric: DistanceType,
    ) {
        let pq = quantizer(num_bits, metric);
        let dim = pq.dimension;
        let width = 2;
        let num_centroids = 1usize << num_bits;
        let centroid: Vec<f32> = (0..dim).map(|d| d as f32 * 0.5 - 1.0).collect();
        let centroid_array = Arc::new(Float32Array::from(centroid.clone())) as ArrayRef;
        let scorer = PQPairScorer::new(&pq, &centroid_array, metric).unwrap();
        // 37 rows: row 1 repeats row 0, and the last row pairs with a tail.
        let rows = 37;
        let subs: Vec<Vec<usize>> = (0..rows)
            .map(|r| {
                let r = if r == 1 { 0 } else { r };
                (0..pq.num_sub_vectors)
                    .map(|m| (r * 5 + m * 3) % num_centroids)
                    .collect()
            })
            .collect();
        let row_major: Vec<u8> = subs
            .iter()
            .flat_map(|codes| {
                if num_bits == 4 {
                    codes
                        .chunks_exact(2)
                        .map(|c| (c[0] | (c[1] << 4)) as u8)
                        .collect::<Vec<_>>()
                } else {
                    codes.iter().map(|&c| c as u8).collect()
                }
            })
            .collect();
        let code_bytes = row_major.len() / rows;
        let column_major = transpose(&UInt8Array::from(row_major), rows, code_bytes);
        let codes =
            FixedSizeListArray::try_new_from_values(column_major, code_bytes as i32).unwrap();
        let source = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(ROW_ID, DataType::UInt64, false),
                Field::new(PQ_CODE_COLUMN, codes.data_type().clone(), false),
            ])),
            vec![
                Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
                Arc::new(codes),
            ],
        )
        .unwrap();
        let stage = |rows: Range<usize>| {
            RecordBatch::try_from_iter(
                scorer
                    .stage(&source, rows)
                    .unwrap()
                    .into_iter()
                    .map(|(field, column)| (field.name().clone(), column)),
            )
            .unwrap()
        };
        let anchor = stage(0..32);
        let tail = stage(32..rows);
        let codebook = pq.codebook.values().as_primitive::<Float32Type>().values();
        let reconstruct = |r: usize| -> Vec<f64> {
            (0..dim)
                .map(|d| {
                    let m = d / width;
                    let word = codebook[(m * num_centroids + subs[r][m]) * width + d % width];
                    let base = if metric == DistanceType::Dot {
                        0.0
                    } else {
                        f64::from(centroid[d])
                    };
                    base + f64::from(word)
                })
                .collect()
        };
        for backend in LookupBackend::supported(num_bits) {
            let scorer = PQPairScorer::with_backend(&pq, &centroid_array, metric, backend).unwrap();
            let kernel = scorer.kernel(&anchor, &anchor).unwrap();
            let tail_kernel = scorer.kernel(&anchor, &tail).unwrap();
            for a in [0, 1, 17, 31] {
                let mut same = vec![0.0; 32];
                kernel.distances(a, 0..32, &mut same);
                let mut later = vec![0.0; rows - 32];
                tail_kernel.distances(a, 0..rows - 32, &mut later);
                // A sub-range yields bit-identical values.
                let mut part = vec![0.0; 5];
                kernel.distances(a, 3..8, &mut part);
                assert_eq!(part, same[3..8]);
                for (b, distance) in same.into_iter().chain(later).enumerate() {
                    let (x, y) = (reconstruct(a), reconstruct(b));
                    let dot: f64 = x.iter().zip(&y).map(|(x, y)| x * y).sum();
                    let norm = |v: &[f64]| v.iter().map(|v| v * v).sum::<f64>().sqrt();
                    let expected = match metric {
                        DistanceType::L2 => x.iter().zip(&y).map(|(x, y)| (x - y).powi(2)).sum(),
                        DistanceType::Dot => 1.0 - dot,
                        _ => 1.0 - dot / (norm(&x) * norm(&y)),
                    };
                    assert!(
                        (f64::from(distance) - expected).abs() <= 1e-5 * expected.abs().max(1.0),
                        "{backend:?} bits={num_bits} {metric} a={a} b={b} actual={distance} expected={expected}"
                    );
                    if subs[a] == subs[b] && metric != DistanceType::Dot {
                        assert_eq!(distance, 0.0, "{backend:?} a={a} b={b}");
                    }
                }
            }
        }
    }

    /// The previous one-sub-vector-per-pass lookup over `[sub][code_a][code_b]`
    /// f32 rows, the bit-exact reference.
    fn reference_lookup(
        kernel: &PQKernel<'_>,
        table: &[f32],
        anchor_row: usize,
        candidates: Range<usize>,
        out: &mut [f32],
    ) {
        let scorer = kernel.scorer;
        let num_centroids = scorer.num_centroids();
        let table_row = |sub: usize, code: u8| {
            let start = (sub * num_centroids + usize::from(code)) * num_centroids;
            &table[start..start + num_centroids]
        };
        out.fill(0.0);
        for byte in 0..scorer.code_bytes {
            let code = kernel.anchor[byte * kernel.anchor_stride + anchor_row];
            let start = byte * kernel.candidate_stride;
            let codes = &kernel.candidates[start + candidates.start..start + candidates.end];
            if scorer.num_bits == 4 {
                let low = table_row(2 * byte, code & 0x0f);
                let high = table_row(2 * byte + 1, code >> 4);
                for (out, &code) in out.iter_mut().zip(codes) {
                    *out += low[usize::from(code & 0x0f)];
                    *out += high[usize::from(code >> 4)];
                }
            } else {
                let row = table_row(byte, code);
                for (out, &code) in out.iter_mut().zip(codes) {
                    *out += row[usize::from(code)];
                }
            }
        }
    }

    /// Every backend sums each candidate's sub-vectors in the same order, so
    /// its values are bit-identical to the reference for any byte grouping,
    /// SIMD block and tail. An 8-bit table is 16x larger per sub-vector than
    /// a 4-bit one, so 8-bit cases stop at 34 sub-vectors to stay fast in
    /// debug builds; any multi-byte case already exercises the VBMI kernel's
    /// next-row prefetch.
    #[rstest]
    #[case::bits4_subs2(4, 2)]
    #[case::bits4_subs18(4, 18)]
    #[case::bits4_subs34(4, 34)]
    #[case::bits4_subs96(4, 96)]
    #[case::bits8_subs2(8, 2)]
    #[case::bits8_subs18(8, 18)]
    #[case::bits8_subs34(8, 34)]
    fn test_pq_lookup_backends_bit_identical(
        #[case] num_bits: u32,
        #[case] num_sub_vectors: usize,
    ) {
        use rand::{Rng, SeedableRng, rngs::SmallRng};

        let mut rng = SmallRng::seed_from_u64(u64::from(num_bits) * 1000 + num_sub_vectors as u64);
        let num_centroids = 1usize << num_bits;
        let code_bytes = num_sub_vectors * num_bits as usize / 8;
        // Magnitudes spanning several decades make any reordering of the sum
        // change the rounded result.
        let table: Vec<f32> = (0..num_sub_vectors * num_centroids * num_centroids)
            .map(|_| rng.random_range(-1.0f32..1.0) * 10f32.powi(rng.random_range(-4..4)))
            .collect();
        let (anchor_stride, candidate_stride) = (40, 200);
        let anchor: Vec<u8> = (0..code_bytes * anchor_stride)
            .map(|_| rng.random())
            .collect();
        let candidates: Vec<u8> = (0..code_bytes * candidate_stride)
            .map(|_| rng.random())
            .collect();
        let backends = LookupBackend::supported(num_bits);
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512f") {
            assert_ne!(backends[0], LookupBackend::Scalar);
        }
        for backend in backends {
            let scorer = PQPairScorer {
                num_bits,
                num_sub_vectors,
                code_bytes,
                metric: DistanceType::L2,
                table: CodewordTable::new(table.clone(), backend),
                squared_norms: Vec::new(),
                backend,
            };
            let kernel = PQKernel {
                scorer: &scorer,
                anchor: &anchor,
                anchor_stride,
                candidate_stride,
                candidates: &candidates,
                anchor_norms: &[],
                candidate_norms: &[],
            };
            for anchor_row in [0, 7, 39] {
                for range in [0..200, 3..200, 60..190, 5..21, 16..32, 50..50, 199..200] {
                    let mut expected = vec![0.0; range.len()];
                    reference_lookup(&kernel, &table, anchor_row, range.clone(), &mut expected);
                    let mut actual = vec![f32::NAN; range.len()];
                    kernel.distances(anchor_row, range.clone(), &mut actual);
                    assert_eq!(
                        actual.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
                        "{backend:?} bits={num_bits} subs={num_sub_vectors} a={anchor_row} {range:?}"
                    );
                }
            }
        }
    }
}
