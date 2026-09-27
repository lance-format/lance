// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact pair scoring over stored IVF_FLAT vectors.

use std::{ops::Range, sync::Arc};

use arrow_array::{
    ArrayRef, Float64Array, RecordBatch,
    cast::AsArray,
    types::{Float16Type, Float32Type, Float64Type, UInt8Type},
};
use arrow_schema::{DataType, Field, Schema};
use half::f16;
use lance_core::{Error, Result};
use lance_linalg::distance::{DistanceType, Dot, L2, hamming::hamming};
use num_traits::AsPrimitive;

use crate::vector::pairwise::{
    NORM_COLUMN, PairKernel, PairScorer, column_values, l2_to_cosine, list_values, norm_field,
};

#[derive(Clone, Copy, Debug)]
enum Element {
    Float16,
    Float32,
    Float64,
    Binary,
}

pub struct FlatPairScorer {
    column: &'static str,
    dim: usize,
    element: Element,
    metric: DistanceType,
}

impl FlatPairScorer {
    pub(crate) fn new(column: &'static str, source: &Schema, metric: DistanceType) -> Result<Self> {
        let field = source.field_with_name(column)?;
        let DataType::FixedSizeList(item, dim) = field.data_type() else {
            return Err(Error::invalid_input(format!(
                "flat vector column {column} must be a fixed-size list, got {}",
                field.data_type()
            )));
        };
        let element = match (item.data_type(), metric) {
            (DataType::UInt8, DistanceType::Hamming) => Element::Binary,
            (DataType::Float16, DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot) => {
                Element::Float16
            }
            (DataType::Float32, DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot) => {
                Element::Float32
            }
            (DataType::Float64, DistanceType::L2 | DistanceType::Cosine | DistanceType::Dot) => {
                Element::Float64
            }
            (other, metric) => {
                return Err(Error::not_supported(format!(
                    "pair scoring of {other} flat vectors with {metric} distance"
                )));
            }
        };
        Ok(Self {
            column,
            dim: *dim as usize,
            element,
            metric,
        })
    }
}

fn norms<T: AsPrimitive<f64>>(values: &[T], dim: usize) -> Vec<f64> {
    values
        .chunks_exact(dim)
        .map(|v| v.iter().map(|x| x.as_().powi(2)).sum::<f64>().sqrt())
        .collect()
}

impl PairScorer for FlatPairScorer {
    type Kernel<'a> = FlatKernel<'a>;

    fn row_bytes(&self) -> usize {
        let width = match self.element {
            Element::Float16 => 2,
            Element::Float32 => 4,
            Element::Float64 => 8,
            Element::Binary => 1,
        };
        let norm = if self.metric == DistanceType::Cosine {
            8
        } else {
            0
        };
        self.dim * width + norm
    }

    fn stage(&self, source: &RecordBatch, rows: Range<usize>) -> Result<Vec<(Field, ArrayRef)>> {
        let (field, vectors) = source
            .schema()
            .column_with_name(self.column)
            .map(|(index, field)| {
                (
                    field.clone(),
                    source.column(index).slice(rows.start, rows.len()),
                )
            })
            .ok_or_else(|| Error::internal(format!("flat batch missing {}", self.column)))?;
        let mut columns = vec![(field, vectors.clone())];
        if self.metric == DistanceType::Cosine {
            let values = vectors
                .as_fixed_size_list_opt()
                .ok_or_else(|| Error::internal("flat vectors must be a fixed-size list"))?
                .values();
            let norms = match self.element {
                Element::Float16 => norms(values.as_primitive::<Float16Type>().values(), self.dim),
                Element::Float32 => norms(values.as_primitive::<Float32Type>().values(), self.dim),
                Element::Float64 => norms(values.as_primitive::<Float64Type>().values(), self.dim),
                Element::Binary => {
                    return Err(Error::internal("binary flat vectors have no cosine norm"));
                }
            };
            columns.push((
                norm_field(),
                Arc::new(Float64Array::from(norms)) as ArrayRef,
            ));
        }
        Ok(columns)
    }

    fn kernel<'a>(
        &'a self,
        anchor: &'a RecordBatch,
        candidates: &'a RecordBatch,
    ) -> Result<Self::Kernel<'a>> {
        fn float<'a, T: arrow_array::ArrowPrimitiveType>(
            scorer: &FlatPairScorer,
            anchor: &'a RecordBatch,
            candidates: &'a RecordBatch,
        ) -> Result<FloatKernel<'a, T::Native>>
        where
            T::Native: FlatFloat,
        {
            let norms = |batch: &'a RecordBatch| -> Result<&'a [f64]> {
                if scorer.metric == DistanceType::Cosine {
                    column_values::<Float64Type>(batch, NORM_COLUMN)
                } else {
                    Ok(&[])
                }
            };
            Ok(FloatKernel {
                dim: scorer.dim,
                metric: scorer.metric,
                anchor: list_values::<T>(anchor, scorer.column)?,
                candidates: list_values::<T>(candidates, scorer.column)?,
                anchor_norms: norms(anchor)?,
                candidate_norms: norms(candidates)?,
                rows: T::Native::rows_kernel(scorer.metric, scorer.dim),
            })
        }
        Ok(match self.element {
            Element::Float16 => {
                FlatKernel::Float16(float::<Float16Type>(self, anchor, candidates)?)
            }
            Element::Float32 => {
                FlatKernel::Float32(float::<Float32Type>(self, anchor, candidates)?)
            }
            Element::Float64 => {
                FlatKernel::Float64(float::<Float64Type>(self, anchor, candidates)?)
            }
            Element::Binary => FlatKernel::Binary {
                dim: self.dim,
                anchor: list_values::<UInt8Type>(anchor, self.column)?,
                candidates: list_values::<UInt8Type>(candidates, self.column)?,
            },
        })
    }
}

pub enum FlatKernel<'a> {
    Float16(FloatKernel<'a, f16>),
    Float32(FloatKernel<'a, f32>),
    Float64(FloatKernel<'a, f64>),
    Binary {
        dim: usize,
        anchor: &'a [u8],
        candidates: &'a [u8],
    },
}

pub struct FloatKernel<'a, T> {
    dim: usize,
    metric: DistanceType,
    anchor: &'a [T],
    candidates: &'a [T],
    anchor_norms: &'a [f64],
    candidate_norms: &'a [f64],
    /// Squared L2 (dot product for [`DistanceType::Dot`]) to consecutive
    /// rows, resolved once per binding.
    rows: RowsFn<T>,
}

/// Squared L2 or dot product from `anchor` to each of the `out.len()`
/// consecutive `anchor.len()`-wide rows in `rows`.
type RowsFn<T> = fn(anchor: &[T], rows: &[T], out: &mut [f32]);

/// Float element types of IVF_FLAT vectors.
pub trait FlatFloat: L2 + Dot {
    fn rows_kernel(metric: DistanceType, dim: usize) -> RowsFn<Self>;
}

/// lance-linalg's batch kernels, one row at a time.
fn batch_rows_kernel<T: L2 + Dot>(metric: DistanceType) -> RowsFn<T> {
    fn l2_rows<T: L2>(anchor: &[T], rows: &[T], out: &mut [f32]) {
        for (out, l2) in out.iter_mut().zip(T::l2_batch(anchor, rows, anchor.len())) {
            *out = l2;
        }
    }
    fn dot_rows<T: Dot>(anchor: &[T], rows: &[T], out: &mut [f32]) {
        for (out, dot) in out.iter_mut().zip(T::dot_batch(anchor, rows, anchor.len())) {
            *out = dot;
        }
    }
    if metric == DistanceType::Dot {
        dot_rows::<T>
    } else {
        l2_rows::<T>
    }
}

impl FlatFloat for f16 {
    fn rows_kernel(metric: DistanceType, _dim: usize) -> RowsFn<Self> {
        batch_rows_kernel(metric)
    }
}

impl FlatFloat for f64 {
    fn rows_kernel(metric: DistanceType, _dim: usize) -> RowsFn<Self> {
        batch_rows_kernel(metric)
    }
}

impl FlatFloat for f32 {
    fn rows_kernel(metric: DistanceType, dim: usize) -> RowsFn<Self> {
        // Up to one lane block, lance-linalg scores f32 with an FMA kernel on
        // x86, which the blocked kernels do not reproduce.
        if dim <= f32_rows::LANES {
            batch_rows_kernel(metric)
        } else {
            f32_rows::select(metric != DistanceType::Dot)
        }
    }
}

impl<T: FlatFloat> FloatKernel<'_, T> {
    fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        let dim = self.dim;
        let x = &self.anchor[anchor_row * dim..(anchor_row + 1) * dim];
        let y = &self.candidates[candidates.start * dim..candidates.end * dim];
        (self.rows)(x, y, out);
        match self.metric {
            DistanceType::Dot => {
                for out in out.iter_mut() {
                    *out = 1.0 - *out;
                }
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

/// f32 squared L2 and dot product over rows wider than one lane block,
/// bit-identical to lance-linalg's `l2_scalar::<f32, f32, 16>` and
/// `dot_scalar::<f32, f32, 16>` (what its batch kernels run above 16
/// dimensions on the x86 AVX2 baseline and on aarch64).
///
/// Each row keeps the same 16 independent lane sums, updated with a separate
/// multiply and add (never FMA), then folds the remainder and the lanes in
/// the same order. On x86 the 16 lanes are one AVX-512 register (or two AVX
/// registers), so a single row's sums form one or two dependent add chains;
/// the x86 kernels score several rows per pass over the anchor to keep the
/// floating-point ports busy instead.
mod f32_rows {
    pub(super) const LANES: usize = 16;

    pub(super) fn select(l2: bool) -> super::RowsFn<f32> {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return if l2 { x86::l2_avx512 } else { x86::dot_avx512 };
            }
            if std::arch::is_x86_feature_detected!("avx") {
                return if l2 { x86::l2_avx } else { x86::dot_avx };
            }
        }
        if l2 {
            portable::<true>
        } else {
            portable::<false>
        }
    }

    /// Every kernel this CPU can run, named.
    #[cfg(test)]
    pub(super) fn backends(l2: bool) -> Vec<(&'static str, super::RowsFn<f32>)> {
        #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
        let mut backends: Vec<(&'static str, super::RowsFn<f32>)> = vec![(
            "portable",
            if l2 {
                portable::<true>
            } else {
                portable::<false>
            },
        )];
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                backends.push(("avx512", if l2 { x86::l2_avx512 } else { x86::dot_avx512 }));
            }
            if std::arch::is_x86_feature_detected!("avx") {
                backends.push(("avx", if l2 { x86::l2_avx } else { x86::dot_avx }));
            }
        }
        backends
    }

    #[inline(always)]
    fn term<const L2: bool>(anchor: f32, row: f32) -> f32 {
        if L2 {
            let diff = anchor - row;
            diff * diff
        } else {
            anchor * row
        }
    }

    /// Remainder dimensions summed in order, plus the lane sums folded in
    /// order, as lance-linalg does.
    #[inline(always)]
    fn finish<const L2: bool>(anchor_tail: &[f32], row_tail: &[f32], lanes: &[f32; LANES]) -> f32 {
        let tail = if anchor_tail.is_empty() {
            0.0
        } else {
            anchor_tail
                .iter()
                .zip(row_tail)
                .map(|(&x, &y)| term::<L2>(x, y))
                .sum::<f32>()
        };
        tail + lanes.iter().copied().sum::<f32>()
    }

    /// One row at a time, in lance-linalg's own loop shape. Autovectorized
    /// NEON already runs the 16 lanes as four independent chains, so blocking
    /// rows gains nothing there.
    fn portable<const L2: bool>(anchor: &[f32], rows: &[f32], out: &mut [f32]) {
        assert_eq!(rows.len(), anchor.len() * out.len());
        for (out, row) in out.iter_mut().zip(rows.chunks_exact(anchor.len())) {
            let x = anchor.chunks_exact(LANES);
            let y = row.chunks_exact(LANES);
            let (x_tail, y_tail) = (x.remainder(), y.remainder());
            let mut sums = [0.0f32; LANES];
            for (x, y) in x.zip(y) {
                for i in 0..LANES {
                    sums[i] += term::<L2>(x[i], y[i]);
                }
            }
            *out = finish::<L2>(x_tail, y_tail, &sums);
        }
    }

    #[cfg(target_arch = "x86_64")]
    mod x86 {
        use std::arch::x86_64::*;

        use super::{LANES, finish};

        /// Rows scored per pass over the anchor.
        const BLOCK: usize = 4;

        /// Score `rows` in blocks of [`BLOCK`], then the leftover rows as one
        /// narrower block.
        #[inline(always)]
        fn blocked(
            anchor: &[f32],
            rows: &[f32],
            out: &mut [f32],
            block: impl Fn(&[f32], &mut [f32]),
        ) {
            let dim = anchor.len();
            assert_eq!(rows.len(), dim * out.len());
            let mut blocks = rows.chunks_exact(dim * BLOCK);
            let mut outs = out.chunks_exact_mut(BLOCK);
            for (rows, out) in (&mut blocks).zip(&mut outs) {
                block(rows, out);
            }
            let out = outs.into_remainder();
            if !out.is_empty() {
                block(blocks.remainder(), out);
            }
        }

        macro_rules! rows_kernel {
            ($name:ident, $block:ident, $l2:literal) => {
                pub(super) fn $name(anchor: &[f32], rows: &[f32], out: &mut [f32]) {
                    // SAFETY: `select` returns this kernel only when the CPU
                    // supports `$block`'s target features.
                    blocked(anchor, rows, out, |rows, out| unsafe {
                        match out.len() {
                            BLOCK => $block::<$l2, BLOCK>(anchor, rows, out),
                            3 => $block::<$l2, 3>(anchor, rows, out),
                            2 => $block::<$l2, 2>(anchor, rows, out),
                            _ => $block::<$l2, 1>(anchor, rows, out),
                        }
                    });
                }
            };
        }

        rows_kernel!(l2_avx512, block_avx512, true);
        rows_kernel!(dot_avx512, block_avx512, false);
        rows_kernel!(l2_avx, block_avx, true);
        rows_kernel!(dot_avx, block_avx, false);

        /// Adds the terms of 16 dimensions at `x` and at `y` plus each row
        /// offset. Masked-off positions load zeros, whose term is `+0.0` and
        /// leaves the sum unchanged.
        #[inline]
        #[target_feature(enable = "avx512f")]
        unsafe fn accumulate_avx512<const L2: bool, const N: usize>(
            x: *const f32,
            y: *const f32,
            dim: usize,
            mask: __mmask16,
            sums: &mut [__m512; N],
        ) {
            // Inlined with a constant full mask, these are plain loads.
            let xv = _mm512_maskz_loadu_ps(mask, x);
            for (r, sums) in sums.iter_mut().enumerate() {
                let yv = _mm512_maskz_loadu_ps(mask, y.wrapping_add(r * dim));
                let term = if L2 {
                    let diff = _mm512_sub_ps(xv, yv);
                    _mm512_mul_ps(diff, diff)
                } else {
                    _mm512_mul_ps(xv, yv)
                };
                *sums = _mm512_add_ps(*sums, term);
            }
        }

        /// Scores `N = out.len()` rows; `rows` holds exactly `N` rows.
        #[inline]
        #[target_feature(enable = "avx512f")]
        unsafe fn block_avx512<const L2: bool, const N: usize>(
            anchor: &[f32],
            rows: &[f32],
            out: &mut [f32],
        ) {
            let dim = anchor.len();
            debug_assert_eq!(rows.len(), N * dim);
            if dim.is_multiple_of(LANES) {
                return block_avx512_aligned::<L2, N>(anchor, rows, out);
            }
            let full = dim - dim % LANES;
            let mut sums = [_mm512_setzero_ps(); N];
            let mut k = 0;
            while k < full {
                accumulate_avx512::<L2, N>(
                    anchor.as_ptr().add(k),
                    rows.as_ptr().add(k),
                    dim,
                    !0,
                    &mut sums,
                );
                k += LANES;
            }
            for (r, sums) in sums.iter().enumerate() {
                let mut lanes = [0.0f32; LANES];
                _mm512_storeu_ps(lanes.as_mut_ptr(), *sums);
                let row = &rows[r * dim..(r + 1) * dim];
                out[r] = finish::<L2>(&anchor[full..], &row[full..], &lanes);
            }
        }

        /// [`block_avx512`] for whole lane blocks, reading the rows along
        /// their 64-byte boundaries.
        ///
        /// Rows are then a multiple of 64 bytes apart, so all start `shift`
        /// floats past a boundary. A 64-byte load that crosses a cache line
        /// costs two, and staged Arrow buffers are commonly only 16-byte
        /// aligned, so the loads start `shift` floats early: register
        /// position `q` then holds dimension `16k - shift + q`, which belongs
        /// to lane `(q - shift) mod 16`. Each lane still adds its dimensions
        /// in order, the partial first and last loads mask off dimensions
        /// outside the row, and the lanes are rotated back before folding.
        #[inline]
        #[target_feature(enable = "avx512f")]
        unsafe fn block_avx512_aligned<const L2: bool, const N: usize>(
            anchor: &[f32],
            rows: &[f32],
            out: &mut [f32],
        ) {
            let dim = anchor.len();
            debug_assert_eq!(dim % LANES, 0);
            let shift = (rows.as_ptr() as usize / size_of::<f32>()) % LANES;
            // Masked lanes are never accessed, so these may point before the
            // slices.
            let x = anchor.as_ptr().wrapping_sub(shift);
            let y = rows.as_ptr().wrapping_sub(shift);
            let mut sums = [_mm512_setzero_ps(); N];
            accumulate_avx512::<L2, N>(x, y, dim, !0 << shift, &mut sums);
            let mut k = LANES;
            while k < dim {
                accumulate_avx512::<L2, N>(
                    x.wrapping_add(k),
                    y.wrapping_add(k),
                    dim,
                    !0,
                    &mut sums,
                );
                k += LANES;
            }
            if shift > 0 {
                accumulate_avx512::<L2, N>(
                    x.wrapping_add(dim),
                    y.wrapping_add(dim),
                    dim,
                    (1 << shift) - 1,
                    &mut sums,
                );
            }
            // Lane j sits at position (j + shift) mod 16.
            let positions = _mm512_and_si512(
                _mm512_add_epi32(
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                    _mm512_set1_epi32(shift as i32),
                ),
                _mm512_set1_epi32(LANES as i32 - 1),
            );
            for (out, sums) in out.iter_mut().zip(&sums) {
                let mut lanes = [0.0f32; LANES];
                _mm512_storeu_ps(lanes.as_mut_ptr(), _mm512_permutexvar_ps(positions, *sums));
                *out = finish::<L2>(&[], &[], &lanes);
            }
        }

        /// [`block_avx512`] with each 16-lane sum split across two registers.
        #[inline]
        #[target_feature(enable = "avx")]
        unsafe fn block_avx<const L2: bool, const N: usize>(
            anchor: &[f32],
            rows: &[f32],
            out: &mut [f32],
        ) {
            let dim = anchor.len();
            let full = dim - dim % LANES;
            debug_assert_eq!(rows.len(), N * dim);
            let x = anchor.as_ptr();
            let y = rows.as_ptr();
            let mut sums = [[_mm256_setzero_ps(); 2]; N];
            let mut k = 0;
            while k < full {
                let xv = [_mm256_loadu_ps(x.add(k)), _mm256_loadu_ps(x.add(k + 8))];
                for (r, sums) in sums.iter_mut().enumerate() {
                    for (half, sums) in sums.iter_mut().enumerate() {
                        let yv = _mm256_loadu_ps(y.add(r * dim + k + 8 * half));
                        let term = if L2 {
                            let diff = _mm256_sub_ps(xv[half], yv);
                            _mm256_mul_ps(diff, diff)
                        } else {
                            _mm256_mul_ps(xv[half], yv)
                        };
                        *sums = _mm256_add_ps(*sums, term);
                    }
                }
                k += LANES;
            }
            for (r, sums) in sums.iter().enumerate() {
                let mut lanes = [0.0f32; LANES];
                _mm256_storeu_ps(lanes.as_mut_ptr(), sums[0]);
                _mm256_storeu_ps(lanes.as_mut_ptr().add(8), sums[1]);
                let row = &rows[r * dim..(r + 1) * dim];
                out[r] = finish::<L2>(&anchor[full..], &row[full..], &lanes);
            }
        }
    }
}

impl PairKernel for FlatKernel<'_> {
    fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        match self {
            Self::Float16(kernel) => kernel.distances(anchor_row, candidates, out),
            Self::Float32(kernel) => kernel.distances(anchor_row, candidates, out),
            Self::Float64(kernel) => kernel.distances(anchor_row, candidates, out),
            Self::Binary {
                dim,
                anchor,
                candidates: values,
            } => {
                let x = &anchor[anchor_row * dim..(anchor_row + 1) * dim];
                let y = &values[candidates.start * dim..candidates.end * dim];
                for (out, y) in out.iter_mut().zip(y.chunks_exact(*dim)) {
                    *out = hamming(x, y);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::flat::storage::FLAT_COLUMN;
    use arrow_array::{
        Array, FixedSizeListArray, Float16Array, Float32Array, UInt8Array, UInt64Array,
    };
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::ROW_ID;
    use rstest::rstest;

    fn source(values: ArrayRef, dim: i32) -> RecordBatch {
        let rows = values.len() / dim as usize;
        let vectors = FixedSizeListArray::try_new_from_values(values, dim).unwrap();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new(ROW_ID, DataType::UInt64, false),
                Field::new(FLAT_COLUMN, vectors.data_type().clone(), false),
            ])),
            vec![
                Arc::new(UInt64Array::from_iter_values(0..rows as u64)),
                Arc::new(vectors),
            ],
        )
        .unwrap()
    }

    fn score_all(scorer: &FlatPairScorer, source: &RecordBatch) -> Vec<Vec<f32>> {
        let rows = source.num_rows();
        let staged = RecordBatch::try_from_iter(
            scorer
                .stage(source, 0..rows)
                .unwrap()
                .into_iter()
                .map(|(field, column)| (field.name().clone(), column)),
        )
        .unwrap();
        let kernel = scorer.kernel(&staged, &staged).unwrap();
        (0..rows)
            .map(|a| {
                let mut out = vec![0.0; rows];
                kernel.distances(a, 0..rows, &mut out);
                out
            })
            .collect()
    }

    #[rstest]
    fn test_flat_distances(
        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)] metric: DistanceType,
        #[values(DataType::Float16, DataType::Float32, DataType::Float64)] element: DataType,
        // Above 16 dimensions f32 takes the blocked kernels, with a remainder.
        #[values(3, 35)] dim: usize,
    ) {
        // Small integers and halves are exact in every float width, and so
        // are all their sums of products here.
        let base = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-1.5, 0.5, 2.0],
            [0.0, 0.0, 0.0],
        ];
        let exact: Vec<f64> = base
            .iter()
            .flat_map(|row| {
                (0..dim).map(|i| row[i % 3] * if (i / 3) % 2 == 0 { 1.0 } else { -0.5 })
            })
            .collect();
        let values: ArrayRef = match element {
            DataType::Float16 => Arc::new(Float16Array::from_iter_values(
                exact.iter().map(|&v| f16::from_f64(v)),
            )),
            DataType::Float32 => Arc::new(Float32Array::from_iter_values(
                exact.iter().map(|&v| v as f32),
            )),
            _ => Arc::new(Float64Array::from(exact.clone())),
        };
        let source = source(values, dim as i32);
        let scorer = FlatPairScorer::new(FLAT_COLUMN, source.schema_ref(), metric).unwrap();
        let actual = score_all(&scorer, &source);
        let rows: Vec<&[f64]> = exact.chunks_exact(dim).collect();
        for (a, x) in rows.iter().enumerate() {
            for (b, y) in rows.iter().enumerate() {
                let dot: f64 = x.iter().zip(*y).map(|(x, y)| x * y).sum();
                let norm = |v: &[f64]| v.iter().map(|v| v * v).sum::<f64>().sqrt();
                let expected = match metric {
                    DistanceType::L2 => x.iter().zip(*y).map(|(x, y)| (x - y).powi(2)).sum(),
                    DistanceType::Dot => 1.0 - dot,
                    _ => 1.0 - dot / (norm(x) * norm(y)),
                };
                let distance = actual[a][b];
                if expected.is_nan() {
                    assert!(distance.is_nan(), "{a} {b} {distance}");
                } else {
                    assert!(
                        (f64::from(distance) - expected).abs() <= 1e-6,
                        "{metric} {element} a={a} b={b} actual={distance} expected={expected}"
                    );
                }
            }
        }
        if metric != DistanceType::Dot {
            // Identical vectors score exactly zero, including after renormalization.
            assert_eq!(actual[0][1], 0.0);
            assert_eq!(actual[3][3], 0.0);
        }
    }

    /// lance-linalg's `dot_scalar::<f32, f32, 16>`, which is private there.
    fn dot_reference(x: &[f32], y: &[f32]) -> f32 {
        let x_chunks = y.chunks_exact(16);
        let y_chunks = x.chunks_exact(16);
        let sum = if x_chunks.remainder().is_empty() {
            0.0
        } else {
            x_chunks
                .remainder()
                .iter()
                .zip(y_chunks.remainder())
                .map(|(&x, &y)| x * y)
                .sum::<f32>()
        };
        let mut sums = [0.0f32; 16];
        for (x, y) in x_chunks.zip(y_chunks) {
            for i in 0..16 {
                sums[i] += x[i] * y[i];
            }
        }
        sum + sums.iter().copied().sum::<f32>()
    }

    fn assert_same_bits(actual: &[f32], expected: &[f32], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.to_bits() == e.to_bits() || (a.is_nan() && e.is_nan()),
                "{context} row={i} actual={a:e} expected={e:e}"
            );
        }
    }

    /// The blocked f32 kernels reproduce lance-linalg's 16-lane kernels bit
    /// for bit on every backend and for every block split of the rows.
    #[rstest]
    fn test_f32_rows_bit_identical(
        #[values(17, 32, 35, 64, 100, 1536)] dim: usize,
        #[values(true, false)] l2: bool,
    ) {
        use rand::{Rng, SeedableRng, rngs::SmallRng};

        let num_rows = 11;
        let mut rng = SmallRng::seed_from_u64(dim as u64);
        let anchor: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut rows: Vec<f32> = (0..num_rows * dim)
            .map(|i| rng.random_range(-1.0f32..1.0) * 10f32.powi((i / dim % 5) as i32 - 2))
            .collect();
        rows[3 * dim..4 * dim].copy_from_slice(&anchor);
        rows[5 * dim + dim / 2] = f32::NAN;
        rows[6 * dim + 1] = f32::INFINITY;

        let reference = |y: &[f32]| {
            if l2 {
                lance_linalg::distance::l2_scalar::<f32, f32, 16>(&anchor, y)
            } else {
                dot_reference(&anchor, y)
            }
        };
        let expected: Vec<f32> = rows.chunks_exact(dim).map(reference).collect();
        if l2 {
            assert_eq!(expected[3], 0.0);
        }
        // lance-linalg's batch kernels are these 16-lane kernels on the
        // default builds; sub-AVX2 x86 builds dispatch per host instead.
        #[cfg(any(
            not(target_arch = "x86_64"),
            all(target_feature = "avx2", target_feature = "fma")
        ))]
        {
            let mut batch = vec![0.0; num_rows];
            batch_rows_kernel::<f32>(if l2 {
                DistanceType::L2
            } else {
                DistanceType::Dot
            })(&anchor, &rows, &mut batch);
            assert_same_bits(&batch, &expected, "lance-linalg batch");
        }

        // Shift both inputs across every float offset within 64 bytes, as
        // the AVX-512 kernel realigns its loads to the rows.
        let shifted = |values: &[f32], offset: usize| {
            let mut buffer = vec![f32::NAN; values.len() + offset];
            buffer[offset..].copy_from_slice(values);
            buffer
        };
        for (anchor_offset, rows_offset) in [(0, 0), (0, 5), (3, 0), (4, 4), (15, 1), (9, 12)] {
            let anchor = shifted(&anchor, anchor_offset);
            let anchor = &anchor[anchor_offset..];
            let rows = shifted(&rows, rows_offset);
            let rows = &rows[rows_offset..];
            for (name, kernel) in f32_rows::backends(l2) {
                for start in 0..num_rows {
                    for end in start + 1..=num_rows {
                        let mut out = vec![0.0; end - start];
                        kernel(anchor, &rows[start * dim..end * dim], &mut out);
                        assert_same_bits(
                            &out,
                            &expected[start..end],
                            &format!(
                                "{name} dim={dim} l2={l2} rows={start}..{end} \
                                 offsets={anchor_offset},{rows_offset}"
                            ),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_flat_hamming() {
        let values = UInt8Array::from(vec![0b1010_1010, 0xff, 0b1010_1011, 0x0f, 0, 0]);
        let source = source(Arc::new(values), 2);
        let scorer =
            FlatPairScorer::new(FLAT_COLUMN, source.schema_ref(), DistanceType::Hamming).unwrap();
        let actual = score_all(&scorer, &source);
        assert_eq!(actual[0], vec![0.0, 5.0, 12.0]);
        assert_eq!(actual[1][2], 9.0);
        let err = FlatPairScorer::new(FLAT_COLUMN, source.schema_ref(), DistanceType::L2)
            .err()
            .unwrap();
        assert!(matches!(err, Error::NotSupported { .. }), "{err}");
    }
}
