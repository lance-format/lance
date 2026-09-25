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
    table: Vec<f32>,
    /// Cosine only: `‖c_sub + codeword‖²` per `[sub][code]`, where `c` is the
    /// partition centroid that residual codes are relative to.
    squared_norms: Vec<f64>,
}

impl PQPairScorer {
    pub(crate) fn new(
        pq: &ProductQuantizer,
        centroid: &ArrayRef,
        metric: DistanceType,
    ) -> Result<Self> {
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
            table,
            squared_norms,
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

impl PairKernel for PQKernel<'_> {
    fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        let scorer = self.scorer;
        let num_centroids = scorer.num_centroids();
        let table_row = |sub: usize, code: u8| {
            let start = (sub * num_centroids + usize::from(code)) * num_centroids;
            &scorer.table[start..start + num_centroids]
        };
        // Sum sub-vectors in order from zero for every candidate, so each
        // distance is independent of the candidate range.
        out.fill(0.0);
        for byte in 0..scorer.code_bytes {
            let code = self.anchor[byte * self.anchor_stride + anchor_row];
            let start = byte * self.candidate_stride;
            let codes = &self.candidates[start + candidates.start..start + candidates.end];
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
        let scorer = PQPairScorer::new(
            &pq,
            &(Arc::new(Float32Array::from(centroid.clone())) as ArrayRef),
            metric,
        )
        .unwrap();
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
                    "bits={num_bits} {metric} a={a} b={b} actual={distance} expected={expected}"
                );
                if subs[a] == subs[b] && metric != DistanceType::Dot {
                    assert_eq!(distance, 0.0, "a={a} b={b}");
                }
            }
        }
    }
}
