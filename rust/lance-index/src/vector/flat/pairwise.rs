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
        ) -> Result<FloatKernel<'a, T::Native>> {
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
}

impl<T: L2 + Dot> FloatKernel<'_, T> {
    fn distances(&self, anchor_row: usize, candidates: Range<usize>, out: &mut [f32]) {
        let dim = self.dim;
        let x = &self.anchor[anchor_row * dim..(anchor_row + 1) * dim];
        let y = &self.candidates[candidates.start * dim..candidates.end * dim];
        if self.metric == DistanceType::Dot {
            for (out, dot) in out.iter_mut().zip(T::dot_batch(x, y, dim)) {
                *out = 1.0 - dot;
            }
            return;
        }
        for (out, l2) in out.iter_mut().zip(T::l2_batch(x, y, dim)) {
            *out = l2;
        }
        if self.metric == DistanceType::Cosine {
            l2_to_cosine(
                out,
                self.anchor_norms[anchor_row],
                &self.candidate_norms[candidates],
            );
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
    ) {
        let dim = 3;
        // Small integers and halves are exact in every float width.
        let exact: Vec<f64> = vec![
            1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            -1.5, 0.5, 2.0, //
            0.0, 0.0, 0.0, //
        ];
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
