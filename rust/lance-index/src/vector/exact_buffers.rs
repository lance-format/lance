// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Index cache entries in buffers of exactly their values' size.
//!
//! The index cache charges an entry what its arrays allocate. A plane-row
//! read of an IVF_RQ file unpacks its values into buffers of exactly their
//! size, while a read of the column layout can keep more: a decoder that
//! joins the pages a read spans rounds its buffers up, and a slice keeps the
//! whole buffer it was cut from. Reads that become cache entries are copied
//! here into exact buffers where they are not already, so the entries of
//! both layouts, and of native and layered indexes alike, weigh their
//! values.

use std::sync::Arc;

use arrow::array::ArrayData;
use arrow::buffer::{Buffer, MutableBuffer};
use arrow::compute::concat_batches;
use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, make_array};
use arrow_schema::{DataType, SchemaRef};
use arrow_select::concat::concat;
use lance_arrow::deepcopy::deep_copy_array_sliced;
use lance_core::Result;

/// Whether `data` holds exactly its values: a fixed-width column, or a
/// fixed-size list of one, without nulls, whose buffer allocates its values'
/// bytes and no more. Other layouts are never exact here.
fn is_exact(data: &ArrayData) -> bool {
    if data.nulls().is_some() {
        return false;
    }
    match data.data_type() {
        DataType::FixedSizeList(_, size) => match data.child_data() {
            [child] => {
                data.offset() == 0 && child.len() == data.len() * *size as usize && is_exact(child)
            }
            _ => false,
        },
        data_type => match (data_type.primitive_width(), data.buffers()) {
            (Some(width), [values]) => values.capacity() == data.len() * width,
            _ => false,
        },
    }
}

/// `pieces`, consecutive parts of one column, as one array in buffers of
/// exactly its values' size; `None` for a layout [`is_exact`] does not size
/// or for pieces with nulls.
fn exact_concat(pieces: &[ArrayData]) -> Option<ArrayData> {
    let data_type = pieces.first()?.data_type();
    if pieces.iter().any(|piece| piece.nulls().is_some()) {
        return None;
    }
    let len = pieces.iter().map(ArrayData::len).sum();
    match data_type {
        DataType::FixedSizeList(_, size) => {
            let size = *size as usize;
            let children = pieces
                .iter()
                .map(|piece| match piece.child_data() {
                    [child] => Some(child.slice(piece.offset() * size, piece.len() * size)),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            let child = exact_concat(&children)?;
            ArrayData::builder(data_type.clone())
                .len(len)
                .child_data(vec![child])
                .build()
                .ok()
        }
        _ => {
            let width = data_type.primitive_width()?;
            let mut values = MutableBuffer::from_len_zeroed(len * width);
            let mut written = 0;
            for piece in pieces {
                let [buffer] = piece.buffers() else {
                    return None;
                };
                let start = piece.offset() * width;
                let bytes = buffer.as_slice().get(start..start + piece.len() * width)?;
                values.as_slice_mut()[written..written + bytes.len()].copy_from_slice(bytes);
                written += bytes.len();
            }
            ArrayData::builder(data_type.clone())
                .len(len)
                .add_buffer(Buffer::from(values))
                .build()
                .ok()
        }
    }
}

/// `pieces`, consecutive parts of one column, as one array whose
/// fixed-width values sit in a buffer of exactly their size. A single piece
/// that already does is kept without a copy, unless `copy` asks for one so
/// that the array never shares the buffer of a larger read. A layout
/// [`exact_concat`] does not size is concatenated, or copied, as is.
pub fn exact_array(pieces: &[&ArrayRef], copy: bool) -> Result<ArrayRef> {
    let data: Vec<ArrayData> = pieces.iter().map(|piece| piece.to_data()).collect();
    if let [single] = data.as_slice()
        && !copy
        && is_exact(single)
    {
        return Ok(pieces[0].clone());
    }
    if let Some(exact) = exact_concat(&data) {
        return Ok(make_array(exact));
    }
    Ok(match pieces {
        [single] if copy => deep_copy_array_sliced(single.as_ref()),
        [single] => (*single).clone(),
        pieces => {
            let pieces: Vec<&dyn Array> = pieces.iter().map(|piece| piece.as_ref()).collect();
            concat(&pieces)?
        }
    })
}

/// `batches`, consecutive reads of `schema`'s columns, as one batch whose
/// columns are [`exact_array`]s.
pub fn exact_batch(schema: &SchemaRef, batches: &[RecordBatch], copy: bool) -> Result<RecordBatch> {
    if batches.is_empty() {
        return Ok(concat_batches(schema, batches)?);
    }
    let rows = batches.iter().map(RecordBatch::num_rows).sum();
    let columns = (0..schema.fields().len())
        .map(|column| {
            let pieces: Vec<&ArrayRef> = batches.iter().map(|batch| batch.column(column)).collect();
            exact_array(&pieces, copy)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )?)
}

/// Bytes `array`'s buffers allocate, as the index cache charges them.
#[cfg(test)]
pub fn allocated_bytes(array: &dyn Array) -> usize {
    fn walk(data: &ArrayData) -> usize {
        data.buffers().iter().map(Buffer::capacity).sum::<usize>()
            + data
                .nulls()
                .map_or(0, |nulls| nulls.inner().inner().capacity())
            + data.child_data().iter().map(walk).sum::<usize>()
    }
    walk(&array.to_data())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::types::{Float32Type, UInt8Type, UInt64Type};
    use arrow_array::{
        FixedSizeListArray, Float32Array, StringArray, UInt8Array, UInt64Array, cast::AsArray,
    };
    use arrow_schema::{Field, Schema};
    use lance_arrow::FixedSizeListArrayExt;

    /// `rows` lists of `width` bytes, in a buffer of exactly their size.
    fn codes(rows: usize, width: i32) -> ArrayRef {
        let values: Vec<u8> = (0..rows * width as usize).map(|v| v as u8).collect();
        Arc::new(FixedSizeListArray::try_new_from_values(UInt8Array::from(values), width).unwrap())
    }

    /// Joined pieces and slices are copied into buffers of exactly their
    /// values, which they equal; exact columns are kept, and copied only
    /// when asked.
    #[test]
    fn exact_array_sizes_fixed_width_columns() {
        let rows: ArrayRef = Arc::new(UInt64Array::from((0..100).collect::<Vec<u64>>()));
        let factors: ArrayRef = Arc::new(Float32Array::from(
            (0..100).map(|v| v as f32).collect::<Vec<_>>(),
        ));
        let codes = codes(100, 3);
        for array in [&rows, &factors, &codes] {
            assert!(is_exact(&array.to_data()), "{array:?}");
            assert!(Arc::ptr_eq(&exact_array(&[array], false).unwrap(), array));
            let copied = exact_array(&[array], true).unwrap();
            assert_eq!(copied.as_ref(), array.as_ref());
            assert_eq!(
                allocated_bytes(copied.as_ref()),
                allocated_bytes(array.as_ref())
            );

            // A slice keeps the whole buffer; its exact copy holds its rows.
            let slice = array.slice(10, 30);
            assert!(!is_exact(&slice.to_data()));
            let exact = exact_array(&[&slice], false).unwrap();
            assert_eq!(exact.as_ref(), slice.as_ref());
            assert!(is_exact(&exact.to_data()));
            assert_eq!(
                allocated_bytes(exact.as_ref()),
                allocated_bytes(array.as_ref()) * 30 / 100
            );

            // Pieces join into one exact buffer, where a concatenation
            // rounds its buffer up.
            let pieces = [array.slice(0, 33), array.slice(33, 67)];
            let joined = exact_array(&[&pieces[0], &pieces[1]], false).unwrap();
            assert_eq!(joined.as_ref(), array.as_ref());
            assert_eq!(
                allocated_bytes(joined.as_ref()),
                allocated_bytes(array.as_ref())
            );
        }
        assert_eq!(
            exact_array(&[&codes.slice(5, 2)], false)
                .unwrap()
                .as_fixed_size_list()
                .values()
                .as_primitive::<UInt8Type>()
                .values()
                .as_ref(),
            &[15, 16, 17, 18, 19, 20]
        );
        assert_eq!(
            exact_array(&[&rows.slice(98, 2)], false)
                .unwrap()
                .as_primitive::<UInt64Type>()
                .values()
                .as_ref(),
            &[98, 99]
        );
        assert_eq!(
            exact_array(&[&factors.slice(0, 1)], true)
                .unwrap()
                .as_primitive::<Float32Type>()
                .value(0),
            0.0
        );
    }

    /// Columns with nulls or without a fixed width keep the decoder's
    /// layout: kept, or concatenated, or copied as a slice.
    #[test]
    fn exact_array_keeps_other_layouts() {
        let nullable: ArrayRef = Arc::new(Float32Array::from(vec![Some(1.0), None, Some(3.0)]));
        let names: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        for array in [&nullable, &names] {
            assert!(!is_exact(&array.to_data()));
            assert_eq!(
                exact_array(&[array], false).unwrap().as_ref(),
                array.as_ref()
            );
            assert_eq!(
                exact_array(&[array], true).unwrap().as_ref(),
                array.as_ref()
            );
            let pieces = [array.slice(0, 1), array.slice(1, 2)];
            assert_eq!(
                exact_array(&[&pieces[0], &pieces[1]], false)
                    .unwrap()
                    .as_ref(),
                array.as_ref()
            );
        }
    }

    #[test]
    fn exact_batch_joins_reads_of_a_schema() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", arrow_schema::DataType::UInt64, false),
            Field::new("codes", codes(1, 4).data_type().clone(), true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from((0..10).collect::<Vec<u64>>())),
                codes(10, 4),
            ],
        )
        .unwrap();
        let joined = exact_batch(&schema, &[batch.slice(0, 4), batch.slice(4, 6)], false).unwrap();
        assert_eq!(joined, batch);
        for column in joined.columns() {
            assert!(is_exact(&column.to_data()));
        }
        assert_eq!(exact_batch(&schema, &[], false).unwrap().num_rows(), 0);
    }
}
