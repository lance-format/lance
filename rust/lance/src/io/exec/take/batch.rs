// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Size checks before combining decoded take results, including repeated rows.

use std::collections::HashMap;

use arrow_array::{Array, RecordBatch, UInt32Array, make_array};
use arrow_schema::{ArrowError, DataType};
use arrow_select::{concat::concat_batches, take::take_record_batch};
use lance_arrow::interleave_batches;

/// Measure logical data rather than backing buffers, which may belong to a
/// much larger sliced array. Counting at least one byte per child value also
/// bounds List<Null> offsets, even though those children have no value buffer.
fn array_size(array: &dyn Array) -> Result<usize, ArrowError> {
    let (child, start, len, offset_bytes) = match array.data_type() {
        DataType::Utf8View | DataType::BinaryView => return Ok(array.get_array_memory_size()),
        DataType::List(_) | DataType::Map(_, _) => {
            let data = array.to_data();
            let offsets = data.buffer::<i32>(0);
            let start = offsets[0] as usize;
            let end = offsets[array.len()] as usize;
            (
                make_array(data.child_data()[0].clone()),
                start,
                end - start,
                (array.len() + 1) * 4,
            )
        }
        DataType::LargeList(_) => {
            let data = array.to_data();
            let offsets = data.buffer::<i64>(0);
            let start = offsets[0] as usize;
            let end = offsets[array.len()] as usize;
            (
                make_array(data.child_data()[0].clone()),
                start,
                end - start,
                (array.len() + 1) * 8,
            )
        }
        DataType::FixedSizeList(_, width) => {
            let data = array.to_data();
            (
                make_array(data.child_data()[0].clone()),
                data.offset() * *width as usize,
                array.len() * *width as usize,
                0,
            )
        }
        DataType::Struct(_) => {
            let data = array.to_data();
            return data
                .child_data()
                .iter()
                .try_fold(array.len(), |size, child| {
                    Ok(size + array_size(make_array(child.clone()).as_ref())?)
                });
        }
        _ => {
            let data = array.to_data();
            // Views need their backing buffers included. Dictionary values
            // and other child layouts are conservatively counted in full.
            let size = match data.get_slice_memory_size() {
                Ok(size) => size,
                Err(ArrowError::NotYetImplemented(_)) => array.get_array_memory_size(),
                Err(error) => return Err(error),
            };
            return data
                .child_data()
                .iter()
                .try_fold(size.max(array.len()), |size, child| {
                    Ok(size + array_size(make_array(child.clone()).as_ref())?)
                });
        }
    };
    Ok(offset_bytes + array.len() + array_size(child.slice(start, len).as_ref())?)
}

/// Concat and reorder only when the result fits. Otherwise interleave bounded
/// selections directly from the decoded batches, never forming an oversized
/// intermediate array. A single valid row is indivisible and may exceed the
/// injected budget.
pub(super) fn take_batches(
    batches: &[RecordBatch],
    indices: Option<&UInt32Array>,
    max_bytes: usize,
) -> Result<Vec<RecordBatch>, ArrowError> {
    let num_rows = batches.iter().map(RecordBatch::num_rows).sum();
    let max_repetitions = if let Some(indices) = indices {
        let mut repetitions = vec![0_usize; num_rows];
        for &index in indices.values() {
            repetitions[index as usize] += 1;
        }
        repetitions.iter().copied().max().unwrap_or(1)
    } else {
        1
    };
    let source_bytes = batches.iter().try_fold(0_usize, |size, batch| {
        batch.columns().iter().try_fold(size, |size, column| {
            Ok::<_, ArrowError>(size + array_size(column.as_ref())?)
        })
    })?;
    if source_bytes
        .checked_mul(max_repetitions)
        .is_some_and(|size| size <= max_bytes)
    {
        let batch = concat_batches(&batches[0].schema(), batches)?;
        return Ok(vec![if let Some(indices) = indices {
            take_record_batch(&batch, indices)?
        } else {
            batch
        }]);
    }

    let mut locations = Vec::with_capacity(num_rows);
    let mut row_bytes = Vec::with_capacity(num_rows);
    for (batch_index, batch) in batches.iter().enumerate() {
        for row in 0..batch.num_rows() {
            locations.push((batch_index, row));
            row_bytes.push(batch.columns().iter().try_fold(0_usize, |size, column| {
                Ok::<_, ArrowError>(size + array_size(column.slice(row, 1).as_ref())?)
            })?);
        }
    }
    let selection = if let Some(indices) = indices {
        indices
            .values()
            .iter()
            .map(|&i| i as usize)
            .collect::<Vec<_>>()
    } else {
        (0..num_rows).collect()
    };
    let mut output = Vec::new();
    let mut selected = Vec::new();
    let mut bytes = 0_usize;
    for index in selection {
        let size = row_bytes[index];
        if !selected.is_empty() && bytes.checked_add(size).is_none_or(|n| n > max_bytes) {
            output.push(interleave_selection(batches, &selected)?);
            selected.clear();
            bytes = 0;
        }
        selected.push(locations[index]);
        bytes += size;
    }
    if !selected.is_empty() {
        output.push(interleave_selection(batches, &selected)?);
    }
    Ok(output)
}

fn interleave_selection(
    batches: &[RecordBatch],
    selection: &[(usize, usize)],
) -> Result<RecordBatch, ArrowError> {
    // Unselected dictionaries must not contribute their values to a concat
    // inside Arrow's interleave kernel.
    let mut sources = Vec::new();
    let mut source_indices = HashMap::new();
    let indices = selection
        .iter()
        .map(|&(source, row)| {
            let source = *source_indices.entry(source).or_insert_with(|| {
                sources.push(batches[source].clone());
                sources.len() - 1
            });
            (source, row)
        })
        .collect::<Vec<_>>();
    interleave_batches(&sources, &indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow_array::builder::{ListBuilder, StringBuilder};
    use arrow_array::{ArrayRef, ListArray, NullArray, StringArray, StructArray, record_batch};
    use arrow_buffer::OffsetBuffer;
    use arrow_schema::Field;
    use rstest::rstest;

    #[rstest]
    #[case::utf8(DataType::Utf8)]
    #[case::large_utf8(DataType::LargeUtf8)]
    #[case::utf8_view(DataType::Utf8View)]
    #[case::binary(DataType::Binary)]
    #[case::large_binary(DataType::LargeBinary)]
    #[case::binary_view(DataType::BinaryView)]
    fn test_byte_limit_before_concat_and_expansion(#[case] data_type: DataType) {
        let large = "x".repeat(80);
        let array = StringArray::from(vec![Some("a"), None, Some(&large), Some("")]);
        let array = arrow_cast::cast(&array, &data_type).unwrap();
        let batch = RecordBatch::try_from_iter([("value", array)]).unwrap();
        // Slices keep the original buffers, but budgeting must use their rows.
        let batches = vec![batch.slice(0, 2), batch.slice(2, 2)];
        let indices = UInt32Array::from(vec![3, 0, 2, 2, 1, 0]);
        let output = take_batches(&batches, Some(&indices), 100).unwrap();
        assert!(output.len() > 1);
        for batch in &output {
            assert!(array_size(batch.column(0).as_ref()).unwrap() <= 100 || batch.num_rows() == 1);
        }
        assert_eq!(
            concat_batches(&batch.schema(), &output).unwrap(),
            take_record_batch(&batch, &indices).unwrap()
        );
    }

    #[rstest]
    #[case::list(false)]
    #[case::struct_of_list(true)]
    fn test_nested_skewed_and_null_rows(#[case] in_struct: bool) {
        let mut builder = ListBuilder::new(StringBuilder::new());
        builder.values().append_value("x".repeat(80));
        builder.append(true);
        builder.append(false);
        builder.append(true);
        builder.values().append_null();
        builder.append(true);
        builder.values().append_value("a");
        builder.append(true);
        let mut array: ArrayRef = Arc::new(builder.finish());
        if in_struct {
            array = Arc::new(StructArray::new(
                vec![Arc::new(Field::new(
                    "items",
                    array.data_type().clone(),
                    true,
                ))]
                .into(),
                vec![array],
                None,
            ));
        }
        let batch = RecordBatch::try_from_iter([("value", array)]).unwrap();
        let batches = vec![batch.slice(0, 2), batch.slice(2, 3)];
        let indices = UInt32Array::from(vec![4, 0, 3, 0, 1, 2]);
        let output = take_batches(&batches, Some(&indices), 110).unwrap();
        assert!(output.len() > 1);
        for batch in &output {
            assert!(array_size(batch.column(0).as_ref()).unwrap() <= 110 || batch.num_rows() == 1);
        }
        assert_eq!(
            concat_batches(&batch.schema(), &output).unwrap(),
            take_record_batch(&batch, &indices).unwrap()
        );
    }

    #[test]
    fn test_list_null_children_consume_offset_capacity() {
        let array = ListArray::new(
            Arc::new(Field::new("item", DataType::Null, true)),
            OffsetBuffer::new(vec![0, 50, 100].into()),
            Arc::new(NullArray::new(100)),
            None,
        );
        let batch = RecordBatch::try_from_iter([("value", Arc::new(array) as ArrayRef)]).unwrap();
        let output = take_batches(std::slice::from_ref(&batch), None, 70).unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(concat_batches(&batch.schema(), &output).unwrap(), batch);
    }

    #[test]
    fn test_batches_within_budget_keep_shape() {
        let batch = record_batch!(("value", Utf8, ["a", "b", "c"])).unwrap();
        let batches = vec![batch.slice(0, 1), batch.slice(1, 2)];
        assert_eq!(take_batches(&batches, None, 100).unwrap(), vec![batch]);
    }
}
