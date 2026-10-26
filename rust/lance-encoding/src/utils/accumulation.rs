// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An accumulation queue accumulates arrays until we have enough data to flush.

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use lance_arrow::deepcopy::{deep_copy_array, deep_copy_array_sliced};
use log::{debug, trace};

#[derive(Debug)]
pub struct AccumulationQueue {
    cache_bytes: u64,
    keep_original_array: bool,
    buffered_arrays: Vec<ArrayRef>,
    current_bytes: u64,
    // Row number of the first item in buffered_arrays, reset on flush
    row_number: u64,
    // Number of top level rows represented in buffered_arrays, reset on flush
    num_rows: u64,
    // This is only for logging / debugging purposes
    column_index: u32,
}

impl AccumulationQueue {
    pub fn new(cache_bytes: u64, column_index: u32, keep_original_array: bool) -> Self {
        Self {
            cache_bytes,
            buffered_arrays: Vec::new(),
            current_bytes: 0,
            column_index,
            keep_original_array,
            row_number: u64::MAX,
            num_rows: 0,
        }
    }

    /// Adds an array to the queue, if there is enough data then the queue is flushed
    /// and returned
    pub fn insert(
        &mut self,
        mut array: ArrayRef,
        row_number: u64,
        num_rows: u64,
    ) -> Option<(Vec<ArrayRef>, u64, u64)> {
        if !self.keep_original_array {
            let data = array.to_data();
            // Compacting nested and view arrays can retain shared child buffers.
            // Preserve their existing deep-copy behavior.
            let can_compact = data.child_data().is_empty()
                && !matches!(data.data_type(), DataType::Utf8View | DataType::BinaryView);
            let bytes = if can_compact {
                data.get_slice_memory_size()
                    .unwrap_or_else(|_| array.get_array_memory_size())
            } else {
                array.get_array_memory_size()
            };
            // A slice can report its parent's capacity. Copy before accounting,
            // unless its visible data already fills the remaining page budget.
            if bytes as u64 <= self.cache_bytes - self.current_bytes {
                array = if can_compact {
                    deep_copy_array_sliced(array.as_ref())
                } else {
                    deep_copy_array(array.as_ref())
                };
            }
        }
        if self.row_number == u64::MAX {
            self.row_number = row_number;
        }
        self.num_rows += num_rows;
        self.current_bytes += array.get_array_memory_size() as u64;
        self.buffered_arrays.push(array);
        if self.current_bytes > self.cache_bytes {
            debug!(
                "Flushing column {} page of size {} bytes (unencoded)",
                self.column_index, self.current_bytes
            );
            self.current_bytes = 0;
            let row_number = self.row_number;
            self.row_number = u64::MAX;
            let num_rows = self.num_rows;
            self.num_rows = 0;
            Some((
                std::mem::take(&mut self.buffered_arrays),
                row_number,
                num_rows,
            ))
        } else {
            trace!(
                "Accumulating data for column {}.  Now at {} bytes",
                self.column_index, self.current_bytes
            );
            None
        }
    }

    pub fn flush(&mut self) -> Option<(Vec<ArrayRef>, u64, u64)> {
        if self.buffered_arrays.is_empty() {
            trace!(
                "No final flush since no data at column {}",
                self.column_index
            );
            None
        } else {
            trace!(
                "Final flush of column {} which has {} bytes",
                self.column_index, self.current_bytes
            );
            self.current_bytes = 0;
            let row_number = self.row_number;
            self.row_number = u64::MAX;
            let num_rows = self.num_rows;
            self.num_rows = 0;
            Some((
                std::mem::take(&mut self.buffered_arrays),
                row_number,
                num_rows,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, BooleanArray, StringArray, UInt64Array};
    use arrow_schema::DataType;
    use lance_arrow::deepcopy::deep_copy_array_sliced;
    use rstest::rstest;

    use super::AccumulationQueue;

    #[rstest]
    #[case::primitive(DataType::UInt64)]
    #[case::string(DataType::Utf8)]
    #[case::boolean(DataType::Boolean)]
    fn test_accumulate_slices_by_retained_size(#[case] data_type: DataType) {
        let parent: ArrayRef = match data_type {
            DataType::UInt64 => Arc::new(UInt64Array::from(
                (0..4096)
                    .map(|i| (i % 3 != 0).then_some(i))
                    .collect::<Vec<_>>(),
            )),
            DataType::Utf8 => Arc::new(StringArray::from(
                (0..4096)
                    .map(|i| (i % 3 != 0).then(|| format!("{i:064}")))
                    .collect::<Vec<_>>(),
            )),
            DataType::Boolean => Arc::new(BooleanArray::from(
                (0..4096)
                    .map(|i| (i % 3 != 0).then_some(i % 2 == 0))
                    .collect::<Vec<_>>(),
            )),
            _ => unreachable!(),
        };
        let slice = parent.slice(17, 32);
        let copied_bytes = deep_copy_array_sliced(slice.as_ref()).get_array_memory_size() as u64;
        let budget = copied_bytes * 2;
        assert!(slice.get_array_memory_size() as u64 > budget);
        let mut queue = AccumulationQueue::new(budget, 0, false);

        assert!(queue.insert(slice.clone(), 17, 32).is_none());
        assert_eq!(queue.current_bytes, copied_bytes);
        assert!(queue.insert(slice.clone(), 49, 32).is_none());
        let (arrays, first_row, rows) = queue.insert(slice.clone(), 81, 32).unwrap();
        assert_eq!((first_row, rows), (17, 96));
        assert_eq!(arrays.len(), 3);
        for array in arrays {
            assert_eq!(array.as_ref(), slice.as_ref());
        }
        assert!(queue.flush().is_none());

        assert!(queue.insert(slice.clone(), 113, 32).is_none());
        let (arrays, first_row, rows) = queue.flush().unwrap();
        assert_eq!((first_row, rows), (113, 32));
        assert_eq!(arrays[0].as_ref(), slice.as_ref());
    }

    #[rstest]
    #[case::keep_original(true, 32)]
    #[case::large_batch(false, 4096)]
    fn test_flush_without_copy(#[case] keep_original: bool, #[case] rows: usize) {
        let parent: ArrayRef = Arc::new(UInt64Array::from(vec![1; 4096]));
        let array = parent.slice(0, rows);
        let mut queue = AccumulationQueue::new(512, 0, keep_original);
        let (arrays, first_row, num_rows) = queue.insert(array.clone(), 0, rows as u64).unwrap();
        assert_eq!((first_row, num_rows), (0, rows as u64));
        assert!(Arc::ptr_eq(&arrays[0], &array));
    }
}
