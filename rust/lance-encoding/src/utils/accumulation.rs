// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! An accumulation queue accumulates arrays until we have enough data to flush.

use arrow_array::cast::AsArray;
use arrow_array::types::{BinaryType, ByteArrayType, LargeBinaryType, LargeUtf8Type, Utf8Type};
use arrow_array::{Array, ArrayRef, GenericByteArray};
use arrow_buffer::{ArrowNativeType, bit_util::round_upto_multiple_of_64};
use arrow_schema::DataType;
use lance_arrow::deepcopy::{deep_copy_array, deep_copy_array_sliced};
use log::{debug, trace};

// Value buffers are rounded to 64 bytes; compact validity buffers use exact byte lengths.
// Inspect lengths without building ArrayData, so the copy decision requires no allocations.
fn flat_copy_sizes(array: &dyn Array) -> Option<(usize, usize)> {
    let (mut copied_bytes, mut compacted_bytes, mut has_unused_bytes) = match array.data_type() {
        DataType::Null => (0, 0, false),
        DataType::Boolean => {
            let buffer_bytes = array.as_boolean().values().inner().len();
            let value_bytes = array.len().div_ceil(8);
            (
                round_upto_multiple_of_64(buffer_bytes),
                round_upto_multiple_of_64(value_bytes),
                buffer_bytes > value_bytes,
            )
        }
        DataType::Utf8 => byte_copy_sizes(array.as_bytes::<Utf8Type>()),
        DataType::LargeUtf8 => byte_copy_sizes(array.as_bytes::<LargeUtf8Type>()),
        DataType::Binary => byte_copy_sizes(array.as_bytes::<BinaryType>()),
        DataType::LargeBinary => byte_copy_sizes(array.as_bytes::<LargeBinaryType>()),
        DataType::FixedSizeBinary(_) => {
            let bytes = round_upto_multiple_of_64(array.as_fixed_size_binary().values().len());
            (bytes, bytes, false)
        }
        data_type => {
            let width = data_type.primitive_width()?;
            let bytes = round_upto_multiple_of_64(array.len() * width);
            (bytes, bytes, false)
        }
    };
    if let Some(nulls) = array.nulls() {
        let validity_bytes = nulls.len().div_ceil(8);
        copied_bytes += round_upto_multiple_of_64(nulls.buffer().len());
        if nulls.null_count() > 0 {
            compacted_bytes += validity_bytes;
        }
        has_unused_bytes |= nulls.buffer().len() > validity_bytes;
    }
    // Reserve compaction for buffers containing data outside the slice.
    if !has_unused_bytes {
        compacted_bytes = copied_bytes;
    }
    let array_bytes = std::mem::size_of_val(array);
    Some((copied_bytes + array_bytes, compacted_bytes + array_bytes))
}

fn byte_copy_sizes<T: ByteArrayType>(array: &GenericByteArray<T>) -> (usize, usize, bool) {
    let offsets = array.value_offsets();
    let offset_bytes = round_upto_multiple_of_64(std::mem::size_of_val(offsets));
    let value_bytes = (offsets[array.len()] - offsets[0]).as_usize();
    let buffer_bytes = array.value_data().len();
    (
        offset_bytes + round_upto_multiple_of_64(buffer_bytes),
        offset_bytes + round_upto_multiple_of_64(value_bytes),
        buffer_bytes > value_bytes,
    )
}

#[derive(Debug)]
pub(crate) struct AccumulatedPage {
    pub(crate) arrays: Vec<ArrayRef>,
    pub(crate) row_number: u64,
    pub(crate) num_rows: u64,
    pub(crate) includes_inserted_array: bool,
}

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
        let mut is_full = false;
        if !self.keep_original_array {
            match flat_copy_sizes(array.as_ref()) {
                Some((copied_bytes, compacted_bytes)) => {
                    let remaining_bytes = self.remaining_bytes();
                    if copied_bytes <= compacted_bytes && copied_bytes as u64 <= remaining_bytes {
                        array = deep_copy_array(array.as_ref());
                    } else if compacted_bytes as u64 <= remaining_bytes {
                        array = deep_copy_array_sliced(array.as_ref());
                    } else {
                        // Copy alignment can exceed the budget even when the original fits.
                        is_full = true;
                    }
                }
                // Nested and view arrays retain their existing copy policy.
                None if array.get_array_memory_size() as u64 <= self.remaining_bytes() => {
                    array = deep_copy_array(array.as_ref());
                }
                None => {}
            }
        }
        if self.row_number == u64::MAX {
            self.row_number = row_number;
        }
        self.num_rows += num_rows;
        self.current_bytes += array.get_array_memory_size() as u64;
        self.buffered_arrays.push(array);
        if is_full || self.current_bytes > self.cache_bytes {
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

    pub(crate) fn insert_with_pages(
        &mut self,
        mut array: ArrayRef,
        row_number: u64,
        num_rows: u64,
    ) -> Vec<AccumulatedPage> {
        let mut pages = Vec::with_capacity(2);
        if !self.keep_original_array && !self.buffered_arrays.is_empty() {
            let required_bytes = match flat_copy_sizes(array.as_ref()) {
                Some((_, compacted_bytes)) => compacted_bytes as u64,
                None => array.get_array_memory_size() as u64,
            };
            if required_bytes > self.remaining_bytes()
                && let Some(page) = self.take_page(false)
            {
                // Do not join a borrowed input that cannot fit to a page whose
                // buffered values are already detached copies.  Encoding that
                // mixed page keeps the foreign source allocation alive while
                // both the copied input and the encoded output are resident.
                pages.push(page);
            }
        }

        let mut is_full = false;
        if !self.keep_original_array {
            match flat_copy_sizes(array.as_ref()) {
                Some((copied_bytes, compacted_bytes)) => {
                    let remaining_bytes = self.remaining_bytes();
                    if copied_bytes <= compacted_bytes && copied_bytes as u64 <= remaining_bytes {
                        array = deep_copy_array(array.as_ref());
                    } else if compacted_bytes as u64 <= remaining_bytes {
                        array = deep_copy_array_sliced(array.as_ref());
                    } else {
                        // Copy alignment can exceed the budget even when the original fits.
                        is_full = true;
                    }
                }
                // Nested and view arrays retain their existing copy policy.
                None if array.get_array_memory_size() as u64 <= self.remaining_bytes() => {
                    array = deep_copy_array(array.as_ref());
                }
                None => {}
            }
        }
        if self.row_number == u64::MAX {
            self.row_number = row_number;
        }
        self.num_rows += num_rows;
        self.current_bytes += array.get_array_memory_size() as u64;
        self.buffered_arrays.push(array);
        if (is_full || self.current_bytes > self.cache_bytes)
            && let Some(page) = self.take_page(true)
        {
            pages.push(page);
        }
        pages
    }

    pub(crate) fn insert_for_page_boundary(
        &mut self,
        array: ArrayRef,
        row_number: u64,
        num_rows: u64,
    ) -> Vec<AccumulatedPage> {
        let mut pages = Vec::with_capacity(2);
        let input_bytes = flat_copy_sizes(array.as_ref())
            .map(|(_, compacted_bytes)| compacted_bytes as u64)
            .unwrap_or_else(|| array.get_array_memory_size() as u64);
        // Keep ordinary page accumulation when the terminal input fits the same budget.
        let coalesce = self.current_bytes.saturating_add(input_bytes) <= self.cache_bytes;
        if !coalesce && let Some(page) = self.take_page(false) {
            pages.push(page);
        }
        if self.row_number == u64::MAX {
            self.row_number = row_number;
        }
        self.num_rows += num_rows;
        self.current_bytes = self
            .current_bytes
            .saturating_add(array.get_array_memory_size() as u64);
        self.buffered_arrays.push(array);
        if let Some(page) = self.take_page(true) {
            pages.push(page);
        }
        pages
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

    fn remaining_bytes(&self) -> u64 {
        self.cache_bytes.saturating_sub(self.current_bytes)
    }

    fn take_page(&mut self, includes_inserted_array: bool) -> Option<AccumulatedPage> {
        if self.buffered_arrays.is_empty() {
            self.current_bytes = 0;
            return None;
        }
        self.current_bytes = 0;
        let row_number = self.row_number;
        self.row_number = u64::MAX;
        let num_rows = self.num_rows;
        self.num_rows = 0;
        Some(AccumulatedPage {
            arrays: std::mem::take(&mut self.buffered_arrays),
            row_number,
            num_rows,
            includes_inserted_array,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, BooleanArray, StringArray, UInt64Array};
    use arrow_schema::DataType;
    use lance_arrow::deepcopy::{deep_copy_array, deep_copy_array_sliced};
    use rstest::rstest;

    use super::AccumulationQueue;

    #[rstest]
    #[case::primitive(DataType::UInt64)]
    #[case::string(DataType::Utf8)]
    #[case::large_string(DataType::LargeUtf8)]
    #[case::binary(DataType::Binary)]
    #[case::large_binary(DataType::LargeBinary)]
    #[case::boolean(DataType::Boolean)]
    fn test_accumulate_slices_by_retained_size(#[case] data_type: DataType) {
        let parent: ArrayRef = match data_type {
            DataType::UInt64 => Arc::new(UInt64Array::from(
                (0..4096)
                    .map(|i| (i % 3 != 0).then_some(i))
                    .collect::<Vec<_>>(),
            )),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
                let strings = StringArray::from(
                    (0..4096)
                        .map(|i| (i % 3 != 0).then(|| format!("{i:064}")))
                        .collect::<Vec<_>>(),
                );
                arrow_cast::cast(&strings, &data_type).unwrap()
            }
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

        let compact = deep_copy_array_sliced(slice.as_ref());
        let copied_bytes = deep_copy_array(compact.as_ref()).get_array_memory_size() as u64;
        let mut queue = AccumulationQueue::new(copied_bytes, 0, false);
        assert!(queue.insert(compact.clone(), 0, 32).is_none());
        assert_eq!(queue.current_bytes, copied_bytes);
        let (arrays, _, _) = queue.flush().unwrap();
        assert_eq!(arrays[0].as_ref(), compact.as_ref());
        assert_ne!(
            arrays[0].to_data().buffers()[0].as_ptr(),
            compact.to_data().buffers()[0].as_ptr()
        );
    }

    #[test]
    fn test_preflush_keeps_detached_copies_out_of_borrowed_page() {
        let parent: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..4096));
        let slice = parent.slice(17, 32);
        let copied_bytes = deep_copy_array_sliced(slice.as_ref()).get_array_memory_size() as u64;
        let mut queue = AccumulationQueue::new(copied_bytes * 3 / 2, 0, false);

        assert!(queue.insert_with_pages(slice.clone(), 17, 32).is_empty());
        let pages = queue.insert_with_pages(slice.clone(), 49, 32);
        assert_eq!(pages.len(), 1);
        assert!(!pages[0].includes_inserted_array);
        assert_ne!(
            pages[0].arrays[0].to_data().buffers()[0].data_ptr(),
            slice.to_data().buffers()[0].data_ptr()
        );

        let (arrays, row_number, num_rows) = queue.flush().unwrap();
        assert_eq!((row_number, num_rows), (49, 32));
        assert_eq!(arrays[0].as_ref(), slice.as_ref());
        assert_ne!(
            arrays[0].to_data().buffers()[0].data_ptr(),
            slice.to_data().buffers()[0].data_ptr()
        );
    }

    #[test]
    fn test_preflush_can_return_previous_and_current_pages() {
        let parent: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..4096));
        let slice = parent.slice(0, 32);
        let copied_bytes = deep_copy_array_sliced(slice.as_ref()).get_array_memory_size() as u64;
        let mut queue = AccumulationQueue::new(copied_bytes, 0, false);

        assert!(queue.insert_with_pages(slice.clone(), 0, 32).is_empty());
        let pages = queue.insert_with_pages(parent.clone(), 32, 4096);
        assert_eq!(pages.len(), 2);
        assert!(!pages[0].includes_inserted_array);
        assert!(pages[1].includes_inserted_array);
        assert_ne!(
            pages[0].arrays[0].to_data().buffers()[0].data_ptr(),
            slice.to_data().buffers()[0].data_ptr()
        );
        assert!(Arc::ptr_eq(&pages[1].arrays[0], &parent));
    }

    #[test]
    fn test_page_boundary_flushes_borrowed_input_without_copy() {
        let parent: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..2_097_152));
        let slice = parent.slice(17, 32_768);
        let mut queue = AccumulationQueue::new(8 * 1024 * 1024, 0, false);

        assert!(
            queue
                .insert_with_pages(slice.clone(), 17, 32_768)
                .is_empty()
        );
        let pages = queue.insert_for_page_boundary(parent.clone(), 32_785, 2_097_152);
        assert_eq!(pages.len(), 2);
        assert!(!pages[0].includes_inserted_array);
        assert!(pages[1].includes_inserted_array);
        assert!(Arc::ptr_eq(&pages[1].arrays[0], &parent));
        assert!(queue.flush().is_none());
    }

    #[test]
    fn test_page_boundary_coalesces_small_tail() {
        let parent: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..128));
        let slice = parent.slice(0, 96);
        let last = parent.slice(96, 32);
        let mut queue = AccumulationQueue::new(4096, 0, false);

        assert!(queue.insert_with_pages(slice.clone(), 0, 96).is_empty());
        let pages = queue.insert_for_page_boundary(last, 96, 32);
        assert_eq!(pages.len(), 1);
        assert!(pages[0].includes_inserted_array);
        assert_eq!(pages[0].num_rows, 128);
    }

    #[rstest]
    #[case::keep_original(true, 4096, 32, 512)]
    #[case::large_batch(false, 4096, 4096, 512)]
    #[case::array_overhead(false, 4096, 32, 256)]
    #[case::buffer_alignment(false, 1, 1, 128)]
    fn test_flush_without_copy(
        #[case] keep_original: bool,
        #[case] parent_rows: usize,
        #[case] rows: usize,
        #[case] budget: u64,
    ) {
        let parent: ArrayRef = Arc::new(UInt64Array::from(vec![1; parent_rows]));
        let array = parent.slice(0, rows);
        let mut queue = AccumulationQueue::new(budget, 0, keep_original);
        let (arrays, first_row, num_rows) = queue.insert(array.clone(), 0, rows as u64).unwrap();
        assert_eq!((first_row, num_rows), (0, rows as u64));
        assert!(Arc::ptr_eq(&arrays[0], &array));
    }
}
