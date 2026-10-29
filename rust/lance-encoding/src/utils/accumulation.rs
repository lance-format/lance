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

fn flat_buffer_ptr(array: &dyn Array) -> Option<usize> {
    array
        .to_data()
        .buffers()
        .first()
        .map(|buffer| buffer.data_ptr().as_ptr() as usize)
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
    retain_shared_slices: bool,
    buffered_arrays: Vec<ArrayRef>,
    current_bytes: u64,
    // The source allocation shared by every retained slice in the current page.
    shared_buffer: Option<usize>,
    // A page completed while another page was already being returned.
    flush_pending: bool,
    // Row number of the first item in buffered_arrays, reset on flush
    row_number: u64,
    // Number of top level rows represented in buffered_arrays, reset on flush
    num_rows: u64,
    // This is only for logging / debugging purposes
    column_index: u32,
}

impl AccumulationQueue {
    pub fn new(cache_bytes: u64, column_index: u32, keep_original_array: bool) -> Self {
        Self::new_with_shared_slice_retention(cache_bytes, column_index, keep_original_array, false)
    }

    pub(crate) fn new_with_shared_slice_retention(
        cache_bytes: u64,
        column_index: u32,
        keep_original_array: bool,
        retain_shared_slices: bool,
    ) -> Self {
        Self {
            cache_bytes,
            retain_shared_slices,
            buffered_arrays: Vec::new(),
            current_bytes: 0,
            shared_buffer: None,
            flush_pending: false,
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
        array: ArrayRef,
        row_number: u64,
        num_rows: u64,
    ) -> Option<(Vec<ArrayRef>, u64, u64)> {
        self.insert_with_page_info(array, row_number, num_rows)
            .map(|page| (page.arrays, page.row_number, page.num_rows))
    }

    pub(crate) fn insert_with_page_info(
        &mut self,
        mut array: ArrayRef,
        row_number: u64,
        num_rows: u64,
    ) -> Option<AccumulatedPage> {
        let mut completed_page = if self.flush_pending {
            self.take_page(false)
        } else {
            None
        };
        let mut is_full = false;
        let mut retained_bytes = None;
        if !self.keep_original_array {
            match flat_copy_sizes(array.as_ref()) {
                Some((copied_bytes, compacted_bytes)) => {
                    let original_bytes = array.get_array_memory_size();
                    let shared_buffer =
                        if self.retain_shared_slices && original_bytes > compacted_bytes {
                            flat_buffer_ptr(array.as_ref())
                        } else {
                            None
                        };
                    if let Some(shared_buffer) = shared_buffer {
                        // A page pins one source allocation.  Retaining slices
                        // from two parents would keep both buffers alive even
                        // though their visible bytes fit in the cache budget.
                        if !self.buffered_arrays.is_empty()
                            && self.shared_buffer != Some(shared_buffer)
                        {
                            debug_assert!(completed_page.is_none());
                            completed_page = self.take_page(false);
                        }
                        if self.buffered_arrays.is_empty() {
                            self.shared_buffer = Some(shared_buffer);
                        }
                        // The source batch stays alive while its slices stream.
                        // A private copy would temporarily add a second page-sized
                        // allocation, so retain the slice and charge visible bytes.
                        retained_bytes = Some(compacted_bytes as u64);
                    } else if self.shared_buffer.is_some() {
                        debug_assert!(completed_page.is_none());
                        completed_page = self.take_page(false);
                    }

                    let remaining_bytes = self.remaining_bytes();
                    if retained_bytes.is_none()
                        && copied_bytes <= compacted_bytes
                        && copied_bytes as u64 <= remaining_bytes
                    {
                        array = deep_copy_array(array.as_ref());
                    } else if retained_bytes.is_none() && compacted_bytes as u64 <= remaining_bytes
                    {
                        array = deep_copy_array_sliced(array.as_ref());
                    } else if retained_bytes.is_none() {
                        // Copy alignment can exceed the budget even when the original fits.
                        is_full = true;
                    }
                }
                // Nested and view arrays retain their existing copy policy.
                None => {
                    if self.shared_buffer.is_some() {
                        debug_assert!(completed_page.is_none());
                        completed_page = self.take_page(false);
                    }
                    if array.get_array_memory_size() as u64 <= self.remaining_bytes() {
                        array = deep_copy_array(array.as_ref());
                    }
                }
            }
        }
        if self.row_number == u64::MAX {
            self.row_number = row_number;
        }
        self.num_rows += num_rows;
        self.current_bytes +=
            retained_bytes.unwrap_or_else(|| array.get_array_memory_size() as u64);
        self.buffered_arrays.push(array);
        if is_full || self.current_bytes > self.cache_bytes {
            if completed_page.is_none() {
                debug!(
                    "Flushing column {} page of size {} bytes (unencoded)",
                    self.column_index, self.current_bytes
                );
                completed_page = self.take_page(true);
            } else {
                // The caller returns one page per inserted array.  Hold this
                // completed page until the next insert or final flush.
                self.flush_pending = true;
            }
        } else {
            trace!(
                "Accumulating data for column {}.  Now at {} bytes",
                self.column_index, self.current_bytes
            );
        }
        completed_page
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
            self.take_page(false)
                .map(|page| (page.arrays, page.row_number, page.num_rows))
        }
    }

    fn remaining_bytes(&self) -> u64 {
        self.cache_bytes.saturating_sub(self.current_bytes)
    }

    fn take_page(&mut self, includes_inserted_array: bool) -> Option<AccumulatedPage> {
        if self.buffered_arrays.is_empty() {
            self.current_bytes = 0;
            self.shared_buffer = None;
            self.flush_pending = false;
            return None;
        }
        self.current_bytes = 0;
        self.shared_buffer = None;
        self.flush_pending = false;
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
        let mut queue = AccumulationQueue::new_with_shared_slice_retention(budget, 0, false, true);

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
    fn test_shared_slice_pages_do_not_mix_source_buffers() {
        let parent_a: ArrayRef = Arc::new(UInt64Array::from(vec![1; 4096]));
        let parent_b: ArrayRef = Arc::new(UInt64Array::from(vec![2; 4096]));
        let slice_a = parent_a.slice(17, 32);
        let slice_b = parent_b.slice(17, 32);
        let copied_bytes = deep_copy_array_sliced(slice_a.as_ref()).get_array_memory_size() as u64;
        let mut queue =
            AccumulationQueue::new_with_shared_slice_retention(copied_bytes * 2, 0, false, true);

        assert!(
            queue
                .insert_with_page_info(slice_a.clone(), 17, 32)
                .is_none()
        );
        let page = queue
            .insert_with_page_info(slice_b.clone(), 49, 32)
            .unwrap();
        assert!(!page.includes_inserted_array);
        assert_eq!((page.row_number, page.num_rows), (17, 32));
        assert_eq!(
            page.arrays[0].to_data().buffers()[0].data_ptr(),
            slice_a.to_data().buffers()[0].data_ptr()
        );

        let (arrays, row_number, num_rows) = queue.flush().unwrap();
        assert_eq!((row_number, num_rows), (49, 32));
        assert_eq!(arrays.len(), 1);
        assert_eq!(arrays[0].as_ref(), slice_b.as_ref());
        assert_eq!(
            arrays[0].to_data().buffers()[0].data_ptr(),
            slice_b.to_data().buffers()[0].data_ptr()
        );
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
