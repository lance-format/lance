// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in inline bitpacking with admission before alignment/output/scratch allocations.

use std::ops::Range;

use arrow_buffer::{ArrowNativeType, Buffer};
use bytemuck::Pod;
use lance_bitpacking::BitPackingUninit;
use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

use super::{ELEMS_PER_CHUNK, InlineBitpacking};
use crate::{
    buffer::LanceBuffer,
    compression::MiniBlockDecodeBudget,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
};

fn checked_input<'a>(
    data: &'a [LanceBuffer],
    num_values: u64,
    range: &Range<u64>,
) -> Result<&'a LanceBuffer> {
    let [data] = data else {
        return Err(Error::invalid_input(format!(
            "Inline bitpacking expects one buffer, got {}",
            data.len()
        )));
    };
    if range.start > range.end || range.end > num_values {
        return Err(Error::corrupt_file_named(
            "inline_bitpacking",
            format!("selection {range:?} exceeds {num_values} decoded values"),
        ));
    }
    Ok(data)
}

pub(super) fn decode(
    bits: u64,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    let data = checked_input(data, num_values, &range)?;
    match bits {
        8 => decode_typed::<u8>(data, num_values, range, workspace),
        16 => decode_typed::<u16>(data, num_values, range, workspace),
        32 => decode_typed::<u32>(data, num_values, range, workspace),
        64 => decode_typed::<u64>(data, num_values, range, workspace),
        _ => Err(Error::invalid_input(format!(
            "unsupported bitpacking width {bits}"
        ))),
    }
}

pub(super) fn plan(
    bits: u64,
    data: &[LanceBuffer],
    num_values: u64,
    range: Range<u64>,
) -> Result<MiniBlockDecodeBudget> {
    let data = checked_input(data, num_values, &range)?;
    let plan = match bits {
        8 => plan_typed::<u8>(data, num_values, &range),
        16 => plan_typed::<u16>(data, num_values, &range),
        32 => plan_typed::<u32>(data, num_values, &range),
        64 => plan_typed::<u64>(data, num_values, &range),
        _ => {
            return Err(Error::invalid_input(format!(
                "unsupported bitpacking width {bits}"
            )));
        }
    }?;
    Ok(plan.budget)
}

fn allocate(workspace: &mut Reservation, len: usize) -> Result<ReservedBuffer> {
    workspace
        .try_allocate(len)?
        .ok_or_else(|| Error::internal("bitpacking workspace changed after complete admission"))
}

struct RangePlan {
    budget: MiniBlockDecodeBudget,
    width: usize,
    scratch_bytes: usize,
    alignment_bytes: usize,
}

fn plan_typed<T: ArrowNativeType + BitPackingUninit + Pod>(
    data: &LanceBuffer,
    num_values: u64,
    range: &Range<u64>,
) -> Result<RangePlan> {
    let width = if num_values == 0 {
        0
    } else {
        InlineBitpacking::validate_chunk::<T>(data, num_values)?
    };
    let selected = (range.end - range.start) as usize;
    let word_size = std::mem::size_of::<T>();
    let output_bytes = selected * word_size;
    // Preserve range decoding: sparse selections unpack individual values; full
    // chunks unpack directly into output. Only dense partial ranges need scratch.
    let scratch_bytes = if selected as u64 > InlineBitpacking::SPARSE_UNPACK_THRESHOLD
        && selected as u64 != ELEMS_PER_CHUNK
    {
        ELEMS_PER_CHUNK as usize * word_size
    } else {
        0
    };
    let alignment_bytes = if selected > 0 && data.as_ptr().align_offset(word_size) != 0 {
        data.len()
    } else {
        0
    };
    let total = output_bytes
        .checked_add(scratch_bytes)
        .and_then(|bytes| bytes.checked_add(alignment_bytes))
        .ok_or_else(|| Error::invalid_input("bitpacking workspace size overflow"))?;
    Ok(RangePlan {
        budget: MiniBlockDecodeBudget {
            output_bytes,
            workspace_bytes: total,
        },
        width,
        scratch_bytes,
        alignment_bytes,
    })
}

fn decode_typed<T: ArrowNativeType + BitPackingUninit + Pod>(
    data: &LanceBuffer,
    num_values: u64,
    range: Range<u64>,
    workspace: &mut Reservation,
) -> Result<Option<DataBlock>> {
    let RangePlan {
        budget,
        width,
        scratch_bytes,
        alignment_bytes,
    } = plan_typed::<T>(data, num_values, &range)?;
    // The exclusive workspace was already admitted to the shared budget. Check
    // every simultaneous allocation before splitting any credit or copying input.
    if budget.workspace_bytes > workspace.len() {
        return Ok(None);
    }
    let selected = (range.end - range.start) as usize;
    let word_size = std::mem::size_of::<T>();
    let mut output = allocate(workspace, budget.output_bytes)?;
    let aligned = if alignment_bytes > 0 {
        let mut aligned = allocate(workspace, alignment_bytes)?;
        aligned.as_mut().copy_from_slice(data);
        Some(aligned)
    } else {
        None
    };
    if selected > 0 {
        let bytes = aligned
            .as_ref()
            .map_or(data.as_ref(), |buffer| buffer.as_ref());
        let words: &[T] = bytemuck::try_cast_slice(bytes)
            .map_err(|error| Error::internal(format!("admitted input layout: {error}")))?;
        let destination: &mut [T] = bytemuck::try_cast_slice_mut(output.as_mut())
            .map_err(|error| Error::internal(format!("admitted output layout: {error}")))?;
        if selected as u64 == ELEMS_PER_CHUNK {
            // SAFETY: validate_chunk checked width and packed length, and the
            // range check proves a full 1024-value destination here.
            unsafe { T::unchecked_unpack(width, &words[1..], destination) };
        } else if scratch_bytes > 0 {
            let mut scratch = allocate(workspace, scratch_bytes)?;
            let scratch_values: &mut [T] = bytemuck::try_cast_slice_mut(scratch.as_mut())
                .map_err(|error| Error::internal(format!("admitted scratch layout: {error}")))?;
            // SAFETY: validated width/packed length and exactly 1024 typed slots.
            unsafe { T::unchecked_unpack(width, &words[1..], scratch_values) };
            destination.copy_from_slice(&scratch_values[range.start as usize..range.end as usize]);
            scratch.recycle(workspace)?;
        } else {
            for (destination, index) in destination.iter_mut().zip(range.start..range.end) {
                // SAFETY: validated width/packed length, index < num_values <= 1024.
                *destination =
                    unsafe { T::unchecked_unpack_single(width, &words[1..], index as usize) };
            }
        }
    }
    if let Some(aligned) = aligned {
        aligned.recycle(workspace)?;
    }
    Ok(Some(DataBlock::FixedWidth(FixedWidthDataBlock {
        data: LanceBuffer::from(Buffer::from(output.freeze())),
        bits_per_value: (word_size * 8) as u64,
        num_values: selected as u64,
        block_info: BlockInfo::new(),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::DataType;
    use lance_bitpacking::BitPacking;
    use lance_core::utils::allocation::AllocationBudget;
    use rstest::rstest;

    use crate::compression::MiniBlockDecompressor;

    fn check_admitted_range<T: ArrowNativeType + BitPackingUninit + Pod>(
        stored: u64,
        range: Range<u64>,
        packed_width: usize,
        unaligned: bool,
    ) {
        let word_size = std::mem::size_of::<T>();
        let packed_width = if packed_width == usize::MAX {
            word_size * 8
        } else {
            packed_width
        };
        let mask = if packed_width == usize::BITS as usize {
            usize::MAX
        } else {
            (1_usize << packed_width) - 1
        };
        let expected: Vec<T> = (0..ELEMS_PER_CHUNK as usize)
            .map(|i| T::from_usize((i * 31 + 7) & mask).unwrap())
            .collect();
        let encoded = if stored == 0 {
            Vec::new()
        } else {
            let mut packed =
                vec![T::from_usize(packed_width).unwrap(); 1 + 128 * packed_width / word_size];
            // SAFETY: exactly 1024 values, a width <= each tested T, and an exact packed destination.
            unsafe { BitPacking::unchecked_pack(packed_width, &expected, &mut packed[1..]) };
            bytemuck::cast_slice(&packed).to_vec()
        };
        let offset = usize::from(unaligned);
        let input_bytes = encoded.len() + offset;
        let output_bytes = (range.end - range.start) as usize * word_size;
        // Independently price the three physical layouts used by the selected
        // algorithm: selected output, full-chunk dense scratch, unaligned input copy.
        let selected = range.end - range.start;
        let dense_partial = selected > 32 && selected < 1024;
        let scratch_bytes = if dense_partial { 1024 * word_size } else { 0 };
        let alignment_bytes = if !range.is_empty() && unaligned && word_size > 1 {
            encoded.len()
        } else {
            0
        };
        let operation_bytes = output_bytes + scratch_bytes + alignment_bytes;
        let budget = AllocationBudget::new(input_bytes + operation_bytes);
        let mut workspace = budget.try_reserve(input_bytes + operation_bytes).unwrap();
        let mut input = workspace.try_allocate(input_bytes).unwrap().unwrap();
        input.as_mut()[offset..].copy_from_slice(&encoded);
        drop(encoded);
        let source = input.freeze();
        let input = source.slice(offset..);
        let buffers = [LanceBuffer::try_from_bytes_in(&input, 1, &mut workspace)
            .unwrap()
            .unwrap()];
        drop(input);
        let codec = InlineBitpacking::new((word_size * 8) as u64);
        let planned = codec.plan_range(&buffers, stored, range.clone()).unwrap();
        assert_eq!(planned.output_bytes, output_bytes);
        assert_eq!(planned.workspace_bytes, operation_bytes);
        assert_eq!(workspace.len(), operation_bytes);
        assert_eq!(budget.reserved(), input_bytes + operation_bytes);
        if operation_bytes > 0 {
            let mut short = workspace.try_split(operation_bytes - 1).unwrap();
            assert!(
                codec
                    .try_decompress_range_in(&buffers, stored, range.clone(), &mut short)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(short.len(), operation_bytes - 1);
            assert_eq!(budget.reserved(), input_bytes + operation_bytes);
            drop(short);
            drop(workspace);
            workspace = budget.try_reserve(operation_bytes).unwrap();
        }
        let output = codec
            .try_decompress_range_in(&buffers, stored, range.clone(), &mut workspace)
            .unwrap()
            .unwrap();
        assert_eq!(workspace.len(), scratch_bytes + alignment_bytes);
        let block = output.as_fixed_width_ref().unwrap();
        assert_eq!(block.num_values, range.end - range.start);
        assert_eq!(
            block.data.borrow_to_typed_slice::<T>().as_ref(),
            &expected[range.start as usize..range.end as usize]
        );
        let pointer = block.data.as_ptr();
        let datatype = match word_size {
            1 => DataType::UInt8,
            2 => DataType::UInt16,
            4 => DataType::UInt32,
            8 => DataType::UInt64,
            _ => unreachable!(),
        };
        let array = output.into_arrow(datatype, true).unwrap();
        assert_eq!(array.buffers()[0].as_ptr(), pointer);
        let last = array.slice(0, usize::from(!range.is_empty()));
        drop(array);
        drop(workspace);
        assert_eq!(budget.reserved(), input_bytes + output_bytes);
        drop(buffers);
        drop(source);
        assert_eq!(budget.reserved(), output_bytes);
        drop(last);
        assert_eq!(budget.reserved(), 0);
    }

    #[rstest]
    #[case::empty_chunk(0, 0..0, 0)]
    #[case::empty_selection(1024, 7..7, 3)]
    #[case::sparse_tail(1023, 991..1023, 3)]
    #[case::dense_tail(1023, 990..1023, 3)]
    #[case::full_chunk(1024, 0..1024, 3)]
    #[case::zero_width_dense(1024, 5..70, 0)]
    #[case::full_width_dense(1024, 5..70, usize::MAX)]
    fn admitted_range_preserves_values_and_complete_transient_credit(
        #[case] stored: u64,
        #[case] range: Range<u64>,
        #[case] packed_width: usize,
        #[values(8, 16, 32, 64)] output_width: usize,
        #[values(false, true)] unaligned: bool,
    ) {
        match output_width {
            8 => check_admitted_range::<u8>(stored, range, packed_width, unaligned),
            16 => check_admitted_range::<u16>(stored, range, packed_width, unaligned),
            32 => check_admitted_range::<u32>(stored, range, packed_width, unaligned),
            64 => check_admitted_range::<u64>(stored, range, packed_width, unaligned),
            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case::short_header(vec![0_u8; 3], 1, 0..1, "header")]
    #[case::bad_width(33_u32.to_ne_bytes().to_vec(), 1, 0..1, "width")]
    #[case::bad_payload(1_u32.to_ne_bytes().to_vec(), 1, 0..1, "payload")]
    #[case::too_many_values(0_u32.to_ne_bytes().to_vec(), 1025, 0..1, "at most")]
    #[case::bad_selection(0_u32.to_ne_bytes().to_vec(), 1, 0..2, "selection")]
    fn corrupt_chunks_do_not_consume_credit(
        #[case] input: Vec<u8>,
        #[case] stored: u64,
        #[case] range: Range<u64>,
        #[case] message: &str,
    ) {
        let budget = AllocationBudget::new(8192);
        let mut workspace = budget.try_reserve(8192).unwrap();
        let codec = InlineBitpacking::new(32);
        let buffers = [LanceBuffer::from(input)];
        let error = codec
            .plan_range(&buffers, stored, range.clone())
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(workspace.len(), 8192);
        let error = codec
            .try_decompress_range_in(&buffers, stored, range, &mut workspace)
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(workspace.len(), 8192);
        assert_eq!(budget.reserved(), 8192);
    }

    #[test]
    fn unsupported_codec_never_falls_back_to_unaccounted_decode() {
        #[derive(Debug)]
        struct Unsupported;
        impl MiniBlockDecompressor for Unsupported {
            fn decompress(&self, _: Vec<LanceBuffer>, _: u64) -> Result<DataBlock> {
                panic!("unadmitted fallback must not run")
            }
        }
        let codec = Unsupported;
        let budget = AllocationBudget::new(1);
        let mut workspace = budget.try_reserve(1).unwrap();
        let error = codec
            .plan_range(&[LanceBuffer::from(vec![7_u8])], 1, 0..1)
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("no admitted range plan"));
        let error = codec
            .try_decompress_range_in(&[LanceBuffer::from(vec![7_u8])], 1, 0..1, &mut workspace)
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("no admitted range decoder"));
        assert_eq!(workspace.len(), 1);
    }

    #[rstest]
    #[case::concat_peak(2..4, 8..10)]
    #[case::later_scratch_peak(2..4, 8..72)]
    #[case::earlier_scratch_peak(1..80, 8..10)]
    #[case::full_ranges(0..1024, 0..1024)]
    fn planned_ranges_cover_decode_and_concat_overlap(
        #[case] first: Range<u64>,
        #[case] second: Range<u64>,
        #[values(false, true)] is_unaligned: bool,
    ) {
        let values = [0_u32, 1].map(|seed| {
            (0..1024)
                .map(|index| (index * 3 + seed) % 8)
                .collect::<Vec<_>>()
        });
        let encoded = values.each_ref().map(|values| {
            let mut encoded = vec![3_u32; 97];
            // SAFETY: 1024 values fit in three bits; 96 words hold the packed payload.
            unsafe { u32::unchecked_pack(3, values, &mut encoded[1..]) };
            bytemuck::cast_slice(&encoded).to_vec()
        });
        let ranges = [first, second];
        let output_bytes = ranges
            .each_ref()
            .map(|range| (range.end - range.start) as usize * 4);
        // Independent physical layouts: 1024 u32 scratch slots for a dense
        // partial range, plus a 388-byte copy for either unaligned input.
        let transient = ranges.each_ref().map(|range| {
            let selected = range.end - range.start;
            usize::from(selected > 32 && selected < 1024) * 4096 + usize::from(is_unaligned) * 388
        });
        let retained = output_bytes.iter().sum::<usize>();
        let expected_peak = (output_bytes[0] + transient[0])
            .max(retained + transient[1])
            .max(retained * 2);
        let prefix = usize::from(is_unaligned);
        let input_bytes = 2 * (388 + prefix);
        let budget = AllocationBudget::new(input_bytes + expected_peak);
        let mut input = budget.try_reserve(input_bytes).unwrap().allocate().unwrap();
        for (index, bytes) in encoded.iter().enumerate() {
            let start = index * (388 + prefix) + prefix;
            input.as_mut()[start..start + 388].copy_from_slice(bytes);
        }
        let owner = input.freeze();
        let inputs = [0, 1].map(|index| {
            let start = index * (388 + prefix) + prefix;
            [LanceBuffer::from(Buffer::from(
                owner.slice(start..start + 388),
            ))]
        });
        drop(owner);
        let codec = InlineBitpacking::new(32);
        let plans = inputs
            .iter()
            .zip(&ranges)
            .map(|(input, range)| codec.plan_range(input, 1024, range.clone()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(budget.reserved(), input_bytes);
        let required = plans[0]
            .workspace_bytes
            .max(plans[0].output_bytes + plans[1].workspace_bytes)
            .max((plans[0].output_bytes + plans[1].output_bytes) * 2);
        assert_eq!(required, expected_peak);
        assert!(budget.try_reserve(required + 1).is_none());
        let mut workspace = budget.try_reserve(required).unwrap();
        let mut decoded = Vec::with_capacity(2);
        for (input, range) in inputs.iter().zip(&ranges) {
            decoded.push(
                codec
                    .try_decompress_range_in(input, 1024, range.clone(), &mut workspace)
                    .unwrap()
                    .unwrap(),
            );
        }
        assert_eq!(workspace.len(), required - retained);
        let output = DataBlock::try_concat_in(&decoded, &mut workspace)
            .unwrap()
            .unwrap();
        assert_eq!(workspace.len(), required - 2 * retained);
        let expected = values
            .iter()
            .zip(&ranges)
            .flat_map(|(values, range)| {
                values[range.start as usize..range.end as usize]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        let array = output
            .try_to_arrow_in(&DataType::UInt32, &mut workspace)
            .unwrap()
            .unwrap();
        assert_eq!(array.buffers()[0].typed_data::<u32>(), expected);
        drop(output);
        drop(decoded);
        drop(inputs);
        drop(workspace);
        assert_eq!(budget.reserved(), retained);
        let last = array.slice(array.len() - 1, 1);
        drop(array);
        assert_eq!(budget.reserved(), retained);
        drop(last);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn finite_decoders_progress_while_a_consumer_holds_output() {
        let output_bytes = 64 * 4;
        let operation_bytes = output_bytes + 1024 * 4;
        let budget = AllocationBudget::new(4 + output_bytes + 2 * operation_bytes);
        let input = budget.try_reserve(4).unwrap().allocate().unwrap().freeze();
        let buffers = [LanceBuffer::from(Buffer::from(input))];
        let codec = InlineBitpacking::new(32);
        let mut slow_workspace = budget.try_reserve(operation_bytes).unwrap();
        let slow_output = codec
            .try_decompress_range_in(&buffers, 1024, 1..65, &mut slow_workspace)
            .unwrap()
            .unwrap();
        drop(slow_workspace);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    for _ in 0..3 {
                        let mut workspace = budget.try_reserve(operation_bytes).unwrap();
                        let output = codec
                            .try_decompress_range_in(&buffers, 1024, 1..65, &mut workspace)
                            .unwrap()
                            .unwrap();
                        assert_eq!(output.num_values(), 64);
                        assert!(
                            output
                                .as_fixed_width_ref()
                                .unwrap()
                                .data
                                .iter()
                                .all(|byte| *byte == 0)
                        );
                        assert_eq!(workspace.len(), 1024 * 4);
                    }
                });
            }
        });
        assert_eq!(budget.reserved(), 4 + output_bytes);
        drop(buffers);
        assert_eq!(budget.reserved(), output_bytes);
        drop(slow_output);
        assert_eq!(budget.reserved(), 0);
    }
}
