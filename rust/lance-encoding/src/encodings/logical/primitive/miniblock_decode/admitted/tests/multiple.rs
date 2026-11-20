// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Array, FixedSizeListArray, UInt32Array};
use arrow_buffer::NullBuffer;
use arrow_schema::Field;

use super::*;
#[cfg(feature = "bitpacking")]
use crate::{
    compression::{BlockCompressor, DecompressionStrategy, DefaultDecompressionStrategy},
    data::{BlockInfo, FixedWidthDataBlock},
    format::EncodingUtils,
};

pub(super) fn select(task: &mut DecodeMiniBlockTask, selections: &[(u64, u64)]) {
    let (template, chunk) = task.instructions.remove(0);
    task.instructions = selections
        .iter()
        .map(|&(start, count)| {
            (
                ChunkDrainInstructions {
                    chunk_instructions: template.chunk_instructions.clone(),
                    rows_to_skip: start,
                    rows_to_take: count,
                    preamble_action: template.preamble_action,
                },
                chunk.clone(),
            )
        })
        .collect();
}

fn list_fixture(codec: Codec, large: bool, workspace: usize) -> Fixture {
    let mut fixture = fixture(
        Some(&[1, 0, 1, 1, 1, 0]),
        Some(&[0, 1, 2, 3, 0, 0]),
        codec,
        large,
        workspace,
    );
    select(&mut fixture.task, &[(0, 2), (2, 2)]);
    fixture
}

fn assert_lists(page: DecodedPage, work: &mut Reservation) {
    assert_values(&page, &[10, 0, 20, 30]);
    let mut structure = CompositeRepDefUnraveler::new(vec![page.repdef]);
    let leaf = structure
        .try_unravel_validity_in(4, work)
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(leaf.iter().collect::<Vec<_>>(), [true, false, true, true]);
    let (offsets, validity) = structure
        .try_unravel_offsets_in::<i32>(work)
        .unwrap()
        .unwrap();
    assert_eq!(offsets.as_ref(), &[0, 2, 2, 2, 4]);
    assert_eq!(
        validity.unwrap().iter().collect::<Vec<_>>(),
        [true, false, true, true]
    );
}

#[rstest]
#[case::flat(Codec::Flat)]
#[case::rle(Codec::Rle)]
fn multiple_selections_preserve_list_null_empty_order(
    #[case] codec: Codec,
    #[values(false, true)] large: bool,
) {
    let Fixture {
        task,
        budget,
        mut work,
        physical,
    } = list_fixture(codec, large, 128);
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    // 24 combined level bytes,16 intermediate value bytes and16 final values.
    assert_eq!(work.len(), 128 - 56);
    assert_eq!(budget.reserved(), physical + 128 - 16);
    assert_lists(page, &mut work);
    drop(work);
    drop(task);
    assert_eq!(budget.reserved(), 0);
}

#[cfg(feature = "bitpacking")]
#[rstest]
fn multiple_bitpacked_structural_streams(#[values(false, true)] large: bool) {
    let Fixture {
        task,
        budget,
        mut work,
        ..
    } = list_fixture(Codec::Bitpacked, large, 4096);
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_lists(page, &mut work);
    drop(work);
    drop(task);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::first_levels(23)]
#[case::first_copy(35)]
#[case::later_copy(47)]
#[case::combined_execution(55)]
fn multiple_refusal_recycles_all_prepared_selections(#[case] available: usize) {
    let Fixture {
        task,
        budget,
        mut work,
        physical,
    } = list_fixture(Codec::Flat, false, 56);
    let mut short = work.try_split(available).unwrap();
    for _ in 0..2 {
        assert!(task.try_decode_in(&mut short).unwrap().is_none());
        assert_eq!(short.len(), available);
        assert_eq!(budget.reserved(), physical + 56);
        assert!(budget.try_reserve(1).is_none());
    }
    drop(short);
    drop(work);
    let mut work = budget.try_reserve(56).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_values(&page, &[10, 0, 20, 30]);
    assert!(work.is_empty());
    drop(page);
    drop(task);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::item_count(0, "item count")]
#[case::selection(1, "repetition boundaries")]
#[case::truncated_last_chunk(2, "buffer")]
fn multiple_late_invalid_input_restores_the_same_workspace(
    #[case] fault: u8,
    #[case] message: &str,
) {
    let Fixture {
        mut task,
        budget,
        mut work,
        physical,
    } = list_fixture(Codec::Flat, false, 128);
    match fault {
        0 => task.instructions[1].1.items_in_chunk += 1,
        1 => task.instructions[1].0.rows_to_take = 10,
        2 => {
            let data = &task.instructions[1].1.data;
            task.instructions[1].1.data = data.slice_with_length(0, data.len() - 17);
        }
        _ => unreachable!(),
    }
    let error = task.try_decode_in(&mut work).err().unwrap();
    assert!(matches!(
        error,
        Error::InvalidInput { .. } | Error::CorruptFile { .. }
    ));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(work.len(), 128);
    assert_eq!(budget.reserved(), physical + 128);
}

#[test]
fn multiple_preamble_and_rows_keep_existing_selection_semantics() {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = fixture(
        Some(&[0, 0, 1, 1, 1, 0]),
        Some(&[0, 1, 2, 3, 0, 0]),
        Codec::Flat,
        false,
        128,
    );
    select(&mut task, &[(0, 0), (0, 3)]);
    task.instructions[0].0.preamble_action = PreambleAction::Take;
    task.instructions[1].0.preamble_action = PreambleAction::Skip;
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_values(&page, &[10, 0, 20, 30]);
    drop(page);
    drop(work);
    drop(task);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn empty_multiple_ranges_preserve_typed_empty_output() {
    let Fixture {
        mut task,
        budget,
        mut work,
        physical,
    } = fixture(None, None, Codec::Flat, false, 0);
    select(&mut task, &[(1, 0), (2, 0)]);
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_values(&page, &[]);
    let arrow = page
        .data
        .try_to_arrow_in(&DataType::UInt32, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(arrow.len(), 0);
    assert_eq!(budget.reserved(), physical);
    drop(arrow);
    drop(page);
    drop(work);
    drop(task);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn distinct_chunks_preserve_order_and_independent_input_owners(#[values(false, true)] large: bool) {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = fixture(None, None, Codec::Flat, large, 1024);
    let values = [100_u32, 200, 300, 400];
    let payload: &[u8] = bytemuck::cast_slice(&values);
    let mut frame = 4_u16.to_le_bytes().to_vec();
    if large {
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    } else {
        frame.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    }
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    frame.extend_from_slice(payload);
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    let physical = frame.len() + 17;
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..1 + frame.len()].copy_from_slice(&frame);
    let input = LanceBuffer::from(Buffer::from(input.freeze()).slice_with_length(1, frame.len()));
    select(&mut task, &[(2, 2), (1, 2)]);
    let start = task.instructions[0].1.byte_range.end;
    task.instructions[1].0.chunk_instructions.chunk_idx = 1;
    task.instructions[1].1 = LoadedChunk {
        byte_range: start..start + frame.len() as u64,
        data: input.clone(),
        items_in_chunk: 4,
        chunk_idx: 1,
    };
    let mut exact = work.try_split(32).unwrap();
    let page = task.try_decode_in(&mut exact).unwrap().unwrap();
    assert_values(&page, &[20, 30, 200, 300]);
    assert!(exact.is_empty());
    let output_alias = page.data.as_fixed_width_ref().unwrap().data.clone();
    drop(page);
    drop(exact);
    drop(work);
    drop(task);
    assert_eq!(budget.reserved(), physical + 16);
    drop(input);
    assert_eq!(budget.reserved(), 16);
    drop(output_alias);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn multiple_outputs_allow_finite_concurrent_progress() {
    let Fixture {
        mut task,
        budget,
        work,
        physical,
    } = fixture(None, None, Codec::Flat, false, 64);
    select(&mut task, &[(0, 2), (2, 2)]);
    drop(work);
    let consume = || {
        let mut work = budget.try_reserve(32).unwrap();
        task.try_decode_in(&mut work).unwrap().unwrap()
    };
    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(consume);
        let second = scope.spawn(consume);
        (first.join().unwrap(), second.join().unwrap())
    });
    assert_values(&first, &[10, 0, 20, 30]);
    assert_values(&second, &[10, 0, 20, 30]);
    assert_eq!(budget.reserved(), physical + 32);
    // A third workspace fits while both completed outputs remain live.
    let third = consume();
    let alias = first.data.as_fixed_width_ref().unwrap().data.clone();
    drop(first);
    assert!(budget.try_reserve(32).is_none());
    drop(alias);
    let fourth = consume();
    assert_values(&fourth, &[10, 0, 20, 30]);
    drop(second);
    drop(third);
    drop(fourth);
    drop(task);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn multiple_nested_vectors_include_child_bitmap_output(#[values(false, true)] large: bool) {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = fixture(None, None, Codec::Flat, large, 1024);
    let values: Vec<u32> = (0..12).collect();
    let buffers: [&[u8]; 2] = [&[0b1011_0110, 0b0000_1101], bytemuck::cast_slice(&values)];
    let mut frame = 4_u16.to_le_bytes().to_vec();
    for buffer in buffers {
        if large {
            frame.extend_from_slice(&(buffer.len() as u32).to_le_bytes());
        } else {
            frame.extend_from_slice(&(buffer.len() as u16).to_le_bytes());
        }
    }
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    for buffer in buffers {
        frame.extend_from_slice(buffer);
        frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    }
    let physical = frame.len() + 1;
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..].copy_from_slice(&frame);
    task.instructions[0].1.data =
        LanceBuffer::from(Buffer::from(input.freeze()).slice_with_length(1, frame.len()));
    task.instructions[0].1.byte_range = 0..frame.len() as u64;
    task.num_buffers = 2;
    let encoding =
        crate::format::EncodingUtils::fsl(3, true, crate::format::EncodingUtils::flat(32, None));
    let metadata::compressive_encoding::Compression::FixedSizeList(fsl) =
        encoding.compression.unwrap()
    else {
        unreachable!()
    };
    task.value_decompressor = Arc::new(ValueDecompressor::from_fsl(&fsl).unwrap());
    select(&mut task, &[(1, 1), (2, 2)]);
    let mut short = work.try_split(75).unwrap();
    assert!(task.try_decode_in(&mut short).unwrap().is_none());
    assert_eq!(short.len(), 75);
    drop(short);
    drop(work);
    let mut work = budget.try_reserve(76).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert!(work.is_empty());
    let child = UInt32Array::new(
        values[3..].to_vec().into(),
        Some(NullBuffer::from(
            (3..12)
                .map(|i| arrow_buffer::bit_util::get_bit(buffers[0], i))
                .collect::<Vec<_>>(),
        )),
    );
    let expected = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::UInt32, true)),
        3,
        Arc::new(child),
        None,
    );
    let actual = page
        .data
        .try_to_arrow_in(expected.data_type(), &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(actual, expected.to_data());
    drop(page);
    drop(actual);
    drop(work);
    drop(task);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn rle_values_compose_with_task_admission(
    #[values(false, true)] large: bool,
    #[values(false, true)] multiple: bool,
) {
    check_rle_values(large, multiple, None);
}

#[cfg(feature = "bitpacking")]
#[rstest]
fn bitpacked_rle_values_compose_with_task_admission(
    #[values(false, true)] large: bool,
    #[values(false, true)] multiple: bool,
    #[values(false, true)] packed_values: bool,
) {
    check_rle_values(large, multiple, Some(packed_values));
}

fn check_rle_values(large: bool, multiple: bool, packed: Option<bool>) {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = fixture(None, None, Codec::Flat, large, 1024);
    let (payloads, codec) = rle_payloads(packed);
    let buffers: [&[u8]; 2] = [&payloads[0], &payloads[1]];
    let mut frame = 4_u16.to_le_bytes().to_vec();
    for buffer in buffers {
        if large {
            frame.extend_from_slice(&(buffer.len() as u32).to_le_bytes());
        } else {
            frame.extend_from_slice(&(buffer.len() as u16).to_le_bytes());
        }
    }
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    for buffer in buffers {
        frame.extend_from_slice(buffer);
        frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    }
    let physical = frame.len() + 17;
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..1 + frame.len()].copy_from_slice(&frame);
    task.instructions[0].1.data =
        LanceBuffer::from(Buffer::from(input.freeze()).slice_with_length(1, frame.len()));
    task.instructions[0].1.byte_range = 0..frame.len() as u64;
    task.num_buffers = 2;
    task.value_decompressor = codec;
    let selections: &[(u64, u64)] = if multiple {
        &[(1, 2), (3, 1)]
    } else {
        &[(1, 2)]
    };
    select(&mut task, selections);
    let expected: &[u32] = if multiple { &[10, 20, 30] } else { &[10, 20] };
    let required = expected.len() * 4 * if multiple { 2 } else { 1 };
    let mut short = work.try_split(required - 1).unwrap();
    assert!(task.try_decode_in(&mut short).unwrap().is_none());
    assert_eq!(short.len(), required - 1);
    drop(short);
    drop(work);
    let mut work = budget.try_reserve(required).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_values(&page, expected);
    assert!(work.is_empty());
    assert_eq!(budget.reserved(), physical + expected.len() * 4);
    drop(task);
    assert_eq!(budget.reserved(), expected.len() * 4);
    drop(page);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

fn rle_payloads(packed: Option<bool>) -> ([Vec<u8>; 2], Arc<dyn MiniBlockDecompressor>) {
    let values = [10_u32, 20, 30];
    let payloads = [bytemuck::cast_slice(&values).to_vec(), vec![2, 1, 1]];
    let Some(packed_values) = packed else {
        return (payloads, Arc::new(RleDecompressor::new(32)));
    };
    #[cfg(feature = "bitpacking")]
    {
        let mut payloads = payloads;
        let index = usize::from(!packed_values);
        let bits = if packed_values { 32 } else { 8 };
        let (encoded, encoding) = BlockCompressor::compress(
            &OutOfLineBitpacking::new(if packed_values { 5 } else { 2 }, bits),
            DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::from(payloads[index].clone()),
                bits_per_value: bits,
                num_values: 3,
                block_info: BlockInfo::new(),
            }),
        )
        .unwrap();
        payloads[index] = encoded.unwrap().to_vec();
        let encoding = if packed_values {
            EncodingUtils::rle(encoding, EncodingUtils::flat(8, None))
        } else {
            EncodingUtils::rle(EncodingUtils::flat(32, None), encoding)
        };
        let strategy = DefaultDecompressionStrategy::default();
        let decoder = strategy
            .create_miniblock_decompressor(&encoding, &strategy)
            .unwrap();
        (payloads, Arc::from(decoder))
    }
    #[cfg(not(feature = "bitpacking"))]
    panic!("bitpacked RLE feature disabled (values={packed_values})")
}

fn general_multiple(scheme: CompressionScheme, large: bool, capacity: usize) -> Fixture {
    let mut fixture = fixture_general(
        Some(&[1, 0, 1, 1, 1, 0]),
        Some(&[0, 1, 2, 3, 0, 0]),
        Codec::Flat,
        large,
        capacity,
        Some(scheme),
    );
    select(&mut fixture.task, &[(0, 2), (2, 2)]);
    fixture
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn multiple_general_preparation_refusal_is_transactional(
    #[case] scheme: CompressionScheme,
    #[values(false, true)] large: bool,
    #[values(39, 51, 67, 79, 87)] capacity: usize,
) {
    // Full levels=24, two selected level copies=24, two raw buffers=32.
    // After preparation, 32 more bytes must fit for values plus concat output.
    let Fixture {
        task,
        budget,
        mut work,
        physical,
    } = general_multiple(scheme, large, capacity);
    for _ in 0..2 {
        assert!(task.try_decode_in(&mut work).unwrap().is_none());
        assert_eq!(work.len(), capacity);
        assert_eq!(budget.reserved(), physical + capacity);
        assert!(budget.try_reserve(1).is_none());
    }
    drop((task, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn multiple_general_preserves_nested_values_and_concat_overlap(
    #[case] scheme: CompressionScheme,
    #[values(false, true)] large: bool,
) {
    let Fixture {
        task,
        budget,
        mut work,
        physical,
    } = general_multiple(scheme, large, 88);
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    // Both preparations have recycled. Combined levels=24, intermediates=16,
    // final values=16 were funded; intermediate charges returned to the pool.
    assert_eq!(work.len(), 32);
    assert_eq!(budget.reserved(), physical + 72);
    assert_lists(page, &mut work);
    drop((task, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn multiple_general_late_inner_error_recycles_earlier_preparation(
    #[case] scheme: CompressionScheme,
    #[values(false, true)] large: bool,
) {
    let Fixture {
        mut task,
        budget,
        mut work,
        physical,
    } = fixture_general(None, None, Codec::Flat, large, 1024, Some(scheme));
    select(&mut task, &[(0, 2), (2, 2)]);
    // Valid outer frame, but only 15 decoded bytes for four u32 values. The
    // earlier instruction must be recycled after this late inner preflight error.
    let compressor =
        GeneralBufferCompressor::get_compressor(CompressionConfig::new(scheme, None)).unwrap();
    let mut payload = Vec::new();
    compressor.compress(&[0; 15], &mut payload).unwrap();
    let mut frame = 4_u16.to_le_bytes().to_vec();
    if large {
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    } else {
        frame.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    }
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    frame.extend_from_slice(&payload);
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    let mut input = work.try_allocate(frame.len()).unwrap().unwrap();
    input.as_mut().copy_from_slice(&frame);
    task.instructions[1].1.data = LanceBuffer::from(Buffer::from(input.freeze()));
    let available = work.len();
    for _ in 0..2 {
        let error = task.try_decode_in(&mut work).err().unwrap();
        assert!(matches!(
            error,
            Error::InvalidInput { .. } | Error::CorruptFile { .. }
        ));
        assert!(error.to_string().contains("buffer"), "{error}");
        assert_eq!(work.len(), available);
        assert_eq!(budget.reserved(), physical + 1024);
    }
    drop((task, work));
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn multiple_general_distinct_chunks_keep_physical_owners(
    #[case] scheme: CompressionScheme,
    #[values(false, true)] large: bool,
) {
    let Fixture {
        mut task,
        budget,
        mut work,
        physical: _,
    } = fixture_general(None, None, Codec::Flat, large, 1024, Some(scheme));
    let frame = task.instructions[0].1.data.as_ref();
    let second_physical = frame.len() + 37;
    let mut second = work.try_allocate(second_physical).unwrap().unwrap();
    second.as_mut()[1..1 + frame.len()].copy_from_slice(frame);
    let second = LanceBuffer::from(Buffer::from(second.freeze()).slice_with_length(1, frame.len()));
    select(&mut task, &[(2, 2), (0, 2)]);
    task.instructions[1].1.data = second.clone();
    task.instructions[1].1.chunk_idx = 1;
    task.instructions[1].0.chunk_instructions.chunk_idx = 1;
    let start = task.instructions[0].1.byte_range.end;
    task.instructions[1].1.byte_range = start..start + second.len() as u64;
    let mut exact = work.try_split(64).unwrap();
    let page = task.try_decode_in(&mut exact).unwrap().unwrap();
    assert_values(&page, &[20, 30, 10, 0]);
    assert_eq!(exact.len(), 32);
    let alias = page
        .data
        .as_fixed_width_ref()
        .unwrap()
        .data
        .slice_with_length(0, 1);
    drop((page, task, work, exact));
    // A one-byte result view still owns the full concat allocation.
    assert_eq!(budget.reserved(), second_physical + 16);
    drop(second);
    assert_eq!(budget.reserved(), 16);
    drop(alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn multiple_general_finite_consumers_progress_after_last_alias(#[case] scheme: CompressionScheme) {
    let Fixture {
        mut task,
        budget,
        work,
        physical,
    } = fixture_general(None, None, Codec::Flat, false, 128, Some(scheme));
    select(&mut task, &[(0, 2), (2, 2)]);
    drop(work);
    let consume = || {
        let mut work = budget.try_reserve(64).unwrap();
        task.try_decode_in(&mut work).unwrap().unwrap()
    };
    let (first, second) = std::thread::scope(|scope| {
        let a = scope.spawn(consume);
        let b = scope.spawn(consume);
        (a.join().unwrap(), b.join().unwrap())
    });
    let third = consume();
    let fourth = consume();
    let fifth = consume();
    for page in [&first, &second, &third, &fourth, &fifth] {
        assert_values(page, &[10, 0, 20, 30]);
    }
    let alias = first.data.as_fixed_width_ref().unwrap().data.clone();
    drop(first);
    assert_eq!(budget.reserved(), physical + 80);
    assert!(budget.try_reserve(64).is_none());
    drop(alias);
    let sixth = consume();
    assert_values(&sixth, &[10, 0, 20, 30]);
    drop((second, third, fourth, fifth, sixth, task));
    assert_eq!(budget.reserved(), 0);
}
