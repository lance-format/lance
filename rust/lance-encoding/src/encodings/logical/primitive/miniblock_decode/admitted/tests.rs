// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_buffer::Buffer;
use arrow_schema::DataType;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::{InlineBitpacking, OutOfLineBitpacking};
#[cfg(feature = "bitpacking")]
use lance_bitpacking::BitPacking;

use super::*;
use crate::{
    buffer::LanceBuffer,
    compression::{BlockDecompressor, MiniBlockDecompressor},
    data::DataBlock,
    decoder::DecodePageTask,
    encodings::{
        logical::primitive::{
            LoadedChunk, MINIBLOCK_ALIGNMENT,
            miniblock_decode::{ChunkDrainInstructions, ChunkInstructions},
        },
        physical::{
            rle::{RleDecompressor, RunLengthWidth},
            value::ValueDecompressor,
        },
    },
    format::metadata,
    repdef::{CompositeRepDefUnraveler, DefinitionInterpretation as Meaning},
};

#[derive(Clone, Copy, Debug)]
enum Codec {
    Flat,
    Rle,
    #[cfg(feature = "bitpacking")]
    Bitpacked,
}

fn levels(values: &[u16], kind: Codec) -> (Arc<dyn BlockDecompressor>, Vec<u8>) {
    match kind {
        Codec::Flat => (
            Arc::new(ValueDecompressor::from_flat(&metadata::Flat {
                bits_per_value: 16,
                data: None,
            })),
            bytemuck::cast_slice(values).to_vec(),
        ),
        Codec::Rle => {
            let mut bytes = ((values.len() * 2) as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(bytemuck::cast_slice(values));
            bytes.extend(std::iter::repeat_n(1, values.len()));
            (
                Arc::new(RleDecompressor::with_run_length_width(
                    16,
                    RunLengthWidth::U8,
                )),
                bytes,
            )
        }
        #[cfg(feature = "bitpacking")]
        Codec::Bitpacked => {
            let mut padded = [0_u16; 1024];
            padded[..values.len()].copy_from_slice(values);
            let mut packed = vec![0_u16; 128];
            // SAFETY: 1024 u16 values fit in two bits; packed output has 128 words.
            unsafe { u16::unchecked_pack(2, &padded, &mut packed) };
            (
                Arc::new(OutOfLineBitpacking::new(2, 16)),
                bytemuck::cast_slice(&packed).to_vec(),
            )
        }
    }
}

struct Fixture {
    task: DecodeMiniBlockTask,
    budget: Arc<AllocationBudget>,
    work: Reservation,
    physical: usize,
}

fn fixture(
    rep: Option<&[u16]>,
    def: Option<&[u16]>,
    kind: Codec,
    has_large_chunk: bool,
    workspace_bytes: usize,
) -> Fixture {
    let values = [10_u32, 0, 20, 30];
    let count = rep.or(def).map_or(4, <[u16]>::len);
    let rep = rep.map(|v| levels(v, kind));
    let def = def.map(|v| levels(v, kind));
    let payload: &[u8] = bytemuck::cast_slice(&values);
    let mut bytes = (count as u16).to_le_bytes().to_vec();
    for (_, data) in rep.iter().chain(def.iter()) {
        bytes.extend_from_slice(&(data.len() as u16).to_le_bytes());
    }
    if has_large_chunk {
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    } else {
        bytes.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    }
    bytes.resize(bytes.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    for data in rep
        .iter()
        .chain(def.iter())
        .map(|(_, b)| b.as_slice())
        .chain([payload])
    {
        bytes.extend_from_slice(data);
        bytes.resize(bytes.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    }
    // Sliced/coalesced backing stays charged, including unreferenced padding.
    let physical = bytes.len() + 38;
    let budget = AllocationBudget::new(physical + workspace_bytes);
    let mut work = budget.try_reserve(physical + workspace_bytes).unwrap();
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..1 + bytes.len()].copy_from_slice(&bytes);
    let input = LanceBuffer::from(Buffer::from(input.freeze()).slice_with_length(1, bytes.len()));
    let has_rep = rep.is_some();
    let meanings: Arc<[Meaning]> = if has_rep {
        Arc::from([Meaning::NullableItem, Meaning::NullableAndEmptyableList])
    } else if def.is_some() {
        Arc::from([Meaning::NullableItem])
    } else {
        Arc::from([Meaning::AllValidItem])
    };
    let task = DecodeMiniBlockTask {
        structural_projection: None,
        rep_decompressor: rep.map(|(c, _)| c),
        def_decompressor: def.map(|(c, _)| c),
        value_decompressor: Arc::new(ValueDecompressor::from_flat(&metadata::Flat {
            bits_per_value: 32,
            data: None,
        })),
        dictionary_data: None,
        max_visible_level: max_visible_level(&meanings),
        def_meaning: meanings,
        num_buffers: 1,
        instructions: vec![(
            ChunkDrainInstructions {
                chunk_instructions: ChunkInstructions {
                    chunk_idx: 0,
                    preamble: PreambleAction::Absent,
                    rows_to_skip: 0,
                    rows_to_take: 4,
                    take_trailer: false,
                },
                rows_to_skip: 1,
                rows_to_take: 3,
                preamble_action: PreambleAction::Absent,
            },
            LoadedChunk {
                byte_range: 0..bytes.len() as u64,
                data: input,
                items_in_chunk: 4,
                chunk_idx: 0,
            },
        )],
        has_large_chunk,
    };
    Fixture {
        task,
        budget,
        work,
        physical,
    }
}

fn assert_values(page: &DecodedPage, expected: &[u32]) {
    let block = page.data.as_fixed_width_ref().unwrap();
    assert_eq!(block.num_values, expected.len() as u64);
    assert_eq!(bytemuck::cast_slice::<u8, u32>(&block.data), expected);
}

fn check_list_selection(codec: Codec, large: bool) {
    let Fixture {
        task,
        budget,
        mut work,
        physical,
    } = fixture(
        Some(&[1, 0, 1, 1, 1, 0]),
        Some(&[0, 1, 2, 3, 0, 0]),
        codec,
        large,
        4096,
    );
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_values(&page, &[20, 30]);
    assert_eq!(
        work.len(),
        4096 - 24,
        "full levels and codec scratch must be recycled"
    );
    let array = page
        .data
        .try_to_arrow_in(&DataType::UInt32, &mut work)
        .unwrap()
        .unwrap();
    let last_value = array.buffers()[0].slice_with_length(0, 4);
    let mut structure = CompositeRepDefUnraveler::new(vec![page.repdef]);
    let leaf = structure
        .try_unravel_validity_in(2, &mut work)
        .unwrap()
        .unwrap()
        .unwrap();
    let (offsets, validity) = structure
        .try_unravel_offsets_in::<i32>(&mut work)
        .unwrap()
        .unwrap();
    assert_eq!(leaf.iter().collect::<Vec<_>>(), [true, true]);
    assert_eq!(offsets.as_ref(), &[0, 0, 0, 2]);
    assert_eq!(
        validity.as_ref().unwrap().iter().collect::<Vec<_>>(),
        [false, true, true]
    );
    drop(work);
    assert_eq!(budget.reserved(), physical + 24 + 18);
    drop(task);
    drop(structure);
    drop(page.data);
    drop(array);
    drop(leaf);
    drop(offsets);
    drop(validity);
    assert_eq!(budget.reserved(), 8);
    drop(last_value);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::flat(Codec::Flat)]
#[case::rle(Codec::Rle)]
fn list_selection_recycles_scratch_and_preserves_aliases(
    #[case] codec: Codec,
    #[values(false, true)] large: bool,
) {
    check_list_selection(codec, large);
}

#[cfg(feature = "bitpacking")]
#[rstest]
fn bitpacked_list_selection_recycles_scratch(#[values(false, true)] large: bool) {
    check_list_selection(Codec::Bitpacked, large);
}

#[rstest]
#[case::before_levels(23)]
#[case::before_selected_copy(39)]
#[case::inside_value_decoder(47)]
fn every_refusal_preserves_task_and_exclusive_credit(#[case] available: usize) {
    let Fixture {
        task,
        budget,
        mut work,
        physical,
    } = fixture(
        Some(&[1, 0, 1, 1, 1, 0]),
        Some(&[0, 1, 2, 3, 0, 0]),
        Codec::Rle,
        false,
        48,
    );
    let mut short = work.try_split(available).unwrap();
    for _ in 0..2 {
        assert!(task.try_decode_in(&mut short).unwrap().is_none());
        assert_eq!(short.len(), available);
        assert_eq!(budget.reserved(), physical + 48);
    }
    drop(short);
    drop(work);
    let mut work = budget.try_reserve(48).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_values(&page, &[20, 30]);
    assert_eq!(work.len(), 24);
    drop(task);
    drop(work);
    assert_eq!(budget.reserved(), 24);
    drop(page);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::skip(PreambleAction::Skip, 1, vec![20, 30])]
#[case::take(PreambleAction::Take, 1, vec![10, 0, 20, 30])]
#[case::preamble_only(PreambleAction::Take, 0, vec![10, 0])]
fn list_preamble_keeps_existing_range_semantics(
    #[case] preamble: PreambleAction,
    #[case] rows: u64,
    #[case] expected: Vec<u32>,
) {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = fixture(
        Some(&[0, 0, 1, 0]),
        Some(&[0, 0, 0, 0]),
        Codec::Rle,
        false,
        128,
    );
    task.instructions[0].0.rows_to_skip = 0;
    task.instructions[0].0.rows_to_take = rows;
    task.instructions[0].0.preamble_action = preamble;
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_values(&page, &expected);
    drop(task);
    drop(page);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn flat_nullable_or_empty_selection(
    #[values(false, true)] nullable: bool,
    #[values(false, true)] empty: bool,
) {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = fixture(
        None,
        nullable.then_some(&[0, 1, 0, 0]),
        Codec::Rle,
        true,
        128,
    );
    if empty {
        task.instructions[0].0.rows_to_take = 0;
    }
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_values(&page, if empty { &[] } else { &[0, 20, 30] });
    drop(task);
    drop(page);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[test]
fn concurrent_consumers_make_progress_after_final_alias_release() {
    let Fixture {
        mut task,
        budget,
        work,
        physical,
    } = fixture(None, None, Codec::Flat, false, 16);
    task.instructions[0].0.rows_to_take = 2;
    drop(work);
    let consume = || {
        let mut workspace = budget.try_reserve(8).unwrap();
        task.try_decode_in(&mut workspace).unwrap().unwrap()
    };
    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(consume);
        let second = scope.spawn(consume);
        (first.join().unwrap(), second.join().unwrap())
    });
    assert_values(&first, &[0, 20]);
    assert_values(&second, &[0, 20]);
    assert_eq!(budget.reserved(), physical + 16);
    assert!(budget.try_reserve(8).is_none());
    let alias = first.data.as_fixed_width_ref().unwrap().data.clone();
    drop(first);
    assert!(budget.try_reserve(8).is_none());
    drop(alias);
    let third = consume();
    assert_values(&third, &[0, 20]);
    drop(third);
    drop(second);
    drop(task);
    assert_eq!(budget.reserved(), 0);
}

#[derive(Debug)]
struct Unsupported;
impl MiniBlockDecompressor for Unsupported {
    fn decompress(&self, _: Vec<LanceBuffer>, _: u64) -> Result<DataBlock> {
        panic!("admission must not use the ordinary decoder");
    }
}

#[rstest]
#[case::invalid_levels(0, "level exceeds")]
#[case::wrong_item_count(1, "item count")]
#[case::invalid_preamble(2, "preamble")]
#[case::out_of_range(3, "repetition boundaries")]
#[case::unsupported_value_codec(4, "no admitted range decoder")]
fn errors_recycle_level_workspace(#[case] fault: u8, #[case] message: &str) {
    let rep = if fault == 2 {
        [0, 0, 1, 1, 1, 0]
    } else {
        [1, 0, 1, 1, 1, 0]
    };
    let def = if fault == 0 {
        [0, 1, 2, 7, 0, 0]
    } else {
        [0, 1, 2, 3, 0, 0]
    };
    let Fixture {
        mut task,
        budget,
        mut work,
        physical,
    } = fixture(Some(&rep), Some(&def), Codec::Rle, false, 128);
    if fault == 1 {
        task.instructions[0].1.items_in_chunk = 5;
    }
    if fault == 2 {
        task.instructions[0].0.rows_to_take = 2;
    }
    if fault == 3 {
        task.instructions[0].0.rows_to_take = 10;
    }
    if fault == 4 {
        task.value_decompressor = Arc::new(Unsupported);
    }
    let error = task.try_decode_in(&mut work).err().unwrap();
    assert!(matches!(
        error,
        Error::InvalidInput { .. } | Error::NotSupported { .. }
    ));
    assert!(error.to_string().contains(message), "{error}");
    assert_eq!(work.len(), 128);
    assert_eq!(budget.reserved(), physical + 128);
}

#[rstest]
fn unsupported_task_shapes_preserve_input_and_credit(#[values(false, true)] dictionary: bool) {
    let Fixture {
        mut task,
        budget,
        mut work,
        physical,
    } = fixture(None, None, Codec::Flat, false, 128);
    if dictionary {
        task.dictionary_data = Some(Arc::new(DataBlock::AllNull(
            crate::data::AllNullDataBlock { num_values: 1 },
        )));
    } else {
        let (instruction, chunk) = &task.instructions[0];
        let duplicate = (
            ChunkDrainInstructions {
                chunk_instructions: instruction.chunk_instructions.clone(),
                rows_to_skip: instruction.rows_to_skip,
                rows_to_take: instruction.rows_to_take,
                preamble_action: instruction.preamble_action,
            },
            chunk.clone(),
        );
        task.instructions.push(duplicate);
    }
    let error = task.try_decode_in(&mut work).err().unwrap();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains(if dictionary {
        "dictionary/projection"
    } else {
        "one instruction"
    }));
    assert_eq!(work.len(), 128);
    assert_eq!(budget.reserved(), physical + 128);
}

#[cfg(feature = "bitpacking")]
#[rstest]
#[case::sparse(4, 1, 2)]
#[case::dense(128, 3, 64)]
#[case::full(1024, 0, 1024)]
#[case::empty(4, 2, 0)]
fn inline_values_preserve_range_decode_and_complete_scratch(
    #[case] stored: u64,
    #[case] start: u64,
    #[case] count: u64,
    #[values(false, true)] large: bool,
) {
    let Fixture {
        mut task,
        budget,
        mut work,
        ..
    } = fixture(None, None, Codec::Flat, large, 16_384);
    let mut values = [0_u32; 1024];
    values[..4].copy_from_slice(&[10, 0, 20, 30]);
    let mut packed = vec![5_u32; 161];
    // SAFETY: 1024 u32 values fit in five bits and the destination has 160 words.
    unsafe { u32::unchecked_pack(5, &values, &mut packed[1..]) };
    let payload: &[u8] = bytemuck::cast_slice(&packed);
    let mut frame = (stored as u16).to_le_bytes().to_vec();
    if large {
        frame.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    } else {
        frame.extend_from_slice(&(payload.len() as u16).to_le_bytes());
    }
    frame.resize(frame.len().next_multiple_of(MINIBLOCK_ALIGNMENT), 0);
    frame.extend_from_slice(payload);
    let physical = frame.len() + 38;
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..1 + frame.len()].copy_from_slice(&frame);
    task.instructions[0].1.data =
        LanceBuffer::from(Buffer::from(input.freeze()).slice_with_length(1, frame.len()));
    task.instructions[0].1.items_in_chunk = stored;
    task.instructions[0].1.byte_range = 0..frame.len() as u64;
    task.instructions[0].0.chunk_instructions.rows_to_take = stored;
    task.instructions[0].0.rows_to_skip = start;
    task.instructions[0].0.rows_to_take = count;
    task.value_decompressor = Arc::new(InlineBitpacking::new(32));
    // Price physical shapes independently: selected u32 values, an unaligned
    // encoded copy, and a full decoded chunk only for dense partial selection.
    let output = count as usize * 4;
    let alignment = if count > 0 { 644 } else { 0 };
    let scratch = if count > 32 && count < 1024 { 4096 } else { 0 };
    let required = output + alignment + scratch;
    if required > 0 {
        let mut short = work.try_split(required - 1).unwrap();
        assert!(task.try_decode_in(&mut short).unwrap().is_none());
        assert_eq!(short.len(), required - 1);
        drop(short);
    }
    drop(work);
    let mut work = budget.try_reserve(required).unwrap();
    let page = task.try_decode_in(&mut work).unwrap().unwrap();
    assert_values(&page, &values[start as usize..(start + count) as usize]);
    assert_eq!(work.len(), alignment + scratch);
    drop(work);
    assert_eq!(budget.reserved(), physical + output);
    drop(task);
    assert_eq!(budget.reserved(), output);
    drop(page);
    assert_eq!(budget.reserved(), 0);
}
