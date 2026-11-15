// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    ops::Range,
    sync::atomic::{AtomicUsize, Ordering},
};

use arrow_array::{Array, BinaryArray, LargeBinaryArray};
use arrow_schema::DataType;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;
use crate::{
    compression::VariablePerValueDecompressor,
    encodings::physical::{
        binary::VariableDecoder,
        fsst::test_utils::{charge_per_value, per_value_fixture},
    },
    format::metadata::compressive_encoding::Compression,
};

#[rstest]
fn per_value_ranges_keep_exact_output_and_shared_input(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] compressed: bool,
    #[values(0..4, 1..3, 1..2, 4..4)] range: Range<usize>,
    #[values(0, 1)] displacement: usize,
) {
    let (codec, mut input, values, _) = per_value_fixture(bits, compressed);
    let width = usize::from(bits / 8);
    let read_offset = |index: usize| {
        let mut word = [0; 8];
        word[..width].copy_from_slice(&input.offsets[index * width..(index + 1) * width]);
        u64::from_le_bytes(word) as usize
    };
    let start = read_offset(range.start);
    let end = read_offset(range.end);
    let mut offsets = Vec::new();
    for row in range.start..=range.end {
        offsets.extend_from_slice(&((read_offset(row) - start) as u64).to_le_bytes()[..width]);
    }
    input.data = input.data.slice_with_length(start, end - start);
    input.offsets = LanceBuffer::from(offsets);
    input.num_values = range.len() as u64;
    let selected = &values[range];
    let needed = (selected.len() + 1) * width + selected.iter().map(Vec::len).sum::<usize>();
    let physical = displacement + input.offsets.len() + 3 + input.data.len() + 37;
    let pool = AllocationBudget::new(physical + needed);
    let mut work = pool.try_reserve(physical + needed).unwrap();
    let (input, sibling, actual_physical) = charge_per_value(input, displacement, &mut work);
    assert_eq!(actual_physical, physical);
    let plan = codec.plan_decode(&input).unwrap();
    assert_eq!(
        plan,
        PerValueDecodeBudget {
            output_bytes: needed,
            workspace_bytes: needed
        }
    );
    let mut short = work.try_split(needed - 1).unwrap();
    for _ in 0..2 {
        assert!(
            codec
                .try_decompress_in(&input, &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), needed - 1);
        assert_eq!(pool.reserved(), physical + needed);
    }
    drop((short, work));
    let mut work = pool.try_reserve(needed).unwrap();
    let output = codec.try_decompress_in(&input, &mut work).unwrap().unwrap();
    let refs = selected.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (ty, expected) = if bits == 32 {
        (DataType::Binary, BinaryArray::from_vec(refs).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(refs).to_data(),
        )
    };
    let array = output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(array, expected);
    assert_eq!(array.get_buffer_memory_size(), needed);
    drop((input, output, work));
    assert_eq!(pool.reserved(), physical + needed);
    drop(sibling);
    assert_eq!(pool.reserved(), needed);
    drop(array);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn per_value_corruption_precedes_admission(
    #[values(32, 64)] bits: u8,
    #[values("decreasing", "signed", "truncated", "escape")] corruption: &str,
) {
    let (codec, mut input, _, _) = per_value_fixture(bits, true);
    let width = usize::from(bits / 8);
    let mut offsets = input.offsets.as_ref().to_vec();
    let expected = if corruption == "escape" {
        let mut word = [0; 8];
        word[..width].copy_from_slice(&offsets[3 * width..4 * width]);
        let last_start = u64::from_le_bytes(word) as usize;
        let mut bytes = input.data.as_ref().to_vec();
        bytes[last_start] = 255;
        offsets[4 * width..5 * width]
            .copy_from_slice(&((last_start + 1) as u64).to_le_bytes()[..width]);
        input.data = LanceBuffer::from(bytes);
        "escape"
    } else {
        match corruption {
            "decreasing" => offsets[4 * width..5 * width].fill(0),
            "signed" => offsets[5 * width - 1] = 128,
            "truncated" => {
                offsets.pop();
            }
            _ => unreachable!(),
        }
        "offset"
    };
    input.offsets = LanceBuffer::from(offsets);
    let physical = 1 + input.offsets.len() + 3 + input.data.len() + 37;
    let pool = AllocationBudget::new(physical);
    let mut work = pool.try_reserve(physical).unwrap();
    let (input, sibling, _) = charge_per_value(input, 1, &mut work);
    for _ in 0..2 {
        let error = codec.try_decompress_in(&input, &mut work).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(work.len(), 0);
        assert_eq!(pool.reserved(), physical);
    }
    drop((input, sibling, work));
    assert_eq!(pool.reserved(), 0);
}

#[derive(Debug)]
struct ReplayError(AtomicUsize);

impl VariablePerValueDecompressor for ReplayError {
    fn decompress(&self, _: VariableWidthBlock) -> Result<DataBlock> {
        panic!("ordinary decoder must not be called")
    }

    fn visit_binary_values(
        &self,
        data: &VariableWidthBlock,
        visitor: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        let replay = self.0.fetch_add(1, Ordering::SeqCst) % 2 == 1;
        VariableDecoder::default().visit_binary_values(data, &mut |value| {
            visitor(value)?;
            if replay {
                Err(Error::corrupt_file_named("test", "injected replay error"))
            } else {
                Ok(())
            }
        })
    }
}

#[test]
fn per_value_replay_error_recycles_exclusive_output() {
    let (_, input, _, description) = per_value_fixture(32, false);
    let Some(Compression::Fsst(description)) = description.compression else {
        panic!("expected FSST")
    };
    let codec = FsstPerValueDecompressor::new(
        LanceBuffer::from_bytes(description.symbol_table, 1),
        Box::new(ReplayError(AtomicUsize::new(0))),
    );
    let pool = AllocationBudget::new(128);
    let mut work = pool.try_reserve(128).unwrap();
    let (input, sibling, _) = charge_per_value(input, 1, &mut work);
    let available = work.len();
    for _ in 0..2 {
        let error = codec.try_decompress_in(&input, &mut work).unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("injected replay error"));
        assert_eq!(work.len(), available);
        assert_eq!(pool.reserved(), 128);
    }
    drop((input, sibling, work));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn per_value_borrowed_traversal_checks_tail_before_callback() {
    let (_, mut input, _, _) = per_value_fixture(32, false);
    let mut offsets = input.offsets.as_ref().to_vec();
    offsets[16..20].fill(0);
    input.offsets = LanceBuffer::from(offsets);
    let mut callbacks = 0;
    let error = VariableDecoder::default()
        .visit_binary_values(&input, &mut |_| {
            callbacks += 1;
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }));
    assert!(error.to_string().contains("offsets"));
    assert_eq!(callbacks, 0);
}

#[derive(Debug)]
struct Unsupported;

impl VariablePerValueDecompressor for Unsupported {
    fn decompress(&self, _: VariableWidthBlock) -> Result<DataBlock> {
        panic!("ordinary decode must not be a hidden allocation fallback")
    }
}

#[test]
fn per_value_unsupported_inner_preserves_workspace() {
    let (_, input, _, description) = per_value_fixture(32, false);
    let Some(Compression::Fsst(description)) = description.compression else {
        panic!("expected FSST")
    };
    let codec = FsstPerValueDecompressor::new(
        LanceBuffer::from_bytes(description.symbol_table, 1),
        Box::new(Unsupported),
    );
    let pool = AllocationBudget::new(128);
    let mut work = pool.try_reserve(128).unwrap();
    let (input, sibling, _) = charge_per_value(input, 1, &mut work);
    let available = work.len();
    let error = codec.try_decompress_in(&input, &mut work).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("borrowed binary traversal"));
    assert_eq!(work.len(), available);
    assert_eq!(pool.reserved(), 128);
    drop((input, sibling, work));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn per_value_finite_consumers_wait_for_final_payload_alias() {
    let (codec, input, values, _) = per_value_fixture(32, true);
    let needed = 20 + values.iter().map(Vec::len).sum::<usize>();
    let physical = 1 + input.offsets.len() + 3 + input.data.len() + 37;
    let pool = AllocationBudget::new(physical + 2 * needed);
    let mut input_work = pool.try_reserve(physical).unwrap();
    let (input, sibling, _) = charge_per_value(input, 1, &mut input_work);
    drop((sibling, input_work));
    let consume = || {
        let mut work = pool.try_reserve(needed).unwrap();
        codec.try_decompress_in(&input, &mut work).unwrap().unwrap()
    };
    let (first, second) = std::thread::scope(|scope| {
        let a = scope.spawn(consume);
        let b = scope.spawn(consume);
        (a.join().unwrap(), b.join().unwrap())
    });
    assert!(pool.try_reserve(1).is_none());
    let DataBlock::VariableWidth(first) = first else {
        panic!("expected binary")
    };
    let alias = first.data.slice_with_length(0, 1);
    drop(first);
    assert!(pool.try_reserve(needed).is_none());
    drop(alias);
    let third = consume();
    drop((second, third, input));
    assert_eq!(pool.reserved(), 0);
}
