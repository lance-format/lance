// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_buffer::Buffer;
use lance_core::{
    Error,
    utils::allocation::{AllocationBudget, Reservation},
};
use rstest::rstest;

use crate::{
    buffer::LanceBuffer, compression::BlockDecompressor,
    encodings::physical::value::ValueDecompressor, format::metadata,
};

mod admitted_rle;

fn owned_input(bytes: &[u8], unaligned: bool, work: &mut Reservation) -> (LanceBuffer, usize) {
    let start = if unaligned { 1 } else { 64 };
    let physical = start + bytes.len() + 37;
    let mut owner = work.try_allocate(physical).unwrap().unwrap();
    owner.as_mut()[start..start + bytes.len()].copy_from_slice(bytes);
    (
        LanceBuffer::from(Buffer::from(owner.freeze()).slice_with_length(start, bytes.len())),
        physical,
    )
}

fn check(
    codec: &dyn BlockDecompressor,
    bytes: &[u8],
    expected: &[u16],
    unaligned: bool,
    scratch: usize,
) {
    let output_bytes = expected.len() * 2;
    let required = output_bytes + scratch;
    let budget = AllocationBudget::new(bytes.len() + 101 + required);
    let mut work = budget.try_reserve(bytes.len() + 101 + required).unwrap();
    let (input, physical) = owned_input(bytes, unaligned, &mut work);
    let plan = codec.plan_levels(&input, expected.len() as u64).unwrap();
    assert_eq!(plan.output_bytes(), output_bytes);
    assert_eq!(plan.workspace_bytes(), required);
    if required > 0 {
        let mut short = work.try_split(required - 1).unwrap();
        assert!(plan.try_decode_in(&mut short).unwrap().is_none());
        assert_eq!(short.len(), required - 1);
        assert_eq!(input.as_ref(), bytes);
        drop(short);
    }
    drop(work);
    let mut work = budget.try_reserve(required).unwrap();
    let output = plan.try_decode_in(&mut work).unwrap().unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, u16>(output.as_ref()), expected);
    assert_eq!(
        work.len(),
        scratch,
        "all temporary scratch must return to this workspace"
    );
    let shared = Buffer::from(output.freeze());
    let last = shared.slice_with_length(0, output_bytes.min(2));
    drop(shared);
    drop(work);
    assert_eq!(budget.reserved(), physical + output_bytes);
    drop(input);
    assert_eq!(budget.reserved(), output_bytes);
    drop(last);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
fn flat_levels_preserve_values_and_owner(
    #[values(0, 1, 65, 1025)] count: usize,
    #[values(false, true)] unaligned: bool,
) {
    let values = (0..count).map(|i| (i % 513) as u16).collect::<Vec<_>>();
    let codec = ValueDecompressor::from_flat(&metadata::Flat {
        bits_per_value: 16,
        data: None,
    });
    check(&codec, bytemuck::cast_slice(&values), &values, unaligned, 0);
}

#[rstest]
#[case::short(16, 3, 2)]
#[case::long(16, 6, 2)]
#[case::wrong_width(32, 4, 2)]
#[case::overflow(16, 0, u64::MAX)]
fn invalid_flat_layout_precedes_admission(
    #[case] width: u64,
    #[case] bytes: usize,
    #[case] count: u64,
) {
    let codec = ValueDecompressor::from_flat(&metadata::Flat {
        bits_per_value: width,
        data: None,
    });
    let input = LanceBuffer::from(vec![0_u8; bytes]);
    let error = codec.plan_levels(&input, count).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("structural"));
}

#[cfg(feature = "bitpacking")]
mod bitpacked {
    use super::*;
    use crate::{
        compression::BlockCompressor,
        data::{BlockInfo, DataBlock, FixedWidthDataBlock},
        encodings::physical::bitpacking::{InlineBitpacking, OutOfLineBitpacking},
    };
    use lance_bitpacking::BitPacking;

    #[rstest]
    #[case::empty(3, 0)]
    #[case::zero_width(0, 1025)]
    #[case::raw_tail(3, 1025)]
    #[case::padded_tail(3, 2047)]
    #[case::complete_chunks(7, 3072)]
    #[case::equal_tail_size(1, 1088)]
    fn writer_payloads(
        #[case] width: u64,
        #[case] count: usize,
        #[values(false, true)] unaligned: bool,
    ) {
        let mask = ((1_u32 << width) - 1) as u16;
        let values = (0..count)
            .map(|i| (i as u16).wrapping_mul(37) & mask)
            .collect::<Vec<_>>();
        let codec = OutOfLineBitpacking::new(width, 16);
        let (input, _) = codec
            .compress(DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(values.clone()),
                bits_per_value: 16,
                num_values: count as u64,
                block_info: BlockInfo::new(),
            }))
            .unwrap();
        let input = input.unwrap();
        let chunk_bytes = width as usize * 128;
        let raw_bytes = count / 1024 * chunk_bytes + count % 1024 * 2;
        let is_padded_tail = !count.is_multiple_of(1024) && input.len() != raw_bytes;
        let scratch = if is_padded_tail { 2048 } else { 0 }
            + if unaligned && (count >= 1024 || is_padded_tail) {
                chunk_bytes
            } else {
                0
            };
        check(&codec, &input, &values, unaligned, scratch);
    }

    #[rstest]
    fn both_stable_tail_layouts(
        #[values(3, 16)] width: usize,
        #[values(false, true)] raw: bool,
        #[values(false, true)] unaligned: bool,
    ) {
        let mask = ((1_u32 << width) - 1) as u16;
        let values = (0..1500)
            .map(|i| (i as u16).wrapping_mul(37) & mask)
            .collect::<Vec<_>>();
        let mut padded = [0_u16; 2048];
        padded[..values.len()].copy_from_slice(&values);
        let chunks = if raw { 1 } else { 2 };
        let words_per_chunk = width * 64;
        let mut input = vec![0_u16; chunks * words_per_chunk];
        // Build both accepted reader layouts without invoking the writer's tail
        // compression heuristic, which requires a width smaller than the word.
        for chunk in 0..chunks {
            // SAFETY: 1024 u16 inputs fit in width bits; output has 1024*width/16 words.
            unsafe {
                u16::unchecked_pack(
                    width,
                    &padded[chunk * 1024..(chunk + 1) * 1024],
                    &mut input[chunk * words_per_chunk..(chunk + 1) * words_per_chunk],
                );
            }
        }
        if raw {
            input.extend_from_slice(&values[1024..]);
        }
        let codec = OutOfLineBitpacking::new(width as u64, 16);
        check(
            &codec,
            bytemuck::cast_slice(&input),
            &values,
            unaligned,
            if raw { 0 } else { 2048 } + if unaligned { words_per_chunk * 2 } else { 0 },
        );
    }

    #[rstest]
    #[case::odd_payload(3, 16, 1024, 383)]
    #[case::truncated_chunk(3, 16, 1024, 382)]
    #[case::extra_payload(3, 16, 1024, 386)]
    #[case::bad_width(17, 16, 1024, 0)]
    #[case::wrong_output(3, 32, 1024, 384)]
    fn malformed_layouts(
        #[case] width: u64,
        #[case] output_width: u64,
        #[case] count: u64,
        #[case] bytes: usize,
    ) {
        let input = LanceBuffer::from(vec![0_u8; bytes]);
        let error = OutOfLineBitpacking::new(width, output_width)
            .plan_levels(&input, count)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { .. } | Error::CorruptFile { .. }
        ));
        assert!(
            error.to_string().contains("structural")
                || error.to_string().contains("bitpacking width")
        );
    }

    #[test]
    fn unsupported_codec_does_not_use_ordinary_decode() {
        let error = InlineBitpacking::new(16)
            .plan_levels(&LanceBuffer::empty(), 0)
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(
            error
                .to_string()
                .contains("no admitted structural-level decoder")
        );
    }
}
