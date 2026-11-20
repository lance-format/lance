// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::format::EncodingUtils;
#[cfg(feature = "bitpacking")]
use crate::{
    compression::BlockCompressor, data::FixedWidthDataBlock,
    encodings::physical::bitpacking::OutOfLineBitpacking,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Copy, Debug)]
enum Codec {
    DeltaFlat,
    #[cfg(feature = "bitpacking")]
    Packed,
    #[cfg(feature = "bitpacking")]
    DeltaPacked,
}

fn payload(bits: u8, codec: Codec, offsets: &[u64]) -> (BinaryMiniBlockDecompressor, LanceBuffer) {
    let delta = match codec {
        Codec::DeltaFlat => true,
        #[cfg(feature = "bitpacking")]
        Codec::Packed => false,
        #[cfg(feature = "bitpacking")]
        Codec::DeltaPacked => true,
    };
    let values = if delta {
        offsets.windows(2).map(|v| v[1] - v[0]).collect::<Vec<_>>()
    } else {
        offsets.to_vec()
    };
    let mut bytes = Vec::new();
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes()[..usize::from(bits / 8)]);
    }
    let raw = LanceBuffer::from(bytes);
    let (data, encoding) = match codec {
        Codec::DeltaFlat => (raw, EncodingUtils::flat(u64::from(bits), None)),
        #[cfg(feature = "bitpacking")]
        _ => {
            let (data, encoding) = OutOfLineBitpacking::new(3, u64::from(bits))
                .compress(DataBlock::FixedWidth(FixedWidthDataBlock {
                    bits_per_value: u64::from(bits),
                    num_values: values.len() as u64,
                    data: raw,
                    block_info: BlockInfo::new(),
                }))
                .unwrap();
            (data.unwrap(), encoding)
        }
    };
    let encoding = if delta {
        EncodingUtils::delta(u64::from(bits), offsets[0], encoding)
    } else {
        encoding
    };
    let encoding = EncodingUtils::variable(encoding, None);
    let Some(Compression::Variable(variable)) = encoding.compression else {
        panic!("expected Variable")
    };
    (
        BinaryMiniBlockDecompressor::from_variable(&variable).unwrap(),
        data,
    )
}

#[rstest]
fn delta_flat_offsets_keep_only_selected_payload(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] empty: bool,
) {
    check_selected_payload(Codec::DeltaFlat, bits, empty);
}

#[cfg(feature = "bitpacking")]
#[rstest]
#[case::packed(Codec::Packed)]
#[case::delta_packed(Codec::DeltaPacked)]
fn bitpacked_offsets_keep_only_selected_payload(
    #[case] kind: Codec,
    #[values(32, 64)] bits: u8,
    #[values(false, true)] empty: bool,
) {
    check_selected_payload(kind, bits, empty);
}

fn check_selected_payload(kind: Codec, bits: u8, empty: bool) {
    let (codec, encoded) = payload(bits, kind, &[0, 1, 1, 3, 4]);
    let range = if empty { 2..2 } else { 1..3 };
    let selected: &[&[u8]] = if empty { &[] } else { &[b"", b"bc"] };
    let required = (selected.len() + 1) * usize::from(bits / 8) + if empty { 0 } else { 2 };
    let physical = encoded.len() + 45;
    let pool = AllocationBudget::new(physical + required);
    let mut work = pool.try_reserve(physical + required).unwrap();
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..1 + encoded.len()].copy_from_slice(&encoded);
    input.as_mut()[encoded.len() + 2..encoded.len() + 6].copy_from_slice(b"abcd");
    let owner = Buffer::from(input.freeze());
    let sibling = owner.slice_with_length(physical - 1, 1);
    let data = [
        LanceBuffer::from(owner.slice_with_length(1, encoded.len())),
        LanceBuffer::from(owner.slice_with_length(encoded.len() + 2, 4)),
    ];
    drop(owner);
    assert_eq!(
        codec
            .plan_range(&data, 4, range.clone())
            .unwrap()
            .workspace_bytes,
        required
    );
    let mut short = work.try_split(required - 1).unwrap();
    for _ in 0..2 {
        assert!(
            codec
                .try_decompress_range_in(&data, 4, range.clone(), &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), required - 1);
        assert_eq!(pool.reserved(), physical + required);
    }
    drop(short);
    drop(work);
    let mut work = pool.try_reserve(required).unwrap();
    let output = codec
        .try_decompress_range_in(&data, 4, range, &mut work)
        .unwrap()
        .unwrap();
    let ty = if bits == 32 {
        DataType::Binary
    } else {
        DataType::LargeBinary
    };
    let actual = output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(actual, expected(selected, bits));
    assert_eq!(actual.get_buffer_memory_size(), required);
    assert!(work.is_empty());
    drop(data);
    drop(output);
    drop(work);
    assert_eq!(pool.reserved(), physical + required);
    drop(sibling);
    assert_eq!(pool.reserved(), required);
    drop(actual);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::late_bounds(0)]
#[case::late_overflow(1)]
#[case::truncated_offsets(2)]
#[case::extra_offsets(3)]
#[case::value_length(4)]
fn corrupt_encoded_offsets_preserve_workspace(
    #[case] fault: u8,
    #[values(32, 64)] bits: u8,
    #[values(false, true)] empty: bool,
) {
    let (codec, encoded) = payload(bits, Codec::DeltaFlat, &[0, 1, 1, 3, 4]);
    let mut bytes = encoded.to_vec();
    let width = usize::from(bits / 8);
    match fault {
        0 => bytes[3 * width..4 * width].copy_from_slice(&100_u64.to_le_bytes()[..width]),
        1 => bytes[3 * width..4 * width].fill(255),
        2 => {
            bytes.pop();
        }
        3 => bytes.extend_from_slice(&0_u64.to_le_bytes()[..width]),
        _ => {}
    }
    let data = [
        LanceBuffer::from(bytes),
        LanceBuffer::from(vec![0_u8; if fault == 4 { 5 } else { 4 }]),
    ];
    let pool = AllocationBudget::new(64);
    let mut work = pool.try_reserve(64).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(&data, 4, if empty { 1..1 } else { 0..1 }, &mut work)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { .. } | Error::CorruptFile { .. }
        ));
        assert_eq!(work.len(), 64);
        assert_eq!(pool.reserved(), 64);
    }
}

#[derive(Debug, Default)]
struct FailingReplay(AtomicUsize);

impl BlockDecompressor for FailingReplay {
    fn decompress(&self, _: Option<LanceBuffer>, _: u64) -> Result<DataBlock> {
        panic!("ordinary decode must not run")
    }

    fn visit_unsigned(
        &self,
        _: Option<&LanceBuffer>,
        _: u64,
        _: u64,
        visitor: &mut dyn FnMut(u64) -> Result<()>,
    ) -> Result<()> {
        let replay = self.0.fetch_add(1, Ordering::SeqCst) % 2 == 1;
        for (i, value) in [0, 1, 1, 3, 4].into_iter().enumerate() {
            visitor(value)?;
            if replay && i == 2 {
                return Err(Error::invalid_input("visitor replay failed"));
            }
        }
        Ok(())
    }
}

#[test]
fn failed_offset_replay_returns_output_credit_to_workspace() {
    let codec = BinaryMiniBlockDecompressor {
        layout: BinaryMiniBlockLayout::Generic {
            bits_per_offset: 32,
            offsets_have_payload: true,
            offsets: Box::<FailingReplay>::default(),
        },
    };
    let data = [LanceBuffer::empty(), LanceBuffer::from(b"abcd".to_vec())];
    let pool = AllocationBudget::new(14);
    let mut work = pool.try_reserve(14).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(&data, 4, 1..3, &mut work)
            .unwrap_err();
        assert!(error.to_string().contains("visitor replay failed"));
        assert_eq!(work.len(), 14);
        assert_eq!(pool.reserved(), 14);
    }
}

#[cfg(feature = "bitpacking")]
#[rstest]
fn current_writer_encoded_offsets_round_trip(#[values(32, 64)] bits: u8) {
    let values = (0..4096)
        .map(|i| vec![b'a' + (i % 20) as u8; [1, 7, 2, 5][i % 4]])
        .collect::<Vec<_>>();
    let refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let array: arrow_array::ArrayRef = if bits == 32 {
        std::sync::Arc::new(BinaryArray::from_vec(refs))
    } else {
        std::sync::Arc::new(LargeBinaryArray::from_vec(refs))
    };
    let (encoded, encoding) =
        BinaryMiniBlockEncoder::with_generic_offsets(Some(1024), CompressionFieldParams::default())
            .compress(
                MiniBlockCompressionContext::new(0, true, true),
                DataBlock::from(array),
            )
            .unwrap();
    let Some(Compression::Variable(variable)) = encoding.compression else {
        panic!("expected Variable")
    };
    assert!(matches!(
        variable.offsets.as_ref().unwrap().compression,
        Some(Compression::Delta(_))
    ));
    assert_eq!(encoded.data.len(), 2);
    assert!(encoded.chunks.len() > 1);
    let codec = BinaryMiniBlockDecompressor::from_variable(&variable).unwrap();
    let mut cursors = [0, 0];
    let mut row = 0;
    for chunk in &encoded.chunks {
        let count = if chunk.log_num_values == 0 {
            encoded.num_values as usize - row
        } else {
            1 << chunk.log_num_values
        };
        let data = chunk
            .buffer_sizes
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let value = encoded.data[i].slice_with_length(cursors[i], *n as usize);
                cursors[i] += *n as usize;
                value
            })
            .collect::<Vec<_>>();
        let selected = values[row..row + count.min(3)]
            .iter()
            .map(Vec::as_slice)
            .collect::<Vec<_>>();
        let bytes = (selected.len() + 1) * usize::from(bits / 8)
            + selected.iter().map(|v| v.len()).sum::<usize>();
        let pool = AllocationBudget::new(bytes);
        let mut work = pool.try_reserve(bytes).unwrap();
        let output = codec
            .try_decompress_range_in(&data, count as u64, 0..selected.len() as u64, &mut work)
            .unwrap()
            .unwrap();
        let ty = if bits == 32 {
            DataType::Binary
        } else {
            DataType::LargeBinary
        };
        assert_eq!(
            output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap(),
            expected(&selected, bits)
        );
        assert!(work.is_empty());
        row += count;
    }
    assert_eq!(row, values.len());
}
