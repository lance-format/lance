// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{
    compression::{MiniBlockDecodeBudget, MiniBlockOutputLayout},
    data::{BlockInfo, FixedWidthDataBlock},
    encodings::physical::{
        binary::BinaryMiniBlockDecompressor,
        block::{CompressionConfig, CompressionScheme},
        rle::{RleDecompressor, RunLengthWidth},
        value::ValueDecompressor,
    },
    format::{EncodingUtils, metadata},
};
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;
use std::sync::Mutex;

#[cfg(feature = "lz4")]
use crate::{
    compression::{DecompressionStrategy, DefaultDecompressionStrategy},
    encodings::{
        logical::primitive::miniblock::{MiniBlockCompressionContext, MiniBlockCompressor},
        physical::{general::GeneralMiniBlockCompressor, value::ValueEncoder},
    },
};

struct Input {
    pool: Arc<AllocationBudget>,
    work: Reservation,
    buffers: Vec<LanceBuffer>,
    sibling: LanceBuffer,
    physical: usize,
}

fn input(parts: &[Vec<u8>], workspace: usize, displacement: usize) -> Input {
    let physical = displacement + parts.iter().map(|p| p.len() + 3).sum::<usize>() + 17;
    let pool = AllocationBudget::new(physical + workspace);
    let mut work = pool.try_reserve(physical + workspace).unwrap();
    let mut root = work.try_allocate(physical).unwrap().unwrap();
    let mut start = displacement;
    for part in parts {
        root.as_mut()[start..start + part.len()].copy_from_slice(part);
        start += part.len() + 3;
    }
    let root = Buffer::from(root.freeze());
    start = displacement;
    let buffers = parts
        .iter()
        .map(|part| {
            let buffer = LanceBuffer::from(root.slice_with_length(start, part.len()));
            start += part.len() + 3;
            buffer
        })
        .collect();
    let sibling = LanceBuffer::from(root.slice_with_length(physical - 1, 1));
    Input {
        pool,
        work,
        buffers,
        sibling,
        physical,
    }
}

fn encode_first(scheme: CompressionScheme, parts: &mut [Vec<u8>]) {
    let codec =
        GeneralBufferCompressor::get_compressor(CompressionConfig::new(scheme, None)).unwrap();
    let mut encoded = Vec::new();
    codec.compress(&parts[0], &mut encoded).unwrap();
    parts[0] = encoded;
}

fn flat() -> ValueDecompressor {
    ValueDecompressor::from_flat(&metadata::Flat {
        bits_per_value: 32,
        data: None,
    })
}

fn fixed_case(scheme: CompressionScheme, rle: bool, displacement: usize, range: Range<u64>) {
    let values = [10_u32, 10, 20, 20, 20, 30];
    let (inner, mut parts): (Box<dyn MiniBlockDecompressor>, _) = if rle {
        (
            Box::new(RleDecompressor::with_run_length_width(
                32,
                RunLengthWidth::U8,
            )),
            vec![
                bytemuck::cast_slice(&[10_u32, 20, 30]).to_vec(),
                vec![2, 3, 1],
            ],
        )
    } else {
        (
            Box::new(flat()),
            vec![bytemuck::cast_slice(&values).to_vec()],
        )
    };
    let scratch = parts[0].len();
    let output = (range.end - range.start) as usize * 4;
    encode_first(scheme, &mut parts);
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        physical,
    } = input(&parts, scratch + output, displacement);
    let codec = GeneralMiniBlockDecompressor::new(inner, CompressionConfig::new(scheme, None));
    let error = codec.plan_range(&buffers, 6, range.clone()).unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("no admitted range plan"));
    let total = work.len();
    // Refuse both before preparation and after it has filled the workspace.
    for capacity in [scratch - 1, total - 1] {
        let mut short = work.try_split(capacity).unwrap();
        for _ in 0..2 {
            assert!(
                codec
                    .try_prepare_and_decompress_range_in(&buffers, 6, range.clone(), &mut short)
                    .unwrap()
                    .is_none()
            );
            assert_eq!(short.len(), capacity);
            assert_eq!(pool.reserved(), physical + total);
        }
        drop(short);
        drop(work);
        work = pool.try_reserve(total).unwrap();
    }
    let decoded = codec
        .try_prepare_and_decompress_range_in(&buffers, 6, range.clone(), &mut work)
        .unwrap()
        .unwrap();
    let block = decoded.as_fixed_width_ref().unwrap();
    assert_eq!(block.num_values, range.end - range.start);
    assert_eq!(
        block.data.as_ref(),
        bytemuck::cast_slice::<u32, u8>(&values[range.start as usize..range.end as usize])
    );
    assert_eq!(work.len(), scratch);
    let last = block.data.clone();
    drop(buffers);
    assert_eq!(pool.reserved(), physical + total);
    drop(sibling);
    drop(work);
    drop(decoded);
    assert_eq!(pool.reserved(), output);
    drop(last);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::empty(2..2)]
#[case::prefix(0..2)]
#[case::cross_run(1..5)]
#[case::suffix(5..6)]
fn prepared_uncompressed_ranges(
    #[case] range: Range<u64>,
    #[values(false, true)] rle: bool,
    #[values(0, 1)] displacement: usize,
) {
    fixed_case(CompressionScheme::None, rle, displacement, range);
}

#[cfg(feature = "lz4")]
#[rstest]
#[case::empty(2..2)]
#[case::prefix(0..2)]
#[case::cross_run(1..5)]
#[case::suffix(5..6)]
fn prepared_lz4_ranges(
    #[case] range: Range<u64>,
    #[values(false, true)] rle: bool,
    #[values(0, 1)] displacement: usize,
) {
    fixed_case(CompressionScheme::Lz4, rle, displacement, range);
}

fn binary_case(bits: u8, scheme: CompressionScheme) {
    let width = usize::from(bits / 8);
    let offsets = [1_u64, 0, 3]
        .into_iter()
        .flat_map(|n| n.to_le_bytes()[..width].to_vec())
        .collect::<Vec<_>>();
    let mut parts = vec![offsets, b"abcd".to_vec()];
    let scratch = parts[0].len();
    encode_first(scheme, &mut parts);
    let description = EncodingUtils::variable(
        EncodingUtils::delta(
            u64::from(bits),
            0,
            EncodingUtils::flat(u64::from(bits), None),
        ),
        None,
    );
    let Some(metadata::compressive_encoding::Compression::Variable(variable)) =
        description.compression
    else {
        panic!("expected Variable")
    };
    let codec = GeneralMiniBlockDecompressor::new(
        Box::new(BinaryMiniBlockDecompressor::from_variable(&variable).unwrap()),
        CompressionConfig::new(scheme, None),
    );
    let output = 3 * width + 3;
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        ..
    } = input(&parts, scratch + output, 1);
    let decoded = codec
        .try_prepare_and_decompress_range_in(&buffers, 3, 1..3, &mut work)
        .unwrap()
        .unwrap();
    let DataBlock::VariableWidth(block) = decoded else {
        panic!("expected binary")
    };
    assert_eq!(block.data.as_ref(), b"bcd");
    let expected = [0_u64, 0, 3]
        .into_iter()
        .flat_map(|n| n.to_le_bytes()[..width].to_vec())
        .collect::<Vec<_>>();
    assert_eq!(block.offsets.as_ref(), expected);
    assert_eq!(work.len(), scratch);
    drop((block, work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn prepared_binary_offsets_and_payload(#[values(32, 64)] bits: u8) {
    binary_case(bits, CompressionScheme::None);
}

#[cfg(feature = "lz4")]
#[rstest]
fn prepared_lz4_binary_offsets_and_payload(#[values(32, 64)] bits: u8) {
    binary_case(bits, CompressionScheme::Lz4);
}

#[test]
fn prepared_finite_consumers_progress_after_final_output_drop() {
    let values = bytemuck::cast_slice(&[10_u32, 20, 30, 40]).to_vec();
    let codec = GeneralMiniBlockDecompressor::new(
        Box::new(flat()),
        CompressionConfig::new(CompressionScheme::None, None),
    );
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        physical,
    } = input(&[values], 32, 0);
    let first = codec
        .try_prepare_and_decompress_range_in(&buffers, 4, 1..3, &mut work)
        .unwrap()
        .unwrap();
    let alias = first.as_fixed_width_ref().unwrap().data.clone();
    let second = codec
        .try_prepare_and_decompress_range_in(&buffers, 4, 2..4, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), 16);
    assert!(
        codec
            .try_prepare_and_decompress_range_in(&buffers, 4, 0..2, &mut work)
            .unwrap()
            .is_none()
    );
    assert_eq!(work.len(), 16);
    drop(first);
    assert_eq!(pool.reserved(), physical + 32);
    drop(alias);
    drop(work);
    let mut work = pool.try_reserve(24).unwrap();
    let third = codec
        .try_prepare_and_decompress_range_in(&buffers, 4, 0..2, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(
        third.as_fixed_width_ref().unwrap().data.as_ref(),
        bytemuck::cast_slice::<u32, u8>(&[10_u32, 20])
    );
    drop((second, third, work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn prepared_nested_general_refuses_unplanned_inner_scratch() {
    let config = CompressionConfig::new(CompressionScheme::None, None);
    let codec = GeneralMiniBlockDecompressor::new(
        Box::new(GeneralMiniBlockDecompressor::new(Box::new(flat()), config)),
        config,
    );
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        ..
    } = input(&[vec![0; 4]], 16, 0);
    let error = codec
        .try_prepare_and_decompress_range_in(&buffers, 1, 0..1, &mut work)
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(error.to_string().contains("no admitted range plan"));
    assert_eq!(work.len(), 16);
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[cfg(feature = "lz4")]
#[rstest]
#[case::short_prefix(vec![4, 0], "no readable decoded length")]
#[case::truncated_literal(vec![4, 0, 0, 0, 0x40, 1], "decompression error")]
fn prepared_corrupt_frame_restores_credit(#[case] frame: Vec<u8>, #[case] message: &str) {
    let codec = GeneralMiniBlockDecompressor::new(
        Box::new(flat()),
        CompressionConfig::new(CompressionScheme::Lz4, None),
    );
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        physical,
    } = input(&[frame], 8, 1);
    for _ in 0..2 {
        let error = codec
            .try_prepare_and_decompress_range_in(&buffers, 1, 0..1, &mut work)
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(work.len(), 8);
        assert_eq!(pool.reserved(), physical + 8);
    }
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[cfg(feature = "zstd")]
#[test]
fn prepared_zstd_never_uses_unaccounted_context() {
    let mut parts = vec![vec![0; 4]];
    encode_first(CompressionScheme::Zstd, &mut parts);
    let codec = GeneralMiniBlockDecompressor::new(
        Box::new(flat()),
        CompressionConfig::new(CompressionScheme::Zstd, None),
    );
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        ..
    } = input(&parts, 8, 0);
    let error = codec
        .try_prepare_and_decompress_range_in(&buffers, 1, 0..1, &mut work)
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }));
    assert!(
        error.to_string().contains("no admitted exact decoder"),
        "{error}"
    );
    assert_eq!(work.len(), 8);
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::short_values(vec![1, 2, 3], None, 1)]
#[case::late_zero_run(bytemuck::cast_slice(&[10_u32, 20]).to_vec(), Some(vec![1, 0]), 2)]
fn prepared_inner_error_restores_credit(
    #[case] values: Vec<u8>,
    #[case] lengths: Option<Vec<u8>>,
    #[case] rows: u64,
) {
    let mut parts = vec![values];
    let inner: Box<dyn MiniBlockDecompressor> = if let Some(lengths) = lengths {
        parts.push(lengths);
        Box::new(RleDecompressor::with_run_length_width(
            32,
            RunLengthWidth::U8,
        ))
    } else {
        Box::new(flat())
    };
    let codec = GeneralMiniBlockDecompressor::new(
        inner,
        CompressionConfig::new(CompressionScheme::None, None),
    );
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        physical,
    } = input(&parts, 64, 1);
    for _ in 0..2 {
        let error = codec
            .try_prepare_and_decompress_range_in(&buffers, rows, 0..0, &mut work)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidInput { .. } | Error::CorruptFile { .. }
        ));
        assert!(
            error.to_string().contains(if parts.len() == 2 {
                "zero run length"
            } else {
                "buffer"
            }),
            "{error}"
        );
        assert_eq!(work.len(), 64);
        assert_eq!(pool.reserved(), physical + 64);
    }
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[derive(Debug)]
struct AliasingInner;
impl MiniBlockDecompressor for AliasingInner {
    fn decompress(&self, _: Vec<LanceBuffer>, _: u64) -> Result<DataBlock> {
        panic!("ordinary fallback")
    }
    fn admitted_output_layout(&self) -> Result<MiniBlockOutputLayout> {
        Ok(MiniBlockOutputLayout::FixedWidth { bits_per_value: 32 })
    }
    fn plan_range(
        &self,
        _: &[LanceBuffer],
        _: u64,
        _: Range<u64>,
    ) -> Result<MiniBlockDecodeBudget> {
        Ok(MiniBlockDecodeBudget {
            output_bytes: 4,
            workspace_bytes: 4,
        })
    }
    fn try_decompress_range_in(
        &self,
        data: &[LanceBuffer],
        _: u64,
        _: Range<u64>,
        _: &mut Reservation,
    ) -> Result<Option<DataBlock>> {
        Ok(Some(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: data[0].clone(),
            bits_per_value: 32,
            num_values: 1,
            block_info: BlockInfo::new(),
        })))
    }
}

#[test]
fn prepared_output_cannot_publish_scratch_alias() {
    let codec = GeneralMiniBlockDecompressor::new(
        Box::new(AliasingInner),
        CompressionConfig::new(CompressionScheme::None, None),
    );
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        physical,
    } = input(&[vec![1, 2, 3, 4]], 8, 0);
    let error = codec
        .try_prepare_and_decompress_range_in(&buffers, 1, 0..1, &mut work)
        .unwrap_err();
    assert!(matches!(error, Error::Internal { .. }));
    assert!(error.to_string().contains("retained preparation scratch"));
    assert_eq!(work.len(), 8);
    assert_eq!(pool.reserved(), physical + 8);
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}

#[derive(Debug)]
struct RetainingInner(Arc<Mutex<Option<LanceBuffer>>>);

impl MiniBlockDecompressor for RetainingInner {
    fn decompress(&self, _: Vec<LanceBuffer>, _: u64) -> Result<DataBlock> {
        panic!("ordinary fallback")
    }

    fn admitted_output_layout(&self) -> Result<MiniBlockOutputLayout> {
        AliasingInner.admitted_output_layout()
    }

    fn plan_range(
        &self,
        data: &[LanceBuffer],
        rows: u64,
        range: Range<u64>,
    ) -> Result<MiniBlockDecodeBudget> {
        AliasingInner.plan_range(data, rows, range)
    }

    fn try_decompress_range_in(
        &self,
        data: &[LanceBuffer],
        _: u64,
        _: Range<u64>,
        _: &mut Reservation,
    ) -> Result<Option<DataBlock>> {
        *self.0.lock().unwrap() = Some(data[0].clone());
        Err(Error::invalid_input("injected inner failure"))
    }
}

#[test]
fn prepared_external_alias_is_never_refunded_while_live() {
    let held = Arc::new(Mutex::new(None));
    let codec = GeneralMiniBlockDecompressor::new(
        Box::new(RetainingInner(held.clone())),
        CompressionConfig::new(CompressionScheme::None, None),
    );
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        physical,
    } = input(&[vec![1, 2, 3, 4]], 8, 0);
    let error = codec
        .try_prepare_and_decompress_range_in(&buffers, 1, 0..1, &mut work)
        .unwrap_err();
    assert!(matches!(error, Error::Internal { .. }));
    assert!(error.to_string().contains("external scratch alias"));
    assert_eq!(work.len(), 4);
    assert_eq!(pool.reserved(), physical + 8);
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 4);
    assert_eq!(held.lock().unwrap().take().unwrap().as_ref(), &[1, 2, 3, 4]);
    assert_eq!(pool.reserved(), 0);
}

#[cfg(feature = "lz4")]
#[test]
fn prepared_real_writer_and_factory_preserve_range_decode() {
    let expected = (0..2048).map(|i| (i / 64) as u32).collect::<Vec<_>>();
    let writer = GeneralMiniBlockCompressor::new(
        Box::new(ValueEncoder::default()),
        CompressionConfig::new(CompressionScheme::Lz4, None),
    );
    let (encoded, description) = writer
        .compress(
            MiniBlockCompressionContext::new(0, true, true),
            DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::reinterpret_vec(expected.clone()),
                bits_per_value: 32,
                num_values: expected.len() as u64,
                block_info: BlockInfo::new(),
            }),
        )
        .unwrap();
    assert!(matches!(
        description.compression,
        Some(metadata::compressive_encoding::Compression::General(_))
    ));
    let strategy = DefaultDecompressionStrategy::default();
    let codec = strategy
        .create_miniblock_decompressor(&description, &strategy)
        .unwrap();
    let mut cursor = 0;
    let mut first_row = 0;
    for chunk in &encoded.chunks {
        let bytes = chunk.buffer_sizes[0] as usize;
        let part = encoded.data[0][cursor..cursor + bytes].to_vec();
        let rows = 1_usize << chunk.log_num_values;
        let range = 3..(rows as u64 - 2);
        let output_bytes = (rows - 5) * 4;
        let Input {
            pool,
            mut work,
            buffers,
            sibling,
            ..
        } = input(&[part], rows * 4 + output_bytes, 1);
        let result = codec
            .try_prepare_and_decompress_range_in(&buffers, rows as u64, range.clone(), &mut work)
            .unwrap()
            .unwrap();
        assert_eq!(
            result.as_fixed_width_ref().unwrap().data.as_ref(),
            bytemuck::cast_slice::<u32, u8>(&expected[first_row + 3..first_row + rows - 2])
        );
        assert_eq!(work.len(), rows * 4);
        drop((result, work, buffers, sibling));
        assert_eq!(pool.reserved(), 0);
        cursor += bytes;
        first_row += rows;
    }
    assert_eq!(first_row, expected.len());
}

#[rstest]
#[case::noop(CompressionScheme::None)]
#[cfg_attr(feature = "lz4", case::lz4(CompressionScheme::Lz4))]
fn prepared_range_can_recycle_before_any_output(#[case] scheme: CompressionScheme) {
    let config = CompressionConfig::new(scheme, None);
    let codec = GeneralMiniBlockDecompressor::new(Box::new(flat()), config);
    let compressor = GeneralBufferCompressor::get_compressor(config).unwrap();
    let mut encoded = Vec::new();
    compressor
        .compress(bytemuck::cast_slice(&[10_u32, 20, 30, 40]), &mut encoded)
        .unwrap();
    let Input {
        pool,
        mut work,
        buffers,
        sibling,
        physical,
    } = input(&[encoded], 32, 1);
    let first = codec
        .try_prepare_range_in(&buffers, 4, 0..2, &mut work)
        .unwrap()
        .unwrap();
    let second = codec
        .try_prepare_range_in(&buffers, 4, 2..4, &mut work)
        .unwrap()
        .unwrap();
    assert!(work.is_empty());
    assert_eq!(first.budget().output_bytes, 8);
    // A caller may abandon one prepared range and execute the other. No input
    // alias or prepared plan can refund the other allocation prematurely.
    first.recycle(&mut work).unwrap();
    assert_eq!(work.len(), 16);
    let result = second.try_decode(&mut work).unwrap().unwrap();
    assert_eq!(
        result.as_fixed_width_ref().unwrap().data.as_ref(),
        bytemuck::cast_slice::<u32, u8>(&[30, 40])
    );
    assert_eq!(work.len(), 24);
    drop(result);
    let abandoned = codec
        .try_prepare_range_in(&buffers, 4, 1..2, &mut work)
        .unwrap()
        .unwrap();
    assert_eq!(work.len(), 8);
    drop(abandoned);
    assert_eq!(work.len(), 8); // Drop refunds the pool, not an unrelated workspace.
    assert_eq!(pool.reserved(), physical + 8);
    drop((work, buffers, sibling));
    assert_eq!(pool.reserved(), 0);
}
