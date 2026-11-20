// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{Array, BinaryArray, LargeBinaryArray};
use arrow_schema::DataType;
use lance_core::utils::allocation::AllocationBudget;
use rstest::rstest;

use super::*;
use crate::{
    compression::{DecompressionStrategy, DefaultDecompressionStrategy, MiniBlockDecompressor},
    encodings::{
        logical::primitive::miniblock::{MiniBlockCompressionContext, MiniBlockCompressor},
        physical::fsst::FsstMiniBlockEncoder,
    },
    format::metadata::{CompressiveEncoding, compressive_encoding::Compression},
};

fn fixture(
    bits: u8,
    compressed: bool,
) -> (
    Box<dyn MiniBlockDecompressor>,
    LanceBuffer,
    Vec<Vec<u8>>,
    CompressiveEncoding,
) {
    let repeat = if compressed { 16384 } else { 1 };
    let values = vec![
        vec![b'a'; repeat],
        vec![],
        vec![b'b'; repeat * 2],
        b"d".to_vec(),
    ];
    let refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let block = if bits == 32 {
        DataBlock::from_array(BinaryArray::from_vec(refs))
    } else {
        DataBlock::from_array(LargeBinaryArray::from_vec(refs))
    };
    let (encoded, description) = FsstMiniBlockEncoder::new(Some(1 << 20))
        .compress(MiniBlockCompressionContext::new(0, true, true), block)
        .unwrap();
    assert_eq!(encoded.chunks.len(), 1);
    assert_eq!(encoded.data.len(), 1);
    let Some(Compression::Fsst(fsst)) = &description.compression else {
        panic!("expected FSST")
    };
    let header = u64::from_ne_bytes(fsst.symbol_table[..8].try_into().unwrap());
    assert_eq!(header & (1 << 24) != 0, compressed);
    let strategy = DefaultDecompressionStrategy::default();
    let codec = strategy
        .create_miniblock_decompressor(&description, &strategy)
        .unwrap();
    (
        codec,
        encoded.data.into_iter().next().unwrap(),
        values,
        description,
    )
}

#[rstest]
fn fsst_ranges_admit_exact_expansion_and_keep_shared_owners(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] compressed: bool,
    #[values(0..4, 1..3, 1..2, 4..4)] range: Range<u64>,
    #[values(0, 1)] displacement: usize,
) {
    let (codec, encoded, values, _) = fixture(bits, compressed);
    let selected = &values[range.start as usize..range.end as usize];
    let width = usize::from(bits / 8);
    let needed = (selected.len() + 1) * width + selected.iter().map(Vec::len).sum::<usize>();
    let physical = encoded.len() + 37;
    let pool = AllocationBudget::new(physical + needed);
    let mut work = pool.try_reserve(physical + needed).unwrap();
    let mut input = work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[displacement..displacement + encoded.len()].copy_from_slice(&encoded);
    let root = Buffer::from(input.freeze());
    let buffers = [LanceBuffer::from(
        root.slice_with_length(displacement, encoded.len()),
    )];
    let sibling = root.slice_with_length(physical - 1, 1);
    drop(root);
    let plan = codec.plan_range(&buffers, 4, range.clone()).unwrap();
    assert_eq!(plan.output_bytes, needed);
    assert_eq!(plan.workspace_bytes, needed);
    let mut short = work.try_split(needed - 1).unwrap();
    for _ in 0..2 {
        assert!(
            codec
                .try_decompress_range_in(&buffers, 4, range.clone(), &mut short)
                .unwrap()
                .is_none()
        );
        assert_eq!(short.len(), needed - 1);
        assert_eq!(pool.reserved(), physical + needed);
    }
    drop((short, work));
    let mut work = pool.try_reserve(needed).unwrap();
    let output = codec
        .try_decompress_range_in(&buffers, 4, range, &mut work)
        .unwrap()
        .unwrap();
    assert!(work.is_empty());
    let refs = selected.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let (ty, expected) = if bits == 32 {
        (DataType::Binary, BinaryArray::from_vec(refs).to_data())
    } else {
        (
            DataType::LargeBinary,
            LargeBinaryArray::from_vec(refs).to_data(),
        )
    };
    let actual = output.try_to_arrow_in(&ty, &mut work).unwrap().unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.get_buffer_memory_size(), needed);
    drop((output, buffers, work));
    assert_eq!(pool.reserved(), physical + needed);
    drop(sibling);
    assert_eq!(pool.reserved(), needed);
    drop(actual);
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
fn fsst_unselected_tail_corruption_precedes_admission(
    #[values(32, 64)] bits: u8,
    #[values(false, true)] offset_corruption: bool,
    #[values(0..1, 4..4)] range: Range<u64>,
) {
    let (codec, encoded, _, _) = fixture(bits, true);
    let mut bytes = encoded.as_ref().to_vec();
    let width = usize::from(bits / 8);
    if offset_corruption {
        bytes[4 * width..5 * width].fill(0);
    } else {
        let mut start = [0; 8];
        start[..width].copy_from_slice(&bytes[3 * width..4 * width]);
        let last_start = u64::from_le_bytes(start) as usize;
        bytes[last_start] = 255;
        // Make the last value exactly one escape byte. Padding and the old
        // literal after that value cannot satisfy its missing payload.
        bytes[4 * width..5 * width]
            .copy_from_slice(&((last_start + 1) as u64).to_le_bytes()[..width]);
    }
    let pool = AllocationBudget::new(16);
    let mut work = pool.try_reserve(16).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(
                &[LanceBuffer::from(bytes.clone())],
                4,
                range.clone(),
                &mut work,
            )
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(
            error.to_string().contains(if offset_corruption {
                "offsets"
            } else {
                "escape"
            }),
            "{error}"
        );
        assert_eq!(work.len(), 16);
        assert_eq!(pool.reserved(), 16);
    }
}

#[derive(Debug)]
struct ReplayError(AtomicUsize);

impl MiniBlockDecompressor for ReplayError {
    fn decompress(&self, _: Vec<LanceBuffer>, _: u64) -> Result<DataBlock> {
        panic!("ordinary fallback")
    }
    fn admitted_output_layout(&self) -> Result<MiniBlockOutputLayout> {
        Ok(MiniBlockOutputLayout::VariableWidth {
            bits_per_offset: 32,
        })
    }
    fn visit_binary_values(
        &self,
        _: &[LanceBuffer],
        _: u64,
        visitor: &mut dyn FnMut(&[u8]) -> Result<()>,
    ) -> Result<()> {
        visitor(b"a")?;
        if self.0.fetch_add(1, Ordering::SeqCst) % 2 == 1 {
            return Err(Error::corrupt_file_named("test", "injected replay error"));
        }
        for value in [b"".as_slice(), b"bb", b"d"] {
            visitor(value)?;
        }
        Ok(())
    }
}

#[test]
fn fsst_replay_error_returns_all_exclusive_output_credit() {
    let (_, _, _, description) = fixture(32, false);
    let Some(Compression::Fsst(description)) = description.compression else {
        panic!("expected FSST")
    };
    let codec =
        FsstMiniBlockDecompressor::new(&description, Box::new(ReplayError(AtomicUsize::new(0))));
    let pool = AllocationBudget::new(64);
    let mut work = pool.try_reserve(64).unwrap();
    for _ in 0..2 {
        let error = codec
            .try_decompress_range_in(&[], 4, 0..4, &mut work)
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("injected replay error"));
        assert_eq!(work.len(), 64);
        assert_eq!(pool.reserved(), 64);
    }
}

#[test]
fn fsst_finite_consumers_resume_after_last_output_alias() {
    let (codec, encoded, values, _) = fixture(32, true);
    let needed = 5 * 4 + values.iter().map(Vec::len).sum::<usize>();
    let physical = encoded.len() + 19;
    let pool = AllocationBudget::new(physical + 2 * needed);
    let mut input_work = pool.try_reserve(physical).unwrap();
    let mut input = input_work.try_allocate(physical).unwrap().unwrap();
    input.as_mut()[1..1 + encoded.len()].copy_from_slice(&encoded);
    let buffer = Buffer::from(input.freeze());
    let buffers = [LanceBuffer::from(
        buffer.slice_with_length(1, encoded.len()),
    )];
    drop((buffer, input_work));
    let consume = || {
        let mut work = pool.try_reserve(needed).unwrap();
        codec
            .try_decompress_range_in(&buffers, 4, 0..4, &mut work)
            .unwrap()
            .unwrap()
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
    let last_alias = first.data.slice_with_length(0, 1);
    drop(first);
    assert!(pool.try_reserve(needed).is_none());
    drop(last_alias);
    let third = consume();
    drop((second, third, buffers));
    assert_eq!(pool.reserved(), 0);
}
