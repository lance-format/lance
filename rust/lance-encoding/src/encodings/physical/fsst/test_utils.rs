// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{BinaryArray, LargeBinaryArray};
use arrow_buffer::Buffer;
use lance_core::utils::allocation::Reservation;

use super::FsstPerValueEncoder;
use crate::{
    buffer::LanceBuffer,
    compression::{
        DecompressionStrategy, DefaultDecompressionStrategy, VariablePerValueDecompressor,
    },
    data::{DataBlock, VariableWidthBlock},
    encodings::{
        logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock},
        physical::binary::VariableEncoder,
    },
    format::metadata::CompressiveEncoding,
};

pub fn per_value_fixture(
    bits: u8,
    compressed: bool,
) -> (
    Box<dyn VariablePerValueDecompressor>,
    VariableWidthBlock,
    Vec<Vec<u8>>,
    CompressiveEncoding,
) {
    let n = if compressed { 16384 } else { 1 };
    let values = vec![vec![b'a'; n], vec![], vec![b'b'; 2 * n], b"d".to_vec()];
    let refs = values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let block = if bits == 32 {
        DataBlock::from_array(BinaryArray::from_vec(refs))
    } else {
        DataBlock::from_array(LargeBinaryArray::from_vec(refs))
    };
    let (encoded, description) = FsstPerValueEncoder::new(Box::new(VariableEncoder::default()))
        .compress(block)
        .unwrap();
    let PerValueDataBlock::Variable(encoded) = encoded else {
        panic!("expected variable values")
    };
    let codec = DefaultDecompressionStrategy::default()
        .create_variable_per_value_decompressor(&description)
        .unwrap();
    (codec, encoded, values, description)
}

/// One charged coalesced owner, with independently sliced data/offsets and a
/// sibling outside both logical slices. Padding must stay charged too.
pub fn charge_per_value(
    mut block: VariableWidthBlock,
    displacement: usize,
    work: &mut Reservation,
) -> (VariableWidthBlock, Buffer, usize) {
    let data_start = displacement + block.offsets.len() + 3;
    let physical = data_start + block.data.len() + 37;
    let mut storage = work.try_allocate(physical).unwrap().unwrap();
    storage.as_mut()[displacement..displacement + block.offsets.len()]
        .copy_from_slice(&block.offsets);
    storage.as_mut()[data_start..data_start + block.data.len()].copy_from_slice(&block.data);
    let root = Buffer::from(storage.freeze());
    block.offsets = LanceBuffer::from(root.slice_with_length(displacement, block.offsets.len()));
    block.data = LanceBuffer::from(root.slice_with_length(data_start, block.data.len()));
    (block, root.slice_with_length(physical - 1, 1), physical)
}
