// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zip a borrowed Flat child with a streaming child, without child arrays.

use super::super::{RleChildDecompressor, RleChildDecompressorInner, RleDecompressor};
use crate::buffer::LanceBuffer;
use lance_core::{Error, Result};

pub(super) fn visit(
    codec: &RleDecompressor,
    data: &[LanceBuffer],
    visitor: &mut dyn FnMut(u64, u64) -> Result<()>,
) -> Result<()> {
    let [values, lengths] = data else {
        return Err(Error::invalid_input(format!(
            "RLE decompressor expects exactly 2 buffers, got {}",
            data.len()
        )));
    };
    let value_width = (codec.bits_per_value / 8) as usize;
    let length_width = codec.run_length_width.bytes_per_value();
    if codec.values.is_identity() && codec.run_lengths.is_identity() {
        codec.validate_buffer_sizes(values, lengths, value_width, "admitted values")?;
        for (value, length) in values
            .chunks_exact(value_width)
            .zip(lengths.chunks_exact(length_width))
        {
            visitor(word(value), word(length))?;
        }
        return Ok(());
    }

    // The current writer never chooses two count-dependent children. A Flat
    // sibling gives the exact count without decoding a compressed child first.
    if codec.values.is_identity() {
        zip_with_flat::<true>(values, value_width, &codec.run_lengths, lengths, visitor)
    } else if codec.run_lengths.is_identity() {
        zip_with_flat::<false>(lengths, length_width, &codec.values, values, visitor)
    } else {
        Err(Error::not_supported(
            "admitted RLE values require at least one flat child to determine the run count",
        ))
    }
}

fn zip_with_flat<const VALUES_ARE_FLAT: bool>(
    flat: &LanceBuffer,
    width: usize,
    child: &RleChildDecompressor,
    payload: &LanceBuffer,
    visitor: &mut dyn FnMut(u64, u64) -> Result<()>,
) -> Result<()> {
    if !flat.len().is_multiple_of(width) {
        return Err(Error::invalid_input("RLE Flat child has a partial value"));
    }
    let count = flat.len() / width;
    let mut seen = 0;
    visit_child(child, payload, count as u64, &mut |value| {
        if seen >= count {
            return Err(Error::invalid_input("RLE child exceeds run count"));
        }
        let other = word(&flat[seen * width..(seen + 1) * width]);
        seen += 1;
        if VALUES_ARE_FLAT {
            visitor(other, value)
        } else {
            visitor(value, other)
        }
    })?;
    if seen != count {
        return Err(Error::invalid_input(format!(
            "RLE child visited {seen} runs, expected {count}"
        )));
    }
    Ok(())
}

fn visit_child(
    child: &RleChildDecompressor,
    data: &LanceBuffer,
    count: u64,
    visitor: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<()> {
    let RleChildDecompressorInner::Block { decompressor, .. } = &child.inner else {
        return Err(Error::internal(
            "RLE streaming child unexpectedly became Flat",
        ));
    };
    let max = u64::MAX >> (64 - child.bits_per_value);
    decompressor.visit_unsigned(Some(data), child.bits_per_value, count, &mut |value| {
        if value > max {
            return Err(Error::invalid_input(
                "RLE child value exceeds its declared width",
            ));
        }
        visitor(value)
    })
}

fn word(bytes: &[u8]) -> u64 {
    let mut word = [0; 8];
    word[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(word)
}
