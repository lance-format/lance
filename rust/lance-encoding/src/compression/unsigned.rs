// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Borrowed unsigned traversal for container preflight. No payload allocations.

use crate::buffer::LanceBuffer;
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::out_of_line_payload_byte_lengths;
use lance_core::{Error, Result};

fn word(data: &[u8], index: usize, bytes: usize) -> u64 {
    let mut value = [0; 8];
    value[..bytes].copy_from_slice(&data[index * bytes..(index + 1) * bytes]);
    u64::from_le_bytes(value)
}

pub fn flat(
    data: &LanceBuffer,
    bits: u64,
    count: u64,
    visitor: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<()> {
    let width = flat_width(data, bits, count)?;
    for i in 0..data.len() / width {
        visitor(word(data, i, width))?;
    }
    Ok(())
}

pub fn flat_at(data: &LanceBuffer, bits: u64, count: u64, index: u64) -> Result<u64> {
    let width = flat_width(data, bits, count)?;
    if index >= count {
        return Err(Error::invalid_input(
            "unsigned Flat lookup index exceeds value count",
        ));
    }
    // Validated payload length makes the index addressable on this platform.
    Ok(word(data, index as usize, width))
}

fn flat_width(data: &LanceBuffer, bits: u64, count: u64) -> Result<usize> {
    if !matches!(bits, 32 | 64) {
        return Err(Error::invalid_input(
            "unsigned Flat visitor requires 32 or 64 bits",
        ));
    }
    let bytes = count
        .checked_mul(bits / 8)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| Error::invalid_input("unsigned Flat payload length overflows"))?;
    if bytes != data.len() {
        return Err(Error::corrupt_file_named(
            "unsigned Flat",
            "payload length disagrees with count",
        ));
    }
    Ok((bits / 8) as usize)
}

#[cfg(feature = "bitpacking")]
pub fn bitpacked(
    data: &LanceBuffer,
    bits: u64,
    packed: u64,
    count: u64,
    visitor: &mut dyn FnMut(u64) -> Result<()>,
) -> Result<()> {
    let raw = validate_bitpacked(data, bits, packed, count)?;
    let chunk_bytes = (1024 * packed / 8) as usize;
    let full_chunks = count / 1024;
    let tail = count % 1024;
    let word_bytes = (bits / 8) as usize;
    for chunk in 0..full_chunks + u64::from(tail > 0) {
        let start = usize::try_from(chunk)
            .ok()
            .and_then(|n| n.checked_mul(chunk_bytes))
            .ok_or_else(|| Error::invalid_input("unsigned bitpacking chunk position overflows"))?;
        let last = chunk == full_chunks;
        let num_values = if last { tail as usize } else { 1024 };
        if last && data.len() as u64 == raw {
            // Preserve the ordinary reader's raw-tail precedence when sizes tie.
            for i in 0..num_values {
                visitor(word(&data[start..], i, word_bytes))?;
            }
        } else {
            let input = &data[start..start + chunk_bytes];
            for i in 0..num_values {
                visitor(packed_value(input, bits as usize, packed as usize, i))?;
            }
        }
    }
    Ok(())
}

#[cfg(feature = "bitpacking")]
pub fn bitpacked_at(
    data: &LanceBuffer,
    bits: u64,
    packed: u64,
    count: u64,
    index: u64,
) -> Result<u64> {
    let raw = validate_bitpacked(data, bits, packed, count)?;
    if index >= count {
        return Err(Error::invalid_input(
            "unsigned bitpacking lookup index exceeds value count",
        ));
    }
    let chunk = index / 1024;
    let chunk_bytes = (1024 * packed / 8) as usize;
    let start = usize::try_from(chunk)
        .ok()
        .and_then(|n| n.checked_mul(chunk_bytes))
        .ok_or_else(|| Error::invalid_input("unsigned bitpacking chunk position overflows"))?;
    let index = (index % 1024) as usize;
    if chunk == count / 1024 && data.len() as u64 == raw {
        Ok(word(&data[start..], index, (bits / 8) as usize))
    } else {
        Ok(packed_value(
            &data[start..start + chunk_bytes],
            bits as usize,
            packed as usize,
            index,
        ))
    }
}

#[cfg(feature = "bitpacking")]
fn validate_bitpacked(data: &LanceBuffer, bits: u64, packed: u64, count: u64) -> Result<u64> {
    if !matches!(bits, 32 | 64) || packed == 0 || packed >= bits {
        return Err(Error::invalid_input(
            "unsigned bitpacking visitor requires a narrower positive width",
        ));
    }
    let [raw, padded] = out_of_line_payload_byte_lengths(count, bits, packed)?;
    if data.len() as u64 != raw && data.len() as u64 != padded {
        return Err(Error::corrupt_file_named(
            "unsigned bitpacking",
            "payload length disagrees with count",
        ));
    }
    Ok(raw)
}

#[cfg(feature = "bitpacking")]
fn packed_value(data: &[u8], bits: usize, width: usize, index: usize) -> u64 {
    // The same FastLanes permutation as BitPacking::unchecked_unpack_single,
    // but byte reads tolerate sliced, unaligned input without copying a chunk.
    let lanes = 1024 / bits;
    let lane = index & (lanes - 1);
    let order = [0, 4, 2, 6, 1, 5, 3, 7];
    let row = (order[(index & 127 & !(lanes - 1)) >> 4] << 3) | (index >> 7);
    let position = row * width;
    let current = position / bits;
    let shift = position % bits;
    let first = word(data, lanes * current + lane, bits / 8) >> shift;
    let value = if shift + width > bits {
        first | (word(data, lanes * (current + 1) + lane, bits / 8) << (bits - shift))
    } else {
        first
    };
    value & ((1_u64 << width) - 1)
}

#[cfg(test)]
mod tests;
