// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};

use crate::encodings::physical::rle::RunLengthWidth;

/// Offsets into the borrowed frame, without typed views that could copy an
/// unaligned input. The immutable frame is validated before output admission.
#[derive(Debug)]
pub(super) struct RleLevelPlan {
    lengths_start: usize,
    width: RunLengthWidth,
}

impl RleLevelPlan {
    pub(super) fn new(data: &[u8], num_values: u64, width: RunLengthWidth) -> Result<Self> {
        let header = data.first_chunk::<8>().ok_or_else(|| {
            Error::invalid_input("RLE structural frame is shorter than its 8-byte header")
        })?;
        let values_bytes = usize::try_from(u64::from_le_bytes(*header))
            .map_err(|_| Error::invalid_input("RLE structural value length exceeds usize"))?;
        let lengths_start = values_bytes
            .checked_add(8)
            .filter(|end| *end <= data.len())
            .ok_or_else(|| Error::invalid_input("RLE structural values exceed frame length"))?;
        if !values_bytes.is_multiple_of(2) {
            return Err(Error::invalid_input(
                "RLE structural values have an incomplete u16",
            ));
        }
        let lengths = &data[lengths_start..];
        let runs = values_bytes / 2;
        match width {
            RunLengthWidth::U8 => validate_lengths::<1>(lengths, runs, num_values)?,
            RunLengthWidth::U16 => validate_lengths::<2>(lengths, runs, num_values)?,
            RunLengthWidth::U32 => validate_lengths::<4>(lengths, runs, num_values)?,
        }
        Ok(Self {
            lengths_start,
            width,
        })
    }

    pub(super) fn decode(&self, data: &[u8], output: &mut [u16]) {
        let values = &data[8..self.lengths_start];
        let lengths = &data[self.lengths_start..];
        match self.width {
            RunLengthWidth::U8 => fill::<1>(values, lengths, output),
            RunLengthWidth::U16 => fill::<2>(values, lengths, output),
            RunLengthWidth::U32 => fill::<4>(values, lengths, output),
        }
    }
}

fn read_length<const N: usize>(bytes: &[u8; N]) -> u32 {
    let mut word = [0_u8; 4];
    word[..N].copy_from_slice(bytes);
    u32::from_le_bytes(word)
}

fn validate_lengths<const N: usize>(bytes: &[u8], runs: usize, expected: u64) -> Result<()> {
    let (lengths, remainder) = bytes.as_chunks::<N>();
    if !remainder.is_empty() || lengths.len() != runs {
        return Err(Error::invalid_input(format!(
            "RLE structural frame has {runs} values but {} run-length bytes at width {N}",
            bytes.len()
        )));
    }
    let mut total = 0_u64;
    for bytes in lengths {
        let length = read_length(bytes);
        if length == 0 {
            return Err(Error::invalid_input("RLE structural run length is zero"));
        }
        total = total
            .checked_add(u64::from(length))
            .filter(|total| *total <= expected)
            .ok_or_else(|| Error::invalid_input("RLE structural runs exceed expected count"))?;
    }
    if total != expected {
        return Err(Error::invalid_input(format!(
            "RLE structural runs cover {total} values, expected {expected}"
        )));
    }
    Ok(())
}

fn fill<const N: usize>(values: &[u8], lengths: &[u8], mut output: &mut [u16]) {
    for (value, length) in values
        .as_chunks::<2>()
        .0
        .iter()
        .zip(lengths.as_chunks::<N>().0)
    {
        // The complete run sum and cardinality were checked against this exact
        // immutable input before allocating output. No run metadata is copied.
        let (run, remaining) = output.split_at_mut(read_length(length) as usize);
        run.fill(u16::from_le_bytes(*value));
        output = remaining;
    }
}
