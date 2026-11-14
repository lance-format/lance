// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact, exclusive output for codecs with no additional heap workspace.

use lance_core::{
    Error, Result,
    utils::allocation::{Reservation, ReservedBuffer},
};

pub(super) fn copy(
    input: &[u8],
    expected: usize,
    workspace: &mut Reservation,
) -> Result<Option<ReservedBuffer>> {
    if input.len() != expected {
        return Err(Error::invalid_input(format!(
            "Uncompressed payload has {} bytes, expected {expected}",
            input.len()
        )));
    }
    let Some(mut output) = workspace.try_allocate(expected)? else {
        return Ok(None);
    };
    output.as_mut().copy_from_slice(input);
    Ok(Some(output))
}

#[cfg(feature = "lz4")]
pub(super) fn lz4(
    input: &[u8],
    expected: usize,
    workspace: &mut Reservation,
) -> Result<Option<ReservedBuffer>> {
    let prefix: [u8; 4] = input
        .get(..4)
        .and_then(|prefix| prefix.try_into().ok())
        .ok_or_else(|| {
            Error::invalid_input("LZ4 payload is shorter than its 4-byte length prefix")
        })?;
    let declared = u32::from_le_bytes(prefix) as usize;
    if declared != expected {
        return Err(Error::invalid_input(format!(
            "LZ4 declares {declared} output bytes, expected {expected}"
        )));
    }
    // Check the pinned wrapper's signed C arguments before its usize-to-i32 cast.
    // The expansion check matches the ordinary reader's corrupt-prefix guard.
    if input.len() - 4 > i32::MAX as usize
        || expected > input.len().saturating_mul(255)
        || ::lz4::block::compress_bound(expected).is_err()
    {
        return Err(Error::invalid_input(
            "LZ4 frame exceeds the codec's size bounds",
        ));
    }
    let Some(mut output) = workspace.try_allocate(expected)? else {
        return Ok(None);
    };
    // lz4 1.28.1 calls LZ4_decompress_safe (no dictionary/context allocation).
    // It writes only to this caller-owned slice. Errors may leave a partial prefix,
    // so recycle the exclusive buffer before exposing either bytes or an error.
    let decoded = ::lz4::block::decompress_to_buffer(input, None, output.as_mut());
    let result = match decoded {
        Ok(written) if written == expected => Ok(()),
        Ok(written) => Err(Error::invalid_input(format!(
            "LZ4 produced {written} bytes, expected {expected}"
        ))),
        Err(error) => Err(Error::invalid_input(format!(
            "LZ4 decompression error: {error}"
        ))),
    };
    if let Err(error) = result {
        output.recycle(workspace)?;
        return Err(error);
    }
    Ok(Some(output))
}

#[cfg(test)]
mod tests;
