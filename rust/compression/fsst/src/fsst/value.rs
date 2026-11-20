// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact single-value expansion into caller-owned storage.

use std::io;

use super::{FSST_ESC, FsstDecompressor, missing_escape_payload_error};

impl FsstDecompressor {
    /// Validate one complete encoded value and return its exact decoded length.
    ///
    /// This does not allocate output or scratch. `input` must end at the value's
    /// boundary: an escape cannot borrow its literal from the next value. An
    /// undefined symbol or a truncated escape returns `InvalidData`. When the
    /// table disables compression, every input byte is literal.
    ///
    /// ```
    /// # use fsst::fsst::FsstDecompressor;
    /// # fn example(decoder: &FsstDecompressor, value: &[u8]) -> std::io::Result<()> {
    /// let bytes = decoder.decompressed_value_len(value)?;
    /// // A bounded caller admits this allocation before creating the buffer.
    /// let mut output = vec![0; bytes];
    /// decoder.decompress_value_into(value, &mut output)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn decompressed_value_len(&self, input: &[u8]) -> io::Result<usize> {
        let mut bytes = 0_usize;
        self.visit_value(input, |symbol| {
            bytes = bytes.checked_add(symbol.len()).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "FSST decoded value length overflows",
                )
            })?;
            Ok(())
        })?;
        Ok(bytes)
    }

    /// Decode one complete value into an exact-sized caller-owned output slice.
    ///
    /// The slice length must equal [`Self::decompressed_value_len`]. No output
    /// or scratch allocation occurs, and no padding or spare output capacity is
    /// required. The entire input and output length are checked before writing;
    /// malformed input or a length mismatch leaves `output` unchanged. See
    /// [`Self::decompressed_value_len`] for an example and value-boundary rules.
    pub fn decompress_value_into(&self, input: &[u8], output: &mut [u8]) -> io::Result<()> {
        let expected = self.decompressed_value_len(input)?;
        if output.len() != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "FSST output has {} bytes, expected {expected}",
                    output.len()
                ),
            ));
        }
        let mut position = 0;
        self.visit_value(input, |symbol| {
            let end = position + symbol.len();
            output[position..end].copy_from_slice(symbol);
            position = end;
            Ok(())
        })
    }

    fn visit_value(
        &self,
        input: &[u8],
        mut visitor: impl FnMut(&[u8]) -> io::Result<()>,
    ) -> io::Result<()> {
        if !self.decoder_switch_on {
            return visitor(input);
        }
        let mut codes = input.iter();
        while let Some(&code) = codes.next() {
            if code == FSST_ESC {
                let literal = codes.next().ok_or_else(missing_escape_payload_error)?;
                visitor(std::slice::from_ref(literal))?;
            } else {
                let len = self.lens[usize::from(code)];
                if len == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("FSST value references undefined symbol {code}"),
                    ));
                }
                // Match the native symbol bytes used by the existing bulk decoder,
                // copying only the declared length instead of an eight-byte store.
                let bytes = self.symbols[usize::from(code)].to_ne_bytes();
                visitor(&bytes[..usize::from(len)])?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
