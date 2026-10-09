// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cardinality-aware decoding for independently framed unsigned sequences.

use std::sync::Arc;

use lance_core::{Error, Result};

use crate::{
    buffer::LanceBuffer,
    compression::{BlockDecompressor, create_fixed_width_block_decompressor},
    data::DataBlock,
    encodings::physical::range::checked_range_last,
    format::pb21::{CompressiveEncoding, compressive_encoding::Compression},
};

/// Validated metadata that a container can use without materializing values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceMetadata {
    Empty,
    Constant(u64),
    Range { start: u64, step: u64 },
}

#[derive(Debug, Clone)]
pub struct SequenceDecoder {
    decoder: Arc<dyn BlockDecompressor>,
    num_values: u64,
    metadata: Option<SequenceMetadata>,
}

impl SequenceDecoder {
    pub(crate) fn try_new(
        encoding: &CompressiveEncoding,
        bits: u64,
        num_values: u64,
    ) -> Result<Self> {
        if matches!(encoding.compression.as_ref(), Some(Compression::Constant(c)) if c.value.is_none())
        {
            if num_values != 0 {
                return Err(Error::invalid_input(
                    "An absent unsigned Constant scalar requires zero values",
                ));
            }
        } else if num_values == 0 {
            return Err(Error::invalid_input(
                "An empty unsigned sequence requires Constant with no scalar",
            ));
        }
        let decoder = create_fixed_width_block_decompressor(encoding, bits)?;
        let metadata = metadata(encoding, bits, num_values)?;
        Ok(Self {
            decoder: Arc::from(decoder),
            num_values,
            metadata,
        })
    }

    pub(crate) fn metadata(&self) -> Option<SequenceMetadata> {
        self.metadata
    }

    pub(crate) fn requires_payload(&self) -> bool {
        self.decoder.requires_payload()
    }

    pub(crate) fn decode(&self, payload: Option<LanceBuffer>) -> Result<DataBlock> {
        self.decoder.decompress(payload, self.num_values)
    }
}

// The typed decoder has already validated scalar widths and bounded child grammar.
fn metadata(
    encoding: &CompressiveEncoding,
    bits: u64,
    count: u64,
) -> Result<Option<SequenceMetadata>> {
    Ok(match encoding.compression.as_ref() {
        Some(Compression::Constant(c)) => {
            let Some(scalar) = c.value.as_ref() else {
                return Ok(Some(SequenceMetadata::Empty));
            };
            let mut bytes = [0; 8];
            bytes[..scalar.len()].copy_from_slice(scalar);
            Some(SequenceMetadata::Constant(u64::from_le_bytes(bytes)))
        }
        Some(Compression::Range(r)) => {
            checked_range_last(bits, r.start, r.step, count)?;
            Some(SequenceMetadata::Range {
                start: r.start,
                step: r.step,
            })
        }
        Some(Compression::Delta(d)) => {
            if count < 2 || (bits == 32 && d.base > u32::MAX as u64) {
                return Err(Error::invalid_input("Invalid Delta cardinality or base"));
            }
            let child = d
                .deltas
                .as_deref()
                .ok_or_else(|| Error::invalid_input("Missing Delta child"))?;
            match metadata(child, bits, count - 1)? {
                Some(SequenceMetadata::Constant(0)) => Some(SequenceMetadata::Constant(d.base)),
                Some(SequenceMetadata::Constant(step)) => {
                    checked_range_last(bits, d.base, step, count)?;
                    Some(SequenceMetadata::Range {
                        start: d.base,
                        step,
                    })
                }
                _ => None,
            }
        }
        _ => None,
    })
}
