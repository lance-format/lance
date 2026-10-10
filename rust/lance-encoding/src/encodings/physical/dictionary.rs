// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dictionary encoding for bounded unsigned block sequences.

use std::{collections::BTreeMap, sync::Arc};

use crate::{
    buffer::LanceBuffer,
    compression::{BlockCompressor, BlockDecompressor},
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::physical::{checked_fixed_values, try_vec_with_capacity},
    format::{ProtobufUtils21, pb21::CompressiveEncoding},
};
use lance_core::{Error, Result};

pub(crate) const MAX_BLOCK_DICTIONARY_ITEMS: usize = 4096;
const BLOCK_FRAME_BYTES: usize = 16;

/// Encodes unsigned values through concrete index and dictionary-item codecs.
#[derive(Debug)]
pub struct BlockDictionaryEncoder {
    bits_per_value: u64,
    dictionary_items: Arc<[u64]>,
    indices: Box<dyn BlockCompressor>,
    items: Box<dyn BlockCompressor>,
}

impl BlockDictionaryEncoder {
    /// Creates a bounded dictionary codec for `u32` or `u64` values.
    pub fn try_new(
        bits_per_value: u64,
        dictionary_items: Arc<[u64]>,
        indices: Box<dyn BlockCompressor>,
        items: Box<dyn BlockCompressor>,
    ) -> Result<Self> {
        validate_dictionary_header(bits_per_value, dictionary_items.len())?;
        if bits_per_value == 32
            && dictionary_items
                .iter()
                .any(|value| *value > u32::MAX as u64)
        {
            return Err(Error::invalid_input(
                "Dictionary contains an item that exceeds u32::MAX",
            ));
        }
        Ok(Self {
            bits_per_value,
            dictionary_items,
            indices,
            items,
        })
    }
}

impl BlockCompressor for BlockDictionaryEncoder {
    fn compress(&self, data: DataBlock) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let DataBlock::FixedWidth(data) = data else {
            return Err(Error::invalid_input(
                "Dictionary block compression requires fixed-width data",
            ));
        };
        if data.bits_per_value != self.bits_per_value {
            return Err(Error::invalid_input(format!(
                "Dictionary codec expects {}-bit values, got {}",
                self.bits_per_value, data.bits_per_value
            )));
        }

        let dictionary_index = self
            .dictionary_items
            .iter()
            .enumerate()
            .map(|(index, value)| (*value, index as u32))
            .collect::<BTreeMap<_, _>>();
        let values = read_unsigned_values(&data, "Dictionary input")?;
        let mut encoded_indices =
            try_vec_with_capacity::<u32>(data.num_values, "Dictionary indices")?;
        for (position, value) in values.iter().enumerate() {
            encoded_indices.push(*dictionary_index.get(value).ok_or_else(|| {
                Error::invalid_input(format!(
                    "Dictionary does not contain input value {value} at position {position}"
                ))
            })?);
        }

        let indices_block = FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::reinterpret_vec(encoded_indices),
            num_values: data.num_values,
            block_info: BlockInfo::default(),
        };
        let items_block = fixed_from_u64_values(
            &self.dictionary_items,
            self.bits_per_value,
            "Dictionary items",
        )?;
        let (indices_payload, indices_encoding) = self
            .indices
            .compress(DataBlock::FixedWidth(indices_block))?;
        let (items_payload, items_encoding) =
            self.items.compress(DataBlock::FixedWidth(items_block))?;
        let encoding = ProtobufUtils21::dictionary(
            indices_encoding,
            items_encoding,
            self.dictionary_items.len() as u32,
        );
        if indices_payload.is_none() && items_payload.is_none() {
            return Ok((None, encoding));
        }

        let indices_payload = indices_payload.unwrap_or_else(LanceBuffer::empty);
        let items_payload = items_payload.unwrap_or_else(LanceBuffer::empty);
        let mut output = try_frame(indices_payload.len(), items_payload.len())?;
        output.extend_from_slice(&(indices_payload.len() as u64).to_le_bytes());
        output.extend_from_slice(&(items_payload.len() as u64).to_le_bytes());
        output.extend_from_slice(&indices_payload);
        output.extend_from_slice(&items_payload);
        Ok((Some(LanceBuffer::from(output)), encoding))
    }
}

/// Decodes a framed block dictionary through concrete child codecs.
#[derive(Debug)]
pub struct BlockDictionaryDecompressor {
    bits_per_value: u64,
    num_dictionary_items: u32,
    indices: Box<dyn BlockDecompressor>,
    items: Box<dyn BlockDecompressor>,
}

impl BlockDictionaryDecompressor {
    /// Creates a bounded dictionary decoder for `u32` or `u64` values.
    pub fn try_new(
        bits_per_value: u64,
        num_dictionary_items: u32,
        indices: Box<dyn BlockDecompressor>,
        items: Box<dyn BlockDecompressor>,
    ) -> Result<Self> {
        validate_dictionary_header(bits_per_value, num_dictionary_items as usize)?;
        Ok(Self {
            bits_per_value,
            num_dictionary_items,
            indices,
            items,
        })
    }
}

impl BlockDecompressor for BlockDictionaryDecompressor {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let indices_have_payload = self.indices.requires_payload();
        let items_have_payload = self.items.requires_payload();
        let has_payload = indices_have_payload || items_have_payload;
        let (indices_payload, items_payload) =
            split_payload(data, indices_have_payload, items_have_payload, has_payload)?;

        let indices = self.indices.decompress(indices_payload, num_values)?;
        let items = self
            .items
            .decompress(items_payload, u64::from(self.num_dictionary_items))?;
        let DataBlock::FixedWidth(indices) = indices else {
            return Err(Error::invalid_input(
                "Dictionary indices decoded to a non fixed-width block",
            ));
        };
        let DataBlock::FixedWidth(items) = items else {
            return Err(Error::invalid_input(
                "Dictionary items decoded to a non fixed-width block",
            ));
        };
        if indices.bits_per_value != 32 {
            return Err(Error::invalid_input(format!(
                "Dictionary indices decoded as {}-bit values, expected 32",
                indices.bits_per_value
            )));
        }
        if items.bits_per_value != self.bits_per_value {
            return Err(Error::invalid_input(format!(
                "Dictionary items decoded as {}-bit values, expected {}",
                items.bits_per_value, self.bits_per_value
            )));
        }

        let indices = checked_fixed_values::<u32>(&indices, "Dictionary indices")?;
        let items = read_unsigned_values(&items, "Dictionary items")?;
        let output = match self.bits_per_value {
            32 => {
                let mut output = try_vec_with_capacity::<u32>(num_values, "Dictionary output")?;
                append_items(
                    &mut output,
                    &indices,
                    &items,
                    self.num_dictionary_items,
                    |value| value as u32,
                )?;
                LanceBuffer::reinterpret_vec(output)
            }
            64 => {
                let mut output = try_vec_with_capacity::<u64>(num_values, "Dictionary output")?;
                append_items(
                    &mut output,
                    &indices,
                    &items,
                    self.num_dictionary_items,
                    |value| value,
                )?;
                LanceBuffer::reinterpret_vec(output)
            }
            _ => unreachable!("dictionary width was validated at construction"),
        };
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bits_per_value,
            data: output,
            num_values,
            block_info: BlockInfo::default(),
        }))
    }

    fn requires_payload(&self) -> bool {
        self.indices.requires_payload() || self.items.requires_payload()
    }
}

fn validate_dictionary_header(bits_per_value: u64, num_dictionary_items: usize) -> Result<()> {
    if !matches!(bits_per_value, 32 | 64) {
        return Err(Error::invalid_input(format!(
            "Dictionary only supports 32 or 64-bit values, got {bits_per_value}"
        )));
    }
    if !(1..=MAX_BLOCK_DICTIONARY_ITEMS).contains(&num_dictionary_items) {
        return Err(Error::invalid_input(format!(
            "Dictionary item count {num_dictionary_items} is outside 1..={MAX_BLOCK_DICTIONARY_ITEMS}"
        )));
    }
    Ok(())
}

fn split_payload(
    data: Option<LanceBuffer>,
    indices_have_payload: bool,
    items_have_payload: bool,
    has_payload: bool,
) -> Result<(Option<LanceBuffer>, Option<LanceBuffer>)> {
    let Some(data) = data else {
        if has_payload {
            return Err(Error::invalid_input("Dictionary requires one payload"));
        }
        return Ok((None, None));
    };
    if !has_payload {
        return Err(Error::invalid_input(
            "Metadata-only Dictionary expects no payload",
        ));
    }
    if data.len() < BLOCK_FRAME_BYTES {
        return Err(Error::invalid_input(format!(
            "Dictionary payload has {} bytes, shorter than its {BLOCK_FRAME_BYTES}-byte header",
            data.len()
        )));
    }

    let indices_size = u64::from_le_bytes(data[..8].try_into().unwrap());
    let items_size = u64::from_le_bytes(data[8..16].try_into().unwrap());
    let indices_size = usize::try_from(indices_size).map_err(|_| {
        Error::invalid_input("Dictionary indices payload length does not fit usize")
    })?;
    let items_size = usize::try_from(items_size)
        .map_err(|_| Error::invalid_input("Dictionary items payload length does not fit usize"))?;
    let items_start = BLOCK_FRAME_BYTES
        .checked_add(indices_size)
        .ok_or_else(|| Error::invalid_input("Dictionary indices payload end overflows"))?;
    let end = items_start
        .checked_add(items_size)
        .ok_or_else(|| Error::invalid_input("Dictionary items payload end overflows"))?;
    if end != data.len() {
        return Err(Error::invalid_input(format!(
            "Dictionary framing describes {end} bytes, payload has {}",
            data.len()
        )));
    }
    if !indices_have_payload && indices_size != 0 {
        return Err(Error::invalid_input(format!(
            "Metadata-only Dictionary indices child has {indices_size} framed payload bytes"
        )));
    }
    if !items_have_payload && items_size != 0 {
        return Err(Error::invalid_input(format!(
            "Metadata-only Dictionary items child has {items_size} framed payload bytes"
        )));
    }
    Ok((
        indices_have_payload.then(|| data.slice_with_length(BLOCK_FRAME_BYTES, indices_size)),
        items_have_payload.then(|| data.slice_with_length(items_start, items_size)),
    ))
}

fn append_items<T>(
    output: &mut Vec<T>,
    indices: &[u32],
    items: &[u64],
    num_dictionary_items: u32,
    convert: impl Fn(u64) -> T,
) -> Result<()> {
    for (position, index) in indices.iter().enumerate() {
        let value = *items.get(*index as usize).ok_or_else(|| {
            Error::invalid_input(format!(
                "Dictionary index {index} at position {position} is out of bounds for {num_dictionary_items} items"
            ))
        })?;
        output.push(convert(value));
    }
    Ok(())
}

fn read_unsigned_values(data: &FixedWidthDataBlock, label: &str) -> Result<Vec<u64>> {
    match data.bits_per_value {
        32 => Ok(checked_fixed_values::<u32>(data, label)?
            .iter()
            .map(|value| u64::from(*value))
            .collect()),
        64 => Ok(checked_fixed_values::<u64>(data, label)?.to_vec()),
        bits_per_value => Err(Error::invalid_input(format!(
            "{label} uses unsupported {bits_per_value}-bit values"
        ))),
    }
}

fn fixed_from_u64_values(
    values: &[u64],
    bits_per_value: u64,
    label: &str,
) -> Result<FixedWidthDataBlock> {
    let data = match bits_per_value {
        32 => LanceBuffer::reinterpret_vec(
            values
                .iter()
                .map(|value| {
                    u32::try_from(*value).map_err(|_| {
                        Error::invalid_input(format!("{label} value {value} exceeds u32::MAX"))
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        64 => LanceBuffer::reinterpret_vec(values.to_vec()),
        _ => unreachable!("dictionary width was validated at construction"),
    };
    Ok(FixedWidthDataBlock {
        bits_per_value,
        data,
        num_values: values.len() as u64,
        block_info: BlockInfo::default(),
    })
}

fn try_frame(indices_payload_bytes: usize, items_payload_bytes: usize) -> Result<Vec<u8>> {
    let capacity = BLOCK_FRAME_BYTES
        .checked_add(indices_payload_bytes)
        .and_then(|capacity| capacity.checked_add(items_payload_bytes))
        .ok_or_else(|| Error::invalid_input("Dictionary frame length overflows usize"))?;
    let mut output = Vec::new();
    output.try_reserve_exact(capacity).map_err(|error| {
        Error::invalid_input(format!(
            "Dictionary could not reserve {capacity} frame bytes: {error}"
        ))
    })?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use crate::encodings::physical::{
        range::{RangeDecompressor, RangeEncoder},
        value::{ValueDecompressor, ValueEncoder},
    };
    use crate::format::pb21::Flat;

    use super::*;

    fn input_u64(values: Vec<u64>) -> DataBlock {
        DataBlock::FixedWidth(FixedWidthDataBlock {
            num_values: values.len() as u64,
            bits_per_value: 64,
            data: LanceBuffer::reinterpret_vec(values),
            block_info: BlockInfo::default(),
        })
    }

    fn flat_decoder(bits_per_value: u64) -> Box<dyn BlockDecompressor> {
        Box::new(ValueDecompressor::from_flat(&Flat {
            bits_per_value,
            data: None,
        }))
    }

    fn decoded_u64(block: DataBlock) -> Vec<u64> {
        block
            .as_fixed_width()
            .unwrap()
            .data
            .borrow_to_typed_slice::<u64>()
            .to_vec()
    }

    #[test]
    fn dictionary_round_trip_with_payload_children() {
        let items: Arc<[u64]> = Arc::from([10, 20]);
        let encoder = BlockDictionaryEncoder::try_new(
            64,
            items,
            Box::new(ValueEncoder::default()),
            Box::new(ValueEncoder::default()),
        )
        .unwrap();
        let (payload, encoding) = encoder
            .compress(input_u64(vec![10, 20, 10, 20, 20]))
            .unwrap();
        let payload = payload.unwrap();
        assert_eq!(u64::from_le_bytes(payload[..8].try_into().unwrap()), 20);
        assert_eq!(u64::from_le_bytes(payload[8..16].try_into().unwrap()), 16);
        assert!(matches!(
            encoding.compression,
            Some(crate::format::pb21::compressive_encoding::Compression::Dictionary(_))
        ));

        let decoder =
            BlockDictionaryDecompressor::try_new(64, 2, flat_decoder(32), flat_decoder(64))
                .unwrap();
        assert_eq!(
            decoded_u64(decoder.decompress(Some(payload), 5).unwrap()),
            vec![10, 20, 10, 20, 20]
        );
    }

    #[test]
    fn dictionary_round_trip_without_payload() {
        let encoder = BlockDictionaryEncoder::try_new(
            64,
            Arc::from([10, 20, 30]),
            Box::new(RangeEncoder::new(32, 0, 1)),
            Box::new(RangeEncoder::new(64, 10, 10)),
        )
        .unwrap();
        let (payload, _) = encoder.compress(input_u64(vec![10, 20, 30])).unwrap();
        assert!(payload.is_none());

        let decoder = BlockDictionaryDecompressor::try_new(
            64,
            3,
            Box::new(RangeDecompressor::new(32, 0, 1)),
            Box::new(RangeDecompressor::new(64, 10, 10)),
        )
        .unwrap();
        assert!(!decoder.requires_payload());
        assert_eq!(
            decoded_u64(decoder.decompress(None, 3).unwrap()),
            vec![10, 20, 30]
        );
    }

    #[test]
    fn dictionary_frames_a_mixed_payload() {
        let encoder = BlockDictionaryEncoder::try_new(
            64,
            Arc::from([10, 20]),
            Box::new(ValueEncoder::default()),
            Box::new(RangeEncoder::new(64, 10, 10)),
        )
        .unwrap();
        let (payload, _) = encoder.compress(input_u64(vec![10, 20, 10])).unwrap();
        let payload = payload.unwrap();
        assert_eq!(u64::from_le_bytes(payload[..8].try_into().unwrap()), 12);
        assert_eq!(u64::from_le_bytes(payload[8..16].try_into().unwrap()), 0);

        let decoder = BlockDictionaryDecompressor::try_new(
            64,
            2,
            flat_decoder(32),
            Box::new(RangeDecompressor::new(64, 10, 10)),
        )
        .unwrap();
        assert_eq!(
            decoded_u64(decoder.decompress(Some(payload), 3).unwrap()),
            vec![10, 20, 10]
        );
    }

    #[test]
    fn dictionary_rejects_out_of_bounds_index() {
        let indices = LanceBuffer::reinterpret_vec(vec![0_u32, 2]);
        let items = LanceBuffer::reinterpret_vec(vec![9_u64]);
        let mut frame = try_frame(indices.len(), items.len()).unwrap();
        frame.extend_from_slice(&(indices.len() as u64).to_le_bytes());
        frame.extend_from_slice(&(items.len() as u64).to_le_bytes());
        frame.extend_from_slice(&indices);
        frame.extend_from_slice(&items);

        let decoder =
            BlockDictionaryDecompressor::try_new(64, 1, flat_decoder(32), flat_decoder(64))
                .unwrap();
        let error = decoder
            .decompress(Some(LanceBuffer::from(frame)), 2)
            .unwrap_err();
        assert!(error.to_string().contains("out of bounds"));
    }

    #[test]
    fn dictionary_rejects_invalid_frame_lengths() {
        let mut frame = Vec::from(16_u64.to_le_bytes());
        frame.extend_from_slice(&0_u64.to_le_bytes());

        let decoder =
            BlockDictionaryDecompressor::try_new(64, 1, flat_decoder(32), flat_decoder(64))
                .unwrap();
        let error = decoder
            .decompress(Some(LanceBuffer::from(frame)), 1)
            .unwrap_err();
        assert!(error.to_string().contains("framing describes"));
    }
}
