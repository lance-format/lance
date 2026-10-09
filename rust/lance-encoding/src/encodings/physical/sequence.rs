// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared bounded codec selection for unsigned block sequences.

use std::{ops::Range, str::FromStr, sync::Arc};

use lance_core::{Error, Result};
use prost::Message;

use crate::{
    buffer::LanceBuffer,
    compression::BlockCompressor,
    compression_config::CompressionFieldParams,
    data::{BlockInfo, DataBlock, FixedWidthDataBlock},
    encodings::physical::{
        block::{
            CompressedBufferEncoder, CompressionConfig, CompressionScheme, GeneralBufferCompressor,
        },
        checked_fixed_values,
        constant::ConstantEncoder,
        delta::DeltaEncoder,
        dictionary::{BlockDictionaryEncoder, MAX_BLOCK_DICTIONARY_ITEMS},
        general::GeneralBlockCompressor,
        range::RangeEncoder,
        rle::{BlockRleEncoder, GENERIC_BLOCK_RLE_HEADER_BYTES},
        value::ValueEncoder,
    },
    format::{ProtobufUtils21, pb21::CompressiveEncoding},
};

#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::{
    OutOfLineBitpacking, out_of_line_payload_byte_lengths,
};

mod decoder;
mod statistics;

pub use decoder::{SequenceDecoder, SequenceMetadata};

use statistics::{FamilyStats, SequenceStats, analyze_family, combine};

/// Container-specific serialized cost applied to every optional payload.
#[derive(Debug, Clone, Copy)]
pub struct BlockCost {
    buffer_overhead_bytes: u64,
    alignment: u64,
}

impl Default for BlockCost {
    fn default() -> Self {
        Self::new(0, 1)
    }
}

impl BlockCost {
    pub(crate) const fn new(buffer_overhead_bytes: u64, alignment: u64) -> Self {
        Self {
            buffer_overhead_bytes,
            alignment: if alignment == 0 { 1 } else { alignment },
        }
    }

    fn payload_bytes(self, has_payload: bool, payload_bytes: u64) -> u64 {
        let aligned = payload_bytes
            .checked_add(self.alignment - 1)
            .map(|padded| padded / self.alignment * self.alignment)
            .unwrap_or(u64::MAX);
        aligned.saturating_add(if has_payload {
            self.buffer_overhead_bytes
        } else {
            0
        })
    }
}

/// One concrete codec reused across independently decoded sequence chunks.
#[derive(Debug)]
pub struct SequenceBlockCodec {
    compressor: Box<dyn BlockCompressor>,
    expected_encoding: CompressiveEncoding,
    has_payload: bool,
    estimated_wire_bytes: u64,
}

impl SequenceBlockCodec {
    pub(crate) fn estimated_wire_bytes(&self) -> u64 {
        self.estimated_wire_bytes
    }

    pub(crate) fn expected_encoding(&self) -> &CompressiveEncoding {
        &self.expected_encoding
    }

    pub(crate) fn has_payload(&self) -> bool {
        self.has_payload
    }

    pub(crate) fn compress(
        &self,
        data: FixedWidthDataBlock,
    ) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let (payload, encoding) = self.compressor.compress(DataBlock::FixedWidth(data))?;
        if encoding != self.expected_encoding {
            return Err(Error::internal(
                "Sequence block codec produced a different descriptor after selection".to_string(),
            ));
        }
        if payload.is_some() != self.has_payload {
            return Err(Error::internal(
                "Sequence block codec changed its payload arity after selection".to_string(),
            ));
        }
        Ok((payload, encoding))
    }
}

#[derive(Debug)]
struct Candidate {
    compressor: Box<dyn BlockCompressor>,
    encoding: CompressiveEncoding,
    has_payload: bool,
    payload_bytes: Vec<u64>,
    wire_bytes: u64,
    transform_depth: u8,
    decode_cpu_rank: u8,
    stable_rank: u8,
}

impl Candidate {
    fn is_better_than(&self, other: &Self) -> bool {
        (
            self.wire_bytes,
            self.transform_depth,
            self.decode_cpu_rank,
            self.stable_rank,
        ) < (
            other.wire_bytes,
            other.transform_depth,
            other.decode_cpu_rank,
            other.stable_rank,
        )
    }

    fn finish(self, normalize: bool) -> SequenceBlockCodec {
        SequenceBlockCodec {
            compressor: if normalize {
                Box::new(NormalizedOffsetEncoder {
                    child: self.compressor,
                })
            } else {
                self.compressor
            },
            expected_encoding: self.encoding,
            has_payload: self.has_payload,
            estimated_wire_bytes: self.wire_bytes,
        }
    }
}

#[derive(Debug)]
struct NormalizedOffsetEncoder {
    child: Box<dyn BlockCompressor>,
}

impl BlockCompressor for NormalizedOffsetEncoder {
    fn compress(&self, data: DataBlock) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let DataBlock::FixedWidth(data) = data else {
            return Err(Error::invalid_input(
                "Offset compression requires fixed-width data",
            ));
        };
        self.child
            .compress(DataBlock::FixedWidth(normalize_offsets(data)?))
    }
}

/// Selects one bounded unsigned codec for all independently framed chunks.
pub fn select_block_codec(
    data: &FixedWidthDataBlock,
    ranges: &[Range<usize>],
    field_params: &CompressionFieldParams,
    cost: BlockCost,
    normalize: bool,
) -> Result<SequenceBlockCodec> {
    let collect_general_sample =
        !matches!(field_params.compression.as_deref(), Some("none" | "fsst"));
    let analysis = analyze_family(data, ranges, collect_general_sample, normalize)?;
    select_analyzed_codec(analysis, data.bits_per_value, field_params, cost, normalize)
}

/// Select a codec for generated u64 values without allocating the input sequence.
pub fn select_u64_sequence_codec(
    values: impl Iterator<Item = u64>,
    field_params: &CompressionFieldParams,
    cost: BlockCost,
) -> Result<SequenceBlockCodec> {
    let collect_sample = !matches!(field_params.compression.as_deref(), Some("none" | "fsst"));
    let analysis = statistics::analyze_sequence(values, collect_sample)?;
    select_analyzed_codec(analysis, 64, field_params, cost, false)
}

fn select_analyzed_codec(
    analysis: FamilyStats,
    bits: u64,
    field_params: &CompressionFieldParams,
    cost: BlockCost,
    normalize: bool,
) -> Result<SequenceBlockCodec> {
    let value_stats = analysis
        .members
        .iter()
        .map(|member| &member.values)
        .collect::<Vec<_>>();

    if value_stats.iter().any(|stats| stats.len == 0) {
        if value_stats.iter().any(|stats| stats.len != 0) {
            return Err(Error::invalid_input(
                "Empty and non-empty sequences cannot share one block descriptor",
            ));
        }
        return Ok(
            metadata_candidate(bits, 0, MetadataCodec::Constant, &value_stats, cost, 0)?
                .finish(false),
        );
    }

    if let Some(value) = common_constant(&value_stats) {
        return Ok(metadata_candidate(
            bits,
            value,
            MetadataCodec::Constant,
            &value_stats,
            cost,
            0,
        )?
        .finish(normalize));
    }
    if let Some((start, step)) = common_range(&value_stats) {
        return Ok(metadata_candidate(
            bits,
            start,
            MetadataCodec::Range(step),
            &value_stats,
            cost,
            1,
        )?
        .finish(normalize));
    }

    let mut candidates = direct_candidates(&value_stats, field_params, cost, true)?;
    if let Some(delta) = delta_candidate(&analysis, field_params, cost)? {
        candidates.push(delta);
    }
    if let Some(rle) = rle_candidate(&analysis, field_params, cost)? {
        candidates.push(rle);
    }
    if let Some(dictionary) = dictionary_candidate(&analysis, field_params, cost)? {
        candidates.push(dictionary);
    }

    candidates
        .into_iter()
        .reduce(|best, candidate| {
            if candidate.is_better_than(&best) {
                candidate
            } else {
                best
            }
        })
        .map(|candidate| candidate.finish(normalize))
        .ok_or_else(|| Error::internal("No unsigned block codec candidate".to_string()))
}

#[derive(Debug, Clone, Copy)]
enum MetadataCodec {
    Constant,
    Range(u64),
}

fn metadata_candidate(
    bits_per_value: u64,
    value: u64,
    codec: MetadataCodec,
    stats: &[&SequenceStats],
    cost: BlockCost,
    stable_rank: u8,
) -> Result<Candidate> {
    let num_values = stats[0].len;
    let (compressor, encoding): (Box<dyn BlockCompressor>, _) = match codec {
        MetadataCodec::Constant => {
            let encoder = ConstantEncoder::new(bits_per_value, value);
            let encoding = encoder.encoding(num_values)?;
            (Box::new(encoder), encoding)
        }
        MetadataCodec::Range(step) => {
            let encoder = RangeEncoder::new(bits_per_value, value, step);
            let encoding = encoder.encoding(num_values)?;
            (Box::new(encoder), encoding)
        }
    };
    let payload_bytes = vec![0; stats.len()];
    Ok(Candidate {
        wire_bytes: wire_bytes(&encoding, false, &payload_bytes, cost),
        compressor,
        encoding,
        has_payload: false,
        payload_bytes,
        transform_depth: 0,
        decode_cpu_rank: 0,
        stable_rank,
    })
}

fn direct_candidates(
    stats: &[&SequenceStats],
    field_params: &CompressionFieldParams,
    cost: BlockCost,
    allow_general: bool,
) -> Result<Vec<Candidate>> {
    let combined = combine(stats)?;
    let bits_per_value = combined.bits_per_value;
    let flat_payloads = stats
        .iter()
        .map(|stats| stats.raw_bytes())
        .collect::<Vec<_>>();
    let flat_encoding = ProtobufUtils21::flat(bits_per_value, None);
    let mut candidates = vec![Candidate {
        wire_bytes: wire_bytes(&flat_encoding, true, &flat_payloads, cost),
        compressor: Box::new(ValueEncoder::default()),
        encoding: flat_encoding,
        has_payload: true,
        payload_bytes: flat_payloads,
        transform_depth: 0,
        decode_cpu_rank: 0,
        stable_rank: 0,
    }];

    #[cfg(feature = "bitpacking")]
    if combined.len > 0 && combined.required_bits() < bits_per_value {
        let compressed_bits = combined.required_bits();
        let compressor = Box::new(OutOfLineBitpacking::new(compressed_bits, bits_per_value));
        let encoding = ProtobufUtils21::out_of_line_bitpacking(
            bits_per_value,
            ProtobufUtils21::flat(compressed_bits, None),
        );
        let payload_bytes = stats
            .iter()
            .map(|stats| {
                out_of_line_payload_byte_lengths(stats.len, bits_per_value, compressed_bits)
                    .map(|[raw, padded]| raw.min(padded))
            })
            .collect::<Result<Vec<_>>>()?;
        candidates.push(Candidate {
            wire_bytes: wire_bytes(&encoding, true, &payload_bytes, cost),
            compressor,
            encoding,
            has_payload: true,
            payload_bytes,
            transform_depth: 1,
            decode_cpu_rank: 1,
            stable_rank: 1,
        });
    }

    if allow_general
        && let Some(config) = general_config(&combined, field_params)?
        && config.scheme != CompressionScheme::None
    {
        let payload_bytes = stats
            .iter()
            .map(|stats| estimate_general_bytes(stats, config))
            .collect::<Result<Vec<_>>>()?;
        let encoding =
            ProtobufUtils21::wrapped(config, ProtobufUtils21::flat(bits_per_value, None))?;
        candidates.push(Candidate {
            wire_bytes: wire_bytes(&encoding, true, &payload_bytes, cost),
            compressor: Box::new(GeneralBlockCompressor::new(
                Box::new(ValueEncoder::default()),
                config,
            )),
            encoding,
            has_payload: true,
            payload_bytes,
            transform_depth: 1,
            decode_cpu_rank: 4,
            stable_rank: 2,
        });
    }
    Ok(candidates)
}

fn leaf_candidate(
    stats: &[&SequenceStats],
    field_params: &CompressionFieldParams,
) -> Result<Candidate> {
    let bits_per_value = stats[0].bits_per_value;
    if let Some(value) = common_constant(stats) {
        return metadata_candidate(
            bits_per_value,
            value,
            MetadataCodec::Constant,
            stats,
            BlockCost::default(),
            0,
        );
    }
    if let Some((start, step)) = common_range(stats) {
        return metadata_candidate(
            bits_per_value,
            start,
            MetadataCodec::Range(step),
            stats,
            BlockCost::default(),
            1,
        );
    }
    direct_candidates(stats, field_params, BlockCost::default(), false)?
        .into_iter()
        .reduce(|best, candidate| {
            if candidate.is_better_than(&best) {
                candidate
            } else {
                best
            }
        })
        .ok_or_else(|| Error::internal("No block leaf candidate".to_string()))
}

fn delta_candidate(
    analysis: &FamilyStats,
    field_params: &CompressionFieldParams,
    cost: BlockCost,
) -> Result<Option<Candidate>> {
    if analysis
        .members
        .iter()
        .any(|member| member.deltas.is_none())
    {
        return Ok(None);
    }
    let base = analysis.members[0]
        .values
        .first
        .expect("non-empty sequence");
    if analysis
        .members
        .iter()
        .any(|member| member.values.first != Some(base))
    {
        return Ok(None);
    }
    let stats = analysis
        .members
        .iter()
        .map(|member| member.deltas.as_ref().expect("delta presence was checked"))
        .collect::<Vec<_>>();
    let child = leaf_candidate(&stats, field_params)?;
    let bits_per_value = analysis.members[0].values.bits_per_value;
    let encoding = ProtobufUtils21::delta(bits_per_value, base, child.encoding);
    Ok(Some(Candidate {
        wire_bytes: wire_bytes(&encoding, child.has_payload, &child.payload_bytes, cost),
        compressor: Box::new(DeltaEncoder::new(bits_per_value, base, child.compressor)),
        encoding,
        has_payload: child.has_payload,
        payload_bytes: child.payload_bytes,
        transform_depth: child.transform_depth.saturating_add(1),
        decode_cpu_rank: 2,
        stable_rank: 5,
    }))
}

fn rle_candidate(
    analysis: &FamilyStats,
    field_params: &CompressionFieldParams,
    cost: BlockCost,
) -> Result<Option<Candidate>> {
    let total_values = analysis
        .members
        .iter()
        .map(|member| member.values.len)
        .sum::<u64>();
    let total_runs = analysis
        .members
        .iter()
        .map(|member| member.run_values.len)
        .sum::<u64>();
    if total_runs == 0 || total_runs >= total_values {
        return Ok(None);
    }
    if let Some(threshold) = field_params.rle_threshold
        && total_runs as f64 >= total_values as f64 * threshold
    {
        return Ok(None);
    }
    if analysis
        .members
        .iter()
        .any(|member| member.run_lengths.max > u32::MAX as u64)
    {
        return Ok(None);
    }

    let value_stats = analysis
        .members
        .iter()
        .map(|member| &member.run_values)
        .collect::<Vec<_>>();
    let length_stats = analysis
        .members
        .iter()
        .map(|member| &member.run_lengths)
        .collect::<Vec<_>>();
    let values = leaf_candidate(&value_stats, field_params)?;
    let mut lengths = leaf_candidate(&length_stats, field_params)?;
    let lengths_are_metadata = matches!(
        lengths.encoding.compression.as_ref(),
        Some(crate::format::pb21::compressive_encoding::Compression::Constant(_))
            | Some(crate::format::pb21::compressive_encoding::Compression::Range(_))
    );
    let values_are_flat = is_flat(&values.encoding);
    if !lengths_are_metadata && !values_are_flat && !is_flat(&lengths.encoding) {
        lengths = flat_candidate(&length_stats)?;
    }

    let has_payload = values.has_payload || lengths.has_payload;
    let payload_bytes = values
        .payload_bytes
        .iter()
        .zip(&lengths.payload_bytes)
        .map(|(values, lengths)| {
            if has_payload {
                (GENERIC_BLOCK_RLE_HEADER_BYTES as u64)
                    .saturating_add(*values)
                    .saturating_add(*lengths)
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let encoding = ProtobufUtils21::rle(values.encoding, lengths.encoding);
    let bits_per_value = analysis.members[0].values.bits_per_value;
    let compressor =
        BlockRleEncoder::try_new(bits_per_value, values.compressor, lengths.compressor)?;
    Ok(Some(Candidate {
        wire_bytes: wire_bytes(&encoding, has_payload, &payload_bytes, cost),
        compressor: Box::new(compressor),
        encoding,
        has_payload,
        payload_bytes,
        transform_depth: 1,
        decode_cpu_rank: 3,
        stable_rank: 3,
    }))
}

fn dictionary_candidate(
    analysis: &FamilyStats,
    field_params: &CompressionFieldParams,
    cost: BlockCost,
) -> Result<Option<Candidate>> {
    let Some(items) = analysis.dictionary_items.as_ref() else {
        return Ok(None);
    };
    let total_values = analysis
        .members
        .iter()
        .map(|member| member.values.len)
        .sum::<u64>();
    if items.len() < 2
        || items.len() > MAX_BLOCK_DICTIONARY_ITEMS
        || items.len() as u64 >= total_values.div_ceil(2)
    {
        return Ok(None);
    }

    let bits_per_value = analysis.members[0].values.bits_per_value;
    let item_stats = sequence_stats_for_values(items, bits_per_value);
    let repeated_items = (0..analysis.members.len())
        .map(|_| &item_stats)
        .collect::<Vec<_>>();
    let index_stats = analysis
        .members
        .iter()
        .map(|member| SequenceStats {
            bits_per_value: 32,
            len: member.values.len,
            first: Some(0),
            min: 0,
            max: items.len().saturating_sub(1) as u64,
            arithmetic_step: None,
            run_count: member.values.len,
            sample: Arc::from([]),
        })
        .collect::<Vec<_>>();
    let index_refs = index_stats.iter().collect::<Vec<_>>();
    let indices = leaf_candidate(&index_refs, field_params)?;
    let dictionary_items = leaf_candidate(&repeated_items, field_params)?;
    let has_payload = indices.has_payload || dictionary_items.has_payload;
    let payload_bytes = indices
        .payload_bytes
        .iter()
        .zip(&dictionary_items.payload_bytes)
        .map(|(indices, items)| {
            if has_payload {
                16_u64.saturating_add(*indices).saturating_add(*items)
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let encoding = ProtobufUtils21::dictionary(
        indices.encoding,
        dictionary_items.encoding,
        items.len() as u32,
    );
    let compressor = BlockDictionaryEncoder::try_new(
        bits_per_value,
        items.clone(),
        indices.compressor,
        dictionary_items.compressor,
    )?;
    Ok(Some(Candidate {
        wire_bytes: wire_bytes(&encoding, has_payload, &payload_bytes, cost),
        compressor: Box::new(compressor),
        encoding,
        has_payload,
        payload_bytes,
        transform_depth: 1,
        decode_cpu_rank: 3,
        stable_rank: 4,
    }))
}

fn flat_candidate(stats: &[&SequenceStats]) -> Result<Candidate> {
    let bits_per_value = stats[0].bits_per_value;
    if stats
        .iter()
        .any(|stats| stats.bits_per_value != bits_per_value)
    {
        return Err(Error::invalid_input(
            "Flat child candidates must have one value width",
        ));
    }
    let payload_bytes = stats
        .iter()
        .map(|stats| stats.raw_bytes())
        .collect::<Vec<_>>();
    let encoding = ProtobufUtils21::flat(bits_per_value, None);
    Ok(Candidate {
        wire_bytes: wire_bytes(&encoding, true, &payload_bytes, BlockCost::default()),
        compressor: Box::new(ValueEncoder::default()),
        encoding,
        has_payload: true,
        payload_bytes,
        transform_depth: 0,
        decode_cpu_rank: 0,
        stable_rank: 0,
    })
}

fn sequence_stats_for_values(values: &[u64], bits_per_value: u64) -> SequenceStats {
    let arithmetic_step = values
        .windows(2)
        .map(|pair| pair[1].checked_sub(pair[0]))
        .try_fold(None, |expected, step| match (expected, step) {
            (None, Some(step)) => Some(Some(step)),
            (Some(expected), Some(step)) if expected == step => Some(Some(expected)),
            _ => None,
        })
        .flatten();
    SequenceStats {
        bits_per_value,
        len: values.len() as u64,
        first: values.first().copied(),
        min: values.first().copied().unwrap_or(0),
        max: values.last().copied().unwrap_or(0),
        arithmetic_step: (values.len() >= 2).then_some(arithmetic_step).flatten(),
        run_count: values.len() as u64,
        sample: Arc::from([]),
    }
}

fn common_constant(stats: &[&SequenceStats]) -> Option<u64> {
    let value = stats.first()?.first?;
    stats
        .iter()
        .all(|stats| stats.len > 0 && stats.min == value && stats.max == value)
        .then_some(value)
}

fn common_range(stats: &[&SequenceStats]) -> Option<(u64, u64)> {
    let first = stats.first()?;
    let start = first.first?;
    let step = first.arithmetic_step?;
    if step == 0 {
        return None;
    }
    stats
        .iter()
        .all(|stats| {
            stats.len >= 2 && stats.first == Some(start) && stats.arithmetic_step == Some(step)
        })
        .then_some((start, step))
}

fn general_config(
    stats: &SequenceStats,
    field_params: &CompressionFieldParams,
) -> Result<Option<CompressionConfig>> {
    match field_params.compression.as_deref() {
        Some("none" | "fsst") => Ok(None),
        Some(name) => {
            let scheme = CompressionScheme::from_str(name)?;
            Ok(Some(CompressionConfig::new(
                scheme,
                field_params.compression_level,
            )))
        }
        None if stats.raw_bytes() > 32 * 1024 => {
            Ok(Some(CompressedBufferEncoder::default().compressor.config()))
        }
        None => Ok(None),
    }
}

fn estimate_general_bytes(stats: &SequenceStats, config: CompressionConfig) -> Result<u64> {
    if stats.raw_bytes() == 0 {
        return Ok(0);
    }
    if stats.sample.is_empty() {
        return Ok(u64::MAX);
    }
    let compressor = GeneralBufferCompressor::get_compressor(config)?;
    let mut compressed = Vec::new();
    compressor.compress(&stats.sample, &mut compressed)?;
    Ok(if stats.sample.len() as u64 == stats.raw_bytes() {
        compressed.len() as u64
    } else {
        (compressed.len() as u64)
            .saturating_mul(stats.raw_bytes())
            .div_ceil(stats.sample.len() as u64)
    })
}

fn wire_bytes(
    encoding: &CompressiveEncoding,
    has_payload: bool,
    payload_bytes: &[u64],
    cost: BlockCost,
) -> u64 {
    (encoding.encoded_len() as u64).saturating_add(
        payload_bytes
            .iter()
            .map(|payload_bytes| cost.payload_bytes(has_payload, *payload_bytes))
            .sum::<u64>(),
    )
}

fn is_flat(encoding: &CompressiveEncoding) -> bool {
    matches!(
        encoding.compression.as_ref(),
        Some(crate::format::pb21::compressive_encoding::Compression::Flat(_))
    )
}

fn normalize_offsets(mut data: FixedWidthDataBlock) -> Result<FixedWidthDataBlock> {
    if data.num_values == 0 {
        return Err(Error::invalid_input("Offset chunks cannot be empty"));
    }
    data.data = match data.bits_per_value {
        32 => {
            let values = checked_fixed_values::<u32>(&data, "Offset chunk")?;
            let base = values[0];
            if base == 0 {
                return Ok(data);
            }
            LanceBuffer::reinterpret_vec(
                values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        value.checked_sub(base).ok_or_else(|| {
                            Error::invalid_input(format!(
                                "Offset chunk decreases below its base at index {index}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        }
        64 => {
            let values = checked_fixed_values::<u64>(&data, "Offset chunk")?;
            let base = values[0];
            if base == 0 {
                return Ok(data);
            }
            LanceBuffer::reinterpret_vec(
                values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        value.checked_sub(base).ok_or_else(|| {
                            Error::invalid_input(format!(
                                "Offset chunk decreases below its base at index {index}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        }
        bits_per_value => {
            return Err(Error::invalid_input(format!(
                "Offset compression only supports 32 or 64-bit values, got {bits_per_value}"
            )));
        }
    };
    data.block_info = BlockInfo::default();
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compression::create_fixed_width_block_decompressor,
        format::pb21::compressive_encoding::Compression,
    };

    #[rstest::rstest]
    #[case(32)]
    #[case(64)]
    fn empty_sequences_share_one_encoding_contract(#[case] bits: u64) {
        let block = FixedWidthDataBlock {
            bits_per_value: bits,
            data: LanceBuffer::empty(),
            num_values: 0,
            block_info: BlockInfo::default(),
        };
        for normalize in [false, true] {
            let codec = select_block_codec(
                &block,
                std::slice::from_ref(&(0..0)),
                &Default::default(),
                BlockCost::default(),
                normalize,
            )
            .unwrap();
            let (payload, encoding) = codec.compress(block.clone()).unwrap();
            assert!(payload.is_none());
            assert_eq!(encoding, ConstantEncoder::new(bits, 0).encoding(0).unwrap());
            let decoder = SequenceDecoder::try_new(&encoding, bits, 0).unwrap();
            assert_eq!(decoder.metadata(), Some(SequenceMetadata::Empty));
            assert_eq!(decoder.decode(payload).unwrap().num_values(), 0);
            let block_decoder = create_fixed_width_block_decompressor(&encoding, bits).unwrap();
            assert_eq!(block_decoder.decompress(None, 0).unwrap().num_values(), 0);
            let error = block_decoder.decompress(None, 1).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("cardinality is zero"));
        }
        let codec = select_u64_sequence_codec(
            std::iter::empty(),
            &Default::default(),
            BlockCost::default(),
        )
        .unwrap();
        assert_eq!(
            codec.expected_encoding(),
            &ConstantEncoder::new(64, 0).encoding(0).unwrap()
        );
    }

    #[test]
    fn empty_and_non_empty_chunks_require_separate_descriptors() {
        let error = select_block_codec(
            &fixed_u64(&[1]),
            &[0..0, 0..1],
            &Default::default(),
            BlockCost::default(),
            false,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(
            error
                .to_string()
                .contains("cannot share one block descriptor")
        );
    }

    fn fixed_u32(values: &[u32]) -> FixedWidthDataBlock {
        FixedWidthDataBlock {
            bits_per_value: 32,
            data: LanceBuffer::reinterpret_vec(values.to_vec()),
            num_values: values.len() as u64,
            block_info: BlockInfo::default(),
        }
    }

    fn fixed_u64(values: &[u64]) -> FixedWidthDataBlock {
        FixedWidthDataBlock {
            bits_per_value: 64,
            data: LanceBuffer::reinterpret_vec(values.to_vec()),
            num_values: values.len() as u64,
            block_info: BlockInfo::default(),
        }
    }

    fn member(data: &FixedWidthDataBlock, range: Range<usize>) -> FixedWidthDataBlock {
        let bytes_per_value = (data.bits_per_value / 8) as usize;
        FixedWidthDataBlock {
            bits_per_value: data.bits_per_value,
            data: data
                .data
                .slice_with_length(range.start * bytes_per_value, range.len() * bytes_per_value),
            num_values: range.len() as u64,
            block_info: BlockInfo::default(),
        }
    }

    fn round_trip(
        codec: &SequenceBlockCodec,
        data: &FixedWidthDataBlock,
        range: Range<usize>,
    ) -> Vec<u64> {
        let num_values = range.len() as u64;
        let (payload, encoding) = codec.compress(member(data, range)).unwrap();
        let decoder =
            create_fixed_width_block_decompressor(&encoding, data.bits_per_value).unwrap();
        let decoded = decoder.decompress(payload, num_values).unwrap();
        let decoded = decoded.as_fixed_width().unwrap();
        match decoded.bits_per_value {
            32 => decoded
                .data
                .borrow_to_typed_slice::<u32>()
                .iter()
                .map(|value| u64::from(*value))
                .collect(),
            64 => decoded.data.borrow_to_typed_slice::<u64>().to_vec(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn selects_shared_metadata_codecs() {
        let constant = fixed_u32(&[5, 5, 5, 10, 10, 10]);
        let codec = select_block_codec(
            &constant,
            &[0..3, 3..6],
            &CompressionFieldParams::default(),
            BlockCost::new(0, 1),
            true,
        )
        .unwrap();
        assert!(matches!(
            codec.expected_encoding().compression,
            Some(Compression::Constant(_))
        ));
        assert!(!codec.has_payload());
        assert_eq!(round_trip(&codec, &constant, 3..6), vec![0, 0, 0]);

        let range = fixed_u32(&[5, 7, 9, 11, 10, 12, 14, 16]);
        let codec = select_block_codec(
            &range,
            &[0..4, 4..8],
            &CompressionFieldParams::default(),
            BlockCost::new(0, 1),
            true,
        )
        .unwrap();
        assert!(matches!(
            codec.expected_encoding().compression,
            Some(Compression::Range(_))
        ));
        assert_eq!(round_trip(&codec, &range, 4..8), vec![0, 2, 4, 6]);
    }

    #[test]
    fn selects_delta_and_round_trips() {
        let data = fixed_u32(&[0, 2, 5, 9, 14]);
        let codec = select_block_codec(
            &data,
            &[0..5],
            &CompressionFieldParams {
                compression: Some("none".to_string()),
                ..Default::default()
            },
            BlockCost::new(0, 1),
            true,
        )
        .unwrap();
        assert!(matches!(
            codec.expected_encoding().compression,
            Some(Compression::Delta(_))
        ));
        assert_eq!(round_trip(&codec, &data, 0..5), vec![0, 2, 5, 9, 14]);
    }

    #[test]
    fn absolute_sequences_preserve_bases_and_accept_non_monotonic_counts() {
        let sequences = [
            vec![9, 9, 9],
            vec![13, 20, 27, 34],
            (0..256).map(|i| 1_000_000 + i * i).collect(),
            (0..512).map(|i| [1, 1000, 2, 9000][i % 4]).collect(),
            (0..32)
                .flat_map(|i| std::iter::repeat_n(if i % 2 == 0 { 300 } else { 1 }, 16))
                .collect(),
        ];
        for values in sequences {
            let data = fixed_u64(&values);
            let codec = select_block_codec(
                &data,
                &[0..values.len()],
                &CompressionFieldParams::default(),
                BlockCost::new(16, 8),
                false,
            )
            .unwrap();
            let decoded = round_trip(&codec, &data, 0..values.len());
            assert_eq!(decoded, values);
        }
    }

    #[test]
    fn selects_rle_and_dictionary_candidates() {
        let rle_member = (0..32_u64)
            .flat_map(|value| std::iter::repeat_n(value, 8))
            .collect::<Vec<_>>();
        let mut rle_values = rle_member.clone();
        rle_values.extend_from_slice(&rle_member);
        let rle_data = fixed_u64(&rle_values);
        let codec = select_block_codec(
            &rle_data,
            &[0..rle_member.len(), rle_member.len()..rle_values.len()],
            &CompressionFieldParams::default(),
            BlockCost::new(0, 1),
            true,
        )
        .unwrap();
        assert!(matches!(
            codec.expected_encoding().compression,
            Some(Compression::Rle(_))
        ));
        assert_eq!(
            round_trip(&codec, &rle_data, 0..rle_member.len()),
            rle_member
        );

        let dictionary_values = [0_u64, 1000, 5000, 9000]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, 128))
            .collect::<Vec<_>>();
        let dictionary_data = fixed_u64(&dictionary_values);
        let analysis =
            analyze_family(&dictionary_data, &[0..dictionary_values.len()], false, true).unwrap();
        let codec = dictionary_candidate(
            &analysis,
            &CompressionFieldParams {
                compression: Some("none".to_string()),
                ..Default::default()
            },
            BlockCost::new(0, 1),
        )
        .unwrap()
        .unwrap()
        .finish(true);
        assert!(matches!(
            codec.expected_encoding().compression,
            Some(Compression::Dictionary(_))
        ));
        assert_eq!(
            round_trip(&codec, &dictionary_data, 0..512),
            dictionary_values
        );

        let dictionary_values = dictionary_values
            .into_iter()
            .map(|value| value as u32)
            .collect::<Vec<_>>();
        let dictionary_data = fixed_u32(&dictionary_values);
        let analysis =
            analyze_family(&dictionary_data, &[0..dictionary_values.len()], false, true).unwrap();
        let codec = dictionary_candidate(
            &analysis,
            &CompressionFieldParams {
                compression: Some("none".to_string()),
                ..Default::default()
            },
            BlockCost::new(0, 1),
        )
        .unwrap()
        .unwrap()
        .finish(true);
        assert!(matches!(
            codec.expected_encoding().compression,
            Some(Compression::Dictionary(_))
        ));
        assert_eq!(
            round_trip(&codec, &dictionary_data, 0..512),
            dictionary_values
                .into_iter()
                .map(u64::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn rejects_decreasing_member() {
        let error = select_block_codec(
            &fixed_u32(&[0, 2, 1]),
            &[0..3],
            &CompressionFieldParams::default(),
            BlockCost::new(0, 1),
            true,
        )
        .unwrap_err();
        assert!(error.to_string().contains("non-decreasing"));
    }
}
