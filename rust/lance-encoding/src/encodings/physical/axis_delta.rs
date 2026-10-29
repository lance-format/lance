// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! First-order differential coding along a tensor axis, with bounded per-page selection.

#[cfg(target_arch = "x86_64")]
mod x86_64;

use std::time::Instant;

use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType;
use lance_core::{Error, Result, datatypes::Field};

use super::{
    block::{CompressionConfig, CompressionScheme, GeneralBufferCompressor},
    cost::{BlockCost, wire_bytes},
    general::FixedWidthGeneralPerValueCompressor,
    value::{ValueDecompressor, ValueEncoder},
};
use crate::{
    buffer::LanceBuffer,
    compression::{FixedPerValueDecompressor, try_raw_per_value},
    compression_config::{AxisDeltaMode, CompressionFieldParams},
    data::{DataBlock, FixedWidthDataBlock},
    encodings::logical::primitive::fullzip::{PerValueCompressor, PerValueDataBlock},
    format::{
        EncodingUtils,
        metadata::{AxisDelta, CompressiveEncoding, compressive_encoding::Compression},
    },
};

const META_KEY: &str = "lance-encoding:axis-delta";
// Whole frames preserve predictor resets and the downstream codec's dictionary scope.
const MAX_SAMPLE_ROWS: usize = 4;
const MAX_SAMPLE_BYTES: usize = 4 * 1024 * 1024;
// Compression and prediction add decoding work; marginal savings do not justify it.
const MIN_SAVING_PERCENT: u128 = 5;

#[derive(Debug, Clone, Copy)]
struct AxisLayout {
    axis_length: usize,
    inner_stride: usize,
    elements: usize,
    element_bytes: usize,
}

impl AxisLayout {
    fn from_description(description: &AxisDelta) -> Result<(Self, ValueDecompressor)> {
        let values = description
            .values
            .as_deref()
            .ok_or_else(|| Error::invalid_input("AxisDelta is missing its values encoding"))?;
        let Some(Compression::FixedSizeList(list)) = &values.compression else {
            return Err(Error::invalid_input(
                "AxisDelta requires a plain fixed-size-list child",
            ));
        };
        let mut encoding = values;
        let mut elements = 1_usize;
        while let Some(Compression::FixedSizeList(list)) = &encoding.compression {
            let dimension = usize::try_from(list.items_per_value)
                .map_err(|_| Error::invalid_input("AxisDelta dimension exceeds usize"))?;
            if list.has_validity || dimension == 0 {
                return Err(Error::invalid_input(
                    "AxisDelta requires positive dimensions without child validity",
                ));
            }
            elements = elements
                .checked_mul(dimension)
                .ok_or_else(|| Error::invalid_input("AxisDelta element count overflows usize"))?;
            encoding = list
                .values
                .as_deref()
                .ok_or_else(|| Error::invalid_input("AxisDelta list child is missing"))?;
        }
        let Some(Compression::Flat(flat)) = &encoding.compression else {
            return Err(Error::invalid_input(
                "AxisDelta requires plain integer elements",
            ));
        };
        if flat.data.is_some() || !matches!(flat.bits_per_value, 8 | 16 | 32 | 64) {
            return Err(Error::invalid_input(
                "AxisDelta requires plain 8/16/32/64-bit elements",
            ));
        }
        let axis_length = usize::try_from(description.axis_length)
            .map_err(|_| Error::invalid_input("AxisDelta axis length exceeds usize"))?;
        let inner_stride = usize::try_from(description.inner_stride)
            .map_err(|_| Error::invalid_input("AxisDelta stride exceeds usize"))?;
        let span = axis_length
            .checked_mul(inner_stride)
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::invalid_input("AxisDelta span is zero or overflows usize"))?;
        if !elements.is_multiple_of(span) {
            return Err(Error::invalid_input(format!(
                "AxisDelta span {span} does not divide {elements} tensor elements"
            )));
        }
        let element_bytes = flat.bits_per_value as usize / 8;
        elements
            .checked_mul(element_bytes)
            .filter(|&n| n <= isize::MAX as usize)
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| Error::invalid_input("AxisDelta tensor byte size overflows"))?;
        Ok((
            Self {
                axis_length,
                inner_stride,
                elements,
                element_bytes,
            },
            ValueDecompressor::from_fsl(list)?,
        ))
    }

    fn row_bytes(self) -> usize {
        self.elements * self.element_bytes
    }

    fn transform_values<const DECODE: bool>(self, bytes: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        if self.axis_length * self.inner_stride * self.element_bytes >= 16
            && std::is_x86_feature_detected!("avx2")
        {
            let stride_bytes = self.inner_stride * self.element_bytes;
            let span_bytes = self.axis_length * stride_bytes;
            // SAFETY: Runtime detection guarantees AVX2. Layout validation and
            // the caller's buffer check guarantee complete, element-aligned spans.
            unsafe {
                x86_64::transform::<DECODE>(bytes, span_bytes, stride_bytes, self.element_bytes);
            }
            return;
        }
        self.transform_scalar::<DECODE>(bytes);
    }

    fn transform_scalar<const DECODE: bool>(self, bytes: &mut [u8]) {
        macro_rules! transform {
            ($ty:ty) => {{
                const WIDTH: usize = std::mem::size_of::<$ty>();
                let stride = self.inner_stride * WIDTH;
                for span in bytes.chunks_exact_mut(self.axis_length * stride) {
                    let mut update = |offset: usize| {
                        let mut current = [0; WIDTH];
                        let mut previous = [0; WIDTH];
                        current.copy_from_slice(&span[offset..offset + WIDTH]);
                        previous.copy_from_slice(&span[offset - stride..offset - stride + WIDTH]);
                        let current = <$ty>::from_le_bytes(current);
                        let previous = <$ty>::from_le_bytes(previous);
                        let result = if DECODE {
                            current.wrapping_add(previous)
                        } else {
                            current.wrapping_sub(previous)
                        };
                        span[offset..offset + WIDTH].copy_from_slice(&result.to_le_bytes());
                    };
                    // Reverse encoding retains original predecessors in the same buffer.
                    if DECODE {
                        for i in self.inner_stride..self.axis_length * self.inner_stride {
                            update(i * WIDTH);
                        }
                    } else {
                        for i in (self.inner_stride..self.axis_length * self.inner_stride).rev() {
                            update(i * WIDTH);
                        }
                    }
                }
            }};
        }
        match self.element_bytes {
            1 => transform!(u8),
            2 => transform!(u16),
            4 => transform!(u32),
            8 => transform!(u64),
            _ => unreachable!("validated element width"),
        }
    }

    fn transform(self, data: &mut FixedWidthDataBlock, decode: bool) -> Result<()> {
        let rows = usize::try_from(data.num_values)
            .map_err(|_| Error::invalid_input("AxisDelta row count exceeds usize"))?;
        if data.bits_per_value != self.row_bytes() as u64 * 8
            || rows.checked_mul(self.row_bytes()) != Some(data.data.len())
        {
            return Err(Error::invalid_input(
                "AxisDelta buffer does not match its tensor shape",
            ));
        }
        let mut output = data.data.as_ref().to_vec();
        if decode {
            self.transform_values::<true>(&mut output);
        } else {
            self.transform_values::<false>(&mut output);
        }
        data.data = LanceBuffer::from(output);
        Ok(())
    }
}

fn request(field: &Field, params: &CompressionFieldParams) -> Result<AxisDeltaMode> {
    let Some(raw) = field.metadata.get(META_KEY) else {
        return Ok(params
            .axis_delta
            .unwrap_or(AxisDeltaMode::Auto { axis: None }));
    };
    if raw == "off" {
        return Ok(AxisDeltaMode::Off);
    }
    if raw == "auto" {
        return Ok(AxisDeltaMode::Auto { axis: None });
    }
    if let Some((mode, axis)) = raw.split_once(':')
        && let Ok(axis) = axis.parse::<usize>()
    {
        match mode {
            "auto" => return Ok(AxisDeltaMode::Auto { axis: Some(axis) }),
            "on" => return Ok(AxisDeltaMode::On { axis }),
            _ => {}
        }
    }
    Err(Error::invalid_input(format!(
        "Invalid {META_KEY} {raw:?}; expected off, auto, auto:<axis>, or on:<axis>"
    )))
}

fn shape(field: &Field, axis: Option<usize>) -> Result<(usize, usize)> {
    let mut shape = Vec::new();
    let data_type = field.data_type();
    let mut leaf = &data_type;
    while let DataType::FixedSizeList(child, dimension) = leaf {
        shape.push(
            usize::try_from(*dimension)
                .map_err(|_| Error::invalid_input("AxisDelta requires positive dimensions"))?,
        );
        leaf = child.data_type();
    }
    if !matches!(
        leaf,
        DataType::Int8
            | DataType::UInt8
            | DataType::Int16
            | DataType::UInt16
            | DataType::Int32
            | DataType::UInt32
            | DataType::Int64
            | DataType::UInt64
    ) || shape.is_empty()
    {
        return Err(Error::not_supported(
            "AxisDelta requires a fixed-size tensor of 8/16/32/64-bit integers",
        ));
    }
    let product = |dimensions: &[usize]| -> Result<usize> {
        dimensions.iter().try_fold(1_usize, |n, &d| {
            n.checked_mul(d)
                .filter(|&n| n > 0)
                .ok_or_else(|| Error::invalid_input("AxisDelta shape is zero or overflows usize"))
        })
    };
    let elements = product(&shape)?;
    let mut physical_axis = axis;
    if field.extension_name() == Some("arrow.fixed_shape_tensor") {
        let metadata = field
            .metadata
            .get("ARROW:extension:metadata")
            .ok_or_else(|| Error::invalid_input("Tensor shape metadata is missing"))?;
        let metadata: serde_json::Value = serde_json::from_str(metadata)
            .map_err(|err| Error::invalid_input(format!("Invalid tensor metadata: {err}")))?;
        let dimensions = |name: &str| -> Result<Vec<usize>> {
            metadata
                .get(name)
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    Error::invalid_input(format!("Tensor {name} must be an integer array"))
                })?
                .iter()
                .map(|n| {
                    n.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| {
                            Error::invalid_input(format!("Invalid tensor {name} value {n}"))
                        })
                })
                .collect()
        };
        shape = dimensions("shape")?;
        if shape.is_empty() || product(&shape)? != elements {
            return Err(Error::invalid_input(
                "Tensor shape does not match fixed-size-list length",
            ));
        }
        if metadata.get("permutation").is_some() {
            let permutation = dimensions("permutation")?;
            let mut sorted = permutation.clone();
            sorted.sort_unstable();
            if sorted != (0..shape.len()).collect::<Vec<_>>() {
                return Err(Error::invalid_input("Invalid tensor axis permutation"));
            }
            if let Some(axis) = axis {
                physical_axis = Some(*permutation.get(axis).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "Axis {axis:?} exceeds tensor rank {}",
                        shape.len()
                    ))
                })?);
            }
        }
    }
    // A fixed candidate bounds selection cost independently of tensor rank.
    let physical_axis = physical_axis.unwrap_or_else(|| {
        shape
            .iter()
            .enumerate()
            .max_by_key(|(_, length)| *length)
            .map_or(0, |(axis, _)| axis)
    });
    let length = *shape.get(physical_axis).ok_or_else(|| {
        Error::invalid_input(format!("Axis {axis:?} exceeds tensor rank {}", shape.len()))
    })?;
    Ok((length, product(&shape[physical_axis + 1..])?))
}

/// Select raw, compressed, or axis-predicted integer tensors for the 2.3 per-value writer.
/// The default samples the longest physical axis when LZ4 or Zstd is configured.
/// Unsupported implicit inputs retain their existing encoding.
/// Metadata `lance-encoding:axis-delta` overrides the typed column/type setting.
pub fn try_tensor_per_value(
    field: &Field,
    data: &DataBlock,
    params: &CompressionFieldParams,
) -> Result<Option<Box<dyn PerValueCompressor>>> {
    let implicit = params.axis_delta.is_none() && !field.metadata.contains_key(META_KEY);
    let mode = request(field, params)?;
    let compression = match params.compression.as_deref() {
        Some("lz4") => Some(CompressionConfig::new(
            CompressionScheme::Lz4,
            params.compression_level,
        )),
        Some("zstd") => Some(CompressionConfig::new(
            CompressionScheme::Zstd,
            params.compression_level,
        )),
        None | Some("none") => None,
        Some(_) if !matches!(mode, AxisDeltaMode::On { .. }) => return Ok(None),
        Some(name) => {
            return Err(Error::invalid_input(format!(
                "Unsupported AxisDelta compression {name:?}"
            )));
        }
    };
    let (axis, auto) = match mode {
        AxisDeltaMode::Off => {
            // Off still honors an explicitly configured codec for integer tensors.
            if !matches!(data, DataBlock::FixedSizeList(_)) {
                return Ok(None);
            }
            let data_type = field.data_type();
            let mut leaf = &data_type;
            while let DataType::FixedSizeList(child, _) = leaf {
                leaf = child.data_type();
            }
            return Ok(compression
                .filter(|_| leaf.is_integer())
                .map(|compression| {
                    Box::new(FixedWidthGeneralPerValueCompressor::new(compression)) as _
                }));
        }
        AxisDeltaMode::Auto { axis } => (axis, true),
        AxisDeltaMode::On { axis } => (Some(axis), false),
    };
    if implicit && compression.is_none() {
        return Ok(None);
    }
    if auto && !matches!(data, DataBlock::FixedSizeList(_)) {
        return Ok(None);
    }
    let (axis_length, inner_stride) = match shape(field, axis) {
        Err(_) if implicit => return Ok(try_raw_per_value(data)),
        Err(Error::NotSupported { .. }) if auto => return Ok(try_raw_per_value(data)),
        result => result?,
    };
    if auto && (compression.is_none() || axis_length == 1) {
        return Ok(try_raw_per_value(data));
    }
    // ValueEncoder owns nested FSL shape and validity. A predictor cannot operate
    // on its packed validity bytes, so reject such children before flattening.
    let mut child = data;
    while let DataBlock::FixedSizeList(list) = child {
        child = list.child.as_ref();
    }
    if !matches!(child, DataBlock::FixedWidth(_)) {
        if auto {
            return Ok(try_raw_per_value(data));
        }
        return Err(Error::not_supported(
            "AxisDelta requires tensor elements without nulls",
        ));
    }
    Ok(Some(Box::new(AxisDeltaCompressor {
        axis_length,
        inner_stride,
        auto,
        compression,
    })))
}

#[derive(Debug)]
struct AxisDeltaCompressor {
    axis_length: usize,
    inner_stride: usize,
    auto: bool,
    compression: Option<CompressionConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Selection {
    Raw,
    Compressed,
    Delta,
}

impl AxisDeltaCompressor {
    fn select(
        &self,
        data: &FixedWidthDataBlock,
        layout: AxisLayout,
        baseline: &CompressiveEncoding,
        predicted: &CompressiveEncoding,
        validity: Option<&BooleanBuffer>,
    ) -> Result<Selection> {
        let start = log::log_enabled!(target: "lance_encoding::axis_delta", log::Level::Debug)
            .then(Instant::now);
        let rows = usize::try_from(data.num_values)
            .map_err(|_| Error::invalid_input("AxisDelta rows exceed usize"))?;
        if validity.is_some_and(|v| v.len() != rows) {
            return Err(Error::invalid_input(
                "AxisDelta sample validity length does not match rows",
            ));
        }
        let valid_rows = validity.map_or(rows, BooleanBuffer::count_set_bits);
        let count = MAX_SAMPLE_ROWS
            .min(MAX_SAMPLE_BYTES / layout.row_bytes())
            .min(valid_rows);
        if count == 0 {
            return Ok(Selection::Raw);
        }
        let config = self
            .compression
            .ok_or_else(|| Error::internal("Automatic AxisDelta requires general compression"))?;
        let compressor = GeneralBufferCompressor::get_compressor(config)?;
        let mut baseline_bytes = 0_u64;
        let mut delta_bytes = 0_u64;
        let mut compressed = Vec::new();
        let mut transformed = Vec::with_capacity(layout.row_bytes());
        let mut sample = 0;
        for (ordinal, row) in (0..rows)
            .filter(|&r| validity.is_none_or(|v| v.value(r)))
            .enumerate()
        {
            let wanted = if count == 1 {
                0
            } else {
                sample * (valid_rows - 1) / (count - 1)
            };
            if ordinal != wanted {
                continue;
            }
            let input =
                &data.data.as_ref()[row * layout.row_bytes()..(row + 1) * layout.row_bytes()];
            compressed.clear();
            compressor.compress(input, &mut compressed)?;
            baseline_bytes += compressed.len() as u64;
            transformed.clear();
            transformed.extend_from_slice(input);
            layout.transform_values::<false>(&mut transformed);
            compressed.clear();
            compressor.compress(&transformed, &mut compressed)?;
            delta_bytes += compressed.len() as u64;
            sample += 1;
            if sample == count {
                break;
            }
        }
        let projected = |bytes: u64| {
            bytes
                .saturating_mul(valid_rows as u64)
                .div_ceil(count as u64)
        };
        let raw_wire = wire_bytes(
            baseline,
            true,
            &[(valid_rows as u64).saturating_mul(layout.row_bytes() as u64)],
            BlockCost::default(),
        );
        // General per-value frames use a 64-bit length for each visible value.
        // Raw fixed-width values need no length. Structural control words are
        // common to the candidates and are excluded from this estimate.
        let addressing_bytes = (valid_rows as u64).saturating_mul(8);
        let baseline_wire = wire_bytes(
            &EncodingUtils::wrapped(config, baseline.clone())?,
            true,
            &[projected(baseline_bytes).saturating_add(addressing_bytes)],
            BlockCost::default(),
        );
        let delta_wire = wire_bytes(
            &EncodingUtils::wrapped(config, predicted.clone())?,
            true,
            &[projected(delta_bytes).saturating_add(addressing_bytes)],
            BlockCost::default(),
        );
        let saves_enough = |candidate: u64, baseline: u64| {
            (candidate as u128) * 100 <= (baseline as u128) * (100 - MIN_SAVING_PERCENT)
        };
        let mut selected = Selection::Raw;
        let mut best_wire = raw_wire;
        if saves_enough(baseline_wire, best_wire) {
            selected = Selection::Compressed;
            best_wire = baseline_wire;
        }
        if saves_enough(delta_wire, best_wire) {
            selected = Selection::Delta;
        }
        if let Some(start) = start {
            log::debug!(target: "lance_encoding::axis_delta", "rows={} bytes={} raw={} baseline={} delta={} selected={} encoding={:?} nanos={}", count, count * layout.row_bytes(), raw_wire, baseline_wire, delta_wire, selected == Selection::Delta, selected, start.elapsed().as_nanos());
        }
        Ok(selected)
    }
}

impl PerValueCompressor for AxisDeltaCompressor {
    fn needs_row_validity(&self) -> bool {
        self.auto
    }
    fn compress(&self, data: DataBlock) -> Result<(PerValueDataBlock, CompressiveEncoding)> {
        self.compress_with_validity(data, None)
    }
    fn compress_with_validity(
        &self,
        data: DataBlock,
        validity: Option<&BooleanBuffer>,
    ) -> Result<(PerValueDataBlock, CompressiveEncoding)> {
        let (data, original) = PerValueCompressor::compress(&ValueEncoder::default(), data)?;
        let PerValueDataBlock::Fixed(mut data) = data else {
            return Err(Error::invalid_input(
                "AxisDelta requires fixed-width tensor rows",
            ));
        };
        let description = AxisDelta {
            values: Some(Box::new(original.clone())),
            axis_length: self.axis_length as u64,
            inner_stride: self.inner_stride as u64,
        };
        let (layout, _) = AxisLayout::from_description(&description)?;
        let predicted = CompressiveEncoding {
            compression: Some(Compression::AxisDelta(Box::new(description))),
        };
        let selected = if self.auto {
            self.select(&data, layout, &original, &predicted, validity)?
        } else {
            Selection::Delta
        };
        let description = match selected {
            Selection::Raw => return Ok((PerValueDataBlock::Fixed(data), original)),
            Selection::Compressed => original,
            Selection::Delta => {
                layout.transform(&mut data, false)?;
                predicted
            }
        };
        if let Some(compression) = self.compression {
            FixedWidthGeneralPerValueCompressor::compress_encoded(data, description, compression)
        } else {
            Ok((PerValueDataBlock::Fixed(data), description))
        }
    }
}

#[derive(Debug)]
pub struct AxisDeltaDecompressor {
    layout: AxisLayout,
    child: ValueDecompressor,
}
impl AxisDeltaDecompressor {
    pub(crate) fn try_new(description: &AxisDelta) -> Result<Self> {
        let (layout, child) = AxisLayout::from_description(description)?;
        Ok(Self { layout, child })
    }
}
impl FixedPerValueDecompressor for AxisDeltaDecompressor {
    fn decompress(&self, mut data: FixedWidthDataBlock, num_values: u64) -> Result<DataBlock> {
        if data.num_values != num_values {
            return Err(Error::invalid_input("AxisDelta row count mismatch"));
        }
        self.layout.transform(&mut data, true)?;
        self.child.decompress(data, num_values)
    }
    fn bits_per_value(&self) -> u64 {
        self.child.bits_per_value()
    }
    fn decoded_size_bytes(&self, num_values: u64) -> Option<u64> {
        self.child.decoded_size_bytes(num_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compression::{DecompressionStrategy, DefaultDecompressionStrategy},
        data::BlockInfo,
    };
    use arrow_array::{Array, ArrayRef, FixedSizeListArray, UInt16Array};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use std::{collections::HashMap, sync::Arc};

    fn descriptor(bits: u64, shape: &[u64], axis: usize) -> AxisDelta {
        let mut values = EncodingUtils::flat(bits, None);
        for &dimension in shape.iter().rev() {
            values = EncodingUtils::fsl(dimension, false, values);
        }
        AxisDelta {
            values: Some(Box::new(values)),
            axis_length: shape[axis],
            inner_stride: shape[axis + 1..].iter().product(),
        }
    }

    #[rstest::rstest]
    fn reference_and_inverse(#[values(8, 16, 32, 64)] bits: u64, #[values(0, 1, 2)] axis: usize) {
        let shape = [2, 3, 5];
        let (layout, _) = AxisLayout::from_description(&descriptor(bits, &shape, axis)).unwrap();
        let mut rng = StdRng::seed_from_u64(193);
        let input: Vec<u8> = (0..2 * layout.row_bytes()).map(|_| rng.random()).collect();
        let mut data = FixedWidthDataBlock {
            data: LanceBuffer::from(input.clone()),
            bits_per_value: layout.row_bytes() as u64 * 8,
            num_values: 2,
            block_info: BlockInfo::new(),
        };
        layout.transform(&mut data, false).unwrap();
        let read = |data: &[u8], offset: usize| {
            let mut bytes = [0; 8];
            bytes[..layout.element_bytes]
                .copy_from_slice(&data[offset..offset + layout.element_bytes]);
            u64::from_le_bytes(bytes)
        };
        let mask = u64::MAX >> (64 - bits);
        for row in 0..2 {
            for i in 0..layout.elements {
                let offset = row * layout.row_bytes() + i * layout.element_bytes;
                let previous = if i / layout.inner_stride % layout.axis_length == 0 {
                    0
                } else {
                    read(&input, offset - layout.inner_stride * layout.element_bytes)
                };
                assert_eq!(
                    read(&data.data, offset),
                    read(&input, offset).wrapping_sub(previous) & mask
                );
            }
        }
        layout.transform(&mut data, true).unwrap();
        assert_eq!(data.data.as_ref(), input);
    }

    #[cfg(target_arch = "x86_64")]
    #[rstest::rstest]
    fn simd_matches_scalar_at_boundaries(
        #[values(1, 2, 4, 8)] element_bytes: usize,
        #[values(1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 32, 33)] inner_stride: usize,
    ) {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = StdRng::seed_from_u64(731);
        for axis_length in [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 65] {
            let layout = AxisLayout {
                axis_length,
                inner_stride,
                elements: 2 * axis_length * inner_stride,
                element_bytes,
            };
            let span_bytes = axis_length * inner_stride * element_bytes;
            // Multiple spans and tensors catch predictor carry leaking across
            // boundaries. Sentinels detect writes outside unaligned slices.
            let input: Vec<u8> = (0..3 * layout.row_bytes()).map(|_| rng.random()).collect();
            let mut expected = input.clone();
            layout.transform_scalar::<false>(&mut expected);
            for offset in [0, 1, 5, 31] {
                let end = offset + input.len();
                let mut actual = vec![0xa7; end + 32];
                actual[offset..end].copy_from_slice(&input);
                // SAFETY: AVX2 is checked above and the generated buffer/layout
                // satisfy the complete-span contract, including short spans.
                unsafe {
                    x86_64::transform::<false>(
                        &mut actual[offset..end],
                        span_bytes,
                        inner_stride * element_bytes,
                        element_bytes,
                    );
                }
                assert_eq!(&actual[offset..end], expected);
                // Decode independently generated scalar bytes with SIMD.
                unsafe {
                    x86_64::transform::<true>(
                        &mut actual[offset..end],
                        span_bytes,
                        inner_stride * element_bytes,
                        element_bytes,
                    );
                }
                assert_eq!(&actual[offset..end], input);
                assert!(actual[..offset].iter().all(|b| *b == 0xa7));
                assert!(actual[end..].iter().all(|b| *b == 0xa7));
            }
        }
    }

    fn fixture(rows: usize, width: usize, noise: bool) -> (Field, DataBlock) {
        let mut rng = StdRng::seed_from_u64(137);
        let values = (0..rows * width)
            .map(|i| {
                if noise {
                    rng.random::<u16>()
                } else {
                    (i as u16).wrapping_mul(37)
                }
            })
            .collect::<Vec<_>>();
        let array: ArrayRef = Arc::new(FixedSizeListArray::new(
            Arc::new(arrow_schema::Field::new("item", DataType::UInt16, false)),
            width as i32,
            Arc::new(UInt16Array::from(values)),
            None,
        ));
        let field = Field::try_from(&arrow_schema::Field::new(
            "tensor",
            array.data_type().clone(),
            false,
        ))
        .unwrap();
        (field, DataBlock::from_array(array))
    }

    fn has_delta(encoding: &CompressiveEncoding) -> bool {
        match &encoding.compression {
            Some(Compression::AxisDelta(_)) => true,
            Some(Compression::General(g)) => has_delta(g.values.as_ref().unwrap()),
            _ => false,
        }
    }

    fn assert_raw(
        data: &DataBlock,
        description: &CompressiveEncoding,
        encoded: &FixedWidthDataBlock,
    ) {
        let (raw, expected) = ValueEncoder::default().compress(data.clone()).unwrap();
        let PerValueDataBlock::Fixed(raw) = raw else {
            panic!("expected raw rows")
        };
        assert_eq!(*description, expected);
        assert_eq!(encoded.data.as_ref(), raw.data.as_ref());
        assert_eq!(encoded.bits_per_value, raw.bits_per_value);
        assert_eq!(encoded.num_values, raw.num_values);
    }

    #[rstest::rstest]
    fn direct_compression_wins(#[values("lz4", "zstd")] codec: &str) {
        let (field, mut data) = fixture(9, 4096, true);
        let DataBlock::FixedSizeList(list) = &mut data else {
            panic!()
        };
        let DataBlock::FixedWidth(values) = list.child.as_mut() else {
            panic!()
        };
        // A short random pattern repeats within each row. Direct compression
        // already captures that repetition; prediction adds no useful savings.
        let pattern = values.data.as_ref()[..128].to_vec();
        values.data = LanceBuffer::from(pattern.repeat(values.data.len() / pattern.len()));
        let params = CompressionFieldParams {
            compression: Some(codec.into()),
            ..Default::default()
        };
        let (encoded, description) = try_tensor_per_value(&field, &data, &params)
            .unwrap()
            .unwrap()
            .compress(data.clone())
            .unwrap();
        assert!(!has_delta(&description));
        let PerValueDataBlock::Variable(encoded) = encoded else {
            panic!("expected compression")
        };
        let decoded = DefaultDecompressionStrategy::default()
            .create_variable_per_value_decompressor(&description)
            .unwrap()
            .decompress(encoded)
            .unwrap();
        assert_eq!(
            decoded.into_arrow(field.data_type(), true).unwrap(),
            data.into_arrow(field.data_type(), true).unwrap()
        );
    }

    #[rstest::rstest]
    fn automatic_selection(
        #[values("lz4", "zstd")] codec: &str,
        #[values(false, true)] noise: bool,
    ) {
        let (field, data) = fixture(9, 4096, noise);
        for (mode, expected) in [
            (AxisDeltaMode::Auto { axis: Some(0) }, !noise),
            (AxisDeltaMode::Auto { axis: None }, !noise),
            (AxisDeltaMode::On { axis: 0 }, true),
            (AxisDeltaMode::Off, false),
        ] {
            let params = CompressionFieldParams {
                axis_delta: Some(mode),
                compression: Some(codec.into()),
                ..Default::default()
            };
            let compressor = try_tensor_per_value(&field, &data, &params)
                .unwrap()
                .unwrap();
            let (encoded, description) = compressor.compress(data.clone()).unwrap();
            assert_eq!(has_delta(&description), expected);
            assert_eq!(
                matches!(encoded, PerValueDataBlock::Fixed(_)),
                noise && matches!(mode, AxisDeltaMode::Auto { .. }),
                "Auto must bypass general compression when both sampled candidates lose"
            );
            let strategy = DefaultDecompressionStrategy::default();
            let decoded = match encoded {
                PerValueDataBlock::Fixed(encoded) => {
                    assert!(noise && matches!(mode, AxisDeltaMode::Auto { .. }));
                    assert_raw(&data, &description, &encoded);
                    strategy
                        .create_fixed_per_value_decompressor(&description)
                        .unwrap()
                        .decompress(encoded, 9)
                        .unwrap()
                }
                PerValueDataBlock::Variable(encoded) => strategy
                    .create_variable_per_value_decompressor(&description)
                    .unwrap()
                    .decompress(encoded)
                    .unwrap(),
            };
            assert_eq!(
                decoded.into_arrow(field.data_type(), true).unwrap(),
                data.clone().into_arrow(field.data_type(), true).unwrap()
            );
        }
    }

    #[test]
    fn sampling_excludes_hidden_null_values() {
        let (field, data) = fixture(9, 4096, true);
        let params = CompressionFieldParams {
            axis_delta: Some(AxisDeltaMode::Auto { axis: Some(0) }),
            compression: Some("zstd".into()),
            ..Default::default()
        };
        let DataBlock::FixedSizeList(mut list) = data else {
            panic!()
        };
        let DataBlock::FixedWidth(mut values) = *list.child else {
            panic!()
        };
        let mut bytes = values.data.as_ref().to_vec();
        // The ordinary four sample positions contain hidden smooth values.
        for row in [0, 2, 5, 8] {
            for (i, value) in bytes[row * 8192..(row + 1) * 8192]
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .enumerate()
            {
                value.copy_from_slice(&(i as u16).wrapping_mul(37).to_le_bytes());
            }
        }
        values.data = LanceBuffer::from(bytes);
        list.child = Box::new(DataBlock::FixedWidth(values));
        let data = DataBlock::FixedSizeList(list);
        let compressor = try_tensor_per_value(&field, &data, &params)
            .unwrap()
            .unwrap();
        let (_, unmasked) = compressor.compress(data.clone()).unwrap();
        assert!(
            has_delta(&unmasked),
            "hidden smooth rows would bias this sample"
        );
        let validity = BooleanBuffer::from_iter((0..9).map(|i| ![0, 2, 5, 8].contains(&i)));
        for validity in [validity, BooleanBuffer::new_unset(9)] {
            let (encoded, description) = compressor
                .compress_with_validity(data.clone(), Some(&validity))
                .unwrap();
            let PerValueDataBlock::Fixed(encoded) = encoded else {
                panic!("expected raw rows")
            };
            assert_raw(&data, &description, &encoded);
        }
    }

    #[test]
    fn bounded_sampling_and_tiny_frames_retain_baseline() {
        for (rows, width) in [(1, 2), (1, MAX_SAMPLE_BYTES / 2 + 1)] {
            let (field, data) = fixture(rows, width, false);
            let params = CompressionFieldParams {
                axis_delta: Some(AxisDeltaMode::Auto { axis: Some(0) }),
                compression: Some("zstd".into()),
                ..Default::default()
            };
            let (encoded, description) = try_tensor_per_value(&field, &data, &params)
                .unwrap()
                .unwrap()
                .compress(data.clone())
                .unwrap();
            let PerValueDataBlock::Fixed(encoded) = encoded else {
                panic!("expected raw rows")
            };
            assert_raw(&data, &description, &encoded);
        }
    }

    #[test]
    fn shape_permutation_and_configuration_validation() {
        let (mut field, data) = fixture(1, 30, false);
        field.metadata = HashMap::from([
            (
                "ARROW:extension:name".into(),
                "arrow.fixed_shape_tensor".into(),
            ),
            (
                "ARROW:extension:metadata".into(),
                r#"{"shape":[2,3,5],"permutation":[2,0,1]}"#.into(),
            ),
        ]);
        assert_eq!(shape(&field, None).unwrap(), (5, 1));
        assert_eq!(shape(&field, Some(0)).unwrap(), (5, 1));
        assert_eq!(shape(&field, Some(1)).unwrap(), (2, 15));
        assert_eq!(shape(&field, Some(2)).unwrap(), (3, 5));
        let params = CompressionFieldParams {
            axis_delta: Some(AxisDeltaMode::On { axis: 0 }),
            ..Default::default()
        };
        field.metadata.insert(META_KEY.into(), "off".into());
        assert!(
            try_tensor_per_value(&field, &data, &params)
                .unwrap()
                .is_none()
        );
        for value in ["invalid", "auto:-1", "on:99", "auto:99"] {
            field.metadata.insert(META_KEY.into(), value.into());
            let error = try_tensor_per_value(&field, &data, &params).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("Axis") || error.to_string().contains(META_KEY));
        }
        field.metadata.insert(META_KEY.into(), "auto:0".into());
        field.metadata.insert(
            "ARROW:extension:metadata".into(),
            r#"{"shape":[2,3,6]}"#.into(),
        );
        let error = try_tensor_per_value(&field, &data, &params).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("shape"));
    }

    #[rstest::rstest]
    #[case::outer_longest([5, 2, 3], (5, 6))]
    #[case::inner_longest([2, 3, 5], (5, 1))]
    #[case::innermost_tie([3, 3, 2], (3, 2))]
    fn default_axis_uses_longest_physical_dimension(
        #[case] dimensions: [usize; 3],
        #[case] expected: (usize, usize),
    ) {
        let (mut field, _) = fixture(1, dimensions.iter().product(), false);
        field.metadata.insert(
            "ARROW:extension:name".into(),
            "arrow.fixed_shape_tensor".into(),
        );
        field.metadata.insert(
            "ARROW:extension:metadata".into(),
            serde_json::json!({"shape": dimensions, "permutation": [2, 0, 1]}).to_string(),
        );
        assert_eq!(shape(&field, None).unwrap(), expected);
    }

    #[test]
    fn nested_tensor_and_unsupported_inputs() {
        let (field, data) = fixture(6, 30, false);
        let array = arrow_array::make_array(data.into_arrow(field.data_type(), true).unwrap());
        let flat = array.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let inner = Arc::new(FixedSizeListArray::new(
            Arc::new(arrow_schema::Field::new("element", DataType::UInt16, false)),
            5,
            flat.values().clone(),
            None,
        ));
        let nested: ArrayRef = Arc::new(FixedSizeListArray::new(
            Arc::new(arrow_schema::Field::new(
                "inner",
                inner.data_type().clone(),
                false,
            )),
            6,
            inner,
            None,
        ));
        let field = Field::try_from(&arrow_schema::Field::new(
            "tensor",
            nested.data_type().clone(),
            false,
        ))
        .unwrap();
        let data = DataBlock::from_array(nested.clone());
        let params = CompressionFieldParams {
            axis_delta: Some(AxisDeltaMode::On { axis: 0 }),
            compression: Some("lz4".into()),
            ..Default::default()
        };
        let (encoded, desc) = try_tensor_per_value(&field, &data, &params)
            .unwrap()
            .unwrap()
            .compress(data)
            .unwrap();
        let PerValueDataBlock::Variable(encoded) = encoded else {
            panic!()
        };
        let decoded = DefaultDecompressionStrategy::default()
            .create_variable_per_value_decompressor(&desc)
            .unwrap()
            .decompress(encoded)
            .unwrap();
        assert_eq!(
            decoded
                .into_arrow(nested.data_type().clone(), true)
                .unwrap(),
            nested.to_data()
        );

        let mut nullable_field = arrow_schema::Field::new(
            "tensor",
            DataType::FixedSizeList(
                Arc::new(arrow_schema::Field::new("item", DataType::Float32, true)),
                30,
            ),
            false,
        );
        let nullable: ArrayRef = Arc::new(FixedSizeListArray::new(
            Arc::new(arrow_schema::Field::new("item", DataType::UInt16, true)),
            30,
            Arc::new(UInt16Array::from(
                (0..60)
                    .map(|i| (i % 7 != 0).then_some(i as u16))
                    .collect::<Vec<_>>(),
            )),
            None,
        ));
        for is_float in [true, false] {
            if !is_float {
                nullable_field =
                    arrow_schema::Field::new("tensor", nullable.data_type().clone(), false);
            }
            let field = Field::try_from(&nullable_field).unwrap();
            let data = DataBlock::from_array(nullable.clone());
            let auto = CompressionFieldParams {
                axis_delta: Some(AxisDeltaMode::Auto { axis: Some(0) }),
                ..params.clone()
            };
            let (encoded, description) = try_tensor_per_value(&field, &data, &auto)
                .unwrap()
                .unwrap()
                .compress(data.clone())
                .unwrap();
            let PerValueDataBlock::Fixed(encoded) = encoded else {
                panic!("expected raw rows")
            };
            assert_raw(&data, &description, &encoded);
            let error = try_tensor_per_value(&field, &data, &params).unwrap_err();
            assert!(matches!(error, Error::NotSupported { .. }));
            assert!(error.to_string().contains("AxisDelta"));
        }
    }

    #[test]
    fn rejects_corrupt_layouts() {
        for (length, stride) in [(0, 1), (3, 0), (u64::MAX, 2), (4, 3)] {
            let mut d = descriptor(16, &[2, 3, 5], 1);
            d.axis_length = length;
            d.inner_stride = stride;
            let error = AxisDeltaDecompressor::try_new(&d).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("AxisDelta"));
        }
        let oversized = descriptor(8, &[1_u64 << 61], 0);
        let error = AxisDeltaDecompressor::try_new(&oversized).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("byte size"));
        let mut d = descriptor(16, &[30], 0);
        d.values = Some(Box::new(EncodingUtils::fsl(
            30,
            true,
            EncodingUtils::flat(16, None),
        )));
        let error = AxisDeltaDecompressor::try_new(&d).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("validity"));
    }
}
