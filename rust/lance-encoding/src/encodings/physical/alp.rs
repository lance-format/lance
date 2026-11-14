// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Modern ALP (Adaptive Lossless floating-Point) miniblock encoding.
//!
//! ALP maps finite-decimal floating-point values onto integers with a base-10
//! exponent transform, then bit-packs the Frame-of-Reference (FOR) or Delta
//! stream. Values that cannot be reconstructed bitwise are stored verbatim as
//! exceptions, which preserves NaN payloads, signed zeros, and subnormals.
//!
//! # Transform
//!
//! Two reconstruction paths are supported per chunk:
//!
//! - **Multiply** (classic `(e, f)` pair): `enc = round_ties_even(v * 10^e *
//!   10^-f)` and `v = enc * 10^f * 10^-e`. The two-step multiplication chain
//!   can diverge by 1 ULP for `f32`, so this path is only selected when it
//!   bitwise round-trips every non-exception value.
//! - **Divide** (`use_div`): `enc = round_ties_even(v * 10^e)` and `v = enc /
//!   10^e`. The reconstruction is a single correctly-rounded division, which
//!   repairs the `f32` truncation failures of the multiply chain.
//!
//! Every value is validated with a bitwise round-trip check (`to_bits()`
//! equality) before it is admitted to the integer stream.
//!
//! # Buffer layout
//!
//! Each mini-block chunk uses four value buffers (see
//! [`MiniBlockChunk::buffer_sizes`]).
//!
//! - **Buffer 0 (payload):** bitpacked offsets, `enc - min` for FOR mode or
//!   `delta - min_delta` for Delta mode, using the same layout as
//!   `bitpack_out_of_line_vec`.
//! - **Buffer 1 (header):** per-chunk header:
//!   - `mode`: bit 0 = Delta, bit 1 = `use_div`
//!   - `e` / `f`: base-10 exponents (f is ignored when `use_div`)
//!   - `bit_width`: bits per packed offset
//!   - FOR: `min`; Delta: `first` then `min_delta` (little-endian i32/i64)
//! - **Buffer 2 (exception positions):** `u16` positions relative to the chunk
//! - **Buffer 3 (exception values):** original IEEE-754 bit patterns
//!
//! # Fallback
//!
//! A page that cannot be encoded with ALP (no encodable values, packed size at
//! or above the raw size, or any chunk exceeding `MAX_MINIBLOCK_BYTES`) falls
//! back to the flat [`ValueEncoder`], guaranteeing ALP never enlarges data.

mod restore;

use std::fmt::Debug;

use crate::buffer::LanceBuffer;
use crate::compression::MiniBlockDecompressor;
use crate::compression_config::AlpMode;
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MAX_MINIBLOCK_BYTES, MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressionContext,
    MiniBlockCompressor,
};
#[cfg(all(test, feature = "bitpacking"))]
use crate::encodings::physical::bitpacking::bitpack_out_of_line_vec;
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::{
    bitpack_out_of_line_slice_into, out_of_line_payload_byte_lengths, unpack_out_of_line,
};
use crate::encodings::physical::value::ValueEncoder;
use crate::format::EncodingUtils;
use crate::format::metadata;
use crate::format::metadata::CompressiveEncoding;
use crate::statistics::{GetStat, Stat};
use arrow_array::cast::AsArray;

#[cfg(all(feature = "bitpacking", target_arch = "x86_64"))]
use lance_core::utils::cpu::CpuFeatures;
use lance_core::{Error, Result};

const MODE_DELTA: u8 = 0b0000_0001;
const MODE_DIV: u8 = 0b0000_0010;
/// ALP-RD splits values into a dictionary-coded left part (high bits) and a
/// bit-packed right part (low bits). See [`encode_chunk_f64_rd`].
const MODE_RD: u8 = 0b0000_0100;

/// Candidate right-part widths for the ALP-RD split.
const RD_SWEEP: [u8; 7] = [8, 16, 24, 32, 40, 48, 56];

const MAX_EXP_F32: u8 = 10;
const MAX_EXP_F64: u8 = 18;

const F10_F32: [f32; MAX_EXP_F32 as usize + 1] = [
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
    10000000000.0,
];
const IF10_F32: [f32; MAX_EXP_F32 as usize + 1] = [
    1.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    0.00001,
    0.000001,
    0.0000001,
    0.00000001,
    0.000000001,
    0.0000000001,
];

const F10_F64: [f64; MAX_EXP_F64 as usize + 1] = [
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
    10000000000.0,
    100000000000.0,
    1000000000000.0,
    10000000000000.0,
    100000000000000.0,
    1000000000000000.0,
    10000000000000000.0,
    100000000000000000.0,
    1000000000000000000.0,
];
const IF10_F64: [f64; MAX_EXP_F64 as usize + 1] = [
    1.0,
    0.1,
    0.01,
    0.001,
    0.0001,
    0.00001,
    0.000001,
    0.0000001,
    0.00000001,
    0.000000001,
    0.0000000001,
    0.00000000001,
    0.000000000001,
    0.0000000000001,
    0.00000000000001,
    0.000000000000001,
    0.0000000000000001,
    0.00000000000000001,
    0.000000000000000001,
];

/// Chunk-level ALP transform parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AlpParams {
    e: u8,
    f: u8,
    use_div: bool,
}

#[inline]
fn bits_required_u128(v: u128) -> u8 {
    if v == 0 {
        0
    } else {
        (128 - v.leading_zeros()) as u8
    }
}

/// Multiply `(e, f)` pairs plus the `use_div` pure-exponent candidates.
fn candidate_params(max_exp: u8) -> impl Iterator<Item = AlpParams> {
    let multiply = (1..=max_exp).flat_map(move |e| {
        (0..e).map(move |f| AlpParams {
            e,
            f,
            use_div: false,
        })
    });
    let divide = (1..=max_exp).map(move |e| AlpParams {
        e,
        f: 0,
        use_div: true,
    });
    multiply.chain(divide)
}

const SAMPLE_SIZE: usize = 32;

/// Copy a strided sample of `words` into `out`, returning the sample length.
///
/// Pages smaller than the sample size are copied in full so the estimator
/// sees every value; larger pages are sampled at a fixed stride.
#[inline]
fn fill_sample<T: Copy>(words: &[T], out: &mut [T; SAMPLE_SIZE]) -> usize {
    let n = words.len();
    if n <= SAMPLE_SIZE {
        out[..n].copy_from_slice(words);
        n
    } else {
        let step = n / SAMPLE_SIZE;
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = words[i * step];
        }
        SAMPLE_SIZE
    }
}

#[inline]
fn try_encode_f32(v: f32, params: AlpParams) -> Option<i32> {
    let bits = v.to_bits();
    if !v.is_finite() || bits == 0x8000_0000 {
        return None;
    }
    let e = params.e as usize;
    if params.use_div {
        let scaled = v * F10_F32[e];
        if !scaled.is_finite() {
            return None;
        }
        let enc = scaled.round_ties_even() as i32;
        let decoded = (enc as f32) / F10_F32[e];
        (decoded.to_bits() == bits).then_some(enc)
    } else {
        let f = params.f as usize;
        let scaled = v * F10_F32[e] * IF10_F32[f];
        if !scaled.is_finite() {
            return None;
        }
        let enc = scaled.round_ties_even() as i32;
        let decoded = (enc as f32) * F10_F32[f] * IF10_F32[e];
        (decoded.to_bits() == bits).then_some(enc)
    }
}

#[inline]
fn try_encode_f64(v: f64, params: AlpParams) -> Option<i64> {
    let bits = v.to_bits();
    if !v.is_finite() || bits == 0x8000_0000_0000_0000 {
        return None;
    }
    let e = params.e as usize;
    if params.use_div {
        let scaled = v * F10_F64[e];
        if !scaled.is_finite() {
            return None;
        }
        let enc = scaled.round_ties_even() as i64;
        let decoded = (enc as f64) / F10_F64[e];
        (decoded.to_bits() == bits).then_some(enc)
    } else {
        let f = params.f as usize;
        let scaled = v * F10_F64[e] * IF10_F64[f];
        if !scaled.is_finite() {
            return None;
        }
        let enc = scaled.round_ties_even() as i64;
        let decoded = (enc as f64) * F10_F64[f] * IF10_F64[e];
        (decoded.to_bits() == bits).then_some(enc)
    }
}

/// Pick the cheapest `(e, f)`/`use_div` candidate on a small sample.
///
/// Candidates run in the same order as the original exhaustive scan so that
/// equal-cost ties resolve identically. A candidate is pruned as soon as its
/// exception cost alone reaches the current best: the full-page estimate is a
/// superset of the partial one, so the pruned candidate cannot win.
fn select_params_f32(words: &[u32]) -> Option<AlpParams> {
    let mut sample_buf = [0u32; SAMPLE_SIZE];
    let sample_len = fill_sample(words, &mut sample_buf);
    let sample = &sample_buf[..sample_len];
    let mut best = None;
    let mut best_cost = usize::MAX;
    for params in candidate_params(MAX_EXP_F32) {
        let Some(cost) = sample_cost_f32(sample, params, best_cost) else {
            continue;
        };
        if cost < best_cost {
            best_cost = cost;
            best = Some(params);
        }
    }
    best
}

fn select_params_f64(words: &[u64]) -> Option<AlpParams> {
    let mut sample_buf = [0u64; SAMPLE_SIZE];
    let sample_len = fill_sample(words, &mut sample_buf);
    let sample = &sample_buf[..sample_len];
    let mut best = None;
    let mut best_cost = usize::MAX;
    for params in candidate_params(MAX_EXP_F64) {
        let Some(cost) = sample_cost_f64(sample, params, best_cost) else {
            continue;
        };
        if cost < best_cost {
            best_cost = cost;
            best = Some(params);
        }
    }
    best
}

/// Estimated candidate cost on a small sample, or `None` when the candidate
/// cannot beat `cutoff` (already provably worse than the current best).
fn sample_cost_f32(words: &[u32], params: AlpParams, cutoff: usize) -> Option<usize> {
    const EXCEPTION_COST: usize = 2 + 4;
    let mut exceptions = 0usize;
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    let mut any = false;
    for &w in words {
        match try_encode_f32(f32::from_bits(w), params) {
            Some(enc) => {
                any = true;
                min = min.min(enc);
                max = max.max(enc);
                // min/max only grow, so the byte estimate is a lower bound on
                // the final cost: a candidate that already cannot beat the
                // current best is pruned exactly as if it finished scanning.
                let data_bytes = (bits_required_u128((max as i128 - min as i128) as u128).max(1)
                    as usize
                    * words.len())
                .div_ceil(8);
                if data_bytes + exceptions * EXCEPTION_COST >= cutoff {
                    return None;
                }
            }
            None => {
                exceptions += 1;
                if exceptions * EXCEPTION_COST >= cutoff {
                    return None;
                }
            }
        }
    }
    if !any {
        return None;
    }
    let range = (max as i128 - min as i128) as u128;
    let bit_width = bits_required_u128(range).max(1);
    let data_bytes = (bit_width as usize * words.len()).div_ceil(8);
    Some(data_bytes + exceptions * EXCEPTION_COST)
}

/// Estimated candidate cost on a small sample, or `None` when the candidate
/// cannot beat `cutoff` (already provably worse than the current best).
fn sample_cost_f64(words: &[u64], params: AlpParams, cutoff: usize) -> Option<usize> {
    const EXCEPTION_COST: usize = 2 + 8;
    let mut exceptions = 0usize;
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    let mut any = false;
    for &w in words {
        match try_encode_f64(f64::from_bits(w), params) {
            Some(enc) => {
                any = true;
                min = min.min(enc);
                max = max.max(enc);
                let data_bytes = (bits_required_u128((max as i128 - min as i128) as u128).max(1)
                    as usize
                    * words.len())
                .div_ceil(8);
                if data_bytes + exceptions * EXCEPTION_COST >= cutoff {
                    return None;
                }
            }
            None => {
                exceptions += 1;
                if exceptions * EXCEPTION_COST >= cutoff {
                    return None;
                }
            }
        }
    }
    if !any {
        return None;
    }
    let range = (max as i128 - min as i128) as u128;
    let bit_width = bits_required_u128(range).max(1);
    let data_bytes = (bit_width as usize * words.len()).div_ceil(8);
    Some(data_bytes + exceptions * EXCEPTION_COST)
}

/// Opt-in ALP encoder, selected via compression metadata/config.
///
/// The encoder is version-gated by the caller (Lance file version >= 2.2). It
/// falls back to [`ValueEncoder`] when ALP is ineffective for the page.
#[derive(Debug, Clone)]
pub struct AlpMiniBlockEncoder {
    bits_per_value: u64,
}

impl AlpMiniBlockEncoder {
    pub fn new(bits_per_value: u64) -> Self {
        assert!(bits_per_value == 32 || bits_per_value == 64);
        Self { bits_per_value }
    }

    fn max_chunk_size(&self) -> usize {
        // The out-of-line bitpack layout pads chunks to 1024-value blocks, so
        // smaller chunks would waste half of every block. 64-bit ALP offsets
        // up to the 63-bit fallback threshold still fit the 8 KiB chunk budget.
        1024
    }
}

/// Decide whether ALP should be attempted for a block, based on the mode and
/// (in `Auto`) the precomputed byte-position entropy.
///
/// The `Auto` threshold is deliberately higher than the one used by BSS: a
/// false positive only wastes one candidate scan that falls back, while a
/// false negative silently gives up a lossless win.
pub fn should_use_alp(data: &FixedWidthDataBlock, mode: AlpMode) -> bool {
    match mode {
        AlpMode::Off => false,
        AlpMode::On => true,
        AlpMode::Auto => low_byte_position_entropy(data),
    }
}

fn low_byte_position_entropy(data: &FixedWidthDataBlock) -> bool {
    // The byte-position statistic samples 64 values, so its upper bound is
    // 6.0 bits. On the 4M-value evaluation corpus (2026-10-23, EC2 i4i.4xlarge),
    // the best ALP-suitable datasets (prices/sensor/drift, f32+f64) peak at
    // 4.504 average bits per byte position, while incompressible random data
    // (gaussian/uniform, f32+f64) bottoms out at 4.72. 4.6 separates the two:
    // suitable data keeps every page, random data never enters the candidate
    // scan (a false positive only wastes one scan that falls back). A higher
    // threshold like 5.0 lets gaussian32/uniform32 through the gate on most
    // pages and costs 4.5-4.8x encode throughput on those columns.
    const AUTO_ENTROPY_THRESHOLD: f64 = 4.6;
    let Some(entropy_stat) = data.get_stat(Stat::BytePositionEntropy) else {
        return false;
    };
    let entropies = entropy_stat.as_primitive::<arrow_array::types::UInt64Type>();
    if entropies.is_empty() {
        return false;
    }
    let sum: u64 = entropies.values().iter().sum();
    let avg_entropy = sum as f64 / entropies.len() as f64 / 1000.0;
    avg_entropy < AUTO_ENTROPY_THRESHOLD
}

impl MiniBlockCompressor for AlpMiniBlockEncoder {
    fn compress(
        &self,
        context: MiniBlockCompressionContext,
        page: DataBlock,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        let DataBlock::FixedWidth(data) = page else {
            return Err(Error::invalid_input(
                "ALP encoding only supports FixedWidth data blocks",
            ));
        };

        if data.bits_per_value != self.bits_per_value {
            return Err(Error::invalid_input("ALP bits_per_value mismatch"));
        }

        if data.num_values == 0 {
            return Ok((
                MiniBlockCompressed {
                    data: vec![
                        LanceBuffer::empty(),
                        LanceBuffer::empty(),
                        LanceBuffer::empty(),
                        LanceBuffer::empty(),
                    ],
                    chunks: vec![],
                    num_values: 0,
                },
                EncodingUtils::alp(self.bits_per_value as u32),
            ));
        }

        let raw_size = data.num_values as usize * (self.bits_per_value as usize / 8);
        let encoded = match self.bits_per_value {
            32 => encode_page_f32(&data, self.max_chunk_size(), raw_size)?,
            64 => encode_page_f64(&data, self.max_chunk_size(), raw_size)?,
            _ => unreachable!(),
        };

        let Some(encoded) = encoded else {
            return ValueEncoder::default().compress(context, DataBlock::FixedWidth(data));
        };
        Ok(encoded)
    }
}

#[cfg(feature = "bitpacking")]
struct EncodeScratch<T, U> {
    encoded: Vec<T>,
    offsets: Vec<U>,
    exception_positions: Vec<u16>,
    exception_bits: Vec<u8>,
}

#[cfg(feature = "bitpacking")]
impl<T: Copy + Default, U: Copy + Default> EncodeScratch<T, U> {
    fn with_chunk_capacity(max_chunk: usize) -> Self {
        Self {
            encoded: vec![T::default(); max_chunk],
            offsets: vec![U::default(); max_chunk],
            exception_positions: Vec::new(),
            exception_bits: Vec::new(),
        }
    }

    fn prepare(&mut self, n: usize) {
        if self.encoded.len() < n {
            self.encoded.resize(n, T::default());
        }
        if self.offsets.len() < n {
            self.offsets.resize(n, U::default());
        }
        self.exception_positions.clear();
        self.exception_bits.clear();
    }
}

#[cfg(feature = "bitpacking")]
fn encode_page_f32(
    data: &FixedWidthDataBlock,
    max_chunk: usize,
    raw_size: usize,
) -> Result<Option<(MiniBlockCompressed, CompressiveEncoding)>> {
    let words = data.data.borrow_to_typed_view::<u32>();
    let Some(params) = select_params_f32(&words) else {
        return Ok(None);
    };

    let num_chunks = words.len().div_ceil(max_chunk);
    let mut buf0: Vec<u32> = Vec::with_capacity(words.len() / 2);
    let mut buf1: Vec<u8> = Vec::with_capacity(num_chunks * 12);
    let mut buf2: Vec<u8> = Vec::new();
    let mut buf3: Vec<u8> = Vec::new();
    let mut chunks = Vec::with_capacity(num_chunks);
    let mut compressed_size = 0usize;
    let mut scratch = EncodeScratch::<i32, u32>::with_chunk_capacity(max_chunk);

    for chunk in words.chunks(max_chunk) {
        let Some(sizes) = encode_chunk_f32_into(
            chunk,
            params,
            &mut scratch,
            &mut buf0,
            &mut buf1,
            &mut buf2,
            &mut buf3,
        )?
        else {
            return Ok(None);
        };
        compressed_size +=
            sizes[0] as usize + sizes[1] as usize + sizes[2] as usize + sizes[3] as usize;

        chunks.push(MiniBlockChunk {
            buffer_sizes: sizes.to_vec(),
            log_num_values: if chunk.len() < max_chunk {
                0
            } else {
                chunk.len().ilog2() as u8
            },
        });
    }

    if compressed_size >= raw_size {
        return Ok(None);
    }
    Ok(Some((
        MiniBlockCompressed {
            data: vec![
                LanceBuffer::reinterpret_vec(buf0),
                LanceBuffer::from(buf1),
                LanceBuffer::from(buf2),
                LanceBuffer::from(buf3),
            ],
            chunks,
            num_values: data.num_values,
        },
        EncodingUtils::alp(32),
    )))
}

#[cfg(not(feature = "bitpacking"))]
fn encode_page_f32(
    _data: &FixedWidthDataBlock,
    _max_chunk: usize,
    _raw_size: usize,
) -> Result<Option<(MiniBlockCompressed, CompressiveEncoding)>> {
    Ok(None)
}

#[cfg(feature = "bitpacking")]
fn encode_page_f64(
    data: &FixedWidthDataBlock,
    max_chunk: usize,
    raw_size: usize,
) -> Result<Option<(MiniBlockCompressed, CompressiveEncoding)>> {
    let words = data.data.borrow_to_typed_view::<u64>();
    let Some(params) = select_params_f64(&words) else {
        return encode_page_f64_rd(&words, max_chunk, raw_size);
    };

    let num_chunks = words.len().div_ceil(max_chunk);
    let mut buf0: Vec<u64> = Vec::with_capacity(words.len() / 2);
    let mut buf1: Vec<u8> = Vec::with_capacity(num_chunks * 20);
    let mut buf2: Vec<u8> = Vec::new();
    let mut buf3: Vec<u8> = Vec::new();
    let mut chunks = Vec::with_capacity(num_chunks);
    let mut compressed_size = 0usize;
    let mut scratch = EncodeScratch::<i64, u64>::with_chunk_capacity(max_chunk);

    for chunk in words.chunks(max_chunk) {
        let Some(sizes) = encode_chunk_f64_into(
            chunk,
            params,
            &mut scratch,
            &mut buf0,
            &mut buf1,
            &mut buf2,
            &mut buf3,
        )?
        else {
            return encode_page_f64_rd(&words, max_chunk, raw_size);
        };
        compressed_size +=
            sizes[0] as usize + sizes[1] as usize + sizes[2] as usize + sizes[3] as usize;

        chunks.push(MiniBlockChunk {
            buffer_sizes: sizes.to_vec(),
            log_num_values: if chunk.len() < max_chunk {
                0
            } else {
                chunk.len().ilog2() as u8
            },
        });
    }

    if compressed_size >= raw_size {
        return encode_page_f64_rd(&words, max_chunk, raw_size);
    }
    Ok(Some((
        MiniBlockCompressed {
            data: vec![
                LanceBuffer::reinterpret_vec(buf0),
                LanceBuffer::from(buf1),
                LanceBuffer::from(buf2),
                LanceBuffer::from(buf3),
            ],
            chunks,
            num_values: data.num_values,
        },
        EncodingUtils::alp(64),
    )))
}

#[cfg(not(feature = "bitpacking"))]
fn encode_page_f64(
    _data: &FixedWidthDataBlock,
    _max_chunk: usize,
    _raw_size: usize,
) -> Result<Option<(MiniBlockCompressed, CompressiveEncoding)>> {
    Ok(None)
}

#[cfg(feature = "bitpacking")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn record_slot_32(
    i: usize,
    w: u32,
    enc: i32,
    ok: bool,
    encoded: &mut [i32],
    first_encodable: &mut Option<i32>,
    prev: &mut i32,
    exception_positions: &mut Vec<u16>,
    exception_bits: &mut Vec<u8>,
) {
    if ok {
        if first_encodable.is_none() {
            *first_encodable = Some(enc);
        }
        *prev = enc;
        encoded[i] = enc;
    } else {
        encoded[i] = *prev;
        exception_positions.push(i as u16);
        exception_bits.extend_from_slice(&w.to_le_bytes());
    }
}

#[cfg(feature = "bitpacking")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn record_slot_64(
    i: usize,
    w: u64,
    enc: i64,
    ok: bool,
    encoded: &mut [i64],
    first_encodable: &mut Option<i64>,
    prev: &mut i64,
    exception_positions: &mut Vec<u16>,
    exception_bits: &mut Vec<u8>,
) {
    if ok {
        if first_encodable.is_none() {
            *first_encodable = Some(enc);
        }
        *prev = enc;
        encoded[i] = enc;
    } else {
        encoded[i] = *prev;
        exception_positions.push(i as u16);
        exception_bits.extend_from_slice(&w.to_le_bytes());
    }
}

#[cfg(feature = "bitpacking")]
fn transform_chunk_f32_scalar(
    words: &[u32],
    params: AlpParams,
    encoded: &mut [i32],
    exception_positions: &mut Vec<u16>,
    exception_bits: &mut Vec<u8>,
) -> Option<i32> {
    let mut first_encodable = None;
    let mut prev = 0i32;
    let e = params.e as usize;

    if params.use_div {
        let scale = F10_F32[e];
        for (i, &w) in words.iter().enumerate() {
            let v = f32::from_bits(w);
            let scaled = v * scale;
            let enc = scaled.round_ties_even() as i32;
            let ok = scaled.is_finite() && ((enc as f32) / scale).to_bits() == w;
            record_slot_32(
                i,
                w,
                enc,
                ok,
                encoded,
                &mut first_encodable,
                &mut prev,
                exception_positions,
                exception_bits,
            );
        }
    } else {
        let f = params.f as usize;
        let scale_e = F10_F32[e];
        let inv_f = IF10_F32[f];
        let reconstruct_scale = F10_F32[f];
        let reconstruct_inv = IF10_F32[e];
        for (i, &w) in words.iter().enumerate() {
            let v = f32::from_bits(w);
            let scaled = (v * scale_e) * inv_f;
            let enc = scaled.round_ties_even() as i32;
            let ok = scaled.is_finite()
                && (((enc as f32) * reconstruct_scale) * reconstruct_inv).to_bits() == w;
            record_slot_32(
                i,
                w,
                enc,
                ok,
                encoded,
                &mut first_encodable,
                &mut prev,
                exception_positions,
                exception_bits,
            );
        }
    }

    first_encodable
}

/// 8-lane AVX2 fast path for `f32` ALP transformation.
///
/// Processes 8 lanes via `_mm256_round_ps::<0x08>` (explicit round-ties-to-even)
/// when all 8 scaled magnitudes are strictly below `2^30` (guaranteeing exact
/// `i32` conversion) and all 8 lanes bitwise round-trip. Any 8-lane group with
/// an out-of-range value, `NaN`/`±inf`, `-0.0`, or exception falls back to the
/// scalar loop for bit-for-bit identical exception bookkeeping.
#[cfg(all(feature = "bitpacking", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn transform_chunk_f32_avx2(
    words: &[u32],
    params: AlpParams,
    encoded: &mut [i32],
    exception_positions: &mut Vec<u16>,
    exception_bits: &mut Vec<u8>,
) -> Option<i32> {
    use std::arch::x86_64::*;

    let n = words.len();
    let mut i = 0usize;
    let mut first_encodable = None;
    let mut prev = 0i32;
    let e = params.e as usize;

    let sign_mask = _mm256_set1_ps(-0.0);
    let limit_2p30 = _mm256_set1_ps(1_073_741_824.0); // 2^30

    if params.use_div {
        let scale = F10_F32[e];
        let scale_v = _mm256_set1_ps(scale);
        while i + 8 <= n {
            // SAFETY: `i + 8 <= n == words.len() <= encoded.len()`.
            let raw_w = unsafe { _mm256_loadu_si256(words.as_ptr().add(i).cast()) };
            let v = _mm256_castsi256_ps(raw_w);
            let scaled = _mm256_mul_ps(v, scale_v);
            let abs_scaled = _mm256_andnot_ps(sign_mask, scaled);
            let in_range = _mm256_cmp_ps::<_CMP_LT_OQ>(abs_scaled, limit_2p30);
            if _mm256_movemask_ps(in_range) == 0xFF {
                let rounded = _mm256_round_ps::<0x08>(scaled);
                let enc_v = _mm256_cvttps_epi32(rounded);
                let enc_f32 = _mm256_cvtepi32_ps(enc_v);
                let recon = _mm256_div_ps(enc_f32, scale_v);
                let eq = _mm256_cmpeq_epi32(_mm256_castps_si256(recon), raw_w);
                if _mm256_movemask_ps(_mm256_castsi256_ps(eq)) == 0xFF {
                    // SAFETY: `encoded` has at least `n` initialized elements.
                    unsafe { _mm256_storeu_si256(encoded.as_mut_ptr().add(i).cast(), enc_v) };
                    if first_encodable.is_none() {
                        first_encodable = Some(encoded[i]);
                    }
                    prev = encoded[i + 7];
                    i += 8;
                    continue;
                }
            }
            for (off, &w) in words[i..i + 8].iter().enumerate() {
                let j = i + off;
                let vf = f32::from_bits(w);
                let sc = vf * scale;
                let enc = sc.round_ties_even() as i32;
                let ok = sc.is_finite() && ((enc as f32) / scale).to_bits() == w;
                record_slot_32(
                    j,
                    w,
                    enc,
                    ok,
                    encoded,
                    &mut first_encodable,
                    &mut prev,
                    exception_positions,
                    exception_bits,
                );
            }
            i += 8;
        }
        for (j, &w) in words[i..n].iter().enumerate() {
            let idx = i + j;
            let vf = f32::from_bits(w);
            let sc = vf * scale;
            let enc = sc.round_ties_even() as i32;
            let ok = sc.is_finite() && ((enc as f32) / scale).to_bits() == w;
            record_slot_32(
                idx,
                w,
                enc,
                ok,
                encoded,
                &mut first_encodable,
                &mut prev,
                exception_positions,
                exception_bits,
            );
        }
    } else {
        let f = params.f as usize;
        let scale_e = F10_F32[e];
        let inv_f = IF10_F32[f];
        let reconstruct_scale = F10_F32[f];
        let reconstruct_inv = IF10_F32[e];
        let scale_e_v = _mm256_set1_ps(scale_e);
        let inv_f_v = _mm256_set1_ps(inv_f);
        let recon_scale_v = _mm256_set1_ps(reconstruct_scale);
        let recon_inv_v = _mm256_set1_ps(reconstruct_inv);

        while i + 8 <= n {
            // SAFETY: `i + 8 <= n == words.len() <= encoded.len()`.
            let raw_w = unsafe { _mm256_loadu_si256(words.as_ptr().add(i).cast()) };
            let v = _mm256_castsi256_ps(raw_w);
            let scaled = _mm256_mul_ps(_mm256_mul_ps(v, scale_e_v), inv_f_v);
            let abs_scaled = _mm256_andnot_ps(sign_mask, scaled);
            let in_range = _mm256_cmp_ps::<_CMP_LT_OQ>(abs_scaled, limit_2p30);
            if _mm256_movemask_ps(in_range) == 0xFF {
                let rounded = _mm256_round_ps::<0x08>(scaled);
                let enc_v = _mm256_cvttps_epi32(rounded);
                let enc_f32 = _mm256_cvtepi32_ps(enc_v);
                let recon = _mm256_mul_ps(_mm256_mul_ps(enc_f32, recon_scale_v), recon_inv_v);
                let eq = _mm256_cmpeq_epi32(_mm256_castps_si256(recon), raw_w);
                if _mm256_movemask_ps(_mm256_castsi256_ps(eq)) == 0xFF {
                    // SAFETY: `encoded` has at least `n` initialized elements.
                    unsafe { _mm256_storeu_si256(encoded.as_mut_ptr().add(i).cast(), enc_v) };
                    if first_encodable.is_none() {
                        first_encodable = Some(encoded[i]);
                    }
                    prev = encoded[i + 7];
                    i += 8;
                    continue;
                }
            }
            for (off, &w) in words[i..i + 8].iter().enumerate() {
                let j = i + off;
                let vf = f32::from_bits(w);
                let sc = (vf * scale_e) * inv_f;
                let enc = sc.round_ties_even() as i32;
                let ok = sc.is_finite()
                    && (((enc as f32) * reconstruct_scale) * reconstruct_inv).to_bits() == w;
                record_slot_32(
                    j,
                    w,
                    enc,
                    ok,
                    encoded,
                    &mut first_encodable,
                    &mut prev,
                    exception_positions,
                    exception_bits,
                );
            }
            i += 8;
        }
        for (j, &w) in words[i..n].iter().enumerate() {
            let idx = i + j;
            let vf = f32::from_bits(w);
            let sc = (vf * scale_e) * inv_f;
            let enc = sc.round_ties_even() as i32;
            let ok = sc.is_finite()
                && (((enc as f32) * reconstruct_scale) * reconstruct_inv).to_bits() == w;
            record_slot_32(
                idx,
                w,
                enc,
                ok,
                encoded,
                &mut first_encodable,
                &mut prev,
                exception_positions,
                exception_bits,
            );
        }
    }

    first_encodable
}

#[cfg(feature = "bitpacking")]
fn transform_chunk_f64_scalar(
    words: &[u64],
    params: AlpParams,
    encoded: &mut [i64],
    exception_positions: &mut Vec<u16>,
    exception_bits: &mut Vec<u8>,
) -> Option<i64> {
    let mut first_encodable = None;
    let mut prev = 0i64;
    let e = params.e as usize;

    if params.use_div {
        let scale = F10_F64[e];
        for (i, &w) in words.iter().enumerate() {
            let v = f64::from_bits(w);
            let scaled = v * scale;
            let enc = scaled.round_ties_even() as i64;
            let ok = scaled.is_finite() && ((enc as f64) / scale).to_bits() == w;
            record_slot_64(
                i,
                w,
                enc,
                ok,
                encoded,
                &mut first_encodable,
                &mut prev,
                exception_positions,
                exception_bits,
            );
        }
    } else {
        let f = params.f as usize;
        let scale_e = F10_F64[e];
        let inv_f = IF10_F64[f];
        let reconstruct_scale = F10_F64[f];
        let reconstruct_inv = IF10_F64[e];
        for (i, &w) in words.iter().enumerate() {
            let v = f64::from_bits(w);
            let scaled = (v * scale_e) * inv_f;
            let enc = scaled.round_ties_even() as i64;
            let ok = scaled.is_finite()
                && (((enc as f64) * reconstruct_scale) * reconstruct_inv).to_bits() == w;
            record_slot_64(
                i,
                w,
                enc,
                ok,
                encoded,
                &mut first_encodable,
                &mut prev,
                exception_positions,
                exception_bits,
            );
        }
    }

    first_encodable
}

/// 4-lane AVX2 magic-double fast path for `f64` ALP transformation.
///
/// AVX2 lacks 64-bit float<->int conversions (`vcvttpd2qq`/`vcvtqq2pd` require
/// AVX-512DQ). For finite `|scaled| < 2^51`, adding `MAGIC = 3 * 2^51`
/// (`0x4338_0000_0000_0000`) places `m = scaled + MAGIC` inside `(2^52, 2^53)`,
/// where IEEE-754 addition rounds to the nearest integer with ties-to-even,
/// `m - MAGIC` recovers `enc as f64`, and `bits(m) - bits(MAGIC)` yields the
/// exact two's-complement `i64`. Any 4-lane group outside `(-2^51, 2^51)` or
/// failing bitwise round-trip falls back to the scalar loop.
#[cfg(all(feature = "bitpacking", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn transform_chunk_f64_avx2(
    words: &[u64],
    params: AlpParams,
    encoded: &mut [i64],
    exception_positions: &mut Vec<u16>,
    exception_bits: &mut Vec<u8>,
) -> Option<i64> {
    use std::arch::x86_64::*;

    const MAGIC_F64: f64 = 6_755_399_441_055_744.0; // 3 * 2^51 = 0x4338_0000_0000_0000
    const MAGIC_BITS: i64 = 0x4338_0000_0000_0000_u64 as i64;
    const LIMIT_2P51: f64 = 2_251_799_813_685_248.0; // 2^51

    let n = words.len();
    let mut i = 0usize;
    let mut first_encodable = None;
    let mut prev = 0i64;
    let e = params.e as usize;

    let sign_mask = _mm256_set1_pd(-0.0);
    let limit_v = _mm256_set1_pd(LIMIT_2P51);
    let magic_pd = _mm256_set1_pd(MAGIC_F64);
    let magic_i64 = _mm256_set1_epi64x(MAGIC_BITS);

    if params.use_div {
        let scale = F10_F64[e];
        let scale_v = _mm256_set1_pd(scale);
        while i + 4 <= n {
            // SAFETY: `i + 4 <= n == words.len() <= encoded.len()`.
            let raw_w = unsafe { _mm256_loadu_si256(words.as_ptr().add(i).cast()) };
            let v = _mm256_castsi256_pd(raw_w);
            let scaled = _mm256_mul_pd(v, scale_v);
            let abs_scaled = _mm256_andnot_pd(sign_mask, scaled);
            let in_range = _mm256_cmp_pd::<_CMP_LT_OQ>(abs_scaled, limit_v);
            if _mm256_movemask_pd(in_range) == 0xF {
                let m = _mm256_add_pd(scaled, magic_pd);
                let rounded = _mm256_sub_pd(m, magic_pd);
                let recon = _mm256_div_pd(rounded, scale_v);
                let eq = _mm256_cmpeq_epi64(_mm256_castpd_si256(recon), raw_w);
                if _mm256_movemask_pd(_mm256_castsi256_pd(eq)) == 0xF {
                    let enc_v = _mm256_sub_epi64(_mm256_castpd_si256(m), magic_i64);
                    // SAFETY: `encoded` has at least `n` initialized elements.
                    unsafe { _mm256_storeu_si256(encoded.as_mut_ptr().add(i).cast(), enc_v) };
                    if first_encodable.is_none() {
                        first_encodable = Some(encoded[i]);
                    }
                    prev = encoded[i + 3];
                    i += 4;
                    continue;
                }
            }
            for (off, &w) in words[i..i + 4].iter().enumerate() {
                let j = i + off;
                let vf = f64::from_bits(w);
                let sc = vf * scale;
                let enc = sc.round_ties_even() as i64;
                let ok = sc.is_finite() && ((enc as f64) / scale).to_bits() == w;
                record_slot_64(
                    j,
                    w,
                    enc,
                    ok,
                    encoded,
                    &mut first_encodable,
                    &mut prev,
                    exception_positions,
                    exception_bits,
                );
            }
            i += 4;
        }
        for (j, &w) in words[i..n].iter().enumerate() {
            let idx = i + j;
            let vf = f64::from_bits(w);
            let sc = vf * scale;
            let enc = sc.round_ties_even() as i64;
            let ok = sc.is_finite() && ((enc as f64) / scale).to_bits() == w;
            record_slot_64(
                idx,
                w,
                enc,
                ok,
                encoded,
                &mut first_encodable,
                &mut prev,
                exception_positions,
                exception_bits,
            );
        }
    } else {
        let f = params.f as usize;
        let scale_e = F10_F64[e];
        let inv_f = IF10_F64[f];
        let reconstruct_scale = F10_F64[f];
        let reconstruct_inv = IF10_F64[e];
        let scale_e_v = _mm256_set1_pd(scale_e);
        let inv_f_v = _mm256_set1_pd(inv_f);
        let recon_scale_v = _mm256_set1_pd(reconstruct_scale);
        let recon_inv_v = _mm256_set1_pd(reconstruct_inv);

        while i + 4 <= n {
            // SAFETY: `i + 4 <= n == words.len() <= encoded.len()`.
            let raw_w = unsafe { _mm256_loadu_si256(words.as_ptr().add(i).cast()) };
            let v = _mm256_castsi256_pd(raw_w);
            let scaled = _mm256_mul_pd(_mm256_mul_pd(v, scale_e_v), inv_f_v);
            let abs_scaled = _mm256_andnot_pd(sign_mask, scaled);
            let in_range = _mm256_cmp_pd::<_CMP_LT_OQ>(abs_scaled, limit_v);
            if _mm256_movemask_pd(in_range) == 0xF {
                let m = _mm256_add_pd(scaled, magic_pd);
                let rounded = _mm256_sub_pd(m, magic_pd);
                let recon = _mm256_mul_pd(_mm256_mul_pd(rounded, recon_scale_v), recon_inv_v);
                let eq = _mm256_cmpeq_epi64(_mm256_castpd_si256(recon), raw_w);
                if _mm256_movemask_pd(_mm256_castsi256_pd(eq)) == 0xF {
                    let enc_v = _mm256_sub_epi64(_mm256_castpd_si256(m), magic_i64);
                    // SAFETY: `encoded` has at least `n` initialized elements.
                    unsafe { _mm256_storeu_si256(encoded.as_mut_ptr().add(i).cast(), enc_v) };
                    if first_encodable.is_none() {
                        first_encodable = Some(encoded[i]);
                    }
                    prev = encoded[i + 3];
                    i += 4;
                    continue;
                }
            }
            for (off, &w) in words[i..i + 4].iter().enumerate() {
                let j = i + off;
                let vf = f64::from_bits(w);
                let sc = (vf * scale_e) * inv_f;
                let enc = sc.round_ties_even() as i64;
                let ok = sc.is_finite()
                    && (((enc as f64) * reconstruct_scale) * reconstruct_inv).to_bits() == w;
                record_slot_64(
                    j,
                    w,
                    enc,
                    ok,
                    encoded,
                    &mut first_encodable,
                    &mut prev,
                    exception_positions,
                    exception_bits,
                );
            }
            i += 4;
        }
        for (j, &w) in words[i..n].iter().enumerate() {
            let idx = i + j;
            let vf = f64::from_bits(w);
            let sc = (vf * scale_e) * inv_f;
            let enc = sc.round_ties_even() as i64;
            let ok = sc.is_finite()
                && (((enc as f64) * reconstruct_scale) * reconstruct_inv).to_bits() == w;
            record_slot_64(
                idx,
                w,
                enc,
                ok,
                encoded,
                &mut first_encodable,
                &mut prev,
                exception_positions,
                exception_bits,
            );
        }
    }

    first_encodable
}

#[cfg(feature = "bitpacking")]
fn encode_chunk_f32_into(
    words: &[u32],
    params: AlpParams,
    scratch: &mut EncodeScratch<i32, u32>,
    buf0: &mut Vec<u32>,
    buf1: &mut Vec<u8>,
    buf2: &mut Vec<u8>,
    buf3: &mut Vec<u8>,
) -> Result<Option<[u32; 4]>> {
    let n = words.len();
    scratch.prepare(n);

    #[cfg(target_arch = "x86_64")]
    let first_opt = if CpuFeatures::host().contains(CpuFeatures::AVX2) {
        // SAFETY: `CpuFeatures::host().contains(CpuFeatures::AVX2)` verified AVX2 support.
        unsafe {
            transform_chunk_f32_avx2(
                words,
                params,
                &mut scratch.encoded[..n],
                &mut scratch.exception_positions,
                &mut scratch.exception_bits,
            )
        }
    } else {
        transform_chunk_f32_scalar(
            words,
            params,
            &mut scratch.encoded[..n],
            &mut scratch.exception_positions,
            &mut scratch.exception_bits,
        )
    };
    #[cfg(not(target_arch = "x86_64"))]
    let first_opt = transform_chunk_f32_scalar(
        words,
        params,
        &mut scratch.encoded[..n],
        &mut scratch.exception_positions,
        &mut scratch.exception_bits,
    );

    let Some(first) = first_opt else {
        return Ok(None);
    };
    if !scratch.exception_positions.is_empty() {
        backfill_exceptions(&mut scratch.encoded[..n], &scratch.exception_positions);
        if scratch.exception_positions.first() == Some(&0) {
            scratch.encoded[0] = first;
        }
    }
    pack_chunk_f32_into(n, params, scratch, buf0, buf1, buf2, buf3)
}

#[cfg(feature = "bitpacking")]
fn encode_chunk_f64_into(
    words: &[u64],
    params: AlpParams,
    scratch: &mut EncodeScratch<i64, u64>,
    buf0: &mut Vec<u64>,
    buf1: &mut Vec<u8>,
    buf2: &mut Vec<u8>,
    buf3: &mut Vec<u8>,
) -> Result<Option<[u32; 4]>> {
    let n = words.len();
    scratch.prepare(n);

    #[cfg(target_arch = "x86_64")]
    let first_opt = if CpuFeatures::host().contains(CpuFeatures::AVX2) {
        // SAFETY: `CpuFeatures::host().contains(CpuFeatures::AVX2)` verified AVX2 support.
        unsafe {
            transform_chunk_f64_avx2(
                words,
                params,
                &mut scratch.encoded[..n],
                &mut scratch.exception_positions,
                &mut scratch.exception_bits,
            )
        }
    } else {
        transform_chunk_f64_scalar(
            words,
            params,
            &mut scratch.encoded[..n],
            &mut scratch.exception_positions,
            &mut scratch.exception_bits,
        )
    };
    #[cfg(not(target_arch = "x86_64"))]
    let first_opt = transform_chunk_f64_scalar(
        words,
        params,
        &mut scratch.encoded[..n],
        &mut scratch.exception_positions,
        &mut scratch.exception_bits,
    );

    let Some(first) = first_opt else {
        return Ok(None);
    };
    if !scratch.exception_positions.is_empty() {
        backfill_exceptions(&mut scratch.encoded[..n], &scratch.exception_positions);
        if scratch.exception_positions.first() == Some(&0) {
            scratch.encoded[0] = first;
        }
    }
    pack_chunk_f64_into(n, params, scratch, buf0, buf1, buf2, buf3)
}

/// Backfill exception slots with the preceding encodable integer so Delta
/// streams do not see artificial spikes.
#[cfg(feature = "bitpacking")]
fn backfill_exceptions<T: Copy>(encoded: &mut [T], exception_positions: &[u16]) {
    let mut running = encoded[0];
    let mut exc = 0usize;
    for (i, e) in encoded.iter_mut().enumerate() {
        if exception_positions.get(exc) == Some(&(i as u16)) {
            *e = running;
            exc += 1;
        } else {
            running = *e;
        }
    }
    debug_assert_eq!(exc, exception_positions.len());
}

/// Pack the cheaper of the FOR and Delta offset streams directly into the page
/// output buffers. Only the winning stream is bit-packed.
#[cfg(feature = "bitpacking")]
fn pack_chunk_f32_into(
    n: usize,
    params: AlpParams,
    scratch: &mut EncodeScratch<i32, u32>,
    buf0: &mut Vec<u32>,
    buf1: &mut Vec<u8>,
    buf2: &mut Vec<u8>,
    buf3: &mut Vec<u8>,
) -> Result<Option<[u32; 4]>> {
    if n == 0 {
        return Ok(None);
    }
    let encoded = &scratch.encoded[..n];
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    for &v in encoded {
        min = min.min(v);
        max = max.max(v);
    }
    let for_range = (max as i64 - min as i64) as u64;
    let for_bit_width = (64 - for_range.leading_zeros()) as u8;
    if for_bit_width >= 32 {
        return Ok(None);
    }
    let for_width = for_bit_width.max(1);

    let mut delta_min = 0i32;
    let mut delta_bit_width = 0u8;
    let mut delta_usable = false;
    if n > 1 {
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        for pair in encoded.windows(2) {
            // Cannot overflow i32 because `max - min < 2^31`.
            let delta = pair[1] - pair[0];
            lo = lo.min(delta);
            hi = hi.max(delta);
        }
        let delta_range = (hi as i64 - lo as i64) as u64;
        let width = (64 - delta_range.leading_zeros()) as u8;
        if width < 32 {
            delta_min = lo;
            delta_bit_width = width.max(1);
            delta_usable = true;
        }
    }

    let for_len = out_of_line_bytes(n, for_width as usize, 32) + 8;
    let delta_len = if delta_usable {
        out_of_line_bytes(n - 1, delta_bit_width as usize, 32) + 12
    } else {
        usize::MAX
    };

    let pos_bytes = scratch.exception_positions.len() * 2;
    let exc_bytes = scratch.exception_bits.len();
    let winning_len = for_len.min(delta_len);
    if winning_len + pos_bytes + exc_bytes > MAX_MINIBLOCK_BYTES as usize {
        return Ok(None);
    }

    let buf0_before = buf0.len();
    let header_bytes = if for_len <= delta_len {
        let min_u32 = min as u32;
        for (&v, dst) in encoded.iter().zip(&mut scratch.offsets[..n]) {
            *dst = (v as u32).wrapping_sub(min_u32);
        }
        bitpack_out_of_line_slice_into::<u32>(&scratch.offsets[..n], for_width as usize, buf0);
        let mut header = [0u8; 8];
        header[0] = if params.use_div { MODE_DIV } else { 0 };
        header[1] = params.e;
        header[2] = params.f;
        header[3] = for_width;
        header[4..8].copy_from_slice(&min.to_le_bytes());
        buf1.extend_from_slice(&header);
        8usize
    } else {
        let first = encoded[0];
        let delta_min_u32 = delta_min as u32;
        for ((&curr, &next), dst) in encoded[..n - 1]
            .iter()
            .zip(&encoded[1..n])
            .zip(&mut scratch.offsets[..n - 1])
        {
            *dst = (next as u32)
                .wrapping_sub(curr as u32)
                .wrapping_sub(delta_min_u32);
        }
        bitpack_out_of_line_slice_into::<u32>(
            &scratch.offsets[..n - 1],
            delta_bit_width as usize,
            buf0,
        );
        let mut header = [0u8; 12];
        header[0] = MODE_DELTA | if params.use_div { MODE_DIV } else { 0 };
        header[1] = params.e;
        header[2] = params.f;
        header[3] = delta_bit_width;
        header[4..8].copy_from_slice(&first.to_le_bytes());
        header[8..12].copy_from_slice(&delta_min.to_le_bytes());
        buf1.extend_from_slice(&header);
        12usize
    };

    let delta_bytes = (buf0.len() - buf0_before) * 4;
    debug_assert_eq!(delta_bytes + header_bytes, winning_len);
    #[cfg(target_endian = "little")]
    buf2.extend_from_slice(bytemuck::cast_slice(&scratch.exception_positions));
    #[cfg(not(target_endian = "little"))]
    for &pos in &scratch.exception_positions {
        buf2.extend_from_slice(&pos.to_le_bytes());
    }
    buf3.extend_from_slice(&scratch.exception_bits);

    Ok(Some([
        delta_bytes as u32,
        header_bytes as u32,
        pos_bytes as u32,
        exc_bytes as u32,
    ]))
}

/// Pack the cheaper of the FOR and Delta offset streams directly into the page
/// output buffers.
#[cfg(feature = "bitpacking")]
fn pack_chunk_f64_into(
    n: usize,
    params: AlpParams,
    scratch: &mut EncodeScratch<i64, u64>,
    buf0: &mut Vec<u64>,
    buf1: &mut Vec<u8>,
    buf2: &mut Vec<u8>,
    buf3: &mut Vec<u8>,
) -> Result<Option<[u32; 4]>> {
    if n == 0 {
        return Ok(None);
    }
    let encoded = &scratch.encoded[..n];
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    for &v in encoded {
        min = min.min(v);
        max = max.max(v);
    }
    let for_bit_width = bits_required_u128((max as i128 - min as i128) as u128);
    if for_bit_width >= 63 {
        return Ok(None);
    }
    let for_width = for_bit_width.max(1);

    let mut delta_min = 0i64;
    let mut delta_bit_width = 0u8;
    let mut delta_usable = false;
    if n > 1 {
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        for pair in encoded.windows(2) {
            // Cannot overflow i64 because `max - min < 2^62`.
            let delta = pair[1] - pair[0];
            lo = lo.min(delta);
            hi = hi.max(delta);
        }
        let delta_range = (hi as u64).wrapping_sub(lo as u64);
        let width = (64 - delta_range.leading_zeros()) as u8;
        if width < 64 {
            delta_min = lo;
            delta_bit_width = width.max(1);
            delta_usable = true;
        }
    }

    let for_len = out_of_line_bytes(n, for_width as usize, 64) + 12;
    let delta_len = if delta_usable {
        out_of_line_bytes(n - 1, delta_bit_width as usize, 64) + 20
    } else {
        usize::MAX
    };

    let pos_bytes = scratch.exception_positions.len() * 2;
    let exc_bytes = scratch.exception_bits.len();
    let winning_len = for_len.min(delta_len);
    if winning_len + pos_bytes + exc_bytes > MAX_MINIBLOCK_BYTES as usize {
        return Ok(None);
    }

    let buf0_before = buf0.len();
    let header_bytes = if for_len <= delta_len {
        let min_u64 = min as u64;
        for (&v, dst) in encoded.iter().zip(&mut scratch.offsets[..n]) {
            *dst = (v as u64).wrapping_sub(min_u64);
        }
        bitpack_out_of_line_slice_into::<u64>(&scratch.offsets[..n], for_width as usize, buf0);
        let mut header = [0u8; 12];
        header[0] = if params.use_div { MODE_DIV } else { 0 };
        header[1] = params.e;
        header[2] = params.f;
        header[3] = for_width;
        header[4..12].copy_from_slice(&min.to_le_bytes());
        buf1.extend_from_slice(&header);
        12usize
    } else {
        let first = encoded[0];
        let delta_min_u64 = delta_min as u64;
        for ((&curr, &next), dst) in encoded[..n - 1]
            .iter()
            .zip(&encoded[1..n])
            .zip(&mut scratch.offsets[..n - 1])
        {
            *dst = (next as u64)
                .wrapping_sub(curr as u64)
                .wrapping_sub(delta_min_u64);
        }
        bitpack_out_of_line_slice_into::<u64>(
            &scratch.offsets[..n - 1],
            delta_bit_width as usize,
            buf0,
        );
        let mut header = [0u8; 20];
        header[0] = MODE_DELTA | if params.use_div { MODE_DIV } else { 0 };
        header[1] = params.e;
        header[2] = params.f;
        header[3] = delta_bit_width;
        header[4..12].copy_from_slice(&first.to_le_bytes());
        header[12..20].copy_from_slice(&delta_min.to_le_bytes());
        buf1.extend_from_slice(&header);
        20usize
    };

    let delta_bytes = (buf0.len() - buf0_before) * 8;
    debug_assert_eq!(delta_bytes + header_bytes, winning_len);
    #[cfg(target_endian = "little")]
    buf2.extend_from_slice(bytemuck::cast_slice(&scratch.exception_positions));
    #[cfg(not(target_endian = "little"))]
    for &pos in &scratch.exception_positions {
        buf2.extend_from_slice(&pos.to_le_bytes());
    }
    buf3.extend_from_slice(&scratch.exception_bits);

    Ok(Some([
        delta_bytes as u32,
        header_bytes as u32,
        pos_bytes as u32,
        exc_bytes as u32,
    ]))
}

/// ALP-RD page fallback for `f64` pages where the scalar transform does not
/// pay off (typically random-looking mantissas).
///
/// Each chunk splits every value into a high dictionary part (`bits >> d`)
/// and a low random part (`bits & mask`). The dictionary is stored verbatim
/// and the per-value indices plus right parts are bit-packed. The split is
/// lossless by construction and each chunk picks the `d` with the smallest
/// packed size, always staying below the raw size and chunk budget.
#[cfg(feature = "bitpacking")]
fn encode_page_f64_rd(
    words: &[u64],
    max_chunk: usize,
    raw_size: usize,
) -> Result<Option<(MiniBlockCompressed, CompressiveEncoding)>> {
    let num_chunks = words.len().div_ceil(max_chunk);
    let mut buf0: Vec<u64> = Vec::with_capacity(words.len() / 2);
    let mut buf1: Vec<u8> = Vec::with_capacity(num_chunks * 8);
    let mut buf2: Vec<u8> = Vec::with_capacity(num_chunks * 128);
    let mut buf3: Vec<u64> = Vec::with_capacity(words.len() / 8);
    let mut chunks = Vec::with_capacity(num_chunks);
    let mut compressed_size = 0usize;
    let mut scratch = vec![0u64; max_chunk];

    for chunk in words.chunks(max_chunk) {
        let Some(sizes) = encode_chunk_f64_rd(
            chunk,
            &mut scratch,
            &mut buf0,
            &mut buf1,
            &mut buf2,
            &mut buf3,
        ) else {
            return Ok(None);
        };
        compressed_size +=
            sizes[0] as usize + sizes[1] as usize + sizes[2] as usize + sizes[3] as usize;

        chunks.push(MiniBlockChunk {
            buffer_sizes: sizes.to_vec(),
            log_num_values: if chunk.len() < max_chunk {
                0
            } else {
                chunk.len().ilog2() as u8
            },
        });
    }

    if compressed_size >= raw_size {
        return Ok(None);
    }
    Ok(Some((
        MiniBlockCompressed {
            data: vec![
                LanceBuffer::reinterpret_vec(buf0),
                LanceBuffer::from(buf1),
                LanceBuffer::from(buf2),
                LanceBuffer::reinterpret_vec(buf3),
            ],
            chunks,
            num_values: words.len() as u64,
        },
        EncodingUtils::alp(64),
    )))
}

/// Encode one chunk with the best ALP-RD split directly into the page buffers,
/// or `None` when no split beats the raw representation within the chunk budget.
#[cfg(feature = "bitpacking")]
fn encode_chunk_f64_rd(
    words: &[u64],
    scratch: &mut Vec<u64>,
    buf0: &mut Vec<u64>,
    buf1: &mut Vec<u8>,
    buf2: &mut Vec<u8>,
    buf3: &mut Vec<u64>,
) -> Option<[u32; 4]> {
    let n = words.len();
    let raw_bytes = n * 8;
    if n == 0 {
        return None;
    }

    // Sort `words` once: for unsigned u64, `a <= b => (a >> d) <= (b >> d)` for every
    // shift `d`, so each `RD_SWEEP` width can count distinct prefixes in a linear pass
    // with zero heap allocations instead of sorting and deduplicating 17 times.
    let mut stack_sorted = [0u64; 1024];
    let mut heap_sorted: Vec<u64>;
    let sorted: &mut [u64] = if n <= stack_sorted.len() {
        stack_sorted[..n].copy_from_slice(words);
        &mut stack_sorted[..n]
    } else {
        heap_sorted = words.to_vec();
        heap_sorted.as_mut_slice()
    };
    sorted.sort_unstable();

    let mut best: Option<(u8, u8, usize)> = None; // (d, index_bits, dict_len)
    let mut best_cost = usize::MAX;
    for &d in &RD_SWEEP {
        let mut dict_len = 1usize;
        let mut prev = sorted[0] >> d;
        for &w in &sorted[1..] {
            let cur = w >> d;
            dict_len += usize::from(cur != prev);
            prev = cur;
        }
        let index_bits = bits_required_u128(dict_len as u128 - 1);
        let right_bytes = out_of_line_bytes(n, d as usize, 64);
        let index_bytes = if index_bits == 0 {
            0
        } else {
            out_of_line_bytes(n, index_bits as usize, 64)
        };
        let cost = right_bytes + index_bytes + dict_len * 8 + 8;
        if cost < raw_bytes && cost <= MAX_MINIBLOCK_BYTES as usize && cost < best_cost {
            best_cost = cost;
            best = Some((d, index_bits, dict_len));
        }
    }

    let (d, index_bits, dict_len) = best?;
    if d >= 64 || index_bits >= 64 {
        return None;
    }
    let mask = (1u64 << d) - 1;
    let high_mask = !mask;

    // Build the sorted dictionary (`(w >> d) << d == w & high_mask`) in one pass over `sorted`.
    let mut stack_dict = [0u64; 1024];
    let mut heap_dict: Vec<u64>;
    let dict: &[u64] = if dict_len <= stack_dict.len() {
        let mut count = 1usize;
        let mut prev_high = sorted[0] & high_mask;
        stack_dict[0] = prev_high;
        for &w in &sorted[1..] {
            let cur_high = w & high_mask;
            if cur_high != prev_high {
                stack_dict[count] = cur_high;
                count += 1;
                prev_high = cur_high;
            }
        }
        debug_assert_eq!(count, dict_len);
        &stack_dict[..dict_len]
    } else {
        heap_dict = Vec::with_capacity(dict_len);
        let mut prev_high = sorted[0] & high_mask;
        heap_dict.push(prev_high);
        for &w in &sorted[1..] {
            let cur_high = w & high_mask;
            if cur_high != prev_high {
                heap_dict.push(cur_high);
                prev_high = cur_high;
            }
        }
        debug_assert_eq!(heap_dict.len(), dict_len);
        heap_dict.as_slice()
    };

    if scratch.len() < n {
        scratch.resize(n, 0);
    }
    for (&w, dst) in words.iter().zip(&mut scratch[..n]) {
        *dst = w & mask;
    }
    let buf0_before = buf0.len();
    bitpack_out_of_line_slice_into::<u64>(&scratch[..n], d as usize, buf0);
    let right_len = (buf0.len() - buf0_before) * 8;

    let buf3_before = buf3.len();
    if index_bits > 0 {
        if dict_len <= 1024 {
            // Open-addressed stack hash table (load factor <= 0.5 since `dict_len <= 1024`
            // for 8 KiB chunks), avoiding `HashMap` heap allocation and SipHash per value.
            const EMPTY: u16 = u16::MAX;
            const HASH_MUL: u64 = 0x9E37_79B9_7F4A_7C15;
            let mut slots = [EMPTY; 2048];
            for (i, &high) in dict.iter().enumerate() {
                let mut slot = (((high >> d).wrapping_mul(HASH_MUL)) >> 53) as usize;
                while slots[slot] != EMPTY {
                    slot = (slot + 1) & 2047;
                }
                slots[slot] = i as u16;
            }
            for (&w, dst) in words.iter().zip(&mut scratch[..n]) {
                let high = w & high_mask;
                let mut slot = (((w >> d).wrapping_mul(HASH_MUL)) >> 53) as usize;
                *dst = loop {
                    let candidate = slots[slot] as usize;
                    if dict[candidate] == high {
                        break candidate as u64;
                    }
                    slot = (slot + 1) & 2047;
                };
            }
        } else {
            for (&w, dst) in words.iter().zip(&mut scratch[..n]) {
                *dst = dict.binary_search(&(w & high_mask)).unwrap() as u64;
            }
        }
        bitpack_out_of_line_slice_into::<u64>(&scratch[..n], index_bits as usize, buf3);
    }
    let index_len = (buf3.len() - buf3_before) * 8;

    #[cfg(target_endian = "little")]
    buf2.extend_from_slice(bytemuck::cast_slice::<u64, u8>(dict));
    #[cfg(not(target_endian = "little"))]
    for &high in dict {
        buf2.extend_from_slice(&high.to_le_bytes());
    }
    let dict_bytes_len = dict.len() * 8;

    let mut header = [0u8; 8];
    header[0] = MODE_RD;
    header[1] = d;
    header[2] = index_bits;
    header[3] = 0;
    header[4..8].copy_from_slice(&(dict.len() as u32).to_le_bytes());
    buf1.extend_from_slice(&header);

    debug_assert_eq!(right_len + 8 + dict_bytes_len + index_len, best_cost);
    Some([right_len as u32, 8, dict_bytes_len as u32, index_len as u32])
}

#[cfg(not(feature = "bitpacking"))]
fn encode_page_f64_rd(
    _words: &[u64],
    _max_chunk: usize,
    _raw_size: usize,
) -> Result<Option<(MiniBlockCompressed, CompressiveEncoding)>> {
    Ok(None)
}

/// Number of payload bytes `bitpack_out_of_line_slice_into` produces for `num_items`
/// values at `bit_width`. Mirrors its tail padding decisions exactly.
#[cfg(feature = "bitpacking")]
fn out_of_line_bytes(num_items: usize, bit_width: usize, bits_per_value: usize) -> usize {
    const ELEMS_PER_CHUNK: usize = 1024;
    let words_per_chunk = (ELEMS_PER_CHUNK * bit_width).div_ceil(bits_per_value);
    let num_chunks = num_items.div_ceil(ELEMS_PER_CHUNK);
    let last_chunk_is_runt = !num_items.is_multiple_of(ELEMS_PER_CHUNK);
    let mut words = (num_chunks - usize::from(last_chunk_is_runt)) * words_per_chunk;
    if last_chunk_is_runt {
        let remaining = num_items - (num_chunks - 1) * ELEMS_PER_CHUNK;
        let padding_cost = bit_width * (ELEMS_PER_CHUNK - remaining);
        let tail_pack_savings = bits_per_value.saturating_sub(bit_width) * remaining;
        if padding_cost < tail_pack_savings {
            words += words_per_chunk;
        } else {
            words += remaining;
        }
    }
    words * bits_per_value / 8
}

/// Decodes values encoded by [`AlpMiniBlockEncoder`].
#[derive(Debug)]
pub struct AlpMiniBlockDecompressor {
    bits_per_value: u64,
}

impl AlpMiniBlockDecompressor {
    pub fn from_description(desc: &metadata::Alp) -> Result<Self> {
        let bits_per_value = desc.bits_per_value as u64;
        if bits_per_value != 32 && bits_per_value != 64 {
            return Err(Error::invalid_input("ALP bits_per_value must be 32 or 64"));
        }
        Ok(Self { bits_per_value })
    }
}

/// Header parsed from buffer 1. The two base values are interpreted by mode.
struct ChunkHeader {
    mode: u8,
    e: u8,
    f: u8,
    bit_width: u8,
    base: i64,
    delta_base: i64,
}

fn parse_header(header: &[u8], bits_per_value: u64) -> Result<ChunkHeader> {
    let value_bytes = (bits_per_value / 8) as usize;
    let (fixed_parts, var_parts) = match bits_per_value {
        32 => (4usize, 8usize),
        64 => (8usize, 12usize),
        _ => unreachable!(),
    };
    if header.len() < 4 {
        return Err(Error::invalid_input("ALP header is too short"));
    }
    let mode = header[0];
    let delta = mode & MODE_DELTA != 0;
    let expected = if delta {
        fixed_parts + var_parts
    } else {
        var_parts
    };
    if header.len() != expected {
        return Err(Error::invalid_input(format!(
            "ALP header buffer size {} does not match expected {expected}",
            header.len()
        )));
    }

    let e = header[1];
    let f = header[2];
    let bit_width = header[3];
    let max_exp = match bits_per_value {
        32 => MAX_EXP_F32,
        64 => MAX_EXP_F64,
        _ => unreachable!(),
    };
    if e as usize > max_exp as usize || f as usize > max_exp as usize {
        return Err(Error::invalid_input("ALP exponents out of range"));
    }
    if mode & MODE_DIV == 0 && f >= e {
        return Err(Error::invalid_input(
            "ALP multiply mode requires exponent_f < exponent_e",
        ));
    }
    let max_bit_width = if bits_per_value == 32 { 31 } else { 63 };
    if bit_width as usize > max_bit_width {
        return Err(Error::invalid_input("ALP bit width out of range"));
    }

    let mut offset = 4usize;
    let read_base = |offset: &mut usize| -> i64 {
        let mut bytes = [0u8; 8];
        bytes[..value_bytes].copy_from_slice(&header[*offset..*offset + value_bytes]);
        *offset += value_bytes;
        if value_bytes == 4 {
            i32::from_le_bytes(bytes[..4].try_into().unwrap()) as i64
        } else {
            i64::from_le_bytes(bytes)
        }
    };
    let base = read_base(&mut offset);
    let delta_base = if delta { read_base(&mut offset) } else { 0 };

    Ok(ChunkHeader {
        mode,
        e,
        f,
        bit_width,
        base,
        delta_base,
    })
}

#[cfg(feature = "bitpacking")]
fn validate_packed_buffer(
    name: &str,
    buffer: &LanceBuffer,
    num_values: u64,
    word_bits: u64,
    bit_width: u8,
) -> Result<()> {
    let [raw_tail_bytes, padded_tail_bytes] =
        out_of_line_payload_byte_lengths(num_values, word_bits, bit_width as u64)?;
    if buffer.len() as u64 != raw_tail_bytes && buffer.len() as u64 != padded_tail_bytes {
        return Err(Error::corrupt_file_named(
            name,
            format!(
                "packed payload has {} bytes, expected {raw_tail_bytes} bytes (raw tail) or \
                 {padded_tail_bytes} bytes (padded tail) for {num_values} values at \
                 {bit_width} bits per value in {word_bits}-bit words",
                buffer.len()
            ),
        ));
    }
    Ok(())
}

impl MiniBlockDecompressor for AlpMiniBlockDecompressor {
    fn num_buffers(&self) -> usize {
        4
    }

    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        if num_values == 0 {
            return Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::empty(),
                bits_per_value: self.bits_per_value,
                num_values: 0,
                block_info: BlockInfo::new(),
            }));
        }

        if data.len() != 4 {
            return Err(Error::invalid_input(format!(
                "ALP decompression expects 4 buffers, got {}",
                data.len()
            )));
        }

        let n = usize::try_from(num_values)
            .map_err(|_| Error::invalid_input("ALP chunk too large for usize"))?;
        let mut iter = data.into_iter();
        let packed = iter.next().unwrap();
        let header = iter.next().unwrap();
        let buffer2 = iter.next().unwrap();
        let buffer3 = iter.next().unwrap();

        if header.as_ref().first() == Some(&MODE_RD) {
            if self.bits_per_value != 64 {
                return Err(Error::invalid_input(
                    "ALP-RD is only defined for 64-bit values",
                ));
            }
            let rd = parse_rd_header(header.as_ref())?;
            #[cfg(feature = "bitpacking")]
            {
                validate_packed_buffer("ALP-RD right parts", &packed, num_values, 64, rd.d)?;
                if rd.index_bits > 0 {
                    validate_packed_buffer(
                        "ALP-RD indices",
                        &buffer3,
                        num_values,
                        64,
                        rd.index_bits,
                    )?;
                }
                return self.decode_rd64(&rd, packed, buffer2, buffer3, n);
            }
            #[cfg(not(feature = "bitpacking"))]
            {
                let _ = (packed, buffer2, buffer3);
                return Err(Error::not_supported("ALP requires bitpacking support"));
            }
        }

        let header = parse_header(header.as_ref(), self.bits_per_value)?;

        #[cfg(feature = "bitpacking")]
        {
            // Delta stores the first value in the header, leaving n - 1 offsets.
            let offset_count = if header.mode & MODE_DELTA != 0 {
                num_values - 1
            } else {
                num_values
            };
            validate_packed_buffer(
                "ALP offsets",
                &packed,
                offset_count,
                self.bits_per_value,
                header.bit_width,
            )?;
        }

        let mut out = vec![0u8; n * (self.bits_per_value as usize / 8)];
        #[cfg(feature = "bitpacking")]
        match self.bits_per_value {
            32 => self.decode_chunk32(&header, packed, n, &mut out)?,
            64 => self.decode_chunk64(&header, packed, n, &mut out)?,
            _ => unreachable!(),
        }
        #[cfg(not(feature = "bitpacking"))]
        {
            let _ = (&header, packed);
            return Err(Error::not_supported("ALP requires bitpacking support"));
        }

        apply_exceptions(self.bits_per_value, &buffer2, &buffer3, n, &mut out)?;

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(out),
            bits_per_value: self.bits_per_value,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

/// ALP-RD chunk header: `[mode][d][index_bits][0][dict_len u32]`.
struct RdHeader {
    d: u8,
    index_bits: u8,
    dict_len: usize,
}

fn parse_rd_header(header: &[u8]) -> Result<RdHeader> {
    if header.len() != 8 {
        return Err(Error::invalid_input(format!(
            "ALP-RD header buffer size {} does not match expected 8",
            header.len()
        )));
    }
    if header[0] != MODE_RD {
        return Err(Error::invalid_input("ALP-RD header mode mismatch"));
    }
    let d = header[1];
    let index_bits = header[2];
    if d == 0 || d > 63 {
        return Err(Error::invalid_input("ALP-RD right width out of range"));
    }
    if index_bits > 31 {
        return Err(Error::invalid_input("ALP-RD index width out of range"));
    }
    let dict_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    if dict_len == 0 {
        return Err(Error::invalid_input("ALP-RD dictionary is empty"));
    }
    if index_bits > 0 && bits_required_u128(dict_len as u128 - 1) > index_bits {
        return Err(Error::invalid_input("ALP-RD index width too small"));
    }
    Ok(RdHeader {
        d,
        index_bits,
        dict_len,
    })
}

#[cfg(feature = "bitpacking")]
impl AlpMiniBlockDecompressor {
    fn decode_chunk32(
        &self,
        header: &ChunkHeader,
        packed: LanceBuffer,
        n: usize,
        out: &mut [u8],
    ) -> Result<()> {
        let delta = header.mode & MODE_DELTA != 0;
        let offset_count = if delta { n.saturating_sub(1) } else { n };
        let offsets_block = if offset_count == 0 {
            None
        } else {
            let block = FixedWidthDataBlock {
                data: packed,
                bits_per_value: 32,
                num_values: n as u64,
                block_info: BlockInfo::new(),
            };
            Some(unpack_out_of_line::<u32>(
                block,
                offset_count,
                header.bit_width as usize,
            ))
        };
        let offsets_buf = offsets_block
            .as_ref()
            .map(|b| b.data.borrow_to_typed_view::<u32>());
        let offsets: &[u32] = offsets_buf.as_deref().unwrap_or(&[]);

        if delta {
            restore::restore_delta_chunk32(
                offsets,
                header.base,
                header.delta_base,
                header.mode & MODE_DIV != 0,
                header.e as usize,
                header.f as usize,
                out,
            );
        } else {
            restore::restore_chunk32(
                offsets,
                header.base,
                header.mode & MODE_DIV != 0,
                header.e as usize,
                header.f as usize,
                out,
            );
        }
        Ok(())
    }

    fn decode_chunk64(
        &self,
        header: &ChunkHeader,
        packed: LanceBuffer,
        n: usize,
        out: &mut [u8],
    ) -> Result<()> {
        let delta = header.mode & MODE_DELTA != 0;
        let offset_count = if delta { n.saturating_sub(1) } else { n };
        let offsets_block = if offset_count == 0 {
            None
        } else {
            let block = FixedWidthDataBlock {
                data: packed,
                bits_per_value: 64,
                num_values: n as u64,
                block_info: BlockInfo::new(),
            };
            Some(unpack_out_of_line::<u64>(
                block,
                offset_count,
                header.bit_width as usize,
            ))
        };
        let offsets_buf = offsets_block
            .as_ref()
            .map(|b| b.data.borrow_to_typed_view::<u64>());
        let offsets: &[u64] = offsets_buf.as_deref().unwrap_or(&[]);

        if delta {
            restore::restore_delta_chunk64(
                offsets,
                header.base,
                header.delta_base,
                header.mode & MODE_DIV != 0,
                header.e as usize,
                header.f as usize,
                out,
            );
        } else {
            restore::restore_chunk64(
                offsets,
                header.base,
                header.mode & MODE_DIV != 0,
                header.e as usize,
                header.f as usize,
                out,
            );
        }
        Ok(())
    }

    fn decode_rd64(
        &self,
        header: &RdHeader,
        packed: LanceBuffer,
        dict_buffer: LanceBuffer,
        index_buffer: LanceBuffer,
        n: usize,
    ) -> Result<DataBlock> {
        if dict_buffer.len() != header.dict_len * 8 {
            return Err(Error::invalid_input(format!(
                "ALP-RD dictionary buffer size {} does not match dict_len {}",
                dict_buffer.len(),
                header.dict_len
            )));
        }

        let mut out: Vec<u64> = if n == 0 {
            Vec::new()
        } else {
            let rights_block = unpack_out_of_line::<u64>(
                FixedWidthDataBlock {
                    data: packed,
                    bits_per_value: 64,
                    num_values: n as u64,
                    block_info: BlockInfo::new(),
                },
                n,
                header.d as usize,
            );
            rights_block
                .data
                .into_buffer()
                .into_vec::<u64>()
                .unwrap_or_else(|buf| {
                    LanceBuffer::from(buf)
                        .borrow_to_typed_view::<u64>()
                        .to_vec()
                })
        };

        let dict_view = dict_buffer.borrow_to_typed_view::<u64>();
        let dict_words = &dict_view[..header.dict_len];
        if header.index_bits == 0 {
            #[cfg(target_endian = "little")]
            let single = dict_words[0];
            #[cfg(not(target_endian = "little"))]
            let single = u64::from_le(dict_words[0]);
            for right in &mut out {
                *right |= single;
            }
        } else if n > 0 {
            let indices_block = unpack_out_of_line::<u64>(
                FixedWidthDataBlock {
                    data: index_buffer,
                    bits_per_value: 64,
                    num_values: n as u64,
                    block_info: BlockInfo::new(),
                },
                n,
                header.index_bits as usize,
            );
            let indices = indices_block.data.borrow_to_typed_view::<u64>();
            for (right, &idx) in out.iter_mut().zip(indices.iter()) {
                let Some(&raw_dict) = dict_words.get(idx as usize) else {
                    return Err(Error::invalid_input("ALP-RD index out of bounds"));
                };
                #[cfg(target_endian = "little")]
                let dict = raw_dict;
                #[cfg(not(target_endian = "little"))]
                let dict = u64::from_le(raw_dict);
                *right |= dict;
            }
        }

        #[cfg(not(target_endian = "little"))]
        for w in &mut out {
            *w = w.to_le();
        }

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(out),
            bits_per_value: 64,
            num_values: n as u64,
            block_info: BlockInfo::new(),
        }))
    }
}

fn apply_exceptions(
    bits_per_value: u64,
    exception_positions: &LanceBuffer,
    exception_bits: &LanceBuffer,
    n: usize,
    out: &mut [u8],
) -> Result<()> {
    let value_bytes = (bits_per_value / 8) as usize;
    let pos_bytes = exception_positions.as_ref();
    let exc_bytes = exception_bits.as_ref();
    if pos_bytes.len() % 2 != 0 {
        return Err(Error::invalid_input(
            "ALP exception positions not u16-aligned",
        ));
    }
    let count = pos_bytes.len() / 2;
    if exc_bytes.len() != count * value_bytes {
        return Err(Error::invalid_input("ALP exception values length mismatch"));
    }
    for i in 0..count {
        let pos = u16::from_le_bytes([pos_bytes[i * 2], pos_bytes[i * 2 + 1]]) as usize;
        if pos >= n {
            return Err(Error::invalid_input("ALP exception position out of range"));
        }
        let start = i * value_bytes;
        let end = start + value_bytes;
        out[pos * value_bytes..(pos + 1) * value_bytes].copy_from_slice(&exc_bytes[start..end]);
    }
    Ok(())
}

#[cfg(all(test, feature = "bitpacking"))]
mod tests {
    use super::*;
    use crate::statistics::ComputeStat;
    use rand::Rng;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rstest::rstest;

    #[cfg_attr(coverage, coverage(off))]
    fn packed_zeros(count: usize, bits: u64, width: usize, is_padded: bool) -> LanceBuffer {
        // Pack full blocks, then either pack a padded tail or append a raw tail.
        let packed_count = if is_padded {
            count.div_ceil(1024) * 1024
        } else {
            count / 1024 * 1024
        };
        let block = FixedWidthDataBlock {
            data: LanceBuffer::from(vec![0u8; packed_count * (bits as usize / 8)]),
            bits_per_value: bits,
            num_values: packed_count as u64,
            block_info: BlockInfo::new(),
        };
        let packed = match bits {
            32 => LanceBuffer::reinterpret_vec(bitpack_out_of_line_vec::<u32>(block, width)),
            64 => LanceBuffer::reinterpret_vec(bitpack_out_of_line_vec::<u64>(block, width)),
            _ => unreachable!(),
        };
        let mut bytes = packed.as_ref().to_vec();
        if !is_padded {
            bytes.resize(bytes.len() + (count % 1024) * (bits as usize / 8), 0);
        }
        LanceBuffer::from(bytes)
    }

    #[rstest]
    #[case::for32(32, false)]
    #[case::delta32(32, true)]
    #[case::for64(64, false)]
    #[case::delta64(64, true)]
    fn test_packed_offsets_buffer_validation(
        #[case] bits: u64,
        #[case] is_delta: bool,
        #[values(1, 17, 1024, 1025)] n: usize,
        #[values(false, true)] is_padded: bool,
    ) {
        let offset_count = n - usize::from(is_delta);
        let packed = packed_zeros(offset_count, bits, 3, is_padded);
        let mut header = vec![MODE_DIV | if is_delta { MODE_DELTA } else { 0 }, 1, 0, 3];
        let word_bytes = bits as usize / 8;
        header.extend_from_slice(&10i64.to_le_bytes()[..word_bytes]);
        if is_delta {
            header.extend_from_slice(&1i64.to_le_bytes()[..word_bytes]);
        }
        let buffers = vec![
            packed,
            LanceBuffer::from(header),
            LanceBuffer::empty(),
            LanceBuffer::empty(),
        ];
        let decompressor = AlpMiniBlockDecompressor {
            bits_per_value: bits,
        };
        let decoded = decompressor.decompress(buffers.clone(), n as u64).unwrap();
        let DataBlock::FixedWidth(decoded) = decoded else {
            panic!("expected fixed width");
        };
        let expected: Vec<u8> = (0..n)
            .flat_map(|i| {
                let value = 10 + if is_delta { i } else { 0 };
                if bits == 32 {
                    ((value as f32) / 10.0).to_le_bytes().to_vec()
                } else {
                    ((value as f64) / 10.0).to_le_bytes().to_vec()
                }
            })
            .collect();
        assert_eq!(decoded.data.as_ref(), expected);

        let len = buffers[0].len();
        let invalid_lengths = if len == 0 {
            vec![word_bytes]
        } else {
            vec![0, len / 2, len - 1, len - word_bytes, len + word_bytes]
        };
        for len in invalid_lengths {
            let mut damaged = buffers.clone();
            damaged[0] = LanceBuffer::from(vec![0u8; len]);
            let error = decompressor.decompress(damaged, n as u64).unwrap_err();
            assert!(matches!(error, Error::CorruptFile { .. }));
            assert!(error.to_string().contains("ALP offsets"), "{error}");
            assert!(error.to_string().contains("expected"), "{error}");
        }
    }

    #[rstest]
    fn test_rd_packed_buffer_validation(
        #[values(19, 1024, 1025)] n: usize,
        #[values(false, true)] is_padded: bool,
        #[values(0, 3)] buffer_index: usize,
    ) {
        let buffers = vec![
            packed_zeros(n, 64, 3, is_padded),
            LanceBuffer::from(vec![MODE_RD, 3, 1, 0, 2, 0, 0, 0]),
            LanceBuffer::reinterpret_vec(vec![1.0f64.to_bits(), (-1.0f64).to_bits()]),
            packed_zeros(n, 64, 1, is_padded),
        ];
        let decompressor = AlpMiniBlockDecompressor { bits_per_value: 64 };
        let decoded = decompressor.decompress(buffers.clone(), n as u64).unwrap();
        let DataBlock::FixedWidth(decoded) = decoded else {
            panic!("expected fixed width");
        };
        assert_eq!(
            decoded.data.borrow_to_typed_slice::<f64>().as_ref(),
            vec![1.0; n]
        );

        let len = buffers[buffer_index].len();
        for len in [0, len / 2, len - 1, len - 8, len + 8] {
            let mut damaged = buffers.clone();
            damaged[buffer_index] = LanceBuffer::from(vec![0u8; len]);
            let error = decompressor.decompress(damaged, n as u64).unwrap_err();
            assert!(matches!(error, Error::CorruptFile { .. }));
            let stream = if buffer_index == 0 {
                "ALP-RD right parts"
            } else {
                "ALP-RD indices"
            };
            assert!(error.to_string().contains(stream), "{error}");
            assert!(error.to_string().contains("expected"), "{error}");
        }
    }

    fn stat_block(values: &[f64]) -> FixedWidthDataBlock {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let mut block = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        block.compute_stat();
        block
    }

    #[test]
    fn test_auto_mode_gates_on_entropy() {
        // Decimal look-alike are low-entropy and pass the Auto gate; random
        // doubles are not.
        let mut rng = StdRng::seed_from_u64(3);
        let decimals: Vec<f64> = (0..1024)
            .map(|_| rng.random_range(0..1_000_000i64) as f64 / 100.0)
            .collect();
        // Wide-magnitude random values have near-uniform byte positions, well
        // above the Auto threshold even on the 64-value entropy sample.
        let random: Vec<f64> = (0..1024)
            .map(|_| (rng.random::<f64>() - 0.5) * 1e12)
            .collect();

        assert!(should_use_alp(&stat_block(&decimals), AlpMode::Auto));
        assert!(!should_use_alp(&stat_block(&random), AlpMode::Auto));
        assert!(should_use_alp(&stat_block(&random), AlpMode::On));
        assert!(!should_use_alp(&stat_block(&decimals), AlpMode::Off));
    }

    fn round_trip_f32(values: &[f32]) -> CompressiveEncoding {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let data = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 32,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        let encoder = AlpMiniBlockEncoder::new(32);
        let (compressed, encoding) = encoder
            .compress(
                MiniBlockCompressionContext::new(0, true, true),
                DataBlock::FixedWidth(data),
            )
            .unwrap();
        check_round_trip(compressed, &encoding, values, |v| v.to_bits() as u64);
        encoding
    }

    fn round_trip_f64(values: &[f64]) -> CompressiveEncoding {
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let data = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        let encoder = AlpMiniBlockEncoder::new(64);
        let (compressed, encoding) = encoder
            .compress(
                MiniBlockCompressionContext::new(0, true, true),
                DataBlock::FixedWidth(data),
            )
            .unwrap();
        check_round_trip(compressed, &encoding, values, |v| v.to_bits());
        encoding
    }

    fn check_round_trip<T: Copy>(
        compressed: MiniBlockCompressed,
        encoding: &CompressiveEncoding,
        expected: &[T],
        to_bits: impl Fn(T) -> u64,
    ) {
        let metadata::compressive_encoding::Compression::Alp(desc) =
            encoding.compression.as_ref().unwrap()
        else {
            // Flat fallback: no payload to decode here; the page-level
            // pipeline round-trips via the flat codec instead.
            assert_eq!(compressed.data.len(), 1);
            return;
        };
        assert!(matches!(desc.bits_per_value, 32 | 64));
        let decompressor = AlpMiniBlockDecompressor::from_description(desc).unwrap();
        assert_eq!(decompressor.num_buffers(), 4);

        let bits = desc.bits_per_value as u64;
        let mut vals_in_prev = 0u64;
        let mut buffer_offsets = vec![0usize; compressed.data.len()];
        let mut out_bits = Vec::with_capacity(expected.len());
        for chunk in &compressed.chunks {
            let chunk_vals = chunk.num_values(vals_in_prev, compressed.num_values);
            vals_in_prev += chunk_vals;
            let buffers = chunk
                .buffer_sizes
                .iter()
                .zip(compressed.data.iter().zip(buffer_offsets.iter_mut()))
                .map(|(sz, (buf, off))| {
                    let start = *off;
                    let end = start + *sz as usize;
                    *off = end;
                    buf.slice_with_length(start, *sz as usize)
                })
                .collect::<Vec<_>>();
            let decoded = decompressor.decompress(buffers, chunk_vals).unwrap();
            let DataBlock::FixedWidth(decoded) = decoded else {
                panic!("expected fixed width")
            };
            if bits == 32 {
                let words = decoded.data.borrow_to_typed_slice::<u32>();
                out_bits.extend(words.as_ref().iter().map(|b| *b as u64));
            } else {
                let words = decoded.data.borrow_to_typed_slice::<u64>();
                out_bits.extend(words.as_ref().iter().copied());
            }
        }
        assert_eq!(out_bits.len(), expected.len());
        for (i, (got, want)) in out_bits.iter().zip(expected.iter()).enumerate() {
            assert_eq!(*got, to_bits(*want), "value mismatch at index {i}");
        }
    }

    #[rstest]
    #[case::full_chunks(4096)]
    #[case::short_chunk(1000)]
    #[case::raw_tail(1025)]
    fn test_round_trip_decimal_f32_uses_alp(#[case] num_values: usize) {
        let mut values = Vec::with_capacity(num_values);
        for i in 0..num_values {
            values.push(((i * 73) % 10000) as f32 / 100.0);
        }
        let encoding = round_trip_f32(&values);
        assert!(matches!(
            encoding.compression.as_ref().unwrap(),
            metadata::compressive_encoding::Compression::Alp(_)
        ));
    }

    #[rstest]
    #[case::full_chunks(4096)]
    #[case::short_chunk(1000)]
    #[case::raw_tail(1025)]
    fn test_round_trip_sensor_f64(#[case] num_values: usize) {
        let mut values = Vec::with_capacity(num_values);
        for i in 0..num_values {
            let v = 20.0 + (i as f64 / 10.0).sin();
            values.push((v * 10.0).round() / 10.0);
        }
        let encoding = round_trip_f64(&values);
        assert!(matches!(
            encoding.compression.as_ref().unwrap(),
            metadata::compressive_encoding::Compression::Alp(_)
        ));
    }

    #[test]
    fn test_round_trip_random_f64_uses_rd() {
        // Random-looking doubles do not survive the base-10 transform; the
        // ALP-RD split must compress them losslessly instead of falling back.
        let mut rng = StdRng::seed_from_u64(0xdead_beef);
        let values: Vec<f64> = (0..2048)
            .map(|_| (rng.random::<f64>() - 0.5) * 2.0)
            .collect();
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let data = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        let encoder = AlpMiniBlockEncoder::new(64);
        let (compressed, encoding) = encoder
            .compress(
                MiniBlockCompressionContext::new(0, true, true),
                DataBlock::FixedWidth(data),
            )
            .unwrap();
        // Both chunks must carry the RD mode in their headers.
        let mut header_offset = 0usize;
        for chunk in &compressed.chunks {
            let header =
                &compressed.data[1][header_offset..header_offset + chunk.buffer_sizes[1] as usize];
            assert_eq!(header[0], MODE_RD);
            header_offset += chunk.buffer_sizes[1] as usize;
        }
        let stored: usize = compressed.data.iter().map(|b| b.len()).sum();
        assert!(stored < values.len() * 8, "RD should shrink random doubles");
        check_round_trip(compressed, &encoding, &values, |v| v.to_bits());
    }

    #[test]
    fn test_rd_shared_high_bits() {
        // Values sharing an exponent contract to a small dictionary; the right
        // parts are the only varying bits.
        let mut rng = StdRng::seed_from_u64(7);
        let values: Vec<f64> = (0..1024)
            .map(|_| {
                let sign = if rng.random::<bool>() { 1.0 } else { -1.0 };
                sign * rng.random_range(1000.0..1001.0)
            })
            .collect();
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let data = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        let encoder = AlpMiniBlockEncoder::new(64);
        let (compressed, encoding) = encoder
            .compress(
                MiniBlockCompressionContext::new(0, true, true),
                DataBlock::FixedWidth(data),
            )
            .unwrap();
        check_round_trip(compressed, &encoding, &values, |v| v.to_bits());
    }

    #[test]
    fn test_rd_single_dictionary_entry_round_trip() {
        // A wide drift page that plain ALP cannot transform collapses to a
        // one-entry RD dictionary (all values share their high bits), so every
        // chunk's index stream is empty (index_bits == 0). The decoder must
        // accept that instead of demanding one packed index per value.
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0x05ee_da1b_2026_1023);
        let gauss = |rng: &mut rand::rngs::SmallRng| {
            let u1: f64 = rng.random_range(1e-12..1.0);
            let u2: f64 = rng.random();
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let values: Vec<f64> = (0..65536)
            .map(|i| 1_000_000.0 + i as f64 * 0.1 + gauss(&mut rng) * 0.01)
            .collect();
        let bytes = values
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>();
        let data = FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };
        let encoder = AlpMiniBlockEncoder::new(64);
        let (compressed, encoding) = encoder
            .compress(
                MiniBlockCompressionContext::new(0, true, true),
                DataBlock::FixedWidth(data),
            )
            .unwrap();
        // The page must stay RD (not fall back) and carry an empty index stream.
        let mut header_offset = 0usize;
        let mut saw_single_dict = false;
        for chunk in &compressed.chunks {
            let header =
                &compressed.data[1][header_offset..header_offset + chunk.buffer_sizes[1] as usize];
            assert_eq!(header[0], MODE_RD);
            saw_single_dict |= header[2] == 0; // index_bits == 0
            header_offset += chunk.buffer_sizes[1] as usize;
        }
        assert!(saw_single_dict, "expected an index_bits == 0 RD chunk");
        check_round_trip(compressed, &encoding, &values, |v| v.to_bits());
    }

    #[test]
    fn test_rd_header_validation() {
        assert!(parse_rd_header(&[]).is_err());
        assert!(parse_rd_header(&[MODE_RD, 0, 0, 0, 1, 0, 0, 0]).is_err());
        assert!(parse_rd_header(&[MODE_RD, 64, 0, 0, 1, 0, 0, 0]).is_err());
        assert!(parse_rd_header(&[MODE_RD, 8, 0, 0, 0, 0, 0, 0]).is_err());
        let ok = parse_rd_header(&[MODE_RD, 40, 7, 0, 100, 0, 0, 0]).unwrap();
        assert_eq!(ok.d, 40);
        assert_eq!(ok.index_bits, 7);
        assert_eq!(ok.dict_len, 100);
    }

    #[test]
    fn test_round_trip_with_exceptions_f32() {
        let mut values = (0..2048).map(|v| v as f32 / 10.0).collect::<Vec<_>>();
        values[3] = -0.0;
        values[7] = f32::from_bits(0x7FC0_0001);
        values[11] = f32::INFINITY;
        values[13] = f32::NEG_INFINITY;
        values[17] = 1e30;
        round_trip_f32(&values);
    }

    #[test]
    fn test_round_trip_with_exceptions_f64() {
        let mut values = (0..1024).map(|v| v as f64 / 100.0).collect::<Vec<_>>();
        values[2] = -0.0;
        values[5] = f64::from_bits(0x7FF8_0000_0000_0001);
        values[9] = f64::NEG_INFINITY;
        values[13] = 1e308;
        values[17] = f64::MAX;
        round_trip_f64(&values);
    }

    #[test]
    fn test_round_trip_subnormals_and_payloads() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let f64s = (0..1024)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                f64::from_bits(state)
            })
            .collect::<Vec<_>>();
        round_trip_f64(&f64s);

        let state = state as u32;
        let f32s = (0..1024)
            .map(|i| f32::from_bits(state.wrapping_mul(31).wrapping_add(i as u32)))
            .collect::<Vec<_>>();
        round_trip_f32(&f32s);
    }

    #[test]
    fn test_fallback_when_not_beneficial() {
        let mut state = 0xdead_beef_u32;
        let values = (0..4096)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                f32::from_bits(state)
            })
            .collect::<Vec<_>>();
        let encoding = round_trip_f32(&values);
        assert!(matches!(
            encoding.compression.as_ref().unwrap(),
            metadata::compressive_encoding::Compression::Flat(_)
        ));
    }

    #[test]
    fn test_empty_page() {
        let data = FixedWidthDataBlock {
            data: LanceBuffer::empty(),
            bits_per_value: 32,
            num_values: 0,
            block_info: BlockInfo::new(),
        };
        let encoder = AlpMiniBlockEncoder::new(32);
        let (compressed, encoding) = encoder
            .compress(
                MiniBlockCompressionContext::new(0, true, true),
                DataBlock::FixedWidth(data),
            )
            .unwrap();
        assert_eq!(compressed.data.len(), 4);
        assert!(matches!(
            encoding.compression.as_ref().unwrap(),
            metadata::compressive_encoding::Compression::Alp(_)
        ));
    }

    #[test]
    fn test_alp_rd_single_and_multi_prefix_round_trip() {
        // 1) Single shared high prefix (`index_bits == 0`).
        let mut state = 0x517c_c1b7_2722_0a95_u64;
        let single_prefix: Vec<f64> = (0..2048)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (0x3FF0_u64 << 48) | (state & ((1u64 << 48) - 1));
                f64::from_bits(bits)
            })
            .collect();
        let enc1 = round_trip_f64(&single_prefix);
        assert!(matches!(
            enc1.compression.as_ref().unwrap(),
            metadata::compressive_encoding::Compression::Alp(_)
        ));

        // 2) 128 distinct 20-bit prefixes with random 44-bit suffixes to stress
        //    the single-sort RD_SWEEP + stack open-addressed dictionary table.
        let multi_prefix: Vec<f64> = (0..2048)
            .map(|i| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let prefix = 0x0003_FE00_u64.wrapping_add((i % 128) as u64);
                let bits = (prefix << 44) | (state & ((1u64 << 44) - 1));
                f64::from_bits(bits)
            })
            .collect();
        let enc2 = round_trip_f64(&multi_prefix);
        assert!(matches!(
            enc2.compression.as_ref().unwrap(),
            metadata::compressive_encoding::Compression::Alp(_)
        ));
    }

    #[cfg(all(feature = "bitpacking", target_arch = "x86_64"))]
    #[test]
    fn test_avx2_transform_matches_scalar_edge_cases() {
        if !CpuFeatures::host().contains(CpuFeatures::AVX2) {
            return;
        }

        // Stress leading exceptions, NaN payloads, signed zeros, subnormals,
        // boundary powers near 2^51 / 2^60, negative ties, and non-4/8 tails.
        let mut f64_vals = vec![
            -0.0_f64,
            f64::NAN,
            f64::from_bits(0x7FF8_1234_5678_9ABC),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            -f64::from_bits(1),
            0.0,
            -0.01,
            -0.02,
            0.05,
            12345.67,
            -9876543.21,
            22_517_998_136_852.47, // near 2^51 / 100
            22_517_998_136_852.49, // crosses 2^51 when scaled by 100
            -22_517_998_136_852.49,
            1.0e18,
            std::f64::consts::PI,
        ];
        for i in 0..77 {
            f64_vals.push(((i as i64 - 38) as f64) * 0.01);
        }
        let words64: Vec<u64> = f64_vals.iter().map(|v| v.to_bits()).collect();

        for params in [
            AlpParams {
                e: 2,
                f: 0,
                use_div: false,
            },
            AlpParams {
                e: 2,
                f: 0,
                use_div: true,
            },
            AlpParams {
                e: 4,
                f: 2,
                use_div: false,
            },
        ] {
            let mut enc_scalar = vec![0i64; words64.len()];
            let mut pos_scalar = Vec::new();
            let mut bits_scalar = Vec::new();
            let first_scalar = transform_chunk_f64_scalar(
                &words64,
                params,
                &mut enc_scalar,
                &mut pos_scalar,
                &mut bits_scalar,
            );

            let mut enc_avx2 = vec![0i64; words64.len()];
            let mut pos_avx2 = Vec::new();
            let mut bits_avx2 = Vec::new();
            // SAFETY: checked `CpuFeatures::host().contains(CpuFeatures::AVX2)` above.
            let first_avx2 = unsafe {
                transform_chunk_f64_avx2(
                    &words64,
                    params,
                    &mut enc_avx2,
                    &mut pos_avx2,
                    &mut bits_avx2,
                )
            };

            assert_eq!(first_avx2, first_scalar);
            assert_eq!(pos_avx2, pos_scalar);
            assert_eq!(bits_avx2, bits_scalar);
            assert_eq!(enc_avx2, enc_scalar);
        }

        let mut f32_vals = vec![
            -0.0_f32,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN_POSITIVE,
            0.0,
            -0.25,
            12.34,
            -56.78,
            10_737_418.0, // near 2^30 / 100
            20_000_000.0, // crosses 2^30 when scaled by 100
            std::f32::consts::E,
        ];
        for i in 0..69 {
            f32_vals.push(((i - 34) as f32) * 0.01);
        }
        let words32: Vec<u32> = f32_vals.iter().map(|v| v.to_bits()).collect();

        for params in [
            AlpParams {
                e: 2,
                f: 0,
                use_div: false,
            },
            AlpParams {
                e: 2,
                f: 0,
                use_div: true,
            },
        ] {
            let mut enc_scalar = vec![0i32; words32.len()];
            let mut pos_scalar = Vec::new();
            let mut bits_scalar = Vec::new();
            let first_scalar = transform_chunk_f32_scalar(
                &words32,
                params,
                &mut enc_scalar,
                &mut pos_scalar,
                &mut bits_scalar,
            );

            let mut enc_avx2 = vec![0i32; words32.len()];
            let mut pos_avx2 = Vec::new();
            let mut bits_avx2 = Vec::new();
            // SAFETY: checked `CpuFeatures::host().contains(CpuFeatures::AVX2)` above.
            let first_avx2 = unsafe {
                transform_chunk_f32_avx2(
                    &words32,
                    params,
                    &mut enc_avx2,
                    &mut pos_avx2,
                    &mut bits_avx2,
                )
            };

            assert_eq!(first_avx2, first_scalar);
            assert_eq!(pos_avx2, pos_scalar);
            assert_eq!(bits_avx2, bits_scalar);
            assert_eq!(enc_avx2, enc_scalar);
        }
    }
}
