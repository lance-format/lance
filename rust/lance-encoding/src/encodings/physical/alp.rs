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
#[cfg(feature = "bitpacking")]
use crate::encodings::physical::bitpacking::{bitpack_out_of_line_vec, unpack_out_of_line};
use crate::encodings::physical::value::ValueEncoder;
use crate::format::EncodingUtils;
use crate::format::metadata;
use crate::format::metadata::CompressiveEncoding;
use crate::statistics::{GetStat, Stat};
use arrow_array::cast::AsArray;

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

fn encode_page_f32(
    data: &FixedWidthDataBlock,
    max_chunk: usize,
    raw_size: usize,
) -> Result<Option<(MiniBlockCompressed, CompressiveEncoding)>> {
    let words = data.data.borrow_to_typed_slice::<u32>();
    let Some(params) = select_params_f32(words.as_ref()) else {
        return Ok(None);
    };

    let mut buf0: Vec<u32> = Vec::new();
    let mut buf1: Vec<u8> = Vec::new();
    let mut buf2: Vec<u8> = Vec::new();
    let mut buf3: Vec<u8> = Vec::new();
    let mut chunks = Vec::new();
    let mut compressed_size = 0usize;

    for chunk in words.as_ref().chunks(max_chunk) {
        let Some(encoded) = encode_chunk_f32(chunk, params)? else {
            return Ok(None);
        };
        if encoded.total_bytes() > MAX_MINIBLOCK_BYTES as usize {
            return Ok(None);
        }
        let delta_bytes = encoded.packed_offsets.len() * 4;
        let header_bytes = encoded.header.len();
        let pos_bytes = encoded.exception_positions.len() * 2;
        let exc_bytes = encoded.exception_bits.len();
        buf0.extend(encoded.packed_offsets);
        buf1.extend_from_slice(&encoded.header);
        for pos in encoded.exception_positions {
            buf2.extend_from_slice(&pos.to_le_bytes());
        }
        buf3.extend_from_slice(&encoded.exception_bits);
        compressed_size += delta_bytes + header_bytes + pos_bytes + exc_bytes;

        chunks.push(MiniBlockChunk {
            buffer_sizes: vec![
                delta_bytes as u32,
                header_bytes as u32,
                pos_bytes as u32,
                exc_bytes as u32,
            ],
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

fn encode_page_f64(
    data: &FixedWidthDataBlock,
    max_chunk: usize,
    raw_size: usize,
) -> Result<Option<(MiniBlockCompressed, CompressiveEncoding)>> {
    let words = data.data.borrow_to_typed_slice::<u64>();
    let Some(params) = select_params_f64(words.as_ref()) else {
        return encode_page_f64_rd(words.as_ref(), max_chunk, raw_size);
    };

    let mut buf0: Vec<u64> = Vec::new();
    let mut buf1: Vec<u8> = Vec::new();
    let mut buf2: Vec<u8> = Vec::new();
    let mut buf3: Vec<u8> = Vec::new();
    let mut chunks = Vec::new();
    let mut compressed_size = 0usize;

    for chunk in words.as_ref().chunks(max_chunk) {
        let Some(encoded) = encode_chunk_f64(chunk, params)? else {
            return encode_page_f64_rd(words.as_ref(), max_chunk, raw_size);
        };
        if encoded.total_bytes() > MAX_MINIBLOCK_BYTES as usize {
            return encode_page_f64_rd(words.as_ref(), max_chunk, raw_size);
        }
        let delta_bytes = encoded.packed_offsets.len() * 8;
        let header_bytes = encoded.header.len();
        let pos_bytes = encoded.exception_positions.len() * 2;
        let exc_bytes = encoded.exception_bits.len();
        buf0.extend(encoded.packed_offsets);
        buf1.extend_from_slice(&encoded.header);
        for pos in encoded.exception_positions {
            buf2.extend_from_slice(&pos.to_le_bytes());
        }
        buf3.extend_from_slice(&encoded.exception_bits);
        compressed_size += delta_bytes + header_bytes + pos_bytes + exc_bytes;

        chunks.push(MiniBlockChunk {
            buffer_sizes: vec![
                delta_bytes as u32,
                header_bytes as u32,
                pos_bytes as u32,
                exc_bytes as u32,
            ],
            log_num_values: if chunk.len() < max_chunk {
                0
            } else {
                chunk.len().ilog2() as u8
            },
        });
    }

    if compressed_size >= raw_size {
        return encode_page_f64_rd(words.as_ref(), max_chunk, raw_size);
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

struct EncodedChunk32 {
    packed_offsets: Vec<u32>,
    header: Vec<u8>,
    exception_positions: Vec<u16>,
    exception_bits: Vec<u8>,
}

impl EncodedChunk32 {
    fn total_bytes(&self) -> usize {
        self.packed_offsets.len() * 4
            + self.header.len()
            + self.exception_positions.len() * 2
            + self.exception_bits.len()
    }
}

struct EncodedChunk64 {
    packed_offsets: Vec<u64>,
    header: Vec<u8>,
    exception_positions: Vec<u16>,
    exception_bits: Vec<u8>,
}

impl EncodedChunk64 {
    fn total_bytes(&self) -> usize {
        self.packed_offsets.len() * 8
            + self.header.len()
            + self.exception_positions.len() * 2
            + self.exception_bits.len()
    }
}

/// Transform a chunk into the integer stream and exception lists.
/// Transform a chunk into the integer stream and exception lists.
///
/// The transformation is written branchless so the compiler can vectorize the
/// multiply/round/cast/divide chain; exception decisions fall out of the
/// bitwise round-trip check instead of special-casing NaN, infinities, signed
/// zeros and subnormals up front.
#[cfg(feature = "bitpacking")]
fn encode_chunk_f32(words: &[u32], params: AlpParams) -> Result<Option<EncodedChunk32>> {
    let n = words.len();
    let mut state = EncodeState::new(n);
    let e = params.e as usize;

    if params.use_div {
        let scale = F10_F32[e];
        for (i, &w) in words.iter().enumerate() {
            let v = f32::from_bits(w);
            let scaled = v * scale;
            let enc = scaled.round_ties_even() as i32;
            let ok = scaled.is_finite() && ((enc as f32) / scale).to_bits() == w;
            state.record32(i, w, enc, ok);
        }
    } else {
        let f = params.f as usize;
        let reconstruct_scale = F10_F32[f];
        let reconstruct_inv = IF10_F32[e];
        for (i, &w) in words.iter().enumerate() {
            let v = f32::from_bits(w);
            let scaled = (v * F10_F32[e]) * IF10_F32[f];
            let enc = scaled.round_ties_even() as i32;
            let ok = scaled.is_finite()
                && (((enc as f32) * reconstruct_scale) * reconstruct_inv).to_bits() == w;
            state.record32(i, w, enc, ok);
        }
    }

    let Some(first) = state.first_encodable else {
        return Ok(None);
    };
    if !state.exception_positions.is_empty() {
        backfill_exceptions(&mut state.encoded, &state.exception_positions);
    }
    if state.exception_positions.first() == Some(&0) {
        state.encoded[0] = first;
    }
    pack_chunk_f32(
        state.encoded,
        params,
        state.exception_positions,
        state.exception_bits,
    )
}

/// Mutable state for one chunk's transform: the integer stream plus the
/// exception bookkeeping that runs alongside it.
struct EncodeState<T> {
    encoded: Vec<T>,
    exception_positions: Vec<u16>,
    exception_bits: Vec<u8>,
    first_encodable: Option<T>,
    prev: Option<T>,
}

impl<T: Copy + Default> EncodeState<T> {
    fn new(n: usize) -> Self {
        Self {
            encoded: Vec::with_capacity(n),
            exception_positions: Vec::new(),
            exception_bits: Vec::new(),
            first_encodable: None,
            prev: None,
        }
    }

    #[inline]
    fn record(&mut self, i: usize, value_bits: u64, bytes_per_value: usize, enc: T, ok: bool) {
        if ok {
            if self.first_encodable.is_none() {
                self.first_encodable = Some(enc);
            }
            self.prev = Some(enc);
            self.encoded.push(enc);
        } else {
            self.encoded.push(self.prev.unwrap_or_default());
            self.exception_positions.push(i as u16);
            self.exception_bits
                .extend_from_slice(&value_bits.to_le_bytes()[..bytes_per_value]);
        }
    }
}

impl EncodeState<i32> {
    fn record32(&mut self, i: usize, w: u32, enc: i32, ok: bool) {
        self.record(i, w as u64, 4, enc, ok);
    }
}

impl EncodeState<i64> {
    fn record64(&mut self, i: usize, w: u64, enc: i64, ok: bool) {
        self.record(i, w, 8, enc, ok);
    }
}

/// Transform a chunk into the integer stream and exception lists.
#[cfg(feature = "bitpacking")]
fn encode_chunk_f64(words: &[u64], params: AlpParams) -> Result<Option<EncodedChunk64>> {
    let n = words.len();
    let mut state = EncodeState::new(n);
    let e = params.e as usize;

    if params.use_div {
        let scale = F10_F64[e];
        for (i, &w) in words.iter().enumerate() {
            let v = f64::from_bits(w);
            let scaled = v * scale;
            let enc = scaled.round_ties_even() as i64;
            let ok = scaled.is_finite() && ((enc as f64) / scale).to_bits() == w;
            state.record64(i, w, enc, ok);
        }
    } else {
        let f = params.f as usize;
        let reconstruct_scale = F10_F64[f];
        let reconstruct_inv = IF10_F64[e];
        for (i, &w) in words.iter().enumerate() {
            let v = f64::from_bits(w);
            let scaled = (v * F10_F64[e]) * IF10_F64[f];
            let enc = scaled.round_ties_even() as i64;
            let ok = scaled.is_finite()
                && (((enc as f64) * reconstruct_scale) * reconstruct_inv).to_bits() == w;
            state.record64(i, w, enc, ok);
        }
    }

    let Some(first) = state.first_encodable else {
        return Ok(None);
    };
    if !state.exception_positions.is_empty() {
        backfill_exceptions(&mut state.encoded, &state.exception_positions);
    }
    if state.exception_positions.first() == Some(&0) {
        state.encoded[0] = first;
    }
    pack_chunk_f64(
        state.encoded,
        params,
        state.exception_positions,
        state.exception_bits,
    )
}

/// Backfill exception slots with the preceding encodable integer so Delta
/// streams do not see artificial spikes.
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

/// Pack the cheaper of the FOR and Delta offset streams. Only the winning
/// stream is bit-packed; the loser is dismissed from its (exact) size so the
/// FOR-vs-Delta decision stays identical to packing both.
#[cfg(feature = "bitpacking")]
fn pack_chunk_f32(
    encoded: Vec<i32>,
    params: AlpParams,
    exception_positions: Vec<u16>,
    exception_bits: Vec<u8>,
) -> Result<Option<EncodedChunk32>> {
    let n = encoded.len();
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    let mut delta_min = 0i128;
    let mut delta_bit_width = 0u8;
    let mut delta_usable = false;
    let mut lo = i64::MAX as i128;
    let mut hi = i64::MIN as i128;
    let mut prev = 0i32;
    for (i, &v) in encoded.iter().enumerate() {
        min = min.min(v);
        max = max.max(v);
        if i > 0 {
            let delta = v as i128 - prev as i128;
            lo = lo.min(delta);
            hi = hi.max(delta);
        }
        prev = v;
    }
    let for_bit_width = bits_required_u128((max as i128 - min as i128) as u128);
    if for_bit_width >= 32 {
        return Ok(None);
    }
    let for_width = for_bit_width.max(1);

    // Delta range is computed in i128 so extreme pairs cannot wrap.
    if n > 1 {
        let width = bits_required_u128((hi - lo) as u128);
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

    if for_len <= delta_len {
        let offsets = encoded
            .iter()
            .map(|&v| v.wrapping_sub(min) as u32)
            .collect();
        let Some(packed) = pack_offsets_u32(offsets, for_width as usize) else {
            return Ok(None);
        };
        let mut header = Vec::with_capacity(8);
        header.push(if params.use_div { MODE_DIV } else { 0 });
        header.push(params.e);
        header.push(params.f);
        header.push(for_width);
        header.extend_from_slice(&min.to_le_bytes());
        Ok(Some(EncodedChunk32 {
            packed_offsets: packed,
            header,
            exception_positions,
            exception_bits,
        }))
    } else {
        let width = delta_bit_width as usize;
        let offsets = encoded
            .windows(2)
            .map(|pair| (pair[1] as i128 - pair[0] as i128 - delta_min) as u32)
            .collect();
        let Some(packed) = pack_offsets_u32(offsets, width) else {
            return Ok(None);
        };
        let mut header = Vec::with_capacity(12);
        header.push(MODE_DELTA | if params.use_div { MODE_DIV } else { 0 });
        header.push(params.e);
        header.push(params.f);
        header.push(delta_bit_width);
        header.extend_from_slice(&encoded[0].to_le_bytes());
        header.extend_from_slice(&(delta_min as i32).to_le_bytes());
        Ok(Some(EncodedChunk32 {
            packed_offsets: packed,
            header,
            exception_positions,
            exception_bits,
        }))
    }
}

/// Pack the cheaper of the FOR and Delta offset streams.
#[cfg(feature = "bitpacking")]
fn pack_chunk_f64(
    encoded: Vec<i64>,
    params: AlpParams,
    exception_positions: Vec<u16>,
    exception_bits: Vec<u8>,
) -> Result<Option<EncodedChunk64>> {
    let n = encoded.len();
    let mut min = i64::MAX;
    let mut max = i64::MIN;
    let mut delta_min = 0i128;
    let mut delta_bit_width = 0u8;
    let mut delta_usable = false;
    let mut lo = i64::MAX as i128;
    let mut hi = i64::MIN as i128;
    let mut prev = 0i64;
    for (i, &v) in encoded.iter().enumerate() {
        min = min.min(v);
        max = max.max(v);
        if i > 0 {
            let delta = v as i128 - prev as i128;
            lo = lo.min(delta);
            hi = hi.max(delta);
        }
        prev = v;
    }
    let for_bit_width = bits_required_u128((max as i128 - min as i128) as u128);
    if for_bit_width >= 63 {
        return Ok(None);
    }
    let for_width = for_bit_width.max(1);

    if n > 1 {
        let width = bits_required_u128((hi - lo) as u128);
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

    if for_len <= delta_len {
        let offsets = encoded
            .iter()
            .map(|&v| (v as i128 - min as i128) as u64)
            .collect();
        let Some(packed) = pack_offsets_u64(offsets, for_width as usize) else {
            return Ok(None);
        };
        let mut header = Vec::with_capacity(12);
        header.push(if params.use_div { MODE_DIV } else { 0 });
        header.push(params.e);
        header.push(params.f);
        header.push(for_width);
        header.extend_from_slice(&min.to_le_bytes());
        Ok(Some(EncodedChunk64 {
            packed_offsets: packed,
            header,
            exception_positions,
            exception_bits,
        }))
    } else {
        let width = delta_bit_width as usize;
        let offsets = encoded
            .windows(2)
            .map(|pair| (pair[1] as i128 - pair[0] as i128 - delta_min) as u64)
            .collect();
        let Some(packed) = pack_offsets_u64(offsets, width) else {
            return Ok(None);
        };
        let mut header = Vec::with_capacity(20);
        header.push(MODE_DELTA | if params.use_div { MODE_DIV } else { 0 });
        header.push(params.e);
        header.push(params.f);
        header.push(delta_bit_width);
        header.extend_from_slice(&encoded[0].to_le_bytes());
        header.extend_from_slice(&(delta_min as i64).to_le_bytes());
        Ok(Some(EncodedChunk64 {
            packed_offsets: packed,
            header,
            exception_positions,
            exception_bits,
        }))
    }
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
    let mut buf0: Vec<u64> = Vec::new();
    let mut buf1: Vec<u8> = Vec::new();
    let mut buf2: Vec<u8> = Vec::new();
    let mut buf3: Vec<u64> = Vec::new();
    let mut chunks = Vec::new();
    let mut compressed_size = 0usize;

    for chunk in words.chunks(max_chunk) {
        let Some(encoded) = encode_chunk_f64_rd(chunk) else {
            return Ok(None);
        };
        if encoded.total_bytes() > MAX_MINIBLOCK_BYTES as usize {
            return Ok(None);
        }
        let right_len = encoded.right.len() * 8;
        let header_len = encoded.header.len();
        let dict_len = encoded.dict.len();
        let index_len = encoded.indices.len() * 8;
        buf0.extend(encoded.right);
        buf1.extend_from_slice(&encoded.header);
        buf2.extend_from_slice(&encoded.dict);
        buf3.extend(encoded.indices);
        compressed_size += right_len + header_len + dict_len + index_len;

        chunks.push(MiniBlockChunk {
            buffer_sizes: vec![
                right_len as u32,
                header_len as u32,
                dict_len as u32,
                index_len as u32,
            ],
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

struct EncodedRdChunk64 {
    right: Vec<u64>,
    header: Vec<u8>,
    dict: Vec<u8>,
    indices: Vec<u64>,
}

impl EncodedRdChunk64 {
    fn total_bytes(&self) -> usize {
        self.right.len() * 8 + self.header.len() + self.dict.len() + self.indices.len() * 8
    }
}

/// Encode one chunk with the best ALP-RD split, or `None` when no split beats
/// the raw representation within the chunk budget.
#[cfg(feature = "bitpacking")]
fn encode_chunk_f64_rd(words: &[u64]) -> Option<EncodedRdChunk64> {
    let n = words.len();
    let raw_bytes = n * 8;
    if n == 0 {
        return None;
    }

    let mut best: Option<(u8, u8, Vec<u64>)> = None; // (d, index bits, sorted dict)
    let mut best_cost = usize::MAX;
    for &d in &RD_SWEEP {
        let mut lefts: Vec<u64> = words.iter().map(|w| w >> d).collect();
        lefts.sort_unstable();
        lefts.dedup();
        let dict_len = lefts.len();
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
            best = Some((d, index_bits, lefts));
        }
    }

    let (d, index_bits, dict) = best?;
    let mask = (1u64 << d) - 1;
    let right_values: Vec<u64> = words.iter().map(|w| w & mask).collect();
    let right = pack_offsets_u64(right_values, d as usize)?;

    let indices: Vec<u64> = if index_bits == 0 {
        Vec::new()
    } else {
        let mut position = std::collections::HashMap::with_capacity(dict.len());
        for (i, &left) in dict.iter().enumerate() {
            position.insert(left, i as u64);
        }
        let index_values: Vec<u64> = words.iter().map(|&w| position[&(w >> d)]).collect();
        pack_offsets_u64(index_values, index_bits as usize)?
    };

    let mut dict_bytes = Vec::with_capacity(dict.len() * 8);
    for left in &dict {
        dict_bytes.extend_from_slice(&(left << d).to_le_bytes());
    }
    let mut header = Vec::with_capacity(8);
    header.push(MODE_RD);
    header.push(d);
    header.push(index_bits);
    header.push(0);
    header.extend_from_slice(&(dict.len() as u32).to_le_bytes());

    Some(EncodedRdChunk64 {
        right,
        header,
        dict: dict_bytes,
        indices,
    })
}

#[cfg(not(feature = "bitpacking"))]
fn encode_page_f64_rd(
    _words: &[u64],
    _max_chunk: usize,
    _raw_size: usize,
) -> Result<Option<(MiniBlockCompressed, CompressiveEncoding)>> {
    Ok(None)
}

/// Number of payload bytes `bitpack_out_of_line_vec` produces for `num_items`
/// values at `bit_width`. Mirrors its tail padding decisions exactly.
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

#[cfg(feature = "bitpacking")]
fn pack_offsets_u32(offsets: Vec<u32>, bit_width: usize) -> Option<Vec<u32>> {
    if offsets.is_empty() || bit_width >= 32 {
        return None;
    }
    let num_values = offsets.len() as u64;
    let block = FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(offsets),
        bits_per_value: 32,
        num_values,
        block_info: BlockInfo::new(),
    };
    Some(bitpack_out_of_line_vec::<u32>(block, bit_width))
}

#[cfg(feature = "bitpacking")]
fn pack_offsets_u64(offsets: Vec<u64>, bit_width: usize) -> Option<Vec<u64>> {
    if offsets.is_empty() || bit_width >= 64 {
        return None;
    }
    let num_values = offsets.len() as u64;
    let block = FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(offsets),
        bits_per_value: 64,
        num_values,
        block_info: BlockInfo::new(),
    };
    Some(bitpack_out_of_line_vec::<u64>(block, bit_width))
}

#[cfg(not(feature = "bitpacking"))]
fn encode_chunk_f32(_words: &[u32], _params: AlpParams) -> Result<Option<EncodedChunk32>> {
    // Without bitpacking there is no payload kernel; degrade to the flat
    // page-level fallback rather than failing the write.
    Ok(None)
}

#[cfg(not(feature = "bitpacking"))]
fn encode_chunk_f64(_words: &[u64], _params: AlpParams) -> Result<Option<EncodedChunk64>> {
    Ok(None)
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
            return self.decode_rd64(&rd, packed, buffer2, buffer3, n);
            #[cfg(not(feature = "bitpacking"))]
            {
                let _ = (packed, buffer2, buffer3);
                return Err(Error::not_supported("ALP requires bitpacking support"));
            }
        }

        let header = parse_header(header.as_ref(), self.bits_per_value)?;

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
            .map(|b| b.data.borrow_to_typed_slice::<u32>());
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
            .map(|b| b.data.borrow_to_typed_slice::<u64>());
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

        let unpack = |buffer: LanceBuffer, count: usize, width: usize| -> Result<Vec<u64>> {
            if count == 0 {
                return Ok(Vec::new());
            }
            let block = FixedWidthDataBlock {
                data: buffer,
                bits_per_value: 64,
                num_values: count as u64,
                block_info: BlockInfo::new(),
            };
            Ok(unpack_out_of_line::<u64>(block, count, width)
                .data
                .borrow_to_typed_slice::<u64>()
                .as_ref()
                .to_vec())
        };

        let rights = unpack(packed, n, header.d as usize)?;
        let indices = if header.index_bits == 0 {
            Vec::new()
        } else {
            let indices = unpack(index_buffer, n, header.index_bits as usize)?;
            if indices.len() != n {
                return Err(Error::invalid_input(format!(
                    "ALP-RD index stream has wrong length: n={n} d={} index_bits={} dict_len={} packed_indices={}",
                    header.d,
                    header.index_bits,
                    header.dict_len,
                    indices.len()
                )));
            }
            indices
        };

        let dict_words = dict_buffer_entries(&dict_buffer, header.dict_len);
        let mut out = Vec::with_capacity(n);
        if header.index_bits == 0 {
            let single = dict_words[0];
            out.extend(rights.iter().map(|&r| single | r));
        } else {
            for (&idx, &right) in indices.iter().zip(rights.iter()) {
                let dict = match dict_words.get(idx as usize) {
                    Some(v) => *v,
                    None => {
                        return Err(Error::invalid_input("ALP-RD index out of bounds"));
                    }
                };
                out.push(dict | right);
            }
        }

        let mut bytes = Vec::with_capacity(n * 8);
        for w in out {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: n as u64,
            block_info: BlockInfo::new(),
        }))
    }
}

fn dict_buffer_entries(dict_buffer: &LanceBuffer, dict_len: usize) -> Vec<u64> {
    let words = dict_buffer.borrow_to_typed_slice::<u64>();
    words.as_ref()[..dict_len].to_vec()
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

    #[test]
    fn test_round_trip_decimal_f32_uses_alp() {
        let mut values = Vec::with_capacity(4096);
        for i in 0..4096 {
            values.push(((i * 73) % 10000) as f32 / 100.0);
        }
        let encoding = round_trip_f32(&values);
        assert!(matches!(
            encoding.compression.as_ref().unwrap(),
            metadata::compressive_encoding::Compression::Alp(_)
        ));
    }

    #[test]
    fn test_round_trip_sensor_f64() {
        let mut values = Vec::with_capacity(4096);
        for i in 0..4096 {
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
}
