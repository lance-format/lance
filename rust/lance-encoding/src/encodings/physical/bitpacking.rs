// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bitpacking encodings
//!
//! These encodings look for unused higher order bits and discard them.  For example, if we
//! have a u32 array and all values are between 0 and 5000 then we only need 12 bits to store
//! each value.  The encoding will discard the upper 20 bits and only store 12 bits.
//!
//! This is a simple encoding that works well for data that has a small range.
//!
//! In order to decode the values we need to know the bit width of the values.  This can be stored
//! inline with the data (miniblock) or in the encoding description, out of line (full zip).
//!
//! The encoding is transparent because the output has a fixed width (just like the input) and
//! we can easily jump to the correct value.
//!
//! # u128 inline-bitpacking dispatch (decimal128)
//!
//! For 128-bit-wide columns (`bits_per_value = 128`, e.g. `Decimal128`) the [`InlineBitpacking`]
//! encoder dispatches per chunk into one of five kernels based on the chunk's `Stat::BitWidth`.
//! The kernel is selected by `u128_kernel_for(bit_width)`, which is the single source of truth
//! for both encode and decode and uses a `match` over disjoint width ranges (so the order of
//! the rows below is the dispatch order):
//!
//! | bit_width | kernel                                         | layout            |
//! |-----------|------------------------------------------------|-------------------|
//! | 0         | zero-fill (no body bytes written)              | empty             |
//! | 1..=32    | reinterpret as `&[u32]`, pack with FastLanes  | u32 lane×row      |
//! | 33..=64   | reinterpret as `&[u64]`, pack with FastLanes  | u64 lane×row      |
//! | 65..=127  | scalar u128 sequential bit-stream              | sequential        |
//! | 128       | memcpy identity (each value occupies one word) | sequential        |
//!
//! **Scope:** this dispatch lives only in the `InlineBitpacking` (miniblock) path. The
//! `OutOfLineBitpacking` (full-zip) path is intentionally **not** touched — full-zip frames
//! whole pages with a single `bit_width` and has no per-chunk header to carry a kernel
//! selector, so the same machinery would either need a new format field or change cross-page
//! framing semantics. Both are out of scope here.
//!
//! ## Design: bit_width is the kernel selector
//!
//! There is **no separate negotiation channel** that records which kernel produced a chunk's
//! body bytes — `bit_width` (already written to the chunk header on the encode path) is reused
//! as the implicit selector. The framing-level invariants are preserved: the 16-byte chunk
//! header layout is unchanged, and the body **byte count** still follows the canonical
//! `bit_width * ELEMS_PER_CHUNK / 8` formula for every kernel (see the `pack_u128_chunk`
//! `# Safety` block for the per-kernel proof).
//!
//! What **does** change is the body **byte interpretation**: at the same `bit_width`, an old
//! single-kernel u128 reader and the new dispatched reader will not agree on the meaning of
//! the bits. Concretely, a chunk written by this encoder at `bit_width = 24` is a FastLanes
//! u32 lane×row interleaved layout, not a sequential u128 bit-stream — an older reader that
//! always uses the sequential u128 unpack would silently produce garbage. Both encode and
//! decode therefore **must** route through `u128_kernel_for(bit_width)`; that mapping is the
//! format contract, not an implementation detail. Adding a new u128 kernel without updating
//! both sides — or introducing a real negotiation field with an explicit version bump — would
//! corrupt every file written under the new mapping.
//!
//! There is **no cross-version interop**: a chunk written by this encoder cannot be read by a
//! reader that lacks the per-chunk dispatch. That is why only the 2.3 strategy composes the
//! selector that offers 128-bit widths (`try_wide_bitpacking_miniblock` /
//! `try_wide_bitpacking_block`); every earlier version composes the narrow pair.
//! A reader that accepts a 2.3 file implements this dispatch; readers limited to older versions
//! never receive a 128-bit bitpacked page, so there is no chunk they could decode incorrectly.
//!
//! ## Sign safety
//!
//! The narrow branches (1..=32 and 33..=64) reinterpret `&[u128]` as `&[u32] / &[u64]` and rely
//! on every value satisfying `v >> bit_width == 0`. This invariant is supplied by
//! `FixedWidthDataBlock::max_bit_widths` (see `crate::statistics`), whose OR-fold + `leading_zeros`
//! algorithm guarantees the high lanes are all zero for any `bit_width < 128`. Negative `i128`
//! values (after `bytemuck::cast_slice` to `u128`) have the high bit set and therefore force
//! `bit_width = 128` for the entire chunk, routing them into the memcpy branch.
//!
//! See `pack_u128_chunk` / `unpack_u128_chunk` for per-branch contracts and `# Safety`.

use arrow_array::types::UInt64Type;
use arrow_array::{Array, PrimitiveArray};
use arrow_buffer::ArrowNativeType;
use byteorder::{ByteOrder, LittleEndian};
use lance_bitpacking::{BitPacking, BitPackingUninit};

use lance_core::{Error, Result};

use crate::buffer::LanceBuffer;
use crate::compression::{
    BlockCompressor, BlockDecompressor, MiniBlockDecompressor, require_block_payload,
};
use crate::data::BlockInfo;
use crate::data::{DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressionContext, MiniBlockCompressor,
};
use crate::format::pb21::CompressiveEncoding;
use crate::format::{ProtobufUtils21, pb21};
use crate::statistics::{GetStat, Stat};
use bytemuck::Pod;

pub(crate) const LOG_ELEMS_PER_CHUNK: u8 = 10;
/// Number of values encoded in each inline bitpacking chunk.
pub const ELEMS_PER_CHUNK: u64 = 1 << LOG_ELEMS_PER_CHUNK;

// `pack_u128_chunk` reinterprets `&mut [u128]` body buffers as `&mut [u32]` /
// `&mut [u64]` via `bytemuck::cast_slice_mut`, which requires the body length
// in bytes to be exact for `unchecked_pack`. The framing constant
// `bit_width * ELEMS_PER_CHUNK / (size_of::<T>() * 8)` is integer-exact for any
// `T ∈ {u32, u64, u128}` only because `ELEMS_PER_CHUNK` is a multiple of 128.
// Catch any future drift at compile time.
const _: () = assert!(ELEMS_PER_CHUNK.is_multiple_of(128));

/// Width-dispatched kernel for the u128 inline bitpacking path.
///
/// `pack_u128_chunk` and `unpack_u128_chunk` choose their kernel by computing
/// this enum from the chunk's `bit_width` (carried in the 16-byte chunk header).
/// Keeping the encoder and decoder both call `u128_kernel_for(bit_width)` makes
/// the dispatch table the single source of truth — adding a new kernel requires
/// updating one function instead of two parallel match arms that could drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum U128Kernel {
    /// `bit_width == 0`: zero-fill, no body bytes.
    Zero,
    /// `1..=32`: reinterpret body as `&[u32]`, FastLanes u32 SIMD pack/unpack.
    NarrowU32,
    /// `33..=64`: reinterpret body as `&[u64]`, FastLanes u64 SIMD pack/unpack.
    NarrowU64,
    /// `65..=127`: scalar sequential u128 bit-stream.
    SequentialU128,
    /// `bit_width == 128`: identity — one u128 word per value, copied verbatim.
    Memcpy,
}

#[inline]
fn u128_kernel_for(bit_width: usize) -> U128Kernel {
    debug_assert!(
        bit_width <= 128,
        "u128_kernel_for: bit_width={bit_width} > 128"
    );
    match bit_width {
        0 => U128Kernel::Zero,
        1..=32 => U128Kernel::NarrowU32,
        33..=64 => U128Kernel::NarrowU64,
        65..=127 => U128Kernel::SequentialU128,
        128 => U128Kernel::Memcpy,
        // Pre-condition asserted above; this arm exists so that any future
        // drift (e.g. caller passing a corrupt header) becomes a defined
        // panic in release builds rather than silently routing to Memcpy.
        _ => unreachable!("u128_kernel_for: bit_width={bit_width} > 128 (precondition violated)"),
    }
}

#[derive(Debug, Default)]
pub struct InlineBitpacking {
    uncompressed_bit_width: u64,
}

impl InlineBitpacking {
    pub fn new(uncompressed_bit_width: u64) -> Self {
        Self {
            uncompressed_bit_width,
        }
    }

    pub fn from_description(description: &pb21::InlineBitpacking) -> Self {
        Self {
            uncompressed_bit_width: description.uncompressed_bits_per_value,
        }
    }

    /// The minimum number of bytes required to actually get compression
    ///
    /// We have to compress in blocks of 1024 values.  For example, we can compress 500 2-byte (1000 bytes)
    /// values into 1024 2-bit values (256 bytes) for a win but we don't want to compress 10 2-byte values
    /// into 1024 2-bit values because that's not a win.
    pub fn min_size_bytes(compressed_bit_width: u64) -> u64 {
        (ELEMS_PER_CHUNK * compressed_bit_width).div_ceil(8)
    }

    /// An empty fixed-width block, used for the `num_values == 0` short-circuit in
    /// both decompressor entry points so empty blocks skip chunk validation entirely.
    fn empty_block(&self) -> DataBlock {
        DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::empty(),
            bits_per_value: self.uncompressed_bit_width,
            num_values: 0,
            block_info: BlockInfo::new(),
        })
    }

    /// Bitpacks a FixedWidthDataBlock into compressed chunks of 1024 values
    ///
    /// Each chunk can have a different bit width
    ///
    /// Each chunk has the compressed bit width stored inline in the chunk itself.
    fn bitpack_chunked<T: ArrowNativeType + BitPackingUninit>(
        data: FixedWidthDataBlock,
    ) -> MiniBlockCompressed {
        debug_assert!(data.num_values > 0);
        let data_buffer = data.data.borrow_to_typed_slice::<T>();
        let data_buffer = data_buffer.as_ref();

        let bit_widths = data.expect_stat(Stat::BitWidth);
        let bit_widths_array = bit_widths
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .unwrap();

        let (packed_chunk_sizes, total_size) = bit_widths_array
            .values()
            .iter()
            .map(|&bit_width| {
                let chunk_size = ((1024 * bit_width) / data.bits_per_value) as usize;
                (chunk_size, chunk_size + 1)
            })
            .fold(
                (Vec::with_capacity(bit_widths_array.len()), 0),
                |(mut sizes, total), (size, inc)| {
                    sizes.push(size);
                    (sizes, total + inc)
                },
            );

        let mut output: Vec<T> = Vec::with_capacity(total_size);
        let mut chunks = Vec::with_capacity(bit_widths_array.len());

        for (i, packed_chunk_size) in packed_chunk_sizes
            .iter()
            .enumerate()
            .take(bit_widths_array.len() - 1)
        {
            let start_elem = i * ELEMS_PER_CHUNK as usize;
            let bit_width = bit_widths_array.value(i) as usize;
            output.push(T::from_usize(bit_width).unwrap());
            let output_len = output.len();
            unsafe {
                BitPackingUninit::unchecked_pack_uninit(
                    bit_width,
                    &data_buffer[start_elem..][..ELEMS_PER_CHUNK as usize],
                    &mut output.spare_capacity_mut()[..*packed_chunk_size],
                );
                // The bitpacking kernel initialized every reserved output word.
                output.set_len(output_len + *packed_chunk_size);
            }
            chunks.push(MiniBlockChunk {
                buffer_sizes: vec![((1 + *packed_chunk_size) * std::mem::size_of::<T>()) as u32],
                log_num_values: LOG_ELEMS_PER_CHUNK,
            });
        }

        // Handle the last chunk
        let last_chunk_elem_num = if data.num_values.is_multiple_of(ELEMS_PER_CHUNK) {
            ELEMS_PER_CHUNK
        } else {
            data.num_values % ELEMS_PER_CHUNK
        };
        let mut last_chunk: Vec<T> = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];
        last_chunk[..last_chunk_elem_num as usize].clone_from_slice(
            &data_buffer[data.num_values as usize - last_chunk_elem_num as usize..],
        );
        let bit_width = bit_widths_array.value(bit_widths_array.len() - 1) as usize;
        output.push(T::from_usize(bit_width).unwrap());
        let output_len = output.len();
        let packed_chunk_size = packed_chunk_sizes[bit_widths_array.len() - 1];
        unsafe {
            BitPackingUninit::unchecked_pack_uninit(
                bit_width,
                &last_chunk,
                &mut output.spare_capacity_mut()[..packed_chunk_size],
            );
            // The bitpacking kernel initialized every reserved output word.
            output.set_len(output_len + packed_chunk_size);
        }
        chunks.push(MiniBlockChunk {
            buffer_sizes: vec![
                ((1 + packed_chunk_sizes[bit_widths_array.len() - 1]) * std::mem::size_of::<T>())
                    as u32,
            ],
            log_num_values: 0,
        });

        MiniBlockCompressed {
            data: vec![LanceBuffer::reinterpret_vec(output)],
            chunks,
            num_values: data.num_values,
        }
    }

    /// u128-specific bitpacker that picks a narrow FastLanes lane (u32 or u64) per chunk
    /// when the chunk's bit_width fits, falling back to the scalar sequential u128 kernel
    /// only for chunks that genuinely need 65..=128 bits.
    ///
    /// The on-disk framing is identical to a generic `bitpack_chunked::<u128>` output:
    /// each chunk is a 16-byte u128 header carrying `bit_width` (low 8 bits used, high
    /// 120 bits zero) followed by `bit_width * 1024 / 8` body bytes. The body bytes for
    /// narrow chunks use the FastLanes lane×row layout of the chosen narrow type; the
    /// decoder reads the same header and dispatches by the same bit_width threshold.
    ///
    /// # Sign correctness
    /// `Stat::BitWidth` for u128 is `bits_per_value - leading_zeros(or_fold)`,
    /// where `or_fold` is the bitwise-OR of every value in the chunk
    /// reinterpreted as `u128`. Any high bit set in any value — including the
    /// sign bit of a negative i128 — propagates into `or_fold` and forces
    /// `bit_width = 128`, routing the chunk to the memcpy path. So a chunk
    /// reaching the narrow branches (`bit_width <= 64`) is guaranteed to have
    /// every value in `0..=2^bit_width - 1`, and narrowing the source
    /// `&[u128]` to `&[u32]`/`&[u64]` is lossless.
    fn bitpack_chunked_u128_dispatch(data: FixedWidthDataBlock) -> MiniBlockCompressed {
        debug_assert!(data.num_values > 0);
        debug_assert_eq!(data.bits_per_value, 128);
        let data_buffer = data.data.borrow_to_typed_slice::<u128>();
        let data_buffer = data_buffer.as_ref();

        // The dispatch is u128-specific by design: u8/u16/u32/u64 already enjoy
        // FastLanes SIMD pack/unpack via `BitPacking`, so per-chunk routing would
        // add scheduling overhead with no measurable speedup. u128 is the only
        // word size in `lance-bitpacking` whose default kernel is a scalar
        // sequential bit-stream, which is why narrowing to u32/u64 (when the
        // chunk fits) wins ~3.7× on real decimal128 workloads. Do not extend
        // this dispatch to narrower word sizes without a benchmark showing it
        // helps — the symmetry is an artifact of u128's missing SIMD kernel,
        // not a generic optimization.
        let bit_widths = data.expect_stat(Stat::BitWidth);
        let bit_widths_array = bit_widths
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt64Type>>()
            .expect(
                "Stat::BitWidth must be UInt64Array (FixedWidthDataBlock::max_bit_widths contract)",
            );

        // Hard-assert at least one chunk's worth of statistics. The trailing-chunk
        // path below indexes `bit_widths_array.len() - 1`, which would underflow on
        // an empty input. `data.num_values > 0` is asserted above (debug-only), so
        // this is the release-build defence for that contract.
        assert!(
            !bit_widths_array.is_empty(),
            "bitpack_chunked_u128_dispatch: Stat::BitWidth must produce at least one entry \
             (data.num_values={}, bits_per_value=128)",
            data.num_values
        );

        // Same chunk-size accounting as the generic path: body bytes = bit_width * ELEMS_PER_CHUNK / 8.
        // Expressed in u128 units that's `ELEMS_PER_CHUNK * bit_width / 128 = 8 * bit_width`. The
        // header is 1 u128 per chunk. We size the output buffer in u128 units regardless
        // of which narrow kernel actually writes the body — the body byte count is
        // identical across u32/u64/u128 packings of the same bit_width.
        let packed_chunk_size_u128 =
            |bit_width: u64| -> usize { ((ELEMS_PER_CHUNK * bit_width) / 128) as usize };
        let total_size: usize = bit_widths_array
            .values()
            .iter()
            .map(|&bit_width| packed_chunk_size_u128(bit_width) + 1)
            .sum();

        let mut output: Vec<u128> = Vec::with_capacity(total_size);
        let mut chunks = Vec::with_capacity(bit_widths_array.len());

        for i in 0..bit_widths_array.len() - 1 {
            let bit_width = bit_widths_array.value(i);
            let start_elem = i * ELEMS_PER_CHUNK as usize;
            append_u128_chunk(
                &mut output,
                &mut chunks,
                &data_buffer[start_elem..][..ELEMS_PER_CHUNK as usize],
                bit_width,
                packed_chunk_size_u128(bit_width),
                LOG_ELEMS_PER_CHUNK,
            );
        }

        // Last chunk (with zero-padding to ELEMS_PER_CHUNK if the input doesn't fill it).
        let last_chunk_elem_num = if data.num_values.is_multiple_of(ELEMS_PER_CHUNK) {
            ELEMS_PER_CHUNK
        } else {
            data.num_values % ELEMS_PER_CHUNK
        };
        let mut last_chunk: Vec<u128> = vec![0u128; ELEMS_PER_CHUNK as usize];
        last_chunk[..last_chunk_elem_num as usize].clone_from_slice(
            &data_buffer[data.num_values as usize - last_chunk_elem_num as usize..],
        );
        let last_bit_width = bit_widths_array.value(bit_widths_array.len() - 1);
        let last_chunk_size = packed_chunk_size_u128(last_bit_width);
        // `log_num_values: 0` here is the MiniBlockChunk runt-chunk convention: any
        // chunk that doesn't carry a full 2^LOG_ELEMS_PER_CHUNK = 1024 values uses 0
        // as a sentinel meaning "use `num_values - sum_of_full_chunks` instead". A full
        // trailing chunk (last_chunk_elem_num == ELEMS_PER_CHUNK) still uses this path
        // because the encoder has no special-case for "trailing-but-full" — the decoder
        // resolves the actual count from the parent MiniBlockCompressed.num_values.
        append_u128_chunk(
            &mut output,
            &mut chunks,
            &last_chunk,
            last_bit_width,
            last_chunk_size,
            0,
        );

        MiniBlockCompressed {
            data: vec![LanceBuffer::reinterpret_vec(output)],
            chunks,
            num_values: data.num_values,
        }
    }

    fn chunk_data(&self, data: FixedWidthDataBlock) -> (MiniBlockCompressed, CompressiveEncoding) {
        assert!(data.bits_per_value.is_multiple_of(8));
        assert_eq!(data.bits_per_value, self.uncompressed_bit_width);
        let bits_per_value = data.bits_per_value;
        // The 128-bit arm is what makes Decimal128 columns benefit from inline bit-packing:
        // common decimal precisions (e.g. Decimal128(7,2)) pack to ~24 bits, well below the
        // full 128-bit storage. The u128 path uses per-chunk narrow dispatch: when a
        // chunk's bit_width fits in u32 or u64 we route the body through the FastLanes
        // SIMD kernels for those types, and only fall back to the scalar sequential
        // u128 kernel when the chunk genuinely needs more than 64 bits. Chunk header
        // (16 bytes carrying `bit_width`) and body byte count are unchanged.
        let compressed = match bits_per_value {
            8 => Self::bitpack_chunked::<u8>(data),
            16 => Self::bitpack_chunked::<u16>(data),
            32 => Self::bitpack_chunked::<u32>(data),
            64 => Self::bitpack_chunked::<u64>(data),
            128 => Self::bitpack_chunked_u128_dispatch(data),
            _ => unreachable!(
                "InlineBitpacking::chunk_data: unsupported bits_per_value={}; \
                 supported widths are 8, 16, 32, 64, or 128",
                bits_per_value
            ),
        };
        (
            compressed,
            ProtobufUtils21::inline_bitpacking(
                bits_per_value,
                // TODO: Could potentially compress the data here
                None,
            ),
        )
    }

    fn unchunk<T: ArrowNativeType + BitPackingUninit + Pod>(
        data: LanceBuffer,
        num_values: u64,
    ) -> Result<DataBlock> {
        // This macro decompresses a chunk(1024 values) of bitpacked values.
        let uncompressed_bit_width = std::mem::size_of::<T>() * 8;
        let word_size = std::mem::size_of::<T>();

        if data.len() < word_size {
            return Err(Error::corrupt_file_named(
                "inline_bitpacking",
                format!(
                    "Inline bitpacking chunk is too small for {}-byte header: {} bytes",
                    word_size,
                    data.len()
                ),
            ));
        }
        if !data.len().is_multiple_of(word_size) {
            return Err(Error::corrupt_file_named(
                "inline_bitpacking",
                format!(
                    "Inline bitpacking chunk size must be a multiple of {} bytes, got {} bytes",
                    word_size,
                    data.len()
                ),
            ));
        }
        if num_values > ELEMS_PER_CHUNK {
            return Err(Error::corrupt_file_named(
                "inline_bitpacking",
                format!(
                    "Inline bitpacking chunk has {} values, expected at most {}",
                    num_values, ELEMS_PER_CHUNK
                ),
            ));
        }

        let chunk_words = data.borrow_to_typed_view::<T>();
        let bit_width_value = chunk_words[0].as_usize();
        if bit_width_value > uncompressed_bit_width {
            return Err(Error::corrupt_file_named(
                "inline_bitpacking",
                format!(
                    "Inline bitpacking width {} exceeds {}-bit values",
                    bit_width_value, uncompressed_bit_width
                ),
            ));
        }
        let chunk = &chunk_words[1..];
        // bit_width_value has already been verified to be <= uncompressed_bit_width
        // (8/16/32/64), so bit_width_value * ELEMS_PER_CHUNK (1024) can never
        // overflow usize on supported targets. Keep checked_mul as defense in depth.
        let expected_num_bits = bit_width_value
            .checked_mul(ELEMS_PER_CHUNK as usize)
            .ok_or_else(|| {
                Error::corrupt_file_named(
                    "inline_bitpacking",
                    format!(
                        "Inline bitpacking width {} overflows chunk bit count",
                        bit_width_value
                    ),
                )
            })?;
        let expected_num_bytes = expected_num_bits / 8;
        let actual_num_bytes = std::mem::size_of_val(chunk);
        if actual_num_bytes != expected_num_bytes {
            return Err(Error::corrupt_file_named(
                "inline_bitpacking",
                format!(
                    "Inline bitpacking payload has {} bytes, expected {} bytes for bit width {}",
                    actual_num_bytes, expected_num_bytes, bit_width_value
                ),
            ));
        }

        let mut decompressed = Vec::with_capacity(ELEMS_PER_CHUNK as usize);
        unsafe {
            BitPackingUninit::unchecked_unpack_uninit(
                bit_width_value,
                chunk,
                &mut decompressed.spare_capacity_mut()[..ELEMS_PER_CHUNK as usize],
            );
            // The bitpacking kernel initialized all 1024 decoded values.
            decompressed.set_len(ELEMS_PER_CHUNK as usize);
        }

        decompressed.truncate(num_values as usize);
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(decompressed),
            bits_per_value: uncompressed_bit_width as u64,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }

    /// u128-specific `unchunk` that mirrors `bitpack_chunked_u128_dispatch`: read the
    /// 16-byte u128 header, validate body length against the header's bit_width, then
    /// dispatch the body decode to the matching narrow kernel (u32 / u64 / u128).
    fn unchunk_u128_dispatch(data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        if data.len() < std::mem::size_of::<u128>() {
            return Err(Error::invalid_input(format!(
                "bitpacking buffer too short for u128 header: got {} bytes, need at least {}",
                data.len(),
                std::mem::size_of::<u128>()
            )));
        }
        if num_values > ELEMS_PER_CHUNK {
            return Err(Error::invalid_input(format!(
                "bitpacking num_values {num_values} exceeds chunk capacity {ELEMS_PER_CHUNK}"
            )));
        }

        let header_bytes = &data[..std::mem::size_of::<u128>()];
        let raw = LittleEndian::read_u128(header_bytes);
        if raw > 128 {
            return Err(Error::invalid_input(format!(
                "bitpacking header out of range: {raw} > 128"
            )));
        }
        let bit_width = raw as usize;

        let body = &data[std::mem::size_of::<u128>()..];
        let expected_bytes = bit_width * ELEMS_PER_CHUNK as usize / 8;
        if body.len() != expected_bytes {
            return Err(Error::invalid_input(format!(
                "bitpacking chunk body length mismatch: got {} bytes, \
                 expected {expected_bytes} bytes for bit_width={bit_width}",
                body.len()
            )));
        }

        // Short-circuit when the caller asked for zero rows from this chunk
        // (e.g. a slice that does not intersect the chunk). Header + body
        // length validation above is still useful (catches corruption), but
        // we can skip the 1024-element scratch unpack + truncate. Mirrors the
        // logical end-state of `decompressed.truncate(0)` while saving a
        // 16 KiB stack/heap unpack and the kernel's per-value work.
        if num_values == 0 {
            return Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::empty(),
                bits_per_value: 128,
                num_values: 0,
                block_info: BlockInfo::new(),
            }));
        }

        let mut decompressed = vec![0u128; ELEMS_PER_CHUNK as usize];
        unsafe { unpack_u128_chunk(bit_width, body, &mut decompressed) };

        decompressed.truncate(num_values as usize);
        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(decompressed),
            bits_per_value: 128,
            num_values,
            block_info: BlockInfo::new(),
        }))
    }
}

impl MiniBlockCompressor for InlineBitpacking {
    fn compress(
        &self,
        _context: MiniBlockCompressionContext,
        chunk: DataBlock,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match chunk {
            DataBlock::FixedWidth(fixed_width) => Ok(self.chunk_data(fixed_width)),
            _ => Err(Error::invalid_input_source(
                format!(
                    "Cannot compress a data block of type {} with BitpackMiniBlockEncoder",
                    chunk.name()
                )
                .into(),
            )),
        }
    }
}

impl BlockCompressor for InlineBitpacking {
    fn compress(&self, data: DataBlock) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let DataBlock::FixedWidth(fixed_width) = data else {
            return Err(Error::invalid_input(
                "Inline bitpacking requires fixed-width data",
            ));
        };
        if fixed_width.bits_per_value != self.uncompressed_bit_width {
            return Err(Error::invalid_input(format!(
                "Inline bitpacking expects {}-bit values, got {}",
                self.uncompressed_bit_width, fixed_width.bits_per_value
            )));
        }
        let (chunked, _) = self.chunk_data(fixed_width);
        let payload =
            chunked.data.into_iter().next().ok_or_else(|| {
                Error::internal("Inline bitpacking produced no payload".to_string())
            })?;
        Ok((
            Some(payload),
            ProtobufUtils21::inline_bitpacking(self.uncompressed_bit_width, None),
        ))
    }
}

impl MiniBlockDecompressor for InlineBitpacking {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        assert_eq!(data.len(), 1);
        let data = data.into_iter().next().unwrap();
        if num_values == 0 {
            // Empty mini-blocks have no inline bit-width header to decode.
            return Ok(self.empty_block());
        }
        match self.uncompressed_bit_width {
            8 => Self::unchunk::<u8>(data, num_values),
            16 => Self::unchunk::<u16>(data, num_values),
            32 => Self::unchunk::<u32>(data, num_values),
            64 => Self::unchunk::<u64>(data, num_values),
            128 => Self::unchunk_u128_dispatch(data, num_values),
            w => Err(Error::invalid_input(format!(
                "Bitpacking word size must be 8, 16, 32, 64, or 128, got {w}"
            ))),
        }
    }

    fn decoded_size_bytes(&self, num_values: u64) -> Option<u64> {
        num_values
            .checked_mul(self.uncompressed_bit_width)
            .map(|bits| bits.div_ceil(8))
    }
}

impl BlockDecompressor for InlineBitpacking {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let data = require_block_payload(data, "Inline bitpacking")?;
        if num_values == 0 {
            // Empty blocks carry no inline bit-width header to decode; avoid
            // spurious "too small for header" corrupt-file errors and mirror
            // the MiniBlockDecompressor path. See #7794.
            return Ok(self.empty_block());
        }
        match self.uncompressed_bit_width {
            8 => Self::unchunk::<u8>(data, num_values),
            16 => Self::unchunk::<u16>(data, num_values),
            32 => Self::unchunk::<u32>(data, num_values),
            64 => Self::unchunk::<u64>(data, num_values),
            128 => Self::unchunk_u128_dispatch(data, num_values),
            w => Err(Error::invalid_input(format!(
                "Bitpacking word size must be 8, 16, 32, 64, or 128, got {w}"
            ))),
        }
    }
}

/// Append one packed u128 chunk to `output` and record its `MiniBlockChunk`
/// entry in `chunks`. Encoder helper extracted from
/// `bitpack_chunked_u128_dispatch`.
///
/// This is intentionally a free function (rather than the captured-state
/// closure used in earlier revisions) so the encoder doesn't borrow the
/// outer scope while writing chunks, and so each call site is independently
/// auditable.
///
/// The body is written by first growing `output` to its final length with
/// zero-init u128 slots, then handing a regular `&mut [u128]` slice to
/// `pack_u128_chunk`. This keeps the function entirely in safe Rust and
/// avoids any `&mut [u128]` reference materialized over uninitialized
/// memory (which would be undefined behavior even if every byte is
/// subsequently overwritten by `pack_u128_chunk`). The zero-init cost is a
/// `memset` of `packed_chunk_size * 16` bytes per chunk (≤ 16 KiB at
/// `bit_width = 128`), bounded by the FastLanes pack work itself.
///
/// Panic safety: u128 has no invalid bit patterns and no `Drop`, so even if
/// `pack_u128_chunk` panics partway through (e.g. a per-value
/// `debug_assert!(v >> bit_width == 0)` fires), the partially-filled body
/// holds well-defined zero / partially-packed u128 values, the `Vec` is at
/// its declared length, and unwinding is no-op-equivalent.
fn append_u128_chunk(
    output: &mut Vec<u128>,
    chunks: &mut Vec<MiniBlockChunk>,
    src: &[u128],
    bit_width_u64: u64,
    packed_chunk_size: usize,
    log_num_values: u8,
) {
    debug_assert!(
        bit_width_u64 <= 128,
        "append_u128_chunk: Stat::BitWidth produced out-of-range value {bit_width_u64} for u128"
    );
    let bit_width = bit_width_u64 as usize;
    output.push(bit_width as u128);
    let header_end = output.len();
    // Zero-init the body slots, then write through a safe `&mut [u128]`.
    // `resize` is `O(packed_chunk_size)` memset + reserve; capacity was
    // already reserved by the caller, so this is just a length bump and
    // memset. See the function-level rustdoc for the soundness rationale
    // (avoids materializing `&mut [u128]` over uninit memory).
    output.resize(header_end + packed_chunk_size, 0u128);
    let body_dest = &mut output[header_end..header_end + packed_chunk_size];
    // SAFETY: `pack_u128_chunk` requires `src.len() == ELEMS_PER_CHUNK`,
    // `bit_width <= 128`, and `dest.len() * 16 == bit_width * ELEMS_PER_CHUNK / 8`.
    // (1) Caller of `append_u128_chunk` slices `src` to exactly one chunk; the
    // call sites in `bitpack_chunked_u128_dispatch` always pass a 1024-element
    // window. (2) `bit_width_u64 <= 128` is debug-asserted at function entry
    // and `Stat::BitWidth` is `<= bits_per_value = 128` by construction.
    // (3) `packed_chunk_size = bit_width * ELEMS_PER_CHUNK / 128` (in u128
    // words), matching the framing formula exactly.
    unsafe { pack_u128_chunk(bit_width, src, body_dest) };
    chunks.push(MiniBlockChunk {
        buffer_sizes: vec![((1 + packed_chunk_size) * std::mem::size_of::<u128>()) as u32],
        log_num_values,
    });
}

/// Pack a 1024-element u128 chunk with width-based dispatch. The body byte count
/// is always `bit_width * 1024 / 8`, identical across all five branches:
///   - bit_width == 0: no body bytes (`dest_u128` is empty).
///   - bit_width 1..=32: write FastLanes u32 lane×row layout into `dest_u128` (cast to u32).
///   - bit_width 33..=64: write FastLanes u64 lane×row layout (cast to u64).
///   - bit_width 65..=127: write the existing scalar sequential u128 layout.
///   - bit_width == 128: identity — body holds raw u128 values, written via `copy_nonoverlapping`.
///
/// The branch is selected by `u128_kernel_for(bit_width)` (single source of truth shared
/// with `unpack_u128_chunk`). `dest_u128.len()` must equal `1024 * bit_width / 128`, the
/// standard u128 framing used by the generic `bitpack_chunked` machinery — alignment is
/// sufficient for `bytemuck::cast_slice_mut::<u128, u32>` / `<u64>` because u128 is
/// 16-byte aligned.
///
/// # Safety
/// Caller guarantees:
/// - `src.len() == ELEMS_PER_CHUNK` (1024).
/// - `dest_u128.len() * 16 == bit_width * ELEMS_PER_CHUNK / 8`.
/// - `dest_u128` is a `&mut [u128]`, so its base pointer is 16-byte aligned;
///   `bytemuck::cast_slice_mut::<u128, u32>` and `<u128, u64>` therefore
///   satisfy their alignment requirements unconditionally — there is no
///   alignment-failure path for the narrow branches.
/// - `bit_width <= 128`.
/// - For `bit_width <= 127`, every value in `src` satisfies `v >> bit_width == 0`
///   (no bit set above position `bit_width - 1`). This is enforced upstream by
///   `Stat::BitWidth`, which OR-folds the chunk's u128-reinterpreted bytes and
///   reports `128 - leading_zeros(or_fold)` — a negative i128 (sign bit set) or
///   any value exceeding `2^bit_width - 1` therefore drives the chunk's reported
///   bit_width to at least the position of its highest set bit, so a chunk
///   reaching the narrow or sequential branches is provably bounded by
///   `2^bit_width`. `debug_assertions` builds re-check each value to catch any
///   future drift in the statistics computation.
/// - The `bit_width == 128` branch holds raw, full-range values that are written
///   verbatim via `copy_nonoverlapping` and is the only branch that can carry
///   negative i128 values.
/// - `BitPacking::unchecked_pack`'s contract requires `bit_width <= size_of::<T>() * 8`,
///   which is satisfied by construction: `T = u32` for 1..=32, `u64` for 33..=64,
///   `u128` for 65..=127. The 0 and 128 arms short-circuit and never call into
///   `BitPacking`.
unsafe fn pack_u128_chunk(bit_width: usize, src: &[u128], dest_u128: &mut [u128]) {
    debug_assert_eq!(src.len(), ELEMS_PER_CHUNK as usize);
    debug_assert!(
        bit_width <= 128,
        "pack_u128_chunk: bit_width={bit_width} > 128"
    );
    let expected_body_bytes = bit_width * ELEMS_PER_CHUNK as usize / 8;
    debug_assert_eq!(std::mem::size_of_val(dest_u128), expected_body_bytes);

    match u128_kernel_for(bit_width) {
        U128Kernel::Zero => {
            // bit_width == 0: every value is implicitly zero, body is empty.
            debug_assert!(
                src.iter().all(|&v| v == 0),
                "pack_u128_chunk: bit_width=0 requires all values zero"
            );
        }
        U128Kernel::Memcpy => {
            // bit_width == 128: identity — at full width every value occupies one whole
            // u128 word in the body, so the FastLanes scalar bit-loop degenerates to a
            // sequence of single-word writes. `copy_nonoverlapping` short-circuits the
            // same work and aligns with the dispatch's documented "memcpy path" —
            // important for the worst case (negative i128 columns), which forces this
            // branch on every chunk.
            debug_assert_eq!(dest_u128.len(), ELEMS_PER_CHUNK as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), dest_u128.as_mut_ptr(), src.len());
            }
        }
        U128Kernel::NarrowU32 => {
            // bit_width 1..=32: every src value is bounded by `2^bit_width` (sign-safety
            // invariant from `Stat::BitWidth`), so the high 96 bits are zero and casting
            // to `u32` is lossless.
            //
            // Release-build sentinel on the first value: corruption modes that drive a
            // value's high bits set while keeping `bit_width <= 32` (e.g. a stale
            // `Stat::BitWidth` reused across calls, or a malformed encoder feeding a
            // negative i128 down this path) would silently truncate to u32 in
            // production. We pay a single check on `src[0]` to turn that into a
            // defined panic; the remaining 1023 values are still covered by the
            // per-value `debug_assert!` in dev/test builds.
            assert!(
                src[0] >> bit_width == 0,
                "pack_u128_chunk narrow-u32: src[0]={:#x} exceeds bit_width={bit_width} \
                 (Stat::BitWidth invariant violated)",
                src[0]
            );
            let mut narrow = [0u32; ELEMS_PER_CHUNK as usize];
            for (n, &v) in narrow.iter_mut().zip(src.iter()) {
                debug_assert!(
                    v >> bit_width == 0,
                    "pack_u128_chunk narrow-u32: value {v:#x} exceeds bit_width={bit_width} \
                     — Stat::BitWidth invariant violated (any high bit, including i128 sign, \
                     must drive bit_width to 128 and route to the sequential/memcpy path)"
                );
                *n = v as u32;
            }
            let dest_u32 = bytemuck::cast_slice_mut::<u128, u32>(dest_u128);
            unsafe { BitPacking::unchecked_pack(bit_width, &narrow, dest_u32) };
        }
        U128Kernel::NarrowU64 => {
            // bit_width 33..=64: every src value fits in 64 bits (same invariant).
            //
            // Release-build sentinel on the first value: mirrors the NarrowU32 branch.
            // A miscomputed `Stat::BitWidth` could otherwise let a high-bit value
            // silently truncate to u64. One assert turns the silent corruption into a
            // defined panic; per-value `debug_assert!` covers dev/test builds.
            assert!(
                src[0] >> bit_width == 0,
                "pack_u128_chunk narrow-u64: src[0]={:#x} exceeds bit_width={bit_width} \
                 (Stat::BitWidth invariant violated)",
                src[0]
            );
            let mut narrow = [0u64; ELEMS_PER_CHUNK as usize];
            for (n, &v) in narrow.iter_mut().zip(src.iter()) {
                debug_assert!(
                    v >> bit_width == 0,
                    "pack_u128_chunk narrow-u64: value {v:#x} exceeds bit_width={bit_width} \
                     — Stat::BitWidth invariant violated (any high bit, including i128 sign, \
                     must drive bit_width to 128 and route to the sequential/memcpy path)"
                );
                *n = v as u64;
            }
            let dest_u64 = bytemuck::cast_slice_mut::<u128, u64>(dest_u128);
            unsafe { BitPacking::unchecked_pack(bit_width, &narrow, dest_u64) };
        }
        U128Kernel::SequentialU128 => {
            // bit_width 65..=127: scalar sequential u128 kernel. Same `v >> bit_width
            // == 0` contract as the narrow branches.
            for &v in src.iter() {
                debug_assert!(
                    v >> bit_width == 0,
                    "pack_u128_chunk u128-sequential: value {v:#x} exceeds bit_width={bit_width} \
                     — Stat::BitWidth invariant violated"
                );
            }
            unsafe { BitPacking::unchecked_pack(bit_width, src, dest_u128) };
        }
    }
}

/// Unpack a 1024-element u128 chunk with width-based dispatch matching
/// `pack_u128_chunk` exactly (both call `u128_kernel_for`). `body_bytes` is the
/// chunk body (after the 16-byte header) and must have length
/// `bit_width * 1024 / 8`. `dest.len()` must be 1024.
///
/// For each kernel branch the body is staged into a stack-local typed buffer
/// (no heap allocation) via byte-level `copy_nonoverlapping`, which has no
/// alignment requirement on the on-disk u8 source — `BitPacking::unchecked_unpack`
/// is then called on the (now correctly aligned) stack buffer. The narrow
/// branches widen each lane back to `u128` lossless-zero-extend; the
/// encoder's sign-safety invariant guarantees those values are in
/// `0..=2^bit_width`. The `bit_width == 128` arm short-circuits to a single
/// byte-level memcpy that mirrors the encoder's `copy_nonoverlapping`
/// fast-path — no intermediate `Vec`, no second `copy_from_slice`.
///
/// # Safety
/// Caller guarantees:
/// - `body_bytes.len() == bit_width * ELEMS_PER_CHUNK / 8`
///   (also asserted hard at the function entry).
/// - `dest.len() == ELEMS_PER_CHUNK` (1024).
/// - `bit_width <= 128`.
/// - `BitPacking::unchecked_unpack`'s contract requires `bit_width <= size_of::<T>() * 8`,
///   which is satisfied by construction: `T = u32` for 1..=32, `u64` for 33..=64,
///   `u128` for 65..=127. The 0 and 128 arms short-circuit (zero-fill and memcpy
///   respectively) and never call into `BitPacking`.
unsafe fn unpack_u128_chunk(bit_width: usize, body_bytes: &[u8], dest: &mut [u128]) {
    debug_assert_eq!(dest.len(), ELEMS_PER_CHUNK as usize);
    debug_assert!(
        bit_width <= 128,
        "unpack_u128_chunk: bit_width={bit_width} > 128"
    );
    // Hard-asserted at the decode-input boundary: the body byte count is the only
    // contract surface that is reachable from on-disk framing, so a corrupt or
    // truncated chunk must produce a defined panic in release builds rather than
    // out-of-bounds reads inside the kernel branches.
    assert_eq!(
        body_bytes.len(),
        bit_width * ELEMS_PER_CHUNK as usize / 8,
        "unpack_u128_chunk: body_bytes.len()={} does not match bit_width={} (expected {})",
        body_bytes.len(),
        bit_width,
        bit_width * ELEMS_PER_CHUNK as usize / 8
    );

    match u128_kernel_for(bit_width) {
        U128Kernel::Zero => {
            // bit_width == 0: body is empty; every value is implicitly zero.
            dest.fill(0);
        }
        U128Kernel::Memcpy => {
            // bit_width == 128: identity. Body is exactly `ELEMS_PER_CHUNK` u128
            // values byte-for-byte, mirroring the encoder's `copy_nonoverlapping`
            // fast-path. Single byte-level memcpy: no alignment requirement on
            // the on-disk source, no intermediate heap allocation, no second
            // copy. Important for the worst case (negative i128 columns) which
            // hits this path on every chunk.
            debug_assert_eq!(
                body_bytes.len(),
                ELEMS_PER_CHUNK as usize * 16,
                "unpack_u128_chunk Memcpy: body bytes must be exactly 1024 * 16; \
                 got {} (header outer-assertion already covers this for bit_width=128)",
                body_bytes.len()
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body_bytes.as_ptr(),
                    dest.as_mut_ptr() as *mut u8,
                    body_bytes.len(),
                );
            }
        }
        U128Kernel::NarrowU32 => {
            // bit_width 1..=32: stage the on-disk packed body into an aligned
            // stack u32 buffer (no heap alloc), unpack into a second stack
            // buffer, then widen each lane to u128 (lossless zero-extend by the
            // sign-safety invariant). `words <= ELEMS_PER_CHUNK` for `bit_width <= 32`.
            let words = bit_width * (ELEMS_PER_CHUNK as usize / 32);
            debug_assert!(words <= ELEMS_PER_CHUNK as usize);
            let mut packed = [0u32; ELEMS_PER_CHUNK as usize];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body_bytes.as_ptr(),
                    packed.as_mut_ptr() as *mut u8,
                    body_bytes.len(),
                );
            }
            let mut narrow = [0u32; ELEMS_PER_CHUNK as usize];
            unsafe { BitPacking::unchecked_unpack(bit_width, &packed[..words], &mut narrow) };
            for (d, &v) in dest.iter_mut().zip(narrow.iter()) {
                *d = v as u128;
            }
        }
        U128Kernel::NarrowU64 => {
            // bit_width 33..=64: same staged-stack pattern with u64 lanes.
            let words = bit_width * (ELEMS_PER_CHUNK as usize / 64);
            debug_assert!(words <= ELEMS_PER_CHUNK as usize);
            let mut packed = [0u64; ELEMS_PER_CHUNK as usize];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body_bytes.as_ptr(),
                    packed.as_mut_ptr() as *mut u8,
                    body_bytes.len(),
                );
            }
            let mut narrow = [0u64; ELEMS_PER_CHUNK as usize];
            unsafe { BitPacking::unchecked_unpack(bit_width, &packed[..words], &mut narrow) };
            for (d, &v) in dest.iter_mut().zip(narrow.iter()) {
                *d = v as u128;
            }
        }
        U128Kernel::SequentialU128 => {
            // bit_width 65..=127: scalar sequential u128 kernel. Stage on-disk
            // bytes into an aligned stack u128 buffer and unpack directly into
            // dest. `words = bit_width * 8` for ELEMS_PER_CHUNK = 1024 / 128.
            let words = bit_width * (ELEMS_PER_CHUNK as usize / 128);
            debug_assert!(words <= ELEMS_PER_CHUNK as usize);
            let mut packed = [0u128; ELEMS_PER_CHUNK as usize];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    body_bytes.as_ptr(),
                    packed.as_mut_ptr() as *mut u8,
                    body_bytes.len(),
                );
            }
            unsafe { BitPacking::unchecked_unpack(bit_width, &packed[..words], dest) };
        }
    }
}

/// Bitpacks a FixedWidthDataBlock with a given bit width.
///
/// Each chunk of 1024 values is packed with a constant bit width. For the tail we compare the
/// cost of padding and packing against storing the raw values: if padding yields a smaller
/// representation we pack; otherwise we append the raw tail.
fn bitpack_out_of_line<T: ArrowNativeType + BitPackingUninit>(
    data: FixedWidthDataBlock,
    compressed_bits_per_value: usize,
) -> LanceBuffer {
    let data_buffer = data.data.borrow_to_typed_slice::<T>();
    let data_buffer = data_buffer.as_ref();

    let num_chunks = data_buffer.len().div_ceil(ELEMS_PER_CHUNK as usize);
    let last_chunk_is_runt = data_buffer.len() % ELEMS_PER_CHUNK as usize != 0;
    let words_per_chunk = (ELEMS_PER_CHUNK as usize * compressed_bits_per_value)
        .div_ceil(data.bits_per_value as usize);
    let mut output: Vec<T> = Vec::with_capacity(num_chunks * words_per_chunk);

    let num_whole_chunks = if last_chunk_is_runt {
        num_chunks - 1
    } else {
        num_chunks
    };

    // Simple case for complete chunks
    for i in 0..num_whole_chunks {
        let input_start = i * ELEMS_PER_CHUNK as usize;
        let input_end = input_start + ELEMS_PER_CHUNK as usize;
        let output_start = output.len();
        unsafe {
            BitPackingUninit::unchecked_pack_uninit(
                compressed_bits_per_value,
                &data_buffer[input_start..input_end],
                &mut output.spare_capacity_mut()[..words_per_chunk],
            );
            // The bitpacking kernel initialized this complete packed chunk.
            output.set_len(output_start + words_per_chunk);
        }
    }

    if !last_chunk_is_runt {
        return LanceBuffer::reinterpret_vec(output);
    }

    let last_chunk_start = num_whole_chunks * ELEMS_PER_CHUNK as usize;
    let remaining_items = data_buffer.len() - last_chunk_start;

    let uncompressed_bits = data.bits_per_value as usize;
    let tail_bit_savings = uncompressed_bits.saturating_sub(compressed_bits_per_value);
    let padding_cost = compressed_bits_per_value * (ELEMS_PER_CHUNK as usize - remaining_items);
    let tail_pack_savings = tail_bit_savings.saturating_mul(remaining_items);
    debug_assert!(remaining_items > 0, "remaining_items must be non-zero");
    debug_assert!(tail_bit_savings > 0, "tail_bit_savings must be non-zero");

    if padding_cost < tail_pack_savings {
        // Padding buys us more than it costs: pad to 1024 values and pack them as a normal chunk.
        let mut last_chunk: Vec<T> = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];
        last_chunk[..remaining_items].copy_from_slice(&data_buffer[last_chunk_start..]);
        let start = output.len();
        unsafe {
            BitPackingUninit::unchecked_pack_uninit(
                compressed_bits_per_value,
                &last_chunk,
                &mut output.spare_capacity_mut()[..words_per_chunk],
            );
            // The bitpacking kernel initialized the padded tail chunk.
            output.set_len(start + words_per_chunk);
        }
    } else {
        // Padding would waste space; append tail values as-is.
        output.extend_from_slice(&data_buffer[last_chunk_start..]);
    }

    LanceBuffer::reinterpret_vec(output)
}

/// Unpacks a FixedWidthDataBlock that has been bitpacked with a constant bit width.
///
/// The compressed bit width is provided while the uncompressed width comes from `T`.
/// Depending on the encoding decision the final chunk may be fully packed (with padding)
/// or stored as raw tail values. We infer the layout from the buffer length.
fn unpack_out_of_line<T: ArrowNativeType + BitPackingUninit>(
    data: FixedWidthDataBlock,
    num_values: usize,
    compressed_bits_per_value: usize,
) -> FixedWidthDataBlock {
    let words_per_chunk = (ELEMS_PER_CHUNK as usize * compressed_bits_per_value)
        .div_ceil(data.bits_per_value as usize);
    let compressed_words = data.data.borrow_to_typed_slice::<T>();

    let num_whole_chunks = num_values / ELEMS_PER_CHUNK as usize;
    let tail_values = num_values % ELEMS_PER_CHUNK as usize;
    let expected_full_words = num_whole_chunks * words_per_chunk;
    let expected_new_len = expected_full_words + tail_values;
    let tail_is_raw = tail_values > 0 && compressed_words.len() == expected_new_len;

    let extra_tail_capacity = ELEMS_PER_CHUNK as usize;
    let mut decompressed: Vec<T> =
        Vec::with_capacity(num_values.saturating_add(extra_tail_capacity));

    for chunk_idx in 0..num_whole_chunks {
        let input_start = chunk_idx * words_per_chunk;
        let input_end = input_start + words_per_chunk;
        let output_start = decompressed.len();
        unsafe {
            BitPackingUninit::unchecked_unpack_uninit(
                compressed_bits_per_value,
                &compressed_words[input_start..input_end],
                &mut decompressed.spare_capacity_mut()[..ELEMS_PER_CHUNK as usize],
            );
            // The bitpacking kernel initialized this complete decoded chunk.
            decompressed.set_len(output_start + ELEMS_PER_CHUNK as usize);
        }
    }

    if tail_values > 0 {
        // The tail might be padded and bit packed or it might be appended raw.  We infer the
        // layout from the buffer length to decode appropriately.
        if tail_is_raw {
            let tail_start = expected_full_words;
            decompressed.extend_from_slice(&compressed_words[tail_start..tail_start + tail_values]);
        } else {
            let tail_start = expected_full_words;
            let output_start = decompressed.len();
            unsafe {
                BitPackingUninit::unchecked_unpack_uninit(
                    compressed_bits_per_value,
                    &compressed_words[tail_start..tail_start + words_per_chunk],
                    &mut decompressed.spare_capacity_mut()[..ELEMS_PER_CHUNK as usize],
                );
                // The kernel initialized a full chunk; only the requested tail stays visible.
                decompressed.set_len(output_start + ELEMS_PER_CHUNK as usize);
            }
            decompressed.truncate(output_start + tail_values);
        }
    }

    debug_assert_eq!(decompressed.len(), num_values);

    FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(decompressed),
        bits_per_value: data.bits_per_value,
        num_values: num_values as u64,
        block_info: BlockInfo::new(),
    }
}

/// A transparent compressor that bit packs data
///
/// In order for the encoding to be transparent we must have a fixed bit width
/// across the entire array.  Chunking within the buffer is not supported.  This
/// means that we will be slightly less efficient than something like the mini-block
/// approach.
///
/// This was an interesting experiment but it can't be used as a per-value compressor
/// at the moment.  The resulting data IS transparent but it's not quite so simple.  We
/// compress in blocks of 1024 and each block has a fixed size but also has some padding.
///
/// We do use this as a block compressor currently.
///
/// In other words, if we try the simple math to access the item at index `i` we will be
/// out of luck because `bits_per_value * i` is not the location.  What we need is something
/// like:
///
/// ```ignore
/// let chunk_idx = i / 1024;
/// let chunk_offset = i % 1024;
/// bits_per_chunk * chunk_idx + bits_per_value * chunk_offset
/// ```
///
/// However, this logic isn't expressible with the per-value traits we have today.  We can
/// enhance these traits should we need to support it at some point in the future.
///
/// # u128 path note (Inline-vs-OutOfLine asymmetry)
///
/// The u128 dispatch added by this PR (per-chunk routing to u32/u64 FastLanes SIMD via
/// `pack_u128_chunk` / `unpack_u128_chunk`) lives only on the **inline** miniblock path.
/// `OutOfLineBitpacking` for `bits_per_value == 128` continues to call the scalar
/// `bitpack_out_of_line::<u128>` kernel (line 1161) — there is no per-chunk dispatch on
/// values, no narrow-lane fast path, and no SIMD. This is intentional for the scope of
/// this PR: the perf hotspot motivating the dispatch (q4 over decimal128 store_sales) is
/// served by the inline miniblock encoder, and full-zip bitpacking is out of scope.
/// Expect the OutOfLine u128 path to be slower than the inline path on identical data;
/// fixing it would require a parallel per-chunk dispatch refactor in `bitpack_out_of_line`.
#[derive(Debug)]
pub struct OutOfLineBitpacking {
    compressed_bit_width: u64,
    uncompressed_bit_width: u64,
}

impl OutOfLineBitpacking {
    pub fn new(compressed_bit_width: u64, uncompressed_bit_width: u64) -> Self {
        Self {
            compressed_bit_width,
            uncompressed_bit_width,
        }
    }
}

impl BlockCompressor for OutOfLineBitpacking {
    fn compress(&self, data: DataBlock) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
        let DataBlock::FixedWidth(fixed_width) = data else {
            return Err(Error::invalid_input(
                "Out-of-line bitpacking requires fixed-width data",
            ));
        };
        if fixed_width.bits_per_value != self.uncompressed_bit_width {
            return Err(Error::invalid_input(format!(
                "Out-of-line bitpacking expects {}-bit values, got {}",
                self.uncompressed_bit_width, fixed_width.bits_per_value
            )));
        }
        let compressed = match fixed_width.bits_per_value {
            8 => bitpack_out_of_line::<u8>(fixed_width, self.compressed_bit_width as usize),
            16 => bitpack_out_of_line::<u16>(fixed_width, self.compressed_bit_width as usize),
            32 => bitpack_out_of_line::<u32>(fixed_width, self.compressed_bit_width as usize),
            64 => bitpack_out_of_line::<u64>(fixed_width, self.compressed_bit_width as usize),
            128 => bitpack_out_of_line::<u128>(fixed_width, self.compressed_bit_width as usize),
            w => {
                return Err(Error::invalid_input(format!(
                    "Bitpacking word size must be 8, 16, 32, 64, or 128, got {w}"
                )));
            }
        };
        Ok((
            Some(compressed),
            ProtobufUtils21::out_of_line_bitpacking(
                self.uncompressed_bit_width,
                ProtobufUtils21::flat(self.compressed_bit_width, None),
            ),
        ))
    }
}

impl BlockDecompressor for OutOfLineBitpacking {
    fn decompress(&self, data: Option<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let data = require_block_payload(data, "Out-of-line bitpacking")?;
        let word_size = match self.uncompressed_bit_width {
            8 => std::mem::size_of::<u8>(),
            16 => std::mem::size_of::<u16>(),
            32 => std::mem::size_of::<u32>(),
            64 => std::mem::size_of::<u64>(),
            128 => std::mem::size_of::<u128>(),
            w => {
                return Err(Error::invalid_input(format!(
                    "Bitpacking word size must be 8, 16, 32, 64, or 128, got {w}"
                )));
            }
        };
        debug_assert_eq!(data.len() % word_size, 0);
        let total_words = (data.len() / word_size) as u64;
        let block = FixedWidthDataBlock {
            data,
            bits_per_value: self.uncompressed_bit_width,
            num_values: total_words,
            block_info: BlockInfo::new(),
        };

        let unpacked = match self.uncompressed_bit_width {
            8 => unpack_out_of_line::<u8>(
                block,
                num_values as usize,
                self.compressed_bit_width as usize,
            ),
            16 => unpack_out_of_line::<u16>(
                block,
                num_values as usize,
                self.compressed_bit_width as usize,
            ),
            32 => unpack_out_of_line::<u32>(
                block,
                num_values as usize,
                self.compressed_bit_width as usize,
            ),
            64 => unpack_out_of_line::<u64>(
                block,
                num_values as usize,
                self.compressed_bit_width as usize,
            ),
            128 => unpack_out_of_line::<u128>(
                block,
                num_values as usize,
                self.compressed_bit_width as usize,
            ),
            // Unreachable: the first match above already returned `Err` for any
            // unsupported width before we reach this dispatch.
            _ => unreachable!(
                "OutOfLineBitpacking::decompress: unsupported uncompressed_bit_width={}; \
                 supported widths are 8, 16, 32, 64, or 128",
                self.uncompressed_bit_width
            ),
        };
        Ok(DataBlock::FixedWidth(unpacked))
    }
}

#[cfg(test)]
mod test {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{Array, Int8Array, Int64Array};
    use arrow_buffer::ArrowNativeType;
    use arrow_schema::DataType;
    use bytemuck::Pod;
    use lance_bitpacking::{BitPacking, BitPackingUninit};
    use rstest::rstest;

    use super::{
        ELEMS_PER_CHUNK, InlineBitpacking, U128Kernel, bitpack_out_of_line, u128_kernel_for,
        unpack_out_of_line,
    };
    use crate::{
        buffer::LanceBuffer,
        compression::{BlockDecompressor, MiniBlockDecompressor},
        data::{BlockInfo, DataBlock, FixedWidthDataBlock},
        encodings::logical::primitive::miniblock::{
            MiniBlockCompressionContext, MiniBlockCompressor,
        },
        statistics::ComputeStat,
        testing::{TestCases, check_round_trip_encoding_of_data},
    };
    use lance_core::Error;
    use proptest::prelude::*;

    /// The framing context a mini-block page hands a value codec. These tests drive the
    /// codec directly, so they use the same defaults the other physical codecs' tests do.
    fn test_compression_context() -> MiniBlockCompressionContext {
        MiniBlockCompressionContext::new(0, true, true)
    }

    #[rstest]
    #[case::u8(8)]
    #[case::u16(16)]
    #[case::u32(32)]
    #[case::u64(64)]
    fn test_inline_bitpacking_decompress_empty_miniblock(#[case] bit_width: u64) {
        let decompressor = InlineBitpacking::new(bit_width);
        let decompressed =
            MiniBlockDecompressor::decompress(&decompressor, vec![LanceBuffer::empty()], 0)
                .unwrap();

        let DataBlock::FixedWidth(block) = decompressed else {
            panic!("Expected FixedWidth block");
        };
        assert_eq!(block.bits_per_value, bit_width);
        assert_eq!(block.num_values, 0);
        assert_eq!(block.data.len(), 0);
    }

    // Regression test for #7794: the block-level decompressor must short-circuit
    // on num_values == 0 the same way the mini-block decompressor does, instead
    // of reporting a spurious "too small for header" corrupt-file error.
    #[rstest]
    #[case::u8(8)]
    #[case::u16(16)]
    #[case::u32(32)]
    #[case::u64(64)]
    fn test_inline_bitpacking_decompress_empty_block(#[case] bit_width: u64) {
        let decompressor = InlineBitpacking::new(bit_width);
        let decompressed =
            BlockDecompressor::decompress(&decompressor, Some(LanceBuffer::empty()), 0).unwrap();

        let DataBlock::FixedWidth(block) = decompressed else {
            panic!("Expected FixedWidth block");
        };
        assert_eq!(block.bits_per_value, bit_width);
        assert_eq!(block.num_values, 0);
        assert_eq!(block.data.len(), 0);
    }

    fn roundtrip_unchunk<T>(values: &[T], bit_width: usize)
    where
        T: ArrowNativeType + BitPackingUninit + Pod,
    {
        assert!(values.len() <= ELEMS_PER_CHUNK as usize);
        let num_values = values.len() as u64;

        let mut padded = vec![T::from_usize(0).unwrap(); ELEMS_PER_CHUNK as usize];
        padded[..values.len()].copy_from_slice(values);

        let packed_words = ELEMS_PER_CHUNK as usize * bit_width / (std::mem::size_of::<T>() * 8);
        let mut chunk: Vec<T> = Vec::with_capacity(1 + packed_words);
        chunk.push(T::from_usize(bit_width).unwrap());
        let out_len = chunk.len();
        chunk.resize(out_len + packed_words, T::from_usize(0).unwrap());
        unsafe {
            BitPacking::unchecked_pack(bit_width, &padded, &mut chunk[out_len..]);
        }

        let data = LanceBuffer::reinterpret_vec(chunk);
        let decoded = InlineBitpacking::unchunk::<T>(data, num_values).unwrap();
        let DataBlock::FixedWidth(fixed) = decoded else {
            panic!("expected FixedWidth DataBlock");
        };
        let decoded_values = fixed.data.borrow_to_typed_view::<T>();
        assert_eq!(decoded_values.as_ref(), values);
    }

    fn assert_corrupt_unchunk<T>(data: LanceBuffer, num_values: u64, expected_message: &str)
    where
        T: ArrowNativeType + BitPackingUninit + Pod,
    {
        let err = InlineBitpacking::unchunk::<T>(data, num_values).unwrap_err();
        assert!(matches!(err, lance_core::Error::CorruptFile { .. }));
        let err = err.to_string();
        assert!(
            err.contains(expected_message),
            "expected error containing {expected_message:?}, got {err:?}"
        );
    }

    #[test]
    fn unchunk_u32_bw12_tail() {
        let values: Vec<u32> = (0..500).map(|i| ((i * 7) % (1 << 12)) as u32).collect();
        roundtrip_unchunk(&values, 12);
    }

    #[test]
    fn unchunk_u64_bw23_full() {
        let values: Vec<u64> = (0..1024).map(|i| ((i * 3) % (1 << 23)) as u64).collect();
        roundtrip_unchunk(&values, 23);
    }

    #[rstest]
    #[case::too_small_header(LanceBuffer::from(vec![1, 2, 3]), 1, "too small")]
    #[case::misaligned_chunk_size(LanceBuffer::from(vec![0, 0, 0, 0, 0]), 1, "multiple")]
    #[case::too_many_values(
        LanceBuffer::reinterpret_vec(vec![0_u32]),
        ELEMS_PER_CHUNK + 1,
        "expected at most"
    )]
    #[case::payload_size_mismatch(LanceBuffer::reinterpret_vec(vec![12_u32]), 1, "payload")]
    #[case::invalid_bit_width(LanceBuffer::reinterpret_vec(vec![33_u32]), 1, "exceeds")]
    fn unchunk_rejects(
        #[case] data: LanceBuffer,
        #[case] num_values: u64,
        #[case] expected_message: &str,
    ) {
        assert_corrupt_unchunk::<u32>(data, num_values, expected_message);
    }

    #[test_log::test(tokio::test)]
    async fn test_miniblock_bitpack() {
        let test_cases = TestCases::default().with_structural_encodings();

        let arrays = vec![
            Arc::new(Int8Array::from(vec![100; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![1; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![16; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![-1; 1024])) as Arc<dyn Array>,
            Arc::new(Int8Array::from(vec![5; 1])) as Arc<dyn Array>,
        ];
        check_round_trip_encoding_of_data(arrays, &test_cases, HashMap::new()).await;

        for data_type in [DataType::Int16, DataType::Int32, DataType::Int64] {
            let int64_arrays = vec![
                Int64Array::from(vec![3; 1024]),
                Int64Array::from(vec![8; 1024]),
                Int64Array::from(vec![16; 1024]),
                Int64Array::from(vec![100; 1024]),
                Int64Array::from(vec![512; 1024]),
                Int64Array::from(vec![1000; 1024]),
                Int64Array::from(vec![2000; 1024]),
                Int64Array::from(vec![-1; 10]),
            ];

            let mut arrays = vec![];
            for int64_array in int64_arrays {
                arrays.push(arrow_cast::cast(&int64_array, &data_type).unwrap());
            }

            check_round_trip_encoding_of_data(arrays, &test_cases, HashMap::new()).await;
        }
    }

    #[test_log::test(tokio::test)]
    async fn test_bitpack_encoding_verification() {
        use arrow_array::Int32Array;

        // Test bitpacking encoding verification with varied small values that should trigger bitpacking
        let test_cases = TestCases::default()
            .with_expected_encoding("inline_bitpacking")
            .with_structural_encodings();

        // Generate data with varied small values to avoid RLE
        // Mix different values but keep them small to trigger bitpacking
        let mut values = Vec::new();
        for i in 0..2048 {
            values.push(i % 16); // Values 0-15, varied enough to avoid RLE
        }

        let arrays = vec![Arc::new(Int32Array::from(values)) as Arc<dyn Array>];

        // Explicitly disable BSS to ensure bitpacking is tested
        let mut metadata = HashMap::new();
        metadata.insert("lance-encoding:bss".to_string(), "off".to_string());

        check_round_trip_encoding_of_data(arrays, &test_cases, metadata.clone()).await;
    }

    #[test_log::test(tokio::test)]
    async fn test_miniblock_bitpack_zero_chunk_selection() {
        use arrow_array::Int32Array;

        let test_cases = TestCases::default()
            .with_expected_encoding("inline_bitpacking")
            .with_structural_encodings();

        // Build 2048 values: first 1024 all zeros (bit_width=0),
        // next 1024 small varied values to avoid RLE and trigger bitpacking.
        let mut vals = vec![0i32; 1024];
        for i in 0..1024 {
            vals.push(i % 16);
        }

        let arrays = vec![Arc::new(Int32Array::from(vals)) as Arc<dyn Array>];

        // Disable BSS and RLE to prefer bitpacking in selection
        let mut metadata = HashMap::new();
        metadata.insert("lance-encoding:bss".to_string(), "off".to_string());
        metadata.insert("lance-encoding:rle-threshold".to_string(), "0".to_string());

        check_round_trip_encoding_of_data(arrays, &test_cases, metadata).await;
    }

    // End-to-end guard for the version gate: on 2.3 (the only version whose strategy composes
    // the wide bitpacking selector) the chooser must pick inline bitpacking for low-magnitude
    // Decimal128 data and the file must round-trip. `TestEncoding::StructuralSparse` is the 2.3
    // shape, so this exercises the encode to decode path through the per-chunk u128 dispatch
    // rather than just the kernel.
    #[test_log::test(tokio::test)]
    async fn test_decimal128_u128_bitpacking_round_trips_on_2_3() {
        use crate::testing::TestEncoding;
        use arrow_array::Decimal128Array;

        let test_cases = TestCases::default()
            .with_expected_encoding("inline_bitpacking")
            .with_encodings([TestEncoding::StructuralSparse]);

        // 2048 small-magnitude values (bit width well under 128) spanning two 1024-value chunks.
        let values: Vec<i128> = (0..2048).map(|i| (i % 1000) as i128).collect();
        let array = Decimal128Array::from(values)
            .with_precision_and_scale(38, 0)
            .unwrap();
        let arrays = vec![Arc::new(array) as Arc<dyn Array>];

        // Disable BSS so bitpacking (not byte-stream split) is the selected encoding.
        let mut metadata = HashMap::new();
        metadata.insert("lance-encoding:bss".to_string(), "off".to_string());

        check_round_trip_encoding_of_data(arrays, &test_cases, metadata).await;
    }

    #[test]
    fn test_out_of_line_bitpack_raw_tail_roundtrip() {
        let bit_width = 8usize;
        let word_bits = std::mem::size_of::<u32>() as u64 * 8;
        let values: Vec<u32> = (0..1025).map(|i| (i % 200) as u32).collect();
        let input = FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(values.clone()),
            bits_per_value: word_bits,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };

        let compressed = bitpack_out_of_line::<u32>(input, bit_width);
        let compressed_words = compressed.borrow_to_typed_slice::<u32>().to_vec();
        let words_per_chunk = (ELEMS_PER_CHUNK as usize * bit_width).div_ceil(word_bits as usize);
        assert_eq!(
            compressed_words.len(),
            words_per_chunk + (values.len() - ELEMS_PER_CHUNK as usize),
        );

        let compressed_block = FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(compressed_words.clone()),
            bits_per_value: word_bits,
            num_values: compressed_words.len() as u64,
            block_info: BlockInfo::new(),
        };

        let decoded = unpack_out_of_line::<u32>(compressed_block, values.len(), bit_width);
        let decoded_values = decoded.data.borrow_to_typed_slice::<u32>();
        assert_eq!(decoded_values.as_ref(), values.as_slice());
    }

    #[test]
    fn test_out_of_line_bitpack_u128_raw_tail_roundtrip() {
        // u128 mirror of `test_out_of_line_bitpack_raw_tail_roundtrip`. Reachable
        // from `try_bitpack_for_block` when `num_values > 1024 && bits_per_value
        // == 128`, so the OutOfLine path must round-trip for u128 just like the
        // Inline path covered by `test_inline_bitpack_u128_width_24_narrow_u32`.
        // 1025 values trips the runt-tail branch (one packed chunk + raw tail).
        let bit_width = 24usize;
        let word_bits = std::mem::size_of::<u128>() as u64 * 8;
        let values: Vec<u128> = (0..1025).map(|i| (i as u128) & 0xFFFFFF).collect();
        let input = FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(values.clone()),
            bits_per_value: word_bits,
            num_values: values.len() as u64,
            block_info: BlockInfo::new(),
        };

        let compressed = bitpack_out_of_line::<u128>(input, bit_width);
        let compressed_words = compressed.borrow_to_typed_slice::<u128>().to_vec();
        let words_per_chunk = (ELEMS_PER_CHUNK as usize * bit_width).div_ceil(word_bits as usize);
        assert_eq!(
            compressed_words.len(),
            words_per_chunk + (values.len() - ELEMS_PER_CHUNK as usize),
        );

        let compressed_block = FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(compressed_words.clone()),
            bits_per_value: word_bits,
            num_values: compressed_words.len() as u64,
            block_info: BlockInfo::new(),
        };

        let decoded = unpack_out_of_line::<u128>(compressed_block, values.len(), bit_width);
        let decoded_values = decoded.data.borrow_to_typed_slice::<u128>();
        assert_eq!(decoded_values.as_ref(), values.as_slice());
    }

    /// Build a `LanceBuffer` for `InlineBitpacking::unchunk::<u128>` whose 16-byte
    /// header advertises `header_bit_width` and whose body has `body_words` u128
    /// elements. The body is zero-filled — sufficient to exercise validation
    /// without needing a valid bit-packed stream.
    fn make_u128_inline_buffer(header_bit_width: u128, body_words: usize) -> LanceBuffer {
        let mut bytes = Vec::with_capacity(16 + body_words * 16);
        bytes.extend_from_slice(&header_bit_width.to_le_bytes());
        bytes.extend(std::iter::repeat_n(0u8, body_words * 16));
        LanceBuffer::from(bytes)
    }

    #[test]
    fn test_inline_bitpack_u128_corrupt_header_returns_err() {
        // Header advertises bit_width = 200, which exceeds the 128-bit ceiling.
        // This must surface as `InvalidInput` rather than wrapping silently and
        // feeding an out-of-range width into `unchecked_unpack` (UB).
        let body_words = (200 * ELEMS_PER_CHUNK) as usize / (8 * 16);
        let buf = make_u128_inline_buffer(200, body_words);

        let decoder = InlineBitpacking::new(128);
        let err = MiniBlockDecompressor::decompress(&decoder, vec![buf], ELEMS_PER_CHUNK)
            .expect_err("corrupt u128 header must yield Err");
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
    }

    #[test]
    fn test_inline_bitpack_u128_body_length_mismatch_returns_err() {
        // Header is valid (24 bits) but the body has only 1 u128 word — far
        // less than (24 * 1024 / 8) = 3072 bytes = 192 u128 words.
        let buf = make_u128_inline_buffer(24, 1);

        let decoder = InlineBitpacking::new(128);
        let err = MiniBlockDecompressor::decompress(&decoder, vec![buf], ELEMS_PER_CHUNK)
            .expect_err("body-length mismatch must yield Err");
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
    }

    #[test]
    fn test_inline_bitpack_u128_buffer_too_short_returns_err() {
        // Buffer is 15 bytes, one short of the 16-byte u128 header. The entry-point
        // validation in `unchunk_u128_dispatch` must reject this with `InvalidInput`
        // rather than letting `LittleEndian::read_u128` panic on an under-length slice.
        // `num_values` has to be nonzero: an empty page returns an empty block before
        // any header is read.
        let buf = LanceBuffer::from(vec![0u8; std::mem::size_of::<u128>() - 1]);

        let decoder = InlineBitpacking::new(128);
        let err = MiniBlockDecompressor::decompress(&decoder, vec![buf], 1)
            .expect_err("under-length buffer must yield Err");
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
    }

    #[test]
    fn test_inline_bitpack_u128_num_values_exceeds_chunk_returns_err() {
        // num_values > ELEMS_PER_CHUNK (1024) violates the per-chunk capacity
        // invariant. `unchunk_u128_dispatch:511` must reject this before
        // dispatching to a kernel — otherwise the post-unpack
        // `decompressed.truncate(num_values)` would silently no-op (per
        // `Vec::truncate` semantics: when `new_len >= self.len()` truncate
        // returns without changing the Vec), and the function would return a
        // `FixedWidthDataBlock { num_values: 1025, data.len()/16 == 1024 }`
        // — a length that the underlying buffer doesn't actually contain.
        let buf = make_u128_inline_buffer(24, (24 * ELEMS_PER_CHUNK) as usize / (8 * 16));

        let decoder = InlineBitpacking::new(128);
        let err = MiniBlockDecompressor::decompress(&decoder, vec![buf], ELEMS_PER_CHUNK + 1)
            .expect_err("num_values > ELEMS_PER_CHUNK must yield Err");
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
    }

    #[test]
    fn test_inline_bitpack_u128_num_values_zero_short_circuits_with_empty_block() {
        // Positive-path coverage for the `num_values == 0` short-circuit at
        // `unchunk_u128_dispatch:542` — must return an empty 128-bit
        // `FixedWidthDataBlock` *after* the four malformed-header guards
        // succeed, not as a side effect of one of those guards firing.
        // Other tests that pass `num_values: 0` deliberately tripwire on
        // an earlier validation branch (under-length buffer, unsupported
        // width, etc.), so this short-circuit was previously only
        // reachable in production.
        let buf = make_u128_inline_buffer(24, (24 * ELEMS_PER_CHUNK) as usize / (8 * 16));

        let decoder = InlineBitpacking::new(128);
        let block = MiniBlockDecompressor::decompress(&decoder, vec![buf], 0)
            .expect("valid header + body with num_values=0 must succeed");
        let DataBlock::FixedWidth(fw) = block else {
            panic!("expected FixedWidth, got: {block:?}");
        };
        assert_eq!(fw.num_values, 0);
        assert_eq!(fw.bits_per_value, 128);
        assert_eq!(fw.data.len(), 0);
    }

    #[test]
    fn test_inline_bitpack_unsupported_width_returns_err() {
        // 7 is not in {8,16,32,64,128}, so the dispatch arm must return Err (it used to
        // panic via `unimplemented!`). `num_values` has to be nonzero: an empty page
        // returns an empty block before the width is dispatched.
        let decoder = InlineBitpacking::new(7);
        let buf = LanceBuffer::from(vec![0u8; 16]);
        let err = MiniBlockDecompressor::decompress(&decoder, vec![buf], 1)
            .expect_err("unsupported width must yield Err");
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );

        // Same for `BlockDecompressor` dispatch.
        let buf = LanceBuffer::from(vec![0u8; 16]);
        let err = BlockDecompressor::decompress(&decoder, Some(buf), 1)
            .expect_err("unsupported width must yield Err");
        assert!(
            matches!(err, Error::InvalidInput { .. }),
            "expected InvalidInput, got: {err:?}"
        );
    }

    #[test]
    fn test_inline_bitpack_u128_width_24_narrow_u32() {
        // Happy-path round-trip through the framed [16-byte header | body] format
        // for the narrow u32 dispatch path (bit_width <= 32). Width=24 mirrors a
        // Decimal128(7,2)-shaped workload — the motivating use case. The body is
        // produced by `BitPacking::unchecked_pack::<u32>` (FastLanes layout),
        // matching what the new u128 encoder writes for narrow chunks.
        let bit_width: usize = 24;
        let values: Vec<u128> = (0..ELEMS_PER_CHUNK as u128).map(|i| i & 0xFFFFFF).collect();
        let narrow: Vec<u32> = values.iter().map(|&v| v as u32).collect();
        let words_per_chunk = ELEMS_PER_CHUNK as usize * bit_width / 32;
        let mut packed: Vec<u32> = vec![0u32; words_per_chunk];
        unsafe {
            BitPacking::unchecked_pack(bit_width, &narrow, &mut packed);
        }

        let mut bytes = Vec::with_capacity(16 + words_per_chunk * 4);
        bytes.extend_from_slice(&(bit_width as u128).to_le_bytes());
        for word in &packed {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        let buf = LanceBuffer::from(bytes);

        let decoder = InlineBitpacking::new(128);
        let block = MiniBlockDecompressor::decompress(&decoder, vec![buf], ELEMS_PER_CHUNK)
            .expect("framed u128 chunk must decode");
        let DataBlock::FixedWidth(fw) = block else {
            panic!("expected FixedWidth DataBlock");
        };
        assert_eq!(fw.bits_per_value, 128);
        assert_eq!(fw.num_values, ELEMS_PER_CHUNK);
        let decoded = fw.data.borrow_to_typed_slice::<u128>().to_vec();
        assert_eq!(decoded, values);
    }

    /// Helper: end-to-end compress→decompress round-trip through the public
    /// MiniBlockCompressor / MiniBlockDecompressor surface. Returns the decoded
    /// values in original order. Used by the per-regime u128 tests below.
    fn u128_compress_decompress_roundtrip(values: Vec<u128>) -> Vec<u128> {
        let total = values.len() as u64;
        let input = FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(values),
            bits_per_value: 128,
            num_values: total,
            block_info: BlockInfo::new(),
        };

        let codec = InlineBitpacking::new(128);
        let mut block = DataBlock::FixedWidth(input);
        block.compute_stat();
        let (compressed, _enc) =
            MiniBlockCompressor::compress(&codec, test_compression_context(), block)
                .expect("u128 compress must succeed");
        assert_eq!(compressed.num_values, total);
        assert_eq!(compressed.data.len(), 1);

        let buf = compressed.data.into_iter().next().unwrap();
        let mut offset = 0usize;
        let mut decoded: Vec<u128> = Vec::with_capacity(total as usize);
        let mut consumed = 0u64;
        for chunk in &compressed.chunks {
            let chunk_len = chunk.buffer_sizes[0] as usize;
            let chunk_buf = buf.slice_with_length(offset, chunk_len);
            offset += chunk_len;
            let nv = chunk.num_values(consumed, total);
            consumed += nv;
            let block = MiniBlockDecompressor::decompress(&codec, vec![chunk_buf], nv)
                .expect("u128 chunk decompress must succeed");
            let DataBlock::FixedWidth(fw) = block else {
                panic!("expected FixedWidth DataBlock");
            };
            assert_eq!(fw.bits_per_value, 128);
            assert_eq!(fw.num_values, nv);
            decoded.extend_from_slice(fw.data.borrow_to_typed_slice::<u128>().as_ref());
        }
        assert_eq!(decoded.len(), total as usize);
        decoded
    }

    #[test]
    fn test_inline_bitpack_u128_width_40_narrow_u64() {
        // bit_width in (32, 64] routes through the u64 narrow dispatch.
        // Use values with bit 33 set (forces width >= 34) but stay <= 40 bits.
        let mask: u128 = (1u128 << 40) - 1;
        let values: Vec<u128> = (0..ELEMS_PER_CHUNK as u128 * 2 + 7)
            .map(|i| ((i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | (1u128 << 33)) & mask)
            .collect();
        let decoded = u128_compress_decompress_roundtrip(values.clone());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_inline_bitpack_u128_width_80_sequential() {
        // bit_width in (64, 127] routes through the u128 sequential bit-stream.
        // Values with bit 79 set force width >= 80; stay below 2^80 so high bits
        // remain zero and width does not escalate to 128.
        let mask: u128 = (1u128 << 80) - 1;
        let values: Vec<u128> = (0..ELEMS_PER_CHUNK as u128 + 5)
            .map(|i| ((i.wrapping_mul(0xDEAD_BEEF_CAFE_F00D)) | (1u128 << 79)) & mask)
            .collect();
        let decoded = u128_compress_decompress_roundtrip(values.clone());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_inline_bitpack_u128_width_128_all_negatives_memcpy() {
        // i128 negatives have the sign bit set ⇒ as u128, bit 127 is set ⇒
        // chunk bit_width = 128 ⇒ no compression possible; encoder writes the
        // raw u128 stream. Decoder must reproduce bit-for-bit.
        let raw_negatives: Vec<i128> = (0..ELEMS_PER_CHUNK as i128 + 3)
            .map(|i| -(i + 1) * 0x0123_4567_89AB_CDEF)
            .collect();
        let values: Vec<u128> = raw_negatives.iter().map(|&v| v as u128).collect();
        let decoded = u128_compress_decompress_roundtrip(values.clone());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_inline_bitpack_u128_mixed_sign_small_magnitude_routes_to_memcpy() {
        // Sign-safety regression test: a chunk with mixed positive/negative i128
        // values where the magnitudes are small (would fit in 24 bits in absolute
        // value) MUST still route to bit_width=128 / memcpy because the negatives'
        // two's-complement bit pattern has the high bit set when reinterpreted as
        // u128. If a future change to `Stat::BitWidth` ever computes width from
        // signed magnitude instead of OR-folding the raw bits, this test fires:
        // the narrow dispatch debug_assert (`v >> bit_width == 0`) trips on the
        // first negative, or the round-trip diverges.
        let mut raw: Vec<i128> = Vec::with_capacity(ELEMS_PER_CHUNK as usize + 17);
        for i in 0..ELEMS_PER_CHUNK as i128 + 17 {
            // Alternate small positives and small negatives, magnitudes <= 2^20.
            let mag = (i.wrapping_mul(7) & 0xF_FFFF) + 1;
            raw.push(if i % 2 == 0 { mag } else { -mag });
        }
        let values: Vec<u128> = raw.iter().map(|&v| v as u128).collect();
        let decoded = u128_compress_decompress_roundtrip(values.clone());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_inline_bitpack_u128_mixed_widths_per_chunk_dispatch() {
        // Three full chunks crafted so each lands in a different dispatch
        // regime: chunk 0 → u32 narrow (width=24), chunk 1 → u64 narrow
        // (width=40), chunk 2 → u128 sequential (width=80). This exercises the
        // per-chunk Stat::BitWidth driving distinct kernels in the same array.
        let mask24: u128 = (1u128 << 24) - 1;
        let mask40: u128 = (1u128 << 40) - 1;
        let mask80: u128 = (1u128 << 80) - 1;

        let mut values: Vec<u128> = Vec::with_capacity(ELEMS_PER_CHUNK as usize * 3);
        // Chunk 0: width 24
        for i in 0..ELEMS_PER_CHUNK as u128 {
            values.push((i.wrapping_mul(31) | (1u128 << 23)) & mask24);
        }
        // Chunk 1: width 40
        for i in 0..ELEMS_PER_CHUNK as u128 {
            values.push((i.wrapping_mul(0x1_0000_0001) | (1u128 << 39)) & mask40);
        }
        // Chunk 2: width 80
        for i in 0..ELEMS_PER_CHUNK as u128 {
            values.push((i.wrapping_mul(0x0BAD_F00D_DEAD_BEEF) | (1u128 << 79)) & mask80);
        }

        let decoded = u128_compress_decompress_roundtrip(values.clone());
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_u128_kernel_for_exhaustive_table_and_roundtrip() {
        // Pin the dispatch table for every legal `bit_width` (0..=128). The
        // table here is a hand-written ground truth; if `u128_kernel_for`
        // changes, this assertion forces the change to be deliberate. We
        // additionally exercise an end-to-end round-trip at every width to
        // catch any width where encode and decode disagree on the kernel.
        for bit_width in 0usize..=128 {
            let expected = match bit_width {
                0 => U128Kernel::Zero,
                1..=32 => U128Kernel::NarrowU32,
                33..=64 => U128Kernel::NarrowU64,
                65..=127 => U128Kernel::SequentialU128,
                128 => U128Kernel::Memcpy,
                _ => unreachable!("bit_width loop bound is 128"),
            };
            assert_eq!(
                u128_kernel_for(bit_width),
                expected,
                "u128_kernel_for({bit_width}) drifted from expected table"
            );

            // End-to-end round-trip: one full chunk (1024 values) plus a few
            // tail values, every value masked to `2^bit_width - 1` so the
            // encoder's `Stat::BitWidth` lands exactly at `bit_width` and the
            // expected kernel fires. The mask is identity at width 128.
            let mask: u128 = match bit_width {
                0 => 0,
                128 => u128::MAX,
                w => (1u128 << w) - 1,
            };
            // Pattern that, when masked, still has bit `bit_width - 1` set
            // for any `bit_width >= 1` (so `Stat::BitWidth` does not fall
            // below the target width). Exception: `bit_width == 0` is all
            // zeros by mask construction.
            let high_bit: u128 = if bit_width == 0 {
                0
            } else if bit_width == 128 {
                1u128 << 127
            } else {
                1u128 << (bit_width - 1)
            };
            let total = ELEMS_PER_CHUNK as usize + 7;
            let values: Vec<u128> = (0..total as u128)
                .map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | high_bit) & mask)
                .collect();
            let decoded = u128_compress_decompress_roundtrip(values.clone());
            assert_eq!(
                decoded, values,
                "round-trip mismatch at bit_width={bit_width}"
            );
        }
    }

    #[test]
    fn test_inline_bitpack_u128_compress_decompress_roundtrip() {
        // End-to-end coverage of the u128 inline path through the public
        // `MiniBlockCompressor` / `MiniBlockDecompressor` surface — i.e. the
        // path that real decimal128 columns travel. Spans 2 full chunks plus
        // a short trailing chunk to also exercise the last-chunk handling.
        let total: u64 = ELEMS_PER_CHUNK * 2 + 137;
        let values: Vec<u128> = (0..total as u128).map(|i| i & 0xFFFFFF).collect();
        let input = FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_vec(values.clone()),
            bits_per_value: 128,
            num_values: total,
            block_info: BlockInfo::new(),
        };

        let codec = InlineBitpacking::new(128);
        let mut block = DataBlock::FixedWidth(input);
        block.compute_stat();
        let (compressed, _enc) =
            MiniBlockCompressor::compress(&codec, test_compression_context(), block)
                .expect("u128 compress must succeed");
        assert_eq!(compressed.num_values, total);
        assert_eq!(compressed.chunks.len(), 3);
        assert_eq!(compressed.data.len(), 1);

        let buf = compressed.data.into_iter().next().unwrap();
        let mut offset = 0usize;
        let mut decoded: Vec<u128> = Vec::with_capacity(total as usize);
        let mut consumed = 0u64;
        for chunk in &compressed.chunks {
            let chunk_len = chunk.buffer_sizes[0] as usize;
            let chunk_buf = buf.slice_with_length(offset, chunk_len);
            offset += chunk_len;
            let nv = chunk.num_values(consumed, total);
            consumed += nv;
            let block = MiniBlockDecompressor::decompress(&codec, vec![chunk_buf], nv)
                .expect("u128 chunk decompress must succeed");
            let DataBlock::FixedWidth(fw) = block else {
                panic!("expected FixedWidth DataBlock");
            };
            assert_eq!(fw.bits_per_value, 128);
            assert_eq!(fw.num_values, nv);
            decoded.extend_from_slice(fw.data.borrow_to_typed_slice::<u128>().as_ref());
        }
        assert_eq!(decoded.len(), total as usize);
        assert_eq!(decoded, values);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Fuzz the u128 inline bitpacking round-trip across the full dispatch
        /// matrix. For each case, we pick:
        ///   * `bit_width` in `0..=128` — covers all five kernels (zero-fill,
        ///     u32 narrow, u64 narrow, u128 sequential, memcpy at width=128).
        ///   * `len` in `1..=3 * ELEMS_PER_CHUNK` — exercises 0..2 full chunks
        ///     and any partial trailing chunk size, including exactly
        ///     `2 * ELEMS_PER_CHUNK` and `3 * ELEMS_PER_CHUNK` (multiple-of-1024
        ///     boundaries). Using a range strategy (rather than a fixed length
        ///     plus a separate `tail`) lets proptest's shrinker minimize both
        ///     the value count and the per-element bytes on failure.
        ///   * Per-value bytes via `prop::collection::vec(any::<u128>(), …)` so
        ///     proptest's shrinker can produce minimal counterexamples on
        ///     failure (a hand-rolled PRNG would only shrink the seed and
        ///     produce noise). Each value is masked to `2^bit_width - 1` after
        ///     generation so every value satisfies the narrow-branch invariant
        ///     by construction; at `bit_width == 128` the mask is `u128::MAX`
        ///     (a no-op) so the unmasked sign-bit / full-magnitude case is
        ///     exercised on the memcpy path.
        ///
        /// We assert the public compress→decompress surface is bit-exact for
        /// every (width, count, payload) triple. This is the safety net behind
        /// the per-regime hand-written tests above: if a future refactor of
        /// the dispatch logic, the FastLanes packers, or the sequential
        /// kernel breaks any width, proptest will surface a minimal failing
        /// example.
        #[test]
        fn proptest_inline_bitpack_u128_roundtrip(
            bit_width in 0usize..=128,
            raw in prop::collection::vec(any::<u128>(), 1..=3 * ELEMS_PER_CHUNK as usize),
        ) {
            // bit_width = 0 → mask = 0 (every value clamped to zero, exercises
            // the Zero kernel). bit_width = 128 → mask = u128::MAX (identity,
            // raw values flow through to the memcpy path including any sign
            // bit / full-magnitude pattern). 1..=127 use the standard
            // `(1 << bit_width) - 1` mask.
            let mask: u128 = match bit_width {
                0 => 0,
                128 => u128::MAX,
                w => (1u128 << w) - 1,
            };
            let values: Vec<u128> = raw.into_iter().map(|v| v & mask).collect();
            let decoded = u128_compress_decompress_roundtrip(values.clone());
            prop_assert_eq!(decoded, values);
        }

        /// Fuzz the per-chunk dispatch with **independent** widths per chunk.
        /// The single-width proptest above proves each kernel correct in
        /// isolation across counts and payloads, but it cannot catch a bug
        /// where chunk N's encode/decode is influenced by chunk N-1's kernel
        /// (e.g. shared mutable state in the encoder, off-by-one in the
        /// per-chunk slice math, or a stale `bit_width` reused across chunks).
        ///
        /// Strategy: pick 1..=3 chunk widths from the full `0..=128` range,
        /// fill `ELEMS_PER_CHUNK` values per chunk masked to that chunk's
        /// width, and assert the round-trip is bit-exact. With multiple
        /// chunks per case, all 5×5 kernel-transition combinations
        /// (zero/u32/u64/u128/memcpy → zero/u32/u64/u128/memcpy) are reached
        /// in expectation across the 128-case proptest budget.
        #[test]
        fn proptest_inline_bitpack_u128_mixed_widths(
            bit_widths in prop::collection::vec(0usize..=128, 1..=3),
            seed in any::<u64>(),
        ) {
            // Deterministic per-chunk payload from the shrinkable seed so
            // failures shrink to a minimal `(widths, seed)` pair. Each chunk
            // is filled with `ELEMS_PER_CHUNK` values whose bytes come from a
            // simple xorshift derived from `seed + chunk_index * salt`, then
            // masked to that chunk's width.
            let mut values: Vec<u128> = Vec::with_capacity(
                bit_widths.len() * ELEMS_PER_CHUNK as usize,
            );
            for (chunk_idx, &bit_width) in bit_widths.iter().enumerate() {
                let mask: u128 = match bit_width {
                    0 => 0,
                    128 => u128::MAX,
                    w => (1u128 << w) - 1,
                };
                // `| 1` defends against the xorshift fixed-point at 0: when
                // proptest's shrinker drives `seed` to 0 and chunk 0 is
                // selected, `seed + chunk_idx * salt = 0` would loop on
                // state==0 forever and produce all-zero values, masking
                // a genuine non-zero failure mode. Forcing the low bit
                // makes every chunk's stream non-degenerate.
                let mut state = seed
                    .wrapping_add((chunk_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    | 1;
                for _ in 0..ELEMS_PER_CHUNK as usize {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let hi = state;
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let lo = state;
                    let v = ((hi as u128) << 64) | (lo as u128);
                    values.push(v & mask);
                }
            }
            let decoded = u128_compress_decompress_roundtrip(values.clone());
            prop_assert_eq!(decoded, values);
        }
    }
}
