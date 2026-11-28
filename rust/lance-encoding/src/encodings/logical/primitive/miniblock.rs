// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Routines for encoding and decoding miniblock data
//!
//! Miniblock encoding is one of the two structural encodings in Lance 2.1.
//! In this approach the data is compressed into a series of chunks put into
//! a single buffer.
//!
//! A chunk must be encoded or decoded as a unit.  There is a small amount of
//! chunk metadata such as the number and size of each buffer in the chunk.
//!
//! Any form of compression can be used since we are compressing and decompressing
//! entire chunks.
use crate::{buffer::LanceBuffer, data::DataBlock, format::metadata::CompressiveEncoding};

use std::sync::LazyLock;

use lance_core::{Error, Result};

pub const MAX_MINIBLOCK_BYTES: u64 = 8 * 1024 - 6;
/// Maximum framed chunk size in a u32 metadata word.
pub(crate) const MAX_LARGE_MINIBLOCK_BYTES: u64 = 1 << 31;

const DEFAULT_MAX_MINIBLOCK_VALUES: u64 = 4096;
/// Largest count expressible by the four bits of `log_num_values`.
pub(crate) const MAX_CONFIGURABLE_MINIBLOCK_VALUES: u64 = 32768;

static CONFIGURED_MAX_MINIBLOCK_VALUES: LazyLock<Option<u64>> = LazyLock::new(|| {
    std::env::var("LANCE_MINIBLOCK_MAX_VALUES")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(|values| values.clamp(1, MAX_CONFIGURABLE_MINIBLOCK_VALUES))
});

pub static MAX_MINIBLOCK_VALUES: LazyLock<u64> =
    LazyLock::new(|| CONFIGURED_MAX_MINIBLOCK_VALUES.unwrap_or(DEFAULT_MAX_MINIBLOCK_VALUES));

/// Maximum number of rep/def levels the structural planner should place into
/// a single mini-block chunk.
pub fn max_repdef_levels_per_chunk(bits_per_level: u64) -> u64 {
    debug_assert!(bits_per_level > 0);
    const REPDEF_BUDGET_BITS: u64 = 16 * 1024 * 8;
    let budgeted_levels = REPDEF_BUDGET_BITS / bits_per_level;
    budgeted_levels.min(u16::MAX as u64)
}

/// Page data that has been compressed into a series of chunks put into
/// a single buffer.
#[derive(Debug)]
pub struct MiniBlockCompressed {
    /// The buffers of compressed data
    pub data: Vec<LanceBuffer>,
    /// Describes the size of each chunk
    pub chunks: Vec<MiniBlockChunk>,
    /// The number of values in the entire page
    pub num_values: u64,
}

/// Per-page framing details that can affect a mini-block compressor's choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MiniBlockCompressionContext {
    common_chunk_buffers: u64,
    support_large_chunk: bool,
    allow_generic_offsets: bool,
    fixed_width_chunk_target: Option<FixedWidthChunkTarget>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FixedWidthChunkTarget {
    bytes: u64,
    max_values: u64,
}

impl MiniBlockCompressionContext {
    /// Creates the framing context supplied by the owning mini-block page.
    pub fn new(
        common_chunk_buffers: u64,
        support_large_chunk: bool,
        allow_generic_offsets: bool,
    ) -> Self {
        Self {
            common_chunk_buffers,
            support_large_chunk,
            allow_generic_offsets,
            fixed_width_chunk_target: None,
        }
    }

    pub(crate) fn with_fixed_width_chunk_target(mut self, bytes: u64) -> Self {
        self.fixed_width_chunk_target = Some(FixedWidthChunkTarget {
            bytes,
            max_values: CONFIGURED_MAX_MINIBLOCK_VALUES
                .unwrap_or(MAX_CONFIGURABLE_MINIBLOCK_VALUES),
        });
        self
    }

    /// Reduce the chunk cardinality when the actual dense level buffers do not fit.
    pub(crate) fn reduce_fixed_width_chunk_limit(&mut self, current_values: u64) -> bool {
        let Some(target) = self.fixed_width_chunk_target.as_mut() else {
            return false;
        };
        // Zero is reserved for the final chunk, so a full chunk needs at least two values.
        if current_values <= 2 || current_values > target.max_values {
            return false;
        }
        target.max_values = 1 << (current_values - 1).ilog2();
        true
    }

    pub(crate) fn fixed_width_values_per_chunk(
        self,
        num_values: u64,
        values_per_word: u64,
        num_value_buffers: u64,
        bytes_for_values: impl Fn(u64) -> u64,
    ) -> Result<Option<u64>> {
        let Some(target) = self
            .fixed_width_chunk_target
            .filter(|_| self.supports_large_chunk())
        else {
            return Ok(None);
        };
        // The wire limit includes headers, level buffers and per-buffer alignment.
        let reserved = self.chunk_header_bytes(num_value_buffers)
            + self.common_chunk_buffers * (u16::MAX as u64).next_multiple_of(8)
            + 7 * num_value_buffers;
        let byte_budget = target
            .bytes
            .min(MAX_LARGE_MINIBLOCK_BYTES.saturating_sub(reserved));
        let min_full_chunk = values_per_word.max(2);
        for log in (min_full_chunk.ilog2()..=target.max_values.ilog2()).rev() {
            let values = 1 << log;
            if bytes_for_values(values) <= byte_budget {
                return Ok(Some(values));
            }
        }
        // A page with just one chunk uses the final-chunk count, and can end mid-byte.
        if num_values <= target.max_values && bytes_for_values(num_values) <= byte_budget {
            return Ok(Some(num_values.max(1)));
        }
        Err(Error::invalid_input(format!(
            "Fixed-width mini-block with {num_values} values needs at least {min_full_chunk} \
             values and {} bytes per non-final chunk, but its limits are {} values and {byte_budget} bytes",
            bytes_for_values(min_full_chunk),
            target.max_values
        )))
    }

    pub(crate) fn chunk_header_bytes(self, value_buffers: u64) -> u64 {
        let value_buffer_size_bytes = if self.support_large_chunk { 4 } else { 2 };
        2_u64
            .saturating_add(self.common_chunk_buffers.saturating_mul(2))
            .saturating_add(value_buffers.saturating_mul(value_buffer_size_bytes))
            .next_multiple_of(8)
    }

    pub(crate) fn allows_generic_offsets(self) -> bool {
        self.allow_generic_offsets
    }

    pub(crate) fn supports_large_chunk(self) -> bool {
        self.support_large_chunk
    }
}

/// Describes the size of a mini-block chunk of data
///
/// Mini-block chunks are designed to be small (just a few disk sectors)
/// and contain a power-of-two number of values (except for the last chunk)
///
/// The default chunking budget is 4Ki values and slightly less than 8KiB of
/// value data. With an explicit byte target, 2.2 and later fixed-width writers
/// can reach the wire limit of 32Ki values. `LANCE_MINIBLOCK_MAX_VALUES` can
/// set a process-wide count limit for these chunks.
#[derive(Debug)]
pub struct MiniBlockChunk {
    // The size in bytes of each buffer in the chunk.
    //
    // In Lance 2.1, the chunk size is limited to 32KiB, so only 16-bits are used.
    // Since Lance 2.2, the chunk size uses u32 to support larger chunk size
    pub buffer_sizes: Vec<u32>,
    // The log (base 2) of the number of values in the chunk.  If this is the final chunk
    // then this should be 0 (the number of values will be calculated by subtracting the
    // size of all other chunks from the total size of the page)
    //
    // For example, 1 would mean there are 2 values in the chunk and 15 would mean there
    // are 32Ki values in the chunk.
    //
    // This must be <= 15 to fit the four-bit metadata field.
    pub log_num_values: u8,
}

impl MiniBlockChunk {
    pub(crate) fn log_for_fixed_width_chunk(values: u64) -> u8 {
        if values.is_power_of_two() {
            values.ilog2() as u8
        } else {
            0
        }
    }

    /// Gets the number of values in this block
    ///
    /// This requires `vals_in_prev_blocks` and `total_num_values` because the
    /// last block in a page is a special case which stores 0 for log_num_values
    /// and, in that case, the number of values is determined by subtracting
    /// `vals_in_prev_blocks` from `total_num_values`
    pub fn num_values(&self, vals_in_prev_blocks: u64, total_num_values: u64) -> u64 {
        if self.log_num_values == 0 {
            total_num_values - vals_in_prev_blocks
        } else {
            1 << self.log_num_values
        }
    }
}

/// Trait for compression algorithms that are suitable for use in the miniblock structural encoding
///
/// These compression algorithms should be capable of encoding the data into small chunks
/// where each chunk (except the last) has 2^N values (N can vary between chunks)
pub trait MiniBlockCompressor: std::fmt::Debug + Send + Sync {
    /// Compress a `page` of data into multiple chunks
    ///
    /// See [`MiniBlockCompressed`] for details on how chunks should be sized.
    ///
    /// This method also returns a description of the encoding applied that will be
    /// used at decode time to read the data.
    fn compress(
        &self,
        context: MiniBlockCompressionContext,
        page: DataBlock,
    ) -> Result<(MiniBlockCompressed, CompressiveEncoding)>;
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::unset(None, None)]
    #[case::custom_value(Some("256"), Some(256))]
    #[case::non_power_of_two(Some("9000"), Some(9000))]
    #[case::can_raise_to_32k(Some("32768"), Some(32768))]
    #[case::clamps_zero_to_one(Some("0"), Some(1))]
    #[case::clamps_above_max(Some("99999"), Some(MAX_CONFIGURABLE_MINIBLOCK_VALUES))]
    #[case::invalid_falls_back_to_default(Some("not_a_number"), None)]
    fn test_parse_max_miniblock_values(
        #[case] env_value: Option<&str>,
        #[case] expected: Option<u64>,
    ) {
        // Other tests initialize `MAX_MINIBLOCK_VALUES` from the process-wide environment, so
        // instead of changing `LANCE_MINIBLOCK_MAX_VALUES` here the test re-runs itself in a child
        // process that starts with it. libtest names the thread after the running test, so the
        // child's filter cannot drift from this test's name.
        const CHILD_ENV: &str = "LANCE_TEST_MINIBLOCK_MAX_VALUES_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            command
                .args([
                    "--exact",
                    std::thread::current().name().unwrap(),
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1");
            match env_value {
                Some(value) => command.env("LANCE_MINIBLOCK_MAX_VALUES", value),
                None => command.env_remove("LANCE_MINIBLOCK_MAX_VALUES"),
            };
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "child test failed: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }

        assert_eq!(*MAX_MINIBLOCK_VALUES, expected.unwrap_or(4096));
        let context = MiniBlockCompressionContext::new(0, true, true)
            .with_fixed_width_chunk_target(256 * 1024);
        let num_values = if expected == Some(1) { 1 } else { 32769 };
        assert_eq!(
            context
                .fixed_width_values_per_chunk(num_values, 1, 1, |values| values * 8)
                .unwrap(),
            Some(1 << expected.unwrap_or(32768).ilog2())
        );
    }
}
