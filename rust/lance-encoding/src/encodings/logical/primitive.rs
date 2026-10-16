// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{any::Any, collections::VecDeque, fmt::Debug, ops::Range, sync::Arc, vec};

use crate::format::pb21::{self, CompressiveEncoding};
use arrow_buffer::BooleanBufferBuilder;
use arrow_schema::DataType;
use bytes::Bytes;
use futures::{FutureExt, future::BoxFuture};
use lance_core::{
    cache::{CacheKey, CacheKeySchema, Context, DeepSizeOf, KeyBuilder},
    error::Error,
    utils::bit::pad_bytes,
};

use crate::{
    compression::{DecompressionStrategy, MiniBlockDecompressor},
    data::DataBlock,
};
use lance_core::Result;

use crate::{
    EncodingsIo,
    buffer::LanceBuffer,
    data::DataBlockBuilder,
    decoder::{DecodePageTask, DecodedPage, StructuralPageDecoder},
    repdef::RepDefUnraveler,
};

// Reached only by `sparse.rs` tests through `use super::*`.
#[cfg(test)]
use crate::{format::ProtobufUtils21, repdef::CompositeRepDefUnraveler};

pub mod blob;
mod chunk_index;
pub mod constant;
pub mod dict;
pub mod fullzip;
mod layout;
pub mod miniblock;
pub(crate) mod sparse;

mod all_null;
mod field;
mod fullzip_decode;
mod miniblock_decode;
mod structural_encoder;

pub(crate) use all_null::LevelCodec;
pub use all_null::{
    ComplexAllNullPageDecoder, ComplexAllNullScheduler, DecodeComplexAllNullTask,
    SimpleAllNullPageDecoder, SimpleAllNullScheduler,
};
pub use field::{
    StructuralCompositeDecodeArrayTask, StructuralPrimitiveFieldDecoder,
    StructuralPrimitiveFieldScheduler,
};
pub use fullzip_decode::FullZipScheduler;
pub use miniblock_decode::MiniBlockScheduler;
pub use structural_encoder::{PrimitivePageEncoding, PrimitiveStructuralEncoder};

const FILL_BYTE: u8 = 0xFE;
const DEFAULT_DICT_DIVISOR: u64 = 2;
const DEFAULT_DICT_MAX_CARDINALITY: u64 = 100_000;
const DEFAULT_DICT_SIZE_RATIO: f64 = 0.8;
const DEFAULT_DICT_VALUES_COMPRESSION: &str = "lz4";

/// Largest level count a direct legacy u16-value/U8-run RLE frame can prove.
///
/// Mini-block level payload sizes are u16. After the eight-byte frame header, each run needs a
/// two-byte value and a one-byte length, and each U8 run can represent at most 255 levels.
const MAX_LEGACY_RLE_LEVELS: u64 = ((u16::MAX as u64 - std::mem::size_of::<u64>() as u64)
    / (std::mem::size_of::<u16>() as u64 + std::mem::size_of::<u8>() as u64))
    * u8::MAX as u64;

struct PageLoadTask {
    decoder_fut: BoxFuture<'static, Result<Box<dyn StructuralPageDecoder>>>,
    num_rows: u64,
}

/// The ordered initialization-buffer slots for one page.
///
/// Optional slots preserve the semantic position of absent buffers so range
/// declaration and buffer consumption share one representation.
#[derive(Debug)]
struct PageInitialization {
    buffer_ranges: Vec<Option<Range<u64>>>,
}

impl PageInitialization {
    fn new(buffer_ranges: impl IntoIterator<Item = Option<Range<u64>>>) -> Self {
        Self {
            buffer_ranges: buffer_ranges.into_iter().collect(),
        }
    }

    fn required_ranges(&self) -> impl Iterator<Item = &Range<u64>> {
        self.buffer_ranges.iter().flatten()
    }

    fn num_required_ranges(&self) -> usize {
        self.required_ranges().count()
    }

    fn with_buffers(self, buffers: Vec<Bytes>) -> Result<PageInitializationBuffers> {
        let expected = self.num_required_ranges();
        if buffers.len() != expected {
            return Err(Error::invalid_input_source(
                format!(
                    "Page initialization received {} buffers for {expected} declared ranges",
                    buffers.len()
                )
                .into(),
            ));
        }

        let mut buffers = buffers.into_iter();
        let mut buffer_slots = Vec::with_capacity(self.buffer_ranges.len());
        for range in self.buffer_ranges {
            if range.is_some() {
                let buffer = buffers.next().ok_or_else(|| {
                    Error::internal("Page initialization buffer was not assigned")
                })?;
                buffer_slots.push(Some(buffer));
            } else {
                buffer_slots.push(None);
            }
        }
        debug_assert!(
            buffers.next().is_none(),
            "validated page initialization buffer count must be exhausted"
        );
        Ok(PageInitializationBuffers { buffer_slots })
    }
}

#[derive(Debug)]
struct PageInitializationBuffers {
    buffer_slots: Vec<Option<Bytes>>,
}

impl PageInitializationBuffers {
    fn try_into_array<const N: usize>(self, layout: &str) -> Result<[Option<Bytes>; N]> {
        let actual = self.buffer_slots.len();
        self.buffer_slots.try_into().map_err(|_| {
            Error::internal(format!(
                "{layout} page initialization declared {actual} buffer slots, expected {N}"
            ))
        })
    }

    fn into_required(self, layout: &str) -> Result<Vec<Bytes>> {
        self.buffer_slots
            .into_iter()
            .enumerate()
            .map(|(index, buffer)| {
                buffer.ok_or_else(|| {
                    Error::internal(format!(
                        "{layout} page initialization buffer slot {index} is absent"
                    ))
                })
            })
            .collect()
    }
}

/// A trait for figuring out how to schedule the data within
/// a single page.
trait StructuralPageScheduler: std::fmt::Debug + Send {
    /// Whether this page requires initialized metadata before scheduling.
    fn needs_initialization(&self) -> bool {
        true
    }

    /// Describes the file byte ranges this page must read to initialize its
    /// metadata (chunk metadata, dictionary, repetition index, ...).
    ///
    /// The field scheduler sorts the combined I/O request by file offset and
    /// restores buffers to the declared slots afterward.
    fn init_layout(&self) -> Result<PageInitialization>;

    /// Returns initialized state from the bytes declared by [`Self::init_layout`].
    ///
    /// The caller attaches the returned state with [`Self::try_load`].
    fn init_from_buffers<'a>(
        &'a mut self,
        buffers: PageInitializationBuffers,
        io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>>;

    /// Tries to attach metadata from a previous initialization.
    fn try_load(&mut self, data: &Arc<dyn CachedPageData>) -> Result<()>;

    // Identifies the requested decode shape (e.g. blob descriptor struct vs
    // raw bytes). Blob columns can produce multiple page scheduler variants
    // for the same physical column depending on the target field's data type,
    // and the cached page state types differ per variant. The stable view is
    // mixed into the cache key so different variants do not collide.
    fn cache_view(&self) -> PageCacheView {
        PageCacheView::Native
    }

    /// Schedules the read of the given ranges in the page
    ///
    /// The read may be split into multiple "shards" if the page is extremely large.
    /// Each shard maps to one or more rows and can be decoded independently.
    ///
    /// Note: this sharding is for splitting up very large pages into smaller reads to
    /// avoid buffering too much data in memory.  It is not related to the batch size or
    /// compute units in any way.
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Vec<PageLoadTask>>;
}

/// Metadata describing the decoded size of a mini-block
#[derive(Debug)]
struct ChunkMeta {
    num_values: u64,
    chunk_size_bytes: u64,
    offset_bytes: u64,
}

/// A chunk that has been loaded by the miniblock scheduler (but not
/// yet decoded)
#[derive(Debug)]
struct LoadedChunk {
    data: LanceBuffer,
    items_in_chunk: u64,
    byte_range: Range<u64>,
    chunk_idx: usize,
}

impl Clone for LoadedChunk {
    fn clone(&self) -> Self {
        Self {
            // Safe as we always create borrowed buffers here
            data: self.data.clone(),
            items_in_chunk: self.items_in_chunk,
            byte_range: self.byte_range.clone(),
            chunk_idx: self.chunk_idx,
        }
    }
}

pub trait CachedPageData: Any + Send + Sync + DeepSizeOf + 'static {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static>;
}

pub struct NoCachedPageData;

impl DeepSizeOf for NoCachedPageData {
    fn deep_size_of_children(&self, _ctx: &mut Context) -> usize {
        0
    }
}
impl CachedPageData for NoCachedPageData {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync + 'static> {
        self
    }
}

#[doc = "Deprecated compatibility wrapper; new code uses per-page PageDataCacheKey entries."]
pub struct CachedFieldData {
    pages: Vec<Arc<dyn CachedPageData>>,
}

impl DeepSizeOf for CachedFieldData {
    fn deep_size_of_children(&self, ctx: &mut Context) -> usize {
        self.pages.deep_size_of_children(ctx)
    }
}

/// Compatibility key for the former whole-field page metadata cache.
#[derive(Debug, Clone)]
pub struct FieldDataCacheKey {
    pub column_index: u32,
    pub view_tag: String,
}

impl CacheKey for FieldDataCacheKey {
    type ValueType = CachedFieldData;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("{}:{}", self.column_index, self.view_tag).into()
    }

    fn type_name() -> &'static str {
        "FieldData"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.encoding.logical.primitive.field-data-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.column_index);
        builder.write_str(&self.view_tag);
    }
}

/// Sized wrapper around a page's cached metadata so [`PageDataCacheKey`] can use
/// the sized [`CacheKey`] path; the concrete per-layout state still lives behind
/// `dyn CachedPageData`.
pub struct CachedPage {
    data: Arc<dyn CachedPageData>,
}

impl DeepSizeOf for CachedPage {
    fn deep_size_of_children(&self, ctx: &mut Context) -> usize {
        self.data.deep_size_of_children(ctx)
    }
}

// Cache key for a single page's initialized metadata, keyed per page (not per
// column) so a read populates and reuses only the pages it touches.
//
// `view` distinguishes decode shapes of one physical column: a blob column
// materializes as a `Struct<position, size>` descriptor in one variant and raw
// `LargeBinary` in another, each caching a different `CachedPageData` type.
// Without it, two readers at the same column/page but different shapes collide
// and crash on a downcast.
/// Decode shape used to isolate cached page metadata for one physical column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PageCacheView {
    /// The column's native non-blob representation.
    Native,
    /// Blob position and size descriptors materialized as a struct.
    BlobDescriptor,
    /// Blob contents materialized as binary values.
    BlobPayload,
}

impl PageCacheView {
    fn cache_tag(self) -> &'static str {
        match self {
            Self::Native => "n",
            Self::BlobDescriptor => "bd",
            Self::BlobPayload => "bp",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PageDataCacheKey {
    pub column_index: u32,
    pub page_index: u32,
    pub view: PageCacheView,
    pub target_data_type_tag: Arc<str>,
}

impl CacheKey for PageDataCacheKey {
    type ValueType = CachedPage;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!(
            "v2/{}/{}/{}/{}",
            self.column_index,
            self.page_index,
            self.view.cache_tag(),
            self.target_data_type_tag
        )
        .into()
    }

    fn type_name() -> &'static str {
        "PageData"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.encoding.logical.primitive.page-data-key", 2)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.column_index);
        builder.write_u32(self.page_index);
        builder.write_variant(match self.view {
            PageCacheView::Native => 0,
            PageCacheView::BlobDescriptor => 1,
            PageCacheView::BlobPayload => 2,
        });
        builder.write_str(&self.target_data_type_tag);
    }
}

// We align and pad mini-blocks to 8 byte boundaries for two reasons.  First,
// to allow us to store a chunk size in 12 bits.
//
// If we directly record the size in bytes with 12 bits we would be limited to
// 4KiB which is too small.  Since we know each mini-block consists of 8 byte
// words we can store the # of words instead which gives us 32KiB.
//
// Second, each chunk in a mini-block is aligned to 8 bytes.  This allows multi-byte
// values like offsets to be stored in a mini-block and safely read back out.  It also
// helps ensure zero-copy reads in cases where zero-copy is possible (e.g. no decoding
// needed).
//
// Note: by "aligned to 8 bytes" we mean BOTH "aligned to 8 bytes from the start of
// the page" and "aligned to 8 bytes from the start of the file."
const MINIBLOCK_ALIGNMENT: usize = 8;

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::sync::Arc;

    use crate::buffer::LanceBuffer;
    use crate::compression::BlockCompressor;
    use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
    use crate::encodings::physical::rle::{RleEncoder, RunLengthWidth};

    use super::PageInitialization;

    #[test]
    fn page_initialization_preserves_optional_buffer_slots() {
        let initialization = PageInitialization::new([None, Some(20..22), None, Some(10..11)]);
        assert_eq!(
            initialization
                .required_ranges()
                .cloned()
                .collect::<Vec<_>>(),
            [20..22, 10..11]
        );

        let buffers = initialization
            .with_buffers(vec![
                bytes::Bytes::from_static(b"ab"),
                bytes::Bytes::from_static(b"c"),
            ])
            .unwrap();
        let [first, second, third, fourth] = buffers.try_into_array("test").unwrap();
        assert!(first.is_none());
        assert_eq!(second.unwrap(), bytes::Bytes::from_static(b"ab"));
        assert!(third.is_none());
        assert_eq!(fourth.unwrap(), bytes::Bytes::from_static(b"c"));

        let error = PageInitialization::new([Some(0..1)])
            .with_buffers(Vec::new())
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
        assert!(error.to_string().contains("1 declared ranges"));
    }

    pub(super) fn encoded_u16_frame(
        levels: &[u16],
        run_length_width: RunLengthWidth,
    ) -> LanceBuffer {
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::reinterpret_slice(Arc::from(levels)),
            bits_per_value: 16,
            num_values: levels.len() as u64,
            block_info: BlockInfo::new(),
        });
        BlockCompressor::compress(&RleEncoder::with_run_length_width(run_length_width), block)
            .unwrap()
            .0
            .unwrap()
    }
}
