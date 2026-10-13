// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sparse dictionary indices are initialized once; item payloads are read on demand.

use super::*;
use crate::compression::{create_fixed_width_block_decompressor, field_metadata_params};
use crate::data::DictionaryDataBlock;
use crate::encodings::physical::checked_fixed_values;
use crate::encodings::physical::sequence::{BlockCost, select_block_codec};

pub(in crate::encodings::logical::primitive) fn validate(
    layout: &pb23::IndexedValues,
    field: &Field,
    decompressors: &dyn DecompressionStrategy,
) -> Result<usize> {
    if !matches!(
        field.data_type(),
        DataType::Binary | DataType::LargeBinary | DataType::Utf8 | DataType::LargeUtf8
    ) {
        return Err(Error::invalid_input(
            "Indexed sparse values require a binary or string leaf",
        ));
    }
    let indices = layout
        .indices
        .as_ref()
        .ok_or_else(|| Error::invalid_input("Indexed values are missing index compression"))?;
    create_fixed_width_block_decompressor(indices, 32)?;
    let items = layout
        .items
        .as_ref()
        .ok_or_else(|| Error::invalid_input("Indexed values are missing dictionary items"))?;
    if items.num_items == 0
        || items.num_items > u32::MAX as u64
        || items.num_items != items.num_visible_items
        || items.levels.is_some()
        || !matches!(
            items.addressing,
            Some(pb23::zipped_values::Addressing::BitsPerOffset(32 | 64))
        )
    {
        return Err(Error::invalid_input(
            "Indexed dictionary item count or addressing is invalid",
        ));
    }
    let compression = items
        .compression
        .as_ref()
        .ok_or_else(|| Error::invalid_input("Indexed dictionary item compression is missing"))?;
    decompressors.create_variable_per_value_decompressor(compression)?;
    Ok(3)
}

pub(super) fn encode(
    ctx: &PrimitiveEncodeContext,
    indices: DataBlock,
    items: DataBlock,
    compression: &dyn CompressionStrategy,
    row_number: u64,
) -> Result<(pb23::value_plane::Codec, Vec<LanceBuffer>)> {
    let DataBlock::FixedWidth(indices) = indices else {
        return Err(Error::internal("Dictionary indices must be fixed-width"));
    };
    let count = usize_from_u64(indices.num_values, "Dictionary index count")?;
    let codec = select_block_codec(
        &indices,
        &[0..count],
        &field_metadata_params(&ctx.field),
        BlockCost::default(),
        false,
    )?;
    let (index_buffer, indices) = codec.compress(indices)?;
    let index_buffer = index_buffer.unwrap_or_else(|| LanceBuffer::from(Vec::new()));
    let num_items = items.num_values();
    let field = PrimitiveStructuralEncoder::build_dict_values_compressor_field(&ctx.field)?;
    let encoded = PrimitiveStructuralEncoder::encode_full_zip(
        ctx.column_idx,
        &field,
        compression,
        items,
        SerializedRepDefs::new(None, None, vec![DefinitionInterpretation::AllValidItem]),
        row_number,
        num_items,
    )?;
    let PageEncoding::Structural(layout) = encoded.description else {
        return Err(Error::internal(
            "Dictionary item encoder returned non-structural metadata",
        ));
    };
    let planar = crate::format::planar::from_dense_execution(layout)?;
    let Some(pb23::ValuePlane {
        codec: Some(pb23::value_plane::Codec::Zipped(mut items)),
    }) = planar.values
    else {
        return Err(Error::internal(
            "Dictionary item encoder returned non-zipped values",
        ));
    };
    items.levels = None;
    let mut buffers = vec![index_buffer];
    buffers.extend(encoded.data);
    Ok((
        pb23::value_plane::Codec::Indexed(pb23::IndexedValues {
            indices: Some(indices),
            items: Some(items),
        }),
        buffers,
    ))
}

struct CachedDictionary {
    indices: Arc<[u32]>,
    items: Arc<dyn CachedPageData>,
}
impl DeepSizeOf for CachedDictionary {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.indices.len() * 4 + self.items.deep_size_of_children(context)
    }
}
impl CachedPageData for CachedDictionary {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }
}

pub(super) struct IndexedScheduler {
    indices: Arc<dyn BlockDecompressor>,
    index_range: Range<u64>,
    slots: u64,
    num_items: u64,
    items: FullZipScheduler,
    state: Option<Arc<CachedDictionary>>,
    cache: Option<Arc<LanceCache>>,
}
impl std::fmt::Debug for IndexedScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexedScheduler")
            .field("slots", &self.slots)
            .finish_non_exhaustive()
    }
}
impl IndexedScheduler {
    pub(super) fn try_new(
        buffers: &[(u64, u64)],
        priority: u64,
        slots: u64,
        layout: &pb23::IndexedValues,
        field: &Field,
        decompressors: &dyn DecompressionStrategy,
    ) -> Result<Self> {
        if buffers.len() != validate(layout, field, decompressors)? {
            return Err(Error::invalid_input(
                "Indexed dictionary buffer count mismatch",
            ));
        }
        let index_encoding = layout
            .indices
            .as_ref()
            .ok_or_else(|| Error::invalid_input("Missing dictionary indices"))?;
        crate::compression::validate_fixed_width_block_sequence(index_encoding, 32, slots)?;
        let item_layout = layout
            .items
            .as_ref()
            .ok_or_else(|| Error::invalid_input("Missing dictionary items"))?;
        if create_fixed_width_block_decompressor(index_encoding, 32)?.requires_payload()
            != (buffers[0].1 > 0)
        {
            return Err(Error::invalid_input(
                "Dictionary index payload does not match its codec",
            ));
        }
        let count = item_layout.num_items + 1;
        if !buffers[2].1.is_multiple_of(count) || ![1, 2, 4, 8].contains(&(buffers[2].1 / count)) {
            return Err(Error::invalid_input(
                "Dictionary item offset buffer has an invalid length",
            ));
        }
        let mut items = FullZipScheduler::try_new(
            &buffers[1..],
            priority,
            item_layout.num_items,
            &item_layout.full_zip(vec![metadata::RepDefLayer::RepdefAllValidItem as i32], None)?,
            decompressors,
        )?;
        items.enable_cache = true;
        Ok(Self {
            indices: Arc::from(create_fixed_width_block_decompressor(index_encoding, 32)?),
            index_range: checked_buffer_range(buffers[0].0, buffers[0].1, "dictionary indices")?,
            slots,
            num_items: item_layout.num_items,
            items,
            state: None,
            cache: None,
        })
    }
}
impl StructuralPageScheduler for IndexedScheduler {
    fn set_cache(&mut self, cache: Arc<LanceCache>) {
        self.cache = (cache.capacity_bytes() != Some(0)).then_some(cache);
    }

    fn init_layout(&self) -> Result<PageInitialization> {
        let mut ranges = vec![
            self.indices
                .requires_payload()
                .then(|| self.index_range.clone()),
        ];
        ranges.extend(self.items.init_layout()?.buffer_ranges);
        Ok(PageInitialization::new(ranges))
    }
    fn init_from_buffers<'a>(
        &'a mut self,
        buffers: PageInitializationBuffers,
        io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        let io = io.clone();
        async move {
            let mut buffers = buffers.buffer_slots.into_iter();
            let payload = buffers
                .next()
                .ok_or_else(|| Error::invalid_input("Missing dictionary index slot"))?;
            let decoder = self.indices.clone();
            let slots = self.slots;
            let num_items = self.num_items;
            let indices = spawn_cpu(move || -> Result<Arc<[u32]>> {
                let data = decoder.decompress(
                    payload.map(|bytes| LanceBuffer::from_bytes(bytes, 4)),
                    slots,
                )?;
                let DataBlock::FixedWidth(data) = data else {
                    return Err(Error::invalid_input(
                        "Dictionary indices are not fixed width",
                    ));
                };
                let indices: Arc<[u32]> =
                    Arc::from(checked_fixed_values::<u32>(&data, "Dictionary indices")?.as_ref());
                if indices.iter().any(|&index| u64::from(index) >= num_items) {
                    return Err(Error::invalid_input("Dictionary index exceeds item count"));
                }
                Ok(indices)
            })
            .await?;
            let item_buffers = buffers.collect::<Vec<_>>();
            // Validate the addressing table before passing it to the shared engine.
            let offsets = item_buffers
                .first()
                .and_then(Option::as_ref)
                .ok_or_else(|| Error::invalid_input("Missing dictionary item offsets"))?;
            let count = usize_from_u64(self.num_items + 1, "Dictionary offset count")?;
            let width = offsets.len() / count;
            if ![1, 2, 4, 8].contains(&width) || !offsets.len().is_multiple_of(count) {
                return Err(Error::invalid_input(
                    "Dictionary offset buffer has an invalid length",
                ));
            }
            let mut previous = 0;
            for (i, bytes) in offsets.chunks_exact(width).enumerate() {
                let mut scalar = [0; 8];
                scalar[..width].copy_from_slice(bytes);
                let value = u64::from_le_bytes(scalar);
                if (i == 0 && value != 0) || value < previous || value > self.items.data_buf_size {
                    return Err(Error::invalid_input(
                        "Dictionary item offsets are out of bounds",
                    ));
                }
                previous = value;
            }
            if previous != self.items.data_buf_size {
                return Err(Error::invalid_input(
                    "Dictionary final offset does not match payload size",
                ));
            }
            let items = self
                .items
                .init_from_buffers(
                    PageInitializationBuffers {
                        buffer_slots: item_buffers,
                    },
                    &io,
                )
                .await?;
            Ok(Arc::new(CachedDictionary { indices, items }) as Arc<dyn CachedPageData>)
        }
        .boxed()
    }
    fn try_load(&mut self, state: &Arc<dyn CachedPageData>) -> Result<()> {
        let state = state
            .clone()
            .as_arc_any()
            .downcast::<CachedDictionary>()
            .map_err(|_| Error::invalid_input("Invalid indexed dictionary cache"))?;
        self.items.try_load(&state.items)?;
        self.state = Some(state);
        Ok(())
    }
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Vec<PageLoadTask>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| Error::internal("Dictionary is not initialized"))?;
        let count = validate_slice_ranges(ranges, self.slots, "dictionary slots")?;
        if count == 0 {
            return Ok(vec![]);
        }
        let (indices, item_ranges) = if ranges == [0..self.slots] {
            // A full-page scan already uses the original dictionary domain.
            // Keep its indices and read items contiguously without remapping.
            (state.indices.clone(), vec![0..self.num_items])
        } else {
            let mut indices = Vec::with_capacity(usize_from_u64(count, "dictionary selection")?);
            for range in ranges {
                indices.extend_from_slice(&state.indices[range.start as usize..range.end as usize]);
            }
            let mut selected = indices.clone();
            selected.sort_unstable();
            selected.dedup();
            let item_ranges = selected
                .iter()
                .map(|&index| u64::from(index)..u64::from(index) + 1)
                .collect::<Vec<_>>();
            for index in &mut indices {
                *index = selected
                    .binary_search(index)
                    .map_err(|_| Error::internal("Dictionary selection lost an index"))?
                    as u32;
            }
            (Arc::from(indices), item_ranges)
        };
        // Reuse zipped range coalescing; duplicate requested items are loaded once.
        let cached_io;
        let item_io = if ranges != [0..self.slots] {
            if let Some(cache) = &self.cache {
                cached_io = Arc::new(DictionaryItemsIo {
                    inner: io.clone(),
                    cache: cache.clone(),
                }) as Arc<dyn EncodingsIo>;
                &cached_io
            } else {
                io
            }
        } else {
            io
        };
        let tasks = self.items.schedule_ranges(&item_ranges, item_io)?;
        Ok(vec![PageLoadTask {
            num_rows: count,
            decoder_fut: async move {
                let mut decoders = Vec::with_capacity(tasks.len());
                for task in tasks {
                    decoders.push((task.decoder_fut.await?, task.num_rows));
                }
                let items = spawn_cpu(move || -> Result<DataBlock> {
                    let mut builder = DataBlockBuilder::with_capacity_estimate(0);
                    for (mut decoder, count) in decoders {
                        let data = decoder.drain(count)?.decode()?.data;
                        builder.append(&data, 0..data.num_values())?;
                    }
                    Ok(builder.finish())
                })
                .await?;
                Ok(Box::new(IndexedDecoder {
                    indices,
                    offset: 0,
                    items: Arc::new(items),
                }) as Box<dyn StructuralPageDecoder>)
            }
            .boxed(),
        }])
    }
}

#[derive(Debug)]
struct IndexedDecoder {
    indices: Arc<[u32]>,
    offset: usize,
    items: Arc<DataBlock>,
}
impl StructuralPageDecoder for IndexedDecoder {
    fn num_rows(&self) -> u64 {
        self.indices.len() as u64
    }
    fn drain(&mut self, count: u64) -> Result<Box<dyn DecodePageTask>> {
        let end = self
            .offset
            .checked_add(usize_from_u64(count, "Dictionary drain")?)
            .filter(|&end| end <= self.indices.len())
            .ok_or_else(|| Error::invalid_input("Dictionary drain exceeds selected rows"))?;
        let indices = LanceBuffer::reinterpret_vec(self.indices[self.offset..end].to_vec());
        self.offset = end;
        Ok(Box::new(IndexedTask {
            indices,
            count,
            items: self.items.clone(),
        }))
    }
}
#[derive(Debug)]
struct IndexedTask {
    indices: LanceBuffer,
    count: u64,
    items: Arc<DataBlock>,
}
impl DecodePageTask for IndexedTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        Ok(DecodedPage {
            data: DataBlock::Dictionary(DictionaryDataBlock::from_parts(
                FixedWidthDataBlock {
                    data: self.indices,
                    bits_per_value: 32,
                    num_values: self.count,
                    block_info: BlockInfo::new(),
                },
                self.items.as_ref().clone(),
            )),
            repdef: RepDefUnraveler::new(
                None,
                None,
                Arc::new([DefinitionInterpretation::AllValidItem]),
                self.count,
            ),
        })
    }
}

// Item bytes share the reader's capacity and eviction policy. Keys are physical
// ranges within its file namespace, so entries can be reused across selections.
#[derive(Debug)]
struct DictionaryItemsIo {
    inner: Arc<dyn EncodingsIo>,
    cache: Arc<LanceCache>,
}

struct DictionaryItemBytes(Bytes);

impl DeepSizeOf for DictionaryItemBytes {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.0.len()
    }
}

struct DictionaryItemKey(Range<u64>);

impl CacheKey for DictionaryItemKey {
    type ValueType = DictionaryItemBytes;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("{}:{}", self.0.start, self.0.end).into()
    }

    fn type_name() -> &'static str {
        "IndexedDictionaryItemBytes"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.encoding.indexed-dictionary-item", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.0.start);
        builder.write_u64(self.0.end);
    }
}

impl EncodingsIo for DictionaryItemsIo {
    fn submit_request(
        &self,
        ranges: Vec<Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
        let cache = self.cache.clone();
        let io = self.inner.clone();
        // Cache lookup is asynchronous, but reads must start during scheduling.
        // Waiting for a decoder to poll this future serializes sibling pages.
        tokio::spawn(async move {
            let mut output = vec![None; ranges.len()];
            let mut misses = Vec::new();
            for (position, range) in ranges.into_iter().enumerate() {
                let key = DictionaryItemKey(range);
                if let Some(value) = cache.get_with_key(&key).await {
                    output[position] = Some(value.0.clone());
                } else {
                    misses.push((position, key));
                }
            }
            if !misses.is_empty() {
                let ranges = misses.iter().map(|(_, key)| key.0.clone()).collect();
                let loaded = io.submit_request(ranges, priority).await?;
                if loaded.len() != misses.len() {
                    return Err(Error::internal("Dictionary item read count mismatch"));
                }
                for ((position, key), bytes) in misses.into_iter().zip(loaded) {
                    // Coalesced reads may share a much larger allocation. Cache
                    // an owned item so its accounted size matches retained memory.
                    if cache
                        .capacity_bytes()
                        .is_none_or(|capacity| bytes.len() < capacity)
                    {
                        let value = Arc::new(DictionaryItemBytes(Bytes::copy_from_slice(&bytes)));
                        cache.insert_with_key(&key, value).await;
                    }
                    output[position] = Some(bytes);
                }
            }
            output
                .into_iter()
                .map(|bytes| bytes.ok_or_else(|| Error::internal("Missing dictionary item read")))
                .collect()
        })
        .map(|result| result.map_err(Error::from).and_then(|result| result))
        .boxed()
    }
}
