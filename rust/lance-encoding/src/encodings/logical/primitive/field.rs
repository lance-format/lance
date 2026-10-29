// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::VecDeque;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;
use std::vec;

use arrow_array::{Array, cast::AsArray, make_array};
use arrow_schema::{DataType, Field as ArrowField};
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::FuturesOrdered;
use futures::{FutureExt, TryStreamExt};
use itertools::Itertools;
use lance_arrow::{DataTypeExt, r#struct::StructArrayExt};
use lance_core::Result;
use lance_core::cache::LanceCache;
use lance_core::datatypes::Field;
use lance_core::error::{Error, LanceOptionExt};
use log::{debug, trace};

use crate::EncodingsIo;
use crate::compression::DecompressionStrategy;
use crate::data::{concat_decoded_arrays, dictionary_batch_limit};
use crate::decoder::{
    ColumnInfo, DecodePageTask, DecodedArray, FilterExpression, IO_REQUEST_BATCH_BYTES,
    LoadedPageShard, MessageType, PageEncoding, PageInfo, RequestBatch, ScheduledScanLine,
    SchedulerContext, StructuralDecodeArrayTask, StructuralFieldDecoder, StructuralFieldScheduler,
    StructuralPageDecoder, StructuralSchedulingJob, UnloadedPageShard,
};
use crate::encodings::logical::primitive::blob::{BlobDescriptionPageScheduler, BlobPageScheduler};
use crate::format::metadata::PageLayout;
use crate::format::{EncodingUtils, metadata};
use crate::repdef::{CompositeRepDefUnraveler, DefinitionInterpretation};

use super::{
    CachedPage, CachedPageData, ComplexAllNullScheduler, FullZipScheduler, LevelCodec,
    MiniBlockScheduler, NoCachedPageData, PageDataCacheKey, PageInitialization,
    SimpleAllNullScheduler, StructuralPageScheduler,
};
use super::{constant, planar, sparse};

#[derive(Debug)]
struct StructuralPrimitiveFieldSchedulingJob<'a> {
    scheduler: &'a StructuralPrimitiveFieldScheduler,
    mappings: std::vec::IntoIter<PageRangeMapping>,
}

impl<'a> StructuralPrimitiveFieldSchedulingJob<'a> {
    pub fn try_new(
        scheduler: &'a StructuralPrimitiveFieldScheduler,
        ranges: Vec<Range<u64>>,
    ) -> Result<Self> {
        let mappings = map_ranges_to_pages(&scheduler.page_row_ends, &ranges)?.into_iter();
        Ok(Self {
            scheduler,
            mappings,
        })
    }
}

impl StructuralSchedulingJob for StructuralPrimitiveFieldSchedulingJob<'_> {
    fn schedule_next(&mut self, context: &mut SchedulerContext) -> Result<Vec<ScheduledScanLine>> {
        let Some(first_mapping) = self.mappings.next() else {
            return Ok(Vec::new());
        };
        // Pages are scheduled in row order until their reads add up to a
        // batch, then submitted together so the I/O scheduler can coalesce
        // neighbouring pages.
        let batch = Arc::new(RequestBatch::new(context.io().clone()));
        let batch_io: Arc<dyn EncodingsIo> = batch.clone();
        let cur_path = context.current_path();
        let mut scan_lines = Vec::new();
        let mut mapping = Some(first_mapping);
        while let Some(cur_mapping) = mapping.take() {
            let cur_page = self.scheduler.built_page(cur_mapping.page_idx)?;

            trace!(
                "Scheduling {} rows across {} ranges from page with {} rows (column_index={}, page_index={})",
                cur_mapping
                    .ranges_in_page
                    .iter()
                    .map(|r| r.end - r.start)
                    .sum::<u64>(),
                cur_mapping.ranges_in_page.len(),
                cur_page.row_range.end - cur_page.row_range.start,
                self.scheduler.column_index,
                cur_page.page_index,
            );

            let page_decoders = cur_page
                .scheduler
                .schedule_ranges(&cur_mapping.ranges_in_page, &batch_io)?;

            scan_lines.extend(page_decoders.into_iter().map(|page_load_task| {
                let cur_path = cur_path.clone();
                let page_decoder = page_load_task.decoder_fut;
                let unloaded_page = async move {
                    let page_decoder = page_decoder.await?;
                    Ok(LoadedPageShard {
                        decoder: page_decoder,
                        path: cur_path,
                    })
                }
                .boxed();
                ScheduledScanLine {
                    decoders: vec![MessageType::UnloadedPage(UnloadedPageShard(unloaded_page))],
                    rows_scheduled: page_load_task.num_rows,
                }
            }));

            if batch.pending_bytes() < IO_REQUEST_BATCH_BYTES {
                mapping = self.mappings.next();
            }
        }
        batch.flush();
        Ok(scan_lines)
    }
}

#[derive(Debug)]
struct PageInfoAndScheduler {
    page_index: usize,
    row_range: Range<u64>,
    scheduler: Box<dyn StructuralPageScheduler>,
}

/// Everything needed to build a page scheduler from its `PageInfo` on demand.
///
/// Building a page scheduler clones buffer tables and instantiates
/// decompressors, so a column with hundreds of thousands of pages must not pay
/// for every page on every read. The field scheduler keeps this source and
/// builds only the pages `initialize` is asked to cover.
#[derive(Debug)]
struct PageSchedulerSource {
    column_info: Arc<ColumnInfo>,
    decompressors: Arc<dyn DecompressionStrategy>,
    cache_repetition_index: bool,
    target_field: Field,
}

/// A scheduler for a leaf node
///
/// Here we look at the layout of the various pages and delegate scheduling to a scheduler
/// appropriate for the layout of the page.
#[derive(Debug)]
pub struct StructuralPrimitiveFieldScheduler {
    /// Exclusive end row of every page in the column, in page order, so page
    /// `i` covers `page_row_ends[i - 1]..page_row_ends[i]` (page 0 starts at 0).
    page_row_ends: Vec<u64>,
    /// Page schedulers built so far, ordered by `page_index`. After
    /// `initialize` this holds exactly the pages the requested ranges touch,
    /// or every page when no ranges were given.
    page_schedulers: Vec<PageInfoAndScheduler>,
    /// Metadata to build pages that are not in `page_schedulers` yet. `None`
    /// once every page has been built.
    page_source: Option<PageSchedulerSource>,
    column_index: u32,
    target_data_type_tag: Arc<str>,
}

impl StructuralPrimitiveFieldScheduler {
    /// Creates the scheduler and builds every page scheduler up front.
    ///
    /// This pays for all pages of the column on every read. Prefer
    /// [`Self::try_new_lazy`], which builds only the pages a read touches.
    #[deprecated(
        since = "13.0.0",
        note = "builds every page scheduler eagerly; use try_new_lazy"
    )]
    pub fn try_new(
        column_info: &ColumnInfo,
        decompressors: &dyn DecompressionStrategy,
        cache_repetition_index: bool,
        target_field: &Field,
    ) -> Result<Self> {
        let page_row_ends = Self::page_row_ends(column_info)?;
        let mut page_schedulers = Vec::with_capacity(page_row_ends.len());
        let mut row_start = 0_u64;
        for (page_index, (page_info, &row_end)) in column_info
            .page_infos
            .iter()
            .zip(&page_row_ends)
            .enumerate()
        {
            page_schedulers.push(Self::page_info_to_scheduler(
                page_info,
                page_index,
                row_start..row_end,
                decompressors,
                cache_repetition_index,
                target_field,
            )?);
            row_start = row_end;
        }
        Ok(Self {
            page_row_ends,
            page_schedulers,
            page_source: None,
            column_index: column_info.index,
            target_data_type_tag: format!("{:?}", target_field.data_type()).into(),
        })
    }

    /// Creates the scheduler without building any page schedulers.
    ///
    /// Only the per-page row boundaries are computed here. Page schedulers
    /// are built in [`StructuralFieldScheduler::initialize`] for the pages the
    /// requested ranges touch (or for every page when no ranges are given), so
    /// a small read from a column with many pages stays cheap.
    pub fn try_new_lazy(
        column_info: Arc<ColumnInfo>,
        decompressors: Arc<dyn DecompressionStrategy>,
        cache_repetition_index: bool,
        target_field: &Field,
    ) -> Result<Self> {
        Ok(Self {
            page_row_ends: Self::page_row_ends(&column_info)?,
            page_schedulers: Vec::new(),
            column_index: column_info.index,
            target_data_type_tag: format!("{:?}", target_field.data_type()).into(),
            page_source: Some(PageSchedulerSource {
                column_info,
                decompressors,
                cache_repetition_index,
                target_field: target_field.clone(),
            }),
        })
    }

    /// Exclusive end row of every page of `column_info`, in page order.
    fn page_row_ends(column_info: &ColumnInfo) -> Result<Vec<u64>> {
        let mut page_row_ends = Vec::with_capacity(column_info.page_infos.len());
        let mut row_start = 0_u64;
        for (page_index, page_info) in column_info.page_infos.iter().enumerate() {
            let row_end = row_start.checked_add(page_info.num_rows).ok_or_else(|| {
                Error::invalid_input_source(
                    format!(
                        "Page row range overflowed for column {} at page {page_index}: start={row_start}, rows={}",
                        column_info.index, page_info.num_rows
                    )
                    .into(),
                )
            })?;
            page_row_ends.push(row_end);
            row_start = row_end;
        }
        Ok(page_row_ends)
    }

    /// Wraps already-built page schedulers; nothing is built lazily.
    #[cfg(test)]
    fn from_page_schedulers(
        page_schedulers: Vec<PageInfoAndScheduler>,
        column_index: u32,
        target_data_type_tag: Arc<str>,
    ) -> Self {
        Self {
            page_row_ends: page_schedulers
                .iter()
                .map(|page| page.row_range.end)
                .collect(),
            page_schedulers,
            page_source: None,
            column_index,
            target_data_type_tag,
        }
    }

    fn page_row_range(&self, page_index: usize) -> Range<u64> {
        let start = match page_index.checked_sub(1) {
            Some(previous) => self.page_row_ends[previous],
            None => 0,
        };
        start..self.page_row_ends[page_index]
    }

    /// Position of `page_index` in `page_schedulers` if it has been built.
    fn built_page_position(&self, page_index: usize) -> Option<usize> {
        self.page_schedulers
            .binary_search_by_key(&page_index, |page| page.page_index)
            .ok()
    }

    fn built_page(&self, page_index: usize) -> Result<&PageInfoAndScheduler> {
        let position = self.built_page_position(page_index).ok_or_else(|| {
            Error::internal(format!(
                "Page {page_index} of column {} was scheduled but not initialized; the scheduled ranges must be covered by the ranges passed to initialize",
                self.column_index
            ))
        })?;
        Ok(&self.page_schedulers[position])
    }

    /// Builds the schedulers for `page_indices` (ascending) that do not exist
    /// yet and returns the positions of all of them in `page_schedulers`.
    fn build_pages(&mut self, page_indices: &[usize]) -> Result<Vec<usize>> {
        if let Some(source) = &self.page_source {
            let mut missing = Vec::new();
            for &page_index in page_indices {
                if self.built_page_position(page_index).is_some() {
                    continue;
                }
                let page_info = source
                    .column_info
                    .page_infos
                    .get(page_index)
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "Page {page_index} is out of bounds for column {} with {} pages",
                            self.column_index,
                            source.column_info.page_infos.len()
                        ))
                    })?;
                missing.push(Self::page_info_to_scheduler(
                    page_info,
                    page_index,
                    self.page_row_range(page_index),
                    source.decompressors.as_ref(),
                    source.cache_repetition_index,
                    &source.target_field,
                )?);
            }
            if !missing.is_empty() {
                self.page_schedulers = std::mem::take(&mut self.page_schedulers)
                    .into_iter()
                    .merge_by(missing, |built, new| built.page_index < new.page_index)
                    .collect();
            }
            if self.page_schedulers.len() == self.page_row_ends.len() {
                self.page_source = None;
            }
        }
        page_indices
            .iter()
            .map(|&page_index| {
                self.built_page_position(page_index).ok_or_else(|| {
                    Error::internal(format!(
                        "Page {page_index} of column {} is missing after construction",
                        self.column_index
                    ))
                })
            })
            .collect()
    }

    fn page_layout_to_scheduler(
        page_info: &PageInfo,
        page_layout: &PageLayout,
        decompressors: &dyn DecompressionStrategy,
        cache_repetition_index: bool,
        target_field: &Field,
    ) -> Result<Box<dyn StructuralPageScheduler>> {
        use metadata::page_layout::Layout;
        Ok(match page_layout.layout.as_ref().expect_ok()? {
            Layout::MiniBlockLayout(mini_block) => Box::new(MiniBlockScheduler::try_new(
                &page_info.buffer_offsets_and_sizes,
                page_info.priority,
                mini_block.num_items,
                mini_block,
                decompressors,
            )?),
            Layout::FullZipLayout(full_zip) => {
                let mut scheduler = FullZipScheduler::try_new(
                    &page_info.buffer_offsets_and_sizes,
                    page_info.priority,
                    page_info.num_rows,
                    full_zip,
                    decompressors,
                )?;
                scheduler.enable_cache = cache_repetition_index;
                Box::new(scheduler)
            }
            Layout::ConstantLayout(constant_layout) => {
                let def_meaning = constant_layout
                    .layers
                    .iter()
                    .map(|l| EncodingUtils::repdef_layer_to_def_interp(*l))
                    .collect::<Vec<_>>();
                let has_scalar_value = constant_layout.inline_value.is_some()
                    || page_info.buffer_offsets_and_sizes.len() == 1
                    || page_info.buffer_offsets_and_sizes.len() == 3;
                if has_scalar_value {
                    Box::new(constant::ConstantPageScheduler::try_new(
                        page_info.buffer_offsets_and_sizes.clone(),
                        constant_layout.inline_value.clone(),
                        target_field.data_type(),
                        def_meaning.into(),
                    )?) as Box<dyn StructuralPageScheduler>
                } else if def_meaning.len() == 1
                    && def_meaning[0] == DefinitionInterpretation::NullableItem
                {
                    Box::new(SimpleAllNullScheduler::default()) as Box<dyn StructuralPageScheduler>
                } else {
                    // RLE levels select a validated cache representation; other
                    // block compressions keep flowing through the eager decompressor.
                    let rep_codec = LevelCodec::try_new(
                        constant_layout.rep_compression.as_ref(),
                        decompressors,
                    )?;
                    let def_codec = LevelCodec::try_new(
                        constant_layout.def_compression.as_ref(),
                        decompressors,
                    )?;

                    Box::new(ComplexAllNullScheduler::new(
                        page_info.buffer_offsets_and_sizes.clone(),
                        def_meaning.into(),
                        rep_codec,
                        def_codec,
                        constant_layout.num_rep_values,
                        constant_layout.num_def_values,
                    )) as Box<dyn StructuralPageScheduler>
                }
            }
            Layout::BlobLayout(blob) => {
                let inner_scheduler = Self::page_layout_to_scheduler(
                    page_info,
                    blob.inner_layout.as_ref().expect_ok()?.as_ref(),
                    decompressors,
                    cache_repetition_index,
                    target_field,
                )?;
                let def_meaning = blob
                    .layers
                    .iter()
                    .map(|l| EncodingUtils::repdef_layer_to_def_interp(*l))
                    .collect::<Vec<_>>();
                if matches!(target_field.data_type(), DataType::Struct(_)) {
                    // User wants to decode blob into struct
                    Box::new(BlobDescriptionPageScheduler::new(
                        inner_scheduler,
                        def_meaning.into(),
                    ))
                } else {
                    // User wants to decode blob into binary data
                    Box::new(BlobPageScheduler::new(
                        inner_scheduler,
                        page_info.priority,
                        page_info.num_rows,
                        def_meaning.into(),
                    ))
                }
            }
        })
    }

    fn page_info_to_scheduler(
        page_info: &PageInfo,
        page_index: usize,
        row_range: Range<u64>,
        decompressors: &dyn DecompressionStrategy,
        cache_repetition_index: bool,
        target_field: &Field,
    ) -> Result<PageInfoAndScheduler> {
        if let PageEncoding::Planar(layout) = &page_info.encoding {
            let scheduler = if layout.sparse().is_some() {
                Box::new(sparse::planar::PlanarSparseScheduler::try_new(
                    page_info,
                    layout,
                    target_field,
                    decompressors,
                )?) as Box<dyn StructuralPageScheduler>
            } else {
                planar::validate_page(page_info, layout, target_field, decompressors)?;
                Self::page_layout_to_scheduler(
                    page_info,
                    &layout.dense_execution_layout()?,
                    decompressors,
                    cache_repetition_index,
                    target_field,
                )?
            };
            return Ok(PageInfoAndScheduler {
                page_index,
                row_range,
                scheduler,
            });
        }
        let page_layout = page_info.encoding.as_structural();
        let scheduler = Self::page_layout_to_scheduler(
            page_info,
            page_layout,
            decompressors,
            cache_repetition_index,
            target_field,
        )?;
        Ok(PageInfoAndScheduler {
            page_index,
            row_range,
            scheduler,
        })
    }
}

/// One page's slice of a scheduling request: the page's index and the
/// page-local row sub-ranges (relative to the page's first row) of the request
/// that fall in it.
#[derive(Debug)]
struct PageRangeMapping {
    page_idx: usize,
    ranges_in_page: Vec<Range<u64>>,
}

/// Visits every intersection between ordered row ranges and ordered pages.
///
/// `page_row_ends` holds the exclusive end row of each page (see
/// [`StructuralPrimitiveFieldScheduler::page_row_ends`]). Each range
/// binary-searches its first overlapping page and then visits only the pages
/// it touches. For `R` ranges, `P` pages, and `T` intersections this takes
/// `O(R log P + T)` time without scanning sparse gaps.
fn for_each_page_range(
    page_row_ends: &[u64],
    ranges: &[Range<u64>],
    mut visit: impl FnMut(usize, Range<u64>),
) -> Result<()> {
    for (range_idx, range) in ranges.iter().enumerate() {
        if range.start > range.end {
            return Err(Error::invalid_input_source(
                format!(
                    "Requested row range {range_idx} is reversed: {}..{}",
                    range.start, range.end
                )
                .into(),
            ));
        }
        if range_idx > 0 {
            let previous = &ranges[range_idx - 1];
            if previous.start >= range.start || previous.end > range.start {
                return Err(Error::invalid_input_source(
                    format!(
                        "Requested row ranges overlap or are not strictly ordered at index {range_idx}: {previous:?}, then {range:?}"
                    )
                    .into(),
                ));
            }
        }
    }

    for range in ranges {
        let mut page_idx = page_row_ends.partition_point(|page_end| *page_end <= range.start);
        let mut page_start = match page_idx.checked_sub(1) {
            Some(previous) => page_row_ends[previous],
            None => 0,
        };
        while let Some(&page_end) = page_row_ends.get(page_idx) {
            if page_start >= range.end {
                break;
            }
            let start_in_page = range.start.max(page_start) - page_start;
            let end_in_page = range.end.min(page_end) - page_start;
            if start_in_page < end_in_page {
                visit(page_idx, start_in_page..end_in_page);
            }
            page_idx += 1;
            page_start = page_end;
        }
    }
    Ok(())
}

/// Maps global row `ranges` onto a column's pages, yielding the page-local
/// sub-ranges for every page any range touches.
///
/// `ranges` must be ascending and non-overlapping (the precondition
/// `schedule_ranges` relies on).
fn map_ranges_to_pages(
    page_row_ends: &[u64],
    ranges: &[Range<u64>],
) -> Result<Vec<PageRangeMapping>> {
    let mut result: Vec<PageRangeMapping> =
        Vec::with_capacity(ranges.len().min(page_row_ends.len()));
    for_each_page_range(page_row_ends, ranges, |page_idx, range_in_page| {
        if let Some(mapping) = result.last_mut()
            && mapping.page_idx == page_idx
        {
            mapping.ranges_in_page.push(range_in_page);
        } else {
            result.push(PageRangeMapping {
                page_idx,
                ranges_in_page: vec![range_in_page],
            });
        }
    })?;
    Ok(result)
}

fn pages_overlapping_ranges(page_row_ends: &[u64], ranges: &[Range<u64>]) -> Result<Vec<usize>> {
    let mut result = Vec::with_capacity(ranges.len().min(page_row_ends.len()));
    for_each_page_range(page_row_ends, ranges, |page_idx, _| {
        if result.last().copied() != Some(page_idx) {
            result.push(page_idx);
        }
    })?;
    Ok(result)
}

#[derive(Debug)]
struct PageCacheMiss {
    /// Position of the page in `StructuralPrimitiveFieldScheduler::page_schedulers`,
    /// which holds only the built pages and so may differ from `page_index`.
    page_idx: usize,
    cache_key: PageDataCacheKey,
    initialization: PageInitialization,
}

async fn read_page_initialization_buffers(
    io: &Arc<dyn EncodingsIo>,
    all_ranges: Vec<Range<u64>>,
) -> Result<Vec<Bytes>> {
    // `submit_request` coalesces (and later un-coalesces) in one forward
    // pass, so submit ranges in file order. Page layouts are not required
    // to declare their initialization ranges in that order, so retain each
    // range's original position and restore it before initializing pages.
    let mut sorted_ranges = all_ranges.into_iter().enumerate().collect::<Vec<_>>();
    sorted_ranges.sort_unstable_by_key(|(_, range)| (range.start, range.end));
    let sorted_requests = sorted_ranges
        .iter()
        .map(|(_, range)| range.clone())
        .collect::<Vec<_>>();
    let expected_buffer_count = sorted_requests.len();
    let sorted_buffers = if sorted_requests.is_empty() {
        Vec::new()
    } else {
        io.submit_request(sorted_requests, 0).await?
    };
    if sorted_buffers.len() != expected_buffer_count {
        return Err(Error::invalid_input_source(
            format!(
                "Page initialization I/O returned {} buffers for {} ranges",
                sorted_buffers.len(),
                expected_buffer_count
            )
            .into(),
        ));
    }
    let mut buffers = vec![None; expected_buffer_count];
    for ((original_idx, _), buffer) in sorted_ranges.into_iter().zip(sorted_buffers) {
        buffers[original_idx] = Some(buffer);
    }
    buffers
        .into_iter()
        .map(|buffer| {
            buffer.ok_or_else(|| Error::internal("Page initialization buffer was not assigned"))
        })
        .collect()
}

async fn cache_initialized_page(
    cache: Arc<LanceCache>,
    cache_key: PageDataCacheKey,
    data: Arc<dyn CachedPageData>,
) {
    if data
        .clone()
        .as_arc_any()
        .downcast::<NoCachedPageData>()
        .is_err()
    {
        cache
            .insert_with_key(&cache_key, Arc::new(CachedPage { data }))
            .await;
    }
}

impl StructuralPrimitiveFieldScheduler {
    /// Checks the built pages at `positions` (ascending indices into
    /// `page_schedulers`) against the cache and returns those still needing
    /// initialization I/O.
    async fn find_cache_misses(
        &mut self,
        positions: impl ExactSizeIterator<Item = usize>,
        cache: &Arc<LanceCache>,
    ) -> Result<Vec<PageCacheMiss>> {
        // Serve cache hits in place; the rest are misses we must read.
        let mut misses = Vec::with_capacity(positions.len());
        for page_idx in positions {
            self.page_schedulers[page_idx]
                .scheduler
                .set_cache(cache.clone());
            let page = &self.page_schedulers[page_idx];
            if !page.scheduler.needs_initialization() {
                continue;
            }
            let page_index = u32::try_from(page.page_index).map_err(|_| {
                Error::invalid_input_source(
                    format!("Page index does not fit u32: {}", page.page_index).into(),
                )
            })?;
            let cache_key = PageDataCacheKey {
                column_index: self.column_index,
                page_index,
                view: page.scheduler.cache_view(),
                target_data_type_tag: self.target_data_type_tag.clone(),
            };
            if let Some(cached_page) = cache.get_with_key(&cache_key).await {
                match self.page_schedulers[page_idx]
                    .scheduler
                    .try_load(&cached_page.data)
                {
                    Ok(()) => continue,
                    Err(error) => {
                        debug!("Replacing invalid cached page data for {cache_key:?}: {error}");
                    }
                }
            }
            let initialization = self.page_schedulers[page_idx].scheduler.init_layout()?;
            misses.push(PageCacheMiss {
                page_idx,
                cache_key,
                initialization,
            });
        }
        Ok(misses)
    }

    fn initialization_ranges(misses: &[PageCacheMiss]) -> Vec<Range<u64>> {
        let num_ranges = misses
            .iter()
            .map(|miss| miss.initialization.num_required_ranges())
            .sum();
        let mut all_ranges = Vec::with_capacity(num_ranges);
        for miss in misses {
            all_ranges.extend(miss.initialization.required_ranges().cloned());
        }
        all_ranges
    }

    async fn initialize_cache_misses(
        &mut self,
        misses: Vec<PageCacheMiss>,
        buffers: Vec<Bytes>,
        cache: Arc<LanceCache>,
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<()> {
        let mut buffers = buffers.into_iter();

        // Initialize each missing page from its slice of the buffers, in file
        // order. `init_from_buffers` can perform further I/O (e.g. loading blob
        // descriptions), so run pages concurrently; peel disjoint `&mut` page
        // borrows off the slice to satisfy the borrow checker without a lookup
        // table.
        let mut tail = self.page_schedulers.as_mut_slice();
        let mut next_page = 0;
        let mut page_inits = FuturesOrdered::new();
        for miss in misses {
            let PageCacheMiss {
                page_idx,
                cache_key,
                initialization,
            } = miss;
            let relative_idx = page_idx
                .checked_sub(next_page)
                .ok_or_else(|| Error::internal("Page initialization indices are not ordered"))?;
            if relative_idx >= tail.len() {
                return Err(Error::internal(format!(
                    "Page initialization index {page_idx} is out of bounds"
                )));
            }
            let (_, rest) = tail.split_at_mut(relative_idx);
            let (page, rest) = rest.split_first_mut().ok_or_else(|| {
                Error::internal(format!(
                    "Page initialization index {page_idx} is out of bounds"
                ))
            })?;
            tail = rest;
            next_page = page_idx + 1;

            let count = initialization.num_required_ranges();
            let page_buffers = buffers.by_ref().take(count).collect::<Vec<_>>();
            if page_buffers.len() != count {
                return Err(Error::invalid_input_source(
                    format!(
                        "Page {page_idx} received {} initialization buffers, expected {count}",
                        page_buffers.len()
                    )
                    .into(),
                ));
            }
            let page_buffers = initialization.with_buffers(page_buffers)?;
            let cache = cache.clone();
            let scheduler = &mut page.scheduler;
            page_inits.push_back(async move {
                let cached = scheduler.init_from_buffers(page_buffers, io).await?;
                scheduler.try_load(&cached)?;
                cache_initialized_page(cache, cache_key, cached).await;
                Ok::<(), Error>(())
            });
        }
        if buffers.next().is_some() {
            return Err(Error::invalid_input_source(
                "Page initialization returned unassigned buffers".into(),
            ));
        }

        page_inits.try_collect::<Vec<()>>().await?;
        Ok(())
    }

    async fn initialize_pages(
        &mut self,
        requested_ranges: Option<&[Range<u64>]>,
        context: &SchedulerContext,
    ) -> Result<()> {
        let cache = context.cache().clone();
        let page_indices = match requested_ranges {
            None => (0..self.page_row_ends.len()).collect::<Vec<_>>(),
            Some(ranges) => pages_overlapping_ranges(&self.page_row_ends, ranges)?,
        };
        let positions = self.build_pages(&page_indices)?;
        let misses = self
            .find_cache_misses(positions.into_iter(), &cache)
            .await?;

        if misses.is_empty() {
            return Ok(());
        }

        // Concatenate the misses' metadata ranges into one request so adjacent
        // ranges coalesce into shared GETs instead of one request per page.
        let all_ranges = Self::initialization_ranges(&misses);
        let buffers = read_page_initialization_buffers(context.io(), all_ranges).await?;
        self.initialize_cache_misses(misses, buffers, cache, context.io())
            .await
    }
}

impl StructuralFieldScheduler for StructuralPrimitiveFieldScheduler {
    fn initialize<'a>(
        &'a mut self,
        requested_ranges: Option<&'a [Range<u64>]>,
        _filter: &'a FilterExpression,
        context: &'a SchedulerContext,
    ) -> BoxFuture<'a, Result<()>> {
        self.initialize_pages(requested_ranges, context).boxed()
    }

    fn schedule_ranges<'a>(
        &'a self,
        ranges: &[Range<u64>],
        _filter: &FilterExpression,
    ) -> Result<Box<dyn StructuralSchedulingJob + 'a>> {
        let ranges = ranges.to_vec();
        Ok(Box::new(StructuralPrimitiveFieldSchedulingJob::try_new(
            self, ranges,
        )?))
    }
}

/// Takes the output from several pages decoders and
/// concatenates them.
#[derive(Debug)]
pub struct StructuralCompositeDecodeArrayTask {
    tasks: Vec<Box<dyn DecodePageTask>>,
    should_validate: bool,
    data_type: DataType,
}

impl StructuralCompositeDecodeArrayTask {
    fn restore_validity(
        array: Arc<dyn Array>,
        unraveler: &mut CompositeRepDefUnraveler,
    ) -> Result<Arc<dyn Array>> {
        let validity = unraveler.unravel_validity(array.len())?;
        let Some(validity) = validity else {
            return Ok(array);
        };
        if array.data_type() == &DataType::Null {
            // We unravel from a null array but we don't add the null buffer because arrow-rs doesn't like it
            return Ok(array);
        }
        if validity.len() != array.len() {
            return Err(Error::invalid_input_source(
                format!(
                    "Structural validity has {} entries for an array with {} values",
                    validity.len(),
                    array.len()
                )
                .into(),
            ));
        }
        // SAFETY: The array buffers have already been validated and the null buffer length
        // matches the array. We are only attaching the null buffer here.
        let array = make_array(unsafe {
            array
                .to_data()
                .into_builder()
                .nulls(Some(validity))
                .build_unchecked()
        });
        if let Some(struct_array) = array.as_struct_opt() {
            // Packed structs (including blob descriptors) restore validity at the
            // parent level. Field access reads the child arrays directly, so those
            // arrays must also mask values beneath a null parent.
            Ok(Arc::new(struct_array.pushdown_nulls()?))
        } else {
            Ok(array)
        }
    }
}

impl StructuralDecodeArrayTask for StructuralCompositeDecodeArrayTask {
    fn decode(self: Box<Self>) -> Result<DecodedArray> {
        let mut arrays = Vec::with_capacity(self.tasks.len());
        let mut unravelers = Vec::with_capacity(self.tasks.len());
        let mut data_size = 0u64;
        for task in self.tasks {
            let decoded = task.decode()?;
            data_size += decoded.data.data_size();
            unravelers.push(decoded.repdef);

            let array = make_array(
                decoded
                    .data
                    .into_arrow(self.data_type.clone(), self.should_validate)?,
            );

            arrays.push(array);
        }
        let array_refs = arrays.iter().map(|arr| arr.as_ref()).collect::<Vec<_>>();
        let array = concat_decoded_arrays(&array_refs)?;
        let mut repdef = CompositeRepDefUnraveler::new(unravelers);

        let array = Self::restore_validity(array, &mut repdef)?;

        Ok(DecodedArray {
            array,
            repdef,
            data_size,
        })
    }
}

/// Returns the total bytes consumed by validity bitmaps of nested child fields
/// for `rows` logical rows of `data_type`.
///
/// Only `FixedSizeList` types have nested children that contribute additional
/// bitmaps; all other fixed-width leaf types return 0.
fn fsl_child_bitmap_bytes(data_type: &arrow_schema::DataType, rows: u64) -> u64 {
    match data_type {
        arrow_schema::DataType::FixedSizeList(child_field, dimension) => {
            let child_rows = rows * *dimension as u64;
            let own = if child_field.is_nullable() {
                child_rows.div_ceil(8)
            } else {
                0
            };
            own + fsl_child_bitmap_bytes(child_field.data_type(), child_rows)
        }
        _ => 0,
    }
}

#[derive(Debug)]
pub struct StructuralPrimitiveFieldDecoder {
    field: Arc<ArrowField>,
    page_decoders: VecDeque<Box<dyn StructuralPageDecoder>>,
    should_validate: bool,
    rows_drained_in_current: u64,
}

impl StructuralPrimitiveFieldDecoder {
    pub fn new(field: &Arc<ArrowField>, should_validate: bool) -> Self {
        Self {
            field: field.clone(),
            page_decoders: VecDeque::new(),
            should_validate,
            rows_drained_in_current: 0,
        }
    }

    fn decoded_bytes_for_rows(&self, rows: u64) -> lance_core::Result<u64> {
        let mut remaining = rows;
        let mut total = 0u64;
        for (page_num, page) in self.page_decoders.iter().enumerate() {
            if remaining == 0 {
                break;
            }
            let available = if page_num == 0 {
                page.num_rows().saturating_sub(self.rows_drained_in_current)
            } else {
                page.num_rows()
            };
            let take = available.min(remaining);
            total += page.decoded_bytes(take)?;
            remaining -= take;
        }
        if remaining > 0 {
            return Err(lance_core::Error::not_supported(format!(
                "plan_decoded_bytes: queued pages cover only {} of {} requested rows",
                rows - remaining,
                rows,
            )));
        }
        Ok(total)
    }
}

impl StructuralFieldDecoder for StructuralPrimitiveFieldDecoder {
    fn max_dictionary_rows(&self) -> Option<u64> {
        self.page_decoders
            .iter()
            .filter_map(|page| page.dictionary_len())
            .filter_map(|len| dictionary_batch_limit(self.field.data_type(), len))
            .min()
    }

    fn accept_page(&mut self, child: LoadedPageShard) -> Result<()> {
        assert!(child.path.is_empty());
        self.page_decoders.push_back(child.decoder);
        Ok(())
    }

    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn StructuralDecodeArrayTask>> {
        let mut remaining = num_rows;
        let mut tasks = Vec::new();
        while remaining > 0 {
            let queued_pages = self.page_decoders.len();
            let Some(cur_page) = self.page_decoders.front_mut() else {
                return Err(Error::internal(format!(
                    "Primitive decoder missing page decoder while draining field '{}' (data_type={:?}, requested_rows={}, remaining_rows={}, rows_drained_in_current={}, queued_pages={})",
                    self.field.name(),
                    self.field.data_type(),
                    num_rows,
                    remaining,
                    self.rows_drained_in_current,
                    queued_pages
                )));
            };
            let num_in_page = cur_page.num_rows() - self.rows_drained_in_current;
            let to_take = num_in_page.min(remaining);

            let task = cur_page.drain(to_take)?;
            tasks.push(task);

            if to_take == num_in_page {
                self.page_decoders.pop_front();
                self.rows_drained_in_current = 0;
            } else {
                self.rows_drained_in_current += to_take;
            }

            remaining -= to_take;
        }
        Ok(Box::new(StructuralCompositeDecodeArrayTask {
            tasks,
            should_validate: self.should_validate,
            data_type: self.field.data_type().clone(),
        }))
    }

    fn data_type(&self) -> &DataType {
        self.field.data_type()
    }

    fn plan_decoded_bytes(&self, rows_remaining: u64) -> lance_core::Result<[u64; 8]> {
        use crate::decoder::CANDIDATE_BATCH_SIZES;
        let data_type = self.field.data_type();
        let is_nullable = self.field.is_nullable();

        // Fixed-width: exact from type metadata, no page inspection needed.
        if let Some(byte_width) = data_type.byte_width_opt() {
            let mut out = [0u64; 8];
            for (i, &c) in CANDIDATE_BATCH_SIZES.iter().enumerate() {
                let rows = (c as u64).min(rows_remaining);
                out[i] = rows * byte_width as u64;
                if is_nullable {
                    out[i] += rows.div_ceil(8);
                }
                out[i] += fsl_child_bitmap_bytes(data_type, rows);
            }
            return Ok(out);
        }

        // Boolean: bit-packed, 1 bit per value.
        if matches!(data_type, DataType::Boolean) {
            let mut out = [0u64; 8];
            for (i, &c) in CANDIDATE_BATCH_SIZES.iter().enumerate() {
                let rows = (c as u64).min(rows_remaining);
                out[i] = rows.div_ceil(8);
                if is_nullable {
                    out[i] += rows.div_ceil(8);
                }
            }
            return Ok(out);
        }

        // Null type: no data bytes.
        if matches!(data_type, DataType::Null) {
            return Ok([0u64; 8]);
        }

        // Variable-width: walk page decoders for exact byte counts.
        let mut out = [0u64; 8];
        for (i, &c) in CANDIDATE_BATCH_SIZES.iter().enumerate() {
            let rows = (c as u64).min(rows_remaining);
            out[i] = self.decoded_bytes_for_rows(rows)?;
            if is_nullable {
                out[i] += rows.div_ceil(8);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use std::ops::Range;
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, Int32Array, StructArray, cast::AsArray};
    use arrow_schema::{DataType, Field as ArrowField};
    use bytes::Bytes;

    use crate::EncodingsIo;
    use crate::decoder::{
        FilterExpression, RequestBatch, SchedulerContext, StructuralFieldDecoder,
        StructuralFieldScheduler,
    };
    use crate::encodings::logical::primitive::{
        CachedPage, CachedPageData, PageCacheView, PageDataCacheKey, SimpleAllNullScheduler,
        StructuralPageScheduler,
    };
    use crate::format::metadata;
    use crate::repdef::{CompositeRepDefUnraveler, DefinitionInterpretation, RepDefUnraveler};
    use crate::testing::SimulatedScheduler;

    use super::{
        PageInfoAndScheduler, StructuralCompositeDecodeArrayTask, StructuralPrimitiveFieldDecoder,
        StructuralPrimitiveFieldScheduler, map_ranges_to_pages, pages_overlapping_ranges,
    };

    #[derive(Debug)]
    struct CacheMarker(u8);

    #[test]
    fn test_packed_struct_restores_child_validity() {
        let array = StructArray::new(
            vec![Arc::new(ArrowField::new("value", DataType::Int32, false))].into(),
            vec![Arc::new(Int32Array::from(vec![5, 0, 7])) as ArrayRef],
            None,
        );
        let mut repdef = CompositeRepDefUnraveler::new(vec![RepDefUnraveler::new(
            None,
            Some(vec![0, 1, 0]),
            vec![DefinitionInterpretation::NullableItem].into(),
            3,
        )]);
        let result =
            StructuralCompositeDecodeArrayTask::restore_validity(Arc::new(array), &mut repdef)
                .unwrap();
        let result = result.as_struct();
        assert_eq!(result.null_count(), 1);
        assert_eq!(result.column(0).nulls(), result.nulls());
        assert!(result.column(0).is_null(1));
        result.to_data().validate_full().unwrap();
    }

    impl lance_core::cache::DeepSizeOf for CacheMarker {
        fn deep_size_of_children(&self, _context: &mut lance_core::cache::Context) -> usize {
            0
        }
    }

    impl CachedPageData for CacheMarker {
        fn as_arc_any(self: Arc<Self>) -> Arc<dyn std::any::Any + Send + Sync + 'static> {
            self
        }
    }

    #[test]
    fn test_primitive_decoder_empty_page_queue_returns_error() {
        let field = Arc::new(ArrowField::new("vector", DataType::Float32, true));
        let mut decoder = StructuralPrimitiveFieldDecoder::new(&field, false);

        let err = decoder.drain(1).unwrap_err();
        assert!(
            matches!(&err, lance_core::Error::Internal { .. }),
            "expected internal error, got: {err:?}"
        );
        let message = err.to_string();
        for expected in [
            "Primitive decoder missing page decoder",
            "field 'vector'",
            "data_type=Float32",
            "requested_rows=1",
            "remaining_rows=1",
            "rows_drained_in_current=0",
            "queued_pages=0",
        ] {
            assert!(
                message.contains(expected),
                "expected error to contain {expected:?}, got: {message}"
            );
        }
    }

    #[test]
    fn test_plan_decoded_bytes_i32() {
        use crate::decoder::CANDIDATE_BATCH_SIZES;

        let field = Arc::new(ArrowField::new("x", DataType::Int32, false));
        let decoder = StructuralPrimitiveFieldDecoder::new(&field, false);

        let rows_remaining = 100u64;
        let bytes = decoder.plan_decoded_bytes(rows_remaining).unwrap();

        for (i, &candidate) in CANDIDATE_BATCH_SIZES.iter().enumerate() {
            let rows = (candidate as u64).min(rows_remaining);
            assert_eq!(bytes[i], rows * 4, "candidate batch size {candidate}");
        }
    }

    #[test]
    fn test_plan_decoded_bytes_i32_with_nulls() {
        use arrow_array::Int32Array;

        // Use N == CANDIDATE_BATCH_SIZES[5] == 1024.  At this size both the
        // values buffer (1024*4 = 4096 bytes) and the validity bitmap
        // (ceil(1024/8) = 128 bytes) are naturally aligned to Arrow's 64-byte
        // allocation boundary, so plan_decoded_bytes matches
        // get_buffer_memory_size exactly without needing alignment arithmetic.
        const N: i32 = 1024;
        let array = Int32Array::from_iter((0..N).map(|i| if i % 2 == 0 { Some(i) } else { None }));
        let expected = array.get_buffer_memory_size() as u64;

        let field = Arc::new(ArrowField::new("x", DataType::Int32, true));
        let decoder = StructuralPrimitiveFieldDecoder::new(&field, false);

        let bytes = decoder.plan_decoded_bytes(N as u64).unwrap();
        // bytes[5] corresponds to CANDIDATE_BATCH_SIZES[5] == 1024 == N
        assert_eq!(
            bytes[5], expected,
            "plan_decoded_bytes({N}) = {} but get_buffer_memory_size = {expected}",
            bytes[5],
        );
    }

    #[test]
    fn test_plan_decoded_bytes_nullable_fsl_with_nullable_items() {
        // FixedSizeList<nullable i32, DIM> where the FSL itself is also nullable.
        // Arrow allocates three buffers:
        //   1. FSL null bitmap:       ceil(rows / 8)
        //   2. Child i32 null bitmap: ceil(rows * DIM / 8)
        //   3. Child i32 values:      rows * DIM * 4
        //
        // N=1024, DIM=4 keeps all three naturally 64-byte aligned so the
        // comparison against get_buffer_memory_size() is exact.
        use arrow_array::{FixedSizeListArray, Int32Array};
        use arrow_buffer::NullBuffer;

        const N: usize = 1024;
        const DIM: i32 = 4;

        let child_values = Int32Array::from_iter(
            (0..(N as i32 * DIM)).map(|i| if i % 2 == 0 { Some(i) } else { None }),
        );
        let item_field = Arc::new(ArrowField::new("item", DataType::Int32, true));
        let fsl_nulls = NullBuffer::from((0..N).map(|i| i % 3 != 0).collect::<Vec<_>>());
        let fsl_array = FixedSizeListArray::new(
            item_field.clone(),
            DIM,
            Arc::new(child_values),
            Some(fsl_nulls),
        );
        let expected = fsl_array.get_buffer_memory_size() as u64;

        let fsl_type = DataType::FixedSizeList(item_field, DIM);
        let field = Arc::new(ArrowField::new("v", fsl_type, true));
        let decoder = StructuralPrimitiveFieldDecoder::new(&field, false);

        let bytes = decoder.plan_decoded_bytes(N as u64).unwrap();
        // bytes[5] corresponds to CANDIDATE_BATCH_SIZES[5] == 1024 == N
        assert_eq!(
            bytes[5], expected,
            "plan_decoded_bytes({N}) = {} but get_buffer_memory_size = {expected}",
            bytes[5],
        );
    }

    #[test]
    fn test_plan_decoded_bytes_fixed_size_list_of_f32() {
        use crate::decoder::CANDIDATE_BATCH_SIZES;

        const DIMENSION: i32 = 8;
        // Items are non-nullable, which is the typical case for vector embeddings.
        let item_field = Arc::new(ArrowField::new("item", DataType::Float32, false));
        let fsl_type = DataType::FixedSizeList(item_field, DIMENSION);
        let field = Arc::new(ArrowField::new("vector", fsl_type, true));
        let decoder = StructuralPrimitiveFieldDecoder::new(&field, false);

        let rows_remaining = 100u64;
        let bytes = decoder.plan_decoded_bytes(rows_remaining).unwrap();

        // Each FSL row is DIMENSION f32 values = DIMENSION * 4 bytes, plus
        // 1 bit of validity bitmap (the field is nullable).
        let bytes_per_row = DIMENSION as u64 * 4;
        for (i, &candidate) in CANDIDATE_BATCH_SIZES.iter().enumerate() {
            let rows = (candidate as u64).min(rows_remaining);
            assert_eq!(
                bytes[i],
                rows * bytes_per_row + rows.div_ceil(8),
                "candidate batch size {candidate}"
            );
        }
    }

    /// Prove that an N-page cache miss issues one `submit_request` carrying every
    /// page's metadata range, rather than one submission per page. The production
    /// I/O scheduler can then coalesce adjacent ranges within that batch.
    #[tokio::test]
    async fn test_initialize_coalesces_missed_page_metadata() {
        use std::ops::Range;

        use futures::FutureExt;
        use futures::future::BoxFuture;

        use super::{
            CachedPage, CachedPageData, Error, PageDataCacheKey, PageInfoAndScheduler,
            PageInitialization, StructuralPrimitiveFieldScheduler,
        };
        use crate::decoder::{FilterExpression, SchedulerContext, StructuralFieldScheduler};
        use crate::encodings::logical::primitive::{
            PageCacheView, PageInitializationBuffers, PageLoadTask,
        };

        // Give each fake page one fixed-size metadata buffer. The ranges are far
        // apart because this test observes batching, not the I/O layer's distance-
        // based range coalescing.
        #[derive(Debug)]
        struct FakePageScheduler {
            meta_range: Range<u64>,
        }
        impl StructuralPageScheduler for FakePageScheduler {
            fn init_layout(&self) -> crate::Result<PageInitialization> {
                Ok(PageInitialization::new([Some(self.meta_range.clone())]))
            }
            fn init_from_buffers<'a>(
                &'a mut self,
                buffers: PageInitializationBuffers,
                _io: &Arc<dyn EncodingsIo>,
            ) -> BoxFuture<'a, crate::Result<Arc<dyn CachedPageData>>> {
                let [buffer] = buffers.try_into_array("fake").unwrap();
                let buffer = buffer.unwrap();
                assert_eq!(
                    &buffer[..8],
                    &self.meta_range.start.to_le_bytes(),
                    "each page gets the buffer for its own range"
                );
                std::future::ready(Ok(Arc::new(CacheMarker(42)) as Arc<dyn CachedPageData>)).boxed()
            }
            fn try_load(&mut self, data: &Arc<dyn CachedPageData>) -> crate::Result<()> {
                data.clone()
                    .as_arc_any()
                    .downcast::<CacheMarker>()
                    .map(|_| ())
                    .map_err(|_| {
                        Error::invalid_input_source(
                            "Fake page cache data has an unexpected type".into(),
                        )
                    })
            }
            fn schedule_ranges(
                &self,
                _ranges: &[Range<u64>],
                _io: &Arc<dyn EncodingsIo>,
            ) -> crate::Result<Vec<PageLoadTask>> {
                unreachable!("initialize never schedules data reads");
            }
        }

        const NUM_PAGES: u64 = 64;
        const META_LEN: u64 = 64;
        const STRIDE: u64 = 1024 * 1024;
        let page_schedulers = (0..NUM_PAGES)
            .map(|i| PageInfoAndScheduler {
                page_index: i as usize,
                row_range: (i * 100)..((i + 1) * 100),
                scheduler: Box::new(FakePageScheduler {
                    meta_range: (i * STRIDE)..(i * STRIDE + META_LEN),
                }),
            })
            .collect();
        let mut scheduler = StructuralPrimitiveFieldScheduler::from_page_schedulers(
            page_schedulers,
            0,
            Arc::from("test"),
        );

        let io = Arc::new(RecordingScheduler::default());
        // A no-op cache always misses, so every page is a miss.
        let cache = Arc::new(lance_core::cache::LanceCache::no_cache());
        let context = SchedulerContext::new(io.clone(), cache.clone());

        // `None` requests every page; against an always-miss cache that's an
        // N-page miss.
        let filter = FilterExpression::no_filter();
        scheduler.initialize(None, &filter, &context).await.unwrap();

        {
            let requests = io.requests.lock().unwrap();
            assert_eq!(
                requests.len(),
                1,
                "an N-page miss must issue exactly one request, not one per page"
            );
            assert_eq!(
                requests[0].len() as u64,
                NUM_PAGES,
                "the single request must carry every missed page's metadata range"
            );
        }

        io.requests.lock().unwrap().clear();
        let cache_misses_before = cache.stats().await.misses;
        let requested_ranges = Arc::<[Range<u64>]>::from([150..151, 6_350..6_351]);
        scheduler
            .initialize(Some(&requested_ranges), &filter, &context)
            .await
            .unwrap();
        assert_eq!(
            cache.stats().await.misses - cache_misses_before,
            2,
            "range-scoped initialization must only look up touched pages"
        );

        {
            let requests = io.requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert_eq!(
                requests[0],
                vec![
                    STRIDE..(STRIDE + META_LEN),
                    63 * STRIDE..(63 * STRIDE + META_LEN),
                ],
                "range-scoped initialization must omit untouched pages"
            );
        }

        io.requests.lock().unwrap().clear();
        let cache = Arc::new(lance_core::cache::LanceCache::with_capacity(4 * 1024));
        let cache_key = PageDataCacheKey {
            column_index: 9,
            page_index: 0,
            view: PageCacheView::Native,
            target_data_type_tag: Arc::from("test"),
        };
        cache
            .insert_with_key(
                &cache_key,
                Arc::new(CachedPage {
                    data: Arc::new(super::NoCachedPageData),
                }),
            )
            .await;
        let mut recovering_scheduler = StructuralPrimitiveFieldScheduler::from_page_schedulers(
            vec![PageInfoAndScheduler {
                page_index: 0,
                row_range: 0..100,
                scheduler: Box::new(FakePageScheduler {
                    meta_range: 0..META_LEN,
                }),
            }],
            9,
            Arc::from("test"),
        );
        let recovering_context = SchedulerContext::new(io.clone(), cache.clone());
        recovering_scheduler
            .initialize(None, &filter, &recovering_context)
            .await
            .unwrap();
        assert_eq!(io.requests.lock().unwrap().as_slice(), &[vec![0..META_LEN]]);

        io.requests.lock().unwrap().clear();
        let mut warm_scheduler = StructuralPrimitiveFieldScheduler::from_page_schedulers(
            vec![PageInfoAndScheduler {
                page_index: 0,
                row_range: 0..100,
                scheduler: Box::new(FakePageScheduler {
                    meta_range: 0..META_LEN,
                }),
            }],
            9,
            Arc::from("test"),
        );
        warm_scheduler
            .initialize(None, &filter, &recovering_context)
            .await
            .unwrap();
        assert!(
            io.requests.lock().unwrap().is_empty(),
            "the rebuilt cache entry must be reusable"
        );

        let mut out_of_order_scheduler = StructuralPrimitiveFieldScheduler::from_page_schedulers(
            vec![
                PageInfoAndScheduler {
                    page_index: 0,
                    row_range: 0..100,
                    scheduler: Box::new(FakePageScheduler {
                        meta_range: STRIDE..(STRIDE + META_LEN),
                    }),
                },
                PageInfoAndScheduler {
                    page_index: 1,
                    row_range: 100..200,
                    scheduler: Box::new(FakePageScheduler {
                        meta_range: 0..META_LEN,
                    }),
                },
            ],
            0,
            Arc::from("test"),
        );
        out_of_order_scheduler
            .initialize(None, &filter, &context)
            .await
            .unwrap();
        assert_eq!(
            io.requests.lock().unwrap().as_slice(),
            &[vec![0..META_LEN, STRIDE..(STRIDE + META_LEN)]],
            "metadata reads must be sorted without changing page buffer assignment"
        );
    }

    #[tokio::test]
    async fn page_cache_key_distinguishes_page_view_and_target_type() {
        let cache = lance_core::cache::LanceCache::with_capacity(4 * 1024);
        let cases = [
            (
                PageDataCacheKey {
                    column_index: 7,
                    page_index: 0,
                    view: PageCacheView::Native,
                    target_data_type_tag: Arc::from("Boolean"),
                },
                10,
            ),
            (
                PageDataCacheKey {
                    column_index: 7,
                    page_index: 1,
                    view: PageCacheView::Native,
                    target_data_type_tag: Arc::from("Boolean"),
                },
                11,
            ),
            (
                PageDataCacheKey {
                    column_index: 7,
                    page_index: 0,
                    view: PageCacheView::Native,
                    target_data_type_tag: Arc::from("UInt8"),
                },
                12,
            ),
            (
                PageDataCacheKey {
                    column_index: 7,
                    page_index: 0,
                    view: PageCacheView::BlobDescriptor,
                    target_data_type_tag: Arc::from("Struct"),
                },
                20,
            ),
            (
                PageDataCacheKey {
                    column_index: 7,
                    page_index: 0,
                    view: PageCacheView::BlobPayload,
                    target_data_type_tag: Arc::from("LargeBinary"),
                },
                21,
            ),
        ];

        for (key, marker) in &cases {
            cache
                .insert_with_key(
                    key,
                    Arc::new(CachedPage {
                        data: Arc::new(CacheMarker(*marker)),
                    }),
                )
                .await;
        }

        assert_eq!(cache.size().await, cases.len());
        for (key, expected) in &cases {
            let cached = cache.get_with_key(key).await.unwrap();
            let marker = cached
                .data
                .clone()
                .as_arc_any()
                .downcast::<CacheMarker>()
                .unwrap();
            assert_eq!(marker.0, *expected);
        }
    }

    #[tokio::test]
    async fn no_op_pages_do_not_create_cache_entries() {
        let mut scheduler = StructuralPrimitiveFieldScheduler::from_page_schedulers(
            vec![PageInfoAndScheduler {
                page_index: 0,
                row_range: 0..10,
                scheduler: Box::new(SimpleAllNullScheduler::default()),
            }],
            3,
            Arc::from("test"),
        );
        let io: Arc<dyn crate::EncodingsIo> =
            Arc::new(SimulatedScheduler::new(bytes::Bytes::new()));
        let cache = Arc::new(lance_core::cache::LanceCache::with_capacity(4 * 1024));
        let context = SchedulerContext::new(io, cache.clone());

        scheduler
            .initialize(None, &FilterExpression::no_filter(), &context)
            .await
            .unwrap();

        assert_eq!(cache.size().await, 0);
    }

    #[tokio::test]
    async fn initialize_builds_only_requested_page_schedulers() {
        use crate::compression::DefaultDecompressionStrategy;
        use crate::decoder::{
            ColumnInfo, FilterExpression, PageEncoding, PageInfo, SchedulerContext,
            StructuralFieldScheduler,
        };
        use crate::format::pb;
        use lance_core::datatypes::Field;

        const NUM_PAGES: u64 = 64;
        const ROWS_PER_PAGE: u64 = 100;
        // All-null constant pages need no initialization I/O, so this test
        // observes construction alone.
        let page_infos = (0..NUM_PAGES)
            .map(|page_index| PageInfo {
                num_rows: ROWS_PER_PAGE,
                priority: page_index * ROWS_PER_PAGE,
                encoding: PageEncoding::Structural(metadata::PageLayout {
                    layout: Some(metadata::page_layout::Layout::ConstantLayout(
                        metadata::ConstantLayout {
                            layers: vec![metadata::RepDefLayer::RepdefNullableItem as i32],
                            ..Default::default()
                        },
                    )),
                }),
                buffer_offsets_and_sizes: Arc::new([]),
            })
            .collect::<Vec<_>>();
        let column_info = Arc::new(ColumnInfo::new(
            3,
            Arc::from(page_infos),
            Vec::new(),
            pb::ColumnEncoding::default(),
        ));
        let field = Field::try_from(&ArrowField::new("x", DataType::Int32, true)).unwrap();
        #[allow(deprecated)]
        let eager = StructuralPrimitiveFieldScheduler::try_new(
            &column_info,
            &DefaultDecompressionStrategy {},
            false,
            &field,
        )
        .unwrap();
        assert_eq!(eager.page_schedulers.len(), NUM_PAGES as usize);
        assert!(eager.page_source.is_none());

        let mut scheduler = StructuralPrimitiveFieldScheduler::try_new_lazy(
            column_info,
            Arc::new(DefaultDecompressionStrategy {}),
            false,
            &field,
        )
        .unwrap();
        assert!(
            scheduler.page_schedulers.is_empty(),
            "construction must not build page schedulers"
        );
        assert_eq!(scheduler.page_row_ends.len(), NUM_PAGES as usize);

        let io: Arc<dyn EncodingsIo> = Arc::new(SimulatedScheduler::new(bytes::Bytes::new()));
        let cache = Arc::new(lance_core::cache::LanceCache::no_cache());
        let mut context = SchedulerContext::new(io, cache);
        let filter = FilterExpression::no_filter();

        let requested_ranges = [150..151, 6_350..6_351];
        scheduler
            .initialize(Some(&requested_ranges), &filter, &context)
            .await
            .unwrap();
        let built_pages = |scheduler: &StructuralPrimitiveFieldScheduler| {
            scheduler
                .page_schedulers
                .iter()
                .map(|page| page.page_index)
                .collect::<Vec<_>>()
        };
        assert_eq!(built_pages(&scheduler), [1, 63]);
        assert_eq!(scheduler.page_schedulers[1].row_range, 6_300..6_400);
        assert!(scheduler.page_source.is_some());

        {
            let mut job = scheduler
                .schedule_ranges(&requested_ranges, &filter)
                .unwrap();
            // Both pages fit one read batch, so one call schedules them together.
            assert_eq!(job.schedule_next(&mut context).unwrap().len(), 2);
            assert!(job.schedule_next(&mut context).unwrap().is_empty());

            // Rows outside the initialized ranges must fail cleanly, not panic
            // on an uninitialized page.
            let mut job = scheduler.schedule_ranges(&[0..1], &filter).unwrap();
            let error = job.schedule_next(&mut context).unwrap_err();
            assert!(matches!(error, lance_core::Error::Internal { .. }));
            assert!(error.to_string().contains("not initialized"));
        }

        // Widening the request builds only the new pages and keeps page order.
        scheduler
            .initialize(Some(&[0..250]), &filter, &context)
            .await
            .unwrap();
        assert_eq!(built_pages(&scheduler), [0, 1, 2, 63]);

        // The eager path builds every page and releases the page source.
        scheduler.initialize(None, &filter, &context).await.unwrap();
        assert_eq!(scheduler.page_schedulers.len(), NUM_PAGES as usize);
        assert!(scheduler.page_source.is_none());
        let mut job = scheduler.schedule_ranges(&[0..1], &filter).unwrap();
        assert_eq!(job.schedule_next(&mut context).unwrap().len(), 1);
    }

    /// Records every `submit_request` (ranges and priority) and answers each
    /// range with a zero buffer that starts with the range's offset, so a test
    /// can count requests and check that every reader got its own ranges back.
    #[derive(Debug, Default)]
    struct RecordingScheduler {
        requests: std::sync::Mutex<Vec<Vec<Range<u64>>>>,
        priorities: std::sync::Mutex<Vec<u64>>,
    }

    impl crate::EncodingsIo for RecordingScheduler {
        fn submit_request(
            &self,
            ranges: Vec<Range<u64>>,
            priority: u64,
        ) -> futures::future::BoxFuture<'static, crate::Result<Vec<Bytes>>> {
            use futures::FutureExt;

            let buffers = ranges
                .iter()
                .map(|range| {
                    let mut buffer = vec![0u8; (range.end - range.start) as usize];
                    let marker = range.start.to_le_bytes();
                    let marker_len = marker.len().min(buffer.len());
                    buffer[..marker_len].copy_from_slice(&marker[..marker_len]);
                    Bytes::from(buffer)
                })
                .collect();
            self.requests.lock().unwrap().push(ranges);
            self.priorities.lock().unwrap().push(priority);
            std::future::ready(Ok(buffers)).boxed()
        }
    }

    #[tokio::test]
    async fn page_read_batch_submits_pending_reads_as_one_request() {
        let io = Arc::new(RecordingScheduler::default());
        let batch = RequestBatch::new(io.clone());
        let first = batch.submit_request(vec![100..108, 300..308], 7);
        let second = batch.submit_request(vec![200..208], 3);
        assert_eq!(batch.pending_bytes(), 24);
        assert!(
            io.requests.lock().unwrap().is_empty(),
            "nothing is read before the flush"
        );

        batch.flush();
        {
            let requests = io.requests.lock().unwrap();
            assert_eq!(requests.len(), 1, "the batch is one request");
            assert_eq!(
                requests[0],
                vec![100..108, 200..208, 300..308],
                "the batch is sorted by file offset"
            );
        }
        assert_eq!(
            io.priorities.lock().unwrap()[0],
            3,
            "the batch takes the lowest priority"
        );
        let first = first.await.unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].as_ref(), &100_u64.to_le_bytes());
        assert_eq!(first[1].as_ref(), &300_u64.to_le_bytes());
        let second = second.await.unwrap();
        assert_eq!(second.len(), 1);
        assert_eq!(second[0].as_ref(), &200_u64.to_le_bytes());

        // A read submitted after the flush is batched until it is awaited.
        let late = batch.submit_request(vec![400..408], 9).await.unwrap();
        assert_eq!(late[0].as_ref(), &400_u64.to_le_bytes());
        assert_eq!(io.requests.lock().unwrap().len(), 2);

        // Queued reads that would exceed the budget are submitted first, so
        // no batch grows past it.
        let io = Arc::new(RecordingScheduler::default());
        let batch = RequestBatch::new(io.clone());
        let budget = crate::decoder::IO_REQUEST_BATCH_BYTES;
        let first = batch.submit_request(vec![0..budget], 1);
        assert!(io.requests.lock().unwrap().is_empty());
        let second = batch.submit_request(vec![budget..(budget + 8)], 2);
        assert_eq!(
            io.requests.lock().unwrap().as_slice(),
            &[vec![0..budget]],
            "the full batch is submitted before the read that would overflow it"
        );
        batch.flush();
        assert_eq!(io.requests.lock().unwrap().len(), 2);
        assert_eq!(first.await.unwrap()[0].len(), budget as usize);
        assert_eq!(second.await.unwrap()[0].as_ref(), &budget.to_le_bytes());
    }

    #[tokio::test]
    async fn request_batch_sorts_a_lone_unsorted_read() {
        let io = Arc::new(RecordingScheduler::default());
        let batch = RequestBatch::new(io.clone());
        let read = batch.submit_request(vec![300..308, 100..108, 200..208], 1);
        batch.flush();
        assert_eq!(
            io.requests.lock().unwrap().as_slice(),
            &[vec![100..108, 200..208, 300..308]],
            "the I/O layer only accepts ranges sorted by offset"
        );
        let starts = read
            .await
            .unwrap()
            .iter()
            .map(|bytes| u64::from_le_bytes(bytes[..8].try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(starts, [300, 100, 200], "results come back in caller order");
    }

    /// Fails every request with the error its function builds.
    #[derive(Debug)]
    struct FailingScheduler(fn() -> lance_core::Error);

    impl crate::EncodingsIo for FailingScheduler {
        fn submit_request(
            &self,
            _ranges: Vec<Range<u64>>,
            _priority: u64,
        ) -> futures::future::BoxFuture<'static, crate::Result<Vec<Bytes>>> {
            futures::FutureExt::boxed(std::future::ready(Err((self.0)())))
        }
    }

    #[derive(Debug)]
    struct Cause;

    impl std::fmt::Display for Cause {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "typed cause")
        }
    }

    impl std::error::Error for Cause {}

    fn chain_has_cause(error: &lance_core::Error) -> bool {
        let mut next: Option<&(dyn std::error::Error + 'static)> = Some(error);
        while let Some(error) = next {
            if error.is::<Cause>() {
                return true;
            }
            next = error.source();
        }
        false
    }

    #[rstest::rstest]
    #[case::io_source(
        || lance_core::Error::io_source(Box::new(Cause)),
        chain_has_cause
    )]
    #[case::invalid_input(
        || lance_core::Error::invalid_input_source(Box::new(Cause)),
        chain_has_cause
    )]
    #[case::corrupt_file(
        || lance_core::Error::corrupt_file_named("data.lance", "bad page"),
        |e: &lance_core::Error| matches!(e.find_root(), lance_core::Error::CorruptFile { path, .. } if path.as_ref() == "data.lance")
    )]
    #[case::internal(|| lance_core::Error::internal("bug"), |_: &lance_core::Error| true)]
    #[case::timeout(|| lance_core::Error::timeout("slow"), |_: &lance_core::Error| true)]
    #[case::not_found(|| lance_core::Error::not_found("data.lance"), |e: &lance_core::Error| e.is_not_found())]
    #[case::wrapped_not_found(
        || lance_core::Error::wrapped(Box::new(lance_core::Error::not_found("data.lance"))),
        |e: &lance_core::Error| e.is_not_found()
    )]
    #[case::commit_status_unknown(
        || lance_core::Error::commit_status_unknown_source(7, Box::new(Cause)),
        |e: &lance_core::Error| e.is_commit_status_unknown() && chain_has_cause(e)
    )]
    #[case::external(
        || lance_core::Error::external(Box::new(Cause)),
        |e: &lance_core::Error| e.external_source().is_some_and(|source| source.is::<Cause>())
    )]
    #[case::fenced(
        || lance_core::Error::fenced_by_peer("epoch 7"),
        |e: &lance_core::Error| e.fence_reason() == Some(lance_core::error::FenceReason::PeerClaimedEpoch)
    )]
    #[case::backpressure(|| lance_core::Error::backpressure("full"), |e: &lance_core::Error| e.is_backpressure())]
    #[tokio::test]
    async fn request_batch_shares_the_original_error(
        #[case] failure: fn() -> lance_core::Error,
        #[case] check: fn(&lance_core::Error) -> bool,
    ) {
        let expected = failure();
        assert!(check(&expected), "{expected:?}");
        let batch = RequestBatch::new(Arc::new(FailingScheduler(failure)));
        let first = batch.submit_request(vec![0..8], 0);
        let second = batch.submit_request(vec![8..16], 0);
        batch.flush();
        for error in [first.await.unwrap_err(), second.await.unwrap_err()] {
            // The batch hands out `Error::Shared` while other readers still hold
            // the error, so compare the variant it wraps.
            assert_eq!(
                std::mem::discriminant(error.find_root()),
                std::mem::discriminant(&expected),
                "{error:?}"
            );
            assert_eq!(error.to_string(), expected.to_string());
            assert!(check(&error), "{error:?}");
        }
    }

    #[tokio::test]
    async fn request_batch_flush_keeps_collecting_and_awaiting_flushes() {
        let io = Arc::new(RecordingScheduler::default());
        let batch = RequestBatch::new(io.clone());
        let first = batch.submit_request(vec![100..108], 1);
        let second = batch.submit_request(vec![300..308], 2);
        batch.flush();
        assert_eq!(
            io.requests.lock().unwrap().as_slice(),
            &[vec![100..108, 300..308]],
            "a flush submits the collected reads as one request"
        );
        // The batch keeps collecting after a flush.
        let third = batch.submit_request(vec![500..508], 3);
        let fourth = batch.submit_request(vec![700..708], 4);
        assert_eq!(batch.pending_bytes(), 16);
        assert_eq!(io.requests.lock().unwrap().len(), 1);
        // Awaiting a read submits whatever is queued, so a read queued outside
        // a scheduling step (indirect I/O) is never stuck behind a missing flush.
        assert_eq!(third.await.unwrap()[0].as_ref(), &500_u64.to_le_bytes());
        assert_eq!(
            io.requests.lock().unwrap().as_slice(),
            &[vec![100..108, 300..308], vec![500..508, 700..708]]
        );
        assert_eq!(fourth.await.unwrap()[0].as_ref(), &700_u64.to_le_bytes());
        assert_eq!(first.await.unwrap()[0].as_ref(), &100_u64.to_le_bytes());
        assert_eq!(second.await.unwrap()[0].as_ref(), &300_u64.to_le_bytes());
        assert_eq!(io.requests.lock().unwrap().len(), 2);

        // Futures that submit when first polled and are polled in the same
        // pass (struct children initializing, list columns fetching items)
        // end up in one request: the flush waits for one yield.
        let (fifth, sixth) = futures::join!(
            async { batch.submit_request(vec![900..908], 5).await.unwrap() },
            async { batch.submit_request(vec![1100..1108], 6).await.unwrap() },
        );
        assert_eq!(fifth[0].as_ref(), &900_u64.to_le_bytes());
        assert_eq!(sixth[0].as_ref(), &1100_u64.to_le_bytes());
        assert_eq!(
            io.requests.lock().unwrap().last().unwrap(),
            &vec![900..908, 1100..1108],
            "reads submitted by sibling futures share one request"
        );
        assert_eq!(io.requests.lock().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn schedule_next_batches_page_reads_up_to_the_budget() {
        use super::{
            CachedPageData, IO_REQUEST_BATCH_BYTES, PageInfoAndScheduler, PageInitialization,
            StructuralPageScheduler, StructuralPrimitiveFieldScheduler,
        };
        use crate::decoder::{FilterExpression, SchedulerContext, StructuralFieldScheduler};
        use crate::encodings::logical::primitive::{PageInitializationBuffers, PageLoadTask};

        /// A page that reads `data_ranges`, one shard (and one load task) per
        /// range. The bytes it gets back are covered by the batch test, so its
        /// load futures just fail and the test never has to build a decoder.
        #[derive(Debug)]
        struct FakeDataPage {
            data_ranges: Vec<Range<u64>>,
        }

        impl StructuralPageScheduler for FakeDataPage {
            fn needs_initialization(&self) -> bool {
                false
            }

            fn init_layout(&self) -> crate::Result<PageInitialization> {
                unreachable!("fake data pages need no initialization")
            }

            fn init_from_buffers<'a>(
                &'a mut self,
                _buffers: PageInitializationBuffers,
                _io: &Arc<dyn EncodingsIo>,
            ) -> futures::future::BoxFuture<'a, crate::Result<Arc<dyn CachedPageData>>>
            {
                unreachable!("fake data pages need no initialization")
            }

            fn try_load(&mut self, _data: &Arc<dyn CachedPageData>) -> crate::Result<()> {
                Ok(())
            }

            fn schedule_ranges(
                &self,
                ranges: &[Range<u64>],
                io: &Arc<dyn EncodingsIo>,
            ) -> crate::Result<Vec<PageLoadTask>> {
                use futures::FutureExt;

                let num_rows = ranges.iter().map(|r| r.end - r.start).sum();
                Ok(self
                    .data_ranges
                    .iter()
                    .enumerate()
                    .map(|(shard, range)| {
                        let read = io.submit_request(vec![range.clone()], range.start);
                        PageLoadTask {
                            decoder_fut: read
                                .map(|_| Err(lance_core::Error::internal("fake page")))
                                .boxed(),
                            // The page's rows are reported once, on the first shard.
                            num_rows: if shard == 0 { num_rows } else { 0 },
                        }
                    })
                    .collect())
            }
        }

        fn scheduler_with_pages(
            page_bytes: u64,
            num_pages: u64,
        ) -> StructuralPrimitiveFieldScheduler {
            // Leave a gap between pages so the test can tell batching from
            // the I/O layer's coalescing of adjacent ranges.
            let stride = page_bytes * 2;
            let pages = (0..num_pages)
                .map(|page| PageInfoAndScheduler {
                    page_index: page as usize,
                    row_range: (page * 10)..((page + 1) * 10),
                    scheduler: Box::new(FakeDataPage {
                        data_ranges: vec![(page * stride)..(page * stride + page_bytes)],
                    }),
                })
                .collect();
            StructuralPrimitiveFieldScheduler::from_page_schedulers(pages, 0, Arc::from("test"))
        }

        let filter = FilterExpression::no_filter();
        let cache = Arc::new(lance_core::cache::LanceCache::no_cache());

        // Small pages: every page of the request is read in one batch.
        let io = Arc::new(RecordingScheduler::default());
        let scheduler = scheduler_with_pages(64, 6);
        let mut context = SchedulerContext::new(io.clone(), cache.clone());
        let mut job = scheduler.schedule_ranges(&[5..45], &filter).unwrap();
        let scan_lines = job.schedule_next(&mut context).unwrap();
        context.flush_io();
        assert_eq!(scan_lines.len(), 5, "one scan line per touched page");
        assert_eq!(
            scan_lines
                .iter()
                .map(|line| line.rows_scheduled)
                .sum::<u64>(),
            40
        );
        assert!(job.schedule_next(&mut context).unwrap().is_empty());
        {
            let requests = io.requests.lock().unwrap();
            assert_eq!(requests.len(), 1, "five pages cost one request");
            assert_eq!(
                requests[0],
                vec![0..64, 128..192, 256..320, 384..448, 512..576],
                "the untouched last page is not read"
            );
        }
        assert_eq!(
            io.priorities.lock().unwrap()[0],
            0,
            "the batch carries the first page's priority"
        );

        // Pages as large as the batch budget are read one request at a time.
        let io = Arc::new(RecordingScheduler::default());
        let scheduler = scheduler_with_pages(IO_REQUEST_BATCH_BYTES, 2);
        let mut context = SchedulerContext::new(io.clone(), cache);
        let mut job = scheduler.schedule_ranges(&[0..20], &filter).unwrap();
        let first = job.schedule_next(&mut context).unwrap();
        context.flush_io();
        assert_eq!(first.len(), 1);
        assert_eq!(io.requests.lock().unwrap().len(), 1);
        let second = job.schedule_next(&mut context).unwrap();
        context.flush_io();
        assert_eq!(second.len(), 1);
        assert_eq!(io.requests.lock().unwrap().len(), 2);
        assert!(job.schedule_next(&mut context).unwrap().is_empty());

        // A page that shards its own reads (like a blob page) keeps every
        // shard in a bounded read instead of one request for the whole page.
        let io = Arc::new(RecordingScheduler::default());
        let shard = IO_REQUEST_BATCH_BYTES;
        let scheduler = StructuralPrimitiveFieldScheduler::from_page_schedulers(
            vec![PageInfoAndScheduler {
                page_index: 0,
                row_range: 0..10,
                scheduler: Box::new(FakeDataPage {
                    data_ranges: vec![0..shard, (2 * shard)..(3 * shard)],
                }),
            }],
            0,
            Arc::from("test"),
        );
        let mut context = SchedulerContext::new(
            io.clone(),
            Arc::new(lance_core::cache::LanceCache::no_cache()),
        );
        let mut job = scheduler.schedule_ranges(&[0..10], &filter).unwrap();
        let scan_lines = job.schedule_next(&mut context).unwrap();
        context.flush_io();
        assert_eq!(scan_lines.len(), 2, "one scan line per shard");
        assert_eq!(
            io.requests.lock().unwrap().as_slice(),
            &[vec![0..shard], vec![(2 * shard)..(3 * shard)]],
            "each shard stays its own bounded request"
        );
        assert!(job.schedule_next(&mut context).unwrap().is_empty());
    }

    #[test]
    fn page_range_mapping_validates_and_splits_ranges() {
        // Exclusive end rows of three ten-row pages.
        let pages = [10_u64, 20, 30];

        let mappings = map_ranges_to_pages(&pages, &[8..22]).unwrap();
        assert_eq!(mappings.len(), 3);
        assert_eq!(mappings[0].page_idx, 0);
        assert_eq!(mappings[0].ranges_in_page, [8..10]);
        assert_eq!(mappings[1].page_idx, 1);
        assert_eq!(mappings[1].ranges_in_page, [0..10]);
        assert_eq!(mappings[2].page_idx, 2);
        assert_eq!(mappings[2].ranges_in_page, [0..2]);

        let sparse_mappings = map_ranges_to_pages(&pages, &[1..2, 3..4, 28..29]).unwrap();
        assert_eq!(sparse_mappings.len(), 2);
        assert_eq!(sparse_mappings[0].page_idx, 0);
        assert_eq!(sparse_mappings[0].ranges_in_page, [1..2, 3..4]);
        assert_eq!(sparse_mappings[1].page_idx, 2);
        assert_eq!(sparse_mappings[1].ranges_in_page, [8..9]);
        assert_eq!(
            pages_overlapping_ranges(&pages, &[1..2, 3..4, 28..29]).unwrap(),
            [0, 2]
        );

        for (ranges, expected_message) in [
            (vec![Range { start: 2, end: 1 }], "is reversed"),
            (vec![15..16, 5..6], "not strictly ordered"),
            (vec![5..15, 10..20], "overlap"),
            (vec![5..5, 5..6], "not strictly ordered"),
        ] {
            let error = map_ranges_to_pages(&pages, &ranges).unwrap_err();
            assert!(matches!(error, lance_core::Error::InvalidInput { .. }));
            assert!(error.to_string().contains(expected_message));
        }
    }
}
