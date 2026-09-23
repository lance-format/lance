// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One segment of a MinHash LSH index: opening its files, candidate lookup
//! through the page table, refining candidates against the signature table,
//! prewarming, and the rebuilds that update, remap or merge segments.

use super::*;

use std::collections::VecDeque;

use futures::future::try_join_all;

/// One logical page of `bands.lance`, cached per page.
#[derive(Debug, DeepSizeOf)]
pub(super) struct BandPage {
    keys: Vec<u64>,
    doc_ids: Vec<u32>,
}

impl BandPage {
    fn try_from_batch(batch: &RecordBatch) -> Result<Self> {
        let corrupt = |message: String| Error::corrupt_file_named(BANDS_FILENAME, message);
        let keys = batch
            .column_by_name(BAND_KEY_COL)
            .and_then(|column| column.as_primitive_opt::<UInt64Type>())
            .filter(|keys| keys.null_count() == 0)
            .ok_or_else(|| corrupt(format!("{BAND_KEY_COL} is not a non-null UInt64 column")))?;
        let doc_ids = batch
            .column_by_name(DOC_ID_COL)
            .and_then(|column| column.as_primitive_opt::<UInt32Type>())
            .filter(|doc_ids| doc_ids.null_count() == 0)
            .ok_or_else(|| corrupt(format!("{DOC_ID_COL} is not a non-null UInt32 column")))?;
        // Copy out of the batch so a cached page neither pins nor is charged
        // for the buffers of every other page read in the same request.
        Ok(Self {
            keys: keys.values().to_vec(),
            doc_ids: doc_ids.values().to_vec(),
        })
    }
}

#[derive(Debug, Clone)]
struct BandPageKey {
    page: u32,
}

impl CacheKey for BandPageKey {
    type ValueType = BandPage;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("band-page-{}", self.page).into()
    }

    fn type_name() -> &'static str {
        "MinHashLshBandPage"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.minhashlsh.band-page-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.page);
    }
}

/// `signature_chunk_docs` consecutive documents of the signature table,
/// resident after a prewarm.
#[derive(Debug, DeepSizeOf)]
pub(super) struct SignatureChunk {
    row_ids: Vec<u64>,
    /// `row_ids.len() * num_hashes` values.
    signatures: Vec<SignatureValue>,
}

#[derive(Debug, Clone)]
struct SignatureChunkKey {
    chunk: u32,
}

impl CacheKey for SignatureChunkKey {
    type ValueType = SignatureChunk;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("signature-chunk-{}", self.chunk).into()
    }

    fn type_name() -> &'static str {
        "MinHashLshSignatureChunk"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.minhashlsh.signature-chunk-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.chunk);
    }
}

/// A single segment of a MinHash LSH index.
pub struct MinHashLshIndex {
    params: MinHashLshIndexParams,
    generator: SignatureGenerator,
    bands: Arc<dyn IndexReader>,
    signatures: Arc<dyn IndexReader>,
    page_rows: usize,
    /// Largest band key of each logical page of `bands.lance`.
    page_max_keys: Vec<u64>,
    num_docs: usize,
    /// Documents per resident signature chunk; see [`RESIDENT_CHUNK_BYTES`].
    signature_chunk_docs: usize,
    /// Candidates held per level of a search and rows per signature read:
    /// one IO batch of signature rows.
    pub(super) candidate_batch: usize,
    /// Pages a walk reads per round: one IO batch of band rows, shared by the
    /// buckets not yet walked to their end.
    pub(super) window_pages: usize,
    cache: WeakLanceCache,
    frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
}

impl std::fmt::Debug for MinHashLshIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinHashLshIndex")
            .field("params", &self.params)
            .field("num_docs", &self.num_docs)
            .field("num_pages", &self.page_max_keys.len())
            .finish()
    }
}

impl DeepSizeOf for MinHashLshIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        // Pages and signature chunks live in the index cache, which sizes them
        // itself; the parameters are a few strings, not worth counting.
        self.page_max_keys.deep_size_of_children(context)
    }
}

fn metadata_value<'a>(
    metadata: &'a HashMap<String, String>,
    file: &str,
    key: &str,
) -> Result<&'a String> {
    metadata.get(key).ok_or_else(|| {
        Error::corrupt_file_named(file, format!("missing schema metadata key {key}"))
    })
}

/// Checks that `reader` is `file` of a segment built from `params`: the exact
/// schema, the details repeated in the metadata and the layout version.
fn check_index_file(
    reader: &dyn IndexReader,
    file: &str,
    expected: &Schema,
    params: &MinHashLshIndexParams,
) -> Result<()> {
    let corrupt = |message: String| Error::corrupt_file_named(file, message);
    let expected = lance_core::datatypes::Schema::try_from(expected)?;
    reader
        .schema()
        .check_compatible(&expected, &SchemaCompareOptions::default())
        .map_err(|err| corrupt(format!("schema does not match the expected schema: {err}")))?;
    let metadata = &reader.schema().metadata;
    let index_version: u32 = metadata_value(metadata, file, INDEX_VERSION_META_KEY)?
        .parse()
        .map_err(|err| corrupt(format!("invalid {INDEX_VERSION_META_KEY}: {err}")))?;
    if index_version > MINHASH_LSH_INDEX_VERSION {
        return Err(Error::not_supported(format!(
            "MinHash LSH index version {index_version} is newer than the supported version {MINHASH_LSH_INDEX_VERSION}"
        )));
    }
    let file_params =
        MinHashLshIndexParams::from_details_hex(metadata_value(metadata, file, DETAILS_META_KEY)?)?;
    if file_params != *params {
        return Err(corrupt(format!(
            "index file details {file_params:?} do not match index details {params:?}"
        )));
    }
    Ok(())
}

impl MinHashLshIndex {
    pub async fn load(
        store: Arc<dyn IndexStore>,
        details: &prost_types::Any,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<Self>> {
        let params = MinHashLshIndexParams::from_details_any(details)?;
        let (bands, signatures) = futures::try_join!(
            store.open_index_file(BANDS_FILENAME),
            store.open_index_file(SIGNATURES_FILENAME)
        )?;

        check_index_file(bands.as_ref(), BANDS_FILENAME, &BANDS_SCHEMA, &params)?;
        check_index_file(
            signatures.as_ref(),
            SIGNATURES_FILENAME,
            &signatures_schema(params.num_hashes as i32),
            &params,
        )?;

        let metadata = &bands.schema().metadata;
        let corrupt = |message: String| Error::corrupt_file_named(BANDS_FILENAME, message);
        let page_rows: usize = metadata_value(metadata, BANDS_FILENAME, PAGE_ROWS_META_KEY)?
            .parse()
            .map_err(|err| corrupt(format!("invalid {PAGE_ROWS_META_KEY}: {err}")))?;
        if page_rows == 0 {
            return Err(corrupt(format!("{PAGE_ROWS_META_KEY} must be positive")));
        }
        let num_docs: usize = metadata_value(metadata, BANDS_FILENAME, NUM_DOCS_META_KEY)?
            .parse()
            .map_err(|err| corrupt(format!("invalid {NUM_DOCS_META_KEY}: {err}")))?;
        if num_docs != signatures.num_rows() {
            return Err(corrupt(format!(
                "bands file records {num_docs} documents but {SIGNATURES_FILENAME} has {} rows",
                signatures.num_rows()
            )));
        }
        let page_table_buffer: u32 =
            metadata_value(metadata, BANDS_FILENAME, PAGE_TABLE_BUFFER_META_KEY)?
                .parse()
                .map_err(|err| corrupt(format!("invalid {PAGE_TABLE_BUFFER_META_KEY}: {err}")))?;
        let page_table = bands.read_global_buffer(page_table_buffer).await?;
        if page_table.len() % 8 != 0 {
            return Err(corrupt(format!(
                "page table buffer length {} is not a multiple of 8",
                page_table.len()
            )));
        }
        let page_max_keys: Vec<u64> = page_table
            .chunks_exact(8)
            .map(|bytes| {
                u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                ])
            })
            .collect();
        let expected_pages = bands.num_rows().div_ceil(page_rows);
        if page_max_keys.len() != expected_pages {
            return Err(corrupt(format!(
                "page table has {} entries but {} rows at {page_rows} rows per page need {expected_pages}",
                page_max_keys.len(),
                bands.num_rows()
            )));
        }

        let generator = SignatureGenerator::try_new(&params)?;
        let row_bytes = signature_row_bytes(params.num_hashes as usize);
        let signature_chunk_docs = (RESIDENT_CHUNK_BYTES / row_bytes).max(1);
        let candidate_batch = rows_per_batch(row_bytes);
        Ok(Arc::new(Self {
            params,
            generator,
            bands,
            signatures,
            page_rows,
            page_max_keys,
            num_docs,
            signature_chunk_docs,
            candidate_batch,
            window_pages: (rows_per_batch(BAND_ROW_BYTES) / page_rows).max(1),
            cache: WeakLanceCache::from(cache),
            frag_reuse_index,
        }))
    }

    pub fn params(&self) -> &MinHashLshIndexParams {
        &self.params
    }

    /// Number of documents (rows with at least one token) in this segment.
    pub fn num_docs(&self) -> usize {
        self.num_docs
    }

    fn signature_source<'a>(&self, transform: RowIdTransform<'a>) -> SignatureSource<'a> {
        SignatureSource {
            reader: self.signatures.clone(),
            num_docs: self.num_docs,
            frag_reuse_index: self.frag_reuse_index.clone(),
            transform,
        }
    }

    fn created_index(&self, files: Vec<IndexFile>) -> Result<CreatedIndex> {
        Ok(CreatedIndex {
            index_details: self.params.details_any()?,
            index_version: MINHASH_LSH_INDEX_VERSION,
            files,
        })
    }

    fn page_range(&self, page: u32) -> Range<usize> {
        let start = page as usize * self.page_rows;
        start..(start + self.page_rows).min(self.bands.num_rows())
    }

    /// Signature and band keys of `text`, or `None` when the text has no
    /// tokens and therefore matches nothing.
    pub fn query_signature(&self, text: &str) -> Option<QuerySignature> {
        QuerySignature::compute(&mut self.generator.clone(), text)
    }

    /// Return the `limit` rows with the smallest Jaccard distance to `text`
    /// among rows selected by `mask`, ordered by ascending distance and then
    /// row id. Which rows are returned among equal distances is not specified.
    pub async fn search_text(
        &self,
        text: &str,
        limit: usize,
        mask: &RowAddrMask,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<MinHashHit>> {
        match self.query_signature(text) {
            Some(query) => self.search_signature(&query, limit, mask, metrics).await,
            None => Ok(Vec::new()),
        }
    }

    /// Like [`Self::search_text`] for a signature computed by
    /// [`Self::query_signature`] on any segment with the same parameters.
    pub async fn search_signature(
        &self,
        query: &QuerySignature,
        limit: usize,
        mask: &RowAddrMask,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<MinHashHit>> {
        if query.signature.len() != self.params.num_hashes as usize {
            return Err(Error::invalid_input(format!(
                "query signature has {} values but the index uses num_hashes={}",
                query.signature.len(),
                self.params.num_hashes
            )));
        }
        if limit == 0 || self.num_docs == 0 {
            return Ok(Vec::new());
        }
        let mut buckets = self.open_buckets(&query.band_keys, metrics).await?;
        if buckets.cursors.is_empty() {
            return Ok(Vec::new());
        }
        self.refine(&mut buckets, &query.signature, limit, mask, metrics)
            .await
    }

    /// Locate the bucket of every band key and position a cursor at its
    /// first row.
    ///
    /// A bucket is a run of equal keys in the sorted bands file, and the page
    /// table locates it without probing: its pages run from the first page
    /// whose max key reaches the key through the first page whose max key
    /// exceeds it, and the pages strictly between hold nothing else. The two
    /// boundary pages of each bucket are fetched through the cache and give
    /// the exact row range by binary search; the cursor then walks the rows
    /// one window of pages at a time, through the cache as well.
    async fn open_buckets(
        &self,
        band_keys: &[u64],
        metrics: &dyn MetricsCollector,
    ) -> Result<BucketScan> {
        let num_pages = self.page_max_keys.len();
        let located: Vec<(u64, usize, usize)> = band_keys
            .iter()
            .filter_map(|&key| {
                let first = self.page_max_keys.partition_point(|&max_key| max_key < key);
                // `first == num_pages` means the key is larger than every stored key.
                (first < num_pages).then(|| {
                    let end = self
                        .page_max_keys
                        .partition_point(|&max_key| max_key <= key);
                    (key, first, end)
                })
            })
            .collect();
        metrics.record_comparisons(band_keys.len());
        let mut boundary_pages: Vec<u32> = located
            .iter()
            .flat_map(|&(_, first, end)| [first as u32, end as u32])
            .filter(|&page| (page as usize) < num_pages)
            .collect();
        boundary_pages.sort_unstable();
        boundary_pages.dedup();
        let pages = self.load_pages(&boundary_pages, metrics).await?;
        let page = |page: usize| {
            pages.get(&(page as u32)).cloned().ok_or_else(|| {
                Error::internal(format!("band page {page} was requested but not loaded"))
            })
        };
        let mut cursors = Vec::with_capacity(located.len());
        for (key, first_page, end_page) in located {
            let first = page(first_page)?;
            let page_start = first_page * self.page_rows;
            let row = page_start + first.keys.partition_point(|&k| k < key);
            let end = if end_page < num_pages {
                end_page * self.page_rows + page(end_page)?.keys.partition_point(|&k| k <= key)
            } else {
                // The bucket reaches the end of the file
                self.bands.num_rows()
            };
            if row < end {
                // The first window is the bucket's rows of the page in hand,
                // all of them for the usual bucket within one page
                let window = first.doc_ids
                    [row - page_start..end.min(page_start + first.doc_ids.len()) - page_start]
                    .to_vec();
                cursors.push(BucketCursor {
                    next: row,
                    end,
                    window: window.into_iter(),
                });
            }
        }
        Ok(BucketScan { cursors })
    }

    /// Fetch pages through the cache; the pages missing from the cache are
    /// read with scattered reads of at most [`IO_BATCH_BYTES`] each.
    async fn load_pages(
        &self,
        pages: &[u32],
        metrics: &dyn MetricsCollector,
    ) -> Result<HashMap<u32, Arc<BandPage>>> {
        let mut loaded = HashMap::with_capacity(pages.len());
        let mut missing = Vec::new();
        for &page in pages {
            match self.cache.get_with_key(&BandPageKey { page }).await {
                Some(cached) => {
                    metrics.record_index_cache_hit();
                    loaded.insert(page, cached);
                }
                None => missing.push(page),
            }
        }
        metrics.record_index_cache_misses(missing.len());
        let pages_per_read = (rows_per_batch(BAND_ROW_BYTES) / self.page_rows).max(1);
        for missing in missing.chunks(pages_per_read) {
            metrics.record_parts_loaded(missing.len());
            tracing::info!(
                target: TRACE_IO_EVENTS,
                r#type = IO_TYPE_LOAD_SCALAR_PART,
                index_type = "minhashlsh",
                num_parts = missing.len(),
            );
            let ranges: Vec<Range<usize>> =
                missing.iter().map(|&page| self.page_range(page)).collect();
            let batch = self.bands.read_ranges(&ranges, None).await?;
            let mut offset = 0;
            for (&page, range) in missing.iter().zip(&ranges) {
                let band_page =
                    Arc::new(BandPage::try_from_batch(&batch.slice(offset, range.len()))?);
                offset += range.len();
                self.cache
                    .insert_with_key(&BandPageKey { page }, band_page.clone())
                    .await;
                loaded.insert(page, band_page);
            }
        }
        Ok(loaded)
    }

    async fn load_signature_chunk(&self, chunk: usize) -> Result<SignatureChunk> {
        let num_hashes = self.params.num_hashes as usize;
        let start = chunk * self.signature_chunk_docs;
        let end = (start + self.signature_chunk_docs).min(self.num_docs);
        let batch = self.signatures.read_range(start..end, None).await?;
        let (row_ids, signatures) = signature_columns(&batch, num_hashes)?;
        if row_ids.len() != end - start {
            return Err(Error::corrupt_file_named(
                SIGNATURES_FILENAME,
                format!(
                    "read of docs {start}..{end} returned {} rows",
                    row_ids.len()
                ),
            ));
        }
        Ok(SignatureChunk {
            row_ids: row_ids.values().to_vec(),
            signatures: signatures.to_vec(),
        })
    }

    /// The doc ids of the `pages` pages of `cursor`'s bucket that follow the
    /// rows it holds, fetched through the page cache (the pages missing from
    /// it are read together).
    async fn load_window(
        &self,
        cursor: &BucketCursor,
        pages: usize,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<u32>> {
        let row = cursor.next + cursor.window.as_slice().len();
        let first_page = row / self.page_rows;
        let last_page = ((cursor.end - 1) / self.page_rows).min(first_page + pages - 1);
        let pages: Vec<u32> = (first_page..=last_page).map(|page| page as u32).collect();
        let loaded = self.load_pages(&pages, metrics).await?;
        let to = cursor.end.min((last_page + 1) * self.page_rows);
        let mut window = Vec::with_capacity(to - row);
        for page in pages {
            let band_page = loaded.get(&page).ok_or_else(|| {
                Error::internal(format!("band page {page} was requested but not loaded"))
            })?;
            let page_start = page as usize * self.page_rows;
            let from = row.max(page_start) - page_start;
            let until = to.min(page_start + band_page.doc_ids.len()) - page_start;
            window.extend_from_slice(&band_page.doc_ids[from..until]);
        }
        Ok(window)
    }

    /// The next doc id in any bucket of `scan`, in ascending order, with the
    /// number of buckets (bands) it appears in; `None` once every bucket is
    /// exhausted. The buckets are sorted by doc id, so this is a merge of
    /// their cursors, each refilled with a bounded window as it runs out.
    async fn next_candidate(
        &self,
        scan: &mut BucketScan,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<(u32, u32)>> {
        let live = |cursor: &BucketCursor| cursor.next < cursor.end;
        let drained = |cursor: &BucketCursor| live(cursor) && cursor.window.as_slice().is_empty();
        if scan.cursors.iter().any(drained) {
            // The page budget is shared by the buckets still being walked, so
            // a walk that outlives the others reads in full batches.
            let live_cursors = scan.cursors.iter().filter(|cursor| live(cursor)).count();
            let pages = (self.window_pages / live_cursors.max(1)).max(1);
            let window_rows = pages * self.page_rows;
            // One round tops up every bucket that is running low, not only the
            // drained one, so buckets advancing together cost one concurrent
            // read per round rather than one read each. A bucket the merge has
            // not reached yet keeps its window and stops reading, so no bucket
            // holds more than two windows.
            let refills: Vec<usize> = scan
                .cursors
                .iter()
                .enumerate()
                .filter(|(_, cursor)| {
                    let held = cursor.window.as_slice().len();
                    held < window_rows && cursor.next + held < cursor.end
                })
                .map(|(index, _)| index)
                .collect();
            let windows = try_join_all(
                refills
                    .iter()
                    .map(|&index| self.load_window(&scan.cursors[index], pages, metrics)),
            )
            .await?;
            for (&index, window) in refills.iter().zip(windows) {
                let cursor = &mut scan.cursors[index];
                if cursor.window.as_slice().is_empty() {
                    cursor.window = window.into_iter();
                } else {
                    let mut held = cursor.window.as_slice().to_vec();
                    held.extend(window);
                    cursor.window = held.into_iter();
                }
            }
        }
        let mut min_doc = u32::MAX;
        let mut shared = 0u32;
        for cursor in &scan.cursors {
            if let Some(&doc_id) = cursor.window.as_slice().first() {
                if doc_id < min_doc {
                    min_doc = doc_id;
                    shared = 1;
                } else if doc_id == min_doc {
                    shared += 1;
                }
            }
        }
        if shared == 0 {
            return Ok(None);
        }
        if min_doc as usize >= self.num_docs {
            return Err(Error::corrupt_file_named(
                BANDS_FILENAME,
                format!(
                    "posting references doc id {min_doc} but the segment has {} documents",
                    self.num_docs
                ),
            ));
        }
        for cursor in &mut scan.cursors {
            if cursor.window.as_slice().first() == Some(&min_doc) {
                cursor.window.next();
                cursor.next += 1;
            }
        }
        Ok(Some((min_doc, shared)))
    }

    /// Walk the buckets from the cursors' positions, sorting every doc id met
    /// into the level of its shared band count. Each level keeps its first
    /// `cap` doc ids and remembers where the scan stood when it filled up, so
    /// the level can be continued from there. Returns when level
    /// `pause_level` fills up or when the buckets are exhausted.
    async fn scan_levels(
        &self,
        scan: &mut BucketScan,
        levels: &mut CandidateLevels,
        pause_level: usize,
        metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        while let Some((doc_id, shared)) = self.next_candidate(scan, metrics).await? {
            let shared = shared as usize;
            levels.counts[shared] += 1;
            let list = &mut levels.lists[shared];
            if list.len() < levels.cap {
                list.push(doc_id);
                if list.len() == levels.cap {
                    levels.resume[shared] = Some(scan.positions());
                    if shared == pause_level {
                        return Ok(());
                    }
                }
            }
        }
        levels.complete = true;
        Ok(())
    }

    /// The next doc id of level `shared` from the cursors' positions.
    async fn next_in_level(
        &self,
        scan: &mut BucketScan,
        shared: usize,
        metrics: &dyn MetricsCollector,
    ) -> Result<Option<u32>> {
        while let Some((doc_id, count)) = self.next_candidate(scan, metrics).await? {
            if count as usize == shared {
                return Ok(Some(doc_id));
            }
        }
        Ok(None)
    }

    /// Up to `cap` further doc ids of level `shared`, and whether the buckets
    /// are exhausted.
    async fn next_level_chunk(
        &self,
        scan: &mut BucketScan,
        shared: usize,
        cap: usize,
        metrics: &dyn MetricsCollector,
    ) -> Result<(Vec<u32>, bool)> {
        let mut doc_ids = Vec::new();
        while doc_ids.len() < cap {
            match self.next_in_level(scan, shared, metrics).await? {
                Some(doc_id) => doc_ids.push(doc_id),
                None => return Ok((doc_ids, true)),
            }
        }
        Ok((doc_ids, false))
    }

    /// Score the candidates of the buckets against the query signature and
    /// keep the best `limit` rows selected by `mask`.
    ///
    /// A candidate sharing `m` of the `b` bands differs from the query in at
    /// least one value of every other band, so its distance is at least
    /// `(b - m) / k`. Candidates are therefore queued by decreasing shared
    /// bands in [`Pending`]: the top level as the walk finds it, then every
    /// lower level, its first batch held by the walk and the rest walked again
    /// from where it overflowed. Nothing is materialized beyond what the
    /// results miss plus one batch per level. Which rows are returned among
    /// equal distances is not specified.
    async fn refine(
        &self,
        scan: &mut BucketScan,
        query: &[SignatureValue],
        limit: usize,
        mask: &RowAddrMask,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<MinHashHit>> {
        let num_hashes = query.len();
        let num_bands = self.params.num_bands as usize;
        let cap = self.candidate_batch;
        let mut scorer = Scorer {
            hits: TopHits::new(limit),
            query,
            mask,
            remapper: self.frag_reuse_index.as_deref(),
        };
        let mut levels = CandidateLevels::new(num_bands, cap);
        let missing = scorer.hits.missing();
        let mut pending = Pending {
            candidates: VecDeque::new(),
            must: missing,
            batch_rows: missing.max(MIN_REFINE_READ_ROWS).min(cap),
        };

        // Top level: candidates sharing every band, at distance zero unless a
        // band hash collided. The walk pauses whenever a batch of them is
        // held, so a large cluster can end the search without being walked.
        loop {
            self.scan_levels(scan, &mut levels, num_bands, metrics)
                .await?;
            pending.queue(0.0, std::mem::take(&mut levels.lists[num_bands]));
            if !self
                .read_queued(&mut pending, false, false, &mut scorer, metrics)
                .await?
            {
                return Ok(scorer.hits.into_sorted());
            }
            if levels.complete {
                break;
            }
        }

        // Lower levels, complete counts known.
        for shared in (1..num_bands).rev() {
            let floor = (num_bands - shared) as f32 / num_hashes as f32;
            if scorer.done(pending.floor().unwrap_or(floor)) {
                return Ok(scorer.hits.into_sorted());
            }
            pending.queue(floor, std::mem::take(&mut levels.lists[shared]));
            if let Some(positions) = levels.resume[shared].take() {
                scan.seek(&positions);
                let remaining = levels.counts[shared] - cap as u64;
                if remaining.saturating_mul(100)
                    > (self.num_docs as u64).saturating_mul(SPARSE_REFINE_READ_PERCENT)
                {
                    if !self
                        .read_queued(&mut pending, true, true, &mut scorer, metrics)
                        .await?
                    {
                        return Ok(scorer.hits.into_sorted());
                    }
                    self.score_level_dense(scan, shared, floor, &mut scorer, metrics)
                        .await?;
                    // The scan read far more than the results missed
                    pending.must = 0;
                    continue;
                }
                loop {
                    if !self
                        .read_queued(&mut pending, false, false, &mut scorer, metrics)
                        .await?
                        || scorer.done(pending.floor().unwrap_or(floor))
                    {
                        return Ok(scorer.hits.into_sorted());
                    }
                    let (doc_ids, exhausted) =
                        self.next_level_chunk(scan, shared, cap, metrics).await?;
                    pending.queue(floor, doc_ids);
                    if exhausted {
                        break;
                    }
                }
            }
            if !self
                .read_queued(&mut pending, true, false, &mut scorer, metrics)
                .await?
            {
                return Ok(scorer.hits.into_sorted());
            }
        }
        self.read_queued(&mut pending, true, true, &mut scorer, metrics)
            .await?;
        Ok(scorer.hits.into_sorted())
    }

    /// Read what `pending` holds as far as its plan allows: first the
    /// candidates the results were missing when the search started, without
    /// a stop check in between, and every queued one when the queue ends at
    /// a level boundary (`level_end`), the only place a search usually stops;
    /// then batches of doubling size, each only while its first candidate can
    /// still improve the results. With `flush`, whatever is queued is read.
    /// Returns false once the search can stop.
    async fn read_queued(
        &self,
        pending: &mut Pending,
        level_end: bool,
        flush: bool,
        scorer: &mut Scorer<'_>,
        metrics: &dyn MetricsCollector,
    ) -> Result<bool> {
        while let Some(floor) = pending.floor() {
            let queued = pending.candidates.len();
            let rows = if pending.must > 0 {
                if queued < pending.must && !flush {
                    return Ok(true);
                }
                if level_end || flush {
                    queued
                } else {
                    pending.must
                }
            } else {
                if scorer.done(floor) {
                    return Ok(false);
                }
                if queued < pending.batch_rows && !flush {
                    return Ok(true);
                }
                pending.batch_rows.min(queued)
            };
            // Sorted as a whole before it is split into IO batches, so that
            // neighbouring rows share a batch and their requests coalesce
            let mut doc_ids: Vec<u32> = pending
                .candidates
                .drain(..rows)
                .map(|(doc_id, _)| doc_id)
                .collect();
            doc_ids.sort_unstable();
            for batch in doc_ids.chunks(self.candidate_batch) {
                self.score_batch(batch, scorer, metrics).await?;
            }
            if pending.must > 0 {
                pending.must = pending.must.saturating_sub(rows);
            } else {
                pending.batch_rows = (pending.batch_rows * 2).min(self.candidate_batch);
            }
        }
        Ok(true)
    }

    /// Score one batch of ascending `doc_ids`: those in resident signature
    /// chunks from memory, the rest with one scattered read.
    async fn score_batch(
        &self,
        doc_ids: &[u32],
        scorer: &mut Scorer<'_>,
        metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        let num_hashes = self.params.num_hashes as usize;
        let chunk_docs = self.signature_chunk_docs;
        metrics.record_comparisons(doc_ids.len());
        let mut pending: Vec<u32> = Vec::new();
        let mut start = 0;
        while start < doc_ids.len() {
            let chunk = doc_ids[start] as usize / chunk_docs;
            let end = start
                + doc_ids[start..].partition_point(|&doc_id| doc_id as usize / chunk_docs == chunk);
            let group = &doc_ids[start..end];
            start = end;
            // A chunk that is not resident is read, not loaded, so only a
            // hit is a cache event.
            let Some(resident) = self
                .cache
                .get_with_key(&SignatureChunkKey {
                    chunk: chunk as u32,
                })
                .await
            else {
                pending.extend_from_slice(group);
                continue;
            };
            metrics.record_index_cache_hit();
            let first_doc = chunk * chunk_docs;
            for &doc_id in group {
                let offset = doc_id as usize - first_doc;
                let Some(row_id) = resident.row_ids.get(offset) else {
                    return Err(Error::corrupt_file_named(
                        SIGNATURES_FILENAME,
                        format!("resident signature chunk {chunk} has no doc {doc_id}"),
                    ));
                };
                scorer.score(
                    *row_id,
                    &resident.signatures[offset * num_hashes..(offset + 1) * num_hashes],
                );
            }
        }
        if pending.is_empty() {
            return Ok(());
        }
        metrics.record_part_load();
        tracing::info!(
            target: TRACE_IO_EVENTS,
            r#type = IO_TYPE_LOAD_SCALAR_PART,
            index_type = "minhashlsh",
            part_id = "signatures",
        );
        let ranges = doc_id_ranges(pending.iter().copied());
        let batch = self.signatures.read_ranges(&ranges, None).await?;
        if batch.num_rows() != pending.len() {
            return Err(Error::corrupt_file_named(
                SIGNATURES_FILENAME,
                format!(
                    "scattered read returned {} rows for {} candidates",
                    batch.num_rows(),
                    pending.len()
                ),
            ));
        }
        let (row_ids, signatures) = signature_columns(&batch, num_hashes)?;
        for (row_id, signature) in row_ids
            .values()
            .iter()
            .zip(signatures.chunks_exact(num_hashes))
        {
            scorer.score(*row_id, signature);
        }
        Ok(())
    }

    /// Score the rest of level `shared` by walking the buckets and the
    /// signature table together in doc id order, both sequentially: cheaper
    /// than scattered reads once the level covers much of the segment.
    async fn score_level_dense(
        &self,
        scan: &mut BucketScan,
        shared: usize,
        floor: f32,
        scorer: &mut Scorer<'_>,
        metrics: &dyn MetricsCollector,
    ) -> Result<()> {
        let num_hashes = self.params.num_hashes as usize;
        let Some(mut pending) = self.next_in_level(scan, shared, metrics).await? else {
            return Ok(());
        };
        metrics.record_part_load();
        tracing::info!(
            target: TRACE_IO_EVENTS,
            r#type = IO_TYPE_LOAD_SCALAR_PART,
            index_type = "minhashlsh",
            part_id = "signatures",
        );
        let mut first_row = pending as usize;
        let mut stream = std::pin::pin!(scan_rows(
            self.signatures.clone(),
            first_row,
            self.num_docs,
            rows_per_batch(signature_row_bytes(num_hashes)),
        ));
        while let Some(batch) = stream.try_next().await? {
            let end_row = first_row + batch.num_rows();
            let (row_ids, signatures) = signature_columns(&batch, num_hashes)?;
            let mut scored = 0;
            while (pending as usize) < end_row {
                let offset = pending as usize - first_row;
                scorer.score(
                    row_ids.value(offset),
                    &signatures[offset * num_hashes..(offset + 1) * num_hashes],
                );
                scored += 1;
                match self.next_in_level(scan, shared, metrics).await? {
                    Some(doc_id) => pending = doc_id,
                    None => {
                        metrics.record_comparisons(scored);
                        return Ok(());
                    }
                }
            }
            metrics.record_comparisons(scored);
            if scorer.done(floor) {
                return Ok(());
            }
            first_row = end_row;
        }
        Ok(())
    }
}

/// One bucket of a search: the rows of one band key, walked in doc id order
/// through a bounded window.
struct BucketCursor {
    /// Absolute row of the head: the next doc id to yield.
    next: usize,
    /// Row after the bucket's last row.
    end: usize,
    /// Doc ids of the rows from `next`: up to two windows of pages, topped
    /// up whenever any bucket runs dry.
    window: std::vec::IntoIter<u32>,
}

/// The cursors of a search's buckets, merged in doc id order by
/// [`MinHashLshIndex::next_candidate`].
struct BucketScan {
    cursors: Vec<BucketCursor>,
}

impl BucketScan {
    /// The absolute row each cursor stands at.
    fn positions(&self) -> Vec<usize> {
        self.cursors.iter().map(|cursor| cursor.next).collect()
    }

    /// Move every cursor to `positions`, dropping the windows held.
    fn seek(&mut self, positions: &[usize]) {
        for (cursor, &row) in self.cursors.iter_mut().zip(positions) {
            cursor.next = row;
            cursor.window = Vec::new().into_iter();
        }
    }
}

/// Candidates sorted by the number of bands they share with the query, each
/// level holding at most `cap` doc ids in ascending order.
struct CandidateLevels {
    cap: usize,
    /// Indexed by shared band count; index 0 is unused.
    lists: Vec<Vec<u32>>,
    /// Candidates met per level, including those beyond `cap`.
    counts: Vec<u64>,
    /// Cursor positions right after a level reached `cap`, where a walk
    /// continuing that level starts.
    resume: Vec<Option<Vec<usize>>>,
    /// Whether the buckets were walked to the end, making `counts` exact.
    complete: bool,
}

impl CandidateLevels {
    fn new(num_bands: usize, cap: usize) -> Self {
        Self {
            cap,
            lists: vec![Vec::new(); num_bands + 1],
            counts: vec![0; num_bands + 1],
            resume: vec![None; num_bands + 1],
            complete: false,
        }
    }
}

/// Candidates waiting for their signatures, in refine order (by decreasing
/// shared bands, each level in doc id order) with the smallest distance each
/// can have, and how much to read at once (see [`MinHashLshIndex::read_queued`]).
struct Pending {
    candidates: VecDeque<(u32, f32)>,
    /// Candidates still to read before the first stop check.
    must: usize,
    /// Candidates of the next read once `must` is read.
    batch_rows: usize,
}

impl Pending {
    /// Queue the doc ids of one level, whose candidates are at least `floor`
    /// away from the query.
    fn queue(&mut self, floor: f32, doc_ids: Vec<u32>) {
        self.candidates
            .extend(doc_ids.into_iter().map(|doc_id| (doc_id, floor)));
    }

    /// The smallest distance the next candidate to read can have.
    fn floor(&self) -> Option<f32> {
        self.candidates.front().map(|&(_, floor)| floor)
    }
}

/// Scores candidate rows against the query signature into a bounded result.
struct Scorer<'a> {
    hits: TopHits,
    query: &'a [SignatureValue],
    mask: &'a RowAddrMask,
    remapper: Option<&'a dyn RowIdRemapper>,
}

impl Scorer<'_> {
    fn score(&mut self, row_id: u64, signature: &[SignatureValue]) {
        let row_id = match self.remapper {
            Some(remapper) => match remapper.remap_row_id(row_id) {
                Some(row_id) => row_id,
                None => return,
            },
            None => row_id,
        };
        if !self.mask.selected(row_id) {
            return;
        }
        self.hits.push(MinHashHit {
            row_id,
            distance: OrderedFloat(1.0 - estimate_jaccard(self.query, signature)),
        });
    }

    /// Whether no candidate at distance `floor` or more can improve the
    /// results held.
    fn done(&self, floor: f32) -> bool {
        self.hits.cutoff().is_some_and(|cutoff| cutoff <= floor)
    }
}

/// Coalesce ascending doc ids into contiguous row ranges.
fn doc_id_ranges(doc_ids: impl Iterator<Item = u32>) -> Vec<Range<usize>> {
    let mut ranges: Vec<Range<usize>> = Vec::new();
    for doc_id in doc_ids {
        let doc_id = doc_id as usize;
        match ranges.last_mut() {
            Some(last) if last.end == doc_id => last.end += 1,
            _ => ranges.push(doc_id..doc_id + 1),
        }
    }
    ranges
}

/// Extract the row id column and the flattened signature values of a batch
/// read from `signatures.lance`.
pub(super) fn signature_columns(
    batch: &RecordBatch,
    num_hashes: usize,
) -> Result<(&UInt64Array, &[SignatureValue])> {
    let corrupt = |message: String| Error::corrupt_file_named(SIGNATURES_FILENAME, message);
    let row_ids = batch
        .column_by_name(ROW_ID)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| corrupt(format!("missing UInt64 column {ROW_ID}")))?;
    let signatures = batch
        .column_by_name(SIGNATURE_COL)
        .and_then(|column| column.as_fixed_size_list_opt())
        .ok_or_else(|| corrupt(format!("missing FixedSizeList column {SIGNATURE_COL}")))?;
    if signatures.value_length() as usize != num_hashes {
        return Err(corrupt(format!(
            "signature width {} does not match num_hashes {num_hashes}",
            signatures.value_length()
        )));
    }
    let values = signatures
        .values()
        .as_primitive_opt::<UInt16Type>()
        .filter(|values| values.null_count() == 0)
        .ok_or_else(|| corrupt(format!("{SIGNATURE_COL} values are not non-null UInt16")))?;
    Ok((row_ids, values.values()))
}

/// Stream rows `first_row..num_rows` of `reader` as `rows_per_read` batches
/// with a few reads in flight.
pub(super) fn scan_rows(
    reader: Arc<dyn IndexReader>,
    first_row: usize,
    num_rows: usize,
    rows_per_read: usize,
) -> impl Stream<Item = Result<RecordBatch>> + Send {
    let ranges: Vec<Range<usize>> = (first_row..num_rows)
        .step_by(rows_per_read.max(1))
        .map(|start| start..(start + rows_per_read).min(num_rows))
        .collect();
    futures::stream::iter(ranges)
        .map(move |range| {
            let reader = reader.clone();
            async move { reader.read_range(range, None).await }
        })
        .buffered(READ_CONCURRENCY)
}

#[async_trait]
impl Index for MinHashLshIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    async fn prewarm(&self) -> Result<()> {
        // Load what the cache can hold, signature chunks first: a scattered
        // signature read costs a query far more than a page read.
        let capacity = self.cache.capacity_bytes();
        if capacity == Some(0) {
            return Ok(());
        }
        let mut budget = capacity.unwrap_or(usize::MAX);
        let num_chunks = self.num_docs.div_ceil(self.signature_chunk_docs);
        let mut chunks_loaded = 0;
        while chunks_loaded < num_chunks {
            let start = chunks_loaded * self.signature_chunk_docs;
            let docs = self.signature_chunk_docs.min(self.num_docs - start);
            let bytes = docs * signature_row_bytes(self.params.num_hashes as usize);
            if bytes > budget {
                break;
            }
            budget -= bytes;
            self.cache
                .get_or_insert_with_key(
                    SignatureChunkKey {
                        chunk: chunks_loaded as u32,
                    },
                    || self.load_signature_chunk(chunks_loaded),
                )
                .await?;
            chunks_loaded += 1;
        }

        let num_rows = self.bands.num_rows();
        let num_pages = self.page_max_keys.len();
        let page_bytes = self.page_rows * BAND_ROW_BYTES;
        let pages_to_load = (budget / page_bytes).min(num_pages);
        let pages_per_read = (RESIDENT_CHUNK_BYTES / page_bytes).max(1);
        for first_page in (0..pages_to_load).step_by(pages_per_read) {
            let last_page = (first_page + pages_per_read).min(pages_to_load);
            let rows = first_page * self.page_rows..(last_page * self.page_rows).min(num_rows);
            let batch = self.bands.read_range(rows, None).await?;
            for page in first_page..last_page {
                let range = self.page_range(page as u32);
                let page_batch =
                    batch.slice(range.start - first_page * self.page_rows, range.len());
                let band_page = Arc::new(BandPage::try_from_batch(&page_batch)?);
                self.cache
                    .insert_with_key(&BandPageKey { page: page as u32 }, band_page)
                    .await;
            }
        }

        if chunks_loaded < num_chunks || pages_to_load < num_pages {
            log::warn!(
                "MinHash LSH prewarm kept {chunks_loaded} of {num_chunks} signature chunks and {pages_to_load} of {num_pages} band pages: the index cache capacity is {} bytes",
                capacity.unwrap_or(0)
            );
        }
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "type": "MinHashLsh",
            "num_docs": self.num_docs,
            "num_pages": self.page_max_keys.len(),
            "num_hashes": self.params.num_hashes,
            "num_bands": self.params.num_bands,
            "shingle_size": self.params.shingle_size,
            "signature_version": SIGNATURE_VERSION,
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::MinHashLsh
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        // The signature table stores row ids, which no longer identify a
        // fragment once stable row ids are enabled; coverage is recorded in
        // the index metadata when a segment is committed.
        Err(Error::not_supported(
            "MinHash LSH indices do not recalculate fragment coverage from their files; the fragment bitmap of the index metadata is authoritative".to_string(),
        ))
    }
}

#[async_trait]
impl ScalarIndex for MinHashLshIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        _metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        Err(Error::not_supported(format!(
            "MinHash LSH index cannot evaluate scalar filter {query:?}; query it with a MinHash similarity search instead"
        )))
    }

    fn can_remap(&self) -> bool {
        true
    }

    async fn remap(
        &self,
        mapping: &RowAddrRemap,
        dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        let files = MinHashLshIndexBuilder::try_new(self.params.clone())?
            .rebuild_from(
                vec![self.signature_source(RowIdTransform::Remap(mapping))],
                None,
                dest_store,
            )
            .await?;
        self.created_index(files)
    }

    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        old_data_filter: Option<OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        let transform = match &old_data_filter {
            Some(filter) => RowIdTransform::Filter(filter),
            None => RowIdTransform::Keep,
        };
        let files = MinHashLshIndexBuilder::try_new(self.params.clone())?
            .rebuild_from(
                vec![self.signature_source(transform)],
                Some(new_data),
                dest_store,
            )
            .await?;
        self.created_index(files)
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(TrainingCriteria::new(TrainingOrdering::None).with_row_id())
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::MinHashLsh).with_params(&self.params))
    }
}

/// Merge segments into one new segment in `dest_store`, keeping only the rows
/// each segment's filter selects. Every segment must have been built with the
/// same parameters.
pub async fn merge_minhash_indices(
    sources: &[(&MinHashLshIndex, Option<&OldIndexDataFilter>)],
    dest_store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let Some((first, _)) = sources.first() else {
        return Err(Error::invalid_input(
            "merging MinHash LSH segments requires at least one segment".to_string(),
        ));
    };
    for (index, _) in sources {
        if index.params != first.params {
            return Err(Error::invalid_input(format!(
                "MinHash LSH segments were built with different parameters and cannot be merged: {:?} vs {:?}; rebuild the index with replace=true instead",
                first.params, index.params
            )));
        }
    }
    let total_docs: u64 = sources.iter().map(|(index, _)| index.num_docs as u64).sum();
    if total_docs > u32::MAX as u64 + 1 {
        return Err(Error::invalid_input(format!(
            "merging these MinHash LSH segments would produce {total_docs} documents but a segment holds at most {}; keep the segments separate",
            u32::MAX as u64 + 1
        )));
    }
    let signature_sources = sources
        .iter()
        .map(|(index, filter)| {
            index.signature_source(match filter {
                Some(filter) => RowIdTransform::Filter(filter),
                None => RowIdTransform::Keep,
            })
        })
        .collect();
    let files = MinHashLshIndexBuilder::try_new(first.params.clone())?
        .rebuild_from(signature_sources, None, dest_store)
        .await?;
    first.created_index(files)
}
