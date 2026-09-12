// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! One segment of a MinHash LSH index: opening its files, candidate lookup
//! through the page table, refining candidates against the signature table,
//! prewarming, and the rebuilds that update, remap or merge segments.

use super::*;

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

    /// Doc ids of the bucket `band_key` within this page.
    fn members(&self, band_key: u64) -> &[u32] {
        let start = self.keys.partition_point(|&key| key < band_key);
        let len = self.keys[start..]
            .iter()
            .take_while(|&&key| key == band_key)
            .count();
        &self.doc_ids[start..start + len]
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
        let signature_chunk_docs =
            (RESIDENT_CHUNK_BYTES / signature_row_bytes(params.num_hashes as usize)).max(1);
        Ok(Arc::new(Self {
            params,
            generator,
            bands,
            signatures,
            page_rows,
            page_max_keys,
            num_docs,
            signature_chunk_docs,
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
    /// among rows selected by `mask`, ordered by ascending distance.
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
        let candidates = self.collect_candidates(&query.band_keys, metrics).await?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        self.refine(candidates, &query.signature, limit, mask, metrics)
            .await
    }

    /// Union of the buckets of `band_keys`.
    ///
    /// A bucket is a run of equal keys in the sorted bands file, and the page
    /// table locates it without probing: its pages run from the first page
    /// whose max key reaches the key through the first page whose max key
    /// exceeds it. The pages of all buckets are fetched through the cache.
    async fn collect_candidates(
        &self,
        band_keys: &[u64],
        metrics: &dyn MetricsCollector,
    ) -> Result<RoaringBitmap> {
        let num_pages = self.page_max_keys.len();
        let buckets: Vec<(u64, Range<u32>)> = band_keys
            .iter()
            .map(|&key| {
                let first = self.page_max_keys.partition_point(|&max_key| max_key < key);
                let end = self
                    .page_max_keys
                    .partition_point(|&max_key| max_key <= key);
                (key, first as u32..(end + 1).min(num_pages) as u32)
            })
            .collect();
        metrics.record_comparisons(band_keys.len());
        let mut pages: Vec<u32> = buckets
            .iter()
            .flat_map(|(_, pages)| pages.clone())
            .collect();
        pages.sort_unstable();
        pages.dedup();
        let pages = self.load_pages(&pages, metrics).await?;
        let mut candidates = RoaringBitmap::new();
        for (key, bucket_pages) in &buckets {
            for page in bucket_pages.clone() {
                let page = pages.get(&page).ok_or_else(|| {
                    Error::internal(format!("band page {page} was requested but not loaded"))
                })?;
                candidates.extend(page.members(*key).iter().copied());
            }
        }
        if let Some(max_doc_id) = candidates.max()
            && max_doc_id as usize >= self.num_docs
        {
            return Err(Error::corrupt_file_named(
                BANDS_FILENAME,
                format!(
                    "posting references doc id {max_doc_id} but the segment has {} documents",
                    self.num_docs
                ),
            ));
        }
        Ok(candidates)
    }

    /// Fetch pages through the cache; the pages missing from the cache are
    /// read with scattered reads of at most [`IO_BATCH_BYTES`] each, so a
    /// bucket spanning many pages never materializes at once.
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

    /// Score every candidate against the query signature and keep the best
    /// `limit` rows selected by `mask`. Candidates in resident signature
    /// chunks are scored from memory; the rest are read with scattered reads,
    /// or with a sequential scan when they cover much of the segment.
    async fn refine(
        &self,
        mut candidates: RoaringBitmap,
        query: &[SignatureValue],
        limit: usize,
        mask: &RowAddrMask,
        metrics: &dyn MetricsCollector,
    ) -> Result<Vec<MinHashHit>> {
        let num_hashes = query.len();
        let mut hits = TopHits::new(limit);
        let mut score = |row_id: u64, signature: &[SignatureValue]| {
            let row_id = match &self.frag_reuse_index {
                Some(remapper) => match remapper.remap_row_id(row_id) {
                    Some(row_id) => row_id,
                    None => return,
                },
                None => row_id,
            };
            if !mask.selected(row_id) {
                return;
            }
            hits.push(MinHashHit {
                row_id,
                distance: 1.0 - estimate_jaccard(query, signature),
            });
        };
        metrics.record_comparisons(candidates.len() as usize);

        let chunk_docs = self.signature_chunk_docs;
        let (Some(first), Some(last)) = (candidates.min(), candidates.max()) else {
            return Ok(Vec::new());
        };
        for chunk in (first as usize / chunk_docs)..=(last as usize / chunk_docs) {
            let first_doc = chunk * chunk_docs;
            let chunk_range =
                first_doc as u32..=((first_doc + chunk_docs - 1).min(u32::MAX as usize)) as u32;
            if candidates.range_cardinality(chunk_range.clone()) == 0 {
                continue;
            }
            // A chunk that is not resident is read, not loaded, so only a
            // hit is a cache event.
            let Some(resident) = self
                .cache
                .get_with_key(&SignatureChunkKey {
                    chunk: chunk as u32,
                })
                .await
            else {
                continue;
            };
            metrics.record_index_cache_hit();
            for doc_id in candidates.range(chunk_range.clone()) {
                let offset = doc_id as usize - first_doc;
                let Some(row_id) = resident.row_ids.get(offset) else {
                    return Err(Error::corrupt_file_named(
                        SIGNATURES_FILENAME,
                        format!("resident signature chunk {chunk} has no doc {doc_id}"),
                    ));
                };
                score(
                    *row_id,
                    &resident.signatures[offset * num_hashes..(offset + 1) * num_hashes],
                );
            }
            candidates.remove_range(chunk_range);
        }
        if candidates.is_empty() {
            return Ok(hits.into_sorted());
        }

        let mut score_batch = |batch: &RecordBatch, keep: &mut dyn FnMut(usize) -> bool| {
            let (row_ids, signatures) = signature_columns(batch, num_hashes)?;
            for (index, (row_id, signature)) in row_ids
                .values()
                .iter()
                .zip(signatures.chunks_exact(num_hashes))
                .enumerate()
            {
                if keep(index) {
                    score(*row_id, signature);
                }
            }
            Ok::<_, Error>(())
        };
        metrics.record_part_load();
        tracing::info!(
            target: TRACE_IO_EVENTS,
            r#type = IO_TYPE_LOAD_SCALAR_PART,
            index_type = "minhashlsh",
            part_id = "signatures",
        );
        let rows_per_read = rows_per_batch(signature_row_bytes(num_hashes));
        if candidates.len().saturating_mul(100)
            <= (self.num_docs as u64).saturating_mul(SPARSE_REFINE_READ_PERCENT)
        {
            let mut doc_ids = candidates.iter();
            loop {
                let ranges = doc_id_ranges(doc_ids.by_ref().take(rows_per_read));
                if ranges.is_empty() {
                    break;
                }
                let expected_rows: usize = ranges.iter().map(|range| range.len()).sum();
                let batch = self.signatures.read_ranges(&ranges, None).await?;
                if batch.num_rows() != expected_rows {
                    return Err(Error::corrupt_file_named(
                        SIGNATURES_FILENAME,
                        format!(
                            "scattered read returned {} rows for {expected_rows} candidates",
                            batch.num_rows()
                        ),
                    ));
                }
                score_batch(&batch, &mut |_| true)?;
            }
        } else {
            let mut stream = std::pin::pin!(scan_rows(
                self.signatures.clone(),
                self.num_docs,
                rows_per_read
            ));
            let mut first_doc = 0usize;
            while let Some(batch) = stream.try_next().await? {
                score_batch(&batch, &mut |index| {
                    u32::try_from(first_doc + index).is_ok_and(|doc_id| candidates.contains(doc_id))
                })?;
                first_doc += batch.num_rows();
            }
        }
        Ok(hits.into_sorted())
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

/// Stream rows `0..num_rows` of `reader` as `rows_per_read` batches with a
/// few reads in flight.
pub(super) fn scan_rows(
    reader: Arc<dyn IndexReader>,
    num_rows: usize,
    rows_per_read: usize,
) -> impl Stream<Item = Result<RecordBatch>> + Send {
    let ranges: Vec<Range<usize>> = (0..num_rows)
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
