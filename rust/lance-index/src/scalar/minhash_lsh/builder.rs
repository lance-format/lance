// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Building the index files: signing text batches on the CPU pool, writing
//! `signatures.lance`, and sorting the (band key, doc id) records into
//! `bands.lance` with an external sort that spills to local temporary files.

use std::path::{Path, PathBuf};

use super::index::{scan_rows, signature_columns};
use super::*;

/// Signatures and band keys of one input batch, for rows that have tokens.
struct SignedBatch {
    row_ids: Vec<u64>,
    /// `row_ids.len() * num_hashes` values.
    signatures: Vec<SignatureValue>,
    /// `row_ids.len() * num_bands` keys.
    band_keys: Vec<u64>,
}

fn sign_batch(mut generator: SignatureGenerator, batch: RecordBatch) -> Result<SignedBatch> {
    let row_ids = batch
        .column_by_name(ROW_ID)
        .and_then(|column| column.as_primitive_opt::<UInt64Type>())
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "MinHash LSH training data must contain a non-null UInt64 column {ROW_ID}"
            ))
        })?;
    let values = batch.column_by_name(VALUE_COLUMN_NAME).ok_or_else(|| {
        Error::invalid_input(format!(
            "MinHash LSH training data must contain a column {VALUE_COLUMN_NAME}"
        ))
    })?;
    let num_rows = batch.num_rows();
    let mut signed = SignedBatch {
        row_ids: Vec::with_capacity(num_rows),
        signatures: Vec::with_capacity(num_rows * generator.num_hashes()),
        band_keys: Vec::with_capacity(num_rows * generator.num_bands()),
    };
    let mut signature = vec![SignatureValue::MAX; generator.num_hashes()];
    for (row, text) in text_values(values.as_ref())?.enumerate() {
        if let Some(text) = text
            && generator.signature(text, &mut signature)
        {
            signed.row_ids.push(row_ids.value(row));
            signed.signatures.extend_from_slice(&signature);
            generator.band_keys(&signature, &mut signed.band_keys);
        }
    }
    Ok(signed)
}

/// Signed batches computed from text batches, `num_cpus` batches in flight.
fn text_signed_batches(
    data: SendableRecordBatchStream,
    generator: SignatureGenerator,
) -> impl Stream<Item = Result<SignedBatch>> + Send {
    data.map(move |batch| {
        let generator = generator.clone();
        async move {
            let batch = batch?;
            spawn_cpu(move || sign_batch(generator, batch)).await
        }
    })
    .buffered(get_num_compute_intensive_cpus())
}

/// How the row ids of an existing signature table are carried into a
/// rebuilt segment.
pub(super) enum RowIdTransform<'a> {
    /// Keep every row with its row id.
    Keep,
    /// Keep only the rows the filter selects.
    Filter(&'a OldIndexDataFilter),
    /// Rewrite row ids through the mapping, dropping rows it deletes.
    Remap(&'a RowAddrRemap),
}

impl RowIdTransform<'_> {
    /// The row id each input row keeps, or `None` for rows to drop.
    ///
    /// Stored row ids predate any deferred compaction the segment was opened
    /// with, so `frag_reuse_index` first brings them into the current address
    /// space, which is the space the filter or mapping is expressed in.
    fn apply(
        &self,
        row_ids: &UInt64Array,
        frag_reuse_index: Option<&dyn RowIdRemapper>,
    ) -> Vec<Option<u64>> {
        let mut row_ids: Vec<Option<u64>> = row_ids
            .values()
            .iter()
            .map(|&row_id| match frag_reuse_index {
                Some(remapper) => remapper.remap_row_id(row_id),
                None => Some(row_id),
            })
            .collect();
        match self {
            Self::Keep => {}
            Self::Filter(filter) => {
                let keep = filter.filter_row_ids(&UInt64Array::from(row_ids.clone()));
                for (row_id, keep) in row_ids.iter_mut().zip(keep.iter()) {
                    if !keep.unwrap_or(false) {
                        *row_id = None;
                    }
                }
            }
            Self::Remap(mapping) => mapping.remap_in_place(&mut row_ids),
        }
        row_ids
    }
}

/// An existing signature table whose surviving rows are carried into a
/// rebuilt segment.
pub(super) struct SignatureSource<'a> {
    pub reader: Arc<dyn IndexReader>,
    pub num_docs: usize,
    /// The deferred compactions the segment was opened with; see
    /// [`RowIdTransform::apply`].
    pub frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
    pub transform: RowIdTransform<'a>,
}

type SignedBatchStream<'a> = Pin<Box<dyn Stream<Item = Result<SignedBatch>> + Send + 'a>>;

impl SignatureSource<'_> {
    fn signed_batches(&self, generator: SignatureGenerator) -> SignedBatchStream<'_> {
        let num_hashes = generator.num_hashes();
        let batches = scan_rows(
            self.reader.clone(),
            self.num_docs,
            rows_per_batch(signature_row_bytes(num_hashes)),
        );
        let transform = &self.transform;
        let frag_reuse_index = self.frag_reuse_index.as_deref();
        let stream = batches
            .map(move |batch| {
                let generator = generator.clone();
                // Row id filtering needs the transform, which cannot move into
                // the CPU task, so it runs here; band keys are computed there.
                let prepared = batch.and_then(|batch| {
                    let (row_ids, _) = signature_columns(&batch, num_hashes)?;
                    let kept = transform.apply(row_ids, frag_reuse_index);
                    Ok((batch, kept))
                });
                async move {
                    let (batch, kept) = prepared?;
                    spawn_cpu(move || resign_batch(generator, batch, kept)).await
                }
            })
            .buffered(get_num_compute_intensive_cpus());
        Box::pin(stream)
    }
}

/// Build a signed batch from stored signatures, keeping the rows whose entry
/// in `row_ids` is `Some` (the row id to store).
fn resign_batch(
    generator: SignatureGenerator,
    batch: RecordBatch,
    row_ids: Vec<Option<u64>>,
) -> Result<SignedBatch> {
    let num_hashes = generator.num_hashes();
    let (_, signatures) = signature_columns(&batch, num_hashes)?;
    let kept = row_ids.iter().filter(|row_id| row_id.is_some()).count();
    let mut signed = SignedBatch {
        row_ids: Vec::with_capacity(kept),
        signatures: Vec::with_capacity(kept * num_hashes),
        band_keys: Vec::with_capacity(kept * generator.num_bands()),
    };
    for (row_id, signature) in row_ids.iter().zip(signatures.chunks_exact(num_hashes)) {
        let Some(row_id) = row_id else {
            continue;
        };
        signed.row_ids.push(*row_id);
        signed.signatures.extend_from_slice(signature);
        generator.band_keys(signature, &mut signed.band_keys);
    }
    Ok(signed)
}

fn signatures_batch(
    schema: &SchemaRef,
    row_ids: Vec<u64>,
    signatures: Vec<SignatureValue>,
    num_hashes: i32,
) -> Result<RecordBatch> {
    let values: ArrayRef = Arc::new(UInt16Array::from(signatures));
    let signatures = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::UInt16, false)),
        num_hashes,
        values,
        None,
    )?;
    Ok(RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt64Array::from(row_ids)) as ArrayRef,
            Arc::new(signatures) as ArrayRef,
        ],
    )?)
}

/// Streams ascending (band key, doc id) rows into `bands.lance` and records
/// the page table as rows are written.
struct BandsWriter {
    writer: Box<dyn IndexWriter>,
    page_rows: usize,
    rows_written: usize,
    /// Key of the most recently written row, for the trailing partial page.
    last_key: Option<u64>,
    page_max_keys: Vec<u64>,
}

impl BandsWriter {
    /// Write one batch of ascending records and extend the page table.
    async fn write_batch(&mut self, keys: Vec<u64>, doc_ids: Vec<u32>) -> Result<()> {
        // Only the rows that end a page matter; this runs on the serial
        // write path, so step to them instead of testing every row.
        let mut page_end = self.page_rows - self.rows_written % self.page_rows;
        while page_end <= keys.len() {
            self.page_max_keys.push(keys[page_end - 1]);
            page_end += self.page_rows;
        }
        self.rows_written += keys.len();
        if let Some(last) = keys.last() {
            self.last_key = Some(*last);
        }
        let batch = RecordBatch::try_new(
            BANDS_SCHEMA.clone(),
            vec![
                Arc::new(UInt64Array::from(keys)) as ArrayRef,
                Arc::new(UInt32Array::from(doc_ids)) as ArrayRef,
            ],
        )?;
        self.writer.write_record_batch(batch).await?;
        Ok(())
    }

    async fn finish(mut self, params: &MinHashLshIndexParams, num_docs: u64) -> Result<IndexFile> {
        if !self.rows_written.is_multiple_of(self.page_rows)
            && let Some(last_key) = self.last_key
        {
            self.page_max_keys.push(last_key);
        }
        let mut page_table = Vec::with_capacity(self.page_max_keys.len() * 8);
        for key in &self.page_max_keys {
            page_table.extend_from_slice(&key.to_le_bytes());
        }
        let page_table_buffer = self
            .writer
            .add_global_buffer(Bytes::from(page_table))
            .await?;
        self.writer
            .finish_with_metadata(HashMap::from([
                (DETAILS_META_KEY.to_string(), params.details_hex()?),
                (
                    INDEX_VERSION_META_KEY.to_string(),
                    MINHASH_LSH_INDEX_VERSION.to_string(),
                ),
                (PAGE_ROWS_META_KEY.to_string(), self.page_rows.to_string()),
                (
                    PAGE_TABLE_BUFFER_META_KEY.to_string(),
                    page_table_buffer.to_string(),
                ),
                (NUM_DOCS_META_KEY.to_string(), num_docs.to_string()),
            ]))
            .await
    }
}

/// The temporary directory holding the spill files of one build, removed
/// when the last run referencing it is dropped. Every spill is accounted
/// against the configured limit before it is written, so a build that
/// cannot fit fails with a clear error.
struct SpillDir {
    dir: tempfile::TempDir,
    limit_bytes: u64,
    written_bytes: AtomicU64,
}

impl SpillDir {
    fn create(limit_bytes: u64) -> Result<Self> {
        let base = std::env::temp_dir();
        let dir = tempfile::Builder::new()
            .prefix("lance-minhash-lsh-")
            .tempdir_in(&base)
            .map_err(|err| {
                Error::io(format!(
                    "cannot create the MinHash LSH spill directory in {}: {err}",
                    base.display()
                ))
            })?;
        Ok(Self {
            dir,
            limit_bytes,
            written_bytes: AtomicU64::new(0),
        })
    }

    /// Account `bytes` of a new run against the limit.
    fn reserve(&self, bytes: u64) -> Result<()> {
        let written = self.written_bytes.fetch_add(bytes, Ordering::SeqCst);
        if written + bytes > self.limit_bytes {
            return Err(Error::io(format!(
                "MinHash LSH build needs {} bytes of temporary disk in {} ({written} already written) but {SPILL_LIMIT_ENV} limits it to {}",
                written + bytes,
                self.dir.path().display(),
                self.limit_bytes
            )));
        }
        Ok(())
    }

    fn run_path(&self, run: usize) -> PathBuf {
        self.dir.path().join(format!("run-{run}.bin"))
    }
}

/// A sorted run of (band key, doc id) records in a spill file, with the row
/// at which each key-range partition starts (`SPILL_PARTITIONS + 1` entries,
/// the last one being the row count).
struct SpilledRun {
    dir: Arc<SpillDir>,
    run: usize,
    partition_starts: Vec<usize>,
}

impl SpilledRun {
    /// Rows of this run that hold `partitions`.
    fn rows(&self, partitions: &Range<usize>) -> Range<usize> {
        self.partition_starts[partitions.start]..self.partition_starts[partitions.end]
    }
}

/// Write sorted records to `path` as little-endian (key, doc id) pairs.
fn write_spill_records(path: &Path, records: &[(u64, u32)]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    let records_per_write = rows_per_batch(BAND_ROW_BYTES);
    let mut bytes = Vec::with_capacity(records_per_write.min(records.len()) * BAND_ROW_BYTES);
    for chunk in records.chunks(records_per_write) {
        bytes.clear();
        for (key, doc_id) in chunk {
            bytes.extend_from_slice(&key.to_le_bytes());
            bytes.extend_from_slice(&doc_id.to_le_bytes());
        }
        file.write_all(&bytes)?;
    }
    Ok(())
}

/// Read the records of `rows` back from a spill file.
fn read_spill_records(path: &Path, rows: Range<usize>) -> std::io::Result<Vec<(u64, u32)>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path)?;
    file.seek(SeekFrom::Start((rows.start * BAND_ROW_BYTES) as u64))?;
    let mut bytes = vec![0u8; rows.len() * BAND_ROW_BYTES];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(BAND_ROW_BYTES)
        .map(|record| {
            let (key, doc_id) = record.split_at(8);
            (
                u64::from_le_bytes(key.try_into().expect("8-byte key")),
                u32::from_le_bytes(doc_id.try_into().expect("4-byte doc id")),
            )
        })
        .collect())
}

/// Key-range partition of a record: its band id, then the high bits of the
/// band hash, so partitions are contiguous in key order and every band's
/// records spread evenly over `partitions_per_band` of them.
fn spill_partition(key: u64, partitions_per_band: usize) -> usize {
    let band = (key >> 56) as usize;
    let hash = key & 0x00FF_FFFF_FFFF_FFFF;
    band * partitions_per_band + ((hash as u128 * partitions_per_band as u128) >> 56) as usize
}

/// Sort one run and write it to a spill file.
async fn spill_run(
    dir: Arc<SpillDir>,
    run: usize,
    records: Vec<(u64, u32)>,
    partitions_per_band: usize,
) -> Result<SpilledRun> {
    let (records, partition_starts) = spawn_cpu(move || {
        let mut records = records;
        records.par_sort_unstable();
        let partition_starts: Vec<usize> = (0..=SPILL_PARTITIONS)
            .map(|partition| {
                records.partition_point(|record| {
                    spill_partition(record.0, partitions_per_band) < partition
                })
            })
            .collect();
        Ok::<_, Error>((records, partition_starts))
    })
    .await?;
    dir.reserve((records.len() * BAND_ROW_BYTES) as u64)?;
    let path = dir.run_path(run);
    // Blocking file IO stays off the runtime workers.
    tokio::task::spawn_blocking(move || write_spill_records(&path, &records))
        .await
        .map_err(|err| Error::internal(format!("spill write task failed: {err}")))?
        .map_err(|err| Error::io(format!("cannot write MinHash LSH spill file: {err}")))?;
    Ok(SpilledRun {
        dir,
        run,
        partition_starts,
    })
}

/// Collects (band key, doc id) records into sorted runs: in memory until
/// `run_records` are held, then spilled to disk while signing continues.
struct RunSorter {
    run: Vec<(u64, u32)>,
    run_records: usize,
    spill_limit_bytes: u64,
    partitions_per_band: usize,
    /// Created by the first spill and shared with every run and spill task.
    spill_dir: Option<Arc<SpillDir>>,
    /// Runs being sorted and written, oldest first; at most
    /// [`MAX_INFLIGHT_SPILLS`] so the backlog of full runs stays bounded.
    spills: FuturesOrdered<tokio::task::JoinHandle<Result<SpilledRun>>>,
    spilled: Vec<SpilledRun>,
}

impl RunSorter {
    async fn push(&mut self, keys: &[u64], doc_id: u32) -> Result<()> {
        self.run.extend(keys.iter().map(|&key| (key, doc_id)));
        if self.run.len() < self.run_records {
            return Ok(());
        }
        let records = std::mem::replace(&mut self.run, Vec::with_capacity(self.run_records));
        self.spill(records)?;
        if self.spills.len() >= MAX_INFLIGHT_SPILLS
            && let Some(spilled) = self.spills.next().await
        {
            self.spilled.push(joined_spill(spilled)?);
        }
        Ok(())
    }

    fn spill(&mut self, records: Vec<(u64, u32)>) -> Result<()> {
        let dir = match &self.spill_dir {
            Some(dir) => dir.clone(),
            None => self
                .spill_dir
                .insert(Arc::new(SpillDir::create(self.spill_limit_bytes)?))
                .clone(),
        };
        let run = self.spilled.len() + self.spills.len();
        self.spills.push_back(tokio::spawn(spill_run(
            dir,
            run,
            records,
            self.partitions_per_band,
        )));
        Ok(())
    }

    /// Sort the last run and wait for every spill.
    async fn finish(mut self) -> Result<SortedRuns> {
        if self.spill_dir.is_none() {
            let mut run = self.run;
            return spawn_cpu(move || {
                run.par_sort_unstable();
                Ok(SortedRuns::Resident(run))
            })
            .await;
        }
        if !self.run.is_empty() {
            let run = std::mem::take(&mut self.run);
            self.spill(run)?;
        }
        while let Some(spilled) = self.spills.next().await {
            self.spilled.push(joined_spill(spilled)?);
        }
        Ok(SortedRuns::Spilled(self.spilled.into()))
    }
}

/// Unwrap a finished spill task, surfacing a panicked or cancelled task as an error.
fn joined_spill(
    spilled: std::result::Result<Result<SpilledRun>, tokio::task::JoinError>,
) -> Result<SpilledRun> {
    spilled.map_err(|err| Error::internal(format!("spill task failed: {err}")))?
}

/// Sorted (band key, doc id) records ready to be written.
enum SortedRuns {
    /// Everything fit in one in-memory run.
    Resident(Vec<(u64, u32)>),
    /// Several runs spilled to temporary files, merged while writing.
    Spilled(Arc<[SpilledRun]>),
}

/// Split the partitions into contiguous groups of about `group_records`
/// records each (at least one partition per group).
fn merge_groups(runs: &[SpilledRun], group_records: usize) -> Vec<Range<usize>> {
    let mut groups = Vec::new();
    let mut start = 0;
    let mut records = 0usize;
    for partition in 0..SPILL_PARTITIONS {
        records += runs
            .iter()
            .map(|run| run.rows(&(partition..partition + 1)).len())
            .sum::<usize>();
        if records >= group_records {
            groups.push(start..partition + 1);
            start = partition + 1;
            records = 0;
        }
    }
    if start < SPILL_PARTITIONS {
        groups.push(start..SPILL_PARTITIONS);
    }
    groups
}

/// Gather the records of `partitions` from every run, sort them and split
/// them into the key and doc id columns.
async fn merge_group(
    runs: &[SpilledRun],
    partitions: Range<usize>,
) -> Result<(Vec<u64>, Vec<u32>)> {
    let mut records: Vec<(u64, u32)> =
        Vec::with_capacity(runs.iter().map(|run| run.rows(&partitions).len()).sum());
    let reads: Vec<_> = runs
        .iter()
        .filter_map(|run| {
            let rows = run.rows(&partitions);
            if rows.is_empty() {
                return None;
            }
            let path = run.dir.run_path(run.run);
            // Spill reads are blocking file IO, kept off the runtime workers.
            Some(async move {
                tokio::task::spawn_blocking(move || read_spill_records(&path, rows))
                    .await
                    .map_err(|err| Error::internal(format!("spill read task failed: {err}")))?
                    .map_err(|err| Error::io(format!("cannot read MinHash LSH spill file: {err}")))
            })
        })
        .collect();
    let mut chunks = futures::stream::iter(reads).buffered(READ_CONCURRENCY);
    while let Some(chunk) = chunks.try_next().await? {
        records.extend(chunk);
    }
    spawn_cpu(move || {
        records.sort_unstable();
        Ok::<_, Error>(records.into_iter().unzip())
    })
    .await
}

/// Builds the two index files from a stream of `value` (text) and `_rowid`
/// batches.
pub struct MinHashLshIndexBuilder {
    params: MinHashLshIndexParams,
    page_rows: usize,
    /// Maximum records held in memory before a sort run is spilled.
    sort_run_records: usize,
    /// Records merged as one group while writing the bands file.
    merge_group_records: usize,
    /// Temporary disk space one build may use for its spill files.
    spill_limit_bytes: u64,
}

impl MinHashLshIndexBuilder {
    /// A builder with the memory budget of `LANCE_MEM_POOL_SIZE` (2 GiB by
    /// default) and the spill limit of `LANCE_MAX_TEMP_DIRECTORY_SIZE`
    /// (100 GiB by default).
    pub fn try_new(params: MinHashLshIndexParams) -> Result<Self> {
        params.validate()?;
        // The runs (the one being filled plus the in-flight spills) get three
        // fifths of the memory budget, the merge groups the rest.
        let memory_bytes = env_bytes(SORT_MEMORY_ENV).unwrap_or(DEFAULT_SORT_MEMORY_BYTES) as usize;
        let record_bytes = std::mem::size_of::<(u64, u32)>();
        let sort_run_records = memory_bytes * 3 / 5 / (MAX_INFLIGHT_SPILLS + 1) / record_bytes;
        let merge_group_records = memory_bytes * 2 / 5 / MERGE_GROUPS_IN_FLIGHT / record_bytes;
        if sort_run_records == 0 || merge_group_records == 0 {
            return Err(Error::invalid_input(format!(
                "MinHash LSH sort memory budget of {memory_bytes} bytes ({SORT_MEMORY_ENV}) is too small"
            )));
        }
        Ok(Self {
            params,
            page_rows: DEFAULT_PAGE_ROWS,
            sort_run_records,
            merge_group_records,
            spill_limit_bytes: env_bytes(SPILL_LIMIT_ENV).unwrap_or(DEFAULT_SPILL_LIMIT_BYTES),
        })
    }

    pub fn params(&self) -> &MinHashLshIndexParams {
        &self.params
    }

    /// Rows per logical page of `bands.lance`; exposed so tests can force
    /// multi-page layouts with small inputs.
    pub fn with_page_rows(mut self, page_rows: usize) -> Result<Self> {
        if page_rows == 0 {
            return Err(Error::invalid_input(
                "MinHash LSH page_rows must be positive".to_string(),
            ));
        }
        self.page_rows = page_rows;
        Ok(self)
    }

    /// Records per in-memory sort run; exposed so tests can force spilling.
    pub fn with_sort_run_records(mut self, sort_run_records: usize) -> Result<Self> {
        if sort_run_records == 0 {
            return Err(Error::invalid_input(
                "MinHash LSH sort_run_records must be positive".to_string(),
            ));
        }
        self.sort_run_records = sort_run_records;
        Ok(self)
    }

    /// Bound the temporary disk space of the build; a build that would
    /// exceed it fails before writing the spill that crosses the limit.
    pub fn with_spill_limit_bytes(mut self, bytes: u64) -> Self {
        self.spill_limit_bytes = bytes;
        self
    }

    /// Build the index files from a stream of `value` (text) and `_rowid`
    /// batches.
    pub async fn train(
        &self,
        data: SendableRecordBatchStream,
        store: &dyn IndexStore,
    ) -> Result<Vec<IndexFile>> {
        let generator = SignatureGenerator::try_new(&self.params)?;
        self.train_signed(text_signed_batches(data, generator), store)
            .await
    }

    /// Build the index files from the surviving rows of existing signature
    /// tables plus optional new text batches, without tokenizing the existing
    /// rows again. Used by segment merges, updates and row id remaps.
    pub(super) async fn rebuild_from(
        &self,
        sources: Vec<SignatureSource<'_>>,
        new_data: Option<SendableRecordBatchStream>,
        store: &dyn IndexStore,
    ) -> Result<Vec<IndexFile>> {
        let generator = SignatureGenerator::try_new(&self.params)?;
        let mut streams: Vec<SignedBatchStream<'_>> = sources
            .iter()
            .map(|source| source.signed_batches(generator.clone()))
            .collect();
        if let Some(new_data) = new_data {
            streams.push(Box::pin(text_signed_batches(new_data, generator)));
        }
        self.train_signed(futures::stream::iter(streams).flatten(), store)
            .await
    }

    /// Write `signatures.lance` and `bands.lance` from signed batches,
    /// assigning doc ids in arrival order.
    async fn train_signed(
        &self,
        signed_batches: impl Stream<Item = Result<SignedBatch>> + Send,
        store: &dyn IndexStore,
    ) -> Result<Vec<IndexFile>> {
        let num_hashes = self.params.num_hashes as usize;
        let num_bands = self.params.num_bands as usize;
        let schema = signatures_schema(self.params.num_hashes as i32);
        let signature_write_rows = rows_per_batch(signature_row_bytes(num_hashes));
        let mut signatures_writer = store
            .new_index_file(SIGNATURES_FILENAME, schema.clone())
            .await?;

        // The signing stream (CPU-pool tasks behind `buffered`) only makes
        // progress while it is polled, and the consumer below also waits on
        // spills and on file writes. A driver keeps polling the stream into
        // a bounded queue and the signature file is written by its own task,
        // so neither wait idles the CPU pool.
        let (signed_tx, mut signed_rx) = tokio::sync::mpsc::channel(SIGNED_BATCH_QUEUE);
        let driver = async move {
            let mut signed_batches = std::pin::pin!(signed_batches);
            while let Some(signed) = signed_batches.next().await
                && signed_tx.send(signed).await.is_ok()
            {}
        };
        let (batch_tx, mut batch_rx) = tokio::sync::mpsc::channel::<RecordBatch>(4);
        let writer = tokio::spawn(async move {
            while let Some(batch) = batch_rx.recv().await {
                signatures_writer.write_record_batch(batch).await?;
            }
            Ok::<_, Error>(signatures_writer)
        });
        let schema = &schema;
        let consumer = async move {
            let mut sorter = RunSorter {
                run: Vec::new(),
                run_records: self.sort_run_records,
                spill_limit_bytes: self.spill_limit_bytes,
                partitions_per_band: SPILL_PARTITIONS / num_bands,
                spill_dir: None,
                spills: FuturesOrdered::new(),
                spilled: Vec::new(),
            };
            let mut num_docs: u64 = 0;
            // Signed batches are coalesced into larger write batches.
            let mut row_ids: Vec<u64> = Vec::new();
            let mut signatures: Vec<SignatureValue> = Vec::new();
            let send = |batch: RecordBatch| async {
                batch_tx.send(batch).await.map_err(|_| {
                    Error::internal("signature writer task stopped before the build finished")
                })
            };
            while let Some(signed) = signed_rx.recv().await {
                let signed = signed?;
                for keys in signed.band_keys.chunks_exact(num_bands) {
                    sorter.push(keys, checked_doc_id(num_docs)?).await?;
                    num_docs += 1;
                }
                row_ids.extend_from_slice(&signed.row_ids);
                signatures.extend_from_slice(&signed.signatures);
                if row_ids.len() >= signature_write_rows {
                    send(signatures_batch(
                        schema,
                        std::mem::take(&mut row_ids),
                        std::mem::take(&mut signatures),
                        num_hashes as i32,
                    )?)
                    .await?;
                }
            }
            if !row_ids.is_empty() {
                send(signatures_batch(
                    schema,
                    row_ids,
                    signatures,
                    num_hashes as i32,
                )?)
                .await?;
            }
            Ok::<_, Error>((sorter, num_docs))
        };
        let ((), consumed) = futures::join!(driver, consumer);
        // The writer's own error explains a consumer that failed to send.
        let mut signatures_writer = writer
            .await
            .map_err(|err| Error::internal(format!("signature writer task failed: {err}")))??;
        let (sorter, num_docs) = consumed?;

        let metadata = HashMap::from([
            (DETAILS_META_KEY.to_string(), self.params.details_hex()?),
            (
                INDEX_VERSION_META_KEY.to_string(),
                MINHASH_LSH_INDEX_VERSION.to_string(),
            ),
        ]);
        let (signatures_file, runs) = futures::try_join!(
            signatures_writer.finish_with_metadata(metadata),
            sorter.finish()
        )?;
        let bands_file = self.write_bands(store, &runs, num_docs).await?;
        Ok(vec![signatures_file, bands_file])
    }

    /// Write the sorted records to `bands.lance`; spilled runs are merged one
    /// key-range group at a time, several groups in flight as independent
    /// tasks, written in order.
    async fn write_bands(
        &self,
        store: &dyn IndexStore,
        runs: &SortedRuns,
        num_docs: u64,
    ) -> Result<IndexFile> {
        let mut bands = BandsWriter {
            writer: store
                .new_index_file(BANDS_FILENAME, BANDS_SCHEMA.clone())
                .await?,
            page_rows: self.page_rows,
            rows_written: 0,
            last_key: None,
            page_max_keys: Vec::new(),
        };
        match runs {
            SortedRuns::Resident(records) => {
                for chunk in records.chunks(rows_per_batch(BAND_ROW_BYTES)) {
                    let (keys, doc_ids) = chunk.iter().copied().unzip();
                    bands.write_batch(keys, doc_ids).await?;
                }
            }
            SortedRuns::Spilled(runs) => {
                // Each group is its own task so gathering and sorting proceed
                // while this loop waits on the writer.
                let mut merged =
                    futures::stream::iter(merge_groups(runs, self.merge_group_records))
                        .map(|group| {
                            let runs = runs.clone();
                            tokio::spawn(async move { merge_group(&runs, group).await })
                        })
                        .buffered(
                            (get_num_compute_intensive_cpus() / 4).clamp(1, MERGE_GROUPS_IN_FLIGHT),
                        );
                while let Some(joined) = merged.next().await {
                    let (keys, doc_ids) = joined
                        .map_err(|err| Error::internal(format!("merge task failed: {err}")))??;
                    if !keys.is_empty() {
                        bands.write_batch(keys, doc_ids).await?;
                    }
                }
            }
        }
        bands.finish(&self.params, num_docs).await
    }
}

/// A byte count from the environment, if set and valid.
fn env_bytes(name: &str) -> Option<u64> {
    let value = std::env::var(name).ok()?;
    match value.parse::<u64>() {
        Ok(bytes) => Some(bytes),
        Err(err) => {
            log::warn!("ignoring {name}={value}: {err}");
            None
        }
    }
}
