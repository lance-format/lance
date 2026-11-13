// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Build-only external merging. Temporary runs have lexically ordered posting
//! rows and a private resource manifest. Neither changes the query format.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::array::AsArray;
use arrow_array::{RecordBatch, UInt32Array, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use fst::Streamer;
use futures::future::join_all;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};

use super::super::{
    DocSet, IndexFile, IndexStore, InnerBuilder, InvertedIndexBuilder, InvertedListFormatVersion,
    LENGTH_COL, MAX_SCORE_COL, NUM_TOKEN_COL, PartitionWriteTarget, PositionRecorder, PostingList,
    PostingListBatchBuilder, PostingListBuilder, TokenDictionary, TokenMap, TokenSet,
    TokenSetFormat, WorkerOutput, inverted_list_schema_for_version_with_block_size,
};
use super::Stop;
use crate::scalar::IndexReader;

const RETAINED: &str = "retained_bytes";
const SCRATCH: &str = "scratch_bytes";
const MAX_FAN_IN: usize = 4;
const METADATA_ROWS: usize = 128;
// Merge scans are sequential and do not share query-time read amplification
// goals. Amortize remote latency for small terms while the byte quota below
// remains the primary bound; cap rows too for zero-sized posting payloads.
const POSTING_READ_ROWS: usize = 4096;

#[derive(Clone, Copy, Default)]
struct Bounds {
    retained: u64,
    scratch: u64,
}

impl Bounds {
    fn read_bytes(self) -> Result<u64> {
        self.retained
            .checked_mul(4)
            .and_then(|n| n.checked_add(256))
            .ok_or_else(|| Error::invalid_input("FTS merge read bound overflow"))
    }

    fn work_bytes(self) -> Result<u64> {
        // Remapped doc ids, impact frontiers, Arrow builders and encoded output
        // can all coexist. Charge per term, not per complete output partition.
        self.retained
            .checked_mul(32)
            .and_then(|n| n.checked_add(self.scratch))
            .ok_or_else(|| Error::invalid_input("FTS merge term bound overflow"))
    }
}

#[derive(Clone)]
struct ManifestCursor {
    reader: Arc<dyn IndexReader>,
    batch: Option<RecordBatch>,
}

impl ManifestCursor {
    fn new(reader: Arc<dyn IndexReader>) -> Self {
        Self {
            reader,
            batch: None,
        }
    }

    async fn at(&mut self, row: usize) -> Result<Bounds> {
        if row >= self.reader.num_rows() {
            return Err(Error::index("FTS run manifest row is out of bounds"));
        }
        // Fixed-width bounds are admitted with dictionary metadata. Keep a
        // shared decoded copy across preflight and posting reads, avoiding a
        // separate remote request for every small manifest window.
        if self.batch.is_none() {
            let batch = self
                .reader
                .read_range(0..self.reader.num_rows(), None)
                .await?;
            if batch.num_rows() != self.reader.num_rows() {
                return Err(Error::index("FTS run manifest row count differs"));
            }
            self.batch = Some(batch);
        }
        let batch = self
            .batch
            .as_ref()
            .ok_or_else(|| Error::internal("missing FTS run manifest batch"))?;
        let index = row;
        Ok(Bounds {
            retained: batch
                .column_by_name(RETAINED)
                .ok_or_else(|| Error::index(format!("missing FTS column {RETAINED}")))?
                .as_primitive::<arrow_array::types::UInt64Type>()
                .value(index),
            scratch: batch
                .column_by_name(SCRATCH)
                .ok_or_else(|| Error::index(format!("missing FTS column {SCRATCH}")))?
                .as_primitive::<arrow_array::types::UInt64Type>()
                .value(index),
        })
    }
}

struct Run {
    id: u64,
    size: u64,
    terms: usize,
    docs: usize,
    rank: usize,
    dictionary_bytes: u64,
}

fn header(reader: &dyn IndexReader, key: &str) -> Result<u64> {
    reader
        .schema()
        .metadata
        .get(key)
        .ok_or_else(|| Error::index(format!("FTS run manifest is missing {key}")))?
        .parse()
        .map_err(|error| Error::index(format!("invalid FTS run manifest {key}: {error}")))
}

struct Term {
    token: String,
    sources: Vec<(usize, usize)>,
}

struct Terms<'a> {
    streams: Vec<fst::map::Stream<'a>>,
    current: Vec<Option<(Vec<u8>, usize)>>,
    next_rows: Vec<usize>,
}

impl<'a> Terms<'a> {
    fn new(tokens: &'a [TokenSet]) -> Result<Self> {
        let streams = tokens
            .iter()
            .map(|tokens| match &tokens.tokens {
                TokenMap::Fst(map) => Ok(map.stream()),
                _ => Err(Error::internal("loaded FTS run dictionary is not an FST")),
            })
            .collect::<Result<Vec<_>>>()?;
        let len = streams.len();
        let mut terms = Self {
            streams,
            current: vec![None; len],
            next_rows: vec![0; len],
        };
        for index in 0..len {
            terms.advance(index)?;
        }
        Ok(terms)
    }

    fn advance(&mut self, source: usize) -> Result<()> {
        self.current[source] = match self.streams[source].next() {
            Some((term, id)) => {
                if id != self.next_rows[source] as u64 {
                    return Err(Error::index(
                        "FTS temporary posting rows are not in lexical order",
                    ));
                }
                self.next_rows[source] += 1;
                Some((term.to_vec(), id as usize))
            }
            None => None,
        };
        Ok(())
    }

    fn next(&mut self) -> Result<Option<Term>> {
        let Some(term) = self
            .current
            .iter()
            .flatten()
            .map(|(term, _)| term)
            .min()
            .cloned()
        else {
            return Ok(None);
        };
        let mut sources = Vec::with_capacity(self.streams.len());
        for index in 0..self.current.len() {
            if let Some((key, row)) = &self.current[index]
                && *key == term
            {
                sources.push((index, *row));
                self.advance(index)?;
            }
        }
        let term = String::from_utf8(term)
            .map_err(|error| Error::index(format!("invalid FTS term: {error}")))?;
        Ok(Some(Term {
            token: term,
            sources,
        }))
    }
}

struct PostingCursor {
    reader: Arc<dyn IndexReader>,
    manifest: ManifestCursor,
    start: usize,
    batch: Option<RecordBatch>,
    limit: u64,
}

impl PostingCursor {
    fn contains(&self, row: usize) -> bool {
        self.batch
            .as_ref()
            .is_some_and(|batch| row >= self.start && row < self.start + batch.num_rows())
    }

    async fn at(&mut self, row: usize) -> Result<RecordBatch> {
        if self
            .batch
            .as_ref()
            .is_none_or(|batch| row < self.start || row >= self.start + batch.num_rows())
        {
            // Drop the previous decode before admitting the next batch.
            self.batch = None;
            self.start = row;
            let mut end = row;
            let mut bytes = 0u64;
            while end < self.reader.num_rows() && end - row < POSTING_READ_ROWS {
                let next = self.manifest.at(end).await?.read_bytes()?;
                if next > self.limit {
                    return Err(Error::index("FTS run row exceeds its admitted read buffer"));
                }
                if end != row
                    && bytes
                        .checked_add(next)
                        .is_none_or(|sum| sum > self.limit / 4)
                {
                    break;
                }
                bytes += next;
                end += 1;
            }
            if end == row {
                return Err(Error::index("FTS run row exceeds its admitted read buffer"));
            }
            let batch = self.reader.read_range(row..end, None).await?;
            // Projected read batches need not carry the file's codec metadata.
            // Reattach the admitted file schema before interpreting posting rows.
            let batch = RecordBatch::try_new(
                Arc::new(self.reader.schema().into()),
                batch.columns().to_vec(),
            )?;
            // Requested slices can retain their containing decoder pages.
            if batch.get_array_memory_size() as u64 > self.limit {
                return Err(Error::index(
                    "FTS run decode exceeds its admitted read buffer",
                ));
            }
            self.batch = Some(batch);
        }
        Ok(self
            .batch
            .as_ref()
            .ok_or_else(|| Error::internal("missing FTS posting batch"))?
            .slice(row - self.start, 1))
    }
}

struct TermInput {
    token: String,
    batches: Vec<(RecordBatch, u32)>,
}

async fn read_term(
    postings: &mut [PostingCursor],
    sources: &[(usize, usize)],
    bases: &[u32],
) -> Result<Vec<(RecordBatch, u32)>> {
    let reads = postings
        .iter_mut()
        .enumerate()
        .filter_map(|(source, cursor)| {
            sources
                .iter()
                .find_map(|&(index, row)| (index == source).then_some(row))
                .map(|row| async move { cursor.at(row).await.map(|batch| (batch, bases[source])) })
        });
    // Every cursor already owns a separate read reservation. Start independent
    // reads together, but drain all accepted work even when one read fails.
    join_all(reads).await.into_iter().collect()
}

fn encode_terms(
    inputs: Vec<TermInput>,
    docs: &DocSet,
    mut tokens: TokenDictionary,
    schema: Arc<Schema>,
    with_position: bool,
    block_size: usize,
) -> Result<(TokenDictionary, RecordBatch)> {
    let mut output = PostingListBatchBuilder::new(
        schema,
        with_position,
        InvertedListFormatVersion::V3,
        inputs.len(),
    );
    for input in inputs {
        tokens.get_or_add(&input.token);
        let mut merged = PostingListBuilder::new_with_posting_tail_codec_and_block_size(
            with_position,
            super::super::PostingTailCodec::VarintDelta,
            block_size,
        );
        for (batch, base) in input.batches {
            let max_score = batch
                .column_by_name(MAX_SCORE_COL)
                .ok_or_else(|| Error::index(format!("missing FTS column {MAX_SCORE_COL}")))?
                .as_primitive::<arrow_array::types::Float32Type>()
                .value(0);
            let length = batch
                .column_by_name(LENGTH_COL)
                .ok_or_else(|| Error::index(format!("missing FTS column {LENGTH_COL}")))?
                .as_primitive::<arrow_array::types::UInt32Type>()
                .value(0);
            let posting = PostingList::from_batch(&batch, Some(max_score), Some(length))?;
            for (doc, frequency, positions) in posting.iter() {
                let doc = u32::try_from(doc)
                    .ok()
                    .and_then(|doc| base.checked_add(doc))
                    .ok_or_else(|| Error::index("FTS merged document id overflow"))?;
                let positions = positions
                    .map(|values| PositionRecorder::Position(values.collect::<Vec<_>>().into()))
                    .unwrap_or(PositionRecorder::Count(frequency));
                merged.add(doc, positions);
            }
        }
        merged.append_to_batch_with_docs(docs, &mut output, InvertedListFormatVersion::V3)?;
    }
    Ok((tokens, output.finish()?))
}

pub(super) struct FinalMerge {
    store: Arc<dyn IndexStore>,
    target: PartitionWriteTarget,
    budget: u64,
    target_bytes: Option<u64>,
    ids: Arc<AtomicU64>,
    fragment_mask: u64,
    token_format: TokenSetFormat,
    with_position: bool,
    block_size: usize,
    stop: Stop,
}

impl FinalMerge {
    pub(super) fn new(
        builder: &InvertedIndexBuilder,
        store: Arc<dyn IndexStore>,
        target: PartitionWriteTarget,
        budget: u64,
        target_bytes: Option<u64>,
        stop: Stop,
    ) -> Self {
        Self {
            store,
            target,
            budget,
            target_bytes,
            stop,
            ids: builder.partition_id_allocator(),
            fragment_mask: builder.fragment_mask.unwrap_or(0),
            token_format: builder.token_set_format,
            with_position: builder.params.with_position,
            block_size: builder.params.block_size,
        }
    }

    async fn runs(&self, output: &WorkerOutput) -> Result<Vec<Run>> {
        let sizes = output
            .files
            .iter()
            .map(|file| (file.path.as_str(), file.size_bytes))
            .collect::<HashMap<_, _>>();
        let mut runs = Vec::with_capacity(output.partitions.len());
        for &id in &output.partitions {
            check_running(&self.stop)?;
            let manifest = self
                .store
                .open_index_file(&manifest_path(self.target, id))
                .await?;
            let docs = usize::try_from(header(manifest.as_ref(), "docs")?)
                .map_err(|_| Error::index("FTS run document count overflows usize"))?;
            let rank = usize::try_from(header(manifest.as_ref(), "coordinate_rank")?)
                .map_err(|_| Error::index("FTS run coordinate rank overflows usize"))?;
            let dictionary_bytes = header(manifest.as_ref(), "dictionary_bytes")?;
            let size = [
                self.target.posting_path(id),
                self.target.token_path(id),
                self.target.doc_path(id),
            ]
            .iter()
            .try_fold(0u64, |sum, path| {
                let size = sizes
                    .get(path.as_str())
                    .ok_or_else(|| Error::internal(format!("missing FTS run file {path}")))?;
                sum.checked_add(*size)
                    .ok_or_else(|| Error::index("FTS run file size overflow"))
            })?;
            runs.push(Run {
                id,
                size,
                terms: manifest.num_rows(),
                docs,
                rank,
                dictionary_bytes,
            });
        }
        Ok(runs)
    }

    pub(super) async fn finish(mut self, original: WorkerOutput) -> Result<WorkerOutput> {
        check_running(&self.stop)?;
        // Bound writer buffering as well as input reads. A custom store that
        // cannot enforce a writer cap keeps the valid first-level output.
        let bounded = match self.store.with_write_buffer_size((self.budget / 16).max(1)) {
            Ok(store) => store,
            Err(Error::NotSupported { .. }) => {
                log::info!("FTS_FINAL_MERGE_UNSUPPORTED retaining first-level partitions");
                self.clean_manifests(&original.partitions).await;
                return Ok(original);
            }
            Err(error) => return Err(error),
        };
        self.store = bounded.with_io_buffer_size((self.budget / 32).max(1));
        let mut runs = self.runs(&original).await?;
        runs.sort_unstable_by_key(|run| (run.size, run.id));
        let mut pending = Vec::new();
        let mut group = Vec::new();
        let mut bytes = 0u64;
        // Like a size-tiered policy, prefer similarly-sized runs and limit
        // fan-in independently of the final serialized-size estimate. The
        // target is independent of indexing-worker and merger counts.
        let target_bytes = self.target_bytes.unwrap_or((self.budget / 2).max(1));
        let floor_bytes = (self.budget / 128).max(1);
        for run in runs {
            let skewed = group.first().is_some_and(|first: &Run| {
                run.size > first.size.max(floor_bytes).saturating_mul(4)
            });
            if !group.is_empty()
                && (group.len() == MAX_FAN_IN
                    || bytes.saturating_add(run.size) > target_bytes
                    || skewed)
            {
                pending.push(std::mem::take(&mut group));
                bytes = 0;
            }
            bytes = bytes
                .checked_add(run.size)
                .ok_or_else(|| Error::index("FTS merge group size overflow"))?;
            group.push(run);
        }
        if !group.is_empty() {
            pending.push(group);
        }

        let mut output = WorkerOutput::default();
        let mut consumed = Vec::new();
        let mut merged = 0;
        let mut rejected = 0;
        while let Some(mut group) = pending.pop() {
            check_running(&self.stop)?;
            if group.len() > 1 {
                if let Some((id, files)) = self.merge_group(&group).await? {
                    consumed.extend(group.iter().map(|run| run.id));
                    output.partitions.push(id);
                    output.files.extend(files);
                    merged += 1;
                    continue;
                }
                rejected += 1;
                let right = group.split_off(group.len() / 2);
                pending.push(right);
                pending.push(group);
                continue;
            }
            for run in group {
                output.partitions.push(run.id);
                for path in [
                    self.target.posting_path(run.id),
                    self.target.token_path(run.id),
                    self.target.doc_path(run.id),
                ] {
                    let file = original
                        .files
                        .iter()
                        .find(|file| file.path == path)
                        .ok_or_else(|| {
                            Error::internal(format!("missing retained FTS file {path}"))
                        })?;
                    output.files.push(file.clone());
                }
            }
        }
        check_running(&self.stop)?;
        let paths = consumed
            .into_iter()
            .flat_map(|id| {
                [
                    self.target.posting_path(id),
                    self.target.token_path(id),
                    self.target.doc_path(id),
                ]
            })
            .collect();
        super::super::delete_index_files_best_effort(&self.store, paths).await;
        self.clean_manifests(&original.partitions).await;
        output.partitions.sort_unstable();
        log::info!(
            "FTS_FINAL_MERGE_COMPLETE inputs={} outputs={} merged_groups={} rejected_groups={} target_bytes={} working_budget_bytes={}",
            original.partitions.len(),
            output.partitions.len(),
            merged,
            rejected,
            target_bytes,
            self.budget
        );
        Ok(output)
    }

    async fn clean_manifests(&self, ids: &[u64]) {
        let paths = ids
            .iter()
            .map(|&id| manifest_path(self.target, id))
            .collect();
        super::super::delete_index_files_best_effort(&self.store, paths).await;
    }

    async fn merge_group(&self, runs: &[Run]) -> Result<Option<(u64, Vec<IndexFile>)>> {
        let mut doc_count = 0usize;
        let mut metadata_bytes = 0u64;
        let mut bases = Vec::with_capacity(runs.len());
        for run in runs {
            if run.rank != runs[0].rank {
                return Err(Error::index("FTS run coordinate ranks differ"));
            }
            bases.push(
                u32::try_from(doc_count)
                    .map_err(|_| Error::index("FTS document base overflows u32"))?,
            );
            doc_count = doc_count
                .checked_add(run.docs)
                .ok_or_else(|| Error::index("FTS merged document count overflow"))?;
            if doc_count > u32::MAX as usize {
                return Ok(None);
            }
            metadata_bytes = metadata_bytes
                .checked_add(4096)
                .and_then(|bytes| bytes.checked_add(run.dictionary_bytes.saturating_mul(6)))
                .and_then(|bytes| bytes.checked_add((run.terms as u64).saturating_mul(32)))
                .and_then(|bytes| bytes.checked_add((run.docs as u64).saturating_mul(16)))
                .ok_or_else(|| Error::index("FTS merge metadata size overflow"))?;
        }
        if doc_count > u32::MAX as usize || metadata_bytes > self.budget / 4 {
            return Ok(None);
        }
        let mut tokens = Vec::with_capacity(runs.len());
        for run in runs {
            check_running(&self.stop)?;
            let reader = self
                .store
                .open_index_file(&self.target.token_path(run.id))
                .await?;
            let dictionary = TokenSet::load(reader, self.token_format).await?;
            if dictionary.len() != run.terms {
                return Err(Error::index(
                    "FTS run dictionary and resource manifest lengths differ",
                ));
            }
            tokens.push(dictionary);
        }
        let read_limit = self.budget / (8 * runs.len() as u64);
        let work_limit = self.budget / 4;
        let mut manifests = Vec::with_capacity(runs.len());
        for run in runs {
            check_running(&self.stop)?;
            let reader = self
                .store
                .open_index_file(&manifest_path(self.target, run.id))
                .await?;
            if reader.num_rows() != run.terms {
                return Err(Error::index("FTS run manifest changed after planning"));
            }
            manifests.push(ManifestCursor::new(reader));
        }
        let mut terms = Terms::new(&tokens)?;
        let mut count = 0u64;
        // Validate every hot term before creating output or reading posting
        // payloads. An oversized term narrows fan-in, never forces an unbounded
        // decode or converts the whole group back into an in-memory builder.
        while let Some(Term { sources, .. }) = terms.next()? {
            let mut work = 0u64;
            for (source, row) in sources {
                let bounds = manifests[source].at(row).await?;
                if bounds.read_bytes()? > read_limit {
                    return Ok(None);
                }
                work = work
                    .checked_add(bounds.work_bytes()?)
                    .ok_or_else(|| Error::index("FTS term working set overflow"))?;
            }
            if work > work_limit || count >= u32::MAX as u64 {
                return Ok(None);
            }
            count += 1;
            if count.is_multiple_of(METADATA_ROWS as u64) {
                check_running(&self.stop)?;
                tokio::task::yield_now().await;
            }
        }
        drop(terms);

        // Coordinate columns are fixed-width but their count is caller-driven.
        // Size document windows by bytes too, rather than assuming a small rank.
        let doc_row_bytes = (runs[0].rank as u64)
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or_else(|| Error::index("FTS document row bound overflow"))?;
        let doc_batch_rows = (self.budget / 32 / doc_row_bytes).min(65_536) as usize;
        if doc_batch_rows == 0 {
            return Ok(None);
        }
        let mut lengths = Vec::with_capacity(doc_count);
        let mut doc_readers = Vec::with_capacity(runs.len());
        let mut postings = Vec::with_capacity(runs.len());
        let schema = inverted_list_schema_for_version_with_block_size(
            self.with_position,
            InvertedListFormatVersion::V3,
            self.block_size,
        );
        for (index, run) in runs.iter().enumerate() {
            check_running(&self.stop)?;
            let reader = self
                .store
                .open_index_file(&self.target.doc_path(run.id))
                .await?;
            if reader.num_rows() != run.docs {
                return Err(Error::index("FTS run document count differs from manifest"));
            }
            // Metadata admission charges sixteen bytes per document. The final
            // lengths slab and one projected UInt32 input each use four, so a
            // single sequential read fits the existing reservation without
            // hundreds of small remote requests. Identities/coordinates below
            // remain byte-windowed because their rank is caller-driven.
            let batch = reader
                .read_range(0..run.docs, Some(&[NUM_TOKEN_COL]))
                .await?;
            lengths.extend_from_slice(
                batch
                    .column_by_name(NUM_TOKEN_COL)
                    .ok_or_else(|| Error::index(format!("missing FTS column {NUM_TOKEN_COL}")))?
                    .as_primitive::<arrow_array::types::UInt32Type>()
                    .values(),
            );
            doc_readers.push(reader);
            let reader = self
                .store
                .open_index_file(&self.target.posting_path(run.id))
                .await?;
            let input_schema: Schema = reader.schema().into();
            if input_schema != *schema {
                return Err(Error::index("FTS run posting schemas differ"));
            }
            if reader.num_rows() != run.terms {
                return Err(Error::index("FTS run posting and manifest lengths differ"));
            }
            postings.push(PostingCursor {
                reader,
                manifest: manifests[index].clone(),
                start: 0,
                batch: None,
                limit: read_limit,
            });
        }
        let docs = Arc::new(DocSet::from_num_tokens_only(&UInt32Array::from(lengths)));
        let id = self
            .ids
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| Error::invalid_input("FTS partition id allocator overflow"))?
            | self.fragment_mask;
        let mut writer = self
            .store
            .new_index_file(&self.target.posting_path(id), schema.clone())
            .await?;
        let mut output_tokens = TokenDictionary::default();
        let mut terms = Terms::new(&tokens)?;
        let mut inputs = Vec::new();
        let mut work = 0u64;
        while let Some(Term { token, sources }) = terms.next()? {
            check_running(&self.stop)?;
            let mut term_work = 0u64;
            for &(source, row) in &sources {
                term_work = term_work
                    .checked_add(manifests[source].at(row).await?.work_bytes()?)
                    .ok_or_else(|| Error::index("FTS term working set overflow"))?;
            }
            if !inputs.is_empty()
                && (work
                    .checked_add(term_work)
                    .is_none_or(|bytes| bytes > work_limit)
                    || sources
                        .iter()
                        .any(|&(source, row)| !postings[source].contains(row)))
            {
                let batch = self
                    .encode_batch(
                        std::mem::take(&mut inputs),
                        docs.clone(),
                        output_tokens,
                        schema.clone(),
                    )
                    .await?;
                output_tokens = batch.0;
                writer.write_record_batch(batch.1).await?;
                work = 0;
            }
            let batches = read_term(&mut postings, &sources, &bases).await?;
            inputs.push(TermInput { token, batches });
            work += term_work;
            if inputs.len() == METADATA_ROWS {
                let batch = self
                    .encode_batch(
                        std::mem::take(&mut inputs),
                        docs.clone(),
                        output_tokens,
                        schema.clone(),
                    )
                    .await?;
                output_tokens = batch.0;
                writer.write_record_batch(batch.1).await?;
                work = 0;
            }
        }
        if !inputs.is_empty() {
            let batch = self
                .encode_batch(inputs, docs.clone(), output_tokens, schema.clone())
                .await?;
            output_tokens = batch.0;
            writer.write_record_batch(batch.1).await?;
        }
        let mut files = vec![writer.finish().await?];
        drop(writer);
        drop(terms);
        drop(tokens);
        drop(postings);
        drop(manifests);
        check_running(&self.stop)?;
        let format = self.token_format;
        let batch = spawn_cpu(move || output_tokens.to_batch(format)).await?;
        check_running(&self.stop)?;
        let mut writer = self
            .store
            .new_index_file(&self.target.token_path(id), batch.schema())
            .await?;
        writer.write_record_batch(batch).await?;
        files.push(writer.finish().await?);
        drop(writer);
        let schema = Arc::new(doc_readers[0].schema().into());
        let mut writer = self
            .store
            .new_index_file(&self.target.doc_path(id), schema)
            .await?;
        for (reader, run) in doc_readers.iter().zip(runs) {
            for start in (0..run.docs).step_by(doc_batch_rows) {
                check_running(&self.stop)?;
                let batch = reader
                    .read_range(start..(start + doc_batch_rows).min(run.docs), None)
                    .await?;
                writer.write_record_batch(batch).await?;
            }
        }
        files.push(
            writer
                .finish_with_metadata(HashMap::from([(
                    super::super::super::documents::TOTAL_TOKENS_KEY.to_owned(),
                    docs.total_tokens_num().to_string(),
                )]))
                .await?,
        );
        check_running(&self.stop)?;
        Ok(Some((id, files)))
    }

    async fn encode_batch(
        &self,
        inputs: Vec<TermInput>,
        docs: Arc<DocSet>,
        tokens: TokenDictionary,
        schema: Arc<Schema>,
    ) -> Result<(TokenDictionary, RecordBatch)> {
        let with_position = self.with_position;
        let block_size = self.block_size;
        let result = spawn_cpu(move || {
            encode_terms(inputs, &docs, tokens, schema, with_position, block_size)
        })
        .await;
        check_running(&self.stop)?;
        result
    }
}

fn manifest_path(target: PartitionWriteTarget, id: u64) -> String {
    format!("{}.merge.lance", target.posting_path(id))
}

fn check_running(stop: &Stop) -> Result<()> {
    if stop.is_cancelled() {
        Err(Error::execution("FTS final merge cancelled"))
    } else {
        Ok(())
    }
}

/// Canonical order lets the external merger read contiguous posting batches
/// instead of issuing one remote random read for every dictionary lookup.
fn order_run(builder: &mut InnerBuilder) {
    let (tokens, mut destinations) = {
        let mut entries = builder.tokens.iter().collect::<Vec<_>>();
        entries.sort_unstable_by(|lhs, rhs| lhs.1.cmp(rhs.1));
        let mut tokens = TokenDictionary::default();
        let mut destinations = vec![0; entries.len()];
        for (new_id, (old_id, token)) in entries.into_iter().enumerate() {
            tokens.get_or_add(token);
            destinations[old_id as usize] = new_id;
        }
        (tokens, destinations)
    };
    for index in 0..destinations.len() {
        while destinations[index] != index {
            let next = destinations[index];
            builder.posting_lists.swap(index, next);
            destinations.swap(index, next);
        }
    }
    builder.tokens = tokens;
}

/// Record bounds while the producer still owns its compressed lists. This
/// permits admission before reading a large Arrow row or decoding positions.
/// The manifest is temporary, and is removed before successful publication.
pub(super) async fn write_run(
    mut builder: InnerBuilder,
    store: &dyn IndexStore,
    target: PartitionWriteTarget,
    stop: &Stop,
) -> Result<(u64, Vec<IndexFile>)> {
    check_running(stop)?;
    let (mut builder, manifest) = spawn_cpu(move || {
        order_run(&mut builder);
        let retained = builder
            .posting_lists
            .iter()
            .map(|posting| posting.size())
            .collect::<Vec<_>>();
        let scratch = builder
            .posting_lists
            .iter()
            .map(|posting| posting.merge_scratch_bytes())
            .collect::<Result<Vec<_>>>()?;
        let schema = Schema::new_with_metadata(
            vec![
                Field::new(RETAINED, DataType::UInt64, false),
                Field::new(SCRATCH, DataType::UInt64, false),
            ],
            HashMap::from([
                (
                    "dictionary_bytes".into(),
                    builder.tokens.memory_size().to_string(),
                ),
                ("docs".into(), builder.docs.len().to_string()),
                (
                    "coordinate_rank".into(),
                    builder.docs.coordinate_rank().to_string(),
                ),
            ]),
        );
        let manifest = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(UInt64Array::from(retained)),
                Arc::new(UInt64Array::from(scratch)),
            ],
        )?;
        Result::Ok((builder, manifest))
    })
    .await?;
    check_running(stop)?;
    let id = builder.id();
    let mut writer = store
        .new_index_file(&manifest_path(target, id), manifest.schema())
        .await?;
    writer.write_record_batch(manifest).await?;
    writer.finish().await?;
    check_running(stop)?;
    let files = builder.write_to(store, target).await?;
    Ok((id, files))
}

#[cfg(test)]
mod tests;
