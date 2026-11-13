// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::scalar::lance_format::LanceIndexStore;
use lance_core::cache::LanceCache;
use lance_io::object_store::ObjectStore;
use object_store::path::Path;

fn store() -> Arc<dyn IndexStore> {
    Arc::new(LanceIndexStore::new(
        Arc::new(ObjectStore::memory()),
        Path::from("index"),
        Arc::new(LanceCache::no_cache()),
    ))
}

fn run(id: u64, positions: bool, coordinates: bool) -> InnerBuilder {
    let mut builder = InnerBuilder::new_with_format_version_and_block_size(
        id,
        positions,
        TokenSetFormat::Fst,
        InvertedListFormatVersion::V3,
        256,
    );
    for doc in 0..8u32 {
        let rank = if coordinates {
            vec![id as u32, doc]
        } else {
            Vec::new()
        };
        builder
            .docs
            .append_with_doc_index(
                id * 100 + u64::from(doc),
                (1..=256).map(|term| term + doc).sum(),
                &rank,
            )
            .unwrap();
    }
    // Deliberately reverse insertion order: the persisted cursor must still
    // match the dictionary and postings after canonicalization.
    for term in (0..256).rev() {
        builder.tokens.get_or_add(&format!("term{term:04}"));
        let mut posting = PostingListBuilder::new_with_posting_tail_codec_and_block_size(
            positions,
            builder.posting_tail_codec,
            builder.block_size,
        );
        for doc in 0..8u32 {
            let frequency = term as u32 + doc + 1;
            let values = if positions {
                PositionRecorder::Position((0..frequency).map(|p| p * 2).collect::<Vec<_>>().into())
            } else {
                PositionRecorder::Count(frequency)
            };
            posting.add(doc, values);
        }
        builder.posting_lists.push(posting);
    }
    builder
}

fn merger(store: Arc<dyn IndexStore>, positions: bool) -> FinalMerge {
    FinalMerge {
        store,
        target: PartitionWriteTarget::Final,
        budget: 4 << 20,
        ids: Arc::new(AtomicU64::new(100)),
        fragment_mask: 0,
        token_format: TokenSetFormat::Fst,
        with_position: positions,
        block_size: 256,
        stop: Stop::default(),
    }
}

async fn inputs(
    store: &dyn IndexStore,
    positions: bool,
    coordinates: bool,
) -> Result<WorkerOutput> {
    let mut output = WorkerOutput::default();
    for id in 0..4 {
        let (id, files) = write_run(
            run(id, positions, coordinates),
            store,
            PartitionWriteTarget::Final,
            &Stop::default(),
        )
        .await?;
        output.partitions.push(id);
        output.files.extend(files);
    }
    Ok(output)
}

#[rstest::rstest]
#[case::rows(false, false)]
#[case::positions(true, false)]
#[case::coordinates(false, true)]
#[case::coordinate_positions(true, true)]
#[tokio::test]
async fn merges_persisted_runs_preserving_documents(
    #[case] positions: bool,
    #[case] coordinates: bool,
) -> Result<()> {
    let store = store();
    let output = merger(store.clone(), positions)
        .finish(inputs(store.as_ref(), positions, coordinates).await?)
        .await?;
    assert_eq!(
        output.partitions.len(),
        1,
        "four persisted partitions must coalesce to one"
    );
    let id = output.partitions[0];
    let docs = DocSet::load(
        store
            .open_index_file(&PartitionWriteTarget::Final.doc_path(id))
            .await?,
        false,
        None,
    )
    .await?;
    assert_eq!(docs.len(), 32);
    assert_eq!(docs.coordinate_rank(), if coordinates { 2 } else { 0 });
    let tokens = TokenSet::load(
        store
            .open_index_file(&PartitionWriteTarget::Final.token_path(id))
            .await?,
        TokenSetFormat::Fst,
    )
    .await?;
    let postings = store
        .open_index_file(&PartitionWriteTarget::Final.posting_path(id))
        .await?;
    assert_eq!(tokens.len(), 256);
    for term in 0..256 {
        let token_id = tokens.get(&format!("term{term:04}")).unwrap() as usize;
        assert_eq!(token_id, term);
        let batch = postings.read_range(token_id..token_id + 1, None).await?;
        let batch =
            RecordBatch::try_new(Arc::new(postings.schema().into()), batch.columns().to_vec())?;
        let max_score = batch[MAX_SCORE_COL]
            .as_primitive::<arrow_array::types::Float32Type>()
            .value(0);
        let length = batch[LENGTH_COL]
            .as_primitive::<arrow_array::types::UInt32Type>()
            .value(0);
        let posting = PostingList::from_batch(&batch, Some(max_score), Some(length))?;
        let mut rows = Vec::new();
        for (doc, frequency, actual_positions) in posting.iter() {
            let row = docs.row_id(doc as u32);
            let source = (row / 100) as u32;
            let local = (row % 100) as u32;
            assert_eq!(frequency, term as u32 + local + 1);
            assert_eq!(docs.num_tokens(doc as u32), 32896 + 256 * local);
            if coordinates {
                assert_eq!(docs.doc_index(doc as u32), vec![source, local]);
            }
            if positions {
                assert_eq!(
                    actual_positions.unwrap().collect::<Vec<_>>(),
                    (0..frequency).map(|p| p * 2).collect::<Vec<_>>()
                );
            } else {
                assert!(actual_positions.is_none());
            }
            rows.push(row);
        }
        rows.sort_unstable();
        assert_eq!(
            rows,
            (0..4)
                .flat_map(|source| (0..8).map(move |doc| source * 100 + doc))
                .collect::<Vec<u64>>()
        );
    }
    assert!(
        store
            .list_files_with_sizes()
            .await?
            .iter()
            .all(|f| !f.path.ends_with(".merge.lance"))
    );
    assert!(
        store
            .open_index_file(&PartitionWriteTarget::Final.posting_path(0))
            .await
            .is_err()
    );
    Ok(())
}

#[tokio::test]
async fn cancelled_final_merge_retains_inputs() -> Result<()> {
    let store = store();
    let input = inputs(store.as_ref(), true, true).await?;
    let ctx = merger(store.clone(), true);
    ctx.stop.cancel();
    let error = ctx
        .finish(input)
        .await
        .err()
        .expect("cancelled merge must fail");
    assert!(matches!(error, Error::Execution { .. }));
    assert!(error.to_string().contains("cancelled"));
    assert!(
        store
            .open_index_file(&PartitionWriteTarget::Final.posting_path(0))
            .await
            .is_ok()
    );
    Ok(())
}

#[tokio::test]
async fn metadata_over_budget_keeps_valid_first_level_outputs() -> Result<()> {
    let store = store();
    let input = inputs(store.as_ref(), false, false).await?;
    let mut ctx = merger(store.clone(), false);
    ctx.budget = 16 << 10;
    let output = ctx.finish(input).await?;
    assert_eq!(output.partitions, vec![0, 1, 2, 3]);
    assert!(
        store
            .open_index_file(&PartitionWriteTarget::Final.posting_path(0))
            .await
            .is_ok()
    );
    Ok(())
}

// The wrapper rejects whole posting reads and observes only the external phase:
// fixtures are created through the underlying real store before wrapping it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    Read,
    ConcurrentRead,
    Open,
    LaterOpen,
    Write,
    Finish,
    Pause,
}

#[derive(Debug)]
struct Probe {
    fault: Fault,
    reads: std::sync::atomic::AtomicUsize,
    doc_reads: std::sync::atomic::AtomicUsize,
    writes: std::sync::atomic::AtomicUsize,
    manifest_live: std::sync::atomic::AtomicUsize,
    manifest_peak: std::sync::atomic::AtomicUsize,
    posting_bytes_peak: std::sync::atomic::AtomicUsize,
    started: tokio::sync::Notify,
    release: tokio::sync::Notify,
    other_started: tokio::sync::Notify,
    other_release: tokio::sync::Notify,
    read_failed: tokio::sync::Notify,
    other_finished: std::sync::atomic::AtomicBool,
}

#[derive(Clone, Debug)]
struct ProbeStore {
    inner: Arc<dyn IndexStore>,
    probe: Arc<Probe>,
}

impl ProbeStore {
    fn new(inner: Arc<dyn IndexStore>, fault: Fault) -> Self {
        Self {
            inner,
            probe: Arc::new(Probe {
                fault,
                reads: Default::default(),
                doc_reads: Default::default(),
                writes: Default::default(),
                manifest_live: Default::default(),
                manifest_peak: Default::default(),
                posting_bytes_peak: Default::default(),
                started: Default::default(),
                release: Default::default(),
                other_started: Default::default(),
                other_release: Default::default(),
                read_failed: Default::default(),
                other_finished: Default::default(),
            }),
        }
    }
}

impl lance_core::deepsize::DeepSizeOf for ProbeStore {
    fn deep_size_of_children(&self, _: &mut lance_core::deepsize::Context) -> usize {
        0
    }
}

#[async_trait::async_trait]
impl IndexStore for ProbeStore {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_arc(&self) -> Arc<dyn IndexStore> {
        Arc::new(self.clone())
    }
    fn io_parallelism(&self) -> usize {
        self.inner.io_parallelism()
    }
    fn with_io_priority(&self, priority: u64) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.with_io_priority(priority),
            ..self.clone()
        })
    }
    fn with_io_buffer_size(&self, bytes: u64) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.with_io_buffer_size(bytes),
            ..self.clone()
        })
    }
    fn with_write_buffer_size(&self, bytes: u64) -> Result<Arc<dyn IndexStore>> {
        Ok(Arc::new(Self {
            inner: self.inner.with_write_buffer_size(bytes)?,
            ..self.clone()
        }))
    }
    async fn new_index_file(
        &self,
        name: &str,
        schema: Arc<Schema>,
    ) -> Result<Box<dyn crate::scalar::IndexWriter>> {
        let opened = self.probe.writes.fetch_add(1, Ordering::Relaxed);
        if self.probe.fault == Fault::Open || (self.probe.fault == Fault::LaterOpen && opened == 3)
        {
            return Err(Error::execution("injected final open"));
        }
        Ok(Box::new(ProbeWriter {
            inner: self.inner.new_index_file(name, schema).await?,
            probe: self.probe.clone(),
        }))
    }
    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
        let reader = self.inner.open_index_file(name).await?;
        let is_manifest = name.ends_with(".merge.lance");
        let is_documents = name.ends_with("docs.lance");
        if is_manifest || is_documents || name.ends_with("invert.lance") {
            if is_manifest {
                let live = self.probe.manifest_live.fetch_add(1, Ordering::Relaxed) + 1;
                self.probe.manifest_peak.fetch_max(live, Ordering::Relaxed);
            }
            Ok(Arc::new(ProbeReader {
                inner: reader,
                probe: self.probe.clone(),
                is_manifest,
                is_documents,
            }))
        } else {
            Ok(reader)
        }
    }
    async fn copy_index_file(&self, name: &str, dest: &dyn IndexStore) -> Result<IndexFile> {
        self.inner.copy_index_file(name, dest).await
    }
    async fn rename_index_file(&self, name: &str, new_name: &str) -> Result<IndexFile> {
        self.inner.rename_index_file(name, new_name).await
    }
    async fn delete_index_file(&self, name: &str) -> Result<()> {
        self.inner.delete_index_file(name).await
    }
    async fn list_files_with_sizes(&self) -> Result<Vec<IndexFile>> {
        self.inner.list_files_with_sizes().await
    }
}

struct ProbeReader {
    inner: Arc<dyn IndexReader>,
    probe: Arc<Probe>,
    is_manifest: bool,
    is_documents: bool,
}

impl Drop for ProbeReader {
    fn drop(&mut self) {
        if self.is_manifest {
            self.probe.manifest_live.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

#[async_trait::async_trait]
impl IndexReader for ProbeReader {
    async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
        self.read_range(
            (n * batch_size) as usize..((n + 1) * batch_size).min(self.num_rows() as u64) as usize,
            None,
        )
        .await
    }
    async fn read_range(
        &self,
        range: std::ops::Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        assert!(
            self.is_manifest || self.is_documents || range.len() <= 4096,
            "external merge must bound rows even for empty payloads"
        );
        let is_posting = !self.is_manifest && !self.is_documents;
        let ticket = if is_posting {
            self.probe.reads.fetch_add(1, Ordering::Relaxed)
        } else {
            0
        };
        if self.is_documents {
            self.probe.doc_reads.fetch_add(1, Ordering::Relaxed);
        }
        if is_posting && self.probe.fault == Fault::Read {
            return Err(Error::execution("injected final read"));
        }
        if is_posting && self.probe.fault == Fault::ConcurrentRead {
            if ticket == 0 {
                self.probe.release.notified().await;
                self.probe.read_failed.notify_one();
                return Err(Error::execution("injected concurrent final read"));
            }
            if ticket == 1 {
                self.probe.other_started.notify_one();
                self.probe.other_release.notified().await;
            }
        }
        let batch = self.inner.read_range(range, projection).await?;
        if is_posting {
            self.probe
                .posting_bytes_peak
                .fetch_max(batch.get_array_memory_size(), Ordering::Relaxed);
            if self.probe.fault == Fault::ConcurrentRead && ticket == 1 {
                self.probe.other_finished.store(true, Ordering::Relaxed);
            }
        }
        Ok(batch)
    }
    async fn num_batches(&self, batch_size: u64) -> u32 {
        self.inner.num_batches(batch_size).await
    }
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }
    fn schema(&self) -> &lance_core::datatypes::Schema {
        self.inner.schema()
    }
}

struct ProbeWriter {
    inner: Box<dyn crate::scalar::IndexWriter>,
    probe: Arc<Probe>,
}

#[async_trait::async_trait]
impl crate::scalar::IndexWriter for ProbeWriter {
    async fn write_record_batch(&mut self, batch: RecordBatch) -> Result<u64> {
        if self.probe.fault == Fault::Write {
            return Err(Error::execution("injected final write"));
        }
        if self.probe.fault == Fault::Pause {
            self.probe.started.notify_one();
            self.probe.release.notified().await;
        }
        self.inner.write_record_batch(batch).await
    }
    async fn finish(&mut self) -> Result<IndexFile> {
        if self.probe.fault == Fault::Finish {
            return Err(Error::execution("injected final finish"));
        }
        self.inner.finish().await
    }
    async fn finish_with_metadata(
        &mut self,
        metadata: HashMap<String, String>,
    ) -> Result<IndexFile> {
        self.inner.finish_with_metadata(metadata).await
    }
}

#[tokio::test]
async fn persisted_postings_are_read_in_bounded_windows() -> Result<()> {
    let inner = store();
    let input = inputs(inner.as_ref(), true, true).await?;
    let store = ProbeStore::new(inner, Fault::None);
    let result = merger(Arc::new(store.clone()), true).finish(input).await?;
    assert_eq!(result.partitions.len(), 1);
    assert!(store.probe.reads.load(Ordering::Relaxed) >= 8);
    assert!(store.probe.posting_bytes_peak.load(Ordering::Relaxed) <= (4 << 20) / (8 * 4));
    Ok(())
}

#[tokio::test]
async fn small_postings_fill_the_admitted_read_window() -> Result<()> {
    let inner = store();
    let mut input = WorkerOutput::default();
    for id in 0..4 {
        let mut builder = run(id, false, false);
        builder.tokens = TokenDictionary::default();
        builder.posting_lists.clear();
        for term in 0..1024 {
            builder.tokens.get_or_add(&format!("small{term:04}"));
            let mut posting = PostingListBuilder::new_with_posting_tail_codec_and_block_size(
                false,
                builder.posting_tail_codec,
                builder.block_size,
            );
            posting.add(0, PositionRecorder::Count(1));
            builder.posting_lists.push(posting);
        }
        let (id, files) = write_run(
            builder,
            inner.as_ref(),
            PartitionWriteTarget::Final,
            &Stop::default(),
        )
        .await?;
        input.partitions.push(id);
        input.files.extend(files);
    }
    let store = ProbeStore::new(inner, Fault::None);
    let mut ctx = merger(Arc::new(store.clone()), false);
    ctx.budget = 64 << 20;
    let output = ctx.finish(input).await?;
    assert_eq!(output.partitions.len(), 1);
    assert!(
        store.probe.reads.load(Ordering::Relaxed) <= 8,
        "small posting rows must amortize remote reads across the admitted byte window"
    );
    assert!(store.probe.posting_bytes_peak.load(Ordering::Relaxed) <= (64 << 20) / (8 * 4));
    Ok(())
}

#[tokio::test]
async fn document_reads_use_admitted_metadata_and_byte_windows() -> Result<()> {
    let inner = store();
    let mut input = WorkerOutput::default();
    for id in 0..4 {
        let mut builder = InnerBuilder::new_with_format_version_and_block_size(
            id,
            false,
            TokenSetFormat::Fst,
            InvertedListFormatVersion::V3,
            256,
        );
        builder.tokens.get_or_add("word");
        let mut posting = PostingListBuilder::new_with_posting_tail_codec_and_block_size(
            false,
            builder.posting_tail_codec,
            builder.block_size,
        );
        for doc in 0..20_000u32 {
            builder
                .docs
                .append_with_doc_index(id * 100_000 + u64::from(doc), 1, &[])
                .unwrap();
            posting.add(doc, PositionRecorder::Count(1));
        }
        builder.posting_lists.push(posting);
        let (id, files) = write_run(
            builder,
            inner.as_ref(),
            PartitionWriteTarget::Final,
            &Stop::default(),
        )
        .await?;
        input.partitions.push(id);
        input.files.extend(files);
    }
    let store = ProbeStore::new(inner, Fault::None);
    let mut ctx = merger(Arc::new(store.clone()), false);
    ctx.budget = 128 << 20;
    let output = ctx.finish(input).await?;
    assert_eq!(output.partitions.len(), 1);
    assert!(
        store.probe.doc_reads.load(Ordering::Relaxed) <= 8,
        "admitted lengths and small document files must not require many remote round trips"
    );
    Ok(())
}

#[tokio::test]
async fn concurrent_posting_read_failure_drains_other_reads() -> Result<()> {
    let inner = store();
    let input = inputs(inner.as_ref(), true, true).await?;
    let store = ProbeStore::new(inner, Fault::ConcurrentRead);
    let ctx = merger(Arc::new(store.clone()), true);
    let mut task = tokio::spawn(async move { ctx.finish(input).await });
    let overlaps = tokio::time::timeout(
        std::time::Duration::from_millis(200),
        store.probe.other_started.notified(),
    )
    .await
    .is_ok();
    store.probe.release.notify_waiters();
    if overlaps {
        store.probe.read_failed.notified().await;
        let early = tokio::time::timeout(std::time::Duration::from_millis(20), &mut task).await;
        store.probe.other_release.notify_waiters();
        assert!(
            early.is_err(),
            "an accepted read must drain before the first error returns"
        );
    } else {
        store.probe.other_release.notify_waiters();
    }
    let error = task.await.unwrap().err().expect("injected read failure");
    assert!(matches!(error, Error::Execution { .. }));
    assert!(error.to_string().contains("injected concurrent final read"));
    assert!(overlaps, "independent admitted input reads must overlap");
    assert!(store.probe.other_finished.load(Ordering::Relaxed));
    Ok(())
}

#[rstest::rstest]
#[case::read(Fault::Read)]
#[case::open(Fault::Open)]
#[case::write(Fault::Write)]
#[case::finish(Fault::Finish)]
#[tokio::test]
async fn final_failure_preserves_all_inputs(#[case] fault: Fault) -> Result<()> {
    let inner = store();
    let input = inputs(inner.as_ref(), true, true).await?;
    let probe = Arc::new(ProbeStore::new(inner.clone(), fault));
    let error = merger(probe, true)
        .finish(input)
        .await
        .err()
        .expect("injected error");
    assert!(matches!(error, Error::Execution { .. }));
    assert!(error.to_string().contains("injected final"));
    for id in 0..4 {
        assert!(
            inner
                .open_index_file(&PartitionWriteTarget::Final.posting_path(id))
                .await
                .is_ok()
        );
    }
    Ok(())
}

#[rstest::rstest]
#[case::retained(false)]
#[case::scratch(true)]
#[tokio::test]
async fn hot_term_is_rejected_before_posting_reads(#[case] scratch: bool) -> Result<()> {
    let inner = store();
    let input = inputs(inner.as_ref(), true, true).await?;
    // Force the first term of every run over the quota using its private
    // manifest. No large allocation is needed to challenge pre-decode admission.
    for id in 0..4 {
        let path = manifest_path(PartitionWriteTarget::Final, id);
        let reader = inner.open_index_file(&path).await?;
        let batch = reader.read_range(0..reader.num_rows(), None).await?;
        let mut columns = batch.columns().to_vec();
        let mut bounds = vec![1; batch.num_rows()];
        bounds[0] = 4 << 20;
        columns[usize::from(scratch)] = Arc::new(UInt64Array::from(bounds));
        let schema: Arc<Schema> = Arc::new(reader.schema().into());
        let mut writer = inner.new_index_file(&path, schema.clone()).await?;
        writer
            .write_record_batch(RecordBatch::try_new(schema, columns)?)
            .await?;
        writer.finish().await?;
    }
    let store = ProbeStore::new(inner, Fault::None);
    let output = merger(Arc::new(store.clone()), true).finish(input).await?;
    assert_eq!(output.partitions.len(), 4);
    assert_eq!(store.probe.reads.load(Ordering::Relaxed), 0);
    assert_eq!(store.probe.writes.load(Ordering::Relaxed), 0);
    Ok(())
}

#[tokio::test]
async fn cancellation_joins_paused_final_write() -> Result<()> {
    let inner = store();
    let input = inputs(inner.as_ref(), true, true).await?;
    let store = ProbeStore::new(inner.clone(), Fault::Pause);
    let merger = merger(Arc::new(store.clone()), true);
    let stop = merger.stop.clone();
    let mut task = tokio::spawn(merger.finish(input));
    tokio::time::timeout(
        std::time::Duration::from_secs(1),
        store.probe.started.notified(),
    )
    .await
    .unwrap();
    stop.cancel();
    assert!(matches!(
        futures::poll!(&mut task),
        std::task::Poll::Pending
    ));
    // Started I/O is still owned. Its release must lead to cancellation,
    // without deleting any source or publishing new root metadata.
    store.probe.release.notify_one();
    let error = tokio::time::timeout(std::time::Duration::from_secs(1), task)
        .await
        .unwrap()
        .unwrap()
        .err()
        .expect("cancelled final merge");
    assert!(matches!(error, Error::Execution { .. }));
    assert!(error.to_string().contains("cancelled"));
    for id in 0..4 {
        assert!(
            inner
                .open_index_file(&PartitionWriteTarget::Final.posting_path(id))
                .await
                .is_ok()
        );
    }
    Ok(())
}

#[tokio::test]
async fn later_group_failure_keeps_previously_merged_sources() -> Result<()> {
    let inner = store();
    let mut input = inputs(inner.as_ref(), true, true).await?;
    for id in 4..8 {
        let (id, files) = write_run(
            run(id, true, true),
            inner.as_ref(),
            PartitionWriteTarget::Final,
            &Stop::default(),
        )
        .await?;
        input.partitions.push(id);
        input.files.extend(files);
    }
    let store = ProbeStore::new(inner.clone(), Fault::LaterOpen);
    let error = merger(Arc::new(store), true)
        .finish(input)
        .await
        .err()
        .expect("second group must fail");
    assert!(matches!(error, Error::Execution { .. }));
    assert!(error.to_string().contains("injected final open"));
    for id in 0..8 {
        assert!(
            inner
                .open_index_file(&PartitionWriteTarget::Final.posting_path(id))
                .await
                .is_ok()
        );
    }
    Ok(())
}

#[tokio::test]
async fn arrow_dictionary_is_preserved_by_final_merge() -> Result<()> {
    let store = store();
    let mut input = WorkerOutput::default();
    for id in 0..4 {
        let mut builder = run(id, true, false);
        builder.token_set_format = TokenSetFormat::Arrow;
        let (id, files) = write_run(
            builder,
            store.as_ref(),
            PartitionWriteTarget::Final,
            &Stop::default(),
        )
        .await?;
        input.partitions.push(id);
        input.files.extend(files);
    }
    let mut ctx = merger(store.clone(), true);
    ctx.token_format = TokenSetFormat::Arrow;
    let output = ctx.finish(input).await?;
    assert_eq!(output.partitions.len(), 1);
    let reader = store
        .open_index_file(&PartitionWriteTarget::Final.token_path(output.partitions[0]))
        .await?;
    let tokens = TokenSet::load(reader, TokenSetFormat::Arrow).await?;
    for term in 0..256 {
        assert_eq!(tokens.get(&format!("term{term:04}")), Some(term));
    }
    Ok(())
}

#[tokio::test]
async fn idle_runs_do_not_keep_manifest_readers_alive() -> Result<()> {
    let inner = store();
    let mut input = WorkerOutput::default();
    for id in 0..16 {
        let (id, files) = write_run(
            run(id, false, false),
            inner.as_ref(),
            PartitionWriteTarget::Final,
            &Stop::default(),
        )
        .await?;
        input.partitions.push(id);
        input.files.extend(files);
    }
    let store = ProbeStore::new(inner, Fault::None);
    let output = merger(Arc::new(store.clone()), false).finish(input).await?;
    assert_eq!(output.partitions.len(), 4);
    assert!(
        store.probe.manifest_peak.load(Ordering::Relaxed) <= 4,
        "idle runs retained file readers outside the admitted merge group"
    );
    assert_eq!(store.probe.manifest_live.load(Ordering::Relaxed), 0);
    Ok(())
}
