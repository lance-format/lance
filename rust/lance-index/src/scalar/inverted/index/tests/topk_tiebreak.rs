// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

// Deterministic top-k tiebreak (score DESC, row_id ASC).
//
// A BM25 top-k that breaks ties arbitrarily is not a stable prefix across `k`:
// top-k1 is not a prefix of top-k2, so paginating a tied-score query (a broad
// term where many documents share `(tf, doc_len)`) duplicates and skips rows.
// Four coupled pieces fix that, all keyed on the row address:
//   1. `ScoredDoc::cmp` is the total order used for the final sort.
//   2. `wand::TopKCollector` evicts on the full key when the walk knows row
//      addresses, and retains the whole k-th-score tie band when it does not.
//   3. `wand::admit_ties_floor` lowers the WAND threshold one ULP so the
//      score-only prune kernels keep, rather than drop, that band, including a
//      slower partition's ties behind the shared cross-partition floor.
//   4. `ModernCandidates::push` keeps the band across partitions and
//      `rank_resolved_documents` re-selects once addresses are resolved.
//
// The tests below cover a single partition, a post-compaction address order that
// diverges from DocId order, several partitions, and a tie band wide enough to
// overflow the collector's buffer and force the address-ordered retry.

struct MaskPreFilter {
    mask: Arc<RowAddrMask>,
}

#[cfg_attr(coverage, coverage(off))]
#[async_trait]
impl PreFilter for MaskPreFilter {
    async fn wait_for_ready(&self) -> Result<()> {
        Ok(())
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn mask(&self) -> Arc<RowAddrMask> {
        self.mask.clone()
    }

    fn filter_row_ids<'a>(&self, row_ids: Box<dyn Iterator<Item = &'a u64> + 'a>) -> Vec<u64> {
        self.mask.selected_indices(row_ids)
    }
}

/// Build one partition whose documents all score identically: term "alpha"
/// once, doc length 1. `row_ids` is indexed by DocId, so passing an unsorted
/// slice models a post-compaction remap.
async fn build_tied_partition(store: &Arc<LanceIndexStore>, partition_id: u64, row_ids: &[u64]) {
    let mut builder = InnerBuilder::new(partition_id, false, TokenSetFormat::default());
    builder.tokens.get_or_add("alpha");
    builder.posting_lists.push(PostingListBuilder::new(false));
    for (doc_id, &row_id) in row_ids.iter().enumerate() {
        builder.posting_lists[0].add(doc_id as u32, PositionRecorder::Count(1));
        builder.docs.append(row_id, 1);
    }
    builder.write(store.as_ref()).await.unwrap();
}

async fn load_tied_index(partitions: &[(u64, Vec<u64>)]) -> (TempObjDir, Arc<InvertedIndex>) {
    load_tied_index_with(partitions, &LanceCache::with_capacity(64 * 1024 * 1024)).await
}

/// Load a tied corpus against an explicit cache. An uncached index never keeps a
/// partition's row addresses resident between queries, so its first walk starts
/// without them.
async fn load_tied_index_with(
    partitions: &[(u64, Vec<u64>)],
    cache: &LanceCache,
) -> (TempObjDir, Arc<InvertedIndex>) {
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    for (partition_id, row_ids) in partitions {
        build_tied_partition(&store, *partition_id, row_ids).await;
    }
    write_test_metadata(
        &store,
        partitions.iter().map(|(id, _)| *id).collect(),
        InvertedIndexParams::default(),
    )
    .await;
    let index = InvertedIndex::load(store, None, cache).await.unwrap();
    (tmpdir, index)
}

async fn search_alpha_with(
    index: &InvertedIndex,
    limit: usize,
    prefilter: Arc<dyn PreFilter>,
) -> Vec<u64> {
    let (row_ids, scores) = index
        .bm25_search(
            Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text)),
            Arc::new(FtsSearchParams::new().with_limit(Some(limit))),
            Operator::Or,
            prefilter,
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap();
    // Every document ties, so the premise of these tests is that the returned
    // scores are all identical; without that the row_id order proves nothing.
    for window in scores.windows(2) {
        assert!(
            (window[0] - window[1]).abs() < 1e-6,
            "premise: scores must be tied, got {scores:?}"
        );
    }
    row_ids
}

async fn search_alpha(index: &InvertedIndex, limit: usize) -> Vec<u64> {
    search_alpha_with(index, limit, Arc::new(NoFilter)).await
}

/// Search for "alpha" and return `(row_id, doc_index)` per hit, so an
/// element-granularity result can be checked down to the element.
async fn search_alpha_documents(index: &InvertedIndex, limit: usize) -> Vec<(u64, Vec<u32>)> {
    index
        .bm25_search_documents(
            Arc::new(Tokens::new(vec!["alpha".to_owned()], DocType::Text)),
            Arc::new(FtsSearchParams::new().with_limit(Some(limit))),
            Operator::Or,
            Arc::new(NoFilter),
            Arc::new(NoOpMetricsCollector),
            None,
        )
        .await
        .unwrap()
        .into_iter()
        .map(|document| (document.row_id, document.doc_index))
        .collect()
}

#[tokio::test]
async fn test_fts_topk_tied_scores_resolve_by_row_id_not_doc_id() {
    // Row addresses no longer ascend with DocId, as after a compaction remap.
    // Postings are still walked in DocId order, so a collector that kept the
    // first documents it saw would return 105, 100, 107 instead of 100, 101, 102.
    let (_tmpdir, index) =
        load_tied_index(&[(0, vec![105, 100, 107, 102, 104, 101, 106, 103])]).await;

    let top3 = search_alpha(&index, 3).await;
    let top5 = search_alpha(&index, 5).await;
    assert_eq!(
        top3,
        vec![100, 101, 102],
        "the tie band must resolve by row_id, not by DocId or encounter order"
    );
    assert_eq!(top5, vec![100, 101, 102, 103, 104]);
    assert_eq!(top3, top5[..3], "top-3 must be a prefix of top-5");
}

#[tokio::test]
async fn test_fts_topk_tied_scores_stable_prefix_across_partitions() {
    // Tied documents split across partitions with interleaved row_ids, so the
    // shared cross-partition WAND floor and the merge must resolve ties by
    // row_id no matter which partition finishes first (`buffer_unordered`).
    let (_tmpdir, index) =
        load_tied_index(&[(0, vec![100, 102, 104, 106]), (1, vec![101, 103, 105, 107])]).await;

    let top3 = search_alpha(&index, 3).await;
    let top6 = search_alpha(&index, 6).await;
    assert_eq!(
        top3,
        vec![100, 101, 102],
        "cross-partition ties must resolve by row_id"
    );
    assert_eq!(top6, vec![100, 101, 102, 103, 104, 105]);
    assert_eq!(top3, top6[..3], "top-3 must be a prefix of top-6");
}

#[tokio::test]
async fn test_fts_topk_tied_scores_are_stable_across_repeated_searches() {
    // Twelve partitions all tied at one score, with row_ids striped across them
    // so the correct top-6 draws one document from each of six partitions. The
    // partitions complete in whatever order `buffer_unordered` yields, so
    // repeating the query checks directly that the result does not depend on it.
    const PARTITIONS: u64 = 12;
    let partitions = (0..PARTITIONS)
        .map(|partition_id| {
            let row_ids = (0..3)
                .map(|offset| partition_id + offset * PARTITIONS)
                .collect::<Vec<_>>();
            (partition_id, row_ids)
        })
        .collect::<Vec<_>>();
    let (_tmpdir, index) = load_tied_index(&partitions).await;

    let expected_top6 = vec![0, 1, 2, 3, 4, 5];
    for _ in 0..8 {
        assert_eq!(search_alpha(&index, 3).await, expected_top6[..3]);
        assert_eq!(search_alpha(&index, 6).await, expected_top6);
    }
}

#[tokio::test]
async fn test_fts_topk_tied_scores_survive_a_prefilter() {
    // An explicit allow-list routes the search off the unfiltered fast path.
    // The partition then loads its addresses to build visibility, so the walk
    // settles the tie itself instead of deferring it to the merge.
    let row_ids = [105u64, 100, 107, 102, 104, 101, 106, 103];
    let (_tmpdir, index) = load_tied_index(&[(0, row_ids.to_vec())]).await;
    let allow_all = || -> Arc<dyn PreFilter> {
        Arc::new(MaskPreFilter {
            mask: Arc::new(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
                row_ids.iter().copied(),
            ))),
        })
    };

    let top3 = search_alpha_with(&index, 3, allow_all()).await;
    let top5 = search_alpha_with(&index, 5, allow_all()).await;
    assert_eq!(
        top3,
        vec![100, 101, 102],
        "the filtered path must resolve ties by row_id too"
    );
    assert_eq!(top5, vec![100, 101, 102, 103, 104]);
    assert_eq!(top3, top5[..3], "top-3 must be a prefix of top-5");
}

/// Build one element-granularity partition. Each entry is `(row_id, doc_index)`
/// and is appended in the given order, so `row_ids` indexed by DocId can
/// disagree with the element coordinates the result must be ordered by.
async fn build_tied_element_partition(
    store: &Arc<LanceIndexStore>,
    partition_id: u64,
    elements: &[(u64, u32)],
) {
    let mut builder = InnerBuilder::new(partition_id, false, TokenSetFormat::default());
    builder.docs = DocSet::with_coordinate_rank(1);
    builder.tokens.get_or_add("alpha");
    builder.posting_lists.push(PostingListBuilder::new(false));
    for (doc_id, &(row_id, doc_index)) in elements.iter().enumerate() {
        builder.posting_lists[0].add(doc_id as u32, PositionRecorder::Count(1));
        builder
            .docs
            .append_with_doc_index(row_id, 1, &[doc_index])
            .unwrap();
    }
    builder.write(store.as_ref()).await.unwrap();
}

#[tokio::test]
async fn test_fts_topk_tied_elements_of_one_row_resolve_by_doc_index() {
    // Element documents of one row share its address, so the WAND ordering key
    // ranks the row but not its elements. The winning element is visited after
    // the heap is full, which a collector that let the address key settle the
    // tie would drop, and no later sort could bring it back.
    let tmpdir = TempObjDir::default();
    let store = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    build_tied_element_partition(&store, 0, &[(100, 1), (100, 2), (100, 0), (50, 0)]).await;
    write_test_metadata(&store, vec![0], InvertedIndexParams::default()).await;
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();

    let expected = vec![(50, vec![0]), (100, vec![0])];
    // Cold: the walk defers addresses, so the whole k-th-score band is retained.
    assert_eq!(search_alpha_documents(&index, 2).await, expected);

    // Prewarmed: the address projection is resident, so the walk ranks by row
    // address and only the elements of one row stay ambiguous.
    index.partitions[0]
        .docs
        .modern()
        .unwrap()
        .prewarm()
        .await
        .unwrap();
    assert!(index.has_resident_document_projections());
    assert_eq!(search_alpha_documents(&index, 2).await, expected);

    // Stable prefix across k, down to the element.
    let top4 = search_alpha_documents(&index, 4).await;
    assert_eq!(
        top4,
        vec![
            (50, vec![0]),
            (100, vec![0]),
            (100, vec![1]),
            (100, vec![2])
        ]
    );
    assert_eq!(top4[..2], expected);
}

#[test]
fn test_push_scored_key_breaks_legacy_ties_by_row_id() {
    // The legacy merge holds row addresses, so it settles a score tie on the
    // spot: a lower row_id evicts the worst tied entry and a higher one loses,
    // whatever order the candidates arrive in.
    let mut candidates = BinaryHeap::new();
    for (row_id, score) in [(20u64, 1.0f32), (10, 1.0), (5, 1.0), (30, 1.0), (15, 0.5)] {
        push_scored_key(&mut candidates, 2, row_id, score);
    }
    let ranked = candidates
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse(doc)| (doc.row_id, doc.score.0))
        .collect::<Vec<_>>();
    assert_eq!(ranked, vec![(5, 1.0), (10, 1.0)]);
}

#[test]
fn test_push_scored_key_rejects_nan_scores() {
    // A NaN score is not a rank. `OrderedFloat` compares with `total_cmp`, which
    // sorts NaN above every real score, so the legacy merge drops it instead of
    // letting it take the top of the result. Before the tiebreak the same
    // candidate slipped into an under-filled heap and came back first.
    let mut candidates = BinaryHeap::new();
    push_scored_key(&mut candidates, 4, 7, f32::NAN);
    assert!(candidates.is_empty(), "NaN must never become a candidate");

    push_scored_key(&mut candidates, 4, 1, 1.0);
    push_scored_key(&mut candidates, 4, 2, f32::NAN);
    let ranked = candidates
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse(doc)| doc.row_id)
        .collect::<Vec<_>>();
    assert_eq!(ranked, vec![1]);
}

#[tokio::test]
async fn test_fts_topk_tied_scores_bound_the_cross_partition_band() {
    // Every document ties, so each partition hands the merge its whole
    // `limit + SCORE_FLOOR_BUFFER` band. Six of those overrun the merge's own
    // bound several times over, which is the one buffer on this path that grows
    // with the number of partitions rather than with `limit`.
    const PARTITIONS: u64 = 6;
    const PER_PARTITION: u64 = 100;
    const LIMIT: usize = 4;
    let partitions = (0..PARTITIONS)
        .map(|partition_id| {
            let base = partition_id * 1_000;
            // Addresses descend with DocId, so the partitions keep handing over
            // their tie band: an address-ascending partition ranks the ties during
            // the walk and the merge never sees enough candidates to compact.
            let row_ids = (0..PER_PARTITION)
                .rev()
                .map(|d| base + d)
                .collect::<Vec<_>>();
            (partition_id, row_ids)
        })
        .collect::<Vec<_>>();
    let (_tmpdir, index) = load_tied_index(&partitions).await;

    // Each partition's band fits its collector, so every document reaches the
    // merge, and together they overrun the merge's own bound.
    assert!(PER_PARTITION as usize <= LIMIT + SCORE_FLOOR_BUFFER);
    assert!(
        (PARTITIONS * PER_PARTITION) as usize > LIMIT + SCORE_FLOOR_BUFFER,
        "premise: the partitions together must overrun the merge's band bound"
    );
    assert_eq!(
        search_alpha(&index, LIMIT).await,
        vec![0, 1, 2, 3],
        "compacting the band must not change the exact top-k"
    );
}

#[tokio::test]
async fn test_fts_topk_tied_scores_cold_and_warm_searches_agree() {
    // A cache that keeps nothing: the partition's row addresses are not resident
    // when its walk starts, so the first pass cannot rank the corpus-wide tie and
    // stops at the first boundary tie. The retry holds the addresses, ranks the
    // ties by row address and prunes every tied document behind the k-th. A
    // prefilter builds visibility from those addresses, so that walk has them from
    // the start. Every variant must return the same exact top-k.
    const NUM_DOCS: u64 = 1_024;
    const LIMIT: usize = 10;
    let row_ids = (0..NUM_DOCS).collect::<Vec<_>>();
    let expected = (0..LIMIT as u64).collect::<Vec<_>>();

    let (_tmpdir, index) =
        load_tied_index_with(&[(0, row_ids.clone())], &LanceCache::no_cache()).await;
    assert!(
        !index.has_resident_document_projections(),
        "premise: the first search must start without resident addresses"
    );
    assert_eq!(
        search_alpha(&index, LIMIT).await,
        expected,
        "the address-ordered retry must return the exact top-k"
    );
    let top3 = search_alpha(&index, 3).await;
    assert_eq!(top3, vec![0, 1, 2]);
    assert_eq!(top3, expected[..3], "top-3 must be a prefix of top-10");

    let (_prefiltered_tmpdir, prefiltered) = load_tied_index(&[(0, row_ids.clone())]).await;
    let mask = Arc::new(MaskPreFilter {
        mask: Arc::new(RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(
            row_ids,
        ))),
    });
    assert_eq!(
        search_alpha_with(&prefiltered, LIMIT, mask).await,
        expected,
        "a walk that already holds the addresses must resolve the tie exactly"
    );
    assert_eq!(
        search_alpha(&prefiltered, LIMIT).await,
        expected,
        "the same index searched unfiltered must agree"
    );
}

#[tokio::test]
async fn test_fts_topk_tied_scores_wider_than_the_collector_buffer() {
    // The tie band is far wider than the collector's buffer, and row addresses
    // descend with DocId, so the winners sit at the very end of the walk. A
    // collector that truncated its band would answer with the highest row_ids;
    // the overflow retry reloads the partition's addresses and orders exactly.
    const NUM_DOCS: u64 = 512;
    const BASE: u64 = 10_000;
    let row_ids = (0..NUM_DOCS)
        .map(|doc_id| BASE + NUM_DOCS - doc_id)
        .collect::<Vec<_>>();
    let (_tmpdir, index) = load_tied_index(&[(0, row_ids)]).await;

    assert!(
        NUM_DOCS as usize > 5 + SCORE_FLOOR_BUFFER,
        "premise: the tie band must overflow the collector's buffer"
    );
    let top3 = search_alpha(&index, 3).await;
    let top5 = search_alpha(&index, 5).await;
    assert_eq!(
        top3,
        vec![BASE + 1, BASE + 2, BASE + 3],
        "the exact lowest row_ids, which the walk only reaches at its last DocIds"
    );
    assert_eq!(top5, vec![BASE + 1, BASE + 2, BASE + 3, BASE + 4, BASE + 5]);
    assert_eq!(top3, top5[..3], "top-3 must be a prefix of top-5");
}

/// Counts reads of the element coordinate columns across every partition.
#[derive(Default, Debug)]
struct CoordinateReadCounter {
    reads: AtomicUsize,
}

struct CountingCoordinateReader {
    inner: Arc<dyn IndexReader>,
    counter: Arc<CoordinateReadCounter>,
}

#[cfg_attr(coverage, coverage(off))]
#[async_trait]
impl IndexReader for CountingCoordinateReader {
    async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
        self.inner.read_record_batch(n, batch_size).await
    }
    async fn read_global_buffer(&self, index: u32) -> Result<bytes::Bytes> {
        self.inner.read_global_buffer(index).await
    }
    async fn read_range(
        &self,
        range: std::ops::Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        if projection.is_some_and(|columns| columns.contains(&doc_index_storage_column(0).as_str()))
        {
            self.counter.reads.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.read_range(range, projection).await
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

#[derive(Debug)]
struct CountingCoordinateStore {
    inner: Arc<dyn IndexStore>,
    counter: Arc<CoordinateReadCounter>,
}

#[cfg_attr(coverage, coverage(off))]
impl DeepSizeOf for CountingCoordinateStore {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.inner.deep_size_of_children(context)
    }
}

#[cfg_attr(coverage, coverage(off))]
#[async_trait]
impl IndexStore for CountingCoordinateStore {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_arc(&self) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.clone(),
            counter: self.counter.clone(),
        })
    }
    fn io_parallelism(&self) -> usize {
        self.inner.io_parallelism()
    }
    fn with_io_priority(&self, io_priority: u64) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.with_io_priority(io_priority),
            counter: self.counter.clone(),
        })
    }
    fn with_io_buffer_size(&self, bytes: u64) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.with_io_buffer_size(bytes),
            counter: self.counter.clone(),
        })
    }
    async fn new_index_file(
        &self,
        name: &str,
        schema: Arc<arrow_schema::Schema>,
    ) -> Result<Box<dyn crate::scalar::IndexWriter>> {
        self.inner.new_index_file(name, schema).await
    }
    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
        Ok(Arc::new(CountingCoordinateReader {
            inner: self.inner.open_index_file(name).await?,
            counter: self.counter.clone(),
        }))
    }
    async fn copy_index_file(
        &self,
        name: &str,
        dest_store: &dyn IndexStore,
    ) -> Result<crate::scalar::IndexFile> {
        self.inner.copy_index_file(name, dest_store).await
    }
    async fn copy_index_file_to(
        &self,
        name: &str,
        new_name: &str,
        dest_store: &dyn IndexStore,
    ) -> Result<crate::scalar::IndexFile> {
        self.inner
            .copy_index_file_to(name, new_name, dest_store)
            .await
    }
    async fn rename_index_file(
        &self,
        name: &str,
        new_name: &str,
    ) -> Result<crate::scalar::IndexFile> {
        self.inner.rename_index_file(name, new_name).await
    }
    async fn delete_index_file(&self, name: &str) -> Result<()> {
        self.inner.delete_index_file(name).await
    }
    async fn list_files_with_sizes(&self) -> Result<Vec<crate::scalar::IndexFile>> {
        self.inner.list_files_with_sizes().await
    }
}

#[tokio::test]
async fn test_element_fts_resolves_coordinates_once_per_partition() {
    // The merge compacts its tie band after every partition once past the bound,
    // and each compaction re-resolves the candidates of every partition merged so
    // far. Resolving coordinates straight from the docs file made that quadratic
    // in the number of partitions, so the columns are read once and cached.
    const PARTITIONS: u64 = 8;
    const PER_PARTITION: u64 = 100;
    let tmpdir = TempObjDir::default();
    let inner = Arc::new(LanceIndexStore::new(
        ObjectStore::local().into(),
        tmpdir.clone(),
        Arc::new(LanceCache::no_cache()),
    ));
    for partition_id in 0..PARTITIONS {
        let elements = (0..PER_PARTITION)
            .map(|element| (partition_id * 1_000 + element, element as u32))
            .collect::<Vec<_>>();
        build_tied_element_partition(&inner, partition_id, &elements).await;
    }
    write_test_metadata(
        &inner,
        (0..PARTITIONS).collect(),
        InvertedIndexParams::default(),
    )
    .await;

    let counter = Arc::new(CoordinateReadCounter::default());
    let store: Arc<dyn IndexStore> = Arc::new(CountingCoordinateStore {
        inner: inner.clone(),
        counter: counter.clone(),
    });
    let cache = LanceCache::with_capacity(64 * 1024 * 1024);
    let index = InvertedIndex::load(store, None, &cache).await.unwrap();

    let documents = search_alpha_documents(&index, 4).await;
    assert_eq!(
        documents,
        vec![(0, vec![0]), (1, vec![1]), (2, vec![2]), (3, vec![3])],
        "the exact lowest row_ids, ordered by row_id then doc_index"
    );
    let reads = counter.reads.load(Ordering::Relaxed);
    assert!(
        reads <= PARTITIONS as usize,
        "coordinates must be read once per partition, got {reads} reads for {PARTITIONS} partitions"
    );
}

#[tokio::test]
async fn test_fts_topk_tied_scores_retry_every_overflowed_partition() {
    // Every partition overflows its tie band at once, which is what a broad
    // single-term query over quantized doc lengths looks like. All of them are
    // rescored against resolved addresses, concurrently, and the answer is still
    // the exact lowest row_ids.
    const PARTITIONS: u64 = 5;
    const PER_PARTITION: u64 = 200;
    const BASE: u64 = 10_000;
    let partitions = (0..PARTITIONS)
        .map(|partition_id| {
            // Row addresses descend with DocId, so the winners sit at the end of
            // each walk, past the point where the band overflows.
            let row_ids = (0..PER_PARTITION)
                .map(|doc_id| BASE + partition_id + (PER_PARTITION - doc_id) * PARTITIONS)
                .collect::<Vec<_>>();
            (partition_id, row_ids)
        })
        .collect::<Vec<_>>();
    let (_tmpdir, index) = load_tied_index(&partitions).await;

    assert!(
        PER_PARTITION as usize > 5 + SCORE_FLOOR_BUFFER,
        "premise: every partition's tie band must overflow the collector's buffer"
    );
    let top5 = search_alpha(&index, 5).await;
    // The lowest addresses are the last DocId of each partition, one per
    // partition: BASE + partition_id + PARTITIONS.
    let expected = (0..PARTITIONS)
        .map(|partition_id| BASE + partition_id + PARTITIONS)
        .collect::<Vec<_>>();
    assert_eq!(top5, expected);
    assert_eq!(search_alpha(&index, 3).await, expected[..3]);
}
