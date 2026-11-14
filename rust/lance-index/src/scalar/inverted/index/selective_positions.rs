// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental bounded position-block reads for standalone exact phrases.

use super::*;
use crate::scalar::IndexBinaryReader;
use crate::scalar::inverted::wand::selective_phrase::{
    PhraseProgress, PhraseSearch, PositionWindow,
};

mod read_ahead;
use read_ahead::ReadAhead;

// Kept opt-in until exact-head CI and matched low-cache VM qualification.
static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_SELECTIVE_POSITIONS").is_ok_and(|value| value == "1")
});

pub(super) fn enabled() -> bool {
    *ENABLED
}

// This bounds encoded blocks held by one partition search, not process RSS.
// Postings, offsets, decode scratch, I/O buffers and other queries are separate.
const MAX_LIVE_WINDOW_BYTES: u64 = 32 * 1024 * 1024;
// Search seed for the opt-in path, pending matched VM qualification.
const POSITION_WINDOW_BYTES: u64 = 2 * 1024 * 1024;
// Bound speculative request metadata even for arbitrarily small codec windows.
// Encoded bytes have the stricter current-plus-predicted clause allowance.
const MAX_PREDICTED_WINDOWS: usize = 7;
const MAX_CLAUSES: usize = 16;
const MAX_OFFSET_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
struct PositionOffsets(Arc<[u32]>);

impl DeepSizeOf for PositionOffsets {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        std::mem::size_of_val(self.0.as_ref()) + 2 * std::mem::size_of::<usize>()
    }
}

#[derive(Debug)]
struct PositionOffsetsKey(u32);

impl CacheKey for PositionOffsetsKey {
    type ValueType = Option<PositionOffsets>;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("position-offsets-{}", self.0).into()
    }

    fn type_name() -> &'static str {
        "PositionOffsets"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.inverted.position-offsets", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.0);
    }
}

impl DeepSizeOf for PositionWindow {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.bytes.len()
            + std::mem::size_of_val(self.offsets.as_ref())
            + 4 * std::mem::size_of::<usize>()
    }
}

#[derive(Debug)]
struct PositionWindowKey {
    token_id: u32,
    first_block: usize,
}
impl CacheKey for PositionWindowKey {
    type ValueType = PositionWindow;
    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!(
            "position-window-{}-{}-{POSITION_WINDOW_BYTES}",
            self.token_id, self.first_block
        )
        .into()
    }
    fn type_name() -> &'static str {
        "EncodedPositionWindow"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.inverted.position-window", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.token_id);
        builder.write_u64(self.first_block as u64);
        builder.write_u64(POSITION_WINDOW_BYTES);
    }
}

/// Greedy byte groups are deterministic for this immutable term. Never split
/// a codec block, including a final block whose endpoint is the value length.
fn position_window_ends(
    offsets: &[u32],
    length: u64,
    target: u64,
    max_block: u64,
) -> Result<Option<Vec<usize>>> {
    if target == 0 || target > max_block {
        return Err(Error::invalid_input(
            "position window target must be positive and fit the clause budget",
        ));
    }
    if offsets.first().is_some_and(|start| *start != 0) {
        return Err(Error::index("position window offsets must start at zero"));
    }
    let mut ends = Vec::new();
    let mut first = 0;
    for (index, start) in offsets.iter().enumerate() {
        let end = offsets.get(index + 1).map_or(length, |end| u64::from(*end));
        if u64::from(*start) > end || end > length {
            return Err(Error::index(
                "position window offsets exceed the value length or decrease",
            ));
        }
        if end - u64::from(*start) > max_block {
            return Ok(None);
        }
        if index > first && end - u64::from(offsets[first]) > target {
            ends.push(index);
            first = index;
        }
    }
    if !offsets.is_empty() {
        ends.push(offsets.len());
    }
    Ok(Some(ends))
}

/// Request-local preparation. Never put the statistics-bound I/O handle in the cache.
struct PreparedPositions {
    token_id: u32,
    codec: PositionStreamCodec,
    offsets: Arc<[u32]>,
    window_ends: Vec<usize>,
    value: Arc<dyn IndexBinaryReader>,
    cache: WeakLanceCache,
}

impl PreparedPositions {
    fn block_range(&self, index: usize) -> Result<Range<u64>> {
        let start = self.offsets.get(index).copied().ok_or_else(|| {
            Error::index(format!(
                "position block {index} is outside {} blocks for token {}",
                self.offsets.len(),
                self.token_id
            ))
        })?;
        let end = self
            .offsets
            .get(index + 1)
            .map_or(self.value.len(), |end| u64::from(*end));
        if u64::from(start) > end || end > self.value.len() {
            return Err(Error::index(format!(
                "invalid position block range {start}..{end} for {} bytes",
                self.value.len()
            )));
        }
        Ok(u64::from(start)..end)
    }

    fn window_bounds(&self, index: usize) -> Result<(usize, usize, Range<u64>)> {
        self.block_range(index)?;
        let ordinal = self.window_ends.partition_point(|end| *end <= index);
        let end = self
            .window_ends
            .get(ordinal)
            .copied()
            .ok_or_else(|| Error::index("position block has no byte window"))?;
        let first = ordinal
            .checked_sub(1)
            .map_or(0, |prior| self.window_ends[prior]);
        let range = self.block_range(first)?.start..self.block_range(end - 1)?.end;
        Ok((first, end, range))
    }

    fn read_ahead_blocks(
        &self,
        current: &PositionWindow,
        clause_budget: u64,
    ) -> Result<Vec<Range<usize>>> {
        if current.offsets.is_empty() {
            return Err(Error::index(
                "cannot prefetch after an empty position window",
            ));
        }
        let mut next = current
            .first_block
            .checked_add(current.offsets.len())
            .ok_or_else(|| Error::index("position window block endpoint overflows"))?;
        let mut retained = current.bytes.len() as u64;
        let mut predictions = Vec::with_capacity(MAX_PREDICTED_WINDOWS);
        for _ in 0..MAX_PREDICTED_WINDOWS {
            if next >= self.offsets.len() {
                break;
            }
            let (first, end, range) = self.window_bounds(next)?;
            let Some(bytes) = retained.checked_add(range.end - range.start) else {
                break;
            };
            if bytes > clause_budget {
                break;
            }
            predictions.push(first..end);
            retained = bytes;
            next = end;
        }
        Ok(predictions)
    }

    async fn read(&self, index: usize, metrics: &dyn MetricsCollector) -> Result<PositionWindow> {
        let (first, end, range) = self.window_bounds(index)?;
        let expected = range.end - range.start;
        let start = range.start;
        let (window, hit) = self
            .cache
            .get_or_insert_with_key_hit(
                PositionWindowKey {
                    token_id: self.token_id,
                    first_block: first,
                },
                || async {
                    let mut buffers = self.value.read_slices(&[range]).await?;
                    if buffers.len() != 1 || buffers[0].len() as u64 != expected {
                        return Err(Error::index(
                            "selective position block read returned an unexpected buffer length",
                        ));
                    }
                    let bytes = buffers.remove(0);
                    let offsets = self.offsets[first..end]
                        .iter()
                        .map(|offset| {
                            u32::try_from(u64::from(*offset) - start)
                                .map_err(|_| Error::index("position window offset exceeds u32"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(PositionWindow {
                        first_block: first,
                        codec: self.codec,
                        offsets: Arc::from(offsets.into_boxed_slice()),
                        bytes: Arc::from(bytes.as_ref()),
                    })
                },
            )
            .await?;
        if hit {
            metrics.record_index_cache_hit();
        } else {
            metrics.record_index_cache_miss();
        }
        Ok(window.as_ref().clone())
    }
}

impl InvertedPartition {
    async fn prepare_selective_positions(
        &self,
        tokens: &Tokens,
        term_ids: Option<PartitionTermIds<'_>>,
    ) -> Result<Option<Vec<PreparedPositions>>> {
        let inverted = &self.inverted_list;
        let PositionsLayout::SharedStream(codec) = inverted.positions_layout else {
            return Ok(None);
        };
        if inverted.is_legacy_layout() {
            return Ok(None);
        }
        let reader = inverted.reader.get().await?;
        let offset_cache = reader
            .metadata_cache()
            .unwrap_or_else(|| inverted.index_cache.clone());
        let mut prepared = Vec::with_capacity(tokens.len());
        // At most 16 clauses; preparation is sequential so metadata I/O is
        // bounded independently of an object store's potentially large limit.
        for token_index in 0..tokens.len() {
            let token_id = match &term_ids {
                Some(ids) => ids.token_id(token_index),
                None => self.map(tokens.get_token(token_index)),
            }
            .ok_or_else(|| Error::internal("selected phrase partition lost a dictionary term"))?;
            if prepared
                .iter()
                .any(|value: &PreparedPositions| value.token_id == token_id)
            {
                continue;
            }
            let Some(value) = reader
                .try_prepare_binary_value(COMPRESSED_POSITION_COL, u64::from(token_id))
                .await?
            else {
                return Ok(None);
            };
            // Keep small immutable offsets in the existing file metadata budget,
            // independent of position payload churn. Readers without that tier
            // retain the index-cache fallback. I/O scopes stay request-local.
            let offsets = offset_cache
                .get_or_insert_with_key(PositionOffsetsKey(token_id), || async {
                    let batch = reader
                        .read_range(
                            token_id as usize..token_id as usize + 1,
                            Some(&[POSITION_BLOCK_OFFSET_COL]),
                        )
                        .await?;
                    let offsets = batch[POSITION_BLOCK_OFFSET_COL].as_list::<i32>().value(0);
                    let offsets = offsets.as_primitive::<UInt32Type>().values();
                    if offsets.len() > MAX_OFFSET_BYTES / std::mem::size_of::<u32>() {
                        return Ok(None);
                    }
                    if offsets.first().is_some_and(|start| *start != 0)
                        || offsets.windows(2).any(|pair| pair[0] > pair[1])
                    {
                        return Err(Error::index(
                            "selective position block offsets are not monotonic from zero",
                        ));
                    }
                    Ok(Some(PositionOffsets(Arc::from(
                        offsets.to_vec().into_boxed_slice(),
                    ))))
                })
                .await?;
            let Some(offsets) = offsets.as_ref() else {
                return Ok(None);
            };
            let Some(window_ends) = position_window_ends(
                &offsets.0,
                value.len(),
                POSITION_WINDOW_BYTES,
                MAX_LIVE_WINDOW_BYTES / tokens.len() as u64,
            )?
            else {
                return Ok(None);
            };
            let plan = PreparedPositions {
                token_id,
                codec,
                offsets: offsets.0.clone(),
                window_ends,
                value,
                cache: inverted.index_cache.clone(),
            };
            prepared.push(plan);
        }
        Ok(Some(prepared))
    }
}

fn eligible(request: &ModernSearchRequest<'_>) -> bool {
    request.operator == Operator::And
        && request.params.phrase_slop == Some(0)
        && request.params.wand_factor == 1.0
        && request.params.limit.is_some_and(|limit| limit > 0)
        && (2..=MAX_CLAUSES).contains(&request.tokens.len())
        && (0..request.tokens.len()).all(|index| {
            (0..index).all(|prior| {
                request.tokens.position_for_query(prior, true)
                    != request.tokens.position_for_query(index, true)
            })
        })
}

impl InvertedIndex {
    pub(super) fn try_selective_phrase_search<'a>(
        &'a self,
        request: ModernSearchRequest<'a>,
    ) -> futures::future::BoxFuture<'a, Result<Option<Vec<Reverse<ScoredPartitionDoc>>>>> {
        async move {
            if !eligible(&request) {
                return Ok(None);
            }
            // Decide all encoding support before loading/consuming postings or
            // publishing a competitive floor. A None cleanly selects ordinary search.
            let mut preparations = Vec::new();
            for (ordinal, part) in self.partitions.iter().enumerate() {
                if !part.may_match_tokens(&request.tokens, request.operator, true) {
                    continue;
                }
                let part = part.clone();
                let tokens = request.tokens.clone();
                let ids = request.term_ids.map(|ids| ids.partition(ordinal));
                preparations.push(
                    async move {
                        if part.docs.modern().is_none() {
                            return Ok(None);
                        }
                        Ok::<_, Error>(
                            part.prepare_selective_positions(&tokens, ids)
                                .await?
                                .map(|positions| (ordinal, part.clone(), positions)),
                        )
                    }
                    .boxed(),
                );
            }
            let prepared = stream::iter(preparations)
                .buffered(4)
                .try_collect::<Vec<_>>()
                .await?;
            let Some(prepared) = prepared.into_iter().collect::<Option<Vec<_>>>() else {
                return Ok(None);
            };
            let shared = Arc::new(AtomicU32::new(
                request
                    .initial_score_floor
                    .unwrap_or(f32::NEG_INFINITY)
                    .to_bits(),
            ));
            let searches = prepared
                .into_iter()
                .map(|(ordinal, part, positions)| {
                    let shared = shared.clone();
                    // Own the request across suspension; erase the future here
                    // rather than borrowing through the outer stream closure.
                    let request = request.clone();
                    async move {
                        let documents = part.docs.modern().ok_or_else(|| {
                            Error::internal("selective phrase partition became legacy")
                        })?;
                        let visibility = documents.visibility(request.mask.clone(), false).await?;
                        if !visibility.may_match().await? {
                            return Result::Ok((ordinal, Vec::new()));
                        }
                        let lengths = documents.lengths().await?;
                        let ids = request.term_ids.map(|ids| ids.partition(ordinal));
                        let Some(fetched) = part
                            .fetch_posting_lists(
                                &request.tokens,
                                &request.params,
                                request.operator,
                                request.metrics.as_ref(),
                                PostingLoadOptions::deferred_positions(),
                                ids,
                            )
                            .await?
                        else {
                            return Result::Ok((ordinal, Vec::new()));
                        };
                        let tokens = request.tokens.clone();
                        let scorer = request.impact_scorer.clone();
                        let limit = request.params.limit.ok_or_else(|| {
                            Error::internal("selective phrase lost its finite limit")
                        })?;
                        let (mut search, idf) = spawn_cpu(move || {
                            let loaded = fetched.into_loaded(&tokens, scorer.as_ref())?;
                            let idf = idf_by_position(&loaded.postings, scorer.as_ref());
                            Ok::<_, Error>((PhraseSearch::new(loaded.postings, limit)?, idf))
                        })
                        .await?;
                        let mut read_ahead = ReadAhead::new(positions.len());
                        let clause_budget = MAX_LIVE_WINDOW_BYTES / request.tokens.len() as u64;
                        loop {
                            let lengths = lengths.clone();
                            let visibility = visibility.clone();
                            let scorer = request.impact_scorer.clone();
                            let shared = shared.clone();
                            let metrics = request.metrics.clone();
                            let advance = spawn_cpu(move || {
                                let docs = ModernWandDocuments::filtered(&lengths, &visibility);
                                let progress =
                                    search.advance(&docs, scorer, shared, metrics.as_ref())?;
                                Ok::<_, Error>((search, progress))
                            });
                            let (next, progress) = read_ahead.drive(advance).await?;
                            search = next;
                            match progress {
                                PhraseProgress::Done(candidates) => {
                                    read_ahead.clear();
                                    drop(search);
                                    let scorer = request.impact_scorer.clone();
                                    let scored = spawn_cpu(move || {
                                        Ok::<_, Error>(rescore_partition_candidates(
                                            PartitionCandidates {
                                                idf_by_position: idf,
                                                grouped_expansions: Vec::new(),
                                                candidates,
                                            },
                                            scorer.as_ref(),
                                        ))
                                    })
                                    .await?;
                                    return Result::Ok((ordinal, scored));
                                }
                                PhraseProgress::Read(requests) => {
                                    let mut reads = Vec::with_capacity(requests.len());
                                    for block_request in requests {
                                        let (term, plan) = positions
                                            .iter()
                                            .enumerate()
                                            .find(|(_, plan)| {
                                                plan.token_id == block_request.token_id
                                            })
                                            .ok_or_else(|| {
                                                Error::internal(
                                                    "position request has no prepared term",
                                                )
                                            })?;
                                        let metrics = request.metrics.as_ref();
                                        let read = read_ahead
                                            .take(term, block_request.index)
                                            .unwrap_or_else(|| {
                                                plan.read(block_request.index, metrics).boxed()
                                            });
                                        reads.push(async move {
                                            Ok::<_, Error>((term, block_request, read.await?))
                                        });
                                    }
                                    let windows = read_ahead
                                        .drive(futures::future::try_join_all(reads))
                                        .await?;
                                    for (term, block_request, window) in windows {
                                        let plan = &positions[term];
                                        // Current plus speculative encoded bytes share the
                                        // original clause budget. Oversized neighbors skip
                                        // speculation; offsets/scratch/cache/I/O are separate.
                                        let predictions =
                                            plan.read_ahead_blocks(&window, clause_budget)?;
                                        read_ahead.schedule(
                                            term,
                                            predictions.into_iter().map(|blocks| {
                                                let future = plan
                                                    .read(blocks.start, request.metrics.as_ref())
                                                    .boxed();
                                                (blocks, future)
                                            }),
                                        )?;
                                        search.supply(block_request, window)?;
                                    }
                                }
                            }
                        }
                    }
                    .boxed()
                })
                .collect::<Vec<_>>();
            let mut results =
                stream::iter(searches).buffer_unordered(get_num_compute_intensive_cpus().min(4));
            let mut ranked = BinaryHeap::new();
            while let Some((ordinal, scored)) = results.try_next().await? {
                for (doc_id, score) in scored {
                    push_scored_partition_doc(
                        &mut ranked,
                        request.limit,
                        PartitionDocId::try_new(ordinal, doc_id)?,
                        score,
                    );
                }
            }
            Ok(Some(ranked.into_sorted_vec()))
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::AtomicUsize;

    #[derive(Debug)]
    struct BinaryValue {
        calls: AtomicUsize,
        short: bool,
        expected_range: Range<u64>,
        pending: bool,
        active: Arc<AtomicUsize>,
        started: tokio::sync::Notify,
        backing: std::sync::Mutex<Option<std::sync::Weak<()>>>,
    }

    struct ActiveRead(Arc<AtomicUsize>);
    impl Drop for ActiveRead {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct Backing {
        bytes: Vec<u8>,
        _lifetime: Arc<()>,
    }
    impl AsRef<[u8]> for Backing {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }

    #[async_trait]
    impl IndexBinaryReader for BinaryValue {
        fn len(&self) -> u64 {
            16
        }
        async fn read_slices(&self, slices: &[Range<u64>]) -> Result<Vec<bytes::Bytes>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.active.fetch_add(1, Ordering::SeqCst);
            let _active = ActiveRead(self.active.clone());
            self.started.notify_one();
            if self.pending {
                return futures::future::pending().await;
            }
            assert_eq!(slices, std::slice::from_ref(&self.expected_range));
            let lifetime = Arc::new(());
            *self.backing.lock().unwrap() = Some(Arc::downgrade(&lifetime));
            let bytes = bytes::Bytes::from_owner(Backing {
                bytes: vec![9; 4096],
                _lifetime: lifetime,
            });
            let range = self.expected_range.start as usize
                ..self.expected_range.end as usize - usize::from(self.short);
            Ok(vec![bytes.slice(range)])
        }
    }

    fn binary_value(short: bool, pending: bool) -> Arc<BinaryValue> {
        Arc::new(BinaryValue {
            calls: AtomicUsize::new(0),
            short,
            expected_range: 12..16,
            pending,
            active: Arc::new(AtomicUsize::new(0)),
            started: tokio::sync::Notify::new(),
            backing: std::sync::Mutex::new(None),
        })
    }

    fn prepared(value: Arc<BinaryValue>, cache: &LanceCache) -> PreparedPositions {
        PreparedPositions {
            token_id: 7,
            codec: PositionStreamCodec::PackedDelta,
            offsets: vec![0, 4, 12].into(),
            window_ends: vec![1, 2, 3],
            value,
            cache: WeakLanceCache::from(cache),
        }
    }

    #[rstest::rstest]
    #[case::greedy(vec![0, 4, 7, 12], 16, 8, 8, Some(vec![2, 3, 4]))]
    #[case::large_single_block(vec![0, 6, 9], 10, 4, 8, Some(vec![1, 3]))]
    #[case::fallback(vec![0, 9], 10, 4, 8, None)]
    #[case::empty(vec![], 0, 4, 8, Some(vec![]))]
    #[case::empty_blocks(vec![0, 0, 0], 0, 4, 8, Some(vec![3]))]
    fn selective_phrase_window_plan_respects_bytes(
        #[case] offsets: Vec<u32>,
        #[case] length: u64,
        #[case] target: u64,
        #[case] max_block: u64,
        #[case] expected: Option<Vec<usize>>,
    ) {
        assert_eq!(
            position_window_ends(&offsets, length, target, max_block).unwrap(),
            expected
        );
    }

    #[rstest::rstest]
    #[case::nonzero_start(vec![1], 8)]
    #[case::decreasing(vec![0, 4, 3], 8)]
    #[case::outside_value(vec![0, 9], 8)]
    fn selective_phrase_window_plan_rejects_invalid_offsets(
        #[case] offsets: Vec<u32>,
        #[case] length: u64,
    ) {
        let error = position_window_ends(&offsets, length, 4, 8).unwrap_err();
        assert!(matches!(error, Error::Index { .. }));
    }

    #[tokio::test]
    async fn selective_phrase_prefetched_neighbors_share_one_read() -> Result<()> {
        let cache = LanceCache::with_capacity(4096);
        let mut value = binary_value(false, false);
        Arc::get_mut(&mut value).unwrap().expected_range = 0..16;
        let mut plan = prepared(value.clone(), &cache);
        plan.window_ends = vec![3];
        let window = plan.read(0, &NoOpMetricsCollector).await?;
        assert_eq!(window.block(0)?, &[9; 4]);
        assert_eq!(window.block(1)?, &[9; 8]);
        assert_eq!(window.block(2)?, &[9; 4]);
        assert!(!window.contains(3));
        assert!(matches!(window.block(3), Err(Error::Index { .. })));
        let neighbor = plan.read(2, &NoOpMetricsCollector).await?;
        assert!(Arc::ptr_eq(&window.bytes, &neighbor.bytes));
        assert_eq!(value.calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn selective_phrase_reads_last_block_compacts_and_reuses_cache() -> Result<()> {
        let cache = LanceCache::with_capacity(4096);
        let value = binary_value(false, false);
        let plan = prepared(value.clone(), &cache);
        let block = plan.read(2, &NoOpMetricsCollector).await?;
        assert_eq!(block.bytes.as_ref(), &[9; 4]);
        assert!(
            value
                .backing
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .upgrade()
                .is_none(),
            "cached block must release oversized custom backing"
        );
        let second = plan.read(2, &NoOpMetricsCollector).await?;
        assert!(Arc::ptr_eq(&block.bytes, &second.bytes));
        assert_eq!(value.calls.load(Ordering::SeqCst), 1);
        assert!(matches!(plan.block_range(3), Err(Error::Index { .. })));
        assert_eq!(value.calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[rstest::rstest]
    #[case::moka(LanceCache::with_capacity(4096))]
    #[case::quick(LanceCache::with_backend(Arc::new(
        lance_core::cache::QuickCacheBackend::with_capacity(4096)
    )))]
    #[case::disabled(LanceCache::no_cache())]
    #[tokio::test]
    async fn selective_phrase_short_payload_is_an_error_not_a_cache_entry(
        #[case] cache: LanceCache,
    ) {
        let value = binary_value(true, false);
        let plan = prepared(value.clone(), &cache);
        for _ in 0..2 {
            let Err(error) = plan.read(2, &NoOpMetricsCollector).await else {
                panic!("short read accepted");
            };
            // Moka can return Shared while another caller retains the same
            // loader failure. The root kind and message remain the contract.
            assert!(
                matches!(error.find_root(), Error::Index { .. }),
                "{error:?}"
            );
            assert!(error.to_string().contains("unexpected buffer length"));
        }
        assert_eq!(value.calls.load(Ordering::SeqCst), 2);
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
    }

    #[rstest::rstest]
    #[case::moka(LanceCache::with_capacity(4096))]
    #[case::quick(LanceCache::with_backend(Arc::new(
        lance_core::cache::QuickCacheBackend::with_capacity(4096)
    )))]
    #[case::disabled(LanceCache::no_cache())]
    #[tokio::test]
    async fn selective_phrase_cancellation_drops_pending_io(#[case] cache: LanceCache) {
        let value = binary_value(false, true);
        let plan = prepared(value.clone(), &cache);
        let metrics = NoOpMetricsCollector;
        let mut read = Box::pin(plan.read(2, &metrics));
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            tokio::select! {
                _ = &mut read => panic!("pending I/O completed before cancellation"),
                _ = value.started.notified() => {}
            }
        })
        .await
        .unwrap();
        assert_eq!(value.active.load(Ordering::SeqCst), 1);
        drop(read);
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn selective_phrase_read_ahead_shares_the_current_window_budget() {
        let cache = LanceCache::no_cache();
        let plan = prepared(binary_value(false, false), &cache);
        let current = PositionWindow {
            first_block: 0,
            codec: PositionStreamCodec::PackedDelta,
            offsets: vec![0].into(),
            bytes: vec![9; 4].into(),
        };
        // The current four bytes and next eight bytes must both fit.
        assert!(plan.read_ahead_blocks(&current, 11).unwrap().is_empty());
        let one = plan.read_ahead_blocks(&current, 12).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0], 1..2);
        assert_eq!(
            plan.read_ahead_blocks(&current, 16).unwrap(),
            vec![1..2, 2..3]
        );
        let final_window = PositionWindow {
            first_block: 2,
            ..current
        };
        assert!(
            plan.read_ahead_blocks(&final_window, 16)
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn selective_phrase_read_ahead_reuses_a_completed_read() -> Result<()> {
        let cache = LanceCache::no_cache();
        let value = binary_value(false, false);
        let plan = prepared(value.clone(), &cache);
        let metrics = NoOpMetricsCollector;
        let mut ahead = ReadAhead::new(1);
        ahead.schedule(0, [(2..3, plan.read(2, &metrics).boxed())])?;
        ahead.drive(value.started.notified()).await;
        let window = ahead.take(0, 2).unwrap().await?;
        assert_eq!(window.block(2)?, &[9; 4]);
        assert!(ahead.take(0, 2).is_none());
        assert_eq!(value.calls.load(Ordering::SeqCst), 1);
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[test]
    fn selective_phrase_read_ahead_limits_tiny_window_predictions() {
        let cache = LanceCache::no_cache();
        let mut plan = prepared(binary_value(false, false), &cache);
        plan.offsets = vec![0; 100].into();
        plan.window_ends = (1..=100).collect();
        let current = PositionWindow {
            first_block: 0,
            codec: PositionStreamCodec::PackedDelta,
            offsets: vec![0].into(),
            bytes: Arc::from([]),
        };
        let predictions = plan.read_ahead_blocks(&current, 32 * 1024 * 1024).unwrap();
        assert_eq!(predictions.len(), MAX_PREDICTED_WINDOWS);
        assert_eq!(predictions.first(), Some(&(1..2)));
        assert_eq!(predictions.last(), Some(&(7..8)));
    }

    #[tokio::test]
    async fn selective_phrase_read_ahead_preserves_remaining_windows() -> Result<()> {
        let calls = Arc::new(AtomicUsize::new(0));
        let predict = |block, calls: Arc<AtomicUsize>| {
            (
                block..block + 1,
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Ok(PositionWindow {
                        first_block: block,
                        codec: PositionStreamCodec::PackedDelta,
                        offsets: vec![0].into(),
                        bytes: vec![block as u8].into(),
                    })
                }
                .boxed(),
            )
        };
        let mut ahead = ReadAhead::new(1);
        ahead.schedule(0, [predict(1, calls.clone()), predict(2, calls.clone())])?;
        // One pending essential poll allows both requests to complete.
        let mut pending = true;
        ahead
            .drive(futures::future::poll_fn(|context| {
                if std::mem::take(&mut pending) {
                    context.waker().wake_by_ref();
                    std::task::Poll::Pending
                } else {
                    std::task::Poll::Ready(())
                }
            }))
            .await;
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(ahead.take(0, 1).unwrap().await?.block(1)?, &[1]);
        ahead.schedule(0, [predict(2, calls.clone()), predict(3, calls.clone())])?;
        assert_eq!(ahead.take(0, 2).unwrap().await?.block(2)?, &[2]);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        // A later skip discards the unused third prediction without polling it.
        assert!(ahead.take(0, 4).is_none());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn selective_phrase_read_ahead_retains_pending_io_across_admission() -> Result<()> {
        let (started, accepted) = tokio::sync::oneshot::channel();
        let (complete, receive) = tokio::sync::oneshot::channel();
        let mut ahead = ReadAhead::new(1);
        ahead.schedule(
            0,
            [(
                2..3,
                async move {
                    started.send(()).unwrap();
                    receive.await.unwrap();
                    Ok(PositionWindow {
                        first_block: 2,
                        codec: PositionStreamCodec::PackedDelta,
                        offsets: vec![0].into(),
                        bytes: vec![2].into(),
                    })
                }
                .boxed(),
            )],
        )?;
        ahead.drive(accepted).await.unwrap();
        ahead.schedule(
            0,
            [(
                2..3,
                async { Err(Error::index("replacement must not restart accepted I/O")) }.boxed(),
            )],
        )?;
        assert!(!complete.is_closed());
        complete.send(()).unwrap();
        assert_eq!(ahead.take(0, 2).unwrap().await?.block(2)?, &[2]);
        Ok(())
    }

    #[rstest::rstest]
    #[case::moka(LanceCache::with_capacity(4096))]
    #[case::quick(LanceCache::with_backend(Arc::new(
        lance_core::cache::QuickCacheBackend::with_capacity(4096)
    )))]
    #[case::disabled(LanceCache::no_cache())]
    #[tokio::test]
    async fn selective_phrase_read_ahead_releases_unused_pending_io(#[case] cache: LanceCache) {
        let value = binary_value(false, true);
        let plan = prepared(value.clone(), &cache);
        let metrics = NoOpMetricsCollector;
        let mut ahead = ReadAhead::new(1);
        ahead
            .schedule(0, [(2..3, plan.read(2, &metrics).boxed())])
            .unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            ahead.drive(value.started.notified()),
        )
        .await
        .unwrap();
        assert_eq!(value.active.load(Ordering::SeqCst), 1);
        // Skipping the predicted window drops the already accepted read.
        assert!(ahead.take(0, 0).is_none());
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
        ahead
            .schedule(0, [(2..3, plan.read(2, &metrics).boxed())])
            .unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            ahead.drive(value.started.notified()),
        )
        .await
        .unwrap();
        assert_eq!(value.active.load(Ordering::SeqCst), 1);
        // Dropping the request owner requires no join, task or sender wakeup.
        drop(ahead);
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn selective_phrase_read_ahead_does_not_start_when_essential_work_is_ready() {
        let cache = LanceCache::no_cache();
        let value = binary_value(false, true);
        let plan = prepared(value.clone(), &cache);
        let metrics = NoOpMetricsCollector;
        let mut ahead = ReadAhead::new(1);
        ahead
            .schedule(0, [(2..3, plan.read(2, &metrics).boxed())])
            .unwrap();
        assert_eq!(ahead.drive(futures::future::ready(7)).await, 7);
        ahead.clear();
        assert_eq!(value.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn selective_phrase_read_ahead_defers_an_unused_failure() -> Result<()> {
        let (send, receive) = tokio::sync::oneshot::channel();
        let mut ahead = ReadAhead::new(1);
        ahead.schedule(
            0,
            [(
                2..3,
                async move {
                    send.send(()).unwrap();
                    Err(Error::index("speculative read unavailable"))
                }
                .boxed(),
            )],
        )?;
        ahead.drive(receive).await.unwrap();
        let error = ahead.take(0, 2).unwrap().await.err().unwrap();
        assert!(error.to_string().contains("speculative read unavailable"));
        Ok(())
    }
}
