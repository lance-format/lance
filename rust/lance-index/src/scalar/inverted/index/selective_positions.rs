// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental bounded position-block reads for standalone exact phrases.

use super::*;
use crate::scalar::IndexBinaryReader;
use crate::scalar::inverted::wand::selective_phrase::{
    PhraseProgress, PhraseSearch, PositionBlock, PositionBlockRequest,
};

// Kept opt-in until exact-head CI and matched low-cache VM qualification.
static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_SELECTIVE_POSITIONS").is_ok_and(|value| value == "1")
});

pub(super) fn enabled() -> bool {
    *ENABLED
}

// This bounds encoded blocks held by one partition search, not process RSS.
// Postings, offsets, decode scratch, I/O buffers and other queries are separate.
const MAX_LIVE_BLOCK_BYTES: u64 = 8 * 1024 * 1024;
const MAX_CLAUSES: usize = 16;
const MAX_OFFSET_BYTES: usize = 1024 * 1024;

struct EncodedPositionBlock(Arc<[u8]>);

impl DeepSizeOf for EncodedPositionBlock {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        self.0.len() + 2 * std::mem::size_of::<usize>()
    }
}

#[derive(Debug)]
struct PositionBlockKey {
    token_id: u32,
    index: usize,
}

impl CacheKey for PositionBlockKey {
    type ValueType = EncodedPositionBlock;

    fn key(&self) -> std::borrow::Cow<'_, str> {
        format!("position-block-{}-{}", self.token_id, self.index).into()
    }

    fn type_name() -> &'static str {
        "EncodedPositionBlock"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.scalar.inverted.position-block", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u32(self.token_id);
        builder.write_u64(self.index as u64);
    }
}

/// Request-local preparation. Never put the statistics-bound I/O handle in the cache.
struct PreparedPositions {
    token_id: u32,
    codec: PositionStreamCodec,
    offsets: Vec<u32>,
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

    async fn read(&self, index: usize, metrics: &dyn MetricsCollector) -> Result<PositionBlock> {
        let range = self.block_range(index)?;
        let expected = range.end - range.start;
        let (block, hit) = self
            .cache
            .get_or_insert_with_key_hit(
                PositionBlockKey {
                    token_id: self.token_id,
                    index,
                },
                || async {
                    let mut buffers = self.value.read_slices(&[range]).await?;
                    if buffers.len() != 1 || buffers[0].len() as u64 != expected {
                        return Err(Error::index(
                            "selective position block read returned an unexpected buffer length",
                        ));
                    }
                    let bytes = buffers.remove(0);
                    // Copy even a uniquely visible slice: it can own a much larger
                    // coalesced buffer through a custom owner. Arc<[u8]> is compact.
                    Ok(EncodedPositionBlock(Arc::from(bytes.as_ref())))
                },
            )
            .await?;
        if hit {
            metrics.record_index_cache_hit();
        } else {
            metrics.record_index_cache_miss();
        }
        Ok(PositionBlock {
            index,
            codec: self.codec,
            bytes: block.0.clone(),
        })
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
            let plan = PreparedPositions {
                token_id,
                codec,
                offsets: offsets.to_vec(),
                value,
                cache: inverted.index_cache.clone(),
            };
            // Check every block before consuming any search state. Oversized
            // values use ordinary loading, never a partial, silently incomplete search.
            for index in 0..plan.offsets.len() {
                let range = plan.block_range(index)?;
                if range.end - range.start > MAX_LIVE_BLOCK_BYTES / tokens.len() as u64 {
                    return Ok(None);
                }
            }
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
    pub(super) async fn try_selective_phrase_search(
        &self,
        request: &ModernSearchRequest<'_>,
    ) -> Result<Option<Vec<Reverse<ScoredPartitionDoc>>>> {
        if !eligible(request) {
            return Ok(None);
        }
        // Decide all encoding support before loading/consuming postings or
        // publishing a competitive floor. A None cleanly selects ordinary search.
        let preparations = self
            .partitions
            .iter()
            .enumerate()
            .filter(|(_, part)| part.may_match_tokens(&request.tokens, request.operator, true))
            .map(|(ordinal, part)| async move {
                if part.docs.modern().is_none() {
                    return Ok(None);
                }
                let ids = request.term_ids.map(|ids| ids.partition(ordinal));
                Ok::<_, Error>(
                    part.prepare_selective_positions(&request.tokens, ids)
                        .await?
                        .map(|positions| (ordinal, part.clone(), positions)),
                )
            });
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
        let searches = prepared.into_iter().map(|(ordinal, part, positions)| {
            let shared = shared.clone();
            async move {
                let documents = part
                    .docs
                    .modern()
                    .ok_or_else(|| Error::internal("selective phrase partition became legacy"))?;
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
                let limit = request
                    .params
                    .limit
                    .ok_or_else(|| Error::internal("selective phrase lost its finite limit"))?;
                let (mut search, idf) = spawn_cpu(move || {
                    let loaded = fetched.into_loaded(&tokens, scorer.as_ref())?;
                    let idf = idf_by_position(&loaded.postings, scorer.as_ref());
                    Ok::<_, Error>((PhraseSearch::new(loaded.postings, limit)?, idf))
                })
                .await?;
                loop {
                    let lengths = lengths.clone();
                    let visibility = visibility.clone();
                    let scorer = request.impact_scorer.clone();
                    let shared = shared.clone();
                    let metrics = request.metrics.clone();
                    let (next, progress) = spawn_cpu(move || {
                        let docs = ModernWandDocuments::filtered(&lengths, &visibility);
                        let progress = search.advance(&docs, scorer, shared, metrics.as_ref())?;
                        Ok::<_, Error>((search, progress))
                    })
                    .await?;
                    search = next;
                    match progress {
                        PhraseProgress::Done(candidates) => {
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
                            // One block per clause, at most 8 MiB in aggregate.
                            // try_join_all cancellation drops accepted I/O futures;
                            // no background tasks or parked CPU workers are spawned.
                            let reads =
                                requests
                                    .into_iter()
                                    .map(|block_request: PositionBlockRequest| {
                                        let positions = &positions;
                                        async move {
                                            let plan = positions
                                                .iter()
                                                .find(|plan| {
                                                    plan.token_id == block_request.token_id
                                                })
                                                .ok_or_else(|| {
                                                    Error::internal(
                                                        "position request has no prepared term",
                                                    )
                                                })?;
                                            Ok::<_, Error>((
                                                block_request,
                                                plan.read(
                                                    block_request.index,
                                                    request.metrics.as_ref(),
                                                )
                                                .await?,
                                            ))
                                        }
                                    });
                            for (request, block) in futures::future::try_join_all(reads).await? {
                                search.supply(request, block)?;
                            }
                        }
                    }
                }
            }
        });
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
        pending: bool,
        active: Arc<AtomicUsize>,
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
            if self.pending {
                return futures::future::pending().await;
            }
            assert_eq!(slices, &[12..16]);
            let lifetime = Arc::new(());
            *self.backing.lock().unwrap() = Some(Arc::downgrade(&lifetime));
            let bytes = bytes::Bytes::from_owner(Backing {
                bytes: vec![9; 4096],
                _lifetime: lifetime,
            });
            Ok(vec![bytes.slice(12..if self.short { 15 } else { 16 })])
        }
    }

    fn binary_value(short: bool, pending: bool) -> Arc<BinaryValue> {
        Arc::new(BinaryValue {
            calls: AtomicUsize::new(0),
            short,
            pending,
            active: Arc::new(AtomicUsize::new(0)),
            backing: std::sync::Mutex::new(None),
        })
    }

    fn prepared(value: Arc<BinaryValue>, cache: &LanceCache) -> PreparedPositions {
        PreparedPositions {
            token_id: 7,
            codec: PositionStreamCodec::PackedDelta,
            offsets: vec![0, 4, 12],
            value,
            cache: WeakLanceCache::from(cache),
        }
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

    #[tokio::test]
    async fn selective_phrase_short_payload_is_an_error_not_a_cache_entry() {
        let cache = LanceCache::with_capacity(4096);
        let value = binary_value(true, false);
        let plan = prepared(value.clone(), &cache);
        for _ in 0..2 {
            let error = match plan.read(2, &NoOpMetricsCollector).await {
                Ok(_) => panic!("short read accepted"),
                Err(error) => error,
            };
            assert!(matches!(error, Error::Index { .. }));
            assert!(error.to_string().contains("unexpected buffer length"));
        }
        assert_eq!(value.calls.load(Ordering::SeqCst), 2);
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn selective_phrase_cancellation_drops_pending_io() {
        let cache = LanceCache::no_cache();
        let value = binary_value(false, true);
        let plan = prepared(value.clone(), &cache);
        let metrics = NoOpMetricsCollector;
        let mut read = Box::pin(plan.read(2, &metrics));
        assert!(futures::poll!(&mut read).is_pending());
        assert_eq!(value.active.load(Ordering::SeqCst), 1);
        drop(read);
        assert_eq!(value.active.load(Ordering::SeqCst), 0);
    }
}
