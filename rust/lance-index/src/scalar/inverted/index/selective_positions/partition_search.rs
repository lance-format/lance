// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Request-owned exact-phrase search over one physical partition.

use super::*;

pub(super) const MAX_ACTIVE_PARTITIONS: usize = 4;

pub(super) struct PartitionSearch {
    pub(super) ordinal: usize,
    pub(super) positions: Vec<PreparedPositions>,
    pub(super) fetched: FetchedPostings,
    pub(super) tokens: Arc<Tokens>,
    pub(super) params: Arc<FtsSearchParams>,
    pub(super) metrics: Arc<dyn MetricsCollector>,
    pub(super) impact_scorer: Arc<MemBM25Scorer>,
    pub(super) lengths: Arc<DocLengths>,
    pub(super) visibility: Arc<DocVisibility>,
    pub(super) shared: Arc<AtomicU32>,
}

impl PartitionSearch {
    pub(super) async fn run(self) -> Result<(usize, Vec<(DocId, f32)>)> {
        let Self {
            ordinal,
            positions,
            fetched,
            tokens,
            params,
            metrics,
            impact_scorer,
            lengths,
            visibility,
            shared,
        } = self;
        let query_tokens = tokens.clone();
        let scorer = impact_scorer.clone();
        let limit = params
            .limit
            .ok_or_else(|| Error::internal("selective phrase lost its finite limit"))?;
        let (mut search, idf) = spawn_cpu(move || {
            let loaded = fetched.into_loaded(&query_tokens, scorer.as_ref())?;
            let idf = idf_by_position(&loaded.postings, scorer.as_ref());
            let search = PhraseSearch::new(loaded.postings, limit)?;
            Ok::<_, Error>((search, idf))
        })
        .await?;
        let mut read_ahead = ReadAhead::new(positions.len());
        let clause_budget = MAX_LIVE_WINDOW_BYTES / tokens.len() as u64;
        loop {
            let lengths = lengths.clone();
            let visibility = visibility.clone();
            let scorer = impact_scorer.clone();
            let shared = shared.clone();
            let cpu_metrics = metrics.clone();
            let advance = spawn_cpu(move || {
                let docs = ModernWandDocuments::filtered(&lengths, visibility.as_ref());
                let progress = search.advance(&docs, scorer, shared, cpu_metrics.as_ref())?;
                Ok::<_, Error>((search, progress))
            });
            let (next, progress) = read_ahead.drive(advance).await?;
            search = next;
            match progress {
                PhraseProgress::Done(candidates) => {
                    read_ahead.clear();
                    drop(search);
                    let scorer = impact_scorer.clone();
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
                            .find(|(_, plan)| plan.token_id == block_request.token_id)
                            .ok_or_else(|| {
                                Error::internal("position request has no prepared term")
                            })?;
                        let metrics = metrics.as_ref();
                        let read = read_ahead
                            .take(term, block_request.index)
                            .unwrap_or_else(|| plan.read(block_request.index, metrics).boxed());
                        reads.push(
                            async move { Ok::<_, Error>((term, block_request, read.await?)) },
                        );
                    }
                    let windows = read_ahead
                        .drive(futures::future::try_join_all(reads))
                        .await?;
                    for (term, block_request, window) in windows {
                        let plan = &positions[term];
                        // Current plus speculative encoded bytes share the
                        // original clause budget. Oversized neighbors skip
                        // speculation; offsets/scratch/cache/I/O are separate.
                        let predictions = plan.read_ahead_blocks(&window, clause_budget)?;
                        read_ahead.schedule(
                            term,
                            predictions.into_iter().map(|blocks| {
                                let future = plan.read(blocks.start, metrics.as_ref()).boxed();
                                (blocks, future)
                            }),
                        )?;
                        search.supply(block_request, window)?;
                    }
                }
            }
        }
    }
}
