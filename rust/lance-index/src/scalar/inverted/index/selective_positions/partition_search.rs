// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Request-owned exact-phrase search over one physical partition.

use super::*;

// Bound async task metadata even when tests use very small byte windows.
const MAX_ACTIVE_PARTITIONS: usize = 16;

#[derive(Clone, Copy, Debug)]
pub(super) struct SearchResources {
    pub(super) active_partitions: usize,
    pub(super) partition_bytes: u64,
}

/// Share one query's encoded allowance across useful partition overlap.
/// A current and next window per clause is the working-set target, bounded
/// by the existing per-partition cap. This is still an opt-in policy seed.
pub(super) fn search_resources(
    cpus: usize,
    partitions: usize,
    clauses: usize,
    window_bytes: u64,
) -> Result<SearchResources> {
    if cpus == 0 || partitions == 0 || !(2..=MAX_CLAUSES).contains(&clauses) || window_bytes == 0 {
        return Err(Error::invalid_input(format!(
            "selective query resource inputs must be positive with 2..={MAX_CLAUSES} clauses: cpus={cpus}, partitions={partitions}, clauses={clauses}, window_bytes={window_bytes}"
        )));
    }
    let overlap_bytes = window_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_mul(clauses as u64))
        .ok_or_else(|| Error::invalid_input("selective query resource byte target overflows u64"))?
        .min(MAX_PARTITION_WINDOW_BYTES);
    // overlap_bytes >= 4, so the quotient fits a 32-bit usize as well.
    let by_bytes = (MAX_QUERY_WINDOW_BYTES / overlap_bytes).max(1) as usize;
    let active_partitions = cpus
        .min(partitions)
        .min(MAX_ACTIVE_PARTITIONS)
        .min(by_bytes);
    Ok(SearchResources {
        active_partitions,
        partition_bytes: (MAX_QUERY_WINDOW_BYTES / active_partitions as u64)
            .min(MAX_PARTITION_WINDOW_BYTES),
    })
}

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
    pub(super) encoded_budget: u64,
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
            encoded_budget,
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
        let clause_budget = encoded_budget / tokens.len() as u64;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    fn selective_phrase_resources_obey_cpu_and_byte_caps(
        #[values(1, 2, 8, 16, 64)] cpus: usize,
        #[values(1, 3, 19, 1000)] partitions: usize,
        #[values(2, 5, 16)] clauses: usize,
    ) {
        let resources = search_resources(cpus, partitions, clauses, POSITION_WINDOW_BYTES).unwrap();
        assert!(
            (1..=cpus.min(partitions).min(MAX_ACTIVE_PARTITIONS))
                .contains(&resources.active_partitions)
        );
        assert!(resources.partition_bytes <= MAX_PARTITION_WINDOW_BYTES);
        assert!(
            resources.partition_bytes * resources.active_partitions as u64
                <= MAX_QUERY_WINDOW_BYTES
        );
        assert!(resources.partition_bytes / clauses as u64 >= POSITION_WINDOW_BYTES);
        let current_and_next = 2 * POSITION_WINDOW_BYTES * clauses as u64;
        if current_and_next <= MAX_PARTITION_WINDOW_BYTES {
            assert!(resources.partition_bytes >= current_and_next);
        }
    }

    #[test]
    fn selective_phrase_resources_keep_cpu1_and_reallocate_short_phrases() {
        let single = search_resources(1, 19, 2, POSITION_WINDOW_BYTES).unwrap();
        assert_eq!(single.active_partitions, 1);
        assert_eq!(single.partition_bytes, MAX_PARTITION_WINDOW_BYTES);
        let short = search_resources(16, 19, 2, POSITION_WINDOW_BYTES).unwrap();
        let longer = search_resources(16, 19, 5, POSITION_WINDOW_BYTES).unwrap();
        assert!(short.active_partitions > longer.active_partitions);
        assert!(short.partition_bytes < longer.partition_bytes);
    }

    #[test]
    fn selective_phrase_resources_reject_invalid_or_overflowing_inputs() {
        for (cpus, partitions, clauses, bytes) in [
            (0, 1, 2, 1),
            (1, 0, 2, 1),
            (1, 1, 0, 1),
            (1, 1, MAX_CLAUSES + 1, 1),
            (1, 1, 2, 0),
            (1, 1, 2, u64::MAX),
        ] {
            let error = search_resources(cpus, partitions, clauses, bytes).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("selective query resource"));
        }
    }
}
