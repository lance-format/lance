// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Independent doc-ID slices sharing one physical partition's fetched inputs.

use super::*;

pub(super) const MAX_ACTIVE_SLICES: usize = 8;
const MAX_SLICES_PER_PARTITION: usize = 5;

pub(super) type SliceTask<'a> = futures::future::BoxFuture<'a, Result<(usize, Vec<(DocId, f32)>)>>;

/// Equal half-open ranges, independent of dataset row addresses and filtering.
/// Statistics and scoring lengths remain those of the complete partition.
pub(super) fn document_ranges(num_docs: usize, target_docs: usize) -> Result<Vec<Range<u64>>> {
    if target_docs == 0 || num_docs as u64 > u64::from(u32::MAX) {
        return Err(Error::invalid_input(format!(
            "slice document count {num_docs} must fit dense DocIds and target {target_docs} must be positive"
        )));
    }
    if num_docs == 0 {
        return Ok(Vec::new());
    }
    let count = num_docs.div_ceil(target_docs).min(MAX_SLICES_PER_PARTITION);
    let docs = num_docs as u64;
    // count <= 5 and docs <= u32::MAX, so the products fit u64.
    Ok((0..count)
        .map(|index| index as u64 * docs / count as u64..(index as u64 + 1) * docs / count as u64)
        .collect())
}

pub(super) struct SliceSearch<'a> {
    pub(super) ordinal: usize,
    pub(super) positions: Arc<[PreparedPositions]>,
    pub(super) fetched: FetchedPostings,
    pub(super) request: ModernSearchRequest<'a>,
    pub(super) lengths: Arc<DocLengths>,
    pub(super) visibility: Arc<DocVisibility>,
    pub(super) range: Range<u64>,
    pub(super) shared: Arc<AtomicU32>,
}

impl SliceSearch<'_> {
    pub(super) async fn run(self) -> Result<(usize, Vec<(DocId, f32)>)> {
        let Self {
            ordinal,
            positions,
            fetched,
            request,
            lengths,
            visibility,
            range,
            shared,
        } = self;
        let tokens = request.tokens.clone();
        let scorer = request.impact_scorer.clone();
        let limit = request
            .params
            .limit
            .ok_or_else(|| Error::internal("selective phrase lost its finite limit"))?;
        let (mut search, idf, block_limits) = spawn_cpu(move || {
            let loaded = fetched.into_loaded(&tokens, scorer.as_ref())?;
            let idf = idf_by_position(&loaded.postings, scorer.as_ref());
            let search = PhraseSearch::new(loaded.postings, limit)?.with_doc_range(range)?;
            let block_limits = search.position_block_limits();
            Ok::<_, Error>((search, idf, block_limits))
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
                let docs = ModernWandDocuments::filtered(&lengths, visibility.as_ref());
                let progress = search.advance(&docs, scorer, shared, metrics.as_ref())?;
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
                            .find(|(_, plan)| plan.token_id == block_request.token_id)
                            .ok_or_else(|| {
                                Error::internal("position request has no prepared term")
                            })?;
                        let metrics = request.metrics.as_ref();
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
                        let end_block = block_limits
                            .iter()
                            .find(|(token, _)| *token == plan.token_id)
                            .map(|(_, limit)| *limit)
                            .ok_or_else(|| {
                                Error::internal("slice term has no position block bound")
                            })?;
                        let predictions = plan
                            .read_ahead_blocks(&window, clause_budget)?
                            .into_iter()
                            .take_while(|blocks| blocks.start < end_block);
                        read_ahead.schedule(
                            term,
                            predictions.map(|blocks| {
                                let future =
                                    plan.read(blocks.start, request.metrics.as_ref()).boxed();
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
    #[case::empty(0, 10)]
    #[case::single(1, 10)]
    #[case::exact(10, 10)]
    #[case::split(11, 10)]
    #[case::cap(999, 1)]
    #[case::max_domain(u32::MAX as usize, 250_000)]
    fn selective_phrase_slice_ranges_cover_dense_docs(#[case] docs: usize, #[case] target: usize) {
        let ranges = document_ranges(docs, target).unwrap();
        assert_eq!(ranges.len(), docs.div_ceil(target).min(5));
        if docs == 0 {
            assert!(ranges.is_empty());
            return;
        }
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, docs as u64);
        assert!(ranges.iter().all(|range| range.start < range.end));
        assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
        let smallest = ranges
            .iter()
            .map(|range| range.end - range.start)
            .min()
            .unwrap();
        assert!(
            ranges
                .iter()
                .all(|range| range.end - range.start <= smallest + 1)
        );
    }

    #[test]
    fn selective_phrase_slice_ranges_reject_invalid_domain() {
        for (docs, target) in [(1, 0), (u32::MAX as usize + 1, 1)] {
            let error = document_ranges(docs, target).unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("must fit dense DocIds"));
        }
    }
}
