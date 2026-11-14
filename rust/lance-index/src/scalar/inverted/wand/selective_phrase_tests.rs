// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::tests::{UnitScorer, generate_posting_list_with_positions, unit_length_docs};
use super::*;
use crate::metrics::NoOpMetricsCollector;
use crate::scalar::inverted::encoding::encode_position_stream_block_into;
use crate::scalar::inverted::index::SharedPositionStream;
use rstest::rstest;

#[rstest]
#[case::packed(PositionStreamCodec::PackedDelta)]
#[case::varint(PositionStreamCodec::VarintDocDelta)]
fn selective_phrase_resumes_blocks_and_keeps_late_high_score(
    #[case] codec: PositionStreamCodec,
    #[values(1, 500)] limit: usize,
    #[values(false, true)] repeated_term: bool,
) -> Result<()> {
    use super::selective_phrase::{PhraseProgress, PhraseSearch, PositionBlock};
    let total = BLOCK_SIZE * 5 + 7;
    let docs = unit_length_docs(total);
    let build = || {
        (0..2)
            .map(|term| {
                let positions = (0..total)
                    .map(|doc| {
                        let count = if doc == total - 1 { 32 } else { 1 };
                        // Even docs and the last doc match. Other docs contain both
                        // terms but never at the required relative phrase offsets.
                        (0..count)
                            .map(|position| {
                                position * 4
                                    + if repeated_term { 0 } else { term }
                                    + if !repeated_term
                                        && term == 1
                                        && doc % 2 == 1
                                        && doc != total - 1
                                    {
                                        1
                                    } else {
                                        0
                                    }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let mut list = generate_posting_list_with_positions(
                    (0..total as u32).collect(),
                    positions.clone(),
                    100.0,
                    true,
                );
                let PostingList::Compressed(list) = &mut list else {
                    unreachable!()
                };
                let mut bytes = Vec::new();
                let mut offsets = Vec::new();
                for block in positions.chunks(list.block_size) {
                    offsets.push(bytes.len() as u32);
                    let frequencies = block
                        .iter()
                        .map(|positions| positions.len() as u32)
                        .collect::<Vec<_>>();
                    let values = block.iter().flatten().copied().collect::<Vec<_>>();
                    encode_position_stream_block_into(&values, &frequencies, codec, &mut bytes)
                        .unwrap();
                }
                list.positions = Some(CompressedPositionStorage::SharedStream(
                    SharedPositionStream::new(codec, offsets, bytes::Bytes::from(bytes)),
                ));
                PostingIterator::with_query_weight(
                    "term".into(),
                    if repeated_term { 0 } else { term },
                    // Repeated terms use a gap of four present in every document.
                    if repeated_term { term * 4 } else { term },
                    1.0,
                    PostingList::Compressed(list.clone()),
                    total,
                )
            })
            .collect::<Vec<_>>()
    };
    let params = FtsSearchParams::default()
        .with_limit(Some(limit))
        .with_phrase_slop(Some(0));
    let metrics = NoOpMetricsCollector;
    let mut ordinary = Wand::new(Operator::And, build().into_iter(), &docs, UnitScorer)
        .with_bulk_and_mode(BulkAndMode::Off);
    let expected = ordinary.search(&params, &metrics)?;
    let sources = build()
        .into_iter()
        .map(|posting| {
            let PostingList::Compressed(list) = posting.list else {
                unreachable!()
            };
            let Some(CompressedPositionStorage::SharedStream(stream)) = list.positions else {
                unreachable!()
            };
            (posting.token_id, stream)
        })
        .collect::<Vec<_>>();
    let mut search = PhraseSearch::new(build(), limit)?;
    let shared = Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()));
    let mut suspensions = 0;
    let actual = loop {
        match search.advance(&docs, UnitScorer, shared.clone(), &metrics)? {
            PhraseProgress::Done(hits) => break hits,
            PhraseProgress::Read(requests) => {
                suspensions += 1;
                for request in requests {
                    let stream = &sources
                        .iter()
                        .find(|(id, _)| *id == request.token_id)
                        .unwrap()
                        .1;
                    search.supply(
                        request,
                        PositionBlock {
                            index: request.index,
                            codec,
                            bytes: Arc::from(stream.block(request.index)),
                        },
                    )?;
                }
            }
        }
    };
    assert!(
        suspensions > 1,
        "fixture must cross asynchronous block boundaries"
    );
    let mut actual_ids = actual
        .iter()
        .map(|hit| hit.posting_doc_id)
        .collect::<Vec<_>>();
    actual_ids.sort_unstable();
    let expected_ids = if limit == 1 || repeated_term {
        vec![(total - 1) as u64]
    } else {
        (0..total)
            .filter(|doc| doc % 2 == 0 || *doc == total - 1)
            .map(|doc| doc as u64)
            .collect()
    };
    assert_eq!(
        actual_ids, expected_ids,
        "membership is derived from fixture positions"
    );
    if limit == 1 && !repeated_term {
        assert!(
            suspensions < total.div_ceil(BLOCK_SIZE),
            "score-floor feedback must skip middle position blocks"
        );
    }

    let normalize = |hits: Vec<DocCandidate<u64>>| {
        let mut rows = hits
            .into_iter()
            .map(|hit| (hit.document, hit.doc_length, hit.freqs))
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    };
    assert_eq!(normalize(actual), normalize(expected));
    if limit == 1 {
        assert!(shared.load(Ordering::Relaxed) != f32::NEG_INFINITY.to_bits());
    }
    Ok(())
}

#[test]
fn selective_phrase_rejects_wrong_block_and_drops_owned_state() -> Result<()> {
    use super::selective_phrase::{PhraseProgress, PhraseSearch, PositionBlock};
    let docs = unit_length_docs(1);
    let postings = (0..2)
        .map(|term| {
            PostingIterator::with_query_weight(
                format!("t{term}"),
                term,
                term,
                1.0,
                generate_posting_list_with_positions(vec![0], vec![vec![term]], 10.0, true),
                1,
            )
        })
        .collect();
    let mut search = PhraseSearch::new(postings, 1)?;
    let PhraseProgress::Read(requests) = search.advance(
        &docs,
        UnitScorer,
        Arc::new(AtomicU32::new(0)),
        &NoOpMetricsCollector,
    )?
    else {
        panic!("expected a position miss")
    };
    let bytes: Arc<[u8]> = Arc::from([1_u8, 2, 3]);
    let weak = Arc::downgrade(&bytes);
    let error = search
        .supply(
            requests[0],
            PositionBlock {
                index: requests[0].index + 1,
                codec: PositionStreamCodec::PackedDelta,
                bytes: bytes.clone(),
            },
        )
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("index differs"));
    search.supply(
        requests[0],
        PositionBlock {
            index: requests[0].index,
            codec: PositionStreamCodec::PackedDelta,
            bytes,
        },
    )?;
    assert!(weak.upgrade().is_some());
    drop(search); // cancellation during I/O has no background or borrowed state
    assert!(weak.upgrade().is_none());
    Ok(())
}
