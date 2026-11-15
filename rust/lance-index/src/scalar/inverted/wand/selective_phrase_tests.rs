// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::selective_phrase::{PhraseProgress, PhraseSearch, PositionWindow};
use super::tests::{UnitScorer, generate_posting_list_with_positions};
use super::*;
use crate::metrics::NoOpMetricsCollector;
use crate::scalar::inverted::encoding::encode_position_stream_block_into;
use crate::scalar::inverted::index::PositionStreamCodec;
use crate::scalar::inverted::index::SharedPositionStream;
use crate::scalar::inverted::{CompressedPositionStorage, DocSet};
use lance_core::Error;
use rstest::rstest;

fn unit_length_docs(num_docs: usize) -> DocSet {
    let mut docs = DocSet::default();
    for doc_id in 0..num_docs {
        docs.append(doc_id as u64, 1);
    }
    docs
}

#[rstest]
#[case::packed(PositionStreamCodec::PackedDelta)]
#[case::varint(PositionStreamCodec::VarintDocDelta)]
fn selective_phrase_resumes_blocks_and_keeps_late_high_score(
    #[case] codec: PositionStreamCodec,
    #[values(1, 500)] limit: usize,
    #[values(false, true)] repeated_term: bool,
    #[values(1, 3)] window_blocks: usize,
) -> Result<()> {
    use super::selective_phrase::{PhraseProgress, PhraseSearch, PositionWindow};
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
                    let first = request.index / window_blocks * window_blocks;
                    let end = (first + window_blocks).min(stream.block_count());
                    let range = stream.block_range(first).start..stream.block_range(end - 1).end;
                    let offsets = stream.block_offsets()[first..end]
                        .iter()
                        .map(|offset| *offset - range.start as u32)
                        .collect::<Vec<_>>();
                    search.supply(
                        request,
                        PositionWindow {
                            first_block: first,
                            codec,
                            offsets: Arc::from(offsets.into_boxed_slice()),
                            bytes: Arc::from(&stream.bytes()[range]),
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
    use super::selective_phrase::{PhraseProgress, PhraseSearch, PositionWindow};
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
            PositionWindow {
                first_block: requests[0].index + 1,
                offsets: Arc::from([0]),
                codec: PositionStreamCodec::PackedDelta,
                bytes: bytes.clone(),
            },
        )
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(
        error
            .to_string()
            .contains("does not contain requested block")
    );
    search.supply(
        requests[0],
        PositionWindow {
            first_block: requests[0].index,
            offsets: Arc::from([0]),
            codec: PositionStreamCodec::PackedDelta,
            bytes,
        },
    )?;
    assert!(weak.upgrade().is_some());
    drop(search); // cancellation during I/O has no background or borrowed state
    assert!(weak.upgrade().is_none());
    Ok(())
}

/// A failed rare-term alignment proves the whole phrase false. The much
/// larger final clause must not require position I/O in that case.
#[rstest]
fn selective_phrase_does_not_read_a_rejected_follower(
    #[values(PositionStreamCodec::VarintDocDelta, PositionStreamCodec::PackedDelta)]
    codec: PositionStreamCodec,
    #[values(false, true)] matches: bool,
    #[values(false, true)] late_match: bool,
) -> Result<()> {
    let docs = unit_length_docs(1);
    // In the late case, the first rare-pair alignment fails on the common
    // clause. Resuming after its I/O must still find the second alignment.
    let positions = if late_match {
        [
            vec![0, 4],
            if matches { vec![1, 5] } else { vec![3, 7] },
            (0..2048).map(|index| index * 3 + 6).collect::<Vec<_>>(),
        ]
    } else {
        [
            vec![0],
            vec![if matches { 1 } else { 3 }],
            (0..2048).map(|index| index * 3 + 2).collect::<Vec<_>>(),
        ]
    };
    let build = || {
        positions
            .iter()
            .enumerate()
            .map(|(term, values)| {
                let mut list =
                    generate_posting_list_with_positions(vec![0], vec![values.clone()], 1.0, true);
                let PostingList::Compressed(list) = &mut list else {
                    unreachable!()
                };
                let mut bytes = Vec::new();
                encode_position_stream_block_into(
                    values,
                    &[values.len() as u32],
                    codec,
                    &mut bytes,
                )
                .unwrap();
                list.positions = Some(CompressedPositionStorage::SharedStream(
                    SharedPositionStream::new(codec, vec![0], bytes::Bytes::from(bytes)),
                ));
                PostingIterator::with_query_weight(
                    "term".into(),
                    term as u32,
                    term as u32,
                    1.0,
                    PostingList::Compressed(list.clone()),
                    1,
                )
            })
            .collect::<Vec<_>>()
    };
    let params = FtsSearchParams::default()
        .with_limit(Some(10))
        .with_phrase_slop(Some(0));
    let metrics = NoOpMetricsCollector;
    let mut ordinary = Wand::new(Operator::And, build().into_iter(), &docs, UnitScorer)
        .with_bulk_and_mode(BulkAndMode::Off);
    let expected = ordinary.search(&params, &metrics)?;
    assert_eq!(expected.len(), usize::from(matches));
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
    let mut search = PhraseSearch::new(build(), 10)?;
    let floor = Arc::new(AtomicU32::new(f32::NEG_INFINITY.to_bits()));
    let mut read_final_clause = false;
    let actual = loop {
        match search.advance(&docs, UnitScorer, floor.clone(), &metrics)? {
            PhraseProgress::Done(hits) => break hits,
            PhraseProgress::Read(requests) => {
                for request in requests {
                    assert!(
                        matches || request.token_id != 2,
                        "a rejected rare-term alignment must not read the final clause"
                    );
                    read_final_clause |= request.token_id == 2;
                    let source = &sources
                        .iter()
                        .find(|(token, _)| *token == request.token_id)
                        .unwrap()
                        .1;
                    search.supply(
                        request,
                        PositionWindow {
                            first_block: 0,
                            codec,
                            offsets: Arc::from(vec![0].into_boxed_slice()),
                            bytes: Arc::from(source.bytes()),
                        },
                    )?;
                }
            }
        }
    };
    assert_eq!(read_final_clause, matches);
    assert_eq!(actual.len(), usize::from(matches));
    assert_eq!(
        actual
            .iter()
            .map(|hit| hit.posting_doc_id)
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|hit| hit.posting_doc_id)
            .collect::<Vec<_>>()
    );
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.document, expected.document);
        assert_eq!(actual.doc_length, expected.doc_length);
        let mut actual_freqs = actual.freqs.clone();
        let mut expected_freqs = expected.freqs.clone();
        actual_freqs.sort_unstable();
        expected_freqs.sort_unstable();
        assert_eq!(actual_freqs, expected_freqs);
    }
    Ok(())
}
