// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

// Deterministic top-k tiebreak (score DESC, row_id ASC).
//
// A BM25 top-k that breaks ties arbitrarily is not a stable prefix across `k`:
// top-k1 is not a prefix of top-k2, so paginating a tied-score query (a broad
// term where many documents share `(tf, doc_len)`) duplicates and skips rows.
// What the collector does with a document tied at the k-th score follows from
// how far the walk can see into the result order:
//
// * addresses resident and ascending with DocId: the tie has already been lost,
//   so the candidate is rejected and nothing is retained;
// * addresses resident, order unknown: the tie is between things the walk cannot
//   rank (the elements of one row), so it is retained;
// * no resident addresses: the tie is unrankable and this pass cannot rank it, so
//   the walk stops and asks the caller to retry with addresses.
//
// `admit_ties_floor` keeps the WAND pruning kernels from dropping the band the
// collector still needs in the two admit modes.

/// Collector for a walk whose row addresses are resident and strictly ascend with
/// the DocId order the walk visits.
fn ascending_address_collector(limit: usize) -> TopKCollector {
    TopKCollector::new(limit, limit, TieHandling::AscendingRowAddresses)
}

/// Collector for a walk whose row addresses are resident but whose DocId order
/// does not follow them, or whose rows own several documents.
fn row_address_collector(limit: usize) -> TopKCollector {
    TopKCollector::new(limit, limit, TieHandling::ByRowAddress)
}

/// Collector for a walk that has no resident row addresses.
fn deferred_address_collector(limit: usize) -> TopKCollector {
    TopKCollector::new(limit, limit, TieHandling::DeferredAddresses)
}

#[test]
fn test_top_k_collector_reuses_frequency_slots() -> Result<()> {
    const LIMIT: usize = 8;
    const NUM_DOCS: usize = 10_000;
    let mut collector = row_address_collector(LIMIT);

    for doc in 0..NUM_DOCS {
        let num_terms = doc % 4 + 1;
        let inserted = collector.insert(
            ScoredDoc::new(doc as u64, doc as f32),
            doc as u64,
            num_terms as u32,
            doc as u64,
            (0..num_terms).map(|term| (term as u32, doc as u32)),
        )?;
        assert!(inserted);
        assert!(collector.num_frequency_slots() <= LIMIT);
    }
    assert_eq!(collector.num_frequency_slots(), LIMIT);

    let mut candidates = collector.into_candidates(|key| key)?;
    candidates.sort_unstable_by_key(|candidate| candidate.posting_doc_id);
    assert_eq!(candidates.len(), LIMIT);
    for (candidate, expected_doc) in candidates.iter().zip(NUM_DOCS - LIMIT..NUM_DOCS) {
        assert_eq!(candidate.posting_doc_id, expected_doc as u64);
        assert_eq!(candidate.document, expected_doc as u64);
        let expected_freqs = (0..expected_doc % 4 + 1)
            .map(|term| (term as u32, expected_doc as u32))
            .collect::<Vec<_>>();
        assert_eq!(candidate.freqs, expected_freqs);
    }
    Ok(())
}

/// The ordering key a walk that defers row addresses hands the collector: one
/// constant for every document, so every score tie compares equal.
const DEFERRED_ORDER_KEY: u64 = u64::MAX;

/// Insert a document the walk can rank by row address. `document_key` doubles
/// as the address so the assertions can read the result back.
fn insert_addressed(collector: &mut TopKCollector, document_key: u64, score: f32) -> Result<bool> {
    insert_at_address(collector, document_key, document_key, score)
}

/// Insert a document at an explicit address, so several documents can share
/// one address the way the elements of a list row do.
fn insert_at_address(
    collector: &mut TopKCollector,
    document_key: u64,
    address: u64,
    score: f32,
) -> Result<bool> {
    collector.insert(
        ScoredDoc::new(document_key, score),
        address,
        1,
        document_key,
        std::iter::once((0u32, 1u32)),
    )
}

/// Insert a document into a collector whose walk defers row addresses.
fn insert_deferred(collector: &mut TopKCollector, document_key: u64, score: f32) -> Result<bool> {
    insert_at_address(collector, document_key, DEFERRED_ORDER_KEY, score)
}

fn collected_keys(collector: TopKCollector) -> Result<Vec<u64>> {
    let mut keys = collector
        .into_candidates(|key| key)?
        .into_iter()
        .map(|candidate| candidate.document)
        .collect::<Vec<_>>();
    keys.sort_unstable();
    Ok(keys)
}

#[rstest]
#[case::one(1.0)]
#[case::small(f32::MIN_POSITIVE)]
#[case::large(1e30)]
fn test_admit_ties_floor_sits_just_below_kth(#[case] kth: f32) {
    // A document tied at the k-th score must pass the `score > threshold` admit
    // test, so the floor sits strictly below the k-th score, exactly one ULP down.
    let floor = admit_ties_floor(kth);
    assert!(floor < kth, "floor {floor} must be < kth {kth}");
    assert_eq!(floor, kth.next_down());
}

#[test]
fn test_admit_ties_floor_zero_negative_and_nan() {
    // Zero and negative inputs drop below zero, which the `threshold > 0.0`
    // prune guards read as "no threshold yet" (pruning disabled).
    assert!(admit_ties_floor(0.0) < 0.0);
    assert!(admit_ties_floor(-1.0) < -1.0);
    // NaN stays NaN; it is never a competitive score and is guarded at insert time.
    assert!(admit_ties_floor(f32::NAN).is_nan());
    // The smallest positive subnormal would step down to zero and disable
    // pruning, so it stays put.
    let tiny = f32::from_bits(1);
    assert_eq!(admit_ties_floor(tiny), tiny);
}

#[test]
fn test_top_k_collector_ascending_addresses_reject_kth_ties() -> Result<()> {
    // The walk reaches documents in address order, so a candidate tied at the
    // k-th score has the higher address and lost the tie before the collector saw
    // it. Retaining it would only make the caller re-rank documents it already
    // knows lose.
    let mut collector = ascending_address_collector(2);
    assert!(insert_addressed(&mut collector, 10, 1.0)?);
    assert!(insert_addressed(&mut collector, 20, 1.0)?); // heap full, k-th is (1.0, 20)
    assert!(!insert_addressed(&mut collector, 30, 1.0)?); // later address, same score
    assert!(!insert_addressed(&mut collector, 40, 1.0)?);
    assert!(!insert_addressed(&mut collector, 15, 0.5)?); // below the k-th score
    // Refusal covers rank ties only. A candidate that beats the k-th score still
    // enters, and the k-th score it publishes is what keeps the walk pruning.
    assert!(
        insert_addressed(&mut collector, 30, 3.0)?,
        "a higher score must enter the heap even when addresses ascend"
    );
    assert_eq!(
        collector.kth_score_if_full(),
        Some(1.0),
        "the k-th score is the weakest live candidate"
    );
    assert!(
        collector.tie_band_len() == 0,
        "an address-ascending walk retains nothing"
    );
    assert!(!collector.score_floor_overflow());
    assert!(!collector.band_truncated());
    assert_eq!(
        collected_keys(collector)?,
        vec![10, 30],
        "the tie stays resolved on the lowest address, and a higher score replaces one"
    );
    Ok(())
}

#[test]
fn test_top_k_collector_ascending_addresses_keep_the_band_empty_under_ties() -> Result<()> {
    // A corpus where every document ties, in address order: the heap fills and
    // every later candidate is rejected, so the working set never grows past
    // `limit` and the caller never has to re-select.
    const LIMIT: usize = 10;
    let mut collector = ascending_address_collector(LIMIT);
    for document_key in 0..4_000u64 {
        insert_addressed(&mut collector, document_key, 1.0)?;
    }
    assert_eq!(
        collector.buffered(),
        LIMIT,
        "nothing may accumulate beyond the heap"
    );
    assert!(collector.tie_band_len() == 0);
    assert!(!collector.score_floor_overflow());
    assert!(!collector.band_truncated());
    assert_eq!(
        collected_keys(collector)?,
        (0..LIMIT as u64).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn test_top_k_collector_by_row_address_evicts_on_full_key() -> Result<()> {
    // The ordering key is the row address but DocId order does not follow it, so
    // a score tie is still decided here: a lower address evicts the worst tied
    // entry, a higher one loses.
    let mut collector = row_address_collector(2);
    assert!(insert_addressed(&mut collector, 20, 1.0)?);
    assert!(insert_addressed(&mut collector, 10, 1.0)?); // heap full {10, 20}
    assert!(insert_addressed(&mut collector, 5, 1.0)?); // ties, lower address
    assert!(!insert_addressed(&mut collector, 30, 1.0)?); // ties, higher address
    assert!(!insert_addressed(&mut collector, 15, 0.5)?); // below the k-th score
    assert!(!collector.score_floor_overflow());
    // Distinct addresses leave nothing the walk cannot rank.
    assert!(collector.tie_band_len() == 0);
    assert_eq!(
        collected_keys(collector)?,
        vec![5, 10],
        "the two lowest addresses among the tie"
    );
    Ok(())
}

#[test]
fn test_top_k_collector_retains_element_documents_of_one_row() -> Result<()> {
    // Element documents of one row share its address, so the address key cannot
    // rank them and only `doc_index` can, which the walk never sees. Evicting
    // one of them here would pick an arbitrary element, so both are retained and
    // the resolved re-selection settles it.
    const ROW: u64 = 100;
    let mut collector = row_address_collector(2);
    assert!(insert_at_address(&mut collector, 11, ROW, 1.0)?);
    assert!(insert_at_address(&mut collector, 12, ROW, 1.0)?);
    // A lower address outranks the row whatever its elements' coordinates are,
    // so it takes a heap slot and the element it displaced stays a candidate.
    assert!(insert_at_address(&mut collector, 10, 50, 1.0)?);
    assert!(!collector.score_floor_overflow());
    assert_eq!(
        collected_keys(collector)?,
        vec![10, 11, 12],
        "both elements of the tied row must reach the resolved re-selection"
    );
    Ok(())
}

#[test]
fn test_top_k_collector_ranks_rows_around_a_tied_element_row() -> Result<()> {
    // The element band is still bounded by the row it belongs to: a strictly
    // higher address loses outright, and a rising k-th key drops the band.
    const ROW: u64 = 100;
    let mut collector = row_address_collector(2);
    assert!(insert_at_address(&mut collector, 11, ROW, 1.0)?);
    assert!(insert_at_address(&mut collector, 12, ROW, 1.0)?);
    assert!(!insert_at_address(&mut collector, 13, 200, 1.0)?); // higher address
    assert!(insert_at_address(&mut collector, 10, 50, 1.0)?); // lower address
    assert!(insert_at_address(&mut collector, 9, 60, 2.0)?); // better score
    assert_eq!(
        collected_keys(collector)?,
        vec![9, 10],
        "a rising k-th key drops the element band"
    );
    Ok(())
}

#[test]
fn test_top_k_collector_retains_kth_score_band_of_one_row() -> Result<()> {
    // Documents the walk cannot rank against the k-th entry survive on both sides
    // of the split, whether they land in the heap or beside it. The DocId proxy
    // only decides which side that is.
    const ROW: u64 = 100;
    let mut collector = row_address_collector(2);
    assert!(insert_at_address(&mut collector, 10, ROW, 1.0)?);
    assert!(insert_at_address(&mut collector, 11, ROW, 1.0)?);
    assert!(insert_at_address(&mut collector, 5, ROW, 1.0)?); // wins the proxy, heap
    assert!(!insert_at_address(&mut collector, 20, ROW, 1.0)?); // loses it, band
    assert!(!collector.score_floor_overflow());
    assert_eq!(
        collected_keys(collector)?,
        vec![5, 10, 11, 20],
        "the whole tie band is kept, not just the first two seen"
    );

    // Once the k-th score rises past the band, none of its members can win.
    let mut collector = row_address_collector(2);
    assert!(insert_at_address(&mut collector, 0, ROW, 1.0)?);
    assert!(insert_at_address(&mut collector, 1, ROW, 1.0)?);
    assert!(!insert_at_address(&mut collector, 2, ROW, 1.0)?);
    assert!(insert_at_address(&mut collector, 3, ROW, 2.0)?); // k-th still 1.0
    assert!(insert_at_address(&mut collector, 4, ROW, 2.0)?); // k-th rises to 2.0
    assert_eq!(collected_keys(collector)?, vec![3, 4]);
    Ok(())
}

#[test]
fn test_top_k_collector_deferred_addresses_stop_when_the_band_fills() -> Result<()> {
    // A walk with no resident addresses cannot rank a k-th-score tie, but a single
    // tie is common and cheap, so it is retained like any other. A band that fills
    // means the corpus ties at the k-th score throughout: scoring the rest of it
    // cannot rank the band either, so the pass stops there with the retry signal.
    const LIMIT: usize = 2;
    let mut collector = deferred_address_collector(LIMIT);
    assert!(insert_deferred(&mut collector, 10, 1.0)?);
    assert!(insert_deferred(&mut collector, 11, 1.0)?); // heap full
    assert!(
        !insert_deferred(&mut collector, 12, 1.0)?,
        "a tie the walk cannot rank is retained rather than ranked"
    );
    assert_eq!(collector.tie_band_len(), 1);
    assert!(
        !collector.aborted(),
        "one boundary tie is not a reason to abandon the pass"
    );

    for key in 13..(12 + SCORE_FLOOR_BUFFER) {
        insert_deferred(&mut collector, key as u64, 1.0)?;
    }
    assert_eq!(
        collector.buffered(),
        LIMIT + SCORE_FLOOR_BUFFER,
        "the retained band is bounded by the same cap as the other modes"
    );
    assert!(!collector.aborted(), "a band that just filled is not over");

    insert_deferred(&mut collector, 9_000, 1.0)?;
    assert!(
        collector.aborted(),
        "a full band means a corpus tied at the k-th score, so stop walking"
    );
    assert!(
        collector.score_floor_overflow(),
        "the caller must retry with addresses"
    );
    Ok(())
}

#[test]
fn test_top_k_collector_deferred_abort_survives_a_later_better_score() -> Result<()> {
    // An abort is not a band that became obsolete: the pass ended early, so what
    // it returned stays incomplete however the k-th score moves afterwards.
    const LIMIT: usize = 2;
    let mut collector = deferred_address_collector(LIMIT);
    for key in 0..(LIMIT + SCORE_FLOOR_BUFFER + 50) {
        insert_deferred(&mut collector, key as u64, 1.0)?;
    }
    assert!(collector.aborted());
    assert_eq!(
        collector.buffered(),
        LIMIT + SCORE_FLOOR_BUFFER,
        "the band stopped growing at its cap once the pass aborted"
    );
    assert!(collector.num_frequency_slots() <= LIMIT);

    assert!(insert_deferred(&mut collector, 1_000, 2.0)?);
    assert!(insert_deferred(&mut collector, 1_001, 2.0)?);
    assert!(
        collector.score_floor_overflow(),
        "a deliberate abort is never obsolete"
    );
    assert_eq!(collected_keys(collector)?, vec![1_000, 1_001]);
    Ok(())
}

#[test]
fn test_top_k_collector_bounds_the_element_band() -> Result<()> {
    // Nothing bounds how many elements a list row holds, so a walk that retains a
    // band caps it too. Past the cap the heap keeps ranking on the DocId proxy,
    // so the documents the result order wants survive even though the rest of the
    // group is dropped.
    const ROW: u64 = 100;
    const LIMIT: usize = 2;
    let max_buffered = LIMIT + SCORE_FLOOR_BUFFER;
    let collect = || -> Result<(Vec<u64>, bool, usize)> {
        let mut collector = row_address_collector(LIMIT);
        // Descending DocIds, so the documents the result order wants arrive last
        // and a collector that kept whatever it saw first would miss them.
        for document_key in (0..(max_buffered as u64 + 50)).rev() {
            insert_at_address(&mut collector, document_key, ROW, 1.0)?;
        }
        let truncated = collector.band_truncated();
        let buffered = collector.buffered();
        assert!(
            !collector.score_floor_overflow(),
            "a walk with resident addresses has nothing to retry"
        );
        Ok((collected_keys(collector)?, truncated, buffered))
    };

    let (kept, truncated, buffered) = collect()?;
    assert!(
        truncated,
        "an element band wider than the buffer must report a truncation"
    );
    assert!(
        buffered <= max_buffered,
        "buffered {buffered} must stay within {max_buffered}"
    );
    assert_eq!(
        kept[..LIMIT],
        [0, 1],
        "the documents the result order picks must survive the truncation"
    );

    let (repeated, _, _) = collect()?;
    assert_eq!(kept, repeated, "the truncation must be deterministic");
    Ok(())
}

#[test]
fn test_top_k_collector_clears_obsolete_band_truncation() -> Result<()> {
    // A truncation describes the k-th-score band it happened in. Once the k-th
    // score rises past that band, nothing it dropped can win, so the collector
    // must stop reporting it.
    const LIMIT: usize = 2;
    const ROW: u64 = 100;
    let mut collector = row_address_collector(LIMIT);
    for key in 0..(LIMIT + SCORE_FLOOR_BUFFER + 1) {
        insert_at_address(&mut collector, key as u64, ROW, 1.0)?;
    }
    assert!(collector.band_truncated());

    assert!(insert_at_address(&mut collector, 1_000, ROW, 2.0)?);
    assert!(insert_at_address(&mut collector, 1_001, ROW, 2.0)?);

    let truncated = collector.band_truncated();
    assert_eq!(collected_keys(collector)?, vec![1_000, 1_001]);
    assert!(!truncated, "an obsolete band must stop being reported");
    Ok(())
}

#[test]
fn test_top_k_collector_reuses_slots_when_no_tie_is_retained() -> Result<()> {
    // A deferred walk over strictly improving scores retains nothing, so every
    // replacement must reuse the displaced entry's slot in place. Allocating a
    // slot for the newcomer and releasing the displaced one instead would add a
    // slot and hand its buffer away, which is exactly what `FrequencySlots`
    // exists to avoid.
    const LIMIT: usize = 4;
    const NUM_PAIRS: u32 = 4;
    let mut collector = deferred_address_collector(LIMIT);
    for doc in 0..1_000u64 {
        collector.insert(
            ScoredDoc::new(doc, doc as f32),
            DEFERRED_ORDER_KEY,
            1,
            doc,
            (0..NUM_PAIRS).map(|term| (term, doc as u32)),
        )?;
    }
    assert!(collector.tie_band_len() == 0);
    assert!(!collector.aborted(), "an improving score is never a tie");
    assert_eq!(
        collector.num_frequency_slots(),
        LIMIT,
        "an improving candidate must not allocate a slot beyond the heap"
    );
    assert!(
        collector.min_frequency_slot_capacity() >= NUM_PAIRS as usize,
        "a reused slot must keep its retained capacity"
    );
    Ok(())
}

#[test]
fn test_top_k_collector_recycles_slots_while_retaining_ties() -> Result<()> {
    // A retained band that the k-th score rises past is dropped wholesale, and
    // the frequency slots of the documents that moved into it must come back.
    const LIMIT: usize = 2;
    const ROW: u64 = 100;
    let mut collector = row_address_collector(LIMIT);
    for round in 0..50u64 {
        for tie in 0..8u64 {
            insert_at_address(&mut collector, round * 8 + tie, ROW, round as f32)?;
        }
    }
    assert!(!collector.band_truncated());
    assert!(
        collector.num_frequency_slots() <= LIMIT + 1,
        "slot count {} grew with the number of dropped bands",
        collector.num_frequency_slots()
    );
    Ok(())
}

#[test]
fn test_top_k_collector_rejects_nan_scores() -> Result<()> {
    let mut collector = row_address_collector(1);
    assert!(insert_addressed(&mut collector, 1, 1.0)?);
    assert!(!insert_addressed(&mut collector, 0, f32::NAN)?);
    assert_eq!(collected_keys(collector)?, vec![1]);
    Ok(())
}

#[test]
fn test_top_k_collector_rejects_nan_while_filling() -> Result<()> {
    // NaN must not take a slot even before the heap is full, or it would rank
    // above every real score in the final order.
    let mut collector = row_address_collector(2);
    assert!(!insert_addressed(&mut collector, 0, f32::NAN)?);
    assert!(insert_addressed(&mut collector, 1, 1.0)?);
    assert_eq!(collected_keys(collector)?, vec![1]);
    Ok(())
}

#[test]
fn test_top_k_collector_rejects_score_admits_kth_ties() -> Result<()> {
    // MaxScore skips candidates `rejects_score` refuses before they reach the
    // collector, so a tie at the k-th score must not be refused wherever the
    // collector still has to rank or retain it.
    let mut collector = row_address_collector(1);
    assert!(
        !collector.rejects_score(1.0),
        "an empty heap rejects nothing"
    );
    assert!(insert_addressed(&mut collector, 10, 1.0)?);
    assert!(
        !collector.rejects_score(1.0),
        "a k-th-score tie may still win"
    );
    assert!(!collector.rejects_score(2.0));
    assert!(collector.rejects_score(0.5));
    assert!(collector.rejects_score(f32::NAN));
    Ok(())
}

#[test]
fn test_top_k_collector_rejects_score_refuses_kth_ties_when_addresses_ascend() -> Result<()> {
    // The same heap in the address-ascending mode: the candidate reaches the walk
    // after the k-th entry, so its address already lost the tie and refusing it
    // here is what lets block-max pruning skip a whole block.
    let mut collector = ascending_address_collector(1);
    assert!(insert_addressed(&mut collector, 10, 1.0)?);
    assert!(
        collector.rejects_score(1.0),
        "a k-th-score tie arriving later has the higher address"
    );
    assert!(!collector.rejects_score(1.1));
    assert!(collector.rejects_score(0.5));
    assert!(collector.rejects_score(f32::NAN));
    Ok(())
}
