// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[test]
fn test_top_k_collector_reuses_frequency_slots() -> Result<()> {
    const LIMIT: usize = 8;
    const NUM_DOCS: usize = 10_000;
    let mut collector = TopKCollector::new(LIMIT, LIMIT);

    for doc in 0..NUM_DOCS {
        let num_terms = doc % 4 + 1;
        let inserted = collector.insert(
            ScoredDoc::new(doc as u64, doc as f32),
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
