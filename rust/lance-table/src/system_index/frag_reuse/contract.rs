// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Invariants every fragment reuse history format upholds, run against each
//! one. Where a format's rule differs by design, the case states the
//! expected outcome for that format rather than moving it to a separate
//! suite.

use bytes::Bytes;
use lance_core::{Error, Result};
use prost::Message;
use roaring::RoaringBitmap;
use rstest::rstest;

use super::history::tests::{
    Source, legacy, ordered, transition_with_unknown_field, unknown_record, whole,
};
use super::history::{
    FragReuseHistory, HistoryFormat, LEGACY_VERSION_FIELD, RawHistory, Record, length_delimited,
};

/// Record one rewrite as the record kind `history`'s format writes.
fn record_natively(
    history: FragReuseHistory,
    dataset_version: u64,
    sources: &[Source],
    destinations: &[(u64, u64)],
) -> FragReuseHistory {
    match history.format() {
        HistoryFormat::Legacy => {
            history.insert_legacy_version(legacy(dataset_version, sources, destinations))
        }
        HistoryFormat::Tagged => history.append_transitions(&[ordered(sources, destinations)]),
    }
    .unwrap()
}

#[rstest]
#[case::legacy(HistoryFormat::Legacy)]
#[case::tagged(HistoryFormat::Tagged)]
fn empty_round_trips(#[case] format: HistoryFormat) {
    let empty = FragReuseHistory::empty(format);
    assert_eq!(empty.format(), format);
    assert!(empty.content().is_empty());
    assert!(empty.ledger().transitions().is_empty());
    assert!(empty.fragment_bitmap().is_empty());
    assert_eq!(
        RawHistory::new(format, empty.content()).validate().unwrap(),
        empty
    );
}

#[rstest]
#[case::legacy(HistoryFormat::Legacy)]
#[case::tagged(HistoryFormat::Tagged)]
fn encode_then_decode_round_trips(#[case] format: HistoryFormat) {
    let history = record_natively(
        FragReuseHistory::empty(format),
        1,
        &[whole(0, 2)],
        &[(1, 2)],
    );
    let history = record_natively(history, 2, &[whole(1, 2)], &[(2, 1), (3, 1)]);

    let decoded = RawHistory::new(format, history.content())
        .validate()
        .unwrap();
    assert_eq!(decoded, history);
    assert_eq!(decoded.fragment_bitmap(), history.fragment_bitmap());
    assert_eq!(decoded.ledger().transitions().len(), 2);
    let (repaired, dropped) = RawHistory::new(format, history.content()).repair().unwrap();
    assert_eq!(repaired, history);
    assert!(dropped.is_empty());
}

/// Only a committed version 0 history is repaired; validation of a
/// proposal, and both paths of a tagged history, are strict.
#[rstest]
#[case::legacy(HistoryFormat::Legacy, Some(vec![2]))]
#[case::tagged(HistoryFormat::Tagged, None)]
fn validate_refuses_what_repair_repairs(
    #[case] format: HistoryFormat,
    #[case] repaired_drops: Option<Vec<u64>>,
) {
    let first = legacy(1, &[whole(0, 2)], &[(1, 2)]);
    let reconsumes = legacy(2, &[whole(0, 2)], &[(2, 2)]);
    let content: Bytes = [&first, &reconsumes]
        .iter()
        .flat_map(|version| length_delimited(LEGACY_VERSION_FIELD, &version.encode_to_vec()))
        .collect::<Vec<u8>>()
        .into();

    let raw = RawHistory::new(format, content);
    let error = raw.clone().validate().unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
    match repaired_drops {
        Some(expected) => {
            let (repaired, dropped) = raw.repair().unwrap();
            assert_eq!(dropped, expected);
            assert_eq!(
                repaired,
                FragReuseHistory::from_legacy_versions(vec![first]).unwrap()
            );
        }
        None => {
            let error = raw.repair().unwrap_err();
            assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        }
    }
}

/// Version 0 names only destinations, as released writers did; tagged names
/// the whole retained lineage.
#[rstest]
#[case::legacy(HistoryFormat::Legacy, &[1, 2])]
#[case::tagged(HistoryFormat::Tagged, &[0, 1, 2])]
fn fragment_bitmap_follows_the_format(#[case] format: HistoryFormat, #[case] expected: &[u32]) {
    let history = record_natively(
        FragReuseHistory::empty(format),
        1,
        &[whole(0, 1)],
        &[(1, 1)],
    );
    let history = record_natively(history, 2, &[whole(1, 1)], &[(2, 1)]);
    assert_eq!(
        history.fragment_bitmap(),
        RoaringBitmap::from_iter(expected.iter().copied())
    );
}

#[test]
fn lifted_legacy_records_name_their_sources() {
    let v0 = FragReuseHistory::empty(HistoryFormat::Legacy)
        .insert_legacy_version(legacy(1, &[whole(0, 1)], &[(1, 1)]))
        .unwrap();
    assert_eq!(v0.fragment_bitmap(), RoaringBitmap::from_iter([1]));
    assert_eq!(
        v0.into_tagged().unwrap().fragment_bitmap(),
        RoaringBitmap::from_iter([0, 1])
    );
}

#[rstest]
#[case::legacy(HistoryFormat::Legacy)]
#[case::tagged(HistoryFormat::Tagged)]
fn each_format_refuses_the_other_record_kind(#[case] format: HistoryFormat) {
    let history = FragReuseHistory::empty(format);
    let error = match format {
        HistoryFormat::Legacy => history
            .append_transitions(&[ordered(&[whole(0, 1)], &[(1, 1)])])
            .unwrap_err(),
        HistoryFormat::Tagged => history
            .insert_legacy_version(legacy(1, &[whole(0, 1)], &[(1, 1)]))
            .unwrap_err(),
    };
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
}

#[rstest]
#[case::legacy(HistoryFormat::Legacy)]
#[case::tagged(HistoryFormat::Tagged)]
fn retain_drops_only_the_rejected_records(#[case] format: HistoryFormat) {
    let history = record_natively(
        FragReuseHistory::empty(format),
        1,
        &[whole(0, 1)],
        &[(1, 1)],
    );
    let history = record_natively(history, 2, &[whole(5, 1)], &[(6, 1)]);
    let trimmed = history
        .retain(&mut |record| match record {
            Record::Transition(transition) => transition.sources[0].id != 0,
            Record::LegacyVersion(version) => version.groups[0].old_fragments[0].id != 0,
            Record::Uninterpretable { .. } => true,
        })
        .unwrap();
    assert_eq!(trimmed.format(), format);
    assert_eq!(trimmed.ledger().transitions().len(), 1);
    assert_eq!(trimmed.ledger().consumer(5), Some(0));
    assert_eq!(trimmed.ledger().consumer(0), None);
}

/// A record only a newer writer understands is carried byte for byte, and
/// no edit that reasons about the whole history goes past it. Version 0 has
/// no transitions, so only an unknown envelope record applies to it.
#[rstest]
#[case::legacy_unknown_envelope_record(HistoryFormat::Legacy, unknown_record())]
#[case::tagged_unknown_envelope_record(HistoryFormat::Tagged, unknown_record())]
#[case::tagged_transition_with_unknown_field(
    HistoryFormat::Tagged,
    transition_with_unknown_field()
)]
fn whole_history_edits_refuse_records_they_cannot_interpret(
    #[case] format: HistoryFormat,
    #[case] record: Vec<u8>,
) {
    let content = Bytes::from(record);
    let history = RawHistory::new(format, content.clone()).validate().unwrap();
    assert_eq!(history.content(), content);
    assert!(!history.is_fully_interpretable());
    let assert_refused = |result: Result<FragReuseHistory>| {
        let error = result.unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(error.to_string().contains("cannot maintain"), "{error}");
    };
    assert_refused(history.retain(&mut |_| true));
    assert_refused(history.remap_stable_partition_bases(&Ok));
    if format == HistoryFormat::Tagged {
        assert_refused(history.append_transitions(&[ordered(&[whole(1, 1)], &[(2, 1)])]));
    }
}
