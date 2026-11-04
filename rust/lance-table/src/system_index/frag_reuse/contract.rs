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
    MAP_A, Source, address, legacy, ordered, partition, transition_with_unknown_field, translate,
    unknown_record, whole,
};
use super::history::{
    FragReuseHistory, HistoryFormat, LEGACY_VERSION_FIELD, RawHistory, Record, RewriteRecord,
    length_delimited,
};
use super::plan::{ReadContext, ReadPlan, RowDomain, SegmentSummary, TranslationSupport};
use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
use crate::format::pb::fragment_reuse_index_details as pb;
use crate::format::{DataFile, Fragment};
use lance_file::version::ConcreteFileVersion;

/// Record one rewrite of `sources` into `destinations` at `dataset_version`.
fn record_natively(
    history: FragReuseHistory,
    dataset_version: u64,
    sources: &[Source],
    destinations: &[(u64, u64)],
) -> FragReuseHistory {
    history
        .append(RewriteRecord::new(
            dataset_version,
            vec![ordered(sources, destinations)],
        ))
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
        .append(RewriteRecord::new(
            1,
            vec![ordered(&[whole(0, 1)], &[(1, 1)])],
        ))
        .unwrap();
    assert_eq!(v0.format(), HistoryFormat::Legacy);
    assert_eq!(v0.fragment_bitmap(), RoaringBitmap::from_iter([1]));
    assert_eq!(
        v0.lift().unwrap().fragment_bitmap(),
        RoaringBitmap::from_iter([0, 1])
    );
}

/// A version 0 history holds only ordered compactions: a stable partition
/// is refused until the history is lifted.
#[test]
fn legacy_refuses_a_transition_only_tagged_records() {
    let rewrite = RewriteRecord::new(1, vec![partition(0, 1, MAP_A)]);
    assert!(rewrite.needs_tagged_format());
    let legacy = FragReuseHistory::empty(HistoryFormat::Legacy);
    let error = legacy.append(rewrite.clone()).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("lift"), "{error}");
    let lifted = legacy.lift().unwrap().append(rewrite).unwrap();
    assert_eq!(lifted.format(), HistoryFormat::Tagged);
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
    assert_refused(history.remap_bases(&Ok));
    if format == HistoryFormat::Tagged {
        assert_refused(history.append(RewriteRecord::new(
            1,
            vec![ordered(&[whole(1, 1)], &[(2, 1)])],
        )));
    }
}

/// How a history is read: each format, and the legacy format on a table
/// with stable row IDs.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Reading {
    Legacy,
    LegacyStableRowIds,
    Tagged,
}

/// The version the rewrite in [`rewritten`] is recorded at, and the FRI
/// entry's version.
const REWRITE_VERSION: u64 = 5;

/// One rewrite at [`REWRITE_VERSION`]: fragments 0 and 3 (one row each)
/// into fragment 1. Fragment 2 is untouched; 1 and 2 are live.
fn rewritten(reading: Reading) -> (FragReuseHistory, RoaringBitmap) {
    let format = match reading {
        Reading::Legacy | Reading::LegacyStableRowIds => HistoryFormat::Legacy,
        Reading::Tagged => HistoryFormat::Tagged,
    };
    let history = record_natively(
        FragReuseHistory::empty(format),
        REWRITE_VERSION,
        &[whole(0, 1), whole(3, 1)],
        &[(1, 2)],
    );
    let history = match reading {
        Reading::LegacyStableRowIds => history.on_stable_row_ids().unwrap(),
        _ => history,
    };
    (history, RoaringBitmap::from_iter([1, 2]))
}

fn segment(
    covered: &[u32],
    dataset_version: u64,
    translation: TranslationSupport,
) -> SegmentSummary {
    SegmentSummary {
        uuid: uuid::Uuid::new_v4(),
        fragment_bitmap: Some(covered.iter().copied().collect()),
        dataset_version,
        domain: RowDomain::Address,
        translation,
    }
}

fn plan_one(reading: Reading, segment: SegmentSummary) -> ReadPlan {
    let (history, live) = rewritten(reading);
    let context = ReadContext {
        entry_version: REWRITE_VERSION,
        live_fragments: &live,
    };
    history
        .plan(&context, std::slice::from_ref(&segment))
        .remove(0)
}

/// A segment built after the rewrite, or covering none of its fragments,
/// is caught up and read directly. The comparison is strict: a segment built
/// at the rewrite's version read its sources.
#[rstest]
#[case::legacy(Reading::Legacy)]
#[case::legacy_stable_row_ids(Reading::LegacyStableRowIds)]
#[case::tagged(Reading::Tagged)]
fn caught_up_segment_is_direct(#[case] reading: Reading) {
    let (history, _) = rewritten(reading);
    let supported = TranslationSupport::Supported;
    for (covered, dataset_version, is_caught_up) in [
        (&[1, 2][..], REWRITE_VERSION + 1, true),
        (&[2][..], 1, true),
        // A fragment that is no longer live needs no translation either.
        (&[2, 7][..], 1, true),
        (&[1, 2][..], REWRITE_VERSION, false),
        (&[0, 3][..], 1, false),
    ] {
        let summary = segment(covered, dataset_version, supported);
        assert_eq!(
            history.is_caught_up(REWRITE_VERSION, &summary),
            is_caught_up,
            "{covered:?} at version {dataset_version}"
        );
        let plan = plan_one(reading, summary);
        assert_eq!(
            plan == ReadPlan::Direct,
            is_caught_up,
            "{covered:?} at version {dataset_version}: {plan:?}"
        );
    }
}

/// Every fragment a segment answers for is live, and either one it covers
/// itself or one a transition produced from fragments its index covers. An
/// excluded segment answers for nothing.
#[rstest]
#[case::legacy(Reading::Legacy)]
#[case::legacy_stable_row_ids(Reading::LegacyStableRowIds)]
#[case::tagged(Reading::Tagged)]
fn effective_coverage_never_claims_an_untranslatable_fragment(#[case] reading: Reading) {
    let (history, live) = rewritten(reading);
    let context = ReadContext {
        entry_version: REWRITE_VERSION,
        live_fragments: &live,
    };
    let supported = TranslationSupport::Supported;
    let unsupported = TranslationSupport::Unsupported;
    let groups = [
        vec![segment(&[0, 3], 1, supported)],
        vec![segment(&[0], 1, supported), segment(&[3], 1, supported)],
        vec![segment(&[0], 1, supported)],
        vec![
            segment(&[0, 3], 1, unsupported),
            segment(&[2], 1, supported),
        ],
        vec![segment(&[0, 3], 1, supported), segment(&[1], 6, supported)],
    ];
    for group in groups {
        let provenance: RoaringBitmap = group
            .iter()
            .flat_map(|segment| segment.fragment_bitmap.iter().flatten())
            .collect();
        let plans = history.plan(&context, &group);
        let coverage = history.effective_coverage(&context, &group);
        for ((segment, plan), coverage) in group.iter().zip(plans).zip(coverage) {
            let own = segment.fragment_bitmap.as_ref().unwrap();
            if plan == ReadPlan::Excluded {
                assert!(coverage.is_empty(), "{group:?}");
            }
            for fragment in &coverage {
                assert!(live.contains(fragment), "{group:?}: {fragment} is not live");
                let is_translated = history.ledger().producer(fragment).is_some_and(|producer| {
                    history.ledger().transitions()[producer]
                        .sources()
                        .iter()
                        .all(|source| provenance.contains(source.id as u32))
                });
                assert!(
                    own.contains(fragment) || is_translated,
                    "{group:?}: {fragment} cannot be translated to"
                );
            }
        }
    }
}

/// A segment is excluded only when it needs translation its kind cannot
/// provide. Legacy trusts every kind (fail-closed plugins come with the
/// `Remapper`); on a table with stable row IDs the legacy reader applies the
/// history to every segment it opens, so a kind that cannot apply it, or one
/// storing stable row IDs, is excluded even when caught up.
#[rstest]
#[case::legacy(Reading::Legacy, [false, false, false])]
#[case::legacy_stable_row_ids(Reading::LegacyStableRowIds, [true, true, true])]
#[case::tagged(Reading::Tagged, [true, false, false])]
fn excluded_only_when_translation_is_needed_and_unsupported(
    #[case] reading: Reading,
    #[case] is_excluded: [bool; 3],
) {
    let unsupported = TranslationSupport::Unsupported;
    let stale = segment(&[0, 3], 1, unsupported);
    let caught_up = segment(&[1], REWRITE_VERSION + 1, unsupported);
    let mut row_ids = segment(&[1], REWRITE_VERSION + 1, TranslationSupport::Supported);
    row_ids.domain = RowDomain::StableRowId;
    for (segment, is_excluded) in [stale, caught_up, row_ids].into_iter().zip(is_excluded) {
        let plan = plan_one(reading, segment.clone());
        assert_eq!(
            plan == ReadPlan::Excluded,
            is_excluded,
            "{segment:?}: {plan:?}"
        );
    }
    for covered in [&[0, 3][..], &[1], &[2], &[]] {
        let plan = plan_one(reading, segment(covered, 1, TranslationSupport::Supported));
        assert_ne!(plan, ReadPlan::Excluded, "{covered:?}");
    }
}

/// A segment with no remaining coverage is still translated, not excluded,
/// so readers that need the index itself (full text search takes its
/// tokenizer from it) still find it: one whose coverage was withdrawn, and,
/// where coverage is derived from the lineage, one covering only part of a
/// transition's sources.
#[rstest]
#[case::legacy(Reading::Legacy)]
#[case::legacy_stable_row_ids(Reading::LegacyStableRowIds)]
#[case::tagged(Reading::Tagged)]
fn empty_remaining_coverage_still_translates(#[case] reading: Reading) {
    let mut cases = vec![segment(
        &[],
        REWRITE_VERSION + 1,
        TranslationSupport::Supported,
    )];
    if reading == Reading::Tagged {
        cases.push(segment(&[0], 1, TranslationSupport::Supported));
    }
    for segment in cases {
        match plan_one(reading, segment.clone()) {
            ReadPlan::Translate(plan) => assert!(plan.coverage().is_empty(), "{segment:?}"),
            other => panic!("{segment:?}: {other:?}"),
        }
    }
}

/// Append, encode as an entry, then decode it as committed and as proposed:
/// each reads back the history that was encoded.
#[rstest]
#[case::legacy(HistoryFormat::Legacy)]
#[case::tagged(HistoryFormat::Tagged)]
fn appended_history_round_trips_through_its_entry(#[case] format: HistoryFormat) {
    let history = record_natively(
        FragReuseHistory::empty(format),
        1,
        &[whole(0, 2)],
        &[(1, 2)],
    );
    let history = record_natively(history, 2, &[whole(1, 2)], &[(2, 1), (3, 1)]);

    let encoded = history.encode(7);
    assert!(encoded.external.is_none());
    let entry = encoded.entry;
    assert_eq!(entry.index_version, format.index_version());
    assert_eq!(entry.dataset_version, 7);
    assert_eq!(entry.fragment_bitmap, Some(history.fragment_bitmap()));
    assert_eq!(FragReuseHistory::locate(&entry).unwrap(), None);
    assert_eq!(FragReuseHistory::decode(&entry, None).unwrap(), history);
    assert_eq!(
        FragReuseHistory::decode_proposed(&entry, None).unwrap(),
        history
    );
    let inline = history.encode_inline(7);
    assert_eq!(inline.index_details, entry.index_details);
}

/// A history above the inline limit spills; decoding takes exactly the bytes
/// the entry references.
#[rstest]
#[case::legacy(HistoryFormat::Legacy)]
#[case::tagged(HistoryFormat::Tagged)]
fn spilled_history_decodes_from_the_external_bytes(#[case] format: HistoryFormat) {
    let transitions = (0..5000)
        .map(|id| ordered(&[whole(id, 1)], &[(100_000 + id, 1)]))
        .collect();
    let history = FragReuseHistory::empty(format)
        .append(RewriteRecord::new(1, transitions))
        .unwrap();
    let encoded = history.encode(1);
    let external = encoded.external.expect("spilled");
    assert_eq!(external, history.content());
    let entry = encoded.entry;
    let file = FragReuseHistory::locate(&entry).unwrap().expect("external");
    assert_eq!(file.size, external.len() as u64);
    assert_eq!(file.offset, 0);
    assert_eq!(
        FragReuseHistory::decode(&entry, Some(external.clone())).unwrap(),
        history
    );

    let error = FragReuseHistory::decode(&entry, None).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    let error = FragReuseHistory::decode(&entry, Some(external.slice(1..))).unwrap_err();
    assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
}

/// A committed version 0 entry is repaired; the same entry proposed by a
/// caller is refused.
#[test]
fn only_a_committed_legacy_entry_is_repaired() {
    let first = legacy(1, &[whole(0, 2)], &[(1, 2)]);
    let reconsumes = legacy(2, &[whole(0, 2)], &[(2, 2)]);
    let content: Vec<u8> = [&first, &reconsumes]
        .iter()
        .flat_map(|version| length_delimited(LEGACY_VERSION_FIELD, &version.encode_to_vec()))
        .collect();
    let mut entry = FragReuseHistory::empty(HistoryFormat::Legacy).encode_inline(2);
    let mut details = entry.index_details.as_deref().unwrap().clone();
    details.value = length_delimited(1, &content);
    entry.index_details = Some(std::sync::Arc::new(details));

    assert_eq!(
        FragReuseHistory::decode(&entry, None).unwrap(),
        FragReuseHistory::from_legacy_versions(vec![first]).unwrap()
    );
    let error = FragReuseHistory::decode_proposed(&entry, None).unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("proposed"), "{error}");
}

fn overlaid_fragment(id: u64) -> Fragment {
    let mut fragment = Fragment::new(id);
    fragment.physical_rows = Some(1);
    fragment.overlays.push(DataOverlayFile {
        data_file: DataFile::new(
            "overlay.lance",
            vec![0],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        ),
        coverage: OverlayCoverage::dense(RoaringBitmap::from_iter([0u32])),
        committed_version: 1,
    });
    fragment
}

/// Append refuses what the history's format cannot record, and nothing
/// else. Rewriting an overlaid source bakes its values into the destination,
/// which tagged provenance would keep serving through translation (the
/// legacy format drops the destination from stale bitmaps instead). A
/// rewrite already recorded would consume its sources twice. A stable
/// partition needs the tagged format.
#[rstest]
#[case::legacy(HistoryFormat::Legacy, false, true)]
#[case::tagged(HistoryFormat::Tagged, true, false)]
fn append_refuses_what_the_format_cannot_record(
    #[case] format: HistoryFormat,
    #[case] refuses_overlays: bool,
    #[case] refuses_stable_partitions: bool,
) {
    let history = record_natively(
        FragReuseHistory::empty(format),
        1,
        &[whole(0, 1)],
        &[(1, 1)],
    );

    let overlaid = RewriteRecord::new(2, vec![ordered(&[whole(5, 1)], &[(6, 1)])])
        .with_source_fragments([&overlaid_fragment(5)]);
    match history.append(overlaid) {
        Err(error) if refuses_overlays => {
            assert!(matches!(error, Error::NotSupported { .. }), "{error}");
            assert!(error.to_string().contains("overlay"), "{error}");
        }
        Ok(_) if !refuses_overlays => {}
        other => panic!("{other:?}"),
    }

    let error = history
        .append(RewriteRecord::new(
            2,
            vec![ordered(&[whole(0, 1)], &[(1, 1)])],
        ))
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");

    match history.append(RewriteRecord::new(2, vec![partition(7, 8, MAP_A)])) {
        Err(error) if refuses_stable_partitions => {
            assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        }
        Ok(appended) if !refuses_stable_partitions => {
            assert_eq!(appended.ledger().transitions().len(), 2);
        }
        other => panic!("{other:?}"),
    }
}

/// Lifting reinterprets the legacy records without re-encoding them, so
/// every address translates exactly as before.
#[test]
fn lift_preserves_every_recorded_translation() {
    let partly_deleted = Source {
        id: 0,
        physical_rows: 3,
        deleted: [1].into(),
    };
    let history = record_natively(
        FragReuseHistory::empty(HistoryFormat::Legacy),
        1,
        &[partly_deleted, whole(1, 2)],
        &[(2, 2), (3, 2)],
    );
    let history = record_natively(history, 2, &[whole(2, 2)], &[(4, 2)]);
    let lifted = history.lift().unwrap();
    assert_eq!(lifted.format(), HistoryFormat::Tagged);
    assert_eq!(lifted.content(), history.content());
    assert_eq!(lifted.lift().unwrap(), lifted);

    assert_eq!(
        translate(lifted.ledger(), address(0, 0)),
        Some(address(4, 0))
    );
    for fragment in 0..5 {
        for offset in 0..3 {
            let addr = address(fragment, offset);
            assert_eq!(
                translate(lifted.ledger(), addr),
                translate(history.ledger(), addr),
                "address {addr:#x}"
            );
        }
    }
}

/// A shallow clone's relocation restamps stable-partition bases and leaves
/// every other part of the history as it was.
#[rstest]
#[case::legacy(HistoryFormat::Legacy)]
#[case::tagged(HistoryFormat::Tagged)]
fn remap_bases_changes_only_bases(#[case] format: HistoryFormat) {
    let history = record_natively(
        FragReuseHistory::empty(format),
        1,
        &[whole(0, 1)],
        &[(1, 1)],
    );
    let history = match format {
        HistoryFormat::Legacy => history,
        HistoryFormat::Tagged => history
            .append(RewriteRecord::new(2, vec![partition(5, 6, MAP_A)]))
            .unwrap(),
    };
    let remapped = history
        .remap_bases(&|base| {
            assert_eq!(base, None);
            Ok(Some(9))
        })
        .unwrap();
    assert_eq!(remapped.format(), history.format());
    assert_eq!(remapped.fragment_bitmap(), history.fragment_bitmap());
    let records: Vec<_> = history.records().cloned().collect();
    let remapped_records: Vec<_> = remapped.records().cloned().collect();
    assert_eq!(records.len(), remapped_records.len());
    for (record, remapped) in records.into_iter().zip(remapped_records) {
        let expected = match record {
            Record::Transition(mut transition) => {
                if let Some(pb::transition::Mapping::StablePartition(partition)) =
                    &mut transition.mapping
                {
                    partition.base_id = Some(9);
                }
                Record::Transition(transition)
            }
            other => other,
        };
        assert_eq!(remapped, expected);
    }
}
