// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::ops::Range;
use std::sync::Mutex;

use arrow_array::UInt16Array;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::utils::stable_partition::{CountsMatrix, CountsMatrixBuilder};
use prost::Message;
use roaring::RoaringTreemap;
use uuid::Uuid;

use super::*;
use crate::system_index::frag_reuse::history::tests::digest;
use crate::system_index::frag_reuse::history::{
    FragReuseHistory, HistoryFormat, LEGACY_VERSION_FIELD, RawHistory, TRANSITION_FIELD,
    length_delimited,
};
use crate::system_index::frag_reuse::plan::RowDomain;
use crate::system_index::frag_reuse::plan::TranslationSupport;

fn addr(fragment: u32, offset: u32) -> u64 {
    RowAddress::new_from_parts(fragment, offset).into()
}

fn bitmap(fragments: &[u32]) -> RoaringBitmap {
    fragments.iter().copied().collect()
}

fn digests(fragments: &[(u64, u64)]) -> Vec<pb::FragmentDigest> {
    fragments
        .iter()
        .map(|&(id, rows)| digest(id, rows, 0))
        .collect()
}

/// The serialized changed-row bitmap of a compaction moving every row of
/// `sources` (`(id, rows)` pairs).
fn changed_rows(sources: &[(u64, u64)]) -> Vec<u8> {
    let mut addrs = RoaringTreemap::new();
    for &(id, rows) in sources {
        for offset in 0..rows as u32 {
            addrs.insert(addr(id as u32, offset));
        }
    }
    let mut bytes = Vec::new();
    addrs.serialize_into(&mut bytes).unwrap();
    bytes
}

/// An ordered compaction of `sources` (`(id, rows)`) into `destinations`.
fn ordered_rows(sources: &[(u64, u64)], destinations: &[(u64, u64)]) -> pb::Transition {
    pb::Transition {
        sources: digests(sources),
        destinations: digests(destinations),
        mapping: Some(pb::transition::Mapping::OrderedCompaction(
            pb::OrderedCompaction {
                changed_row_addrs: changed_rows(sources),
            },
        )),
    }
}

/// An ordered compaction moving every row of four-row `sources` into
/// equally sized `destinations`, in scan order.
fn ordered(sources: &[u64], destinations: &[u64]) -> pb::Transition {
    let per_destination = 4 * sources.len() as u64 / destinations.len() as u64;
    ordered_rows(
        &sources.iter().map(|&id| (id, 4)).collect::<Vec<_>>(),
        &destinations
            .iter()
            .map(|&id| (id, per_destination))
            .collect::<Vec<_>>(),
    )
}

/// A stable partition with explicit `(id, rows)` fragments.
fn partition_rows(
    map_id: &str,
    sources: &[(u64, u64)],
    destinations: &[(u64, u64)],
) -> pb::Transition {
    pb::Transition {
        sources: digests(sources),
        destinations: digests(destinations),
        mapping: Some(pb::transition::Mapping::StablePartition(
            pb::StablePartition {
                map_id: map_id.to_string(),
                map_size_bytes: 1,
                base_id: None,
            },
        )),
    }
}

/// A stable partition of four-row `sources` into equally sized
/// `destinations`.
fn partition(sources: &[u64], destinations: &[u64]) -> pb::Transition {
    let per_destination = 4 * sources.len() as u64 / destinations.len() as u64;
    partition_rows(
        &Uuid::new_v4().to_string(),
        &sources.iter().map(|&id| (id, 4)).collect::<Vec<_>>(),
        &destinations
            .iter()
            .map(|&id| (id, per_destination))
            .collect::<Vec<_>>(),
    )
}

/// A v0 reuse version stamped `dataset_version` with one compaction group.
fn legacy_version(
    dataset_version: u64,
    sources: &[(u64, u64)],
    destinations: &[(u64, u64)],
) -> pb::Version {
    pb::Version {
        dataset_version,
        groups: vec![pb::Group {
            changed_row_addrs: changed_rows(sources),
            old_fragments: digests(sources),
            new_fragments: digests(destinations),
        }],
    }
}

/// A tagged history lifting `legacy_versions` beside its own `transitions`.
fn history_with_legacy(
    legacy_versions: Vec<pb::Version>,
    transitions: Vec<pb::Transition>,
) -> FragReuseHistory {
    let mut content = Vec::new();
    for version in legacy_versions {
        content.extend(length_delimited(
            LEGACY_VERSION_FIELD,
            &version.encode_to_vec(),
        ));
    }
    for transition in transitions {
        content.extend(length_delimited(
            TRANSITION_FIELD,
            &transition.encode_to_vec(),
        ));
    }
    RawHistory::new(HistoryFormat::Tagged, content.into())
        .validate()
        .unwrap()
}

fn history(transitions: Vec<pb::Transition>) -> FragReuseHistory {
    history_with_legacy(vec![], transitions)
}

/// Walk against a table where only the ledger retires fragments: live is
/// everything named minus every source, the lineage is everything the
/// ledger names.
fn plan(history: &FragReuseHistory, provenance: &[u32]) -> Walk {
    let ledger = history.ledger();
    let mut lineage = RoaringBitmap::new();
    let mut sources = RoaringBitmap::new();
    for transition in ledger.transitions() {
        sources |= ids(transition.sources());
        lineage |= ids(transition.sources());
        lineage |= ids(transition.destinations());
    }
    let provenance = bitmap(provenance);
    let live = (&provenance | &lineage) - &sources;
    walk(ledger, &provenance, &live, &lineage)
}

/// The hops of a compaction-only walk, composed into one remap.
fn composed(history: &FragReuseHistory, hops: &[RewriteHop]) -> RowAddrRemap {
    RowAddrRemap::chained(hops.iter().map(|hop| {
        match history.ledger().transitions()[hop.transition].mapping() {
            Mapping::OrderedCompaction(remap) => remap.as_ref().clone(),
            Mapping::StablePartition(_) => {
                panic!("hop {} needs a row map", hop.transition)
            }
        }
    }))
}

fn positions(hops: &[RewriteHop]) -> Vec<usize> {
    hops.iter().map(|hop| hop.transition).collect()
}

/// Two lifted v0 rounds (v10 `{2,3} -> D`, v12 `{0,1,D} -> X`) and a tagged
/// partition of X: the effective provenance of a segment whose bitmap v0
/// swapped to `{X}` depends on which rounds its stamp predates, unwound
/// newest first.
#[test]
fn effective_provenance_unwinds_legacy_swaps_by_stamp() {
    const D: u64 = 20;
    const X: u64 = 30;
    const Y: u64 = 40;
    let history = history_with_legacy(
        vec![
            legacy_version(10, &[(2, 4), (3, 4)], &[(D, 8)]),
            legacy_version(12, &[(0, 4), (1, 4), (D, 8)], &[(X, 16)]),
        ],
        vec![partition_rows(MAP, &[(X, 16)], &[(Y, 16)])],
    );
    let ledger = history.ledger();
    let stored = bitmap(&[X as u32]);
    // Built before both rounds: the file holds the original fragments'
    // addresses.
    assert_eq!(
        effective_provenance(ledger, &stored, 5),
        bitmap(&[0, 1, 2, 3])
    );
    // Built after the second round: the bitmap is the truth.
    assert_eq!(effective_provenance(ledger, &stored, 13), stored);
    // Built at the second round's watermark: it still predates the round.
    assert_eq!(
        effective_provenance(ledger, &stored, 12),
        effective_provenance(ledger, &stored, 11)
    );
    // Built between the rounds: only the second swap is unwound.
    assert_eq!(
        effective_provenance(ledger, &stored, 11),
        bitmap(&[0, 1, D as u32])
    );
    // A bitmap that already names the sources is left alone.
    let sources = bitmap(&[0, 1, 2, 3]);
    assert_eq!(effective_provenance(ledger, &sources, 5), sources);
    // The tagged partition is never unwound, whatever the stamp.
    let restamped = bitmap(&[Y as u32]);
    assert_eq!(effective_provenance(ledger, &restamped, 0), restamped);
}

/// A segment stamped after the lifted version was remapped (or built) after
/// that compaction: its destination bitmap is kept.
#[test]
fn caught_up_destination_bitmap_is_left_alone() {
    const D: u64 = 20;
    let history = history_with_legacy(
        vec![legacy_version(10, &[(2, 4), (3, 4)], &[(D, 8)])],
        vec![],
    );
    let ledger = history.ledger();
    let stored = bitmap(&[D as u32]);
    assert_eq!(effective_provenance(ledger, &stored, 11), stored);
    // Stamped at the watermark itself: still predates the compaction.
    assert_eq!(effective_provenance(ledger, &stored, 10), bitmap(&[2, 3]));
    // A bitmap naming only part of the destinations, or one of the sources
    // beside them, is not a v0 swap and is kept too.
    let mixed = bitmap(&[2, D as u32]);
    assert_eq!(effective_provenance(ledger, &mixed, 5), mixed);
}

const MAP: &str = "00000000-0000-0000-0000-0000000000c1";

#[test]
fn coverage_disjoint_from_the_ledger_is_current() {
    let history = history(vec![ordered(&[1], &[2])]);
    assert!(matches!(plan(&history, &[7]), Walk::Identity));
}

#[test]
fn single_hop_compaction_remaps() {
    let history = history(vec![ordered(&[1, 2], &[5, 6])]);
    let Walk::Remap { hops, coverage, .. } = plan(&history, &[1, 2]) else {
        panic!("expected a remap plan");
    };
    assert_eq!(coverage, bitmap(&[5, 6]));
    let remap = composed(&history, &hops);
    assert_eq!(remap.get(addr(1, 0)), Some(Some(addr(5, 0))));
    assert_eq!(remap.get(addr(2, 3)), Some(Some(addr(6, 3))));
    assert_eq!(remap.get(addr(9, 0)), None);
}

#[test]
fn multi_hop_compaction_composes() {
    let history = history(vec![ordered(&[1], &[2]), ordered(&[2], &[3])]);
    let Walk::Remap { hops, coverage, .. } = plan(&history, &[1]) else {
        panic!("expected a remap plan");
    };
    assert_eq!(coverage, bitmap(&[3]));
    assert_eq!(
        composed(&history, &hops).get(addr(1, 2)),
        Some(Some(addr(3, 2)))
    );
}

/// A fully covered stable partition is the free case: applied like a
/// compaction hop, its bitmap restamped onto the destinations.
#[test]
fn fully_covered_stable_partition_remaps() {
    let history = history(vec![partition(&[1], &[2])]);
    let Walk::Remap { hops, coverage, .. } = plan(&history, &[1]) else {
        panic!("expected a remap plan");
    };
    assert_eq!(coverage, bitmap(&[2]));
    assert_eq!(positions(&hops), vec![0]);
}

/// A partially covered stable partition cannot be restamped (the segment
/// owns only part of the destinations), so the whole plan blocks.
#[test]
fn partially_covered_stable_partition_blocks() {
    let history = history(vec![partition(&[5, 6], &[7, 8])]);
    let walk = plan(&history, &[5]);
    assert!(
        matches!(walk, Walk::Blocked { fragment: 5 }),
        "expected Blocked on fragment 5, got {walk:?}"
    );
}

/// A partial compaction reached AFTER a fully covered stable partition
/// blocks: the restamped file would hold addresses inside a fragment the
/// compaction consumes, while its bitmap no longer names the partition's
/// sources, so nothing would pin the mapping those addresses need.
#[test]
fn partial_compaction_after_restamp_blocks() {
    let history = history(vec![partition(&[1, 2], &[3, 4]), ordered(&[3, 5], &[6])]);
    let walk = plan(&history, &[1, 2]);
    assert!(
        matches!(walk, Walk::Blocked { fragment: 3 }),
        "expected Blocked on fragment 3, got {walk:?}"
    );
}

/// A partial compaction BEFORE a stable partition blocks only when the
/// chain is connected: here the partition consumes the compaction's
/// destination, which the straddle fallback removed from coverage.
#[test]
fn connected_partial_compaction_before_stable_partition_blocks() {
    let history = history(vec![
        ordered(&[1, 5], &[9]),
        partition_rows(MAP, &[(9, 8), (2, 4)], &[(3, 6), (4, 6)]),
    ]);
    let walk = plan(&history, &[1, 2]);
    assert!(
        matches!(walk, Walk::Blocked { fragment: 2 }),
        "expected Blocked on fragment 2, got {walk:?}"
    );
}

/// The same two hops on DISCONNECTED branches do not interact.
#[test]
fn disconnected_partial_compaction_does_not_block_stable_partition() {
    let history = history(vec![ordered(&[5, 7], &[8]), partition(&[1, 2], &[3, 4])]);
    let Walk::Remap { hops, coverage, .. } = plan(&history, &[1, 2, 5]) else {
        panic!("expected a remap plan");
    };
    assert_eq!(coverage, bitmap(&[3, 4]));
    let partition_position = history.ledger().consumer(1).unwrap();
    assert_eq!(positions(&hops), vec![partition_position]);
}

#[test]
fn downstream_stable_partition_composes() {
    let history = history(vec![ordered(&[1], &[2]), partition(&[2], &[3])]);
    let Walk::Remap { hops, coverage, .. } = plan(&history, &[1]) else {
        panic!("expected a remap plan");
    };
    assert_eq!(coverage, bitmap(&[3]));
    assert_eq!(positions(&hops), vec![0, 1]);
}

/// A segment covering only {3} of the chain 3,4 -> 5,6 -> 7,8 straddles the
/// first hop: the group's coverage is dropped (sources removed, destinations
/// NOT added) and nothing downstream applies.
#[test]
fn straddled_first_hop_drops_group_coverage() {
    let history = history(vec![ordered(&[3, 4], &[5, 6]), ordered(&[5, 6], &[7, 8])]);
    let Walk::Remap { hops, coverage, .. } = plan(&history, &[3]) else {
        panic!("expected a (coverage-only) remap plan");
    };
    assert!(coverage.is_empty());
    assert!(hops.is_empty(), "dropped rows are not moved");

    let Walk::Remap { hops, coverage, .. } = plan(&history, &[3, 4]) else {
        panic!("expected a remap plan");
    };
    assert_eq!(coverage, bitmap(&[7, 8]));
    assert_eq!(
        composed(&history, &hops).get(addr(3, 0)),
        Some(Some(addr(7, 0)))
    );
}

#[test]
fn straddle_drop_is_per_group() {
    let history = history(vec![ordered(&[1], &[2]), ordered(&[5, 6], &[7, 8])]);
    let Walk::Remap { hops, coverage, .. } = plan(&history, &[1, 5]) else {
        panic!("expected a remap plan");
    };
    assert_eq!(coverage, bitmap(&[2]));
    let remap = composed(&history, &hops);
    assert_eq!(remap.get(addr(1, 0)), Some(Some(addr(2, 0))));
    assert_eq!(remap.get(addr(5, 0)), None);
}

/// A retired fragment the entry's lineage records but the ledger no longer
/// mentions has lost its mapping: the plan blocks instead of treating its
/// addresses as deletions.
#[test]
fn retired_provenance_without_a_mapping_blocks() {
    let history = history(vec![ordered(&[1], &[2])]);
    let walk = walk(
        history.ledger(),
        &bitmap(&[1, 5]),
        &bitmap(&[2]),
        &bitmap(&[1, 2, 5]),
    );
    assert!(
        matches!(walk, Walk::Blocked { fragment: 5 }),
        "expected Blocked on fragment 5, got {walk:?}"
    );
}

/// The admitted set spans the retained lineage: provenance plus the
/// destinations of every applied hop, intermediates included. A withdrawn
/// source's transition is never applied, so nothing past it is admitted.
#[test]
fn admitted_spans_the_retained_lineage() {
    let chain = history(vec![partition(&[1, 2], &[3, 4]), ordered(&[3, 4], &[5, 6])]);
    let Walk::Remap {
        coverage, admitted, ..
    } = plan(&chain, &[1, 2])
    else {
        panic!("expected a remap plan");
    };
    assert_eq!(coverage, bitmap(&[5, 6]));
    assert_eq!(admitted, bitmap(&[1, 2, 3, 4, 5, 6]));

    let two_lineages = history(vec![ordered(&[1], &[3]), ordered(&[7], &[8])]);
    let Walk::Remap { admitted, .. } = plan(&two_lineages, &[7]) else {
        panic!("expected a remap plan");
    };
    assert_eq!(admitted, bitmap(&[7, 8]));
}

/// A retired fragment the ledger still names (a destination deleted
/// wholesale since) or never recorded is gone, not unmapped: the walk
/// proceeds and the bitmap swap narrows to what the reader still serves.
#[test]
fn deleted_fragments_do_not_block() {
    let history = history(vec![partition(&[1], &[2, 3])]);
    let Walk::Remap {
        coverage: claimed, ..
    } = walk(
        history.ledger(),
        &bitmap(&[1, 7]),
        &bitmap(&[3]),
        &bitmap(&[1, 2, 3]),
    )
    else {
        panic!("expected a remap plan");
    };
    assert_eq!(claimed, bitmap(&[2, 3, 7]));
}

/// Row maps held in memory, recording the reads translation makes.
#[derive(Debug, Default)]
struct InMemoryMappings {
    maps: HashMap<String, (Arc<CountsMatrix>, UInt16Array)>,
    reads: Mutex<Vec<(String, Range<u64>)>>,
}

impl InMemoryMappings {
    fn with_map(mut self, map_id: &str, labels: &[Option<u16>], destinations: u32) -> Self {
        let mut counts = CountsMatrixBuilder::try_new(destinations, 2).unwrap();
        for &label in labels {
            counts.push(label).unwrap();
        }
        self.maps.insert(
            map_id.to_string(),
            (
                Arc::new(counts.finish()),
                UInt16Array::from(labels.to_vec()),
            ),
        );
        self
    }

    fn reads_of(&self, map_id: &str) -> usize {
        self.reads
            .lock()
            .unwrap()
            .iter()
            .filter(|(id, _)| id == map_id)
            .count()
    }
}

#[async_trait::async_trait]
impl MappingSource for InMemoryMappings {
    async fn counts(&self, mapping: &MappingRef) -> Result<Arc<CountsMatrix>> {
        Ok(self.maps[mapping.map_id()].0.clone())
    }

    async fn read(&self, mapping: &MappingRef, rows: Range<u64>) -> Result<UInt16Array> {
        self.reads
            .lock()
            .unwrap()
            .push((mapping.map_id().to_string(), rows.clone()));
        Ok(self.maps[mapping.map_id()]
            .1
            .slice(rows.start as usize, (rows.end - rows.start) as usize))
    }
}

const SP1: &str = "00000000-0000-0000-0000-0000000000d1";
const SP2: &str = "00000000-0000-0000-0000-0000000000d2";

/// C1 F0 -> F2; SP1 keeps F2's even rows in F4 and deletes the odd ones;
/// C2 F4 -> F6; SP2 sends F6's row 0 to F9 and row 1 to F8. Fragment 11
/// is never rewritten.
fn chain() -> (FragReuseHistory, Arc<InMemoryMappings>) {
    let history = history(vec![
        ordered_rows(&[(0, 4)], &[(2, 4)]),
        pb::Transition {
            // Rows 1 and 3 were deleted before the partition.
            sources: vec![digest(2, 4, 2)],
            ..partition_rows(SP1, &[], &[(4, 2)])
        },
        ordered_rows(&[(4, 2)], &[(6, 2)]),
        partition_rows(SP2, &[(6, 2)], &[(8, 1), (9, 1)]),
    ]);
    let mappings = InMemoryMappings::default()
        .with_map(SP1, &[Some(0), None, Some(0), None], 1)
        .with_map(SP2, &[Some(1), Some(0)], 2);
    (history, Arc::new(mappings))
}

fn compose_chain(history: &FragReuseHistory, provenance: &[u32]) -> TranslationPlan {
    let live = bitmap(&[8, 9, 11]);
    let lineage = history.fragment_bitmap();
    let rows = HashMap::from([(8, 1), (9, 1), (11, 6)]);
    let context = RewriteContext {
        live_fragments: &live,
        lineage: &lineage,
        fragment_rows: &rows,
        served: None,
    };
    let segment = SegmentSummary {
        uuid: Uuid::new_v4(),
        fragment_bitmap: Some(bitmap(provenance)),
        dataset_version: 1,
        domain: RowDomain::Address,
        translation: TranslationSupport::Supported,
    };
    match history.compose(&context, &segment).unwrap() {
        Composition::Rewrite(plan) => plan,
        other => panic!("expected a rewrite, got {other:?}"),
    }
}

/// Hops apply in ledger order per batch (a compaction after a partition
/// sees the partition's output), addresses outside every hop pass through,
/// deleted rows stay deleted, and input order and duplicates survive one to
/// one. Each row map is read once for the batch.
#[tokio::test]
async fn rewrite_hops_apply_in_ledger_order_per_batch() {
    let (history, mappings) = chain();
    let plan = compose_chain(&history, &[0, 11]);
    assert_eq!(plan.coverage(), &bitmap(&[8, 9, 11]));
    assert!(plan.moves_addresses());
    let remapper = plan.bind(mappings.clone());
    let mut addrs = vec![
        Some(addr(0, 0)),
        Some(addr(0, 1)),
        Some(addr(11, 5)),
        Some(addr(0, 0)),
        Some(addr(0, 2)),
        None,
    ];
    remapper.translate_in_place(&mut addrs).await.unwrap();
    assert_eq!(
        addrs,
        vec![
            Some(addr(9, 0)),
            None,
            Some(addr(11, 5)),
            Some(addr(9, 0)),
            Some(addr(8, 0)),
            None,
        ]
    );
    assert_eq!(mappings.reads_of(SP1), 1);
    assert_eq!(mappings.reads_of(SP2), 1);
    assert!(remapper.fragment_needs_translation(0));
    assert!(!remapper.fragment_needs_translation(11));
    assert_eq!(remapper.translate_fragments(&bitmap(&[0])), bitmap(&[8, 9]));
}

/// Only admitted addresses enter the hops, and only claimed fragments
/// survive them: restricting the coverage (a sibling took a fragment over)
/// drops the rows that land there.
#[tokio::test]
async fn rewrite_drops_what_it_neither_admits_nor_claims() {
    let (history, mappings) = chain();
    let plan = compose_chain(&history, &[0, 11]).restrict_coverage(&bitmap(&[8, 11]));
    assert_eq!(plan.coverage(), &bitmap(&[8, 11]));
    let remapper = plan.bind(mappings);
    let mut addrs = vec![
        Some(addr(0, 0)),
        Some(addr(0, 2)),
        Some(addr(5, 0)),
        Some(addr(11, 1)),
    ];
    remapper.translate_in_place(&mut addrs).await.unwrap();
    assert_eq!(addrs, vec![None, Some(addr(8, 0)), None, Some(addr(11, 1))]);
}

/// The bound rewrite sizes the fragments it translates from the history's
/// digests, so materializing needs nothing from the caller beyond the
/// fragments the index stores, and an untouched fragment is not enumerated.
#[tokio::test]
async fn rewrite_materializes_from_the_histories_digests() {
    let (history, mappings) = chain();
    let remapper = compose_chain(&history, &[0, 11]).bind(mappings);
    let remap = remapper.materialize(Some(&bitmap(&[0, 11]))).await.unwrap();
    assert_eq!(remap.get(addr(0, 0)), Some(Some(addr(9, 0))));
    assert_eq!(remap.get(addr(0, 1)), Some(None));
    assert_eq!(remap.get(addr(0, 2)), Some(Some(addr(8, 0))));
    assert_eq!(remap.get(addr(0, 3)), Some(None));
    assert_eq!(remap.get(addr(11, 0)), None);
}

/// A rewrite that only drops coverage moves nothing.
#[test]
fn straddle_only_rewrite_moves_no_address() {
    let history = history(vec![ordered(&[3, 4], &[5])]);
    let live = bitmap(&[5]);
    let lineage = history.fragment_bitmap();
    let rows = HashMap::new();
    let context = RewriteContext {
        live_fragments: &live,
        lineage: &lineage,
        fragment_rows: &rows,
        served: None,
    };
    let segment = SegmentSummary {
        uuid: Uuid::new_v4(),
        fragment_bitmap: Some(bitmap(&[3])),
        dataset_version: 1,
        domain: RowDomain::Address,
        translation: TranslationSupport::Supported,
    };
    let Composition::Rewrite(plan) = history.compose(&context, &segment).unwrap() else {
        panic!("expected a coverage-only rewrite");
    };
    assert!(!plan.moves_addresses());
    assert!(plan.coverage().is_empty());
}

fn compose_one(
    history: &FragReuseHistory,
    fragment_bitmap: Option<RoaringBitmap>,
    served: Option<&RoaringBitmap>,
) -> Composition {
    let live = bitmap(&[5, 6]);
    let lineage = history.fragment_bitmap();
    let rows = HashMap::new();
    let context = RewriteContext {
        live_fragments: &live,
        lineage: &lineage,
        fragment_rows: &rows,
        served,
    };
    let segment = SegmentSummary {
        uuid: Uuid::new_v4(),
        fragment_bitmap,
        dataset_version: 1,
        domain: RowDomain::Address,
        translation: TranslationSupport::Supported,
    };
    history.compose(&context, &segment).unwrap()
}

/// A rewrite that moves addresses publishes only what the reader serves the
/// segment for; one that only drops coverage moves nothing the reader could
/// cede, so the reader's coverage does not narrow it.
#[test]
fn a_moving_rewrite_publishes_only_what_the_reader_serves() {
    let history = history(vec![ordered(&[3, 4], &[5, 6])]);
    let served = bitmap(&[5]);
    let Composition::Rewrite(plan) = compose_one(&history, Some(bitmap(&[3, 4])), Some(&served))
    else {
        panic!("a covered compaction is rewritten");
    };
    assert!(plan.moves_addresses());
    assert_eq!(plan.published_bitmap(), Some(&bitmap(&[5])));

    let history = history_with_legacy(vec![], vec![ordered(&[3, 4], &[5])]);
    let served = bitmap(&[]);
    let Composition::Rewrite(plan) = compose_one(&history, Some(bitmap(&[3, 7])), Some(&served))
    else {
        panic!("a straddled compaction is rewritten");
    };
    assert!(!plan.moves_addresses());
    assert_eq!(plan.published_bitmap(), Some(&bitmap(&[7])));
}

/// Without a bitmap the history cannot tell which addresses moved.
#[test]
fn a_segment_without_a_bitmap_has_no_known_lineage() {
    let history = history(vec![ordered(&[3, 4], &[5, 6])]);
    assert_eq!(
        compose_one(&history, None, None),
        Composition::LineageUnknown
    );
}
