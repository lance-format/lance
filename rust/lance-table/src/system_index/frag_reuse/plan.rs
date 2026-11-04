// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read planning: how a query reads each index segment under a fragment
//! reuse history.
//!
//! A segment's stored row addresses were written when it was built. A rewrite
//! the history records since then may have moved those rows, so a reader
//! either uses the segment as is ([`ReadPlan::Direct`]), translates its
//! addresses through the history ([`ReadPlan::Translate`]), or, when its kind
//! cannot translate, leaves it out and scans its fragments instead
//! ([`ReadPlan::Excluded`]).
//!
//! The plan is a pure function of the history, the segment's own metadata
//! ([`SegmentSummary`]) and what the reader knows about the table
//! ([`ReadContext`]). Segments are planned per logical index, since a
//! fragment one segment covers directly is not translated into for its
//! siblings.

use std::collections::HashMap;
use std::sync::Arc;

use lance_select::Remapper;
use roaring::RoaringBitmap;
use uuid::Uuid;

use super::history::FragReuseHistory;
use super::impls::FriImpl;
use super::ledger::FragReuseLedger;
use super::mapping::MappingSource;

/// What the identifiers a segment stores refer to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowDomain {
    /// Physical row addresses: a rewrite moves them.
    Address,
    /// Stable row IDs: a rewrite leaves them in place.
    StableRowId,
}

/// Whether a segment's index kind can translate its stored addresses through
/// the history. The caller computes it from the index kind; planning only
/// consults it for a segment that needs translation
/// ([`super::history::FragReuseHistory::needs_translation_support`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranslationSupport {
    /// The kind translates its addresses through the history.
    Supported,
    /// The kind cannot translate its addresses.
    Unsupported,
    /// The segment's metadata cannot be interpreted (missing or unknown
    /// details), so it is not known whether the kind translates. Planned as
    /// [`Self::Unsupported`]; maintenance refuses such a segment rather than
    /// skipping it.
    Uninterpretable,
}

/// What an index segment knows about itself: the input to every
/// per-segment question the history answers.
#[derive(Debug, Clone)]
pub struct SegmentSummary {
    pub uuid: Uuid,
    /// The fragments the segment's files were built over (its provenance).
    /// `None` when an older writer recorded no coverage, which no read plan
    /// can trust.
    pub fragment_bitmap: Option<RoaringBitmap>,
    /// The dataset version the segment's data was read at.
    pub dataset_version: u64,
    pub domain: RowDomain,
    pub translation: TranslationSupport,
}

/// What read planning needs to know about the table, beyond the history.
#[derive(Debug, Clone, Copy)]
pub struct ReadContext<'a> {
    /// `dataset_version` of the FRI entry holding the history. Every write of
    /// the entry stamps the version it is committed on top of, so no
    /// transition the history records was committed after it.
    pub entry_version: u64,
    /// The fragments of the snapshot being read.
    pub live_fragments: &'a RoaringBitmap,
}

/// How a reader uses one index segment.
#[derive(Debug, Clone, PartialEq)]
pub enum ReadPlan {
    /// The segment is caught up with the history: its addresses are used as
    /// stored, and it answers for the fragments it covers that are still
    /// live.
    Direct,
    /// The segment's addresses are translated through the plan. Its
    /// remaining coverage may be empty; such a segment is still listed, so
    /// readers that need the index itself (full text search takes its
    /// tokenizer from it) find it, and every fragment it covered is scanned.
    Translate(TranslationPlan),
    /// The segment needs translation its kind cannot provide, or the history
    /// cannot be interpreted: it is left out, and its fragments are scanned.
    Excluded,
}

/// How one segment's addresses are translated.
///
/// Opaque to readers: they [`bind`](Self::bind) it to the source of the
/// history's external mappings and hand the resulting [`Remapper`] to the
/// segment's index.
#[derive(Clone)]
pub struct TranslationPlan {
    history: Arc<dyn FriImpl>,
    coverage: RoaringBitmap,
    excluded_fragments: RoaringBitmap,
    stops: RoaringBitmap,
    fingerprint: [u8; 32],
    /// Set on a plan [`FragReuseHistory::compose`] made: the hops a rewrite
    /// moves the segment's stored addresses through.
    rewrite: Option<Arc<RewritePath>>,
}

/// The hops a rewrite moves a segment's stored addresses through, in
/// lineage order, as [`FragReuseHistory::compose`] planned them.
#[derive(Debug)]
pub(super) struct RewritePath {
    pub hops: Vec<RewriteHop>,
    /// Every fragment the segment's rows may sit in along the retained
    /// lineage; an address anywhere else is dropped before the first hop.
    pub admitted: RoaringBitmap,
    /// Physical row counts of every fragment the history or the snapshot
    /// knows, for [`lance_select::Remapper::materialize`].
    pub fragment_rows: HashMap<u32, u64>,
}

/// One hop of a [`RewritePath`]: the position of a transition in the
/// ledger and the fragments of its sources the segment's addresses occupy
/// when they reach it. Only addresses in those fragments enter the hop.
#[derive(Debug)]
pub(super) struct RewriteHop {
    pub transition: usize,
    pub enter: RoaringBitmap,
}

/// What [`FragReuseHistory::compose`] needs to know about the table,
/// beyond the history and the segment.
#[derive(Debug, Clone, Copy)]
pub struct RewriteContext<'a> {
    /// The fragments of the snapshot being rewritten.
    pub live_fragments: &'a RoaringBitmap,
    /// The FRI entry's `fragment_bitmap`, the lineage it recorded. A
    /// fragment it names that the history no longer maps lost its mapping.
    pub lineage: &'a RoaringBitmap,
    /// The physical row counts of the snapshot's fragments.
    pub fragment_rows: &'a HashMap<u32, u64>,
}

/// How [`FragReuseHistory::compose`] answers for one segment.
#[derive(Debug, Clone, PartialEq)]
pub enum Composition {
    /// No recorded transition moves the segment's stored addresses.
    CaughtUp,
    /// The segment cannot be rewritten through the history: at `fragment`
    /// its coverage reaches a transition it only partially covers (a stable
    /// partition, or any hop after a stable-partition restamp), or a
    /// retired fragment the history no longer maps. It keeps translating
    /// through the history until it is rebuilt.
    Blocked { fragment: u32 },
    /// The segment is rewritten through the plan: bind it and translate the
    /// segment's files through the [`Remapper`], then publish the plan's
    /// [`coverage`](TranslationPlan::coverage) as the segment's bitmap.
    Rewrite(TranslationPlan),
}

impl PartialEq for TranslationPlan {
    fn eq(&self, other: &Self) -> bool {
        self.coverage == other.coverage
            && self.excluded_fragments == other.excluded_fragments
            && self.stops == other.stops
            && self.fingerprint == other.fingerprint
    }
}

impl std::fmt::Debug for TranslationPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranslationPlan")
            .field("format", &self.history.format())
            .field("coverage", &self.coverage)
            .field("excluded_fragments", &self.excluded_fragments)
            .field("stops", &self.stops)
            .finish_non_exhaustive()
    }
}

impl TranslationPlan {
    /// A plan under `history` translating every address to nothing: the
    /// segment answers for no fragment. Maintenance reads a segment the
    /// reader derives no coverage for through it.
    pub fn empty(history: &FragReuseHistory) -> Self {
        Self::nothing(history.imp().clone())
    }

    pub(super) fn nothing(history: Arc<dyn FriImpl>) -> Self {
        let empty = RoaringBitmap::new();
        Self::new(history, empty.clone(), empty.clone(), empty, &[])
    }

    pub(super) fn new(
        history: Arc<dyn FriImpl>,
        coverage: RoaringBitmap,
        excluded_fragments: RoaringBitmap,
        stops: RoaringBitmap,
        path: &[usize],
    ) -> Self {
        let fingerprint = translation_fingerprint(
            &history.records().ledger,
            &coverage,
            &excluded_fragments,
            path,
        );
        Self {
            history,
            coverage,
            excluded_fragments,
            stops,
            fingerprint,
            rewrite: None,
        }
    }

    /// A rewrite plan publishing `coverage`, whose addresses move through
    /// `path`.
    pub(super) fn rewrite(
        history: Arc<dyn FriImpl>,
        coverage: RoaringBitmap,
        path: RewritePath,
    ) -> Self {
        let mut plan = Self::nothing(history);
        plan.coverage = coverage;
        plan.rewrite = Some(Arc::new(path));
        plan
    }

    pub(super) fn rewrite_path(&self) -> Option<&Arc<RewritePath>> {
        self.rewrite.as_ref()
    }

    /// Whether binding the plan moves any stored address. A rewrite plan
    /// that only drops coverage (a partially covered compaction leaves its
    /// rows to the scan) changes the segment's bitmap, not its files.
    pub fn moves_addresses(&self) -> bool {
        self.rewrite
            .as_ref()
            .is_none_or(|path| !path.hops.is_empty())
    }

    /// The same plan answering only for the fragments of `fragments` it
    /// already covers: a rewrite drops every address the published bitmap
    /// does not claim, so the rewritten files carry nothing it leaves out
    /// (rows a sibling segment covers directly, fragments an overlay made
    /// stale).
    pub fn restrict_coverage(mut self, fragments: &RoaringBitmap) -> Self {
        self.coverage &= fragments;
        self
    }

    /// The translation an index loads the segment with. `mappings` reads
    /// the external mappings the plan's path passes through, when the
    /// history has any, lazily and once per translated batch.
    pub fn bind(self, mappings: Arc<dyn MappingSource>) -> Remapper {
        self.history.clone().bind(self, mappings)
    }

    /// The live fragments the segment answers for: its coverage in the
    /// index listing. A reader also uses it to tell why a translated segment
    /// answers for nothing.
    pub fn coverage(&self) -> &RoaringBitmap {
        &self.coverage
    }

    /// Fragments other segments of the same index cover directly: a
    /// translated path entering one of them is dropped. Exposed so a reader
    /// caching plans can size them.
    pub fn excluded_fragments(&self) -> &RoaringBitmap {
        &self.excluded_fragments
    }

    /// Live fragments the history still records as rewritten: a translated
    /// path stops at one rather than following the rewrite.
    pub(super) fn stops(&self) -> &RoaringBitmap {
        &self.stops
    }

    /// The translation identity: a hash of everything that decides the
    /// translated output, so cached objects that embed translated addresses
    /// can be keyed by it and go cold only when that state changes.
    pub fn fingerprint(&self) -> &[u8; 32] {
        &self.fingerprint
    }
}

/// One segment's derived query-time inputs from the coverage backtrack.
#[derive(Clone, Debug, Default, PartialEq)]
#[doc(hidden)]
pub struct SegmentPlanParts {
    /// Live fragments this segment answers for in the current snapshot.
    pub coverage: RoaringBitmap,
    /// Fragments directly covered by other selected group members; translated
    /// paths entering them belong to those siblings.
    pub excluded: RoaringBitmap,
    /// Positions (in ledger order) of the transitions this segment's rows
    /// pass through on their way to `coverage`: every transition whose
    /// resolved contributors include this segment.
    pub path: Vec<usize>,
}

/// Resolve each segment's query coverage and sibling exclusions in one pass,
/// so "direct coverage wins" is owned by one algorithm.
///
/// Coverage: each live fragment backtracks to the nearest available index
/// coverage. A mapping requires complete source coverage; failure for one
/// destination does not discard another destination's coverage. Every
/// contributing segment is retained.
///
/// Exclusions: the fragments directly covered by other group members,
/// derived from the same `direct` map the backtrack builds. This is
/// deliberately an over-approximation of each segment's true sibling
/// contention: fragments that are not on a path this segment translates are
/// never checked per-hop, so extra members are inert. Do not "optimize"
/// this to exact per-path sets.
#[doc(hidden)]
pub fn segment_plans(
    ledger: &FragReuseLedger,
    live_fragments: &RoaringBitmap,
    provenance: &[RoaringBitmap],
) -> Vec<SegmentPlanParts> {
    let mut direct: HashMap<u32, Vec<usize>> = HashMap::new();
    for (segment, bitmap) in provenance.iter().enumerate() {
        for fragment in bitmap {
            direct.entry(fragment).or_default().push(segment);
        }
    }
    // Direct coverage takes precedence: a fragment resolves to its direct
    // segments when present and to its producing transition otherwise. The
    // transition resolution is shared by all of that transition's
    // destinations; None records incomplete source coverage.
    let mut transitions: HashMap<usize, Option<Vec<usize>>> = HashMap::new();
    let mut coverage = vec![RoaringBitmap::new(); provenance.len()];
    let mut pending = Vec::new();
    for destination in live_fragments.iter() {
        let producer = match (direct.get(&destination), ledger.producer(destination)) {
            (Some(segments), _) => {
                for &segment in segments {
                    coverage[segment].insert(destination);
                }
                continue;
            }
            (None, Some(producer)) => producer,
            (None, None) => continue,
        };
        pending.push((producer, false));
        while let Some((index, expanded)) = pending.pop() {
            if transitions.contains_key(&index) {
                continue;
            }
            let sources = ledger.transitions()[index].sources();
            if !expanded {
                pending.push((index, true));
                pending.extend(sources.iter().filter_map(|source| {
                    let source = source.id as u32;
                    (!direct.contains_key(&source))
                        .then(|| ledger.producer(source))
                        .flatten()
                        .map(|producer| (producer, false))
                }));
                continue;
            }
            let mut segments = Vec::new();
            let complete = sources.iter().all(|source| {
                let source = source.id as u32;
                let contributors = direct.get(&source).or_else(|| {
                    ledger
                        .producer(source)
                        .and_then(|producer| transitions.get(&producer))
                        .and_then(|resolution| resolution.as_ref())
                });
                match contributors {
                    Some(contributors) => {
                        segments.extend_from_slice(contributors);
                        true
                    }
                    None => false,
                }
            });
            let resolution = complete.then(|| {
                segments.sort_unstable();
                segments.dedup();
                segments
            });
            transitions.insert(index, resolution);
        }
        if let Some(Some(segments)) = transitions.get(&producer) {
            for &segment in segments {
                coverage[segment].insert(destination);
            }
        }
    }
    let group_direct: RoaringBitmap = direct.keys().copied().collect();
    // Invert the resolutions: a transition whose contributors include a
    // segment is on that segment's translation path.
    let mut paths = vec![Vec::new(); provenance.len()];
    for (&index, resolution) in &transitions {
        for &segment in resolution.iter().flatten() {
            paths[segment].push(index);
        }
    }
    provenance
        .iter()
        .zip(coverage)
        .zip(paths)
        .map(|((own, coverage), mut path)| {
            path.sort_unstable();
            SegmentPlanParts {
                coverage,
                excluded: &group_direct - own,
                path,
            }
        })
        .collect()
}

/// The translation identity of one segment (see
/// [`TranslationPlan::fingerprint`]).
///
/// Translation is a function of the mappings on the segment's path, the
/// liveness of the fragments along it, `excluded` and `coverage`. The hash
/// therefore covers `coverage` (live fragments only, so a fragment dropped
/// from the manifest changes it), `excluded` restricted to the destinations
/// of the path (sibling churn elsewhere in the group cannot change this
/// segment's output), the fingerprints of the path transitions in ledger
/// order (a trimmed or replaced transition on the path changes it; one
/// elsewhere does not) and whether the ledger dropped unsupported
/// transitions. The FRI entry's UUID is deliberately not part of it: every
/// rewrite and trim mints a new one, even for transitions this segment never
/// touches.
#[doc(hidden)]
pub fn translation_fingerprint(
    ledger: &FragReuseLedger,
    coverage: &RoaringBitmap,
    excluded: &RoaringBitmap,
    path: &[usize],
) -> [u8; 32] {
    let transitions = ledger.transitions();
    let path_destinations: RoaringBitmap = path
        .iter()
        .flat_map(|&index| transitions[index].destinations())
        .map(|destination| destination.id as u32)
        .collect();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"fri-translation/1");
    hasher.update(&[u8::from(ledger.has_unsupported_transitions())]);
    hasher.update(b"coverage");
    // The serialized form is canonical for a given set; the hasher
    // implements `io::Write`, so nothing is buffered.
    coverage
        .serialize_into(&mut hasher)
        .expect("hashing a bitmap cannot fail");
    hasher.update(b"excluded");
    (excluded & path_destinations)
        .serialize_into(&mut hasher)
        .expect("hashing a bitmap cannot fail");
    hasher.update(b"path");
    for &index in path {
        hasher.update(transitions[index].fingerprint());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_index::frag_reuse::history::tests::{ordered, whole};
    use crate::system_index::frag_reuse::history::{
        FragReuseHistory, HistoryFormat, RewriteRecord,
    };

    /// A,B -> C,D -> E,F. Each destination holds one row. Coverage is
    /// conservative for each mapping and requires both source fragments.
    fn chain() -> FragReuseHistory {
        FragReuseHistory::empty(HistoryFormat::Tagged)
            .append(RewriteRecord::new(
                1,
                vec![
                    ordered(&[whole(0, 1), whole(1, 1)], &[(2, 1), (3, 1)]),
                    ordered(&[whole(2, 1), whole(3, 1)], &[(4, 1), (5, 1)]),
                ],
            ))
            .unwrap()
    }

    fn bitmaps(values: Vec<Vec<u32>>) -> Vec<RoaringBitmap> {
        values.into_iter().map(RoaringBitmap::from_iter).collect()
    }

    #[rstest::rstest]
    #[case::nearest_coverage(vec![vec![0], vec![1], vec![2, 3]], vec![vec![], vec![], vec![4, 5]])]
    #[case::mixed_depths(vec![vec![0], vec![1], vec![2]], vec![vec![4, 5], vec![4, 5], vec![4, 5]])]
    #[case::independent_destination(vec![vec![0], vec![4]], vec![vec![], vec![4]])]
    #[case::direct_and_derived(vec![vec![0], vec![1], vec![4]], vec![vec![5], vec![5], vec![4]])]
    fn coverage_resolves_each_destination_independently(
        #[case] provenance: Vec<Vec<u32>>,
        #[case] expected: Vec<Vec<u32>>,
    ) {
        let history = chain();
        let live = RoaringBitmap::from_iter([4, 5]);
        let coverage: Vec<RoaringBitmap> =
            segment_plans(history.ledger(), &live, &bitmaps(provenance))
                .into_iter()
                .map(|parts| parts.coverage)
                .collect();
        assert_eq!(coverage, bitmaps(expected));
    }

    #[test]
    fn segment_plans_excludes_the_sibling_direct_union() {
        let history = chain();
        let ledger = history.ledger();
        let live = RoaringBitmap::from_iter([4, 5]);
        // Segments X, Y, Z with mixed direct and derived coverage.
        let provenance = bitmaps(vec![vec![0], vec![1], vec![4]]);
        let plans = segment_plans(ledger, &live, &provenance);
        // Exclusions equal the sibling-union formula: everything any group
        // member covers directly, minus the segment's own provenance.
        let group: RoaringBitmap = provenance.iter().flatten().collect();
        for (plan, own) in plans.iter().zip(&provenance) {
            assert_eq!(plan.excluded, &group - own);
        }
        assert_eq!(plans[0].excluded, RoaringBitmap::from_iter([1, 4]));
        assert_eq!(plans[1].excluded, RoaringBitmap::from_iter([0, 4]));
        assert_eq!(plans[2].excluded, RoaringBitmap::from_iter([0, 1]));
        // X and Y reach F5 through both transitions; Z covers F4 directly and
        // passes through none.
        assert_eq!(plans[0].path, vec![0, 1]);
        assert_eq!(plans[1].path, vec![0, 1]);
        assert!(plans[2].path.is_empty());

        // The translation identity follows exactly the inputs of the
        // translated output: coverage, the exclusions on the path's
        // destinations, and the path itself.
        let fingerprint = |coverage: &RoaringBitmap, excluded: &RoaringBitmap, path: &[usize]| {
            translation_fingerprint(ledger, coverage, excluded, path)
        };
        let of = |plan: &SegmentPlanParts| fingerprint(&plan.coverage, &plan.excluded, &plan.path);
        let base = of(&plans[0]);
        assert_eq!(base, of(&plans[0]), "deterministic");
        // X and Y share coverage, path and on-path exclusions ({4}), so they
        // share an identity; their own content is told apart by the index
        // UUID namespace layered on top.
        assert_eq!(base, of(&plans[1]));
        assert_ne!(base, of(&plans[2]), "a direct segment differs");
        assert_ne!(
            base,
            fingerprint(&RoaringBitmap::new(), &plans[0].excluded, &plans[0].path),
            "coverage is part of the identity"
        );
        assert_ne!(
            base,
            fingerprint(&plans[0].coverage, &plans[0].excluded, &plans[0].path[..1]),
            "a transition dropped from the path changes the identity"
        );
        // Exclusions outside the path's destinations {2, 3, 4, 5} are inert:
        // sibling churn elsewhere in the group cannot change X's output.
        let mut off_path = plans[0].excluded.clone();
        off_path.insert(99);
        assert_eq!(
            base,
            fingerprint(&plans[0].coverage, &off_path, &plans[0].path),
            "an exclusion off the path leaves the identity unchanged"
        );
        let mut on_path = plans[0].excluded.clone();
        on_path.insert(5);
        assert_ne!(
            base,
            fingerprint(&plans[0].coverage, &on_path, &plans[0].path),
            "an exclusion on a path destination changes the identity"
        );
    }
}
