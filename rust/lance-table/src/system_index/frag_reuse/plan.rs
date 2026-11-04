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

use roaring::RoaringBitmap;
use uuid::Uuid;

use super::history::FragReuseHistory;
use super::ledger::FragReuseLedger;

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
    /// The segment is caught up with the history and its fragments are all
    /// live: its addresses are used as stored.
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
/// Opaque to readers: they hand it back to the module that translates.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationPlan {
    coverage: RoaringBitmap,
    excluded_fragments: RoaringBitmap,
    fingerprint: [u8; 32],
}

impl TranslationPlan {
    /// A plan under `history` translating every address to nothing: the
    /// segment answers for no fragment. Maintenance reads a segment the
    /// reader derives no coverage for through it.
    pub fn empty(history: &FragReuseHistory) -> Self {
        Self::nothing(history.ledger())
    }

    pub(super) fn nothing(ledger: &FragReuseLedger) -> Self {
        let empty = RoaringBitmap::new();
        Self::new(ledger, empty.clone(), empty, &[])
    }

    pub(super) fn new(
        ledger: &FragReuseLedger,
        coverage: RoaringBitmap,
        excluded_fragments: RoaringBitmap,
        path: &[usize],
    ) -> Self {
        let fingerprint = translation_fingerprint(ledger, &coverage, &excluded_fragments, path);
        Self {
            coverage,
            excluded_fragments,
            fingerprint,
        }
    }

    /// The live fragments the segment answers for. Transitional: the lance
    /// crate builds its translator from this until translation is bound
    /// through the plan.
    pub fn coverage(&self) -> &RoaringBitmap {
        &self.coverage
    }

    /// Fragments other segments of the same index cover directly: a
    /// translated path entering one of them is dropped. Transitional, as
    /// [`Self::coverage`].
    pub fn excluded_fragments(&self) -> &RoaringBitmap {
        &self.excluded_fragments
    }

    /// The translation identity: a hash of everything that decides the
    /// translated output, so cached objects that embed translated addresses
    /// can be keyed by it and go cold only when that state changes.
    /// Transitional, as [`Self::coverage`].
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
