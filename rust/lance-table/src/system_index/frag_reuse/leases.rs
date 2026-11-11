// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What listed retention leases require of a fragment reuse history.
//!
//! A lease holder computes results against the fragments of its leased
//! version. Until it publishes, every transition that connects those fragments
//! to the current layout must stay recorded: compaction records them and
//! trimming keeps them. See the "Required transitions" section of the
//! [Retention Leases specification](https://lance.org/format/table/retention_leases/).
//!
//! ```
//! use lance_table::system_index::frag_reuse::leases::{
//!     LeaseRequirement, LeasedFragments, RecordLineage,
//! };
//! use roaring::RoaringBitmap;
//!
//! let records = [
//!     RecordLineage::transition([4], [6]),
//!     RecordLineage::transition([6], [7]),
//!     RecordLineage::transition([8], [9]),
//! ];
//! let leased = LeasedFragments::Known(RoaringBitmap::from_iter([3, 4, 5]));
//! let required = LeaseRequirement::compute(&leased, &records);
//! assert!(required.requires_record(0));
//! assert!(required.requires_record(1));
//! assert!(!required.requires_record(2));
//! // Compacting fragment 7 must record a transition; fragment 9 need not.
//! assert!(required.requires_sources([7]));
//! assert!(!required.requires_sources([9]));
//! ```

use std::collections::HashMap;

use roaring::RoaringBitmap;

use super::FragReuseVersion;
use crate::format::pb::fragment_reuse_index_details as pb;

/// The fragment lineage of one FRI record, the unit trimming keeps or
/// removes: a tagged transition, or a legacy reuse version whose groups are
/// kept or removed together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordLineage {
    /// `(sources, destinations)` fragment ids of each group.
    groups: Vec<(Vec<u64>, Vec<u64>)>,
}

impl RecordLineage {
    pub fn transition(
        sources: impl IntoIterator<Item = u64>,
        destinations: impl IntoIterator<Item = u64>,
    ) -> Self {
        Self {
            groups: vec![(
                sources.into_iter().collect(),
                destinations.into_iter().collect(),
            )],
        }
    }

    /// A legacy reuse version: selecting any group selects every group.
    pub fn legacy_version(groups: impl IntoIterator<Item = (Vec<u64>, Vec<u64>)>) -> Self {
        Self {
            groups: groups.into_iter().collect(),
        }
    }

    /// Every group's source fragment ids.
    pub fn sources(&self) -> impl Iterator<Item = u64> + '_ {
        self.groups
            .iter()
            .flat_map(|(sources, _)| sources.iter().copied())
    }

    fn destinations(&self) -> impl Iterator<Item = u64> + '_ {
        self.groups
            .iter()
            .flat_map(|(_, destinations)| destinations.iter().copied())
    }
}

impl From<&pb::Transition> for RecordLineage {
    fn from(transition: &pb::Transition) -> Self {
        Self::transition(
            transition.sources.iter().map(|digest| digest.id),
            transition.destinations.iter().map(|digest| digest.id),
        )
    }
}

impl From<&pb::Version> for RecordLineage {
    fn from(version: &pb::Version) -> Self {
        Self::legacy_version(version.groups.iter().map(|group| {
            (
                group.old_fragments.iter().map(|digest| digest.id).collect(),
                group.new_fragments.iter().map(|digest| digest.id).collect(),
            )
        }))
    }
}

impl From<&FragReuseVersion> for RecordLineage {
    fn from(version: &FragReuseVersion) -> Self {
        Self::legacy_version(version.groups.iter().map(|group| {
            (
                group.old_frags.iter().map(|digest| digest.id).collect(),
                group.new_frags.iter().map(|digest| digest.id).collect(),
            )
        }))
    }
}

/// The fragment ids of the versions the listed leases retain.
#[derive(Debug, Clone, PartialEq)]
pub enum LeasedFragments {
    /// The union of every leased version's fragment ids; empty when no lease
    /// is listed.
    Known(RoaringBitmap),
    /// Some leased version's manifest could not be read, so its fragments are
    /// unknown. Tagged transitions carry no version, so the ones recorded after
    /// that version cannot be told apart from older ones: every record is
    /// required, and every rewrite must record its transitions.
    Unknown,
}

/// The records and fragments the listed leases require under the FRI index
/// version 1 rule: select each record whose sources intersect the leased
/// fragments, add its destinations, and repeat until nothing more is
/// selected. A selected legacy reuse version contributes every group's
/// destinations.
#[derive(Debug, Clone, PartialEq)]
pub struct LeaseRequirement {
    /// Indexed by position in the records passed to [`Self::compute`].
    required_records: Vec<bool>,
    /// The leased fragments plus every destination reached from them. `None`
    /// when every fragment is required.
    required_fragments: Option<RoaringBitmap>,
}

impl LeaseRequirement {
    /// Evaluate `leased` against a history's records, in any order. Callers
    /// must query [`Self::requires_record`] with positions in that same slice.
    pub fn compute(leased: &LeasedFragments, records: &[RecordLineage]) -> Self {
        let LeasedFragments::Known(leased) = leased else {
            return Self {
                required_records: vec![true; records.len()],
                required_fragments: None,
            };
        };
        let mut consumers: HashMap<u64, Vec<usize>> = HashMap::new();
        for (position, record) in records.iter().enumerate() {
            for source in record.sources() {
                consumers.entry(source).or_default().push(position);
            }
        }
        let mut fragments = leased.clone();
        let mut selected = vec![false; records.len()];
        let mut pending: Vec<usize> = records
            .iter()
            .enumerate()
            .filter(|(_, record)| record.sources().any(|source| contains(&fragments, source)))
            .map(|(position, _)| position)
            .collect();
        while let Some(position) = pending.pop() {
            if std::mem::replace(&mut selected[position], true) {
                continue;
            }
            for destination in records[position].destinations() {
                let Ok(id) = u32::try_from(destination) else {
                    continue;
                };
                if fragments.insert(id)
                    && let Some(next) = consumers.get(&destination)
                {
                    pending.extend(next.iter().filter(|next| !selected[**next]));
                }
            }
        }
        Self {
            required_records: selected,
            required_fragments: Some(fragments),
        }
    }

    /// Whether the record at `position` in the evaluated history is required.
    pub fn requires_record(&self, position: usize) -> bool {
        self.required_records
            .get(position)
            .copied()
            .unwrap_or(false)
    }

    /// Whether a rewrite consuming `sources` must record a transition for
    /// them: some source is a leased fragment or reached from one.
    pub fn requires_sources(&self, sources: impl IntoIterator<Item = u64>) -> bool {
        match &self.required_fragments {
            None => true,
            Some(fragments) => sources
                .into_iter()
                .any(|source| contains(fragments, source)),
        }
    }
}

/// The legacy-only (FRI index version 0) rule: a reuse version is required
/// when stamped at or after the oldest leased version. A version is stamped
/// with the version its rewrite was based on, so one stamped earlier rewrote
/// fragments no leased version contains.
pub fn requires_legacy_version(oldest_leased_version: Option<u64>, stamp: u64) -> bool {
    oldest_leased_version.is_some_and(|oldest| stamp >= oldest)
}

// Fragment ids beyond the row-address range never appear in a leased version's
// address space, so they are never required.
fn contains(fragments: &RoaringBitmap, id: u64) -> bool {
    u32::try_from(id).is_ok_and(|id| fragments.contains(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    fn known(ids: &[u32]) -> LeasedFragments {
        LeasedFragments::Known(ids.iter().copied().collect())
    }

    fn required_positions(leased: &LeasedFragments, records: &[RecordLineage]) -> Vec<usize> {
        let requirement = LeaseRequirement::compute(leased, records);
        (0..records.len())
            .filter(|position| requirement.requires_record(*position))
            .collect()
    }

    #[test]
    fn spec_example_keeps_chain_from_leased_fragments() {
        // The Trimming example of the FRI specification: leased version 12
        // holds {3, 4, 5}.
        let records = [
            RecordLineage::transition([1, 2], [5]),
            RecordLineage::transition([4], [6]),
            RecordLineage::transition([6], [7]),
            RecordLineage::transition([8], [9]),
        ];
        // {1, 2} -> {5} is required by an index, not by the lease: fragment 5
        // is the transition's output, which the lease already holds.
        assert_eq!(required_positions(&known(&[3, 4, 5]), &records), vec![1, 2]);
    }

    #[rstest]
    // Serialized order does not matter: the consumer precedes its producer.
    #[case::reverse_order(
        vec![RecordLineage::transition([6], [7]), RecordLineage::transition([4], [6])],
        &[4],
        vec![0, 1]
    )]
    #[case::no_leases(vec![RecordLineage::transition([4], [6])], &[], vec![])]
    #[case::fan_in(
        vec![
            RecordLineage::transition([1], [3]),
            RecordLineage::transition([2, 3], [4]),
        ],
        &[1],
        vec![0, 1]
    )]
    #[case::legacy_version_selects_every_group(
        vec![
            RecordLineage::legacy_version([(vec![1], vec![10]), (vec![2], vec![11])]),
            RecordLineage::transition([11], [12]),
            RecordLineage::transition([10], [13]),
        ],
        &[1],
        vec![0, 1, 2]
    )]
    #[case::out_of_range_ids_are_never_required(
        vec![RecordLineage::transition([u64::from(u32::MAX) + 1], [7])],
        &[],
        vec![]
    )]
    fn closure_cases(
        #[case] records: Vec<RecordLineage>,
        #[case] leased: &[u32],
        #[case] expected: Vec<usize>,
    ) {
        assert_eq!(required_positions(&known(leased), &records), expected);
    }

    #[test]
    fn unknown_leased_fragments_require_everything() {
        let records = [RecordLineage::transition([8], [9])];
        let requirement = LeaseRequirement::compute(&LeasedFragments::Unknown, &records);
        assert!(requirement.requires_record(0));
        assert!(requirement.requires_sources([100]));
    }

    #[test]
    fn rewrite_sources_reached_from_lease_are_required() {
        let records = [RecordLineage::transition([4], [6])];
        let requirement = LeaseRequirement::compute(&known(&[3, 4]), &records);
        assert!(requirement.requires_sources([3]));
        assert!(requirement.requires_sources([6, 20]));
        assert!(!requirement.requires_sources([20, 21]));
    }

    #[rstest]
    #[case::no_leases(None, 5, false)]
    #[case::stamped_before(Some(10), 9, false)]
    #[case::stamped_at(Some(10), 10, true)]
    #[case::stamped_after(Some(10), 11, true)]
    fn legacy_rule(#[case] oldest: Option<u64>, #[case] stamp: u64, #[case] expected: bool) {
        assert_eq!(requires_legacy_version(oldest, stamp), expected);
    }

    #[test]
    fn lineage_from_pb_records() {
        let digest = |id| pb::FragmentDigest {
            id,
            ..Default::default()
        };
        let version = pb::Version {
            dataset_version: 3,
            groups: vec![pb::Group {
                old_fragments: vec![digest(1)],
                new_fragments: vec![digest(2)],
                ..Default::default()
            }],
        };
        assert_eq!(
            RecordLineage::from(&version),
            RecordLineage::legacy_version([(vec![1], vec![2])])
        );
        let transition = pb::Transition {
            sources: vec![digest(2)],
            destinations: vec![digest(3)],
            mapping: None,
        };
        assert_eq!(
            RecordLineage::from(&transition),
            RecordLineage::transition([2], [3])
        );
    }
}
