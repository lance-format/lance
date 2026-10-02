// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Which versions of a dataset must survive, and which may be discarded.
//!
//! Shared by [`cleanup`](super::cleanup) and [`expire`](super::expire). The two differ
//! only in what they delete and in where they get a version's age from; the rules for
//! what is protected are the same and live here so they cannot drift apart.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use chrono::{DateTime, Utc};

use lance_table::io::commit::read_version_from_hint;

use super::Dataset;
use crate::Result;

/// Versions that must survive regardless of age or policy.
///
/// Resolved by reading the dataset's refs, never by reading manifests. Listing failures
/// propagate rather than yielding an empty set: a tag that cannot be read must never
/// license a deletion.
#[derive(Clone, Debug, Default)]
pub struct ProtectedVersions {
    /// The newest version. Deleting it would destroy the dataset.
    pub latest: u64,
    /// Versions a tag on the current branch points at.
    pub tagged: HashSet<u64>,
    /// Versions a branch is rooted at. Removing one orphans that branch's history.
    pub branch_referenced: HashSet<u64>,
    /// The version recorded in the version hint, when the dataset keeps one.
    ///
    /// Everything from here up must stay contiguous. Resolving the latest version starts at
    /// the hint and probes upward one version at a time, stopping at the first that is
    /// missing — so a hole above the hint does not merely lose that version, it hides every
    /// version above it and the dataset reads as though it were rolled back. Deleting the
    /// hinted version itself is survivable, since a miss there falls back to a full listing,
    /// but there is nothing to gain by allowing it.
    pub hint_floor: Option<u64>,
}

impl ProtectedVersions {
    /// Resolve the protected set for `dataset`.
    ///
    /// Only tags on the current branch are collected, matching cleanup: a tag on another
    /// branch protects manifests in that branch's own path, not in this one.
    pub async fn resolve(dataset: &Dataset) -> Result<Self> {
        let tags = dataset.tags().list().await?;
        let current_branch = &dataset.manifest.branch;
        let tagged = Self::tagged_versions(&tags, current_branch.as_ref());

        let branch_identifier = dataset.branch_identifier().await?;
        let branches = dataset.branches().list().await?;
        let branch_referenced = branch_identifier
            .collect_referenced_versions(&branches)
            .into_iter()
            .map(|(_name, version)| version)
            .collect();

        let hint_floor = read_version_from_hint(&dataset.object_store, &dataset.base).await;

        Ok(Self {
            latest: dataset.manifest.version,
            tagged,
            branch_referenced,
            hint_floor,
        })
    }

    /// Tags on `current_branch`, as cleanup selects them.
    fn tagged_versions(
        tags: &HashMap<String, crate::dataset::refs::TagContents>,
        current_branch: Option<&String>,
    ) -> HashSet<u64> {
        tags.values()
            .filter(|tag| match (tag.branch.as_ref(), current_branch) {
                (Some(branch_of_tag), Some(current_branch)) => branch_of_tag == current_branch,
                (None, None) => true,
                _ => false,
            })
            .map(|tag| tag.version)
            .collect()
    }

    /// True when `version` must survive whatever the policy says.
    pub fn contains(&self, version: u64) -> bool {
        version >= self.latest
            || self.tagged.contains(&version)
            || self.branch_referenced.contains(&version)
            || self.hint_floor.is_some_and(|floor| version >= floor)
    }

    /// Protected versions that the policy would otherwise have expired, so a caller can
    /// report them rather than silently keeping them.
    pub fn tagged_but_expired<I>(&self, expired: I) -> HashSet<u64>
    where
        I: IntoIterator<Item = u64>,
    {
        expired
            .into_iter()
            .filter(|v| self.tagged.contains(v))
            .collect()
    }
}

/// Keep the newest version in each bucket of `width`, discarding the rest.
///
/// Buckets are absolute rather than relative to the newest candidate, so the survivors do
/// not shift as the table grows: a version's bucket depends only on its own timestamp.
/// Returns the versions to keep.
///
/// Ties on timestamp resolve to the higher version, so the survivor of a bucket is always
/// the latest state within it.
///
/// Bucketing is exact to the nanosecond. Rounding `width` up to a coarser unit would
/// silently delete more than the caller asked for, and version history does not come back.
/// A zero `width` keeps every candidate; callers reject it up front so that asking for
/// zero is an error rather than quietly doing nothing.
pub fn thin_to_one_per(candidates: &[(u64, DateTime<Utc>)], width: Duration) -> HashSet<u64> {
    if width.is_zero() {
        return candidates.iter().map(|(version, _)| *version).collect();
    }
    // i128 throughout: nanoseconds since the epoch overflow i64 outside ~1677..2262, and a
    // width may legitimately be years.
    let width_nanos = width.as_nanos() as i128;
    let mut newest_in_bucket: HashMap<i128, (u64, DateTime<Utc>)> = HashMap::new();
    for (version, timestamp) in candidates {
        let nanos = (timestamp.timestamp() as i128) * 1_000_000_000
            + timestamp.timestamp_subsec_nanos() as i128;
        let bucket = nanos.div_euclid(width_nanos);
        newest_in_bucket
            .entry(bucket)
            .and_modify(|held| {
                if (*timestamp, *version) > (held.1, held.0) {
                    *held = (*version, *timestamp);
                }
            })
            .or_insert((*version, *timestamp));
    }
    newest_in_bucket
        .into_values()
        .map(|(version, _)| version)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(secs: i64) -> DateTime<Utc> {
        DateTime::from_timestamp(secs, 0).unwrap()
    }

    #[test]
    fn protected_versions_covers_latest_tags_and_branches() {
        let protected = ProtectedVersions {
            latest: 100,
            tagged: HashSet::from([7]),
            branch_referenced: HashSet::from([42]),
            hint_floor: None,
        };

        assert!(protected.contains(100), "the latest version is protected");
        assert!(
            protected.contains(101),
            "anything at or past latest is protected"
        );
        assert!(protected.contains(7), "a tagged version is protected");
        assert!(
            protected.contains(42),
            "a branch-referenced version is protected"
        );
        assert!(!protected.contains(41), "an ordinary old version is not");
    }

    #[test]
    fn protected_versions_treats_the_hint_as_a_floor() {
        // Resolving the latest version probes upward from the hint and stops at the first
        // gap, so a hole above the hint hides every version above it. Nothing at or above
        // the hint may be removed.
        let protected = ProtectedVersions {
            latest: 100,
            tagged: HashSet::new(),
            branch_referenced: HashSet::new(),
            hint_floor: Some(50),
        };

        assert!(
            protected.contains(50),
            "the hinted version itself is protected"
        );
        assert!(protected.contains(51), "and everything above it");
        assert!(
            protected.contains(99),
            "including versions well below latest"
        );
        assert!(
            !protected.contains(49),
            "but not the versions below the hint"
        );

        // With no hint there is no probe to break, so the floor does not apply.
        let no_hint = ProtectedVersions {
            hint_floor: None,
            ..protected
        };
        assert!(!no_hint.contains(50));
        assert!(!no_hint.contains(99));
    }

    #[test]
    fn thinning_keeps_exactly_one_per_bucket() {
        // Three versions in the first hour, two in the second.
        let hour = Duration::from_secs(3600);
        let candidates = vec![
            (1, ts(0)),
            (2, ts(600)),
            (3, ts(3599)),
            (4, ts(3600)),
            (5, ts(7000)),
        ];

        let kept = thin_to_one_per(&candidates, hour);
        assert_eq!(kept.len(), 2, "one survivor per hour bucket");
        assert!(
            kept.contains(&3),
            "the newest of hour 0 survives, not 1 or 2"
        );
        assert!(kept.contains(&5), "the newest of hour 1 survives, not 4");
    }

    #[test]
    fn thinning_survivor_is_the_newest_not_an_arbitrary_one() {
        // Deliberately out of order: a naive implementation that keeps the first seen
        // would keep version 1 here.
        let candidates = vec![(1, ts(10)), (9, ts(20)), (5, ts(15))];
        let kept = thin_to_one_per(&candidates, Duration::from_secs(3600));
        assert_eq!(kept, HashSet::from([9]));
    }

    #[test]
    fn thinning_breaks_timestamp_ties_toward_the_higher_version() {
        // Equal timestamps happen when commits land inside one clock tick; the later
        // version is the later state.
        let candidates = vec![(4, ts(10)), (6, ts(10)), (5, ts(10))];
        let kept = thin_to_one_per(&candidates, Duration::from_secs(3600));
        assert_eq!(kept, HashSet::from([6]));
    }

    #[test]
    fn thinning_a_zero_width_keeps_every_version() {
        // Not a divide-by-zero, and not "collapse into one bucket and delete the rest".
        // Candidates deliberately share a second, so this cannot pass by accident the way
        // one-second-apart candidates would.
        let candidates = vec![
            (1, DateTime::from_timestamp(0, 0).unwrap()),
            (2, DateTime::from_timestamp(0, 1).unwrap()),
            (3, DateTime::from_timestamp(0, 2).unwrap()),
        ];
        let kept = thin_to_one_per(&candidates, Duration::from_secs(0));
        assert_eq!(kept.len(), 3, "a zero width must not collapse the history");
    }

    #[test]
    fn thinning_is_exact_at_nanosecond_widths() {
        let candidates = vec![
            (1, DateTime::from_timestamp(0, 0).unwrap()),
            (2, DateTime::from_timestamp(0, 1).unwrap()),
        ];
        // One nanosecond apart: distinct buckets at a 1ns width...
        assert_eq!(
            thin_to_one_per(&candidates, Duration::from_nanos(1)),
            HashSet::from([1, 2])
        );
        // ...and the same bucket at 2ns.
        assert_eq!(
            thin_to_one_per(&candidates, Duration::from_nanos(2)),
            HashSet::from([2])
        );
    }

    #[test]
    fn thinning_honors_subsecond_widths() {
        let candidates = vec![
            (1, DateTime::from_timestamp(0, 100_000_000).unwrap()),
            (2, DateTime::from_timestamp(0, 600_000_000).unwrap()),
        ];
        let kept = thin_to_one_per(&candidates, Duration::from_millis(500));
        assert_eq!(kept, HashSet::from([1, 2]));
    }

    #[test]
    fn thinning_nothing_keeps_nothing() {
        let kept = thin_to_one_per(&[], Duration::from_secs(3600));
        assert!(kept.is_empty());
    }
}
