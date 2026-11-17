// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Decides which versions cleanup expires, from version numbers and commit
//! times alone. Listing and reading manifests stays in the cleanup task.
//!
//! A version can be removed in two ways:
//!
//! * **Expiration.** It is older than a cutoff: the legacy `before_timestamp`,
//!   or `now - older_than` of a `RemoveAll` rule. With neither, a legacy policy
//!   expires every version its version filters match.
//! * **Thinning.** A `KeepOne { per_interval, older_than }` rule keeps only one
//!   version per `per_interval` of history older than `older_than`. With
//!   `per_interval = 1h` and `older_than = 2 days`, history older than two days
//!   is cut down to one version per hour.
//!
//! Thinning splits history into windows of `per_interval`, aligned to the Unix
//! epoch. One window of one rule is a [`Bucket`]. The newest version in a bucket
//! is its *survivor* and is kept; every other version in the bucket is removed.
//!
//! ```text
//! KeepOne { per_interval: 1h, older_than: 2 days }
//!
//!          thinned: older than (anchor - 2 days)       | kept: newer
//!   10:00       11:00       12:00       13:00          |
//!     | v3 v4 v5 | v6       |          | v7 v8 |  ...  | v11 v12 v13
//!             ^     ^                        ^         |
//!   survivors v5,   v6,                      v8        |
//! ```
//!
//! A survivor is only known once every version in its bucket has been seen, so
//! thinning takes two steps: [`VersionExpiry::classify`] puts a version in its
//! bucket, and [`ThinningSurvivors`] picks the newest version of each bucket.

use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::time::Duration;

use chrono::{DateTime, TimeDelta, Utc};
use lance_core::{Error, Result};

use super::CleanupPolicy;
use super::retention::CleanupRule;

/// The commit `KeepOne` ages are measured from: the hinted commit, or the
/// dataset's own commit when no hint exists.
#[derive(Clone, Copy, Debug)]
pub(super) struct ThinningAnchor {
    pub timestamp: DateTime<Utc>,
    /// This version and all later ones are never thinned, even with disordered clocks.
    pub version: u64,
}

#[derive(Clone, Copy, Debug)]
enum Expiration {
    Never,
    /// Every version matching the version filters.
    All,
    Before(DateTime<Utc>),
}

/// A `KeepOne` rule resolved against the anchor.
#[derive(Clone, Copy, Debug)]
struct ThinningRule {
    per_interval: Duration,
    /// Versions committed before this are thinned by this rule, unless a later
    /// (older) rule applies.
    cutoff: DateTime<Utc>,
}

/// One `per_interval` window of one `KeepOne` rule.
///
/// Each rule thins on its own, so two rules' windows never share a bucket,
/// even when their window numbers are equal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Bucket {
    /// Position of the rule among the policy's `KeepOne` rules.
    rule: usize,
    /// Window number since the Unix epoch: `commit time / per_interval`. Signed
    /// so that a commit time before 1970, from a skewed clock, still gets a window.
    window: i128,
}

/// Whether a version is kept, given its commit time. See [`VersionExpiry::classify`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    Retained,
    /// Old enough to expire, but kept for its tag.
    TaggedOld,
    Expired,
    /// In a thinning bucket: removed unless it is the bucket's survivor, which
    /// [`ThinningSurvivors`] decides once every version has been seen.
    Thinned(ThinningCandidate),
}

impl Verdict {
    pub(super) fn is_retained(self) -> bool {
        matches!(self, Self::Retained | Self::TaggedOld)
    }
}

/// A version in a thinning bucket, waiting to learn whether it is the survivor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ThinningCandidate {
    bucket: Bucket,
    timestamp: DateTime<Utc>,
    version: u64,
}

/// The survivor of each thinning bucket: its newest version, with the higher
/// version breaking commit-time ties.
///
/// Candidates can be offered in any order. `Handle` is whatever the caller needs
/// to act on a survivor afterwards. Cleanup passes the manifest location, so it
/// can re-read each survivor's manifest and keep its files.
#[derive(Debug)]
pub(super) struct ThinningSurvivors<Handle> {
    newest: HashMap<Bucket, (ThinningCandidate, Handle)>,
}

impl<Handle> Default for ThinningSurvivors<Handle> {
    fn default() -> Self {
        Self {
            newest: HashMap::new(),
        }
    }
}

impl<Handle> ThinningSurvivors<Handle> {
    pub(super) fn offer(&mut self, candidate: ThinningCandidate, handle: Handle) {
        match self.newest.entry(candidate.bucket) {
            Entry::Occupied(mut occupied) => {
                let (newest, _) = occupied.get();
                if (candidate.timestamp, candidate.version) > (newest.timestamp, newest.version) {
                    occupied.insert((candidate, handle));
                }
            }
            Entry::Vacant(vacant) => {
                vacant.insert((candidate, handle));
            }
        }
    }

    pub(super) fn into_survivors(self) -> impl Iterator<Item = Handle> {
        self.newest.into_values().map(|(_, handle)| handle)
    }
}

/// Which versions a cleanup run expires.
///
/// Combines the legacy [`CleanupPolicy`] filters, the retention rules resolved
/// against the current time and thinning anchor, and the versions protected by
/// tags, branches, and the read version.
///
/// Tags and branch-held versions are added after construction with
/// [`Self::with_tagged`] and [`Self::with_branch_held`], because cleanup only
/// learns them after it must already know which versions are old: [`Self::is_old`]
/// ignores them so it can pick the branches to clean first.
#[derive(Clone, Debug)]
pub(super) struct VersionExpiry {
    legacy_policy: CleanupPolicy,
    expiration: Expiration,
    /// Ordered by increasing age, like the `KeepOne` rules they come from.
    thinning: Vec<ThinningRule>,
    /// Versions at or above this are never thinned: the lower of the anchor and
    /// read versions.
    first_unthinned_version: u64,
    read_version: u64,
    tagged: HashSet<u64>,
    branch_held: HashSet<u64>,
}

impl VersionExpiry {
    /// `RemoveAll` ages are measured from `now`. `anchor` is required when
    /// `rules` contains `KeepOne`.
    pub(super) fn new(
        policy: &CleanupPolicy,
        rules: Option<&[CleanupRule]>,
        now: DateTime<Utc>,
        anchor: Option<ThinningAnchor>,
        read_version: u64,
    ) -> Result<Self> {
        let mut expire_before = policy.before_timestamp;
        let mut thinning = Vec::new();
        for rule in rules.unwrap_or_default() {
            match *rule {
                CleanupRule::RemoveAll { older_than } => {
                    let cutoff = now
                        .checked_sub_signed(to_time_delta(older_than, "remove_all")?)
                        .ok_or_else(|| {
                            Error::invalid_input(format!(
                                "remove_all.older_than={older_than:?} underflows current time"
                            ))
                        })?;
                    expire_before = Some(expire_before.map_or(cutoff, |ts| ts.max(cutoff)));
                }
                CleanupRule::KeepOne {
                    per_interval,
                    older_than,
                } => {
                    let anchor = anchor.ok_or_else(|| {
                        Error::internal("keep_one retention requires a thinning anchor")
                    })?;
                    let cutoff = anchor
                        .timestamp
                        .checked_sub_signed(to_time_delta(older_than, "keep_one")?)
                        .ok_or_else(|| {
                            Error::invalid_input(format!(
                                "keep_one.older_than={older_than:?} underflows anchor timestamp {}",
                                anchor.timestamp
                            ))
                        })?;
                    thinning.push(ThinningRule {
                        per_interval,
                        cutoff,
                    });
                }
            }
        }
        // Without a time limit, a retention policy only thins, while a legacy
        // policy expires every version its filters match.
        let expiration = match (expire_before, rules) {
            (Some(timestamp), _) => Expiration::Before(timestamp),
            (None, None) => Expiration::All,
            (None, Some(_)) => Expiration::Never,
        };
        Ok(Self {
            legacy_policy: policy.clone(),
            expiration,
            thinning,
            first_unthinned_version: anchor
                .map_or(read_version, |anchor| anchor.version.min(read_version)),
            read_version,
            tagged: HashSet::new(),
            branch_held: HashSet::new(),
        })
    }

    pub(super) fn with_tagged(mut self, tagged: HashSet<u64>) -> Self {
        self.tagged = tagged;
        self
    }

    /// Versions whose files descendant branches still use.
    pub(super) fn with_branch_held(mut self, branch_held: HashSet<u64>) -> Self {
        self.branch_held = branch_held;
        self
    }

    fn is_hard_expired(&self, timestamp: DateTime<Utc>) -> bool {
        match self.expiration {
            Expiration::Never => false,
            Expiration::All => true,
            Expiration::Before(cutoff) => timestamp < cutoff,
        }
    }

    /// The index of the oldest thinning rule whose age the version has reached.
    fn thinning_rule(&self, version: u64, timestamp: DateTime<Utc>) -> Option<usize> {
        if version >= self.first_unthinned_version {
            return None;
        }
        self.thinning
            .iter()
            .rposition(|rule| timestamp < rule.cutoff)
    }

    /// True when the policy would expire or thin this version, ignoring every
    /// protection (tags, branches, the read version).
    pub(super) fn is_old(&self, version: u64, timestamp: DateTime<Utc>) -> bool {
        self.legacy_policy.matches_version_filters(version)
            && (self.is_hard_expired(timestamp) || self.thinning_rule(version, timestamp).is_some())
    }

    pub(super) fn classify(&self, version: u64, timestamp: DateTime<Utc>) -> Verdict {
        // Versions at or above the read version are current, in progress, or
        // committed since this cleanup started. Detached versions set the high
        // bit, so they are above any attached read version and never expire.
        if version >= self.read_version || !self.legacy_policy.matches_version_filters(version) {
            return Verdict::Retained;
        }
        let hard_expired = self.is_hard_expired(timestamp);
        let rule = self.thinning_rule(version, timestamp);
        if !hard_expired && rule.is_none() {
            Verdict::Retained
        } else if self.tagged.contains(&version) {
            Verdict::TaggedOld
        } else if self.branch_held.contains(&version) {
            Verdict::Retained
        } else if let Some(rule) = rule.filter(|_| !hard_expired) {
            let per_interval = self.thinning[rule].per_interval.as_nanos() as i128;
            Verdict::Thinned(ThinningCandidate {
                bucket: Bucket {
                    rule,
                    // Rounds toward negative infinity, so windows before the
                    // epoch are as wide as windows after it.
                    window: timestamp_nanos(timestamp).div_euclid(per_interval),
                },
                timestamp,
                version,
            })
        } else {
            Verdict::Expired
        }
    }

    /// True only when the version alone proves it expired, so its manifest need
    /// not be read.
    ///
    /// Timestamp policies require the logical commit time: storage clocks and
    /// timestamp precision do not make object write time an upper bound.
    pub(super) fn is_expired_by_version(&self, version: u64) -> bool {
        matches!(self.expiration, Expiration::All)
            && (self.legacy_policy.before_version.is_some()
                || self.legacy_policy.versions.is_some())
            && version < self.read_version
            && !self.tagged.contains(&version)
            && !self.branch_held.contains(&version)
            && self.legacy_policy.matches_version_filters(version)
    }
}

fn to_time_delta(age: Duration, rule: &str) -> Result<TimeDelta> {
    TimeDelta::from_std(age)
        .map_err(|_| Error::invalid_input(format!("{rule}.older_than is out of range: {age:?}")))
}

fn timestamp_nanos(timestamp: DateTime<Utc>) -> i128 {
    i128::from(timestamp.timestamp()) * 1_000_000_000
        + i128::from(timestamp.timestamp_subsec_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    use crate::dataset::cleanup::retention::MIN_THINNING_AGE;

    const HOUR: u64 = 3600;
    const DAY: u64 = 24 * HOUR;

    fn at_hour(hour: u64) -> DateTime<Utc> {
        DateTime::from_timestamp(i64::try_from(hour * HOUR).unwrap(), 0).unwrap()
    }

    fn keep_one(per_interval: u64, older_than: u64) -> CleanupRule {
        CleanupRule::KeepOne {
            per_interval: Duration::from_secs(per_interval),
            older_than: Duration::from_secs(older_than),
        }
    }

    fn remove_all(older_than: u64) -> CleanupRule {
        CleanupRule::RemoveAll {
            older_than: Duration::from_secs(older_than),
        }
    }

    /// Commits versions 1..=n at the given hours, read from the latest version.
    fn history(hours: &[u64]) -> Vec<(u64, DateTime<Utc>)> {
        (1..).zip(hours.iter().map(|hour| at_hour(*hour))).collect()
    }

    fn latest(history: &[(u64, DateTime<Utc>)]) -> ThinningAnchor {
        let (version, timestamp) = *history.last().unwrap();
        ThinningAnchor { timestamp, version }
    }

    /// The versions a cleanup run keeps. Manifests are listed in no particular
    /// order, so this checks that oldest-first and newest-first agree.
    fn kept(expiry: &VersionExpiry, history: &[(u64, DateTime<Utc>)]) -> Vec<u64> {
        let oldest_first = kept_in_order(expiry, history.iter());
        let newest_first = kept_in_order(expiry, history.iter().rev());
        assert_eq!(
            oldest_first, newest_first,
            "survivors depend on listing order"
        );
        oldest_first
    }

    fn kept_in_order<'a>(
        expiry: &VersionExpiry,
        listing: impl Iterator<Item = &'a (u64, DateTime<Utc>)>,
    ) -> Vec<u64> {
        let mut kept = Vec::new();
        let mut survivors = ThinningSurvivors::default();
        for (version, timestamp) in listing {
            match expiry.classify(*version, *timestamp) {
                Verdict::Retained | Verdict::TaggedOld => kept.push(*version),
                Verdict::Thinned(candidate) => survivors.offer(candidate, *version),
                Verdict::Expired => {}
            }
        }
        kept.extend(survivors.into_survivors());
        kept.sort_unstable();
        kept
    }

    #[test]
    fn retention_ladder() {
        let history = history(&[
            0, 1, 240, 264, 576, 577, 769, 770, 792, 792, 793, 864, 864, 865, 911, 912, 912, 913,
            959, 960,
        ]);
        let rules = [
            keep_one(HOUR, 2 * DAY),
            keep_one(DAY, 7 * DAY),
            remove_all(30 * DAY),
        ];
        let expiry = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&rules),
            at_hour(960),
            Some(latest(&history)),
            20,
        )
        .unwrap();
        assert_eq!(
            kept(&expiry, &history),
            vec![3, 4, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20]
        );

        // Tags and version filters keep versions the rules would remove.
        let policy = CleanupPolicy {
            versions: Some(
                (1..=20)
                    .filter(|version| ![2, 5].contains(version))
                    .collect(),
            ),
            ..Default::default()
        };
        let expiry = VersionExpiry::new(
            &policy,
            Some(&rules),
            at_hour(960),
            Some(latest(&history)),
            20,
        )
        .unwrap()
        .with_tagged(HashSet::from([1]));
        assert_eq!(
            kept(&expiry, &history),
            vec![1, 2, 3, 4, 5, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20]
        );
        assert_eq!(expiry.classify(1, at_hour(0)), Verdict::TaggedOld);
    }

    #[test]
    fn retention_bands_have_independent_buckets() {
        // Version 1's quarter-hour bucket and version 2's hourly bucket have the
        // same index (800), but they are in different bands.
        let history = history(&[200, 800, 801, 960]);
        let expiry = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&[keep_one(HOUR, 2 * DAY), keep_one(15 * 60, 7 * DAY)]),
            at_hour(960),
            Some(latest(&history)),
            4,
        )
        .unwrap();
        assert_eq!(kept(&expiry, &history), vec![1, 2, 3, 4]);
    }

    #[rstest]
    #[case::daily(24, vec![4, 7, 8, 9, 10])]
    #[case::twice_daily(12, vec![2, 4, 7, 8, 9, 10])]
    fn keeps_newest_per_period(#[case] period_hours: u64, #[case] expected: Vec<u64>) {
        // Out-of-order clocks: version 3 commits before version 2.
        let history = history(&[0, 2, 1, 12, 24, 26, 26, 48, 49, 96]);
        let expiry = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&[keep_one(period_hours * HOUR, 2 * DAY)]),
            at_hour(96),
            Some(latest(&history)),
            10,
        )
        .unwrap();
        assert_eq!(kept(&expiry, &history), expected);
    }

    #[rstest]
    #[case::current_hint(None, vec![5, 7, 9, 10, 11, 12, 13, 14])]
    #[case::stale_hint(Some(9), vec![5, 7, 8, 9, 10, 11, 12, 13, 14])]
    fn expiration_and_thinning_are_separate(
        #[case] hinted_version: Option<u64>,
        #[case] mut expected: Vec<u64>,
        #[values(false, true)] filtered: bool,
    ) {
        let history = history(&[0, 1, 24, 25, 72, 96, 97, 168, 169, 192, 193, 216, 217, 240]);
        let anchor = hinted_version.map_or(latest(&history), |version| ThinningAnchor {
            version,
            timestamp: history[version as usize - 1].1,
        });
        let mut policy = CleanupPolicy {
            before_timestamp: Some(at_hour(240 - 7 * 24)),
            ..Default::default()
        };
        if filtered {
            policy.versions = Some((2..14).filter(|version| *version != 6).collect());
            expected.extend([1, 6]);
            expected.sort_unstable();
        }
        let expiry = VersionExpiry::new(
            &policy,
            Some(&[keep_one(DAY, 2 * DAY)]),
            at_hour(240),
            Some(anchor),
            14,
        )
        .unwrap();
        assert_eq!(kept(&expiry, &history), expected);
    }

    #[rstest]
    #[case::default(MIN_THINNING_AGE.as_secs(), vec![2, 3, 4, 5, 6, 7, 8, 9])]
    #[case::three_days(3 * DAY, vec![1, 2, 3, 4, 5, 6, 7, 8, 9])]
    fn thinning_anchors_to_hint(#[case] older_than: u64, #[case] expected: Vec<u64>) {
        // Version 6 has a skewed clock; the hint at version 7 protects it and
        // every later version from thinning.
        let history = history(&[0, 1, 24, 25, 48, 1000, 72, 1, 120]);
        let anchor = ThinningAnchor {
            version: 7,
            timestamp: at_hour(72),
        };
        let expiry = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&[keep_one(DAY, older_than)]),
            at_hour(30 * 24),
            Some(anchor),
            9,
        )
        .unwrap();
        assert_eq!(kept(&expiry, &history), expected);
    }

    #[test]
    fn remove_all_uses_wall_clock_and_expires_samples() {
        let history = history(&[0, 1]);
        let rules = [keep_one(DAY, 2 * DAY), remove_all(2 * DAY)];
        let anchor = ThinningAnchor {
            version: 1,
            timestamp: at_hour(0),
        };
        let expiry = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&rules),
            at_hour(24),
            Some(anchor),
            2,
        )
        .unwrap();
        assert_eq!(kept(&expiry, &history), vec![1, 2]);
        let expiry = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&rules),
            at_hour(72),
            Some(anchor),
            2,
        )
        .unwrap();
        assert_eq!(kept(&expiry, &history), vec![2]);
    }

    #[test]
    fn stricter_of_legacy_timestamp_and_remove_all_wins() {
        let history = history(&[0, 24, 48, 72]);
        let policy = CleanupPolicy {
            before_timestamp: Some(at_hour(30)),
            ..Default::default()
        };
        let expiry =
            VersionExpiry::new(&policy, Some(&[remove_all(DAY)]), at_hour(72), None, 4).unwrap();
        assert_eq!(kept(&expiry, &history), vec![3, 4]);
    }

    #[test]
    fn thinning_respects_filters_and_tags() {
        let history = history(&[0, 0, 0, 72]);
        let policy = CleanupPolicy {
            versions: Some(HashSet::from([1, 2, 3])),
            ..Default::default()
        };
        let expiry = VersionExpiry::new(
            &policy,
            Some(&[keep_one(1, 2 * DAY)]),
            at_hour(72),
            Some(latest(&history)),
            4,
        )
        .unwrap()
        .with_tagged(HashSet::from([1]));
        assert_eq!(expiry.classify(1, at_hour(0)), Verdict::TaggedOld);
        assert_eq!(kept(&expiry, &history), vec![1, 3, 4]);
    }

    #[test]
    fn survivor_ties_go_to_the_higher_version() {
        // Versions 1 and 2 commit at the same instant, in the same bucket.
        let history = history(&[0, 0, 72]);
        let expiry = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&[keep_one(DAY, 2 * DAY)]),
            at_hour(72),
            Some(latest(&history)),
            3,
        )
        .unwrap();
        assert_eq!(kept(&expiry, &history), vec![2, 3]);
    }

    #[test]
    fn tag_protection_reports_tagged_old_even_when_a_branch_holds_the_version() {
        let policy = CleanupPolicy {
            before_timestamp: Some(at_hour(50)),
            ..Default::default()
        };
        let expiry = VersionExpiry::new(&policy, None, at_hour(72), None, 3)
            .unwrap()
            .with_tagged(HashSet::from([1]))
            .with_branch_held(HashSet::from([1]));
        assert_eq!(expiry.classify(1, at_hour(0)), Verdict::TaggedOld);
    }

    #[test]
    fn tagged_version_outside_the_filters_is_not_tagged_old() {
        let policy = CleanupPolicy {
            before_timestamp: Some(at_hour(50)),
            versions: Some(HashSet::from([2])),
            ..Default::default()
        };
        let expiry = VersionExpiry::new(&policy, None, at_hour(72), None, 3)
            .unwrap()
            .with_tagged(HashSet::from([1, 2]));
        assert_eq!(expiry.classify(1, at_hour(0)), Verdict::Retained);
        assert_eq!(expiry.classify(2, at_hour(0)), Verdict::TaggedOld);
    }

    #[test]
    fn branch_held_versions_are_retained_without_a_tag_error() {
        let history = history(&[0, 1, 72]);
        let policy = CleanupPolicy {
            before_timestamp: Some(at_hour(50)),
            ..Default::default()
        };
        let expiry = VersionExpiry::new(&policy, None, at_hour(72), None, 3)
            .unwrap()
            .with_branch_held(HashSet::from([1]));
        assert_eq!(expiry.classify(1, at_hour(0)), Verdict::Retained);
        assert!(expiry.is_old(1, at_hour(0)));
        assert_eq!(kept(&expiry, &history), vec![1, 3]);
    }

    #[test]
    fn read_version_newer_and_detached_are_never_expired() {
        let expiry = VersionExpiry::new(
            &CleanupPolicy {
                before_timestamp: Some(at_hour(100)),
                ..Default::default()
            },
            None,
            at_hour(100),
            None,
            3,
        )
        .unwrap()
        .with_tagged(HashSet::from([3]));
        assert_eq!(expiry.classify(2, at_hour(0)), Verdict::Expired);
        assert_eq!(expiry.classify(3, at_hour(0)), Verdict::Retained);
        assert_eq!(expiry.classify(4, at_hour(0)), Verdict::Retained);
        let detached = lance_table::format::DETACHED_VERSION_MASK | 1;
        assert_eq!(expiry.classify(detached, at_hour(0)), Verdict::Retained);
        assert!(!expiry.is_expired_by_version(detached));
        // The read-version check is a protection, not part of the age policy.
        assert!(expiry.is_old(4, at_hour(0)));
    }

    #[rstest]
    #[case::before_version(Some(3), None, vec![3, 4, 5])]
    #[case::versions(None, Some(vec![2]), vec![1, 3, 4, 5])]
    #[case::both(Some(3), Some(vec![2, 3]), vec![1, 3, 4, 5])]
    fn legacy_version_filters(
        #[case] before_version: Option<u64>,
        #[case] versions: Option<Vec<u64>>,
        #[case] expected: Vec<u64>,
    ) {
        let policy = CleanupPolicy {
            before_version,
            versions: versions.map(HashSet::from_iter),
            ..Default::default()
        };
        let expiry = VersionExpiry::new(&policy, None, at_hour(0), None, 5).unwrap();
        assert_eq!(kept(&expiry, &history(&[0, 1, 2, 3, 4])), expected);
    }

    #[test]
    fn legacy_timestamp_and_version_filters_must_all_match() {
        let history = history(&[0, 24, 48, 72, 96]);
        let policy = CleanupPolicy {
            before_timestamp: Some(at_hour(60)),
            before_version: Some(3),
            ..Default::default()
        };
        let expiry = VersionExpiry::new(&policy, None, at_hour(96), None, 5).unwrap();
        assert_eq!(kept(&expiry, &history), vec![3, 4, 5]);
    }

    #[test]
    fn retention_without_time_limit_expires_nothing_outside_buckets() {
        let history = history(&[0, 1]);
        let expiry = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&[keep_one(DAY, 2 * DAY)]),
            at_hour(1),
            Some(latest(&history)),
            2,
        )
        .unwrap();
        assert_eq!(kept(&expiry, &history), vec![1, 2]);
    }

    #[test]
    fn expired_by_version_refuses_to_guess() {
        let expiry = |policy: CleanupPolicy| {
            VersionExpiry::new(&policy, None, at_hour(0), None, 20).unwrap()
        };
        assert!(!expiry(CleanupPolicy::default()).is_expired_by_version(1));

        let by_version = expiry(CleanupPolicy {
            before_version: Some(10),
            ..Default::default()
        });
        assert!(by_version.is_expired_by_version(9));
        assert!(!by_version.is_expired_by_version(10));
        assert!(!by_version.is_expired_by_version(11));
        let protected = by_version
            .clone()
            .with_tagged(HashSet::from([2]))
            .with_branch_held(HashSet::from([3]));
        assert!(!protected.is_expired_by_version(2));
        assert!(!protected.is_expired_by_version(3));
        assert!(protected.is_expired_by_version(4));

        let listed = CleanupPolicy {
            versions: Some(HashSet::from([3, 4, 25])),
            ..Default::default()
        };
        assert!(expiry(listed.clone()).is_expired_by_version(3));
        assert!(!expiry(listed.clone()).is_expired_by_version(5));
        assert!(!expiry(listed.clone()).is_expired_by_version(25));
        let combined = CleanupPolicy {
            before_version: Some(4),
            ..listed
        };
        assert!(expiry(combined.clone()).is_expired_by_version(3));
        assert!(!expiry(combined.clone()).is_expired_by_version(4));
        assert!(!expiry(combined.clone()).is_expired_by_version(2));

        let timed = CleanupPolicy {
            before_timestamp: Some(at_hour(100)),
            ..combined
        };
        assert!(!expiry(timed).is_expired_by_version(3));

        let retention = VersionExpiry::new(
            &CleanupPolicy {
                before_version: Some(10),
                ..Default::default()
            },
            Some(&[keep_one(DAY, 2 * DAY)]),
            at_hour(0),
            Some(ThinningAnchor {
                version: 20,
                timestamp: at_hour(0),
            }),
            20,
        )
        .unwrap();
        assert!(!retention.is_expired_by_version(1));
    }

    #[test]
    fn keep_one_without_anchor_is_an_internal_error() {
        let error = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&[keep_one(DAY, 2 * DAY)]),
            at_hour(0),
            None,
            1,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Internal { .. }), "{error}");
    }

    #[rstest]
    #[case::remove_all(remove_all(u64::MAX / 2), "remove_all.older_than")]
    #[case::keep_one(keep_one(DAY, u64::MAX / 2), "keep_one.older_than")]
    #[case::remove_all_underflow(remove_all(10_000_000_000_000), "underflows current time")]
    #[case::keep_one_underflow(keep_one(DAY, 10_000_000_000_000), "underflows anchor timestamp")]
    fn out_of_range_ages_are_rejected(#[case] rule: CleanupRule, #[case] message: &str) {
        let error = VersionExpiry::new(
            &CleanupPolicy::default(),
            Some(&[rule]),
            at_hour(0),
            Some(ThinningAnchor {
                version: 1,
                timestamp: at_hour(0),
            }),
            1,
        )
        .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains(message), "{error}");
    }
}
