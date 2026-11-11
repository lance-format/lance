// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! History retention rule: the versions cleanup has expired on a branch.
//!
//! Cleanup writes the rule before deleting anything, so writers that publish
//! a reference to an old version (tags, branches, restore) can tell whether
//! that version may be in the middle of deletion. See the "History Retention"
//! section of the branch and tag format specification.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

/// Directory under a branch root that holds the branch's retention files.
pub const RETENTION_DIR: &str = "_retention";

/// The history retention rule of one branch, as stored in a retention file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", from = "RetentionFile")]
pub struct HistoryRetention {
    expired_before: u64,
    retained: BTreeSet<u64>,
}

/// The file contents before the invariant that retained versions are below
/// the floor is enforced.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RetentionFile {
    expired_before: u64,
    #[serde(default)]
    retained: BTreeSet<u64>,
}

impl From<RetentionFile> for HistoryRetention {
    fn from(file: RetentionFile) -> Self {
        Self::new(file.expired_before, file.retained)
    }
}

impl HistoryRetention {
    /// Builds a rule. Retained versions at or above `expired_before` are not
    /// expired anyway, so they are dropped.
    pub fn new(expired_before: u64, retained: impl IntoIterator<Item = u64>) -> Self {
        let retained = retained
            .into_iter()
            .filter(|version| *version < expired_before)
            .collect();
        Self {
            expired_before,
            retained,
        }
    }

    /// Parses a retention file.
    ///
    /// A malformed file yields `None`, which expires nothing. Failing open is
    /// deliberate: the rule is best-effort protection, and a damaged file must
    /// not make operations fail that would otherwise succeed.
    pub fn from_json(bytes: &[u8]) -> Option<Self> {
        match serde_json::from_slice(bytes) {
            Ok(rule) => Some(rule),
            Err(error) => {
                log::warn!("Ignoring malformed history retention file ({error}); expiring nothing");
                None
            }
        }
    }

    pub fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("a history retention rule always serializes")
    }

    pub fn expired_before(&self) -> u64 {
        self.expired_before
    }

    pub fn retained(&self) -> &BTreeSet<u64> {
        &self.retained
    }

    /// Whether `version` is expired: below the floor and not retained.
    pub fn is_expired(&self, version: u64) -> bool {
        version < self.expired_before && !self.retained.contains(&version)
    }

    /// The rule that expires every version either rule expires.
    ///
    /// Every new rule is merged with its predecessor this way, so expiration
    /// is never lifted while an earlier cleanup may still be deleting. The
    /// floor is the higher of the two, and a version below it is retained only
    /// if neither rule expires it. A version kept by both rules is below the
    /// higher floor, so it is in that rule's retained list; the union of both
    /// lists therefore covers every candidate.
    pub fn merge(&self, other: &Self) -> Self {
        let expired_before = self.expired_before.max(other.expired_before);
        let retained = self
            .retained
            .union(&other.retained)
            .copied()
            .filter(|version| !self.is_expired(*version) && !other.is_expired(*version));
        Self::new(expired_before, retained)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[test]
    fn expiration_respects_floor_and_retained() {
        let rule = HistoryRetention::new(106, [102, 105, 200]);
        assert_eq!(rule.retained(), &BTreeSet::from([102, 105]));
        assert!(rule.is_expired(101));
        assert!(!rule.is_expired(102));
        assert!(!rule.is_expired(105));
        assert!(!rule.is_expired(106));
    }

    #[test]
    fn json_round_trip() {
        let rule = HistoryRetention::new(106, [105, 102]);
        let json = rule.to_json();
        assert_eq!(
            std::str::from_utf8(&json).unwrap(),
            r#"{"expiredBefore":106,"retained":[102,105]}"#
        );
        assert_eq!(HistoryRetention::from_json(&json), Some(rule));
    }

    #[test]
    fn parsing_drops_retained_versions_above_the_floor() {
        assert_eq!(
            HistoryRetention::from_json(br#"{"expiredBefore":5,"retained":[2,9]}"#),
            Some(HistoryRetention::new(5, [2]))
        );
        assert_eq!(
            HistoryRetention::from_json(br#"{"expiredBefore":5}"#),
            Some(HistoryRetention::new(5, []))
        );
    }

    #[rstest]
    #[case::empty(b"")]
    #[case::not_json(b"expired")]
    #[case::missing_floor(br#"{"retained":[1]}"#)]
    #[case::negative_floor(br#"{"expiredBefore":-1}"#)]
    #[case::bad_retained(br#"{"expiredBefore":10,"retained":["x"]}"#)]
    fn malformed_files_expire_nothing(#[case] bytes: &[u8]) {
        assert_eq!(HistoryRetention::from_json(bytes), None);
    }

    #[test]
    fn merge_keeps_only_versions_neither_rule_expires() {
        // Rule A expires below 106 except 102 and 105; rule B expires below 110
        // except 105 and 108. 102 is expired by B, 108 is above A's floor.
        let a = HistoryRetention::new(106, [102, 105]);
        let b = HistoryRetention::new(110, [105, 108]);
        let merged = HistoryRetention::new(110, [105, 108]);
        assert_eq!(a.merge(&b), merged);
        assert_eq!(b.merge(&a), merged);
    }

    #[test]
    fn merge_never_lifts_expiration() {
        // A later plan that keeps version 1 (say, a tag appeared mid-run) must
        // not make it referenceable again.
        let current = HistoryRetention::new(2, []);
        let planned = HistoryRetention::new(3, [1]);
        assert_eq!(planned.merge(&current), HistoryRetention::new(3, []));
    }
}
