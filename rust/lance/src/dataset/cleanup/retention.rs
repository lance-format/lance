// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ordered cleanup rules, policy defaults and validation.

use std::time::Duration;

use chrono::TimeDelta;
use humantime::parse_duration;
use lance_core::{Error, Result};
use serde::{Deserialize, Deserializer};

pub(super) const MIN_THINNING_AGE: Duration = Duration::from_secs(2 * 86_400);

/// An age-based rule in a cleanup retention policy, ordered by increasing age.
///
/// Each version uses the oldest matching `KeepOne` rule. `RemoveAll` may only
/// appear last and expires sampled versions too. Current versions, tags and
/// branch dependencies retain their usual protection.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CleanupRule {
    /// Keep the newest eligible commit per UTC epoch-aligned interval.
    KeepOne {
        /// Positive bucket width; each age band has independent buckets.
        #[serde(deserialize_with = "deserialize_retention_duration")]
        per_interval: Duration,
        /// Full-history window, at least two days, relative to the hinted commit
        /// (or this dataset's commit when no hint exists).
        #[serde(deserialize_with = "deserialize_retention_duration")]
        older_than: Duration,
    },
    /// Expire all matching versions older than a wall-clock age, including samples.
    RemoveAll {
        /// Non-negative expiration age. Zero expires every non-protected version.
        #[serde(deserialize_with = "deserialize_retention_duration")]
        older_than: Duration,
    },
}

fn deserialize_retention_duration<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Duration, D::Error> {
    let value = String::deserialize(deserializer)?;
    parse_duration(&value).map_err(serde::de::Error::custom)
}

/// Ordered retention rules. By default, expire versions older than two weeks.
///
/// Use with [`super::CleanupOperation::with_retention_policy`]. The legacy
/// [`super::CleanupPolicy`] remains available for version and file-cleanup filters.
/// ```
/// # use lance::dataset::cleanup::retention::{CleanupPolicy, CleanupRule};
/// # use std::time::Duration;
/// let policy = CleanupPolicy::new(vec![
///     CleanupRule::KeepOne { per_interval: Duration::from_secs(3600), older_than: Duration::from_secs(2 * 86_400) },
///     CleanupRule::RemoveAll { older_than: Duration::from_secs(30 * 86_400) },
/// ]);
/// ```
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CleanupPolicy {
    pub(super) rules: Vec<CleanupRule>,
}

impl CleanupPolicy {
    /// Create a policy from ordered rules, validated when applied to cleanup.
    pub fn new(rules: Vec<CleanupRule>) -> Self {
        Self { rules }
    }

    /// The ordered rules; the oldest matching sampling rule applies.
    pub fn rules(&self) -> &[CleanupRule] {
        &self.rules
    }
}

impl Default for CleanupPolicy {
    fn default() -> Self {
        Self::new(vec![CleanupRule::RemoveAll {
            older_than: Duration::from_secs(14 * 86_400),
        }])
    }
}

/// Parse a cleanup policy containing ordered rules with duration strings.
///
/// ```
/// # use lance::dataset::cleanup::parse_cleanup_policy;
/// let policy = parse_cleanup_policy(r#"{"rules":[
///   {"type":"keep_one","per_interval":"1h","older_than":"2days"},
///   {"type":"remove_all","older_than":"30days"}
/// ]}"#)?;
/// # Ok::<_, lance::Error>(())
/// ```
pub fn parse_cleanup_policy(value: &str) -> Result<CleanupPolicy> {
    let policy: CleanupPolicy = serde_json::from_str(value).map_err(|error| {
        Error::invalid_input(format!("Invalid cleanup retention policy: {error}"))
    })?;
    validate_retention_policy(&policy.rules)?;
    Ok(policy)
}

pub(super) fn validate_retention_policy(rules: &[CleanupRule]) -> Result<()> {
    if rules.is_empty() {
        return Err(Error::invalid_input("retention policy must not be empty"));
    }
    let mut previous_age = None;
    for (index, rule) in rules.iter().enumerate() {
        let age = match *rule {
            CleanupRule::KeepOne {
                per_interval,
                older_than,
            } => {
                if per_interval.is_zero() {
                    return Err(Error::invalid_input(format!(
                        "keep_one.per_interval must be greater than zero at policy[{index}]"
                    )));
                }
                validate_thinning_age(older_than)?;
                older_than
            }
            CleanupRule::RemoveAll { older_than } => {
                if index + 1 != rules.len() {
                    return Err(Error::invalid_input(format!(
                        "remove_all must be the final retention rule, got policy[{index}]"
                    )));
                }
                TimeDelta::from_std(older_than).map_err(|_| {
                    Error::invalid_input(format!(
                        "remove_all.older_than is out of range: {older_than:?}"
                    ))
                })?;
                older_than
            }
        };
        if previous_age.is_some_and(|previous| age <= previous) {
            return Err(Error::invalid_input(format!(
                "retention policy ages must be strictly increasing; policy[{index}] has older_than={age:?}"
            )));
        }
        previous_age = Some(age);
    }
    Ok(())
}

fn validate_thinning_age(age: Duration) -> Result<()> {
    TimeDelta::from_std(age).map_err(|_| {
        Error::invalid_input(format!("keep_one.older_than is out of range: {age:?}"))
    })?;
    if age < MIN_THINNING_AGE {
        return Err(Error::invalid_input(format!(
            "keep_one.older_than must be at least 2 days, got {age:?}"
        )));
    }
    Ok(())
}
