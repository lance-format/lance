// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Moving between the legacy and tagged formats: the only code that knows
//! both.
//!
//! A legacy compaction version and a group of ordered-compaction transitions
//! are informationally equivalent (same digest ordering, same
//! `changed_row_addrs` semantics), and a legacy record is a valid tagged
//! record, so lifting a history reinterprets its bytes and never re-encodes
//! them.

use std::sync::Arc;

use lance_core::{Error, Result};

use super::history::RewriteRecord;
use super::imp::{FriImpl, Records, concat};
use super::tagged::TaggedHistory;
use crate::format::pb::fragment_reuse_index_details::{self as pb, transition};

/// The tagged history holding the same records as the legacy `records`.
pub fn lift_legacy(records: &Records) -> Result<Arc<dyn FriImpl>> {
    Ok(Arc::new(TaggedHistory::decode(concat(&records.stored))?))
}

/// One ordered-compaction transition per group of `version`.
pub fn version_transitions(version: &pb::Version) -> Vec<pb::Transition> {
    version
        .groups
        .iter()
        .map(|group| pb::Transition {
            sources: group.old_fragments.clone(),
            destinations: group.new_fragments.clone(),
            mapping: Some(transition::Mapping::OrderedCompaction(
                pb::OrderedCompaction {
                    changed_row_addrs: group.changed_row_addrs.clone(),
                },
            )),
        })
        .collect()
}

/// The compaction version a legacy history records for `rewrite`, refusing a
/// transition only the tagged format can hold.
pub fn legacy_version(rewrite: &RewriteRecord) -> Result<pb::Version> {
    let groups = rewrite
        .transitions()
        .iter()
        .map(|transition| match &transition.mapping {
            Some(transition::Mapping::OrderedCompaction(ordered)) => Ok(pb::Group {
                changed_row_addrs: ordered.changed_row_addrs.clone(),
                old_fragments: transition.sources.clone(),
                new_fragments: transition.destinations.clone(),
            }),
            _ => Err(Error::invalid_input(format!(
                "the transition from fragments {:?} is not an ordered compaction, which only \
                 the tagged fragment reuse format can record; lift the history first",
                transition
                    .sources
                    .iter()
                    .map(|digest| digest.id)
                    .collect::<Vec<_>>()
            ))),
        })
        .collect::<Result<_>>()?;
    Ok(pb::Version {
        dataset_version: rewrite.dataset_version(),
        groups,
    })
}

impl RewriteRecord {
    /// The rewrite a legacy compaction `version` records: one
    /// ordered-compaction transition per group, at the version's dataset
    /// version.
    pub fn from_legacy_version(version: &pb::Version) -> Self {
        Self::new(version.dataset_version, version_transitions(version))
    }
}
