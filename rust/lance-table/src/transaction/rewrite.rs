// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rewrite-operation payload types.
//!
//! `RewriteGroup`, `RewrittenIndex`, and `DataReplacementGroup` describe the
//! per-group payloads carried by [`Operation::Rewrite`] and
//! [`Operation::DataReplacement`] (still defined in `lance::dataset::transaction`).
//! They are pure data with only `lance-table` / `uuid` / `prost-types`
//! dependencies, so they live alongside the other relocated transaction
//! payload types here.

use crate::format::pb;
use crate::format::{DataFile, Fragment, IndexFile};
use deepsize::DeepSizeOf;
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use uuid::Uuid;

#[derive(Debug, Clone, DeepSizeOf, PartialEq)]
pub struct DataReplacementGroup(pub u64, pub DataFile);

#[derive(Debug, Clone, PartialEq)]
pub struct RewrittenIndex {
    pub old_id: Uuid,
    pub new_id: Uuid,
    pub new_index_details: prost_types::Any,
    pub new_index_version: u32,
    /// Files in the new index with their sizes.
    /// Empty list from older writers that didn't persist this field.
    pub new_index_files: Option<Vec<IndexFile>>,
}

impl DeepSizeOf for RewrittenIndex {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.new_index_details
            .type_url
            .deep_size_of_children(context)
            + self.new_index_details.value.deep_size_of_children(context)
    }
}

#[derive(Debug, Clone, DeepSizeOf)]
pub struct RewriteGroup {
    pub old_fragments: Vec<Fragment>,
    pub new_fragments: Vec<Fragment>,
}

impl PartialEq for RewriteGroup {
    fn eq(&self, other: &Self) -> bool {
        fn compare_vec<T: PartialEq>(a: &[T], b: &[T]) -> bool {
            a.len() == b.len() && a.iter().all(|f| b.contains(f))
        }
        compare_vec(&self.old_fragments, &other.old_fragments)
            && compare_vec(&self.new_fragments, &other.new_fragments)
    }
}

impl From<&DataReplacementGroup> for pb::transaction::DataReplacementGroup {
    fn from(DataReplacementGroup(fragment_id, new_file): &DataReplacementGroup) -> Self {
        Self {
            fragment_id: *fragment_id,
            new_file: Some(new_file.into()),
        }
    }
}

/// Convert a protobug DataReplacementGroup to a rust native DataReplacementGroup
/// this is unfortunately TryFrom instead of From because of the Option in the pb::DataReplacementGroup
impl TryFrom<pb::transaction::DataReplacementGroup> for DataReplacementGroup {
    type Error = Error;

    fn try_from(message: pb::transaction::DataReplacementGroup) -> Result<Self> {
        Ok(Self(
            message.fragment_id,
            message
                .new_file
                .ok_or(Error::invalid_input(
                    "DataReplacementGroup must have a new_file",
                ))?
                .try_into()?,
        ))
    }
}

impl TryFrom<&pb::transaction::rewrite::RewrittenIndex> for RewrittenIndex {
    type Error = Error;

    fn try_from(message: &pb::transaction::rewrite::RewrittenIndex) -> Result<Self> {
        Ok(Self {
            old_id: message
                .old_id
                .as_ref()
                .map(Uuid::try_from)
                .ok_or_else(|| {
                    Error::invalid_input("required field (old_id) missing from message".to_string())
                })??,
            new_id: message
                .new_id
                .as_ref()
                .map(Uuid::try_from)
                .ok_or_else(|| {
                    Error::invalid_input("required field (new_id) missing from message".to_string())
                })??,
            new_index_details: message
                .new_index_details
                .as_ref()
                .ok_or_else(|| {
                    Error::invalid_input("new_index_details is a required field".to_string())
                })?
                .clone(),
            new_index_version: message.new_index_version,
            new_index_files: if message.new_index_files.is_empty() {
                None
            } else {
                Some(
                    message
                        .new_index_files
                        .iter()
                        .map(|f| IndexFile {
                            path: f.path.clone(),
                            size_bytes: f.size_bytes,
                        })
                        .collect(),
                )
            },
        })
    }
}

impl TryFrom<pb::transaction::rewrite::RewriteGroup> for RewriteGroup {
    type Error = Error;

    fn try_from(message: pb::transaction::rewrite::RewriteGroup) -> Result<Self> {
        Ok(Self {
            old_fragments: message
                .old_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
            new_fragments: message
                .new_fragments
                .into_iter()
                .map(Fragment::try_from)
                .collect::<Result<Vec<_>>>()?,
        })
    }
}

impl From<&RewrittenIndex> for pb::transaction::rewrite::RewrittenIndex {
    fn from(value: &RewrittenIndex) -> Self {
        Self {
            old_id: Some((&value.old_id).into()),
            new_id: Some((&value.new_id).into()),
            new_index_details: Some(value.new_index_details.clone()),
            new_index_version: value.new_index_version,
            new_index_files: value
                .new_index_files
                .as_ref()
                .map(|files| {
                    files
                        .iter()
                        .map(|f| pb::IndexFile {
                            path: f.path.clone(),
                            size_bytes: f.size_bytes,
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

impl From<&RewriteGroup> for pb::transaction::rewrite::RewriteGroup {
    fn from(value: &RewriteGroup) -> Self {
        Self {
            old_fragments: value
                .old_fragments
                .iter()
                .map(pb::DataFragment::from)
                .collect(),
            new_fragments: value
                .new_fragments
                .iter()
                .map(pb::DataFragment::from)
                .collect(),
        }
    }
}

/// Re-point an index `fragment_bitmap` across a list of rewrite groups.
///
/// An index covers a fragment when its bitmap contains the fragment's id. A
/// rewrite group either replaces a covered set wholesale — every old fragment
/// was indexed, so every new fragment inherits coverage — or it does not
/// touch the index at all. A group whose old fragments are a split of indexed
/// and non-indexed data is rejected: the legacy compaction planner
/// (`plan_compaction`) refuses to emit such a group, so this is a writer-bug
/// guard rather than a runtime concern.
///
/// Shared between the legacy `Transaction::build_manifest` compaction path and
/// the action-based `IndexCoverageRemap::RewriteGroups` apply.
pub fn recalculate_fragment_bitmap(
    old: &RoaringBitmap,
    groups: &[RewriteGroup],
) -> Result<RoaringBitmap> {
    let mut new_bitmap = old.clone();
    for group in groups {
        let any_in_index = group
            .old_fragments
            .iter()
            .any(|frag| old.contains(frag.id as u32));
        let all_in_index = group
            .old_fragments
            .iter()
            .all(|frag| old.contains(frag.id as u32));
        // Any rewrite group may or may not be covered by the index.  However, if any fragment
        // in a rewrite group was previously covered by the index then all fragments in the rewrite
        // group must have been previously covered by the index.  plan_compaction takes care of
        // this for us so this should be safe to assume.
        if any_in_index {
            if all_in_index {
                for frag_id in group.old_fragments.iter().map(|frag| frag.id as u32) {
                    new_bitmap.remove(frag_id);
                }
                new_bitmap.extend(group.new_fragments.iter().map(|frag| frag.id as u32));
            } else {
                return Err(Error::invalid_input(
                    "The compaction plan included a rewrite group that was a split of indexed and non-indexed data",
                ));
            }
        }
    }
    Ok(new_bitmap)
}
