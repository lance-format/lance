// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Restamp the row-version metadata of fragments rewritten in place.

use super::apply::ApplyState;
use super::{Coordinate, Footprint};
use crate::format::pb;
use crate::rowids::version::refresh_row_latest_update_meta_for_full_frag_rewrite_cols;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Record that every row of these fragments was updated by this operation.
///
/// Under stable row ids a fragment carries a `last_updated_at_version` sequence
/// per row. Rewriting a fragment's columns in place leaves the rows where they
/// are, so nothing else in the operation restates when they last changed; this
/// action does, mirroring the refresh a legacy `Merge` performs implicitly.
///
/// `created_at_version` is deliberately untouched: the rows are the same rows,
/// and a row minted by this operation gets both stamps from the
/// [`AddFragment`](super::AddFragment) that minted it.
///
/// Fragments are named by committed id. A fragment this operation minted has
/// nothing to restamp.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RefreshRowVersionMetadata {
    pub fragment_ids: Vec<u64>,
}

impl RefreshRowVersionMetadata {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        if !self.fragment_ids.is_empty() && !state.uses_stable_row_ids() {
            return Err(Error::invalid_input(
                "RefreshRowVersionMetadata restamps the per-row version sequences, which only \
                 exist on a dataset using stable row ids",
            ));
        }
        let new_version = state.new_version();
        for fragment_id in &self.fragment_ids {
            let fragment = state.fragment_mut(*fragment_id, "RefreshRowVersionMetadata")?;
            refresh_row_latest_update_meta_for_full_frag_rewrite_cols(fragment, new_version)?;
        }
        Ok(())
    }

    /// The rows themselves are restated elsewhere in the operation -- by the
    /// data files it writes -- so this is bookkeeping about that change, not a
    /// change of its own.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        for fragment_id in &self.fragment_ids {
            footprint.add(Coordinate::FragmentRowVersions(*fragment_id));
        }
    }
}

impl From<&RefreshRowVersionMetadata> for pb::RefreshRowVersionMetadata {
    fn from(value: &RefreshRowVersionMetadata) -> Self {
        Self {
            fragment_ids: value.fragment_ids.clone(),
        }
    }
}

impl TryFrom<pb::RefreshRowVersionMetadata> for RefreshRowVersionMetadata {
    type Error = Error;

    fn try_from(message: pb::RefreshRowVersionMetadata) -> Result<Self> {
        Ok(Self {
            fragment_ids: message.fragment_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{DataFile, Fragment, RowIdMeta};
    use crate::rowids::{RowIdSequence, write_row_ids};
    use crate::transaction::action::test_support::{apply, backed_manifest};
    use crate::transaction::action::{Action, CompositeOperation, UserAction};
    use crate::transaction::test_support::make_stable_row_id_manifest;
    use std::sync::Arc;

    fn refresh(fragment_ids: Vec<u64>) -> Action {
        Action::RefreshRowVersionMetadata(RefreshRowVersionMetadata { fragment_ids })
    }

    /// A stable-row-id manifest whose fragment 0 has `rows` rows, all last
    /// updated at version 1.
    fn manifest_with_rows(rows: usize) -> crate::format::Manifest {
        let row_ids = RowIdSequence::from((0..rows as u64).collect::<Vec<_>>().as_slice());
        let mut fragment = Fragment {
            id: 0,
            files: vec![DataFile::new(
                "data.lance",
                vec![0],
                vec![0],
                2,
                0,
                None,
                None,
            )],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: Some(RowIdMeta::Inline(write_row_ids(&row_ids))),
            physical_rows: Some(rows),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };
        refresh_row_latest_update_meta_for_full_frag_rewrite_cols(&mut fragment, 1).unwrap();
        make_stable_row_id_manifest(vec![fragment])
    }

    fn last_updated_versions(fragment: &Fragment) -> Vec<u64> {
        let meta = fragment
            .last_updated_at_version_meta
            .as_ref()
            .expect("the fragment should carry a last-updated sequence");
        let sequence = meta.load_sequence().unwrap();
        (0..fragment.physical_rows.unwrap())
            .map(|offset| sequence.version_at(offset).unwrap())
            .collect()
    }

    #[test]
    fn test_refresh_stamps_every_row_with_the_version_the_commit_produces() {
        let manifest = manifest_with_rows(3);
        let expected = manifest.version + 1;

        let out = apply(&manifest, vec![refresh(vec![0])]).unwrap();

        assert_eq!(last_updated_versions(&out.fragments[0]), vec![expected; 3]);
    }

    #[test]
    fn test_refresh_leaves_the_created_at_sequence_alone() {
        let mut manifest = manifest_with_rows(3);
        let mut fragment = manifest.fragments[0].clone();
        fragment.created_at_version_meta = fragment.last_updated_at_version_meta.clone();
        let created_at = fragment.created_at_version_meta.clone();
        manifest.fragments = Arc::new(vec![fragment]);

        let out = apply(&manifest, vec![refresh(vec![0])]).unwrap();

        assert_eq!(out.fragments[0].created_at_version_meta, created_at);
    }

    #[test]
    fn test_refreshing_no_fragments_is_allowed_on_any_dataset() {
        // An empty list is what a translated operation produces when nothing was
        // rewritten in place, so it must not depend on stable row ids.
        apply(&backed_manifest(), vec![refresh(vec![])]).unwrap();
    }

    #[test]
    fn test_refreshing_without_stable_row_ids_is_rejected() {
        let error = apply(&backed_manifest(), vec![refresh(vec![0])]).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("stable row ids"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_refreshing_a_fragment_that_is_not_there_is_rejected() {
        let error = apply(&manifest_with_rows(3), vec![refresh(vec![7])]).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("RefreshRowVersionMetadata targets fragment 7, which does not exist"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_two_writers_refreshing_the_same_fragment_conflict() {
        let footprint = |actions| {
            Footprint::from(&CompositeOperation::new(vec![UserAction::new(
                "step", actions,
            )]))
        };

        let ours = footprint(vec![refresh(vec![0])]);
        let same = footprint(vec![refresh(vec![0, 1])]);
        let other = footprint(vec![refresh(vec![1])]);

        assert!(ours.conflicts_with(&same));
        assert!(!ours.conflicts_with(&other));
    }
}
