// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Remove a fragment entirely.

use super::apply::ApplyState;
use super::proto::required;
use super::{Footprint, Ref};
use crate::format::pb_actions;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Remove a fragment entirely -- every row deleted, or the fragment replaced by
/// compaction.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RemoveFragment {
    fragment: Ref,
    data_change: bool,
}

impl RemoveFragment {
    /// Remove `fragment` and every row in it.
    pub fn new(fragment: Ref) -> Self {
        Self {
            fragment,
            data_change: true,
        }
    }

    /// Remove `fragment`, saying whether its rows leave the dataset.
    ///
    /// `data_change = true` is the same as [`new`](Self::new).
    ///
    /// # Contract
    ///
    /// With `data_change = false` the caller certifies, and nothing downstream
    /// checks, that every row of the fragment is added back by an
    /// [`AddFragment`](super::AddFragment) in the same operation, unchanged and
    /// with its lineage, as a compaction does.
    pub fn new_unchecked(fragment: Ref, data_change: bool) -> Self {
        Self {
            fragment,
            data_change,
        }
    }

    /// The fragment removed.
    pub fn fragment(&self) -> Ref {
        self.fragment
    }

    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let fragment_id = state.resolve_fragment(self.fragment)?;
        if !state.remove_fragment(fragment_id, self.data_change) {
            return Err(Error::invalid_input(format!(
                "RemoveFragment targets fragment {fragment_id}, which does not exist"
            )));
        }
        Ok(())
    }

    /// Whether the fragment's rows leave the dataset. See
    /// [`new_unchecked`](Self::new_unchecked).
    pub fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// Every coordinate inside the fragment, which cannot be enumerated, so the
    /// removal is recorded as such and matched by fragment id. A removal that
    /// is not a data change moved the rows somewhere rather than deleting them.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        if let Some(id) = self.fragment.committed() {
            if self.data_change {
                footprint.remove_fragment(id);
            } else {
                footprint.move_rows_out_of(id);
            }
        }
    }
}

impl From<&RemoveFragment> for pb_actions::RemoveFragment {
    fn from(value: &RemoveFragment) -> Self {
        Self {
            fragment: Some(value.fragment.into()),
            data_unchanged: !value.data_change,
        }
    }
}

impl TryFrom<pb_actions::RemoveFragment> for RemoveFragment {
    type Error = Error;

    fn try_from(message: pb_actions::RemoveFragment) -> Result<Self> {
        Ok(Self::new_unchecked(
            required(message.fragment, "RemoveFragment.fragment")?.try_into()?,
            !message.data_unchanged,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::action::test_support::{apply, backed_manifest};
    use crate::transaction::action::{Action, AddFragment, AddFragmentParts};

    #[test]
    fn test_remove_fragment() {
        let next = apply(
            &backed_manifest(),
            vec![Action::RemoveFragment(RemoveFragment::new(Ref::Committed(
                0,
            )))],
        )
        .unwrap();

        assert!(next.fragments.is_empty());
    }

    #[test]
    fn test_remove_fragment_can_drop_one_minted_in_the_same_operation() {
        let next = apply(
            &backed_manifest(),
            vec![
                Action::AddFragment(AddFragment::new_unchecked(AddFragmentParts {
                    id: Ref::Local(0),
                    physical_rows: 10,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                })),
                Action::RemoveFragment(RemoveFragment::new(Ref::Local(0))),
            ],
        )
        .unwrap();

        assert_eq!(
            next.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn test_remove_fragment_rejects_a_missing_fragment() {
        let error = apply(
            &backed_manifest(),
            vec![Action::RemoveFragment(RemoveFragment::new(Ref::Committed(
                7,
            )))],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("fragment 7"), "{error}");
    }
}
