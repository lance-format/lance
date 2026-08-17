// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Remove a fragment entirely.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{Footprint, Ref};
use crate::format::pb;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Remove a fragment entirely -- every row deleted, or the fragment replaced by
/// compaction.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct RemoveFragment {
    pub fragment: Ref,
    /// See [`AddFragment::data_change`](super::AddFragment::data_change).
    pub data_change: bool,
}

impl RemoveFragment {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let fragment_id = state.resolve_fragment(self.fragment)?;
        if !state.remove_fragment(fragment_id) {
            return Err(Error::invalid_input(format!(
                "RemoveFragment targets fragment {fragment_id}, which does not exist"
            )));
        }
        Ok(())
    }

    /// Every coordinate inside the fragment, which cannot be enumerated, so the
    /// removal is recorded as such and matched by fragment id.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        if let Some(id) = self.fragment.committed() {
            footprint.remove_fragment(id);
        }
    }
}

impl From<&RemoveFragment> for pb::RemoveFragment {
    fn from(value: &RemoveFragment) -> Self {
        Self {
            fragment: Some(value.fragment.into()),
            data_change: data_change_to_wire(value.data_change),
        }
    }
}

impl TryFrom<pb::RemoveFragment> for RemoveFragment {
    type Error = Error;

    fn try_from(message: pb::RemoveFragment) -> Result<Self> {
        Ok(Self {
            fragment: required(message.fragment, "RemoveFragment.fragment")?.try_into()?,
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::action::test_support::{apply, backed_manifest};
    use crate::transaction::action::{Action, AddFragment};

    #[test]
    fn test_remove_fragment() {
        let next = apply(
            &backed_manifest(),
            vec![Action::RemoveFragment(RemoveFragment {
                fragment: Ref::Committed(0),
                data_change: true,
            })],
        )
        .unwrap();

        assert!(next.fragments.is_empty());
    }

    #[test]
    fn test_remove_fragment_can_drop_one_minted_in_the_same_operation() {
        let next = apply(
            &backed_manifest(),
            vec![
                Action::AddFragment(AddFragment {
                    local: 0,
                    physical_rows: 10,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                Action::RemoveFragment(RemoveFragment {
                    fragment: Ref::Local(0),
                    data_change: true,
                }),
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
            vec![Action::RemoveFragment(RemoveFragment {
                fragment: Ref::Committed(7),
                data_change: true,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("fragment 7"), "{error}");
    }
}
