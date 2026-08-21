// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reserve a range of fragment ids for a later writer.

use super::Footprint;
use super::apply::ApplyState;
use crate::format::pb;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Reserve a contiguous range of fragment ids from the counter, for a later
/// (possibly distributed) writer to populate.
///
/// The range starts wherever the counter stands when this action is applied, so
/// the reserving writer learns which ids it got by reading the committed
/// manifest's high-water mark: the range is the `count` ids ending there.
/// Fragments written against the range name those ids as
/// [`Ref::Committed`](super::Ref::Committed).
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct ReserveFragmentIds {
    pub count: u32,
}

impl ReserveFragmentIds {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        state.reserve_fragment_ids(self.count);
        Ok(())
    }

    /// A reserved id backs no rows until something is written to it.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// Nothing. Ids come off a monotonic counter, so two operations reserving
    /// at once get disjoint ranges rather than colliding.
    pub(super) fn footprint(&self, _footprint: &mut Footprint) {}
}

impl From<&ReserveFragmentIds> for pb::ReserveFragmentIds {
    fn from(value: &ReserveFragmentIds) -> Self {
        Self { count: value.count }
    }
}

impl TryFrom<pb::ReserveFragmentIds> for ReserveFragmentIds {
    type Error = Error;

    fn try_from(message: pb::ReserveFragmentIds) -> Result<Self> {
        Ok(Self {
            count: message.count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::action::test_support::{apply, backed_manifest};
    use crate::transaction::action::{Action, AddFragment};

    fn reserve(count: u32) -> Action {
        Action::ReserveFragmentIds(ReserveFragmentIds { count })
    }

    fn add_fragment(local: u32) -> Action {
        Action::AddFragment(AddFragment {
            local,
            physical_rows: 10,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    #[test]
    fn test_reserving_raises_the_high_water_mark_without_adding_fragments() {
        let manifest = backed_manifest();
        assert_eq!(manifest.max_fragment_id(), Some(0));

        let next = apply(&manifest, vec![reserve(3)]).unwrap();

        // Ids 1, 2 and 3 are now spoken for, but no fragment exists for them.
        assert_eq!(next.max_fragment_id(), Some(3));
        assert_eq!(next.fragments.len(), 1);
    }

    #[test]
    fn test_a_later_mint_skips_the_reserved_range() {
        let next = apply(&backed_manifest(), vec![reserve(3), add_fragment(0)]).unwrap();

        assert_eq!(
            next.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0, 4],
            "the minted fragment must not land inside the reserved range"
        );
    }

    #[test]
    fn test_a_reserved_id_is_usable_by_the_next_operation() {
        let reserved = apply(&backed_manifest(), vec![reserve(2)]).unwrap();
        assert_eq!(reserved.max_fragment_id(), Some(2));

        // The next operation mints past the range rather than into it: the
        // reservation holds even though nothing was written against it.
        let next = apply(&reserved, vec![add_fragment(0)]).unwrap();
        assert_eq!(
            next.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0, 3]
        );
    }

    #[test]
    fn test_reserving_nothing_is_a_no_op() {
        let manifest = backed_manifest();
        let next = apply(&manifest, vec![reserve(0)]).unwrap();

        assert_eq!(next.max_fragment_id(), manifest.max_fragment_id());
    }
}
