// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reserve a range of stable row ids for a later writer.

use super::Footprint;
use super::apply::ApplyState;
use crate::format::pb;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Reserve a contiguous range of stable row ids from the counter, for a later
/// writer to populate.
///
/// The counterpart of [`ReserveFragmentIds`](super::ReserveFragmentIds) for the
/// row id space, and needed alongside it: reserving a fragment id fixes the
/// high half of a row address, but on a dataset with stable row ids an index
/// records row ids instead, and those come off their own counter.
///
/// The reservation is taken after this operation's own fragments are numbered,
/// so the reserved range is always the `count` ids ending at the committed
/// manifest's `next_row_id`. The reserving writer learns which ids it got by
/// reading that back, and a fragment written against the range carries those
/// ids in its row id sequence, which apply then leaves as it finds them.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct ReserveRowIds {
    pub count: u64,
}

impl ReserveRowIds {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        state.reserve_row_ids(self.count);
        Ok(())
    }

    /// A reserved id backs no row until something is written against it.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// Nothing. Ids come off a monotonic counter, so two operations reserving
    /// at once get disjoint ranges rather than colliding.
    pub(super) fn footprint(&self, _footprint: &mut Footprint) {}
}

impl From<&ReserveRowIds> for pb::ReserveRowIds {
    fn from(value: &ReserveRowIds) -> Self {
        Self { count: value.count }
    }
}

impl TryFrom<pb::ReserveRowIds> for ReserveRowIds {
    type Error = Error;

    fn try_from(message: pb::ReserveRowIds) -> Result<Self> {
        Ok(Self {
            count: message.count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{Fragment, RowIdMeta};
    use crate::rowids::{RowIdSequence, read_row_ids, write_row_ids};
    use crate::transaction::action::test_support::{apply, backed_manifest};
    use crate::transaction::action::{Action, AddFragment, Ref};
    use crate::transaction::test_support::make_stable_row_id_manifest;

    fn reserve(count: u64) -> Action {
        Action::ReserveRowIds(ReserveRowIds { count })
    }

    fn add_fragment(local: u32, physical_rows: u64) -> Action {
        Action::AddFragment(AddFragment {
            id: Ref::Local(local),
            physical_rows,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    fn row_id_meta(row_ids: std::ops::Range<u64>) -> RowIdMeta {
        let sequence = RowIdSequence::try_from_iter(row_ids).unwrap();
        RowIdMeta::Inline(write_row_ids(&sequence).into())
    }

    /// One committed fragment holding row ids 0..10, with the counter at 1000.
    fn stable_manifest() -> crate::format::Manifest {
        let mut fragment = Fragment::new(1);
        fragment.physical_rows = Some(10);
        fragment.row_id_meta = Some(row_id_meta(0..10));
        make_stable_row_id_manifest(vec![fragment])
    }

    #[test]
    fn test_reserving_raises_the_watermark_without_adding_rows() {
        let manifest = stable_manifest();
        assert_eq!(manifest.next_row_id, 1000);

        let next = apply(&manifest, vec![reserve(50)]).unwrap();

        // Ids 1000..1050 are now spoken for, but no fragment holds them.
        assert_eq!(next.next_row_id, 1050);
        assert_eq!(next.fragments.len(), 1);
    }

    #[test]
    fn test_the_reservation_sits_above_the_operations_own_row_ids() {
        let next = apply(&stable_manifest(), vec![reserve(50), add_fragment(0, 10)]).unwrap();

        // The appended fragment numbers from the old watermark and the
        // reservation lands on top, so the reserved range is always the
        // trailing `count` ids -- whichever order the actions arrived in.
        assert_eq!(next.next_row_id, 1060);
        let added = next.fragments.iter().find(|f| f.id == 2).unwrap();
        let Some(RowIdMeta::Inline(data)) = &added.row_id_meta else {
            panic!("the added fragment should carry an inline row id sequence");
        };
        let row_ids = read_row_ids(data).unwrap().iter().collect::<Vec<_>>();
        assert_eq!(row_ids, (1000..1010).collect::<Vec<_>>());
    }

    #[test]
    fn test_reservations_accumulate() {
        let next = apply(&stable_manifest(), vec![reserve(10), reserve(5)]).unwrap();

        assert_eq!(next.next_row_id, 1015);
    }

    #[test]
    fn test_reserving_nothing_is_a_no_op() {
        let manifest = stable_manifest();
        let next = apply(&manifest, vec![reserve(0)]).unwrap();

        assert_eq!(next.next_row_id, manifest.next_row_id);
    }

    #[test]
    fn test_a_fragment_may_carry_reserved_row_ids() {
        // The whole point of the action: reserve in one commit, then write data
        // (and an index over it) against ids that are already known.
        let reserved = apply(&stable_manifest(), vec![reserve(10)]).unwrap();
        assert_eq!(reserved.next_row_id, 1010);

        let next = apply(
            &reserved,
            vec![Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 10,
                row_id_meta: Some(row_id_meta(1000..1010)),
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            })],
        )
        .unwrap();

        // The reservation already moved the counter, so applying the fragment
        // leaves it where it is rather than numbering the rows again.
        assert_eq!(next.next_row_id, 1010);
        let added = next.fragments.iter().find(|f| f.id == 2).unwrap();
        let Some(RowIdMeta::Inline(data)) = &added.row_id_meta else {
            panic!("the supplied sequence should have been kept");
        };
        assert_eq!(
            read_row_ids(data).unwrap().iter().collect::<Vec<_>>(),
            (1000..1010).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_unreserved_row_ids_are_rejected() {
        // 1000 onwards is where the counter stands, so these were never set
        // aside and the next append would hand them out a second time.
        let error = apply(
            &stable_manifest(),
            vec![Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 10,
                row_id_meta: Some(row_id_meta(1000..1010)),
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            })],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(
            error.to_string().contains("no commit has reserved"),
            "{error}"
        );
    }

    #[test]
    fn test_reserving_without_stable_row_ids_is_rejected() {
        // `backed_manifest` has no row id counter to reserve from.
        let error = apply(&backed_manifest(), vec![reserve(10)]).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
        assert!(error.to_string().contains("stable row ids"), "{error}");
    }
}
