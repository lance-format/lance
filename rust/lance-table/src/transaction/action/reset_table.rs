// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reset the table to an empty state.

use super::Footprint;
use super::apply::ApplyState;
use crate::format::pb;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;

/// Reset the table to an empty state, in preparation for a fresh schema and
/// data written by later actions in the same operation.
///
/// This is how a full `Overwrite` / `CREATE OR REPLACE` decomposes. It drops the
/// entire schema, the schema metadata, all fragments and all indices, and
/// preserves the table config, the table metadata and the base paths -- change
/// those with a [`ConfigUpdate`](super::ConfigUpdate) or an
/// [`AddBase`](super::AddBase) in the same operation.
///
/// The id counters are not reset. A field or fragment added after the reset gets
/// a fresh id, so a stale file naming an old id can never be mistaken for the
/// new table's data.
#[derive(Debug, Clone, PartialEq, DeepSizeOf, Default)]
pub struct ResetTable;

impl ResetTable {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        state.reset();
        Ok(())
    }

    /// Everything. A reset writes every coordinate there is, including ones a
    /// concurrent set would only mint, so it takes the table exclusively rather
    /// than enumerating them.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.take_exclusive();
    }
}

impl From<&ResetTable> for pb::ResetTable {
    fn from(_value: &ResetTable) -> Self {
        Self {}
    }
}

impl TryFrom<pb::ResetTable> for ResetTable {
    type Error = lance_core::Error;

    fn try_from(_message: pb::ResetTable) -> Result<Self> {
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::transaction::action::test_support::{
        added_field, apply, apply_with_indices, backed_manifest,
    };
    use crate::transaction::action::{Action, AddDataFile, AddField, AddFragment, Ref};
    use crate::transaction::test_support::sample_index_metadata;

    fn reset() -> Action {
        Action::ResetTable(ResetTable)
    }

    #[test]
    fn test_reset_empties_the_table() {
        let manifest = backed_manifest();
        assert!(!manifest.fragments.is_empty());
        assert!(!manifest.schema.fields.is_empty());

        let (next, indices) =
            apply_with_indices(&manifest, vec![reset()], vec![sample_index_metadata("idx")])
                .unwrap();

        assert!(next.fragments.is_empty());
        assert!(next.schema.fields.is_empty());
        assert!(indices.is_empty());
    }

    #[test]
    fn test_reset_preserves_config_and_base_paths() {
        let mut manifest = backed_manifest();
        manifest.config.insert("lance.keep".into(), "yes".into());
        manifest.table_metadata.insert("owner".into(), "me".into());

        let next = apply(&manifest, vec![reset()]).unwrap();

        assert_eq!(next.config.get("lance.keep"), Some(&"yes".to_string()));
        assert_eq!(next.table_metadata.get("owner"), Some(&"me".to_string()));
    }

    #[test]
    fn test_reset_then_rebuild_in_one_operation() {
        let next = apply(
            &backed_manifest(),
            vec![
                reset(),
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("fresh"),
                }),
                Action::AddFragment(AddFragment {
                    local: 0,
                    physical_rows: 4,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Local(0),
                    file: DataFile::new_unstarted("data/fresh.lance", 2, 0),
                    field_ids: vec![Ref::Local(0)],
                    data_change: true,
                }),
            ],
        )
        .unwrap();

        // The rebuilt table holds only what the operation wrote after the reset,
        // and its ids continue past the ones the old table used.
        assert_eq!(next.schema.fields.len(), 1);
        let field = next.schema.field("fresh").unwrap();
        assert_ne!(field.id, 0, "a fresh field must not reuse a dropped id");
        assert_eq!(next.fragments.len(), 1);
        assert_ne!(next.fragments[0].id, 0);
        assert_eq!(next.fragments[0].files[0].fields.as_ref(), &[field.id]);
    }

    #[test]
    fn test_reset_conflicts_with_everything() {
        use crate::transaction::action::{CompositeOperation, Footprint, UserAction};

        let footprint = |actions| {
            Footprint::from(&CompositeOperation::new(vec![UserAction::new(
                "step", actions,
            )]))
        };

        let reset = footprint(vec![reset()]);
        // Even a pure append, which writes no committed coordinate at all, is
        // preempted: its rows would either vanish or survive the reset
        // depending on which commit landed first.
        let append = footprint(vec![Action::AddFragment(AddFragment {
            local: 0,
            physical_rows: 4,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })]);

        assert!(reset.conflicts_with(&append));
        assert!(append.conflicts_with(&reset));
        assert!(reset.conflicts_with(&reset.clone()));
        assert!(!append.conflicts_with(&append.clone()));
    }
}
