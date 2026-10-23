// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reset the table to an empty state.

use super::Footprint;
use super::apply::ApplyState;
use crate::format::pb_actions;
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
/// It discards objects but never rewinds the id counters. A field or fragment
/// added after the reset gets a fresh id, so a stale file naming an old id can
/// never be mistaken for the new table's data, and the committed high-water
/// marks still cover every id handed out -- the discarded table's, and any
/// this operation minted before the reset. (The field mark is recorded only
/// on a table under the non-reusable field id contract; elsewhere field ids
/// are derived from the schema and files.)
///
/// Actions before it in the same operation are allowed, and it discards their
/// changes to the schema, fragments, and indices. After it, a committed
/// reference to a fragment from before it is rejected as naming a fragment
/// that does not exist.
#[derive(Debug, Clone, PartialEq, DeepSizeOf, Default)]
pub struct ResetTable;

impl ResetTable {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        state.reset();
        Ok(())
    }

    /// Emptying the table discards every row it held.
    pub(super) fn is_data_change(&self) -> bool {
        true
    }

    /// Everything. A reset writes every coordinate there is, including ones a
    /// concurrent set would only mint, so it takes the table exclusively rather
    /// than enumerating them.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.take_exclusive();
    }
}

impl From<&ResetTable> for pb_actions::ResetTable {
    fn from(_value: &ResetTable) -> Self {
        Self {}
    }
}

impl TryFrom<pb_actions::ResetTable> for ResetTable {
    type Error = lance_core::Error;

    fn try_from(_message: pb_actions::ResetTable) -> Result<Self> {
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature_flags::FLAG_NON_REUSABLE_FIELD_IDS;
    use crate::format::DataFile;
    use crate::transaction::action::test_support::{
        added_field, apply, apply_with_indices, backed_manifest, footprint,
    };
    use crate::transaction::action::{
        Action, AddDataFile, AddField, AddFragment, FieldRef, Ref, ReserveFragmentIds,
    };
    use crate::transaction::test_support::sample_index_metadata;
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;

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
                    id: Ref::Local(0),
                    physical_rows: 4,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                Action::AddDataFile(AddDataFile {
                    fragment: Ref::Local(0),
                    file: DataFile::new_unstarted("data/fresh.lance", ConcreteFileVersion::V2_0),
                    field_ids: vec![FieldRef::Local(0)],
                    data_change: true,
                    base: None,
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
    fn test_reset_discards_what_earlier_actions_in_the_operation_did() {
        let mint_fragment = |token| {
            Action::AddFragment(AddFragment {
                id: Ref::Local(token),
                physical_rows: 4,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: true,
            })
        };
        let write = |token, path: &str| {
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(token),
                file: DataFile::new_unstarted(path, ConcreteFileVersion::V2_0),
                field_ids: vec![FieldRef::Local(token)],
                data_change: true,
                base: None,
            })
        };
        let next = apply(
            &backed_manifest(),
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("early"),
                }),
                mint_fragment(0),
                write(0, "data/early.lance"),
                reset(),
                Action::AddField(AddField {
                    local: 1,
                    parent: None,
                    def: added_field("fresh"),
                }),
                mint_fragment(1),
                write(1, "data/fresh.lance"),
            ],
        )
        .unwrap();

        let names: Vec<_> = next.schema.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["fresh"]);
        let paths: Vec<_> = next
            .fragments
            .iter()
            .flat_map(|fragment| fragment.files.iter().map(|file| file.path.as_str()))
            .collect();
        assert_eq!(paths, vec!["data/fresh.lance"]);
    }

    fn mint_fragment(token: u32) -> Action {
        Action::AddFragment(AddFragment {
            id: Ref::Local(token),
            physical_rows: 4,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    fn mint_field(token: u32, name: &str) -> Action {
        Action::AddField(AddField {
            local: token,
            parent: None,
            def: added_field(name),
        })
    }

    /// A reset discards objects but never rewinds a counter: the committed
    /// marks cover every id the table or this operation handed out, including
    /// ids the reset then discarded. `backed_manifest` holds fragment 0 and
    /// field 0, with no fragment mark recorded yet.
    #[rstest]
    #[case::bare_reset(vec![reset()], 0, 0)]
    #[case::mint_then_reset(vec![mint_field(0, "early"), mint_fragment(0), reset()], 1, 1)]
    #[case::mint_reset_mint(
        vec![
            mint_field(0, "early"),
            mint_fragment(0),
            reset(),
            mint_field(1, "fresh"),
            mint_fragment(1),
        ],
        2,
        2
    )]
    fn test_reset_never_rewinds_the_id_counters(
        #[case] actions: Vec<Action>,
        #[case] fragment_mark: u64,
        #[case] field_mark: i32,
    ) {
        let mut manifest = backed_manifest();
        manifest.activate_non_reusable_field_ids();
        manifest.writer_feature_flags |= FLAG_NON_REUSABLE_FIELD_IDS;

        let next = apply(&manifest, actions).unwrap();

        assert_eq!(next.max_fragment_id(), Some(fragment_mark));
        assert_eq!(next.max_field_id(), field_mark);
    }

    #[test]
    fn test_reset_conflicts_with_everything() {
        let reset = footprint(vec![reset()]);
        // Even a pure append, which writes no committed coordinate at all, is
        // preempted: its rows would either vanish or survive the reset
        // depending on which commit landed first.
        let append = footprint(vec![Action::AddFragment(AddFragment {
            id: Ref::Local(0),
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

        // Nor does a set have to write anything: one that only requires a
        // coordinate (a compaction-shaped set, which is not even an insert),
        // or records nothing at all, still cannot land on either side of a
        // reset.
        let requires_only = footprint(vec![
            Action::AddFragment(AddFragment {
                id: Ref::Local(0),
                physical_rows: 4,
                row_id_meta: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                data_change: false,
            }),
            Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(0),
                file: DataFile::new_unstarted("data/compacted.lance", ConcreteFileVersion::V2_0),
                field_ids: vec![FieldRef::Committed(0)],
                data_change: false,
                base: None,
            }),
        ]);
        let records_nothing = footprint(vec![Action::ReserveFragmentIds(ReserveFragmentIds {
            count: 3,
        })]);
        for other in [requires_only, records_nothing] {
            assert!(reset.conflicts_with(&other));
            assert!(other.conflicts_with(&reset));
        }
    }
}
