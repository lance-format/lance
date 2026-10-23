// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Assert that no concurrent commit inserted a colliding key.

use super::apply::ApplyState;
use super::proto::required;
use super::{FieldRef, Footprint};
use crate::format::key_existence::KeyExistenceFilter;
use crate::format::pb_actions;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// A precondition rather than a delta: the keys this operation inserts must not
/// collide with keys a concurrent commit inserted.
///
/// This is the home for merge-insert's strict primary-key conflict detection.
/// The key columns are an unenforced primary key, so nothing in the manifest
/// records which keys exist -- the filter of inserted key hashes travels with
/// the operation because it cannot be recovered from any post-image.
///
/// Two operations that both carry one are compatible when they agree on the key
/// columns and their filters do not intersect. An operation carrying one is not
/// compatible with a concurrent operation that inserts rows without saying which
/// keys they carry, because there is nothing to compare against.
///
/// An assertion speaks for *every* row the operation inserts, over its key
/// columns. An operation may carry several, one per key column set, but each
/// must still cover all of its inserted rows: two merge-inserts over different
/// keys squashed into one operation cannot each assert only their own rows,
/// because the other's rows carry values in those columns too. Conflict
/// detection matches assertions by key column set on that assumption.
///
/// An assertion over a column the same operation mints ([`FieldRef::Local`]) guards
/// nothing concurrent: no other writer can insert into that column, so conflict
/// detection ignores it.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AssertUniqueKeys {
    /// The key columns, in order, and the only key list: the field ids the
    /// filter type carries are an artifact of its use by the legacy operation,
    /// cleared on encode and rejected on decode.
    pub key_fields: Vec<FieldRef>,
    /// Hashes of the key values of every row this operation inserts, over
    /// `key_fields`.
    pub filter: KeyExistenceFilter,
}

impl AssertUniqueKeys {
    /// Nothing to apply -- the assertion is checked when two operations are
    /// compared, not when one is folded into a manifest. The key columns are
    /// still resolved and looked up, so an assertion naming a field that is not
    /// there fails at the commit that carries it rather than silently guarding
    /// nothing.
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        if self.key_fields.is_empty() {
            return Err(Error::invalid_input(
                "AssertUniqueKeys names no key column, so there is nothing for it to assert",
            ));
        }
        for key_field in &self.key_fields {
            let field_id = state.resolve_schema_field(*key_field, "AssertUniqueKeys")?;
            if state.schema().field_by_id(field_id).is_none() {
                return Err(Error::invalid_input(format!(
                    "AssertUniqueKeys names key field {field_id}, which is not in the schema"
                )));
            }
        }
        Ok(())
    }

    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.assert_unique_keys(self.key_fields.clone(), self.filter.clone());
    }
}

impl From<&AssertUniqueKeys> for pb_actions::AssertUniqueKeys {
    fn from(value: &AssertUniqueKeys) -> Self {
        let mut filter = pb_actions::KeyExistenceFilter::from(&value.filter);
        // `key_fields` is the authoritative list; the filter's own copy is an
        // artifact of the type it shares with the legacy Update operation.
        filter.field_ids.clear();
        Self {
            key_fields: value.key_fields.iter().map(|&f| f.into()).collect(),
            filter: Some(filter),
        }
    }
}

impl TryFrom<pb_actions::AssertUniqueKeys> for AssertUniqueKeys {
    type Error = Error;

    fn try_from(message: pb_actions::AssertUniqueKeys) -> Result<Self> {
        let filter = required(message.filter, "AssertUniqueKeys.filter")?;
        let key_fields = message
            .key_fields
            .into_iter()
            .map(FieldRef::try_from)
            .collect::<Result<Vec<_>>>()?;
        if !filter.field_ids.is_empty() {
            return Err(Error::invalid_input(format!(
                "AssertUniqueKeys carries filter.field_ids {:?} beside key_fields {key_fields:?}; \
                 filter.field_ids must be empty, since key_fields is the only key list",
                filter.field_ids
            )));
        }
        Ok(Self {
            key_fields,
            filter: KeyExistenceFilter::try_from(&filter)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::DataFile;
    use crate::format::key_existence::FilterType;
    use crate::format::overlay::{DataOverlayFile, OverlayCoverage};
    use crate::transaction::action::test_support::{apply, backed_manifest, footprint};
    use crate::transaction::action::{
        Action, AddFragment, AddOverlays, Ref, RemoveFragment, TombstoneFieldData,
    };
    use lance_file::version::ConcreteFileVersion;
    use roaring::RoaringBitmap;
    use std::sync::Arc;

    fn assertion(key_fields: Vec<FieldRef>, hashes: &[u64]) -> Action {
        Action::AssertUniqueKeys(AssertUniqueKeys {
            key_fields,
            filter: KeyExistenceFilter {
                field_ids: Vec::new(),
                filter: FilterType::ExactSet(hashes.iter().copied().collect()),
            },
        })
    }

    fn append(local: u32) -> Action {
        Action::AddFragment(AddFragment {
            id: Ref::Local(local),
            physical_rows: 5,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    fn compact(local: u32) -> Action {
        let Action::AddFragment(mut fragment) = append(local) else {
            unreachable!()
        };
        fragment.data_change = false;
        Action::AddFragment(fragment)
    }

    /// `key_fields` is the only key list. The filter type keeps a field list
    /// of its own for the legacy operation, and a second list here could only
    /// disagree with the first.
    #[test]
    fn test_decoding_rejects_a_filter_with_its_own_field_ids() {
        let Action::AssertUniqueKeys(assertion) = assertion(vec![FieldRef::Committed(0)], &[1])
        else {
            unreachable!()
        };
        let mut message = pb_actions::AssertUniqueKeys::from(&assertion);
        message.filter.as_mut().unwrap().field_ids = vec![0];

        let error = AssertUniqueKeys::try_from(message).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .contains("filter.field_ids [0] beside key_fields [Committed(0)]"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_an_assertion_changes_nothing() {
        let manifest = backed_manifest();
        let out = apply(
            &manifest,
            vec![assertion(vec![FieldRef::Committed(0)], &[1, 2, 3])],
        )
        .unwrap();

        assert_eq!(out.fragments, manifest.fragments);
        assert_eq!(out.schema, manifest.schema);
    }

    #[test]
    fn test_asserting_over_no_key_column_is_rejected() {
        let error = apply(&backed_manifest(), vec![assertion(vec![], &[1])]).unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("names no key column"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_asserting_over_a_field_that_is_not_there_is_rejected() {
        let error = apply(
            &backed_manifest(),
            vec![assertion(vec![FieldRef::Committed(9)], &[1])],
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("not in the schema"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_inserts_with_disjoint_keys_do_not_conflict() {
        let ours = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[1, 2]),
        ]);
        let theirs = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[3, 4]),
        ]);

        assert!(!ours.conflicts_with(&theirs));
        assert!(!theirs.conflicts_with(&ours));
    }

    #[test]
    fn test_inserts_sharing_a_key_conflict() {
        let ours = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[1, 2]),
        ]);
        let theirs = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[2, 3]),
        ]);

        assert!(ours.conflicts_with(&theirs));
        assert!(theirs.conflicts_with(&ours));
    }

    /// A local key field names a column the operation mints, which no
    /// concurrent writer can insert into, so the assertion has nothing to
    /// guard. Two operations' `Local(3)` also name different columns, so
    /// matching them would compare hashes of unrelated values.
    #[test]
    fn test_an_assertion_over_a_minted_key_column_does_not_conflict() {
        let over = |key_field: FieldRef, hashes: &[u64]| {
            footprint(vec![append(0), assertion(vec![key_field], hashes)])
        };
        let plain_append = footprint(vec![append(0)]);

        assert!(
            !over(FieldRef::Local(3), &[1, 2]).conflicts_with(&over(FieldRef::Local(3), &[2, 3]))
        );
        assert!(!over(FieldRef::Local(3), &[1, 2]).conflicts_with(&plain_append));
        assert!(!plain_append.conflicts_with(&over(FieldRef::Local(3), &[1, 2])));
        // The same overlap over a committed column is a collision.
        assert!(
            over(FieldRef::Committed(3), &[1, 2])
                .conflicts_with(&over(FieldRef::Committed(3), &[2, 3]))
        );
    }

    #[test]
    fn test_assertions_over_different_key_columns_conflict() {
        // Two filters over different columns hash different values, so neither
        // says anything about the other's keys.
        let ours = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[1]),
        ]);
        let theirs = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(1)], &[2]),
        ]);

        assert!(ours.conflicts_with(&theirs));
    }

    /// Each assertion speaks for every row the set inserts, over its own key
    /// columns. A set carrying two of them -- a table with two unique keys,
    /// each asserted over all the inserted rows -- is compared column set by
    /// column set, not every assertion against every other.
    #[test]
    fn test_assertions_are_matched_by_key_column() {
        let ours = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[1]),
            assertion(vec![FieldRef::Committed(1)], &[5]),
        ]);
        let disjoint = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[2]),
            assertion(vec![FieldRef::Committed(1)], &[6]),
        ]);
        let overlapping_on_the_second = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[2]),
            assertion(vec![FieldRef::Committed(1)], &[5]),
        ]);
        let silent_on_the_second = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[2]),
        ]);

        assert!(!ours.conflicts_with(&disjoint));
        assert!(ours.conflicts_with(&overlapping_on_the_second));
        // An insert that says nothing about one of our key columns could have
        // put anything in it.
        assert!(ours.conflicts_with(&silent_on_the_second));
    }

    /// A key does not have to arrive in a new row. Rewriting a key column in
    /// place -- a column rewrite, an overlay -- can put any value in it, and the
    /// writer asserts nothing about which, so the assertion cannot be checked.
    #[test]
    fn test_an_assertion_conflicts_with_an_in_place_write_to_its_key_column() {
        let ours = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[1]),
        ]);
        let rewrite = |field: i32| {
            footprint(vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![FieldRef::Committed(field)],
                data_change: true,
            })])
        };

        assert!(ours.conflicts_with(&rewrite(0)));
        assert!(rewrite(0).conflicts_with(&ours));
        assert!(!ours.conflicts_with(&rewrite(1)));
    }

    /// An overlay writes only some cells of a key column, but those cells can
    /// hold any key, just as a full rewrite's can.
    #[test]
    fn test_an_assertion_conflicts_with_an_overlay_of_its_key_column() {
        let ours = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[1]),
        ]);
        let overlay = |field: i32| {
            footprint(vec![Action::AddOverlays(AddOverlays {
                fragment: Ref::Committed(0),
                overlays: vec![DataOverlayFile {
                    data_file: DataFile::new(
                        "overlay.lance",
                        vec![field],
                        vec![0],
                        ConcreteFileVersion::V2_0,
                        None,
                        None,
                    ),
                    coverage: OverlayCoverage::Shared(Arc::new(RoaringBitmap::from([0]))),
                    committed_version: 0,
                }],
                data_change: true,
            })])
        };

        assert!(ours.conflicts_with(&overlay(0)));
        assert!(overlay(0).conflicts_with(&ours));
        assert!(!ours.conflicts_with(&overlay(1)));
        assert!(!overlay(1).conflicts_with(&ours));
    }

    /// Bloom filters built for different sizes or error rates cannot be
    /// intersected, so neither can clear the other's keys.
    #[test]
    fn test_assertions_with_incomparable_filters_conflict() {
        let bloom = |number_of_items: u64| {
            Action::AssertUniqueKeys(AssertUniqueKeys {
                key_fields: vec![FieldRef::Committed(0)],
                filter: KeyExistenceFilter {
                    field_ids: Vec::new(),
                    filter: FilterType::Bloom {
                        bitmap: vec![0; 32],
                        num_bits: 256,
                        number_of_items,
                        probability: 0.01,
                    },
                },
            })
        };
        let ours = footprint(vec![append(0), bloom(100)]);
        let theirs = footprint(vec![append(0), bloom(200)]);

        assert!(ours.conflicts_with(&theirs));
        assert!(theirs.conflicts_with(&ours));
    }

    #[test]
    fn test_an_assertion_conflicts_with_an_unqualified_insert() {
        let ours = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[1]),
        ]);
        let plain_append = footprint(vec![append(0)]);

        assert!(ours.conflicts_with(&plain_append));
        assert!(plain_append.conflicts_with(&ours));
    }

    #[test]
    fn test_two_plain_appends_still_do_not_conflict() {
        assert!(!footprint(vec![append(0)]).conflicts_with(&footprint(vec![append(0)])));
    }

    #[test]
    fn test_an_assertion_ignores_a_concurrent_change_that_inserts_no_rows() {
        let ours = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[1]),
        ]);
        let removal = footprint(vec![Action::RemoveFragment(RemoveFragment {
            fragment: Ref::Committed(3),
            data_change: true,
        })]);

        assert!(!ours.conflicts_with(&removal));
    }

    #[test]
    fn test_a_compaction_is_not_an_insert() {
        // Compaction rewrites rows that are already there, so it cannot have
        // introduced a key the assertion would have to rule out.
        let ours = footprint(vec![
            append(0),
            assertion(vec![FieldRef::Committed(0)], &[1]),
        ]);
        let compaction = footprint(vec![compact(0)]);

        assert!(!ours.conflicts_with(&compaction));
    }
}
