// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Assert that no concurrent commit inserted a colliding key.

use super::apply::ApplyState;
use super::proto::required;
use super::{Footprint, Ref};
use crate::format::key_existence::KeyExistenceFilter;
use crate::format::pb;
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
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AssertUniqueKeys {
    /// The key columns, in order. This is the authoritative list; the field ids
    /// the filter carries are an artifact of the shared filter type and are left
    /// empty here.
    pub key_fields: Vec<Ref>,
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
            let field_id = state.resolve_field(*key_field)?;
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

impl From<&AssertUniqueKeys> for pb::AssertUniqueKeys {
    fn from(value: &AssertUniqueKeys) -> Self {
        let mut filter = pb::KeyExistenceFilter::from(&value.filter);
        // `key_fields` is the authoritative list; the filter's own copy is an
        // artifact of the type it shares with the legacy Update operation.
        filter.field_ids.clear();
        Self {
            key_fields: value.key_fields.iter().map(|&f| f.into()).collect(),
            filter: Some(filter),
        }
    }
}

impl TryFrom<pb::AssertUniqueKeys> for AssertUniqueKeys {
    type Error = Error;

    fn try_from(message: pb::AssertUniqueKeys) -> Result<Self> {
        let filter = required(message.filter, "AssertUniqueKeys.filter")?;
        Ok(Self {
            key_fields: message
                .key_fields
                .into_iter()
                .map(Ref::try_from)
                .collect::<Result<Vec<_>>>()?,
            filter: KeyExistenceFilter::try_from(&filter)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::key_existence::FilterType;
    use crate::transaction::action::test_support::{apply, backed_manifest};
    use crate::transaction::action::{
        Action, AddFragment, CompositeOperation, RemoveFragment, UserAction,
    };

    fn assertion(key_fields: Vec<Ref>, hashes: &[u64]) -> Action {
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
            local,
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

    fn footprint(actions: Vec<Action>) -> Footprint {
        Footprint::from(&CompositeOperation::new(vec![UserAction::new(
            "step", actions,
        )]))
    }

    #[test]
    fn test_an_assertion_changes_nothing() {
        let manifest = backed_manifest();
        let out = apply(
            &manifest,
            vec![assertion(vec![Ref::Committed(0)], &[1, 2, 3])],
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
            vec![assertion(vec![Ref::Committed(9)], &[1])],
        )
        .unwrap_err();

        assert!(
            error.to_string().contains("not in the schema"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_inserts_with_disjoint_keys_do_not_conflict() {
        let ours = footprint(vec![append(0), assertion(vec![Ref::Committed(0)], &[1, 2])]);
        let theirs = footprint(vec![append(0), assertion(vec![Ref::Committed(0)], &[3, 4])]);

        assert!(!ours.conflicts_with(&theirs));
        assert!(!theirs.conflicts_with(&ours));
    }

    #[test]
    fn test_inserts_sharing_a_key_conflict() {
        let ours = footprint(vec![append(0), assertion(vec![Ref::Committed(0)], &[1, 2])]);
        let theirs = footprint(vec![append(0), assertion(vec![Ref::Committed(0)], &[2, 3])]);

        assert!(ours.conflicts_with(&theirs));
        assert!(theirs.conflicts_with(&ours));
    }

    #[test]
    fn test_assertions_over_different_key_columns_conflict() {
        // Two filters over different columns hash different values, so neither
        // says anything about the other's keys.
        let ours = footprint(vec![append(0), assertion(vec![Ref::Committed(0)], &[1])]);
        let theirs = footprint(vec![append(0), assertion(vec![Ref::Committed(1)], &[2])]);

        assert!(ours.conflicts_with(&theirs));
    }

    #[test]
    fn test_an_assertion_conflicts_with_an_unqualified_insert() {
        let ours = footprint(vec![append(0), assertion(vec![Ref::Committed(0)], &[1])]);
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
        let ours = footprint(vec![append(0), assertion(vec![Ref::Committed(0)], &[1])]);
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
        let ours = footprint(vec![append(0), assertion(vec![Ref::Committed(0)], &[1])]);
        let compaction = footprint(vec![compact(0)]);

        assert!(!ours.conflicts_with(&compaction));
    }
}
