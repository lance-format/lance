// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Assert that no concurrent commit inserted a colliding key.

use super::apply::ApplyState;
use super::assert_cells_unchanged::check_assertable;
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
    key_fields: Vec<FieldRef>,
    filter: KeyExistenceFilter,
}

impl AssertUniqueKeys {
    /// Assert that the keys hashed in `filter`, over the key columns
    /// `key_fields` in order, are new.
    ///
    /// `key_fields` is the only key list: the field ids the filter type carries
    /// are an artifact of its use by the legacy operation, ignored here and
    /// cleared on encode.
    ///
    /// # Contract
    ///
    /// The caller certifies, and nothing downstream checks, that `filter` holds
    /// the hash of the key of every row the operation inserts, over
    /// `key_fields`. A row left out can collide with a concurrent insert
    /// unnoticed, leaving the table with a duplicate key.
    ///
    /// # Errors
    ///
    /// Rejects an empty `key_fields`.
    pub fn try_new_unchecked(
        key_fields: Vec<FieldRef>,
        filter: KeyExistenceFilter,
    ) -> Result<Self> {
        if key_fields.is_empty() {
            return Err(Error::invalid_input(
                "AssertUniqueKeys names no key column, so there is nothing for it to assert",
            ));
        }
        Ok(Self { key_fields, filter })
    }

    /// The key columns, in order.
    pub fn key_fields(&self) -> &[FieldRef] {
        &self.key_fields
    }

    /// Hashes of the key values of every row the operation inserts.
    pub fn filter(&self) -> &KeyExistenceFilter {
        &self.filter
    }

    /// Nothing to apply -- the assertion is checked when two operations are
    /// compared, not when one is folded into a manifest. The key columns are
    /// still resolved and looked up, so an assertion naming a field that is not
    /// there fails at the commit that carries it rather than silently guarding
    /// nothing.
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        for key_field in &self.key_fields {
            let field_id = state.resolve_schema_field(*key_field, "AssertUniqueKeys")?;
            let field = state.schema().field_by_id(field_id).ok_or_else(|| {
                Error::invalid_input(format!(
                    "AssertUniqueKeys names key field {field_id}, which is not in the schema"
                ))
            })?;
            check_assertable(field, "AssertUniqueKeys")?;
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
        // A committed transaction file carries the certification of the
        // writer that staged it.
        Self::try_new_unchecked(key_fields, KeyExistenceFilter::try_from(&filter)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::format::key_existence::FilterType;

    use crate::transaction::action::Action;
    use crate::transaction::action::test_support::{apply, backed_manifest};

    use std::sync::Arc;

    fn assertion(key_fields: Vec<FieldRef>, hashes: &[u64]) -> Action {
        Action::AssertUniqueKeys(
            AssertUniqueKeys::try_new_unchecked(
                key_fields,
                KeyExistenceFilter {
                    field_ids: Vec::new(),
                    filter: FilterType::ExactSet(hashes.iter().copied().collect()),
                },
            )
            .unwrap(),
        )
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
        let error = AssertUniqueKeys::try_new_unchecked(
            vec![],
            KeyExistenceFilter {
                field_ids: Vec::new(),
                filter: FilterType::ExactSet([1].into_iter().collect()),
            },
        )
        .unwrap_err();

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

        // Field 9 is beyond anything the read version handed out.
        assert!(
            error.to_string().contains("field 9 by a Committed ref"),
            "unexpected error: {error}"
        );
    }

    /// Conflict detection matches key fields to written field ids exactly, but
    /// a write to a nested value is recorded under its descendants' ids -- a
    /// list overlay under its item's -- so only a top-level leaf column (or a
    /// packed struct) can be a key. Fields: `id` (0), unpacked struct `s` (1)
    /// with child `x` (2), list `tags` (3) with item (4).
    #[rstest::rstest]
    #[case::a_top_level_column(0, None)]
    #[case::an_unpacked_struct(
        1,
        Some("names field 1 ('s'), which is nested or holds nested fields")
    )]
    #[case::a_struct_child(2, Some("names field 2 ('x'), which is nested or holds nested fields"))]
    #[case::a_list(
        3,
        Some("names field 3 ('tags'), which is nested or holds nested fields")
    )]
    #[case::a_list_item(
        4,
        Some("names field 4 ('item'), which is nested or holds nested fields")
    )]
    fn test_asserting_over_a_nested_key_is_rejected(
        #[case] key_field: i32,
        #[case] expected_error: Option<&str>,
    ) {
        use arrow_schema::{DataType, Field as ArrowField, Fields};
        let mut manifest = backed_manifest();
        manifest.schema =
            lance_core::datatypes::Schema::try_from(&arrow_schema::Schema::new(vec![
                ArrowField::new("id", DataType::Int32, false),
                ArrowField::new(
                    "s",
                    DataType::Struct(Fields::from(vec![ArrowField::new(
                        "x",
                        DataType::Int32,
                        true,
                    )])),
                    true,
                ),
                ArrowField::new(
                    "tags",
                    DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
                    true,
                ),
            ]))
            .unwrap();

        let result = apply(
            &manifest,
            vec![assertion(vec![FieldRef::Committed(key_field)], &[99])],
        );

        match expected_error {
            None => {
                result.unwrap();
            }
            Some(message) => {
                let error = result.unwrap_err();
                assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
                assert!(
                    error.to_string().contains(message),
                    "unexpected error: {error}"
                );
            }
        }
    }
}
