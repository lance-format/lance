// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Mint a new base path.

use super::apply::ApplyState;
use super::proto::required;
use super::{Coordinate, Footprint};
use crate::format::{BasePath, pb};
use lance_core::deepsize::DeepSizeOf;
use lance_core::{Error, Result};

/// Mint a new base path.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct AddBase {
    /// Token standing in for the base id until it is allocated at apply.
    pub local: u32,
    /// The base path. Its `id` is ignored and stamped in at apply.
    pub base: BasePath,
}

impl AddBase {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let id = state.mint_base(self.local)?;

        // Two separate uniqueness rules, reported separately: a caller that
        // trips one needs to know which, and the two are fixed differently.
        if let Some(name) = &self.base.name {
            // Only a name that is actually set: the name is an optional alias,
            // and the format puts no uniqueness rule on its absence.
            if let Some(conflicting) = state.bases().find(|base| base.name.as_ref() == Some(name)) {
                return Err(Error::invalid_input(format!(
                    "a base path named {:?} already exists (id {}, at '{}'); base names are \
                     unique within a dataset",
                    name, conflicting.id, conflicting.path
                )));
            }
        }
        if let Some(conflicting) = state.bases().find(|base| base.path == self.base.path) {
            return Err(Error::invalid_input(format!(
                "a base path at '{}' already exists (id {}, named {:?}); base locations \
                 are unique within a dataset",
                self.base.path, conflicting.id, conflicting.name
            )));
        }

        let mut base = self.base.clone();
        base.id = id;
        state.push_base(base);
        Ok(())
    }

    /// A base path is a location, not data.
    pub(super) fn is_data_change(&self) -> bool {
        false
    }

    /// The base id is minted, but the name and location are not: both must be
    /// unique, so two operations claiming either collide. An unset name claims
    /// nothing -- it is an optional alias, not a name to share.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        if let Some(name) = &self.base.name {
            footprint.add(Coordinate::BaseName(name.clone()));
        }
        footprint.add(Coordinate::BaseLocation(self.base.path.clone()));
    }
}

impl From<&AddBase> for pb::AddBase {
    fn from(value: &AddBase) -> Self {
        Self {
            local: value.local,
            base: Some(pb::BasePath::from(value.base.clone())),
        }
    }
}

impl TryFrom<pb::AddBase> for AddBase {
    type Error = Error;

    fn try_from(message: pb::AddBase) -> Result<Self> {
        Ok(Self {
            local: message.local,
            base: BasePath::from(required(message.base, "AddBase.base")?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::action::Action;
    use crate::transaction::action::test_support::apply;
    use crate::transaction::test_support::sample_manifest;

    #[test]
    fn test_add_base_mints_an_id_and_rejects_duplicates() {
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![Action::AddBase(AddBase {
                local: 0,
                base: BasePath::new(0, "s3://bucket/a".into(), Some("a".into()), false),
            })],
        )
        .unwrap();
        assert_eq!(next.base_paths.len(), 1);
        assert_eq!(next.base_paths[&1].path, "s3://bucket/a");

        // A fresh name, but the location is taken.
        let error = apply(
            &next,
            vec![Action::AddBase(AddBase {
                local: 0,
                base: BasePath::new(0, "s3://bucket/a".into(), Some("other".into()), false),
            })],
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("base path at 's3://bucket/a' already exists"),
            "unexpected error: {error}"
        );
    }

    /// The name is an optional alias, so leaving it unset is not a name two
    /// bases can share. `Operation::UpdateBases` guards its name comparison
    /// with `is_some()` for the same reason.
    #[test]
    fn test_unnamed_bases_do_not_collide_with_each_other() {
        let manifest = sample_manifest();
        let next = apply(
            &manifest,
            vec![
                Action::AddBase(AddBase {
                    local: 0,
                    base: BasePath::new(0, "s3://bucket/a".into(), None, false),
                }),
                Action::AddBase(AddBase {
                    local: 1,
                    base: BasePath::new(0, "s3://bucket/b".into(), None, false),
                }),
            ],
        )
        .unwrap();
        assert_eq!(next.base_paths.len(), 2);
    }

    /// The second base takes a different id and a different location, but the
    /// same name, and the first is only visible to it because both land in one
    /// operation.
    #[test]
    fn test_two_add_bases_in_one_operation_see_each_other() {
        let manifest = sample_manifest();
        let error = apply(
            &manifest,
            vec![
                Action::AddBase(AddBase {
                    local: 0,
                    base: BasePath::new(0, "s3://bucket/a".into(), Some("a".into()), false),
                }),
                Action::AddBase(AddBase {
                    local: 1,
                    base: BasePath::new(0, "s3://bucket/b".into(), Some("a".into()), false),
                }),
            ],
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("base path named \"a\" already exists"),
            "unexpected error: {error}"
        );
    }
}
