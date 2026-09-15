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

        let conflicting = state
            .bases()
            .find(|base| base.name == self.base.name || base.path == self.base.path);
        if let Some(conflicting) = conflicting {
            return Err(Error::invalid_input(format!(
                "Conflict detected: Base path with name '{:?}' or path '{}' already exists. \
                 Existing: name='{:?}', path='{}'",
                self.base.name, self.base.path, conflicting.name, conflicting.path
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

    /// The base id is minted, but the name and location are not: the manifest
    /// requires both to be unique, so two operations claiming either collide.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add(Coordinate::BaseName(self.base.name.clone()));
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

        let error = apply(
            &next,
            vec![Action::AddBase(AddBase {
                local: 0,
                base: BasePath::new(0, "s3://bucket/a".into(), Some("other".into()), false),
            })],
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("already exists"),
            "unexpected error: {error}"
        );
    }

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
            error.to_string().contains("already exists"),
            "unexpected error: {error}"
        );
    }
}
