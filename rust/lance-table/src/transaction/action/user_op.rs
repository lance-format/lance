// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! User-operation transaction types.
//!
//! Extracted from `action.rs` to keep that module focused on the `Action`
//! trait and dispatch. Items here are re-exported from `action.rs`, so the
//! public surface of `crate::transaction::action::*` is unchanged.

use deepsize::DeepSizeOf;

use super::Action;

/// A description plus a list of actions (`Box<dyn Action>`).
///
/// Per Weston's suggestion in the design discussion, this middle layer
/// preserves human-readable intent ("append batch", "rebuild index") for
/// compound transactions without splitting them into separate commits.
/// When applying to the manifest, the [`UserAction`] lists are flattened.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UserAction {
    /// Human-readable description. Maps to (for example) `Delete.predicate`
    /// when translating from old [`Operation`] variants.
    pub description: Option<String>,
    /// The actions that make up this user action, applied in order.
    pub actions: Vec<Box<dyn Action>>,
}

impl UserAction {
    /// Create a [`UserAction`] from a list of actions, with no description.
    pub fn new(actions: Vec<Box<dyn Action>>) -> Self {
        Self {
            description: None,
            actions,
        }
    }

    /// Attach a human-readable description to this action.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// A normal composable operation.
///
/// `UserOperation` is one of the three top-level transaction kinds (see
/// [`TransactionKind`]); the others are [`TransactionKind::Restore`] and
/// [`TransactionKind::Clone`], which cannot be expressed as actions.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UserOperation {
    /// Unique identifier for this transaction.
    pub uuid: String,
    /// The dataset version this transaction was built against.
    pub read_version: u64,
    /// The grouped user actions; flattened to a single action list on apply.
    pub user_actions: Vec<UserAction>,
}

impl UserOperation {
    /// Iterate over the flattened action list, ignoring the
    /// [`UserAction`] grouping.
    pub fn actions(&self) -> impl Iterator<Item = &Box<dyn Action>> {
        self.user_actions.iter().flat_map(|ua| ua.actions.iter())
    }

    /// Mutable counterpart of [`actions`](Self::actions) — flat over the
    /// nested [`UserAction`] groups. Used by post-translation enrichment
    /// (PR B's action resolver) to fill late payload fields that cannot be
    /// populated during sync translation.
    pub fn actions_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Action>> {
        self.user_actions
            .iter_mut()
            .flat_map(|ua| ua.actions.iter_mut())
    }
}

/// Top-level transaction kind. Mirrors the protobuf `oneof` the new format
/// will introduce in Phase 2.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum TransactionKind {
    UserOperation(UserOperation),
    /// Full manifest restore. Cannot be expressed as a list of actions, so it
    /// stays as a distinct top-level variant.
    Restore {
        version: u64,
    },
    /// Dataset clone. Cannot be expressed as a list of actions for the same
    /// reason as `Restore`; semantics live at the dataset level, not the
    /// manifest level.
    Clone {
        is_shallow: bool,
        ref_name: Option<String>,
        ref_version: u64,
        ref_path: String,
        branch_name: Option<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::super::*;
    use super::*;

    #[test]
    fn user_operation_flattens_actions() {
        let user_op = UserOperation {
            uuid: "test".into(),
            read_version: 0,
            user_actions: vec![
                UserAction::new(vec![Box::new(AddFragments {
                    fragments: vec![sample_fragment(0)],
                    inserted_rows_filter: None,
                })])
                .with_description("first batch"),
                UserAction::new(vec![Box::new(AddFragments {
                    fragments: vec![sample_fragment(0), sample_fragment(0)],
                    inserted_rows_filter: None,
                })])
                .with_description("second batch"),
            ],
        };

        assert_eq!(user_op.actions().count(), 2);
    }
}
