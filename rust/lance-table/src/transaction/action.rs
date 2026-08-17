// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The action vocabulary of Transaction V2.
//!
//! Where an [`Operation`](super::Operation) names one whole change and carries a
//! post-image of the parts of the manifest it touches, a [`UserOperation`] is an
//! ordered list of [`Action`]s, each recording a single *delta*. Composing
//! several changes into one atomic commit and replaying a change onto a
//! different version both fall out of that, with no per-operation logic.
//!
//! The wire format and the reasoning behind it live in
//! `protos/transaction/actions.proto`; the two definitions must stay in step.
//! Only the subset of the drafted vocabulary that is implemented appears here --
//! an action this build does not know is rejected on load rather than skipped.
//!
//! Each action lives in its own module and owns everything about itself: its
//! definition, how it is applied, which coordinates it writes, and its wire
//! encoding. This module holds the shared vocabulary and dispatches to them.
//!
//! ```text
//! action                the vocabulary and the dispatch (this module)
//! action::<an_action>   one action, end to end
//! action::apply         the working state an action set is applied against
//! action::footprint     comparing two action sets for conflicts
//! action::proto         the envelope around the per-action encodings
//! action::translate     lowering a named operation into actions
//! ```
//!
//! # Stability
//!
//! Transaction V2 is a pre-vote draft. Nothing in this module is a compatibility
//! contract, and a transaction carrying a [`UserOperation`] is rejected outright
//! by libraries that predate it.

mod add_base;
mod add_data_file;
mod add_field;
mod add_fragment;
mod alter_field;
mod apply;
mod config_update;
mod drop_field;
mod footprint;
mod proto;
mod remove_fragment;
mod reserve_fragment_ids;
mod reset_table;
mod set_deletion_file;
mod tombstone_field_data;
mod translate;

#[cfg(test)]
mod test_support;

pub use add_base::AddBase;
pub use add_data_file::AddDataFile;
pub use add_field::AddField;
pub use add_fragment::AddFragment;
pub use alter_field::AlterField;
pub use config_update::{ConfigUpdate, FieldMetadataUpdate};
pub use drop_field::DropField;
pub use footprint::{ConfigMap, Coordinate, Footprint};
pub use remove_fragment::RemoveFragment;
pub use reserve_fragment_ids::ReserveFragmentIds;
pub use reset_table::ResetTable;
pub use set_deletion_file::SetDeletionFile;
pub use tombstone_field_data::TombstoneFieldData;

use apply::ApplyState;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;

/// A reference to a counter-allocated identifier -- a field id, fragment id, or
/// base id -- that may not have been assigned yet.
///
/// [`Ref::Committed`] is a concrete id that already exists in the manifest.
/// [`Ref::Local`] is a placeholder token minted by an `Add*` action earlier in
/// the same [`UserOperation`]; it resolves to a freshly-allocated id at apply,
/// and re-resolves against the target's counters when the operation is replayed
/// onto a newer version. That re-resolution is what lets two independent
/// `AddField`s on divergent branches become two distinct fields rather than a
/// collision.
///
/// Local tokens are scoped to one [`UserOperation`] and must be distinct within
/// it. The three id spaces do not share a token namespace: a fragment token 0
/// and a field token 0 are unrelated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, DeepSizeOf)]
pub enum Ref {
    Committed(u64),
    Local(u32),
}

impl Ref {
    /// The committed id, or `None` if this is an unresolved local token.
    pub fn committed(&self) -> Option<u64> {
        match self {
            Self::Committed(id) => Some(*id),
            Self::Local(_) => None,
        }
    }

    /// The local token, or `None` if this reference is already committed.
    pub fn local(&self) -> Option<u32> {
        match self {
            Self::Local(token) => Some(*token),
            Self::Committed(_) => None,
        }
    }
}

/// A composable transaction: an ordered list of user actions that commit
/// atomically as a single manifest change.
///
/// The `uuid` and `read_version` carried on the wire mirror the enclosing
/// [`Transaction`](super::Transaction) and are filled in from it, so they are
/// not repeated here.
#[derive(Debug, Clone, PartialEq, DeepSizeOf, Default)]
pub struct UserOperation {
    /// Human-readable description of the whole operation, e.g. `"INSERT INTO t"`.
    pub description: String,
    /// The ordered steps this operation applies.
    pub actions: Vec<UserAction>,
}

impl UserOperation {
    pub fn new(description: impl Into<String>, actions: Vec<UserAction>) -> Self {
        Self {
            description: description.into(),
            actions,
        }
    }

    /// Every action in every step, in application order.
    pub fn iter_actions(&self) -> impl Iterator<Item = &Action> {
        self.actions.iter().flat_map(|step| step.actions.iter())
    }
}

/// A single user-recognizable step within a [`UserOperation`], e.g. "append
/// batch" or "rebuild index".
///
/// The description keeps transaction history readable: when a range of versions
/// is squashed, each original operation collapses into one step, so the sequence
/// a user performed survives even though the deltas are flattened when applied.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UserAction {
    pub description: String,
    pub actions: Vec<Action>,
}

impl UserAction {
    pub fn new(description: impl Into<String>, actions: Vec<Action>) -> Self {
        Self {
            description: description.into(),
            actions,
        }
    }
}

/// A single granular change to the manifest.
///
/// The drafted vocabulary is larger than this; the variants here are the ones
/// this build implements end to end. Each one is defined, applied, and encoded
/// in the module named after it.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub enum Action {
    AddFragment(AddFragment),
    AddDataFile(AddDataFile),
    AddField(AddField),
    AddBase(AddBase),
    TombstoneFieldData(TombstoneFieldData),
    RemoveFragment(RemoveFragment),
    SetDeletionFile(SetDeletionFile),
    AlterField(AlterField),
    DropField(DropField),
    ReserveFragmentIds(ReserveFragmentIds),
    ResetTable(ResetTable),
    ConfigUpdate(ConfigUpdate),
}

impl Action {
    pub fn name(&self) -> &'static str {
        match self {
            Self::AddFragment(_) => "AddFragment",
            Self::AddDataFile(_) => "AddDataFile",
            Self::AddField(_) => "AddField",
            Self::AddBase(_) => "AddBase",
            Self::TombstoneFieldData(_) => "TombstoneFieldData",
            Self::RemoveFragment(_) => "RemoveFragment",
            Self::SetDeletionFile(_) => "SetDeletionFile",
            Self::AlterField(_) => "AlterField",
            Self::DropField(_) => "DropField",
            Self::ReserveFragmentIds(_) => "ReserveFragmentIds",
            Self::ResetTable(_) => "ResetTable",
            Self::ConfigUpdate(_) => "ConfigUpdate",
        }
    }

    /// Whether this action changes the data a reader would see, as opposed to
    /// rearranging how it is stored (compaction, a segment rebuild).
    ///
    /// CDC and streaming consumers use this to skip commits that cannot have
    /// changed any row's value.
    pub fn is_data_change(&self) -> bool {
        match self {
            Self::AddFragment(action) => action.data_change,
            Self::AddDataFile(action) => action.data_change,
            Self::TombstoneFieldData(action) => action.data_change,
            Self::RemoveFragment(action) => action.data_change,
            Self::SetDeletionFile(action) => action.data_change,
            // Dropping a field discards the values it held.
            Self::DropField(_) => true,
            // Other schema and base-path changes touch no row values.
            // Emptying the table discards every row it held.
            Self::ResetTable(_) => true,
            // Reserving ids writes no rows either.
            Self::AddField(_)
            | Self::AddBase(_)
            | Self::AlterField(_)
            | Self::ReserveFragmentIds(_)
            | Self::ConfigUpdate(_) => false,
        }
    }

    /// Fold this action into the state the next manifest is built from.
    fn apply(&self, state: &mut ApplyState) -> Result<()> {
        match self {
            Self::AddFragment(action) => action.apply(state),
            Self::AddDataFile(action) => action.apply(state),
            Self::AddField(action) => action.apply(state),
            Self::AddBase(action) => action.apply(state),
            Self::TombstoneFieldData(action) => action.apply(state),
            Self::RemoveFragment(action) => action.apply(state),
            Self::SetDeletionFile(action) => action.apply(state),
            Self::AlterField(action) => action.apply(state),
            Self::DropField(action) => action.apply(state),
            Self::ReserveFragmentIds(action) => action.apply(state),
            Self::ResetTable(action) => action.apply(state),
            Self::ConfigUpdate(action) => action.apply(state),
        }
    }

    /// Record the coordinates this action writes.
    fn footprint(&self, footprint: &mut Footprint) {
        match self {
            Self::AddFragment(action) => action.footprint(footprint),
            Self::AddDataFile(action) => action.footprint(footprint),
            Self::AddField(action) => action.footprint(footprint),
            Self::AddBase(action) => action.footprint(footprint),
            Self::TombstoneFieldData(action) => action.footprint(footprint),
            Self::RemoveFragment(action) => action.footprint(footprint),
            Self::SetDeletionFile(action) => action.footprint(footprint),
            Self::AlterField(action) => action.footprint(footprint),
            Self::DropField(action) => action.footprint(footprint),
            Self::ReserveFragmentIds(action) => action.footprint(footprint),
            Self::ResetTable(action) => action.footprint(footprint),
            Self::ConfigUpdate(action) => action.footprint(footprint),
        }
    }
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_accessors() {
        assert_eq!(Ref::Committed(7).committed(), Some(7));
        assert_eq!(Ref::Committed(7).local(), None);
        assert_eq!(Ref::Local(2).local(), Some(2));
        assert_eq!(Ref::Local(2).committed(), None);
    }

    #[test]
    fn test_data_change_defaults_by_action_kind() {
        let alter = Action::AlterField(AlterField {
            field: 1,
            name: Some("renamed".into()),
            ..Default::default()
        });
        assert!(!alter.is_data_change(), "a rename changes no row values");

        let remove = Action::RemoveFragment(RemoveFragment {
            fragment: Ref::Committed(3),
            data_change: true,
        });
        assert!(remove.is_data_change());
    }

    #[test]
    fn test_iter_actions_flattens_steps_in_order() {
        let operation = UserOperation::new(
            "two steps",
            vec![
                UserAction::new(
                    "first",
                    vec![Action::RemoveFragment(RemoveFragment {
                        fragment: Ref::Committed(1),
                        data_change: true,
                    })],
                ),
                UserAction::new(
                    "second",
                    vec![Action::RemoveFragment(RemoveFragment {
                        fragment: Ref::Committed(2),
                        data_change: true,
                    })],
                ),
            ],
        );
        let fragments = operation
            .iter_actions()
            .map(|action| match action {
                Action::RemoveFragment(remove) => remove.fragment,
                other => panic!("unexpected action {other}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(fragments, vec![Ref::Committed(1), Ref::Committed(2)]);
    }
}
