// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The action vocabulary of Transaction V2.
//!
//! Where an [`Operation`](super::Operation) names one whole change and carries a
//! post-image of the parts of the manifest it touches, a [`CompositeOperation`] is an
//! ordered list of [`Action`]s, each recording a single *delta*. Composing
//! several changes into one atomic commit and replaying a change onto a
//! different version both fall out of that, with no per-operation logic.
//!
//! The wire format and the reasoning behind it live in
//! `protos/transaction/actions.proto`; the two definitions must stay in step.
//! Only the subset of the specified vocabulary that is implemented appears here --
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
//! ```
//!
//! # Stability
//!
//! Transaction V2 is an experimental format feature: breaking changes may be
//! made to it without a separate vote, and it is removed outright if its
//! stabilization vote does not pass. Nothing in this module is a
//! compatibility contract.
//!
//! Committing a [`CompositeOperation`] is a durable compatibility break, not
//! merely a feature an older reader ignores: a table version carrying one
//! cannot be *opened* by Lance before v12.0.0. Those releases decode the
//! manifest's inline transaction while opening the table, and an operation
//! they do not know decodes to no operation at all, which fails the open. The
//! fix shipped in v12.0.0 and is not expected to be backported, so writing one
//! raises that table's minimum reader version. See
//! <https://github.com/lance-format/lance/issues/9454>.

/// The action vocabulary, as one list.
///
/// Every per-variant `match` over an [`Action`] -- the enum itself, its
/// forwarding methods, and both directions of its wire encoding -- is generated
/// from this, so an action is added by writing its module and adding one name
/// here. A module that does not supply the full set of methods fails to
/// compile.
///
/// The variant name doubles as the protobuf `oneof` variant name and as the
/// action's name in errors and logs, so the three cannot drift apart.
///
/// This is defined ahead of the modules below so it is in scope for all of
/// them; `macro_rules!` visibility runs from the definition to the end of the
/// enclosing module, children included.
macro_rules! for_each_action {
    ($emit:ident) => {
        $emit! {
            AddFragment,
            AddDataFile,
            AddField,
            AddBase,
            TombstoneFieldData,
            RemoveFragment,
            SetDeletionFile,
            AlterField,
            DropField,
            ReserveFragmentIds,
            ReserveRowIds,
            ResetTable,
            ConfigUpdate,
        }
    };
}

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
mod reserve_row_ids;
mod reset_table;
mod set_deletion_file;
mod tombstone_field_data;

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
pub use reserve_row_ids::ReserveRowIds;
pub use reset_table::ResetTable;
pub use set_deletion_file::SetDeletionFile;
pub use tombstone_field_data::TombstoneFieldData;

use apply::ApplyState;
use lance_core::deepsize::DeepSizeOf;
use lance_core::{
    ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID, ROW_LAST_UPDATED_AT_VERSION_FIELD_ID, Result,
};

/// A reference to a counter-allocated fragment id or base id that may not have
/// been assigned yet. Field ids use [`FieldRef`].
///
/// [`Ref::Committed`] is a concrete id that already exists in the manifest.
/// [`Ref::Local`] is a placeholder token minted by an `Add*` action earlier in
/// the same [`CompositeOperation`]; it resolves to a freshly-allocated id at apply,
/// and re-resolves against the target's counters when the operation is replayed
/// onto a newer version. That re-resolution is what lets two independent
/// `AddField`s on divergent branches become two distinct fields rather than a
/// collision.
///
/// A committed id must exist at the operation's read version: apply rejects
/// one above that version's high-water mark, since a concurrent commit may
/// have minted it since. An object minted in the same operation is named by
/// its local token.
///
/// Local tokens are scoped to one [`CompositeOperation`] and must be distinct within
/// it. The id spaces (fragment, field, base) do not share a token namespace: a
/// fragment token 0 and a field token 0 are unrelated.
///
/// ```
/// # use lance_table::transaction::action::Ref;
/// // A fragment the dataset already has.
/// let existing = Ref::Committed(3);
/// assert_eq!(existing.committed(), Some(3));
///
/// // A fragment this operation is about to mint. It has no id until apply
/// // resolves it against whichever version the commit lands on.
/// let minted = Ref::Local(0);
/// assert_eq!(minted.committed(), None);
/// ```
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

/// A reference to a field id that may not have been assigned yet: the field-id
/// form of [`Ref`].
///
/// A separate type because field ids are signed. Schema fields are `>= 0`, and
/// [`AddDataFile::field_ids`] may also bind the reserved ids of the hidden row
/// lineage columns ([`ROW_ID_FIELD_ID`](lance_core::ROW_ID_FIELD_ID) and its
/// two siblings), the way a data file's own field list does. Every other field
/// reference takes a schema field id; apply rejects anything else.
///
/// ```
/// # use lance_table::transaction::action::FieldRef;
/// let existing = FieldRef::Committed(3);
/// assert_eq!(existing.committed(), Some(3));
///
/// // A field minted by an AddField with `local: 0` earlier in the operation.
/// let minted = FieldRef::Local(0);
/// assert_eq!(minted.local(), Some(0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, DeepSizeOf)]
pub enum FieldRef {
    Committed(i32),
    Local(u32),
}

impl FieldRef {
    /// The committed id, or `None` if this is an unresolved local token.
    pub fn committed(&self) -> Option<i32> {
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

/// A field id as apply resolved it: a schema field, or one of the reserved
/// negative ids a data file's field list may carry.
///
/// Scoped to the action layer, where a [`FieldRef`] can name either kind and
/// each action has to say which it accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct FieldId(i32);

impl FieldId {
    pub(super) fn get(self) -> i32 {
        self.0
    }

    /// A field of the dataset schema, as opposed to a reserved id.
    pub(super) fn is_schema(self) -> bool {
        self.0 >= 0
    }

    /// One of the hidden row lineage columns a spilled sequence lives in.
    pub(super) fn is_row_lineage(self) -> bool {
        matches!(
            self.0,
            ROW_ID_FIELD_ID
                | ROW_CREATED_AT_VERSION_FIELD_ID
                | ROW_LAST_UPDATED_AT_VERSION_FIELD_ID
        )
    }
}

impl std::fmt::Display for FieldId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A composable transaction: an ordered list of user steps that commit
/// atomically as a single manifest change.
///
/// The `uuid` and `read_version` carried on the wire mirror the enclosing
/// [`Transaction`](super::Transaction) and are filled in from it, so they are
/// not repeated here.
///
/// The steps are applied in order, and a step may name what an earlier one
/// minted: below, the data file is attached to the fragment added in the step
/// before it, which has no committed id yet. Doing that as two named
/// operations would take two commits, and a reader could observe the empty
/// fragment in between.
///
/// ```
/// # use lance_file::version::ConcreteFileVersion;
/// # use lance_table::format::DataFile;
/// use lance_table::transaction::action::{
///     Action, AddDataFile, AddFragment, CompositeOperation, FieldRef, Ref, UserStep,
/// };
///
/// let operation = CompositeOperation::new(vec![
///     UserStep::new(
///         "add a fragment",
///         vec![Action::AddFragment(AddFragment {
///             id: Ref::Local(0),
///             physical_rows: 10,
///             row_id_meta: None,
///             last_updated_at_version_meta: None,
///             created_at_version_meta: None,
///             data_change: true,
///         })],
///     ),
///     UserStep::new(
///         "back its one column",
///         vec![Action::AddDataFile(AddDataFile {
///             fragment: Ref::Local(0),
///             file: DataFile::new_unstarted("data/new.lance", ConcreteFileVersion::V2_0),
///             field_ids: vec![FieldRef::Committed(0)],
///             data_change: true,
///             base: None,
///         })],
///     ),
/// ]);
///
/// // The steps are what a user did; the actions are the deltas they became.
/// assert_eq!(operation.steps.len(), 2);
/// assert_eq!(operation.iter_actions().count(), 2);
/// ```
///
/// Committing one goes through
/// `CommitBuilder::with_experimental_composite_operations`, which is the
/// caller's acknowledgement that this is an experimental format (see
/// [Stability](self#stability)).
#[derive(Debug, Clone, PartialEq, DeepSizeOf, Default)]
pub struct CompositeOperation {
    /// The ordered steps this operation applies.
    pub steps: Vec<UserStep>,
}

impl CompositeOperation {
    pub fn new(steps: Vec<UserStep>) -> Self {
        Self { steps }
    }

    /// Every action in every step, in application order.
    pub fn iter_actions(&self) -> impl Iterator<Item = &Action> {
        self.steps.iter().flat_map(|step| step.actions.iter())
    }
}

/// A single user-recognizable step within a [`CompositeOperation`], e.g. "append
/// batch" or "rebuild index".
///
/// The description keeps transaction history readable. The intent is that
/// squashing a range of versions concatenates their steps, so the sequence a
/// user performed survives even though the deltas are flattened when applied;
/// the exact squash behavior is up to the implementing system.
#[derive(Debug, Clone, PartialEq, DeepSizeOf)]
pub struct UserStep {
    pub description: String,
    pub actions: Vec<Action>,
}

impl UserStep {
    pub fn new(description: impl Into<String>, actions: Vec<Action>) -> Self {
        Self {
            description: description.into(),
            actions,
        }
    }
}

macro_rules! define_action {
    ($($variant:ident,)*) => {
        /// A single granular change to the manifest.
        ///
        /// The specified vocabulary is larger than this; the variants here are the
        /// ones this build implements end to end. Each one is defined, applied,
        /// and encoded in the module named after it, and appears here only
        /// because it is listed in `for_each_action!`.
        #[derive(Debug, Clone, PartialEq, DeepSizeOf)]
        pub enum Action {
            $($variant($variant),)*
        }

        impl Action {
            pub fn name(&self) -> &'static str {
                match self {
                    $(Self::$variant(_) => stringify!($variant),)*
                }
            }

            /// Whether this action changes the data a reader would see, as
            /// opposed to rearranging how it is stored (compaction, a segment
            /// rebuild).
            ///
            /// CDC and streaming consumers use this to skip commits that cannot
            /// have changed any row's value.
            pub fn is_data_change(&self) -> bool {
                match self {
                    $(Self::$variant(action) => action.is_data_change(),)*
                }
            }

            /// Fold this action into the state the next manifest is built from.
            fn apply(&self, state: &mut ApplyState) -> Result<()> {
                match self {
                    $(Self::$variant(action) => action.apply(state),)*
                }
            }

            /// Record the coordinates this action writes.
            fn footprint(&self, footprint: &mut Footprint) {
                match self {
                    $(Self::$variant(action) => action.footprint(footprint),)*
                }
            }
        }
    };
}

for_each_action!(define_action);

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
        assert_eq!(FieldRef::Committed(-3).committed(), Some(-3));
        assert_eq!(FieldRef::Local(2).local(), Some(2));
    }

    #[test]
    fn test_field_id_kinds() {
        assert!(FieldId(0).is_schema());
        assert!(!FieldId(0).is_row_lineage());
        for reserved in [
            ROW_ID_FIELD_ID,
            ROW_CREATED_AT_VERSION_FIELD_ID,
            ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
        ] {
            assert!(FieldId(reserved).is_row_lineage());
            assert!(!FieldId(reserved).is_schema());
        }
        for sentinel in [-1, -2, -6] {
            assert!(!FieldId(sentinel).is_schema());
            assert!(!FieldId(sentinel).is_row_lineage());
        }
    }

    #[test]
    fn test_data_change_defaults_by_action_kind() {
        let alter = Action::AlterField(AlterField {
            field: FieldRef::Committed(1),
            name: Some("renamed".into()),
            nullable: None,
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
        let operation = CompositeOperation::new(vec![
            UserStep::new(
                "first",
                vec![Action::RemoveFragment(RemoveFragment {
                    fragment: Ref::Committed(1),
                    data_change: true,
                })],
            ),
            UserStep::new(
                "second",
                vec![Action::RemoveFragment(RemoveFragment {
                    fragment: Ref::Committed(2),
                    data_change: true,
                })],
            ),
        ]);
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
