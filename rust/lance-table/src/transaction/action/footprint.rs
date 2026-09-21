// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The set of coordinates an action set writes.
//!
//! Two concurrent action sets can both commit when neither writes anything the
//! other writes. This is a structural test over coordinates rather than a
//! matrix over operation pairs, so it stays a single rule as the vocabulary
//! grows -- adding an action means saying which coordinates it writes, not
//! extending an N-by-N table.
//!
//! Footprints are derived from the actions at conflict time and never
//! serialized. A writer cannot pin down what a reader considers a conflict, and
//! the rule can be tightened in a later release without a format change.
//!
//! Which coordinates an action writes is decided by that action, in its own
//! module. This module holds the coordinate space and the comparison.

use super::{CompositeOperation, Ref};
use crate::transaction::UpdateMap;
use std::collections::HashSet;
use std::sync::Arc;
use uuid::Uuid;

/// One thing an action set writes.
///
/// Only committed coordinates appear. A minted fragment, field, or base has no
/// id in the read version, so no concurrent writer can be naming the same
/// thing; relocation re-resolves it against whatever version wins.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Coordinate {
    /// Whether a committed fragment is still part of the dataset.
    FragmentExistence(u64),
    /// A committed fragment's deletion file.
    FragmentDeletions(u64),
    /// The data backing one field within one committed fragment.
    FieldData { fragment: u64, field: i32 },
    /// A field's definition in the schema.
    FieldDefinition(i32),
    /// A base path's name, where it has one. Names must be unique; an unset
    /// name is an absent alias rather than a name shared with every other
    /// unnamed base, so it coordinates nothing.
    BaseName(String),
    /// A base path's location, which the manifest requires to be unique.
    BaseLocation(String),
    /// One key in one of the manifest's string maps.
    ConfigEntry { map: ConfigMap, key: String },
    /// One index segment, by uuid.
    IndexSegment(Uuid),
}

/// One of the string maps a manifest carries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConfigMap {
    /// Dataset config.
    Config,
    /// Table metadata.
    TableMetadata,
    /// Schema-level metadata.
    SchemaMetadata,
    /// One field's metadata, by field id.
    Field(i32),
}

impl Coordinate {
    /// The fragment this coordinate lives in, if it is fragment-scoped.
    fn fragment(&self) -> Option<u64> {
        match self {
            Self::FragmentExistence(id) | Self::FragmentDeletions(id) => Some(*id),
            Self::FieldData { fragment, .. } => Some(*fragment),
            Self::FieldDefinition(_)
            | Self::BaseName(_)
            | Self::BaseLocation(_)
            | Self::ConfigEntry { .. }
            | Self::IndexSegment(_) => None,
        }
    }

    /// The field this coordinate belongs to, if it is field-scoped.
    fn field(&self) -> Option<i32> {
        match self {
            Self::FieldData { field, .. } => Some(*field),
            Self::FieldDefinition(id) => Some(*id),
            // A field's metadata goes with the field, so dropping the field
            // writes over a concurrent update to its metadata.
            Self::ConfigEntry {
                map: ConfigMap::Field(id),
                ..
            } => Some(*id),
            Self::FragmentExistence(_)
            | Self::FragmentDeletions(_)
            | Self::BaseName(_)
            | Self::BaseLocation(_)
            | Self::ConfigEntry { .. }
            | Self::IndexSegment(_) => None,
        }
    }

    /// The string map this coordinate is a key in, if it is one.
    fn config_map(&self) -> Option<&ConfigMap> {
        match self {
            Self::ConfigEntry { map, .. } => Some(map),
            _ => None,
        }
    }
}

/// Everything an action set writes, in one set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Footprint {
    writes: HashSet<Coordinate>,
    /// Fragments this set removes outright. Removing a fragment writes every
    /// coordinate inside it, which cannot be enumerated, so it is tracked
    /// separately and matched against the other set by fragment id.
    removed_fragments: HashSet<u64>,
    /// Fields this set drops from the schema. Like a fragment removal, this
    /// writes every coordinate belonging to the field -- its definition and its
    /// data in every fragment -- so it is matched by field id.
    ///
    /// Only the named field, not its descendants: a footprint has no schema to
    /// expand a struct with. A concurrent write to a child of a dropped struct
    /// is therefore not caught here and fails when it is applied against the
    /// version where the child no longer exists.
    removed_fields: HashSet<i32>,
    /// Fields this set needs to still be in the schema, without writing their
    /// definitions. Writing data into a fragment this operation mints is the
    /// case this exists for: the cells are invisible to a concurrent writer, so
    /// they are not written coordinates, but the fields they hold data for have
    /// committed ids that a concurrent [`DropField`](super::DropField) can take
    /// away -- leaving the new fragment carrying data for a field the manifest
    /// no longer has.
    required_fields: HashSet<i32>,
    /// Coordinates this set reads and needs to still hold what it read, without
    /// writing them itself.
    ///
    /// Data written for a committed field is one case: the values are encoded
    /// in the type the schema names, so the definition has to still say what it
    /// said. An index segment is the other: it describes values it does not
    /// change, and the format gives a reader no way to notice that those values
    /// were replaced underneath it. Unlike a write, two sets may require the
    /// same coordinate -- two readers of one column do not collide.
    requires: HashSet<Coordinate>,
    /// String maps this set replaces outright rather than merging into. Like a
    /// fragment removal, this writes every key in the map, including keys it
    /// does not name, so it is matched by map rather than by key.
    replaced_maps: HashSet<ConfigMap>,
    /// Whether this set rewrites the table wholesale. Such a set writes every
    /// coordinate there is, including ones a concurrent set would only mint, so
    /// it is tracked as a flag rather than enumerated.
    exclusive: bool,
    /// What this set writes into logical indices. A segment's uuid is a
    /// coordinate, but the thing two writers can collide over is the index the
    /// segment joins, which is not a set of ids -- see [`IndexClaim`].
    index_claims: Vec<IndexClaim>,
}

/// What an action set writes into one logical index.
///
/// An index is the set of segments sharing a name, and the query path unions
/// them, so two writers may extend one index at the same time -- what they may
/// not do is describe the same rows twice, or disagree about what the index is.
#[derive(Debug, Clone, PartialEq)]
struct IndexClaim {
    name: String,
    /// What the set says the index is. `None` for an action that edits a
    /// segment without restating the index's definition.
    identity: Option<IndexIdentity>,
    /// The committed fragments this set brings under the index, or `None` when
    /// the reach is not stated -- what the system indices carry, and what makes
    /// a claim collide with every other claim on the same index.
    ///
    /// Fragments this operation mints are left out. They have no id in the read
    /// version, so a concurrent writer cannot be covering one.
    coverage: Option<HashSet<u64>>,
}

/// What an index is, for the purpose of deciding whether two writers are
/// building the same one.
///
/// `details` is compared as the opaque blob it is. Two segments of one index
/// built by the same writer serialize identical config, so equality is the
/// right test until index config is lifted out of the per-segment details.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct IndexIdentity {
    pub fields: Vec<Ref>,
    /// Which of `fields` are merely carried: two writers that disagree about
    /// this disagree about what the index answers for, not just what it holds.
    pub covering_fields: Vec<Ref>,
    pub details: Option<Arc<prost_types::Any>>,
    pub index_version: i32,
}

impl IndexClaim {
    fn conflicts_with(&self, other: &Self) -> bool {
        if self.name != other.name {
            return false;
        }
        if let (Some(ours), Some(theirs)) = (&self.identity, &other.identity)
            && ours != theirs
        {
            return true;
        }
        match (&self.coverage, &other.coverage) {
            (Some(ours), Some(theirs)) => !ours.is_disjoint(theirs),
            // An unstated reach could be any fragment, including one the other
            // side is claiming.
            _ => true,
        }
    }
}

impl Footprint {
    /// Whether this set can still commit on top of `committed`, a set that
    /// landed after this one read.
    ///
    /// Asymmetric, in the requirement check alone: what `committed` required
    /// held when it committed, and it serializes first, so nothing this set
    /// writes can retroactively break it. This set's requirements are the open
    /// question, because it read before `committed` landed.
    ///
    /// The distinction is not academic. An index segment requires the data it
    /// describes; a rewrite of that data writes it. Checking both directions
    /// would also reject the rewrite that arrives *after* a segment -- an order
    /// the system already handles, by pruning the stale fragment out of the
    /// segment's coverage as the rewrite applies.
    ///
    /// Everything else is a claim on the same coordinate from both sides and
    /// stays symmetric.
    pub fn conflicts_with(&self, committed: &Self) -> bool {
        // A wholesale rewrite leaves nothing for a concurrent set to land on --
        // not even an append, whose rows the reset would discard or resurrect
        // depending on which commit won.
        if self.exclusive || committed.exclusive {
            return true;
        }
        if !self.writes.is_disjoint(&committed.writes) {
            return true;
        }
        // Only this side's requirements. See the note on direction above.
        if !self.requires.is_disjoint(&committed.writes) {
            return true;
        }
        // Two sets replacing the same map collide even when neither names a
        // key, since clearing a map is a replacement with no entries.
        if !self.replaced_maps.is_disjoint(&committed.replaced_maps) {
            return true;
        }
        if self.index_claims.iter().any(|ours| {
            committed
                .index_claims
                .iter()
                .any(|theirs| ours.conflicts_with(theirs))
        }) {
            return true;
        }
        self.removes_something_touched_by(committed) || committed.removes_something_touched_by(self)
    }

    /// Whether this set wipes out something -- a fragment, a field, a whole
    /// string map -- that `other` also writes to or needs to still be there.
    fn removes_something_touched_by(&self, other: &Self) -> bool {
        if !self.removed_fields.is_disjoint(&other.required_fields) {
            return true;
        }
        other.writes.iter().any(|coordinate| {
            coordinate
                .fragment()
                .is_some_and(|id| self.removed_fragments.contains(&id))
                || coordinate
                    .field()
                    .is_some_and(|id| self.removed_fields.contains(&id))
                || coordinate
                    .config_map()
                    .is_some_and(|map| self.replaced_maps.contains(map))
        })
    }

    pub(super) fn add(&mut self, coordinate: Coordinate) {
        self.writes.insert(coordinate);
    }

    /// The data of each field within `fragment`.
    ///
    /// A field this operation mints records nothing -- no concurrent writer can
    /// be naming one. Nor does a fragment this operation mints write a
    /// coordinate; but the committed fields the data belongs to are still
    /// required to be there when it lands.
    ///
    /// Either way the field's definition is required. Values are written in the
    /// type the schema names, so a concurrent
    /// [`AlterField`](super::AlterField) casting it would leave the manifest
    /// describing these values as something they are not -- and unlike a
    /// [`DropField`](super::DropField), which `required_fields` catches, a cast
    /// leaves the field in place for no later check to notice.
    ///
    /// This requires the whole definition, not the type alone, so it also
    /// rejects a concurrent rename or nullability change that would not have
    /// invalidated the data. That matches the granularity of the coordinate:
    /// `AlterField` writes one `FieldDefinition` whichever facet it sets.
    pub(super) fn add_field_data(&mut self, fragment: Ref, fields: impl IntoIterator<Item = Ref>) {
        let fields: Vec<i32> = fields.into_iter().filter_map(committed_field).collect();
        for field in fields.iter().copied() {
            self.requires.insert(Coordinate::FieldDefinition(field));
        }
        match fragment.committed() {
            Some(fragment) => {
                for field in fields {
                    self.add(Coordinate::FieldData { fragment, field });
                }
            }
            None => self.required_fields.extend(fields),
        }
    }

    /// Record that this set reads `fields` in `fragment` and needs them to
    /// still hold what it read.
    ///
    /// A fragment this operation mints records nothing: no concurrent writer
    /// can have replaced data that did not exist when they planned.
    pub(super) fn require_field_data(
        &mut self,
        fragment: u64,
        fields: impl IntoIterator<Item = Ref>,
    ) {
        for field in fields.into_iter().filter_map(committed_field) {
            self.requires
                .insert(Coordinate::FieldData { fragment, field });
        }
    }

    /// A field's entry in the schema.
    pub(super) fn add_field_definition(&mut self, field: Ref) {
        if let Some(field) = committed_field(field) {
            self.add(Coordinate::FieldDefinition(field));
        }
    }

    /// Record that this set adds a segment to `name`, defining the index as
    /// `identity` and describing `coverage`.
    pub(super) fn build_index(
        &mut self,
        name: String,
        identity: IndexIdentity,
        coverage: Option<impl IntoIterator<Item = Ref>>,
    ) {
        self.index_claims.push(IndexClaim {
            name,
            identity: Some(identity),
            coverage: coverage.map(committed_fragments),
        });
    }

    /// Record that this set brings `fragments` under `name` by widening a
    /// segment that is already there, without restating what the index is.
    pub(super) fn extend_index_coverage(
        &mut self,
        name: String,
        fragments: impl IntoIterator<Item = Ref>,
    ) {
        self.index_claims.push(IndexClaim {
            name,
            identity: None,
            coverage: Some(committed_fragments(fragments)),
        });
    }

    pub(super) fn remove_fragment(&mut self, fragment: u64) {
        self.add(Coordinate::FragmentExistence(fragment));
        self.removed_fragments.insert(fragment);
    }

    /// Record an edit to one of the manifest's string maps: the keys it names,
    /// or the whole map when it replaces rather than merges.
    pub(super) fn add_map_update(&mut self, map: ConfigMap, update: &UpdateMap) {
        if update.replace {
            self.replaced_maps.insert(map);
            return;
        }
        for entry in &update.update_entries {
            self.add(Coordinate::ConfigEntry {
                map: map.clone(),
                key: entry.key.clone(),
            });
        }
    }

    /// Mark this set as rewriting the whole table, conflicting with any
    /// concurrent set whatsoever.
    pub(super) fn take_exclusive(&mut self) {
        self.exclusive = true;
    }

    pub(super) fn remove_field(&mut self, field: Ref) {
        if let Some(field) = committed_field(field) {
            self.add(Coordinate::FieldDefinition(field));
            self.removed_fields.insert(field);
        }
    }
}

/// A field id a concurrent writer could also be naming, or `None` for a field
/// this operation mints, which no one else can see yet.
fn committed_field(reference: Ref) -> Option<i32> {
    i32::try_from(reference.committed()?).ok()
}

fn committed_fragments(fragments: impl IntoIterator<Item = Ref>) -> HashSet<u64> {
    fragments
        .into_iter()
        .filter_map(|fragment| fragment.committed())
        .collect()
}

impl From<&CompositeOperation> for Footprint {
    fn from(composite_operation: &CompositeOperation) -> Self {
        let mut footprint = Self::default();
        for action in composite_operation.iter_actions() {
            action.footprint(&mut footprint);
        }
        footprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BasePath, DataFile};
    use crate::transaction::action::{
        Action, AddBase, AddDataFile, AddField, AddFragment, AlterField, DropField, RemoveFragment,
        SetDeletionFile, TombstoneFieldData, UserAction,
    };
    use arrow_schema::{DataType, Field as ArrowField};
    use lance_core::datatypes::{Field, LogicalType};
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;

    fn footprint(actions: Vec<Action>) -> Footprint {
        Footprint::from(&CompositeOperation::new(vec![UserAction::new(
            "step", actions,
        )]))
    }

    fn add_fragment(id: Ref) -> Action {
        Action::AddFragment(AddFragment {
            id,
            physical_rows: 10,
            row_id_meta: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            data_change: true,
        })
    }

    fn add_data_file(fragment: Ref, fields: &[i32]) -> Action {
        Action::AddDataFile(AddDataFile {
            fragment,
            file: DataFile::new_unstarted("data/x.lance", ConcreteFileVersion::V2_0),
            field_ids: fields
                .iter()
                .map(|field| Ref::Committed(*field as u64))
                .collect(),
            data_change: true,
        })
    }

    fn cast_field(field: i32) -> Action {
        Action::AlterField(AlterField {
            field: Ref::Committed(field as u64),
            name: None,
            logical_type: Some(LogicalType::from("int64")),
            nullable: None,
        })
    }

    fn tombstone(fragment: u64, fields: &[i32]) -> Action {
        Action::TombstoneFieldData(TombstoneFieldData {
            fragment: Ref::Committed(fragment),
            field_ids: fields.iter().map(|id| Ref::Committed(*id as u64)).collect(),
            data_change: true,
        })
    }

    fn remove_fragment(fragment: u64) -> Action {
        Action::RemoveFragment(RemoveFragment {
            fragment: Ref::Committed(fragment),
            data_change: true,
        })
    }

    fn set_deletion_file(fragment: u64) -> Action {
        Action::SetDeletionFile(SetDeletionFile {
            fragment,
            deletion_file: None,
            data_change: true,
        })
    }

    fn add_base(local: u32, name: &str, path: &str) -> Action {
        Action::AddBase(AddBase {
            local,
            base: BasePath::new(0, path.into(), Some(name.into()), false),
        })
    }

    fn add_unnamed_base(local: u32, path: &str) -> Action {
        Action::AddBase(AddBase {
            local,
            base: BasePath::new(0, path.into(), None, false),
        })
    }

    #[test]
    fn test_minting_actions_write_nothing() {
        let minting = footprint(vec![
            add_fragment(Ref::Local(0)),
            Action::AddField(AddField {
                local: 1,
                parent: None,
                def: Field::try_from(ArrowField::new("new", DataType::Int32, true)).unwrap(),
            }),
            add_data_file(Ref::Local(0), &[0]),
        ]);

        // Two writers appending at the same time never collide.
        assert!(!minting.conflicts_with(&minting.clone()));
    }

    #[rstest]
    #[case::same_field_in_same_fragment(
        vec![tombstone(0, &[1])],
        vec![add_data_file(Ref::Committed(0), &[1])],
        true,
    )]
    #[case::different_fields_in_same_fragment(
        vec![tombstone(0, &[1])],
        vec![add_data_file(Ref::Committed(0), &[2])],
        false,
    )]
    #[case::same_field_in_different_fragments(
        vec![tombstone(0, &[1])],
        vec![tombstone(1, &[1])],
        false,
    )]
    #[case::deletions_do_not_collide_with_field_data(
        vec![set_deletion_file(0)],
        vec![tombstone(0, &[1])],
        false,
    )]
    #[case::concurrent_deletes_of_one_fragment(
        vec![set_deletion_file(0)],
        vec![set_deletion_file(0)],
        true,
    )]
    #[case::removal_swallows_the_whole_fragment(
        vec![remove_fragment(0)],
        vec![tombstone(0, &[1])],
        true,
    )]
    #[case::removal_leaves_other_fragments_alone(
        vec![remove_fragment(0)],
        vec![tombstone(1, &[1])],
        false,
    )]
    #[case::same_field_definition(
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: Some("a".into()), logical_type: None, nullable: None })],
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: None, logical_type: None, nullable: Some(true) })],
        true,
    )]
    #[case::different_field_definitions(
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: None, logical_type: None, nullable: None })],
        vec![Action::AlterField(AlterField { field: Ref::Committed(2), name: None, logical_type: None, nullable: None })],
        false,
    )]
    #[case::dropping_a_field_collides_with_altering_it(
        vec![Action::DropField(DropField { field: Ref::Committed(1) })],
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: None, logical_type: None, nullable: Some(true) })],
        true,
    )]
    #[case::dropping_a_field_collides_with_rewriting_its_data(
        vec![Action::DropField(DropField { field: Ref::Committed(1) })],
        vec![tombstone(0, &[1])],
        true,
    )]
    #[case::dropping_a_field_leaves_other_fields_alone(
        vec![Action::DropField(DropField { field: Ref::Committed(1) })],
        vec![tombstone(0, &[2])],
        false,
    )]
    #[case::dropping_a_field_leaves_deletions_alone(
        vec![Action::DropField(DropField { field: Ref::Committed(1) })],
        vec![set_deletion_file(0)],
        false,
    )]
    // An append writes no coordinate -- its fragment is minted, so no
    // concurrent writer can name a cell inside it -- but the data it carries
    // still belongs to committed fields. Dropping one of those fields out from
    // under it would leave the new fragment holding data for a field the
    // manifest no longer has.
    #[case::dropping_a_field_a_concurrent_append_writes_data_for(
        vec![Action::DropField(DropField { field: Ref::Committed(1) })],
        vec![add_fragment(Ref::Local(0)), add_data_file(Ref::Local(0), &[1])],
        true,
    )]
    #[case::dropping_a_field_a_concurrent_append_does_not_write(
        vec![Action::DropField(DropField { field: Ref::Committed(1) })],
        vec![add_fragment(Ref::Local(0)), add_data_file(Ref::Local(0), &[2])],
        false,
    )]
    // A cast and data for an unrelated field never interact, in either order.
    #[case::casting_a_field_a_concurrent_append_does_not_write(
        vec![cast_field(1)],
        vec![add_fragment(Ref::Local(0)), add_data_file(Ref::Local(0), &[2])],
        false,
    )]
    #[case::bases_with_the_same_name(
        vec![add_base(0, "a", "s3://bucket/one")],
        vec![add_base(0, "a", "s3://bucket/two")],
        true,
    )]
    #[case::bases_with_the_same_location(
        vec![add_base(0, "a", "s3://bucket/one")],
        vec![add_base(0, "b", "s3://bucket/one")],
        true,
    )]
    #[case::unrelated_bases(
        vec![add_base(0, "a", "s3://bucket/one")],
        vec![add_base(0, "b", "s3://bucket/two")],
        false,
    )]
    // The name is an optional alias. Two writers who both decline to set one
    // have not picked the same name, so only their locations coordinate.
    #[case::unnamed_bases_at_different_locations(
        vec![add_unnamed_base(0, "s3://bucket/one")],
        vec![add_unnamed_base(0, "s3://bucket/two")],
        false,
    )]
    #[case::unnamed_bases_at_the_same_location(
        vec![add_unnamed_base(0, "s3://bucket/one")],
        vec![add_unnamed_base(0, "s3://bucket/one")],
        true,
    )]
    // A reservation is meant to be one writer's alone, but nothing in the
    // format enforces that, so claiming a reserved id has to be a write.
    #[case::the_same_reserved_fragment_id(
        vec![add_fragment(Ref::Committed(1))],
        vec![add_fragment(Ref::Committed(1))],
        true,
    )]
    #[case::different_reserved_fragment_ids(
        vec![add_fragment(Ref::Committed(1))],
        vec![add_fragment(Ref::Committed(2))],
        false,
    )]
    #[case::claiming_a_reserved_id_a_concurrent_set_removes(
        vec![add_fragment(Ref::Committed(1))],
        vec![remove_fragment(1)],
        true,
    )]
    fn test_conflicts(
        #[case] ours: Vec<Action>,
        #[case] theirs: Vec<Action>,
        #[case] expected: bool,
    ) {
        let ours = footprint(ours);
        let theirs = footprint(theirs);
        assert_eq!(ours.conflicts_with(&theirs), expected);
        // The relation has to hold whichever side is asking.
        assert_eq!(theirs.conflicts_with(&ours), expected);
    }

    /// Pairs whose answer depends on which set committed first, so each order
    /// is stated rather than assumed to match the other.
    ///
    /// Writing data for a field requires that field's definition, and a
    /// requirement is only checked against what committed *after* the set was
    /// planned. A cast that arrives second is the unsafe order: the data was
    /// encoded in the old type and the manifest now names the new one. A cast
    /// that arrives first is not, because applying it rebinds the field in
    /// every fragment the manifest has by then, including one a concurrent
    /// append just added.
    #[rstest]
    #[case::an_append_landing_after_a_cast(
        vec![add_fragment(Ref::Local(0)), add_data_file(Ref::Local(0), &[1])],
        vec![cast_field(1)],
        true,
    )]
    #[case::a_cast_landing_after_an_append(
        vec![cast_field(1)],
        vec![add_fragment(Ref::Local(0)), add_data_file(Ref::Local(0), &[1])],
        false,
    )]
    #[case::a_rewrite_landing_after_a_cast(
        vec![add_data_file(Ref::Committed(7), &[1])],
        vec![cast_field(1)],
        true,
    )]
    #[case::a_cast_landing_after_a_rewrite(
        vec![cast_field(1)],
        vec![add_data_file(Ref::Committed(7), &[1])],
        false,
    )]
    fn test_directional_conflicts(
        #[case] ours: Vec<Action>,
        #[case] committed: Vec<Action>,
        #[case] expected: bool,
    ) {
        assert_eq!(
            footprint(ours).conflicts_with(&footprint(committed)),
            expected
        );
    }
}
