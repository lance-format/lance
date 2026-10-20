// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! What an action set writes and what it requires of the version it lands on.
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

use super::{Action, CompositeOperation, Ref};
use crate::transaction::UpdateMap;
use std::collections::{HashMap, HashSet};

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
    /// The schema new rows are written in. Every set that adds rows requires
    /// it, because a writer prepares rows against the schema it read: a row
    /// minted later carries no value for a non-nullable column added in
    /// between. A set that adds such a column, or makes a column required,
    /// writes it. So does one that adds a child under a committed parent, which
    /// a stale writer may write packed without the child. A stale writer
    /// cannot name a field it does not write, so the field's own coordinates
    /// cannot say this.
    NewRowSchema,
    /// A field's name. Names only have to be unique among siblings, but a
    /// footprint has no schema to place a field among its siblings, so a name
    /// coordinates table-wide: two writers adding a `c` under different structs
    /// collide as if they had added it at the top level. The cost is a spurious
    /// conflict in that case; the alternative is two fields called `c` side by
    /// side, which no reader can tell apart.
    FieldName(String),
    /// A base path's name, where it has one. Names must be unique; an unset
    /// name is an absent alias rather than a name shared with every other
    /// unnamed base, so it coordinates nothing.
    BaseName(String),
    /// A base path's location, which the manifest requires to be unique.
    BaseLocation(String),
    /// One key in one of the manifest's string maps.
    ConfigEntry { map: ConfigMap, key: String },
    /// One of the table-wide keys declared through field metadata. Each is set
    /// once, on one set of fields, so two writers declaring it on fields of
    /// their own collide even though they name different fields' metadata.
    UnenforcedKey(UnenforcedKey),
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

/// A key the table declares once, across its fields, through reserved field
/// metadata entries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnenforcedKey {
    Primary,
    Clustering,
}

impl Coordinate {
    /// The fragment this coordinate lives in, if it is fragment-scoped.
    fn fragment(&self) -> Option<u64> {
        match self {
            Self::FragmentExistence(id) | Self::FragmentDeletions(id) => Some(*id),
            Self::FieldData { fragment, .. } => Some(*fragment),
            Self::FieldDefinition(_)
            | Self::NewRowSchema
            | Self::FieldName(_)
            | Self::BaseName(_)
            | Self::BaseLocation(_)
            | Self::ConfigEntry { .. }
            | Self::UnenforcedKey(_) => None,
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

/// How an action set touches a coordinate. Ordered weakest to strongest, so
/// that a set touching one coordinate two ways is summarized by the maximum.
///
/// The whole conflict rule between two sets is the table in
/// [`pair_conflicts`]: one row per mode of the set being committed, one column
/// per mode of the set that committed before it. A new way of touching a
/// coordinate is a new variant here and a new row and column there.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Mode {
    /// Reads the coordinate and needs it to still hold what it read. Data
    /// written for a committed field requires the field's definition, since
    /// the values are encoded in the type it names. Unlike a write, two sets
    /// may require the same coordinate -- two readers of one column do not
    /// collide.
    Requires,
    /// Replaces the coordinate. Two of these on one coordinate never commute.
    Writes,
}

/// A part of the manifest an action set touches wholesale, without being able
/// to enumerate the coordinates inside it. Compared against the other set by
/// containment: a region write is a write of every coordinate the region holds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Region {
    /// A committed fragment: its existence, its deletions, and the data of
    /// every field in it.
    Fragment(u64),
    /// One of the manifest's string maps, every key included.
    Map(ConfigMap),
}

impl Region {
    fn contains(&self, coordinate: &Coordinate) -> bool {
        match self {
            Self::Fragment(id) => coordinate.fragment() == Some(*id),
            Self::Map(map) => coordinate.config_map() == Some(map),
        }
    }
}

/// Whether a claim of mode `committing` collides with a claim of mode
/// `committed` on the same coordinate, where the `committed` set landed after
/// the `committing` set read.
///
/// | committing \ committed | Writes   | Requires |
/// |------------------------|----------|----------|
/// | Writes                 | conflict | ok       |
/// | Requires               | conflict | ok       |
///
/// The `Requires` column is all "ok": what the committed set required held when
/// it committed, and it serialized first, so nothing arriving later can
/// retroactively break it. The `Requires` row is the open question, because
/// the committing set read before the other landed.
///
/// The table is monotone along [`Mode`]'s order in every row and column, which
/// is what lets one coordinate touched two ways by one set be summarized by
/// its strongest mode.
fn pair_conflicts(committing: Mode, committed: Mode) -> bool {
    match (committing, committed) {
        (Mode::Writes | Mode::Requires, Mode::Writes) => true,
        (_, Mode::Requires) => false,
    }
}

/// Something that must not have happened in a set that landed after this one
/// read.
///
/// A [`Mode::Requires`] claim is checked against what the other set wrote, but
/// never the reverse: what the committed set required held when it committed.
/// That leaves a gap for a set that is valid only against the rows or values it
/// read, such as a stricter nullability, whose validity a later write can
/// revoke without touching any coordinate the first set names. The committed
/// set is queried for the fact itself.
///
/// Preconditions are one-way by design. The set that landed first was valid
/// when it landed, and a precondition says nothing about the order the other
/// way around; that order is covered by the claims the later set makes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Precondition {
    /// No values were written for this field, in any fragment. An added row
    /// counts as writing every field, since it holds a value -- if only a null
    /// -- for each.
    NoValuesWritten(i32),
    /// No fragment was added.
    NoRowsAdded,
}

/// Everything an action set writes, and everything it requires to still hold
/// when it lands: definitions it depends on, fragments that must survive, and
/// preconditions on what a concurrent set may have written.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Footprint {
    /// Every coordinate this set touches, with the strongest [`Mode`] it
    /// touches it in.
    ///
    /// There is no region for a dropped field, the way there is for a removed
    /// fragment. Dropping a field writes its definition, and anything that
    /// depends on the field -- data written for it, its metadata, a child added
    /// under it -- requires that definition, so the drop-then-write order is a
    /// `Requires`-vs-`Writes` collision. The write-then-drop order is safe,
    /// because [`DropField`](super::DropField) applies to every fragment the
    /// manifest holds by then, including one a concurrent set just added, and
    /// tombstones the field out of each.
    claims: HashMap<Coordinate, Mode>,
    /// Regions this set writes wholesale: fragments it removes, string maps it
    /// replaces rather than merges into. A region write counts as a write of
    /// every coordinate inside the region, and of the region itself -- two sets
    /// replacing one map name no key in common, and still collide.
    regions: HashMap<Region, Mode>,
    /// Fragments this set needs to still be there, without writing anything a
    /// concurrent set could name inside them. Data for a field this set mints,
    /// written into a committed fragment, is the case this exists for: the
    /// field id is invisible to a concurrent writer, so the cells are not a
    /// coordinate, but they are gone if the fragment is.
    ///
    /// Not a `Requires` claim on the region, because the check is symmetric: a
    /// removal that lands second destroys the cells just as surely as one that
    /// lands first. And not a write, because it must not collide with a
    /// concurrent write to some other coordinate in the same fragment.
    required_fragments: HashSet<u64>,
    /// Facts about the table this set is valid only against. See
    /// [`Precondition`].
    preconditions: Vec<Precondition>,
    /// Committed fields this set writes values for, in any fragment, including
    /// one it mints. Unlike [`Coordinate::FieldData`] this is not a claim: two
    /// sets writing one field's values into different fragments commute, but a
    /// set that needs the field to hold no new values cannot commit after
    /// either.
    values_written: HashSet<i32>,
    /// Whether this set adds a fragment.
    rows_added: bool,
    /// Whether this set rewrites the table wholesale. Such a set collides with
    /// every concurrent set, including one that records no claim at all -- an
    /// append's rows would vanish or survive the reset depending on which
    /// commit landed first -- so it is a flag rather than a region.
    exclusive: bool,
}

impl Footprint {
    /// Whether this set can still commit on top of `committed`, a set that
    /// landed after this one read.
    ///
    /// Coordinates and regions are compared through `pair_conflicts`, with
    /// this set in the committing role and `committed` in the committed role.
    /// That is where the one asymmetry lives: a requirement of this set is
    /// tested against what `committed` wrote, never the reverse, because
    /// `committed` serialized first and nothing arriving later can break what
    /// it required. A set whose validity depends on what it read rather than on
    /// a coordinate -- a tightened nullability, a replaced field -- adds
    /// [`Precondition`]s, which are tested against the facts `committed`
    /// recorded, and are likewise one-way.
    pub fn conflicts_with(&self, committed: &Self) -> bool {
        // A wholesale rewrite leaves nothing for a concurrent set to land on --
        // not even an append, whose rows the reset would discard or resurrect
        // depending on which commit won.
        if self.exclusive || committed.exclusive {
            return true;
        }
        if self.claims_conflict_with(committed) || self.preconditions_violated_by(committed) {
            return true;
        }
        self.required_fragment_removed_by(committed) || committed.required_fragment_removed_by(self)
    }

    /// Whether any coordinate touched by both sets, directly or through a
    /// region, collides under [`pair_conflicts`] with this set committing after
    /// `committed`.
    ///
    /// Every pair is enumerated from this side's point of view: our claims
    /// against their claims and their regions, their claims against our
    /// regions, and our regions against theirs. A region only matches a region
    /// that is equal to it; containment is between a region and a coordinate.
    fn claims_conflict_with(&self, committed: &Self) -> bool {
        let ours_against_theirs = self.claims.iter().flat_map(|(coordinate, ours)| {
            let same_coordinate = committed.claims.get(coordinate).copied();
            let containing_regions = committed
                .regions
                .iter()
                .filter(|(region, _)| region.contains(coordinate))
                .map(|(_, theirs)| *theirs);
            same_coordinate
                .into_iter()
                .chain(containing_regions)
                .map(move |theirs| (*ours, theirs))
        });
        let theirs_inside_our_regions = committed.claims.iter().flat_map(|(coordinate, theirs)| {
            self.regions
                .iter()
                .filter(|(region, _)| region.contains(coordinate))
                .map(move |(_, ours)| (*ours, *theirs))
        });
        let same_region = self.regions.iter().filter_map(|(region, ours)| {
            committed.regions.get(region).map(|theirs| (*ours, *theirs))
        });
        ours_against_theirs
            .chain(theirs_inside_our_regions)
            .chain(same_region)
            .any(|(ours, theirs)| pair_conflicts(ours, theirs))
    }

    fn preconditions_violated_by(&self, committed: &Self) -> bool {
        self.preconditions
            .iter()
            .any(|precondition| match precondition {
                Precondition::NoValuesWritten(field) => {
                    committed.rows_added || committed.values_written.contains(field)
                }
                Precondition::NoRowsAdded => committed.rows_added,
            })
    }

    /// Whether `other` removes a fragment this set needs to still be there.
    fn required_fragment_removed_by(&self, other: &Self) -> bool {
        self.required_fragments
            .iter()
            .any(|fragment| other.regions.contains_key(&Region::Fragment(*fragment)))
    }

    /// Record a claim on `coordinate`, keeping the strongest mode if the set
    /// already touches it another way. See [`pair_conflicts`] for why the
    /// strongest mode is the right summary.
    fn claim(&mut self, coordinate: Coordinate, mode: Mode) {
        self.claims
            .entry(coordinate)
            .and_modify(|current| *current = (*current).max(mode))
            .or_insert(mode);
    }

    /// Record that this set writes `coordinate`. See [`Mode::Writes`].
    pub(super) fn write(&mut self, coordinate: Coordinate) {
        self.claim(coordinate, Mode::Writes);
    }

    /// Record that this set reads `coordinate` and needs it to still hold what
    /// it read. See [`Mode::Requires`].
    pub(super) fn require(&mut self, coordinate: Coordinate) {
        self.claim(coordinate, Mode::Requires);
    }

    /// The data of each field within `fragment`.
    ///
    /// A field this operation mints writes no coordinate -- no concurrent
    /// writer can be naming one -- and neither does a fragment this operation
    /// mints. But data written into a committed fragment lives or dies with
    /// that fragment whichever kind of field it is for, so the fragment is
    /// required to still be there: a concurrent compaction that removed it
    /// would otherwise discard the new column's cells for those rows, and the
    /// replacement fragment would read them back as null.
    ///
    /// Every committed field's definition is required. Values are written in
    /// the type the schema names, so a concurrent
    /// [`AlterField`](super::AlterField) changing its definition could leave the
    /// manifest describing these values as something they are not, and a
    /// concurrent [`DropField`](super::DropField) would leave a file carrying
    /// data for a field the manifest no longer has. Both write the definition,
    /// so both are caught when they land first. Landing second, a drop tombstones the field
    /// out of every fragment's files, so it needs no symmetric check. A
    /// replacement does, because it rewrites the values it read; see
    /// [`Footprint`]'s `From<&CompositeOperation>`.
    ///
    /// This requires the whole definition, not the type alone, so it also
    /// rejects a concurrent rename or nullability change that would not have
    /// invalidated the data. That matches the granularity of the coordinate:
    /// `AlterField` writes one `FieldDefinition` whichever facet it sets.
    pub(super) fn write_field_data(
        &mut self,
        fragment: Ref,
        fields: impl IntoIterator<Item = Ref>,
    ) {
        let fields: Vec<i32> = fields.into_iter().filter_map(committed_field).collect();
        for field in fields.iter().copied() {
            self.require(Coordinate::FieldDefinition(field));
            self.values_written.insert(field);
        }
        let Some(fragment) = fragment.committed() else {
            return;
        };
        self.require_fragment(fragment);
        for field in fields {
            self.write(Coordinate::FieldData { fragment, field });
        }
    }

    /// Record that this set adds a fragment, whose rows were prepared against
    /// the schema this set read. See [`Coordinate::NewRowSchema`].
    pub(super) fn add_rows(&mut self) {
        self.rows_added = true;
        self.require(Coordinate::NewRowSchema);
    }

    /// Record that this set adds a column that rows must carry a value for, or
    /// makes an existing one required. Rows a stale writer adds afterwards
    /// would not carry one, and rows already added are not covered by what
    /// this set read or wrote.
    pub(super) fn add_required_column(&mut self) {
        self.write(Coordinate::NewRowSchema);
        self.preconditions.push(Precondition::NoRowsAdded);
    }

    /// Record that this set only holds if `field` has had no values written
    /// since it read it, such as a stricter nullability the apply does not
    /// check the values against.
    pub(super) fn require_no_values_written(&mut self, field: Ref) {
        if let Some(field) = committed_field(field) {
            self.preconditions
                .push(Precondition::NoValuesWritten(field));
        }
    }

    /// A field's entry in the schema.
    pub(super) fn write_field_definition(&mut self, field: Ref) {
        if let Some(field) = committed_field(field) {
            self.write(Coordinate::FieldDefinition(field));
        }
    }

    /// Record that this set depends on `field` still being defined as it was,
    /// without redefining it: a child added under it, metadata written on it.
    pub(super) fn require_field_definition(&mut self, field: Ref) {
        if let Some(field) = committed_field(field) {
            self.require(Coordinate::FieldDefinition(field));
        }
    }

    /// Note that this set only works if `fragment` is still part of the dataset,
    /// without claiming anything inside it.
    pub(super) fn require_fragment(&mut self, fragment: u64) {
        self.required_fragments.insert(fragment);
    }

    /// Record that this set removes `fragment` outright: its existence, and
    /// with it every coordinate inside it.
    pub(super) fn remove_fragment(&mut self, fragment: u64) {
        self.write(Coordinate::FragmentExistence(fragment));
        self.regions
            .insert(Region::Fragment(fragment), Mode::Writes);
    }

    /// Record an edit to one of the manifest's string maps: the keys it names,
    /// or the whole map when it replaces rather than merges.
    pub(super) fn write_map_update(&mut self, map: ConfigMap, update: &UpdateMap) {
        if update.replace {
            self.regions.insert(Region::Map(map), Mode::Writes);
            return;
        }
        for entry in &update.update_entries {
            self.write(Coordinate::ConfigEntry {
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
}

/// A field id a concurrent writer could also be naming, or `None` for a field
/// this operation mints, which no one else can see yet.
fn committed_field(reference: Ref) -> Option<i32> {
    i32::try_from(reference.committed()?).ok()
}

impl From<&CompositeOperation> for Footprint {
    fn from(composite_operation: &CompositeOperation) -> Self {
        let mut footprint = Self::default();
        for action in composite_operation.iter_actions() {
            action.footprint(&mut footprint);
        }

        // Dropping a field and adding one by the same name is how a cast is
        // spelled, since [`AlterField`](super::AlterField) has no type facet:
        // the new field's data is rewritten from the dropped field's
        // values as they were read. A footprint has no schema to match the
        // names, so any drop alongside any add is treated as a replacement. The
        // cost is a spurious conflict for an unrelated drop and add in one set.
        let adds_field = composite_operation
            .iter_actions()
            .any(|action| matches!(action, Action::AddField(_)));
        if adds_field {
            for action in composite_operation.iter_actions() {
                if let Action::DropField(drop) = action {
                    footprint.require_no_values_written(drop.field);
                }
            }
        }
        footprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{BasePath, DataFile};
    use crate::transaction::action::test_support::footprint;
    use crate::transaction::action::{
        Action, AddBase, AddDataFile, AddField, AddFragment, AlterField, DropField, RemoveFragment,
        SetDeletionFile, TombstoneFieldData,
    };
    use arrow_schema::{DataType, Field as ArrowField};
    use lance_core::datatypes::Field;
    use lance_file::version::ConcreteFileVersion;
    use rstest::rstest;

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
        add_data_file_for(
            fragment,
            fields
                .iter()
                .map(|field| Ref::Committed(*field as u64))
                .collect(),
        )
    }

    fn add_data_file_for(fragment: Ref, field_ids: Vec<Ref>) -> Action {
        Action::AddDataFile(AddDataFile {
            fragment,
            file: DataFile::new_unstarted("data/x.lance", ConcreteFileVersion::V2_0),
            field_ids,
            data_change: true,
        })
    }

    fn add_field(local: u32, name: &str, parent: Option<Ref>) -> Action {
        Action::AddField(AddField {
            local,
            parent,
            def: Field::try_from(ArrowField::new(name, DataType::Int32, true)).unwrap(),
        })
    }

    fn add_non_nullable_field(local: u32, name: &str) -> Action {
        Action::AddField(AddField {
            local,
            parent: None,
            def: Field::try_from(ArrowField::new(name, DataType::Int32, false)).unwrap(),
        })
    }

    fn set_nullability(field: i32, nullable: bool) -> Action {
        Action::AlterField(AlterField {
            field: Ref::Committed(field as u64),
            name: None,
            nullable: Some(nullable),
        })
    }

    fn tighten_local_field(local: u32) -> Action {
        Action::AlterField(AlterField {
            field: Ref::Local(local),
            name: None,
            nullable: Some(false),
        })
    }

    fn append(field: i32) -> Vec<Action> {
        vec![
            add_fragment(Ref::Local(0)),
            add_data_file(Ref::Local(0), &[field]),
        ]
    }

    /// The add-column shape: mint a field, then back it in a committed fragment.
    fn add_column(name: &str, fragment: u64) -> Vec<Action> {
        vec![
            add_field(1, name, None),
            add_data_file_for(Ref::Committed(fragment), vec![Ref::Local(1)]),
        ]
    }

    fn rename_field(field: i32, name: &str) -> Action {
        Action::AlterField(AlterField {
            field: Ref::Committed(field as u64),
            name: Some(name.into()),
            nullable: None,
        })
    }

    fn drop_field(field: i32) -> Action {
        Action::DropField(DropField {
            field: Ref::Committed(field as u64),
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
    // Two compactions of one fragment: the second would remove something the
    // first already took away.
    #[case::removing_the_same_fragment_twice(
        vec![remove_fragment(0)],
        vec![remove_fragment(0)],
        true,
    )]
    // A new column's cells in a committed fragment are not a coordinate -- the
    // field is minted -- but they are gone if the fragment is, whichever order
    // the two land in.
    #[case::adding_a_column_into_a_fragment_a_concurrent_set_removes(
        add_column("c", 7),
        vec![remove_fragment(7)],
        true,
    )]
    #[case::adding_a_column_leaves_other_writes_to_the_fragment_alone(
        add_column("c", 7),
        vec![tombstone(7, &[2]), set_deletion_file(7)],
        false,
    )]
    #[case::same_field_definition(
        vec![rename_field(1, "a")],
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: None, nullable: Some(true) })],
        true,
    )]
    #[case::different_field_definitions(
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: None, nullable: None })],
        vec![Action::AlterField(AlterField { field: Ref::Committed(2), name: None, nullable: None })],
        false,
    )]
    #[case::dropping_a_field_collides_with_altering_it(
        vec![drop_field(1)],
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: None, nullable: Some(true) })],
        true,
    )]
    #[case::dropping_a_field_leaves_other_fields_alone(
        vec![drop_field(1)],
        vec![tombstone(0, &[2])],
        false,
    )]
    #[case::dropping_a_field_leaves_deletions_alone(
        vec![drop_field(1)],
        vec![set_deletion_file(0)],
        false,
    )]
    #[case::dropping_a_field_a_concurrent_append_does_not_write(
        vec![drop_field(1)],
        vec![add_fragment(Ref::Local(0)), add_data_file(Ref::Local(0), &[2])],
        false,
    )]
    // Names coordinate table-wide, whether a writer takes one by adding a
    // field or by renaming one to it.
    #[case::adding_two_fields_with_the_same_name(
        vec![add_field(0, "c", None)],
        vec![add_field(0, "c", None)],
        true,
    )]
    #[case::adding_two_fields_with_different_names(
        vec![add_field(0, "c", None)],
        vec![add_field(0, "d", None)],
        false,
    )]
    #[case::adding_a_field_named_like_a_concurrent_rename(
        vec![add_field(0, "c", None)],
        vec![rename_field(1, "c")],
        true,
    )]
    #[case::renaming_two_fields_to_the_same_name(
        vec![rename_field(1, "c")],
        vec![rename_field(2, "c")],
        true,
    )]
    #[case::adding_the_same_name_under_different_parents(
        vec![add_field(0, "c", Some(Ref::Committed(1)))],
        vec![add_field(0, "c", Some(Ref::Committed(2)))],
        true,
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
    /// Anything that depends on a field -- data written for it, a child added
    /// under it -- requires that field's definition, and a requirement is only
    /// checked against what committed *after* the set was planned. A drop that
    /// arrives second is safe, because applying it tombstones the field out of
    /// every fragment's files, including one a concurrent set just added. A
    /// set that is valid only against the values it read -- a replacement, a
    /// tightening -- is not: it carries a precondition instead.
    #[rstest]
    // A cast is a drop, an add under the same name, and the data. The footprint
    // has no schema to match the name, so any drop paired with an add counts as
    // a replacement, and the replacement's data would miss a value a concurrent
    // writer put in the dropped field.
    #[case::a_replacement_landing_after_an_append(
        vec![drop_field(1), add_field(0, "a", None)],
        append(1),
        true,
    )]
    #[case::a_replacement_landing_after_a_rewrite(
        vec![drop_field(1), add_field(0, "a", None)],
        vec![add_data_file(Ref::Committed(7), &[1])],
        true,
    )]
    // An added row counts as writing every field, so the replaced field's
    // value in it -- null -- was never part of the replacement's data.
    #[case::a_replacement_landing_after_an_append_omitting_the_field(
        vec![drop_field(1), add_field(0, "a", None)],
        append(2),
        true,
    )]
    #[case::a_replacement_landing_after_a_rewrite_of_another_field(
        vec![drop_field(1), add_field(0, "a", None)],
        vec![add_data_file(Ref::Committed(7), &[2])],
        false,
    )]
    // The reverse order is already caught: the append requires the definition
    // the replacement drops.
    #[case::an_append_landing_after_a_replacement(
        append(1),
        vec![drop_field(1), add_field(0, "a", None)],
        true,
    )]
    // New rows must carry a value for every non-nullable field, so a stale
    // append cannot land after one is added, and an add cannot land after rows
    // the add's data never covered. A nullable add needs neither.
    #[case::an_append_landing_after_a_non_nullable_add(
        append(1),
        vec![add_non_nullable_field(0, "c")],
        true,
    )]
    #[case::a_non_nullable_add_landing_after_an_append(
        vec![add_non_nullable_field(0, "c")],
        append(1),
        true,
    )]
    #[case::an_append_landing_after_a_nullable_add(
        append(1),
        vec![add_field(0, "c", None)],
        false,
    )]
    #[case::a_nullable_add_landing_after_an_append(
        vec![add_field(0, "c", None)],
        append(1),
        false,
    )]
    // Tightening is valid against the version it read; values written since
    // may hold nulls the apply never sees.
    #[case::a_tightening_landing_after_an_append_to_the_field(
        vec![set_nullability(1, false)],
        append(1),
        true,
    )]
    #[case::a_tightening_landing_after_a_rewrite_of_the_field(
        vec![set_nullability(1, false)],
        vec![add_data_file(Ref::Committed(7), &[1])],
        true,
    )]
    #[case::a_tightening_landing_after_an_append_omitting_the_field(
        vec![set_nullability(1, false)],
        append(2),
        true,
    )]
    #[case::a_tightening_landing_after_a_rewrite_of_another_field(
        vec![set_nullability(1, false)],
        vec![add_data_file(Ref::Committed(7), &[2])],
        false,
    )]
    #[case::an_append_landing_after_a_tightening(
        append(1),
        vec![set_nullability(1, false)],
        true,
    )]
    // The append cannot name a field it does not write, so the tightening has
    // to fence the schema new rows are written in.
    #[case::an_append_omitting_the_field_landing_after_a_tightening(
        append(2),
        vec![set_nullability(1, false)],
        true,
    )]
    // Adding a nullable field and tightening it in the same set is a
    // non-nullable add by another name.
    #[case::an_append_landing_after_a_local_field_is_added_and_tightened(
        append(1),
        vec![add_field(0, "c", None), tighten_local_field(0)],
        true,
    )]
    #[case::a_local_field_added_and_tightened_landing_after_an_append(
        vec![add_field(0, "c", None), tighten_local_field(0)],
        append(1),
        true,
    )]
    #[case::a_relaxation_landing_after_an_append(
        vec![set_nullability(1, true)],
        append(1),
        false,
    )]
    #[case::an_append_landing_after_a_drop(
        vec![add_fragment(Ref::Local(0)), add_data_file(Ref::Local(0), &[1])],
        vec![drop_field(1)],
        true,
    )]
    #[case::a_drop_landing_after_an_append(
        vec![drop_field(1)],
        vec![add_fragment(Ref::Local(0)), add_data_file(Ref::Local(0), &[1])],
        false,
    )]
    #[case::a_rewrite_landing_after_a_drop(
        vec![tombstone(0, &[1])],
        vec![drop_field(1)],
        true,
    )]
    #[case::a_drop_landing_after_a_rewrite(
        vec![drop_field(1)],
        vec![tombstone(0, &[1])],
        false,
    )]
    // A child added under a struct needs the struct to still be there; the
    // struct dropped afterwards takes the child with it, as intended.
    #[case::a_child_landing_after_its_parent_was_dropped(
        vec![add_field(0, "child", Some(Ref::Committed(1)))],
        vec![drop_field(1)],
        true,
    )]
    #[case::a_parent_dropped_after_a_child_was_added(
        vec![drop_field(1)],
        vec![add_field(0, "child", Some(Ref::Committed(1)))],
        false,
    )]
    // A stale append may write the parent packed, without the new child.
    #[case::an_append_landing_after_a_child_was_added_under_a_committed_parent(
        append(2),
        vec![add_field(0, "child", Some(Ref::Committed(1)))],
        true,
    )]
    #[case::an_append_landing_after_a_child_was_added_under_a_local_parent(
        append(2),
        vec![
            add_field(0, "parent", None),
            add_field(1, "child", Some(Ref::Local(0))),
        ],
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
