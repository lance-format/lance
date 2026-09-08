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
use std::collections::{HashMap, HashSet};
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
            | Self::FieldName(_)
            | Self::BaseName(_)
            | Self::BaseLocation(_)
            | Self::ConfigEntry { .. }
            | Self::UnenforcedKey(_)
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

/// Everything an action set writes, in one set.
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
    /// Whether this set rewrites the table wholesale. Such a set collides with
    /// every concurrent set, including one that records no claim at all -- an
    /// append's rows would vanish or survive the reset depending on which
    /// commit landed first -- so it is a flag rather than a region.
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
    /// Coordinates and regions are compared through [`pair_conflicts`], with
    /// this set in the committing role and `committed` in the committed role.
    /// That is where the one asymmetry lives: a requirement of this set is
    /// tested against what `committed` wrote, never the reverse, because
    /// `committed` serialized first and nothing arriving later can break what
    /// it required. The distinction is not academic. Data written for a field
    /// requires the field's definition; a cast of that field writes it.
    /// Checking both directions would also reject the cast that arrives
    /// *after* the data -- an order [`AlterField`](super::AlterField) already
    /// handles, by rebinding the field in every fragment the manifest has by
    /// then.
    pub fn conflicts_with(&self, committed: &Self) -> bool {
        // A wholesale rewrite leaves nothing for a concurrent set to land on --
        // not even an append, whose rows the reset would discard or resurrect
        // depending on which commit won.
        if self.exclusive || committed.exclusive {
            return true;
        }
        if self.claims_conflict_with(committed) {
            return true;
        }
        if self.anchors_removed_by(committed) || committed.anchors_removed_by(self) {
            return true;
        }
        // Index claims are not coordinates and are compared on their own terms,
        // symmetrically.
        self.index_claims.iter().any(|ours| {
            committed
                .index_claims
                .iter()
                .any(|theirs| ours.conflicts_with(theirs))
        })
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

    /// Whether `other` removes a fragment this set needs to still be there.
    fn anchors_removed_by(&self, other: &Self) -> bool {
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
    /// [`AlterField`](super::AlterField) casting it would leave the manifest
    /// describing these values as something they are not, and a concurrent
    /// [`DropField`](super::DropField) would leave a file carrying data for a
    /// field the manifest no longer has. Both write the definition, so both are
    /// caught when they land first. Landing second, a cast rebinds the field in
    /// every fragment the manifest has by then and a drop tombstones it out of
    /// every fragment's files, so neither order needs a symmetric check.
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
        }
        let Some(fragment) = fragment.committed() else {
            return;
        };
        self.require_fragment(fragment);
        for field in fields {
            self.write(Coordinate::FieldData { fragment, field });
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

    /// Record that this set drops `field` from the schema.
    ///
    /// Only the definition is written, and only the named field's: a footprint
    /// has no schema to expand a struct with. A concurrent write to a child of
    /// a dropped struct requires the child's definition, which this does not
    /// write, so it is not caught here and fails when it is applied against the
    /// version where the child no longer exists.
    pub(super) fn remove_field(&mut self, field: Ref) {
        self.write_field_definition(field);
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
    use crate::transaction::action::test_support::footprint;
    use crate::transaction::action::{
        Action, AddBase, AddDataFile, AddField, AddFragment, AlterField, DropField, RemoveFragment,
        SetDeletionFile, TombstoneFieldData,
    };
    use arrow_schema::{DataType, Field as ArrowField};
    use lance_core::datatypes::{Field, LogicalType};
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
            logical_type: None,
            nullable: None,
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
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: None, logical_type: None, nullable: Some(true) })],
        true,
    )]
    #[case::different_field_definitions(
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: None, logical_type: None, nullable: None })],
        vec![Action::AlterField(AlterField { field: Ref::Committed(2), name: None, logical_type: None, nullable: None })],
        false,
    )]
    #[case::dropping_a_field_collides_with_altering_it(
        vec![drop_field(1)],
        vec![Action::AlterField(AlterField { field: Ref::Committed(1), name: None, logical_type: None, nullable: Some(true) })],
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
    // A cast and data for an unrelated field never interact, in either order.
    #[case::casting_a_field_a_concurrent_append_does_not_write(
        vec![cast_field(1)],
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
    /// checked against what committed *after* the set was planned. A cast or a
    /// drop that arrives second is the unsafe order: the data was encoded
    /// against a definition the manifest no longer carries. Arriving first they
    /// are not, because applying a cast rebinds the field in every fragment the
    /// manifest has by then, and applying a drop tombstones the field out of
    /// every fragment's files -- including one a concurrent set just added.
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
