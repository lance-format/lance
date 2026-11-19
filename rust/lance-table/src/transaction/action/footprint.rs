// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Whether a pending transaction can still land on the commits that landed
//! after it read.
//!
//! A pending transaction states the parts of the table it read or writes
//! ([`Part`]). The commits that landed since it read are seen as one change
//! of state, from the manifest it read to the latest one ([`Window`]): each
//! part is looked up in both and compared, and a part that changed is a
//! conflict. Writes and reads are treated alike, so two writers of one part
//! collide and the first to commit wins.
//!
//! The manifests cannot say everything. What only a committed transaction
//! knows is what it [`Did`]: whether it inserted rows and with which keys,
//! which fragments it moved rows out of to rewrite them elsewhere, which
//! fields it wrote values for, which indices it wrote, and whether it reset
//! the table. A pending transaction states what it cannot land over as
//! [`Forbid`]s, each checked against what every committed transaction did.
//!
//! Nothing here diffs whole manifests: a part is found by id in each manifest,
//! so the cost grows with what the pending transaction touches, not with the
//! table.
//!
//! Which parts an action touches is decided by that action, in its own
//! module; the comparison is in `window`.

mod window;

pub use window::Window;

use super::{Action, CompositeOperation, FieldRef, Ref};
use crate::format::BasePath;
use crate::format::key_existence::KeyExistenceFilter;
use crate::system_index::mem_wal::MEM_WAL_INDEX_NAME;
use crate::transaction::{Operation, Transaction, UpdateMap};
use lance_core::datatypes::{Field, Schema};

use roaring::RoaringTreemap;
use std::collections::HashSet;
use std::sync::Arc;
use uuid::Uuid;

/// A part of the table's state, found by key in a manifest. Any change to it
/// between the read and the latest manifest is a conflict, except where the
/// part says otherwise.
///
/// Only parts the read version already holds appear, plus fragment ids it
/// reserved. A fragment, field, or base the transaction mints has no id until
/// it lands, so nothing concurrent can be touching it.
///
/// A fragment is split so that every one of its fields is compared by exactly
/// one part: whether it is there, its data files, its overlays, its row
/// versions, and its deletions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Part {
    /// Whether one fragment is there at all.
    FragmentExists(u64),
    /// One fragment's data files, row count, clustering, and metadata.
    DataFiles(u64),
    /// One fragment's overlays.
    Overlays(u64),
    /// One fragment's row ids and the versions its rows were created and
    /// last updated at.
    RowVersions(u64),
    /// One fragment's deletion file.
    Deletions(u64),
    /// The data file and column backing one field in one fragment.
    FieldData { fragment: u64, field: i32 },
    /// The data an index segment was built from: what backs, in one
    /// fragment, every field that is, contains, or is nested in one field the
    /// index keys or carries. A fragment removed since is no change: a
    /// segment naming a removed fragment reaches its rows through the
    /// fragment reuse history or loses that coverage.
    ///
    /// TODO: a stand-in until index coverage is repaired after the check
    /// instead, which withdraws the rewritten fragments rather than
    /// conflicting.
    IndexedFieldData { fragment: u64, field: i32 },
    /// What a reader sees of one field in one fragment: its data and the
    /// overlays over it.
    Cells { fragment: u64, field: i32 },
    /// Whether a field is in the schema.
    FieldExists(i32),
    /// A field's definition: its name, type, nullability, and children.
    FieldDefinition(i32),
    /// The schema new rows and values are written in. It changes when a
    /// field becomes required -- added non-nullable, or tightened -- and when
    /// a child is added under a field the read version holds, which a writer
    /// that read before may write packed without the child.
    NewRowSchema,
    /// One key of one of the manifest's string maps.
    MapEntry { map: ConfigMap, key: String },
    /// One of the manifest's string maps, every key included.
    Map(ConfigMap),
    /// Whether a base by this name exists.
    BaseName(String),
    /// Whether a base at this location exists.
    BaseLocation(String),
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

/// What a transaction did that its manifests do not show. A committed
/// transaction's is what a pending one's [`Forbid`]s are checked against.
///
/// Set-valued facts are unions over the whole transaction, so checking a
/// forbid costs one lookup per committed transaction, not one per fragment it
/// touched.
#[derive(Debug, Clone, Default, PartialEq)]
struct Did {
    /// Brought rows into the table that were not there before.
    /// Over-approximated: the rows a merge insert updates arrive in a new
    /// fragment too.
    inserted_rows: bool,
    /// Which keys the inserted rows carry, over which columns, one per
    /// [`AssertUniqueKeys`](super::AssertUniqueKeys).
    inserted_keys: Vec<Arc<KeyAssertion>>,
    /// Committed fields it wrote values for in committed fragments, in data
    /// files or overlays.
    wrote_values: HashSet<i32>,
    /// Committed fragments it moved some rows out of into a fragment of its
    /// own, deleting them in place, as a row-moving update does. The moved
    /// rows carry the values it read.
    relocated_rows_from: HashSet<u64>,
    /// Row ids its new fragments arrive carrying, out of a reservation or
    /// kept by a compaction.
    supplied_row_ids: RoaringTreemap,
    /// What it wrote into logical indices.
    index_writes: Vec<IndexWrite>,
    /// Index segments it removed or edited.
    edited_segments: HashSet<Uuid>,
    /// The committed fields, by committed fragment, it wrote new base values
    /// for, which tombstone every overlay on them as they land.
    superseded_overlays: HashSet<(u64, i32)>,
    /// Rewrote the table wholesale. Landing first, it preempts everything
    /// that read before it; landing second, it wins over what landed in
    /// between, and only its parts and forbids are checked.
    reset_table: bool,
    /// What it touched cannot be stated. Conflicts with everything, in either
    /// order.
    opaque: bool,
}

/// Something a pending transaction cannot land over, which only what the
/// committed transactions [`Did`] can say.
///
/// A write only one transaction may make -- a row id supplied, an index
/// written, a segment edited -- is both recorded in [`Did`] and forbidden
/// here, so two transactions that both make it collide in either order.
#[derive(Debug, Clone, PartialEq)]
enum Forbid {
    /// No rows were inserted. Rows prepared against an older schema carry no
    /// value for a column added required since, or may hold a null in one
    /// made required since.
    Inserts,
    /// No values were written for this field the read version holds, nor,
    /// through a struct, for an ancestor or descendant of it, in any fragment;
    /// nor rows inserted, which carry a value for every field. The
    /// transaction validated or copied the values it read: a tightened
    /// nullability, or a field replaced by a rewrite of its values.
    ValueWrites(i32),
    /// No key this assertion covers was inserted, nor any value written into
    /// its key columns. One-way, as legacy merge insert's inserted-key filter
    /// is: a uniqueness assertion is a precondition on the commit that
    /// carries it, not an invariant on the table.
    CollidingKeys(Arc<KeyAssertion>),
    /// No rows were moved out of this fragment, whose files the transaction
    /// writes: the moved rows were copied without the write, in either order.
    RelocationFrom(u64),
    /// No new fragment carries any of these row ids: one id names one row.
    RowIds(RoaringTreemap),
    /// No write into the same index that this one cannot coexist with; see
    /// [`IndexWrite::conflicts_with`].
    IndexWrite(IndexWrite),
    /// No edit of this index segment.
    SegmentEdit(Uuid),
    /// No new base values were written for this field in this fragment: they
    /// tombstone every overlay on it as they land, this transaction's
    /// included, so its values would be lost without a trace.
    OverlaySuperseded { fragment: u64, field: i32 },
}

impl Forbid {
    /// Whether a committed transaction that did `theirs` did what this
    /// forbids, with fields related through `schema`, the one the window
    /// starts at. The whole rule for what the manifests do not show.
    fn occurs_in(&self, theirs: &Did, schema: &Schema) -> bool {
        match self {
            Self::Inserts => theirs.inserted_rows,
            Self::ValueWrites(field) => {
                theirs.inserted_rows
                    || related_fields(schema, *field)
                        .iter()
                        .any(|field| theirs.wrote_values.contains(field))
            }
            Self::CollidingKeys(ours) => {
                let wrote_a_key_column = ours
                    .key_fields
                    .iter()
                    .filter_map(|field| field.committed())
                    .any(|field| theirs.wrote_values.contains(&field));
                // Inserting rows is only compatible if `theirs` says which keys
                // it inserted over the same columns, and the filters provably do
                // not intersect. Anything less leaves the assertion unverifiable.
                let mut over_same_columns = theirs
                    .inserted_keys
                    .iter()
                    .filter(|theirs| theirs.key_fields == ours.key_fields)
                    .peekable();
                let may_collide = theirs.inserted_rows
                    && (over_same_columns.peek().is_none()
                        || over_same_columns.any(|theirs| {
                            !matches!(ours.filter.intersects(&theirs.filter), Ok((false, _)))
                        }));
                wrote_a_key_column || may_collide
            }
            Self::RelocationFrom(fragment) => theirs.relocated_rows_from.contains(fragment),
            Self::RowIds(ours) => !ours.is_disjoint(&theirs.supplied_row_ids),
            Self::IndexWrite(ours) => theirs
                .index_writes
                .iter()
                .any(|theirs| ours.conflicts_with(theirs)),
            Self::SegmentEdit(segment) => theirs.edited_segments.contains(segment),
            Self::OverlaySuperseded { fragment, field } => {
                theirs.superseded_overlays.contains(&(*fragment, *field))
            }
        }
    }

    /// What broke, for a conflict's reason.
    fn describe(&self) -> String {
        match self {
            Self::Inserts => "rows were inserted".into(),
            Self::ValueWrites(field) => format!("values were written for field {field}"),
            Self::CollidingKeys(_) => "a key this one asserts absent may have been inserted".into(),
            Self::RelocationFrom(fragment) => {
                format!("rows were moved out of fragment {fragment}, whose files this one writes")
            }
            Self::RowIds(_) => "a new fragment carries some of the same row ids".into(),
            Self::IndexWrite(write) => format!("index '{}' was written", write.name),
            Self::SegmentEdit(segment) => format!("index segment {segment} was edited"),
            Self::OverlaySuperseded { fragment, field } => format!(
                "new values for field {field} in fragment {fragment} tombstone the overlay \
                 this one adds"
            ),
        }
    }
}

/// What a transaction reads and writes, as far as conflict resolution is
/// concerned.
///
/// Pending, its parts are checked against a [`Window`], and its forbids
/// against what each committed transaction did. Committed, only what it did
/// is read. Derived from the actions or the operation at conflict time and
/// never serialized, so the rule can change without a format change.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Footprint {
    /// Every part this transaction read or writes.
    parts: HashSet<Part>,
    /// What it did that the manifests do not show.
    did: Did,
    /// What the committed transactions must not have done.
    forbids: Vec<Forbid>,
}

/// A claim about which keys a transaction inserts, and over which columns.
#[derive(Debug, Clone, PartialEq)]
struct KeyAssertion {
    key_fields: Vec<FieldRef>,
    filter: KeyExistenceFilter,
}

/// What a transaction writes into one logical index.
///
/// An index is the set of segments sharing a name, and the query path unions
/// them, so two writers may extend one index at the same time -- what they may
/// not do is describe the same rows twice, or disagree about what the index is.
#[derive(Debug, Clone, PartialEq)]
struct IndexWrite {
    name: String,
    /// What the transaction says the index is. `None` for an action that
    /// edits a segment without restating the index's definition.
    identity: Option<IndexIdentity>,
    /// The committed fragments it brings under the index, or `None` when that
    /// is not stated -- what the system indices carry, and what a rewrite of
    /// the whole index reaches.
    adds: Option<HashSet<u64>>,
}

/// What an index is, for the purpose of deciding whether two writers are
/// building the same one.
///
/// Keyed fields, carried fields, and the index details compared byte for byte.
/// `lance-table` cannot tell which details matter to a query, so any
/// difference counts; a conflict is only a retry, so being strict costs
/// little. The index version is left out: a newer library may write segments
/// of the same index.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct IndexIdentity {
    pub fields: Vec<FieldRef>,
    /// Which of `fields` are merely carried: two writers that disagree about
    /// this disagree about what the index answers for, not just what it holds.
    pub covering_fields: Vec<FieldRef>,
    pub index_details: Option<Arc<prost_types::Any>>,
}

impl IndexWrite {
    fn conflicts_with(&self, other: &Self) -> bool {
        if self.name != other.name {
            return false;
        }
        if let (Some(ours), Some(theirs)) = (&self.identity, &other.identity)
            && ours != theirs
        {
            return true;
        }
        match (&self.adds, &other.adds) {
            (Some(ours), Some(theirs)) => !ours.is_disjoint(theirs),
            // An unstated reach could be any fragment, including one the other
            // side is covering.
            _ => true,
        }
    }
}

/// What it takes for a pending transaction to land on the latest version.
#[derive(Debug)]
pub enum Verdict {
    /// Nothing it depends on changed; it lands as it is.
    Compatible,
    /// It has to be planned again. Says what changed.
    Conflict(String),
}

impl Footprint {
    /// What `transaction` reads and writes. An action set's footprint is its
    /// actions'. A named operation has no actions to ask, so what it touches
    /// cannot be stated: it is opaque, and conflicts with everything.
    pub fn of(transaction: &Transaction) -> Self {
        match &transaction.operation {
            Operation::CompositeOperation(composite_operation) => Self::from(composite_operation),
            _ => {
                let mut footprint = Self::default();
                footprint.mark_opaque();
                footprint
            }
        }
    }

    /// Record that this transaction needs `part` exactly as it read it.
    pub(super) fn read(&mut self, part: Part) {
        self.parts.insert(part);
    }

    /// Record that this transaction needs `fragment` still there, but
    /// tolerates any other change to it: values for a field it mints, an
    /// overlay, a widened index segment, a restamp.
    pub(super) fn require_fragment(&mut self, fragment: u64) {
        self.read(Part::FragmentExists(fragment));
    }

    /// Record that this transaction needs `field` to still be in the schema.
    pub(super) fn require_field(&mut self, field: FieldRef) {
        if let Some(field) = committed_field(field) {
            self.read(Part::FieldExists(field));
        }
    }

    /// Record that this transaction changes `field`'s definition: a rename, a
    /// nullability change, a drop. Any concurrent change to it collides, a
    /// child added under it included.
    pub(super) fn write_field_definition(&mut self, field: FieldRef) {
        if let Some(field) = committed_field(field) {
            self.read(Part::FieldDefinition(field));
        }
    }

    /// Record that this transaction writes `fields`' data in `fragment`.
    ///
    /// Each committed field must still be there, in a fragment this
    /// transaction mints as in a committed one: apply rejects a file naming a
    /// field the schema no longer has.
    ///
    /// In a committed fragment, each committed field's backing is written,
    /// so two writers of one field's data collide. A field this transaction
    /// mints writes nothing a concurrent writer can name, but its cells are
    /// lost if the fragment is, so the fragment must stay.
    ///
    /// New values (`data_change`) are prepared against the schema the
    /// transaction read, so they need [`Part::NewRowSchema`].
    pub(super) fn write_field_data(
        &mut self,
        fragment: Ref,
        fields: impl IntoIterator<Item = FieldRef>,
        data_change: bool,
    ) {
        let fields: Vec<i32> = fields.into_iter().filter_map(committed_field).collect();
        if data_change && !fields.is_empty() {
            self.read(Part::NewRowSchema);
        }
        for field in fields.iter().copied() {
            self.read(Part::FieldExists(field));
        }
        let Some(fragment) = fragment.committed() else {
            return;
        };
        self.require_fragment(fragment);
        self.forbids.push(Forbid::RelocationFrom(fragment));
        for field in fields.iter().copied() {
            self.read(Part::FieldData { fragment, field });
        }
        if data_change {
            self.did.wrote_values.extend(fields);
        }
    }

    /// Record that this transaction overlays some of the cells of each of
    /// `fields` in `fragment`, leaving the rest as they were.
    ///
    /// Overlays commute with each other and with a rewrite of the data
    /// beneath: whichever lands second, the overlay is the newer value and
    /// stays on top. So only the fragment's rows must stay where they are.
    pub(super) fn write_overlays(
        &mut self,
        fragment: Ref,
        fields: impl IntoIterator<Item = FieldRef>,
    ) {
        let fields: Vec<i32> = fields.into_iter().filter_map(committed_field).collect();
        if !fields.is_empty() {
            self.read(Part::NewRowSchema);
        }
        for field in fields.iter().copied() {
            self.read(Part::FieldExists(field));
        }
        if let Some(fragment) = fragment.committed() {
            self.require_fragment(fragment);
            self.forbids.push(Forbid::RelocationFrom(fragment));
            self.forbids
                .extend(fields.iter().map(|field| Forbid::OverlaySuperseded {
                    fragment,
                    field: *field,
                }));
            self.did.wrote_values.extend(fields);
        }
    }

    /// Record that this transaction's new base values for `fields` in
    /// `fragment` tombstone every overlay on them as they land, an overlay
    /// committed since it read included. So no overlay may have landed on
    /// those cells since, and none may land after it (see
    /// [`Forbid::OverlaySuperseded`]).
    ///
    /// TODO: an overlay committed after this transaction read is newer than
    /// its values. `TombstoneFieldData` already keeps such an overlay; once
    /// `AddDataFile` does too, the two can commute.
    pub(super) fn supersede_overlays(
        &mut self,
        fragment: Ref,
        fields: impl IntoIterator<Item = FieldRef>,
    ) {
        let Some(fragment) = fragment.committed() else {
            return;
        };
        for field in fields.into_iter().filter_map(committed_field) {
            self.did.superseded_overlays.insert((fragment, field));
            self.read(Part::Cells { fragment, field });
        }
    }

    /// Record that this transaction adds deletions to `fragment` and removes
    /// none, and needs the rest of the fragment only to still be there.
    pub(super) fn add_deletions(&mut self, fragment: u64) {
        self.require_fragment(fragment);
        self.read(Part::Deletions(fragment));
    }

    /// Record that this transaction removes `fragment` outright, deleting its
    /// rows. Overlays added over the rows and row versions restamped are
    /// discarded along with them. A new value in a data file is not discarded
    /// so quietly, since a reader may have expected the deletion to see it,
    /// nor is a change to the deletions.
    pub(super) fn remove_fragment(&mut self, fragment: u64) {
        self.require_fragment(fragment);
        self.read(Part::DataFiles(fragment));
        self.read(Part::Deletions(fragment));
    }

    /// Record that this transaction removes `fragment` after moving its rows
    /// into a fragment of its own, as a compaction does. The rows carry
    /// whatever the fragment held when it read, so nothing written into it
    /// since can be discarded.
    pub(super) fn move_rows_out_of(&mut self, fragment: u64) {
        self.read_whole_fragment(fragment);
    }

    /// Record that this transaction adds a fragment, whose rows were prepared
    /// against the schema it read, and whether its rows are new ones.
    pub(super) fn add_rows(&mut self, inserts: bool) {
        self.read(Part::NewRowSchema);
        self.did.inserted_rows |= inserts;
    }

    /// Record row ids a fragment this transaction adds arrives carrying.
    pub(super) fn supply_row_ids(&mut self, row_ids: &RoaringTreemap) {
        self.did.supplied_row_ids |= row_ids;
        self.forbids.push(Forbid::RowIds(row_ids.clone()));
    }

    /// Record that this transaction makes rows carry a value they may not
    /// have carried before: a non-nullable column added, a column made
    /// required. Rows inserted since it read are not covered by what it read
    /// or wrote.
    pub(super) fn add_required_column(&mut self) {
        self.forbids.push(Forbid::Inserts);
    }

    /// Record that this transaction only holds if no values were written for
    /// `field` since it read it, such as a stricter nullability the apply does
    /// not check the values against. A field it mints has no concurrent
    /// writers, so it needs nothing.
    pub(super) fn require_no_values_written(&mut self, field: FieldRef) {
        if let Some(field) = committed_field(field) {
            self.forbids.push(Forbid::ValueWrites(field));
        }
    }

    /// Record that this transaction adds a segment to `name`, defining the
    /// index as `identity` and describing `coverage`.
    pub(super) fn claim_index_segment(
        &mut self,
        name: String,
        identity: IndexIdentity,
        coverage: Option<impl IntoIterator<Item = Ref>>,
    ) {
        self.write_index(IndexWrite {
            name,
            identity: Some(identity),
            adds: coverage.map(committed_fragments),
        });
    }

    /// Record that this transaction replaces or drops the index `name`
    /// wholesale, reaching whatever segments it holds when it applies.
    pub(super) fn rewrite_index(&mut self, name: String) {
        self.write_index(IndexWrite {
            name,
            identity: None,
            adds: None,
        });
    }

    /// Record that this transaction brings `fragments` under `name` by
    /// widening a segment that is already there.
    pub(super) fn extend_index_coverage(
        &mut self,
        name: String,
        fragments: impl IntoIterator<Item = Ref>,
    ) {
        self.write_index(IndexWrite {
            name,
            identity: None,
            adds: Some(committed_fragments(fragments)),
        });
    }

    /// Record that this transaction brings committed `fragment` under an
    /// index built from the data of `fields` as it read it: a segment over
    /// it would describe values a rewrite since replaced, and a reader has no
    /// way to tell. See [`Part::IndexedFieldData`].
    pub(super) fn require_indexed_field_data(
        &mut self,
        fragment: u64,
        fields: impl IntoIterator<Item = FieldRef>,
    ) {
        for field in fields.into_iter().filter_map(committed_field) {
            self.read(Part::IndexedFieldData { fragment, field });
        }
    }

    /// Record that this transaction removes or edits the segment `uuid`.
    pub(super) fn write_segment(&mut self, uuid: Uuid) {
        self.did.edited_segments.insert(uuid);
        self.forbids.push(Forbid::SegmentEdit(uuid));
    }

    /// Record that the rows this transaction inserts carry keys `filter`
    /// describes, which no concurrent commit may have inserted.
    ///
    /// Only an assertion whose key columns are all committed is recorded. A
    /// local key field names a column this transaction mints, which no
    /// concurrent writer can insert into, and a local token means a different
    /// column in each transaction, so it could not be matched either.
    pub(super) fn assert_unique_keys(
        &mut self,
        key_fields: Vec<FieldRef>,
        filter: KeyExistenceFilter,
    ) {
        if key_fields.iter().any(|field| field.committed().is_none()) {
            return;
        }
        let assertion = Arc::new(KeyAssertion { key_fields, filter });
        self.forbids.push(Forbid::CollidingKeys(assertion.clone()));
        self.did.inserted_keys.push(assertion);
    }

    /// Record that this transaction read `fields` in `fragments` and needs
    /// what a reader sees of them unchanged. A deletion does not count, since
    /// a deleted row's value is invisible. Fragment granularity, which
    /// over-rejects a selection of some of a fragment's rows.
    pub(super) fn assert_cells_unchanged(&mut self, fields: &[i32], fragments: &[u64]) {
        for fragment in fragments.iter().copied() {
            // A row-moving update changes only the deletions here and adds the
            // rows elsewhere, with new values no cell of this fragment shows.
            self.forbids.push(Forbid::RelocationFrom(fragment));
            for field in fields.iter().copied() {
                self.read(Part::Cells { fragment, field });
            }
        }
    }

    /// Record an edit to one of the manifest's string maps: the keys it names,
    /// or the whole map when it replaces rather than merges.
    pub(super) fn write_map_update(&mut self, map: ConfigMap, update: &UpdateMap) {
        if update.replace {
            self.read(Part::Map(map));
            return;
        }
        for entry in &update.update_entries {
            self.read(Part::MapEntry {
                map: map.clone(),
                key: entry.key.clone(),
            });
        }
    }

    /// Record that this transaction adds `base`. Its name, where it has one,
    /// and its location must both still be free.
    pub(super) fn add_base(&mut self, base: &BasePath) {
        if let Some(name) = &base.name {
            self.read(Part::BaseName(name.clone()));
        }
        self.read(Part::BaseLocation(base.path.clone()));
    }

    /// Mark this transaction as one whose footprint cannot be stated.
    pub(super) fn mark_opaque(&mut self) {
        self.did.opaque = true;
    }

    /// Mark this transaction as rewriting the whole table.
    ///
    /// The MemWAL index is claimed as well. A reset drops every index, which
    /// for an ordinary index loses nothing a rebuild cannot recover, but the
    /// MemWAL index records compaction progress that exists nowhere else.
    pub(super) fn reset_table(&mut self) {
        self.did.reset_table = true;
        self.rewrite_index(MEM_WAL_INDEX_NAME.to_string());
    }

    /// Record that this transaction needs every piece of `fragment` as it
    /// read it.
    fn read_whole_fragment(&mut self, fragment: u64) {
        self.read(Part::FragmentExists(fragment));
        self.read(Part::DataFiles(fragment));
        self.read(Part::Overlays(fragment));
        self.read(Part::RowVersions(fragment));
        self.read(Part::Deletions(fragment));
    }

    /// Record that this transaction moves some of `fragment`'s rows into a
    /// fragment of its own. See [`Did::relocated_rows_from`]. The fragment's
    /// files are what the moved rows are copied from, so they must be as read.
    fn relocate_rows_out_of(&mut self, fragment: u64) {
        self.did.relocated_rows_from.insert(fragment);
        self.read(Part::DataFiles(fragment));
        self.read(Part::Overlays(fragment));
    }

    /// Record a write into a logical index, which no concurrent write it
    /// conflicts with may have landed before.
    fn write_index(&mut self, write: IndexWrite) {
        self.did.index_writes.push(write.clone());
        self.forbids.push(Forbid::IndexWrite(write));
    }
}

/// A schema field id a concurrent writer could also be naming, or `None` for
/// a field this transaction mints, which no one else can see yet. A reserved
/// id is `None` too: a lineage column belongs to a fragment this transaction
/// adds, so no concurrent writer can name it either.
fn committed_field(reference: FieldRef) -> Option<i32> {
    reference.committed().filter(|id| *id >= 0)
}

/// `field`, its ancestors, and its descendants in `schema`: a write of any of
/// them may change whether `field` holds a null.
fn related_fields(schema: &Schema, field: i32) -> Vec<i32> {
    let Some(ancestry) = schema.field_ancestry_by_id(field) else {
        return vec![field];
    };
    let mut related: Vec<i32> = ancestry.iter().map(|field| field.id).collect();
    let mut descendants: Vec<&Field> = ancestry
        .last()
        .map(|field| field.children.iter().collect())
        .unwrap_or_default();
    while let Some(descendant) = descendants.pop() {
        related.push(descendant.id);
        descendants.extend(descendant.children.iter());
    }
    related
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

        // Dropping a field and adding one by the same name is how a cast is
        // spelled: the new field's data is rewritten from the dropped field's
        // values as they were read, so values written since would be lost.
        // Names are not matched, so any drop alongside any add counts; the
        // cost is a spurious conflict for an unrelated drop and add in one set.
        let adds_field = composite_operation
            .iter_actions()
            .any(|action| matches!(action, Action::AddField(_)));
        if adds_field {
            for action in composite_operation.iter_actions() {
                if let Action::DropField(drop) = action {
                    footprint.require_no_values_written(drop.field());
                }
            }
        }

        // A row-moving update is spelled as deletions plus the rows written
        // anew, and nothing says which added rows came from which deleted
        // ones. So every fragment a set that inserts rows deletes from, or
        // deletes outright, is taken to have had rows moved out. The cost is a
        // spurious conflict between an unrelated delete and append in one set
        // and a concurrent write to the deleted-from fragment.
        if footprint.did.inserted_rows {
            let deleted_from: Vec<u64> = composite_operation
                .iter_actions()
                .filter_map(|action| match action {
                    Action::SetDeletionFile(set) => Some(set.fragment()),
                    Action::RemoveFragment(remove) if remove.is_data_change() => {
                        remove.fragment().committed()
                    }
                    _ => None,
                })
                .collect();
            for fragment in deleted_from {
                footprint.relocate_rows_out_of(fragment);
            }
        }
        footprint
    }
}

#[cfg(test)]
mod tests;
