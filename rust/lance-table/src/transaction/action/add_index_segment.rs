// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Add an index segment.

use super::apply::ApplyState;
use super::proto::{data_change_from_wire, data_change_to_wire, required};
use super::{Coordinate, Footprint, Ref};
use crate::format::{IndexFile, IndexMetadata, pb};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use roaring::RoaringBitmap;
use std::sync::Arc;
use uuid::Uuid;

/// Add an index segment.
///
/// The format has no first-class "index" apart from its segments: a logical
/// index is the set of segments sharing a `name`, and a brand-new index is
/// written as its first segment. So this one action covers both creating an
/// index and extending an existing one -- the difference is only whether any
/// segment already carries the name.
///
/// The segment's `uuid` is chosen by the writer rather than minted from a
/// counter, so unlike a fragment or a field it needs no [`Ref`]: two writers
/// cannot pick the same one, and replaying the action onto another version
/// leaves it untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct AddIndexSegment {
    /// Identifies the segment, and names the directory its files live in.
    pub uuid: Uuid,
    /// The logical index this segment belongs to.
    pub name: String,
    /// The indexed fields.
    pub fields: Vec<Ref>,
    /// Index-type-specific metadata, opaque to the transaction layer.
    pub index_details: Option<Arc<prost_types::Any>>,
    pub index_version: i32,
    /// The fragments this segment covers, or `None` when the coverage was not
    /// recorded -- what the system indices carry. An empty list is the
    /// different statement that the segment covers nothing.
    pub covered_fragments: Option<Vec<Ref>>,
    /// The segment's files and their sizes, empty when the writer did not
    /// record them.
    pub files: Vec<IndexFile>,
    /// The base path the files live under, for a segment imported from another
    /// dataset. `None` means the dataset's own index directory.
    pub base: Option<Ref>,
    /// When the segment was built, or `None` when the writer did not record
    /// it. Carried rather than derived, because it describes a build that
    /// replaying this action does not redo.
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    /// The dataset version whose data this segment reflects, or `None` for the
    /// version the operation reads -- what a freshly built segment reflects.
    ///
    /// A segment merged from older ones reflects only as much as its oldest
    /// input, so this is not always the read version and is not derivable from
    /// it. The overlay version gate reads it: an overlay committed at or before
    /// it counts as already folded into the index.
    pub dataset_version: Option<u64>,
    pub data_change: bool,
}

impl AddIndexSegment {
    pub(super) fn apply(&self, state: &mut ApplyState) -> Result<()> {
        let fields = self
            .fields
            .iter()
            .map(|field| state.resolve_field(*field))
            .collect::<Result<Vec<_>>>()?;
        let fragment_bitmap = self
            .covered_fragments
            .as_ref()
            .map(|fragments| self.resolve_coverage(fragments, state))
            .transpose()?;
        let base_id = self.base.map(|base| state.resolve_base(base)).transpose()?;

        state.add_index_segment(IndexMetadata {
            uuid: self.uuid,
            name: self.name.clone(),
            fields,
            dataset_version: self.reflected_version(state.read_version())?,
            fragment_bitmap,
            index_details: self.index_details.clone(),
            index_version: self.index_version,
            created_at: self.created_at,
            base_id,
            files: (!self.files.is_empty()).then(|| self.files.clone()),
        })
    }

    /// The version this segment reflects, defaulting to the one the operation
    /// reads.
    fn reflected_version(&self, read_version: u64) -> Result<u64> {
        let Some(version) = self.dataset_version else {
            return Ok(read_version);
        };
        if version > read_version {
            return Err(Error::invalid_input(format!(
                "AddIndexSegment for index '{}' reflects dataset version {version}, which is newer                  than the version {read_version} this operation reads; a segment cannot reflect                  data it could not have seen",
                self.name
            )));
        }
        Ok(version)
    }

    fn resolve_coverage(&self, fragments: &[Ref], state: &ApplyState) -> Result<RoaringBitmap> {
        fragments
            .iter()
            .map(|fragment| {
                let id = state.resolve_fragment(*fragment)?;
                u32::try_from(id).map_err(|_| {
                    Error::invalid_input(format!(
                        "AddIndexSegment for index '{}' covers fragment {id}, which is beyond the \
                         largest fragment id an index can record ({})",
                        self.name,
                        u32::MAX
                    ))
                })
            })
            .collect()
    }

    /// Left to the writer. An index is derived state, so building one normally
    /// changes nothing a reader sees -- but the MemWAL index holds real
    /// unflushed rows, so the answer is not the same for every segment.
    pub(super) fn is_data_change(&self) -> bool {
        self.data_change
    }

    /// The logical index, by name. Two writers may add segments to *different*
    /// indices at once, and may extend one index concurrently with any change to
    /// the data it covers -- an index is derived state, and a segment that has
    /// fallen behind is pruned rather than being wrong. What they may not do is
    /// both build the same index, which would leave two segments each claiming
    /// to cover the same fragments.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.add(Coordinate::IndexName(self.name.clone()));
    }
}

impl DeepSizeOf for AddIndexSegment {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        // `index_details` is an opaque protobuf whose size the deepsize crate
        // cannot reach, matching how `IndexMetadata` accounts for itself.
        self.uuid.as_bytes().deep_size_of_children(context)
            + self.name.deep_size_of_children(context)
            + self.fields.deep_size_of_children(context)
            + self.covered_fragments.deep_size_of_children(context)
            + self.files.deep_size_of_children(context)
    }
}

impl From<&AddIndexSegment> for pb::AddIndexSegment {
    fn from(value: &AddIndexSegment) -> Self {
        Self {
            uuid: Some((&value.uuid).into()),
            name: value.name.clone(),
            fields: value.fields.iter().map(|field| (*field).into()).collect(),
            index_details: value
                .index_details
                .as_ref()
                .map(|details| details.as_ref().clone()),
            index_version: Some(value.index_version),
            covered_fragments: value.covered_fragments.as_ref().map(|fragments| {
                pb::FragmentCoverage {
                    fragments: fragments.iter().map(|id| (*id).into()).collect(),
                }
            }),
            files: value
                .files
                .iter()
                .map(|file| pb::IndexFile {
                    path: file.path.clone(),
                    size_bytes: file.size_bytes,
                })
                .collect(),
            data_change: data_change_to_wire(value.data_change),
            base: value.base.map(pb::Ref::from),
            created_at: value
                .created_at
                .map(|created_at| created_at.timestamp_millis() as u64),
            dataset_version: value.dataset_version,
        }
    }
}

impl TryFrom<pb::AddIndexSegment> for AddIndexSegment {
    type Error = Error;

    fn try_from(message: pb::AddIndexSegment) -> Result<Self> {
        let created_at = message
            .created_at
            .map(|millis| {
                chrono::DateTime::from_timestamp_millis(millis as i64).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "AddIndexSegment.created_at is {millis}ms since the epoch, which is not a \
                         representable timestamp"
                    ))
                })
            })
            .transpose()?;

        Ok(Self {
            uuid: Uuid::try_from(&required(message.uuid, "AddIndexSegment.uuid")?)?,
            name: message.name,
            fields: message
                .fields
                .into_iter()
                .map(Ref::try_from)
                .collect::<Result<Vec<_>>>()?,
            index_details: message.index_details.map(Arc::new),
            index_version: message.index_version.unwrap_or_default(),
            covered_fragments: message
                .covered_fragments
                .map(|coverage| {
                    coverage
                        .fragments
                        .into_iter()
                        .map(Ref::try_from)
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?,
            files: message
                .files
                .into_iter()
                .map(|file| IndexFile {
                    path: file.path,
                    size_bytes: file.size_bytes,
                })
                .collect(),
            base: message.base.map(Ref::try_from).transpose()?,
            created_at,
            dataset_version: message.dataset_version,
            data_change: data_change_from_wire(message.data_change),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transaction::action::test_support::{
        added_field, apply_with_indices, backed_manifest,
    };
    use crate::transaction::action::{Action, AddField, AddFragment, DropField};
    use crate::transaction::test_support::sample_index_metadata;

    fn segment(name: &str, fields: Vec<Ref>) -> AddIndexSegment {
        AddIndexSegment {
            uuid: Uuid::new_v4(),
            name: name.into(),
            fields,
            index_details: None,
            index_version: 1,
            covered_fragments: Some(vec![Ref::Committed(0)]),
            files: Vec::new(),
            base: None,
            created_at: None,
            dataset_version: None,
            data_change: false,
        }
    }

    #[test]
    fn test_add_index_segment_records_coverage_and_the_version_it_describes() {
        let manifest = backed_manifest();
        let action = AddIndexSegment {
            files: vec![IndexFile {
                path: "index.idx".into(),
                size_bytes: 1024,
            }],
            ..segment("by_a", vec![Ref::Committed(0)])
        };

        let (next, indices) = apply_with_indices(
            &manifest,
            vec![Action::AddIndexSegment(action.clone())],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(indices.len(), 1);
        let index = &indices[0];
        assert_eq!(index.uuid, action.uuid);
        assert_eq!(index.name, "by_a");
        assert_eq!(index.fields, vec![0]);
        assert_eq!(index.fragment_bitmap, Some([0].into_iter().collect()));
        assert_eq!(index.files, Some(action.files));
        // A segment that does not say what it reflects reflects the version the
        // operation read, so replaying it elsewhere restamps it.
        assert_eq!(index.dataset_version, manifest.version);
        assert!(index.dataset_version < next.version);
    }

    #[test]
    fn test_a_merged_segment_keeps_the_older_version_it_reflects() {
        let manifest = backed_manifest();
        let (_, indices) = apply_with_indices(
            &manifest,
            vec![Action::AddIndexSegment(AddIndexSegment {
                dataset_version: Some(manifest.version - 1),
                ..segment("by_a", vec![Ref::Committed(0)])
            })],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(indices[0].dataset_version, manifest.version - 1);
    }

    #[test]
    fn test_a_segment_reflecting_a_future_version_is_rejected() {
        let manifest = backed_manifest();
        let error = apply_with_indices(
            &manifest,
            vec![Action::AddIndexSegment(AddIndexSegment {
                dataset_version: Some(manifest.version + 1),
                ..segment("by_a", vec![Ref::Committed(0)])
            })],
            Vec::new(),
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("could not have seen"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_a_second_segment_extends_the_same_index() {
        let existing = sample_index_metadata("by_a");
        let added = segment("by_a", vec![Ref::Committed(0)]);

        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![Action::AddIndexSegment(added.clone())],
            vec![existing.clone()],
        )
        .unwrap();

        // A logical index is the set of segments sharing a name, so adding one
        // leaves the other in place rather than replacing it.
        assert_eq!(indices.len(), 2);
        let uuids = indices.iter().map(|index| index.uuid).collect::<Vec<_>>();
        assert!(uuids.contains(&existing.uuid));
        assert!(uuids.contains(&added.uuid));
    }

    #[test]
    fn test_re_adding_a_segment_is_rejected() {
        let existing = sample_index_metadata("by_a");
        let error = apply_with_indices(
            &backed_manifest(),
            vec![Action::AddIndexSegment(AddIndexSegment {
                uuid: existing.uuid,
                ..segment("by_a", vec![Ref::Committed(0)])
            })],
            vec![existing],
        )
        .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(
            error.to_string().contains("added once"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_unrecorded_coverage_is_not_the_same_as_covering_nothing() {
        let unknown = AddIndexSegment {
            covered_fragments: None,
            ..segment("system", vec![])
        };
        let empty = AddIndexSegment {
            covered_fragments: Some(Vec::new()),
            ..segment("empty", vec![])
        };

        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![
                Action::AddIndexSegment(unknown),
                Action::AddIndexSegment(empty),
            ],
            Vec::new(),
        )
        .unwrap();

        let bitmap_of = |name: &str| {
            indices
                .iter()
                .find(|index| index.name == name)
                .unwrap()
                .fragment_bitmap
                .clone()
        };
        assert_eq!(bitmap_of("system"), None);
        assert_eq!(bitmap_of("empty"), Some(RoaringBitmap::new()));
    }

    #[test]
    fn test_a_segment_can_cover_a_fragment_minted_in_the_same_operation() {
        // Indexing what an operation just wrote is the point of composing the
        // two, and the fragment has no committed id until this apply runs.
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![
                Action::AddFragment(AddFragment {
                    local: 0,
                    physical_rows: 10,
                    row_id_meta: None,
                    last_updated_at_version_meta: None,
                    created_at_version_meta: None,
                    data_change: true,
                }),
                Action::AddIndexSegment(AddIndexSegment {
                    covered_fragments: Some(vec![Ref::Committed(0), Ref::Local(0)]),
                    ..segment("by_a", vec![Ref::Committed(0)])
                }),
            ],
            Vec::new(),
        )
        .unwrap();

        assert_eq!(
            indices[0].fragment_bitmap,
            Some([0, 1].into_iter().collect())
        );
    }

    #[test]
    fn test_a_segment_can_index_a_field_minted_in_the_same_operation() {
        let (next, indices) = apply_with_indices(
            &backed_manifest(),
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("added"),
                }),
                Action::AddIndexSegment(segment("by_added", vec![Ref::Local(0)])),
            ],
            Vec::new(),
        )
        .unwrap();

        let field = next.schema.field("added").unwrap();
        assert_eq!(indices[0].fields, vec![field.id]);
    }

    #[test]
    fn test_a_segment_over_a_dropped_field_does_not_survive_the_commit() {
        // The assembly prunes indices whose fields left the schema, so an
        // operation that drops a field and indexes it cannot smuggle one in.
        let (_, indices) = apply_with_indices(
            &backed_manifest(),
            vec![
                Action::DropField(DropField {
                    field: Ref::Committed(0),
                }),
                Action::AddIndexSegment(segment("by_a", vec![Ref::Committed(0)])),
            ],
            Vec::new(),
        )
        .unwrap();

        assert!(indices.is_empty());
    }

    #[test]
    fn test_two_writers_building_the_same_index_conflict() {
        use crate::transaction::action::{CompositeOperation, Footprint, UserAction};

        let footprint = |action: AddIndexSegment| {
            Footprint::from(&CompositeOperation::new(vec![UserAction::new(
                "step",
                vec![Action::AddIndexSegment(action)],
            )]))
        };

        let ours = footprint(segment("by_a", vec![Ref::Committed(0)]));
        let same_name = footprint(segment("by_a", vec![Ref::Committed(0)]));
        let other_name = footprint(segment("by_b", vec![Ref::Committed(1)]));

        assert!(ours.conflicts_with(&same_name));
        assert!(!ours.conflicts_with(&other_name));
    }
}
