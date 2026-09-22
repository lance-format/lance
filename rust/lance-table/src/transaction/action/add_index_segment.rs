// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Add an index segment.

use super::apply::ApplyState;
use super::footprint::IndexIdentity;
use super::proto::{data_change_from_wire, data_change_to_wire, non_empty, required};
use super::{Footprint, Ref};
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
    /// The fields the segment is keyed on -- the columns it can be searched
    /// by. Empty only for a system index keyed on no column at all, such as
    /// `mem_wal` or `frag_reuse`.
    ///
    /// Keys only, unlike [`IndexMetadata::fields`] under the legacy contract,
    /// where it means keyed columns followed by carried ones. `apply` derives
    /// that representation.
    pub fields: Vec<Ref>,
    /// The fields whose values the segment carries but is not necessarily
    /// keyed on, independent of [`Self::fields`] rather than a subset of it.
    /// Empty for a segment that carries no extra column.
    ///
    /// A field in both lists is one the segment is keyed on *and* carries.
    /// That needs `FLAG_INDEPENDENT_COVERING_FIELDS`, which no release
    /// implements, so `apply` rejects it for now.
    pub covering_fields: Vec<Ref>,
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

        let covering_fields = self
            .covering_fields
            .iter()
            .map(|field| state.resolve_field(*field))
            .collect::<Result<Vec<_>>>()?;
        let (fields, covering_fields) = Self::lower_covering(&self.name, fields, covering_fields)?;

        let metadata = IndexMetadata {
            uuid: self.uuid,
            name: self.name.clone(),
            fields,
            covering_fields,
            dataset_version: self.reflected_version(state.read_version())?,
            fragment_bitmap,
            index_details: self.index_details.clone(),
            index_version: self.index_version,
            created_at: self.created_at,
            base_id,
            files: (!self.files.is_empty()).then(|| self.files.clone()),
        };
        // Belt and braces: `lower_covering` builds the legacy form by
        // construction, so a failure here means the two have drifted apart.
        metadata.validate_covering_fields()?;

        state.add_index_segment(metadata)
    }

    /// Render the action's independent key/covering declaration as the
    /// `IndexMetadata` form a manifest can carry today.
    ///
    /// The action names keys and carried columns separately (the contract in
    /// lance-format/lance#9159), while a manifest without
    /// `FLAG_INDEPENDENT_COVERING_FIELDS` carries the legacy form: one
    /// `fields` list of keyed columns followed by carried ones, with
    /// `covering_fields` naming that trailing subset.
    ///
    /// A disjoint declaration converts exactly. An overlapping one -- a column
    /// both keyed and carried -- has no legacy representation and is rejected
    /// rather than silently flattened, which would drop the fact that the
    /// column is carried. Both this lowering and the rejection come out when a
    /// release implements the flag; the action's own shape does not change.
    fn lower_covering(
        name: &str,
        keys: Vec<i32>,
        covering: Vec<i32>,
    ) -> Result<(Vec<i32>, Vec<i32>)> {
        if covering.is_empty() {
            return Ok((keys, covering));
        }

        let overlapping: Vec<i32> = covering
            .iter()
            .copied()
            .filter(|field| keys.contains(field))
            .collect();
        if !overlapping.is_empty() {
            return Err(Error::not_supported(format!(
                "index '{name}' declares fields {overlapping:?} as both keyed and \
                 covering; that requires FLAG_INDEPENDENT_COVERING_FIELDS, which no \
                 release implements yet (keys {keys:?}, covering {covering:?})"
            )));
        }

        let mut lowered = keys;
        lowered.extend_from_slice(&covering);
        Ok((lowered, covering))
    }

    /// The version this segment reflects, defaulting to the one the operation
    /// reads.
    fn reflected_version(&self, read_version: u64) -> Result<u64> {
        let Some(version) = self.dataset_version else {
            return Ok(read_version);
        };
        if version > read_version {
            return Err(Error::invalid_input(format!(
                "AddIndexSegment for index '{}' reflects dataset version {version}, which is \
                 newer than the version {read_version} this operation reads; a segment cannot \
                 reflect data it could not have seen",
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

    /// A claim on the logical index -- this is what the index is, and these are
    /// the fragments the new segment describes -- plus a requirement on the
    /// data the segment was built from.
    ///
    /// An index is derived state whose segments the query path unions, so two
    /// writers may extend the same index at once. What they may not do is
    /// describe the same fragment twice, which would double-count rows, or
    /// disagree about what the index is.
    ///
    /// A segment that has fallen behind is pruned rather than being wrong, but
    /// only the commit that does the changing can prune: a rewrite rebinds the
    /// field and drops it from every segment the manifest carries. A segment
    /// arriving *after* the rewrite prunes nothing, and the format gives a
    /// reader no way to catch it -- an in-place column rewrite "cannot be
    /// detected just by examining metadata", so a reader trusts
    /// `fragment_bitmap` as it finds it. Requiring the data keeps that trust
    /// earned. Deletions and overlays are excluded on purpose: both leave a
    /// trace a reader can act on, so both may still run alongside a build.
    pub(super) fn footprint(&self, footprint: &mut Footprint) {
        footprint.build_index(
            self.name.clone(),
            IndexIdentity {
                fields: self.fields.clone(),
                covering_fields: self.covering_fields.clone(),
                details: self.index_details.clone(),
                index_version: self.index_version,
            },
            self.covered_fragments.clone(),
        );

        // Keyed and carried columns alike: a carried column can be rewritten
        // while the keyed one is untouched, and the segment would then answer
        // from an obsolete carried value.
        let dependencies: Vec<Ref> = self
            .fields
            .iter()
            .chain(self.covering_fields.iter())
            .copied()
            .collect();
        // The segment describes values in the type the schema named when it
        // was built. A cast landing first would leave it describing values as
        // something they are not, and a drop would leave it over a field the
        // manifest no longer has; both write the definition. Landing second,
        // either prunes or discards the segment as it applies, so this is only
        // required, not written.
        for field in dependencies.iter().copied() {
            footprint.require_field_definition(field);
        }
        let Some(covered) = &self.covered_fragments else {
            // Unstated reach. The claim already collides with every other claim
            // on the index; there is no fragment list to require.
            return;
        };
        for fragment in covered.iter().filter_map(|fragment| fragment.committed()) {
            footprint.require_field_data(fragment, dependencies.iter().copied());
        }
    }
}

impl DeepSizeOf for AddIndexSegment {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        // `prost_types::Any` does not implement DeepSizeOf, but it is only a
        // type url and a byte string, so both are measured directly.
        let index_details = self.index_details.as_ref().map_or(0, |details| {
            std::mem::size_of::<prost_types::Any>()
                + details.type_url.capacity()
                + details.value.capacity()
        });
        self.uuid.as_bytes().deep_size_of_children(context)
            + self.name.deep_size_of_children(context)
            + self.fields.deep_size_of_children(context)
            + self.covering_fields.deep_size_of_children(context)
            + index_details
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
            covering_fields: value
                .covering_fields
                .iter()
                .map(|field| (*field).into())
                .collect(),
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
            name: non_empty(message.name, "AddIndexSegment.name")?,
            fields: message
                .fields
                .into_iter()
                .map(Ref::try_from)
                .collect::<Result<Vec<_>>>()?,
            covering_fields: message
                .covering_fields
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
    use crate::transaction::action::{
        Action, AddField, AddFragment, AlterField, CompositeOperation, DropField, Footprint,
        RemoveFragment, TombstoneFieldData, UserAction,
    };
    use crate::transaction::test_support::{default_build_config, sample_index_metadata};
    use crate::transaction::{Operation, Transaction};
    use rstest::rstest;

    fn segment(name: &str, fields: Vec<Ref>) -> AddIndexSegment {
        AddIndexSegment {
            uuid: Uuid::new_v4(),
            name: name.into(),
            fields,
            covering_fields: Vec::new(),
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

    /// A segment that does not say what it reflects reflects the version its
    /// writer read -- not whatever version the set ends up landing on. The two
    /// are the same only when the commit is uncontended; after a lost race the
    /// manifest has moved on while the segment has not.
    #[test]
    fn test_a_relocated_segment_reflects_the_version_it_was_built_against() {
        // Stand in for a retry: the set was built against `read_version` and is
        // being applied to a manifest two commits further along.
        let mut manifest = backed_manifest();
        manifest.version += 2;
        let read_version = manifest.version - 2;

        let transaction = Transaction::new(
            read_version,
            Operation::CompositeOperation(CompositeOperation::new(vec![UserAction::new(
                "step",
                vec![Action::AddIndexSegment(segment(
                    "by_a",
                    vec![Ref::Committed(0)],
                ))],
            )])),
            None,
        );
        let (_, indices) = transaction
            .build_manifest(
                Some(&manifest),
                Vec::new(),
                "tx.txn",
                &default_build_config(),
            )
            .unwrap();

        assert_eq!(indices[0].dataset_version, read_version);
    }

    /// The bound on an explicit `dataset_version` is the version the set read,
    /// for the same reason: a segment cannot reflect data its writer could not
    /// have seen, however far the manifest has moved since.
    #[test]
    fn test_a_segment_cannot_reflect_a_version_committed_after_the_one_it_read() {
        let mut manifest = backed_manifest();
        manifest.version += 2;
        let read_version = manifest.version - 2;

        let transaction = Transaction::new(
            read_version,
            Operation::CompositeOperation(CompositeOperation::new(vec![UserAction::new(
                "step",
                vec![Action::AddIndexSegment(AddIndexSegment {
                    dataset_version: Some(read_version + 1),
                    ..segment("by_a", vec![Ref::Committed(0)])
                })],
            )])),
            None,
        );
        let error = transaction
            .build_manifest(
                Some(&manifest),
                Vec::new(),
                "tx.txn",
                &default_build_config(),
            )
            .unwrap_err();

        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
        assert!(error.to_string().contains("could not have seen"), "{error}");
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

    /// The action names keys and carried columns independently, but a manifest
    /// can only carry the legacy form -- one `fields` list with the carried
    /// columns as its tail -- until a release implements
    /// `FLAG_INDEPENDENT_COVERING_FIELDS`. Apply converts between the two.
    #[test]
    fn test_a_disjoint_covering_declaration_lowers_to_the_legacy_form() {
        let (next, indices) = apply_with_indices(
            &backed_manifest(),
            vec![
                Action::AddField(AddField {
                    local: 0,
                    parent: None,
                    def: added_field("carried"),
                }),
                Action::AddIndexSegment(AddIndexSegment {
                    covering_fields: vec![Ref::Local(0)],
                    ..segment("by_a", vec![Ref::Committed(0)])
                }),
            ],
            Vec::new(),
        )
        .unwrap();

        let carried = next.schema.field("carried").unwrap().id;
        assert_eq!(indices[0].fields, vec![0, carried]);
        assert_eq!(indices[0].covering_fields, vec![carried]);
    }

    /// A column both keyed and carried has no legacy representation: dropping
    /// it from `covering_fields` would leave the manifest claiming only that
    /// the index is keyed on the column, losing the fact that it also serves
    /// the column's values. Reject rather than publish the weaker claim.
    #[test]
    fn test_a_covering_field_that_is_also_a_key_is_rejected_for_now() {
        let error = apply_with_indices(
            &backed_manifest(),
            vec![Action::AddIndexSegment(AddIndexSegment {
                covering_fields: vec![Ref::Committed(0)],
                ..segment("by_a", vec![Ref::Committed(0)])
            })],
            Vec::new(),
        )
        .unwrap_err();

        assert!(matches!(error, Error::NotSupported { .. }), "{error:?}");
        assert!(
            error
                .to_string()
                .contains("FLAG_INDEPENDENT_COVERING_FIELDS"),
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
                    id: Ref::Local(0),
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

    /// Two builds over one column do not collide: a requirement is a claim
    /// that nothing moved, not a claim on the column.
    #[test]
    fn test_two_indices_over_one_column_and_fragment_do_not_conflict() {
        let ours = index_footprint(covering("by_a", Some(vec![0])));
        let theirs = index_footprint(covering("by_b", Some(vec![0])));

        assert!(!ours.conflicts_with(&theirs));
        assert!(!theirs.conflicts_with(&ours));
    }

    /// The direction is the whole point, so it gets its own test.
    ///
    /// A segment requires the data it describes; a rewrite of that data writes
    /// it. Arriving after the rewrite, the segment would publish coverage of
    /// values it never saw and a reader has no way to detect that, so it must
    /// be rejected. Arriving before, the rewrite prunes the segment's coverage
    /// as it applies, so rejecting it would be a conflict over nothing.
    #[test]
    fn test_a_build_loses_to_a_committed_rewrite_but_not_the_reverse() {
        let build = index_footprint(covering("by_a", Some(vec![0])));
        let rewrite = Footprint::from(&CompositeOperation::new(vec![UserAction::new(
            "step",
            vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![Ref::Committed(0)],
                data_change: true,
            })],
        )]));

        assert!(
            build.conflicts_with(&rewrite),
            "a segment cannot describe values a committed rewrite replaced"
        );
        assert!(
            !rewrite.conflicts_with(&build),
            "the rewrite prunes the committed segment's coverage as it applies"
        );
    }

    /// A rewrite of a column the segment merely carries invalidates it just as
    /// a keyed one does -- the segment would answer from an obsolete carried
    /// value. `covering_fields` is independent of `fields`, so the requirement
    /// is over the union.
    #[test]
    fn test_a_build_requires_the_columns_it_carries_too() {
        let mut segment = covering("by_a", Some(vec![0]));
        segment.covering_fields = vec![Ref::Committed(1)];
        let build = index_footprint(segment);

        let rewrite = Footprint::from(&CompositeOperation::new(vec![UserAction::new(
            "step",
            vec![Action::TombstoneFieldData(TombstoneFieldData {
                fragment: Ref::Committed(0),
                field_ids: vec![Ref::Committed(1)],
                data_change: true,
            })],
        )]));

        assert!(build.conflicts_with(&rewrite));
    }

    /// A cast landing first leaves the segment describing values in a type the
    /// schema no longer names; a cast landing second rebinds the field in every
    /// fragment and prunes the segment's coverage as it applies. Same shape as
    /// the rewrite case above, one level up: the definition rather than the
    /// data.
    #[test]
    fn test_a_build_loses_to_a_committed_cast_but_not_the_reverse() {
        let build = index_footprint(covering("by_a", Some(vec![0])));
        let cast = Footprint::from(&CompositeOperation::new(vec![UserAction::new(
            "step",
            vec![Action::AlterField(AlterField {
                field: Ref::Committed(0),
                name: None,
                logical_type: Some("int64".into()),
                nullable: None,
            })],
        )]));

        assert!(build.conflicts_with(&cast));
        assert!(!cast.conflicts_with(&build));

        // A segment of unstated reach depends on the definition just the same.
        assert!(index_footprint(covering("by_a", None)).conflicts_with(&cast));
    }

    /// A segment requires the data of the fragments it covers, and that data
    /// is gone if the fragment is: a compaction landing first would leave the
    /// segment describing rows the dataset no longer has, in a fragment the
    /// replacement does not cover. Landing second, the compaction is the one
    /// answerable for the index, and the footprint has nothing to say.
    #[test]
    fn test_a_build_loses_to_a_committed_removal_of_a_covered_fragment() {
        let build = index_footprint(covering("by_a", Some(vec![0])));
        let removal = |fragment: u64| {
            Footprint::from(&CompositeOperation::new(vec![UserAction::new(
                "step",
                vec![Action::RemoveFragment(RemoveFragment {
                    fragment: Ref::Committed(fragment),
                    data_change: false,
                })],
            )]))
        };

        assert!(build.conflicts_with(&removal(0)));
        assert!(!removal(0).conflicts_with(&build));
        assert!(!build.conflicts_with(&removal(1)));
    }

    fn index_footprint(action: AddIndexSegment) -> Footprint {
        Footprint::from(&CompositeOperation::new(vec![UserAction::new(
            "step",
            vec![Action::AddIndexSegment(action)],
        )]))
    }

    fn covering(name: &str, fragments: Option<Vec<u64>>) -> AddIndexSegment {
        AddIndexSegment {
            covered_fragments: fragments
                .map(|ids| ids.into_iter().map(Ref::Committed).collect::<Vec<_>>()),
            ..segment(name, vec![Ref::Committed(0)])
        }
    }

    #[rstest]
    #[case::disjoint_coverage_of_one_index(
        covering("by_a", Some(vec![0, 1])),
        covering("by_a", Some(vec![2, 3])),
        false,
    )]
    #[case::overlapping_coverage_of_one_index(
        covering("by_a", Some(vec![0, 1])),
        covering("by_a", Some(vec![1, 2])),
        true,
    )]
    #[case::different_indices(
        covering("by_a", Some(vec![0, 1])),
        covering("by_b", Some(vec![0, 1])),
        false,
    )]
    #[case::unstated_coverage_reaches_everywhere(
        covering("by_a", None),
        covering("by_a", Some(vec![7])),
        true,
    )]
    #[case::two_segments_of_unstated_coverage(covering("by_a", None), covering("by_a", None), true)]
    #[case::disagreeing_about_which_fields_the_index_is_over(
        AddIndexSegment { fields: vec![Ref::Committed(1)], ..covering("by_a", Some(vec![0])) },
        covering("by_a", Some(vec![2])),
        true,
    )]
    #[case::disagreeing_about_the_index_config(
        AddIndexSegment {
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "type.googleapis.com/lance.index.pb.VectorIndexDetails".into(),
                value: vec![1, 2, 3],
            })),
            ..covering("by_a", Some(vec![0]))
        },
        covering("by_a", Some(vec![2])),
        true,
    )]
    #[case::disagreeing_about_the_index_version(
        AddIndexSegment { index_version: 2, ..covering("by_a", Some(vec![0])) },
        covering("by_a", Some(vec![2])),
        true,
    )]
    fn test_two_writers_extending_one_index(
        #[case] ours: AddIndexSegment,
        #[case] theirs: AddIndexSegment,
        #[case] expected: bool,
    ) {
        let ours = index_footprint(ours);
        let theirs = index_footprint(theirs);
        assert_eq!(ours.conflicts_with(&theirs), expected);
        assert_eq!(theirs.conflicts_with(&ours), expected);
    }

    #[test]
    fn test_two_writers_indexing_what_they_each_just_wrote_do_not_conflict() {
        // Neither segment names a committed fragment, and the fragments they do
        // name get distinct ids once both commits are ordered.
        let local = AddIndexSegment {
            covered_fragments: Some(vec![Ref::Local(0)]),
            ..segment("by_a", vec![Ref::Committed(0)])
        };

        let ours = index_footprint(local.clone());
        let theirs = index_footprint(local);
        assert!(!ours.conflicts_with(&theirs));
    }
}
