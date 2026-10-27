// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fixtures shared by the per-action test modules.

use super::{Action, CompositeOperation, Footprint, UserStep};
use crate::format::{BasePath, DataFile, Fragment, IndexMetadata, Manifest};
use crate::transaction::test_support::{default_build_config, sample_manifest};
use crate::transaction::{Operation, Transaction};
use arrow_schema::{DataType, Field as ArrowField};
use lance_core::Result;
use lance_core::datatypes::Field;
use lance_file::version::ConcreteFileVersion;
use std::sync::Arc;

pub(super) fn apply(manifest: &Manifest, actions: Vec<Action>) -> Result<Manifest> {
    apply_with_indices(manifest, actions, Vec::new()).map(|(manifest, _)| manifest)
}

pub(super) fn apply_with_indices(
    manifest: &Manifest,
    actions: Vec<Action>,
    indices: Vec<IndexMetadata>,
) -> Result<(Manifest, Vec<IndexMetadata>)> {
    let transaction = Transaction::new(
        manifest.version,
        Operation::CompositeOperation(CompositeOperation::new(vec![UserStep::new(
            "step", actions,
        )])),
        None,
    );
    transaction.build_manifest(Some(manifest), indices, "tx.txn", &default_build_config())
}

/// The footprint of one action set, for pair tests. Every conflict test in the
/// action modules builds its two sides with this, so a pair reads as two lists
/// of actions and an expected answer.
pub(super) fn footprint(actions: Vec<Action>) -> Footprint {
    Footprint::from(&CompositeOperation::new(vec![UserStep::new(
        "step", actions,
    )]))
}

pub(super) fn added_field(name: &str) -> Field {
    Field::try_from(ArrowField::new(name, DataType::Int32, true)).unwrap()
}

/// A one-column table whose fragment 0 has no data files. The column is
/// nullable, so a fragment may be added without data for it, which lets tests
/// about ids and reservations skip writing any.
pub(super) fn unbacked_manifest() -> Manifest {
    let mut manifest = sample_manifest();
    manifest.schema.field_by_id_mut(0).unwrap().nullable = true;
    manifest
}

/// `unbacked_manifest` with fragment 0 actually backed by a data file, so the
/// reference-stable actions have something committed to point at.
pub(super) fn backed_manifest() -> Manifest {
    let mut manifest = unbacked_manifest();
    let mut fragment = Fragment::new(0);
    fragment.physical_rows = Some(10);
    fragment.files.push(DataFile::new(
        "data/0.lance",
        vec![0],
        vec![0],
        ConcreteFileVersion::V2_0,
        None,
        None,
    ));
    manifest.fragments = Arc::new(vec![fragment]);
    manifest
}

/// `backed_manifest` with one base path besides the dataset's own, at id 1,
/// so an action can name a committed base. A base minted on top of it gets
/// id 2.
pub(super) fn manifest_with_base() -> Manifest {
    let mut manifest = backed_manifest();
    manifest.base_paths.insert(
        1,
        BasePath::new(1, "s3://bucket/other".into(), Some("other".into()), false),
    );
    manifest
}

/// An action minting base token 0 at a location no fixture uses.
pub(super) fn mint_base() -> Action {
    Action::AddBase(super::AddBase {
        local: 0,
        base: BasePath::new(0, "s3://bucket/minted".into(), None, false),
    })
}

/// A v2 data file with one top-level column, as a writer hands it to
/// `AddDataFile`. A v2 file with no column indices stores nothing a reader can
/// find, so it would not back the field it names.
pub(super) fn one_column_file(path: &str) -> DataFile {
    DataFile {
        column_indices: Arc::from([0]),
        ..DataFile::new_unstarted(path, ConcreteFileVersion::V2_0)
    }
}

/// Fixtures for the fragment reuse actions: a table of `fragments` fragments of
/// `ROWS` rows each, every one backed by a file for field 0.
pub(super) mod frag_reuse {
    use super::super::{
        Action, AddDataFile, AddFragment, AppendFragReuseTransition, FieldRef, FragReuseMapping,
        Ref, RemoveFragment,
    };
    use crate::format::pb::fragment_reuse_index_details as pb_fri;
    use crate::format::{DataFile, Fragment, IndexMetadata, Manifest, RowIdMeta, pb};
    use crate::rowids::version::{RowDatasetVersionMeta, RowDatasetVersionSequence};
    use crate::rowids::write_row_ids;
    use crate::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use crate::system_index::frag_reuse::ledger::FragReuseLedger;
    use crate::transaction::test_support::{make_stable_row_id_manifest, sample_manifest};
    use lance_core::utils::address::RowAddress;
    use lance_file::version::ConcreteFileVersion;
    use prost::Message;
    use roaring::RoaringTreemap;
    use std::sync::Arc;
    use uuid::Uuid;

    pub const ROWS: u64 = 10;

    pub fn table(fragments: u64, stable_row_ids: bool) -> Manifest {
        let fragments: Vec<Fragment> = (0..fragments)
            .map(|id| {
                let mut fragment = Fragment::new(id);
                fragment.physical_rows = Some(ROWS as usize);
                fragment.files.push(DataFile::new(
                    format!("data/{id}.lance"),
                    vec![0],
                    vec![0],
                    ConcreteFileVersion::V2_0,
                    None,
                    None,
                ));
                if stable_row_ids {
                    let start = id * ROWS;
                    fragment.row_id_meta = Some(RowIdMeta::Inline(
                        write_row_ids(&(start..start + ROWS).into()).into(),
                    ));
                }
                fragment
            })
            .collect();
        let mut manifest = if stable_row_ids {
            make_stable_row_id_manifest(fragments)
        } else {
            let mut manifest = sample_manifest();
            manifest.fragments = Arc::new(fragments);
            manifest
        };
        manifest.update_max_fragment_id();
        manifest
    }

    /// The surviving-row bitmap of an ordered compaction keeping every row of
    /// `fragments` except `deleted` (fragment, offset) pairs.
    pub fn survivors(fragments: &[u64], deleted: &[(u64, u32)]) -> Vec<u8> {
        let mut bitmap = RoaringTreemap::new();
        for &fragment in fragments {
            for offset in 0..ROWS as u32 {
                if !deleted.contains(&(fragment, offset)) {
                    bitmap.insert(u64::from(RowAddress::new_from_parts(
                        fragment as u32,
                        offset,
                    )));
                }
            }
        }
        let mut bytes = Vec::new();
        bitmap.serialize_into(&mut bytes).unwrap();
        bytes
    }

    pub fn ordered(changed_row_addrs: Vec<u8>) -> FragReuseMapping {
        FragReuseMapping::OrderedCompaction(pb_fri::OrderedCompaction { changed_row_addrs })
    }

    pub fn stable_partition() -> FragReuseMapping {
        FragReuseMapping::StablePartition(pb_fri::StablePartition {
            map_id: Uuid::new_v4().to_string(),
            map_size_bytes: 1,
            base_id: None,
        })
    }

    /// Compact `sources` into one fragment minted under `token`, recording
    /// `mapping` as the transition when one is given.
    pub fn compaction(
        sources: &[u64],
        token: u32,
        stable_row_ids: bool,
        mapping: Option<FragReuseMapping>,
    ) -> Vec<Action> {
        let rows = ROWS * sources.len() as u64;
        let mut actions: Vec<Action> = sources
            .iter()
            .map(|&source| {
                Action::RemoveFragment(RemoveFragment {
                    fragment: Ref::Committed(source),
                    data_change: false,
                })
            })
            .collect();
        let versions = || {
            RowDatasetVersionMeta::from_sequence(
                &RowDatasetVersionSequence::from_uniform_row_count(rows, 1),
            )
            .unwrap()
        };
        let row_ids: Vec<u64> = sources
            .iter()
            .flat_map(|source| source * ROWS..(source + 1) * ROWS)
            .collect();
        actions.push(Action::AddFragment(AddFragment {
            id: Ref::Local(token),
            physical_rows: rows,
            row_id_meta: stable_row_ids
                .then(|| RowIdMeta::Inline(write_row_ids(&row_ids.as_slice().into()).into())),
            last_updated_at_version_meta: stable_row_ids.then(versions),
            created_at_version_meta: stable_row_ids.then(versions),
            data_change: false,
        }));
        actions.push(Action::AddDataFile(AddDataFile {
            fragment: Ref::Local(token),
            file: DataFile::new_unstarted(
                format!("data/compacted-{token}.lance"),
                ConcreteFileVersion::V2_0,
            ),
            field_ids: vec![FieldRef::Committed(0)],
            data_change: false,
            base: None,
        }));
        if let Some(mapping) = mapping {
            actions.push(transition(sources, &[Ref::Local(token)], mapping));
        }
        actions
    }

    pub fn transition(sources: &[u64], destinations: &[Ref], mapping: FragReuseMapping) -> Action {
        Action::AppendFragReuseTransition(AppendFragReuseTransition {
            sources: sources.iter().copied().map(Ref::Committed).collect(),
            destinations: destinations.to_vec(),
            mapping,
        })
    }

    pub fn digest(id: u64, physical_rows: u64) -> pb_fri::FragmentDigest {
        pb_fri::FragmentDigest {
            id,
            physical_rows,
            num_deleted_rows: 0,
        }
    }

    /// An ordered-compaction transition moving every row of `sources` into
    /// `destinations`, as a history records it.
    pub fn recorded(sources: &[u64], destinations: &[u64]) -> pb_fri::Transition {
        let per_destination = ROWS * sources.len() as u64 / destinations.len() as u64;
        pb_fri::Transition {
            sources: sources.iter().map(|&id| digest(id, ROWS)).collect(),
            destinations: destinations
                .iter()
                .map(|&id| digest(id, per_destination))
                .collect(),
            mapping: Some(pb_fri::transition::Mapping::OrderedCompaction(
                pb_fri::OrderedCompaction {
                    changed_row_addrs: survivors(sources, &[]),
                },
            )),
        }
    }

    /// A history entry carrying `legacy_versions` and `transitions` inline.
    pub fn history(
        index_version: i32,
        legacy_versions: Vec<pb_fri::Version>,
        transitions: Vec<pb_fri::Transition>,
    ) -> IndexMetadata {
        let content = pb_fri::InlineContent {
            legacy_versions,
            transitions,
        };
        let mut fragment_bitmap = roaring::RoaringBitmap::new();
        let details = prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: pb::FragmentReuseIndexDetails {
                content: Some(pb::fragment_reuse_index_details::Content::Inline(content)),
            }
            .encode_to_vec(),
        };
        let ledger = FragReuseLedger::decode_inline(index_version, &details)
            .unwrap()
            .unwrap();
        for transition in ledger.transitions() {
            for digest in transition.sources().iter().chain(transition.destinations()) {
                fragment_bitmap.insert(digest.id as u32);
            }
        }
        IndexMetadata {
            uuid: Uuid::new_v4(),
            name: FRAG_REUSE_INDEX_NAME.into(),
            fields: Vec::new(),
            covering_fields: Vec::new(),
            dataset_version: 1,
            fragment_bitmap: Some(fragment_bitmap),
            index_details: Some(Arc::new(details)),
            index_version,
            created_at: None,
            base_id: None,
            files: None,
        }
    }

    /// The history `indices` carries: its index version, legacy versions and
    /// transitions.
    pub fn read_history(
        indices: &[IndexMetadata],
    ) -> Option<(i32, Vec<pb_fri::Version>, Vec<pb_fri::Transition>)> {
        let entry = indices
            .iter()
            .find(|index| index.name == FRAG_REUSE_INDEX_NAME)?;
        let details =
            pb::FragmentReuseIndexDetails::decode(entry.index_details.as_ref()?.value.as_slice())
                .unwrap();
        let Some(pb::fragment_reuse_index_details::Content::Inline(content)) = details.content
        else {
            panic!("the history is inline in these tests");
        };
        Some((
            entry.index_version,
            content.legacy_versions,
            content.transitions,
        ))
    }
}
