// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Translating a legacy [`Operation`] into the action vocabulary.
//!
//! Each named operation is a fixed recipe over the granular actions. Expressing
//! the recipe here rather than in the commit loop is what lets several
//! operations be squashed into one commit: once both are action sets, combining
//! them is concatenation.
//!
//! Translation is fail-closed. An operation whose recipe is not written yet, or
//! one carrying a detail the actions cannot express, is rejected rather than
//! silently translated into something narrower.
//!
//! `Merge` and `Project` are not translated. Both hand over a whole new schema
//! rather than a description of what changed, so recovering the delta needs the
//! read version's schema to diff against -- which this conversion, taking only
//! the operation, does not have. The actions themselves are sufficient:
//! `Project` is a set of [`DropField`](super::DropField)s and `Merge` a set of
//! [`AddField`](super::AddField)s plus their data files.

use super::{
    Action, AddBase, AddDataFile, AddFragment, Ref, RemoveFragment, SetDeletionFile,
    TombstoneFieldData, UserAction,
};
use crate::format::Fragment;
use crate::transaction::{DataReplacementGroup, Operation};
use lance_core::{Error, Result};

impl TryFrom<&Operation> for Vec<UserAction> {
    type Error = Error;

    fn try_from(operation: &Operation) -> Result<Self> {
        match operation {
            Operation::Append { fragments } => Ok(vec![UserAction::new(
                format!("append {} fragments", fragments.len()),
                append_actions(fragments)?,
            )]),
            Operation::Delete {
                updated_fragments,
                deleted_fragment_ids,
                predicate,
            } => Ok(vec![UserAction::new(
                format!("delete rows matching {predicate}"),
                delete_actions(updated_fragments, deleted_fragment_ids),
            )]),
            Operation::UpdateBases { new_bases } => Ok(vec![UserAction::new(
                format!("add {} base paths", new_bases.len()),
                new_bases
                    .iter()
                    .enumerate()
                    .map(|(index, base)| {
                        Action::AddBase(AddBase {
                            local: index as u32,
                            base: base.clone(),
                        })
                    })
                    .collect(),
            )]),
            Operation::DataReplacement { replacements } => Ok(vec![UserAction::new(
                format!("replace data files in {} fragments", replacements.len()),
                data_replacement_actions(replacements)?,
            )]),
            other => Err(Error::not_supported(format!(
                "translating a {} operation into actions",
                other.name()
            ))),
        }
    }
}

fn append_actions(fragments: &[Fragment]) -> Result<Vec<Action>> {
    let mut actions = Vec::with_capacity(fragments.len() * 2);
    for (index, fragment) in fragments.iter().enumerate() {
        let local = index as u32;
        let physical_rows = fragment.physical_rows.ok_or_else(|| {
            Error::invalid_input(
                "an appended fragment must know its physical row count to become an AddFragment",
            )
        })?;
        // A fragment being appended has no committed state, so anything that
        // only makes sense against committed data means the caller built the
        // operation by hand out of an existing fragment.
        if fragment.deletion_file.is_some() || !fragment.overlays.is_empty() {
            return Err(Error::invalid_input(format!(
                "appended fragment {} carries a deletion file or overlays, which an append \
                 cannot produce",
                fragment.id
            )));
        }

        actions.push(Action::AddFragment(AddFragment {
            local,
            physical_rows: physical_rows as u64,
            row_id_meta: fragment.row_id_meta.clone(),
            last_updated_at_version_meta: fragment.last_updated_at_version_meta.clone(),
            created_at_version_meta: fragment.created_at_version_meta.clone(),
            data_change: true,
        }));

        for file in &fragment.files {
            actions.push(Action::AddDataFile(AddDataFile {
                fragment: Ref::Local(local),
                file: file.clone(),
                field_ids: committed_field_refs(file.fields.as_ref())?,
                data_change: true,
            }));
        }
    }
    Ok(actions)
}

/// A legacy delete replaces whole fragments, but the only thing it ever changes
/// on one is its deletion file, so that is what the translation carries over.
fn delete_actions(updated_fragments: &[Fragment], deleted_fragment_ids: &[u64]) -> Vec<Action> {
    let mut actions = Vec::with_capacity(updated_fragments.len() + deleted_fragment_ids.len());
    for fragment in updated_fragments {
        actions.push(Action::SetDeletionFile(SetDeletionFile {
            fragment: fragment.id,
            deletion_file: fragment.deletion_file.clone(),
            data_change: true,
        }));
    }
    for id in deleted_fragment_ids {
        actions.push(Action::RemoveFragment(RemoveFragment {
            fragment: Ref::Committed(*id),
            data_change: true,
        }));
    }
    actions
}

/// Replacing a field's data is a drop of the old backing file followed by an
/// add of the new one. The legacy path swaps the path on the existing file in
/// place instead, so the resulting fragment holds the same set of data files
/// but not necessarily in the same order -- files are addressed by field, so
/// the order carries no meaning.
fn data_replacement_actions(replacements: &[DataReplacementGroup]) -> Result<Vec<Action>> {
    let mut actions = Vec::with_capacity(replacements.len() * 2);
    for DataReplacementGroup(fragment_id, new_file) in replacements {
        let fragment = Ref::Committed(*fragment_id);
        actions.push(Action::TombstoneFieldData(TombstoneFieldData {
            fragment,
            field_ids: committed_field_refs(new_file.fields.as_ref())?,
            data_change: true,
        }));
        actions.push(Action::AddDataFile(AddDataFile {
            fragment,
            file: new_file.clone(),
            field_ids: committed_field_refs(new_file.fields.as_ref())?,
            data_change: true,
        }));
    }
    Ok(actions)
}

fn committed_field_refs(field_ids: &[i32]) -> Result<Vec<Ref>> {
    field_ids
        .iter()
        .map(|id| {
            u64::try_from(*id).map(Ref::Committed).map_err(|_| {
                Error::invalid_input(format!(
                    "a data file in this operation lists field id {id}, which is not a committed \
                     field"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{
        BasePath, DataFile, DeletionFile, DeletionFileType, IndexMetadata, Manifest, RowIdMeta,
    };
    use crate::rowids::{RowIdSequence, write_row_ids};
    use crate::transaction::Transaction;
    use crate::transaction::action::CompositeOperation;
    use crate::transaction::test_support::{
        default_build_config, make_stable_row_id_manifest, sample_manifest,
    };
    use lance_file::version::ConcreteFileVersion;
    use std::sync::Arc;

    /// Build the same manifest twice -- once down the legacy path, once by
    /// translating the operation to actions -- and assert they agree.
    fn assert_parity(manifest: &Manifest, operation: Operation) -> Manifest {
        let (legacy, legacy_indices) = build(manifest, operation.clone());

        let actions = Vec::<UserAction>::try_from(&operation).unwrap();
        let (translated, translated_indices) = build(
            manifest,
            Operation::CompositeOperation(CompositeOperation::new(actions)),
        );

        // Data files are addressed by field, so the two paths are allowed to
        // hold them in different orders within a fragment.
        assert_eq!(translated.fragments.len(), legacy.fragments.len());
        for (translated, legacy) in translated.fragments.iter().zip(legacy.fragments.iter()) {
            assert_eq!(sorted_files(translated), sorted_files(legacy));
            assert_eq!(
                Fragment {
                    files: Vec::new(),
                    ..translated.clone()
                },
                Fragment {
                    files: Vec::new(),
                    ..legacy.clone()
                }
            );
        }
        assert_eq!(translated.schema, legacy.schema);
        assert_eq!(translated.base_paths, legacy.base_paths);
        assert_eq!(translated.next_row_id, legacy.next_row_id);
        assert_eq!(translated.max_fragment_id, legacy.max_fragment_id);
        assert_eq!(translated_indices, legacy_indices);
        translated
    }

    fn sorted_files(fragment: &Fragment) -> Vec<DataFile> {
        let mut files = fragment.files.clone();
        files.sort_by(|a, b| a.path.cmp(&b.path));
        files
    }

    fn build(manifest: &Manifest, operation: Operation) -> (Manifest, Vec<IndexMetadata>) {
        Transaction::new(manifest.version, operation, None)
            .build_manifest(
                Some(manifest),
                Vec::new(),
                "tx.txn",
                &default_build_config(),
            )
            .unwrap()
    }

    fn appendable_fragment(path: &str) -> Fragment {
        let mut fragment = Fragment::new(0);
        fragment.physical_rows = Some(10);
        fragment.files.push(DataFile::new(
            path,
            vec![0],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        ));
        fragment
    }

    fn manifest_with_fragments(fragments: Vec<Fragment>) -> Manifest {
        let mut manifest = sample_manifest();
        manifest.fragments = Arc::new(fragments);
        manifest
    }

    #[test]
    fn test_append_matches_the_legacy_path() {
        let manifest = manifest_with_fragments(vec![appendable_fragment("data/0.lance")]);
        let next = assert_parity(
            &manifest,
            Operation::Append {
                fragments: vec![
                    appendable_fragment("data/1.lance"),
                    appendable_fragment("data/2.lance"),
                ],
            },
        );

        // The appended fragments take ids from the manifest's counter, not the
        // zero they were handed to the operation with.
        assert_eq!(
            next.fragments.iter().map(|f| f.id).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn test_append_assigns_stable_row_ids_like_the_legacy_path() {
        let mut existing = appendable_fragment("data/1.lance");
        existing.id = 1;
        existing.row_id_meta = Some(RowIdMeta::Inline(
            write_row_ids(&RowIdSequence::from(0u64..10)).into(),
        ));
        let manifest = make_stable_row_id_manifest(vec![existing]);

        let next = assert_parity(
            &manifest,
            Operation::Append {
                fragments: vec![appendable_fragment("data/2.lance")],
            },
        );

        assert_eq!(next.next_row_id, 1010);
        let appended = next.fragments.iter().find(|f| f.id == 2).unwrap();
        assert!(appended.row_id_meta.is_some());
        assert!(appended.created_at_version_meta.is_some());
    }

    #[test]
    fn test_append_rejects_a_fragment_without_a_row_count() {
        let operation = Operation::Append {
            fragments: vec![Fragment::new(0)],
        };
        let error = Vec::<UserAction>::try_from(&operation).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
    }

    #[test]
    fn test_delete_matches_the_legacy_path() {
        let manifest = manifest_with_fragments(vec![appendable_fragment("data/0.lance"), {
            let mut fragment = appendable_fragment("data/1.lance");
            fragment.id = 1;
            fragment
        }]);

        let mut updated = manifest.fragments[0].clone();
        updated.deletion_file = Some(DeletionFile {
            read_version: manifest.version,
            id: 7,
            file_type: DeletionFileType::Array,
            num_deleted_rows: Some(3),
            base_id: None,
        });

        let next = assert_parity(
            &manifest,
            Operation::Delete {
                updated_fragments: vec![updated],
                deleted_fragment_ids: vec![1],
                predicate: "id > 5".into(),
            },
        );

        assert_eq!(next.fragments.len(), 1);
        assert!(next.fragments[0].deletion_file.is_some());
    }

    #[test]
    fn test_update_bases_matches_the_legacy_path() {
        let manifest = manifest_with_fragments(vec![appendable_fragment("data/0.lance")]);
        let next = assert_parity(
            &manifest,
            Operation::UpdateBases {
                new_bases: vec![
                    BasePath::new(0, "s3://bucket/a".into(), Some("a".into()), false),
                    BasePath::new(0, "s3://bucket/b".into(), Some("b".into()), false),
                ],
            },
        );

        assert_eq!(next.base_paths.len(), 2);
    }

    #[test]
    fn test_data_replacement_matches_the_legacy_path() {
        let mut fragment = appendable_fragment("data/0.lance");
        fragment.files.push(DataFile::new(
            "data/0b.lance",
            vec![1],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        ));
        let manifest = manifest_with_fragments(vec![fragment]);

        let replacement = DataFile::new(
            "data/0-new.lance",
            vec![0],
            vec![0],
            ConcreteFileVersion::V2_0,
            None,
            None,
        );
        let next = assert_parity(
            &manifest,
            Operation::DataReplacement {
                replacements: vec![DataReplacementGroup(0, replacement)],
            },
        );

        let paths = sorted_files(&next.fragments[0])
            .iter()
            .map(|file| file.path.clone())
            .collect::<Vec<_>>();
        assert_eq!(paths, vec!["data/0-new.lance", "data/0b.lance"]);
    }

    #[test]
    fn test_data_replacement_of_an_unbacked_field_is_rejected() {
        let manifest = manifest_with_fragments(vec![appendable_fragment("data/0.lance")]);
        let operation = Operation::DataReplacement {
            replacements: vec![DataReplacementGroup(
                0,
                DataFile::new(
                    "data/0-new.lance",
                    vec![9],
                    vec![0],
                    ConcreteFileVersion::V2_0,
                    None,
                    None,
                ),
            )],
        };
        let actions = Vec::<UserAction>::try_from(&operation).unwrap();
        let error = Transaction::new(
            manifest.version,
            Operation::CompositeOperation(CompositeOperation::new(actions)),
            None,
        )
        .build_manifest(
            Some(&manifest),
            Vec::new(),
            "tx.txn",
            &default_build_config(),
        )
        .unwrap_err();

        // The legacy path treats this as an all-NULL column gaining real data;
        // the action form has no way to say "drop this if it is there".
        assert!(matches!(error, Error::InvalidInput { .. }), "{error:?}");
    }

    #[test]
    fn test_an_untranslated_operation_is_rejected() {
        let operation = Operation::ReserveFragments { num_fragments: 3 };
        let error = Vec::<UserAction>::try_from(&operation).unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }), "{error:?}");
        assert!(error.to_string().contains("ReserveFragments"), "{error}");
    }
}
