// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    collections::{HashMap, HashSet},
    num::NonZero,
    sync::Arc,
};

use futures::{StreamExt, TryFutureExt, TryStreamExt, future::Either};
use lance_core::{Error, Result};
use lance_io::utils::CachedFileSize;
use lance_table::format::{
    DeletionFile, Fragment, IndexMetadata, Manifest, WriterVersion, list_index_files_with_sizes,
};
use lance_table::system_index::is_system_index;
use roaring::RoaringBitmap;

use crate::io::deletion::read_table_deletion_file;
use crate::{FileFragment, Table};

use super::CommitIndexHook;

/// before certain new operations. An easy way to force a migration is to run
/// `dataset.delete(false)`, which won't modify data but will cause a migration.
/// However, you don't want to always have to do this, so we provide this method
/// to check if a migration is needed.
pub fn manifest_needs_migration(manifest: &Manifest, indices: &[IndexMetadata]) -> bool {
    manifest.writer_version.is_none()
        || manifest.fragments.iter().any(|f| {
            f.physical_rows.is_none()
                || (f
                    .deletion_file
                    .as_ref()
                    .map(|d| d.num_deleted_rows.is_none())
                    .unwrap_or(false))
        })
        || indices
            .iter()
            .any(|i| must_recalculate_fragment_bitmap(i, manifest.writer_version.as_ref()))
}

/// Update manifest with new metadata fields.
///
/// Fields such as `physical_rows` and `num_deleted_rows` may not have been
/// in older datasets. To bring these old manifests up-to-date, we add them here.
pub(super) async fn migrate_manifest(
    dataset: &Table,
    manifest: &mut Manifest,
    recompute_stats: bool,
) -> Result<()> {
    if !recompute_stats
        && manifest.fragments.iter().all(|f| {
            f.num_rows().map(|n| n > 0).unwrap_or(false)
                && f.files.iter().all(|f| f.file_size_bytes.get().is_some())
        })
    {
        return Ok(());
    }

    manifest.fragments =
        Arc::new(migrate_fragments(dataset, &manifest.fragments, recompute_stats).await?);

    Ok(())
}

#[cfg(test)]
fn check_storage_version(manifest: &mut Manifest) -> Result<()> {
    lance_table::format::storage_contract::check_manifest_storage_version(manifest)
}

/// Reject a manifest in which two fragments share an id. Per-fragment state is
/// keyed by fragment id — deletion file paths, cached row id sequences, row
/// addresses — so a duplicate makes it ambiguous which rows that state describes.
///
/// Runs after the legacy fixups above, so a dataset that needs a rollback for some
/// other reason is diagnosed with that first. Relies on `build_manifest` leaving
/// the fragments sorted by id.
pub(super) fn check_fragment_ids(manifest: &Manifest) -> Result<()> {
    if let Some(pair) = manifest.fragments.windows(2).find(|p| p[0].id == p[1].id) {
        return Err(Error::invalid_input(format!(
            "The commit would produce two fragments with id {}. Fragment ids must be \
             unique. Datasets written by Lance 0.16 and earlier may already contain \
             duplicate ids; those have to be rewritten, or rolled back to a version \
             without the duplicate.",
            pair[0].id
        )));
    }
    Ok(())
}

#[cfg(test)]
fn check_column_indices(manifest: &Manifest) -> Result<()> {
    lance_table::format::storage_contract::validate_column_indices(manifest)
}

/// Fix schema in case of duplicate field ids.
///
/// See test dataset v0.10.5/corrupt_schema
pub fn fix_schema(manifest: &mut Manifest) -> Result<()> {
    // We can short-circuit if there is only one file per fragment or no fragments.
    if manifest.fragments.iter().all(|f| f.files.len() <= 1) {
        return Ok(());
    }

    // First, see which, if any fields have duplicate ids, within any fragment.
    // Data file field lists are interned `Arc<[i32]>`s shared across fragments, so
    // fragments whose files carry the same lists were already checked.
    let mut fields_with_duplicate_ids = HashSet::new();
    let mut seen_fields = HashSet::new();
    let mut checked_field_lists: HashSet<Vec<usize>> = HashSet::new();
    for fragment in manifest.fragments.iter() {
        let field_lists: Vec<usize> = fragment
            .files
            .iter()
            .map(|file| file.fields.as_ptr() as usize)
            .collect();
        if !checked_field_lists.insert(field_lists) {
            continue;
        }
        for file in fragment.files.iter() {
            for field_id in file.fields.iter() {
                if *field_id >= 0 && !seen_fields.insert(*field_id) {
                    fields_with_duplicate_ids.insert(*field_id);
                }
            }
        }
        seen_fields.clear();
    }
    if fields_with_duplicate_ids.is_empty() {
        return Ok(());
    }

    if manifest.uses_non_reusable_field_ids() {
        return Err(Error::invalid_input(
            "Cannot repair duplicate field IDs after non-reusable field identity is activated; repair would change an existing identity",
        ));
    }

    // Now, we need to remap the field ids to be unique.
    let mut old_field_id_mapping: HashMap<i32, i32> = HashMap::new();
    let mut fields_with_duplicate_ids = fields_with_duplicate_ids.into_iter().collect::<Vec<_>>();
    fields_with_duplicate_ids.sort_unstable();
    let next_field_ids = i64::from(manifest.max_field_id()) + 1..;
    for (next_field_id, field_id) in next_field_ids.zip(fields_with_duplicate_ids) {
        let assigned_field_id = i32::try_from(next_field_id).map_err(|_| {
            Error::invalid_input(
                "Cannot repair duplicate field IDs because the field-ID space is exhausted",
            )
        })?;
        old_field_id_mapping.insert(field_id, assigned_field_id);
    }

    let mut fragments = manifest.fragments.as_ref().clone();

    // Apply mapping to fragment files list
    // We iterate over files in reverse order so that we only map the last field id
    seen_fields.clear();
    for fragment in fragments.iter_mut() {
        for file in fragment.files.iter_mut().rev() {
            let new_fields: Arc<[i32]> = file
                .fields
                .iter()
                .map(|field_id| {
                    if let Some(new_field_id) = old_field_id_mapping.get(field_id)
                        && seen_fields.insert(*field_id)
                    {
                        *new_field_id
                    } else {
                        *field_id
                    }
                })
                .collect::<Vec<_>>()
                .into();
            file.fields = new_fields;
        }
        seen_fields.clear();
    }

    // Apply mapping to the schema. Children carry an independent `parent_id`
    // that is serialized verbatim, so re-parent them onto the new id as well.
    // A field's children are named by this field's id only, so each entry can
    // be applied independently of the order the mapping is iterated in.
    for (old_field_id, new_field_id) in &old_field_id_mapping {
        let field = manifest.schema.mut_field_by_id(*old_field_id).ok_or_else(|| {
            Error::invalid_input(format!(
                "Cannot repair duplicate field id {} because it does not exist in the manifest schema",
                old_field_id
            ))
        })?;
        field.id = *new_field_id;
        for child in field.children.iter_mut() {
            child.parent_id = *new_field_id;
        }
    }

    // Drop data files that are no longer in use.
    let remaining_field_ids = manifest
        .schema
        .fields_pre_order()
        .map(|f| f.id)
        .collect::<HashSet<_>>();
    for fragment in fragments.iter_mut() {
        fragment.files.retain(|file| {
            file.fields
                .iter()
                .any(|field_id| remaining_field_ids.contains(field_id))
        });
    }

    manifest.fragments = Arc::new(fragments);

    Ok(())
}

/// Get updated vector of fragments that has `physical_rows` and `num_deleted_rows`
/// filled in. This is no-op for newer tables, but may do IO for tables written
/// with older versions of Lance.
pub async fn migrate_fragments(
    dataset: &Table,
    fragments: &[Fragment],
    recompute_stats: bool,
) -> Result<Vec<Fragment>> {
    let dataset = Arc::new(dataset.clone());
    let new_fragments = futures::stream::iter(fragments)
        .map(|fragment| async {
            let physical_rows = if recompute_stats {
                None
            } else {
                fragment.physical_rows
            };
            let physical_rows = if let Some(physical_rows) = physical_rows {
                Either::Right(futures::future::ready(Ok(physical_rows)))
            } else {
                let file_fragment = FileFragment::new(dataset.clone(), fragment.clone());
                Either::Left(async move { file_fragment.physical_rows().await })
            };
            let num_deleted_rows = match &fragment.deletion_file {
                None => Either::Left(futures::future::ready(Ok(None))),
                Some(DeletionFile {
                    num_deleted_rows: Some(deleted_rows),
                    ..
                }) if !recompute_stats => {
                    Either::Left(futures::future::ready(Ok(Some(*deleted_rows))))
                }
                Some(deletion_file) => Either::Right(async {
                    let deletion_vector =
                        read_table_deletion_file(dataset.as_ref(), fragment.id, deletion_file)
                            .await?;
                    Ok(Some(deletion_vector.len()))
                }),
            };

            let (physical_rows, num_deleted_rows) =
                futures::future::try_join(physical_rows, num_deleted_rows).await?;

            let mut data_files = fragment.files.clone();

            // For each of the data files in the fragment, we need to get the file size.
            // Resolve each file against its own storage base: multi-base datasets
            // keep data files outside the dataset root (DataFile.base_id).
            let get_sizes = data_files
                .iter()
                .map(|file| {
                    if let Some(size) = file.file_size_bytes.get() {
                        Either::Left(futures::future::ready(Ok(size)))
                    } else {
                        let dataset = dataset.clone();
                        Either::Right(async move {
                            let object_store = dataset.object_store(file.base_id).await?;
                            let data_dir = dataset.view().data_file_dir_for_base(file.base_id)?;
                            object_store
                                .size(&data_dir.join(file.path.clone()))
                                .map_ok(|size| {
                                    NonZero::new(size).ok_or_else(|| {
                                        Error::internal(format!("File {} has size 0", file.path))
                                    })
                                })
                                .await?
                        })
                    }
                })
                .collect::<Vec<_>>();
            let sizes = futures::future::try_join_all(get_sizes).await?;
            data_files.iter_mut().zip(sizes).for_each(|(file, size)| {
                file.file_size_bytes = CachedFileSize::new(size.into());
            });

            let deletion_file = fragment
                .deletion_file
                .as_ref()
                .map(|deletion_file| DeletionFile {
                    num_deleted_rows,
                    ..deletion_file.clone()
                });

            Ok::<_, Error>(Fragment {
                physical_rows: Some(physical_rows),
                deletion_file,
                files: data_files,
                ..fragment.clone()
            })
        })
        .buffered(dataset.state.object_store.io_parallelism())
        // Filter out empty fragments
        .try_filter(|frag| futures::future::ready(frag.num_rows().map(|n| n > 0).unwrap_or(false)))
        .boxed();

    new_fragments.try_collect().await
}

fn must_recalculate_fragment_bitmap(
    index: &IndexMetadata,
    version: Option<&WriterVersion>,
) -> bool {
    if index.fragment_bitmap.is_none() {
        return true;
    }
    // If the fragment bitmap was written by an old version of lance then we need to recalculate
    // it because it could be corrupt due to a bug in versions < 0.8.15
    if let Some(version) = version {
        if version.library != "lance" {
            // We assume a different library is not affected by the bug.
            return false;
        }

        let cutoff = semver::Version::new(0, 8, 15);
        version
            .lance_lib_version()
            .map(|lance_lib_version| lance_lib_version < cutoff)
            .unwrap_or(true)
    } else {
        // Older versions of Lance library didn't record writer version at all.
        true
    }
}

/// Update indices with new fields.
///
/// Indices might be missing `fragment_bitmap`, so this function will add it.
/// Indices might also be missing `files` (file sizes), so this function will collect them.
///
/// Returns the logical indices whose `fragment_bitmap` this replaced. Those are
/// the only changes here that alter what an index covers, and the caller has to
/// withdraw MemWAL catch-up for them: this runs after the coverage derivation,
/// and it keeps the segment's UUID.
pub(super) async fn migrate_indices(
    dataset: &Table,
    indices: &mut [IndexMetadata],
    index_hook: Option<&dyn CommitIndexHook>,
) -> Result<Vec<String>> {
    let Some(index_hook) = index_hook else {
        return Ok(Vec::new());
    };
    index_hook.infer_legacy_details(dataset, indices).await;
    let index_is_usable = index_hook.usability_predicate();
    let mut recovered_coverage = Vec::new();
    let needs_recalculating = match detect_overlapping_fragments(indices) {
        Ok(()) => vec![],
        Err(BadFragmentBitmapError { bad_indices }) => {
            bad_indices.into_iter().map(|(name, _)| name).collect()
        }
    };
    for index in indices.iter_mut() {
        // Migration is skipped for an index this build has no reader for: every
        // branch below would have to open it to recalculate anything, which is
        // exactly what this build cannot do, and failing here would fail an
        // unrelated commit. Its stored coverage must remain unchanged rather
        // than persisting a fragment-reuse projection this writer cannot verify.
        if !index_is_usable(index) {
            continue;
        }
        // Also true when the bitmap is missing entirely, so the failure path below
        // pairs it with `is_some` to mean "written before the 0.8.15 fix".
        let bitmap_missing_or_legacy =
            must_recalculate_fragment_bitmap(index, dataset.manifest.writer_version.as_ref());
        if needs_recalculating.contains(&index.name)
            || bitmap_missing_or_legacy && !is_system_index(index)
        {
            // `fields[0]` is the one query input used to recover the bitmap.
            // Plugin-declared dependencies and carried columns may follow it;
            // the index hook resolves which entries are query inputs.
            debug_assert!(
                !index.keyed_fields().is_empty(),
                "migrate_indices expects a single keyed field, got fields {:?} carrying {:?}",
                index.fields,
                index.covering_fields,
            );
            dataset.schema().field_by_id(index.fields[0]).ok_or_else(|| Error::internal(format!(
                "Index with uuid {} referred to field with id {} which did not exist in dataset",
                index.uuid, index.fields[0]
            )))?;
            let recalculated = index_hook.recover_coverage(dataset, index).await;
            match recalculated {
                Ok(fragment_bitmap) => {
                    if index.fragment_bitmap.as_ref() != Some(&fragment_bitmap) {
                        recovered_coverage.push(index.name.clone());
                    }
                    index.fragment_bitmap = Some(fragment_bitmap);
                }
                Err(e) => {
                    // Recalculating means opening the index, and failing here fails
                    // every commit the dataset takes, since migration runs on all of
                    // them. A missing bitmap and overlapping segment bitmaps are both
                    // re-derived from the index metadata, so they ask again on their
                    // own; the pre-0.8.15 trigger reads the previous manifest's writer
                    // version, which this commit replaces with the current one, and a
                    // bitmap left in place would look migrated from here on.
                    let repair_ends_with_this_commit =
                        index.fragment_bitmap.is_some() && bitmap_missing_or_legacy;
                    log::warn!(
                        "Could not recalculate the fragment bitmap for index {} (uuid: {}): {}. {}",
                        index.name,
                        index.uuid,
                        e,
                        if repair_ends_with_this_commit {
                            "Dropping its coverage to unknown so a build that can open the index recalculates it."
                        } else {
                            "Leaving the repair to a build that can open the index."
                        }
                    );
                    if repair_ends_with_this_commit {
                        index.fragment_bitmap = None;
                        // Derivation ran before this and may have credited a
                        // catch-up position off the bitmap being dropped here.
                        recovered_coverage.push(index.name.clone());
                    }
                }
            }
        }
        // We can't reliably recalculate the index type for label_list and bitmap indices and so we can't migrate this field.
        // However, we still log for visibility and to help potentially diagnose issues in the future if we grow to rely on the field.
        if index.index_details.is_none() {
            log::debug!(
                "the index with uuid {} is missing index metadata.  This probably means it was written with Lance version <= 0.19.2.  This is not a problem.",
                index.uuid
            );
        }

        // Migrate file sizes for indices that don't have them.
        // Use indice_files_dir to handle shallow-cloned indices with base_id.
        if index.files.is_none() && !is_system_index(index) {
            let result = async {
                let index_dir = dataset
                    .view()
                    .indice_files_dir(index)?
                    .join(index.uuid.to_string());
                let object_store = dataset.object_store(index.base_id).await?;
                list_index_files_with_sizes(&object_store, &index_dir).await
            }
            .await;
            match result {
                Ok(files) => {
                    log::debug!(
                        "Migrated file sizes for index {} (uuid: {}): {} files",
                        index.name,
                        index.uuid,
                        files.len()
                    );
                    index.files = Some(files);
                }
                Err(e) => {
                    // Log but don't fail - file sizes are optional
                    log::debug!(
                        "Could not collect file sizes for index {} (uuid: {}): {}",
                        index.name,
                        index.uuid,
                        e
                    );
                }
            }
        }
    }

    Ok(recovered_coverage)
}

pub struct BadFragmentBitmapError {
    pub bad_indices: Vec<(String, Vec<u32>)>,
}

/// Detect whether a given index has overlapping fragment bitmaps in its index
/// segments.
pub fn detect_overlapping_fragments(
    indices: &[IndexMetadata],
) -> std::result::Result<(), BadFragmentBitmapError> {
    let mut bitmaps_by_name: HashMap<&str, Vec<&RoaringBitmap>> = HashMap::new();
    for index in indices {
        if let Some(fragment_bitmap) = index.fragment_bitmap.as_ref() {
            bitmaps_by_name
                .entry(index.name.as_str())
                .or_default()
                .push(fragment_bitmap);
        }
    }
    let mut bad_indices = Vec::new(); // (index_name, overlapping_fragments)
    for (name, fragment_bitmaps) in bitmaps_by_name {
        // A single segment (the common case) cannot overlap with itself, so
        // skip it before hashing every fragment id it covers.
        if fragment_bitmaps.len() < 2 {
            continue;
        }
        let mut seen_fragment_ids = HashSet::new();
        let mut overlap = Vec::new();
        for fragment_bitmap in fragment_bitmaps {
            for fragment in fragment_bitmap {
                if !seen_fragment_ids.insert(fragment) {
                    overlap.push(fragment);
                }
            }
        }
        if !overlap.is_empty() {
            bad_indices.push((name.to_string(), overlap));
        }
    }
    if bad_indices.is_empty() {
        Ok(())
    } else {
        Err(BadFragmentBitmapError { bad_indices })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field as ArrowField};
    use lance_core::datatypes::{Field, Schema};
    use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
    use lance_table::feature_flags::FLAG_MIXED_DATA_FILE_VERSIONS;
    use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};
    use lance_table::format::{DataFile, DataStorageFormat};

    #[test]
    fn test_fix_schema() {
        // Manifest has a fragment with no fields in use
        // Manifest has a duplicate field id in one fragment but not others.
        let mut field0 =
            Field::try_from(ArrowField::new("a", arrow_schema::DataType::Int64, false)).unwrap();
        field0.set_id(-1, &mut 0);
        let mut field2 =
            Field::try_from(ArrowField::new("b", arrow_schema::DataType::Int64, false)).unwrap();
        field2.set_id(-1, &mut 2);

        let schema = Schema {
            fields: vec![field0.clone(), field2.clone()],
            metadata: Default::default(),
        };
        let fragments = vec![
            Fragment {
                id: 0,
                files: vec![
                    DataFile::new_legacy_from_fields("path1", vec![0, 1, 2], None),
                    DataFile::new_legacy_from_fields("unused", vec![9], None),
                ],
                overlays: vec![],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                clustering_id: None,
                metadata: Default::default(),
            },
            Fragment {
                id: 1,
                files: vec![
                    DataFile::new_legacy_from_fields("path2", vec![0, 1, 2], None),
                    DataFile::new_legacy_from_fields("path3", vec![2], None),
                ],
                overlays: vec![],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                clustering_id: None,
                metadata: Default::default(),
            },
        ];

        let mut manifest = Manifest::new(
            schema,
            Arc::new(fragments),
            DataStorageFormat::default(),
            HashMap::new(),
        );

        fix_schema(&mut manifest).unwrap();

        // Because of the duplicate field id, the field id of field2 should have been changed to 10
        field2.id = 10;
        let expected_schema = Schema {
            fields: vec![field0, field2],
            metadata: Default::default(),
        };
        assert_eq!(manifest.schema, expected_schema);

        // The fragment with just field 9 should have been removed, since it's
        // not used in the current schema.
        // The field 2 should have been changed to 10, except in the first
        // file of the second fragment.
        let expected_fragments = vec![
            Fragment {
                id: 0,
                files: vec![DataFile::new_legacy_from_fields(
                    "path1",
                    vec![0, 1, 10],
                    None,
                )],
                overlays: vec![],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                clustering_id: None,
                metadata: Default::default(),
            },
            Fragment {
                id: 1,
                files: vec![
                    DataFile::new_legacy_from_fields("path2", vec![0, 1, 2], None),
                    DataFile::new_legacy_from_fields("path3", vec![10], None),
                ],
                overlays: vec![],
                deletion_file: None,
                row_id_meta: None,
                physical_rows: None,
                last_updated_at_version_meta: None,
                created_at_version_meta: None,
                clustering_id: None,
                metadata: Default::default(),
            },
        ];
        assert_eq!(manifest.fragments.as_ref(), &expected_fragments);
    }

    #[test]
    fn test_fix_schema_nested() {
        // A duplicated struct field is renumbered, so its children must be
        // re-parented onto the new id. Otherwise the manifest serializes a
        // parent_id that no longer exists and can never be read back.
        let mut field0 = Field::try_from(ArrowField::new("a", DataType::Int64, false)).unwrap();
        field0.set_id(-1, &mut 0);

        let inner = ArrowField::new(
            "inner",
            DataType::Struct(vec![ArrowField::new("ordinal", DataType::Int32, false)].into()),
            false,
        );
        let mut outer = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("text", DataType::Utf8, false), inner].into()),
            false,
        ))
        .unwrap();
        // ids: s = 1, text = 2, inner = 3, ordinal = 4
        outer.set_id(-1, &mut 1);

        let schema = Schema {
            fields: vec![field0, outer],
            metadata: Default::default(),
        };
        // Both the struct (1) and the nested struct (3) have duplicate coverage
        // within this fragment, so both get renumbered.
        let fragments = vec![Fragment {
            id: 0,
            files: vec![
                DataFile::new_legacy_from_fields("path1", vec![0, 1, 2, 3, 4], None),
                DataFile::new_legacy_from_fields("path2", vec![1, 3], None),
            ],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            clustering_id: None,
            metadata: Default::default(),
        }];

        let mut manifest = Manifest::new(
            schema,
            Arc::new(fragments),
            DataStorageFormat::default(),
            HashMap::new(),
        );

        fix_schema(&mut manifest).unwrap();

        // max_field_id was 4, so 1 -> 5 and 3 -> 6.
        let outer = &manifest.schema.fields[1];
        assert_eq!(outer.id, 5);
        let text = &outer.children[0];
        let inner = &outer.children[1];
        assert_eq!(
            text.parent_id, 5,
            "child of a renumbered struct kept a stale parent_id"
        );
        assert_eq!(
            inner.parent_id, 5,
            "child of a renumbered struct kept a stale parent_id"
        );
        assert_eq!(inner.id, 6);
        assert_eq!(
            inner.children[0].parent_id, 6,
            "grandchild of a renumbered struct kept a stale parent_id"
        );

        // Every non-root field must name a parent that still exists in the schema.
        let live_ids = manifest
            .schema
            .fields_pre_order()
            .map(|f| f.id)
            .collect::<HashSet<_>>();
        for field in manifest.schema.fields_pre_order() {
            if field.parent_id >= 0 {
                assert!(
                    live_ids.contains(&field.parent_id),
                    "field '{}' (id={}) references parent id {}, which no longer exists",
                    field.name,
                    field.id,
                    field.parent_id
                );
            }
        }
    }

    #[test]
    fn test_fix_schema_rejects_field_id_exhaustion() {
        let mut field =
            Field::try_from(ArrowField::new("a", arrow_schema::DataType::Int64, false)).unwrap();
        field.id = i32::MAX;
        let schema = Schema {
            fields: vec![field],
            metadata: Default::default(),
        };
        let mut fragment = Fragment::new(0);
        fragment.files = vec![
            DataFile::new_legacy_from_fields("first", vec![i32::MAX], None),
            DataFile::new_legacy_from_fields("second", vec![i32::MAX], None),
        ];
        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![fragment]),
            DataStorageFormat::default(),
            HashMap::new(),
        );

        let err = fix_schema(&mut manifest).unwrap_err();

        assert!(err.to_string().contains("space is exhausted"), "{err}");
    }

    #[test]
    fn test_fix_schema_does_not_reassign_non_reusable_field_identity() {
        let mut field =
            Field::try_from(ArrowField::new("a", arrow_schema::DataType::Int64, false)).unwrap();
        field.id = 0;
        let schema = Schema {
            fields: vec![field],
            metadata: Default::default(),
        };
        let mut fragment = Fragment::new(0);
        fragment.files = vec![
            DataFile::new_legacy_from_fields("first", vec![0], None),
            DataFile::new_legacy_from_fields("second", vec![0], None),
        ];
        let mut manifest = Manifest::new(
            schema,
            Arc::new(vec![fragment]),
            DataStorageFormat::default(),
            HashMap::new(),
        );
        manifest.activate_non_reusable_field_ids();

        let err = fix_schema(&mut manifest).unwrap_err();

        assert!(
            err.to_string().contains("non-reusable field identity"),
            "{err}"
        );
        assert_eq!(manifest.schema.field("a").unwrap().id, 0);
    }
    /// Helper to build a simple manifest for check_column_indices tests.
    fn make_manifest_with_file(
        schema: Schema,
        data_file: DataFile,
        data_storage_version: LanceFileVersion,
    ) -> Manifest {
        let fragment = Fragment {
            id: 0,
            files: vec![data_file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(100),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            clustering_id: None,
            metadata: Default::default(),
        };
        Manifest::new(
            schema,
            Arc::new(vec![fragment]),
            DataStorageFormat::new(data_storage_version.resolve()),
            HashMap::new(),
        )
    }

    fn make_storage_contract_manifest(
        fallback: ConcreteFileVersion,
        file_versions: &[ConcreteFileVersion],
    ) -> Manifest {
        let files = file_versions
            .iter()
            .enumerate()
            .map(|(index, version)| {
                DataFile::new(
                    format!("data-{index}.lance"),
                    vec![],
                    vec![],
                    *version,
                    None,
                    None,
                )
            })
            .collect();
        let fragment = Fragment {
            id: 0,
            files,
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(0),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            clustering_id: None,
            metadata: Default::default(),
        };
        Manifest::new(
            Schema::default(),
            Arc::new(vec![fragment]),
            DataStorageFormat::new(fallback),
            HashMap::new(),
        )
    }

    fn enable_mixed_file_versions(manifest: &mut Manifest) {
        manifest.reader_feature_flags |= FLAG_MIXED_DATA_FILE_VERSIONS;
        manifest.writer_feature_flags |= FLAG_MIXED_DATA_FILE_VERSIONS;
    }

    #[test]
    fn storage_contract_accepts_all_exact_v2_combinations() {
        let versions = [
            ConcreteFileVersion::V2_0,
            ConcreteFileVersion::V2_1,
            ConcreteFileVersion::V2_2,
            ConcreteFileVersion::V2_3,
        ];
        for fallback in versions {
            for other in versions {
                let mut manifest = make_storage_contract_manifest(fallback, &[fallback, other]);
                enable_mixed_file_versions(&mut manifest);
                check_storage_version(&mut manifest).unwrap();
            }
        }
    }

    #[test]
    fn storage_contract_requires_capability_for_non_fallback_file() {
        let mut manifest =
            make_storage_contract_manifest(ConcreteFileVersion::V2_0, &[ConcreteFileVersion::V2_1]);

        let err = check_storage_version(&mut manifest).unwrap_err();

        assert!(err.to_string().contains("not enabled"), "{err}");
    }

    #[test]
    fn storage_contract_finalization_derives_capability() {
        let mut manifest = make_storage_contract_manifest(
            ConcreteFileVersion::V2_0,
            &[ConcreteFileVersion::V2_0, ConcreteFileVersion::V2_2],
        );

        lance_table::format::storage_contract::finalize_manifest_storage_version(&mut manifest)
            .unwrap();

        assert_ne!(
            manifest.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
        assert_ne!(
            manifest.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
        check_storage_version(&mut manifest).unwrap();
    }

    #[test]
    fn storage_contract_finalization_rejects_v1_non_fallback() {
        let mut manifest =
            make_storage_contract_manifest(ConcreteFileVersion::V2_1, &[ConcreteFileVersion::V1]);

        let err =
            lance_table::format::storage_contract::finalize_manifest_storage_version(&mut manifest)
                .unwrap_err();

        assert!(err.to_string().contains("cannot be mixed"), "{err}");
        assert_eq!(
            manifest.reader_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
        assert_eq!(
            manifest.writer_feature_flags & FLAG_MIXED_DATA_FILE_VERSIONS,
            0
        );
    }

    #[test]
    fn storage_contract_checks_overlay_versions() {
        let mut manifest =
            make_storage_contract_manifest(ConcreteFileVersion::V2_0, &[ConcreteFileVersion::V2_0]);
        manifest.fragments = Arc::new(vec![Fragment {
            overlays: vec![DataOverlayFile {
                data_file: DataFile::new(
                    "overlay.lance",
                    vec![],
                    vec![],
                    ConcreteFileVersion::V2_2,
                    None,
                    None,
                ),
                coverage: OverlayCoverage::dense(roaring::RoaringBitmap::from_iter([0])),
                committed_version: 1,
            }],
            ..manifest.fragments[0].clone()
        }]);

        assert!(check_storage_version(&mut manifest).is_err());
        enable_mixed_file_versions(&mut manifest);
        check_storage_version(&mut manifest).unwrap();
    }

    #[test]
    fn storage_contract_rejects_v1_v2_mixing_even_with_capability() {
        let mut manifest = make_storage_contract_manifest(
            ConcreteFileVersion::V2_0,
            &[ConcreteFileVersion::V1, ConcreteFileVersion::V2_0],
        );
        enable_mixed_file_versions(&mut manifest);

        let err = check_storage_version(&mut manifest).unwrap_err();

        assert!(err.to_string().contains("mixes V1 and V2"), "{err}");
    }

    #[test]
    fn storage_contract_rejects_unknown_file_identity() {
        let mut manifest =
            make_storage_contract_manifest(ConcreteFileVersion::V2_0, &[ConcreteFileVersion::V2_0]);
        Arc::make_mut(&mut manifest.fragments)[0].files[0].file_major_version = 99;
        enable_mixed_file_versions(&mut manifest);

        let err = check_storage_version(&mut manifest).unwrap_err();

        assert!(err.to_string().contains("99"), "{err}");
    }

    #[test]
    fn storage_contract_preserves_uniform_legacy_repair() {
        let mut manifest = make_storage_contract_manifest(
            ConcreteFileVersion::V1,
            &[ConcreteFileVersion::V2_1, ConcreteFileVersion::V2_1],
        );

        check_storage_version(&mut manifest).unwrap();

        assert_eq!(
            manifest.data_storage_format.lance_file_format(),
            ConcreteFileVersion::V1
        );

        lance_table::format::storage_contract::finalize_manifest_storage_version(&mut manifest)
            .unwrap();

        assert_eq!(
            manifest.data_storage_format.lance_file_format(),
            ConcreteFileVersion::V2_1
        );
    }

    #[test]
    fn storage_contract_does_not_extend_legacy_repair_to_mixed_v2() {
        let mut manifest = make_storage_contract_manifest(
            ConcreteFileVersion::V1,
            &[ConcreteFileVersion::V2_0, ConcreteFileVersion::V2_1],
        );

        let err = check_storage_version(&mut manifest).unwrap_err();

        assert!(matches!(err, Error::Internal { .. }));
        assert!(
            err.to_string().contains("mixture of file versions"),
            "{err}"
        );
    }

    #[test]
    fn storage_contract_rejects_v1_files_with_capability() {
        let mut manifest =
            make_storage_contract_manifest(ConcreteFileVersion::V2_0, &[ConcreteFileVersion::V1]);
        enable_mixed_file_versions(&mut manifest);

        let err = check_storage_version(&mut manifest).unwrap_err();

        assert!(err.to_string().contains("references V1"), "{err}");
    }

    #[test]
    fn storage_contract_rejects_empty_v1_fallback_with_capability() {
        let mut manifest = make_storage_contract_manifest(ConcreteFileVersion::V1, &[]);
        enable_mixed_file_versions(&mut manifest);

        let err = check_storage_version(&mut manifest).unwrap_err();

        assert!(matches!(err, Error::InvalidInput { .. }));
        assert!(err.to_string().contains("requires a V2 default"), "{err}");
    }

    #[test]
    fn storage_contract_validates_column_indices_per_file_version() {
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);
        let data_file = DataFile::new(
            "data.lance",
            vec![0, 1],
            vec![0, 1],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let mut manifest = make_manifest_with_file(
            Schema {
                fields: vec![struct_field],
                metadata: Default::default(),
            },
            data_file,
            LanceFileVersion::V2_0,
        );
        enable_mixed_file_versions(&mut manifest);

        let err = check_storage_version(&mut manifest).unwrap_err();

        assert!(err.to_string().contains("Non-leaf field"), "{err}");
    }

    #[test]
    fn test_fix_schema_checks_distinct_field_lists_after_shared_ones() {
        // Fragments sharing interned field lists are scanned once; a later
        // fragment with a distinct duplicate-id list must still be remapped.
        let mut field0 =
            Field::try_from(ArrowField::new("a", arrow_schema::DataType::Int64, false)).unwrap();
        field0.set_id(-1, &mut 0);
        let mut field1 =
            Field::try_from(ArrowField::new("b", arrow_schema::DataType::Int64, false)).unwrap();
        field1.set_id(-1, &mut 1);
        let schema = Schema {
            fields: vec![field0.clone(), field1.clone()],
            metadata: Default::default(),
        };

        let shared_a = DataFile::new_legacy_from_fields("a.lance", vec![0], None);
        let shared_b = DataFile::new_legacy_from_fields("b.lance", vec![1], None);
        let make_fragment = |id: u64, files: Vec<DataFile>| Fragment {
            id,
            files,
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: None,
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            clustering_id: None,
            metadata: Default::default(),
        };
        let fragments = vec![
            make_fragment(0, vec![shared_a.clone(), shared_b.clone()]),
            make_fragment(1, vec![shared_a.clone(), shared_b]),
            make_fragment(
                2,
                vec![
                    shared_a,
                    DataFile::new_legacy_from_fields("dup.lance", vec![0, 1], None),
                ],
            ),
        ];
        let mut manifest = Manifest::new(
            schema,
            Arc::new(fragments),
            DataStorageFormat::default(),
            HashMap::new(),
        );

        fix_schema(&mut manifest).unwrap();

        // Field 0 is duplicated only in fragment 2 -> remapped to max_field_id + 1 = 2.
        field0.id = 2;
        let expected_schema = Schema {
            fields: vec![field0, field1],
            metadata: Default::default(),
        };
        assert_eq!(manifest.schema, expected_schema);
        // Shared lists in fragments 0/1 are remapped too; fragment 2's first
        // file only held the old id and is dropped as no longer in use.
        assert_eq!(manifest.fragments[0].files[0].fields.as_ref(), &[2]);
        assert_eq!(manifest.fragments[0].files[1].fields.as_ref(), &[1]);
        let frag2 = &manifest.fragments[2];
        assert_eq!(frag2.files.len(), 1);
        assert_eq!(frag2.files[0].path, "dup.lance");
        assert_eq!(frag2.files[0].fields.as_ref(), &[2, 1]);
    }

    #[test]
    fn test_check_column_indices_rejects_struct_with_column() {
        // Struct (non-leaf) field with column_index=0 in v2.1 should be rejected.
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        // field ids: struct=0, leaf=1; give struct a real column_index (wrong)
        let data_file = DataFile::new(
            "data.lance",
            vec![0, 1],
            vec![0, 1],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        let result = check_column_indices(&manifest);
        assert!(
            result.is_err(),
            "Expected error for struct with column_index=0"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Non-leaf field"), "{msg}");
    }

    #[test]
    fn test_check_column_indices_rejects_list_with_column() {
        // List (non-leaf) field with column_index=0 in v2.1 should be rejected.
        let mut list_field = Field::try_from(ArrowField::new(
            "l",
            DataType::List(Arc::new(ArrowField::new("item", DataType::Int32, true))),
            false,
        ))
        .unwrap();
        list_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![list_field],
            metadata: Default::default(),
        };

        // field ids: list=0, item=1; give list a real column_index (wrong)
        let data_file = DataFile::new(
            "data.lance",
            vec![0, 1],
            vec![0, 1],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        let result = check_column_indices(&manifest);
        assert!(
            result.is_err(),
            "Expected error for list with column_index=0"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("Non-leaf field"), "{msg}");
    }

    #[test]
    fn test_check_column_indices_allows_correct_v21() {
        // Non-leaf with column_index=-1 and leaf with column_index>=0 should pass.
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        // struct=-1 (correct), leaf=0 (correct)
        let data_file = DataFile::new(
            "data.lance",
            vec![0, 1],
            vec![-1, 0],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        assert!(check_column_indices(&manifest).is_ok());
    }

    #[test]
    fn test_check_column_indices_allows_packed_struct() {
        // Packed struct with a real column_index in v2.1 should be allowed.
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);
        struct_field
            .metadata
            .insert("lance-encoding:packed".to_string(), "true".to_string());

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        // packed struct=0 (allowed), leaf=1
        let data_file = DataFile::new(
            "data.lance",
            vec![0, 1],
            vec![0, 1],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        assert!(check_column_indices(&manifest).is_ok());
    }

    #[test]
    fn test_check_column_indices_skips_v20() {
        // Non-leaf with column_index>=0 in v2.0 should be allowed (no validation).
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        let data_file = DataFile::new(
            "data.lance",
            vec![0, 1],
            vec![0, 1],
            ConcreteFileVersion::V2_0,
            None,
            None,
        );
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_0);
        assert!(check_column_indices(&manifest).is_ok());
    }

    #[test]
    fn test_check_column_indices_rejects_mismatched_lengths() {
        // fields and column_indices must have the same length.
        let mut leaf_field = Field::try_from(ArrowField::new("x", DataType::Int32, false)).unwrap();
        leaf_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![leaf_field],
            metadata: Default::default(),
        };

        // 1 field id but 2 column indices
        let data_file = DataFile::new(
            "data.lance",
            vec![0],
            vec![0, 1],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        let result = check_column_indices(&manifest);
        assert!(result.is_err(), "Expected error for mismatched lengths");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("1 field ids but 2 column indices"), "{msg}");
    }

    #[test]
    fn test_check_column_indices_skips_unknown_field_id() {
        // A field id not present in the schema is skipped (schema evolution).
        let mut leaf_field = Field::try_from(ArrowField::new("x", DataType::Int32, false)).unwrap();
        leaf_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![leaf_field],
            metadata: Default::default(),
        };

        // field id 99 does not exist in the schema — should be skipped
        let data_file = DataFile::new(
            "data.lance",
            vec![0, 99],
            vec![0, 1],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        assert!(check_column_indices(&manifest).is_ok());
    }

    #[test]
    fn test_check_column_indices_rejects_leaf_with_negative_one() {
        // A leaf field with column_index=-1 in v2.1 should be rejected.
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        // struct=-1 (correct), but leaf=-1 (wrong — leaf must have a real column)
        let data_file = DataFile::new(
            "data.lance",
            vec![0, 1],
            vec![-1, -1],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let manifest = make_manifest_with_file(schema, data_file, LanceFileVersion::V2_1);
        let result = check_column_indices(&manifest);
        assert!(
            result.is_err(),
            "Expected error for leaf with column_index=-1"
        );
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("must have a valid column index"), "{msg}");
    }

    #[test]
    fn test_check_column_indices_rejects_after_dedup() {
        let mut struct_field = Field::try_from(ArrowField::new(
            "s",
            DataType::Struct(vec![ArrowField::new("x", DataType::Int32, false)].into()),
            false,
        ))
        .unwrap();
        struct_field.set_id(-1, &mut 0);

        let schema = Schema {
            fields: vec![struct_field],
            metadata: Default::default(),
        };

        // struct=-1, leaf=0: valid layout; clones share the same Arcs.
        let shared_file = DataFile::new(
            "shared.lance",
            vec![0, 1],
            vec![-1, 0],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        // Wrongly gives the struct a real column index.
        let bad_file = DataFile::new(
            "bad.lance",
            vec![0, 1],
            vec![0, 1],
            ConcreteFileVersion::V2_1,
            None,
            None,
        );
        let make_fragment = |id: u64, file: DataFile| Fragment {
            id,
            files: vec![file],
            overlays: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(100),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
            clustering_id: None,
            metadata: Default::default(),
        };
        let manifest = Manifest::new(
            schema,
            Arc::new(vec![
                make_fragment(0, shared_file.clone()),
                make_fragment(1, shared_file),
                make_fragment(2, bad_file),
            ]),
            DataStorageFormat::new(LanceFileVersion::V2_1.resolve()),
            HashMap::new(),
        );
        let msg = check_column_indices(&manifest).unwrap_err().to_string();
        assert!(msg.contains("Non-leaf field"), "{msg}");
        assert!(msg.contains("bad.lance"), "{msg}");
    }
}
