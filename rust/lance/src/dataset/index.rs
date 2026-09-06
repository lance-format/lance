// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

pub mod frag_reuse;

use lance_core::utils::row_addr_remap::RowAddrRemap;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::Dataset;
use crate::dataset::optimize::RemappedIndex;
use crate::dataset::optimize::remapping::RemapResult;
use crate::index::remap_index;
use crate::index::scalar::infer_scalar_index_details;
use crate::index::vector::ivf::compaction_vector_segment_merge_groups;
use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
use arrow_schema::DataType;
use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_file::version::ConcreteFileVersion;
use lance_index::is_system_index;
use lance_index::pb::VectorIndexDetails;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::IndexMetadata;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::optimize::{IndexRemapper, IndexRemapperOptions};
use super::versions;

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetIndexRemapperOptions {}

/// Loads index metadata when compaction has at least one index to remap.
///
/// Returns all usable index metadata, including system indices, so the remapper
/// uses a consistent snapshot. Returns `None` when there are no usable
/// non-system indices.
pub(crate) async fn load_indices_for_remapping(
    dataset: &Dataset,
) -> Result<Option<Arc<Vec<IndexMetadata>>>> {
    if dataset.manifest.index_section.is_none() {
        return Ok(None);
    }

    let indices = dataset.load_indices().await?;
    let has_remappable_index = indices.iter().any(|index| !is_system_index(index));
    Ok(has_remappable_index.then_some(indices))
}

#[async_trait]
impl IndexRemapperOptions for DatasetIndexRemapperOptions {
    async fn create_remapper(&self, dataset: &Dataset) -> Result<Option<Box<dyn IndexRemapper>>> {
        let Some(indices) = load_indices_for_remapping(dataset).await? else {
            return Ok(None);
        };

        Ok(Some(Box::new(DatasetIndexRemapper {
            dataset: Arc::new(dataset.clone()),
            indices,
        })))
    }
}

struct DatasetIndexRemapper {
    dataset: Arc<Dataset>,
    indices: Arc<Vec<IndexMetadata>>,
}

pub(crate) async fn vector_segment_merge_groups_for_compaction<'a>(
    dataset: &Dataset,
    segments: &[&'a IndexMetadata],
) -> Vec<Vec<&'a IndexMetadata>> {
    let singleton_groups = || segments.iter().map(|segment| vec![*segment]).collect();
    if segments.len() < 2
        || segments.iter().any(|segment| {
            !segment
                .index_details
                .as_ref()
                .is_some_and(|details| details.type_url.ends_with("VectorIndexDetails"))
        })
        || segments
            .iter()
            .any(|segment| segment.fields != segments[0].fields)
        || segments[0].fields.len() != 1
    {
        return singleton_groups();
    }

    let Ok(column) = dataset.schema().field_path(segments[0].fields[0]) else {
        return singleton_groups();
    };
    let Ok(logical_index) = dataset
        .open_logical_vector_index(&column, &segments[0].name)
        .await
    else {
        return singleton_groups();
    };
    let Ok(ivf) = logical_index.as_ivf() else {
        return singleton_groups();
    };

    let mut segments_by_id = segments
        .iter()
        .map(|segment| (segment.uuid, *segment))
        .collect::<HashMap<_, _>>();
    let groups = compaction_vector_segment_merge_groups(&ivf)
        .into_iter()
        .filter_map(|group| {
            let group = group
                .into_iter()
                .filter_map(|segment_id| segments_by_id.remove(&segment_id))
                .collect::<Vec<_>>();
            (!group.is_empty()).then_some(group)
        })
        .collect::<Vec<_>>();

    if segments_by_id.is_empty() {
        groups
    } else {
        singleton_groups()
    }
}

impl DatasetIndexRemapper {
    async fn remap_index(
        &self,
        index: &IndexMetadata,
        mapping: &RowAddrRemap,
    ) -> Result<RemapResult> {
        remap_index(&self.dataset, &index.uuid, mapping).await
    }

    async fn remapped_index_from_result(
        &self,
        index: &IndexMetadata,
        remap_result: RemapResult,
    ) -> Result<Option<RemappedIndex>> {
        match remap_result {
            RemapResult::Drop => Ok(None),
            RemapResult::Keep(id) => {
                let index_details = match &index.index_details {
                    Some(index_details) => index_details.as_ref().clone(),
                    None => {
                        // Migration path, if we didn't store details before then use the
                        // default details. This only supports a single keyed field, not a
                        // composite index.
                        let Some(field) = index.keyed_field() else {
                            return Err(Error::index(format!(
                                "Index {} has fields {:?} (carried fields {:?}); the \
                                 legacy index-details migration path only supports a \
                                 single keyed field",
                                index.uuid, index.fields, index.covering_fields
                            )));
                        };
                        let field = self.dataset.schema().field_by_id(field).ok_or_else(|| {
                            Error::internal(format!(
                                "Index {} references field {} which does not exist",
                                index.uuid, field
                            ))
                        })?;

                        if matches!(field.data_type(), DataType::FixedSizeList(..)) {
                            prost_types::Any::from_msg(&VectorIndexDetails::default())?
                        } else {
                            infer_scalar_index_details(&self.dataset, &field.name, index)
                                .await?
                                .as_ref()
                                .clone()
                        }
                    }
                };
                Ok(Some(RemappedIndex {
                    old_id: id,
                    new_id: id,
                    index_details,
                    index_version: index.index_version as u32,
                    files: index.files.clone(),
                }))
            }
            RemapResult::Remapped(remapped_index) => Ok(Some(remapped_index)),
        }
    }

    async fn remap_and_merge_vector_segments(
        &self,
        segments: &[&IndexMetadata],
        mapping: &RowAddrRemap,
    ) -> Result<Option<Vec<RemappedIndex>>> {
        if segments.len() < 2
            || segments.iter().any(|segment| {
                !segment
                    .index_details
                    .as_ref()
                    .is_some_and(|details| details.type_url.ends_with("VectorIndexDetails"))
                    || segment.fragment_bitmap.is_none()
            })
        {
            return Ok(None);
        }

        let mut segment_results = Vec::with_capacity(segments.len());
        for segment in segments {
            // See the call-site note in `remap_indices` about boxing this future.
            let result = Box::pin(self.remap_index(segment, mapping)).await?;
            segment_results.push((*segment, result));
        }

        if segment_results
            .iter()
            .any(|(_, result)| !matches!(result, RemapResult::Remapped(_)))
        {
            let mut remapped = Vec::with_capacity(segment_results.len());
            for (segment, result) in segment_results {
                if let Some(remapped_index) =
                    self.remapped_index_from_result(segment, result).await?
                {
                    remapped.push(remapped_index);
                }
            }
            return Ok(Some(remapped));
        }

        let mut remapped_segments = Vec::with_capacity(segments.len());
        let mut remapped_results = Vec::with_capacity(segments.len());
        for (segment, result) in segment_results {
            let RemapResult::Remapped(remapped) = result else {
                unreachable!("non-remapped results returned above")
            };
            let mut metadata = (*segment).clone();
            metadata.uuid = remapped.new_id;
            metadata.index_details = Some(Arc::new(remapped.index_details.clone()));
            metadata.index_version = remapped.index_version as i32;
            metadata.base_id = None;
            metadata.files = remapped.files.clone();
            remapped_segments.push(metadata);
            remapped_results.push(remapped);
        }

        let merged =
            crate::index::vector::ivf::merge_remapped_segments(&self.dataset, remapped_segments)
                .await?;
        let merged_details = merged.index_details.as_ref().ok_or_else(|| {
            Error::internal(format!(
                "Merged vector index {} is missing index details",
                merged.uuid
            ))
        })?;
        let merged_version = u32::try_from(merged.index_version).map_err(|_| {
            Error::internal(format!(
                "Merged vector index {} has invalid version {}",
                merged.uuid, merged.index_version
            ))
        })?;

        for remapped in &remapped_results {
            let intermediate_dir = self.dataset.indices_dir().join(remapped.new_id.to_string());
            if let Err(error) = self
                .dataset
                .object_store
                .remove_dir_all(intermediate_dir)
                .await
            {
                log::warn!(
                    "Failed to remove intermediate remapped index {}: {}",
                    remapped.new_id,
                    error
                );
            }
        }

        Ok(Some(
            segments
                .iter()
                .map(|segment| RemappedIndex {
                    old_id: segment.uuid,
                    new_id: merged.uuid,
                    index_details: merged_details.as_ref().clone(),
                    index_version: merged_version,
                    files: merged.files.clone(),
                })
                .collect(),
        ))
    }
}

#[async_trait]
impl IndexRemapper for DatasetIndexRemapper {
    async fn remap_indices(
        &self,
        mapping: RowAddrRemap,
        affected_fragment_ids: &[u64],
    ) -> Result<Vec<RemappedIndex>> {
        let affected_frag_ids = HashSet::<u64>::from_iter(affected_fragment_ids.iter().copied());
        let mut affected_by_name = HashMap::<&str, Vec<&IndexMetadata>>::new();
        for index in self.indices.iter() {
            let needs_remapped = !is_system_index(index)
                && match &index.fragment_bitmap {
                    None => true,
                    Some(fragment_bitmap) => fragment_bitmap
                        .iter()
                        .any(|frag_idx| affected_frag_ids.contains(&(frag_idx as u64))),
                };
            if needs_remapped {
                affected_by_name
                    .entry(index.name.as_str())
                    .or_default()
                    .push(index);
            }
        }

        let mut remapped = Vec::with_capacity(self.indices.len());
        let mut visited_names = HashSet::new();
        for index in self.indices.iter() {
            let Some(segments) = affected_by_name.get(index.name.as_str()) else {
                continue;
            };
            if !visited_names.insert(index.name.as_str()) {
                continue;
            }

            for merge_group in
                vector_segment_merge_groups_for_compaction(&self.dataset, segments).await
            {
                if let Some(merged) = self
                    .remap_and_merge_vector_segments(&merge_group, &mapping)
                    .await?
                {
                    remapped.extend(merged);
                    continue;
                }

                for segment in merge_group {
                    // Box the remap future at the call site: inlining `remap_index` into this
                    // loop's async layout otherwise exceeds rustc's depth limit. It has to be
                    // boxed here, not inside `remap_index` — boxing internally turns the
                    // future's `Send` check into a `Box<Future>: Send` trait obligation that
                    // overflows the solver through the cache types (E0275 downstream).
                    let remap_result = Box::pin(self.remap_index(segment, &mapping)).await?;
                    if let Some(remapped_index) = self
                        .remapped_index_from_result(segment, remap_result)
                        .await?
                    {
                        remapped.push(remapped_index);
                    }
                }
            }
        }
        Ok(remapped)
    }
}

#[async_trait]
pub trait LanceIndexStoreExt {
    /// Create an index store for a new index (will always be absolute with no base id)
    fn from_dataset_for_new(dataset: &Dataset, uuid: &Uuid) -> Result<Self>
    where
        Self: Sized;

    /// Open an index store for an existing index (might be relative or absolute)
    async fn from_dataset_for_existing(dataset: &Dataset, index: &IndexMetadata) -> Result<Self>
    where
        Self: Sized;
}

/// Select the exact file version used for index files in this dataset version.
///
/// Index files should never use the legacy format. If the dataset uses legacy
/// format, V2_0 is selected explicitly by the dataset composition table.
pub(crate) fn dataset_format_version(dataset: &Dataset) -> ConcreteFileVersion {
    versions::index_file_version(dataset.manifest.data_storage_format.lance_file_format())
}

#[async_trait]
impl LanceIndexStoreExt for LanceIndexStore {
    fn from_dataset_for_new(dataset: &Dataset, uuid: &Uuid) -> Result<Self> {
        let index_dir = dataset.indices_dir().join(uuid.to_string());
        let cache = dataset.metadata_cache.file_metadata_cache(&index_dir);
        let format_version = dataset_format_version(dataset);
        Ok(Self::with_format_version(
            dataset.object_store.clone(),
            index_dir,
            Arc::new(cache),
            format_version,
        ))
    }

    async fn from_dataset_for_existing(dataset: &Dataset, index: &IndexMetadata) -> Result<Self> {
        let index_dir = dataset
            .indice_files_dir(index)?
            .join(index.uuid.to_string());
        let cache = dataset.metadata_cache.file_metadata_cache(&index_dir);
        let format_version = dataset_format_version(dataset);
        let object_store = dataset.object_store_for_index(index).await?;
        let store =
            Self::with_format_version(object_store, index_dir, Arc::new(cache), format_version);
        Ok(store.with_file_sizes(index.file_size_map()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::WriteParams;
    use crate::dataset::transaction::{Operation, Transaction};
    use crate::index::frag_reuse::build_frag_reuse_index_metadata;
    use crate::index::vector::VectorIndexParams;
    use crate::index::{DatasetIndexExt, IntoIndexSegment};
    use lance_datagen::{BatchCount, RowCount, array};
    use lance_index::IndexType;
    use lance_index::frag_reuse::{FRAG_REUSE_INDEX_NAME, FragReuseIndexDetails};
    use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
    use lance_linalg::distance::MetricType;
    use std::collections::HashMap;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_remapper_not_created_without_remappable_indices() {
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(1), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        let options = DatasetIndexRemapperOptions::default();

        assert!(options.create_remapper(&dataset).await.unwrap().is_none());

        let frag_reuse_index = build_frag_reuse_index_metadata(
            &dataset,
            None,
            FragReuseIndexDetails {
                versions: Vec::new(),
            },
            Default::default(),
        )
        .await
        .unwrap();
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![frag_reuse_index],
                removed_indices: Vec::new(),
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let indices = dataset.load_indices().await.unwrap();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0].name, FRAG_REUSE_INDEX_NAME);
        assert!(options.create_remapper(&dataset).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_remapper_not_created_for_unknown_index_type() {
        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(1), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                Some("id_idx".to_string()),
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();

        let current = dataset.load_indices().await.unwrap();
        let unknown = IndexMetadata {
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "type.googleapis.com/example.ForeignIndexDetails".to_string(),
                value: Vec::new(),
            })),
            fragment_bitmap: None,
            ..current[0].clone()
        };
        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![unknown],
                removed_indices: current.to_vec(),
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        assert!(dataset.load_indices().await.unwrap().is_empty());
        assert!(
            DatasetIndexRemapperOptions::default()
                .create_remapper(&dataset)
                .await
                .unwrap()
                .is_none(),
            "compaction must not migrate an index type this build cannot open"
        );
    }

    #[tokio::test]
    async fn test_remapper_only_touches_segments_with_affected_fragments() {
        let test_dir = tempfile::tempdir().unwrap();
        let test_uri = test_dir.path().to_str().unwrap();

        let reader = lance_datagen::gen_batch()
            .col("id", array::step::<arrow_array::types::Int32Type>())
            .col(
                "vector",
                array::rand_vec::<arrow_array::types::Float32Type>(16.into()),
            )
            .into_reader_rows(RowCount::from(40), BatchCount::from(2));

        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 20,
                max_rows_per_group: 20,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let fragments = dataset.get_fragments();
        assert!(
            fragments.len() >= 2,
            "expected at least two fragments for this test"
        );
        let target_fragments = fragments.iter().take(2).collect::<Vec<_>>();

        let params = VectorIndexParams::ivf_flat(2, MetricType::L2);
        let first_segment_uuid = Uuid::new_v4();
        let second_segment_uuid = Uuid::new_v4();
        let built_index = dataset
            .create_index_builder(&["vector"], IndexType::Vector, &params)
            .name("vector_idx".to_string())
            .index_uuid(first_segment_uuid)
            .execute_uncommitted()
            .await
            .unwrap();
        let first_segment_dir = dataset.indices_dir().join(first_segment_uuid.to_string());
        let second_segment_dir = dataset.indices_dir().join(second_segment_uuid.to_string());
        for file_name in ["index.idx", "auxiliary.idx"] {
            dataset
                .object_store
                .as_ref()
                .copy(
                    &first_segment_dir.clone().join(file_name),
                    &second_segment_dir.clone().join(file_name),
                )
                .await
                .unwrap();
        }

        let segments = [
            IndexMetadata {
                uuid: first_segment_uuid,
                fragment_bitmap: Some(std::iter::once(target_fragments[0].id() as u32).collect()),
                ..built_index.clone()
            },
            IndexMetadata {
                uuid: second_segment_uuid,
                fragment_bitmap: Some(std::iter::once(target_fragments[1].id() as u32).collect()),
                ..built_index
            },
        ];

        let segments = segments
            .iter()
            .map(|segment| {
                segment
                    .clone()
                    .into_index_segment()
                    .expect("test segment metadata should convert to an index segment")
            })
            .collect::<Vec<_>>();

        dataset
            .commit_existing_index_segments("vector_idx", "vector", segments)
            .await
            .unwrap();
        let committed = dataset.load_indices_by_name("vector_idx").await.unwrap();
        let committed_ids = committed
            .iter()
            .map(|segment| segment.uuid)
            .collect::<Vec<_>>();
        let unaffected_segment_id = committed
            .iter()
            .find(|segment| {
                segment
                    .fragment_bitmap
                    .as_ref()
                    .is_some_and(|bitmap| bitmap.contains(target_fragments[1].id() as u32))
            })
            .map(|segment| segment.uuid)
            .expect("expected one committed segment to cover the unaffected fragment");

        let remapper = DatasetIndexRemapperOptions::default()
            .create_remapper(&dataset)
            .await
            .unwrap()
            .expect("vector index should require a remapper");
        let remapped = remapper
            .remap_indices(RowAddrRemap::empty(), &[target_fragments[0].id() as u64])
            .await
            .unwrap();

        assert_eq!(remapped.len(), 1);
        assert!(committed_ids.contains(&remapped[0].old_id));
        assert_ne!(remapped[0].old_id, unaffected_segment_id);
        assert_ne!(remapped[0].new_id, unaffected_segment_id);
    }

    /// A covered index must be withdrawn from remapping rather than remapped.
    /// No index type carries the declared payload through a remap, so a
    /// replacement would republish a covering claim its storage does not back.
    /// Withdrawal also means the legacy `index_details: None` migration path is
    /// never reached for such an index, so it cannot panic there either.
    #[tokio::test]
    async fn test_remapper_migration_path_withdraws_covered_index() {
        let reader = lance_datagen::gen_batch()
            .col("a", array::step::<arrow_array::types::Int32Type>())
            .col("b", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        dataset
            .create_index(
                &["a"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();

        let a_id = dataset.schema().field("a").unwrap().id;
        let b_id = dataset.schema().field("b").unwrap().id;
        let current = dataset.load_indices().await.unwrap();
        let mut legacy_covered = current[0].clone();
        legacy_covered.fields = vec![a_id, b_id];
        legacy_covered.covering_fields = vec![b_id];
        // Force the legacy migration path this fix touches.
        legacy_covered.index_details = None;

        let transaction = Transaction::new(
            dataset.manifest.version,
            Operation::CreateIndex {
                new_indices: vec![legacy_covered],
                removed_indices: current.to_vec(),
            },
            None,
        );
        dataset
            .apply_commit(transaction, &Default::default(), &Default::default())
            .await
            .unwrap();

        let index_uuid = dataset.load_indices().await.unwrap()[0].uuid;

        // Fully delete every row so `remap_index` returns `RemapResult::Keep`,
        // landing in the `index_details: None` migration branch this fix touches.
        let remap_to_empty = (0..dataset.count_all_rows().await.unwrap())
            .map(|i| (i as u64, None))
            .collect::<HashMap<_, _>>();
        let remapper = DatasetIndexRemapperOptions::default()
            .create_remapper(&dataset)
            .await
            .unwrap()
            .expect("a real index should require a remapper");
        let remapped = remapper
            .remap_indices(RowAddrRemap::direct(remap_to_empty), &[0])
            .await
            .unwrap();

        // Withdrawn, not remapped: no index type carries the declared payload
        // through a remap, so producing a replacement would republish a covering
        // claim its storage does not back. The original entry stays in the
        // manifest and simply stops covering the rewritten fragments.
        assert!(
            remapped.is_empty(),
            "a covered index must be withdrawn from remapping, got {remapped:?}"
        );
        let _ = index_uuid;
    }

    /// The same migration path must reject -- cleanly, not with a panic -- a
    /// malformed `covering_fields` longer than `fields`.
    ///
    /// Note this is a genuinely synthetic scenario, not one a real commit can
    /// produce: `crate::index::remap_index`'s own `keyed > 1` guard (a sibling
    /// fix in this same phase) already rejects every organically-committable
    /// composite index *before* this migration path ever runs, since it is
    /// only reached from that function's `RemapResult::Keep` arm. And
    /// `validate_covering_fields` rejects a fully-consumed `covering_fields`
    /// (`keyed == 0` with non-empty `fields`) at `Operation::CreateIndex`
    /// commit time, and `TryFrom<pb::IndexMetadata>` rejects it again when a
    /// manifest is decoded. The only way left to reach this branch's rejection
    /// is metadata that passed through neither, which is how this test reaches
    /// it: seeding the index cache directly rather than committing through a
    /// `Transaction`.
    #[tokio::test]
    async fn test_remapper_migration_path_rejects_malformed_covering_fields() {
        let reader = lance_datagen::gen_batch()
            .col("a", array::step::<arrow_array::types::Int32Type>())
            .into_reader_rows(RowCount::from(20), BatchCount::from(1));
        let mut dataset = Dataset::write(reader, "memory://", None).await.unwrap();
        dataset
            .create_index(
                &["a"],
                IndexType::BTree,
                None,
                &ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                false,
            )
            .await
            .unwrap();

        let index_uuid = dataset.load_indices().await.unwrap()[0].uuid;
        let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
        for idx in &mut indices {
            if idx.uuid == index_uuid {
                // Force the legacy migration path, and malform `covering_fields`
                // to be longer than `fields` -- more carried fields than fields
                // at all. No normal commit can produce this.
                idx.index_details = None;
                idx.covering_fields = idx
                    .fields
                    .iter()
                    .copied()
                    .chain(std::iter::once(999))
                    .collect();
            }
        }
        let metadata_key = crate::session::index_caches::IndexMetadataKey {
            version: dataset.version().version,
            store_identity: &dataset.object_store.store_prefix,
            e_tag: dataset.manifest_location.e_tag.as_deref(),
        };
        dataset
            .index_cache
            .insert_with_key(&metadata_key, Arc::new(indices))
            .await;

        let remap_to_empty = (0..dataset.count_all_rows().await.unwrap())
            .map(|i| (i as u64, None))
            .collect::<HashMap<_, _>>();
        let remapper = DatasetIndexRemapperOptions::default()
            .create_remapper(&dataset)
            .await
            .unwrap()
            .expect("a real index should require a remapper");
        let error = remapper
            .remap_indices(RowAddrRemap::direct(remap_to_empty), &[0])
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("are not among its fields"),
            "malformed covering must fail closed via the validator, not be \
             withdrawn as an ordinary covered index; got: {error}"
        );
    }
}
