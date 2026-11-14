// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bridge common FRI address translation to scalar and vector index loading.
use super::frag_reuse_reader::FragmentReuseIndex;
use crate::Dataset;
use async_trait::async_trait;
use lance_core::Result;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::utils::address::RowAddress;
use lance_index::INDEX_FILE_NAME;
use lance_table::format::IndexMetadata;
use roaring::RoaringBitmap;
use std::sync::Arc;
#[derive(Clone)]
struct VectorFormatKey(uuid::Uuid);

impl CacheKey for VectorFormatKey {
    type ValueType = bool;
    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.0.to_string().into()
    }
    fn type_name() -> &'static str {
        "VectorBatchRemapping"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.vector-batch-remapping", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.0.as_bytes());
    }
}

#[derive(Clone)]
struct InvertedFormatKey(uuid::Uuid);

impl CacheKey for InvertedFormatKey {
    type ValueType = bool;
    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.0.to_string().into()
    }
    fn type_name() -> &'static str {
        "InvertedBatchRemapping"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.inverted-batch-remapping", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.0.as_bytes());
    }
}

/// Whether a full-text segment can be loaded under a translating remapper, or
/// `None` when the question has no answer yet.
///
/// The plugin supports batch remapping, but a legacy-layout index (no
/// `metadata.lance`; the loader's own format probe) stores per-document
/// lengths aligned with the row ids it was built with and cannot be
/// translated. Such a segment is excluded from coverage so the planner scans
/// its fragments; an untouched (identity) legacy segment still loads.
///
/// A segment holding any file under `staging/` has not published its root
/// metadata yet, so that file's absence is not the legacy layout -- it is a
/// build in progress, including a finalize interrupted partway that left root
/// partitions beside the staging directory. Reporting `None` there keeps the
/// withdrawal selector from treating a modern distributed build as
/// untranslatable, and the non-verdict is never memoized: the finalize that
/// publishes the root metadata makes the same call definitive. A segment being
/// replayed into a commit carries no recorded file list, so the layout is read
/// from its index directory rather than assumed legacy.
///
/// Only a confirmed missing metadata file reports the legacy layout. Any
/// other failure to read it propagates and nothing is cached: an
/// unreachable object store says nothing about the format, and a coverage
/// withdrawal derived from it would be persisted.
pub async fn inverted_supports_batch_remapping(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> Result<Option<bool>> {
    if let Some(cached) = dataset
        .index_cache
        .get_with_key(&InvertedFormatKey(index.uuid))
        .await
    {
        return Ok(Some(*cached));
    }
    use crate::dataset::index::LanceIndexStoreExt;
    let store = lance_index::scalar::lance_format::LanceIndexStore::from_dataset_for_existing(
        dataset, index,
    )
    .await?;
    let supported = match lance_index::scalar::IndexStore::open_index_file(
        &store,
        lance_index::scalar::inverted::METADATA_FILE,
    )
    .await
    {
        Ok(_) => true,
        // Only a segment that has published its layout answers for it. An
        // unfinalized build's root metadata is simply not written yet.
        Err(e) if e.is_not_found() => {
            if segment_files_are_staged_or_listed(dataset, index).await? {
                return Ok(None);
            }
            false
        }
        Err(e) => return Err(e),
    };
    dataset
        .index_cache
        .insert_with_key(&InvertedFormatKey(index.uuid), Arc::new(supported))
        .await;
    Ok(Some(supported))
}

/// Whether this segment holds ANY staged build artifact, i.e. its build has
/// not published its final layout. A finalize interrupted partway leaves root
/// partitions beside the staging directory, so only a layout with no staged
/// file at all is the legacy one.
///
/// A segment replayed at commit time carries no file list (the commit collects
/// the files from the index directory), so the recorded list cannot answer and
/// the segment's own index location is listed instead -- resolved through its
/// `base_id`, the same directory and store the root metadata was probed in. A
/// recorded list keeps the zero-store-call shortcut.
async fn segment_files_are_staged_or_listed(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> Result<bool> {
    if let Some(files) = index.files.as_ref() {
        return Ok(has_staged_file(files));
    }
    let index_dir = dataset
        .indice_files_dir(index)?
        .join(index.uuid.to_string());
    let store = dataset.object_store_for_index(index).await?;
    let files = lance_table::format::list_index_files_with_sizes(&store, &index_dir).await?;
    Ok(has_staged_file(&files))
}

fn has_staged_file(files: &[lance_table::format::IndexFile]) -> bool {
    files
        .iter()
        .any(|file| lance_index::scalar::inverted::builder::is_staged_path(&file.path))
}

/// Legacy vector file readers keep their existing V1 behavior and scan fallback.
pub async fn vector_supports_batch_remapping(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> Result<bool> {
    let supported = dataset
        .index_cache
        .get_or_insert_with_key(VectorFormatKey(index.uuid), || async {
            let path = dataset
                .indice_files_dir(index)?
                .join(index.uuid.to_string())
                .join(INDEX_FILE_NAME);
            let store = dataset.object_store_for_index(index).await?;
            let reader = super::vector::open_index_file(
                store.as_ref(),
                &path,
                INDEX_FILE_NAME,
                &index.file_size_map(),
            )
            .await?;
            let tail = lance_io::utils::read_last_block(reader.as_ref()).await?;
            Ok(matches!(
                lance_io::utils::read_version(&tail)?,
                (0, 3) | (2, _)
            ))
        })
        .await?;
    Ok(*supported)
}

/// Applies the shared FRI history and limits translated rows to this segment's coverage.
pub(super) struct QueryRowIdRemapper {
    mapping: Arc<FragmentReuseIndex>,
    coverage: RoaringBitmap,
    excluded_fragments: RoaringBitmap,
}

impl std::fmt::Debug for QueryRowIdRemapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryRowIdRemapper")
            .field("coverage", &self.coverage)
            .field("excluded_fragments", &self.excluded_fragments)
            .finish_non_exhaustive()
    }
}

impl QueryRowIdRemapper {
    pub(crate) fn new(
        mapping: Arc<FragmentReuseIndex>,
        coverage: RoaringBitmap,
        excluded_fragments: RoaringBitmap,
    ) -> Self {
        Self {
            mapping,
            coverage,
            excluded_fragments,
        }
    }
}

#[async_trait]
impl lance_index::scalar::BatchRowIdRemapper for QueryRowIdRemapper {
    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        lance_index::scalar::check_batch_remapping_entry()?;
        Ok(self
            .mapping
            .remap_row_ids_excluding(row_ids, &self.excluded_fragments)
            .await?
            .into_iter()
            .map(|address| {
                address.filter(|row_id| {
                    self.coverage
                        .contains(RowAddress::from(*row_id).fragment_id())
                })
            })
            .collect())
    }
}
