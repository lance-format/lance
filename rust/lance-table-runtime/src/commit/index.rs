// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::sync::Arc;

use arrow_schema::DataType;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder};
use lance_core::{Error, Result};
use lance_table::format::{IndexMetadata, Manifest};
use lance_table::io::manifest::read_manifest_indexes;
use lance_table::system_index::frag_reuse::{
    CompactFragReuseIndex, FRAG_REUSE_INDEX_NAME, metadata::is_tagged,
};
use lance_table::system_index::is_system_index;
use lance_table::transaction::{Operation, action::Action};

use crate::Table;
use crate::caches::{DerivedIndexListingKey, FragReuseIndexKey, IndexMetadataKey};
use crate::frag_reuse::{load_frag_reuse_index_details, v0_cache_identity};

use super::CommitIndexHook;

impl Table {
    /// Read every index entry stored in this snapshot's manifest, including
    /// entries for unknown index implementations and opaque system metadata.
    ///
    /// This listing retains the stored coverage. A legacy fragment-reuse
    /// history can translate that coverage to different fragment IDs.
    pub async fn load_index_metadata(&self) -> Result<Arc<Vec<IndexMetadata>>> {
        self.index_cache()
            .get_or_insert_with_key(self.index_metadata_key(), || async {
                let indices = read_manifest_indexes(
                    &self.state.object_store,
                    &self.state.manifest_location,
                    &self.manifest,
                )
                .await?;
                self.state
                    .session
                    .observe_index_metadata(&self.manifest, &indices);
                Ok(indices)
            })
            .await
    }

    pub(crate) fn index_metadata_key(&self) -> IndexMetadataKey<'_> {
        IndexMetadataKey {
            version: self.manifest.version,
            store_identity: &self.state.object_store.store_prefix,
            e_tag: self.state.manifest_location.e_tag.as_deref(),
        }
    }
}

struct InferredIndexListingKey<'a> {
    snapshot: DerivedIndexListingKey<'a>,
    namespace: &'static str,
}

impl CacheKey for InferredIndexListingKey<'_> {
    type ValueType = Vec<IndexMetadata>;

    fn key(&self) -> Cow<'_, str> {
        Cow::Owned(format!("{}/{}", self.namespace, self.snapshot.key()))
    }

    fn type_name() -> &'static str {
        "InferredIndexListing"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.inferred-listing", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(self.namespace);
        self.snapshot.write_key(builder);
    }
}

/// The complete index listing for commit bookkeeping. Version-zero FRI
/// coverage is translated, and tagged histories retain their stored provenance.
/// Query-time filters and coverage projections must never seed a commit.
pub async fn load_all_indices(
    table: &Table,
    hook: Option<&dyn CommitIndexHook>,
) -> Result<Arc<Vec<IndexMetadata>>> {
    let raw = table.load_index_metadata().await?;
    let metadata_key = table.index_metadata_key();
    let key = DerivedIndexListingKey {
        version: metadata_key.version,
        store_identity: metadata_key.store_identity,
        e_tag: metadata_key.e_tag,
    };
    let mut indices = listing_with_v0_coverage(table, raw, key).await?;
    if let Some(hook) = hook
        && indices
            .iter()
            .any(|index| hook.needs_legacy_details(table, index))
    {
        let inferred_key =
            hook.legacy_details_cache_namespace()
                .map(|namespace| InferredIndexListingKey {
                    snapshot: key,
                    namespace,
                });
        if let Some(key) = &inferred_key
            && let Some(cached) = table.index_cache().get_with_key(key).await
        {
            indices = cached;
            if !indices
                .iter()
                .any(|index| hook.needs_legacy_details(table, index))
            {
                return Ok(indices);
            }
        }
        let mut inferred = indices.as_ref().clone();
        hook.infer_legacy_details(table, &mut inferred).await;
        if inferred != *indices {
            indices = Arc::new(inferred);
            // Keep partial results while retrying unreadable entries on the
            // next read. The raw listing retains the manifest's original Any.
            if let Some(key) = inferred_key {
                table
                    .index_cache()
                    .insert_with_key(&key, indices.clone())
                    .await;
            }
        }
    }
    Ok(indices)
}

async fn listing_with_v0_coverage(
    table: &Table,
    raw: Arc<Vec<IndexMetadata>>,
    key: DerivedIndexListingKey<'_>,
) -> Result<Arc<Vec<IndexMetadata>>> {
    let mut entries = raw
        .iter()
        .filter(|index| index.name == FRAG_REUSE_INDEX_NAME);
    let Some(fri) = entries.next() else {
        return Ok(raw);
    };
    if is_tagged(fri) || entries.clone().any(is_tagged) {
        if entries.next().is_some() {
            return Err(Error::corrupt_file_named(
                "FRI metadata",
                "tagged history requires a single FRI entry",
            ));
        }
        return Ok(raw);
    }
    if let Some(cached) = table.index_cache().get_with_key(&key).await {
        return Ok(cached);
    }
    let identity = v0_cache_identity(&table.manifest, fri);
    let history = table
        .index_cache()
        .get_or_insert_with_key(FragReuseIndexKey { uuid: &identity }, || async {
            let details = load_frag_reuse_index_details(table, fri).await?;
            CompactFragReuseIndex::try_new(fri.uuid, Arc::unwrap_or_clone(details))
        })
        .await?;
    let mut remapped = raw.as_ref().clone();
    for index in &mut remapped {
        if let Some(bitmap) = &mut index.fragment_bitmap {
            history.remap_fragment_bitmap(bitmap)?;
        }
    }
    let remapped = Arc::new(remapped);
    table
        .index_cache()
        .insert_with_key(&key, remapped.clone())
        .await;
    Ok(remapped)
}

/// Metadata-only operations can publish the original entries when no user
/// index implementation is present. Coverage-sensitive operations still use
/// the version-zero FRI projection.
pub(super) async fn load_attempt_indices(
    table: &Table,
    operation: &Operation,
    hook: Option<&dyn CommitIndexHook>,
) -> Result<Arc<Vec<IndexMetadata>>> {
    if hook.is_none()
        && matches!(
            operation,
            Operation::Append { .. }
                | Operation::ReserveFragments { .. }
                | Operation::UpdateConfig { .. }
                | Operation::UpdateBases { .. }
                | Operation::UpdateMemWalState { .. }
        )
    {
        table.load_index_metadata().await
    } else {
        load_all_indices(table, hook).await
    }
}

pub fn index_usability(hook: Option<&dyn CommitIndexHook>) -> fn(&IndexMetadata) -> bool {
    // Every segment's provenance can pin history. Without an index reader,
    // no user segment can prove direct coverage that releases that history.
    hook.map_or(is_system_index, |hook| hook.usability_predicate())
}

pub(super) async fn validate_new_segments(
    hook: Option<&dyn CommitIndexHook>,
    table: &Table,
    operation: &Operation,
    candidate: &Manifest,
    indices: &[IndexMetadata],
) -> Result<()> {
    if let Some(hook) = hook {
        return hook
            .validate_new_segments(table, operation, candidate, indices)
            .await;
    }
    let added_names: BTreeSet<_> = match operation {
        Operation::CreateIndex { new_indices, .. } => new_indices
            .iter()
            .filter(|index| !is_system_index(index))
            .map(|index| index.name.as_str())
            .collect(),
        Operation::CompositeOperation(composite) => composite
            .iter_actions()
            .filter_map(|action| match action {
                Action::AddIndexSegment(index) => Some(index.name.as_str()),
                _ => None,
            })
            .collect(),
        _ => return Ok(()),
    };
    for name in added_names {
        let mut segments = indices.iter().filter(|index| index.name == name);
        if segments.clone().count() > 1
            && segments.any(|index| {
                index
                    .fields
                    .first()
                    .and_then(|&id| candidate.schema.field_by_id(id))
                    .is_some_and(|field| match field.data_type() {
                        DataType::FixedSizeList(_, _) => true,
                        DataType::List(inner) => {
                            matches!(inner.data_type(), DataType::FixedSizeList(_, _))
                        }
                        _ => false,
                    })
            })
        {
            return Err(Error::not_supported(format!(
                "Cannot validate the files of multiple segments of index '{name}' without a commit index hook"
            )));
        }
    }
    Ok(())
}
