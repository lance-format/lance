// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Streaming remapping of one committed index during data rewrites.

// These helpers stay crate-visible so callers do not depend on private parameter types.
#![allow(clippy::redundant_pub_crate)]

use std::sync::Arc;

use lance_core::utils::address::RowAddress;
use lance_index::scalar::RowAddrTranslator;
use lance_index::scalar::inverted::InvertedIndexPlugin;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::{TrainingCriteria, TrainingOrdering};
use lance_index::scalar::zonemap::{ZoneMapIndex, ZoneMapRemapNeedsWithdrawal};
use lance_index::scalar::{CreatedIndex, index_files_to_table, table_files_to_index};
use lance_index::{IndexType, metrics::NoOpMetricsCollector, progress::NoopIndexBuildProgress};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::dataset::index::LanceIndexStoreExt;
use crate::dataset::optimize::RemappedIndex;
use crate::dataset::optimize::remapping::RemapResult;
use crate::index::frag_reuse;
use crate::index::loading::DatasetIndexInternalExt;
use crate::index::scalar::{
    IndexDetails, TrainingScanOptions, fetch_index_details, load_training_data,
};
use crate::index::vector::details::vector_index_details_default;
use crate::index::vector::remap_vector_index;
use crate::{Dataset, Error, Result};

/// Addresses checked per translation while deciding whether a remap deletes
/// every indexed row; the check stops at the first surviving row.
const REMAP_DELETE_CHECK_BATCH: u32 = 64 * 1024;

async fn remap_deletes_all_indexed_rows(
    dataset: &Dataset,
    indexed_fragments: &RoaringBitmap,
    row_id_map: &RowAddrTranslator,
) -> Result<bool> {
    for fragment_id in indexed_fragments.iter() {
        let Some(fragment) = dataset.get_fragment(fragment_id as usize) else {
            return Ok(false);
        };
        let Some(physical_rows) = fragment.metadata().physical_rows else {
            // Legacy fragments may not record their physical row count, so the
            // remap cannot prove that every possible address was deleted.
            return Ok(false);
        };
        let Ok(physical_rows) = u32::try_from(physical_rows) else {
            return Ok(false);
        };
        for start in (0..physical_rows).step_by(REMAP_DELETE_CHECK_BATCH as usize) {
            let end = start
                .saturating_add(REMAP_DELETE_CHECK_BATCH)
                .min(physical_rows);
            let addrs: Vec<u64> = (start..end)
                .map(|offset| u64::from(RowAddress::new_from_parts(fragment_id, offset)))
                .collect();
            if row_id_map
                .remap_row_addrs(&addrs)
                .await?
                .iter()
                .any(|row| row.is_some())
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Restrict rows to `owned_fragments` only for sources of a staging merge,
/// before translating their addresses. Independent and deferred remaps pass None.
///
/// A remapped segment that would have to stop covering some destination
/// fragments (see [`remap_index_with_withdrawals`]) is dropped here: these
/// callers publish coverage that follows whole rewrite groups and cannot
/// subtract fragments from it, and dropping is what they did for such an
/// index type before it could be remapped.
pub(crate) async fn remap_index(
    dataset: &Dataset,
    index_id: &Uuid,
    row_id_map: &RowAddrTranslator,
    owned_fragments: Option<&RoaringBitmap>,
) -> Result<RemapResult> {
    let (result, withdrawn_fragments) =
        remap_index_with_withdrawals(dataset, index_id, row_id_map, owned_fragments).await?;
    if withdrawn_fragments.is_empty() {
        return Ok(result);
    }
    log::warn!(
        "Remapping index {} would withdraw its coverage of fragments {:?}, which this \
         rewrite cannot publish, so the index is not remapped. Rebuild it to cover them.",
        index_id,
        withdrawn_fragments.iter().collect::<Vec<_>>()
    );
    if let RemapResult::Remapped(remapped) = &result {
        let index_dir = dataset.indices_dir().join(remapped.new_id.to_string());
        if let Err(error) = dataset.object_store.remove_dir_all(index_dir).await {
            log::warn!(
                "Failed to remove dropped remapped index {}: {}",
                remapped.new_id,
                error
            );
        }
    }
    Ok(RemapResult::Drop)
}

/// [`remap_index`] for a caller that can narrow the published coverage: the
/// second value lists the destination fragments the remapped segment must
/// not claim, because no sound statistics could be written for some of
/// their rows (today only a ZoneMap reports any). Empty for every other
/// outcome.
pub(crate) async fn remap_index_with_withdrawals(
    dataset: &Dataset,
    index_id: &Uuid,
    row_id_map: &RowAddrTranslator,
    owned_fragments: Option<&RoaringBitmap>,
) -> Result<(RemapResult, RoaringBitmap)> {
    let mut withdrawn_fragments = RoaringBitmap::new();
    // A remap is maintenance: a segment the tagged reader excludes from the
    // listing is still reachable here (its caller decided whether to touch
    // it), so the metadata comes from the maintenance lookup.
    let matched = dataset
        .load_index_with_purpose(index_id, frag_reuse::OpenPurpose::Maintenance)
        .await?
        .ok_or_else(|| Error::index(format!("Index with id {} does not exist", index_id)))?;
    let matched = &matched;

    // Corrupt metadata fails closed before anything else: a declaration that
    // breaks the covering rules cannot be reasoned about at all, and the
    // withdrawal below would otherwise swallow it as an ordinary covered index.
    matched.validate_covering_fields()?;

    // A covered index cannot survive a remap yet. Nothing writes the carried
    // values in the first place, and each index type's `remap` rewrites only the
    // schema that type knows about, so the remapped segment would still declare
    // payload its storage no longer holds. Withdraw it instead, exactly as an
    // index whose plugin reports `can_remap(details) == false` is withdrawn below: an
    // absent index costs a fallback scan, whereas a surviving false declaration
    // is answered from data that is not there. Erroring here would instead block
    // compaction of the whole table.
    if !matched.covering_fields.is_empty() {
        log::warn!(
            "Index '{}' declares covering fields {:?}, which no index builder \
             writes or preserves yet, so it is not remapped. Rebuild it to \
             cover the rewritten fragments.",
            matched.name,
            matched.covering_fields,
        );
        return Ok((RemapResult::Drop, withdrawn_fragments));
    }

    // With covering withdrawn above, `fields` is entirely keyed, so more than one
    // entry means a genuinely composite index.
    if matched.fields.len() > 1 {
        return Err(Error::index(format!(
            "Remapping index '{}' is not supported: it has {} keyed fields {:?}; \
             only one keyed field is supported",
            matched.name,
            matched.fields.len(),
            matched.fields,
        )));
    }

    if let Some(fragments) = matched.fragment_bitmap.as_ref()
        && remap_deletes_all_indexed_rows(dataset, fragments, row_id_map).await?
    {
        // If remap deleted all rows, we can just return the same index ID.
        // This can happen if there is a bug where the index is covering empty
        // fragment that haven't been cleaned up. They should be cleaned up
        // outside of this function.
        return Ok((RemapResult::Keep(*index_id), withdrawn_fragments));
    }

    let field_id = matched
        .fields
        .first()
        .expect("An index existed with no fields");
    let field_path = dataset.schema().field_path(*field_id)?;

    let new_id = Uuid::new_v4();

    let generic = match dataset
        .open_generic_index(&field_path, index_id, &NoOpMetricsCollector)
        .await
    {
        Ok(g) => g,
        Err(e) => {
            log::warn!(
                "Cannot open index '{}' on '{}': {}, so it is not remapped. Rebuild the index.",
                index_id,
                field_path,
                e
            );
            return Ok((RemapResult::Drop, withdrawn_fragments));
        }
    };

    let created_index = match generic.index_type() {
        it if it.is_scalar() => {
            let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_id)?;

            let scalar_index = dataset
                .open_scalar_index(&field_path, index_id, &NoOpMetricsCollector)
                .await?;
            // Keep the deprecated instance-level check for external indexes
            // written before ScalarIndexPlugin::can_remap existed. The plugin
            // capability below covers built-ins and new implementations.
            #[allow(deprecated)]
            if !scalar_index.can_remap() {
                return Ok((RemapResult::Drop, withdrawn_fragments));
            }
            let details = fetch_index_details(dataset, &field_path, matched).await?;
            if !IndexDetails(details.clone())
                .get_plugin()?
                .can_remap(details.as_ref())?
            {
                return Ok((RemapResult::Drop, withdrawn_fragments));
            }

            match scalar_index.index_type() {
                IndexType::Inverted => {
                    let inverted_index = scalar_index
                        .as_any()
                        .downcast_ref::<lance_index::scalar::inverted::InvertedIndex>()
                        .ok_or(Error::index("expected inverted index".to_string()))?;
                    if inverted_index.is_legacy() {
                        log::warn!(
                            "reindex because of legacy format, index_type: {}, index_id: {}, field: {}",
                            scalar_index.index_type(),
                            index_id,
                            field_path
                        );
                        let training_data = load_training_data(
                            dataset,
                            &field_path,
                            &TrainingCriteria::new(TrainingOrdering::None),
                            None,
                            true, // Legacy reindexing should always train
                            None,
                            TrainingScanOptions::default(),
                        )
                        .await?;
                        InvertedIndexPlugin::train_inverted_index(
                            training_data,
                            &new_store,
                            inverted_index.params().clone(),
                            None,
                            Arc::new(NoopIndexBuildProgress),
                        )
                        .await?
                    } else {
                        scalar_index.remap_streaming(row_id_map, &new_store).await?
                    }
                }
                _ => {
                    // A ZoneMap remapped through a synchronous map reports
                    // the coverage it withdraws; through a batch translator
                    // it takes the trait default, which declines.
                    let zonemap = scalar_index.as_any().downcast_ref::<ZoneMapIndex>();
                    match (row_id_map, zonemap) {
                        (RowAddrTranslator::Sync(mapping), Some(zonemap)) => {
                            let (created, withdrawn) =
                                zonemap.remap_with_withdrawals(mapping, &new_store).await?;
                            withdrawn_fragments = withdrawn;
                            created
                        }
                        _ => match scalar_index.remap_streaming(row_id_map, &new_store).await {
                            Ok(created) => created,
                            // A wrapper over a ZoneMap reaches the plain
                            // remap, which cannot report withdrawn coverage
                            // and refuses instead.
                            Err(error) => match ZoneMapRemapNeedsWithdrawal::from_error(&error) {
                                Some(reason) => {
                                    log::warn!(
                                        "Index '{}' on '{}' is not remapped: {reason}. Rebuild \
                                         the index.",
                                        index_id,
                                        field_path
                                    );
                                    return Ok((RemapResult::Drop, withdrawn_fragments));
                                }
                                None => return Err(error),
                            },
                        },
                    }
                }
            }
        }
        it if it.is_vector() => {
            let index_version = u32::try_from(matched.index_version).map_err(|_| {
                Error::index(format!(
                    "Invalid vector index version {} on index {}",
                    matched.index_version, matched.name
                ))
            })?;
            let files = remap_vector_index(
                Arc::new(dataset.clone()),
                &field_path,
                index_id,
                &new_id,
                matched,
                row_id_map,
                owned_fragments,
            )
            .await?;

            CreatedIndex {
                // A remap keeps the IVF model, quantizer and sub-index; only row
                // addresses change, so the details carry over.
                index_details: matched
                    .index_details
                    .as_deref()
                    .cloned()
                    .unwrap_or_else(vector_index_details_default),
                index_version,
                files: table_files_to_index(files),
            }
        }
        _ => {
            return Err(Error::index(format!(
                "Index type {} is not supported",
                generic.index_type()
            )));
        }
    };

    Ok((
        RemapResult::Remapped(RemappedIndex {
            old_id: *index_id,
            new_id,
            index_details: created_index.index_details,
            index_version: created_index.index_version,
            files: Some(index_files_to_table(created_index.files)),
        }),
        withdrawn_fragments,
    ))
}

#[cfg(test)]
mod tests;
