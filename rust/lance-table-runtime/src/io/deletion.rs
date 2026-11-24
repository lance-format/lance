// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Table;
use crate::caches::DeletionFileKey;
use lance_core::utils::deletion::DeletionVector;
use lance_table::format::DeletionFile;
use lance_table::io::deletion::read_deletion_file;
use std::sync::Arc;

/// Read and cache the deletion vector referenced by a fragment's metadata.
pub async fn read_table_deletion_file(
    table: &Table,
    fragment_id: u64,
    deletion_file: &DeletionFile,
) -> lance_core::Result<Arc<DeletionVector>> {
    let key = DeletionFileKey {
        fragment_id,
        deletion_file,
    };

    table
        .state
        .metadata_cache
        .get_or_insert_with_key(key, || async {
            // Resolve the store inside the loader so cache hits skip it.
            let dataset_dir = table.view().dataset_dir_for_deletion(deletion_file)?;
            let object_store = table.object_store(deletion_file.base_id).await?;
            read_deletion_file(
                fragment_id,
                deletion_file,
                &dataset_dir,
                object_store.as_ref(),
            )
            .await
        })
        .await
}

/// Whether a previously deleted row became live at the same physical address.
///
/// The loaders resolve deletion files through their respective snapshots. They
/// are only called when both fragments have different deletion files.
pub async fn revives_rows<'a, 'b, Before, After>(
    then: &'a lance_table::format::Fragment,
    now: &'b lance_table::format::Fragment,
    before: impl FnOnce(&'a DeletionFile) -> Before,
    after: impl FnOnce(&'b DeletionFile) -> After,
) -> lance_core::Result<bool>
where
    Before: std::future::Future<Output = lance_core::Result<Arc<DeletionVector>>>,
    After: std::future::Future<Output = lance_core::Result<Arc<DeletionVector>>>,
{
    deletion_mask_shrank(
        then.deletion_file.as_ref(),
        now.deletion_file.as_ref(),
        before,
        after,
    )
    .await
}

pub(crate) async fn deletion_mask_shrank<'a, 'b, Before, After>(
    then: Option<&'a DeletionFile>,
    now: Option<&'b DeletionFile>,
    before: impl FnOnce(&'a DeletionFile) -> Before,
    after: impl FnOnce(&'b DeletionFile) -> After,
) -> lance_core::Result<bool>
where
    Before: std::future::Future<Output = lance_core::Result<Arc<DeletionVector>>>,
    After: std::future::Future<Output = lance_core::Result<Arc<DeletionVector>>>,
{
    let Some(deleted_then) = then else {
        return Ok(false);
    };
    let Some(deleted_now) = now else {
        return Ok(true);
    };
    if deleted_now == deleted_then {
        return Ok(false);
    }
    let (rows_then, rows_now) = futures::try_join!(before(deleted_then), after(deleted_now))?;
    Ok(rows_then.iter().any(|row| !rows_now.contains(row)))
}

/// Table config key: the largest bitmap encoding, in bytes, of a deletion
/// vector that writers keep in the manifest rather than in a file under
/// `_deletions/`. Unset or `0` always writes a file.
///
/// A reader pays one request per deletion file it needs, so a table whose
/// fragments each lost a few rows pays one request per fragment on its first
/// take; inline vectors come with the manifest instead.
pub const INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY: &str = "lance.deletions.inline_max_bytes";

/// The configured maximum bytes of an inline deletion vector, or zero when unset.
pub fn inline_deletions_max_bytes(
    manifest: &lance_table::format::Manifest,
) -> lance_core::Result<usize> {
    let Some(value) = manifest.config.get(INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY) else {
        return Ok(0);
    };
    value.trim().parse().map_err(|err| {
        lance_core::Error::invalid_input(format!(
            "table config {INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY} must be a number of bytes, got `{value}`: {err}"
        ))
    })
}
