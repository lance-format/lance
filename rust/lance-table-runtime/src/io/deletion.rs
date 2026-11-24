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
