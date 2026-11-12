// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use crate::session::caches::DeletionFileKey;
use lance_core::utils::deletion::DeletionVector;
use lance_table::format::DeletionFile;
use lance_table::io::deletion::read_deletion_file;
use std::sync::Arc;

pub async fn read_dataset_deletion_file(
    dataset: &Dataset,
    fragment_id: u64,
    deletion_file: &DeletionFile,
) -> lance_core::Result<Arc<DeletionVector>> {
    let key = DeletionFileKey {
        fragment_id,
        deletion_file,
    };

    dataset
        .metadata_cache
        .get_or_insert_with_key(key, || async {
            // Resolve the store inside the loader so cache hits skip it.
            let dataset_dir = dataset.dataset_dir_for_deletion(deletion_file)?;
            let object_store = dataset.object_store_for_deletion(deletion_file).await?;
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
