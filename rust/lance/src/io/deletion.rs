// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Dataset;
use lance_core::utils::deletion::DeletionVector;
use lance_table::format::DeletionFile;
use std::sync::Arc;

pub async fn read_dataset_deletion_file(
    dataset: &Dataset,
    fragment_id: u64,
    deletion_file: &DeletionFile,
) -> lance_core::Result<Arc<DeletionVector>> {
    lance_table_runtime::io::deletion::read_table_deletion_file(
        &dataset.table_view().to_table(),
        fragment_id,
        deletion_file,
    )
    .await
}
