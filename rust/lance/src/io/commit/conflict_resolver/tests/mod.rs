// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::num::NonZero;
use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use lance_core::{Error, utils::deletion::DeletionVector};
use lance_file::version::LanceFileVersion;
use lance_io::assert_io_eq;
use lance_select::RowAddrTreeMap;
use lance_table::format::{Fragment, IndexMetadata};
use lance_table::io::deletion::{deletion_file_path, read_deletion_file, write_deletion_file};
use lance_table::system_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_table::system_index::mem_wal::{CompactedSsTable, MEM_WAL_INDEX_NAME};
use lance_table::transaction::{Operation, RewriteGroup, UpdateMap};
use uuid::Uuid;

use super::*;
use crate::dataset::write::WriteMode;
use crate::dataset::{CommitBuilder, InsertBuilder, WriteParams};
use crate::session::caches::DeletionFileKey;

mod index_and_config;
mod rebase;

/// Concurrent-writer coverage for the stable-partition rows of the
/// conflict matrix. Where cross-process behavior matters, the second
/// writer reopens the dataset with a fresh session so nothing rides the
/// shared in-memory cache: every decision has to come from transaction
/// files and the manifests themselves.
mod tagged_rewrite_conflicts;

async fn test_dataset(num_rows: usize, num_fragments: usize) -> Dataset {
    let write_params = WriteParams {
        max_rows_per_file: num_rows / num_fragments,
        ..Default::default()
    };
    let data = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, true),
        ])),
        vec![
            Arc::new(Int32Array::from_iter_values(0..num_rows as i32)),
            Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(
                0, num_rows,
            ))),
        ],
    )
    .unwrap();

    InsertBuilder::new("memory://")
        .with_params(&write_params)
        .execute(vec![data])
        .await
        .unwrap()
}

async fn apply_deletion(
    delete_rows: &[u32],
    fragment: &mut Fragment,
    dataset: &Dataset,
) -> Fragment {
    let mut current_deletions = if let Some(deletion_file) = &fragment.deletion_file {
        read_deletion_file(
            fragment.id,
            deletion_file,
            // Reference deletion file should never enter this apply_deletion. So base path is fine.
            &dataset.table_state.base,
            dataset.table_state.object_store.as_ref(),
        )
        .await
        .unwrap()
    } else {
        DeletionVector::default()
    };

    current_deletions.extend(delete_rows.iter().copied());

    fragment.deletion_file = write_deletion_file(
        &dataset.table_state.base,
        fragment.id,
        dataset.manifest.version,
        &current_deletions,
        dataset.table_state.object_store.as_ref(),
        0,
    )
    .await
    .unwrap();

    let deletion_file = fragment.deletion_file.as_ref().unwrap();
    let key = DeletionFileKey {
        fragment_id: fragment.id,
        deletion_file,
    };
    dataset
        .table_state
        .metadata_cache
        .insert_with_key(&key, Arc::new(current_deletions))
        .await;

    fragment.clone()
}
