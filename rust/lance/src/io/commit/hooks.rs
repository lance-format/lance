// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use lance_core::{Error, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_io::object_store::ObjectStore;
use lance_table::format::{IndexMetadata, Manifest};
use lance_table::transaction::Operation;
use lance_table_runtime::Table;
use lance_table_runtime::commit::{CommitHooks, CommitIndexHook, CommitLifecycleHook};
use object_store::path::Path;
use roaring::RoaringBitmap;

use crate::dataset::cleanup::auto_cleanup_hook;
use crate::dataset::index::frag_reuse::derive_superseded_segments;
use crate::index::field_dependencies::input_field_ids;
use crate::index::frag_reuse_reader::segment_supports_batch_remapping;
use crate::index::frag_reuse_with_stable_row_ids::validate_stable_row_id_user_indices;
use crate::index::loading::DatasetIndexInternalExt;
use crate::index::segment::index_is_usable;
use crate::index::vector::details::{infer_missing_vector_details, needs_vector_details_inference};
use crate::index::vector::ivf::validate_committed_vector_segments;
use crate::utils::temporal::timestamp_to_nanos;
use crate::{Dataset, session::Session};

pub struct LanceCommitHook {
    session: Arc<Session>,
}

impl LanceCommitHook {
    pub(crate) fn new(session: Arc<Session>) -> Self {
        Self { session }
    }

    pub(crate) fn hooks(&self) -> CommitHooks<'_> {
        CommitHooks {
            index: Some(self),
            lifecycle: Some(self),
        }
    }

    pub(crate) fn dataset(&self, table: &Table) -> Dataset {
        Dataset::from_committed_table(table.clone(), self.session.clone())
    }
}

#[async_trait]
impl CommitIndexHook for LanceCommitHook {
    fn usability_predicate(&self) -> fn(&IndexMetadata) -> bool {
        index_is_usable
    }

    fn needs_legacy_details(&self, table: &Table, index: &IndexMetadata) -> bool {
        needs_vector_details_inference(index, table.schema())
    }

    fn legacy_details_cache_namespace(&self) -> Option<&'static str> {
        Some("lance.vector-details")
    }

    async fn infer_legacy_details(&self, table: &Table, indices: &mut [IndexMetadata]) {
        infer_missing_vector_details(&self.dataset(table), indices).await;
    }

    async fn recover_coverage(
        &self,
        table: &Table,
        index: &IndexMetadata,
    ) -> Result<RoaringBitmap> {
        // Additional dependencies and carried columns do not change the one
        // query input used to recover the bitmap.
        let inputs = input_field_ids(index)?;
        let field = match inputs.as_ref() {
            [id] => table.schema().field_by_id(*id),
            _ => None,
        }
        .ok_or_else(|| {
            Error::internal(format!(
                "Cannot recover coverage of index {} with keyed fields {:?}",
                index.uuid, index.fields,
            ))
        })?;
        let dataset = self.dataset(table);
        let index = dataset
            .open_generic_index(&field.name, &index.uuid, &NoOpMetricsCollector)
            .await?;
        index.calculate_included_frags().await
    }

    async fn validate_new_segments(
        &self,
        table: &Table,
        operation: &Operation,
        candidate: &Manifest,
        indices: &[IndexMetadata],
    ) -> Result<()> {
        validate_committed_vector_segments(&self.dataset(table), operation, candidate, indices)
            .await
    }

    async fn supports_batch_remapping(
        &self,
        table: &Table,
        index: &IndexMetadata,
    ) -> Result<Option<bool>> {
        segment_supports_batch_remapping(&self.dataset(table), index).await
    }

    fn validate_stable_row_id_indices(
        &self,
        manifest: &Manifest,
        indices: &[IndexMetadata],
    ) -> Result<()> {
        validate_stable_row_id_user_indices(manifest, indices)
    }

    async fn superseded_segments(&self, table: &Table) -> Result<Vec<IndexMetadata>> {
        derive_superseded_segments(&self.dataset(table)).await
    }
}

#[async_trait]
impl CommitLifecycleHook for LanceCommitHook {
    async fn copy_managed_blobs(
        &self,
        source: Table,
        destination_store: &ObjectStore,
        destination_base: &Path,
        destination_uri: &str,
        manifest: &mut Manifest,
    ) -> Result<()> {
        let source = Arc::new(Dataset::from_table(source, self.session.clone()));
        crate::dataset::blob::clone::copy_blob_columns(
            source,
            Arc::new(destination_store.clone()),
            destination_base.clone(),
            destination_uri,
            manifest,
        )
        .await
    }

    async fn after_commit(&self, table: &Table, manifest: &Manifest) {
        match auto_cleanup_hook(&self.dataset(table), manifest).await {
            Ok(Some(stats)) => log::info!("Auto cleanup triggered: {:?}", stats),
            Err(e) => log::warn!("Auto cleanup failed after a successful commit: {}", e),
            _ => {}
        }
    }

    fn timestamp_nanos(&self) -> u128 {
        timestamp_to_nanos(None)
    }

    #[cfg(test)]
    fn inline_transaction_limit(&self) -> usize {
        super::MAX_INLINE_TRANSACTION_BYTES
    }
}
