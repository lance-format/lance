// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result, utils::deletion::DeletionVector};
use lance_table::io::deletion::write_deletion_file;

use crate::FileFragment;
use crate::io::deletion::inline_deletions_max_bytes;

impl FileFragment {
    /// Stage deletions by physical row offset, merging with existing deletions.
    ///
    /// Returns `None` when the entire fragment is deleted. Publish the resulting
    /// metadata with a delete transaction to make the changes visible.
    pub async fn extend_deletions(
        self,
        new_deletions: impl IntoIterator<Item = u32>,
    ) -> Result<Option<Self>> {
        let mut deletion_vector = self
            .get_deletion_vector()
            .await?
            .unwrap_or_default()
            .as_ref()
            .clone();

        deletion_vector.extend(new_deletions);

        self.write_deletions(deletion_vector).await
    }

    #[doc(hidden)]
    pub async fn write_deletions(
        mut self,
        deletion_vector: DeletionVector,
    ) -> Result<Option<Self>> {
        let physical_rows = self.physical_rows().await?;
        if deletion_vector.len() == physical_rows
            && deletion_vector.contains_range(0..physical_rows as u32)
        {
            return Ok(None);
        } else if deletion_vector.len() >= physical_rows {
            let dv_len = deletion_vector.len();
            let examples: Vec<u32> = deletion_vector
                .into_iter()
                .filter(|x| *x >= physical_rows as u32)
                .take(5)
                .collect();
            return Err(Error::internal(format!(
                "Deletion vector includes rows that aren't in the fragment. \
            Num physical rows {}; Deletion vector length: {}; \
            Examples: {:?}",
                physical_rows, dv_len, examples
            )));
        }

        let deletion_file = write_deletion_file(
            &self.table().state.base,
            self.metadata().id,
            self.table().version_id(),
            &deletion_vector,
            self.table().state.object_store.as_ref(),
            inline_deletions_max_bytes(self.table().manifest())?,
        )
        .await?;
        self.metadata_mut().deletion_file = deletion_file;

        Ok(Some(self))
    }
}
