// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Rebase dataset transactions over concurrent commits.

use std::sync::Arc;

use lance_select::RowAddrTreeMap;
use lance_table::transaction::{TaggedRewriteAssembly, Transaction};
use lance_table_runtime::commit::conflict_resolver as runtime;

use crate::{Dataset, Result};

use super::LanceCommitHook;

/// Rebase a staged transaction against a dataset's committed changes.
#[derive(Debug)]
pub struct TransactionRebase<'a>(runtime::TransactionRebase<'a>);

impl<'a> TransactionRebase<'a> {
    /// Capture the staged transaction and its read snapshot.
    pub async fn try_new(
        dataset: &Dataset,
        transaction: Transaction,
        affected_rows: Option<&'a RowAddrTreeMap>,
    ) -> Result<Self> {
        Ok(Self(
            runtime::TransactionRebase::try_new(
                &dataset.table_view().to_table(),
                transaction,
                affected_rows,
            )
            .await?,
        ))
    }

    pub(crate) fn for_staged_replay(self) -> Self {
        Self(self.0.for_staged_replay())
    }

    /// Load the landing snapshot metadata needed by the conflict checks.
    pub async fn load_current_lineage(&mut self, dataset: &Dataset) -> Result<()> {
        self.0
            .load_current_lineage(&dataset.table_view().to_table())
            .await
    }

    /// Check a committed transaction and record its effect on this rebase.
    pub fn check_txn(&mut self, transaction: &Transaction, version: u64) -> Result<()> {
        self.0.check_txn(transaction, version)
    }

    /// Finish the rebase, writing merged deletion vectors when required.
    pub async fn finish(self, dataset: &Dataset) -> Result<Transaction> {
        Ok(self.finish_with_tagged_rewrite(dataset).await?.0)
    }

    /// Finish the rebase and return any tagged fragment-reuse history assembled
    /// for the manifest build.
    pub async fn finish_with_tagged_rewrite(
        self,
        dataset: &Dataset,
    ) -> Result<(Transaction, Option<Arc<TaggedRewriteAssembly>>)> {
        let hook = LanceCommitHook::new(dataset.session.clone());
        self.0
            .finish_with_index_hook(&dataset.table_view().to_table(), Some(&hook))
            .await
    }
}

#[cfg(test)]
mod tests;
