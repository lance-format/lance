// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Publish transactions over prewritten table fragments.

mod builder;
mod create;
mod detached;
mod hooks;
pub(crate) mod index;
mod migration;
mod publish;
mod transaction;
mod transactions;

#[doc(hidden)]
pub mod conflict_resolver;
#[doc(hidden)]
pub mod retry;

pub use builder::{BatchCommitResult, CommitBuilder, CommitDestination, DEFAULT_COMMIT_TIMEOUT};
pub use hooks::CommitIndexHook;

#[doc(hidden)]
pub use builder::resolve_commit_handler;
#[doc(hidden)]
pub use create::{assign_new_table_field_ids, commit_new_dataset};
#[doc(hidden)]
pub use hooks::{CommitHooks, CommitLifecycleHook};
#[doc(hidden)]
pub use index::load_all_indices;
#[doc(hidden)]
pub use migration::{
    BadFragmentBitmapError, detect_overlapping_fragments, fix_schema, manifest_needs_migration,
    migrate_fragments,
};
#[doc(hidden)]
pub use publish::{
    MAX_INLINE_TRANSACTION_BYTES, read_transaction_file, write_manifest_file,
    write_transaction_file,
};
#[doc(hidden)]
pub use transaction::{CommitTransactionOptions, commit_transaction};
#[doc(hidden)]
pub use transactions::{TransactionReadSource, load_and_sort_new_transactions};

#[cfg(feature = "dynamodb")]
pub use lance_table::io::commit::dynamodb;
pub use lance_table::io::commit::external_manifest;

#[cfg(test)]
mod tests;
