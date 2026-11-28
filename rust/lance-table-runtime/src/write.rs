// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Physical table writers, streaming inserts, and host callbacks.

mod bases;
#[doc(hidden)]
pub mod dictionary_budget;
mod file;
mod fragments;
mod insert;
mod params;
mod parts;
pub mod progress;
pub(crate) mod schema;
mod schema_adapter;
mod seeds;

#[doc(hidden)]
pub use bases::{
    PRIMARY_BASE_ID, TargetBaseInfo, validate_and_resolve_target_bases,
    validate_and_resolve_target_bases_with_primary,
};
#[doc(hidden)]
pub use file::{
    GenericWriter, open_update_writer, open_update_writer_with_options, writes_blob_v2,
};
#[doc(hidden)]
pub use fragments::{FragmentWriteOptions, cleanup_data_fragments, write_fragments};
pub use insert::InsertBuilder;
#[doc(hidden)]
pub use params::WriteParams as WriteParamsWithSession;
pub use params::{AutoCleanupParams, ExternalBlobMode, WriteMode};

/// Options for writing a table with a [`crate::TableSession`].
pub type WriteParams = WriteParamsWithSession<crate::TableSession>;
#[doc(hidden)]
pub use bases::blob_v2_external_base_resolver;
pub(crate) use file::{WriterOptions, open_writer};
pub(crate) use fragments::write_fragments_direct;
pub use parts::{DataFilePart, DataFileTarget};
pub use progress::{WriteProgressFn, WriteStats};
#[doc(hidden)]
pub use schema::{promote_legacy_blob_schema, schema_compare_options, validate_row_limits};
#[doc(hidden)]
pub use schema_adapter::SchemaAdapter;
#[doc(hidden)]
pub use seeds::SeedCollector;
pub use seeds::WriteSeedHook;
