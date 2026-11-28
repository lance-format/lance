// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Dataset
//!

use arrow_array::{RecordBatch, RecordBatchReader};
use chrono::Duration;
use futures::future::BoxFuture;
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::{FutureExt, Stream};
use lance_core::deepsize::DeepSizeOf;

use crate::dataset::metadata::UpdateFieldMetadataBuilder;
use crate::dataset::transaction::translate_schema_metadata_updates;
use crate::session::index_caches::DSIndexCache;
use itertools::Itertools;
use lance_core::ROW_ADDR;
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Projectable, Projection};
use lance_core::memory::ObservedOperation;
use lance_core::traits::DatasetTakeRows;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tracing::{
    DATASET_DELETING_EVENT, DATASET_DROPPING_COLUMN_EVENT, TRACE_DATASET_EVENTS,
};
use lance_datafusion::projection::ProjectionPlan;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::versions as file_versions;
use lance_index::{IndexType, progress::IndexBuildProgress};
use lance_io::object_store::{
    LanceNamespaceStorageOptionsProvider, ObjectStore, ObjectStoreParams, StorageOptions,
    StorageOptionsAccessor, StorageOptionsProvider, WrappingObjectStore,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_namespace::LanceNamespace;
use lance_table::format::{
    DataFile, Fragment, IndexMetadata, Manifest, ManifestBuildConfig, RowIdMeta,
};
use lance_table::io::commit::{
    CommitConfig, CommitHandler, CommitLock, ManifestLocation, ManifestNamingScheme, VERSIONS_DIR,
    external_manifest::ExternalManifestCommitHandler, migrate_scheme_to_v2,
};

use crate::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;
use lance_table::io::manifest::{read_manifest, read_manifest_indexes};
use lance_table_runtime::base::has_any_entry;
pub(crate) use lance_table_runtime::paths::{DATA_DIR, INDICES_DIR, TRANSACTIONS_DIR};
use lance_table_runtime::{Table, TableState, TableView};
pub use lance_table_runtime::{Version, VersionRef, VersionTransaction};
use object_store::path::Path;
use rand::{Rng, seq::index::sample as sample_indices};
use roaring::RoaringBitmap;
use rowids::get_row_id_index;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::num::NonZero;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{Instrument, debug_span, field::Empty, info, instrument, warn};

pub(crate) mod blob;
pub(crate) use lance_table_runtime::branch_location;
pub mod builder;
pub mod cleanup;
pub mod compaction_stats;
mod data_file;
pub mod deletions;
pub mod delta;
pub mod expire;
pub mod files;
pub mod fragment;
pub(crate) mod fragment_slice;
mod hash_joiner;
pub mod index;
pub mod mem_wal;
mod metadata;
pub mod optimize;
pub(crate) mod overlay;
pub mod progress;
pub use lance_table_runtime::refs;
pub mod rowids;
pub mod scanner;
mod schema_evolution;
pub mod sql;
pub mod statistics;
mod take;
/// Transaction definitions for updating datasets
///
/// Prior to creating a new manifest, a transaction must be created representing
/// the changes being made to the dataset. By representing them as incremental
/// changes, we can detect whether concurrent operations are compatible with
/// one another. We can also rebuild manifests when retrying committing a
/// manifest.
///
/// The definitions live in [`lance_table::transaction`]: building a manifest from
/// a transaction reads and writes only table metadata, so it belongs at the table
/// layer. This module re-exports them at the path callers have always used.
///
/// For more details please refer to the
/// [Transaction Specification](https://lance.org/format/table/transaction/#transaction-types).
pub mod transaction {
    pub use lance_table::transaction::{
        DataOverlayGroup, DataReplacementGroup, Operation, ReadVersionState, RewriteGroup,
        RewrittenIndex, TaggedRewriteAssembly, Transaction, TransactionBuilder, UpdateMap,
        UpdateMapEntry, UpdateMode, UpdatedFragmentOffsets, translate_config_updates,
        translate_schema_metadata_updates, validate_operation,
    };
}
pub mod udtf;
mod update_join;
#[cfg(feature = "update-join-bench")]
#[doc(hidden)]
pub mod update_join_bench;
mod update_join_options;
pub mod updater;
mod utils;
pub mod version_retention;
pub(crate) mod versions;
pub mod write;

pub use data_file::DataFileTarget;
pub use lance_table_runtime::write::DataFilePart;

pub(crate) use take::row_offsets_to_row_addresses;

use self::builder::DatasetBuilder;
use self::cleanup::RemovalStats;
use self::fragment::FileFragment;
use self::refs::Refs;
use self::scanner::{DatasetRecordBatchStream, Scanner};
use self::statistics::DatasetStatistics;
use self::transaction::{Operation, Transaction, TransactionBuilder, UpdateMapEntry};
use crate::dataset::branch_location::BranchLocation;
use crate::dataset::cleanup::{CleanupOperation, CleanupPolicy, CleanupPolicyBuilder};
use crate::dataset::refs::{BranchContents, BranchIdentifier, BranchReservation, Branches, Tags};
use crate::dataset::sql::SqlQueryBuilder;
use crate::datatypes::Schema;
use crate::index::memory_progress::ObservedBuild;
use crate::io::commit::{
    CommitTransactionOptions, commit_transaction, detect_overlapping_fragments, fix_schema,
};
use crate::session::Session;
use crate::utils::temporal::utc_now;
use crate::{Error, Result};
pub use blob::{
    BlobFile, BlobRangeRequest, BlobReadRange, BlobV2BatchRewritePlan, ReadBlob, ReadBlobRange,
    ReadBlobRangesBuilder, ReadBlobRangesStream, ReadBlobsBuilder, ReadBlobsStream,
    preserve_managed_descriptors,
};
use hash_joiner::HashJoiner;
pub use lance_core::ROW_ID;
use lance_index::scalar::BuiltinIndexType;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_namespace::models::{DeclareTableRequest, DescribeTableRequest};
use lance_table::feature_flags::ensure_can_write_manifest;
use lance_table::io::deletion::relative_deletion_file_path;
use lance_table::rowids::{RowIdSequence, write_row_ids};
pub use overlay::writer::{OverlayWriter, WriteOverlayError};
pub use schema_evolution::{
    BatchInfo, BatchUDF, ColumnAlteration, NewColumnTransform, UDFCheckpointStore,
};
pub use take::TakeBuilder;
pub use update_join_options::{UpdateJoinOptions, UpdateJoinStrategy};
use uuid::Uuid;
pub use write::merge_insert::{
    MergeInsertAnalysis, MergeInsertBuilder, MergeInsertJob, MergeInsertJoin,
    MergeInsertJoinStrategy, MergeInsertWriteMode, MergeInsertWriteStrategy, MergeOperatorSpills,
    MergeStats, MergeSubplan, SourceDedupeBehavior, UncommittedMergeInsert, WhenMatched,
    WhenNotMatched, WhenNotMatchedBySource,
};

use crate::dataset::index::LanceIndexStoreExt;
pub use write::update::{UpdateBuilder, UpdateJob};
#[allow(deprecated)]
pub use write::{
    AutoCleanupParams, CommitBuilder, DEFAULT_COMMIT_TIMEOUT, DeleteBuilder, DeleteResult,
    ExternalBlobMode, InsertBuilder, UncommittedDelete, WriteDestination, WriteMode, WriteParams,
    WriteProgressFn, WriteStats, write_fragments,
};

const DEFAULT_MAX_STREAM_COPY_PARALLELISM: usize = 4;

// Full samples already have their final ordering and need no randomness.
fn sample_row_offsets(rng: &mut impl Rng, num_rows: usize, n: usize) -> Vec<u64> {
    let sample_size = n.min(num_rows);
    if sample_size == 0 {
        return Vec::new();
    }
    if sample_size == num_rows {
        return (0..num_rows as u64).collect();
    }
    let mut offsets: Vec<u64> = sample_indices(rng, num_rows, sample_size)
        .into_iter()
        .map(|offset| offset as u64)
        .collect();
    offsets.sort_unstable();
    offsets
}

fn parse_deep_clone_stream_concurrency(value: &str) -> Result<usize> {
    value
        .parse::<NonZero<usize>>()
        .map(NonZero::get)
        .map_err(|_| {
            Error::invalid_input(format!(
                "LANCE_DEEP_CLONE_STREAM_CONCURRENCY must be a positive integer, got {value:?}"
            ))
        })
}

fn deep_clone_copy_parallelism(
    configured_io_parallelism: usize,
    uses_streaming_copy: bool,
    stream_copy_parallelism: Option<usize>,
) -> usize {
    if !uses_streaming_copy {
        configured_io_parallelism
    } else if let Some(value) = stream_copy_parallelism {
        value
    } else {
        configured_io_parallelism.min(DEFAULT_MAX_STREAM_COPY_PARALLELISM)
    }
}

// We default to 6GB for the index cache, since indices are often large but
// worth caching.
pub const DEFAULT_INDEX_CACHE_SIZE: usize = 6 * 1024 * 1024 * 1024;
// Default to 1 GiB for the metadata cache. Column metadata can be like 40MB,
// so this should be enough for a few hundred columns. Other metadata is much
// smaller.
pub use lance_table_runtime::session::DEFAULT_METADATA_CACHE_SIZE;

/// Lance Dataset
///
/// The manifest and refs are available directly on a checked-out handle.
///
/// ```
/// # use lance::Dataset;
/// # use std::sync::Arc;
/// # fn inspect(mut dataset: Dataset) {
/// let Dataset { manifest, refs, .. } = &dataset;
/// let _version = manifest.version;
/// let _tags = refs.tags();
/// Arc::make_mut(&mut dataset.manifest).tag = Some("local".into());
/// # }
/// ```
#[derive(Clone)]
pub struct Dataset {
    pub(crate) table_state: TableState,
    pub manifest: Arc<Manifest>,
    pub refs: Refs,
    pub(crate) session: Arc<Session>,
    pub(crate) index_cache: Arc<DSIndexCache>,
}

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("uri", &self.table_state.uri)
            .field("base", &self.table_state.base)
            .field("version", &self.manifest.version)
            .field("cache_num_items", &self.session.approx_num_items())
            .field(
                "base_store_params",
                &self.table_state.base_store_params.is_some(),
            )
            .finish()
    }
}

/// Customize read behavior of a dataset.
#[derive(Clone, Debug)]
pub struct ReadParams {
    /// Size of the index cache in bytes. This cache stores index data in memory
    /// for faster lookups. The default is 6 GiB.
    pub index_cache_size_bytes: usize,

    /// Size of the metadata cache in bytes. This cache stores metadata in memory
    /// for faster open table and scans. The default is 1 GiB.
    pub metadata_cache_size_bytes: usize,

    /// If present, dataset will use this shared [`Session`] instead creating a new one.
    ///
    /// This is useful for sharing the same session across multiple datasets.
    pub session: Option<Arc<Session>>,

    pub store_options: Option<ObjectStoreParams>,

    /// If present, dataset will use this to resolve the latest version
    ///
    /// Lance needs to be able to make atomic updates to the manifest.  This involves
    /// coordination between readers and writers and we can usually rely on the filesystem
    /// to do this coordination for us.
    ///
    /// Some file systems (e.g. S3) do not support atomic operations.  In this case, for
    /// safety, we recommend an external commit mechanism (such as dynamodb) and, on the
    /// read path, we need to reach out to that external mechanism to figure out the latest
    /// version of the dataset.
    ///
    /// If this is not set then a default behavior is chosen that is appropriate for the
    /// filesystem.
    ///
    /// If a custom object store is provided (via store_params.object_store) then this
    /// must also be provided.
    pub commit_handler: Option<Arc<dyn CommitHandler>>,

    /// File reader options to use when reading data files.
    ///
    /// This allows control over features like caching repetition indices and validation.
    /// Options set here act as dataset-level defaults and can be overridden on a
    /// per-scan basis via [`Scanner::batch_size_bytes`](crate::dataset::scanner::Scanner::batch_size_bytes) or
    /// [`Scanner::with_file_reader_options`](crate::dataset::scanner::Scanner::with_file_reader_options).
    pub file_reader_options: Option<FileReaderOptions>,
}

impl ReadParams {
    /// Set the cache size for indices. Set to zero, to disable the cache.
    #[deprecated(
        since = "0.30.0",
        note = "Use `index_cache_size_bytes` instead, which accepts a size in bytes."
    )]
    pub fn index_cache_size(&mut self, cache_size: usize) -> &mut Self {
        let assumed_entry_size = 20 * 1024 * 1024; // 20 MiB per entry
        self.index_cache_size_bytes = cache_size * assumed_entry_size;
        self
    }

    pub fn index_cache_size_bytes(&mut self, cache_size: usize) -> &mut Self {
        self.index_cache_size_bytes = cache_size;
        self
    }

    /// Set the cache size for the file metadata. Set to zero to disable this cache.
    #[deprecated(
        since = "0.30.0",
        note = "Use `metadata_cache_size_bytes` instead, which accepts a size in bytes."
    )]
    pub fn metadata_cache_size(&mut self, cache_size: usize) -> &mut Self {
        let assumed_entry_size = 10 * 1024 * 1024; // 10 MiB per entry
        self.metadata_cache_size_bytes = cache_size * assumed_entry_size;
        self
    }

    /// Set the cache size for the file metadata in bytes.
    pub fn metadata_cache_size_bytes(&mut self, cache_size: usize) -> &mut Self {
        self.metadata_cache_size_bytes = cache_size;
        self
    }

    /// Set a shared session for the datasets.
    pub fn session(&mut self, session: Arc<Session>) -> &mut Self {
        self.session = Some(session);
        self
    }

    /// Use the explicit locking to resolve the latest version
    pub fn set_commit_lock<T: CommitLock + Send + Sync + 'static>(&mut self, lock: Arc<T>) {
        self.commit_handler = Some(Arc::new(lock));
    }

    /// Set the file reader options.
    pub fn file_reader_options(&mut self, options: FileReaderOptions) -> &mut Self {
        self.file_reader_options = Some(options);
        self
    }
}

impl Default for ReadParams {
    fn default() -> Self {
        Self {
            index_cache_size_bytes: DEFAULT_INDEX_CACHE_SIZE,
            metadata_cache_size_bytes: DEFAULT_METADATA_CACHE_SIZE,
            session: None,
            store_options: None,
            commit_handler: None,
            file_reader_options: None,
        }
    }
}

/// Runtime-only configuration applied to an already-open [`Dataset`] handle.
///
/// Every field is optional so a caller can change one part of the handle without
/// rebuilding the manifest, object stores, or inherited base bindings.
#[derive(Default)]
pub struct DatasetRuntimeConfig {
    /// New index cache capacity. Ignored when `session` is also set.
    pub index_cache_size_bytes: Option<usize>,
    /// New metadata cache capacity. Ignored when `session` is also set.
    pub metadata_cache_size_bytes: Option<usize>,
    /// New primary object-store block-size hint.
    pub block_size: Option<usize>,
    /// Replacement session. Wins over the cache sizes above.
    pub session: Option<Arc<Session>>,
    /// Replacement commit handler.
    pub commit_handler: Option<Arc<dyn CommitHandler>>,
    /// Replacement storage-options accessor for the primary store. The primary
    /// store is rebuilt through the session registry so subsequent primary I/O
    /// consults this accessor.
    pub storage_options_accessor: Option<Arc<StorageOptionsAccessor>>,
    /// Explicit per-base bindings to merge. Caller entries win over inherited ones.
    pub base_store_params: HashMap<String, ObjectStoreParams>,
}

#[derive(Debug, Clone)]
pub enum ProjectionRequest {
    Schema(Arc<Schema>),
    Sql(Vec<(String, String)>),
}

impl ProjectionRequest {
    /// Project `columns`, by name or nested path, out of `dataset_schema`.
    ///
    /// # Panics
    ///
    /// If a column, or a nested path below it, is not in the schema. Use
    /// [`Self::try_from_columns`] for column names that come from a user.
    pub fn from_columns(
        columns: impl IntoIterator<Item = impl AsRef<str>>,
        dataset_schema: &Schema,
    ) -> Self {
        Self::try_from_columns(columns, dataset_schema).unwrap()
    }

    /// Project `columns`, by name or nested path, out of `dataset_schema`, or
    /// return [`Error::FieldNotFound`] for a column the schema does not have.
    ///
    /// ```
    /// # use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    /// # use lance::dataset::ProjectionRequest;
    /// # use lance_core::datatypes::Schema;
    /// let schema = Schema::try_from(&ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]))?;
    /// assert!(ProjectionRequest::try_from_columns(["id"], &schema).is_ok());
    /// assert!(ProjectionRequest::try_from_columns(["missing"], &schema).is_err());
    /// # Ok::<(), lance::Error>(())
    /// ```
    pub fn try_from_columns(
        columns: impl IntoIterator<Item = impl AsRef<str>>,
        dataset_schema: &Schema,
    ) -> Result<Self> {
        let columns = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();
        let schema = dataset_schema.project_preserve_system_columns(&columns)?;
        Ok(Self::Schema(Arc::new(schema)))
    }

    pub fn from_schema(schema: Schema) -> Self {
        Self::Schema(Arc::new(schema))
    }

    /// Provide a list of projection with SQL transform.
    ///
    /// # Parameters
    /// - `columns`: A list of tuples where the first element is resulted column name and the second
    ///   element is the SQL expression.
    pub fn from_sql(
        columns: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    ) -> Self {
        Self::Sql(
            columns
                .into_iter()
                .map(|(a, b)| (a.into(), b.into()))
                .collect(),
        )
    }

    pub fn into_projection_plan(self, dataset: Arc<Dataset>) -> Result<ProjectionPlan> {
        match self {
            Self::Schema(schema) => {
                // The schema might contain system columns (_rowid, _rowaddr) which are not
                // in the dataset schema. We handle these specially in ProjectionPlan::from_schema.
                let system_columns_present = schema
                    .fields
                    .iter()
                    .any(|f| lance_core::is_system_column(&f.name));

                if system_columns_present {
                    // If system columns are present, we can't use project_by_schema directly
                    // Just pass the schema to ProjectionPlan::from_schema which handles it
                    ProjectionPlan::from_schema(dataset, schema.as_ref())
                } else {
                    // No system columns, use normal path with validation
                    let projection = dataset.schema().project_by_schema(
                        schema.as_ref(),
                        OnMissing::Error,
                        OnTypeMismatch::Error,
                    )?;
                    ProjectionPlan::from_schema(dataset, &projection)
                }
            }
            Self::Sql(columns) => ProjectionPlan::from_expressions(dataset, &columns),
        }
    }
}

impl From<Arc<Schema>> for ProjectionRequest {
    fn from(schema: Arc<Schema>) -> Self {
        Self::Schema(schema)
    }
}

impl From<Schema> for ProjectionRequest {
    fn from(schema: Schema) -> Self {
        Self::from(Arc::new(schema))
    }
}

impl Dataset {
    pub(crate) fn from_table(table: Table, session: Arc<Session>) -> Self {
        let index_cache = Arc::new(session.index_cache.for_dataset(table.uri()));
        Self::from_table_with_index_cache(table, session, index_cache)
    }

    pub(crate) fn from_committed_table(table: Table, session: Arc<Session>) -> Self {
        let index_cache = Arc::new(DSIndexCache(table.index_cache().clone()));
        Self::from_table_with_index_cache(table, session, index_cache)
    }

    pub(crate) fn from_table_with_index_cache(
        table: Table,
        session: Arc<Session>,
        index_cache: Arc<DSIndexCache>,
    ) -> Self {
        let (table_state, manifest, refs) = table.into_parts();
        Self {
            table_state,
            manifest,
            refs,
            session,
            index_cache,
        }
    }

    pub(crate) fn table_view(&self) -> TableView<'_> {
        TableView::new(&self.table_state, &self.manifest, &self.refs)
    }

    /// Open an existing dataset.
    ///
    /// See also [DatasetBuilder].
    #[instrument]
    pub async fn open(uri: &str) -> Result<Self> {
        DatasetBuilder::from_uri(uri).load().await
    }

    /// Check out a dataset version with a ref.
    ///
    /// In reference contexts, `"main"` is an alias for the default branch and
    /// is equivalent to `None`.
    pub async fn checkout_version(&self, version: impl Into<refs::Ref>) -> Result<Self> {
        match self.table_view().checkout_version(version).await? {
            Some(table) => Ok(Self::from_table(table, self.session.clone())),
            None => Ok(self.clone()),
        }
    }

    pub fn tags(&self) -> Tags<'_> {
        self.refs.tags()
    }

    /// A handle for cheap, index-derived statistics about this dataset (e.g. a
    /// column's global value range) that never scan data.
    pub fn statistics(&self) -> DatasetStatistics<'_> {
        DatasetStatistics::new(self)
    }

    pub fn branches(&self) -> Branches<'_> {
        self.refs.branches()
    }

    /// Check out the latest version of the dataset
    pub async fn checkout_latest(&mut self) -> Result<()> {
        let (manifest, manifest_location) = self.latest_manifest().await?;
        self.set_manifest(manifest, manifest_location);
        Ok(())
    }

    /// Replace the manifest, refreshing derived state. Base stores are kept
    /// when `base_paths` is unchanged.
    fn set_manifest(&mut self, manifest: Arc<Manifest>, manifest_location: ManifestLocation) {
        self.table_state
            .set_manifest(&mut self.manifest, manifest, manifest_location);
    }

    /// Check out the latest version of the branch.
    ///
    /// Use `"main"` to check out the latest version of the default branch.
    pub async fn checkout_branch(&self, branch: &str) -> Result<Self> {
        self.checkout_version((branch, None)).await
    }

    /// Create a branch from an existing version with a shallow clone.
    ///
    /// This is a two-phase operation:
    /// - Atomically reserve the name by creating the branch metadata (`BranchContents`).
    /// - Create the branch dataset by shallow cloning.
    ///
    /// These two phases are not atomic. We consider `BranchContents` as the source of truth
    /// for the branch. Callers that must check their own state between the phases can run
    /// them separately with [`Self::reserve_branch`] and [`Self::complete_branch`].
    ///
    /// A concurrent creator that cannot reserve the name returns a conflict without
    /// writing to the branch dataset.
    ///
    /// Destinations with leftover version data are rejected without reserving the name.
    /// Use [`Self::force_delete_branch`] to explicitly clean up the leftover data before
    /// retrying. If cloning fails or is interrupted, the name remains reserved: the
    /// clone may have succeeded, and releasing the reservation cannot atomically
    /// verify that it still belongs to this attempt. Use [`Self::force_delete_branch`]
    /// to explicitly remove a failed creation before retrying.
    ///
    /// `"main"` is reserved for the default branch and cannot be used as a new branch name.
    /// A segment after the first cannot be a reserved dataset directory.
    /// A literal `.` path segment is rejected.
    ///
    /// New branch names are limited to a portable budget of 211 bytes, measured
    /// after the name is path-part encoded twice: once for the branch metadata
    /// filename and once for the local atomic staging filename. The check runs
    /// before any branch metadata or dataset files are written. Longer names
    /// that already exist remain available for lookup, checkout, metadata
    /// updates, and deletion.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &mut Dataset) -> Result<()> {
    /// let experiment = dataset.create_branch("experiment", ("main", None), None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn create_branch(
        &mut self,
        branch: &str,
        version: impl Into<refs::Ref>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        let reservation = self.reserve_branch(branch, version).await?;
        // The reservation was just written, so skip complete_branch's ref reread.
        self.clone_reserved_branch(reservation, store_params).await
    }

    /// Reserve a branch name at an exact source version without creating the branch dataset.
    ///
    /// This is the first phase of [`Self::create_branch`]. It applies the same name and
    /// destination checks, resolves `version` to a fixed source branch and version, and
    /// atomically creates the branch metadata (`BranchContents`). A concurrent reservation
    /// of the same name returns a conflict.
    ///
    /// Until [`Self::complete_branch`] clones the source, the reserved branch is listed but
    /// cannot be checked out. To abandon the reservation, delete the branch with
    /// [`Self::delete_branch`].
    ///
    /// A pending reservation does not keep [`Self::cleanup_old_versions`] from removing its
    /// source version: cleanup finds the versions a branch depends on through the branch's
    /// manifests, and a reservation has none yet. If cleanup can run before the reservation
    /// is completed, tag the source version until then, or completion can fail because the
    /// source manifest was removed.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let reservation = dataset.reserve_branch("experiment", ("main", None)).await?;
    /// // Check any state that must still hold before the branch is published.
    /// let experiment = dataset.complete_branch(reservation, None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn reserve_branch(
        &self,
        branch: &str,
        version: impl Into<refs::Ref>,
    ) -> Result<BranchReservation> {
        self.table_view().reserve_branch(branch, version).await
    }

    /// Clone the source version of a reservation from [`Self::reserve_branch`] into its
    /// branch.
    ///
    /// This is the second phase of [`Self::create_branch`]. Because a reservation can be held
    /// across other work, it first rereads the branch metadata and refuses to clone if the
    /// reservation was deleted ([`Error::RefNotFound`]) or the name was deleted and reserved
    /// again ([`Error::RefConflict`]). That check is not atomic with the clone commit.
    ///
    /// The clone commits through this dataset's object store and commit handler, so any
    /// handle of the same dataset can complete the reservation. `store_params` defaults to
    /// this dataset's store parameters. As with [`Self::create_branch`], a failed or
    /// interrupted clone keeps the reservation because the clone may have succeeded. Use
    /// [`Self::force_delete_branch`] to remove the failed creation before retrying.
    pub async fn complete_branch(
        &self,
        reservation: BranchReservation,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        let branch = reservation.branch();
        let contents = self
            .branches()
            .get(branch)
            .await
            .map_err(|error| match error {
                Error::RefNotFound { .. } => Error::RefNotFound {
                    message: format!("branch {branch} was deleted after it was reserved"),
                },
                error => error,
            })?;
        if &contents.identifier != reservation.identifier() {
            return Err(Error::RefConflict {
                message: format!("branch {branch} was deleted and recreated after it was reserved"),
            });
        }
        self.clone_reserved_branch(reservation, store_params).await
    }

    async fn clone_reserved_branch(
        &self,
        reservation: BranchReservation,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        let branch = reservation.branch();
        let source_branch = reservation.source_branch();
        let source_version = reservation.source_version();
        let branch_location = self.branch_location().find_branch(Some(branch))?;
        let source_location = self.branch_location().find_branch(source_branch)?;
        let clone_op = Operation::Clone {
            is_shallow: true,
            ref_name: source_branch.map(ToOwned::to_owned),
            ref_version: source_version,
            ref_path: source_location.uri,
            branch_name: Some(branch.to_owned()),
        };
        let transaction = Transaction::new(source_version, clone_op, None);

        CommitBuilder::new(WriteDestination::Uri(branch_location.uri.as_str()))
            // Fall back to the dataset's own store params
            .with_store_params(
                store_params.unwrap_or(
                    self.table_state
                        .store_params
                        .as_deref()
                        .cloned()
                        .unwrap_or_default(),
                ),
            )
            .with_object_store(Arc::new(self.table_state.object_store.as_ref().clone()))
            .with_commit_handler(self.table_state.commit_handler.clone())
            .with_exact_storage_format(self.manifest.data_storage_format.lance_file_format())
            .execute(transaction)
            .await
    }

    /// Deletes a branch and removes its directory when safe.
    ///
    /// The ref is removed before directory cleanup. If cleanup fails, this
    /// returns the cleanup error even though the ref may already be gone. Use
    /// [`Self::force_delete_branch`] to retry failed cleanup.
    ///
    /// Legacy branches like `outer/data` may share a directory with another
    /// branch. Their files are intentionally retained, but checkout requires a
    /// live branch ref.
    pub async fn delete_branch(&mut self, branch: &str) -> Result<()> {
        self.branches().delete(branch, false).await
    }

    /// Delete the branch even if the BranchContents is not found.
    /// This could be useful when we have zombie branches and want to clean them up immediately.
    pub async fn force_delete_branch(&mut self, branch: &str) -> Result<()> {
        self.branches().delete(branch, true).await
    }

    pub async fn list_branches(&self) -> Result<HashMap<String, BranchContents>> {
        self.branches().list().await
    }

    #[cfg(test)]
    pub(crate) async fn load_manifest(
        object_store: &ObjectStore,
        manifest_location: &ManifestLocation,
        uri: &str,
        session: &Session,
    ) -> Result<Manifest> {
        Table::load_manifest(object_store, manifest_location, uri, &session.table_session).await
    }

    /// Write to or Create a [Dataset] with a stream of [RecordBatch]s.
    ///
    /// `dest` can be a `&str`, `object_store::path::Path` or `Arc<Dataset>`.
    ///
    /// Returns the newly created [`Dataset`].
    /// Or Returns [Error] if the dataset already exists.
    ///
    pub async fn write(
        batches: impl RecordBatchReader + Send + 'static,
        dest: impl Into<WriteDestination<'_>>,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        let mut builder = InsertBuilder::new(dest);
        if let Some(params) = &params {
            builder = builder.with_params(params);
        }
        Box::pin(builder.execute_stream(Box::new(batches) as Box<dyn RecordBatchReader + Send>))
            .await
    }

    /// Write into a namespace client-managed table with automatic credential vending.
    ///
    /// For CREATE mode, calls declare_table() to initialize the table.
    /// For other modes, calls describe_table() and opens dataset with namespace client credentials.
    ///
    /// # Arguments
    ///
    /// * `batches` - The record batches to write
    /// * `namespace_client` - The namespace client to use for table management
    /// * `table_id` - The table identifier
    /// * `params` - Write parameters
    pub async fn write_into_namespace(
        batches: impl RecordBatchReader + Send + 'static,
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        Self::write_into_namespace_impl(batches, namespace_client, table_id, None, params).await
    }

    /// Write into a branch of a namespace client-managed table.
    ///
    /// Behaves like [`write_into_namespace`](Self::write_into_namespace), but APPEND and
    /// OVERWRITE open and commit against `branch` instead of main. CREATE is rejected,
    /// since a branch forks from an existing version.
    pub async fn write_into_namespace_on_branch(
        batches: impl RecordBatchReader + Send + 'static,
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
        branch: &str,
        params: Option<WriteParams>,
    ) -> Result<Self> {
        Self::write_into_namespace_impl(
            batches,
            namespace_client,
            table_id,
            Some(branch.to_string()),
            params,
        )
        .await
    }

    async fn write_into_namespace_impl(
        batches: impl RecordBatchReader + Send + 'static,
        namespace_client: Arc<dyn LanceNamespace>,
        table_id: Vec<String>,
        branch: Option<String>,
        mut params: Option<WriteParams>,
    ) -> Result<Self> {
        let mut write_params = params.take().unwrap_or_default();

        match write_params.mode {
            WriteMode::Create => {
                if branch.is_some() {
                    return Err(Error::not_supported_source(
                        "cannot create a table on a branch; create on main first, then branch it"
                            .into(),
                    ));
                }
                let declare_request = DeclareTableRequest {
                    id: Some(table_id.clone()),
                    ..Default::default()
                };
                let response = namespace_client
                    .declare_table(declare_request)
                    .await
                    .map_err(|e| Error::namespace_source(Box::new(e)))?;

                let uri = response.location.ok_or_else(|| {
                    Error::namespace_source(Box::new(std::io::Error::other(
                        "Table location not found in declare_table response",
                    )))
                })?;

                // Set up commit handler when managed_versioning is enabled
                if response.managed_versioning == Some(true) {
                    // The store derives the branch a request targets from the
                    // base path it is handed, resolved against the table root.
                    let external_store = LanceNamespaceExternalManifestStore::for_table_uri(
                        namespace_client.clone(),
                        table_id.clone(),
                        &uri,
                    )?;
                    let commit_handler: Arc<dyn CommitHandler> =
                        Arc::new(ExternalManifestCommitHandler {
                            external_manifest_store: Arc::new(external_store),
                        });
                    write_params.commit_handler = Some(commit_handler);
                }

                // Set initial credentials and provider from namespace_client
                if let Some(namespace_storage_options) = response.storage_options {
                    let provider: Arc<dyn StorageOptionsProvider> = Arc::new(
                        LanceNamespaceStorageOptionsProvider::new(namespace_client, table_id),
                    );

                    // Merge namespace client storage options with any existing options
                    let mut merged_options = write_params
                        .store_params
                        .as_ref()
                        .and_then(|p| p.storage_options().cloned())
                        .unwrap_or_default();
                    merged_options.extend(namespace_storage_options);

                    let accessor = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
                        merged_options,
                        provider,
                    ));

                    let existing_params = write_params.store_params.take().unwrap_or_default();
                    write_params.store_params = Some(ObjectStoreParams {
                        storage_options_accessor: Some(accessor),
                        ..existing_params
                    });
                }

                Self::write(batches, uri.as_str(), Some(write_params)).await
            }
            WriteMode::Append | WriteMode::Overwrite => {
                let request = DescribeTableRequest {
                    id: Some(table_id.clone()),
                    ..Default::default()
                };
                let response = namespace_client
                    .describe_table(request)
                    .await
                    .map_err(|e| Error::namespace_source(Box::new(e)))?;

                let uri = response.location.ok_or_else(|| {
                    Error::namespace_source(Box::new(std::io::Error::other(
                        "Table location not found in describe_table response",
                    )))
                })?;

                // Set up commit handler when managed_versioning is enabled.
                // It must ride on the dataset opened below: InsertBuilder
                // commits through the destination dataset's handler and does
                // not consult write params for Dataset destinations.
                let commit_handler: Option<Arc<dyn CommitHandler>> =
                    if response.managed_versioning == Some(true) {
                        // The store derives the branch a request targets from the
                        // base path it is handed, resolved against the table root.
                        let external_store = LanceNamespaceExternalManifestStore::for_table_uri(
                            namespace_client.clone(),
                            table_id.clone(),
                            uri.as_str(),
                        )?;
                        Some(Arc::new(ExternalManifestCommitHandler {
                            external_manifest_store: Arc::new(external_store),
                        }))
                    } else {
                        None
                    };

                // Set initial credentials and provider from namespace_client
                if let Some(namespace_storage_options) = response.storage_options {
                    let provider: Arc<dyn StorageOptionsProvider> =
                        Arc::new(LanceNamespaceStorageOptionsProvider::new(
                            namespace_client.clone(),
                            table_id.clone(),
                        ));

                    // Merge namespace client storage options with any existing options
                    let mut merged_options = write_params
                        .store_params
                        .as_ref()
                        .and_then(|p| p.storage_options().cloned())
                        .unwrap_or_default();
                    merged_options.extend(namespace_storage_options);

                    let accessor = Arc::new(StorageOptionsAccessor::with_initial_and_provider(
                        merged_options,
                        provider,
                    ));

                    let existing_params = write_params.store_params.take().unwrap_or_default();
                    write_params.store_params = Some(ObjectStoreParams {
                        storage_options_accessor: Some(accessor),
                        ..existing_params
                    });
                }

                // For APPEND/OVERWRITE modes, we must open the existing dataset first
                // and pass it to InsertBuilder. If we pass just the URI, InsertBuilder
                // assumes no dataset exists and converts the mode to CREATE.
                let mut builder = DatasetBuilder::from_uri(uri.as_str());
                if let Some(ref store_params) = write_params.store_params
                    && let Some(accessor) = &store_params.storage_options_accessor
                {
                    builder = builder.with_storage_options_accessor(accessor.clone());
                }
                if let Some(commit_handler) = commit_handler {
                    builder = builder.with_commit_handler(commit_handler);
                }
                if let Some(branch) = &branch {
                    builder = builder.with_branch(branch, None);
                }
                let dataset = Arc::new(builder.load().await?);

                Self::write(batches, dataset, Some(write_params)).await
            }
        }
    }

    /// Append to existing [Dataset] with a stream of [RecordBatch]s
    ///
    /// Returns void result or Returns [Error]
    pub async fn append(
        &mut self,
        batches: impl RecordBatchReader + Send + 'static,
        params: Option<WriteParams>,
    ) -> Result<()> {
        let write_params = WriteParams {
            mode: WriteMode::Append,
            ..params.unwrap_or_default()
        };

        let new_dataset = InsertBuilder::new(WriteDestination::Dataset(Arc::new(self.clone())))
            .with_params(&write_params)
            .execute_stream(Box::new(batches) as Box<dyn RecordBatchReader + Send>)
            .await?;

        *self = new_dataset;

        Ok(())
    }

    /// Get the fully qualified URI of this dataset.
    pub fn uri(&self) -> &str {
        &self.table_state.uri
    }

    /// The commit handler this dataset commits through. Lets a caller wrap it
    /// for one commit via [`CommitBuilder::with_commit_handler`], for example
    /// to fence that commit to an exact version, without reopening the
    /// dataset or knowing how the handler was chosen.
    pub fn commit_handler(&self) -> Arc<dyn CommitHandler> {
        self.table_state.commit_handler.clone()
    }

    pub fn branch_location(&self) -> BranchLocation {
        self.table_view().branch_location()
    }

    pub async fn branch_identifier(&self) -> Result<BranchIdentifier> {
        self.table_view().branch_identifier().await
    }

    /// Get the full manifest of the dataset version.
    pub fn manifest(&self) -> &Manifest {
        self.table_view().manifest()
    }

    pub fn manifest_location(&self) -> &ManifestLocation {
        self.table_view().manifest_location()
    }

    /// Create a [`delta::DatasetDeltaBuilder`] to explore changes between dataset versions.
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let delta = dataset.delta()
    ///     .compared_against_version(5)
    ///     .build()?;
    /// let inserted = delta.get_inserted_rows().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn delta(&self) -> delta::DatasetDeltaBuilder {
        delta::DatasetDeltaBuilder::new(self.clone())
    }

    pub async fn latest_manifest(&self) -> Result<(Arc<Manifest>, ManifestLocation)> {
        self.table_view().latest_manifest().await
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    pub async fn read_transaction(&self) -> Result<Option<Transaction>> {
        self.table_view().read_transaction().await
    }

    /// Read the transaction (if any) and commit timestamp of a version of the
    /// dataset. `version` is a version number on this dataset's current branch.
    ///
    /// When `version` is the version this handle is checked out at, the
    /// in-memory manifest is used and the transaction is read directly from
    /// storage, so no manifest download is needed. Otherwise the version's
    /// manifest is read transiently: no historical `Dataset` is constructed,
    /// no `IndexSection` is decoded, and no session cache is written, so
    /// scanning many historical versions does not fill the shared caches. A
    /// manifest that the session cache already holds (for example after
    /// `checkout_version`) is reused instead of being downloaded again.
    ///
    /// Neither path consults or populates the session's transaction cache
    /// (unlike [`Self::read_transaction`]): that cache is keyed only by
    /// version number, which is not safe to reuse across a dataset dropped
    /// and recreated at the same URI within the same session (see #8916).
    ///
    /// Returns an error if the version does not exist (for example, if it has
    /// been cleaned up).
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let record = dataset.read_version_transaction(5).await?;
    /// let committed_at = record.timestamp;
    /// let operation = record.transaction.as_ref().map(|t| t.operation.name());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_version_transaction(&self, version: u64) -> Result<VersionTransaction> {
        self.table_view().read_version_transaction(version).await
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    ///
    /// Does not populate the session caches; see
    /// [`Self::read_version_transaction`].
    ///
    /// # Example
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let transaction = dataset.read_transaction_by_version(5).await?;
    /// let operation = transaction.as_ref().map(|t| t.operation.name());
    /// # Ok(())
    /// # }
    /// ```
    pub async fn read_transaction_by_version(&self, version: u64) -> Result<Option<Transaction>> {
        self.table_view().read_transaction_by_version(version).await
    }

    /// List transactions for the dataset, up to a maximum number.
    ///
    /// This method iterates through dataset versions, starting from the current version,
    /// and collects the transaction for each version. It stops when either `recent_transactions`
    /// is reached or there are no more versions.
    ///
    /// # Arguments
    ///
    /// * `recent_transactions` - Maximum number of transactions to return
    ///
    /// # Returns
    ///
    /// A vector of optional transactions. Each element corresponds to a version,
    /// and may be None if no transaction file exists for that version.
    pub async fn get_transactions(
        &self,
        recent_transactions: usize,
    ) -> Result<Vec<Option<Transaction>>> {
        self.table_view()
            .get_transactions(recent_transactions)
            .await
    }

    /// Restore the currently checked out version of the dataset as the latest version.
    pub async fn restore(&mut self) -> Result<()> {
        let (latest_manifest, _) = self.latest_manifest().await?;
        let latest_version = latest_manifest.version;

        let transaction = Transaction::new(
            latest_version,
            Operation::Restore {
                version: self.manifest.version,
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    /// Restore the currently checked out version if `expected_latest_version`
    /// is still the dataset's latest version.
    ///
    /// Unlike [`Self::restore`], this method never rebases across commits that
    /// landed after the caller inspected the table. If the latest version
    /// changed, or another writer publishes while this restore is being built,
    /// the commit fails with a retryable conflict and the caller must reload
    /// and validate the new snapshot before trying again.
    ///
    /// The restore is published through this dataset's installed commit
    /// handler, including namespace-managed version registration.
    ///
    /// # Arguments
    ///
    /// * `expected_latest_version` - The latest version the caller observed
    ///   and validated before deciding to restore.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let expected_latest_version = dataset.latest_version_id().await?;
    /// let mut restore_target = dataset.checkout_version(3).await?;
    /// restore_target
    ///     .restore_from(expected_latest_version)
    ///     .await?;
    /// # Ok(())
    /// }
    /// ```
    pub async fn restore_from(&mut self, expected_latest_version: u64) -> Result<()> {
        let transaction = Transaction::new(
            expected_latest_version,
            Operation::Restore {
                version: self.manifest.version,
            },
            None,
        );
        self.apply_commit_with_expected_latest_version(
            transaction,
            &Default::default(),
            &Default::default(),
            Some(expected_latest_version),
        )
        .await
    }

    /// Removes old versions of the dataset from disk
    ///
    /// This function will remove all versions of the dataset that are older than the provided
    /// timestamp.  This function will not remove the current version of the dataset.
    ///
    /// Once a version is removed it can no longer be checked out or restored.  Any data unique
    /// to that version will be lost.
    ///
    /// Returns an invalid-input error when this dataset is checked out at a detached version.
    ///
    /// # Arguments
    ///
    /// * `older_than` - Versions older than this will be deleted.
    /// * `delete_unverified` - If false (the default), files that are not referenced by any
    ///                        manifest will only be deleted if they are more than 7 days old.
    ///                        Otherwise these files will be kept since they cannot be
    ///                        distinguished from an in-progress transaction. Set to true to
    ///                        delete these files if you are sure there are no other
    ///                        in-progress dataset operations.
    ///
    /// # Returns
    ///
    /// * `RemovalStats` - Statistics about the removal operation
    #[instrument(level = "debug", skip(self))]
    pub fn cleanup_old_versions(
        &self,
        older_than: Duration,
        delete_unverified: Option<bool>,
        error_if_tagged_old_versions: Option<bool>,
    ) -> BoxFuture<'_, Result<RemovalStats>> {
        let mut builder = CleanupPolicyBuilder::default();
        builder = builder.before_timestamp(utc_now() - older_than);
        if let Some(v) = delete_unverified {
            builder = builder.delete_unverified(v);
        }
        if let Some(v) = error_if_tagged_old_versions {
            builder = builder.error_if_tagged_old_versions(v);
        }

        self.cleanup_with_policy(builder.build())
    }

    /// Removes old versions of the dataset from storage
    ///
    /// This function will remove all versions of the dataset that satisfies the given policy.
    /// This function will not remove the current version of the dataset.
    ///
    /// Once a version is removed it can no longer be checked out or restored.  Any data unique
    /// to that version will be lost.
    ///
    /// Returns an invalid-input error when this dataset is checked out at a detached version.
    ///
    /// # Arguments
    ///
    /// * `policy` - `CleanupPolicy` determines the behaviour of cleanup.
    ///
    /// # Returns
    ///
    /// * `RemovalStats` - Statistics about the removal operation
    #[instrument(level = "debug", skip(self))]
    pub fn cleanup_with_policy(
        &self,
        policy: CleanupPolicy,
    ) -> BoxFuture<'_, Result<RemovalStats>> {
        async move { self.cleanup(policy).execute().await }.boxed()
    }

    /// Creates a cleanup operation for this dataset.
    ///
    /// The returned operation can be explained without deleting files, or
    /// executed to re-evaluate the current dataset state and remove files.
    /// Both return an invalid-input error when this dataset is checked out at a
    /// detached version.
    pub fn cleanup(&self, policy: CleanupPolicy) -> CleanupOperation<'_> {
        CleanupOperation::new(self, policy)
    }

    #[allow(clippy::too_many_arguments)]
    async fn do_commit(
        base_uri: WriteDestination<'_>,
        operation: Operation,
        read_version: Option<u64>,
        store_params: Option<ObjectStoreParams>,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        session: Arc<Session>,
        enable_v2_manifest_paths: bool,
        detached: bool,
    ) -> Result<Self> {
        let read_version = read_version.map_or_else(
            || match operation {
                Operation::Overwrite { .. } | Operation::Restore { .. } => Ok(0),
                _ => Err(Error::invalid_input(
                    "read_version must be specified for this operation",
                )),
            },
            Ok,
        )?;

        let transaction = Transaction::new(read_version, operation, None);

        let mut builder = CommitBuilder::new(base_uri)
            .enable_v2_manifest_paths(enable_v2_manifest_paths)
            .with_session(session)
            .with_detached(detached);

        if let Some(store_params) = store_params {
            builder = builder.with_store_params(store_params);
        }

        if let Some(commit_handler) = commit_handler {
            builder = builder.with_commit_handler(commit_handler);
        }

        builder.execute(transaction).await
    }

    /// Commit changes to the dataset
    ///
    /// This operation is not needed if you are using append/write/delete to manipulate the dataset.
    /// It is used to commit changes to the dataset that are made externally.  For example, a bulk
    /// import tool may import large amounts of new data and write the appropriate lance files
    /// directly instead of using the write function.
    ///
    /// This method can be used to commit this change to the dataset's manifest.  This method will
    /// not verify that the provided fragments exist and correct, that is the caller's responsibility.
    /// Some validation can be performed using the function
    /// [crate::dataset::transaction::validate_operation].
    ///
    /// That responsibility includes the files' format requirements: a caller publishing
    /// externally produced fragments must make sure their contents comply with the
    /// destination's format and with the feature flags the destination carries. The commit
    /// does not discover every capability from arbitrary file contents. For example, data
    /// files that embed version 1 zone write seeds (such as files copied from another table
    /// whose manifest carries `FLAG_ZONE_WRITE_SEEDS`) may only be committed into a
    /// destination that already carries that flag.
    ///
    /// If this commit is a change to an existing dataset then it will often need to be based on an
    /// existing version of the dataset.  For example, if this change is a `delete` operation then
    /// the caller will have read in the existing data (at some version) to determine which fragments
    /// need to be deleted.  The base version that the caller used should be supplied as the `read_version`
    /// parameter.  Some operations (e.g. Overwrite) do not depend on a previous version and `read_version`
    /// can be None.  An error will be returned if the `read_version` is needed for an operation and
    /// it is not specified.
    ///
    /// All operations except Overwrite will fail if the dataset does not already exist.
    ///
    /// # Arguments
    ///
    /// * `base_uri` - The base URI of the dataset
    /// * `operation` - A description of the change to commit
    /// * `read_version` - The version of the dataset that this change is based on
    /// * `store_params` Parameters controlling object store access to the manifest
    /// * `enable_v2_manifest_paths`: If set to true, and this is a new dataset, uses the new v2 manifest
    ///   paths. These allow constant-time lookups for the latest manifest on object storage.
    ///   This parameter has no effect on existing datasets. To migrate an existing
    ///   dataset, use the [`Self::migrate_manifest_paths_v2`] method. WARNING: turning
    ///   this on will make the dataset unreadable for older versions of Lance
    ///   (prior to 0.17.0). Default is False.
    pub async fn commit(
        dest: impl Into<WriteDestination<'_>>,
        operation: Operation,
        read_version: Option<u64>,
        store_params: Option<ObjectStoreParams>,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        session: Arc<Session>,
        enable_v2_manifest_paths: bool,
    ) -> Result<Self> {
        Self::do_commit(
            dest.into(),
            operation,
            read_version,
            store_params,
            commit_handler,
            session,
            enable_v2_manifest_paths,
            /*detached=*/ false,
        )
        .await
    }

    /// Commits changes exactly the same as [`Self::commit`] but the commit will
    /// not be associated with the dataset lineage.
    ///
    /// The commit will not show up in the dataset's history and will never be
    /// the latest version of the dataset.
    ///
    /// This can be used to stage changes or to handle "secondary" datasets whose
    /// lineage is tracked elsewhere.
    pub async fn commit_detached(
        dest: impl Into<WriteDestination<'_>>,
        operation: Operation,
        read_version: Option<u64>,
        store_params: Option<ObjectStoreParams>,
        commit_handler: Option<Arc<dyn CommitHandler>>,
        session: Arc<Session>,
        enable_v2_manifest_paths: bool,
    ) -> Result<Self> {
        Self::do_commit(
            dest.into(),
            operation,
            read_version,
            store_params,
            commit_handler,
            session,
            enable_v2_manifest_paths,
            /*detached=*/ true,
        )
        .await
    }

    pub(crate) async fn apply_commit(
        &mut self,
        transaction: Transaction,
        build_config: &ManifestBuildConfig,
        commit_config: &CommitConfig,
    ) -> Result<()> {
        self.apply_commit_with_options(
            transaction,
            CommitTransactionOptions::new(
                build_config,
                commit_config,
                self.table_state.manifest_location.naming_scheme,
            ),
        )
        .await
    }

    pub(crate) async fn apply_tagged_frag_reuse_trim(
        &mut self,
        transaction: Transaction,
    ) -> Result<()> {
        let build_config = ManifestBuildConfig::default();
        let commit_config = CommitConfig::default();
        self.apply_commit_with_options(
            transaction,
            CommitTransactionOptions::new(
                &build_config,
                &commit_config,
                self.table_state.manifest_location.naming_scheme,
            )
            .with_tagged_frag_reuse_trim(),
        )
        .await
    }

    async fn apply_commit_with_expected_latest_version(
        &mut self,
        transaction: Transaction,
        build_config: &ManifestBuildConfig,
        commit_config: &CommitConfig,
        expected_latest_version: Option<u64>,
    ) -> Result<()> {
        let mut options = CommitTransactionOptions::new(
            build_config,
            commit_config,
            self.table_state.manifest_location.naming_scheme,
        );
        if let Some(version) = expected_latest_version {
            options = options.with_expected_latest_version(version);
        }
        self.apply_commit_with_options(transaction, options).await
    }

    async fn apply_commit_with_options(
        &mut self,
        transaction: Transaction,
        options: CommitTransactionOptions<'_>,
    ) -> Result<()> {
        // Keep the retry and rebase state machine out of every caller's future.
        let (manifest, manifest_location) = Box::pin(commit_transaction(
            self,
            self.table_state.object_store.as_ref(),
            self.table_state.commit_handler.as_ref(),
            &transaction,
            options,
        ))
        .await?;

        self.set_manifest(Arc::new(manifest), manifest_location);

        Ok(())
    }

    /// Create a Scanner to scan the dataset.
    pub fn scan(&self) -> Scanner {
        Scanner::new(Arc::new(self.clone()))
    }

    /// Count the number of rows in the dataset.
    ///
    /// It offers a fast path of counting rows by just computing via metadata.
    #[instrument(skip_all)]
    pub async fn count_rows(&self, filter: Option<String>) -> Result<usize> {
        // TODO: consolidate the count_rows into Scanner plan.
        if let Some(filter) = filter {
            let mut scanner = self.scan();
            scanner.filter(&filter)?;
            Ok(scanner
                .project::<String>(&[])?
                .with_row_id() // TODO: fix scan plan to not require row_id for count_rows.
                .count_rows()
                .await? as usize)
        } else {
            self.count_all_rows().await
        }
    }

    pub(crate) async fn count_all_rows(&self) -> Result<usize> {
        self.table_view().to_table().count_rows().await
    }

    /// Take rows by indices.
    #[instrument(skip_all, fields(num_rows=row_indices.len()))]
    pub async fn take(
        &self,
        row_indices: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<RecordBatch> {
        let span = debug_span!(
            "take_io",
            num_rows = row_indices.len(),
            fragments_selected = Empty,
            logical_ranges = Empty,
            logical_requested_bytes = Empty,
            logical_union_bytes = Empty,
            physical_iops = Empty,
            scheduler_requests = Empty,
            physical_bytes = Empty,
            cloud_reader_attempts = Empty,
            cloud_reader_failed_attempts = Empty,
            cloud_reader_body_bytes = Empty,
            peak_active_reads = Empty,
            max_scheduler_queue_time_ns = Empty,
            max_child_read_time_ns = Empty,
            elapsed_ns = Empty,
            outcome = Empty,
        );
        let summary_span = (!span.is_disabled()).then(|| span.clone());
        take::take(self, row_indices, projection.into(), summary_span)
            .instrument(span)
            .await
    }

    /// Take rows with a maximum number of active fragment-reader data reads in the primary base.
    ///
    /// Current-format primary-base data and overlay files opened through the
    /// fragment reader share one scheduler with this request ceiling, and
    /// fragment-reader fanout is capped at `min(store_parallelism, io_parallelism)`.
    /// This does not increase fragment-reader fanout above the store default, so
    /// actual concurrency may be lower. Legacy v1 files, additional-base
    /// schedulers, and auxiliary metadata or lineage reads (such as deletion
    /// vectors and spilled row-ID or row-version sequences) keep their existing
    /// schedulers. It must be greater than zero.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let rows = dataset
    ///     .take_with_io_parallelism(&[0, 1, 2], dataset.schema().clone(), 64)
    ///     .await?;
    /// # let _ = rows;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip_all, fields(num_rows=row_indices.len(), io_parallelism = io_parallelism))]
    pub async fn take_with_io_parallelism(
        &self,
        row_indices: &[u64],
        projection: impl Into<ProjectionRequest>,
        io_parallelism: u32,
    ) -> Result<RecordBatch> {
        let span = debug_span!(
            "take_io",
            num_rows = row_indices.len(),
            io_parallelism,
            fragments_selected = Empty,
            logical_ranges = Empty,
            logical_requested_bytes = Empty,
            logical_union_bytes = Empty,
            physical_iops = Empty,
            scheduler_requests = Empty,
            physical_bytes = Empty,
            cloud_reader_attempts = Empty,
            cloud_reader_failed_attempts = Empty,
            cloud_reader_body_bytes = Empty,
            peak_active_reads = Empty,
            max_scheduler_queue_time_ns = Empty,
            max_child_read_time_ns = Empty,
            elapsed_ns = Empty,
            outcome = Empty,
        );
        let summary_span = (!span.is_disabled()).then(|| span.clone());
        take::take_with_io_parallelism(
            self,
            row_indices,
            projection.into(),
            Some(io_parallelism),
            summary_span,
        )
        .instrument(span)
        .await
    }

    /// Take Rows by the internal ROW ids.
    ///
    /// In Lance format, each row has a unique `u64` id, which is used to identify the row globally.
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use tokio::runtime::Runtime;
    /// # use arrow_array::{RecordBatch, RecordBatchIterator, Int64Array};
    /// # use arrow_schema::{Schema, Field, DataType};
    /// # use lance::dataset::{WriteParams, Dataset, ProjectionRequest};
    /// #
    /// # let mut rt = Runtime::new().unwrap();
    /// # rt.block_on(async {
    /// # let test_dir = tempfile::tempdir().unwrap();
    /// # let uri = test_dir.path().to_str().unwrap().to_string();
    /// #
    /// # let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    /// # let write_params = WriteParams::default();
    /// # let array = Arc::new(Int64Array::from_iter(0..128));
    /// # let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
    /// # let reader = RecordBatchIterator::new(
    /// #    vec![batch].into_iter().map(Ok), schema
    /// # );
    /// # let dataset = Dataset::write(reader, &uri, Some(write_params)).await.unwrap();
    /// #
    /// let schema = dataset.schema().clone();
    /// let row_ids = vec![0, 4, 7];
    /// let rows = dataset.take_rows(&row_ids, schema).await.unwrap();
    ///
    /// // We can have more fine-grained control over the projection, i.e., SQL projection.
    /// let projection = ProjectionRequest::from_sql([("identity", "id * 2")]);
    /// let rows = dataset.take_rows(&row_ids, projection).await.unwrap();
    /// # });
    /// ```
    pub async fn take_rows(
        &self,
        row_ids: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<RecordBatch> {
        Arc::new(self.clone())
            .take_builder(row_ids, projection)?
            .execute()
            .await
    }

    pub fn take_builder(
        self: &Arc<Self>,
        row_ids: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<TakeBuilder> {
        TakeBuilder::try_new_from_ids(self.clone(), row_ids.to_vec(), projection.into())
    }

    /// Take [BlobFile] by row IDs.
    ///
    /// The returned vector has one element per row ID. Null blob values are
    /// represented as `None`; valid empty blobs return a `BlobFile` with size
    /// zero.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::dataset::Dataset;
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let blobs = dataset.take_blobs(&[42], "images").await?;
    /// match &blobs[0] {
    ///     None => { /* The selected blob is null. */ }
    ///     Some(blob) if blob.size() == 0 => { /* The selected blob is valid but empty. */ }
    ///     Some(blob) => { let _size = blob.size(); }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_blobs(
        self: &Arc<Self>,
        row_ids: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<Option<BlobFile>>> {
        self.table_view()
            .to_table()
            .take_blobs(row_ids, column)
            .await
    }

    /// Take [BlobFile] by row addresses.
    ///
    /// Row addresses are `u64` values encoding `(fragment_id << 32) | row_offset`.
    /// Use this method when you already have row addresses, for example from
    /// a scan with `with_row_address()`. For row IDs (stable identifiers), use
    /// [`Self::take_blobs`]. For row indices (offsets), use
    /// [`Self::take_blobs_by_indices`]. The result has the same null and empty
    /// blob representation as [`Self::take_blobs`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::dataset::Dataset;
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>, row_address: u64) -> Result<()> {
    /// let blobs = dataset
    ///     .take_blobs_by_addresses(&[row_address], "images")
    ///     .await?;
    /// match &blobs[0] {
    ///     None => { /* The selected blob is null. */ }
    ///     Some(blob) if blob.size() == 0 => { /* The selected blob is valid but empty. */ }
    ///     Some(blob) => { let _size = blob.size(); }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_blobs_by_addresses(
        self: &Arc<Self>,
        row_addrs: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<Option<BlobFile>>> {
        self.table_view()
            .to_table()
            .take_blobs_by_addresses(row_addrs, column)
            .await
    }

    /// Scan every row of this dataset with its blob v2 columns as values another
    /// dataset can be written from: the payload bytes of every blob this dataset
    /// stores, and an External blob's absolute URI and byte range, at any depth.
    ///
    /// Each batch holds the payloads of its blobs in memory.
    pub async fn scan_blobs_as_logical(self: &Arc<Self>) -> Result<DatasetRecordBatchStream> {
        let mut scan = self.scan();
        scan.with_row_address();
        let batches = scan.try_into_stream().await?;
        let plan = Arc::new(blob::BlobV2BatchRewritePlan::try_new(
            self.schema(),
            lance_io::stream::RecordBatchStream::schema(&batches).as_ref(),
            false,
            false,
        )?);
        let schema = plan.output_schema().clone();
        let dataset = self.clone();
        let batches = batches
            .and_then(move |batch| {
                let (plan, dataset) = (plan.clone(), dataset.clone());
                async move { plan.transform_batch(&dataset, batch).await }
            })
            .map_err(|err| datafusion::error::DataFusionError::External(Box::new(err)));
        Ok(DatasetRecordBatchStream::new(Box::pin(
            datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, batches),
        )))
    }

    /// Take [BlobFile] by row indices (offsets in the dataset).
    ///
    /// The result has the same null and empty blob representation as
    /// [`Self::take_blobs`].
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use lance::dataset::Dataset;
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let blobs = dataset.take_blobs_by_indices(&[0], "images").await?;
    /// match &blobs[0] {
    ///     None => { /* The selected blob is null. */ }
    ///     Some(blob) if blob.size() == 0 => { /* The selected blob is valid but empty. */ }
    ///     Some(blob) => { let _size = blob.size(); }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn take_blobs_by_indices(
        self: &Arc<Self>,
        row_indices: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<Option<BlobFile>>> {
        self.table_view()
            .to_table()
            .take_blobs_by_indices(row_indices, column)
            .await
    }

    /// Create a planned blob reader for a blob column.
    ///
    /// This API complements [`Self::take_blobs`]. `take_blobs` returns
    /// [`BlobFile`] handles for caller-driven random access, while
    /// `read_blobs` builds a streaming read plan for sequential or batched blob
    /// retrieval. Every selected row produces one result: null blob values have
    /// `ReadBlob::data` set to `None`, while valid empty blobs contain an empty
    /// buffer.
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use futures::TryStreamExt;
    /// # use lance::dataset::Dataset;
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let blobs = dataset
    ///     .read_blobs("images")?
    ///     .with_row_indices(vec![0, 1, 2])
    ///     .execute()
    ///     .await?;
    /// # let _ = blobs;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_blobs(self: &Arc<Self>, column: impl AsRef<str>) -> Result<ReadBlobsBuilder> {
        self.table_view().to_table().read_blobs(column)
    }

    /// Create a planned reader for row-specific blob-local byte ranges.
    ///
    /// Each [`BlobRangeRequest`] contains both its row selector and byte range,
    /// so requests can be repeated or reordered without coordinating parallel
    /// selector and range lists. Every request produces one result. A null blob
    /// has `ReadBlobRange::data` set to `None`; an empty range on a non-null blob
    /// contains an empty buffer.
    ///
    /// ```rust
    /// # use std::sync::Arc;
    /// # use lance::dataset::{BlobRangeRequest, Dataset};
    /// # use lance::Result;
    /// # async fn example(dataset: Arc<Dataset>) -> Result<()> {
    /// let ranges = dataset
    ///     .read_blob_ranges("images")?
    ///     .with_row_indices([
    ///         BlobRangeRequest::new(7, 0, 1024),
    ///         BlobRangeRequest::new(7, 4096, 1024),
    ///     ])
    ///     .execute()
    ///     .await?;
    /// # let _ = ranges;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_blob_ranges(
        self: &Arc<Self>,
        column: impl AsRef<str>,
    ) -> Result<ReadBlobRangesBuilder> {
        self.table_view().to_table().read_blob_ranges(column)
    }

    /// Get a stream of batches based on iterator of ranges of row numbers.
    ///
    /// `batch_readahead` must be greater than zero; otherwise the stream yields an invalid-input error.
    ///
    /// This is an experimental API. It may change at any time.
    pub fn take_scan(
        &self,
        row_ranges: Pin<Box<dyn Stream<Item = Result<Range<u64>>> + Send>>,
        projection: Arc<Schema>,
        batch_readahead: usize,
    ) -> DatasetRecordBatchStream {
        take::take_scan(self, row_ranges, projection, batch_readahead)
    }

    /// Randomly sample up to `n` live rows uniformly without replacement.
    ///
    /// Returns all available live rows if `n` exceeds their count.
    ///
    /// If `fragment_ids` is provided, sampling is limited to rows from those
    /// fragments in the current dataset version.
    ///
    /// The returned rows are in ascending logical row-offset order within the
    /// dataset or selected fragments, allowing efficient sorted reads.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn example(dataset: &Dataset) -> Result<()> {
    /// let batch = dataset.sample(10, dataset.schema(), None).await?;
    /// assert!(batch.num_rows() <= 10);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn sample(
        &self,
        n: usize,
        projection: &Schema,
        fragment_ids: Option<&[u32]>,
    ) -> Result<RecordBatch> {
        match fragment_ids {
            None => {
                let num_rows = self.count_rows(None).await?;
                let offsets = sample_row_offsets(&mut rand::rng(), num_rows, n);
                self.take(&offsets, projection.clone()).await
            }
            Some(fragment_ids) => {
                if fragment_ids.is_empty() {
                    return Err(Error::invalid_input(
                        "Dataset::sample does not accept an empty fragment_ids list".to_string(),
                    ));
                }

                let selected_fragments = self.get_fragments_from_ids(fragment_ids)?;

                let num_rows = stream::iter(selected_fragments.iter().cloned())
                    .map(|fragment| async move { fragment.count_rows(None).await })
                    .buffer_unordered(16)
                    .try_fold(0_u64, |acc, rows| async move {
                        acc.checked_add(rows as u64).ok_or_else(|| {
                            Error::invalid_input(format!(
                                "Dataset::sample selected live row count overflow: {acc} + {rows}"
                            ))
                        })
                    })
                    .await?;

                let num_rows = usize::try_from(num_rows).map_err(|_| {
                    Error::invalid_input(format!(
                        "Dataset::sample selected live row count {num_rows} exceeds usize::MAX ({})",
                        usize::MAX
                    ))
                })?;
                let offsets = sample_row_offsets(&mut rand::rng(), num_rows, n);

                let row_addrs = row_offsets_to_row_addresses(&selected_fragments, &offsets).await?;
                let dataset = Arc::new(self.clone());
                let projection = Arc::new(
                    ProjectionRequest::from(projection.clone())
                        .into_projection_plan(dataset.clone())?,
                );
                TakeBuilder::try_new_from_addresses(dataset, row_addrs, projection)?
                    .execute()
                    .await
            }
        }
    }

    /// Delete rows based on a predicate.
    pub async fn delete(&mut self, predicate: &str) -> Result<write::delete::DeleteResult> {
        info!(target: TRACE_DATASET_EVENTS, event=DATASET_DELETING_EVENT, uri = &self.table_state.uri, predicate=predicate);
        write::delete::delete(self, predicate).await
    }

    /// Truncate the dataset by deleting all rows.
    pub async fn truncate_table(&mut self) -> Result<()> {
        self.delete("true").await.map(|_| ())
    }

    /// Add new base paths to the dataset.
    ///
    /// This method allows you to register additional storage locations (buckets)
    /// that can be used for future data writes. The base paths are added to the
    /// dataset's manifest and can be referenced by name in subsequent write operations.
    ///
    /// # Arguments
    ///
    /// * `new_bases` - A vector of `lance_table::format::BasePath` objects representing the new storage
    ///   locations to add. Each base path should have a unique name and path.
    ///
    /// # Returns
    ///
    /// Returns a new `Dataset` instance with the updated manifest containing the
    /// new base paths.
    pub async fn add_bases(
        self: &Arc<Self>,
        new_bases: Vec<lance_table::format::BasePath>,
        transaction_properties: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let operation = Operation::UpdateBases { new_bases };

        let transaction = TransactionBuilder::new(self.manifest.version, operation)
            .transaction_properties(transaction_properties.map(Arc::new))
            .build();

        let new_dataset = CommitBuilder::new(self.clone())
            .execute(transaction)
            .await?;

        Ok(new_dataset)
    }

    pub async fn count_deleted_rows(&self) -> Result<usize> {
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.count_deletions().await })
            .buffer_unordered(self.table_state.object_store.io_parallelism())
            .try_fold(0, |acc, x| futures::future::ready(Ok(acc + x)))
            .await
    }

    /// Clone this dataset with a different object store binding.
    ///
    /// The returned dataset shares metadata, session state, and caches with the
    /// original dataset, but its primary object store — and the dataset-level
    /// default params used for base paths that have no explicit binding — are
    /// replaced by the supplied ones. Explicit per-base bindings still win: a base
    /// path bound through [`DatasetBuilder::with_base_store_params`], or inherited
    /// from the dataset a shallow clone was taken from, keeps reading through its
    /// own store.
    pub fn with_object_store(
        &self,
        object_store: Arc<ObjectStore>,
        store_params: Option<ObjectStoreParams>,
    ) -> Self {
        let mut cloned = self.clone();
        cloned
            .table_state
            .set_object_store(object_store, store_params);
        cloned
    }

    /// Apply runtime configuration to this dataset handle.
    ///
    /// The returned handle keeps the loaded manifest and every store binding that
    /// `config` does not replace. An explicit session wins over cache sizes, and
    /// cache sizes create a new session on the same store registry so custom
    /// object-store providers keep resolving. When `block_size` is set, the
    /// primary store keeps its existing backend and only changes its read hint.
    /// Replacing `storage_options_accessor` rebuilds the primary store through
    /// the session's registry, because the accessor is consumed when the store is
    /// constructed and must reach the backend that serves primary I/O.
    pub async fn reconfigure(&self, config: DatasetRuntimeConfig) -> Result<Self> {
        let mut ds = self.clone();

        let session = match config.session {
            Some(session) => Some(session),
            None if config.index_cache_size_bytes.is_some()
                || config.metadata_cache_size_bytes.is_some() =>
            {
                Some(Arc::new(Session::new(
                    config
                        .index_cache_size_bytes
                        .unwrap_or(DEFAULT_INDEX_CACHE_SIZE),
                    config
                        .metadata_cache_size_bytes
                        .unwrap_or(DEFAULT_METADATA_CACHE_SIZE),
                    ds.session.store_registry(),
                )))
            }
            None => None,
        };
        if let Some(session) = session {
            ds.index_cache = Arc::new(session.index_cache.for_dataset(&ds.table_state.uri));
            ds.table_state.metadata_cache =
                Arc::new(session.metadata_cache().for_dataset(ds.uri()));
            ds.table_state.set_session(session.table_session.clone());
            ds.session = session;
        }

        let mut store_params = ds.table_state.store_params.as_deref().cloned();
        let accessor_replaced = config.storage_options_accessor.is_some();
        if let Some(accessor) = config.storage_options_accessor {
            let params = store_params.get_or_insert_default();
            params.storage_options_accessor = Some(accessor);
        }
        if let Some(block_size) = config.block_size {
            let params = store_params.get_or_insert_default();
            params.block_size = Some(block_size);
        }

        if accessor_replaced {
            // A store consumes its accessor at construction time, so replacing it
            // in `store_params` alone leaves the opened backend on the previous
            // (possibly expiring) credentials. Rebuild through the handle's
            // session registry so the new accessor, and every other params-bound
            // setting, reach the backend that serves primary metadata, refs, and
            // commits. Explicit base bindings stay untouched; `with_object_store`
            // drops the derived fallback cache so bases are re-resolved from their
            // own bindings, or from the new default params when they have none,
            // instead of keeping the replaced stores.
            let params = store_params.clone().unwrap_or_default();
            let (object_store, _) = ObjectStore::from_uri_and_params(
                ds.session.store_registry(),
                &ds.table_state.uri,
                &params,
            )
            .await?;
            ds = ds.with_object_store(object_store, Some(params));
        } else if let Some(block_size) = config.block_size {
            let params = store_params.unwrap_or_default();
            ds = ds.with_object_store(
                Arc::new(ds.table_state.object_store.with_block_size(block_size)),
                Some(params),
            );
        } else if store_params.is_some() {
            ds.table_state.store_params = store_params.map(Box::new);
        }

        let commit_handler_replaced = config.commit_handler.is_some();
        if let Some(commit_handler) = config.commit_handler {
            ds.table_state.commit_handler = commit_handler;
        }
        if accessor_replaced || config.block_size.is_some() || commit_handler_replaced {
            // Refs binds the primary store and commit handler when it is built.
            ds.refs = Refs::new(
                ds.table_state.object_store.clone(),
                ds.table_state.commit_handler.clone(),
                ds.branch_location(),
            );
        }

        if !config.base_store_params.is_empty() {
            let mut base_store_params = ds
                .table_state
                .base_store_params
                .as_deref()
                .cloned()
                .unwrap_or_default();
            base_store_params.extend(config.base_store_params);
            ds.table_state.base_store_params = Some(Arc::new(base_store_params));
            ds.table_state.clear_base_object_stores();
        }

        Ok(ds)
    }

    /// Clone this dataset with extra object store wrappers applied to all read stores.
    ///
    /// The returned dataset uses the wrappers for the already-open primary object
    /// store and appends the same wrappers to the dataset-level and base-specific
    /// object store params used when additional base stores are opened later.
    pub fn with_object_store_wrappers(
        &self,
        wrappers: impl IntoIterator<Item = Arc<dyn WrappingObjectStore>>,
    ) -> Self {
        let table = self.table_view().with_object_store_wrappers(wrappers);
        Self::from_table_with_index_cache(table, self.session.clone(), self.index_cache.clone())
    }

    pub(crate) fn store_params_for_base(
        &self,
        base_path: Option<&lance_table::format::BasePath>,
    ) -> ObjectStoreParams {
        self.table_state.store_params_for_base(base_path)
    }

    /// Returns the initial storage options used when opening this dataset, if any.
    ///
    /// This returns the static initial options without triggering any refresh.
    /// For the latest refreshed options, use [`Self::latest_storage_options`].
    #[deprecated(since = "0.25.0", note = "Use initial_storage_options() instead")]
    pub fn storage_options(&self) -> Option<&HashMap<String, String>> {
        self.initial_storage_options()
    }

    /// Returns the initial storage options without triggering any refresh.
    ///
    /// For the latest refreshed options, use [`Self::latest_storage_options`].
    pub fn initial_storage_options(&self) -> Option<&HashMap<String, String>> {
        self.table_state.initial_storage_options()
    }

    /// Returns the storage options provider used when opening this dataset, if any.
    pub fn storage_options_provider(
        &self,
    ) -> Option<Arc<dyn lance_io::object_store::StorageOptionsProvider>> {
        self.table_state.storage_options_provider()
    }

    /// Returns the unified storage options accessor for this dataset, if any.
    ///
    /// The accessor handles both static and dynamic storage options with automatic
    /// caching and refresh. Use [`StorageOptionsAccessor::get_storage_options`] to
    /// get the latest options.
    pub fn storage_options_accessor(&self) -> Option<Arc<StorageOptionsAccessor>> {
        self.table_state.storage_options_accessor()
    }

    /// Returns the latest (possibly refreshed) storage options.
    ///
    /// If a dynamic storage options provider is configured, this will return
    /// the cached options if still valid, or fetch fresh options if expired.
    ///
    /// For the initial static options without refresh, use [`Self::storage_options`].
    ///
    /// # Returns
    ///
    /// - `Ok(Some(options))` - Storage options are available (static or refreshed)
    /// - `Ok(None)` - No storage options were configured for this dataset
    /// - `Err(...)` - Error occurred while fetching/refreshing options from provider
    pub async fn latest_storage_options(&self) -> Result<Option<StorageOptions>> {
        self.table_state.latest_storage_options().await
    }

    pub fn data_dir(&self) -> Path {
        self.table_state.data_dir()
    }

    pub fn indices_dir(&self) -> Path {
        self.table_state.indices_dir()
    }

    pub fn transactions_dir(&self) -> Path {
        self.table_state.transactions_dir()
    }

    pub fn deletions_dir(&self) -> Path {
        self.table_state.deletions_dir()
    }

    pub fn versions_dir(&self) -> Path {
        self.table_state.versions_dir()
    }

    pub(crate) fn data_file_dir(&self, data_file: &DataFile) -> Result<Path> {
        self.data_file_dir_for_base(data_file.base_id)
    }

    /// Resolve the object store and complete object key of a manifest data file.
    ///
    /// Uses the file's registered base when present, including dataset-root bases
    /// used by shallow clones. Otherwise uses this dataset's `data/` directory.
    /// This does not read the file, so it also works for missing objects.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn inspect(dataset: &Dataset) -> Result<()> {
    /// for fragment in dataset.iter_fragments() {
    ///     for file in &fragment.files {
    ///         let (store, path) = dataset.resolve_data_file_location(file).await?;
    ///         let exists = store.exists(&path).await?;
    /// #       let _ = exists;
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn resolve_data_file_location(
        &self,
        data_file: &DataFile,
    ) -> Result<(Arc<ObjectStore>, Path)> {
        let path = self.data_file_dir(data_file)?.join(data_file.path.as_str());
        let store = self.object_store_for_data_file(data_file).await?;
        Ok((store, path))
    }

    /// Create a [`DataFile`] by reading metadata from an existing lance file.
    ///
    /// This reads the file's schema and version information, matches columns to
    /// the dataset's schema to determine field IDs, and calculates column indices.
    /// This is useful for constructing `DataFile` metadata needed for operations
    /// like [`Operation::DataReplacement`].
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the data file, relative to the dataset's data directory.
    /// * `base_id` - The base path ID if the file is outside the dataset directory.
    pub async fn create_data_file(&self, path: &str, base_id: Option<u32>) -> Result<DataFile> {
        let data_dir = self.data_file_dir_for_base(base_id)?;
        let filepath = data_dir.clone().join(path);

        let object_store = self.object_store(base_id).await?;

        // Get file size
        let file_size = object_store.size(&filepath).await?;

        // Read file metadata
        let scheduler = ScanScheduler::new(
            object_store.clone(),
            SchedulerConfig::new(2 * 1024 * 1024 * 1024),
        );
        let file = scheduler
            .open_file(&filepath, &CachedFileSize::new(file_size))
            .await?;
        let file_metadata = FileReader::read_all_metadata(&file).await?;

        let lance_file_format = file_metadata.version;
        let physical_columns = file_metadata.column_metadatas.len();
        let has_footer_orphans = file_metadata.file_schema.fields.len() > physical_columns;
        let dataset_schema = self.schema();
        let mut represented_columns = 0usize;
        let mut column_names = Vec::new();
        let mut consumed_top_level_fields = 0usize;

        fn field_contains_blob(field: &lance_core::datatypes::Field) -> bool {
            field.is_blob() || field.children.iter().any(field_contains_blob)
        }

        fn field_names_match(
            fields: &[lance_core::datatypes::Field],
            start: usize,
            expected: &arrow_schema::Fields,
        ) -> bool {
            fields
                .get(start..start + expected.len())
                .is_some_and(|candidate| {
                    candidate
                        .iter()
                        .zip(expected.iter())
                        .all(|(field, expected)| field.name == expected.name().as_str())
                })
        }

        fn blob_descriptor_orphan_len(
            fields: &[lance_core::datatypes::Field],
            start: usize,
        ) -> usize {
            if field_names_match(fields, start, &lance_core::datatypes::BLOB_V2_DESC_FIELDS) {
                lance_core::datatypes::BLOB_V2_DESC_FIELDS.len()
            } else if field_names_match(fields, start, &lance_core::datatypes::BLOB_DESC_FIELDS) {
                lance_core::datatypes::BLOB_DESC_FIELDS.len()
            } else {
                0
            }
        }

        fn validate_file_field_matches_dataset(
            dataset_field: &lance_core::datatypes::Field,
            file_field: &lance_core::datatypes::Field,
            path: &str,
        ) -> Result<()> {
            use arrow_schema::DataType::{LargeList, List, Timestamp};

            if dataset_field.name != file_field.name {
                return Err(Error::invalid_input(format!(
                    "Schema mismatch: expected field '{}' but file has '{}'",
                    path, file_field.name
                )));
            }

            if dataset_field.is_blob() && file_field.is_blob() {
                return Ok(());
            }

            // Reads decode a replacement file's values as the dataset field
            // types, so values stored as another type read back as other
            // values or fail to decode. List offset widths and timestamp time
            // zone names do not change the stored values, and list items are
            // compared as children below. Blob columns are skipped: their
            // file layout can legitimately differ from the dataset's type.
            let same_values = match (dataset_field.data_type(), file_field.data_type()) {
                (List(_) | LargeList(_), List(_) | LargeList(_)) => true,
                // A naive timestamp is a wall-clock time and one with a time
                // zone is an instant, so only the zone's name may differ.
                (Timestamp(dataset_unit, dataset_tz), Timestamp(file_unit, file_tz)) => {
                    dataset_unit == file_unit && dataset_tz.is_some() == file_tz.is_some()
                }
                _ => dataset_field.logical_type == file_field.logical_type,
            };
            if !same_values
                && !field_contains_blob(dataset_field)
                && !field_contains_blob(file_field)
            {
                return Err(Error::invalid_input(format!(
                    "Schema mismatch: field '{}' has type {} in dataset schema but {} in file schema",
                    path, dataset_field.logical_type, file_field.logical_type
                )));
            }

            if dataset_field.children.len() != file_field.children.len() {
                return Err(Error::invalid_input(format!(
                    "Schema mismatch: field '{}' has {} children in dataset schema but {} children in file schema",
                    path,
                    dataset_field.children.len(),
                    file_field.children.len()
                )));
            }

            for (dataset_child, file_child) in
                dataset_field.children.iter().zip(&file_field.children)
            {
                let child_path = format!("{}.{}", path, dataset_child.name);
                validate_file_field_matches_dataset(dataset_child, file_child, &child_path)?;
            }

            Ok(())
        }

        let file_schema_fields = &file_metadata.file_schema.fields;
        let mut idx = 0usize;
        while represented_columns < physical_columns {
            let Some(field) = file_schema_fields.get(idx) else {
                return Err(Error::invalid_input(format!(
                    "Schema mismatch: file schema ended after representing {} physical columns but file has {} columns",
                    represented_columns, physical_columns
                )));
            };

            let Some(dataset_field) = dataset_schema.field(&field.name) else {
                return Err(Error::invalid_input(format!(
                    "Schema mismatch: file has extra field '{}'",
                    field.name
                )));
            };
            validate_file_field_matches_dataset(dataset_field, field, &field.name)?;

            represented_columns += file_versions::physical_column_count(lance_file_format, field);
            column_names.push(field.name.as_str());
            consumed_top_level_fields = idx + 1;
            idx += 1;

            if has_footer_orphans && field_contains_blob(field) {
                loop {
                    let skipped = blob_descriptor_orphan_len(file_schema_fields, idx);
                    if skipped == 0 {
                        break;
                    }
                    consumed_top_level_fields = idx + skipped;
                    idx += skipped;
                }
            }
        }

        if represented_columns != physical_columns {
            return Err(Error::invalid_input(format!(
                "Schema mismatch: file schema represents {} physical columns but file has {} columns",
                represented_columns, physical_columns
            )));
        }

        if let Some(field) = file_schema_fields.get(consumed_top_level_fields) {
            return Err(Error::invalid_input(format!(
                "Schema mismatch: file has extra field '{}'",
                field.name
            )));
        }

        let projected_ds_schema = self.schema().project(&column_names)?;

        let (fields, column_indices) =
            file_versions::data_file_columns(lance_file_format, &projected_ds_schema);
        let represented_dataset_columns = column_indices
            .iter()
            .filter(|column_index| **column_index >= 0)
            .count();

        if represented_dataset_columns != physical_columns {
            return Err(Error::invalid_input(format!(
                "Schema mismatch: dataset projection maps to {} physical columns but file has {} columns",
                represented_dataset_columns, physical_columns
            )));
        }

        if fields.is_empty() && physical_columns > 0 {
            return Err(Error::invalid_input(
                "Schema mismatch: file has columns but none matched the dataset schema",
            ));
        }

        let file_size_nz = NonZero::new(file_size);
        Ok(DataFile::new(
            path,
            fields,
            column_indices,
            lance_file_format,
            file_size_nz,
            base_id,
        ))
    }

    /// Resolve the data directory for a given base_id.
    ///
    /// If `base_id` is `None`, returns the default data directory.
    pub(crate) fn data_file_dir_for_base(&self, base_id: Option<u32>) -> Result<Path> {
        self.table_view().data_file_dir_for_base(base_id)
    }

    /// Resolve the object store for the primary dataset or an additional base.
    ///
    /// Pass `None` to get the primary dataset object store. Pass `Some(base_id)`
    /// when resolving a file whose metadata references an additional base.
    pub async fn object_store(&self, base_id: Option<u32>) -> Result<Arc<ObjectStore>> {
        self.table_view().object_store(base_id).await
    }

    /// The `ObjectStoreParams` this dataset was opened with, or `None` when
    /// opened without explicit params. Lets a caller re-open a derived path
    /// (e.g. a MemWAL SSTable) with the same store this dataset used.
    pub fn store_params(&self) -> Option<&ObjectStoreParams> {
        self.table_state.store_params.as_deref()
    }

    /// Dataset-level file-reader defaults supplied through [`ReadParams`].
    ///
    /// `None` means no explicit options were supplied and readers use their
    /// defaults. Clone these options when creating a per-read override so that
    /// unrelated settings are preserved without changing the shared dataset.
    ///
    /// ```
    /// # use lance::Dataset;
    /// # use lance::dataset::scanner::Scanner;
    /// fn scan_with_byte_limit(dataset: &Dataset) -> Scanner {
    ///     let mut options = dataset.file_reader_options().cloned().unwrap_or_default();
    ///     options.batch_size_bytes = Some(1024 * 1024);
    ///     let mut scanner = dataset.scan();
    ///     scanner.with_file_reader_options(options);
    ///     scanner
    /// }
    /// ```
    pub fn file_reader_options(&self) -> Option<&FileReaderOptions> {
        self.table_state.file_reader_options.as_deref()
    }

    /// Return a copy of this dataset with updated dataset-level file-reader
    /// defaults.
    ///
    /// The returned handle shares the manifest, object stores, session, and
    /// caches with this dataset. Only the defaults used by subsequent scans
    /// change; see [`Self::file_reader_options`].
    pub fn with_file_reader_options(&self, file_reader_options: Option<FileReaderOptions>) -> Self {
        let mut cloned = self.clone();
        cloned.table_state.file_reader_options = file_reader_options.map(Arc::new);
        cloned
    }

    pub(crate) async fn object_store_for_data_file(
        &self,
        data_file: &DataFile,
    ) -> Result<Arc<ObjectStore>> {
        self.object_store(data_file.base_id).await
    }

    pub(crate) async fn object_store_for_index(
        &self,
        index: &IndexMetadata,
    ) -> Result<Arc<ObjectStore>> {
        self.object_store(index.base_id).await
    }

    /// This table as a reader of `manifest`, another manifest of it, sees its
    /// files: through that manifest's base paths, with base stores opened
    /// afresh when those differ, since `base_object_stores` caches by id alone.
    pub(crate) fn with_manifest_view(&self, manifest: Arc<Manifest>) -> Self {
        let table = self.table_view().with_manifest_view(manifest);
        Self::from_table_with_index_cache(table, self.session.clone(), self.index_cache.clone())
    }

    /// Get the indices directory for a specific index, considering its base_id
    pub(crate) fn indice_files_dir(&self, index: &IndexMetadata) -> Result<Path> {
        self.table_view().indice_files_dir(index)
    }

    pub fn session(&self) -> Arc<Session> {
        self.session.clone()
    }

    /// Get the currently checked-out version id.
    ///
    /// This is a cheap accessor that reads the id directly from the loaded
    /// manifest without constructing the full [Version] summary.
    pub fn version_id(&self) -> u64 {
        self.table_view().version_id()
    }

    /// Get the currently checked-out version details.
    ///
    /// This constructs a full [Version], including summary metadata derived
    /// from the loaded manifest fragments.
    pub fn version(&self) -> Version {
        self.table_view().version()
    }

    /// Get the number of entries currently in the index cache.
    pub async fn index_cache_entry_count(&self) -> usize {
        self.session.index_cache.size().await
    }

    /// Get cache hit ratio.
    pub async fn index_cache_hit_rate(&self) -> f32 {
        let stats = self.session.index_cache_stats().await;
        stats.hit_ratio()
    }

    pub fn cache_size_bytes(&self) -> u64 {
        self.session.deep_size_of() as u64
    }

    /// Get all versions, in ascending order.
    ///
    /// Every manifest in the history is read. Use [`Self::versions_with_options`] to
    /// read only a page, or [`Self::count_versions`] to avoid reading manifests at all.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        self.table_state.versions().await
    }

    /// Get one page of versions in the current version history.
    ///
    /// Versions are ordered ascending by version, or descending when `descending`
    /// is true. `offset` is the number of versions to skip and `limit` is the
    /// maximum number to return, so `versions_with_options(0, None, false)` is the
    /// whole history. `None` means no limit, `Some(0)` returns an empty page without
    /// listing or reading any manifest, and an `offset` past the end of the history
    /// returns an empty page. Detached versions are never part of the history.
    ///
    /// Only the manifests in the returned page are read, concurrently within the
    /// store's effective IO window.
    ///
    /// With `descending` on a store whose listing is lexically ordered and whose
    /// manifests are V2 paths, listing stops after `offset + limit` manifest entries
    /// have been observed, so a page near the newest version does not enumerate the
    /// history. Every other case (ascending pages, V1 manifest paths, or a listing
    /// that is not lexically ordered) enumerates every manifest location before
    /// selecting the page. These are listing-entry bounds: a store may still
    /// materialize more keys, or issue more requests, than the entries it yields.
    pub async fn versions_with_options(
        &self,
        offset: usize,
        limit: Option<usize>,
        descending: bool,
    ) -> Result<Vec<Version>> {
        self.table_state
            .versions_with_options(offset, limit, descending)
            .await
    }

    /// Get the number of versions in the current version history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest.
    pub async fn count_versions(&self) -> Result<u64> {
        self.table_state.count_versions().await
    }

    /// List lightweight references to all attached versions in the current branch's history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest. The references are sorted by version in ascending order.
    /// Detached manifests are excluded; see [`Self::list_detached_manifests`].
    ///
    /// Use [`Self::latest_version_id`] instead when only the latest version is needed.
    pub async fn version_refs(&self) -> Result<Vec<VersionRef>> {
        self.table_state.version_refs().await
    }

    /// List all detached manifest locations.
    ///
    /// Detached manifests are versions that are not part of the main version history.
    /// They are created by `commit_detached` and can be used for staging changes.
    ///
    /// To read transaction properties from a detached manifest:
    /// ```ignore
    /// let detached = dataset.list_detached_manifests().await?;
    /// for location in detached {
    ///     let ds = dataset.checkout_version(location.version).await?;
    ///     let tx = ds.read_transaction().await?;
    ///     // Access tx.transaction_properties
    /// }
    /// ```
    pub async fn list_detached_manifests(&self) -> Result<Vec<ManifestLocation>> {
        self.table_state.list_detached_manifests().await
    }

    /// Get the latest version of the dataset
    /// This is meant to be a fast path for checking if a dataset has changed. This is why
    /// we don't return the full version struct.
    pub async fn latest_version_id(&self) -> Result<u64> {
        self.table_state.latest_version_id().await
    }

    /// Return whether the dataset has a newer committed version.
    pub async fn is_stale(&self) -> Result<bool> {
        self.table_view().is_stale().await
    }

    /// Return whether the immediate attached successor manifest exists.
    ///
    /// This is a fast contiguous-history probe. It does not resolve the latest
    /// version and may return `false` if intermediate manifests have been
    /// removed. Callers that need a general freshness check should use
    /// [`Self::is_stale`].
    #[doc(hidden)]
    pub async fn has_successor_version(&self) -> Result<bool> {
        self.table_view().has_successor_version().await
    }

    pub fn count_fragments(&self) -> usize {
        self.table_view().count_fragments()
    }

    /// Get the schema of the dataset
    pub fn schema(&self) -> &Schema {
        self.table_view().schema()
    }

    /// Similar to [Self::schema], but only returns fields that are not marked as blob columns
    /// Creates a new empty projection into the dataset schema
    pub fn empty_projection(self: &Arc<Self>) -> Projection {
        Projection::empty(self.clone())
    }

    /// Creates a projection that includes all columns in the dataset
    pub fn full_projection(self: &Arc<Self>) -> Projection {
        Projection::full(self.clone())
    }

    /// Get fragments.
    pub fn get_fragments(&self) -> Vec<FileFragment> {
        let dataset = Arc::new(self.clone());
        let table = Arc::new(self.table_view().to_table());
        self.iter_fragments()
            .map(|fragment| {
                FileFragment::from_table(dataset.clone(), table.clone(), fragment.clone())
            })
            .collect()
    }

    /// Iterate over manifest fragments without allocating [`FileFragment`] wrappers.
    pub fn iter_fragments(&self) -> impl Iterator<Item = &Fragment> {
        self.table_view().iter_fragments()
    }

    pub fn get_fragment(&self, fragment_id: usize) -> Option<FileFragment> {
        let metadata = self.find_fragment(fragment_id as u64)?.clone();
        Some(FileFragment::new(Arc::new(self.clone()), metadata))
    }

    pub fn fragments(&self) -> &Arc<Vec<Fragment>> {
        self.table_view().fragments()
    }

    /// Ids of the fragments in this version.
    pub fn fragment_bitmap(&self) -> &Arc<RoaringBitmap> {
        self.table_view().fragment_bitmap()
    }

    pub(crate) fn normalize_fragment_ids(fragment_ids: &[u32]) -> Vec<u32> {
        let mut ids = fragment_ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    pub(crate) fn get_fragments_from_ids(&self, fragment_ids: &[u32]) -> Result<Vec<FileFragment>> {
        let ordered_ids = Self::normalize_fragment_ids(fragment_ids);
        let fragments = self.get_frags_from_ordered_ids(&ordered_ids);
        if let Some(missing_id) = fragments
            .iter()
            .zip(ordered_ids.iter())
            .find_map(|(fragment, fragment_id)| fragment.is_none().then_some(*fragment_id))
        {
            return Err(Error::invalid_input(format!(
                "Unknown fragment id {missing_id} in fragment filter; not part of the current dataset version"
            )));
        }

        Ok(fragments.into_iter().flatten().collect())
    }

    pub(crate) fn get_existing_fragments_from_ids(
        &self,
        fragment_ids: &[u32],
    ) -> Vec<FileFragment> {
        let ordered_ids = Self::normalize_fragment_ids(fragment_ids);
        self.get_frags_from_ordered_ids(&ordered_ids)
            .into_iter()
            .flatten()
            .collect()
    }

    pub(crate) fn get_fragment_metadata_from_ids(
        &self,
        fragment_ids: &[u32],
    ) -> Result<Vec<Fragment>> {
        Ok(self
            .get_fragments_from_ids(fragment_ids)?
            .into_iter()
            .map(|fragment| fragment.metadata().clone())
            .collect())
    }

    pub(crate) fn get_existing_fragment_metadata_from_ids(
        &self,
        fragment_ids: &[u32],
    ) -> Vec<Fragment> {
        self.get_existing_fragments_from_ids(fragment_ids)
            .into_iter()
            .map(|fragment| fragment.metadata().clone())
            .collect()
    }

    pub(crate) async fn count_rows_in_fragments(&self, fragment_ids: &[u32]) -> Result<usize> {
        let fragments = self.get_fragments_from_ids(fragment_ids)?;
        self.count_rows_in_resolved_fragments(fragments).await
    }

    pub(crate) async fn count_rows_in_existing_fragments(
        &self,
        fragment_ids: &[u32],
    ) -> Result<usize> {
        let fragments = self.get_existing_fragments_from_ids(fragment_ids);
        self.count_rows_in_resolved_fragments(fragments).await
    }

    async fn count_rows_in_resolved_fragments(
        &self,
        fragments: Vec<FileFragment>,
    ) -> Result<usize> {
        let counts = stream::iter(fragments)
            .map(|fragment| async move { fragment.count_rows(None).await })
            .buffer_unordered(16)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(counts.iter().sum())
    }

    /// Resolves fragments for the given ids without scanning the manifest.
    ///
    /// The ids do not need to be sorted or deduplicated. Each id is resolved
    /// independently via the fragment bitmap.
    pub fn get_frags_from_ordered_ids(&self, ordered_ids: &[u32]) -> Vec<Option<FileFragment>> {
        let dataset = Arc::new(self.clone());
        let table = Arc::new(self.table_view().to_table());
        ordered_ids
            .iter()
            .map(|id| {
                if !self.table_state.fragment_bitmap.contains(*id) {
                    return None;
                }
                let fragment_index = self.table_state.fragment_bitmap.rank(*id) as usize - 1;
                let fragment = self.manifest.fragments.get(fragment_index)?;
                debug_assert_eq!(
                    fragment.id, *id as u64,
                    "fragment_bitmap rank({id}) resolved to fragment {}, but fragment_bitmap and manifest.fragments are expected to stay in sync",
                    fragment.id
                );
                Some(FileFragment::from_table(dataset.clone(), table.clone(), fragment.clone()))
            })
            .collect()
    }

    /// Look up the fragment with `id` in the manifest.
    ///
    /// `Manifest::fragments` is kept sorted by id, so this binary searches
    /// rather than scanning. Two kinds of manifest predate that invariant and
    /// are still readable: those written before fragments were forced into id
    /// order (Lance 0.10 and earlier), and those with duplicate fragment ids
    /// (Lance 0.16 and earlier). Neither is rejected on read, so the search
    /// result is checked and a scan takes over when it does not match --
    /// returning some other fragment's data would be silent corruption.
    pub(crate) fn find_fragment(&self, id: u64) -> Option<&Fragment> {
        self.table_view().find_fragment(id)
    }

    // This method filters deleted items from `addr_or_ids` using `addrs` as a reference
    async fn filter_addr_or_ids(&self, addr_or_ids: &[u64], addrs: &[u64]) -> Result<Vec<u64>> {
        // The final zip pairs these positionally; misalignment must fail
        // loud rather than truncate.
        if addr_or_ids.len() != addrs.len() {
            return Err(Error::internal(format!(
                "filter_addr_or_ids: addr_or_ids has {} entries but addrs has {}",
                addr_or_ids.len(),
                addrs.len()
            )));
        }
        if addrs.is_empty() {
            return Ok(Vec::new());
        }

        let mut perm = permutation::sort(addrs);
        // First we sort the addrs, then we transform from Vec<u64> to Vec<Option<u64>> and then
        // we un-sort and use the None values to filter `addr_or_ids`
        let sorted_addrs = perm.apply_slice(addrs);

        // Only collect deletion vectors for the fragments referenced by the given addrs
        let referenced_frag_ids = sorted_addrs
            .iter()
            .map(|addr| RowAddress::from(*addr).fragment_id())
            .dedup()
            .collect::<Vec<_>>();
        let frags = self.get_frags_from_ordered_ids(&referenced_frag_ids);
        let dv_futs = frags
            .iter()
            .map(|frag| {
                if let Some(frag) = frag {
                    frag.get_deletion_vector().boxed()
                } else {
                    std::future::ready(Ok(None)).boxed()
                }
            })
            .collect::<Vec<_>>();
        let dvs = stream::iter(dv_futs)
            .buffered(self.table_state.object_store.io_parallelism())
            .try_collect::<Vec<_>>()
            .await?;

        // Iterate through the sorted addresses and sorted fragments (and sorted deletion vectors)
        // and filter out addresses that have been deleted
        let mut filtered_sorted_addrs = Vec::with_capacity(sorted_addrs.len());
        let mut sorted_addr_iter = sorted_addrs.into_iter().map(RowAddress::from);
        let mut next_addr = sorted_addr_iter.next().unwrap();
        let mut exhausted = false;

        for frag_dv in frags.iter().zip(dvs).zip(referenced_frag_ids.iter()) {
            let ((frag, dv), frag_id) = frag_dv;
            if frag.is_some() {
                // Frag exists
                if let Some(dv) = dv.as_ref() {
                    // Deletion vector exists, scan DV
                    for deleted in dv.to_sorted_iter() {
                        while next_addr.fragment_id() == *frag_id
                            && next_addr.row_offset() < deleted
                        {
                            filtered_sorted_addrs.push(Some(u64::from(next_addr)));
                            if let Some(next) = sorted_addr_iter.next() {
                                next_addr = next;
                            } else {
                                exhausted = true;
                                break;
                            }
                        }
                        if exhausted {
                            break;
                        }
                        if next_addr.fragment_id() != *frag_id {
                            break;
                        }
                        if next_addr.row_offset() == deleted {
                            filtered_sorted_addrs.push(None);
                            if let Some(next) = sorted_addr_iter.next() {
                                next_addr = next;
                            } else {
                                exhausted = true;
                                break;
                            }
                        }
                    }
                }
                if exhausted {
                    break;
                }
                // Either no deletion vector, or we've exhausted it, keep everything else
                // in this frag
                while next_addr.fragment_id() == *frag_id {
                    filtered_sorted_addrs.push(Some(u64::from(next_addr)));
                    if let Some(next) = sorted_addr_iter.next() {
                        next_addr = next;
                    } else {
                        break;
                    }
                }
            } else {
                // Frag doesn't exist (possibly deleted), delete all items
                while next_addr.fragment_id() == *frag_id {
                    filtered_sorted_addrs.push(None);
                    if let Some(next) = sorted_addr_iter.next() {
                        next_addr = next;
                    } else {
                        break;
                    }
                }
            }
        }

        // filtered_sorted_ids is now a Vec with the same size as sorted_addrs, but with None
        // values where the corresponding address was deleted.  We now need to un-sort it and
        // filter out the deleted addresses.
        perm.apply_inv_slice_in_place(&mut filtered_sorted_addrs);
        Ok(addr_or_ids
            .iter()
            .zip(filtered_sorted_addrs)
            .filter_map(|(addr_or_id, maybe_addr)| maybe_addr.map(|_| *addr_or_id))
            .collect())
    }

    pub(crate) async fn filter_deleted_ids(&self, ids: &[u64]) -> Result<Vec<u64>> {
        let (ids, addresses) = if let Some(row_id_index) = get_row_id_index(self).await? {
            // Ids absent from the deletion-aware index are deleted; drop
            // them from both lists to keep the zip aligned. ids.len() is an
            // upper bound on the output size, so allocate once up front.
            let mut live_ids = Vec::with_capacity(ids.len());
            let mut addresses = Vec::with_capacity(ids.len());
            for id in ids {
                if let Some(address) = row_id_index.get(*id)? {
                    live_ids.push(*id);
                    addresses.push(u64::from(address));
                }
            }
            (Cow::Owned(live_ids), Cow::Owned(addresses))
        } else {
            (Cow::Borrowed(ids), Cow::Borrowed(ids))
        };

        self.filter_addr_or_ids(&ids, &addresses).await
    }

    /// Gets the number of files that are so small they don't even have a full
    /// group. These are considered too small because reading many of them is
    /// much less efficient than reading a single file because the separate files
    /// split up what would otherwise be single IO requests into multiple.
    pub async fn num_small_files(&self, max_rows_per_group: usize) -> usize {
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.physical_rows().await })
            .buffered(self.table_state.object_store.io_parallelism())
            .try_filter(|row_count| futures::future::ready(*row_count < max_rows_per_group))
            .count()
            .await
    }

    pub async fn validate(&self) -> Result<()> {
        // All fragments have unique ids
        let id_counts =
            self.manifest
                .fragments
                .iter()
                .map(|f| f.id)
                .fold(HashMap::new(), |mut acc, id| {
                    *acc.entry(id).or_insert(0) += 1;
                    acc
                });
        for (id, count) in id_counts {
            if count > 1 {
                return Err(Error::corrupt_file(
                    self.table_state.base.clone(),
                    format!(
                        "Duplicate fragment id {} found in dataset {:?}",
                        id, self.table_state.base
                    ),
                ));
            }
        }

        // Fragments are sorted in increasing fragment id order
        self.manifest
            .fragments
            .iter()
            .map(|f| f.id)
            .try_fold(0, |prev, id| {
                if id < prev {
                    Err(Error::corrupt_file(self.table_state.base.clone(), format!(
                        "Fragment ids are not sorted in increasing fragment-id order. Found {} after {} in dataset {:?}",
                        id, prev, self.table_state.base
                    )))
                } else {
                    Ok(id)
                }
            })?;

        // All fragments have equal lengths
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.validate().await })
            .buffer_unordered(self.table_state.object_store.io_parallelism())
            .try_collect::<Vec<()>>()
            .await?;

        rowids::validate_stable_row_ids(self).await?;

        // Validate indices. Over the complete list: these checks are about what
        // the manifest says, not about what this build can use, and duplicate
        // uuids or overlapping coverage are no less corrupt for involving an
        // index this build has no reader for. `migrate_indices` already runs the
        // same overlap check over the complete list on every commit.
        let indices = crate::index::loading::load_all_indices(self).await?;
        self.validate_indices(&indices)?;

        Ok(())
    }

    fn validate_indices(&self, indices: &[IndexMetadata]) -> Result<()> {
        // Make sure there are no duplicate ids
        let mut index_ids = HashSet::new();
        for index in indices.iter() {
            if !index_ids.insert(&index.uuid) {
                return Err(Error::corrupt_file(
                    self.table_state.manifest_location.path.clone(),
                    format!(
                        "Duplicate index id {} found in dataset {:?}",
                        index.uuid, self.table_state.base
                    ),
                ));
            }
        }

        // For each index name, make sure there is no overlap in fragment bitmaps
        if let Err(err) = detect_overlapping_fragments(indices) {
            let mut message = "Overlapping fragments detected in dataset.".to_string();
            for (index_name, overlapping_frags) in err.bad_indices {
                message.push_str(&format!(
                    "\nIndex {:?} has overlapping fragments: {:?}",
                    index_name, overlapping_frags
                ));
            }
            return Err(Error::corrupt_file(
                self.table_state.manifest_location.path.clone(),
                message,
            ));
        };

        Ok(())
    }

    /// Migrate the dataset to use the new manifest path scheme.
    ///
    /// This function will rename all V1 manifests to [ManifestNamingScheme::V2].
    /// These paths provide more efficient opening of datasets with many versions
    /// on object stores.
    ///
    /// This function is idempotent, and can be run multiple times without
    /// changing the state of the object store.
    ///
    /// However, it should not be run while other concurrent operations are happening.
    /// And it should also run until completion before resuming other operations.
    ///
    /// ```rust
    /// # use lance::dataset::Dataset;
    /// # use lance_table::io::commit::ManifestNamingScheme;
    /// # use lance_datagen::{array, RowCount, BatchCount};
    /// # use arrow_array::types::Int32Type;
    /// # use lance::dataset::write::WriteParams;
    /// # let data = lance_datagen::gen_batch()
    /// #  .col("key", array::step::<Int32Type>())
    /// #  .into_reader_rows(RowCount::from(10), BatchCount::from(1));
    /// # let fut = async {
    /// # let params = WriteParams {
    /// #     enable_v2_manifest_paths: false,
    /// #     ..Default::default()
    /// # };
    /// let mut dataset = Dataset::write(data, "memory://test", Some(params)).await.unwrap();
    /// assert_eq!(dataset.manifest_location().naming_scheme, ManifestNamingScheme::V1);
    ///
    /// dataset.migrate_manifest_paths_v2().await.unwrap();
    /// assert_eq!(dataset.manifest_location().naming_scheme, ManifestNamingScheme::V2);
    /// # };
    /// # tokio::runtime::Runtime::new().unwrap().block_on(fut);
    /// ```
    pub async fn migrate_manifest_paths_v2(&mut self) -> Result<()> {
        migrate_scheme_to_v2(
            self.table_state.object_store.as_ref(),
            &self.table_state.base,
        )
        .await?;
        // We need to re-open.
        let latest_version = self.latest_version_id().await?;
        *self = self.checkout_version(latest_version).await?;
        Ok(())
    }

    /// Assign a stable row ID sequence to every fragment, contiguously from
    /// `start`, and return the resulting `next_row_id` high-water mark. Only
    /// reached when the feature is off, so no fragment has a sequence yet.
    fn assign_stable_row_ids_for_migration(fragments: &mut [Fragment], start: u64) -> Result<u64> {
        let mut next_row_id = start;
        for fragment in fragments.iter_mut() {
            let physical_rows = fragment.physical_rows.ok_or_else(|| {
                Error::internal(format!(
                    "Fragment {} is missing physical_rows; cannot assign stable row IDs",
                    fragment.id
                ))
            })? as u64;
            let end = next_row_id
                .checked_add(physical_rows)
                .ok_or_else(|| Error::internal("Row ID overflow during stable row ID migration"))?;
            let sequence = RowIdSequence::from(next_row_id..end);
            fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&sequence).into()));
            next_row_id = end;
        }
        Ok(next_row_id)
    }

    /// Migrate a table to use stable row IDs.
    ///
    /// Afterwards a row keeps the same id for its lifetime: compaction, update
    /// and merge insert relocate or rewrite the row without changing its id.
    ///
    /// Migration assigns a fresh id to every physical row position, in fragment
    /// order, starting from the dataset's `next_row_id` high-water mark. Before
    /// migration `_rowid` is a row address, a different namespace from the ids
    /// assigned here, so a `_rowid` recorded beforehand must not be reused as a
    /// row id afterwards.
    ///
    /// A single Merge commit assigns row ID sequences to all fragments and
    /// activates the stable row ID feature flag atomically. Because `Merge`
    /// conflicts with all data-modifying operations, a successful commit
    /// guarantees no concurrent write occurred — no separate validation step
    /// is needed.
    ///
    /// **No retries are attempted.** Callers should quiesce concurrent writes
    /// before running this migration. If a conflicting write is detected, this
    /// method returns an error and the caller must retry.
    ///
    /// This method is idempotent: if the table already uses stable row IDs,
    /// it returns `Ok(())` immediately.
    pub async fn migrate_to_stable_row_ids(&mut self) -> Result<()> {
        if self.manifest.uses_stable_row_ids() {
            return Ok(());
        }

        let mut fragments = self.manifest.fragments.as_ref().clone();
        // Restore carries the high-water mark forward across a version that
        // predates activation, so a re-migration must allocate above it rather
        // than reissue ids the earlier versions still hold.
        let next_row_id =
            Self::assign_stable_row_ids_for_migration(&mut fragments, self.manifest.next_row_id)?;
        let schema = self.manifest.schema.clone();
        let read_version = self.manifest.version;

        let transaction = Transaction::new(
            read_version,
            Operation::Merge {
                fragments,
                schema,
                preserves_nullability: true,
            },
            None,
        );

        let new_ds = CommitBuilder::new(Arc::new(self.clone()))
            .with_max_retries(0)
            .with_stable_row_id_migration_activation(next_row_id)
            .execute(transaction)
            .await?;

        *self = new_ds;
        Ok(())
    }

    /// Activate monotonic, non-reusable field IDs for a legacy dataset.
    ///
    /// The activation commit records the current maximum referenced field ID as
    /// a persistent high-water mark. Later schema changes allocate above it even
    /// after fields and their files are dropped. Activation is one-way and
    /// idempotent.
    ///
    /// Before calling this method, ensure that all clients that can write to the
    /// dataset enforce writer feature flags, rejecting writes when they do not
    /// support a required flag. Clients that ignore these flags must no longer
    /// write to the dataset: they may discard the high-water mark and allow
    /// field IDs to be reused. This method cannot enforce that client policy.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # async fn activate(dataset: &mut Dataset) -> Result<()> {
    /// dataset.migrate_to_non_reusable_field_ids().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn migrate_to_non_reusable_field_ids(&mut self) -> Result<()> {
        if self.manifest.uses_non_reusable_field_ids() {
            return Ok(());
        }

        let mut repaired_manifest = self.manifest.as_ref().clone();
        fix_schema(&mut repaired_manifest)?;
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Merge {
                fragments: repaired_manifest.fragments.as_ref().clone(),
                schema: repaired_manifest.schema,
                preserves_nullability: true,
            },
            None,
        );
        let new_ds = CommitBuilder::new(Arc::new(self.clone()))
            .with_max_retries(0)
            .with_non_reusable_field_id_migration_activation()
            .execute(transaction)
            .await?;
        *self = new_ds;
        Ok(())
    }

    /// Shared clone-target preflight for `shallow_clone` and `deep_clone`:
    /// permit the clone only when the target definitively holds no dataset.
    /// Only the codebase-wide "dataset absent" pair passes: the built-in
    /// resolver reports an empty `_versions/` listing as `NotFound`, while
    /// handlers with an external source of truth use `DatasetNotFound` (the
    /// same discrimination the write path's destination probe applies). Any
    /// other resolver failure (storage, auth, corrupt manifest listing)
    /// propagates instead of letting the clone write into a target it failed
    /// to inspect.
    async fn ensure_clone_target_absent(
        commit_handler: &dyn CommitHandler,
        target_base: &Path,
        target_store: &ObjectStore,
        target_path: &str,
    ) -> Result<()> {
        match commit_handler
            .resolve_latest_location(target_base, target_store)
            .await
        {
            Ok(_) => Err(Error::dataset_already_exists(target_path.to_string())),
            Err(Error::NotFound { .. } | Error::DatasetNotFound { .. }) => Ok(()),
            Err(error) => Err(error),
        }
    }

    /// Shallow clone the target version into a new dataset at target_path.
    /// 'target_path': the uri string to clone the dataset into.
    /// 'version': the version cloned from, could be a version number or tag.
    /// 'store_params': the object store params to use for the new dataset.
    ///
    /// The returned handle retains the source's store parameters for shared files.
    /// When reopening a clone across stores, supply the source parameters with
    /// [`DatasetBuilder::with_base_store_params`].
    pub async fn shallow_clone(
        &mut self,
        target_path: &str,
        version: impl Into<refs::Ref>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        let store_params = match store_params {
            Some(params) => params,
            None => {
                let mut params = self
                    .table_state
                    .store_params
                    .as_deref()
                    .cloned()
                    .unwrap_or_default();
                // Inherit the custom store, but not its source dataset location.
                #[allow(deprecated)]
                if let Some((_, location)) = params.object_store.as_mut() {
                    *location = lance_io::object_store::uri_to_url(target_path)?;
                }
                params
            }
        };
        let (target_store, _) = ObjectStore::from_uri_and_params(
            self.session.store_registry(),
            target_path,
            &store_params,
        )
        .await?;

        // Prevent cloning into an existing target dataset (parity with
        // `deep_clone`) before anything is written there: a tagged clone
        // stages its relocated FRI details in the target's `_indices/`
        // ahead of the manifest commit, which must not pollute a live
        // dataset. The check goes through the same store and commit handler
        // the commit below writes through. Only a definitive "no dataset
        // here" permits the clone; see `ensure_clone_target_absent`.
        // CommitBuilder extracts its path from the URI when given an explicit store.
        let target_base =
            ObjectStore::extract_path_from_uri(self.session.store_registry(), target_path)?;
        Self::ensure_clone_target_absent(
            self.table_state.commit_handler.as_ref(),
            &target_base,
            &target_store,
            target_path,
        )
        .await?;

        let (ref_name, version_number) = self.resolve_reference(version.into()).await?;
        let source_location = self.branch_location().find_branch(ref_name.as_deref())?;
        let clone_op = Operation::Clone {
            is_shallow: true,
            ref_name,
            ref_version: version_number,
            ref_path: source_location.uri.clone(),
            branch_name: None,
        };
        let transaction = Transaction::new(version_number, clone_op, None);

        let builder = CommitBuilder::new(WriteDestination::Uri(target_path))
            .with_store_params(store_params)
            .with_object_store(target_store)
            .with_source_store(self.table_state.object_store.clone())
            .with_session(self.session.clone())
            .with_commit_handler(self.table_state.commit_handler.clone())
            .with_exact_storage_format(self.manifest.data_storage_format.lance_file_format());
        let mut cloned = builder.execute(transaction).await?;
        // Destination credentials must not replace the bindings for shared
        // source files, including bases inherited through earlier shallow clones.
        cloned.table_state.base_store_params = Some(Arc::new(
            cloned
                .manifest
                .base_paths
                .values()
                .map(|base| {
                    let params = if base.path == source_location.uri {
                        self.store_params_for_base(None)
                    } else {
                        self.store_params_for_base(Some(base))
                    };
                    (base.path.clone(), params)
                })
                .collect(),
        ));
        Ok(cloned)
    }

    /// Deep clone the target version into a new dataset at target_path.
    /// This copies all relevant dataset files (data files, deletion files, and
    /// index files) into the target dataset with bounded memory use.
    ///
    /// The source files are read through this dataset's own object store while the
    /// copies are written through the target object store built from `store_params`.
    /// This makes the clone work across accounts/stores (e.g. between two abfss
    /// accounts). Object-store files are streamed through this process by default;
    /// `LANCE_IO_SERVER_SIDE_COPY_ENABLED` opts same-store copies into
    /// provider-native copy operations. Cross-store copies continue to stream, and
    /// local files retain their filesystem copy path.
    ///
    /// Parameters:
    /// - `target_path`: the URI string to clone the dataset into.
    /// - `version`: the version cloned from, could be a version number, branch head, or tag.
    /// - `store_params`: the object store params for the target dataset (e.g. the
    ///   credentials of the target account).
    ///
    /// Note: external `base_paths` referenced by the source manifest are read through
    /// this dataset's object store; per-base distinct source credentials are not yet
    /// supported (see <https://github.com/lance-format/lance/issues/6093>).
    /// Object-store streaming defaults to at most four concurrent file copies;
    /// `LANCE_DEEP_CLONE_STREAM_CONCURRENCY` overrides that limit for this operation.
    pub async fn deep_clone(
        &mut self,
        target_path: &str,
        version: impl Into<refs::Ref>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        use futures::StreamExt;

        // Resolve source dataset and its manifest using checkout_version
        let src_ds = self.checkout_version(version).await?;
        ensure_can_write_manifest(&src_ds.manifest)?;
        // Rejects a tagged FRI history this writer cannot fully interpret
        // before anything is copied or written to the target.
        let src_paths = src_ds.collect_paths().await?;

        // Prepare target object store and base path
        let (target_store, target_base) = ObjectStore::from_uri_and_params(
            self.session.store_registry(),
            target_path,
            &store_params.clone().unwrap_or_default(),
        )
        .await?;

        // Prevent cloning into an existing target dataset. Only a definitive
        // "no dataset here" permits the clone; see
        // `ensure_clone_target_absent`.
        Self::ensure_clone_target_absent(
            self.table_state.commit_handler.as_ref(),
            &target_base,
            &target_store,
            target_path,
        )
        .await?;

        let build_absolute_path = |relative_path: &str, base: &Path| -> Path {
            let mut path = base.clone();
            for seg in relative_path.split('/') {
                if !seg.is_empty() {
                    path = path.clone().join(seg);
                }
            }
            path
        };

        let configured_io_parallelism = src_ds.table_state.object_store.io_parallelism();
        // Provider-native copy can fall back to streaming for large objects, so every
        // non-direct-local transfer stays within the bounded file-copy window.
        let uses_streaming_copy = !(src_ds.table_state.object_store.has_direct_local_paths()
            && target_store.has_direct_local_paths());
        let stream_copy_parallelism = match std::env::var("LANCE_DEEP_CLONE_STREAM_CONCURRENCY") {
            Ok(value) => Some(parse_deep_clone_stream_concurrency(&value)?),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(value)) => {
                return Err(Error::invalid_input(format!(
                    "LANCE_DEEP_CLONE_STREAM_CONCURRENCY must be valid UTF-8 and a positive \
                     integer, got {value:?}"
                )));
            }
        };
        // Limit the number of concurrently buffered transfers by default while
        // preserving efficient local copies and the operation-specific override.
        let io_parallelism = deep_clone_copy_parallelism(
            configured_io_parallelism,
            uses_streaming_copy,
            stream_copy_parallelism,
        );
        let copy_futures = src_paths
            .iter()
            .map(|(relative_path, base)| {
                let source_store = Arc::clone(&src_ds.table_state.object_store);
                let target_store = Arc::clone(&target_store);
                let src_path = build_absolute_path(relative_path, base);
                let target_path = build_absolute_path(relative_path, &target_base);
                async move {
                    source_store
                        .copy_bulk(&src_path, &target_store, &target_path)
                        .await?;
                    Result::Ok(())
                }
            })
            .collect::<Vec<_>>();

        futures::stream::iter(copy_futures)
            .buffer_unordered(io_parallelism)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        // Record a Clone operation and commit via CommitBuilder
        let ref_name = src_ds.manifest.branch.clone();
        let ref_version = src_ds.table_state.manifest_location.version;
        let clone_op = Operation::Clone {
            is_shallow: false,
            ref_name,
            ref_version,
            ref_path: src_ds.uri().to_string(),
            branch_name: None,
        };
        let txn = Transaction::new(ref_version, clone_op, None);
        let builder = CommitBuilder::new(WriteDestination::Uri(target_path))
            .with_store_params(store_params.clone().unwrap_or_default())
            .with_object_store(target_store.clone())
            .with_source_store(src_ds.table_state.object_store.clone())
            .with_commit_handler(self.table_state.commit_handler.clone())
            .with_exact_storage_format(self.manifest.data_storage_format.lance_file_format())
            .with_deep_clone_files_copied();
        let new_ds = builder.execute(txn).await?;
        Ok(new_ds)
    }

    async fn resolve_reference(&self, reference: refs::Ref) -> Result<(Option<String>, u64)> {
        self.table_view().resolve_reference(reference).await
    }

    /// Collect all (relative_path, path) of the dataset files.
    async fn collect_paths(&self) -> Result<Vec<(String, Path)>> {
        let mut file_paths: Vec<(String, Path)> = Vec::new();
        let mut blob_dirs = HashSet::new();
        for fragment in self.manifest.fragments.iter() {
            for data_file in fragment.referenced_lance_files() {
                let base_root = if let Some(base_id) = data_file.base_id {
                    let base_path =
                        self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                            Error::internal(format!("base_id {} not found", base_id))
                        })?;
                    Path::parse(base_path.path.as_str())?
                } else {
                    self.table_state.base.clone()
                };
                file_paths.push((
                    format!("{}/{}", DATA_DIR, data_file.path.clone()),
                    base_root.clone(),
                ));

                if !data_file
                    .schema(self.schema())
                    .fields_pre_order()
                    .any(|field| field.is_blob_v2())
                {
                    continue;
                }

                // Blob v2 sidecars are not listed in the manifest. Their directory is
                // derived from the owning data file, so enumerate it to copy packed and
                // dedicated payloads without decoding blob descriptors. External blobs
                // remain caller-owned references and are deliberately not collected.
                let data_file_key = blob::data_file_key_from_path(data_file.path.as_str());
                let relative_blob_dir = format!("{}/{}", DATA_DIR, data_file_key);
                let blob_dir = base_root.clone().join(DATA_DIR).join(data_file_key);
                // Overlays can make the same data file reachable from multiple fragments.
                if blob_dirs.insert(blob_dir.clone()) {
                    let mut stream = self.table_state.object_store.read_dir_all(&blob_dir, None);
                    while let Some(meta) = stream.next().await.transpose()? {
                        if let Some(filename) = meta.location.filename() {
                            file_paths.push((
                                format!("{}/{}", relative_blob_dir, filename),
                                base_root.clone(),
                            ));
                        }
                    }
                }
            }
            if let Some(deletion_file) = fragment
                .deletion_file
                .as_ref()
                .filter(|file| !file.is_inline())
            {
                let base_root = if let Some(base_id) = deletion_file.base_id {
                    let base_path =
                        self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                            Error::internal(format!("base_id {} not found", base_id))
                        })?;
                    Path::parse(base_path.path.as_str())?
                } else {
                    self.table_state.base.clone()
                };
                file_paths.push((
                    relative_deletion_file_path(fragment.id, deletion_file),
                    base_root,
                ));
            }
        }

        let indices = read_manifest_indexes(
            self.table_state.object_store.as_ref(),
            &self.table_state.manifest_location,
            &self.manifest,
        )
        .await?;

        for index in &indices {
            if lance_table::system_index::frag_reuse::metadata::is_tagged(index) {
                // The clone commit rewrites a tagged FRI entry under a fresh
                // uuid with local references (its details spill included), so
                // the entry's own `_indices/<uuid>/` directory is not copied;
                // the row maps it references are, into the clone's `_fri/`.
                file_paths.extend(
                    crate::index::frag_reuse::collect_tagged_row_map_paths(self, index).await?,
                );
                continue;
            }
            let base_root = if let Some(base_id) = index.base_id {
                let base_path = self
                    .manifest
                    .base_paths
                    .get(&base_id)
                    .ok_or_else(|| Error::internal(format!("base_id {} not found", base_id)))?;
                Path::parse(base_path.path.as_str())?
            } else {
                self.table_state.base.clone()
            };
            let index_root = base_root
                .clone()
                .join(INDICES_DIR)
                .join(index.uuid.to_string());
            let mut stream = self
                .table_state
                .object_store
                .read_dir_all(&index_root, None);
            while let Some(meta) = stream.next().await.transpose()? {
                if let Some(filename) = meta.location.filename() {
                    file_paths.push((
                        format!("{}/{}/{}", INDICES_DIR, index.uuid, filename),
                        base_root.clone(),
                    ));
                }
            }
        }
        Ok(file_paths)
    }

    /// Run a SQL query against the dataset.
    /// The underlying SQL engine is DataFusion.
    /// Please refer to the DataFusion documentation for supported SQL syntax.
    pub fn sql(&self, sql: &str) -> SqlQueryBuilder {
        SqlQueryBuilder::new(self.clone(), sql)
    }
}

#[cfg(test)]
pub(crate) use lance_table_runtime::commit::TransactionReadSource;

/// # Schema Evolution
///
/// Lance datasets support evolving the schema. Several operations are
/// supported that mirror common SQL operations:
///
/// - [Self::add_columns()]: Add new columns to the dataset, similar to `ALTER TABLE ADD COLUMN`.
/// - [Self::drop_columns()]: Drop columns from the dataset, similar to `ALTER TABLE DROP COLUMN`.
/// - [Self::alter_columns()]: Modify columns in the dataset, changing their name, type, or nullability.
///   Similar to `ALTER TABLE ALTER COLUMN`.
///
/// In addition, one operation is unique to Lance: [`merge`](Self::merge). This
/// operation allows inserting precomputed data into the dataset.
///
/// Because these operations change the schema of the dataset, they will conflict
/// with most other concurrent operations. Therefore, they should be performed
/// when no other write operations are being run.
impl Dataset {
    /// Append new columns to the dataset.
    pub async fn add_columns(
        &mut self,
        transforms: NewColumnTransform,
        read_columns: Option<Vec<String>>,
        batch_size: Option<u32>,
    ) -> Result<()> {
        schema_evolution::add_columns(self, transforms, read_columns, batch_size).await
    }

    /// Modify columns in the dataset, changing their name, type, or nullability.
    ///
    /// If only changing the name or nullability of a column, this is a zero-copy
    /// operation and any indices will be preserved. If changing the type of a
    /// column, the data for that column will be rewritten and any indices will
    /// be dropped. The old column data will not be immediately deleted. To remove
    /// it, call [optimize::compact_files()] and then
    /// [cleanup::cleanup_old_versions()] on the dataset.
    pub async fn alter_columns(&mut self, alterations: &[ColumnAlteration]) -> Result<()> {
        schema_evolution::alter_columns(self, alterations).await
    }

    /// Remove columns from the dataset.
    ///
    /// This is a metadata-only operation and does not remove the data from the
    /// underlying storage. In order to remove the data, you must subsequently
    /// call [optimize::compact_files()] to rewrite the data without the removed columns and
    /// then call [cleanup::cleanup_old_versions()] to remove the old files.
    /// The schema a [`Self::drop_columns`] of `columns` would project to,
    /// with every validation that drop performs and no mutation of its own.
    ///
    /// For a caller that must not write anything until the whole projection
    /// is known to be valid against this revision.
    pub fn plan_drop_columns(&self, columns: &[&str]) -> Result<Schema> {
        schema_evolution::plan_drop_columns(self, columns)
    }

    pub async fn drop_columns(&mut self, columns: &[&str]) -> Result<()> {
        info!(target: TRACE_DATASET_EVENTS, event=DATASET_DROPPING_COLUMN_EVENT, uri = &self.table_state.uri, columns = columns.join(","));
        schema_evolution::drop_columns(self, columns).await
    }

    /// Drop columns from the dataset and return updated dataset. Note that this
    /// is a zero-copy operation and column is not physically removed from the
    /// dataset.
    /// Parameters:
    /// - `columns`: the list of column names to drop.
    #[deprecated(since = "0.9.12", note = "Please use `drop_columns` instead.")]
    pub async fn drop(&mut self, columns: &[&str]) -> Result<()> {
        self.drop_columns(columns).await
    }

    async fn merge_impl(
        &mut self,
        stream: Box<dyn RecordBatchReader + Send>,
        left_on: &str,
        right_on: &str,
    ) -> Result<()> {
        // Sanity check.
        if self.schema().field(left_on).is_none() && left_on != ROW_ID && left_on != ROW_ADDR {
            return Err(Error::invalid_input(format!(
                "Column {} does not exist in the left side dataset",
                left_on
            )));
        };
        let right_schema = stream.schema();
        if right_schema.field_with_name(right_on).is_err() {
            return Err(Error::invalid_input(format!(
                "Column {} does not exist in the right side dataset",
                right_on
            )));
        };
        for field in right_schema.fields() {
            if field.name() == right_on {
                // right_on is allowed to exist in the dataset, since it may be
                // the same as left_on.
                continue;
            }
            if self.schema().field(field.name()).is_some() {
                return Err(Error::invalid_input(format!(
                    "Column {} exists in both sides of the dataset",
                    field.name()
                )));
            }
        }

        // Hash join
        let joiner = Arc::new(HashJoiner::try_new(stream, right_on).await?);
        // Final schema is union of current schema, plus the RHS schema without
        // the right_on key.
        let mut new_schema: Schema = self.schema().merge(joiner.out_schema().as_ref())?;
        new_schema.try_set_field_id(Some(self.manifest.max_field_id()))?;

        // Write new data file to each fragment. Parallelism is done over columns,
        // so no parallelism done at this level.
        let updated_fragments: Vec<Fragment> = stream::iter(self.get_fragments())
            .then(|f| {
                let joiner = joiner.clone();
                async move { f.merge(left_on, &joiner).await.map(Fragment::from) }
            })
            .try_collect::<Vec<_>>()
            .await?;

        let preserves_nullability =
            !schema_evolution::merge_introduces_required_field(self.schema(), &new_schema);
        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Merge {
                fragments: updated_fragments,
                schema: new_schema,
                preserves_nullability,
            },
            None,
        );

        self.apply_commit(transaction, &Default::default(), &Default::default())
            .await?;

        Ok(())
    }

    /// Merge this dataset with another arrow Table / Dataset, and returns a new version of dataset.
    ///
    /// Parameters:
    ///
    /// - `stream`: the stream of [`RecordBatch`] to merge.
    /// - `left_on`: the column name to join on the left side (self).
    /// - `right_on`: the column name to join on the right side (stream).
    ///
    /// Returns: a new version of dataset.
    ///
    /// It performs a left-join on the two datasets. The right-side join column must
    /// contain unique keys, including at most one null. Duplicate right-side keys
    /// return an input error before the dataset is modified.
    pub async fn merge(
        &mut self,
        stream: impl RecordBatchReader + Send + 'static,
        left_on: &str,
        right_on: &str,
    ) -> Result<()> {
        let stream = Box::new(stream);
        self.merge_impl(stream, left_on, right_on).await
    }

    /// Merge a distributed scalar index into a single root artifact and report
    /// progress via the supplied callback.
    pub async fn merge_index_metadata(
        &self,
        index_uuid: &Uuid,
        index_type: IndexType,
        _batch_readhead: Option<usize>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<()> {
        let store = LanceIndexStore::from_dataset_for_new(self, index_uuid)?;
        let index_dir = self.indices_dir().join(index_uuid.to_string());
        match index_type {
            IndexType::Inverted => {
                let observed = ObservedBuild::start(
                    progress,
                    ObservedOperation::MergeIndexMetadata,
                    || BuiltinIndexType::Inverted.as_str().to_string(),
                )?;
                lance_index::scalar::inverted::builder::merge_index_files(
                    self.table_state.object_store.as_ref(),
                    &index_dir,
                    Arc::new(store),
                    observed.progress(),
                )
                .await?;
                observed.finish();
                Ok(())
            }
            IndexType::BTree => {
                Err(Error::invalid_input(
                    "BTree distributed indexing no longer supports merge_index_metadata; \
                     build segments, optionally merge groups with merge_existing_index_segments(...), \
                     and commit with commit_existing_index_segments(...)"
                        .to_string(),
                ))
            }
            IndexType::Bitmap => {
                Err(Error::invalid_input(
                    "Bitmap distributed indexing no longer supports merge_index_metadata; \
                     build segments with create_index_uncommitted(...), merge them with \
                     merge_existing_index_segments(...), and commit with \
                     commit_existing_index_segments(...)"
                        .to_string(),
                ))
            }
            IndexType::IvfFlat | IndexType::IvfPq | IndexType::IvfSq | IndexType::Vector => {
                Err(Error::invalid_input(
                    "Vector distributed indexing no longer supports merge_index_metadata; \
                     build segments, optionally merge groups with merge_existing_index_segments(...), \
                     and commit with commit_existing_index_segments(...)"
                        .to_string(),
                ))
            }
            _ => Err(Error::invalid_input_source(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unsupported index type (patched): {}", index_type),
            )))),
        }
    }
}

/// # Dataset metadata APIs
///
/// There are four kinds of metadata on datasets:
///
///  - **Schema metadata**: metadata about the data itself.
///  - **Field metadata**: metadata about the dataset itself.
///  - **Dataset metadata**: metadata about the dataset. For example, this could
///    store a created_at date.
///  - **Dataset config**: configuration values controlling how engines should
///    manage the dataset. This configures things like auto-cleanup.
///
/// You can get
impl Dataset {
    /// Get dataset metadata.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.manifest.table_metadata
    }

    /// Get the dataset config from manifest
    pub fn config(&self) -> &HashMap<String, String> {
        &self.manifest.config
    }

    /// Delete keys from the config.
    #[deprecated(
        note = "Use the new update_config(values, replace) method - pass None values to delete keys"
    )]
    pub async fn delete_config_keys(&mut self, delete_keys: &[&str]) -> Result<()> {
        let updates = delete_keys.iter().map(|key| (*key, None));
        self.update_config(updates).await?;
        Ok(())
    }

    /// Update table metadata.
    ///
    /// Pass `None` for a value to remove that key.
    ///
    /// Use `.replace()` to replace the entire metadata map instead of merging.
    ///
    /// Returns the updated metadata map after the operation.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::transaction::UpdateMapEntry;
    /// # async fn test_update_metadata(dataset: &mut Dataset) -> Result<()> {
    /// // Update single key
    /// dataset.update_metadata([("key", "value")]).await?;
    ///
    /// // Remove a key
    /// dataset.update_metadata([("to_delete", None)]).await?;
    ///
    /// // Clear all metadata
    /// dataset.update_metadata([] as [UpdateMapEntry; 0]).replace().await?;
    ///
    /// // Replace full metadata
    /// dataset.update_metadata([("k1", "v1"), ("k2", "v2")]).replace().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_metadata(
        &mut self,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
    ) -> metadata::UpdateMetadataBuilder<'_> {
        metadata::UpdateMetadataBuilder::new(self, values, metadata::MetadataType::TableMetadata)
    }

    /// Update config.
    ///
    /// Pass `None` for a value to remove that key.
    ///
    /// Use `.replace()` to replace the entire config map instead of merging.
    ///
    /// Returns the updated config map after the operation.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::transaction::UpdateMapEntry;
    /// # async fn test_update_config(dataset: &mut Dataset) -> Result<()> {
    /// // Update single key
    /// dataset.update_config([("key", "value")]).await?;
    ///
    /// // Remove a key
    /// dataset.update_config([("to_delete", None)]).await?;
    ///
    /// // Clear all config
    /// dataset.update_config([] as [UpdateMapEntry; 0]).replace().await?;
    ///
    /// // Replace full config
    /// dataset.update_config([("k1", "v1"), ("k2", "v2")]).replace().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_config(
        &mut self,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
    ) -> metadata::UpdateMetadataBuilder<'_> {
        metadata::UpdateMetadataBuilder::new(self, values, metadata::MetadataType::Config)
    }

    /// Update schema metadata.
    ///
    /// Pass `None` for a value to remove that key.
    ///
    /// Use `.replace()` to replace the entire schema metadata map instead of merging.
    ///
    /// Returns the updated schema metadata map after the operation.
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::transaction::UpdateMapEntry;
    /// # async fn test_update_schema_metadata(dataset: &mut Dataset) -> Result<()> {
    /// // Update single key
    /// dataset.update_schema_metadata([("key", "value")]).await?;
    ///
    /// // Remove a key
    /// dataset.update_schema_metadata([("to_delete", None)]).await?;
    ///
    /// // Clear all schema metadata
    /// dataset.update_schema_metadata([] as [UpdateMapEntry; 0]).replace().await?;
    ///
    /// // Replace full schema metadata
    /// dataset.update_schema_metadata([("k1", "v1"), ("k2", "v2")]).replace().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_schema_metadata(
        &mut self,
        values: impl IntoIterator<Item = impl Into<UpdateMapEntry>>,
    ) -> metadata::UpdateMetadataBuilder<'_> {
        metadata::UpdateMetadataBuilder::new(self, values, metadata::MetadataType::SchemaMetadata)
    }

    /// Update schema metadata
    #[deprecated(note = "Use the new update_schema_metadata(values).replace() instead")]
    pub async fn replace_schema_metadata(
        &mut self,
        new_values: impl IntoIterator<Item = (String, String)>,
    ) -> Result<()> {
        let new_values = new_values
            .into_iter()
            .map(|(k, v)| (k, Some(v)))
            .collect::<HashMap<_, _>>();
        self.update_schema_metadata(new_values).replace().await?;
        Ok(())
    }

    /// Update field metadata
    ///
    /// ```
    /// # use lance::{Dataset, Result};
    /// # use lance::dataset::transaction::UpdateMapEntry;
    /// # async fn test_update_field_metadata(dataset: &mut Dataset) -> Result<()> {
    /// // Update metadata by field path
    /// dataset.update_field_metadata()
    ///     .update("path.to_field", [("key", "value")])?
    ///     .await?;
    ///
    /// // Update metadata by field id
    /// dataset.update_field_metadata()
    ///     .update(12, [("key", "value")])?
    ///     .await?;
    ///
    /// // Clear field metadata
    /// dataset.update_field_metadata()
    ///     .replace("path.to_field", [] as [UpdateMapEntry; 0])?
    ///     .replace(12, [] as [UpdateMapEntry; 0])?
    ///     .await?;
    ///
    /// // Replace field metadata
    /// dataset.update_field_metadata()
    ///     .replace("field_name", [("k1", "v1"), ("k2", "v2")])?
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn update_field_metadata(&mut self) -> UpdateFieldMetadataBuilder<'_> {
        UpdateFieldMetadataBuilder::new(self)
    }

    /// Update field metadata
    pub async fn replace_field_metadata(
        &mut self,
        new_values: impl IntoIterator<Item = (u32, HashMap<String, String>)>,
    ) -> Result<()> {
        let new_values = new_values.into_iter().collect::<HashMap<_, _>>();
        let field_metadata_updates = new_values
            .into_iter()
            .map(|(field_id, metadata)| {
                (
                    field_id as i32,
                    translate_schema_metadata_updates(&metadata),
                )
            })
            .collect();
        metadata::execute_metadata_update(
            self,
            Operation::UpdateConfig {
                config_updates: None,
                table_metadata_updates: None,
                schema_metadata_updates: None,
                field_metadata_updates,
            },
        )
        .await
    }
}

#[async_trait::async_trait]
impl DatasetTakeRows for Dataset {
    fn schema(&self) -> &Schema {
        Self::schema(self)
    }

    async fn take_rows(&self, row_ids: &[u64], projection: &Schema) -> Result<RecordBatch> {
        Self::take_rows(self, row_ids, projection.clone()).await
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn write_manifest_file(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    config: &ManifestBuildConfig,
    naming_scheme: ManifestNamingScheme,
    transaction: Option<lance_table::format::Transaction>,
    may_change_schema: bool,
) -> std::result::Result<ManifestLocation, lance_table::io::commit::CommitError> {
    let hook = crate::io::commit::LanceCommitHook::new(Arc::default());
    lance_table_runtime::commit::write_manifest_file(
        object_store,
        commit_handler,
        base_path,
        manifest,
        indices,
        config,
        naming_scheme,
        transaction,
        may_change_schema,
        hook.hooks(),
    )
    .await
}

impl Projectable for Dataset {
    fn schema(&self) -> &Schema {
        self.schema()
    }
}

/// Marker files that `DirectoryNamespace` writes for a table that is declared or
/// deregistered but was never materialized. Spelled out here because
/// `lance-namespace-impls` depends on this crate, not the other way around.
const NAMESPACE_TABLE_MARKERS: &[&str] = &[".lance-reserved", ".lance-deregistered"];

/// Check that `base` is a Lance dataset root before deleting it recursively.
///
/// Dropping a dataset removes whatever the caller pointed at, so a mistyped or
/// misconfigured URI — a warehouse root, a bucket root, a home directory — destroys
/// unrelated data with no way back. Requiring the target to actually be a dataset
/// turns that class of mistake into an error instead of silent data loss.
///
/// A path qualifies on positive evidence only, which is one of:
///
/// * a file under `_versions/` that both parses as a manifest location and deserializes
///   as a manifest, attached or detached. Every dataset that has ever committed has one,
///   whatever naming scheme or commit handler produced it.
/// * a `DirectoryNamespace` declare or deregister marker, for a table that a namespace
///   reserved but never wrote.
///
/// Nothing weaker qualifies. A non-empty `_versions/` is not evidence, because any file
/// can be put there; neither are data files, which look identical to a storage root whose
/// only prefix happens to be `data/`. Leftovers from a write that never committed,
/// manifests that are corrupt, and the staging manifest an external store writes before
/// it materializes the canonical path therefore need an explicit storage-level delete
/// rather than a weaker default guard here. That costs little: leftovers do not block
/// re-creating the dataset, because creation only refuses a path that already holds a
/// manifest, and [`Dataset::cleanup_old_versions`] removes data files no manifest
/// references.
///
/// Unmanaged files that a user keeps next to a committed dataset do not change the
/// answer, matching the way cleanup leaves them alone. Note that the recursive delete
/// this guards still removes them.
///
/// A missing or empty path also qualifies, so callers keep whatever not-found behavior
/// they have today rather than seeing a new error kind.
///
/// This cannot protect files that another dataset references through `base_paths`;
/// shallow-clone sources still need the reference tracking discussed in
/// [#7514](https://github.com/lance-format/lance/issues/7514).
pub async fn validate_dataset_root_for_drop(object_store: &ObjectStore, base: &Path) -> Result<()> {
    if holds_readable_manifest(object_store, base).await? {
        return Ok(());
    }

    for marker in NAMESPACE_TABLE_MARKERS {
        if object_store.exists(&base.clone().join(*marker)).await? {
            return Ok(());
        }
    }

    // Rejecting a path that holds nothing would replace the not-found error callers
    // already handle, and `ignore_not_found` relies on, with a different error kind.
    if !has_any_entry(object_store, base).await? {
        return Ok(());
    }

    Err(Error::invalid_input(format!(
        "Refusing to drop '{base}': no readable Lance manifest was found under \
         '{VERSIONS_DIR}', so this is not a dataset root. Check that the path points at a \
         dataset and not at a parent directory, and check the logs for manifests that \
         could not be read. A path holding only data files, or only manifests that cannot \
         be read, needs an explicit storage-level delete instead: such leftovers neither \
         block re-creating the dataset nor survive cleanup."
    )))
}

/// Whether `base` holds a manifest that actually deserializes, which is the only proof
/// that a dataset was ever committed here.
///
/// Returns on the first manifest that reads, so a real dataset costs one listing plus one
/// manifest read no matter how many versions it has.
async fn holds_readable_manifest(object_store: &ObjectStore, base: &Path) -> Result<bool> {
    let mut entries = object_store.list(Some(base.clone().join(VERSIONS_DIR)));
    loop {
        let meta = match entries.try_next().await {
            Ok(Some(meta)) => meta,
            Ok(None) => return Ok(false),
            // Local filesystems report a missing directory as an error where object stores
            // return an empty listing. Neither holds a manifest.
            Err(e) if e.is_not_found() => return Ok(false),
            Err(e) => return Err(e),
        };

        if !is_manifest_location(&meta) {
            continue;
        }

        match read_manifest(object_store, &meta.location, Some(meta.size)).await {
            Ok(_) => return Ok(true),
            // A file that only looks like a manifest proves nothing, so keep looking
            // rather than authorizing the delete. The reason is logged because a read
            // that failed for an unrelated cause, such as a transient storage error,
            // otherwise leaves no trace of why the path was refused.
            Err(e) => warn!(
                "Ignoring '{}' while checking whether '{base}' is a dataset root: {e}",
                meta.location
            ),
        }
    }
}

/// Whether `meta` names a manifest, using the same parsing that manifest discovery uses.
fn is_manifest_location(meta: &object_store::ObjectMeta) -> bool {
    if ManifestLocation::try_from(meta.clone()).is_ok() {
        return true;
    }
    meta.location
        .filename()
        .and_then(ManifestNamingScheme::parse_detached_version)
        .is_some()
}

#[cfg(test)]
mod tests;
