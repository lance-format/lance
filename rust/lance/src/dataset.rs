// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Dataset
//!

use arrow_array::{RecordBatch, RecordBatchReader};
use byteorder::{ByteOrder, LittleEndian};
use chrono::{Duration, prelude::*};
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use futures::{FutureExt, Stream};
use lance_core::deepsize::DeepSizeOf;

use crate::dataset::metadata::UpdateFieldMetadataBuilder;
use crate::dataset::transaction::translate_schema_metadata_updates;
use crate::session::caches::{DSMetadataCache, ManifestKey, TransactionKey};
use crate::session::index_caches::DSIndexCache;
use itertools::Itertools;
use lance_core::ROW_ADDR;
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Projectable, Projection};
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
    ChainedWrappingObjectStore, LanceNamespaceStorageOptionsProvider, ObjectStore,
    ObjectStoreParams, StorageOptions, StorageOptionsAccessor, StorageOptionsProvider,
    WrappingObjectStore,
};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::{
    CachedFileSize, read_last_block, read_message, read_metadata_offset, read_struct,
};
use lance_namespace::LanceNamespace;
use lance_table::format::{
    DataFile, DataStorageFormat, DeletionFile, Fragment, IndexMetadata, MAGIC, Manifest,
    ManifestBuildConfig, RowIdMeta, pb, populate_manifest_schema_dictionaries,
};
use lance_table::io::commit::{
    CommitConfig, CommitError, CommitHandler, CommitLock, ManifestLocation, ManifestNamingScheme,
    VERSIONS_DIR, external_manifest::ExternalManifestCommitHandler, migrate_scheme_to_v2,
    write_manifest_file_to_path,
};

use crate::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;
use lance_table::io::manifest::{read_manifest, read_manifest_indexes};
use object_store::ObjectStoreExt;
use object_store::path::Path;
use prost::Message;
use roaring::RoaringBitmap;
use rowids::get_row_id_index;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Debug;
use std::num::NonZero;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{info, instrument, warn};

pub(crate) mod blob;
pub(crate) mod branch_location;
pub mod builder;
pub mod cleanup;
mod data_file;
pub mod delta;
pub mod files;
pub mod fragment;
mod hash_joiner;
pub mod index;
pub mod mem_wal;
mod metadata;
pub mod optimize;
pub(crate) mod overlay;
pub mod progress;
pub mod refs;
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
        RewrittenIndex, Transaction, TransactionBuilder, UpdateMap, UpdateMapEntry, UpdateMode,
        UpdatedFragmentOffsets, translate_config_updates, translate_schema_metadata_updates,
        validate_operation,
    };
}
pub mod udtf;
pub mod updater;
mod utils;
pub(crate) mod versions;
pub mod write;

pub use data_file::{DataFilePart, DataFileTarget};

pub(crate) use take::row_offsets_to_row_addresses;

use self::builder::DatasetBuilder;
use self::cleanup::RemovalStats;
use self::fragment::FileFragment;
use self::refs::Refs;
use self::scanner::{DatasetRecordBatchStream, Scanner};
use self::statistics::DatasetStatistics;
use self::transaction::{Operation, Transaction, TransactionBuilder, UpdateMapEntry};
use self::write::cleanup_data_fragments;
use crate::dataset::branch_location::BranchLocation;
use crate::dataset::cleanup::{CleanupOperation, CleanupPolicy, CleanupPolicyBuilder};
use crate::dataset::refs::{BranchContents, BranchIdentifier, Branches, Tags};
use crate::dataset::sql::SqlQueryBuilder;
use crate::datatypes::Schema;
use crate::io::commit::{
    DEFAULT_COMMIT_RETRY_TIMEOUT, commit_detached_transaction, commit_new_dataset,
    commit_transaction, detect_overlapping_fragments,
};
use crate::session::Session;
use crate::utils::temporal::{SystemTime, timestamp_to_nanos, utc_now};
use crate::{Error, Result};
pub use blob::{
    BlobFile, BlobRangeRequest, BlobReadRange, ReadBlob, ReadBlobRange, ReadBlobRangesBuilder,
    ReadBlobRangesStream, ReadBlobsBuilder, ReadBlobsStream,
};
use hash_joiner::HashJoiner;
pub use lance_core::ROW_ID;
use lance_core::box_error;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_namespace::models::{DeclareTableRequest, DescribeTableRequest};
use lance_table::feature_flags::{
    apply_feature_flags, ensure_can_read_manifest, ensure_can_write_manifest,
    validate_paired_feature_flags,
};
use lance_table::io::deletion::{DELETIONS_DIR, relative_deletion_file_path};
use lance_table::rowids::{RowIdSequence, write_row_ids};
pub use overlay::writer::{OverlayWriter, WriteOverlayError};
pub use schema_evolution::{
    BatchInfo, BatchUDF, ColumnAlteration, NewColumnTransform, UDFCheckpointStore,
};
pub use take::TakeBuilder;
use uuid::Uuid;
pub use write::merge_insert::{
    MergeInsertBuilder, MergeInsertJob, MergeInsertWriteMode, MergeStats, UncommittedMergeInsert,
    WhenMatched, WhenNotMatched, WhenNotMatchedBySource,
};

use crate::dataset::index::LanceIndexStoreExt;
pub use write::update::{UpdateBuilder, UpdateJob};
#[allow(deprecated)]
pub use write::{
    AutoCleanupParams, CommitBuilder, DEFAULT_COMMIT_TIMEOUT, DeleteBuilder, DeleteResult,
    ExternalBlobMode, InsertBuilder, UncommittedDelete, WriteDestination, WriteMode, WriteParams,
    WriteProgressFn, WriteStats, write_fragments,
};

pub(crate) const INDICES_DIR: &str = "_indices";
pub(crate) const DATA_DIR: &str = "data";
pub(crate) const TRANSACTIONS_DIR: &str = "_transactions";
const DEFAULT_MAX_STREAM_COPY_PARALLELISM: usize = 4;

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
pub const DEFAULT_METADATA_CACHE_SIZE: usize = 1024 * 1024 * 1024;

/// Lance Dataset
#[derive(Clone)]
pub struct Dataset {
    /// The primary dataset object store. Use [`Self::object_store`] when
    /// resolving files that may carry a base id.
    pub(crate) object_store: Arc<ObjectStore>,
    pub(crate) commit_handler: Arc<dyn CommitHandler>,
    /// Uri of the dataset.
    ///
    /// On cloud storage, we can not use [Dataset::base] to build the full uri because the
    /// `bucket` is swallowed in the inner [ObjectStore].
    uri: String,
    pub(crate) base: Path,
    pub manifest: Arc<Manifest>,
    // Path for the manifest that is loaded. Used to get additional information,
    // such as the index metadata.
    pub(crate) manifest_location: ManifestLocation,
    pub(crate) session: Arc<Session>,
    pub refs: Refs,

    // Bitmap of fragment ids in this dataset.
    pub(crate) fragment_bitmap: Arc<RoaringBitmap>,

    // These are references to session caches, but with the dataset URI as a prefix.
    pub(crate) index_cache: Arc<DSIndexCache>,
    pub(crate) metadata_cache: Arc<DSMetadataCache>,

    /// File reader options to use when reading data files.
    pub(crate) file_reader_options: Option<FileReaderOptions>,

    /// Object store parameters used when opening this dataset.
    /// These are used when creating object stores for additional base paths.
    pub(crate) store_params: Option<Box<ObjectStoreParams>>,
    /// Optional runtime-only object store parameters keyed by base path URI.
    pub(crate) base_store_params: Option<Arc<HashMap<String, ObjectStoreParams>>>,
    /// Object stores for additional base paths, normally shared across clones.
    /// Applying new object store wrappers starts a fresh cache scope.
    pub(crate) base_object_stores: BaseObjectStores,
}

/// The `OnceCell` coalesces concurrent first resolutions into one build.
pub(crate) type BaseObjectStores =
    Arc<std::sync::Mutex<HashMap<u32, Arc<tokio::sync::OnceCell<Arc<ObjectStore>>>>>>;

impl std::fmt::Debug for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dataset")
            .field("uri", &self.uri)
            .field("base", &self.base)
            .field("version", &self.manifest.version)
            .field("cache_num_items", &self.session.approx_num_items())
            .field("base_store_params", &self.base_store_params.is_some())
            .finish()
    }
}

/// Dataset Version
#[derive(Deserialize, Serialize, Debug)]
pub struct Version {
    /// version number
    pub version: u64,

    /// Timestamp of dataset creation in UTC.
    pub timestamp: DateTime<Utc>,

    /// Key-value pairs of metadata.
    pub metadata: BTreeMap<String, String>,
}

/// A lightweight reference to an attached dataset version, which could be used to uniquely identify a version.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct VersionRef {
    /// Version number within the current branch's history.
    pub version: u64,
}

/// Convert Manifest to Data Version.
impl From<&Manifest> for Version {
    fn from(m: &Manifest) -> Self {
        Self {
            version: m.version,
            timestamp: m.timestamp(),
            metadata: m.summary().into(),
        }
    }
}

/// The transaction that produced a version of the dataset, along with the
/// version's commit timestamp.
///
/// Returned by [`Dataset::read_version_transaction`], which reads this
/// information directly from storage without checking out the version.
#[derive(Debug, Clone)]
pub struct VersionTransaction {
    /// Version number.
    pub version: u64,

    /// Timestamp the version was committed, in UTC.
    pub timestamp: DateTime<Utc>,

    /// The transaction that produced this version, if one was recorded.
    pub transaction: Option<Transaction>,
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

#[derive(Debug, Clone)]
pub enum ProjectionRequest {
    Schema(Arc<Schema>),
    Sql(Vec<(String, String)>),
}

impl ProjectionRequest {
    pub fn from_columns(
        columns: impl IntoIterator<Item = impl AsRef<str>>,
        dataset_schema: &Schema,
    ) -> Self {
        let columns = columns
            .into_iter()
            .map(|s| s.as_ref().to_string())
            .collect::<Vec<_>>();

        let schema = dataset_schema
            .project_preserve_system_columns(&columns)
            .unwrap();
        Self::Schema(Arc::new(schema))
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
    /// Open an existing dataset.
    ///
    /// See also [DatasetBuilder].
    #[instrument]
    pub async fn open(uri: &str) -> Result<Self> {
        DatasetBuilder::from_uri(uri).load().await
    }

    /// Check out a dataset version with a ref
    pub async fn checkout_version(&self, version: impl Into<refs::Ref>) -> Result<Self> {
        let reference: refs::Ref = version.into();
        match reference {
            refs::Ref::Version(branch, version_number) => {
                self.checkout_by_ref(version_number, branch.as_deref())
                    .await
            }
            refs::Ref::VersionNumber(version_number) => {
                self.checkout_by_ref(Some(version_number), self.manifest.branch.as_deref())
                    .await
            }
            refs::Ref::Tag(tag_name) => {
                let tag_contents = self.tags().get(tag_name.as_str()).await?;
                self.checkout_by_ref(Some(tag_contents.version), tag_contents.branch.as_deref())
                    .await
            }
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
        if manifest.base_paths != self.manifest.base_paths {
            self.base_object_stores = Default::default();
        }
        self.manifest = manifest;
        self.manifest_location = manifest_location;
        self.fragment_bitmap = Arc::new(
            self.manifest
                .fragments
                .iter()
                .map(|f| f.id as u32)
                .collect(),
        );
    }

    /// Check out the latest version of the branch
    pub async fn checkout_branch(&self, branch: &str) -> Result<Self> {
        self.checkout_by_ref(None, Some(branch)).await
    }

    /// This is a two-phase operation:
    /// - Create the branch dataset by shallow cloning.
    /// - Create the branch metadata (a.k.a. `BranchContents`).
    ///
    /// These two phases are not atomic. We consider `BranchContents` as the source of truth
    /// for the branch.
    ///
    /// The cleanup procedure should:
    /// - Clean up zombie branch datasets that have no related `BranchContents`.
    /// - Delete broken `BranchContents` entries that have no related branch dataset.
    ///
    /// If `create_branch` stops at phase 1, it may leave a zombie branch dataset,
    /// which can be cleaned up later. Such a zombie dataset may cause a branch creation
    /// failure if we use the same name to `create_branch`. In that case, you need to call
    /// `force_delete_branch` to interactively clean up the zombie dataset.
    pub async fn create_branch(
        &mut self,
        branch: &str,
        version: impl Into<refs::Ref>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        let (source_branch, version_number) = self.resolve_reference(version.into()).await?;
        let branch_location = self.branch_location().find_branch(Some(branch))?;
        let source_location = self
            .branch_location()
            .find_branch(source_branch.as_deref())?;
        let clone_op = Operation::Clone {
            is_shallow: true,
            ref_name: source_branch.clone(),
            ref_version: version_number,
            ref_path: source_location.uri,
            branch_name: Some(branch.to_string()),
        };
        let transaction = Transaction::new(version_number, clone_op, None);

        let builder = CommitBuilder::new(WriteDestination::Uri(branch_location.uri.as_str()))
            // Fall back to the dataset's own store params
            .with_store_params(
                store_params.unwrap_or(self.store_params.as_deref().cloned().unwrap_or_default()),
            )
            .with_object_store(Arc::new(self.object_store.as_ref().clone()))
            .with_commit_handler(self.commit_handler.clone())
            .with_exact_storage_format(self.manifest.data_storage_format.lance_file_format());
        let dataset = builder.execute(transaction).await?;

        // Create BranchContents after shallow_clone
        self.branches()
            .create(branch, version_number, source_branch.as_deref())
            .await?;
        Ok(dataset)
    }

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

    fn already_checked_out(&self, location: &ManifestLocation, branch_name: Option<&str>) -> bool {
        // The ETag is an opaque object-generation token, not a content hash.
        // Comparing the token still prevents reusing this Dataset's manifest
        // after the physical object was replaced, for example by a recent
        // drop/recreate at the same URI and version.
        self.manifest.branch.as_deref() == branch_name
            && self.manifest.version == location.version
            && self.manifest_location.naming_scheme == location.naming_scheme
            && location.e_tag.as_ref().is_some_and(|e_tag| {
                self.manifest_location
                    .e_tag
                    .as_ref()
                    .is_some_and(|current_e_tag| e_tag == current_e_tag)
            })
    }

    async fn checkout_by_ref(
        &self,
        version_number: Option<u64>,
        branch: Option<&str>,
    ) -> Result<Self> {
        // Reject malformed names at the boundary (mirroring the branch CRUD
        // paths) so they fail as InvalidRef instead of tripping the wrong-chain
        // check below
        if let Some(branch_name) = branch
            && !Branches::is_main_branch(branch)
        {
            refs::check_valid_branch(branch_name)?;
        }

        let new_location = self.branch_location().find_branch(branch)?;

        let manifest_location = if let Some(version_number) = version_number {
            self.commit_handler
                .resolve_version_location(
                    &new_location.path,
                    version_number,
                    &self.object_store.inner,
                )
                .await?
        } else {
            self.commit_handler
                .resolve_latest_location(&new_location.path, &self.object_store)
                .await?
        };

        if self.already_checked_out(&manifest_location, branch) {
            return Ok(self.clone());
        }

        let manifest = Self::get_manifest(
            self.object_store.as_ref(),
            &manifest_location,
            &new_location.uri,
            self.session.as_ref(),
        )
        .await?;

        // The resolved manifest must belong to the requested branch. A mismatch
        // means the commit handler resolved against a different chain (for
        // example an external manifest store that ignores branch-qualified
        // paths); error loudly rather than hand back another branch's data.
        let requested_branch = branch.and_then(refs::standardize_branch);
        if manifest.branch.as_deref() != requested_branch.as_deref() {
            return Err(Error::internal(format!(
                "checkout of branch '{}' at version {} resolved a manifest belonging to branch '{}'",
                refs::normalize_branch(branch),
                manifest.version,
                refs::normalize_branch(manifest.branch.as_deref()),
            )));
        }

        Self::checkout_manifest(
            self.object_store.clone(),
            new_location.path,
            new_location.uri,
            manifest,
            manifest_location,
            self.session.clone(),
            self.commit_handler.clone(),
            self.file_reader_options.clone(),
            self.store_params.as_deref().cloned(),
            self.base_store_params.clone(),
        )
    }

    pub(crate) async fn load_manifest(
        object_store: &ObjectStore,
        manifest_location: &ManifestLocation,
        uri: &str,
        session: &Session,
    ) -> Result<Manifest> {
        let object_reader = if let Some(size) = manifest_location.size {
            object_store
                .open_with_size(&manifest_location.path, size as usize)
                .await
        } else {
            object_store.open(&manifest_location.path).await
        };
        let object_reader = object_reader.map_err(|e| match &e {
            Error::NotFound { uri, .. } => Error::dataset_not_found(uri.clone(), box_error(e)),
            _ => e,
        })?;

        let last_block =
            read_last_block(object_reader.as_ref())
                .await
                .map_err(|err| match err {
                    object_store::Error::NotFound { path, source } => {
                        Error::dataset_not_found(path, source)
                    }
                    _ => Error::io_source(err.into()),
                })?;

        // A stale cached size yields a bogus footer offset. Detect it (the block
        // lacks the trailing magic) and retry with the true size, like
        // read_manifest.
        if manifest_location.size.is_some() && !last_block.ends_with(MAGIC) {
            let manifest_location = ManifestLocation {
                size: None,
                ..manifest_location.clone()
            };
            return Box::pin(Self::load_manifest(
                object_store,
                &manifest_location,
                uri,
                session,
            ))
            .await;
        }

        let offset = read_metadata_offset(&last_block)?;

        // If manifest is in the last block, we can decode directly from memory.
        let manifest_size = object_reader.size().await?;
        let mut manifest = if manifest_size - offset <= last_block.len() {
            let manifest_len = manifest_size - offset;
            let offset_in_block = last_block.len() - manifest_len;
            let message_len =
                LittleEndian::read_u32(&last_block[offset_in_block..offset_in_block + 4]) as usize;
            let message_data = &last_block[offset_in_block + 4..offset_in_block + 4 + message_len];
            Manifest::try_from(lance_table::format::pb::Manifest::decode(message_data)?)
        } else {
            read_struct(object_reader.as_ref(), offset).await
        }?;

        ensure_can_read_manifest(&manifest)?;

        // If indices were also in the last block, we can take the opportunity to
        // decode them now and cache them.
        if let Some(index_offset) = manifest.index_section
            && manifest_size - index_offset <= last_block.len()
        {
            let offset_in_block = last_block.len() - (manifest_size - index_offset);
            let message_len =
                LittleEndian::read_u32(&last_block[offset_in_block..offset_in_block + 4]) as usize;
            let message_data = &last_block[offset_in_block + 4..offset_in_block + 4 + message_len];
            let section = lance_table::format::pb::IndexSection::decode(message_data)?;
            // Cached unfiltered: this is the same cache the commit path reads
            // from, and an index this build cannot decode still has to survive
            // into the next manifest. Version filtering happens on the way out,
            // in `DatasetIndexExt::load_indices`.
            let indices: Vec<IndexMetadata> = section
                .indices
                .into_iter()
                .map(IndexMetadata::try_from)
                .collect::<Result<Vec<_>>>()?;
            crate::index::warn_about_unsupported_indices(&indices);
            let ds_index_cache = session.index_cache.for_dataset(uri);
            let metadata_key = crate::session::index_caches::IndexMetadataKey {
                version: manifest_location.version,
                store_identity: &object_store.store_prefix,
                e_tag: manifest_location.e_tag.as_deref(),
            };
            ds_index_cache
                .insert_with_key(&metadata_key, Arc::new(indices))
                .await;
        }

        // If transaction is also in the last block, we can take the opportunity to
        // decode them now and cache them.
        if let Some(transaction_offset) = manifest.transaction_section
            && manifest_size - transaction_offset <= last_block.len()
        {
            let offset_in_block = last_block.len() - (manifest_size - transaction_offset);
            let message_len =
                LittleEndian::read_u32(&last_block[offset_in_block..offset_in_block + 4]) as usize;
            let message_data = &last_block[offset_in_block + 4..offset_in_block + 4 + message_len];
            if let Some(transaction) =
                decode_inline_transaction(message_data, manifest_location.version)
            {
                let metadata_cache = session.metadata_cache.for_dataset(uri);
                let metadata_key = TransactionKey {
                    version: manifest_location.version,
                };
                metadata_cache
                    .insert_with_key(&metadata_key, Arc::new(transaction))
                    .await;
            }
        }

        populate_manifest_schema_dictionaries(&mut manifest, object_reader.as_ref()).await?;

        Ok(manifest)
    }

    /// Fetch the manifest for `manifest_location` from the session metadata
    /// cache, loading and caching it on a miss.
    pub(crate) async fn get_manifest(
        object_store: &ObjectStore,
        manifest_location: &ManifestLocation,
        uri: &str,
        session: &Session,
    ) -> Result<Arc<Manifest>> {
        if manifest_location.size.is_none() {
            return Ok(Arc::new(
                Self::load_manifest(object_store, manifest_location, uri, session).await?,
            ));
        }
        let metadata_cache = session.metadata_cache.for_dataset(uri);
        let manifest_key = ManifestKey {
            version: manifest_location.version,
            e_tag: manifest_location.e_tag.as_deref(),
        };
        if let Some(cached) = metadata_cache.get_with_key(&manifest_key).await {
            ensure_can_read_manifest(&cached)?;
            return Ok(cached);
        }
        let loaded =
            Arc::new(Self::load_manifest(object_store, manifest_location, uri, session).await?);
        metadata_cache
            .insert_with_key(&manifest_key, loaded.clone())
            .await;
        Ok(loaded)
    }

    #[allow(clippy::too_many_arguments)]
    fn checkout_manifest(
        object_store: Arc<ObjectStore>,
        base_path: Path,
        uri: String,
        manifest: Arc<Manifest>,
        manifest_location: ManifestLocation,
        session: Arc<Session>,
        commit_handler: Arc<dyn CommitHandler>,
        file_reader_options: Option<FileReaderOptions>,
        store_params: Option<ObjectStoreParams>,
        base_store_params: Option<Arc<HashMap<String, ObjectStoreParams>>>,
    ) -> Result<Self> {
        let refs = Refs::new(
            object_store.clone(),
            commit_handler.clone(),
            BranchLocation {
                path: base_path.clone(),
                uri: uri.clone(),
                branch: manifest.branch.clone(),
            },
        );
        let metadata_cache = Arc::new(session.metadata_cache.for_dataset(&uri));
        let index_cache = Arc::new(session.index_cache.for_dataset(&uri));
        let fragment_bitmap = Arc::new(manifest.fragments.iter().map(|f| f.id as u32).collect());
        write::log_unregistered_base_scoped_options(
            store_params.as_ref(),
            &manifest.base_paths,
            log::Level::Debug,
        );
        Ok(Self {
            object_store,
            base: base_path,
            uri,
            manifest,
            manifest_location,
            commit_handler,
            session,
            refs,
            fragment_bitmap,
            metadata_cache,
            index_cache,
            file_reader_options,
            store_params: store_params.map(Box::new),
            base_store_params,
            base_object_stores: Default::default(),
        })
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
        &self.uri
    }

    pub fn branch_location(&self) -> BranchLocation {
        BranchLocation {
            path: self.base.clone(),
            uri: self.uri.clone(),
            branch: self.manifest.branch.clone(),
        }
    }

    pub async fn branch_identifier(&self) -> Result<BranchIdentifier> {
        self.refs
            .branches()
            .get_identifier(self.manifest.branch.as_deref())
            .await
    }

    /// Get the full manifest of the dataset version.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn manifest_location(&self) -> &ManifestLocation {
        &self.manifest_location
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
        let location = self
            .commit_handler
            .resolve_latest_location(&self.base, &self.object_store)
            .await?;

        if self.already_checked_out(&location, self.manifest.branch.as_deref()) {
            ensure_can_read_manifest(&self.manifest)?;
            return Ok((self.manifest.clone(), self.manifest_location.clone()));
        }
        let manifest =
            Self::get_manifest(&self.object_store, &location, &self.uri, &self.session).await?;
        Ok((manifest, location))
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    pub async fn read_transaction(&self) -> Result<Option<Transaction>> {
        let transaction_key = TransactionKey {
            version: self.manifest.version,
        };
        if let Some(transaction) = self.metadata_cache.get_with_key(&transaction_key).await {
            return Ok(Some((*transaction).clone()));
        }

        let transaction = self
            .read_transaction_from_storage(&self.manifest, &self.manifest_location)
            .await?;

        if let Some(tx) = transaction.as_ref() {
            self.metadata_cache
                .insert_with_key(&transaction_key, Arc::new(tx.clone()))
                .await;
        }
        Ok(transaction)
    }

    /// Read the transaction recorded by `manifest` directly from storage,
    /// without consulting or populating any session cache.
    async fn read_transaction_from_storage(
        &self,
        manifest: &Manifest,
        manifest_location: &ManifestLocation,
    ) -> Result<Option<Transaction>> {
        // Prefer inline transaction from manifest when available
        if let Some(pos) = manifest.transaction_section {
            let reader = match manifest_location.size {
                Some(size) => {
                    self.object_store
                        .open_with_size(&manifest_location.path, size as usize)
                        .await?
                }
                None => self.object_store.open(&manifest_location.path).await?,
            };

            // A concurrent overwrite can leave the listed size too small; retry
            // once with the true size.
            let tx: pb::Transaction = match read_message(reader.as_ref(), pos).await {
                Err(e)
                    if manifest_location.size.is_some()
                        && e.to_string().contains("file size is too small") =>
                {
                    let reader = self.object_store.open(&manifest_location.path).await?;
                    read_message(reader.as_ref(), pos).await?
                }
                other => other?,
            };
            Transaction::try_from(tx).map(Some)
        } else if let Some(path) = &manifest.transaction_file {
            // Fallback: read external transaction file if present
            let path = self.transactions_dir().join(path.as_str());
            let data = self.object_store.inner.get(&path).await?.bytes().await?;
            let transaction = lance_table::format::pb::Transaction::decode(data)?;
            Transaction::try_from(transaction).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Read the transaction (if any) and commit timestamp of a version of the
    /// dataset. `version` is a version number on this dataset's current branch.
    ///
    /// Reads the version's manifest transiently: no historical `Dataset` is
    /// constructed, no `IndexSection` is decoded, and no session cache is read
    /// or written, so scanning many historical versions does not fill the
    /// shared caches.
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
        // Resolve against this dataset's current branch.
        let manifest_location = self
            .commit_handler
            .resolve_version_location(&self.base, version, &self.object_store.inner)
            .await?;

        // Keep the DatasetNotFound variant callers expect for a missing version.
        let manifest = read_manifest(
            &self.object_store,
            &manifest_location.path,
            manifest_location.size,
        )
        .await
        .map_err(|e| match &e {
            Error::NotFound { uri, .. } => Error::dataset_not_found(uri.clone(), box_error(e)),
            _ => e,
        })?;

        // The resolved manifest must belong to this dataset's branch. A
        // mismatch means the commit handler resolved against a different chain
        // (for example an external manifest store that ignores
        // branch-qualified paths); error loudly rather than hand back another
        // branch's transaction.
        if manifest.branch != self.manifest.branch {
            return Err(Error::internal(format!(
                "reading version {} on branch '{}' resolved a manifest belonging to branch '{}'",
                version,
                refs::normalize_branch(self.manifest.branch.as_deref()),
                refs::normalize_branch(manifest.branch.as_deref()),
            )));
        }

        let transaction = self
            .read_transaction_from_storage(&manifest, &manifest_location)
            .await?;

        Ok(VersionTransaction {
            version: manifest.version,
            timestamp: manifest.timestamp(),
            transaction,
        })
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
        Ok(self.read_version_transaction(version).await?.transaction)
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
        let mut transactions = vec![];
        let mut dataset = self.clone();

        loop {
            let transaction = dataset.read_transaction().await?;
            transactions.push(transaction);

            if transactions.len() >= recent_transactions {
                break;
            } else {
                match dataset
                    .checkout_version(dataset.version().version - 1)
                    .await
                {
                    Ok(ds) => dataset = ds,
                    Err(Error::DatasetNotFound { .. }) => break,
                    Err(err) => return Err(err),
                }
            }
        }

        Ok(transactions)
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

    /// Removes old versions of the dataset from disk
    ///
    /// This function will remove all versions of the dataset that are older than the provided
    /// timestamp.  This function will not remove the current version of the dataset.
    ///
    /// Once a version is removed it can no longer be checked out or restored.  Any data unique
    /// to that version will be lost.
    ///
    /// # Arguments
    ///
    /// * `older_than` - Versions older than this will be deleted.
    /// * `delete_unverified` - If false (the default) then files will only be deleted if they
    ///                        are listed in at least one manifest.  Otherwise these files will
    ///                        be kept since they cannot be distinguished from an in-progress
    ///                        transaction.  Set to true to delete these files if you are sure
    ///                        there are no other in-progress dataset operations.
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
        write_config: &ManifestWriteConfig,
        commit_config: &CommitConfig,
    ) -> Result<()> {
        let (manifest, manifest_location) = commit_transaction(
            self,
            self.object_store.as_ref(),
            self.commit_handler.as_ref(),
            &transaction,
            write_config,
            commit_config,
            DEFAULT_COMMIT_RETRY_TIMEOUT,
            self.manifest_location.naming_scheme,
            None,
        )
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
        let cnts = stream::iter(self.get_fragments())
            .map(|f| async move { f.count_rows(None).await })
            .buffer_unordered(16)
            .try_collect::<Vec<_>>()
            .await?;
        Ok(cnts.iter().sum())
    }

    /// Take rows by indices.
    #[instrument(skip_all, fields(num_rows=row_indices.len()))]
    pub async fn take(
        &self,
        row_indices: &[u64],
        projection: impl Into<ProjectionRequest>,
    ) -> Result<RecordBatch> {
        take::take(self, row_indices, projection.into()).await
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
        blob::take_blobs(self, row_ids, column.as_ref()).await
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
        blob::take_blobs_by_addresses(self, row_addrs, column.as_ref()).await
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
        let fragments = self.get_fragments();
        let row_addrs = row_offsets_to_row_addresses(&fragments, row_indices).await?;
        blob::take_blobs_by_addresses(self, &row_addrs, column.as_ref()).await
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
        let column = column.as_ref();
        let blob_field_id = blob::validate_blob_column(self, column)?;
        Ok(ReadBlobsBuilder::new(
            self.clone(),
            column.to_string(),
            blob_field_id,
        ))
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
        Ok(ReadBlobRangesBuilder::new(self.read_blobs(column)?))
    }

    /// Get a stream of batches based on iterator of ranges of row numbers.
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

    /// Randomly sample `n` rows from the dataset.
    ///
    /// If `fragment_ids` is provided, sampling is limited to rows from those
    /// fragments in the current dataset version.
    ///
    /// The returned rows are in row-id order (not random order), which allows
    /// the underlying take operation to use an efficient sorted code path.
    pub async fn sample(
        &self,
        n: usize,
        projection: &Schema,
        fragment_ids: Option<&[u32]>,
    ) -> Result<RecordBatch> {
        use rand::seq::IteratorRandom;

        match fragment_ids {
            None => {
                let num_rows = self.count_rows(None).await?;
                let mut ids = (0..num_rows as u64).choose_multiple(&mut rand::rng(), n);
                ids.sort_unstable();
                self.take(&ids, projection.clone()).await
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
                    .try_fold(0_u64, |acc, rows| async move { Ok(acc + rows as u64) })
                    .await?;

                let mut offsets = (0..num_rows).choose_multiple(&mut rand::rng(), n);
                offsets.sort_unstable();

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
        info!(target: TRACE_DATASET_EVENTS, event=DATASET_DELETING_EVENT, uri = &self.uri, predicate=predicate);
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
            .buffer_unordered(self.object_store.io_parallelism())
            .try_fold(0, |acc, x| futures::future::ready(Ok(acc + x)))
            .await
    }

    /// Clone this dataset with a different object store binding.
    ///
    /// The returned dataset shares metadata, session state, and caches with the
    /// original dataset, but all subsequent operations on the returned dataset
    /// use the supplied object store.
    pub fn with_object_store(
        &self,
        object_store: Arc<ObjectStore>,
        store_params: Option<ObjectStoreParams>,
    ) -> Self {
        let mut cloned = self.clone();
        cloned.object_store = object_store;
        cloned.base_object_stores = Default::default();
        if let Some(store_params) = store_params {
            cloned.store_params = Some(Box::new(store_params));
        }
        cloned
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
        let wrappers = wrappers.into_iter().collect::<Vec<_>>();
        if wrappers.is_empty() {
            return self.clone();
        }

        let mut cloned = self.clone();
        // Each wrapper application defines a new store lifetime. Keep base
        // stores alive within the derived dataset without sharing stateful
        // provider layers (such as an AIMD throttle) with other scopes.
        cloned.base_object_stores = Default::default();
        let mut object_store = self.object_store.as_ref().clone();
        for wrapper in &wrappers {
            object_store.apply_wrapper(wrapper.as_ref());
        }
        cloned.object_store = Arc::new(object_store);
        cloned.refs = Refs::new(
            cloned.object_store.clone(),
            cloned.commit_handler.clone(),
            cloned.branch_location(),
        );

        let store_params = self.store_params.as_deref().cloned().unwrap_or_default();
        cloned.store_params = Some(Box::new(Self::append_object_store_wrappers(
            store_params,
            &wrappers,
        )));
        cloned.base_store_params = self.base_store_params.as_ref().map(|base_store_params| {
            Arc::new(
                base_store_params
                    .iter()
                    .map(|(base_path, store_params)| {
                        (
                            base_path.clone(),
                            Self::append_object_store_wrappers(store_params.clone(), &wrappers),
                        )
                    })
                    .collect(),
            )
        });
        cloned
    }

    fn append_object_store_wrappers(
        mut store_params: ObjectStoreParams,
        wrappers: &[Arc<dyn WrappingObjectStore>],
    ) -> ObjectStoreParams {
        let mut all_wrappers = Vec::with_capacity(
            store_params.object_store_wrapper.as_ref().map_or(0, |_| 1) + wrappers.len(),
        );
        if let Some(wrapper) = store_params.object_store_wrapper.take() {
            all_wrappers.push(wrapper);
        }
        all_wrappers.extend(wrappers.iter().cloned());
        store_params.object_store_wrapper = match all_wrappers.len() {
            0 => None,
            1 => all_wrappers.pop(),
            _ => Some(Arc::new(ChainedWrappingObjectStore::new(all_wrappers))),
        };
        store_params
    }

    pub(crate) fn store_params_for_base(
        &self,
        base_path: Option<&lance_table::format::BasePath>,
    ) -> ObjectStoreParams {
        // Base-specific bindings are exact ObjectStoreParams keyed by
        // `BasePath.path` and are used as-is. Otherwise the dataset-level
        // default params are resolved for the base scope: `base_<id>.<key>`
        // storage options overlay the shared defaults for that base.
        if let Some(params) = base_path.and_then(|base_path| {
            self.base_store_params
                .as_ref()
                .and_then(|params| params.get(&base_path.path))
        }) {
            return params.clone();
        }
        let default_params = self.store_params.as_deref().cloned().unwrap_or_default();
        match default_params.scoped_to_base(base_path.map(|base_path| base_path.id)) {
            Cow::Owned(scoped_params) => scoped_params,
            Cow::Borrowed(_) => default_params,
        }
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
        self.store_params
            .as_ref()
            .and_then(|params| params.storage_options())
    }

    /// Returns the storage options provider used when opening this dataset, if any.
    pub fn storage_options_provider(
        &self,
    ) -> Option<Arc<dyn lance_io::object_store::StorageOptionsProvider>> {
        self.store_params
            .as_ref()
            .and_then(|params| params.storage_options_accessor.as_ref())
            .and_then(|accessor| accessor.provider().cloned())
    }

    /// Returns the unified storage options accessor for this dataset, if any.
    ///
    /// The accessor handles both static and dynamic storage options with automatic
    /// caching and refresh. Use [`StorageOptionsAccessor::get_storage_options`] to
    /// get the latest options.
    pub fn storage_options_accessor(&self) -> Option<Arc<StorageOptionsAccessor>> {
        self.store_params
            .as_ref()
            .and_then(|params| params.get_accessor())
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
        // First check if we have an accessor (handles both static and dynamic options)
        if let Some(accessor) = self.storage_options_accessor() {
            let options = accessor.get_storage_options().await?;
            return Ok(Some(options));
        }

        // Fallback to initial storage options if no accessor
        Ok(self.initial_storage_options().cloned().map(StorageOptions))
    }

    pub fn data_dir(&self) -> Path {
        self.base.clone().join(DATA_DIR)
    }

    pub fn indices_dir(&self) -> Path {
        self.base.clone().join(INDICES_DIR)
    }

    pub fn transactions_dir(&self) -> Path {
        self.base.clone().join(TRANSACTIONS_DIR)
    }

    pub fn deletions_dir(&self) -> Path {
        self.base.clone().join(DELETIONS_DIR)
    }

    pub fn versions_dir(&self) -> Path {
        self.base.clone().join(VERSIONS_DIR)
    }

    pub(crate) fn data_file_dir(&self, data_file: &DataFile) -> Result<Path> {
        self.data_file_dir_for_base(data_file.base_id)
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
            if dataset_field.name != file_field.name {
                return Err(Error::invalid_input(format!(
                    "Schema mismatch: expected field '{}' but file has '{}'",
                    path, file_field.name
                )));
            }

            if dataset_field.is_blob() && file_field.is_blob() {
                return Ok(());
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
        match base_id {
            Some(base_id) => {
                let base_path = self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                    Error::invalid_input(format!("base_path id {} not found", base_id))
                })?;
                let path = base_path.extract_path(self.session.store_registry())?;
                if base_path.is_dataset_root {
                    Ok(path.join(DATA_DIR))
                } else {
                    Ok(path)
                }
            }
            None => Ok(self.base.clone().join(DATA_DIR)),
        }
    }

    async fn base_object_store(&self, base_id: u32) -> Result<Arc<ObjectStore>> {
        let base_path = self.manifest.base_paths.get(&base_id).ok_or_else(|| {
            Error::invalid_input(format!("Dataset base path with ID {} not found", base_id))
        })?;
        let store_params = self.store_params_for_base(Some(base_path));

        let cell = {
            let mut stores = self.base_object_stores.lock().unwrap();
            stores.entry(base_id).or_default().clone()
        };
        let store = cell
            .get_or_try_init(|| async {
                // Wrappers define a request or execution scope. Keep the
                // fully resolved store in this dataset's OnceCell, but do not
                // also put it in the global registry: provider-local state
                // such as GCS AIMD token buckets must not cross that scope.
                let (store, _) = if store_params.object_store_wrapper.is_some() {
                    ObjectStore::from_uri_and_params_uncached(
                        self.session.store_registry(),
                        &base_path.path,
                        &store_params,
                    )
                    .await?
                } else {
                    ObjectStore::from_uri_and_params(
                        self.session.store_registry(),
                        &base_path.path,
                        &store_params,
                    )
                    .await?
                };
                Ok::<_, Error>(store)
            })
            .await?;
        Ok(store.clone())
    }

    /// Resolve the object store for the primary dataset or an additional base.
    ///
    /// Pass `None` to get the primary dataset object store. Pass `Some(base_id)`
    /// when resolving a file whose metadata references an additional base.
    pub async fn object_store(&self, base_id: Option<u32>) -> Result<Arc<ObjectStore>> {
        match base_id {
            Some(base_id) => self.base_object_store(base_id).await,
            None => Ok(self.object_store.clone()),
        }
    }

    /// The `ObjectStoreParams` this dataset was opened with, or `None` when
    /// opened without explicit params. Lets a caller re-open a derived path
    /// (e.g. a MemWAL SSTable) with the same store this dataset used.
    pub fn store_params(&self) -> Option<&ObjectStoreParams> {
        self.store_params.as_deref()
    }

    pub(crate) async fn object_store_for_data_file(
        &self,
        data_file: &DataFile,
    ) -> Result<Arc<ObjectStore>> {
        self.object_store(data_file.base_id).await
    }

    pub(crate) async fn object_store_for_deletion(
        &self,
        deletion_file: &DeletionFile,
    ) -> Result<Arc<ObjectStore>> {
        self.object_store(deletion_file.base_id).await
    }

    pub(crate) async fn object_store_for_index(
        &self,
        index: &IndexMetadata,
    ) -> Result<Arc<ObjectStore>> {
        self.object_store(index.base_id).await
    }

    pub(crate) fn dataset_dir_for_deletion(&self, deletion_file: &DeletionFile) -> Result<Path> {
        match deletion_file.base_id.as_ref() {
            Some(base_id) => {
                let base_paths = &self.manifest.base_paths;
                let base_path = base_paths.get(base_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "base_path id {} not found for deletion_file {:?}",
                        base_id, deletion_file
                    ))
                })?;

                if !base_path.is_dataset_root {
                    return Err(Error::internal(format!(
                        "base_path id {} is not a dataset root for deletion_file {:?}",
                        base_id, deletion_file
                    )));
                }
                base_path.extract_path(self.session.store_registry())
            }
            None => Ok(self.base.clone()),
        }
    }

    /// Get the indices directory for a specific index, considering its base_id
    pub(crate) fn indice_files_dir(&self, index: &IndexMetadata) -> Result<Path> {
        match index.base_id.as_ref() {
            Some(base_id) => {
                let base_paths = &self.manifest.base_paths;
                let base_path = base_paths.get(base_id).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "base_path id {} not found for index {}",
                        base_id, index.uuid
                    ))
                })?;
                let path = base_path.extract_path(self.session.store_registry())?;
                if base_path.is_dataset_root {
                    Ok(path.join(INDICES_DIR))
                } else {
                    // For non-dataset-root base paths, we assume the path already points to the indices directory
                    Ok(path)
                }
            }
            None => Ok(self.base.clone().join(INDICES_DIR)),
        }
    }

    pub fn session(&self) -> Arc<Session> {
        self.session.clone()
    }

    /// Get the currently checked-out version id.
    ///
    /// This is a cheap accessor that reads the id directly from the loaded
    /// manifest without constructing the full [Version] summary.
    pub fn version_id(&self) -> u64 {
        self.manifest.version
    }

    /// Get the currently checked-out version details.
    ///
    /// This constructs a full [Version], including summary metadata derived
    /// from the loaded manifest fragments.
    pub fn version(&self) -> Version {
        Version::from(self.manifest.as_ref())
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

    /// Get all versions.
    pub async fn versions(&self) -> Result<Vec<Version>> {
        let mut versions: Vec<Version> = self
            .commit_handler
            .list_manifest_locations(&self.base, &self.object_store, false)
            .try_filter_map(|location| async move {
                match read_manifest(&self.object_store, &location.path, location.size).await {
                    Ok(manifest) => Ok(Some(Version::from(&manifest))),
                    Err(e) => Err(e),
                }
            })
            .try_collect()
            .await?;

        // TODO: this API should support pagination
        versions.sort_by_key(|v| v.version);

        Ok(versions)
    }

    /// Get the number of versions in the current version history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest.
    pub async fn count_versions(&self) -> Result<u64> {
        self.commit_handler
            .list_manifest_locations(&self.base, &self.object_store, false)
            .try_fold(0_u64, |count, _| async move { Ok(count + 1) })
            .await
    }

    /// List lightweight references to all attached versions in the current branch's history.
    ///
    /// Unlike [`Self::versions`], this only enumerates manifest locations and does not read or
    /// deserialize every manifest. The references are sorted by version in ascending order.
    /// Detached manifests are excluded; see [`Self::list_detached_manifests`].
    ///
    /// Use [`Self::latest_version_id`] instead when only the latest version is needed.
    pub async fn version_refs(&self) -> Result<Vec<VersionRef>> {
        let mut versions: Vec<_> = self
            .commit_handler
            .list_manifest_locations(&self.base, &self.object_store, false)
            .map_ok(|location| VersionRef {
                version: location.version,
            })
            .try_collect()
            .await?;
        versions.sort_unstable_by_key(|version| version.version);
        Ok(versions)
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
        self.commit_handler
            .list_detached_manifest_locations(&self.base, &self.object_store)
            .try_collect()
            .await
    }

    /// Get the latest version of the dataset
    /// This is meant to be a fast path for checking if a dataset has changed. This is why
    /// we don't return the full version struct.
    pub async fn latest_version_id(&self) -> Result<u64> {
        Ok(self
            .commit_handler
            .resolve_latest_location(&self.base, &self.object_store)
            .await?
            .version)
    }

    /// Return whether the dataset has a newer committed version.
    pub async fn is_stale(&self) -> Result<bool> {
        let latest_version = self.latest_version_id().await?;
        Ok(latest_version != self.manifest.version)
    }

    /// Return whether the immediate attached successor manifest exists.
    ///
    /// This is a fast contiguous-history probe. It does not resolve the latest
    /// version and may return `false` if intermediate manifests have been
    /// removed. Callers that need a general freshness check should use
    /// [`Self::is_stale`].
    #[doc(hidden)]
    pub async fn has_successor_version(&self) -> Result<bool> {
        let Some(next_version) = self.manifest.version.checked_add(1) else {
            return Ok(false);
        };
        if lance_table::format::is_detached_version(next_version) {
            return Ok(false);
        }

        let exists = self
            .commit_handler
            .version_exists(
                &self.base,
                next_version,
                self.object_store.inner.as_ref(),
                self.manifest_location.naming_scheme,
            )
            .await?;
        Ok(exists)
    }

    pub fn count_fragments(&self) -> usize {
        self.manifest.fragments.len()
    }

    /// Get the schema of the dataset
    pub fn schema(&self) -> &Schema {
        &self.manifest.schema
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
        self.manifest
            .fragments
            .iter()
            .map(|f| FileFragment::new(dataset.clone(), f.clone()))
            .collect()
    }

    /// Iterate over manifest fragments without allocating [`FileFragment`] wrappers.
    pub fn iter_fragments(&self) -> impl Iterator<Item = &Fragment> {
        self.manifest.fragments.iter()
    }

    pub fn get_fragment(&self, fragment_id: usize) -> Option<FileFragment> {
        let metadata = self.find_fragment(fragment_id as u64)?.clone();
        Some(FileFragment::new(Arc::new(self.clone()), metadata))
    }

    pub fn fragments(&self) -> &Arc<Vec<Fragment>> {
        &self.manifest.fragments
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
        ordered_ids
            .iter()
            .map(|id| {
                if !self.fragment_bitmap.contains(*id) {
                    return None;
                }
                let fragment_index = self.fragment_bitmap.rank(*id) as usize - 1;
                let fragment = self.manifest.fragments.get(fragment_index)?;
                debug_assert_eq!(
                    fragment.id, *id as u64,
                    "fragment_bitmap rank({id}) resolved to fragment {}, but fragment_bitmap and manifest.fragments are expected to stay in sync",
                    fragment.id
                );
                Some(FileFragment::new(dataset.clone(), fragment.clone()))
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
    fn find_fragment(&self, id: u64) -> Option<&Fragment> {
        if !u32::try_from(id).is_ok_and(|id| self.fragment_bitmap.contains(id)) {
            return None;
        }
        let fragments = self.manifest.fragments.as_slice();
        let index = fragments.partition_point(|fragment| fragment.id < id);
        match fragments.get(index) {
            Some(fragment) if fragment.id == id => Some(fragment),
            _ => fragments.iter().find(|fragment| fragment.id == id),
        }
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
            .buffered(self.object_store.io_parallelism())
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
            .buffered(self.object_store.io_parallelism())
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
                    self.base.clone(),
                    format!(
                        "Duplicate fragment id {} found in dataset {:?}",
                        id, self.base
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
                    Err(Error::corrupt_file(self.base.clone(), format!(
                        "Fragment ids are not sorted in increasing fragment-id order. Found {} after {} in dataset {:?}",
                        id, prev, self.base
                    )))
                } else {
                    Ok(id)
                }
            })?;

        // All fragments have equal lengths
        futures::stream::iter(self.get_fragments())
            .map(|f| async move { f.validate().await })
            .buffer_unordered(self.object_store.io_parallelism())
            .try_collect::<Vec<()>>()
            .await?;

        rowids::validate_stable_row_ids(self).await?;

        // Validate indices. Over the complete list: these checks are about what
        // the manifest says, not about what this build can use, and duplicate
        // uuids or overlapping coverage are no less corrupt for involving an
        // index this build has no reader for. `migrate_indices` already runs the
        // same overlap check over the complete list on every commit.
        let indices = crate::index::load_all_indices(self).await?;
        self.validate_indices(&indices)?;

        Ok(())
    }

    fn validate_indices(&self, indices: &[IndexMetadata]) -> Result<()> {
        // Make sure there are no duplicate ids
        let mut index_ids = HashSet::new();
        for index in indices.iter() {
            if !index_ids.insert(&index.uuid) {
                return Err(Error::corrupt_file(
                    self.manifest_location.path.clone(),
                    format!(
                        "Duplicate index id {} found in dataset {:?}",
                        index.uuid, self.base
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
                self.manifest_location.path.clone(),
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
        migrate_scheme_to_v2(self.object_store.as_ref(), &self.base).await?;
        // We need to re-open.
        let latest_version = self.latest_version_id().await?;
        *self = self.checkout_version(latest_version).await?;
        Ok(())
    }

    /// Assign stable row ID sequences to fragments that do not yet have them,
    /// contiguously from `start`, and return the resulting `next_row_id`
    /// high-water mark.
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
    /// Stable row IDs assign a persistent identifier to each row that remains
    /// stable across compaction operations. This enables more efficient updates
    /// to secondary indices.
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

    /// Shallow clone the target version into a new dataset at target_path.
    /// 'target_path': the uri string to clone the dataset into.
    /// 'version': the version cloned from, could be a version number or tag.
    /// 'store_params': the object store params to use for the new dataset.
    pub async fn shallow_clone(
        &mut self,
        target_path: &str,
        version: impl Into<refs::Ref>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        let (ref_name, version_number) = self.resolve_reference(version.into()).await?;
        let source_location = self.branch_location().find_branch(ref_name.as_deref())?;
        let clone_op = Operation::Clone {
            is_shallow: true,
            ref_name,
            ref_version: version_number,
            ref_path: source_location.uri,
            branch_name: None,
        };
        let transaction = Transaction::new(version_number, clone_op, None);

        let builder = CommitBuilder::new(WriteDestination::Uri(target_path))
            .with_store_params(
                store_params.unwrap_or(self.store_params.as_deref().cloned().unwrap_or_default()),
            )
            .with_object_store(Arc::new(self.object_store.as_ref().clone()))
            .with_commit_handler(self.commit_handler.clone())
            .with_exact_storage_format(self.manifest.data_storage_format.lance_file_format());
        builder.execute(transaction).await
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
        let src_paths = src_ds.collect_paths().await?;

        // Prepare target object store and base path
        let (target_store, target_base) = ObjectStore::from_uri_and_params(
            self.session.store_registry(),
            target_path,
            &store_params.clone().unwrap_or_default(),
        )
        .await?;

        // Prevent cloning into an existing target dataset
        if self
            .commit_handler
            .resolve_latest_location(&target_base, &target_store)
            .await
            .is_ok()
        {
            return Err(Error::dataset_already_exists(target_path.to_string()));
        }

        let build_absolute_path = |relative_path: &str, base: &Path| -> Path {
            let mut path = base.clone();
            for seg in relative_path.split('/') {
                if !seg.is_empty() {
                    path = path.clone().join(seg);
                }
            }
            path
        };

        let configured_io_parallelism = src_ds.object_store.io_parallelism();
        // Provider-native copy can fall back to streaming for large objects, so every
        // non-direct-local transfer stays within the bounded file-copy window.
        let uses_streaming_copy = !(src_ds.object_store.has_direct_local_paths()
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
                let source_store = Arc::clone(&src_ds.object_store);
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
        let ref_version = src_ds.manifest_location.version;
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
            .with_source_store(src_ds.object_store.clone())
            .with_commit_handler(self.commit_handler.clone())
            .with_exact_storage_format(self.manifest.data_storage_format.lance_file_format());
        let new_ds = builder.execute(txn).await?;
        Ok(new_ds)
    }

    async fn resolve_reference(&self, reference: refs::Ref) -> Result<(Option<String>, u64)> {
        match reference {
            refs::Ref::Version(branch, version_number) => {
                if let Some(version_number) = version_number {
                    Ok((branch, version_number))
                } else {
                    let branch_location = self.branch_location().find_branch(branch.as_deref())?;
                    let version_number = self
                        .commit_handler
                        .resolve_latest_location(&branch_location.path, &self.object_store)
                        .await?
                        .version;
                    Ok((branch, version_number))
                }
            }
            refs::Ref::VersionNumber(version_number) => {
                Ok((self.manifest.branch.clone(), version_number))
            }
            refs::Ref::Tag(tag_name) => {
                let tag_contents = self.tags().get(tag_name.as_str()).await?;
                Ok((tag_contents.branch, tag_contents.version))
            }
        }
    }

    /// Collect all (relative_path, path) of the dataset files.
    async fn collect_paths(&self) -> Result<Vec<(String, Path)>> {
        let mut file_paths: Vec<(String, Path)> = Vec::new();
        for fragment in self.manifest.fragments.iter() {
            if let Some(RowIdMeta::External(external_file)) = &fragment.row_id_meta {
                return Err(Error::internal(format!(
                    "External row_id_meta is not supported yet. external file path: {}",
                    external_file.path
                )));
            }
            for data_file in fragment.referenced_lance_files() {
                let base_root = if let Some(base_id) = data_file.base_id {
                    let base_path =
                        self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                            Error::internal(format!("base_id {} not found", base_id))
                        })?;
                    Path::parse(base_path.path.as_str())?
                } else {
                    self.base.clone()
                };
                file_paths.push((
                    format!("{}/{}", DATA_DIR, data_file.path.clone()),
                    base_root,
                ));
            }
            if let Some(deletion_file) = &fragment.deletion_file {
                let base_root = if let Some(base_id) = deletion_file.base_id {
                    let base_path =
                        self.manifest.base_paths.get(&base_id).ok_or_else(|| {
                            Error::internal(format!("base_id {} not found", base_id))
                        })?;
                    Path::parse(base_path.path.as_str())?
                } else {
                    self.base.clone()
                };
                file_paths.push((
                    relative_deletion_file_path(fragment.id, deletion_file),
                    base_root,
                ));
            }
        }

        let indices = read_manifest_indexes(
            self.object_store.as_ref(),
            &self.manifest_location,
            &self.manifest,
        )
        .await?;

        for index in &indices {
            let base_root = if let Some(base_id) = index.base_id {
                let base_path = self
                    .manifest
                    .base_paths
                    .get(&base_id)
                    .ok_or_else(|| Error::internal(format!("base_id {} not found", base_id)))?;
                Path::parse(base_path.path.as_str())?
            } else {
                self.base.clone()
            };
            let index_root = base_root
                .clone()
                .join(INDICES_DIR)
                .join(index.uuid.to_string());
            let mut stream = self.object_store.read_dir_all(&index_root, None);
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

pub(crate) struct NewTransactionResult<'a> {
    pub dataset: BoxFuture<'a, Result<Dataset>>,
    pub new_transactions: BoxStream<'a, Result<(u64, Arc<Transaction>)>>,
}

pub(crate) fn load_new_transactions(dataset: &Dataset) -> NewTransactionResult<'_> {
    // Resolve every manifest with version > our current version (the latest plus
    // the ones in between). On non-lexically-ordered stores this uses the version
    // hint to avoid an O(n) listing.
    let io_parallelism = dataset.object_store.as_ref().io_parallelism();
    let locations = dataset.commit_handler.list_manifest_locations_since(
        &dataset.base,
        dataset.object_store.as_ref(),
        dataset.manifest.version,
    );

    // Will send the latest manifest via a channel.
    let (latest_tx, latest_rx) = tokio::sync::oneshot::channel();
    let mut latest_tx = Some(latest_tx);

    let manifests = locations
        .map_ok(move |location| {
            let latest_tx = latest_tx.take();
            async move {
                let manifest = Dataset::get_manifest(
                    dataset.object_store.as_ref(),
                    &location,
                    &dataset.uri,
                    dataset.session.as_ref(),
                )
                .await?;

                if let Some(latest_tx) = latest_tx {
                    // We ignore the error, since we don't care if the receiver is dropped.
                    let _ = latest_tx.send((manifest.clone(), location.clone()));
                }

                Ok((manifest, location))
            }
        })
        .try_buffer_unordered(io_parallelism / 2);
    let transactions = manifests
        .map_ok(move |(manifest, location)| async move {
            let manifest_copy = manifest.clone();
            let tx_key = TransactionKey {
                version: manifest.version,
            };
            let transaction =
                if let Some(cached) = dataset.metadata_cache.get_with_key(&tx_key).await {
                    cached
                } else {
                    let dataset_version = Dataset::checkout_manifest(
                        dataset.object_store.clone(),
                        dataset.base.clone(),
                        dataset.uri.clone(),
                        manifest_copy.clone(),
                        location,
                        dataset.session(),
                        dataset.commit_handler.clone(),
                        dataset.file_reader_options.clone(),
                        dataset.store_params.as_deref().cloned(),
                        dataset.base_store_params.clone(),
                    )?;
                    let loaded =
                        Arc::new(dataset_version.read_transaction().await?.ok_or_else(|| {
                            Error::internal(format!(
                                "Dataset version {} does not have a transaction file",
                                manifest_copy.version
                            ))
                        })?);
                    dataset
                        .metadata_cache
                        .insert_with_key(&tx_key, loaded.clone())
                        .await;
                    loaded
                };
            Ok((manifest.version, transaction))
        })
        .try_buffer_unordered(io_parallelism / 2);

    let dataset = async move {
        if let Ok((latest_manifest, location)) = latest_rx.await {
            // If we got the latest manifest, we can checkout the dataset.
            Dataset::checkout_manifest(
                dataset.object_store.clone(),
                dataset.base.clone(),
                dataset.uri.clone(),
                latest_manifest,
                location,
                dataset.session(),
                dataset.commit_handler.clone(),
                dataset.file_reader_options.clone(),
                dataset.store_params.as_deref().cloned(),
                dataset.base_store_params.clone(),
            )
        } else {
            // If we didn't get the latest manifest, we can still return the dataset
            // with the current manifest.
            Ok(dataset.clone())
        }
    }
    .boxed();

    let new_transactions = transactions.boxed();

    NewTransactionResult {
        dataset,
        new_transactions,
    }
}

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
    pub async fn drop_columns(&mut self, columns: &[&str]) -> Result<()> {
        info!(target: TRACE_DATASET_EVENTS, event=DATASET_DROPPING_COLUMN_EVENT, uri = &self.uri, columns = columns.join(","));
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
        new_schema.set_field_id(Some(self.manifest.max_field_id()));

        // Write new data file to each fragment. Parallelism is done over columns,
        // so no parallelism done at this level.
        let updated_fragments: Vec<Fragment> = stream::iter(self.get_fragments())
            .then(|f| {
                let joiner = joiner.clone();
                async move { f.merge(left_on, &joiner).await.map(|f| f.metadata) }
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
    /// It performs a left-join on the two datasets.
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
                // Call merge_index_files function for inverted index
                lance_index::scalar::inverted::builder::merge_index_files(
                    self.object_store.as_ref(),
                    &index_dir,
                    Arc::new(store),
                    progress,
                )
                .await
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

#[derive(Debug)]
pub(crate) struct ManifestWriteConfig {
    auto_set_feature_flags: bool,              // default true
    timestamp: Option<SystemTime>,             // default None
    use_stable_row_ids: bool,                  // default false
    use_legacy_format: Option<bool>,           // default None
    storage_format: Option<DataStorageFormat>, // default None
    disable_transaction_file: bool,            // default false
    /// When `Some`, this commit is the second step of `migrate_to_stable_row_ids`.
    /// It bypasses the "cannot enable stable row ids on existing dataset" guard and
    /// sets `manifest.next_row_id` to the provided value before activating the flag.
    migration_next_row_id: Option<u64>, // default None
}

impl Default for ManifestWriteConfig {
    fn default() -> Self {
        Self {
            auto_set_feature_flags: true,
            timestamp: None,
            use_stable_row_ids: false,
            disable_transaction_file: false,
            use_legacy_format: None,
            storage_format: None,
            migration_next_row_id: None,
        }
    }
}

impl ManifestWriteConfig {
    pub fn disable_transaction_file(&self) -> bool {
        self.disable_transaction_file
    }

    #[cfg(test)]
    pub(crate) fn with_transaction_file_disabled(mut self) -> Self {
        self.disable_transaction_file = true;
        self
    }

    /// Resolve into the config `Transaction::build_manifest` consumes.
    ///
    /// The timestamp is resolved here rather than during the build so it goes
    /// through this crate's mockable `SystemTime`.
    pub(crate) fn to_build_config(&self) -> ManifestBuildConfig {
        ManifestBuildConfig {
            auto_set_feature_flags: self.auto_set_feature_flags,
            timestamp_nanos: timestamp_to_nanos(self.timestamp),
            use_stable_row_ids: self.use_stable_row_ids,
            use_legacy_format: self.use_legacy_format,
            storage_format: self.storage_format.clone(),
            disable_transaction_file: self.disable_transaction_file,
            migration_next_row_id: self.migration_next_row_id,
        }
    }
}

/// Decode an inline transaction section for opportunistic caching.
///
/// Returns `None` instead of failing when the transaction cannot be decoded:
/// the section may have been written by a newer version of Lance with an
/// operation type this version does not know, and that must not prevent
/// opening the dataset. Paths that need the transaction contents surface the
/// error at their call sites instead.
fn decode_inline_transaction(message_data: &[u8], version: u64) -> Option<Transaction> {
    match lance_table::format::pb::Transaction::decode(message_data)
        .map_err(Error::from)
        .and_then(Transaction::try_from)
    {
        Ok(transaction) => Some(transaction),
        Err(err) => {
            log::warn!(
                "Failed to decode the inline transaction of version {}; \
                 it may have been written by a newer version of Lance: {}",
                version,
                err
            );
            None
        }
    }
}

/// Commit a manifest file and create a copy at the latest manifest path.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn write_manifest_file(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    config: &ManifestWriteConfig,
    naming_scheme: ManifestNamingScheme,
    transaction: Option<lance_table::format::Transaction>,
    may_change_schema: bool,
) -> std::result::Result<ManifestLocation, CommitError> {
    validate_paired_feature_flags(manifest)?;
    // Every manifest write funnels through here, including restore and clone,
    // which rebuild a manifest from a stored one rather than from an Arrow
    // schema, so this is where the invariant holds for a schema that never
    // passed through that conversion.
    //
    // Only for transactions that can change the schema. Released versions could
    // install a key on a nullable column through the metadata path, and
    // validating every write would make such a table read-only on upgrade --
    // including through the delete that removes the offending rows, which is
    // the first step of repairing it. A repair still has to pass: it changes
    // the schema, and the schema it produces is valid.
    //
    // The caller classifies the operation, rather than this reading it off
    // `transaction`, which is None whenever the encoded bytes were too large
    // to inline. Deriving it here would make the verdict depend on payload
    // size, so the same operation would be exempt while small and validated
    // once it spilled -- and a MemWAL table spills routinely, since its
    // transactions carry mem-table state.
    if may_change_schema {
        manifest
            .schema
            .verify_primary_key()
            .map_err(CommitError::OtherError)?;
    }

    if config.auto_set_feature_flags {
        // build_manifest may have already set FLAG_STABLE_ROW_IDS on the manifest.
        // Preserve it here so this second apply_feature_flags call does not clear it
        // when config.use_stable_row_ids is false (the ManifestWriteConfig default).
        let use_stable_row_ids = config.use_stable_row_ids || manifest.uses_stable_row_ids();
        apply_feature_flags(
            manifest,
            use_stable_row_ids,
            config.disable_transaction_file,
        )?;
    }

    manifest.set_timestamp(timestamp_to_nanos(config.timestamp));

    manifest.update_max_fragment_id();

    commit_handler
        .commit(
            manifest,
            indices,
            base_path,
            object_store,
            write_manifest_file_to_path,
            naming_scheme,
            transaction,
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

/// Whether anything at all lives under `prefix`. Stops at the first entry, so this stays
/// cheap even on a storage root holding millions of objects.
async fn has_any_entry(object_store: &ObjectStore, prefix: &Path) -> Result<bool> {
    match object_store.list(Some(prefix.clone())).try_next().await {
        Ok(entry) => Ok(entry.is_some()),
        // Local filesystems report a missing directory as an error where object stores
        // return an empty listing. Neither has anything to protect.
        Err(e) if e.is_not_found() => Ok(false),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests;
