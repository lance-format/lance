// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance Dataset
//!

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::DataType;
use byteorder::{ByteOrder, LittleEndian};
use chrono::{Duration, prelude::*};
use deepsize::DeepSizeOf;
use futures::future::BoxFuture;
use futures::stream::{self, BoxStream, StreamExt, TryStreamExt};
use futures::{FutureExt, Stream};

use crate::dataset::metadata::UpdateFieldMetadataBuilder;
use crate::dataset::transaction::translate_schema_metadata_updates;
use crate::index::DatasetIndexExt;
use crate::session::caches::{DSMetadataCache, ManifestKey, TransactionKey};
use crate::session::index_caches::DSIndexCache;
use itertools::Itertools;
use lance_core::ROW_ADDR;
use lance_core::datatypes::{OnMissing, OnTypeMismatch, Projectable, Projection};
use lance_core::traits::DatasetTakeRows;
use lance_core::utils::address::RowAddress;
use lance_core::utils::tracing::{
    DATASET_CLEANING_EVENT, DATASET_DELETING_EVENT, DATASET_DROPPING_COLUMN_EVENT,
    TRACE_DATASET_EVENTS,
};
use lance_datafusion::projection::ProjectionPlan;
use lance_file::datatypes::populate_schema_dictionary;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::version::LanceFileVersion;
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
    DataFile, DataStorageFormat, DeletionFile, Fragment, IndexMetadata, Manifest, RowIdMeta, pb,
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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Debug;
use std::num::NonZero;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use tracing::{info, instrument};

pub(crate) mod blob;
mod branch_location;
pub mod builder;
pub mod cleanup;
pub mod delta;
pub mod fragment;
mod hash_joiner;
pub mod index;
pub mod mem_wal;
mod metadata;
pub mod optimize;
pub mod progress;
pub mod refs;
pub(crate) mod rowids;
pub mod scanner;
mod schema_evolution;
pub mod sql;
pub mod statistics;
mod take;
pub mod transaction;
pub mod udtf;
pub mod updater;
mod utils;
pub mod write;

pub(crate) use take::row_offsets_to_row_addresses;

use self::builder::DatasetBuilder;
use self::cleanup::RemovalStats;
use self::fragment::FileFragment;
use self::refs::Refs;
use self::scanner::{DatasetRecordBatchStream, Scanner};
use self::transaction::{Operation, Transaction, TransactionBuilder, UpdateMapEntry};
use self::write::{cleanup_data_fragments, write_fragments_internal};
use crate::dataset::branch_location::BranchLocation;
use crate::dataset::cleanup::{CleanupPolicy, CleanupPolicyBuilder};
use crate::dataset::refs::{BranchContents, BranchIdentifier, Branches, Tags};
use crate::dataset::sql::SqlQueryBuilder;
use crate::datatypes::Schema;
use crate::index::retain_supported_indices;
use crate::io::commit::{
    commit_detached_transaction, commit_new_dataset, commit_transaction,
    detect_overlapping_fragments,
};
use crate::session::Session;
use crate::utils::temporal::{SystemTime, timestamp_to_nanos, utc_now};
use crate::{Error, Result};
pub use blob::{BlobFile, ReadBlob, ReadBlobsBuilder, ReadBlobsStream};
use hash_joiner::HashJoiner;
pub use lance_core::ROW_ID;
use lance_core::box_error;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_namespace::models::{DeclareTableRequest, DescribeTableRequest};
use lance_table::feature_flags::{apply_feature_flags, can_read_dataset};
use lance_table::io::deletion::{DELETIONS_DIR, relative_deletion_file_path};
pub use schema_evolution::{
    BatchInfo, BatchUDF, ColumnAlteration, NewColumnTransform, UDFCheckpointStore,
};
pub use take::TakeBuilder;
pub use write::merge_insert::{
    MergeInsertBuilder, MergeInsertJob, MergeStats, UncommittedMergeInsert, WhenMatched,
    WhenNotMatched, WhenNotMatchedBySource,
};

use crate::dataset::index::LanceIndexStoreExt;
pub use write::update::{UpdateBuilder, UpdateJob};
#[allow(deprecated)]
pub use write::{
    AutoCleanupParams, CommitBuilder, DeleteBuilder, DeleteResult, ExternalBlobMode, InsertBuilder,
    UncommittedDelete, WriteDestination, WriteMode, WriteParams, WriteProgressFn, WriteStats,
    write_fragments,
};

pub(crate) const INDICES_DIR: &str = "_indices";
pub(crate) const DATA_DIR: &str = "data";
pub(crate) const TRANSACTIONS_DIR: &str = "_transactions";

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
}

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

    pub fn branches(&self) -> Branches<'_> {
        self.refs.branches()
    }

    /// Check out the latest version of the dataset
    pub async fn checkout_latest(&mut self) -> Result<()> {
        let (manifest, manifest_location) = self.latest_manifest().await?;
        self.manifest = manifest;
        self.manifest_location = manifest_location;
        self.fragment_bitmap = Arc::new(
            self.manifest
                .fragments
                .iter()
                .map(|f| f.id as u32)
                .collect(),
        );
        Ok(())
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
        let clone_op = Operation::Clone {
            is_shallow: true,
            ref_name: source_branch.clone(),
            ref_version: version_number,
            ref_path: String::from(self.uri()),
            branch_name: Some(branch.to_string()),
        };
        let transaction = Transaction::new(version_number, clone_op, None);

        let builder = CommitBuilder::new(WriteDestination::Uri(branch_location.uri.as_str()))
            .with_store_params(store_params.unwrap_or_default())
            .with_object_store(Arc::new(self.object_store.as_ref().clone()))
            .with_commit_handler(self.commit_handler.clone())
            .with_storage_format(self.manifest.data_storage_format.lance_file_version()?);
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
        // We check the e_tag here just in case it has been overwritten. This can
        // happen if the table has been dropped then re-created recently.
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

        let manifest = Self::load_manifest(
            self.object_store.as_ref(),
            &manifest_location,
            &new_location.uri,
            self.session.as_ref(),
        )
        .await?;
        Self::checkout_manifest(
            self.object_store.clone(),
            new_location.path,
            new_location.uri,
            Arc::new(manifest),
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

        if !can_read_dataset(manifest.reader_feature_flags) {
            let message = format!(
                "This dataset cannot be read by this version of Lance. \
                 Please upgrade Lance to read this dataset.\n Flags: {}",
                manifest.reader_feature_flags
            );
            return Err(Error::not_supported_source(message.into()));
        }

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
            let mut indices: Vec<IndexMetadata> = section
                .indices
                .into_iter()
                .map(IndexMetadata::try_from)
                .collect::<Result<Vec<_>>>()?;
            retain_supported_indices(&mut indices);
            let ds_index_cache = session.index_cache.for_dataset(uri);
            let metadata_key = crate::session::index_caches::IndexMetadataKey {
                version: manifest_location.version,
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
            let transaction: Transaction =
                lance_table::format::pb::Transaction::decode(message_data)?.try_into()?;

            let metadata_cache = session.metadata_cache.for_dataset(uri);
            let metadata_key = TransactionKey {
                version: manifest_location.version,
            };
            metadata_cache
                .insert_with_key(&metadata_key, Arc::new(transaction))
                .await;
        }

        if manifest.should_use_legacy_format() {
            populate_schema_dictionary(&mut manifest.schema, object_reader.as_ref()).await?;
        }

        Ok(manifest)
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
        mut params: Option<WriteParams>,
    ) -> Result<Self> {
        let mut write_params = params.take().unwrap_or_default();

        match write_params.mode {
            WriteMode::Create => {
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
                    let external_store = LanceNamespaceExternalManifestStore::new(
                        namespace_client.clone(),
                        table_id.clone(),
                    );
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

                // Set up commit handler when managed_versioning is enabled
                if response.managed_versioning == Some(true) {
                    let external_store = LanceNamespaceExternalManifestStore::new(
                        namespace_client.clone(),
                        table_id.clone(),
                    );
                    let commit_handler: Arc<dyn CommitHandler> =
                        Arc::new(ExternalManifestCommitHandler {
                            external_manifest_store: Arc::new(external_store),
                        });
                    write_params.commit_handler = Some(commit_handler);
                }

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

    pub fn commit_handler(&self) -> &dyn CommitHandler {
        self.commit_handler.as_ref()
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

    // TODO: Cache this
    pub(crate) fn is_legacy_storage(&self) -> bool {
        self.manifest
            .data_storage_format
            .lance_file_version()
            .unwrap()
            == LanceFileVersion::Legacy
    }

    pub async fn latest_manifest(&self) -> Result<(Arc<Manifest>, ManifestLocation)> {
        let location = self
            .commit_handler
            .resolve_latest_location(&self.base, &self.object_store)
            .await?;

        // Check if manifest is in cache before reading from storage
        let manifest_key = ManifestKey {
            version: location.version,
            e_tag: location.e_tag.as_deref(),
        };
        let cached_manifest = self.metadata_cache.get_with_key(&manifest_key).await;
        if let Some(cached_manifest) = cached_manifest {
            return Ok((cached_manifest, location));
        }

        if self.already_checked_out(&location, self.manifest.branch.as_deref()) {
            return Ok((self.manifest.clone(), self.manifest_location.clone()));
        }
        let mut manifest = read_manifest(&self.object_store, &location.path, location.size).await?;
        if manifest.schema.has_dictionary_types() && manifest.should_use_legacy_format() {
            let reader = if let Some(size) = location.size {
                self.object_store
                    .open_with_size(&location.path, size as usize)
                    .await?
            } else {
                self.object_store.open(&location.path).await?
            };
            populate_schema_dictionary(&mut manifest.schema, reader.as_ref()).await?;
        }
        let manifest_arc = Arc::new(manifest);
        self.metadata_cache
            .insert_with_key(&manifest_key, manifest_arc.clone())
            .await;
        Ok((manifest_arc, location))
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

        // Prefer inline transaction from manifest when available
        let transaction = if let Some(pos) = self.manifest.transaction_section {
            let reader = if let Some(size) = self.manifest_location.size {
                self.object_store
                    .open_with_size(&self.manifest_location.path, size as usize)
                    .await?
            } else {
                self.object_store.open(&self.manifest_location.path).await?
            };

            let tx: pb::Transaction = read_message(reader.as_ref(), pos).await?;
            Transaction::try_from(tx).map(Some)?
        } else if let Some(path) = &self.manifest.transaction_file {
            // Fallback: read external transaction file if present
            let path = self.transactions_dir().join(path.as_str());
            let data = self.object_store.inner.get(&path).await?.bytes().await?;
            let transaction = lance_table::format::pb::Transaction::decode(data)?;
            Transaction::try_from(transaction).map(Some)?
        } else {
            None
        };

        if let Some(tx) = transaction.as_ref() {
            self.metadata_cache
                .insert_with_key(&transaction_key, Arc::new(tx.clone()))
                .await;
        }
        Ok(transaction)
    }

    /// Read the transaction file for this version of the dataset.
    ///
    /// If there was no transaction file written for this version of the dataset
    /// then this will return None.
    pub async fn read_transaction_by_version(&self, version: u64) -> Result<Option<Transaction>> {
        let dataset_version = self.checkout_version(version).await?;
        dataset_version.read_transaction().await
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
        info!(target: TRACE_DATASET_EVENTS, event=DATASET_CLEANING_EVENT, uri=&self.uri);
        cleanup::cleanup_old_versions(self, policy).boxed()
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
            self.manifest_location.naming_scheme,
            None,
            None,
        )
        .await?;

        self.manifest = Arc::new(manifest);
        self.manifest_location = manifest_location;
        self.fragment_bitmap = Arc::new(
            self.manifest
                .fragments
                .iter()
                .map(|f| f.id as u32)
                .collect(),
        );

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
    pub async fn take_blobs(
        self: &Arc<Self>,
        row_ids: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<BlobFile>> {
        blob::take_blobs(self, row_ids, column.as_ref()).await
    }

    /// Take [BlobFile] by row addresses.
    ///
    /// Row addresses are `u64` values encoding `(fragment_id << 32) | row_offset`.
    /// Use this method when you already have row addresses, for example from
    /// a scan with `with_row_address()`. For row IDs (stable identifiers), use
    /// [`Self::take_blobs`]. For row indices (offsets), use
    /// [`Self::take_blobs_by_indices`].
    pub async fn take_blobs_by_addresses(
        self: &Arc<Self>,
        row_addrs: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<BlobFile>> {
        blob::take_blobs_by_addresses(self, row_addrs, column.as_ref()).await
    }

    /// Take [BlobFile] by row indices (offsets in the dataset).
    pub async fn take_blobs_by_indices(
        self: &Arc<Self>,
        row_indices: &[u64],
        column: impl AsRef<str>,
    ) -> Result<Vec<BlobFile>> {
        let fragments = self.get_fragments();
        let row_addrs = row_offsets_to_row_addresses(&fragments, row_indices).await?;
        blob::take_blobs_by_addresses(self, &row_addrs, column.as_ref()).await
    }

    /// Create a planned blob reader for a blob column.
    ///
    /// This API complements [`Self::take_blobs`]. `take_blobs` returns
    /// [`BlobFile`] handles for caller-driven random access, while
    /// `read_blobs` builds a streaming read plan for sequential or batched blob
    /// retrieval.
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

                let selected_fragment_ids = fragment_ids.iter().copied().collect::<BTreeSet<_>>();
                let selected_fragments = self
                    .get_fragments()
                    .into_iter()
                    .filter(|fragment| selected_fragment_ids.contains(&(fragment.id() as u32)))
                    .collect::<Vec<_>>();

                if selected_fragments.len() != selected_fragment_ids.len() {
                    let present_fragment_ids = selected_fragments
                        .iter()
                        .map(|fragment| fragment.id() as u32)
                        .collect::<HashSet<_>>();
                    let missing_fragment_ids = selected_fragment_ids
                        .into_iter()
                        .filter(|fragment_id| !present_fragment_ids.contains(fragment_id))
                        .collect::<Vec<_>>();
                    return Err(Error::invalid_input(format!(
                        "Dataset::sample received fragment ids that are not part of the current dataset version: {missing_fragment_ids:?}",
                    )));
                }

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
        let mut object_store = self.object_store.as_ref().clone();
        for wrapper in &wrappers {
            object_store.inner =
                wrapper.wrap(&object_store.store_prefix, object_store.inner.clone());
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

    fn store_params_for_base(
        &self,
        base_path: Option<&lance_table::format::BasePath>,
    ) -> ObjectStoreParams {
        // Base-specific bindings are exact ObjectStoreParams keyed by
        // `BasePath.path`. If a base has no explicit binding then reads fall back
        // to the dataset-level default store params.
        base_path
            .and_then(|base_path| {
                self.base_store_params
                    .as_ref()
                    .and_then(|params| params.get(&base_path.path))
            })
            .cloned()
            .unwrap_or_else(|| self.store_params.as_deref().cloned().unwrap_or_default())
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

        let file_version = LanceFileVersion::try_from_major_minor(
            file_metadata.major_version as u32,
            file_metadata.minor_version as u32,
        )?;

        // Get top-level column names from file schema in file order
        let column_names: Vec<&str> = file_metadata
            .file_schema
            .fields
            .iter()
            .map(|f| f.name.as_str())
            .collect();

        // Project dataset schema by file column names to get dataset field IDs
        let projected_ds_schema = self.schema().project(&column_names)?;

        // Walk both schemas in parallel to build fields and column_indices
        let is_structural = file_version >= LanceFileVersion::V2_1;
        let ds_fields: Vec<_> = projected_ds_schema.fields_pre_order().collect();
        let file_fields: Vec<_> = file_metadata.file_schema.fields_pre_order().collect();

        if ds_fields.len() != file_fields.len() {
            return Err(Error::invalid_input(format!(
                "Schema mismatch: dataset projection has {} fields but file has {} fields",
                ds_fields.len(),
                file_fields.len()
            )));
        }

        let mut fields = Vec::new();
        let mut column_indices = Vec::new();
        let mut curr_column_idx: i32 = 0;
        let mut packed_struct_fields_num: usize = 0;

        for (ds_field, file_field) in ds_fields.iter().zip(file_fields.iter()) {
            if ds_field.name != file_field.name {
                return Err(Error::invalid_input(format!(
                    "Schema mismatch: expected field '{}' but file has '{}'",
                    ds_field.name, file_field.name
                )));
            }

            if packed_struct_fields_num > 0 {
                packed_struct_fields_num -= 1;
                continue;
            }

            if file_field.is_packed_struct() {
                fields.push(ds_field.id);
                column_indices.push(curr_column_idx);
                curr_column_idx += 1;
                packed_struct_fields_num = file_field.children.len();
            } else if file_field.children.is_empty() || !is_structural {
                fields.push(ds_field.id);
                column_indices.push(curr_column_idx);
                curr_column_idx += 1;
            }
        }

        let file_size_nz = NonZero::new(file_size);
        Ok(DataFile::new(
            path,
            fields,
            column_indices,
            file_metadata.major_version as u32,
            file_metadata.minor_version as u32,
            file_size_nz,
            base_id,
        ))
    }

    /// Resolve the data directory for a given base_id.
    ///
    /// If `base_id` is `None`, returns the default data directory.
    fn data_file_dir_for_base(&self, base_id: Option<u32>) -> Result<Path> {
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

        let (store, _) = ObjectStore::from_uri_and_params(
            self.session.store_registry(),
            &base_path.path,
            &store_params,
        )
        .await?;

        Ok(store)
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
        let dataset = Arc::new(self.clone());
        let fragment = self
            .manifest
            .fragments
            .iter()
            .find(|f| f.id == fragment_id as u64)?;
        Some(FileFragment::new(dataset, fragment.clone()))
    }

    pub fn fragments(&self) -> &Arc<Vec<Fragment>> {
        &self.manifest.fragments
    }

    // Gets a filtered list of fragments from ids in O(N) time instead of using
    // `get_fragment` which would require O(N^2) time.
    pub fn get_frags_from_ordered_ids(&self, ordered_ids: &[u32]) -> Vec<Option<FileFragment>> {
        let mut fragments = Vec::with_capacity(ordered_ids.len());
        let mut id_iter = ordered_ids.iter();
        let mut id = id_iter.next();
        // This field is just used to assert the ids are in order
        let mut last_id: i64 = -1;
        for frag in self.manifest.fragments.iter() {
            let mut the_id = if let Some(id) = id { *id } else { break };
            // Assert the given ids are, in fact, in order
            assert!(the_id as i64 > last_id);
            // For any IDs we've passed we can assume that no fragment exists any longer
            // with that ID.
            while the_id < frag.id as u32 {
                fragments.push(None);
                last_id = the_id as i64;
                id = id_iter.next();
                the_id = if let Some(id) = id { *id } else { break };
            }

            if the_id == frag.id as u32 {
                fragments.push(Some(FileFragment::new(
                    Arc::new(self.clone()),
                    frag.clone(),
                )));
                last_id = the_id as i64;
                id = id_iter.next();
            }
        }
        fragments
    }

    // This method filters deleted items from `addr_or_ids` using `addrs` as a reference
    async fn filter_addr_or_ids(&self, addr_or_ids: &[u64], addrs: &[u64]) -> Result<Vec<u64>> {
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
        let addresses = if let Some(row_id_index) = get_row_id_index(self).await? {
            let addresses = ids
                .iter()
                .filter_map(|id| row_id_index.get(*id).map(|address| address.into()))
                .collect::<Vec<_>>();
            Cow::Owned(addresses)
        } else {
            Cow::Borrowed(ids)
        };

        self.filter_addr_or_ids(ids, &addresses).await
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

        // Validate indices
        let indices = self.load_indices().await?;
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
                        &index.uuid, self.base
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
        let clone_op = Operation::Clone {
            is_shallow: true,
            ref_name,
            ref_version: version_number,
            ref_path: self.uri.clone(),
            branch_name: None,
        };
        let transaction = Transaction::new(version_number, clone_op, None);

        let builder = CommitBuilder::new(WriteDestination::Uri(target_path))
            .with_store_params(
                store_params.unwrap_or(self.store_params.as_deref().cloned().unwrap_or_default()),
            )
            .with_object_store(Arc::new(self.object_store.as_ref().clone()))
            .with_commit_handler(self.commit_handler.clone())
            .with_storage_format(self.manifest.data_storage_format.lance_file_version()?);
        builder.execute(transaction).await
    }

    /// Deep clone the target version into a new dataset at target_path.
    /// This performs a server-side copy of all relevant dataset files (data files,
    /// deletion files, and any external row-id files) into the target dataset
    /// without loading data into memory.
    ///
    /// Parameters:
    /// - `target_path`: the URI string to clone the dataset into.
    /// - `version`: the version cloned from, could be a version number, branch head, or tag.
    /// - `store_params`: the object store params to use for the new dataset.
    pub async fn deep_clone(
        &mut self,
        target_path: &str,
        version: impl Into<refs::Ref>,
        store_params: Option<ObjectStoreParams>,
    ) -> Result<Self> {
        use futures::StreamExt;

        // Resolve source dataset and its manifest using checkout_version
        let src_ds = self.checkout_version(version).await?;
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

        // TODO: Leverage object store bulk copy for efficient deep_clone
        //
        // All cloud storage providers support batch copy APIs that would provide significant
        // performance improvements. We use single file copy before we have upstream support.
        //
        // Tracked by: https://github.com/lance-format/lance/issues/5435
        let io_parallelism = self.object_store.io_parallelism();
        let copy_futures = src_paths
            .iter()
            .map(|(relative_path, base)| {
                let store = Arc::clone(&target_store);
                let src_path = build_absolute_path(relative_path, base);
                let target_path = build_absolute_path(relative_path, &target_base);
                async move { store.copy(&src_path, &target_path).await.map(|_| ()) }
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
            .with_commit_handler(self.commit_handler.clone())
            .with_storage_format(self.manifest.data_storage_format.lance_file_version()?);
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
            for data_file in fragment.files.iter() {
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

    /// Returns true if Lance supports writing this datatype with nulls.
    pub(crate) fn lance_supports_nulls(&self, datatype: &DataType) -> bool {
        match self
            .manifest()
            .data_storage_format
            .lance_file_version()
            .unwrap_or(LanceFileVersion::Legacy)
            .resolve()
        {
            LanceFileVersion::Legacy => matches!(
                datatype,
                DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Binary
                    | DataType::List(_)
                    | DataType::FixedSizeBinary(_)
                    | DataType::FixedSizeList(_, _)
            ),
            LanceFileVersion::V2_0 => !matches!(datatype, DataType::Struct(..)),
            _ => true,
        }
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
                let manifest_key = ManifestKey {
                    version: location.version,
                    e_tag: location.e_tag.as_deref(),
                };
                let manifest = if let Some(cached) =
                    dataset.metadata_cache.get_with_key(&manifest_key).await
                {
                    cached
                } else {
                    let loaded = Arc::new(
                        Dataset::load_manifest(
                            dataset.object_store.as_ref(),
                            &location,
                            &dataset.uri,
                            dataset.session.as_ref(),
                        )
                        .await?,
                    );
                    dataset
                        .metadata_cache
                        .insert_with_key(&manifest_key, loaded.clone())
                        .await;
                    loaded
                };

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

        let transaction = Transaction::new(
            self.manifest.version,
            Operation::Merge {
                fragments: updated_fragments,
                schema: new_schema,
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
        index_uuid: &str,
        index_type: IndexType,
        batch_readhead: Option<usize>,
        progress: Arc<dyn IndexBuildProgress>,
    ) -> Result<()> {
        let store = LanceIndexStore::from_dataset_for_new(self, index_uuid)?;
        let index_dir = self.indices_dir().join(index_uuid);
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
                // Call merge_index_files function for btree index
                lance_index::scalar::btree::merge_index_files(
                    self.object_store.as_ref(),
                    &index_dir,
                    Arc::new(store),
                    batch_readhead,
                    progress,
                )
                .await
            }
            IndexType::Bitmap => {
                lance_index::scalar::bitmap::merge_index_files(
                    self.object_store.as_ref(),
                    &index_dir,
                    Arc::new(store),
                    progress,
                )
                .await
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
pub struct ManifestWriteConfig {
    auto_set_feature_flags: bool,              // default true
    timestamp: Option<SystemTime>,             // default None
    use_stable_row_ids: bool,                  // default false
    use_legacy_format: Option<bool>,           // default None
    storage_format: Option<DataStorageFormat>, // default None
    disable_transaction_file: bool,            // default false
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
        }
    }
}

impl ManifestWriteConfig {
    pub fn disable_transaction_file(&self) -> bool {
        self.disable_transaction_file
    }
}

/// Commit a manifest file and create a copy at the latest manifest path.
#[allow(clippy::too_many_arguments)]
pub async fn write_manifest_file(
    object_store: &ObjectStore,
    commit_handler: &dyn CommitHandler,
    base_path: &Path,
    manifest: &mut Manifest,
    indices: Option<Vec<IndexMetadata>>,
    config: &ManifestWriteConfig,
    naming_scheme: ManifestNamingScheme,
    mut transaction: Option<&Transaction>,
) -> std::result::Result<ManifestLocation, CommitError> {
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
            transaction.take().map(|tx| tx.into()),
        )
        .await
}

impl Projectable for Dataset {
    fn schema(&self) -> &Schema {
        self.schema()
    }
}

#[cfg(test)]
mod tests;
