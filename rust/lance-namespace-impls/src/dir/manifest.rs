// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Manifest-based namespace implementation
//!
//! This module provides a namespace implementation that uses a manifest table
//! to track tables and nested namespaces.

use arrow::array::{
    Array, LargeBinaryArray, LargeStringArray, RecordBatch, RecordBatchIterator, StringArray,
    UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use arrow_ipc::reader::StreamReader;
use async_trait::async_trait;
use bytes::Bytes;
use datafusion_common::DataFusionError;
use datafusion_physical_plan::{
    SendableRecordBatchStream,
    stream::RecordBatchStreamAdapter as DatafusionRecordBatchStreamAdapter,
};
use futures::{
    FutureExt, TryStreamExt,
    stream::{self, StreamExt},
};
use lance::dataset::index::LanceIndexStoreExt;
use lance::dataset::transaction::Operation;
use lance::dataset::{
    InsertBuilder, ManifestWriteConfig, ReadParams, WhenMatched, WriteMode, WriteParams,
    builder::DatasetBuilder, write_manifest_file,
};
use lance::session::Session;
use lance::{Dataset, dataset::scanner::Scanner};
use lance_arrow::json::{JsonArray, decode_json, json_field};
use lance_core::datatypes::LANCE_UNENFORCED_PRIMARY_KEY_POSITION;
use lance_core::{Error as LanceError, ROW_ID};
use lance_core::{Error, Result};
use lance_index::progress::noop_progress;
use lance_index::registry::IndexPluginRegistry;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::registry::VALUE_COLUMN_NAME;
use lance_index::scalar::{BuiltinIndexType, CreatedIndex, InvertedIndexParams, ScalarIndexParams};
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_namespace::LanceNamespace;
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{
    CreateNamespaceRequest, CreateNamespaceResponse, CreateTableRequest, CreateTableResponse,
    DeclareTableRequest, DeclareTableResponse, DeregisterTableRequest, DeregisterTableResponse,
    DescribeNamespaceRequest, DescribeNamespaceResponse, DescribeTableRequest,
    DescribeTableResponse, DescribeTableVersionResponse, DropNamespaceRequest,
    DropNamespaceResponse, DropTableRequest, DropTableResponse, ListNamespacesRequest,
    ListNamespacesResponse, ListTableVersionsResponse, ListTablesRequest, ListTablesResponse,
    NamespaceExistsRequest, RegisterTableRequest, RegisterTableResponse, TableExistsRequest,
    TableVersion,
};
use lance_namespace::schema::arrow_schema_to_json;
use lance_table::format::{IndexMetadata, Manifest};
use lance_table::io::commit::CommitError;
use object_store::{Error as ObjectStoreError, path::Path};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex as StdMutex, MutexGuard as StdMutexGuard},
};
use tokio::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use uuid::Uuid;

const MANIFEST_TABLE_NAME: &str = "__manifest";
const SUPER_MANIFEST_TABLE_NAME: &str = "__super_manifest";
const SHARDED_MANIFEST_TABLE_PREFIX: &str = "__manifest_shard";
const DELIMITER: &str = "$";
/// Bounded concurrency for per-table `_versions/` probes when filtering declared tables.
/// Higher values reduce latency but increase burst load against the object store.
pub(crate) const DECLARED_FILTER_CONCURRENCY: usize = 16;

// Index names for the __manifest table
/// BTREE index on the object_id column for fast lookups
const OBJECT_ID_INDEX_NAME: &str = "object_id_btree";
/// Bitmap index on the object_type column for filtering by type
const OBJECT_TYPE_INDEX_NAME: &str = "object_type_bitmap";
/// JSON FTS index on the metadata column for metadata text search
const METADATA_INDEX_NAME: &str = "metadata_fts";
// Each retry reloads and rewrites the full manifest. Match the regular Lance
// commit retry budget so multi-process namespace writes can make progress.
const DEFAULT_MANIFEST_REWRITE_COMMIT_RETRIES: u32 = 20;
const MANIFEST_INDEX_BATCH_SIZE: usize = 8192;

/// Object types that can be stored in the manifest
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Namespace,
    Table,
    TableVersion,
    CatalogBranch,
}

impl ObjectType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Namespace => "namespace",
            Self::Table => "table",
            Self::TableVersion => "table_version",
            Self::CatalogBranch => "catalog_branch",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "namespace" => Ok(Self::Namespace),
            "table" => Ok(Self::Table),
            "table_version" => Ok(Self::TableVersion),
            "catalog_branch" => Ok(Self::CatalogBranch),
            _ => Err(NamespaceError::Internal {
                message: format!("Invalid object type: {}", s),
            }
            .into()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CreateTableMode {
    Create,
    ExistOk,
    Overwrite,
}

impl CreateTableMode {
    fn parse(mode: Option<&str>) -> Result<Self> {
        match mode {
            None => Ok(Self::Create),
            Some(mode) if mode.eq_ignore_ascii_case("create") => Ok(Self::Create),
            Some(mode)
                if mode.eq_ignore_ascii_case("existok")
                    || mode.eq_ignore_ascii_case("exist_ok") =>
            {
                Ok(Self::ExistOk)
            }
            Some(mode) if mode.eq_ignore_ascii_case("overwrite") => Ok(Self::Overwrite),
            Some(mode) => Err(NamespaceError::InvalidInput {
                message: format!(
                    "Unsupported create_table mode '{}'. Supported modes are: 'Create', 'ExistOk', 'Overwrite'",
                    mode
                ),
            }
            .into()),
        }
    }

    fn write_mode(self) -> WriteMode {
        match self {
            Self::Overwrite => WriteMode::Overwrite,
            Self::Create | Self::ExistOk => WriteMode::Create,
        }
    }
}

/// Information about a table stored in the manifest
#[derive(Debug, Clone)]
pub struct TableInfo {
    pub namespace: Vec<String>,
    pub name: String,
    pub location: String,
    pub metadata: Option<HashMap<String, String>>,
}

/// An entry to be inserted into the manifest table.
///
/// This struct makes the meaning of each field explicit, replacing the
/// previous tuple-based API `(String, ObjectType, Option<String>, Option<String>)`.
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    /// The unique object identifier (e.g., table name or version object_id)
    pub object_id: String,
    /// The type of the object (Namespace, Table, or TableVersion)
    pub object_type: ObjectType,
    /// The storage location (e.g., directory name for tables)
    pub location: Option<String>,
    /// Additional metadata serialized as JSON
    pub metadata: Option<String>,
}

struct CopyOnWriteMutation<T> {
    result: T,
    has_changes: bool,
}

impl<T> CopyOnWriteMutation<T> {
    fn updated(result: T) -> Self {
        Self {
            result,
            has_changes: true,
        }
    }

    fn unchanged(result: T) -> Self {
        Self {
            result,
            has_changes: false,
        }
    }
}

struct ManifestIndexBuildInput {
    index_name: &'static str,
    column_name: &'static str,
    params: ScalarIndexParams,
    field: Field,
    stream: SendableRecordBatchStream,
}

struct ManifestTrainedIndex {
    index_name: &'static str,
    column_name: &'static str,
    uuid: Uuid,
    created_index: CreatedIndex,
}

struct ManifestRowValue {
    object_id: String,
    object_type: ObjectType,
    location: Option<String>,
    metadata: Option<String>,
}

struct ManifestOutputRow<'a> {
    object_id: &'a str,
    object_type: ObjectType,
    location: Option<&'a str>,
    metadata: Option<&'a str>,
}

#[derive(Default)]
struct ManifestIndexAccumulator {
    object_ids: BTreeMap<Arc<str>, u64>,
    object_types: BTreeMap<&'static str, RoaringBitmap>,
    metadata_values: Vec<Option<String>>,
    metadata_row_ids: Vec<u64>,
    row_count: u64,
}

impl ManifestIndexAccumulator {
    fn next_row_id(&self) -> Result<u64> {
        if self.row_count > u64::from(u32::MAX) {
            return Err(NamespaceError::Internal {
                message: format!(
                    "Manifest rewrite exceeded maximum single-fragment row count: {}",
                    self.row_count
                ),
            }
            .into());
        }
        Ok(self.row_count)
    }

    fn push(&mut self, row: &ManifestOutputRow<'_>) -> Result<u64> {
        let row_id = self.next_row_id()?;
        if self
            .object_ids
            .insert(Arc::<str>::from(row.object_id), row_id)
            .is_some()
        {
            return Err(NamespaceError::Internal {
                message: format!("Manifest contains duplicate object_id '{}'", row.object_id),
            }
            .into());
        }
        self.object_types
            .entry(row.object_type.as_str())
            .or_default()
            .insert(row_id as u32);
        self.metadata_values
            .push(row.metadata.map(ToString::to_string));
        self.metadata_row_ids.push(row_id);
        self.row_count += 1;
        Ok(row_id)
    }
}

struct ManifestBatchBuilder {
    object_ids: Vec<String>,
    object_types: Vec<&'static str>,
    locations: Vec<Option<String>>,
    metadatas: Vec<Option<String>>,
    row_ids: Vec<u64>,
}

impl ManifestBatchBuilder {
    fn new() -> Self {
        Self {
            object_ids: Vec::new(),
            object_types: Vec::new(),
            locations: Vec::new(),
            metadatas: Vec::new(),
            row_ids: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.object_ids.is_empty()
    }

    fn append(
        &mut self,
        index_data: &mut ManifestIndexAccumulator,
        row: ManifestOutputRow<'_>,
    ) -> Result<()> {
        let row_id = index_data.push(&row)?;
        self.object_ids.push(row.object_id.to_string());
        self.object_types.push(row.object_type.as_str());
        self.locations.push(row.location.map(ToString::to_string));
        self.metadatas.push(row.metadata.map(ToString::to_string));
        self.row_ids.push(row_id);
        Ok(())
    }

    fn finish(self) -> Result<RecordBatch> {
        let metadata_array = Arc::new(
            JsonArray::try_from_iter(self.metadatas.iter().map(|v| v.as_deref()))
                .map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!("Failed to encode manifest metadata as JSON: {}", e),
                    })
                })?
                .into_inner(),
        );
        RecordBatch::try_new(
            ManifestNamespace::manifest_schema(),
            vec![
                Arc::new(StringArray::from(self.object_ids)),
                Arc::new(StringArray::from(self.object_types)),
                Arc::new(StringArray::from(self.locations)),
                metadata_array,
            ],
        )
        .map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to create manifest snapshot batch: {:?}", e),
            })
        })
    }
}

trait ManifestStreamMutation: Send {
    type Output: Clone + Send + 'static;

    fn process_existing_row(
        &mut self,
        row: ManifestRowValue,
        output: &mut ManifestBatchBuilder,
        index_data: &mut ManifestIndexAccumulator,
    ) -> Result<()>;

    fn append_rows(
        &mut self,
        output: &mut ManifestBatchBuilder,
        index_data: &mut ManifestIndexAccumulator,
    ) -> Result<()>;

    fn finish(&self) -> CopyOnWriteMutation<Self::Output>;
}

struct ManifestRewriteShared<M: ManifestStreamMutation> {
    mutation: M,
    index_data: Option<ManifestIndexAccumulator>,
    result: Option<CopyOnWriteMutation<M::Output>>,
    error: Option<LanceError>,
}

impl<M: ManifestStreamMutation> ManifestRewriteShared<M> {
    fn new(mutation: M) -> Self {
        Self {
            mutation,
            index_data: Some(ManifestIndexAccumulator::default()),
            result: None,
            error: None,
        }
    }
}

struct UpsertManifestMutation {
    entries: Vec<ManifestEntry>,
    entry_positions: HashMap<String, usize>,
    matched: Vec<bool>,
    when_matched: WhenMatched,
}

impl UpsertManifestMutation {
    fn new(entries: Vec<ManifestEntry>, when_matched: WhenMatched) -> Self {
        let entry_positions = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.object_id.clone(), index))
            .collect();
        let matched = vec![false; entries.len()];
        Self {
            entries,
            entry_positions,
            matched,
            when_matched,
        }
    }

    fn entry_row(&self, index: usize) -> ManifestOutputRow<'_> {
        let entry = &self.entries[index];
        ManifestOutputRow {
            object_id: &entry.object_id,
            object_type: entry.object_type,
            location: entry.location.as_deref(),
            metadata: entry.metadata.as_deref(),
        }
    }
}

impl ManifestStreamMutation for UpsertManifestMutation {
    type Output = ();

    fn process_existing_row(
        &mut self,
        row: ManifestRowValue,
        output: &mut ManifestBatchBuilder,
        index_data: &mut ManifestIndexAccumulator,
    ) -> Result<()> {
        if let Some(index) = self.entry_positions.get(&row.object_id).copied() {
            match self.when_matched {
                WhenMatched::Fail => {
                    return Err(NamespaceError::ConcurrentModification {
                        message: format!(
                            "Object '{}' was concurrently created by another operation",
                            row.object_id
                        ),
                    }
                    .into());
                }
                WhenMatched::UpdateAll => {
                    self.matched[index] = true;
                    output.append(index_data, self.entry_row(index))?;
                    return Ok(());
                }
                _ => {
                    return Err(NamespaceError::Internal {
                        message: format!(
                            "Unsupported manifest rewrite matched action: {:?}",
                            self.when_matched
                        ),
                    }
                    .into());
                }
            }
        }

        output.append(
            index_data,
            ManifestOutputRow {
                object_id: &row.object_id,
                object_type: row.object_type,
                location: row.location.as_deref(),
                metadata: row.metadata.as_deref(),
            },
        )
    }

    fn append_rows(
        &mut self,
        output: &mut ManifestBatchBuilder,
        index_data: &mut ManifestIndexAccumulator,
    ) -> Result<()> {
        for index in 0..self.entries.len() {
            if !self.matched[index] {
                output.append(index_data, self.entry_row(index))?;
            }
        }
        Ok(())
    }

    fn finish(&self) -> CopyOnWriteMutation<Self::Output> {
        CopyOnWriteMutation::updated(())
    }
}

struct DeleteObjectMutation {
    object_id: String,
    deleted: bool,
}

impl ManifestStreamMutation for DeleteObjectMutation {
    type Output = ();

    fn process_existing_row(
        &mut self,
        row: ManifestRowValue,
        output: &mut ManifestBatchBuilder,
        index_data: &mut ManifestIndexAccumulator,
    ) -> Result<()> {
        if row.object_id == self.object_id {
            self.deleted = true;
            return Ok(());
        }

        output.append(
            index_data,
            ManifestOutputRow {
                object_id: &row.object_id,
                object_type: row.object_type,
                location: row.location.as_deref(),
                metadata: row.metadata.as_deref(),
            },
        )
    }

    fn append_rows(
        &mut self,
        _output: &mut ManifestBatchBuilder,
        _index_data: &mut ManifestIndexAccumulator,
    ) -> Result<()> {
        Ok(())
    }

    fn finish(&self) -> CopyOnWriteMutation<Self::Output> {
        if self.deleted {
            CopyOnWriteMutation::updated(())
        } else {
            CopyOnWriteMutation::unchanged(())
        }
    }
}

enum DeleteTableVersionsTarget {
    ObjectIds(HashSet<String>),
    Ranges(Vec<DeleteTableVersionRangeTarget>),
}

type TableVersionRange = (i64, i64);
type TableVersionRangeRequest = (String, Vec<TableVersionRange>);

#[derive(Clone)]
struct DeleteTableVersionRangeTarget {
    object_id_prefix: String,
    ranges: Vec<TableVersionRange>,
}

impl DeleteTableVersionRangeTarget {
    fn matches(&self, object_id: &str) -> bool {
        let Some(version) = object_id
            .strip_prefix(&self.object_id_prefix)
            .and_then(|suffix| suffix.parse::<i64>().ok())
        else {
            return false;
        };

        self.ranges
            .iter()
            .any(|(start, end)| *start <= version && version <= *end)
    }
}

impl DeleteTableVersionsTarget {
    fn matches(&self, object_id: &str) -> bool {
        match self {
            Self::ObjectIds(object_ids) => object_ids.contains(object_id),
            Self::Ranges(targets) => targets.iter().any(|target| target.matches(object_id)),
        }
    }
}

struct DeleteTableVersionsMutation {
    target: DeleteTableVersionsTarget,
    deleted_count: i64,
}

impl ManifestStreamMutation for DeleteTableVersionsMutation {
    type Output = i64;

    fn process_existing_row(
        &mut self,
        row: ManifestRowValue,
        output: &mut ManifestBatchBuilder,
        index_data: &mut ManifestIndexAccumulator,
    ) -> Result<()> {
        if row.object_type == ObjectType::TableVersion && self.target.matches(&row.object_id) {
            self.deleted_count += 1;
            return Ok(());
        }

        output.append(
            index_data,
            ManifestOutputRow {
                object_id: &row.object_id,
                object_type: row.object_type,
                location: row.location.as_deref(),
                metadata: row.metadata.as_deref(),
            },
        )
    }

    fn append_rows(
        &mut self,
        _output: &mut ManifestBatchBuilder,
        _index_data: &mut ManifestIndexAccumulator,
    ) -> Result<()> {
        Ok(())
    }

    fn finish(&self) -> CopyOnWriteMutation<Self::Output> {
        if self.deleted_count > 0 {
            CopyOnWriteMutation::updated(self.deleted_count)
        } else {
            CopyOnWriteMutation::unchanged(0)
        }
    }
}

/// Information about a namespace stored in the manifest
#[derive(Debug, Clone)]
pub struct NamespaceInfo {
    pub namespace: Vec<String>,
    pub name: String,
    pub metadata: Option<HashMap<String, String>>,
}

/// A wrapper around a Dataset that provides concurrent access.
///
/// This can be cloned cheaply. It supports concurrent reads or exclusive writes.
/// The manifest dataset is always kept strongly consistent by reloading on each read.
#[derive(Debug, Clone)]
pub struct DatasetConsistencyWrapper(Arc<RwLock<Dataset>>);

impl DatasetConsistencyWrapper {
    /// Create a new wrapper with the given dataset.
    pub fn new(dataset: Dataset) -> Self {
        Self(Arc::new(RwLock::new(dataset)))
    }

    /// Get an immutable reference to the dataset.
    /// Always reloads to ensure strong consistency.
    pub async fn get(&self) -> Result<DatasetReadGuard<'_>> {
        self.reload().await?;
        Ok(DatasetReadGuard {
            guard: self.0.read().await,
        })
    }

    /// Get the cached dataset without reloading.
    /// Use when the caller knows the dataset is already at an acceptable version
    /// (e.g., under a mutation lock where we just committed or are about to detect
    /// conflicts at commit time).
    pub async fn get_cached(&self) -> DatasetReadGuard<'_> {
        DatasetReadGuard {
            guard: self.0.read().await,
        }
    }

    /// Reload the dataset and return a reference.
    /// Use on retry paths where we know the version is stale.
    pub async fn get_refreshed(&self) -> Result<DatasetReadGuard<'_>> {
        self.reload().await?;
        Ok(DatasetReadGuard {
            guard: self.0.read().await,
        })
    }

    /// Get a mutable reference to the dataset.
    /// Always reloads to ensure strong consistency.
    pub async fn get_mut(&self) -> Result<DatasetWriteGuard<'_>> {
        self.reload().await?;
        Ok(DatasetWriteGuard {
            guard: self.0.write().await,
        })
    }

    /// Provide a known latest version of the dataset.
    ///
    /// This is usually done after some write operation, which inherently will
    /// have the latest version.
    pub async fn set_latest(&self, dataset: Dataset) {
        let mut write_guard = self.0.write().await;
        if dataset.manifest().version > write_guard.manifest().version {
            *write_guard = dataset;
        }
    }

    /// Reload the dataset to the latest version.
    async fn reload(&self) -> Result<()> {
        // First check if we need to reload (with read lock)
        let read_guard = self.0.read().await;
        let dataset_uri = read_guard.uri().to_string();
        let current_version = read_guard.version().version;
        log::debug!(
            "Reload starting for uri={}, current_version={}",
            dataset_uri,
            current_version
        );
        let latest_version = read_guard.latest_version_id().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to get latest version: {:?}", e),
            })
        })?;
        log::debug!(
            "Reload got latest_version={} for uri={}, current_version={}",
            latest_version,
            dataset_uri,
            current_version
        );
        drop(read_guard);

        // If already up-to-date, return early
        if latest_version == current_version {
            log::debug!("Already up-to-date for uri={}", dataset_uri);
            return Ok(());
        }

        // Need to reload, acquire write lock
        let mut write_guard = self.0.write().await;

        // Double-check after acquiring write lock (someone else might have reloaded)
        let latest_version = write_guard.latest_version_id().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to get latest version: {:?}", e),
            })
        })?;

        if latest_version != write_guard.version().version {
            write_guard.checkout_latest().await.map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to checkout latest: {:?}", e),
                })
            })?;
        }

        Ok(())
    }
}

pub struct DatasetReadGuard<'a> {
    guard: RwLockReadGuard<'a, Dataset>,
}

impl Deref for DatasetReadGuard<'_> {
    type Target = Dataset;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

pub struct DatasetWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, Dataset>,
}

impl Deref for DatasetWriteGuard<'_> {
    type Target = Dataset;

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl DerefMut for DatasetWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

/// Manifest-based namespace implementation
///
/// Uses a special `__manifest` Lance table to track tables and nested namespaces.
pub struct ManifestNamespace {
    root: String,
    manifest_table_name: String,
    manifest_branch: Option<String>,
    storage_options: Option<HashMap<String, String>>,
    session: Option<Arc<Session>>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    manifest_dataset: DatasetConsistencyWrapper,
    /// Whether directory listing is enabled in dual mode
    /// If true, root namespace tables use {table_name}.lance naming
    /// If false, they use namespace-prefixed names
    dir_listing_enabled: bool,
    /// When true, copy-on-write manifest rewrites build replacement indices
    /// (BTree, Bitmap, FTS) inline during the overwrite. When false, only the
    /// data file is rewritten and indices are expected to be created offline.
    inline_optimization_enabled: bool,
    /// Number of retries for commit operations on the manifest table.
    /// If None, defaults to [`lance_table::io::commit::CommitConfig`] default (20).
    commit_retries: Option<u32>,
    /// Serialize manifest mutations within a single namespace instance so concurrent
    /// create/drop calls do not compete with each other on the same in-memory snapshot.
    manifest_mutation_lock: Arc<Mutex<()>>,
}

impl std::fmt::Debug for ManifestNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManifestNamespace")
            .field("root", &self.root)
            .field("manifest_table_name", &self.manifest_table_name)
            .field("manifest_branch", &self.manifest_branch)
            .field("storage_options", &self.storage_options)
            .field("dir_listing_enabled", &self.dir_listing_enabled)
            .field(
                "inline_optimization_enabled",
                &self.inline_optimization_enabled,
            )
            .finish()
    }
}

/// Convert a Lance commit error to an appropriate namespace error.
///
/// Maps lance commit errors to namespace errors:
/// - `CommitConflict`: version collision retries exhausted -> Throttling (safe to retry)
/// - `TooMuchWriteContention`: RetryableCommitConflict (semantic conflict) retries exhausted -> ConcurrentModification
/// - `IncompatibleTransaction`: incompatible concurrent change -> ConcurrentModification
/// - Errors containing "matched/duplicate/already exists": ConcurrentModification (from WhenMatched::Fail)
/// - Other errors: IO error with the operation description
fn convert_lance_commit_error(e: &LanceError, operation: &str, object_id: Option<&str>) -> Error {
    match e {
        // Safe-to-retry commit conflicts exhausted retries -> Throttling.
        LanceError::CommitConflict { .. } | LanceError::RetryableCommitConflict { .. } => {
            NamespaceError::Throttling {
                message: format!("Too many concurrent writes, please retry later: {:?}", e),
            }
            .into()
        }
        // TooMuchWriteContention: RetryableCommitConflict (semantic conflict) retries exhausted -> ConcurrentModification
        // IncompatibleTransaction: incompatible concurrent change -> ConcurrentModification
        LanceError::TooMuchWriteContention { .. } | LanceError::IncompatibleTransaction { .. } => {
            let message = if let Some(id) = object_id {
                format!(
                    "Object '{}' was concurrently modified by another operation: {:?}",
                    id, e
                )
            } else {
                format!(
                    "Object was concurrently modified by another operation: {:?}",
                    e
                )
            };
            NamespaceError::ConcurrentModification { message }.into()
        }
        // Other errors: check message for semantic conflicts (matched/duplicate from WhenMatched::Fail)
        _ => {
            let error_msg = e.to_string();
            if error_msg.contains("matched")
                || error_msg.contains("duplicate")
                || error_msg.contains("already exists")
            {
                let message = if let Some(id) = object_id {
                    format!(
                        "Object '{}' was concurrently created by another operation: {:?}",
                        id, e
                    )
                } else {
                    format!(
                        "Object was concurrently created by another operation: {:?}",
                        e
                    )
                };
                return NamespaceError::ConcurrentModification { message }.into();
            }
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("{}: {:?}", operation, e),
            })
        }
    }
}

impl ManifestNamespace {
    /// Create a new ManifestNamespace from an existing DirectoryNamespace
    #[allow(clippy::too_many_arguments)]
    pub async fn from_directory(
        root: String,
        storage_options: Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        dir_listing_enabled: bool,
        inline_optimization_enabled: bool,
        commit_retries: Option<u32>,
        table_version_storage_enabled: bool,
    ) -> Result<Self> {
        Self::from_directory_with_manifest_table(
            root,
            MANIFEST_TABLE_NAME.to_string(),
            None,
            storage_options,
            session,
            object_store,
            base_path,
            dir_listing_enabled,
            inline_optimization_enabled,
            commit_retries,
            table_version_storage_enabled,
        )
        .await
    }

    /// Create a namespace using a specific Lance table to store manifest rows.
    ///
    /// `root` remains the catalog root used for user table locations. The
    /// `manifest_table_name` controls only the system table that stores catalog rows.
    #[allow(clippy::too_many_arguments)]
    pub async fn from_directory_with_manifest_table(
        root: String,
        manifest_table_name: String,
        manifest_branch: Option<String>,
        storage_options: Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        dir_listing_enabled: bool,
        inline_optimization_enabled: bool,
        commit_retries: Option<u32>,
        table_version_storage_enabled: bool,
    ) -> Result<Self> {
        let manifest_dataset = Self::ensure_manifest_table_up_to_date(
            &root,
            &manifest_table_name,
            manifest_branch.as_deref(),
            &storage_options,
            session.clone(),
            table_version_storage_enabled,
        )
        .await?;

        Ok(Self {
            root,
            manifest_table_name,
            manifest_branch,
            storage_options,
            session,
            object_store,
            base_path,
            manifest_dataset,
            dir_listing_enabled,
            inline_optimization_enabled,
            commit_retries,
            manifest_mutation_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Build object ID from namespace path and name
    pub fn build_object_id(namespace: &[String], name: &str) -> String {
        if namespace.is_empty() {
            name.to_string()
        } else {
            let mut id = namespace.join(DELIMITER);
            id.push_str(DELIMITER);
            id.push_str(name);
            id
        }
    }

    /// Parse object ID into namespace path and name
    pub fn parse_object_id(object_id: &str) -> (Vec<String>, String) {
        let parts: Vec<&str> = object_id.split(DELIMITER).collect();
        if parts.len() == 1 {
            (Vec::new(), parts[0].to_string())
        } else {
            let namespace = parts[..parts.len() - 1]
                .iter()
                .map(|s| s.to_string())
                .collect();
            let name = parts[parts.len() - 1].to_string();
            (namespace, name)
        }
    }

    /// Split an object ID (vec of strings) into namespace and table name
    pub fn split_object_id(object_id: &[String]) -> (Vec<String>, String) {
        if object_id.len() == 1 {
            (vec![], object_id[0].clone())
        } else {
            (
                object_id[..object_id.len() - 1].to_vec(),
                object_id[object_id.len() - 1].clone(),
            )
        }
    }

    /// Convert an ID (vec of strings) to an object_id string
    pub fn str_object_id(object_id: &[String]) -> String {
        object_id.join(DELIMITER)
    }

    fn format_table_id(table_id: &[String]) -> String {
        format!("table id '{}'", Self::str_object_id(table_id))
    }

    /// Format a version number as a zero-padded lexicographically sortable string.
    ///
    /// Versions are stored as 20-digit zero-padded integers (e.g., `00000000000000000001`
    /// for version 1) so that string-based range queries and sorting work correctly.
    pub fn format_table_version(version: i64) -> String {
        format!("{:020}", version)
    }

    /// Build the object_id for a table version entry.
    ///
    /// Format: `{table_object_id}${zero_padded_version}`
    pub fn build_version_object_id(table_object_id: &str, version: i64) -> String {
        format!(
            "{}{}{}",
            table_object_id,
            DELIMITER,
            Self::format_table_version(version)
        )
    }

    fn build_version_object_id_prefix(table_object_id: &str) -> String {
        format!("{}{}", table_object_id, DELIMITER)
    }

    fn normalize_table_version_ranges(ranges: &[(i64, i64)]) -> Vec<(i64, i64)> {
        let mut ranges = ranges
            .iter()
            .copied()
            .filter(|(start, end)| start <= end)
            .collect::<Vec<_>>();
        ranges.sort_unstable();

        let mut merged: Vec<(i64, i64)> = Vec::with_capacity(ranges.len());
        for (start, end) in ranges {
            if let Some((_, last_end)) = merged.last_mut()
                && start <= last_end.saturating_add(1)
            {
                *last_end = (*last_end).max(end);
                continue;
            }
            merged.push((start, end));
        }
        merged
    }

    /// Parse a version number from the version suffix of a table version object_id.
    ///
    /// The object_id is formatted as `{table_id}${zero_padded_version}`.
    pub fn parse_version_from_object_id(object_id: &str) -> Option<i64> {
        let (_namespace, name) = Self::parse_object_id(object_id);
        name.parse::<i64>().ok()
    }

    /// Generate a new directory name in format: `<hash>_<object_id>`
    /// The hash is used to (1) optimize object store throughput,
    /// (2) have high enough entropy in a short period of time to prevent issues like
    /// failed table creation, delete and create new table of the same name, etc.
    /// The object_id is added after the hash to ensure
    /// dir name uniqueness and make debugging easier.
    pub fn generate_dir_name(object_id: &str) -> String {
        // Generate a random number for uniqueness
        let random_num: u64 = rand::random();

        // Create hash from random number + object_id
        let mut hasher = DefaultHasher::new();
        random_num.hash(&mut hasher);
        object_id.hash(&mut hasher);
        let hash = hasher.finish();

        // Format as lowercase hex (8 characters - sufficient entropy for uniqueness)
        format!("{:08x}_{}", (hash & 0xFFFFFFFF) as u32, object_id)
    }

    /// Construct a full URI from root and relative location
    pub(crate) fn construct_full_uri(root: &str, relative_location: &str) -> Result<String> {
        let mut base_url = lance_io::object_store::uri_to_url(root)?;

        // Ensure the base URL has a trailing slash so that path segment mutation
        // appends rather than replaces the last path segment.
        // Without this fix, appending "table.lance" to "s3://bucket/path/subdir"
        // would incorrectly produce "s3://bucket/path/table.lance" (missing subdir).
        if !base_url.path().ends_with('/') {
            base_url.set_path(&format!("{}/", base_url.path()));
        }

        let mut full_url = base_url.clone();
        full_url
            .path_segments_mut()
            .map_err(|_| {
                lance_core::Error::from(NamespaceError::InvalidInput {
                    message: format!("Cannot modify path segments for URI '{}'", root),
                })
            })?
            .pop_if_empty()
            .extend(
                relative_location
                    .split('/')
                    .filter(|segment| !segment.is_empty()),
            );

        // Clear any query string to avoid trailing "?" in the URL.
        // Use set_query(None) instead of set_query("") because the latter
        // would still add a trailing '?' to the URL when serialized.
        full_url.set_query(None);

        Ok(full_url.to_string())
    }

    /// Get the manifest schema
    fn manifest_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            // Set unenforced primary key on object_id for bloom filter conflict detection
            Field::new("object_id", DataType::Utf8, false).with_metadata(
                [(
                    LANCE_UNENFORCED_PRIMARY_KEY_POSITION.to_string(),
                    "0".to_string(),
                )]
                .into_iter()
                .collect(),
            ),
            Field::new("object_type", DataType::Utf8, false),
            Field::new("location", DataType::Utf8, true),
            json_field("metadata", true),
        ]))
    }

    /// Get a scanner for the manifest dataset
    async fn manifest_scanner(&self) -> Result<Scanner> {
        let dataset_guard = self.manifest_dataset.get().await?;
        Ok(dataset_guard.scan())
    }

    /// Helper to execute a scanner and collect results into a Vec
    async fn execute_scanner(scanner: Scanner) -> Result<Vec<RecordBatch>> {
        let mut stream = scanner.try_into_stream().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to create stream: {:?}", e),
            })
        })?;

        let mut batches = Vec::new();
        while let Some(batch) = stream.next().await {
            batches.push(batch.map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to read batch: {:?}", e),
                })
            })?);
        }

        Ok(batches)
    }

    /// Helper to get a string column from a record batch
    fn get_string_column<'a>(batch: &'a RecordBatch, column_name: &str) -> Result<&'a StringArray> {
        let column = batch.column_by_name(column_name).ok_or_else(|| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Column '{}' not found", column_name),
            })
        })?;
        column
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Column '{}' is not a string array", column_name),
                })
            })
    }

    fn required_string_value<'a>(
        array: &'a StringArray,
        row: usize,
        column_name: &str,
    ) -> Result<&'a str> {
        if array.is_null(row) {
            return Err(NamespaceError::Internal {
                message: format!("Manifest column '{}' has null at row {}", column_name, row),
            }
            .into());
        }
        Ok(array.value(row))
    }

    fn optional_string_value(array: &StringArray, row: usize) -> Option<String> {
        (!array.is_null(row)).then(|| array.value(row).to_string())
    }

    fn metadata_column_values(
        batch: &RecordBatch,
        column_name: &str,
    ) -> Result<Vec<Option<String>>> {
        let column = batch.column_by_name(column_name).ok_or_else(|| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Column '{}' not found", column_name),
            })
        })?;

        if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
            return Ok((0..array.len())
                .map(|row| Self::optional_string_value(array, row))
                .collect());
        }

        if let Some(array) = column.as_any().downcast_ref::<LargeStringArray>() {
            return Ok((0..array.len())
                .map(|row| (!array.is_null(row)).then(|| array.value(row).to_string()))
                .collect());
        }

        if let Some(array) = column.as_any().downcast_ref::<LargeBinaryArray>() {
            return Ok((0..array.len())
                .map(|row| (!array.is_null(row)).then(|| decode_json(array.value(row))))
                .collect());
        }

        Err(NamespaceError::Internal {
            message: format!(
                "Column '{}' is not a supported metadata array: {:?}",
                column_name,
                column.data_type()
            ),
        }
        .into())
    }

    fn projected_schema(dataset: &Dataset) -> Result<SchemaRef> {
        let projected_columns = ["object_id", "object_type", "location", "metadata"];
        let lance_schema = dataset.schema();
        let fields: Vec<_> = projected_columns
            .iter()
            .filter_map(|name| {
                let f = lance_schema.field(name)?;
                Some(Field::new(*name, f.data_type(), f.nullable))
            })
            .collect();
        Ok(Arc::new(ArrowSchema::new(fields)))
    }

    async fn manifest_projected_stream(dataset: &Dataset) -> Result<SendableRecordBatchStream> {
        // Use the dataset's own schema so that old datasets with Utf8 metadata
        // (instead of JSON/LargeBinary) stream correctly. The downstream
        // rewrite path (metadata_column_values) handles both column types.
        let schema = Self::projected_schema(dataset)?;
        let mut scanner = dataset.scan();
        scanner
            .project(&["object_id", "object_type", "location", "metadata"])
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to project manifest columns: {:?}", e),
                })
            })?;
        let stream = scanner.try_into_stream().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to create manifest stream: {:?}", e),
            })
        })?;
        let stream = stream.map_err(|err| DataFusionError::External(Box::new(err)));
        Ok(Box::pin(DatafusionRecordBatchStreamAdapter::new(
            schema,
            stream.fuse(),
        )))
    }

    fn manifest_rewrite_commit_retries(&self) -> u32 {
        self.commit_retries
            .unwrap_or(DEFAULT_MANIFEST_REWRITE_COMMIT_RETRIES)
    }

    fn value_row_id_schema(value_field: Field) -> SchemaRef {
        Arc::new(ArrowSchema::new(vec![
            value_field,
            Field::new(ROW_ID, DataType::UInt64, false),
        ]))
    }

    fn metadata_index_schema() -> SchemaRef {
        Self::value_row_id_schema(json_field(VALUE_COLUMN_NAME, true))
    }

    fn metadata_index_stream(
        metadata_values: Vec<Option<String>>,
        metadata_row_ids: Vec<u64>,
    ) -> SendableRecordBatchStream {
        let schema = Self::metadata_index_schema();
        let stream_schema = schema.clone();
        let stream = stream::unfold(
            (
                metadata_values.into_iter().zip(metadata_row_ids).peekable(),
                false,
                schema,
            ),
            |state| async move {
                let (mut iter, emitted, schema) = state;
                let mut values = Vec::with_capacity(MANIFEST_INDEX_BATCH_SIZE);
                let mut row_ids = Vec::with_capacity(MANIFEST_INDEX_BATCH_SIZE);
                for _ in 0..MANIFEST_INDEX_BATCH_SIZE {
                    let Some((value, row_id)) = iter.next() else {
                        break;
                    };
                    values.push(value);
                    row_ids.push(row_id);
                }
                if values.is_empty() {
                    if emitted {
                        None
                    } else {
                        let batch = Self::json_row_id_batch(schema.clone(), values, row_ids)
                            .map_err(|err| DataFusionError::External(Box::new(err)));
                        Some((batch, (iter, true, schema)))
                    }
                } else {
                    let batch = Self::json_row_id_batch(schema.clone(), values, row_ids)
                        .map_err(|err| DataFusionError::External(Box::new(err)));
                    Some((batch, (iter, true, schema)))
                }
            },
        );
        Box::pin(DatafusionRecordBatchStreamAdapter::new(
            stream_schema,
            stream.fuse(),
        ))
    }

    fn metadata_index_params() -> ScalarIndexParams {
        ScalarIndexParams::for_builtin(BuiltinIndexType::Inverted)
            .with_params(&InvertedIndexParams::default().lance_tokenizer("json".to_string()))
    }

    fn string_row_id_batch(
        schema: SchemaRef,
        values: Vec<String>,
        row_ids: Vec<u64>,
    ) -> Result<RecordBatch> {
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(values)),
                Arc::new(UInt64Array::from(row_ids)),
            ],
        )
        .map_err(Into::into)
    }

    fn json_row_id_batch(
        schema: SchemaRef,
        values: Vec<Option<String>>,
        row_ids: Vec<u64>,
    ) -> Result<RecordBatch> {
        let json_array = Arc::new(
            JsonArray::try_from_iter(values.iter().map(|v| v.as_deref()))
                .map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!("Failed to encode metadata index JSON: {}", e),
                    })
                })?
                .into_inner(),
        );
        RecordBatch::try_new(
            schema,
            vec![json_array, Arc::new(UInt64Array::from(row_ids))],
        )
        .map_err(Into::into)
    }

    fn object_id_index_stream(object_ids: BTreeMap<Arc<str>, u64>) -> SendableRecordBatchStream {
        let schema =
            Self::value_row_id_schema(Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false));
        let stream_schema = schema.clone();
        let stream = stream::unfold(
            (object_ids.into_iter(), false, schema),
            |state| async move {
                let (mut iter, emitted, schema) = state;
                let mut values = Vec::with_capacity(MANIFEST_INDEX_BATCH_SIZE);
                let mut row_ids = Vec::with_capacity(MANIFEST_INDEX_BATCH_SIZE);
                for _ in 0..MANIFEST_INDEX_BATCH_SIZE {
                    let Some((value, row_id)) = iter.next() else {
                        break;
                    };
                    values.push(value.to_string());
                    row_ids.push(row_id);
                }
                if values.is_empty() {
                    if emitted {
                        None
                    } else {
                        let batch = Self::string_row_id_batch(schema.clone(), values, row_ids)
                            .map_err(|err| DataFusionError::External(Box::new(err)));
                        Some((batch, (iter, true, schema)))
                    }
                } else {
                    let batch = Self::string_row_id_batch(schema.clone(), values, row_ids)
                        .map_err(|err| DataFusionError::External(Box::new(err)));
                    Some((batch, (iter, true, schema)))
                }
            },
        );
        Box::pin(DatafusionRecordBatchStreamAdapter::new(
            stream_schema,
            stream.fuse(),
        ))
    }

    fn object_type_index_stream(
        object_types: BTreeMap<&'static str, RoaringBitmap>,
    ) -> SendableRecordBatchStream {
        let schema =
            Self::value_row_id_schema(Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false));
        let stream_schema = schema.clone();
        let entries = object_types
            .into_iter()
            .map(|(value, bitmap)| {
                (
                    value,
                    Box::new(bitmap.into_iter()) as Box<dyn Iterator<Item = u32> + Send>,
                )
            })
            .collect::<Vec<_>>()
            .into_iter();
        let stream = stream::unfold(
            (entries, None, false, schema),
            |(mut entries, mut current, emitted, schema)| async move {
                let mut values = Vec::with_capacity(MANIFEST_INDEX_BATCH_SIZE);
                let mut row_ids = Vec::with_capacity(MANIFEST_INDEX_BATCH_SIZE);
                while values.len() < MANIFEST_INDEX_BATCH_SIZE {
                    if current.is_none() {
                        current = entries.next();
                    }
                    let Some((value, iter)) = current.as_mut() else {
                        break;
                    };
                    if let Some(row_id) = iter.next() {
                        values.push((*value).to_string());
                        row_ids.push(u64::from(row_id));
                    } else {
                        current = None;
                    }
                }

                if values.is_empty() {
                    if emitted {
                        None
                    } else {
                        let batch = Self::string_row_id_batch(schema.clone(), values, row_ids)
                            .map_err(|err| DataFusionError::External(Box::new(err)));
                        Some((batch, (entries, current, true, schema)))
                    }
                } else {
                    let batch = Self::string_row_id_batch(schema.clone(), values, row_ids)
                        .map_err(|err| DataFusionError::External(Box::new(err)));
                    Some((batch, (entries, current, true, schema)))
                }
            },
        );
        Box::pin(DatafusionRecordBatchStreamAdapter::new(
            stream_schema,
            stream.fuse(),
        ))
    }

    async fn train_manifest_index(
        dataset: &Dataset,
        input: ManifestIndexBuildInput,
        index_uuid: Uuid,
    ) -> Result<ManifestTrainedIndex> {
        let index_store = LanceIndexStore::from_dataset_for_new(dataset, &index_uuid.to_string())?;
        let registry = IndexPluginRegistry::with_default_plugins();
        let plugin = registry.get_plugin_by_name(&input.params.index_type)?;
        let training_request = plugin
            .new_training_request(input.params.params.as_deref().unwrap_or("{}"), &input.field)?;
        let created_index = plugin
            .train_index(
                input.stream,
                &index_store,
                training_request,
                None,
                noop_progress(),
            )
            .await?;

        Ok(ManifestTrainedIndex {
            index_name: input.index_name,
            column_name: input.column_name,
            uuid: index_uuid,
            created_index,
        })
    }

    fn manifest_index_metadata(
        lance_schema: &lance_core::datatypes::Schema,
        fragment_bitmap: &RoaringBitmap,
        dataset_version: u64,
        trained_index: ManifestTrainedIndex,
    ) -> Result<IndexMetadata> {
        Ok(IndexMetadata {
            uuid: trained_index.uuid,
            fields: vec![lance_schema.field_id(trained_index.column_name)?],
            name: trained_index.index_name.to_string(),
            dataset_version,
            fragment_bitmap: Some(fragment_bitmap.clone()),
            index_details: Some(Arc::new(trained_index.created_index.index_details)),
            index_version: trained_index.created_index.index_version as i32,
            created_at: None,
            base_id: None,
            files: trained_index.created_index.files,
        })
    }

    async fn build_manifest_index(
        dataset: &Dataset,
        lance_schema: &lance_core::datatypes::Schema,
        input: ManifestIndexBuildInput,
        fragment_bitmap: &RoaringBitmap,
        dataset_version: u64,
        index_uuid: Uuid,
    ) -> Result<IndexMetadata> {
        let trained_index = Self::train_manifest_index(dataset, input, index_uuid).await?;
        Self::manifest_index_metadata(
            lance_schema,
            fragment_bitmap,
            dataset_version,
            trained_index,
        )
    }

    async fn build_manifest_indices(
        dataset: &Dataset,
        manifest: &Manifest,
        index_data: ManifestIndexAccumulator,
        index_uuids: &mut Vec<Uuid>,
    ) -> Result<Vec<IndexMetadata>> {
        let num_fragments = manifest.fragments.len();
        let fragment_bitmap = RoaringBitmap::from_iter(0..num_fragments as u32);
        let schema = &manifest.schema;
        let ManifestIndexAccumulator {
            object_ids,
            object_types,
            metadata_values,
            metadata_row_ids,
            ..
        } = index_data;
        let object_id_uuid = Uuid::new_v4();
        let object_type_uuid = Uuid::new_v4();
        let metadata_uuid = Uuid::new_v4();
        index_uuids.extend([object_id_uuid, object_type_uuid, metadata_uuid]);

        let dataset_version = manifest.version;
        let object_id_index_fut = Self::build_manifest_index(
            dataset,
            schema,
            ManifestIndexBuildInput {
                index_name: OBJECT_ID_INDEX_NAME,
                column_name: "object_id",
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
                field: Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
                stream: Self::object_id_index_stream(object_ids),
            },
            &fragment_bitmap,
            dataset_version,
            object_id_uuid,
        );
        let object_type_index_fut = Self::build_manifest_index(
            dataset,
            schema,
            ManifestIndexBuildInput {
                index_name: OBJECT_TYPE_INDEX_NAME,
                column_name: "object_type",
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
                field: Field::new(VALUE_COLUMN_NAME, DataType::Utf8, false),
                stream: Self::object_type_index_stream(object_types),
            },
            &fragment_bitmap,
            dataset_version,
            object_type_uuid,
        );
        let metadata_index_fut = Self::build_manifest_index(
            dataset,
            schema,
            ManifestIndexBuildInput {
                index_name: METADATA_INDEX_NAME,
                column_name: "metadata",
                params: Self::metadata_index_params(),
                field: json_field(VALUE_COLUMN_NAME, true),
                stream: Self::metadata_index_stream(metadata_values, metadata_row_ids),
            },
            &fragment_bitmap,
            dataset_version,
            metadata_uuid,
        );

        let (object_id_index, object_type_index, metadata_index) = futures::join!(
            object_id_index_fut,
            object_type_index_fut,
            metadata_index_fut
        );

        Ok(vec![object_id_index?, object_type_index?, metadata_index?])
    }

    fn lock_manifest_rewrite_shared<M: ManifestStreamMutation>(
        shared: &Arc<StdMutex<ManifestRewriteShared<M>>>,
    ) -> Result<StdMutexGuard<'_, ManifestRewriteShared<M>>> {
        shared.lock().map_err(|_| {
            lance_core::Error::from(NamespaceError::Internal {
                message: "Manifest rewrite state mutex was poisoned".to_string(),
            })
        })
    }

    fn set_manifest_rewrite_error<M: ManifestStreamMutation>(
        shared: &Arc<StdMutex<ManifestRewriteShared<M>>>,
        err: LanceError,
    ) {
        match shared.lock() {
            Ok(mut guard) => {
                guard.error = Some(err);
            }
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.error = Some(err);
            }
        }
    }

    fn take_manifest_rewrite_error<M: ManifestStreamMutation>(
        shared: &Arc<StdMutex<ManifestRewriteShared<M>>>,
    ) -> Result<Option<LanceError>> {
        let mut guard = Self::lock_manifest_rewrite_shared(shared)?;
        Ok(guard.error.take())
    }

    fn process_manifest_rewrite_batch<M: ManifestStreamMutation>(
        batch: RecordBatch,
        shared: &Arc<StdMutex<ManifestRewriteShared<M>>>,
    ) -> Result<Option<RecordBatch>> {
        let object_ids = Self::get_string_column(&batch, "object_id")?;
        let object_types = Self::get_string_column(&batch, "object_type")?;
        let locations = Self::get_string_column(&batch, "location")?;
        let metadatas = Self::metadata_column_values(&batch, "metadata")?;
        let mut output = ManifestBatchBuilder::new();
        let mut guard = Self::lock_manifest_rewrite_shared(shared)?;
        let mut index_data = guard.index_data.take().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::Internal {
                message: "Manifest rewrite index state is unavailable".to_string(),
            })
        })?;
        for (row, metadata) in metadatas.into_iter().enumerate().take(batch.num_rows()) {
            let row_value = ManifestRowValue {
                object_id: Self::required_string_value(object_ids, row, "object_id")?.to_string(),
                object_type: ObjectType::parse(Self::required_string_value(
                    object_types,
                    row,
                    "object_type",
                )?)?,
                location: Self::optional_string_value(locations, row),
                metadata,
            };
            guard
                .mutation
                .process_existing_row(row_value, &mut output, &mut index_data)?;
        }
        guard.index_data = Some(index_data);
        if output.is_empty() {
            return Ok(None);
        }
        Ok(Some(output.finish()?))
    }

    fn finish_manifest_rewrite_stream<M: ManifestStreamMutation>(
        shared: &Arc<StdMutex<ManifestRewriteShared<M>>>,
    ) -> Result<Option<RecordBatch>> {
        let mut output = ManifestBatchBuilder::new();
        let mut guard = Self::lock_manifest_rewrite_shared(shared)?;
        let mut index_data = guard.index_data.take().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::Internal {
                message: "Manifest rewrite index state is unavailable".to_string(),
            })
        })?;
        guard.mutation.append_rows(&mut output, &mut index_data)?;
        let result = guard.mutation.finish();
        let force_empty_batch = index_data.row_count == 0;
        guard.result = Some(result);
        guard.index_data = Some(index_data);
        if output.is_empty() && !force_empty_batch {
            Ok(None)
        } else {
            Ok(Some(output.finish()?))
        }
    }

    fn manifest_rewrite_output_stream<M: ManifestStreamMutation + 'static>(
        source: SendableRecordBatchStream,
        shared: Arc<StdMutex<ManifestRewriteShared<M>>>,
    ) -> SendableRecordBatchStream {
        enum Phase {
            Source,
            Finish,
            Done,
        }

        let schema = Self::manifest_schema();
        let stream = stream::unfold(
            (source, shared, Phase::Source),
            |(mut source, shared, mut phase)| async move {
                loop {
                    match phase {
                        Phase::Source => match source.next().await {
                            Some(Ok(batch)) => {
                                match Self::process_manifest_rewrite_batch(batch, &shared) {
                                    Ok(Some(batch)) => {
                                        return Some((Ok(batch), (source, shared, phase)));
                                    }
                                    Ok(None) => continue,
                                    Err(err) => {
                                        let message = err.to_string();
                                        Self::set_manifest_rewrite_error(&shared, err);
                                        return Some((
                                            Err(DataFusionError::External(Box::new(
                                                std::io::Error::other(message),
                                            ))),
                                            (source, shared, Phase::Done),
                                        ));
                                    }
                                }
                            }
                            Some(Err(err)) => {
                                return Some((Err(err), (source, shared, Phase::Done)));
                            }
                            None => phase = Phase::Finish,
                        },
                        Phase::Finish => {
                            phase = Phase::Done;
                            match Self::finish_manifest_rewrite_stream(&shared) {
                                Ok(Some(batch)) => {
                                    return Some((Ok(batch), (source, shared, phase)));
                                }
                                Ok(None) => continue,
                                Err(err) => {
                                    let message = err.to_string();
                                    Self::set_manifest_rewrite_error(&shared, err);
                                    return Some((
                                        Err(DataFusionError::External(Box::new(
                                            std::io::Error::other(message),
                                        ))),
                                        (source, shared, Phase::Done),
                                    ));
                                }
                            }
                        }
                        Phase::Done => return None,
                    }
                }
            },
        );
        Box::pin(DatafusionRecordBatchStreamAdapter::new(
            schema,
            stream.fuse(),
        ))
    }

    fn take_manifest_rewrite_result<M: ManifestStreamMutation>(
        shared: &Arc<StdMutex<ManifestRewriteShared<M>>>,
    ) -> Result<(CopyOnWriteMutation<M::Output>, ManifestIndexAccumulator)> {
        let mut guard = Self::lock_manifest_rewrite_shared(shared)?;
        let result = guard.result.take().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::Internal {
                message: "Manifest rewrite stream did not finish".to_string(),
            })
        })?;
        let index_data = guard.index_data.take().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::Internal {
                message: "Manifest rewrite index state is unavailable".to_string(),
            })
        })?;
        Ok((result, index_data))
    }

    async fn rewrite_manifest<M, F>(
        &self,
        operation: &str,
        mut make_mutation: F,
    ) -> Result<M::Output>
    where
        M: ManifestStreamMutation + 'static,
        F: FnMut() -> M,
    {
        let _mutation_guard = self.manifest_mutation_lock.lock().await;
        let max_retries = self.manifest_rewrite_commit_retries();
        let mut retries = 0;

        let build_indices = self.inline_optimization_enabled;

        loop {
            // First attempt uses cached dataset (no I/O). Retries reload to pick up
            // the version that won the conflict.
            let dataset_guard = if retries == 0 {
                self.manifest_dataset.get_cached().await
            } else {
                self.manifest_dataset.get_refreshed().await?
            };
            let dataset = Arc::new(dataset_guard.clone());
            drop(dataset_guard);

            let source = Self::manifest_projected_stream(&dataset).await?;
            let shared = Arc::new(StdMutex::new(ManifestRewriteShared::new(make_mutation())));
            let output_stream = Self::manifest_rewrite_output_stream(source, shared.clone());
            let write_params = WriteParams {
                mode: WriteMode::Overwrite,
                session: self.session.clone(),
                max_rows_per_file: u32::MAX as usize,
                skip_auto_cleanup: true,
                ..WriteParams::default()
            };

            let transaction = match InsertBuilder::new(dataset.clone())
                .with_params(&write_params)
                .execute_uncommitted_stream(output_stream)
                .await
            {
                Ok(transaction) => transaction,
                Err(err) => {
                    if let Some(stream_err) = Self::take_manifest_rewrite_error(&shared)? {
                        return Err(stream_err);
                    }
                    return Err(convert_lance_commit_error(&err, operation, None));
                }
            };

            let (mutation, index_data) = Self::take_manifest_rewrite_result(&shared)?;
            if !mutation.has_changes {
                return Ok(mutation.result);
            }

            // Extract fragments and schema from the overwrite transaction
            let Operation::Overwrite {
                fragments, schema, ..
            } = transaction.operation
            else {
                return Err(NamespaceError::Internal {
                    message: "Manifest rewrite transaction is not an overwrite".to_string(),
                }
                .into());
            };
            let mut manifest =
                Manifest::new_from_previous(dataset.manifest(), schema, Arc::new(fragments));

            let indices = if build_indices {
                let mut index_uuids = vec![];
                Some(
                    Self::build_manifest_indices(&dataset, &manifest, index_data, &mut index_uuids)
                        .await?,
                )
            } else {
                None
            };

            let object_store = dataset.object_store(None).await?;
            let base_path = dataset.branch_location().path;
            let naming_scheme = dataset.manifest_location().naming_scheme;
            let config = ManifestWriteConfig::default();

            let result = write_manifest_file(
                &object_store,
                dataset.commit_handler(),
                &base_path,
                &mut manifest,
                indices,
                &config,
                naming_scheme,
                None,
            )
            .await;

            match result {
                Ok(_) => {
                    // Promote cache to the committed version. checkout_version reads
                    // the manifest we just wrote, which is typically served from the
                    // object-store HTTP cache and avoids a full reload cycle.
                    if let Ok(new_dataset) = dataset.checkout_version(manifest.version).await {
                        self.manifest_dataset.set_latest(new_dataset).await;
                    }
                    return Ok(mutation.result);
                }
                Err(CommitError::CommitConflict) if retries < max_retries => {
                    retries += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(10 * u64::from(retries)))
                        .await;
                }
                Err(err) => {
                    let lance_err: LanceError = err.into();
                    return Err(convert_lance_commit_error(&lance_err, operation, None));
                }
            }
        }
    }

    /// Check if the manifest contains an object with the given ID
    pub(crate) async fn manifest_contains_object(&self, object_id: &str) -> Result<bool> {
        let escaped_id = object_id.replace('\'', "''");
        let filter = format!("object_id = '{}'", escaped_id);

        let dataset_guard = self.manifest_dataset.get().await?;
        let mut scanner = dataset_guard.scan();

        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;

        // Project no columns and enable row IDs for count_rows to work
        scanner.project::<&str>(&[]).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to project: {:?}", e),
            })
        })?;

        scanner.with_row_id();

        let count = scanner.count_rows().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to count rows: {:?}", e),
            })
        })?;

        Ok(count > 0)
    }

    pub(crate) async fn count_objects_with_prefix(&self, object_id_prefix: &str) -> Result<u64> {
        let escaped_prefix = object_id_prefix.replace('\'', "''");
        let filter = format!("starts_with(object_id, '{}')", escaped_prefix);
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner.project::<&str>(&[]).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to project: {:?}", e),
            })
        })?;
        scanner.with_row_id();
        scanner.count_rows().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to count rows: {:?}", e),
            })
        })
    }

    pub(crate) async fn query_manifest_entry(
        &self,
        object_id: &str,
        object_type: ObjectType,
    ) -> Result<Option<ManifestEntry>> {
        let escaped_id = object_id.replace('\'', "''");
        let filter = format!(
            "object_id = '{}' AND object_type = '{}'",
            escaped_id,
            object_type.as_str()
        );
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner
            .project(&["object_id", "object_type", "location", "metadata"])
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to project: {:?}", e),
                })
            })?;
        let batches = Self::execute_scanner(scanner).await?;

        let mut found = None;
        let mut total_rows = 0;
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }
            total_rows += batch.num_rows();
            if total_rows > 1 {
                return Err(NamespaceError::Internal {
                    message: format!(
                        "Expected exactly 1 manifest entry with id '{}', found {}",
                        object_id, total_rows
                    ),
                }
                .into());
            }

            let object_ids = Self::get_string_column(&batch, "object_id")?;
            let object_types = Self::get_string_column(&batch, "object_type")?;
            let locations = Self::get_string_column(&batch, "location")?;
            let metadatas = Self::metadata_column_values(&batch, "metadata")?;
            found = Some(ManifestEntry {
                object_id: object_ids.value(0).to_string(),
                object_type: ObjectType::parse(object_types.value(0))?,
                location: Self::optional_string_value(locations, 0),
                metadata: metadatas[0].clone(),
            });
        }

        Ok(found)
    }

    pub(crate) async fn create_manifest_branch(&self, branch: &str) -> Result<()> {
        let dataset_guard = self.manifest_dataset.get().await?;
        let mut dataset = dataset_guard.clone();
        drop(dataset_guard);
        dataset
            .create_branch(branch, (self.manifest_branch.as_deref(), None), None)
            .await
            .map(|_| ())
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to create branch '{}' for manifest table '{}': {:?}",
                        branch, self.manifest_table_name, e
                    ),
                })
            })
    }

    /// Query the manifest for a table with the given object ID
    async fn query_manifest_for_table(&self, object_id: &str) -> Result<Option<TableInfo>> {
        let escaped_id = object_id.replace('\'', "''");
        let filter = format!("object_id = '{}' AND object_type = 'table'", escaped_id);
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner
            .project(&["object_id", "location", "metadata"])
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to project: {:?}", e),
                })
            })?;
        let batches = Self::execute_scanner(scanner).await?;

        let mut found_result: Option<TableInfo> = None;
        let mut total_rows = 0;

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            total_rows += batch.num_rows();
            if total_rows > 1 {
                return Err(NamespaceError::Internal {
                    message: format!(
                        "Expected exactly 1 table with id '{}', found {}",
                        object_id, total_rows
                    ),
                }
                .into());
            }

            let object_id_array = Self::get_string_column(&batch, "object_id")?;
            let location_array = Self::get_string_column(&batch, "location")?;
            let metadata_values = Self::metadata_column_values(&batch, "metadata")?;
            let location = location_array.value(0).to_string();
            let metadata = if let Some(metadata_str) = &metadata_values[0] {
                match serde_json::from_str::<HashMap<String, String>>(metadata_str) {
                    Ok(map) => Some(map),
                    Err(e) => {
                        return Err(NamespaceError::Internal {
                            message: format!(
                                "Failed to deserialize metadata for table '{}': {}",
                                object_id, e
                            ),
                        }
                        .into());
                    }
                }
            } else {
                None
            };
            let (namespace, name) = Self::parse_object_id(object_id_array.value(0));
            found_result = Some(TableInfo {
                namespace,
                name,
                location,
                metadata,
            });
        }

        Ok(found_result)
    }

    pub(crate) fn serialize_metadata(
        properties: Option<&HashMap<String, String>>,
        object_type: &str,
        object_id: &str,
    ) -> Result<Option<String>> {
        match properties {
            Some(properties) if !properties.is_empty() => {
                serde_json::to_string(properties).map(Some).map_err(|e| {
                    LanceError::from(NamespaceError::Internal {
                        message: format!(
                            "Failed to serialize {} metadata for '{}': {}",
                            object_type, object_id, e
                        ),
                    })
                })
            }
            _ => Ok(None),
        }
    }

    pub(crate) async fn path_has_actual_manifests(
        object_store: &ObjectStore,
        table_path: &Path,
    ) -> Result<bool> {
        let versions_path = table_path
            .clone()
            .join(lance_table::io::commit::VERSIONS_DIR);
        // `_versions/` should only contain manifest files, so probing the first entry is enough
        // to distinguish declared-only tables (empty `_versions/`) from created tables.
        Ok(object_store
            .list(Some(versions_path))
            .try_next()
            .await?
            .is_some())
    }

    pub(crate) fn relative_location_path(base_path: &Path, location: &str) -> Path {
        location
            .split('/')
            .filter(|segment| !segment.is_empty())
            .fold(base_path.clone(), |path, segment| path.join(segment))
    }

    async fn location_has_actual_manifests(&self, location: &str) -> Result<bool> {
        let table_path = Self::relative_location_path(&self.base_path, location);
        Self::path_has_actual_manifests(&self.object_store, &table_path).await
    }

    pub(crate) fn is_not_found_load_error(err: &LanceError) -> bool {
        match err {
            LanceError::NotFound { .. } => true,
            LanceError::IO { source, .. } => source
                .downcast_ref::<ObjectStoreError>()
                .is_some_and(|source| matches!(source, ObjectStoreError::NotFound { .. })),
            LanceError::DatasetNotFound { source, .. } => {
                source
                    .downcast_ref::<LanceError>()
                    .is_some_and(|source| matches!(source, LanceError::NotFound { .. }))
                    || source
                        .downcast_ref::<ObjectStoreError>()
                        .is_some_and(|source| matches!(source, ObjectStoreError::NotFound { .. }))
            }
            _ => false,
        }
    }

    /// List all table locations in the manifest (for root namespace only)
    /// Returns a set of table locations (e.g., "table_name.lance")
    pub async fn list_manifest_table_locations(&self) -> Result<std::collections::HashSet<String>> {
        let filter = "object_type = 'table' AND NOT contains(object_id, '$')";
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner.project(&["location"]).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to project: {:?}", e),
            })
        })?;

        let batches = Self::execute_scanner(scanner).await?;
        let mut locations = std::collections::HashSet::new();

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }
            let location_array = Self::get_string_column(&batch, "location")?;
            for i in 0..location_array.len() {
                locations.insert(location_array.value(i).to_string());
            }
        }

        Ok(locations)
    }

    /// Insert an entry into the manifest table
    async fn insert_into_manifest(
        &self,
        object_id: String,
        object_type: ObjectType,
        location: Option<String>,
    ) -> Result<()> {
        self.insert_into_manifest_with_metadata(vec![ManifestEntry {
            object_id,
            object_type,
            location,
            metadata: None,
        }])
        .await
    }

    /// Insert one or more entries into the manifest table with metadata.
    ///
    /// This is the unified entry point for both single and batch inserts.
    /// If any entry already exists (matching object_id), the entire batch fails.
    pub async fn insert_into_manifest_with_metadata(
        &self,
        entries: Vec<ManifestEntry>,
    ) -> Result<()> {
        self.merge_into_manifest_with_metadata(entries, WhenMatched::Fail)
            .await
    }

    pub(crate) async fn upsert_into_manifest_with_metadata(
        &self,
        entries: Vec<ManifestEntry>,
    ) -> Result<()> {
        self.merge_into_manifest_with_metadata(entries, WhenMatched::UpdateAll)
            .await
    }

    async fn merge_into_manifest_with_metadata(
        &self,
        entries: Vec<ManifestEntry>,
        when_matched: WhenMatched,
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        self.rewrite_manifest("Failed to overwrite manifest", || {
            UpsertManifestMutation::new(entries.clone(), when_matched.clone())
        })
        .await
    }

    /// Delete an entry from the manifest table
    pub async fn delete_from_manifest(&self, object_id: &str) -> Result<()> {
        let object_id = object_id.to_string();
        self.rewrite_manifest("Failed to delete from manifest", || DeleteObjectMutation {
            object_id: object_id.clone(),
            deleted: false,
        })
        .await
    }

    /// Query the manifest for all versions of a table, sorted by version.
    ///
    /// Returns a list of (version, metadata_json_string) tuples where metadata_json_string
    /// contains the full metadata JSON stored in the manifest (manifest_path, manifest_size,
    /// e_tag, naming_scheme).
    ///
    /// **Known limitation**: All matching rows are loaded into memory, sorted in Rust,
    /// and then truncated. For tables with a very large number of versions this may be
    /// expensive. Pushing sort/limit into the scan is not yet supported by Lance.
    pub async fn query_table_versions(
        &self,
        object_id: &str,
        descending: bool,
        limit: Option<i32>,
    ) -> Result<Vec<(i64, String)>> {
        let escaped_id = object_id.replace('\'', "''");
        // table_version object_ids are formatted as "{object_id}${zero_padded_version}"
        let filter = format!(
            "object_type = 'table_version' AND starts_with(object_id, '{}{}')",
            escaped_id, DELIMITER
        );
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner.project(&["object_id", "metadata"]).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to project: {:?}", e),
            })
        })?;
        let batches = Self::execute_scanner(scanner).await?;

        let mut versions: Vec<(i64, String)> = Vec::new();
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }
            let object_id_array = Self::get_string_column(&batch, "object_id")?;
            let metadata_values = Self::metadata_column_values(&batch, "metadata")?;
            for (i, metadata) in metadata_values.iter().enumerate().take(batch.num_rows()) {
                let oid = object_id_array.value(i);
                // Parse version from object_id
                if let Some(version) = Self::parse_version_from_object_id(oid) {
                    versions.push((version, metadata.clone().unwrap_or_default()));
                }
            }
        }

        if descending {
            versions.sort_by(|a, b| b.0.cmp(&a.0));
        } else {
            versions.sort_by(|a, b| a.0.cmp(&b.0));
        }

        if let Some(limit) = limit {
            versions.truncate(limit as usize);
        }

        Ok(versions)
    }

    /// Query the manifest for a specific version of a table.
    ///
    /// Returns the full metadata JSON string if found, which contains
    /// manifest_path, manifest_size, e_tag, and naming_scheme.
    ///
    pub async fn query_table_version(
        &self,
        object_id: &str,
        version: i64,
    ) -> Result<Option<String>> {
        let version_object_id = Self::build_version_object_id(object_id, version);
        self.query_table_version_by_object_id(&version_object_id)
            .await
    }

    /// Query a specific table version by its exact object_id.
    async fn query_table_version_by_object_id(
        &self,
        version_object_id: &str,
    ) -> Result<Option<String>> {
        let escaped_id = version_object_id.replace('\'', "''");
        let filter = format!(
            "object_id = '{}' AND object_type = 'table_version'",
            escaped_id
        );
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner.project(&["metadata"]).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to project: {:?}", e),
            })
        })?;
        let batches = Self::execute_scanner(scanner).await?;

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }
            let metadata_values = Self::metadata_column_values(&batch, "metadata")?;
            return Ok(metadata_values[0].clone());
        }

        Ok(None)
    }

    async fn delete_table_version_rows_by_object_ids(&self, object_ids: &[String]) -> Result<i64> {
        let object_ids = object_ids.iter().cloned().collect::<HashSet<_>>();
        self.rewrite_manifest("Failed to delete table versions from manifest", || {
            DeleteTableVersionsMutation {
                target: DeleteTableVersionsTarget::ObjectIds(object_ids.clone()),
                deleted_count: 0,
            }
        })
        .await
    }

    /// Delete table version entries from the manifest for a given table and version ranges.
    ///
    /// Each range is (start_version, end_version) inclusive. Deletes all matching
    /// `object_type = 'table_version'` entries whose object_id matches
    /// `{object_id}${zero_padded_version}`.
    ///
    /// Applies the ranges while streaming the manifest rewrite, without expanding
    /// sparse ranges into every possible version object id.
    pub async fn delete_table_versions(
        &self,
        object_id: &str,
        ranges: &[(i64, i64)],
    ) -> Result<i64> {
        self.batch_delete_table_versions_by_ranges(&[(object_id.to_string(), ranges.to_vec())])
            .await
    }

    /// Atomically delete table version entries from the manifest for multiple
    /// tables and version ranges.
    pub async fn batch_delete_table_versions_by_ranges(
        &self,
        table_ranges: &[(String, Vec<(i64, i64)>)],
    ) -> Result<i64> {
        let targets = table_ranges
            .iter()
            .filter_map(|(object_id, ranges)| {
                let ranges = Self::normalize_table_version_ranges(ranges);
                if ranges.is_empty() {
                    None
                } else {
                    Some(DeleteTableVersionRangeTarget {
                        object_id_prefix: Self::build_version_object_id_prefix(object_id),
                        ranges,
                    })
                }
            })
            .collect::<Vec<_>>();
        if targets.is_empty() {
            return Ok(0);
        }

        self.rewrite_manifest("Failed to delete table versions from manifest", || {
            DeleteTableVersionsMutation {
                target: DeleteTableVersionsTarget::Ranges(targets.clone()),
                deleted_count: 0,
            }
        })
        .await
    }

    /// Atomically delete table version entries from the manifest by their object_ids.
    ///
    /// This method supports multi-table transactional deletion: all specified
    /// object_ids (which may span multiple tables) are deleted in a single atomic
    /// copy-on-write manifest rewrite. Either all entries are removed or none are.
    ///
    /// Object IDs are formatted as `{table_id}${version}`.
    pub async fn batch_delete_table_versions_by_object_ids(
        &self,
        object_ids: &[String],
    ) -> Result<i64> {
        if object_ids.is_empty() {
            return Ok(0);
        }

        self.delete_table_version_rows_by_object_ids(object_ids)
            .await
    }

    /// Set a property flag in the __manifest table's metadata key-value map.
    ///
    /// This uses `dataset.update_metadata()` to persist the flag in the
    /// __manifest dataset's table metadata, rather than inserting a row.
    /// If the property already exists with the same value, this is a no-op.
    pub async fn set_property(&self, name: &str, value: &str) -> Result<()> {
        let _mutation_guard = self.manifest_mutation_lock.lock().await;
        let dataset_guard = self.manifest_dataset.get().await?;
        if dataset_guard.metadata().get(name) == Some(&value.to_string()) {
            return Ok(());
        }
        drop(dataset_guard);

        let mut dataset_guard = self.manifest_dataset.get_mut().await?;
        dataset_guard
            .update_metadata([(name, value)])
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to set property '{}' in __manifest metadata: {}",
                        name, e
                    ),
                })
            })?;
        Ok(())
    }

    /// Check if a property flag exists in the __manifest table's metadata key-value map.
    pub async fn has_property(&self, name: &str) -> Result<bool> {
        let dataset_guard = self.manifest_dataset.get().await?;
        Ok(dataset_guard.metadata().contains_key(name))
    }

    /// Parse metadata JSON into a `TableVersion`.
    ///
    /// Returns `None` if metadata is invalid or missing required fields.
    fn parse_table_version(version: i64, metadata_str: &str) -> Option<TableVersion> {
        let meta: serde_json::Value = match serde_json::from_str(metadata_str) {
            Ok(v) => v,
            Err(e) => {
                log::warn!(
                    "Skipping version {} due to invalid metadata JSON: {}",
                    version,
                    e
                );
                return None;
            }
        };
        let manifest_path = match meta.get("manifest_path").and_then(|v| v.as_str()) {
            Some(p) => p.to_string(),
            None => {
                log::warn!(
                    "Skipping version {} due to missing 'manifest_path' in metadata — \
                     this may indicate data corruption",
                    version
                );
                return None;
            }
        };
        let manifest_size = meta.get("manifest_size").and_then(|v| v.as_i64());
        let e_tag = meta
            .get("e_tag")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        Some(TableVersion {
            version,
            manifest_path,
            manifest_size,
            e_tag,
            timestamp_millis: None,
            metadata: None,
        })
    }

    /// List table versions from the __manifest table.
    ///
    /// Queries the manifest for all versions of the given table and returns
    /// them as a `ListTableVersionsResponse`.
    pub async fn list_table_versions(
        &self,
        table_id: &[String],
        descending: bool,
        limit: Option<i32>,
    ) -> Result<ListTableVersionsResponse> {
        let object_id = Self::str_object_id(table_id);
        let manifest_versions = self
            .query_table_versions(&object_id, descending, limit)
            .await?;

        let table_versions: Vec<TableVersion> = manifest_versions
            .into_iter()
            .filter_map(|(version, metadata_str)| Self::parse_table_version(version, &metadata_str))
            .collect();

        Ok(ListTableVersionsResponse {
            versions: table_versions,
            page_token: None,
        })
    }

    /// Describe a specific table version from the __manifest table.
    ///
    /// Queries the manifest for a specific version and returns it as a
    /// `DescribeTableVersionResponse`. Returns an error if the version is not found.
    pub async fn describe_table_version(
        &self,
        table_id: &[String],
        version: i64,
    ) -> Result<DescribeTableVersionResponse> {
        let object_id = Self::str_object_id(table_id);
        if let Some(metadata_str) = self.query_table_version(&object_id, version).await?
            && let Some(tv) = Self::parse_table_version(version, &metadata_str)
        {
            return Ok(DescribeTableVersionResponse {
                version: Box::new(tv),
            });
        }
        Err(NamespaceError::TableVersionNotFound {
            message: format!("version {} for table {:?}", version, table_id),
        }
        .into())
    }

    /// Register a table in the manifest without creating the physical table (internal helper for migration)
    pub async fn register_table(&self, name: &str, location: String) -> Result<()> {
        let object_id = Self::build_object_id(&[], name);
        if self.manifest_contains_object(&object_id).await? {
            return Err(NamespaceError::Internal {
                message: format!("Table '{}' already exists", name),
            }
            .into());
        }

        self.insert_into_manifest(object_id, ObjectType::Table, Some(location))
            .await
    }

    /// Validate that all levels of a namespace path exist
    async fn validate_namespace_levels_exist(&self, namespace_path: &[String]) -> Result<()> {
        for i in 1..=namespace_path.len() {
            let partial_path = &namespace_path[..i];
            let object_id = partial_path.join(DELIMITER);
            if !self.manifest_contains_object(&object_id).await? {
                return Err(NamespaceError::NamespaceNotFound {
                    message: format!("parent namespace '{}'", object_id),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Query the manifest for a namespace with the given object ID
    async fn query_manifest_for_namespace(&self, object_id: &str) -> Result<Option<NamespaceInfo>> {
        let escaped_id = object_id.replace('\'', "''");
        let filter = format!("object_id = '{}' AND object_type = 'namespace'", escaped_id);
        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner.project(&["object_id", "metadata"]).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to project: {:?}", e),
            })
        })?;
        let batches = Self::execute_scanner(scanner).await?;

        let mut found_result: Option<NamespaceInfo> = None;
        let mut total_rows = 0;

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            total_rows += batch.num_rows();
            if total_rows > 1 {
                return Err(NamespaceError::Internal {
                    message: format!(
                        "Expected exactly 1 namespace with id '{}', found {}",
                        object_id, total_rows
                    ),
                }
                .into());
            }

            let object_id_array = Self::get_string_column(&batch, "object_id")?;
            let metadata_values = Self::metadata_column_values(&batch, "metadata")?;

            let object_id_str = object_id_array.value(0);
            let metadata = if let Some(metadata_str) = &metadata_values[0] {
                match serde_json::from_str::<HashMap<String, String>>(metadata_str) {
                    Ok(map) => Some(map),
                    Err(e) => {
                        return Err(NamespaceError::Internal {
                            message: format!(
                                "Failed to deserialize metadata for namespace '{}': {}",
                                object_id, e
                            ),
                        }
                        .into());
                    }
                }
            } else {
                None
            };

            let (namespace, name) = Self::parse_object_id(object_id_str);
            found_result = Some(NamespaceInfo {
                namespace,
                name,
                metadata,
            });
        }

        Ok(found_result)
    }

    /// Create or load the manifest dataset, ensuring it has the latest schema setup.
    ///
    /// This function will:
    /// 1. Try to load an existing manifest table
    /// 2. If it exists, check and migrate the schema if needed (e.g., add primary key metadata)
    /// 3. If it doesn't exist, create a new manifest table with the current schema
    /// 4. Persist feature flags (e.g., table_version_storage_enabled) if requested
    async fn ensure_manifest_table_up_to_date(
        root: &str,
        manifest_table_name: &str,
        manifest_branch: Option<&str>,
        storage_options: &Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
        table_version_storage_enabled: bool,
    ) -> Result<DatasetConsistencyWrapper> {
        let manifest_path = format!("{}/{}", root, manifest_table_name);
        log::debug!("Attempting to load manifest from {}", manifest_path);
        let store_options = ObjectStoreParams {
            storage_options_accessor: storage_options.as_ref().map(|opts| {
                Arc::new(
                    lance_io::object_store::StorageOptionsAccessor::with_static_options(
                        opts.clone(),
                    ),
                )
            }),
            ..Default::default()
        };
        let read_params = ReadParams {
            session: session.clone(),
            store_options: Some(store_options.clone()),
            ..Default::default()
        };
        let dataset_result = DatasetBuilder::from_uri(&manifest_path)
            .with_read_params(read_params)
            .load()
            .await;
        if let Ok(mut dataset) = dataset_result {
            if let Some(branch) = manifest_branch {
                dataset = dataset.checkout_branch(branch).await.map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!(
                            "Failed to checkout manifest table '{}' branch '{}': {:?}",
                            manifest_table_name, branch, e
                        ),
                    })
                })?;
            }

            // Check if the object_id field has primary key metadata, migrate if not
            let needs_pk_migration = dataset
                .schema()
                .field("object_id")
                .map(|f| {
                    !f.metadata
                        .contains_key(LANCE_UNENFORCED_PRIMARY_KEY_POSITION)
                })
                .unwrap_or(false);

            if needs_pk_migration {
                log::info!(
                    "Migrating {} table to add primary key metadata on object_id",
                    manifest_table_name
                );
                dataset
                    .update_field_metadata()
                    .update("object_id", [(LANCE_UNENFORCED_PRIMARY_KEY_POSITION, "0")])
                    .map_err(|e| {
                        lance_core::Error::from(NamespaceError::Internal {
                            message: format!(
                                "Failed to find object_id field for migration: {:?}",
                                e
                            ),
                        })
                    })?
                    .await
                    .map_err(|e| {
                        lance_core::Error::from(NamespaceError::Internal {
                            message: format!("Failed to migrate primary key metadata: {:?}", e),
                        })
                    })?;
            }

            // Persist table_version_storage_enabled flag in the manifest table so that once
            // enabled, it becomes a permanent property of this namespace.
            if table_version_storage_enabled {
                let needs_flag = dataset
                    .metadata()
                    .get("table_version_storage_enabled")
                    .map(|v| v != "true")
                    .unwrap_or(true);

                if needs_flag
                    && let Err(e) = dataset
                        .update_metadata([("table_version_storage_enabled", "true")])
                        .await
                {
                    log::warn!(
                        "Failed to persist table_version_storage_enabled flag in {}: {:?}",
                        manifest_table_name,
                        e
                    );
                }
            }

            Ok(DatasetConsistencyWrapper::new(dataset))
        } else {
            log::info!("Creating new manifest table at {}", manifest_path);
            let schema = Self::manifest_schema();
            let empty_batch = RecordBatch::new_empty(schema.clone());
            let reader = RecordBatchIterator::new(vec![Ok(empty_batch)], schema.clone());

            let store_params = ObjectStoreParams {
                storage_options_accessor: storage_options.as_ref().map(|opts| {
                    Arc::new(
                        lance_io::object_store::StorageOptionsAccessor::with_static_options(
                            opts.clone(),
                        ),
                    )
                }),
                ..Default::default()
            };
            let write_params = WriteParams {
                session: session.clone(),
                store_params: Some(store_params),
                ..Default::default()
            };

            let dataset =
                Dataset::write(Box::new(reader), &manifest_path, Some(write_params)).await;

            // Handle race condition where another process created the manifest concurrently
            match dataset {
                Ok(dataset) => {
                    let dataset = if let Some(branch) = manifest_branch {
                        let mut dataset = dataset;
                        dataset
                            .create_branch(branch, (None, None), None)
                            .await
                            .map_err(|e| {
                                lance_core::Error::from(NamespaceError::Internal {
                                    message: format!(
                                        "Failed to initialize manifest table '{}' branch '{}': {:?}",
                                        manifest_table_name, branch, e
                                    ),
                                })
                            })?
                    } else {
                        dataset
                    };
                    log::info!(
                        "Successfully created manifest table at {}, version={}, uri={}",
                        manifest_path,
                        dataset.version().version,
                        dataset.uri()
                    );
                    Ok(DatasetConsistencyWrapper::new(dataset))
                }
                Err(ref e)
                    if matches!(
                        e,
                        LanceError::DatasetAlreadyExists { .. }
                            | LanceError::CommitConflict { .. }
                            | LanceError::IncompatibleTransaction { .. }
                            | LanceError::RetryableCommitConflict { .. }
                    ) =>
                {
                    // Another process created the manifest concurrently, try to load it
                    log::info!(
                        "Manifest table was created by another process, loading it: {}",
                        manifest_path
                    );
                    let recovery_store_options = ObjectStoreParams {
                        storage_options_accessor: storage_options.as_ref().map(|opts| {
                            Arc::new(
                                lance_io::object_store::StorageOptionsAccessor::with_static_options(
                                    opts.clone(),
                                ),
                            )
                        }),
                        ..Default::default()
                    };
                    let recovery_read_params = ReadParams {
                        session,
                        store_options: Some(recovery_store_options),
                        ..Default::default()
                    };
                    let dataset = DatasetBuilder::from_uri(&manifest_path)
                        .with_read_params(recovery_read_params)
                        .load()
                        .await
                        .map_err(|e| {
                            lance_core::Error::from(NamespaceError::Internal {
                                message: format!(
                                    "Failed to load manifest dataset after creation conflict: {}",
                                    e
                                ),
                            })
                        })?;
                    Ok(DatasetConsistencyWrapper::new(dataset))
                }
                Err(e) => Err(lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to create manifest dataset: {:?}", e),
                })),
            }
        }
    }

    /// Sorts names alphabetically and applies pagination using page_token (start_after) and limit.
    ///
    /// Returns the next page token (last item in this page) if more results exist beyond the limit,
    /// or `None` if this is the last page.
    fn apply_pagination(
        names: &mut Vec<String>,
        page_token: Option<String>,
        limit: Option<i32>,
    ) -> Option<String> {
        names.sort();

        if let Some(start_after) = page_token {
            if let Some(index) = names
                .iter()
                .position(|name| name.as_str() > start_after.as_str())
            {
                names.drain(0..index);
            } else {
                names.clear();
            }
        }

        if let Some(limit) = limit
            && limit >= 0
        {
            let limit = limit as usize;
            if names.len() > limit {
                let next_page_token = if limit > 0 {
                    Some(names[limit - 1].clone())
                } else {
                    None
                };
                names.truncate(limit);
                return next_page_token;
            }
        }

        None
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CatalogBranchMetadata {
    branch: String,
    shard_branch: Option<String>,
    generation: u64,
    parent_branch: Option<String>,
    parent_generation: Option<u64>,
    shard_count: usize,
}

/// Manifest-backed catalog split across multiple Lance manifest tables.
///
/// Each shard is itself a [`ManifestNamespace`]. Catalog branches are represented
/// by Lance branches on every shard table, with `__super_manifest` mapping the
/// logical catalog branch name to the shard-table branch to read.
pub struct ShardedManifestNamespace {
    root: String,
    catalog_branch: String,
    branch_metadata: CatalogBranchMetadata,
    storage_options: Option<HashMap<String, String>>,
    session: Option<Arc<Session>>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    super_manifest: Arc<ManifestNamespace>,
    shards: Vec<Arc<ManifestNamespace>>,
}

impl std::fmt::Debug for ShardedManifestNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedManifestNamespace")
            .field("root", &self.root)
            .field("catalog_branch", &self.catalog_branch)
            .field("shard_count", &self.shards.len())
            .field("branch_metadata", &self.branch_metadata)
            .finish()
    }
}

impl ShardedManifestNamespace {
    #[allow(clippy::too_many_arguments)]
    pub async fn from_directory(
        root: String,
        storage_options: Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        dir_listing_enabled: bool,
        inline_optimization_enabled: bool,
        commit_retries: Option<u32>,
        table_version_storage_enabled: bool,
        shard_count: usize,
        catalog_branch: String,
    ) -> Result<Self> {
        if shard_count == 0 {
            return Err(NamespaceError::InvalidInput {
                message: "manifest_shard_count must be greater than 0".to_string(),
            }
            .into());
        }

        let super_manifest = Arc::new(
            ManifestNamespace::from_directory_with_manifest_table(
                root.clone(),
                SUPER_MANIFEST_TABLE_NAME.to_string(),
                None,
                storage_options.clone(),
                session.clone(),
                object_store.clone(),
                base_path.clone(),
                dir_listing_enabled,
                inline_optimization_enabled,
                commit_retries,
                false,
            )
            .await?,
        );

        Self::ensure_base_shards(
            &root,
            &storage_options,
            session.clone(),
            object_store.clone(),
            base_path.clone(),
            dir_listing_enabled,
            inline_optimization_enabled,
            commit_retries,
            table_version_storage_enabled,
            shard_count,
        )
        .await?;

        if Self::read_branch_metadata(&super_manifest, "main")
            .await?
            .is_none()
        {
            let metadata = CatalogBranchMetadata {
                branch: "main".to_string(),
                shard_branch: None,
                generation: 0,
                parent_branch: None,
                parent_generation: None,
                shard_count,
            };
            Self::write_branch_metadata(&super_manifest, &metadata).await?;
        }

        let branch_metadata = Self::read_branch_metadata(&super_manifest, &catalog_branch)
            .await?
            .ok_or_else(|| {
                lance_core::Error::from(NamespaceError::NamespaceNotFound {
                    message: format!("catalog branch '{}'", catalog_branch),
                })
            })?;

        let shards = Self::open_shards(
            &root,
            &storage_options,
            session.clone(),
            object_store.clone(),
            base_path.clone(),
            dir_listing_enabled,
            inline_optimization_enabled,
            commit_retries,
            table_version_storage_enabled,
            branch_metadata.shard_count,
            branch_metadata.shard_branch.as_deref(),
        )
        .await?;

        Ok(Self {
            root,
            catalog_branch,
            branch_metadata,
            storage_options,
            session,
            object_store,
            base_path,
            super_manifest,
            shards,
        })
    }

    pub async fn create_catalog_branch(&self, branch: &str) -> Result<()> {
        if branch == "main" {
            return Err(NamespaceError::InvalidInput {
                message: "catalog branch name 'main' is reserved".to_string(),
            }
            .into());
        }
        lance::dataset::refs::check_valid_branch(branch)?;
        if Self::read_branch_metadata(&self.super_manifest, branch)
            .await?
            .is_some()
        {
            return Err(NamespaceError::NamespaceAlreadyExists {
                message: format!("catalog branch '{}'", branch),
            }
            .into());
        }
        let parent_metadata =
            Self::read_branch_metadata(&self.super_manifest, &self.catalog_branch)
                .await?
                .ok_or_else(|| {
                    lance_core::Error::from(NamespaceError::NamespaceNotFound {
                        message: format!("catalog branch '{}'", self.catalog_branch),
                    })
                })?;
        if parent_metadata.generation != self.branch_metadata.generation
            || parent_metadata.shard_branch != self.branch_metadata.shard_branch
        {
            return Err(NamespaceError::ConcurrentModification {
                message: format!(
                    "Catalog branch '{}' changed; reopen the namespace before branching",
                    self.catalog_branch
                ),
            }
            .into());
        }

        for shard in &self.shards {
            shard.create_manifest_branch(branch).await?;
        }

        let metadata = CatalogBranchMetadata {
            branch: branch.to_string(),
            shard_branch: Some(branch.to_string()),
            generation: 0,
            parent_branch: Some(self.catalog_branch.clone()),
            parent_generation: Some(parent_metadata.generation),
            shard_count: parent_metadata.shard_count,
        };
        Self::write_branch_metadata(&self.super_manifest, &metadata).await
    }

    pub async fn promote_catalog_branch(&self, branch: &str, target_branch: &str) -> Result<()> {
        let source = Self::read_branch_metadata(&self.super_manifest, branch)
            .await?
            .ok_or_else(|| {
                lance_core::Error::from(NamespaceError::NamespaceNotFound {
                    message: format!("catalog branch '{}'", branch),
                })
            })?;
        let target = Self::read_branch_metadata(&self.super_manifest, target_branch)
            .await?
            .ok_or_else(|| {
                lance_core::Error::from(NamespaceError::NamespaceNotFound {
                    message: format!("catalog branch '{}'", target_branch),
                })
            })?;

        if source.parent_branch.as_deref() != Some(target_branch)
            || source.parent_generation != Some(target.generation)
        {
            return Err(NamespaceError::ConcurrentModification {
                message: format!(
                    "Catalog branch '{}' is not a fast-forward update for '{}'",
                    branch, target_branch
                ),
            }
            .into());
        }

        let promoted = CatalogBranchMetadata {
            branch: target_branch.to_string(),
            shard_branch: source.shard_branch.clone(),
            generation: target.generation + 1,
            parent_branch: target.parent_branch.clone(),
            parent_generation: target.parent_generation,
            shard_count: source.shard_count,
        };
        Self::write_branch_metadata(&self.super_manifest, &promoted).await
    }

    #[allow(clippy::too_many_arguments)]
    async fn ensure_base_shards(
        root: &str,
        storage_options: &Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        dir_listing_enabled: bool,
        inline_optimization_enabled: bool,
        commit_retries: Option<u32>,
        table_version_storage_enabled: bool,
        shard_count: usize,
    ) -> Result<()> {
        for shard_index in 0..shard_count {
            ManifestNamespace::from_directory_with_manifest_table(
                root.to_string(),
                Self::shard_table_name(shard_index),
                None,
                storage_options.clone(),
                session.clone(),
                object_store.clone(),
                base_path.clone(),
                dir_listing_enabled,
                inline_optimization_enabled,
                commit_retries,
                table_version_storage_enabled,
            )
            .await?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn open_shards(
        root: &str,
        storage_options: &Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        dir_listing_enabled: bool,
        inline_optimization_enabled: bool,
        commit_retries: Option<u32>,
        table_version_storage_enabled: bool,
        shard_count: usize,
        shard_branch: Option<&str>,
    ) -> Result<Vec<Arc<ManifestNamespace>>> {
        let mut shards = Vec::with_capacity(shard_count);
        for shard_index in 0..shard_count {
            let shard = ManifestNamespace::from_directory_with_manifest_table(
                root.to_string(),
                Self::shard_table_name(shard_index),
                shard_branch.map(ToString::to_string),
                storage_options.clone(),
                session.clone(),
                object_store.clone(),
                base_path.clone(),
                dir_listing_enabled,
                inline_optimization_enabled,
                commit_retries,
                table_version_storage_enabled,
            )
            .await?;
            shards.push(Arc::new(shard));
        }
        Ok(shards)
    }

    fn shard_table_name(shard_index: usize) -> String {
        format!("{}_{:06}", SHARDED_MANIFEST_TABLE_PREFIX, shard_index)
    }

    async fn read_branch_metadata(
        super_manifest: &ManifestNamespace,
        branch: &str,
    ) -> Result<Option<CatalogBranchMetadata>> {
        let Some(entry) = super_manifest
            .query_manifest_entry(branch, ObjectType::CatalogBranch)
            .await?
        else {
            return Ok(None);
        };
        let Some(metadata) = entry.metadata else {
            return Err(NamespaceError::Internal {
                message: format!("Catalog branch '{}' is missing metadata", branch),
            }
            .into());
        };
        serde_json::from_str(&metadata).map(Some).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to parse catalog branch '{}': {}", branch, e),
            })
        })
    }

    async fn write_branch_metadata(
        super_manifest: &ManifestNamespace,
        metadata: &CatalogBranchMetadata,
    ) -> Result<()> {
        let metadata_json = serde_json::to_string(metadata).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!(
                    "Failed to serialize catalog branch '{}': {}",
                    metadata.branch, e
                ),
            })
        })?;
        super_manifest
            .upsert_into_manifest_with_metadata(vec![ManifestEntry {
                object_id: metadata.branch.clone(),
                object_type: ObjectType::CatalogBranch,
                location: None,
                metadata: Some(metadata_json),
            }])
            .await
    }

    fn stable_hash(value: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in value.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    fn table_version_table_id(version_object_id: &str) -> &str {
        version_object_id
            .rsplit_once(DELIMITER)
            .map(|(table_id, _)| table_id)
            .unwrap_or(version_object_id)
    }

    fn shard_key_for_entry(entry: &ManifestEntry) -> &str {
        match entry.object_type {
            ObjectType::TableVersion => Self::table_version_table_id(&entry.object_id),
            _ => &entry.object_id,
        }
    }

    fn shard_index_for_key(&self, key: &str) -> usize {
        Self::stable_hash(key) as usize % self.shards.len()
    }

    fn shard_for_object_id(&self, object_id: &str) -> &ManifestNamespace {
        self.shards[self.shard_index_for_key(object_id)].as_ref()
    }

    fn shard_for_table_id(&self, table_id: &[String]) -> &ManifestNamespace {
        let object_id = ManifestNamespace::str_object_id(table_id);
        self.shard_for_object_id(&object_id)
    }

    fn relative_location_from_path(&self, table_path: &Path) -> Result<String> {
        let Some(parts) = table_path.prefix_match(&self.base_path) else {
            return Err(NamespaceError::Internal {
                message: format!(
                    "Materialized table path '{}' is not under catalog base path '{}'",
                    table_path.as_ref(),
                    self.base_path.as_ref()
                ),
            }
            .into());
        };
        Ok(parts
            .map(|part| part.as_ref().to_string())
            .collect::<Vec<_>>()
            .join("/"))
    }

    fn branch_relative_location(location: &str, branch: &str) -> String {
        let table_root = location
            .trim_end_matches('/')
            .rfind("/tree/")
            .map(|index| &location[..index])
            .unwrap_or_else(|| location.trim_end_matches('/'));
        let branch_path = branch
            .split('/')
            .fold(String::from("tree"), |mut path, segment| {
                path.push('/');
                path.push_str(segment);
                path
            });
        format!("{}/{}", table_root, branch_path)
    }

    async fn load_table_dataset(&self, table_uri: &str) -> Result<Dataset> {
        let mut builder = DatasetBuilder::from_uri(table_uri);
        if let Some(opts) = &self.storage_options {
            builder = builder.with_storage_options(opts.clone());
        }
        if let Some(session) = &self.session {
            builder = builder.with_session(session.clone());
        }
        builder.load().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!(
                    "Failed to load table '{}' for catalog branch: {}",
                    table_uri, e
                ),
            })
        })
    }

    async fn materialize_table_branch(&self, table_id: &[String]) -> Result<()> {
        let Some(branch) = self.branch_metadata.shard_branch.as_deref() else {
            return Ok(());
        };
        let object_id = ManifestNamespace::str_object_id(table_id);
        let shard = self.shard_for_object_id(&object_id);
        let Some(entry) = shard
            .query_manifest_entry(&object_id, ObjectType::Table)
            .await?
        else {
            return Ok(());
        };
        let Some(location) = entry.location.as_deref() else {
            return Ok(());
        };
        if location == Self::branch_relative_location(location, branch) {
            return Ok(());
        }

        let table_path = ManifestNamespace::relative_location_path(&self.base_path, location);
        let materialized_location =
            if ManifestNamespace::path_has_actual_manifests(&self.object_store, &table_path).await?
            {
                let table_uri = ManifestNamespace::construct_full_uri(&self.root, location)?;
                let mut dataset = self.load_table_dataset(&table_uri).await?;
                let source_branch = dataset.branch_location().branch;
                let store_params = ObjectStoreParams {
                    storage_options_accessor: self.storage_options.as_ref().map(|opts| {
                        Arc::new(
                            lance_io::object_store::StorageOptionsAccessor::with_static_options(
                                opts.clone(),
                            ),
                        )
                    }),
                    ..Default::default()
                };
                let branch_dataset = match dataset
                    .create_branch(branch, (source_branch.as_deref(), None), Some(store_params))
                    .await
                {
                    Ok(dataset) => dataset,
                    Err(LanceError::RefConflict { .. }) => {
                        dataset.checkout_branch(branch).await.map_err(|e| {
                            lance_core::Error::from(NamespaceError::Internal {
                                message: format!(
                                    "Failed to checkout existing table branch '{}' for '{}': {}",
                                    branch, object_id, e
                                ),
                            })
                        })?
                    }
                    Err(e) => {
                        return Err(NamespaceError::Internal {
                            message: format!(
                                "Failed to create table branch '{}' for '{}': {}",
                                branch, object_id, e
                            ),
                        }
                        .into());
                    }
                };
                self.relative_location_from_path(&branch_dataset.branch_location().path)?
            } else {
                Self::branch_relative_location(location, branch)
            };

        if location == materialized_location {
            return Ok(());
        }

        shard
            .upsert_into_manifest_with_metadata(vec![ManifestEntry {
                object_id,
                object_type: ObjectType::Table,
                location: Some(materialized_location),
                metadata: entry.metadata,
            }])
            .await
    }

    fn apply_pagination(
        names: &mut Vec<String>,
        page_token: Option<String>,
        limit: Option<i32>,
    ) -> Option<String> {
        names.sort();
        names.dedup();

        if let Some(start_after) = page_token {
            if let Some(index) = names
                .iter()
                .position(|name| name.as_str() > start_after.as_str())
            {
                names.drain(0..index);
            } else {
                names.clear();
            }
        }

        if let Some(limit) = limit
            && limit >= 0
        {
            let limit = limit as usize;
            if names.len() > limit {
                let next_page_token = if limit > 0 {
                    Some(names[limit - 1].clone())
                } else {
                    None
                };
                names.truncate(limit);
                return next_page_token;
            }
        }

        None
    }

    async fn validate_namespace_levels_exist(&self, namespace_path: &[String]) -> Result<()> {
        for i in 1..=namespace_path.len() {
            let partial_path = &namespace_path[..i];
            let object_id = partial_path.join(DELIMITER);
            if !self
                .shard_for_object_id(&object_id)
                .manifest_contains_object(&object_id)
                .await?
            {
                return Err(NamespaceError::NamespaceNotFound {
                    message: format!("parent namespace '{}'", object_id),
                }
                .into());
            }
        }
        Ok(())
    }

    pub async fn list_manifest_table_locations(&self) -> Result<HashSet<String>> {
        let mut locations = HashSet::new();
        for shard in &self.shards {
            locations.extend(shard.list_manifest_table_locations().await?);
        }
        Ok(locations)
    }

    pub async fn register_table(&self, name: &str, location: String) -> Result<()> {
        let object_id = ManifestNamespace::build_object_id(&[], name);
        self.shard_for_object_id(&object_id)
            .insert_into_manifest(object_id, ObjectType::Table, Some(location))
            .await
    }

    pub async fn insert_into_manifest_with_metadata(
        &self,
        entries: Vec<ManifestEntry>,
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let mut grouped: HashMap<usize, Vec<ManifestEntry>> = HashMap::new();
        for entry in entries {
            let shard_index = self.shard_index_for_key(Self::shard_key_for_entry(&entry));
            grouped.entry(shard_index).or_default().push(entry);
        }

        for (shard_index, entries) in grouped {
            self.shards[shard_index]
                .insert_into_manifest_with_metadata(entries)
                .await?;
        }
        Ok(())
    }

    pub async fn query_table_versions(
        &self,
        object_id: &str,
        descending: bool,
        limit: Option<i32>,
    ) -> Result<Vec<(i64, String)>> {
        self.shard_for_object_id(object_id)
            .query_table_versions(object_id, descending, limit)
            .await
    }

    pub async fn list_table_versions(
        &self,
        table_id: &[String],
        descending: bool,
        limit: Option<i32>,
    ) -> Result<ListTableVersionsResponse> {
        self.shard_for_table_id(table_id)
            .list_table_versions(table_id, descending, limit)
            .await
    }

    pub async fn describe_table_version(
        &self,
        table_id: &[String],
        version: i64,
    ) -> Result<DescribeTableVersionResponse> {
        self.shard_for_table_id(table_id)
            .describe_table_version(table_id, version)
            .await
    }

    pub async fn batch_delete_table_versions_by_ranges(
        &self,
        table_ranges: &[(String, Vec<(i64, i64)>)],
    ) -> Result<i64> {
        let mut grouped: HashMap<usize, Vec<TableVersionRangeRequest>> = HashMap::new();
        for (object_id, ranges) in table_ranges {
            let shard_index = self.shard_index_for_key(object_id);
            grouped
                .entry(shard_index)
                .or_default()
                .push((object_id.clone(), ranges.clone()));
        }

        let mut deleted = 0;
        for (shard_index, ranges) in grouped {
            deleted += self.shards[shard_index]
                .batch_delete_table_versions_by_ranges(&ranges)
                .await?;
        }
        Ok(deleted)
    }
}

#[async_trait]
impl LanceNamespace for ShardedManifestNamespace {
    fn namespace_id(&self) -> String {
        format!("{}#{}", self.root, self.catalog_branch)
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        let mut all_tables = Vec::new();
        for shard in &self.shards {
            let mut shard_request = request.clone();
            shard_request.limit = None;
            shard_request.page_token = None;
            all_tables.extend(shard.list_tables(shard_request).await?.tables);
        }

        let next_page_token =
            Self::apply_pagination(&mut all_tables, request.page_token, request.limit);
        let mut response = ListTablesResponse::new(all_tables);
        response.page_token = next_page_token;
        Ok(response)
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        let table_id = request.id.clone().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;
        self.materialize_table_branch(&table_id).await?;
        self.shard_for_table_id(&table_id)
            .describe_table(request)
            .await
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;
        self.shard_for_table_id(table_id)
            .table_exists(request)
            .await
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        data: Bytes,
    ) -> Result<CreateTableResponse> {
        let table_id = request.id.clone().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;
        self.materialize_table_branch(&table_id).await?;
        self.shard_for_table_id(&table_id)
            .create_table(request, data)
            .await
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        let table_id = request.id.clone().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;
        self.materialize_table_branch(&table_id).await?;
        self.shard_for_table_id(&table_id).drop_table(request).await
    }

    async fn declare_table(&self, request: DeclareTableRequest) -> Result<DeclareTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;
        self.shard_for_table_id(table_id)
            .declare_table(request)
            .await
    }

    async fn register_table(&self, request: RegisterTableRequest) -> Result<RegisterTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;
        LanceNamespace::register_table(self.shard_for_table_id(table_id), request).await
    }

    async fn deregister_table(
        &self,
        request: DeregisterTableRequest,
    ) -> Result<DeregisterTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;
        self.shard_for_table_id(table_id)
            .deregister_table(request)
            .await
    }

    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        let mut all_namespaces = Vec::new();
        for shard in &self.shards {
            let mut shard_request = request.clone();
            shard_request.limit = None;
            shard_request.page_token = None;
            all_namespaces.extend(shard.list_namespaces(shard_request).await?.namespaces);
        }

        let next_page_token =
            Self::apply_pagination(&mut all_namespaces, request.page_token, request.limit);
        let mut response = ListNamespacesResponse::new(all_namespaces);
        response.page_token = next_page_token;
        Ok(response)
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;
        if namespace_id.is_empty() {
            return Ok(DescribeNamespaceResponse {
                properties: Some(HashMap::new()),
            });
        }
        let object_id = namespace_id.join(DELIMITER);
        self.shard_for_object_id(&object_id)
            .describe_namespace(request)
            .await
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;
        if namespace_id.is_empty() {
            return Err(NamespaceError::NamespaceAlreadyExists {
                message: "root namespace".to_string(),
            }
            .into());
        }
        if namespace_id.len() > 1 {
            self.validate_namespace_levels_exist(&namespace_id[..namespace_id.len() - 1])
                .await?;
        }
        let object_id = namespace_id.join(DELIMITER);
        let shard = self.shard_for_object_id(&object_id);
        if shard.manifest_contains_object(&object_id).await? {
            return Err(NamespaceError::NamespaceAlreadyExists { message: object_id }.into());
        }
        let metadata = ManifestNamespace::serialize_metadata(
            request.properties.as_ref(),
            "namespace",
            &object_id,
        )?;
        shard
            .insert_into_manifest_with_metadata(vec![ManifestEntry {
                object_id,
                object_type: ObjectType::Namespace,
                location: None,
                metadata,
            }])
            .await?;
        Ok(CreateNamespaceResponse {
            properties: request.properties,
            ..Default::default()
        })
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;
        if namespace_id.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Root namespace cannot be dropped".to_string(),
            }
            .into());
        }
        let object_id = namespace_id.join(DELIMITER);
        let shard = self.shard_for_object_id(&object_id);
        if !shard.manifest_contains_object(&object_id).await? {
            return Err(NamespaceError::NamespaceNotFound { message: object_id }.into());
        }

        let prefix = format!("{}{}", object_id, DELIMITER);
        let mut child_count = 0;
        for shard in &self.shards {
            child_count += shard.count_objects_with_prefix(&prefix).await?;
        }
        if child_count > 0 {
            return Err(NamespaceError::NamespaceNotEmpty {
                message: format!("'{}' (contains {} child objects)", object_id, child_count),
            }
            .into());
        }

        shard.delete_from_manifest(&object_id).await?;
        Ok(DropNamespaceResponse::default())
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;
        if namespace_id.is_empty() {
            return Ok(());
        }
        let object_id = namespace_id.join(DELIMITER);
        if self
            .shard_for_object_id(&object_id)
            .manifest_contains_object(&object_id)
            .await?
        {
            Ok(())
        } else {
            Err(NamespaceError::NamespaceNotFound { message: object_id }.into())
        }
    }
}

#[async_trait]
impl LanceNamespace for ManifestNamespace {
    fn namespace_id(&self) -> String {
        self.root.clone()
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;

        // Build filter to find tables in this namespace
        let filter = if namespace_id.is_empty() {
            // Root namespace: find tables without a namespace prefix
            "object_type = 'table' AND NOT contains(object_id, '$')".to_string()
        } else {
            // Namespaced: find tables that start with namespace$ but have no additional $
            let prefix = namespace_id.join(DELIMITER);
            format!(
                "object_type = 'table' AND starts_with(object_id, '{}{}') AND NOT contains(substring(object_id, {}), '$')",
                prefix,
                DELIMITER,
                prefix.len() + 2
            )
        };

        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner.project(&["object_id", "location"]).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to project: {:?}", e),
            })
        })?;

        let batches = Self::execute_scanner(scanner).await?;

        let mut table_entries = Vec::new();
        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            let object_id_array = Self::get_string_column(&batch, "object_id")?;
            let location_array = Self::get_string_column(&batch, "location")?;
            for i in 0..batch.num_rows() {
                let object_id = object_id_array.value(i);
                let location = location_array.value(i);
                let (_namespace, name) = Self::parse_object_id(object_id);
                table_entries.push((name, location.to_string()));
            }
        }

        let mut tables: Vec<String> = if request.include_declared.unwrap_or(true) {
            table_entries.into_iter().map(|(name, _)| name).collect()
        } else {
            let mut stream = futures::stream::iter(table_entries.into_iter().map(
                |(name, location)| async move {
                    // `include_declared=false` is an explicit opt-in. We still pay one
                    // `_versions/` probe per table so declared-state is derived from actual
                    // manifests. This is linear in the total number of listed tables, and we do
                    // the probes with bounded concurrency before pagination.
                    if self.location_has_actual_manifests(&location).await? {
                        Ok::<Option<String>, Error>(Some(name))
                    } else {
                        Ok::<Option<String>, Error>(None)
                    }
                },
            ))
            .buffered(DECLARED_FILTER_CONCURRENCY);

            let mut filtered = Vec::new();
            while let Some(result) = stream.next().await {
                if let Some(name) = result? {
                    filtered.push(name);
                }
            }
            filtered
        };

        let next_page_token =
            Self::apply_pagination(&mut tables, request.page_token, request.limit);
        let mut response = ListTablesResponse::new(tables);
        response.page_token = next_page_token;
        Ok(response)
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;

        if table_id.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Table ID cannot be empty".to_string(),
            }
            .into());
        }

        let object_id = Self::str_object_id(table_id);
        let table_info = self.query_manifest_for_table(&object_id).boxed().await?;

        // Extract table name and namespace from table_id
        let table_name = table_id.last().cloned().unwrap_or_default();
        let namespace_id: Vec<String> = if table_id.len() > 1 {
            table_id[..table_id.len() - 1].to_vec()
        } else {
            vec![]
        };

        let load_detailed_metadata = request.load_detailed_metadata.unwrap_or(false);
        let should_check_declared =
            load_detailed_metadata || request.check_declared.unwrap_or(false);
        // For backwards compatibility, only skip vending credentials when explicitly set to false
        let vend_credentials = request.vend_credentials.unwrap_or(true);

        match table_info {
            Some(info) => {
                // Construct full URI from relative location
                let table_uri = Self::construct_full_uri(&self.root, &info.location)?;

                let storage_options = if vend_credentials {
                    self.storage_options.clone()
                } else {
                    None
                };
                let is_only_declared = if should_check_declared {
                    Some(!self.location_has_actual_manifests(&info.location).await?)
                } else {
                    None
                };

                if !load_detailed_metadata {
                    return Ok(DescribeTableResponse {
                        table: Some(table_name),
                        namespace: Some(namespace_id),
                        location: Some(table_uri.clone()),
                        table_uri: Some(table_uri),
                        storage_options,
                        properties: info.metadata,
                        is_only_declared,
                        ..Default::default()
                    });
                }

                if is_only_declared == Some(true) {
                    return Ok(DescribeTableResponse {
                        table: Some(table_name),
                        namespace: Some(namespace_id),
                        location: Some(table_uri.clone()),
                        table_uri: Some(table_uri),
                        storage_options,
                        properties: info.metadata,
                        is_only_declared,
                        ..Default::default()
                    });
                }

                let mut builder = DatasetBuilder::from_uri(&table_uri);
                if let Some(opts) = &self.storage_options {
                    builder = builder.with_storage_options(opts.clone());
                }
                if let Some(session) = &self.session {
                    builder = builder.with_session(session.clone());
                }

                match builder.load().await {
                    Ok(mut dataset) => {
                        // If a specific version is requested, checkout that version
                        if let Some(requested_version) = request.version {
                            dataset = dataset.checkout_version(requested_version as u64).await?;
                        }

                        let version = dataset.version().version;
                        let lance_schema = dataset.schema();
                        let arrow_schema: arrow_schema::Schema = lance_schema.into();
                        let json_schema = arrow_schema_to_json(&arrow_schema)?;

                        Ok(DescribeTableResponse {
                            table: Some(table_name.clone()),
                            namespace: Some(namespace_id.clone()),
                            version: Some(version as i64),
                            location: Some(table_uri.clone()),
                            table_uri: Some(table_uri),
                            schema: Some(Box::new(json_schema)),
                            storage_options,
                            properties: info.metadata.clone(),
                            is_only_declared,
                            ..Default::default()
                        })
                    }
                    Err(err) => Err(NamespaceError::Internal {
                        message: format!(
                            "Table exists in manifest but failed to load dataset '{}': {}",
                            object_id, err
                        ),
                    }
                    .into()),
                }
            }
            None => Err(NamespaceError::TableNotFound {
                message: Self::format_table_id(table_id),
            }
            .into()),
        }
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;

        if table_id.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Table ID cannot be empty".to_string(),
            }
            .into());
        }

        let object_id = Self::str_object_id(table_id);
        let exists = self.manifest_contains_object(&object_id).await?;
        if exists {
            Ok(())
        } else {
            Err(NamespaceError::TableNotFound {
                message: Self::format_table_id(table_id),
            }
            .into())
        }
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        data: Bytes,
    ) -> Result<CreateTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;

        if table_id.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Table ID cannot be empty".to_string(),
            }
            .into());
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        let existing_table = self.query_manifest_for_table(&object_id).await?;
        let existing_has_manifests = if let Some(existing_table) = &existing_table {
            Some(
                self.location_has_actual_manifests(&existing_table.location)
                    .await?,
            )
        } else {
            None
        };

        if existing_has_manifests == Some(false)
            && request
                .properties
                .as_ref()
                .is_some_and(|properties| !properties.is_empty())
        {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "create_table cannot set properties for already declared table '{}'",
                    object_id
                ),
            }
            .into());
        }

        let create_mode = if existing_has_manifests == Some(false) {
            CreateTableMode::Create
        } else {
            CreateTableMode::parse(request.mode.as_deref())?
        };
        let dir_name = if let Some(existing_table) = &existing_table {
            existing_table.location.clone()
        } else if namespace.is_empty() && self.dir_listing_enabled {
            format!("{}.lance", table_name)
        } else {
            Self::generate_dir_name(&object_id)
        };
        let table_uri = Self::construct_full_uri(&self.root, &dir_name)?;
        let overwriting_existing_table =
            existing_has_manifests == Some(true) && create_mode == CreateTableMode::Overwrite;

        if existing_has_manifests == Some(true) {
            match create_mode {
                CreateTableMode::Create => {
                    return Err(NamespaceError::TableAlreadyExists {
                        message: table_name.clone(),
                    }
                    .into());
                }
                CreateTableMode::ExistOk => {
                    let properties = existing_table
                        .as_ref()
                        .and_then(|table| table.metadata.clone());
                    return Ok(CreateTableResponse {
                        location: Some(table_uri),
                        storage_options: self.storage_options.clone(),
                        properties,
                        ..Default::default()
                    });
                }
                CreateTableMode::Overwrite => {}
            }
        }

        // Validate that request_data is provided
        if data.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Request data (Arrow IPC stream) is required for create_table".to_string(),
            }
            .into());
        }

        // Write the data using Lance Dataset
        let cursor = Cursor::new(data.to_vec());
        let stream_reader = StreamReader::try_new(cursor, None).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to read IPC stream: {:?}", e),
            })
        })?;

        let batches: Vec<RecordBatch> = stream_reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to collect batches: {:?}", e),
            })
        })?;

        if batches.is_empty() {
            return Err(NamespaceError::Internal {
                message: "No data provided for table creation".to_string(),
            }
            .into());
        }

        let schema = batches[0].schema();
        let batch_results: Vec<std::result::Result<RecordBatch, arrow_schema::ArrowError>> =
            batches.into_iter().map(Ok).collect();
        let reader = RecordBatchIterator::new(batch_results, schema);

        let mut write_storage_options = self.storage_options.clone().unwrap_or_default();
        if let Some(request_storage_options) = request.storage_options.as_ref() {
            write_storage_options.extend(request_storage_options.clone());
        }

        let store_params = ObjectStoreParams {
            storage_options_accessor: (!write_storage_options.is_empty()).then(|| {
                Arc::new(
                    lance_io::object_store::StorageOptionsAccessor::with_static_options(
                        write_storage_options,
                    ),
                )
            }),
            ..Default::default()
        };
        let write_params = WriteParams {
            mode: create_mode.write_mode(),
            session: self.session.clone(),
            store_params: Some(store_params),
            ..Default::default()
        };
        let dataset = Dataset::write(Box::new(reader), &table_uri, Some(write_params))
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to write dataset: {:?}", e),
                })
            })?;
        let version = dataset.version().version as i64;

        if overwriting_existing_table {
            let metadata =
                Self::serialize_metadata(request.properties.as_ref(), "table", &object_id)?;
            self.upsert_into_manifest_with_metadata(vec![ManifestEntry {
                object_id,
                object_type: ObjectType::Table,
                location: Some(dir_name),
                metadata,
            }])
            .await?;

            Ok(CreateTableResponse {
                version: Some(version),
                location: Some(table_uri),
                storage_options: self.storage_options.clone(),
                properties: request.properties,
                ..Default::default()
            })
        } else {
            match existing_table {
                Some(existing_table) => Ok(CreateTableResponse {
                    version: Some(version),
                    location: Some(table_uri),
                    storage_options: self.storage_options.clone(),
                    properties: existing_table.metadata,
                    ..Default::default()
                }),
                None => {
                    let metadata =
                        Self::serialize_metadata(request.properties.as_ref(), "table", &object_id)?;
                    // Register in manifest (store dir_name, not full URI)
                    self.insert_into_manifest_with_metadata(vec![ManifestEntry {
                        object_id,
                        object_type: ObjectType::Table,
                        location: Some(dir_name.clone()),
                        metadata,
                    }])
                    .await?;

                    Ok(CreateTableResponse {
                        version: Some(version),
                        location: Some(table_uri),
                        storage_options: self.storage_options.clone(),
                        properties: request.properties,
                        ..Default::default()
                    })
                }
            }
        }
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;

        if table_id.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Table ID cannot be empty".to_string(),
            }
            .into());
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Query manifest for table location
        let table_info = self.query_manifest_for_table(&object_id).boxed().await?;

        match table_info {
            Some(info) => {
                // Delete from manifest first
                self.delete_from_manifest(&object_id).boxed().await?;

                // Delete physical data directory using the dir_name from manifest
                let table_path = Self::relative_location_path(&self.base_path, &info.location);
                let table_uri = Self::construct_full_uri(&self.root, &info.location)?;

                // Remove the table directory
                self.object_store
                    .remove_dir_all(table_path)
                    .boxed()
                    .await
                    .map_err(|e| {
                        lance_core::Error::from(NamespaceError::Internal {
                            message: format!("Failed to delete table directory: {:?}", e),
                        })
                    })?;

                Ok(DropTableResponse {
                    id: request.id.clone(),
                    location: Some(table_uri),
                    ..Default::default()
                })
            }
            None => Err(NamespaceError::TableNotFound {
                message: table_name.to_string(),
            }
            .into()),
        }
    }

    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        let parent_namespace = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;

        // Build filter to find direct child namespaces
        let filter = if parent_namespace.is_empty() {
            // Root namespace: find all namespaces without a parent
            "object_type = 'namespace' AND NOT contains(object_id, '$')".to_string()
        } else {
            // Non-root: find namespaces that start with parent$ but have no additional $
            let prefix = parent_namespace.join(DELIMITER);
            format!(
                "object_type = 'namespace' AND starts_with(object_id, '{}{}') AND NOT contains(substring(object_id, {}), '$')",
                prefix,
                DELIMITER,
                prefix.len() + 2
            )
        };

        let mut scanner = self.manifest_scanner().await?;
        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner.project(&["object_id"]).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to project: {:?}", e),
            })
        })?;

        let batches = Self::execute_scanner(scanner).await?;
        let mut namespaces = Vec::new();

        for batch in batches {
            if batch.num_rows() == 0 {
                continue;
            }

            let object_id_array = Self::get_string_column(&batch, "object_id")?;
            for i in 0..batch.num_rows() {
                let object_id = object_id_array.value(i);
                let (_namespace, name) = Self::parse_object_id(object_id);
                namespaces.push(name);
            }
        }

        let next_page_token =
            Self::apply_pagination(&mut namespaces, request.page_token, request.limit);
        let mut response = ListNamespacesResponse::new(namespaces);
        response.page_token = next_page_token;
        Ok(response)
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;

        // Root namespace always exists
        if namespace_id.is_empty() {
            #[allow(clippy::needless_update)]
            return Ok(DescribeNamespaceResponse {
                properties: Some(HashMap::new()),
                ..Default::default()
            });
        }

        // Check if namespace exists in manifest
        let object_id = namespace_id.join(DELIMITER);
        let namespace_info = self.query_manifest_for_namespace(&object_id).await?;

        match namespace_info {
            #[allow(clippy::needless_update)]
            Some(info) => Ok(DescribeNamespaceResponse {
                properties: info.metadata,
                ..Default::default()
            }),
            None => Err(NamespaceError::NamespaceNotFound {
                message: object_id.to_string(),
            }
            .into()),
        }
    }

    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;

        // Root namespace always exists and cannot be created
        if namespace_id.is_empty() {
            return Err(NamespaceError::NamespaceAlreadyExists {
                message: "root namespace".to_string(),
            }
            .into());
        }

        // Validate parent namespaces exist (but not the namespace being created)
        if namespace_id.len() > 1 {
            self.validate_namespace_levels_exist(&namespace_id[..namespace_id.len() - 1])
                .await?;
        }

        let object_id = namespace_id.join(DELIMITER);
        if self.manifest_contains_object(&object_id).await? {
            return Err(NamespaceError::NamespaceAlreadyExists {
                message: object_id.to_string(),
            }
            .into());
        }

        let metadata =
            Self::serialize_metadata(request.properties.as_ref(), "namespace", &object_id)?;

        self.insert_into_manifest_with_metadata(vec![ManifestEntry {
            object_id,
            object_type: ObjectType::Namespace,
            location: None,
            metadata,
        }])
        .await?;

        Ok(CreateNamespaceResponse {
            properties: request.properties,
            ..Default::default()
        })
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;

        // Root namespace always exists and cannot be dropped
        if namespace_id.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Root namespace cannot be dropped".to_string(),
            }
            .into());
        }

        let object_id = namespace_id.join(DELIMITER);

        // Check if namespace exists
        if !self.manifest_contains_object(&object_id).boxed().await? {
            return Err(NamespaceError::NamespaceNotFound {
                message: object_id.to_string(),
            }
            .into());
        }

        // Check for child namespaces
        let escaped_id = object_id.replace('\'', "''");
        let prefix = format!("{}{}", escaped_id, DELIMITER);
        let filter = format!("starts_with(object_id, '{}')", prefix);
        let mut scanner = self.manifest_scanner().boxed().await?;
        scanner.filter(&filter).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to filter: {:?}", e),
            })
        })?;
        scanner.project::<&str>(&[]).map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to project: {:?}", e),
            })
        })?;
        scanner.with_row_id();
        let count = scanner.count_rows().boxed().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to count rows: {:?}", e),
            })
        })?;

        if count > 0 {
            return Err(NamespaceError::NamespaceNotEmpty {
                message: format!("'{}' (contains {} child objects)", object_id, count),
            }
            .into());
        }

        self.delete_from_manifest(&object_id).boxed().await?;

        Ok(DropNamespaceResponse::default())
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        let namespace_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Namespace ID is required".to_string(),
            })
        })?;

        // Root namespace always exists
        if namespace_id.is_empty() {
            return Ok(());
        }

        let object_id = namespace_id.join(DELIMITER);
        if self.manifest_contains_object(&object_id).await? {
            Ok(())
        } else {
            Err(NamespaceError::NamespaceNotFound {
                message: object_id.to_string(),
            }
            .into())
        }
    }

    async fn declare_table(&self, request: DeclareTableRequest) -> Result<DeclareTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;

        if table_id.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Table ID cannot be empty".to_string(),
            }
            .into());
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Check if table already exists in manifest
        let existing = self.query_manifest_for_table(&object_id).await?;
        if existing.is_some() {
            return Err(NamespaceError::TableAlreadyExists {
                message: table_name.to_string(),
            }
            .into());
        }

        // Create table location path with hash-based naming
        // When dir_listing_enabled is true and it's a root table, use directory-style naming: {table_name}.lance
        // Otherwise, use hash-based naming: {hash}_{object_id}
        let dir_name = if namespace.is_empty() && self.dir_listing_enabled {
            // Root table with directory listing enabled: use {table_name}.lance
            format!("{}.lance", table_name)
        } else {
            // Child namespace table or dir listing disabled: use hash-based naming
            Self::generate_dir_name(&object_id)
        };
        let table_path = self.base_path.clone().join(dir_name.as_str());
        let table_uri = Self::construct_full_uri(&self.root, &dir_name)?;

        // Validate location if provided
        if let Some(req_location) = &request.location {
            let req_location = req_location.trim_end_matches('/');
            if req_location != table_uri {
                return Err(NamespaceError::InvalidInput {
                    message: format!(
                        "Cannot declare table {} at location {}, must be at location {}",
                        table_name, req_location, table_uri
                    ),
                }
                .into());
            }
        }

        // Create the .lance-reserved file to mark the table as existing
        let reserved_file_path = table_path.clone().join(".lance-reserved");

        self.object_store
            .create(&reserved_file_path)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to create .lance-reserved file for table {}: {}",
                        table_name, e
                    ),
                })
            })?
            .shutdown()
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!(
                        "Failed to finalize .lance-reserved file for table {}: {}",
                        table_name, e
                    ),
                })
            })?;

        let metadata = Self::serialize_metadata(request.properties.as_ref(), "table", &object_id)?;

        // Add entry to manifest marking this as a declared table (store dir_name, not full path)
        self.insert_into_manifest_with_metadata(vec![ManifestEntry {
            object_id,
            object_type: ObjectType::Table,
            location: Some(dir_name),
            metadata,
        }])
        .await?;

        log::info!(
            "Declared table '{}' in manifest at {}",
            table_name,
            table_uri
        );

        // For backwards compatibility, only skip vending credentials when explicitly set to false
        let vend_credentials = request.vend_credentials.unwrap_or(true);
        let storage_options = if vend_credentials {
            self.storage_options.clone()
        } else {
            None
        };

        Ok(DeclareTableResponse {
            location: Some(table_uri),
            storage_options,
            properties: request.properties,
            ..Default::default()
        })
    }

    async fn register_table(&self, request: RegisterTableRequest) -> Result<RegisterTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;

        if table_id.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Table ID cannot be empty".to_string(),
            }
            .into());
        }

        let location = request.location.clone();

        // Validate that location is a relative path within the root directory
        // We don't allow absolute URIs or paths that escape the root
        if location.contains("://") {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "Absolute URIs are not allowed for register_table. Location must be a relative path within the root directory: {}",
                    location
                ),
            }
            .into());
        }

        if location.starts_with('/') {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "Absolute paths are not allowed for register_table. Location must be a relative path within the root directory: {}",
                    location
                ),
            }
            .into());
        }

        // Check for path traversal attempts
        if location.contains("..") {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "Path traversal is not allowed. Location must be a relative path within the root directory: {}",
                    location
                ),
            }
            .into());
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Validate that parent namespaces exist (if not root)
        if !namespace.is_empty() {
            self.validate_namespace_levels_exist(&namespace).await?;
        }

        // Check if table already exists
        if self.manifest_contains_object(&object_id).await? {
            return Err(NamespaceError::TableAlreadyExists {
                message: object_id.to_string(),
            }
            .into());
        }

        // Register the table with its location in the manifest
        self.insert_into_manifest(object_id, ObjectType::Table, Some(location.clone()))
            .await?;

        Ok(RegisterTableResponse {
            location: Some(location),
            ..Default::default()
        })
    }

    async fn deregister_table(
        &self,
        request: DeregisterTableRequest,
    ) -> Result<DeregisterTableResponse> {
        let table_id = request.id.as_ref().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: "Table ID is required".to_string(),
            })
        })?;

        if table_id.is_empty() {
            return Err(NamespaceError::InvalidInput {
                message: "Table ID cannot be empty".to_string(),
            }
            .into());
        }

        let (namespace, table_name) = Self::split_object_id(table_id);
        let object_id = Self::build_object_id(&namespace, &table_name);

        // Get table info before deleting
        let table_info = self.query_manifest_for_table(&object_id).await?;

        let table_uri = match table_info {
            Some(info) => {
                // Delete from manifest only (leave physical data intact)
                self.delete_from_manifest(&object_id).boxed().await?;
                Self::construct_full_uri(&self.root, &info.location)?
            }
            None => {
                return Err(NamespaceError::TableNotFound {
                    message: object_id.to_string(),
                }
                .into());
            }
        };

        Ok(DeregisterTableResponse {
            id: request.id.clone(),
            location: Some(table_uri),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MANIFEST_TABLE_NAME, METADATA_INDEX_NAME, OBJECT_ID_INDEX_NAME, OBJECT_TYPE_INDEX_NAME,
        ObjectType, convert_lance_commit_error,
    };
    use crate::{DirectoryNamespaceBuilder, ManifestNamespace};
    use arrow::datatypes::DataType;
    use bytes::Bytes;
    use lance::dataset::builder::DatasetBuilder;
    use lance::index::DatasetIndexExt;
    use lance_core::utils::tempfile::TempStdDir;
    use lance_namespace::LanceNamespace;
    use lance_namespace::models::{
        CreateNamespaceRequest, CreateTableRequest, DescribeTableRequest, DropTableRequest,
        ListNamespacesRequest, ListTablesRequest, TableExistsRequest,
    };
    use rstest::rstest;
    use std::collections::HashMap;

    fn create_test_ipc_data() -> Vec<u8> {
        use arrow::array::{Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::ipc::writer::StreamWriter;
        use arrow::record_batch::RecordBatch;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
        )
        .unwrap();

        let mut buffer = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        buffer
    }

    fn namespace_request(name: &str) -> CreateNamespaceRequest {
        let mut request = CreateNamespaceRequest::new();
        request.id = Some(vec![name.to_string()]);
        request
    }

    async fn load_manifest_dataset(root: &str) -> lance::Dataset {
        DatasetBuilder::from_uri(format!("{}/{}", root, MANIFEST_TABLE_NAME))
            .load()
            .await
            .unwrap()
    }

    async fn manifest_object_ids(root: &str) -> Vec<String> {
        let dataset = load_manifest_dataset(root).await;
        let mut scanner = dataset.scan();
        scanner.project(&["object_id"]).unwrap();
        let batches = ManifestNamespace::execute_scanner(scanner).await.unwrap();
        let mut ids = Vec::new();
        for batch in batches {
            let object_ids = ManifestNamespace::get_string_column(&batch, "object_id").unwrap();
            for row in 0..batch.num_rows() {
                ids.push(object_ids.value(row).to_string());
            }
        }
        ids
    }

    #[test]
    fn test_retryable_commit_conflict_maps_to_throttling() {
        let conflict = lance_core::Error::retryable_commit_conflict_source(
            1,
            Box::new(std::io::Error::other("conflict")),
        );
        let converted = convert_lance_commit_error(&conflict, "test operation", None);

        let lance_core::Error::Namespace { source, .. } = converted else {
            panic!("expected namespace error");
        };
        let namespace_error = source.downcast_ref::<lance_namespace::NamespaceError>();
        assert!(matches!(
            namespace_error,
            Some(lance_namespace::NamespaceError::Throttling { .. })
        ));
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_basic_create_and_list(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create a DirectoryNamespace with manifest enabled (default)
        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Verify we can list tables (should be empty)
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);

        // Create a test table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);

        let _response = dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // List tables again - should see our new table
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);
        assert_eq!(response.tables[0], "test_table");
    }

    #[tokio::test]
    async fn test_manifest_rewrite_from_properties() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let mut properties = HashMap::new();
        properties.insert("root".to_string(), temp_path.to_string());
        properties.insert(
            "inline_optimization_enabled".to_string(),
            "true".to_string(),
        );

        let namespace = DirectoryNamespaceBuilder::from_properties(properties, None)
            .unwrap()
            .build()
            .await
            .unwrap();

        namespace
            .create_namespace(namespace_request("workspace"))
            .await
            .unwrap();

        let mut request = ListNamespacesRequest::new();
        request.id = Some(vec![]);
        let response = namespace.list_namespaces(request).await.unwrap();
        assert_eq!(response.namespaces, vec!["workspace".to_string()]);

        let dataset = load_manifest_dataset(temp_path).await;
        let metadata_field = dataset.schema().field("metadata").unwrap();
        assert_eq!(metadata_field.logical_type.to_string(), "json");
        assert!(
            dataset.schema().field("base_objects").is_none(),
            "manifest rewrites should drop the old base_objects column"
        );

        let indices = dataset.load_indices().await.unwrap();
        assert!(
            indices
                .iter()
                .any(|index| index.name == OBJECT_ID_INDEX_NAME),
            "manifest rewrites should create the object_id btree index synchronously"
        );
        assert!(
            indices
                .iter()
                .any(|index| index.name == OBJECT_TYPE_INDEX_NAME),
            "manifest rewrites should create the object_type bitmap index synchronously"
        );
        assert!(
            indices
                .iter()
                .any(|index| index.name == METADATA_INDEX_NAME),
            "manifest rewrites should create the metadata JSON FTS index synchronously"
        );
        assert!(
            indices.iter().all(|index| index.fields.len() == 1),
            "manifest rewrite indexes should be single-column indexes"
        );
    }

    #[tokio::test]
    async fn test_manifest_rewrite_overwrites_compact_ordered_snapshot() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(true)
            .build()
            .await
            .unwrap();

        for name in ["zeta", "alpha", "middle"] {
            namespace
                .create_namespace(namespace_request(name))
                .await
                .unwrap();
        }

        let dataset = load_manifest_dataset(temp_path).await;
        assert_eq!(
            dataset.manifest().fragments.len(),
            1,
            "copy-on-write manifest should keep the current snapshot compact"
        );

        let object_ids = manifest_object_ids(temp_path).await;
        assert_eq!(
            object_ids,
            vec![
                "zeta".to_string(),
                "alpha".to_string(),
                "middle".to_string()
            ]
        );

        let indices = dataset.load_indices().await.unwrap();
        assert!(
            indices
                .iter()
                .any(|index| index.name == OBJECT_ID_INDEX_NAME),
            "manifest rewrites should maintain the object_id btree index"
        );
        assert!(
            indices
                .iter()
                .any(|index| index.name == OBJECT_TYPE_INDEX_NAME),
            "manifest rewrites should maintain the object_type bitmap index"
        );
        assert!(
            indices
                .iter()
                .any(|index| index.name == METADATA_INDEX_NAME),
            "manifest rewrites should maintain the metadata JSON FTS index"
        );
    }

    #[tokio::test]
    async fn test_manifest_rewrite_table_lifecycle() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        namespace
            .create_table(create_request, Bytes::from(create_test_ipc_data()))
            .await
            .unwrap();

        let mut list_request = ListTablesRequest::new();
        list_request.id = Some(vec![]);
        let response = namespace.list_tables(list_request).await.unwrap();
        assert_eq!(response.tables, vec!["test_table".to_string()]);

        let mut drop_request = DropTableRequest::new();
        drop_request.id = Some(vec!["test_table".to_string()]);
        namespace.drop_table(drop_request).await.unwrap();

        let mut list_request = ListTablesRequest::new();
        list_request.id = Some(vec![]);
        let response = namespace.list_tables(list_request).await.unwrap();
        assert!(response.tables.is_empty());

        let object_ids = manifest_object_ids(temp_path).await;
        assert!(object_ids.is_empty());
    }

    #[tokio::test]
    async fn test_manifest_rewrite_table_version_query_preserves_null_metadata_row() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();
        let table_object_id = "table";
        let manifest_ns = namespace.manifest_ns.as_ref().unwrap();
        manifest_ns
            .insert_into_manifest_with_metadata(vec![super::ManifestEntry {
                object_id: ManifestNamespace::build_version_object_id(table_object_id, 1),
                object_type: super::ObjectType::TableVersion,
                location: None,
                metadata: None,
            }])
            .await
            .unwrap();

        let versions = manifest_ns
            .query_table_versions(table_object_id, false, None)
            .await
            .unwrap();
        assert_eq!(versions, vec![(1, String::new())]);
    }

    #[tokio::test]
    async fn test_manifest_delete_table_versions_streams_large_ranges() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();
        let manifest_ns = namespace.manifest_ns.as_ref().unwrap();
        manifest_ns
            .insert_into_manifest_with_metadata(vec![
                super::ManifestEntry {
                    object_id: ManifestNamespace::build_version_object_id("table", 1),
                    object_type: super::ObjectType::TableVersion,
                    location: None,
                    metadata: Some("{}".to_string()),
                },
                super::ManifestEntry {
                    object_id: ManifestNamespace::build_version_object_id("table", 10),
                    object_type: super::ObjectType::TableVersion,
                    location: None,
                    metadata: Some("{}".to_string()),
                },
                super::ManifestEntry {
                    object_id: ManifestNamespace::build_version_object_id("other", 2),
                    object_type: super::ObjectType::TableVersion,
                    location: None,
                    metadata: Some("{}".to_string()),
                },
            ])
            .await
            .unwrap();

        let deleted = manifest_ns
            .delete_table_versions("table", &[(1, 1_000_000_000)])
            .await
            .unwrap();
        assert_eq!(deleted, 2);

        let object_ids = manifest_object_ids(temp_path).await;
        assert_eq!(
            object_ids,
            vec![ManifestNamespace::build_version_object_id("other", 2)]
        );

        let deleted = manifest_ns
            .batch_delete_table_versions_by_ranges(&[("other".to_string(), vec![(2, 2)])])
            .await
            .unwrap();
        assert_eq!(deleted, 1);
        assert!(manifest_object_ids(temp_path).await.is_empty());
    }

    #[tokio::test]
    async fn test_manifest_rewrite_concurrent_compatible_creates_succeed() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let first = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();
        let second = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        let (alpha, beta) = tokio::join!(
            first.create_namespace(namespace_request("alpha")),
            second.create_namespace(namespace_request("beta"))
        );

        alpha.unwrap();
        beta.unwrap();

        let mut request = ListNamespacesRequest::new();
        request.id = Some(vec![]);
        let mut response = first.list_namespaces(request).await.unwrap();
        response.namespaces.sort();
        assert_eq!(
            response.namespaces,
            vec!["alpha".to_string(), "beta".to_string()]
        );
    }

    #[tokio::test]
    async fn test_manifest_rewrite_concurrent_duplicate_create_fails() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let first = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();
        let second = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        let (first_result, second_result) = tokio::join!(
            first.create_namespace(namespace_request("duplicate")),
            second.create_namespace(namespace_request("duplicate"))
        );

        let results = [first_result, second_result];
        let ok_count = results.iter().filter(|result| result.is_ok()).count();
        let err_count = results.iter().filter(|result| result.is_err()).count();
        assert_eq!(ok_count, 1);
        assert_eq!(err_count, 1);
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_table_exists(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Check non-existent table
        let mut request = TableExistsRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);
        let result = dir_namespace.table_exists(request).await;
        assert!(result.is_err());

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // Check existing table
        let mut request = TableExistsRequest::new();
        request.id = Some(vec!["test_table".to_string()]);
        let result = dir_namespace.table_exists(request).await;
        assert!(result.is_ok());
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_describe_table(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Describe non-existent table
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["nonexistent".to_string()]);
        let result = dir_namespace.describe_table(request).await;
        assert!(result.is_err());

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // Describe existing table
        let mut request = DescribeTableRequest::new();
        request.id = Some(vec!["test_table".to_string()]);
        let response = dir_namespace.describe_table(request).await.unwrap();
        assert!(response.location.is_some());
        assert!(response.location.unwrap().contains("test_table"));
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_drop_table(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // Verify table exists
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);

        // Drop table
        let mut drop_request = DropTableRequest::new();
        drop_request.id = Some(vec!["test_table".to_string()]);
        let _response = dir_namespace.drop_table(drop_request).await.unwrap();

        // Verify table is gone
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);
    }

    #[tokio::test]
    async fn test_list_tables_pagination_limit_zero() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["alpha".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        let response = dir_namespace
            .list_tables(ListTablesRequest {
                id: Some(vec![]),
                limit: Some(0),
                ..Default::default()
            })
            .await
            .unwrap();

        assert!(response.tables.is_empty());
        assert!(response.page_token.is_none());
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_namespace_multiple_tables(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create multiple tables
        let buffer = create_test_ipc_data();
        for i in 1..=3 {
            let mut create_request = CreateTableRequest::new();
            create_request.id = Some(vec![format!("table{}", i)]);
            dir_namespace
                .create_table(create_request, Bytes::from(buffer.clone()))
                .await
                .unwrap();
        }

        // List all tables
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 3);
        assert!(response.tables.contains(&"table1".to_string()));
        assert!(response.tables.contains(&"table2".to_string()));
        assert!(response.tables.contains(&"table3".to_string()));
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_directory_only_mode(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create a DirectoryNamespace with manifest disabled
        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(false)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Verify we can list tables (should be empty)
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0);

        // Create a test table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);

        // Create table - this should use directory-only mode
        let _response = dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // List tables - should see our new table
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);
        assert_eq!(response.tables[0], "test_table");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_dual_mode_merge(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create a DirectoryNamespace with both manifest and directory enabled
        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(true)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create tables through manifest
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["table1".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // List tables - should see table from both manifest and directory
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);
        assert_eq!(response.tables[0], "table1");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_manifest_only_mode(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Create a DirectoryNamespace with only manifest enabled
        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .manifest_enabled(true)
            .dir_listing_enabled(false)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // List tables - should only use manifest
        let mut request = ListTablesRequest::new();
        request.id = Some(vec![]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 1);
        assert_eq!(response.tables[0], "test_table");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_drop_nonexistent_table(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Try to drop non-existent table
        let mut drop_request = DropTableRequest::new();
        drop_request.id = Some(vec!["nonexistent".to_string()]);
        let result = dir_namespace.drop_table(drop_request).await;
        assert!(result.is_err());
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_duplicate_table_fails(#[case] inline_optimization: bool) {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer.clone()))
            .await
            .unwrap();

        // Try to create table with same name - should fail
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        let result = dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await;
        assert!(result.is_err());
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_child_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            CreateNamespaceRequest, ListNamespacesRequest, NamespaceExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a child namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns1".to_string()]);
        let result = dir_namespace.create_namespace(create_req).await;
        assert!(
            result.is_ok(),
            "Failed to create child namespace: {:?}",
            result.err()
        );

        // Verify namespace exists
        let exists_req = NamespaceExistsRequest {
            id: Some(vec!["ns1".to_string()]),
            ..Default::default()
        };
        let result = dir_namespace.namespace_exists(exists_req).await;
        assert!(result.is_ok(), "Namespace should exist");

        // List child namespaces of root
        let list_req = ListNamespacesRequest {
            id: Some(vec![]),
            page_token: None,
            limit: None,
            ..Default::default()
        };
        let result = dir_namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let namespaces = result.unwrap();
        assert_eq!(namespaces.namespaces.len(), 1);
        assert_eq!(namespaces.namespaces[0], "ns1");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_nested_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            CreateNamespaceRequest, ListNamespacesRequest, NamespaceExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create parent namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string()]);
        dir_namespace.create_namespace(create_req).await.unwrap();

        // Create nested child namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string(), "child".to_string()]);
        let result = dir_namespace.create_namespace(create_req).await;
        assert!(
            result.is_ok(),
            "Failed to create nested namespace: {:?}",
            result.err()
        );

        // Verify nested namespace exists
        let exists_req = NamespaceExistsRequest {
            id: Some(vec!["parent".to_string(), "child".to_string()]),
            ..Default::default()
        };
        let result = dir_namespace.namespace_exists(exists_req).await;
        assert!(result.is_ok(), "Nested namespace should exist");

        // List child namespaces of parent
        let list_req = ListNamespacesRequest {
            id: Some(vec!["parent".to_string()]),
            page_token: None,
            limit: None,
            ..Default::default()
        };
        let result = dir_namespace.list_namespaces(list_req).await;
        assert!(result.is_ok());
        let namespaces = result.unwrap();
        assert_eq!(namespaces.namespaces.len(), 1);
        assert_eq!(namespaces.namespaces[0], "child");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_namespace_without_parent_fails(#[case] inline_optimization: bool) {
        use lance_namespace::models::CreateNamespaceRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Try to create nested namespace without parent
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["nonexistent_parent".to_string(), "child".to_string()]);
        let result = dir_namespace.create_namespace(create_req).await;
        assert!(result.is_err(), "Should fail when parent doesn't exist");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_drop_child_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            CreateNamespaceRequest, DropNamespaceRequest, NamespaceExistsRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a child namespace
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns1".to_string()]);
        dir_namespace.create_namespace(create_req).await.unwrap();

        // Drop the namespace
        let mut drop_req = DropNamespaceRequest::new();
        drop_req.id = Some(vec!["ns1".to_string()]);
        let result = dir_namespace.drop_namespace(drop_req).await;
        assert!(
            result.is_ok(),
            "Failed to drop namespace: {:?}",
            result.err()
        );

        // Verify namespace no longer exists
        let exists_req = NamespaceExistsRequest {
            id: Some(vec!["ns1".to_string()]),
            ..Default::default()
        };
        let result = dir_namespace.namespace_exists(exists_req).await;
        assert!(result.is_err(), "Namespace should not exist after drop");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_drop_namespace_with_children_fails(#[case] inline_optimization: bool) {
        use lance_namespace::models::{CreateNamespaceRequest, DropNamespaceRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create parent and child namespaces
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string()]);
        dir_namespace.create_namespace(create_req).await.unwrap();

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["parent".to_string(), "child".to_string()]);
        dir_namespace.create_namespace(create_req).await.unwrap();

        // Try to drop parent namespace - should fail because it has children
        let mut drop_req = DropNamespaceRequest::new();
        drop_req.id = Some(vec!["parent".to_string()]);
        let result = dir_namespace.drop_namespace(drop_req).await;
        assert!(result.is_err(), "Should fail when namespace has children");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_create_table_in_child_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            CreateNamespaceRequest, CreateTableRequest, ListTablesRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a child namespace
        let mut create_ns_req = CreateNamespaceRequest::new();
        create_ns_req.id = Some(vec!["ns1".to_string()]);
        dir_namespace.create_namespace(create_ns_req).await.unwrap();

        // Create a table in the child namespace
        let buffer = create_test_ipc_data();
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
        let result = dir_namespace
            .create_table(create_table_req, Bytes::from(buffer))
            .await;
        assert!(
            result.is_ok(),
            "Failed to create table in child namespace: {:?}",
            result.err()
        );

        // List tables in the namespace
        let list_req = ListTablesRequest {
            id: Some(vec!["ns1".to_string()]),
            page_token: None,
            limit: None,
            ..Default::default()
        };
        let result = dir_namespace.list_tables(list_req).await;
        assert!(result.is_ok());
        let tables = result.unwrap();
        assert_eq!(tables.tables.len(), 1);
        assert_eq!(tables.tables[0], "table1");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_describe_child_namespace(#[case] inline_optimization: bool) {
        use lance_namespace::models::{CreateNamespaceRequest, DescribeNamespaceRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a child namespace with properties
        let mut properties = std::collections::HashMap::new();
        properties.insert("key1".to_string(), "value1".to_string());

        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec!["ns1".to_string()]);
        create_req.properties = Some(properties.clone());
        dir_namespace.create_namespace(create_req).await.unwrap();

        // Describe the namespace
        let describe_req = DescribeNamespaceRequest {
            id: Some(vec!["ns1".to_string()]),
            ..Default::default()
        };
        let result = dir_namespace.describe_namespace(describe_req).await;
        assert!(
            result.is_ok(),
            "Failed to describe namespace: {:?}",
            result.err()
        );
        let response = result.unwrap();
        assert!(response.properties.is_some());
        assert_eq!(
            response.properties.unwrap().get("key1"),
            Some(&"value1".to_string())
        );
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_concurrent_create_and_drop_single_instance(#[case] inline_optimization: bool) {
        use futures::future::join_all;
        use std::sync::Arc;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = Arc::new(
            DirectoryNamespaceBuilder::new(temp_path)
                .inline_optimization_enabled(inline_optimization)
                .build()
                .await
                .unwrap(),
        );

        // Initialize namespace first - create parent namespace to ensure __manifest table
        // is created before concurrent operations
        let mut create_ns_request = CreateNamespaceRequest::new();
        create_ns_request.id = Some(vec!["test_ns".to_string()]);
        dir_namespace
            .create_namespace(create_ns_request)
            .await
            .unwrap();

        let num_tables = 10;
        let mut handles = Vec::new();

        for i in 0..num_tables {
            let ns = dir_namespace.clone();
            let handle = async move {
                let table_name = format!("concurrent_table_{}", i);
                let table_id = vec!["test_ns".to_string(), table_name.clone()];
                let buffer = create_test_ipc_data();

                // Create table
                let mut create_request = CreateTableRequest::new();
                create_request.id = Some(table_id.clone());
                ns.create_table(create_request, Bytes::from(buffer))
                    .await
                    .unwrap_or_else(|e| panic!("Failed to create table {}: {}", table_name, e));

                // Drop table
                let mut drop_request = DropTableRequest::new();
                drop_request.id = Some(table_id);
                ns.drop_table(drop_request)
                    .await
                    .unwrap_or_else(|e| panic!("Failed to drop table {}: {}", table_name, e));

                Ok::<_, lance_core::Error>(())
            };
            handles.push(handle);
        }

        let results = join_all(handles).await;
        for result in results {
            assert!(result.is_ok(), "All concurrent operations should succeed");
        }

        // Verify all tables are dropped
        let mut request = ListTablesRequest::new();
        request.id = Some(vec!["test_ns".to_string()]);
        let response = dir_namespace.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0, "All tables should be dropped");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_concurrent_create_and_drop_multiple_instances(#[case] inline_optimization: bool) {
        use futures::future::join_all;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap().to_string();

        // Initialize namespace first with a single instance to ensure __manifest
        // table is created and parent namespace exists before concurrent operations
        let init_ns = DirectoryNamespaceBuilder::new(&temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();
        let mut create_ns_request = CreateNamespaceRequest::new();
        create_ns_request.id = Some(vec!["test_ns".to_string()]);
        init_ns.create_namespace(create_ns_request).await.unwrap();

        let num_tables = 10;
        let mut handles = Vec::new();

        for i in 0..num_tables {
            let path = temp_path.clone();
            let handle = async move {
                // Each task creates its own namespace instance
                let ns = DirectoryNamespaceBuilder::new(&path)
                    .inline_optimization_enabled(inline_optimization)
                    .build()
                    .await
                    .unwrap();

                let table_name = format!("multi_ns_table_{}", i);
                let table_id = vec!["test_ns".to_string(), table_name.clone()];
                let buffer = create_test_ipc_data();

                // Create table
                let mut create_request = CreateTableRequest::new();
                create_request.id = Some(table_id.clone());
                ns.create_table(create_request, Bytes::from(buffer))
                    .await
                    .unwrap_or_else(|e| panic!("Failed to create table {}: {}", table_name, e));

                // Drop table
                let mut drop_request = DropTableRequest::new();
                drop_request.id = Some(table_id);
                ns.drop_table(drop_request)
                    .await
                    .unwrap_or_else(|e| panic!("Failed to drop table {}: {}", table_name, e));

                Ok::<_, lance_core::Error>(())
            };
            handles.push(handle);
        }

        let results = join_all(handles).await;
        for result in results {
            assert!(result.is_ok(), "All concurrent operations should succeed");
        }

        // Verify with a fresh namespace instance
        let verify_ns = DirectoryNamespaceBuilder::new(&temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        let mut request = ListTablesRequest::new();
        request.id = Some(vec!["test_ns".to_string()]);
        let response = verify_ns.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0, "All tables should be dropped");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_concurrent_create_then_drop_from_different_instance(
        #[case] inline_optimization: bool,
    ) {
        use futures::future::join_all;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap().to_string();

        // Initialize namespace first with a single instance to ensure __manifest
        // table is created and parent namespace exists before concurrent operations
        let init_ns = DirectoryNamespaceBuilder::new(&temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();
        let mut create_ns_request = CreateNamespaceRequest::new();
        create_ns_request.id = Some(vec!["test_ns".to_string()]);
        init_ns.create_namespace(create_ns_request).await.unwrap();

        let num_tables = 10;

        // Phase 1: Create all tables concurrently using separate namespace instances
        let mut create_handles = Vec::new();
        for i in 0..num_tables {
            let path = temp_path.clone();
            let handle = async move {
                let ns = DirectoryNamespaceBuilder::new(&path)
                    .inline_optimization_enabled(inline_optimization)
                    .build()
                    .await
                    .unwrap();

                let table_name = format!("cross_instance_table_{}", i);
                let table_id = vec!["test_ns".to_string(), table_name.clone()];
                let buffer = create_test_ipc_data();

                let mut create_request = CreateTableRequest::new();
                create_request.id = Some(table_id);
                ns.create_table(create_request, Bytes::from(buffer))
                    .await
                    .unwrap_or_else(|e| panic!("Failed to create table {}: {}", table_name, e));

                Ok::<_, lance_core::Error>(())
            };
            create_handles.push(handle);
        }

        let create_results = join_all(create_handles).await;
        for result in create_results {
            assert!(result.is_ok(), "All create operations should succeed");
        }

        // Phase 2: Drop all tables concurrently using NEW namespace instances
        let mut drop_handles = Vec::new();
        for i in 0..num_tables {
            let path = temp_path.clone();
            let handle = async move {
                let ns = DirectoryNamespaceBuilder::new(&path)
                    .inline_optimization_enabled(inline_optimization)
                    .build()
                    .await
                    .unwrap();

                let table_name = format!("cross_instance_table_{}", i);
                let table_id = vec!["test_ns".to_string(), table_name.clone()];

                let mut drop_request = DropTableRequest::new();
                drop_request.id = Some(table_id);
                ns.drop_table(drop_request)
                    .await
                    .unwrap_or_else(|e| panic!("Failed to drop table {}: {}", table_name, e));

                Ok::<_, lance_core::Error>(())
            };
            drop_handles.push(handle);
        }

        let drop_results = join_all(drop_handles).await;
        for result in drop_results {
            assert!(result.is_ok(), "All drop operations should succeed");
        }

        // Verify all tables are dropped
        let verify_ns = DirectoryNamespaceBuilder::new(&temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        let mut request = ListTablesRequest::new();
        request.id = Some(vec!["test_ns".to_string()]);
        let response = verify_ns.list_tables(request).await.unwrap();
        assert_eq!(response.tables.len(), 0, "All tables should be dropped");
    }

    #[test]
    fn test_construct_full_uri_with_cloud_urls() {
        // Test S3-style URL with nested path (no trailing slash)
        let s3_result =
            ManifestNamespace::construct_full_uri("s3://bucket/path/subdir", "table.lance")
                .unwrap();
        assert_eq!(
            s3_result, "s3://bucket/path/subdir/table.lance",
            "S3 URL should correctly append table name to nested path"
        );

        // Test Azure-style URL with nested path (no trailing slash)
        let az_result =
            ManifestNamespace::construct_full_uri("az://container/path/subdir", "table.lance")
                .unwrap();
        assert_eq!(
            az_result, "az://container/path/subdir/table.lance",
            "Azure URL should correctly append table name to nested path"
        );

        // Test GCS-style URL with nested path (no trailing slash)
        let gs_result =
            ManifestNamespace::construct_full_uri("gs://bucket/path/subdir", "table.lance")
                .unwrap();
        assert_eq!(
            gs_result, "gs://bucket/path/subdir/table.lance",
            "GCS URL should correctly append table name to nested path"
        );

        // Test with deeper nesting
        let deep_result =
            ManifestNamespace::construct_full_uri("s3://bucket/a/b/c/d", "my_table.lance").unwrap();
        assert_eq!(
            deep_result, "s3://bucket/a/b/c/d/my_table.lance",
            "Deeply nested path should work correctly"
        );

        // Test with root-level path (single segment after bucket)
        let shallow_result =
            ManifestNamespace::construct_full_uri("s3://bucket", "table.lance").unwrap();
        assert_eq!(
            shallow_result, "s3://bucket/table.lance",
            "Single-level nested path should work correctly"
        );

        // Test that URLs with trailing slash already work (no regression)
        let trailing_slash_result =
            ManifestNamespace::construct_full_uri("s3://bucket/path/subdir/", "table.lance")
                .unwrap();
        assert_eq!(
            trailing_slash_result, "s3://bucket/path/subdir/table.lance",
            "URL with existing trailing slash should still work"
        );

        // Test that URLs with empty query string don't include trailing "?"
        // This is important because URL::to_string() can add "?" for empty queries
        let empty_query_result =
            ManifestNamespace::construct_full_uri("s3://bucket/path?", "table.lance").unwrap();
        assert_eq!(
            empty_query_result, "s3://bucket/path/table.lance",
            "URL with empty query string should not include trailing '?'"
        );

        // Test that URLs with actual query parameters have them stripped
        // (query parameters are not meaningful for storage paths)
        let query_param_result =
            ManifestNamespace::construct_full_uri("s3://bucket/path?param=value", "table.lance")
                .unwrap();
        assert_eq!(
            query_param_result, "s3://bucket/path/table.lance",
            "URL with query parameters should have them stripped"
        );
    }

    #[test]
    fn test_construct_full_uri_with_dollar_sign() {
        let result =
            ManifestNamespace::construct_full_uri("/tmp/root", "hash_workspace$test_table")
                .unwrap();

        assert!(
            result.ends_with("/tmp/root/hash_workspace$test_table"),
            "local file URI should preserve dollar signs without adding empty path segments: {}",
            result
        );
        assert!(
            !result.contains("//hash_workspace$test_table"),
            "local file URI should not add a double slash before table directory: {}",
            result
        );
    }

    #[test]
    fn test_construct_full_uri_with_nested_relative_location() {
        let result =
            ManifestNamespace::construct_full_uri("/tmp/root", "workspace/physical_table.lance")
                .unwrap();

        assert!(
            result.ends_with("/tmp/root/workspace/physical_table.lance"),
            "nested relative location should preserve path separators: {}",
            result
        );
        assert!(
            !result.contains("%2Fphysical_table.lance"),
            "nested relative location should not encode path separators: {}",
            result
        );
    }

    /// Test that concurrent create_table calls for the same table name don't
    /// create duplicate entries in the manifest. Uses two independent
    /// ManifestNamespace instances pointing at the same directory to simulate
    /// two separate OS processes racing on table creation. The conflict_retries
    /// setting on the MergeInsert ensures the second operation properly detects
    /// the duplicate via WhenMatched::Fail after retrying against the latest data.
    #[tokio::test]
    async fn test_concurrent_create_table_no_duplicates() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        // Two independent namespace instances = two separate "processes"
        // sharing the same underlying filesystem directory.
        let ns1 = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(false)
            .build()
            .await
            .unwrap();
        let ns2 = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(false)
            .build()
            .await
            .unwrap();

        let buffer = create_test_ipc_data();

        let mut req1 = CreateTableRequest::new();
        req1.id = Some(vec!["race_table".to_string()]);
        let mut req2 = CreateTableRequest::new();
        req2.id = Some(vec!["race_table".to_string()]);

        // Launch both create_table calls concurrently
        let (result1, result2) = tokio::join!(
            ns1.create_table(req1, Bytes::from(buffer.clone())),
            ns2.create_table(req2, Bytes::from(buffer.clone())),
        );

        // Exactly one should succeed and one should fail
        let success_count = [&result1, &result2].iter().filter(|r| r.is_ok()).count();
        let failure_count = [&result1, &result2].iter().filter(|r| r.is_err()).count();
        assert_eq!(
            success_count, 1,
            "Exactly one create should succeed, got: result1={:?}, result2={:?}",
            result1, result2
        );
        assert_eq!(
            failure_count, 1,
            "Exactly one create should fail, got: result1={:?}, result2={:?}",
            result1, result2
        );

        // Verify only one table entry exists in the manifest
        let ns_check = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(false)
            .build()
            .await
            .unwrap();
        let mut list_request = ListTablesRequest::new();
        list_request.id = Some(vec![]);
        let response = ns_check.list_tables(list_request).await.unwrap();
        assert_eq!(
            response.tables.len(),
            1,
            "Should have exactly 1 table, found: {:?}",
            response.tables
        );
        assert_eq!(response.tables[0], "race_table");

        // Also verify describe_table works (no "found 2" error)
        let mut describe_request = DescribeTableRequest::new();
        describe_request.id = Some(vec!["race_table".to_string()]);
        let describe_result = ns_check.describe_table(describe_request).await;
        assert!(
            describe_result.is_ok(),
            "describe_table should not fail with duplicate entries: {:?}",
            describe_result
        );
    }

    // --- apply_pagination unit tests ---

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn test_apply_pagination_no_token_no_limit() {
        let mut n = names(&["b", "a", "c"]);
        let next = ManifestNamespace::apply_pagination(&mut n, None, None);
        assert_eq!(n, names(&["a", "b", "c"]));
        assert_eq!(next, None);
    }

    #[test]
    fn test_apply_pagination_limit_truncates_and_returns_token() {
        let mut n = names(&["c", "a", "b"]);
        let next = ManifestNamespace::apply_pagination(&mut n, None, Some(2));
        assert_eq!(n, names(&["a", "b"]));
        assert_eq!(next, Some("b".to_string()));
    }

    #[test]
    fn test_apply_pagination_limit_zero_returns_empty_no_token() {
        let mut n = names(&["a", "b", "c"]);
        let next = ManifestNamespace::apply_pagination(&mut n, None, Some(0));
        assert!(n.is_empty());
        assert_eq!(next, None);
    }

    #[test]
    fn test_apply_pagination_page_token_in_list() {
        // "b" is in the list; should start from "c" (strict >)
        let mut n = names(&["a", "b", "c", "d"]);
        let next = ManifestNamespace::apply_pagination(&mut n, Some("b".to_string()), None);
        assert_eq!(n, names(&["c", "d"]));
        assert_eq!(next, None);
    }

    #[test]
    fn test_apply_pagination_page_token_past_all_items() {
        let mut n = names(&["a", "b", "c"]);
        let next = ManifestNamespace::apply_pagination(&mut n, Some("z".to_string()), None);
        assert!(n.is_empty());
        assert_eq!(next, None);
    }

    #[test]
    fn test_apply_pagination_token_and_limit_combined() {
        let mut n = names(&["a", "b", "c", "d", "e"]);
        let next = ManifestNamespace::apply_pagination(&mut n, Some("b".to_string()), Some(2));
        assert_eq!(n, names(&["c", "d"]));
        assert_eq!(next, Some("d".to_string()));
    }

    /// Seed a __manifest dataset using the legacy schema (Utf8 metadata +
    /// base_objects List column) so that backward-compat tests start from a
    /// real pre-migration state.
    async fn seed_old_schema_manifest(
        root: &str,
        rows: &[(&str, &str, Option<&str>, Option<&str>)],
    ) {
        use arrow::array::builder::{ListBuilder, StringBuilder};
        use arrow::array::{RecordBatchIterator, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use lance::dataset::WriteParams;
        use std::sync::Arc;

        let old_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("object_id", DataType::Utf8, false),
            Field::new("object_type", DataType::Utf8, false),
            Field::new("location", DataType::Utf8, true),
            Field::new("metadata", DataType::Utf8, true),
            Field::new(
                "base_objects",
                DataType::List(Arc::new(Field::new("object_id", DataType::Utf8, true))),
                true,
            ),
        ]));

        let mut object_ids = Vec::new();
        let mut object_types = Vec::new();
        let mut locations: Vec<Option<String>> = Vec::new();
        let mut metadatas: Vec<Option<String>> = Vec::new();
        let mut base_objects_builder = ListBuilder::new(StringBuilder::new())
            .with_field(Field::new("object_id", DataType::Utf8, true));

        for &(oid, otype, loc, meta) in rows {
            object_ids.push(oid.to_string());
            object_types.push(otype.to_string());
            locations.push(loc.map(ToString::to_string));
            metadatas.push(meta.map(ToString::to_string));
            base_objects_builder.append(true);
        }

        let batch = arrow::array::RecordBatch::try_new(
            old_schema.clone(),
            vec![
                Arc::new(StringArray::from(object_ids)),
                Arc::new(StringArray::from(object_types)),
                Arc::new(StringArray::from(locations)),
                Arc::new(StringArray::from(metadatas)),
                Arc::new(base_objects_builder.finish()),
            ],
        )
        .unwrap();

        let reader = RecordBatchIterator::new(vec![Ok(batch)], old_schema);
        let manifest_path = format!("{}/{}", root, MANIFEST_TABLE_NAME);
        let write_params = WriteParams::default();
        lance::Dataset::write(Box::new(reader), &manifest_path, Some(write_params))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_backward_compat_read_old_schema_manifest() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let meta_json = r#"{"key":"value"}"#;
        seed_old_schema_manifest(
            temp_path,
            &[
                ("ns1", ObjectType::Namespace.as_str(), None, None),
                (
                    "my_table",
                    ObjectType::Table.as_str(),
                    Some("some/location"),
                    Some(meta_json),
                ),
            ],
        )
        .await;

        // Verify old dataset has Utf8 metadata and base_objects columns
        let old_dataset = load_manifest_dataset(temp_path).await;
        assert!(old_dataset.schema().field("base_objects").is_some());
        assert_eq!(
            old_dataset.schema().field("metadata").unwrap().data_type(),
            DataType::Utf8,
        );

        // Open via the namespace builder — read path should work.
        // Disable dir listing so list_tables goes through the manifest path.
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .dir_listing_enabled(false)
            .build()
            .await
            .unwrap();

        let mut list_ns_req = ListNamespacesRequest::new();
        list_ns_req.id = Some(vec![]);
        let ns_response = namespace.list_namespaces(list_ns_req).await.unwrap();
        assert_eq!(ns_response.namespaces, vec!["ns1".to_string()]);

        let mut list_tables_req = ListTablesRequest::new();
        list_tables_req.id = Some(vec![]);
        let table_response = namespace.list_tables(list_tables_req).await.unwrap();
        assert_eq!(table_response.tables, vec!["my_table".to_string()]);

        let describe_req = DescribeTableRequest {
            id: Some(vec!["my_table".to_string()]),
            ..Default::default()
        };
        let describe_resp = namespace.describe_table(describe_req).await.unwrap();
        let properties = describe_resp.properties.unwrap();
        assert_eq!(properties.get("key").unwrap(), "value");
    }

    #[tokio::test]
    async fn test_backward_compat_write_migrates_old_schema() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        seed_old_schema_manifest(
            temp_path,
            &[("existing_ns", ObjectType::Namespace.as_str(), None, None)],
        )
        .await;

        // Open namespace and perform a write (which triggers copy-on-write rewrite)
        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();

        namespace
            .create_namespace(namespace_request("new_ns"))
            .await
            .unwrap();

        // After the write, the manifest should have been rewritten with new schema
        let dataset = load_manifest_dataset(temp_path).await;
        assert!(
            dataset.schema().field("base_objects").is_none(),
            "copy-on-write rewrite should drop the old base_objects column"
        );
        let metadata_field = dataset.schema().field("metadata").unwrap();
        assert_eq!(
            metadata_field.logical_type.to_string(),
            "json",
            "copy-on-write rewrite should migrate metadata to JSON"
        );

        // Verify both old and new rows are present
        let mut list_ns_req = ListNamespacesRequest::new();
        list_ns_req.id = Some(vec![]);
        let response = namespace.list_namespaces(list_ns_req).await.unwrap();
        assert!(response.namespaces.contains(&"existing_ns".to_string()));
        assert!(response.namespaces.contains(&"new_ns".to_string()));

        // Verify indices are built
        let indices = dataset.load_indices().await.unwrap();
        assert!(indices.iter().any(|i| i.name == OBJECT_ID_INDEX_NAME));
        assert!(indices.iter().any(|i| i.name == OBJECT_TYPE_INDEX_NAME));
        assert!(indices.iter().any(|i| i.name == METADATA_INDEX_NAME));
    }

    #[tokio::test]
    async fn test_manifest_indices_are_complete_and_versioned() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(true)
            .build()
            .await
            .unwrap();

        // Create enough objects to exercise all index types
        namespace
            .create_namespace(namespace_request("ns1"))
            .await
            .unwrap();
        namespace
            .create_namespace(namespace_request("ns2"))
            .await
            .unwrap();
        let buffer = create_test_ipc_data();
        let mut props1 = HashMap::new();
        props1.insert("env".to_string(), "prod".to_string());
        props1.insert("owner".to_string(), "alice".to_string());
        let mut req = CreateTableRequest::new();
        req.id = Some(vec!["tbl1".to_string()]);
        req.properties = Some(props1);
        namespace
            .create_table(req, Bytes::from(buffer.clone()))
            .await
            .unwrap();
        let mut props2 = HashMap::new();
        props2.insert("env".to_string(), "staging".to_string());
        props2.insert("owner".to_string(), "bob".to_string());
        let mut req = CreateTableRequest::new();
        req.id = Some(vec!["tbl2".to_string()]);
        req.properties = Some(props2);
        namespace
            .create_table(req, Bytes::from(buffer))
            .await
            .unwrap();

        let dataset = load_manifest_dataset(temp_path).await;
        let indices = dataset.load_indices().await.unwrap();

        // All 3 indices must exist
        assert_eq!(
            indices.len(),
            3,
            "Expected exactly 3 indices, got: {:?}",
            indices.iter().map(|i| &i.name).collect::<Vec<_>>()
        );

        let btree = indices
            .iter()
            .find(|i| i.name == OBJECT_ID_INDEX_NAME)
            .expect("object_id btree index missing");
        let bitmap = indices
            .iter()
            .find(|i| i.name == OBJECT_TYPE_INDEX_NAME)
            .expect("object_type bitmap index missing");
        let fts = indices
            .iter()
            .find(|i| i.name == METADATA_INDEX_NAME)
            .expect("metadata FTS index missing");

        // Each index covers exactly one column
        assert_eq!(btree.fields.len(), 1);
        assert_eq!(bitmap.fields.len(), 1);
        assert_eq!(fts.fields.len(), 1);

        // All indices are stamped with the current dataset version
        let version = dataset.manifest().version;
        assert_eq!(
            btree.dataset_version, version,
            "btree index dataset_version mismatch"
        );
        assert_eq!(
            bitmap.dataset_version, version,
            "bitmap index dataset_version mismatch"
        );
        assert_eq!(
            fts.dataset_version, version,
            "fts index dataset_version mismatch"
        );

        // All indices must have fragment bitmaps covering the single manifest fragment
        for idx in [btree, bitmap, fts] {
            let bitmap = idx
                .fragment_bitmap
                .as_ref()
                .unwrap_or_else(|| panic!("{} should have a fragment bitmap", idx.name));
            assert_eq!(
                bitmap.len(),
                dataset.manifest().fragments.len() as u64,
                "{} fragment bitmap length mismatch",
                idx.name
            );
        }

        // All indices reference distinct UUIDs
        let mut uuids: Vec<_> = indices.iter().map(|i| i.uuid).collect();
        uuids.sort();
        uuids.dedup();
        assert_eq!(uuids.len(), 3, "index UUIDs should be unique");
    }

    #[tokio::test]
    async fn test_manifest_reads_use_indexed_scans() {
        use lance_index::scalar::FullTextSearchQuery;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(true)
            .build()
            .await
            .unwrap();

        // Populate manifest with a namespace and tables with metadata
        namespace
            .create_namespace(namespace_request("ns"))
            .await
            .unwrap();
        let buffer = create_test_ipc_data();
        let mut props = HashMap::new();
        props.insert(
            "description".to_string(),
            "important production data".to_string(),
        );
        let mut req = CreateTableRequest::new();
        req.id = Some(vec!["my_table".to_string()]);
        req.properties = Some(props);
        namespace
            .create_table(req, Bytes::from(buffer))
            .await
            .unwrap();

        let dataset = load_manifest_dataset(temp_path).await;

        // 1. object_id equality filter should use BTree (ScalarIndexQuery)
        {
            let mut scanner = dataset.scan();
            scanner.filter("object_id = 'ns'").unwrap();
            scanner.prefilter(true);
            let plan = scanner.explain_plan(true).await.unwrap();
            assert!(
                plan.contains("ScalarIndexQuery"),
                "object_id filter should use ScalarIndexQuery (btree), plan:\n{}",
                plan
            );
        }

        // 2. object_type equality filter should use Bitmap (ScalarIndexQuery)
        {
            let mut scanner = dataset.scan();
            scanner.filter("object_type = 'table'").unwrap();
            scanner.prefilter(true);
            let plan = scanner.explain_plan(true).await.unwrap();
            assert!(
                plan.contains("ScalarIndexQuery"),
                "object_type filter should use ScalarIndexQuery (bitmap), plan:\n{}",
                plan
            );
        }

        // 3. Combined object_id + object_type filter should use ScalarIndexQuery
        {
            let mut scanner = dataset.scan();
            scanner
                .filter("object_id = 'my_table' AND object_type = 'table'")
                .unwrap();
            scanner.prefilter(true);
            let plan = scanner.explain_plan(true).await.unwrap();
            assert!(
                plan.contains("ScalarIndexQuery"),
                "combined filter should use ScalarIndexQuery, plan:\n{}",
                plan
            );
        }

        // 4. FTS query on metadata should use the inverted index (MatchQuery)
        {
            let mut scanner = dataset.scan();
            let query = FullTextSearchQuery::new("production".to_owned())
                .with_column("metadata".to_string())
                .unwrap();
            scanner.full_text_search(query).unwrap();
            let plan = scanner.explain_plan(true).await.unwrap();
            assert!(
                plan.contains("MatchQuery"),
                "metadata FTS query should use MatchQuery (inverted index), plan:\n{}",
                plan
            );
        }
    }
}
