// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Manifest-based namespace implementation
//!
//! This module provides a namespace implementation that uses a manifest table
//! to track tables and nested namespaces.

use super::manifest_feature_flags::{ensure_readable, ensure_writable};
use arrow::array::builder::{ListBuilder, StringBuilder};
use arrow::array::{Array, ListArray, RecordBatch, RecordBatchIterator, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
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
use lance::Dataset;
use lance::dataset::transaction::{Operation, Transaction};
use lance::dataset::{
    InsertBuilder, ReadParams, WhenMatched, WriteMode, WriteParams, builder::DatasetBuilder,
};
use lance::session::Session;
use lance_core::Error as LanceError;
use lance_core::datatypes::LANCE_UNENFORCED_PRIMARY_KEY_POSITION;
use lance_core::{Error, Result, box_error};
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use lance_io::stream::RecordBatchStream as LanceRecordBatchStream;
use lance_namespace::LanceNamespace;
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{
    AlterTableAddColumnsRequest, AlterTableAddColumnsResponse, AlterTableAlterColumnsRequest,
    AlterTableAlterColumnsResponse, AlterTableDropColumnsRequest, AlterTableDropColumnsResponse,
    CreateNamespaceRequest, CreateNamespaceResponse, CreateTableRequest, CreateTableResponse,
    DeclareTableRequest, DeclareTableResponse, DeregisterTableRequest, DeregisterTableResponse,
    DescribeNamespaceRequest, DescribeNamespaceResponse, DescribeTableRequest,
    DescribeTableResponse, DropNamespaceRequest, DropNamespaceResponse, DropTableRequest,
    DropTableResponse, ListNamespacesRequest, ListNamespacesResponse, ListTablesRequest,
    ListTablesResponse, NamespaceExistsRequest, RegisterTableRequest, RegisterTableResponse,
    TableExistsRequest,
};
use lance_namespace::schema::arrow_schema_to_json;
use lance_table::feature_flags::{apply_feature_flags, ensure_can_write_manifest};
use lance_table::format::{Fragment, Manifest};
use lance_table::io::commit::{
    CommitError, CommitHandler, commit_handler_from_url, write_manifest_file_to_path,
};
use object_store::{Error as ObjectStoreError, path::Path};
use std::io::Cursor;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{
    collections::{HashMap, HashSet},
    hash::{DefaultHasher, Hash, Hasher},
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex as StdMutex, MutexGuard as StdMutexGuard},
};
use tokio::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

const MANIFEST_TABLE_NAME: &str = "__manifest";
const LANCE_DATA_DIR: &str = "data";
const DELIMITER: &str = "$";
/// Bounded concurrency for per-table `_versions/` probes when filtering declared tables.
/// Higher values reduce latency but increase burst load against the object store.
pub(crate) const DECLARED_FILTER_CONCURRENCY: usize = 16;

// Each retry reloads and rewrites the full manifest. Match the regular Lance
// commit retry budget so multi-process namespace writes can make progress.
const DEFAULT_MANIFEST_REWRITE_COMMIT_RETRIES: u32 = 20;
const MANIFEST_REWRITE_BATCH_SIZE: usize = 8192;

/// Object types that can be stored in the manifest
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectType {
    Namespace,
    Table,
}

impl ObjectType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Namespace => "namespace",
            Self::Table => "table",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "namespace" => Ok(Self::Namespace),
            "table" => Ok(Self::Table),
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
    /// The type of the object (Namespace or Table)
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

#[derive(Debug, Clone)]
struct ManifestRowValue {
    object_id: String,
    object_type: ObjectType,
    location: Option<String>,
    metadata: Option<String>,
    base_objects: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct ManifestSnapshot {
    rows: Arc<[ManifestRowValue]>,
}

impl ManifestSnapshot {
    async fn load(dataset: &Dataset) -> Result<Self> {
        ensure_readable(dataset.metadata())?;
        let mut scanner = dataset.scan();
        scanner
            .project(&[
                "object_id",
                "object_type",
                "location",
                "metadata",
                "base_objects",
            ])
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to project manifest columns: {:?}", e),
                })
            })?;
        let mut stream = scanner.try_into_stream().await.map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to load manifest snapshot: {:?}", e),
            })
        })?;
        let mut rows = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = batch.map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to read manifest snapshot: {:?}", e),
                })
            })?;
            let object_ids = ManifestNamespace::get_string_column(&batch, "object_id")?;
            let object_types = ManifestNamespace::get_string_column(&batch, "object_type")?;
            let locations = ManifestNamespace::get_string_column(&batch, "location")?;
            let metadatas = ManifestNamespace::get_string_column(&batch, "metadata")?;
            let base_objects = ManifestNamespace::base_objects_column_values(&batch)?;
            rows.reserve(batch.num_rows());
            for (row, base_objects) in base_objects.into_iter().enumerate() {
                let object_id =
                    ManifestNamespace::required_string_value(object_ids, row, "object_id")?
                        .to_string();
                let object_type = ObjectType::parse(ManifestNamespace::required_string_value(
                    object_types,
                    row,
                    "object_type",
                )?)?;
                let location = ManifestNamespace::optional_string_value(locations, row);
                if object_type == ObjectType::Table && location.is_none() {
                    return Err(NamespaceError::Internal {
                        message: format!("Manifest table '{}' has no location", object_id),
                    }
                    .into());
                }
                let metadata = ManifestNamespace::optional_string_value(metadatas, row);
                ManifestNamespace::deserialize_metadata(
                    metadata.as_deref(),
                    object_type.as_str(),
                    &object_id,
                )?;
                rows.push(ManifestRowValue {
                    object_id,
                    object_type,
                    location,
                    metadata,
                    base_objects,
                });
            }
        }
        if rows.len() > u32::MAX as usize {
            return Err(NamespaceError::Internal {
                message: format!(
                    "Manifest snapshot exceeds maximum single-fragment row count: {}",
                    rows.len()
                ),
            }
            .into());
        }
        rows.sort_unstable_by(|left, right| left.object_id.cmp(&right.object_id));
        if let Some(duplicate) = rows
            .windows(2)
            .find(|pair| pair[0].object_id == pair[1].object_id)
        {
            return Err(NamespaceError::Internal {
                message: format!(
                    "Manifest contains duplicate object_id '{}'",
                    duplicate[0].object_id
                ),
            }
            .into());
        }
        Ok(Self { rows: rows.into() })
    }

    fn get(&self, object_id: &str) -> Option<&ManifestRowValue> {
        self.rows
            .binary_search_by(|row| row.object_id.as_str().cmp(object_id))
            .ok()
            .map(|index| &self.rows[index])
    }

    fn prefix_range(&self, prefix: &str) -> (usize, usize) {
        let start = self
            .rows
            .partition_point(|row| row.object_id.as_str() < prefix);
        let end =
            start + self.rows[start..].partition_point(|row| row.object_id.starts_with(prefix));
        (start, end)
    }

    fn descendant_count(&self, object_id: &str) -> usize {
        let prefix = format!("{}{}", object_id, DELIMITER);
        let (start, end) = self.prefix_range(&prefix);
        end - start
    }

    fn direct_children(
        &self,
        parent: &[String],
        object_type: ObjectType,
    ) -> Vec<&ManifestRowValue> {
        let prefix = if parent.is_empty() {
            None
        } else {
            Some(format!("{}{}", parent.join(DELIMITER), DELIMITER))
        };
        let candidates = if let Some(prefix) = &prefix {
            let (start, end) = self.prefix_range(prefix);
            &self.rows[start..end]
        } else {
            &self.rows
        };
        candidates
            .iter()
            .filter(|row| {
                if row.object_type != object_type {
                    return false;
                }
                let relative_id = prefix
                    .as_deref()
                    .and_then(|prefix| row.object_id.strip_prefix(prefix))
                    .unwrap_or(&row.object_id);
                !relative_id.contains(DELIMITER)
            })
            .collect()
    }

    fn rows(&self) -> &[ManifestRowValue] {
        &self.rows
    }
}

struct ManifestOutputRow<'a> {
    object_id: &'a str,
    object_type: ObjectType,
    location: Option<&'a str>,
    metadata: Option<&'a str>,
    base_objects: Option<&'a [String]>,
}

struct ManifestBatchBuilder {
    object_ids: Vec<String>,
    object_types: Vec<&'static str>,
    locations: Vec<Option<String>>,
    metadatas: Vec<Option<String>>,
    base_objects: Vec<Option<Vec<String>>>,
}

impl ManifestBatchBuilder {
    fn new() -> Self {
        Self {
            object_ids: Vec::new(),
            object_types: Vec::new(),
            locations: Vec::new(),
            metadatas: Vec::new(),
            base_objects: Vec::new(),
        }
    }

    fn is_empty(&self) -> bool {
        self.object_ids.is_empty()
    }

    fn len(&self) -> usize {
        self.object_ids.len()
    }

    fn append(&mut self, row: ManifestOutputRow<'_>) -> Result<()> {
        self.object_ids.push(row.object_id.to_string());
        self.object_types.push(row.object_type.as_str());
        self.locations.push(row.location.map(ToString::to_string));
        self.metadatas.push(row.metadata.map(ToString::to_string));
        self.base_objects
            .push(row.base_objects.map(|objects| objects.to_vec()));
        Ok(())
    }

    fn finish(self) -> Result<RecordBatch> {
        let base_objects_array = ManifestNamespace::base_objects_array(&self.base_objects);
        RecordBatch::try_new(
            ManifestNamespace::manifest_schema(),
            vec![
                Arc::new(StringArray::from(self.object_ids)),
                Arc::new(StringArray::from(self.object_types)),
                Arc::new(StringArray::from(self.locations)),
                Arc::new(StringArray::from(self.metadatas)),
                Arc::new(base_objects_array),
            ],
        )
        .map_err(|e| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!("Failed to create manifest snapshot batch: {:?}", e),
            })
        })
    }
}

/// How to resolve a storage commit conflict (or an ambiguous commit error that did
/// not land) against the latest catalog state, without re-staging the full rewrite.
enum ConflictResolution<O> {
    /// Re-read the latest manifest and re-apply the mutation (upserts, version-range
    /// deletes). The staged data/index files are discarded and a new rewrite is attempted.
    Retry,
    /// Creating these object ids with fail-on-conflict semantics. If any of them now
    /// exists in the latest manifest, the create lost the race and must fail with a
    /// concurrent-modification error; otherwise retry the rewrite.
    FailIfExists(Vec<String>),
    /// Deleting `object_id`. If it is already absent from the latest manifest the delete
    /// has effectively happened, so return `output` as success; otherwise retry.
    SucceedIfAbsent { object_id: String, output: O },
}

trait ManifestStreamMutation: Send {
    type Output: Clone + Send + 'static;

    fn process_existing_row(
        &mut self,
        row: ManifestRowValue,
        output: &mut ManifestBatchBuilder,
    ) -> Result<()>;

    fn append_rows(&mut self, output: &mut ManifestBatchBuilder) -> Result<()>;

    fn finish(&self) -> CopyOnWriteMutation<Self::Output>;

    /// Declares how a storage commit conflict should be resolved against the latest
    /// committed catalog state. Defaults to re-reading and re-applying.
    fn conflict_resolution(&self) -> ConflictResolution<Self::Output> {
        ConflictResolution::Retry
    }
}

struct ManifestRewriteShared<M: ManifestStreamMutation> {
    mutation: M,
    result: Option<CopyOnWriteMutation<M::Output>>,
    error: Option<LanceError>,
    output_row_count: usize,
}

impl<M: ManifestStreamMutation> ManifestRewriteShared<M> {
    fn new(mutation: M) -> Self {
        Self {
            mutation,
            result: None,
            error: None,
            output_row_count: 0,
        }
    }
}

struct UpsertManifestMutation {
    entries: Vec<ManifestEntry>,
    base_objects: Vec<Option<Vec<String>>>,
    entry_positions: HashMap<String, usize>,
    matched: Vec<bool>,
    when_matched: WhenMatched,
}

impl UpsertManifestMutation {
    fn new(
        entries: Vec<ManifestEntry>,
        base_objects: Option<Vec<String>>,
        when_matched: WhenMatched,
    ) -> Self {
        let entry_positions = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.object_id.clone(), index))
            .collect();
        let matched = vec![false; entries.len()];
        let mut entry_base_objects = vec![None; entries.len()];
        if !entry_base_objects.is_empty() {
            entry_base_objects[0] = base_objects;
        }
        Self {
            entries,
            base_objects: entry_base_objects,
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
            base_objects: self.base_objects[index].as_deref(),
        }
    }
}

impl ManifestStreamMutation for UpsertManifestMutation {
    type Output = ();

    fn process_existing_row(
        &mut self,
        row: ManifestRowValue,
        output: &mut ManifestBatchBuilder,
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
                    output.append(self.entry_row(index))?;
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

        output.append(ManifestOutputRow {
            object_id: &row.object_id,
            object_type: row.object_type,
            location: row.location.as_deref(),
            metadata: row.metadata.as_deref(),
            base_objects: row.base_objects.as_deref(),
        })
    }

    fn append_rows(&mut self, output: &mut ManifestBatchBuilder) -> Result<()> {
        for index in 0..self.entries.len() {
            if !self.matched[index] {
                output.append(self.entry_row(index))?;
            }
        }
        Ok(())
    }

    fn finish(&self) -> CopyOnWriteMutation<Self::Output> {
        CopyOnWriteMutation::updated(())
    }

    fn conflict_resolution(&self) -> ConflictResolution<Self::Output> {
        match self.when_matched {
            // Fail-on-conflict create: a concurrent writer may have created one of these
            // ids. Re-applying would still fail, so check directly instead of re-staging.
            WhenMatched::Fail => ConflictResolution::FailIfExists(
                self.entries.iter().map(|e| e.object_id.clone()).collect(),
            ),
            // Metadata upsert is last-writer-wins: re-read and re-apply.
            _ => ConflictResolution::Retry,
        }
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
    ) -> Result<()> {
        if row.object_id == self.object_id {
            self.deleted = true;
            return Ok(());
        }

        output.append(ManifestOutputRow {
            object_id: &row.object_id,
            object_type: row.object_type,
            location: row.location.as_deref(),
            metadata: row.metadata.as_deref(),
            base_objects: row.base_objects.as_deref(),
        })
    }

    fn append_rows(&mut self, _output: &mut ManifestBatchBuilder) -> Result<()> {
        Ok(())
    }

    fn finish(&self) -> CopyOnWriteMutation<Self::Output> {
        if self.deleted {
            CopyOnWriteMutation::updated(())
        } else {
            CopyOnWriteMutation::unchanged(())
        }
    }

    fn conflict_resolution(&self) -> ConflictResolution<Self::Output> {
        // If a concurrent writer already removed the object, the delete is satisfied.
        ConflictResolution::SucceedIfAbsent {
            object_id: self.object_id.clone(),
            output: (),
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
/// The manifest dataset uses contiguous attached versions and this module never
/// runs old-version cleanup on it, allowing reads to check only the immediate
/// successor manifest before deciding whether a reload is needed.
#[derive(Debug)]
struct ManifestState {
    dataset: Dataset,
    snapshot: Arc<ManifestSnapshot>,
}

#[derive(Debug, Clone)]
pub struct DatasetConsistencyWrapper(Arc<RwLock<ManifestState>>);

impl DatasetConsistencyWrapper {
    /// Create a new wrapper with the given dataset.
    pub async fn new(dataset: Dataset) -> Result<Self> {
        debug_assert!(
            !dataset
                .manifest()
                .config
                .keys()
                .any(|key| key.starts_with("lance.auto_cleanup.")),
            "the directory manifest dataset must not enable old-version cleanup"
        );
        let snapshot = Arc::new(ManifestSnapshot::load(&dataset).await?);
        Ok(Self(Arc::new(RwLock::new(ManifestState {
            dataset,
            snapshot,
        }))))
    }

    /// Get an immutable reference to the dataset.
    /// Always reloads to ensure strong consistency.
    pub async fn get(&self) -> Result<DatasetReadGuard<'_>> {
        self.reload().await?;
        let guard = DatasetReadGuard {
            guard: self.0.read().await,
        };
        // Refuse manifests written with a reader feature flag this build does
        // not understand instead of misreading them.
        ensure_readable(guard.metadata())?;
        Ok(guard)
    }

    /// Reload the dataset and return a reference.
    pub async fn get_refreshed(&self) -> Result<DatasetReadGuard<'_>> {
        self.reload().await?;
        let guard = DatasetReadGuard {
            guard: self.0.read().await,
        };
        ensure_readable(guard.metadata())?;
        Ok(guard)
    }

    /// Get a mutable reference to the dataset.
    /// Always reloads to ensure strong consistency.
    pub async fn get_mut(&self) -> Result<DatasetWriteGuard<'_>> {
        self.reload().await?;
        let guard = DatasetWriteGuard {
            guard: self.0.write().await,
        };
        ensure_readable(guard.metadata())?;
        ensure_writable(guard.metadata())?;
        Ok(guard)
    }

    /// Provide a known latest version of the dataset.
    pub async fn set_latest(&self, dataset: Dataset) {
        {
            let read_guard = self.0.read().await;
            if dataset.manifest().version <= read_guard.dataset.manifest().version {
                return;
            }
        }
        let snapshot = match ManifestSnapshot::load(&dataset).await {
            Ok(snapshot) => Arc::new(snapshot),
            Err(err) => {
                log::warn!("Failed to materialize known latest manifest: {}", err);
                return;
            }
        };
        let mut write_guard = self.0.write().await;
        if dataset.manifest().version > write_guard.dataset.manifest().version {
            write_guard.dataset = dataset;
            write_guard.snapshot = snapshot;
        }
    }

    /// Reload the dataset to the latest version.
    async fn reload(&self) -> Result<()> {
        // First check if we need to reload (with read lock)
        let read_guard = self.0.read().await;
        let dataset_uri = read_guard.dataset.uri().to_string();
        let current_version = read_guard.dataset.version().version;
        log::debug!(
            "Reload starting for uri={}, current_version={}",
            dataset_uri,
            current_version
        );
        // The directory manifest table uses contiguous attached versions and
        // does not run old-version cleanup, so the immediate successor probe is
        // enough to detect changes without resolving or loading the latest
        // manifest on every namespace read.
        let has_successor_version =
            read_guard
                .dataset
                .has_successor_version()
                .await
                .map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!("Failed to check dataset staleness: {:?}", e),
                    })
                })?;
        log::debug!(
            "Reload checked successor_version_exists={} for uri={}, current_version={}",
            has_successor_version,
            dataset_uri,
            current_version
        );
        drop(read_guard);

        // If already up-to-date, return early
        if !has_successor_version {
            log::debug!("Already up-to-date for uri={}", dataset_uri);
            return Ok(());
        }

        // Need to reload, acquire write lock
        let mut write_guard = self.0.write().await;

        // Double-check after acquiring write lock (someone else might have reloaded)
        let has_successor_version =
            write_guard
                .dataset
                .has_successor_version()
                .await
                .map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!("Failed to check dataset staleness: {:?}", e),
                    })
                })?;

        if has_successor_version {
            let mut latest_dataset = write_guard.dataset.clone();
            latest_dataset.checkout_latest().await.map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to checkout latest: {:?}", e),
                })
            })?;
            let latest_snapshot = ManifestSnapshot::load(&latest_dataset).await?;
            write_guard.dataset = latest_dataset;
            write_guard.snapshot = Arc::new(latest_snapshot);
        }

        Ok(())
    }
}

pub struct DatasetReadGuard<'a> {
    guard: RwLockReadGuard<'a, ManifestState>,
}

impl DatasetReadGuard<'_> {
    fn snapshot(&self) -> &Arc<ManifestSnapshot> {
        &self.guard.snapshot
    }
}

impl Deref for DatasetReadGuard<'_> {
    type Target = Dataset;

    fn deref(&self) -> &Self::Target {
        &self.guard.dataset
    }
}

pub struct DatasetWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, ManifestState>,
}

impl Deref for DatasetWriteGuard<'_> {
    type Target = Dataset;

    fn deref(&self) -> &Self::Target {
        &self.guard.dataset
    }
}

impl DerefMut for DatasetWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard.dataset
    }
}

/// Manifest-based namespace implementation
///
/// Uses a special `__manifest` Lance table to track tables and nested namespaces.
pub struct ManifestNamespace {
    root: String,
    storage_options: Option<HashMap<String, String>>,
    session: Option<Arc<Session>>,
    object_store: Arc<ObjectStore>,
    base_path: Path,
    manifest_dataset: DatasetConsistencyWrapper,
    /// Whether directory listing is enabled in dual mode
    /// If true, root namespace tables use {table_name}.lance naming
    /// If false, they use namespace-prefixed names
    dir_listing_enabled: bool,
    /// Legacy compatibility setting. In-memory reads do not build replacement indices.
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
        // CommitConflict: version collision retries exhausted -> Throttling (safe to retry)
        LanceError::CommitConflict { .. } => NamespaceError::Throttling {
            message: format!("Too many concurrent writes, please retry later: {:?}", e),
        }
        .into(),
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
    ) -> Result<Self> {
        let manifest_dataset =
            Self::ensure_manifest_table_up_to_date(&root, &storage_options, session.clone())
                .await?;

        Ok(Self::new(
            root,
            storage_options,
            session,
            object_store,
            base_path,
            manifest_dataset,
            dir_listing_enabled,
            inline_optimization_enabled,
            commit_retries,
        ))
    }

    /// Open an existing manifest dataset without creating or migrating it.
    #[allow(clippy::too_many_arguments)]
    pub async fn open_from_directory(
        root: String,
        storage_options: Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        dir_listing_enabled: bool,
        inline_optimization_enabled: bool,
        commit_retries: Option<u32>,
    ) -> Result<Self> {
        let manifest_dataset =
            Self::open_manifest_table(&root, &storage_options, session.clone()).await?;

        Ok(Self::new(
            root,
            storage_options,
            session,
            object_store,
            base_path,
            manifest_dataset,
            dir_listing_enabled,
            inline_optimization_enabled,
            commit_retries,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        root: String,
        storage_options: Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
        object_store: Arc<ObjectStore>,
        base_path: Path,
        manifest_dataset: DatasetConsistencyWrapper,
        dir_listing_enabled: bool,
        inline_optimization_enabled: bool,
        commit_retries: Option<u32>,
    ) -> Self {
        Self {
            root,
            storage_options,
            session,
            object_store,
            base_path,
            manifest_dataset,
            dir_listing_enabled,
            inline_optimization_enabled,
            commit_retries,
            manifest_mutation_lock: Arc::new(Mutex::new(())),
        }
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

    fn string_list_array(values: &[Option<Vec<String>>], child_name: &str) -> ListArray {
        let string_builder = StringBuilder::new();
        let mut list_builder = ListBuilder::new(string_builder).with_field(Arc::new(Field::new(
            child_name,
            DataType::Utf8,
            true,
        )));
        for value in values {
            match value {
                Some(objects) => {
                    for object in objects {
                        list_builder.values().append_value(object);
                    }
                    list_builder.append(true);
                }
                None => list_builder.append_null(),
            }
        }
        list_builder.finish()
    }

    fn base_objects_array(values: &[Option<Vec<String>>]) -> ListArray {
        Self::string_list_array(values, "object_id")
    }

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
            Field::new("metadata", DataType::Utf8, true),
            Field::new(
                "base_objects",
                DataType::List(Arc::new(Field::new("object_id", DataType::Utf8, true))),
                true,
            ),
        ]))
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

    fn base_objects_column_values(batch: &RecordBatch) -> Result<Vec<Option<Vec<String>>>> {
        let Some(column) = batch.column_by_name("base_objects") else {
            return Ok(vec![None; batch.num_rows()]);
        };
        let array = column.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::Internal {
                message: format!(
                    "Column 'base_objects' is not a list array: {:?}",
                    column.data_type()
                ),
            })
        })?;

        let mut values = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            if array.is_null(row) {
                values.push(None);
                continue;
            }
            let row_values = array.value(row);
            let row_values = row_values
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: "Column 'base_objects' values are not strings".to_string(),
                    })
                })?;
            let mut objects = Vec::with_capacity(row_values.len());
            for value_index in 0..row_values.len() {
                if row_values.is_null(value_index) {
                    return Err(NamespaceError::Internal {
                        message: format!(
                            "Manifest column 'base_objects' has null item at row {} item {}",
                            row, value_index
                        ),
                    }
                    .into());
                }
                objects.push(row_values.value(value_index).to_string());
            }
            values.push(Some(objects));
        }
        Ok(values)
    }

    fn manifest_snapshot_stream(snapshot: Arc<ManifestSnapshot>) -> SendableRecordBatchStream {
        let schema = Self::manifest_schema();
        let stream_schema = schema.clone();
        let stream = stream::unfold((snapshot, 0), |(snapshot, offset)| async move {
            if offset >= snapshot.rows().len() {
                return None;
            }
            let end = (offset + MANIFEST_REWRITE_BATCH_SIZE).min(snapshot.rows().len());
            let mut output = ManifestBatchBuilder::new();
            let batch = snapshot.rows()[offset..end]
                .iter()
                .try_for_each(|row| {
                    output.append(ManifestOutputRow {
                        object_id: &row.object_id,
                        object_type: row.object_type,
                        location: row.location.as_deref(),
                        metadata: row.metadata.as_deref(),
                        base_objects: row.base_objects.as_deref(),
                    })
                })
                .and_then(|()| output.finish())
                .map_err(|err| DataFusionError::External(Box::new(err)));
            Some((batch, (snapshot, end)))
        });
        Box::pin(DatafusionRecordBatchStreamAdapter::new(
            stream_schema,
            stream.fuse(),
        ))
    }

    fn manifest_rewrite_commit_retries(&self) -> u32 {
        self.commit_retries
            .unwrap_or(DEFAULT_MANIFEST_REWRITE_COMMIT_RETRIES)
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
        let metadatas = Self::get_string_column(&batch, "metadata")?;
        let base_objects = Self::base_objects_column_values(&batch)?;
        let mut output = ManifestBatchBuilder::new();
        let mut guard = Self::lock_manifest_rewrite_shared(shared)?;
        for (row, base_objects) in base_objects.into_iter().enumerate().take(batch.num_rows()) {
            let row_value = ManifestRowValue {
                object_id: Self::required_string_value(object_ids, row, "object_id")?.to_string(),
                object_type: ObjectType::parse(Self::required_string_value(
                    object_types,
                    row,
                    "object_type",
                )?)?,
                location: Self::optional_string_value(locations, row),
                metadata: Self::optional_string_value(metadatas, row),
                base_objects,
            };
            guard
                .mutation
                .process_existing_row(row_value, &mut output)?;
        }
        guard.output_row_count = guard
            .output_row_count
            .checked_add(output.len())
            .ok_or_else(|| NamespaceError::Internal {
                message: "Manifest rewrite row count overflow".to_string(),
            })?;
        if guard.output_row_count > u32::MAX as usize {
            return Err(NamespaceError::Internal {
                message: format!(
                    "Manifest rewrite exceeded maximum single-fragment row count: {}",
                    guard.output_row_count
                ),
            }
            .into());
        }
        drop(guard);
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
        guard.mutation.append_rows(&mut output)?;
        let result = guard.mutation.finish();
        let output_row_count = guard
            .output_row_count
            .checked_add(output.len())
            .ok_or_else(|| NamespaceError::Internal {
                message: "Manifest rewrite row count overflow".to_string(),
            })?;
        if output_row_count > u32::MAX as usize {
            return Err(NamespaceError::Internal {
                message: format!(
                    "Manifest rewrite exceeded maximum single-fragment row count: {}",
                    output_row_count
                ),
            }
            .into());
        }
        let force_empty_batch = output_row_count == 0;
        guard.output_row_count = output_row_count;
        guard.result = Some(result);
        drop(guard);
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
    ) -> Result<CopyOnWriteMutation<M::Output>> {
        let mut guard = Self::lock_manifest_rewrite_shared(shared)?;
        guard.result.take().ok_or_else(|| {
            lance_core::Error::from(NamespaceError::Internal {
                message: "Manifest rewrite stream did not finish".to_string(),
            })
        })
    }

    fn manifest_from_overwrite_transaction(
        previous: &Manifest,
        schema: lance_core::datatypes::Schema,
        fragments: &[Fragment],
    ) -> Manifest {
        let mut next_fragment_id = 0;
        let mut fragments = fragments
            .iter()
            .cloned()
            .map(|mut fragment| {
                if fragment.id == 0 {
                    fragment.id = next_fragment_id;
                    next_fragment_id += 1;
                } else {
                    next_fragment_id = next_fragment_id.max(fragment.id + 1);
                }
                fragment
            })
            .collect::<Vec<_>>();
        fragments.sort_by_key(|fragment| fragment.id);
        Manifest::new_from_previous(previous, schema, Arc::new(fragments))
    }

    /// Delete the staged (uncommitted) data files for a rewrite.
    /// Only call this once the rewrite is known *not* to have landed (a put-if-not-exists
    /// conflict, or an ambiguous error whose target version does not reference our data
    /// file) — otherwise it would orphan files a committed manifest still references.
    async fn cleanup_staged_manifest_files(
        &self,
        object_store: &ObjectStore,
        data_files: &HashSet<String>,
    ) {
        let data_dir = self
            .base_path
            .clone()
            .join(MANIFEST_TABLE_NAME)
            .join(LANCE_DATA_DIR);
        for path in data_files {
            let data_path = data_dir.clone().join(path.as_str());
            if let Err(err) = object_store.delete(&data_path).await {
                log::warn!(
                    "Failed to clean up uncommitted manifest rewrite data file '{}': {}",
                    data_path,
                    err
                );
            }
        }
    }

    /// Resolve the commit handler for the `__manifest` dataset's storage backend.
    async fn manifest_commit_handler(&self) -> Result<Arc<dyn CommitHandler>> {
        commit_handler_from_url(&self.root, &None)
            .await
            .map_err(|e| {
                lance_core::Error::from(NamespaceError::Internal {
                    message: format!("Failed to resolve manifest commit handler: {:?}", e),
                })
            })
    }

    /// Directly write the rewritten `__manifest` as a new version using the storage
    /// backend's atomic put-if-not-exists. The overwrite transaction is embedded inline
    /// (no separate transaction file) and the commit handler writes the version hint.
    async fn commit_manifest_overwrite(
        &self,
        dataset: &Dataset,
        commit_handler: &dyn CommitHandler,
        manifest: &mut Manifest,
        transaction: Transaction,
    ) -> std::result::Result<(), CommitError> {
        ensure_can_write_manifest(manifest).map_err(CommitError::from)?;
        apply_feature_flags(manifest, false, false).map_err(CommitError::from)?;
        let timestamp_nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        manifest.set_timestamp(timestamp_nanos);
        manifest.update_max_fragment_id();

        // Commit through the dataset's own object store, not `self.object_store`: for
        // stores like `memory://` the namespace and the dataset can hold different
        // instances, and a commit written to the wrong one is invisible to reads.
        let object_store = dataset
            .object_store(None)
            .await
            .map_err(CommitError::from)?;
        let base_path = self.base_path.clone().join(MANIFEST_TABLE_NAME);
        let naming_scheme = dataset.manifest_location().naming_scheme;
        commit_handler
            .commit(
                manifest,
                None,
                &base_path,
                &object_store,
                write_manifest_file_to_path,
                naming_scheme,
                Some((&transaction).into()),
            )
            .await
            .map(|_location| ())
    }

    /// After an ambiguous commit error, determine whether our overwrite actually landed at
    /// `target_version`. A network failure can leave the manifest committed even though the
    /// client observed an error; in that case the committed version references one of our
    /// staged data files, and deleting them would corrupt the catalog.
    async fn manifest_commit_landed(
        &self,
        dataset: &Dataset,
        target_version: u64,
        data_files: &HashSet<String>,
    ) -> bool {
        let Ok(committed) = dataset.checkout_version(target_version).await else {
            return false;
        };
        committed.manifest().fragments.iter().any(|fragment| {
            fragment
                .files
                .iter()
                .any(|file| data_files.contains(file.path.as_str()))
        })
    }

    /// Resolve a storage commit conflict against the latest committed catalog state.
    /// Returns `Some(output)` when the mutation's intent is already satisfied (no retry
    /// needed), `Ok(None)` to retry the rewrite, or an error for a terminal conflict.
    async fn resolve_manifest_conflict<O: Clone>(
        &self,
        resolution: &ConflictResolution<O>,
    ) -> Result<Option<O>> {
        match resolution {
            ConflictResolution::Retry => Ok(None),
            ConflictResolution::FailIfExists(object_ids) => {
                for object_id in object_ids {
                    if self.manifest_contains_object(object_id).await? {
                        return Err(NamespaceError::ConcurrentModification {
                            message: format!(
                                "Object '{}' was concurrently created by another operation",
                                object_id
                            ),
                        }
                        .into());
                    }
                }
                Ok(None)
            }
            ConflictResolution::SucceedIfAbsent { object_id, output } => {
                if self.manifest_contains_object(object_id).await? {
                    Ok(None)
                } else {
                    Ok(Some(output.clone()))
                }
            }
        }
    }

    /// Validate that this build can write the current `__manifest` before a
    /// mutating operation performs any side effect (e.g. writing table data), so
    /// a refused write leaves nothing orphaned behind. The eventual
    /// `rewrite_manifest` commit re-checks `ensure_writable` on each retry, so a
    /// concurrent upgrade in between is still caught.
    async fn ensure_manifest_writable(&self) -> Result<()> {
        let dataset_guard = self.manifest_dataset.get().await?;
        ensure_can_write_manifest(dataset_guard.manifest())?;
        ensure_writable(dataset_guard.metadata())
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
        let commit_handler = self.manifest_commit_handler().await?;

        loop {
            let dataset_guard = self.manifest_dataset.get_refreshed().await?;
            ensure_can_write_manifest(dataset_guard.manifest())?;
            let dataset = Arc::new(dataset_guard.clone());
            let snapshot = Arc::clone(dataset_guard.snapshot());
            drop(dataset_guard);
            // The namespace format has its own capabilities in table metadata,
            // separate from the Lance manifest capabilities checked above.
            ensure_writable(dataset.metadata())?;
            // Staged files, the commit, and cleanup must all use the dataset's
            // own object store (see `commit_manifest_overwrite`).
            let object_store = dataset.object_store(None).await?;

            let source = Self::manifest_snapshot_stream(snapshot);
            let mutation = make_mutation();
            let resolution = mutation.conflict_resolution();
            let shared = Arc::new(StdMutex::new(ManifestRewriteShared::new(mutation)));
            let output_stream = Self::manifest_rewrite_output_stream(source, shared.clone());
            // Pin both limits so the catalog remains a single-fragment snapshot.
            let write_params = WriteParams {
                mode: WriteMode::Overwrite,
                session: self.session.clone(),
                max_rows_per_file: u32::MAX as usize,
                max_bytes_per_file: usize::MAX,
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

            let mutation = Self::take_manifest_rewrite_result(&shared)?;

            let Operation::Overwrite {
                fragments, schema, ..
            } = &transaction.operation
            else {
                return Err(NamespaceError::Internal {
                    message: "Manifest rewrite transaction is not an overwrite".to_string(),
                }
                .into());
            };
            // Unique data files this attempt staged. Used to clean up orphans and to
            // attribute an ambiguous commit error back to us.
            let staged_data_files = fragments
                .iter()
                .flat_map(|fragment| fragment.files.iter())
                .filter(|file| file.base_id.is_none())
                .map(|file| file.path.clone())
                .collect::<HashSet<_>>();

            if !mutation.has_changes {
                self.cleanup_staged_manifest_files(&object_store, &staged_data_files)
                    .await;
                return Ok(mutation.result);
            }

            let mut manifest = Self::manifest_from_overwrite_transaction(
                dataset.manifest(),
                schema.clone(),
                fragments,
            );
            let target_version = manifest.version;

            let commit_result = self
                .commit_manifest_overwrite(
                    &dataset,
                    commit_handler.as_ref(),
                    &mut manifest,
                    transaction,
                )
                .await;

            match commit_result {
                Ok(()) => {
                    self.manifest_dataset.get_refreshed().await?;
                    return Ok(mutation.result);
                }
                Err(err) => {
                    // The put may have landed even though the client saw an error (lost
                    // ack). Verify before deleting anything so we never orphan files that a
                    // committed manifest still references.
                    if self
                        .manifest_commit_landed(&dataset, target_version, &staged_data_files)
                        .await
                    {
                        self.manifest_dataset.get_refreshed().await?;
                        return Ok(mutation.result);
                    }
                    self.cleanup_staged_manifest_files(&object_store, &staged_data_files)
                        .await;
                    match err {
                        CommitError::CommitConflict => {
                            if let Some(output) =
                                self.resolve_manifest_conflict(&resolution).await?
                            {
                                return Ok(output);
                            }
                            if retries >= max_retries {
                                return Err(NamespaceError::ConcurrentModification {
                                    message: format!(
                                        "{}: still conflicting after {} retries",
                                        operation, max_retries
                                    ),
                                }
                                .into());
                            }
                            retries += 1;
                            tokio::time::sleep(std::time::Duration::from_millis(
                                10 * u64::from(retries),
                            ))
                            .await;
                        }
                        CommitError::OtherError(err) => {
                            return Err(convert_lance_commit_error(&err, operation, None));
                        }
                    }
                }
            }
        }
    }

    /// Check if the manifest contains an object with the given ID
    async fn manifest_contains_object(&self, object_id: &str) -> Result<bool> {
        let state = self.manifest_dataset.get().await?;
        Ok(state.snapshot().get(object_id).is_some())
    }

    fn deserialize_metadata(
        metadata: Option<&str>,
        object_type: &str,
        object_id: &str,
    ) -> Result<Option<HashMap<String, String>>> {
        metadata
            .map(|metadata| {
                serde_json::from_str(metadata).map_err(|e| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!(
                            "Failed to deserialize metadata for {} '{}': {}",
                            object_type, object_id, e
                        ),
                    })
                })
            })
            .transpose()
    }

    /// Query the manifest for a table with the given object ID
    async fn query_manifest_for_table(&self, object_id: &str) -> Result<Option<TableInfo>> {
        let state = self.manifest_dataset.get().await?;
        let Some(row) = state.snapshot().get(object_id) else {
            return Ok(None);
        };
        if row.object_type != ObjectType::Table {
            return Ok(None);
        }
        let location = row
            .location
            .clone()
            .ok_or_else(|| NamespaceError::Internal {
                message: format!("Manifest table '{}' has no location", object_id),
            })?;
        let metadata = Self::deserialize_metadata(row.metadata.as_deref(), "table", object_id)?;
        let (namespace, name) = Self::parse_object_id(object_id);
        Ok(Some(TableInfo {
            namespace,
            name,
            location,
            metadata,
        }))
    }

    fn serialize_metadata(
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

    async fn location_has_actual_manifests(&self, location: &str) -> Result<bool> {
        Self::path_has_actual_manifests(&self.object_store, &self.base_path.clone().join(location))
            .await
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
        let state = self.manifest_dataset.get().await?;
        state
            .snapshot()
            .rows()
            .iter()
            .filter(|row| {
                row.object_type == ObjectType::Table && !row.object_id.contains(DELIMITER)
            })
            .map(|row| {
                row.location.clone().ok_or_else(|| {
                    lance_core::Error::from(NamespaceError::Internal {
                        message: format!("Manifest table '{}' has no location", row.object_id),
                    })
                })
            })
            .collect()
    }

    /// Insert an entry into the manifest table
    async fn insert_into_manifest(
        &self,
        object_id: String,
        object_type: ObjectType,
        location: Option<String>,
    ) -> Result<()> {
        self.insert_into_manifest_with_metadata(
            vec![ManifestEntry {
                object_id,
                object_type,
                location,
                metadata: None,
            }],
            None,
        )
        .await
    }

    /// Insert one or more entries into the manifest table with metadata and base_objects.
    ///
    /// This is the unified entry point for both single and batch inserts.
    /// If any entry already exists (matching object_id), the entire batch fails.
    pub async fn insert_into_manifest_with_metadata(
        &self,
        entries: Vec<ManifestEntry>,
        base_objects: Option<Vec<String>>,
    ) -> Result<()> {
        self.merge_into_manifest_with_metadata(entries, base_objects, WhenMatched::Fail)
            .await
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

    async fn upsert_into_manifest_with_metadata(
        &self,
        entries: Vec<ManifestEntry>,
        base_objects: Option<Vec<String>>,
    ) -> Result<()> {
        self.merge_into_manifest_with_metadata(entries, base_objects, WhenMatched::UpdateAll)
            .await
    }

    async fn merge_into_manifest_with_metadata(
        &self,
        entries: Vec<ManifestEntry>,
        base_objects: Option<Vec<String>>,
        when_matched: WhenMatched,
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let mut object_ids = HashSet::with_capacity(entries.len());
        if let Some(duplicate) = entries
            .iter()
            .find(|entry| !object_ids.insert(entry.object_id.as_str()))
        {
            return Err(NamespaceError::InvalidInput {
                message: format!(
                    "Manifest mutation contains duplicate object_id '{}'",
                    duplicate.object_id
                ),
            }
            .into());
        }

        self.rewrite_manifest("Failed to overwrite manifest", || {
            UpsertManifestMutation::new(entries.clone(), base_objects.clone(), when_matched.clone())
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
        let state = self.manifest_dataset.get().await?;
        let Some(row) = state.snapshot().get(object_id) else {
            return Ok(None);
        };
        if row.object_type != ObjectType::Namespace {
            return Ok(None);
        }
        let metadata = Self::deserialize_metadata(row.metadata.as_deref(), "namespace", object_id)?;
        let (namespace, name) = Self::parse_object_id(object_id);
        Ok(Some(NamespaceInfo {
            namespace,
            name,
            metadata,
        }))
    }

    /// Load an existing manifest dataset without creating or migrating it.
    async fn open_manifest_table(
        root: &str,
        storage_options: &Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
    ) -> Result<DatasetConsistencyWrapper> {
        let manifest_path = format!("{}/{}", root, MANIFEST_TABLE_NAME);
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
            session,
            store_options: Some(store_options),
            ..Default::default()
        };
        let dataset = DatasetBuilder::from_uri(&manifest_path)
            .with_read_params(read_params)
            .load()
            .await?;
        ensure_readable(dataset.metadata())?;
        DatasetConsistencyWrapper::new(dataset).await
    }

    /// Create or load the manifest dataset, ensuring it has the latest schema setup.
    ///
    /// This function will:
    /// 1. Try to load an existing manifest table
    /// 2. If it exists, check and migrate the schema if needed (e.g., add primary key metadata)
    /// 3. If it doesn't exist, create a new manifest table with the current schema
    async fn ensure_manifest_table_up_to_date(
        root: &str,
        storage_options: &Option<HashMap<String, String>>,
        session: Option<Arc<Session>>,
    ) -> Result<DatasetConsistencyWrapper> {
        let manifest_path = format!("{}/{}", root, MANIFEST_TABLE_NAME);
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
        match dataset_result {
            Ok(mut dataset) => {
                // Reject a manifest written with a reader feature flag this build
                // does not understand before touching it.
                ensure_readable(dataset.metadata())?;

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
                    // This legacy migration writes to the manifest, so confirm this
                    // build is allowed to write the current format first.
                    ensure_writable(dataset.metadata())?;
                    log::info!(
                        "Migrating __manifest table to add primary key metadata on object_id"
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

                DatasetConsistencyWrapper::new(dataset).await
            }
            Err(err) if Self::is_not_found_load_error(&err) => {
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
                        log::info!(
                            "Successfully created manifest table at {}, version={}, uri={}",
                            manifest_path,
                            dataset.version().version,
                            dataset.uri()
                        );
                        DatasetConsistencyWrapper::new(dataset).await
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
                        DatasetConsistencyWrapper::new(dataset).await
                    }
                    Err(e) => Err(lance_core::Error::from(NamespaceError::Internal {
                        message: format!("Failed to create manifest dataset: {:?}", e),
                    })),
                }
            }
            Err(err) => Err(err),
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

        let table_entries = {
            let state = self.manifest_dataset.get().await?;
            state
                .snapshot()
                .direct_children(namespace_id, ObjectType::Table)
                .into_iter()
                .map(|row| {
                    let (_, name) = Self::parse_object_id(&row.object_id);
                    let location = row.location.clone().ok_or_else(|| {
                        lance_core::Error::from(NamespaceError::Internal {
                            message: format!("Manifest table '{}' has no location", row.object_id),
                        })
                    })?;
                    Ok((name, location))
                })
                .collect::<Result<Vec<_>>>()?
        };

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

        // Refuse before writing any table data if this build cannot write the
        // manifest, so a refused create leaves no orphaned dataset behind.
        self.ensure_manifest_writable().await?;

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
            self.upsert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id,
                    object_type: ObjectType::Table,
                    location: Some(dir_name),
                    metadata,
                }],
                None,
            )
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
                    self.insert_into_manifest_with_metadata(
                        vec![ManifestEntry {
                            object_id,
                            object_type: ObjectType::Table,
                            location: Some(dir_name.clone()),
                            metadata,
                        }],
                        None,
                    )
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
                let table_path = self.base_path.clone().join(info.location.as_str());
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

        let state = self.manifest_dataset.get().await?;
        let mut namespaces = state
            .snapshot()
            .direct_children(parent_namespace, ObjectType::Namespace)
            .into_iter()
            .map(|row| Self::parse_object_id(&row.object_id).1)
            .collect();

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

        self.insert_into_manifest_with_metadata(
            vec![ManifestEntry {
                object_id,
                object_type: ObjectType::Namespace,
                location: None,
                metadata,
            }],
            None,
        )
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

        let count = self
            .manifest_dataset
            .get()
            .boxed()
            .await?
            .snapshot()
            .descendant_count(&object_id);

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

        self.ensure_manifest_writable().await?;

        // Atomically create the .lance-reserved file to mark the table as declared.
        // Shared with DirectoryNamespace via put_marker_file_atomic (dotfile-safe
        // staging + MarkerFileError::AlreadyExists → TableAlreadyExists).
        let reserved_file_path = table_path.clone().join(".lance-reserved");
        super::put_marker_file_atomic(
            &self.object_store,
            &reserved_file_path,
            &format!("table {}", table_name),
        )
        .await
        .map_err(|e| match e {
            super::MarkerFileError::AlreadyExists { .. } => {
                lance_core::Error::from(NamespaceError::TableAlreadyExists {
                    message: table_name.to_string(),
                })
            }
            super::MarkerFileError::Other { message } => {
                lance_core::Error::from(NamespaceError::Internal { message })
            }
        })?;

        let metadata = Self::serialize_metadata(request.properties.as_ref(), "table", &object_id)?;

        // Add entry to manifest marking this as a declared table (store dir_name, not full path)
        self.insert_into_manifest_with_metadata(
            vec![ManifestEntry {
                object_id,
                object_type: ObjectType::Table,
                location: Some(dir_name),
                metadata,
            }],
            None,
        )
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

    /// Add columns to a table.
    ///
    /// Converts the API `AddColumnsEntry` (SQL expressions) into Lance's
    /// `NewColumnTransform::SqlExpressions` and delegates to `Dataset::add_columns`.
    async fn alter_table_add_columns(
        &self,
        request: AlterTableAddColumnsRequest,
    ) -> Result<AlterTableAddColumnsResponse> {
        let table_id = request
            .id
            .as_ref()
            .ok_or_else(|| Error::invalid_input_source("Table ID is required".into()))?;

        if table_id.is_empty() {
            return Err(Error::invalid_input_source(
                "Table ID cannot be empty".into(),
            ));
        }

        let object_id = Self::str_object_id(table_id);
        let table_info = self.query_manifest_for_table(&object_id).boxed().await?;

        match table_info {
            Some(info) => {
                let table_uri = Self::construct_full_uri(&self.root, &info.location)?;
                // Use DatasetBuilder with storage options to align with describe_table
                // and to support custom storage backends (e.g. S3 with custom endpoints).
                let mut builder = DatasetBuilder::from_uri(&table_uri);
                if let Some(opts) = &self.storage_options {
                    builder = builder.with_storage_options(opts.clone());
                }
                if let Some(session) = &self.session {
                    builder = builder.with_session(session.clone());
                }
                let mut dataset = builder.load().await.map_err(|e| {
                    Error::io_source(box_error(std::io::Error::other(format!(
                        "Failed to open dataset: {}",
                        e
                    ))))
                })?;

                // Use shared helper to build SQL expressions, ensuring a clear error when expression is missing
                let sql_expressions = super::build_sql_expressions(&request.new_columns)?;

                dataset
                    .add_columns(
                        lance::dataset::NewColumnTransform::SqlExpressions(sql_expressions),
                        None,
                        None,
                    )
                    .await
                    .map_err(|e| {
                        // Surface specific commit/conflict errors (CommitConflict,
                        // RetryableCommitConflict, IncompatibleTransaction, ...) rather than
                        // collapsing every failure into a generic IO error.
                        convert_lance_commit_error(&e, "add_columns", Some(&object_id))
                    })?;

                let version = dataset.version().version as i64;
                Ok(AlterTableAddColumnsResponse::new(version))
            }
            None => Err(NamespaceError::TableNotFound { message: object_id }.into()),
        }
    }

    /// Alter columns in a table (rename, change type, change nullability).
    ///
    /// Converts the API `AlterColumnsEntry` into Lance's `ColumnAlteration`
    /// and delegates to `Dataset::alter_columns`.
    async fn alter_table_alter_columns(
        &self,
        request: AlterTableAlterColumnsRequest,
    ) -> Result<AlterTableAlterColumnsResponse> {
        let table_id = request
            .id
            .as_ref()
            .ok_or_else(|| Error::invalid_input_source("Table ID is required".into()))?;

        if table_id.is_empty() {
            return Err(Error::invalid_input_source(
                "Table ID cannot be empty".into(),
            ));
        }

        let object_id = Self::str_object_id(table_id);
        let table_info = self.query_manifest_for_table(&object_id).boxed().await?;

        match table_info {
            Some(info) => {
                let table_uri = Self::construct_full_uri(&self.root, &info.location)?;
                let mut builder = DatasetBuilder::from_uri(&table_uri);
                if let Some(opts) = &self.storage_options {
                    builder = builder.with_storage_options(opts.clone());
                }
                if let Some(session) = &self.session {
                    builder = builder.with_session(session.clone());
                }
                let mut dataset = builder.load().await.map_err(|e| {
                    Error::io_source(box_error(std::io::Error::other(format!(
                        "Failed to open dataset: {}",
                        e
                    ))))
                })?;

                // Use shared helper to build column alterations, ensuring a clear error when data_type conversion fails
                let alterations = super::build_column_alterations(&request.alterations)?;

                dataset.alter_columns(&alterations).await.map_err(|e| {
                    convert_lance_commit_error(&e, "alter_columns", Some(&object_id))
                })?;

                let version = dataset.version().version as i64;
                Ok(AlterTableAlterColumnsResponse::new(version))
            }
            None => Err(NamespaceError::TableNotFound { message: object_id }.into()),
        }
    }

    /// Drop columns from a table.
    ///
    /// Delegates to `Dataset::drop_columns` with the column names from the request.
    async fn alter_table_drop_columns(
        &self,
        request: AlterTableDropColumnsRequest,
    ) -> Result<AlterTableDropColumnsResponse> {
        let table_id = request
            .id
            .as_ref()
            .ok_or_else(|| Error::invalid_input_source("Table ID is required".into()))?;

        if table_id.is_empty() {
            return Err(Error::invalid_input_source(
                "Table ID cannot be empty".into(),
            ));
        }

        let object_id = Self::str_object_id(table_id);
        let table_info = self.query_manifest_for_table(&object_id).boxed().await?;

        match table_info {
            Some(info) => {
                let table_uri = Self::construct_full_uri(&self.root, &info.location)?;
                let mut builder = DatasetBuilder::from_uri(&table_uri);
                if let Some(opts) = &self.storage_options {
                    builder = builder.with_storage_options(opts.clone());
                }
                if let Some(session) = &self.session {
                    builder = builder.with_session(session.clone());
                }
                let mut dataset = builder.load().await.map_err(|e| {
                    Error::io_source(box_error(std::io::Error::other(format!(
                        "Failed to open dataset: {}",
                        e
                    ))))
                })?;

                let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
                dataset.drop_columns(&columns).await.map_err(|e| {
                    convert_lance_commit_error(&e, "drop_columns", Some(&object_id))
                })?;

                let version = dataset.version().version as i64;
                Ok(AlterTableDropColumnsResponse::new(version))
            }
            None => Err(NamespaceError::TableNotFound { message: object_id }.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ConflictResolution, CopyOnWriteMutation, DeleteObjectMutation, LANCE_DATA_DIR,
        MANIFEST_TABLE_NAME, ManifestBatchBuilder, ManifestEntry, ManifestNamespace,
        ManifestOutputRow, ManifestRowValue, ManifestStreamMutation, ObjectType,
    };
    use crate::DirectoryNamespaceBuilder;
    use arrow::datatypes::DataType;
    use bytes::Bytes;
    use futures::StreamExt;
    use lance::index::DatasetIndexExt;
    use lance_core::utils::tempfile::TempStdDir;
    use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
    use lance_namespace::LanceNamespace;
    use lance_namespace::models::{
        CreateNamespaceRequest, CreateTableRequest, DeclareTableRequest, DescribeTableRequest,
        DropTableRequest, ListTablesRequest, TableExistsRequest,
    };
    use lance_table::feature_flags::FLAG_UNKNOWN;
    use lance_table::format::Fragment;
    use rstest::rstest;
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    async fn create_manifest_namespace(
        root: &str,
        inline_optimization_enabled: bool,
    ) -> ManifestNamespace {
        create_manifest_namespace_with_retries(root, inline_optimization_enabled, None).await
    }

    async fn create_manifest_namespace_with_retries(
        root: &str,
        inline_optimization_enabled: bool,
        commit_retries: Option<u32>,
    ) -> ManifestNamespace {
        let (object_store, base_path) = ObjectStore::from_uri_and_params(
            Arc::new(ObjectStoreRegistry::default()),
            root,
            &ObjectStoreParams::default(),
        )
        .await
        .unwrap();
        ManifestNamespace::from_directory(
            root.to_string(),
            None,
            None,
            object_store,
            base_path,
            true,
            inline_optimization_enabled,
            commit_retries,
        )
        .await
        .unwrap()
    }

    struct CommitConflictAfterRewriteMutation {
        root: String,
        conflict_object_id: String,
    }

    impl ManifestStreamMutation for CommitConflictAfterRewriteMutation {
        type Output = ();

        fn process_existing_row(
            &mut self,
            row: ManifestRowValue,
            output: &mut ManifestBatchBuilder,
        ) -> lance_core::Result<()> {
            output.append(ManifestOutputRow {
                object_id: &row.object_id,
                object_type: row.object_type,
                location: row.location.as_deref(),
                metadata: row.metadata.as_deref(),
                base_objects: row.base_objects.as_deref(),
            })
        }

        fn append_rows(&mut self, output: &mut ManifestBatchBuilder) -> lance_core::Result<()> {
            output.append(ManifestOutputRow {
                object_id: "attempted_table",
                object_type: ObjectType::Table,
                location: Some("attempted_table.lance"),
                metadata: None,
                base_objects: None,
            })
        }

        fn finish(&self) -> CopyOnWriteMutation<Self::Output> {
            let root = self.root.clone();
            let object_id = self.conflict_object_id.clone();
            std::thread::spawn(move || {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async move {
                    let writer = create_manifest_namespace(&root, false).await;
                    writer
                        .insert_into_manifest_with_metadata(
                            vec![ManifestEntry {
                                object_id,
                                object_type: ObjectType::Table,
                                location: Some("conflicting_table.lance".to_string()),
                                metadata: None,
                            }],
                            None,
                        )
                        .await
                        .unwrap();
                });
            })
            .join()
            .unwrap();
            CopyOnWriteMutation::updated(())
        }
    }

    /// A delete mutation that, during staging, has a concurrent writer delete the same
    /// object and commit first, so our own commit hits a conflict while the object is
    /// already gone — exercising `ConflictResolution::SucceedIfAbsent`.
    struct ConcurrentDeleteBeforeCommitMutation {
        inner: DeleteObjectMutation,
        root: String,
        target: String,
    }

    impl ManifestStreamMutation for ConcurrentDeleteBeforeCommitMutation {
        type Output = ();

        fn process_existing_row(
            &mut self,
            row: ManifestRowValue,
            output: &mut ManifestBatchBuilder,
        ) -> lance_core::Result<()> {
            self.inner.process_existing_row(row, output)
        }

        fn append_rows(&mut self, output: &mut ManifestBatchBuilder) -> lance_core::Result<()> {
            self.inner.append_rows(output)
        }

        fn finish(&self) -> CopyOnWriteMutation<Self::Output> {
            let root = self.root.clone();
            let target = self.target.clone();
            std::thread::spawn(move || {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                runtime.block_on(async move {
                    let writer = create_manifest_namespace(&root, false).await;
                    writer.delete_from_manifest(&target).await.unwrap();
                });
            })
            .join()
            .unwrap();
            self.inner.finish()
        }

        fn conflict_resolution(&self) -> ConflictResolution<Self::Output> {
            ConflictResolution::SucceedIfAbsent {
                object_id: self.target.clone(),
                output: (),
            }
        }
    }

    async fn manifest_base_objects(
        manifest_ns: &ManifestNamespace,
    ) -> HashMap<String, Option<Vec<String>>> {
        manifest_ns
            .manifest_dataset
            .get()
            .await
            .unwrap()
            .snapshot()
            .unwrap()
            .rows()
            .iter()
            .map(|row| (row.object_id.clone(), row.base_objects.clone()))
            .collect()
    }

    async fn manifest_data_paths(manifest_ns: &ManifestNamespace) -> HashSet<String> {
        let data_dir = manifest_ns
            .base_path
            .clone()
            .join(MANIFEST_TABLE_NAME)
            .join(LANCE_DATA_DIR);
        let mut stream = manifest_ns.object_store.read_dir_all(&data_dir, None);
        let mut paths = HashSet::new();
        while let Some(meta) = stream.next().await.transpose().unwrap() {
            paths.insert(meta.location.to_string());
        }
        paths
    }

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

    /// Open the `__manifest` dataset directly and set a table-metadata key,
    /// simulating a future Lance client that persisted a feature flag.
    async fn set_manifest_table_metadata(temp_path: &str, key: &str, value: &str) {
        use lance::dataset::builder::DatasetBuilder;
        let mut ds = DatasetBuilder::from_uri(format!("{}/{}", temp_path, MANIFEST_TABLE_NAME))
            .load()
            .await
            .unwrap();
        ds.update_metadata([(key, value)]).await.unwrap();
    }

    async fn create_namespace_with_one_table(temp_path: &str) {
        let ns = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["t1".to_string()]);
        ns.create_table(create_request, Bytes::from(create_test_ipc_data()))
            .await
            .unwrap();
    }

    /// This is a forward-compatibility checker only: it must not set any feature
    /// flag, so existing clients keep treating the manifest as compatible.
    #[tokio::test]
    async fn test_manifest_has_no_feature_flags_by_default() {
        use lance::dataset::builder::DatasetBuilder;
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        create_namespace_with_one_table(temp_path).await;

        let ds = DatasetBuilder::from_uri(format!("{}/{}", temp_path, MANIFEST_TABLE_NAME))
            .load()
            .await
            .unwrap();
        assert!(
            !ds.metadata()
                .contains_key(crate::dir::manifest_feature_flags::READER_FEATURE_FLAGS_KEY)
        );
        assert!(
            !ds.metadata()
                .contains_key(crate::dir::manifest_feature_flags::WRITER_FEATURE_FLAGS_KEY)
        );
    }

    /// An unknown reader feature flag must block opening the catalog with a clear
    /// "please upgrade" error rather than silently degrading to directory listing.
    #[tokio::test]
    async fn test_unknown_reader_flag_blocks_access() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        create_namespace_with_one_table(temp_path).await;
        set_manifest_table_metadata(
            temp_path,
            crate::dir::manifest_feature_flags::READER_FEATURE_FLAGS_KEY,
            "1",
        )
        .await;

        let err = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .expect_err("opening a manifest with an unknown reader flag should fail");
        assert!(
            err.to_string().to_lowercase().contains("upgrade"),
            "expected an upgrade error, got: {err}"
        );
    }

    /// An unknown writer feature flag must still allow reads but block writes.
    #[tokio::test]
    async fn test_unknown_writer_flag_blocks_writes_but_allows_reads() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        create_namespace_with_one_table(temp_path).await;
        set_manifest_table_metadata(
            temp_path,
            crate::dir::manifest_feature_flags::WRITER_FEATURE_FLAGS_KEY,
            "1",
        )
        .await;

        let ns = DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .expect("reads should still be allowed with only a writer flag set");
        let mut list_request = ListTablesRequest::new();
        list_request.id = Some(vec![]);
        assert_eq!(ns.list_tables(list_request).await.unwrap().tables.len(), 1);

        // A refused write must not leave an orphaned table dataset behind.
        let entries_before = dir_entry_names(temp_path);
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["t2".to_string()]);
        let err = ns
            .create_table(create_request, Bytes::from(create_test_ipc_data()))
            .await
            .expect_err("writing through an unknown writer flag should fail");
        assert!(
            err.to_string().to_lowercase().contains("upgrade"),
            "expected an upgrade error, got: {err}"
        );
        assert_eq!(
            entries_before,
            dir_entry_names(temp_path),
            "a refused create_table must not create an orphaned table directory"
        );

        // Mutations that go straight through rewrite_manifest (no early
        // create_table check) must also be refused: an insert (create_namespace)
        // and a delete (drop_table). This proves the writer check is enforced at
        // the single copy-on-write chokepoint, not just on the create_table path.
        let mut create_ns = CreateNamespaceRequest::new();
        create_ns.id = Some(vec!["ns1".to_string()]);
        let err = ns
            .create_namespace(create_ns)
            .await
            .expect_err("create_namespace through an unknown writer flag should fail");
        assert!(
            err.to_string().to_lowercase().contains("upgrade"),
            "expected an upgrade error, got: {err}"
        );

        let mut drop_request = DropTableRequest::new();
        drop_request.id = Some(vec!["t1".to_string()]);
        let err = ns
            .drop_table(drop_request)
            .await
            .expect_err("drop_table through an unknown writer flag should fail");
        assert!(
            err.to_string().to_lowercase().contains("upgrade"),
            "expected an upgrade error, got: {err}"
        );
    }

    fn dir_entry_names(path: &str) -> std::collections::BTreeSet<String> {
        std::fs::read_dir(path)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect()
    }

    #[tokio::test]
    async fn test_manifest_rewrite_preserves_utf8_metadata_and_base_objects() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace(temp_path, true).await;

        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "view".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("view.lance".to_string()),
                    metadata: Some(r#"{"kind":"view"}"#.to_string()),
                }],
                Some(vec!["base_a".to_string(), "base_b".to_string()]),
            )
            .await
            .unwrap();
        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "other".to_string(),
                    object_type: ObjectType::Namespace,
                    location: None,
                    metadata: Some(r#"{"kind":"namespace"}"#.to_string()),
                }],
                None,
            )
            .await
            .unwrap();

        let dataset_guard = manifest_ns.manifest_dataset.get().await.unwrap();
        let metadata_field = dataset_guard.schema().field("metadata").unwrap();
        assert_eq!(metadata_field.data_type(), DataType::Utf8);
        drop(dataset_guard);

        let base_objects = manifest_base_objects(&manifest_ns).await;
        assert_eq!(
            base_objects.get("view").cloned().unwrap(),
            Some(vec!["base_a".to_string(), "base_b".to_string()])
        );
        assert_eq!(base_objects.get("other").cloned().unwrap(), None);
    }

    #[tokio::test]
    async fn test_manifest_rewrite_updates_snapshot_without_indices() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace(temp_path, true).await;

        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table.lance".to_string()),
                    metadata: None,
                }],
                Some(vec!["base".to_string()]),
            )
            .await
            .unwrap();

        let dataset_guard = manifest_ns.manifest_dataset.get().await.unwrap();
        let indices = dataset_guard.load_indices().await.unwrap();
        assert!(indices.is_empty());
        let row = dataset_guard.snapshot().get("table").unwrap();
        assert_eq!(row.location.as_deref(), Some("table.lance"));
        assert_eq!(
            row.base_objects.as_deref(),
            Some(["base".to_string()].as_slice())
        );
    }

    #[tokio::test]
    async fn test_manifest_rewrite_rejects_duplicate_input_ids() {
        let temp_dir = TempStdDir::default();
        let manifest_ns = create_manifest_namespace(temp_dir.to_str().unwrap(), false).await;
        let version_before = manifest_ns
            .manifest_dataset
            .get()
            .await
            .unwrap()
            .version()
            .version;
        let entries = ["first.lance", "second.lance"]
            .into_iter()
            .map(|location| ManifestEntry {
                object_id: "duplicate".to_string(),
                object_type: ObjectType::Table,
                location: Some(location.to_string()),
                metadata: None,
            })
            .collect();

        let err = manifest_ns
            .insert_into_manifest_with_metadata(entries, None)
            .await
            .unwrap_err();

        assert!(matches!(&err, lance_core::Error::Namespace { .. }));
        assert!(err.to_string().contains("duplicate object_id 'duplicate'"));
        let version_after = manifest_ns
            .manifest_dataset
            .get()
            .await
            .unwrap()
            .version()
            .version;
        assert_eq!(version_after, version_before);
    }

    #[tokio::test]
    async fn test_manifest_rewrite_empty_manifest_updates_snapshot() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace(temp_path, true).await;

        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap();
        manifest_ns.delete_from_manifest("table").await.unwrap();

        assert!(!manifest_ns.manifest_contains_object("table").await.unwrap());
        let dataset_guard = manifest_ns.manifest_dataset.get().await.unwrap();
        assert!(dataset_guard.snapshot().rows().is_empty());
        let indices = dataset_guard.load_indices().await.unwrap();
        assert!(indices.is_empty());
    }

    #[tokio::test]
    async fn test_manifest_rewrite_assigns_unique_fragment_ids() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace(temp_path, false).await;
        let dataset_guard = manifest_ns.manifest_dataset.get().await.unwrap();
        let fragments = vec![Fragment::new(0), Fragment::new(0), Fragment::new(7)];

        let manifest = ManifestNamespace::manifest_from_overwrite_transaction(
            dataset_guard.manifest(),
            dataset_guard.manifest().schema.clone(),
            &fragments,
        );

        let fragment_ids = manifest
            .fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>();
        assert_eq!(fragment_ids, vec![0, 1, 7]);
    }

    #[tokio::test]
    async fn test_manifest_writes_reject_unknown_writer_flag_before_staging() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace(temp_path, false).await;
        let data_paths_before = manifest_data_paths(&manifest_ns).await;
        let original_version = {
            let mut dataset = manifest_ns.manifest_dataset.get_mut().await.unwrap();
            let mut manifest = dataset.manifest().clone();
            manifest.writer_feature_flags |= FLAG_UNKNOWN << 1;
            let version = manifest.version;
            dataset.manifest = Arc::new(manifest);
            version
        };

        let entries_before = dir_entry_names(temp_path);
        let mut declare_request = DeclareTableRequest::new();
        declare_request.id = Some(vec!["declared_table".to_string()]);
        let error = manifest_ns
            .declare_table(declare_request)
            .await
            .unwrap_err();
        assert!(
            error.to_string().to_lowercase().contains("upgrade"),
            "expected an upgrade error, got: {error}"
        );
        assert_eq!(dir_entry_names(temp_path), entries_before);

        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["new_table".to_string()]);
        let error = manifest_ns
            .create_table(create_request, Bytes::from(create_test_ipc_data()))
            .await
            .unwrap_err();
        assert!(
            error.to_string().to_lowercase().contains("upgrade"),
            "expected an upgrade error, got: {error}"
        );
        assert_eq!(dir_entry_names(temp_path), entries_before);

        let error = manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap_err();

        assert!(
            error.to_string().to_lowercase().contains("upgrade"),
            "expected an upgrade error, got: {error}"
        );
        assert_eq!(
            manifest_ns
                .manifest_dataset
                .get()
                .await
                .unwrap()
                .version()
                .version,
            original_version
        );
        assert_eq!(manifest_data_paths(&manifest_ns).await, data_paths_before);
    }

    #[tokio::test]
    async fn test_manifest_noop_delete_uses_latest_snapshot() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let stale_ns = create_manifest_namespace(temp_path, false).await;
        let writer_ns = create_manifest_namespace(temp_path, false).await;

        writer_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "late_table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("late_table.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap();

        stale_ns.delete_from_manifest("late_table").await.unwrap();

        let check_ns = create_manifest_namespace(temp_path, false).await;
        assert!(
            !check_ns
                .manifest_contains_object("late_table")
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_manifest_noop_delete_cleans_uncommitted_data_file() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace(temp_path, false).await;

        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap();

        let before = manifest_data_paths(&manifest_ns).await;
        assert!(!before.is_empty());

        manifest_ns
            .delete_from_manifest("missing_table")
            .await
            .unwrap();

        let after = manifest_data_paths(&manifest_ns).await;
        assert_eq!(after, before);
    }

    #[tokio::test]
    async fn test_manifest_final_commit_failure_cleans_uncommitted_rewrite_files() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace_with_retries(temp_path, true, Some(0)).await;

        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap();

        let before_data_paths = manifest_data_paths(&manifest_ns).await;

        let result = manifest_ns
            .rewrite_manifest("Failed to test manifest cleanup", || {
                CommitConflictAfterRewriteMutation {
                    root: temp_path.to_string(),
                    conflict_object_id: "conflicting_table".to_string(),
                }
            })
            .await;
        assert!(result.is_err());

        let after_data_paths = manifest_data_paths(&manifest_ns).await;
        assert!(before_data_paths.is_subset(&after_data_paths));
        assert_eq!(after_data_paths.len(), before_data_paths.len() + 1);
        assert!(
            manifest_ns
                .manifest_contains_object("conflicting_table")
                .await
                .unwrap()
        );
        assert!(
            !manifest_ns
                .manifest_contains_object("attempted_table")
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_manifest_commit_visible_on_memory_store() {
        // Regression: the commit must use the same object store the manifest dataset reads
        // from. On `memory://` the namespace store and the dataset store can be different
        // in-memory instances, so a commit written to the wrong one is invisible to reads
        // (manifests as stale version -> endless conflict / "not found").
        let manifest_ns = create_manifest_namespace("memory://test_commit_visible", false).await;
        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap();
        assert!(manifest_ns.manifest_contains_object("table").await.unwrap());
        // A second sequential commit must not falsely conflict.
        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table2".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table2.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap();
        assert!(
            manifest_ns
                .manifest_contains_object("table2")
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn test_manifest_commit_uses_inline_transaction() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace(temp_path, false).await;

        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap();

        let dataset_guard = manifest_ns.manifest_dataset.get().await.unwrap();
        let manifest = dataset_guard.manifest();
        // The overwrite transaction is embedded inline in the manifest, never written as a
        // separate _transactions/*.txn file.
        assert!(manifest.transaction_section.is_some());
        assert!(manifest.transaction_file.is_none());
    }

    #[tokio::test]
    async fn test_manifest_commit_landed_attributes_data_file() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace(temp_path, false).await;

        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap();

        let dataset = Arc::new(manifest_ns.manifest_dataset.get().await.unwrap().clone());
        let version = dataset.manifest().version;
        let our_files = dataset
            .manifest()
            .fragments
            .iter()
            .flat_map(|fragment| fragment.files.iter())
            .map(|file| file.path.clone())
            .collect::<HashSet<_>>();
        assert!(!our_files.is_empty());

        // The committed version references our data file => attributed to us (a lost-ack
        // commit must be treated as success, not cleaned up).
        assert!(
            manifest_ns
                .manifest_commit_landed(&dataset, version, &our_files)
                .await
        );
        // A different file set is not attributed to us.
        let other = HashSet::from(["missing.lance".to_string()]);
        assert!(
            !manifest_ns
                .manifest_commit_landed(&dataset, version, &other)
                .await
        );
        // A version that does not exist did not land.
        assert!(
            !manifest_ns
                .manifest_commit_landed(&dataset, version + 100, &our_files)
                .await
        );
    }

    #[tokio::test]
    async fn test_manifest_delete_conflict_with_concurrent_delete_succeeds() {
        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();
        let manifest_ns = create_manifest_namespace_with_retries(temp_path, false, Some(0)).await;

        manifest_ns
            .insert_into_manifest_with_metadata(
                vec![ManifestEntry {
                    object_id: "table".to_string(),
                    object_type: ObjectType::Table,
                    location: Some("table.lance".to_string()),
                    metadata: None,
                }],
                None,
            )
            .await
            .unwrap();
        assert!(manifest_ns.manifest_contains_object("table").await.unwrap());

        // A concurrent writer deletes "table" and commits first, so our own delete commit
        // conflicts while "table" is already gone. Native resolution treats the goal as
        // achieved and succeeds instead of erroring or retrying forever.
        let result = manifest_ns
            .rewrite_manifest("Failed to delete from manifest", || {
                ConcurrentDeleteBeforeCommitMutation {
                    inner: DeleteObjectMutation {
                        object_id: "table".to_string(),
                        deleted: false,
                    },
                    root: temp_path.to_string(),
                    target: "table".to_string(),
                }
            })
            .await;

        assert!(result.is_ok(), "delete should succeed: {result:?}");
        assert!(!manifest_ns.manifest_contains_object("table").await.unwrap());
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
    /// two separate OS processes racing on table creation. Copy-on-write rewrite
    /// retries ensure the second operation detects the duplicate after retrying
    /// against the latest data.
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

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_alter_table_add_columns(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            AddColumnsEntry, AlterTableAddColumnsRequest, DescribeTableRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a table with id and name columns
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // Add a new column using SQL expression
        let mut new_col = AddColumnsEntry::new("doubled_id".to_string());
        new_col.expression = Some(Some("id * 2".to_string()));
        let mut add_request = AlterTableAddColumnsRequest::new(vec![new_col]);
        add_request.id = Some(vec!["test_table".to_string()]);

        let response = dir_namespace
            .alter_table_add_columns(add_request)
            .await
            .unwrap();
        // Version should have incremented
        assert!(response.version > 1);

        // Verify the column was added by describing the table with detailed metadata
        let mut describe_request = DescribeTableRequest::new();
        describe_request.id = Some(vec!["test_table".to_string()]);
        describe_request.load_detailed_metadata = Some(true);
        let describe_response = dir_namespace
            .describe_table(describe_request)
            .await
            .unwrap();
        assert!(describe_response.schema.is_some());

        let schema = describe_response.schema.unwrap();
        let field_names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
        assert!(
            field_names.contains(&"doubled_id"),
            "Column 'doubled_id' should exist after add_columns, got: {:?}",
            field_names
        );
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_alter_table_add_columns_missing_id(#[case] inline_optimization: bool) {
        use lance_namespace::models::{AddColumnsEntry, AlterTableAddColumnsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Request without ID should fail
        let new_col = AddColumnsEntry::new("col".to_string());
        let request = AlterTableAddColumnsRequest::new(vec![new_col]);
        let result = dir_namespace.alter_table_add_columns(request).await;
        assert!(result.is_err(), "Should fail when table ID is missing");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_alter_table_add_columns_nonexistent_table(#[case] inline_optimization: bool) {
        use lance_namespace::models::{AddColumnsEntry, AlterTableAddColumnsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Request with non-existent table should fail
        let new_col = AddColumnsEntry::new("col".to_string());
        let mut request = AlterTableAddColumnsRequest::new(vec![new_col]);
        request.id = Some(vec!["nonexistent".to_string()]);
        let result = dir_namespace.alter_table_add_columns(request).await;
        assert!(result.is_err(), "Should fail when table does not exist");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_alter_table_alter_columns_rename(#[case] inline_optimization: bool) {
        use lance_namespace::models::{
            AlterColumnsEntry, AlterTableAlterColumnsRequest, DescribeTableRequest,
        };

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a table
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // Rename the "name" column to "full_name"
        let mut entry = AlterColumnsEntry::new("name".to_string());
        entry.rename = Some(Some("full_name".to_string()));
        let mut alter_request = AlterTableAlterColumnsRequest::new(vec![entry]);
        alter_request.id = Some(vec!["test_table".to_string()]);

        let response = dir_namespace
            .alter_table_alter_columns(alter_request)
            .await
            .unwrap();
        assert!(response.version > 1);

        // Verify the column was renamed
        let mut describe_request = DescribeTableRequest::new();
        describe_request.id = Some(vec!["test_table".to_string()]);
        describe_request.load_detailed_metadata = Some(true);
        let describe_response = dir_namespace
            .describe_table(describe_request)
            .await
            .unwrap();
        assert!(describe_response.schema.is_some());

        let schema = describe_response.schema.unwrap();
        let field_names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
        assert!(
            field_names.contains(&"full_name"),
            "Column should be renamed to 'full_name', got: {:?}",
            field_names
        );
        assert!(
            !field_names.contains(&"name"),
            "Old column name 'name' should no longer exist, got: {:?}",
            field_names
        );
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_alter_table_alter_columns_missing_id(#[case] inline_optimization: bool) {
        use lance_namespace::models::{AlterColumnsEntry, AlterTableAlterColumnsRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        let entry = AlterColumnsEntry::new("name".to_string());
        let request = AlterTableAlterColumnsRequest::new(vec![entry]);
        let result = dir_namespace.alter_table_alter_columns(request).await;
        assert!(result.is_err(), "Should fail when table ID is missing");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_alter_table_drop_columns(#[case] inline_optimization: bool) {
        use lance_namespace::models::{AlterTableDropColumnsRequest, DescribeTableRequest};

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        // Create a table with id and name columns
        let buffer = create_test_ipc_data();
        let mut create_request = CreateTableRequest::new();
        create_request.id = Some(vec!["test_table".to_string()]);
        dir_namespace
            .create_table(create_request, Bytes::from(buffer))
            .await
            .unwrap();

        // Drop the "name" column
        let mut drop_request = AlterTableDropColumnsRequest::new(vec!["name".to_string()]);
        drop_request.id = Some(vec!["test_table".to_string()]);

        let response = dir_namespace
            .alter_table_drop_columns(drop_request)
            .await
            .unwrap();
        assert!(response.version > 1);

        // Verify the column was dropped
        let mut describe_request = DescribeTableRequest::new();
        describe_request.id = Some(vec!["test_table".to_string()]);
        describe_request.load_detailed_metadata = Some(true);
        let describe_response = dir_namespace
            .describe_table(describe_request)
            .await
            .unwrap();
        assert!(describe_response.schema.is_some());

        let schema = describe_response.schema.unwrap();
        let field_names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
        assert!(
            !field_names.contains(&"name"),
            "Column 'name' should have been dropped, got: {:?}",
            field_names
        );
        assert!(
            field_names.contains(&"id"),
            "Column 'id' should still exist, got: {:?}",
            field_names
        );
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_alter_table_drop_columns_missing_id(#[case] inline_optimization: bool) {
        use lance_namespace::models::AlterTableDropColumnsRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        let request = AlterTableDropColumnsRequest::new(vec!["col".to_string()]);
        let result = dir_namespace.alter_table_drop_columns(request).await;
        assert!(result.is_err(), "Should fail when table ID is missing");
    }

    #[rstest]
    #[case::with_optimization(true)]
    #[case::without_optimization(false)]
    #[tokio::test]
    async fn test_alter_table_drop_columns_nonexistent_table(#[case] inline_optimization: bool) {
        use lance_namespace::models::AlterTableDropColumnsRequest;

        let temp_dir = TempStdDir::default();
        let temp_path = temp_dir.to_str().unwrap();

        let dir_namespace = DirectoryNamespaceBuilder::new(temp_path)
            .inline_optimization_enabled(inline_optimization)
            .build()
            .await
            .unwrap();

        let mut request = AlterTableDropColumnsRequest::new(vec!["col".to_string()]);
        request.id = Some(vec!["nonexistent".to_string()]);
        let result = dir_namespace.alter_table_drop_columns(request).await;
        assert!(result.is_err(), "Should fail when table does not exist");
    }
}
