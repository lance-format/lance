// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;

use chrono::TimeDelta;
use lance_core::{Error, Result};
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_file::writer::FileWriterOptions;
use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry};
use lance_table::format::BasePath;
use lance_table::io::commit::CommitHandler;

use super::progress::{NoopFragmentWriteProgress, WriteFragmentProgress, WriteProgressFn};
use crate::TableSession;

/// The mode to write dataset.
#[derive(Debug, Clone, Copy)]
pub enum WriteMode {
    /// Create a new dataset. Expect the dataset does not exist.
    Create,
    /// Append to an existing dataset.
    Append,
    /// Overwrite a dataset as a new version, or create new dataset if not exist.
    Overwrite,
}

impl TryFrom<&str> for WriteMode {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            "create" => Ok(Self::Create),
            "append" => Ok(Self::Append),
            "overwrite" => Ok(Self::Overwrite),
            _ => Err(Error::invalid_input(format!(
                "Invalid write mode: {}",
                value
            ))),
        }
    }
}

/// The strategy for handling external blob URIs on write.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ExternalBlobMode {
    /// Store the URI as an external blob reference.
    #[default]
    Reference,
    /// Read the external bytes during write and store them in Lance-managed storage.
    Ingest,
}

impl TryFrom<&str> for ExternalBlobMode {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            "reference" => Ok(Self::Reference),
            "ingest" => Ok(Self::Ingest),
            _ => Err(Error::invalid_input(format!(
                "Invalid external blob mode: {}",
                value
            ))),
        }
    }
}

/// Auto cleanup parameters
#[derive(Debug, Clone)]
pub struct AutoCleanupParams {
    pub interval: usize,
    pub older_than: TimeDelta,
}

impl Default for AutoCleanupParams {
    fn default() -> Self {
        Self {
            interval: 20,
            older_than: TimeDelta::days(14),
        }
    }
}

/// Dataset Write Parameters
#[derive(Debug, Clone)]
pub struct WriteParams<S> {
    /// Max number of records per file.
    ///
    /// Must not exceed `u32::MAX`, the row capacity of a fragment.
    pub max_rows_per_file: usize,

    /// Max number of rows per row group.
    pub max_rows_per_group: usize,

    /// Max file size in bytes.
    ///
    /// This is a soft limit. The actual file size may be larger than this value
    /// by a few megabytes, since once we detect we hit this limit, we still
    /// need to flush the footer.
    ///
    /// This limit is checked after writing each group, so if max_rows_per_group
    /// is set to a large value, this limit may be exceeded by a large amount.
    ///
    /// The default is 90 GB. If you are using an object store such as S3, we
    /// currently have a hard 100 GB limit.
    pub max_bytes_per_file: usize,

    /// Write mode
    pub mode: WriteMode,

    /// Default object store params for the write.
    ///
    /// Storage options may carry base-scoped entries (`base_<id>.<key>`) that
    /// apply only to the registered base path with that id, overriding the
    /// unscoped options that every base inherits.
    pub store_params: Option<ObjectStoreParams>,

    /// Exact object store params per base path URI, taking precedence over
    /// `base_<id>.<key>` storage options in [`Self::store_params`]. See
    /// [`Self::with_base_store_params`].
    pub base_store_params: Option<HashMap<String, ObjectStoreParams>>,

    pub progress: Arc<dyn WriteFragmentProgress>,

    /// Optional callback invoked after each batch is written.
    ///
    /// Receives cumulative [`super::WriteStats`] so callers can render a progress bar
    /// or compute throughput. The callback must be cheap and non-blocking;
    /// spawn a task if you need async work.
    pub write_progress: Option<WriteProgressFn>,

    /// If present, dataset will use this to update the latest version
    ///
    /// If not set, the default will be based on the object store.  Generally this will
    /// be RenameCommitHandler unless the object store does not handle atomic renames (e.g. S3)
    ///
    /// If a custom object store is provided (via store_params.object_store) then this
    /// must also be provided.
    pub commit_handler: Option<Arc<dyn CommitHandler>>,

    /// The format version to use when writing data.
    ///
    /// Newer versions are more efficient but the data can only be read by more recent versions
    /// of lance.
    /// Lance file version 2.3 enables RLE v2 run length widths by default.
    ///
    /// For an existing dataset, an explicit version is the exact target for
    /// this operation; if omitted, the manifest default storage version is used.
    /// New datasets default to the latest stable version.
    pub data_storage_version: Option<LanceFileVersion>,

    /// Experimental: if set to true, the writer will use stable row ids.
    /// A row then keeps the same id for its lifetime: compaction, update and
    /// merge insert relocate or rewrite the row without changing its id.
    ///
    /// This only applies to new datasets. A write to an existing dataset does
    /// not change that dataset's setting, so this value can neither enable nor
    /// disable the feature after creation; to enable it later, use the
    /// `migrate_to_stable_row_ids` method. Default is false.
    pub enable_stable_row_ids: bool,

    /// If set to true, and this is a new dataset, uses the new v2 manifest paths.
    /// These allow constant-time lookups for the latest manifest on object storage.
    /// This parameter has no effect on existing datasets. To migrate an existing
    /// dataset, use the `migrate_manifest_paths_v2` method.
    /// Default is True.
    pub enable_v2_manifest_paths: bool,

    /// Session supplying the object-store registry and caches.
    pub session: Option<Arc<S>>,

    /// If Some and this is a new dataset, old dataset versions will be
    /// automatically cleaned up after commits according to the parameters set
    /// out in [`AutoCleanupParams`]. This parameter has no effect on existing
    /// datasets. To add auto-cleanup to an existing dataset, use
    /// `update_config` to set `lance.auto_cleanup.interval` and
    /// `lance.auto_cleanup.older_than`. Both parameters must be set to invoke
    /// auto-cleanup.
    ///
    /// Defaults to `None` (auto-cleanup disabled). Enabling it makes every
    /// `interval`-th commit run a full cleanup pass, which lists and reads every
    /// manifest in the dataset even when nothing is old enough to delete; on
    /// object stores this adds noticeable per-commit latency that grows with the
    /// version count. Prefer calling `cleanup_old_versions` explicitly
    /// when you actually need to reclaim space.
    pub auto_cleanup: Option<AutoCleanupParams>,

    /// If true, skip auto cleanup during commits. This should be set to true
    /// for high frequency writes to improve performance. This is also useful
    /// if the writer does not have delete permissions and the clean up would
    /// just try and log a failure anyway. Default is false.
    pub skip_auto_cleanup: bool,

    /// If true, this write's commit stores its transaction only inline in the
    /// manifest and skips the separate file under `_transactions/`, saving one
    /// object write per commit. The transaction is inlined regardless of its
    /// size (up to `u32::MAX` bytes), so very large transactions grow the
    /// manifest instead of spilling to a separate file.
    ///
    /// The new manifest gets a writer feature flag, so Lance releases older than
    /// v1.0.0 refuse to write to that version. The setting applies only to this
    /// commit: later commits that leave it unset write the transaction file again
    /// and drop the flag.
    ///
    /// Readers older than Lance 14.0 return no transaction for such a version
    /// and cannot rebase a concurrent commit across it. Keep the transaction
    /// file when clients older than 14.0 read transaction history or commit
    /// concurrently.
    ///
    /// Default is false.
    pub disable_transaction_file: bool,

    /// Configuration key-value pairs for this write operation.
    /// This can include commit messages, engine information, etc.
    /// this properties map will be persisted as part of Transaction object.
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,

    /// New base paths to register in the manifest during dataset creation.
    /// An id of 0 requests an unused id; explicit non-zero ids must not
    /// collide with existing bases or each other.
    /// Only used in CREATE/OVERWRITE modes for manifest registration.
    pub initial_bases: Option<Vec<BasePath>>,

    /// Target base IDs for writing data files.
    /// When provided, all new data files will be written to bases with these IDs.
    /// Used in all modes (CREATE, APPEND, OVERWRITE) to specify where data should be written.
    /// The IDs must correspond to either:
    /// - IDs in initial_bases (for CREATE/OVERWRITE modes)
    /// - IDs already registered in the existing dataset manifest (for APPEND mode)
    /// - [`crate::write::PRIMARY_BASE_ID`] (0), which targets the dataset's primary storage
    ///   and participates in the round-robin like any other entry
    pub target_bases: Option<Vec<u32>>,

    /// Target base names or paths as strings (unresolved).
    /// These will be resolved to IDs when the write operation executes.
    /// Resolution happens at builder execution time when dataset context is available.
    /// An entry equal to the dataset's URI targets the dataset's primary storage.
    pub target_base_names_or_paths: Option<Vec<String>>,

    /// Target every base registered in the dataset manifest, resolved when the
    /// write executes. `Some(include_primary)`: when `include_primary` is true
    /// the dataset's primary storage participates in the rotation as the first
    /// slot. Cannot be combined with `target_bases` or
    /// `target_base_names_or_paths`.
    pub target_all_bases: Option<bool>,

    /// Allow writing external blob URIs that cannot be mapped to any registered
    /// non-dataset-root base path. When disabled, such rows are rejected.
    pub allow_external_blob_outside_bases: bool,

    /// The strategy used when writing external blob URIs.
    pub external_blob_mode: ExternalBlobMode,

    /// Maximum size in bytes for blob v2 pack (.blob) sidecar files.
    /// When a pack file reaches this size, a new one is started.
    /// If not set, defaults to 1 GiB.
    pub blob_pack_file_size_threshold: Option<usize>,

    /// File writer options to use when writing data files.
    ///
    /// Options set here apply to current-format data files. They have no effect
    /// when writing legacy V1 files. If not set, the file writer uses its
    /// configured defaults.
    pub file_writer_options: Option<FileWriterOptions>,
}

impl<S> Default for WriteParams<S> {
    fn default() -> Self {
        Self {
            max_rows_per_file: 1024 * 1024, // 1 million
            max_rows_per_group: 1024,
            // object-store has a 100GB limit, so we should at least make sure
            // we are under that.
            max_bytes_per_file: 90 * 1024 * 1024 * 1024, // 90 GB
            mode: WriteMode::Create,
            store_params: None,
            base_store_params: None,
            progress: Arc::new(NoopFragmentWriteProgress::new()),
            write_progress: None,
            commit_handler: None,
            data_storage_version: None,
            enable_stable_row_ids: false,
            enable_v2_manifest_paths: true,
            session: None,
            auto_cleanup: None,
            skip_auto_cleanup: false,
            disable_transaction_file: false,
            transaction_properties: None,
            initial_bases: None,
            target_bases: None,
            target_base_names_or_paths: None,
            target_all_bases: None,
            allow_external_blob_outside_bases: false,
            external_blob_mode: ExternalBlobMode::Reference,
            blob_pack_file_size_threshold: None,
            file_writer_options: None,
        }
    }
}

impl<S> WriteParams<S> {
    /// Use the same write options with a different session binding.
    #[doc(hidden)]
    pub fn with_session_type<T>(self, session: Option<Arc<T>>) -> WriteParams<T> {
        WriteParams {
            max_rows_per_file: self.max_rows_per_file,
            max_rows_per_group: self.max_rows_per_group,
            max_bytes_per_file: self.max_bytes_per_file,
            mode: self.mode,
            store_params: self.store_params,
            base_store_params: self.base_store_params,
            progress: self.progress,
            write_progress: self.write_progress,
            commit_handler: self.commit_handler,
            data_storage_version: self.data_storage_version,
            enable_stable_row_ids: self.enable_stable_row_ids,
            enable_v2_manifest_paths: self.enable_v2_manifest_paths,
            session,
            auto_cleanup: self.auto_cleanup,
            skip_auto_cleanup: self.skip_auto_cleanup,
            disable_transaction_file: self.disable_transaction_file,
            transaction_properties: self.transaction_properties,
            initial_bases: self.initial_bases,
            target_bases: self.target_bases,
            target_base_names_or_paths: self.target_base_names_or_paths,
            target_all_bases: self.target_all_bases,
            allow_external_blob_outside_bases: self.allow_external_blob_outside_bases,
            external_blob_mode: self.external_blob_mode,
            blob_pack_file_size_threshold: self.blob_pack_file_size_threshold,
            file_writer_options: self.file_writer_options,
        }
    }

    /// Create a new WriteParams with the given storage version.
    /// The other fields are set to their default values.
    pub fn with_storage_version(version: LanceFileVersion) -> Self {
        Self {
            data_storage_version: Some(version),
            ..Default::default()
        }
    }

    pub fn storage_version_or_default(&self) -> ConcreteFileVersion {
        self.data_storage_version.unwrap_or_default().resolve()
    }

    pub fn store_registry(&self) -> Arc<ObjectStoreRegistry>
    where
        S: AsRef<TableSession>,
    {
        self.session
            .as_ref()
            .map(|session| session.as_ref().as_ref().store_registry())
            .unwrap_or_default()
    }

    /// Set exact runtime object store params for a registered base path.
    ///
    /// These params are used as-is for that base, taking precedence over
    /// `base_<id>.<key>` storage options in `store_params`. The write-level
    /// default `store_params` remain the fallback for bases without an
    /// explicit binding.
    pub fn with_base_store_params(
        mut self,
        base_path: impl AsRef<str>,
        store_params: ObjectStoreParams,
    ) -> Self {
        self.base_store_params
            .get_or_insert_with(HashMap::new)
            .insert(base_path.as_ref().to_string(), store_params);
        self
    }

    /// Set the properties for this WriteParams.
    pub fn with_transaction_properties(self, properties: HashMap<String, String>) -> Self {
        Self {
            transaction_properties: Some(Arc::new(properties)),
            ..self
        }
    }

    /// Set the initial_bases for this WriteParams.
    ///
    /// This specifies new base paths to register in the manifest during dataset creation.
    /// An id of 0 requests an unused id; explicit non-zero ids must not collide
    /// with existing bases or each other.
    /// Only used in CREATE/OVERWRITE modes for manifest registration.
    pub fn with_initial_bases(self, bases: Vec<BasePath>) -> Self {
        Self {
            initial_bases: Some(bases),
            ..self
        }
    }

    /// Set the target_bases for this WriteParams.
    ///
    /// This specifies the base IDs where data files should be written.
    /// The IDs must correspond to either:
    /// - IDs in initial_bases (for CREATE/OVERWRITE modes)
    /// - IDs already registered in the existing dataset manifest (for APPEND mode)
    pub fn with_target_bases(self, base_ids: Vec<u32>) -> Self {
        Self {
            target_bases: Some(base_ids),
            ..self
        }
    }

    /// Store target base names or paths for deferred resolution.
    ///
    /// This method stores the references in `target_base_names_or_paths` field
    /// to be resolved later at execution time when the dataset manifest is available.
    ///
    /// Resolution will happen at write execution time and will try to match:
    /// 1. initial_bases by name
    /// 2. initial_bases by path
    /// 3. existing manifest by name
    /// 4. existing manifest by path
    ///
    /// # Arguments
    ///
    /// * `references` - Vector of base names or paths to be resolved later
    pub fn with_target_base_names_or_paths(self, references: Vec<String>) -> Self {
        Self {
            target_base_names_or_paths: Some(references),
            ..self
        }
    }

    /// Target every base registered in the dataset manifest, resolved when the
    /// write executes. When `include_primary` is true the dataset's primary
    /// storage participates in the rotation as the first slot.
    ///
    /// Cannot be combined with [`Self::with_target_bases`] or
    /// [`Self::with_target_base_names_or_paths`].
    pub fn with_target_all_bases(self, include_primary: bool) -> Self {
        Self {
            target_all_bases: Some(include_primary),
            ..self
        }
    }

    /// Configure whether external blobs outside registered bases are allowed.
    pub fn with_allow_external_blob_outside_bases(self, allow: bool) -> Self {
        Self {
            allow_external_blob_outside_bases: allow,
            ..self
        }
    }

    /// Configure how external blob URIs are handled during writes.
    pub fn with_external_blob_mode(self, mode: ExternalBlobMode) -> Self {
        Self {
            external_blob_mode: mode,
            ..self
        }
    }

    /// Set the maximum size in bytes for blob v2 pack (.blob) sidecar files.
    pub fn with_blob_pack_file_size_threshold(self, max_bytes: usize) -> Self {
        Self {
            blob_pack_file_size_threshold: Some(max_bytes),
            ..self
        }
    }
}
