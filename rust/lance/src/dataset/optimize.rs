// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Table maintenance for optimizing table layout.
//!
//! As a table is updated, its layout can become suboptimal. For example, if
//! a series of small streaming appends are performed, eventually there will be
//! a large number of small files. This imposes an overhead to track the large
//! number of files and for very small files can make it harder to read data
//! efficiently. In this case, files can be compacted into fewer larger files.
//!
//! To compact files in a table, use the [compact_files] method. This currently
//! can compact in two cases:
//!
//! 1. If a fragment has fewer rows than the target number of rows per fragment.
//!    The fragment must also have neighbors that are also candidates for
//!    compaction.
//! 2. If a fragment has a higher percentage of deleted rows than the provided
//!    threshold.
//!
//! In addition to the rules above there may be restrictions due to indexes.
//! When a fragment is compacted its row ids change and any index that contained
//! that fragment will be remapped.  However, we cannot combine indexed fragments
//! with unindexed fragments.
//!
//! ```rust
//! # use std::sync::Arc;
//! # use tokio::runtime::Runtime;
//! # use arrow_array::{RecordBatch, RecordBatchIterator, Int64Array};
//! # use arrow_schema::{Schema, Field, DataType};
//! use lance::{dataset::WriteParams, Dataset, dataset::optimize::compact_files};
//! // Remapping indices is ignored in this example.
//! use lance::dataset::optimize::IgnoreRemap;
//!
//! # let mut rt = Runtime::new().unwrap();
//! # rt.block_on(async {
//! #
//! # let test_dir = lance_core::utils::tempfile::TempStrDir::default();
//! # let uri = test_dir.to_string();
//! let schema = Arc::new(Schema::new(vec![Field::new("test", DataType::Int64, false)]));
//! let data = RecordBatch::try_new(
//!     schema.clone(),
//!     vec![Arc::new(Int64Array::from_iter_values(0..10_000))]
//! ).unwrap();
//! let reader = RecordBatchIterator::new(vec![Ok(data)], schema);
//!
//! // Write 100 small files
//! let write_params = WriteParams { max_rows_per_file: 100, ..Default::default()};
//! let mut dataset = Dataset::write(reader, &uri, Some(write_params)).await.unwrap();
//! assert_eq!(dataset.get_fragments().len(), 100);
//!
//! // Use compact_files() to consolidate the data to 1 fragment
//! let metrics = compact_files(&mut dataset, Default::default(), None).await.unwrap();
//! assert_eq!(metrics.fragments_removed, 100);
//! assert_eq!(metrics.fragments_added, 1);
//! assert_eq!(dataset.get_fragments().len(), 1);
//! # })
//! ```
//!
//! ## Distributed execution
//!
//! The [compact_files] method internally can use multiple threads, but
//! sometimes you might want to run it across multiple machines. To do this,
//! use the task API.
//!
//! ```text
//!                                      ┌──► CompactionTask.execute() ─► RewriteResult ─┐
//! plan_compaction() ─► CompactionPlan ─┼──► CompactionTask.execute() ─► RewriteResult ─┼─► commit_compaction()
//!                                      └──► CompactionTask.execute() ─► RewriteResult ─┘
//! ```
//!
//! [plan_compaction()] produces a [CompactionPlan]. This can be split into multiple
//! [CompactionTask], which can be serialized and sent to other machines. Calling
//! [CompactionTask::execute()] performs the compaction and returns a [RewriteResult].
//! The [RewriteResult] can be sent back to the coordinator, which can then call
//! [commit_compaction()] to commit the changes to the dataset.
//!
//! It's not required that all tasks are passed to [commit_compaction]. If some
//! didn't complete successfully or before a deadline, they can be omitted and
//! the successful tasks can be committed. You can also commit in batches if
//! you wish. As long as the tasks don't rewrite any of the same fragments,
//! they can be committed in any order.
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Cursor;
use std::ops::{AddAssign, Range};
use std::sync::Arc;

use super::fragment::FileFragment;
use super::index::DatasetIndexRemapperOptions;
use super::rowids::load_row_id_sequences;
use super::transaction::{
    Operation, RewriteGroup, RewrittenIndex, Transaction, TransactionBuilder,
};
use super::utils::make_rowid_capture_stream;
use super::{WriteMode, WriteParams, write_fragments_internal};
use crate::Dataset;
use crate::Result;
use crate::dataset::utils::CapturedRowIds;
use crate::index::DatasetIndexExt;
use crate::io::commit::{commit_transaction, migrate_fragments};
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::{StreamExt, TryStreamExt};
use lance_core::Error;
use lance_core::datatypes::BlobHandling;
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::utils::tracing::{DATASET_COMPACTING_EVENT, TRACE_DATASET_EVENTS};
use lance_index::frag_reuse::FragReuseGroup;
use lance_table::format::{Fragment, RowIdMeta};
use roaring::{RoaringBitmap, RoaringTreemap};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

mod binary_copy;
pub mod remapping;

use crate::index::frag_reuse::build_new_frag_reuse_index;
use crate::io::deletion::read_dataset_deletion_file;
use binary_copy::rewrite_files_binary_copy;
pub use remapping::{IgnoreRemap, IndexRemapper, IndexRemapperOptions, RemappedIndex};

/// Controls how data is rewritten during compaction.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CompactionMode {
    /// Decode and re-encode data (default).
    Reencode,
    /// Try binary copy if fragments are compatible, fall back to [`Reencode`](CompactionMode::Reencode) otherwise.
    TryBinaryCopy,
    /// Use binary copy or fail if fragments are not compatible.
    ForceBinaryCopy,
}

impl TryFrom<&str> for CompactionMode {
    type Error = Error;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "reencode" => Ok(Self::Reencode),
            "try_binary_copy" => Ok(Self::TryBinaryCopy),
            "force_binary_copy" => Ok(Self::ForceBinaryCopy),
            _ => Err(Error::invalid_input(format!(
                "Invalid compaction mode \"{}\". Valid values: \"reencode\", \"try_binary_copy\", \"force_binary_copy\"",
                value
            ))),
        }
    }
}

/// Options to be passed to [compact_files].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionOptions {
    /// Target number of rows per file. Defaults to 1 million.
    ///
    /// This is used to determine which fragments need compaction, as any
    /// fragments that have fewer rows than this value will be candidates for
    /// compaction.
    pub target_rows_per_fragment: usize,
    /// Max number of rows per group
    ///
    /// This does not affect which fragments need compaction, but does affect
    /// how they are re-written if selected.
    pub max_rows_per_group: usize,
    /// Max number of bytes per file
    ///
    /// This does not affect which frgamnets need compaction, but does affect
    /// how they are re-written if selected.
    ///
    /// If not specified then the default (see [`WriteParams`]) will be used.
    pub max_bytes_per_file: Option<usize>,
    /// Whether to compact fragments with deletions so there are no deletions.
    /// Defaults to true.
    pub materialize_deletions: bool,
    /// The fraction of rows that need to be deleted in a fragment before
    /// materializing the deletions. Defaults to 10% (0.1). Setting to zero (or
    /// lower) will materialize deletions for all fragments with deletions.
    /// Setting above 1.0 will never materialize deletions.
    pub materialize_deletions_threshold: f32,
    /// The number of threads to use (how many compaction tasks to run in parallel).
    /// Defaults to the number of compute-intensive CPUs.  Not used when running
    /// tasks manually using [`plan_compaction`]
    pub num_threads: Option<usize>,
    /// The batch size to use when scanning the input fragments.  If not
    /// specified then the default (see
    /// [`crate::dataset::Scanner::batch_size`]) will be used.
    pub batch_size: Option<usize>,
    /// Whether to defer remapping indices during compaction. If true, indices will
    /// not be remapped during this compaction operation. Instead, the fragment reuse index
    /// is updated and will be used to perform remapping later.
    pub defer_index_remap: bool,
    /// The compaction mode to use. When set, this takes priority over the
    /// deprecated `enable_binary_copy` and `enable_binary_copy_force` fields.
    ///
    /// Defaults to `None` (falls back to legacy boolean fields).
    pub compaction_mode: Option<CompactionMode>,
    /// Deprecated: use `compaction_mode` instead.
    #[deprecated(note = "Use `compaction_mode` instead")]
    pub enable_binary_copy: bool,
    /// Deprecated: use `compaction_mode` instead.
    #[deprecated(note = "Use `compaction_mode` instead")]
    pub enable_binary_copy_force: bool,
    /// The batch size in bytes for reading during binary copy operations.
    /// Controls how much data is read at once when performing binary copy.
    /// Defaults to 16MB (16 * 1024 * 1024).
    pub binary_copy_read_batch_bytes: Option<usize>,
    /// Maximum number of source fragments to compact in a single run. When set,
    /// the planner stops collecting fragment metrics once enough candidate
    /// fragments have been found. The final plan is strictly
    /// limited to at most this many source fragments across all tasks.
    /// If the first task alone exceeds this limit, no tasks are returned.
    /// Defaults to `None` (no limit, all eligible fragments are compacted).
    pub max_source_fragments: Option<usize>,
    /// Maximum total bytes of source data to compact in a single run. When set,
    /// the planner stops collecting fragment metrics once enough candidate
    /// data has been found. The final plan is strictly limited to
    /// at most this many bytes across all tasks. Fragments whose
    /// `file_size_bytes` is unknown are treated as 0 bytes for budget
    /// purposes. If the first task alone exceeds this limit, no tasks are
    /// returned.
    /// Defaults to `None` (no limit).
    pub max_compaction_bytes: Option<u64>,
    /// Transaction properties to store with this commit.
    ///
    /// These key-value pairs are stored in the transaction file
    /// and can be read later to identify the source of the commit
    /// (e.g., job_id for tracking completed compaction jobs).
    #[serde(skip)]
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
}

#[allow(deprecated)]
impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            // Matching defaults for WriteParams
            target_rows_per_fragment: 1024 * 1024,
            max_rows_per_group: 1024,
            materialize_deletions: true,
            materialize_deletions_threshold: 0.1,
            num_threads: None,
            max_bytes_per_file: None,
            batch_size: None,
            defer_index_remap: false,
            compaction_mode: None,
            enable_binary_copy: false,
            enable_binary_copy_force: false,
            binary_copy_read_batch_bytes: Some(16 * 1024 * 1024),
            max_source_fragments: None,
            max_compaction_bytes: None,
            transaction_properties: None,
        }
    }
}

/// Config key prefix for compaction options stored in the dataset manifest.
pub const COMPACTION_CONFIG_PREFIX: &str = "lance.compaction.";

#[allow(deprecated)]
impl CompactionOptions {
    /// Create [`CompactionOptions`] by starting with defaults and applying any
    /// overrides found in the dataset manifest config.
    ///
    /// Config keys are prefixed with `lance.compaction.` and map to fields:
    /// - `lance.compaction.target_rows_per_fragment`
    /// - `lance.compaction.max_rows_per_group`
    /// - `lance.compaction.max_bytes_per_file`
    /// - `lance.compaction.materialize_deletions`
    /// - `lance.compaction.materialize_deletions_threshold`
    /// - `lance.compaction.defer_index_remap`
    /// - `lance.compaction.batch_size`
    /// - `lance.compaction.compaction_mode`
    /// - `lance.compaction.binary_copy_read_batch_bytes`
    /// - `lance.compaction.max_source_fragments`
    /// - `lance.compaction.max_compaction_bytes`
    pub fn from_dataset_config(config: &HashMap<String, String>) -> Result<Self> {
        let mut opts = Self::default();
        opts.apply_dataset_config(config)?;
        Ok(opts)
    }

    /// Apply overrides from the dataset manifest config to this options struct.
    ///
    /// Only fields with corresponding config keys are modified; other fields
    /// retain their current values.
    pub fn apply_dataset_config(&mut self, config: &HashMap<String, String>) -> Result<()> {
        for (key, value) in config {
            let Some(field) = key.strip_prefix(COMPACTION_CONFIG_PREFIX) else {
                continue;
            };
            match field {
                "target_rows_per_fragment" => {
                    self.target_rows_per_fragment = value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?;
                }
                "max_rows_per_group" => {
                    self.max_rows_per_group = value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?;
                }
                "max_bytes_per_file" => {
                    self.max_bytes_per_file = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "materialize_deletions" => {
                    self.materialize_deletions = match value.to_lowercase().as_str() {
                        "true" => true,
                        "false" => false,
                        _ => {
                            return Err(Error::invalid_input(format!(
                                "Invalid value for {}: '{}' (expected 'true' or 'false')",
                                key, value
                            )));
                        }
                    };
                }
                "materialize_deletions_threshold" => {
                    self.materialize_deletions_threshold = value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a float between 0.0 and 1.0)",
                            key, value
                        ))
                    })?;
                }
                "defer_index_remap" => {
                    self.defer_index_remap = match value.to_lowercase().as_str() {
                        "true" => true,
                        "false" => false,
                        _ => {
                            return Err(Error::invalid_input(format!(
                                "Invalid value for {}: '{}' (expected 'true' or 'false')",
                                key, value
                            )));
                        }
                    };
                }
                "batch_size" => {
                    self.batch_size = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "compaction_mode" => {
                    self.compaction_mode = Some(CompactionMode::try_from(value.as_str())?);
                }
                "binary_copy_read_batch_bytes" => {
                    self.binary_copy_read_batch_bytes = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "max_source_fragments" => {
                    self.max_source_fragments = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "max_compaction_bytes" => {
                    self.max_compaction_bytes = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                _ => {
                    warn!("Ignoring unknown compaction config key: {}", key);
                }
            }
        }
        Ok(())
    }

    pub fn validate(&mut self) {
        // If threshold is 100%, same as turning off deletion materialization.
        if self.materialize_deletions && self.materialize_deletions_threshold >= 1.0 {
            self.materialize_deletions = false;
        }
    }

    /// Returns the effective [`CompactionMode`], preferring the new
    /// `compaction_mode` field and falling back to the deprecated boolean
    /// fields for backwards compatibility.
    pub fn compaction_mode(&self) -> CompactionMode {
        if let Some(mode) = self.compaction_mode {
            return mode;
        }
        // Fall back to deprecated booleans
        match (self.enable_binary_copy, self.enable_binary_copy_force) {
            (true, true) => CompactionMode::ForceBinaryCopy,
            (true, false) => CompactionMode::TryBinaryCopy,
            _ => CompactionMode::Reencode,
        }
    }

    /// Set transaction properties to store in the commit manifest.
    pub fn transaction_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.transaction_properties = Some(Arc::new(properties));
        self
    }
}

/// Determine if page-level binary copy can safely merge the provided fragments.
///
/// Preconditions checked in order:
/// - Compaction mode is not `Reencode`
/// - Dataset storage format is non-legacy
/// - Fragment list is non-empty
/// - All data files share identical Lance file versions
/// - No fragment has a deletion file
///   TODO: Need to support schema evolution case like add column and drop column
/// - All data files share identical schema mappings (`fields`, `column_indices`)
/// - Input data files must not contain extra global buffers (beyond schema / file descriptor)
async fn can_use_binary_copy(
    dataset: &Dataset,
    options: &CompactionOptions,
    fragments: &[Fragment],
) -> bool {
    can_use_binary_copy_impl(dataset, options, fragments)
        .await
        .unwrap_or_else(|err| {
            log::warn!("Binary copy disabled due to error: {}", err);
            false
        })
}

async fn can_use_binary_copy_impl(
    dataset: &Dataset,
    options: &CompactionOptions,
    fragments: &[Fragment],
) -> Result<bool> {
    use lance_file::reader::FileReader as LFReader;
    use lance_file::version::LanceFileVersion;
    use lance_io::scheduler::{ScanScheduler, SchedulerConfig};

    if matches!(options.compaction_mode(), CompactionMode::Reencode) {
        log::debug!("Binary copy disabled: compaction mode is Reencode");
        return Ok(false);
    }

    let has_blob_columns = dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob());
    if has_blob_columns {
        log::debug!("Binary copy disabled: dataset contains blob columns");
        return Ok(false);
    }

    let storage_ok = dataset
        .manifest
        .data_storage_format
        .lance_file_version()
        .map(|v| !matches!(v.resolve(), LanceFileVersion::Legacy))
        .unwrap_or(false);
    if !storage_ok {
        log::debug!("Binary copy disabled: dataset uses legacy storage format");
        return Ok(false);
    }

    if fragments.is_empty() {
        log::debug!("Binary copy disabled: no fragments to compact");
        return Ok(false);
    }

    let storage_file_version = dataset
        .manifest
        .data_storage_format
        .lance_file_version()?
        .resolve();

    if fragments[0].files.is_empty() {
        log::debug!(
            "Binary copy disabled: fragment {} has no data files",
            fragments[0].id
        );
        return Ok(false);
    }
    let ref_fields = &fragments[0].files[0].fields;
    let ref_cols = &fragments[0].files[0].column_indices;
    let mut is_same_version = true;

    for fragment in fragments {
        if fragment.deletion_file.is_some() {
            log::debug!(
                "Binary copy disabled: fragment {} has a deletion file",
                fragment.id
            );
            return Ok(false);
        }

        for data_file in &fragment.files {
            let version_ok = LanceFileVersion::try_from_major_minor(
                data_file.file_major_version,
                data_file.file_minor_version,
            )
            .map(|v| v.resolve())
            .is_ok_and(|v| v == storage_file_version);

            if !version_ok {
                is_same_version = false;
            }
            if data_file.fields != *ref_fields || data_file.column_indices != *ref_cols {
                return Ok(false);
            }

            // check file global buffer
            let object_store = match data_file.base_id {
                Some(base_id) => dataset.object_store(Some(base_id)).await?,
                None => dataset.object_store.clone(),
            };
            let full_path = dataset
                .data_file_dir(data_file)?
                .clone()
                .join(data_file.path.as_str());
            let scan_scheduler = ScanScheduler::new(
                object_store.clone(),
                SchedulerConfig::max_bandwidth(&object_store),
            );
            let file_scheduler = scan_scheduler
                .open_file_with_priority(&full_path, 0, &data_file.file_size_bytes)
                .await?;
            let file_meta = LFReader::read_all_metadata(&file_scheduler).await?;
            // Binary copy only preserves page and column-buffer bytes. The output file's footer
            // (including global buffers) is re-generated, not copied from inputs.
            //
            // Therefore, we reject input files that contain any additional global buffers beyond
            // the required schema / file descriptor global buffer (global buffer index 0).
            if file_meta.file_buffers.len() > 1 {
                log::debug!(
                    "Binary copy disabled: data file has extra global buffers (len={})",
                    file_meta.file_buffers.len()
                );
                return Ok(false);
            }
        }
    }

    if !is_same_version {
        log::debug!("Binary copy disabled: data files use different file versions");
        return Ok(false);
    }

    Ok(true)
}

/// Metrics returned by [compact_files].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionMetrics {
    /// The number of fragments that have been overwritten.
    pub fragments_removed: usize,
    /// The number of new fragments that have been added.
    pub fragments_added: usize,
    /// The number of files that have been removed, including deletion files.
    pub files_removed: usize,
    /// The number of files that have been added, which is always equal to the
    /// number of fragments.
    pub files_added: usize,
}

impl AddAssign for CompactionMetrics {
    fn add_assign(&mut self, rhs: Self) {
        self.fragments_removed += rhs.fragments_removed;
        self.fragments_added += rhs.fragments_added;
        self.files_removed += rhs.files_removed;
        self.files_added += rhs.files_added;
    }
}

/// Trait for implementing custom compaction planning strategies.
///
/// This trait allows users to define their own compaction strategies by implementing
/// the `plan` method. The default implementation is provided by [`DefaultCompactionPlanner`].
#[async_trait::async_trait]
pub trait CompactionPlanner: Send + Sync {
    /// Build compaction plan.
    ///
    /// This method analyzes the dataset's fragments and generates a [`CompactionPlan`]
    /// containing a list of compaction tasks to execute.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Reference to the dataset to be compacted
    async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan>;
}

/// Formulate a plan to compact the files in a dataset
///
/// The compaction plan will contain a list of tasks to execute. Each task
/// will contain approximately `target_rows_per_fragment` rows and will be
/// rewriting fragments that are adjacent in the dataset's fragment list. Some
/// tasks may contain a single fragment when that fragment has deletions that
/// are being materialized and doesn't have any neighbors that need to be
/// compacted.
#[derive(Debug, Clone, Default)]
pub struct DefaultCompactionPlanner {
    options: CompactionOptions,
}

impl DefaultCompactionPlanner {
    pub fn new(mut options: CompactionOptions) -> Self {
        options.validate();
        Self { options }
    }

    fn exceeds_budget(&self, candidate_fragments: usize, candidate_bytes: u64) -> bool {
        if let Some(max_frags) = self.options.max_source_fragments
            && candidate_fragments > max_frags
        {
            return true;
        }
        if let Some(max_bytes) = self.options.max_compaction_bytes
            && candidate_bytes > max_bytes
        {
            return true;
        }
        false
    }

    fn finalize_bin(
        &self,
        completed_bin: CandidateBin,
        effective_candidate_fragments: &mut usize,
        effective_candidate_bytes: &mut u64,
        candidate_bins: &mut Vec<CandidateBin>,
    ) -> bool {
        if !completed_bin.is_noop() {
            *effective_candidate_fragments += completed_bin.fragments.len();
            *effective_candidate_bytes += completed_bin.known_bytes;
        }
        candidate_bins.push(completed_bin);
        self.exceeds_budget(*effective_candidate_fragments, *effective_candidate_bytes)
    }

    fn apply_budget_limits(&self, tasks: Vec<TaskData>) -> Vec<TaskData> {
        let mut total_frags = 0usize;
        let mut total_bytes = 0u64;
        let mut result = Vec::new();
        for task in tasks {
            let task_frags = task.fragments.len();
            let task_bytes: u64 = task
                .fragments
                .iter()
                .flat_map(|f| f.files.iter())
                .filter_map(|f| f.file_size_bytes.get())
                .map(|sz| sz.get())
                .sum();
            if let Some(max_frags) = self.options.max_source_fragments
                && total_frags + task_frags > max_frags
            {
                break;
            }
            if let Some(max_bytes) = self.options.max_compaction_bytes
                && total_bytes + task_bytes > max_bytes
            {
                break;
            }
            total_frags += task_frags;
            total_bytes += task_bytes;
            result.push(task);
        }
        result
    }
}

#[async_trait::async_trait]
impl CompactionPlanner for DefaultCompactionPlanner {
    async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan> {
        let fragments = dataset.get_fragments();

        debug_assert!(
            fragments.windows(2).all(|w| w[0].id() < w[1].id()),
            "fragments in manifest are not sorted"
        );
        let mut fragment_metrics = futures::stream::iter(fragments)
            .map(|fragment| async move {
                match collect_metrics(&fragment).await {
                    Ok(metrics) => Ok((fragment.metadata, metrics)),
                    Err(e) => Err(e),
                }
            })
            .buffered(dataset.object_store.as_ref().io_parallelism());

        let index_fragmaps = load_index_fragmaps(dataset).await?;
        let indices_containing_frag = |frag_id: u32| {
            index_fragmaps
                .iter()
                .enumerate()
                .filter(|(_, bitmap)| bitmap.contains(frag_id))
                .map(|(pos, _)| pos)
                .collect::<Vec<_>>()
        };

        let mut candidate_bins: Vec<CandidateBin> = Vec::new();
        let mut current_bin: Option<CandidateBin> = None;
        let mut effective_candidate_fragments: usize = 0;
        let mut effective_candidate_bytes: u64 = 0;
        let mut i = 0;

        let has_budget = self.options.max_source_fragments.is_some()
            || self.options.max_compaction_bytes.is_some();

        while let Some(res) = fragment_metrics.next().await {
            let (fragment, metrics) = res?;

            let candidacy = if self.options.materialize_deletions
                && metrics.deletion_percentage() > self.options.materialize_deletions_threshold
            {
                Some(CompactionCandidacy::CompactItself)
            } else if metrics.physical_rows < self.options.target_rows_per_fragment {
                Some(CompactionCandidacy::CompactWithNeighbors)
            } else {
                None
            };

            let indices = indices_containing_frag(fragment.id as u32);

            match (candidacy, &mut current_bin) {
                (None, None) => {}
                (Some(candidacy), None) => {
                    let known_bytes = CandidateBin::fragment_known_bytes(&fragment);
                    current_bin = Some(CandidateBin {
                        fragments: vec![fragment],
                        pos_range: i..(i + 1),
                        candidacy: vec![candidacy],
                        row_counts: vec![metrics.num_rows()],
                        indices,
                        known_bytes,
                    });
                }
                (Some(candidacy), Some(bin)) => {
                    if bin.indices == indices {
                        let new_fragment_bytes = CandidateBin::fragment_known_bytes(&fragment);
                        bin.fragments.push(fragment);
                        bin.pos_range.end += 1;
                        bin.candidacy.push(candidacy);
                        bin.row_counts.push(metrics.num_rows());
                        bin.known_bytes += new_fragment_bytes;
                        if has_budget && !bin.is_noop() {
                            let running_fragments =
                                effective_candidate_fragments + bin.fragments.len();
                            let running_bytes = effective_candidate_bytes + new_fragment_bytes;
                            if self.exceeds_budget(running_fragments, running_bytes) {
                                bin.fragments.pop();
                                bin.pos_range.end -= 1;
                                bin.candidacy.pop();
                                bin.row_counts.pop();
                                bin.known_bytes -= new_fragment_bytes;
                                candidate_bins.push(current_bin.take().unwrap());
                                break;
                            }
                        }
                    } else {
                        if let Some(completed_bin) = current_bin.take()
                            && self.finalize_bin(
                                completed_bin,
                                &mut effective_candidate_fragments,
                                &mut effective_candidate_bytes,
                                &mut candidate_bins,
                            )
                        {
                            break;
                        }
                        let known_bytes = CandidateBin::fragment_known_bytes(&fragment);
                        current_bin = Some(CandidateBin {
                            fragments: vec![fragment],
                            pos_range: i..(i + 1),
                            candidacy: vec![candidacy],
                            row_counts: vec![metrics.num_rows()],
                            indices,
                            known_bytes,
                        });
                    }
                }
                (None, Some(_)) => {
                    if let Some(completed_bin) = current_bin.take()
                        && self.finalize_bin(
                            completed_bin,
                            &mut effective_candidate_fragments,
                            &mut effective_candidate_bytes,
                            &mut candidate_bins,
                        )
                    {
                        break;
                    }
                }
            }

            i += 1;
        }

        if let Some(bin) = current_bin {
            candidate_bins.push(bin);
        }

        let all_tasks: Vec<TaskData> = candidate_bins
            .into_iter()
            .filter(|bin| !bin.is_noop())
            .flat_map(|bin| bin.split_for_size(self.options.target_rows_per_fragment))
            .map(|bin| TaskData {
                fragments: bin.fragments,
            })
            .collect();

        let tasks = self.apply_budget_limits(all_tasks);

        let mut compaction_plan =
            CompactionPlan::new(dataset.manifest.version, self.options.clone());
        compaction_plan.extend_tasks(tasks);

        Ok(compaction_plan)
    }
}

/// Compacts the files in the dataset without reordering them.
///
/// By default, this does a few things:
///  * Removes deleted rows from fragments.
///  * Removes dropped columns from fragments.
///  * Merges fragments that are too small.
///
/// This method tries to preserve the insertion order of rows in the dataset.
///
/// If no compaction is needed, this method will not make a new version of the table.
pub async fn compact_files(
    dataset: &mut Dataset,
    options: CompactionOptions,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>, // These will be deprecated later
) -> Result<CompactionMetrics> {
    info!(target: TRACE_DATASET_EVENTS, event=DATASET_COMPACTING_EVENT, uri = &dataset.uri);
    let planner = DefaultCompactionPlanner::new(options);
    compact_files_with_planner(dataset, remap_options, &planner).await
}

pub async fn compact_files_with_planner(
    dataset: &mut Dataset,
    remap_options: Option<Arc<dyn IndexRemapperOptions>>, // These will be deprecated later
    planner: &dyn CompactionPlanner,
) -> Result<CompactionMetrics> {
    let compaction_plan: CompactionPlan = planner.plan(dataset).await?;

    // If nothing to compact, don't make a commit.
    if compaction_plan.tasks().is_empty() {
        return Ok(CompactionMetrics::default());
    }

    let dataset_ref = &dataset.clone();

    let result_stream = futures::stream::iter(compaction_plan.tasks.into_iter())
        .map(|task| rewrite_files(Cow::Borrowed(dataset_ref), task, &compaction_plan.options))
        .buffer_unordered(
            compaction_plan
                .options
                .num_threads
                .unwrap_or_else(get_num_compute_intensive_cpus),
        );

    let completed_tasks: Vec<RewriteResult> = result_stream.try_collect().await?;
    let remap_options = remap_options.unwrap_or(Arc::new(DatasetIndexRemapperOptions::default()));
    let metrics = commit_compaction(
        dataset,
        completed_tasks,
        remap_options,
        &compaction_plan.options,
    )
    .await?;

    Ok(metrics)
}

/// Information about a fragment used to decide its fate in compaction
#[derive(Debug)]
struct FragmentMetrics {
    /// The number of original rows in the fragment
    pub physical_rows: usize,
    /// The number of rows that have been deleted
    pub num_deletions: usize,
}

impl FragmentMetrics {
    /// The fraction of rows that have been deleted
    fn deletion_percentage(&self) -> f32 {
        if self.physical_rows > 0 {
            self.num_deletions as f32 / self.physical_rows as f32
        } else {
            0.0
        }
    }

    /// The number of rows that are still in the fragment
    fn num_rows(&self) -> usize {
        self.physical_rows - self.num_deletions
    }
}

async fn collect_metrics(fragment: &FileFragment) -> Result<FragmentMetrics> {
    let physical_rows = fragment.physical_rows();
    let num_deletions = fragment.count_deletions();
    let (physical_rows, num_deletions) =
        futures::future::try_join(physical_rows, num_deletions).await?;
    Ok(FragmentMetrics {
        physical_rows,
        num_deletions,
    })
}

/// A plan for what groups of fragments to compact.
///
/// See [plan_compaction()] for more details.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompactionPlan {
    pub tasks: Vec<TaskData>,
    pub read_version: u64,
    pub options: CompactionOptions,
}

impl CompactionPlan {
    /// Retrieve standalone tasks that be be executed in a distributed fashion.
    pub fn compaction_tasks(&self) -> impl Iterator<Item = CompactionTask> + '_ {
        let read_version = self.read_version;
        let options = self.options.clone();
        self.tasks.iter().map(move |task| CompactionTask {
            task: task.clone(),
            read_version,
            options: options.clone(),
        })
    }

    /// The number of tasks in the plan.
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// The version of the dataset that was read to produce this plan.
    pub fn read_version(&self) -> u64 {
        self.read_version
    }

    /// The options used to produce this plan.
    pub fn options(&self) -> &CompactionOptions {
        &self.options
    }
}

/// Build a scan reader for rewrite and optionally capture row IDs.
///
/// Parameters:
/// - `dataset`: Dataset handle used to create the scanner.
/// - `fragments`: When `with_frags` is true, restrict the scan to these old fragments
///   and preserve insertion order.
/// - `batch_size`: Optional batch size; if provided, set it on the scanner to control
///   read batching.
/// - `with_frags`: Whether to scan only the specified old fragments and force
///   in-order reading.
/// - `capture_row_ids`: When index remapping is needed, include and capture the
///   `_rowid` column from the stream.
///
/// Returns:
/// - `SendableRecordBatchStream`: The batch stream (with `_rowid` removed if captured)
///   to feed the rewrite path.
/// - `Option<Receiver<CapturedRowIds>>`: A receiver to obtain captured row IDs after the
///   stream completes; `None` if not capturing.
async fn prepare_reader(
    dataset: &Dataset,
    fragments: &[Fragment],
    batch_size: Option<usize>,
    with_frags: bool,
    capture_row_ids: bool,
) -> Result<(
    SendableRecordBatchStream,
    Option<std::sync::mpsc::Receiver<CapturedRowIds>>,
)> {
    let mut scanner = dataset.scan();
    let has_blob_columns = dataset
        .schema()
        .fields_pre_order()
        .any(|field| field.is_blob());
    if has_blob_columns {
        scanner.blob_handling(BlobHandling::AllBinary);
    }
    if let Some(bs) = batch_size {
        scanner.batch_size(bs);
    }
    if with_frags {
        scanner
            .with_fragments(fragments.to_vec())
            .scan_in_order(true);
    }
    if capture_row_ids {
        scanner.with_row_id();
        let data = SendableRecordBatchStream::from(scanner.try_into_stream().await?);
        let (data_no_row_ids, rx) =
            make_rowid_capture_stream(data, dataset.manifest.uses_stable_row_ids())?;
        Ok((data_no_row_ids, Some(rx)))
    } else {
        Ok((
            SendableRecordBatchStream::from(scanner.try_into_stream().await?),
            None,
        ))
    }
}

/// A single group of fragments to compact, which is a view into the compaction
/// plan. We keep the `replace_range` indices so we can map the result of the
/// compact back to the fragments it replaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskData {
    /// The fragments to compact.
    pub fragments: Vec<Fragment>,
}

/// A standalone task that can be serialized and sent to another machine for
/// execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionTask {
    pub task: TaskData,
    pub read_version: u64,
    pub options: CompactionOptions,
}

impl CompactionTask {
    /// Run the compaction task and return the result.
    ///
    /// This result should be later passed to [commit_compaction()] to commit
    /// the changes to the dataset.
    ///
    /// Note: you should pass the version of the dataset that is the same as
    /// the read version for this task (the same version from which the
    /// compaction was planned).
    pub async fn execute(&self, dataset: &Dataset) -> Result<RewriteResult> {
        let dataset = if dataset.manifest.version == self.read_version {
            Cow::Borrowed(dataset)
        } else {
            Cow::Owned(dataset.checkout_version(self.read_version).await?)
        };
        rewrite_files(dataset, self.task.clone(), &self.options).await
    }
}

impl CompactionPlan {
    fn new(read_version: u64, options: CompactionOptions) -> Self {
        Self {
            tasks: Vec::new(),
            read_version,
            options,
        }
    }

    fn extend_tasks(&mut self, tasks: impl IntoIterator<Item = TaskData>) {
        self.tasks.extend(tasks);
    }

    fn tasks(&self) -> &[TaskData] {
        &self.tasks
    }
}

#[derive(Debug, Clone)]
enum CompactionCandidacy {
    /// Compact the fragment if it has neighbors that are also candidates
    CompactWithNeighbors,
    /// Compact the fragment regardless.
    CompactItself,
}

/// Internal struct used for planning compaction.
struct CandidateBin {
    pub fragments: Vec<Fragment>,
    pub pos_range: Range<usize>,
    pub candidacy: Vec<CompactionCandidacy>,
    pub row_counts: Vec<usize>,
    pub indices: Vec<usize>,
    pub known_bytes: u64,
}

impl CandidateBin {
    fn fragment_known_bytes(fragment: &Fragment) -> u64 {
        fragment
            .files
            .iter()
            .filter_map(|f| f.file_size_bytes.get())
            .map(|sz| sz.get())
            .sum()
    }

    fn is_noop(&self) -> bool {
        if self.fragments.is_empty() {
            return true;
        }
        if self.fragments.len() == 1 {
            matches!(self.candidacy[0], CompactionCandidacy::CompactWithNeighbors)
        } else {
            false
        }
    }

    fn split_for_size(mut self, min_num_rows: usize) -> Vec<Self> {
        let mut bins = Vec::new();

        loop {
            let mut bin_len = 0;
            let mut bin_row_count = 0;
            while bin_row_count < min_num_rows && bin_len < self.row_counts.len() {
                bin_row_count += self.row_counts[bin_len];
                bin_len += 1;
            }

            if self.row_counts[bin_len..].iter().sum::<usize>() >= min_num_rows {
                bins.push(Self {
                    fragments: self.fragments.drain(0..bin_len).collect(),
                    pos_range: self.pos_range.start..(self.pos_range.start + bin_len),
                    candidacy: self.candidacy.drain(0..bin_len).collect(),
                    row_counts: self.row_counts.drain(0..bin_len).collect(),
                    indices: Vec::new(),
                    known_bytes: 0,
                });
                self.pos_range.start += bin_len;
            } else {
                bins.push(self);
                break;
            }
        }

        bins
    }
}

async fn load_index_fragmaps(dataset: &Dataset) -> Result<Vec<RoaringBitmap>> {
    let indices = dataset.load_indices().await?;
    let mut index_fragmaps = Vec::with_capacity(indices.len());
    for index in indices.iter() {
        if let Some(fragment_bitmap) = index.fragment_bitmap.as_ref() {
            index_fragmaps.push(fragment_bitmap.clone());
        } else {
            let dataset_at_index = dataset.checkout_version(index.dataset_version).await?;
            let frags = 0..dataset_at_index.manifest.max_fragment_id.unwrap_or(0);
            index_fragmaps.push(RoaringBitmap::from_sorted_iter(frags).unwrap());
        }
    }
    Ok(index_fragmaps)
}

pub async fn plan_compaction(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Result<CompactionPlan> {
    let planner = DefaultCompactionPlanner::new(options.clone());
    planner.plan(dataset).await
}

/// The result of a single compaction task.
///
/// This should be passed to [commit_compaction()] to commit the operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RewriteResult {
    pub metrics: CompactionMetrics,
    pub new_fragments: Vec<Fragment>,
    /// The version of the dataset that was read to perform this compaction.
    pub read_version: u64,
    /// The original fragments being replaced
    pub original_fragments: Vec<Fragment>,
    /// Serialized `RoaringTreemap` of the row addresses from the original
    /// fragments that were read during compaction.
    ///
    /// - `None` when configured with stable row IDs because the row ID
    ///   sequences are rechunked directly.
    /// - `Some` then these addresses are either (1) written to storage for
    ///   deferred index remap post-processing, or (2) used with reserved
    ///   fragment IDs to build old-to-new mappings.
    pub row_addrs: Option<Vec<u8>>,
}

async fn reserve_fragment_ids(
    dataset: &Dataset,
    fragments: impl ExactSizeIterator<Item = &mut Fragment>,
) -> Result<()> {
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::ReserveFragments {
            num_fragments: fragments.len() as u32,
        },
        None,
    );

    let (manifest, _) = commit_transaction(
        dataset,
        dataset.object_store.as_ref(),
        dataset.commit_handler.as_ref(),
        &transaction,
        &Default::default(),
        &Default::default(),
        dataset.manifest_location.naming_scheme,
        None,
    )
    .await?;

    // Need +1 since max_fragment_id is inclusive in this case and ranges are exclusive
    let new_max_exclusive = manifest.max_fragment_id.unwrap_or(0) + 1;
    let reserved_ids = (new_max_exclusive - fragments.len() as u32)..(new_max_exclusive);

    for (fragment, new_id) in fragments.zip(reserved_ids) {
        fragment.id = new_id as u64;
    }

    Ok(())
}

/// Rewrite the files in a single task.
///
/// This assumes that the dataset is the correct read version to be compacted.
async fn rewrite_files(
    dataset: Cow<'_, Dataset>,
    task: TaskData,
    options: &CompactionOptions,
) -> Result<RewriteResult> {
    let mut metrics = CompactionMetrics::default();

    if task.fragments.is_empty() {
        return Ok(RewriteResult {
            metrics,
            new_fragments: Vec::new(),
            read_version: dataset.manifest.version,
            original_fragments: task.fragments,
            row_addrs: None,
        });
    }

    let previous_writer_version = &dataset.manifest.writer_version;
    // The versions of Lance prior to when we started writing the writer version
    // sometimes wrote incorrect `Fragment.physical_rows` values, so we should
    // make sure to recompute them.
    // See: https://github.com/lance-format/lance/issues/1531
    let recompute_stats = previous_writer_version.is_none();

    // It's possible the fragments are old and don't have physical rows or
    // num deletions recorded. If that's the case, we need to grab and set that
    // information.
    let fragments = migrate_fragments(dataset.as_ref(), &task.fragments, recompute_stats).await?;
    let num_rows = fragments
        .iter()
        .map(|f| f.physical_rows.unwrap() as u64)
        .sum::<u64>();
    // If we aren't using stable row ids, then we need to remap indices.
    let needs_remapping = !dataset.manifest.uses_stable_row_ids();
    let mut new_fragments: Vec<Fragment>;
    let task_id = uuid::Uuid::new_v4();
    log::info!(
        "Compaction task {}: Begin compacting {} rows across {} fragments",
        task_id,
        num_rows,
        fragments.len()
    );
    let mode = options.compaction_mode();
    let can_binary_copy = can_use_binary_copy(dataset.as_ref(), options, &fragments).await;
    if !can_binary_copy && matches!(mode, CompactionMode::ForceBinaryCopy) {
        return Err(Error::not_supported_source(
            format!("compaction task {}: binary copy is not supported", task_id).into(),
        ));
    }
    let mut row_ids_rx: Option<std::sync::mpsc::Receiver<CapturedRowIds>> = None;
    let mut reader: Option<SendableRecordBatchStream> = None;

    if !can_binary_copy {
        let (prepared_reader, rx_initial) = prepare_reader(
            dataset.as_ref(),
            &fragments,
            options.batch_size,
            true,
            needs_remapping,
        )
        .await?;
        row_ids_rx = rx_initial;

        let mut rows_read = 0;
        let schema = prepared_reader.schema();
        let reader_with_progress = prepared_reader.inspect_ok(move |batch| {
            rows_read += batch.num_rows();
            log::info!(
                "Compaction task {}: Read progress {}/{}",
                task_id,
                rows_read,
                num_rows,
            );
        });
        reader = Some(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            reader_with_progress,
        )));
    }

    let mut params = WriteParams {
        max_rows_per_file: options.target_rows_per_fragment,
        max_rows_per_group: options.max_rows_per_group,
        mode: WriteMode::Append,
        ..Default::default()
    };
    if let Some(max_bytes_per_file) = options.max_bytes_per_file {
        params.max_bytes_per_file = max_bytes_per_file;
    }

    if dataset.manifest.uses_stable_row_ids() {
        params.enable_stable_row_ids = true;
    }

    if can_binary_copy {
        new_fragments = rewrite_files_binary_copy(
            dataset.as_ref(),
            &fragments,
            &params,
            options.binary_copy_read_batch_bytes,
        )
        .await?;

        if new_fragments.is_empty() && matches!(mode, CompactionMode::ForceBinaryCopy) {
            return Err(Error::not_supported_source(
                format!("compaction task {}: binary copy is not supported", task_id).into(),
            ));
        }

        if needs_remapping {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut addrs = RoaringTreemap::new();
            for frag in &fragments {
                let frag_id = frag.id as u32;
                let count = u64::try_from(frag.physical_rows.unwrap_or(0)).map_err(|_| {
                    Error::internal(format!(
                        "Fragment {} has too many physical rows to represent as row addresses",
                        frag.id
                    ))
                })?;
                let start = u64::from(lance_core::utils::address::RowAddress::first_row(frag_id));
                addrs.insert_range(start..start + count);
            }
            let captured = CapturedRowIds::AddressStyle(addrs);
            let _ = tx.send(captured);
            row_ids_rx = Some(rx);
        }
    } else {
        let (frags, _) = write_fragments_internal(
            Some(dataset.as_ref()),
            dataset.object_store.clone(),
            &dataset.base,
            dataset.schema().clone(),
            reader.expect("reader must be prepared for non-binary-copy path"),
            params,
            None,
        )
        .await?;
        new_fragments = frags;
    }

    log::info!("Compaction task {}: file written", task_id);

    let row_addrs = if let Some(row_ids_rx) = row_ids_rx {
        let captured_ids = row_ids_rx
            .try_recv()
            .map_err(|err| Error::internal(format!("Failed to receive row ids: {}", err)))?;
        let row_addrs = captured_ids.row_addrs(None).into_owned();
        let mut serialized = Vec::with_capacity(row_addrs.serialized_size());
        row_addrs.serialize_into(&mut serialized)?;
        Some(serialized)
    } else {
        if dataset.manifest.uses_stable_row_ids() {
            log::info!("Compaction task {}: rechunking stable row ids", task_id);
            rechunk_stable_row_ids(dataset.as_ref(), &mut new_fragments, &fragments).await?;
            recalc_versions_for_rewritten_fragments(
                dataset.as_ref(),
                &mut new_fragments,
                &fragments,
            )
            .await?;
        }
        None
    };

    metrics.files_removed = task
        .fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();
    metrics.fragments_removed = task.fragments.len();
    metrics.fragments_added = new_fragments.len();
    metrics.files_added = new_fragments
        .iter()
        .map(|f| f.files.len() + f.deletion_file.is_some() as usize)
        .sum();

    log::info!("Compaction task {}: completed", task_id);

    Ok(RewriteResult {
        metrics,
        new_fragments,
        read_version: dataset.manifest.version,
        original_fragments: fragments,
        row_addrs,
    })
}

async fn rechunk_stable_row_ids(
    dataset: &Dataset,
    new_fragments: &mut [Fragment],
    old_fragments: &[Fragment],
) -> Result<()> {
    let mut old_sequences = load_row_id_sequences(dataset, old_fragments)
        .try_collect::<Vec<_>>()
        .await?;
    // Should sort them back into original order.
    old_sequences.sort_by_key(|(frag_id, _)| {
        old_fragments
            .iter()
            .position(|frag| frag.id as u32 == *frag_id)
            .expect("Fragment not found")
    });

    // Need to remove deleted rows
    futures::stream::iter(old_sequences.iter_mut().zip(old_fragments.iter()))
        .map(Ok)
        .try_for_each(|((_, seq), frag)| async move {
            if let Some(deletion_file) = &frag.deletion_file {
                let deletions = read_dataset_deletion_file(dataset, frag.id, deletion_file).await?;

                let mut new_seq = seq.as_ref().clone();
                new_seq.mask(deletions.to_sorted_iter())?;
                *seq = Arc::new(new_seq);
            }
            Ok::<(), crate::Error>(())
        })
        .await?;

    debug_assert_eq!(
        { old_sequences.iter().map(|(_, seq)| seq.len()).sum::<u64>() },
        {
            new_fragments
                .iter()
                .map(|frag| frag.physical_rows.unwrap() as u64)
                .sum::<u64>()
        },
        "{:?}",
        old_sequences
    );

    let new_sequences = lance_table::rowids::rechunk_sequences(
        old_sequences
            .into_iter()
            .map(|(_, seq)| seq.as_ref().clone()),
        new_fragments
            .iter()
            .map(|frag| frag.physical_rows.unwrap() as u64),
        false,
    )?;

    for (fragment, sequence) in new_fragments.iter_mut().zip(new_sequences) {
        // TODO: if large enough, serialize to separate file
        let serialized = lance_table::rowids::write_row_ids(&sequence);
        fragment.row_id_meta = Some(RowIdMeta::Inline(serialized));
    }

    Ok(())
}

/// After row id rechunking, preserve per-row latest update versions by masking deletions and rechunking
async fn recalc_versions_for_rewritten_fragments(
    dataset: &Dataset,
    new_fragments: &mut [Fragment],
    old_fragments: &[Fragment],
) -> Result<()> {
    // Load old per-row last_updated_at version sequences
    let mut old_last_updated_sequences: Vec<lance_table::format::RowDatasetVersionSequence> =
        Vec::with_capacity(old_fragments.len());
    // Load old per-row created_at version sequences
    let mut old_created_at_sequences: Vec<lance_table::format::RowDatasetVersionSequence> =
        Vec::with_capacity(old_fragments.len());

    for frag in old_fragments.iter() {
        let row_count = if let Some(row_id_meta) = &frag.row_id_meta {
            match row_id_meta {
                RowIdMeta::Inline(data) => lance_table::rowids::read_row_ids(data)?.len(),
                RowIdMeta::External(_file) => frag.physical_rows.unwrap_or(0) as u64,
            }
        } else {
            frag.physical_rows.unwrap_or(0) as u64
        };

        // Load created_at sequence (default to version 1 if missing)
        let mut created_at_seq = if let Some(version_meta) = &frag.created_at_version_meta {
            version_meta.load_sequence().map_err(|e| {
                Error::internal(format!("Failed to load created_at version sequence: {}", e))
            })?
        } else {
            // Default: treat all rows as created at version 1
            lance_table::format::RowDatasetVersionSequence::from_uniform_row_count(row_count, 1)
        };

        // Load last_updated_at sequence (default to same as created_at sequence)
        let mut last_updated_seq = if let Some(version_meta) = &frag.last_updated_at_version_meta {
            version_meta.load_sequence().map_err(|e| {
                Error::internal(format!(
                    "Failed to load last_updated_at version sequence: {}",
                    e
                ))
            })?
        } else {
            created_at_seq.clone()
        };

        // Apply deletion mask if present (positions are local offsets)
        if let Some(deletion_file) = &frag.deletion_file {
            let deletions = read_dataset_deletion_file(dataset, frag.id, deletion_file).await?;
            last_updated_seq.mask(deletions.to_sorted_iter())?;
            created_at_seq.mask(deletions.to_sorted_iter())?;
        }

        old_last_updated_sequences.push(last_updated_seq);
        old_created_at_sequences.push(created_at_seq);
    }

    // Ensure row counts match new fragments total
    let old_total: u64 = old_last_updated_sequences.iter().map(|s| s.len()).sum();
    let new_total: u64 = new_fragments
        .iter()
        .map(|f| f.physical_rows.unwrap_or(0) as u64)
        .sum();
    debug_assert_eq!(old_total, new_total);

    // Rechunk version runs aligned to new fragment sizes
    let chunk_sizes: Vec<u64> = new_fragments
        .iter()
        .map(|f| f.physical_rows.unwrap_or(0) as u64)
        .collect();

    let new_last_updated_sequences = lance_table::rowids::version::rechunk_version_sequences(
        old_last_updated_sequences,
        chunk_sizes.clone(),
        false,
    )?;

    let new_created_at_sequences = lance_table::rowids::version::rechunk_version_sequences(
        old_created_at_sequences,
        chunk_sizes,
        false,
    )?;

    // Set both version metadata on new fragments
    for ((fragment, last_updated_seq), created_at_seq) in new_fragments
        .iter_mut()
        .zip(new_last_updated_sequences.into_iter())
        .zip(new_created_at_sequences.into_iter())
    {
        fragment.last_updated_at_version_meta = Some(
            lance_table::format::RowDatasetVersionMeta::from_sequence(&last_updated_seq).unwrap(),
        );
        fragment.created_at_version_meta = Some(
            lance_table::format::RowDatasetVersionMeta::from_sequence(&created_at_seq).unwrap(),
        );
    }

    Ok(())
}

/// Commit the results of file compaction.
///
/// It is not required that all tasks are passed to this method. If some failed,
/// they can be omitted and the successful tasks can be committed. However, once
/// some of the tasks have been committed, the remainder of the tasks will not
/// be able to be committed and should be considered cancelled.
pub async fn commit_compaction(
    dataset: &mut Dataset,
    completed_tasks: Vec<RewriteResult>,
    remap_options: Arc<dyn IndexRemapperOptions>,
    options: &CompactionOptions,
) -> Result<CompactionMetrics> {
    if completed_tasks.is_empty() {
        return Ok(CompactionMetrics::default());
    }

    // If we aren't using stable row ids, then we need to remap indices.
    let needs_remapping = !dataset.manifest.uses_stable_row_ids() && !options.defer_index_remap;

    // Determine the earliest version at which compaction tasks were planned/executed.
    //
    // In distributed mode (e.g. Spark) the caller opens *two separate* Dataset
    // handles: one for `plan_compaction` (at version V) and a fresh one for
    // `commit_compaction` (at the latest version V+N).  Using `dataset.manifest.version`
    // (= V+N) as the transaction's `read_version` would cause the conflict checker to
    // scan only versions after V+N — finding nothing — and therefore silently skip any
    // concurrent DELETE/UPDATE that landed between V and V+N, resurrecting deleted rows.
    //
    // By anchoring `read_version` to the minimum version carried in the RewriteResults
    // we ensure the conflict checker covers the full range [V, V+N] and will reject the
    // commit with a retryable conflict error if a concurrent write touched the same
    // fragments.
    let tasks_read_version = completed_tasks
        .iter()
        .map(|t| t.read_version)
        .min()
        .unwrap_or(dataset.manifest.version);

    let mut completed_tasks = completed_tasks;

    // Single reserve_fragment_ids for all address-style tasks
    let has_address_style = completed_tasks.iter().any(|t| t.row_addrs.is_some());
    if has_address_style {
        let frags: Vec<&mut Fragment> = completed_tasks
            .iter_mut()
            .filter(|t| t.row_addrs.is_some())
            .flat_map(|t| t.new_fragments.iter_mut())
            .collect();
        reserve_fragment_ids(dataset, frags.into_iter()).await?;
    }

    let mut rewrite_groups = Vec::with_capacity(completed_tasks.len());
    let mut metrics = CompactionMetrics::default();

    let mut row_id_map: HashMap<u64, Option<u64>> = HashMap::default();
    let mut frag_reuse_groups: Vec<FragReuseGroup> = Vec::new();
    let mut new_fragment_bitmap: RoaringBitmap = RoaringBitmap::new();

    for task in completed_tasks {
        metrics += task.metrics;
        let rewrite_group = RewriteGroup {
            old_fragments: task.original_fragments.clone(),
            new_fragments: task.new_fragments.clone(),
        };

        if needs_remapping {
            if let Some(row_addrs_bytes) = task.row_addrs {
                let row_addrs =
                    RoaringTreemap::deserialize_from(&mut Cursor::new(&row_addrs_bytes))?;
                let transposed = remapping::transpose_row_addrs(
                    row_addrs,
                    &task.original_fragments,
                    &task.new_fragments,
                );
                row_id_map.extend(transposed);
            }
        } else if options.defer_index_remap {
            let changed_row_addrs = task.row_addrs.ok_or_else(|| {
                Error::internal(
                    "defer_index_remap requires row_addrs but none were provided".to_string(),
                )
            })?;
            frag_reuse_groups.push(FragReuseGroup {
                changed_row_addrs,
                old_frags: task.original_fragments.iter().map(|f| f.into()).collect(),
                new_frags: task.new_fragments.iter().map(|f| f.into()).collect(),
            });

            task.new_fragments.iter().for_each(|frag| {
                new_fragment_bitmap.insert(frag.id as u32);
            });
        }
        rewrite_groups.push(rewrite_group);
    }

    let rewritten_indices = if needs_remapping {
        let index_remapper = remap_options.create_remapper(dataset)?;
        let affected_ids = rewrite_groups
            .iter()
            .flat_map(|group| group.old_fragments.iter().map(|frag| frag.id))
            .collect::<Vec<_>>();

        let remapped_indices = index_remapper
            .remap_indices(row_id_map, &affected_ids)
            .await?;
        remapped_indices
            .into_iter()
            .map(|rewritten| RewrittenIndex {
                old_id: rewritten.old_id,
                new_id: rewritten.new_id,
                new_index_details: rewritten.index_details,
                new_index_version: rewritten.index_version,
                new_index_files: rewritten.files,
            })
            .collect()
    } else if !options.defer_index_remap && !has_address_style {
        // We need to reserve fragment ids here so that the fragment bitmap
        // can be updated for each index. Only needed for stable row IDs
        // since address-style IDs were already reserved above.
        let new_fragments = rewrite_groups
            .iter_mut()
            .flat_map(|group| group.new_fragments.iter_mut())
            .collect::<Vec<_>>();
        reserve_fragment_ids(dataset, new_fragments.into_iter()).await?;
        Vec::new()
    } else {
        Vec::new()
    };

    let frag_reuse_index = if options.defer_index_remap {
        Some(build_new_frag_reuse_index(dataset, frag_reuse_groups, new_fragment_bitmap).await?)
    } else {
        None
    };

    let transaction = TransactionBuilder::new(
        // Use the version at which the compaction tasks were *planned*, not the
        // version of the dataset handle passed to this function.  In distributed
        // mode the caller may open a fresh dataset at a later version (V+N), but
        // the tasks were executed against an older snapshot (V).  Anchoring the
        // transaction to V ensures the OCC conflict checker scans all writes that
        // landed between V and the commit point, detecting concurrent DELETE
        // transactions that would otherwise cause deleted rows to reappear.
        tasks_read_version,
        Operation::Rewrite {
            groups: rewrite_groups,
            rewritten_indices,
            frag_reuse_index,
        },
    )
    .transaction_properties(options.transaction_properties.clone())
    .build();

    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await?;

    Ok(metrics)
}

#[cfg(test)]
mod tests {

    mod binary_copy;
    use self::remapping::RemappedIndex;
    use super::*;
    use crate::dataset::WriteDestination;
    use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
    use crate::dataset::optimize::remapping::{transpose_row_addrs, transpose_row_ids_from_digest};
    use crate::index::frag_reuse::{load_frag_reuse_index_details, open_frag_reuse_index};
    use crate::index::vector::{StageParams, VectorIndexParams};
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
    use arrow_array::{
        ArrayRef, Float32Array, Int32Array, Int64Array, LargeBinaryArray, LargeStringArray,
        PrimitiveArray, RecordBatch, RecordBatchIterator,
    };
    use arrow_schema::{DataType, Field, Schema};
    use arrow_select::concat::concat_batches;
    use async_trait::async_trait;
    use lance_arrow::BLOB_META_KEY;
    use lance_core::Error;
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_datagen::Dimension;
    use lance_file::version::LanceFileVersion;
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use lance_index::scalar::{
        BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
    };
    use lance_index::vector::ivf::IvfBuildParams;
    use lance_index::vector::pq::PQBuildParams;
    use lance_index::{Index, IndexType};
    use lance_linalg::distance::{DistanceType, MetricType};
    use lance_table::io::manifest::read_manifest_indexes;
    use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
    use rstest::rstest;
    use std::collections::HashSet;
    use std::io::Cursor;
    use std::sync::Arc;
    use uuid::Uuid;

    #[test]
    fn test_candidate_bin() {
        let empty_bin = CandidateBin {
            fragments: vec![],
            pos_range: 0..0,
            candidacy: vec![],
            row_counts: vec![],
            indices: vec![],
            known_bytes: 0,
        };
        assert!(empty_bin.is_noop());

        let fragment = Fragment {
            id: 0,
            files: vec![],
            deletion_file: None,
            row_id_meta: None,
            physical_rows: Some(0),
            last_updated_at_version_meta: None,
            created_at_version_meta: None,
        };
        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactWithNeighbors],
            row_counts: vec![100],
            indices: vec![],
            known_bytes: 0,
        };
        assert!(single_bin.is_noop());

        let single_bin = CandidateBin {
            fragments: vec![fragment.clone()],
            pos_range: 0..1,
            candidacy: vec![CompactionCandidacy::CompactItself],
            row_counts: vec![100],
            indices: vec![],
            known_bytes: 0,
        };
        // Not a no-op because it's CompactItself
        assert!(!single_bin.is_noop());

        let big_bin = CandidateBin {
            fragments: std::iter::repeat_n(fragment, 8).collect(),
            pos_range: 0..8,
            candidacy: std::iter::repeat_n(CompactionCandidacy::CompactItself, 8).collect(),
            row_counts: vec![100, 400, 200, 200, 400, 300, 300, 100],
            indices: vec![],
            known_bytes: 0,
        };
        assert!(!big_bin.is_noop());
        let split = big_bin.split_for_size(500);
        assert_eq!(split.len(), 3);
        assert_eq!(split[0].pos_range, 0..2);
        assert_eq!(split[1].pos_range, 2..5);
        assert_eq!(split[2].pos_range, 5..8);
    }

    fn sample_data() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from_iter_values(0..10_000))],
        )
        .unwrap()
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct MockIndexRemapperExpectation {
        expected: HashMap<u64, Option<u64>>,
        answer: Vec<RemappedIndex>,
    }

    #[derive(Debug, Default, Clone, PartialEq)]
    struct MockIndexRemapper {
        expectations: Vec<MockIndexRemapperExpectation>,
    }

    impl MockIndexRemapper {
        fn stringify_map(map: &HashMap<u64, Option<u64>>) -> String {
            let mut sorted_keys = map.keys().collect::<Vec<_>>();
            sorted_keys.sort();
            let mut first_keys = sorted_keys
                .into_iter()
                .take(10)
                .map(|key| {
                    format!(
                        "{}:{:?}",
                        RowAddress::from(*key),
                        map[key].map(RowAddress::from)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            if map.len() > 10 {
                first_keys.push_str(", ...");
            }
            let mut result_str = format!("(len={})", map.len());
            result_str.push_str(&first_keys);
            result_str
        }

        fn in_any_order(expectations: &[Self]) -> Self {
            let expectations = expectations
                .iter()
                .flat_map(|item| item.expectations.clone())
                .collect::<Vec<_>>();
            Self { expectations }
        }
    }

    #[async_trait]
    impl IndexRemapper for MockIndexRemapper {
        async fn remap_indices(
            &self,
            index_map: HashMap<u64, Option<u64>>,
            _: &[u64],
        ) -> Result<Vec<RemappedIndex>> {
            for expectation in &self.expectations {
                if expectation.expected == index_map {
                    return Ok(expectation.answer.clone());
                }
            }
            panic!(
                "Unexpected index map (len={}): {}\n  Options: {}",
                index_map.len(),
                Self::stringify_map(&index_map),
                self.expectations
                    .iter()
                    .map(|expectation| Self::stringify_map(&expectation.expected))
                    .collect::<Vec<_>>()
                    .join("\n  ")
            );
        }
    }

    impl IndexRemapperOptions for MockIndexRemapper {
        fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
            Ok(Box::new(self.clone()))
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_empty(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Compact an empty table
        let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

        let reader = RecordBatchIterator::new(vec![].into_iter().map(Ok), Arc::new(schema));
        let mut dataset = Dataset::write(
            reader,
            test_uri,
            Some(WriteParams {
                data_storage_version: Some(data_storage_version),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.tasks().len(), 0);

        let metrics = compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();

        assert_eq!(metrics, CompactionMetrics::default());
        assert_eq!(dataset.manifest.version, 1);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_all_good(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // Compact a table with nothing to do
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        // Just one file
        let write_params = WriteParams {
            max_rows_per_file: 10_000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // There's only one file, so we can't compact any more if we wanted to.
        let plan = plan_compaction(&dataset, &CompactionOptions::default())
            .await
            .unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Now split across multiple files
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 3_000,
            max_rows_per_group: 1_000,
            data_storage_version: Some(data_storage_version),
            mode: WriteMode::Overwrite,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);
    }

    #[tokio::test]
    async fn test_compact_blob_columns() {
        let test_dir = TempStrDir::default();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("blob", DataType::LargeBinary, false)
                .with_metadata([(BLOB_META_KEY.to_string(), "true".to_string())].into()),
        ]));
        let expected_payload: Vec<Vec<u8>> =
            vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9, 10], vec![11]];
        let id_column: ArrayRef = Arc::new(Int32Array::from_iter_values(
            0..expected_payload.len() as i32,
        ));
        let blob_array: ArrayRef = Arc::new(LargeBinaryArray::from_iter(
            expected_payload.iter().map(|value| Some(value.as_slice())),
        ));
        let batch = RecordBatch::try_new(schema.clone(), vec![id_column, blob_array]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

        let mut dataset = Dataset::write(
            reader,
            &test_dir,
            Some(WriteParams {
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset.validate().await.unwrap();
        assert!(dataset.get_fragments().len() > 1);

        compact_files(&mut dataset, CompactionOptions::default(), None)
            .await
            .unwrap();
        dataset.validate().await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 1);

        let dataset = Arc::new(dataset);
        let row_indices: Vec<u64> = (0..expected_payload.len() as u64).collect();
        let blobs = dataset
            .take_blobs_by_indices(&row_indices, "blob")
            .await
            .unwrap();
        assert_eq!(blobs.len(), expected_payload.len());
        for (blob, expected) in blobs.iter().zip(expected_payload.iter()) {
            let bytes = blob.read().await.unwrap();
            assert_eq!(bytes.as_ref(), expected.as_slice());
        }
    }

    fn row_addrs(frag_idx: u32, offsets: Range<u32>) -> Range<u64> {
        let start = RowAddress::new_from_parts(frag_idx, offsets.start);
        let end = RowAddress::new_from_parts(frag_idx, offsets.end);
        start.into()..end.into()
    }

    // The outer list has one item per new fragment
    // The inner list has ranges of old row ids that map to the new fragment, in order
    fn expect_remap(
        ranges: &[Vec<(Range<u64>, bool)>],
        starting_new_frag_idx: u32,
    ) -> MockIndexRemapper {
        let mut expected_remap: HashMap<u64, Option<u64>> = HashMap::default();
        expected_remap.reserve(ranges.iter().map(|r| r.len()).sum());
        for (new_frag_offset, new_frag_ranges) in ranges.iter().enumerate() {
            let new_frag_idx = starting_new_frag_idx + new_frag_offset as u32;
            let mut row_offset = 0;
            for (old_id_range, is_found) in new_frag_ranges.iter() {
                for old_id in old_id_range.clone() {
                    if *is_found {
                        let new_id = RowAddress::new_from_parts(new_frag_idx, row_offset);
                        expected_remap.insert(old_id, Some(new_id.into()));
                        row_offset += 1;
                    } else {
                        expected_remap.insert(old_id, None);
                    }
                }
            }
        }
        MockIndexRemapper {
            expectations: vec![MockIndexRemapperExpectation {
                expected: expected_remap,
                answer: vec![],
            }],
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_many(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 3 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1200))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 400,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Append 2 large fragments (1k rows)
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(1200, 2000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            data_storage_version: Some(data_storage_version),
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Delete 1 row from first large fragment
        dataset.delete("a = 1300").await.unwrap();

        // Delete 20% of rows from second large fragment
        dataset.delete("a >= 2400 AND a < 2600").await.unwrap();

        // Append 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(3200, 600))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 300,
            data_storage_version: Some(data_storage_version),
            mode: WriteMode::Append,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        let first_new_frag_idx = 7;
        // Predicting the remap is difficult.  One task will remap to fragments 7/8 and the other
        // will remap to fragments 9/10 but we don't know which is which and so we just allow ourselves
        // to expect both possibilities.
        let remap_a = expect_remap(
            &[
                vec![
                    // 3 small fragments are rewritten to frags 7 & 8
                    (row_addrs(0, 0..400), true),
                    (row_addrs(1, 0..400), true),
                    (row_addrs(2, 0..200), true),
                ],
                vec![(row_addrs(2, 200..400), true)],
                // frag 3 is skipped since it does not have enough missing data
                // Frags 4, 5, and 6 are rewritten to frags 9 & 10
                vec![
                    // Only 800 of the 1000 rows taken from frag 4
                    (row_addrs(4, 0..200), true),
                    (row_addrs(4, 200..400), false),
                    (row_addrs(4, 400..1000), true),
                    // frags 5 compacted with frag 4
                    (row_addrs(5, 0..200), true),
                ],
                vec![(row_addrs(5, 200..300), true), (row_addrs(6, 0..300), true)],
            ],
            first_new_frag_idx,
        );
        let remap_b = expect_remap(
            &[
                // Frags 4, 5, and 6 are rewritten to frags 7 & 8
                vec![
                    (row_addrs(4, 0..200), true),
                    (row_addrs(4, 200..400), false),
                    (row_addrs(4, 400..1000), true),
                    (row_addrs(5, 0..200), true),
                ],
                vec![(row_addrs(5, 200..300), true), (row_addrs(6, 0..300), true)],
                // 3 small fragments rewritten to frags 9 & 10
                vec![
                    (row_addrs(0, 0..400), true),
                    (row_addrs(1, 0..400), true),
                    (row_addrs(2, 0..200), true),
                ],
                vec![(row_addrs(2, 200..400), true)],
            ],
            first_new_frag_idx,
        );

        // Create compaction plan
        let options = CompactionOptions {
            target_rows_per_fragment: 1000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 2);
        assert_eq!(plan.tasks()[0].fragments.len(), 3);
        assert_eq!(plan.tasks()[1].fragments.len(), 3);

        assert_eq!(
            plan.tasks()[0]
                .fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            plan.tasks()[1]
                .fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![4, 5, 6]
        );

        let mock_remapper = MockIndexRemapper::in_any_order(&[remap_a, remap_b]);

        // Run compaction
        let metrics = compact_files(&mut dataset, options, Some(Arc::new(mock_remapper)))
            .await
            .unwrap();

        // Assert on metrics
        assert_eq!(metrics.fragments_removed, 6);
        assert_eq!(metrics.fragments_added, 4);
        assert_eq!(metrics.files_removed, 7); // 6 data files + 1 deletion file
        assert_eq!(metrics.files_added, 4);

        let fragment_ids = dataset
            .get_fragments()
            .iter()
            .map(|f| f.id())
            .collect::<Vec<_>>();
        assert_eq!(fragment_ids, vec![3, 7, 8, 9, 10]);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_data_files(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 2 small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 5_000,
            max_rows_per_group: 1_000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Add a column
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("x", DataType::Float32, false),
        ]);

        let data = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from_iter_values(0..10_000)),
                Arc::new(Float32Array::from_iter_values(
                    (0..10_000).map(|x| x as f32 * std::f32::consts::PI),
                )),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], data.schema());

        dataset.merge(reader, "a", "a").await.unwrap();

        let expected_remap = expect_remap(
            &[vec![
                // 3 small fragments are rewritten entirely
                (row_addrs(0, 0..5000), true),
                (row_addrs(1, 0..5000), true),
            ]],
            2,
        );

        let plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                ..Default::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(plan.tasks().len(), 1);
        assert_eq!(plan.tasks()[0].fragments.len(), 2);

        let metrics = compact_files(&mut dataset, plan.options, Some(Arc::new(expected_remap)))
            .await
            .unwrap();

        assert_eq!(metrics.files_removed, 4); // 2 fragments with 2 data files
        assert_eq!(metrics.files_added, 1); // 1 fragment with 1 data file
        assert_eq!(metrics.fragments_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        // Assert order unchanged and data is all there.
        let scanner = dataset.scan();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let scanned_data = concat_batches(&batches[0].schema(), &batches).unwrap();

        assert_eq!(scanned_data, data);
    }

    #[rstest]
    #[tokio::test]
    async fn test_compact_deletions(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
    ) {
        // For files that have few rows, we don't want to compact just 1 since
        // that won't do anything. But if there are deletions to materialize,
        // we want to do groups of 1. This test checks that.
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Create a table with 1 fragment
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 1000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            data_storage_version: Some(data_storage_version),
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        dataset.delete("a <= 500").await.unwrap();

        // Threshold must be satisfied
        let mut options = CompactionOptions {
            materialize_deletions_threshold: 0.8,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Ignore deletions if materialize_deletions is false
        options.materialize_deletions_threshold = 0.1;
        options.materialize_deletions = false;
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 0);

        // Materialize deletions if threshold is met
        options.materialize_deletions = true;
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 1);

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert_eq!(metrics.fragments_removed, 1);
        assert_eq!(metrics.files_removed, 2);
        assert_eq!(metrics.fragments_added, 1);

        let fragments = dataset.get_fragments();
        assert_eq!(fragments.len(), 1);
        assert!(fragments[0].metadata.deletion_file.is_none());
    }

    #[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
    struct IgnoreRemap {}

    #[async_trait]
    impl IndexRemapper for IgnoreRemap {
        async fn remap_indices(
            &self,
            _: HashMap<u64, Option<u64>>,
            _: &[u64],
        ) -> Result<Vec<RemappedIndex>> {
            Ok(Vec::new())
        }
    }

    impl IndexRemapperOptions for IgnoreRemap {
        fn create_remapper(&self, _: &Dataset) -> Result<Box<dyn IndexRemapper>> {
            Ok(Box::new(Self {}))
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn test_compact_distributed(
        #[values(LanceFileVersion::Legacy, LanceFileVersion::Stable)]
        data_storage_version: LanceFileVersion,
        #[values(false, true)] use_stable_row_id: bool,
    ) {
        // Can run the tasks independently
        // Can provide subset of tasks to commit_compaction
        // Once committed, can't commit remaining tasks
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();

        // Write dataset as 9 1k row fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.slice(0, 9000))], data.schema());
        let write_params = WriteParams {
            max_rows_per_file: 1000,
            data_storage_version: Some(data_storage_version),
            enable_stable_row_ids: use_stable_row_id,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        // Plan compaction with 3 tasks
        let options = CompactionOptions {
            target_rows_per_fragment: 3_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 3);

        let dataset_ref = &dataset;
        let mut results = futures::stream::iter(plan.compaction_tasks())
            .then(|task| async move { task.execute(dataset_ref).await.unwrap() })
            .collect::<Vec<_>>()
            .await;

        assert_eq!(results.len(), 3);

        assert_eq!(
            results[0]
                .original_fragments
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(results[0].metrics.files_removed, 3);
        assert_eq!(results[0].metrics.files_added, 1);

        // Just commit the last task
        commit_compaction(
            &mut dataset,
            vec![results.pop().unwrap()],
            Arc::new(IgnoreRemap::default()),
            &options,
        )
        .await
        .unwrap();

        // 1 commit for reserve fragments and 1 for final commit, both
        // from the call to commit_compaction
        assert_eq!(dataset.manifest.version, 3);

        // Can commit the remaining tasks
        commit_compaction(
            &mut dataset,
            results,
            Arc::new(IgnoreRemap::default()),
            &options,
        )
        .await
        .unwrap();
        // 1 commit for reserve fragments and 1 for final commit, both
        // from the call to commit_compaction
        assert_eq!(dataset.manifest.version, 5);

        assert_eq!(dataset.manifest.uses_stable_row_ids(), use_stable_row_id,);
    }

    #[tokio::test]
    async fn test_stable_row_indices() {
        // Validate behavior of indices after compaction with stable row ids.
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(16).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));
        let mut dataset = Dataset::write(
            data_gen.batch(500),
            "memory://test/table",
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 100, // 5 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Delete first 110 rows so rowids != final rowaddrs
        // First 100 rows deletes first file. Next 10 deletes part of second
        // file, so we will trigger the with deletions code path.
        dataset.delete("i < 110").await.unwrap();

        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("scalar".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let params = VectorIndexParams::ivf_pq(1, 8, 1, MetricType::L2, 50);
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                Some("vector".into()),
                &params,
                false,
            )
            .await
            .unwrap();

        async fn index_set(dataset: &Dataset) -> HashSet<Uuid> {
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .map(|index| index.uuid)
                .collect()
        }
        let indices = index_set(&dataset).await;

        async fn vector_query(dataset: &Dataset) -> RecordBatch {
            let mut scanner = dataset.scan();

            let query = Float32Array::from(vec![0.0f32; 16]);
            scanner
                .nearest("vec", &query, 10)
                .unwrap()
                .project(&["i"])
                .unwrap();

            scanner.try_into_batch().await.unwrap()
        }

        async fn scalar_query(dataset: &Dataset) -> RecordBatch {
            let mut scanner = dataset.scan();

            scanner.filter("i = 100").unwrap().project(&["i"]).unwrap();

            scanner.try_into_batch().await.unwrap()
        }

        let before_vec_result = vector_query(&dataset).await;
        let before_scalar_result = scalar_query(&dataset).await;

        let options = CompactionOptions {
            target_rows_per_fragment: 180,
            ..Default::default()
        };
        let _metrics = compact_files(&mut dataset, options, None).await.unwrap();

        // The indices should be unchanged after compaction, since we are using
        // stable row ids.
        let current_indices = index_set(&dataset).await;
        assert_eq!(indices, current_indices);

        let after_vec_result = vector_query(&dataset).await;
        assert_eq!(before_vec_result, after_vec_result);

        let after_scalar_result = scalar_query(&dataset).await;
        assert_eq!(before_scalar_result, after_scalar_result);
    }

    // Regression test for https://github.com/lancedb/lance/issues/6161
    // When FragReuseIndexDetails exceeds 204800 bytes it is written to an external
    // file. Previously the file was silently dropped (temp file deleted) because
    // tokio::io::AsyncWriteExt::shutdown was called instead of
    // lance_io::traits::Writer::shutdown, which persists the temp file.
    #[tokio::test]
    async fn test_defer_index_remap_large_external_file() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Create ~150 fragments × 1000 rows to produce a FragReuseIndexDetails
        // that exceeds the 204800-byte inline threshold (~302 KB serialized).
        let num_fragments = 150usize;
        let rows_per_fragment = 1000usize;
        let total_rows = num_fragments * rows_per_fragment;

        let schema = Arc::new(Schema::new(vec![Field::new("i", DataType::Int32, false)]));

        let mut dataset = Dataset::write(
            RecordBatchIterator::new(
                vec![Ok(RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(Int32Array::from_iter_values(0..total_rows as i32)) as ArrayRef],
                )
                .unwrap())],
                schema.clone(),
            ),
            test_uri,
            Some(WriteParams {
                max_rows_per_file: rows_per_fragment,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset.get_fragments().len(), num_fragments);

        // Delete a few rows from each fragment so compaction has something to do.
        dataset.delete("i % 1000 = 0").await.unwrap();

        compact_files(
            &mut dataset,
            CompactionOptions {
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Loading the FragReuseIndex details must succeed even when the details
        // were written to an external file.
        let frag_reuse_meta = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .expect("fragment reuse index must exist after compaction");

        load_frag_reuse_index_details(&dataset, &frag_reuse_meta)
            .await
            .expect("loading large frag reuse index details must not fail");
    }

    #[tokio::test]
    async fn test_defer_index_remap() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create another same dataset to mimic behavior without deferred index remap
        let mut data_gen2 = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset2 = Dataset::write(
            data_gen2.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Delete some rows to create deletions
        dataset.delete("i < 500").await.unwrap();
        dataset2.delete("i < 500").await.unwrap();

        // Create a scalar index to check this is not touched
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                Some("scalar".into()),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Verify the initial state - no fragment reuse index should exist
        let initial_indices = dataset.load_indices().await.unwrap();
        assert_eq!(initial_indices.len(), 1);
        assert_eq!(initial_indices[0].name, "scalar");

        // Store the original scalar index UUID for comparison
        let original_scalar_uuid = initial_indices[0].uuid;

        // Plan and execute compaction manually
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };
        let options2 = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: false,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let plan2 = plan_compaction(&dataset2, &options2).await.unwrap();

        let mut expected_all_old_frag_ids = Vec::new();
        let mut expected_all_new_frag_ids = Vec::new();
        let mut expected_all_new_frag_bitmap = RoaringBitmap::new();
        let mut expected_all_row_id_map = HashMap::new();
        let mut deferred_results = Vec::new();
        let mut immediate_results = Vec::new();

        for (task, task2) in plan.tasks().iter().zip(plan2.tasks()) {
            let deferred_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
                .await
                .unwrap();
            let immediate_result =
                rewrite_files(Cow::Borrowed(&dataset2), task2.clone(), &options2)
                    .await
                    .unwrap();

            // Both should produce row_addrs (address-style row IDs)
            assert!(deferred_result.row_addrs.is_some());
            assert!(!deferred_result.row_addrs.as_ref().unwrap().is_empty());
            assert!(!deferred_result.row_addrs.as_ref().unwrap().is_empty());
            assert!(!deferred_result.original_fragments.is_empty());
            assert!(!deferred_result.new_fragments.is_empty());

            assert!(immediate_result.row_addrs.is_some());
            assert!(!immediate_result.original_fragments.is_empty());
            assert!(!immediate_result.new_fragments.is_empty());

            // Both should capture the same row addresses
            assert_eq!(deferred_result.row_addrs, immediate_result.row_addrs);

            deferred_results.push(deferred_result);
            immediate_results.push(immediate_result);
        }

        // Reserve fragment IDs for immediate results to build expected values
        {
            let frags: Vec<&mut Fragment> = immediate_results
                .iter_mut()
                .flat_map(|r| r.new_fragments.iter_mut())
                .collect();
            reserve_fragment_ids(&dataset2, frags.into_iter())
                .await
                .unwrap();
        }

        // Build expected values by transposing using the immediate results
        for immediate_result in &immediate_results {
            let row_addrs_bytes = immediate_result.row_addrs.as_ref().unwrap();
            let row_addrs =
                RoaringTreemap::deserialize_from(&mut Cursor::new(row_addrs_bytes)).unwrap();
            let transposed = transpose_row_addrs(
                row_addrs,
                &immediate_result.original_fragments,
                &immediate_result.new_fragments,
            );
            expected_all_row_id_map.extend(transposed);
            immediate_result.new_fragments.iter().for_each(|frag| {
                expected_all_new_frag_bitmap.insert(frag.id as u32);
            });
            expected_all_new_frag_ids.extend(
                immediate_result
                    .new_fragments
                    .iter()
                    .map(|s| s.id)
                    .collect::<Vec<_>>(),
            );
            expected_all_old_frag_ids.extend(
                immediate_result
                    .original_fragments
                    .iter()
                    .map(|s| s.id)
                    .collect::<Vec<_>>(),
            );
        }

        // Now commit the first compaction (using deferred results)
        let first_metrics = commit_compaction(
            &mut dataset,
            deferred_results.clone(),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Verify compaction happened
        assert!(first_metrics.fragments_removed > 0);
        assert!(first_metrics.fragments_added > 0);

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };

        assert_eq!(
            frag_reuse_index_meta.fragment_bitmap.clone().unwrap(),
            expected_all_new_frag_bitmap
        );
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        let frag_reuse_index =
            open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
                .await
                .unwrap();
        let stats = frag_reuse_index.statistics().unwrap();
        assert_eq!(
            serde_json::to_string(&stats).unwrap(),
            dataset
                .index_statistics(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
        );

        // Verify the index has one version with the correct dataset version
        let compaction_version = &frag_reuse_index.details.versions[0];
        assert_eq!(frag_reuse_index.details.versions.len(), 1);
        assert_eq!(
            compaction_version.dataset_version,
            frag_reuse_index_meta.dataset_version
        );

        // Verify the index compaction version information matches the RewriteResults
        let mut compacted_all_old_frag_digests = Vec::new();
        let mut compacted_all_new_frag_digests = Vec::new();
        let mut transposed_map = HashMap::new();
        for group in compaction_version.groups.iter() {
            let changed_row_addr_bytes = &group.changed_row_addrs;
            let mut cursor = Cursor::new(&changed_row_addr_bytes);
            let changed_row_addrs = RoaringTreemap::deserialize_from(&mut cursor).unwrap();
            compacted_all_old_frag_digests.extend(group.old_frags.clone());
            compacted_all_new_frag_digests.extend(group.new_frags.clone());

            let group_transposed_map = transpose_row_ids_from_digest(
                changed_row_addrs,
                &group.old_frags,
                &group.new_frags,
            );
            transposed_map.extend(group_transposed_map);
        }
        assert_eq!(transposed_map, expected_all_row_id_map);
        assert_eq!(
            compacted_all_old_frag_digests
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            expected_all_old_frag_ids
        );
        assert_eq!(
            compacted_all_new_frag_digests
                .iter()
                .map(|f| f.id)
                .collect::<Vec<_>>(),
            expected_all_new_frag_ids
        );

        // Verify the scalar index UUID is unchanged (it should not be remapped yet)
        let Some(current_scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };
        assert_eq!(current_scalar_index.uuid, original_scalar_uuid);
    }

    #[tokio::test]
    async fn test_defer_index_remap_multiple_compactions() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let mut compact_read_versions = Vec::new();
        for i in 0..10 {
            dataset
                .delete(&format!("i < {}", 500 * (i + 1)))
                .await
                .unwrap();
            let read_version = dataset.manifest.version;
            compact_files(&mut dataset, options.clone(), None)
                .await
                .unwrap();

            // Record the read version for verification if compaction has happened
            if dataset.manifest.version > read_version {
                compact_read_versions.push(read_version);
            }

            // Load and verify the fragment reuse index content
            let Some(frag_reuse_index_meta) = dataset
                .load_index_by_name(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap()
            else {
                panic!("Fragment reuse index must be available");
            };
            let frag_reuse_details =
                load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
                    .await
                    .unwrap();
            let frag_reuse_index =
                open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
                    .await
                    .unwrap();

            // Verify the index has one version with the correct dataset version
            assert_eq!(
                frag_reuse_index
                    .details
                    .versions
                    .iter()
                    .map(|v| v.dataset_version)
                    .collect::<Vec<_>>(),
                compact_read_versions
            );
        }
    }

    #[tokio::test]
    async fn test_remap_index_after_compaction() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create a index to be remapped
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        // Remap without a frag reuse index should yield unsupported
        let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };

        let result = remapping::remap_column_index(&mut dataset, &["i"], index_name.clone()).await;
        assert!(matches!(result, Err(Error::NotSupported { .. })));

        let plan = plan_compaction(&dataset, &options).await.unwrap();

        // Commit each rewrite task separately to simulate 3 compaction runs
        // being accumulated in the fragment reuse index
        for task in plan.tasks().iter() {
            let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), task.clone(), &options)
                .await
                .unwrap();

            commit_compaction(
                &mut dataset,
                Vec::from([rewrite_result]),
                Arc::new(DatasetIndexRemapperOptions::default()),
                &options,
            )
            .await
            .unwrap();
        }

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        let frag_reuse_index =
            open_frag_reuse_index(frag_reuse_index_meta.uuid, frag_reuse_details.as_ref())
                .await
                .unwrap();

        assert_eq!(frag_reuse_index.details.versions.len(), plan.tasks().len());

        // Check auto-remap
        let mut all_fragment_bitmap = RoaringBitmap::new();
        dataset.fragments().iter().for_each(|f| {
            all_fragment_bitmap.insert(f.id as u32);
        });
        let Some(scalar_index_before_remap) = dataset.load_index_by_name("scalar").await.unwrap()
        else {
            panic!("scalar index must be available");
        };
        assert_eq!(
            scalar_index_before_remap.fragment_bitmap.unwrap(),
            all_fragment_bitmap
        );

        // Trigger index remap
        remapping::remap_column_index(&mut dataset, &["i"], index_name.clone())
            .await
            .unwrap();

        // Compare against original index
        let indices = read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap();
        let Some(remapped_scalar_index) = indices.into_iter().find(|idx| idx.name == "scalar")
        else {
            panic!("scalar index must be available");
        };
        assert_ne!(remapped_scalar_index.uuid, scalar_index.uuid);
        assert_eq!(
            remapped_scalar_index.fragment_bitmap.unwrap(),
            all_fragment_bitmap
        );
    }

    #[tokio::test]
    async fn test_concurrent_compaction_reindex_compaction_commit_first() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create an index
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Write some more data for reindexing
        Dataset::write(
            data_gen.batch(6_000),
            WriteDestination::Dataset(Arc::new(dataset.clone())),
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset.checkout_latest().await.unwrap();
        let mut dataset_clone = dataset.clone();

        // First commit a compaction with deferred remap
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Concurrent reindex should succeed
        dataset_clone
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Check new index does not cover the compacted files
        dataset.checkout_latest().await.unwrap();

        let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };
        let index_frags = scalar_index
            .fragment_bitmap
            .unwrap()
            .iter()
            .collect::<HashSet<_>>();
        assert_eq!(
            index_frags,
            dataset
                .fragments()
                .iter()
                .map(|f| f.id as u32)
                .collect::<HashSet<_>>()
        )
    }

    #[tokio::test]
    async fn test_concurrent_compaction_reindex_reindex_commit_first() {
        let mut data_gen = BatchGenerator::new()
            .col(Box::new(
                RandomVector::new().vec_width(128).named("vec".to_owned()),
            ))
            .col(Box::new(IncrementingInt32::new().named("i".to_owned())));

        let mut dataset = Dataset::write(
            data_gen.batch(6_000),
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Create an index
        let index_name = Some("scalar".into());
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();

        // Write some more data for reindexing
        Dataset::write(
            data_gen.batch(6_000),
            WriteDestination::Dataset(Arc::new(dataset.clone())),
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                mode: WriteMode::Append,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        dataset.checkout_latest().await.unwrap();
        let mut dataset_clone = dataset.clone();

        // Concurrent reindex should succeed
        dataset
            .create_index(
                &["i"],
                IndexType::Scalar,
                index_name.clone(),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // First commit a compaction with deferred remap
        compact_files(
            &mut dataset_clone,
            CompactionOptions {
                target_rows_per_fragment: 2_000,
                defer_index_remap: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();

        // Check new index is auto-remapped
        dataset.checkout_latest().await.unwrap();
        let Some(scalar_index) = dataset.load_index_by_name("scalar").await.unwrap() else {
            panic!("scalar index must be available");
        };
        let index_frags = scalar_index
            .fragment_bitmap
            .unwrap()
            .iter()
            .collect::<HashSet<_>>();
        assert_eq!(
            index_frags,
            dataset
                .fragments()
                .iter()
                .map(|f| f.id as u32)
                .collect::<HashSet<_>>()
        )
    }

    #[tokio::test]
    async fn test_concurrent_cleanup_and_compaction_rebase_cleanup() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let tasks = plan.tasks();

        // Only compact the first task, record the state of the dataset
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        let mut dataset_clone = dataset.clone();

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };

        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        // First commit the remaining 2 compaction tasks.
        let rewrite_result2 = rewrite_files(Cow::Borrowed(&dataset), tasks[1].clone(), &options)
            .await
            .unwrap();
        let rewritten_frags2 = rewrite_result2
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result2]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Get the new fragment IDs from the frag_reuse_index after commit
        let frag_reuse_index_meta2 = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .unwrap();
        let frag_reuse_details2 = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta2)
            .await
            .unwrap();
        let new_frags2 = frag_reuse_details2.versions.last().unwrap().new_frag_ids();

        let rewrite_result3 = rewrite_files(Cow::Borrowed(&dataset), tasks[2].clone(), &options)
            .await
            .unwrap();
        let rewritten_frags3 = rewrite_result3
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result3]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Get the new fragment IDs from the frag_reuse_index after commit
        let frag_reuse_index_meta3 = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
            .unwrap();
        let frag_reuse_details3 = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta3)
            .await
            .unwrap();
        let new_frags3 = frag_reuse_details3.versions.last().unwrap().new_frag_ids();

        // Concurrently commit a frag_reuse_index cleanup operation.
        // Because there is no index, it should remove the first version.
        // but after rebase it should contain the new compaction versions.
        cleanup_frag_reuse_index(&mut dataset_clone).await.unwrap();

        // Load and verify the fragment reuse index content
        dataset.checkout_latest().await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 2);
        assert_eq!(
            frag_reuse_details.versions[0].old_frag_ids(),
            rewritten_frags2
        );
        assert_eq!(frag_reuse_details.versions[0].new_frag_ids(), new_frags2);
        assert_eq!(
            frag_reuse_details.versions[1].old_frag_ids(),
            rewritten_frags3
        );
        assert_eq!(frag_reuse_details.versions[1].new_frag_ids(), new_frags3);
    }

    #[tokio::test]
    async fn test_concurrent_cleanup_and_compaction_rebase_compaction() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let tasks = plan.tasks();

        // Only compact the first task, record the state of the dataset
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        let mut dataset_clone = dataset.clone();

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        // First commit the frag_reuse_index cleanup
        // Because there is no index, it should remove the first version.
        cleanup_frag_reuse_index(&mut dataset).await.unwrap();

        // Load and verify the fragment reuse index content
        dataset.checkout_latest().await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 0);

        // Concurrently commit a rewrite
        // After rebase it should only contain the latest reuse version
        let rewrite_result2 =
            rewrite_files(Cow::Borrowed(&dataset_clone), tasks[1].clone(), &options)
                .await
                .unwrap();
        let rewritten_frags2 = rewrite_result2
            .original_fragments
            .iter()
            .map(|f| f.id)
            .collect::<Vec<_>>();
        commit_compaction(
            &mut dataset_clone,
            Vec::from([rewrite_result2]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Load and verify the fragment reuse index content
        dataset.checkout_latest().await.unwrap();
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);
        assert_eq!(
            frag_reuse_details.versions[0].old_frag_ids(),
            rewritten_frags2
        );
        // Verify new fragment IDs are non-zero (allocated by commit_compaction)
        let new_frags2 = frag_reuse_details.versions[0].new_frag_ids();
        assert!(new_frags2.iter().all(|id| *id != 0));
    }

    #[tokio::test]
    async fn test_concurrent_compactions_with_defer_index_remap() {
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let plan = plan_compaction(&dataset, &options).await.unwrap();
        let tasks = plan.tasks();

        let mut dataset_clone = dataset.clone();

        // Only compact the first task, record the state of the dataset
        let rewrite_result = rewrite_files(Cow::Borrowed(&dataset), tasks[0].clone(), &options)
            .await
            .unwrap();

        commit_compaction(
            &mut dataset,
            Vec::from([rewrite_result]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .unwrap();

        // Load and verify the fragment reuse index content
        let Some(frag_reuse_index_meta) = dataset
            .load_index_by_name(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap()
        else {
            panic!("Fragment reuse index must be available");
        };
        let frag_reuse_details = load_frag_reuse_index_details(&dataset, &frag_reuse_index_meta)
            .await
            .unwrap();
        assert_eq!(frag_reuse_details.versions.len(), 1);

        // Concurrently commit a rewrite should fail
        let rewrite_result2 =
            rewrite_files(Cow::Borrowed(&dataset_clone), tasks[1].clone(), &options)
                .await
                .unwrap();
        let result = commit_compaction(
            &mut dataset_clone,
            Vec::from([rewrite_result2]),
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await;
        assert!(matches!(result, Err(Error::RetryableCommitConflict { .. })));
    }

    #[tokio::test]
    async fn test_read_bitmap_index_with_defer_index_remap() {
        // Create a dataset with categorical values
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "category",
                lance_datagen::array::cycle::<Int32Type>(vec![1, 2, 3]),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get initial counts for each category
        let count1 = dataset
            .count_rows(Some("category = 1".to_owned()))
            .await
            .unwrap();
        let count2 = dataset
            .count_rows(Some("category = 2".to_owned()))
            .await
            .unwrap();
        let count3 = dataset
            .count_rows(Some("category = 3".to_owned()))
            .await
            .unwrap();

        // Create a bitmap index on the category column
        let index_name = Some("category_idx".into());
        dataset
            .create_index(
                &["category"],
                IndexType::Bitmap,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices
            .iter()
            .find(|idx| idx.name == "category_idx")
            .unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("category_idx").await.unwrap() else {
            panic!("category index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some("category = 1".to_owned()))
                .await
                .unwrap(),
            count1
        );
        assert_eq!(
            dataset
                .count_rows(Some("category = 2".to_owned()))
                .await
                .unwrap(),
            count2
        );
        assert_eq!(
            dataset
                .count_rows(Some("category = 3".to_owned()))
                .await
                .unwrap(),
            count3
        );

        // Verify that after index creation and compaction, scan uses bitmap index scan
        let mut scanner = dataset.scan();
        scanner.filter("category = 1").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[category = 1]@category_idx(Bitmap)"),
            "Expected index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_btree_index_with_defer_index_remap() {
        // Create a dataset with an incremental ID column
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col("id", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(110), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get initial counts for some ID ranges
        let count_low = dataset
            .count_rows(Some("id < 1000".to_owned()))
            .await
            .unwrap();
        let count_mid = dataset
            .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
            .await
            .unwrap();
        let count_high = dataset
            .count_rows(Some("id >= 5000".to_owned()))
            .await
            .unwrap();

        // Create a btree index on the id column
        let index_name = Some("id_idx".into());
        dataset
            .create_index(
                &["id"],
                IndexType::BTree,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "id_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 50_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("id_idx").await.unwrap() else {
            panic!("id index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some("id < 1000".to_owned()))
                .await
                .unwrap(),
            count_low
        );
        assert_eq!(
            dataset
                .count_rows(Some("id >= 2000 and id < 3000".to_owned()))
                .await
                .unwrap(),
            count_mid
        );
        assert_eq!(
            dataset
                .count_rows(Some("id >= 5000".to_owned()))
                .await
                .unwrap(),
            count_high
        );

        // Verify that after index creation and compaction, scan uses btree index scan
        let mut scanner = dataset.scan();
        scanner.filter("id >= 2000 and id < 3000").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[id >= 2000 && id < 3000]@id_idx(BTree)"),
            "Expected scalar index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_inverted_index_with_defer_index_remap() {
        // Generate random words using lance-datagen
        let mut words_gen = lance_datagen::array::random_sentence(1, 100, true);
        let doc_col = words_gen
            .generate_default(lance_datagen::RowCount::from(6000))
            .unwrap();

        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new("doc", DataType::LargeUtf8, false)]).into(),
            vec![doc_col.clone()],
        )
        .unwrap();
        let schema_ref = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
        let mut dataset = Dataset::write(
            stream,
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Get initial counts for some word searches
        // Extract some test words from the generated documents
        let large_string_array = doc_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
        let sample_words: Vec<String> = large_string_array
            .value(0)
            .split_whitespace()
            .take(10)
            .map(|s| s.to_string())
            .collect();
        let test_word1 = &sample_words[0];
        let test_word2 = &sample_words[1];
        let test_word3 = &sample_words[2];

        // Create an inverted index on the doc column
        let index_name = Some("doc_idx".into());
        dataset
            .create_index(
                &["doc"],
                IndexType::Inverted,
                index_name.clone(),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "doc_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("doc_idx").await.unwrap() else {
            panic!("doc index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Initial scan
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let count1 = scanner.count_rows().await.unwrap();
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word2.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let count2 = scanner.count_rows().await.unwrap();
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word3.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let count3 = scanner.count_rows().await.unwrap();

        // Verify that after index creation and compaction, scan uses inverted index scan
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(true).await.unwrap();
        assert!(
            plan.contains("MatchQuery"),
            "Expected inverted index scan in plan: {}",
            plan
        );
        assert!(
            !plan.contains("LanceScan"),
            "Expected no fragment scan in plan: {}",
            plan
        );

        // Reindex to the latest
        dataset
            .create_index(
                &["doc"],
                IndexType::Inverted,
                index_name.clone(),
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();

        // Verify that scans still work correctly and return the same counts
        let mut scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word1.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(scanner.count_rows().await.unwrap(), count1);
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word2.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(scanner.count_rows().await.unwrap(), count2);
        scanner = dataset.scan();
        scanner
            .full_text_search(FullTextSearchQuery::new(test_word3.clone()))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        assert_eq!(scanner.count_rows().await.unwrap(), count3);
    }

    #[tokio::test]
    async fn test_read_ngram_index_with_defer_index_remap() {
        // Generate random words using lance-datagen
        let mut words_gen = lance_datagen::array::random_sentence(1, 100, true);
        let doc_col = words_gen
            .generate_default(lance_datagen::RowCount::from(6000))
            .unwrap();

        let batch = RecordBatch::try_new(
            Schema::new(vec![Field::new("doc", DataType::LargeUtf8, false)]).into(),
            vec![doc_col.clone()],
        )
        .unwrap();
        let schema_ref = batch.schema();
        let stream = RecordBatchIterator::new(vec![batch].into_iter().map(Ok), schema_ref);
        let mut dataset = Dataset::write(
            stream,
            "memory://test/table",
            Some(WriteParams {
                max_rows_per_file: 1_000, // 6 files
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        // Get initial counts for some word searches
        // Extract some test words from the generated documents
        let large_string_array = doc_col.as_any().downcast_ref::<LargeStringArray>().unwrap();
        let sample_words: Vec<String> = large_string_array
            .value(0)
            .split_whitespace()
            .take(10)
            .map(|s| s.to_string())
            .collect();
        let test_word1 = &sample_words[0];
        let test_word2 = &sample_words[1];
        let test_word3 = &sample_words[2];

        // Create an inverted index on the doc column
        let index_name = Some("doc_idx".into());
        dataset
            .create_index(
                &["doc"],
                IndexType::NGram,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "doc_idx").unwrap();

        // Initial scan
        let count1 = dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word1)))
            .await
            .unwrap();
        let count2 = dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word2)))
            .await
            .unwrap();
        let count3 = dataset
            .count_rows(Some(format!("contains(doc, '{}')", test_word3)))
            .await
            .unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("doc_idx").await.unwrap() else {
            panic!("doc index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some(format!("contains(doc, '{}')", test_word1)))
                .await
                .unwrap(),
            count1
        );
        assert_eq!(
            dataset
                .count_rows(Some(format!("contains(doc, '{}')", test_word2)))
                .await
                .unwrap(),
            count2
        );
        assert_eq!(
            dataset
                .count_rows(Some(format!("contains(doc, '{}')", test_word3)))
                .await
                .unwrap(),
            count3
        );

        // Verify that after index creation and compaction, scan uses inverted index scan
        let mut scanner = dataset.scan();
        scanner
            .filter(&format!("contains(doc, '{}')", test_word1))
            .unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ScalarIndexQuery: query=[contains(doc, Utf8"),
            "Expected scalar index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_label_list_index_with_defer_index_remap() {
        // Create a dataset with list data for labels
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .col(
                "labels",
                lance_datagen::array::rand_list_any(
                    lance_datagen::array::cycle::<Int64Type>(vec![1, 2, 3, 4, 5]),
                    false,
                ),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get initial counts for different label values
        let count1 = dataset
            .count_rows(Some("array_has_any(labels, [1])".to_owned()))
            .await
            .unwrap();
        let count2 = dataset
            .count_rows(Some("array_has_any(labels, [5])".to_owned()))
            .await
            .unwrap();
        let count3 = dataset
            .count_rows(Some("array_has_any(labels, [10])".to_owned()))
            .await
            .unwrap();

        // Create a label list index on the labels column
        let index_name = Some("labels_idx".into());
        dataset
            .create_index(
                &["labels"],
                IndexType::LabelList,
                index_name.clone(),
                &ScalarIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "labels_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2000,
            defer_index_remap: true,
            ..Default::default()
        };
        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify that the index UUID remains unchanged
        let indices = dataset.load_indices().await.unwrap();
        let current_index = indices.iter().find(|idx| idx.name == "labels_idx").unwrap();
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that scans still work correctly and return the same counts
        assert_eq!(
            dataset
                .count_rows(Some("array_has_any(labels, [1])".to_owned()))
                .await
                .unwrap(),
            count1
        );
        assert_eq!(
            dataset
                .count_rows(Some("array_has_any(labels, [5])".to_owned()))
                .await
                .unwrap(),
            count2
        );
        assert_eq!(
            dataset
                .count_rows(Some("array_has_any(labels, [10])".to_owned()))
                .await
                .unwrap(),
            count3
        );

        // Verify that after index creation and compaction, scan uses label list index scan
        let mut scanner = dataset.scan();
        scanner.filter("array_has_any(labels, [1])").unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains(
                "ScalarIndexQuery: query=[array_has_any(labels, List([1]))]@labels_idx(LabelList)",
            ),
            "Expected scalar index query in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_read_ivf_pq_index_v3_with_defer_index_remap() {
        // Create a dataset with vector data
        let mut dataset = lance_datagen::gen_batch()
            .col(
                "vec",
                lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(128)),
            )
            .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
            .await
            .unwrap();

        // Get some query vectors for KNN search
        let query_vec1: PrimitiveArray<Float32Type> =
            PrimitiveArray::from_iter_values(std::iter::repeat_n(0.0, 128));
        let query_vec2: PrimitiveArray<Float32Type> =
            PrimitiveArray::from_iter_values(std::iter::repeat_n(1.1, 128));
        let query_vec3: PrimitiveArray<Float32Type> =
            PrimitiveArray::from_iter_values(std::iter::repeat_n(2.2, 128));

        // Get initial KNN search results
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec1, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let results1 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count1 = results1.len();

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec2, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let results2 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count2 = results2.len();

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec3, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let results3 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let count3 = results3.len();

        // Create an IVF-PQ index on the vec column
        let index_name = Some("vec_idx".into());
        dataset
            .create_index(
                &["vec"],
                IndexType::Vector,
                index_name.clone(),
                &VectorIndexParams {
                    metric_type: DistanceType::L2,
                    stages: vec![
                        StageParams::Ivf(IvfBuildParams {
                            max_iters: 2,
                            num_partitions: Some(2),
                            sample_rate: 2,
                            ..Default::default()
                        }),
                        StageParams::PQ(PQBuildParams {
                            max_iters: 2,
                            num_sub_vectors: 2,
                            ..Default::default()
                        }),
                    ],
                    version: crate::index::vector::IndexFileVersion::V3,
                    skip_transpose: false,
                    runtime_hints: Default::default(),
                },
                false,
            )
            .await
            .unwrap();
        let indices = dataset.load_indices().await.unwrap();
        let original_index = indices.iter().find(|idx| idx.name == "vec_idx").unwrap();

        // Run compaction with deferred index remapping
        let options = CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        };

        let metrics = compact_files(&mut dataset, options, None).await.unwrap();
        assert!(metrics.fragments_removed > 0);
        assert!(metrics.fragments_added > 0);

        // Verify the index UUID is unchanged (it should not be remapped yet)
        let Some(current_index) = dataset.load_index_by_name("vec_idx").await.unwrap() else {
            panic!("vec index must be available");
        };
        assert_eq!(current_index.uuid, original_index.uuid);

        // Verify that KNN searches still work correctly and return the same counts
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec1, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let new_results1 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(new_results1.len(), count1);

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec2, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let new_results2 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(new_results2.len(), count2);

        scanner = dataset.scan();
        scanner.nearest("vec", &query_vec3, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let new_results3 = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(new_results3.len(), count3);

        // Verify that after index creation and compaction, scan uses vector index scan
        let mut scanner = dataset.scan();
        scanner.nearest("vec", &query_vec1, 10).unwrap();
        scanner.project::<String>(&[]).unwrap().with_row_id();
        let plan = scanner.explain_plan(false).await.unwrap();
        assert!(
            plan.contains("ANNSubIndex"),
            "Expected vector index scan in plan: {}",
            plan
        );
        assert!(
            !plan.contains("LanceScan"),
            "Expected no fragment scan in plan: {}",
            plan
        );
    }

    #[tokio::test]
    async fn test_default_compaction_planner() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let schema = data.schema();

        // Create dataset with multiple small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], schema.clone());
        let write_params = WriteParams {
            max_rows_per_file: 2000,
            ..Default::default()
        };
        let dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.get_fragments().len(), 5);

        // Test default planner
        let options = CompactionOptions {
            target_rows_per_fragment: 5000,
            materialize_deletions_threshold: 2.0,
            ..Default::default()
        };

        let planner = DefaultCompactionPlanner::new(options);
        let plan = planner.plan(&dataset).await.unwrap();

        // Should create tasks to compact small fragments
        assert!(!plan.tasks.is_empty());
        assert_eq!(plan.read_version, dataset.manifest.version);
        // make sure options.validate() worked
        assert!(!plan.options.materialize_deletions);
    }

    #[test]
    fn test_from_dataset_config() {
        let config = HashMap::from([
            (
                "lance.compaction.target_rows_per_fragment".to_string(),
                "500000".to_string(),
            ),
            (
                "lance.compaction.max_rows_per_group".to_string(),
                "2048".to_string(),
            ),
            (
                "lance.compaction.max_bytes_per_file".to_string(),
                "1000000".to_string(),
            ),
            (
                "lance.compaction.materialize_deletions".to_string(),
                "false".to_string(),
            ),
            (
                "lance.compaction.materialize_deletions_threshold".to_string(),
                "0.25".to_string(),
            ),
            (
                "lance.compaction.defer_index_remap".to_string(),
                "true".to_string(),
            ),
            (
                "lance.compaction.batch_size".to_string(),
                "4096".to_string(),
            ),
            (
                "lance.compaction.compaction_mode".to_string(),
                "try_binary_copy".to_string(),
            ),
            (
                "lance.compaction.binary_copy_read_batch_bytes".to_string(),
                "8388608".to_string(),
            ),
        ]);

        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        assert_eq!(opts.target_rows_per_fragment, 500_000);
        assert_eq!(opts.max_rows_per_group, 2048);
        assert_eq!(opts.max_bytes_per_file, Some(1_000_000));
        assert!(!opts.materialize_deletions);
        assert!((opts.materialize_deletions_threshold - 0.25).abs() < f32::EPSILON);
        assert!(opts.defer_index_remap);
        assert_eq!(opts.batch_size, Some(4096));
        assert_eq!(opts.compaction_mode, Some(CompactionMode::TryBinaryCopy));
        assert_eq!(opts.binary_copy_read_batch_bytes, Some(8_388_608));
    }

    #[test]
    fn test_from_dataset_config_empty() {
        let config = HashMap::new();
        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        let defaults = CompactionOptions::default();
        assert_eq!(
            opts.target_rows_per_fragment,
            defaults.target_rows_per_fragment
        );
        assert_eq!(opts.max_rows_per_group, defaults.max_rows_per_group);
        assert_eq!(opts.max_bytes_per_file, defaults.max_bytes_per_file);
        assert_eq!(opts.materialize_deletions, defaults.materialize_deletions);
        assert_eq!(
            opts.materialize_deletions_threshold,
            defaults.materialize_deletions_threshold
        );
        assert_eq!(opts.defer_index_remap, defaults.defer_index_remap);
        assert_eq!(opts.batch_size, defaults.batch_size);
        assert_eq!(opts.compaction_mode, defaults.compaction_mode);
        assert_eq!(
            opts.binary_copy_read_batch_bytes,
            defaults.binary_copy_read_batch_bytes
        );
    }

    #[test]
    fn test_from_dataset_config_partial() {
        let config = HashMap::from([(
            "lance.compaction.target_rows_per_fragment".to_string(),
            "500000".to_string(),
        )]);

        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        assert_eq!(opts.target_rows_per_fragment, 500_000);
        // Other fields should remain at defaults
        let defaults = CompactionOptions::default();
        assert_eq!(opts.max_rows_per_group, defaults.max_rows_per_group);
        assert_eq!(opts.max_bytes_per_file, defaults.max_bytes_per_file);
        assert_eq!(opts.materialize_deletions, defaults.materialize_deletions);
        assert_eq!(opts.defer_index_remap, defaults.defer_index_remap);
        assert_eq!(opts.batch_size, defaults.batch_size);
        assert_eq!(opts.compaction_mode, defaults.compaction_mode);
        assert_eq!(
            opts.binary_copy_read_batch_bytes,
            defaults.binary_copy_read_batch_bytes
        );
    }

    #[test]
    fn test_from_dataset_config_ignores_other_keys() {
        let config = HashMap::from([
            (
                "lance.compaction.target_rows_per_fragment".to_string(),
                "500000".to_string(),
            ),
            (
                "lance.auto_cleanup.interval".to_string(),
                "3600".to_string(),
            ),
            ("some.other.key".to_string(), "value".to_string()),
        ]);

        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        assert_eq!(opts.target_rows_per_fragment, 500_000);
    }

    #[test]
    fn test_from_dataset_config_invalid_value() {
        let config = HashMap::from([(
            "lance.compaction.target_rows_per_fragment".to_string(),
            "not_a_number".to_string(),
        )]);

        let result = CompactionOptions::from_dataset_config(&config);
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("target_rows_per_fragment"));
        assert!(err_msg.contains("not_a_number"));
    }

    #[test]
    fn test_from_dataset_config_invalid_bool() {
        let config = HashMap::from([(
            "lance.compaction.materialize_deletions".to_string(),
            "yes".to_string(),
        )]);

        let result = CompactionOptions::from_dataset_config(&config);
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("materialize_deletions"));
        assert!(err_msg.contains("yes"));
    }

    #[test]
    fn test_from_dataset_config_unknown_compaction_key() {
        // Unknown keys should be ignored (with a warning) for forwards compatibility
        let config = HashMap::from([(
            "lance.compaction.unknown_key".to_string(),
            "value".to_string(),
        )]);

        let opts = CompactionOptions::from_dataset_config(&config).unwrap();
        // Should return defaults since the unknown key is skipped
        let defaults = CompactionOptions::default();
        assert_eq!(
            opts.target_rows_per_fragment,
            defaults.target_rows_per_fragment
        );
    }

    #[test]
    fn test_from_dataset_config_invalid_compaction_mode() {
        let config = HashMap::from([(
            "lance.compaction.compaction_mode".to_string(),
            "invalid_mode".to_string(),
        )]);

        let result = CompactionOptions::from_dataset_config(&config);
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("invalid_mode"));
    }

    #[test]
    fn test_apply_dataset_config_overrides() {
        let config = HashMap::from([(
            "lance.compaction.target_rows_per_fragment".to_string(),
            "500000".to_string(),
        )]);

        let mut opts = CompactionOptions {
            max_rows_per_group: 4096,
            ..Default::default()
        };
        opts.apply_dataset_config(&config).unwrap();

        // Config value should be applied
        assert_eq!(opts.target_rows_per_fragment, 500_000);
        // Explicitly set value should be preserved (config didn't have this key)
        assert_eq!(opts.max_rows_per_group, 4096);
    }

    #[test]
    fn test_apply_dataset_config_overwrites_matching_field() {
        let config = HashMap::from([(
            "lance.compaction.max_rows_per_group".to_string(),
            "2048".to_string(),
        )]);

        let mut opts = CompactionOptions {
            max_rows_per_group: 4096,
            ..Default::default()
        };
        opts.apply_dataset_config(&config).unwrap();

        // Config value should overwrite the pre-set value
        assert_eq!(opts.max_rows_per_group, 2048);
    }

    #[tokio::test]
    async fn test_max_source_fragments() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let schema = data.schema();

        // Create 10 small fragments (100 rows each) via 10 appends
        let write_params = WriteParams {
            max_rows_per_file: 100,
            ..Default::default()
        };
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.slice(0, 100))], schema.clone()),
            test_uri,
            Some(write_params.clone()),
        )
        .await
        .unwrap();
        for i in 1..10 {
            let mut append_params = write_params.clone();
            append_params.mode = WriteMode::Append;
            Dataset::write(
                RecordBatchIterator::new(vec![Ok(data.slice(i * 100, 100))], schema.clone()),
                test_uri,
                Some(append_params),
            )
            .await
            .unwrap();
        }

        let dataset = Dataset::open(test_uri).await.unwrap();
        assert_eq!(dataset.get_fragments().len(), 10);

        // Plan without limit - all 10 fragments should be candidates.
        // Use a target that splits the 10 fragments into multiple tasks.
        let opts_no_limit = CompactionOptions {
            target_rows_per_fragment: 250,
            ..Default::default()
        };
        let plan_all = plan_compaction(&dataset, &opts_no_limit).await.unwrap();
        let total_source_frags: usize = plan_all.tasks().iter().map(|t| t.fragments.len()).sum();
        assert_eq!(total_source_frags, 10);
        assert!(
            plan_all.num_tasks() > 2,
            "need multiple tasks to test bounding, got {}",
            plan_all.num_tasks()
        );

        // Plan with max_source_fragments=4 should include tasks covering
        // at most 4 source fragments.
        let opts_bounded = CompactionOptions {
            target_rows_per_fragment: 250,
            max_source_fragments: Some(4),
            ..Default::default()
        };
        let plan_bounded = plan_compaction(&dataset, &opts_bounded).await.unwrap();
        let bounded_source_frags: usize =
            plan_bounded.tasks().iter().map(|t| t.fragments.len()).sum();
        assert!(
            bounded_source_frags <= 4,
            "expected at most 4 source fragments, got {bounded_source_frags}"
        );
        assert!(
            bounded_source_frags > 0,
            "expected at least 1 source fragment in bounded plan"
        );
        assert!(
            plan_bounded.num_tasks() < plan_all.num_tasks(),
            "bounded plan ({}) should have fewer tasks than unbounded ({})",
            plan_bounded.num_tasks(),
            plan_all.num_tasks()
        );

        // Execute bounded compaction incrementally
        let mut dataset = dataset;
        compact_files(&mut dataset, opts_bounded, None)
            .await
            .unwrap();
        let after_first = dataset.get_fragments().len();
        assert!(
            after_first < 10,
            "expected fewer than 10 fragments after first compaction, got {after_first}"
        );
        assert!(
            after_first > 1,
            "expected partial compaction (not fully compacted), got {after_first}"
        );

        // Run again to make more progress
        let opts_bounded = CompactionOptions {
            target_rows_per_fragment: 250,
            max_source_fragments: Some(4),
            ..Default::default()
        };
        compact_files(&mut dataset, opts_bounded, None)
            .await
            .unwrap();
        let after_second = dataset.get_fragments().len();
        assert!(
            after_second <= after_first,
            "expected progress: {after_second} should be <= {after_first}"
        );
    }

    #[tokio::test]
    async fn test_compaction_uses_manifest_config() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        let data = sample_data();
        let schema = data.schema();

        // Create dataset with small fragments
        let reader = RecordBatchIterator::new(vec![Ok(data.clone())], schema.clone());
        let write_params = WriteParams {
            max_rows_per_file: 2000,
            ..Default::default()
        };
        let mut dataset = Dataset::write(reader, test_uri, Some(write_params))
            .await
            .unwrap();

        assert_eq!(dataset.get_fragments().len(), 5);

        // Set compaction config in manifest
        dataset
            .update_config([
                ("lance.compaction.target_rows_per_fragment", "5000"),
                ("lance.compaction.materialize_deletions_threshold", "2.0"),
            ])
            .await
            .unwrap();

        // Build options from the dataset config (as the bindings do)
        let opts = CompactionOptions::from_dataset_config(&dataset.manifest.config).unwrap();
        assert_eq!(opts.target_rows_per_fragment, 5000);
        assert!((opts.materialize_deletions_threshold - 2.0).abs() < f32::EPSILON);

        // Verify the config flows through plan_compaction
        let plan = plan_compaction(&dataset, &opts).await.unwrap();
        assert!(!plan.tasks.is_empty());
        assert_eq!(plan.options.target_rows_per_fragment, 5000);
        // validate() should have turned off materialize_deletions since threshold >= 1.0
        assert!(!plan.options.materialize_deletions);
    }

    // check_rewrite_txn takes the (None, Some(_)) branch when a Rewrite with
    // defer_index_remap=true is committed against a previously committed
    // CreateIndex, declaring COMPATIBLE without verifying that the Rewrite's
    // FRI groups don't straddle the CreateIndex's fragment bitmap. When a
    // group mixes indexed and unindexed fragments, commit succeeds and later
    // queries fail at load_indices with "split of indexed and non-indexed
    // data".
    #[tokio::test]
    async fn test_rewrite_fri_vs_create_index_conflict() {
        use crate::index::DatasetIndexExt;
        use crate::index::vector::VectorIndexParams;
        use futures::TryStreamExt;
        use lance_datagen::{BatchCount, Dimension, RowCount, array, gen_batch};
        use lance_index::IndexType;
        use lance_linalg::distance::MetricType;

        async fn append_fragment(uri: &str, rows: u64) -> Dataset {
            let reader = gen_batch()
                .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
                .into_reader_rows(RowCount::from(rows), BatchCount::from(1));
            let params = WriteParams {
                max_rows_per_file: rows as usize,
                mode: WriteMode::Append,
                ..Default::default()
            };
            Dataset::write(reader, uri, Some(params)).await.unwrap()
        }

        let tmpdir = TempStrDir::default();
        let uri = format!("file://{}", tmpdir.as_str());

        // frag0 (256 rows) with a base IVF index.
        let reader = gen_batch()
            .col("vec", array::rand_vec::<Float32Type>(Dimension::from(16)))
            .into_reader_rows(RowCount::from(256), BatchCount::from(1));
        let mut dataset = Dataset::write(
            reader,
            &uri,
            Some(WriteParams {
                max_rows_per_file: 256,
                mode: WriteMode::Overwrite,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        let index_params = VectorIndexParams::ivf_pq(2, 8, 2, MetricType::L2, 50);
        dataset
            .create_index(&["vec"], IndexType::Vector, None, &index_params, true)
            .await
            .unwrap();

        // Append frag1 (unindexed), snapshot a stale handle pointing here,
        // then append frag2 (also unindexed).
        dataset = append_fragment(&uri, 64).await;
        let mut stale = dataset.clone();
        dataset = append_fragment(&uri, 64).await;

        // Plan + execute compaction of frag1+frag2 with deferred remap.
        let options = CompactionOptions {
            defer_index_remap: true,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset, &options).await.unwrap();
        assert!(!plan.tasks.is_empty());
        let snapshot = dataset.clone();
        let completed: Vec<RewriteResult> = futures::stream::iter(plan.tasks.into_iter())
            .map(|task| rewrite_files(Cow::Borrowed(&snapshot), task, &options))
            .buffer_unordered(1)
            .try_collect()
            .await
            .unwrap();

        // optimize_indices on the stale handle indexes frag1 only (frag2
        // didn't exist at that version), commits as CreateIndex. `dataset`
        // stays at its pre-optimize version so the Rewrite commit has to
        // conflict-check against this CreateIndex.
        stale
            .optimize_indices(&lance_index::optimize::OptimizeOptions::append())
            .await
            .unwrap();

        // Commit the pre-executed Rewrite. The FRI group [frag1, frag2]
        // straddles the new CreateIndex bitmap (frag1 indexed, frag2 not), so
        // check_rewrite_txn must reject this as a retryable conflict rather
        // than letting it commit into a broken state that fails queries.
        let err = commit_compaction(
            &mut dataset,
            completed,
            Arc::new(DatasetIndexRemapperOptions::default()),
            &options,
        )
        .await
        .expect_err("commit should fail with retryable conflict");
        assert!(
            matches!(err, Error::RetryableCommitConflict { .. }),
            "unexpected error: {err}"
        );
    }

    /// Reproduce the distributed-compaction concurrent-delete data-resurrection bug.
    ///
    /// In the distributed (Spark) path the caller opens **two separate** `Dataset` handles:
    ///
    /// 1. dataset_plan  — used for `plan_compaction` (version = V)
    /// 2. dataset_commit — opened **fresh** for `commit_compaction` (version = V+N)
    ///
    /// Because `commit_compaction` builds the `Rewrite` transaction with
    /// `dataset.manifest.version` (= V+N), `load_and_sort_new_transactions` only
    /// scans versions after V+N and finds nothing.  Any concurrent DELETE that
    /// happened between V and V+N is silently ignored, causing the deleted rows to
    /// reappear in the compacted fragment.
    ///
    /// After the fix, `commit_compaction` uses `min(tasks.read_version)` (= V) as
    /// the transaction `read_version`, so the conflict checker correctly loads and
    /// rejects the DELETE, returning a retryable conflict error instead of silently
    /// resurrecting data.
    #[tokio::test]
    async fn test_distributed_compact_concurrent_delete_no_resurrection() {
        let test_dir = TempStrDir::default();
        let test_uri = &test_dir;

        // Write 4 fragments × 1 000 rows each (a=0..4000).
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let data = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..4_000))],
        )
        .unwrap();
        let mut dataset_plan = Dataset::write(
            RecordBatchIterator::new(vec![Ok(data)], schema.clone()),
            test_uri,
            Some(WriteParams {
                max_rows_per_file: 1_000,
                ..Default::default()
            }),
        )
        .await
        .unwrap();

        assert_eq!(dataset_plan.manifest.version, 1);
        assert_eq!(dataset_plan.get_fragments().len(), 4);

        // ── Step 1: plan compaction at version V=1 ───────────────────────────────
        let options = CompactionOptions {
            target_rows_per_fragment: 10_000,
            ..Default::default()
        };
        let plan = plan_compaction(&dataset_plan, &options).await.unwrap();
        assert_eq!(plan.tasks().len(), 1, "expected one compaction task");

        // ── Step 2: execute tasks (simulating distributed executors at V=1) ──────
        // Clone dataset_plan so the closure can own its copy while the original
        // remains available for the concurrent DELETE in Step 3.
        let dataset_for_tasks = dataset_plan.clone();
        let results: Vec<RewriteResult> = futures::stream::iter(plan.compaction_tasks())
            .then(|task| {
                let ds = dataset_for_tasks.clone();
                async move {
                    // Executors open the dataset at the planned read_version
                    task.execute(&ds).await.unwrap()
                }
            })
            .collect()
            .await;
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].read_version, 1,
            "tasks must carry read_version=1"
        );

        // ── Step 3: concurrent DELETE commits at V=2 ─────────────────────────────
        // Delete rows where a < 1000 (the first 1 000 rows in fragment 0).
        dataset_plan.delete("a < 1000").await.unwrap();
        assert_eq!(dataset_plan.manifest.version, 2);

        // ── Step 4: the Spark driver opens a *fresh* dataset (latest = V=2) ──────
        // This is exactly what OptimizeExec.scala does for commitCompaction.
        let mut dataset_commit = Dataset::open(test_uri).await.unwrap();
        assert_eq!(
            dataset_commit.manifest.version, 2,
            "fresh dataset must be at the post-delete version"
        );

        // ── Step 5: commit_compaction with the stale results ─────────────────────
        let commit_result = commit_compaction(
            &mut dataset_commit,
            results,
            Arc::new(IgnoreRemap::default()),
            &options,
        )
        .await;

        // ── Step 6: assert correct behaviour ─────────────────────────────────────
        // BEFORE fix: commit_result is Ok(…) and the deleted rows are resurrected.
        // AFTER  fix: commit_result is Err(retryable conflict), protecting data integrity.
        assert!(
            commit_result.is_err(),
            "commit_compaction must fail with a conflict error when a concurrent \
             DELETE touched the same fragments; got Ok instead — deleted rows were \
             silently resurrected"
        );
        let err_msg = commit_result.unwrap_err().to_string();
        assert!(
            err_msg.contains("retryable")
                || err_msg.contains("conflict")
                || err_msg.contains("preempted"),
            "expected a retryable conflict error, got: {err_msg}"
        );

        // The on-disk table must still reflect the DELETE (a < 1000 remains absent).
        let latest = Dataset::open(test_uri).await.unwrap();
        let row_count = latest
            .count_rows(Some("a < 1000".to_string()))
            .await
            .unwrap();
        assert_eq!(
            row_count, 0,
            "rows deleted before compaction must not be resurrected; found {row_count}"
        );
    }

    #[tokio::test]
    async fn test_compaction_plan_early_termination_max_source_fragments() {
        use lance_datagen::generator::array;

        let dataset = lance_datagen::BatchGeneratorBuilder::new()
            .col("key", array::cycle::<Int32Type>((0..100).collect()))
            .into_ram_dataset(FragmentCount::from(100), FragmentRowCount::from(100))
            .await
            .unwrap();

        assert_eq!(dataset.get_fragments().len(), 100);

        // With target_rows_per_fragment=250, 100 small fragments (100 rows each)
        // split into multiple tasks, so max_source_fragments can truncate.
        let unlimited_plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                target_rows_per_fragment: 250,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let unlimited_frags: usize = unlimited_plan.tasks.iter().map(|t| t.fragments.len()).sum();
        assert_eq!(unlimited_frags, 100);
        assert!(
            unlimited_plan.num_tasks() > 1,
            "need multiple tasks to test bounding"
        );

        let limited_plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                target_rows_per_fragment: 250,
                max_source_fragments: Some(20),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let limited_frags: usize = limited_plan.tasks.iter().map(|t| t.fragments.len()).sum();
        assert!(
            limited_frags <= 20,
            "expected at most 20 candidate fragments, got {}",
            limited_frags
        );
        assert!(
            limited_frags > 0,
            "expected at least 1 candidate fragment, got 0"
        );
        assert!(
            limited_frags < unlimited_frags,
            "limited plan should have fewer fragments than unlimited"
        );
    }

    #[tokio::test]
    async fn test_compaction_plan_early_termination_max_compaction_bytes() {
        use lance_datagen::generator::array;

        let dataset = lance_datagen::BatchGeneratorBuilder::new()
            .col("key", array::cycle::<Int32Type>((0..100).collect()))
            .into_ram_dataset(FragmentCount::from(100), FragmentRowCount::from(100))
            .await
            .unwrap();

        let unlimited_plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                target_rows_per_fragment: 250,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let unlimited_frags: usize = unlimited_plan.tasks.iter().map(|t| t.fragments.len()).sum();
        assert!(
            unlimited_frags > 0,
            "need at least some candidate fragments"
        );

        let first_task_bytes: u64 = unlimited_plan.tasks[0]
            .fragments
            .iter()
            .flat_map(|f| f.files.iter())
            .filter_map(|f| f.file_size_bytes.get())
            .map(|sz| sz.get())
            .sum();

        let limited_plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                target_rows_per_fragment: 250,
                max_compaction_bytes: Some(first_task_bytes),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let limited_frags: usize = limited_plan.tasks.iter().map(|t| t.fragments.len()).sum();

        assert!(
            limited_frags > 0,
            "expected at least 1 candidate fragment, got 0"
        );
        assert!(
            limited_frags < unlimited_frags,
            "limited plan should have fewer fragments than unlimited ({} vs {})",
            limited_frags,
            unlimited_frags
        );
    }

    #[tokio::test]
    async fn test_compaction_plan_no_budget_unlimited() {
        use lance_datagen::generator::array;

        let dataset = lance_datagen::BatchGeneratorBuilder::new()
            .col("key", array::cycle::<Int32Type>((0..100).collect()))
            .into_ram_dataset(FragmentCount::from(50), FragmentRowCount::from(100))
            .await
            .unwrap();

        let plan = plan_compaction(
            &dataset,
            &CompactionOptions {
                target_rows_per_fragment: 1_000_000,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        let total_frags: usize = plan.tasks.iter().map(|t| t.fragments.len()).sum();
        assert_eq!(
            total_frags, 50,
            "without budget, all fragments should be planned"
        );
    }
}
