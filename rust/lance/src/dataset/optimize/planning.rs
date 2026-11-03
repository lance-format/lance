// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compaction candidate selection and distributed task planning.
//!
//! This module owns the serializable [`CompactionOptions`], [`CompactionPlan`],
//! and [`TaskData`] contracts, but never rewrites data or publishes a manifest.
//! Tests live in `optimize/tests/planning.rs`.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use lance_core::{Error, Result};
use lance_file::version::{ConcreteFileVersion, LanceFileVersion};
use lance_index::is_system_index;
use lance_index::metrics::NoOpMetricsCollector;
use lance_table::format::overlay::TOMBSTONE_FIELD_ID;
use lance_table::format::{
    Fragment, IndexMetadata, ROW_CREATED_AT_VERSION_FIELD_ID, ROW_ID_FIELD_ID,
    ROW_LAST_UPDATED_AT_VERSION_FIELD_ID,
};
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::RewriteResult;
use super::remapping::IndexRemapperOptions;
use crate::Dataset;
use crate::dataset::fragment::FileFragment;
use crate::dataset::index::{
    load_indices_for_remapping, vector_segment_merge_groups_for_compaction,
};
use crate::dataset::versions;
use crate::index::frag_reuse_with_stable_row_ids::ensure_frag_reuse_recording_allowed;
use crate::index::vector::details::needs_vector_details_inference;
use crate::index::{
    loading::{DatasetIndexInternalExt, load_all_indices},
    segment::{index_is_usable, segment_has_vector_details},
};

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

/// Controls how the old-to-new row-address mapping is built when remapping
/// indices during compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IndexRemapMode {
    /// Store a compact remap and compute row-address mappings during lookup.
    ///
    /// Best for large compactions where peak memory is the constraint. Uses
    /// less memory, but each lookup does extra bitmap/range computation.
    Compact,
    /// Store the full row-address remap in memory for fast direct lookups.
    ///
    /// Best when the remap fits comfortably in memory and remap speed is the
    /// priority. Uses more peak memory because every rewritten/deleted row has
    /// a materialized mapping entry.
    #[default]
    Direct,
}

impl TryFrom<&str> for IndexRemapMode {
    type Error = Error;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "compact" => Ok(Self::Compact),
            "direct" => Ok(Self::Direct),
            _ => Err(Error::invalid_input(format!(
                "Invalid index remap mode \"{}\". Valid values: \"compact\", \"direct\"",
                value
            ))),
        }
    }
}

/// Options to be passed to [`crate::dataset::optimize::compact_files`].
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
    /// Max number of bytes per output data file.
    ///
    /// By default, this only affects how selected fragments are rewritten.
    /// With [`DefaultCompactionPlanner::with_byte_aware_consolidation`], the
    /// effective limit also gates ordinary consolidation when the fragment's
    /// main-file size is reliable: one data file, without deletions, overlays,
    /// or obsolete columns. In that case, fragments at or above 95% of the
    /// limit are left out of ordinary consolidation, while forced deletion,
    /// overlay, and dead-column rewrites remain eligible. Unknown and complex
    /// layouts retain row-only candidacy.
    ///
    /// If not specified then the default (see [`crate::dataset::WriteParams`]) will be used.
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
    /// Must be greater than 0. Defaults to the number of compute-intensive CPUs.
    /// Not used when running tasks manually using [`plan_compaction`]
    pub num_threads: Option<usize>,
    /// The batch size to use when scanning the input fragments.  If not
    /// specified then the default (see
    /// [`crate::dataset::Scanner::batch_size`]) will be used.
    pub batch_size: Option<usize>,
    /// The number of bytes to allow to queue up in the I/O buffer when scanning
    /// the input fragments.  If not specified then the default (see
    /// [`crate::dataset::Scanner::io_buffer_size`]) will be used.
    ///
    /// Increasing this can avoid a deadlock that occurs when a single batch of
    /// data is larger than the I/O buffer size.
    pub io_buffer_size: Option<u64>,
    /// Whether to defer remapping indices during compaction. If true, indices will
    /// not be remapped during this compaction operation. Instead, the fragment reuse index
    /// is updated and will be used to perform remapping later.
    ///
    /// With stable row IDs, every rewritten group's row-address moves are recorded,
    /// indexed or not. Every index must store row addresses and apply the fragment reuse
    /// index, and keeps its coverage through it. Not yet supported in release builds.
    ///
    /// When this is false (eager, the default) and the dataset uses stable row ids then
    /// the behavior depends on the index.  Some legacy indexes store row ids.  These
    /// indexes remain valid and their fragment bitmaps are updated.  Modern indexes
    /// store row addresses and these indexes need to be remapped as if we were not using
    /// stable row ids. An existing fragment reuse index does not by itself enable recording.
    ///
    /// Eager remapping is rejected while a fragment reuse index still has pending
    /// moves from an earlier deferred compaction. Continue deferring the remap, or
    /// explicitly remap the indices and clean up the fragment reuse index first.
    pub defer_index_remap: bool,
    /// How the old-to-new row-address mapping used to remap indices is built.
    /// Defaults to [`IndexRemapMode::Direct`].
    #[serde(default)]
    pub index_remap_mode: IndexRemapMode,
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
    /// tasks are included in the plan until adding the next task would exceed
    /// this limit. This allows for incremental compaction (e.g., compact 20
    /// fragments at a time).
    /// Defaults to `None` (no limit, all eligible fragments are compacted).
    pub max_source_fragments: Option<usize>,
    /// Maximum number of source rows to compact in a single run. Rows are
    /// counted as live rows (physical rows minus soft-deleted rows). When
    /// set, tasks are included in the plan until adding the next task would
    /// exceed this limit.
    /// Defaults to `None` (no limit).
    pub max_source_rows: Option<usize>,
    /// Maximum number of source bytes to compact in a single run, measured as
    /// the total size of the source fragments' data and overlay files. When
    /// set, tasks are included in the plan until adding the next task would
    /// exceed this limit.
    /// Blob v2 payloads live in separate blob files and are not counted, so
    /// this is not a cap on total compaction I/O for datasets with blob
    /// columns.
    /// Defaults to `None` (no limit).
    pub max_source_bytes: Option<u64>,
    /// Fragment IDs to exclude from compaction planning.
    ///
    /// Excluded fragments act as boundaries between adjacent compaction candidates,
    /// so fragments on opposite sides of an exclusion are never combined into the
    /// same task. IDs that are duplicated or absent from the dataset are ignored.
    /// Defaults to an empty list.
    #[serde(default)]
    pub excluded_fragment_ids: Vec<u32>,
    /// Maximum number of data overlay files a fragment may carry before it is
    /// fully compacted. When set, any fragment with more than this many overlays
    /// is rewritten into a fresh fragment with its overlays (and deletions)
    /// materialized into the base data, dropping the fragment from any index
    /// left stale by those overlays.
    /// Defaults to `Some(10)`. Set to `Some(0)` to compact every fragment that
    /// carries any overlay, or `None` to disable the overlay-count trigger
    /// entirely.
    pub max_overlays_per_fragment: Option<usize>,
    /// Exact data file version for compacted output.
    ///
    /// If omitted, use the dataset's default write version without changing it.
    /// The planner resolves release selectors before distributing tasks.
    /// Targets cannot cross the V1/V2 boundary.
    ///
    /// ```
    /// # use lance::dataset::optimize::CompactionOptions;
    /// # use lance_file::version::LanceFileVersion;
    /// let options = CompactionOptions {
    ///     data_storage_version: Some(LanceFileVersion::V2_2),
    ///     ..Default::default()
    /// };
    /// ```
    pub data_storage_version: Option<LanceFileVersion>,
    /// Transaction properties to store with this commit.
    ///
    /// These key-value pairs are stored in the transaction file
    /// and can be read later to identify the source of the commit
    /// (e.g., job_id for tracking completed compaction jobs).
    #[serde(skip)]
    pub transaction_properties: Option<Arc<HashMap<String, String>>>,
    /// Internal: cached result of `has_eager_remappable_index` for the
    /// dataset state the plan was computed against. Not meant to be set by
    /// callers; `pub` only because `CompactionOptions` is constructed with
    /// struct literal syntax outside this crate, which requires every field
    /// to be visible even when filled in via `..Default::default()`.
    ///
    /// [`DefaultCompactionPlanner::plan`] fills this in once, and
    /// `rewrite_files` reuses it instead of reloading index metadata for
    /// every task. Left as `None` by any options a caller builds directly
    /// (including via [`Default`]), which makes `rewrite_files` fall back to
    /// loading it itself. `commit_compaction` never reads this: it always
    /// reloads live, since its dataset handle can have moved past the plan's
    /// read version by the time it runs.
    #[doc(hidden)]
    #[serde(default)]
    pub has_eager_remappable_index: Option<bool>,

    /// How many bytes to use for buffering column data.
    ///
    /// The budget is divided evenly across columns. The default is 8 MiB per
    /// column.
    /// See [`lance_file::writer::FileWriterOptions::data_cache_bytes`].
    pub data_cache_bytes: Option<u64>,
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
            io_buffer_size: None,
            defer_index_remap: false,
            index_remap_mode: IndexRemapMode::Direct,
            compaction_mode: None,
            enable_binary_copy: false,
            enable_binary_copy_force: false,
            binary_copy_read_batch_bytes: Some(16 * 1024 * 1024),
            max_source_fragments: None,
            max_source_rows: None,
            max_source_bytes: None,
            excluded_fragment_ids: Vec::new(),
            max_overlays_per_fragment: Some(10),
            data_storage_version: None,
            transaction_properties: None,
            has_eager_remappable_index: None,
            data_cache_bytes: None,
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
    /// - `lance.compaction.index_remap_mode`
    /// - `lance.compaction.batch_size`
    /// - `lance.compaction.io_buffer_size`
    /// - `lance.compaction.compaction_mode`
    /// - `lance.compaction.binary_copy_read_batch_bytes`
    /// - `lance.compaction.max_source_fragments`
    /// - `lance.compaction.max_source_rows`
    /// - `lance.compaction.max_source_bytes`
    /// - `lance.compaction.max_overlays_per_fragment`
    /// - `lance.compaction.data_storage_version`
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
                "index_remap_mode" => {
                    self.index_remap_mode = IndexRemapMode::try_from(value.as_str())?;
                }
                "batch_size" => {
                    self.batch_size = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "io_buffer_size" => {
                    self.io_buffer_size = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "compaction_mode" => {
                    self.compaction_mode = Some(CompactionMode::try_from(value.as_str())?);
                }
                "data_storage_version" => {
                    self.data_storage_version = Some(value.parse().map_err(|error| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}': {}",
                            key, value, error
                        ))
                    })?);
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
                "max_source_rows" => {
                    self.max_source_rows = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "max_source_bytes" => {
                    self.max_source_bytes = Some(value.parse().map_err(|_| {
                        Error::invalid_input(format!(
                            "Invalid value for {}: '{}' (expected a non-negative integer)",
                            key, value
                        ))
                    })?);
                }
                "max_overlays_per_fragment" => {
                    // The default is `Some(10)`, so an explicit "none" is the only
                    // way to disable the trigger through the manifest config.
                    self.max_overlays_per_fragment = match value.to_ascii_lowercase().as_str() {
                        "none" => None,
                        _ => Some(value.parse().map_err(|_| {
                            Error::invalid_input(format!(
                                "Invalid value for {}: '{}' (expected a non-negative integer or 'none')",
                                key, value
                            ))
                        })?),
                    };
                }
                _ => {
                    warn!("Ignoring unknown compaction config key: {}", key);
                }
            }
        }
        Ok(())
    }

    pub fn validate(&mut self) -> Result<()> {
        // If threshold is 100%, same as turning off deletion materialization.
        if self.materialize_deletions && self.materialize_deletions_threshold >= 1.0 {
            self.materialize_deletions = false;
        }

        for (name, value) in [
            (
                "max_source_fragments",
                self.max_source_fragments.map(|v| v as u64),
            ),
            ("max_source_rows", self.max_source_rows.map(|v| v as u64)),
            ("max_source_bytes", self.max_source_bytes),
        ] {
            if value == Some(0) {
                return Err(Error::invalid_input(format!(
                    "CompactionOptions::{} must be greater than 0 (use None for no limit)",
                    name
                )));
            }
        }
        Ok(())
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

    pub(super) fn write_version(&self, dataset: &Dataset) -> ConcreteFileVersion {
        self.data_storage_version
            .map(LanceFileVersion::resolve)
            .unwrap_or_else(|| dataset.manifest.data_storage_format.lance_file_format())
    }
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

/// A single group of fragments to compact, which is a view into the compaction
/// plan. We keep the `replace_range` indices so we can map the result of the
/// compact back to the fragments it replaces.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskData {
    /// The fragments to compact.
    pub fragments: Vec<Fragment>,
    /// `Fragment::clustering_id` of every output fragment, chosen by the
    /// planner. `None`, the outputs carry no definition, whatever the inputs
    /// carried: Lance does not infer it from the inputs, since a compaction
    /// may span fragments of different clusterings. The commit refuses an id
    /// whose definition the landing manifest does not hold. Independent of
    /// `output_metadata`: metadata needs no definition.
    ///
    /// Serialized tasks stay readable in both directions, but the output
    /// properties are honoured only by a worker that knows the fields: a
    /// worker built before them rewrites unlabelled outputs with empty
    /// metadata. A planner that hands such tasks to other processes must know
    /// they run this version or newer; the Java binding refuses to export a
    /// task carrying either property for that reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_clustering_id: Option<u32>,
    /// `Fragment::metadata` of every output fragment, replacing whatever the
    /// writer set; never copied from the inputs. Opaque to Lance. Empty, the
    /// outputs carry no metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub output_metadata: HashMap<String, String>,
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

    pub(super) fn tasks(&self) -> &[TaskData] {
        &self.tasks
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
    excluded_fragment_ids: RoaringBitmap,
    can_replace_vector_segments: bool,
    byte_aware_consolidation: bool,
}

impl DefaultCompactionPlanner {
    pub fn new(mut options: CompactionOptions) -> Result<Self> {
        options.validate()?;
        let excluded_fragment_ids = options.excluded_fragment_ids.iter().copied().collect();
        Ok(Self {
            options,
            excluded_fragment_ids,
            can_replace_vector_segments: false,
            byte_aware_consolidation: false,
        })
    }

    /// Leave byte-full fragments out of ordinary consolidation.
    ///
    /// Deletion and overlay rewrites remain eligible. Unknown sizes retain
    /// row-only behavior. See [`super::consolidation`] for measurement details.
    ///
    /// ```
    /// use lance::dataset::optimize::{CompactionOptions, DefaultCompactionPlanner};
    /// let options = CompactionOptions {
    ///     max_bytes_per_file: Some(64 * 1024 * 1024),
    ///     ..Default::default()
    /// };
    /// let planner = DefaultCompactionPlanner::new(options)?
    ///     .with_byte_aware_consolidation();
    /// # Ok::<(), lance::Error>(())
    /// ```
    pub fn with_byte_aware_consolidation(mut self) -> Self {
        self.byte_aware_consolidation = true;
        self
    }

    /// Select the publication contract used by this plan. Without an explicit
    /// remapper, planning preserves physical index boundaries. Distributed callers
    /// using [`crate::dataset::index::DatasetIndexRemapperOptions`] at commit may opt into atomic merges.
    ///
    /// ```
    /// use lance::dataset::index::DatasetIndexRemapperOptions;
    /// use lance::dataset::optimize::{CompactionOptions, DefaultCompactionPlanner};
    /// let planner = DefaultCompactionPlanner::new(CompactionOptions::default())?
    ///     .with_index_remapper(&DatasetIndexRemapperOptions::default());
    /// # Ok::<(), lance::Error>(())
    /// ```
    pub fn with_index_remapper(mut self, remapper: &dyn IndexRemapperOptions) -> Self {
        self.can_replace_vector_segments = remapper.supports_vector_segment_replacement();
        self
    }
}

impl DefaultCompactionPlanner {
    /// Why one fragment is a compaction candidate, if it is: the same rule for
    /// ordinary planning and for caller-chosen groups.
    fn candidacy_of(
        &self,
        fragment: &Fragment,
        metrics: &FragmentMetrics,
        schema_field_ids: &HashSet<i32>,
        is_binary_copy_only: bool,
    ) -> Option<CompactionCandidacy> {
        let over_overlay_limit = self
            .options
            .max_overlays_per_fragment
            .is_some_and(|max| fragment.overlays.len() > max);
        if over_overlay_limit {
            // Too many overlays: fully compact this fragment on its own,
            // regardless of its size or deletion count.
            Some(CompactionCandidacy::CompactItself)
        } else if !is_binary_copy_only
            && pins_dead_user_columns_for_lineage(fragment, schema_field_ids)
        {
            // Only a rewrite moves the row lineage out of a file whose user
            // columns are all gone and lets the file go.
            Some(CompactionCandidacy::CompactItself)
        } else if self.options.materialize_deletions
            && metrics.deletion_percentage() > self.options.materialize_deletions_threshold
        {
            Some(CompactionCandidacy::CompactItself)
        } else if metrics.physical_rows < self.options.target_rows_per_fragment
            && (!self.byte_aware_consolidation
                || super::consolidation::needs_consolidation(
                    metrics.physical_rows as u64,
                    super::consolidation::fragment_file_bytes(fragment, schema_field_ids),
                    self.options.target_rows_per_fragment,
                    self.options.max_bytes_per_file,
                ))
        {
            // Worth compacting only with neighbors, to make a larger fragment.
            Some(CompactionCandidacy::CompactWithNeighbors)
        } else {
            None
        }
    }
}

/// A caller-chosen set of fragments allowed to compact together, and only
/// together, with the properties their outputs carry. A group is a boundary,
/// not an order to rewrite every member: Lance applies its ordinary candidacy
/// inside the group and leaves the rest untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionGroup {
    /// The group's fragments, in ascending id order.
    pub fragments: Vec<Fragment>,
    /// [`TaskData::output_clustering_id`] of every task of the group.
    pub output_clustering_id: Option<u32>,
    /// [`TaskData::output_metadata`] of every task of the group.
    pub output_metadata: HashMap<String, String>,
}

impl DefaultCompactionPlanner {
    /// Plan the compaction of explicitly chosen, mutually disjoint fragment
    /// groups, which need not be adjacent in the manifest. Lance interprets
    /// nothing about why fragments were grouped: the caller chooses the groups
    /// and the output properties, Lance selects and binds the work.
    ///
    /// Inside a group the ordinary rules apply, through the same code as
    /// [`CompactionPlanner::plan`]: a fragment under the target compacts with
    /// its group, one over the target is left alone unless deletions, overlays
    /// or dead columns call for rewriting it by itself, candidates are split at
    /// index boundaries and by output size, and a lone fragment that would
    /// only gain from neighbours is not rewritten. Several small fragments of
    /// one group compact together even when their total stays under the
    /// target. Inputs of different groups are never mixed.
    ///
    /// Outputs replace exactly the selected sources; every other fragment
    /// keeps its id, files and metadata, and the outputs take fresh ids at the
    /// end of the manifest, as any compaction does on this version.
    ///
    /// Refused before anything is written: an empty group, a group out of
    /// ascending id order, a fragment absent from the dataset, named by two
    /// groups, or excluded by [`CompactionOptions::excluded_fragment_ids`]; and
    /// a selected source covered by an index this build cannot read (a member
    /// left alone may be covered). `max_source_fragments`,
    /// `max_source_rows` and `max_source_bytes` bound the selected tasks, in
    /// group order.
    pub async fn plan_groups(
        &self,
        dataset: &Dataset,
        groups: Vec<CompactionGroup>,
    ) -> Result<CompactionPlan> {
        let write_version = self.options.write_version(dataset);
        versions::validate_write_version(
            dataset.manifest.data_storage_format.lance_file_format(),
            write_version,
        )?;
        if self.options.defer_index_remap && dataset.manifest.uses_stable_row_ids() {
            ensure_frag_reuse_recording_allowed(
                &dataset.manifest,
                &load_all_indices(dataset).await?,
            )?;
        }
        let eager_remap_capable =
            if !self.options.defer_index_remap && dataset.manifest.uses_stable_row_ids() {
                Some(has_eager_remappable_index(dataset).await?)
            } else {
                None
            };
        if !self.options.defer_index_remap
            && (!dataset.manifest.uses_stable_row_ids() || eager_remap_capable == Some(true))
        {
            reject_eager_remap_over_undrained_fri(dataset).await?;
        }

        // The manifest's own copy of each fragment is what gets planned, so a
        // caller's stale listing cannot carry outdated files or counts.
        let present: HashMap<u64, FileFragment> = dataset
            .get_fragments()
            .into_iter()
            .map(|fragment| (fragment.id() as u64, fragment))
            .collect();
        let mut named = HashSet::new();
        for group in &groups {
            if group.fragments.is_empty() {
                return Err(Error::invalid_input(
                    "a compaction group must name at least one fragment",
                ));
            }
            if let Some(pair) = group
                .fragments
                .windows(2)
                .find(|pair| pair[0].id >= pair[1].id)
            {
                return Err(Error::invalid_input(format!(
                    "a compaction group's fragments must be in ascending id order: fragment {} precedes fragment {}",
                    pair[0].id, pair[1].id
                )));
            }
            for fragment in &group.fragments {
                if !present.contains_key(&fragment.id) {
                    return Err(Error::invalid_input(format!(
                        "compaction group names fragment {}, which is not in the dataset",
                        fragment.id
                    )));
                }
                if u32::try_from(fragment.id)
                    .is_ok_and(|id| self.excluded_fragment_ids.contains(id))
                {
                    return Err(Error::invalid_input(format!(
                        "compaction group names fragment {}, which is excluded from compaction",
                        fragment.id
                    )));
                }
                if !named.insert(fragment.id) {
                    return Err(Error::invalid_input(format!(
                        "fragment {} is named by more than one compaction group",
                        fragment.id
                    )));
                }
            }
        }
        // Coverage by an index this build cannot read is checked against the
        // sources actually selected below, not against every named member:
        // a member ordinary candidacy leaves alone is never rewritten, so its
        // coverage is no obstacle to the rest of its group.
        let unreadable_coverage =
            if !dataset.manifest.uses_stable_row_ids() && !self.options.defer_index_remap {
                unremappable_index_coverage(dataset).await?
            } else {
                Vec::new()
            };

        let schema_field_ids = dataset
            .schema()
            .fields_pre_order()
            .map(|field| field.id)
            .collect::<HashSet<_>>();
        let is_binary_copy_only = matches!(
            self.options.compaction_mode(),
            CompactionMode::ForceBinaryCopy
        );

        // One candidate bin per group: its members that ordinary planning
        // would compact, in order. A member it would leave alone stays out of
        // the bin and untouched.
        // Metrics for every member of every group, read with the same bounded
        // parallelism as ordinary planning; `buffered` keeps input order, so
        // the results line up with the groups.
        let present = &present;
        let members = groups
            .iter()
            .enumerate()
            .flat_map(|(group_index, group)| {
                group
                    .fragments
                    .iter()
                    .map(move |listed| (group_index, present[&listed.id].clone()))
            })
            .collect::<Vec<_>>();
        let mut measured: Vec<(usize, Fragment, FragmentMetrics)> = futures::stream::iter(members)
            .map(|(group_index, fragment)| async move {
                collect_metrics(&fragment)
                    .await
                    .map(|metrics| (group_index, fragment.metadata, metrics))
            })
            .buffered(dataset.object_store.as_ref().io_parallelism())
            .try_collect()
            .await?;
        measured.reverse();

        let mut bins = Vec::with_capacity(groups.len());
        let mut properties = Vec::with_capacity(groups.len());
        for (group_index, group) in groups.into_iter().enumerate() {
            let mut fragments = Vec::new();
            let mut candidacy = Vec::new();
            let mut row_counts = Vec::new();
            for _ in &group.fragments {
                let Some((measured_group, fragment, metrics)) = measured.pop() else {
                    return Err(Error::internal(
                        "grouped compaction lost a fragment's metrics",
                    ));
                };
                debug_assert_eq!(measured_group, group_index);
                if let Some(reason) =
                    self.candidacy_of(&fragment, &metrics, &schema_field_ids, is_binary_copy_only)
                {
                    candidacy.push(reason);
                    row_counts.push(metrics.num_rows());
                    fragments.push(fragment);
                }
            }
            let len = fragments.len();
            bins.push(CandidateBin {
                fragments,
                pos_range: 0..len,
                candidacy,
                row_counts,
                indices: Vec::new(),
            });
            properties.push((group.output_clustering_id, group.output_metadata));
        }

        // Split each group at index boundaries exactly as ordinary planning
        // splits its candidate bins, so no task straddles an index segment.
        let can_merge_vector_segments = self.can_replace_vector_segments
            && !self.options.defer_index_remap
            && !dataset.manifest.uses_stable_row_ids();
        let candidate_windows = bins
            .iter()
            .filter(|bin| bin.fragments.len() > 1)
            .map(|bin| {
                bin.fragments
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect::<RoaringBitmap>()
            })
            .collect::<Vec<_>>();
        let index_fragmaps =
            load_index_fragmaps(dataset, can_merge_vector_segments, &candidate_windows).await?;
        let indices_containing_frag = |frag_id: u32| {
            index_fragmaps
                .iter()
                .enumerate()
                .filter(|(_, bitmap)| bitmap.contains(frag_id))
                .map(|(pos, _)| pos)
                .collect::<Vec<_>>()
        };

        let mut all_tasks: Vec<(TaskData, usize)> = Vec::new();
        for (bin, (output_clustering_id, output_metadata)) in bins.into_iter().zip(properties) {
            let mut constrained = Vec::new();
            let mut current: Option<CandidateBin> = None;
            for (offset, ((fragment, candidacy), row_count)) in bin
                .fragments
                .into_iter()
                .zip(bin.candidacy)
                .zip(bin.row_counts)
                .enumerate()
            {
                let indices = indices_containing_frag(fragment.id as u32);
                if current
                    .as_ref()
                    .is_some_and(|current| current.indices != indices)
                {
                    constrained.extend(current.take());
                }
                let current = current.get_or_insert_with(|| CandidateBin {
                    fragments: Vec::new(),
                    candidacy: Vec::new(),
                    row_counts: Vec::new(),
                    pos_range: offset..offset,
                    indices,
                });
                current.fragments.push(fragment);
                current.candidacy.push(candidacy);
                current.row_counts.push(row_count);
                current.pos_range.end = offset + 1;
            }
            constrained.extend(current);
            // A split can leave one fragment on its own; it is rewritten only
            // for the reasons ordinary compaction rewrites a fragment alone.
            all_tasks.extend(
                constrained
                    .into_iter()
                    .filter(|bin| !bin.is_noop())
                    .flat_map(|bin| bin.split_for_size(self.options.target_rows_per_fragment))
                    .filter(|bin| !bin.is_noop())
                    .map(|bin| {
                        let live_rows = bin.row_counts.iter().sum();
                        (
                            TaskData {
                                fragments: bin.fragments,
                                output_clustering_id,
                                output_metadata: output_metadata.clone(),
                            },
                            live_rows,
                        )
                    }),
            );
        }

        // Only the selected sources change identity, so `max_source_fragments`
        // bounds them directly: tasks are kept in group order until the next
        // would exceed it, as the row and byte budgets are applied below.
        let all_tasks = if let Some(max_frags) = self.options.max_source_fragments {
            let num_candidate_tasks = all_tasks.len();
            let mut total = 0_usize;
            let mut kept = Vec::with_capacity(all_tasks.len());
            for (task, live_rows) in all_tasks {
                total += task.fragments.len();
                if total > max_frags {
                    break;
                }
                kept.push((task, live_rows));
            }
            if kept.is_empty() && num_candidate_tasks > 0 {
                warn!(
                    "Grouped compaction plan is empty: the first of {num_candidate_tasks} tasks \
                     already exceeds max_source_fragments={max_frags}"
                );
            }
            kept
        } else {
            all_tasks
        };
        let tasks = limit_tasks_to_source_budget(&self.options, dataset.schema(), all_tasks)?;

        // Checked last, over the tasks the budgets kept: only a source that
        // will really be rewritten can refuse the plan, the same set the
        // commit's safeguard checks.
        for (index, covered) in &unreadable_coverage {
            if let Some(fragment) = tasks
                .iter()
                .flat_map(|task| &task.fragments)
                .find(|fragment| u32::try_from(fragment.id).is_ok_and(|id| covered.contains(id)))
            {
                return Err(Error::invalid_input(format!(
                    "compaction group selects fragment {} for rewriting, which is covered by \
                     index '{index}' that this build cannot read and so cannot remap",
                    fragment.id
                )));
            }
        }

        let mut options = self.options.clone();
        options.data_storage_version = Some(write_version.to_selector());
        options.has_eager_remappable_index = eager_remap_capable;
        let mut compaction_plan = CompactionPlan::new(dataset.manifest.version, options);
        compaction_plan.extend_tasks(tasks);
        Ok(compaction_plan)
    }
}

#[async_trait::async_trait]
impl CompactionPlanner for DefaultCompactionPlanner {
    async fn plan(&self, dataset: &Dataset) -> Result<CompactionPlan> {
        let write_version = self.options.write_version(dataset);
        versions::validate_write_version(
            dataset.manifest.data_storage_format.lance_file_format(),
            write_version,
        )?;
        if self.options.defer_index_remap && dataset.manifest.uses_stable_row_ids() {
            ensure_frag_reuse_recording_allowed(
                &dataset.manifest,
                &load_all_indices(dataset).await?,
            )?;
        }
        // Computed at most once here, rather than reloaded by every task in
        // `rewrite_files` and the planner above: see the doc comment on
        // `CompactionOptions::has_eager_remappable_index`.
        let eager_remap_capable =
            if !self.options.defer_index_remap && dataset.manifest.uses_stable_row_ids() {
                Some(has_eager_remappable_index(dataset).await?)
            } else {
                None
            };
        if !self.options.defer_index_remap
            && (!dataset.manifest.uses_stable_row_ids() || eager_remap_capable == Some(true))
        {
            reject_eager_remap_over_undrained_fri(dataset).await?;
        }

        // get_fragments should be returning fragments in sorted order (by id)
        // and fragment ids should be unique
        let fragments = dataset.get_fragments();

        debug_assert!(
            fragments.windows(2).all(|w| w[0].id() < w[1].id()),
            "fragments in manifest are not sorted"
        );
        // Without stable row ids a rewrite moves every row address, so the
        // indices over the rewritten fragments have to be remapped in the same
        // commit, and an index this build cannot open cannot be remapped. With
        // stable row ids this never applies: every index's coverage is fixed
        // up from manifest metadata alone (follow the data, or drop the
        // rewritten fragments), which needs no reader for the index at all.
        // An unremappable index's fragments join the caller's own exclusions
        // and are left uncompacted - taking the same path, which terminates
        // the current bin rather than letting the candidates on either side
        // of the gap be planned together.
        let mut excluded_fragment_ids = self.excluded_fragment_ids.clone();
        if !dataset.manifest.uses_stable_row_ids() && !self.options.defer_index_remap {
            let unremappable = unremappable_index_coverage(dataset)
                .await?
                .into_iter()
                .fold(RoaringBitmap::new(), |mut covered, (_, fragments)| {
                    covered |= fragments;
                    covered
                });
            if !unremappable.is_empty() {
                // Otherwise a compaction that plans nothing looks like a
                // compaction that found nothing to do.
                log::info!(
                    "holding {} fragment(s) back from compaction: they are covered by an index \
                     this build cannot read, and so cannot be remapped here",
                    unremappable.len(),
                );
            }
            excluded_fragment_ids |= unremappable;
        }
        let excluded_fragment_ids = &excluded_fragment_ids;

        let mut fragment_metrics = futures::stream::iter(fragments)
            .map(|fragment| async move {
                if u32::try_from(fragment.id())
                    .is_ok_and(|fragment_id| excluded_fragment_ids.contains(fragment_id))
                {
                    Ok(None)
                } else {
                    collect_metrics(&fragment)
                        .await
                        .map(|metrics| Some((fragment.metadata, metrics)))
                }
            })
            .buffered(dataset.object_store.as_ref().io_parallelism());

        let schema_field_ids = dataset
            .schema()
            .fields_pre_order()
            .map(|field| field.id)
            .collect::<HashSet<_>>();
        // Only reencoding reclaims a lineage carrier whose user columns are
        // gone: its fields are not the schema's, so binary copy never applies
        // to it, and under `ForceBinaryCopy` its task would fail the whole run.
        let is_binary_copy_only = matches!(
            self.options.compaction_mode(),
            CompactionMode::ForceBinaryCopy
        );

        let mut candidate_bins: Vec<CandidateBin> = Vec::new();
        let mut current_bin: Option<CandidateBin> = None;
        let mut i = 0;

        while let Some(res) = fragment_metrics.next().await {
            let Some((fragment, metrics)) = res? else {
                // Exclusions preserve adjacency semantics: they terminate the
                // current bin instead of allowing candidates on either side to
                // be planned together.
                if let Some(bin) = current_bin.take() {
                    candidate_bins.push(bin);
                }
                i += 1;
                continue;
            };

            let candidacy =
                self.candidacy_of(&fragment, &metrics, &schema_field_ids, is_binary_copy_only);

            let indices = Vec::new();

            match (candidacy, &mut current_bin) {
                (None, None) => {} // keep searching
                (Some(candidacy), None) => {
                    // Start a new bin
                    current_bin = Some(CandidateBin {
                        fragments: vec![fragment],
                        pos_range: i..(i + 1),
                        candidacy: vec![candidacy],
                        row_counts: vec![metrics.num_rows()],
                        indices,
                    });
                }
                (Some(candidacy), Some(bin)) => {
                    bin.fragments.push(fragment);
                    bin.pos_range.end += 1;
                    bin.candidacy.push(candidacy);
                    bin.row_counts.push(metrics.num_rows());
                }
                (None, Some(_)) => {
                    // Bin is complete
                    candidate_bins.push(current_bin.take().unwrap());
                }
            }

            i += 1;
        }

        // Flush the last bin
        if let Some(bin) = current_bin {
            candidate_bins.push(bin);
        }

        // Candidate windows precede index constraints: the old boundary-constrained
        // plan can be empty even when crossing a compatible boundary would help.
        let candidate_windows = candidate_bins
            .iter()
            .filter(|bin| bin.fragments.len() > 1)
            .map(|bin| {
                bin.fragments
                    .iter()
                    .map(|fragment| fragment.id as u32)
                    .collect::<RoaringBitmap>()
            })
            .collect::<Vec<_>>();
        // Removing a physical index boundary is safe only when publication remaps
        // and atomically replaces every segment in the resulting merge group.
        let can_merge_vector_segments = self.can_replace_vector_segments
            && !self.options.defer_index_remap
            && !dataset.manifest.uses_stable_row_ids();
        let index_fragmaps =
            load_index_fragmaps(dataset, can_merge_vector_segments, &candidate_windows).await?;
        let indices_containing_frag = |frag_id: u32| {
            index_fragmaps
                .iter()
                .enumerate()
                .filter(|(_, bitmap)| bitmap.contains(frag_id))
                .map(|(pos, _)| pos)
                .collect::<Vec<_>>()
        };

        let mut constrained_bins = Vec::new();
        for bin in candidate_bins {
            let mut current: Option<CandidateBin> = None;
            for (offset, ((fragment, candidacy), row_count)) in bin
                .fragments
                .into_iter()
                .zip(bin.candidacy)
                .zip(bin.row_counts)
                .enumerate()
            {
                let indices = indices_containing_frag(fragment.id as u32);
                if current
                    .as_ref()
                    .is_some_and(|current| current.indices != indices)
                {
                    constrained_bins.extend(current.take());
                }
                let pos = bin.pos_range.start + offset;
                let current = current.get_or_insert_with(|| CandidateBin {
                    fragments: Vec::new(),
                    candidacy: Vec::new(),
                    row_counts: Vec::new(),
                    pos_range: pos..pos,
                    indices,
                });
                current.fragments.push(fragment);
                current.candidacy.push(candidacy);
                current.row_counts.push(row_count);
                current.pos_range.end = pos + 1;
            }
            constrained_bins.extend(current);
        }

        if log::log_enabled!(log::Level::Debug) {
            // Bound both dimensions of the candidate bin previews.
            const MAX_PREVIEW_BINS: usize = 100;
            const MAX_PREVIEW_ITEMS: usize = 100;
            log::debug!(
                "Compaction constrained_bins: count={}, omitted_bins={}, index_fragmaps={:?}",
                constrained_bins.len(),
                constrained_bins.len().saturating_sub(MAX_PREVIEW_BINS),
                index_fragmaps,
            );
            for (bin_index, bin) in constrained_bins.iter().take(MAX_PREVIEW_BINS).enumerate() {
                let fragment_ids: Vec<_> = bin
                    .fragments
                    .iter()
                    .take(MAX_PREVIEW_ITEMS)
                    .map(|fragment| fragment.id)
                    .collect();
                log::debug!(
                    "Compaction constrained_bins[{}]: pos_range={:?}, fragment_count={}, \
                     fragment_ids_preview={:?}, candidacy_preview={:?}, row_counts_preview={:?}, \
                     index_count={}, indices_preview={:?}, preview_item_limit={}",
                    bin_index,
                    bin.pos_range,
                    bin.fragments.len(),
                    fragment_ids,
                    &bin.candidacy[..bin.candidacy.len().min(MAX_PREVIEW_ITEMS)],
                    &bin.row_counts[..bin.row_counts.len().min(MAX_PREVIEW_ITEMS)],
                    bin.indices.len(),
                    &bin.indices[..bin.indices.len().min(MAX_PREVIEW_ITEMS)],
                    MAX_PREVIEW_ITEMS,
                );
            }
        }

        let all_tasks: Vec<(TaskData, usize)> = constrained_bins
            .into_iter()
            .filter(|bin| !bin.is_noop())
            .flat_map(|bin| bin.split_for_size(self.options.target_rows_per_fragment))
            .map(|bin| {
                let live_rows = bin.row_counts.iter().sum();
                (
                    TaskData {
                        fragments: bin.fragments,
                        output_clustering_id: None,
                        output_metadata: HashMap::new(),
                    },
                    live_rows,
                )
            })
            .collect();

        let tasks = limit_tasks_to_source_budget(&self.options, dataset.schema(), all_tasks)?;

        let mut options = self.options.clone();
        options.data_storage_version = Some(write_version.to_selector());
        options.has_eager_remappable_index = eager_remap_capable;
        let mut compaction_plan = CompactionPlan::new(dataset.manifest.version, options);
        compaction_plan.extend_tasks(tasks);

        Ok(compaction_plan)
    }
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

/// Whether one of `fragment`'s data files is referenced only for the row
/// lineage columns it carries, while the user columns next to them have all
/// been dropped or replaced. `schema_field_ids` holds the ids of the schema's
/// fields.
///
/// Compaction writes spilled lineage into the data file of each fragment it
/// writes. Once every user column in that file is dropped, cast or replaced,
/// the lineage alone keeps the file, and its dead bytes, referenced: cleanup
/// cannot remove it and nothing else rewrites it. A lineage-only file, which
/// binary copy and update write, holds no dead user column and never matches;
/// nor does any file compaction writes, so compacting a fragment that matches
/// cannot leave one that matches again.
pub(super) fn pins_dead_user_columns_for_lineage(
    fragment: &Fragment,
    schema_field_ids: &HashSet<i32>,
) -> bool {
    fragment.files.iter().any(|file| {
        let mut holds_dead_user_column = false;
        let mut holds_lineage = false;
        for field_id in file.fields.iter() {
            if schema_field_ids.contains(field_id) {
                return false;
            }
            match *field_id {
                ROW_ID_FIELD_ID
                | ROW_CREATED_AT_VERSION_FIELD_ID
                | ROW_LAST_UPDATED_AT_VERSION_FIELD_ID => holds_lineage = true,
                TOMBSTONE_FIELD_ID => holds_dead_user_column = true,
                // A user field the schema no longer has: dropping a column
                // leaves the ids of the files it keeps as they are.
                other => holds_dead_user_column |= other >= 0,
            }
        }
        holds_dead_user_column && holds_lineage
    })
}

/// Truncates a planned task list to the configured per-run source budgets
/// (`max_source_fragments`, `max_source_rows`, `max_source_bytes`).
///
/// All configured budgets apply together: tasks are kept, in order, until
/// adding the next task would exceed any one of them. The budgets are hard
/// upper bounds, so if the first task already exceeds one of them the
/// returned plan is empty and a warning is logged, since compaction would
/// otherwise stall silently.
///
/// Each task is paired with the number of live rows in its source fragments.
fn limit_tasks_to_source_budget(
    options: &CompactionOptions,
    schema: &lance_core::datatypes::Schema,
    all_tasks: Vec<(TaskData, usize)>,
) -> Result<Vec<TaskData>> {
    if options.max_source_fragments.is_none()
        && options.max_source_rows.is_none()
        && options.max_source_bytes.is_none()
    {
        return Ok(all_tasks.into_iter().map(|(task, _)| task).collect());
    }

    // Only needed for the bytes budget: files whose fields are all absent
    // from the current schema only back dropped columns, which compaction
    // does not read.
    let schema_field_ids: HashSet<i32> = if options.max_source_bytes.is_some() {
        schema.field_ids().into_iter().collect()
    } else {
        HashSet::new()
    };

    let num_candidate_tasks = all_tasks.len();
    let mut total_fragments = 0_usize;
    let mut total_rows = 0_usize;
    let mut total_bytes = 0_u64;
    let mut tasks = Vec::with_capacity(all_tasks.len());
    for (task, live_rows) in all_tasks {
        total_fragments += task.fragments.len();
        total_rows = total_rows.saturating_add(live_rows);
        if options.max_source_bytes.is_some() {
            total_bytes = total_bytes.saturating_add(task_source_bytes(&task, &schema_field_ids)?);
        }

        let over_budget = options
            .max_source_fragments
            .is_some_and(|max| total_fragments > max)
            || options.max_source_rows.is_some_and(|max| total_rows > max)
            || options
                .max_source_bytes
                .is_some_and(|max| total_bytes > max);
        if over_budget {
            break;
        }

        tasks.push(task);
    }

    if tasks.is_empty() && num_candidate_tasks > 0 {
        warn!(
            "Compaction plan is empty: the first of {} candidate tasks already exceeds a source \
             budget (max_source_fragments={:?}, max_source_rows={:?}, max_source_bytes={:?}); \
             compaction cannot make progress until the budget is raised",
            num_candidate_tasks,
            options.max_source_fragments,
            options.max_source_rows,
            options.max_source_bytes
        );
    }

    Ok(tasks)
}

/// Returns the total size in bytes of a task's source data and overlay files.
///
/// Files whose fields are all absent from `schema_field_ids` only back
/// dropped columns; compaction does not read them, so they are neither
/// counted nor required to have a recorded size.
/// Only sizes recorded in the manifest are used: a missing size is an error
/// rather than a metadata request against object storage, which would turn
/// planning into one round trip per file. Deletion files are not counted.
fn task_source_bytes(task: &TaskData, schema_field_ids: &HashSet<i32>) -> Result<u64> {
    let mut total_bytes = 0_u64;
    for fragment in &task.fragments {
        let overlay_files = fragment.overlays.iter().map(|overlay| &overlay.data_file);
        for data_file in fragment.files.iter().chain(overlay_files) {
            if !data_file
                .fields
                .iter()
                .any(|field_id| schema_field_ids.contains(field_id))
            {
                continue;
            }
            let size = data_file.file_size_bytes.get().ok_or_else(|| {
                Error::invalid_input(format!(
                    "max_source_bytes is set but file '{}' of fragment {} has no size recorded \
                     in the manifest; unset max_source_bytes to compact this dataset",
                    data_file.path, fragment.id
                ))
            })?;
            total_bytes = total_bytes.saturating_add(size.get());
        }
    }
    Ok(total_bytes)
}

#[derive(Debug, Clone)]
pub(super) enum CompactionCandidacy {
    /// Compact the fragment if it has neighbors that are also candidates
    CompactWithNeighbors,
    /// Compact the fragment regardless.
    CompactItself,
}

/// Internal struct used for planning compaction.
pub(super) struct CandidateBin {
    pub fragments: Vec<Fragment>,
    pub pos_range: Range<usize>,
    pub candidacy: Vec<CompactionCandidacy>,
    pub row_counts: Vec<usize>,
    pub indices: Vec<usize>,
}

impl CandidateBin {
    /// Return true if compacting these fragments wouldn't do anything.
    pub(super) fn is_noop(&self) -> bool {
        if self.fragments.is_empty() {
            return true;
        }
        // If there's only one fragment, it's a noop if it's not CompactItself
        if self.fragments.len() == 1 {
            matches!(self.candidacy[0], CompactionCandidacy::CompactWithNeighbors)
        } else {
            false
        }
    }

    /// Split into one or more bins with at least `min_num_rows` in them.
    pub(super) fn split_for_size(self, min_num_rows: usize) -> Vec<Self> {
        let total_rows = self.row_counts.iter().sum::<usize>();
        let mut remaining_rows = total_rows;
        let mut current_rows = 0;
        let mut current_len = 0;
        let mut split_lengths = Vec::new();

        for row_count in &self.row_counts {
            current_rows += *row_count;
            current_len += 1;
            remaining_rows -= *row_count;

            // Only split once the current bin is large enough and there is
            // enough left over to form another worthwhile non-empty bin.
            if current_rows >= min_num_rows && remaining_rows > 0 && remaining_rows >= min_num_rows
            {
                split_lengths.push(current_len);
                current_rows = 0;
                current_len = 0;
            }
        }

        if split_lengths.is_empty() {
            return vec![self];
        }

        let mut bins = Vec::with_capacity(split_lengths.len() + 1);
        let mut fragments = self.fragments.into_iter();
        let mut candidacy = self.candidacy.into_iter();
        let mut row_counts = self.row_counts.into_iter();
        let mut pos_start = self.pos_range.start;

        for bin_len in split_lengths {
            bins.push(Self {
                fragments: fragments.by_ref().take(bin_len).collect(),
                pos_range: pos_start..(pos_start + bin_len),
                candidacy: candidacy.by_ref().take(bin_len).collect(),
                row_counts: row_counts.by_ref().take(bin_len).collect(),
                // By the time we are splitting for size we are done considering indices
                indices: Vec::new(),
            });
            pos_start += bin_len;
        }

        bins.push(Self {
            fragments: fragments.collect(),
            pos_range: pos_start..self.pos_range.end,
            candidacy: candidacy.collect(),
            row_counts: row_counts.collect(),
            indices: self.indices,
        });

        bins
    }
}

pub(super) async fn load_index_fragmaps(
    dataset: &Dataset,
    can_merge_vector_segments: bool,
    candidate_windows: &[RoaringBitmap],
) -> Result<Vec<RoaringBitmap>> {
    // Coverage, not usability: these bitmaps decide the rewrite groups. Under
    // stable row ids `Transaction::recalculate_fragment_bitmap` then rejects any
    // group that splits an index's coverage, and it walks every index the new
    // manifest carries - including the ones this build cannot read. Binning from
    // the filtered view fails that check outright on a dataset holding an index
    // written by a newer Lance. The same bitmaps also decide, through
    // `any_group_indexed`, whether a deferred compaction writes the
    // fragment-reuse index that a build which can read that index needs to
    // repair its coverage.
    let indices = load_all_indices(dataset).await?;
    let mut index_fragmaps = Vec::with_capacity(indices.len());
    let mut segments_by_name = HashMap::<&str, Vec<_>>::new();
    for index in indices.iter().filter(|idx| !is_system_index(idx)) {
        segments_by_name
            .entry(index.name.as_str())
            .or_default()
            .push(index);
    }

    // Apply all manifest-known boundaries before loading any stored models.
    // Eligible segments can potentially share coverage; scalar, opaque and
    // unsupported vector layouts retain their individual boundaries.
    let mut potential_boundaries = Vec::new();
    if can_merge_vector_segments {
        for segments in segments_by_name.values() {
            let mut eligible_coverage = RoaringBitmap::new();
            for segment in segments {
                let coverage = index_fragment_coverage(dataset, segment).await?;
                if crate::dataset::index::is_merge_candidate(segment) {
                    eligible_coverage |= coverage;
                } else {
                    potential_boundaries.push(coverage);
                }
            }
            if !eligible_coverage.is_empty() {
                potential_boundaries.push(eligible_coverage);
            }
        }
    }
    let candidate_windows = crate::dataset::index::constrain_candidate_windows(
        candidate_windows,
        &potential_boundaries,
    );

    let mut visited_names = HashSet::new();
    // System indices (fragment-reuse, mem-wal) don't define data coverage and
    // aren't remapped per rewrite group, so they must not constrain compaction
    // bins -- otherwise deferred compaction's fragment-reuse index repeatedly
    // splits the small-fragment run and they never coalesce.
    for index in indices.iter().filter(|idx| !is_system_index(idx)) {
        if !visited_names.insert(index.name.as_str()) {
            continue;
        }

        let segments = &segments_by_name[index.name.as_str()];
        let merge_groups = if can_merge_vector_segments {
            let (relevant, other): (Vec<_>, Vec<_>) =
                segments.iter().copied().partition(|segment| {
                    segment.fragment_bitmap.as_ref().is_some_and(|coverage| {
                        candidate_windows.iter().any(|window| {
                            !coverage.is_disjoint(window) && !window.is_subset(coverage)
                        })
                    })
                });
            let mut groups = vector_segment_merge_groups_for_compaction(dataset, &relevant).await;
            groups.extend(other.into_iter().map(|segment| vec![segment]));
            groups
        } else {
            segments.iter().map(|segment| vec![*segment]).collect()
        };
        for merge_group in merge_groups {
            let mut group_fragment_bitmap = RoaringBitmap::new();
            for segment in merge_group {
                group_fragment_bitmap |= index_fragment_coverage(dataset, segment).await?;
            }
            index_fragmaps.push(group_fragment_bitmap);
        }
    }
    Ok(index_fragmaps)
}

/// The fragments an index segment covers, reconstructing the coverage of a
/// legacy segment that predates the bitmap from the dataset it was written
/// against.
async fn index_fragment_coverage(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> Result<RoaringBitmap> {
    if let Some(fragment_bitmap) = index.fragment_bitmap.as_ref() {
        return Ok(fragment_bitmap.clone());
    }

    let dataset_at_index = dataset.checkout_version(index.dataset_version).await?;
    // max_fragment_id is inclusive (the highest id); +1 for an exclusive
    // upper bound so the last fragment is covered (None => empty range).
    let frags = 0..dataset_at_index
        .manifest
        .max_fragment_id
        .map_or(0, |m| m + 1);
    let mut coverage = RoaringBitmap::from_sorted_iter(frags).unwrap();
    // Reconstructed in the id space of the version the index was written
    // against, which a later compaction has already moved on from.
    // `load_all_indices` puts a stored bitmap into the current space by running
    // it through the fragment-reuse index and leaves a `None` one alone, so a
    // reconstruction has to take that step itself. Skipping it names the
    // fragments a deferred compaction moved these rows out of, which is a set
    // no rewrite can intersect - the guards below then wave through the rewrite
    // of the fragment the rows actually live in.
    if let Some(frag_reuse_index) = dataset.open_frag_reuse_index(&NoOpMetricsCollector).await? {
        frag_reuse_index.remap_fragment_bitmap(&mut coverage)?;
    }
    Ok(coverage)
}

/// Each index this build has no reader for, by name, and the fragments it covers.
///
/// A rewrite moves every row address in the fragments it touches, and putting an
/// index back in step means opening it. A build that cannot open one cannot
/// remap it, so compacting the fragments it covers would leave it addressing
/// rows that are gone -- worse than the erase this whole path exists to prevent.
/// They are held out of the plan instead, and the rest of the table still
/// compacts.
///
/// Only the non-stable-row-id eager remap path needs this: there, a rewrite
/// group fixes up an index's coverage by remapping it or not at all, so an
/// index this build cannot open is left stranded on deleted fragment ids. On
/// a stable-row-id table every index's coverage is instead fixed up from
/// manifest metadata alone (follow the data, or drop the rewritten
/// fragments), which needs no reader for the index at all, so this check
/// does not apply there; callers only run it for non-stable-row-id tables.
async fn unremappable_index_coverage(dataset: &Dataset) -> Result<Vec<(String, RoaringBitmap)>> {
    let mut coverage = Vec::new();
    for index in load_all_indices(dataset).await?.iter() {
        if index_is_usable(index) {
            continue;
        }
        coverage.push((
            index.name.clone(),
            index_fragment_coverage(dataset, index).await?,
        ));
    }
    Ok(coverage)
}

/// Refuse a plan that rewrites fragments an index this build cannot read covers.
///
/// [`DefaultCompactionPlanner`] keeps those fragments out of the plan, but
/// nothing forces a caller through it: `compact_files_with_planner` takes any
/// planner, [`CompactionPlan`] is public and serializable, and a distributed
/// driver hands [`crate::dataset::optimize::commit_compaction`] results planned elsewhere. Committing such
/// a plan strands the index on fragment ids the rewrite deleted, so the commit
/// boundary refuses it rather than the planner alone. Only relevant without
/// stable row ids; see [`unremappable_index_coverage`].
pub(super) async fn reject_unremappable_rewrite(
    dataset: &Dataset,
    completed_tasks: &[RewriteResult],
) -> Result<()> {
    let rewritten = completed_tasks
        .iter()
        .flat_map(|task| task.original_fragments.iter())
        .filter_map(|fragment| u32::try_from(fragment.id).ok())
        .collect::<RoaringBitmap>();

    for (name, covered) in unremappable_index_coverage(dataset).await? {
        let blocked = covered & &rewritten;
        if !blocked.is_empty() {
            return Err(Error::invalid_input(format!(
                "compaction would rewrite fragment(s) {:?}, which index {:?} covers. This build \
                 has no reader for that index, so it cannot be remapped onto the rewritten \
                 fragments and the commit would leave it addressing rows that no longer exist. \
                 Plan with DefaultCompactionPlanner, which holds those fragments back, set \
                 defer_index_remap, or compact from a build that can read the index.",
                blocked.iter().collect::<Vec<_>>(),
                name,
            )));
        }
    }
    Ok(())
}

pub async fn plan_compaction(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Result<CompactionPlan> {
    let planner = DefaultCompactionPlanner::new(options.clone())?;
    planner.plan(dataset).await
}

/// Whether any non-system index on `dataset` both stores row addresses and
/// its type supports being eagerly remapped onto them
/// (`IndexMetadata::supports_eager_remap`). Metadata-only, no index opened:
/// an index whose type can never remap (ZoneMap, BloomFilter, FM) does not
/// count, since an eager remap would do nothing for it but materialize and
/// then discard an old-to-new move for every rewritten row -- on a large
/// compaction, table-scale memory the rewrite files have already been
/// written to produce.
pub(super) async fn has_eager_remappable_index(dataset: &Dataset) -> Result<bool> {
    let Some(indices) = load_indices_for_remapping(dataset).await? else {
        return Ok(false);
    };
    Ok(indices
        .iter()
        .any(|index| !is_system_index(index) && index.supports_eager_remap()))
}

/// Refuse an eager index remap while the table's fragment reuse index is not
/// fully drained, regardless of whether the table uses stable row ids.
///
/// A remappable index (e.g. BTree) is remapped by reading its persisted
/// pages' stored addresses directly and translating them with this
/// compaction's own old-to-new move map; it never composes that read with a
/// fragment reuse index's pending moves. If the table still carries one -- a
/// prior deferred compaction left addresses that only the FRI translates --
/// remapping now would write those untranslated addresses into the rebuilt
/// index and the manifest would then claim full coverage, silently dropping
/// every row the FRI still had to translate. Explicitly remapping the indices
/// and cleaning up the FRI first, or compacting with `defer_index_remap` set
/// instead, keeps the index's pages in step with its coverage.
pub(super) async fn reject_eager_remap_over_undrained_fri(dataset: &Dataset) -> Result<()> {
    if let Some(fri) = dataset.frag_reuse_index().await?
        && !fri.is_empty()
    {
        return Err(Error::not_supported(
            "Cannot eagerly compact while the fragment reuse index has pending remaps from \
             an earlier compaction. Continue compacting with defer_index_remap=true, or \
             explicitly remap the indices and clean up the fragment reuse index first. \
             optimize_indices alone may leave pending remaps.",
        ));
    }
    Ok(())
}

pub(super) async fn validate_fragment_groups_preserve_index_boundaries<'a>(
    dataset: &Dataset,
    groups: impl Iterator<Item = &'a [Fragment]>,
) -> Result<()> {
    let indices = load_all_indices(dataset).await?;
    let mut physical_index_boundaries = Vec::new();
    // Scalar indexes heal partial coverage through the fragment-reuse path.
    // Only vector segments require all-or-nothing physical replacement here.
    for index in indices.iter().filter(|index| {
        !is_system_index(index)
            && (segment_has_vector_details(index)
                || needs_vector_details_inference(index, dataset.schema()))
    }) {
        physical_index_boundaries.push((
            index.name.clone(),
            index.uuid,
            index_fragment_coverage(dataset, index).await?,
        ));
    }

    for (task_idx, fragments) in groups.enumerate() {
        let old_fragment_ids = fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>();
        for (index_name, segment_id, fragment_bitmap) in &physical_index_boundaries {
            let covered_fragment_ids = old_fragment_ids
                .iter()
                .copied()
                .filter(|fragment_id| {
                    u32::try_from(*fragment_id)
                        .is_ok_and(|fragment_id| fragment_bitmap.contains(fragment_id))
                })
                .collect::<Vec<_>>();
            if !covered_fragment_ids.is_empty()
                && covered_fragment_ids.len() != old_fragment_ids.len()
            {
                return Err(Error::invalid_input(format!(
                    "Compaction rewrite group {task_idx} with old fragments {old_fragment_ids:?} \
                     crosses the physical coverage boundary of index '{index_name}' segment \
                     {segment_id} (covered fragments {covered_fragment_ids:?}), but the selected \
                     remapper/publication path cannot atomically replace vector segments. \
                     Use a boundary-preserving DefaultCompactionPlanner plan, or select the \
                     built-in DatasetIndexRemapperOptions with eager address remapping"
                )));
            }
        }
    }
    Ok(())
}
