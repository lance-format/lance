// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memtable index kinds as plugins.
//!
//! A [`MemIndexPlugin`] declares which base-table index it maintains and builds
//! [`MemIndex`] instances, one per memtable. An instance indexes rows as they
//! are written and answers queries over rows not yet flushed. At flush it gives
//! Lance's regular index builder what it needs, so the index on disk is one
//! Lance already writes.

use std::any::Any;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arrow_array::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_file::version::ConcreteFileVersion;
use lance_index::IndexType;
use lance_index::scalar::ScalarIndexParams;
use lance_index::scalar::expression::ScalarQueryParser;
use lance_index::scalar::registry::TrainingCriteria;
use lance_io::object_store::ObjectStore;
use lance_table::format::IndexMetadata;
use object_store::path::Path;
use uuid::Uuid;

use super::RowPosition;
use super::query::{MemMatches, MemQuery, SearchContext};
use crate::Dataset;
use crate::dataset::INDICES_DIR;
use crate::dataset::mem_wal::memtable::batch_store::StoredBatch;

/// Build settings a plugin resolves for one index. Implemented for every
/// `PartialEq` type; compared to tell whether an index changed.
pub trait MemIndexParams: Any + Send + Sync + std::fmt::Debug {
    /// Whether `other` holds the same settings.
    fn same_as(&self, other: &dyn MemIndexParams) -> bool;
}

impl<T: Any + Send + Sync + std::fmt::Debug + PartialEq> MemIndexParams for T {
    fn same_as(&self, other: &dyn MemIndexParams) -> bool {
        (other as &dyn Any).downcast_ref::<T>() == Some(self)
    }
}

/// What an index needs to build itself, before any row is written.
pub struct MemIndexBuildContext<'a> {
    /// Index name, matching the base-table index this maintains.
    pub name: &'a str,
    /// The shard schema: the base table's, plus the tombstone column.
    pub schema: &'a LanceSchema,
    /// Field ids of the covered columns, in the order the base-table index
    /// names them.
    pub field_ids: &'a [i32],
    /// Those fields' column names, in the same order.
    pub columns: &'a [String],
    /// Most rows the memtable holds before it flushes; 0 while validating.
    pub capacity_rows: usize,
    /// Most batches the memtable holds; 0 while validating.
    pub capacity_batches: usize,
    /// Whatever [`MemIndexPlugin::resolve`] returned for this index.
    pub params: &'a dyn MemIndexParams,
}

impl MemIndexBuildContext<'_> {
    /// The single column this index covers, or an error if it covers several.
    pub fn single_column(&self) -> Result<(&str, i32)> {
        match (self.columns, self.field_ids) {
            ([column], [field_id]) => Ok((column.as_str(), *field_id)),
            _ => Err(Error::invalid_input(format!(
                "index '{}' covers {} columns, but this kind covers exactly one",
                self.name,
                self.columns.len()
            ))),
        }
    }

    /// Check that every covered column is in the schema under the field id the
    /// spec names. A plugin indexing a nested path checks that path itself.
    pub fn check_columns_resolve(&self) -> Result<()> {
        check_column_count(self.name, self.columns, self.field_ids)?;
        for (column, field_id) in self.columns.iter().zip(self.field_ids) {
            let field = self.schema.field(column).ok_or_else(|| {
                Error::invalid_input(format!(
                    "index '{}' is configured on column '{}', which is not in the shard schema; \
                     available columns: [{}]",
                    self.name,
                    column,
                    self.schema
                        .fields
                        .iter()
                        .map(|f| f.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;
            if field.id != *field_id {
                return Err(Error::invalid_input(format!(
                    "index '{}' is configured with field_id {field_id} but its column '{column}' \
                     has field_id {} in the shard schema",
                    self.name, field.id,
                )));
            }
        }
        Ok(())
    }

    /// [`Self::check_columns_resolve`], and that each covered column is a
    /// top-level field, for a kind that reads its column from a batch by name.
    pub fn check_top_level_columns(&self) -> Result<()> {
        self.check_columns_resolve()?;
        for column in self.columns {
            if !self.schema.fields.iter().any(|field| &field.name == column) {
                return Err(Error::invalid_input(format!(
                    "index '{}' covers the nested column '{column}'; this kind maintains only \
                     top-level columns",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// `params` as `P`, the type this plugin's own
    /// [`resolve`](MemIndexPlugin::resolve) returned.
    pub fn params<P: Any>(&self) -> Result<&P> {
        downcast_params(self.name, self.params)
    }
}

fn downcast_params<'a, P: Any>(name: &str, params: &'a dyn MemIndexParams) -> Result<&'a P> {
    (params as &dyn Any).downcast_ref::<P>().ok_or_else(|| {
        Error::internal(format!(
            "index '{name}' was built with params of an unexpected type"
        ))
    })
}

/// What an index needs to resolve its build params from the base table.
pub struct ResolveContext<'a> {
    /// Index name on the base table.
    pub name: &'a str,
    /// The base table.
    pub dataset: &'a Dataset,
    /// The base-table index entry.
    pub index_meta: &'a IndexMetadata,
    /// The shard schema.
    pub schema: &'a LanceSchema,
    /// The base-table index's columns, as their full paths.
    pub columns: &'a [String],
    /// The writer's settings for this index, read through
    /// [`Self::overrides`].
    overrides: Option<&'a (dyn Any + Send + Sync)>,
    /// Whether the plugin asked for `overrides`.
    overrides_read: AtomicBool,
}

impl<'a> ResolveContext<'a> {
    pub(crate) fn new(
        name: &'a str,
        dataset: &'a Dataset,
        index_meta: &'a IndexMetadata,
        schema: &'a LanceSchema,
        columns: &'a [String],
        overrides: Option<&'a (dyn Any + Send + Sync)>,
    ) -> Self {
        Self {
            name,
            dataset,
            index_meta,
            schema,
            columns,
            overrides,
            overrides_read: AtomicBool::new(false),
        }
    }

    /// The writer's settings for this index as `P`, or `Ok(None)` when the
    /// writer gave none. Settings of another type are an error.
    pub fn overrides<P: Any>(&self) -> Result<Option<&P>> {
        self.overrides_read.store(true, Ordering::Relaxed);
        let Some(overrides) = self.overrides else {
            return Ok(None);
        };
        overrides.downcast_ref::<P>().map(Some).ok_or_else(|| {
            Error::invalid_input(format!(
                "index '{}' was given writer settings of a type its plugin does not read; \
                 it reads {}",
                self.name,
                std::any::type_name::<P>()
            ))
        })
    }

    /// Whether the writer gave settings the plugin never asked for.
    pub(crate) fn overrides_ignored(&self) -> bool {
        self.overrides.is_some() && !self.overrides_read.load(Ordering::Relaxed)
    }
}

/// What a plugin resolved about one index against the base table.
#[derive(Debug)]
pub struct ResolvedIndex {
    /// The columns the index covers, as a query names them: a full-text index
    /// over a list of structs covers `tags.name`, not `tags`.
    pub columns: Vec<String>,
    /// Each column's field id, when the plugin resolved them itself; `None`
    /// keeps the base index's ids for its own columns and looks up any other.
    pub field_ids: Option<Vec<i32>>,
    /// Whatever the plugin needs when it builds an index.
    pub params: Arc<dyn MemIndexParams>,
}

impl ResolvedIndex {
    /// The base table's columns, and no parameters.
    pub fn plain(columns: Vec<String>) -> Self {
        Self::with_params(columns, ())
    }

    /// Parameters for the columns the base table names.
    pub fn with_params<P: MemIndexParams>(columns: Vec<String>, params: P) -> Self {
        Self {
            columns,
            field_ids: None,
            params: Arc::new(params),
        }
    }

    /// The field ids the plugin resolved for its columns, in the same order.
    pub fn with_field_ids(mut self, field_ids: Vec<i32>) -> Self {
        self.field_ids = Some(field_ids);
        self
    }
}

/// The flushed generation, already committed, for an index that writes its own
/// file into it.
pub struct GenerationWrite<'a> {
    /// Directory holding the generation's files.
    pub path: &'a Path,
    /// Store to write through.
    pub object_store: &'a Arc<ObjectStore>,
    /// The generation as committed.
    pub dataset: &'a Dataset,
    /// How many rows the generation holds.
    pub total_rows: usize,
    /// The name this index is recorded under.
    pub name: &'a str,
    /// File version the index is written at, matching the flushed data.
    pub storage_version: ConcreteFileVersion,
}

impl GenerationWrite<'_> {
    /// A fresh index id and the directory its files go in.
    pub fn new_index_dir(&self) -> (Uuid, Path) {
        let uuid = Uuid::new_v4();
        let dir = self.path.clone().join(INDICES_DIR).join(uuid.to_string());
        (uuid, dir)
    }

    /// The metadata recording an index written into this generation.
    pub fn index_metadata(
        &self,
        uuid: Uuid,
        field_ids: Vec<i32>,
        index_details: prost_types::Any,
        index_version: i32,
    ) -> IndexMetadata {
        IndexMetadata {
            uuid,
            name: self.name.to_string(),
            fields: field_ids,
            covering_fields: vec![],
            dataset_version: self.dataset.version().version,
            fragment_bitmap: Some(self.dataset.fragment_bitmap.as_ref().clone()),
            index_details: Some(Arc::new(index_details)),
            index_version,
            created_at: Some(chrono::Utc::now()),
            base_id: None,
            files: None,
        }
    }
}

/// What a flush needs from an index.
pub struct FlushContext<'a> {
    /// Rows per batch when handing back training data.
    pub batch_size: usize,
    /// The generation to write into; set whenever the flusher calls.
    pub generation: Option<&'a GenerationWrite<'a>>,
}

impl<'a> FlushContext<'a> {
    /// A flush that only collects training data.
    pub fn training_only(batch_size: usize) -> Self {
        Self {
            batch_size,
            generation: None,
        }
    }

    /// The generation to write into.
    pub fn generation(&self) -> Result<&'a GenerationWrite<'a>> {
        self.generation.ok_or_else(|| {
            Error::internal("an index that writes its own file was flushed with no generation")
        })
    }
}

/// What an index gives the flush.
pub enum FlushOutcome {
    /// Write no index for this generation, such as a vector index whose
    /// vectors are all null.
    Skip,
    /// Build the on-disk index from the generation's rows with
    /// [`MemIndexPlugin::flush_params`]. Scalar index types only.
    BuildFromGeneration,
    /// Rows for the on-disk builder, in the shape
    /// [`MemIndexPlugin::training_criteria`] declares, built with
    /// [`MemIndexPlugin::flush_params`]. Scalar index types only.
    TrainingData(SendableRecordBatchStream),
    /// The index wrote its own file into the generation; record this metadata.
    Wrote(Box<IndexMetadata>),
}

impl FlushOutcome {
    /// Training data from batches already in memory; none builds from the
    /// generation.
    pub fn from_batches(batches: Vec<RecordBatch>) -> Self {
        let Some(first) = batches.first() else {
            return Self::BuildFromGeneration;
        };
        let schema = first.schema();
        Self::TrainingData(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::iter(batches.into_iter().map(Ok)),
        )))
    }
}

/// One index, resident in one memtable.
///
/// Inserts are serialized, but every other call may run alongside them, so an
/// implementation synchronizes itself. A failed insert stops the writer.
#[async_trait::async_trait]
pub trait MemIndex: Send + Sync + std::fmt::Debug + Any {
    /// The columns this index covers, in the order the base-table index names
    /// them.
    fn columns(&self) -> &[String];

    /// Whether this index can answer a ranked search, asked while planning with
    /// the query [`search`](Self::search) will receive. Only finding which
    /// full-text granularities exist asks a probe instead. A full-text query's
    /// search options must not decide the answer. Filters are not asked: they
    /// reach the index through [`MemIndexPlugin::query_parser`].
    fn can_answer(&self, query: &dyn MemQuery) -> bool;

    /// Index every row of `batch`. Row `n` occupies position `row_offset + n`.
    fn insert(&self, batch: &RecordBatch, row_offset: RowPosition) -> Result<()>;

    /// Index, in order, every batch of one write. The default inserts them one
    /// at a time.
    fn insert_batches(&self, batches: &[StoredBatch]) -> Result<()> {
        batches
            .iter()
            .try_for_each(|stored| self.insert(&stored.data, stored.row_offset))
    }

    /// Heap bytes held by this index. The flush trigger budgets from it, so it
    /// must not report less than the index holds.
    fn resident_bytes(&self) -> usize;

    /// Answer `query` with positions at or below
    /// [`SearchContext::max_visible`]; an empty answer means no row matches.
    ///
    /// A filter is answered with [`MemMatches::Filter`]. `None` declines it,
    /// as an index may when it matches more than
    /// [`SearchContext::match_budget`]; the caller then reads every row.
    ///
    /// A search is answered with [`MemMatches::Ranked`], scored as every other
    /// source scores it: the exact distance in the query's metric, or the
    /// built-in full-text score. `None` declines a search
    /// [`can_answer`](Self::can_answer) rejects; declining one it accepted is
    /// an error.
    fn search(&self, query: &dyn MemQuery, ctx: &SearchContext) -> Result<Option<MemMatches>>;

    /// Hand the flush what this index can save it. Called once, after the last
    /// insert.
    async fn flush(&self, ctx: &FlushContext<'_>) -> Result<FlushOutcome>;

    /// This index, if it can also serve as the memtable's primary-key index,
    /// which then needs no index of its own.
    fn as_primary_key(self: Arc<Self>) -> Option<Arc<dyn PrimaryKeyIndex>> {
        None
    }
}

/// An index that can also serve as the memtable's primary-key index.
///
/// Only a user index on exactly the key column is shared this way; otherwise
/// the memtable keeps a B-tree of its own. Until a key is rewritten, the store
/// inserts through [`insert_and_report_existing`](Self::insert_and_report_existing).
pub trait PrimaryKeyIndex: Send + Sync {
    /// Index a batch and report whether any row's key was already held, or
    /// repeats earlier in the batch.
    fn insert_and_report_existing(
        &self,
        batch: &RecordBatch,
        row_offset: RowPosition,
    ) -> Result<bool>;

    /// The newest position holding `key` at or below `max_visible`.
    fn newest_visible(&self, key: &ScalarValue, max_visible: RowPosition) -> Option<RowPosition>;

    /// Every key in order, with the positions holding it, for training the
    /// generation's deduplication index.
    fn training_batches(&self, batch_size: usize) -> Result<Vec<RecordBatch>>;

    /// Whether this index holds no key at all.
    fn is_empty(&self) -> bool;
}

/// Declares one memtable index kind and builds instances of it.
///
/// A writer tells kinds apart by their Rust type and [`version`](Self::version),
/// so whatever else decides how an index is built belongs in the params
/// [`resolve`](Self::resolve) returns.
#[async_trait::async_trait]
pub trait MemIndexPlugin: Send + Sync + std::fmt::Debug + Any {
    /// A short name for plans and errors, for example `BTree`.
    fn name(&self) -> &str;

    /// The name of the details message identifying the base-table index this
    /// plugin maintains, without its package, for example `BTreeIndexDetails`.
    fn details_message(&self) -> &str;

    /// This plugin's version. A writer treats each index whose plugin changed
    /// version as changed.
    fn version(&self) -> u32 {
        1
    }

    /// The index type the flush builds on disk.
    fn flush_index_type(&self) -> IndexType;

    /// The shape of the rows [`FlushOutcome::TrainingData`] carries.
    fn training_criteria(&self) -> TrainingCriteria;

    /// The parameters for a [`FlushOutcome::BuildFromGeneration`] or
    /// [`FlushOutcome::TrainingData`] build; an index returning
    /// [`FlushOutcome::Wrote`] writes its own. Derive them from `spec.params`,
    /// so a changed setting is a different index under
    /// [`MemIndexSpec::same_index`]. Called when the writer opens, so an error
    /// fails the open, not a flush. By default, the index type's default
    /// parameters.
    fn flush_params(&self, _spec: &MemIndexSpec) -> Result<ScalarIndexParams> {
        Ok(ScalarIndexParams::default())
    }

    /// The parser the on-disk index of this kind uses, so a filter expression
    /// it claims reaches this index as the same query. `index_details` are the
    /// base-table index's, absent for a memtable configured directly. `None`
    /// for a kind no filter expression names.
    fn query_parser(
        &self,
        _index_name: String,
        _index_details: Option<&prost_types::Any>,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        None
    }

    /// Resolve this index against the base table: the columns it covers and
    /// what it needs to build one. Runs each time a writer opens or refreshes
    /// its index set; do any I/O here so [`create`](Self::create) need not.
    ///
    /// A plugin that takes settings reads the writer's settings for this index
    /// with [`ResolveContext::overrides`], checks them, and keeps them in the
    /// params it returns, so they reach both [`create`](Self::create) and
    /// [`flush_params`](Self::flush_params). Settings the plugin never reads
    /// are ignored, with a warning. The default reads none.
    async fn resolve(&self, ctx: &ResolveContext<'_>) -> Result<ResolvedIndex> {
        Ok(ResolvedIndex::plain(ctx.columns.to_vec()))
    }

    /// Reject a column this index cannot maintain, before any row is written.
    /// A writer opens on this alone, so [`create`](Self::create) must not fail
    /// for a spec it accepts.
    fn validate(&self, ctx: &MemIndexBuildContext<'_>) -> Result<()>;

    /// Build an empty index.
    fn create(&self, ctx: &MemIndexBuildContext<'_>) -> Result<Arc<dyn MemIndex>>;
}

/// The plugins a writer can maintain.
#[derive(Debug, Clone)]
pub struct MemIndexRegistry {
    plugins: Vec<Arc<dyn MemIndexPlugin>>,
}

impl MemIndexRegistry {
    /// A registry with no plugins; [`Default`] holds the built-in kinds.
    pub fn empty() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Add a plugin. Two plugins cannot claim the same base-table index.
    pub fn add_plugin(&mut self, plugin: Arc<dyn MemIndexPlugin>) -> Result<()> {
        check_details_message(plugin.as_ref())?;
        let message = plugin.details_message();
        if let Some(existing) = self
            .plugins
            .iter()
            .find(|other| other.details_message() == message)
        {
            return Err(Error::invalid_input(format!(
                "plugin '{}' claims details message '{message}', which '{}' already claims",
                plugin.name(),
                existing.name(),
            )));
        }
        self.plugins.push(plugin);
        Ok(())
    }

    /// [`Self::add_plugin`], by value.
    pub fn with_plugin(mut self, plugin: Arc<dyn MemIndexPlugin>) -> Result<Self> {
        self.add_plugin(plugin)?;
        Ok(self)
    }

    /// Replace the plugin claiming the same base-table index, or add it.
    pub fn replace_plugin(&mut self, plugin: Arc<dyn MemIndexPlugin>) -> Result<()> {
        check_details_message(plugin.as_ref())?;
        self.plugins
            .retain(|other| other.details_message() != plugin.details_message());
        self.plugins.push(plugin);
        Ok(())
    }

    /// The plugin maintaining a base-table index with this details type url.
    pub fn plugin_for_details_url(&self, type_url: &str) -> Option<&Arc<dyn MemIndexPlugin>> {
        let message = type_url.rsplit(['/', '.']).next().unwrap_or(type_url);
        self.plugins
            .iter()
            .find(|plugin| plugin.details_message() == message)
    }

    /// Every registered plugin.
    pub fn plugins(&self) -> &[Arc<dyn MemIndexPlugin>] {
        &self.plugins
    }
}

/// A plugin's details message must be a bare message name, the only part of a
/// type url the registry matches on.
fn check_details_message(plugin: &dyn MemIndexPlugin) -> Result<()> {
    let message = plugin.details_message();
    if message.is_empty() || message.contains(['.', '/']) {
        return Err(Error::invalid_input(format!(
            "plugin '{}' claims details message '{message}', which is not a bare message name",
            plugin.name()
        )));
    }
    Ok(())
}

/// One index a memtable maintains: its plugin, its columns, and what the plugin
/// resolved for it.
#[derive(Clone)]
pub struct MemIndexSpec {
    /// Index name, matching the base-table index it maintains.
    pub name: String,
    /// Field ids of the covered columns.
    pub field_ids: Vec<i32>,
    /// Those fields' column names.
    pub columns: Vec<String>,
    /// The plugin that builds and answers for it.
    pub plugin: Arc<dyn MemIndexPlugin>,
    /// What [`MemIndexPlugin::resolve`] returned.
    pub params: Arc<dyn MemIndexParams>,
    /// The base-table index's details message, which the query parser is built
    /// from.
    pub details: Option<Arc<prost_types::Any>>,
}

impl MemIndexSpec {
    /// One index over one column, maintained by `plugin` with no parameters.
    pub fn for_plugin(
        name: impl Into<String>,
        field_id: i32,
        column: impl Into<String>,
        plugin: Arc<dyn MemIndexPlugin>,
    ) -> Self {
        Self::single_column(name, field_id, column, plugin, Arc::new(()))
    }

    /// One index over one column, with the parameters its plugin builds from.
    pub fn single_column(
        name: impl Into<String>,
        field_id: i32,
        column: impl Into<String>,
        plugin: Arc<dyn MemIndexPlugin>,
        params: Arc<dyn MemIndexParams>,
    ) -> Self {
        Self {
            name: name.into(),
            field_ids: vec![field_id],
            columns: vec![column.into()],
            plugin,
            params,
            details: None,
        }
    }

    /// Whether `other` describes the same index: name, columns, field ids,
    /// plugin type, plugin version, settings and base-table details.
    pub fn same_index(&self, other: &Self) -> bool {
        self.name == other.name
            && self.columns == other.columns
            && self.field_ids == other.field_ids
            && (self.plugin.as_ref() as &dyn Any).type_id()
                == (other.plugin.as_ref() as &dyn Any).type_id()
            && self.plugin.details_message() == other.plugin.details_message()
            && self.plugin.version() == other.plugin.version()
            && self.params.same_as(other.params.as_ref())
            && self.details == other.details
    }

    /// `params` as `P`, the type its plugin's own
    /// [`resolve`](MemIndexPlugin::resolve) returned.
    pub fn params<P: Any>(&self) -> Result<&P> {
        downcast_params(&self.name, self.params.as_ref())
    }

    /// The context this spec's plugin validates and builds against.
    fn build_context<'a>(
        &'a self,
        schema: &'a LanceSchema,
        capacity_rows: usize,
        capacity_batches: usize,
    ) -> MemIndexBuildContext<'a> {
        MemIndexBuildContext {
            name: &self.name,
            schema,
            field_ids: &self.field_ids,
            columns: &self.columns,
            capacity_rows,
            capacity_batches,
            params: self.params.as_ref(),
        }
    }

    /// Check that this spec's plugin can maintain it against `schema`.
    pub fn validate(&self, schema: &LanceSchema) -> Result<()> {
        check_column_count(&self.name, &self.columns, &self.field_ids)?;
        self.plugin.validate(&self.build_context(schema, 0, 0))?;
        if self.plugin.flush_index_type().is_scalar() {
            self.plugin.flush_params(self)?;
        }
        Ok(())
    }

    /// Build the index this spec describes.
    pub fn build(
        &self,
        schema: &LanceSchema,
        capacity_rows: usize,
        capacity_batches: usize,
    ) -> Result<Arc<dyn MemIndex>> {
        check_column_count(&self.name, &self.columns, &self.field_ids)?;
        self.plugin
            .create(&self.build_context(schema, capacity_rows, capacity_batches))
    }
}

/// Columns and field ids are parallel lists; a plugin indexes them pairwise.
fn check_column_count(name: &str, columns: &[String], field_ids: &[i32]) -> Result<()> {
    if columns.len() != field_ids.len() {
        return Err(Error::invalid_input(format!(
            "index '{name}' names {} columns but {} field ids",
            columns.len(),
            field_ids.len()
        )));
    }
    Ok(())
}

impl std::fmt::Debug for MemIndexSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemIndexSpec")
            .field("name", &self.name)
            .field("columns", &self.columns)
            .field("field_ids", &self.field_ids)
            .field("plugin", &self.plugin.name())
            .field("version", &self.plugin.version())
            .field("params", &self.params)
            .finish()
    }
}
