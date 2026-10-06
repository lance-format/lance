// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The extension point for memtable-resident indexes.
//!
//! A memtable index is not an index format. It is a mutable accumulator that
//! lives for one memtable: it takes rows as they are written, answers queries
//! against rows that have not been flushed yet, and at flush hands the ordinary
//! index builder whatever it can save it. The index that lands on disk is
//! always built by that ordinary builder, so a plugin adds a maintenance
//! strategy, never a storage format.
//!
//! Registering a plugin is the whole integration. Nothing else in the memtable
//! names an index kind, so a registered plugin is discovered from the base
//! table's index metadata, maintained on every write path, charged to the
//! memory budget, consulted by the query planner, and flushed without further
//! wiring.
//!
//! # What a plugin has to write
//!
//! Two traits. [`MemIndexPlugin`] declares the kind — which base-table index it
//! maintains, how to build one, how a filter reaches it. [`MemIndex`] is one
//! instance: take rows, answer questions, say what you cost, hand something
//! over at flush.
//!
//! A scalar plugin does not define a query vocabulary. It returns the same
//! [`ScalarQueryParser`] its on-disk index uses from
//! [`MemIndexPlugin::query_parser`], and then answers the same
//! [`AnyQuery`](lance_index::scalar::AnyQuery) types in
//! [`MemIndex::search`]. Everything a filter can express against the on-disk
//! index — comparisons, ranges, `IN`, `IS NULL`, `LIKE`, a spatial or
//! full-text function — reaches the memtable index for free, and the two
//! cannot drift apart.

use std::any::Any;
use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion::physical_plan::SendableRecordBatchStream;
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

use super::RowPosition;
use super::query::{MemMatches, MemQuery, SearchContext};

/// What a plugin builds an index with.
///
/// Comparable, so a writer can tell whether a refreshed set changed without
/// knowing any plugin's settings. Implemented for every `PartialEq` type; a
/// plugin whose settings hold a cache implements `PartialEq` to leave it out.
/// Settings must equal themselves: a value that does not, such as a NaN float,
/// reads as a changed set on every refresh and seals a memtable each time.
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
    /// Most rows the memtable holds before it flushes, for an index that would
    /// rather allocate once than grow.
    pub capacity_rows: usize,
    /// Most batches the memtable holds, for an index that indexes per batch.
    pub capacity_batches: usize,
    /// Whatever [`MemIndexPlugin::resolve`] returned for this index.
    pub params: &'a dyn MemIndexParams,
}

impl MemIndexBuildContext<'_> {
    /// The single column this index covers.
    ///
    /// Most kinds cover exactly one and would otherwise all repeat the same
    /// bounds check.
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
    /// spec names.
    ///
    /// For a kind whose columns are plain schema paths. A kind that resolves
    /// its own path — full-text search over a list of structs, where the path
    /// runs through list elements the schema walk does not follow — checks
    /// that itself.
    ///
    /// The field id matters as much as the name: index selection keys off it,
    /// so a spec whose field id names a *different* column would be bound
    /// under the wrong identity, serving stale reads and flushing the wrong
    /// column.
    pub fn check_columns_resolve(&self) -> Result<()> {
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

    /// `params`, if they are of type `P`.
    ///
    /// The params come from this plugin's own
    /// [`resolve`](MemIndexPlugin::resolve), so a mismatch is a
    /// wiring error rather than bad input.
    pub fn params<P: Any>(&self) -> Result<&P> {
        (self.params as &dyn Any)
            .downcast_ref::<P>()
            .ok_or_else(|| {
                Error::internal(format!(
                    "index '{}' was built with params of an unexpected type",
                    self.name
                ))
            })
    }
}

/// What an index needs to resolve its build params from the base table.
pub struct ParamsContext<'a> {
    /// Index name on the base table.
    pub name: &'a str,
    /// The base table, for an index whose params live in the trained artifact:
    /// a vector index reads the distance type its graph was built with.
    pub dataset: &'a crate::Dataset,
    /// The base-table index entry: its details message, field ids, fragments.
    pub index_meta: &'a IndexMetadata,
    /// The shard schema, for an index that must resolve a path through nested
    /// types before it knows what it covers.
    pub schema: &'a LanceSchema,
    /// Those fields' column names.
    pub columns: &'a [String],
    /// Build overrides the writer was configured with for this index, if any.
    ///
    /// How a deployment tunes one index without changing the base table: the
    /// writer is handed values keyed by index name, and each plugin recognises
    /// its own. A vector index takes its graph parameters this way.
    pub overrides: Option<&'a (dyn Any + Send + Sync)>,
}

impl ParamsContext<'_> {
    /// The writer's settings for this index, read as `P`.
    ///
    /// `Ok(None)` when the writer supplied none. Settings of any other type
    /// are an error naming the index, so a mistyped override cannot quietly
    /// leave the index on its defaults.
    pub fn overrides<P: Any>(&self) -> Result<Option<&P>> {
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
}

/// What a plugin resolved about one index against the base table.
#[derive(Debug)]
pub struct ResolvedIndex {
    /// The columns the index actually covers.
    ///
    /// Usually the base-table index's own fields. A kind that resolves a path
    /// through nested types reports the resolved path instead, because that is
    /// the name a query will use: a full-text index over a list of structs
    /// covers `tags.name`, not `tags`.
    pub columns: Vec<String>,
    /// The field id of each column, when the plugin resolved them itself.
    ///
    /// `None` looks each column up by name in the shard schema. A kind whose
    /// path runs through list elements must supply them: a lookup by name
    /// follows the list's physical `item` child and cannot find `tags.name`.
    pub field_ids: Option<Vec<i32>>,
    /// Whatever the plugin needs when it builds an index.
    pub params: Arc<dyn MemIndexParams>,
}

impl ResolvedIndex {
    /// Nothing resolved: the index covers what the base table says, and needs
    /// no parameters.
    pub fn plain(columns: Vec<String>) -> Self {
        Self {
            columns,
            field_ids: None,
            params: Arc::new(()),
        }
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

/// The flushed generation, for an index that writes its own file into it.
///
/// Already committed as a dataset by the time an index sees it, so its
/// fragment bitmap, version and schema are settled and an index can record
/// itself against them.
pub struct GenerationWrite<'a> {
    /// Directory holding the generation's files.
    pub path: &'a Path,
    /// Store to write through.
    pub object_store: &'a Arc<ObjectStore>,
    /// The generation as committed.
    pub dataset: &'a crate::Dataset,
    /// How many rows the generation holds.
    pub total_rows: usize,
    /// The name this index is recorded under.
    pub name: &'a str,
    /// File version the index is written at, matching the flushed data.
    pub storage_version: ConcreteFileVersion,
}

/// What a flush needs from an index.
pub struct FlushContext<'a> {
    /// Rows per batch when handing back training data.
    pub batch_size: usize,
    /// The generation to write into, for an index whose file is not built from
    /// a stream of values.
    ///
    /// Absent when nothing is being written — a caller asking only what an
    /// index would hand a trainer. An index that writes its own file says it
    /// has nothing to give rather than inventing a place to put it.
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

    /// The generation to write into, or an error naming the index.
    ///
    /// For an index whose [`FlushOutcome`] is always
    /// [`Wrote`](FlushOutcome::Wrote): reaching this without a generation is a
    /// caller asking a question this index cannot answer.
    pub fn generation(&self, index_name: &str) -> Result<&'a GenerationWrite<'a>> {
        self.generation.ok_or_else(|| {
            Error::invalid_input(format!(
                "index '{index_name}' writes its own file and has no training data to give"
            ))
        })
    }
}

/// What an index gives the flush.
pub enum FlushOutcome {
    /// Write no index for this generation: the index holds nothing to save,
    /// such as a vector index over a generation whose vectors are all null.
    Skip,
    /// Build the on-disk index the ordinary way, by reading the generation the
    /// flush has just written, with [`MemIndexPlugin::flush_params`].
    ///
    /// The right answer for a kind whose builder wants its input in an order
    /// the memtable does not hold it in, where re-reading costs less than
    /// sorting twice. Only a scalar [`flush_index_type`] can be built this way;
    /// any other kind builds its own file from the generation in the
    /// [`FlushContext`] and returns [`Self::Wrote`], and asking the flush to
    /// build it is an error.
    ///
    /// [`flush_index_type`]: MemIndexPlugin::flush_index_type
    BuildFromGeneration,
    /// Rows for the on-disk builder, in the shape
    /// [`MemIndexPlugin::training_criteria`] promises, built with
    /// [`MemIndexPlugin::flush_params`].
    ///
    /// This is the saving a memtable index exists to make: a B-tree wants its
    /// input sorted by value, and the memtable already holds it that way.
    TrainingData(SendableRecordBatchStream),
    /// The index wrote its own file into the generation, and this is the
    /// metadata to record.
    ///
    /// For an index whose file is not built from a stream of values at all — a
    /// vector graph and its storage, an inverted index assembled from
    /// partitions the memtable already holds.
    Wrote(Box<IndexMetadata>),
}

impl FlushOutcome {
    /// Training data from batches already in memory.
    pub fn from_batches(batches: Vec<RecordBatch>) -> Self {
        use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
        use futures::stream;

        let Some(first) = batches.first() else {
            return Self::BuildFromGeneration;
        };
        let schema = first.schema();
        Self::TrainingData(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream::iter(batches.into_iter().map(Ok)),
        )))
    }
}

/// One index, resident in one memtable.
///
/// Inserts into one instance never overlap: the writer serializes them. Every
/// other method may run concurrently with an insert and with each other, so
/// implementations own their synchronization; nothing here takes `&mut self`.
///
/// A failed insert stops the writer, and the instance is dropped with its
/// memtable. Rows become visible only after every index has taken them, so a
/// partly applied insert is never read. `flush` runs once, after the memtable
/// is frozen and its last insert has returned.
#[async_trait::async_trait]
pub trait MemIndex: Send + Sync + std::fmt::Debug + Any {
    /// The columns this index covers, in the order the base-table index names
    /// them.
    ///
    /// With [`Self::can_answer`] this is how a query finds its index. Resolving
    /// by what an index covers and what it can answer, rather than by which
    /// type it is, is what lets a plugin stand in for a kind Lance also builds
    /// in.
    fn columns(&self) -> &[String];

    /// Whether this index can answer `query`.
    ///
    /// Asked while planning, before any row is read, so the planner can choose
    /// between an index and a scan. An index recognises the queries it answers
    /// by downcasting and declines anything else — including a query of a shape
    /// it knows but a configuration it cannot serve, such as a vector search
    /// asking for a metric its graph was not built with.
    ///
    /// Planning asks with the query it will run, and then `search` must answer
    /// it. The one exception is discovering what exists before any query does —
    /// which full-text granularities a column has — where it asks a probe: a
    /// query of the same type carrying only the attributes routing turns on.
    /// Answer a probe truthfully on those. Search options a full-text query
    /// carries are execution hints, so they must not decide the answer.
    fn can_answer(&self, query: &dyn MemQuery) -> bool;

    /// Index every row of `batch`. Row `n` occupies position `row_offset + n`.
    ///
    /// Every batch carries the shard schema, so one missing a covered column is
    /// an error.
    fn insert(&self, batch: &RecordBatch, row_offset: RowPosition) -> Result<()>;

    /// Heap bytes held by this index.
    ///
    /// The flush trigger budgets from this, so it must cover the real footprint
    /// rather than approach it from below; an index that under-reports lets the
    /// memtable grow past its ceiling.
    fn resident_bytes(&self) -> usize;

    /// Answer `query`, returning only positions at or below
    /// [`SearchContext::max_visible`].
    ///
    /// Returning positions rather than rows is what keeps the batch store, the
    /// visibility cursors and the execution-plan machinery on the caller's side
    /// of this boundary.
    ///
    /// `None` means "I cannot help with this one" and the caller scans. That is
    /// a different answer from an empty result, which claims no row matches: a
    /// bloom filter asked for a range has no opinion, and asked for a value it
    /// has never seen it has a firm one. Returning `None` for a query
    /// [`can_answer`](Self::can_answer) accepted is an error the caller reports.
    ///
    /// A filter query is answered with [`MemMatches::Filter`], a search with
    /// [`MemMatches::Ranked`]. Ranked scores are merged with every other
    /// source's, so they must be on the same scale: a vector index returns the
    /// exact distance in the query's metric, refining any approximation before
    /// it returns, and a full-text index the score the built-in one would give.
    fn search(&self, query: &dyn MemQuery, ctx: &SearchContext) -> Result<Option<MemMatches>>;

    /// Hand the flush whatever this index can save it.
    ///
    /// Called once, after the generation's data is written and opened, so an
    /// index that writes its own file has a settled dataset to record against.
    async fn flush(&self, ctx: &FlushContext<'_>) -> Result<FlushOutcome>;

    /// This index, if it can also serve as the memtable's primary-key index.
    ///
    /// The memtable keeps one index on the primary-key column for deduplication
    /// and point lookups. When a user index on that column can serve the same
    /// purpose, sharing it halves both the memory and the insert work, so a
    /// kind that qualifies says so here rather than being recognised by type.
    fn as_primary_key(self: Arc<Self>) -> Option<Arc<dyn PrimaryKeyIndex>> {
        None
    }
}

/// An index that can also serve as the memtable's primary-key index.
///
/// Implemented by a kind that keys whole values and can find one, which in
/// practice means an ordered map. The memtable needs more of it than a query
/// does: point lookups by key, and whether an insert replaced a key already
/// held.
///
/// Only a user index on a single-column key is shared this way. The index the
/// memtable creates when none qualifies, and the one over a composite key's
/// encoded tuple, are always the built-in B-tree: replacing the B-tree plugin
/// does not replace them.
pub trait PrimaryKeyIndex: MemIndex {
    /// Index a batch and report whether any row replaced a key this index
    /// already held.
    ///
    /// The memtable needs the answer because an overwrite invalidates the
    /// append-only fast path a vector search would otherwise take: a stale row
    /// that is still in the graph must not consume a top-k slot.
    fn insert_and_report_existing(
        &self,
        batch: &RecordBatch,
        row_offset: RowPosition,
    ) -> Result<bool>;

    /// The newest position holding `key` at or below `max_visible`.
    fn newest_visible(
        &self,
        key: &datafusion::common::ScalarValue,
        max_visible: RowPosition,
    ) -> Option<RowPosition>;

    /// Every key in order, with the positions holding it, for training the
    /// generation's deduplication index.
    fn training_batches(&self, batch_size: usize) -> Result<Vec<RecordBatch>>;

    /// Whether this index holds no key at all.
    fn is_empty(&self) -> bool;
}

/// Declares one memtable index kind and builds instances of it.
#[async_trait::async_trait]
pub trait MemIndexPlugin: Send + Sync + std::fmt::Debug {
    /// A short name, used in plans and errors. Conventionally the index type's
    /// own name, for example `Bitmap`.
    fn name(&self) -> &str;

    /// The name of the protobuf details message identifying the base-table
    /// index this plugin maintains, for example `BitmapIndexDetails`.
    ///
    /// The name alone, without a package: the package varies with the dataset
    /// version, and a type url matches when its message name is exactly this.
    fn details_message(&self) -> &str;

    /// This plugin's version.
    ///
    /// A memtable index is never read back from disk, so this gates no
    /// compatibility. It shows in diagnostics, and a writer whose plugin changes
    /// version treats each index it maintains as changed.
    fn version(&self) -> u32 {
        1
    }

    /// The index type the flush builds on disk. Selects an ordinary index
    /// builder, so the persisted artifact stays the one Lance already writes.
    fn flush_index_type(&self) -> IndexType;

    /// The shape of the rows [`FlushOutcome::TrainingData`] carries.
    ///
    /// Compared against what the flush index type's trainer requires before any
    /// training happens, so a mismatch is an error naming both sides rather
    /// than an index that disagrees with its own data.
    fn training_criteria(&self) -> TrainingCriteria;

    /// The parameters the flush builds a scalar on-disk index with, from
    /// [`FlushOutcome::TrainingData`] or from the generation.
    ///
    /// The index type's defaults unless overridden. A kind whose base-table
    /// index was built with tuned parameters returns them, usually from the
    /// settings its [`resolve`](Self::resolve) read off that index.
    fn flush_params(&self, _spec: &MemIndexSpec) -> ScalarIndexParams {
        ScalarIndexParams::default()
    }

    /// How a filter expression reaches this index.
    ///
    /// Return the parser the on-disk index of the same kind uses, and every
    /// expression it already claims — comparisons, ranges, `IN`, `IS NULL`,
    /// `LIKE`, a scalar function — reaches this index too, producing the same
    /// [`AnyQuery`](lance_index::scalar::AnyQuery) that
    /// [`MemIndex::search`] then answers.
    ///
    /// The details are the base-table index's, for a kind whose claims depend
    /// on how it was built. They are absent when a caller configured the
    /// memtable directly rather than from a base table, so a plugin that needs
    /// them declines without them.
    ///
    /// `None` for a kind no filter expression names: a vector index and a
    /// full-text index are reached from the scan API instead.
    fn query_parser(
        &self,
        _index_name: String,
        _index_details: Option<&prost_types::Any>,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        None
    }

    /// Resolve this index against the base table: which columns it really
    /// covers, and whatever it needs to build one.
    ///
    /// Runs once, when a memtable is configured, so the per-memtable
    /// [`create`](Self::create) stays synchronous and cannot do I/O on the
    /// write path. A kind with nothing to resolve does not implement this.
    async fn resolve(&self, ctx: &ParamsContext<'_>) -> Result<ResolvedIndex> {
        Ok(ResolvedIndex::plain(ctx.columns.to_vec()))
    }

    /// Reject a column this index cannot maintain, before any row is written.
    ///
    /// Separate from [`create`](Self::create) because a caller validates a set
    /// of maintained indexes without building any of them.
    fn validate(&self, ctx: &MemIndexBuildContext<'_>) -> Result<()>;

    /// Build an empty index.
    fn create(&self, ctx: &MemIndexBuildContext<'_>) -> Result<Arc<dyn MemIndex>>;
}

/// The plugins a writer can maintain.
///
/// Held by the shard writer rather than a process global, so one test cannot
/// disturb another and two writers in one process can be configured
/// differently.
///
/// The default holds the kinds Lance builds in. A deployment starts from it
/// and adds its own, or replaces one with its own implementation of the same
/// kind.
#[derive(Debug, Clone)]
pub struct MemIndexRegistry {
    plugins: Vec<Arc<dyn MemIndexPlugin>>,
}

impl Default for MemIndexRegistry {
    fn default() -> Self {
        Self {
            plugins: vec![
                Arc::new(super::btree::BTreeMemIndexPlugin),
                Arc::new(super::hnsw::HnswMemIndexPlugin),
                Arc::new(super::fts::FtsMemIndexPlugin),
            ],
        }
    }
}

impl MemIndexRegistry {
    /// A registry maintaining nothing, for a caller that registers every
    /// plugin itself.
    pub fn empty() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Add a plugin. Two plugins cannot claim the same base-table index.
    pub fn add_plugin(&mut self, plugin: Arc<dyn MemIndexPlugin>) -> Result<()> {
        let message = plugin.details_message();
        if message.is_empty() || message.contains(['.', '/']) {
            return Err(Error::invalid_input(format!(
                "plugin '{}' claims details message '{message}', which is not a bare \
                 message name",
                plugin.name()
            )));
        }
        if let Some(existing) = self
            .plugins
            .iter()
            .find(|other| other.details_message() == plugin.details_message())
        {
            return Err(Error::invalid_input(format!(
                "plugin '{}' claims details message '{}', which '{}' already claims",
                plugin.name(),
                plugin.details_message(),
                existing.name(),
            )));
        }
        self.plugins.push(plugin);
        Ok(())
    }

    /// Builder form of [`Self::add_plugin`], for assembling a registry inline.
    pub fn with_plugin(mut self, plugin: Arc<dyn MemIndexPlugin>) -> Result<Self> {
        self.add_plugin(plugin)?;
        Ok(self)
    }

    /// Replace the plugin claiming the same base-table index, or add it.
    ///
    /// How a deployment substitutes its own implementation for one Lance
    /// builds in: the last plugin registered for a details message wins.
    pub fn replace_plugin(&mut self, plugin: Arc<dyn MemIndexPlugin>) {
        self.plugins
            .retain(|other| other.details_message() != plugin.details_message());
        self.plugins.push(plugin);
    }

    /// The plugin maintaining a base-table index with this details type url.
    pub fn plugin_for_details_url(&self, type_url: &str) -> Option<&Arc<dyn MemIndexPlugin>> {
        let message = type_url.rsplit(['/', '.']).next().unwrap_or(type_url);
        self.plugins
            .iter()
            .find(|plugin| plugin.details_message() == message)
    }

    /// Every registered plugin, for diagnostics.
    pub fn plugins(&self) -> &[Arc<dyn MemIndexPlugin>] {
        &self.plugins
    }
}

/// One index a memtable is to maintain, resolved against the base table.
///
/// The single description of a maintained index: which plugin, which columns,
/// and whatever that plugin resolved for itself. There is no per-kind
/// configuration type, so adding a kind adds no variant here.
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
    /// The base-table index's details message.
    ///
    /// Carried because the query parser is built from it: the same details the
    /// on-disk index was written with decide which expressions this index
    /// claims, so the memtable and the base table claim the same ones.
    pub details: Option<Arc<prost_types::Any>>,
}

impl MemIndexSpec {
    /// The single column this index covers, for the many callers that only
    /// handle single-column indexes.
    pub fn column(&self) -> &str {
        self.columns.first().map(String::as_str).unwrap_or("")
    }

    /// The field id of that column.
    pub fn field_id(&self) -> i32 {
        self.field_ids.first().copied().unwrap_or(-1)
    }

    /// Whether `other` describes the same index: same name, columns and
    /// plugin, built with the same settings. An index rebuilt under one name
    /// with a different metric or tokenizer is a different index.
    pub fn same_index(&self, other: &Self) -> bool {
        self.name == other.name
            && self.columns == other.columns
            && self.field_ids == other.field_ids
            && self.plugin.name() == other.plugin.name()
            && self.plugin.version() == other.plugin.version()
            && self.params.same_as(other.params.as_ref())
            && self.details == other.details
    }

    /// Build the index this spec describes.
    pub fn build(
        &self,
        schema: &LanceSchema,
        capacity_rows: usize,
        capacity_batches: usize,
    ) -> Result<Arc<dyn MemIndex>> {
        self.plugin.create(&MemIndexBuildContext {
            name: &self.name,
            schema,
            field_ids: &self.field_ids,
            columns: &self.columns,
            capacity_rows,
            capacity_batches,
            params: self.params.as_ref(),
        })
    }
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

/// Convenience constructors for the kinds Lance builds in.
///
/// The production path resolves specs from the base table through
/// [`MemIndexPlugin::resolve`]; these are for a caller that already knows what
/// it wants, which in practice means tests and low-level wiring.
impl MemIndexSpec {
    /// One index over one column, maintained by `plugin` with no resolved
    /// parameters.
    ///
    /// For a kind whose [`MemIndexPlugin::resolve`] returns nothing, which is
    /// every kind that reads its whole configuration from the column it covers.
    pub fn for_plugin(
        name: impl Into<String>,
        field_id: i32,
        column: impl Into<String>,
        plugin: Arc<dyn MemIndexPlugin>,
    ) -> Self {
        Self {
            name: name.into(),
            field_ids: vec![field_id],
            columns: vec![column.into()],
            plugin,
            params: Arc::new(()),
            details: None,
        }
    }

    /// A B-tree over one column.
    pub fn btree(name: impl Into<String>, field_id: i32, column: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_ids: vec![field_id],
            columns: vec![column.into()],
            plugin: Arc::new(super::btree::BTreeMemIndexPlugin),
            params: Arc::new(()),
            details: None,
        }
    }

    /// An HNSW graph over one vector column, with default graph parameters.
    pub fn hnsw(
        name: impl Into<String>,
        field_id: i32,
        column: impl Into<String>,
        distance_type: lance_linalg::distance::DistanceType,
    ) -> Self {
        Self::hnsw_with_params(
            name,
            field_id,
            column,
            distance_type,
            super::hnsw::mem_wal_hnsw_default(),
        )
    }

    /// An HNSW graph over one vector column.
    pub fn hnsw_with_params(
        name: impl Into<String>,
        field_id: i32,
        column: impl Into<String>,
        distance_type: lance_linalg::distance::DistanceType,
        build_params: lance_index::vector::hnsw::builder::HnswBuildParams,
    ) -> Self {
        Self {
            name: name.into(),
            field_ids: vec![field_id],
            columns: vec![column.into()],
            plugin: Arc::new(super::hnsw::HnswMemIndexPlugin),
            params: Arc::new(super::hnsw::HnswParams {
                distance_type,
                build_params,
            }),
            details: None,
        }
    }

    /// A full-text index over one column, resolving its path from the first
    /// batch it sees.
    pub fn fts(name: impl Into<String>, field_id: i32, column: impl Into<String>) -> Self {
        Self::fts_with_params(name, field_id, column, Default::default())
    }

    /// A full-text index over one column.
    pub fn fts_with_params(
        name: impl Into<String>,
        field_id: i32,
        column: impl Into<String>,
        params: lance_index::scalar::InvertedIndexParams,
    ) -> Self {
        Self {
            name: name.into(),
            field_ids: vec![field_id],
            columns: vec![column.into()],
            plugin: Arc::new(super::fts::FtsMemIndexPlugin),
            params: Arc::new(super::fts::FtsParams {
                params,
                resolved_field: None,
            }),
            details: None,
        }
    }
}
