// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The indexes one memtable maintains. Each kind is a [`MemIndexPlugin`];
//! B-tree, HNSW and full-text search are built in.

pub mod arena_skiplist;
mod btree;
mod filter;
mod fts;
mod hnsw;
mod pk;
mod pk_key;
mod plugin;
mod query;
#[cfg(test)]
pub(crate) mod test_plugin;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arrow_array::RecordBatch;
use arrow_schema::Schema as ArrowSchema;
use lance_core::datatypes::Schema as LanceSchema;
use lance_core::{Error, Result};
use lance_index::scalar::InvertedIndexParams;
use lance_index::scalar::inverted::DocumentGranularity;
use lance_linalg::distance::DistanceType;
use tracing::instrument;

use super::memtable::batch_store::StoredBatch;
use super::wal::WriterCursors;
use pk::{OwnedPk, PkIndex};

pub use btree::{BTreeMemIndex, BTreeMemIndexPlugin};
pub use filter::{IndexedFilter, MemIndexCatalog, evaluate as evaluate_index_filter, plan_filter};
pub use fts::{FtsEntry, FtsMemIndex, FtsMemIndexPlugin, FtsParams, FtsQueryExpr, SearchOptions};
pub(crate) use fts::{QueryLocalFtsIndex, QueryLocalFtsStats, search_cross_column};
pub use hnsw::{HnswMemIndex, HnswMemIndexPlugin, HnswParams};
pub use pk_key::encode_pk_tuple;
pub use plugin::{
    FlushContext, FlushOutcome, GenerationWrite, MemIndex, MemIndexBuildContext, MemIndexParams,
    MemIndexPlugin, MemIndexRegistry, MemIndexSpec, PrimaryKeyIndex, ResolveContext, ResolvedIndex,
};
pub use query::{
    FtsMemQuery, MemMatches, MemQuery, MemSearchResult, PositionSet, RankedMatch, ScalarQuery,
    SearchContext, VectorMemQuery,
};

/// Row position in a memtable: the row's position across all its batches,
/// which becomes its row id in the flushed file.
pub type RowPosition = u64;

/// At or below this many rows, index on the calling thread: a thread per index
/// costs more than the work. Tune with `benches/mem_wal/vector/mem_wal_index_micro.rs`.
const PARALLEL_INDEX_MIN_ROWS: usize = 64;

impl Default for MemIndexRegistry {
    /// The kinds Lance builds in.
    fn default() -> Self {
        let mut registry = Self::empty();
        for plugin in [
            Arc::new(BTreeMemIndexPlugin) as Arc<dyn MemIndexPlugin>,
            Arc::new(HnswMemIndexPlugin),
            Arc::new(FtsMemIndexPlugin),
        ] {
            registry
                .add_plugin(plugin)
                .expect("the built-in kinds claim distinct details messages");
        }
        registry
    }
}

/// Reject index specs and primary-key columns that would fail every insert,
/// so a writer never accepts a row that replay could not index. Call once,
/// when a shard opens.
pub fn validate_index_specs(
    specs: &[MemIndexSpec],
    schema: &ArrowSchema,
    lance_schema: &LanceSchema,
    pk_columns: &[String],
) -> Result<()> {
    let mut seen = HashMap::new();
    for spec in specs {
        if !first_under_its_name(&mut seen, spec)? {
            continue;
        }
        spec.validate(lance_schema)?;
    }
    for column in pk_columns {
        let field = schema.field_with_name(column).map_err(|_| {
            Error::invalid_input(format!(
                "primary-key column '{column}' is not in the shard schema"
            ))
        })?;
        // Only a composite key is encoded, so only it restricts types.
        if pk_columns.len() > 1 && !pk_key::is_encodable(field.data_type()) {
            return Err(Error::invalid_input(format!(
                "composite primary-key column '{column}' has type {:?}, which has no \
                 order-preserving key encoding",
                field.data_type()
            )));
        }
    }
    Ok(())
}

/// Whether `spec` is the first under its name. A name listed again for the same
/// index is that one index; for a different one it is refused.
fn first_under_its_name<'a>(
    seen: &mut HashMap<&'a str, &'a MemIndexSpec>,
    spec: &'a MemIndexSpec,
) -> Result<bool> {
    match seen.insert(spec.name.as_str(), spec) {
        None => Ok(true),
        Some(first) if first.same_index(spec) => Ok(false),
        Some(_) => Err(Error::invalid_input(format!(
            "index '{}' is configured twice, as two different indexes",
            spec.name
        ))),
    }
}

/// The error for an index no registered plugin maintains, naming the index.
pub(crate) fn unsupported_index_type(
    index_name: &str,
    type_url: &str,
    registry: &MemIndexRegistry,
) -> Error {
    let supported = registry
        .plugins()
        .iter()
        .map(|plugin| plugin.details_message())
        .collect::<Vec<_>>()
        .join(", ");
    Error::invalid_input(format!(
        "index '{index_name}' has type {type_url}, which no registered plugin maintains. \
         Supported: [{supported}]"
    ))
}

/// Which prefix of a MemTable a reader may see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemTableVisibility {
    /// [`IndexStore::visible_count`]. Required for every reader but the writer
    /// itself: a row past this bound can still fail its append and never exist.
    #[default]
    Published,
    /// [`IndexStore::indexed_count`], which also covers writes whose append is
    /// outstanding.
    ///
    /// Sound only for a writer reading its own prefix under the lock that makes
    /// it the sole writer. Both cursors advance over contiguous prefixes, so a
    /// row derived from `p` cannot be published before `p`, and a failed append
    /// poisons the writer before either is acknowledged.
    Indexed,
}

/// The indexes one memtable maintains, keyed by name, and its two cursors:
/// what every index holds, and what readers may see.
#[derive(Default)]
pub struct IndexStore {
    /// Sorted by name, so the same query always reaches the same index.
    indexes: BTreeMap<String, Arc<dyn MemIndex>>,
    /// How a filter expression reaches these indexes.
    filter_catalog: Arc<MemIndexCatalog>,
    pk_index: Option<PkIndex>,
    /// Batches every index holds, as an exclusive count. Not a visibility
    /// bound: readers use [`Self::visible_count`].
    indexed_count: AtomicUsize,
    /// The writer's cursors and this memtable's offset in them; `None` in tests
    /// and benches.
    durability: Option<(Arc<WriterCursors>, usize)>,
    pk_has_overrides: AtomicBool,
    /// Whether any row has reached the indexes; an index added after would
    /// miss it.
    has_rows: AtomicBool,
}

impl std::fmt::Debug for IndexStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexStore")
            .field("indexes", &self.indexes.keys().collect::<Vec<_>>())
            .field("pk_index", &self.pk_index.as_ref().map(PkIndex::describe))
            .field("indexed_count", &self.indexed_count())
            .field("pk_has_overrides", &self.pk_has_overrides())
            .finish()
    }
}

/// The rows one insert hands every index.
#[derive(Clone, Copy)]
enum Rows<'a> {
    One(&'a RecordBatch, RowPosition),
    Batches(&'a [StoredBatch]),
}

impl Rows<'_> {
    fn num_rows(&self) -> usize {
        match self {
            Self::One(batch, _) => batch.num_rows(),
            Self::Batches(batches) => batches.iter().map(|stored| stored.num_rows).sum(),
        }
    }

    /// Feed every batch to `insert`, reporting whether any call did.
    fn each(
        &self,
        mut insert: impl FnMut(&RecordBatch, RowPosition) -> Result<bool>,
    ) -> Result<bool> {
        match self {
            Self::One(batch, row_offset) => insert(batch, *row_offset),
            Self::Batches(batches) => batches.iter().try_fold(false, |seen, stored| {
                Ok(insert(&stored.data, stored.row_offset)? || seen)
            }),
        }
    }

    fn insert_into(&self, index: &dyn MemIndex) -> Result<()> {
        match self {
            Self::One(batch, row_offset) => index.insert(batch, *row_offset),
            Self::Batches(batches) => index.insert_batches(batches),
        }
    }
}

/// One index's share of an insert, reporting whether it saw a rewritten key.
type IndexTask<'a> = Box<dyn Fn() -> Result<bool> + Send + Sync + 'a>;

impl IndexStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build every index a memtable is to maintain. `max_rows` and
    /// `max_batches` are the memtable's capacity, which sizes structures an
    /// index allocates up front.
    pub fn from_specs(
        specs: &[MemIndexSpec],
        schema: &LanceSchema,
        max_rows: usize,
        max_batches: usize,
    ) -> Result<Self> {
        let mut store = Self::new();
        let mut seen = HashMap::new();
        for spec in specs {
            if !first_under_its_name(&mut seen, spec)? {
                continue;
            }
            store.add_index(
                spec.name.clone(),
                spec.build(schema, max_rows, max_batches)?,
            );
        }
        store.filter_catalog = Arc::new(MemIndexCatalog::new(specs, schema));
        Ok(store)
    }

    /// Add a built index. Indexes are added before any row is inserted.
    pub fn add_index(&mut self, name: String, index: Arc<dyn MemIndex>) {
        assert!(
            !self.has_rows.load(Ordering::Acquire),
            "indexes must be added before any row is inserted"
        );
        self.indexes.insert(name, index);
    }

    /// Add a B-tree over one column.
    pub fn add_btree(&mut self, name: String, field_id: i32, column: String) {
        self.add_index(name, Arc::new(BTreeMemIndex::new(field_id, column)));
    }

    /// Add an HNSW graph over one vector column, with default graph parameters.
    pub fn add_hnsw(
        &mut self,
        name: String,
        field_id: i32,
        column: String,
        distance_type: DistanceType,
        capacity: usize,
        max_batches: usize,
    ) {
        self.add_index(
            name,
            Arc::new(HnswMemIndex::with_capacity(
                field_id,
                column,
                distance_type,
                Default::default(),
                capacity,
                max_batches,
            )),
        );
    }

    /// Add a full-text index over one column, with default parameters.
    pub fn add_fts(&mut self, name: String, field_id: i32, column: String) {
        self.add_index(name, Arc::new(FtsMemIndex::new(field_id, column)));
    }

    /// Add a full-text index over one column.
    pub fn add_fts_with_params(
        &mut self,
        name: String,
        field_id: i32,
        column: String,
        params: InvertedIndexParams,
    ) -> Result<()> {
        self.add_index(
            name,
            Arc::new(FtsMemIndex::try_with_params(field_id, column, params)?),
        );
        Ok(())
    }

    /// Insert a batch into every index.
    pub fn insert(&self, batch: &RecordBatch, row_offset: RowPosition) -> Result<()> {
        self.insert_with_batch_position(batch, row_offset, None)
    }

    /// Insert a batch into every index, then count it indexed if it is the
    /// batch at `batch_position`.
    #[instrument(
        name = "idx_insert_batch",
        level = "debug",
        skip_all,
        fields(num_rows = batch.num_rows(), row_offset, batch_position = ?batch_position)
    )]
    pub fn insert_with_batch_position(
        &self,
        batch: &RecordBatch,
        row_offset: RowPosition,
        batch_position: Option<usize>,
    ) -> Result<()> {
        self.index_rows(Rows::One(batch, row_offset))?;
        if let Some(position) = batch_position {
            self.advance_indexed_count(position + 1);
        }
        Ok(())
    }

    /// Insert the batches of one write into every index, each index on its own
    /// thread above a few dozen rows, then count them indexed.
    #[instrument(name = "idx_insert_batches", level = "debug", skip_all, fields(batch_count = batches.len()))]
    pub fn insert_batches(&self, batches: &[StoredBatch]) -> Result<()> {
        let Some(last) = batches.iter().map(|stored| stored.batch_position).max() else {
            return Ok(());
        };
        self.index_rows(Rows::Batches(batches))?;
        self.advance_indexed_count(last + 1);
        Ok(())
    }

    /// Feed `rows` to every index. A failure stops the writer, so every index
    /// still finishes and the first error is returned.
    fn index_rows(&self, rows: Rows<'_>) -> Result<()> {
        self.has_rows.store(true, Ordering::Release);
        let track_overrides = self.should_track_pk_overrides();

        let mut tasks: Vec<(&str, IndexTask<'_>)> = Vec::with_capacity(self.indexes.len() + 1);
        for (name, index) in &self.indexes {
            let task: IndexTask<'_> = match self.shared_pk_named(name).filter(|_| track_overrides) {
                Some(pk) => Box::new(move || {
                    rows.each(|batch, row_offset| pk.insert_and_report_existing(batch, row_offset))
                }),
                None => Box::new(move || rows.insert_into(index.as_ref()).map(|()| false)),
            };
            tasks.push((name, task));
        }
        if let Some(pk) = self.pk_index.as_ref().and_then(PkIndex::owned) {
            tasks.push((
                "primary key",
                Box::new(move || {
                    rows.each(|batch, row_offset| pk.insert(batch, row_offset, track_overrides))
                }),
            ));
        }

        let inline = matches!(rows, Rows::One(..))
            || tasks.len() < 2
            || rows.num_rows() <= PARALLEL_INDEX_MIN_ROWS;
        let results: Vec<Result<bool>> = if inline {
            tasks.iter().map(|(_, task)| task()).collect()
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = tasks
                    .iter()
                    .map(|(name, task)| (*name, scope.spawn(task)))
                    .collect();
                handles
                    .into_iter()
                    .map(|(name, handle)| {
                        handle.join().unwrap_or_else(|panic| {
                            Err(Error::internal(format!(
                                "index '{name}' panicked: {}",
                                panic_message(panic.as_ref())
                            )))
                        })
                    })
                    .collect()
            })
        };

        let mut had_existing = false;
        for result in results {
            had_existing |= result?;
        }
        // Set before the indexed count advances, so a reader that can see a
        // rewrite also sees the flag.
        self.mark_pk_overrides(had_existing);
        Ok(())
    }

    /// Advance the indexed prefix to at least `count` batches; it never moves
    /// back.
    pub(crate) fn advance_indexed_count(&self, count: usize) {
        self.indexed_count.fetch_max(count, Ordering::AcqRel);
    }

    /// How a filter expression reaches these indexes.
    pub fn filter_catalog(&self) -> &MemIndexCatalog {
        &self.filter_catalog
    }

    /// The index named `name`.
    pub fn get_index(&self, name: &str) -> Option<&Arc<dyn MemIndex>> {
        self.indexes.get(name)
    }

    /// The B-tree the memtable keeps over a single key column, when no user
    /// index serves as the key index.
    pub(crate) fn own_key_index(&self) -> Option<Arc<dyn MemIndex>> {
        match &self.pk_index {
            Some(PkIndex::Owned(OwnedPk::Single(index))) => {
                Some(index.clone() as Arc<dyn MemIndex>)
            }
            _ => None,
        }
    }

    /// The first index, by name, covering `column` that can answer `query`;
    /// the memtable's own key index last.
    pub fn index_answering(&self, column: &str, query: &dyn MemQuery) -> Option<Arc<dyn MemIndex>> {
        self.indexes
            .values()
            .cloned()
            .chain(self.own_key_index())
            .find(|index| {
                index.columns().iter().any(|covered| covered == column) && index.can_answer(query)
            })
    }

    /// The granularities a full-text index on `column` answers, row before list
    /// element.
    pub fn fts_granularities_on(&self, column: &str) -> Vec<DocumentGranularity> {
        [DocumentGranularity::Row, DocumentGranularity::ListElement]
            .into_iter()
            .filter(|granularity| {
                self.index_answering(column, &FtsMemQuery::probe(*granularity))
                    .is_some()
            })
            .collect()
    }

    /// Whether the store holds no named index.
    pub fn is_empty(&self) -> bool {
        self.indexes.is_empty()
    }

    /// Every named index, sorted.
    pub fn index_names(&self) -> Vec<String> {
        self.indexes.keys().cloned().collect()
    }

    /// How many named indexes the store holds.
    pub fn len(&self) -> usize {
        self.indexes.len()
    }

    /// Heap bytes held by every index. Not part of `MemTable::row_bytes`; add
    /// it when budgeting memory.
    pub fn resident_bytes(&self) -> usize {
        let named: usize = self
            .indexes
            .values()
            .map(|index| index.resident_bytes())
            .sum();
        let owned_pk = self
            .pk_index
            .as_ref()
            .and_then(PkIndex::owned)
            .map_or(0, |pk| pk.btree().resident_bytes());
        named + owned_pk
    }

    /// Batches every index holds, as an exclusive count. Readers snapshot
    /// [`Self::visible_count`] instead.
    pub fn indexed_count(&self) -> usize {
        self.indexed_count.load(Ordering::Acquire)
    }

    /// The prefix readers may see: indexed and, under a writer, durable.
    /// Derived on every call, so no cached value can go stale.
    pub fn visible_count(&self) -> usize {
        let indexed = self.indexed_count();
        match &self.durability {
            Some((cursors, global_offset)) => cursors.visible_count(indexed, *global_offset),
            None => indexed,
        }
    }

    /// The prefix readable under `visibility`.
    pub fn prefix_count(&self, visibility: MemTableVisibility) -> usize {
        match visibility {
            MemTableVisibility::Published => self.visible_count(),
            MemTableVisibility::Indexed => self.indexed_count(),
        }
    }

    /// Bind this memtable's indexes to the writer's cursors. Called once at
    /// construction, before the memtable is published.
    pub(crate) fn set_durability(&mut self, cursors: Arc<WriterCursors>, global_offset: usize) {
        self.durability = Some((cursors, global_offset));
    }
}

/// The message a panicking index task left, if it left text.
fn panic_message(panic: &(dyn std::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("no message")
}
