// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Full-text search planner for LSM scanner (local scoring).
//!
//! Builds an execution plan that scores an FTS query across the base
//! table, flushed memtable generations, and the active/frozen-undrained
//! in-memory memtables, returning rows ordered by BM25 `_score` DESC.
//!
//! # Scoring
//!
//! Each source scores with its own corpus statistics (local BM25), and
//! the coordinator unions the per-source plans and merges by `_score`
//! (per-partition top-k sort + sort-preserving merge). The plan is
//! single-pass and never coordinates statistics across sources, so
//! cross-source `_score` values are only approximately comparable — but
//! within each source the ranking is exact. This mirrors the default
//! `query_then_fetch` trade-off of distributed search systems.
//!
//! A globally-consistent scoring mode (aggregate corpus statistics
//! across sources, then rescore) is a deliberate follow-up: the
//! benchmark in this PR shows it carries a real latency penalty, so the
//! local path lands first and the global option is optimized separately.
//!
//! Staleness: within a flushed generation, the deletion vector written
//! at flush time (see #6929) already masks rows superseded by a newer
//! generation, so per-source results are clean within each tier. The
//! same primary key can still appear across tiers (active vs flushed)
//! when an updated row sits in the active memtable while the older
//! copy lives in a flushed generation; cross-tier deduplication is
//! left to the caller in local mode.
//!
//! Everything here is contained in the `mem_wal` module — it reuses the
//! existing per-source FTS read paths (`scanner.full_text_search` for
//! base/flushed Lance datasets, `MemTableScanner` for the active
//! memtable) and requires no changes to `lance-index`.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion::physical_plan::union::UnionExec;
use lance_core::{Error, ROW_ID, Result, is_system_column};
use lance_index::scalar::FullTextSearchQuery;
use lance_index::scalar::inverted::query::FtsQuery as IndexFtsQuery;
use tracing::instrument;

use super::block_list::compute_source_block_lists;
use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::exec::{DedupDirection, PkHashFilterExec, WithinSourceDedupExec};
use super::flushed_cache::{FlushedMemTableCache, open_flushed_dataset};
use super::projection::project_to_canonical;
use crate::dataset::mem_wal::memtable::scanner::MemTableScanner;
use crate::session::Session;

/// `_score` column name in FTS results — kept aligned with
/// `lance_index::scalar::inverted::SCORE_COL` so this module doesn't
/// require an import for one string constant.
pub const SCORE_COLUMN: &str = "_score";

/// Default over-fetch multiple for blocked sources. `1.0` keeps cross-generation
/// dedup on with no over-fetch; callers (e.g. the sophon WAL handler) raise it
/// so a blocked source still yields `k` live rows after the block-list filter.
const DEFAULT_OVERFETCH_FACTOR: f64 = 1.0;

/// Plans local-scoring FTS queries over LSM data.
pub struct LsmFtsSearchPlanner {
    collector: LsmDataSourceCollector,
    pk_columns: Vec<String>,
    base_schema: SchemaRef,
    /// Session threaded into flushed-generation opens (shared caches).
    session: Option<Arc<Session>>,
    /// Cache of opened flushed-generation datasets.
    flushed_cache: Option<Arc<FlushedMemTableCache>>,
    /// Over-fetch multiple for blocked sources (clamped to `>= 1.0`).
    overfetch_factor: f64,
}

impl LsmFtsSearchPlanner {
    /// Create a new planner.
    pub fn new(
        collector: LsmDataSourceCollector,
        pk_columns: Vec<String>,
        base_schema: SchemaRef,
    ) -> Self {
        Self {
            collector,
            pk_columns,
            base_schema,
            session: None,
            flushed_cache: None,
            overfetch_factor: DEFAULT_OVERFETCH_FACTOR,
        }
    }

    /// Set the over-fetch multiple for blocked sources so they still yield `k`
    /// live rows after cross-generation block-list filtering. Clamped to `>= 1.0`.
    pub fn with_overfetch_factor(mut self, factor: f64) -> Self {
        self.overfetch_factor = factor;
        self
    }

    /// Thread a session into flushed-generation opens so the first open
    /// populates the shared index / file-metadata caches.
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Inject a cache of opened flushed-generation datasets, making repeated
    /// searches against the same generation a pure `Arc::clone`.
    pub fn with_flushed_cache(mut self, cache: Arc<FlushedMemTableCache>) -> Self {
        self.flushed_cache = Some(cache);
        self
    }

    /// Build the FTS execution plan (local scoring).
    ///
    /// # Arguments
    ///
    /// * `column` — text column to search.
    /// * `query` — the FTS query (match / phrase / boolean / fuzzy for
    ///   base/flushed Lance sources; the active memtable currently
    ///   supports `MatchQuery`).
    /// * `k` — global top-k to return.
    /// * `projection` — user columns to project. PK columns are
    ///   auto-included; `_score` is always appended.
    ///
    /// Each source is scored independently (local BM25), normalized to a
    /// canonical schema, unioned, and merged by `_score` DESC with a
    /// top-k cap pushed into each partition.
    #[instrument(
        name = "lsm_fts_search",
        level = "info",
        skip_all,
        fields(column = %column, k)
    )]
    pub async fn plan_search(
        &self,
        column: &str,
        query: FullTextSearchQuery,
        k: usize,
        projection: Option<&[String]>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let sources = self.collector.collect()?;
        let target_schema = self.canonical_fts_schema(projection);

        if sources.is_empty() {
            return self.empty_plan(&target_schema);
        }

        // Per-source PK-hash block sets for cross-generation dedup (NEWER(G)
        // per shard; base = union of all gens). Query-type-agnostic — same
        // call the vector planner makes. `Box::pin` keeps the future off
        // `clippy::large_futures`.
        let block_lists = Box::pin(compute_source_block_lists(
            &sources,
            &self.pk_columns,
            self.session.as_ref(),
            self.flushed_cache.as_ref(),
        ))
        .await?;
        let overfetch = self.overfetch_factor.max(1.0);

        // Stage the per-source over-fetch decisions, then build every source
        // plan concurrently — the builds are independent and a sequential loop
        // was the dominant serial planning cost at multiple generations.
        let arm_inputs: Vec<_> = sources
            .iter()
            .map(|source| {
                let is_active = matches!(source, LsmDataSource::ActiveMemTable { .. });
                let blocked = block_lists.get(&(source.shard_id(), source.generation()));
                // Over-fetch a blocked source so the post-filter still yields k live
                // rows. The active arm returns all matches (no builder limit), so its
                // within-source dedup needs no over-fetch hint.
                let fetch_k = if blocked.is_some() {
                    ((k as f64) * overfetch).ceil() as usize
                } else {
                    k
                };
                (source, is_active, blocked, fetch_k)
            })
            .collect();
        let built =
            futures::future::try_join_all(arm_inputs.iter().map(
                |(source, is_active, _, fetch_k)| {
                    Box::pin(self.build_source_plan(
                        source, column, &query, *fetch_k, projection, *is_active,
                    ))
                },
            ))
            .await?;

        let mut per_source_plans: Vec<Arc<dyn ExecutionPlan>> = Vec::with_capacity(sources.len());
        for ((_, is_active, blocked, _), plan) in arm_inputs.iter().zip(built) {
            let is_active = *is_active;
            let blocked = *blocked;
            // Dedup, mirroring LsmVectorSearchPlanner:
            //  * active: collapse duplicate-PK appends to the newest insert
            //    (larger _rowid = inserted later). The FTS index is append-only,
            //    so an in-memtable update leaves both versions searchable.
            //  * flushed/base: drop rows superseded by a newer generation via the
            //    block-list (within-gen is handled by the flushed deletion vector).
            let deduped = if is_active {
                Arc::new(WithinSourceDedupExec::new(
                    plan,
                    self.pk_columns.clone(),
                    ROW_ID,
                    DedupDirection::KeepMaxRowAddr,
                )) as Arc<dyn ExecutionPlan>
            } else if let Some(set) = blocked {
                Arc::new(PkHashFilterExec::new(
                    plan,
                    self.pk_columns.clone(),
                    set.clone(),
                    k,
                )) as Arc<dyn ExecutionPlan>
            } else {
                plan
            };

            // Normalize to canonical. This also drops the active arm's _rowid,
            // which the canonical FTS schema omits — it served only the dedup.
            let normalized = project_to_canonical(deduped, &target_schema)?;
            per_source_plans.push(normalized);
        }

        // Single source: skip Union and the merge.
        let merged: Arc<dyn ExecutionPlan> = if per_source_plans.len() == 1 {
            per_source_plans.into_iter().next().unwrap()
        } else {
            #[allow(deprecated)]
            // The downstream `SortPreservingMergeExec` already spawns one driver
            // task per input partition (one per union arm) via `spawn_buffered`,
            // so each arm's per-arm CPU (posting decode, BM25) runs on its own
            // task without an extra repartition.
            Arc::new(UnionExec::new(per_source_plans))
        };

        let score_idx = merged.schema().index_of(SCORE_COLUMN).map_err(|_| {
            Error::internal(format!(
                "{SCORE_COLUMN} missing from canonical FTS schema after merge"
            ))
        })?;

        let sort_expr = vec![PhysicalSortExpr {
            expr: Arc::new(Column::new(SCORE_COLUMN, score_idx)),
            options: SortOptions {
                descending: true,
                nulls_first: false,
            },
        }];
        let lex_ordering = LexOrdering::new(sort_expr).ok_or_else(|| {
            Error::internal("Failed to build LexOrdering for FTS _score sort".to_string())
        })?;

        // Per-partition sort with `fetch=k` so each upstream partition
        // can early-terminate at k; the preserving merge then does a
        // K-way heap merge also capped at k. Same pattern as
        // LsmVectorSearchPlanner.
        let per_partition_sorted: Arc<dyn ExecutionPlan> = Arc::new(
            SortExec::new(lex_ordering.clone(), merged)
                .with_preserve_partitioning(true)
                .with_fetch(Some(k)),
        );
        let merged_sorted: Arc<dyn ExecutionPlan> = Arc::new(
            SortPreservingMergeExec::new(lex_ordering, per_partition_sorted).with_fetch(Some(k)),
        );

        Ok(merged_sorted)
    }

    async fn build_source_plan(
        &self,
        source: &LsmDataSource,
        column: &str,
        query: &FullTextSearchQuery,
        k: usize,
        projection: Option<&[String]>,
        emit_row_id: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match source {
            LsmDataSource::BaseTable { dataset } => {
                let mut scanner = dataset.scan();
                let mut cols = self.fts_scanner_projection(projection);
                // fts_scanner_projection strips _score; add it back since the
                // dataset scanner no longer auto-projects scoring columns for
                // explicit projections.
                if !cols.iter().any(|c| c == SCORE_COLUMN) {
                    cols.push(SCORE_COLUMN.to_string());
                }
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                let bound_query = query
                    .clone()
                    .with_column(column.to_string())?
                    .limit(Some(k as i64));
                scanner.full_text_search(bound_query)?;
                scanner.create_plan().await
            }
            LsmDataSource::FlushedMemTable { path, .. } => {
                let dataset =
                    open_flushed_dataset(path, self.session.as_ref(), self.flushed_cache.as_ref())
                        .await?;
                let mut scanner = dataset.scan();
                let mut cols = self.fts_scanner_projection(projection);
                if !cols.iter().any(|c| c == SCORE_COLUMN) {
                    cols.push(SCORE_COLUMN.to_string());
                }
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                let bound_query = query
                    .clone()
                    .with_column(column.to_string())?
                    .limit(Some(k as i64));
                scanner.full_text_search(bound_query)?;
                scanner.create_plan().await
            }
            LsmDataSource::ActiveMemTable {
                batch_store,
                index_store,
                schema,
                ..
            } => {
                let mut scanner =
                    MemTableScanner::new(batch_store.clone(), index_store.clone(), schema.clone());
                let cols = self.fts_scanner_projection(projection);
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                // Emit `_rowid` (row position) so the planner can collapse
                // duplicate-PK appends via WithinSourceDedupExec before the union.
                if emit_row_id {
                    scanner.with_row_id();
                }
                // `MemTableScanner::full_text_search` takes a raw match
                // string; richer query shapes (phrase/boolean/fuzzy) can
                // be plumbed through once the MemTable scanner accepts a
                // structured query.
                let match_str = match &query.query {
                    IndexFtsQuery::Match(m) => m.terms.clone(),
                    other => {
                        return Err(Error::not_supported(format!(
                            "Active memtable FTS via LsmFtsSearchPlanner currently only \
                             supports MatchQuery, got: {other:?}"
                        )));
                    }
                };
                let _ = scanner.full_text_search(column, &match_str);
                // Active arm doesn't take a top-K hint via the builder
                // today; the per-partition Sort+fetch above bounds the
                // emitted rows.
                let _ = k;
                scanner.create_plan().await
            }
        }
    }

    /// Columns to pass to the underlying scanner: user projection minus
    /// system / `_score`, with PK columns appended.
    fn fts_scanner_projection(&self, user_projection: Option<&[String]>) -> Vec<String> {
        let mut cols: Vec<String> = if let Some(p) = user_projection {
            p.iter()
                .filter(|c| !is_system_column(c) && c.as_str() != SCORE_COLUMN)
                .cloned()
                .collect()
        } else {
            self.base_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        };
        for pk in &self.pk_columns {
            if !cols.contains(pk) {
                cols.push(pk.clone());
            }
        }
        cols
    }

    /// Canonical FTS output: user-projected cols + PK + `_score`.
    fn canonical_fts_schema(&self, user_projection: Option<&[String]>) -> SchemaRef {
        let mut ordered: Vec<String> = if let Some(p) = user_projection {
            p.to_vec()
        } else {
            self.base_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect()
        };
        for pk in &self.pk_columns {
            if !ordered.contains(pk) {
                ordered.push(pk.clone());
            }
        }
        if !ordered.iter().any(|c| c == SCORE_COLUMN) {
            ordered.push(SCORE_COLUMN.to_string());
        }
        let fields: Vec<Arc<Field>> = ordered
            .iter()
            .filter_map(|name| {
                if name == SCORE_COLUMN {
                    Some(Arc::new(Field::new(SCORE_COLUMN, DataType::Float32, true)))
                } else if is_system_column(name) {
                    Some(Arc::new(Field::new(name.clone(), DataType::UInt64, true)))
                } else {
                    self.base_schema
                        .field_with_name(name)
                        .ok()
                        .map(|f| Arc::new(f.clone()))
                }
            })
            .collect();
        Arc::new(Schema::new(fields))
    }

    fn empty_plan(&self, schema: &SchemaRef) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_plan::empty::EmptyExec;
        Ok(Arc::new(EmptyExec::new(schema.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
    use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
    use crate::dataset::{Dataset, WriteParams};
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use std::collections::HashMap;

    fn fts_schema() -> Arc<ArrowSchema> {
        let mut id_meta = HashMap::new();
        id_meta.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(id_meta);
        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new("text", DataType::Utf8, true),
        ]))
    }

    fn make_batch(schema: &ArrowSchema, ids: &[i32], texts: &[&str]) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(texts.to_vec())),
            ],
        )
        .unwrap()
    }

    async fn write_dataset(uri: &str, batches: Vec<RecordBatch>) -> Dataset {
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn local_mode_unions_base_and_active_with_consistent_score_schema() {
        // Regression for the `_score` nullability mismatch between
        // FtsIndexExec (active arm) and FTS_SCHEMA (base/flushed). The
        // active-only test below would not catch this — UnionExec rejects
        // schema-inequality, so we need at least one base + one active
        // source to exercise that code path.
        use crate::index::DatasetIndexExt;
        use lance_index::IndexType;
        use lance_index::scalar::inverted::tokenizer::InvertedIndexParams;

        let schema = fts_schema();
        let tmp = tempfile::tempdir().unwrap();

        // Base Lance dataset with FTS index on the `text` column.
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        let mut base_ds = write_dataset(
            &base_uri,
            vec![make_batch(
                &schema,
                &[1, 2],
                &["lance rocks", "unrelated text"],
            )],
        )
        .await;
        base_ds
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("text_fts".to_string()),
                &InvertedIndexParams::default(),
                false,
            )
            .await
            .unwrap();
        let base_ds = Arc::new(Dataset::open(&base_uri).await.unwrap());

        // Active memtable with its own FTS index, containing a matching row.
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut indexes = IndexStore::new();
        indexes.add_fts("text_fts".to_string(), 1, "text".to_string());
        let active_batch = make_batch(
            &schema,
            &[3, 4],
            &["lance memwal goes fast", "completely unrelated"],
        );
        batch_store.append(active_batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&active_batch, 0, Some(0))
            .unwrap();
        let indexes = Arc::new(indexes);

        let collector = LsmDataSourceCollector::new(base_ds, vec![]).with_in_memory_memtables(
            uuid::Uuid::new_v4(),
            InMemoryMemTables {
                active: InMemoryMemTableRef {
                    batch_store,
                    index_store: indexes,
                    schema: schema.clone(),
                    generation: 1,
                },
                frozen: vec![],
            },
        );

        let planner = LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema);
        let plan = planner
            .plan_search(
                "text",
                FullTextSearchQuery::new("lance".to_string()),
                10,
                None,
            )
            .await
            .expect("planner should produce a base+active union plan");

        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        // Both base id=1 ("lance rocks") and active id=3 ("lance memwal ...")
        // should match. id=2 / id=4 do not contain "lance".
        assert!(
            total >= 2,
            "expected at least the 2 'lance' rows from base+active, got {total}"
        );

        // Both sources must agree on _score nullability — verifies the fix.
        let out = batches[0].schema();
        let score_field = out
            .field_with_name(SCORE_COLUMN)
            .expect("_score column missing from output");
        assert!(
            score_field.is_nullable(),
            "_score must be nullable to stay union-compatible across base+active"
        );

        // Sanity: ids contain at least one base hit (id=1) and one active hit (id=3).
        let mut ids: Vec<i32> = Vec::new();
        for b in &batches {
            let col = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            for i in 0..b.num_rows() {
                ids.push(col.value(i));
            }
        }
        assert!(ids.contains(&1), "missing base hit id=1; got ids={ids:?}");
        assert!(ids.contains(&3), "missing active hit id=3; got ids={ids:?}");
    }

    #[tokio::test]
    async fn local_mode_active_memtable_only_returns_score_sorted_hits() {
        let schema = fts_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut indexes = IndexStore::new();
        // text column has field_id 1 in fts_schema()
        indexes.add_fts("text_fts".to_string(), 1, "text".to_string());
        let batch = make_batch(
            &schema,
            &[1, 2, 3, 4],
            &[
                "lance is a columnar data format",
                "memwal handles streaming writes",
                "lance memwal lance lance",
                "completely unrelated",
            ],
        );
        batch_store.append(batch.clone()).unwrap();
        indexes
            .insert_with_batch_position(&batch, 0, Some(0))
            .unwrap();
        let indexes = Arc::new(indexes);

        let tmp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                uuid::Uuid::new_v4(),
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store: indexes,
                        schema: schema.clone(),
                        generation: 1,
                    },
                    frozen: vec![],
                },
            );

        let planner = LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema);
        let plan = planner
            .plan_search(
                "text",
                FullTextSearchQuery::new("lance".to_string()),
                10,
                None,
            )
            .await
            .expect("local mode planner should produce a plan");

        // Plan executes and emits _score-sorted rows.
        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert!(
            total >= 2,
            "expected at least the 2 'lance' rows, got {total}"
        );

        // Schema must include _score and the PK id.
        let out = batches[0].schema();
        assert!(out.field_with_name(SCORE_COLUMN).is_ok());
        assert!(out.field_with_name("id").is_ok());

        // _score must be non-ascending across the result.
        let mut prev_score: Option<f32> = None;
        for batch in &batches {
            let score = batch
                .column_by_name(SCORE_COLUMN)
                .unwrap()
                .as_any()
                .downcast_ref::<arrow_array::Float32Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                let s = score.value(i);
                if let Some(p) = prev_score {
                    assert!(p >= s, "scores not sorted DESC: {p} then {s}");
                }
                prev_score = Some(s);
            }
        }
    }

    #[tokio::test]
    async fn local_mode_active_dedups_updated_pk_keeping_newest() {
        // The active memtable is an append log and the FTS index is
        // append-only, so a PK updated before flush is searchable as two
        // row-positions. WithinSourceDedupExec(KeepMaxRowAddr) must collapse
        // them to the newest insert. Without it the same PK would surface
        // twice (criterion 2 violation).
        let schema = fts_schema();
        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut indexes = IndexStore::new();
        indexes.add_fts("text_fts".to_string(), 1, "text".to_string());

        // First append (positions 0,1): id=1 is the stale version of the PK.
        let batch_old = make_batch(&schema, &[1, 2], &["lance stale version", "other doc"]);
        batch_store.append(batch_old.clone()).unwrap();
        indexes
            .insert_with_batch_position(&batch_old, 0, Some(0))
            .unwrap();

        // Second append (position 2): id=1 updated — same PK, later row.
        let batch_new = make_batch(&schema, &[1], &["lance fresh version"]);
        batch_store.append(batch_new.clone()).unwrap();
        indexes
            .insert_with_batch_position(&batch_new, 2, Some(1))
            .unwrap();
        let indexes = Arc::new(indexes);

        let tmp = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", tmp.path().to_str().unwrap());
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                uuid::Uuid::new_v4(),
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store: indexes,
                        schema: schema.clone(),
                        generation: 1,
                    },
                    frozen: vec![],
                },
            );

        let planner = LsmFtsSearchPlanner::new(collector, vec!["id".to_string()], schema);
        let plan = planner
            .plan_search(
                "text",
                FullTextSearchQuery::new("lance".to_string()),
                10,
                None,
            )
            .await
            .expect("planner should produce an active-only plan");

        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let mut rows: Vec<(i32, String)> = Vec::new();
        for b in &batches {
            let ids = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let texts = b
                .column_by_name("text")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..b.num_rows() {
                rows.push((ids.value(i), texts.value(i).to_string()));
            }
        }

        // id=1 must appear exactly once, and it must be the *newest* version.
        let id1: Vec<&(i32, String)> = rows.iter().filter(|(id, _)| *id == 1).collect();
        assert_eq!(
            id1.len(),
            1,
            "updated PK id=1 must be deduped to one row; got {rows:?}"
        );
        assert_eq!(
            id1[0].1, "lance fresh version",
            "dedup must keep the newest (max row-position) version"
        );
    }
}
