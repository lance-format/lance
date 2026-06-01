// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Point lookup planner for LSM scanner.
//!
//! Provides efficient primary key-based point lookups across LSM levels.

use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::common::ScalarValue;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::prelude::Expr;
use lance_core::Result;
use lance_core::utils::bloomfilter::sbbf::Sbbf;
use tracing::instrument;

use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use super::exec::{
    BloomFilterGuardExec, CoalesceFirstExec, DedupDirection, WithinSourceDedupExec,
    compute_pk_hash_from_scalars,
};
use super::flushed_cache::{FlushedMemTableCache, open_flushed_dataset};
use super::projection::{
    build_scanner_projection, canonical_output_schema, null_columns, project_to_canonical,
    wants_row_address, wants_row_id,
};
use crate::session::Session;

/// Plans point lookup queries over LSM data.
///
/// Point lookups are optimized for primary key-based queries where we expect
/// to find at most one row. The query plan uses:
///
/// 1. **Bloom filter guards**: Skip generations that definitely don't contain the key
/// 2. **Short-circuit evaluation**: Stop after finding the first match
/// 3. **Newest-first ordering**: Check newer generations before older ones
///
/// # Query Plan Structure
///
/// Since data is stored in reverse order (newest first), we use `GlobalLimitExec`
/// with limit=1 to take the first (most recent) matching row.
///
/// ```text
/// CoalesceFirstExec: return_first_non_null
///   BloomFilterGuardExec: gen=3
///     GlobalLimitExec: limit=1
///       FilterExec: pk = target
///         ScanExec: memtable_gen_3
///   BloomFilterGuardExec: gen=2
///     GlobalLimitExec: limit=1
///       FilterExec: pk = target
///         ScanExec: flushed_gen_2
///   BloomFilterGuardExec: gen=1
///     GlobalLimitExec: limit=1
///       FilterExec: pk = target
///         ScanExec: flushed_gen_1
///   GlobalLimitExec: limit=1
///     FilterExec: pk = target
///       ScanExec: base_table
/// ```
///
/// The base table doesn't use a bloom filter guard because:
/// - It's the fallback when no memtable has the key
/// - Bloom filters for the base table would be too large
pub struct LsmPointLookupPlanner {
    /// Data source collector.
    collector: LsmDataSourceCollector,
    /// Primary key column names.
    pk_columns: Vec<String>,
    /// Schema of the base table.
    base_schema: SchemaRef,
    /// Bloom filters for each memtable generation.
    /// Map: generation -> bloom filter
    bloom_filters: std::collections::HashMap<u64, Arc<Sbbf>>,
    /// Session threaded into flushed-generation opens (shared caches).
    session: Option<Arc<Session>>,
    /// Cache of opened flushed-generation datasets.
    flushed_cache: Option<Arc<FlushedMemTableCache>>,
}

impl LsmPointLookupPlanner {
    /// Create a new planner.
    ///
    /// # Arguments
    ///
    /// * `collector` - Data source collector
    /// * `pk_columns` - Primary key column names
    /// * `base_schema` - Schema of the base table
    pub fn new(
        collector: LsmDataSourceCollector,
        pk_columns: Vec<String>,
        base_schema: SchemaRef,
    ) -> Self {
        Self {
            collector,
            pk_columns,
            base_schema,
            bloom_filters: std::collections::HashMap::new(),
            session: None,
            flushed_cache: None,
        }
    }

    /// Thread a session into flushed-generation opens so the first open
    /// populates the shared index / file-metadata caches.
    pub fn with_session(mut self, session: Arc<Session>) -> Self {
        self.session = Some(session);
        self
    }

    /// Inject a cache of opened flushed-generation datasets, making repeated
    /// lookups against the same generation a pure `Arc::clone`.
    pub fn with_flushed_cache(mut self, cache: Arc<FlushedMemTableCache>) -> Self {
        self.flushed_cache = Some(cache);
        self
    }

    /// Add a bloom filter for a generation.
    ///
    /// Bloom filters are optional but improve performance by skipping
    /// generations that definitely don't contain the target key.
    pub fn with_bloom_filter(mut self, generation: u64, bloom_filter: Arc<Sbbf>) -> Self {
        self.bloom_filters.insert(generation, bloom_filter);
        self
    }

    /// Add multiple bloom filters.
    pub fn with_bloom_filters(
        mut self,
        bloom_filters: impl IntoIterator<Item = (u64, Arc<Sbbf>)>,
    ) -> Self {
        self.bloom_filters.extend(bloom_filters);
        self
    }

    /// Create a point lookup plan for the given primary key values.
    ///
    /// # Arguments
    ///
    /// * `pk_values` - Primary key values to look up (one value per pk column)
    /// * `projection` - Columns to include in output (None = all columns)
    ///
    /// # Returns
    ///
    /// An execution plan that returns at most one row - the newest version
    /// of the row with the given primary key.
    #[instrument(name = "lsm_point_lookup", level = "debug", skip_all, fields(pk_column_count = self.pk_columns.len()))]
    pub async fn plan_lookup(
        &self,
        pk_values: &[ScalarValue],
        projection: Option<&[String]>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if pk_values.len() != self.pk_columns.len() {
            return Err(lance_core::Error::invalid_input(format!(
                "Expected {} primary key values, got {}",
                self.pk_columns.len(),
                pk_values.len()
            )));
        }

        let pk_hash = compute_pk_hash_from_scalars(pk_values);
        let filter_expr = self.build_pk_filter_expr(pk_values)?;
        let sources = self.collector.collect()?;

        if sources.is_empty() {
            return self.empty_plan(projection);
        }

        // Sort by generation DESC (newest first)
        let mut sources: Vec<_> = sources.into_iter().collect();
        sources.sort_by_key(|b| std::cmp::Reverse(b.generation()));

        let mut source_plans = Vec::new();

        for source in sources {
            let generation = source.generation().as_u64();

            let scan = self
                .build_source_scan(&source, projection, &filter_expr)
                .await?;

            // Data is stored in reverse order, so first match is newest
            let limited: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(scan, 0, Some(1)));

            let guarded_plan: Arc<dyn ExecutionPlan> =
                if let Some(bf) = self.bloom_filters.get(&generation) {
                    Arc::new(BloomFilterGuardExec::new(
                        limited,
                        bf.clone(),
                        pk_hash,
                        generation,
                    ))
                } else {
                    limited
                };

            source_plans.push(guarded_plan);
        }

        let plan: Arc<dyn ExecutionPlan> = if source_plans.len() == 1 {
            source_plans.remove(0)
        } else {
            Arc::new(CoalesceFirstExec::new(source_plans))
        };

        Ok(plan)
    }

    /// Build the filter expression for primary key equality.
    fn build_pk_filter_expr(&self, pk_values: &[ScalarValue]) -> Result<Expr> {
        use datafusion::prelude::{col, lit};

        let mut expr: Option<Expr> = None;

        for (col_name, value) in self.pk_columns.iter().zip(pk_values.iter()) {
            let eq_expr = col(col_name.as_str()).eq(lit(value.clone()));

            expr = Some(match expr {
                Some(e) => e.and(eq_expr),
                None => eq_expr,
            });
        }

        expr.ok_or_else(|| lance_core::Error::invalid_input("No primary key columns specified"))
    }

    /// Build scan plan for a single data source.
    ///
    /// Output is projected to the canonical schema so user-requested system
    /// columns appear at the requested position — NULL where the source
    /// doesn't produce them or where per-source values aren't meaningful.
    async fn build_source_scan(
        &self,
        source: &LsmDataSource,
        projection: Option<&[String]>,
        filter: &Expr,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let cols = build_scanner_projection(projection, &self.base_schema, &self.pk_columns);
        let target =
            canonical_output_schema(projection, &self.base_schema, &self.pk_columns, false);
        let want_row_id = wants_row_id(projection);
        let want_row_addr = wants_row_address(projection);
        let scan: Arc<dyn ExecutionPlan> = match source {
            LsmDataSource::BaseTable { dataset } => {
                let mut scanner = dataset.scan();
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                // Only the base produces row IDs callers can use against the
                // dataset (e.g. `take_rows`); non-base arms NULL via canonical.
                if want_row_id {
                    scanner.with_row_id();
                }
                if want_row_addr {
                    scanner.with_row_address();
                }
                scanner.filter_expr(filter.clone());
                scanner.create_plan().await?
            }
            LsmDataSource::FlushedMemTable { path, .. } => {
                let dataset =
                    open_flushed_dataset(path, self.session.as_ref(), self.flushed_cache.as_ref())
                        .await?;
                let mut scanner = dataset.scan();
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>())?;
                scanner.filter_expr(filter.clone());
                scanner.create_plan().await?
            }
            LsmDataSource::ActiveMemTable {
                batch_store,
                index_store,
                schema,
                ..
            } => {
                use crate::dataset::mem_wal::memtable::scanner::MemTableScanner;

                let mut scanner =
                    MemTableScanner::new(batch_store.clone(), index_store.clone(), schema.clone());
                scanner.project(&cols.iter().map(|s| s.as_str()).collect::<Vec<_>>());
                scanner.filter_expr(filter.clone());
                // Expose `_rowid` (the BatchStore row offset, monotonic with
                // insert order) so we can pick the most recently inserted
                // duplicate below. Without this, a `FilterExec → LIMIT 1`
                // over insert-ordered scan would return the *oldest* of
                // multiple rows sharing the target primary key.
                scanner.with_row_id();
                let raw = scanner.create_plan().await?;
                // Within the active memtable, larger `_rowid` = newer
                // insert. After dedup there is exactly one row per PK.
                let deduped: Arc<dyn ExecutionPlan> = Arc::new(WithinSourceDedupExec::new(
                    raw,
                    self.pk_columns.clone(),
                    lance_core::ROW_ID,
                    DedupDirection::KeepMaxRowAddr,
                ));
                // Per-source `_rowid` would collide with the base table's;
                // NULL it before canonicalization (the value is internal to
                // this arm). project_to_canonical drops it entirely when
                // the user didn't request `_rowid` in the projection.
                null_columns(deduped, &[lance_core::ROW_ID])?
            }
        };
        project_to_canonical(scan, &target)
    }

    /// Create an empty execution plan with the canonical output schema.
    fn empty_plan(&self, projection: Option<&[String]>) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_plan::empty::EmptyExec;

        let schema =
            canonical_output_schema(projection, &self.base_schema, &self.pk_columns, false);
        Ok(Arc::new(EmptyExec::new(schema)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator, StringArray};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use datafusion::physical_plan::displayable;
    use std::collections::HashMap;
    use uuid::Uuid;

    use crate::dataset::mem_wal::scanner::data_source::ShardSnapshot;
    use crate::dataset::{Dataset, WriteParams};

    fn create_pk_schema() -> Arc<ArrowSchema> {
        let mut id_metadata = HashMap::new();
        id_metadata.insert(
            "lance-schema:unenforced-primary-key".to_string(),
            "true".to_string(),
        );
        let id_field = Field::new("id", DataType::Int32, false).with_metadata(id_metadata);

        Arc::new(ArrowSchema::new(vec![
            id_field,
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    fn create_test_batch(schema: &ArrowSchema, ids: &[i32], name_prefix: &str) -> RecordBatch {
        let names: Vec<String> = ids
            .iter()
            .map(|id| format!("{}_{}", name_prefix, id))
            .collect();
        RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(ids.to_vec())),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    async fn create_dataset(uri: &str, batches: Vec<RecordBatch>) -> Dataset {
        let schema = batches[0].schema();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        Dataset::write(reader, uri, Some(WriteParams::default()))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_point_lookup_plan_structure() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create collector without memtables
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone());

        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        // Verify plan structure
        let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));

        // Should have GlobalLimitExec with limit=1 (data is stored in reverse order)
        assert!(
            plan_str.contains("GlobalLimitExec"),
            "Should have GlobalLimitExec in plan: {}",
            plan_str
        );
    }

    #[tokio::test]
    async fn test_point_lookup_with_memtables() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create shard snapshot
        let shard_id = Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, shard_id);
        let gen1_batch = create_test_batch(&schema, &[2], "gen1"); // Update id=2
        create_dataset(&gen1_uri, vec![gen1_batch]).await;

        let shard_snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(2)
            .with_flushed_generation(1, "gen_1".to_string());

        // Create collector
        let collector = LsmDataSourceCollector::new(base_dataset, vec![shard_snapshot]);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone());

        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        // Verify plan structure - should have CoalesceFirstExec with multiple children
        let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));

        assert!(
            plan_str.contains("CoalesceFirstExec") || plan_str.contains("GlobalLimitExec"),
            "Should have CoalesceFirstExec or GlobalLimitExec in plan: {}",
            plan_str
        );
    }

    #[tokio::test]
    async fn test_point_lookup_with_bloom_filter() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // Create base table
        let base_uri = format!("{}/base", base_path);
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        // Create collector
        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        // Create a bloom filter for generation 1 (simulating a memtable)
        let mut bf = Sbbf::with_ndv_fpp(100, 0.01).unwrap();
        let pk_hash = compute_pk_hash_from_scalars(&[ScalarValue::Int32(Some(2))]);
        bf.insert_hash(pk_hash);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema.clone())
            .with_bloom_filter(1, Arc::new(bf));

        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        // Plan should be valid
        assert!(plan.schema().field_with_name("id").is_ok());
    }

    #[tokio::test]
    async fn test_pk_filter_expr() {
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[1], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let pk_values = vec![ScalarValue::Int32(Some(42))];
        let expr = planner.build_pk_filter_expr(&pk_values).unwrap();

        // Verify expression is an equality
        let expr_str = format!("{}", expr);
        assert!(
            expr_str.contains("id"),
            "Expression should contain column name"
        );
    }

    #[tokio::test]
    async fn test_point_lookup_without_base_table() {
        use futures::TryStreamExt;

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();

        // No base dataset is created. We still need a base URI so the collector
        // can resolve flushed-generation paths.
        let base_uri = format!("{}/base", base_path);

        // Create a flushed generation under {base_uri}/_mem_wal/{shard}/gen_1
        let shard_id = Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, shard_id);
        let gen1_batch = create_test_batch(&schema, &[2, 3], "gen1");
        create_dataset(&gen1_uri, vec![gen1_batch]).await;

        let shard_snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(2)
            .with_flushed_generation(1, "gen_1".to_string());

        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![shard_snapshot]);
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        // id=3 lives in the flushed generation
        let pk_values = vec![ScalarValue::Int32(Some(3))];
        let plan = planner.plan_lookup(&pk_values, None).await.unwrap();

        let plan_str = format!("{}", displayable(plan.as_ref()).indent(true));
        assert!(
            !plan_str.contains("base/data"),
            "Plan must not scan base table, got: {}",
            plan_str
        );
        assert!(plan_str.contains("gen_1"));

        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);

        // id=99 doesn't exist anywhere → empty
        let plan = planner
            .plan_lookup(&[ScalarValue::Int32(Some(99))], None)
            .await
            .unwrap();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[tokio::test]
    async fn test_point_lookup_projection_with_system_columns() {
        // Regression: system columns in projection used to error in the
        // active-arm MemTableScanner or get silently dropped. Verify they're
        // surfaced at the requested position with the correct NULL/real mix.
        use futures::TryStreamExt;
        use lance_core::is_system_column;

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());
        let base_batch = create_test_batch(&schema, &[1, 2, 3], "base");
        let base_dataset = Arc::new(create_dataset(&base_uri, vec![base_batch]).await);

        let collector = LsmDataSourceCollector::new(base_dataset, vec![]);
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        // User requests `_rowaddr` between `id` and `name`, plus `_rowoffset` at end.
        let projection = vec![
            "id".to_string(),
            "_rowaddr".to_string(),
            "name".to_string(),
            "_rowoffset".to_string(),
        ];
        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner
            .plan_lookup(&pk_values, Some(&projection))
            .await
            .expect("planner must accept system columns in projection");

        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "expected exactly one matching row");

        let out_schema = batches[0].schema();
        let out_cols: Vec<String> = out_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            out_cols,
            vec![
                "id".to_string(),
                "_rowaddr".to_string(),
                "name".to_string(),
                "_rowoffset".to_string(),
            ],
            "system columns must appear at the user's requested position"
        );

        // Hit row is from base → `_rowaddr` is real. `_rowoffset` stays
        // NULL (no scanner produces it).
        // (Test 5 — empty-plan with system columns — lives in the next
        // test below.)
        let rowaddr = batches[0].column_by_name("_rowaddr").unwrap();
        assert!(
            !rowaddr.is_null(0),
            "_rowaddr from base should be populated, got: {:?}",
            rowaddr
        );
        let rowoffset = batches[0].column_by_name("_rowoffset").unwrap();
        assert!(is_system_column("_rowoffset"));
        assert!(
            rowoffset.is_null(0),
            "_rowoffset has no per-source flag, must be NULL across LSM, got: {:?}",
            rowoffset
        );
    }

    #[tokio::test]
    async fn test_point_lookup_empty_plan_with_system_columns() {
        // Test 5 (point_lookup slice): with no sources, the empty plan
        // must still expose user-requested system columns at the
        // requested position.
        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());

        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![]);
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let projection = vec![
            "id".to_string(),
            "_rowaddr".to_string(),
            "name".to_string(),
            "_rowid".to_string(),
        ];
        let pk_values = vec![ScalarValue::Int32(Some(2))];
        let plan = planner
            .plan_lookup(&pk_values, Some(&projection))
            .await
            .expect("empty plan must accept system columns in projection");

        let names: Vec<String> = plan
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(
            names,
            vec![
                "id".to_string(),
                "_rowaddr".to_string(),
                "name".to_string(),
                "_rowid".to_string(),
            ],
            "empty point-lookup plan must honor user column order including system columns"
        );
    }

    #[tokio::test]
    async fn test_point_lookup_active_memtable_returns_newest_duplicate() {
        // Regression: same primary key inserted twice into one active
        // memtable must return the *newest* row. The bug was that
        // `FilterExec → LIMIT 1` over an insert-ordered scan returned the
        // first (oldest) match. `WithinSourceDedupExec` collapses by PK,
        // keeping the row with the largest `_rowid` (insert order).
        use crate::dataset::mem_wal::scanner::collector::{InMemoryMemTableRef, InMemoryMemTables};
        use crate::dataset::mem_wal::write::{BatchStore, IndexStore};
        use futures::TryStreamExt;

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_uri = format!("{}/base", temp_dir.path().to_str().unwrap());

        let batch_store = Arc::new(BatchStore::with_capacity(16));
        let mut index_store = IndexStore::new();
        // BTree on the PK so that `max_visible_batch_position` advances as
        // we insert, otherwise the scanner sees no batches at all.
        index_store.add_btree("id_idx".to_string(), 0, "id".to_string());

        // Two writes to pk=1, then an unrelated pk=2. The "new" row goes
        // *second* so its `_rowid` is larger.
        let b_old = create_test_batch(&schema, &[1], "old");
        let b_new = create_test_batch(&schema, &[1], "new");
        let b_other = create_test_batch(&schema, &[2], "two");
        let (_, _, bp_old) = batch_store.append(b_old.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_old, 0, Some(bp_old))
            .unwrap();
        let (_, _, bp_new) = batch_store.append(b_new.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_new, 1, Some(bp_new))
            .unwrap();
        let (_, _, bp_other) = batch_store.append(b_other.clone()).unwrap();
        index_store
            .insert_with_batch_position(&b_other, 2, Some(bp_other))
            .unwrap();
        let index_store = Arc::new(index_store);

        let shard_id = Uuid::new_v4();
        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![])
            .with_in_memory_memtables(
                shard_id,
                InMemoryMemTables {
                    active: InMemoryMemTableRef {
                        batch_store,
                        index_store,
                        schema: schema.clone(),
                        generation: 1,
                    },
                    frozen: vec![],
                },
            );

        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let plan = planner
            .plan_lookup(&[ScalarValue::Int32(Some(1))], None)
            .await
            .unwrap();
        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "expected exactly one row for pk=1");
        let name_col = batches[0].column_by_name("name").unwrap();
        let name_arr = name_col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            name_arr.value(0),
            "new_1",
            "active-arm lookup must return the newer insert, not the oldest"
        );
    }

    #[tokio::test]
    async fn test_point_lookup_flushed_memtable_returns_newest_duplicate() {
        // Regression / invariant pin: when a flushed memtable contains two
        // rows for the same PK, the lookup must return the newer one. The
        // flushed dataset is reverse-written (newest at the smallest
        // physical position), so we simulate that here by writing the
        // dataset with the new row first. The point-lookup plan today
        // returns the first match (smallest `_rowid`) under reverse-write,
        // and remains so after this change.
        use futures::TryStreamExt;

        let schema = create_pk_schema();
        let temp_dir = tempfile::tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap();
        let base_uri = format!("{}/base", base_path);

        // Simulated reverse-write: newest insert lives at row 0.
        let shard_id = Uuid::new_v4();
        let gen1_uri = format!("{}/_mem_wal/{}/gen_1", base_uri, shard_id);
        let row_new = create_test_batch(&schema, &[1], "new");
        let row_old = create_test_batch(&schema, &[1], "old");
        create_dataset(&gen1_uri, vec![row_new, row_old]).await;

        let shard_snapshot = ShardSnapshot::new(shard_id)
            .with_current_generation(2)
            .with_flushed_generation(1, "gen_1".to_string());

        let collector = LsmDataSourceCollector::without_base_table(base_uri, vec![shard_snapshot]);
        let planner = LsmPointLookupPlanner::new(collector, vec!["id".to_string()], schema);

        let plan = planner
            .plan_lookup(&[ScalarValue::Int32(Some(1))], None)
            .await
            .unwrap();
        let ctx = datafusion::prelude::SessionContext::new();
        let stream = plan.execute(0, ctx.task_ctx()).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1, "expected exactly one row for pk=1");
        let name_col = batches[0].column_by_name("name").unwrap();
        let name_arr = name_col.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            name_arr.value(0),
            "new_1",
            "flushed-arm lookup must return the row at the smallest _rowid (newest under reverse-write)"
        );
    }
}
