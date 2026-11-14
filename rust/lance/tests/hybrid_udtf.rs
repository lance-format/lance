// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int32Type, Int64Type, UInt8Type, UInt64Type};
use arrow_array::{
    Array, FixedSizeListArray, Float32Array, LargeBinaryArray, RecordBatch, RecordBatchIterator,
    StructArray, record_batch,
};
use arrow_schema::{DataType, Field};
use datafusion::common::DataFusionError;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_expr::{Volatility, create_udf};
use datafusion_physical_plan::{ExecutionPlan, collect, displayable};
use lance::dataset::WriteParams;
use lance::dataset::udtf::HybridQueryUDTFBuilder;
use lance::index::vector::VectorIndexParams;
use lance::index::{DatasetIndexExt, DatasetIndexInternalExt};
use lance::io::exec::filtered_read::FilteredReadExec;
use lance::io::exec::fts::{FlatMatchQueryExec, MatchQueryExec, SharedFtsScorerExec};
use lance::{BlobArrayBuilder, Dataset, blob_field};
use lance_core::datatypes::BLOB_V2_DESC_TYPE;
use lance_datafusion::utils::PARTITIONS_SEARCHED_METRIC;
use lance_file::version::LanceFileVersion;
use lance_index::IndexType;
use lance_index::scalar::InvertedIndexParams;
use lance_index::scalar::inverted::tokenizer::DocumentGranularity;
use lance_linalg::distance::MetricType;
use rstest::rstest;
use serde_json::{Value, json};

#[path = "hybrid_udtf/payload.rs"]
mod payload;

fn batch() -> RecordBatch {
    let batch = record_batch!(
        ("id", Int32, [0, 1, 2, 3]),
        ("text", Utf8, ["other", "needle needle", "needle", "other"])
    )
    .unwrap();
    let vectors = Arc::new(
        FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            2,
            Arc::new(Float32Array::from(vec![
                0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 8.0, 0.0,
            ])),
            None,
        )
        .unwrap(),
    );
    let mut fields = batch.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "embedding",
        vectors.data_type().clone(),
        false,
    )));
    let mut columns = batch.columns().to_vec();
    columns.push(vectors);
    RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns).unwrap()
}

async fn dataset(stable: bool, indexed: bool) -> Dataset {
    let batch = batch();
    let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(
        reader,
        "memory://",
        Some(WriteParams {
            max_rows_per_file: 2,
            enable_stable_row_ids: stable,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(dataset.get_fragments().len(), 2);
    if indexed {
        dataset
            .create_index(
                &["text"],
                IndexType::Inverted,
                Some("hybrid_fts".to_string()),
                &InvertedIndexParams::default(),
                true,
            )
            .await
            .unwrap();
    }
    dataset
}

fn context(dataset: Arc<Dataset>, partitions: usize, batch_size: usize) -> SessionContext {
    let ctx = SessionContext::new_with_config(
        SessionConfig::new()
            .with_target_partitions(partitions)
            .with_batch_size(batch_size),
    );
    ctx.register_udtf(
        "hybrid",
        Arc::new(
            HybridQueryUDTFBuilder::builder()
                .register_table("docs", dataset)
                .build(),
        ),
    );
    ctx
}

fn relation(mut options: Value) -> String {
    for (key, value) in [
        ("vector_candidates", json!(2)),
        ("fts_candidates", json!(2)),
        ("limit", json!(3)),
        ("with_row_id", json!(true)),
    ] {
        if options.get(key).is_none() {
            options[key] = value;
        }
    }
    format!(
        "hybrid('docs', '{{\"column\":\"embedding\",\"q\":[0,0]}}', '{{\"match\":{{\"column\":\"text\",\"terms\":\"needle\"}}}}', '{}')",
        options.to_string().replace('\'', "''")
    )
}

async fn query(ctx: &SessionContext, sql: &str) -> RecordBatch {
    let frame = ctx.sql(sql).await.unwrap();
    let schema = frame.schema().as_arrow().clone();
    let batches = frame.collect().await.unwrap();
    arrow_select::concat::concat_batches(&Arc::new(schema), &batches).unwrap()
}

async fn collect_indexed(
    ctx: &SessionContext,
    plan: Arc<dyn ExecutionPlan>,
    dataset: &Dataset,
    unindexed: usize,
) -> Vec<RecordBatch> {
    let indices = dataset.load_indices_by_name("hybrid_fts").await.unwrap();
    assert_eq!(indices.len(), 1);
    assert_eq!(
        indices[0].fields,
        vec![dataset.schema().field("text").unwrap().id]
    );
    let residual = dataset.unindexed_fragments("hybrid_fts").await.unwrap();
    assert_eq!(residual.len(), unindexed);
    let mut pending = vec![plan.clone()];
    let mut indexed = Vec::new();
    let mut corpus = Vec::new();
    let mut flat = 0;
    while let Some(node) = pending.pop() {
        if node.is::<SharedFtsScorerExec>() {
            corpus.push(node.clone());
        }
        if let Some(search) = node.downcast_ref::<MatchQueryExec>() {
            assert_eq!(search.query().column.as_deref(), Some("text"));
            assert_eq!(
                search.dataset().version().version,
                dataset.version().version
            );
            if let Some(selected) = search.explicit_segment_uuids() {
                assert_eq!(selected, vec![indices[0].uuid]);
            }
            indexed.push(node.clone());
        }
        if let Some(search) = node.downcast_ref::<FlatMatchQueryExec>() {
            flat += 1;
            let children = search.children();
            let reader = children[0].downcast_ref::<FilteredReadExec>().unwrap();
            let actual = reader.options().fragments.as_ref().unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|fragment| fragment.id)
                    .collect::<Vec<_>>(),
                residual
                    .iter()
                    .map(|fragment| fragment.id)
                    .collect::<Vec<_>>()
            );
        }
        pending.extend(node.children().into_iter().cloned());
    }
    let shown = displayable(plan.as_ref()).indent(true).to_string();
    assert_eq!(indexed.len(), 1, "{shown}");
    assert_eq!(flat, usize::from(unindexed > 0), "{shown}");
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    // MatchCorpus rebuilds its template leaves with a shared scorer at execute
    // time and owns their final metrics. ANN metrics cannot satisfy this check.
    assert!(corpus.len() <= 1, "{shown}");
    let expected_counters = if corpus.is_empty() { 1 } else { 2 };
    for node in if corpus.is_empty() { indexed } else { corpus } {
        let searched = node
            .metrics()
            .unwrap()
            .iter()
            .filter(|metric| metric.value().name() == PARTITIONS_SEARCHED_METRIC)
            .map(|metric| metric.value().as_usize())
            .collect::<Vec<_>>();
        // The residual leaf also reads partitions for corpus statistics. Both
        // distinct counters must be present to prove indexed search executed.
        assert_eq!(searched.len(), expected_counters, "{searched:?}: {shown}");
        assert!(
            searched.iter().all(|count| *count > 0),
            "FTS index node did not search a segment: {shown}"
        );
    }
    batches
}

#[rstest]
#[case::flat_addresses(false, false, 1, 1024)]
#[case::flat_stable(true, false, 4, 1)]
#[case::indexed_addresses(false, true, 4, 2)]
#[case::indexed_stable(true, true, 1, 1)]
#[tokio::test]
async fn hybrid_union_rrf(
    #[case] stable: bool,
    #[case] indexed: bool,
    #[case] partitions: usize,
    #[case] batch_size: usize,
) {
    let data = Arc::new(dataset(stable, indexed).await);
    let ctx = context(data.clone(), partitions, batch_size);
    let sql = format!(
        "SELECT id, _relevance_score, _rowid FROM {} ORDER BY _relevance_score DESC, _rowid",
        relation(json!({}))
    );
    let result = if indexed {
        let plan = ctx
            .sql(&sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let schema = plan.schema();
        let batches = collect_indexed(&ctx, plan, &data, 0).await;
        arrow_select::concat::concat_batches(&schema, &batches).unwrap()
    } else {
        query(&ctx, &sql).await
    };
    assert_eq!(
        result["id"].as_primitive::<Int32Type>().values(),
        &[1, 0, 2]
    );
    let scores = result["_relevance_score"]
        .as_primitive::<Float64Type>()
        .values();
    for (actual, expected) in scores
        .iter()
        .zip([1.0 / 62.0 + 1.0 / 61.0, 1.0 / 61.0, 1.0 / 62.0])
    {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }
    assert!(
        !result
            .schema()
            .field_with_name("_relevance_score")
            .unwrap()
            .is_nullable()
    );
    assert_eq!(
        result["_rowid"].as_primitive::<UInt64Type>().value(2),
        if stable { 2 } else { 1 << 32 }
    );
    assert_eq!(result, query(&ctx, &sql).await);
}

fn final_payload_readers(plan: &Arc<dyn ExecutionPlan>) -> Vec<Arc<dyn ExecutionPlan>> {
    let mut pending = vec![plan.clone()];
    let mut found = Vec::new();
    while let Some(node) = pending.pop() {
        if let Some(reader) = node.downcast_ref::<FilteredReadExec>()
            && let Some(input) = reader.row_stream_input()
            && input.schema().index_of("_relevance_score").is_ok()
            && input.schema().index_of("_rowid").is_ok()
        {
            found.push(node.clone());
        }
        pending.extend(node.children().into_iter().cloned());
    }
    found
}

#[rstest]
#[case::stable(true)]
#[case::addresses(false)]
#[tokio::test]
async fn hybrid_outer_operations_and_late_materialization(#[case] stable: bool) {
    let ctx = context(Arc::new(dataset(stable, false).await), 4, 1);
    let table = relation(json!({"limit":2}));
    let full = query(
        &ctx,
        &format!("SELECT * FROM {table} ORDER BY _relevance_score DESC"),
    )
    .await;
    assert_eq!(full.num_columns(), 5);
    assert_eq!(full.num_rows(), 2);
    for suffix in ["WHERE id = 2", "WHERE _relevance_score > 1", "LIMIT 0"] {
        assert_eq!(
            query(&ctx, &format!("SELECT * FROM {table} {suffix}"))
                .await
                .num_rows(),
            0
        );
    }
    let filtered = relation(json!({"limit":2,"filter":"id = 2"}));
    assert_eq!(
        query(&ctx, &format!("SELECT id FROM {filtered}")).await["id"]
            .as_primitive::<Int32Type>()
            .values(),
        &[2]
    );
    let top = query(
        &ctx,
        &format!("SELECT id, _relevance_score FROM {table} ORDER BY _relevance_score DESC LIMIT 1"),
    )
    .await;
    assert_eq!(top["id"].as_primitive::<Int32Type>().values(), &[1]);
    assert_eq!(
        top["_relevance_score"]
            .as_primitive::<Float64Type>()
            .value(0),
        1.0 / 62.0 + 1.0 / 61.0
    );
    let second = query(
        &ctx,
        &format!("SELECT id FROM {table} ORDER BY _relevance_score DESC LIMIT 1 OFFSET 1"),
    )
    .await;
    assert_eq!(second["id"].as_primitive::<Int32Type>().values(), &[0]);
    let count = query(&ctx, &format!("SELECT count(*) AS n FROM {table}")).await;
    assert_eq!(count["n"].as_primitive::<Int64Type>().value(0), 2);
    let payload_plan = ctx
        .sql(&format!("SELECT id FROM {table}"))
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let readers = final_payload_readers(&payload_plan);
    assert_eq!(readers.len(), 1);
    let reader = readers[0].downcast_ref::<FilteredReadExec>().unwrap();
    let input = reader.row_stream_input().unwrap();
    assert!(input.schema().index_of("id").is_err());
    assert!(reader.schema().index_of("id").is_ok());
    let shown = displayable(input.as_ref()).indent(true).to_string();
    assert!(shown.contains("fetch=2"), "{shown}");
    let winners = collect(input.clone(), ctx.task_ctx()).await.unwrap();
    assert_eq!(winners.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    for projection in ["_relevance_score", "_rowid", "count(*)"] {
        let plan = ctx
            .sql(&format!("SELECT {projection} FROM {table}"))
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let display = displayable(plan.as_ref()).indent(true).to_string();
        assert!(display.contains("Aggregate"), "{display}");
        assert!(final_payload_readers(&plan).is_empty(), "{display}");
    }
    let reordered = query(
        &ctx,
        &format!(
            "SELECT _rowid, text, _relevance_score, id FROM {table} ORDER BY _relevance_score DESC"
        ),
    )
    .await;
    assert_eq!(
        reordered
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect::<Vec<_>>(),
        ["_rowid", "text", "_relevance_score", "id"]
    );
    assert_eq!(&reordered["id"], &full["id"]);
    let join = query(
        &ctx,
        &format!("SELECT a.id FROM {table} a JOIN (VALUES (1)) b(id) ON a.id=b.id"),
    )
    .await;
    assert_eq!(join["id"].as_primitive::<Int32Type>().values(), &[1]);
}

#[rstest]
#[case::stable(true)]
#[case::addresses(false)]
#[tokio::test]
async fn hybrid_pinned_snapshot_and_deletions(#[case] stable: bool) {
    let mut data = dataset(stable, true).await;
    let old = context(Arc::new(data.clone()), 4, 1);
    data.delete("id = 1").await.unwrap();
    let current = context(Arc::new(data), 4, 1);
    let table = relation(json!({}));
    let sql = format!("SELECT id FROM {table} ORDER BY _relevance_score DESC, _rowid");
    assert_eq!(
        query(&old, &sql).await["id"]
            .as_primitive::<Int32Type>()
            .values(),
        &[1, 0, 2]
    );
    let result = query(&current, &sql).await;
    assert_eq!(result["id"].as_primitive::<Int32Type>().values(), &[2, 0]);
    for projection in ["_rowid", "_relevance_score", "1"] {
        assert_eq!(
            query(&current, &format!("SELECT {projection} FROM {table}"))
                .await
                .num_rows(),
            2
        );
    }
}

#[rstest]
#[case::no_rows(json!({"filter":"id < 0"}), 0)]
#[case::vector_only(json!({"filter":"id = 0"}), 1)]
#[case::hidden_row_id(json!({"with_row_id":false}), 3)]
#[tokio::test]
async fn hybrid_empty_and_schema(#[case] options: Value, #[case] rows: usize) {
    let ctx = context(Arc::new(dataset(true, false).await), 4, 1);
    let table = relation(options.clone());
    let result = query(&ctx, &format!("SELECT * FROM {table}")).await;
    assert_eq!(result.num_rows(), rows);
    assert_eq!(
        result.num_columns(),
        if options["with_row_id"] == false {
            4
        } else {
            5
        }
    );
    assert_eq!(
        result
            .schema()
            .field_with_name("_relevance_score")
            .unwrap()
            .data_type(),
        &DataType::Float64
    );
}

#[rstest]
#[case::zero_candidates(json!({"vector_candidates":0}), "vector_candidates")]
#[case::negative_candidates(json!({"fts_candidates":-1}), "invalid hybrid options JSON")]
#[case::zero_limit(json!({"limit":0}), "limit")]
#[case::fractional_limit(json!({"limit":1.5}), "invalid hybrid options JSON")]
#[case::zero_rrf(json!({"rrf_k":0}), "rrf_k")]
#[case::unknown(json!({"offset":1}), "unknown field")]
#[case::wrong_boolean(json!({"with_row_id":"true"}), "boolean")]
#[case::volatile(json!({"filter":"random() > 0.5"}), "deterministic")]
#[case::stable_function(json!({"filter":"now() IS NOT NULL"}), "deterministic")]
#[case::derived_filter(json!({"filter":"_relevance_score > 0"}), "_relevance_score")]
#[case::non_boolean(json!({"filter":"id + 1"}), "Boolean")]
#[tokio::test]
async fn hybrid_invalid_options(#[case] options: Value, #[case] message: &str) {
    let ctx = context(Arc::new(dataset(false, false).await), 1, 1024);
    let error = ctx
        .sql(&format!("SELECT * FROM {}", relation(options)))
        .await
        .unwrap_err();
    assert!(
        matches!(
            error.find_root(),
            DataFusionError::Plan(_)
                | DataFusionError::Context(_, _)
                | DataFusionError::SchemaError(_, _)
        ),
        "{error:?}"
    );
    assert!(error.to_string().contains(message), "{error}");
}

#[tokio::test]
async fn hybrid_provider_projection_and_limit_hint() {
    let ctx = context(Arc::new(dataset(true, false).await), 4, 1);
    let plan = ctx
        .sql(&format!("SELECT * FROM {}", relation(json!({}))))
        .await
        .unwrap()
        .into_unoptimized_plan();
    let datafusion_expr::LogicalPlan::Projection(projection) = plan else {
        panic!("expected projection")
    };
    let datafusion_expr::LogicalPlan::TableScan(scan) = projection.input.as_ref() else {
        panic!("expected scan")
    };
    let provider = datafusion::datasource::source_as_provider(&scan.source).unwrap();
    for indices in [vec![], vec![3], vec![4, 3], vec![4, 1, 0]] {
        let physical = provider
            .scan(&ctx.state(), Some(&indices), &[], Some(1))
            .await
            .unwrap();
        assert_eq!(physical.schema().fields().len(), indices.len());
        let batches = collect(physical, ctx.task_ctx()).await.unwrap();
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
    }
}

#[rstest]
#[case::indexed(false)]
#[case::partial_coverage(true)]
#[tokio::test]
async fn hybrid_vector_index_and_residuals(#[case] append: bool) {
    let mut data = dataset(true, true).await;
    data.create_index(
        &["embedding"],
        IndexType::IvfFlat,
        None,
        &VectorIndexParams::ivf_flat(1, MetricType::L2),
        true,
    )
    .await
    .unwrap();
    if append {
        let rows = batch().slice(2, 1);
        data.append(
            RecordBatchIterator::new(vec![Ok(rows.clone())], rows.schema()),
            None,
        )
        .await
        .unwrap();
    }
    let data = Arc::new(data);
    let mut candidate_sets = Vec::with_capacity(2);
    for use_index in [false, true] {
        let mut scan = data.scan();
        scan.nearest("embedding", &Float32Array::from(vec![0.0, 0.0]), 4)
            .unwrap()
            .use_index(use_index)
            .project(&["_rowid"])
            .unwrap();
        let batch = scan.try_into_batch().await.unwrap();
        candidate_sets.push(
            batch["_rowid"]
                .as_primitive::<UInt64Type>()
                .values()
                .to_vec(),
        );
    }
    let recall = candidate_sets[1]
        .iter()
        .filter(|id| candidate_sets[0].contains(id))
        .count() as f64
        / candidate_sets[0].len() as f64;
    assert!(recall >= 0.5, "vector recall={recall}");
    if append {
        assert!(
            candidate_sets[1].contains(&4),
            "unindexed vector row was omitted"
        );
    }
    let ctx = context(data.clone(), 4, 1);
    let table = relation(json!({"vector_candidates":4,"fts_candidates":4,"limit":5}));
    let plan = ctx.sql(&format!("SELECT id, _relevance_score, _rowid FROM {table} ORDER BY _relevance_score DESC, _rowid"))
        .await.unwrap().create_physical_plan().await.unwrap();
    let display = displayable(plan.as_ref()).indent(true).to_string();
    assert!(display.contains("ANN"), "{display}");
    let batches = collect_indexed(&ctx, plan, &data, usize::from(append)).await;
    let ids = batches
        .iter()
        .flat_map(|b| b["id"].as_primitive::<Int32Type>().values().to_vec())
        .collect::<Vec<_>>();
    assert!(
        ids.contains(&0),
        "vector-only nearest candidate is absent: {ids:?}"
    );
    assert!(ids.contains(&1) && ids.contains(&2));
    if append {
        assert_eq!(ids.iter().filter(|&&id| id == 2).count(), 2);
        let residual_score = batches
            .iter()
            .find_map(|batch| {
                let rows = batch["_rowid"].as_primitive::<UInt64Type>();
                let scores = batch["_relevance_score"].as_primitive::<Float64Type>();
                rows.values()
                    .iter()
                    .position(|&row| row == 4)
                    .map(|index| scores.value(index))
            })
            .unwrap();
        assert!(
            residual_score > 1.0 / 61.0,
            "the unindexed row must contribute from both branches: {residual_score}"
        );
    } else {
        assert!(
            ids.contains(&3),
            "vector-only farthest candidate is absent: {ids:?}"
        );
    }
}

#[tokio::test]
async fn hybrid_stable_identity_after_compaction() {
    let mut data = dataset(true, true).await;
    data.delete("id = 0").await.unwrap();
    let previous = Arc::new(data.clone());
    let old = context(previous.clone(), 1, 1);
    lance::dataset::optimize::compact_files(
        &mut data,
        lance::dataset::optimize::CompactionOptions::default(),
        None,
    )
    .await
    .unwrap();
    let data = Arc::new(data);
    let current = context(data.clone(), 4, 2);
    let sql = format!(
        "SELECT _rowid, id, text, _relevance_score FROM {} ORDER BY _relevance_score DESC, _rowid",
        relation(json!({}))
    );
    let mut results = Vec::new();
    for (ctx, snapshot) in [(&old, previous), (&current, data)] {
        let plan = ctx
            .sql(&sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let schema = plan.schema();
        let batches = collect_indexed(ctx, plan, &snapshot, 0).await;
        results.push(arrow_select::concat::concat_batches(&schema, &batches).unwrap());
    }
    assert_eq!(results[0], results[1]);
}

#[rstest]
#[case::dimension(json!({"column":"embedding","q":[1]}), "expected 2")]
#[case::batch(json!({"column":"embedding","q":[[0,0]]}), "vector JSON")]
#[case::null(json!({"column":"embedding","q":[null,0]}), "vector JSON")]
#[case::unknown(json!({"column":"embedding","q":[0,0],"limit":1}), "unknown field")]
#[case::wrong_column(json!({"column":"id","q":[0,0]}), "FixedSizeList")]
#[case::overflow(json!({"column":"embedding","q":[1e100,0]}), "representable")]
#[case::positive_underflow(json!({"column":"embedding","q":[1e-100,0]}), "representable")]
#[case::negative_underflow(json!({"column":"embedding","q":[-1e-100,0]}), "representable")]
#[tokio::test]
async fn hybrid_invalid_vector(#[case] vector: Value, #[case] message: &str) {
    let ctx = context(Arc::new(dataset(false, false).await), 1, 1024);
    let sql = format!(
        "SELECT * FROM hybrid('docs', '{}', '{{\"match\":{{\"column\":\"text\",\"terms\":\"needle\"}}}}', '{{\"vector_candidates\":2,\"fts_candidates\":2,\"limit\":3}}')",
        vector
    );
    let error = ctx.sql(&sql).await.unwrap_err();
    assert!(
        matches!(
            error,
            DataFusionError::Plan(_) | DataFusionError::Context(_, _)
        ),
        "{error:?}"
    );
    assert!(error.to_string().contains(message), "{error}");
}

#[rstest]
#[tokio::test]
async fn hybrid_blob_descriptor_schema_and_filter(
    #[values(LanceFileVersion::V2_2, LanceFileVersion::V2_3)] version: LanceFileVersion,
    #[values(false, true)] stable: bool,
) {
    let base = batch();
    let mut blobs = BlobArrayBuilder::new(4);
    blobs.push_bytes(b"a").unwrap();
    blobs.push_null().unwrap();
    blobs.push_bytes(b"ccc").unwrap();
    blobs.push_bytes(b"dddd").unwrap();
    let blobs = blobs.finish().unwrap();
    let nested = Arc::new(StructArray::new(
        vec![blob_field("blob", true)].into(),
        vec![blobs.clone()],
        None,
    ));
    let plain = Arc::new(LargeBinaryArray::from(vec![
        Some(b"zero".as_slice()),
        Some(b"one".as_slice()),
        Some(b"two".as_slice()),
        Some(b"three".as_slice()),
    ]));
    let mut fields = base.schema().fields().to_vec();
    fields.extend([
        Arc::new(blob_field("direct", true)),
        Arc::new(Field::new("nested", nested.data_type().clone(), false)),
        Arc::new(Field::new("plain", DataType::LargeBinary, false)),
    ]);
    let mut columns = base.columns().to_vec();
    columns.extend([blobs, nested as _, plain as _]);
    let rows = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns).unwrap();
    let data = Arc::new(
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(rows.clone())], rows.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 2,
                enable_stable_row_ids: stable,
                data_storage_version: Some(version),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    assert_eq!(data.get_fragments().len(), 2);
    let expected = data
        .scan()
        .filter("id < 3")
        .unwrap()
        .project(&["id", "direct", "nested", "plain"])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(expected["direct"].data_type(), &*BLOB_V2_DESC_TYPE);
    assert_eq!(
        expected["nested"].as_struct()["blob"].data_type(),
        &*BLOB_V2_DESC_TYPE
    );
    let ctx = context(data, 4, 1);
    let table = relation(json!({}));
    let frame = ctx
        .sql(&format!(
            "SELECT id, direct, nested, plain FROM {table} ORDER BY id"
        ))
        .await
        .unwrap();
    let logical_schema = Arc::new(frame.schema().as_arrow().clone());
    assert_eq!(logical_schema, expected.schema());
    let plan = frame.create_physical_plan().await.unwrap();
    assert_eq!(plan.schema(), logical_schema);
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    let actual = arrow_select::concat::concat_batches(&logical_schema, &batches).unwrap();
    assert_eq!(actual, expected);
    let direct = actual["direct"].as_struct();
    assert!(direct.is_null(1));
    assert_eq!(direct["size"].as_primitive::<UInt64Type>().value(0), 1);
    assert_eq!(direct["size"].as_primitive::<UInt64Type>().value(2), 3);
    let children = query(&ctx, &format!(
        "SELECT id, direct['size'] AS size, nested['blob']['kind'] AS kind FROM {table} ORDER BY id"
    )).await;
    assert_eq!(children["size"].as_primitive::<UInt64Type>().value(2), 3);
    assert!(children["size"].is_null(1));
    let kind = direct["kind"].as_primitive::<UInt8Type>().value(0);
    let filtered = relation(json!({
        "vector_candidates":4, "fts_candidates":4, "limit":4,
        "filter":format!("direct['kind'] IS NOT NULL AND nested['blob']['kind'] = {kind}")
    }));
    let result = query(
        &ctx,
        &format!("SELECT id, _relevance_score FROM {filtered} ORDER BY id"),
    )
    .await;
    assert_eq!(
        result["id"].as_primitive::<Int32Type>().values(),
        &[0, 2, 3]
    );
    assert!(
        (result["_relevance_score"]
            .as_primitive::<Float64Type>()
            .value(1)
            - (1.0 / 62.0 + 1.0 / 61.0))
            .abs()
            < 1e-12
    );
    assert_eq!(
        query(&ctx, &format!("SELECT count(*) AS n FROM {table}")).await["n"]
            .as_primitive::<Int64Type>()
            .value(0),
        3
    );
}

#[rstest]
#[case::relevance("_relevance_score", 0)]
#[case::score_number("_score", 0)]
#[case::distance_text("_distance", 1)]
#[case::score_case("_ScOrE", 0)]
#[case::distance_case("_DISTANCE", 1)]
#[tokio::test]
async fn hybrid_schema_collision(#[case] name: &str, #[case] index: usize) {
    let batch = batch();
    let mut fields = batch.schema().fields().to_vec();
    fields[index] = Arc::new(fields[index].as_ref().clone().with_name(name));
    let batch = RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(fields)),
        batch.columns().to_vec(),
    )
    .unwrap();
    let data = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let ctx = context(Arc::new(data), 1, 1024);
    for options in [
        json!({}),
        json!({"with_row_id":false,"filter":"embedding IS NOT NULL"}),
    ] {
        let error = ctx
            .sql(&format!("SELECT * FROM {}", relation(options)))
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                DataFusionError::Plan(_) | DataFusionError::Context(_, _)
            ),
            "{error:?}"
        );
        assert!(
            error.to_string().contains(&format!("field {name}")),
            "{error}"
        );
    }
}

#[tokio::test]
async fn hybrid_respects_caller_memory_pool() {
    let pool = Arc::new(GreedyMemoryPool::new(1));
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(pool.clone())
        .build()
        .unwrap();
    let ctx = SessionContext::new_with_config_rt(SessionConfig::new(), Arc::new(runtime));
    ctx.register_udtf(
        "hybrid",
        Arc::new(
            HybridQueryUDTFBuilder::builder()
                .register_table("docs", Arc::new(dataset(true, false).await))
                .build(),
        ),
    );
    let error = ctx
        .sql(&format!(
            "SELECT _relevance_score FROM {}",
            relation(json!({}))
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("memory") || error.to_string().contains("Memory"),
        "{error}"
    );
    assert_eq!(pool.reserved(), 0);
}

#[rstest]
#[case::no_args("hybrid()", "requires")]
#[case::wrong_type("hybrid(1, '{}', '{}', '{}')", "argument 1")]
#[case::null("hybrid(NULL, '{}', '{}', '{}')", "argument 1")]
#[case::unregistered("hybrid('missing', '{}', '{}', '{}')", "not registered")]
#[case::invalid_json("hybrid('docs', '{', '{}', '{}')", "vector JSON")]
#[tokio::test]
async fn hybrid_argument_errors(#[case] call: &str, #[case] message: &str) {
    let ctx = context(Arc::new(dataset(false, false).await), 1, 1024);
    let error = ctx.sql(&format!("SELECT * FROM {call}")).await.unwrap_err();
    assert!(
        matches!(
            error,
            DataFusionError::Plan(_) | DataFusionError::Context(_, _)
        ),
        "{error:?}"
    );
    assert!(error.to_string().contains(message), "{error}");
}

#[tokio::test]
async fn hybrid_null_vector_lexical_candidate() {
    let batch = batch();
    let vector = batch["embedding"].as_fixed_size_list();
    let vector = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        2,
        vector.values().clone(),
        Some(arrow_buffer::NullBuffer::from(vec![
            true, true, false, true,
        ])),
    )
    .unwrap();
    let mut fields = batch.schema().fields().to_vec();
    fields[2] = Arc::new(fields[2].as_ref().clone().with_nullable(true));
    let mut columns = batch.columns().to_vec();
    columns[2] = Arc::new(vector);
    let batch = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns).unwrap();
    let data = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let ctx = context(Arc::new(data), 4, 1);
    let table = relation(json!({"filter":"id = 2"}));
    let result = query(&ctx, &format!("SELECT id, _relevance_score FROM {table}")).await;
    assert_eq!(result["id"].as_primitive::<Int32Type>().values(), &[2]);
    assert_eq!(
        result["_relevance_score"]
            .as_primitive::<Float64Type>()
            .value(0),
        1.0 / 61.0
    );
}

#[tokio::test]
async fn hybrid_nested_metric_name_is_payload() {
    let batch = batch();
    let nested = arrow_array::StructArray::from(vec![(
        Arc::new(Field::new("_score", DataType::Int32, false)),
        batch["id"].clone(),
    )]);
    let mut fields = batch.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "nested",
        nested.data_type().clone(),
        false,
    )));
    let mut columns = batch.columns().to_vec();
    columns.push(Arc::new(nested));
    let batch = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns).unwrap();
    let data = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let ctx = context(Arc::new(data), 4, 1);
    let table = relation(json!({}));
    let result = query(
        &ctx,
        &format!(
            "SELECT nested['_score'] AS stored FROM {table} ORDER BY _relevance_score DESC, _rowid"
        ),
    )
    .await;
    assert_eq!(
        result["stored"].as_primitive::<Int32Type>().values(),
        &[1, 0, 2]
    );
}

#[tokio::test]
async fn hybrid_custom_rrf_constant() {
    let ctx = context(Arc::new(dataset(true, false).await), 1, 1);
    let table = relation(json!({"rrf_k":1}));
    let result = query(
        &ctx,
        &format!("SELECT id, _relevance_score FROM {table} ORDER BY _relevance_score DESC, _rowid"),
    )
    .await;
    assert_eq!(
        result["id"].as_primitive::<Int32Type>().values(),
        &[1, 0, 2]
    );
    assert_eq!(
        result["_relevance_score"]
            .as_primitive::<Float64Type>()
            .values(),
        &[1.0 / 3.0 + 0.5, 0.5, 1.0 / 3.0]
    );
}

#[tokio::test]
async fn hybrid_final_tie_uses_row_id() {
    let ctx = context(Arc::new(dataset(true, true).await), 4, 1);
    let table = relation(json!({"vector_candidates":1,"fts_candidates":1,"limit":1}));
    let result = query(&ctx, &format!("SELECT id, _relevance_score FROM {table}")).await;
    assert_eq!(result["id"].as_primitive::<Int32Type>().values(), &[0]);
    assert_eq!(
        result["_relevance_score"]
            .as_primitive::<Float64Type>()
            .values(),
        &[1.0 / 61.0]
    );
}

#[tokio::test]
async fn hybrid_filter_uses_caller_function_registry() {
    let ctx = context(Arc::new(dataset(true, true).await), 4, 1);
    ctx.register_udf(create_udf(
        "hybrid_id",
        vec![DataType::Int32],
        DataType::Int32,
        Volatility::Immutable,
        Arc::new(|args| Ok(args[0].clone())),
    ));
    let table = relation(json!({"filter":"hybrid_id(id) = 2"}));
    let result = query(&ctx, &format!("SELECT id, _relevance_score FROM {table}")).await;
    assert_eq!(result["id"].as_primitive::<Int32Type>().values(), &[2]);
    assert_eq!(
        result["_relevance_score"]
            .as_primitive::<Float64Type>()
            .values(),
        &[2.0 / 61.0]
    );
}

#[tokio::test]
async fn hybrid_fts_column_inference_and_existing_json_parser() {
    let ctx = context(Arc::new(dataset(true, true).await), 1, 1);
    let table = relation(json!({})).replace("\"column\":\"text\",", "");
    let table = table.replace(
        "\"terms\":\"needle\"",
        "\"terms\":\"needle\",\"unused\":true",
    );
    let result = query(
        &ctx,
        &format!("SELECT id FROM {table} ORDER BY _relevance_score DESC, _rowid"),
    )
    .await;
    assert_eq!(
        result["id"].as_primitive::<Int32Type>().values(),
        &[1, 0, 2]
    );
}

#[rstest]
#[case::plain("id")]
#[case::backtick("a`b")]
#[tokio::test]
async fn hybrid_field_path_payload(#[case] name: &str) {
    let batch = batch();
    let mut fields = batch.schema().fields().to_vec();
    fields[0] = Arc::new(fields[0].as_ref().clone().with_name(name));
    let batch = RecordBatch::try_new(
        Arc::new(arrow_schema::Schema::new(fields)),
        batch.columns().to_vec(),
    )
    .unwrap();
    let data = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let ctx = context(Arc::new(data), 1, 1);
    for projection in ["*".to_string(), format!("\"{name}\"")] {
        let result = query(
            &ctx,
            &format!(
                "SELECT {projection} FROM {} ORDER BY _relevance_score DESC, _rowid",
                relation(json!({}))
            ),
        )
        .await;
        assert_eq!(
            result[name].null_count(),
            0,
            "hybrid projection={projection}: {result:?}"
        );
        assert_eq!(
            result[name].as_primitive::<Int32Type>().values(),
            &[1, 0, 2],
            "hybrid projection={projection}"
        );
    }
    let table = relation(json!({"filter":format!("\"{name}\" = 2")}));
    let filtered = query(&ctx, &format!("SELECT * FROM {table}")).await;
    assert_eq!(filtered[name].as_primitive::<Int32Type>().values(), &[2]);
    assert_eq!(
        filtered["_relevance_score"]
            .as_primitive::<Float64Type>()
            .values(),
        &[2.0 / 61.0]
    );
}

#[tokio::test]
async fn hybrid_field_path_vector() {
    let batch = batch();
    let nested = arrow_array::StructArray::from(vec![(
        Arc::new(batch.schema().field(2).clone().with_name("vec.with.dot")),
        batch["embedding"].clone(),
    )]);
    let mut fields = batch.schema().fields().to_vec();
    fields[2] = Arc::new(Field::new("payload", nested.data_type().clone(), false));
    let mut columns = batch.columns().to_vec();
    columns[2] = Arc::new(nested);
    let batch = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns).unwrap();
    let data = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let ctx = context(Arc::new(data), 1, 1);
    let table = relation(json!({})).replace(
        "\"column\":\"embedding\"",
        "\"column\":\"payload.`vec.with.dot`\"",
    );
    let result = query(
        &ctx,
        &format!("SELECT id FROM {table} ORDER BY _relevance_score DESC, _rowid"),
    )
    .await;
    assert_eq!(
        result["id"].as_primitive::<Int32Type>().values(),
        &[1, 0, 2]
    );
}

#[rstest]
#[case::f16(DataType::Float16)]
#[case::f32(DataType::Float32)]
#[case::f64(DataType::Float64)]
#[case::u8(DataType::UInt8)]
#[tokio::test]
async fn hybrid_vector_representations(#[case] dtype: DataType) {
    let batch = batch();
    let is_byte = dtype == DataType::UInt8;
    let dtype = DataType::FixedSizeList(Arc::new(Field::new("item", dtype, true)), 2);
    let mut fields = batch.schema().fields().to_vec();
    fields[2] = Arc::new(Field::new("embedding", dtype.clone(), false));
    let mut columns = batch.columns().to_vec();
    columns[2] = arrow_cast::cast(&batch["embedding"], &dtype).unwrap();
    if is_byte {
        // Hamming distance must distinguish the second and fourth rows too.
        columns[2] = Arc::new(FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::UInt8, true)),
            2,
            Arc::new(arrow_array::UInt8Array::from(vec![
                0, 0, 1, 0, 3, 0, 255, 0,
            ])),
            None,
        ));
    }
    let batch = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns).unwrap();
    let data = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let ctx = context(Arc::new(data), 1, 1);
    let result = query(
        &ctx,
        &format!(
            "SELECT id FROM {} ORDER BY _relevance_score DESC, _rowid",
            relation(json!({}))
        ),
    )
    .await;
    assert_eq!(
        result["id"].as_primitive::<Int32Type>().values(),
        &[1, 0, 2]
    );
}

#[tokio::test]
async fn hybrid_rejects_multivector_column() {
    let batch = batch();
    let vectors = arrow_array::ListArray::new(
        Arc::new(Field::new(
            "item",
            batch["embedding"].data_type().clone(),
            true,
        )),
        arrow_buffer::OffsetBuffer::new(vec![0, 1, 2, 3, 4].into()),
        batch["embedding"].clone(),
        None,
    );
    let mut fields = batch.schema().fields().to_vec();
    fields[2] = Arc::new(Field::new("embedding", vectors.data_type().clone(), false));
    let mut columns = batch.columns().to_vec();
    columns[2] = Arc::new(vectors);
    let batch = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns).unwrap();
    let data = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    let ctx = context(Arc::new(data), 1, 1);
    let error = ctx
        .sql(&format!("SELECT * FROM {}", relation(json!({}))))
        .await
        .unwrap_err();
    assert!(
        matches!(error.find_root(), DataFusionError::Plan(_)),
        "{error}"
    );
    assert!(
        error.to_string().contains("single FixedSizeList vector"),
        "{error}"
    );
}

#[rstest]
#[case::explicit(false)]
#[case::inferred_from_index(true)]
#[tokio::test]
async fn hybrid_rejects_list_element_fts(#[case] indexed: bool) {
    let batch = batch();
    let text = arrow_array::ListArray::new(
        Arc::new(Field::new("item", DataType::Utf8, true)),
        arrow_buffer::OffsetBuffer::new(vec![0, 1, 2, 3, 4].into()),
        batch["text"].clone(),
        None,
    );
    let mut fields = batch.schema().fields().to_vec();
    fields[1] = Arc::new(Field::new("text", text.data_type().clone(), false));
    let mut columns = batch.columns().to_vec();
    columns[1] = Arc::new(text);
    let batch = RecordBatch::try_new(Arc::new(arrow_schema::Schema::new(fields)), columns).unwrap();
    let mut data = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
        "memory://",
        None,
    )
    .await
    .unwrap();
    if indexed {
        data.create_index(
            &["text"],
            IndexType::Inverted,
            None,
            &InvertedIndexParams::default().document_granularity(DocumentGranularity::ListElement),
            true,
        )
        .await
        .unwrap();
    }
    let ctx = context(Arc::new(data), 1, 1);
    let mut table = relation(json!({}));
    if !indexed {
        table = table.replace(
            "\"terms\":\"needle\"",
            "\"terms\":\"needle\",\"document_granularity\":\"list_element\"",
        );
    }
    let error = ctx
        .sql(&format!("SELECT * FROM {table}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    assert!(
        matches!(error.find_root(), DataFusionError::Plan(_)),
        "{error}"
    );
    assert!(
        error
            .to_string()
            .contains("ListElement queries are unsupported"),
        "{error}"
    );
}
