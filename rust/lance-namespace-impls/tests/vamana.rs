// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF_VAMANA_* indices through the directory namespace.
//!
//! This file exists only in the Vamana fork of Lance. It tests the hook in
//! `src/dir.rs` that maps the IVF_VAMANA_* type names to their build
//! parameters, through the public namespace API, and reads the table past the
//! namespace only for what that API does not report.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{Array, AsArray, FixedSizeListArray, Float32Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Float32Type, Int32Type, Schema};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::StreamWriter;
use bytes::Bytes;
use lance::Dataset;
use lance::index::DatasetIndexExt;
use lance_core::Error;
use lance_core::utils::tempfile::TempStrDir;
use lance_linalg::distance::DistanceType;
use lance_namespace::models::{
    CreateTableIndexRequest, CreateTableIndexResponse, CreateTableRequest,
    DescribeTableIndexStatsRequest, DescribeTableRequest, IndexContent, ListTableIndicesRequest,
    QueryTableRequest, QueryTableRequestVector,
};
use lance_namespace::{ErrorCode, LanceNamespace, NamespaceError};
use lance_namespace_impls::{DirectoryNamespace, DirectoryNamespaceBuilder};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rstest::rstest;
use serde_json::{Value, json};

const TABLE: &str = "vectors";
const INDEX: &str = "vector_idx";
/// Fewer rows than the 256 that eight-bit product codes train on: an
/// IVF_VAMANA_PQ index is recorded untrained, so no graph is drawn over
/// product codes, which takes seconds in a debug build at the default degree.
const ROWS: i64 = 100;
/// A width the default product quantizer's 16 sub-vectors divide, so that the
/// untrained IVF_VAMANA_PQ index is one the column can train later.
const DIM: i32 = 16;
const K: i32 = 5;

fn table_id() -> Option<Vec<String>> {
    Some(vec![TABLE.to_string()])
}

fn random_values(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..len).map(|_| rng.random::<f32>()).collect()
}

/// A namespace holding [`TABLE`]: an `id` and a [`DIM`]-wide `vector` in
/// each of [`ROWS`] rows.
async fn namespace_with_table(dir: &TempStrDir) -> DirectoryNamespace {
    let namespace = DirectoryNamespaceBuilder::new(dir.as_str())
        .build()
        .await
        .unwrap();
    let vectors = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM,
        Arc::new(Float32Array::from(random_values(
            42,
            (ROWS * DIM as i64) as usize,
        ))),
        None,
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("vector", vectors.data_type().clone(), true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(0..ROWS as i32)),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    let mut ipc = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut ipc, &schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    let mut request = CreateTableRequest::new();
    request.id = table_id();
    namespace
        .create_table(request, Bytes::from(ipc))
        .await
        .unwrap();
    namespace
}

/// The table as a Lance dataset, opened past the namespace.
async fn dataset(namespace: &DirectoryNamespace) -> Dataset {
    let mut request = DescribeTableRequest::new();
    request.id = table_id();
    let location = namespace
        .describe_table(request)
        .await
        .unwrap()
        .location
        .unwrap();
    Dataset::open(&location).await.unwrap()
}

async fn create_index(
    namespace: &DirectoryNamespace,
    index_type: &str,
    distance_type: Option<&str>,
) -> lance_core::Result<CreateTableIndexResponse> {
    let mut request = CreateTableIndexRequest::new("vector".into(), index_type.into());
    request.id = table_id();
    request.name = Some(INDEX.into());
    request.distance_type = distance_type.map(Into::into);
    namespace.create_table_index(request).await
}

async fn indices(namespace: &DirectoryNamespace) -> Vec<IndexContent> {
    namespace
        .list_table_indices(ListTableIndicesRequest {
            id: table_id(),
            ..Default::default()
        })
        .await
        .unwrap()
        .indexes
}

fn error_code(error: &Error) -> ErrorCode {
    match error {
        Error::Namespace { source, .. } => source
            .downcast_ref::<NamespaceError>()
            .expect("a namespace error")
            .code(),
        other => panic!("expected a namespace error, got {other:?}"),
    }
}

/// Each type is built with Vamana's default graph and the distance asked
/// for, and listed and described as the namespace lists IVF_HNSW_*.
#[rstest]
#[case::flat("IVF_VAMANA_FLAT", None, true)]
#[case::sq("IVF_VAMANA_SQ", Some(json!({"type": "sq", "num_bits": 8})), true)]
#[case::pq(
    "IVF_VAMANA_PQ",
    Some(json!({"type": "pq", "num_bits": 8, "num_sub_vectors": 16})),
    false
)]
#[tokio::test]
async fn each_type_is_created_with_vamanas_defaults(
    #[case] index_type: &str,
    #[case] compression: Option<Value>,
    #[case] trained: bool,
    #[values((None, "L2"), (Some("cosine"), "COSINE"))] distance: (Option<&str>, &str),
) {
    let (distance_type, metric_type) = distance;
    let dir = TempStrDir::default();
    let namespace = namespace_with_table(&dir).await;
    let response = create_index(&namespace, index_type, distance_type)
        .await
        .unwrap();
    assert!(response.transaction_id.is_some());

    let indices = indices(&namespace).await;
    assert_eq!(indices.len(), 1);
    let index = &indices[0];
    assert_eq!(index.index_name, INDEX);
    assert_eq!(index.columns, vec!["vector"]);
    assert_eq!(index.index_type.as_deref(), Some(index_type));
    let mut details: Value = serde_json::from_str(index.index_details.as_deref().unwrap()).unwrap();
    // The hints hold the defaults of the IVF and quantizer stages, which the
    // namespace takes as HNSW's arms do.
    details.as_object_mut().unwrap().remove("runtime_hints");
    let mut expected = json!({
        "metric_type": metric_type,
        "vamana": {"max_degree": 64, "search_list_size": 100, "alpha": 1.2},
    });
    if let Some(compression) = compression {
        expected["compression"] = compression;
    }
    assert_eq!(details, expected);

    let indexed = if trained { ROWS } else { 0 };
    assert_eq!(index.num_indexed_rows, Some(indexed));
    assert_eq!(index.num_unindexed_rows, Some(ROWS - indexed));
    let statistics = namespace
        .describe_table_index_stats(DescribeTableIndexStatsRequest {
            id: table_id(),
            index_name: Some(INDEX.into()),
            ..Default::default()
        })
        .await
        .unwrap();
    // An untrained index has no file to read statistics from, and Lance names
    // the type of every such index N/A.
    let statistics_type = if trained { index_type } else { "N/A" };
    assert_eq!(statistics.index_type.as_deref(), Some(statistics_type));
    assert_eq!(statistics.num_indexed_rows, Some(indexed));
    assert_eq!(statistics.num_unindexed_rows, Some(ROWS - indexed));

    // The partition count is in no namespace response. The IVF stage keeps
    // its default, a partition per 1 << 20 rows; an untrained index records
    // no count.
    let statistics: Value = serde_json::from_str(
        &dataset(&namespace)
            .await
            .index_statistics(INDEX)
            .await
            .unwrap(),
    )
    .unwrap();
    let partitions = if trained { json!(1) } else { Value::Null };
    assert_eq!(statistics["indices"][0]["num_partitions"], partitions);
}

fn query_vector() -> Vec<f32> {
    random_values(7, DIM as usize)
}

fn query(distance_type: DistanceType, ef: i32) -> QueryTableRequest {
    let mut vector = QueryTableRequestVector::new();
    vector.single_vector = Some(query_vector());
    let mut request = QueryTableRequest::new(K, vector);
    request.id = table_id();
    request.distance_type = Some(distance_type.to_string());
    request.ef = Some(ef);
    request
}

fn ids_and_distances(batch: &RecordBatch) -> (Vec<i32>, Vec<f32>) {
    (
        batch["id"].as_primitive::<Int32Type>().values().to_vec(),
        batch["_distance"]
            .as_primitive::<Float32Type>()
            .values()
            .to_vec(),
    )
}

/// A query through the namespace finds the nearest rows through the graph,
/// at the distances a scan of every row measures.
#[rstest]
#[tokio::test]
async fn a_namespace_query_is_answered_by_the_graph(
    #[values("IVF_VAMANA_FLAT", "IVF_VAMANA_SQ")] index_type: &str,
    #[values(DistanceType::L2, DistanceType::Cosine)] distance_type: DistanceType,
) {
    let dir = TempStrDir::default();
    let namespace = namespace_with_table(&dir).await;
    create_index(&namespace, index_type, Some(&distance_type.to_string()))
        .await
        .unwrap();

    // A scan of every row ignores ef, and only Vamana's search refuses a list
    // shorter than the results: this path reaches the graph.
    let error = namespace
        .query_table(query(distance_type, K - 1))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("ef must be greater than or equal to k"),
        "{error}"
    );

    let ipc = namespace
        .query_table(query(distance_type, 60))
        .await
        .unwrap();
    let batches = FileReader::try_new(Cursor::new(ipc), None)
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(batches.len(), 1);
    let (ids, distances) = ids_and_distances(&batches[0]);

    let dataset = dataset(&namespace).await;
    let mut scan = dataset.scan();
    scan.nearest("vector", &Float32Array::from(query_vector()), K as usize)
        .unwrap()
        .use_index(false)
        .distance_metric(distance_type);
    let (nearest_ids, nearest_distances) = ids_and_distances(&scan.try_into_batch().await.unwrap());

    assert_eq!(ids.len(), K as usize);
    assert_eq!(ids, nearest_ids);
    for (distance, nearest) in distances.iter().zip(&nearest_distances) {
        assert!(
            (distance - nearest).abs() <= 1e-5,
            "{distances:?} vs {nearest_distances:?}"
        );
    }
}

/// The scalar endpoint refuses the IVF_VAMANA_* types, as it refuses
/// IVF_HNSW_*, instead of building a vector index under its name.
#[rstest]
#[tokio::test]
async fn vector_types_are_not_scalar_indices(
    #[values("IVF_VAMANA_FLAT", "IVF_VAMANA_SQ", "IVF_VAMANA_PQ")] index_type: &str,
) {
    let dir = TempStrDir::default();
    let namespace = namespace_with_table(&dir).await;
    let mut request = CreateTableIndexRequest::new("vector".into(), index_type.into());
    request.id = table_id();
    let error = namespace
        .create_table_scalar_index(request)
        .await
        .unwrap_err();
    assert_eq!(error_code(&error), ErrorCode::InvalidInput);
    assert!(
        error.to_string().contains(&format!(
            "only supports scalar index types, got {index_type}"
        )),
        "{error}"
    );
    assert!(indices(&namespace).await.is_empty());
}

/// Vamana refuses dot before a row is read, and the namespace passes the
/// refusal on as it passes on every failed build.
#[tokio::test]
async fn the_dot_distance_is_refused() {
    let dir = TempStrDir::default();
    let namespace = namespace_with_table(&dir).await;
    let error = create_index(&namespace, "IVF_VAMANA_FLAT", Some("dot"))
        .await
        .unwrap_err();
    assert_eq!(error_code(&error), ErrorCode::Internal);
    assert!(
        error
            .to_string()
            .contains("IVF_VAMANA supports the l2 and cosine distances, not dot"),
        "{error}"
    );
    assert!(indices(&namespace).await.is_empty());
}
