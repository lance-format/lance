// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

use std::io::Cursor;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::record_batch::RecordBatchIterator;
use arrow_ipc::reader::{FileReader, StreamReader};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::DatasetIndexExt;
use lance::index::vector::{StageParams, VectorIndexParams};
use lance_core::utils::tempfile::{TempStdDir, TempStrDir};
use lance_core::utils::testing::CountingObjectStore;
use lance_index::IndexType;
use lance_index::scalar::inverted::query::{FtsQuery, Operator};
use lance_io::object_store::providers::local::FileStoreProvider;
use lance_io::object_store::{ObjectStoreParams, ObjectStoreRegistry, uri_to_url};
use lance_linalg::distance::MetricType;
use lance_namespace::error::{ErrorCode, NamespaceError};
use lance_namespace::models::{
    BatchDeleteTableVersionsRequest, BatchDeleteTableVersionsResponse, CountTableRowsRequest,
    CreateNamespaceRequest, CreateNamespaceResponse, CreateTableBranchRequest,
    CreateTableIndexRequest, CreateTableRequest, CreateTableResponse, CreateTableTagRequest,
    CreateTableVersionRequest, CreateTableVersionResponse, DeclareTableRequest,
    DeclareTableResponse, DeleteFromTableRequest, DeleteTableBranchRequest, DeleteTableTagRequest,
    DescribeNamespaceRequest, DescribeNamespaceResponse, DescribeTableRequest,
    DescribeTableResponse, DescribeTableVersionRequest, DescribeTableVersionResponse,
    DropNamespaceRequest, DropNamespaceResponse, DropTableRequest, DropTableResponse,
    GetTableTagVersionRequest, InsertIntoTableRequest, JsonArrowDataType, JsonArrowField,
    JsonArrowSchema, ListNamespacesRequest, ListNamespacesResponse, ListTableBranchesRequest,
    ListTableTagsRequest, ListTableVersionsRequest, ListTableVersionsResponse, ListTablesRequest,
    ListTablesResponse, NamespaceExistsRequest, QueryTableRequest, QueryTableRequestColumns,
    TableExistsRequest, TableVersion, UpdateTableRequest, UpdateTableTagRequest,
};
use lance_namespace::schema::convert_json_arrow_schema;
use lance_table::io::commit::{ManifestNamingScheme, VERSIONS_DIR};
use object_store::{
    Error as ObjectStoreError, ObjectStore as OSObjectStore, ObjectStoreExt, PutOptions,
};
use rstest::rstest;
use url::Url;

use super::index_params::DirectoryIndexParams;
use super::query::build_engine_fts_query;

mod branches;
/// End-to-end integration test module for table version tracking.
mod e2e_table_version_tracking;
mod namespaces;
mod register_declare;
mod storage;
mod table_versions;
mod tables;
mod unit;

fn assert_plan_contains_all(plan: &str, expected_fragments: &[&str], context: &str) {
    for expected_fragment in expected_fragments {
        assert!(
            plan.contains(expected_fragment),
            "{}. Missing fragment: '{}'. Plan:\n{}",
            context,
            expected_fragment,
            plan
        );
    }
}

fn mutation_error_code(err: lance_core::Error) -> ErrorCode {
    match err {
        lance_core::Error::Namespace { source, .. } => source
            .downcast_ref::<NamespaceError>()
            .expect("mutation error should wrap a NamespaceError")
            .code(),
        other => panic!("expected Namespace error, got: {other:?}"),
    }
}

/// Helper to create a test DirectoryNamespace with a temporary directory
async fn create_test_namespace() -> (DirectoryNamespace, TempStdDir) {
    let temp_dir = TempStdDir::default();

    let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
        .build()
        .await
        .unwrap();
    (namespace, temp_dir)
}

/// Helper to create test IPC data from a schema
fn create_test_ipc_data(schema: &JsonArrowSchema) -> Vec<u8> {
    use arrow::ipc::writer::StreamWriter;

    let arrow_schema = convert_json_arrow_schema(schema).unwrap();
    let arrow_schema = Arc::new(arrow_schema);
    let batch = arrow::record_batch::RecordBatch::new_empty(arrow_schema.clone());
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }
    buffer
}

fn create_ipc_data_from_batches(
    schema: Arc<arrow_schema::Schema>,
    batches: Vec<arrow::record_batch::RecordBatch>,
) -> Vec<u8> {
    use arrow::ipc::writer::StreamWriter;

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema).unwrap();
        for batch in &batches {
            writer.write(batch).unwrap();
        }
        writer.finish().unwrap();
    }
    buffer
}

fn create_non_empty_test_ipc_data() -> Vec<u8> {
    use arrow::array::{Int32Array, StringArray};
    use arrow::record_batch::RecordBatch;

    let schema = Arc::new(convert_json_arrow_schema(&create_test_schema()).unwrap());
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob")])),
        ],
    )
    .unwrap();
    create_ipc_data_from_batches(schema, vec![batch])
}

fn create_single_row_test_ipc_data() -> Vec<u8> {
    use arrow::array::{Int32Array, StringArray};
    use arrow::record_batch::RecordBatch;

    let schema = Arc::new(convert_json_arrow_schema(&create_test_schema()).unwrap());
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![10])),
            Arc::new(StringArray::from(vec![Some("carol")])),
        ],
    )
    .unwrap();
    create_ipc_data_from_batches(schema, vec![batch])
}

/// Helper to create a simple test schema
fn create_test_schema() -> JsonArrowSchema {
    let int_type = JsonArrowDataType::new("int32".to_string());
    let string_type = JsonArrowDataType::new("utf8".to_string());

    let id_field = JsonArrowField {
        name: "id".to_string(),
        r#type: Box::new(int_type),
        nullable: false,
        metadata: None,
    };

    let name_field = JsonArrowField {
        name: "name".to_string(),
        r#type: Box::new(string_type),
        nullable: true,
        metadata: None,
    };

    JsonArrowSchema {
        fields: vec![id_field, name_field],
        metadata: None,
    }
}

fn create_scalar_table_ipc_data() -> Vec<u8> {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["alice", "bob", "cory"])),
        ],
    )
    .unwrap();
    create_ipc_data_from_batches(schema, vec![batch])
}

async fn create_legacy_manifest_without_primary_key_metadata(root: &str) {
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("object_id", DataType::Utf8, false),
        Field::new("object_type", DataType::Utf8, false),
        Field::new("location", DataType::Utf8, true),
        Field::new("metadata", DataType::Utf8, true),
        Field::new(
            "base_objects",
            DataType::List(Arc::new(Field::new("object_id", DataType::Utf8, true))),
            true,
        ),
    ]));
    let batch = RecordBatch::new_empty(schema.clone());
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Dataset::write(Box::new(reader), &format!("{}/__manifest", root), None)
        .await
        .unwrap();
}

async fn manifest_has_primary_key_metadata(root: &str) -> bool {
    let dataset = Dataset::open(&format!("{}/__manifest", root))
        .await
        .unwrap();
    dataset
        .schema()
        .field("object_id")
        .map(|field| {
            field
                .metadata
                .contains_key(lance_core::datatypes::LANCE_UNENFORCED_PRIMARY_KEY_POSITION)
        })
        .unwrap_or(false)
}

fn create_vector_table_ipc_data() -> Vec<u8> {
    use arrow::array::{FixedSizeListArray, Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2),
            true,
        ),
    ]));
    let vector_field = Arc::new(Field::new("item", DataType::Float32, true));
    let vectors = FixedSizeListArray::try_new(
        vector_field,
        2,
        Arc::new(Float32Array::from(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
        None,
    )
    .unwrap();
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(vectors)],
    )
    .unwrap();
    create_ipc_data_from_batches(schema, vec![batch])
}

async fn create_scalar_table(namespace: &DirectoryNamespace, table_name: &str) {
    let mut create_table_request = CreateTableRequest::new();
    create_table_request.id = Some(vec![table_name.to_string()]);
    namespace
        .create_table(
            create_table_request,
            Bytes::from(create_scalar_table_ipc_data()),
        )
        .await
        .unwrap();
}

async fn create_vector_table(namespace: &DirectoryNamespace, table_name: &str) {
    let mut create_table_request = CreateTableRequest::new();
    create_table_request.id = Some(vec![table_name.to_string()]);
    namespace
        .create_table(
            create_table_request,
            Bytes::from(create_vector_table_ipc_data()),
        )
        .await
        .unwrap();
}

async fn open_dataset(namespace: &DirectoryNamespace, table_name: &str) -> Dataset {
    let mut describe_request = DescribeTableRequest::new();
    describe_request.id = Some(vec![table_name.to_string()]);
    let table_uri = namespace
        .describe_table(describe_request)
        .await
        .unwrap()
        .location
        .expect("table location should exist");
    Dataset::open(&table_uri).await.unwrap()
}

async fn create_scalar_index(
    namespace: &DirectoryNamespace,
    table_name: &str,
    index_name: &str,
) -> Option<String> {
    use lance_namespace::models::CreateTableIndexRequest;

    let mut create_index_request =
        CreateTableIndexRequest::new("id".to_string(), "BTREE".to_string());
    create_index_request.id = Some(vec![table_name.to_string()]);
    create_index_request.name = Some(index_name.to_string());
    namespace
        .create_table_scalar_index(create_index_request)
        .await
        .unwrap()
        .transaction_id
}

/// Fork `branch_name` from the table's current version and append
/// `extra_versions` commits to it (each a new version on the branch, written
/// with the default V2 naming). The main branch is left untouched. Returns
/// the branch's storage URI (`<root>/tree/<branch>`).
async fn create_branch_with_commits(
    namespace: &DirectoryNamespace,
    table_name: &str,
    branch_name: &str,
    extra_versions: usize,
) -> String {
    let mut main = open_dataset(namespace, table_name).await;
    let fork_version = main.version().version;
    let branch = main
        .create_branch(branch_name, fork_version, None)
        .await
        .unwrap();
    let branch_uri = branch.uri().to_string();
    for i in 0..extra_versions {
        append_scalar_version(&branch_uri, (i as i32 + 1) * 100).await;
    }
    branch_uri
}

/// Append one scalar-schema batch to the dataset at `uri`, creating a new
/// version (default V2 naming). Shared by branch and main chain setup.
async fn append_scalar_version(uri: &str, seed: i32) {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![seed, seed + 1])),
            Arc::new(StringArray::from(vec![Some("x"), Some("y")])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
}

/// List a table's versions on `branch` (None == main) via the namespace.
async fn list_versions(
    namespace: &DirectoryNamespace,
    table_name: &str,
    branch: Option<&str>,
) -> Result<Vec<TableVersion>> {
    let req = ListTableVersionsRequest {
        id: Some(vec![table_name.to_string()]),
        branch: branch.map(|b| b.to_string()),
        ..Default::default()
    };
    namespace.list_table_versions(req).await.map(|r| r.versions)
}
