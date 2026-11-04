// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

use std::sync::atomic::{AtomicUsize, Ordering};

/// Tracking wrapper around a namespace that counts method invocations.
struct TrackingNamespace {
    inner: DirectoryNamespace,
    create_table_version_count: AtomicUsize,
    describe_table_version_count: AtomicUsize,
    list_table_versions_count: AtomicUsize,
}

impl TrackingNamespace {
    fn new(inner: DirectoryNamespace) -> Self {
        Self {
            inner,
            create_table_version_count: AtomicUsize::new(0),
            describe_table_version_count: AtomicUsize::new(0),
            list_table_versions_count: AtomicUsize::new(0),
        }
    }

    fn create_table_version_calls(&self) -> usize {
        self.create_table_version_count.load(Ordering::SeqCst)
    }

    fn describe_table_version_calls(&self) -> usize {
        self.describe_table_version_count.load(Ordering::SeqCst)
    }

    fn list_table_versions_calls(&self) -> usize {
        self.list_table_versions_count.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for TrackingNamespace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackingNamespace")
            .field(
                "create_table_version_calls",
                &self.create_table_version_calls(),
            )
            .finish()
    }
}

#[async_trait]
impl LanceNamespace for TrackingNamespace {
    async fn create_namespace(
        &self,
        request: CreateNamespaceRequest,
    ) -> Result<CreateNamespaceResponse> {
        self.inner.create_namespace(request).await
    }

    async fn describe_namespace(
        &self,
        request: DescribeNamespaceRequest,
    ) -> Result<DescribeNamespaceResponse> {
        self.inner.describe_namespace(request).await
    }

    async fn namespace_exists(&self, request: NamespaceExistsRequest) -> Result<()> {
        self.inner.namespace_exists(request).await
    }

    async fn list_namespaces(
        &self,
        request: ListNamespacesRequest,
    ) -> Result<ListNamespacesResponse> {
        self.inner.list_namespaces(request).await
    }

    async fn drop_namespace(&self, request: DropNamespaceRequest) -> Result<DropNamespaceResponse> {
        self.inner.drop_namespace(request).await
    }

    async fn list_tables(&self, request: ListTablesRequest) -> Result<ListTablesResponse> {
        self.inner.list_tables(request).await
    }

    async fn describe_table(&self, request: DescribeTableRequest) -> Result<DescribeTableResponse> {
        self.inner.describe_table(request).await
    }

    async fn table_exists(&self, request: TableExistsRequest) -> Result<()> {
        self.inner.table_exists(request).await
    }

    async fn drop_table(&self, request: DropTableRequest) -> Result<DropTableResponse> {
        self.inner.drop_table(request).await
    }

    async fn create_table(
        &self,
        request: CreateTableRequest,
        request_data: Bytes,
    ) -> Result<CreateTableResponse> {
        self.inner.create_table(request, request_data).await
    }

    async fn declare_table(&self, request: DeclareTableRequest) -> Result<DeclareTableResponse> {
        self.inner.declare_table(request).await
    }

    async fn list_table_versions(
        &self,
        request: ListTableVersionsRequest,
    ) -> Result<ListTableVersionsResponse> {
        self.list_table_versions_count
            .fetch_add(1, Ordering::SeqCst);
        self.inner.list_table_versions(request).await
    }

    async fn create_table_version(
        &self,
        request: CreateTableVersionRequest,
    ) -> Result<CreateTableVersionResponse> {
        self.create_table_version_count
            .fetch_add(1, Ordering::SeqCst);
        self.inner.create_table_version(request).await
    }

    async fn describe_table_version(
        &self,
        request: DescribeTableVersionRequest,
    ) -> Result<DescribeTableVersionResponse> {
        self.describe_table_version_count
            .fetch_add(1, Ordering::SeqCst);
        self.inner.describe_table_version(request).await
    }

    async fn batch_delete_table_versions(
        &self,
        request: BatchDeleteTableVersionsRequest,
    ) -> Result<BatchDeleteTableVersionsResponse> {
        self.inner.batch_delete_table_versions(request).await
    }

    fn namespace_id(&self) -> String {
        self.inner.namespace_id()
    }
}

#[tokio::test]
async fn test_describe_table_returns_managed_versioning() {
    use lance_namespace::models::{CreateNamespaceRequest, DescribeTableRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace with table_version_tracking_enabled and manifest_enabled
    let ns = DirectoryNamespaceBuilder::new(temp_path)
        .table_version_tracking_enabled(true)
        .manifest_enabled(true)
        .build()
        .await
        .unwrap();

    // Create parent namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["workspace".to_string()]);
    ns.create_namespace(create_ns_req).await.unwrap();

    // Create a table with multi-level ID (namespace + table)
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["workspace".to_string(), "test_table".to_string()]);
    ns.create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Describe table should return managed_versioning=true
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["workspace".to_string(), "test_table".to_string()]);
    let describe_resp = ns.describe_table(describe_req).await.unwrap();

    // managed_versioning should be true
    assert_eq!(
        describe_resp.managed_versioning,
        Some(true),
        "managed_versioning should be true when table_version_tracking_enabled=true"
    );
}

#[tokio::test]
async fn test_external_manifest_store_invokes_namespace_apis() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use lance::Dataset;
    use lance::dataset::builder::DatasetBuilder;
    use lance::dataset::{WriteMode, WriteParams};
    use lance_namespace::models::CreateNamespaceRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace with table_version_tracking_enabled and manifest_enabled
    let inner_ns = DirectoryNamespaceBuilder::new(temp_path)
        .table_version_tracking_enabled(true)
        .manifest_enabled(true)
        .build()
        .await
        .unwrap();

    let tracking_ns = Arc::new(TrackingNamespace::new(inner_ns));
    let ns: Arc<dyn LanceNamespace> = tracking_ns.clone();

    // Create parent namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["workspace".to_string()]);
    ns.create_namespace(create_ns_req).await.unwrap();

    // Create a table with multi-level ID (namespace + table)
    let table_id = vec!["workspace".to_string(), "test_table".to_string()];

    // Create some initial data
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();

    // Create a table using write_into_namespace
    let batches = RecordBatchIterator::new(vec![Ok(batch.clone())], arrow_schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Create,
        ..Default::default()
    };
    let mut dataset =
        Dataset::write_into_namespace(batches, ns.clone(), table_id.clone(), Some(write_params))
            .await
            .unwrap();
    assert_eq!(dataset.version().version, 1);

    // Verify create_table_version was called once during initial write_into_namespace
    assert_eq!(
        tracking_ns.create_table_version_calls(),
        1,
        "create_table_version should have been called once during initial write_into_namespace"
    );

    // Append data - this should call create_table_version again
    let append_batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5, 6])),
            Arc::new(StringArray::from(vec!["d", "e", "f"])),
        ],
    )
    .unwrap();
    let append_batches = RecordBatchIterator::new(vec![Ok(append_batch)], arrow_schema);
    dataset.append(append_batches, None).await.unwrap();

    assert_eq!(
        tracking_ns.create_table_version_calls(),
        2,
        "create_table_version should have been called twice (once for create, once for append)"
    );

    // checkout_latest should call list_table_versions exactly once
    let initial_list_calls = tracking_ns.list_table_versions_calls();
    let latest_dataset = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .load()
        .await
        .unwrap();
    assert_eq!(latest_dataset.version().version, 2);
    assert_eq!(
        tracking_ns.list_table_versions_calls(),
        initial_list_calls + 1,
        "list_table_versions should have been called exactly once during checkout_latest"
    );

    // checkout to specific version should call describe_table_version exactly once
    let initial_describe_calls = tracking_ns.describe_table_version_calls();
    let v1_dataset = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .with_version(1)
        .load()
        .await
        .unwrap();
    assert_eq!(v1_dataset.version().version, 1);
    assert_eq!(
        tracking_ns.describe_table_version_calls(),
        initial_describe_calls + 1,
        "describe_table_version should have been called exactly once during checkout to version 1"
    );
}

#[tokio::test]
async fn test_dataset_commit_with_external_manifest_store() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use futures::TryStreamExt;
    use lance::dataset::{Dataset, WriteMode, WriteParams};
    use lance_namespace::models::CreateNamespaceRequest;
    use lance_table::io::commit::ManifestNamingScheme;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace with table_version_tracking_enabled and manifest_enabled
    let inner_ns = DirectoryNamespaceBuilder::new(temp_path)
        .table_version_tracking_enabled(true)
        .manifest_enabled(true)
        .build()
        .await
        .unwrap();

    let tracking_ns: Arc<dyn LanceNamespace> = Arc::new(TrackingNamespace::new(inner_ns));

    // Create parent namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["workspace".to_string()]);
    tracking_ns.create_namespace(create_ns_req).await.unwrap();

    // Create a table using write_into_namespace
    let table_id = vec!["workspace".to_string(), "test_table".to_string()];
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Create,
        ..Default::default()
    };
    let dataset = Dataset::write_into_namespace(
        batches,
        tracking_ns.clone(),
        table_id.clone(),
        Some(write_params),
    )
    .await
    .unwrap();
    assert_eq!(dataset.version().version, 1);

    // Append data using write_into_namespace (APPEND mode)
    let batch2 = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5, 6])),
            Arc::new(StringArray::from(vec!["d", "e", "f"])),
        ],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch2)], arrow_schema);
    let write_params = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };
    Dataset::write_into_namespace(
        batches,
        tracking_ns.clone(),
        table_id.clone(),
        Some(write_params),
    )
    .await
    .unwrap();

    // Verify version 2 was created using the dataset's object_store
    // List manifests in the versions directory to find the V2 named manifest
    let manifest_metas: Vec<_> = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .list(Some(&dataset.versions_dir()))
        .try_collect()
        .await
        .unwrap();
    let version_2_found = manifest_metas.iter().any(|m| {
        m.location
            .filename()
            .map(|f| {
                f.ends_with(".manifest") && ManifestNamingScheme::V2.parse_version(f) == Some(2)
            })
            .unwrap_or(false)
    });
    assert!(
        version_2_found,
        "Version 2 manifest should exist in versions directory"
    );
}

/// Helper: create a namespace and a table with some rows, returning (namespace, table_id)
async fn create_ns_with_table() -> (DirectoryNamespace, TempStdDir, Vec<String>) {
    use arrow::array::{Int32Array, StringArray};
    use arrow::ipc::writer::StreamWriter;

    let (namespace, temp_dir) = create_test_namespace().await;

    let schema = create_test_schema();
    let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
    let arrow_schema = Arc::new(arrow_schema);

    let id_array = Int32Array::from(vec![1, 2, 3]);
    let name_array = StringArray::from(vec!["Alice", "Bob", "Charlie"]);
    let batch = arrow::record_batch::RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(id_array), Arc::new(name_array)],
    )
    .unwrap();

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }

    let mut request = CreateTableRequest::new();
    let table_id = vec!["test_ops_table".to_string()];
    request.id = Some(table_id.clone());

    namespace
        .create_table(request, Bytes::from(buffer))
        .await
        .unwrap();

    (namespace, temp_dir, table_id)
}

#[tokio::test]
async fn test_count_table_rows_basic() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = CountTableRowsRequest {
        id: Some(table_id),
        version: None,
        predicate: None,
        ..Default::default()
    };

    let count = namespace.count_table_rows(request).await.unwrap();
    assert_eq!(count, 3);
}

#[tokio::test]
async fn test_count_table_rows_with_predicate() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = CountTableRowsRequest {
        id: Some(table_id),
        version: None,
        predicate: Some("id > 1".to_string()),
        ..Default::default()
    };

    let count = namespace.count_table_rows(request).await.unwrap();
    assert_eq!(count, 2);
}

#[tokio::test]
async fn test_query_table_invalid_distance_type() {
    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

    let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
        single_vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
        multi_vector: None,
    });

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 2,
        vector,
        vector_column: Some("vector".to_string()),
        distance_type: Some("invalid_metric".to_string()),
        filter: None,
        offset: None,
        version: None,
        ..Default::default()
    };

    let result = namespace.query_table(request).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Unknown distance type"),
        "Expected error about unknown distance type, got: {}",
        err_msg
    );
}

#[tokio::test]
async fn test_insert_into_table_append() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::ipc::writer::StreamWriter;

    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    // Prepare new data to insert
    let schema = create_test_schema();
    let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
    let arrow_schema = Arc::new(arrow_schema);

    let id_array = Int32Array::from(vec![4, 5]);
    let name_array = StringArray::from(vec!["Dave", "Eve"]);
    let batch = arrow::record_batch::RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(id_array), Arc::new(name_array)],
    )
    .unwrap();

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }

    let request = InsertIntoTableRequest {
        id: Some(table_id.clone()),
        mode: Some("append".to_string()),
        ..Default::default()
    };

    let response = namespace
        .insert_into_table(request, Bytes::from(buffer))
        .await
        .unwrap();
    assert!(response.transaction_id.is_none());
    assert_eq!(response.num_inserted_rows, Some(2));
    assert_eq!(response.version, Some(2));

    // Verify total rows
    let count_req = CountTableRowsRequest {
        id: Some(table_id),
        version: None,
        predicate: None,
        ..Default::default()
    };
    let count = namespace.count_table_rows(count_req).await.unwrap();
    assert_eq!(count, 5);
}

#[tokio::test]
async fn test_insert_into_table_overwrite() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::ipc::writer::StreamWriter;

    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let schema = create_test_schema();
    let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
    let arrow_schema = Arc::new(arrow_schema);

    let id_array = Int32Array::from(vec![10, 20]);
    let name_array = StringArray::from(vec!["X", "Y"]);
    let batch = arrow::record_batch::RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(id_array), Arc::new(name_array)],
    )
    .unwrap();

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }

    let request = InsertIntoTableRequest {
        id: Some(table_id.clone()),
        mode: Some("overwrite".to_string()),
        ..Default::default()
    };

    let response = namespace
        .insert_into_table(request, Bytes::from(buffer))
        .await
        .unwrap();
    assert_eq!(response.num_inserted_rows, Some(2));
    assert_eq!(response.version, Some(2));

    // Verify overwrite: only 2 rows remain
    let count_req = CountTableRowsRequest {
        id: Some(table_id),
        version: None,
        predicate: None,
        ..Default::default()
    };
    let count = namespace.count_table_rows(count_req).await.unwrap();
    assert_eq!(count, 2);
}

#[tokio::test]
async fn test_insert_into_table_empty_data() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = InsertIntoTableRequest {
        id: Some(table_id),
        mode: None,
        ..Default::default()
    };

    let result = namespace.insert_into_table(request, Bytes::new()).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Arrow IPC stream) is required")
    );
}

#[tokio::test]
async fn test_insert_into_table_with_storage_options() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::ipc::writer::StreamWriter;

    let temp_dir = TempStdDir::default();

    // Build namespace with a (no-op) storage option so self.storage_options is Some
    let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
        .storage_option("allow_http", "true")
        .build()
        .await
        .unwrap();

    // Create a table first
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    let table_id = vec!["so_table".to_string()];
    create_req.id = Some(table_id.clone());
    namespace
        .create_table(create_req, Bytes::from(ipc_data))
        .await
        .unwrap();

    // Insert with storage_options present — covers store_params closure
    let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
    let arrow_schema = Arc::new(arrow_schema);

    let id_array = Int32Array::from(vec![10, 20]);
    let name_array = StringArray::from(vec!["X", "Y"]);
    let batch = arrow::record_batch::RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(id_array), Arc::new(name_array)],
    )
    .unwrap();

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }

    let request = InsertIntoTableRequest {
        id: Some(table_id.clone()),
        mode: Some("append".to_string()),
        ..Default::default()
    };

    let response = namespace
        .insert_into_table(request, Bytes::from(buffer))
        .await
        .unwrap();
    assert!(response.transaction_id.is_none());
    assert_eq!(response.num_inserted_rows, Some(2));
    assert_eq!(response.version, Some(2));

    // Verify rows were inserted
    let count_req = CountTableRowsRequest {
        id: Some(table_id),
        version: None,
        predicate: None,
        ..Default::default()
    };
    let count = namespace.count_table_rows(count_req).await.unwrap();
    assert_eq!(count, 2);
}

#[tokio::test]
async fn test_query_table_basic() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 10,
        filter: None,
        offset: None,
        version: None,
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();

    // Decode IPC and verify
    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);
}

#[tokio::test]
async fn test_query_table_with_filter() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 10,
        filter: Some("id <= 2".to_string()),
        offset: None,
        version: None,
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);
}

#[tokio::test]
async fn test_query_table_with_limit_and_offset() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 2,
        filter: None,
        offset: Some(1),
        version: None,
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);
}

#[tokio::test]
async fn test_query_table_no_limit() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    // k=0 means no limit
    let request = QueryTableRequest {
        id: Some(table_id),
        k: 0,
        filter: None,
        offset: None,
        version: None,
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);
}

#[tokio::test]
async fn test_query_table_with_columns() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let columns = Box::new(lance_namespace::models::QueryTableRequestColumns {
        column_names: Some(vec!["id".to_string()]),
        column_aliases: None,
    });

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 10,
        filter: None,
        offset: None,
        version: None,
        columns: Some(columns),
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let schema = reader.schema();
    assert_eq!(schema.fields().len(), 1);
    assert_eq!(schema.field(0).name(), "id");
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);
}

#[tokio::test]
async fn test_count_table_rows_with_version() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::ipc::writer::StreamWriter;

    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    // Insert more data to create version 2
    let schema = create_test_schema();
    let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
    let arrow_schema = Arc::new(arrow_schema);

    let id_array = Int32Array::from(vec![4, 5]);
    let name_array = StringArray::from(vec!["Dave", "Eve"]);
    let batch = arrow::record_batch::RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(id_array), Arc::new(name_array)],
    )
    .unwrap();

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }

    let request = InsertIntoTableRequest {
        id: Some(table_id.clone()),
        mode: None,
        ..Default::default()
    };
    namespace
        .insert_into_table(request, Bytes::from(buffer))
        .await
        .unwrap();

    // Version 1 should have 3 rows
    let count_req = CountTableRowsRequest {
        id: Some(table_id.clone()),
        version: Some(1),
        predicate: None,
        ..Default::default()
    };
    let count = namespace.count_table_rows(count_req).await.unwrap();
    assert_eq!(count, 3);

    // Latest version should have 5 rows
    let count_req = CountTableRowsRequest {
        id: Some(table_id),
        version: None,
        predicate: None,
        ..Default::default()
    };
    let count = namespace.count_table_rows(count_req).await.unwrap();
    assert_eq!(count, 5);
}

#[tokio::test]
async fn test_query_table_with_version() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::ipc::writer::StreamWriter;

    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    // Insert more data to create version 2
    let schema = create_test_schema();
    let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
    let arrow_schema = Arc::new(arrow_schema);

    let id_array = Int32Array::from(vec![4, 5]);
    let name_array = StringArray::from(vec!["Dave", "Eve"]);
    let batch = arrow::record_batch::RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(id_array), Arc::new(name_array)],
    )
    .unwrap();

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }

    let request = InsertIntoTableRequest {
        id: Some(table_id.clone()),
        mode: None,
        ..Default::default()
    };
    namespace
        .insert_into_table(request, Bytes::from(buffer))
        .await
        .unwrap();

    // Query version 1 should return 3 rows
    let request = QueryTableRequest {
        id: Some(table_id.clone()),
        k: 100,
        filter: None,
        offset: None,
        version: Some(1),
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();
    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);

    // Query latest version should return 5 rows
    let request = QueryTableRequest {
        id: Some(table_id),
        k: 100,
        filter: None,
        offset: None,
        version: None,
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();
    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 5);
}

/// Helper to create a namespace with a table that has a vector column for
/// vector search tests.
async fn create_ns_with_vector_table() -> (DirectoryNamespace, TempStdDir, Vec<String>) {
    use arrow::array::{FixedSizeListArray, Float32Array, Int32Array};
    use arrow::ipc::writer::StreamWriter;

    let (namespace, temp_dir) = create_test_namespace().await;

    // Build schema: id (int32), vector (fixed_size_list<float32>[4])
    let arrow_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int32, false),
        arrow::datatypes::Field::new(
            "vector",
            arrow::datatypes::DataType::FixedSizeList(
                Arc::new(arrow::datatypes::Field::new(
                    "item",
                    arrow::datatypes::DataType::Float32,
                    true,
                )),
                4,
            ),
            true,
        ),
    ]));

    let id_array = Int32Array::from(vec![1, 2, 3]);
    let values = Float32Array::from(vec![
        1.0, 0.0, 0.0, 0.0, // vector for id=1
        0.0, 1.0, 0.0, 0.0, // vector for id=2
        0.0, 0.0, 1.0, 0.0, // vector for id=3
    ]);
    let vector_array = FixedSizeListArray::try_new(
        Arc::new(arrow::datatypes::Field::new(
            "item",
            arrow::datatypes::DataType::Float32,
            true,
        )),
        4,
        Arc::new(values),
        None,
    )
    .unwrap();

    let batch = arrow::record_batch::RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(id_array), Arc::new(vector_array)],
    )
    .unwrap();

    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }

    // Write as a Lance dataset directly
    let table_name = "vector_table";
    let table_uri = format!("{}/{}.lance", temp_dir.to_str().unwrap(), table_name);
    let reader =
        arrow::record_batch::RecordBatchIterator::new(vec![Ok(batch)], arrow_schema.clone());
    Dataset::write(reader, &table_uri, None).await.unwrap();

    let table_id = vec![table_name.to_string()];
    (namespace, temp_dir, table_id)
}

#[tokio::test]
async fn test_query_table_vector_search() {
    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

    let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
        single_vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
        multi_vector: None,
    });

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 2,
        vector,
        filter: None,
        offset: None,
        version: None,
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);
    assert!(batches[0].column_by_name("query_index").is_none());
}

#[tokio::test]
async fn test_query_table_multi_vector_batch_search() {
    use arrow::array::{Array, AsArray};
    use arrow::datatypes::Int32Type;

    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;
    let request = QueryTableRequest {
        id: Some(table_id),
        k: 2,
        vector: Box::new(lance_namespace::models::QueryTableRequestVector {
            single_vector: None,
            multi_vector: Some(vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ]),
        }),
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();
    let reader = FileReader::try_new(Cursor::new(bytes.to_vec()), None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter()).unwrap();

    assert_eq!(batch.num_rows(), 6);
    let schema = batch.schema();
    let query_index_field = schema.field(0);
    assert_eq!(query_index_field.name(), "query_index");
    assert!(!query_index_field.is_nullable());
    let query_index = batch["query_index"].as_primitive::<Int32Type>();
    assert_eq!(query_index.null_count(), 0);
    assert_eq!(query_index.values().as_ref(), &[0, 0, 1, 1, 2, 2]);

    let ids = batch["id"].as_primitive::<Int32Type>();
    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(2), 3);
    assert_eq!(ids.value(4), 2);
}

#[tokio::test]
async fn test_query_table_multi_vector_offset_boundary() {
    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;
    let vector = || {
        Box::new(lance_namespace::models::QueryTableRequestVector {
            single_vector: None,
            multi_vector: Some(vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
            ]),
        })
    };

    let request = QueryTableRequest {
        id: Some(table_id.clone()),
        k: 2,
        vector: vector(),
        offset: Some(0),
        ..Default::default()
    };
    let bytes = namespace.query_table(request).await.unwrap();
    let reader = FileReader::try_new(Cursor::new(bytes.to_vec()), None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter()).unwrap();
    assert_eq!(batch.num_rows(), 6);

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 2,
        vector: vector(),
        offset: Some(1),
        ..Default::default()
    };
    let err = namespace.query_table(request).await.unwrap_err();
    let lance_core::Error::Namespace { source, .. } = &err else {
        panic!("expected a Namespace error, got: {}", err);
    };
    let ns_err = source
        .downcast_ref::<NamespaceError>()
        .expect("expected a NamespaceError source");
    assert_eq!(ns_err.code(), ErrorCode::InvalidInput);
    assert!(
        err.to_string()
            .contains("multi_vector batch search does not support offset > 0"),
        "unexpected error: {}",
        err
    );
}

#[tokio::test]
async fn test_query_table_multi_vector_batch_search_with_filter_and_prefilter() {
    use arrow::array::AsArray;
    use arrow::datatypes::Int32Type;

    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;
    let request = QueryTableRequest {
        id: Some(table_id),
        k: 1,
        vector: Box::new(lance_namespace::models::QueryTableRequestVector {
            single_vector: None,
            multi_vector: Some(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]),
        }),
        filter: Some("id = 3".to_string()),
        prefilter: Some(true),
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();
    let reader = FileReader::try_new(Cursor::new(bytes.to_vec()), None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter()).unwrap();

    assert_eq!(batch.num_rows(), 2);
    let ids = batch["id"].as_primitive::<Int32Type>();
    assert_eq!(ids.values().as_ref(), &[3, 3]);
    let query_index = batch["query_index"].as_primitive::<Int32Type>();
    assert_eq!(query_index.values().as_ref(), &[0, 1]);
}

#[tokio::test]
async fn test_query_table_multi_vector_empty_result_keeps_query_index_schema() {
    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;
    let request = QueryTableRequest {
        id: Some(table_id),
        k: 2,
        vector: Box::new(lance_namespace::models::QueryTableRequestVector {
            single_vector: None,
            multi_vector: Some(vec![vec![1.0, 0.0, 0.0, 0.0]]),
        }),
        filter: Some("id > 99".to_string()),
        prefilter: Some(true),
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();
    let reader = FileReader::try_new(Cursor::new(bytes.to_vec()), None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter()).unwrap();

    assert_eq!(batch.num_rows(), 0);
    let schema = batch.schema();
    let query_index_field = schema.field(0);
    assert_eq!(query_index_field.name(), "query_index");
    assert!(!query_index_field.is_nullable());
}

#[tokio::test]
async fn test_query_table_multi_vector_on_multivector_column() {
    use arrow::array::{Array, FixedSizeListBuilder, Float32Builder, Int32Array, ListBuilder};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    let (namespace, temp_dir) = create_test_namespace().await;

    let mut builder = ListBuilder::new(FixedSizeListBuilder::new(Float32Builder::new(), 2));
    for row in [
        vec![[1.0, 0.0], [0.0, 1.0]],
        vec![[5.0, 5.0]],
        vec![[-1.0, 0.0], [0.0, -1.0]],
    ] {
        for vector in row {
            builder.values().values().append_slice(&vector);
            builder.values().append(true);
        }
        builder.append(true);
    }
    let vectors = builder.finish();
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("vector", vectors.data_type().clone(), true),
    ]));
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3])), Arc::new(vectors)],
    )
    .unwrap();
    let table_uri = format!("{}/multivec.lance", temp_dir.to_str().unwrap());
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
    Dataset::write(reader, &table_uri, None).await.unwrap();

    let request = QueryTableRequest {
        id: Some(vec!["multivec".to_string()]),
        k: 1,
        vector: Box::new(lance_namespace::models::QueryTableRequestVector {
            single_vector: None,
            multi_vector: Some(vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
        }),
        distance_type: Some("cosine".to_string()),
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();
    let reader = FileReader::try_new(Cursor::new(bytes.to_vec()), None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let batch = arrow::compute::concat_batches(&batches[0].schema(), batches.iter()).unwrap();

    assert!(batch.column_by_name("query_index").is_none());
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(
        batch["id"]
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .value(0),
        1
    );
}

#[rstest]
#[case::ragged_rows(
    None,
    Some(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0]]),
    "row 0 has 4 values but row 1 has 2"
)]
#[case::empty_row(
    None,
    Some(vec![vec![1.0, 0.0, 0.0, 0.0], vec![]]),
    "multi_vector row 1 is empty"
)]
#[case::empty_first_row(
    None,
    Some(vec![vec![], vec![1.0, 0.0, 0.0, 0.0]]),
    "multi_vector row 0 is empty"
)]
#[case::single_and_multi(
    Some(vec![1.0, 0.0, 0.0, 0.0]),
    Some(vec![vec![1.0, 0.0, 0.0, 0.0]]),
    "provide either single_vector or multi_vector, not both"
)]
#[tokio::test]
async fn test_query_table_rejects_invalid_query_vectors(
    #[case] single_vector: Option<Vec<f32>>,
    #[case] multi_vector: Option<Vec<Vec<f32>>>,
    #[case] expected: &str,
) {
    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;
    let request = QueryTableRequest {
        id: Some(table_id),
        k: 2,
        vector: Box::new(lance_namespace::models::QueryTableRequestVector {
            single_vector,
            multi_vector,
        }),
        ..Default::default()
    };

    let err = namespace.query_table(request).await.unwrap_err();
    let lance_core::Error::Namespace { source, .. } = &err else {
        panic!("expected a Namespace error, got: {}", err);
    };
    let ns_err = source
        .downcast_ref::<NamespaceError>()
        .expect("expected a NamespaceError source");
    assert_eq!(ns_err.code(), ErrorCode::InvalidInput);
    assert!(
        err.to_string().contains(expected),
        "unexpected error: {}",
        err
    );
}

#[tokio::test]
async fn test_explain_table_query_plan_multi_vector() {
    use lance_namespace::models::ExplainTableQueryPlanRequest;

    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;
    let query = QueryTableRequest {
        k: 2,
        vector: Box::new(lance_namespace::models::QueryTableRequestVector {
            single_vector: None,
            multi_vector: Some(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]),
        }),
        ..Default::default()
    };
    let mut request = ExplainTableQueryPlanRequest::new(query);
    request.id = Some(table_id);

    let plan = namespace.explain_table_query_plan(request).await.unwrap();
    assert!(plan.contains("query_index"), "unexpected plan: {}", plan);

    let query = QueryTableRequest {
        k: 2,
        vector: Box::new(lance_namespace::models::QueryTableRequestVector {
            single_vector: None,
            multi_vector: Some(vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]]),
        }),
        offset: Some(1),
        ..Default::default()
    };
    let mut request = ExplainTableQueryPlanRequest::new(query);
    request.id = Some(vec!["vector_table".to_string()]);
    let err = namespace
        .explain_table_query_plan(request)
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("multi_vector batch search does not support offset > 0"),
        "unexpected error: {}",
        err
    );
}

#[tokio::test]
async fn test_query_table_vector_search_with_distance_type() {
    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

    let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
        single_vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
        multi_vector: None,
    });

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 3,
        vector,
        filter: None,
        offset: None,
        version: None,
        distance_type: Some("cosine".to_string()),
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);
}

#[tokio::test]
async fn test_query_table_vector_search_with_filter() {
    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

    let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
        single_vector: Some(vec![1.0, 0.0, 0.0, 0.0]),
        multi_vector: None,
    });

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 10,
        vector,
        filter: Some("id <= 2".to_string()),
        offset: None,
        version: None,
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows <= 2);
}

#[tokio::test]
async fn test_query_table_vector_search_with_nprobes_and_refine() {
    let (namespace, _temp_dir, table_id) = create_ns_with_vector_table().await;

    let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
        single_vector: Some(vec![0.0, 1.0, 0.0, 0.0]),
        multi_vector: None,
    });

    let request = QueryTableRequest {
        id: Some(table_id),
        k: 2,
        vector,
        filter: None,
        offset: None,
        version: None,
        nprobes: Some(1),
        refine_factor: Some(1),
        prefilter: Some(true),
        ..Default::default()
    };

    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);
}

#[tokio::test]
async fn test_explain_vector_probe_fields_are_applied_independently() {
    use lance_namespace::models::ExplainTableQueryPlanRequest;

    let (namespace, temp_dir, table_id) = create_ns_with_vector_table().await;
    let table_uri = format!("{}/vector_table.lance", temp_dir.to_str().unwrap());
    let mut dataset = Dataset::open(&table_uri).await.unwrap();
    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
            false,
        )
        .await
        .unwrap();
    let vector = || {
        Box::new(lance_namespace::models::QueryTableRequestVector {
            single_vector: Some(vec![0.0, 1.0, 0.0, 0.0]),
            multi_vector: None,
        })
    };

    let query = QueryTableRequest {
        id: None,
        k: 2,
        vector: vector(),
        nprobes: Some(20),
        minimum_nprobes: Some(3),
        maximum_nprobes: Some(10),
        ..Default::default()
    };
    let mut request = ExplainTableQueryPlanRequest::new(query);
    request.id = Some(table_id.clone());

    let plan = namespace.explain_table_query_plan(request).await.unwrap();
    assert!(plan.contains("minimum_nprobes=3"), "{plan}");
    assert!(plan.contains("maximum_nprobes=Some(10)"), "{plan}");

    let query = QueryTableRequest {
        id: None,
        k: 2,
        vector: vector(),
        nprobes: Some(20),
        ..Default::default()
    };
    let mut request = ExplainTableQueryPlanRequest::new(query);
    request.id = Some(table_id.clone());

    let plan = namespace.explain_table_query_plan(request).await.unwrap();
    assert!(plan.contains("minimum_nprobes=20"), "{plan}");
    assert!(plan.contains("maximum_nprobes=Some(20)"), "{plan}");

    let query = QueryTableRequest {
        id: None,
        k: 2,
        vector: vector(),
        nprobes: Some(1),
        maximum_nprobes: Some(0),
        ..Default::default()
    };
    let mut request = ExplainTableQueryPlanRequest::new(query);
    request.id = Some(table_id.clone());

    let plan = namespace.explain_table_query_plan(request).await.unwrap();
    assert!(plan.contains("minimum_nprobes=1"), "{plan}");
    assert!(plan.contains("maximum_nprobes=None"), "{plan}");

    let request = QueryTableRequest {
        id: Some(table_id.clone()),
        k: 2,
        vector: vector(),
        nprobes: Some(1),
        maximum_nprobes: Some(0),
        ..Default::default()
    };
    let bytes = namespace.query_table(request).await.unwrap();
    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let total_rows: usize = reader.map(|batch| batch.unwrap().num_rows()).sum();
    assert_eq!(total_rows, 2);

    let query = QueryTableRequest {
        id: None,
        k: 2,
        vector: vector(),
        nprobes: Some(-1),
        ..Default::default()
    };
    let mut request = ExplainTableQueryPlanRequest::new(query);
    request.id = Some(table_id);

    let err = namespace
        .explain_table_query_plan(request)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("nprobes must be non-negative"));
}

#[tokio::test]
async fn test_namespace_id() {
    let (namespace, _temp_dir) = create_test_namespace().await;
    let id = namespace.namespace_id();
    assert!(id.contains("DirectoryNamespace"));
    assert!(id.contains("root"));
}

#[tokio::test]
async fn test_query_table_empty_table() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create table with empty IPC data (schema only, no rows)
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_request = CreateTableRequest::new();
    create_request.id = Some(vec!["empty_table".to_string()]);
    namespace
        .create_table(create_request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Query the empty table — should hit the "no batches" else branch
    let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
        single_vector: None,
        multi_vector: None,
    });
    let request = QueryTableRequest {
        id: Some(vec!["empty_table".to_string()]),
        k: 10,
        vector,
        ..Default::default()
    };
    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.collect::<std::result::Result<Vec<_>, _>>().unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 0, "empty table should yield no rows");
}

#[tokio::test]
async fn test_query_table_with_plain_filter_no_vector() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    // Query with filter but no vector (plain scan path + filter)
    let vector = Box::new(lance_namespace::models::QueryTableRequestVector {
        single_vector: None,
        multi_vector: None,
    });
    let request = QueryTableRequest {
        id: Some(table_id),
        k: 0,
        vector,
        filter: Some("id > 1".to_string()),
        ..Default::default()
    };
    let bytes = namespace.query_table(request).await.unwrap();

    let cursor = Cursor::new(bytes.to_vec());
    let reader = FileReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = reader.into_iter().map(|b| b.unwrap()).collect();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(total_rows > 0);
    assert!(total_rows < 3);
}

// ---------------------- update_table / delete_from_table ----------------------

#[tokio::test]
async fn test_update_full_table() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    // Capture base version so we can assert the update bumped it.
    let base_version = open_dataset(&namespace, &table_id[0])
        .await
        .version()
        .version;

    let request = UpdateTableRequest {
        id: Some(table_id.clone()),
        updates: vec![vec!["name".to_string(), "'updated'".to_string()]],
        predicate: None,
        ..Default::default()
    };

    let response = namespace.update_table(request).await.unwrap();
    assert_eq!(response.updated_rows, 3);
    assert!(response.version as u64 > base_version);

    // Validate that all rows now carry the new value.
    let count_req = CountTableRowsRequest {
        id: Some(table_id),
        version: None,
        predicate: Some("name = 'updated'".to_string()),
        ..Default::default()
    };
    assert_eq!(namespace.count_table_rows(count_req).await.unwrap(), 3);
}

#[tokio::test]
async fn test_update_with_predicate() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = UpdateTableRequest {
        id: Some(table_id.clone()),
        updates: vec![vec!["name".to_string(), "'matched'".to_string()]],
        predicate: Some("id > 1".to_string()),
        ..Default::default()
    };

    let response = namespace.update_table(request).await.unwrap();
    assert_eq!(response.updated_rows, 2);

    // Rows that did not match the predicate must remain unchanged.
    let untouched = CountTableRowsRequest {
        id: Some(table_id.clone()),
        version: None,
        predicate: Some("name = 'Alice'".to_string()),
        ..Default::default()
    };
    assert_eq!(namespace.count_table_rows(untouched).await.unwrap(), 1);

    let touched = CountTableRowsRequest {
        id: Some(table_id),
        version: None,
        predicate: Some("name = 'matched'".to_string()),
        ..Default::default()
    };
    assert_eq!(namespace.count_table_rows(touched).await.unwrap(), 2);
}

#[tokio::test]
async fn test_update_invalid_expression_returns_invalid_input() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = UpdateTableRequest {
        id: Some(table_id),
        // Reference an unknown column on the right-hand side.
        updates: vec![vec!["name".to_string(), "no_such_column + 1".to_string()]],
        predicate: None,
        ..Default::default()
    };

    let err = namespace.update_table(request).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Invalid input"),
        "expected Invalid input error, got: {}",
        msg
    );
}

#[tokio::test]
async fn test_update_rejects_duplicate_columns() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = UpdateTableRequest {
        id: Some(table_id),
        updates: vec![
            vec!["name".to_string(), "'a'".to_string()],
            vec!["name".to_string(), "'b'".to_string()],
        ],
        predicate: None,
        ..Default::default()
    };

    let err = namespace.update_table(request).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Invalid input") && msg.contains("more than once"),
        "expected duplicate column InvalidInput error, got: {}",
        msg
    );
}

#[tokio::test]
async fn test_delete_with_predicate() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = DeleteFromTableRequest {
        id: Some(table_id.clone()),
        predicate: "id > 1".to_string(),
        ..Default::default()
    };

    let response = namespace.delete_from_table(request).await.unwrap();
    assert!(response.version.is_some());

    let count_req = CountTableRowsRequest {
        id: Some(table_id),
        version: None,
        predicate: None,
        ..Default::default()
    };
    // Original rows = 3; after deleting `id > 1` only row id=1 remains.
    assert_eq!(namespace.count_table_rows(count_req).await.unwrap(), 1);
}

#[tokio::test]
async fn test_delete_empty_predicate_returns_invalid_input() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    let request = DeleteFromTableRequest {
        id: Some(table_id),
        predicate: "   ".to_string(),
        ..Default::default()
    };

    let err = namespace.delete_from_table(request).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Invalid input") && msg.contains("non-empty predicate"),
        "expected non-empty predicate InvalidInput error, got: {}",
        msg
    );
}

#[tokio::test]
async fn test_delete_table_not_found() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    let request = DeleteFromTableRequest {
        id: Some(vec!["does_not_exist".to_string()]),
        predicate: "id = 1".to_string(),
        ..Default::default()
    };

    let err = namespace.delete_from_table(request).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Table not found"),
        "expected TableNotFound for missing table, got: {}",
        msg
    );
}

#[tokio::test]
async fn test_delete_invalid_predicate_returns_invalid_input() {
    let (namespace, _temp_dir, table_id) = create_ns_with_table().await;

    // A predicate referencing a column that does not exist reaches `Dataset::delete`
    // and surfaces as `Error::InvalidInput`, which must map to `InvalidInput` rather
    // than a generic `Internal`.
    let request = DeleteFromTableRequest {
        id: Some(table_id),
        predicate: "no_such_column = 1".to_string(),
        ..Default::default()
    };

    let err = namespace.delete_from_table(request).await.unwrap_err();
    let lance_core::Error::Namespace { source, .. } = &err else {
        panic!("expected a Namespace error, got: {}", err);
    };
    let ns_err = source
        .downcast_ref::<NamespaceError>()
        .expect("expected a NamespaceError source");
    assert_eq!(
        ns_err.code(),
        lance_namespace::ErrorCode::InvalidInput,
        "expected InvalidInput for an invalid delete predicate, got: {}",
        err
    );
}
