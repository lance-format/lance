// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[tokio::test]
async fn test_register_table() {
    use lance_namespace::models::{RegisterTableRequest, TableExistsRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .dir_listing_to_manifest_migration_enabled(true)
        .build()
        .await
        .unwrap();

    // Create a physical table first using lance directly
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let table_uri = format!("{}/external_table.lance", temp_path);
    let cursor = Cursor::new(ipc_data);
    let stream_reader = StreamReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = stream_reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    let schema = batches[0].schema();
    let batch_results: Vec<_> = batches.into_iter().map(Ok).collect();
    let reader = RecordBatchIterator::new(batch_results, schema);
    Dataset::write(Box::new(reader), &table_uri, None)
        .await
        .unwrap();

    // Register the table
    let mut register_req = RegisterTableRequest::new("external_table.lance".to_string());
    register_req.id = Some(vec!["registered_table".to_string()]);

    let response = namespace.register_table(register_req).await.unwrap();
    assert_eq!(response.location, Some("external_table.lance".to_string()));

    // Verify table exists in namespace
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["registered_table".to_string()]);
    assert!(namespace.table_exists(exists_req).await.is_ok());

    // Verify we can list the table
    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    let tables = namespace.list_tables(list_req).await.unwrap();
    assert!(tables.tables.contains(&"registered_table".to_string()));
}

#[tokio::test]
async fn test_register_table_duplicate_fails() {
    use lance_namespace::models::RegisterTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .build()
        .await
        .unwrap();

    // Register a table
    let mut register_req = RegisterTableRequest::new("test_table.lance".to_string());
    register_req.id = Some(vec!["test_table".to_string()]);

    namespace
        .register_table(register_req.clone())
        .await
        .unwrap();

    // Try to register again - should fail
    let result = namespace.register_table(register_req).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));
}

#[tokio::test]
async fn test_deregister_table() {
    use lance_namespace::models::{DeregisterTableRequest, TableExistsRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace with manifest-only mode (no directory listing fallback)
    // This ensures deregistered tables are truly invisible
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Verify table exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_table".to_string()]);
    assert!(namespace.table_exists(exists_req.clone()).await.is_ok());

    // Deregister the table
    let mut deregister_req = DeregisterTableRequest::new();
    deregister_req.id = Some(vec!["test_table".to_string()]);
    let response = namespace.deregister_table(deregister_req).await.unwrap();

    // Should return location and id
    assert!(
        response.location.is_some(),
        "Deregister should return location"
    );
    let location = response.location.as_ref().unwrap();
    // Location should be a proper file:// URI with the temp path
    // Use uri_to_url to normalize the temp path to a URL for comparison
    let expected_url =
        lance_io::object_store::uri_to_url(temp_path).expect("Failed to convert temp path to URL");
    let expected_prefix = expected_url.to_string();
    assert!(
        location.starts_with(&expected_prefix),
        "Location should start with '{}', got: {}",
        expected_prefix,
        location
    );
    assert!(
        location.contains("test_table"),
        "Location should contain table name: {}",
        location
    );
    assert_eq!(response.id, Some(vec!["test_table".to_string()]));

    // Verify table no longer exists in namespace (removed from manifest)
    assert!(namespace.table_exists(exists_req).await.is_err());

    // Verify physical data still exists at the returned location
    let dataset = Dataset::open(location).await;
    assert!(
        dataset.is_ok(),
        "Physical table data should still exist at {}",
        location
    );
}

#[tokio::test]
async fn test_deregister_table_in_child_namespace() {
    use lance_namespace::models::{
        CreateNamespaceRequest, DeregisterTableRequest, TableExistsRequest,
    };

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .build()
        .await
        .unwrap();

    // Create child namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["test_ns".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Create a table in the child namespace
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_ns".to_string(), "test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Deregister the table
    let mut deregister_req = DeregisterTableRequest::new();
    deregister_req.id = Some(vec!["test_ns".to_string(), "test_table".to_string()]);
    let response = namespace.deregister_table(deregister_req).await.unwrap();

    // Should return location and id in child namespace
    assert!(
        response.location.is_some(),
        "Deregister should return location"
    );
    let location = response.location.as_ref().unwrap();
    // Location should be a proper file:// URI with the temp path
    // Use uri_to_url to normalize the temp path to a URL for comparison
    let expected_url =
        lance_io::object_store::uri_to_url(temp_path).expect("Failed to convert temp path to URL");
    let expected_prefix = expected_url.to_string();
    assert!(
        location.starts_with(&expected_prefix),
        "Location should start with '{}', got: {}",
        expected_prefix,
        location
    );
    assert!(
        location.contains("test_ns") && location.contains("test_table"),
        "Location should contain namespace and table name: {}",
        location
    );
    assert_eq!(
        response.id,
        Some(vec!["test_ns".to_string(), "test_table".to_string()])
    );

    // Verify table no longer exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_ns".to_string(), "test_table".to_string()]);
    assert!(namespace.table_exists(exists_req).await.is_err());
}

#[tokio::test]
async fn test_register_without_manifest_fails() {
    use lance_namespace::models::RegisterTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace without manifest
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();

    // Try to register - should fail (register requires manifest)
    let mut register_req = RegisterTableRequest::new("test_table.lance".to_string());
    register_req.id = Some(vec!["test_table".to_string()]);
    let result = namespace.register_table(register_req).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("manifest mode is enabled")
    );

    // Note: deregister_table now works in V1 mode via .lance-deregistered marker files
    // See test_deregister_table_v1_mode for that test case
}

#[tokio::test]
async fn test_register_table_rejects_absolute_uri() {
    use lance_namespace::models::RegisterTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .build()
        .await
        .unwrap();

    // Try to register with absolute URI - should fail
    let mut register_req = RegisterTableRequest::new("s3://bucket/table.lance".to_string());
    register_req.id = Some(vec!["test_table".to_string()]);
    let result = namespace.register_table(register_req).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Absolute URIs are not allowed"));
}

#[tokio::test]
async fn test_register_table_rejects_absolute_path() {
    use lance_namespace::models::RegisterTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .build()
        .await
        .unwrap();

    // Try to register with absolute path - should fail
    let mut register_req = RegisterTableRequest::new("/tmp/table.lance".to_string());
    register_req.id = Some(vec!["test_table".to_string()]);
    let result = namespace.register_table(register_req).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Absolute paths are not allowed"));
}

#[tokio::test]
async fn test_register_table_rejects_path_traversal() {
    use lance_namespace::models::RegisterTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .build()
        .await
        .unwrap();

    // Try to register with path traversal - should fail
    let mut register_req = RegisterTableRequest::new("../outside/table.lance".to_string());
    register_req.id = Some(vec!["test_table".to_string()]);
    let result = namespace.register_table(register_req).await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Path traversal is not allowed"));
}

#[tokio::test]
async fn test_namespace_write() {
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};
    use lance::dataset::{Dataset, WriteMode, WriteParams};
    use lance_namespace::LanceNamespace;

    let (namespace, _temp_dir) = create_test_namespace().await;
    let namespace = Arc::new(namespace) as Arc<dyn LanceNamespace>;

    // Use child namespace instead of root
    let table_id = vec!["test_ns".to_string(), "test_table".to_string()];
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", DataType::Int32, false),
        ArrowField::new("b", DataType::Int32, false),
    ]));

    // Test 1: CREATE mode
    let data1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(Int32Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();

    let reader1 = RecordBatchIterator::new(vec![data1].into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write_into_namespace(reader1, namespace.clone(), table_id.clone(), None)
        .await
        .unwrap();

    assert_eq!(dataset.count_rows(None).await.unwrap(), 3);
    assert_eq!(dataset.version().version, 1);

    // Test 2: APPEND mode
    let data2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![4, 5])),
            Arc::new(Int32Array::from(vec![40, 50])),
        ],
    )
    .unwrap();

    let params_append = WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    };

    let reader2 = RecordBatchIterator::new(vec![data2].into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write_into_namespace(
        reader2,
        namespace.clone(),
        table_id.clone(),
        Some(params_append),
    )
    .await
    .unwrap();

    assert_eq!(dataset.count_rows(None).await.unwrap(), 5);
    assert_eq!(dataset.version().version, 2);

    // Test 3: OVERWRITE mode
    let data3 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![100, 200])),
            Arc::new(Int32Array::from(vec![1000, 2000])),
        ],
    )
    .unwrap();

    let params_overwrite = WriteParams {
        mode: WriteMode::Overwrite,
        ..Default::default()
    };

    let reader3 = RecordBatchIterator::new(vec![data3].into_iter().map(Ok), schema.clone());
    let dataset = Dataset::write_into_namespace(
        reader3,
        namespace.clone(),
        table_id.clone(),
        Some(params_overwrite),
    )
    .await
    .unwrap();

    assert_eq!(dataset.count_rows(None).await.unwrap(), 2);
    assert_eq!(dataset.version().version, 3);

    // Verify old data was replaced
    let result = dataset.scan().try_into_batch().await.unwrap();
    let a_col = result
        .column_by_name("a")
        .unwrap()
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(a_col.values(), &[100, 200]);
}

// ============================================================
// Tests for declare_table
// ============================================================

#[tokio::test]
async fn test_declare_table_v1_mode() {
    use lance_namespace::models::{
        DeclareTableRequest, DescribeTableRequest, ListTablesRequest, TableExistsRequest,
    };

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace in V1 mode (no manifest)
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();

    // Declare a table
    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    let response = namespace.declare_table(declare_req).await.unwrap();

    // Should return location
    assert!(response.location.is_some());
    let location = response.location.as_ref().unwrap();
    assert!(location.ends_with("test_table.lance"));

    // Table should exist (via reserved file)
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_table".to_string()]);
    assert!(namespace.table_exists(exists_req).await.is_ok());

    // Describe should work but return no version/schema (not written yet)
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert!(describe_response.location.is_some());
    assert!(describe_response.version.is_none()); // Not written yet
    assert!(describe_response.schema.is_none()); // Not written yet
    assert_eq!(describe_response.is_only_declared, None);

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    describe_req.check_declared = Some(true);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(describe_response.is_only_declared, Some(true));

    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    let list_response = namespace.list_tables(list_req.clone()).await.unwrap();
    assert_eq!(list_response.tables, vec!["test_table".to_string()]);

    list_req.include_declared = Some(false);
    let list_response = namespace.list_tables(list_req).await.unwrap();
    assert!(list_response.tables.is_empty());
}

#[tokio::test]
async fn test_insert_into_declared_table_promotes_it_from_declared_state() {
    use lance_namespace::models::{
        DeclareTableRequest, DescribeTableRequest, InsertIntoTableRequest,
    };

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    namespace.declare_table(declare_req).await.unwrap();

    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut insert_req = InsertIntoTableRequest::new();
    insert_req.id = Some(vec!["test_table".to_string()]);
    let response = namespace
        .insert_into_table(insert_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();
    assert_eq!(response.num_inserted_rows, Some(0));
    assert_eq!(response.version, Some(1));

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    describe_req.load_detailed_metadata = Some(true);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();

    assert_eq!(describe_response.is_only_declared, Some(false));
    assert_eq!(describe_response.version, Some(1));
    assert!(describe_response.schema.is_some());

    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    list_req.include_declared = Some(false);
    assert_eq!(
        namespace.list_tables(list_req).await.unwrap().tables,
        vec!["test_table".to_string()]
    );
}

#[tokio::test]
async fn test_create_table_after_declare_table_v1_mode_creates_table() {
    use lance_namespace::models::{DeclareTableRequest, DescribeTableRequest, ListTablesRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    namespace.declare_table(declare_req).await.unwrap();

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    let response = namespace
        .create_table(
            create_req,
            bytes::Bytes::from(create_non_empty_test_ipc_data()),
        )
        .await
        .unwrap();

    assert_eq!(response.version, Some(1));

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    describe_req.load_detailed_metadata = Some(true);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(describe_response.is_only_declared, Some(false));
    assert_eq!(describe_response.version, Some(1));

    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    list_req.include_declared = Some(false);
    assert_eq!(
        namespace.list_tables(list_req).await.unwrap().tables,
        vec!["test_table".to_string()]
    );
}

#[tokio::test]
async fn test_insert_into_declared_table_with_manifest_promotes_it() {
    use lance_namespace::models::{
        DeclareTableRequest, DescribeTableRequest, InsertIntoTableRequest, ListTablesRequest,
    };

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    namespace.declare_table(declare_req).await.unwrap();

    let mut insert_req = InsertIntoTableRequest::new();
    insert_req.id = Some(vec!["test_table".to_string()]);
    let response = namespace
        .insert_into_table(
            insert_req,
            bytes::Bytes::from(create_non_empty_test_ipc_data()),
        )
        .await
        .unwrap();
    assert_eq!(response.num_inserted_rows, Some(2));
    assert_eq!(response.version, Some(1));

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    describe_req.load_detailed_metadata = Some(true);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(describe_response.is_only_declared, Some(false));
    assert_eq!(describe_response.version, Some(1));

    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    list_req.include_declared = Some(false);
    assert_eq!(
        namespace.list_tables(list_req).await.unwrap().tables,
        vec!["test_table".to_string()]
    );
}

#[tokio::test]
async fn test_create_table_after_declare_table_with_manifest_creates_table() {
    use lance_namespace::models::{
        CreateTableRequest, DeclareTableRequest, DescribeTableRequest, ListTablesRequest,
    };

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    declare_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
    namespace.declare_table(declare_req).await.unwrap();

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    create_req.mode = Some("Overwrite".to_string());
    let response = namespace
        .create_table(
            create_req,
            bytes::Bytes::from(create_non_empty_test_ipc_data()),
        )
        .await
        .unwrap();

    assert_eq!(response.version, Some(1));
    assert_eq!(
        response
            .properties
            .as_ref()
            .and_then(|properties| properties.get("owner")),
        Some(&"alice".to_string())
    );

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    describe_req.load_detailed_metadata = Some(true);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(describe_response.is_only_declared, Some(false));
    assert_eq!(describe_response.version, Some(1));
    assert_eq!(
        describe_response
            .properties
            .as_ref()
            .and_then(|properties| properties.get("owner")),
        Some(&"alice".to_string())
    );

    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    list_req.include_declared = Some(false);
    assert_eq!(
        namespace.list_tables(list_req).await.unwrap().tables,
        vec!["test_table".to_string()]
    );
}

#[tokio::test]
async fn test_create_table_after_declare_table_with_manifest_rejects_new_properties() {
    use lance_namespace::models::{CreateTableRequest, DeclareTableRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    declare_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
    namespace.declare_table(declare_req).await.unwrap();

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    create_req.properties = Some(HashMap::from([("owner".to_string(), "bob".to_string())]));

    let result = namespace
        .create_table(
            create_req,
            bytes::Bytes::from(create_non_empty_test_ipc_data()),
        )
        .await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("cannot set properties for already declared table")
    );
}

#[tokio::test]
async fn test_create_table_with_manifest_exist_ok_keeps_existing_table() {
    use lance_namespace::models::{CreateTableRequest, DescribeTableRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    create_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
    namespace
        .create_table(
            create_req,
            bytes::Bytes::from(create_non_empty_test_ipc_data()),
        )
        .await
        .unwrap();

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    create_req.mode = Some("ExistOk".to_string());
    create_req.properties = Some(HashMap::from([("owner".to_string(), "bob".to_string())]));
    let response = namespace
        .create_table(
            create_req,
            bytes::Bytes::from(create_single_row_test_ipc_data()),
        )
        .await
        .unwrap();

    assert_eq!(
        response
            .properties
            .as_ref()
            .and_then(|properties| properties.get("owner")),
        Some(&"alice".to_string())
    );
    assert_eq!(
        open_dataset(&namespace, "test_table")
            .await
            .count_rows(None)
            .await
            .unwrap(),
        2
    );

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(
        describe_response
            .properties
            .as_ref()
            .and_then(|properties| properties.get("owner")),
        Some(&"alice".to_string())
    );
}

#[tokio::test]
async fn test_create_table_with_manifest_overwrite_replaces_existing_table() {
    use lance_namespace::models::{CreateTableRequest, DescribeTableRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    create_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
    namespace
        .create_table(
            create_req,
            bytes::Bytes::from(create_non_empty_test_ipc_data()),
        )
        .await
        .unwrap();

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    create_req.mode = Some("overwrite".to_string());
    create_req.properties = Some(HashMap::from([("owner".to_string(), "bob".to_string())]));
    let response = namespace
        .create_table(
            create_req,
            bytes::Bytes::from(create_single_row_test_ipc_data()),
        )
        .await
        .unwrap();

    assert_eq!(response.version, Some(2));
    assert_eq!(
        response
            .properties
            .as_ref()
            .and_then(|properties| properties.get("owner")),
        Some(&"bob".to_string())
    );
    assert_eq!(
        open_dataset(&namespace, "test_table")
            .await
            .count_rows(None)
            .await
            .unwrap(),
        1
    );

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(
        describe_response
            .properties
            .as_ref()
            .and_then(|properties| properties.get("owner")),
        Some(&"bob".to_string())
    );
}

#[tokio::test]
async fn test_create_table_with_manifest_invalid_mode_rejected() {
    use lance_namespace::models::CreateTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    create_req.mode = Some("append".to_string());
    let result = namespace
        .create_table(
            create_req,
            bytes::Bytes::from(create_non_empty_test_ipc_data()),
        )
        .await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Unsupported create_table mode")
    );
}

#[tokio::test]
async fn test_merge_insert_into_declared_table_v1_mode_creates_table() {
    use lance_namespace::models::{
        DeclareTableRequest, DescribeTableRequest, ListTablesRequest, MergeInsertIntoTableRequest,
    };

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    namespace.declare_table(declare_req).await.unwrap();

    let mut merge_req = MergeInsertIntoTableRequest::new();
    merge_req.id = Some(vec!["test_table".to_string()]);
    merge_req.on = Some(vec!["id".to_string()]);
    let response = namespace
        .merge_insert_into_table(
            merge_req,
            bytes::Bytes::from(create_non_empty_test_ipc_data()),
        )
        .await
        .unwrap();

    assert_eq!(response.num_inserted_rows, Some(2));
    assert_eq!(response.num_updated_rows, Some(0));

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    describe_req.load_detailed_metadata = Some(true);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(describe_response.is_only_declared, Some(false));
    assert_eq!(describe_response.version, Some(1));

    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    list_req.include_declared = Some(false);
    assert_eq!(
        namespace.list_tables(list_req).await.unwrap().tables,
        vec!["test_table".to_string()]
    );
}

#[tokio::test]
async fn test_merge_insert_into_declared_table_with_manifest_creates_table() {
    use lance_namespace::models::{
        DeclareTableRequest, DescribeTableRequest, ListTablesRequest, MergeInsertIntoTableRequest,
    };

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    namespace.declare_table(declare_req).await.unwrap();

    let mut merge_req = MergeInsertIntoTableRequest::new();
    merge_req.id = Some(vec!["test_table".to_string()]);
    merge_req.on = Some(vec!["id".to_string()]);
    let response = namespace
        .merge_insert_into_table(
            merge_req,
            bytes::Bytes::from(create_non_empty_test_ipc_data()),
        )
        .await
        .unwrap();

    assert_eq!(response.num_inserted_rows, Some(2));
    assert_eq!(response.num_updated_rows, Some(0));

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    describe_req.load_detailed_metadata = Some(true);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(describe_response.is_only_declared, Some(false));
    assert_eq!(describe_response.version, Some(1));

    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    list_req.include_declared = Some(false);
    assert_eq!(
        namespace.list_tables(list_req).await.unwrap().tables,
        vec!["test_table".to_string()]
    );
}

/// `(region, id, value)` rows, for merge inserts keyed on `region` + `id`.
///
/// `region` is nullable so tests can cover a NULL in one half of the key.
fn create_composite_key_ipc_data(rows: &[(Option<&str>, i32, &str)]) -> Vec<u8> {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("region", DataType::Utf8, true),
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from_iter(
                rows.iter().map(|(region, _, _)| *region),
            )),
            Arc::new(Int32Array::from_iter_values(
                rows.iter().map(|(_, id, _)| *id),
            )),
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|(_, _, value)| *value),
            )),
        ],
    )
    .unwrap();
    create_ipc_data_from_batches(schema, vec![batch])
}

/// `test_table`'s rows as `(region, id, value)`, sorted for a stable comparison.
async fn read_composite_key_rows(root: &str) -> Vec<(Option<String>, i32, String)> {
    use arrow::array::Array;

    let dataset = Dataset::open(&format!("{}/test_table.lance", root))
        .await
        .unwrap();
    let batch = dataset.scan().try_into_batch().await.unwrap();
    let regions = batch["region"]
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap();
    let ids = batch["id"]
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap();
    let values = batch["value"]
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .unwrap();

    let mut rows: Vec<_> = (0..batch.num_rows())
        .map(|row| {
            (
                regions
                    .is_valid(row)
                    .then(|| regions.value(row).to_string()),
                ids.value(row),
                values.value(row).to_string(),
            )
        })
        .collect();
    rows.sort_unstable();
    rows
}

#[tokio::test]
async fn test_merge_insert_matches_on_every_column_of_a_composite_key() {
    use lance_namespace::models::{DeclareTableRequest, MergeInsertIntoTableRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    namespace.declare_table(declare_req).await.unwrap();

    let seed = create_composite_key_ipc_data(&[
        (Some("us"), 1, "a"),
        (Some("us"), 2, "b"),
        (Some("eu"), 1, "c"),
    ]);
    let mut merge_req = MergeInsertIntoTableRequest::new();
    merge_req.id = Some(vec!["test_table".to_string()]);
    merge_req.on = Some(vec!["region".to_string(), "id".to_string()]);
    namespace
        .merge_insert_into_table(merge_req, bytes::Bytes::from(seed))
        .await
        .unwrap();

    // ("us", 1) matches an existing row; ("eu", 2) matches nothing even though a row
    // with region "eu" and a row with id 2 both exist.
    let mut merge_req = MergeInsertIntoTableRequest::new();
    merge_req.id = Some(vec!["test_table".to_string()]);
    merge_req.on = Some(vec!["region".to_string(), "id".to_string()]);
    merge_req.when_matched_update_all = Some(true);
    let response = namespace
        .merge_insert_into_table(
            merge_req,
            bytes::Bytes::from(create_composite_key_ipc_data(&[
                (Some("us"), 1, "updated"),
                (Some("eu"), 2, "inserted"),
            ])),
        )
        .await
        .unwrap();

    assert_eq!(response.num_updated_rows, Some(1));
    assert_eq!(response.num_inserted_rows, Some(1));

    assert_eq!(
        read_composite_key_rows(temp_path).await,
        vec![
            // ("eu", 1) keeps its value: matching on `id` alone would have clobbered it.
            (Some("eu".into()), 1, "c".into()),
            (Some("eu".into()), 2, "inserted".into()),
            (Some("us".into()), 1, "updated".into()),
            (Some("us".into()), 2, "b".into()),
        ]
    );
}

/// Core switches NULL join semantics on the arity of the match key
/// (`merge_insert.rs`, `NullEquality`): a single-column key treats NULL as equal to
/// NULL, while a composite key uses standard SQL equality, under which it is not. So
/// adding a second key column changes whether NULL-keyed rows match at all.
#[tokio::test]
async fn test_merge_insert_composite_key_never_matches_a_null_key_column() {
    use lance_namespace::models::{DeclareTableRequest, MergeInsertIntoTableRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    namespace.declare_table(declare_req).await.unwrap();

    let mut merge_req = MergeInsertIntoTableRequest::new();
    merge_req.id = Some(vec!["test_table".to_string()]);
    merge_req.on = Some(vec!["region".to_string(), "id".to_string()]);
    namespace
        .merge_insert_into_table(
            merge_req,
            bytes::Bytes::from(create_composite_key_ipc_data(&[
                (None, 1, "seeded"),
                (Some("us"), 1, "us-seeded"),
            ])),
        )
        .await
        .unwrap();

    let mut merge_req = MergeInsertIntoTableRequest::new();
    merge_req.id = Some(vec!["test_table".to_string()]);
    merge_req.on = Some(vec!["region".to_string(), "id".to_string()]);
    merge_req.when_matched_update_all = Some(true);
    let response = namespace
        .merge_insert_into_table(
            merge_req,
            bytes::Bytes::from(create_composite_key_ipc_data(&[(None, 1, "not-a-match")])),
        )
        .await
        .unwrap();

    // The incoming row is byte-identical to the seeded one, and still does not match.
    assert_eq!(response.num_updated_rows, Some(0));
    assert_eq!(response.num_inserted_rows, Some(1));

    assert_eq!(
        read_composite_key_rows(temp_path).await,
        vec![
            (None, 1, "not-a-match".into()),
            (None, 1, "seeded".into()),
            (Some("us".into()), 1, "us-seeded".into()),
        ]
    );
}

#[rstest::rstest]
#[case::missing(None, "'on' field is required")]
#[case::empty(Some(vec![]), "must name at least one column")]
#[case::duplicate(
    Some(vec!["region".to_string(), "region".to_string()]),
    "names column 'region' more than once"
)]
#[tokio::test]
async fn test_merge_insert_rejects_an_invalid_on_key(
    #[case] on: Option<Vec<String>>,
    #[case] expected_message: &str,
) {
    use lance_namespace::models::{DeclareTableRequest, MergeInsertIntoTableRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    namespace.declare_table(declare_req).await.unwrap();

    let mut merge_req = MergeInsertIntoTableRequest::new();
    merge_req.id = Some(vec!["test_table".to_string()]);
    merge_req.on = on;
    let error = namespace
        .merge_insert_into_table(
            merge_req,
            bytes::Bytes::from(create_composite_key_ipc_data(&[(Some("us"), 1, "a")])),
        )
        .await
        .unwrap_err();

    let lance_core::Error::Namespace { source, .. } = &error else {
        panic!("expected a Namespace error, got: {}", error);
    };
    let ns_err = source
        .downcast_ref::<NamespaceError>()
        .expect("expected a NamespaceError source");
    assert_eq!(ns_err.code(), lance_namespace::ErrorCode::InvalidInput);
    assert!(
        error.to_string().contains(expected_message),
        "unexpected error message: {error}"
    );
}

#[tokio::test]
async fn test_declare_table_with_manifest() {
    use lance_namespace::models::{
        DeclareTableRequest, DescribeTableRequest, ListTablesRequest, TableExistsRequest,
    };

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace with manifest
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    // Declare a table
    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    declare_req.properties = Some(HashMap::from([("owner".to_string(), "alice".to_string())]));
    let response = namespace.declare_table(declare_req).await.unwrap();

    // Should return location
    assert!(response.location.is_some());
    assert_eq!(
        response
            .properties
            .as_ref()
            .and_then(|properties| properties.get("owner")),
        Some(&"alice".to_string())
    );

    // Table should exist in manifest
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_table".to_string()]);
    assert!(namespace.table_exists(exists_req).await.is_ok());

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(describe_response.is_only_declared, None);

    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    describe_req.check_declared = Some(true);
    let describe_response = namespace.describe_table(describe_req).await.unwrap();
    assert_eq!(describe_response.is_only_declared, Some(true));
    assert_eq!(
        describe_response
            .properties
            .as_ref()
            .and_then(|properties| properties.get("owner")),
        Some(&"alice".to_string())
    );

    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    assert_eq!(
        namespace
            .list_tables(list_req.clone())
            .await
            .unwrap()
            .tables,
        vec!["test_table".to_string()]
    );
    list_req.include_declared = Some(false);
    assert!(
        namespace
            .list_tables(list_req)
            .await
            .unwrap()
            .tables
            .is_empty()
    );
}

#[tokio::test]
async fn test_declare_table_with_manifest_marker_already_exists() {
    // Pre-existing .lance-reserved (concurrent/incomplete declare) must map to
    // TableAlreadyExists, not Internal.
    use lance_namespace::error::ErrorCode;
    use lance_namespace::models::DeclareTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    let table_dir = temp_dir.join("test_table.lance");
    std::fs::create_dir_all(&table_dir).unwrap();
    std::fs::write(table_dir.join(".lance-reserved"), b"reserved").unwrap();

    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    let err = namespace
        .declare_table(declare_req)
        .await
        .expect_err("declare with existing marker must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("already exists") || msg.contains("TableAlreadyExists"),
        "expected TableAlreadyExists, got: {msg}"
    );
    assert_eq!(
        mutation_error_code(err),
        ErrorCode::TableAlreadyExists,
        "expected TableAlreadyExists error code"
    );
}

#[tokio::test]
async fn test_declare_table_when_table_exists() {
    use lance_namespace::models::DeclareTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();

    // First create a table with actual data
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Try to declare the same table - should fail because it already has data
    let mut declare_req = DeclareTableRequest::new();
    declare_req.id = Some(vec!["test_table".to_string()]);
    let result = namespace.declare_table(declare_req).await;
    assert!(result.is_err());
}

// ============================================================
// Tests for deregister_table in V1 mode
// ============================================================

#[tokio::test]
async fn test_deregister_table_v1_mode() {
    use lance_namespace::models::{DeregisterTableRequest, TableExistsRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace in V1 mode (no manifest, with dir listing)
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    // Create a table with data
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Verify table exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_table".to_string()]);
    assert!(namespace.table_exists(exists_req.clone()).await.is_ok());

    // Deregister the table
    let mut deregister_req = DeregisterTableRequest::new();
    deregister_req.id = Some(vec!["test_table".to_string()]);
    let response = namespace.deregister_table(deregister_req).await.unwrap();

    // Should return location
    assert!(response.location.is_some());
    let location = response.location.as_ref().unwrap();
    assert!(location.contains("test_table"));

    // Table should no longer exist (deregistered)
    let result = namespace.table_exists(exists_req).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("deregistered"));

    // Physical data should still exist
    let dataset = Dataset::open(location).await;
    assert!(dataset.is_ok(), "Physical table data should still exist");
}

#[tokio::test]
async fn test_deregister_table_v1_already_deregistered() {
    use lance_namespace::models::DeregisterTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Deregister once
    let mut deregister_req = DeregisterTableRequest::new();
    deregister_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .deregister_table(deregister_req.clone())
        .await
        .unwrap();

    // Try to deregister again - should fail
    let result = namespace.deregister_table(deregister_req).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("already deregistered")
    );
}

// ============================================================
// Tests for list_tables skipping deregistered tables
// ============================================================

#[tokio::test]
async fn test_list_tables_skips_deregistered_v1() {
    use lance_namespace::models::DeregisterTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    // Create two tables
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut create_req1 = CreateTableRequest::new();
    create_req1.id = Some(vec!["table1".to_string()]);
    namespace
        .create_table(create_req1, bytes::Bytes::from(ipc_data.clone()))
        .await
        .unwrap();

    let mut create_req2 = CreateTableRequest::new();
    create_req2.id = Some(vec!["table2".to_string()]);
    namespace
        .create_table(create_req2, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // List tables - should see both (root namespace = empty vec)
    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    let list_response = namespace.list_tables(list_req.clone()).await.unwrap();
    assert_eq!(list_response.tables.len(), 2);

    // Deregister table1
    let mut deregister_req = DeregisterTableRequest::new();
    deregister_req.id = Some(vec!["table1".to_string()]);
    namespace.deregister_table(deregister_req).await.unwrap();

    // List tables - should only see table2
    let list_response = namespace.list_tables(list_req).await.unwrap();
    assert_eq!(list_response.tables.len(), 1);
    assert!(list_response.tables.contains(&"table2".to_string()));
    assert!(!list_response.tables.contains(&"table1".to_string()));
}

// ============================================================
// Tests for describe_table and table_exists with deregistered tables
// ============================================================

#[tokio::test]
async fn test_describe_table_fails_for_deregistered_v1() {
    use lance_namespace::models::{DeregisterTableRequest, DescribeTableRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Describe should work before deregistration
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    assert!(namespace.describe_table(describe_req.clone()).await.is_ok());

    // Deregister
    let mut deregister_req = DeregisterTableRequest::new();
    deregister_req.id = Some(vec!["test_table".to_string()]);
    namespace.deregister_table(deregister_req).await.unwrap();

    // Describe should fail after deregistration
    let result = namespace.describe_table(describe_req).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, Error::Namespace { .. }));
    let err_msg = err.to_string();
    assert!(err_msg.contains("deregistered"));
    assert!(err_msg.contains("table id 'test_table'"));
}

#[tokio::test]
async fn test_table_exists_fails_for_deregistered_v1() {
    use lance_namespace::models::{DeregisterTableRequest, TableExistsRequest};

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Table exists should work before deregistration
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_table".to_string()]);
    assert!(namespace.table_exists(exists_req.clone()).await.is_ok());

    // Deregister
    let mut deregister_req = DeregisterTableRequest::new();
    deregister_req.id = Some(vec!["test_table".to_string()]);
    namespace.deregister_table(deregister_req).await.unwrap();

    // Table exists should fail after deregistration
    let result = namespace.table_exists(exists_req).await;
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(matches!(err, Error::Namespace { .. }));
    let err_msg = err.to_string();
    assert!(err_msg.contains("deregistered"));
    assert!(err_msg.contains("table id 'test_table'"));
}

#[tokio::test]
async fn test_atomic_table_status_check() {
    // This test verifies that the TableStatus check is atomic
    // by ensuring a single directory listing is used

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Table status should show exists=true, is_deregistered=false
    let status = namespace.check_table_status("test_table").await.unwrap();
    assert!(status.exists);
    assert!(!status.is_deregistered);
    assert!(!status.has_reserved_file);
}
