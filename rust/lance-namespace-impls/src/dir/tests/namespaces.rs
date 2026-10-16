// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[tokio::test]
async fn test_root_namespace_operations() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Test list_namespaces - should return empty list for root
    let mut request = ListNamespacesRequest::new();
    request.id = Some(vec![]);
    let result = namespace.list_namespaces(request).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().namespaces.len(), 0);

    // Test describe_namespace - should succeed for root
    let mut request = DescribeNamespaceRequest::new();
    request.id = Some(vec![]);
    let result = namespace.describe_namespace(request).await;
    assert!(result.is_ok());

    // Test namespace_exists - root always exists
    let mut request = NamespaceExistsRequest::new();
    request.id = Some(vec![]);
    let result = namespace.namespace_exists(request).await;
    assert!(result.is_ok());

    // Test create_namespace - root cannot be created
    let mut request = CreateNamespaceRequest::new();
    request.id = Some(vec![]);
    let result = namespace.create_namespace(request).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("already exists"));

    // Test drop_namespace - root cannot be dropped
    let mut request = DropNamespaceRequest::new();
    request.id = Some(vec![]);
    let result = namespace.drop_namespace(request).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("cannot be dropped")
    );
}

#[tokio::test]
async fn test_non_root_namespace_operations() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // With manifest enabled (default), child namespaces are now supported
    // Test create_namespace for non-root - should succeed with manifest
    let mut request = CreateNamespaceRequest::new();
    request.id = Some(vec!["child".to_string()]);
    let result = namespace.create_namespace(request).await;
    assert!(
        result.is_ok(),
        "Child namespace creation should succeed with manifest enabled"
    );

    // Test namespace_exists for non-root - should exist after creation
    let mut request = NamespaceExistsRequest::new();
    request.id = Some(vec!["child".to_string()]);
    let result = namespace.namespace_exists(request).await;
    assert!(
        result.is_ok(),
        "Child namespace should exist after creation"
    );

    // Test drop_namespace for non-root - should succeed
    let mut request = DropNamespaceRequest::new();
    request.id = Some(vec!["child".to_string()]);
    let result = namespace.drop_namespace(request).await;
    assert!(
        result.is_ok(),
        "Child namespace drop should succeed with manifest enabled"
    );

    // Verify namespace no longer exists
    let mut request = NamespaceExistsRequest::new();
    request.id = Some(vec!["child".to_string()]);
    let result = namespace.namespace_exists(request).await;
    assert!(
        result.is_err(),
        "Child namespace should not exist after drop"
    );
}

#[tokio::test]
async fn test_config_custom_root() {
    let temp_dir = TempStdDir::default();
    let custom_path = temp_dir.join("custom");
    std::fs::create_dir(&custom_path).unwrap();

    let namespace = DirectoryNamespaceBuilder::new(custom_path.to_string_lossy().to_string())
        .build()
        .await
        .unwrap();

    // Create test IPC data
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    // Create a table and verify location
    let mut request = CreateTableRequest::new();
    request.id = Some(vec!["test_table".to_string()]);

    let response = namespace
        .create_table(request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    assert!(response.location.unwrap().contains("custom"));
}

#[tokio::test]
async fn test_config_storage_options() {
    let temp_dir = TempStdDir::default();

    let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
        .storage_option("option1", "value1")
        .storage_option("option2", "value2")
        .build()
        .await
        .unwrap();

    // Create test IPC data
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    // Create a table and check storage options are included
    let mut request = CreateTableRequest::new();
    request.id = Some(vec!["test_table".to_string()]);

    let response = namespace
        .create_table(request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    let storage_options = response.storage_options.unwrap();
    assert_eq!(storage_options.get("option1"), Some(&"value1".to_string()));
    assert_eq!(storage_options.get("option2"), Some(&"value2".to_string()));
}

/// When no credential vendor is configured, `describe_table` and
/// `declare_table` must strip credential keys from storage options
/// while preserving non-credential config (region, endpoint, etc.).
#[tokio::test]
async fn test_no_storage_options_without_vendor() {
    use lance_namespace::models::DeclareTableRequest;

    let temp_dir = TempStdDir::default();

    // No manifest, no credential vendor, but storage options with credentials
    let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
        .manifest_enabled(false)
        .storage_option("aws_access_key_id", "AKID")
        .storage_option("aws_secret_access_key", "SECRET")
        .storage_option("region", "us-east-1")
        .build()
        .await
        .unwrap();

    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    // create_table
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["t1".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // describe_table should not return storage options without a vendor
    let mut desc_req = DescribeTableRequest::new();
    desc_req.id = Some(vec!["t1".to_string()]);
    let resp = namespace.describe_table(desc_req).await.unwrap();
    assert!(resp.storage_options.is_none());

    // declare_table should not return storage options without a vendor
    let mut decl_req = DeclareTableRequest::new();
    decl_req.id = Some(vec!["t2".to_string()]);
    let resp = namespace.declare_table(decl_req).await.unwrap();
    assert!(resp.storage_options.is_none());
}

/// Same test with manifest mode enabled.
#[tokio::test]
async fn test_no_storage_options_without_vendor_manifest() {
    let temp_dir = TempStdDir::default();

    let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
        .storage_option("aws_access_key_id", "AKID")
        .storage_option("aws_secret_access_key", "SECRET")
        .storage_option("region", "us-east-1")
        .build()
        .await
        .unwrap();

    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["t1".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // describe_table through manifest should not return storage options without a vendor
    let mut desc_req = DescribeTableRequest::new();
    desc_req.id = Some(vec!["t1".to_string()]);
    let resp = namespace.describe_table(desc_req).await.unwrap();
    assert!(resp.storage_options.is_none());
}

#[tokio::test]
async fn test_from_properties_manifest_enabled() {
    let temp_dir = TempStdDir::default();

    let mut properties = HashMap::new();
    properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());
    properties.insert("manifest_enabled".to_string(), "true".to_string());
    properties.insert("dir_listing_enabled".to_string(), "false".to_string());

    let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
    assert!(builder.manifest_enabled);
    assert!(!builder.dir_listing_enabled);

    let namespace = builder.build().await.unwrap();

    // Create test IPC data
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    // Create a table
    let mut request = CreateTableRequest::new();
    request.id = Some(vec!["test_table".to_string()]);

    let response = namespace
        .create_table(request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    assert!(response.location.is_some());
}

#[tokio::test]
async fn test_from_properties_dir_listing_enabled() {
    let temp_dir = TempStdDir::default();

    let mut properties = HashMap::new();
    properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());
    properties.insert("manifest_enabled".to_string(), "false".to_string());
    properties.insert("dir_listing_enabled".to_string(), "true".to_string());

    let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
    assert!(!builder.manifest_enabled);
    assert!(builder.dir_listing_enabled);

    let namespace = builder.build().await.unwrap();

    // Create test IPC data
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    // Create a table
    let mut request = CreateTableRequest::new();
    request.id = Some(vec!["test_table".to_string()]);

    let response = namespace
        .create_table(request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    assert!(response.location.is_some());
}

#[tokio::test]
async fn test_from_properties_defaults() {
    let temp_dir = TempStdDir::default();

    let mut properties = HashMap::new();
    properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());

    let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
    assert!(builder.manifest_enabled);
    assert!(builder.dir_listing_enabled);
    assert!(!builder.inline_optimization_enabled);
}

#[test]
fn test_builder_disables_inline_optimization_by_default() {
    let builder = DirectoryNamespaceBuilder::new("memory://");
    assert!(!builder.inline_optimization_enabled);
}

#[tokio::test]
async fn test_from_properties_with_storage_options() {
    let temp_dir = TempStdDir::default();

    let mut properties = HashMap::new();
    properties.insert("root".to_string(), temp_dir.to_str().unwrap().to_string());
    properties.insert("manifest_enabled".to_string(), "true".to_string());
    properties.insert("storage.region".to_string(), "us-west-2".to_string());
    properties.insert("storage.bucket".to_string(), "my-bucket".to_string());

    let builder = DirectoryNamespaceBuilder::from_properties(properties, None).unwrap();
    assert!(builder.manifest_enabled);
    assert!(builder.storage_options.is_some());

    let storage_options = builder.storage_options.unwrap();
    assert_eq!(
        storage_options.get("region"),
        Some(&"us-west-2".to_string())
    );
    assert_eq!(
        storage_options.get("bucket"),
        Some(&"my-bucket".to_string())
    );
}

#[tokio::test]
async fn test_various_arrow_types() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create schema with various types
    let fields = vec![
        JsonArrowField {
            name: "bool_col".to_string(),
            r#type: Box::new(JsonArrowDataType::new("bool".to_string())),
            nullable: true,
            metadata: None,
        },
        JsonArrowField {
            name: "int8_col".to_string(),
            r#type: Box::new(JsonArrowDataType::new("int8".to_string())),
            nullable: true,
            metadata: None,
        },
        JsonArrowField {
            name: "float64_col".to_string(),
            r#type: Box::new(JsonArrowDataType::new("float64".to_string())),
            nullable: true,
            metadata: None,
        },
        JsonArrowField {
            name: "binary_col".to_string(),
            r#type: Box::new(JsonArrowDataType::new("binary".to_string())),
            nullable: true,
            metadata: None,
        },
    ];

    let schema = JsonArrowSchema {
        fields,
        metadata: None,
    };

    // Create IPC data
    let ipc_data = create_test_ipc_data(&schema);

    let mut request = CreateTableRequest::new();
    request.id = Some(vec!["complex_table".to_string()]);

    let response = namespace
        .create_table(request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    assert!(response.location.is_some());
}

#[tokio::test]
async fn test_connect_dir() {
    let temp_dir = TempStdDir::default();

    let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
        .build()
        .await
        .unwrap();

    // Test basic operation through the concrete type
    let mut request = ListTablesRequest::new();
    request.id = Some(vec![]);
    let response = namespace.list_tables(request).await.unwrap();
    assert_eq!(response.tables.len(), 0);
}

#[tokio::test]
async fn test_create_table_with_ipc_data() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::ipc::writer::StreamWriter;

    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create a schema with some fields
    let schema = create_test_schema();

    // Create some test data that matches the schema
    let arrow_schema = convert_json_arrow_schema(&schema).unwrap();
    let arrow_schema = Arc::new(arrow_schema);

    // Create a RecordBatch with actual data
    let id_array = Int32Array::from(vec![1, 2, 3]);
    let name_array = StringArray::from(vec!["Alice", "Bob", "Charlie"]);
    let batch = arrow::record_batch::RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(id_array), Arc::new(name_array)],
    )
    .unwrap();

    // Write the batch to an IPC stream
    let mut buffer = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buffer, &arrow_schema).unwrap();
        writer.write(&batch).unwrap();
        writer.finish().unwrap();
    }

    // Create table with the IPC data
    let mut request = CreateTableRequest::new();
    request.id = Some(vec!["test_table_with_data".to_string()]);

    let response = namespace
        .create_table(request, Bytes::from(buffer))
        .await
        .unwrap();

    assert_eq!(response.version, Some(1));
    assert!(
        response
            .location
            .unwrap()
            .contains("test_table_with_data.lance")
    );

    // Verify table exists
    let mut exists_request = TableExistsRequest::new();
    exists_request.id = Some(vec!["test_table_with_data".to_string()]);
    namespace.table_exists(exists_request).await.unwrap();
}

#[tokio::test]
async fn test_child_namespace_create_and_list() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create multiple child namespaces
    for i in 1..=3 {
        let mut create_req = CreateNamespaceRequest::new();
        create_req.id = Some(vec![format!("ns{}", i)]);
        let result = namespace.create_namespace(create_req).await;
        assert!(result.is_ok(), "Failed to create child namespace ns{}", i);
    }

    // List child namespaces
    let list_req = ListNamespacesRequest {
        id: Some(vec![]),
        ..Default::default()
    };
    let result = namespace.list_namespaces(list_req).await;
    assert!(result.is_ok());
    let namespaces = result.unwrap().namespaces;
    assert_eq!(namespaces.len(), 3);
    assert!(namespaces.contains(&"ns1".to_string()));
    assert!(namespaces.contains(&"ns2".to_string()));
    assert!(namespaces.contains(&"ns3".to_string()));
}

#[tokio::test]
async fn test_nested_namespace_hierarchy() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create parent namespace
    let mut create_req = CreateNamespaceRequest::new();
    create_req.id = Some(vec!["parent".to_string()]);
    namespace.create_namespace(create_req).await.unwrap();

    // Create nested children
    let mut create_req = CreateNamespaceRequest::new();
    create_req.id = Some(vec!["parent".to_string(), "child1".to_string()]);
    namespace.create_namespace(create_req).await.unwrap();

    let mut create_req = CreateNamespaceRequest::new();
    create_req.id = Some(vec!["parent".to_string(), "child2".to_string()]);
    namespace.create_namespace(create_req).await.unwrap();

    // List children of parent
    let list_req = ListNamespacesRequest {
        id: Some(vec!["parent".to_string()]),
        ..Default::default()
    };
    let result = namespace.list_namespaces(list_req).await;
    assert!(result.is_ok());
    let children = result.unwrap().namespaces;
    assert_eq!(children.len(), 2);
    assert!(children.contains(&"child1".to_string()));
    assert!(children.contains(&"child2".to_string()));

    // List root should only show parent
    let list_req = ListNamespacesRequest {
        id: Some(vec![]),
        ..Default::default()
    };
    let result = namespace.list_namespaces(list_req).await;
    assert!(result.is_ok());
    let root_namespaces = result.unwrap().namespaces;
    assert_eq!(root_namespaces.len(), 1);
    assert_eq!(root_namespaces[0], "parent");
}

#[tokio::test]
async fn test_table_in_child_namespace() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create child namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["test_ns".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Create table in child namespace
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_table_req = CreateTableRequest::new();
    create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    let result = namespace
        .create_table(create_table_req, bytes::Bytes::from(ipc_data))
        .await;
    assert!(result.is_ok(), "Failed to create table in child namespace");

    // List tables in child namespace
    let list_req = ListTablesRequest {
        id: Some(vec!["test_ns".to_string()]),
        ..Default::default()
    };
    let result = namespace.list_tables(list_req).await;
    assert!(result.is_ok());
    let tables = result.unwrap().tables;
    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0], "table1");

    // Verify table exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    let result = namespace.table_exists(exists_req).await;
    assert!(result.is_ok());

    // Describe table in child namespace
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    let result = namespace.describe_table(describe_req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.location.is_some());
}

/// Regression: a connection built before `__manifest` exists must still
/// resolve a child-namespaced table that a *different* connection registers
/// afterwards. Phalanx caches one DirectoryNamespace per db and the first op
/// on a fresh db is usually a read, so without a self-healing read path the
/// cached reader pins an empty manifest cell and every describe/exists/list
/// on the table reports "not found" forever -- even though `create_table`
/// reports it already exists. This is the geneva `__system$geneva_jobs`
/// open->create->open livelock.
#[tokio::test]
async fn test_read_self_heals_after_manifest_created_by_other_connection() {
    let temp_dir = TempStdDir::default();
    let root = temp_dir.to_str().unwrap();

    // Reader is built while no `__manifest` exists yet -> its read cell is
    // empty and, before the fix, stays empty forever.
    let reader = DirectoryNamespaceBuilder::new(root).build().await.unwrap();

    // A *separate* connection creates the child namespace + table, which
    // lazily creates `__manifest` and registers the entry.
    let writer = DirectoryNamespaceBuilder::new(root).build().await.unwrap();
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["test_ns".to_string()]);
    writer.create_namespace(create_ns_req).await.unwrap();
    let ipc_data = create_test_ipc_data(&create_test_schema());
    let mut create_table_req = CreateTableRequest::new();
    create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    writer
        .create_table(create_table_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // The reader, though built before the manifest existed, must now resolve
    // the table on every read path (was TableNotFound before the fix).
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    let resp = reader
        .describe_table(describe_req)
        .await
        .expect("describe_table must resolve a table registered after build");
    assert!(resp.location.is_some());

    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    reader
        .table_exists(exists_req)
        .await
        .expect("table_exists must resolve a table registered after build");

    let list_req = ListTablesRequest {
        id: Some(vec!["test_ns".to_string()]),
        ..Default::default()
    };
    let tables = reader.list_tables(list_req).await.unwrap().tables;
    assert_eq!(tables, vec!["table1".to_string()]);
}

/// Migration mode promises manifest-first lookup even at the root, so a
/// reader built before `__manifest` existed must still resolve a
/// manifest-only alias (`registered_table` -> `external_table.lance`) that
/// dir-listing cannot produce. Before the read path self-healed in migration
/// mode, the root gate bypassed the manifest probe and fell back to
/// dir-listing, which sees `external_table` but never `registered_table` ->
/// permanent TableNotFound for every registration made after build.
#[tokio::test]
async fn test_migration_root_read_self_heals_registered_alias() {
    use lance_namespace::models::RegisterTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Reader built while the root is empty -> no `__manifest`, read cell
    // empty (and, before the fix, frozen empty forever).
    let reader = DirectoryNamespaceBuilder::new(temp_path)
        .dir_listing_enabled(true)
        .dir_listing_to_manifest_migration_enabled(true)
        .build()
        .await
        .unwrap();

    // A separate connection writes an external dataset and registers it in
    // the manifest under a *different* logical name -- an alias dir-listing
    // cannot resolve. This is what lazily creates `__manifest`.
    let writer = DirectoryNamespaceBuilder::new(temp_path)
        .dir_listing_enabled(true)
        .dir_listing_to_manifest_migration_enabled(true)
        .build()
        .await
        .unwrap();
    let ipc_data = create_test_ipc_data(&create_test_schema());
    let table_uri = format!("{}/external_table.lance", temp_path);
    let cursor = Cursor::new(ipc_data);
    let stream_reader = StreamReader::try_new(cursor, None).unwrap();
    let batches: Vec<_> = stream_reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    let schema = batches[0].schema();
    let batch_results: Vec<_> = batches.into_iter().map(Ok).collect();
    let batch_reader = RecordBatchIterator::new(batch_results, schema);
    Dataset::write(Box::new(batch_reader), &table_uri, None)
        .await
        .unwrap();
    let mut register_req = RegisterTableRequest::new("external_table.lance".to_string());
    register_req.id = Some(vec!["registered_table".to_string()]);
    writer.register_table(register_req).await.unwrap();

    // The reader, built before `__manifest` existed, must now resolve the
    // manifest-only alias on every read path.
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["registered_table".to_string()]);
    reader
        .describe_table(describe_req)
        .await
        .expect("describe_table must resolve a manifest alias registered after build");

    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["registered_table".to_string()]);
    reader
        .table_exists(exists_req)
        .await
        .expect("table_exists must resolve a manifest alias registered after build");

    let list_req = ListTablesRequest {
        id: Some(vec![]),
        ..Default::default()
    };
    let tables = reader.list_tables(list_req).await.unwrap().tables;
    assert!(
        tables.contains(&"registered_table".to_string()),
        "list_tables must include the manifest alias registered after build, got {:?}",
        tables
    );
}

#[tokio::test]
async fn test_multiple_tables_in_child_namespace() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create child namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["test_ns".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Create multiple tables
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    for i in 1..=3 {
        let mut create_table_req = CreateTableRequest::new();
        create_table_req.id = Some(vec!["test_ns".to_string(), format!("table{}", i)]);
        namespace
            .create_table(create_table_req, bytes::Bytes::from(ipc_data.clone()))
            .await
            .unwrap();
    }

    // List tables
    let list_req = ListTablesRequest {
        id: Some(vec!["test_ns".to_string()]),
        ..Default::default()
    };
    let result = namespace.list_tables(list_req).await;
    assert!(result.is_ok());
    let tables = result.unwrap().tables;
    assert_eq!(tables.len(), 3);
    assert!(tables.contains(&"table1".to_string()));
    assert!(tables.contains(&"table2".to_string()));
    assert!(tables.contains(&"table3".to_string()));
}

#[tokio::test]
async fn test_drop_table_in_child_namespace() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create child namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["test_ns".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Create table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_table_req = CreateTableRequest::new();
    create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    namespace
        .create_table(create_table_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Drop table
    let mut drop_req = DropTableRequest::new();
    drop_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    let result = namespace.drop_table(drop_req).await;
    assert!(result.is_ok(), "Failed to drop table in child namespace");

    // Verify table no longer exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    let result = namespace.table_exists(exists_req).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_deeply_nested_namespace() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create deeply nested namespace hierarchy
    let mut create_req = CreateNamespaceRequest::new();
    create_req.id = Some(vec!["level1".to_string()]);
    namespace.create_namespace(create_req).await.unwrap();

    let mut create_req = CreateNamespaceRequest::new();
    create_req.id = Some(vec!["level1".to_string(), "level2".to_string()]);
    namespace.create_namespace(create_req).await.unwrap();

    let mut create_req = CreateNamespaceRequest::new();
    create_req.id = Some(vec![
        "level1".to_string(),
        "level2".to_string(),
        "level3".to_string(),
    ]);
    namespace.create_namespace(create_req).await.unwrap();

    // Create table in deeply nested namespace
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_table_req = CreateTableRequest::new();
    create_table_req.id = Some(vec![
        "level1".to_string(),
        "level2".to_string(),
        "level3".to_string(),
        "table1".to_string(),
    ]);
    let result = namespace
        .create_table(create_table_req, bytes::Bytes::from(ipc_data))
        .await;
    assert!(
        result.is_ok(),
        "Failed to create table in deeply nested namespace"
    );

    // Verify table exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec![
        "level1".to_string(),
        "level2".to_string(),
        "level3".to_string(),
        "table1".to_string(),
    ]);
    let result = namespace.table_exists(exists_req).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_namespace_with_properties() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create namespace with properties
    let mut properties = HashMap::new();
    properties.insert("owner".to_string(), "test_user".to_string());
    properties.insert("description".to_string(), "Test namespace".to_string());

    let mut create_req = CreateNamespaceRequest::new();
    create_req.id = Some(vec!["test_ns".to_string()]);
    create_req.properties = Some(properties.clone());
    namespace.create_namespace(create_req).await.unwrap();

    // Describe namespace and verify properties
    let describe_req = DescribeNamespaceRequest {
        id: Some(vec!["test_ns".to_string()]),
        ..Default::default()
    };
    let result = namespace.describe_namespace(describe_req).await;
    assert!(result.is_ok());
    let response = result.unwrap();
    assert!(response.properties.is_some());
    let props = response.properties.unwrap();
    assert_eq!(props.get("owner"), Some(&"test_user".to_string()));
    assert_eq!(
        props.get("description"),
        Some(&"Test namespace".to_string())
    );
}

#[tokio::test]
async fn test_cannot_drop_namespace_with_tables() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["test_ns".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Create table in namespace
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_table_req = CreateTableRequest::new();
    create_table_req.id = Some(vec!["test_ns".to_string(), "table1".to_string()]);
    namespace
        .create_table(create_table_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Try to drop namespace - should fail
    let mut drop_req = DropNamespaceRequest::new();
    drop_req.id = Some(vec!["test_ns".to_string()]);
    let result = namespace.drop_namespace(drop_req).await;
    assert!(
        result.is_err(),
        "Should not be able to drop namespace with tables"
    );
}

#[tokio::test]
async fn test_isolation_between_namespaces() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create two namespaces
    let mut create_req = CreateNamespaceRequest::new();
    create_req.id = Some(vec!["ns1".to_string()]);
    namespace.create_namespace(create_req).await.unwrap();

    let mut create_req = CreateNamespaceRequest::new();
    create_req.id = Some(vec!["ns2".to_string()]);
    namespace.create_namespace(create_req).await.unwrap();

    // Create table with same name in both namespaces
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut create_table_req = CreateTableRequest::new();
    create_table_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
    namespace
        .create_table(create_table_req, bytes::Bytes::from(ipc_data.clone()))
        .await
        .unwrap();

    let mut create_table_req = CreateTableRequest::new();
    create_table_req.id = Some(vec!["ns2".to_string(), "table1".to_string()]);
    namespace
        .create_table(create_table_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // List tables in each namespace
    let list_req = ListTablesRequest {
        id: Some(vec!["ns1".to_string()]),
        page_token: None,
        limit: None,
        ..Default::default()
    };
    let result = namespace.list_tables(list_req).await.unwrap();
    assert_eq!(result.tables.len(), 1);
    assert_eq!(result.tables[0], "table1");

    let list_req = ListTablesRequest {
        id: Some(vec!["ns2".to_string()]),
        page_token: None,
        limit: None,
        ..Default::default()
    };
    let result = namespace.list_tables(list_req).await.unwrap();
    assert_eq!(result.tables.len(), 1);
    assert_eq!(result.tables[0], "table1");

    // Drop table in ns1 shouldn't affect ns2
    let mut drop_req = DropTableRequest::new();
    drop_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
    namespace.drop_table(drop_req).await.unwrap();

    // Verify ns1 table is gone but ns2 table still exists
    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["ns1".to_string(), "table1".to_string()]);
    assert!(namespace.table_exists(exists_req).await.is_err());

    let mut exists_req = TableExistsRequest::new();
    exists_req.id = Some(vec!["ns2".to_string(), "table1".to_string()]);
    assert!(namespace.table_exists(exists_req).await.is_ok());
}

#[tokio::test]
async fn test_migrate_directory_tables() {
    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Step 1: Create tables in directory-only mode
    let dir_only_ns = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    // Create some tables
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    for i in 1..=3 {
        let mut create_req = CreateTableRequest::new();
        create_req.id = Some(vec![format!("table{}", i)]);
        dir_only_ns
            .create_table(create_req, bytes::Bytes::from(ipc_data.clone()))
            .await
            .unwrap();
    }

    drop(dir_only_ns);

    // Step 2: Create namespace with dual mode (manifest + directory listing)
    let dual_mode_ns = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    // Before migration, tables should be visible (via directory listing fallback)
    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    let tables = dual_mode_ns.list_tables(list_req).await.unwrap().tables;
    assert_eq!(tables.len(), 3);

    // Run migration
    let migrated_count = dual_mode_ns.migrate().await.unwrap();
    assert_eq!(migrated_count, 3, "Should migrate all 3 tables");

    // Verify tables are now in manifest
    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    let tables = dual_mode_ns.list_tables(list_req).await.unwrap().tables;
    assert_eq!(tables.len(), 3);

    // Run migration again - should be idempotent
    let migrated_count = dual_mode_ns.migrate().await.unwrap();
    assert_eq!(
        migrated_count, 0,
        "Should not migrate already-migrated tables"
    );

    drop(dual_mode_ns);

    // Step 3: Create namespace with manifest-only mode
    let manifest_only_ns = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();

    // Tables should still be accessible (now from manifest only)
    let mut list_req = ListTablesRequest::new();
    list_req.id = Some(vec![]);
    let tables = manifest_only_ns.list_tables(list_req).await.unwrap().tables;
    assert_eq!(tables.len(), 3);
    assert!(tables.contains(&"table1".to_string()));
    assert!(tables.contains(&"table2".to_string()));
    assert!(tables.contains(&"table3".to_string()));
}

#[tokio::test]
async fn test_migrate_without_manifest() {
    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace without manifest
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .manifest_enabled(false)
        .dir_listing_enabled(true)
        .build()
        .await
        .unwrap();

    // migrate() should return 0 when manifest is not enabled
    let migrated_count = namespace.migrate().await.unwrap();
    assert_eq!(migrated_count, 0);
}
