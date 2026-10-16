// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[tokio::test]
async fn test_create_table() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create test IPC data
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut request = CreateTableRequest::new();
    request.id = Some(vec!["test_table".to_string()]);

    let response = namespace
        .create_table(request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    assert!(response.location.is_some());
    assert!(response.location.unwrap().ends_with("test_table.lance"));
    assert_eq!(response.version, Some(1));
}

#[tokio::test]
async fn test_create_table_without_data() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    let mut request = CreateTableRequest::new();
    request.id = Some(vec!["test_table".to_string()]);

    let result = namespace.create_table(request, bytes::Bytes::new()).await;
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Arrow IPC stream) is required")
    );
}

#[tokio::test]
async fn test_create_table_with_invalid_id() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create test IPC data
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    // Test with empty ID
    let mut request = CreateTableRequest::new();
    request.id = Some(vec![]);

    let result = namespace
        .create_table(request, bytes::Bytes::from(ipc_data.clone()))
        .await;
    assert!(result.is_err());

    // Test with multi-level ID - should now work with manifest enabled
    // First create the parent namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["test_namespace".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Now create table in the namespace
    let mut request = CreateTableRequest::new();
    request.id = Some(vec!["test_namespace".to_string(), "table".to_string()]);

    let result = namespace
        .create_table(request, bytes::Bytes::from(ipc_data))
        .await;
    // Should succeed with manifest enabled
    assert!(
        result.is_ok(),
        "Multi-level table IDs should work with manifest enabled"
    );
}

#[tokio::test]
async fn test_list_tables() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Initially, no tables
    let mut request = ListTablesRequest::new();
    request.id = Some(vec![]);
    let response = namespace.list_tables(request).await.unwrap();
    assert_eq!(response.tables.len(), 0);

    // Create test IPC data
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    // Create a table
    let mut create_request = CreateTableRequest::new();
    create_request.id = Some(vec!["table1".to_string()]);
    namespace
        .create_table(create_request, bytes::Bytes::from(ipc_data.clone()))
        .await
        .unwrap();

    // Create another table
    let mut create_request = CreateTableRequest::new();
    create_request.id = Some(vec!["table2".to_string()]);
    namespace
        .create_table(create_request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // List tables should return both
    let mut request = ListTablesRequest::new();
    request.id = Some(vec![]);
    let response = namespace.list_tables(request).await.unwrap();
    let tables = response.tables;
    assert_eq!(tables.len(), 2);
    assert!(tables.contains(&"table1".to_string()));
    assert!(tables.contains(&"table2".to_string()));
}

#[tokio::test]
async fn test_list_tables_pagination() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    for name in ["alpha", "bravo", "charlie"] {
        let mut req = CreateTableRequest::new();
        req.id = Some(vec![name.to_string()]);
        namespace
            .create_table(req, bytes::Bytes::from(ipc_data.clone()))
            .await
            .unwrap();
    }

    // First page: limit=2, no page_token
    let first_page = namespace
        .list_tables(ListTablesRequest {
            id: Some(vec![]),
            limit: Some(2),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(first_page.tables, vec!["alpha", "bravo"]);
    assert_eq!(first_page.page_token.as_deref(), Some("bravo"));

    // Second page: use page_token from first response
    let second_page = namespace
        .list_tables(ListTablesRequest {
            id: Some(vec![]),
            limit: Some(2),
            page_token: first_page.page_token.clone(),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(second_page.tables, vec!["charlie"]);
    assert!(second_page.page_token.is_none());
}

#[tokio::test]
async fn test_list_tables_pagination_limit_zero() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut req = CreateTableRequest::new();
    req.id = Some(vec!["alpha".to_string()]);
    namespace
        .create_table(req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    let response = namespace
        .list_tables(ListTablesRequest {
            id: Some(vec![]),
            limit: Some(0),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(response.tables.is_empty());
    assert!(response.page_token.is_none());
}

#[tokio::test]
async fn test_list_tables_with_namespace_id() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // First create a child namespace
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["test_namespace".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Now list tables in the child namespace
    let mut request = ListTablesRequest::new();
    request.id = Some(vec!["test_namespace".to_string()]);

    let result = namespace.list_tables(request).await;
    // Should succeed (with manifest enabled) and return empty list (no tables yet)
    assert!(
        result.is_ok(),
        "list_tables should work with child namespace when manifest is enabled"
    );
    let response = result.unwrap();
    assert_eq!(
        response.tables.len(),
        0,
        "Namespace should have no tables yet"
    );
}

#[tokio::test]
async fn test_create_scalar_index() {
    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;

    let transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;
    let dataset = open_dataset(&namespace, "users").await;
    let expected_transaction_id = dataset
        .read_transaction()
        .await
        .unwrap()
        .map(|transaction| transaction.uuid);
    assert_eq!(transaction_id, expected_transaction_id);
    let indices = dataset.load_indices().await.unwrap();
    assert!(indices.iter().any(|index| index.name == "users_id_idx"));
}

#[tokio::test]
async fn test_create_vector_index() {
    use lance_namespace::models::CreateTableIndexRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_vector_table(&namespace, "vectors").await;

    let mut create_index_request =
        CreateTableIndexRequest::new("vector".to_string(), "IVF_FLAT".to_string());
    create_index_request.id = Some(vec!["vectors".to_string()]);
    create_index_request.name = Some("vector_idx".to_string());
    create_index_request.distance_type = Some("l2".to_string());
    let transaction_id = namespace
        .create_table_index(create_index_request)
        .await
        .unwrap()
        .transaction_id;

    let dataset = open_dataset(&namespace, "vectors").await;
    let expected_transaction_id = dataset
        .read_transaction()
        .await
        .unwrap()
        .map(|transaction| transaction.uuid);
    assert_eq!(transaction_id, expected_transaction_id);
    let indices = dataset.load_indices().await.unwrap();
    assert!(indices.iter().any(|index| index.name == "vector_idx"));
}

#[tokio::test]
async fn test_list_table_indices() {
    use lance_namespace::models::{CreateTableIndexRequest, ListTableIndicesRequest};

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    create_scalar_index(&namespace, "users", "a_idx").await;
    create_scalar_index(&namespace, "users", "b_idx").await;
    let transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;

    let response = namespace
        .list_table_indices(ListTableIndicesRequest {
            id: Some(vec!["users".to_string()]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(response.indexes.len(), 3);
    assert_eq!(response.indexes[0].index_name, "a_idx");
    assert_eq!(response.indexes[1].index_name, "b_idx");
    assert_eq!(response.indexes[2].index_name, "users_id_idx");
    assert!(response.page_token.is_none());
    let users_id_idx = response
        .indexes
        .iter()
        .find(|index| index.index_name == "users_id_idx")
        .unwrap();
    assert_eq!(users_id_idx.columns, vec!["id"]);
    assert_eq!(users_id_idx.status, "SUCCEEDED");

    // Enriched fields populated from the index metadata for a scalar index.
    assert_eq!(users_id_idx.index_type.as_deref(), Some("BTree"));
    assert!(
        users_id_idx
            .type_url
            .as_deref()
            .is_some_and(|s| !s.is_empty())
    );
    assert_eq!(users_id_idx.num_indexed_rows, Some(3));
    assert_eq!(users_id_idx.num_unindexed_rows, Some(0));
    assert_eq!(users_id_idx.num_segments, Some(1));
    assert!(users_id_idx.size_bytes.is_some_and(|size| size > 0));
    assert!(users_id_idx.created_at.is_some());
    assert!(users_id_idx.index_version.is_some());
    assert!(users_id_idx.index_details.is_some());

    let dataset = open_dataset(&namespace, "users").await;
    let expected_transaction_id = dataset
        .read_transaction()
        .await
        .unwrap()
        .map(|transaction| transaction.uuid);
    assert_eq!(transaction_id, expected_transaction_id);
    let indices = dataset.load_indices().await.unwrap();
    assert_eq!(
        indices
            .iter()
            .filter(|index| index.name == "users_id_idx")
            .count(),
        1
    );

    let first_page = namespace
        .list_table_indices(ListTableIndicesRequest {
            id: Some(vec!["users".to_string()]),
            limit: Some(2),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(first_page.indexes.len(), 2);
    assert_eq!(first_page.indexes[0].index_name, "a_idx");
    assert_eq!(first_page.indexes[1].index_name, "b_idx");
    assert_eq!(first_page.page_token.as_deref(), Some("b_idx"));

    let second_page = namespace
        .list_table_indices(ListTableIndicesRequest {
            id: Some(vec!["users".to_string()]),
            page_token: first_page.page_token.clone(),
            limit: Some(2),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(second_page.indexes.len(), 1);
    assert_eq!(second_page.indexes[0].index_name, "users_id_idx");
    assert!(second_page.page_token.is_none());

    // A vector index exercises a different type_url, index_type, and details payload.
    create_vector_table(&namespace, "vectors").await;
    let mut create_index_request =
        CreateTableIndexRequest::new("vector".to_string(), "IVF_FLAT".to_string());
    create_index_request.id = Some(vec!["vectors".to_string()]);
    create_index_request.name = Some("vector_idx".to_string());
    create_index_request.distance_type = Some("l2".to_string());
    namespace
        .create_table_index(create_index_request)
        .await
        .unwrap();

    let vector_response = namespace
        .list_table_indices(ListTableIndicesRequest {
            id: Some(vec!["vectors".to_string()]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(vector_response.indexes.len(), 1);
    let vector_idx = &vector_response.indexes[0];
    assert_eq!(vector_idx.index_name, "vector_idx");
    assert_eq!(vector_idx.columns, vec!["vector"]);
    assert_eq!(vector_idx.index_type.as_deref(), Some("IVF_FLAT"));
    assert!(
        vector_idx
            .type_url
            .as_deref()
            .is_some_and(|s| !s.is_empty())
    );
    assert!(vector_idx.num_indexed_rows.is_some());
    assert!(vector_idx.num_unindexed_rows.is_some());
    assert_eq!(vector_idx.num_segments, Some(1));
    assert!(vector_idx.created_at.is_some());
    assert!(vector_idx.index_version.is_some());
    assert!(vector_idx.index_details.is_some());
}

#[tokio::test]
async fn test_describe_table_index_stats() {
    use lance_namespace::models::DescribeTableIndexStatsRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;

    let response = namespace
        .describe_table_index_stats(DescribeTableIndexStatsRequest {
            id: Some(vec!["users".to_string()]),
            index_name: Some("users_id_idx".to_string()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.index_type, Some("BTree".to_string()));
    assert_eq!(response.num_indices, Some(1));
    assert_eq!(response.num_indexed_rows, Some(3));
    assert_eq!(response.num_unindexed_rows, Some(0));

    let dataset = open_dataset(&namespace, "users").await;
    let expected_transaction_id = dataset
        .read_transaction()
        .await
        .unwrap()
        .map(|transaction| transaction.uuid);
    assert_eq!(transaction_id, expected_transaction_id);
    let stats: serde_json::Value =
        serde_json::from_str(&dataset.index_statistics("users_id_idx").await.unwrap()).unwrap();
    assert_eq!(stats["index_type"], "BTree");
    assert_eq!(stats["num_indices"], 1);
    assert_eq!(stats["num_indexed_rows"], 3);
    assert_eq!(stats["num_unindexed_rows"], 0);
}

#[tokio::test]
async fn test_describe_transaction() {
    use lance_namespace::models::DescribeTransactionRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;
    let dataset = open_dataset(&namespace, "users").await;
    let latest_transaction = dataset.read_transaction().await.unwrap();
    assert_eq!(
        transaction_id,
        latest_transaction
            .as_ref()
            .map(|transaction| transaction.uuid.clone())
    );

    if let Some(transaction_id) = transaction_id {
        let response = namespace
            .describe_transaction(DescribeTransactionRequest {
                id: Some(vec!["users".to_string(), transaction_id.clone()]),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(response.status, "SUCCEEDED");
        assert_eq!(
            response
                .properties
                .as_ref()
                .and_then(|props| props.get("operation")),
            Some(&"CreateIndex".to_string())
        );
        assert_eq!(
            response
                .properties
                .as_ref()
                .and_then(|props| props.get("uuid")),
            Some(&transaction_id)
        );
    } else {
        assert!(latest_transaction.is_none());
    }
}

#[tokio::test]
async fn test_drop_table_index() {
    use lance_namespace::models::{DropTableIndexRequest, ListTableIndicesRequest};

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let create_transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;

    let drop_transaction_id = namespace
        .drop_table_index(DropTableIndexRequest {
            id: Some(vec!["users".to_string()]),
            index_name: Some("users_id_idx".to_string()),
            ..Default::default()
        })
        .await
        .unwrap()
        .transaction_id;

    let dataset = open_dataset(&namespace, "users").await;
    let previous_dataset = dataset
        .checkout_version(dataset.version().version - 1)
        .await
        .unwrap();
    let previous_transaction_id = previous_dataset
        .read_transaction()
        .await
        .unwrap()
        .map(|transaction| transaction.uuid);
    assert_eq!(create_transaction_id, previous_transaction_id);
    let expected_drop_transaction_id = dataset
        .read_transaction()
        .await
        .unwrap()
        .map(|transaction| transaction.uuid);
    assert_eq!(drop_transaction_id, expected_drop_transaction_id);
    let indices = dataset.load_indices().await.unwrap();
    assert!(!indices.iter().any(|index| index.name == "users_id_idx"));

    let list_response = namespace
        .list_table_indices(ListTableIndicesRequest {
            id: Some(vec!["users".to_string()]),
            ..Default::default()
        })
        .await
        .unwrap();
    assert!(list_response.indexes.is_empty());
}

#[tokio::test]
async fn test_describe_table() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create a table first
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut create_request = CreateTableRequest::new();
    create_request.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Describe the table
    let mut request = DescribeTableRequest::new();
    request.id = Some(vec!["test_table".to_string()]);
    let response = namespace.describe_table(request).await.unwrap();

    assert!(response.location.is_some());
    assert!(response.location.unwrap().ends_with("test_table.lance"));
}

#[tokio::test]
async fn test_describe_nonexistent_table() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    let mut request = DescribeTableRequest::new();
    request.id = Some(vec!["nonexistent".to_string()]);

    let result = namespace.describe_table(request).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Table not found"));
}

#[tokio::test]
async fn test_table_exists() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut create_request = CreateTableRequest::new();
    create_request.id = Some(vec!["existing_table".to_string()]);
    namespace
        .create_table(create_request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Check existing table
    let mut request = TableExistsRequest::new();
    request.id = Some(vec!["existing_table".to_string()]);
    let result = namespace.table_exists(request).await;
    assert!(result.is_ok());

    // Check non-existent table
    let mut request = TableExistsRequest::new();
    request.id = Some(vec!["nonexistent".to_string()]);
    let result = namespace.table_exists(request).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Table not found"));
}

#[tokio::test]
async fn test_drop_table() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);

    let mut create_request = CreateTableRequest::new();
    create_request.id = Some(vec!["table_to_drop".to_string()]);
    namespace
        .create_table(create_request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Verify it exists
    let mut exists_request = TableExistsRequest::new();
    exists_request.id = Some(vec!["table_to_drop".to_string()]);
    assert!(namespace.table_exists(exists_request.clone()).await.is_ok());

    // Drop the table
    let mut drop_request = DropTableRequest::new();
    drop_request.id = Some(vec!["table_to_drop".to_string()]);
    let response = namespace.drop_table(drop_request).await.unwrap();
    assert!(response.location.is_some());

    // Verify it no longer exists
    assert!(namespace.table_exists(exists_request).await.is_err());
}

#[tokio::test]
async fn test_drop_nonexistent_table() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    let mut request = DropTableRequest::new();
    request.id = Some(vec!["nonexistent".to_string()]);

    // Should not fail when dropping non-existent table (idempotent)
    let result = namespace.drop_table(request).await;
    // The operation might succeed or fail depending on implementation
    // But it should not panic
    let _ = result;
}

#[tokio::test]
async fn test_list_all_tables() {
    use lance_namespace::models::ListTablesRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "alpha").await;
    create_scalar_table(&namespace, "beta").await;

    let request = ListTablesRequest {
        id: Some(vec![]),
        page_token: None,
        limit: None,
        ..Default::default()
    };
    let response = namespace.list_all_tables(request).await.unwrap();
    let mut tables = response.tables;
    tables.sort();
    assert_eq!(tables, vec!["alpha", "beta"]);
}

#[tokio::test]
async fn test_restore_table() {
    use lance_namespace::models::RestoreTableRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;

    // Create a second version by creating a scalar index (this adds a new version)
    create_scalar_index(&namespace, "users", "users_id_idx").await;

    let dataset = open_dataset(&namespace, "users").await;
    let current_version = dataset.version().version;
    assert!(current_version >= 2, "Should have at least 2 versions");

    // Restore to version 1
    let mut restore_req = RestoreTableRequest::new(1);
    restore_req.id = Some(vec!["users".to_string()]);
    let response = namespace.restore_table(restore_req).await.unwrap();

    // transaction_id should be present (the restore operation)
    assert!(
        response.transaction_id.is_some(),
        "restore_table should return a transaction_id"
    );

    // Verify the dataset now has a new version (restore creates a new version)
    let dataset_after = open_dataset(&namespace, "users").await;
    assert!(
        dataset_after.version().version > current_version,
        "Restore should create a new version"
    );
}

#[tokio::test]
async fn test_alter_table_add_columns() {
    use lance_namespace::models::{
        AddColumnsEntry, AlterTableAddColumnsRequest, DescribeTableRequest,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_request = CreateTableRequest::new();
    create_request.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Add a new column
    let mut new_col = AddColumnsEntry::new("doubled_id".to_string());
    new_col.expression = Some(Some("id * 2".to_string()));
    let mut add_request = AlterTableAddColumnsRequest::new(vec![new_col]);
    add_request.id = Some(vec!["test_table".to_string()]);

    let response = namespace
        .alter_table_add_columns(add_request)
        .await
        .unwrap();
    assert!(
        response.version > 1,
        "Version should increment after adding columns"
    );

    // Verify via describe_table
    let mut describe_request = DescribeTableRequest::new();
    describe_request.id = Some(vec!["test_table".to_string()]);
    describe_request.load_detailed_metadata = Some(true);
    let describe_response = namespace.describe_table(describe_request).await.unwrap();
    assert!(describe_response.schema.is_some());

    let resp_schema = describe_response.schema.unwrap();
    let field_names: Vec<&str> = resp_schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(
        field_names.contains(&"doubled_id"),
        "Column 'doubled_id' should exist, got: {:?}",
        field_names
    );
}

#[tokio::test]
async fn test_update_table_schema_metadata() {
    use lance_namespace::models::UpdateTableSchemaMetadataRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "products").await;

    let mut metadata = HashMap::new();
    metadata.insert("owner".to_string(), "team_a".to_string());
    metadata.insert("version".to_string(), "1.0".to_string());

    let mut req = UpdateTableSchemaMetadataRequest::new();
    req.id = Some(vec!["products".to_string()]);
    req.metadata = Some(metadata.clone());

    let response = namespace.update_table_schema_metadata(req).await.unwrap();

    assert!(response.metadata.is_some());
    let returned = response.metadata.unwrap();
    assert_eq!(returned.get("owner"), Some(&"team_a".to_string()));
    assert_eq!(returned.get("version"), Some(&"1.0".to_string()));
    assert!(
        response.transaction_id.is_some(),
        "update_table_schema_metadata should return a transaction_id"
    );
}

#[tokio::test]
async fn test_alter_table_add_columns_missing_id() {
    use lance_namespace::models::{AddColumnsEntry, AlterTableAddColumnsRequest};

    let (namespace, _temp_dir) = create_test_namespace().await;

    let new_col = AddColumnsEntry::new("col".to_string());
    let request = AlterTableAddColumnsRequest::new(vec![new_col]);
    let result = namespace.alter_table_add_columns(request).await;
    assert!(result.is_err(), "Should fail when table ID is missing");
}

#[tokio::test]
async fn test_alter_table_alter_columns_rename() {
    use lance_namespace::models::{
        AlterColumnsEntry, AlterTableAlterColumnsRequest, DescribeTableRequest,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_request = CreateTableRequest::new();
    create_request.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Rename "name" to "full_name"
    let mut entry = AlterColumnsEntry::new("name".to_string());
    entry.rename = Some(Some("full_name".to_string()));
    let mut alter_request = AlterTableAlterColumnsRequest::new(vec![entry]);
    alter_request.id = Some(vec!["test_table".to_string()]);

    let response = namespace
        .alter_table_alter_columns(alter_request)
        .await
        .unwrap();
    assert!(
        response.version > 1,
        "Version should increment after altering columns"
    );

    // Verify the rename
    let mut describe_request = DescribeTableRequest::new();
    describe_request.id = Some(vec!["test_table".to_string()]);
    describe_request.load_detailed_metadata = Some(true);
    let describe_response = namespace.describe_table(describe_request).await.unwrap();
    assert!(describe_response.schema.is_some());

    let resp_schema = describe_response.schema.unwrap();
    let field_names: Vec<&str> = resp_schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(
        field_names.contains(&"full_name"),
        "Column should be renamed to 'full_name', got: {:?}",
        field_names
    );
    assert!(
        !field_names.contains(&"name"),
        "Old column 'name' should not exist, got: {:?}",
        field_names
    );
}

#[tokio::test]
async fn test_get_table_stats() {
    use lance_namespace::models::GetTableStatsRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "items").await;
    create_scalar_index(&namespace, "items", "items_id_idx").await;

    let mut req = GetTableStatsRequest::new();
    req.id = Some(vec!["items".to_string()]);

    let response = namespace.get_table_stats(req).await.unwrap();
    assert_eq!(response.num_rows, 3);
    assert_eq!(response.num_indices, 1);
}

#[tokio::test]
async fn test_explain_table_query_plan() {
    use lance_namespace::models::QueryTableRequestVector;
    use lance_namespace::models::{ExplainTableQueryPlanRequest, QueryTableRequest};

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "catalog").await;

    let mut query = QueryTableRequest::new(1, QueryTableRequestVector::new());
    query.filter = Some("id > 1".to_string());
    query.columns = Some(Box::new(QueryTableRequestColumns {
        column_names: Some(vec!["id".to_string(), "name".to_string()]),
        column_aliases: None,
    }));
    query.with_row_id = Some(true);

    let mut req = ExplainTableQueryPlanRequest::new(query);
    req.id = Some(vec!["catalog".to_string()]);

    let plan_str = namespace.explain_table_query_plan(req).await.unwrap();
    assert_plan_contains_all(
        &plan_str,
        &[
            "ProjectionExec: expr=[id@0 as id, name@2 as name",
            "projection=[name], source=stream(_rowid)",
            "LanceRead: uri=",
            "projection=[id]",
            "row_id=true, row_addr=false",
            "full_filter=id > Int32(1)",
            "refine_filter=id > Int32(1)",
        ],
        "Filtered explain plan should preserve late materialization and filter pushdown",
    );
}

#[tokio::test]
async fn test_alter_table_alter_columns_missing_id() {
    use lance_namespace::models::{AlterColumnsEntry, AlterTableAlterColumnsRequest};

    let (namespace, _temp_dir) = create_test_namespace().await;

    let entry = AlterColumnsEntry::new("name".to_string());
    let request = AlterTableAlterColumnsRequest::new(vec![entry]);
    let result = namespace.alter_table_alter_columns(request).await;
    assert!(result.is_err(), "Should fail when table ID is missing");
}

#[tokio::test]
async fn test_alter_table_drop_columns() {
    use lance_namespace::models::{AlterTableDropColumnsRequest, DescribeTableRequest};

    let (namespace, _temp_dir) = create_test_namespace().await;

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_request = CreateTableRequest::new();
    create_request.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_request, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Drop the "name" column
    let mut drop_request = AlterTableDropColumnsRequest::new(vec!["name".to_string()]);
    drop_request.id = Some(vec!["test_table".to_string()]);

    let response = namespace
        .alter_table_drop_columns(drop_request)
        .await
        .unwrap();
    assert!(
        response.version > 1,
        "Version should increment after dropping columns"
    );

    // Verify column was dropped
    let mut describe_request = DescribeTableRequest::new();
    describe_request.id = Some(vec!["test_table".to_string()]);
    describe_request.load_detailed_metadata = Some(true);
    let describe_response = namespace.describe_table(describe_request).await.unwrap();
    assert!(describe_response.schema.is_some());

    let resp_schema = describe_response.schema.unwrap();
    let field_names: Vec<&str> = resp_schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(
        !field_names.contains(&"name"),
        "Column 'name' should be dropped, got: {:?}",
        field_names
    );
    assert!(
        field_names.contains(&"id"),
        "Column 'id' should still exist, got: {:?}",
        field_names
    );
}

#[tokio::test]
async fn test_analyze_table_query_plan() {
    use lance_namespace::models::AnalyzeTableQueryPlanRequest;
    use lance_namespace::models::QueryTableRequestVector;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "catalog").await;

    let mut req = AnalyzeTableQueryPlanRequest::new(1, QueryTableRequestVector::new());
    req.id = Some(vec!["catalog".to_string()]);
    req.filter = Some("id > 0".to_string());
    req.columns = Some(Box::new(QueryTableRequestColumns {
        column_names: Some(vec!["id".to_string(), "name".to_string()]),
        column_aliases: None,
    }));
    req.with_row_id = Some(true);

    let analysis_str = namespace.analyze_table_query_plan(req).await.unwrap();
    assert_plan_contains_all(
        &analysis_str,
        &[
            "AnalyzeExec verbose=true",
            "ProjectionExec: elapsed=",
            "expr=[id@0 as id, name@2 as name",
            "projection=[name], source=stream(_rowid)",
            "LanceRead: elapsed=",
            "projection=[id]",
            "row_id=true, row_addr=false",
            "full_filter=id > Int32(0)",
            "refine_filter=id > Int32(0)",
            "metrics=[output_rows=",
        ],
        "Filtered analyze plan should preserve late materialization and filter pushdown",
    );
}

#[tokio::test]
async fn test_alter_table_drop_columns_missing_id() {
    use lance_namespace::models::AlterTableDropColumnsRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;

    let request = AlterTableDropColumnsRequest::new(vec!["col".to_string()]);
    let result = namespace.alter_table_drop_columns(request).await;
    assert!(result.is_err(), "Should fail when table ID is missing");
}

#[tokio::test]
async fn test_alter_table_drop_columns_nonexistent_table() {
    use lance_namespace::models::AlterTableDropColumnsRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;

    let mut request = AlterTableDropColumnsRequest::new(vec!["col".to_string()]);
    request.id = Some(vec!["nonexistent".to_string()]);
    let result = namespace.alter_table_drop_columns(request).await;
    assert!(result.is_err(), "Should fail when table does not exist");
}

#[tokio::test]
async fn test_alter_transaction_set_status() {
    use lance_namespace::models::{
        AlterTransactionAction, AlterTransactionRequest, AlterTransactionSetStatus,
        DescribeTransactionRequest,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let txn_id = create_scalar_index(&namespace, "users", "users_id_idx")
        .await
        .expect("create_scalar_index should return a transaction id");

    // First verify the transaction exists
    let describe_resp = namespace
        .describe_transaction(DescribeTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(describe_resp.status, "SUCCEEDED");

    // Alter the transaction status
    let response = namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            actions: vec![AlterTransactionAction {
                set_status_action: Some(Box::new(AlterTransactionSetStatus {
                    status: Some("Canceled".to_string()),
                })),
                set_property_action: None,
                unset_property_action: None,
            }],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.status, "Canceled");
    assert!(response.properties.is_some());
    let props = response.properties.unwrap();
    assert_eq!(props.get("uuid"), Some(&txn_id));
    assert_eq!(props.get("operation"), Some(&"CreateIndex".to_string()));
}

#[tokio::test]
async fn test_alter_transaction_set_property() {
    use lance_namespace::models::{
        AlterTransactionAction, AlterTransactionRequest, AlterTransactionSetProperty,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let txn_id = create_scalar_index(&namespace, "users", "users_id_idx")
        .await
        .expect("create_scalar_index should return a transaction id");

    let response = namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            actions: vec![AlterTransactionAction {
                set_status_action: None,
                set_property_action: Some(Box::new(AlterTransactionSetProperty {
                    key: Some("custom_key".to_string()),
                    value: Some("custom_value".to_string()),
                    mode: None,
                })),
                unset_property_action: None,
            }],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.status, "SUCCEEDED");
    let props = response.properties.unwrap();
    assert_eq!(props.get("custom_key"), Some(&"custom_value".to_string()));
}

#[tokio::test]
async fn test_alter_transaction_set_property_fail_mode() {
    use lance_namespace::models::{
        AlterTransactionAction, AlterTransactionRequest, AlterTransactionSetProperty,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let txn_id = create_scalar_index(&namespace, "users", "users_id_idx")
        .await
        .expect("create_scalar_index should return a transaction id");

    // First, set a non-reserved property so it exists in the sidecar.
    namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            actions: vec![AlterTransactionAction {
                set_status_action: None,
                set_property_action: Some(Box::new(AlterTransactionSetProperty {
                    key: Some("custom_key".to_string()),
                    value: Some("initial_value".to_string()),
                    mode: None,
                })),
                unset_property_action: None,
            }],
            ..Default::default()
        })
        .await
        .unwrap();

    // Now try to set the same property again with Fail mode, which must
    // exercise the mode='Fail' branch (not the reserved-key guard).
    let result = namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            actions: vec![AlterTransactionAction {
                set_status_action: None,
                set_property_action: Some(Box::new(AlterTransactionSetProperty {
                    key: Some("custom_key".to_string()),
                    value: Some("new_value".to_string()),
                    mode: Some("Fail".to_string()),
                })),
                unset_property_action: None,
            }],
            ..Default::default()
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_alter_transaction_unset_property() {
    use lance_namespace::models::{
        AlterTransactionAction, AlterTransactionRequest, AlterTransactionSetProperty,
        AlterTransactionUnsetProperty,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let txn_id = create_scalar_index(&namespace, "users", "users_id_idx")
        .await
        .expect("create_scalar_index should return a transaction id");

    // First set a custom property, then unset it
    let response = namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            actions: vec![
                AlterTransactionAction {
                    set_status_action: None,
                    set_property_action: Some(Box::new(AlterTransactionSetProperty {
                        key: Some("temp_key".to_string()),
                        value: Some("temp_value".to_string()),
                        mode: None,
                    })),
                    unset_property_action: None,
                },
                AlterTransactionAction {
                    set_status_action: None,
                    set_property_action: None,
                    unset_property_action: Some(Box::new(AlterTransactionUnsetProperty {
                        key: Some("temp_key".to_string()),
                        mode: None,
                    })),
                },
            ],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(response.status, "SUCCEEDED");
    let props = response.properties.unwrap();
    assert!(!props.contains_key("temp_key"));
}

#[tokio::test]
async fn test_alter_transaction_invalid_status() {
    use lance_namespace::models::{
        AlterTransactionAction, AlterTransactionRequest, AlterTransactionSetStatus,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let txn_id = create_scalar_index(&namespace, "users", "users_id_idx")
        .await
        .expect("create_scalar_index should return a transaction id");

    let result = namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            actions: vec![AlterTransactionAction {
                set_status_action: Some(Box::new(AlterTransactionSetStatus {
                    status: Some("InvalidStatus".to_string()),
                })),
                set_property_action: None,
                unset_property_action: None,
            }],
            ..Default::default()
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_alter_transaction_not_found() {
    use lance_namespace::models::{
        AlterTransactionAction, AlterTransactionRequest, AlterTransactionSetStatus,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;

    // Try to alter a non-existent transaction
    let result = namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string(), "non_existent_txn".to_string()]),
            actions: vec![AlterTransactionAction {
                set_status_action: Some(Box::new(AlterTransactionSetStatus {
                    status: Some("Canceled".to_string()),
                })),
                set_property_action: None,
                unset_property_action: None,
            }],
            ..Default::default()
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_alter_transaction_missing_id() {
    use lance_namespace::models::{
        AlterTransactionAction, AlterTransactionRequest, AlterTransactionSetStatus,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;

    // Try with missing id
    let result = namespace
        .alter_transaction(AlterTransactionRequest {
            id: None,
            actions: vec![AlterTransactionAction {
                set_status_action: Some(Box::new(AlterTransactionSetStatus {
                    status: Some("Canceled".to_string()),
                })),
                set_property_action: None,
                unset_property_action: None,
            }],
            ..Default::default()
        })
        .await;
    assert!(result.is_err());

    // Try with insufficient id parts
    let result = namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string()]),
            actions: vec![AlterTransactionAction {
                set_status_action: Some(Box::new(AlterTransactionSetStatus {
                    status: Some("Canceled".to_string()),
                })),
                set_property_action: None,
                unset_property_action: None,
            }],
            ..Default::default()
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_alter_transaction_persists_changes() {
    use lance_namespace::models::{
        AlterTransactionAction, AlterTransactionRequest, AlterTransactionSetProperty,
        AlterTransactionSetStatus, DescribeTransactionRequest,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let transaction_id = create_scalar_index(&namespace, "users", "users_id_idx").await;

    let txn_id = transaction_id.expect("scalar index should produce a transaction id");

    // Alter status and set a custom property.
    namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            actions: vec![
                AlterTransactionAction {
                    set_status_action: Some(Box::new(AlterTransactionSetStatus {
                        status: Some("Canceled".to_string()),
                    })),
                    set_property_action: None,
                    unset_property_action: None,
                },
                AlterTransactionAction {
                    set_status_action: None,
                    set_property_action: Some(Box::new(AlterTransactionSetProperty {
                        key: Some("owner".to_string()),
                        value: Some("alice".to_string()),
                        mode: None,
                    })),
                    unset_property_action: None,
                },
            ],
            ..Default::default()
        })
        .await
        .unwrap();

    // The changes must survive across a fresh describe_transaction call,
    // proving the alteration was persisted to the transaction file.
    let describe_resp = namespace
        .describe_transaction(DescribeTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            ..Default::default()
        })
        .await
        .unwrap();
    let props = describe_resp.properties.expect("properties should be set");
    assert_eq!(props.get("owner"), Some(&"alice".to_string()));
    // The internal `_status` marker should not leak into the response but
    // must be present on disk so subsequent alter_transaction calls can
    // observe the previously set status.
    assert!(!props.contains_key("_status"));

    let follow_up = namespace
        .alter_transaction(AlterTransactionRequest {
            id: Some(vec!["users".to_string(), txn_id.clone()]),
            actions: vec![],
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(follow_up.status, "Canceled");
    let follow_up_props = follow_up.properties.unwrap();
    assert_eq!(follow_up_props.get("owner"), Some(&"alice".to_string()));
}
