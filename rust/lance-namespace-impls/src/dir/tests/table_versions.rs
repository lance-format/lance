// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[tokio::test]
async fn test_table_version_tracking_enabled_managed_versioning() {
    use lance_namespace::models::DescribeTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace with table_version_tracking_enabled=true
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .table_version_tracking_enabled(true)
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

    // Describe table should return managed_versioning=true
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    let describe_resp = namespace.describe_table(describe_req).await.unwrap();

    // managed_versioning should be true
    assert_eq!(
        describe_resp.managed_versioning,
        Some(true),
        "managed_versioning should be true when table_version_tracking_enabled=true"
    );
}

#[tokio::test]
async fn test_table_version_tracking_disabled_no_managed_versioning() {
    use lance_namespace::models::DescribeTableRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    // Create namespace with table_version_tracking_enabled=false (default)
    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .table_version_tracking_enabled(false)
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

    // Describe table should not have managed_versioning set
    let mut describe_req = DescribeTableRequest::new();
    describe_req.id = Some(vec!["test_table".to_string()]);
    let describe_resp = namespace.describe_table(describe_req).await.unwrap();

    // managed_versioning should be None when table_version_tracking_enabled=false
    assert!(
        describe_resp.managed_versioning.is_none(),
        "managed_versioning should be None when table_version_tracking_enabled=false, got: {:?}",
        describe_resp.managed_versioning
    );
}

#[tokio::test]
async fn test_list_table_versions() {
    use arrow::array::{Int32Array, RecordBatchIterator};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use lance::dataset::{Dataset, WriteMode, WriteParams};
    use lance_namespace::models::{CreateNamespaceRequest, ListTableVersionsRequest};

    let temp_dir = TempStrDir::default();
    let temp_path: &str = &temp_dir;

    let namespace: Arc<dyn LanceNamespace> = Arc::new(
        DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap(),
    );

    // Create parent namespace first
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["workspace".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Create a table using write_into_namespace (version 1)
    let table_id = vec!["workspace".to_string(), "test_table".to_string()];
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch.clone())], arrow_schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Create,
        ..Default::default()
    };
    let mut dataset = Dataset::write_into_namespace(
        batches,
        namespace.clone(),
        table_id.clone(),
        Some(write_params),
    )
    .await
    .unwrap();

    // Append to create version 2
    let batch2 = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch2)], arrow_schema.clone());
    dataset.append(batches, None).await.unwrap();

    // Append to create version 3
    let batch3 = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![300, 400]))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch3)], arrow_schema);
    dataset.append(batches, None).await.unwrap();

    // List versions - should have versions 1, 2, and 3
    let mut list_req = ListTableVersionsRequest::new();
    list_req.id = Some(table_id.clone());
    let list_resp = namespace.list_table_versions(list_req).await.unwrap();

    assert_eq!(
        list_resp.versions.len(),
        3,
        "Should have 3 versions, got: {:?}",
        list_resp.versions
    );

    // Verify each version
    for expected_version in 1..=3 {
        let version = list_resp
            .versions
            .iter()
            .find(|v| v.version == expected_version)
            .unwrap_or_else(|| panic!("Expected version {}", expected_version));

        assert!(
            !version.manifest_path.is_empty(),
            "manifest_path should be set for version {}",
            expected_version
        );
        assert!(
            version.manifest_path.contains(".manifest"),
            "manifest_path should contain .manifest for version {}",
            expected_version
        );
        assert!(
            version.manifest_size.is_some(),
            "manifest_size should be set for version {}",
            expected_version
        );
        assert!(
            version.manifest_size.unwrap() > 0,
            "manifest_size should be > 0 for version {}",
            expected_version
        );
        assert!(
            version.timestamp_millis.is_some(),
            "timestamp_millis should be set for version {}",
            expected_version
        );
    }
}

#[tokio::test]
async fn test_describe_table_version() {
    use arrow::array::{Int32Array, RecordBatchIterator};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use lance::dataset::{Dataset, WriteMode, WriteParams};
    use lance_namespace::models::{CreateNamespaceRequest, DescribeTableVersionRequest};

    let temp_dir = TempStrDir::default();
    let temp_path: &str = &temp_dir;

    let namespace: Arc<dyn LanceNamespace> = Arc::new(
        DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap(),
    );

    // Create parent namespace first
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["workspace".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Create a table using write_into_namespace (version 1)
    let table_id = vec!["workspace".to_string(), "test_table".to_string()];
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Create,
        ..Default::default()
    };
    let mut dataset = Dataset::write_into_namespace(
        batches,
        namespace.clone(),
        table_id.clone(),
        Some(write_params),
    )
    .await
    .unwrap();

    // Append data to create version 2
    let batch2 = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch2)], arrow_schema);
    dataset.append(batches, None).await.unwrap();

    // Describe version 1
    let mut describe_req = DescribeTableVersionRequest::new();
    describe_req.id = Some(table_id.clone());
    describe_req.version = Some(1);
    let describe_resp = namespace
        .describe_table_version(describe_req)
        .await
        .unwrap();

    let version = &describe_resp.version;
    assert_eq!(version.version, 1);
    assert!(version.timestamp_millis.is_some());
    assert!(
        !version.manifest_path.is_empty(),
        "manifest_path should be set"
    );
    assert!(
        version.manifest_path.contains(".manifest"),
        "manifest_path should contain .manifest"
    );
    assert!(
        version.manifest_size.is_some(),
        "manifest_size should be set"
    );
    assert!(
        version.manifest_size.unwrap() > 0,
        "manifest_size should be > 0"
    );

    // Describe version 2
    let mut describe_req = DescribeTableVersionRequest::new();
    describe_req.id = Some(table_id.clone());
    describe_req.version = Some(2);
    let describe_resp = namespace
        .describe_table_version(describe_req)
        .await
        .unwrap();

    let version = &describe_resp.version;
    assert_eq!(version.version, 2);
    assert!(version.timestamp_millis.is_some());
    assert!(
        !version.manifest_path.is_empty(),
        "manifest_path should be set"
    );
    assert!(
        version.manifest_size.is_some(),
        "manifest_size should be set"
    );
    assert!(
        version.manifest_size.unwrap() > 0,
        "manifest_size should be > 0"
    );
}

#[tokio::test]
async fn test_describe_table_version_latest() {
    use arrow::array::{Int32Array, RecordBatchIterator};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use lance::dataset::{Dataset, WriteMode, WriteParams};
    use lance_namespace::models::{CreateNamespaceRequest, DescribeTableVersionRequest};

    let temp_dir = TempStrDir::default();
    let temp_path: &str = &temp_dir;

    let namespace: Arc<dyn LanceNamespace> = Arc::new(
        DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap(),
    );

    // Create parent namespace first
    let mut create_ns_req = CreateNamespaceRequest::new();
    create_ns_req.id = Some(vec!["workspace".to_string()]);
    namespace.create_namespace(create_ns_req).await.unwrap();

    // Create a table using write_into_namespace (version 1)
    let table_id = vec!["workspace".to_string(), "test_table".to_string()];
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Create,
        ..Default::default()
    };
    let mut dataset = Dataset::write_into_namespace(
        batches,
        namespace.clone(),
        table_id.clone(),
        Some(write_params),
    )
    .await
    .unwrap();

    // Append to create version 2
    let batch2 = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![100, 200]))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch2)], arrow_schema.clone());
    dataset.append(batches, None).await.unwrap();

    // Append to create version 3
    let batch3 = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![300, 400]))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(batch3)], arrow_schema);
    dataset.append(batches, None).await.unwrap();

    // Describe latest version (no version specified)
    let mut describe_req = DescribeTableVersionRequest::new();
    describe_req.id = Some(table_id.clone());
    describe_req.version = None;
    let describe_resp = namespace
        .describe_table_version(describe_req)
        .await
        .unwrap();

    // Should return version 3 as it's the latest
    assert_eq!(describe_resp.version.version, 3);
}

#[tokio::test]
async fn test_create_table_version() {
    use futures::TryStreamExt;
    use lance::dataset::builder::DatasetBuilder;
    use lance_namespace::models::CreateTableVersionRequest;

    let temp_dir = TempStrDir::default();
    let temp_path: &str = &temp_dir;

    let namespace: Arc<dyn LanceNamespace> = Arc::new(
        DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap(),
    );

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Open the dataset using from_namespace to get proper object_store and paths
    let table_id = vec!["test_table".to_string()];
    let dataset = DatasetBuilder::from_namespace(namespace.clone(), table_id.clone())
        .await
        .unwrap()
        .load()
        .await
        .unwrap();

    // Use dataset's object_store to find and copy the manifest
    let versions_path = dataset.versions_dir();
    let manifest_metas: Vec<_> = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .list(Some(&versions_path))
        .try_collect()
        .await
        .unwrap();

    let manifest_meta = manifest_metas
        .iter()
        .find(|m| {
            m.location
                .filename()
                .map(|f| f.ends_with(".manifest"))
                .unwrap_or(false)
        })
        .expect("No manifest file found");

    // Read the existing manifest data
    let manifest_data = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .get(&manifest_meta.location)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    // Write to a staging location using the dataset's object_store
    let staging_path = dataset.versions_dir().join("staging_manifest");
    dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .put(&staging_path, manifest_data.into())
        .await
        .unwrap();

    // Create version 2 from staging manifest
    // Use the same naming scheme as the existing dataset (V2)
    let mut create_version_req = CreateTableVersionRequest::new(2, staging_path.to_string());
    create_version_req.id = Some(table_id.clone());
    create_version_req.naming_scheme = Some("V2".to_string());

    let result = namespace.create_table_version(create_version_req).await;
    assert!(
        result.is_ok(),
        "create_table_version should succeed: {:?}",
        result
    );

    // Verify version 2 was created at the path returned in the response
    let response = result.unwrap();
    let version_info = response
        .version
        .expect("response should contain version info");
    let version_2_path = Path::parse(&version_info.manifest_path).unwrap();
    let head_result = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .head(&version_2_path)
        .await;
    assert!(
        head_result.is_ok(),
        "Version 2 manifest should exist at {}",
        version_2_path
    );

    // Verify the staging file has been deleted
    let staging_head_result = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .head(&staging_path)
        .await;
    assert!(
        staging_head_result.is_err(),
        "Staging manifest should have been deleted after create_table_version"
    );
}

#[tokio::test]
async fn test_create_table_version_idempotent() {
    // A network retry of create_table_version with the same staging content
    // must succeed (not ConcurrentModification) once the version is published.
    use futures::TryStreamExt;
    use lance::dataset::builder::DatasetBuilder;
    use lance_namespace::models::CreateTableVersionRequest;

    let temp_dir = TempStrDir::default();
    let temp_path: &str = &temp_dir;

    let namespace: Arc<dyn LanceNamespace> = Arc::new(
        DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap(),
    );

    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    let table_id = vec!["test_table".to_string()];
    let dataset = DatasetBuilder::from_namespace(namespace.clone(), table_id.clone())
        .await
        .unwrap()
        .load()
        .await
        .unwrap();

    let versions_path = dataset.versions_dir();
    let manifest_metas: Vec<_> = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .list(Some(&versions_path))
        .try_collect()
        .await
        .unwrap();

    let manifest_meta = manifest_metas
        .iter()
        .find(|m| {
            m.location
                .filename()
                .map(|f| f.ends_with(".manifest"))
                .unwrap_or(false)
        })
        .expect("No manifest file found");

    let manifest_data = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .get(&manifest_meta.location)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let staging_path = dataset.versions_dir().join("staging_manifest");
    dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .put(&staging_path, manifest_data.clone().into())
        .await
        .unwrap();

    let mut create_version_req = CreateTableVersionRequest::new(2, staging_path.to_string());
    create_version_req.id = Some(table_id.clone());
    create_version_req.naming_scheme = Some("V2".to_string());
    let first = namespace
        .create_table_version(create_version_req)
        .await
        .expect("first create_table_version should succeed");

    // Re-stage identical bytes (simulates Lance commit retry rewriting staging).
    let retry_staging = dataset.versions_dir().join("staging_manifest_retry");
    dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .put(&retry_staging, manifest_data.into())
        .await
        .unwrap();

    let mut retry_req = CreateTableVersionRequest::new(2, retry_staging.to_string());
    retry_req.id = Some(table_id.clone());
    retry_req.naming_scheme = Some("V2".to_string());
    let second = namespace
        .create_table_version(retry_req)
        .await
        .expect("idempotent retry must succeed");

    assert_eq!(
        first.version.as_ref().map(|v| v.version),
        second.version.as_ref().map(|v| v.version)
    );
    assert_eq!(
        first.version.as_ref().map(|v| &v.manifest_path),
        second.version.as_ref().map(|v| &v.manifest_path)
    );
}

#[tokio::test]
async fn test_create_table_version_conflict() {
    // Same version with different content must fail ConcurrentModification.
    use futures::TryStreamExt;
    use lance::dataset::builder::DatasetBuilder;
    use lance_namespace::models::CreateTableVersionRequest;

    let temp_dir = TempStrDir::default();
    let temp_path: &str = &temp_dir;

    let namespace: Arc<dyn LanceNamespace> = Arc::new(
        DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap(),
    );

    // Create a table
    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    // Open the dataset using from_namespace to get proper object_store and paths
    let table_id = vec!["test_table".to_string()];
    let dataset = DatasetBuilder::from_namespace(namespace.clone(), table_id.clone())
        .await
        .unwrap()
        .load()
        .await
        .unwrap();

    // Use dataset's object_store to find and copy the manifest
    let versions_path = dataset.versions_dir();
    let manifest_metas: Vec<_> = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .list(Some(&versions_path))
        .try_collect()
        .await
        .unwrap();

    let manifest_meta = manifest_metas
        .iter()
        .find(|m| {
            m.location
                .filename()
                .map(|f| f.ends_with(".manifest"))
                .unwrap_or(false)
        })
        .expect("No manifest file found");

    // Read the existing manifest data
    let manifest_data = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .get(&manifest_meta.location)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    // Write to a staging location using the dataset's object_store
    let staging_path = dataset.versions_dir().join("staging_manifest");
    dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .put(&staging_path, manifest_data.into())
        .await
        .unwrap();

    // First create version 2 (should succeed)
    let mut create_version_req = CreateTableVersionRequest::new(2, staging_path.to_string());
    create_version_req.id = Some(table_id.clone());
    create_version_req.naming_scheme = Some("V2".to_string());
    let first_result = namespace.create_table_version(create_version_req).await;
    assert!(
        first_result.is_ok(),
        "First create_table_version for version 2 should succeed: {:?}",
        first_result
    );

    // Get the path from the response for verification
    let version_2_path = Path::parse(
        &first_result
            .unwrap()
            .version
            .expect("response should contain version info")
            .manifest_path,
    )
    .unwrap();

    // Different content for the same version number must conflict.
    let conflict_staging = dataset.versions_dir().join("staging_manifest_conflict");
    dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .put(
            &conflict_staging,
            bytes::Bytes::from_static(b"not-a-real-manifest").into(),
        )
        .await
        .unwrap();

    let mut create_version_req = CreateTableVersionRequest::new(2, conflict_staging.to_string());
    create_version_req.id = Some(table_id.clone());
    create_version_req.naming_scheme = Some("V2".to_string());

    let result = namespace.create_table_version(create_version_req).await;
    assert!(
        result.is_err(),
        "create_table_version should fail for existing version with different content"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("already exists") || err.contains("ConcurrentModification"),
        "expected ConcurrentModification, got: {err}"
    );

    // Verify version 2 still exists using the dataset's object_store
    let head_result = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .head(&version_2_path)
        .await;
    assert!(
        head_result.is_ok(),
        "Version 2 manifest should still exist at {}",
        version_2_path
    );
}

#[tokio::test]
async fn test_create_table_version_cas_rejects_gap() {
    // Strict CAS: version must be latest+1; skipping ahead is ConcurrentModification.
    use futures::TryStreamExt;
    use lance::dataset::builder::DatasetBuilder;
    use lance_namespace::models::CreateTableVersionRequest;

    let temp_dir = TempStrDir::default();
    let temp_path: &str = &temp_dir;

    let namespace: Arc<dyn LanceNamespace> = Arc::new(
        DirectoryNamespaceBuilder::new(temp_path)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap(),
    );

    let schema = create_test_schema();
    let ipc_data = create_test_ipc_data(&schema);
    let mut create_req = CreateTableRequest::new();
    create_req.id = Some(vec!["test_table".to_string()]);
    namespace
        .create_table(create_req, bytes::Bytes::from(ipc_data))
        .await
        .unwrap();

    let table_id = vec!["test_table".to_string()];
    let dataset = DatasetBuilder::from_namespace(namespace.clone(), table_id.clone())
        .await
        .unwrap()
        .load()
        .await
        .unwrap();

    let versions_path = dataset.versions_dir();
    let manifest_metas: Vec<_> = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .list(Some(&versions_path))
        .try_collect()
        .await
        .unwrap();
    let manifest_meta = manifest_metas
        .iter()
        .find(|m| {
            m.location
                .filename()
                .map(|f| f.ends_with(".manifest"))
                .unwrap_or(false)
        })
        .expect("No manifest file found");
    let manifest_data = dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .get(&manifest_meta.location)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let staging_path = dataset.versions_dir().join("staging_gap");
    dataset
        .object_store(None)
        .await
        .unwrap()
        .inner
        .put(&staging_path, manifest_data.into())
        .await
        .unwrap();

    // After create_table, latest is 1; requesting 5 must fail CAS.
    let mut req = CreateTableVersionRequest::new(5, staging_path.to_string());
    req.id = Some(table_id);
    req.naming_scheme = Some("V2".to_string());
    let err = namespace
        .create_table_version(req)
        .await
        .expect_err("gap create must fail CAS");
    let msg = err.to_string();
    assert!(
        msg.contains("CAS") || msg.contains("ConcurrentModification"),
        "expected CAS ConcurrentModification, got: {msg}"
    );
}

#[tokio::test]
async fn test_create_table_version_branch_cas_requires_parent_version() {
    // Empty branch chain with BranchContents must bootstrap at parent_version,
    // not an arbitrary version (e.g. 1 when forked from v2).
    use futures::TryStreamExt;
    use lance_namespace::models::CreateTableVersionRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let main_uri = open_dataset(&namespace, "users").await.uri().to_string();
    append_scalar_version(&main_uri, 10).await; // main -> v2

    let mut main = open_dataset(&namespace, "users").await;
    let fork_version = main.version().version;
    assert_eq!(fork_version, 2);
    let branch_uri = main
        .create_branch("exp", fork_version, None)
        .await
        .unwrap()
        .uri()
        .to_string();

    let branch_ds = Dataset::open(&branch_uri).await.unwrap();
    let versions_dir = branch_ds.versions_dir();
    let store = branch_ds.object_store(None).await.unwrap();
    let manifests: Vec<_> = store
        .inner
        .list(Some(&versions_dir))
        .try_collect()
        .await
        .unwrap();
    for meta in &manifests {
        if meta
            .location
            .filename()
            .is_some_and(|f| f.ends_with(".manifest"))
        {
            store.inner.delete(&meta.location).await.unwrap();
        }
    }
    // Confirm the branch object-store chain is empty (do not open the dataset:
    // with no manifests, Dataset::open would fail).
    let remaining_manifests = store
        .inner
        .list(Some(&versions_dir))
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .filter(|m| {
            m.location
                .filename()
                .is_some_and(|f| f.ends_with(".manifest"))
        })
        .count();
    assert_eq!(
        remaining_manifests, 0,
        "branch version chain should be empty after deleting manifests"
    );

    // Stage bytes from a main manifest.
    let main_ds = open_dataset(&namespace, "users").await;
    let main_versions = main_ds.versions_dir();
    let main_store = main_ds.object_store(None).await.unwrap();
    let source_meta = main_store
        .inner
        .list(Some(&main_versions))
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .find(|m| {
            m.location
                .filename()
                .is_some_and(|f| f.ends_with(".manifest"))
        })
        .expect("main should have a manifest");
    let source_bytes = main_store
        .inner
        .get(&source_meta.location)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();

    let staging_wrong = versions_dir.clone().join("staging_wrong");
    store
        .inner
        .put(&staging_wrong, source_bytes.clone().into())
        .await
        .unwrap();
    let err = namespace
        .create_table_version(CreateTableVersionRequest {
            id: Some(vec!["users".to_string()]),
            version: 1,
            manifest_path: staging_wrong.to_string(),
            naming_scheme: Some("V2".to_string()),
            branch: Some("exp".to_string()),
            ..Default::default()
        })
        .await
        .expect_err("bootstrap at v1 must fail when parent_version is 2");
    let msg = err.to_string();
    assert!(
        msg.contains("CAS") || msg.contains("ConcurrentModification"),
        "expected CAS ConcurrentModification, got: {msg}"
    );

    let staging_ok = versions_dir.join("staging_ok");
    store
        .inner
        .put(&staging_ok, source_bytes.into())
        .await
        .unwrap();
    let resp = namespace
        .create_table_version(CreateTableVersionRequest {
            id: Some(vec!["users".to_string()]),
            version: 2,
            manifest_path: staging_ok.to_string(),
            naming_scheme: Some("V2".to_string()),
            branch: Some("exp".to_string()),
            ..Default::default()
        })
        .await
        .expect("bootstrap at parent_version must succeed");
    assert_eq!(resp.version.as_ref().map(|v| v.version), Some(2));
}

#[tokio::test]
async fn test_create_table_version_table_not_found() {
    use lance_namespace::models::CreateTableVersionRequest;

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = DirectoryNamespaceBuilder::new(temp_path)
        .table_version_tracking_enabled(true)
        .build()
        .await
        .unwrap();

    // Try to create version for non-existent table
    let mut create_version_req =
        CreateTableVersionRequest::new(1, "/some/staging/path".to_string());
    create_version_req.id = Some(vec!["non_existent_table".to_string()]);

    let result = namespace.create_table_version(create_version_req).await;
    assert!(
        result.is_err(),
        "create_table_version should fail for non-existent table"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Table not found"),
        "Error should mention table not found, got: {}",
        err_msg
    );
}
