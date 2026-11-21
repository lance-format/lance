// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

use arrow::array::{Array, ListArray, StringArray};
use lance::dataset::UpdateBuilder;
use lance_namespace::models::{RegisterTableRequest, RenameTableRequest, TableExistsRequest};

async fn create_manifest_only_namespace() -> (DirectoryNamespace, TempStdDir) {
    let temp_dir = TempStdDir::default();
    let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
        .manifest_enabled(true)
        .dir_listing_enabled(false)
        .build()
        .await
        .unwrap();
    (namespace, temp_dir)
}

async fn create_table_at(
    namespace: &DirectoryNamespace,
    id: &[&str],
    data: Vec<u8>,
    properties: Option<HashMap<String, String>>,
) -> CreateTableResponse {
    let mut request = CreateTableRequest::new();
    request.id = Some(id.iter().map(|part| part.to_string()).collect());
    request.properties = properties;
    namespace
        .create_table(request, Bytes::from(data))
        .await
        .unwrap()
}

async fn describe_table_at(namespace: &DirectoryNamespace, id: &[&str]) -> DescribeTableResponse {
    let mut request = DescribeTableRequest::new();
    request.id = Some(id.iter().map(|part| part.to_string()).collect());
    namespace.describe_table(request).await.unwrap()
}

async fn assert_table_missing(namespace: &DirectoryNamespace, id: &[&str]) {
    let mut request = TableExistsRequest::new();
    request.id = Some(id.iter().map(|part| part.to_string()).collect());
    let error = namespace.table_exists(request).await.unwrap_err();
    assert_eq!(mutation_error_code(error), ErrorCode::TableNotFound);
}

async fn assert_row_count(location: &str, expected: usize) {
    let dataset = Dataset::open(location).await.unwrap();
    assert_eq!(dataset.count_rows(None).await.unwrap(), expected);
}

async fn manifest_version(root: &str) -> u64 {
    Dataset::open(&format!("{root}/__manifest"))
        .await
        .unwrap()
        .version()
        .version
}

async fn manifest_base_objects(root: &str, object_id: &str) -> Option<Vec<String>> {
    let dataset = Dataset::open(&format!("{root}/__manifest")).await.unwrap();
    let mut scanner = dataset.scan();
    scanner.project(&["base_objects"]).unwrap();
    scanner
        .filter(&format!("object_id = '{}'", object_id.replace('\'', "''")))
        .unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    assert_eq!(batch.num_rows(), 1);

    let base_objects = batch
        .column_by_name("base_objects")
        .unwrap()
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    if base_objects.is_null(0) {
        return None;
    }
    let values_array = base_objects.value(0);
    let values = values_array.as_any().downcast_ref::<StringArray>().unwrap();
    Some(
        (0..values.len())
            .map(|index| values.value(index).to_string())
            .collect(),
    )
}

fn relative_location(root: &str, location: &str) -> String {
    let root_url = uri_to_url(root).unwrap();
    let location_url = uri_to_url(location).unwrap();
    location_url
        .path()
        .strip_prefix(root_url.path().trim_end_matches('/'))
        .unwrap()
        .trim_start_matches('/')
        .to_string()
}

#[tokio::test]
async fn test_rename_table_is_metadata_only_and_recreates_old_name_fresh() {
    let (namespace, temp_dir) = create_manifest_only_namespace().await;
    let root = temp_dir.to_str().unwrap();
    let properties = HashMap::from([("owner".to_string(), "source".to_string())]);

    let created = create_table_at(
        &namespace,
        &["source"],
        create_single_row_test_ipc_data(),
        Some(properties.clone()),
    )
    .await;
    let retained_location = created.location.unwrap();

    let manifest_dataset = Dataset::open(&format!("{root}/__manifest")).await.unwrap();
    let mut update = UpdateBuilder::new(Arc::new(manifest_dataset));
    update = update.set("base_objects", "['base_a', 'base_b']").unwrap();
    update.build().unwrap().execute().await.unwrap();

    let version_before_rename = manifest_version(root).await;
    let mut rename = RenameTableRequest::new("renamed".to_string());
    rename.id = Some(vec!["source".to_string()]);
    namespace.rename_table(rename).await.unwrap();
    assert_eq!(manifest_version(root).await, version_before_rename + 1);

    let renamed = describe_table_at(&namespace, &["renamed"]).await;
    assert_eq!(
        renamed.location.as_deref(),
        Some(retained_location.as_str())
    );
    assert_eq!(renamed.properties, Some(properties));
    assert_eq!(
        manifest_base_objects(root, "renamed").await,
        Some(vec!["base_a".to_string(), "base_b".to_string()])
    );
    assert_table_missing(&namespace, &["source"]).await;
    assert_row_count(&retained_location, 1).await;

    let recreated = create_table_at(
        &namespace,
        &["source"],
        create_single_row_test_ipc_data(),
        None,
    )
    .await;
    let recreated_location = recreated.location.unwrap();
    assert_ne!(recreated_location, retained_location);
    assert_row_count(&recreated_location, 1).await;
    assert_row_count(&retained_location, 1).await;
}

#[tokio::test]
async fn test_rename_table_preserves_shared_registration() {
    let (namespace, temp_dir) = create_manifest_only_namespace().await;
    let root = temp_dir.to_str().unwrap();

    let created = create_table_at(
        &namespace,
        &["source"],
        create_single_row_test_ipc_data(),
        None,
    )
    .await;
    let source_location = created.location.unwrap();

    let mut register = RegisterTableRequest::new(relative_location(root, &source_location));
    register.id = Some(vec!["peer".to_string()]);
    namespace.register_table(register).await.unwrap();

    let mut rename = RenameTableRequest::new("renamed".to_string());
    rename.id = Some(vec!["source".to_string()]);
    namespace.rename_table(rename).await.unwrap();

    let peer = describe_table_at(&namespace, &["peer"]).await;
    let renamed = describe_table_at(&namespace, &["renamed"]).await;
    assert_eq!(peer.location.as_deref(), Some(source_location.as_str()));
    assert_eq!(renamed.location.as_deref(), Some(source_location.as_str()));
    assert_row_count(&source_location, 1).await;
    assert_table_missing(&namespace, &["source"]).await;
}

#[tokio::test]
async fn test_rename_table_across_child_namespaces_preserves_location() {
    let temp_dir = TempStdDir::default();
    let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
        .build()
        .await
        .unwrap();

    for id in ["source_ns", "destination_ns"] {
        let mut request = CreateNamespaceRequest::new();
        request.id = Some(vec![id.to_string()]);
        namespace.create_namespace(request).await.unwrap();
    }

    let created = create_table_at(
        &namespace,
        &["source_ns", "source"],
        create_single_row_test_ipc_data(),
        None,
    )
    .await;
    let location = created.location.unwrap();

    let mut rename = RenameTableRequest::new("renamed".to_string());
    rename.id = Some(vec!["source_ns".to_string(), "source".to_string()]);
    rename.new_namespace_id = Some(vec!["destination_ns".to_string()]);
    namespace.rename_table(rename).await.unwrap();

    let renamed = describe_table_at(&namespace, &["destination_ns", "renamed"]).await;
    assert_eq!(renamed.location.as_deref(), Some(location.as_str()));
    assert_row_count(&location, 1).await;
    assert_table_missing(&namespace, &["source_ns", "source"]).await;
}

#[rstest]
#[case::directory_listing(false)]
#[case::migration(true)]
#[tokio::test]
async fn test_rename_table_rejects_directory_resolved_root_tables(#[case] migration_enabled: bool) {
    let temp_dir = TempStdDir::default();
    let root = temp_dir.to_str().unwrap();
    let namespace = DirectoryNamespaceBuilder::new(root)
        .dir_listing_to_manifest_migration_enabled(migration_enabled)
        .build()
        .await
        .unwrap();
    create_table_at(
        &namespace,
        &["source"],
        create_single_row_test_ipc_data(),
        None,
    )
    .await;
    let version_before_rename = manifest_version(root).await;

    let mut rename = RenameTableRequest::new("renamed".to_string());
    rename.id = Some(vec!["source".to_string()]);
    let error = namespace.rename_table(rename).await.unwrap_err();
    assert_eq!(mutation_error_code(error), ErrorCode::Unsupported);
    assert_eq!(manifest_version(root).await, version_before_rename);
    describe_table_at(&namespace, &["source"]).await;
    assert_table_missing(&namespace, &["renamed"]).await;
}

#[tokio::test]
async fn test_rename_table_rejects_manifest_disabled_namespace() {
    let temp_dir = TempStdDir::default();
    let namespace = DirectoryNamespaceBuilder::new(temp_dir.to_str().unwrap())
        .manifest_enabled(false)
        .build()
        .await
        .unwrap();
    create_table_at(
        &namespace,
        &["source"],
        create_single_row_test_ipc_data(),
        None,
    )
    .await;

    let mut rename = RenameTableRequest::new("renamed".to_string());
    rename.id = Some(vec!["source".to_string()]);
    let error = namespace.rename_table(rename).await.unwrap_err();
    assert_eq!(mutation_error_code(error), ErrorCode::Unsupported);
    describe_table_at(&namespace, &["source"]).await;
    assert_table_missing(&namespace, &["renamed"]).await;
}

#[tokio::test]
async fn test_rename_table_rejects_invalid_inputs_and_missing_source() {
    let (namespace, _temp_dir) = create_manifest_only_namespace().await;

    let mut empty_id = RenameTableRequest::new("renamed".to_string());
    empty_id.id = Some(vec![]);
    let error = namespace.rename_table(empty_id).await.unwrap_err();
    assert_eq!(mutation_error_code(error), ErrorCode::InvalidInput);

    let mut empty_name = RenameTableRequest::new("   ".to_string());
    empty_name.id = Some(vec!["missing".to_string()]);
    let error = namespace.rename_table(empty_name).await.unwrap_err();
    assert_eq!(mutation_error_code(error), ErrorCode::InvalidInput);

    let mut missing = RenameTableRequest::new("renamed".to_string());
    missing.id = Some(vec!["missing".to_string()]);
    let error = namespace.rename_table(missing).await.unwrap_err();
    assert_eq!(mutation_error_code(error), ErrorCode::TableNotFound);
}

#[tokio::test]
async fn test_rename_table_rejects_occupied_destination() {
    let (namespace, _temp_dir) = create_manifest_only_namespace().await;

    let source = create_table_at(
        &namespace,
        &["source"],
        create_single_row_test_ipc_data(),
        None,
    )
    .await
    .location
    .unwrap();
    let destination = create_table_at(
        &namespace,
        &["destination"],
        create_single_row_test_ipc_data(),
        None,
    )
    .await
    .location
    .unwrap();

    let mut rename = RenameTableRequest::new("destination".to_string());
    rename.id = Some(vec!["source".to_string()]);
    let error = namespace.rename_table(rename).await.unwrap_err();
    assert_eq!(mutation_error_code(error), ErrorCode::TableAlreadyExists);
    assert_eq!(
        describe_table_at(&namespace, &["source"])
            .await
            .location
            .as_deref(),
        Some(source.as_str())
    );
    assert_eq!(
        describe_table_at(&namespace, &["destination"])
            .await
            .location
            .as_deref(),
        Some(destination.as_str())
    );
}

#[tokio::test]
async fn test_rename_table_uses_canonical_object_ids() {
    let (namespace, _temp_dir) = create_manifest_only_namespace().await;
    let created = create_table_at(
        &namespace,
        &["source$v1%"],
        create_single_row_test_ipc_data(),
        None,
    )
    .await;
    let location = created.location.unwrap();

    let mut rename = RenameTableRequest::new("target$v2%".to_string());
    rename.id = Some(vec!["source$v1%".to_string()]);
    namespace.rename_table(rename).await.unwrap();

    assert_eq!(
        describe_table_at(&namespace, &["target$v2%"])
            .await
            .location
            .as_deref(),
        Some(location.as_str())
    );
    assert_table_missing(&namespace, &["source$v1%"]).await;
}
