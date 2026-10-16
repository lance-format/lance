// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

#[tokio::test]
async fn test_list_table_versions_on_branch() {
    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    create_branch_with_commits(&namespace, "users", "exp", 2).await;

    // The branch lists its own chain, and every version resolves to a
    // manifest under the branch's tree path.
    let branch_versions = list_versions(&namespace, "users", Some("exp"))
        .await
        .unwrap();
    assert!(branch_versions.len() >= 2);
    assert!(
        branch_versions
            .iter()
            .all(|v| v.manifest_path.contains("tree/exp")),
        "branch versions must resolve to branch manifests: {:?}",
        branch_versions
    );

    // Unset and "main" behave identically and never see the tree path.
    let main_versions = list_versions(&namespace, "users", None).await.unwrap();
    let main_explicit = list_versions(&namespace, "users", Some("main"))
        .await
        .unwrap();
    assert_eq!(main_versions.len(), main_explicit.len());
    assert!(
        main_versions
            .iter()
            .all(|v| !v.manifest_path.contains("tree/"))
    );

    // A non-existent branch is a clean not-found, not an empty list.
    let missing = list_versions(&namespace, "users", Some("does-not-exist")).await;
    assert!(missing.is_err());
    assert!(missing.unwrap_err().to_string().contains("not found"));
}

#[tokio::test]
async fn test_describe_table_version_on_branch() {
    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    create_branch_with_commits(&namespace, "users", "exp", 2).await;

    let branch_versions = list_versions(&namespace, "users", Some("exp"))
        .await
        .unwrap();
    let latest = branch_versions.iter().map(|v| v.version).max().unwrap();

    // Describe latest on the branch returns the branch's manifest_path.
    let req = DescribeTableVersionRequest {
        id: Some(vec!["users".to_string()]),
        branch: Some("exp".to_string()),
        ..Default::default()
    };
    let resp = namespace.describe_table_version(req).await.unwrap();
    assert_eq!(resp.version.version, latest);
    assert!(resp.version.manifest_path.contains("tree/exp"));

    // A specific existing branch version resolves.
    let req = DescribeTableVersionRequest {
        id: Some(vec!["users".to_string()]),
        version: Some(latest),
        branch: Some("exp".to_string()),
        ..Default::default()
    };
    assert!(namespace.describe_table_version(req).await.is_ok());

    // A version absent on the branch is not found.
    let req = DescribeTableVersionRequest {
        id: Some(vec!["users".to_string()]),
        version: Some(999_999),
        branch: Some("exp".to_string()),
        ..Default::default()
    };
    assert!(namespace.describe_table_version(req).await.is_err());

    // A non-existent branch is not found.
    let req = DescribeTableVersionRequest {
        id: Some(vec!["users".to_string()]),
        branch: Some("nope".to_string()),
        ..Default::default()
    };
    let err = namespace.describe_table_version(req).await;
    assert!(err.is_err() && err.unwrap_err().to_string().contains("not found"));
}

#[tokio::test]
async fn test_restore_table_on_branch() {
    use lance_namespace::models::RestoreTableRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    create_branch_with_commits(&namespace, "users", "exp", 2).await;

    let before = list_versions(&namespace, "users", Some("exp"))
        .await
        .unwrap();
    let branch_latest = before.iter().map(|v| v.version).max().unwrap();
    let earliest = before.iter().map(|v| v.version).min().unwrap();
    let main_before = list_versions(&namespace, "users", None)
        .await
        .unwrap()
        .len();

    // Restoring the branch to an earlier version commits a NEW version on
    // the branch (restore is itself a commit), and must not touch main.
    let req = RestoreTableRequest {
        id: Some(vec!["users".to_string()]),
        version: earliest,
        branch: Some("exp".to_string()),
        ..Default::default()
    };
    let resp = namespace.restore_table(req).await.unwrap();
    assert!(resp.transaction_id.is_some());

    let after = list_versions(&namespace, "users", Some("exp"))
        .await
        .unwrap();
    let new_latest = after.iter().map(|v| v.version).max().unwrap();
    assert!(
        new_latest > branch_latest,
        "restore should add a branch version"
    );

    let main_after = list_versions(&namespace, "users", None)
        .await
        .unwrap()
        .len();
    assert_eq!(main_after, main_before, "main must be unaffected");
}

#[tokio::test]
async fn test_batch_delete_table_versions_on_branch() {
    use lance_namespace::models::{BatchDeleteTableVersionsRequest, VersionRange};

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    create_branch_with_commits(&namespace, "users", "exp", 2).await;

    let before = list_versions(&namespace, "users", Some("exp"))
        .await
        .unwrap();
    let main_before = list_versions(&namespace, "users", None).await.unwrap();

    // Delete the branch's whole history with a through-latest range (end = -1).
    // The branch manifests use V2 naming (inverted, zero-padded), so a nonzero
    // deleted_count proves the V2 fix: the old code constructed
    // "{version}.manifest" and silently matched nothing.
    let req = BatchDeleteTableVersionsRequest {
        id: Some(vec!["users".to_string()]),
        branch: Some("exp".to_string()),
        ranges: vec![VersionRange::new(0, -1)],
        ..Default::default()
    };
    let resp = namespace.batch_delete_table_versions(req).await.unwrap();
    assert_eq!(
        resp.deleted_count,
        Some(before.len() as i64),
        "every branch manifest should be physically deleted"
    );

    // The emptied branch now reads as not-found, and main is untouched.
    assert!(
        list_versions(&namespace, "users", Some("exp"))
            .await
            .is_err()
    );
    let main_after = list_versions(&namespace, "users", None).await.unwrap();
    assert_eq!(
        main_after.len(),
        main_before.len(),
        "main must be untouched"
    );
}

#[tokio::test]
async fn test_create_table_version_on_branch() {
    use futures::TryStreamExt;
    use lance_namespace::models::CreateTableVersionRequest;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;
    let branch_uri = create_branch_with_commits(&namespace, "users", "exp", 1).await;

    // Stage a manifest by copying one of the branch's existing manifests.
    let branch_ds = Dataset::open(&branch_uri).await.unwrap();
    let versions_dir = branch_ds.versions_dir();
    let store = branch_ds.object_store(None).await.unwrap();
    let existing = store
        .inner
        .list(Some(&versions_dir))
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .find(|m| {
            m.location
                .filename()
                .map(|f| f.ends_with(".manifest"))
                .unwrap_or(false)
        })
        .expect("a branch manifest");
    let bytes = store
        .inner
        .get(&existing.location)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let staging = versions_dir.join("staging_manifest");
    store.inner.put(&staging, bytes.into()).await.unwrap();

    let main_before = list_versions(&namespace, "users", None)
        .await
        .unwrap()
        .len();
    let new_version = list_versions(&namespace, "users", Some("exp"))
        .await
        .unwrap()
        .iter()
        .map(|v| v.version)
        .max()
        .unwrap()
        + 1;

    let req = CreateTableVersionRequest {
        id: Some(vec!["users".to_string()]),
        version: new_version,
        manifest_path: staging.to_string(),
        naming_scheme: Some("V2".to_string()),
        branch: Some("exp".to_string()),
        ..Default::default()
    };
    let resp = namespace.create_table_version(req).await.unwrap();
    let info = resp.version.expect("version info");
    // The new manifest must land under the branch's tree path.
    assert!(
        info.manifest_path.contains("tree/exp"),
        "got {}",
        info.manifest_path
    );

    // It is visible on the branch, and main did not gain a version.
    let after = list_versions(&namespace, "users", Some("exp"))
        .await
        .unwrap();
    assert!(after.iter().any(|v| v.version == new_version));
    let main_after = list_versions(&namespace, "users", None)
        .await
        .unwrap()
        .len();
    assert_eq!(main_after, main_before, "main must be unaffected");
}

/// The namespace-managed commit store derives the branch a request targets
/// from the base path it is handed, so a single store serves every branch of
/// the table: a branch-qualified base resolves and commits against the
/// branch chain while the table root targets main.
#[tokio::test]
async fn test_external_manifest_store_resolves_branch_from_base_path() {
    use futures::TryStreamExt;
    use lance::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;
    use lance_table::io::commit::external_manifest::ExternalManifestStore;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await; // main: version 1
    let branch_uri = create_branch_with_commits(&namespace, "users", "exp", 2).await;

    let namespace = Arc::new(namespace);
    let table_id = vec!["users".to_string()];
    let branch_ds = Dataset::open(&branch_uri).await.unwrap();
    let branch_base = branch_ds.branch_location().path;
    let root_base = branch_ds.branch_location().find_main().unwrap().path;
    let store = LanceNamespaceExternalManifestStore::new(
        namespace.clone(),
        table_id.clone(),
        root_base.clone(),
    );

    // The branch-qualified base resolves the branch chain, the root base
    // resolves main: proof the base path reaches list_table_versions.
    let (branch_latest, branch_path) = store
        .get_latest_version(branch_base.as_ref())
        .await
        .unwrap()
        .expect("branch has versions");
    let (_main_latest, main_path) = store
        .get_latest_version(root_base.as_ref())
        .await
        .unwrap()
        .expect("main has versions");
    assert!(
        branch_path.contains("tree/exp"),
        "branch latest must resolve to the branch tree: {}",
        branch_path
    );
    assert!(
        !main_path.contains("tree/exp"),
        "main latest must not resolve to a branch tree: {}",
        main_path
    );

    // describe (get) with the branch base also resolves to the branch tree.
    let described = store
        .get(branch_base.as_ref(), branch_latest)
        .await
        .unwrap();
    assert!(
        described.contains("tree/exp"),
        "describe on the branch must resolve to the branch tree: {}",
        described
    );

    // A base that is neither the root nor a branch chain is rejected.
    assert!(store.get_latest_version("somewhere/else").await.is_err());

    // Commit (put) with the branch base: the new version must land on the
    // branch chain. Stage a manifest by copying an existing branch manifest.
    let versions_dir = branch_ds.versions_dir();
    let obj = branch_ds.object_store(None).await.unwrap();
    let existing = obj
        .inner
        .list(Some(&versions_dir))
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .find(|m| {
            m.location
                .filename()
                .map(|f| f.ends_with(".manifest"))
                .unwrap_or(false)
        })
        .expect("a branch manifest");
    let bytes = obj
        .inner
        .get(&existing.location)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let size = bytes.len() as u64;
    let staging = versions_dir.clone().join("staging_manifest");
    obj.inner.put(&staging, bytes.into()).await.unwrap();

    let committed = store
        .put(
            &branch_base,
            branch_latest + 1,
            &staging,
            size,
            None,
            obj.inner.as_ref(),
            ManifestNamingScheme::V2,
        )
        .await
        .unwrap();
    assert!(
        committed.path.to_string().contains("tree/exp"),
        "a commit through a branch-qualified base must land on the branch tree: {}",
        committed.path
    );
}

/// write_into_namespace_on_branch must append against the branch chain
/// THROUGH the managed commit handler: the version is registered with the
/// namespace (create_table_version), lands on the branch tree, and main's
/// catalog is untouched. The ops-metrics assertions exist because a
/// physical-only commit is invisible to DirectoryNamespace branch listing
/// (it lists storage), while a catalog-authoritative namespace would
/// silently lose the version.
#[tokio::test]
async fn test_write_into_namespace_on_branch_appends_to_branch() {
    use lance::dataset::builder::DatasetBuilder;
    use lance_namespace::models::CreateTableBranchRequest;

    let temp = TempStdDir::default();
    let namespace = Arc::new(
        DirectoryNamespaceBuilder::new(temp.to_str().unwrap())
            .manifest_enabled(true)
            .table_version_tracking_enabled(true)
            .ops_metrics_enabled(true)
            .build()
            .await
            .unwrap(),
    );
    let ns: Arc<dyn LanceNamespace> = namespace.clone();
    let table_id = vec!["t".to_string()];
    create_managed_table(&ns, &table_id).await; // main: v1 (id=1), v2 (id=2)
    ns.create_table_branch(CreateTableBranchRequest {
        id: Some(table_id.clone()),
        name: "exp".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();

    let main_chain_len = |ns: Arc<dyn LanceNamespace>, table_id: Vec<String>| async move {
        ns.list_table_versions(ListTableVersionsRequest {
            id: Some(table_id),
            ..Default::default()
        })
        .await
        .unwrap()
        .versions
        .len()
    };
    let main_before = main_chain_len(ns.clone(), table_id.clone()).await;
    let commits_before = namespace
        .retrieve_ops_metrics()
        .get("create_table_version")
        .copied()
        .unwrap_or(0);

    let branch_ds = Dataset::write_into_namespace_on_branch(
        RecordBatchIterator::new(vec![Ok(single_int_batch(3))], single_int_schema()),
        ns.clone(),
        table_id.clone(),
        "exp",
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(branch_ds.manifest.branch.as_deref(), Some("exp"));
    assert_eq!(scan_id_column(&branch_ds).await, vec![1, 2, 3]);

    // The append must commit through the namespace, not just write a
    // physical manifest under the branch tree.
    let commits_after = namespace
        .retrieve_ops_metrics()
        .get("create_table_version")
        .copied()
        .unwrap_or(0);
    assert_eq!(
        commits_after,
        commits_before + 1,
        "the branch append must register its version via create_table_version"
    );
    let exp_versions = ns
        .list_table_versions(ListTableVersionsRequest {
            id: Some(table_id.clone()),
            branch: Some("exp".to_string()),
            ..Default::default()
        })
        .await
        .unwrap()
        .versions;
    assert!(
        exp_versions
            .iter()
            .all(|v| v.manifest_path.contains("tree/exp")),
        "branch versions must resolve to the branch tree: {:?}",
        exp_versions
    );
    assert_eq!(
        main_chain_len(ns.clone(), table_id.clone()).await,
        main_before,
        "main's catalog must be untouched by the branch append"
    );

    // A managed main append through the same entry point must register in
    // the catalog too, so a fresh managed open resolves the new latest.
    Dataset::write_into_namespace(
        RecordBatchIterator::new(vec![Ok(single_int_batch(100))], single_int_schema()),
        ns.clone(),
        table_id.clone(),
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert_eq!(
        main_chain_len(ns.clone(), table_id.clone()).await,
        main_before + 1,
        "a managed main append must register its version in the catalog"
    );
    let fresh = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .load()
        .await
        .unwrap();
    assert_eq!(
        scan_id_column(&fresh).await,
        vec![1, 2, 100],
        "a fresh managed open must resolve the appended version, not a stale latest"
    );
}

/// CREATE on a branch is rejected: a branch forks from an existing version.
#[tokio::test]
async fn test_write_into_namespace_on_branch_rejects_create() {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    let (namespace, _temp_dir) = create_test_namespace().await;
    let namespace = Arc::new(namespace);

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = arrow::record_batch::RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(StringArray::from(vec![Some("a")])),
        ],
    )
    .unwrap();
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());

    let result = Dataset::write_into_namespace_on_branch(
        reader,
        namespace.clone(),
        vec!["new_table".to_string()],
        "exp",
        Some(WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        }),
    )
    .await;
    assert!(result.is_err(), "create on a branch must be rejected");
    assert!(
        result.unwrap_err().to_string().contains("branch"),
        "error should mention the branch restriction"
    );
}

#[tokio::test]
async fn test_branch_name_validation_rejects_traversal() {
    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;

    // A traversal-style branch name is rejected as invalid input before any
    // storage path is built from it.
    let err = list_versions(&namespace, "users", Some("../evil")).await;
    assert!(err.is_err());
    assert!(err.unwrap_err().to_string().contains("invalid branch name"));
}

#[tokio::test]
async fn test_branch_ops_reject_zombie_branch() {
    use futures::TryStreamExt;
    use lance_namespace::models::{
        BatchDeleteTableVersionsRequest, CreateTableVersionRequest, RestoreTableRequest,
        VersionRange,
    };

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;

    let dataset = open_dataset(&namespace, "users").await;
    let store = dataset.object_store(None).await.unwrap();
    let manifest = store
        .inner
        .list(Some(&dataset.versions_dir()))
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
        .into_iter()
        .find(|m| {
            m.location
                .filename()
                .map(|f| f.ends_with(".manifest"))
                .unwrap_or(false)
        })
        .expect("a manifest");
    let bytes = store
        .inner
        .get(&manifest.location)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let zombie = dataset
        .branch_location()
        .find_branch(Some("ghost"))
        .unwrap()
        .path
        .join(VERSIONS_DIR)
        .join(manifest.location.filename().unwrap());
    store.inner.put(&zombie, bytes.into()).await.unwrap();

    assert!(dataset.branches().get("ghost").await.is_err());

    fn rejected<T: std::fmt::Debug>(label: &str, r: Result<T>) {
        match r {
            Ok(v) => panic!("{label} must reject the zombie branch, got Ok({v:?})"),
            Err(e) => assert!(e.to_string().contains("not found"), "{label}: {e}"),
        }
    }

    rejected(
        "list",
        list_versions(&namespace, "users", Some("ghost")).await,
    );
    rejected(
        "describe",
        namespace
            .describe_table_version(DescribeTableVersionRequest {
                id: Some(vec!["users".to_string()]),
                branch: Some("ghost".to_string()),
                ..Default::default()
            })
            .await,
    );
    rejected(
        "create",
        namespace
            .create_table_version(CreateTableVersionRequest {
                id: Some(vec!["users".to_string()]),
                version: 2,
                manifest_path: zombie.to_string(),
                branch: Some("ghost".to_string()),
                ..Default::default()
            })
            .await,
    );
    rejected(
        "restore",
        namespace
            .restore_table(RestoreTableRequest {
                id: Some(vec!["users".to_string()]),
                version: 1,
                branch: Some("ghost".to_string()),
                ..Default::default()
            })
            .await,
    );
    rejected(
        "batch_delete",
        namespace
            .batch_delete_table_versions(BatchDeleteTableVersionsRequest {
                id: Some(vec!["users".to_string()]),
                branch: Some("ghost".to_string()),
                ranges: vec![VersionRange::new(1, 1)],
                ..Default::default()
            })
            .await,
    );
}

/// V2 is the default naming scheme, and the pre-rewrite delete path
/// constructed `{version}.manifest` (a V1 name) and silently matched nothing
/// on a V2 table, returning deleted_count 0. This pins the fix on the main
/// chain (branch=None), which previously had no batch_delete coverage at all.
#[tokio::test]
async fn test_batch_delete_table_versions_main_v2() {
    use lance_namespace::models::{BatchDeleteTableVersionsRequest, VersionRange};

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await; // version 1
    let main_uri = open_dataset(&namespace, "users").await.uri().to_string();
    append_scalar_version(&main_uri, 100).await; // version 2
    append_scalar_version(&main_uri, 200).await; // version 3

    let before = list_versions(&namespace, "users", None).await.unwrap();
    assert!(before.len() >= 3);
    // Confirm these really are V2-named manifests (20-digit inverted version
    // + ".manifest" == 29 chars), i.e. the case the old code skipped.
    assert!(
        before
            .iter()
            .all(|v| v.manifest_path.rsplit('/').next().unwrap().len() == 29),
        "expected V2-named manifests: {:?}",
        before
    );
    let min_v = before.iter().map(|v| v.version).min().unwrap();
    let max_v = before.iter().map(|v| v.version).max().unwrap();

    // Delete everything except the latest version. end is exclusive, so
    // [min_v, max_v) keeps max_v.
    let req = BatchDeleteTableVersionsRequest {
        id: Some(vec!["users".to_string()]),
        ranges: vec![VersionRange::new(min_v, max_v)],
        ..Default::default()
    };
    let resp = namespace.batch_delete_table_versions(req).await.unwrap();
    assert_eq!(
        resp.deleted_count,
        Some((before.len() - 1) as i64),
        "V2 manifests must actually be deleted (was 0 before the fix)"
    );

    let after = list_versions(&namespace, "users", None).await.unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].version, max_v);
}

/// Pins the exclusive end of VersionRange: [v, v+1) must match only v.
#[tokio::test]
async fn test_batch_delete_end_is_exclusive() {
    use lance_namespace::models::{BatchDeleteTableVersionsRequest, VersionRange};

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await; // version 1
    let main_uri = open_dataset(&namespace, "users").await.uri().to_string();
    append_scalar_version(&main_uri, 100).await; // version 2
    append_scalar_version(&main_uri, 200).await; // version 3

    let before = list_versions(&namespace, "users", None).await.unwrap();
    let min_v = before.iter().map(|v| v.version).min().unwrap();

    let req = BatchDeleteTableVersionsRequest {
        id: Some(vec!["users".to_string()]),
        ranges: vec![VersionRange::new(min_v, min_v + 1)],
        ..Default::default()
    };
    let resp = namespace.batch_delete_table_versions(req).await.unwrap();
    assert_eq!(
        resp.deleted_count,
        Some(1),
        "only min_v is in [min_v, min_v+1)"
    );

    let after = list_versions(&namespace, "users", None).await.unwrap();
    assert!(
        !after.iter().any(|v| v.version == min_v),
        "min_v must be deleted"
    );
    assert_eq!(after.len(), before.len() - 1, "exactly one version removed");
}

#[tokio::test]
async fn test_batch_delete_rejects_unbounded_range() {
    use lance_namespace::models::{BatchDeleteTableVersionsRequest, VersionRange};

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await;

    // An unbounded range must be rejected up front, not turned into ~10^19
    // iterations / an unbounded id list.
    let req = BatchDeleteTableVersionsRequest {
        id: Some(vec!["users".to_string()]),
        ranges: vec![VersionRange::new(0, i64::MAX)],
        ..Default::default()
    };
    let err = namespace.batch_delete_table_versions(req).await;
    assert!(err.is_err());
    assert!(
        err.unwrap_err().to_string().contains("limit"),
        "expected a range-too-large error"
    );
}

/// Build a managed (manifest-tracked) namespace over `path`.
async fn create_managed_namespace(path: &str) -> Arc<dyn LanceNamespace> {
    Arc::new(
        DirectoryNamespaceBuilder::new(path)
            .manifest_enabled(true)
            .table_version_tracking_enabled(true)
            .build()
            .await
            .unwrap(),
    )
}

fn single_int_schema() -> Arc<arrow::datatypes::Schema> {
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]))
}

fn single_int_batch(seed: i32) -> arrow::record_batch::RecordBatch {
    use arrow::array::Int32Array;
    arrow::record_batch::RecordBatch::try_new(
        single_int_schema(),
        vec![Arc::new(Int32Array::from(vec![seed]))],
    )
    .unwrap()
}

/// Create a managed table with versions v1 (id=1) and v2 (id=2) on main and
/// return the main dataset handle.
async fn create_managed_table(ns: &Arc<dyn LanceNamespace>, table_id: &[String]) -> Dataset {
    let mut ds = Dataset::write_into_namespace(
        RecordBatchIterator::new(vec![Ok(single_int_batch(1))], single_int_schema()),
        ns.clone(),
        table_id.to_vec(),
        Some(WriteParams {
            mode: WriteMode::Create,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    ds.append(
        RecordBatchIterator::new(vec![Ok(single_int_batch(2))], single_int_schema()),
        None,
    )
    .await
    .unwrap();
    ds
}

/// Sorted values of the `id` column across a full scan.
async fn scan_id_column(ds: &Dataset) -> Vec<i32> {
    use arrow::array::Int32Array;
    use futures::TryStreamExt;
    let batches: Vec<arrow::record_batch::RecordBatch> = ds
        .scan()
        .try_into_stream()
        .await
        .unwrap()
        .try_collect()
        .await
        .unwrap();
    let mut ids: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    ids.sort();
    ids
}

/// E2e for the managed branch path through the builder: create a branch via the
/// namespace op, open it with `from_namespace(managed).with_branch`, commit on
/// it, and confirm the dataset is rooted at the branch chain (manifest, base
/// path and data placement) while main's catalog is untouched.
#[tokio::test]
async fn test_managed_branch_open_and_commit() {
    use futures::TryStreamExt;
    use lance::dataset::builder::DatasetBuilder;
    use lance_namespace::models::CreateTableBranchRequest;

    let temp = TempStdDir::default();
    let ns = create_managed_namespace(temp.to_str().unwrap()).await;
    let table_id = vec!["t".to_string()];
    create_managed_table(&ns, &table_id).await;
    let main_before = ns
        .list_table_versions(ListTableVersionsRequest {
            id: Some(table_id.clone()),
            ..Default::default()
        })
        .await
        .unwrap()
        .versions
        .len();

    // Create a branch via the namespace op (the FS-handler path, which succeeds
    // on a managed table).
    ns.create_table_branch(CreateTableBranchRequest {
        id: Some(table_id.clone()),
        name: "exp".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();

    // Open the managed table on the branch: the base path is qualified up
    // front and the manifest store derives the branch from it.
    let mut branch_ds = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .with_branch("exp", None)
        .load()
        .await
        .unwrap();
    assert_eq!(
        branch_ds.manifest.branch.as_deref(),
        Some("exp"),
        "with_branch on a managed table must open the branch chain"
    );
    let branch_base = branch_ds.branch_location().path;
    assert!(
        branch_base.as_ref().ends_with("tree/exp"),
        "the branch dataset must be rooted at the branch chain: {}",
        branch_base
    );
    let branch_v_before = branch_ds.version().version;

    // Commit on the branch.
    branch_ds
        .append(
            RecordBatchIterator::new(vec![Ok(single_int_batch(3))], single_int_schema()),
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        branch_ds.manifest.branch.as_deref(),
        Some("exp"),
        "the commit must stay on the branch"
    );
    assert!(
        branch_ds.version().version > branch_v_before,
        "the branch version must advance after the commit"
    );
    assert_eq!(scan_id_column(&branch_ds).await, vec![1, 2, 3]);

    // The committed data files live under the branch chain, not main's data
    // dir, so unmanaged readers of the branch and main's cleanup see a
    // consistent layout.
    let store = branch_ds.object_store(None).await.unwrap();
    let branch_data = branch_base.clone().join("data");
    let branch_files = store
        .inner
        .list(Some(&branch_data))
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    assert!(
        !branch_files.is_empty(),
        "the branch commit must place data files under the branch chain"
    );

    // The same branch is readable through the unmanaged (path-based) open.
    let table_uri = ns
        .describe_table(DescribeTableRequest {
            id: Some(table_id.clone()),
            ..Default::default()
        })
        .await
        .unwrap()
        .location
        .unwrap();
    let fs_branch_ds = DatasetBuilder::from_uri(&table_uri)
        .with_branch("exp", None)
        .load()
        .await
        .unwrap();
    assert_eq!(fs_branch_ds.manifest.branch.as_deref(), Some("exp"));
    assert_eq!(scan_id_column(&fs_branch_ds).await, vec![1, 2, 3]);

    // Main's catalog is untouched (branches are not tracked in __manifest),
    // and main still reads its own data.
    let main_after = ns
        .list_table_versions(ListTableVersionsRequest {
            id: Some(table_id.clone()),
            ..Default::default()
        })
        .await
        .unwrap()
        .versions
        .len();
    assert_eq!(
        main_after, main_before,
        "committing on the branch must not change main's chain"
    );
    let main_ds = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .load()
        .await
        .unwrap();
    assert_eq!(main_ds.manifest.branch, None);
    assert_eq!(scan_id_column(&main_ds).await, vec![1, 2]);
}

/// Branch-pointing tags on a managed table: create them through the normal
/// API (from both the main and the branch handle), open the table at the
/// tag, and check the tag out from an already-open dataset. All of these
/// must resolve the branch chain, never main's chain.
#[tokio::test]
async fn test_managed_branch_tags() {
    use lance::dataset::builder::DatasetBuilder;
    use lance::dataset::refs::Ref;
    use lance_namespace::models::CreateTableBranchRequest;

    let temp = TempStdDir::default();
    let ns = create_managed_namespace(temp.to_str().unwrap()).await;
    let table_id = vec!["t".to_string()];
    let main_ds = create_managed_table(&ns, &table_id).await;
    ns.create_table_branch(CreateTableBranchRequest {
        id: Some(table_id.clone()),
        name: "exp".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();
    let mut branch_ds = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .with_branch("exp", None)
        .load()
        .await
        .unwrap();
    branch_ds
        .append(
            RecordBatchIterator::new(vec![Ok(single_int_batch(3))], single_int_schema()),
            None,
        )
        .await
        .unwrap();
    let branch_version = branch_ds.version().version;

    // A branch-pointing tag created from the main handle must validate
    // against the branch chain (the version does not exist on main).
    main_ds
        .tags()
        .create("exp-tag", ("exp", Some(branch_version)))
        .await
        .unwrap();
    let tag = main_ds.tags().get("exp-tag").await.unwrap();
    assert_eq!(tag.branch.as_deref(), Some("exp"));
    assert_eq!(tag.version, branch_version);

    // A tag created from the branch handle resolves the branch implicitly.
    branch_ds
        .tags()
        .create("exp-tag2", branch_version)
        .await
        .unwrap();
    let tag2 = branch_ds.tags().get("exp-tag2").await.unwrap();
    assert_eq!(tag2.branch.as_deref(), Some("exp"));

    // Opening the managed table at the branch-pointing tag checks out the
    // branch chain.
    let tag_open = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .with_tag("exp-tag")
        .load()
        .await
        .unwrap();
    assert_eq!(tag_open.manifest.branch.as_deref(), Some("exp"));
    assert_eq!(tag_open.version().version, branch_version);
    assert_eq!(scan_id_column(&tag_open).await, vec![1, 2, 3]);

    // So does checking the tag out from an already-open main dataset.
    let tag_checkout = main_ds
        .checkout_version(Ref::Tag("exp-tag".to_string()))
        .await
        .unwrap();
    assert_eq!(tag_checkout.manifest.branch.as_deref(), Some("exp"));
    assert_eq!(scan_id_column(&tag_checkout).await, vec![1, 2, 3]);

    // A missing tag on a managed table errors at open.
    let err = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .with_tag("no-such-tag")
        .load()
        .await;
    assert!(err.is_err(), "a missing tag must error");
}

/// Cross-branch checkout on a managed table, including version numbers that
/// exist on both chains (branch numbering continues from the fork point, so
/// overlap is the common case). Every checkout must land on the requested
/// chain and read that chain's data.
#[tokio::test]
async fn test_managed_cross_branch_checkout() {
    use lance::dataset::builder::DatasetBuilder;
    use lance::dataset::refs::Ref;
    use lance_namespace::models::CreateTableBranchRequest;

    let temp = TempStdDir::default();
    let ns = create_managed_namespace(temp.to_str().unwrap()).await;
    let table_id = vec!["t".to_string()];
    let mut main_ds = create_managed_table(&ns, &table_id).await;
    ns.create_table_branch(CreateTableBranchRequest {
        id: Some(table_id.clone()),
        name: "exp".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();

    // exp gets id=3 at its tip; main gets id=100 at the same version number.
    let mut branch_ds = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .with_branch("exp", None)
        .load()
        .await
        .unwrap();
    branch_ds
        .append(
            RecordBatchIterator::new(vec![Ok(single_int_batch(3))], single_int_schema()),
            None,
        )
        .await
        .unwrap();
    let overlap_version = branch_ds.version().version;
    while main_ds.version().version < overlap_version {
        main_ds
            .append(
                RecordBatchIterator::new(vec![Ok(single_int_batch(100))], single_int_schema()),
                None,
            )
            .await
            .unwrap();
    }

    // main -> branch at the overlapping version number: must read the
    // branch's data, not main's same-numbered version.
    let on_branch = main_ds
        .checkout_version(Ref::Version(Some("exp".to_string()), Some(overlap_version)))
        .await
        .unwrap();
    assert_eq!(on_branch.manifest.branch.as_deref(), Some("exp"));
    assert_eq!(scan_id_column(&on_branch).await, vec![1, 2, 3]);

    // main -> branch latest.
    let mut on_branch_latest = main_ds.checkout_branch("exp").await.unwrap();
    assert_eq!(on_branch_latest.manifest.branch.as_deref(), Some("exp"));
    assert_eq!(on_branch_latest.version().version, overlap_version);

    // A commit through the checked-out handle (which shares main's commit
    // handler) must land on the branch chain, not main's.
    let main_chain_len = |ns: Arc<dyn LanceNamespace>, table_id: Vec<String>| async move {
        ns.list_table_versions(ListTableVersionsRequest {
            id: Some(table_id),
            ..Default::default()
        })
        .await
        .unwrap()
        .versions
        .len()
    };
    let main_before = main_chain_len(ns.clone(), table_id.clone()).await;
    on_branch_latest
        .append(
            RecordBatchIterator::new(vec![Ok(single_int_batch(4))], single_int_schema()),
            None,
        )
        .await
        .unwrap();
    assert_eq!(on_branch_latest.manifest.branch.as_deref(), Some("exp"));
    assert_eq!(scan_id_column(&on_branch_latest).await, vec![1, 2, 3, 4]);
    assert_eq!(
        main_chain_len(ns.clone(), table_id.clone()).await,
        main_before,
        "a commit on the checked-out branch must not advance main's chain"
    );

    // branch -> main at a specific version.
    let on_main = branch_ds
        .checkout_version(Ref::Version(None, Some(1)))
        .await
        .unwrap();
    assert_eq!(on_main.manifest.branch, None);
    assert_eq!(scan_id_column(&on_main).await, vec![1]);

    // branch -> another branch.
    ns.create_table_branch(CreateTableBranchRequest {
        id: Some(table_id.clone()),
        name: "exp2".to_string(),
        ..Default::default()
    })
    .await
    .unwrap();
    let on_branch2 = branch_ds.checkout_branch("exp2").await.unwrap();
    assert_eq!(on_branch2.manifest.branch.as_deref(), Some("exp2"));

    // A version missing from the branch chain errors loudly.
    let err = main_ds
        .checkout_version(Ref::Version(Some("exp".to_string()), Some(999)))
        .await;
    assert!(err.is_err(), "a version missing from the branch must error");
}

/// CommitBuilder must honor an explicitly supplied commit handler for a
/// Dataset destination: a managed-versioning commit through a dataset that
/// was opened without the namespace handler (as the Java and Python commit
/// APIs allow) must still register with the catalog instead of silently
/// writing a physical manifest the catalog never sees.
#[tokio::test]
async fn test_commit_builder_honors_explicit_handler_for_dataset_dest() {
    use lance::dataset::write::{CommitBuilder, InsertBuilder};
    use lance::dataset::{WriteDestination, builder::DatasetBuilder};
    use lance::io::commit::namespace_manifest::LanceNamespaceExternalManifestStore;
    use lance_table::io::commit::external_manifest::ExternalManifestCommitHandler;

    let temp = TempStdDir::default();
    let namespace = Arc::new(
        DirectoryNamespaceBuilder::new(temp.to_str().unwrap())
            .manifest_enabled(true)
            .table_version_tracking_enabled(true)
            .ops_metrics_enabled(true)
            .build()
            .await
            .unwrap(),
    );
    let ns: Arc<dyn LanceNamespace> = namespace.clone();
    let table_id = vec!["t".to_string()];
    create_managed_table(&ns, &table_id).await; // main: v1 (id=1), v2 (id=2)

    // Open WITHOUT the namespace handler, the way a binding caller can.
    let table_uri = ns
        .describe_table(DescribeTableRequest {
            id: Some(table_id.clone()),
            ..Default::default()
        })
        .await
        .unwrap()
        .location
        .unwrap();
    let plain_ds = Arc::new(Dataset::open(&table_uri).await.unwrap());

    let transaction = InsertBuilder::new(WriteDestination::Dataset(plain_ds.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![single_int_batch(3)])
        .await
        .unwrap();

    let handler = Arc::new(ExternalManifestCommitHandler {
        external_manifest_store: Arc::new(
            LanceNamespaceExternalManifestStore::for_table_uri(
                ns.clone(),
                table_id.clone(),
                &table_uri,
            )
            .unwrap(),
        ),
    });
    let commits_before = namespace
        .retrieve_ops_metrics()
        .get("create_table_version")
        .copied()
        .unwrap_or(0);
    let committed = CommitBuilder::new(WriteDestination::Dataset(plain_ds))
        .with_commit_handler(handler)
        .execute(transaction)
        .await
        .unwrap();
    assert_eq!(scan_id_column(&committed).await, vec![1, 2, 3]);

    let commits_after = namespace
        .retrieve_ops_metrics()
        .get("create_table_version")
        .copied()
        .unwrap_or(0);
    assert_eq!(
        commits_after,
        commits_before + 1,
        "the explicit handler must route the commit through create_table_version"
    );
    let fresh = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .load()
        .await
        .unwrap();
    assert_eq!(
        scan_id_column(&fresh).await,
        vec![1, 2, 3],
        "a fresh managed open must resolve the committed version"
    );
}

/// A branch forked from a non-latest version opens on its own chain.
#[tokio::test]
async fn test_managed_branch_from_non_latest_fork() {
    use lance::dataset::builder::DatasetBuilder;
    use lance_namespace::models::CreateTableBranchRequest;

    let temp = TempStdDir::default();
    let ns = create_managed_namespace(temp.to_str().unwrap()).await;
    let table_id = vec!["t".to_string()];
    create_managed_table(&ns, &table_id).await; // main: v1 (id=1), v2 (id=2)

    ns.create_table_branch(CreateTableBranchRequest {
        id: Some(table_id.clone()),
        name: "old".to_string(),
        from_version: Some(1),
        ..Default::default()
    })
    .await
    .unwrap();

    let old_ds = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .with_branch("old", None)
        .load()
        .await
        .unwrap();
    assert_eq!(old_ds.manifest.branch.as_deref(), Some("old"));
    assert_eq!(
        scan_id_column(&old_ds).await,
        vec![1],
        "the fork must contain only the fork-point data"
    );
}

/// Helper used by tag tests: creates a table with `versions` total versions
/// (1 create + N-1 appends) and returns the namespace plus the table id.
async fn create_tagged_test_table(
    versions: u32,
) -> (Arc<DirectoryNamespace>, TempStdDir, Vec<String>) {
    use arrow::array::{Int32Array, RecordBatchIterator};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use arrow::record_batch::RecordBatch;
    use lance::dataset::{Dataset, WriteMode, WriteParams};

    assert!(versions >= 1, "versions must be at least 1");

    let temp_dir = TempStdDir::default();
    let temp_path = temp_dir.to_str().unwrap();

    let namespace = Arc::new(
        DirectoryNamespaceBuilder::new(temp_path)
            .build()
            .await
            .unwrap(),
    );
    let table_id = vec!["tag_table".to_string()];
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int32,
        false,
    )]));
    let initial_batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    let batches = RecordBatchIterator::new(vec![Ok(initial_batch)], arrow_schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Create,
        ..Default::default()
    };

    let mut dataset = Dataset::write_into_namespace(
        batches,
        namespace.clone() as Arc<dyn LanceNamespace>,
        table_id.clone(),
        Some(write_params),
    )
    .await
    .unwrap();

    for i in 1..versions {
        let value_start = (i as i32) * 10;
        let batch = RecordBatch::try_new(
            arrow_schema.clone(),
            vec![Arc::new(Int32Array::from(vec![
                value_start,
                value_start + 1,
            ]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], arrow_schema.clone());
        dataset.append(batches, None).await.unwrap();
    }

    (namespace, temp_dir, table_id)
}

/// Downcast a lance-core error to its NamespaceError code for precise assertions.
fn namespace_code(err: &Error) -> Option<ErrorCode> {
    match err {
        Error::Namespace { source, .. } => {
            source.downcast_ref::<NamespaceError>().map(|e| e.code())
        }
        _ => None,
    }
}

#[tokio::test]
async fn test_create_and_list_branches() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(3).await;

    namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id.clone()),
            name: "dev".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
    namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id.clone()),
            name: "staging".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();

    let resp = namespace
        .list_table_branches(ListTableBranchesRequest {
            id: Some(table_id.clone()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        resp.branches.len(),
        2,
        "expected 2 branches, got: {:?}",
        resp.branches
    );
    assert!(resp.branches.contains_key("dev"));
    assert!(resp.branches.contains_key("staging"));
    assert!(resp.page_token.is_none());

    // Deleting one branch is reflected in a subsequent list.
    namespace
        .delete_table_branch(DeleteTableBranchRequest {
            id: Some(table_id.clone()),
            name: "dev".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();

    let resp = namespace
        .list_table_branches(ListTableBranchesRequest {
            id: Some(table_id),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(resp.branches.len(), 1, "expected 1 branch after delete");
    assert!(!resp.branches.contains_key("dev"));
    assert!(resp.branches.contains_key("staging"));
}

#[tokio::test]
async fn test_create_branch_from_version() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(3).await;

    // Fork explicitly from version 1 of main.
    namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id.clone()),
            name: "from-v1".to_string(),
            from_version: Some(1),
            ..Default::default()
        })
        .await
        .unwrap();

    let resp = namespace
        .list_table_branches(ListTableBranchesRequest {
            id: Some(table_id),
            ..Default::default()
        })
        .await
        .unwrap();
    let branch = resp
        .branches
        .get("from-v1")
        .expect("forked branch should be listed");
    assert_eq!(
        branch.parent_version, 1,
        "branch should fork from version 1"
    );
    assert!(
        branch.parent_branch.is_none(),
        "a branch forked from main has no parent branch"
    );
}

/// Forking from a NON-main source branch must clone that branch's chain.
/// Both chains are given a version 2 with diverged content, so a clone that
/// wrongly resolves the version under main succeeds silently with main's
/// data instead of erroring.
#[tokio::test]
async fn test_create_branch_from_other_branch() {
    use lance::dataset::builder::DatasetBuilder;

    let (namespace, _temp_dir) = create_test_namespace().await;
    create_scalar_table(&namespace, "users").await; // main v1: ids [1, 2, 3]
    // dev: forked at v1, one append (ids 100, 101) -> dev v2
    create_branch_with_commits(&namespace, "users", "dev", 1).await;
    // Diverge main to the same version number with different content.
    let main_ds = open_dataset(&namespace, "users").await;
    append_scalar_version(main_ds.uri(), 500).await; // main v2: + ids [500, 501]

    namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(vec!["users".to_string()]),
            name: "child".to_string(),
            from_branch: Some("dev".to_string()),
            from_version: Some(2),
            ..Default::default()
        })
        .await
        .unwrap();

    let child_ds = DatasetBuilder::from_uri(main_ds.uri())
        .with_branch("child", None)
        .load()
        .await
        .unwrap();
    let ids = scan_id_column(&child_ds).await;
    assert!(
        ids.contains(&100) && ids.contains(&101),
        "child must contain dev's appended rows, got: {:?}",
        ids
    );
    assert!(
        !ids.contains(&500),
        "child must not contain main's diverged rows, got: {:?}",
        ids
    );

    // The recorded metadata and the cloned data must agree on the parent.
    let listed = namespace
        .list_table_branches(ListTableBranchesRequest {
            id: Some(vec!["users".to_string()]),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(
        listed
            .branches
            .get("child")
            .unwrap()
            .parent_branch
            .as_deref(),
        Some("dev")
    );
}

#[tokio::test]
async fn test_create_existing_branch_conflict() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id.clone()),
            name: "dev".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();

    let err = namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id),
            name: "dev".to_string(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(
        namespace_code(&err),
        Some(ErrorCode::TableBranchAlreadyExists),
        "expected TableBranchAlreadyExists, got: {}",
        err
    );
    assert!(
        err.to_string().to_lowercase().contains("already exists"),
        "expected already-exists message, got: {}",
        err
    );
}

#[tokio::test]
async fn test_delete_unknown_branch() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    let err = namespace
        .delete_table_branch(DeleteTableBranchRequest {
            id: Some(table_id),
            name: "does-not-exist".to_string(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(
        namespace_code(&err),
        Some(ErrorCode::TableBranchNotFound),
        "expected TableBranchNotFound, got: {}",
        err
    );
    assert!(
        err.to_string().to_lowercase().contains("not found"),
        "expected not-found message, got: {}",
        err
    );
}

#[tokio::test]
async fn test_delete_referenced_branch_conflict() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    // A child forked from `parent` (via from_branch) makes `parent` a referenced branch.
    namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id.clone()),
            name: "parent".to_string(),
            ..Default::default()
        })
        .await
        .unwrap();
    namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id.clone()),
            name: "child".to_string(),
            from_branch: Some("parent".to_string()),
            ..Default::default()
        })
        .await
        .unwrap();

    // from_branch resolution: the child records its parent branch as its fork point.
    let listed = namespace
        .list_table_branches(ListTableBranchesRequest {
            id: Some(table_id.clone()),
            ..Default::default()
        })
        .await
        .unwrap();
    let child = listed
        .branches
        .get("child")
        .expect("child branch should be listed");
    assert_eq!(
        child.parent_branch.as_deref(),
        Some("parent"),
        "child should record parent branch as its fork point"
    );
    assert!(
        child.parent_version >= 1,
        "child should record the parent version it forked from, got {}",
        child.parent_version
    );

    // Deleting a branch that still has dependents is refused. The delete spec has no 409,
    // so it surfaces as a documented InvalidInput (400), not a conflict status.
    let err = namespace
        .delete_table_branch(DeleteTableBranchRequest {
            id: Some(table_id),
            name: "parent".to_string(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(
        namespace_code(&err),
        Some(ErrorCode::InvalidInput),
        "expected InvalidInput for deleting a referenced branch, got: {}",
        err
    );
    assert!(
        err.to_string().to_lowercase().contains("referenced"),
        "error should explain the branch is still referenced, got: {}",
        err
    );
}

#[tokio::test]
async fn test_branch_name_required() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    let create_err = namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id.clone()),
            name: String::new(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(
        namespace_code(&create_err),
        Some(ErrorCode::InvalidInput),
        "empty name on create should be InvalidInput, got: {}",
        create_err
    );
    assert!(
        create_err
            .to_string()
            .to_lowercase()
            .contains("must not be empty")
    );

    let delete_err = namespace
        .delete_table_branch(DeleteTableBranchRequest {
            id: Some(table_id),
            name: String::new(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(
        namespace_code(&delete_err),
        Some(ErrorCode::InvalidInput),
        "empty name on delete should be InvalidInput, got: {}",
        delete_err
    );
    assert!(
        delete_err
            .to_string()
            .to_lowercase()
            .contains("must not be empty")
    );
}

#[tokio::test]
async fn test_create_branch_rejects_negative_from_version() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    let err = namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id),
            name: "dev".to_string(),
            from_version: Some(-1),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(
        namespace_code(&err),
        Some(ErrorCode::InvalidInput),
        "negative from_version should be InvalidInput, got: {}",
        err
    );
    assert!(err.to_string().to_lowercase().contains("from_version"));
}

#[tokio::test]
async fn test_create_branch_nonexistent_from_version() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    // Version 999 does not exist (the table has 2 versions). create_branch's clone phase
    // raises DatasetNotFound, which we map to a documented InvalidInput (400).
    let err = namespace
        .create_table_branch(CreateTableBranchRequest {
            id: Some(table_id),
            name: "dev".to_string(),
            from_version: Some(999),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(
        namespace_code(&err),
        Some(ErrorCode::InvalidInput),
        "non-existent from_version should map to InvalidInput, got: {}",
        err
    );
    assert!(
        err.to_string().to_lowercase().contains("does not exist"),
        "error should name the missing source, got: {}",
        err
    );
}

#[tokio::test]
async fn test_create_and_list_tags() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(3).await;

    let mut req = CreateTableTagRequest::new("v1".to_string(), 1);
    req.id = Some(table_id.clone());
    namespace.create_table_tag(req).await.unwrap();

    let mut req = CreateTableTagRequest::new("v2".to_string(), 2);
    req.id = Some(table_id.clone());
    namespace.create_table_tag(req).await.unwrap();

    let mut list_req = ListTableTagsRequest::new();
    list_req.id = Some(table_id);
    let resp = namespace.list_table_tags(list_req).await.unwrap();

    assert_eq!(resp.tags.len(), 2, "expected 2 tags, got: {:?}", resp.tags);
    assert_eq!(resp.tags.get("v1").unwrap().version, 1);
    assert_eq!(resp.tags.get("v2").unwrap().version, 2);
    assert!(resp.page_token.is_none());
}

#[tokio::test]
async fn test_create_existing_tag_conflict() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    let mut req = CreateTableTagRequest::new("v1".to_string(), 1);
    req.id = Some(table_id.clone());
    namespace.create_table_tag(req).await.unwrap();

    let mut req = CreateTableTagRequest::new("v1".to_string(), 2);
    req.id = Some(table_id);
    let err = namespace.create_table_tag(req).await.unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("already exists"),
        "expected already-exists error, got: {}",
        err
    );
}

#[tokio::test]
async fn test_get_tag_version() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(3).await;

    let mut req = CreateTableTagRequest::new("release".to_string(), 2);
    req.id = Some(table_id.clone());
    namespace.create_table_tag(req).await.unwrap();

    let mut get_req = GetTableTagVersionRequest::new("release".to_string());
    get_req.id = Some(table_id);
    let resp = namespace.get_table_tag_version(get_req).await.unwrap();
    assert_eq!(resp.version, 2);
    assert_eq!(resp.branch, None);
}

#[tokio::test]
async fn test_get_unknown_tag() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    let mut get_req = GetTableTagVersionRequest::new("does-not-exist".to_string());
    get_req.id = Some(table_id);
    let err = namespace.get_table_tag_version(get_req).await.unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("not found"),
        "expected not-found error, got: {}",
        err
    );
}

#[tokio::test]
async fn test_update_tag_to_new_version() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(3).await;

    let mut req = CreateTableTagRequest::new("rolling".to_string(), 1);
    req.id = Some(table_id.clone());
    namespace.create_table_tag(req).await.unwrap();

    let mut update_req = UpdateTableTagRequest::new("rolling".to_string(), 3);
    update_req.id = Some(table_id.clone());
    namespace.update_table_tag(update_req).await.unwrap();

    let mut get_req = GetTableTagVersionRequest::new("rolling".to_string());
    get_req.id = Some(table_id);
    let resp = namespace.get_table_tag_version(get_req).await.unwrap();
    assert_eq!(resp.version, 3);
}

#[tokio::test]
async fn test_update_unknown_tag() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    let mut update_req = UpdateTableTagRequest::new("ghost".to_string(), 1);
    update_req.id = Some(table_id);
    let err = namespace.update_table_tag(update_req).await.unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("not found"),
        "expected not-found error, got: {}",
        err
    );
}

#[tokio::test]
async fn test_delete_tag() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    let mut req = CreateTableTagRequest::new("doomed".to_string(), 1);
    req.id = Some(table_id.clone());
    namespace.create_table_tag(req).await.unwrap();

    let mut delete_req = DeleteTableTagRequest::new("doomed".to_string());
    delete_req.id = Some(table_id.clone());
    namespace.delete_table_tag(delete_req).await.unwrap();

    let mut list_req = ListTableTagsRequest::new();
    list_req.id = Some(table_id.clone());
    let resp = namespace.list_table_tags(list_req).await.unwrap();
    assert!(resp.tags.is_empty(), "tag should be removed after delete");

    // A second get should return NotFound.
    let mut get_req = GetTableTagVersionRequest::new("doomed".to_string());
    get_req.id = Some(table_id);
    let err = namespace.get_table_tag_version(get_req).await.unwrap_err();
    assert!(err.to_string().to_lowercase().contains("not found"));
}

#[tokio::test]
async fn test_delete_unknown_tag() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    let mut delete_req = DeleteTableTagRequest::new("nope".to_string());
    delete_req.id = Some(table_id);
    let err = namespace.delete_table_tag(delete_req).await.unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("not found"),
        "expected not-found error, got: {}",
        err
    );
}

#[tokio::test]
async fn test_create_tag_invalid_version() {
    let (namespace, _temp_dir, table_id) = create_tagged_test_table(2).await;

    // version 0 should be rejected as InvalidInput before reaching the dataset.
    let mut req = CreateTableTagRequest::new("v0".to_string(), 0);
    req.id = Some(table_id.clone());
    let err = namespace.create_table_tag(req).await.unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("positive"),
        "expected positive-version error, got: {}",
        err
    );

    // empty tag name should also be rejected.
    let mut req = CreateTableTagRequest::new(String::new(), 1);
    req.id = Some(table_id);
    let err = namespace.create_table_tag(req).await.unwrap_err();
    assert!(
        err.to_string().to_lowercase().contains("must not be empty"),
        "expected empty-tag-name error, got: {}",
        err
    );
}

#[tokio::test]
async fn test_create_tag_table_not_found() {
    let (namespace, _temp_dir) = create_test_namespace().await;

    let mut req = CreateTableTagRequest::new("v1".to_string(), 1);
    req.id = Some(vec!["does_not_exist".to_string()]);
    let err = namespace.create_table_tag(req).await.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Table") && msg.to_lowercase().contains("not found"),
        "expected TableNotFound error, got: {}",
        err
    );
}

#[tokio::test]
async fn test_create_branch_on_managed_dataset_succeeds() {
    use lance::dataset::builder::DatasetBuilder;

    let temp = TempStdDir::default();
    let ns = create_managed_namespace(temp.to_str().unwrap()).await;
    let table_id = vec!["t".to_string()];
    let mut main = create_managed_table(&ns, &table_id).await;

    let fork_version = main.version().version;
    let branch = main
        .create_branch("exp", fork_version, None)
        .await
        .expect("create_branch failed");
    assert_eq!(branch.manifest.branch.as_deref(), Some("exp"));
    assert_eq!(scan_id_column(&branch).await, vec![1, 2]);

    let reopened = DatasetBuilder::from_namespace(ns.clone(), table_id.clone())
        .await
        .unwrap()
        .with_branch("exp", None)
        .load()
        .await
        .expect("reopen branch failed");
    assert_eq!(scan_id_column(&reopened).await, vec![1, 2]);
}
