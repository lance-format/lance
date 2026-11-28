// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, datatypes::Schema};
use lance_file::version::ConcreteFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::format::{Fragment, IndexFile, IndexMetadata, Manifest, ManifestBuildConfig};
use lance_table::io::commit::{
    CommitConfig, CommitError, CommitHandler, ConditionalPutCommitHandler, ManifestLocation,
    ManifestNamingScheme, ManifestWriter, migrate_scheme_to_v2,
};
use lance_table::io::manifest::read_manifest_indexes;
use lance_table::system_index::frag_reuse::{
    FragDigest, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
};
use lance_table::transaction::{Operation, Transaction};
use object_store::path::Path;
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStoreExt,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};
use roaring::{RoaringBitmap, RoaringTreemap};
use uuid::Uuid;

use super::{CommitHooks, CommitTransactionOptions, commit_transaction};
use crate::test_util::{test_commit_table, write_file};
use crate::{CommitBuilder, Table};

async fn append_transaction(table: &Table, name: &str, values: &[i32]) -> Transaction {
    let a = Int32Array::from_iter_values(values.iter().copied());
    let b = Int32Array::from(vec![Some(0); values.len()]);
    let batch = RecordBatch::try_new(
        Arc::new(ArrowSchema::from(table.schema())),
        vec![Arc::new(a), Arc::new(b)],
    )
    .unwrap();
    let file = write_file(
        &table.state.object_store,
        &table.state.base,
        name,
        &batch,
        ConcreteFileVersion::V2_1,
    )
    .await;
    Transaction::new(
        table.version_id(),
        Operation::Append {
            fragments: vec![Fragment {
                files: vec![file],
                physical_rows: Some(values.len()),
                ..Fragment::new(0)
            }],
        },
        None,
    )
}

#[derive(Debug, Default)]
struct ConflictOnce {
    attempts: AtomicU32,
    latest_resolutions: AtomicU32,
}

#[async_trait::async_trait]
impl CommitHandler for ConflictOnce {
    fn is_version_not_found_definitive(&self) -> bool {
        true
    }

    async fn resolve_latest_location(
        &self,
        base_path: &Path,
        object_store: &ObjectStore,
    ) -> lance_core::Result<ManifestLocation> {
        self.latest_resolutions.fetch_add(1, Ordering::Relaxed);
        ConditionalPutCommitHandler
            .resolve_latest_location(base_path, object_store)
            .await
    }

    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<lance_table::format::Transaction>,
    ) -> Result<ManifestLocation, CommitError> {
        if self.attempts.fetch_add(1, Ordering::Relaxed) == 0 {
            return Err(CommitError::CommitConflict);
        }
        ConditionalPutCommitHandler
            .commit(
                manifest,
                indices,
                base_path,
                object_store,
                manifest_writer,
                naming_scheme,
                transaction,
            )
            .await
    }
}

/// Model a lazy backend with two keys per listing request.
#[derive(Debug)]
struct PagedManifestStore {
    inner: Arc<dyn object_store::ObjectStore>,
    pages: Arc<AtomicU32>,
}

impl Display for PagedManifestStore {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "PagedManifestStore")
    }
}

#[async_trait::async_trait]
impl object_store::ObjectStore for PagedManifestStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        options: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, options).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        options: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, options).await
    }

    async fn get_opts(
        &self,
        location: &Path,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.inner.get_opts(location, options).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, object_store::Result<Path>>,
    ) -> BoxStream<'static, object_store::Result<Path>> {
        self.inner.delete_stream(locations)
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        let pages = self.pages.clone();
        let is_manifest_listing = prefix.is_some_and(|path| path.filename() == Some("_versions"));
        self.inner
            .list(prefix)
            .chunks(2)
            .flat_map(move |page| {
                if is_manifest_listing {
                    pages.fetch_add(1, Ordering::Relaxed);
                }
                futures::stream::iter(page)
            })
            .boxed()
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &Path,
        to: &Path,
        options: CopyOptions,
    ) -> object_store::Result<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

async fn v1_manifest_history() -> Table {
    let mut table = test_commit_table(4, 2).await;
    let v1_path = ManifestNamingScheme::V1.manifest_path(&table.state.base, 1);
    table
        .state
        .object_store
        .inner
        .rename_opts(
            &table.state.manifest_location.path,
            &v1_path,
            Default::default(),
        )
        .await
        .unwrap();
    table.state.manifest_location = table
        .state
        .commit_handler
        .resolve_latest_location(&table.state.base, &table.state.object_store)
        .await
        .unwrap();
    for version in 2..=5 {
        let transaction = append_transaction(&table, &format!("{version}.lance"), &[1]).await;
        table = CommitBuilder::new(Arc::new(table))
            .execute(transaction)
            .await
            .unwrap();
    }
    table
}

#[rstest::rstest]
#[case::up_to_date(5, false)]
#[case::catch_up(4, false)]
#[case::conditional(5, true)]
#[tokio::test]
async fn standalone_v1_commit_lists_history_once(
    #[case] read_version: u64,
    #[case] is_conditional: bool,
) {
    let mut table = v1_manifest_history().await;
    let pages = Arc::new(AtomicU32::new(0));
    let store = Arc::make_mut(&mut table.state.object_store);
    store.inner = Arc::new(PagedManifestStore {
        inner: store.inner.clone(),
        pages: pages.clone(),
    });
    let table = table.checkout_version(read_version).await.unwrap();
    let transaction = append_transaction(&table, "staged.lance", &[9]).await;
    let build_config = ManifestBuildConfig::default();
    let commit_config = CommitConfig::default();
    let mut options =
        CommitTransactionOptions::new(&build_config, &commit_config, ManifestNamingScheme::V1);
    if is_conditional {
        options = options.with_expected_latest_version(read_version);
    }
    let (manifest, location) = commit_transaction(
        &table,
        &table.state.object_store,
        table.state.commit_handler.as_ref(),
        &transaction,
        options,
        CommitHooks::default(),
    )
    .await
    .unwrap();

    assert_eq!(manifest.version, 6);
    assert_eq!(manifest.fragments.len(), 7);
    assert_eq!(location.naming_scheme, ManifestNamingScheme::V1);
    assert_eq!(
        pages.load(Ordering::Relaxed),
        3,
        "five V1 manifests need one scan of three pages"
    );
}

#[rstest::rstest]
#[case::same_bindings(true, true, 0)]
#[case::different_handler(false, true, 1)]
#[case::different_store(true, false, 1)]
#[tokio::test]
async fn standalone_migrated_commit_retains_v2_across_retries(
    #[case] same_handler: bool,
    #[case] same_store: bool,
    #[case] expected_resolutions: u32,
) {
    let mut table = v1_manifest_history().await;
    migrate_scheme_to_v2(&table.state.object_store, &table.state.base)
        .await
        .unwrap();
    let handler = Arc::new(ConflictOnce::default());
    if same_handler {
        table.state.commit_handler = handler.clone();
    }
    let other_store = table.state.object_store.as_ref().clone();
    let write_store = if same_store {
        table.state.object_store.as_ref()
    } else {
        &other_store
    };
    let transaction = append_transaction(&table, "staged.lance", &[9]).await;
    let build_config = ManifestBuildConfig::default();
    let commit_config = CommitConfig {
        num_retries: 2,
        ..Default::default()
    };
    let (manifest, location) = commit_transaction(
        &table,
        write_store,
        handler.as_ref(),
        &transaction,
        CommitTransactionOptions::new(&build_config, &commit_config, ManifestNamingScheme::V1),
        CommitHooks::default(),
    )
    .await
    .unwrap();
    assert_eq!(manifest.version, 6);
    assert_eq!(location.naming_scheme, ManifestNamingScheme::V2);
    assert_eq!(handler.attempts.load(Ordering::Relaxed), 2);
    assert_eq!(
        handler.latest_resolutions.load(Ordering::Relaxed),
        expected_resolutions
    );
}

#[tokio::test]
async fn standalone_strict_overwrite_after_migration_with_pruned_history() {
    let latest = v1_manifest_history().await;
    let stale = latest.checkout_version(4).await.unwrap();
    migrate_scheme_to_v2(&latest.state.object_store, &latest.state.base)
        .await
        .unwrap();
    latest
        .state
        .object_store
        .inner
        .delete(&ManifestNamingScheme::V2.manifest_path(&latest.state.base, 4))
        .await
        .unwrap();
    let transaction = Transaction::new(
        4,
        Operation::Overwrite {
            schema: stale.schema().clone(),
            fragments: stale.manifest.fragments.as_ref().clone(),
            config_upsert_values: None,
            initial_bases: None,
        },
        None,
    );
    let error = CommitBuilder::new(Arc::new(stale))
        .with_max_retries(0)
        .execute(transaction)
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CommitConflict { .. }), "{error}");
    assert!(error.to_string().contains("Failed to commit"), "{error}");
    assert!(
        !latest
            .state
            .object_store
            .inner
            .list(Some(&latest.state.base.clone().join("_versions")))
            .map_ok(|meta| ManifestLocation::try_from(meta).unwrap().naming_scheme)
            .try_any(|scheme| futures::future::ready(scheme == ManifestNamingScheme::V1))
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn standalone_append_rebases_and_retries_manifest_conflict() {
    let initial = test_commit_table(2, 1).await;
    let staged = append_transaction(&initial, "staged.lance", &[3, 4]).await;
    let concurrent = append_transaction(&initial, "concurrent.lance", &[2]).await;
    let competing = CommitBuilder::new(Arc::new(initial.clone()))
        .execute(concurrent)
        .await
        .unwrap();
    assert_eq!(competing.version_id(), 2);

    let handler = Arc::new(ConflictOnce::default());
    let committed = CommitBuilder::new(Arc::new(initial.clone()))
        .with_commit_handler(handler.clone())
        .with_max_retries(2)
        .execute(staged.clone())
        .await
        .unwrap();
    assert_eq!(handler.attempts.load(Ordering::Relaxed), 2);
    assert_eq!(committed.version_id(), 3);
    assert_eq!(initial.version_id(), 1);
    let reopened = crate::TableBuilder::from_uri(initial.uri())
        .with_session(initial.session())
        .load()
        .await
        .unwrap();
    assert_eq!(reopened.version_id(), 3);
    assert_eq!(reopened.count_fragments(), 3);
    assert_eq!(reopened.count_rows().await.unwrap(), 5);
    let stored = reopened
        .read_version_transaction(3)
        .await
        .unwrap()
        .transaction
        .unwrap();
    assert_eq!(stored.uuid, staged.uuid);
    let actual = reopened
        .take(
            &[0, 1, 2, 3, 4],
            &reopened.schema().project(&["a"]).unwrap(),
        )
        .await
        .unwrap();
    let actual = actual["a"].as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(actual, &Int32Array::from(vec![0, 1, 2, 3, 4]));
}

#[tokio::test]
async fn standalone_conflicting_overwrite_leaves_winner() {
    let initial = Arc::new(test_commit_table(4, 2).await);
    let make_overwrite = || {
        Transaction::new(
            1,
            Operation::Overwrite {
                schema: initial.schema().clone(),
                fragments: Vec::new(),
                config_upsert_values: None,
                initial_bases: None,
            },
            None,
        )
    };
    let winner = make_overwrite();
    let latest = CommitBuilder::new(initial.clone())
        .execute(winner.clone())
        .await
        .unwrap();
    let error = CommitBuilder::new(initial.clone())
        .execute(make_overwrite())
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("Overwrite"), "{error}");
    assert_eq!(latest.latest_version_id().await.unwrap(), 2);
    assert_eq!(
        latest
            .read_version_transaction(2)
            .await
            .unwrap()
            .transaction
            .unwrap()
            .uuid,
        winner.uuid,
    );
}

fn opaque_segment(name: &str, fragment_bitmap: Option<RoaringBitmap>) -> IndexMetadata {
    IndexMetadata {
        uuid: Uuid::new_v4(),
        name: name.to_string(),
        fields: vec![0],
        covering_fields: Vec::new(),
        dataset_version: 1,
        fragment_bitmap,
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "type.example/OtherWriterIndex".into(),
            value: vec![0x91, 0xff, 1],
        })),
        index_version: 73,
        created_at: None,
        base_id: None,
        files: Some(vec![IndexFile {
            path: "opaque.bin".into(),
            size_bytes: 41,
            file_metadata_size_bytes: None,
        }]),
        clustering_id: None,
        metadata: HashMap::from([("writer".to_string(), "example".to_string())]),
    }
}

async fn read_stored_indices(table: &Table) -> Vec<IndexMetadata> {
    read_manifest_indexes(
        &table.state.object_store,
        table.manifest_location(),
        table.manifest(),
    )
    .await
    .unwrap()
}

#[rstest::rstest]
#[case::without_fragment_reuse(false)]
#[case::with_v0_fragment_reuse(true)]
#[tokio::test]
async fn standalone_append_preserves_unknown_index_entries(#[case] has_v0_fri: bool) {
    let table = test_commit_table(4, 2).await;
    let indexed_fragment = if has_v0_fri { 7 } else { 0 };
    let unknown = opaque_segment(
        "opaque",
        Some(RoaringBitmap::from_iter([indexed_fragment, 1])),
    );
    let mut stored = vec![unknown.clone()];
    if has_v0_fri {
        let old_rows = RoaringTreemap::from_iter(
            (0..2).map(|offset| u64::from(RowAddress::new_from_parts(indexed_fragment, offset))),
        );
        let mut changed_row_addrs = Vec::new();
        old_rows.serialize_into(&mut changed_row_addrs).unwrap();
        let details = FragReuseIndexDetails {
            versions: vec![FragReuseVersion {
                dataset_version: 1,
                groups: vec![FragReuseGroup {
                    changed_row_addrs,
                    old_frags: vec![FragDigest {
                        id: u64::from(indexed_fragment),
                        physical_rows: 2,
                        num_deleted_rows: 0,
                    }],
                    new_frags: vec![FragDigest::from(&table.manifest().fragments[0])],
                }],
            }],
        };
        let mut fri = crate::frag_reuse::build_frag_reuse_index_metadata(
            &table,
            None,
            details,
            RoaringBitmap::from_iter([indexed_fragment, 0]),
        )
        .await
        .unwrap();
        // Manifest index timestamps have millisecond precision.
        fri.created_at = chrono::DateTime::from_timestamp_millis(1);
        stored.push(fri);
    }
    let indexed = CommitBuilder::new(Arc::new(table))
        .execute(Transaction::new(
            1,
            Operation::CreateIndex {
                new_indices: stored.clone(),
                removed_indices: vec![],
            },
            None,
        ))
        .await
        .unwrap();
    assert_eq!(read_stored_indices(&indexed).await, stored);
    let computed = super::load_all_indices(&indexed, None).await.unwrap();
    let computed = computed
        .iter()
        .find(|index| index.uuid == unknown.uuid)
        .unwrap();
    assert_eq!(
        computed.fragment_bitmap,
        Some(RoaringBitmap::from_iter([0, 1]))
    );
    assert_eq!(
        indexed.load_index_metadata().await.unwrap().as_ref(),
        &stored
    );

    let append = append_transaction(&indexed, "append.lance", &[4, 5]).await;
    let committed = CommitBuilder::new(Arc::new(indexed))
        .execute(append)
        .await
        .unwrap();
    assert_eq!(committed.version_id(), 3);
    assert_eq!(committed.count_rows().await.unwrap(), 6);
    assert_eq!(read_stored_indices(&committed).await, stored);
}

#[tokio::test]
async fn standalone_refuses_multisegment_vector_validation_without_index_hook() {
    let schema = ArrowSchema::new(vec![ArrowField::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(ArrowField::new("item", DataType::Float32, true)),
            2,
        ),
        true,
    )]);
    let table = CommitBuilder::new("memory://")
        .execute(Transaction::new(
            0,
            Operation::Overwrite {
                schema: Schema::try_from(&schema).unwrap(),
                fragments: vec![],
                config_upsert_values: None,
                initial_bases: None,
            },
            None,
        ))
        .await
        .unwrap();
    let version = table.version_id();
    let error = CommitBuilder::new(Arc::new(table.clone()))
        .execute(Transaction::new(
            version,
            Operation::CreateIndex {
                new_indices: vec![
                    opaque_segment("vector_index", Some(RoaringBitmap::new())),
                    opaque_segment("vector_index", Some(RoaringBitmap::new())),
                ],
                removed_indices: vec![],
            },
            None,
        ))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NotSupported { .. }), "{error}");
    assert!(
        error
            .to_string()
            .contains("multiple segments of index 'vector_index'"),
        "{error}"
    );
    assert_eq!(table.latest_version_id().await.unwrap(), version);
}
