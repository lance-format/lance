// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, datatypes::Schema};
use lance_file::version::ConcreteFileVersion;
use lance_io::object_store::ObjectStore;
use lance_table::format::{Fragment, IndexFile, IndexMetadata, Manifest};
use lance_table::io::commit::{
    CommitError, CommitHandler, ConditionalPutCommitHandler, ManifestLocation,
    ManifestNamingScheme, ManifestWriter,
};
use lance_table::io::manifest::read_manifest_indexes;
use lance_table::system_index::frag_reuse::{
    FragDigest, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
};
use lance_table::transaction::{Operation, Transaction};
use object_store::path::Path;
use roaring::{RoaringBitmap, RoaringTreemap};
use uuid::Uuid;

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
}

#[async_trait::async_trait]
impl CommitHandler for ConflictOnce {
    fn is_version_not_found_definitive(&self) -> bool {
        true
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
        local_fragment_aliases: Default::default(),
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
