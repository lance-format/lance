// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Clustering definitions and generic object metadata through the real
//! dataset lifecycle: same-object operations keep them, new objects do not
//! inherit them, and restore, branch and clone carry them.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    FixedSizeListArray, Float32Array, Int32Array, RecordBatch, RecordBatchIterator,
    RecordBatchReader,
};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance_core::Error;
use lance_core::utils::tempfile::TempStrDir;
use lance_index::IndexType;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{BuiltinIndexType, ScalarIndexParams};
use lance_linalg::distance::MetricType;
use lance_table::feature_flags::{FLAG_NON_REUSABLE_FIELD_IDS, FLAG_OBJECT_METADATA};
use lance_table::format::{ClusteringDefinition, IndexMetadata, Manifest};
use lance_table::io::manifest::read_manifest_indexes;
use rstest::rstest;
use uuid::Uuid;

use crate::Dataset;
use crate::dataset::cleanup::CleanupPolicyBuilder;
use crate::dataset::optimize::{CompactionOptions, compact_files};
use crate::dataset::transaction::{Operation, Transaction, UpdateMap, UpdateMapEntry};
use crate::dataset::write::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
use crate::dataset::{ManifestWriteConfig, NewColumnTransform, write_manifest_file};
use crate::index::DatasetIndexExt;
use crate::index::vector::VectorIndexParams;

const PROVIDER: &str = "org.example.cluster";
const BUCKET_KEY: &str = "org.example.cluster.bucket";
const UNKNOWN_KEY: &str = "org.example.unknown";

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "i",
        DataType::Int32,
        false,
    )]))
}

fn rows(range: std::ops::Range<i32>) -> impl RecordBatchReader + Send + 'static {
    let batch = RecordBatch::try_new(
        schema(),
        vec![Arc::new(Int32Array::from_iter_values(range))],
    )
    .unwrap();
    RecordBatchIterator::new(vec![Ok(batch)], schema())
}

/// Two fragments of fifty rows each.
async fn write_dataset(uri: &str) -> Dataset {
    Dataset::write(
        rows(0..100),
        uri,
        Some(WriteParams {
            max_rows_per_file: 50,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

fn definition(id: u32) -> ClusteringDefinition {
    ClusteringDefinition {
        id,
        columns: vec![0],
        provider: PROVIDER.into(),
        version: 1,
    }
}

fn object_metadata(bucket: &str) -> HashMap<String, String> {
    HashMap::from([
        (BUCKET_KEY.to_string(), bucket.to_string()),
        (UNKNOWN_KEY.to_string(), "opaque".to_string()),
        ("org.example.empty".to_string(), String::new()),
    ])
}

/// Commit a hand-edited manifest. No transaction allocates definitions yet,
/// so tests stamp them (and object metadata) the way a provider's commit
/// eventually will, then exercise the real operations on top.
async fn commit_edited_manifest(
    dataset: &Dataset,
    edit: impl FnOnce(&mut Manifest, &mut Vec<IndexMetadata>),
) -> Dataset {
    let mut manifest = dataset.manifest().clone();
    let mut indices =
        read_manifest_indexes(&dataset.object_store, &dataset.manifest_location, &manifest)
            .await
            .unwrap();
    edit(&mut manifest, &mut indices);
    manifest.version += 1;
    write_manifest_file(
        dataset.object_store.as_ref(),
        dataset.commit_handler.as_ref(),
        &dataset.base,
        &mut manifest,
        Some(indices),
        &ManifestWriteConfig {
            auto_set_feature_flags: true,
            timestamp: None,
            use_stable_row_ids: false,
            use_legacy_format: None,
            storage_format: None,
            disable_transaction_file: false,
            migration_next_row_id: None,
            activate_non_reusable_field_ids: false,
            tagged_frag_reuse_trim: false,
            clustering_definitions: None,
        },
        dataset.manifest_location.naming_scheme,
        None,
        false,
    )
    .await
    .unwrap();
    Dataset::open(dataset.uri()).await.unwrap()
}

fn stamp_fragment(manifest: &mut Manifest, position: usize, clustering_id: u32, bucket: &str) {
    let fragment = &mut Arc::make_mut(&mut manifest.fragments)[position];
    fragment.clustering_id = Some(clustering_id);
    fragment.metadata = object_metadata(bucket);
}

fn fragment_by_id(dataset: &Dataset, id: u64) -> &lance_table::format::Fragment {
    dataset
        .fragments()
        .iter()
        .find(|fragment| fragment.id == id)
        .unwrap_or_else(|| panic!("fragment {id} missing"))
}

#[tokio::test]
async fn same_object_operations_keep_fragment_metadata_and_set_the_writer_flag() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let mut dataset = commit_edited_manifest(&dataset, |manifest, _| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
        stamp_fragment(manifest, 0, 1, "3");
    })
    .await;
    assert_ne!(
        dataset.manifest().writer_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
    assert_eq!(
        dataset.manifest().reader_feature_flags & FLAG_OBJECT_METADATA,
        0
    );

    // Delete: same fragment, new deletion file.
    dataset.delete("i < 5").await.unwrap();
    let fragment = fragment_by_id(&dataset, 0);
    assert!(fragment.deletion_file.is_some());
    assert_eq!(fragment.clustering_id, Some(1));
    assert_eq!(fragment.metadata, object_metadata("3"));
    assert_eq!(fragment_by_id(&dataset, 1).clustering_id, None);
    assert!(fragment_by_id(&dataset, 1).metadata.is_empty());

    // Add a column: same fragments, merged files.
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("j".to_string(), "i + 1".to_string())]),
            None,
            None,
        )
        .await
        .unwrap();
    let fragment = fragment_by_id(&dataset, 0);
    assert_eq!(fragment.files.len(), 2);
    assert_eq!(fragment.clustering_id, Some(1));
    assert_eq!(fragment.metadata, object_metadata("3"));
    assert_eq!(dataset.manifest().clustering, vec![definition(1)]);
    assert_eq!(dataset.manifest().max_clustering_id, 1);
    assert_ne!(
        dataset.manifest().writer_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
    dataset.validate().await.unwrap();
}

#[tokio::test]
async fn compaction_outputs_do_not_inherit_fragment_metadata() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let mut dataset = commit_edited_manifest(&dataset, |manifest, _| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
        stamp_fragment(manifest, 0, 1, "3");
        stamp_fragment(manifest, 1, 1, "3");
    })
    .await;

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1_000,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.fragments().len(), 1);
    let output = &dataset.fragments()[0];
    assert_eq!(output.clustering_id, None);
    assert!(output.metadata.is_empty());
    // The definitions and the high-water mark outlive the fragments that used them.
    assert_eq!(dataset.manifest().clustering, vec![definition(1)]);
    assert_eq!(dataset.manifest().max_clustering_id, 1);
    assert_ne!(
        dataset.manifest().writer_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
}

#[tokio::test]
async fn append_commit_validates_fragment_references() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let dataset = commit_edited_manifest(&dataset, |manifest, _| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
    })
    .await;

    let stamped = |clustering_id: u32| {
        let dataset = dataset.clone();
        async move {
            let mut transaction = InsertBuilder::new(Arc::new(dataset.clone()))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                })
                .execute_uncommitted(vec![
                    RecordBatch::try_new(
                        schema(),
                        vec![Arc::new(Int32Array::from_iter_values(100..110))],
                    )
                    .unwrap(),
                ])
                .await
                .unwrap();
            let Operation::Append { fragments } = &mut transaction.operation else {
                unreachable!()
            };
            for fragment in fragments {
                fragment.clustering_id = Some(clustering_id);
                fragment.metadata = object_metadata("5");
            }
            CommitBuilder::new(Arc::new(dataset))
                .execute(transaction)
                .await
        }
    };

    let error = stamped(9).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("references clustering definition 9"),
        "{error}"
    );

    let committed = stamped(1).await.unwrap();
    let appended = committed.fragments().last().unwrap();
    assert_eq!(appended.clustering_id, Some(1));
    assert_eq!(appended.metadata, object_metadata("5"));
}

#[tokio::test]
async fn restore_keeps_the_high_water_mark_and_merges_definitions() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await; // version 1: nothing allocated
    let before_allocation = dataset.version().version;

    let dataset = commit_edited_manifest(&dataset, |manifest, _| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
        manifest
            .table_metadata
            .insert("org.example.cluster.active_id".into(), "1".into());
        stamp_fragment(manifest, 0, 1, "3");
    })
    .await;
    let first_definition = dataset.version().version;

    let dataset = commit_edited_manifest(&dataset, |manifest, _| {
        manifest.clustering.push(definition(2));
        manifest.max_clustering_id = 2;
        manifest
            .table_metadata
            .insert("org.example.cluster.active_id".into(), "2".into());
    })
    .await;

    // Restore to before any allocation: the historical selection comes back
    // (none), the mark does not rewind, and every definition this lineage
    // handed out stays defined.
    let mut restored = dataset.checkout_version(before_allocation).await.unwrap();
    restored.restore().await.unwrap();
    assert!(
        !restored
            .manifest()
            .table_metadata
            .contains_key("org.example.cluster.active_id")
    );
    assert_eq!(restored.manifest().max_clustering_id, 2);
    assert_eq!(
        restored.manifest().clustering,
        vec![definition(1), definition(2)]
    );
    assert_eq!(restored.fragments()[0].clustering_id, None);
    assert_ne!(
        restored.manifest().writer_feature_flags & FLAG_OBJECT_METADATA,
        0
    );

    // Allocating again continues above the mark; id 1 and 2 are never reused.
    let restored = commit_edited_manifest(&restored, |manifest, _| {
        manifest.clustering.push(definition(3));
        manifest.max_clustering_id = 3;
    })
    .await;
    assert_eq!(restored.manifest().max_clustering_id, 3);

    // Restore to the first definition's version: its selection and its
    // fragment references come back, with the later definitions retained.
    let mut restored = restored.checkout_version(first_definition).await.unwrap();
    restored.restore().await.unwrap();
    assert_eq!(
        restored.manifest().table_metadata["org.example.cluster.active_id"],
        "1"
    );
    assert_eq!(restored.manifest().max_clustering_id, 3);
    assert_eq!(
        restored.manifest().clustering,
        vec![definition(1), definition(2), definition(3)]
    );
    assert_eq!(restored.fragments()[0].clustering_id, Some(1));
    assert_eq!(restored.fragments()[0].metadata, object_metadata("3"));
}

#[tokio::test]
async fn restore_rejects_a_definition_that_changed_meaning() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let dataset = commit_edited_manifest(&dataset, |manifest, _| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
    })
    .await;
    let original = dataset.version().version;
    // Simulate a writer that redefined id 1 in place, which the contract forbids.
    let dataset = commit_edited_manifest(&dataset, |manifest, _| {
        manifest.clustering[0].provider = "org.example.other".into();
    })
    .await;

    let mut restored = dataset.checkout_version(original).await.unwrap();
    let error = restored.restore().await.unwrap_err();
    assert!(
        error.to_string().contains("differs between manifests"),
        "{error}"
    );
}

#[tokio::test]
async fn branch_and_shallow_clone_carry_definitions_and_metadata() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let mut dataset = commit_edited_manifest(&dataset, |manifest, _| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
        stamp_fragment(manifest, 0, 1, "3");
    })
    .await;
    let version = dataset.version().version;

    let branch = dataset.create_branch("b", version, None).await.unwrap();
    assert_eq!(branch.manifest().clustering, vec![definition(1)]);
    assert_eq!(branch.manifest().max_clustering_id, 1);
    assert_eq!(branch.fragments()[0].clustering_id, Some(1));
    assert_eq!(branch.fragments()[0].metadata, object_metadata("3"));

    let clone_dir = TempStrDir::default();
    let clone = dataset
        .shallow_clone(&clone_dir, version, None)
        .await
        .unwrap();
    assert_eq!(clone.manifest().clustering, vec![definition(1)]);
    assert_eq!(clone.manifest().max_clustering_id, 1);
    assert_eq!(clone.fragments()[0].clustering_id, Some(1));
    assert_eq!(clone.fragments()[0].metadata, object_metadata("3"));
    assert_ne!(
        clone.manifest().writer_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
}

async fn named_segments(dataset: &Dataset, name: &str) -> Vec<IndexMetadata> {
    read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        dataset.manifest(),
    )
    .await
    .unwrap()
    .into_iter()
    .filter(|index| index.name == name)
    .collect()
}

/// A remapped segment is the same segment under new addresses and keeps its
/// metadata, even though it gets a new UUID; a merged segment is a new
/// object and carries only what its creator set.
#[tokio::test]
async fn index_segment_metadata_survives_remap_but_not_merge() {
    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let dataset = commit_edited_manifest(&dataset, |manifest, indices| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
        let segment = indices.iter_mut().find(|idx| idx.name == "idx").unwrap();
        segment.clustering_id = Some(1);
        segment.metadata = object_metadata("3");
    })
    .await;
    let original_uuid = named_segments(&dataset, "idx").await[0].uuid;

    // Compaction remaps the BTree segment inline: same segment, new UUID.
    let mut dataset = dataset;
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1_000,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    let remapped = named_segments(&dataset, "idx").await;
    assert_eq!(remapped.len(), 1);
    assert_ne!(remapped[0].uuid, original_uuid);
    assert_eq!(remapped[0].clustering_id, Some(1));
    assert_eq!(remapped[0].metadata, object_metadata("3"));

    // New rows and an appended delta segment: the new segment carries nothing.
    Dataset::write(
        rows(100..150),
        dataset.uri(),
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.checkout_latest().await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    let mut segments = named_segments(&dataset, "idx").await;
    segments.sort_by_key(|segment| segment.clustering_id.is_none());
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].clustering_id, Some(1));
    assert_eq!(segments[0].metadata, object_metadata("3"));
    assert_eq!(segments[1].clustering_id, None);
    assert!(segments[1].metadata.is_empty());

    // Merging both into one segment creates a new object: no inheritance.
    dataset
        .optimize_indices(&OptimizeOptions::default().num_indices_to_merge(Some(2)))
        .await
        .unwrap();
    let merged = named_segments(&dataset, "idx").await;
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].clustering_id, None);
    assert!(merged[0].metadata.is_empty());
    // The definition and mark remain even with nothing referencing them.
    assert_eq!(dataset.manifest().clustering, vec![definition(1)]);
    assert_eq!(dataset.manifest().max_clustering_id, 1);
}

/// Every merge builder creates a new segment, and none may carry an input's
/// label forward. Both inputs are labelled so the result does not depend on
/// which input a builder happens to start from.
#[rstest]
#[case::btree(IndexType::BTree)]
#[case::bitmap(IndexType::Bitmap)]
#[case::zonemap(IndexType::ZoneMap)]
#[tokio::test]
async fn merged_segments_do_not_inherit_either_input_label(#[case] index_type: IndexType) {
    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    let params = match index_type {
        IndexType::Bitmap => ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
        IndexType::ZoneMap => ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
        _ => ScalarIndexParams::default(),
    };
    dataset
        .create_index(&["i"], index_type, Some("idx".into()), &params, true)
        .await
        .unwrap();
    Dataset::write(
        rows(100..150),
        dataset.uri(),
        Some(WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.checkout_latest().await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(named_segments(&dataset, "idx").await.len(), 2);

    let mut dataset = commit_edited_manifest(&dataset, |manifest, indices| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
        for (position, segment) in indices
            .iter_mut()
            .filter(|segment| segment.name == "idx")
            .enumerate()
        {
            segment.clustering_id = Some(1);
            segment.metadata = object_metadata(&position.to_string());
        }
    })
    .await;

    dataset
        .optimize_indices(&OptimizeOptions::default().num_indices_to_merge(Some(2)))
        .await
        .unwrap();
    let merged = named_segments(&dataset, "idx").await;
    assert_eq!(merged.len(), 1, "{index_type:?}");
    assert_eq!(merged[0].clustering_id, None, "{index_type:?}");
    assert!(merged[0].metadata.is_empty(), "{index_type:?}");
}

/// The explicit remap route (`remap_column_index`) builds a new segment entry
/// for the same content: the nonzero reference and the map survive the new UUID.
#[tokio::test]
async fn explicit_remap_keeps_reference_and_metadata() {
    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let mut dataset = commit_edited_manifest(&dataset, |manifest, indices| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
        let segment = indices.iter_mut().find(|idx| idx.name == "idx").unwrap();
        segment.clustering_id = Some(1);
        segment.metadata = object_metadata("3");
    })
    .await;
    let before = named_segments(&dataset, "idx").await[0].clone();

    // Deferred remap leaves the segment addressing the old fragments; the
    // explicit remap then re-addresses it under a new UUID.
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1_000,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(named_segments(&dataset, "idx").await[0].uuid, before.uuid);
    crate::dataset::optimize::remapping::remap_column_index(
        &mut dataset,
        &["i"],
        Some("idx".into()),
    )
    .await
    .unwrap();
    let after = named_segments(&dataset, "idx").await;
    assert_eq!(after.len(), 1);
    assert_ne!(after[0].uuid, before.uuid);
    assert_eq!(after[0].clustering_id, Some(1));
    assert_eq!(after[0].metadata, object_metadata("3"));
}

/// A table whose ONLY clustering feature is a map on one index segment keeps
/// its writer fence through the clone and branch paths, which publish the
/// manifest without `build_manifest`.
#[tokio::test]
async fn index_only_metadata_keeps_its_writer_fence_through_clone_and_branch() {
    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let mut dataset = commit_edited_manifest(&dataset, |_, indices| {
        let segment = indices.iter_mut().find(|idx| idx.name == "idx").unwrap();
        segment.metadata = object_metadata("3");
    })
    .await;
    assert!(dataset.manifest().clustering.is_empty());
    assert_eq!(dataset.manifest().max_clustering_id, 0);
    assert!(dataset.fragments().iter().all(|f| f.metadata.is_empty()));
    assert_ne!(
        dataset.manifest().writer_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
    let version = dataset.version().version;

    let clone_dir = TempStrDir::default();
    let clone = dataset
        .shallow_clone(&clone_dir, version, None)
        .await
        .unwrap();
    assert_ne!(
        clone.manifest().writer_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
    assert_eq!(
        clone.manifest().reader_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
    assert_eq!(
        named_segments(&clone, "idx").await[0].metadata,
        object_metadata("3")
    );

    let branch = dataset.create_branch("b", version, None).await.unwrap();
    assert_ne!(
        branch.manifest().writer_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
    assert_eq!(
        branch.manifest().reader_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
    assert_eq!(
        named_segments(&branch, "idx").await[0].metadata,
        object_metadata("3")
    );
}

#[tokio::test]
async fn metadata_only_on_an_index_segment_sets_the_writer_flag() {
    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let mut dataset = commit_edited_manifest(&dataset, |_, indices| {
        let segment = indices.iter_mut().find(|idx| idx.name == "idx").unwrap();
        segment.metadata.insert(UNKNOWN_KEY.into(), "opaque".into());
    })
    .await;
    assert!(dataset.manifest().clustering.is_empty());
    assert_eq!(dataset.manifest().max_clustering_id, 0);

    // The next real commit derives the bit from the final index section.
    dataset.delete("i < 1").await.unwrap();
    assert_ne!(
        dataset.manifest().writer_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
    assert_eq!(
        dataset.manifest().reader_feature_flags & FLAG_OBJECT_METADATA,
        0
    );
    assert_eq!(
        named_segments(&dataset, "idx").await[0].metadata[UNKNOWN_KEY],
        "opaque"
    );
}

// ---- complete-list submission alongside UpdateConfig (commit-time input) ---

const ACTIVE_KEY: &str = "org.example.cluster.active_id";

fn spec_key(id: u32) -> String {
    format!("{PROVIDER}.{id}.spec")
}

/// A plain `UpdateConfig` with the provider's key upserts, plus the writer's
/// complete desired definitions list to commit alongside it.
fn submission(
    read_version: u64,
    desired: Vec<ClusteringDefinition>,
    metadata: Vec<(String, Option<String>)>,
) -> (Vec<ClusteringDefinition>, Transaction) {
    let transaction = Transaction::new(
        read_version,
        Operation::UpdateConfig {
            config_updates: None,
            table_metadata_updates: Some(UpdateMap {
                update_entries: metadata.into_iter().map(UpdateMapEntry::from).collect(),
                replace: false,
            }),
            schema_metadata_updates: None,
            field_metadata_updates: HashMap::new(),
        },
        None,
    );
    (desired, transaction)
}

async fn commit(
    dataset: &Dataset,
    (desired, transaction): (Vec<ClusteringDefinition>, Transaction),
) -> crate::Result<Dataset> {
    CommitBuilder::new(Arc::new(dataset.clone()))
        .with_clustering_definitions(desired)
        .execute(transaction)
        .await
}

fn ids(dataset: &Dataset) -> Vec<u32> {
    dataset.manifest().clustering.iter().map(|d| d.id).collect()
}

fn field_ids(dataset: &Dataset) -> Vec<(String, i32)> {
    dataset
        .schema()
        .fields
        .iter()
        .map(|field| (field.name.clone(), field.id))
        .collect()
}

async fn write_named_column(uri: &str, name: &str, mode: WriteMode) -> Dataset {
    write_named_columns(uri, &[name], mode).await
}

/// Add `region` as its own data file, so dropping it releases its field id on
/// a reusable-id table.
async fn with_region_column(mut dataset: Dataset) -> Dataset {
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("region".to_string(), "a + 1".to_string())]),
            None,
            None,
        )
        .await
        .unwrap();
    dataset
}

/// Int32 columns in the given order, so the last name gets the highest id.
async fn write_named_columns(uri: &str, names: &[&str], mode: WriteMode) -> Dataset {
    let schema = Arc::new(ArrowSchema::new(
        names
            .iter()
            .map(|name| ArrowField::new(*name, DataType::Int32, false))
            .collect::<Vec<_>>(),
    ));
    let batch = RecordBatch::try_new(
        schema.clone(),
        names
            .iter()
            .map(|_| Arc::new(Int32Array::from_iter_values(0..10)) as _)
            .collect(),
    )
    .unwrap();
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema),
        uri,
        Some(WriteParams {
            mode,
            ..Default::default()
        }),
    )
    .await
    .unwrap()
}

/// Allocating the first definition on a reusable-id table activates the
/// non-reusable field-id contract in the same version: definitions, both
/// watermarks, the metadata patch and the writer bit land together, and the
/// existing field ids are untouched. Later submissions, removals and a clear
/// leave the activation and its mark alone.
#[tokio::test]
async fn the_first_definition_activates_non_reusable_field_ids_atomically() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    assert!(!dataset.manifest().uses_non_reusable_field_ids());
    let before = dataset.manifest().version;
    let ids_before = field_ids(&dataset);
    let mark_before = dataset.manifest().max_field_id();

    let dataset = commit(
        &dataset,
        submission(
            before,
            vec![definition(1)],
            vec![("org.example.note".to_string(), Some("x".to_string()))],
        ),
    )
    .await
    .unwrap();
    assert_eq!(
        dataset.manifest().version,
        before + 1,
        "exactly one version"
    );
    let check = |dataset: &Dataset| {
        let manifest = dataset.manifest();
        assert_eq!(ids(dataset), vec![1]);
        assert_eq!(manifest.max_clustering_id, 1);
        assert_eq!(manifest.max_allocated_field_id, Some(mark_before));
        assert_ne!(
            manifest.writer_feature_flags & FLAG_NON_REUSABLE_FIELD_IDS,
            0
        );
        assert_eq!(
            manifest.reader_feature_flags & FLAG_NON_REUSABLE_FIELD_IDS,
            0
        );
        assert_eq!(manifest.table_metadata["org.example.note"], "x");
        assert_eq!(field_ids(dataset), ids_before);
    };
    check(&dataset);
    check(&Dataset::open(&dir).await.unwrap());

    // Already active: a second allocation adds a version, not a new mark.
    let dataset = commit(
        &dataset,
        submission(
            dataset.manifest().version,
            vec![definition(1), definition(2)],
            vec![],
        ),
    )
    .await
    .unwrap();
    assert_eq!(dataset.manifest().version, before + 2);
    assert_eq!(dataset.manifest().max_allocated_field_id, Some(mark_before));

    // Removal and a clear keep the activation.
    let dataset = commit(
        &dataset,
        submission(dataset.manifest().version, vec![], vec![]),
    )
    .await
    .unwrap();
    assert!(dataset.manifest().clustering.is_empty());
    assert_eq!(dataset.manifest().max_allocated_field_id, Some(mark_before));
    assert_ne!(
        dataset.manifest().writer_feature_flags & FLAG_NON_REUSABLE_FIELD_IDS,
        0
    );
}

/// A refused first submission publishes nothing: no activation, no
/// definitions, no mark, no metadata. An empty target on an unconfigured table
/// commits its metadata without activating anything.
#[tokio::test]
async fn a_refused_first_submission_activates_nothing() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let version = dataset.manifest().version;
    let note = ("org.example.note".to_string(), Some("x".to_string()));

    let error = commit(
        &dataset,
        submission(
            version,
            vec![ClusteringDefinition {
                columns: vec![7],
                ..definition(1)
            }],
            vec![note.clone()],
        ),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("not in the schema"), "{error}");
    let error = commit(
        &dataset,
        submission(version, vec![definition(2)], vec![note.clone()]),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, Error::CommitConflict { .. }), "{error:?}");

    let reopened = Dataset::open(&dir).await.unwrap();
    assert_eq!(reopened.manifest().version, version);
    assert!(!reopened.manifest().uses_non_reusable_field_ids());
    assert_eq!(
        reopened.manifest().writer_feature_flags & FLAG_NON_REUSABLE_FIELD_IDS,
        0
    );
    assert!(reopened.manifest().clustering.is_empty());
    assert_eq!(reopened.manifest().max_clustering_id, 0);
    assert!(
        !reopened
            .manifest()
            .table_metadata
            .contains_key("org.example.note")
    );

    let dataset = commit(&dataset, submission(version, vec![], vec![note]))
        .await
        .unwrap();
    assert_eq!(dataset.manifest().table_metadata["org.example.note"], "x");
    assert!(!dataset.manifest().uses_non_reusable_field_ids());
}

/// A table that already uses non-reusable field ids keeps its mark: the
/// allocation neither resets nor re-activates it.
#[tokio::test]
async fn an_already_activated_table_keeps_its_field_id_mark() {
    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    dataset.migrate_to_non_reusable_field_ids().await.unwrap();
    let version = dataset.manifest().version;
    let mark = dataset.manifest().max_allocated_field_id;
    assert!(mark.is_some());

    let dataset = commit(&dataset, submission(version, vec![definition(1)], vec![]))
        .await
        .unwrap();
    assert_eq!(dataset.manifest().version, version + 1);
    assert_eq!(dataset.manifest().max_allocated_field_id, mark);
    assert_eq!(ids(&dataset), vec![1]);
}

/// With the contract active, a definition can lose its key field but never
/// silently name another one: a column added after a drop and the fields of
/// an overwrite, unrelated or same-named, all get ids above the mark.
#[tokio::test]
async fn definitions_keep_their_field_identity_across_drop_add_and_overwrite() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let mut dataset = commit(
        &dataset,
        submission(dataset.manifest().version, vec![definition(1)], vec![]),
    )
    .await
    .unwrap();
    let key_id = dataset.schema().field("i").unwrap().id;
    assert_eq!(definition(1).columns, vec![key_id]);

    // Drop the keyed column, then add one: the new column's id is fresh.
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("j".to_string(), "i + 1".to_string())]),
            None,
            None,
        )
        .await
        .unwrap();
    let mark_after_add = dataset.manifest().max_field_id();
    dataset.drop_columns(&["i"]).await.unwrap();
    dataset
        .add_columns(
            NewColumnTransform::SqlExpressions(vec![("k".to_string(), "j + 1".to_string())]),
            None,
            None,
        )
        .await
        .unwrap();
    let k_id = dataset.schema().field("k").unwrap().id;
    assert!(
        k_id > mark_after_add,
        "k got {k_id}, mark was {mark_after_add}"
    );
    assert_ne!(k_id, key_id);
    assert!(dataset.schema().field_by_id(key_id).is_none());
    assert_eq!(dataset.manifest().clustering, vec![definition(1)]);

    // Overwrite with an unrelated schema, then with the original one: both
    // allocate above the mark; the definitions and marks survive.
    let mark = dataset.manifest().max_field_id();
    let overwritten = write_named_column(&dir, "user_id", WriteMode::Overwrite).await;
    let user_id = overwritten.schema().field("user_id").unwrap().id;
    assert!(user_id > mark, "user_id got {user_id}, mark was {mark}");
    assert_eq!(overwritten.manifest().clustering, vec![definition(1)]);
    assert_eq!(overwritten.manifest().max_clustering_id, 1);
    assert!(overwritten.manifest().uses_non_reusable_field_ids());

    let again = write_named_column(&dir, "i", WriteMode::Overwrite).await;
    let i_again = again.schema().field("i").unwrap().id;
    assert!(i_again > user_id, "i got {i_again} after {user_id}");
    assert_ne!(
        i_again, key_id,
        "the same name does not get the old id back"
    );
    assert_eq!(again.manifest().clustering, vec![definition(1)]);
}

/// Writer A plans the first definition from version v. Before A lands, B
/// either adds a column or activates the contract explicitly. A is rebased and
/// activation is computed on the landing candidate: the mark covers B's
/// column, and B's own activation is kept rather than recomputed.
#[tokio::test]
async fn a_stale_first_submission_activates_at_the_landing_mark() {
    for b_does in ["append", "add_column", "activate"] {
        let b_activates = b_does == "activate";
        let dir = TempStrDir::default();
        let dataset = write_dataset(&dir).await;
        let planned_at = dataset.manifest().version;

        let mut b = dataset.clone();
        if b_activates {
            b.migrate_to_non_reusable_field_ids().await.unwrap();
        } else if b_does == "append" {
            b = Dataset::write(
                rows(100..110),
                dataset.uri(),
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        } else {
            b.add_columns(
                NewColumnTransform::SqlExpressions(vec![("j".to_string(), "i + 1".to_string())]),
                None,
                None,
            )
            .await
            .unwrap();
        }
        assert_eq!(b.manifest().version, planned_at + 1);
        let expected_mark = if b_activates {
            b.manifest().max_allocated_field_id.unwrap()
        } else {
            b.manifest().max_field_id()
        };

        let landed = commit(
            &dataset,
            submission(planned_at, vec![definition(1)], vec![]),
        )
        .await
        .unwrap();
        assert_eq!(landed.manifest().version, planned_at + 2);
        assert_eq!(ids(&landed), vec![1]);
        assert_eq!(
            landed.manifest().max_allocated_field_id,
            Some(expected_mark),
            "{b_does}"
        );
        assert_ne!(
            landed.manifest().writer_feature_flags & FLAG_NON_REUSABLE_FIELD_IDS,
            0
        );
    }
}

/// A key that was already absent when the writer read the table is history,
/// not a changed dependency: a list that retains such a definition, allocates
/// another and removes an unreferenced one still lands.
#[tokio::test]
async fn a_historical_absence_is_not_a_changed_dependency() {
    let dir = TempStrDir::default();
    let read = with_region_column(write_named_column(&dir, "a", WriteMode::Create).await).await;
    let key_id = read.schema().field("region").unwrap().id;
    let a_id = read.schema().field("a").unwrap().id;
    let on_region = ClusteringDefinition {
        columns: vec![key_id],
        ..definition(1)
    };
    let stale = ClusteringDefinition {
        columns: vec![key_id],
        ..definition(2)
    };
    let read = commit(
        &read,
        submission(
            read.manifest().version,
            vec![on_region.clone(), stale.clone()],
            vec![],
        ),
    )
    .await
    .unwrap();
    let mut read = read;
    read.drop_columns(&["region"]).await.unwrap();
    assert!(read.schema().field_by_id(key_id).is_none());

    let on_a = ClusteringDefinition {
        columns: vec![a_id],
        ..definition(3)
    };
    let landed = commit(
        &read,
        submission(
            read.manifest().version,
            vec![on_region.clone(), on_a.clone()],
            vec![],
        ),
    )
    .await
    .unwrap();
    assert_eq!(landed.manifest().clustering, vec![on_region, on_a]);
    assert_eq!(landed.manifest().max_clustering_id, 3);
}

/// The first-allocation fence judges the concurrent transaction, not the final
/// shape: a Merge that only adds fields (a new column, a new struct column
/// with its children) is compatible and the allocation rebases over it, while a Merge
/// that renames a field the writer read is refused, since on a table with
/// reusable field ids it may have moved a key's id.
#[tokio::test]
async fn a_first_allocation_rebases_over_an_additive_merge_but_not_a_rename() {
    // Additive: a new top-level column and a new struct column with children.
    let dir = TempStrDir::default();
    let read = with_region_column(write_named_column(&dir, "a", WriteMode::Create).await).await;
    let mut b = read.clone();
    b.add_columns(
        NewColumnTransform::SqlExpressions(vec![
            ("price".to_string(), "a * 2".to_string()),
            (
                "info".to_string(),
                "named_struct('zip', a, 'note', a)".to_string(),
            ),
        ]),
        None,
        None,
    )
    .await
    .unwrap();
    let key_id = read.schema().field("region").unwrap().id;
    let landed = commit(
        &read,
        submission(
            read.manifest().version,
            vec![ClusteringDefinition {
                columns: vec![key_id],
                ..definition(1)
            }],
            vec![],
        ),
    )
    .await
    .unwrap();
    assert_eq!(ids(&landed), vec![1]);
    assert!(landed.schema().field("price").is_some());
    assert!(landed.schema().field("info").is_some());

    // A rename through a Merge in the window is refused: the renamed field is
    // one the writer read, and on this table its id could have moved.
    let dir = TempStrDir::default();
    let read = with_region_column(write_named_column(&dir, "a", WriteMode::Create).await).await;
    let mut b = Dataset::open(&dir).await.unwrap();
    b.alter_columns(&[crate::dataset::ColumnAlteration::new("a".into()).rename("b".into())])
        .await
        .unwrap();
    let key_id = read.schema().field("region").unwrap().id;
    let error = commit(
        &read,
        submission(
            read.manifest().version,
            vec![ClusteringDefinition {
                columns: vec![key_id],
                ..definition(1)
            }],
            vec![],
        ),
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("may have given one of those ids to another column"),
        "{error}"
    );
}

/// A submission depends on the key fields of the definitions it allocates,
/// not on every retained definition's. An unchanged list with unrelated
/// metadata lands over another writer's rename, retype or drop of a retained
/// key. A submission allocating two more definitions, the lower one keyed on
/// that field, lands over the rename and is refused as a whole over the
/// retype or drop; planned again from the landing snapshot, it lands.
#[rstest]
#[case::rename("rename")]
#[case::cast("cast")]
#[case::drop("drop")]
#[tokio::test]
async fn a_retained_keys_change_blocks_only_a_new_allocation_on_it(#[case] change: &'static str) {
    let dir = TempStrDir::default();
    let read = with_region_column(write_named_column(&dir, "a", WriteMode::Create).await).await;
    let key_id = read.schema().field("region").unwrap().id;
    let a_id = read.schema().field("a").unwrap().id;
    let configured = ClusteringDefinition {
        columns: vec![key_id],
        ..definition(1)
    };
    let read = commit(
        &read,
        submission(read.manifest().version, vec![configured.clone()], vec![]),
    )
    .await
    .unwrap();
    let planned_at = read.manifest().version;
    let mut b = Dataset::open(&dir).await.unwrap();
    match change {
        "drop" => b.drop_columns(&["region"]).await.unwrap(),
        "rename" => b
            .alter_columns(&[
                crate::dataset::ColumnAlteration::new("region".into()).rename("zone".into())
            ])
            .await
            .unwrap(),
        _ => b
            .alter_columns(&[crate::dataset::ColumnAlteration::new("region".into())
                .cast_to(arrow_schema::DataType::Int64)])
            .await
            .unwrap(),
    }
    // Unrelated to the retained definition: lands over any of the changes.
    let note = ("org.example.note".to_string(), Some("x".to_string()));
    let landed = commit(
        &read,
        submission(planned_at, vec![configured.clone()], vec![note.clone()]),
    )
    .await
    .unwrap();
    assert_eq!(landed.manifest().clustering, vec![configured.clone()]);
    assert_eq!(
        landed.manifest().table_metadata["org.example.note"],
        "x",
        "{change}: the unrelated submission landed"
    );

    // Allocating on the changed key (definition 2) and on an untouched one
    // (definition 3) at once: a rename keeps the key, a retype or a drop
    // refuses the submission for definition 2 and nothing of it lands.
    let on_region = ClusteringDefinition {
        columns: vec![key_id],
        ..definition(2)
    };
    let on_a = ClusteringDefinition {
        columns: vec![a_id],
        ..definition(3)
    };
    let spec = ("org.example.spec.2".to_string(), Some("{}".to_string()));
    let result = commit(
        &read,
        submission(
            planned_at,
            vec![configured.clone(), on_region.clone(), on_a.clone()],
            vec![spec.clone()],
        ),
    )
    .await;
    let reopened = Dataset::open(&dir).await.unwrap();
    if change == "rename" {
        let landed = result.unwrap();
        assert_eq!(ids(&landed), vec![1, 2, 3]);
        assert_eq!(
            reopened.manifest().table_metadata["org.example.spec.2"],
            "{}"
        );
    } else {
        let error = result.unwrap_err();
        assert!(
            matches!(error, crate::Error::CommitConflict { .. }),
            "{change}: {error:?}"
        );
        assert!(
            error
                .to_string()
                .contains(&format!("definition 2 is keyed on field id {key_id}")),
            "{change}: {error}"
        );
        assert_eq!(ids(&reopened), vec![1], "{change}: no allocation landed");
        assert!(
            !reopened
                .manifest()
                .table_metadata
                .contains_key("org.example.spec.2"),
            "{change}: the refused submission landed nothing"
        );
        // Planned again from the landing snapshot, with the key as it is
        // now (or without it, once dropped), it lands; ids continue the mark.
        let mut replanned = vec![configured.clone()];
        if let Some(region) = reopened.schema().field("region") {
            replanned.push(ClusteringDefinition {
                columns: vec![region.id],
                ..definition(2)
            });
        }
        replanned.push(ClusteringDefinition {
            columns: vec![a_id],
            ..definition(replanned.len() as u32 + 1)
        });
        let landed = commit(
            &reopened,
            submission(reopened.manifest().version, replanned.clone(), vec![spec]),
        )
        .await
        .unwrap();
        assert_eq!(landed.manifest().clustering, replanned);
        assert_eq!(landed.manifest().table_metadata["org.example.spec.2"], "{}");
    }
}

/// Writer A resolves its key field in a reusable-id snapshot; before A lands,
/// writer B replaces the schema in a way that can hand A's key id to another
/// column: an overwrite, a drop followed by an add, the same with the original
/// name and type (the final shape hides the change), or a replacement followed
/// by an explicit activation. Every such transaction in A's window is a
/// conflict, and A publishes nothing. Once the table is under the contract,
/// B's columns get fresh ids and A's stale key simply no longer exists, which
/// the allocation's own schema check refuses.
#[rstest]
#[case::overwrite("overwrite")]
#[case::drop_then_add("drop_add")]
#[case::drop_then_readd_same_name("drop_readd")]
#[case::overwrite_same_shape("overwrite_same")]
#[case::drop_add_then_activate("drop_add_activate")]
#[tokio::test]
async fn a_stale_allocation_cannot_cross_a_field_identity_change(#[case] race: &'static str) {
    async fn replace(uri: &str, read: &Dataset, race: &str) {
        let mut b = read.clone();
        let add = |b: &Dataset, name: &str| {
            let expression = vec![(name.to_string(), "a * 2".to_string())];
            let mut b = b.clone();
            async move {
                b.add_columns(NewColumnTransform::SqlExpressions(expression), None, None)
                    .await
                    .unwrap();
                b
            }
        };
        match race {
            "overwrite" => {
                write_named_columns(uri, &["x", "user_id"], WriteMode::Overwrite).await;
            }
            "overwrite_same" => {
                write_named_columns(uri, &["a", "region"], WriteMode::Overwrite).await;
            }
            "drop_add" | "drop_add_activate" => {
                b.drop_columns(&["region"]).await.unwrap();
                let mut b = add(&b, "price").await;
                if race == "drop_add_activate" {
                    b.migrate_to_non_reusable_field_ids().await.unwrap();
                }
            }
            "drop_readd" => {
                b.drop_columns(&["region"]).await.unwrap();
                add(&b, "region").await;
            }
            _ => unreachable!(),
        }
    }

    // Reusable ids: the key is the highest id and lives in its own data file,
    // so a drop releases it to the next column; an overwrite numbers from zero.
    let dir = TempStrDir::default();
    let read = with_region_column(write_named_column(&dir, "a", WriteMode::Create).await).await;
    let key_id = read.schema().field("region").unwrap().id;
    let planned_at = read.manifest().version;
    replace(&dir, &read, race).await;
    let latest = Dataset::open(&dir).await.unwrap();
    let now_named = latest
        .schema()
        .field_by_id(key_id)
        .map(|field| field.name.clone());
    match race {
        "overwrite" => assert_eq!(now_named.as_deref(), Some("user_id")),
        "drop_readd" | "overwrite_same" => assert_eq!(now_named.as_deref(), Some("region")),
        _ => assert_eq!(now_named.as_deref(), Some("price")),
    }
    let stale = ClusteringDefinition {
        columns: vec![key_id],
        ..definition(1)
    };
    let error = commit(&read, submission(planned_at, vec![stale], vec![]))
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("may have given one of those ids to another column"),
        "{race}: {error}"
    );
    let reopened = Dataset::open(&dir).await.unwrap();
    assert_eq!(
        reopened.manifest().version,
        latest.manifest().version,
        "{race}: A landed nothing"
    );
    assert!(reopened.manifest().clustering.is_empty(), "{race}");
    assert_eq!(reopened.manifest().max_clustering_id, 0, "{race}");
    assert_eq!(
        reopened.manifest().uses_non_reusable_field_ids(),
        race == "drop_add_activate",
        "{race}: only B's own activation may be present"
    );

    // Under the contract from the start, the same race leaves A's key without
    // a field; nothing in the window can give its id to another column, and
    // the vanished key is still a changed dependency, not bad input.
    if matches!(race, "overwrite" | "drop_add") {
        let dir = TempStrDir::default();
        let mut read =
            with_region_column(write_named_column(&dir, "a", WriteMode::Create).await).await;
        read.migrate_to_non_reusable_field_ids().await.unwrap();
        let key_id = read.schema().field("region").unwrap().id;
        let planned_at = read.manifest().version;
        replace(&dir, &read, race).await;
        let stale = ClusteringDefinition {
            columns: vec![key_id],
            ..definition(1)
        };
        let error = commit(&read, submission(planned_at, vec![stale], vec![]))
            .await
            .unwrap_err();
        assert!(
            matches!(error, crate::Error::CommitConflict { .. })
                && error.to_string().contains("no longer exists"),
            "{race}: {error:?}"
        );
    }
}

/// The fence is decided from the version the transaction read, whatever handle
/// it is submitted through: the original snapshot, a handle already at the
/// latest (activated) version, or the table URI. A same-name overwrite then an
/// explicit activation is refused through all three. When a version inside the
/// window has been cleaned up the window cannot be judged and is refused too.
/// Without a replacement, the same submissions land and allocate.
#[rstest]
#[case::read_handle("read", true, false)]
#[case::latest_handle("latest", true, false)]
#[case::uri("uri", true, false)]
#[case::cleaned_window("read", true, true)]
#[case::latest_handle_no_replacement("latest", false, false)]
#[case::uri_no_replacement("uri", false, false)]
#[case::latest_handle_unrelated_append("latest-append", false, false)]
#[case::uri_unrelated_append("uri-append", false, false)]
#[tokio::test]
async fn the_fence_follows_the_read_version_not_the_handle(
    #[case] destination: &str,
    #[case] replaced: bool,
    #[case] cleaned: bool,
) {
    let dir = TempStrDir::default();
    let read = write_named_columns(&dir, &["a", "region"], WriteMode::Create).await;
    let read_version = read.manifest().version;
    let key_id = read.schema().field("region").unwrap().id;
    let (destination, appended) = match destination.strip_suffix("-append") {
        Some(destination) => (destination, true),
        None => (destination, false),
    };
    let mut latest = if replaced {
        // Reusable ids give the replacement the old id, name and type.
        write_named_columns(&dir, &["a", "region"], WriteMode::Overwrite).await
    } else if appended {
        // An unrelated append in the window changes no field identity.
        write_named_columns(&dir, &["a", "region"], WriteMode::Append).await
    } else {
        read.clone()
    };
    assert_eq!(latest.schema().field("region").unwrap().id, key_id);
    latest.migrate_to_non_reusable_field_ids().await.unwrap();
    let before = latest.manifest().version;
    if cleaned {
        let policy = CleanupPolicyBuilder::default()
            .versions(vec![read_version + 1])
            .unwrap()
            .build();
        let removed = latest.cleanup_with_policy(policy).await.unwrap();
        assert_eq!(removed.old_versions, 1);
    }

    let (_, transaction) = submission(read_version, vec![], vec![]);
    let desired = vec![ClusteringDefinition {
        columns: vec![key_id],
        ..definition(1)
    }];
    let builder = match destination {
        "read" => CommitBuilder::new(Arc::new(read.clone())),
        "latest" => CommitBuilder::new(Arc::new(Dataset::open(&dir).await.unwrap())),
        "uri" => CommitBuilder::new(dir.as_str()),
        _ => unreachable!(),
    };
    let result = builder
        .with_clustering_definitions(desired)
        .execute(transaction)
        .await;
    let reopened = Dataset::open(&dir).await.unwrap();
    if replaced {
        let Err(error) = result else {
            panic!("{destination} cleaned={cleaned}: a replaced key was allocated")
        };
        let message = error.to_string();
        assert!(
            message.contains("may have given one of those ids to another column")
                || message.contains("cannot be checked"),
            "{destination} cleaned={cleaned}: {error}"
        );
        assert_eq!(
            reopened.manifest().version,
            before,
            "{destination}: nothing landed"
        );
        assert!(reopened.manifest().clustering.is_empty());
    } else {
        let landed = result.unwrap();
        assert_eq!(landed.manifest().version, before + 1);
        assert_eq!(landed.manifest().clustering[0].columns, vec![key_id]);
        assert_eq!(reopened.manifest().version, before + 1);
        assert_eq!(
            reopened.fragments().len(),
            if appended { 2 } else { 1 },
            "{destination}: the window's append is kept"
        );
    }
}

/// A definitions list rides only an `UpdateConfig`. A shallow clone into a
/// destination that does not exist yet is committed by a path that never
/// reaches the manifest builder, so the refusal happens before any path is
/// chosen and nothing is written to the destination.
#[tokio::test]
async fn definitions_on_a_clone_are_refused_before_anything_is_written() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let target = TempStrDir::default();
    let target_uri = format!("{}/cloned", target.as_str());
    let transaction = Transaction::new(
        dataset.manifest().version,
        Operation::Clone {
            is_shallow: true,
            ref_name: dataset.manifest().branch.clone(),
            ref_version: dataset.manifest().version,
            ref_path: dataset.uri().to_string(),
            branch_name: None,
        },
        None,
    );
    let error = CommitBuilder::new(target_uri.as_str())
        .with_clustering_definitions(vec![definition(1)])
        .execute(transaction)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("only be committed with UpdateConfig"),
        "{error}"
    );
    assert!(
        Dataset::open(&target_uri).await.is_err(),
        "nothing was cloned"
    );
}

/// Allocating the first definition on a reusable-id table activates the
/// non-reusable field-id contract, and that contract refuses restoring any
/// version from before activation. The old version stays readable.
#[tokio::test]
async fn the_first_allocation_makes_earlier_versions_unrestorable_but_readable() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let before = dataset.manifest().version;
    let dataset = commit(&dataset, submission(before, vec![definition(1)], vec![]))
        .await
        .unwrap();
    assert!(dataset.manifest().uses_non_reusable_field_ids());

    let mut earlier = dataset.checkout_version(before).await.unwrap();
    assert_eq!(
        earlier.count_rows(None).await.unwrap(),
        100,
        "still readable"
    );
    let error = earlier.restore().await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("non-reusable field IDs were activated after that version"),
        "{error}"
    );

    // A version from after activation restores as usual.
    let activated_at = dataset.manifest().version;
    let dataset = commit(
        &dataset,
        submission(activated_at, vec![definition(1), definition(2)], vec![]),
    )
    .await
    .unwrap();
    let mut restored = dataset.checkout_version(activated_at).await.unwrap();
    restored.restore().await.unwrap();
    assert_eq!(
        ids(&restored),
        vec![1, 2],
        "restore keeps every allocated definition"
    );
}

/// A build that does not know segment labels caches the index list with every
/// segment unlabelled, under the key schema this build used to share with it.
/// Such an entry on a shared persistent cache must miss: the next commit
/// refills from the manifest and keeps the label, and the definition that
/// segment names still cannot be removed.
#[tokio::test]
async fn a_prior_writers_index_list_cache_entry_is_not_consulted() {
    use lance_core::cache::{CacheCodec, CacheKey, CacheKeySchema, KeyBuilder};
    use std::borrow::Cow;

    /// The real `IndexMetadataKey`, field for field and byte for byte, at the
    /// schema version this build shared with the prior writer. The version is
    /// the only difference, so a hit here would be a hit for that writer.
    struct PriorWriterIndexMetadataKey<'a> {
        version: u64,
        store_identity: &'a str,
        e_tag: Option<&'a str>,
    }

    impl CacheKey for PriorWriterIndexMetadataKey<'_> {
        type ValueType = Vec<IndexMetadata>;

        fn key(&self) -> Cow<'_, str> {
            Cow::Owned(format!(
                "{}:{}/{}/{}",
                self.store_identity.len(),
                self.store_identity,
                self.version,
                self.e_tag.unwrap_or("")
            ))
        }

        fn type_name() -> &'static str {
            "Vec<IndexMetadata>"
        }

        fn schema() -> CacheKeySchema {
            CacheKeySchema::new("lance.index.metadata-key", 2)
        }

        fn write_key(&self, builder: &mut KeyBuilder) {
            builder.write_str(self.store_identity);
            builder.write_u64(self.version);
            match self.e_tag {
                Some(e_tag) => {
                    builder.write_some();
                    builder.write_str(e_tag);
                }
                None => builder.write_none(),
            }
        }

        fn codec() -> Option<CacheCodec> {
            Some(lance_table::format::index_metadata_codec())
        }
    }

    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let mut dataset = commit_edited_manifest(&dataset, |manifest, indices| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
        let segment = indices.iter_mut().find(|idx| idx.name == "idx").unwrap();
        segment.clustering_id = Some(1);
        segment.metadata = object_metadata("3");
    })
    .await;

    // What the prior writer would have cached for this version: the same
    // list with the fields it did not know stripped.
    let stripped = named_segments(&dataset, "idx")
        .await
        .into_iter()
        .map(|segment| IndexMetadata {
            clustering_id: None,
            metadata: Default::default(),
            ..segment
        })
        .collect::<Vec<_>>();
    dataset
        .index_cache
        .insert_with_key(
            &PriorWriterIndexMetadataKey {
                version: dataset.manifest().version,
                store_identity: &dataset.object_store.store_prefix,
                e_tag: dataset.manifest_location.e_tag.as_deref(),
            },
            Arc::new(stripped),
        )
        .await;

    // A commit that republishes the index section must not trust that entry.
    dataset.delete("false").await.unwrap();
    let segment = named_segments(&dataset, "idx").await.remove(0);
    assert_eq!(
        segment.clustering_id,
        Some(1),
        "the label survived the commit"
    );
    assert_eq!(segment.metadata, object_metadata("3"));

    // And the definition that segment names is still pinned.
    let error = commit(
        &dataset,
        submission(dataset.manifest().version, vec![], vec![]),
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot remove clustering definition 1: index segment"),
        "{error}"
    );
}

const VEC_DIM: i32 = 4;

fn vector_rows(ids: std::ops::Range<i32>) -> (Arc<ArrowSchema>, RecordBatch) {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", DataType::Int32, false),
        ArrowField::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(ArrowField::new("item", DataType::Float32, true)),
                VEC_DIM,
            ),
            true,
        ),
    ]));
    let values = Float32Array::from_iter_values(
        ids.clone()
            .flat_map(|id| (0..VEC_DIM).map(move |d| (id * VEC_DIM + d) as f32)),
    );
    let vectors = FixedSizeListArray::try_new(
        Arc::new(ArrowField::new("item", DataType::Float32, true)),
        VEC_DIM,
        Arc::new(values),
        None,
    )
    .unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from_iter_values(ids)),
            Arc::new(vectors),
        ],
    )
    .unwrap();
    (schema, batch)
}

/// Compaction without stable row ids remaps the vector segments that cover
/// the rewritten fragments into one new segment. That segment is a new
/// object: with the two sources labelled differently, the published segment
/// carries neither label nor metadata.
#[tokio::test]
async fn compaction_merging_vector_segments_publishes_an_unlabelled_segment() {
    let dir = TempStrDir::default();
    let (schema, batch) = vector_rows(0..64);
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(batch)], schema.clone()),
        &dir,
        Some(WriteParams {
            max_rows_per_file: 32,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".into()),
            &VectorIndexParams::ivf_flat(1, MetricType::L2),
            true,
        )
        .await
        .unwrap();
    let (_, more) = vector_rows(64..128);
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(more)], schema),
        dataset.uri(),
        Some(WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file: 32,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.checkout_latest().await.unwrap();
    dataset
        .optimize_indices(&OptimizeOptions::append())
        .await
        .unwrap();
    assert_eq!(named_segments(&dataset, "vec_idx").await.len(), 2);

    let mut dataset = commit_edited_manifest(&dataset, |manifest, indices| {
        manifest.clustering = vec![definition(1), definition(2)];
        manifest.max_clustering_id = 2;
        for (position, segment) in indices
            .iter_mut()
            .filter(|segment| segment.name == "vec_idx")
            .enumerate()
        {
            segment.clustering_id = Some(position as u32 + 1);
            segment.metadata = object_metadata(&position.to_string());
        }
    })
    .await;

    // Every fragment compacts into one, so both segments are replaced by one.
    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1_000,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert_eq!(dataset.fragments().len(), 1);
    let merged = named_segments(&dataset, "vec_idx").await;
    assert_eq!(merged.len(), 1, "{merged:?}");
    assert_eq!(
        merged[0].clustering_id, None,
        "a merged segment is a new object"
    );
    assert!(merged[0].metadata.is_empty());
    // The definitions themselves are untouched by the rewrite.
    assert_eq!(ids(&dataset), vec![1, 2]);
}

/// A detached commit is outside the version chain; it cannot carry a
/// definitions list.
#[tokio::test]
async fn definitions_cannot_be_committed_detached() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let (desired, transaction) =
        submission(dataset.manifest().version, vec![definition(1)], vec![]);
    let error = CommitBuilder::new(Arc::new(dataset.clone()))
        .with_detached(true)
        .with_clustering_definitions(desired)
        .execute(transaction)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("committed detached"), "{error}");
    let reopened = Dataset::open(&dir).await.unwrap();
    assert!(reopened.manifest().clustering.is_empty());
    assert!(!reopened.manifest().uses_non_reusable_field_ids());
}

/// Append ten rows as one fragment labelled `clustering_id`, through the real
/// commit path.
async fn append_labelled(dataset: &Dataset, clustering_id: u32) -> crate::Result<Dataset> {
    let mut transaction = InsertBuilder::new(Arc::new(dataset.clone()))
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        })
        .execute_uncommitted(vec![
            RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int32Array::from_iter_values(100..110))],
            )
            .unwrap(),
        ])
        .await
        .unwrap();
    let Operation::Append { fragments } = &mut transaction.operation else {
        unreachable!()
    };
    for fragment in fragments {
        fragment.clustering_id = Some(clustering_id);
        fragment.metadata = object_metadata("5");
    }
    CommitBuilder::new(Arc::new(dataset.clone()))
        .execute(transaction)
        .await
}

/// A definition is removed only while nothing references it. Definitions
/// 1 and 2 are pinned by a fragment and by an index segment respectively;
/// only 3 can go, one referenced id refuses a whole mixed update, and once
/// 3 is gone no new object can name it.
#[tokio::test]
async fn a_referenced_definition_cannot_be_removed() {
    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let dataset = commit_edited_manifest(&dataset, |manifest, indices| {
        manifest.clustering = vec![definition(1), definition(2), definition(3)];
        manifest.max_clustering_id = 3;
        stamp_fragment(manifest, 0, 1, "0");
        let segment = indices.iter_mut().find(|idx| idx.name == "idx").unwrap();
        segment.clustering_id = Some(2);
        segment.metadata = object_metadata("3");
    })
    .await;
    let version = dataset.manifest().version;
    let note = ("org.example.note".to_string(), Some("x".to_string()));

    // A live fragment reference blocks the removal of 1.
    let error = commit(
        &dataset,
        submission(
            version,
            vec![definition(2), definition(3)],
            vec![note.clone()],
        ),
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot remove clustering definition 1: fragment 0"),
        "{error}"
    );

    // An index-only reference blocks the removal of 2.
    let error = commit(
        &dataset,
        submission(
            version,
            vec![definition(1), definition(3)],
            vec![note.clone()],
        ),
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot remove clustering definition 2: index segment"),
        "{error}"
    );

    // One referenced id refuses the whole mixed update, and clearing the list.
    let error = commit(
        &dataset,
        submission(
            version,
            vec![definition(2), definition(4)],
            vec![note.clone()],
        ),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("definition 1"), "{error}");
    let error = commit(&dataset, submission(version, vec![], vec![note.clone()]))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("cannot remove"), "{error}");

    // Nothing landed: no removal, no allocation, no watermark move, no patch.
    let reopened = Dataset::open(&dir).await.unwrap();
    assert_eq!(reopened.manifest().version, version);
    assert_eq!(ids(&reopened), vec![1, 2, 3]);
    assert_eq!(reopened.manifest().max_clustering_id, 3);
    assert!(
        !reopened
            .manifest()
            .table_metadata
            .contains_key("org.example.note")
    );
    assert_eq!(fragment_by_id(&reopened, 0).clustering_id, Some(1));
    assert_eq!(
        named_segments(&reopened, "idx").await[0].clustering_id,
        Some(2)
    );

    // The unreferenced definition can go, with the patch, keeping the mark.
    let dataset = commit(
        &dataset,
        submission(version, vec![definition(1), definition(2)], vec![note]),
    )
    .await
    .unwrap();
    assert_eq!(ids(&dataset), vec![1, 2]);
    assert_eq!(dataset.manifest().max_clustering_id, 3);
    assert_eq!(dataset.manifest().table_metadata["org.example.note"], "x");

    // Deletion landed first: a new object cannot name the absent definition.
    let error = append_labelled(&dataset, 3).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("references clustering definition 3"),
        "{error}"
    );
    let appended = append_labelled(&dataset, 1).await.unwrap();
    assert_eq!(appended.fragments().last().unwrap().clustering_id, Some(1));
}

/// Writer A plans to remove an unreferenced definition; before A lands,
/// writer B publishes a fragment that references it, changing neither the
/// definitions nor the watermark. A is rebased onto B's manifest and the
/// removal is checked against that landing snapshot, so it is refused.
#[tokio::test]
async fn a_concurrent_reference_blocks_a_planned_removal() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let dataset = commit(
        &dataset,
        submission(dataset.manifest().version, vec![definition(1)], vec![]),
    )
    .await
    .unwrap();
    let planned_at = dataset.manifest().version;

    // B lands first.
    let after_b = append_labelled(&dataset, 1).await.unwrap();
    assert_eq!(after_b.manifest().version, planned_at + 1);
    assert_eq!(ids(&after_b), vec![1]);
    assert_eq!(after_b.manifest().max_clustering_id, 1);

    // A, from its stale snapshot, is rebased and refused on the new landing
    // candidate, not on the snapshot it planned against.
    let error = commit(
        &dataset,
        submission(
            planned_at,
            vec![],
            vec![("org.example.note".to_string(), Some("x".to_string()))],
        ),
    )
    .await
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot remove clustering definition 1: fragment"),
        "{error}"
    );
    let reopened = Dataset::open(&dir).await.unwrap();
    assert_eq!(
        reopened.manifest().version,
        planned_at + 1,
        "A landed nothing"
    );
    assert_eq!(ids(&reopened), vec![1]);
    assert!(
        !reopened
            .manifest()
            .table_metadata
            .contains_key("org.example.note")
    );
    assert_eq!(
        reopened.fragments().last().unwrap().clustering_id,
        Some(1),
        "B's fragment keeps its label"
    );
}

/// A CreateIndex published through the real commit path checks every segment
/// it carries, including one that replaces an existing segment: an undefined
/// label is refused, a defined one is kept.
#[tokio::test]
async fn create_index_publication_checks_the_segment_label() {
    let dir = TempStrDir::default();
    let mut dataset = write_dataset(&dir).await;
    dataset
        .create_index(
            &["i"],
            IndexType::BTree,
            Some("idx".into()),
            &ScalarIndexParams::default(),
            true,
        )
        .await
        .unwrap();
    let dataset = commit_edited_manifest(&dataset, |manifest, _| {
        manifest.clustering = vec![definition(1)];
        manifest.max_clustering_id = 1;
    })
    .await;
    let old = named_segments(&dataset, "idx").await.remove(0);

    let replace_with = |clustering_id: u32| {
        let dataset = dataset.clone();
        let old = old.clone();
        async move {
            let mut replacement = old.clone();
            replacement.uuid = Uuid::new_v4();
            replacement.clustering_id = Some(clustering_id);
            replacement.metadata = object_metadata("3");
            let transaction = Transaction::new(
                dataset.manifest().version,
                Operation::CreateIndex {
                    new_indices: vec![replacement],
                    removed_indices: vec![old],
                },
                None,
            );
            CommitBuilder::new(Arc::new(dataset))
                .execute(transaction)
                .await
        }
    };

    let error = replace_with(9).await.unwrap_err();
    assert!(
        error
            .to_string()
            .contains("references clustering definition 9"),
        "{error}"
    );
    assert_eq!(named_segments(&dataset, "idx").await[0].uuid, old.uuid);

    let committed = replace_with(1).await.unwrap();
    let segments = named_segments(&committed, "idx").await;
    assert_eq!(segments.len(), 1);
    assert_ne!(segments[0].uuid, old.uuid);
    assert_eq!(segments[0].clustering_id, Some(1));
    assert_eq!(segments[0].metadata, object_metadata("3"));
}

/// A definitions list only applies to `UpdateConfig`. Attaching one to any
/// other operation must fail rather than commit with the list ignored, and the
/// table must be left exactly as it was. Exercised through
/// `CommitBuilder::execute`, since the production commit loop reaches the
/// builder by a different path than the convenience wrapper.
#[tokio::test]
async fn definitions_on_an_append_are_refused_and_change_nothing() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let before = dataset.manifest().version;

    let transaction = Transaction::new(
        before,
        Operation::Append {
            fragments: dataset
                .get_fragments()
                .iter()
                .map(|f| f.metadata.clone())
                .collect(),
        },
        None,
    );
    let error = CommitBuilder::new(Arc::new(dataset.clone()))
        .with_clustering_definitions(vec![definition(1)])
        .execute(transaction)
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("UpdateConfig"),
        "an Append carrying definitions must be refused: {error:#}"
    );

    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(
        reopened.manifest().version,
        before,
        "nothing may have landed"
    );
    assert!(reopened.manifest().clustering.is_empty());
    assert_eq!(reopened.manifest().max_clustering_id, 0);
}

/// An EMPTY list is still an input: it must be refused on an unsupported
/// operation rather than treated as "nothing to do".
#[tokio::test]
async fn an_empty_definitions_list_is_still_refused_on_an_append() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let before = dataset.manifest().version;

    let transaction = Transaction::new(
        before,
        Operation::Append {
            fragments: dataset
                .get_fragments()
                .iter()
                .map(|f| f.metadata.clone())
                .collect(),
        },
        None,
    );
    let error = CommitBuilder::new(Arc::new(dataset.clone()))
        .with_clustering_definitions(vec![])
        .execute(transaction)
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("UpdateConfig"),
        "an empty list is an input, not an absence: {error:#}"
    );
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(reopened.manifest().version, before);
}

/// Restore dispatches separately from the prepared build path, so it needs its
/// own rejection: it would otherwise ignore the list and report success.
#[tokio::test]
async fn definitions_on_a_restore_are_refused_and_change_nothing() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let first = dataset.manifest().version;
    // Land a second version so there is an earlier one to restore to.
    let dataset = commit(&dataset, submission(first, vec![definition(1)], vec![]))
        .await
        .unwrap();
    let before = dataset.manifest().version;

    let transaction = Transaction::new(before, Operation::Restore { version: first }, None);
    let error = CommitBuilder::new(Arc::new(dataset.clone()))
        .with_clustering_definitions(vec![definition(1)])
        .execute(transaction)
        .await
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("UpdateConfig"),
        "a Restore carrying definitions must be refused: {error:#}"
    );
    let reopened = Dataset::open(dataset.uri()).await.unwrap();
    assert_eq!(
        reopened.manifest().version,
        before,
        "nothing may have landed"
    );
}

#[tokio::test]
async fn complete_list_submission_commits_definitions_and_keys_in_one_version() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    assert_eq!(dataset.manifest().max_clustering_id, 0);

    let dataset = commit(
        &dataset,
        submission(
            dataset.manifest().version,
            vec![definition(1)],
            vec![
                (ACTIVE_KEY.to_string(), Some("1".to_string())),
                (spec_key(1), Some("{\"v\":1}".to_string())),
            ],
        ),
    )
    .await
    .unwrap();
    let manifest = dataset.manifest();
    assert_eq!(manifest.clustering, vec![definition(1)]);
    assert_eq!(manifest.max_clustering_id, 1);
    assert_eq!(manifest.table_metadata[ACTIVE_KEY], "1");
    assert_eq!(manifest.table_metadata[&spec_key(1)], "{\"v\":1}");
    assert_eq!(
        manifest.writer_feature_flags & FLAG_OBJECT_METADATA,
        FLAG_OBJECT_METADATA
    );

    // the durable record is a plain UpdateConfig: the key upserts are in the
    // transaction file, the definitions only in the manifest (a fresh open
    // reads the file instead of the committing session's cached transaction)
    let recorded = Dataset::open(&dir)
        .await
        .unwrap()
        .read_transaction()
        .await
        .unwrap()
        .unwrap();
    let Operation::UpdateConfig {
        table_metadata_updates,
        ..
    } = recorded.operation
    else {
        panic!("not an UpdateConfig");
    };
    assert_eq!(table_metadata_updates.unwrap().update_entries.len(), 2);

    // the complete list grows: retained entries plus two new ones, in id order
    let dataset = commit(
        &dataset,
        submission(
            dataset.manifest().version,
            vec![
                definition(1),
                ClusteringDefinition {
                    id: 2,
                    columns: vec![0],
                    provider: "org.example.other".into(),
                    version: 3,
                },
                definition(3),
            ],
            vec![(ACTIVE_KEY.to_string(), Some("3".to_string()))],
        ),
    )
    .await
    .unwrap();
    assert_eq!(ids(&dataset), vec![1, 2, 3]);
    assert_eq!(dataset.manifest().max_clustering_id, 3);
    assert_eq!(dataset.manifest().table_metadata[ACTIVE_KEY], "3");
    assert_eq!(dataset.manifest().table_metadata[&spec_key(1)], "{\"v\":1}");
}

/// The desired list is the target list: an omission the writer read is an
/// intended removal, removal and allocation can land together, and the
/// watermark never moves backwards so a freed id is spent.
#[tokio::test]
async fn complete_list_removes_omitted_definitions_and_keeps_the_mark() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let dataset = commit(
        &dataset,
        submission(dataset.manifest().version, vec![definition(1)], vec![]),
    )
    .await
    .unwrap();
    let dataset = commit(
        &dataset,
        submission(
            dataset.manifest().version,
            vec![definition(1), definition(2)],
            vec![],
        ),
    )
    .await
    .unwrap();
    assert_eq!(ids(&dataset), vec![1, 2]);
    assert_eq!(dataset.manifest().max_clustering_id, 2);

    // Removal only: definition 2 goes, the mark stays so its id is not reused.
    let removed = commit(
        &dataset,
        submission(dataset.manifest().version, vec![definition(1)], vec![]),
    )
    .await
    .unwrap();
    assert_eq!(ids(&removed), vec![1], "definition 2 is removed");
    assert_eq!(
        removed.manifest().max_clustering_id,
        2,
        "a removal never releases the id"
    );

    // The freed id is spent: resubmitting 2 is at the mark, not above it.
    let error = commit(
        &removed,
        submission(
            removed.manifest().version,
            vec![definition(1), definition(2)],
            vec![],
        ),
    )
    .await
    .unwrap_err();
    assert!(
        error.to_string().contains("does not continue"),
        "a spent id is not a fresh allocation: {error}"
    );

    // Removal and allocation in one list: 3 continues the mark.
    let mixed = commit(
        &removed,
        submission(
            removed.manifest().version,
            vec![definition(3)],
            vec![("org.example.note".to_string(), Some("x".to_string()))],
        ),
    )
    .await
    .unwrap();
    assert_eq!(ids(&mixed), vec![3], "1 removed and 3 allocated");
    assert_eq!(mixed.manifest().max_clustering_id, 3);
    assert_eq!(
        mixed.manifest().table_metadata["org.example.note"],
        "x",
        "the metadata patch publishes with the definitions"
    );

    // An empty target clears the set and still preserves the mark.
    let cleared = commit(&mixed, submission(mixed.manifest().version, vec![], vec![]))
        .await
        .unwrap();
    assert!(cleared.manifest().clustering.is_empty());
    assert_eq!(cleared.manifest().max_clustering_id, 3);
}

#[tokio::test]
async fn complete_list_keeps_current_definitions_by_identity() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let dataset = commit(
        &dataset,
        submission(dataset.manifest().version, vec![definition(1)], vec![]),
    )
    .await
    .unwrap();

    // order is not significant: the retained entry may come after the new one
    let dataset = commit(
        &dataset,
        submission(
            dataset.manifest().version,
            vec![definition(2), definition(1)],
            vec![],
        ),
    )
    .await
    .unwrap();
    assert_eq!(ids(&dataset), vec![1, 2]);
    let version = dataset.manifest().version;

    let changed = ClusteringDefinition {
        columns: vec![0],
        provider: "org.example.other".into(),
        ..definition(2)
    };
    let cases: Vec<(Vec<ClusteringDefinition>, &str)> = vec![
        (vec![definition(1), changed], "immutable"),
        (
            vec![definition(1), definition(2), definition(4)],
            "does not continue",
        ),
    ];
    for (desired, expected) in cases {
        let error = commit(&dataset, submission(version, desired, vec![]))
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::CommitConflict { .. }),
            "{expected}: {error:?}"
        );
        assert!(error.to_string().contains(expected), "{expected}: {error}");
    }
    // malformed requests are plain input errors
    let error = commit(
        &dataset,
        submission(
            version,
            vec![definition(1), definition(2), definition(3), definition(3)],
            vec![],
        ),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("more than once"), "{error}");
    let error = commit(
        &dataset,
        submission(
            version,
            vec![
                definition(1),
                definition(2),
                ClusteringDefinition {
                    columns: vec![7],
                    ..definition(3)
                },
            ],
            vec![],
        ),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("not in the schema"), "{error}");
    let reopened = Dataset::open(&dir).await.unwrap();
    assert_eq!(reopened.manifest().version, version, "nothing landed");
    assert_eq!(reopened.manifest().max_clustering_id, 2);

    // the identical list adds nothing: the commit is a no-op for definitions
    let dataset = commit(
        &dataset,
        submission(
            version,
            vec![definition(1), definition(2)],
            vec![("org.example.note".to_string(), Some("x".to_string()))],
        ),
    )
    .await
    .unwrap();
    assert_eq!(ids(&dataset), vec![1, 2]);
    assert_eq!(dataset.manifest().max_clustering_id, 2);
    assert_eq!(dataset.manifest().table_metadata["org.example.note"], "x");
}

#[tokio::test]
async fn rebased_submission_is_rechecked_against_the_landing_manifest() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let read_version = dataset.manifest().version;

    // X allocates id 1 first
    let after_x = commit(
        &dataset,
        submission(read_version, vec![definition(1)], vec![]),
    )
    .await
    .unwrap();
    assert_eq!(after_x.manifest().max_clustering_id, 1);

    // W built its list from the same stale snapshot and also wants id 1, with
    // different content: Lance rebases the UpdateConfig (its record touches no
    // common key), and the landing check refuses the stale id
    let stale = ClusteringDefinition {
        provider: "org.example.other".into(),
        ..definition(1)
    };
    let error = commit(&dataset, submission(read_version, vec![stale], vec![]))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::CommitConflict { .. }), "{error:?}");
    assert!(error.to_string().contains("immutable"), "{error}");

    // the same configuration from the same stale snapshot is a no-op after
    // the rebase: nothing is allocated twice
    let dataset = commit(
        &dataset,
        submission(
            read_version,
            vec![definition(1)],
            vec![(ACTIVE_KEY.to_string(), Some("1".to_string()))],
        ),
    )
    .await
    .unwrap();
    assert_eq!(ids(&dataset), vec![1]);
    assert_eq!(dataset.manifest().max_clustering_id, 1);
    assert_eq!(dataset.manifest().table_metadata[ACTIVE_KEY], "1");

    // an ordinary append from the stale snapshot rebases over the allocation
    // and keeps it
    let appended = InsertBuilder::new(Arc::new(
        after_x.checkout_version(read_version).await.unwrap(),
    ))
    .with_params(&WriteParams {
        mode: WriteMode::Append,
        ..Default::default()
    })
    .execute(vec![
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int32Array::from_iter_values(100..110))],
        )
        .unwrap(),
    ])
    .await
    .unwrap();
    assert_eq!(ids(&appended), vec![1]);
    assert_eq!(appended.manifest().max_clustering_id, 1);
    assert_eq!(appended.manifest().fragments.len(), 3);
}

#[tokio::test]
async fn ambiguous_success_is_recognized_by_transaction_identity() {
    use crate::utils::test::{AmbiguousCommitHandler, AmbiguousFailure};
    let dir = TempStrDir::default();
    let handler = Arc::new(AmbiguousCommitHandler::default());
    let dataset = Dataset::write(
        rows(0..100),
        &dir,
        Some(WriteParams {
            max_rows_per_file: 50,
            commit_handler: Some(handler.clone()),
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // the store applies the write but reports a conflict; the commit path
    // reads the version back and recognizes its own transaction (by the
    // serialized record, which includes the uuid), so the caller sees success
    handler.fail_next(AmbiguousFailure::LandAndConflict);
    let submitted = submission(
        dataset.manifest().version,
        vec![definition(1)],
        vec![(ACTIVE_KEY.to_string(), Some("1".to_string()))],
    );
    let uuid = submitted.1.uuid.clone();
    let committed = commit(&dataset, submitted).await.unwrap();
    assert_eq!(ids(&committed), vec![1]);
    assert_eq!(committed.manifest().table_metadata[ACTIVE_KEY], "1");
    let recorded = committed.read_transaction().await.unwrap().unwrap();
    assert_eq!(recorded.uuid, uuid);
}

#[tokio::test]
async fn restored_manifest_keeps_the_mark_for_later_submissions() {
    let dir = TempStrDir::default();
    let dataset = write_dataset(&dir).await;
    let dataset = commit(
        &dataset,
        submission(dataset.manifest().version, vec![definition(1)], vec![]),
    )
    .await
    .unwrap();
    let with_one = dataset.manifest().version;
    let dataset = commit(
        &dataset,
        submission(
            dataset.manifest().version,
            vec![definition(1), definition(2)],
            vec![(ACTIVE_KEY.to_string(), Some("2".to_string()))],
        ),
    )
    .await
    .unwrap();

    let mut restored = dataset.checkout_version(with_one).await.unwrap();
    restored.restore().await.unwrap();
    // definitions merge and the mark never goes down; the provider keys are
    // the restored version's
    assert_eq!(ids(&restored), vec![1, 2]);
    assert_eq!(restored.manifest().max_clustering_id, 2);
    assert!(!restored.manifest().table_metadata.contains_key(ACTIVE_KEY));

    // a list built from the pre-restore snapshot still describes the current
    // definitions exactly, so the next allocation continues at 3
    let dataset = commit(
        &restored,
        submission(
            restored.manifest().version,
            vec![definition(1), definition(2), definition(3)],
            vec![(ACTIVE_KEY.to_string(), Some("3".to_string()))],
        ),
    )
    .await
    .unwrap();
    assert_eq!(ids(&dataset), vec![1, 2, 3]);
    assert_eq!(dataset.manifest().max_clustering_id, 3);
}
