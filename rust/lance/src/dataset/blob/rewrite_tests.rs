// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Acceptance tests for [`BlobV2BatchRewritePlan`], driven through its public API
//! the way compaction drives it: scan with row addresses, build one plan from the
//! dataset and batch schemas, transform every batch, and publish the result as new
//! fragments of the same dataset.

use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayRef, LargeBinaryArray, LargeListArray, ListArray, RecordBatch, RecordBatchIterator,
    StringArray, StructArray, UInt32Array, UInt64Array,
};
use arrow_buffer::{OffsetBuffer, ScalarBuffer};
use arrow_schema::{DataType, Field, Fields, Schema, Schema as ArrowSchema};
use futures::TryStreamExt;
use lance_arrow::RecordBatchExt;
use lance_core::ROW_ID;
use lance_core::datatypes::{BLOB_V2_LOGICAL_TYPE, BlobKind, BlobV2Layout};
use lance_core::utils::tempfile::TempStrDir;
use lance_file::version::LanceFileVersion;
use lance_table::format::Fragment;
use lance_table::rowids::RowIdSequence;
use lance_table::rowids::version::RowDatasetVersionSequence;

use crate::Result;
use crate::blob::{BlobArrayBuilder, BlobFieldOptions, blob_field_with_options};
use crate::blob_field;
use crate::dataset::Dataset;
use crate::dataset::blob::rewrite::BlobV2BatchRewritePlan;
use crate::dataset::blob::test_util::managed_blob_schema;
use crate::dataset::cleanup::{CleanupPolicy, cleanup_old_versions};
use crate::dataset::optimize::execution::rechunk_row_lineage;
use crate::dataset::rowids::{RowLineage, place_row_lineage};
use crate::dataset::scanner::ColumnOrdering;
use crate::dataset::transaction::{Operation, RewriteGroup, Transaction};
use crate::dataset::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};

/// Destinations here are written at V2_2, which is the version at which
/// compaction turns managed preservation on.
const PRESERVE_MANAGED: bool = true;

/// A Blob v2 dataset with managed blobs off, split into one-row fragments so a
/// scan yields more than one batch.
async fn blob_v2_dataset(dir: &TempStrDir, rows: usize) -> Arc<Dataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        blob_field("blob", true),
    ]));
    let mut blobs = BlobArrayBuilder::new(rows);
    for row in 0..rows {
        blobs.push_bytes(vec![row as u8; 16]).unwrap();
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..rows as u32)),
            blobs.finish().unwrap(),
        ],
    )
    .unwrap();
    Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                max_rows_per_file: 1,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    )
}

async fn scan_with_row_addresses(dataset: &Dataset) -> Vec<RecordBatch> {
    let mut scan = dataset.scan();
    scan.with_row_address();
    scan.try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap()
}

/// Build the plan the way compaction does, from the dataset schema and the
/// schema of a row-address scan of it.
fn plan_for(
    dataset: &Arc<Dataset>,
    input_schema: &ArrowSchema,
    keep_row_addr: bool,
    preserve_managed: bool,
) -> Result<BlobV2BatchRewritePlan> {
    BlobV2BatchRewritePlan::try_new(
        dataset.schema(),
        input_schema,
        keep_row_addr,
        preserve_managed,
    )
}

#[tokio::test]
async fn the_output_schema_is_known_before_any_batch_and_every_batch_matches_it() {
    let dir = TempStrDir::default();
    let dataset = blob_v2_dataset(&dir, 4).await;
    let batches = scan_with_row_addresses(&dataset).await;
    assert!(
        batches.len() >= 2,
        "need at least two batches to show the schema is stable across them"
    );

    let plan = plan_for(&dataset, batches[0].schema().as_ref(), false, false).unwrap();
    let output_schema = plan.output_schema().clone();

    for batch in batches {
        let rewritten = plan.transform_batch(&dataset, batch).await.unwrap();
        assert_eq!(rewritten.schema(), output_schema);
    }
}

#[tokio::test]
async fn retain_row_addresses_controls_only_the_output_column() {
    let dir = TempStrDir::default();
    let dataset = blob_v2_dataset(&dir, 2).await;
    let input_schema = scan_with_row_addresses(&dataset).await[0].schema();

    let dropped = plan_for(&dataset, input_schema.as_ref(), false, false)
        .unwrap()
        .output_schema()
        .clone();
    let retained = plan_for(&dataset, input_schema.as_ref(), true, false)
        .unwrap()
        .output_schema()
        .clone();

    assert!(dropped.column_with_name(lance_core::ROW_ADDR).is_none());
    assert!(retained.column_with_name(lance_core::ROW_ADDR).is_some());
    assert_eq!(dropped.fields().len() + 1, retained.fields().len());
}

#[tokio::test]
async fn a_blob_column_already_in_logical_layout_needs_no_rewrite() {
    let dir = TempStrDir::default();
    let dataset = blob_v2_dataset(&dir, 2).await;

    let DataType::Struct(logical_fields) = BLOB_V2_LOGICAL_TYPE.clone() else {
        panic!("the logical blob v2 type should be a struct");
    };
    assert_eq!(
        BlobV2Layout::classify(&logical_fields),
        Some(BlobV2Layout::Logical),
        "this test only pins the fix if its input column really is in logical layout"
    );

    let input_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt32, false),
        Field::new("blob", DataType::Struct(logical_fields.clone()), true),
        Field::new(lance_core::ROW_ADDR, DataType::UInt64, false),
    ]));

    let plan = plan_for(&dataset, input_schema.as_ref(), false, false).unwrap();
    let blobs = StructArray::new(
        logical_fields,
        vec![
            Arc::new(LargeBinaryArray::from_iter_values([
                b"zero".as_slice(),
                b"one".as_slice(),
            ])),
            Arc::new(StringArray::new_null(2)),
            Arc::new(UInt64Array::new_null(2)),
            Arc::new(UInt64Array::new_null(2)),
        ],
        None,
    );
    let input = RecordBatch::try_new(
        input_schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..2)),
            Arc::new(blobs),
            Arc::new(UInt64Array::from_iter_values(0..2)),
        ],
    )
    .unwrap();

    let output = plan.transform_batch(&dataset, input.clone()).await.unwrap();
    // The row address column is the only difference: a logical-layout blob column
    // is already in its final form and is handed back untouched.
    assert_eq!(output, input.drop_column(lance_core::ROW_ADDR).unwrap());
}

/// A Blob v2 array of `rows` payloads, each 16 bytes of the row index.
fn blob_values(rows: usize) -> ArrayRef {
    let mut blobs = BlobArrayBuilder::new(rows);
    for row in 0..rows {
        blobs.push_bytes(vec![row as u8; 16]).unwrap();
    }
    blobs.finish().unwrap()
}

/// Write a single-column dataset at V2_2, so its blob payloads are stored
/// out of line and the column scans back in descriptor layout.
async fn nested_dataset(dir: &TempStrDir, field: Field, array: ArrayRef) -> Arc<Dataset> {
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
    Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    )
}

/// The blob struct reached through a column, by descending `path` from the named
/// column through list items and struct children.
fn nested_blob_fields(schema: &ArrowSchema, column: &str, path: &[&str]) -> Fields {
    let mut data_type = schema.field_with_name(column).unwrap().data_type().clone();
    for step in path {
        data_type = match (&data_type, *step) {
            (DataType::List(item), "item") | (DataType::LargeList(item), "item") => {
                item.data_type().clone()
            }
            (DataType::Struct(children), name) => children
                .find(name)
                .map(|(_, child)| child.data_type().clone())
                .unwrap_or_else(|| panic!("struct has no child '{name}'")),
            (other, step) => panic!("cannot descend '{step}' into {other:?}"),
        };
    }
    let DataType::Struct(fields) = data_type else {
        panic!("the path {path:?} should end at the blob struct, got {data_type:?}");
    };
    fields
}

/// Assert that a nested descriptor column is found and rewritten to logical
/// layout, and return the batches scanned with row addresses.
async fn assert_nested_blob_is_rewritten(
    dataset: &Arc<Dataset>,
    column: &str,
    path: &[&str],
) -> Vec<RecordBatch> {
    let batches = scan_with_row_addresses(dataset).await;
    let input_schema = batches[0].schema();
    assert_eq!(
        BlobV2Layout::classify(&nested_blob_fields(&input_schema, column, path)),
        Some(BlobV2Layout::Descriptor),
        "this test only pins the fix if the column really arrives in descriptor layout"
    );

    let plan = plan_for(dataset, input_schema.as_ref(), false, false).unwrap();
    let output_schema = plan.output_schema().clone();
    assert_ne!(
        output_schema.field_with_name(column).unwrap().data_type(),
        input_schema.field_with_name(column).unwrap().data_type(),
        "a rewritten column must not keep its descriptor type"
    );
    assert_eq!(
        BlobV2Layout::classify(&nested_blob_fields(&output_schema, column, path)),
        Some(BlobV2Layout::Logical)
    );
    batches
}

/// Every blob payload reachable under a list column, in row order.
fn list_blob_payloads<O: arrow_array::OffsetSizeTrait>(
    batch: &RecordBatch,
    column: &str,
    struct_child: Option<&str>,
) -> Vec<Vec<u8>> {
    let list = batch
        .column_by_name(column)
        .unwrap()
        .as_any()
        .downcast_ref::<arrow_array::GenericListArray<O>>()
        .unwrap();
    let values = list.values();
    let blobs = match struct_child {
        None => values.clone(),
        Some(name) => values
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap()
            .column_by_name(name)
            .unwrap()
            .clone(),
    };
    let blobs = blobs.as_any().downcast_ref::<StructArray>().unwrap();
    let data = blobs
        .column_by_name("data")
        .unwrap()
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap();
    (0..data.len())
        .map(|row| data.value(row).to_vec())
        .collect()
}

#[tokio::test]
async fn a_list_of_descriptor_blobs_is_rewritten_not_passed_through() {
    let dir = TempStrDir::default();
    let item = blob_field("item", true);
    let values = blob_values(2);
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 1, 2]));
    let list: ArrayRef =
        Arc::new(ListArray::try_new(Arc::new(item.clone()), offsets, values, None).unwrap());
    let dataset = nested_dataset(
        &dir,
        Field::new("blobs", DataType::List(Arc::new(item)), true),
        list,
    )
    .await;

    let batches = assert_nested_blob_is_rewritten(&dataset, "blobs", &["item"]).await;

    let plan = plan_for(&dataset, batches[0].schema().as_ref(), false, false).unwrap();
    let mut payloads = Vec::new();
    for batch in batches {
        let rewritten = plan.transform_batch(&dataset, batch).await.unwrap();
        assert_eq!(rewritten.schema(), *plan.output_schema());
        payloads.extend(list_blob_payloads::<i32>(&rewritten, "blobs", None));
    }
    assert_eq!(payloads, vec![vec![0u8; 16], vec![1u8; 16]]);
}

#[tokio::test]
async fn a_large_list_of_descriptor_blobs_is_rewritten_not_passed_through() {
    let dir = TempStrDir::default();
    let item = blob_field("item", true);
    let values = blob_values(2);
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i64, 1, 2]));
    let list: ArrayRef =
        Arc::new(LargeListArray::try_new(Arc::new(item.clone()), offsets, values, None).unwrap());
    let dataset = nested_dataset(
        &dir,
        Field::new("blobs", DataType::LargeList(Arc::new(item)), true),
        list,
    )
    .await;

    let batches = assert_nested_blob_is_rewritten(&dataset, "blobs", &["item"]).await;

    let plan = plan_for(&dataset, batches[0].schema().as_ref(), false, false).unwrap();
    let mut payloads = Vec::new();
    for batch in batches {
        let rewritten = plan.transform_batch(&dataset, batch).await.unwrap();
        assert_eq!(rewritten.schema(), *plan.output_schema());
        payloads.extend(list_blob_payloads::<i64>(&rewritten, "blobs", None));
    }
    assert_eq!(payloads, vec![vec![0u8; 16], vec![1u8; 16]]);
}

#[tokio::test]
async fn a_list_of_structs_containing_a_blob_is_rewritten_not_passed_through() {
    let dir = TempStrDir::default();
    let item_fields: Fields = vec![
        Field::new("name", DataType::Utf8, false),
        blob_field("blob", true),
    ]
    .into();
    let values: ArrayRef = Arc::new(
        StructArray::try_new(
            item_fields.clone(),
            vec![
                Arc::new(StringArray::from_iter_values(["zero", "one"])) as ArrayRef,
                blob_values(2),
            ],
            None,
        )
        .unwrap(),
    );
    let item = Arc::new(Field::new("item", DataType::Struct(item_fields), true));
    let offsets = OffsetBuffer::new(ScalarBuffer::from(vec![0i32, 1, 2]));
    let list: ArrayRef = Arc::new(ListArray::try_new(item.clone(), offsets, values, None).unwrap());
    let dataset = nested_dataset(&dir, Field::new("items", DataType::List(item), true), list).await;

    let batches = assert_nested_blob_is_rewritten(&dataset, "items", &["item", "blob"]).await;

    let plan = plan_for(&dataset, batches[0].schema().as_ref(), false, false).unwrap();
    let mut payloads = Vec::new();
    for batch in batches {
        let rewritten = plan.transform_batch(&dataset, batch).await.unwrap();
        assert_eq!(rewritten.schema(), *plan.output_schema());
        payloads.extend(list_blob_payloads::<i32>(&rewritten, "items", Some("blob")));
    }
    assert_eq!(payloads, vec![vec![0u8; 16], vec![1u8; 16]]);
}

/// The managed blob column used by the acceptance tests below. The small
/// thresholds force the payload out of the data file, so with managed blobs
/// enabled it becomes a Lance-owned object reached through a descriptor.
fn managed_blob_column() -> Field {
    blob_field_with_options(
        "blob",
        true,
        BlobFieldOptions {
            inline_size_threshold: Some(4),
            dedicated_size_threshold: Some(NonZeroUsize::new(8).unwrap()),
        },
    )
}

/// A one-row managed Blob v2 dataset holding `payload` in primary storage.
async fn managed_blob_dataset(dir: &TempStrDir, payload: &[u8]) -> Arc<Dataset> {
    let schema = Arc::new(managed_blob_schema(vec![
        Field::new("id", DataType::UInt32, false),
        managed_blob_column(),
    ]));
    let mut blobs = BlobArrayBuilder::new(1);
    blobs.push_bytes(payload).unwrap();
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..1)),
            blobs.finish().unwrap(),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    let stored = dataset.take_blobs_by_indices(&[0], "blob").await.unwrap();
    assert_eq!(
        stored[0].as_ref().unwrap().kind(),
        BlobKind::Managed,
        "the source payload must already be a managed object, otherwise this \
         test would be about materialization rather than preservation"
    );
    dataset
}

/// A `rows`-row managed Blob v2 dataset with stable row ids, where row `i`
/// holds [`named_payload`]`("row", i)` so every row's payload is distinct.
async fn stable_row_id_managed_blob_dataset(dir: &TempStrDir, rows: usize) -> Arc<Dataset> {
    let schema = Arc::new(managed_blob_schema(vec![
        Field::new("id", DataType::UInt32, false),
        managed_blob_column(),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(UInt32Array::from_iter_values(0..rows as u32)),
            named_blob_values("row", rows),
        ],
    )
    .unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch)], schema),
            dir,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                enable_stable_row_ids: true,
                ..Default::default()
            }),
        )
        .await
        .unwrap(),
    );
    assert!(dataset.manifest.uses_stable_row_ids());
    dataset
}

/// The backing object a blob row resolves to: its path, byte range and kind.
///
/// A preserved reference and a fresh copy of the same payload both read back
/// equal bytes, so only the object identity distinguishes them.
async fn blob_object_identity(
    dataset: &Arc<Dataset>,
    column: &str,
) -> (String, u64, u64, BlobKind) {
    let blobs = dataset.take_blobs_by_indices(&[0], column).await.unwrap();
    let blob = blobs[0].as_ref().unwrap();
    (
        blob.data_path().to_string(),
        blob.position(),
        blob.size(),
        blob.kind(),
    )
}

#[tokio::test]
async fn a_same_dataset_rewrite_keeps_its_preserved_payload_through_cleanup() {
    let dir = TempStrDir::default();
    let payload = vec![7u8; 64];
    let dataset = managed_blob_dataset(&dir, &payload).await;

    let source_object = blob_object_identity(&dataset, "blob").await;
    let old_fragments = dataset.fragments().to_vec();
    let batches = scan_with_row_addresses(&dataset).await;
    assert_eq!(batches.len(), 1);
    let plan = plan_for(
        &dataset,
        batches[0].schema().as_ref(),
        false,
        PRESERVE_MANAGED,
    )
    .unwrap();
    let transformed = plan
        .transform_batch(&dataset, batches[0].clone())
        .await
        .unwrap();
    assert_eq!(transformed.schema(), *plan.output_schema());
    assert_eq!(
        BlobV2Layout::classify(
            transformed
                .column_by_name("blob")
                .unwrap()
                .as_struct()
                .fields()
        ),
        Some(BlobV2Layout::Prepared),
        "a preserved managed reference must arrive in prepared layout"
    );

    // Write the rewritten rows as new fragments of this same dataset and swap
    // them in, the way compaction does.
    let committed = publish_rewrite(&dataset, old_fragments, transformed)
        .await
        .unwrap();

    // Drop the pre-rewrite snapshot: holding it open would protect the old
    // files and hide what cleanup does.
    drop(dataset);
    let removed = cleanup_old_versions(
        &committed,
        CleanupPolicy {
            before_version: Some(committed.version_id()),
            delete_unverified: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert!(
        removed.data_files_removed > 0,
        "the rewrite left every source file in use, so this test would not \
         exercise cleanup at all"
    );
    assert!(
        removed.old_versions > 0,
        "the versions holding the source fragments are still readable, so \
         nothing stops them from keeping the payload alive"
    );

    // The payload object survived, and the committed descriptor is still a
    // reference to it rather than a fresh copy.
    let reopened = Arc::new(Dataset::open(&dir).await.unwrap());
    let read = reopened.take_blobs_by_indices(&[0], "blob").await.unwrap();
    let blob = read[0].as_ref().unwrap();
    assert_eq!(blob.kind(), BlobKind::Managed);
    assert_eq!(blob.read().await.unwrap().as_ref(), payload);
    // Equal bytes alone would also hold for a fresh copy, so pin the object:
    // the committed descriptor still points at the very bytes the source row
    // did, at the same path, offset and length.
    assert_eq!(
        blob_object_identity(&reopened, "blob").await,
        source_object,
        "the rewrite published a different object instead of preserving the \
         source reference"
    );
}

/// A rewrite that reorders rows under stable row ids: each row id has to end
/// up on the blob it started on, not on whichever row landed at its old
/// offset. The lineage is therefore built from the written order rather than
/// carried over positionally, and it has to survive cleanup of the old files.
#[tokio::test]
async fn a_reordered_rewrite_with_stable_row_ids_keeps_every_row_id_on_its_blob() {
    const ROWS: usize = 4;
    let dir = TempStrDir::default();
    let dataset = stable_row_id_managed_blob_dataset(&dir, ROWS).await;

    // Scan in reverse `id` order, so the rewritten fragment holds the rows in
    // the opposite order from the one they were written in.
    let batch = dataset
        .scan()
        .with_row_id()
        .with_row_address()
        .order_by(Some(vec![ColumnOrdering::desc_nulls_last("id".into())]))
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    assert_eq!(batch.num_rows(), ROWS);
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_primitive::<UInt32Type>();
    let row_ids = batch
        .column_by_name(ROW_ID)
        .unwrap()
        .as_primitive::<UInt64Type>();
    assert_eq!(
        ids.values(),
        &(0..ROWS as u32).rev().collect::<Vec<_>>(),
        "the scan did not reverse the rows, so this test would not reorder anything"
    );
    // What each row id must still read back after the rewrite, captured
    // before any of it moves.
    let expected: Vec<(u64, u32, Vec<u8>)> = (0..ROWS)
        .map(|row| {
            (
                row_ids.value(row),
                ids.value(row),
                named_payload("row", ids.value(row) as usize),
            )
        })
        .collect();

    // `_rowid` is not a field of the dataset schema, so the plan would carry
    // it through as an ordinary column of the new fragment. Drop it from the
    // input and keep the ids separately, in the order the rows are written.
    let written_row_ids: Vec<u64> = row_ids.values().to_vec();
    let batch = batch.drop_column(ROW_ID).unwrap();

    let old_fragments = dataset.fragments().to_vec();
    let plan = plan_for(&dataset, batch.schema().as_ref(), false, PRESERVE_MANAGED).unwrap();
    let transformed = plan.transform_batch(&dataset, batch).await.unwrap();
    assert_eq!(transformed.schema(), *plan.output_schema());

    let written = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        })
        .execute_uncommitted(vec![transformed])
        .await
        .unwrap();
    let Operation::Append {
        fragments: mut new_fragments,
    } = written.operation
    else {
        unreachable!("an uncommitted insert always carries Append");
    };

    // The rows moved, so `rechunk_row_lineage` -- which reassigns the old
    // sequences positionally -- would hand each offset the wrong row id.
    // Build each fragment's sequence from the ids actually written to it.
    let source_version = dataset.manifest.version;
    let mut remaining = written_row_ids.as_slice();
    for fragment in new_fragments.iter_mut() {
        let (ids, rest) = remaining.split_at(fragment.physical_rows.unwrap());
        remaining = rest;
        let versions =
            RowDatasetVersionSequence::from_uniform_row_count(ids.len() as u64, source_version);
        let lineage = RowLineage {
            row_ids: RowIdSequence::try_from_iter(ids.iter().copied()).unwrap(),
            created_at: versions.clone(),
            last_updated_at: versions,
        };
        place_row_lineage(dataset.as_ref(), &lineage)
            .await
            .unwrap()
            .apply(fragment);
    }
    assert!(remaining.is_empty());

    let committed = CommitBuilder::new(dataset.clone())
        .execute(Transaction::new_from_version(
            dataset.manifest.version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments: old_fragments.clone(),
                    new_fragments,
                }],
                rewritten_indices: vec![],
                frag_reuse_index: None,
            },
        ))
        .await
        .unwrap();

    // Drop the pre-rewrite snapshot: holding it open would protect the old
    // files and hide whether the new ones carry the payload.
    drop(dataset);
    let removed = cleanup_old_versions(
        &committed,
        CleanupPolicy {
            before_version: Some(committed.version_id()),
            delete_unverified: true,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert!(
        removed.data_files_removed > 0,
        "the rewrite left every source file in use, so this test would not \
         exercise cleanup at all"
    );

    let reopened = Arc::new(Dataset::open(&dir).await.unwrap());
    for (row_id, id, payload) in expected {
        let blobs = reopened.take_blobs(&[row_id], "blob").await.unwrap();
        let blob = blobs[0]
            .as_ref()
            .unwrap_or_else(|| panic!("row id {row_id} lost its blob"));
        assert_eq!(
            blob.read().await.unwrap().as_ref(),
            payload,
            "row id {row_id} resolved to another row's payload"
        );
        let row = reopened
            .scan()
            .with_row_id()
            .filter(&format!("{ROW_ID} = {row_id}"))
            .unwrap()
            .project(&["id"])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        assert_eq!(row.num_rows(), 1, "row id {row_id} is no longer readable");
        assert_eq!(
            row.column_by_name("id")
                .unwrap()
                .as_primitive::<UInt32Type>()
                .value(0),
            id,
            "row id {row_id} moved to another row's values"
        );
    }
}

/// Distinguishable payloads for a struct child, so a child resolved against
/// the wrong field id is visible in the bytes rather than only in the layout.
fn named_blob_values(prefix: &str, rows: usize) -> ArrayRef {
    let mut blobs = BlobArrayBuilder::new(rows);
    for row in 0..rows {
        blobs.push_bytes(named_payload(prefix, row)).unwrap();
    }
    blobs.finish().unwrap()
}

fn named_payload(prefix: &str, row: usize) -> Vec<u8> {
    format!("payload-{prefix}-{row}").into_bytes()
}

/// A dataset with `info: Struct{a: Blob v2, b: Blob v2}`: two children of the
/// same type, so only the field-to-payload binding tells them apart.
async fn struct_of_two_blobs_dataset(dir: &TempStrDir, rows: usize) -> Arc<Dataset> {
    let children: Fields = vec![blob_field("a", true), blob_field("b", true)].into();
    let column: ArrayRef = Arc::new(
        StructArray::try_new(
            children.clone(),
            vec![named_blob_values("a", rows), named_blob_values("b", rows)],
            None,
        )
        .unwrap(),
    );
    nested_dataset(
        dir,
        Field::new("info", DataType::Struct(children), true),
        column,
    )
    .await
}

/// Write `transformed` as new fragments of `dataset` and swap them in for
/// `old_fragments`, the way compaction publishes a rewrite.
async fn publish_rewrite(
    dataset: &Arc<Dataset>,
    old_fragments: Vec<Fragment>,
    transformed: RecordBatch,
) -> Result<Dataset> {
    let written = InsertBuilder::new(dataset.clone())
        .with_params(&WriteParams {
            mode: WriteMode::Append,
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        })
        .execute_uncommitted(vec![transformed])
        .await?;
    let Operation::Append {
        fragments: mut new_fragments,
    } = written.operation
    else {
        unreachable!("an uncommitted insert always carries Append");
    };
    if dataset.manifest.uses_stable_row_ids() {
        rechunk_row_lineage(dataset.as_ref(), &mut new_fragments, &old_fragments).await?;
    }
    CommitBuilder::new(dataset.clone())
        .execute(Transaction::new_from_version(
            dataset.manifest.version,
            Operation::Rewrite {
                groups: vec![RewriteGroup {
                    old_fragments,
                    new_fragments,
                }],
                rewritten_indices: vec![],
                frag_reuse_index: None,
            },
        ))
        .await
}

/// Rewrite every batch, publish the result as a `Rewrite` of this same
/// dataset, and reopen it.
async fn rewrite_and_republish(dataset: &Arc<Dataset>, batches: Vec<RecordBatch>) -> Dataset {
    let old_fragments = dataset.fragments().to_vec();
    let plan = plan_for(dataset, batches[0].schema().as_ref(), false, false).unwrap();
    let mut transformed = Vec::with_capacity(batches.len());
    for batch in batches {
        transformed.push(plan.transform_batch(dataset, batch).await.unwrap());
    }
    assert_eq!(transformed.len(), 1, "this helper publishes a single batch");
    publish_rewrite(
        dataset,
        old_fragments,
        transformed.into_iter().next().unwrap(),
    )
    .await
    .unwrap()
}

async fn blob_payloads(dataset: &Arc<Dataset>, column: &str, rows: usize) -> Vec<Vec<u8>> {
    let indices = (0..rows as u64).collect::<Vec<_>>();
    let blobs = dataset
        .take_blobs_by_indices(&indices, column)
        .await
        .unwrap();
    let mut payloads = Vec::with_capacity(rows);
    for blob in blobs {
        payloads.push(blob.unwrap().read().await.unwrap().to_vec());
    }
    payloads
}

/// The full struct in the dataset's own child order is the shape every
/// in-crate caller builds, and it keeps each child's payload.
#[tokio::test]
async fn a_full_dataset_ordered_struct_keeps_every_child_payload() {
    let dir = TempStrDir::default();
    let rows = 3;
    let dataset = struct_of_two_blobs_dataset(&dir, rows).await;

    let batches = scan_with_row_addresses(&dataset).await;
    assert_eq!(
        batches[0]
            .column_by_name("info")
            .unwrap()
            .as_struct()
            .column_names(),
        vec!["a", "b"],
        "the scan must hand back the dataset's own child order"
    );

    let reopened = Arc::new(rewrite_and_republish(&dataset, batches).await);
    let expected_a = (0..rows)
        .map(|row| named_payload("a", row))
        .collect::<Vec<_>>();
    let expected_b = (0..rows)
        .map(|row| named_payload("b", row))
        .collect::<Vec<_>>();
    assert_eq!(blob_payloads(&reopened, "info.a", rows).await, expected_a);
    assert_eq!(blob_payloads(&reopened, "info.b", rows).await, expected_b);
}
