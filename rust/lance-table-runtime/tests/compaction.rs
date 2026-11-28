// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::types::{Int32Type, UInt64Type};
use arrow_array::{RecordBatch, RecordBatchIterator, cast::AsArray, record_batch};
use futures::stream;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID, ROW_LAST_UPDATED_AT_VERSION};
use lance_file::reader::FileReader;
use lance_file::version::LanceFileVersion;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_table::format::IndexMetadata;
use lance_table::transaction::{Operation, Transaction};
use lance_table_runtime::optimize::{
    CompactionMode, CompactionOptions, CompactionPlan, CompactionScope, CompactionTask,
    CompactionTaskKind, RewriteResult, TaskData, commit_compaction, compact_files, plan_compaction,
};
use lance_table_runtime::{
    CommitBuilder, InsertBuilder, Table, TableBuilder, WriteMode, WriteParams,
};
use rstest::rstest;
use serde::{Serialize, de::DeserializeOwned};
use uuid::Uuid;

async fn write(table_uri: &str, batch: RecordBatch, params: &WriteParams) -> Table {
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    InsertBuilder::new(table_uri)
        .with_params(params)
        .execute_stream(reader)
        .await
        .unwrap()
}

async fn commit(table: &Table, operation: Operation) -> Table {
    CommitBuilder::new(table)
        .execute(Transaction::new(table.version_id(), operation, None))
        .await
        .unwrap()
}

async fn delete_offset(table: &Table, fragment_id: usize, offset: u32) -> Table {
    let fragment = table.get_fragment(fragment_id).unwrap();
    let deleted = fragment.extend_deletions([offset]).await.unwrap().unwrap();
    commit(
        table,
        Operation::Delete {
            updated_fragments: vec![deleted.into()],
            deleted_fragment_ids: vec![],
            predicate: "physical offset".into(),
        },
    )
    .await
}

async fn overlay_first_row(table: &Table) -> Table {
    let fragment = &table.get_fragments()[0];
    let projection = table.schema().project(&["value"]).unwrap();
    let address = u64::from(RowAddress::first_row(fragment.id() as u32));
    let batch = record_batch!((ROW_ADDR, UInt64, [address]), ("value", Int32, [100])).unwrap();
    let mut writer = fragment.write_overlay(&projection).await.unwrap();
    writer.write_batch(&batch).await.unwrap();
    let group = writer.finish().await.unwrap().unwrap();
    commit(
        table,
        Operation::DataOverlay {
            groups: vec![group],
        },
    )
    .await
}

fn round_trip<T: Serialize + DeserializeOwned>(value: &T) -> T {
    serde_json::from_slice(&serde_json::to_vec(value).unwrap()).unwrap()
}

fn values(batch: &RecordBatch) -> &[i32] {
    batch["value"].as_primitive::<Int32Type>().values()
}

fn system_values<'a>(batch: &'a RecordBatch, column: &str) -> &'a [u64] {
    batch[column].as_primitive::<UInt64Type>().values()
}

#[rstest]
#[case::address_ids(LanceFileVersion::V2_3, false)]
#[case::stable_ids(LanceFileVersion::V2_3, true)]
#[case::legacy(LanceFileVersion::Legacy, false)]
#[tokio::test]
async fn distributed_rewrite_preserves_rows_and_lineage(
    #[case] version: LanceFileVersion,
    #[case] stable_ids: bool,
) {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let params = WriteParams {
        max_rows_per_file: 3,
        max_rows_per_group: 3,
        data_storage_version: Some(version),
        enable_stable_row_ids: stable_ids,
        ..Default::default()
    };
    write(
        uri,
        record_batch!(("value", Int32, (0..9).collect::<Vec<i32>>())).unwrap(),
        &params,
    )
    .await;
    let append_params = WriteParams {
        mode: WriteMode::Append,
        ..params
    };
    let table = write(
        uri,
        record_batch!(("value", Int32, (9..12).collect::<Vec<i32>>())).unwrap(),
        &append_params,
    )
    .await;
    assert_eq!(table.count_fragments(), 4);
    let table = if version == LanceFileVersion::Legacy {
        table
    } else {
        overlay_first_row(&table).await
    };
    let table = delete_offset(&table, 0, 1).await;
    let mut table = delete_offset(&table, 1, 0).await;

    let mut columns = vec!["value", ROW_ID, ROW_ADDR];
    if stable_ids {
        columns.extend([ROW_CREATED_AT_VERSION, ROW_LAST_UPDATED_AT_VERSION]);
    }
    let projection = table
        .schema()
        .project_preserve_system_columns(&columns)
        .unwrap();
    let offsets = (0..10).collect::<Vec<_>>();
    let before = table.take(&offsets, &projection).await.unwrap();
    let first = if version == LanceFileVersion::Legacy {
        0
    } else {
        100
    };
    let expected = [first, 2, 4, 5, 6, 7, 8, 9, 10, 11];
    assert_eq!(values(&before), &expected);
    let plan = plan_compaction(
        &table,
        &CompactionOptions {
            target_rows_per_fragment: 12,
            batch_size: Some(2),
            io_buffer_size: Some(64 * 1024 * 1024),
            compaction_mode: Some(CompactionMode::Reencode),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(plan.num_tasks(), 1);
    let remote_plan: CompactionPlan = round_trip(&plan);
    assert_eq!(remote_plan, plan);
    let task = remote_plan.compaction_tasks().next().unwrap();
    let remote_task: CompactionTask = round_trip(&task);
    assert_eq!(remote_task, task);
    let worker = TableBuilder::from_uri(uri)
        .with_version(task.read_version)
        .load()
        .await
        .unwrap();
    let result = remote_task.execute(&worker).await.unwrap();
    let remote_result: RewriteResult = round_trip(&result);
    assert_eq!(remote_result, result);
    let metrics = commit_compaction(&mut table, vec![remote_result], plan.options())
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 4);
    assert_eq!(metrics.fragments_added, 1);

    let reopened = Table::open(uri).await.unwrap();
    assert_eq!(reopened.version_id(), table.version_id());
    assert_eq!(reopened.count_rows().await.unwrap(), 10);
    let fragment = &reopened.fragments()[0];
    assert_eq!(fragment.physical_rows, Some(10));
    assert!(fragment.deletion_file.is_none() && fragment.overlays.is_empty());
    let batch = reopened.take(&offsets, &projection).await.unwrap();
    assert_eq!(values(&batch), &expected);
    let addresses = (0..10)
        .map(|offset| u64::from(RowAddress::new_from_parts(fragment.id as u32, offset)))
        .collect::<Vec<_>>();
    assert_eq!(system_values(&batch, ROW_ADDR), addresses);
    assert_ne!(system_values(&before, ROW_ADDR), addresses);
    if stable_ids {
        assert_eq!(
            system_values(&batch, ROW_ID),
            &[0, 2, 4, 5, 6, 7, 8, 9, 10, 11]
        );
        assert_eq!(
            system_values(&batch, ROW_CREATED_AT_VERSION),
            &[1, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        );
        assert_eq!(
            system_values(&batch, ROW_LAST_UPDATED_AT_VERSION),
            system_values(&before, ROW_LAST_UPDATED_AT_VERSION),
        );
    } else {
        assert_eq!(system_values(&batch, ROW_ID), addresses);
    }
}

#[tokio::test]
async fn forced_binary_copy_preserves_source_pages() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let mut table = write(
        uri,
        record_batch!(("value", Int32, (0..12).collect::<Vec<i32>>())).unwrap(),
        &WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_3),
            max_rows_per_file: 4,
            enable_stable_row_ids: true,
            ..Default::default()
        },
    )
    .await;
    assert_eq!(table.count_fragments(), 3);
    let metrics = compact_files(
        &mut table,
        CompactionOptions {
            target_rows_per_fragment: 12,
            compaction_mode: Some(CompactionMode::ForceBinaryCopy),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(metrics.fragments_added, 1);
    assert_eq!(metrics.fragments_removed, 3);

    let reopened = Table::open(uri).await.unwrap();
    let file = &reopened.fragments()[0].files[0];
    let store = reopened.object_store(None).await.unwrap();
    let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
    let reader = scheduler
        .open_file_with_priority(
            &reopened.data_dir().join(file.path.as_str()),
            0,
            &file.file_size_bytes,
        )
        .await
        .unwrap();
    let metadata = FileReader::read_all_metadata(&reader).await.unwrap();
    let page_rows = metadata.column_infos[0]
        .page_infos
        .iter()
        .map(|page| page.num_rows)
        .collect::<Vec<_>>();
    assert_eq!(page_rows, [4, 4, 4]);
    let projection = reopened
        .schema()
        .project_preserve_system_columns(&["value", ROW_ID])
        .unwrap();
    let offsets = (0..12).collect::<Vec<_>>();
    let batch = reopened.take(&offsets, &projection).await.unwrap();
    assert_eq!(values(&batch), (0..12).collect::<Vec<_>>());
    assert_eq!(system_values(&batch, ROW_ID), offsets);
}

#[tokio::test]
async fn distributed_repack_keeps_fragment_addresses_deletions_and_overlays() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let table = write(
        uri,
        record_batch!(
            ("value", Int32, (0..6).collect::<Vec<i32>>()),
            ("extra", Int32, (10..16).collect::<Vec<i32>>())
        )
        .unwrap(),
        &WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_3),
            max_rows_per_file: 3,
            ..Default::default()
        },
    )
    .await;
    let fragment = table.get_fragment(0).unwrap();
    let schema = table.schema().project(&["extra"]).unwrap();
    let column = record_batch!(("extra", Int32, (10..13).collect::<Vec<i32>>())).unwrap();
    let replacement = fragment
        .write_columns(stream::iter([Ok(column)]), &schema)
        .await
        .unwrap();
    let table = commit(
        &table,
        Operation::DataReplacement {
            replacements: vec![replacement],
            data_change: false,
        },
    )
    .await;
    let table = overlay_first_row(&table).await;
    let mut table = delete_offset(&table, 0, 1).await;
    let first = table.fragments()[0].clone();
    let second = table.fragments()[1].clone();
    assert_eq!(first.files.len(), 2);
    let plan = plan_compaction(
        &table,
        &CompactionOptions {
            max_data_files_per_fragment: Some(1),
            scope: CompactionScope::RepackColumns,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert_eq!(plan.num_tasks(), 1);
    let task: CompactionTask = round_trip(&plan.compaction_tasks().next().unwrap());
    assert!(matches!(
        task.task.kind,
        CompactionTaskKind::RepackColumns { .. }
    ));
    let result = task.execute(&table).await.unwrap();
    let result: RewriteResult = round_trip(&result);
    assert_eq!(
        result.repacked_files.as_ref().unwrap().fragment_id,
        first.id
    );
    let metrics = commit_compaction(&mut table, vec![result], plan.options())
        .await
        .unwrap();
    assert_eq!(metrics.fragments_removed, 0);
    assert_eq!(metrics.files_removed, 2);
    assert_eq!(metrics.files_added, 1);

    let reopened = Table::open(uri).await.unwrap();
    assert_eq!(reopened.fragments()[1], second);
    let repacked = &reopened.fragments()[0];
    assert_eq!(repacked.id, first.id);
    assert_eq!(repacked.files.len(), 1);
    assert_eq!(repacked.deletion_file, first.deletion_file);
    assert_eq!(repacked.overlays, first.overlays);
    let projection = reopened
        .schema()
        .project_preserve_system_columns(&["value", "extra", ROW_ADDR])
        .unwrap();
    let batch = reopened.take(&[0, 1, 2, 3, 4], &projection).await.unwrap();
    assert_eq!(values(&batch), &[100, 2, 3, 4, 5]);
    assert_eq!(
        batch["extra"].as_primitive::<Int32Type>().values(),
        &[10, 12, 13, 14, 15]
    );
    let addresses = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)]
        .map(|(id, offset)| u64::from(RowAddress::new_from_parts(id, offset)));
    assert_eq!(system_values(&batch, ROW_ADDR), addresses);
}

#[tokio::test]
async fn fresh_coordinator_rejects_rewrite_over_concurrent_delete() {
    let dir = tempfile::tempdir().unwrap();
    let uri = dir.path().to_str().unwrap();
    let table = write(
        uri,
        record_batch!(("value", Int32, (0..6).collect::<Vec<i32>>())).unwrap(),
        &WriteParams {
            max_rows_per_file: 3,
            ..Default::default()
        },
    )
    .await;
    let plan = plan_compaction(&table, &CompactionOptions::default())
        .await
        .unwrap();
    let task = plan.compaction_tasks().next().unwrap();
    let result = task.execute(&table).await.unwrap();

    delete_offset(&table, 0, 1).await;
    let mut latest = Table::open(uri).await.unwrap();
    assert!(latest.version_id() > result.read_version);
    let error = commit_compaction(&mut latest, vec![result], plan.options())
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::RetryableCommitConflict { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("Delete"), "{error}");
    let reopened = Table::open(uri).await.unwrap();
    let batch = reopened
        .take(&[0, 1, 2, 3, 4], reopened.schema())
        .await
        .unwrap();
    assert_eq!(values(&batch), &[0, 2, 3, 4, 5]);
}

#[rstest]
#[case::address_ids(false)]
#[case::stable_ids(true)]
#[tokio::test]
async fn eager_compaction_without_bindings_preserves_opaque_index(#[case] stable_ids: bool) {
    let table = write(
        "memory://",
        record_batch!(("value", Int32, (0..9).collect::<Vec<i32>>())).unwrap(),
        &WriteParams {
            max_rows_per_file: 3,
            enable_stable_row_ids: stable_ids,
            ..Default::default()
        },
    )
    .await;
    let protected = table.fragments()[0].clone();
    let index = IndexMetadata {
        uuid: Uuid::new_v4(),
        fields: vec![table.schema().field_id("value").unwrap()],
        name: "opaque_index".into(),
        dataset_version: table.version_id(),
        fragment_bitmap: Some([protected.id as u32].into_iter().collect()),
        index_version: 42,
        index_details: Some(Arc::new(prost_types::Any {
            type_url: "type.example.com/FutureIndex".into(),
            value: vec![1, 2, 3],
        })),
        covering_fields: vec![],
        created_at: None,
        base_id: None,
        files: None,
        clustering_id: None,
        metadata: Default::default(),
    };
    let mut table = commit(
        &table,
        Operation::CreateIndex {
            new_indices: vec![index.clone()],
            removed_indices: vec![],
        },
    )
    .await;
    let plan = plan_compaction(&table, &CompactionOptions::default())
        .await
        .unwrap();
    assert_eq!(plan.num_tasks(), 1);
    assert_eq!(plan.tasks[0].fragments.len(), 2);
    assert!(
        !plan.tasks[0]
            .fragments
            .iter()
            .any(|fragment| fragment.id == protected.id)
    );

    let task = CompactionTask {
        task: TaskData::rewrite_fragments(vec![protected.clone()]),
        read_version: table.version_id(),
        options: plan.options().clone(),
    };
    let result = task.execute(&table).await.unwrap();
    let error = commit_compaction(&mut table, vec![result], plan.options())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }), "{error}");
    assert!(error.to_string().contains("opaque_index"), "{error}");

    let metrics = compact_files(&mut table, plan.options).await.unwrap();
    assert_eq!(metrics.fragments_removed, 2);
    assert_eq!(table.fragments()[0], protected);
    let stored = table.load_index_metadata().await.unwrap();
    assert_eq!(*stored, vec![index]);
}
