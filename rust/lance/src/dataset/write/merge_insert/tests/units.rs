// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// The fill's plan depth must not grow with the number of columns the
/// source omits. Building it with a `DataFrame::with_column` per missing
/// field — the obvious way, and what this did before #9504 — nests one
/// `Projection` per column; the optimizer's recursive walk over that chain
/// aborted the process on a 315-column table. No other test in this suite
/// builds a partial-schema merge wide enough to notice, so a regression to
/// that form would surface in production rather than here.
///
/// Only the depth is asserted. What the fill produces is covered by the
/// partial-schema merge tests below.
#[test]
fn test_fill_missing_target_columns_depth_is_width_independent() {
    fn fill_depth(num_cols: usize) -> usize {
        fn depth(plan: &LogicalPlan) -> usize {
            1 + plan.inputs().iter().map(|i| depth(i)).max().unwrap_or(0)
        }

        let ctx = SessionContext::new();
        let dataset_schema = Schema::new(
            (0..num_cols)
                .map(|i| Field::new(format!("c{i}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        );
        let target =
            Arc::new(MemTable::try_new(Arc::new(dataset_schema.clone()), vec![vec![]]).unwrap());
        // The source carries only the join key, so every other column is
        // filled from the target side.
        let source = Arc::new(
            MemTable::try_new(
                Arc::new(Schema::new(vec![Field::new("c0", DataType::Int32, true)])),
                vec![vec![]],
            )
            .unwrap(),
        );
        let joined = ctx
            .read_table(target)
            .unwrap()
            .alias("target")
            .unwrap()
            .join(
                ctx.read_table(source).unwrap().alias("source").unwrap(),
                JoinType::Left,
                &["\"c0\""],
                &["\"c0\""],
                None,
            )
            .unwrap();
        let source_field_names: HashSet<String> = std::iter::once("c0".to_string()).collect();
        let filled =
            fill_missing_target_columns(joined, &dataset_schema, &source_field_names, |_| true)
                .unwrap();
        depth(filled.logical_plan())
    }

    assert_eq!(fill_depth(8), fill_depth(512));
}

#[test]
fn test_inserted_key_tracker_preserves_logical_nulls() {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Null, true)])),
        vec![Arc::new(NullArray::new(2))],
    )
    .unwrap();
    let mut tracker = InsertedKeyTracker::default();
    let on_columns = ["id".to_string()];

    assert!(tracker.insert(&batch, 0, &on_columns).unwrap());
    assert!(tracker.insert(&batch, 1, &on_columns).unwrap());
}

#[test]
fn test_fragment_update_partitions_include_working_memory() {
    assert_eq!(
        max_fragment_update_partitions(150 * 1024 * 1024),
        1,
        "a 150 MiB pool has room for one merge reservation and two capped working batches"
    );
}

#[test]
fn test_merge_insert_pool_scales_without_changing_plan_parallelism() {
    let options = merge_insert_execution_options();
    let expected_pool_size = LanceExecutionOptions {
        use_spilling: true,
        batch_size: Some(INDEXED_JOIN_BATCH_ROWS),
        target_partition: Some(get_num_compute_intensive_cpus().min(8)),
        ..Default::default()
    }
    .mem_pool_size();

    assert_eq!(options.mem_pool_size(), expected_pool_size);
    assert_eq!(options.target_partition, None);
}

#[tokio::test]
async fn test_cloned_merge_jobs_do_not_reuse_failed_spill_accounting() {
    let batch = record_batch!(("id", Int32, [1])).unwrap();
    let dataset = Arc::new(
        Dataset::write(
            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
            "memory://",
            None,
        )
        .await
        .unwrap(),
    );
    let mut first_job = MergeInsertBuilder::try_new(dataset, vec!["id".to_string()])
        .unwrap()
        .when_matched(WhenMatched::UpdateAll)
        .try_build()
        .unwrap();
    first_job.spill_execution_options.max_temp_directory_size = Some(1);
    first_job.spill_session_context = new_session_context(&first_job.spill_execution_options);

    let second_job = first_job.clone();
    let first_manager = first_job
        .spill_session_context
        .runtime_env()
        .disk_manager
        .clone();
    let second_manager = second_job
        .spill_session_context
        .runtime_env()
        .disk_manager
        .clone();

    assert!(
        !Arc::ptr_eq(&first_manager, &second_manager),
        "cloned executable jobs must own independent disk managers"
    );
    let retry_job = second_job.clone_for_retry();
    assert!(
        Arc::ptr_eq(
            &second_manager,
            &retry_job.spill_session_context.runtime_env().disk_manager
        ),
        "retry attempts within one execution must share a disk manager"
    );

    let rejected = first_manager
        .create_tmp_file("testing rejected merge job spill")
        .unwrap();
    let mut writer = rejected.open_writer().unwrap();
    let error = writer.write_all(&[0, 0]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("used disk space during the spilling process has exceeded")
    );
    drop(writer);
    drop(rejected);

    let next_spill = second_manager
        .create_tmp_file("testing independent merge job spill")
        .unwrap();
    let mut writer = next_spill.open_writer().unwrap();
    writer.write_all(&[0]).unwrap();
    writer.flush().unwrap();
    drop(writer);
    assert_eq!(second_manager.used_disk_space(), 1);
}
