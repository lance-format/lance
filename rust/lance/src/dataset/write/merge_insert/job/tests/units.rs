// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::record_batch;
use datafusion::execution::memory_pool::{FairSpillPool, MemoryConsumer, MemoryPool};

#[rstest::rstest]
#[case::small(1024 * 1024, 1)]
#[case::uneven(8 * 1024 * 1024 + 7, 1)]
#[case::reservation_ceiling(120 * 1024 * 1024, 1)]
#[case::one_fragment_partition(150 * 1024 * 1024, 1)]
#[case::batch_ceiling(200 * 1024 * 1024, 2)]
#[case::large_row(512 * 1024 * 1024, 5)]
fn test_merge_insert_budget(#[case] bytes: usize, #[case] partitions: usize) {
    let options = LanceExecutionOptions {
        mem_pool_size: Some(bytes as u64),
        ..Default::default()
    };
    let budget = MergeInsertBudget::new(&options);
    assert_eq!(budget.fragment_partitions, partitions);
    assert_eq!(budget.batch_bytes, (bytes / 8).min(25 * 1024 * 1024));
    assert_eq!(budget.sorted_source_bytes, bytes / 8);
    let pool: Arc<dyn MemoryPool> = Arc::new(FairSpillPool::new(bytes));
    let headroom = MemoryConsumer::new("sort merge").register(&pool);
    headroom
        .try_grow(options.sort_spill_reservation_bytes())
        .unwrap();
    let sort = MemoryConsumer::new("sort")
        .with_can_spill(true)
        .register(&pool);
    let other = MemoryConsumer::new("join")
        .with_can_spill(true)
        .register(&pool);
    // Two spilling consumers share the space left after a merge reservation.
    sort.try_grow(2 * budget.batch_bytes).unwrap();
    drop(other);
    // A lone sort must fit an indivisible quarter-pool row and its copy.
    let row = bytes / 4;
    sort.try_resize(2 * row).unwrap();
    drop(headroom);
    drop(sort);
    // The final merge needs two input runs, each reserving twice the row.
    let first_run = MemoryConsumer::new("first run").register(&pool);
    let second_run = MemoryConsumer::new("second run").register(&pool);
    first_run.try_grow(2 * row).unwrap();
    second_run.try_grow(2 * row).unwrap();
}

#[rstest::rstest]
fn test_merge_insert_budget_below_one_batch(#[values(0, 1, 7)] bytes: u64) {
    let options = LanceExecutionOptions {
        mem_pool_size: Some(bytes),
        ..Default::default()
    };
    let budget = MergeInsertBudget::new(&options);
    assert_eq!(budget.batch_bytes, 1);
    assert_eq!(budget.sorted_source_bytes, 0);
    assert_eq!(budget.fragment_partitions, 1);
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
