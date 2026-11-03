// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use arrow_array::record_batch;

#[test]
fn test_fragment_update_partitions_include_working_memory() {
    assert_eq!(
        max_fragment_update_partitions(150 * 1024 * 1024),
        1,
        "a 150 MiB pool has room for one merge reservation and two capped working batches"
    );
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
