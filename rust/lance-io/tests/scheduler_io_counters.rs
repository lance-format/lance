// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The global IO counters, from both schedulers.
//!
//! Standalone process: `iops_counter()` and `bytes_read_counter()` are
//! process-wide, so sharing a process with other reads would make the exact
//! counts here somebody else's business too. Both cases run in the one test
//! for the same reason.

use std::sync::Arc;

use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::traits::{Reader, Writer};
use lance_io::{bytes_read_counter, iops_counter};
use object_store::path::Path;
use tokio::io::AsyncWriteExt;

/// Both schedulers answer an empty range without a request and count the
/// requested length for the rest, so their contribution to the counters is
/// the same.
///
/// The two modes are measured in one test, in sequence: the counters are
/// process-wide, so two tests measuring them would each see the other's
/// reads.
#[tokio::test]
async fn both_schedulers_publish_the_io_counters() {
    for use_lite_scheduler in [false, true] {
        assert_counters_for(use_lite_scheduler).await;
    }
}

async fn assert_counters_for(use_lite_scheduler: bool) {
    let store = Arc::new(ObjectStore::memory());
    let path = Path::from("data.bin");
    let mut writer = store.create(&path).await.unwrap();
    writer.write_all(&vec![7u8; 1024]).await.unwrap();
    Writer::shutdown(&mut writer).await.unwrap();

    let reader: Arc<dyn Reader> = Arc::from(store.open(&path).await.unwrap());
    let config = SchedulerConfig {
        use_lite_scheduler: Some(use_lite_scheduler),
        ..SchedulerConfig::default_for_testing()
    };
    let scheduler = ScanScheduler::new(store, config);

    let iops_before = iops_counter();
    let bytes_before = bytes_read_counter();

    let read = scheduler
        .submit_request(reader.clone(), vec![0..100, 100..300], 0, false)
        .await
        .unwrap();
    assert_eq!(read.iter().map(|bytes| bytes.len()).sum::<usize>(), 300);

    // An empty range reaches the scheduler, and neither path reads for it.
    let empty = scheduler
        .submit_request(reader, vec![300..300], 0, false)
        .await
        .unwrap();
    assert!(empty[0].is_empty());

    assert_eq!(
        iops_counter() - iops_before,
        2,
        "use_lite_scheduler={use_lite_scheduler}"
    );
    assert_eq!(
        bytes_read_counter() - bytes_before,
        300,
        "use_lite_scheduler={use_lite_scheduler}"
    );
}
