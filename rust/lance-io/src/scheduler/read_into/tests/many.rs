// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

fn allocations(
    work: &mut lance_core::utils::allocation::Reservation,
    starts: &[u64],
) -> Vec<(Range<u64>, ReservedBuffer)> {
    starts
        .iter()
        .map(|start| (*start..*start + 8, work.try_allocate(8).unwrap().unwrap()))
        .collect()
}

#[rstest]
#[tokio::test]
async fn batched_owned_reads_preserve_order_and_credit(#[values(false, true)] lite: bool) {
    let (scheduler, file, reader) = setup(lite, 8);
    let budget = AllocationBudget::new(24);
    let mut work = budget.try_reserve(24).unwrap();
    let requests = allocations(&mut work, &[24, 8, 24]);
    let pointers: Vec<_> = requests
        .iter()
        .map(|(_, buffer)| buffer.as_ref().as_ptr())
        .collect();
    let outputs = file.read_many_into(requests, 0).await.unwrap();
    assert_eq!(reader.batches.load(Ordering::SeqCst), usize::from(!lite));
    assert_eq!(*reader.calls.lock().unwrap(), [24, 8, 24]);
    for ((output, pointer), start) in outputs.iter().zip(pointers).zip([24, 8, 24]) {
        assert_eq!(output.as_ref().as_ptr(), pointer);
        assert_eq!(output.as_ref(), (start..start + 8).collect::<Vec<u8>>());
    }
    assert_eq!(scheduler.stats().iops, 3);
    assert_eq!(scheduler.stats().bytes_read, 24);
    assert_eq!(scheduler.reserved_bytes(), 0);
    assert_eq!(budget.reserved(), 24);
    for output in outputs {
        output.recycle(&mut work).unwrap();
    }
    assert_eq!(work.len(), 24);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn partial_batch_credit_is_released_before_fallback() {
    let (scheduler, file, reader) = setup(false, 8);
    let IoQueueType::Standard(queue) = &scheduler.io_queue else {
        unreachable!()
    };
    {
        let mut state = queue.state.lock().unwrap();
        state.iops_avail = 1;
        state.io_capacity = 1;
    }
    let budget = AllocationBudget::new(16);
    let mut work = budget.try_reserve(16).unwrap();
    let output = file
        .read_many_into(allocations(&mut work, &[8, 16]), 0)
        .await
        .unwrap();
    assert_eq!(reader.batches.load(Ordering::SeqCst), 0);
    assert_eq!(*reader.calls.lock().unwrap(), [8, 16]);
    assert_eq!(queue.state.lock().unwrap().iops_avail, 1);
    assert_eq!(scheduler.reserved_bytes(), 0);
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
}

#[tokio::test]
async fn batch_does_not_jump_queued_reads() {
    let (scheduler, file, reader) = setup(false, 8);
    let budget = AllocationBudget::new(24);
    let mut work = budget.try_reserve(24).unwrap();
    let mut blocker = file.read_single_into(0..8, 0, work.try_allocate(8).unwrap().unwrap());
    assert!(futures::poll!(&mut blocker).is_pending());
    let mut earlier = Box::pin(file.submit_single(8..16, 5));
    assert!(futures::poll!(&mut earlier).is_pending());
    let mut batch = file.read_many_into(allocations(&mut work, &[24, 32]), 10);
    assert!(futures::poll!(&mut batch).is_pending());
    assert_eq!(reader.batches.load(Ordering::SeqCst), 0);
    assert_eq!(*reader.calls.lock().unwrap(), [0]);
    reader.gate.add_permits(1);
    let output = blocker.await.unwrap();
    earlier.await.unwrap();
    let outputs = batch.await.unwrap();
    assert_eq!(*reader.calls.lock().unwrap(), [0, 8, 24, 32]);
    drop((output, outputs, work));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(scheduler.reserved_bytes(), 0);
}

#[tokio::test]
async fn batch_cancel_keeps_detached_backend_owner() {
    let (scheduler, file, reader) = setup(false, 16);
    let budget = AllocationBudget::new(16);
    let mut work = budget.try_reserve(16).unwrap();
    let mut pending = file.read_many_into(allocations(&mut work, &[8, 64]), 0);
    assert!(futures::poll!(&mut pending).is_pending());
    assert_eq!(reader.batches.load(Ordering::SeqCst), 1);
    drop((pending, work));
    assert_eq!(
        budget.reserved(),
        8,
        "only detached second read keeps memory"
    );
    assert_eq!(scheduler.reserved_bytes(), 0);
    let peer = file
        .read_single_into(
            16..24,
            0,
            budget.try_reserve(8).unwrap().allocate().unwrap(),
        )
        .await
        .unwrap();
    drop(peer);
    reader.gate.add_permits(1);
    wait_until("detached batch input released", || budget.reserved() == 0).await;
}

#[rstest]
#[case::replacement(96, true)]
#[case::failure(128, false)]
#[tokio::test]
async fn bad_batch_backend_releases_all_inputs(#[case] start: u64, #[case] replacement: bool) {
    let (scheduler, file, _) = setup(false, 16);
    let budget = AllocationBudget::new(16);
    let mut work = budget.try_reserve(16).unwrap();
    let error = file
        .read_many_into(allocations(&mut work, &[8, start]), 0)
        .await
        .unwrap_err();
    if replacement {
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("allocations unchanged"));
    } else {
        assert!(matches!(error, Error::IO { .. }));
        assert!(error.to_string().contains("controlled read failure"));
    }
    drop(work);
    assert_eq!(budget.reserved(), 0);
    assert_eq!(scheduler.reserved_bytes(), 0);
}

#[tokio::test]
async fn invalid_batch_is_rejected_before_any_read() {
    let (scheduler, file, reader) = setup(false, 16);
    let budget = AllocationBudget::new(16);
    let mut work = budget.try_reserve(16).unwrap();
    let mut requests = allocations(&mut work, &[8, 16]);
    requests[1].0.end -= 1;
    let error = file.read_many_into(requests, 0).await.unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("scheduled range"));
    assert!(reader.calls.lock().unwrap().is_empty());
    assert_eq!(scheduler.stats().iops, 0);
    drop(work);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[tokio::test]
async fn local_batch_preserves_owned_ranges(#[values(false, true)] lite: bool) {
    let temp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(temp.path(), (0..32u8).collect::<Vec<_>>()).unwrap();
    let reader = crate::local::LocalObjectReader::open_local_path(temp.path(), 4096, Some(32))
        .await
        .unwrap();
    let scheduler = ScanScheduler::new(
        Arc::new(ObjectStore::local()),
        SchedulerConfig {
            io_buffer_size_bytes: 16,
            use_lite_scheduler: Some(lite),
        },
    );
    let file = scheduler.open_reader(Arc::from(reader));
    let budget = AllocationBudget::new(16);
    let mut work = budget.try_reserve(16).unwrap();
    let requests = allocations(&mut work, &[24, 0]);
    let pending = file.read_many_into(requests, 0);
    drop(file);
    let output = pending.await.unwrap();
    assert_eq!(output[0].as_ref(), &(24..32u8).collect::<Vec<_>>());
    assert_eq!(output[1].as_ref(), &(0..8u8).collect::<Vec<_>>());
    drop((output, work));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(scheduler.reserved_bytes(), 0);
}
