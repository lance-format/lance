// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;
use crate::{object_reader::CloudObjectReader, scheduler::tests::wait_until};
use lance_core::{
    cache::{Context, DeepSizeOf},
    utils::allocation::AllocationBudget,
};
use rstest::rstest;
use std::{
    future::Future,
    sync::atomic::AtomicUsize,
    task::{Context as TaskContext, Poll, Wake, Waker},
};
use tokio::sync::Semaphore;

mod many;

#[derive(Clone, Copy, Debug)]
enum ReadMode {
    Submitted,
    Polled,
}

impl FileScheduler {
    fn read_for_test(
        &self,
        mode: ReadMode,
        range: Range<u64>,
        priority: u64,
        output: ReservedBuffer,
    ) -> BoxFuture<'static, Result<ReservedBuffer>> {
        match mode {
            ReadMode::Submitted => self.submit_single_into(range, priority, output),
            ReadMode::Polled => self.read_single_into(range, priority, output),
        }
    }
}

#[derive(Debug)]
struct ControlledReader {
    path: Path,
    calls: Arc<Mutex<Vec<usize>>>,
    gate: Arc<Semaphore>,
    batches: std::sync::atomic::AtomicUsize,
}

impl DeepSizeOf for ControlledReader {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        0
    }
}

impl Reader for ControlledReader {
    fn get_ranges_into(
        &self,
        requests: Vec<(Range<usize>, ReservedBuffer)>,
    ) -> BoxFuture<'_, Result<Vec<ReservedBuffer>>> {
        self.batches.fetch_add(1, Ordering::SeqCst);
        async move {
            let mut outputs = Vec::new();
            for (range, output) in requests {
                outputs.push(self.get_range_into(range, output).await?);
            }
            Ok(outputs)
        }
        .boxed()
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn block_size(&self) -> usize {
        4096
    }

    fn io_parallelism(&self) -> usize {
        4
    }

    fn size(&self) -> BoxFuture<'_, object_store::Result<usize>> {
        async { Ok(256) }.boxed()
    }

    fn get_range(&self, range: Range<usize>) -> BoxFuture<'static, object_store::Result<Bytes>> {
        let calls = self.calls.clone();
        async move {
            calls.lock().unwrap().push(range.start);
            Ok(Bytes::from(range.map(|n| n as u8).collect::<Vec<_>>()))
        }
        .boxed()
    }

    fn get_all(&self) -> BoxFuture<'_, object_store::Result<Bytes>> {
        self.get_range(0..256)
    }

    fn get_range_into(
        &self,
        range: Range<usize>,
        mut output: ReservedBuffer,
    ) -> BoxFuture<'_, Result<ReservedBuffer>> {
        async move {
            self.calls.lock().unwrap().push(range.start);
            if range.start == 0 {
                self.gate.acquire().await.unwrap().forget();
            }
            if range.start == 64 {
                let gate = self.gate.clone();
                return Ok(tokio::spawn(async move {
                    gate.acquire().await.unwrap().forget();
                    output
                })
                .await
                .unwrap());
            }
            if range.start == 96 {
                let replacement = AllocationBudget::new(output.capacity())
                    .try_reserve(output.capacity())
                    .unwrap()
                    .allocate()
                    .unwrap();
                drop(output);
                return Ok(replacement);
            }
            if range.start == 128 {
                return Err(Error::io("controlled read failure"));
            }
            for (dest, value) in output.as_mut().iter_mut().zip(range) {
                *dest = value as u8;
            }
            Ok(output)
        }
        .boxed()
    }
}

fn setup(lite: bool, io_bytes: u64) -> (Arc<ScanScheduler>, FileScheduler, Arc<ControlledReader>) {
    let scheduler = ScanScheduler::new(
        Arc::new(ObjectStore::memory()),
        SchedulerConfig {
            io_buffer_size_bytes: io_bytes,
            use_lite_scheduler: Some(lite),
        },
    );
    let reader = Arc::new(ControlledReader {
        path: Path::from("owned"),
        calls: Arc::new(Mutex::new(Vec::new())),
        gate: Arc::new(Semaphore::new(0)),
        batches: std::sync::atomic::AtomicUsize::new(0),
    });
    let file = scheduler.open_reader(reader.clone());
    (scheduler, file, reader)
}

#[rstest]
#[case::standard(false)]
#[case::lite(true)]
#[tokio::test]
async fn queued_owned_reads_preserve_priority_and_separate_delivery_credit(
    #[case] lite: bool,
    #[values(ReadMode::Submitted, ReadMode::Polled)] mode: ReadMode,
) {
    let (scheduler, file, reader) = setup(lite, 8);
    let extra = IoStats::new();
    let file = file.with_io_stats(extra.recorder());
    let budget = AllocationBudget::new(24);
    let mut work = budget.try_reserve(24).unwrap();
    let first =
        tokio::spawn(file.read_for_test(mode, 0..8, 0, work.try_allocate(8).unwrap().unwrap()));
    wait_until("first read blocked", || {
        reader.calls.lock().unwrap().len() == 1
    })
    .await;
    let mut ordinary =
        scheduler.submit_request(reader.clone(), vec![Range { start: 8, end: 16 }], 5, false);
    assert!(futures::poll!(&mut ordinary).is_pending());
    let ordinary = tokio::spawn(ordinary);
    let mut last = file.read_for_test(mode, 24..32, 20, work.try_allocate(8).unwrap().unwrap());
    let mut second = file.read_for_test(mode, 16..24, 10, work.try_allocate(8).unwrap().unwrap());
    assert!(futures::poll!(&mut last).is_pending());
    assert!(futures::poll!(&mut second).is_pending());
    let last = tokio::spawn(last);
    let second = tokio::spawn(second);
    assert_eq!(*reader.calls.lock().unwrap(), [0]);
    assert_eq!(scheduler.reserved_bytes(), 8);
    assert_eq!(budget.reserved(), 24);
    reader.gate.add_permits(1);
    let first = first.await.unwrap().unwrap();
    let second = second.await.unwrap().unwrap();
    let last = last.await.unwrap().unwrap();
    assert_eq!(
        &ordinary.await.unwrap().unwrap()[0][..],
        &[8, 9, 10, 11, 12, 13, 14, 15]
    );
    assert_eq!(*reader.calls.lock().unwrap(), [0, 8, 16, 24]);
    assert_eq!(first.as_ref(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(second.as_ref(), &[16, 17, 18, 19, 20, 21, 22, 23]);
    assert_eq!(last.as_ref(), &[24, 25, 26, 27, 28, 29, 30, 31]);
    assert_eq!(scheduler.reserved_bytes(), 0);
    assert_eq!(budget.reserved(), 24);
    assert!(budget.try_reserve(1).is_none());
    // The direct ordinary scheduler request above doesn't use FileScheduler's
    // per-file request recorder. Only the three admitted requests do.
    assert_eq!(extra.snapshot().iops, 3);
    assert_eq!(extra.snapshot().bytes_read, 24);
    first.recycle(&mut work).unwrap();
    second.recycle(&mut work).unwrap();
    let shared = last.freeze();
    let final_alias = shared.slice(2..3);
    drop(shared);
    drop(work);
    assert_eq!(budget.reserved(), 8);
    assert_eq!(&final_alias[..], &[26]);
    drop(final_alias);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::standard(false)]
#[case::lite(true)]
#[tokio::test]
async fn queued_and_running_cancellation_refund_without_blocking_a_peer(
    #[case] lite: bool,
    #[values(ReadMode::Submitted, ReadMode::Polled)] mode: ReadMode,
) {
    let (scheduler, file, reader) = setup(lite, 8);
    let budget = AllocationBudget::new(24);
    let first = tokio::spawn(file.read_for_test(
        mode,
        0..8,
        0,
        budget.try_reserve(8).unwrap().allocate().unwrap(),
    ));
    wait_until("first read blocked", || {
        reader.calls.lock().unwrap().len() == 1
    })
    .await;
    let mut queued = file.read_for_test(
        mode,
        16..24,
        10,
        budget.try_reserve(8).unwrap().allocate().unwrap(),
    );
    assert!(futures::poll!(&mut queued).is_pending());
    assert_eq!(budget.reserved(), 16);
    drop(queued);
    assert_eq!(budget.reserved(), 8);
    // A distinct peer uses the explicit existing bypass policy, while the slow
    // read keeps its own payload charged. Cancellation must not touch the peer.
    let bypass = file.with_bypass_backpressure();
    for _ in 0..3 {
        let output = bypass
            .read_for_test(
                mode,
                32..40,
                30,
                budget.try_reserve(8).unwrap().allocate().unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(output.as_ref(), &[32, 33, 34, 35, 36, 37, 38, 39]);
        assert_eq!(scheduler.reserved_bytes(), 8);
        assert_eq!(budget.reserved(), 16);
        drop(output);
    }
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    wait_until("cancelled owned read released", || {
        budget.reserved() == 0 && scheduler.reserved_bytes() == 0
    })
    .await;
    assert!(!reader.calls.lock().unwrap().contains(&16));
    let output = file
        .read_for_test(
            mode,
            40..48,
            40,
            budget.try_reserve(8).unwrap().allocate().unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.as_ref(), &[40, 41, 42, 43, 44, 45, 46, 47]);
    drop(output);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::standard(false)]
#[case::lite(true)]
#[tokio::test]
async fn invalid_empty_and_failed_owned_reads_leave_no_credit(
    #[case] lite: bool,
    #[values(ReadMode::Submitted, ReadMode::Polled)] mode: ReadMode,
) {
    let (scheduler, file, reader) = setup(lite, 8);
    let budget = AllocationBudget::new(8);
    let reversed = Range { start: 8, end: 7 };
    for range in [0..7, reversed] {
        let error = file
            .read_for_test(
                mode,
                range,
                0,
                budget.try_reserve(8).unwrap().allocate().unwrap(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("scheduled range"));
        assert_eq!(budget.reserved(), 0);
    }
    let empty = file
        .read_for_test(
            mode,
            7..7,
            0,
            budget.try_reserve(0).unwrap().allocate().unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(empty.capacity(), 0);
    assert!(reader.calls.lock().unwrap().is_empty());
    assert_eq!(scheduler.stats().iops, 0);
    let error = file
        .read_for_test(
            mode,
            128..136,
            0,
            budget.try_reserve(8).unwrap().allocate().unwrap(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::IO { .. }));
    assert!(error.to_string().contains("controlled read failure"));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(scheduler.reserved_bytes(), 0);
    let error = file
        .read_for_test(
            mode,
            96..104,
            0,
            budget.try_reserve(8).unwrap().allocate().unwrap(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidInput { .. }));
    assert!(error.to_string().contains("supplied allocation unchanged"));
    assert_eq!(budget.reserved(), 0);
    assert_eq!(scheduler.reserved_bytes(), 0);
    let result = file.submit_single(16..24, 0).await.unwrap();
    assert_eq!(&result[..], &[16, 17, 18, 19, 20, 21, 22, 23]);
}

#[rstest]
#[case::standard(false)]
#[case::lite(true)]
#[tokio::test]
async fn scheduled_local_read_preserves_allocation_and_survives_handle_drop(
    #[case] lite: bool,
    #[values(ReadMode::Submitted, ReadMode::Polled)] mode: ReadMode,
) {
    let temp = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(temp.path(), b"abcdefgh").unwrap();
    let reader = crate::local::LocalObjectReader::open_local_path(temp.path(), 4096, Some(8))
        .await
        .unwrap();
    let scheduler = ScanScheduler::new(
        Arc::new(ObjectStore::local()),
        SchedulerConfig {
            io_buffer_size_bytes: 4,
            use_lite_scheduler: Some(lite),
        },
    );
    let file = scheduler.open_reader(Arc::from(reader));
    let budget = AllocationBudget::new(4);
    let output = budget.try_reserve(4).unwrap().allocate().unwrap();
    let pointer = output.as_ref().as_ptr() as usize;
    let pending = file.read_for_test(mode, 2..6, 0, output);
    drop(file);
    drop(scheduler);
    let output = pending.await.unwrap();
    assert_eq!(output.as_ref(), b"cdef");
    assert_eq!(output.as_ref().as_ptr() as usize, pointer);
    assert_eq!(budget.reserved(), 4);
    drop(output);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::small_reader(false, "Invalid range 6..10 for object 'body' of size 8 bytes")]
#[case::cloud_stream(true, "body for body range 6..10 has 2 bytes, expected 4")]
#[tokio::test]
async fn scheduled_reader_rejects_incomplete_ranges(
    #[values(false, true)] lite: bool,
    #[values(ReadMode::Submitted, ReadMode::Polled)] mode: ReadMode,
    #[case] cloud_stream: bool,
    #[case] expected_message: &str,
) {
    let store = Arc::new(ObjectStore::memory());
    let path = Path::from("body");
    store.put(&path, b"abcdefgh").await.unwrap();
    let scheduler = ScanScheduler::new(
        store.clone(),
        SchedulerConfig {
            io_buffer_size_bytes: 8,
            use_lite_scheduler: Some(lite),
        },
    );
    // open_file chooses SmallReader for this tiny object. Use the actual cloud
    // reader explicitly to cover streamed-body validation as well as that path.
    let file = if cloud_stream {
        scheduler.open_reader(Arc::new(
            CloudObjectReader::new(store.inner.clone(), path.clone(), 4096, Some(8), 0).unwrap(),
        ))
    } else {
        scheduler
            .open_file(&path, &CachedFileSize::unknown())
            .await
            .unwrap()
    };
    let budget = AllocationBudget::new(4);
    let output = file
        .read_for_test(
            mode,
            1..5,
            0,
            budget.try_reserve(4).unwrap().allocate().unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(output.as_ref(), b"bcde");
    drop(output);
    let error = file
        .read_for_test(
            mode,
            6..10,
            0,
            budget.try_reserve(4).unwrap().allocate().unwrap(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, Error::IO { .. }));
    assert!(error.to_string().contains(expected_message), "{error}");
    assert_eq!(budget.reserved(), 0);
    assert_eq!(scheduler.reserved_bytes(), 0);
}

#[rstest]
#[case::standard(false)]
#[case::lite(true)]
#[tokio::test]
async fn cancellation_keeps_charge_until_detached_backend_stops(
    #[case] lite: bool,
    #[values(ReadMode::Submitted, ReadMode::Polled)] mode: ReadMode,
) {
    let (scheduler, file, reader) = setup(lite, 8);
    let budget = AllocationBudget::new(8);
    let request = tokio::spawn(file.read_for_test(
        mode,
        64..72,
        0,
        budget.try_reserve(8).unwrap().allocate().unwrap(),
    ));
    wait_until("backend has taken buffer", || {
        !reader.calls.lock().unwrap().is_empty()
    })
    .await;
    request.abort();
    assert!(request.await.unwrap_err().is_cancelled());
    wait_until("scheduler credit released", || {
        scheduler.reserved_bytes() == 0
    })
    .await;
    assert_eq!(budget.reserved(), 8);
    assert!(budget.try_reserve(1).is_none());
    reader.gate.add_permits(1);
    wait_until("detached backend released allocation", || {
        budget.reserved() == 0
    })
    .await;
}

#[rstest]
#[case::standard(false)]
#[case::lite(true)]
#[tokio::test]
async fn finite_peers_progress_with_slow_read_and_fixed_shared_budget(
    #[case] lite: bool,
    #[values(ReadMode::Submitted, ReadMode::Polled)] mode: ReadMode,
) {
    let (scheduler, file, reader) = setup(lite, 16);
    let budget = AllocationBudget::new(16);
    let slow = tokio::spawn(file.read_for_test(
        mode,
        0..8,
        0,
        budget.try_reserve(8).unwrap().allocate().unwrap(),
    ));
    wait_until("slow read started", || {
        !reader.calls.lock().unwrap().is_empty()
    })
    .await;
    for _ in 0..3 {
        let output = file
            .read_for_test(
                mode,
                16..24,
                1,
                budget.try_reserve(8).unwrap().allocate().unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(output.as_ref(), &[16, 17, 18, 19, 20, 21, 22, 23]);
        assert_eq!(budget.reserved(), 16);
        assert_eq!(scheduler.reserved_bytes(), 8);
        drop(output);
    }
    reader.gate.add_permits(1);
    drop(slow.await.unwrap().unwrap());
    assert_eq!(budget.reserved(), 0);
    assert_eq!(scheduler.reserved_bytes(), 0);
}

#[rstest]
#[case::standard(false)]
#[case::lite(true)]
#[tokio::test]
async fn no_io_backpressure_still_retains_payload_charge(
    #[case] lite: bool,
    #[values(ReadMode::Submitted, ReadMode::Polled)] mode: ReadMode,
) {
    let (scheduler, file, _) = setup(lite, 0);
    let budget = AllocationBudget::new(8);
    let output = file
        .read_for_test(
            mode,
            8..16,
            0,
            budget.try_reserve(8).unwrap().allocate().unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(scheduler.reserved_bytes(), 0);
    assert_eq!(budget.reserved(), 8);
    drop(output);
    assert_eq!(budget.reserved(), 0);
}

#[rstest]
#[case::standard(false)]
#[case::lite(true)]
#[tokio::test]
async fn unpolled_drop_refunds_owned_read(#[case] lite: bool) {
    let (scheduler, file, reader) = setup(lite, 8);
    let budget = AllocationBudget::new(8);
    let request =
        file.submit_single_into(8..16, 0, budget.try_reserve(8).unwrap().allocate().unwrap());
    if !lite {
        wait_until("standard read completed without polling", || {
            let IoQueueType::Standard(queue) = &scheduler.io_queue else {
                unreachable!()
            };
            let state = queue.state.lock().unwrap();
            state.iops_avail == state.io_capacity && !reader.calls.lock().unwrap().is_empty()
        })
        .await;
    } else {
        assert!(reader.calls.lock().unwrap().is_empty());
    }
    assert_eq!(budget.reserved(), 8);
    drop(request);
    wait_until("unpolled response refunded", || {
        budget.reserved() == 0 && scheduler.reserved_bytes() == 0
    })
    .await;
}

#[tokio::test]
async fn uncontended_polled_read_completes_without_scheduler_dispatch() {
    let (scheduler, file, reader) = setup(false, 8);
    let stats = IoStats::new();
    let file = file.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(8);
    for _ in 0..3 {
        let output = file
            .read_single_into(8..16, 0, budget.try_reserve(8).unwrap().allocate().unwrap())
            .now_or_never()
            .expect("an immediately ready backend needs no scheduler task")
            .unwrap();
        assert_eq!(output.as_ref(), &[8, 9, 10, 11, 12, 13, 14, 15]);
        assert_eq!(scheduler.reserved_bytes(), 0);
        assert_eq!(budget.reserved(), 8);
        drop(output);
    }
    assert_eq!(*reader.calls.lock().unwrap(), [8, 8, 8]);
    assert_eq!(stats.snapshot().iops, 3);
    assert_eq!(stats.snapshot().bytes_read, 24);
    let IoQueueType::Standard(queue) = &scheduler.io_queue else {
        unreachable!()
    };
    let state = queue.state.lock().unwrap();
    assert_eq!(state.iops_avail, state.io_capacity);
    assert!(state.pending_requests.is_empty());
    assert!(!state.priorities_in_flight.contains(0));
}

#[rstest]
#[tokio::test]
async fn unpolled_lazy_read_does_not_take_credit_or_start_io(#[values(false, true)] lite: bool) {
    let (scheduler, file, reader) = setup(lite, 8);
    let stats = IoStats::new();
    let file = file.with_io_stats(stats.recorder());
    let budget = AllocationBudget::new(16);
    let pending =
        file.read_single_into(0..8, 0, budget.try_reserve(8).unwrap().allocate().unwrap());
    tokio::task::yield_now().await;
    assert!(reader.calls.lock().unwrap().is_empty());
    assert_eq!(scheduler.reserved_bytes(), 0);
    assert_eq!(stats.snapshot().iops, 0);
    assert_eq!(budget.reserved(), 8);
    let peer = file
        .read_single_into(8..16, 1, budget.try_reserve(8).unwrap().allocate().unwrap())
        .await
        .unwrap();
    drop(peer);
    drop(pending);
    assert_eq!(*reader.calls.lock().unwrap(), [8]);
    assert_eq!(budget.reserved(), 0);
    assert_eq!(scheduler.reserved_bytes(), 0);
    assert_eq!(stats.snapshot().iops, 1);
}

#[tokio::test]
async fn direct_read_respects_iop_limit_and_cancel_wakes_same_priority_peer() {
    let (scheduler, file, reader) = setup(false, 16);
    let IoQueueType::Standard(queue) = &scheduler.io_queue else {
        unreachable!()
    };
    {
        let mut state = queue.state.lock().unwrap();
        state.io_capacity = 1;
        state.iops_avail = 1;
    }
    let budget = AllocationBudget::new(16);
    let mut slow =
        file.read_single_into(0..8, 0, budget.try_reserve(8).unwrap().allocate().unwrap());
    assert!(futures::poll!(&mut slow).is_pending());
    let mut peer =
        file.read_single_into(8..16, 0, budget.try_reserve(8).unwrap().allocate().unwrap());
    assert!(futures::poll!(&mut peer).is_pending());
    assert_eq!(*reader.calls.lock().unwrap(), [0]);
    assert_eq!(queue.state.lock().unwrap().pending_requests.len(), 1);
    assert_eq!(scheduler.reserved_bytes(), 8);
    drop(slow);
    let output = peer.await.unwrap();
    assert_eq!(output.as_ref(), &[8, 9, 10, 11, 12, 13, 14, 15]);
    assert_eq!(scheduler.reserved_bytes(), 0);
    assert_eq!(budget.reserved(), 8);
    drop(output);
    assert_eq!(budget.reserved(), 0);
    let state = queue.state.lock().unwrap();
    assert_eq!(state.iops_avail, 1);
    assert!(!state.priorities_in_flight.contains(0));
}

#[derive(Default)]
struct WakeCount(AtomicUsize);

impl Wake for WakeCount {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[rstest]
fn direct_credit_refund_leaves_empty_queue_asleep(#[values(1, 2)] count: u32) {
    // Poll a standalone queue: an actual scheduler must remain its sole consumer.
    let queue = Arc::new(IoQueue::new(count, u64::from(count) * 8, IoStats::new()));
    let wakes = Arc::new(WakeCount::default());
    let waker = Waker::from(wakes.clone());
    let mut cx = TaskContext::from_waker(&waker);
    let mut pending = Box::pin(queue.pop());
    assert!(pending.as_mut().poll(&mut cx).is_pending());
    let credits: Vec<_> = (0..count)
        .map(|_| DirectReadCredit::try_new(&queue, 5, 8, false).unwrap())
        .collect();
    assert_eq!(queue.reserved_bytes(), u64::from(count) * 8);
    drop(credits);
    assert_eq!(wakes.0.load(Ordering::SeqCst), 0);
    assert_eq!(queue.reserved_bytes(), 0);
    {
        let state = queue.state.lock().unwrap();
        assert_eq!(state.iops_avail, count);
        assert!(state.priorities_in_flight.is_empty());
    }
    queue.close();
    assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
    assert!(matches!(pending.as_mut().poll(&mut cx), Poll::Ready(None)));
}

#[rstest]
fn queued_read_wakes_before_or_after_direct_credit_refund(
    #[values(false, true)] is_queued_before_refund: bool,
) {
    let queue = Arc::new(IoQueue::new(1, 8, IoStats::new()));
    let credit = DirectReadCredit::try_new(&queue, 5, 8, false).unwrap();
    let wakes = Arc::new(WakeCount::default());
    let waker = Waker::from(wakes.clone());
    let mut cx = TaskContext::from_waker(&waker);
    let mut pending = Box::pin(queue.pop());
    assert!(pending.as_mut().poll(&mut cx).is_pending());
    let task = IoTask {
        reader: Arc::new(ControlledReader {
            path: Path::from("owned"),
            calls: Arc::new(Mutex::new(Vec::new())),
            gate: Arc::new(Semaphore::new(0)),
            batches: AtomicUsize::new(0),
        }),
        to_read: 8..16,
        buffer: None,
        when_done: Box::new(|_, _| {}),
        priority: 7,
        bypass_backpressure: false,
        request_cancellation: None,
    };
    if is_queued_before_refund {
        queue.push(task);
        assert_eq!(wakes.0.swap(0, Ordering::SeqCst), 1);
        assert!(pending.as_mut().poll(&mut cx).is_pending());
        drop(credit);
    } else {
        drop(credit);
        assert_eq!(wakes.0.load(Ordering::SeqCst), 0);
        queue.push(task);
    }
    assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
    let Poll::Ready(Some(task)) = pending.as_mut().poll(&mut cx) else {
        panic!("refunded credit must make the queued read runnable");
    };
    assert_eq!(task.to_read, 8..16);
    assert_eq!(task.priority, 7);
    assert_eq!(queue.reserved_bytes(), 8);
    let state = queue.state.lock().unwrap();
    assert_eq!(state.iops_avail, 0);
    assert!(state.pending_requests.is_empty());
    assert!(!state.priorities_in_flight.contains(5));
    assert!(state.priorities_in_flight.contains(7));
}

#[test]
fn close_wakes_queue_with_live_direct_credit() {
    let queue = Arc::new(IoQueue::new(1, 8, IoStats::new()));
    let credit = DirectReadCredit::try_new(&queue, 5, 8, false).unwrap();
    let wakes = Arc::new(WakeCount::default());
    let waker = Waker::from(wakes.clone());
    let mut cx = TaskContext::from_waker(&waker);
    let mut pending = Box::pin(queue.pop());
    assert!(pending.as_mut().poll(&mut cx).is_pending());
    queue.close();
    assert_eq!(wakes.0.load(Ordering::SeqCst), 1);
    assert!(matches!(pending.as_mut().poll(&mut cx), Poll::Ready(None)));
    assert!(DirectReadCredit::try_new(&queue, 7, 8, false).is_none());
    drop(credit);
    assert_eq!(queue.reserved_bytes(), 0);
    let state = queue.state.lock().unwrap();
    assert_eq!(state.iops_avail, 1);
    assert!(state.priorities_in_flight.is_empty());
}
