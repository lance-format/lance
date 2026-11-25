// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! How a [`RequestBatch`] sends the reads that 2.0 pages queue once their own
//! reads arrive (late reads), without waiting for the decoder to await them.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex, Weak};

use bytes::Bytes;
use futures::future::BoxFuture;
use lance_core::Result;

use super::{BatchShared, BatchState, IO_REQUEST_BATCH_BYTES, PendingRead, RequestBatch};
use crate::EncodingsIo;

/// Bounds the late reads that a 2.0 run, with its nested list batches, has sent
/// ahead of the decoder and the decoder has not consumed yet. The bound is one
/// request batch ([`IO_REQUEST_BATCH_BYTES`]), so sending late reads early holds
/// about one batch more than waiting for the decoder to reach each page.
///
/// A scan holds one window per file it has open ahead of the decoder. A read
/// sent ahead also holds its share of the scan's I/O buffer until the decoder
/// reaches it, so each window stays within an eighth of that buffer. When the
/// windows fill the buffer, the I/O scheduler still admits the read with the
/// lowest priority, the one the decoder waits for. An eighth is a heuristic:
/// with small buffers, a quarter, an eighth and a sixteenth measured within a
/// few percent of each other.
pub(super) struct LateWindow {
    state: Mutex<WindowState>,
    capacity: u64,
}

#[derive(Default)]
struct WindowState {
    used: u64,
    /// Batches that hold late reads until the window has room.
    waiting: VecDeque<Weak<BatchShared>>,
}

impl LateWindow {
    fn new(io_buffer_size: Option<u64>) -> Self {
        let capacity = io_buffer_size
            // lance-io applies no backpressure at a size of 0, so there is no
            // buffer to stay within.
            .filter(|&size| size > 0)
            .map_or(IO_REQUEST_BATCH_BYTES, |size| {
                (size / 8).min(IO_REQUEST_BATCH_BYTES)
            });
        Self {
            state: Mutex::default(),
            capacity,
        }
    }

    /// Whether `bytes` more fit, reserving them if `should_reserve`. If they do
    /// not, `batch` waits for room. An empty full-size window admits a group of any
    /// size; a window cut down to the I/O buffer admits only groups that fit.
    fn admit(&self, bytes: u64, should_reserve: bool, batch: &Arc<BatchShared>) -> bool {
        let mut state = self.state.lock().unwrap();
        if (state.used == 0 && self.is_full_size()) || state.used + bytes <= self.capacity {
            if should_reserve {
                state.used += bytes;
            }
            return true;
        }
        if !state
            .waiting
            .iter()
            .any(|waiting| waiting.as_ptr() == Arc::as_ptr(batch))
        {
            state.waiting.push_back(Arc::downgrade(batch));
        }
        false
    }

    /// Whether the window has the full size of one request batch.
    fn is_full_size(&self) -> bool {
        self.capacity == IO_REQUEST_BATCH_BYTES
    }

    fn release(&self, bytes: u64) {
        let waiting = {
            let mut state = self.state.lock().unwrap();
            state.used -= bytes;
            if state.used >= self.capacity {
                return;
            }
            std::mem::take(&mut state.waiting)
        };
        for batch in waiting.iter().filter_map(Weak::upgrade) {
            batch.flush_late();
        }
    }
}

/// A read's share of its run's [`LateWindow`], returned when the read's future
/// finishes or is dropped.
pub(super) struct Charge {
    window: Arc<LateWindow>,
    bytes: u64,
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.window.release(self.bytes);
    }
}

/// Held by a task that will queue a read after its own reads arrive, until it
/// has queued that read or given up. It is also the batch's I/O, so a 2.0 page
/// scheduler that keeps the I/O it was given to queue a late read (a string
/// page's bytes) holds the guard until then.
#[derive(Debug)]
pub struct LateReadGuard {
    pub(super) batch: Arc<RequestBatch>,
    group: u64,
}

impl Drop for LateReadGuard {
    fn drop(&mut self) {
        self.batch.shared.producer_done(self.group);
    }
}

impl EncodingsIo for LateReadGuard {
    fn submit_request(
        &self,
        ranges: Vec<Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
        self.batch.submit_request(ranges, priority)
    }
}

impl BatchShared {
    fn window(&self) -> Arc<LateWindow> {
        match &self.outer {
            Some(outer) => outer.window(),
            None => self
                .window
                .get_or_init(|| Arc::new(LateWindow::new(self.inner.io_buffer_size())))
                .clone(),
        }
    }

    /// Where this batch's scheduling steps send their reads: `inner`, or for a
    /// nested batch, the outer batch, as nested step reads.
    pub(super) fn step_io(&self) -> Arc<dyn EncodingsIo> {
        match &self.outer {
            Some(outer) => Arc::new(NestedSteps(RequestBatch {
                shared: outer.clone(),
            })),
            None => self.inner.clone(),
        }
    }

    fn producer_done(self: &Arc<Self>, group: u64) {
        {
            let mut state = self.state.lock().unwrap();
            let Some(idx) = state
                .open_producers
                .iter()
                .position(|(open_group, _)| *open_group == group)
            else {
                debug_assert!(false, "late read group {group} has no open guard");
                return;
            };
            state.open_producers[idx].1 -= 1;
            if state.open_producers[idx].1 > 0 {
                return;
            }
            state.open_producers.swap_remove(idx);
            if !state.is_scheduling_finished || state.reads.is_empty() {
                return;
            }
            if state.is_late_flush_queued {
                return;
            }
            state.is_late_flush_queued = true;
        }
        // Yield first so groups that close in the same pass share a request. One
        // request per group (one per page for a string column) slows the I/O
        // scheduler's dispatch.
        match tokio::runtime::Handle::try_current() {
            Ok(runtime) => {
                let batch = self.clone();
                runtime.spawn(async move {
                    tokio::task::yield_now().await;
                    batch.flush_late();
                });
            }
            // Off a Tokio runtime there is nothing to spawn the yield on, so
            // the queued reads are sent at once.
            Err(_) => self.flush_late(),
        }
    }

    /// Sends the queued reads on if they fit in the run's window.
    fn flush_late(self: &Arc<Self>) {
        let reads = {
            let mut state = self.state.lock().unwrap();
            state.is_late_flush_queued = false;
            self.take_admitted(&mut state)
        };
        if reads.is_empty() {
            return;
        }
        RequestBatch::submit_batch(&self.inner, reads);
        // A nested batch queues its late reads in the outer batch. Until the outer
        // run finishes, its step flushes send them; after that, `outer.flush_late`
        // sends them through the window.
        if let Some(outer) = &self.outer {
            let is_outer_finished = outer.state.lock().unwrap().is_scheduling_finished;
            if is_outer_finished {
                outer.flush_late();
            }
        }
    }

    /// Takes the queued reads if all of them fit in the run's window. A nested
    /// batch only checks for room: the top-level batch charges the reads to
    /// the window when it sends them to the I/O.
    fn take_admitted(self: &Arc<Self>, state: &mut BatchState) -> Vec<PendingRead> {
        let window = self.window();
        let should_reserve = self.outer.is_none();
        state.reads.retain(|read| !read.tx.is_canceled());
        let bytes = RequestBatch::queued_bytes(&state.reads);
        if state.reads.is_empty() || !window.admit(bytes, should_reserve, self) {
            return Vec::new();
        }
        let mut reads = std::mem::take(&mut state.reads);
        if should_reserve {
            for read in &mut reads {
                read.charge = Some(Charge {
                    window: window.clone(),
                    bytes: RequestBatch::queued_bytes(std::slice::from_ref(read)),
                });
            }
        }
        reads
    }

    /// Submits the queued reads when one of them is awaited. After the run has
    /// finished, awaiting a nested step read sends the nested step reads, and the
    /// late reads of a top-level batch only if all of them fit in its window. A
    /// window cut down to the I/O buffer holds back most late reads, so then every
    /// queued read goes.
    pub(super) fn submit_queued(self: &Arc<Self>, is_nested_step: bool) {
        let mut state = self.state.lock().unwrap();
        if !is_nested_step {
            let reads = std::mem::take(&mut state.reads);
            drop(state);
            RequestBatch::submit_batch(&self.inner, reads);
            return;
        }
        let reads = if state.is_scheduling_finished && self.window().is_full_size() {
            let (mut steps, late): (Vec<_>, Vec<_>) = std::mem::take(&mut state.reads)
                .into_iter()
                .partition(|read| read.is_nested_step);
            state.reads = late;
            if self.outer.is_none() {
                steps.extend(self.take_admitted(&mut state));
            }
            steps
        } else {
            std::mem::take(&mut state.reads)
        };
        drop(state);
        RequestBatch::submit_batch(&self.step_io(), reads);
    }
}

impl RequestBatch {
    pub(crate) fn late_read_guard(self: &Arc<Self>) -> LateReadGuard {
        let mut state = self.shared.state.lock().unwrap();
        let group = state.next_group;
        if state.is_scheduling_finished {
            // No step flush comes to close the group, so the guard is a group
            // of its own.
            state.next_group += 1;
        }
        match state
            .open_producers
            .iter_mut()
            .find(|(open_group, _)| *open_group == group)
        {
            Some((_, num_open)) => *num_open += 1,
            None => state.open_producers.push((group, 1)),
        }
        LateReadGuard {
            batch: self.clone(),
            group,
        }
    }
}

/// Queues the reads of a nested batch's scheduling steps in its outer batch.
#[derive(Debug)]
struct NestedSteps(RequestBatch);

impl EncodingsIo for NestedSteps {
    fn submit_request(
        &self,
        ranges: Vec<Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
        self.0.queue(ranges, priority, true)
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::DataType;
    use futures::{FutureExt, StreamExt};
    use lance_core::cache::LanceCache;
    use lance_core::datatypes::Schema;
    use lance_datagen::{RowCount, array, gen_batch};
    use rstest::rstest;
    use tokio::sync::mpsc::{self, unbounded_channel};

    use super::*;
    use crate::BufferScheduler;
    use crate::decoder::{
        DecodeBatchScheduler, DecoderConfig, DecoderMessage, DecoderPlugins, FilterExpression,
        create_decode_stream,
    };
    use crate::encoder::{EncodedBatch, EncodingOptions, encode_batch};
    use crate::testing::{TestEncoding, test_encoding_strategy};

    /// Records the ranges of every request. Answers from `data` if it is set,
    /// and otherwise with a zero buffer per range that starts with the range's
    /// offset, so a test can check that each reader got its own ranges back.
    #[derive(Debug, Default)]
    struct RecordingIo {
        data: Option<BufferScheduler>,
        requests: Mutex<Vec<Vec<Range<u64>>>>,
        io_buffer_size: Option<u64>,
    }

    impl EncodingsIo for RecordingIo {
        fn submit_request(
            &self,
            ranges: Vec<Range<u64>>,
            priority: u64,
        ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
            self.requests.lock().unwrap().push(ranges.clone());
            if let Some(data) = &self.data {
                return data.submit_request(ranges, priority);
            }
            let buffers = ranges
                .iter()
                .map(|range| {
                    let mut buffer = vec![0u8; (range.end - range.start) as usize];
                    let marker = range.start.to_le_bytes();
                    let marker_len = marker.len().min(buffer.len());
                    buffer[..marker_len].copy_from_slice(&marker[..marker_len]);
                    Bytes::from(buffer)
                })
                .collect();
            std::future::ready(Ok(buffers)).boxed()
        }

        fn io_buffer_size(&self) -> Option<u64> {
            self.io_buffer_size
        }
    }

    /// Lets a late flush that waits for a yield run.
    async fn settle() {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn request_batch_sends_a_late_group_when_its_last_guard_drops() {
        let io = Arc::new(RecordingIo::default());
        let requests = || io.requests.lock().unwrap().clone();
        let batch = Arc::new(RequestBatch::new(io.clone()));
        let early_guard = batch.late_read_guard();
        let early = batch.submit_request(vec![100..108], 1);
        drop(early_guard);
        settle().await;
        assert!(
            requests().is_empty(),
            "before the run finishes, the step flushes send the reads"
        );
        let first_guard = batch.late_read_guard();
        let second_guard = batch.late_read_guard();
        batch.finish_scheduling();
        assert_eq!(requests(), [vec![100..108]]);

        // A guard taken after the run finishes is a group of its own, so it waits
        // for no other guard.
        let late_guard = batch.late_read_guard();
        let other_late_guard = batch.late_read_guard();
        let late = batch.submit_request(vec![300..308], 2);
        drop(late_guard);
        settle().await;
        assert_eq!(requests(), [vec![100..108], vec![300..308]]);
        drop(other_late_guard);

        // Guards taken in the same step are one group.
        let first = batch.submit_request(vec![500..508], 3);
        drop(first_guard);
        settle().await;
        assert_eq!(requests().len(), 2, "a group waits for its last guard");
        let second = batch.submit_request(vec![700..708], 4);
        drop(second_guard);
        settle().await;
        assert_eq!(
            requests(),
            [vec![100..108], vec![300..308], vec![500..508, 700..708]]
        );

        // Groups that close before the flush's yield share its request. Here the
        // second group closes after the flush's task has started.
        let third_guard = batch.late_read_guard();
        let fourth_guard = batch.late_read_guard();
        let _third = batch.submit_request(vec![900..908], 5);
        drop(third_guard);
        tokio::task::yield_now().await;
        let _fourth = batch.submit_request(vec![1100..1108], 6);
        drop(fourth_guard);
        settle().await;
        assert_eq!(requests()[3..], [vec![900..908, 1100..1108]]);
        for (read, start) in [(early, 100_u64), (late, 300), (first, 500), (second, 700)] {
            assert_eq!(read.await.unwrap()[0].as_ref(), &start.to_le_bytes());
        }
    }

    #[tokio::test]
    async fn request_batch_skips_reads_whose_future_was_dropped() {
        let io = Arc::new(RecordingIo::default());
        let requests = || io.requests.lock().unwrap().clone();
        let batch = Arc::new(RequestBatch::new(io.clone()));
        drop(batch.submit_request(vec![100..108], 1));
        batch.flush();
        assert!(requests().is_empty(), "a dropped read is not sent");

        // A dropped read also takes no room in the window: `first` holds half of
        // it, and the dropped read would fill the other half.
        let half = IO_REQUEST_BATCH_BYTES / 2;
        let first_guard = batch.late_read_guard();
        batch.flush();
        let second_guard = batch.late_read_guard();
        batch.finish_scheduling();
        let first = batch.submit_request(vec![0..half], 2);
        drop(first_guard);
        settle().await;
        drop(batch.submit_request(vec![half..2 * half], 3));
        let second = batch.submit_request(vec![2 * half..2 * half + 8], 4);
        drop(second_guard);
        settle().await;
        assert_eq!(requests(), [vec![0..half], vec![2 * half..2 * half + 8]]);
        assert_eq!(second.await.unwrap()[0].as_ref(), &(2 * half).to_le_bytes());
        drop(first);
    }

    #[tokio::test]
    async fn request_batch_holds_late_groups_beyond_the_window() {
        let size = IO_REQUEST_BATCH_BYTES * 3 / 4;
        // lance-io applies no backpressure at an io_buffer_size of 0, so the
        // window keeps its full size.
        let io = Arc::new(RecordingIo {
            io_buffer_size: Some(0),
            ..Default::default()
        });
        let requests = || io.requests.lock().unwrap().clone();
        let batch = Arc::new(RequestBatch::new(io.clone()));
        let mut guards = Vec::new();
        for _ in 0..4 {
            guards.push(batch.late_read_guard());
            batch.flush();
        }
        let mut guards = guards.into_iter();
        batch.finish_scheduling();

        let first = batch.submit_request(vec![0..size], 1);
        drop(guards.next());
        settle().await;
        let second = batch.submit_request(vec![size..2 * size], 2);
        drop(guards.next());
        settle().await;
        assert_eq!(
            requests().len(),
            1,
            "a group that overflows the window waits"
        );
        first.await.unwrap();
        assert_eq!(requests().len(), 2, "consuming a read makes room");

        let third = batch.submit_request(vec![2 * size..3 * size], 3);
        drop(guards.next());
        settle().await;
        assert_eq!(requests().len(), 2);
        drop(second);
        assert_eq!(requests().len(), 3, "dropping a read also makes room");

        let fourth = batch.submit_request(vec![3 * size..4 * size], 4);
        drop(guards.next());
        settle().await;
        assert_eq!(requests().len(), 3);
        fourth.await.unwrap();
        assert_eq!(
            requests(),
            [
                vec![0..size],
                vec![size..2 * size],
                vec![2 * size..3 * size],
                vec![3 * size..4 * size],
            ],
            "a read held by a full window is sent when it is awaited"
        );
        drop(third);

        let oversized_guard = batch.late_read_guard();
        let _oversized = batch.submit_request(vec![4 * size..6 * size], 5);
        drop(oversized_guard);
        settle().await;
        assert_eq!(
            requests().len(),
            5,
            "a group larger than the window is sent once the window is empty"
        );
    }

    #[tokio::test]
    async fn nested_request_batches_send_late_groups_through_the_run_window() {
        let size = IO_REQUEST_BATCH_BYTES * 3 / 4;
        let io = Arc::new(RecordingIo::default());
        let requests = || io.requests.lock().unwrap().clone();
        let outer = Arc::new(RequestBatch::new(io.clone()));
        let outer_guard = outer.late_read_guard();
        outer.finish_scheduling();
        let first = outer.submit_request(vec![0..size], 1);
        drop(outer_guard);
        settle().await;

        // Each list page schedules its items through a nested batch of its own.
        let mut nested_reads = Vec::new();
        for idx in 1..3 {
            let nested = Arc::new(RequestBatch::nested_in(outer.clone()));
            let nested_guard = nested.late_read_guard();
            nested.finish_scheduling();
            nested_reads.push(nested.submit_request(vec![idx * size..(idx + 1) * size], idx));
            drop(nested_guard);
            settle().await;
        }
        assert_eq!(
            requests().len(),
            1,
            "nested groups wait for room in the run's window"
        );
        drop(first);
        assert_eq!(requests(), [vec![0..size], vec![size..2 * size]]);
        let mut nested_reads = nested_reads.into_iter();
        nested_reads.next().unwrap().await.unwrap();
        assert_eq!(
            requests(),
            [
                vec![0..size],
                vec![size..2 * size],
                vec![2 * size..3 * size]
            ],
            "consuming a nested read makes room for the next nested group"
        );
    }

    #[tokio::test]
    async fn awaiting_a_nested_step_read_leaves_the_held_late_reads_queued() {
        let size = IO_REQUEST_BATCH_BYTES * 3 / 4;
        let io = Arc::new(RecordingIo::default());
        let requests = || io.requests.lock().unwrap().clone();
        let outer = Arc::new(RequestBatch::new(io.clone()));
        let first_guard = outer.late_read_guard();
        outer.flush();
        let second_guard = outer.late_read_guard();
        outer.finish_scheduling();
        let first = outer.submit_request(vec![0..size], 1);
        drop(first_guard);
        settle().await;
        let held = outer.submit_request(vec![size..2 * size], 2);
        drop(second_guard);
        settle().await;
        assert_eq!(
            requests(),
            [vec![0..size]],
            "the second read waits for room"
        );

        // A list page's task schedules its items through a nested batch and
        // awaits their offsets at once.
        let nested = RequestBatch::nested_in(outer.clone());
        let offsets = nested.submit_request(vec![4 * size..4 * size + 8], 3);
        nested.flush();
        offsets.await.unwrap();
        assert_eq!(requests(), [vec![0..size], vec![4 * size..4 * size + 8]]);

        // Also when the items' reads reach a request batch before their step
        // flushes.
        let nested = RequestBatch::nested_in(outer.clone());
        let items = nested.submit_request(vec![7 * size..7 * size + 8], 4);
        let _more_items =
            nested.submit_request(vec![8 * size..8 * size + IO_REQUEST_BATCH_BYTES], 5);
        items.await.unwrap();
        assert_eq!(requests()[2], [7 * size..7 * size + 8]);

        drop(first);
        assert_eq!(requests().len(), 4, "consuming a read sends the held one");
        held.await.unwrap();

        // Late reads that fit in the window go with the nested step reads.
        let _open_guard = outer.late_read_guard();
        let _late = outer.submit_request(vec![5 * size..5 * size + 8], 6);
        let nested = RequestBatch::nested_in(outer.clone());
        let offsets = nested.submit_request(vec![6 * size..6 * size + 8], 7);
        nested.flush();
        offsets.await.unwrap();
        assert_eq!(
            requests()[4],
            [5 * size..5 * size + 8, 6 * size..6 * size + 8]
        );
    }

    #[tokio::test]
    async fn awaiting_a_nested_step_read_before_the_run_finishes_sends_every_queued_read() {
        let size = IO_REQUEST_BATCH_BYTES * 3 / 4;
        let io = Arc::new(RecordingIo::default());
        let requests = || io.requests.lock().unwrap().clone();
        let outer = Arc::new(RequestBatch::new(io.clone()));
        let late_guard = outer.late_read_guard();
        let _step = outer.submit_request(vec![0..size], 1);
        let nested = RequestBatch::nested_in(outer.clone());
        let offsets = nested.submit_request(vec![2 * size..2 * size + 8], 2);
        nested.flush();
        offsets.await.unwrap();
        assert_eq!(requests(), [vec![0..size, 2 * size..2 * size + 8]]);

        // The step's read takes no room in the window.
        outer.finish_scheduling();
        let _late = outer.submit_request(vec![size..2 * size], 3);
        drop(late_guard);
        settle().await;
        assert_eq!(requests().len(), 2);
    }

    #[tokio::test]
    async fn request_batch_keeps_late_reads_within_an_eighth_of_the_io_buffer() {
        // A 1 MiB read-ahead budget leaves 128 KiB for late reads sent ahead.
        let io = Arc::new(RecordingIo {
            io_buffer_size: Some(1 << 20),
            ..Default::default()
        });
        let requests = || io.requests.lock().unwrap().clone();
        let batch = Arc::new(RequestBatch::new(io.clone()));
        let first_guard = batch.late_read_guard();
        batch.flush();
        let second_guard = batch.late_read_guard();
        batch.finish_scheduling();

        let size = 96 << 10;
        let first = batch.submit_request(vec![0..size], 1);
        drop(first_guard);
        settle().await;
        let second = batch.submit_request(vec![size..2 * size], 2);
        drop(second_guard);
        settle().await;
        assert_eq!(
            requests(),
            [vec![0..size]],
            "the second group waits for room"
        );
        first.await.unwrap();
        assert_eq!(requests().len(), 2, "consuming a read makes room");
        drop(second);

        // A group larger than the window is not sent ahead, but awaiting a
        // nested step read takes it along.
        let third_guard = batch.late_read_guard();
        let third = batch.submit_request(vec![1 << 20..(1 << 20) + (256 << 10)], 3);
        drop(third_guard);
        settle().await;
        assert_eq!(requests().len(), 2, "an oversized group is not sent ahead");
        let nested = RequestBatch::nested_in(batch.clone());
        let offsets = nested.submit_request(vec![2 << 20..(2 << 20) + 8], 4);
        nested.flush();
        offsets.await.unwrap();
        assert_eq!(
            requests()[2],
            [1 << 20..(1 << 20) + (256 << 10), 2 << 20..(2 << 20) + 8]
        );
        drop(third);
    }

    /// Encodes 100 rows of `data_type` as 2.0 and schedules all of them on a
    /// [`RecordingIo`]. The pages' background tasks have not run when this returns.
    async fn schedule_v2_0(
        data_type: &DataType,
    ) -> (
        EncodedBatch,
        Arc<RecordingIo>,
        mpsc::UnboundedReceiver<Result<DecoderMessage>>,
    ) {
        let batch = gen_batch()
            .col("value", array::rand_type(data_type))
            .into_batch_rows(RowCount::from(100))
            .unwrap();
        let schema = Schema::try_from(batch.schema().as_ref()).unwrap();
        let strategy = test_encoding_strategy(TestEncoding::Array);
        let encoded = encode_batch(
            &batch,
            Arc::new(schema),
            strategy.as_ref(),
            &EncodingOptions::default(),
        )
        .await
        .unwrap();
        let io = Arc::new(RecordingIo {
            data: Some(BufferScheduler::new(encoded.data.clone())),
            ..Default::default()
        });
        let filter = FilterExpression::no_filter();
        let mut scheduler = DecodeBatchScheduler::try_new(
            encoded.schema.as_ref(),
            &encoded.top_level_columns,
            &encoded.page_table,
            &vec![],
            encoded.num_rows,
            Arc::new(DecoderPlugins::default()),
            io.clone(),
            Arc::new(LanceCache::no_cache()),
            &filter,
            &DecoderConfig::default(),
        )
        .await
        .unwrap();
        let (tx, rx) = unbounded_channel();
        scheduler.schedule_range(0..encoded.num_rows, &filter, tx, io.clone());
        (encoded, io, rx)
    }

    // A string page reads its bytes, and a list page its items, once its offsets
    // arrive. A list of strings nests one in the other.
    #[rstest]
    #[case::strings(DataType::Utf8)]
    #[case::list_of_ints(DataType::new_list(DataType::Int32, true))]
    #[case::list_of_strings(DataType::new_list(DataType::Utf8, true))]
    #[tokio::test]
    async fn test_v2_0_indirect_reads_are_sent_before_decode(#[case] data_type: DataType) {
        let (encoded, io, rx) = schedule_v2_0(&data_type).await;
        // In-memory reads complete at once, so yielding lets background tasks finish.
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        let requests_before_decode = io.requests.lock().unwrap().len();

        let mut decode_stream = create_decode_stream(
            &encoded.schema,
            encoded.num_rows,
            /*batch_size=*/ 1024,
            /*is_structural=*/ false,
            /*should_validate=*/ true,
            /*spawn_structural_batch_decode_tasks=*/ false,
            rx,
            None,
        )
        .unwrap();
        while let Some(task) = decode_stream.next().await {
            task.task.await.unwrap();
        }
        assert_eq!(
            io.requests.lock().unwrap().len(),
            requests_before_decode,
            "a read was held until the decoder awaited it"
        );
    }

    #[rstest]
    #[case::strings(DataType::Utf8)]
    #[case::list_of_ints(DataType::new_list(DataType::Int32, true))]
    #[case::list_of_strings(DataType::new_list(DataType::Utf8, true))]
    #[tokio::test]
    async fn test_v2_0_indirect_reads_are_skipped_once_the_decoder_is_dropped(
        #[case] data_type: DataType,
    ) {
        let (_, io, rx) = schedule_v2_0(&data_type).await;
        let offsets_requests = io.requests.lock().unwrap().len();
        assert!(offsets_requests > 0);

        // The background tasks run once this test yields, after the receiver
        // has dropped the page decoders.
        drop(rx);
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            io.requests.lock().unwrap().len(),
            offsets_requests,
            "an indirect read was sent for a dropped decoder"
        );
    }
}
