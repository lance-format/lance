// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dedicated thread for io_uring operations.
//!
//! This module provides a background thread that owns an io_uring instance
//! and processes read requests from a channel. Readers send requests via
//! an MPSC channel, and the thread handles submission and completion processing.

use super::DEFAULT_URING_QUEUE_DEPTH;
use super::requests::IoRequest;
use io_uring::{IoUring, opcode, types};
use std::collections::HashMap;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, sync_channel};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

/// Handle to the io_uring background thread.
///
/// This provides a channel sender for submitting read requests to the thread.
pub(super) struct UringThreadHandle {
    pub request_tx: SyncSender<QueuedRequest>,
    pub is_alive: Arc<AtomicBool>,
}

/// Owns the obligation to fail a request until a worker accepts it.
///
/// Dropping the receiver also drops every queued item. Keeping the failure
/// obligation with the queued item guarantees that a request accepted during
/// worker shutdown cannot be abandoned without waking its future.
pub(super) struct QueuedRequest {
    request: Option<Arc<IoRequest>>,
}

impl QueuedRequest {
    pub(super) fn new(request: Arc<IoRequest>) -> Self {
        SUBMITTED_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            request: Some(request),
        }
    }

    fn into_request(mut self) -> Arc<IoRequest> {
        SUBMITTED_COUNTER.fetch_sub(1, Ordering::Relaxed);
        self.request.take().unwrap()
    }

    fn fail(mut self, error: io::Error) {
        SUBMITTED_COUNTER.fetch_sub(1, Ordering::Relaxed);
        self.request.take().unwrap().fail(error);
    }
}

impl Drop for QueuedRequest {
    fn drop(&mut self) {
        if let Some(request) = self.request.take() {
            SUBMITTED_COUNTER.fetch_sub(1, Ordering::Relaxed);
            request.fail(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "io_uring worker stopped before accepting request",
            ));
        }
    }
}

pub(super) struct UringThreadPool {
    pub threads: Vec<UringThreadHandle>,
    pub initialization_errors: Vec<String>,
}

/// Lazy-initialized io_uring thread pool.
///
/// Multiple threads are spawned on first access and run until process exit.
pub(super) static URING_THREADS: LazyLock<UringThreadPool> = LazyLock::new(|| {
    let queue_depth = get_queue_depth();
    let thread_count = get_thread_count();

    let mut threads = Vec::with_capacity(thread_count);
    let mut initialization_errors = Vec::new();

    for i in 0..thread_count {
        match start_uring_thread(queue_depth, i) {
            Ok(thread) => threads.push(thread),
            Err(error) => {
                let message = format!("thread {i}: {error}");
                log::error!("Failed to start io_uring {message}");
                initialization_errors.push(message);
            }
        }
    }

    log::info!(
        "io_uring thread pool spawned ({} threads, queue_depth={})",
        threads.len(),
        queue_depth
    );

    UringThreadPool {
        threads,
        initialization_errors,
    }
});

fn start_uring_thread(queue_depth: usize, thread_id: usize) -> io::Result<UringThreadHandle> {
    // Initialize the ring before publishing its sender so a request can never be
    // accepted by a worker that subsequently fails during startup.
    let ring = IoUring::builder().build(queue_depth as u32)?;
    let (request_tx, request_rx) = sync_channel(queue_depth);
    let is_alive = Arc::new(AtomicBool::new(true));
    let worker_is_alive = Arc::clone(&is_alive);

    std::thread::Builder::new()
        .name(format!("lance-uring-{}", thread_id))
        .spawn(move || run_uring_thread(ring, request_rx, worker_is_alive, thread_id))?;

    Ok(UringThreadHandle {
        request_tx,
        is_alive,
    })
}

/// Atomic counter for round-robin thread selection.
pub(super) static THREAD_SELECTOR: AtomicU64 = AtomicU64::new(0);

/// Counter for generating unique user_data values.
///
/// Each io_uring operation needs a unique user_data ID to match completions
/// with their corresponding requests.
static USER_DATA_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Counter for requests that have been submitted to the thread but not yet received.
///
/// This tracks requests sitting in the channel queue waiting to be received by the thread.
pub(super) static SUBMITTED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Default batch size for submission - how many requests to batch before calling submit().
const DEFAULT_SUBMIT_BATCH_SIZE: usize = 128;

/// Default number of io_uring threads.
const DEFAULT_URING_THREAD_COUNT: usize = 2;

/// Get the configured queue depth from environment variable.
fn get_queue_depth() -> usize {
    std::env::var("LANCE_URING_QUEUE_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_URING_QUEUE_DEPTH)
}

/// Get the configured poll timeout from environment variable.
fn get_poll_timeout() -> Duration {
    let timeout_ms = std::env::var("LANCE_URING_POLL_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    Duration::from_millis(timeout_ms)
}

/// Get the configured submit batch size from environment variable.
fn get_submit_batch_size() -> usize {
    std::env::var("LANCE_URING_SUBMIT_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SUBMIT_BATCH_SIZE)
}

/// Get the configured number of uring threads from environment variable.
fn get_thread_count() -> usize {
    std::env::var("LANCE_URING_THREAD_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_URING_THREAD_COUNT)
}

/// Main loop for the io_uring thread.
///
/// This thread:
/// 1. Receives requests from the channel
/// 2. Submits them to io_uring
/// 3. Processes completions
/// 4. Wakes futures via their wakers
fn run_uring_thread(
    mut ring: IoUring,
    request_rx: Receiver<QueuedRequest>,
    is_alive: Arc<AtomicBool>,
    thread_id: usize,
) {
    let queue_depth = ring.submission().capacity();
    let mut pending: HashMap<u64, Arc<IoRequest>> = HashMap::with_capacity(queue_depth);
    let poll_timeout = get_poll_timeout();
    let submit_batch_size = get_submit_batch_size();
    let mut last_log = Instant::now();
    let log_interval = Duration::from_millis(100);
    let mut completed_iops = 0usize;
    let mut completed_sectors = 0usize;
    let mut min_in_flight = usize::MAX;

    loop {
        // Track minimum in-flight count
        let in_flight = pending.len();
        min_in_flight = min_in_flight.min(in_flight);

        // Log in-flight requests every 100ms
        let now = Instant::now();
        if now.duration_since(last_log) >= log_interval {
            let submitted = SUBMITTED_COUNTER.load(Ordering::Relaxed);
            log::info!(
                "io_uring[{}]: {} submitted, {} in flight (min {}), {} iops completed, {} sectors completed",
                thread_id,
                submitted,
                in_flight,
                min_in_flight,
                completed_iops,
                completed_sectors
            );
            last_log = now;
            completed_iops = 0; // Reset counter after logging
            completed_sectors = 0; // Reset counter after logging
            min_in_flight = usize::MAX; // Reset min tracker
        }

        // Process all available completions first
        let mut needs_submit = false;
        let completions = process_completions(&mut ring, &mut pending);
        match completions {
            Ok(result) => {
                completed_iops += result.iops;
                completed_sectors += result.sectors;

                // Resubmit any short-read retries
                for request in result.retries {
                    if let Err(e) = push_to_sq(&mut ring, &mut pending, request) {
                        log::error!("Failed to resubmit short read: {}", e);
                    } else {
                        needs_submit = true;
                    }
                }
            }
            Err(e) => {
                log::error!("Error processing io_uring completions: {}", e);
            }
        }

        min_in_flight = min_in_flight.min(pending.len());

        // Batch submit requests - keep pulling from channel and pushing to SQ
        // until we hit batch size or channel is empty
        let mut batch_count = 0;
        loop {
            // Try to receive new request
            // Use recv_timeout only when pending is empty, otherwise use try_recv
            let recv_result = if pending.is_empty() && batch_count == 0 {
                // No operations in flight and no batch started - we can afford to wait with timeout
                request_rx.recv_timeout(poll_timeout).map_err(|e| match e {
                    RecvTimeoutError::Timeout => std::sync::mpsc::TryRecvError::Empty,
                    RecvTimeoutError::Disconnected => std::sync::mpsc::TryRecvError::Disconnected,
                })
            } else {
                // Operations in flight or batch in progress - busy loop with try_recv
                request_rx.try_recv()
            };

            match recv_result {
                Ok(request) => {
                    let request = request.into_request();

                    // Push to submission queue (but don't submit yet)
                    if let Err(e) = push_to_sq(&mut ring, &mut pending, request) {
                        log::error!("Failed to push to io_uring SQ: {}", e);
                    } else {
                        batch_count += 1;
                    }

                    // Break if we've hit the batch size limit
                    if batch_count >= submit_batch_size {
                        break;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // No more requests in channel - break to submit the batch
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    // All senders dropped - submit batch and shutdown
                    if batch_count > 0 {
                        let queued = ring.submission().len();
                        if let Err(error) = submit_all(queued, || ring.submit()) {
                            shutdown_with_error(
                                ring,
                                pending,
                                &request_rx,
                                &is_alive,
                                thread_id,
                                error,
                            );
                            return;
                        }
                    }
                    log::info!(
                        "io_uring thread {} shutting down (channel disconnected)",
                        thread_id
                    );
                    return;
                }
            }
        }

        // Submit if we have any requests (from channel or retries)
        if batch_count > 0 || needs_submit {
            let queued = ring.submission().len();
            if let Err(error) = submit_all(queued, || ring.submit()) {
                shutdown_with_error(ring, pending, &request_rx, &is_alive, thread_id, error);
                return;
            }
        }
    }
}

/// Submit every entry currently published to the submission queue.
///
/// `io_uring_enter` may be interrupted or accept only part of a batch. The
/// remaining entries stay in the userspace submission queue and must be retried;
/// otherwise their requests remain pending without a possible completion.
fn submit_all(mut queued: usize, mut submit: impl FnMut() -> io::Result<usize>) -> io::Result<()> {
    while queued > 0 {
        match submit() {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    format!("io_uring submitted 0 of {queued} queued requests"),
                ));
            }
            Ok(submitted) if submitted <= queued => queued -= submitted,
            Ok(submitted) => {
                return Err(io::Error::other(format!(
                    "io_uring reported {submitted} submissions for {queued} queued requests"
                )));
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }

    Ok(())
}

fn shutdown_with_error(
    ring: IoUring,
    mut pending: HashMap<u64, Arc<IoRequest>>,
    request_rx: &Receiver<QueuedRequest>,
    is_alive: &AtomicBool,
    thread_id: usize,
    error: io::Error,
) {
    let error_kind = error.kind();
    let error_message = format!("io_uring worker {thread_id} stopped: {error}");
    log::error!("{}", error_message);

    // Closing the ring cancels in-flight operations before their request buffers
    // can be released by the futures receiving the errors below.
    drop(ring);
    is_alive.store(false, Ordering::Release);

    for request in pending.drain().map(|(_, request)| request) {
        request.fail(io::Error::new(error_kind, error_message.clone()));
    }
    for request in request_rx.try_iter() {
        request.fail(io::Error::new(error_kind, error_message.clone()));
    }
}

/// Push a read request to the io_uring submission queue (without submitting).
///
/// This generates a unique user_data ID, prepares the read operation,
/// and pushes it to the SQ. The caller is responsible for calling ring.submit().
pub(super) fn push_to_sq(
    ring: &mut IoUring,
    pending: &mut HashMap<u64, Arc<IoRequest>>,
    request: Arc<IoRequest>,
) -> io::Result<()> {
    // Generate unique user_data
    let user_data = USER_DATA_COUNTER.fetch_add(1, Ordering::Relaxed);

    // Get buffer pointer, adjusting for any bytes already read (short read retry)
    let (buffer_ptr, read_offset, read_length) = {
        let state = request.state.lock().unwrap();
        let br = state.bytes_read;
        (
            unsafe { state.buffer.as_ptr().add(br) as *mut u8 },
            request.offset + br as u64,
            (request.length - br) as u32,
        )
    };

    // Prepare read operation
    let read_op =
        opcode::Read::new(types::Fd(request.fd), buffer_ptr, read_length).offset(read_offset);

    // Get submission queue
    let mut sq = ring.submission();

    // Check if SQ has space
    if sq.is_full() {
        drop(sq);
        request.fail(io::Error::new(
            io::ErrorKind::WouldBlock,
            "io_uring submission queue full",
        ));
        return Err(io::Error::new(
            io::ErrorKind::WouldBlock,
            "io_uring submission queue full",
        ));
    }

    // Push to SQ
    unsafe {
        if sq.push(&read_op.build().user_data(user_data)).is_err() {
            drop(sq);
            request.fail(io::Error::other("Failed to push to SQ"));
            return Err(io::Error::other("Failed to push to SQ"));
        }
    }
    drop(sq);

    // Track request in pending map
    pending.insert(user_data, request);

    Ok(())
}

struct CompletionResult {
    iops: usize,
    sectors: usize,
    retries: Vec<Arc<IoRequest>>,
}

/// Process all available completions from the io_uring.
///
/// This iterates through the completion queue, matches completions to requests,
/// updates their state, and wakes any waiting futures. Short reads are collected
/// into `retries` for resubmission; EOF before a full read is an error.
///
/// Returns completion stats and a list of requests needing resubmission.
fn process_completions(
    ring: &mut IoUring,
    pending: &mut HashMap<u64, Arc<IoRequest>>,
) -> io::Result<CompletionResult> {
    let mut iops = 0;
    let mut sectors = 0;
    let mut retries = Vec::new();

    // Process all available completions
    for cqe in ring.completion() {
        let user_data = cqe.user_data();
        let result = cqe.result();

        // Look up request
        if let Some(request) = pending.remove(&user_data) {
            let mut state = request.state.lock().unwrap();

            if result < 0 {
                // Kernel error
                state.err = Some(io::Error::from_raw_os_error(-result));
                state.completed = true;
            } else if result == 0 {
                // EOF before full read completed
                let br = state.bytes_read;
                state.err = Some(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("unexpected EOF: read {} of {} bytes", br, request.length),
                ));
                state.buffer.truncate(br);
                state.completed = true;
            } else {
                // Positive result: n bytes read
                let n = result as usize;
                state.bytes_read += n;
                let br = state.bytes_read;

                if br >= request.length {
                    // Full read complete
                    state.buffer.truncate(br);
                    state.completed = true;

                    if request.length > 0 {
                        let first_sector = request.offset / 4096;
                        let last_sector = (request.offset + request.length as u64 - 1) / 4096;
                        let num_sectors = (last_sector - first_sector + 1) as usize;
                        sectors += num_sectors;
                    }
                } else {
                    // Short read — need retry; don't mark completed or wake
                    drop(state);
                    retries.push(request);
                    continue;
                }
            }

            // Wake the future if it's waiting
            if let Some(waker) = state.waker.take() {
                drop(state); // Release lock before waking
                waker.wake();
            }

            iops += 1;
        } else {
            log::warn!("Received completion for unknown user_data: {}", user_data);
        }
    }

    Ok(CompletionResult {
        iops,
        sectors,
        retries,
    })
}

#[cfg(test)]
mod tests {
    use super::{QueuedRequest, start_uring_thread, submit_all};
    use crate::uring::requests::{IoRequest, RequestState};
    use bytes::BytesMut;
    use std::collections::VecDeque;
    use std::io;
    use std::sync::mpsc::sync_channel;
    use std::sync::{Arc, Barrier, Mutex};
    use std::thread;

    #[test]
    fn test_submit_all_retries_interrupted_and_partial_submissions() {
        let mut results = VecDeque::from([
            Err(io::Error::from(io::ErrorKind::Interrupted)),
            Ok(1),
            Ok(2),
        ]);

        submit_all(3, || results.pop_front().unwrap()).unwrap();

        assert!(results.is_empty());
    }

    #[test]
    fn test_submit_all_rejects_zero_progress() {
        let error = submit_all(1, || Ok(0)).unwrap_err();

        assert_eq!(error.kind(), io::ErrorKind::WriteZero);
        assert!(error.to_string().contains("submitted 0 of 1"));
    }

    #[test]
    fn test_worker_is_not_published_when_ring_initialization_fails() {
        let result = start_uring_thread(0, 0);

        assert!(result.is_err());
    }

    #[test]
    fn test_late_queued_request_is_failed_when_worker_receiver_drops() {
        let request = Arc::new(IoRequest {
            fd: -1,
            offset: 0,
            length: 1,
            thread_id: thread::current().id(),
            state: Mutex::new(RequestState {
                completed: false,
                waker: None,
                err: None,
                buffer: BytesMut::zeroed(1),
                bytes_read: 0,
            }),
        });
        let (request_tx, request_rx) = sync_channel(1);
        let drained = Arc::new(Barrier::new(2));
        let request_sent = Arc::new(Barrier::new(2));

        let sender = {
            let request = Arc::clone(&request);
            let drained = Arc::clone(&drained);
            let request_sent = Arc::clone(&request_sent);
            thread::spawn(move || {
                drained.wait();
                assert!(request_tx.send(QueuedRequest::new(request)).is_ok());
                request_sent.wait();
            })
        };

        assert!(request_rx.try_recv().is_err());
        drained.wait();
        request_sent.wait();
        drop(request_rx);
        sender.join().unwrap();

        let state = request.state.lock().unwrap();
        assert!(state.completed);
        assert_eq!(
            state.err.as_ref().unwrap().kind(),
            io::ErrorKind::BrokenPipe
        );
    }
}
