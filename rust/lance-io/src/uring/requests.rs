// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Protocol types for communication between UringReader and the io_uring thread.

use bytes::BytesMut;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::ops::Deref;
use std::sync::{Arc, Mutex};
use std::task::Waker;
use std::thread::ThreadId;

pub(super) struct RequestState {
    pub completed: bool,
    pub waker: Option<Waker>,
    pub err: Option<io::Error>,
    pub buffer: BytesMut,
    /// Accumulated bytes read across retries (for handling short reads).
    pub bytes_read: usize,
}

/// I/O request object that contains all state for a single read operation.
/// This is shared between the submitter, uring thread, and future via Arc.
pub(super) struct IoRequest {
    /// File to read from, held so that its descriptor cannot be closed and reused by another
    /// file before the read is submitted.
    pub file: Arc<File>,

    /// Byte offset to start reading from.
    pub offset: u64,

    /// Number of bytes to read.
    pub length: usize,

    pub thread_id: ThreadId,

    /// Completion flag - set to true when operation completes.
    pub state: Mutex<RequestState>,
}

impl IoRequest {
    /// Mark this request as failed with the given error.
    ///
    /// Sets the error, marks completed, and wakes any waiting future.
    /// Used when a request cannot be submitted (e.g. SQ full).
    pub(super) fn fail(&self, err: io::Error) {
        let mut state = self.state.lock().unwrap();
        if state.completed {
            return;
        }
        state.err = Some(err);
        state.completed = true;
        if let Some(waker) = state.waker.take() {
            drop(state);
            waker.wake();
        }
    }
}

/// Requests pushed to a ring and not yet reaped, keyed by their `user_data`.
///
/// Closing a ring need not wait for the reads the kernel has already started, so their buffers
/// must outlive the ring. When a ring goes away without reaping them all (a worker stopping after
/// an error or a panic, or a thread-local ring that cannot wait at thread exit), dropping this map
/// leaks the remaining requests, with their buffers and files, instead of freeing them.
#[derive(Default)]
pub(super) struct PendingRequests(HashMap<u64, Arc<IoRequest>>);

impl PendingRequests {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }

    pub(super) fn insert(&mut self, user_data: u64, request: Arc<IoRequest>) {
        self.0.insert(user_data, request);
    }

    /// Takes a request out once the completion of its read has been reaped.
    pub(super) fn remove(&mut self, user_data: &u64) -> Option<Arc<IoRequest>> {
        self.0.remove(user_data)
    }
}

impl Deref for PendingRequests {
    type Target = HashMap<u64, Arc<IoRequest>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for PendingRequests {
    fn drop(&mut self) {
        for (_, request) in self.0.drain() {
            std::mem::forget(request);
        }
    }
}
