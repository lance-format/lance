// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Coordinate overlapping Linux owned reads without parking an executor thread
//! or serializing disjoint ranges. Ordinary local reads do not consult this gate.
//! The registry holds only weak metadata references, never file handles or payload.

use std::fs::File;
use std::ops::Range;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::{collections::HashMap, os::unix::fs::MetadataExt, sync::OnceLock, sync::Weak};
#[cfg(target_os = "linux")]
use tokio::sync::{OnceCell, watch};

use super::join_local_io;

#[derive(Debug)]
pub(super) enum ReadRanges {
    Single(Range<usize>),
    Batch(Vec<Range<usize>>),
}

impl ReadRanges {
    fn as_slice(&self) -> &[Range<usize>] {
        match self {
            Self::Single(range) => std::slice::from_ref(range),
            Self::Batch(ranges) => ranges,
        }
    }

    #[cfg(target_os = "linux")]
    fn overlaps(&self, other: &Self) -> bool {
        self.as_slice().iter().any(|a| {
            !a.is_empty()
                && other
                    .as_slice()
                    .iter()
                    .any(|b| !b.is_empty() && a.start < b.end && b.start < a.end)
        })
    }
}

#[cfg(target_os = "linux")]
#[derive(Debug)]
struct Request {
    ranges: ReadRanges,
    completed: watch::Receiver<()>,
}

#[cfg(target_os = "linux")]
#[derive(Debug, Default)]
struct Coordinator {
    requests: std::sync::Mutex<Vec<Weak<Request>>>,
}

#[cfg(target_os = "linux")]
struct ReadPermit {
    // Keep the inode identity live even if the last reader is dropped while its
    // blocking read is detached. Completion closes the channel without a send.
    _coordinator: Arc<Coordinator>,
    _request: Arc<Request>,
    _completed: watch::Sender<()>,
}

#[cfg(target_os = "linux")]
impl Coordinator {
    async fn acquire(self: &Arc<Self>, ranges: ReadRanges) -> std::io::Result<ReadPermit> {
        let (permit, blockers) = {
            let mut requests = self
                .requests
                .lock()
                .map_err(|_| std::io::Error::other("local read coordination poisoned"))?;
            requests.retain(|request| request.strong_count() != 0);
            let retained_capacity = requests.len().saturating_mul(2);
            requests.shrink_to(retained_capacity);
            let blockers = requests
                .iter()
                .filter_map(Weak::upgrade)
                .filter(|request| ranges.overlaps(&request.ranges))
                .map(|request| request.completed.clone())
                .collect::<Vec<_>>();
            let (completed, receiver) = watch::channel(());
            let request = Arc::new(Request {
                ranges,
                completed: receiver,
            });
            requests.push(Arc::downgrade(&request));
            (
                ReadPermit {
                    _coordinator: self.clone(),
                    _request: request,
                    _completed: completed,
                },
                blockers,
            )
        };
        // Queue order applies only to conflicting ranges. Register before the
        // first wait so later overlaps cannot bypass us. A cancelled waiter drops
        // its permit and closes its channel; detached I/O keeps its permit instead.
        for mut blocker in blockers {
            // The sender never publishes a value: channel closure means the
            // earlier operation ended or was cancelled before starting I/O.
            let _ = blocker.changed().await;
        }
        Ok(permit)
    }
}

#[derive(Debug, Default)]
pub(super) struct ReadGate {
    #[cfg(target_os = "linux")]
    gate: OnceCell<Arc<Coordinator>>,
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct Registry {
    gates: HashMap<(u64, u64), Weak<Coordinator>>,
}

#[cfg(target_os = "linux")]
impl Registry {
    fn get(&mut self, key: (u64, u64)) -> Arc<Coordinator> {
        self.gates.retain(|_, gate| gate.strong_count() != 0);
        // Reclaim metadata from a previous high-water mark as readers close.
        self.gates.shrink_to(self.gates.len().saturating_mul(2));
        if let Some(gate) = self.gates.get(&key).and_then(Weak::upgrade) {
            return gate;
        }
        let gate = Arc::new(Coordinator::default());
        self.gates.insert(key, Arc::downgrade(&gate));
        gate
    }
}

impl ReadGate {
    #[cfg(target_os = "linux")]
    async fn get(&self, file: Arc<File>) -> object_store::Result<Arc<Coordinator>> {
        self.gate
            .get_or_try_init(|| async move {
                join_local_io(tokio::task::spawn_blocking(move || {
                    // fstat identifies the opened file, including hard links and
                    // paths replaced after open. It runs only on first owned I/O.
                    let metadata = file.metadata()?;
                    static REGISTRY: OnceLock<std::sync::Mutex<Registry>> = OnceLock::new();
                    let mut registry = REGISTRY
                        .get_or_init(Default::default)
                        .lock()
                        .map_err(|_| std::io::Error::other("local read gate registry poisoned"))?;
                    Ok(registry.get((metadata.dev(), metadata.ino())))
                }))
                .await
            })
            .await
            .cloned()
    }

    pub(super) async fn run<T, F>(
        &self,
        file: Arc<File>,
        ranges: ReadRanges,
        operation: F,
    ) -> object_store::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(Arc<File>) -> std::io::Result<T> + Send + 'static,
    {
        // Empty reads need neither an inode lookup nor a coordination permit.
        let is_empty = ranges.as_slice().iter().all(Range::is_empty);
        #[cfg(target_os = "linux")]
        let guard = if is_empty {
            None
        } else {
            Some(
                self.get(file.clone())
                    .await?
                    .acquire(ranges)
                    .await
                    .map_err(|error| object_store::Error::Generic {
                        store: "LocalFileSystem",
                        source: error.into(),
                    })?,
            )
        };
        #[cfg(not(target_os = "linux"))]
        let _ = is_empty;
        join_local_io(tokio::task::spawn_blocking(move || {
            // Cancellation of the async waiter detaches this closure. Both the
            // gate and the caller's buffers must remain here until I/O ends.
            #[cfg(target_os = "linux")]
            let _guard = guard;
            operation(file)
        }))
        .await
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests;
