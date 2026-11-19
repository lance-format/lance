// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Coordinate Linux owned reads to the same inode without parking an executor
//! thread. Ordinary local reads do not consult this gate. The registry holds
//! only weak metadata references, never file handles or payload allocations.

use std::fs::File;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use std::{collections::HashMap, os::unix::fs::MetadataExt, sync::OnceLock, sync::Weak};
#[cfg(target_os = "linux")]
use tokio::sync::{Mutex, OnceCell};

use super::join_local_io;

#[derive(Debug, Default)]
pub(super) struct ReadGate {
    #[cfg(target_os = "linux")]
    gate: OnceCell<Arc<Mutex<()>>>,
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct Registry {
    gates: HashMap<(u64, u64), Weak<Mutex<()>>>,
}

#[cfg(target_os = "linux")]
impl Registry {
    fn get(&mut self, key: (u64, u64)) -> Arc<Mutex<()>> {
        self.gates.retain(|_, gate| gate.strong_count() != 0);
        // Reclaim metadata from a previous high-water mark as readers close.
        self.gates.shrink_to(self.gates.len().saturating_mul(2));
        if let Some(gate) = self.gates.get(&key).and_then(Weak::upgrade) {
            return gate;
        }
        let gate = Arc::new(Mutex::new(()));
        self.gates.insert(key, Arc::downgrade(&gate));
        gate
    }
}

impl ReadGate {
    #[cfg(target_os = "linux")]
    async fn get(&self, file: Arc<File>) -> object_store::Result<Arc<Mutex<()>>> {
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

    pub(super) async fn run<T, F>(&self, file: Arc<File>, operation: F) -> object_store::Result<T>
    where
        T: Send + 'static,
        F: FnOnce(Arc<File>) -> std::io::Result<T> + Send + 'static,
    {
        #[cfg(target_os = "linux")]
        let guard = self.get(file.clone()).await?.lock_owned().await;
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
