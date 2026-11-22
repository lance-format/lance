// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicUsize, Ordering};

use super::format::{DEFAULT_FTS_SEARCH_CHUNK, configured_fts_search_chunk};

/// Preserve the original partition-load window when scoring batches shrink.
pub(super) fn buffered_search_chunks(
    partitions: usize,
    chunk: usize,
    workers: usize,
    io_parallelism: usize,
) -> usize {
    if chunk >= DEFAULT_FTS_SEARCH_CHUNK || configured_fts_search_chunk().is_some() {
        return workers;
    }
    // The caller caps workers at 32, bounding this product to 512. Keeping
    // only `workers` small batches in flight would reduce remote I/O overlap.
    let load_window = partitions.div_ceil(DEFAULT_FTS_SEARCH_CHUNK).min(workers)
        * DEFAULT_FTS_SEARCH_CHUNK.min(io_parallelism);
    load_window.div_ceil(chunk.min(io_parallelism)).max(1)
}

// All indices share the CPU pool, so query pressure must span index instances.
pub(super) static MODERN_SEARCHES: SearchConcurrency = SearchConcurrency {
    active: AtomicUsize::new(0),
};

#[derive(Default)]
pub(super) struct SearchConcurrency {
    active: AtomicUsize,
}

impl SearchConcurrency {
    pub(super) fn enter(&self) -> SearchGuard<'_> {
        self.active.fetch_add(1, Ordering::Relaxed);
        SearchGuard(self)
    }
}

pub(super) struct SearchGuard<'a>(&'a SearchConcurrency);

impl SearchGuard<'_> {
    pub(super) fn chunk_size(&self, partitions: usize, workers: usize) -> usize {
        configured_fts_search_chunk().unwrap_or_else(|| {
            // Keep roughly two waves of work per available worker so uneven
            // partitions do not leave a lone query waiting on a large batch.
            // Concurrent searches already supply parallel work and use larger
            // batches. Power-of-two sizes avoid small changes in query pressure
            // repeatedly changing the task shape.
            let active = self.0.active.load(Ordering::Relaxed).max(1);
            let workers_per_search = workers.div_ceil(active).max(1);
            (partitions / workers_per_search / 2)
                .clamp(2, DEFAULT_FTS_SEARCH_CHUNK)
                .next_power_of_two()
        })
    }
}

impl Drop for SearchGuard<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::Relaxed);
    }
}

/// File-backed phrase cursors perform blocking reads and on a cache fault
/// may await canonical I/O that itself uses Tokio's blocking pool. A separate
/// worker pool leaves both the runtime and its CPU/IO workers available.
pub(in crate::scalar::inverted) async fn run_fts_scoring<T: Send + 'static>(
    reads_positions: bool,
    score: impl FnOnce() -> lance_core::Result<T> + Send + 'static,
) -> lance_core::Result<T> {
    if !reads_positions {
        return lance_core::utils::tokio::spawn_cpu(score).await;
    }
    static FILE_SCORERS: std::sync::LazyLock<
        Result<rayon::ThreadPool, rayon::ThreadPoolBuildError>,
    > = std::sync::LazyLock::new(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(lance_core::utils::tokio::get_num_compute_intensive_cpus().clamp(1, 32))
            .thread_name(|i| format!("lance-fts-read-{i}"))
            .build()
    });
    let pool = FILE_SCORERS
        .as_ref()
        .map_err(|e| lance_core::Error::io(e.to_string()))?;
    let runtime = tokio::runtime::Handle::current();
    let (tx, rx) = tokio::sync::oneshot::channel();
    pool.spawn_fifo(move || {
        if tx.is_closed() {
            return;
        }
        let _runtime = runtime.enter();
        let _ = tx.send(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            score,
        )));
    });
    match rx
        .await
        .map_err(|e| lance_core::Error::internal(e.to_string()))?
    {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_scoring_leaves_tokio_blocking_capacity_for_origin_reads() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let result = rt.block_on(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(5),
                super::run_fts_scoring(true, || {
                    tokio::runtime::Handle::current()
                        .block_on(async { Ok(tokio::task::spawn_blocking(|| 41).await? + 1) })
                }),
            )
            .await
        });
        rt.shutdown_timeout(std::time::Duration::from_secs(1));
        assert_eq!(result.unwrap().unwrap(), 42);
    }

    #[rstest::rstest]
    #[case::remote(128, 2, 32, 128, 128)]
    #[case::local(128, 2, 32, 8, 64)]
    #[case::partial_last_batch(37, 2, 32, 8, 21)]
    #[case::few_workers(128, 4, 2, 8, 16)]
    #[case::empty(0, 2, 32, 8, 0)]
    fn scoring_batches_preserve_load_parallelism(
        #[case] partitions: usize,
        #[case] chunk: usize,
        #[case] workers: usize,
        #[case] io_parallelism: usize,
        #[case] baseline_loads: usize,
    ) {
        let concurrency = buffered_search_chunks(partitions, chunk, workers, io_parallelism);
        let partitions = (0..partitions).collect::<Vec<_>>();
        let initial_loads: usize = partitions
            .chunks(chunk)
            .take(concurrency)
            .map(|batch| batch.len().min(io_parallelism))
            .sum();
        assert!(initial_loads >= baseline_loads);
        assert!(concurrency > 0);
    }

    #[test]
    fn concurrent_searches_share_workers() {
        let concurrency = SearchConcurrency::default();
        let first = concurrency.enter();
        assert_eq!(first.chunk_size(128, 32), 2);
        let second = concurrency.enter();
        assert_eq!(second.chunk_size(128, 32), 4);
        let third = concurrency.enter();
        let fourth = concurrency.enter();
        assert_eq!(fourth.chunk_size(128, 32), 8);
        drop((second, third, fourth));
        assert_eq!(first.chunk_size(128, 32), 2);
    }

    #[rstest::rstest]
    #[case::few_partitions(2, 32, 2)]
    #[case::one_worker(128, 1, 16)]
    #[case::many_partitions(1024, 32, 16)]
    fn batches_remain_bounded(
        #[case] partitions: usize,
        #[case] workers: usize,
        #[case] expected: usize,
    ) {
        let concurrency = SearchConcurrency::default();
        assert_eq!(
            concurrency.enter().chunk_size(partitions, workers),
            expected
        );
    }

    #[tokio::test]
    async fn cancellation_releases_query_pressure() {
        let concurrency = std::sync::Arc::new(SearchConcurrency::default());
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn({
            let concurrency = concurrency.clone();
            async move {
                let _guard = concurrency.enter();
                ready_tx.send(()).unwrap();
                std::future::pending::<()>().await;
            }
        });
        ready_rx.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(concurrency.enter().chunk_size(128, 32), 2);
    }
}
