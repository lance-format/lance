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

#[cfg(test)]
mod tests {
    use super::*;

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
