// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicUsize, Ordering};

use futures::{Stream, TryStreamExt};
use lance_core::{
    Error, Result,
    utils::tokio::{spawn, spawn_cpu},
};
use tokio::sync::OwnedSemaphorePermit;

use super::format::{DEFAULT_FTS_SEARCH_CHUNK, configured_fts_search_chunk};

/// Score loaded partitions without waiting for unrelated remote reads. At most
/// two CPU batches per load window overlap, keeping both stages bounded.
pub(super) async fn score_ready_partitions<T, R, S, F>(
    loads: S,
    chunk_size: usize,
    score: F,
) -> Result<Vec<R>>
where
    T: Send + 'static,
    R: Send + 'static,
    S: Stream<Item = Result<(T, OwnedSemaphorePermit)>>,
    F: Fn(Vec<T>) -> Result<Vec<R>> + Clone + Send + 'static,
{
    loads
        .try_ready_chunks(chunk_size)
        .map_err(|error| error.1)
        .map_ok(move |loaded| {
            let score = score.clone();
            let (loaded, permits): (Vec<_>, Vec<_>) = loaded.into_iter().unzip();
            // Keep admission until accepted CPU work finishes even when the
            // caller is cancelled. Release permits on the async thread, never
            // from a CPU closure. Dropping a JoinHandle leaves this finite drain
            // running; it does not admit replacement work prematurely.
            let task = spawn(async move {
                let result = spawn_cpu(move || score(loaded)).await;
                drop(permits);
                result
            });
            async move {
                task.await.map_err(|error| {
                    Error::internal(format!("modern FTS scoring task failed: {error}"))
                })?
            }
        })
        .try_buffer_unordered(2)
        .try_concat()
        .await
}

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
    pub(super) fn partition_load_limit(&self, workers: usize) -> usize {
        // Share two waves of loading/scoring work across active queries. Keep
        // two slots even with one CPU so I/O and pure CPU work can overlap.
        let active = self.0.active.load(Ordering::Relaxed).max(1);
        (workers * 2).div_ceil(active).max(2)
    }

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
    use futures::StreamExt;
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    fn admitted<T>(
        loads: impl Stream<Item = Result<T>>,
    ) -> impl Stream<Item = Result<(T, OwnedSemaphorePermit)>> {
        let permits = Arc::new(Semaphore::new(64));
        loads.then(move |item| {
            let permits = permits.clone();
            async move { Ok((item?, permits.acquire_owned().await.unwrap())) }
        })
    }

    #[tokio::test]
    async fn ready_partition_scores_before_pending_load() {
        let scored = std::sync::Arc::new(AtomicUsize::new(0));
        let progress = scored.clone();
        let loads = futures::stream::once(async { Ok(7) }).chain(futures::stream::once(async {
            // Model a remote read that cannot complete until already loaded
            // work has released its resources.
            while progress.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
            Ok(11)
        }));
        let mut result = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            score_ready_partitions(admitted(loads), 16, move |items| {
                scored.fetch_add(items.len(), Ordering::Release);
                Ok(items)
            }),
        )
        .await
        .expect("ready partitions must score without waiting for the whole load window")
        .unwrap();
        result.sort_unstable();
        assert_eq!(result, vec![7, 11]);
    }

    #[tokio::test]
    async fn ready_batches_bound_retained_inputs() {
        struct Input(std::sync::Arc<AtomicUsize>, usize);
        impl Drop for Input {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::Relaxed);
            }
        }
        let live = std::sync::Arc::new(AtomicUsize::new(0));
        let peak = std::sync::Arc::new(AtomicUsize::new(0));
        let loads = futures::stream::iter(0..33).map(|id| {
            let count = live.fetch_add(1, Ordering::Relaxed) + 1;
            peak.fetch_max(count, Ordering::Relaxed);
            Ok(Input(live.clone(), id))
        });
        let mut result = score_ready_partitions(admitted(loads), 4, |items| {
            assert!(items.len() <= 4);
            Ok(items.into_iter().map(|item| item.1).collect())
        })
        .await
        .unwrap();
        result.sort_unstable();
        assert_eq!(result, (0..33).collect::<Vec<_>>());
        assert_eq!(live.load(Ordering::Relaxed), 0);
        assert!(peak.load(Ordering::Relaxed) <= 8);
    }

    #[tokio::test]
    async fn large_load_window_drains_with_two_slots() {
        let permits = Arc::new(Semaphore::new(2));
        let loads = futures::stream::iter(0..33)
            .map(|id| {
                let permits = permits.clone();
                async move { Ok((id, permits.acquire_owned().await.unwrap())) }
            })
            .buffer_unordered(16);
        let mut result = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            score_ready_partitions(loads, 16, Ok),
        )
        .await
        .expect("a load window larger than admission must still drain")
        .unwrap();
        result.sort_unstable();
        assert_eq!(result, (0..33).collect::<Vec<_>>());
        assert_eq!(permits.available_permits(), 2);
    }

    #[tokio::test]
    async fn cancelled_scoring_retains_admission_until_cpu_finishes() {
        let permits = Arc::new(Semaphore::new(1));
        let permit = permits.clone().acquire_owned().await.unwrap();
        let stage = Arc::new(AtomicUsize::new(0));
        let progress = stage.clone();
        let task = tokio::spawn(score_ready_partitions(
            futures::stream::iter([Ok((7, permit))]),
            16,
            move |items| {
                progress.store(1, Ordering::Release);
                let mut checksum = 0_u64;
                // Finite pure CPU work; no parked CPU thread or release gate.
                for value in 0..1_000_000_u64 {
                    checksum = std::hint::black_box(checksum.wrapping_add(value).rotate_left(7));
                }
                std::hint::black_box(checksum);
                progress.store(2, Ordering::Release);
                Ok(items)
            },
        ));
        tokio::time::timeout(std::time::Duration::from_millis(500), async {
            while stage.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        task.abort();
        let _ = task.await;
        tokio::time::timeout(std::time::Duration::from_millis(500), async {
            loop {
                if permits.available_permits() == 1 {
                    assert_eq!(stage.load(Ordering::Acquire), 2);
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("accepted CPU work must finish and release admission after cancellation");
    }

    #[tokio::test]
    async fn ready_batches_preserve_errors_and_empty_input() {
        let empty = futures::stream::empty::<Result<usize>>();
        let result: Vec<usize> = score_ready_partitions(admitted(empty), 4, |_| {
            panic!("empty input must not dispatch a CPU task")
        })
        .await
        .unwrap();
        assert!(result.is_empty());

        let loads = futures::stream::iter([Ok(7), Err(lance_core::Error::io("load failed"))]);
        let error = score_ready_partitions(admitted(loads), 4, Ok)
            .await
            .unwrap_err();
        assert!(matches!(error, lance_core::Error::IO { .. }));
        assert!(error.to_string().contains("load failed"));

        let loads = futures::stream::iter([Ok(7)]);
        let error = score_ready_partitions::<_, usize, _, _>(admitted(loads), 4, |_| {
            Err(lance_core::Error::index("score failed"))
        })
        .await
        .unwrap_err();
        assert!(matches!(error, lance_core::Error::Index { .. }));
        assert!(error.to_string().contains("score failed"));
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
        assert_eq!(first.partition_load_limit(32), 64);
        let second = concurrency.enter();
        assert_eq!(second.chunk_size(128, 32), 4);
        assert_eq!(second.partition_load_limit(32), 32);
        let third = concurrency.enter();
        let fourth = concurrency.enter();
        assert_eq!(fourth.chunk_size(128, 32), 8);
        assert_eq!(fourth.partition_load_limit(32), 16);
        assert_eq!(fourth.partition_load_limit(1), 2);
        drop((second, third, fourth));
        assert_eq!(first.chunk_size(128, 32), 2);
        assert_eq!(first.partition_load_limit(32), 64);
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
