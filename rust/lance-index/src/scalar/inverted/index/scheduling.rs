// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Poll;

use futures::{Stream, TryStreamExt};
use lance_core::{Result, utils::tokio::spawn_cpu};

use super::format::{DEFAULT_FTS_SEARCH_CHUNK, configured_fts_search_chunk};

// A large partition can retain much more position data than a small one. Bound
// the data queued with each CPU task independently of the partition count.
// One indivisible partition may exceed this target and is scored alone.
const SCORING_BATCH_BYTES: usize = 16 * 1024 * 1024;

fn ready_scoring_chunks<T, S, W>(
    loads: S,
    max_items: usize,
    max_bytes: usize,
    weight: W,
) -> impl Stream<Item = Result<Vec<T>>>
where
    S: Stream<Item = Result<T>>,
    W: Fn(&T) -> usize,
{
    debug_assert!(
        max_items > 0 && max_bytes > 0,
        "scoring chunk item and byte targets must be positive"
    );
    let mut loads = Box::pin(loads);
    let mut batch = Vec::new();
    let mut batch_bytes = 0usize;
    let mut pending = None;
    let mut finished = false;
    futures::stream::poll_fn(move |cx| {
        if finished {
            return Poll::Ready(None);
        }
        loop {
            let (item, bytes) = match pending.take() {
                Some(item) => item,
                None => match loads.as_mut().poll_next(cx) {
                    Poll::Ready(Some(Ok(item))) => {
                        let bytes = weight(&item);
                        (item, bytes)
                    }
                    Poll::Ready(Some(Err(error))) => {
                        finished = true;
                        batch.clear();
                        return Poll::Ready(Some(Err(error)));
                    }
                    Poll::Ready(None) => {
                        finished = true;
                        return Poll::Ready(
                            (!batch.is_empty()).then(|| Ok(std::mem::take(&mut batch))),
                        );
                    }
                    Poll::Pending if batch.is_empty() => return Poll::Pending,
                    Poll::Pending => {
                        batch_bytes = 0;
                        return Poll::Ready(Some(Ok(std::mem::take(&mut batch))));
                    }
                },
            };
            if !batch.is_empty() && batch_bytes.saturating_add(bytes) > max_bytes {
                // Retain only the next item, rather than materializing a whole
                // count-based chunk before splitting it into smaller tasks.
                pending = Some((item, bytes));
                batch_bytes = 0;
                return Poll::Ready(Some(Ok(std::mem::take(&mut batch))));
            }
            batch_bytes = batch_bytes.saturating_add(bytes);
            batch.push(item);
            if batch.len() >= max_items || batch_bytes >= max_bytes {
                batch_bytes = 0;
                return Poll::Ready(Some(Ok(std::mem::take(&mut batch))));
            }
        }
    })
}

/// Consume ready inputs before unrelated reads finish. The caller retains its
/// original bounded I/O window; at most one CPU batch and one pending input
/// are retained here. The byte target bounds queued cache references, not RSS
/// or the upstream decoder's working memory.
pub(super) async fn score_ready_partitions<T, R, S, W, F>(
    loads: S,
    chunk_size: usize,
    weight: W,
    score: F,
) -> Result<Vec<R>>
where
    T: Send + 'static,
    R: Send + 'static,
    S: Stream<Item = Result<T>>,
    W: Fn(&T) -> usize,
    F: Fn(Vec<T>) -> Result<Vec<R>> + Clone + Send + 'static,
{
    ready_scoring_chunks(loads, chunk_size, SCORING_BATCH_BYTES, weight)
        .map_ok(move |loaded| {
            let score = score.clone();
            async move { spawn_cpu(move || score(loaded)).await }
        })
        // Each original window already supplies concurrent I/O, and concurrent
        // windows and searches supply CPU work. Queuing another batch here pins
        // loaded position streams while the shared CPU pool is saturated.
        .try_buffer_unordered(1)
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

    #[tokio::test]
    async fn ready_partition_releases_before_pending_load() {
        let released = std::sync::Arc::new(AtomicUsize::new(0));
        let progress = released.clone();
        let loads = futures::stream::once(async { Ok(7) }).chain(futures::stream::once(async {
            // The next read cannot acquire its resource until the first
            // result has been consumed, even though both fit one CPU batch.
            while progress.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
            Ok(11)
        }));
        let mut result = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            score_ready_partitions(
                loads,
                16,
                |_| 1,
                move |items| {
                    released.fetch_add(items.len(), Ordering::Release);
                    Ok(items)
                },
            ),
        )
        .await
        .expect("ready work must not wait for the whole I/O window")
        .unwrap();
        result.sort_unstable();
        assert_eq!(result, vec![7, 11]);
    }

    #[tokio::test]
    async fn ready_scoring_bounds_retained_batches() {
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
        let mut result = score_ready_partitions(
            loads,
            4,
            |_| 1,
            |items| {
                assert!(items.len() <= 4);
                Ok(items.into_iter().map(|item| item.1).collect())
            },
        )
        .await
        .unwrap();
        result.sort_unstable();
        assert_eq!(result, (0..33).collect::<Vec<_>>());
        assert!(peak.load(Ordering::Relaxed) <= 4);
        assert_eq!(live.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn ready_scoring_limits_bytes_without_splitting_partitions() {
        let mib = 1024 * 1024;
        let weights = [12 * mib, 8 * mib, 8 * mib, 20 * mib, usize::MAX, 0];
        let loads = futures::stream::iter(weights.into_iter().enumerate().map(Ok));
        let mut result = score_ready_partitions(
            loads,
            16,
            |item| item.1,
            |items| {
                let bytes = items.iter().map(|item| item.1).sum::<usize>();
                assert!(bytes <= SCORING_BATCH_BYTES || items.len() == 1);
                Ok(items.into_iter().map(|item| item.0).collect())
            },
        )
        .await
        .unwrap();
        result.sort_unstable();
        assert_eq!(result, (0..weights.len()).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn ready_scoring_limits_zero_byte_item_count() {
        let batches = ready_scoring_chunks(
            futures::stream::iter((0..7).map(Ok)),
            2,
            SCORING_BATCH_BYTES,
            |_| 0,
        )
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
        assert!(batches.iter().all(|batch| batch.len() <= 2));
        assert_eq!(batches.concat(), (0..7).collect::<Vec<_>>());
    }

    struct WeightedInput {
        live: std::sync::Arc<AtomicUsize>,
        bytes: usize,
        id: usize,
    }

    impl Drop for WeightedInput {
        fn drop(&mut self) {
            self.live.fetch_sub(self.bytes, Ordering::Relaxed);
        }
    }

    #[tokio::test]
    async fn ready_scoring_releases_weighted_inputs() {
        let live = std::sync::Arc::new(AtomicUsize::new(0));
        let peak = std::sync::Arc::new(AtomicUsize::new(0));
        let bytes = 12 * 1024 * 1024;
        let loads = futures::stream::iter(0..17).map(|id| {
            let retained = live.fetch_add(bytes, Ordering::Relaxed) + bytes;
            peak.fetch_max(retained, Ordering::Relaxed);
            Ok(WeightedInput {
                live: live.clone(),
                bytes,
                id,
            })
        });
        let mut result = score_ready_partitions(
            loads,
            16,
            |item| item.bytes,
            |items| Ok(items.into_iter().map(|item| item.id).collect()),
        )
        .await
        .unwrap();
        result.sort_unstable();
        assert_eq!(result, (0..17).collect::<Vec<_>>());
        // One submitted batch plus the next partition that did not fit.
        assert!(peak.load(Ordering::Relaxed) <= 2 * bytes);
        assert_eq!(live.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn ready_scoring_drop_releases_pending_input() {
        let live = std::sync::Arc::new(AtomicUsize::new(0));
        let bytes = 12 * 1024 * 1024;
        let loads = futures::stream::iter(0..3).map(|id| {
            live.fetch_add(bytes, Ordering::Relaxed);
            Ok(WeightedInput {
                live: live.clone(),
                bytes,
                id,
            })
        });
        let mut batches = Box::pin(ready_scoring_chunks(
            loads,
            16,
            SCORING_BATCH_BYTES,
            |item| item.bytes,
        ));
        let first = batches.next().await.unwrap().unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(live.load(Ordering::Relaxed), 2 * bytes);
        drop(first);
        drop(batches);
        assert_eq!(live.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn ready_scoring_error_releases_unsubmitted_input() {
        let live = std::sync::Arc::new(AtomicUsize::new(12));
        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let scored = calls.clone();
        let loads = futures::stream::iter([
            Ok(WeightedInput {
                live: live.clone(),
                bytes: 12,
                id: 0,
            }),
            Err(lance_core::Error::index("load failed")),
        ]);
        let error = score_ready_partitions(
            loads,
            16,
            |item| item.bytes,
            move |items| {
                scored.fetch_add(1, Ordering::Relaxed);
                Ok(items.into_iter().map(|item| item.id).collect())
            },
        )
        .await
        .unwrap_err();
        assert!(matches!(&error, lance_core::Error::Index { .. }));
        assert!(error.to_string().contains("load failed"));
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert_eq!(live.load(Ordering::Relaxed), 0);
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
