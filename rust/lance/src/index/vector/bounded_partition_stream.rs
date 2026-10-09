// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::task::AtomicWaker;
use futures::{Stream, StreamExt};
use lance_core::{Error, Result};
use tokio::sync::OwnedSemaphorePermit;

/// Work admitted to [`BoundedPartitionStream`].
type WeightedJobStarter<T> = Box<
    dyn FnOnce(AdmissionPermit) -> BoxFuture<'static, Result<(T, AdmissionPermit)>>
        + Send
        + 'static,
>;

pub(super) struct WeightedJob<T> {
    weight_bytes: usize,
    start: WeightedJobStarter<T>,
}

impl<T> WeightedJob<T> {
    #[cfg(test)]
    pub(super) fn new(
        weight_bytes: usize,
        future: impl Future<Output = Result<T>> + Send + 'static,
    ) -> Self {
        Self {
            weight_bytes,
            start: Box::new(move |permit| {
                Box::pin(async move { future.await.map(|value| (value, permit)) })
            }),
        }
    }

    pub(super) fn with_permit<F, Fut>(weight_bytes: usize, start: F) -> Self
    where
        F: FnOnce(AdmissionPermit) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(T, AdmissionPermit)>> + Send + 'static,
    {
        Self {
            weight_bytes,
            start: Box::new(move |permit| Box::pin(start(permit))),
        }
    }
}

/// A completed job whose admission charge remains held until the result is dropped.
///
/// Keeping the permit with a result bounds both active builds and results waiting for
/// an earlier partition to finish.
pub(super) struct Budgeted<T> {
    pub(super) value: T,
    pub(super) permit: Option<Arc<AdmissionPermit>>,
    pub(super) entry_permit: Option<OwnedSemaphorePermit>,
}

impl<T> Budgeted<T> {
    pub(super) fn untracked(value: T) -> Self {
        Self {
            value,
            permit: None,
            entry_permit: None,
        }
    }
}

/// Buffers out-of-order partition results and exposes only the next id to write.
pub(super) struct OrderedPartitionResults<T> {
    next_partition_id: usize,
    num_partitions: usize,
    pending: BTreeMap<usize, T>,
}

impl<T> OrderedPartitionResults<T> {
    pub(super) fn new(num_partitions: usize) -> Self {
        Self {
            next_partition_id: 0,
            num_partitions,
            pending: BTreeMap::new(),
        }
    }

    pub(super) fn push(&mut self, partition_id: usize, value: T) -> Result<()> {
        if partition_id >= self.num_partitions {
            return Err(Error::internal(format!(
                "partition build returned out-of-range partition id {} for {} partitions",
                partition_id, self.num_partitions
            )));
        }
        if partition_id < self.next_partition_id
            || self.pending.insert(partition_id, value).is_some()
        {
            return Err(Error::internal(format!(
                "partition build returned duplicate partition id {}",
                partition_id
            )));
        }
        Ok(())
    }

    pub(super) fn pop_next(&mut self) -> Option<(usize, T)> {
        let partition_id = self.next_partition_id;
        let value = self.pending.remove(&partition_id)?;
        self.next_partition_id += 1;
        Some((partition_id, value))
    }

    pub(super) fn finish(&self) -> Result<()> {
        if self.next_partition_id != self.num_partitions {
            return Err(Error::internal(format!(
                "partition build stream ended before partition {} of {}; buffered partition ids: {:?}",
                self.next_partition_id,
                self.num_partitions,
                self.pending.keys().copied().collect::<Vec<_>>()
            )));
        }
        Ok(())
    }
}

pub(super) struct AdmissionPermit {
    budget: Arc<Budget>,
    charged_bytes: usize,
}

impl AdmissionPermit {
    /// Reconcile the pre-decode admission charge to the materialized size.
    ///
    /// Oversized values remain charged at the cap. Estimates are conservative,
    /// so this normally releases capacity; an underestimate is still reflected
    /// in the budget to prevent admitting additional work against stale usage.
    pub(super) fn reconcile(&mut self, actual_bytes: usize) {
        let charged_bytes = actual_bytes.min(self.budget.max_bytes);
        match charged_bytes.cmp(&self.charged_bytes) {
            std::cmp::Ordering::Less => {
                self.budget
                    .current_bytes
                    .fetch_sub(self.charged_bytes - charged_bytes, Ordering::AcqRel);
            }
            std::cmp::Ordering::Greater => {
                let additional_bytes = charged_bytes - self.charged_bytes;
                let current_bytes = self
                    .budget
                    .current_bytes
                    .fetch_add(additional_bytes, Ordering::AcqRel)
                    + additional_bytes;
                #[cfg(not(test))]
                let _ = current_bytes;
                #[cfg(test)]
                self.budget
                    .peak_bytes
                    .fetch_max(current_bytes, Ordering::AcqRel);
            }
            std::cmp::Ordering::Equal => {}
        }
        self.charged_bytes = charged_bytes;
        self.budget.waker.wake();
    }
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.budget
            .current_bytes
            .fetch_sub(self.charged_bytes, Ordering::AcqRel);
        self.budget.current_entries.fetch_sub(1, Ordering::AcqRel);
        self.budget.waker.wake();
    }
}

struct Budget {
    max_bytes: usize,
    max_entries: usize,
    current_bytes: AtomicUsize,
    current_entries: AtomicUsize,
    waker: AtomicWaker,
    #[cfg(test)]
    peak_bytes: AtomicUsize,
    #[cfg(test)]
    peak_entries: AtomicUsize,
}

impl Budget {
    fn can_admit(&self, weight_bytes: usize) -> bool {
        let current_bytes = self.current_bytes.load(Ordering::Acquire);
        let current_entries = self.current_entries.load(Ordering::Acquire);
        if current_entries >= self.max_entries {
            return false;
        }
        if weight_bytes > self.max_bytes {
            // An oversized (hotspot) partition is charged at the cap and must run
            // alone. It cannot strand the oldest partition behind later work.
            return current_entries == 0;
        }
        current_bytes
            .checked_add(weight_bytes)
            .is_some_and(|total| total <= self.max_bytes)
    }

    fn admit(self: &Arc<Self>, weight_bytes: usize) -> AdmissionPermit {
        let charged_bytes = weight_bytes.min(self.max_bytes);
        let current_bytes = self
            .current_bytes
            .fetch_add(charged_bytes, Ordering::AcqRel)
            + charged_bytes;
        let current_entries = self.current_entries.fetch_add(1, Ordering::AcqRel) + 1;
        #[cfg(not(test))]
        let _ = (current_bytes, current_entries);
        #[cfg(test)]
        {
            self.peak_bytes.fetch_max(current_bytes, Ordering::AcqRel);
            self.peak_entries
                .fetch_max(current_entries, Ordering::AcqRel);
        }
        AdmissionPermit {
            budget: self.clone(),
            charged_bytes,
        }
    }
}

/// Runs partition jobs out of order while bounding active and completed work.
///
/// The input is polled in partition order. A job is admitted only when both its
/// byte charge and the total number of active/completed entries fit. Oversized
/// jobs are charged at the byte cap and admitted only when the budget is empty.
pub(super) struct BoundedPartitionStream<S, T> {
    input: S,
    pending: Option<WeightedJob<T>>,
    in_flight: FuturesUnordered<BoxFuture<'static, Result<Budgeted<T>>>>,
    budget: Arc<Budget>,
    max_concurrency: usize,
    is_input_done: bool,
    is_failed: bool,
}

impl<S, T> BoundedPartitionStream<S, T>
where
    S: Stream<Item = Result<WeightedJob<T>>> + Unpin,
{
    pub(super) fn try_new(
        input: S,
        max_concurrency: usize,
        max_bytes: usize,
        max_entries: usize,
    ) -> Result<Self> {
        if max_concurrency == 0 || max_bytes == 0 || max_entries == 0 {
            return Err(Error::invalid_input(format!(
                "bounded partition stream limits must be non-zero: max_concurrency={}, max_bytes={}, max_entries={}",
                max_concurrency, max_bytes, max_entries
            )));
        }
        Ok(Self {
            input,
            pending: None,
            in_flight: FuturesUnordered::new(),
            budget: Arc::new(Budget {
                max_bytes,
                max_entries,
                current_bytes: AtomicUsize::new(0),
                current_entries: AtomicUsize::new(0),
                waker: AtomicWaker::new(),
                #[cfg(test)]
                peak_bytes: AtomicUsize::new(0),
                #[cfg(test)]
                peak_entries: AtomicUsize::new(0),
            }),
            max_concurrency,
            is_input_done: false,
            is_failed: false,
        })
    }

    fn fail(&mut self) {
        self.is_failed = true;
        self.pending = None;
        self.in_flight = FuturesUnordered::new();
    }

    #[cfg(test)]
    fn stats(&self) -> (usize, usize, usize, usize) {
        (
            self.budget.current_bytes.load(Ordering::Acquire),
            self.budget.peak_bytes.load(Ordering::Acquire),
            self.budget.current_entries.load(Ordering::Acquire),
            self.budget.peak_entries.load(Ordering::Acquire),
        )
    }
}

impl<S, T: 'static> Stream for BoundedPartitionStream<S, T>
where
    S: Stream<Item = Result<WeightedJob<T>>> + Unpin,
{
    type Item = Result<Budgeted<T>>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_failed {
            return Poll::Ready(None);
        }
        self.budget.waker.register(cx.waker());

        loop {
            if self.in_flight.len() >= self.max_concurrency {
                break;
            }

            if let Some(job) = self.pending.take() {
                if !self.budget.can_admit(job.weight_bytes) {
                    self.pending = Some(job);
                    break;
                }
                let permit = self.budget.admit(job.weight_bytes);
                let future = (job.start)(permit);
                self.in_flight.push(Box::pin(async move {
                    future.await.map(|(value, permit)| Budgeted {
                        value,
                        permit: Some(Arc::new(permit)),
                        entry_permit: None,
                    })
                }));
                continue;
            }

            if self.is_input_done {
                break;
            }
            match Pin::new(&mut self.input).poll_next(cx) {
                Poll::Ready(Some(Ok(job))) => self.pending = Some(job),
                Poll::Ready(Some(Err(error))) => {
                    self.fail();
                    return Poll::Ready(Some(Err(error)));
                }
                Poll::Ready(None) => self.is_input_done = true,
                Poll::Pending => break,
            }
        }

        match self.in_flight.poll_next_unpin(cx) {
            Poll::Ready(Some(Ok(output))) => Poll::Ready(Some(Ok(output))),
            Poll::Ready(Some(Err(error))) => {
                self.fail();
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) if self.is_input_done && self.pending.is_none() => Poll::Ready(None),
            Poll::Ready(None) | Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use futures::stream;
    use futures::{StreamExt, TryStreamExt};

    use super::*;

    #[tokio::test]
    async fn slow_head_does_not_block_later_jobs() {
        let jobs = (0..4).map(|partition_id| {
            Ok(WeightedJob::new(1, async move {
                tokio::time::sleep(Duration::from_millis(if partition_id == 0 {
                    40
                } else {
                    1
                }))
                .await;
                Ok(partition_id)
            }))
        });
        let mut output = BoundedPartitionStream::try_new(stream::iter(jobs), 4, 4, 4).unwrap();
        let first = output.next().await.unwrap().unwrap();
        assert_ne!(first.value, 0);
        let mut completed = vec![first.value];
        completed.extend(
            output
                .map(|result| result.unwrap().value)
                .collect::<Vec<_>>()
                .await,
        );
        completed.sort_unstable();
        assert_eq!(completed, vec![0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn byte_and_entry_caps_include_completed_results() {
        let jobs =
            (0..5).map(|partition_id| Ok(WeightedJob::new(3, async move { Ok(partition_id) })));
        let mut output = BoundedPartitionStream::try_new(stream::iter(jobs), 5, 6, 2).unwrap();
        let first = output.next().await.unwrap().unwrap();
        let second = output.next().await.unwrap().unwrap();
        let (_, peak_bytes, current_entries, peak_entries) = output.stats();
        assert_eq!(peak_bytes, 6);
        assert_eq!(current_entries, 2);
        assert_eq!(peak_entries, 2);
        drop((first, second));
        let mut remaining = 0;
        while let Some(result) = output.next().await {
            drop(result.unwrap());
            remaining += 1;
        }
        assert_eq!(remaining, 3);
        let (current_bytes, peak_bytes, current_entries, peak_entries) = output.stats();
        assert_eq!(current_bytes, 0);
        assert_eq!(peak_bytes, 6);
        assert_eq!(current_entries, 0);
        assert_eq!(peak_entries, 2);
    }

    #[tokio::test]
    async fn oversized_job_is_exclusive() {
        let jobs = vec![
            Ok(WeightedJob::new(2, async { Ok(0) })),
            Ok(WeightedJob::new(20, async { Ok(1) })),
            Ok(WeightedJob::new(2, async { Ok(2) })),
        ];
        let mut output = BoundedPartitionStream::try_new(stream::iter(jobs), 3, 8, 3).unwrap();
        let first = output.next().await.unwrap().unwrap();
        assert_eq!(first.value, 0);
        drop(first);
        let oversized = output.next().await.unwrap().unwrap();
        assert_eq!(oversized.value, 1);
        let (current_bytes, peak_bytes, current_entries, _) = output.stats();
        assert_eq!(current_bytes, 8);
        assert_eq!(peak_bytes, 8);
        assert_eq!(current_entries, 1);
        drop(oversized);
        assert_eq!(output.next().await.unwrap().unwrap().value, 2);
    }

    #[tokio::test]
    async fn oversized_materialization_starts_only_after_exclusive_admission() {
        let oversized_materialized = Arc::new(AtomicBool::new(false));
        let marker = oversized_materialized.clone();
        let jobs = vec![
            Ok(WeightedJob::new(2, async { Ok(0) })),
            Ok(WeightedJob::with_permit(
                20,
                move |mut admission| async move {
                    marker.store(true, Ordering::Release);
                    admission.reconcile(20);
                    Ok((1, admission))
                },
            )),
        ];
        let mut output = BoundedPartitionStream::try_new(stream::iter(jobs), 2, 8, 2).unwrap();

        let first = output.next().await.unwrap().unwrap();
        assert_eq!(first.value, 0);
        assert!(!oversized_materialized.load(Ordering::Acquire));

        drop(first);
        let oversized = output.next().await.unwrap().unwrap();
        assert_eq!(oversized.value, 1);
        assert!(oversized_materialized.load(Ordering::Acquire));
        let (current_bytes, peak_bytes, current_entries, _) = output.stats();
        assert_eq!(current_bytes, 8);
        assert_eq!(peak_bytes, 8);
        assert_eq!(current_entries, 1);
    }

    #[tokio::test]
    async fn actual_size_reconciliation_releases_admission_capacity() {
        let jobs = vec![
            Ok(WeightedJob::with_permit(8, |mut admission| async move {
                admission.reconcile(2);
                Ok((0, admission))
            })),
            Ok(WeightedJob::new(6, async { Ok(1) })),
        ];
        let mut output = BoundedPartitionStream::try_new(stream::iter(jobs), 2, 8, 2).unwrap();

        let first = output.next().await.unwrap().unwrap();
        assert_eq!(first.value, 0);
        let (current_bytes, peak_bytes, current_entries, peak_entries) = output.stats();
        assert_eq!(current_bytes, 2);
        assert_eq!(peak_bytes, 8);
        assert_eq!(current_entries, 1);
        assert_eq!(peak_entries, 1);

        let second = output.next().await.unwrap().unwrap();
        assert_eq!(second.value, 1);
        let (current_bytes, _, current_entries, peak_entries) = output.stats();
        assert_eq!(current_bytes, 8);
        assert_eq!(current_entries, 2);
        assert_eq!(peak_entries, 2);
        drop((first, second));
    }

    #[tokio::test]
    async fn expanded_window_results_respect_partition_entry_cap() {
        struct ResidentResult(usize, Arc<AtomicUsize>);

        impl ResidentResult {
            fn new(value: usize, resident: Arc<AtomicUsize>) -> Self {
                resident.fetch_add(1, Ordering::AcqRel);
                Self(value, resident)
            }
        }

        impl Drop for ResidentResult {
            fn drop(&mut self) {
                self.1.fetch_sub(1, Ordering::AcqRel);
            }
        }

        let resident = Arc::new(AtomicUsize::new(0));
        let partition_entries = Arc::new(tokio::sync::Semaphore::new(1));
        let resident_for_job = resident.clone();
        let entries_for_job = partition_entries.clone();
        let jobs = vec![Ok(WeightedJob::with_permit(
            1,
            move |admission| async move {
                let builds = stream::iter((0..8).map(move |value| {
                    let resident = resident_for_job.clone();
                    let partition_entries = entries_for_job.clone();
                    async move {
                        let entry_permit = partition_entries.acquire_owned().await.unwrap();
                        Ok::<_, Error>((ResidentResult::new(value, resident), entry_permit))
                    }
                }))
                .buffer_unordered(8)
                .boxed();
                Ok((builds, admission))
            },
        ))];
        let windows = BoundedPartitionStream::try_new(stream::iter(jobs), 1, 1, 1).unwrap();
        let mut output = windows
            .map_ok(|window| {
                let Budgeted {
                    value: builds,
                    permit,
                    entry_permit,
                } = window;
                assert!(entry_permit.is_none());
                builds.map_ok(move |(value, entry_permit)| Budgeted {
                    value,
                    permit: permit.clone(),
                    entry_permit: Some(entry_permit),
                })
            })
            .try_flatten_unordered(Some(1))
            .boxed();

        let first = output.next().await.unwrap().unwrap();
        assert!(first.value.0 < 8);
        assert_eq!(resident.load(Ordering::Acquire), 1);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), output.next())
                .await
                .is_err()
        );

        drop(first);
        while let Some(result) = output.next().await {
            let result = result.unwrap();
            assert_eq!(resident.load(Ordering::Acquire), 1);
            drop(result);
        }
        assert_eq!(resident.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn dropping_a_held_result_wakes_budget_blocked_stream() {
        let jobs = vec![
            Ok(WeightedJob::new(1, async { Ok(0) })),
            Ok(WeightedJob::new(1, async { Ok(1) })),
        ];
        let mut output = BoundedPartitionStream::try_new(stream::iter(jobs), 2, 1, 1).unwrap();
        let first = output.next().await.unwrap().unwrap();
        let next = output.next();
        tokio::pin!(next);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut next)
                .await
                .is_err()
        );
        drop(first);
        let second = tokio::time::timeout(Duration::from_millis(100), &mut next)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(second.value, 1);
    }

    #[tokio::test]
    async fn error_drops_pending_work() {
        let was_dropped = Arc::new(AtomicBool::new(false));
        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let guard = DropFlag(was_dropped.clone());
        let jobs = vec![
            Ok(WeightedJob::new(1, async {
                Err::<usize, _>(Error::internal("build failed"))
            })),
            Ok(WeightedJob::new(1, async move {
                let _guard = guard;
                futures::future::pending::<()>().await;
                Ok(1)
            })),
        ];
        let mut output = BoundedPartitionStream::try_new(stream::iter(jobs), 2, 2, 2).unwrap();
        let Err(error) = output.next().await.unwrap() else {
            panic!("expected build failure");
        };
        assert!(error.to_string().contains("build failed"));
        assert!(was_dropped.load(Ordering::Acquire));
        assert!(output.next().await.is_none());
    }

    #[tokio::test]
    async fn dropping_stream_cancels_in_flight_work() {
        let was_dropped = Arc::new(AtomicBool::new(false));
        struct DropFlag(Arc<AtomicBool>);
        impl Drop for DropFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let guard = DropFlag(was_dropped.clone());
        let jobs = vec![Ok(WeightedJob::new(1, async move {
            let _guard = guard;
            futures::future::pending::<()>().await;
            Ok(0)
        }))];
        let mut output = BoundedPartitionStream::try_new(stream::iter(jobs), 1, 1, 1).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), output.next())
                .await
                .is_err()
        );
        drop(output);
        assert!(was_dropped.load(Ordering::Acquire));
    }

    #[test]
    fn ordered_results_drain_in_partition_order_and_include_empty_values() {
        let mut results = OrderedPartitionResults::new(4);
        results.push(2, Some(2)).unwrap();
        assert!(results.pop_next().is_none());
        results.push(0, Some(0)).unwrap();
        assert_eq!(results.pop_next(), Some((0, Some(0))));
        results.push(1, None).unwrap();
        assert_eq!(results.pop_next(), Some((1, None)));
        assert_eq!(results.pop_next(), Some((2, Some(2))));
        results.push(3, Some(3)).unwrap();
        assert_eq!(results.pop_next(), Some((3, Some(3))));
        results.finish().unwrap();
    }

    #[test]
    fn ordered_results_reject_duplicate_out_of_range_and_missing() {
        let mut pending_duplicate = OrderedPartitionResults::new(2);
        pending_duplicate.push(1, 1).unwrap();
        let error = pending_duplicate.push(1, 1).unwrap_err();
        assert!(error.to_string().contains("duplicate partition id 1"));

        let mut written_duplicate = OrderedPartitionResults::new(2);
        written_duplicate.push(0, 0).unwrap();
        assert_eq!(written_duplicate.pop_next(), Some((0, 0)));
        let error = written_duplicate.push(0, 0).unwrap_err();
        assert!(error.to_string().contains("duplicate partition id 0"));

        let mut out_of_range = OrderedPartitionResults::new(2);
        let error = out_of_range.push(2, 2).unwrap_err();
        assert!(error.to_string().contains("out-of-range partition id 2"));

        let mut missing = OrderedPartitionResults::new(3);
        missing.push(1, 1).unwrap();
        let error = missing.finish().unwrap_err();
        assert!(error.to_string().contains("ended before partition 0 of 3"));
        assert!(error.to_string().contains("[1]"));
    }
}
