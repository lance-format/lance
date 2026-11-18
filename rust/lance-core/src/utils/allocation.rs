// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Opt-in admission for owned payload allocations. This is not an RSS limit:
//! allocator overhead, transport buffers and allocations outside this API are excluded.
//! Reserve a complete operation workspace before starting dependent reads. A failed
//! admission returns immediately so the caller can drain/replan instead of waiting
//! while holding the input needed to release the same budget.

use std::alloc::Layout;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_buffer::{MutableBuffer, alloc::ALIGNMENT};
use bytes::Bytes;
use tokio::sync::Notify;

use crate::{Error, Result};

mod writer;
pub use writer::ReservedBufferWriter;

/// A shared ceiling for outstanding payload/workspace reservations.
///
/// ```
/// use lance_core::utils::allocation::AllocationBudget;
/// let budget = AllocationBudget::new(4096);
/// let mut work = budget.try_reserve(4096).unwrap();
/// let bytes = work.try_split(1024).unwrap().allocate().unwrap().freeze();
/// drop(work);
/// assert_eq!(budget.reserved(), 1024);
/// drop(bytes);
/// assert_eq!(budget.reserved(), 0);
/// ```
#[derive(Debug)]
pub struct AllocationBudget {
    limit: usize,
    reserved: AtomicUsize,
    released: Notify,
}

impl AllocationBudget {
    /// Create a budget shared by all readers/operations that use this instance.
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            reserved: AtomicUsize::new(0),
            released: Notify::new(),
        })
    }

    /// Currently reserved bytes, including unallocated workspace headroom.
    pub fn reserved(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }

    /// Wait until at least `bytes` of shared capacity is available, without
    /// reserving it. Rejects requests above this pool's limit. A competing caller
    /// can consume capacity before the next [`Self::try_reserve`], so admission
    /// must still be atomic. This does not promise admission order or fairness.
    ///
    /// Use only in outer scheduling, after releasing a failed operation's inputs
    /// and workspace. Never wait while holding data needed to release that same
    /// capacity. Retained consumer output must have an independent drain path.
    /// This does not prove that a row/codec fits the requested workspace: a decode
    /// refusal with enough available capacity needs replanning, not this wait.
    /// Cancelling the wait owns no credit and cannot consume another waiter's wake.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use lance_core::{Result, utils::allocation::{AllocationBudget, Reservation}};
    /// async fn admit(budget: &Arc<AllocationBudget>, bytes: usize) -> Result<Reservation> {
    ///     // The caller holds no input/output needed to release this capacity.
    ///     loop {
    ///         budget.wait_for_capacity(bytes).await?;
    ///         if let Some(workspace) = budget.try_reserve(bytes) {
    ///             return Ok(workspace);
    ///         }
    ///     }
    /// }
    /// ```
    pub async fn wait_for_capacity(&self, bytes: usize) -> Result<()> {
        if bytes > self.limit {
            return Err(Error::invalid_input(format!(
                "requested capacity {bytes} exceeds allocation budget {}",
                self.limit
            )));
        }
        loop {
            // notify_waiters reaches futures created before the notification,
            // even before their first poll. Subscribe before checking credit so
            // a refund between the check and await cannot strand this caller.
            let released = self.released.notified();
            if self.reserved.load(Ordering::Acquire) <= self.limit - bytes {
                return Ok(());
            }
            released.await;
        }
    }

    /// Atomically admit the entire workspace or return `None` without allocating.
    /// `None` is a scheduling/replanning signal, not a total-result limit or EOF.
    pub fn try_reserve(self: &Arc<Self>, bytes: usize) -> Option<Reservation> {
        self.reserved
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|next| *next <= self.limit)
            })
            .ok()?;
        Some(Reservation {
            budget: self.clone(),
            bytes,
        })
    }

    /// Atomically reserve available credit up to `maximum`, without waiting.
    /// Returns `None` for a zero maximum or an exhausted pool. The caller must
    /// plan against the returned length, which may be smaller than requested.
    /// This never grows a workspace already holding input. Drop a failed attempt
    /// before retrying; it does not guarantee fairness between competing callers.
    ///
    /// ```
    /// use lance_core::utils::allocation::AllocationBudget;
    /// let budget = AllocationBudget::new(4096);
    /// let held = budget.try_reserve(1024).unwrap();
    /// let work = budget.try_reserve_up_to(4096).unwrap();
    /// assert_eq!(work.len(), 3072);
    /// assert!(budget.try_reserve_up_to(1).is_none());
    /// drop((held, work));
    /// assert_eq!(budget.reserved(), 0);
    /// ```
    pub fn try_reserve_up_to(self: &Arc<Self>, maximum: usize) -> Option<Reservation> {
        let previous = self
            .reserved
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                let bytes = self.limit.checked_sub(used)?.min(maximum);
                if bytes == 0 {
                    None
                } else {
                    used.checked_add(bytes)
                }
            })
            .ok()?;
        Some(Reservation {
            budget: self.clone(),
            bytes: (self.limit - previous).min(maximum),
        })
    }
}

/// Exclusive credit for one workspace. Splitting transfers credit without charging twice.
/// There is deliberately no wait/grow operation on a reservation that may hold input.
#[derive(Debug)]
pub struct Reservation {
    budget: Arc<AllocationBudget>,
    bytes: usize,
}

impl Reservation {
    /// Credit still owned by this reservation (excluding previously split children).
    pub fn len(&self) -> usize {
        self.bytes
    }

    /// Whether no credit remains.
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Transfer credit to another allocation/stage, or leave this reservation unchanged.
    pub fn try_split(&mut self, bytes: usize) -> Option<Self> {
        self.bytes = self.bytes.checked_sub(bytes)?;
        Some(Self {
            budget: self.budget.clone(),
            bytes,
        })
    }

    /// Split credit and allocate an Arrow-aligned payload, or return `None` without
    /// changing this workspace. Invalid allocation layouts also leave it unchanged.
    pub fn try_allocate(&mut self, bytes: usize) -> Result<Option<ReservedBuffer>> {
        validate_layout(bytes)?;
        self.try_split(bytes).map(Self::allocate).transpose()
    }

    /// Allocate and initialize exactly this payload layout after admission.
    /// The pointer has Arrow's native alignment, including for an empty buffer.
    /// The charge travels with the buffer, including after conversion to shared `Bytes`.
    pub fn allocate(self) -> Result<ReservedBuffer> {
        validate_layout(self.bytes)?;
        // Unlike MutableBuffer::with_capacity, from_len_zeroed uses an exact-sized
        // allocation layout. Keep it private so no unadmitted reserve/growth is possible.
        let data = MutableBuffer::from_len_zeroed(self.bytes);
        Ok(ReservedBuffer {
            data,
            reservation: self,
        })
    }
}

fn validate_layout(bytes: usize) -> Result<()> {
    Layout::from_size_align(bytes, ALIGNMENT)
        .map(|_| ())
        .map_err(|_| Error::invalid_input("reserved payload exceeds addressable allocation size"))
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.bytes != 0 {
            self.budget.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
            self.budget.released.notify_waiters();
        }
    }
}

/// An admitted payload allocation. Its private fields prevent credit detachment.
#[derive(Debug)]
pub struct ReservedBuffer {
    // Field order matters: release the physical payload before refunding its credit.
    data: MutableBuffer,
    reservation: Reservation,
}

impl ReservedBuffer {
    /// Free an exclusive scratch buffer and return its credit to the same workspace.
    /// No global admission is needed to reuse the returned headroom. On a budget
    /// mismatch the buffer is still consumed/freed, but `workspace` is unchanged.
    pub fn recycle(self, workspace: &mut Reservation) -> Result<()> {
        if !Arc::ptr_eq(&self.reservation.budget, &workspace.budget) {
            return Err(Error::invalid_input(
                "buffer and workspace use different allocation budgets",
            ));
        }
        let recycled = workspace
            .bytes
            .checked_add(self.reservation.bytes)
            .ok_or_else(|| Error::internal("allocation credit overflow"))?;
        let Self {
            data,
            mut reservation,
        } = self;
        drop(data);
        workspace.bytes = recycled;
        reservation.bytes = 0;
        Ok(())
    }

    /// Replace this exclusive buffer only after reserving the entire new allocation.
    /// Both layouts remain charged during the copy. On success, the old allocation's
    /// credit becomes workspace headroom again. On refusal, neither buffer nor credit
    /// changes. Shared/frozen bytes intentionally cannot be resized through this API.
    pub fn try_resize(&mut self, len: usize, workspace: &mut Reservation) -> Result<bool> {
        if !Arc::ptr_eq(&self.reservation.budget, &workspace.budget) {
            return Err(Error::invalid_input(
                "buffer and workspace use different allocation budgets",
            ));
        }
        if len == self.data.len() {
            return Ok(true);
        }
        validate_layout(len)?;
        if len > workspace.bytes {
            return Ok(false);
        }
        let recycled = (workspace.bytes - len)
            .checked_add(self.reservation.bytes)
            .ok_or_else(|| Error::internal("allocation credit overflow"))?;
        let Some(credit) = workspace.try_split(len) else {
            return Ok(false);
        };
        let mut replacement = credit.allocate()?;
        let copy_len = len.min(self.data.len());
        replacement.data.as_slice_mut()[..copy_len].copy_from_slice(&self.data[..copy_len]);
        let Self {
            data,
            mut reservation,
        } = std::mem::replace(self, replacement);
        drop(data);
        workspace.bytes = recycled;
        reservation.bytes = 0;
        Ok(true)
    }

    /// Publish immutable bytes. Clones/slices share this owner and its single charge.
    pub fn freeze(self) -> Bytes {
        Bytes::from_owner(self)
    }

    /// The physical payload layout charged to the shared budget.
    pub fn capacity(&self) -> usize {
        self.reservation.len()
    }
}

impl AsRef<[u8]> for ReservedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.data.as_slice()
    }
}

impl AsMut<[u8]> for ReservedBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.data.as_slice_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use rstest::rstest;

    #[rstest]
    #[case::zero(0, true)]
    #[case::exact(64, true)]
    #[case::too_large(65, false)]
    #[tokio::test]
    async fn capacity_wait_validates_pool_limit(#[case] bytes: usize, #[case] valid: bool) {
        let budget = AllocationBudget::new(64);
        let result = budget.wait_for_capacity(bytes).now_or_never().unwrap();
        if valid {
            result.unwrap();
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("exceeds allocation budget"));
        }
        assert_eq!(budget.reserved(), 0);
    }

    #[rstest]
    #[case::before_poll(false)]
    #[case::after_poll(true)]
    #[tokio::test]
    async fn capacity_wait_observes_last_physical_owner(#[case] poll_first: bool) {
        let budget = AllocationBudget::new(64);
        let bytes = budget.try_reserve(64).unwrap().allocate().unwrap().freeze();
        let alias = bytes.slice(0..1);
        let mut wait = Box::pin(budget.wait_for_capacity(64));
        if poll_first {
            assert!(wait.as_mut().now_or_never().is_none());
        }
        drop(bytes);
        assert_eq!(budget.reserved(), 64);
        if poll_first {
            assert!(wait.as_mut().now_or_never().is_none());
        }
        drop(alias);
        assert_eq!(budget.reserved(), 0);
        wait.now_or_never()
            .expect("final owner must wake admission")
            .unwrap();
        assert_eq!(budget.reserved(), 0, "waiting does not reserve credit");
    }

    #[tokio::test]
    async fn capacity_wait_broadcasts_and_rechecks_after_competing_admission() {
        let budget = AllocationBudget::new(64);
        let mut work = budget.try_reserve(64).unwrap();
        let first = work.try_split(32).unwrap();
        let mut small = Box::pin(budget.wait_for_capacity(32));
        let mut large = Box::pin(budget.wait_for_capacity(64));
        let mut cancelled = Box::pin(budget.wait_for_capacity(32));
        assert!(small.as_mut().now_or_never().is_none());
        assert!(large.as_mut().now_or_never().is_none());
        assert!(cancelled.as_mut().now_or_never().is_none());
        drop(cancelled);
        drop(first);
        small
            .now_or_never()
            .expect("enough for small waiter")
            .unwrap();
        assert!(large.as_mut().now_or_never().is_none());
        drop(work);
        // Another caller can consume the refunded space before a notified future
        // is polled. Notification must not be mistaken for reserved capacity.
        let competitor = budget.try_reserve(64).unwrap();
        assert!(large.as_mut().now_or_never().is_none());
        let mut second = Box::pin(budget.wait_for_capacity(64));
        assert!(second.as_mut().now_or_never().is_none());
        drop(competitor);
        large
            .now_or_never()
            .expect("large waiter must wake")
            .unwrap();
        second
            .now_or_never()
            .expect("all eligible waiters must wake")
            .unwrap();
        assert_eq!(budget.reserved(), 0);
    }

    #[tokio::test]
    async fn recycled_scratch_does_not_release_shared_capacity() {
        let budget = AllocationBudget::new(64);
        let mut work = budget.try_reserve(64).unwrap();
        let scratch = work.try_allocate(64).unwrap().unwrap();
        let mut wait = Box::pin(budget.wait_for_capacity(1));
        assert!(wait.as_mut().now_or_never().is_none());
        scratch.recycle(&mut work).unwrap();
        drop(budget.try_reserve(0).unwrap());
        assert_eq!(budget.reserved(), 64);
        assert!(wait.as_mut().now_or_never().is_none());
        drop(work);
        wait.now_or_never()
            .expect("workspace refund must wake")
            .unwrap();
    }

    #[rstest]
    #[case::zero(0, None)]
    #[case::smaller(17, Some(17))]
    #[case::larger(64, Some(41))]
    #[case::maximum(usize::MAX, Some(41))]
    fn available_workspaces_preserve_shared_owner_credit(
        #[case] maximum: usize,
        #[case] expected: Option<usize>,
    ) {
        let budget = AllocationBudget::new(64);
        let held = budget.try_reserve(23).unwrap().allocate().unwrap().freeze();
        let alias = held.slice(0..1);
        drop(held);
        let work = budget.try_reserve_up_to(maximum);
        assert_eq!(work.as_ref().map(Reservation::len), expected);
        assert_eq!(budget.reserved(), 23 + expected.unwrap_or(0));
        drop(work);
        assert_eq!(budget.reserved(), 23);
        drop(alias);
        assert_eq!(budget.reserved(), 0);
        let all = budget.try_reserve_up_to(64).unwrap();
        assert_eq!(all.len(), 64);
        assert!(budget.try_reserve_up_to(1).is_none());
        drop(all);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn available_workspaces_are_atomic_under_contention() {
        let budget = AllocationBudget::new(10000);
        let start = std::sync::Barrier::new(4);
        let workspaces = std::thread::scope(|scope| {
            let jobs = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        budget.try_reserve_up_to(4096)
                    })
                })
                .collect::<Vec<_>>();
            jobs.into_iter()
                .filter_map(|job| job.join().unwrap())
                .collect::<Vec<_>>()
        });
        let mut sizes = workspaces.iter().map(Reservation::len).collect::<Vec<_>>();
        sizes.sort_unstable();
        assert_eq!(sizes, [1808, 4096, 4096]);
        assert_eq!(budget.reserved(), 10000);
        drop(workspaces);
        assert_eq!(budget.reserved(), 0);
    }

    #[rstest]
    #[case::empty(0)]
    #[case::single_byte(1)]
    #[case::unaligned_length(63)]
    #[case::beyond_cache_line(65)]
    fn admitted_layout_is_exact_and_arrow_aligned(#[case] len: usize) {
        let budget = AllocationBudget::new(len);
        let mut workspace = budget.try_reserve(len).unwrap();
        let output = workspace.try_allocate(len).unwrap().unwrap();
        assert_eq!(output.data.capacity(), len);
        assert_eq!(output.as_ref(), vec![0; len]);
        assert_eq!(output.as_ref().as_ptr().align_offset(ALIGNMENT), 0);
        assert_eq!(budget.reserved(), len);
        assert!(workspace.try_allocate(1).unwrap().is_none());
        let last = output.freeze();
        assert_eq!(last.as_ptr().align_offset(ALIGNMENT), 0);
        drop(last);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn invalid_layout_does_not_consume_workspace() {
        let budget = AllocationBudget::new(8);
        let mut workspace = budget.try_reserve(8).unwrap();
        let error = workspace.try_allocate(usize::MAX).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("addressable allocation size"));
        assert_eq!(workspace.len(), 8);
        assert_eq!(budget.reserved(), 8);
    }

    #[test]
    fn exclusive_scratch_credit_is_reusable_without_new_admission() {
        let budget = AllocationBudget::new(8);
        let mut workspace = budget.try_reserve(8).unwrap();
        for _ in 0..3 {
            let scratch = workspace.try_allocate(8).unwrap().unwrap();
            assert!(budget.try_reserve(1).is_none());
            scratch.recycle(&mut workspace).unwrap();
            assert_eq!(workspace.len(), 8);
            assert_eq!(budget.reserved(), 8);
        }
        let other = AllocationBudget::new(8);
        let foreign = other.try_reserve(8).unwrap().allocate().unwrap();
        let error = foreign.recycle(&mut workspace).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("different allocation budgets"));
        assert_eq!(workspace.len(), 8);
        assert_eq!(other.reserved(), 0);
    }

    #[test]
    fn whole_workspaces_are_atomic_under_contention() {
        let budget = AllocationBudget::new(64);
        let start = std::sync::Barrier::new(8);
        let admitted = std::sync::Barrier::new(8);
        let count = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    start.wait();
                    let workspace = budget.try_reserve(64);
                    if workspace.is_some() {
                        count.fetch_add(1, Ordering::SeqCst);
                    }
                    admitted.wait();
                    drop(workspace);
                });
            }
        });
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(budget.reserved(), 0);
        assert!(budget.try_reserve(64).is_some());
    }

    #[test]
    fn credit_follows_last_shared_owner() {
        let budget = AllocationBudget::new(8192);
        let mut work = budget.try_reserve(8192).unwrap();
        let input = work.try_split(4096).unwrap().allocate().unwrap().freeze();
        let first = input.slice(0..1);
        let last = input.slice(4000..4001);
        drop(input);
        drop(first);
        drop(work);
        assert_eq!(budget.reserved(), 4096);
        assert!(budget.try_reserve(4097).is_none());
        drop(last);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn reserve_overlap_before_reallocation() {
        let budget = AllocationBudget::new(12);
        let mut work = budget.try_reserve(12).unwrap();
        let old = work.try_split(4).unwrap().allocate().unwrap().freeze();
        assert!(work.try_split(9).is_none());
        assert_eq!(work.len(), 8);
        let new = work.try_split(8).unwrap().allocate().unwrap().freeze();
        assert_eq!(budget.reserved(), 12);
        drop(old);
        assert_eq!(budget.reserved(), 8);
        drop(new);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn resize_requires_old_and_new_layouts_to_fit() {
        let budget = AllocationBudget::new(20);
        let mut work = budget.try_reserve(20).unwrap();
        let mut data = work.try_split(8).unwrap().allocate().unwrap();
        data.as_mut().copy_from_slice(b"abcdefgh");
        assert!(!data.try_resize(13, &mut work).unwrap());
        assert_eq!(data.as_ref(), b"abcdefgh");
        assert_eq!(work.len(), 12);
        assert!(data.try_resize(12, &mut work).unwrap());
        assert_eq!(data.as_ref(), b"abcdefgh\0\0\0\0");
        assert_eq!(work.len(), 8);
        assert_eq!(budget.reserved(), 20);
        assert!(data.try_resize(4, &mut work).unwrap());
        assert_eq!(data.as_ref(), b"abcd");
        assert_eq!(work.len(), 16);
        drop(work);
        assert_eq!(budget.reserved(), 4);
        drop(data);
        assert_eq!(budget.reserved(), 0);
    }

    #[test]
    fn overflow_and_failed_admission_do_not_change_credit() {
        let budget = AllocationBudget::new(usize::MAX);
        let first = budget.try_reserve(usize::MAX).unwrap();
        assert!(budget.try_reserve(1).is_none());
        let error = first.allocate().unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("addressable allocation"));
        assert_eq!(budget.reserved(), 0);
        assert!(budget.try_reserve(0).unwrap().is_empty());
    }
}
