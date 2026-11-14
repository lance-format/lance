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

use crate::{Error, Result};

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
}

impl AllocationBudget {
    /// Create a budget shared by all readers/operations that use this instance.
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            reserved: AtomicUsize::new(0),
        })
    }

    /// Currently reserved bytes, including unallocated workspace headroom.
    pub fn reserved(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
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
        self.budget.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
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
    use rstest::rstest;

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
