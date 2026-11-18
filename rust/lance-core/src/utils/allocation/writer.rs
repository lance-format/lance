// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sequential initialization without a redundant full-payload zeroing pass.

use std::{alloc::Layout, ptr::NonNull};

use arrow_buffer::alloc::ALIGNMENT;
use bytes::Bytes;

use super::Reservation;
use crate::{Error, Result};

#[derive(Debug)]
struct Storage {
    pointer: NonNull<u8>,
    layout: Layout,
}

// SAFETY: Storage owns its allocation. Its private pointer is only accessed
// through the writer's exclusive borrow or the finished owner's immutable view.
unsafe impl Send for Storage {}
// SAFETY: shared access never writes; the writer exposes mutation only via &mut.
unsafe impl Sync for Storage {}

impl Drop for Storage {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            // SAFETY: this pointer was allocated with this exact nonzero layout
            // and has a single owner. No borrowed view can outlive that owner.
            unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
        }
    }
}

impl Reservation {
    /// Allocate an exact Arrow-aligned layout for sequential initialization.
    ///
    /// No bytes are initially readable. The writer can append copied bytes or
    /// zeros and mutate its initialized prefix. It cannot grow or publish an
    /// incomplete buffer. The entire layout remains charged during initialization
    /// and through the final shared alias. Refusal/invalid layout leaves this
    /// workspace unchanged; dropping an incomplete writer frees its allocation.
    ///
    /// ```
    /// use lance_core::utils::allocation::AllocationBudget;
    /// let budget = AllocationBudget::new(5);
    /// let mut work = budget.try_reserve(5).unwrap();
    /// let mut writer = work.try_allocate_writer(5).unwrap().unwrap();
    /// writer.extend_zeros(2).unwrap();
    /// writer.extend_from_slice(b"abc").unwrap();
    /// writer.initialized_mut()[0] = 7;
    /// let bytes = writer.finish().unwrap();
    /// assert_eq!(&bytes[..], &[7, 0, b'a', b'b', b'c']);
    /// drop((bytes, work));
    /// assert_eq!(budget.reserved(), 0);
    /// ```
    pub fn try_allocate_writer(&mut self, bytes: usize) -> Result<Option<ReservedBufferWriter>> {
        let layout = Layout::from_size_align(bytes, ALIGNMENT).map_err(|_| {
            Error::invalid_input("reserved payload exceeds addressable allocation size")
        })?;
        let Some(reservation) = self.try_split(bytes) else {
            return Ok(None);
        };
        let pointer = if bytes == 0 {
            // A non-null aligned dangling pointer is sufficient for empty views.
            NonNull::new(ALIGNMENT as *mut u8)
                .ok_or_else(|| Error::internal("Arrow alignment must be nonzero"))?
        } else {
            // SAFETY: layout is valid and nonzero. No readable slice is created
            // until its bytes have been initialized by an append operation.
            NonNull::new(unsafe { std::alloc::alloc(layout) })
                .unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
        };
        Ok(Some(ReservedBufferWriter {
            storage: Storage { pointer, layout },
            reservation,
            initialized: 0,
        }))
    }
}

/// An admitted allocation whose initialized prefix grows through checked appends.
///
/// Construct with [`Reservation::try_allocate_writer`]. No uninitialized slice or
/// raw pointer is exposed; [`Self::finish`] requires the complete layout to have
/// been initialized. It never reallocates or obtains additional credit.
#[derive(Debug)]
pub struct ReservedBufferWriter {
    // Free the allocation before returning its reservation, including on error.
    storage: Storage,
    reservation: Reservation,
    initialized: usize,
}

impl ReservedBufferWriter {
    fn appended_len(&self, additional: usize) -> Result<usize> {
        self.initialized
            .checked_add(additional)
            .filter(|end| *end <= self.storage.layout.size())
            .ok_or_else(|| Error::invalid_input("write exceeds reserved allocation length"))
    }

    /// Copy bytes into the next uninitialized region. Oversized writes change nothing.
    pub fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<()> {
        let end = self.appended_len(bytes.len())?;
        // SAFETY: the destination is within our exclusive allocation. A safe
        // source slice cannot overlap this mutable borrow. Advance the readable
        // prefix only after every copied byte has been initialized.
        unsafe {
            self.storage
                .pointer
                .as_ptr()
                .add(self.initialized)
                .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
        }
        self.initialized = end;
        Ok(())
    }

    /// Initialize the next bytes to zero. Oversized writes change nothing.
    pub fn extend_zeros(&mut self, bytes: usize) -> Result<()> {
        let end = self.appended_len(bytes)?;
        // SAFETY: the checked destination range lies within our exclusive
        // allocation, and no slice exposes it until initialization completes.
        unsafe {
            self.storage
                .pointer
                .as_ptr()
                .add(self.initialized)
                .write_bytes(0, bytes);
        }
        self.initialized = end;
        Ok(())
    }

    /// Mutate only the already initialized prefix, without changing its length.
    pub fn initialized_mut(&mut self) -> &mut [u8] {
        // SAFETY: only successful appends advance initialized. The pointer is
        // aligned/non-null even for an empty buffer, and &mut ensures exclusivity.
        unsafe { std::slice::from_raw_parts_mut(self.storage.pointer.as_ptr(), self.initialized) }
    }

    /// Publish a fully initialized allocation, or consume/refund an incomplete one.
    pub fn finish(self) -> Result<Bytes> {
        if self.initialized != self.storage.layout.size() {
            return Err(Error::invalid_input(format!(
                "reserved allocation has {} initialized bytes, expected {}",
                self.initialized,
                self.storage.layout.size()
            )));
        }
        Ok(Bytes::from_owner(InitializedStorage {
            storage: self.storage,
            _reservation: self.reservation,
        }))
    }
}

struct InitializedStorage {
    storage: Storage,
    _reservation: Reservation,
}

impl AsRef<[u8]> for InitializedStorage {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: finish proves the entire allocation initialized. This owner
        // exposes no mutable access and remains alive through every Bytes alias.
        unsafe {
            std::slice::from_raw_parts(self.storage.pointer.as_ptr(), self.storage.layout.size())
        }
    }
}

#[cfg(test)]
mod tests;
