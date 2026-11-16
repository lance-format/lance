// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cooperative memory admission shared by encoding, I/O, and index stages.

use crate::{Error, Result};

/// An admitted workspace whose allowance may shrink after temporaries are freed.
/// Backends may retain the larger allowance when they cannot resize it. The
/// owner still releases the entire remaining reservation on drop.
pub trait MemoryLease: Send + Sync {
    /// Reduce to `bytes` without acquiring memory. Return `true` if the new
    /// allowance is in effect, or `false` if the original allowance remains.
    /// Growing is invalid and must leave the original reservation unchanged.
    fn try_shrink(&mut self, bytes: usize) -> Result<bool>;
}

struct FixedMemoryLease {
    _owner: Box<dyn Send + Sync>,
    bytes: usize,
}

impl MemoryLease for FixedMemoryLease {
    fn try_shrink(&mut self, bytes: usize) -> Result<bool> {
        if bytes > self.bytes {
            return Err(Error::invalid_input(format!(
                "memory lease cannot grow: requested_bytes={bytes}, reserved_bytes={}",
                self.bytes,
            )));
        }
        Ok(bytes == self.bytes)
    }
}

/// Reserve memory before allocating a buffer or retaining build-owned data.
///
/// The returned owner releases its charge on drop. Keep it with the backing
/// allocation, including detached tasks and every retained slice, until that
/// allocation is freed. Existing caller-owned inputs have a separate boundary;
/// retaining or copying them during an operation must follow that operation's
/// accounting contract. Implementations return contextual resource errors when
/// an allocation cannot fit, without waiting for another stage to release data.
///
/// This interface does not prescribe a pool implementation. Its reservations can
/// share the same pool as DataFusion without making encoding depend on DataFusion.
///
/// ```
/// # use lance_core::{Result, utils::memory::MemoryReserver};
/// fn allocate(pool: &dyn MemoryReserver) -> Result<()> {
///     let owner = pool.reserve("index output", 1024)?;
///     let bytes = vec![0_u8; 1024];
///     drop(bytes);
///     drop(owner);
///     Ok(())
/// }
/// ```
pub trait MemoryReserver: std::fmt::Debug + Send + Sync + 'static {
    /// Completion tracking shared by this operation's CPU and I/O tasks.
    /// Existing implementations retain their original untracked behavior.
    fn task_group(&self) -> Option<std::sync::Arc<super::tasks::ResourceTaskGroup>> {
        None
    }

    /// Known admission ceiling for choosing default cache and tile sizes.
    /// `None` means unknown or unrestricted; every allocation must still call
    /// [`Self::reserve`] before relying on the available allowance.
    fn memory_limit_bytes(&self) -> Option<usize> {
        None
    }

    /// Identify a typed admission denial so a caller can reduce uncommitted work.
    /// Unknown errors return false. Retry only before exposing output or other
    /// side effects, and release the failed attempt's owners before retrying.
    ///
    /// ```
    /// # use lance_core::{Error, utils::memory::MemoryReserver};
    /// fn can_reduce(pool: &dyn MemoryReserver, error: &Error) -> bool {
    ///     pool.is_capacity_error(error)
    /// }
    /// ```
    fn is_capacity_error(&self, _error: &Error) -> bool {
        false
    }

    /// Charge `bytes` to `stage`, returning an owner that releases the charge.
    fn reserve(&self, stage: &str, bytes: usize) -> Result<Box<dyn Send + Sync>>;

    /// Reserve workspace with optional in-place shrinking. The default retains
    /// the full original allowance, preserving existing reserver implementations.
    /// Call shrink only in the async driver after actual CPU work/temporary
    /// allocations have finished; it may access the backend's accounting locks.
    ///
    /// ```
    /// # use lance_core::{Result, utils::memory::MemoryReserver};
    /// fn workspace(pool: &dyn MemoryReserver) -> Result<()> {
    ///     let mut owner = pool.reserve_lease("output workspace", 1024)?;
    ///     let temporary = vec![0_u8; 512];
    ///     let output = vec![1_u8; 512];
    ///     drop(temporary);
    ///     let _reduced = owner.try_shrink(output.capacity())?;
    ///     drop(output);
    ///     drop(owner);
    ///     Ok(())
    /// }
    /// ```
    fn reserve_lease(&self, stage: &str, bytes: usize) -> Result<Box<dyn MemoryLease>> {
        Ok(Box::new(FixedMemoryLease {
            _owner: self.reserve(stage, bytes)?,
            bytes,
        }))
    }
}
