// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_buffer::{Buffer, MutableBuffer};

/// Observes new, exact-capacity allocations made by one IVF partition read.
#[doc(hidden)]
pub struct IvfPartitionCopies {
    copied_bytes: u64,
    #[cfg(feature = "metrics")]
    live: metrics::Gauge,
}

impl IvfPartitionCopies {
    /// Captures the observation switch once for the read.
    pub fn start() -> Option<Self> {
        #[cfg(feature = "metrics")]
        if super::observer::observing() {
            return Some(Self {
                copied_bytes: 0,
                live: metrics::gauge!(super::METRIC_IVF_PARTITION_BACKING),
            });
        }
        None
    }

    /// Transfers the exclusive allocation to an owner shared by every Arrow
    /// clone and slice. An exact buffer keeps its original capacity and pointer.
    pub fn finish(&mut self, values: MutableBuffer) -> Buffer {
        #[cfg(feature = "metrics")]
        if !values.is_empty() && values.len() == values.capacity() {
            let size = values.capacity() as u64;
            self.copied_bytes += size;
            self.live.increment(size as f64);
            return Buffer::from(bytes::Bytes::from_owner(PartitionBuffer {
                values,
                live: self.live.clone(),
            }));
        }
        values.into()
    }

    /// Total bytes materialized through this observation, including buffers
    /// already released. The process gauge reports those still alive.
    pub fn copied_bytes(&self) -> u64 {
        self.copied_bytes
    }
}

#[cfg(feature = "metrics")]
struct PartitionBuffer {
    values: MutableBuffer,
    live: metrics::Gauge,
}

#[cfg(feature = "metrics")]
impl AsRef<[u8]> for PartitionBuffer {
    fn as_ref(&self) -> &[u8] {
        self.values.as_slice()
    }
}

#[cfg(feature = "metrics")]
impl Drop for PartitionBuffer {
    fn drop(&mut self) {
        self.live.decrement(self.values.capacity() as f64);
    }
}
