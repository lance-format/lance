// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::format::metadata::CompressiveEncoding;
use prost::Message;

/// Container-specific serialized cost applied to every optional payload.
#[derive(Debug, Clone, Copy)]
pub struct BlockCost {
    buffer_overhead_bytes: u64,
    alignment: u64,
}

impl Default for BlockCost {
    fn default() -> Self {
        Self::new(0, 1)
    }
}

impl BlockCost {
    pub(crate) const fn new(buffer_overhead_bytes: u64, alignment: u64) -> Self {
        Self {
            buffer_overhead_bytes,
            alignment: if alignment == 0 { 1 } else { alignment },
        }
    }

    fn payload_bytes(self, has_payload: bool, payload_bytes: u64) -> u64 {
        let aligned = payload_bytes
            .checked_add(self.alignment - 1)
            .map(|padded| padded / self.alignment * self.alignment)
            .unwrap_or(u64::MAX);
        aligned.saturating_add(if has_payload {
            self.buffer_overhead_bytes
        } else {
            0
        })
    }
}

pub fn wire_bytes(
    encoding: &CompressiveEncoding,
    has_payload: bool,
    payload_bytes: &[u64],
    cost: BlockCost,
) -> u64 {
    (encoding.encoded_len() as u64).saturating_add(
        payload_bytes
            .iter()
            .map(|payload_bytes| cost.payload_bytes(has_payload, *payload_bytes))
            .sum::<u64>(),
    )
}
