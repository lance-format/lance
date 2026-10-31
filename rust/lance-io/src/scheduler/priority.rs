// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use crate::traits::Reader;

// Reservations live until the response is consumed, not just until I/O finishes.
// Keep the reader alive as well so its identity cannot be reused in that interval.
pub(super) struct PrioritiesInFlight {
    in_flight: Vec<(u128, Arc<dyn Reader>)>,
}

impl PrioritiesInFlight {
    pub(super) fn new(capacity: u32) -> Self {
        Self {
            in_flight: Vec::with_capacity(capacity as usize * 2),
        }
    }

    pub(super) fn min_in_flight(&self) -> u128 {
        self.in_flight.first().map_or(u128::MAX, |(prio, _)| *prio)
    }

    pub(super) fn can_advance(&self, priority: u128, reader: &Arc<dyn Reader>) -> bool {
        let minimum = self.min_in_flight();
        if priority <= minimum {
            return true;
        }
        let reader_minimum = self
            .in_flight
            .iter()
            .find(|(prio, owner)| prio >> 64 == priority >> 64 && Arc::ptr_eq(owner, reader))
            .map(|(prio, _)| *prio);

        // Finish all chunks at an already admitted reader priority, even when
        // an earlier fragment still holds a response.
        if self
            .in_flight
            .iter()
            .any(|(prio, owner)| *prio == priority && Arc::ptr_eq(owner, reader))
        {
            return true;
        }

        // Files in the earliest fragment can depend on different row offsets.
        // Let each file advance independently, but stop at its own first
        // unconsumed response. Later fragments still share the byte budget.
        priority >> 64 == minimum >> 64 && reader_minimum.is_none_or(|min| priority <= min)
    }

    pub(super) fn push(&mut self, priority: u128, reader: Arc<dyn Reader>) {
        let index = self.in_flight.partition_point(|(prio, _)| *prio < priority);
        self.in_flight.insert(index, (priority, reader));
    }

    pub(super) fn remove(&mut self, priority: u128, reader: &Arc<dyn Reader>) {
        if let Some(index) = self
            .in_flight
            .iter()
            .position(|(prio, owner)| *prio == priority && Arc::ptr_eq(owner, reader))
        {
            self.in_flight.remove(index);
        }
    }

    pub(super) fn len(&self) -> usize {
        self.in_flight.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.in_flight.is_empty()
    }
}
