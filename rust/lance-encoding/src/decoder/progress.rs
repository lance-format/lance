// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Row accounting and drain planning shared by the decode drivers.

use log::trace;

/// Result of limiting a drain request to one output array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainLimit {
    /// How many requested rows fit the byte-based i32 offset budget.
    pub rows: u64,
    /// The variable-width bytes (at the most constrained i32-offset nesting level)
    /// those rows consume from the caller's budget.
    pub bytes: u64,
    /// An independent batch-level row limit imposed by dictionary key capacity.
    /// Unlike `rows`, this limit does not accumulate across pages.
    /// Composites that change row coordinates must translate this limit rather
    /// than forwarding it.
    pub dictionary_rows: Option<u64>,
}

impl DrainLimit {
    /// Creates a limit that accepts all requested rows without consuming bytes.
    pub(crate) fn unconstrained(num_rows: u64) -> Self {
        Self {
            rows: num_rows,
            bytes: 0,
            dictionary_rows: None,
        }
    }

    /// Combines the limits of sibling columns in the same struct batch.
    ///
    /// Siblings decode into separate Arrow arrays, so each gets the full byte
    /// budget. The struct batch therefore takes the smallest row limit, the
    /// largest byte consumption, and the smallest independent dictionary limit.
    pub(crate) fn combine_sibling(self, sibling: Self) -> Self {
        Self {
            rows: self.rows.min(sibling.rows),
            bytes: self.bytes.max(sibling.bytes),
            dictionary_rows: match (self.dictionary_rows, sibling.dictionary_rows) {
                (Some(current), Some(sibling)) => Some(current.min(sibling)),
                (Some(current), None) => Some(current),
                (None, Some(sibling)) => Some(sibling),
                (None, None) => None,
            },
        }
    }
}

/// The absolute cumulative row range (`start..end`) of one batch.
///
/// `start` is the number of rows drained before this batch and `end` is the
/// cumulative row count that must be scheduled for the full request. Storing the
/// range this way makes the scheduler target explicit and lets batch sizing be
/// derived from the range instead of kept in a parallel counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DrainPlan {
    pub(super) start: u64,
    pub(super) end: u64,
}

impl DrainPlan {
    /// Rows requested before applying scheduler availability or decoder limits.
    pub(super) fn requested_rows(&self) -> u64 {
        self.end - self.start
    }

    /// Returns how many rows in the range the scheduler has made available.
    ///
    /// A scheduler may stop early or report progress beyond the planned end, so
    /// availability is clamped to the range.
    pub(super) fn rows_to_drain(&self, rows_scheduled: u64) -> u64 {
        debug_assert!(
            rows_scheduled >= self.start,
            "scheduler progress must cover rows already drained"
        );
        rows_scheduled.clamp(self.start, self.end) - self.start
    }

    /// Returns the inclusive index of the last row a drain of `rows_to_drain`
    /// rows needs loaded.
    ///
    /// `wait_for_loaded` waits until the loaded row count is strictly greater
    /// than this index.
    pub(super) fn last_row_needed(&self, rows_to_drain: u64) -> u64 {
        debug_assert!(rows_to_drain > 0, "a decode batch must contain rows");
        debug_assert!(
            rows_to_drain <= self.requested_rows(),
            "cannot wait for more rows than the batch requests"
        );
        self.start + rows_to_drain - 1
    }
}

/// Row counters and batch planning shared by every decode driver.
///
/// `rows_scheduled` and `rows_drained` are absolute cumulative counts.
/// `rows_drained <= rows_scheduled <=` the scheduler's final count, and the rows
/// left to decode are `num_rows - rows_drained`. A scheduler may stop early, so
/// batch plans reduce their request to the rows actually scheduled.
#[derive(Debug)]
pub(super) struct DecodeProgress {
    /// Total number of rows this stream will decode. Fusing after an error lowers
    /// this to the rows already drained.
    pub(super) num_rows: u64,
    rows_scheduled: u64,
    pub(super) rows_drained: u64,
}

impl DecodeProgress {
    pub(super) fn new(num_rows: u64) -> Self {
        Self {
            num_rows,
            rows_scheduled: 0,
            rows_drained: 0,
        }
    }

    pub(super) fn rows_scheduled(&self) -> u64 {
        self.rows_scheduled
    }

    pub(super) fn set_rows_scheduled(&mut self, rows_scheduled: u64) {
        debug_assert!(
            rows_scheduled >= self.rows_scheduled,
            "scheduler progress must be monotonic"
        );
        debug_assert!(
            rows_scheduled >= self.rows_drained,
            "scheduler progress must cover rows already drained"
        );
        self.rows_scheduled = rows_scheduled;
    }

    /// Plans the next batch, clamped to the rows left and the caller's batch bound.
    ///
    /// `max_rows` is the driver's row limit, already tightened by any byte-size
    /// estimate. Returns `None` when no rows remain. The scheduler target is
    /// absolute, so later batches continue from the rows already drained.
    pub(super) fn plan_batch(&self, max_rows: u64) -> Option<DrainPlan> {
        debug_assert!(
            self.rows_drained <= self.num_rows,
            "cannot drain more rows than the stream contains"
        );
        let rows_remaining = self.num_rows - self.rows_drained;
        trace!(
            "Draining batch task (rows_remaining={} rows_drained={} rows_scheduled={})",
            rows_remaining, self.rows_drained, self.rows_scheduled,
        );
        let requested_rows = rows_remaining.min(max_rows);
        if requested_rows == 0 {
            return None;
        }

        let plan = DrainPlan {
            start: self.rows_drained,
            end: self.rows_drained + requested_rows,
        };
        let scheduled_need = plan.end.saturating_sub(self.rows_scheduled);
        trace!(
            "scheduled_need = {} because rows_drained = {} and to_take = {} and rows_scheduled = {}",
            scheduled_need, self.rows_drained, requested_rows, self.rows_scheduled
        );
        if scheduled_need > 0 {
            trace!(
                "Draining from scheduler (desire at least {} scheduled rows)",
                plan.end
            );
        }
        Some(plan)
    }

    /// Applies the decoder drain limit and records the reserved rows as drained.
    ///
    /// This is the only place `rows_drained` advances, so a reserved batch cannot
    /// be decoded without also being recorded. A batch always advances by at
    /// least one row: a first row whose (possibly over-estimated) size exceeds the
    /// whole budget, or whose nested child cannot fit a declared dictionary key
    /// type, is emitted alone. A genuinely oversized value or dictionary then
    /// surfaces the same decoding error it always did, while an over-estimated
    /// value decodes fine.
    pub(super) fn reserve_drain(&mut self, limit: DrainLimit, requested_rows: u64) -> u64 {
        if requested_rows == 0 {
            return 0;
        }
        let planned_rows = limit
            .dictionary_rows
            .map(|dictionary_rows| limit.rows.min(dictionary_rows))
            .unwrap_or(limit.rows);
        let rows = planned_rows.clamp(1, requested_rows);
        debug_assert!(self.rows_drained + rows <= self.num_rows);
        debug_assert!(self.rows_drained + rows <= self.rows_scheduled);
        self.rows_drained += rows;
        rows
    }

    /// Stops future batch planning after an error.
    ///
    /// A failed drain already fuses the driver, so treating the reserved rows as
    /// drained does not change what the caller observes.
    pub(super) fn fuse_after_error(&mut self) {
        self.num_rows = self.rows_drained;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drain_limit_combines_sibling_constraints() {
        let combined = DrainLimit {
            rows: 8,
            bytes: 128,
            dictionary_rows: Some(6),
        }
        .combine_sibling(DrainLimit {
            rows: 5,
            bytes: 256,
            dictionary_rows: Some(7),
        });

        assert_eq!(
            combined,
            DrainLimit {
                rows: 5,
                bytes: 256,
                dictionary_rows: Some(6),
            }
        );
    }

    #[test]
    fn test_plan_batch_clamps_to_rows_remaining() {
        let plan = DecodeProgress::new(3).plan_batch(8).unwrap();
        assert_eq!((plan.start, plan.end), (0, 3));
        assert_eq!(plan.requested_rows(), 3);

        assert!(DecodeProgress::new(0).plan_batch(8).is_none());
    }

    #[test]
    fn test_drain_plan_accounts_for_scheduler_shortfall_and_overshoot() {
        let mut progress = DecodeProgress::new(10);
        progress.set_rows_scheduled(2);

        let plan = progress.plan_batch(4).unwrap();
        assert_eq!((plan.start, plan.end), (0, 4));
        assert_eq!(plan.requested_rows(), 4);
        assert_eq!(plan.rows_to_drain(0), 0);
        assert_eq!(plan.rows_to_drain(2), 2);
        assert_eq!(plan.rows_to_drain(3), 3);
        assert_eq!(plan.rows_to_drain(4), 4);
        // Scheduler progress beyond the planned end must not drain extra rows.
        assert_eq!(plan.rows_to_drain(6), 4);

        let rows_to_drain = plan.rows_to_drain(3);
        progress.set_rows_scheduled(3);
        let reserved =
            progress.reserve_drain(DrainLimit::unconstrained(rows_to_drain), rows_to_drain);
        assert_eq!(reserved, 3);
        assert_eq!(progress.num_rows - progress.rows_drained, 7);
        assert_eq!(progress.rows_drained, 3);
    }

    #[test]
    fn test_drain_plan_clamps_to_planned_range() {
        let mut progress = DecodeProgress::new(10);
        progress.set_rows_scheduled(4);
        assert_eq!(progress.reserve_drain(DrainLimit::unconstrained(2), 2), 2);

        let plan = progress.plan_batch(4).unwrap();
        assert_eq!((plan.start, plan.end), (2, 6));
        assert_eq!(plan.rows_to_drain(2), 0);
        assert_eq!(plan.rows_to_drain(4), 2);
        assert_eq!(plan.rows_to_drain(6), 4);
        assert_eq!(plan.last_row_needed(2), 3);
    }

    #[test]
    fn test_fuse_after_error_keeps_drained_rows_and_stops_planning() {
        let mut progress = DecodeProgress::new(10);
        progress.set_rows_scheduled(10);
        assert_eq!(progress.reserve_drain(DrainLimit::unconstrained(4), 4), 4);

        progress.fuse_after_error();

        assert_eq!(progress.num_rows, 4);
        assert_eq!(progress.rows_drained, 4);
        assert!(progress.plan_batch(4).is_none());
    }

    #[rstest::rstest]
    #[case::over_budget(0, None, 4, 1)]
    #[case::dictionary_capacity(4, Some(2), 4, 2)]
    #[case::oversized_dictionary(4, Some(0), 4, 1)]
    #[case::request_limit(8, None, 4, 4)]
    #[case::empty_request(0, Some(0), 0, 0)]
    fn test_reserve_drain_constraints(
        #[case] rows: u64,
        #[case] dictionary_rows: Option<u64>,
        #[case] requested_rows: u64,
        #[case] expected_rows: u64,
    ) {
        let mut progress = DecodeProgress::new(10);
        progress.set_rows_scheduled(10);
        let limit = DrainLimit {
            rows,
            bytes: 0,
            dictionary_rows,
        };

        let reserved = progress.reserve_drain(limit, requested_rows);
        assert_eq!(reserved, expected_rows);
        assert_eq!(
            progress.num_rows - progress.rows_drained,
            10 - expected_rows
        );
        assert_eq!(progress.rows_drained, expected_rows);
    }
}
