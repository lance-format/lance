// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::atomic::{AtomicU64, Ordering};

use super::{CacheBackendDiagnostics, CacheBackendKind, telemetry};

static NEXT_POOL_ID: AtomicU64 = AtomicU64::new(1);

/// One state per physical backend, independent of typed wrapper lifetimes.
/// Hooks use bounded stack state or atomics; exporter work stays outside locks.
#[derive(Debug)]
pub(super) struct BackendCounters {
    pool_id: u64,
    kind: CacheBackendKind,
    has_size_removal_metrics: bool,
    write_attempts: AtomicU64,
    write_bytes: AtomicU64,
    size_removals: AtomicU64,
    size_removed_bytes: AtomicU64,
    disabled_rejections: AtomicU64,
    disabled_bypasses: AtomicU64,
    lost_placeholders: AtomicU64,
    pub weight_saturations: AtomicU64,
}

impl BackendCounters {
    pub fn new(kind: CacheBackendKind, has_size_removal_metrics: bool) -> Self {
        Self {
            pool_id: NEXT_POOL_ID.fetch_add(1, Ordering::Relaxed),
            kind,
            has_size_removal_metrics,
            write_attempts: AtomicU64::new(0),
            write_bytes: AtomicU64::new(0),
            size_removals: AtomicU64::new(0),
            size_removed_bytes: AtomicU64::new(0),
            disabled_rejections: AtomicU64::new(0),
            disabled_bypasses: AtomicU64::new(0),
            lost_placeholders: AtomicU64::new(0),
            weight_saturations: AtomicU64::new(0),
        }
    }

    pub fn write(&self, bytes: u64) -> u64 {
        let submission_id = self.write_attempts.fetch_add(1, Ordering::Relaxed);
        add_bytes(&self.write_bytes, bytes);
        telemetry::backend_write(self.kind, bytes);
        submission_id
    }

    pub fn size_removal(&self, count: u64, bytes: u64) {
        self.size_removals.fetch_add(count, Ordering::Relaxed);
        add_bytes(&self.size_removed_bytes, bytes);
        telemetry::size_removal(self.kind, count, bytes);
    }

    pub fn disabled_rejection(&self) {
        self.disabled_rejections.fetch_add(1, Ordering::Relaxed);
        telemetry::rejection(self.kind, "disabled");
    }

    pub fn disabled_bypass(&self) {
        self.disabled_bypasses.fetch_add(1, Ordering::Relaxed);
        telemetry::bypass(self.kind, "disabled");
    }

    pub fn lost_placeholder(&self) {
        self.lost_placeholders.fetch_add(1, Ordering::Relaxed);
        telemetry::rejection(self.kind, "lost_placeholder");
    }

    pub fn snapshot(
        &self,
        kind: CacheBackendKind,
        capacity: usize,
        size: usize,
        entries: usize,
    ) -> CacheBackendDiagnostics {
        CacheBackendDiagnostics {
            kind,
            pool_id: Some(self.pool_id),
            capacity_bytes: Some(capacity as u64),
            enabled: Some(capacity > 0),
            size_bytes: Some(size as u64),
            num_entries: Some(entries as u64),
            write_attempts: Some(self.write_attempts.load(Ordering::Relaxed)),
            write_bytes: Some(self.write_bytes.load(Ordering::Relaxed)),
            size_removals: self
                .has_size_removal_metrics
                .then(|| self.size_removals.load(Ordering::Relaxed)),
            size_removed_bytes: self
                .has_size_removal_metrics
                .then(|| self.size_removed_bytes.load(Ordering::Relaxed)),
            disabled_write_rejections: Some(self.disabled_rejections.load(Ordering::Relaxed)),
            disabled_bypasses: Some(self.disabled_bypasses.load(Ordering::Relaxed)),
            lost_placeholder_rejections: (kind == CacheBackendKind::Quick)
                .then(|| self.lost_placeholders.load(Ordering::Relaxed)),
            weight_saturations: Some(self.weight_saturations.load(Ordering::Relaxed)),
            ..Default::default()
        }
    }
}

/// Synthetic declared weights can exhaust a byte counter without allocating
/// that memory. Keep cumulative byte totals monotonic at the representation limit.
fn add_bytes(counter: &AtomicU64, bytes: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(bytes))
    });
}
