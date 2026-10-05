// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Optional `metrics` facade export for cache diagnostics.
#[cfg(feature = "metrics")]
use super::MAX_CACHE_TYPE_SERIES;
use super::{CacheBackend, CacheBackendKind, CacheMetricsKind};

#[cfg(feature = "metrics")]
use metrics::{Key, Label};

/// Cache lookup outcomes, labelled by `cache`, `backend`, and `outcome`.
pub const METRIC_LOOKUPS: &str = "lance_cache_lookups_total";
/// Cache lookup failures, labelled by `cache`, `backend`, and `reason`.
pub const METRIC_LOOKUP_ERRORS: &str = "lance_cache_lookup_errors_total";
/// Physical backend write submissions, labelled by `backend`.
pub const METRIC_WRITE_ATTEMPTS: &str = "lance_cache_write_attempts_total";
/// Accounted bytes in physical backend write submissions, labelled by `backend`.
pub const METRIC_WRITE_BYTES: &str = "lance_cache_write_bytes_total";
/// Physical backend size-removal notifications, labelled by `backend`.
pub const METRIC_SIZE_REMOVALS: &str = "lance_cache_size_removals_total";
/// Accounted bytes in size-removal notifications, labelled by `backend`.
pub const METRIC_SIZE_REMOVED_BYTES: &str = "lance_cache_size_removed_bytes_total";
/// Positively identified rejected writes, labelled by `backend` and `reason`.
pub const METRIC_WRITE_REJECTIONS: &str = "lance_cache_write_rejections_total";
/// Cache insertion bypasses, labelled by `backend` and `reason`.
pub const METRIC_BYPASSES: &str = "lance_cache_bypasses_total";
/// Reserved for backends that can positively identify shared-load results.
/// The built-in backends currently do not export samples for this metric.
pub const METRIC_COALESCED_LOADS: &str = "lance_cache_coalesced_loads_total";
/// Explicit prewarm cache calls, labelled by `cache` and `backend`.
pub const METRIC_WARM_ATTEMPTS: &str = "lance_cache_warm_attempts_total";
/// Explicit prewarm calls served without a new materialization.
pub const METRIC_WARM_HITS: &str = "lance_cache_warm_hits_total";
/// Explicit prewarm materialization outcomes.
pub const METRIC_WARM_LOADS: &str = "lance_cache_warm_loads_total";
/// Accounted bytes successfully materialized by explicit prewarming.
pub const METRIC_WARM_LOAD_BYTES: &str = "lance_cache_warm_load_bytes_total";
/// Failed explicit prewarm cache calls.
pub const METRIC_WARM_ERRORS: &str = "lance_cache_warm_errors_total";
/// Executed loader outcomes, labelled by `cache`, `backend`, and `outcome`.
pub const METRIC_LOADS: &str = "lance_cache_loads_total";
/// Executing loader futures, labelled by `cache` and `backend`.
pub const METRIC_LOADS_IN_FLIGHT: &str = "lance_cache_loads_in_flight";
/// Executed loader duration in seconds, labelled by `cache`, `backend`, and `outcome`.
pub const METRIC_LOAD_DURATION: &str = "lance_cache_load_duration_seconds";
/// Accounted submitted entry size in bytes, labelled by `backend`.
pub const METRIC_ENTRY_SIZE: &str = "lance_cache_entry_size_bytes";
/// Configured capacity of distinct live pools, summed by `backend`.
pub const METRIC_CAPACITY: &str = "lance_cache_capacity_bytes";
/// Approximate occupancy of distinct live pools, summed by `backend`.
pub const METRIC_SIZE: &str = "lance_cache_size_bytes";
/// Approximate entry count of distinct live pools, summed by `backend`.
pub const METRIC_ENTRIES: &str = "lance_cache_entries";
/// Activity events assigned to the bounded `other` type label.
pub const METRIC_TYPE_OVERFLOW: &str = "lance_cache_type_overflow_total";

/// Recommended fixed buckets for executed loader duration, in seconds.
pub const LOAD_DURATION_BOUNDS: &[f64] = &[
    0.000_01, 0.000_05, 0.000_1, 0.000_5, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0,
    60.0,
];

/// Recommended fixed buckets for accounted submitted entry sizes, in bytes.
pub const ENTRY_SIZE_BOUNDS: &[f64] = &[
    64.0,
    256.0,
    1_024.0,
    4_096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    4_194_304.0,
    16_777_216.0,
    67_108_864.0,
    268_435_456.0,
    1_073_741_824.0,
];

/// Recommended fixed histogram boundaries as `(metric_name, boundaries)`.
pub fn histogram_bounds() -> &'static [(&'static str, &'static [f64])] {
    &[
        (METRIC_LOAD_DURATION, LOAD_DURATION_BOUNDS),
        (METRIC_ENTRY_SIZE, ENTRY_SIZE_BOUNDS),
    ]
}

/// Register descriptions for every cache metric with the installed recorder.
///
/// Call this after installing a recorder and before enumerating its catalog.
#[cfg(feature = "metrics")]
pub fn describe_metrics() {
    metrics::describe_counter!(
        METRIC_LOOKUPS,
        metrics::Unit::Count,
        "Cache lookup outcomes."
    );
    metrics::describe_counter!(
        METRIC_LOOKUP_ERRORS,
        metrics::Unit::Count,
        "Cache lookup failures by known reason."
    );
    metrics::describe_counter!(
        METRIC_WRITE_ATTEMPTS,
        metrics::Unit::Count,
        "Physical cache backend write submissions."
    );
    metrics::describe_counter!(
        METRIC_WRITE_BYTES,
        metrics::Unit::Bytes,
        "Accounted bytes submitted to physical cache backends."
    );
    metrics::describe_counter!(
        METRIC_SIZE_REMOVALS,
        metrics::Unit::Count,
        "Physical cache backend size-removal notifications."
    );
    metrics::describe_counter!(
        METRIC_SIZE_REMOVED_BYTES,
        metrics::Unit::Bytes,
        "Accounted bytes in physical cache backend size-removal notifications."
    );
    metrics::describe_counter!(
        METRIC_WRITE_REJECTIONS,
        metrics::Unit::Count,
        "Positively identified physical cache backend write rejections."
    );
    metrics::describe_counter!(
        METRIC_BYPASSES,
        metrics::Unit::Count,
        "Cache insertions bypassed for a known reason."
    );
    metrics::describe_counter!(
        METRIC_COALESCED_LOADS,
        metrics::Unit::Count,
        "Cache calls that received another concurrent loader's result."
    );
    metrics::describe_counter!(
        METRIC_WARM_ATTEMPTS,
        metrics::Unit::Count,
        "Cache calls attributed to explicit prewarming."
    );
    metrics::describe_counter!(
        METRIC_WARM_HITS,
        metrics::Unit::Count,
        "Explicit prewarm calls served without a new materialization."
    );
    metrics::describe_counter!(
        METRIC_WARM_LOADS,
        metrics::Unit::Count,
        "Explicit prewarm materialization outcomes."
    );
    metrics::describe_counter!(
        METRIC_WARM_LOAD_BYTES,
        metrics::Unit::Bytes,
        "Accounted bytes successfully materialized by explicit prewarming."
    );
    metrics::describe_counter!(
        METRIC_WARM_ERRORS,
        metrics::Unit::Count,
        "Failed explicit prewarm cache calls."
    );
    metrics::describe_counter!(
        METRIC_LOADS,
        metrics::Unit::Count,
        "Executed cache loader outcomes."
    );
    metrics::describe_gauge!(
        METRIC_LOADS_IN_FLIGHT,
        metrics::Unit::Count,
        "Cache loader futures currently executing."
    );
    metrics::describe_histogram!(
        METRIC_LOAD_DURATION,
        metrics::Unit::Seconds,
        "Executed cache loader duration in seconds."
    );
    metrics::describe_histogram!(
        METRIC_ENTRY_SIZE,
        metrics::Unit::Bytes,
        "Accounted size of entries submitted to physical cache backends."
    );
    metrics::describe_gauge!(
        METRIC_CAPACITY,
        metrics::Unit::Bytes,
        "Configured capacity of distinct live cache pools."
    );
    metrics::describe_gauge!(
        METRIC_SIZE,
        metrics::Unit::Bytes,
        "Approximate accounted occupancy of distinct live cache pools."
    );
    metrics::describe_gauge!(
        METRIC_ENTRIES,
        metrics::Unit::Count,
        "Approximate entry count of distinct live cache pools."
    );
    metrics::describe_counter!(
        METRIC_TYPE_OVERFLOW,
        metrics::Unit::Count,
        "Cache activity events assigned to the bounded other type label."
    );
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn register_backend(_backend: &std::sync::Arc<dyn CacheBackend>) {}

#[cfg(feature = "metrics")]
mod enabled {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicU16, Ordering};
    use std::sync::{Arc, LazyLock, Mutex, Weak};

    use super::*;

    static BACKENDS: LazyLock<Mutex<Vec<Weak<dyn CacheBackend>>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));
    static TYPE_LABELS: LazyLock<Mutex<Vec<&'static str>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));
    // Three metrics for each of three backend kinds.
    static SEEN_GAUGES: AtomicU16 = AtomicU16::new(0);

    fn kind_index(kind: CacheBackendKind) -> usize {
        match kind {
            CacheBackendKind::Quick => 0,
            CacheBackendKind::Moka => 1,
            CacheBackendKind::Custom => 2,
        }
    }

    pub(super) fn register_backend(backend: &Arc<dyn CacheBackend>) {
        let backend = Arc::downgrade(backend);
        let mut backends = BACKENDS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        backends.retain(|registered| registered.strong_count() > 0);
        if !backends
            .iter()
            .any(|registered| Weak::ptr_eq(registered, &backend))
        {
            backends.push(backend);
        }
    }

    pub(super) fn register_type_label(type_name: &'static str) -> (&'static str, bool) {
        let mut labels = TYPE_LABELS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if labels.contains(&type_name) {
            return (type_name, false);
        }
        if labels.len() < MAX_CACHE_TYPE_SERIES {
            labels.push(type_name);
            (type_name, false)
        } else {
            ("other", true)
        }
    }

    #[derive(Default)]
    struct GaugeGroup {
        capacity: Option<u64>,
        size: Option<u64>,
        entries: Option<u64>,
    }

    fn add(target: &mut Option<u64>, value: Option<u64>) {
        if let Some(value) = value {
            *target = Some(target.unwrap_or_default().saturating_add(value));
        }
    }

    fn publish(
        kind: CacheBackendKind,
        metric_index: usize,
        name: &'static str,
        value: Option<u64>,
    ) {
        let bit = 1 << (metric_index * 3 + kind_index(kind));
        if let Some(value) = value {
            SEEN_GAUGES.fetch_or(bit, Ordering::Relaxed);
            metrics::gauge!(name, "backend" => kind.as_str()).set(value as f64);
        } else if SEEN_GAUGES.load(Ordering::Relaxed) & bit != 0 {
            metrics::gauge!(name, "backend" => kind.as_str()).set(0.0);
        }
    }

    pub(super) fn refresh_metrics() {
        let live = {
            let mut backends = BACKENDS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut live = Vec::with_capacity(backends.len());
            backends.retain(|backend| {
                if let Some(backend) = backend.upgrade() {
                    live.push(backend);
                    true
                } else {
                    false
                }
            });
            live
        };

        let mut pool_ids = HashSet::with_capacity(live.len());
        let mut groups: [GaugeGroup; 3] = Default::default();
        for backend in live {
            let diagnostics = backend.diagnostics();
            let Some(pool_id) = diagnostics.pool_id else {
                // Without physical identity, multiple wrappers cannot be
                // deduplicated safely.
                continue;
            };
            if !pool_ids.insert(pool_id) {
                continue;
            }
            let group = &mut groups[kind_index(diagnostics.kind)];
            add(&mut group.capacity, diagnostics.capacity_bytes);
            add(&mut group.size, diagnostics.size_bytes);
            add(&mut group.entries, diagnostics.num_entries);
        }

        for kind in [
            CacheBackendKind::Quick,
            CacheBackendKind::Moka,
            CacheBackendKind::Custom,
        ] {
            let group = &groups[kind_index(kind)];
            publish(kind, 0, METRIC_CAPACITY, group.capacity);
            publish(kind, 1, METRIC_SIZE, group.size);
            publish(kind, 2, METRIC_ENTRIES, group.entries);
        }
    }
}

#[cfg(feature = "metrics")]
pub(crate) fn register_backend(backend: &std::sync::Arc<dyn CacheBackend>) {
    enabled::register_backend(backend);
}

#[cfg(feature = "metrics")]
pub(super) fn register_type_label(type_name: &'static str) -> (&'static str, bool) {
    enabled::register_type_label(type_name)
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn type_overflow(cache: CacheMetricsKind, backend: CacheBackendKind) {
    metrics::counter!(
        METRIC_TYPE_OVERFLOW,
        "cache" => cache.as_str(),
        "backend" => backend.as_str()
    )
    .increment(1);
}

/// Refresh aggregate occupancy gauges from distinct live physical backends.
///
/// This performs only cheap approximate accounting. It removes expired weak
/// registrations, does not keep caches alive, and does not scan cache entries.
#[cfg(feature = "metrics")]
pub fn refresh_metrics() {
    enabled::refresh_metrics();
}

#[cfg(feature = "metrics")]
#[derive(Debug)]
pub(super) struct LookupMetricKeys {
    hit: Key,
    miss: Key,
}

#[cfg(feature = "metrics")]
impl LookupMetricKeys {
    pub(super) fn aggregate(cache: CacheMetricsKind, backend: CacheBackendKind) -> Self {
        Self::new(cache, backend, None)
    }

    pub(super) fn by_type(
        cache: CacheMetricsKind,
        backend: CacheBackendKind,
        type_name: &'static str,
    ) -> Self {
        Self::new(cache, backend, Some(type_name))
    }

    fn new(
        cache: CacheMetricsKind,
        backend: CacheBackendKind,
        type_name: Option<&'static str>,
    ) -> Self {
        Self {
            hit: lookup_key(cache, backend, type_name, "hit"),
            miss: lookup_key(cache, backend, type_name, "miss"),
        }
    }

    pub(super) fn get(&self, is_hit: bool) -> &Key {
        if is_hit { &self.hit } else { &self.miss }
    }
}

#[cfg(feature = "metrics")]
fn lookup_key(
    cache: CacheMetricsKind,
    backend: CacheBackendKind,
    type_name: Option<&'static str>,
    outcome: &'static str,
) -> Key {
    let mut labels = Vec::with_capacity(if type_name.is_some() { 4 } else { 3 });
    labels.push(Label::new("cache", cache.as_str()));
    labels.push(Label::new("backend", backend.as_str()));
    if let Some(type_name) = type_name {
        labels.push(Label::new("type", type_name));
    }
    labels.push(Label::new("outcome", outcome));
    Key::from_parts(METRIC_LOOKUPS, labels)
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn lookup(aggregate_key: &Key, type_key: &Key) {
    static METADATA: metrics::Metadata<'static> =
        metrics::Metadata::new(module_path!(), metrics::Level::INFO, Some(module_path!()));
    metrics::with_recorder(|recorder| {
        recorder
            .register_counter(aggregate_key, &METADATA)
            .increment(1);
        recorder.register_counter(type_key, &METADATA).increment(1);
    });
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn lookup_error(
    cache: CacheMetricsKind,
    backend: CacheBackendKind,
    type_name: &'static str,
    reason: &'static str,
) {
    metrics::counter!(
        METRIC_LOOKUP_ERRORS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "reason" => reason
    )
    .increment(1);
    metrics::counter!(
        METRIC_LOOKUP_ERRORS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name,
        "reason" => reason
    )
    .increment(1);
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn load_started(
    cache: CacheMetricsKind,
    backend: CacheBackendKind,
    type_name: &'static str,
) {
    metrics::gauge!(
        METRIC_LOADS_IN_FLIGHT,
        "cache" => cache.as_str(),
        "backend" => backend.as_str()
    )
    .increment(1.0);
    metrics::gauge!(
        METRIC_LOADS_IN_FLIGHT,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name
    )
    .increment(1.0);
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn load_finished(
    cache: CacheMetricsKind,
    backend: CacheBackendKind,
    type_name: &'static str,
    outcome: &'static str,
    duration_ns: u64,
) {
    metrics::gauge!(
        METRIC_LOADS_IN_FLIGHT,
        "cache" => cache.as_str(),
        "backend" => backend.as_str()
    )
    .decrement(1.0);
    metrics::gauge!(
        METRIC_LOADS_IN_FLIGHT,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name
    )
    .decrement(1.0);
    metrics::counter!(
        METRIC_LOADS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "outcome" => outcome
    )
    .increment(1);
    metrics::counter!(
        METRIC_LOADS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name,
        "outcome" => outcome
    )
    .increment(1);
    metrics::histogram!(
        METRIC_LOAD_DURATION,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "outcome" => outcome
    )
    .record(duration_ns as f64 / 1_000_000_000.0);
    metrics::histogram!(
        METRIC_LOAD_DURATION,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name,
        "outcome" => outcome
    )
    .record(duration_ns as f64 / 1_000_000_000.0);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn load_finished(
    _cache: CacheMetricsKind,
    _backend: CacheBackendKind,
    _type_name: &'static str,
    _outcome: &'static str,
    _duration_ns: u64,
) {
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn backend_write(kind: CacheBackendKind, bytes: u64) {
    metrics::counter!(METRIC_WRITE_ATTEMPTS, "backend" => kind.as_str()).increment(1);
    metrics::counter!(METRIC_WRITE_BYTES, "backend" => kind.as_str()).increment(bytes);
    metrics::histogram!(METRIC_ENTRY_SIZE, "backend" => kind.as_str()).record(bytes as f64);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn backend_write(_kind: CacheBackendKind, _bytes: u64) {}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn size_removal(kind: CacheBackendKind, count: u64, bytes: u64) {
    metrics::counter!(METRIC_SIZE_REMOVALS, "backend" => kind.as_str()).increment(count);
    metrics::counter!(METRIC_SIZE_REMOVED_BYTES, "backend" => kind.as_str()).increment(bytes);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn size_removal(_kind: CacheBackendKind, _count: u64, _bytes: u64) {}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn rejection(kind: CacheBackendKind, reason: &'static str) {
    metrics::counter!(
        METRIC_WRITE_REJECTIONS,
        "backend" => kind.as_str(),
        "reason" => reason
    )
    .increment(1);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn rejection(_kind: CacheBackendKind, _reason: &'static str) {}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn bypass(kind: CacheBackendKind, reason: &'static str) {
    metrics::counter!(
        METRIC_BYPASSES,
        "backend" => kind.as_str(),
        "reason" => reason
    )
    .increment(1);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn bypass(_kind: CacheBackendKind, _reason: &'static str) {}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn warm_attempt(
    cache: CacheMetricsKind,
    backend: CacheBackendKind,
    type_name: &'static str,
) {
    metrics::counter!(
        METRIC_WARM_ATTEMPTS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str()
    )
    .increment(1);
    metrics::counter!(
        METRIC_WARM_ATTEMPTS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name
    )
    .increment(1);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn warm_attempt(
    _cache: CacheMetricsKind,
    _backend: CacheBackendKind,
    _type_name: &'static str,
) {
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn warm_hit(
    cache: CacheMetricsKind,
    backend: CacheBackendKind,
    type_name: &'static str,
) {
    metrics::counter!(
        METRIC_WARM_HITS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str()
    )
    .increment(1);
    metrics::counter!(
        METRIC_WARM_HITS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name
    )
    .increment(1);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn warm_hit(
    _cache: CacheMetricsKind,
    _backend: CacheBackendKind,
    _type_name: &'static str,
) {
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn warm_load(
    cache: CacheMetricsKind,
    backend: CacheBackendKind,
    type_name: &'static str,
    outcome: &'static str,
) {
    metrics::counter!(
        METRIC_WARM_LOADS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "outcome" => outcome
    )
    .increment(1);
    metrics::counter!(
        METRIC_WARM_LOADS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name,
        "outcome" => outcome
    )
    .increment(1);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn warm_load(
    _cache: CacheMetricsKind,
    _backend: CacheBackendKind,
    _type_name: &'static str,
    _outcome: &'static str,
) {
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn warm_load_bytes(
    cache: CacheMetricsKind,
    backend: CacheBackendKind,
    type_name: &'static str,
    bytes: u64,
) {
    metrics::counter!(
        METRIC_WARM_LOAD_BYTES,
        "cache" => cache.as_str(),
        "backend" => backend.as_str()
    )
    .increment(bytes);
    metrics::counter!(
        METRIC_WARM_LOAD_BYTES,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name
    )
    .increment(bytes);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn warm_load_bytes(
    _cache: CacheMetricsKind,
    _backend: CacheBackendKind,
    _type_name: &'static str,
    _bytes: u64,
) {
}

#[cfg(feature = "metrics")]
#[inline]
pub(crate) fn warm_error(
    cache: CacheMetricsKind,
    backend: CacheBackendKind,
    type_name: &'static str,
) {
    metrics::counter!(
        METRIC_WARM_ERRORS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str()
    )
    .increment(1);
    metrics::counter!(
        METRIC_WARM_ERRORS,
        "cache" => cache.as_str(),
        "backend" => backend.as_str(),
        "type" => type_name
    )
    .increment(1);
}

#[cfg(not(feature = "metrics"))]
#[inline]
pub(crate) fn warm_error(
    _cache: CacheMetricsKind,
    _backend: CacheBackendKind,
    _type_name: &'static str,
) {
}

#[cfg(all(test, feature = "metrics"))]
mod tests {
    use std::borrow::Cow;
    use std::pin::Pin;
    use std::sync::Arc;

    use async_trait::async_trait;
    use futures::Future;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder, Snapshotter};

    use super::*;
    use crate::Result;
    use crate::cache::{
        CacheBackendDiagnostics, CacheCodec, CacheEntry, CacheKey, CacheLoadOrigin,
        CacheSnapshotMode, InternalCacheKey, LanceCache, MokaCacheBackend, QuickCacheBackend,
    };

    #[derive(Debug)]
    struct GaugeBackend;

    struct TestKey;

    impl CacheKey for TestKey {
        type ValueType = u64;

        fn key(&self) -> Cow<'_, str> {
            Cow::Borrowed("telemetry-test")
        }

        fn type_name() -> &'static str {
            "TelemetryTest"
        }
    }

    #[async_trait]
    impl CacheBackend for GaugeBackend {
        async fn get(
            &self,
            _key: &InternalCacheKey,
            _codec: Option<CacheCodec>,
        ) -> Option<CacheEntry> {
            None
        }

        async fn insert(
            &self,
            _key: &InternalCacheKey,
            _entry: CacheEntry,
            _size_bytes: usize,
            _codec: Option<CacheCodec>,
        ) {
        }

        async fn get_or_insert<'a>(
            &self,
            _key: &InternalCacheKey,
            loader: Pin<Box<dyn Future<Output = Result<(CacheEntry, usize)>> + Send + 'a>>,
            _codec: Option<CacheCodec>,
        ) -> Result<(CacheEntry, bool)> {
            loader.await.map(|(entry, _)| (entry, false))
        }

        async fn clear(&self) {}

        async fn num_entries(&self) -> usize {
            7
        }

        async fn size_bytes(&self) -> usize {
            123
        }

        fn diagnostics(&self) -> CacheBackendDiagnostics {
            CacheBackendDiagnostics {
                kind: CacheBackendKind::Custom,
                pool_id: Some(u64::MAX),
                capacity_bytes: Some(321),
                enabled: Some(true),
                size_bytes: Some(123),
                num_entries: Some(7),
                ..Default::default()
            }
        }
    }

    #[allow(clippy::collapsible_if)]
    fn gauge(snapshotter: &Snapshotter, name: &str, backend: &str) -> Option<f64> {
        snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .find_map(|(key, _, _, value)| {
                let key = key.key();
                let has_backend = key
                    .labels()
                    .any(|label| label.key() == "backend" && label.value() == backend);
                if key.name() == name && has_backend {
                    if let DebugValue::Gauge(value) = value {
                        return Some(value.0);
                    }
                }
                None
            })
    }

    #[allow(clippy::collapsible_if)]
    fn counter(
        metrics: &[(metrics::Key, DebugValue)],
        name: &str,
        labels: &[(&str, &str)],
    ) -> Option<u64> {
        metrics.iter().find_map(|(key, value)| {
            let has_labels = labels.iter().all(|(expected_key, expected_value)| {
                key.labels()
                    .any(|label| label.key() == *expected_key && label.value() == *expected_value)
            });
            if key.name() == name && has_labels {
                if let DebugValue::Counter(value) = value {
                    return Some(*value);
                }
            }
            None
        })
    }

    #[test]
    fn cache_operations_emit_activity_and_backend_events() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let cache = LanceCache::with_backend_and_metrics_kind(
            Arc::new(QuickCacheBackend::with_capacity(1 << 20)),
            CacheMetricsKind::Index,
        );

        metrics::with_local_recorder(&recorder, || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            runtime.block_on(async {
                assert!(cache.get_with_key(&TestKey).await.is_none());
                assert_eq!(
                    *cache
                        .get_or_insert_with_key(TestKey, || async { Ok(42) })
                        .await
                        .unwrap(),
                    42
                );
                assert_eq!(*cache.get_with_key(&TestKey).await.unwrap(), 42);
            });
        });

        let metrics = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(key, _, _, value)| (key.key().clone(), value))
            .collect::<Vec<_>>();
        assert_eq!(
            counter(
                &metrics,
                METRIC_LOOKUPS,
                &[
                    ("cache", "index"),
                    ("backend", "quick"),
                    ("outcome", "miss"),
                ],
            ),
            Some(2)
        );
        assert_eq!(
            counter(
                &metrics,
                METRIC_LOOKUPS,
                &[
                    ("cache", "index"),
                    ("backend", "quick"),
                    ("type", "TelemetryTest"),
                    ("outcome", "miss"),
                ],
            ),
            Some(2)
        );
        assert_eq!(
            counter(
                &metrics,
                METRIC_LOOKUPS,
                &[("cache", "index"), ("backend", "quick"), ("outcome", "hit"),],
            ),
            Some(1)
        );
        assert_eq!(
            counter(
                &metrics,
                METRIC_LOADS,
                &[
                    ("cache", "index"),
                    ("backend", "quick"),
                    ("outcome", "success"),
                ],
            ),
            Some(1)
        );
        assert_eq!(
            counter(&metrics, METRIC_WRITE_ATTEMPTS, &[("backend", "quick")],),
            Some(1)
        );
        assert_eq!(cache.diagnostics().activity.hits, 1);
        assert_eq!(cache.diagnostics().activity.misses, 2);
    }

    #[test]
    fn cached_lookup_keys_use_the_current_local_recorder() {
        let first_recorder = DebuggingRecorder::new();
        let first_snapshotter = first_recorder.snapshotter();
        let second_recorder = DebuggingRecorder::new();
        let second_snapshotter = second_recorder.snapshotter();
        let cache = LanceCache::with_backend_and_metrics_kind(
            Arc::new(QuickCacheBackend::with_capacity(1 << 20)),
            CacheMetricsKind::Index,
        );
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        runtime.block_on(async {
            cache.insert_with_key(&TestKey, Arc::new(42)).await;
        });

        metrics::with_local_recorder(&first_recorder, || {
            assert!(runtime.block_on(cache.get_with_key(&TestKey)).is_some());
        });
        metrics::with_local_recorder(&second_recorder, || {
            assert!(runtime.block_on(cache.get_with_key(&TestKey)).is_some());
        });

        for snapshotter in [&first_snapshotter, &second_snapshotter] {
            let metrics = snapshotter
                .snapshot()
                .into_vec()
                .into_iter()
                .map(|(key, _, _, value)| (key.key().clone(), value))
                .collect::<Vec<_>>();
            assert_eq!(
                counter(
                    &metrics,
                    METRIC_LOOKUPS,
                    &[("cache", "index"), ("backend", "quick"), ("outcome", "hit"),],
                ),
                Some(1)
            );
            assert_eq!(
                counter(
                    &metrics,
                    METRIC_LOOKUPS,
                    &[
                        ("cache", "index"),
                        ("backend", "quick"),
                        ("type", "TelemetryTest"),
                        ("outcome", "hit"),
                    ],
                ),
                Some(1)
            );
        }
    }

    #[test]
    fn moka_exports_size_removals_only_when_enabled() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            runtime.block_on(async {
                let key = InternalCacheKey::from_bytes([0; 16]);
                let disabled = MokaCacheBackend::with_capacity(256);
                disabled.insert(&key, Arc::new(()), 1024, None).await;
                disabled
                    .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
                    .await;
                let snapshot = snapshotter.snapshot().into_vec();
                let metrics = snapshot
                    .into_iter()
                    .map(|(key, _, _, value)| (key.key().clone(), value))
                    .collect::<Vec<_>>();
                assert_eq!(
                    counter(&metrics, METRIC_SIZE_REMOVALS, &[("backend", "moka")]),
                    None
                );

                let enabled = MokaCacheBackend::builder(256)
                    .with_size_removal_metrics()
                    .build();
                enabled.insert(&key, Arc::new(()), 1024, None).await;
                enabled
                    .diagnostics_with_mode(CacheSnapshotMode::Refreshed)
                    .await;
            });
        });
        let metrics = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(key, _, _, value)| (key.key().clone(), value))
            .collect::<Vec<_>>();
        assert_eq!(
            counter(&metrics, METRIC_SIZE_REMOVALS, &[("backend", "moka")]),
            Some(1)
        );
        assert_eq!(
            counter(&metrics, METRIC_SIZE_REMOVED_BYTES, &[("backend", "moka")]),
            Some(1040)
        );
        assert_eq!(
            counter(&metrics, METRIC_COALESCED_LOADS, &[("backend", "moka")]),
            None
        );
    }

    #[test]
    fn occupancy_gauges_deduplicate_and_clear_expired_pools() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        metrics::with_local_recorder(&recorder, || {
            let backend: Arc<dyn CacheBackend> = Arc::new(GaugeBackend);
            let first = LanceCache::with_backend(backend.clone());
            let second = LanceCache::with_backend(backend.clone());

            refresh_metrics();
            assert_eq!(gauge(&snapshotter, METRIC_CAPACITY, "custom"), Some(321.0));
            assert_eq!(gauge(&snapshotter, METRIC_SIZE, "custom"), Some(123.0));
            assert_eq!(gauge(&snapshotter, METRIC_ENTRIES, "custom"), Some(7.0));

            drop(first);
            drop(second);
            drop(backend);
            refresh_metrics();
            assert_eq!(gauge(&snapshotter, METRIC_CAPACITY, "custom"), Some(0.0));
            assert_eq!(gauge(&snapshotter, METRIC_SIZE, "custom"), Some(0.0));
            assert_eq!(gauge(&snapshotter, METRIC_ENTRIES, "custom"), Some(0.0));
        });
    }

    #[test]
    fn warm_origin_emits_aggregate_and_typed_events() {
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();
        let cache = LanceCache::with_backend_and_metrics_kind(
            Arc::new(QuickCacheBackend::with_capacity(1 << 20)),
            CacheMetricsKind::Index,
        );
        let warm = cache.with_load_origin(CacheLoadOrigin::Warm);
        metrics::with_local_recorder(&recorder, || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap();
            runtime.block_on(async {
                warm.get_or_insert_with_key(TestKey, || async { Ok(42) })
                    .await
                    .unwrap();
                assert_eq!(*warm.get_with_key(&TestKey).await.unwrap(), 42);
            });
        });
        let metrics = snapshotter
            .snapshot()
            .into_vec()
            .into_iter()
            .map(|(key, _, _, value)| (key.key().clone(), value))
            .collect::<Vec<_>>();
        assert_eq!(
            counter(
                &metrics,
                METRIC_WARM_ATTEMPTS,
                &[
                    ("cache", "index"),
                    ("backend", "quick"),
                    ("type", "TelemetryTest")
                ],
            ),
            Some(2)
        );
        assert_eq!(
            counter(
                &metrics,
                METRIC_WARM_HITS,
                &[
                    ("cache", "index"),
                    ("backend", "quick"),
                    ("type", "TelemetryTest")
                ],
            ),
            Some(1)
        );
        assert_eq!(
            counter(
                &metrics,
                METRIC_WARM_LOADS,
                &[
                    ("cache", "index"),
                    ("backend", "quick"),
                    ("outcome", "success")
                ],
            ),
            Some(1)
        );
        assert!(
            counter(
                &metrics,
                METRIC_WARM_LOAD_BYTES,
                &[("type", "TelemetryTest")]
            )
            .unwrap()
                > 0
        );
        assert_eq!(cache.diagnostics().activity.warm.attempts, 2);
    }
}
