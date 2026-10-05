// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Native diagnostics, available independently of an installed metrics recorder.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Maximum number of distinct cache type names retained by one logical cache.
///
/// Additional names are attributed to the `other` bucket. Exported type labels
/// have the same process-wide bound.
pub const MAX_CACHE_TYPE_SERIES: usize = 64;

/// Stable context passed from typed cache APIs to a physical backend.
///
/// Custom backends can override the context-aware methods on
/// [`super::CacheBackend`] to retain this type identity. Calls through the
/// original backend methods carry an untagged context.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CacheOperationContext {
    type_name: Option<&'static str>,
    origin: CacheLoadOrigin,
}

impl CacheOperationContext {
    pub(super) fn new(type_name: &'static str, origin: CacheLoadOrigin) -> Self {
        Self {
            type_name: Some(type_name),
            origin,
        }
    }

    /// Return the stable type name, or `None` for an untagged backend call.
    pub fn type_name(self) -> Option<&'static str> {
        self.type_name
    }

    /// Return whether the operation came from demand or explicit warming.
    pub fn origin(self) -> CacheLoadOrigin {
        self.origin
    }
}

/// Why a cache operation was started.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CacheLoadOrigin {
    /// A normal read or query requested the value.
    #[default]
    Demand,
    /// An explicit prewarm path requested the value before demand.
    Warm,
}

impl CacheLoadOrigin {
    /// Stable label used by metrics exporters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Demand => "demand",
            Self::Warm => "warm",
        }
    }
}

/// How a successful get-or-load operation obtained its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheLoadOutcome {
    /// This call executed the loader.
    Loaded,
    /// The value was already resident when the backend checked the key.
    ResidentHit,
    /// This call received the result of another concurrent loader.
    Coalesced,
    /// The loader was skipped, but the backend cannot distinguish a resident
    /// hit from a shared concurrent load.
    LoaderSkippedUnknown,
}

impl CacheLoadOutcome {
    /// Whether this call skipped its loader.
    pub fn was_loader_skipped(self) -> bool {
        !matches!(self, Self::Loaded)
    }
}

/// How backend occupancy is collected. Neither mode is an atomic snapshot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CacheSnapshotMode {
    /// Read cheap accounting only; asynchronous maintenance may still be pending.
    #[default]
    Approximate,
    /// Request backend maintenance before reading occupancy. Concurrent writes
    /// can still change fields between reads.
    Refreshed,
}

/// Bounded classification of a physical cache pool.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CacheBackendKind {
    /// The built-in sharded Quick cache.
    Quick,
    /// The built-in Moka cache, with asynchronous maintenance.
    Moka,
    /// A backend whose lifecycle capabilities must be explicitly reported.
    #[default]
    Custom,
}

/// Bounded logical cache classification used by exported activity metrics.
///
/// Physical backend metrics deliberately omit this label because one backend
/// can be shared by wrappers serving different logical caches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CacheMetricsKind {
    /// Session index data and opened indices.
    Index,
    /// Session dataset and file metadata.
    Metadata,
    /// A cache without a built-in session role.
    #[default]
    Other,
}

impl CacheMetricsKind {
    /// Stable label used by metrics exporters.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Index => "index",
            Self::Metadata => "metadata",
            Self::Other => "other",
        }
    }
}

impl CacheBackendKind {
    /// Stable label used by exporters and language bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quick => "quick",
            Self::Moka => "moka",
            Self::Custom => "custom",
        }
    }
}

/// Lifetime wrapper activity, shared by clones and child namespaces.
///
/// Counts never reset on [`super::LanceCache::clear`]. Loader failures are
/// lookup errors, not misses. Type mismatches retain the legacy miss convention
/// and additionally record a lookup error. Coalesced successful loads are hits.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CacheActivity {
    /// Successful lookups whose loader was skipped, including shared loads.
    pub hits: u64,
    /// Absent entries, executed successful loads, and type mismatches.
    pub misses: u64,
    /// Failed backend calls and type mismatches, including propagated loader errors.
    pub lookup_errors: u64,
    /// Returned entries that could not be downcast to the requested type.
    pub type_mismatches: u64,
    /// Executing loaders started, excluding callers that share their result.
    pub loads_started: u64,
    /// Executing loaders that returned a value, before size accounting.
    pub loads_succeeded: u64,
    /// Executing loaders that returned an error.
    pub loads_failed: u64,
    /// Executing loader futures dropped before completion.
    pub loads_cancelled: u64,
    /// Executing loader futures currently active.
    pub loads_in_flight: u64,
    /// Total elapsed successful loader time, in nanoseconds.
    pub load_success_duration_ns: u64,
    /// Total elapsed failed loader time, in nanoseconds.
    pub load_error_duration_ns: u64,
    /// Total elapsed cancelled loader time, in nanoseconds.
    pub load_cancelled_duration_ns: u64,
    /// Activity explicitly attributed to prewarming.
    pub warm: CacheWarmActivity,
}

/// Lifetime activity from operations explicitly attributed to prewarming.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CacheWarmActivity {
    /// Cache calls made by a warm-origin handle.
    pub attempts: u64,
    /// Warm attempts served without executing or submitting a load.
    pub hits: u64,
    /// Warm materializations started, including direct warm insertions.
    pub loads_started: u64,
    /// Warm materializations successfully submitted to the cache.
    pub loads_succeeded: u64,
    /// Warm loader executions that returned an error.
    pub loads_failed: u64,
    /// Warm loader executions dropped before completion.
    pub loads_cancelled: u64,
    /// Accounted bytes from successful warm materializations.
    pub load_bytes: u64,
    /// Failed warm cache calls, including loader failures and type mismatches.
    pub errors: u64,
}

impl CacheWarmActivity {
    fn saturating_add_assign(&mut self, other: &Self) {
        self.attempts = self.attempts.saturating_add(other.attempts);
        self.hits = self.hits.saturating_add(other.hits);
        self.loads_started = self.loads_started.saturating_add(other.loads_started);
        self.loads_succeeded = self.loads_succeeded.saturating_add(other.loads_succeeded);
        self.loads_failed = self.loads_failed.saturating_add(other.loads_failed);
        self.loads_cancelled = self.loads_cancelled.saturating_add(other.loads_cancelled);
        self.load_bytes = self.load_bytes.saturating_add(other.load_bytes);
        self.errors = self.errors.saturating_add(other.errors);
    }

    fn saturating_sub(&self, other: &Self) -> Self {
        Self {
            attempts: self.attempts.saturating_sub(other.attempts),
            hits: self.hits.saturating_sub(other.hits),
            loads_started: self.loads_started.saturating_sub(other.loads_started),
            loads_succeeded: self.loads_succeeded.saturating_sub(other.loads_succeeded),
            loads_failed: self.loads_failed.saturating_sub(other.loads_failed),
            loads_cancelled: self.loads_cancelled.saturating_sub(other.loads_cancelled),
            load_bytes: self.load_bytes.saturating_sub(other.load_bytes),
            errors: self.errors.saturating_sub(other.errors),
        }
    }

    fn is_empty(&self) -> bool {
        self.attempts == 0
            && self.hits == 0
            && self.loads_started == 0
            && self.loads_succeeded == 0
            && self.loads_failed == 0
            && self.loads_cancelled == 0
            && self.load_bytes == 0
            && self.errors == 0
    }
}

impl CacheActivity {
    fn saturating_add_assign(&mut self, other: &Self) {
        self.hits = self.hits.saturating_add(other.hits);
        self.misses = self.misses.saturating_add(other.misses);
        self.lookup_errors = self.lookup_errors.saturating_add(other.lookup_errors);
        self.type_mismatches = self.type_mismatches.saturating_add(other.type_mismatches);
        self.loads_started = self.loads_started.saturating_add(other.loads_started);
        self.loads_succeeded = self.loads_succeeded.saturating_add(other.loads_succeeded);
        self.loads_failed = self.loads_failed.saturating_add(other.loads_failed);
        self.loads_cancelled = self.loads_cancelled.saturating_add(other.loads_cancelled);
        self.loads_in_flight = self.loads_in_flight.saturating_add(other.loads_in_flight);
        self.load_success_duration_ns = self
            .load_success_duration_ns
            .saturating_add(other.load_success_duration_ns);
        self.load_error_duration_ns = self
            .load_error_duration_ns
            .saturating_add(other.load_error_duration_ns);
        self.load_cancelled_duration_ns = self
            .load_cancelled_duration_ns
            .saturating_add(other.load_cancelled_duration_ns);
        self.warm.saturating_add_assign(&other.warm);
    }

    fn saturating_sub(&self, other: &Self) -> Self {
        Self {
            hits: self.hits.saturating_sub(other.hits),
            misses: self.misses.saturating_sub(other.misses),
            lookup_errors: self.lookup_errors.saturating_sub(other.lookup_errors),
            type_mismatches: self.type_mismatches.saturating_sub(other.type_mismatches),
            loads_started: self.loads_started.saturating_sub(other.loads_started),
            loads_succeeded: self.loads_succeeded.saturating_sub(other.loads_succeeded),
            loads_failed: self.loads_failed.saturating_sub(other.loads_failed),
            loads_cancelled: self.loads_cancelled.saturating_sub(other.loads_cancelled),
            loads_in_flight: self.loads_in_flight.saturating_sub(other.loads_in_flight),
            load_success_duration_ns: self
                .load_success_duration_ns
                .saturating_sub(other.load_success_duration_ns),
            load_error_duration_ns: self
                .load_error_duration_ns
                .saturating_sub(other.load_error_duration_ns),
            load_cancelled_duration_ns: self
                .load_cancelled_duration_ns
                .saturating_sub(other.load_cancelled_duration_ns),
            warm: self.warm.saturating_sub(&other.warm),
        }
    }

    fn is_empty(&self) -> bool {
        self.hits == 0
            && self.misses == 0
            && self.lookup_errors == 0
            && self.type_mismatches == 0
            && self.loads_started == 0
            && self.loads_succeeded == 0
            && self.loads_failed == 0
            && self.loads_cancelled == 0
            && self.loads_in_flight == 0
            && self.load_success_duration_ns == 0
            && self.load_error_duration_ns == 0
            && self.load_cancelled_duration_ns == 0
            && self.warm.is_empty()
    }
}

/// Physical backend measurements. `None` means unavailable, never measured zero.
///
/// Independently created wrappers over one backend share these values, but have
/// separate [`CacheActivity`]. Weighted bytes include the physical key and may
/// count shared allocations more than once; they do not measure RSS or bytes
/// actually freed. Lifecycle counters never reset on clear.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CacheBackendDiagnostics {
    /// Backend implementation class, suitable for bounded telemetry labels.
    pub kind: CacheBackendKind,
    /// Opaque process-local pool identifier; unavailable for uninstrumented
    /// custom backends. Never use it as a default exported label.
    pub pool_id: Option<u64>,
    /// Configured weighted capacity, or absent if unknown / unbounded.
    pub capacity_bytes: Option<u64>,
    /// Whether caching is enabled, or absent when the backend cannot report it.
    pub enabled: Option<bool>,
    /// Current accounted occupancy, or absent if cheap accounting is unsupported.
    pub size_bytes: Option<u64>,
    /// Current resident entry count, or absent if cheap accounting is unsupported.
    pub num_entries: Option<u64>,
    /// Backend write submissions, including replacements and rejected candidates.
    pub write_attempts: Option<u64>,
    /// Accounted bytes submitted, in the same units as backend occupancy.
    pub write_bytes: Option<u64>,
    /// Size-removal notifications, including never-admitted candidates.
    /// `None` when Moka's optional eviction listener is disabled.
    pub size_removals: Option<u64>,
    /// Accounted bytes in size-removal notifications, not freed heap bytes.
    /// `None` when Moka's optional eviction listener is disabled.
    pub size_removed_bytes: Option<u64>,
    /// Explicit write attempts into a disabled cache.
    pub disabled_write_rejections: Option<u64>,
    /// Loader calls bypassing insertion because caching is disabled.
    pub disabled_bypasses: Option<u64>,
    /// Loader submissions whose Quick placeholder was superseded.
    pub lost_placeholder_rejections: Option<u64>,
    /// Submitted sizes exceeding the backend's weight representation.
    pub weight_saturations: Option<u64>,
    /// Whether every rejected submission has a positively identified reason.
    /// False for the built-ins: size callbacks conflate rejection and eviction.
    pub write_rejections_complete: bool,
    /// Proven resident capacity evictions. Unavailable when callbacks also
    /// include admission rejection without distinguishing it.
    pub resident_evictions: Option<u64>,
    /// Proven successful policy admissions. A write call alone cannot prove it.
    pub admissions: Option<u64>,
    /// Exact shared-load observations; `None` for both built-in backends because
    /// neither can distinguish resident hits from loader sharing in every case.
    pub coalesced_loads: Option<u64>,
}

/// Lifetime activity attributed to one stable cache key type.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CacheTypeActivity {
    /// Stable [`super::CacheKey`] or [`super::UnsizedCacheKey`] identity.
    /// The value `other` contains activity beyond the bounded type registry.
    pub type_name: String,
    /// Activity for this type. After quiescence, all rows sum to aggregate
    /// activity even when some names were assigned to `other`.
    pub activity: CacheActivity,
}

/// Approximate occupancy attributed to one stable cache key type.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CacheTypeOccupancy {
    /// Stable cache key identity.
    pub type_name: String,
    /// Accounted backend weight, using the same units as aggregate occupancy.
    pub size_bytes: u64,
    /// Number of resident entries carrying this tag.
    pub num_entries: u64,
}

/// Occupancy collected by scanning tagged backend records.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CacheOccupancyByType {
    /// Occupancy for tagged entries, sorted by type name by built-in backends.
    pub types: Vec<CacheTypeOccupancy>,
    /// Accounted weight for entries inserted through an untagged backend API.
    pub untagged_size_bytes: u64,
    /// Resident entries inserted through an untagged backend API.
    pub untagged_num_entries: u64,
}

/// Explicit per-type diagnostic detail for one logical cache wrapper.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CacheByTypeDiagnostics {
    /// Bounded lifetime activity rows.
    pub activity: Vec<CacheTypeActivity>,
    /// Approximate occupancy, or `None` when the backend cannot enumerate
    /// tagged records.
    pub occupancy: Option<CacheOccupancyByType>,
    /// Activity events whose exported type label was collapsed to `other`.
    pub type_label_overflow_events: u64,
}

/// Sampled diagnostics for one wrapper and its physical backend.
///
/// # Example
/// ```
/// use lance_core::cache::LanceCache;
/// let cache = LanceCache::with_capacity(1024);
/// let snapshot = cache.diagnostics();
/// assert_eq!(snapshot.backend.capacity_bytes, Some(1024));
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct CacheDiagnostics {
    /// Wrapper lifetime lookup and loader measurements.
    pub activity: CacheActivity,
    /// Shared physical pool accounting and availability information.
    pub backend: CacheBackendDiagnostics,
    /// Occupancy / configured capacity; absent at zero or unknown capacity,
    /// or unavailable occupancy. Not clamped and not a process-memory metric.
    pub utilization: Option<f64>,
    /// Explicit per-type detail. Ordinary diagnostics keep this absent so they
    /// remain constant cost.
    pub by_type: Option<CacheByTypeDiagnostics>,
}

impl CacheDiagnostics {
    pub(super) fn new(activity: CacheActivity, backend: CacheBackendDiagnostics) -> Self {
        let utilization = backend
            .capacity_bytes
            .filter(|capacity| *capacity > 0)
            .zip(backend.size_bytes)
            .map(|(capacity, size)| size as f64 / capacity as f64);
        Self {
            activity,
            backend,
            utilization,
            by_type: None,
        }
    }

    pub(super) fn with_by_type(
        activity: CacheActivity,
        backend: CacheBackendDiagnostics,
        by_type: CacheByTypeDiagnostics,
    ) -> Self {
        let mut diagnostics = Self::new(activity, backend);
        diagnostics.by_type = Some(by_type);
        diagnostics
    }
}

#[derive(Debug)]
struct TypeRegistration {
    diagnostic_name: &'static str,
    #[cfg(feature = "metrics")]
    export_name: &'static str,
    #[cfg(feature = "metrics")]
    export_overflow: bool,
}

#[derive(Debug)]
pub(super) struct TypeActivitySlot {
    registration: OnceLock<TypeRegistration>,
    #[cfg(feature = "metrics")]
    lookup_metric_keys: OnceLock<super::telemetry::LookupMetricKeys>,
    hits: AtomicU64,
    misses: AtomicU64,
    activity: ActivityCounters,
}

#[derive(Clone, Copy)]
pub(super) struct TypeActivityHandle(u8);

impl TypeActivityHandle {
    fn is_primary(self) -> bool {
        self.0 == 0
    }
}

impl TypeActivitySlot {
    fn new() -> Self {
        Self {
            registration: OnceLock::new(),
            #[cfg(feature = "metrics")]
            lookup_metric_keys: OnceLock::new(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            activity: ActivityCounters::default(),
        }
    }

    fn registration(&self) -> &TypeRegistration {
        self.registration
            .get()
            .expect("cache type activity slot must be registered before use")
    }

    pub(super) fn diagnostic_name(&self) -> &'static str {
        self.registration().diagnostic_name
    }

    fn has_activity(&self) -> bool {
        self.hits.load(Ordering::Relaxed) != 0
            || self.misses.load(Ordering::Relaxed) != 0
            || !self.activity.is_empty()
    }

    fn snapshot(&self) -> CacheActivity {
        self.activity.snapshot(
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }
}

/// A fixed-size registry. Registration takes a lock once per new local type;
/// activity on an existing type uses bounded `OnceLock` reads and atomics.
#[derive(Debug)]
pub(super) struct TypeActivityRegistry {
    slots: Box<[TypeActivitySlot]>,
    other: TypeActivitySlot,
    registration: Mutex<()>,
    type_label_overflow_events: AtomicU64,
    #[cfg(feature = "metrics")]
    cache_kind: CacheMetricsKind,
    #[cfg(feature = "metrics")]
    backend_kind: CacheBackendKind,
    #[cfg(feature = "metrics")]
    aggregate_lookup_metric_keys: super::telemetry::LookupMetricKeys,
}

impl TypeActivityRegistry {
    pub(super) fn new(cache_kind: CacheMetricsKind, backend_kind: CacheBackendKind) -> Self {
        Self::with_capacity(MAX_CACHE_TYPE_SERIES, cache_kind, backend_kind)
    }

    fn with_capacity(
        capacity: usize,
        cache_kind: CacheMetricsKind,
        backend_kind: CacheBackendKind,
    ) -> Self {
        #[cfg(not(feature = "metrics"))]
        let _ = (cache_kind, backend_kind);
        assert!(capacity > 0);
        assert!(capacity < u8::MAX as usize);
        let other = TypeActivitySlot::new();
        other
            .registration
            .set(TypeRegistration {
                diagnostic_name: "other",
                #[cfg(feature = "metrics")]
                export_name: "other",
                #[cfg(feature = "metrics")]
                export_overflow: true,
            })
            .unwrap_or_else(|_| unreachable!());
        #[cfg(feature = "metrics")]
        other
            .lookup_metric_keys
            .set(super::telemetry::LookupMetricKeys::by_type(
                cache_kind,
                backend_kind,
                "other",
            ))
            .unwrap_or_else(|_| unreachable!());
        Self {
            slots: (0..capacity).map(|_| TypeActivitySlot::new()).collect(),
            other,
            registration: Mutex::new(()),
            type_label_overflow_events: AtomicU64::new(0),
            #[cfg(feature = "metrics")]
            cache_kind,
            #[cfg(feature = "metrics")]
            backend_kind,
            #[cfg(feature = "metrics")]
            aggregate_lookup_metric_keys: super::telemetry::LookupMetricKeys::aggregate(
                cache_kind,
                backend_kind,
            ),
        }
    }

    #[inline]
    pub(super) fn get(&self, type_name: &'static str) -> TypeActivityHandle {
        if type_name == "other" {
            return self.other_handle();
        }
        if let Some(handle) = self.find(type_name) {
            return handle;
        }

        let _registration = self
            .registration
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(handle) = self.find(type_name) {
            return handle;
        }
        let Some((index, slot)) = self
            .slots
            .iter()
            .enumerate()
            .find(|(_, slot)| slot.registration.get().is_none())
        else {
            return self.other_handle();
        };
        #[cfg(feature = "metrics")]
        let (export_name, export_overflow) = super::telemetry::register_type_label(type_name);
        slot.registration
            .set(TypeRegistration {
                diagnostic_name: type_name,
                #[cfg(feature = "metrics")]
                export_name,
                #[cfg(feature = "metrics")]
                export_overflow,
            })
            .unwrap_or_else(|_| unreachable!());
        #[cfg(feature = "metrics")]
        slot.lookup_metric_keys
            .set(super::telemetry::LookupMetricKeys::by_type(
                self.cache_kind,
                self.backend_kind,
                export_name,
            ))
            .unwrap_or_else(|_| unreachable!());
        TypeActivityHandle(index as u8)
    }

    #[inline]
    #[allow(clippy::collapsible_if)]
    fn find(&self, type_name: &str) -> Option<TypeActivityHandle> {
        if let Some(registration) = self.slots[0].registration.get() {
            if Self::same_type(registration.diagnostic_name, type_name) {
                return Some(TypeActivityHandle(0));
            }
        }
        self.slots
            .iter()
            .enumerate()
            .skip(1)
            .find_map(|(index, slot)| {
                let registration = slot.registration.get()?;
                Self::same_type(registration.diagnostic_name, type_name)
                    .then_some(TypeActivityHandle(index as u8))
            })
    }

    #[inline]
    fn same_type(registered: &str, requested: &str) -> bool {
        let same_pointer = registered.len() == requested.len()
            && std::ptr::eq(registered.as_ptr(), requested.as_ptr());
        same_pointer || registered == requested
    }

    #[inline]
    fn slot(&self, handle: TypeActivityHandle) -> &TypeActivitySlot {
        if handle.0 as usize == self.slots.len() {
            &self.other
        } else {
            &self.slots[handle.0 as usize]
        }
    }

    fn other_handle(&self) -> TypeActivityHandle {
        TypeActivityHandle(self.slots.len() as u8)
    }

    #[cfg(feature = "metrics")]
    pub(super) fn export_name(&self, handle: TypeActivityHandle) -> &'static str {
        self.slot(handle).registration().export_name
    }

    #[cfg(feature = "metrics")]
    pub(super) fn lookup_metric_keys(
        &self,
        handle: TypeActivityHandle,
        is_hit: bool,
    ) -> (&metrics::Key, &metrics::Key) {
        let type_key = self
            .slot(handle)
            .lookup_metric_keys
            .get()
            .expect("cache type lookup metric keys must be registered before use")
            .get(is_hit);
        (self.aggregate_lookup_metric_keys.get(is_hit), type_key)
    }

    #[cfg(feature = "metrics")]
    pub(super) fn note_event(
        &self,
        handle: TypeActivityHandle,
        cache_kind: CacheMetricsKind,
        backend_kind: CacheBackendKind,
    ) {
        if self.slot(handle).registration().export_overflow {
            self.type_label_overflow_events
                .fetch_add(1, Ordering::Relaxed);
            super::telemetry::type_overflow(cache_kind, backend_kind);
        }
    }

    pub(super) fn record_hit(&self, handle: TypeActivityHandle) {
        if !handle.is_primary() {
            self.slot(handle).hits.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn record_miss(&self, handle: TypeActivityHandle) {
        if !handle.is_primary() {
            self.slot(handle).misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn record_lookup_error(&self, handle: TypeActivityHandle) {
        if !handle.is_primary() {
            self.slot(handle)
                .activity
                .lookup_errors
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn record_type_mismatch(&self, handle: TypeActivityHandle) {
        if !handle.is_primary() {
            self.slot(handle)
                .activity
                .type_mismatches
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn activity(&self, handle: TypeActivityHandle) -> Option<&ActivityCounters> {
        (!handle.is_primary()).then_some(&self.slot(handle).activity)
    }

    pub(super) fn snapshot(&self, aggregate: &CacheActivity) -> (Vec<CacheTypeActivity>, u64) {
        let mut activity = Vec::new();
        let mut secondary_total = CacheActivity::default();
        for slot in self.slots.iter().skip(1) {
            if slot.registration.get().is_some() && slot.has_activity() {
                let snapshot = slot.snapshot();
                secondary_total.saturating_add_assign(&snapshot);
                activity.push(CacheTypeActivity {
                    type_name: slot.diagnostic_name().to_string(),
                    activity: snapshot,
                });
            }
        }
        if self.other.has_activity() {
            let snapshot = self.other.snapshot();
            secondary_total.saturating_add_assign(&snapshot);
            activity.push(CacheTypeActivity {
                type_name: self.other.diagnostic_name().to_string(),
                activity: snapshot,
            });
        }
        if let Some(registration) = self.slots[0].registration.get() {
            let primary = aggregate.saturating_sub(&secondary_total);
            if !primary.is_empty() {
                activity.push(CacheTypeActivity {
                    type_name: registration.diagnostic_name.to_string(),
                    activity: primary,
                });
            }
        }
        activity.sort_unstable_by(|left, right| left.type_name.cmp(&right.type_name));
        (
            activity,
            self.type_label_overflow_events.load(Ordering::Relaxed),
        )
    }
}

#[derive(Debug, Default)]
pub(super) struct ActivityCounters {
    pub lookup_errors: AtomicU64,
    pub type_mismatches: AtomicU64,
    started: AtomicU64,
    succeeded: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    in_flight: AtomicU64,
    success_duration: AtomicU64,
    error_duration: AtomicU64,
    cancelled_duration: AtomicU64,
    warm: WarmActivityCounters,
}

#[derive(Debug, Default)]
struct WarmActivityCounters {
    attempts: AtomicU64,
    hits: AtomicU64,
    loads_started: AtomicU64,
    loads_succeeded: AtomicU64,
    loads_failed: AtomicU64,
    loads_cancelled: AtomicU64,
    load_bytes: AtomicU64,
    errors: AtomicU64,
}

impl WarmActivityCounters {
    fn snapshot(&self) -> CacheWarmActivity {
        CacheWarmActivity {
            attempts: self.attempts.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            loads_started: self.loads_started.load(Ordering::Relaxed),
            loads_succeeded: self.loads_succeeded.load(Ordering::Relaxed),
            loads_failed: self.loads_failed.load(Ordering::Relaxed),
            loads_cancelled: self.loads_cancelled.load(Ordering::Relaxed),
            load_bytes: self.load_bytes.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }

    fn is_empty(&self) -> bool {
        self.attempts.load(Ordering::Relaxed) == 0
            && self.hits.load(Ordering::Relaxed) == 0
            && self.loads_started.load(Ordering::Relaxed) == 0
            && self.loads_succeeded.load(Ordering::Relaxed) == 0
            && self.loads_failed.load(Ordering::Relaxed) == 0
            && self.loads_cancelled.load(Ordering::Relaxed) == 0
            && self.load_bytes.load(Ordering::Relaxed) == 0
            && self.errors.load(Ordering::Relaxed) == 0
    }
}

fn add_bytes(counter: &AtomicU64, bytes: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(bytes))
    });
}

impl ActivityCounters {
    pub fn snapshot(&self, hits: u64, misses: u64) -> CacheActivity {
        CacheActivity {
            hits,
            misses,
            lookup_errors: self.lookup_errors.load(Ordering::Relaxed),
            type_mismatches: self.type_mismatches.load(Ordering::Relaxed),
            loads_started: self.started.load(Ordering::Relaxed),
            loads_succeeded: self.succeeded.load(Ordering::Relaxed),
            loads_failed: self.failed.load(Ordering::Relaxed),
            loads_cancelled: self.cancelled.load(Ordering::Relaxed),
            loads_in_flight: self.in_flight.load(Ordering::Relaxed),
            load_success_duration_ns: self.success_duration.load(Ordering::Relaxed),
            load_error_duration_ns: self.error_duration.load(Ordering::Relaxed),
            load_cancelled_duration_ns: self.cancelled_duration.load(Ordering::Relaxed),
            warm: self.warm.snapshot(),
        }
    }

    fn is_empty(&self) -> bool {
        self.lookup_errors.load(Ordering::Relaxed) == 0
            && self.type_mismatches.load(Ordering::Relaxed) == 0
            && self.started.load(Ordering::Relaxed) == 0
            && self.succeeded.load(Ordering::Relaxed) == 0
            && self.failed.load(Ordering::Relaxed) == 0
            && self.cancelled.load(Ordering::Relaxed) == 0
            && self.in_flight.load(Ordering::Relaxed) == 0
            && self.warm.is_empty()
    }

    pub fn record_warm_attempt(
        &self,
        by_type: Option<&Self>,
        cache_kind: CacheMetricsKind,
        backend_kind: CacheBackendKind,
        type_name: &'static str,
    ) {
        self.warm.attempts.fetch_add(1, Ordering::Relaxed);
        if let Some(by_type) = by_type {
            by_type.warm.attempts.fetch_add(1, Ordering::Relaxed);
        }
        super::telemetry::warm_attempt(cache_kind, backend_kind, type_name);
    }

    pub fn record_warm_hit(
        &self,
        by_type: Option<&Self>,
        cache_kind: CacheMetricsKind,
        backend_kind: CacheBackendKind,
        type_name: &'static str,
    ) {
        self.warm.hits.fetch_add(1, Ordering::Relaxed);
        if let Some(by_type) = by_type {
            by_type.warm.hits.fetch_add(1, Ordering::Relaxed);
        }
        super::telemetry::warm_hit(cache_kind, backend_kind, type_name);
    }

    pub fn record_warm_insert(
        &self,
        by_type: Option<&Self>,
        cache_kind: CacheMetricsKind,
        backend_kind: CacheBackendKind,
        type_name: &'static str,
        bytes: u64,
    ) {
        self.warm.loads_started.fetch_add(1, Ordering::Relaxed);
        self.warm.loads_succeeded.fetch_add(1, Ordering::Relaxed);
        add_bytes(&self.warm.load_bytes, bytes);
        if let Some(by_type) = by_type {
            by_type.warm.loads_started.fetch_add(1, Ordering::Relaxed);
            by_type.warm.loads_succeeded.fetch_add(1, Ordering::Relaxed);
            add_bytes(&by_type.warm.load_bytes, bytes);
        }
        super::telemetry::warm_load(cache_kind, backend_kind, type_name, "success");
        super::telemetry::warm_load_bytes(cache_kind, backend_kind, type_name, bytes);
    }

    pub fn record_warm_load_bytes(
        &self,
        by_type: Option<&Self>,
        cache_kind: CacheMetricsKind,
        backend_kind: CacheBackendKind,
        type_name: &'static str,
        bytes: u64,
    ) {
        add_bytes(&self.warm.load_bytes, bytes);
        if let Some(by_type) = by_type {
            add_bytes(&by_type.warm.load_bytes, bytes);
        }
        super::telemetry::warm_load_bytes(cache_kind, backend_kind, type_name, bytes);
    }

    pub fn record_warm_error(
        &self,
        by_type: Option<&Self>,
        cache_kind: CacheMetricsKind,
        backend_kind: CacheBackendKind,
        type_name: &'static str,
    ) {
        self.warm.errors.fetch_add(1, Ordering::Relaxed);
        if let Some(by_type) = by_type {
            by_type.warm.errors.fetch_add(1, Ordering::Relaxed);
        }
        super::telemetry::warm_error(cache_kind, backend_kind, type_name);
    }

    pub fn start_load<'a>(
        &'a self,
        by_type: Option<&'a Self>,
        cache_kind: CacheMetricsKind,
        backend_kind: CacheBackendKind,
        type_name: &'static str,
        origin: CacheLoadOrigin,
    ) -> LoadGuard<'a> {
        self.started.fetch_add(1, Ordering::Relaxed);
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        if let Some(by_type) = by_type {
            by_type.started.fetch_add(1, Ordering::Relaxed);
            by_type.in_flight.fetch_add(1, Ordering::Relaxed);
        }
        if origin == CacheLoadOrigin::Warm {
            self.warm.loads_started.fetch_add(1, Ordering::Relaxed);
            if let Some(by_type) = by_type {
                by_type.warm.loads_started.fetch_add(1, Ordering::Relaxed);
            }
        }
        #[cfg(feature = "metrics")]
        super::telemetry::load_started(cache_kind, backend_kind, type_name);
        #[cfg(not(feature = "metrics"))]
        let _ = type_name;
        LoadGuard {
            counters: self,
            by_type,
            start: Some(Instant::now()),
            cache_kind,
            backend_kind,
            origin,
            type_name,
        }
    }
}

pub(super) struct LoadGuard<'a> {
    counters: &'a ActivityCounters,
    by_type: Option<&'a ActivityCounters>,
    // Taking the start time marks completion without a separate flag. Keeping
    // this guard small also limits the size of skipped boxed loader futures.
    start: Option<Instant>,
    cache_kind: CacheMetricsKind,
    backend_kind: CacheBackendKind,
    origin: CacheLoadOrigin,
    type_name: &'static str,
}

impl LoadGuard<'_> {
    pub fn finish(mut self, is_success: bool) {
        let (count, duration) = if is_success {
            (&self.counters.succeeded, &self.counters.success_duration)
        } else {
            (&self.counters.failed, &self.counters.error_duration)
        };
        if self.origin == CacheLoadOrigin::Warm {
            let warm_count = if is_success {
                &self.counters.warm.loads_succeeded
            } else {
                &self.counters.warm.loads_failed
            };
            warm_count.fetch_add(1, Ordering::Relaxed);
            if let Some(by_type) = self.by_type {
                let by_type_count = if is_success {
                    &by_type.warm.loads_succeeded
                } else {
                    &by_type.warm.loads_failed
                };
                by_type_count.fetch_add(1, Ordering::Relaxed);
            }
            super::telemetry::warm_load(
                self.cache_kind,
                self.backend_kind,
                self.type_name,
                if is_success { "success" } else { "error" },
            );
        }
        self.record(
            count,
            duration,
            if is_success { "success" } else { "error" },
        );
    }

    fn record(&mut self, count: &AtomicU64, duration: &AtomicU64, outcome: &'static str) {
        let Some(start) = self.start.take() else {
            return;
        };
        let elapsed = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
        count.fetch_add(1, Ordering::Relaxed);
        duration.fetch_add(elapsed, Ordering::Relaxed);
        if let Some(by_type) = self.by_type {
            let (by_type_count, by_type_duration) = match outcome {
                "success" => (&by_type.succeeded, &by_type.success_duration),
                "error" => (&by_type.failed, &by_type.error_duration),
                _ => unreachable!("load guard completion outcome"),
            };
            by_type_count.fetch_add(1, Ordering::Relaxed);
            by_type_duration.fetch_add(elapsed, Ordering::Relaxed);
            by_type.in_flight.fetch_sub(1, Ordering::Relaxed);
        }
        self.counters.in_flight.fetch_sub(1, Ordering::Relaxed);
        #[cfg(feature = "metrics")]
        super::telemetry::load_finished(
            self.cache_kind,
            self.backend_kind,
            self.type_name,
            outcome,
            elapsed,
        );
        #[cfg(not(feature = "metrics"))]
        super::telemetry::load_finished(self.cache_kind, self.backend_kind, "", outcome, elapsed);
    }
}

impl Drop for LoadGuard<'_> {
    fn drop(&mut self) {
        let Some(start) = self.start.take() else {
            return;
        };
        let elapsed = start.elapsed().as_nanos().try_into().unwrap_or(u64::MAX);
        self.counters.cancelled.fetch_add(1, Ordering::Relaxed);
        if self.origin == CacheLoadOrigin::Warm {
            self.counters
                .warm
                .loads_cancelled
                .fetch_add(1, Ordering::Relaxed);
            if let Some(by_type) = self.by_type {
                by_type.warm.loads_cancelled.fetch_add(1, Ordering::Relaxed);
            }
            super::telemetry::warm_load(
                self.cache_kind,
                self.backend_kind,
                self.type_name,
                "cancelled",
            );
        }
        self.counters
            .cancelled_duration
            .fetch_add(elapsed, Ordering::Relaxed);
        if let Some(by_type) = self.by_type {
            by_type.cancelled.fetch_add(1, Ordering::Relaxed);
            by_type
                .cancelled_duration
                .fetch_add(elapsed, Ordering::Relaxed);
            by_type.in_flight.fetch_sub(1, Ordering::Relaxed);
        }
        self.counters.in_flight.fetch_sub(1, Ordering::Relaxed);
        #[cfg(feature = "metrics")]
        super::telemetry::load_finished(
            self.cache_kind,
            self.backend_kind,
            self.type_name,
            "cancelled",
            elapsed,
        );
        #[cfg(not(feature = "metrics"))]
        super::telemetry::load_finished(
            self.cache_kind,
            self.backend_kind,
            "",
            "cancelled",
            elapsed,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_activity_registry_bounds_names_and_preserves_totals() {
        let capacity = 2;
        let registry = TypeActivityRegistry::with_capacity(
            capacity,
            CacheMetricsKind::Other,
            CacheBackendKind::Quick,
        );
        let mut aggregate = CacheActivity::default();
        for index in 0..=capacity {
            let type_name = Box::leak(format!("test.Type{index}").into_boxed_str());
            let slot = registry.get(type_name);
            registry.record_hit(slot);
            #[cfg(feature = "metrics")]
            registry.note_event(slot, CacheMetricsKind::Other, CacheBackendKind::Quick);
            aggregate.hits += 1;
        }

        let (activity, overflow_events) = registry.snapshot(&aggregate);
        assert_eq!(
            activity
                .iter()
                .map(|activity| activity.activity.hits)
                .sum::<u64>(),
            capacity as u64 + 1
        );
        assert!(
            activity
                .iter()
                .any(|activity| activity.type_name == "other")
        );
        #[cfg(feature = "metrics")]
        assert!(overflow_events >= 1);
        #[cfg(not(feature = "metrics"))]
        assert_eq!(overflow_events, 0);
    }
}
