// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bridge from the [`metrics`] crate facade to Java OpenTelemetry.
//!
//! The Java layer owns the actual OpenTelemetry instruments. This module
//! installs the process-global Rust [`metrics::Recorder`], keeps cumulative
//! metric state, and exposes catalog/snapshot calls over JNI.

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Mutex, OnceLock, RwLock};

use jni::JNIEnv;
use jni::objects::{JClass, JObject, JValue};
use jni::sys::{jboolean, jobject};
use metrics::atomics::AtomicU64;
use metrics::{Counter, Gauge, Histogram, Key, KeyName, Metadata, Recorder, SharedString, Unit};
use metrics_util::registry::{Registry, Storage};

use crate::error::Result;

const DEFAULT_BOUNDS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];

#[derive(Clone, Copy)]
enum MetricKind {
    Counter,
    Gauge,
    Histogram,
}

impl MetricKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Counter => "counter",
            Self::Gauge => "gauge",
            Self::Histogram => "histogram",
        }
    }
}

struct MetricDescription {
    kind: MetricKind,
    unit: Option<String>,
    description: String,
}

static CATALOG: LazyLock<Mutex<HashMap<String, MetricDescription>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static HISTOGRAM_BOUNDS: LazyLock<RwLock<HashMap<String, Arc<[f64]>>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

static REGISTRY: OnceLock<Arc<Registry<Key, LanceStorage>>> = OnceLock::new();

fn bounds_for(name: &str) -> Arc<[f64]> {
    HISTOGRAM_BOUNDS
        .read()
        .unwrap()
        .get(name)
        .cloned()
        .unwrap_or_else(|| Arc::from(DEFAULT_BOUNDS))
}

struct BucketedHistogram {
    bounds: Arc<[f64]>,
    counts: Box<[AtomicU64]>,
    count: AtomicU64,
    sum_bits: AtomicU64,
}

impl BucketedHistogram {
    fn new(bounds: Arc<[f64]>) -> Self {
        let counts = (0..=bounds.len())
            .map(|_| AtomicU64::new(0))
            .collect::<Box<[_]>>();
        Self {
            bounds,
            counts,
            count: AtomicU64::new(0),
            sum_bits: AtomicU64::new(0),
        }
    }

    fn add_to_sum(&self, value: f64) {
        let mut current = self.sum_bits.load(Ordering::Relaxed);
        loop {
            let updated = (f64::from_bits(current) + value).to_bits();
            match self.sum_bits.compare_exchange_weak(
                current,
                updated,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }

    fn snapshot(&self) -> MetricValue {
        let mut cumulative = 0u64;
        let mut buckets = Vec::with_capacity(self.bounds.len() + 1);
        for (i, bound) in self.bounds.iter().enumerate() {
            cumulative += self.counts[i].load(Ordering::Relaxed);
            buckets.push((bound.to_string(), cumulative));
        }
        cumulative += self.counts[self.bounds.len()].load(Ordering::Relaxed);
        buckets.push(("+Inf".to_string(), cumulative));
        MetricValue::Histogram {
            buckets,
            count: self.count.load(Ordering::Relaxed),
            sum: f64::from_bits(self.sum_bits.load(Ordering::Relaxed)),
        }
    }
}

impl metrics::HistogramFn for BucketedHistogram {
    fn record(&self, value: f64) {
        let idx = self.bounds.partition_point(|&bound| bound < value);
        self.counts[idx].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.add_to_sum(value);
    }
}

struct LanceStorage;

impl Storage<Key> for LanceStorage {
    type Counter = Arc<AtomicU64>;
    type Gauge = Arc<AtomicU64>;
    type Histogram = Arc<BucketedHistogram>;

    fn counter(&self, _key: &Key) -> Self::Counter {
        Arc::new(AtomicU64::new(0))
    }

    fn gauge(&self, _key: &Key) -> Self::Gauge {
        Arc::new(AtomicU64::new(0))
    }

    fn histogram(&self, key: &Key) -> Self::Histogram {
        Arc::new(BucketedHistogram::new(bounds_for(key.name())))
    }
}

struct LanceRecorder {
    registry: Arc<Registry<Key, LanceStorage>>,
}

impl LanceRecorder {
    fn describe(
        &self,
        key: KeyName,
        kind: MetricKind,
        unit: Option<Unit>,
        description: SharedString,
    ) {
        CATALOG.lock().unwrap().insert(
            key.as_str().to_string(),
            MetricDescription {
                kind,
                unit: unit.map(|u| u.as_canonical_label().to_string()),
                description: description.into_owned(),
            },
        );
    }
}

impl Recorder for LanceRecorder {
    fn describe_counter(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.describe(key, MetricKind::Counter, unit, description);
    }

    fn describe_gauge(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.describe(key, MetricKind::Gauge, unit, description);
    }

    fn describe_histogram(&self, key: KeyName, unit: Option<Unit>, description: SharedString) {
        self.describe(key, MetricKind::Histogram, unit, description);
    }

    fn register_counter(&self, key: &Key, _metadata: &Metadata<'_>) -> Counter {
        self.registry
            .get_or_create_counter(key, |counter| Counter::from_arc(counter.clone()))
    }

    fn register_gauge(&self, key: &Key, _metadata: &Metadata<'_>) -> Gauge {
        self.registry
            .get_or_create_gauge(key, |gauge| Gauge::from_arc(gauge.clone()))
    }

    fn register_histogram(&self, key: &Key, _metadata: &Metadata<'_>) -> Histogram {
        self.registry
            .get_or_create_histogram(key, |histogram| Histogram::from_arc(histogram.clone()))
    }
}

fn register_bounds() {
    let mut bounds = HISTOGRAM_BOUNDS.write().unwrap();
    for (name, values) in lance_io::object_store::metrics::histogram_bounds() {
        bounds.insert((*name).to_string(), Arc::from(*values));
    }
}

fn describe_all() {
    lance_io::object_store::metrics::describe_metrics();
}

enum MetricValue {
    Scalar(f64),
    Histogram {
        buckets: Vec<(String, u64)>,
        count: u64,
        sum: f64,
    },
}

struct MetricPoint {
    name: String,
    kind: &'static str,
    attributes: HashMap<String, String>,
    value: MetricValue,
}

fn labels(key: &Key) -> HashMap<String, String> {
    key.labels()
        .map(|label| (label.key().to_string(), label.value().to_string()))
        .collect()
}

fn collect_points(registry: &Registry<Key, LanceStorage>) -> Vec<MetricPoint> {
    let mut points = Vec::new();
    for (key, handle) in registry.get_counter_handles() {
        points.push(MetricPoint {
            name: key.name().to_string(),
            kind: "counter",
            attributes: labels(&key),
            value: MetricValue::Scalar(handle.load(Ordering::Relaxed) as f64),
        });
    }
    for (key, handle) in registry.get_gauge_handles() {
        points.push(MetricPoint {
            name: key.name().to_string(),
            kind: "gauge",
            attributes: labels(&key),
            value: MetricValue::Scalar(f64::from_bits(handle.load(Ordering::Relaxed))),
        });
    }
    for (key, handle) in registry.get_histogram_handles() {
        points.push(MetricPoint {
            name: key.name().to_string(),
            kind: "histogram",
            attributes: labels(&key),
            value: handle.snapshot(),
        });
    }
    points
}

fn register_lance_metrics_recorder() -> bool {
    if REGISTRY.get().is_some() {
        return true;
    }
    let registry = Arc::new(Registry::new(LanceStorage));
    let recorder = LanceRecorder {
        registry: registry.clone(),
    };
    register_bounds();
    match metrics::set_global_recorder(recorder) {
        Ok(()) => {
            let _ = REGISTRY.set(registry);
            describe_all();
            true
        }
        Err(_) => false,
    }
}

fn object_list<'local>(env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
    Ok(env.new_object("java/util/ArrayList", "()V", &[])?)
}

fn add_to_list(env: &mut JNIEnv, list: &JObject, item: &JObject) -> Result<()> {
    env.call_method(
        list,
        "add",
        "(Ljava/lang/Object;)Z",
        &[JValue::Object(item)],
    )?;
    Ok(())
}

fn string_object<'local>(env: &mut JNIEnv<'local>, value: Option<&str>) -> Result<JObject<'local>> {
    match value {
        Some(value) => Ok(env.new_string(value)?.into()),
        None => Ok(JObject::null()),
    }
}

fn attributes_to_java<'local>(
    env: &mut JNIEnv<'local>,
    attributes: &HashMap<String, String>,
) -> Result<JObject<'local>> {
    let java_map = env.new_object("java/util/HashMap", "()V", &[])?;
    for (key, value) in attributes {
        let java_key = env.new_string(key)?;
        let java_value = env.new_string(value)?;
        env.call_method(
            &java_map,
            "put",
            "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;",
            &[JValue::Object(&java_key), JValue::Object(&java_value)],
        )?;
        env.delete_local_ref(java_key)?;
        env.delete_local_ref(java_value)?;
    }
    Ok(java_map)
}

fn buckets_to_java<'local>(
    env: &mut JNIEnv<'local>,
    buckets: Option<&[(String, u64)]>,
) -> Result<JObject<'local>> {
    let Some(buckets) = buckets else {
        return Ok(JObject::null());
    };
    let list = object_list(env)?;
    for (le, cumulative_count) in buckets {
        let le = env.new_string(le)?;
        let bucket = env.new_object(
            "org/lance/otel/MetricBucket",
            "(Ljava/lang/String;J)V",
            &[JValue::Object(&le), JValue::Long(*cumulative_count as i64)],
        )?;
        add_to_list(env, &list, &bucket)?;
        env.delete_local_ref(le)?;
        env.delete_local_ref(bucket)?;
    }
    Ok(list)
}

fn boxed_double<'local>(env: &mut JNIEnv<'local>, value: Option<f64>) -> Result<JObject<'local>> {
    match value {
        Some(value) => Ok(env.new_object("java/lang/Double", "(D)V", &[JValue::Double(value)])?),
        None => Ok(JObject::null()),
    }
}

fn boxed_long<'local>(env: &mut JNIEnv<'local>, value: Option<u64>) -> Result<JObject<'local>> {
    match value {
        Some(value) => {
            Ok(env.new_object("java/lang/Long", "(J)V", &[JValue::Long(value as i64)])?)
        }
        None => Ok(JObject::null()),
    }
}

fn metric_description_to_java<'local>(
    env: &mut JNIEnv<'local>,
    name: &str,
    desc: &MetricDescription,
) -> Result<JObject<'local>> {
    let name = env.new_string(name)?;
    let kind = env.new_string(desc.kind.as_str())?;
    let unit = string_object(env, desc.unit.as_deref())?;
    let description = env.new_string(&desc.description)?;
    Ok(env.new_object(
        "org/lance/otel/MetricDescription",
        "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V",
        &[
            JValue::Object(&name),
            JValue::Object(&kind),
            JValue::Object(&unit),
            JValue::Object(&description),
        ],
    )?)
}

fn metric_point_to_java<'local>(
    env: &mut JNIEnv<'local>,
    point: MetricPoint,
) -> Result<JObject<'local>> {
    let (value, buckets, count, sum) = match point.value {
        MetricValue::Scalar(value) => (Some(value), None, None, None),
        MetricValue::Histogram {
            buckets,
            count,
            sum,
        } => (None, Some(buckets), Some(count), Some(sum)),
    };
    let name = env.new_string(&point.name)?;
    let kind = env.new_string(point.kind)?;
    let attributes = attributes_to_java(env, &point.attributes)?;
    let value = boxed_double(env, value)?;
    let buckets = buckets_to_java(env, buckets.as_deref())?;
    let count = boxed_long(env, count)?;
    let sum = boxed_double(env, sum)?;
    Ok(env.new_object(
        "org/lance/otel/MetricPoint",
        "(Ljava/lang/String;Ljava/lang/String;Ljava/util/Map;Ljava/lang/Double;Ljava/util/List;Ljava/lang/Long;Ljava/lang/Double;)V",
        &[
            JValue::Object(&name),
            JValue::Object(&kind),
            JValue::Object(&attributes),
            JValue::Object(&value),
            JValue::Object(&buckets),
            JValue::Object(&count),
            JValue::Object(&sum),
        ],
    )?)
}

fn lance_metrics_catalog_native<'local>(env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
    let catalog = CATALOG.lock().unwrap();
    let list = object_list(env)?;
    for (name, desc) in catalog.iter() {
        env.with_local_frame(16, |env| {
            let item = metric_description_to_java(env, name, desc)?;
            add_to_list(env, &list, &item)
        })?;
    }
    Ok(list)
}

fn snapshot_lance_metrics_native<'local>(env: &mut JNIEnv<'local>) -> Result<JObject<'local>> {
    let list = object_list(env)?;
    if let Some(registry) = REGISTRY.get() {
        for point in collect_points(registry) {
            env.with_local_frame(64, |env| {
                let item = metric_point_to_java(env, point)?;
                add_to_list(env, &list, &item)
            })?;
        }
    }
    Ok(list)
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_otel_LanceMetrics_registerLanceMetricsRecorderNative(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    register_lance_metrics_recorder() as jboolean
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_otel_LanceMetrics_lanceMetricsCatalogNative(
    mut env: JNIEnv,
    _class: JClass,
) -> jobject {
    ok_or_throw_with_return!(
        env,
        lance_metrics_catalog_native(&mut env),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_org_lance_otel_LanceMetrics_snapshotLanceMetricsNative(
    mut env: JNIEnv,
    _class: JClass,
) -> jobject {
    ok_or_throw_with_return!(
        env,
        snapshot_lance_metrics_native(&mut env),
        std::ptr::null_mut()
    )
    .into_raw()
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::HistogramFn;

    fn bucket_count(buckets: &[(String, u64)], le: &str) -> u64 {
        buckets
            .iter()
            .find(|(bound, _)| bound == le)
            .map(|(_, count)| *count)
            .unwrap_or_else(|| panic!("no bucket with le={le}"))
    }

    #[test]
    fn test_bucketed_histogram_records_cumulative_buckets() {
        let histogram = BucketedHistogram::new(Arc::from([0.1f64, 1.0, 10.0].as_slice()));
        histogram.record(0.05);
        histogram.record(0.1);
        histogram.record(0.5);
        histogram.record(1.0);
        histogram.record(5.0);
        histogram.record(50.0);

        let MetricValue::Histogram {
            buckets,
            count,
            sum,
        } = histogram.snapshot()
        else {
            panic!("expected histogram");
        };

        assert_eq!(bucket_count(&buckets, "0.1"), 2);
        assert_eq!(bucket_count(&buckets, "1"), 4);
        assert_eq!(bucket_count(&buckets, "10"), 5);
        assert_eq!(bucket_count(&buckets, "+Inf"), 6);
        assert_eq!(bucket_count(&buckets, "+Inf"), count);
        assert!((sum - 56.65).abs() < 1e-9);
    }

    #[test]
    fn test_registry_aggregates_counters_and_gauges() {
        let registry = Arc::new(Registry::new(LanceStorage));
        let recorder = LanceRecorder {
            registry: registry.clone(),
        };
        metrics::with_local_recorder(&recorder, || {
            metrics::counter!("test_counter", "operation" => "get").increment(2);
            metrics::counter!("test_counter", "operation" => "get").increment(3);
            metrics::gauge!("test_gauge", "operation" => "get").increment(3.5);
            metrics::gauge!("test_gauge", "operation" => "get").decrement(1.25);
        });

        let points = collect_points(&registry);
        let counter = points
            .iter()
            .find(|point| point.name == "test_counter")
            .expect("counter recorded");
        let gauge = points
            .iter()
            .find(|point| point.name == "test_gauge")
            .expect("gauge recorded");

        assert!(matches!(counter.value, MetricValue::Scalar(value) if value == 5.0));
        assert!(matches!(gauge.value, MetricValue::Scalar(value) if (value - 2.25).abs() < 1e-9));
    }
}
