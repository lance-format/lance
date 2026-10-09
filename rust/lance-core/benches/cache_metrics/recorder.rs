// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! A collecting recorder with the same registry and fixed-bucket aggregation
//! strategy as the bindings. Raw histogram observations never accumulate.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use metrics::{
    Counter, Gauge, Histogram, HistogramFn, Key, KeyName, Metadata, Recorder, SharedString, Unit,
};
use metrics_util::registry::{Registry, Storage};

const DURATION_BOUNDS: &[f64] = &[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0];
const SIZE_BOUNDS: &[f64] = &[
    64.0,
    1024.0,
    16384.0,
    262144.0,
    4194304.0,
    67108864.0,
    1073741824.0,
];

struct Buckets {
    bounds: &'static [f64],
    counts: Box<[AtomicU64]>,
    count: AtomicU64,
    sum: AtomicU64,
}

impl HistogramFn for Buckets {
    fn record(&self, value: f64) {
        let index = self.bounds.partition_point(|bound| *bound < value);
        self.counts[index].fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        let mut current = self.sum.load(Ordering::Relaxed);
        loop {
            let next = (f64::from_bits(current) + value).to_bits();
            match self.sum.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current = actual,
            }
        }
    }
}

struct BucketStorage;

impl Storage<Key> for BucketStorage {
    type Counter = Arc<AtomicU64>;
    type Gauge = Arc<AtomicU64>;
    type Histogram = Arc<Buckets>;

    fn counter(&self, _: &Key) -> Self::Counter {
        Arc::new(AtomicU64::new(0))
    }
    fn gauge(&self, _: &Key) -> Self::Gauge {
        Arc::new(AtomicU64::new(0))
    }
    fn histogram(&self, key: &Key) -> Self::Histogram {
        let bounds = if key.name().ends_with("_bytes") {
            SIZE_BOUNDS
        } else {
            DURATION_BOUNDS
        };
        Arc::new(Buckets {
            bounds,
            counts: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            count: AtomicU64::new(0),
            sum: AtomicU64::new(0),
        })
    }
}

struct CollectingRecorder(Registry<Key, BucketStorage>);

impl Recorder for CollectingRecorder {
    fn describe_counter(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_gauge(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn describe_histogram(&self, _: KeyName, _: Option<Unit>, _: SharedString) {}
    fn register_counter(&self, key: &Key, _: &Metadata<'_>) -> Counter {
        self.0
            .get_or_create_counter(key, |counter| Counter::from_arc(counter.clone()))
    }
    fn register_gauge(&self, key: &Key, _: &Metadata<'_>) -> Gauge {
        self.0
            .get_or_create_gauge(key, |gauge| Gauge::from_arc(gauge.clone()))
    }
    fn register_histogram(&self, key: &Key, _: &Metadata<'_>) -> Histogram {
        self.0
            .get_or_create_histogram(key, |histogram| Histogram::from_arc(histogram.clone()))
    }
}

pub fn install() {
    metrics::set_global_recorder(CollectingRecorder(Registry::new(BucketStorage))).unwrap();
}
