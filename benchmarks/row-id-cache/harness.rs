// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

// Read-only measurement harness for immutable baseline and candidate builds.
use arrow_array::UInt64Array;
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use lance::index::DatasetIndexExt;
use lance::{
    Dataset,
    dataset::{ProjectionRequest, builder::DatasetBuilder},
};
use lance_core::{cache::CacheStats, utils::address::RowAddress};
use lance_table::rowids::RowIdIndex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Weak},
    time::Instant,
};

const SOURCE_REVISION: &str = env!("ROW_ID_BENCH_REVISION");

static ROW_ID_WORK: std::sync::Mutex<Vec<Value>> = std::sync::Mutex::new(Vec::new());

struct RowIdTelemetry;

impl tracing::Subscriber for RowIdTelemetry {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target().starts_with("lance::dataset::rowids")
            && metadata.fields().field("event").is_some()
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        tracing::span::Id::from_u64(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(serde_json::Map<String, Value>);
        impl tracing::field::Visit for Fields {
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                self.0.insert(field.name().into(), value.into());
            }
            fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
                self.0.insert(field.name().into(), value.into());
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
            fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
                self.0.insert(field.name().into(), value.into());
            }
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0
                    .insert(field.name().into(), format!("{value:?}").into());
            }
        }
        let mut fields = Fields(serde_json::Map::new());
        event.record(&mut fields);
        if matches!(
            fields.0.get("event").and_then(Value::as_str),
            Some(
                "row_id_point_scan"
                    | "row_id_index_build"
                    | "row_id_fragment_resolution"
                    | "row_id_fragment_promotion_fallback"
                    | "row_id_fragment_batch_promotion"
            )
        ) {
            ROW_ID_WORK.lock().unwrap().push(Value::Object(fields.0));
        }
    }
}

fn take_row_id_work() -> Vec<Value> {
    std::mem::take(&mut *ROW_ID_WORK.lock().unwrap())
}

async fn resolve(
    dataset: &Dataset,
    ids: &[u64],
) -> (Vec<Option<RowAddress>>, Option<Weak<RowIdIndex>>) {
    let addresses = lance::dataset::rowids::resolve_row_ids(dataset, ids, true)
        .await
        .unwrap()
        .unwrap();
    (addresses, None)
}

#[derive(Clone, Serialize, Deserialize)]
struct Keys {
    ids: Vec<u64>,
    addrs: Vec<u64>,
}
#[derive(Deserialize)]
struct Oracle {
    uri: String,
    version: u64,
    rows: usize,
    random: Vec<Keys>,
    local: Vec<Keys>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Config {
    name: String,
    engine: String,
    #[serde(default)]
    mode: String,
    #[serde(default)]
    exact_repeat_after_cold: bool,
    #[serde(default)]
    distinct_prefix_calls: usize,
    #[serde(default)]
    after_point_bulk: usize,
    #[serde(default)]
    resolve_before_bulk: bool,
    #[serde(default = "memory_limit")]
    memory_limit_bytes: u64,
    #[serde(default)]
    concurrency: usize,
    #[serde(default)]
    throughput_waves: usize,
    query_file: String,
    path: String,
    locality: String,
    batch: usize,
    metadata_bytes: usize,
    index_bytes: usize,
    expected_row_index_bytes: usize,
    prewarm_batches: usize,
    warm_calls: usize,
    nonresident_calls: usize,
    round: usize,
}

fn memory_limit() -> u64 {
    16 * 1024 * 1024 * 1024
}

fn process_stats() -> Value {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap();
    let fields: Vec<&str> = stat[stat.rfind(')').unwrap() + 2..]
        .split_whitespace()
        .collect();
    let ticks = fields[11].parse::<u64>().unwrap() + fields[12].parse::<u64>().unwrap();
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let bytes = |name: &str| {
        status
            .lines()
            .find(|line| line.starts_with(name))
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<u64>()
            .unwrap()
            * 1024
    };
    json!({"cpu_ticks": ticks, "rss_bytes": bytes("VmRSS:"), "peak_rss_bytes": bytes("VmHWM:")})
}

fn cache_stats(stats: CacheStats) -> Value {
    json!({"hits":stats.hits,"misses":stats.misses,"entries":stats.num_entries,
        "charged_resident_bytes":stats.size_bytes})
}

fn cgroup_memory() -> Value {
    let group = std::fs::read_to_string("/proc/self/cgroup").unwrap();
    let relative = group
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .unwrap();
    let root = format!("/sys/fs/cgroup{relative}");
    let read = |name: &str| {
        std::fs::read_to_string(format!("{root}/{name}"))
            .ok()
            .map(|value| value.trim().to_string())
    };
    json!({"path":relative,"memory_max":read("memory.max"),
        "memory_peak":read("memory.peak"),"memory_current":read("memory.current"),
        "swap_max":read("memory.swap.max")})
}

async fn caches(dataset: &Dataset, config: &Config) -> Value {
    let session = dataset.session();
    json!({
        "metadata":cache_stats(session.metadata_cache_stats().await),
        "index":cache_stats(session.index_cache_stats().await),
        "metadata_capacity_bytes":session.file_metadata_cache().capacity_bytes(),
        "metadata_reported_max_entry_bytes":session.file_metadata_cache().max_entry_bytes(),
        "index_configured_capacity_bytes":config.index_bytes,
    })
}

fn payload(key: u64, lane: u64) -> u64 {
    key.wrapping_mul(0x9e3779b97f4a7c15)
        .rotate_left((lane * 11) as u32)
        ^ lane
}

async fn query(dataset: &Dataset, keys: &Keys, path: &str) -> (u128, Option<Weak<RowIdIndex>>) {
    let start = Instant::now();
    if path == "resolver" {
        let (addresses, weak) = resolve(dataset, &keys.ids).await;
        let elapsed = start.elapsed().as_nanos();
        assert!(addresses.iter().all(Option::is_some));
        assert!(
            addresses
                .iter()
                .map(|address| u64::from(address.unwrap()))
                .eq(keys.addrs.iter().copied())
        );
        (elapsed, weak)
    } else {
        assert_eq!(path, "take");
        let batch = dataset
            .take_rows(&keys.ids, ProjectionRequest::from(dataset.schema().clone()))
            .await
            .unwrap();
        let elapsed = start.elapsed().as_nanos();
        let actual = batch
            .column_by_name("key")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(actual.values().as_ref(), keys.ids.as_slice());
        for lane in 0..4 {
            let values = batch
                .column_by_name(&format!("p{lane}"))
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            for (position, &key) in keys.ids.iter().enumerate() {
                assert_eq!(values.value(position), payload(key, lane));
            }
        }
        (elapsed, None)
    }
}

async fn io_stores(dataset: &Dataset) -> Vec<Arc<lance_io::object_store::ObjectStore>> {
    let mut base_ids = vec![None];
    for fragment in dataset.fragments().iter() {
        for file in &fragment.files {
            if !base_ids.contains(&file.base_id) {
                base_ids.push(file.base_id);
            }
        }
        if let Some(file) = &fragment.deletion_file {
            if !base_ids.contains(&file.base_id) {
                base_ids.push(file.base_id);
            }
        }
    }
    let mut stores: Vec<Arc<lance_io::object_store::ObjectStore>> = Vec::new();
    for base_id in base_ids {
        let store = dataset.object_store(base_id).await.unwrap();
        // ObjectStore clones share both their tracked inner store and IOTracker.
        // An imported base can resolve to the primary binding; count it once.
        if !stores
            .iter()
            .any(|other| Arc::ptr_eq(&other.inner, &store.inner))
        {
            stores.push(store);
        }
    }
    stores
}

fn io_totals(
    stores: &[Arc<lance_io::object_store::ObjectStore>],
) -> lance_io::utils::tracking_store::IoStats {
    let mut total = lance_io::utils::tracking_store::IoStats::default();
    for store in stores {
        let stats = store.io_stats_snapshot();
        total.read_iops += stats.read_iops;
        total.read_bytes += stats.read_bytes;
        total.write_iops += stats.write_iops;
        total.written_bytes += stats.written_bytes;
    }
    total
}

async fn sample(
    dataset: &Dataset,
    keys: &Keys,
    config: &Config,
    phase: &str,
    call: usize,
) -> Value {
    let stores = io_stores(dataset).await;
    let cache_before = caches(dataset, config).await;
    let before = io_totals(&stores);
    let process_before = process_stats();
    take_row_id_work();
    let (elapsed, weak) = query(dataset, keys, &config.path).await;
    let row_id_work = take_row_id_work();
    let after = io_totals(&stores);
    let process_after = process_stats();
    // A Weak reference cannot keep the whole map alive or refresh cache recency.
    let rowid_resident = weak.as_ref().map(|value| value.strong_count() > 0);
    let cache_after = caches(dataset, config).await;
    let value = json!({"type":"sample","phase":phase,"call":call,"latency_ns":elapsed,
        "read_iops":after.read_iops-before.read_iops,"read_bytes":after.read_bytes-before.read_bytes,
        "io_store_count":stores.len(),"io_accounting":"all referenced data/deletion bases plus primary; shared tracked stores counted once",
        "cache_before":cache_before,"cache_after":cache_after,"rowid_alive_after_query":rowid_resident,
        "before":process_before,"after":process_after,"row_id_work":row_id_work,"correct":true});
    println!("{value}");
    value
}

async fn profile(config: Config) {
    let memory = cgroup_memory();
    assert_eq!(
        memory["memory_max"].as_str(),
        Some(config.memory_limit_bytes.to_string().as_str()),
        "all paired arms require the configured common cgroup limit"
    );
    assert_eq!(memory["swap_max"].as_str(), Some("0"));
    let oracle: Oracle =
        serde_json::from_slice(&std::fs::read(&config.query_file).unwrap()).unwrap();
    assert!(oracle.uri.starts_with("s3://") || oracle.uri.starts_with("gs://"));
    assert!(config.batch > 0 && config.batch <= 1000);
    assert!(config.prewarm_batches > 0 && config.warm_calls > 0 && config.nonresident_calls > 0);
    let pool = if config.locality == "random" {
        &oracle.random
    } else {
        assert_eq!(config.locality, "local");
        &oracle.local
    };
    assert!(config.prewarm_batches <= pool.len());
    let keys = |position: usize| {
        let source = &pool[position % config.prewarm_batches];
        Keys {
            ids: source.ids[..config.batch].to_vec(),
            addrs: source.addrs[..config.batch].to_vec(),
        }
    };
    let start = Instant::now();
    let dataset = DatasetBuilder::from_uri(&oracle.uri)
        .with_version(oracle.version)
        .with_metadata_cache_size_bytes(config.metadata_bytes)
        .with_index_cache_size_bytes(config.index_bytes)
        .load()
        .await
        .unwrap();
    let open_ns = start.elapsed().as_nanos();
    assert_eq!(dataset.version().version, oracle.version);
    let store = dataset.object_store(None).await.unwrap();
    let open_io = store.io_stats_snapshot();
    println!(
        "{}",
        json!({"type":"config","source":SOURCE_REVISION,"engine":config.engine,"config":config,
        "uri":oracle.uri,"version":oracle.version,"rows":oracle.rows,"fragments":dataset.fragments().len(),
        "manifest_path":dataset.manifest_location().path.to_string(),"manifest_etag":dataset.manifest_location().e_tag,
        "open_ns":open_ns,"open_read_iops":open_io.read_iops,"open_read_bytes":open_io.read_bytes,
        "initial_cache":caches(&dataset,&config).await,"initial_process":process_stats(),
        "cgroup_memory":memory})
    );
    let first = sample(&dataset, &keys(0), &config, "cold", 0).await;
    let nonresident = if config.engine != "rowid" {
        false
    } else if let Some(resident) = first["rowid_alive_after_query"].as_bool() {
        !resident
    } else {
        // Full take does not return an index handle. Inspect tagged residency
        // after the cold timed call instead of assuming bytes per row.
        let diagnostics = dataset
            .session()
            .cache_diagnostics_by_type(lance_core::cache::CacheSnapshotMode::Approximate)
            .await;
        !diagnostics
            .index
            .by_type
            .unwrap()
            .occupancy
            .unwrap()
            .types
            .iter()
            .any(|entry| {
                matches!(
                    entry.type_name.as_str(),
                    "RowIdIndex"
                        | "CachedRowIdIndex"
                        | "RowIdPointResolution"
                        | "PreparedRowIdFragment"
                        | "MergedRowIdGroup"
                ) && entry.num_entries > 0
            })
    };
    let mut warmup_ns = first["latency_ns"].as_u64().unwrap() as u128;
    let mut warmup_reads = first["read_iops"].as_u64().unwrap();
    let mut warmup_bytes = first["read_bytes"].as_u64().unwrap();
    if config.exact_repeat_after_cold {
        let repeat = sample(&dataset, &keys(0), &config, "exact_repeat", 0).await;
        warmup_ns += repeat["latency_ns"].as_u64().unwrap() as u128;
        warmup_reads += repeat["read_iops"].as_u64().unwrap();
        warmup_bytes += repeat["read_bytes"].as_u64().unwrap();
    }
    let primes = if config.distinct_prefix_calls > 0 {
        config.distinct_prefix_calls
    } else if nonresident {
        1
    } else {
        config.prewarm_batches
    };
    for position in 1..primes {
        let value = sample(&dataset, &keys(position), &config, "prime", position).await;
        warmup_ns += value["latency_ns"].as_u64().unwrap() as u128;
        warmup_reads += value["read_iops"].as_u64().unwrap();
        warmup_bytes += value["read_bytes"].as_u64().unwrap();
    }
    println!(
        "{}",
        json!({"type":"warmup","queries":primes+usize::from(config.exact_repeat_after_cold),"query_time_ns":warmup_ns,
        "read_iops":warmup_reads,"read_bytes":warmup_bytes,"cold_resolution_entry_observed":!nonresident,
        "cold_residency_scope":"tagged entries; does not establish complete prepared coverage",
        "cache":caches(&dataset,&config).await,"process":process_stats()})
    );
    let calls = if config.distinct_prefix_calls > 0 {
        config.warm_calls
    } else if nonresident {
        config.nonresident_calls
    } else {
        config.warm_calls
    };
    for position in 0..calls {
        sample(&dataset, &keys(position), &config, "warm", position).await;
    }
    if config.concurrency > 0 && !nonresident {
        let dataset = Arc::new(dataset.clone());
        let stores = io_stores(&dataset).await;
        let before = io_totals(&stores);
        let start = Instant::now();
        let mut latencies = Vec::new();
        for wave in 0..config.throughput_waves {
            let mut tasks = Vec::new();
            for lane in 0..config.concurrency {
                let dataset = dataset.clone();
                let config = config.clone();
                let request = keys(wave * config.concurrency + lane);
                tasks.push(tokio::spawn(async move {
                    query(&dataset, &request, &config.path).await.0
                }));
            }
            for task in tasks {
                latencies.push(task.await.unwrap());
            }
        }
        let elapsed = start.elapsed().as_nanos();
        let after = io_totals(&stores);
        println!(
            "{}",
            json!({"type":"throughput","concurrency":config.concurrency,
            "calls":latencies.len(),"elapsed_ns":elapsed,"latencies_ns":latencies,
            "read_iops":after.read_iops-before.read_iops,"read_bytes":after.read_bytes-before.read_bytes,
            "cache":caches(&dataset,&config).await,"process":process_stats(),"correct":true})
        );
    }
    if config.after_point_bulk > 0 {
        let count = config.after_point_bulk;
        let source = &oracle.random[0];
        let bulk_keys = Keys {
            ids: source.ids[..count].to_vec(),
            addrs: source.addrs[..count].to_vec(),
        };
        if config.resolve_before_bulk {
            let mut resolver_config = config.clone();
            resolver_config.path = "resolver".into();
            resolver_config.batch = count;
            sample(
                &dataset,
                &bulk_keys,
                &resolver_config,
                "point_to_bulk_resolver",
                0,
            )
            .await;
        }
        sample(&dataset, &bulk_keys, &config, "point_to_bulk", 0).await;
        sample(&dataset, &keys(0), &config, "after_bulk_point", 0).await;
    }
    let workload_process = process_stats();
    let workload_cgroup = cgroup_memory();
    let final_cache = caches(&dataset, &config).await;
    // Walk cache allocations only after recording the workload peak RSS.
    let session_deep_bytes = dataset.session().size_bytes();
    println!(
        "{}",
        json!({"type":"summary","source":SOURCE_REVISION,"engine":config.engine,
        "cold_queries":1,"exact_repeat_queries":usize::from(config.exact_repeat_after_cold),"prime_queries":primes-1,"warm_queries":calls,
        "post_point_bulk_queries":usize::from(config.after_point_bulk>0),
        "post_point_resolver_queries":usize::from(config.after_point_bulk>0 && config.resolve_before_bulk),
        "post_bulk_point_queries":usize::from(config.after_point_bulk>0),"correct":true,
        "cache":final_cache,"workload_process":workload_process,"workload_cgroup":workload_cgroup,"session_deep_bytes":session_deep_bytes,
        "after_deep_size_diagnostic":process_stats()})
    );
    println!(
        "{}",
        json!({"type":"dataset_metadata",
        "indices":dataset.load_indices().await.unwrap().iter().map(|index|json!({"name":index.name,
            "uuid":index.uuid,"fields":index.fields,"index_version":index.index_version,
            "coverage":index.fragment_bitmap.as_ref().map(|bitmap|bitmap.len())})).collect::<Vec<_>>(),
        "fragments":dataset.fragments()})
    );
}

#[derive(Clone, Serialize, Deserialize)]
struct ChurnSnapshot {
    version: u64,
    fragments: usize,
    ids: Vec<u64>,
    addresses: Vec<Option<u64>>,
    values: Vec<Option<[u64; 5]>>,
}

#[derive(Serialize, Deserialize)]
struct ChurnOracle {
    uri: String,
    snapshots: Vec<ChurnSnapshot>,
}

async fn churn_query(dataset: &Dataset, oracle: &ChurnSnapshot, path: &str) -> u128 {
    let start = Instant::now();
    if path == "resolver" {
        let result = lance::dataset::rowids::resolve_row_ids(dataset, &oracle.ids, true)
            .await
            .unwrap()
            .unwrap();
        let elapsed = start.elapsed().as_nanos();
        assert_eq!(
            result.iter().map(|v| v.map(u64::from)).collect::<Vec<_>>(),
            oracle.addresses
        );
        elapsed
    } else {
        assert_eq!(path, "take");
        let result = dataset
            .take_rows(
                &oracle.ids,
                ProjectionRequest::from(dataset.schema().clone()),
            )
            .await
            .unwrap();
        let elapsed = start.elapsed().as_nanos();
        let expected: Vec<_> = oracle.values.iter().flatten().collect();
        assert_eq!(result.num_rows(), expected.len());
        for (row, values) in expected.iter().enumerate() {
            for (column, name) in ["key", "p0", "p1", "p2", "p3"].iter().enumerate() {
                assert_eq!(
                    result[*name].as_primitive::<UInt64Type>().value(row),
                    values[column]
                );
            }
        }
        elapsed
    }
}

async fn churn_sample(
    dataset: &Dataset,
    oracle: &ChurnSnapshot,
    config: &Config,
    phase: &str,
    call: usize,
) {
    let stores = io_stores(dataset).await;
    let before = io_totals(&stores);
    let process_before = process_stats();
    take_row_id_work();
    let elapsed = churn_query(dataset, oracle, &config.path).await;
    let after = io_totals(&stores);
    println!(
        "{}",
        json!({"type":"sample","phase":phase,"call":call,"version":oracle.version,"fragments":oracle.fragments,"latency_ns":elapsed,"read_bytes":after.read_bytes-before.read_bytes,"read_iops":after.read_iops-before.read_iops,"row_id_work":take_row_id_work(),"cache_after":caches(dataset,config).await,"before":process_before,"after":process_stats(),"correct":true,"cgroup":cgroup_memory()})
    );
}

async fn churn(config: Config) {
    let memory = cgroup_memory();
    assert_eq!(
        memory["memory_max"].as_str(),
        Some(config.memory_limit_bytes.to_string().as_str())
    );
    assert_eq!(memory["swap_max"].as_str(), Some("0"));
    let oracle: ChurnOracle =
        serde_json::from_slice(&std::fs::read(&config.query_file).unwrap()).unwrap();
    assert!(oracle.snapshots.len() > 1);
    let first = DatasetBuilder::from_uri(&oracle.uri)
        .with_version(oracle.snapshots[0].version)
        .with_metadata_cache_size_bytes(config.metadata_bytes)
        .with_index_cache_size_bytes(config.index_bytes)
        .load()
        .await
        .unwrap();
    let mut readers = Vec::new();
    for (cycle, snapshot) in oracle.snapshots.iter().enumerate() {
        let start = Instant::now();
        let dataset = first.checkout_version(snapshot.version).await.unwrap();
        println!(
            "{}",
            json!({"type":"checkout","cycle":cycle,"version":snapshot.version,"latency_ns":start.elapsed().as_nanos()})
        );
        churn_sample(&dataset, snapshot, &config, "version_first", cycle).await;
        for call in 0..4 {
            churn_sample(&dataset, snapshot, &config, "version_warm", call).await;
        }
        readers.push(dataset);
    }
    // Dataset snapshots share one bounded session; no index Arc is held.
    for call in 0..3 {
        for (dataset, snapshot) in readers.iter().zip(&oracle.snapshots) {
            churn_sample(dataset, snapshot, &config, "snapshot_roundtrip", call).await;
        }
    }
    let last = Arc::new(readers.last().unwrap().clone());
    let snapshot = Arc::new(oracle.snapshots.last().unwrap().clone());
    let mut completed = 0;
    let start = Instant::now();
    for _ in 0..config.throughput_waves {
        let tasks: Vec<_> = (0..config.concurrency)
            .map(|_| {
                let dataset = last.clone();
                let snapshot = snapshot.clone();
                let path = config.path.clone();
                tokio::spawn(async move { churn_query(&dataset, &snapshot, &path).await })
            })
            .collect();
        for result in futures::future::join_all(tasks).await {
            result.unwrap();
        }
        completed += config.concurrency;
    }
    let elapsed = start.elapsed().as_nanos();
    println!(
        "{}",
        json!({"type":"summary","source":SOURCE_REVISION,"correct":true,"mode":"churn","snapshots":readers.len(),"explicit_old_index_arc_held":false,"throughput_calls":completed,"throughput_wall_ns":elapsed,"cache":caches(&last,&config).await,"process":process_stats(),"cgroup":cgroup_memory()})
    );
}

fn main() {
    tracing::subscriber::set_global_default(RowIdTelemetry).unwrap();
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 2, "one config file is required");
    let config: Config = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    assert_eq!(config.engine, "rowid");
    assert!(matches!(config.mode.as_str(), "" | "profile" | "churn"));
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(16)
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            if config.mode == "churn" {
                churn(config).await
            } else {
                profile(config).await
            }
        });
}
