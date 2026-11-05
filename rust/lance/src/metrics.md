Lance publishes metrics through the [`metrics`](https://docs.rs/metrics) crate
facade. Install any recorder (Prometheus, OpenTelemetry, etc.) in your
application and Lance will emit into it; when no recorder is installed, emission
is a cheap no-op. Metrics are only emitted when Lance is built with the
`metrics` feature.

## Object store metrics

These track I/O against the underlying object store. The `base` label
identifies the store; its cardinality is controlled by the
`LANCE_OBJECT_STORE_METRICS_LABEL` environment variable:

- `scheme` (default) — the scheme only (`s3`, `gs`, `az`, `file`, `memory`);
  low, bounded cardinality.
- `full` — the store's unique prefix (`s3$my-bucket`, `az$container@account`
  where Azure's account also matters), so multiple buckets on the same cloud
  can be told apart. Cardinality grows with the number of stores accessed.
- `dataset` — `base` as in `full`, plus a `dataset` label carrying the URI the
  store was opened for (`s3://my-bucket/path/table.lance`), so requests can be
  attributed to a dataset. In this mode each URI gets its own object store and
  HTTP client instead of sharing one per bucket, while the AIMD throttle budget
  stays shared per bucket. Cardinality grows with the number of datasets opened.
- `off` — omit the `base` label entirely.

`operation` is one of `get`, `put`, `put_part`, `head`, `list`, `delete`,
`copy`, `rename`, `create_multipart`, `complete_multipart`, or
`abort_multipart`. A multipart upload reports `create_multipart`, one
`put_part` per part, and then `complete_multipart` or `abort_multipart`.

Request counts are per logical operation: a `list` or `delete` that spans many
objects is one request, matching how backends batch them.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `lance_object_store_requests_total` | counter | `operation`, `base` | Object store requests issued. |
| `lance_object_store_request_bytes_total` | counter | `operation`, `base` | Bytes transferred by `get`/`put` requests. A `get` is counted once its response body has been fully read. |
| `lance_object_store_request_duration_seconds` | histogram | `operation`, `base` | Per-request latency, in seconds. For `get` this covers the full body transfer, not just time-to-first-byte. |
| `lance_object_store_errors_total` | counter | `operation`, `base` | Requests that returned an error. |
| `lance_object_store_in_flight_requests` | gauge | `operation`, `base` | Requests currently in flight. |
| `lance_object_store_throttle_total` | counter | `status`, `base` | Throttle responses (HTTP 429 / 503) seen at the HTTP layer, counted per attempt including retries. The `status` label is the numeric HTTP status. |
| `lance_object_store_retryable_responses_total` | counter | `status`, `base` | Retryable responses (HTTP 5xx / 429 / 408) seen at the HTTP layer, counted per attempt including retries. A superset of `throttle_total`; 409 (conflict) is excluded so commit conflicts are not counted. |

`lance_object_store_throttle_total` and
`lance_object_store_retryable_responses_total` are recorded only for the native
cloud stores (S3, GCS, Azure); Opendal-backed stores bypass the HTTP client
where the counters are installed, so they report the other object store metrics
but not throttle/retryable counts.

## Scan metrics

The scan node mirrors its `decode_time` and `task_wait_time` plan timers as
cumulative counters so an exporter can read the elapsed time while a scan is
still running. The counters hold nanoseconds, and intervals are recorded when
they complete or are dropped. Consequently, `rate() / 1e9` estimates the
average number of measured decode or task-wait intervals in progress per
second; in-flight intervals contribute only after they are recorded.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `lance_scan_decode_time_nanoseconds_total` | counter | `base` (+ `dataset` in dataset mode) | Wall time spent decoding scan tasks, summed over concurrent tasks. |
| `lance_scan_task_wait_time_nanoseconds_total` | counter | `base` (+ `dataset` in dataset mode) | Time scan partitions spent waiting for scheduled I/O. |

The labels follow `LANCE_OBJECT_STORE_METRICS_LABEL`; `dataset` is also present
in dataset mode. Both counters use the primary dataset store's labels even when
fragments are read through additional bases, so they cannot be joined
one-for-one with secondary-base I/O metrics.

## Cache metrics

Cache activity metrics use bounded labels. `cache` is `index`, `metadata`, or
`other`; `backend` is `quick`, `moka`, or `custom`. Activity series always carry
a stable cache-key `type`; sum over this label for aggregate activity.
Lance retains at most 64 exported type names process-wide; later names use
`type="other"` and increment `lance_cache_type_overflow_total`. Physical backend
metrics do not have a `cache` label because multiple logical cache wrappers can
share one physical pool.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `lance_cache_lookups_total` | counter | `cache`, `backend`, `outcome`, `type` | Lookups whose outcome is `hit` or `miss`. A caller that shares another caller's load is a hit because its loader was skipped. |
| `lance_cache_lookup_errors_total` | counter | `cache`, `backend`, `reason`, `type` | Lookup failures. Known reasons are `load` and `type_mismatch`. |
| `lance_cache_loads_total` | counter | `cache`, `backend`, `outcome`, `type` | Loaders that actually ran, with outcome `success`, `error`, or `cancelled`. Waiters sharing a load do not increment this counter. |
| `lance_cache_loads_in_flight` | gauge | `cache`, `backend`, `type` | Loaders currently executing. |
| `lance_cache_load_duration_seconds` | histogram | `cache`, `backend`, `outcome`, `type` | Time spent executing loaders. It excludes time in callers whose loaders were skipped. |
| `lance_cache_coalesced_loads_total` | counter | `backend` | Reserved for backends that can prove a caller received another loader's result. Neither built-in backend emits samples. |
| `lance_cache_warm_attempts_total` | counter | `cache`, `backend`, `type` | Cache calls made through explicit prewarm-origin handles. |
| `lance_cache_warm_hits_total` | counter | `cache`, `backend`, `type` | Prewarm calls served without new materialization. |
| `lance_cache_warm_loads_total` | counter | `cache`, `backend`, `outcome`, `type` | Prewarm materializations, including direct insertions, by outcome. |
| `lance_cache_warm_load_bytes_total` | counter | `cache`, `backend`, `type` | Accounted bytes successfully materialized by prewarming. |
| `lance_cache_warm_errors_total` | counter | `cache`, `backend`, `type` | Failed explicit prewarm cache calls. |
| `lance_cache_type_overflow_total` | counter | `cache`, `backend` | Activity events assigned to `type="other"` after the exported type-name bound is reached. |
| `lance_cache_write_attempts_total` | counter | `backend` | Physical write submissions, including replacements and candidates that were not admitted. |
| `lance_cache_write_bytes_total` | counter | `backend` | Accounted weight submitted by write attempts. |
| `lance_cache_entry_size_bytes` | histogram | `backend` | Accounted weight of submitted entries. |
| `lance_cache_size_removals_total` | counter | `backend` | Size-policy removal notifications. Quick emits them by default; Moka emits them only when its size-removal listener is enabled. Callbacks can include both resident victims and candidates that were never admitted. |
| `lance_cache_size_removed_bytes_total` | counter | `backend` | Accounted weight in size-policy removal notifications, with the same backend support as `lance_cache_size_removals_total`. |
| `lance_cache_write_rejections_total` | counter | `backend`, `reason` | Positively identified rejected writes. Known reasons are `disabled` and `lost_placeholder`. |
| `lance_cache_bypasses_total` | counter | `backend`, `reason` | Insertions skipped before a write attempt. The known reason is `disabled`. |
| `lance_cache_capacity_bytes` | gauge | `backend` | Configured capacity summed across distinct live physical pools. |
| `lance_cache_size_bytes` | gauge | `backend` | Approximate accounted occupancy summed across distinct live physical pools. |
| `lance_cache_entries` | gauge | `backend` | Approximate resident entry count summed across distinct live physical pools. |

The Python and Java OpenTelemetry bridges register the descriptions, histogram
bounds, and occupancy refresh automatically. A Rust application using another
recorder can use `lance::cache::histogram_bounds()` while configuring its
recorder, call `lance::cache::describe_metrics()` after installing the recorder,
and call `lance::cache::refresh_metrics()` from its collection loop. Refreshing
uses cheap backend accounting and does not scan entries. It deduplicates shared
pools by process-local identity, removes expired weak registrations, and reports
zero after the last pool in a previously reported backend group is dropped.
Custom backends without a pool identity or cheap accounting are excluded from
these gauges.

Event counters and histograms begin when a recorder is installed. Native cache
diagnostics retain lifetime counts even when the recorder is installed later.
The gauges report current state at refresh time, so they do not need event
history. Summing the `type` series yields aggregate activity.

Resident-only reads, reads without promotion, and selected-row reads count as
native diagnostics and exported lookups. Residency and tier probes (`peek_*`)
do not record activity. The legacy `CacheStats` and `index_cache_hit_rate`
remain limited to `get`, `get_unsized`, and `get_or_insert`.

Cache byte metrics are policy weights. They include key accounting and can count
shared allocations once per entry. They do not represent process RSS, heap bytes
freed, or the deep-size result returned by `Session::size_bytes()`. Moka's cheap
occupancy may also lag until backend maintenance runs.

Standard Moka reports `coalesced_loads` and both size-removal fields as unsupported
by default. Enable the latter with
`MokaCacheBackend::builder(capacity).with_size_removal_metrics().build()` or the
`size_removal_metrics=true` Moka URI option when the extra listener work is
acceptable. Neither built-in backend provides an exact coalesced-load count.

A high utilization value alone does not diagnose a cache problem. A full cache
with a high warm-hit rate can be healthy. Interpret occupancy together with
misses, executed loads, load duration, and size-removal churn. Frequent size
removals with repeated loads indicate capacity or admission pressure. A disabled
cache reports zero capacity and may report rejected writes or bypasses. An entry
can also be too large for policy admission while aggregate occupancy remains
low; compare the entry-size histogram with the configured capacity.

## Memory metrics

These report the memory of the Lance process. They are off by default: set `LANCE_MEMORY_OBSERVE` to `1`, `true`, `on` or `yes` to turn them on. Unset, empty, `0`, `false`, `off` or `no` keeps them off, and any other value makes the observed operations fail with an invalid-input error. Lance reads the variable once, the first time it needs it, so set it before the process starts. Readings come from `/proc/self/status` and the process's cgroup, so they are available on Linux only; elsewhere, and in builds without the `metrics` feature, setting the variable logs a warning and reports nothing.

The gauges are read when the recorder collects. The Python and Java OpenTelemetry bridges refresh them before every snapshot. A Rust application using another recorder can use `lance::memory::histogram_bounds()` while configuring it, call `lance::memory::describe_metrics()` after installing it, and call `lance::memory::refresh_metrics()` from its collection loop. A gauge whose source is unavailable, such as the limit of a cgroup that has none, is not reported, and a gauge keeps its last value if its source disappears later.

The peak histograms cover index builds started through `create_index` (or `CreateIndexBuilder`) and the inverted index branch of `merge_index_metadata`. While a build runs, a background thread samples the resident set size (RSS) every 100 ms, and every stage of the build, as well as the build as a whole, reports the largest value seen. The readings taken when a stage starts and ends count for every stage open at the time, so a build's peak is never below the peaks of its stages. A spike between two readings is only caught when it raises the peak RSS the kernel keeps for the process (`VmHWM`), so the values are lower bounds. Index builds that run as part of `optimize_indices` or of a MemWAL flush are not reported yet.

An operation peak is recorded only when the build completes successfully. A build that fails or is cancelled before that records no operation peak, and a stage that never completes records no stage peak. Stages that completed earlier in the same build keep their samples.

Peaks describe the whole process, so work running at the same time, including other index builds, counts towards them. For most scalar indices the `load_data` stage only prepares the scan: reading, sorting and training happen after it and show up in the operation peak. `operation` is `create_index` or `merge_index_metadata`, and `index_type` is the lowercase type name (`btree`, `inverted`, `ivf_pq`, ...). The `index_type` and `stage` labels each keep at most 64 distinct values per process; later values are reported as `other`.

The process gauges are best-effort readings of the live process. They cannot be exported after the process exits or is killed, so they do not provide a postmortem for a hard out-of-memory kill.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `lance_memory_rss_bytes` | gauge | none | Resident set size of the process, in bytes. |
| `lance_memory_rss_peak_bytes` | gauge | none | Largest resident set size of the process since it started, in bytes. |
| `lance_memory_cgroup_bytes` | gauge | none | Memory charged to the cgroup that sets the tightest limit on the process, or to its own cgroup when no group sets one (`memory.current`, or `memory.usage_in_bytes` on cgroup v1), in bytes. Unlike RSS it includes page cache and files written to tmpfs, such as spill files under a tmpfs `/tmp`. |
| `lance_memory_cgroup_limit_bytes` | gauge | none | Tightest memory limit over the process's cgroup and its ancestors, in bytes. |
| `lance_memory_stage_peak_bytes` | histogram | `operation`, `index_type`, `stage` | Peak RSS during each stage of an index build, in bytes. |
| `lance_memory_operation_peak_bytes` | histogram | `operation`, `index_type` | Peak RSS during each index build, in bytes. |
