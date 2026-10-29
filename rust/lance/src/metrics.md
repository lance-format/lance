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

## Memory metrics

These report the memory of the Lance process. They are off by default: set `LANCE_MEMORY_OBSERVE` to `1`, `true`, `on` or `yes` to turn them on. Unset, empty, `0`, `false`, `off` or `no` keeps them off, and any other value makes the observed operations fail with an invalid-input error. Lance reads the variable once, the first time it needs it, so set it before the process starts. Readings come from `/proc/self/status` and the process's cgroup, so they are available on Linux only; elsewhere, and in builds without the `metrics` feature, setting the variable logs a warning and reports nothing.

The gauges are read when the recorder collects. The Python and Java OpenTelemetry bridges refresh them before every snapshot. A Rust application using another recorder can use `lance::memory::histogram_bounds()` while configuring it, call `lance::memory::describe_metrics()` after installing it, and call `lance::memory::refresh_metrics()` from its collection loop. A gauge whose source is unavailable, such as the limit of a cgroup that has none, is not reported, and a gauge keeps its last value if its source disappears later.

The peak histograms cover index builds started through `create_index` (or `CreateIndexBuilder`) and the inverted index branch of `merge_index_metadata`. While a build runs, a background thread samples the resident set size (RSS) every 100 ms, and every stage of the build, as well as the build as a whole, reports the largest value seen. The readings taken when a stage starts and ends count for every stage open at the time, so a build's peak is never below the peaks of its stages. A spike between two readings is only caught when it raises the peak RSS the kernel keeps for the process (`VmHWM`), so the values are lower bounds. Index builds that run as part of `optimize_indices` or of a MemWAL flush are not reported yet.

Peaks describe the whole process, so work running at the same time, including other index builds, counts towards them. For most scalar indices the `load_data` stage only prepares the scan: reading, sorting and training happen after it and show up in the operation peak. `operation` is `create_index` or `merge_index_metadata`, and `index_type` is the lowercase type name (`btree`, `inverted`, `ivf_pq`, ...). The `index_type` and `stage` labels each keep at most 64 distinct values per process; later values are reported as `other`.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `lance_memory_rss_bytes` | gauge | none | Resident set size of the process, in bytes. |
| `lance_memory_rss_peak_bytes` | gauge | none | Largest resident set size of the process since it started, in bytes. |
| `lance_memory_cgroup_bytes` | gauge | none | Memory charged to the cgroup that sets the tightest limit on the process, or to its own cgroup when no group sets one (`memory.current`, or `memory.usage_in_bytes` on cgroup v1), in bytes. Unlike RSS it includes page cache and files written to tmpfs, such as spill files under a tmpfs `/tmp`. |
| `lance_memory_cgroup_limit_bytes` | gauge | none | Tightest memory limit over the process's cgroup and its ancestors, in bytes. |
| `lance_memory_stage_peak_bytes` | histogram | `operation`, `index_type`, `stage` | Peak RSS during each stage of an index build, in bytes. |
| `lance_memory_operation_peak_bytes` | histogram | `operation`, `index_type` | Peak RSS during each index build, in bytes. |
