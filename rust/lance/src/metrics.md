Lance publishes metrics through the [`metrics`](https://docs.rs/metrics) crate
facade. Install any recorder (Prometheus, OpenTelemetry, etc.) in your
application and Lance will emit into it; when no recorder is installed, emission
is a cheap no-op. Metrics are only emitted when Lance is built with the
`metrics` feature.

## Execution diagnostics

Execution summaries and structured reports do not require the `metrics` feature
or an installed process recorder. `ExecutionSummaryCounts` keeps its existing
fields, count/time maps, and callback contract. The additive `spill_count()`,
`spilled_bytes()`, and `spilled_rows()` accessors return `None` when no operator
registered that measurement and `Some(0)` for a registered zero.

`lance_datafusion::exec::analyze_plan_report` drains an execution batch by batch
and returns its formatted plan, independent execution wall time, outcome, summary,
and detailed operator measurements. Execution failures produce a partial report
with error text. Planning/configuration errors return an error. The caller owns
any write side effects and transaction cleanup. Do not execute the same physical
plan again to compare reports: nodes can be one-shot or retain cumulative metrics.

Detailed measurements retain the operator name, child-index path from the root,
partition, labels, and metric kind. Counts are additive only when their producers
describe the same quantity. Gauges remain separate. `PeakMemoryUsage` is exposed
as an operator peak in bytes; summing operator or partition peaks does not produce
a query peak. Custom nonnumeric metrics, timestamps, and unsupported readings are
omitted, not replaced by zero.

| Measurement | Unit | Scope and aggregation |
| --- | --- | --- |
| `iops`, `requests`, `bytes_read` | operations, requests, bytes | Existing attributed query I/O; sums instrumented producers. Separate from process object-store totals. |
| `index_cache_hits`, `index_cache_misses` | page lookups | Query boundaries already instrumented. A hit includes sharing a successful in-flight load; streaming bypasses count as misses. |
| ANN/FTS stage timers and operator `elapsed_compute` | nanoseconds in maps | Stage/operator sums can overlap and include waits; never sum these into query latency. |
| `spill_count`, `spilled_bytes`, `spilled_rows` | files, bytes, rows | DataFusion operator spill variants, summed once. Excludes scratch and replay producers. |
| `current_mem_used`, operator peak memory | bytes | Individual producer gauges/peaks, preserved without aggregation. Names depend on the operator. |
| `pool_peak_reserved_bytes` | bytes | Fresh operation pool's maximum simultaneous DataFusion reservations. Not RSS, cache policy weight, allocator retention, or all Arrow allocations. |
| `merge_write_time` | nanoseconds | Inclusive lazy merge sink execution, including upstream waits and writes. On the imperative scalar-index route, inclusive orchestration including interleaved planning/staging. Overlaps join timers. |

The default structured analyzer creates a fresh `PeakRecordingPool` around the
configured DataFusion pool. It preserves memory limits and starts no process
sampler. The context-taking analyzer reports the supplied pool's lifetime peak
when it is a `PeakRecordingPool`; otherwise the reading is absent. That reading
is exclusive to an operation only when its context is operation-owned.

`execute_plan_with_diagnostics` provides an opt-in terminal callback: once on
successful exhaustion, first stream error, or early drop. Error and cancellation
snapshots are partial and can precede background task shutdown. The existing scan
statistics callback still runs only on exhaustion. Callback code must not panic.

For cold/warm query diagnosis, compare separate executions under the same session:
check attributed I/O and page cache lookups, then inspect operator stage times.
HNSW graph/codebook lookups remain outside the existing cache counters. Physical
bitmap batch `output_rows` is not a logical candidate count. Search-specific
comparisons retain their producer-specific meaning; no per-document FTS counters
are added.

## Merge-insert analysis

`MergeInsertJob::analyze_plan_report`, `analyze_plan_report_batches`, and
`analyze_plan_report_provider` return `MergeInsertAnalysis`. The Python
`MergeInsertBuilder.analyze_plan_report` returns a typed dictionary with the same
units and availability. Java's `Dataset.analyzeMergeInsert` accepts an Arrow stream
and returns a protobuf-backed `MergeInsertAnalysis` with typed accessors. The wire
message is runtime transport and is never stored in a dataset. Existing text APIs
retain their signatures and plan-only eligibility.
Materialized input retains its statistics; streaming and deduplicated input use
their actual execution routes. Join strategy, partition mode, and physical join
type come from the physical nodes, without parsing plan text.
The report's `operation_id` is also attached to the `merge_insert_analysis`
tracing span. It is never exported as a process metric label.

Analysis executes writes without committing a dataset version, then deletes
uncommitted data/deletion files and managed blob payloads. Cleanup failures are
logged and left for orphan cleanup. Dropping the analysis future can interrupt
cleanup. Logical change statistics are absent after failed execution. Data-file
bytes exclude deletion files, blob payloads, and all spill producers. Touched
fragments count existing fragments modified or removed, excluding new fragments.
Structured reporting also supports the current scalar-index writer when every
join column is indexed and `when_not_matched_by_source=Keep`. It runs the existing
uncommitted writer, preserving source routing, first-seen deduplication, and
resource limits. Legacy writers and unsupported configurations return
`NotSupported`.

The root plan has identity zero. `subplans` captures separately executed plans
before releasing them: `index_probe`, `indexed_source_sort`, `indexed_join`, and
`fragment_update_sort`.
Each has an operation-local nonzero `id`, purpose, execution snapshot, and joins.
Operator child paths are local to that identity; a join's `subplan_id` resolves
its path. Indexed orchestration has no root operator tree; its physical nodes
live in subplans. Completed, failed, and dropped streams each finalize once.

| Report field | Meaning and availability |
| --- | --- |
| `operator_spills` | Root/nested operators and the indexed-source materialization barrier's DataFusion spill counts, rows, and bytes, each producer counted once. Read bytes/times are absent unless registered. |
| `replay_spills` | Operation-owned Arrow replay files for source normalization, mapped targets, and fragment-update staging; includes IPC metadata and repeated reads. An entirely buffered operation reports zero. |
| `scratch_spills` | Absent for current merge routes, which allocate no scratch-store files. Other scratch users require an allocation-owner scope; session snapshot subtraction is never used. |
| `target_rows_read`, `fragments_read` | Logical target-scan visits, excluding native source scans, including self-merges. Repeated reads count again; these are not unique target/fragment cardinalities. Absent without an attributable producer. |
| `index_used` | True when the scalar probe ran, false for an observed non-indexed execution, absent when failure prevented observation. |
| `join_index_segments` | Distinct segment UUIDs selected for opening by actual probes, including cache-served opens. Zero for a measured non-indexed execution; absent when index use is unavailable. |
| `join_duration` / `join_duration_ns` | Actual join-node times. Overlap the inclusive `merge_write_duration` / `merge_write_duration_ns`; neither is an exclusive phase measurement. |
| `joined_rows` | Join output before merge actions, including repeated matches, when registered. |
| `mapped_target_rows` | Distinct target addresses emitted by index mapping; an upper bound on matching rows, absent without a mapper. Bitmap payload batch rows are never substituted. |

Rust durations are `Duration`; Python and Java durations use integer nanoseconds.
Byte fields are integer bytes. Optional fields preserve absent versus measured
zero. Replay and nested measurements are operation-owned even when executions
share caches and disk quotas. Snapshots follow stream finalization and best-effort
uncommitted-file cleanup; outstanding I/O and delayed reclamation may advance
lifetime counters after a failed or cancelled snapshot. Pool peaks measure only
DataFusion reservations and must not be interpreted as process memory.

```python
report = dataset.merge_insert("id").when_matched_update_all().analyze_plan_report(source)
if report["outcome"] == "complete":
    print(report["write_mode"], report["stats"], report["fragments_touched"])
    print(report["replay_spills"]["bytes_written"], report["operator_spills"]["bytes_written"])
else:
    print(report["error"])
    # Complete change statistics are unavailable; partial producer readings remain useful.
    print(report["replay_spills"], report["subplans"])
```

```java
MergeInsertAnalysis report = dataset.analyzeMergeInsert(params, sourceStream);
if (report.getExecution().getOutcome() == MergeInsertAnalysis.Outcome.FAILED) {
  report.getExecution().getError().ifPresent(System.err::println);
}
report.getReplaySpills().getBytesWritten().ifPresent(System.out::println);
```

For index maintenance, use the existing stage progress callbacks and traces
alongside scratch usage. Progress totals keep their original units. Process
RSS/cgroup observation and session cache diagnostics remain owned by their
existing upstream projects; these diagnostics introduce no competing sampler.

## Scratch spills

`LocalSpillStore::metrics().snapshot()` reports cumulative traffic and physical
lifetime accounting, with measured zero before the first spill. To attribute
concurrent jobs sharing a store, allocate a `LocalSpillStore::scope()` per job.
Scopes share the directory, unique file allocator, and authoritative quota, but
have independent accounting. Parent metrics cover only allocations through that
parent; subtracting snapshots of a session-shared store is not attribution.

Accepted bytes include buffered writes. Successfully transferred write/read
bytes are separate; repeated replay of scratch readers counts repeated reads.
Charged bytes include accepted writes that fail or are aborted, and are released
only after successful deletion (or confirmed absence). Failed cleanup retains
both the charge and live-file count through retries and directory cleanup.
Retain the metrics handle after the operation to observe delayed reclamation.
Snapshots are observational and not atomic across fields. Peak charged bytes
measure retained accepted storage, excluding transient unaccepted reservations.

With the `metrics` feature and a recorder, the following bounded process
aggregates are exported. Python/Java OpenTelemetry bridges register their units.
They have no operation/dataset labels. The duration histogram has only the
bounded `operation=read|write` label.

| Metric | Type | Unit |
| --- | --- | --- |
| `lance_spill_accepted_bytes_total` | counter | bytes |
| `lance_spill_write_bytes_total` | counter | bytes |
| `lance_spill_read_bytes_total` | counter | bytes |
| `lance_spill_files_created_total` | counter | paths allocated |
| `lance_spill_charged_bytes` | gauge | bytes retained |
| `lance_spill_live_files` | gauge | paths not yet reclaimed |
| `lance_spill_quota_failures_total` | counter | quota rejections |
| `lance_spill_cleanup_failures_total` | counter | deletion attempts failed |
| `lance_spill_io_duration_seconds` | histogram | seconds of local I/O, excluding blocking-pool queue wait |

Call `lance_io::spill::describe_metrics()` after installing a Rust recorder.
These scratch metrics exclude DataFusion operator spills and Arrow replay
spills. `SpillReceiver::metrics()` and `spilling_table_provider_with_metrics`
expose a separate `ReplaySpillUsage` scope for Arrow replay files. It measures
successful underlying read/write bytes (including IPC metadata and read-ahead),
spill file count, and I/O durations. An entirely buffered source reports measured
zero; repeated scans count repeated reads. Replay storage shares the job's disk
manager/quota where configured, without duplicating its physical quota accounting.
Do not add dataset object-store
snapshots, scratch totals, and operator counters as if they were one exact query
I/O total.

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

These report the memory of the Lance process. They are off by default: set `LANCE_MEMORY_OBSERVE` to `1`, `true`, `on` or `yes` to turn them on. Unset, empty, `0`, `false`, `off` or `no` keeps them off, and any other value makes the observed operations fail with an invalid-input error and reports no memory metric. Lance reads the variable once, the first time it needs it, so set it before the process starts. The process and cgroup gauges and the peaks come from `/proc/self/status` and the process's cgroup, so they are available on Linux only; elsewhere, setting the variable logs a warning and reports only `lance_memory_component_bytes`. Builds without the `metrics` feature report no memory gauge or histogram. The same switch turns on trace events on the `lance::memory` target, which need neither a recorder nor the `metrics` feature; the Memory Events section of the performance guide describes them.

The gauges are read when the recorder collects. The Python and Java OpenTelemetry bridges refresh them before every snapshot. A Rust application using another recorder can use `lance::memory::histogram_bounds()` while configuring it, call `lance::memory::describe_metrics()` after installing it, and call `lance::memory::refresh_metrics()` from its collection loop. A process or cgroup gauge whose source is unavailable, such as the limit of a cgroup that has none, is not reported, and keeps its last value if its source disappears later.

The peak histograms cover index builds started through `create_index` (or `CreateIndexBuilder`) and the inverted index branch of `merge_index_metadata`. While a build runs, a background thread samples the resident set size (RSS) every 100 ms, and every stage of the build, as well as the build as a whole, reports the largest value seen. The readings taken when a stage starts and ends count for every stage open at the time, so a build's peak is never below the peaks of its stages. A spike between two readings is only caught when it raises the peak RSS the kernel keeps for the process (`VmHWM`), so the values are lower bounds. Index builds that run as part of `optimize_indices` or of a MemWAL flush are not reported yet.

An operation peak is recorded only when the build completes successfully, which for a build that `create_index` commits is after its commit, so its peak covers the commit. A build that fails or is cancelled before that records no operation peak, and a stage that never completes records no stage peak. Stages that completed earlier in the same build keep their samples.

Peaks describe the whole process, so work running at the same time, including other index builds, counts towards them. For most scalar indices the `load_data` stage only prepares the scan: reading, sorting and training happen after it and show up in the operation peak. `operation` is `create_index` or `merge_index_metadata`, and `index_type` is the lowercase type name (`btree`, `inverted`, `ivf_pq`, ...). The `index_type` and `stage` labels each keep at most 64 distinct values per process; later values are reported as `other`.

The process gauges are best-effort readings of the live process. They cannot be exported after the process exits or is killed, so they do not provide a postmortem for a hard out-of-memory kill. The memory events are emitted as each build starts and completes, so a build that was killed shows as a start without a completion.

`lance_memory_component_bytes` reports how many bytes three parts of Lance hold right now, from their own accounting, on every platform. `io_scheduler` is what the I/O schedulers admitted for reading and their callers have not received yet, reads still in flight included. Each scheduler bounds its share by its own I/O buffer size, which is the `io_buffer_size` of the scan for a scan and 32 MiB per I/O thread for most other reads, such as takes and index loads. Reads admitted by priority can go past that bound, and reads that bypass backpressure, as well as schedulers without an I/O buffer size, are not counted. `datafusion_pool` is the reservations in the DataFusion memory pools that Lance builds from its execution options, with or without spilling, as scans and index builds use. Plans that run in a default DataFusion session are not counted yet: the joins of `merge_insert`, MemWAL scans and point lookups, `Dataset::sql`, sessions from the DataFusion namespace `SessionBuilder`, and sessions an application builds itself. `mem_wal` is what the MemWAL shard writers hold, as `ShardWriter::memory().retained_bytes()` reports it: the memtables not flushed yet or kept for `frozen_memtable_grace`, with their indexes, or the pending batches of a WAL-only writer. A writer registers with the gauge after `ShardWriter::open` replays the WAL, so the memory that replay uses is not sampled, while the memtables it rebuilds stay resident and count as soon as `open` returns. The value is read when the recorder collects, so an instance created and dropped between two collections is not seen. A component is not reported until Lance first creates one in the process, and reads 0 while none is alive. The sum of the three is what these parts account for, not a breakdown of the RSS: each part estimates its own buffers, and the I/O schedulers count reads before their bytes arrive. The session caches report their size separately, as `lance_cache_size_bytes` (see Cache metrics).

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `lance_memory_rss_bytes` | gauge | none | Resident set size of the process, in bytes. |
| `lance_memory_rss_peak_bytes` | gauge | none | Largest resident set size of the process since it started, in bytes. |
| `lance_memory_cgroup_bytes` | gauge | none | Memory charged to the cgroup that sets the tightest limit on the process, or to its own cgroup when no group sets one (`memory.current`, or `memory.usage_in_bytes` on cgroup v1), in bytes. Unlike RSS it includes page cache and files written to tmpfs, such as spill files under a tmpfs `/tmp`. |
| `lance_memory_cgroup_limit_bytes` | gauge | none | Tightest memory limit over the process's cgroup and its ancestors, in bytes. |
| `lance_memory_stage_peak_bytes` | histogram | `operation`, `index_type`, `stage` | Peak RSS during each stage of an index build, in bytes. |
| `lance_memory_operation_peak_bytes` | histogram | `operation`, `index_type` | Peak RSS during each index build, in bytes. |
| `lance_memory_component_bytes` | gauge | `component` | Bytes that a part of Lance holds right now: `io_scheduler`, `datafusion_pool` or `mem_wal`. |
