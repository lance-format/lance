# Lance Performance Guide

This guide provides tips and tricks for optimizing the performance of your Lance applications.

## Logging

Lance uses the `log` crate to log messages. Displaying these log messages will depend on the client
library you are using. For rust, you will need to configure a logging subscriber. For more details
ses the [log](https://docs.rs/log/latest/log/) docs. The Python and Java clients configure a default
logging subscriber that logs to stderr.

The Python/Java logger can be configured with several environment variables:

- `LANCE_LOG`: Controls log filtering based on log level and target. See the [env_logger](https://docs.rs/env_logger/latest/env_logger/) docs for more details. The `LANCE_LOG` environment variable replaces the `RUST_LOG` environment variable.
- `LANCE_TRACING`: Controls tracing filtering based on log level. Key tracing events described below are emitted at
  the `info` level. However, additional spans and events are available at the `debug` level which may be useful for
  debugging performance issues. The default tracing level is `info`.
- `LANCE_LOG_STYLE`: Controls whether colors are used in the log messages. Valid values are `auto`, `always`, `never`.
- `LANCE_LOG_TS_PRECISION`: The precision of the timestamp in the log messages. Valid values are `ns`, `us`, `ms`, `s`.
- `LANCE_LOG_FILE`: Redirects Rust log messages to the specified file path instead of stderr. When set, Lance will create the file and any necessary parent directories. If the file cannot be created (e.g., due to permission issues), Lance will fall back to logging to stderr.

## Trace Events

Lance uses tracing to log events. If you are running `pylance` then these events will be emitted
as log messages. For Rust connections you can use the `tracing` crate to capture these events.

Rust tracing targets are listed below. In `pylance` logs, trace events are emitted under a
`lance::events::` prefix so they can be filtered separately from normal log records. For example,
`LANCE_LOG="warn,lance::events::object_store::throttle=info"` shows storage throttling events
without enabling other Lance event logs.

### File Audit

File audit events are emitted when significant files are created or deleted.

| Event               | Parameter | Description                                                                |
| ------------------- | --------- | -------------------------------------------------------------------------- |
| `lance::file_audit` | `mode`    | The mode of I/O operation (create, delete, delete_unverified)              |
| `lance::file_audit` | `type`    | The type of file affected (manifest, data file, index file, deletion file) |

### Dataset Events

Dataset events are emitted when datasets are loaded, written, committed, deleted, compacted, or cleaned.

| Event                   | Parameter   | Description                                                               |
| ----------------------- | ----------- | ------------------------------------------------------------------------- |
| `lance::dataset_events` | `event`     | The dataset event type (loading, writing, committed, deleting, and others) |
| `lance::dataset_events` | `uri`       | The dataset URI                                                           |
| `lance::dataset_events` | `mode`      | The write mode                                                            |
| `lance::dataset_events` | `operation` | The committed transaction operation                                       |
| `lance::dataset_events` | `predicate` | The delete predicate                                                      |
| `lance::dataset_events` | `columns`   | The removed columns                                                       |

### Object Store Throttle Events

Object store throttle events are emitted when Lance observes cloud storage throttle responses and
reduces or retries request rates.

| Event                            | Parameter       | Description                              |
| -------------------------------- | --------------- | ---------------------------------------- |
| `lance::object_store::throttle`  | `previous_rate` | The request rate before AIMD adjustment  |
| `lance::object_store::throttle`  | `new_rate`      | The request rate after AIMD adjustment   |
| `lance::object_store::throttle`  | `attempt`       | The retry attempt for retry debug events |
| `lance::object_store::throttle`  | `error`         | The underlying object store throttle error      |

### I/O Events

I/O events are emitted when significant I/O operations are performed, particularly
those related to indices. These events are NOT emitted when the index is loaded from
the in-memory cache. Correct cache utilization is important for performance and these
events are intended to help you debug cache usage.

| Event              | Parameter | Description                                                                                          |
| ------------------ | --------- | ---------------------------------------------------------------------------------------------------- |
| `lance::io_events` | `type`    | The type of I/O operation (open_scalar_index, open_vector_index, load_vector_part, load_scalar_part) |

### Execution Events

Execution events are emitted when an execution plan is run. These events are useful for
debugging query performance.

| Event              | Parameter           | Description                                                    |
| ------------------ | ------------------- | -------------------------------------------------------------- |
| `lance::execution` | `type`              | The type of execution event (plan_run is the only type today)  |
| `lance::execution` | `output_rows`       | The number of rows in the output of the plan                   |
| `lance::execution` | `iops`              | The number of I/O operations performed by the plan             |
| `lance::execution` | `bytes_read`        | The number of bytes read by the plan                           |
| `lance::execution` | `indices_loaded`    | The number of indices loaded by the plan                       |
| `lance::execution` | `parts_loaded`      | The number of index partitions loaded by the plan              |
| `lance::execution` | `index_comparisons` | The number of comparisons performed inside the various indices |

## Threading Model

Lance is designed to be thread-safe and performant. Lance APIs can be called concurrently unless
explicitly stated otherwise. Users may create multiple tables and share tables between threads.
Operations may run in parallel on the same table, but some operations may lead to conflicts. For
details see [conflict resolution](../format/table/transaction.md/#conflict-resolution).

Most Lance operations will use multiple threads to perform work in parallel. There are two thread
pools in lance: the IO thread pool and the compute thread pool. The IO thread pool is used for
reading and writing data from disk. The compute thread pool is used for performing computations
on data. The number of threads in each pool can be configured by the user.

The IO thread pool is used for reading and writing data from disk. The number of threads in the IO
thread pool is determined by the object store that the operation is working with. Local object stores
will use 8 threads by default. Cloud object stores will use 64 threads by default. This is a fairly
conservative default and you may need 128 or 256 threads to saturate network bandwidth on some cloud
providers. The `LANCE_IO_THREADS` environment variable can be used to override the number of IO
threads. If you increase this variable you may also want to increase the `io_buffer_size`.

The compute thread pool is used for performing computations on data. The number of threads in the
compute thread pool is determined by the number of cores on the machine. The number of threads in
the compute thread pool can be overridden by setting the `LANCE_CPU_THREADS` environment variable.
This is commonly done when running multiple Lance processes on the same machine (e.g when working with
tools like Ray). Keep in mind that decoding data is a compute intensive operation, even if a workload
seems I/O bound (like scanning a table) it may still need quite a few compute threads to achieve peak
performance.

## Memory Requirements

Lance is designed to be memory efficient. Operations should stream data from disk and not require
loading the entire dataset into memory. However, there are a few components of Lance that can use
a lot of memory.

### Metadata Cache

Lance uses a metadata cache to speed up operations. This cache holds various pieces of metadata such
as file metadata, dataset manifests, etc. This cache is an LRU cache that is sized by bytes. The default
size is 1 GiB.

The metadata cache is not shared between tables by default. For best performance you should create a
single table and share it across your application. Alternatively, you can create a single session
and specify it when you open tables.

Keys are often a composite of multiple fields and all keys are scoped to the dataset URI. The following items are stored in the metadata cache:

| Item              | Key                                              | What is stored                      |
| ----------------- | ------------------------------------------------ | ----------------------------------- |
| Dataset Manifests | Dataset URI, version, and etag                   | The manifest for the dataset        |
| Transactions      | Dataset URI, version                             | The transaction for the dataset     |
| Deletion Files    | Dataset URI, fragment_id, version, id, file_type | The deletion vector for a frag      |
| Row Id Mask       | Dataset URI, version                             | The row id sequence for the dataset |
| Row Id Index      | Dataset URI, version                             | The row id index for the dataset    |
| Row Id Sequence   | Dataset URI, fragment_id, row_id_meta            | The row id sequence for a fragment  |
| Index Metadata    | Dataset URI, version                             | The index metadata for the dataset  |
| Index Details¹    | Dataset URI, index uuid                          | The index details for an index      |
| File Global Meta  | Dataset URI, file path                           | The global metadata for a file      |
| File Column Meta  | Dataset URI, file path, column index             | The search cache for a column       |

Notes:

1. This is only stored for very old indexes which don't store their details in the manifest.

### Index Cache

Lance uses an index cache to speed up queries. This caches vector and scalar indices in memory. The
max size of this cache can be configured when creating a `LanceDataset` using the `index_cache_size_bytes`
parameter. This cache is an LRU cached that is sized by bytes. The default size is 6 GiB.
You can view the size of this cache by inspecting the result of `dataset.session().size_bytes()`.

The index cache is not shared between tables. For best performance you should create a single table and
share it across your application.

**Note**: `index_cache_size` (specified in entries) was deprecated since version 0.30.0. Use
`index_cache_size_bytes` (specified in bytes) for new code.

### Scanning Data

Searches (e.g. vector search, full text search) do not use a lot of memory to hold data because they don't
typically return a lot of data. However, scanning data can use a lot of memory. Scanning is a streaming
operation but we need enough memory to hold the data that we are scanning. The amount of memory needed is
largely determined by the `io_buffer_size` and the `batch_size` variables.

Each I/O thread should have enough memory to buffer an entire page of data. Pages today are typically between
8 and 32 MB. This means, as a rule of thumb, you should generally have about 32MB of memory per I/O thread.
The default `io_buffer_size` is 2GB which is enough to buffer 64 pages of data. If you increase the number
of I/O threads you should also increase the `io_buffer_size`.

Scans will also decode data (and run any filtering or compute) in parallel on CPU threads. The amount of data
decoded at any one time is determined by the `batch_size` and the size of your rows. Each CPU thread will
need enough memory to hold one batch. Once batches are delivered to your application, they are no longer tracked
by Lance and so if memory is a concern then you should also be careful not to accumulate memory in your own
application (e.g. by running `to_table` or otherwise collecting all batches in memory.)

The default `batch_size` is 8192 rows. When you are working with mostly scalar data you want to keep batches
around 1MB and so the amount of memory needed by the compute threads is fairly small. However, when working with
large data you may need to turn down the `batch_size` to keep memory usage under control. For example, when
working with 1024-dimensional vector embeddings (e.g. 32-bit floats) then 8192 rows would be 32MB of data. If you
spread that across 16 CPU threads then you would need 512MB of compute memory per scan. You might find working
with 1024 rows per batch is more appropriate.

#### Tuning remote scans

An ordered dataset scan still overlaps I/O from multiple fragments. `scan_in_order=True` controls the order in
which batches are returned; it does not make fragment reads sequential. This is why a dataset scan can issue
more concurrent requests than scanning one fragment directly. The following controls tune different parts of
the scan:

* `fragment_readahead` limits how many fragments may have reads scheduled concurrently. Set it to `1` to match
  the fragment-level I/O pattern, then increase it if the storage connection has spare bandwidth.
* `LANCE_IO_THREADS` limits concurrent storage requests for the process. Cloud stores default to 64, which is
  intended for high-bandwidth, in-region access and can be too aggressive across regions or over the public
  internet.
* `io_buffer_size` limits buffered I/O bytes and applies backpressure when decoding falls behind.
* `batch_readahead` limits concurrent batch decoding. It does not control the size of storage range requests.

For a bandwidth-constrained remote connection, start with conservative settings and tune upward:

```shell
LANCE_IO_THREADS=8 python scan.py
```

```python
scanner = dataset.scanner(
    fragment_readahead=1,
    batch_readahead=2,
    io_buffer_size=64 * 1024 * 1024,
)
for batch in scanner.to_batches():
    process(batch)
```

Lance reads encoded pages from storage, so reducing `batch_size` changes the returned and decoded batch sizes
but may not reduce the initial range request. The first batch can require loading one encoded page for each
selected column.

In summary, scans could use up to `(2 * io_buffer_size) + (batch_size * num_compute_threads)` bytes of memory.
Keep in mind that `io_buffer_size` is a soft limit (e.g. we cannot read less than one page at a time right now)
and so it is not necessarily a bug if you see memory usage exceed this limit by a small margin.

### Cloud Store Throttling

Cloud object stores (S3, GCS, Azure) are automatically wrapped with an AIMD (Additive Increase / Multiplicative
Decrease) rate limiter. When the store returns throttle errors (HTTP 429/503), the request rate decreases
multiplicatively. During sustained success, the rate increases additively. This applies to all operations
(reads, writes, deletes, lists) and replaces the old `LANCE_PROCESS_IO_THREADS_LIMIT` process-wide cap.

Local and in-memory stores are **not** throttled.

The AIMD throttle can be tuned via storage options or environment variables. Storage options take precedence
over environment variables:

| Setting            | Storage Option Key              | Env Var                         | Default |
| ------------------ | ------------------------------- | ------------------------------- | ------- |
| Initial rate       | `lance_aimd_initial_rate`       | `LANCE_AIMD_INITIAL_RATE`       | 2000    |
| Min rate           | `lance_aimd_min_rate`           | `LANCE_AIMD_MIN_RATE`           | 1       |
| Max rate           | `lance_aimd_max_rate`           | `LANCE_AIMD_MAX_RATE`           | 5000    |
| Decrease factor    | `lance_aimd_decrease_factor`    | `LANCE_AIMD_DECREASE_FACTOR`    | 0.5     |
| Additive increment | `lance_aimd_additive_increment` | `LANCE_AIMD_ADDITIVE_INCREMENT` | 300     |
| Burst capacity     | `lance_aimd_burst_capacity`     | `LANCE_AIMD_BURST_CAPACITY`     | 100     |

These initial settings are balanced and should work for most
use cases. For example, S3 can typically get up to 5000
req/s and with these settings we should get there in about
10 seconds.

## Fragment Sizing

A Lance table is a collection of fragments tracked by a manifest. How you size those fragments
trades off two classes of work:

- **Manifest-level operations** scale with the *number* of fragments. Every dataset mutation
  (appends, metadata updates, schema changes, compactions, etc.) rewrites the manifest, so a
  larger fragment list makes every write slower. Reads pay a similar cost up front: opening a
  dataset, listing fragments, planning a scan, and resolving transaction conflicts at the
  dataset level all walk the manifest.
- **Fragment-level operations** scale with the *size* of a fragment. These include scans
  against a matching fragment, compaction, updates, deletes, and `merge_insert`. Conflict
  detection for these operations is also done at the fragment level.

Fewer, larger fragments make manifest-level operations cheap but make each fragment-level
operation heavier and increase the chance of conflicts when many writers target the same
fragment. More, smaller fragments do the reverse.

Practical guidance:

- The default of 1M rows per fragment works well up to ~1B rows. Past that, bumping toward
  ~100M rows per fragment is reasonable, though fragment-count limits are rarely the bottleneck
  in practice.
- Tens of thousands of fragments per table is generally fine.
- Keep individual fragments well under object-store object-size limits (S3 caps at 5 TB, and
  stores tend to misbehave well before that). 10 GB–100 GB per fragment is a reasonable upper
  range; 1 TB is a hard ceiling.
- If you run many concurrent updates, deletes, or `merge_insert` operations, err toward more
  fragments — conflict detection is per-fragment, so too few fragments leads to excess
  retries.

## Conflict Handling

Lance supports concurrent operations on the same table using optimistic concurrency control. When two
operations conflict, one of them must be retried. Retries are handled automatically but they repeat
work that has already been done, which can hurt throughput. Understanding and minimizing conflicts is
important for maintaining good performance in write-heavy workloads.

Common sources of conflicts include:

- Concurrent compaction and index building, since both need to modify the same indices
- Update operations that affect the same fragments, since both need to rewrite the same data files

For more details on which operations conflict with each other, see
[conflict resolution](../format/table/transaction.md#conflict-resolution).

### Fragment Reuse Index

Compaction is one of the most expensive write operations because it rewrites data files and, by
default, remaps all indices to reflect the new row addresses. When compaction and index building
run concurrently, they often conflict because both need to modify the same indices. This typically
causes the compaction to fail and retry, and repeated failures can cause table layout to degrade
over time.

The Fragment Reuse Index (FRI) solves this by allowing compaction to skip the index remap step.
Instead of immediately updating indices, compaction records a mapping from old fragment row
addresses to new ones. When indices are loaded into the cache, the FRI is applied to translate
the old row addresses to the current ones. This adds a small cost to index load time but does
not affect query performance once the index is cached.

This decoupling means compaction and index building no longer conflict, which is especially
valuable for tables that are continuously ingesting data while also maintaining indices.

To enable the FRI, set `defer_index_remap=True` when compacting:

```python
dataset.optimize.compact_files(defer_index_remap=True)
```

For details on the index format and usage patterns, see the
[Fragment Reuse Index specification](../format/index/system/frag_reuse.md).

## Indexes

Training and searching indexes can have unique requirements for compute and memory. This section provides some
guidance on what can be expected for different index types.

### BTree Index

The BTree index is a two-level structure that provides efficient range queries and sorted access.
It strikes a balance between an expensive memory structure containing all values and an expensive disk
structure that can't be efficiently searched.

Training a BTree index is done by sorting the column. This is done using an [external sort](https://en.wikipedia.org/wiki/External_sorting) to constrain the total memory usage to a reasonable amount. Updating a BTree index does not
require re-sorting the entire column. The new values are sorted and the existing values are merged into the new sorted
values in linear time.

#### Storage Requirements

The BTree index is essentially a sorted copy of a column. The storage requirements are therefore the same as the column
but an additional 4 bytes per value is required to store the row ID and there is a small lookup structure which
should be roughly 0.001% of the size of the column.

#### Memory Requirements

Training a BTree index requires some RAM but the current implementation spills to disk rather aggressively and so the
total memory usage is fairly low.

When searching a BTree index, the index is loaded into the index cache in pages. Each page contains 4096 values.

#### Performance

The sort stage is the most expensive step in training a BTree index. The time complexity is O(n log n) where n is the number of rows in the column. At very large scales this can be a bottleneck and a distributed sort may be necessary. Lance currently does
not have anything builtin for this but work is underway to add this functionality. Training an index in parts as the data grows
may be slightly more efficient than training the entire index at once if you have the flexibility to do so.

When the BTree index is fully loaded into the index cache, the search time scales linearly with the number of rows that match the
query. When the BTree index is not fully loaded into the index cache, the search time will be controlled by the number of pages
that need to be loaded from disk and the speed of storage. The parts_loaded metric in the execution metrics can tell you how many
pages were loaded from disk to satisfy a query.

### Bitmap Index

The Bitmap index is an inverted lookup table that stores a bitmap for each possible value in the column. These bitmaps are compressed and serialized as a [Roaring Bitmap](https://roaringbitmap.org/).

A bitmap index is currently trained by accumulating the column into a hash map from value to a vector of row ids. Each value
is then serialized into a bitmap and stored in a file.

### Storage Requirements

The size of a bitmap index is difficult to calculate precisely but will generally scale with the number of unique values in the
column since a unique bitmap is required for each value and a single bitmap with all rows will compress more efficiently than
many bitmaps with a small number of rows.

#### Memory Requirements

Since training a bitmap index requires collecting the values into a hash map you will need at least 8 bytes of memory per row.
In addition, if you have many unique values, then you will need additional memory for the keys of the hash map. Training large
bitmaps with many unique values at scale can be memory intensive.

When a bitmap index is searched, bitmaps are loaded into the session cache individually. The size of the bitmap will depend on
the number of rows that match the token.

### Performance

When the bitmap index is fully loaded into the index cache, the search time scales linearly with the number of values that the
query requires. This makes the bitmap very fast for equality queries or very small ranges. Queries against large ranges are
currently extremely slow and the btree index is much faster for large range queries.

When a bitmap index is not fully loaded into the index cache, the search time will be controlled by the number of bitmaps that
need to be loaded from disk and the speed of storage. The parts_loaded metric in the execution metrics can tell you how many
bitmaps were loaded from disk to satisfy a query.

### Vector Index

Vector indexes (IVF_PQ, IVF_HNSW_SQ, etc.) are built in multiple phases, each with different memory requirements.

#### IVF Training

The IVF (Inverted File) phase clusters vectors into partitions using KMeans. To train the KMeans model, a sample of the
dataset is loaded into memory. The size of this sample is determined by:

```
training_data = num_partitions * sample_rate * dimension * sizeof(data_type)
```

The default `sample_rate` is 256. For example, with 1024 partitions, 768-dimensional float32 vectors, and the default
sample rate:

```
1024 * 256 * 768 * 4 bytes = 768 MiB
```

In addition to the training data, each KMeans iteration allocates membership and distance vectors proportional to the
number of training vectors (8 bytes per vector). The centroids themselves require `num_partitions * dimension *
sizeof(data_type)` bytes. In practice, the training data dominates and these additional allocations are small in
comparison.

If the dataset has fewer rows than `num_partitions * sample_rate`, the entire dataset is used for training instead.

#### Quantizer Training

After IVF training, a quantizer (e.g. PQ, SQ) is trained to compress vectors. This phase may sample some of the
dataset, but the sample size is tied to properties of the quantizer and the vector dimension rather than the size of the
dataset. As a result, quantizer training typically requires very little RAM compared to the IVF phase.

#### Shuffling

The final phase scans the entire vector column, transforms each vector (assigning it to an IVF partition and quantizing
it), and writes the results into per-partition files on disk. This is a streaming operation — data is not accumulated in
memory.

The input scan uses a 2 GiB I/O readahead buffer by default (configurable via `LANCE_DEFAULT_IO_BUFFER_SIZE`) and reads
batches of 8,192 rows. Incoming batches are transformed in parallel, with `num_cpus - 2` batches in flight at a time
(configurable via `LANCE_CPU_THREADS`). Each batch is sorted by partition ID and the slices are written directly to the
corresponding partition file. The in-flight memory during this phase is roughly:

```
io_readahead_buffer + num_cpu_threads * batch_size * (raw_vector_size + transformed_vector_size)
```

Each partition has an open file writer with roughly 8 MiB of accumulation buffer. In practice there shouldn't be that
much data accumulated in a single partition anyways. Instead, the max accumulation will be roughly the final size of
the partitions which comes out to `num_rows * (num_sub_vectors + 8) bytes`. For example, 100M rows with a 1536-dimensional
vector will have 96 sub-vectors and so the max accumulation will be ~10GB. The additional 8 bytes per row is for the row ID.

#### Storage Requirements

The on-disk size of a vector index consists of the IVF centroids and the quantized vectors.

The centroids require:

```
num_partitions * dimension * sizeof(data_type)
```

This is typically small. For example, 10K partitions with 768-dimensional float32 vectors is only 30 MiB.

The quantized vectors make up the bulk of the index. Each row stores a quantized code plus an 8-byte row ID. The
exact size depends on the quantizer:

**PQ (Product Quantization):** Each sub-vector is quantized to a single byte, so each row requires
`num_sub_vectors + 8` bytes. For example, 100M rows with 96 sub-vectors:

```
100M * (96 + 8) = ~9.7 GiB
```

**SQ (Scalar Quantization):** Each dimension is independently quantized to a single byte, so each row requires
`dimension + 8` bytes. SQ preserves more information than PQ but requires more storage. For example, 100M rows with
768-dimensional vectors:

```
100M * (768 + 8) = ~72.3 GiB
```

**RQ (RaBitQ):** New indexes default to 5 bits per dimension. Every bit width stores a 1-bit sign
code plus three 4-byte correction factors. Multi-bit indexes also store the remaining bits in
64-dimension-padded blocks and two additional 4-byte correction factors. Including the 8-byte row
ID, the approximate size per row is:

- 1-bit: `dimension / 8 + 20` bytes
- Multi-bit: `dimension / 8 + round_up(dimension, 64) * (num_bits - 1) / 8 + 28` bytes

For example, the default 5-bit index for 100M rows with 768 dimensions requires:

```
100M * (768 / 8 + 768 * 4 / 8 + 28) = ~47.3 GiB
```

The 5-bit default retains more information for the higher-fidelity distance estimates used by
`Normal` and `Accurate` search modes, at the cost of more quantization work and index I/O during
the build and a larger index. `Fast` search mode uses only the 1-bit sign code even when the index
stores additional bits. Set `num_bits=1` explicitly to minimize index size and build I/O; the same
100M-row example uses about 10.8 GiB, but searches cannot use the multi-bit distance estimate and
may have lower recall.

#### AMX Acceleration

On Linux x86_64 with an AMX-FP16 CPU (Intel Granite Rapids / Xeon 6 and newer), a `float16`
vector column indexed with `dot` distance uses the AMX tile instructions, provided the build
machine had clang >= 16 or gcc >= 13 to compile the kernel. There is nothing to enable —
Lance checks the CPU at run time and falls back to the previous implementation everywhere else.

The accelerated paths are also shape-gated, because below these sizes a tile pass costs more
than it saves and the kernel declines the work:

| Condition | Why |
|---|---|
| `float16` vectors, `dot` distance | The kernel is fp16-specific; other types and metrics keep their existing paths |
| `dimension >= 32` | One tile pass covers 32 dimensions; a shorter vector would be all scalar cleanup |
| `num_centroids >= 32` | The GEMM steps its centroid loop by 32 and has no partial-tile path |

Anything outside them behaves exactly as it does today, so a small dataset or a low-dimensional
column simply keeps the previous implementation rather than changing behaviour.

Index build also changes algorithm where all of the above hold: comparing every vector against
every centroid becomes affordable, so partition assignment is exact instead of approximated with
a graph search over the centroids. Recall improves, and partition assignments differ from what an
older build produced.

Set `LANCE_DISABLE_AMX=1` to take the AMX paths out of service without rebuilding — for
A/B measurement, or to get the previous behaviour back. Because it also moves partition
assignment back to the approximate path, an index built with it set is not equivalent to one
built without it; compare recall, not just build time.

#### Covering Columns

A vector index can carry the values of extra columns beside its vectors, so a query whose
projection they satisfy is answered from the index without a take against the base table.
That trade only pays off where the search settles into a single global top-k heap, because
the covering read is then bounded by the query's survivors. Every HNSW index, and any query
with `query_parallelism` above one, emits results per partition instead, so covering reads
scale with the number of partitions probed rather than with `k`, and are issued serially.

On those shapes covering is *slower* than the base-table take it exists to avoid — roughly
2.8x a plain index's latency warm at nprobe 32, and 1.18x cold. Prefer IVF_PQ or IVF_FLAT
when declaring covering columns; index creation logs a warning when covering is combined
with HNSW.
