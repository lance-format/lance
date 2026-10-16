# Update Join Strategies

Fragment updates join existing rows with an input containing replacement values. For a small
update input, an in-memory hash join is usually fastest. For a large or wide input, retaining the
entire input and its hash table can require too much memory. Lance therefore provides configurable
update join strategies, including a spillable sort-merge join whose DataFusion memory use is
bounded by a configurable pool.

This is a low-level, fragment-oriented API. An update returns new fragment metadata and the IDs of
the modified fields; the caller must commit those results to the dataset. See
[Update Columns](distributed_write.md#update-columns) for the complete distributed update and
commit workflow.

## Choosing a strategy

The `strategy` option accepts the following values:

| Python value | Rust value | Behavior |
| --- | --- | --- |
| `"auto"` | `UpdateJoinStrategy::Auto` | Uses the hash join only while the right-hand side (RHS) is within both configured hash thresholds. Otherwise, uses the sort-merge join. This is the default. |
| `"hash"` | `UpdateJoinStrategy::Hash` | Always builds the in-memory hash join. Hash thresholds, the external memory pool, and the temporary-disk limit do not constrain this strategy. |
| `"sort_merge"` | `UpdateJoinStrategy::SortMerge` | Uses the spillable external plan for every non-empty update input. |

Use `auto` for most workloads. It retains the lower overhead of the hash join for small inputs and
switches to the memory-bounded path for larger inputs.

Use `hash` only when the RHS is known to fit comfortably in memory. This strategy reads all RHS
batches into memory and builds a hash index; forcing it for a large input can exhaust process
memory.

Use `sort_merge` when predictable memory use is more important than avoiding temporary I/O. The
external plan sorts the join keys, performs a sort-merge join, and spills intermediate data when
necessary. It late-materializes update payload columns so wide values do not pass through every key
sort and join stage.

## Automatic selection

In `auto` mode, Lance reads the RHS incrementally and keeps the hash strategy eligible only while
both of these conditions are true:

- the RHS contains at most `max_hash_rows` rows; and
- the estimated RHS allocation is at most `max_hash_bytes` bytes.

The defaults are 250,000 rows and 1 GiB. Exceeding either threshold selects sort-merge. The byte
estimate includes Arrow buffers, join keys, and estimated per-row hash-table overhead, so it is not
the serialized or on-disk size of the input. It is a selection estimate, not a hard memory limit.

The external memory pool is not part of automatic strategy selection. Changing
`external_memory_pool_bytes` controls the chosen sort-merge execution but does not change which
strategy `auto` selects.

## Python usage

The options are keyword arguments on `LanceFragment.update_columns`:

```python
MIB = 1024 * 1024
GIB = 1024 * MIB

updated_fragment, fields_modified = fragment.update_columns(
    updates,
    left_on="id",
    right_on="id",
    strategy="auto",
    max_hash_rows=250_000,
    max_hash_bytes=GIB,
    external_memory_pool_bytes=256 * MIB,
    max_temp_directory_bytes=20 * GIB,
)
```

To require bounded external execution for every non-empty input:

```python
updated_fragment, fields_modified = fragment.update_columns(
    updates,
    left_on="id",
    right_on="id",
    strategy="sort_merge",
    external_memory_pool_bytes=256 * 1024 * 1024,
    max_temp_directory_bytes=20 * 1024 * 1024 * 1024,
)
```

`max_hash_rows` and `max_hash_bytes` must either both be omitted or both be specified in Python.

## Rust usage

Use `FileFragment::update_columns_with_options` to configure an individual update:

```rust
# use arrow_array::RecordBatchReader;
# use lance::Result;
use lance::dataset::{UpdateJoinOptions, UpdateJoinStrategy};
use lance::dataset::fragment::FileFragment;

# async fn update(
#     fragment: &mut FileFragment,
#     updates: impl RecordBatchReader + Send + 'static,
# ) -> Result<()> {
let options = UpdateJoinOptions::default()
    .with_strategy(UpdateJoinStrategy::Auto)
    .with_hash_thresholds(250_000, 1024 * 1024 * 1024)
    .with_external_memory_pool_bytes(256 * 1024 * 1024)
    .with_max_temp_directory_bytes(20 * 1024 * 1024 * 1024);

let result = fragment
    .update_columns_with_options(updates, "id", "id", options)
    .await?;

// Commit result.fragment and result.fields_modified to the dataset.
# Ok(())
# }
```

`FileFragment::update_columns` and `FileFragment::update_columns_with_offsets` use
`UpdateJoinOptions::default()`.

## Configuration reference

All sizes are integer byte counts.

| Option | Default | Applies to | Description |
| --- | ---: | --- | --- |
| `strategy` | `auto` | All inputs | Selects automatic, hash, or sort-merge execution. |
| `max_hash_rows` | 250,000 | `auto` | Largest RHS row count still eligible for the hash join. |
| `max_hash_bytes` | 1 GiB | `auto` | Largest estimated RHS allocation still eligible for the hash join. |
| `external_memory_pool_bytes` | See below | Sort-merge | Size of the DataFusion memory pool used by the external plan. It must be greater than zero. |
| `max_temp_directory_bytes` | See below | Sort-merge | Maximum temporary spill-directory usage. It must be greater than zero. The update fails with a resource-exhaustion error if the limit is insufficient. |

Per-operation settings take precedence over environment variables. If an external memory pool is
not set on the operation, Lance uses `LANCE_MEM_POOL_SIZE`; if that is also unset, update joins use
256 MiB per execution partition. The current update plan uses one execution partition.

If a temporary-disk limit is not set on the operation, Lance uses
`LANCE_MAX_TEMP_DIRECTORY_SIZE`; if that is also unset, the default is 100 GiB. Both environment
variables contain plain integer byte counts, for example:

```bash
export LANCE_MEM_POOL_SIZE=268435456
export LANCE_MAX_TEMP_DIRECTORY_SIZE=21474836480
```

Spill files use the process's temporary filesystem and are removed after the execution releases
them. Make sure that filesystem has enough free space before starting large updates. The temporary
limit is an upper bound, not a reservation, and the external plan can write several intermediate
representations of the input. Required spill capacity can therefore be larger than the RHS itself.

!!! warning

    Setting `LANCE_BYPASS_SPILLING` disables DataFusion spilling. If `auto` selects sort-merge, or
    `sort_merge` is explicitly selected, the update fails instead of falling back to an unbounded
    in-memory execution.

## Memory behavior

`external_memory_pool_bytes` bounds memory registered by DataFusion operators. It does not place a
hard bound on total resident set size (RSS). The process also contains input and output Arrow
buffers, fragment readers and writers, runtime state, and memory retained by the system allocator.

On Linux systems using glibc, applications that need lower RSS can limit the number of allocator
arenas before starting the process:

```bash
MALLOC_ARENA_MAX=4 python update_fragments.py
```

This is a process-wide allocator setting, not a Lance setting. A smaller value can reduce retained
heap memory during spill-heavy updates, but it can also affect unrelated concurrent allocation.
Benchmark the complete application before adopting it. `MALLOC_ARENA_MAX=1` is a more aggressive
choice when minimum RSS is the priority and allocation concurrency is low.

When several fragment updates run concurrently, each operation has its own external pool and spill
work. Plan memory and temporary-disk capacity for their combined usage.

## Tuning guidance

- Start with `auto` and the default thresholds.
- Lower the hash thresholds when input rows are unusually wide or the application has a tight
  process-memory budget.
- Raise the hash thresholds only after measuring peak RSS with representative data.
- Keep the 256 MiB external pool unless benchmarks show that a different value is better for the
  workload. A smaller pool can increase spill volume and merge passes; a larger pool can improve
  throughput at the cost of memory.
- Set `max_temp_directory_bytes` from measured peak spill usage with headroom. Do not assume it can
  be derived directly from the RHS size.
- Avoid duplicate RHS join keys. If duplicates are present, Lance selects one matching RHS row, but
  which row is chosen is not defined.

## Update semantics and constraints

The strategy changes execution characteristics, not the logical update operation:

- Rows with a matching RHS key receive the supplied replacement column values.
- Rows without a match retain their existing values.
- The RHS must contain the join column and at least one existing fragment column to update.
- Left and right join-key types must match.
- The join column itself is not updated.
- Reserved metadata columns such as `_rowid` and `_rowaddr` cannot be replacement columns.
- An empty RHS is a no-op and does not run either join algorithm.

