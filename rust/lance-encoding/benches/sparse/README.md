# Sparse structural read-path benchmarks

Measures the cost of reading pages written with `Layout::SparseLayout`, split into the two
things a change to the sparse chunk index can move independently:

| target                    | harness    | measures                                                     |
| ------------------------- | ---------- | ------------------------------------------------------------ |
| `bench sparse_footprint`  | plain main | exact resident bytes and allocation volume, no statistics    |
| `bench sparse_decode`     | Criterion  | initialize, full scan and scattered take latency             |

Memory is deliberately not a Criterion benchmark. The quantities are exact byte counts, so
reporting them as a distribution with confidence intervals would add noise and hide the
signal.

## Running

```bash
# exact memory report
cargo bench -p lance-encoding --bench sparse_footprint

# timings; use -- <filter> to narrow, e.g. -- sparse_take
cargo bench -p lance-encoding --bench sparse_decode

# check the inputs still land on the sparse layout (also runs in CI)
cargo test -p lance-encoding --test sparse_bench_layouts
```

## Comparing two revisions

`run_ab.sh` is the reproducible path. It builds two refs, runs identical benchmark code
against both, and reports a paired comparison.

```bash
CPUS=0,2,4,6,8,10,12,14 ./run_ab.sh <before-ref> <after-ref> 8
```

It handles the three things that most often make a hand-rolled A/B wrong:

1. **Identical benchmark code on both arms.** The bench sources from the invoking tree are
   copied into both worktrees, so the comparison cannot silently become "old bench versus
   new bench".
2. **Separate target directories per arm.** Cargo's unit hash covers package name, version,
   profile and features but not the source directory. Two worktrees at the same crate
   version share a hash, so with a shared `CARGO_TARGET_DIR` cargo reports the first arm's
   stale artifact as fresh and both arms measure the same binary. Verify with `md5sum` on
   the two bench executables if you are ever unsure.
3. **Interleaved, rotated arms.** Arm order alternates each round so thermal drift and
   background load bias both arms equally rather than whichever arm runs first.

Timing comparisons need at least 6 rounds. `compare.py` uses an exact two-sided sign-flip
permutation test over per-round paired deltas, whose smallest attainable p-value is
`2/2**rounds`; at 5 rounds that floor is 0.0625, so no effect of any size could clear
p<0.05.

Pin to physical cores for timing. Find them with `lscpu -p=CPU,CORE` and take one CPU per
`CORE` value; sibling hyperthreads share execution resources and inflate variance.

## Reading the output

`sparse_footprint` reports three numbers per case:

- **`cache_bytes`** — what `LanceCache::size_bytes()` holds after `DecodeBatchScheduler::try_new`
  has initialized every field scheduler. This is the headline: the state is retained for as
  long as the dataset is open, once per page per column, so it sets the cost of holding a
  wide 2.3 table open.
- **`init_bytes`** — bytes passed to the global allocator during initialize. Catches
  transient allocations that never reach the cache, which `cache_bytes` cannot see.
- **`init_allocs`** — allocation count over the same region. Separates "allocates less" from
  "allocates fewer times"; a change can do either without the other.

`cache_bytes` is derived from `DeepSizeOf`. When comparing two revisions, check that neither
revision changed how the cached types report their own size, or the comparison measures
accounting rather than memory.

## Input matrix

`cases.rs` holds the matrix. Every case declares the layout it expects and
`sparse_bench_layouts.rs` asserts it, because layout selection is a **writer-side**
heuristic: a page is read by the sparse scheduler only because it was written with
`SparseLayout`. Without that assertion a change to the rep/def budget or to automatic sparse
selection would quietly move cases onto the dense mini-block path, leaving the benches green
while measuring nothing.

Note that an all-null page resolves to `ConstantLayout` rather than `SparseLayout`, as do
zero-row and all-empty-list pages: the writer recognises that the page holds no distinct leaf
values and emits a constant page, which stores no chunk index at all. `all_null_lists` is
kept in the matrix to pin that behaviour down, so a future writer change that starts routing
it through the sparse layout surfaces as an assertion failure rather than as a silent
regression.

`Case::columns` rather than page count is the knob for multiplying resident state.
`encode_batch` emits one page per column here regardless of the page budget, because
`AccumulationQueue` flushes whole arrays and never splits one.

## What these benches do not cover

- **The encode path.** Nothing here measures writing. That is deliberate for changes scoped
  to the reader, where encode timings serve only as a neutrality check.
- **Real IO.** Pages are served from memory via `BufferScheduler`, so results isolate CPU and
  allocation cost and exclude storage latency.
- **Whole-dataset reads.** These benches live in `lance-encoding`, which cannot depend on
  `lance-file` or `lance`. Multi-fragment and full-table scan behaviour belongs in the
  benches in `rust/lance/benches/`.
