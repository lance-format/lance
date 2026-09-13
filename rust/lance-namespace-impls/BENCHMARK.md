# `__manifest` catalog benchmark

Measures manifest-only directory catalog startup, reads, and copy-on-write mutations as
the manifest scales. The current catalog loads the complete `__manifest` into an in-memory
snapshot, queries that snapshot, and refreshes it asynchronously after a successful update.

The catalog commits every mutation by rewriting the whole `__manifest` (copy-on-write)
and atomically writing a new manifest version. This benchmark characterizes:

- **Startup** — open the namespace and load the entire manifest snapshot.
- **Read** — list namespaces, list tables, or describe a table from an already loaded
  snapshot.
- **Continuous commit** — a single process commits `N` times into a manifest already
  holding `rows` entries (per-commit latency + throughput).
- **Concurrent commit** — `C` processes commit continuously for a fixed duration against
  a manifest of `rows` entries (steady, contended TPS).
- **Mixed read/write** — concurrent clients share one catalog instance and each issue an
  exact number of reads for every write, with their write positions staggered so reads
  and writes overlap.

## Binary: `examples/manifest_bench.rs`

```text
manifest_bench seed-large --root <uri> --count <rows> \
    [--storage-option aws_region=us-east-1]
manifest_bench run --root <uri> --operation startup \
    --concurrency 1 --operations 1 --initial-entries <rows>
manifest_bench run --root <uri> --operation warm-read-describe-table \
    --concurrency 1 --operations 1000 --warmup 10 --initial-entries <rows>
manifest_bench run --root <uri> --operation write-create-namespace \
    --concurrency 1 --operations 100 --initial-entries <rows>
manifest_bench run --root <uri> --operation write-create-namespace \
    --concurrency 50 --duration-secs 30 --initial-entries <rows>
manifest_bench mixed --root <uri> --concurrency 4 --writes-per-worker 5 \
    --reads-per-write 100 --warmup 10 --initial-entries <rows>
```

- `seed-large` bootstraps all `count` rows in one direct Lance dataset write. It does not
  call catalog create operations per entry and does not perform an index-building rewrite.
- `run` spawns `--concurrency` worker subprocesses. With `--operations` it runs a fixed
  operation budget; with `--duration-secs` each worker commits until the deadline. It
  prints one JSON `BenchResult` per concurrency level with throughput and p50/p90/p99
  latency.
- `startup`, `warm-read-list-namespaces`, `warm-read-list-tables`, and
  `warm-read-describe-table` cover the in-memory load and query paths.
- The committed operation defaults to `write-create-namespace`, the cheapest pure
  `__manifest` mutation. `write-create-table` and `write-declare-table` are also available.
- `mixed` shares one namespace across `--concurrency` asynchronous clients. Each client
  performs `--writes-per-worker` cycles containing exactly `--reads-per-write` table
  descriptions and one namespace creation. Write positions are staggered across clients;
  the JSON result reports read and write latency, throughput, and errors separately.

S3 requires the default `dir-aws` feature (on by default) and AWS credentials in the
environment; pass `--storage-option aws_region=<region>`.

## Legacy sweep panel: `benches/manifest_commit_sweep.sh`

The legacy sweep script runs sizes × {inline index, no index} × {continuous,
concurrent×C}. It remains useful with a build from before the in-memory implementation.
Use isolated S3 prefixes and the same one-shot seed for every variant so each run starts
from the same catalog contents.

```bash
cargo build --release --example manifest_bench -p lance-namespace-impls
S3_BASE=s3://<bucket>/manifest-cow-bench/$(date -u +%Y%m%dT%H%M%SZ) \
  rust/lance-namespace-impls/benches/manifest_commit_sweep.sh
```

The default legacy panel can be overridden with `SIZES`, `CONCURRENCY`,
`INLINE_VARIANTS`, `CONT_OPS`, and `CONC_DURATION_SECS`. Results land in `$OUT_DIR`.

## Representative results

EC2 `c7i.12xlarge`, S3 `us-east-1`, upstream commit `577091e5` versus the in-memory
implementation at `ff72b913`. Startup is the median of five fresh processes; other values
are operation p50. Every catalog was bootstrapped in one direct dataset write before one
preparation mutation established its steady-state representation.

| rows | variant | startup | startup RSS | warm exact read | list tables | serial writes/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1K | indexed | 97.7 ms | 44.5 MiB | 35.0 ms | 61.7 ms | 3.339 |
| 1K | no index | 100.2 ms | 43.1 MiB | 60.9 ms | 58.6 ms | 4.563 |
| 1K | in memory | 136.3 ms | 75.3 MiB | 9.1 ms | 9.4 ms | 5.677 |
| 100K | indexed | 96.4 ms | 44.6 MiB | 39.5 ms | 138.7 ms | 1.731 |
| 100K | no index | 98.9 ms | 43.2 MiB | 65.5 ms | 101.6 ms | 2.167 |
| 100K | in memory | 231.5 ms | 106.0 MiB | 9.5 ms | 19.1 ms | 2.578 |
| 1M | indexed | 104.3 ms | 44.3 MiB | 92.8 ms | 461.0 ms | 0.565 |
| 1M | no index | 98.2 ms | 43.3 MiB | 84.1 ms | 463.8 ms | 0.781 |
| 1M | in memory | 603.3 ms | 386.7 MiB | 11.0 ms | 125.9 ms | 0.684 |

At 1M rows, the in-memory snapshot is 8.4× faster than the indexed implementation for an
exact lookup, 3.7× faster for a full table listing, and 21% faster for serial writes. Its
fresh startup costs about 0.5 seconds and 342 MiB more RSS. The legacy no-index writer is
12% faster for serial writes at 1M, while its read latency remains close to the indexed
implementation. With ten contending writers, in-memory and no-index were effectively tied
(0.731 versus 0.737 ops/s) and both were about 43–45% faster than indexed writes.

## Mixed 100:1 read/write results

Four asynchronous clients shared one catalog instance. Each client ran five cycles with
exactly 100 table descriptions and one namespace creation, with writes staggered across
clients. These results use the blocking post-commit refresh at `9a337af5`. Each value is
the median of three isolated S3-backed runs; all 54,000 reads and 540 writes across the
matrix succeeded.

| rows | variant | read/s | read p50 | read p99 | write/s | write p50 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1K | indexed | 92.46 | 34.64 ms | 193.17 ms | 0.925 | 258.60 ms |
| 1K | no index | 61.59 | 59.26 ms | 136.94 ms | 0.616 | 211.86 ms |
| 1K | in memory | 258.57 | 9.28 ms | 129.47 ms | 2.586 | 196.31 ms |
| 100K | indexed | 74.94 | 37.29 ms | 281.35 ms | 0.749 | 484.13 ms |
| 100K | no index | 52.92 | 65.03 ms | 213.95 ms | 0.529 | 393.33 ms |
| 100K | in memory | 186.83 | 9.37 ms | 220.90 ms | 1.868 | 420.09 ms |
| 1M | indexed | 49.92 | 39.66 ms | 236.14 ms | 0.499 | 1,964.58 ms |
| 1M | no index | 37.04 | 86.59 ms | 261.71 ms | 0.370 | 1,205.96 ms |
| 1M | in memory | 79.91 | 9.17 ms | 113.04 ms | 0.799 | 3,238.71 ms |

At 1M rows, the in-memory design delivers 1.60× the mixed-workload throughput of indexed
and 2.16× that of no-index. Its read p50 is 4.32× faster than indexed and remains close to
the read-only result, although read p99 rises to 113 ms during overlapping writes. The
tradeoff is write latency: in-memory write p50 reaches 3.24 seconds under four-client
contention, versus 1.96 seconds indexed and 1.21 seconds no-index.

### Asynchronous post-commit refresh

A focused 1M-row A/B comparison on the same `c7i.12xlarge` moved the snapshot refresh
off the commit path. With one writer performing one write followed by 100 reads, median
write latency fell from 2,028 ms to 1,384 ms (32%). Total cycle time remained effectively
unchanged (3,118 ms versus 3,092 ms), because the first read waits for the refresh when
the background task has not completed.

Under the four-client 100:1 mix, write p50 fell from 4,052 ms to 3,528 ms (13%), while
throughput remained effectively unchanged at 66.9 versus 66.7 reads/s. Read p50 stayed
near 9.4 ms, but read p99 increased from 120 ms to 677 ms because refresh latency is now
charged to reads that arrive immediately after a commit. All 13,000 reads and 130 writes
across the focused comparison succeeded.
