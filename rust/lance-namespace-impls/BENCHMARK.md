# `__manifest` catalog benchmark

Measures manifest-only directory catalog startup, reads, and copy-on-write mutations as
the manifest scales. The current catalog loads the complete `__manifest` into an in-memory
snapshot, queries that snapshot, and reloads it after a successful update.

The catalog commits every mutation by rewriting the whole `__manifest` (copy-on-write)
and atomically writing a new manifest version. This benchmark characterizes:

- **Startup** — open the namespace and load the entire manifest snapshot.
- **Read** — list namespaces, list tables, or describe a table from an already loaded
  snapshot.
- **Continuous commit** — a single process commits `N` times into a manifest already
  holding `rows` entries (per-commit latency + throughput).
- **Concurrent commit** — `C` processes commit continuously for a fixed duration against
  a manifest of `rows` entries (steady, contended TPS).

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

Record the instance type, region, catalog size, startup latency and peak RSS, warm read
latency, serial write latency, concurrent throughput, and error count for each compared
implementation. Results should distinguish the legacy indexed and no-index builds from
the in-memory build.
