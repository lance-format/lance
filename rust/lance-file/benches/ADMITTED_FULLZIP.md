# Admitted FullZip file comparison

This harness qualifies the opt-in file reader before dataset/server adoption.
It does not activate it in ordinary scanners or Sophon. It is not the T-210
PE/QN/HTTP/Flight acceptance harness, and its results cannot replace that matrix.

Correctness and test builds run in exact-head CI. Optimized builds and all timed
runs belong on a managed VM. Do not run this example as a local performance probe.

## Controls

Use `ordinary` and `admitted` on the **same binary, immutable file and source head**.
This isolates the read strategy; it is not a before/after PR speedup measurement.
Record the upstream base and direct parent as well. Any later PR-wide performance
claim also requires a compatible ordinary-reader baseline build from that parent.

Keep CPU count, `LANCE_CPU_THREADS`, queue, IO budget (64 MiB), maximum batch rows,
file layout, source region and process memory limit identical. Ordinary reads use
batch readahead 1; admitted reads have no speculative read-ahead. Record this
behavioral difference. Use a single explicit shared payload budget for admission;
it does not limit metadata, transport buffering or whole-process RSS.

Build once with the pinned toolchain and configured mbx shim:

```sh
cargo build --profile release-with-debug --locked -p lance-file --example admitted_fullzip
```

Copy the binary out of the managed target, record its SHA256 and exact source/
lockfile/toolchain/recipe hashes, then stop builds before measuring. Record actual
VM instance, region, architecture, CPU/RAM, disk and kernel. Preserve negative
results. Never divide by an OOM or incomplete run to claim a speedup.

## Fixture and sink

The v2.2 writer emits UTF8 row IDs and inline `List<LargeBinary>` payloads using
the FullZip encoding hint. IDs and frame bytes depend on the absolute row index.
Null/empty lists and null/empty children occur at different intervals. Payload
words are deterministic mixed integers rather than one repeated compressible
value. Each writer batch flushes independently; the writer never retains the
whole fixture. The fixture's 64 MiB generation-batch guard is not a reader or
product result limit.

Use a new task-owned local path or S3 prefix. Do not rewrite retained T-210 server
fixtures. Example CLI (replace `BIN` and `URI` with recorded VM paths):

```text
BIN write URI ROWS VALUE_BYTES PAGE_ROWS
BIN read URI ordinary|admitted EXPECTED_ROWS MAX_ROWS WORKSPACE_BYTES VALUE_BYTES rows|full DELAY_MS standard|lite
```

The `rows` sink checks every row ID, list/child validity, lengths and final row
count without scanning/generating the full reference payload. Use it for primary
full-drain timings. `full` additionally verifies every payload word and emits an
order-dependent fingerprint; use a separate validation run for each variant and
fixture. A fingerprint is not a cryptographic hash; the acceptance here is the
independent per-word comparison. Both sinks discard each batch before requesting
more. Neither collects the output. Refusal is a failed/incomplete run, never EOF.

The output reports complete drain wall time, first batch, sink time, rows,
logical payload bytes, batch count and scheduler payload IOPS/bytes. Open/metadata
latency is included in wall/first-batch time, but the per-reader I/O counters are
attached after open. First batch is not first HTTP byte. Do not subtract sink time
and call the remainder server latency. Store exact CLI/options with every result.
Measure process maximum RSS/CPU externally (`/usr/bin/time -v`), and save cgroup
peak/current anonymous/file memory and `memory.events` under a finite limit with
swap disabled. No sampled `budget.reserved()` value is a process peak estimate.

## Smallest useful matrix

Begin with a small diagnostic fixture. Confirm file layout, complete counts,
full payload validation, raw metrics and actual cgroup enforcement before scaling.
Then use two cumulative row counts with the same row width, page size and fixed
memory budget. For example, 40960/81920 rows, 131072 bytes per frame and 32 rows
per writer batch produce roughly 8/16 GiB after null/empty exclusions. Record
actual payload bytes; do not label these exact 8/16 GiB fixtures. Each value fits
comfortably within the workspace. Start with a 64 MiB payload workspace and a
4 GiB process memory limit, not a larger limit to hide accumulation.

Add a small-value control (64 bytes per frame with more rows per page) to expose
per-batch/per-column dispatch and request overhead. Alternate ordinary/admitted
order across three repetitions. Separate local filesystem from same-region S3;
never combine their ratios. Describe fresh processes and uncontrolled OS/S3
cache state honestly; do not claim cold S3. Add a bounded slow consumer using
`DELAY_MS`, with the delay excluded from throughput claims. Existing CI covers
finite peers/cancellation; this single-process CLI does not prove shared-budget
multi-process or PE/QN concurrency. Those remain in the original server matrix.

Stop and diagnose a candidate error or material regression before expanding the
matrix. Archive raw samples, exit codes, binary/source identity and results before
VM cleanup. Do not retain a task-only VM without a concrete near-term next use.
