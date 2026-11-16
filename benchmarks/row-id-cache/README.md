# Row-ID cache comparison

This is the measurement portion of the helper used for the deletion-cache
experiments in #10800. It compares the actual resolver and public `take_rows`
path, with independent expected physical addresses and payloads. Unrelated
BTree experiments and fixture-writing modes have been removed. The measured
operations, timer boundaries, memory checks and snapshot schedule are retained.
This extracted helper has not yet been built or run against the latest head.

The historical measurements belong to commit `266355944c437cad88c4a2bbedc0ab48f6057908`,
compared with `b5ee153064583d39019ca15939d4de9dfeeef815`. They are **not** current
head-vs-main results. Current acceptance is still pending. Fixture generation,
cache admission and concurrency must be recorded separately from query timing.

## Build two frozen revisions

Run benchmark builds and measurements on a managed Linux VM with cgroup v2,
the repository's pinned Rust toolchain and configured mbx Cargo integration.
Keep the same CPU architecture, compiler, features and `release-with-debug`
profile for both arms. These scripts do not provision cloud resources.

```bash
# From this checkout, on the VM. Pass full 40-character commits.
bash benchmarks/row-id-cache/build.sh "$MAIN_SHA" /absolute/output/main
bash benchmarks/row-id-cache/build.sh "$HEAD_SHA" /absolute/output/head
```

The script exports the commit without modifying a checkout or branch, records
the toolchain and executable hashes, and adds the same helper to both exports.
In the PR export it only makes the real resolver visible to the helper. In the
main export it adds a wrapper around `get_row_id_index_with` + `get_many`.
Neither adapter changes the public take path or lookup policy. The exported
source therefore includes a documented benchmark-only visibility adapter; it
is not byte-identical to the product tree.

## Inputs and scenarios

The example configs describe the historical workload settings. Replace
`query_file` with an absolute path to a saved oracle. Run both arms against the
**same immutable dataset versions and same oracle file**, and record its hash.
The historical temporary mutation datasets have been cleaned up; these configs
are not bundled datasets or claims that the old S3 objects still exist.

| Config | Dataset | Query path | Index cache | Metadata cache | Total memory |
| --- | --- | --- | --- | --- | --- |
| `no-deletes.json` | 100M shuffled IDs, 1000 fragments | Public take, 1000 IDs | 4 GiB | 1 GiB | 4 GiB |
| `small-updates.json` | Seven snapshots, 32 updates and one deletion per round | Public take | 4 GiB | 1 GiB | 8 GiB |
| `heavy-deletes.json` | Same snapshot schedule after 90% initial deletions | Public take | 4 GiB | 1 GiB | 8 GiB |
| `small-cache.json` | Heavy-deletion snapshot | Public take, 64 IDs | 128 MiB | 1 GiB | 4 GiB |

For isolated lookup, change only `path` to `resolver`. The historical churn
resolver budget was 4 GiB, versus 8 GiB for full takes. Preserve an OOM as a
failure outcome instead of increasing only one arm's budget. For the longer
warm-take follow-ups, set `warm_calls` to 32 in both static configs. Do not pool
those samples with the original eight-call run.

The profile oracle is JSON with `uri`, `version`, `rows`, `random` and `local`.
Each query pool contains parallel `ids` and `addrs` arrays. `addrs` contains
physical row addresses from a direct scan, not the resolver being benchmarked.
The table has five UInt64 columns: `key`, `p0`, `p1`, `p2`, `p3`. Initially,
`key` equals the stable row ID and payload lane `i` is:

```text
rotate_left(key * 0x9e3779b97f4a7c15 modulo 2^64, i * 11) XOR i
```

The churn oracle contains `uri` and a `snapshots` array. Each snapshot has
`version`, `fragments`, `ids`, `addresses` and `values`; the last two are arrays
of optional addresses and optional five-element payloads. A missing/deleted ID
has `null` in both. Include a duplicate request and an unknown ID. Generate
addresses with an unindexed scan using `_rowaddr`, and check payloads against
the known write/update history before saving the oracle. Save all seven
versions before timing either arm; fixture construction is not query latency.

The churn schedule measures each version's first query and four warm queries,
then visits all seven held snapshots three times: 56 calls total. It holds
Dataset snapshots sharing one bounded session, not explicit index handles.

## Run and interpret

Run serially in fresh processes. Reverse the arm order in a second round.
Use the VM's existing authorized S3 role; do not put credentials in configs.
For example, with an 8 GiB snapshot-take config:

```bash
sudo systemd-run --wait --pipe --unit=row-id-head-round0 \
  --uid="$(id -un)" --working-directory="$PWD" \
  --property=MemoryMax=8589934592 --property=MemorySwapMax=0 \
  --setenv=AWS_REGION="$AWS_REGION" \
  /absolute/output/head/row_id_cache_bench /absolute/small-updates.json \
  2>&1 | tee head-round0.log
```

Use shell pipeline failure propagation, preserve the actual process exit code
and `systemctl show` result, and retain failed/OOM runs. The helper checks its
actual cgroup memory and swap caps before querying. It reports per-call latency,
all referenced stores' reads/bytes (deduplicating shared tracked stores), cache
charge, RSS, peak RSS and output validity as JSON lines. Cold caches mean a fresh
Lance session, not a claim that S3 or every OS cache is cold.

Report p50/p95, first-call and cumulative snapshot costs, throughput, charged
cache bytes and actual peak memory separately. Compare full takes with full
takes and resolvers with resolvers; never substitute the isolated lookup gain
for end-to-end performance. The current review additionally requires a realistic
default-compaction workload to justify pruning, and a matched current-head/main
run without any requested scenario regressing beyond about 20%, including the
reported warm resolver cases. Neither condition is
established by the historical measurements or by checking in this helper.
