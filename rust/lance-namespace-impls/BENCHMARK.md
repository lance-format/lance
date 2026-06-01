# Copy-on-Write Directory Manifest Benchmark Results

## Test Environment

- **Instance:** c7i.12xlarge / c7i.48xlarge EC2 (us-east-1)
- **Storage:** S3 Standard, S3 Express One Zone (same AZ)
- **Concurrency:** Multi-process (each worker is a separate OS process)
- **Operations per level:** 100-200

## Changes Benchmarked

1. **v1 → v2:** CommitBuilder replaced with direct `Manifest::new_from_previous` + `write_manifest_file`
2. **v2 → v3:** Removed redundant dataset reload before each write (use cached dataset, detect conflicts at commit time)
3. **Index vs no-index:** With/without BTree + Bitmap + FTS index building during CoW rewrite

---

## Write Operation Comparison (v3, S3 Express, c=1)

| Scale | Index | create-ns | declare-table | create-table |
|---|---|---:|---:|---:|
| 1K | idx | 174ms (5.7/s) | 193ms (5.1/s) | 385ms (2.6/s) |
| 1K | noidx | 146ms (6.8/s) | 159ms (6.2/s) | 352ms (2.8/s) |
| 100K | idx | 257ms (3.9/s) | 291ms (3.4/s) | 512ms (1.9/s) |
| 100K | noidx | 158ms (6.3/s) | 196ms (5.0/s) | 417ms (2.4/s) |

### Latency Breakdown (S3 Express, c=1, no-index, 1K entries)

```
create-namespace: 146ms  (manifest CoW rewrite only)
declare-table:    159ms  (manifest CoW rewrite + .lance-reserved PUT)
create-table:     352ms  (manifest CoW rewrite + .lance-reserved PUT + table data write)

.lance-reserved overhead:  ~13ms
table data write overhead: ~193ms
```

---

## Write Throughput Improvement (v2 → v3, reload fix)

### S3 Express

| Scale | Index | Operation | v2 p50 (tput) | v3 p50 (tput) | p50 delta | tput delta |
|---|---|---|---:|---:|---:|---:|
| 1K | idx | create-ns | 273ms (3.7/s) | 164ms (6.0/s) | **-40%** | **+62%** |
| 1K | idx | create-table | 485ms (2.1/s) | 345ms (2.9/s) | **-29%** | **+38%** |
| 100K | idx | create-ns | 280ms (3.5/s) | 265ms (3.8/s) | -5% | +9% |
| 100K | idx | create-table | 588ms (1.7/s) | 460ms (2.2/s) | **-22%** | **+29%** |

### S3 Standard

| Scale | Index | Operation | v2 p50 (tput) | v3 p50 (tput) | p50 delta | tput delta |
|---|---|---|---:|---:|---:|---:|
| 1K | idx | create-ns | 410ms (2.4/s) | 327ms (2.9/s) | **-20%** | **+21%** |
| 1K | idx | create-table | 663ms (1.5/s) | 584ms (1.6/s) | -12% | +7% |
| 100K | idx | create-ns | 693ms (1.4/s) | 692ms (1.4/s) | 0% | 0% |
| 100K | idx | create-table | 981ms (1.0/s) | 958ms (1.0/s) | -2% | 0% |

---

## Scale Benchmark (v2, 100K–1M entries, c7i.48xlarge)

### CoW Rewrite + Index Build Time (initial seed)

| Scale | S3 Standard | S3 Express |
|---:|---:|---:|
| 100K | 0.7s | 0.4s |
| 500K | 1.6s | 1.1s |
| 1M | 3.2s | 2.0s |

### Write Throughput at Scale (c=1, with index)

| Scale | S3 create-ns p50 (tput) | S3X create-ns p50 (tput) |
|---:|---:|---:|
| 100K | 693ms (1.4/s) | 280ms (3.5/s) |
| 300K | 985ms (1.0/s) | 582ms (1.7/s) |
| 500K | 1559ms (0.6/s) | 933ms (1.1/s) |
| 700K | 2174ms (0.5/s) | 1304ms (0.8/s) |
| 1M | 2873ms (0.3/s) | 1791ms (0.6/s) |

### Indexed Point Lookup — describe-table (warm, c=1)

| Scale | S3 p50 | S3X p50 |
|---:|---:|---:|
| 100K | 45ms | 9ms |
| 500K | 47ms | 9ms |
| 1M | 54ms | 9ms |

**Flat from 100K to 1M** — BTree index makes point lookups O(log n).

### Bitmap Scan — list-namespaces (warm, c=1)

| Scale | S3 p50 | S3X p50 |
|---:|---:|---:|
| 100K | 51ms | 14ms |
| 500K | 67ms | 34ms |
| 1M | 101ms | 65ms |

Linear with result count. Still under 100ms on S3X at 1M.

---

## Summary

| Metric | S3 Standard | S3 Express |
|---|---:|---:|
| Pure manifest write (1K, no-idx, c=1) | 224ms (4.3/s) | **146ms (6.8/s)** |
| Declare table (1K, no-idx, c=1) | 251ms (3.8/s) | **159ms (6.2/s)** |
| Declare table (1K, idx, c=1) | 355ms (2.7/s) | **193ms (5.1/s)** |
| Indexed point lookup (1M, warm) | 54ms | **9ms** |
| List namespaces (1M, warm) | 101ms | **65ms** |
| CoW full rewrite + 3 indices (1M) | 3.2s | **2.0s** |
