# VIBE MSMARCO IVF_RQ Search Latency

Compare Lance IVF_RQ vector-search latency across recent releases on
[`lance-format/vibe-msmarco-qwen-1024`](https://huggingface.co/datasets/lance-format/vibe-msmarco-qwen-1024).

The chart covers **IVF_RQ1** and **IVF_RQ5** in two cache states:

- **warm** — every index partition is loaded with `prewarm_index` before timing
- **cold** — the index metadata is opened (`describe_indices` / `index_stats`)
  but partitions are not prewarmed

Queries project only `_rowid` so the timed path does not read payload columns.

## Dataset

| Field | Value |
| --- | --- |
| Corpus | 8,840,823 unit-normalized 1024-d float32 vectors |
| Queries | 1,000 (default timed subset: first 100) |
| Metric | cosine |
| Index | `IVF_RQ` with `num_bits=1` and `num_bits=5` |
| Partitions | 1024 |
| Search | `k=10`, `nprobes=20` |

## Run

```sh
# 1. Download the Hugging Face snapshot (~36 GB)
python bench.py download --data-dir /tmp/lance-vibe-msmarco

# 2. Build indexes and time one pylance interpreter
python bench.py run \
    --data-dir /tmp/lance-vibe-msmarco \
    --label v12.0.0 \
    --out results/v12.0.0.json

# 3. Plot every results/*.json
python plot.py --results-dir results --out results/ivf_rq_latency.png
```

`run_versions.py` installs isolated interpreters for the last two stable
pylance wheels plus a local checkout of this repo, then runs the full matrix.

Each version builds its own `IVF_RQ1` and `IVF_RQ5` indexes in a private
work corpus (fragment files are hardlinked from the snapshot; manifests are
copied). Do not share one on-disk index across runtimes when comparing
version performance.

## Measured results (2026-09-19)

Machine: 4 vCPU, 15 GiB RAM. 100 queries, `k=10`, `nprobes=20`, `_rowid` only.
Each pylance version writes its own `IVF_RQ1` / `IVF_RQ5` in a private work
corpus.

| Index | Version | Cold mean (ms) | Cold median (ms) | Warm mean (ms) | Warm QPS |
| --- | --- | ---: | ---: | ---: | ---: |
| IVF_RQ1 | v11.0.0 | 17.9 | 16.7 | 6.02 | 166 |
| IVF_RQ1 | v12.0.0 | 14.4 | 14.1 | 4.91 | 204 |
| IVF_RQ1 | c8f182179 (main) | 20.0 | 18.2 | 5.03 | 199 |
| IVF_RQ5 | v11.0.0 | 24.4 | 23.3 | 12.8 | 78 |
| IVF_RQ5 | v12.0.0 | 22.0 | 21.2 | 10.6 | 95 |
| IVF_RQ5 | c8f182179 (main) | 25.1 | 24.4 | 10.4 | 97 |

Warm search is close across the two stables and main (RQ1 ~5–6 ms, RQ5
~10–13 ms). Cold search is also in the same band once each version reads
an index it wrote. The earlier chart that reused a v11-written index made
main look much slower on cold; that was a cross-version reader path, not
main's own IVF_RQ performance.

Index build wall time on this box: v11 RQ1/RQ5 420s/466s, v12 227s/457s,
main 223s/461s.

![IVF_RQ search latency](results/ivf_rq_latency.png)
