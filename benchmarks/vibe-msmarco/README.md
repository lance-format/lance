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

Indexes are built once with the oldest runtime (`v11.0.0`) so every version
queries the same on-disk IVF_RQ files. Current Lance can read those released
indexes; older runtimes cannot read indexes written by the latest writer.
