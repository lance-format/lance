# Hierarchical Lloyd vs hierarchical e-means on DINO10B

Compares Lance's default IVF k-means, hierarchical k-means whose nodes run
Lloyd, with the same hierarchy running e-means (arXiv:2611.09501) at every
node (`HierarchicalNodeAlgorithm::EMeans`). Both train through `train_kmeans`
with the parameters Lance's IVF build uses: fan-out 16, 50 Lloyd iterations,
balance factor 1 and `256 * k` training rows. e-means uses its defaults (3
epochs, batch size and step from the paper's scaling laws) and ignores the
balance factor.

It reports:

- training time and validation MSE over the 100,000 held-out DINO10B queries,
  over several seeds, with a paired bootstrap interval on the MSE ratio,
- with one thread, where training time goes: node assignment and update,
  assigning every row of a node to its children, leaf means and sampling,
- with every thread, how thread-seconds (threads times wall time) split into
  busy work by kind and idle time, how the busy work grows over the
  one-thread run, and how many threads are busy over time,
- how wall time and parallel efficiency scale with the thread count, against
  the speedup an embarrassingly parallel distance computation reaches on the
  same machine.

The time breakdown comes from the `kmeans-profile` feature of `lance-index`
(see `vector::kmeans::profile`); speed and quality come from a build without
it.

## Data

DINO10B is shuffled, so a prefix of the first chunk is a uniform sample. Each
`.bvecs` row is a little-endian `u32` dimension (1,024) followed by 1,024
`u8` components.

```bash
mkdir -p ~/data/dino10b && cd ~/data/dino10b
URL=https://dl.fbaipublicfiles.com/large_objects/dino_vitl_10B
curl -O $URL/queries_clean.bvecs
curl -r 0-$((8388608 * 1028 - 1)) -o base.bvecs $URL/chunked_base_10B/chunk_0000.bvecs
```

The default scale, 8,388,608 vectors held as `f32`, needs about 36 GB of
memory.

## Build

The profiled and unprofiled builds differ in a feature of `lance-index`, so
build one, copy it, then build the other:

```bash
mkdir -p ~/data/dino10b/bin
cargo build --release -p lance-index --example emeans_bench
cp target/release/examples/emeans_bench ~/data/dino10b/bin/emeans_bench_timing
cargo build --release -p lance-index --features kmeans-profile --example emeans_bench
cp target/release/examples/emeans_bench ~/data/dino10b/bin/emeans_bench_profile
```

## Run

```bash
DATA=~/data/dino10b OUT=results ./benchmarks/emeans/run.sh
uv run benchmarks/emeans/analyze.py --results results
```

`ROWS`, `K`, `SEEDS`, `MAX_THREADS` and `SCALING_THREADS` override the scale,
seeds and thread counts. `analyze.py` writes `summary.md`, `headline.png`,
`breakdown.png`, `scaling.png` and `utilization.png` into the results folder.

DINO10B is described at <https://swannbessa.github.io/blog/dino10b/>; cite the
e-means paper when using it.

## Sweep

`sweep.sh` runs both methods across dimensions (PCA of DINO10B, fitted with `pca.py`), e-means batch sizes and epochs, thread counts, and 33.5M vectors with k=131,072; `sweep_analyze.py` writes `summary.md` and figures into the sweep folder:

```bash
uv run benchmarks/emeans/pca.py --base ~/data/dino10b/base.bvecs --out ~/data/dino10b/pca.f32
DATA=~/data/dino10b OUT=sweep ./benchmarks/emeans/sweep.sh
uv run benchmarks/emeans/sweep_analyze.py --sweep sweep
```

The 33.5M runs need the first 33,554,432 rows of the chunk (34.5 GB).
