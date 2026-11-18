# Hierarchical Lloyd vs hierarchical e-means: sweep

Base scale: 67,108,864 DINO10B vectors, k=262,144, 96 threads. Dimensions below 1,024 project onto the leading principal components.

## By dimension

| d | Lloyd s | e-means s | speedup | 1-thread speedup | MSE e-means / Lloyd | efficiency Lloyd | efficiency e-means | idle Lloyd | idle e-means | root share Lloyd | root share e-means |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 136.7 | 10.4 | 13.11x |  | 0.9981 [0.9975, 0.9987] |  |  | 91% | 77% | 2% | 15% |
| 128 | 132.5 | 15.8 | 8.38x | 2.94x | 0.9984 [0.9979, 0.9989] | 3% | 10% | 89% | 75% | 2% | 13% |
| 256 | 139.7 | 24.0 | 5.81x |  | 0.9989 [0.9985, 0.9994] |  |  | 84% | 70% | 2% | 10% |
| 512 | 155.9 | 42.2 | 3.69x |  | 0.9986 [0.9982, 0.9990] |  |  | 75% | 59% | 2% | 7% |
| 1024 | 205.1 | 67.9 | 3.02x |  | 0.9983 [0.9980, 0.9986] |  |  | 56% | 45% | 2% | 6% |

## One-thread time split (seconds)

| d | algorithm | node training: assign | node training: update | membership: assign all node rows | leaf means | sampling and gather | other | assign share | update share |
|---|---|---|---|---|---|---|---|---|---|
| 128 | lloyd | 462.9 | 44.0 | 108.8 | 4.9 | 8.3 | 2.7 | 92.1% | 7.9% |
| 128 | emeans | 49.4 | 3.1 | 108.6 | 4.9 | 8.2 | 3.7 | 95.2% | 4.8% |

## e-means before and after this revision

Same seeds; `before` is `bin/emeans_bench_baseline`.

| d | before s | after s | speedup | MSE after / before |
|---|---|---|---|---|
| 64 | 10.8 | 10.4 | 1.04x | 1.0000 |
| 128 | 15.1 | 16.2 | 0.93x | 1.0000 |
| 256 | 22.9 | 22.8 | 1.01x | 1.0000 |
| 512 | 44.9 | 42.0 | 1.07x | 1.0000 |
| 1024 | 96.6 | 54.1 | 1.78x | 1.0000 |

## e-means batch size (x the batch law)

| d | value | time / Lloyd | MSE / Lloyd | idle share |
|---|---|---|---|---|
| 128 | 0.5 | 0.11 | 0.9981 [0.9976, 0.9986] | 74% |
| 128 | 1.0 | 0.12 | 0.9984 [0.9979, 0.9989] | 75% |
| 128 | 2.0 | 0.12 | 0.9988 [0.9983, 0.9992] | 74% |

## e-means epochs

| d | value | time / Lloyd | MSE / Lloyd | idle share |
|---|---|---|---|---|
| 128 | 2 | 0.11 | 0.9998 [0.9993, 1.0003] |  |
| 128 | 3 | 0.12 | 0.9984 [0.9979, 0.9989] | 75% |
| 128 | 5 | 0.14 | 0.9975 [0.9971, 0.9980] |  |
