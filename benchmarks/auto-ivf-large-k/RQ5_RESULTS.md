# IVF_RQ 5bit recall with adaptive IVF probing

Auto's existing metric/k profiles now apply across IVF index types. The
partition policy does not inspect the sub-index or quantization type, and
historical IVF readers do not need prepared-search metadata to use it.
The measured revision retained Float32 and refinement guards and enabled
profiles through k=1000. The current default policy also supports other vector
types, arbitrary refinement factors and profiles through k=100000; this report
retains the original workload, source identity and measurements.

This experiment uses IVF_FLAT to isolate routing effects, then measures
the final recall of IVF_RQ 5bit with the same centroids, exact row-to-partition
membership, query split and frozen Auto parameters. All 15,360 paired queries
search the same number of partitions under FLAT Auto and RQ5 Auto.

RQ5 Auto mean strict-ID recall ranges from **86.719% to 96.227%**.
**6/30** corpus/k groups reach 95% mean final recall.
The FLAT routing calibration target is not a final-recall guarantee for RQ5.
Parameters and RQ models were frozen before held-out evaluation and were not
adjusted to improve these results.

## RQ5 Auto: final recall

| Corpus | Metric | k=1 | k=10 | k=100 | k=200 | k=500 | k=1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINO | l2 | 86.719% | 90.391% | 92.508% | 93.438% | 93.997% | 94.158% |
| LAION | cosine | 91.211% | 93.828% | 95.094% | 96.227% | 96.115% | 96.023% |
| FineWeb | cosine | 91.992% | 92.910% | 94.094% | 94.806% | 95.041% | 95.113% |
| Wiki-Cohere | dot | 92.188% | 90.625% | 92.021% | 92.189% | 92.416% | 92.548% |
| DPR | dot | 89.063% | 90.957% | 92.295% | 92.391% | 92.643% | 92.732% |

Each entry is the mean over the same 512 held-out queries for that corpus.
All queries return k distinct IDs. Ground truth uses the original float64
exact-neighbor prefixes; alternative tied IDs count as misses.

## FLAT Auto on the same queries

| Corpus | Metric | k=1 | k=10 | k=100 | k=200 | k=500 | k=1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINO | l2 | 96.484% | 95.215% | 95.107% | 96.028% | 96.029% | 95.941% |
| LAION | cosine | 94.336% | 95.859% | 96.074% | 97.189% | 96.992% | 96.750% |
| FineWeb | cosine | 95.117% | 94.902% | 95.256% | 96.005% | 96.076% | 96.091% |
| Wiki-Cohere | dot | 94.336% | 94.121% | 94.998% | 95.280% | 95.507% | 95.672% |
| DPR | dot | 96.484% | 96.113% | 95.787% | 95.682% | 95.680% | 95.676% |

These are fresh measurements from the candidate runtime, with the same
partition budgets as RQ5 Auto. They are not the earlier full-held-out
population from the original FLAT campaign.

## RQ5 with every partition selected

| Corpus | Metric | k=1 | k=10 | k=100 | k=200 | k=500 | k=1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINO | l2 | 89.063% | 93.828% | 95.832% | 96.131% | 96.605% | 96.885% |
| LAION | cosine | 96.289% | 97.344% | 98.232% | 98.373% | 98.408% | 98.511% |
| FineWeb | cosine | 96.875% | 96.914% | 97.648% | 97.778% | 97.840% | 97.883% |
| Wiki-Cohere | dot | 97.266% | 95.039% | 95.398% | 95.319% | 95.185% | 95.160% |
| DPR | dot | 92.188% | 94.043% | 95.160% | 95.285% | 95.502% | 95.525% |

This arm fixes nprobes to the full partition count. It still uses normal-mode
quantized scoring and pruning, so it is not an exact search or a pure
quantization-only oracle. Its recall need not be an upper bound for every
individual query. Routing and quantized-search losses are not assumed to add
or multiply independently.

## Original RQ probing heuristic

| Corpus | Metric | k=1 | k=10 | k=100 | k=200 | k=500 | k=1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINO | l2 | 51.367% | 93.828% | 95.832% | 96.131% | 96.605% | 96.885% |
| LAION | cosine | 55.273% | 97.344% | 98.232% | 98.373% | 98.408% | 98.511% |
| FineWeb | cosine | 47.266% | 96.914% | 97.648% | 97.778% | 97.840% | 97.883% |
| Wiki-Cohere | dot | 40.820% | 95.039% | 95.398% | 95.319% | 95.185% | 95.160% |
| DPR | dot | 92.188% | 26.230% | 23.223% | 21.816% | 19.598% | 17.625% |

This arm uses an explicit full maximum_nprobes on the candidate to select
the original signed-distance heuristic. Before measurement, its returned IDs,
partition counts and comparison counts match the preserved pre-change PR
binary on two calibration queries at every k for every corpus (60 controls).
The full 512-query arm is this verified policy reproduction, not a second
512-query run of the old binary. No latency or throughput improvement is
claimed by this recall experiment.

## Full-partition top-1 miss audit

| Corpus | Full-RQ top-1 misses | Median true rank | Maximum true rank | Numerically indistinguishable |
| --- | ---: | ---: | ---: | ---: |
| DINO | 56 / 512 | 2 | 4 | 0 |
| LAION | 19 / 512 | 2 | 3 | 0 |
| FineWeb | 16 / 512 | 2 | 3 | 0 |
| Wiki-Cohere | 14 / 512 | 2 | 3 | 0 |
| DPR | 40 / 512 | 2 | 3 | 0 |

For every full-partition RQ5 top-1 miss, raw source vectors are rescored in
float64 and the returned ID is located in the frozen exact top-100,000 list.
The ground-truth vector's recomputed distance agrees with the stored oracle.
Numerical indistinguishability uses a tolerance of
1e-10 * max(1, abs(oracle_distance)); it does not remove any strict-ID miss.
[rq5-numerical-audit.json](rq5-numerical-audit.json) preserves every checked ID,
true distance, rank and distance regret. Original recall scores are unchanged.

## Reproduction and validation

- Contract: [RQ5_PROTOCOL.md](RQ5_PROTOCOL.md). RQ5 uses 5 bits, the default
  fast rotation and packed storage, approx_mode=normal, no refinement, no
  filters, and query_parallelism=1. Full-partition queries are run separately
  for every k because pruning can depend on k.
- Host: AWS r8i.8xlarge, 32 vCPUs / 256 GiB, Intel Xeon 6975P-C, repository
  release-with-debug profile. Recall collection uses eight concurrent query
  workers, CPU affinity 0-15, 16 Lance/Rayon threads and one BLAS/OMP thread.
  Saved query durations are diagnostic and are not serial latency measurements.
- Frozen inputs: DINO 10M x 1024, LAION/FineWeb 10M x 768, Wiki-Cohere 35M x
  768 and DPR 21,015,300 x 768. Partition counts are 2,441, 2,441, 2,441,
  8,545 and 5,131. Each RQ5 index is built in a separate shallow clone using
  the original centroid matrix and precomputed assignments. Every row's
  membership and every centroid value are checked for equality with FLAT.
- [rq5-native-results.csv](rq5-native-results.csv) contains all 120 arm/k/corpus
  summaries, recall distributions, partition/comparison counts and paired
  differences. Recall and overlap columns are fractions; difference columns
  use percentage points. [rq5-models.json](rq5-models.json) preserves the
  random rotation models. [rq5-run-integrity.json](rq5-run-integrity.json)
  records source, binary, input and index identities.
- Independent audit recomputes recall from every saved returned-ID array,
  checks query coverage and split separation, validates full-partition counts,
  and confirms zero reported storage bytes read after prewarming. A small
  end-to-end test also verifies that duplicate returned IDs are rejected.
- Implementation validation: 314 Rust KNN tests, 34 Python Auto-probe tests
  on development and optimized builds, the full vector-index module
  (138 passed / 11 existing skips), workspace Clippy/formatting and Python
  lint/type checks pass. The final regression fails on the old binary for
  all 12 non-FLAT index/segment cases, while both FLAT controls pass.

These results cover the recorded unfiltered Float32 corpora and one frozen
RQ5 model per corpus. They do not establish final recall for PQ, SQ, HNSW,
other data, other RQ modes, filtering or refinement.
