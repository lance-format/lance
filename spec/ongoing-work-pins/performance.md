# Ongoing-work pins: performance expectations

This accompanies the [design](README.md) and [test cases](tests.md). Pins add manifest writes, authoritative-head resolution, history checks, and retained storage. Measure each cost separately. Do not remove correctness checks to meet a latency target.

## Expectations

| Area | Expectation | Evidence |
| --- | --- | --- |
| Lifecycle | Register, renew, release, and inspection operate on metadata; no table-row scan or rewrite. No-op mutations publish no new version. | Storage requests and bytes by operation, commit count, median/p95 latency, and retries. |
| Ordinary writes | Preserve the existing commit/head-discovery protocol. Carrying pins and pruning expired records add no complete-history listing or separate commit. Use the attempt's predecessor and persist required evidence. | Request counts and CPU time for never-flagged, flagged-empty, unexpired, expired, and mixed pin lists. Separate metadata processing from conflict/retry work. |
| Expiration and publication release | Expiration rides an existing attached commit. Cleanup never creates an expiration version. Combined publication/release uses one commit. | Count committed versions and storage requests; include an idle table, where retention persists until a write or explicit release. |
| Tables without active pins | No unexplained regression. Measure never-flagged tables separately from sticky-flag tables with empty lists. Pin and maintenance decisions still use authoritative resolution. | Open/read, ordinary write, and cleanup comparisons against the same base revision. |
| More pins | Stored metadata grows with the records. Reuse source metadata and file deduplication; do not scan identical source data once per pin. | Manifest size, source-manifest reads, commit/cleanup time, and request counts at 0, 1, 10, and 100 pins. These are workloads, not a supported pin limit. |
| Pin and maintenance head resolution | Honor authoritative discovery, including all unordered pages. Account for history-dependent listing costs where the stronger check is required. | Listing pages, reads, finalization requests, and latency versus history length, separately from pin processing and ordinary-write baselines. |
| FRI checks | Determine retention from metadata and fragment references, without reading row-map payloads solely to decide retention. | CPU/time, source reads per attempt, cache behavior, and whole-record retention costs across long and overlapping chains. |
| Version window | Retain the complete agreed window, with explainable storage cost and reclamation when requirements end. | Unique bytes by data, index, overlay, deletion, mapping, and manifest/transaction class; cleanup and publication-history replay costs. |

## Proposed benchmark matrix

Use an instrumented local object store for deterministic request counts, plus the direct and external commit approaches in the test matrix. Add representative remote runs for the deployment backends being enabled.

- Pin counts: 0, 1, 10, and 100, including shared-source and distinct-source metadata. Use prepared fixtures where necessary; fresh current-version registrations naturally advance the head.
- Attached history: 10, 100, and 1,000 versions, including reclaimed gaps outside the retained window and unordered pagination.
- Writer contention: no competing writer, one writer, and four writers; include renewal traffic. Report registration retry rates and successful latency while preserving the request ID across rebases.
- Expiration: compare ordinary commits with unexpired, expired, and mixed lists. Measure pruning CPU cost and verify no extra commit or complete-history listing. Compare atomic publication/release with two separate commits, and confirm cleanup performs no expiration commits on idle tables.
- Retention workloads: append-only, repeated compaction, column replacement, and overlapping jobs at different sources. Vary oldest-source age through logical timestamps and actual rewritten bytes. A small fixture's age alone does not model a day's write volume.
- FRI workloads: version 0, version 1, and version 1 histories containing legacy records; short and long chains; converging fragment paths; cold and warm metadata caches. Separate movement generation cost from the added retention checks.

These are proposed dimensions for review, not a request to run their full Cartesian product. Choose a small baseline and vary one factor at a time. Agree on representative data sizes and write rates before drawing rollout conclusions.

Use the same fixtures and compiler for base-versus-change comparisons, with the repository's optimized `release-with-debug` profile. Record machine, backend, revision, configuration, repetitions, and variability. Keep setup outside operation timings. Compare equivalent outcomes: a failed publication is not a successful publication baseline.

## Existing feasibility evidence

A local experiment at revision `ca3d16dccc136cdea9e207dddc6ce664434a3ddf` compared source-plus-FRI retention with complete-window retention using real cleanup and staged BTree publication. Pins were simulated with tags and a cleanup floor; production pin APIs were not implemented.

The source had 4,096 rows in eight fragments, two integer fields, and a 128-character payload. Three workloads each ran three times: three 512-row appends; those appends plus three deferred compactions of the original fragments; and three replacements of a column other than the pending index's key. The fixture used physical row addresses and FRI version 1, with staged output inside the ordinary unverified-file grace period.

All nine source-plus-FRI runs failed safely because an intermediate manifest was missing. All nine complete-window runs published and passed indexed-query comparisons with scans. Extra unique file-length bytes were:

| Workload | Extra retained bytes | Of which data |
| --- | ---: | ---: |
| Appends | 4,872 | 0 |
| Appends and compactions | 1,130,936–1,131,000 | 1,109,818–1,109,882 |
| Column replacements | 42,772 | 35,296 |

About 98.1% of the compaction fixture's additional storage was data. Ending the last oldest simulated job advanced the retention floor and allowed reclamation. The accepted first implementation keeps this complete window; a future metadata-only protocol is separate work.

This was a small, warm, unoptimized local experiment. It establishes neither a production latency target nor a universal publication protocol. It did not test real pin lifecycle races, external stores, SRID, spilled transactions, external row maps, overlays, or high commit rates. Make its raw measurements and reproduction harness available with the review evidence; do not treat this summary as a reproducible benchmark by itself.

## Approval and regression gates

The structural expectations above can become automated checks: no row I/O for lifecycle operations, no version on a no-op, deduplicated resource accounting, and preserved correctness under all approved workloads.

Numerical latency, throughput, request-count, and storage-envelope limits remain to be agreed after baseline measurements. They are not yet satisfied or waived. Before implementation approval, record:

| Item | Value to agree |
| --- | --- |
| Representative baseline | Table/fragment size, history length, write rate, and direct/external backend environments. |
| Baseline regression limits | Allowed changes for never-flagged and flagged-empty tables, separately. |
| Active-pin operation budgets | Latency, metadata I/O, and contention expectations at the selected pin counts. |
| Retention envelope | Expected rewrite volume and oldest-source age, plus retained-byte accounting and release/expiration reclaim checks. |

Retained bytes depend on rewritten generations and how long the oldest pin remains listed. The 24-hour default does not cap this cost: a quiet table may retain expired pins indefinitely until another commit or explicit release. Report that delay separately from cleanup scheduling. Agree an operational envelope using representative workload volume; do not silently weaken retention or remove a renewed pin to enforce it.

Put the agreed performance goals and verification plan in the PR description as well. At final review, attach repeated optimized comparisons, the exact environment, and an explanation for material regressions. Keep noisy wall-clock benchmarks separate from deterministic request-count tests.
