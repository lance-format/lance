# Lance architecture guide

This guide maps the main ownership boundaries in the Lance repository. Use it to
find the module that owns a behavior before changing an implementation, its
persistent representation, or its language binding.

## Repository layers

Lance has three visible surfaces and a layered Rust core:

```text
Python API -> PyO3 bindings --\
Java API -----> JNI bindings ---+-> lance (Dataset and query orchestration)
                                |
                                +-> scanner planners -> io::exec physical nodes
                                |                          |
                                |                          +-> lance-index
                                |                          +-> lance-file / lance-io
                                |
                                +-> lance-table (manifest and transactions)
                                +-> lance-index (index algorithms and storage)
                                +-> lance-file / lance-encoding / lance-io
                                +-> lance-datafusion* / lance-linalg / lance-select
```

The dependency direction is downward. Bindings translate Python or Java values
and preserve lifetime semantics; they do not define table behavior. The
`lance` crate coordinates datasets, queries, writes, and commits. Lower-level
crates own durable metadata, index algorithms, physical file I/O, encoding,
expression evaluation, and row-selection primitives.

## Repository map

The tree below shows the main code entry points. It is intentionally a
navigation map, not an exhaustive file inventory.

```text
lance/
|-- rust/
|   |-- lance/                         # Dataset API and orchestration
|   |   `-- src/
|   |       |-- dataset/
|   |       |   |-- scanner.rs         # Scanner state and public query entry
|   |       |   |-- scanner/           # builder, plan, source, FTS, vector, ...
|   |       |   |-- optimize.rs        # compaction workflow entry
|   |       |   |-- optimize/          # planning, execution, commit
|   |       |   |-- blob/              # Blob v2 codec and batch rewrite
|   |       |   |-- write/merge_insert.rs
|   |       |   |-- write/merge_insert/ # builder, job, merger, fragment update
|   |       |   |-- mem_wal/write.rs   # MemWAL shard-writer entry
|   |       |   `-- mem_wal/write/     # coordination, replay, handlers, tests
|   |       `-- index/
|   |           |-- api.rs             # Dataset index public trait
|   |           |-- lifecycle.rs       # create/drop/merge/commit/optimize entry
|   |           |-- provenance.rs      # source freshness and coverage
|   |           |-- loading.rs         # raw listing, cache keys, opening
|   |           |-- statistics.rs      # descriptions and statistics
|   |           |-- prewarm.rs         # cache residency and diagnostics
|   |           |-- segment.rs         # segment metadata/capability rules
|   |           `-- remap.rs           # streaming row-address remapping
|   |-- lance-table/                   # manifests, transactions, format metadata
|   |-- lance-index/                   # index formats, builders, scorers, search
|   |-- lance-file/                    # Lance file reader/writer
|   |-- lance-encoding/                # column encodings and decoders
|   |-- lance-io/                      # object stores, scheduling, file I/O
|   |-- lance-datafusion*/             # DataFusion planning and execution glue
|   `-- lance-select/                  # row addresses, masks, selection plans
|-- python/
|   |-- python/lance/_dataset/         # Python dataset capability modules
|   `-- src/dataset/                   # PyO3 objects grouped by capability
|-- java/
|   |-- src/main/java/org/lance/       # public Java API
|   `-- lance-jni/src/blocking_dataset/ # JNI adapters grouped by capability
`-- docs/                              # user and format documentation
```

## Shared types and invariants

Shared contracts stay with the layer that can enforce them:

| Contract | Owner |
| --- | --- |
| Stable logical row IDs, physical row addresses, deleted rows, and system columns | `lance-core` types plus `lance-table` metadata; callers pass those types rather than re-encoding the bit layout |
| Manifest, fragment, transaction, and index metadata | `rust/lance-table/src/format/` and `rust/lance-table/src/transaction/` |
| Row masks and selection algebra | `rust/lance-select/src/` |
| Index query, build, persistence, scoring, and posting-list algorithms | `rust/lance-index/src/` |
| File layout and codec invariants | `lance-file` and `lance-encoding` |
| Dataset-level lifecycle, coordination, and public API behavior | `rust/lance/src/dataset/` and `rust/lance/src/index/` |
| Binding argument conversion and foreign-language object lifetime | `python/src/` and `java/lance-jni/src/` |

A type moves with the behavior that constructs, validates, serializes, and
updates its invariant. Avoid introducing generic `types`, `validation`, or
`utils` modules as a destination for otherwise unrelated behavior.

## Capability ownership

### Dataset queries and scanner planning

`rust/lance/src/dataset/scanner.rs` owns the user-visible `Scanner` state and
entry points. The child modules divide the query pipeline by responsibility:

- `scanner/builder.rs` applies configuration setters and leaves cross-option
  validation to planning.
- `scanner/plan.rs` validates scan state, chooses the source family, and adds
  final ordering, limit, and projection stages.
- `scanner/source.rs` builds physical row sources, including filtered reads,
  indexed takes, overlays, deletion masks, and distributed takes.
- `scanner/fts.rs` owns FTS field and granularity resolution, coverage
  classification, residual search, and indexed/flat fallback planning.
- `scanner/vector.rs` and `scanner/minhash.rs` own their search planning and
  uncovered-row fallback.
- `scanner/aggregate.rs` owns aggregate expression construction and lowering.

The scanner selects and composes a source with the coverage, row identity,
deletion, and overlay handling required by the query. `io::exec` nodes execute
the selected plan and apply row filters and masks while streaming.

### FTS execution

`rust/lance/src/io/exec/fts.rs` is the execution facade. Its child modules own
physical query shapes:

- `match_query`, `phrase`, `combined_fields`, and `boolean` own indexed query
  nodes.
- `flat_match` and `flat_combined_fields` own flat scans over uncovered rows.
- `compound`, `corpus`, and `selection` own mixed indexed/flat coordination,
  shared BM25 corpus state, and exact segment selection.
- `segments`, `document`, `matches`, and `query_tokens` adapt persisted index
  segments and schema-shaped documents into the execution nodes.

Dataset coverage and fallback decisions remain in `dataset/scanner/fts.rs`.
Query syntax, posting lists, BM25 preparation, and ranking algorithms remain in
`lance-index`.

### Compaction and Blob rewrite

`rust/lance/src/dataset/optimize.rs` keeps the workflow entry and re-exports the
public compaction contracts. The adjacent modules separate the lifecycle:

- `optimize/planning.rs` owns serializable options and plans, candidate
  selection, index-aware grouping, and task construction. It does not rewrite
  data or publish a manifest.
- `optimize/execution.rs` owns task execution and `RewriteResult`. It writes
  replacement fragments but does not publish them.
- `optimize/commit.rs` owns concurrent commit coordination, index remapping,
  fragment publication, and partial-result semantics.
- `dataset/blob/rewrite.rs` owns shared Blob v2 batch classification,
  descriptor-to-payload conversion, null/offset preservation, and the
  `BlobV2BatchRewritePlan` type.

Blob rewrite is a Blob-domain capability shared by compaction, updates,
merge-insert, scans, and managed clones. A caller supplies a schema, a batch,
and row addresses; it does not duplicate Blob traversal or descriptor logic.

### Dataset index lifecycle

`rust/lance/src/index.rs` is the module facade and keeps
`DatasetIndexExt` as one coherent public implementation entry. Its child modules
own the lifecycle stages:

- `lifecycle.rs` creates, drops, merges, commits, lists, and optimizes logical
  indexes. It coordinates the stages below without taking their private state.
- `provenance.rs` determines whether committed segment coverage is still fresh
  under staged replay, overlays, row revival, and fragment-reuse remapping.
- `loading.rs` performs raw manifest listings, FRI bookkeeping, cache-key
  construction, opening, and the internal loading trait.
- `statistics.rs` describes indexes and computes statistics, including the
  existing legacy auto-migration path.
- `prewarm.rs` loads selected segments into caches and reports residency.
- `segment.rs` validates persisted segment metadata and capability predicates.
- `remap.rs` streams row-address translation for data rewrites.

The dependency direction is from lifecycle coordination toward provenance,
loading, statistics, prewarm, segment, and remap. `statistics.rs` may consume the
raw FRI manifest lookup from `loading.rs`; loading must not depend on
statistics. `lance-index` remains the owner of concrete index formats,
algorithms, and readers.

### Merge-insert

`rust/lance/src/dataset/write/merge_insert.rs` keeps the public operation types
and the overall entry point. Child modules divide one execution:

- `builder.rs` validates keys and operations and builds an immutable job.
- `job.rs` plans and executes the source/target join, target-base resolution,
  retries, and transaction assembly.
- `merger.rs` applies matched and not-matched semantics to joined rows and
  tracks staged mutations.
- `update_fragments.rs` groups updates, writes replacement fragments, preserves
  row lineage, and reconciles row addresses.

The merge plan, Blob inventory, stable row IDs, deletion state, and target
version are carried explicitly through these stages. Publication stays with the
dataset commit path.

### MemWAL writes

`rust/lance/src/dataset/mem_wal/write.rs` exposes the shard-writer entry.
Adjacent modules own independent parts of one persistent writer lifecycle:

- `config.rs` owns shard-writer settings and validation.
- `writer.rs` owns `ShardWriter`, resident memtables, write admission, and
  lifecycle coordination.
- `backpressure.rs` owns memory accounting, durability notifications, and
  admission control.
- `tasks.rs` and `handlers.rs` own ordered background dispatch and WAL, index,
  and memtable-flush handlers.
- `schema.rs` derives the storage schema, validates incoming writes, and shapes
  rows and tombstones for persistence.
- `replay.rs` reconstructs memtables and resumes cursors from the WAL.
- `stats.rs` exposes snapshots of write, WAL, and memtable state.

WAL ordering, writer epochs, generation boundaries, and durable cursors are
lifecycle invariants. They stay in the MemWAL write modules rather than being
copied into Python, Java, or benchmark code.

### Python and PyO3

`python/python/lance/dataset.py` remains the public `LanceDataset` entry.
`python/python/lance/_dataset/` contains capability-oriented Python objects and
helpers such as `scanner`, `index`, `optimize`, `cleanup`, `refs`,
`transaction`, `delta`, `sql`, `stats`, and `merge_insert`.

`python/src/dataset.rs` remains the central `_Dataset` PyO3 object.
`python/src/dataset/` contains capability modules for operations that have
distinct conversion or lifetime rules, including writes, merge-insert, SQL,
delta, base paths, cleanup, I/O statistics, compaction, and statistics.

Python modules may validate Python-only calling conventions, coerce Python
inputs, and preserve Python object behavior. Dataset semantics, row identity,
transactions, and error meaning stay in Rust.

### Java and JNI

`java/src/main/java/org/lance/Dataset.java` remains the public Java dataset
object. It owns the native handle, allocator/session lifetime, close behavior,
and the lifecycle and index-build locks shared by all operations.

`java/lance-jni/src/blocking_dataset/` divides the native adapter surface:

- `construction` and `conversions` open datasets and translate handles,
  versions, storage options, and namespace objects.
- `read` and `schema` own reads, statistics, schema inspection, and schema
  evolution.
- `mutation`, `maintenance`, `index`, and `refs` own writes, cleanup/compaction,
  index operations, and version/tag/branch operations.

JNI modules borrow the Rust dataset through the single Java handle. They must
not create a second handle lifecycle or copy Java lock behavior into native
state.

## Finding tests

Tests stay with the contract they protect:

| Area | Test entry |
| --- | --- |
| Scanner planning and behavior | Module-local tests in `dataset/scanner/` and integration tests in `dataset/tests/dataset_scanner_*.rs` |
| FTS execution | Inline tests in `io/exec/fts.rs` and its owner submodules; dataset-level plan and segment behavior in `dataset/tests/dataset_scanner_*.rs` |
| Compaction | `dataset/optimize/tests/` grouped by planning, execution, commit, overlays, lineage, indexes, vectors, and Blob behavior |
| Blob rewrite | Unit tests in `dataset/blob/rewrite.rs` and compaction-path tests in `dataset/optimize/tests/blob.rs` |
| Dataset indexes | Private-state tests beside each index owner module; cross-module behavior in `index/tests/` grouped by create, lifecycle, loading, merge, nested fields, optimize, and statistics |
| Merge-insert | Job-owned planning, unit-state, indexed, and lineage tests in `dataset/write/merge_insert/job/tests/`; public operation, concurrency, semantics, deferred-condition, subcolumn, and type tests in `dataset/write/merge_insert/tests/` |
| MemWAL writes | Tests beside owner modules plus `dataset/mem_wal/write/tests/` for end-to-end writes, WAL replay, schema evolution, and integration |
| Python API and PyO3 | Python tests grouped by user capability, plus Rust binding tests where FFI conversion or lifetime behavior is the contract |
| Java API and JNI | Java suites by capability, plus JNI-local tests for native conversion and error behavior |

Put a test where it can observe the invariant without widening production
visibility. Use cross-module integration tests for behavior that spans planner,
execution, and commit boundaries.

## Maintenance entry points

### Change compaction candidate selection

Start in `rust/lance/src/dataset/optimize/planning.rs`:

1. Read `DefaultCompactionPlanner::candidacy_of` for the per-fragment decision
   between self-compaction, neighbor compaction, and no rewrite.
2. Follow `CompactionPlanner::plan` for index boundaries, grouping, output-size
   limits, and serializable `CompactionPlan` construction.
3. Read `optimize/execution.rs` only to understand what the selected task will
   write; publication remains in `optimize/commit.rs`.
4. Update `dataset/optimize/tests/planning.rs` for selection and grouping
   contracts.

### Change FTS coverage or fallback

Start in `rust/lance/src/dataset/scanner/fts.rs`:

1. Follow `fts_search_source` and the FTS arm of `Scanner::create_plan` in
   `scanner/plan.rs` to see how the scanner chooses an indexed, residual, hybrid,
   or flat source.
2. Read `has_complete_hybrid_fts_coverage` and the segment/overlay helpers in
   the same module before changing when the index can replace a scan.
3. Change `io/exec/fts/` only when the physical node behavior changes; the
   dataset planner remains the owner of coverage and fallback selection.
4. Update `dataset/tests/dataset_scanner_plans.rs`,
   `dataset_scanner_segments.rs`, and the relevant `io/exec/fts` owner tests.

### Change Blob rewrite

Start in `rust/lance/src/dataset/blob/rewrite.rs`:

1. Read `BlobV2BatchRewritePlan::{try_new, output_schema, transform_batch}` and
   `transform_blob_v2_batch`.
2. Check every caller with Blob v2 data: scan, update, merge-insert, managed
   clone, and compaction. They share this module instead of carrying a local
   copy.
3. Preserve nulls, nested struct/list offsets, external references, row count,
   `_rowaddr` handling, and managed descriptor behavior.
4. Update the module tests and `dataset/optimize/tests/blob.rs` when the
   transformation or a compaction integration path changes.
