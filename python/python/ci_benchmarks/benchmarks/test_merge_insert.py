# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmarks for `merge_insert`.

merge_insert has two execution paths and the routing between them is
structural, not a flag.  `use_index` is the one knob that selects between them,
so every benchmark here is parametrized on it:

  ``v1_indexed`` — ``use_index(True)`` with an indexed key.  Takes the legacy
      indexed-scan path (``create_indexed_scan_joined_stream``).
  ``v2_hash`` — ``use_index(False)``.  Disables the index gate in
      ``can_use_create_plan``, so the DataFusion path
      (``LanceRead + HashJoin``) runs instead.

For a partial-schema source the write sink is a second, independent choice, made
with ``write_mode``.  Under the default ``"auto"`` v1 patches columns in place
(``UpdateMode::RewriteColumns``) while v2 rewrites whole rows
(``RewriteRows``); the ``v2_rewrite_columns`` variants of
``test_update_subset_*`` ask v2 for the patching sink instead.  That makes
``write_bytes`` the interesting metric for the ``test_update_*`` benchmarks:
which sink writes less depends on how wide the omitted columns are and how many
rows are matched, so these sweeps are where the crossover shows up.

Targets are mutated, so each measured run is preceded by an untimed restore to
the ``merge_insert_base`` tag written by ``datagen/merge_insert.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Sequence

import lance
import numpy as np
import pyarrow as pa
import pytest
from ci_benchmarks.datagen.merge_insert import (
    BASE_TAG,
    DELETED_ROW_STRIDE,
    FRAGS_NUM_ROWS,
    FRAGS_SCHEMA,
    NARROW_NUM_ROWS,
    NARROW_SCHEMA,
    UNINDEXED_TAIL_ROWS,
    WIDE_NUM_ROWS,
    WIDE_ROWS_PER_FRAGMENT,
    WIDE_SCALAR_COLUMNS,
    WIDE_SCHEMA,
    WIDE_VECTOR_DIM,
    narrow_batch,
)
from ci_benchmarks.datasets import get_dataset_uri

if TYPE_CHECKING:
    from lance.dataset import ExecuteResult

PLANS = ["v1_indexed", "v2_hash"]

# Partial-column updates have a third shape: v2 asked for the column-patching
# sink, which is the only v2 sink that does not rewrite whole rows.
WRITE_PLANS = ["v1_indexed", "v2_hash", "v2_rewrite_columns"]

# Brackets the cold-random break-even, which the design analysis puts at
# roughly target_rows / 4096 -- about 2.4K rows for the 10M-row narrow target.
SOURCE_SIZES = [1_000, 10_000, 100_000]

NARROW_KEYS = ["id_int", "id_uuid7", "id_uuid4"]

# DataFusion accounts each 8,192-row slice of the 1.32 GiB full-schema source
# for its shared backing buffers. Budget the resulting ~159 GiB of reservations
# without changing the source layout measured by this benchmark.
_MEM_POOL_SIZE = 160 * 1024**3


@pytest.fixture(scope="module", autouse=True)
def _merge_insert_mem_pool() -> Iterable[None]:
    """Use a bounded pool large enough for every merge_insert benchmark."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("LANCE_MEM_POOL_SIZE", str(_MEM_POOL_SIZE))
        yield


# ---------------------------------------------------------------------------
# Target management
# ---------------------------------------------------------------------------


@dataclass
class Target:
    """A merge_insert target that can be rewound to a pristine state.

    ``reset`` returns a handle sitting on a fresh version whose contents match
    the ``merge_insert_base`` tag.  It is called from an untimed benchmark
    setup hook, so the restore never lands in the measurement.
    """

    uri: str
    dataset: lance.LanceDataset
    base_version: int

    def reset(self, cold: bool = False) -> lance.LanceDataset:
        if cold:
            # A new session means an empty index cache. `checkout_version`
            # deliberately shares the cache, so a cold run cannot reuse
            # `self.dataset`.
            handle = lance.dataset(self.uri)
        else:
            handle = self.dataset
        reverted = handle.checkout_version(self.base_version)
        reverted.restore()
        return reverted


def _open_target(name: str) -> Iterable[Target]:
    uri = get_dataset_uri(name)
    dataset = lance.dataset(uri)
    tags = dataset.tags.list()
    if BASE_TAG not in tags:
        pytest.skip(
            f"Dataset {name} has no {BASE_TAG} tag; "
            "run python/ci_benchmarks/datagen/gen_all.py"
        )
    base_version = tags[BASE_TAG]["version"]

    yield Target(uri=uri, dataset=dataset, base_version=base_version)

    # Leave the dataset pristine and drop the versions and data files the
    # benchmarks produced. Cleanup never deletes a tagged version, so the tag
    # has to move onto the restored version first or the old base would be
    # retained forever.
    final = dataset.checkout_version(base_version)
    final.restore()
    final.tags.update(BASE_TAG, final.version)
    final.cleanup_old_versions(older_than=timedelta(0), delete_unverified=True)


@pytest.fixture(scope="module")
def narrow() -> Iterable[Target]:
    yield from _open_target("merge_insert_narrow")


@pytest.fixture(scope="module")
def wide() -> Iterable[Target]:
    yield from _open_target("merge_insert_wide")


@pytest.fixture(scope="module")
def frags() -> Iterable[Target]:
    yield from _open_target("merge_insert_frags")


@pytest.fixture(scope="module")
def deleted() -> Iterable[Target]:
    yield from _open_target("merge_insert_deleted")


@pytest.fixture(scope="module")
def unindexed_tail() -> Iterable[Target]:
    yield from _open_target("merge_insert_unindexed_tail")


# ---------------------------------------------------------------------------
# Source construction
# ---------------------------------------------------------------------------
#
# Sources are built from a contiguous run of row indices.  Which key column the
# merge joins on then decides whether those keys are clustered or scattered in
# the index's key order: `id_int` and `id_uuid7` are monotonic in the row index,
# `id_uuid4` scrambles it.  This keeps one source builder for all key shapes.


def narrow_source(row_indices: np.ndarray) -> pa.Table:
    """A full-schema narrow source. ``value`` is offset so updates are real."""
    return pa.Table.from_batches([narrow_batch(row_indices, value_offset=1)])


def existing_rows(num_rows: int, offset: int = 0) -> np.ndarray:
    return np.arange(offset, offset + num_rows, dtype=np.int64)


def new_rows(num_rows: int) -> np.ndarray:
    """Row indices past the end of the target, so every key is a new key."""
    return np.arange(NARROW_NUM_ROWS, NARROW_NUM_ROWS + num_rows, dtype=np.int64)


def wide_row_indices(fraction: float) -> np.ndarray:
    """Row indices covering ``fraction`` of every fragment's rows.

    Spread evenly within each fragment rather than contiguously: that is the
    harder case for the in-place column updater, which has to interleave
    updated and untouched values.
    """
    per_fragment = max(1, round(fraction * WIDE_ROWS_PER_FRAGMENT))
    num_fragments = WIDE_NUM_ROWS // WIDE_ROWS_PER_FRAGMENT
    return np.concatenate(
        [
            fragment * WIDE_ROWS_PER_FRAGMENT
            + np.linspace(0, WIDE_ROWS_PER_FRAGMENT - 1, per_fragment, dtype=np.int64)
            for fragment in range(num_fragments)
        ]
    )


def wide_source(row_indices: np.ndarray, columns: Sequence[str]) -> pa.Table:
    """A wide source carrying ``id_int`` plus ``columns``."""
    arrays = [pa.array(row_indices)]
    names = ["id_int"]
    for column in columns:
        names.append(column)
        if column == "vec":
            values = np.linspace(
                1.0,
                2.0,
                num=len(row_indices) * WIDE_VECTOR_DIM,
                dtype=np.float32,
            )
            arrays.append(
                pa.FixedSizeListArray.from_arrays(pa.array(values), WIDE_VECTOR_DIM)
            )
        elif WIDE_SCHEMA.field(column).type == pa.string():
            arrays.append(
                pa.array([f"updated_{column}_{v}" for v in row_indices], pa.string())
            )
        else:
            arrays.append(pa.array(row_indices + 1))
    return pa.table(arrays, schema=pa.schema([WIDE_SCHEMA.field(n) for n in names]))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    benchmark,
    target: Target,
    job: Callable[[lance.LanceDataset], ExecuteResult],
    *,
    rounds: int = 3,
    cold: bool = False,
    warmup: bool = True,
    expected_rows: Optional[int] = None,
) -> None:
    """Benchmark ``job``, restoring the target before every round.

    ``cold`` opens a fresh session for each round so the index cache starts
    empty.  ``warmup`` runs one unmeasured round first; turn it off for
    expensive shapes where a second pass is not worth the wall clock.

    ``expected_rows`` guards against a benchmark that silently stops doing
    work: a semantics regression should fail the run, not post a fast time.
    Row count is checked instead of the returned stats because it is
    independent of which execution path ran.
    """
    state: dict = {}

    def setup() -> None:
        state["dataset"] = target.reset(cold=cold)

    def bench() -> None:
        dataset = state["dataset"]
        state["stats"] = job(dataset)
        if expected_rows is not None:
            state["row_count"] = dataset.count_rows()

    benchmark.pedantic(
        bench,
        setup=setup,
        rounds=rounds,
        iterations=1,
        # `setup` runs before each warmup round too, so a warmup never leaves
        # the target dirty for the measured rounds.
        warmup_rounds=1 if warmup and not cold else 0,
    )

    if expected_rows is not None:
        assert state["row_count"] == expected_rows, (
            f"expected {expected_rows} rows after merge_insert, "
            f"got {state['row_count']} (stats: {state['stats']})"
        )


def upsert(
    key: str, source: pa.Table, *, use_index: bool
) -> Callable[[lance.LanceDataset], ExecuteResult]:
    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        return (
            dataset.merge_insert(key)
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .use_index(use_index)
            .execute(source)
        )

    return job


def uses_index(plan: str) -> bool:
    return plan == "v1_indexed"


def write_mode_for(plan: str) -> str:
    return "rewrite_columns" if plan == "v2_rewrite_columns" else "auto"


# ---------------------------------------------------------------------------
# A. Cost model core -- merge_insert_narrow, 10M rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plan", PLANS)
def test_upsert_point(benchmark, narrow: Target, plan: str) -> None:
    """Single-row upsert latency -- the extreme where a probe should win."""
    source = narrow_source(existing_rows(1))
    run(
        benchmark,
        narrow,
        upsert("id_int", source, use_index=uses_index(plan)),
        rounds=5,
        expected_rows=NARROW_NUM_ROWS,
    )


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("key", NARROW_KEYS)
@pytest.mark.parametrize("num_rows", SOURCE_SIZES)
def test_upsert_ratio_sweep(
    benchmark, narrow: Target, num_rows: int, key: str, plan: str
) -> None:
    """Source/target ratio sweep against a warm index."""
    source = narrow_source(existing_rows(num_rows))
    run(
        benchmark,
        narrow,
        upsert(key, source, use_index=uses_index(plan)),
        expected_rows=NARROW_NUM_ROWS,
    )


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("key", ["id_int", "id_uuid4"])
@pytest.mark.parametrize("num_rows", [1_000, 100_000])
def test_upsert_cold_ratio_sweep(
    benchmark, narrow: Target, num_rows: int, key: str, plan: str
) -> None:
    """Same sweep with a cold index cache, where page reads are not free."""
    source = narrow_source(existing_rows(num_rows))
    run(
        benchmark,
        narrow,
        upsert(key, source, use_index=uses_index(plan)),
        cold=True,
        expected_rows=NARROW_NUM_ROWS,
    )


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("num_rows", [10_000, 100_000])
def test_upsert_all_new_keys(
    benchmark, narrow: Target, num_rows: int, plan: str
) -> None:
    """Time-ordered ingest: every key is new, so no index page should be read."""
    source = narrow_source(new_rows(num_rows))
    run(
        benchmark,
        narrow,
        upsert("id_uuid7", source, use_index=uses_index(plan)),
        expected_rows=NARROW_NUM_ROWS + num_rows,
    )


@pytest.mark.parametrize("num_rows", [1_000, 100_000])
def test_upsert_unindexed_baseline(benchmark, narrow: Target, num_rows: int) -> None:
    """Joining on a column with no index at all -- the no-index reference."""
    source = narrow_source(existing_rows(num_rows))
    run(
        benchmark,
        narrow,
        upsert("id_no_index", source, use_index=True),
        expected_rows=NARROW_NUM_ROWS,
    )


# ---------------------------------------------------------------------------
# B. Write path -- merge_insert_wide, 1M rows
# ---------------------------------------------------------------------------

# Fractions are of each fragment's rows, which is what decides whether the
# in-place updater interleaves or rewrites a whole column file.
ROW_FRACTIONS = [0.001, 0.01, 0.1, 1.0]
FRACTION_IDS = ["0.1pct", "1pct", "10pct", "100pct"]

# Isolates the two pressures a partial-column update can exert: field count
# (scalar columns, cheap per field) and byte volume (the vector column).
PROJECTIONS = {
    "one_scalar": WIDE_SCALAR_COLUMNS[:1],
    "ten_scalars": WIDE_SCALAR_COLUMNS[:10],
    "vector": ["vec"],
    "vector_and_ten_scalars": ["vec"] + WIDE_SCALAR_COLUMNS[:10],
}


def _wide_rounds(fraction: float) -> int:
    # A full-fragment update rewrites every column file it touches, which for
    # the vector column is ~1 GB per round.
    return 1 if fraction == 1.0 else 3


def update_subset(
    source: pa.Table, *, use_index: bool, write_mode: str = "auto"
) -> Callable[[lance.LanceDataset], ExecuteResult]:
    """Partial-schema update. No insert clause: matched rows only."""

    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        return (
            dataset.merge_insert("id_int")
            .when_matched_update_all()
            .use_index(use_index)
            .write_mode(write_mode)
            .execute(source)
        )

    return job


@pytest.mark.parametrize("plan", WRITE_PLANS)
@pytest.mark.parametrize("fraction", ROW_FRACTIONS, ids=FRACTION_IDS)
def test_update_subset_row_fraction(
    benchmark, wide: Target, fraction: float, plan: str
) -> None:
    """Row-fraction sweep at a fixed, minimal projection.

    Locates the crossover between patching a column in place and rewriting the
    whole column file, without byte volume confounding it.
    """
    source = wide_source(wide_row_indices(fraction), PROJECTIONS["one_scalar"])
    run(
        benchmark,
        wide,
        update_subset(
            source,
            use_index=uses_index(plan),
            write_mode=write_mode_for(plan),
        ),
        rounds=_wide_rounds(fraction),
        warmup=fraction != 1.0,
        expected_rows=WIDE_NUM_ROWS,
    )


@pytest.mark.parametrize("plan", WRITE_PLANS)
@pytest.mark.parametrize("fraction", [0.01, 1.0], ids=["1pct", "100pct"])
@pytest.mark.parametrize("projection", list(PROJECTIONS), ids=list(PROJECTIONS))
def test_update_subset_projection(
    benchmark, wide: Target, projection: str, fraction: float, plan: str
) -> None:
    """Projection sweep: field count vs byte volume, plus the cross term."""
    source = wide_source(wide_row_indices(fraction), PROJECTIONS[projection])
    run(
        benchmark,
        wide,
        update_subset(
            source,
            use_index=uses_index(plan),
            write_mode=write_mode_for(plan),
        ),
        rounds=_wide_rounds(fraction),
        warmup=fraction != 1.0,
        expected_rows=WIDE_NUM_ROWS,
    )


@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("fraction", [0.01, 1.0], ids=["1pct", "100pct"])
def test_upsert_wide_full_schema(
    benchmark, wide: Target, fraction: float, plan: str
) -> None:
    """Full-schema baseline for the partial-column benchmarks above."""
    row_indices = wide_row_indices(fraction)
    source = wide_source(
        row_indices, [f.name for f in WIDE_SCHEMA if f.name != "id_int"]
    )
    run(
        benchmark,
        wide,
        upsert("id_int", source, use_index=uses_index(plan)),
        rounds=_wide_rounds(fraction),
        warmup=fraction != 1.0,
        expected_rows=WIDE_NUM_ROWS,
    )


# ---------------------------------------------------------------------------
# C. Clause shapes -- merge_insert_narrow, 10K-row source
# ---------------------------------------------------------------------------

CLAUSE_SOURCE_ROWS = 10_000


@pytest.mark.parametrize("plan", PLANS)
def test_insert_if_not_exists(benchmark, narrow: Target, plan: str) -> None:
    """Dedup ingest: half the keys already exist, matched rows are untouched.

    The probe only needs to know whether a key exists, so no target payload has
    to be read.
    """
    half = CLAUSE_SOURCE_ROWS // 2
    row_indices = np.concatenate([existing_rows(half), new_rows(half)])
    source = narrow_source(row_indices)

    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        return (
            dataset.merge_insert("id_int")
            .when_not_matched_insert_all()
            .use_index(uses_index(plan))
            .execute(source)
        )

    run(benchmark, narrow, job, expected_rows=NARROW_NUM_ROWS + half)


@pytest.mark.parametrize("plan", PLANS)
def test_update_only(benchmark, narrow: Target, plan: str) -> None:
    """No insert clause: unmatched source rows are dropped."""
    half = CLAUSE_SOURCE_ROWS // 2
    row_indices = np.concatenate([existing_rows(half), new_rows(half)])
    source = narrow_source(row_indices)

    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        return (
            dataset.merge_insert("id_int")
            .when_matched_update_all()
            .use_index(uses_index(plan))
            .execute(source)
        )

    run(benchmark, narrow, job, expected_rows=NARROW_NUM_ROWS)


def test_delete_by_source(benchmark, narrow: Target) -> None:
    """Deleting rows absent from the source requires a full target scan.

    The index gate in `can_use_create_plan` rejects this shape outright, so
    there is no v1 variant to compare against.
    """
    source = narrow_source(existing_rows(CLAUSE_SOURCE_ROWS))

    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        return (
            dataset.merge_insert("id_int")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .when_not_matched_by_source_delete()
            .execute(source)
        )

    run(benchmark, narrow, job, expected_rows=CLAUSE_SOURCE_ROWS)


@pytest.mark.parametrize("plan", PLANS)
def test_conditional_update(benchmark, narrow: Target, plan: str) -> None:
    """A condition on target columns forces the target payload to be read."""
    source = narrow_source(existing_rows(CLAUSE_SOURCE_ROWS))

    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        return (
            dataset.merge_insert("id_int")
            .when_matched_update_all(condition="source.value > target.value")
            .when_not_matched_insert_all()
            .use_index(uses_index(plan))
            .execute(source)
        )

    run(benchmark, narrow, job, expected_rows=NARROW_NUM_ROWS)


@pytest.mark.parametrize("plan", PLANS)
def test_composite_key_fully_indexed(benchmark, narrow: Target, plan: str) -> None:
    """Both key columns indexed: v1 probes each index and AND-folds."""
    source = narrow_source(existing_rows(CLAUSE_SOURCE_ROWS))

    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        return (
            dataset.merge_insert(["composite_a", "composite_b"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .use_index(uses_index(plan))
            .execute(source)
        )

    run(benchmark, narrow, job, expected_rows=NARROW_NUM_ROWS)


def test_composite_key_partially_indexed(benchmark, narrow: Target) -> None:
    """One key column unindexed, which forces the hash join regardless of flag.

    A partial index probe would under-match, so `can_use_create_plan` keeps
    this shape off the indexed path.
    """
    source = narrow_source(existing_rows(CLAUSE_SOURCE_ROWS))

    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        # `composite_a` is indexed, `id_no_index` is not. Both are equal to the
        # row index in source and target, so every source row matches.
        return (
            dataset.merge_insert(["composite_a", "id_no_index"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(source)
        )

    run(benchmark, narrow, job, expected_rows=NARROW_NUM_ROWS)


# ---------------------------------------------------------------------------
# D. Target shape
# ---------------------------------------------------------------------------

TARGET_SHAPE_ROWS = 10_000


@pytest.mark.parametrize("plan", PLANS)
def test_upsert_unindexed_tail(benchmark, unindexed_tail: Target, plan: str) -> None:
    """10% of the target was appended after indexing, so a scan is unioned in."""
    source = narrow_source(existing_rows(TARGET_SHAPE_ROWS))
    total_rows = NARROW_NUM_ROWS + UNINDEXED_TAIL_ROWS
    run(
        benchmark,
        unindexed_tail,
        upsert("id_int", source, use_index=uses_index(plan)),
        expected_rows=total_rows,
    )


@pytest.mark.parametrize("plan", PLANS)
def test_upsert_with_deletion_files(benchmark, deleted: Target, plan: str) -> None:
    """Every fragment carries a deletion file, which the probe has to mask."""
    source = narrow_source(existing_rows(TARGET_SHAPE_ROWS))
    # Row 0 and every DELETED_ROW_STRIDE-th row after it were deleted at
    # generation time, so those source rows insert rather than update.
    reinserted = TARGET_SHAPE_ROWS // DELETED_ROW_STRIDE
    total_rows = NARROW_NUM_ROWS - NARROW_NUM_ROWS // DELETED_ROW_STRIDE + reinserted
    run(
        benchmark,
        deleted,
        upsert("id_int", source, use_index=uses_index(plan)),
        expected_rows=total_rows,
    )


@pytest.mark.parametrize("plan", PLANS)
def test_upsert_many_small_fragments(benchmark, frags: Target, plan: str) -> None:
    """10K fragments of 1K rows, to expose per-fragment overhead."""
    row_indices = existing_rows(TARGET_SHAPE_ROWS)
    source = pa.table(
        [pa.array(row_indices), pa.array(row_indices + 1)], schema=FRAGS_SCHEMA
    )
    run(
        benchmark,
        frags,
        upsert("id_int", source, use_index=uses_index(plan)),
        expected_rows=FRAGS_NUM_ROWS,
    )


# ---------------------------------------------------------------------------
# E. Memory and regime edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plan", PLANS)
def test_upsert_streaming_source(benchmark, narrow: Target, plan: str) -> None:
    """A one-shot reader source, which v1 has to buffer in full to fork it.

    Peak memory is the number of interest here.
    """
    num_rows = 1_000_000
    batch_size = 100_000

    def make_reader() -> pa.RecordBatchReader:
        def batches():
            for start in range(0, num_rows, batch_size):
                yield narrow_batch(
                    existing_rows(batch_size, offset=start), value_offset=1
                )

        return pa.RecordBatchReader.from_batches(NARROW_SCHEMA, batches())

    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        return (
            dataset.merge_insert("id_int")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .use_index(uses_index(plan))
            .execute(make_reader())
        )

    run(
        benchmark,
        narrow,
        job,
        rounds=1,
        warmup=False,
        expected_rows=NARROW_NUM_ROWS,
    )


def test_upsert_source_equals_target(benchmark, narrow: Target) -> None:
    """Source the same size as the target -- the probe must not be chosen here.

    v2 only: the v1 indexed path cannot run this shape at all. Its source-side
    hash join asks for more than the whole memory pool and the operation fails
    with "Resources exhausted". So there is no v1 baseline to compare against,
    and this benchmark exists to keep the v2 path honest at this size.
    """
    source = narrow_source(existing_rows(NARROW_NUM_ROWS))
    run(
        benchmark,
        narrow,
        upsert("id_int", source, use_index=False),
        rounds=1,
        warmup=False,
        expected_rows=NARROW_NUM_ROWS,
    )


# ---------------------------------------------------------------------------
# IO / memory variants
# ---------------------------------------------------------------------------
#
# A subset of the above, re-run under the io_memory_benchmark fixture. Write
# amplification (`write_bytes`) is the headline for the partial-column cases and
# peak memory is the headline for the streaming source.


@pytest.mark.io_memory_benchmark()
@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("key", NARROW_KEYS)
@pytest.mark.parametrize("num_rows", [1_000, 100_000])
def test_io_mem_upsert_ratio(
    io_mem_benchmark, narrow: Target, num_rows: int, key: str, plan: str
) -> None:
    source = narrow_source(existing_rows(num_rows))
    job = upsert(key, source, use_index=uses_index(plan))
    io_mem_benchmark(job, narrow.dataset, setup=lambda: narrow.reset())


@pytest.mark.io_memory_benchmark()
@pytest.mark.parametrize("plan", WRITE_PLANS)
@pytest.mark.parametrize("fraction", ROW_FRACTIONS, ids=FRACTION_IDS)
def test_io_mem_update_subset_row_fraction(
    io_mem_benchmark, wide: Target, fraction: float, plan: str
) -> None:
    source = wide_source(wide_row_indices(fraction), PROJECTIONS["one_scalar"])
    job = update_subset(
        source, use_index=uses_index(plan), write_mode=write_mode_for(plan)
    )
    io_mem_benchmark(
        job,
        wide.dataset,
        warmup=fraction != 1.0,
        setup=lambda: wide.reset(),
    )


@pytest.mark.io_memory_benchmark()
@pytest.mark.parametrize("plan", WRITE_PLANS)
@pytest.mark.parametrize("projection", list(PROJECTIONS), ids=list(PROJECTIONS))
def test_io_mem_update_subset_projection(
    io_mem_benchmark, wide: Target, projection: str, plan: str
) -> None:
    source = wide_source(wide_row_indices(0.01), PROJECTIONS[projection])
    job = update_subset(
        source, use_index=uses_index(plan), write_mode=write_mode_for(plan)
    )
    io_mem_benchmark(job, wide.dataset, setup=lambda: wide.reset())


@pytest.mark.io_memory_benchmark()
@pytest.mark.parametrize("plan", PLANS)
def test_io_mem_upsert_streaming_source(
    io_mem_benchmark, narrow: Target, plan: str
) -> None:
    num_rows = 1_000_000
    batch_size = 100_000

    def job(dataset: lance.LanceDataset) -> ExecuteResult:
        def batches():
            for start in range(0, num_rows, batch_size):
                yield narrow_batch(
                    existing_rows(batch_size, offset=start), value_offset=1
                )

        reader = pa.RecordBatchReader.from_batches(NARROW_SCHEMA, batches())
        return (
            dataset.merge_insert("id_int")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .use_index(uses_index(plan))
            .execute(reader)
        )

    io_mem_benchmark(job, narrow.dataset, warmup=False, setup=lambda: narrow.reset())
