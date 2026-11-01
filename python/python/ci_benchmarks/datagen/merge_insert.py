# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Generate the merge_insert benchmark datasets.

merge_insert benchmarks mutate their target, so each dataset carries a
``merge_insert_base`` tag pointing at a pristine version.  Benchmarks restore
to that tag before every measured run (see ``benchmarks/test_merge_insert.py``),
which makes the suite tolerant of a crashed run leaving stale versions behind.

Datasets
--------
``merge_insert_narrow``
    10M rows, 10 fragments.  The key columns differ along two independent
    axes so benchmarks can separate them:

      * ``id_int`` vs ``id_uuid7`` -- key *type* (int64 vs string), both
        clustered.
      * ``id_uuid7`` vs ``id_uuid4`` -- key *distribution* (clustered vs
        random), both string.

    ``id_no_index`` holds the same values as ``id_int`` with no index, for the
    "user never built an index" baseline.  ``composite_a``/``composite_b`` are
    both indexed, for composite-key probes.

``merge_insert_wide``
    1M rows, 10 fragments.  20 narrow scalar columns (~200 MB) plus one
    256-dim float32 vector column (~1 GB).  These exert different pressures on
    a partial-column update: the scalar columns stress field count and the
    number of distinct buffers to schedule, the vector column stresses raw
    byte volume.

``merge_insert_frags``
    10M rows in 10K fragments of 1K rows, for per-fragment overhead.

``merge_insert_deleted``
    ``merge_insert_narrow`` layout with a deletion file on every fragment.

``merge_insert_unindexed_tail``
    ``merge_insert_narrow`` layout where 10% of the rows were appended after
    the index was built, so the probe must union an unindexed scan.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import lance
import numpy as np
import pyarrow as pa
from lance.log import LOGGER

from ci_benchmarks.datasets import get_dataset_uri

# Tag marking the pristine version that benchmarks restore to.
BASE_TAG = "merge_insert_base"

# BTREE training sorts the full column under DataFusion's FairSpillPool. The
# default 150 MiB pool is tight for 10M 32-byte string keys (uuid columns):
# ExternalSorterMerge has peaked near the pool edge and aborted datagen with
# ResourcesExhausted. Index build is one-shot setup, not a measured path, so
# temporarily enlarge the pool only around create_scalar_index.
_INDEX_BUILD_MEM_POOL_SIZE = str(1024 * 1024 * 1024)

NARROW_NUM_ROWS = 10_000_000
NARROW_ROWS_PER_FRAGMENT = 1_000_000

WIDE_NUM_ROWS = 1_000_000
WIDE_ROWS_PER_FRAGMENT = 100_000
WIDE_NUM_SCALAR_COLUMNS = 20
WIDE_VECTOR_DIM = 256

FRAGS_NUM_ROWS = 10_000_000
FRAGS_ROWS_PER_FRAGMENT = 1_000

# Rows appended to `merge_insert_unindexed_tail` after the index is built.
UNINDEXED_TAIL_ROWS = NARROW_NUM_ROWS // 10

# `merge_insert_deleted` deletes every Nth row, which puts a deletion file on
# every fragment.
DELETED_ROW_STRIDE = 1000

_BATCH_SIZE = 1_000_000

NARROW_INDEXED_COLUMNS = [
    "id_int",
    "id_uuid7",
    "id_uuid4",
    "composite_a",
    "composite_b",
]

NARROW_SCHEMA = pa.schema(
    [
        ("id_int", pa.int64()),
        ("id_uuid7", pa.string()),
        ("id_uuid4", pa.string()),
        ("id_no_index", pa.int64()),
        ("composite_a", pa.int64()),
        ("composite_b", pa.int64()),
        ("value", pa.int64()),
    ]
)

FRAGS_SCHEMA = pa.schema([("id_int", pa.int64()), ("value", pa.int64())])

WIDE_SCALAR_COLUMNS = [f"scalar_{i}" for i in range(WIDE_NUM_SCALAR_COLUMNS)]

# Half int64, half string, so a partial-column update touches a mix of
# fixed-width and variable-width buffers.
WIDE_SCHEMA = pa.schema(
    [("id_int", pa.int64())]
    + [
        (name, pa.int64() if i % 2 == 0 else pa.string())
        for i, name in enumerate(WIDE_SCALAR_COLUMNS)
    ]
    + [("vec", pa.list_(pa.float32(), WIDE_VECTOR_DIM))]
)


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)
_HEX_DIGITS = np.frombuffer(b"0123456789abcdef", dtype="S1")
_NIBBLE_SHIFTS = np.arange(60, -4, -4, dtype=np.uint64)
# Two uint64 halves rendered as hex.
_KEY_WIDTH = 32


def _scramble(values: np.ndarray) -> np.ndarray:
    """splitmix64 finalizer. Wrapping uint64 arithmetic, vectorized."""
    x = values.astype(np.uint64) * _GOLDEN
    x = (x ^ (x >> np.uint64(30))) * _MIX1
    x = (x ^ (x >> np.uint64(27))) * _MIX2
    return x ^ (x >> np.uint64(31))


def _hex_keys(high: np.ndarray, low: np.ndarray) -> pa.Array:
    """Format two uint64 columns as 32-character lowercase hex strings.

    Vectorized because the narrow dataset needs 10M of these and the
    benchmarks rebuild source keys from row indices on every run.
    """
    # One nibble at a time into a preallocated byte matrix. Broadcasting all 16
    # shifts at once would materialize an (n, 16) uint64 array per half, which is
    # 2.5 GB of scratch for a 10M-row source.
    num_rows = len(high)
    half = len(_NIBBLE_SHIFTS)
    digits = np.empty((num_rows, _KEY_WIDTH), dtype="S1")
    for position, shift in enumerate(_NIBBLE_SHIFTS):
        digits[:, position] = _HEX_DIGITS[(high >> shift) & np.uint64(0xF)]
        digits[:, position + half] = _HEX_DIGITS[(low >> shift) & np.uint64(0xF)]

    # Built from buffers rather than `pa.array`, which splits a numpy byte array
    # into chunks above ~1M elements. RecordBatch needs a contiguous Array.
    offsets = np.arange(0, _KEY_WIDTH * (num_rows + 1), _KEY_WIDTH, dtype=np.int32)
    return pa.StringArray.from_buffers(
        num_rows, pa.py_buffer(offsets), pa.py_buffer(digits)
    )


def uuid7_keys(row_indices: np.ndarray) -> pa.Array:
    """Sortable, UUIDv7-shaped keys: monotonic prefix, scrambled suffix.

    Real UUIDv7 keys are time-ordered, so a stream of them lands in a narrow,
    advancing slice of the index.  Reproducing that ordering is what matters
    for the benchmark; the exact bit layout is not.

    Deterministic in the row index so benchmarks can reconstruct a key without
    reading the dataset.
    """
    row_indices = row_indices.astype(np.uint64)
    return _hex_keys(row_indices, _scramble(row_indices))


def uuid4_keys(row_indices: np.ndarray) -> pa.Array:
    """Same width as `uuid7_keys`, but scattered across the index key space.

    Deterministic for the same reason, but the leading bytes are scrambled, so
    a contiguous run of row indices maps to keys spread over the whole index
    rather than a narrow slice.
    """
    row_indices = row_indices.astype(np.uint64)
    return _hex_keys(_scramble(row_indices), row_indices)


def narrow_batch(row_indices: np.ndarray, value_offset: int = 0) -> pa.RecordBatch:
    """Build a full-schema narrow batch from row indices.

    Shared with the benchmarks, which reconstruct source rows from row indices.
    ``value_offset`` shifts the payload column so an update writes a value that
    differs from what the target already holds.
    """
    return pa.record_batch(
        [
            pa.array(row_indices),
            uuid7_keys(row_indices),
            uuid4_keys(row_indices),
            pa.array(row_indices),
            pa.array(row_indices),
            # `composite_b` is a deterministic function of `composite_a` so a
            # source row can target an existing composite key without the
            # benchmark tracking extra state.
            pa.array(row_indices % 1024),
            pa.array(row_indices + value_offset),
        ],
        schema=NARROW_SCHEMA,
    )


def _narrow_data(num_rows: int, offset: int = 0):
    LOGGER.info("Generating %d narrow rows starting at %d", num_rows, offset)
    for start in range(offset, offset + num_rows, _BATCH_SIZE):
        count = min(_BATCH_SIZE, offset + num_rows - start)
        yield narrow_batch(np.arange(start, start + count, dtype=np.int64))


def _frags_data(num_rows: int):
    LOGGER.info("Generating %d small-fragment rows", num_rows)
    for start in range(0, num_rows, _BATCH_SIZE):
        ids = np.arange(
            start, start + min(_BATCH_SIZE, num_rows - start), dtype=np.int64
        )
        yield pa.record_batch([pa.array(ids), pa.array(ids)], schema=FRAGS_SCHEMA)


def _wide_batch(offset: int, num_rows: int) -> pa.RecordBatch:
    ids = np.arange(offset, offset + num_rows, dtype=np.int64)
    columns: list[pa.Array] = [pa.array(ids)]
    for i in range(WIDE_NUM_SCALAR_COLUMNS):
        if i % 2 == 0:
            columns.append(pa.array(ids + i))
        else:
            columns.append(pa.array([f"s{i}_{v}" for v in ids], type=pa.string()))
    # Deterministic float payload; the values are irrelevant, the byte volume
    # is the point.
    vectors = np.linspace(
        0.0, 1.0, num=num_rows * WIDE_VECTOR_DIM, dtype=np.float32
    ).reshape(num_rows, WIDE_VECTOR_DIM)
    columns.append(
        pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.reshape(-1)), WIDE_VECTOR_DIM
        )
    )
    return pa.record_batch(columns, schema=WIDE_SCHEMA)


def _wide_data(num_rows: int):
    LOGGER.info("Generating %d wide rows", num_rows)
    # The vector column makes full batches large, so use a smaller batch here.
    batch_size = WIDE_ROWS_PER_FRAGMENT
    for start in range(0, num_rows, batch_size):
        yield _wide_batch(start, min(batch_size, num_rows - start))


def _tag_base(ds: lance.LanceDataset) -> None:
    """Point BASE_TAG at the current version, creating the tag if needed."""
    if BASE_TAG in ds.tags.list():
        ds.tags.update(BASE_TAG, ds.version)
    else:
        ds.tags.create(BASE_TAG, ds.version)


def _already_generated(uri: str, expected_rows: int) -> bool:
    """True when a usable dataset with a pristine base tag already exists.

    A previous benchmark run may have left extra versions behind, so the row
    count is checked at the tagged version rather than at the latest one.

    Incomplete generations (dataset written, tag never created) and missing
    tags both return False so the caller can overwrite and retag.
    """
    try:
        ds = lance.dataset(uri)
    except ValueError:
        return False
    tags = ds.tags.list()
    if BASE_TAG not in tags:
        return False
    base_version = tags[BASE_TAG]["version"]
    return ds.checkout_version(base_version).count_rows() == expected_rows


@contextmanager
def _enlarged_mem_pool_for_index_build():
    """Raise LANCE_MEM_POOL_SIZE only for the duration of index training."""
    previous = os.environ.get("LANCE_MEM_POOL_SIZE")
    os.environ["LANCE_MEM_POOL_SIZE"] = _INDEX_BUILD_MEM_POOL_SIZE
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("LANCE_MEM_POOL_SIZE", None)
        else:
            os.environ["LANCE_MEM_POOL_SIZE"] = previous


def _gen(
    name: str,
    data,
    schema: pa.Schema,
    expected_rows: int,
    rows_per_fragment: int,
    indexed_columns: list[str],
) -> lance.LanceDataset:
    dataset_uri = get_dataset_uri(name)
    if _already_generated(dataset_uri, expected_rows):
        LOGGER.info("Dataset %s already exists, skipping", name)
        return lance.dataset(dataset_uri)

    LOGGER.info("Creating dataset %s", name)
    ds = lance.write_dataset(
        data,
        dataset_uri,
        schema=schema,
        mode="overwrite",
        max_rows_per_file=rows_per_fragment,
        max_rows_per_group=min(rows_per_fragment, 100_000),
    )
    with _enlarged_mem_pool_for_index_build():
        for column in indexed_columns:
            LOGGER.info("Building BTREE index on %s.%s", name, column)
            ds.create_scalar_index(column, "BTREE")
    _tag_base(ds)
    return ds


def gen_merge_insert_narrow() -> lance.LanceDataset:
    return _gen(
        "merge_insert_narrow",
        _narrow_data(NARROW_NUM_ROWS),
        NARROW_SCHEMA,
        NARROW_NUM_ROWS,
        NARROW_ROWS_PER_FRAGMENT,
        NARROW_INDEXED_COLUMNS,
    )


def gen_merge_insert_wide() -> lance.LanceDataset:
    return _gen(
        "merge_insert_wide",
        _wide_data(WIDE_NUM_ROWS),
        WIDE_SCHEMA,
        WIDE_NUM_ROWS,
        WIDE_ROWS_PER_FRAGMENT,
        ["id_int"],
    )


def gen_merge_insert_frags() -> lance.LanceDataset:
    return _gen(
        "merge_insert_frags",
        _frags_data(FRAGS_NUM_ROWS),
        FRAGS_SCHEMA,
        FRAGS_NUM_ROWS,
        FRAGS_ROWS_PER_FRAGMENT,
        ["id_int"],
    )


def gen_merge_insert_deleted() -> lance.LanceDataset:
    """Narrow layout with a deletion file on every fragment.

    One row in every `DELETED_ROW_STRIDE` is deleted, which lands in every
    fragment.
    """
    name = "merge_insert_deleted"
    dataset_uri = get_dataset_uri(name)
    expected_rows = NARROW_NUM_ROWS - NARROW_NUM_ROWS // DELETED_ROW_STRIDE
    if _already_generated(dataset_uri, expected_rows):
        LOGGER.info("Dataset %s already exists, skipping", name)
        return lance.dataset(dataset_uri)

    ds = _gen(
        name,
        _narrow_data(NARROW_NUM_ROWS),
        NARROW_SCHEMA,
        NARROW_NUM_ROWS,
        NARROW_ROWS_PER_FRAGMENT,
        NARROW_INDEXED_COLUMNS,
    )
    LOGGER.info("Deleting rows from %s to create deletion files", name)
    ds.delete(f"id_int % {DELETED_ROW_STRIDE} == 0")
    _tag_base(ds)
    return ds


def gen_merge_insert_unindexed_tail() -> lance.LanceDataset:
    """Narrow layout where the last 10% of rows are not covered by the index."""
    name = "merge_insert_unindexed_tail"
    dataset_uri = get_dataset_uri(name)
    expected_rows = NARROW_NUM_ROWS + UNINDEXED_TAIL_ROWS
    if _already_generated(dataset_uri, expected_rows):
        LOGGER.info("Dataset %s already exists, skipping", name)
        return lance.dataset(dataset_uri)

    _gen(
        name,
        _narrow_data(NARROW_NUM_ROWS),
        NARROW_SCHEMA,
        NARROW_NUM_ROWS,
        NARROW_ROWS_PER_FRAGMENT,
        NARROW_INDEXED_COLUMNS,
    )
    LOGGER.info("Appending %d unindexed rows to %s", UNINDEXED_TAIL_ROWS, name)
    ds = lance.write_dataset(
        _narrow_data(UNINDEXED_TAIL_ROWS, offset=NARROW_NUM_ROWS),
        dataset_uri,
        schema=NARROW_SCHEMA,
        mode="append",
        max_rows_per_file=NARROW_ROWS_PER_FRAGMENT,
    )
    _tag_base(ds)
    return ds


def gen_merge_insert() -> None:
    gen_merge_insert_narrow()
    gen_merge_insert_wide()
    gen_merge_insert_frags()
    gen_merge_insert_deleted()
    gen_merge_insert_unindexed_tail()
