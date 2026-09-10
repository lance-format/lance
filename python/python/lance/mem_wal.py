# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""
Python wrappers for Lance MemWAL functionality.

The MemWAL feature enables high-throughput, low-latency writes to a Lance
dataset via an LSM-tree structure.  Data flows through three levels:

1. **WAL** – append-only durable log (raw writes)
2. **Active MemTable** – in-memory write buffer
3. **SSTable** – Lance files written to object store
4. **Base table** – canonical Lance dataset files (after SSTable compaction)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Union

import pyarrow as pa

from .lance import (
    _CompactedSsTable,
    _evaluate_sharding_spec,
    _ExecutionPlan,
    _LsmPointLookupPlanner,
    _LsmScanner,
    _LsmVectorSearchPlanner,
    _ShardSnapshot,
    _ShardWriter,
)
from .types import _coerce_reader

if TYPE_CHECKING:
    import lance

__all__ = [
    "ShardingField",
    "ShardingSpec",
    "evaluate_sharding_spec",
    "CompactedSsTable",
    "ShardSnapshot",
    "ShardWriter",
    "LsmScanner",
    "ExecutionPlan",
    "LsmPointLookupPlanner",
    "LsmVectorSearchPlanner",
]


# ---------------------------------------------------------------------------
# ShardingSpec
# ---------------------------------------------------------------------------


@dataclass
class ShardingField:
    """Defines one MemWAL sharding field.

    Parameters
    ----------
    field_id : str
        Identifier for the derived field.
    source_ids : list of int
        Source field IDs used as inputs.
    transform : str, optional
        Optional transform name applied to the source fields.
    expression : str, optional
        Optional expression used to derive the field value.
    result_type : str
        Output type name for the derived field.
    parameters : dict of str to str, optional
        Extra transform parameters.
    """

    field_id: str
    source_ids: List[int]
    transform: Optional[str] = None
    expression: Optional[str] = None
    result_type: str = ""
    parameters: Dict[str, str] = field(default_factory=dict)


@dataclass
class ShardingSpec:
    """Specification for deriving MemWAL shard routing values."""

    spec_id: int
    fields: List[ShardingField]


def evaluate_sharding_spec(
    batch: pa.RecordBatch,
    spec: Union[ShardingSpec, Mapping[str, object]],
    schema: "lance.schema.LanceSchema",
) -> pa.RecordBatch:
    """Evaluate a MemWAL sharding spec against one PyArrow RecordBatch.

    Parameters
    ----------
    batch : pyarrow.RecordBatch
        Input batch containing the sharding source columns.
    spec : ShardingSpec or dict
        MemWAL sharding spec to evaluate.
    schema : LanceSchema
        Lance table schema used to resolve source field IDs in the spec to
        input batch column names.
    """
    if not isinstance(batch, pa.RecordBatch):
        raise TypeError(f"Expected pyarrow.RecordBatch, got {type(batch)!r}")
    return _evaluate_sharding_spec(
        batch,
        _sharding_spec_to_dict(spec),
        schema,
    )


def _sharding_spec_to_dict(spec: Union[ShardingSpec, Mapping[str, object]]) -> dict:
    if isinstance(spec, Mapping):
        return dict(spec)
    return asdict(spec)


@dataclass
class CompactedSsTable:
    """Points to an SSTable compacted into the base table.

    Pass a list of these to mark_sstables_as_compacted
    so Lance can record compaction progress.

    Parameters
    ----------
    shard_id : str
        UUID string for the write shard.
    generation : int
        Generation number of the compacted SSTable (as passed to
        :meth:`ShardSnapshot.with_sstable`).
    """

    shard_id: str
    generation: int


class ShardSnapshot:
    """Snapshot of a MemWAL shard's state, used when constructing scanners.

    Parameters
    ----------
    shard_id : str
        UUID string for the write shard.
    """

    def __init__(self, shard_id: str) -> None:
        self._raw = _ShardSnapshot(shard_id)

    @property
    def shard_id(self) -> str:
        """UUID string for this shard."""
        return self._raw.shard_id

    def with_spec_id(self, spec_id: int) -> "ShardSnapshot":
        """Set the sharding spec ID."""
        self._raw = self._raw.with_spec_id(spec_id)
        return self

    def with_current_generation(self, generation: int) -> "ShardSnapshot":
        """Set the current (active) generation number."""
        self._raw = self._raw.with_current_generation(generation)
        return self

    def with_sstable(self, generation: int, path: str) -> "ShardSnapshot":
        """Add an SSTable with its storage path."""
        self._raw = self._raw.with_sstable(generation, path)
        return self

    def __repr__(self) -> str:
        return repr(self._raw)


class ShardWriter:
    """Stateful writer for one MemWAL shard.

    Obtain an instance via mem_wal_writer.
    Use as a context manager so the writer is closed automatically::

        with dataset.mem_wal_writer(shard_id) as writer:
            writer.put(batch)
            writer.delete(pa.table({"id": [1]}))

    Parameters
    ----------
    _raw : _ShardWriter
        Internal PyO3 object — do not construct directly.
    """

    def __init__(self, _raw: _ShardWriter) -> None:
        self._raw = _raw

    @property
    def shard_id(self) -> str:
        """UUID string for this writer's shard."""
        return self._raw.shard_id

    def put(self, data, *, schema: Optional[pa.Schema] = None) -> None:
        """Write data to the MemWAL.

        Parameters
        ----------
        data : ReaderLike
            Any Arrow-compatible data `pyarrow.Table`,
            `pyarrow.RecordBatch`, ``RecordBatchReader``, pandas
            DataFrame, etc.
        schema : pa.Schema, optional
            Schema hint, needed when *data* is a generator.

        Raises
        ------
        IOError
            On WAL flush failure.
        RuntimeError
            If the writer has already been closed.
        """
        reader = _coerce_reader(data, schema)
        self._raw.put(reader)

    def delete(self, keys, *, schema: Optional[pa.Schema] = None) -> None:
        """Delete rows by primary key from the MemWAL.

        Parameters
        ----------
        keys : ReaderLike
            Any Arrow-compatible data containing this shard's primary key
            column(s). Non-primary-key columns, if present, are ignored by
            the Rust core delete path.
        schema : pa.Schema, optional
            Schema hint, needed when *keys* is a generator.

        Raises
        ------
        IOError
            If delete validation fails, WAL flush fails, or the writer has
            already been closed. Delete validation is centralized in Rust and
            includes checks for primary-key metadata and tombstone-compatible
            nullable non-key columns.

        Examples
        --------
        ::

            with dataset.mem_wal_writer(shard_id) as writer:
                writer.delete(pa.table({"id": [42]}))
        """
        reader = _coerce_reader(keys, schema)
        self._raw.delete(reader)

    def close(self) -> None:
        """Flush and close the writer.

        After ``close()``, calling :meth:`put` raises an error.
        Automatically called when used as a context manager.
        """
        self._raw.close()

    def stats(self) -> dict:
        """Return a snapshot of write statistics.

        Returns
        -------
        dict
            Keys: ``put_count``, ``put_time_ms``, ``wal_flush_count``,
            ``wal_flush_bytes``, ``wal_flush_time_ms``,
            ``memtable_flush_count``, ``memtable_flush_rows``,
            ``memtable_flush_time_ms``.
        """
        return self._raw.stats()

    def memtable_stats(self) -> dict:
        """Return current MemTable statistics.

        Returns
        -------
        dict
            Keys: ``row_count``, ``batch_count``, ``estimated_size_bytes``,
            ``generation``, ``frozen_count``, ``frozen_bytes``.
        """
        return self._raw.memtable_stats()

    def lsm_scanner(
        self, shard_snapshots: Optional[List[ShardSnapshot]] = None
    ) -> "LsmScanner":
        """Create an LSM scanner that includes the active MemTable.

        This scanner covers the base table, the given SSTables,
        and the current active MemTable — providing strong read-your-writes
        consistency.

        Parameters
        ----------
        shard_snapshots : list of ShardSnapshot, optional
            Snapshots of other shards to include.  This writer's own shard
            is automatically included.

        Returns
        -------
        LsmScanner
        """
        raw_snaps = [s._raw for s in (shard_snapshots or [])]
        return LsmScanner(self._raw.lsm_scanner(raw_snaps))

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False


class LsmScanner:
    """LSM-aware scanner covering all data levels.

    Deduplicates by primary key, always returning the newest version of
    each row across base table, SSTables, and the active MemTable.

    Obtain an instance from `ShardWriter.lsm_scanner` (includes
    active MemTable) or `LsmScanner.from_snapshots` (SSTables only).

    The builder methods (`project`, `filter`, `limit`)
    return ``self`` for chaining.

    Examples
    --------
    scanner = LsmScanner.from_snapshots(dataset, [snapshot])
    scanner.project(["id", "value"]).filter("value > 0.5")
    table = scanner.to_table()
    """

    def __init__(self, _raw: _LsmScanner) -> None:
        self._raw = _raw

    @staticmethod
    def from_snapshots(
        dataset: "lance.LanceDataset",
        shard_snapshots: List[ShardSnapshot],
    ) -> "LsmScanner":
        """Create a scanner from dataset and shard snapshots.

        Does **not** include the active MemTable; use
        `ShardWriter.lsm_scanner` for that.

        Parameters
        ----------
        dataset : LanceDataset
            The base dataset to scan.
        shard_snapshots : list of ShardSnapshot
            Shard snapshots specifying SSTables to include.
        """
        raw = _LsmScanner.from_snapshots(dataset._ds, [s._raw for s in shard_snapshots])
        return LsmScanner(raw)

    def project(self, columns: List[str]) -> "LsmScanner":
        """Select specific columns to return."""
        self._raw = self._raw.project(columns)
        return self

    def filter(self, expr: str) -> "LsmScanner":
        """Set a SQL filter expression (e.g. ``"value > 0.5"``)."""
        self._raw = self._raw.filter(expr)
        return self

    def limit(
        self, n: Optional[int] = None, offset: Optional[int] = None
    ) -> "LsmScanner":
        """Limit rows returned, optionally with an offset."""
        self._raw = self._raw.limit(n, offset)
        return self

    def with_row_address(self) -> "LsmScanner":
        """Include the ``_rowaddr`` internal column in results."""
        self._raw = self._raw.with_row_address()
        return self

    def with_memtable_gen(self) -> "LsmScanner":
        """Include the ``_memtable_gen`` internal column in results."""
        self._raw = self._raw.with_memtable_gen()
        return self

    def to_batch(self) -> pa.RecordBatch:
        """Execute the scan and return a single merged :class:`~pyarrow.RecordBatch`."""
        return self._raw.to_batch()

    def to_batches(self) -> List[pa.RecordBatch]:
        """Execute the scan and return a list of :class:`~pyarrow.RecordBatch`."""
        return list(self._raw.to_batches())

    def to_table(self) -> pa.Table:
        """Execute the scan and return a :class:`~pyarrow.Table`."""
        batch = self.to_batch()
        return pa.Table.from_batches([batch])

    def count_rows(self) -> int:
        """Return the row count without loading all column data."""
        return self._raw.count_rows()


class ExecutionPlan:
    """Executable physical plan returned by MemWAL planners.

    This wraps the Rust/DataFusion physical plan object. Planner classes only
    construct plans; execution happens through this class.

    Parameters
    ----------
    _raw : _ExecutionPlan
    """

    def __init__(self, _raw: _ExecutionPlan) -> None:
        self._raw = _raw

    @property
    def schema(self) -> pa.Schema:
        """Output schema of this physical plan."""
        return self._raw.schema

    @property
    def dataset_schema(self) -> pa.Schema:
        """Base dataset schema used to construct this plan."""
        return self._raw.dataset_schema

    def explain(self) -> str:
        """Return the physical plan as an indented string."""
        return self._raw.explain()

    def to_reader(self) -> pa.RecordBatchReader:
        """Execute the plan and return a streaming reader."""
        return self._raw.to_reader()

    def to_batches(self) -> List[pa.RecordBatch]:
        """Execute the plan and return all record batches."""
        return list(self._raw.to_batches())

    def to_table(self) -> pa.Table:
        """Execute the plan and return a table."""
        return self.to_reader().read_all()


class LsmPointLookupPlanner:
    """Plans primary-key point lookups across all LSM levels.

    More efficient than `LsmScanner` for known-PK lookups due to
    bloom filter optimizations and short-circuit evaluation.

    Parameters
    ----------
    dataset : LanceDataset
        The base dataset.
    shard_snapshots : list of ShardSnapshot
        Shard snapshots specifying SSTables to include.
    pk_columns : list of str, optional
        Primary key column names.  Inferred from schema metadata if omitted.

    Examples
    --------
    planner = LsmPointLookupPlanner(dataset, [snapshot])
    plan = planner.plan_lookup(pa.array([42], type=pa.int64()))
    result = plan.to_table()
    """

    def __init__(
        self,
        dataset: "lance.LanceDataset",
        shard_snapshots: List[ShardSnapshot],
        pk_columns: Optional[List[str]] = None,
    ) -> None:
        self._raw = _LsmPointLookupPlanner(
            dataset._ds,
            [s._raw for s in shard_snapshots],
            pk_columns,
        )

    def plan_lookup(
        self,
        pk_value: pa.Array,
        columns: Optional[List[str]] = None,
    ) -> ExecutionPlan:
        """Plan a point lookup by primary key value.

        Parameters
        ----------
        pk_value : pa.Array
            For single-column primary keys, a PyArrow array with exactly one
            element. For composite primary keys, a single-row
            `pyarrow.StructArray` with one field per primary-key column.
        columns : list of str, optional
            Columns to project. Returns all columns if omitted.

        Returns
        -------
        ExecutionPlan
            Physical plan for the lookup. Execute it via `to_table`,
            `to_reader`, or `to_batches`.
        """
        return ExecutionPlan(self._raw.plan_lookup(pk_value, columns))


class LsmVectorSearchPlanner:
    """Plans IVF-PQ vector KNN search across all LSM levels.

    Results include staleness filtering to return only the latest version
    of each row. The output schema includes the ``_distance`` column.

    Parameters
    ----------
    dataset : LanceDataset
        The base dataset.
    shard_snapshots : list of ShardSnapshot
        Shard snapshots specifying SSTables to include.
    vector_column : str
        Name of the ``FixedSizeList<float32>`` vector column.
    pk_columns : list of str, optional
        Primary key columns.  Inferred from schema metadata if omitted.
    distance_type : str, optional
        Distance metric — one of ``"l2"`` (default), ``"cosine"``,
        ``"dot"``, ``"hamming"``.
    filter : str, optional
        SQL predicate applied as a prefilter to every LSM source before vector
        candidate selection.

    Examples
    --------
    import numpy as np
    planner = LsmVectorSearchPlanner(dataset, [snapshot], "vector")
    query = pa.array(np.random.rand(128).astype("float32"))
    plan = planner.plan_search(query, k=10)
    result = plan.to_table()
    """

    def __init__(
        self,
        dataset: "lance.LanceDataset",
        shard_snapshots: List[ShardSnapshot],
        vector_column: str,
        pk_columns: Optional[List[str]] = None,
        distance_type: Optional[str] = None,
        filter: Optional[str] = None,
    ) -> None:
        kwargs = {}
        if pk_columns is not None:
            kwargs["pk_columns"] = pk_columns
        if distance_type is not None:
            kwargs["distance_type"] = distance_type
        if filter is not None:
            kwargs["filter"] = filter
        self._raw = _LsmVectorSearchPlanner(
            dataset._ds,
            [s._raw for s in shard_snapshots],
            vector_column,
            **kwargs,
        )

    def plan_search(
        self,
        query: pa.Array,
        k: int = 10,
        nprobes: int = 20,
        columns: Optional[List[str]] = None,
        refine_base_table: bool = False,
        overfetch_factor: float = 1.0,
    ) -> ExecutionPlan:
        """Plan a KNN vector search.

        Parameters
        ----------
        query : pa.Array
            A flat ``Float32Array`` of length ``vector_dim``.
        k : int, optional
            Number of nearest neighbours to return (default: 10).
        nprobes : int, optional
            Number of IVF partitions to probe (default: 20).
        columns : list of str, optional
            Columns to project.  Returns all columns + ``_distance`` if
            omitted.
        refine_base_table : bool, optional
            When true, the base-table arm re-ranks its candidates with exact
            distances.  Useful when the base index is approximate (e.g.
            IVF-PQ).  Memtable arms use exact HNSW and are unaffected.
            Auto-enabled whenever stale filtering is on (see ``overfetch_factor``).
        overfetch_factor : float, optional
            Over-fetch multiple for sources with rows superseded by newer
            generations. Must be at least ``1.0``. At ``1.0`` each affected
            source fetches exactly ``k`` candidates; above ``1.0`` it fetches
            ``ceil(k * overfetch_factor)`` candidates so dropping stale rows is
            less likely to under-fill the result.

        Returns
        -------
        ExecutionPlan
            Physical plan for the vector search. Execute it via
            `to_table`, `to_reader`, or `to_batches`.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if nprobes <= 0:
            raise ValueError(f"nprobes must be positive, got {nprobes}")
        if not math.isfinite(overfetch_factor) or overfetch_factor < 1.0:
            raise ValueError(
                f"overfetch_factor must be finite and >= 1.0, got {overfetch_factor}"
            )
        return ExecutionPlan(
            self._raw.plan_search(
                query,
                k,
                nprobes,
                columns,
                refine_base_table,
                overfetch_factor,
            )
        )


def _unwrap_shard_id(shard_id: str) -> str:
    """Validate shard_id is a UUID string."""
    import uuid as _uuid

    _uuid.UUID(shard_id)  # raises ValueError if invalid
    return shard_id


def _to_raw_compacted_sstables(
    sstables: Iterable[CompactedSsTable],
) -> list:
    """Convert Python CompactedSsTable list to PyO3 _CompactedSsTable list."""
    return [_CompactedSsTable(s.shard_id, s.generation) for s in sstables]
