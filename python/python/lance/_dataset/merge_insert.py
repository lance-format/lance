# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Python-facing merge-insert builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import pyarrow as pa

from ..lance import _MergeInsertBuilder
from ..types import _coerce_source

if TYPE_CHECKING:
    from datetime import timedelta

    from .. import mem_wal
    from ..types import ReaderLike
    from .transaction import Transaction


def _type_contains_blob_v2(data_type: pa.DataType) -> bool:
    if isinstance(data_type, pa.ExtensionType):
        return data_type.extension_name == "lance.blob.v2"
    if pa.types.is_struct(data_type):
        return any(
            _type_contains_blob_v2(data_type.field(index).type)
            for index in range(data_type.num_fields)
        )
    if (
        pa.types.is_list(data_type)
        or pa.types.is_large_list(data_type)
        or pa.types.is_fixed_size_list(data_type)
    ):
        return _type_contains_blob_v2(data_type.value_type)
    if pa.types.is_map(data_type):
        return _type_contains_blob_v2(data_type.item_type)
    return False


def _is_native_dataset_source(data_obj: ReaderLike) -> bool:
    """Whether a merge can use the Lance dataset handle without a Python scanner."""
    # Local import: ``lance.dataset`` imports this module.
    from ..dataset import LanceDataset

    return (
        type(data_obj) is LanceDataset
        and not data_obj._default_scan_options
        and not any(_type_contains_blob_v2(field.type) for field in data_obj._ds.schema)
    )


class MergeInsertBuilder(_MergeInsertBuilder):
    def execute(self, data_obj: ReaderLike, *, schema: Optional[pa.Schema] = None):
        """Executes the merge insert operation

        This function updates the original dataset and returns a dictionary with
        information about merge statistics - i.e. the number of inserted, updated,
        and deleted rows.

        Parameters
        ----------

        data_obj: ReaderLike
            The new data to use as the source table for the operation.  This parameter
            can be any source of data (e.g. table / dataset) that
            :func:`~lance.write_dataset` accepts.
        schema: Optional[pa.Schema]
            The schema of the data.  This only needs to be supplied whenever the data
            source is some kind of generator.

        Notes
        -----
        Repeatable sources provide fresh readers for conflict retries.
        Keep external source files unchanged until the operation finishes.
        Sources without a repeatability contract retain the replay buffer.
        """
        if _is_native_dataset_source(data_obj):
            return super(MergeInsertBuilder, self).execute_dataset(data_obj._ds)
        source = _coerce_source(data_obj, schema)
        if source.mode == "replayable":
            return super(MergeInsertBuilder, self).execute_replayable(
                source.schema, source.open_reader
            )
        reader = source.open_reader()
        if source.mode == "materialized":
            return super(MergeInsertBuilder, self).execute_batches(reader)
        return super(MergeInsertBuilder, self).execute(reader)

    def execute_uncommitted(
        self, data_obj: ReaderLike, *, schema: Optional[pa.Schema] = None
    ) -> Tuple[Transaction, Dict[str, Any]]:
        """Executes the merge insert operation without committing

        This function updates the original dataset and returns a dictionary with
        information about merge statistics - i.e. the number of inserted, updated,
        and deleted rows.

        Parameters
        ----------

        data_obj: ReaderLike
            The new data to use as the source table for the operation.  This parameter
            can be any source of data (e.g. table / dataset) that
            :func:`~lance.write_dataset` accepts.
        schema: Optional[pa.Schema]
            The schema of the data.  This only needs to be supplied whenever the data
            source is some kind of generator.
        """
        if _is_native_dataset_source(data_obj):
            return super(MergeInsertBuilder, self).execute_uncommitted_dataset(
                data_obj._ds
            )
        source = _coerce_source(data_obj, schema)
        reader = source.open_reader()

        if source.mode == "materialized":
            return super(MergeInsertBuilder, self).execute_uncommitted_batches(reader)

        return super(MergeInsertBuilder, self).execute_uncommitted(reader)

    # These next three overrides exist only to document the methods

    def when_matched_update_all(
        self, condition: Optional[str] = None
    ) -> "MergeInsertBuilder":
        """
        Configure the operation to update matched rows

        After this method is called, when the merge insert operation executes,
        any rows that match both the source table and the target table will be
        updated.  The rows from the target table will be removed and the rows
        from the source table will be added.

        An optional condition may be specified.  This should be an SQL filter
        and, if present, then only matched rows that also satisfy this filter will
        be updated.  The SQL filter should use the prefix `target.` to refer to
        columns in the target table and the prefix `source.` to refer to columns
        in the source table.  For example, `source.last_update < target.last_update`.

        If a condition is specified and rows do not satisfy the condition then these
        rows will not be updated.  Failure to satisfy the filter does not cause
        a "matched" row to become a "not matched" row.
        """
        return super(MergeInsertBuilder, self).when_matched_update_all(condition)

    def when_matched_delete(self) -> "MergeInsertBuilder":
        """
        Configure the operation to delete matched rows in the target table.

        After this method is called, when the merge insert operation executes,
        any rows that match both the source table and the target table will be
        deleted.
        """
        return super(MergeInsertBuilder, self).when_matched_delete()

    def when_matched_fail(self) -> "MergeInsertBuilder":
        """
        Configure the operation to fail if any rows match

        After this method is called, when the merge insert operation executes,
        if there are any rows that match between the source table and the target table,
        the entire operation will fail with an exception.
        This is useful for ensuring that no existing rows are overwritten or modified.
        """
        return super(MergeInsertBuilder, self).when_matched_fail()

    def when_not_matched_insert_all(self) -> "MergeInsertBuilder":
        """
        Configure the operation to insert not matched rows

        After this method is called, when the merge insert operation executes,
        any rows that exist only in the source table will be inserted into
        the target table.
        """
        return super(MergeInsertBuilder, self).when_not_matched_insert_all()

    def when_not_matched_by_source_delete(
        self, expr: Optional[str] = None
    ) -> "MergeInsertBuilder":
        """
        Configure the operation to delete source rows that do not match

        After this method is called, when the merge insert operation executes,
        any rows that exist only in the target table will be deleted.  An
        optional filter can be specified to limit the scope of the delete
        operation.  If given (as an SQL filter) then only rows which match
        the filter will be deleted.
        """
        return super(MergeInsertBuilder, self).when_not_matched_by_source_delete(expr)

    def conflict_retries(self, max_retries: int) -> "MergeInsertBuilder":
        """
        Set number of times to retry the operation if there is contention.

        If this is set > 0, then the operation will keep a copy of the input data
        either in memory or on disk (depending on the size of the data) and will
        retry the operation if there is contention.

        Default is 10.
        """
        return super(MergeInsertBuilder, self).conflict_retries(max_retries)

    def retry_timeout(self, timeout: timedelta) -> "MergeInsertBuilder":
        """
        Set the timeout used to limit retries.

        This is the maximum time to spend on the operation before giving up. At
        least one attempt will be made, regardless of how long it takes to complete.
        Subsequent attempts will be cancelled once this timeout is reached. If
        the timeout has been reached during the first attempt, the operation
        will be cancelled immediately before making a second attempt.

        The default is 30 seconds.
        """
        return super(MergeInsertBuilder, self).retry_timeout(timeout)

    def data_storage_version(self, version: str) -> "MergeInsertBuilder":
        """Set the exact data storage version for files written by this operation.

        If omitted, use the dataset's default write version without changing it.
        Accepts "2.0", "2.1", "2.2", "2.3", "stable", or "next" for V2
        datasets. Release selectors are resolved by the engine; V1/V2
        cross-family targets are rejected.

        Examples
        --------
        ``dataset.merge_insert("id").data_storage_version("2.2")`` selects V2.2
        for the files written when the builder executes.
        """
        return super(MergeInsertBuilder, self).data_storage_version(version)

    def use_index(self, use_index: bool) -> "MergeInsertBuilder":
        """
        Controls whether to use indices for the merge operation.

        When set to False, forces a full table scan even if an index exists on
        the join key. This can be useful for benchmarking or when the optimizer
        chooses a suboptimal path.

        Parameters
        ----------
        use_index : bool
            If True (default), uses an index if available. If False, forces a
            full table scan.

        Returns
        -------
        MergeInsertBuilder
            The builder instance for method chaining.
        """
        return super(MergeInsertBuilder, self).use_index(use_index)

    def write_mode(
        self, mode: Literal["auto", "rewrite_rows", "rewrite_columns"]
    ) -> "MergeInsertBuilder":
        """
        Selects how the merged rows are written to disk.

        For a partial-schema update (the source omits some dataset columns) the
        two modes have different cost shapes. ``rewrite_columns`` never reads or
        writes the columns the source omits, but its replacement column file
        covers every row of each fragment it touches, so the bytes written barely
        fall as fewer rows match. ``rewrite_rows`` instead scales with the number
        of matched rows. Patching columns wins once the fraction of rows matched
        exceeds roughly the fraction of each row's bytes the source columns
        occupy: for a KB-scale update of a MB-per-row table that is nearly
        always, and for a table whose columns are all narrow it may never be.

        The per-fragment matched row count that decides this is only known once
        the join has run, so the caller picks rather than the planner guessing.

        Parameters
        ----------
        mode : {'auto', 'rewrite_rows', 'rewrite_columns'}
            ``auto`` (default) lets the engine choose. It rewrites whole rows,
            except on the one path that predates this parameter: a
            partial-schema update whose join key carries a scalar index patches
            columns. ``rewrite_rows`` deletes the matched rows and writes whole
            rows into new fragments; for a partial-schema update it gives up
            that index probe, since the indexed path only ever patches columns.
            ``rewrite_columns`` attaches new data files holding the source
            columns to the fragments that already hold the matched rows; it
            raises if the merge cannot be expressed that way, which requires
            updating matched rows only (no inserts, no matched deletes, no
            delete-by-source) with a source that omits at least one dataset
            column, carries at least one column besides the join key, and
            carries no blob column.

        Returns
        -------
        MergeInsertBuilder
            The builder instance for method chaining.
        """
        return super(MergeInsertBuilder, self).write_mode(mode)

    def target_bases(self, bases: List[str]) -> "MergeInsertBuilder":
        """
        Write new fragments produced by this merge insert to these bases.

        Each entry references a base path registered in the dataset manifest,
        by name or by path URI, like the ``target_bases`` parameter of
        :func:`~lance.write_dataset`. An entry equal to the dataset's URI
        includes the dataset's primary storage in the rotation, e.g.
        ``[ds.uri, "base1", "base2"]`` spreads new data files across primary
        storage and both bases. New data files are distributed across the
        target bases round-robin. Data files that patch existing fragments
        and deletion files are always written to the dataset's primary
        storage.

        Parameters
        ----------
        bases : List[str]
            Base names or path URIs to write new data files to.

        Returns
        -------
        MergeInsertBuilder
            The builder instance for method chaining.
        """
        return super(MergeInsertBuilder, self).target_bases(bases)

    def target_all_bases(self, include_primary: bool = True) -> "MergeInsertBuilder":
        """
        Write new fragments to every base registered in the dataset manifest.

        The bases are resolved when the merge insert executes, so bases added
        later are picked up automatically. When ``include_primary`` is True
        (the default), the dataset's primary storage participates in the
        round-robin rotation as the first slot. Cannot be combined with
        :meth:`target_bases`.

        Parameters
        ----------
        include_primary : bool, default True
            Whether the dataset's primary storage is part of the rotation.

        Returns
        -------
        MergeInsertBuilder
            The builder instance for method chaining.
        """
        return super(MergeInsertBuilder, self).target_all_bases(include_primary)

    def explain_plan(
        self, schema: Optional[pa.Schema] = None, verbose: bool = False
    ) -> str:
        """
        Generate the execution plan for the merge insert operation.

        This reports the plan a *streaming* source of the given schema would run.
        It takes a schema rather than data, so it cannot know how ``execute`` would
        wrap the source; see the note under the example.

        Parameters
        ----------
        schema : Optional[pa.Schema], default None
            The schema of the source data that would be merged into the dataset.
            If None, defaults to the dataset's schema. This determines the columns
            available and their types, which affects the generated execution plan.
        verbose : bool, default False
            If True, provides more detailed information in the plan output.

        Returns
        -------
        str
            A formatted string representation of the execution plan.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table([
        ...     pa.array([1, 2, 3]),
        ...     pa.array(['alice', 'bob', 'charlie']),
        ...     pa.array([100.0, 200.0, 300.0])
        ... ], names=['id', 'name', 'score'])
        >>> dataset = lance.write_dataset(data, "memory://test_dataset")

        >>> # Using default dataset schema
        >>> builder = dataset.merge_insert("id")
        >>> builder = builder.when_matched_update_all().when_not_matched_insert_all()
        >>> plan = builder.explain_plan()  # Uses dataset schema
        >>> print(plan) # doctest: +ELLIPSIS
        MergeInsert: on=[id], when_matched=UpdateAll, when_not_matched=InsertAll, ...
          CoalescePartitionsExec
            ProjectionExec: expr=[...]
              HashJoinExec: mode=CollectLeft, join_type=Right, ...
                LanceRead: uri=test_dataset/data, projection=[id], ...
                RepartitionExec: ...
                  ProjectionExec: expr=[..., true as __merge_source_sentinel]
                    StreamingTableExec: partition_sizes=1, ...
        <BLANKLINE>

        This is always the streaming shape. `explain_plan` receives a schema rather
        than data, so it cannot know how `execute` would wrap the source, and the
        wrapping affects the plan. Use `analyze_plan`, which receives the real
        source, when that matters. Note that `analyze_plan` runs the merge to
        collect metrics and may write data files, whereas `explain_plan` writes
        nothing.

        >>> # Or with explicit schema
        >>> source_schema = pa.schema([
        ...     pa.field("id", pa.int64()),
        ...     pa.field("name", pa.string()),
        ...     pa.field("score", pa.float64())
        ... ])
        >>> plan = builder.explain_plan(source_schema, verbose=True)
        >>> print(plan) # doctest: +ELLIPSIS
        MergeInsert: on=[id], when_matched=UpdateAll, when_not_matched=InsertAll, ...
          CoalescePartitionsExec
            ProjectionExec: expr=[...]
              HashJoinExec: mode=CollectLeft, join_type=Right, ...
                ...
        """
        return super(MergeInsertBuilder, self).explain_plan(schema, verbose=verbose)

    def analyze_plan(
        self,
        data_obj: ReaderLike,
        *,
        schema: Optional[pa.Schema] = None,
    ) -> str:
        """
        Generate the execution plan and analyze its performance with real data.

        This method creates the execution plan using the provided source data,
        executes it to collect performance metrics, and returns the analysis
        as a formatted string. This is useful for understanding the actual
        performance characteristics of the merge insert operation with your data.

        .. warning::
            This method executes the merge insert operation to collect metrics
            but **does not commit the changes**. While data files may be written
            to storage during execution, they will not be referenced by any dataset
            version and the dataset remains unchanged. This is intended for
            performance analysis only.

        Parameters
        ----------
        data_obj : ReaderLike
            The source data that would be merged into the dataset. This parameter
            can be any source of data (e.g. table / dataset) that
            :func:`~lance.write_dataset` accepts.
        schema : Optional[pa.Schema], default None
            The schema of the data. This only needs to be supplied whenever the data
            source is some kind of generator.

        Returns
        -------
        str
            A formatted string representation of the plan analysis with metrics.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table([
        ...     pa.array([1, 2, 3]),
        ...     pa.array(['alice', 'bob', 'charlie']),
        ...     pa.array([100.0, 200.0, 300.0])
        ... ], names=['id', 'name', 'score'])
        >>> dataset = lance.write_dataset(data, "memory://analyze_dataset")
        >>> new_data = pa.table([
        ...     pa.array([1, 4]),
        ...     pa.array(["updated_alice", "david"]),
        ...     pa.array([150.0, 400.0])
        ... ], names=["id", "name", "score"])
        >>> builder = dataset.merge_insert("id")
        >>> builder = builder.when_matched_update_all().when_not_matched_insert_all()
        >>> analysis = builder.analyze_plan(new_data)
        >>> print(analysis) # doctest: +ELLIPSIS
            MergeInsert: elapsed=..., on=[id], ..., metrics=[..., bytes_written=..., ...]
              CoalescePartitionsExec, elapsed=..., metrics=[output_rows=..., elapsed_compute=...]
                ProjectionExec: elapsed=..., expr=[...], metrics=[...]
                  RepartitionExec: ...
                    HashJoinExec: elapsed=..., mode=CollectLeft, join_type=Right, ...
                      LanceRead: elapsed=..., ..., metrics=[..., bytes_read=..., ...]
                      ProjectionExec: elapsed=..., expr=[..., true as __merge_source_sentinel], metrics=[...]
                        DataSourceExec: ..., metrics=[]

        The reported plan follows how the source was passed. `new_data` above is a
        `pa.Table`, so it is wrapped in an in-memory table that reports exact
        statistics, while a `pa.RecordBatchReader` reports none. DataFusion picks
        the side it collects from those statistics and from the two sides' sizes.
        The target here is three rows, so it is the collected side either way; with
        a larger target the same merge can plan differently depending on which
        source you hand it. Use `explain_plan` only for the streaming shape: it
        takes a schema rather than data, so it cannot know how the source would be
        wrapped.

        The two key parts of the plan analysis are LanceRead and MergeInsert.
        LanceRead scans join keys and columns in conditions. MergeInsert writes
        data files to storage.

        Key metrics in MergeInsert operations:
        - bytes_written: total bytes written to storage
        - num_files_written: number of new data files created
        - num_inserted_rows: rows added that didn't match existing data
        - num_updated_rows: existing rows that were updated
        - num_deleted_rows: rows removed (when using delete conditions)

        Key metrics in LanceRead operations:
        - bytes_read: bytes read from storage
        - fragments_scanned: number of data fragments accessed
        - rows_scanned: total rows examined during the scan
        - iops: number of I/O operations performed
        - requests: number of storage requests made
        - task_wait_time: time spent waiting for scheduled I/O
        - decode_time: time spent decoding, summed over concurrent decode tasks
        """  # noqa: E501
        if _is_native_dataset_source(data_obj):
            return super(MergeInsertBuilder, self).analyze_plan_dataset(data_obj._ds)
        source = _coerce_source(data_obj, schema)
        if source.mode == "replayable":
            return super(MergeInsertBuilder, self).analyze_plan_replayable(
                source.schema, source.open_reader
            )
        reader = source.open_reader()
        if source.mode == "materialized":
            return super(MergeInsertBuilder, self).analyze_plan_batches(reader)

        return super(MergeInsertBuilder, self).analyze_plan(reader)

    def mark_sstables_as_compacted(
        self, sstables: "List[mem_wal.CompactedSsTable]", dir_id: str = ""
    ) -> "MergeInsertBuilder":
        """Mark MemWAL SSTables as compacted into the base table.

        Call this before executing merge_insert when it compacts MemWAL SSTables.
        The progress is recorded in the same commit as the data.

        For multi-pass compaction, call this only on the final successful
        data-changing pass. Intermediate passes must not carry compaction
        progress. Lance cannot tell whether a caller has another pass planned,
        so it cannot enforce this: if a delete pass carried the marker and the
        process then died before the matching upsert, the recorded progress
        would claim rows were copied in that never were.

        Parameters
        ----------
        sstables : list of CompactedSsTable
            SSTables to mark as compacted.
        dir_id : str, optional
            The ``dir_id`` of the MemWAL the SSTables belong to; empty for a
            MemWAL without one. The commit fails if the table's MemWAL has
            since been disabled or replaced.
        """
        from ..mem_wal import _to_raw_compacted_sstables

        raw_sstables = _to_raw_compacted_sstables(sstables)
        super(MergeInsertBuilder, self).mark_sstables_as_compacted(raw_sstables, dir_id)
        return self
