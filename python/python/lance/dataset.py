# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import copy
import dataclasses
import json
import os
import random
import time
import uuid
import warnings
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)

import pyarrow as pa
import pyarrow.dataset
from pyarrow import RecordBatch, Schema

from lance.log import LOGGER

from .blob import BlobFile
from .cuvs import is_cuvs_accelerator
from .dependencies import (
    _check_for_numpy,
    _check_for_torch,
    torch,
)
from .dependencies import numpy as np
from .dependencies import pandas as pd
from .fragment import DataFile, FragmentMetadata, LanceFragment
from .indices import IndexConfig, SupportedDistributedIndices
from .lance import (
    CleanupStats,
    Compaction,
    CompactionMetrics,
    DatasetBasePath,
    IOStats,
    LanceSchema,
    PySearchFilter,
    ScanStatistics,
    _Dataset,
    _MergeInsertBuilder,
    _Scanner,
    _write_dataset,
    indices,
)
from .lance import __version__ as __version__
from .lance import _Session as Session
from .query import FullTextQuery
from .types import _coerce_reader
from .udf import BatchUDF, normalize_transform
from .udf import BatchUDFCheckpoint as BatchUDFCheckpoint
from .udf import batch_udf as batch_udf
from .util import _target_partition_size_to_num_partitions, td_to_micros

if TYPE_CHECKING:
    from pyarrow._compute import Expression

    from lance.namespace import LanceNamespace

    from .commit import CommitLock
    from .lance.indices import IndexDescription
    from .progress import FragmentWriteProgress, IndexProgress
    from .types import ReaderLike

    QueryVectorLike = Union[
        pd.Series,
        pa.Array,
        pa.Scalar,
        np.ndarray,
        Iterable[float],
    ]
LANCE_COMMIT_MESSAGE_KEY = "__lance_commit_message"
_BLOB_PANDAS_MODE_LAZY = "lazy"
_BLOB_PANDAS_MODE_BYTES = "bytes"
_BLOB_PANDAS_MODE_DESCRIPTIONS = "descriptions"
_BLOB_PANDAS_MODES = frozenset(
    {
        _BLOB_PANDAS_MODE_LAZY,
        _BLOB_PANDAS_MODE_BYTES,
        _BLOB_PANDAS_MODE_DESCRIPTIONS,
    }
)
_BLOB_ROW_ADDR_COLUMN = "_rowaddr"


def _field_metadata_value(field: pa.Field, key: str) -> Optional[bytes]:
    metadata = field.metadata
    if metadata is None:
        return None
    return metadata.get(key.encode("utf-8"))


def _is_blob_field(field: pa.Field) -> bool:
    return (
        _field_metadata_value(field, "lance-encoding:blob") == b"true"
        or _field_metadata_value(field, "ARROW:extension:name") == b"lance.blob.v2"
    )


def _blob_columns_in_schema(schema: pa.Schema) -> set[str]:
    return {field.name for field in schema if _is_blob_field(field)}


def _normalize_blob_pandas_mode(
    blob_mode: str,
) -> Literal["lazy", "bytes", "descriptions"]:
    if blob_mode not in _BLOB_PANDAS_MODES:
        raise ValueError("blob_mode must be one of: 'lazy', 'bytes', 'descriptions'")
    return cast("Literal['lazy', 'bytes', 'descriptions']", blob_mode)


def _simple_source_column(expr: str) -> Optional[str]:
    expr = expr.strip()
    if expr.startswith("`") and expr.endswith("`") and expr.count("`") == 2:
        return expr[1:-1]
    if "." in expr or any(ch.isspace() for ch in expr):
        return None
    if not expr:
        return None
    allowed = set("_$")
    if not (expr[0].isalpha() or expr[0] == "_"):
        return None
    if any(not (ch.isalnum() or ch in allowed) for ch in expr[1:]):
        return None
    return expr


def _blob_column_sources(
    schema: pa.Schema,
    snapshot: Dict[str, Any],
    dataset_schema: pa.Schema,
) -> dict[str, str]:
    blob_columns = {}
    output_blob_columns = [field.name for field in schema if _is_blob_field(field)]
    columns_with_transform = snapshot.get("_columns_with_transform")
    if not columns_with_transform:
        return {name: name for name in output_blob_columns}

    source_is_blob = {field.name for field in dataset_schema if _is_blob_field(field)}
    for name in output_blob_columns:
        expr = dict(columns_with_transform).get(name)
        source = _simple_source_column(expr) if expr is not None else name
        if source is None or source not in source_is_blob:
            raise NotImplementedError(
                "blob-aware to_pandas only supports direct blob column references "
                "for transformed projections"
            )
        blob_columns[name] = source
    return blob_columns


def _snapshot_scanner_builder(builder: "ScannerBuilder") -> Dict[str, Any]:
    """Capture Python-side scanner config needed to rebuild the scan later.

    The native scanner object does not preserve the original builder arguments.
    We need these values to recreate the same scan when `to_pandas` switches blob
    handling modes or injects `_rowaddr` for lazy blob export.
    """

    def snapshot_value(value: Any) -> Any:
        try:
            return copy.deepcopy(value)
        except (TypeError, AttributeError):
            return value

    return {
        key: snapshot_value(value)
        for key, value in vars(builder).items()
        if key != "ds"
    }


def _scanner_from_snapshot(
    ds: "LanceDataset", snapshot: Dict[str, Any]
) -> "LanceScanner":
    builder = ScannerBuilder(ds)
    for key, value in snapshot.items():
        if key == "_columns" and value is not None:
            setattr(builder, key, list(value))
        elif key == "_columns_with_transform" and value is not None:
            setattr(builder, key, list(value))
        elif key == "_fragments" and value is not None:
            setattr(builder, key, list(value))
        elif key == "_orderings" and value is not None:
            setattr(builder, key, list(value))
        else:
            setattr(builder, key, value)
    return builder.to_scanner()


def _is_null_blob_description(description: Any) -> bool:
    if description is None:
        return True
    if not isinstance(description, dict):
        return False
    if description.keys() == {"position", "size"}:
        return description["position"] == 1 and description["size"] == 0
    if description.keys() == {"kind", "position", "size", "blob_id", "blob_uri"}:
        return (
            description["kind"] == 0
            and description["position"] == 0
            and description["size"] == 0
            and description["blob_id"] == 0
            and description["blob_uri"] == ""
        )
    return False


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
        """
        reader = _coerce_reader(data_obj, schema)

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
        reader = _coerce_reader(data_obj, schema)

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

    def explain_plan(
        self, schema: Optional[pa.Schema] = None, verbose: bool = False
    ) -> str:
        """
        Generate the execution plan for the merge insert operation.

        This method creates the execution plan that would be used for the given
        source schema and returns it as a formatted string for debugging and
        analysis purposes.

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
                CooperativeExec
                  LanceRead: uri=test_dataset/data, projection=[id], ...
                RepartitionExec: ...
                  ProjectionExec: expr=[..., true as __merge_source_sentinel]
                    StreamingTableExec: partition_sizes=1, ...
        <BLANKLINE>

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
                  HashJoinExec: elapsed=..., mode=CollectLeft, join_type=Right, ...
                    CooperativeExec, elapsed=..., metrics=[]
                      LanceRead: elapsed=..., ..., metrics=[..., bytes_read=..., ...]
                    RepartitionExec: ...
                      ProjectionExec: elapsed=..., expr=[..., true as __merge_source_sentinel], metrics=[...]
                        StreamingTableExec: ..., metrics=[]

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
        """  # noqa: E501
        reader = _coerce_reader(data_obj, schema)
        return super(MergeInsertBuilder, self).analyze_plan(reader)


class LanceDataset(pa.dataset.Dataset):
    """A Lance Dataset in Lance format where the data is stored at the given uri."""

    def __init__(
        self,
        uri: Union[str, Path],
        version: Optional[int | str] = None,
        block_size: Optional[int] = None,
        index_cache_size: Optional[int] = None,
        metadata_cache_size: Optional[int] = None,
        commit_lock: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        serialized_manifest: Optional[bytes] = None,
        default_scan_options: Optional[Dict[str, Any]] = None,
        metadata_cache_size_bytes: Optional[int] = None,
        index_cache_size_bytes: Optional[int] = None,
        read_params: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None,
        namespace_client: Optional[Any] = None,
        table_id: Optional[List[str]] = None,
        namespace_client_managed_versioning: bool = False,
    ):
        uri = os.fspath(uri) if isinstance(uri, Path) else uri
        self._uri = uri
        self._storage_options = storage_options

        # Handle deprecation warning for index_cache_size
        if index_cache_size is not None:
            warnings.warn(
                "The 'index_cache_size' parameter is deprecated. "
                "Use 'index_cache_size_bytes' instead. "
                "The old parameter will be converted to bytes using 20 MiB per entry.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Store namespace_client and table_id for credential refresh in file operations
        self._namespace_client = namespace_client
        self._table_id = table_id
        self._namespace_client_managed_versioning = namespace_client_managed_versioning

        # Storage options provider is automatically created in Rust when
        # namespace_client and table_id are provided
        self._ds = _Dataset(
            uri,
            version,
            block_size,
            index_cache_size,
            metadata_cache_size,
            commit_lock,
            storage_options,
            serialized_manifest,
            metadata_cache_size_bytes=metadata_cache_size_bytes,
            index_cache_size_bytes=index_cache_size_bytes,
            read_params=read_params,
            session=session,
            namespace_client=namespace_client,
            table_id=table_id,
            namespace_client_managed_versioning=namespace_client_managed_versioning,
        )
        self._default_scan_options = default_scan_options
        self._read_params = read_params

    @classmethod
    def __deserialize__(
        cls,
        uri: str,
        storage_options: Optional[Dict[str, str]],
        version: int,
        manifest: bytes,
        default_scan_options: Optional[Dict[str, Any]],
        read_params: Optional[Dict[str, Any]] = None,
    ):
        return cls(
            uri,
            version,
            storage_options=storage_options,
            serialized_manifest=manifest,
            default_scan_options=default_scan_options,
            read_params=read_params,
        )

    def __reduce__(self):
        return type(self).__deserialize__, (
            self.uri,
            self._storage_options,
            self._ds.version(),
            self._ds.serialized_manifest(),
            self._default_scan_options,
            self._read_params,
        )

    def __getstate__(self):
        return (
            self.uri,
            self._storage_options,
            self._ds.version(),
            self._ds.serialized_manifest(),
            self._default_scan_options,
            self._read_params,
        )

    def __setstate__(self, state):
        # Handle backwards compatibility - state may not have read_params
        (
            self._uri,
            self._storage_options,
            version,
            manifest,
            default_scan_options,
            *rest,  # Capture optional read_params
        ) = state
        read_params = rest[0] if rest else None
        self._ds = _Dataset(
            self._uri,
            version,
            storage_options=self._storage_options,
            manifest=manifest,
            default_scan_options=default_scan_options,
            read_params=read_params,
        )
        self._default_scan_options = default_scan_options
        self._read_params = read_params
        self._namespace_client = None
        self._table_id = None
        self._namespace_client_managed_versioning = False

    def __copy__(self):
        ds = LanceDataset.__new__(LanceDataset)
        ds._uri = self._uri
        ds._storage_options = self._storage_options
        ds._namespace_client = self._namespace_client
        ds._table_id = self._table_id
        ds._namespace_client_managed_versioning = (
            self._namespace_client_managed_versioning
        )
        ds._ds = copy.copy(self._ds)
        ds._default_scan_options = self._default_scan_options
        ds._read_params = self._read_params.copy() if self._read_params else None
        return ds

    def __len__(self):
        return self.count_rows()

    @property
    def uri(self) -> str:
        """
        The location of the data
        """
        return self._uri

    @property
    def tags(self) -> Tags:
        """Tag management for the dataset.

        Similar to Git, tags are a way to add metadata to a specific version of the
        dataset.

        .. warning::

            Tagged versions are exempted from the :py:meth:`cleanup_old_versions()`
            process.

            To remove a version that has been tagged, you must first
            :py:meth:`~Tags.delete` the associated tag.

        Examples
        --------

        .. code-block:: python

            ds = lance.open("dataset.lance")
            ds.tags.create("v2-prod-20250203", 10)

            tags = ds.tags.list()

        """
        return Tags(self._ds)

    @property
    def branches(self) -> "Branches":
        """Branch management for the dataset.

        Examples
        --------
        --------

        .. code-block:: python

            ds = lance.open("dataset.lance")
            ds.create_branch("v2-prod-20250203")

            branches = ds.branches.list()

        """
        return Branches(self._ds)

    def create_branch(
        self,
        branch: str,
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]] = None,
        storage_options: Optional[Dict[str, str]] = None,
    ) -> "LanceDataset":
        """Create a new branch from a version or tag.

        Parameters
        ----------
        branch: str
            Name of the branch to create.
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]
            An integer specifies a version number in the current branch; a string
            specifies a tag name; a Tuple[Optional[str], Optional[int]] specifies
            a version number in a specified branch. (None, None) means the latest
            version_number on the main branch.
        storage_options: Optional[Dict[str, str]]
            Storage options for the underlying object store. If not provided,
            the storage options from the current dataset will be used.

        Returns
        -------
        LanceDataset
            A dataset instance pointing to the new branch.
        """
        if storage_options is None:
            storage_options = self._storage_options
        new_ds = self._ds.create_branch(branch, reference, storage_options)
        ds = LanceDataset.__new__(LanceDataset)
        ds._ds = new_ds
        ds._uri = new_ds.uri
        ds._storage_options = self._storage_options
        ds._namespace_client = self._namespace_client
        ds._table_id = self._table_id
        ds._default_scan_options = self._default_scan_options
        ds._read_params = self._read_params
        return ds

    def checkout_latest(self):
        """Check out the latest version of the current branch."""
        self._ds.checkout_latest()

    def list_indices(self) -> List[Index]:
        """
        Returns physical index segment information for all indices in the dataset.

        This method is deprecated as it requires loading the statistics for each index
        which can be a very expensive operation.  It also exposes physical index
        segments directly.  Instead use describe_indices() for logical index
        descriptions and index_statistics() to get the statistics for individual
        indexes of interest.
        """
        warnings.warn(
            "The 'list_indices' method is deprecated. It may be removed in a future "
            "version. Use describe_indices() instead.",
            DeprecationWarning,
        )

        return self._ds.load_indices()

    def describe_indices(self) -> List[IndexDescription]:
        """Returns logical index information aggregated across all segments."""
        return self._ds.describe_indices()

    def index_statistics(self, index_name: str) -> Dict[str, Any]:
        warnings.warn(
            "LanceDataset.index_statistics() is deprecated, "
            + "use LanceDataset.stats.index_stats() instead",
            DeprecationWarning,
        )
        return json.loads(self._ds.index_statistics(index_name))

    @property
    def has_index(self):
        return len(self.describe_indices()) > 0

    def _apply_default_scan_options(self, builder: ScannerBuilder):
        if self._default_scan_options:
            builder.apply_defaults(self._default_scan_options)
        return builder

    def scanner(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[
            Union[str, pa.compute.Expression, FullTextQuery, VectorSearchQuery, dict]
        ] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
        batch_size: Optional[int] = None,
        batch_size_bytes: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: Optional[bool] = None,
        fragments: Optional[Iterable[LanceFragment]] = None,
        full_text_query: Optional[Union[str, dict, FullTextQuery]] = None,
        *,
        prefilter: Optional[bool] = None,
        with_row_id: Optional[bool] = None,
        with_row_address: Optional[bool] = None,
        use_stats: Optional[bool] = None,
        fast_search: Optional[bool] = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        blob_handling: Optional[
            Literal["all_binary", "blobs_descriptions", "all_descriptions"]
        ] = None,
        use_scalar_index: Optional[bool] = None,
        include_deleted_rows: Optional[bool] = None,
        scan_stats_callback: Optional[Callable[[ScanStatistics], None]] = None,
        strict_batch_size: Optional[bool] = None,
        order_by: Optional[List[Union[ColumnOrdering, str]]] = None,
        disable_scoring_autoprojection: Optional[bool] = None,
    ) -> LanceScanner:
        """Return a Scanner that can support various pushdowns.

        Parameters
        ----------
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
        filter: pa.compute.Expression, str, VectorSearchQuery, FullTextQuery or dict
            Lance supports 2 kinds of filters: expression filter and search filter.

            - Expression filter is pa.compute.Expression or str that is a valid SQL
             where clause. See `Lance filter pushdown
             <https://lance.org/guide/read_and_write/#filter-push-down>`_
             for valid SQL expressions. Expression filter is applied to filtered scan,
             full text search and vector search.

            - VectorSearchQuery is a vector search that can only be applied to full
             text search. Example:
           .. code-block:: python

             filter=VectorSearchQuery(
                        "vector",
                        np.array([12, 17, 300, 10], dtype=np.float32),
                        5,
                        20,
                        True,
                    )

            - FullTextQuery is a full text search that can only be applied to vector
             search. Example:
           .. code-block:: python

             filter=PhraseQuery("hello world", "col")

            - Dictionary is a combined filter containing both expression filter with
             key `expr_filter` and search filter with key `search_filter`. Example:
           .. code-block:: python

                scanner = ds.scanner(
                    nearest={
                        "column": "vector",
                        "q": np.array([12, 17, 300, 10], dtype=np.float32),
                        "k": 5,
                        "minimum_nprobes": 20,
                        "use_index": True,
                    },
                    filter={
                        "expr_filter": "category='geography'",
                        "search_filter": PhraseQuery("hello world", "col"),
                    },
                )
        limit: int, default None
            Fetch up to this many rows. All rows if None or unspecified.
        offset: int, default None
            Fetch starting with this row. 0 if None or unspecified.
        nearest: dict, default None
            Get the rows corresponding to the K most similar vectors. Example:

            .. code-block:: python

                {
                    "column": <embedding col name>,
                    "q": <query vector as pa.Float32Array>,
                    "k": 10,
                    "minimum_nprobes": 1,
                    "maximum_nprobes": 50,
                    "refine_factor": 1,
                    "distance_range": (0.0, 1.0),
                }

        batch_size: int, default None
            The maximum number of rows per batch.  In some cases batches can be
            smaller than this size.  Note: this can be overridden by
            ``batch_size_bytes`` or by a dataset-level ``batch_size_bytes``
            configured via ``FileReaderOptions``.
        batch_size_bytes: int, default None
            If set, the scanner will produce batches whose total size in bytes
            is approximately this value, overriding the row-based ``batch_size``.
            This can also be configured at the dataset level via
            ``FileReaderOptions``.  A scanner-level setting takes precedence
            over the dataset-level default.
        io_buffer_size: int, default None
            The size of the IO buffer.  See ``ScannerBuilder.io_buffer_size``
            for more information.
        batch_readahead: int, optional
            The number of batches to read ahead.
        fragment_readahead: int, optional
            The number of fragments to read ahead.
        scan_in_order: bool, default True
            Whether to read the fragments and batches in order. If false,
            throughput may be higher, but batches will be returned out of order
            and memory use might increase.
        fragments: iterable of LanceFragment, default None
            If specified, only scan these fragments. If scan_in_order is True, then
            the fragments will be scanned in the order given.
        prefilter: bool, default False
            If True then the filter will be applied before the vector query is run.
            This will generate more correct results but it may be a more costly
            query.  It's generally good when the filter is highly selective.

            If False then the filter will be applied after the vector query is run.
            This will perform well but the results may have fewer than the requested
            number of rows (or be empty) if the rows closest to the query do not
            match the filter.  It's generally good when the filter is not very
            selective.
        use_scalar_index: bool, default True
            Lance will automatically use scalar indices to optimize a query.  In some
            corner cases this can make query performance worse and this parameter can
            be used to disable scalar indices in these cases.
        late_materialization: bool or List[str], default None
            Allows custom control over late materialization.  Late materialization
            fetches non-query columns using a take operation after the filter.  This
            is useful when there are few results or columns are very large.

            Early materialization can be better when there are many results or the
            columns are very narrow.

            If True, then all columns are late materialized.
            If False, then all columns are early materialized.
            If a list of strings, then only the columns in the list are
            late materialized.

            The default uses a heuristic that assumes filters will select about 0.1%
            of the rows.  If your filter is more selective (e.g. find by id) you may
            want to set this to True.  If your filter is not very selective (e.g.
            matches 20% of the rows) you may want to set this to False.
        blob_handling: str, default None
            Controls how blob columns are returned.

            - "all_binary": read blob columns as binary / large_binary values
            - "blobs_descriptions": read blob columns as descriptions (default)
            - "all_descriptions": read all binary columns as descriptions
        full_text_query: str or dict, optional
            query string to search for, the results will be ranked by BM25.
            e.g. "hello world", would match documents containing "hello" or "world".
            or a dictionary with the following keys:

            - columns: list[str]
                The columns to search,
                currently only supports a single column in the columns list.
            - query: str
                The query string to search for.
        fast_search:  bool, default False
            If True, then the search will only be performed on the indexed data, which
            yields faster search time.
        scan_stats_callback: Callable[[ScanStatistics], None], default None
            A callback function that will be called with the scan statistics after the
            scan is complete.  Errors raised by the callback will be logged but not
            re-raised.
        include_deleted_rows: bool, default False
            If True, then rows that have been deleted, but are still present in the
            fragment, will be returned.  These rows will have the _rowid column set
            to null.  All other columns will reflect the value stored on disk and may
            not be null.

            Note: if this is a search operation, or a take operation (including scalar
            indexed scans) then deleted rows cannot be returned.
        order_by: list of ColumnOrdering or str, default None
            If not specified, the rows will be returned as the file order
            if scan_in_order is true. Otherwise it will fellow as a random order.
            If specified, the return rows will follow the orderings. If a string is
            specified, it will assume ascending and nulls last ordering.
        disable_scoring_autoprojection: bool, default False
            Currently, when a search (vector or full text) is performed, the scoring
            column (_distance, _score) is added to the end of the output even when a
            projection is specified.  In the future, this will change.  The columns will
            only be present if there is no projection or if they are explicitly
            specified in the projection.

            This parameter allows you to opt-in to the new behavior early, to avoid
            being subject to breaking changes in the future.


        .. note::

            For now, if BOTH filter and nearest is specified, then:

            1. nearest is executed first.
            2. The results are filtered afterwards.


        For debugging ANN results, you can choose to not use the index
        even if present by specifying ``use_index=False``. For example,
        the following will always return exact KNN results:


        .. code-block:: python

            dataset.to_table(nearest={
                "column": "vector",
                "k": 10,
                "q": <query vector>,
                "use_index": False
            }

        """
        builder = ScannerBuilder(self)
        builder = self._apply_default_scan_options(builder)

        if order_by is not None:
            order_by = [
                ColumnOrdering(o) if isinstance(o, str) else o for o in order_by
            ]

        # Calls the setter if the user provided a non-None value
        # We need to avoid calling the setter with a None value so
        # we don't override any defaults from _default_scan_options
        def setopt(opt, val):
            if val is not None:
                opt(val)

        setopt(builder.filter, filter)
        setopt(builder.prefilter, prefilter)
        setopt(builder.limit, limit)
        setopt(builder.offset, offset)
        setopt(builder.batch_size, batch_size)
        setopt(builder.batch_size_bytes, batch_size_bytes)
        setopt(builder.io_buffer_size, io_buffer_size)
        setopt(builder.batch_readahead, batch_readahead)
        setopt(builder.fragment_readahead, fragment_readahead)
        setopt(builder.scan_in_order, scan_in_order)
        setopt(builder.with_fragments, fragments)
        setopt(builder.late_materialization, late_materialization)
        setopt(builder.blob_handling, blob_handling)
        setopt(builder.with_row_id, with_row_id)
        setopt(builder.with_row_address, with_row_address)
        setopt(builder.use_stats, use_stats)
        setopt(builder.use_scalar_index, use_scalar_index)
        setopt(builder.fast_search, fast_search)
        setopt(builder.include_deleted_rows, include_deleted_rows)
        setopt(builder.scan_stats_callback, scan_stats_callback)
        setopt(builder.strict_batch_size, strict_batch_size)
        setopt(builder.order_by, order_by)
        setopt(builder.disable_scoring_autoprojection, disable_scoring_autoprojection)
        # columns=None has a special meaning. we can't treat it as "user didn't specify"
        if self._default_scan_options is None:
            # No defaults, use user-provided, if any
            builder = builder.columns(columns)
        else:
            default_columns = self._default_scan_options.get("columns", None)
            if default_columns is None:
                # No default_columns, use user-provided, if any
                builder = builder.columns(columns)
            else:
                if columns is not None:
                    # User supplied None, fallback to default (no way to override
                    # default to None)
                    builder = builder.columns(columns)
                else:
                    # User supplied non-None, use that
                    builder = builder.columns(default_columns)

        if full_text_query is not None:
            if isinstance(full_text_query, (str, FullTextQuery)):
                builder = builder.full_text_search(full_text_query)
            elif isinstance(full_text_query, dict):
                builder = builder.full_text_search(**full_text_query)
        if nearest is not None:
            builder = builder.nearest(**nearest)
        return builder.to_scanner()

    @property
    def schema(self) -> pa.Schema:
        """
        The pyarrow Schema for this dataset
        """
        if self._default_scan_options is None:
            return self._ds.schema
        else:
            return self.scanner().projected_schema

    @property
    def lance_schema(self) -> "LanceSchema":
        """
        The LanceSchema for this dataset
        """
        return self._ds.lance_schema

    @property
    def data_storage_version(self) -> str:
        """
        The version of the data storage format this dataset is using
        """
        return self._ds.data_storage_version

    @property
    def max_field_id(self) -> int:
        """
        The max_field_id in manifest
        """
        return self._ds.max_field_id

    def to_table(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
        batch_size: Optional[int] = None,
        batch_size_bytes: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: Optional[bool] = None,
        *,
        prefilter: Optional[bool] = None,
        with_row_id: Optional[bool] = None,
        with_row_address: Optional[bool] = None,
        use_stats: Optional[bool] = None,
        fast_search: Optional[bool] = None,
        full_text_query: Optional[Union[str, dict, FullTextQuery]] = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        blob_handling: Optional[str] = None,
        use_scalar_index: Optional[bool] = None,
        include_deleted_rows: Optional[bool] = None,
        order_by: Optional[List[ColumnOrdering]] = None,
        disable_scoring_autoprojection: Optional[bool] = None,
    ) -> pa.Table:
        """Read the data into memory as a :py:class:`pyarrow.Table`

        Parameters
        ----------
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
        filter : pa.compute.Expression or str
            Expression or str that is a valid SQL where clause. See
            `Lance filter pushdown <https://lance.org/guide/read_and_write/#filter-push-down>`_
            for valid SQL expressions.
        limit: int, default None
            Fetch up to this many rows. All rows if None or unspecified.
        offset: int, default None
            Fetch starting with this row. 0 if None or unspecified.
        nearest: dict, default None
            Get the rows corresponding to the K most similar vectors. Example:

            .. code-block:: python

                {
                    "column": <embedding col name>,
                    "q": <query vector as pa.Float32Array>,
                    "k": 10,
                    "metric": "cosine",
                    "minimum_nprobes": 1,
                    "maximum_nprobes": 50,
                    "refine_factor": 1
                }

        batch_size: int, optional
            The number of rows to read at a time.
        io_buffer_size: int, default None
            The size of the IO buffer.  See ``ScannerBuilder.io_buffer_size``
            for more information.
        batch_readahead: int, optional
            The number of batches to read ahead.
        fragment_readahead: int, optional
            The number of fragments to read ahead.
        scan_in_order: bool, optional, default True
            Whether to read the fragments and batches in order. If false,
            throughput may be higher, but batches will be returned out of order
            and memory use might increase.
        prefilter: bool, optional, default False
            Run filter before the vector search.
        late_materialization: bool or List[str], default None
            Allows custom control over late materialization.  See
            ``ScannerBuilder.late_materialization`` for more information.
        blob_handling: str, default None
            Controls how blob columns are returned. See ``LanceDataset.scanner`` for
            details.
        use_scalar_index: bool, default True
            Allows custom control over scalar index usage.  See
            ``ScannerBuilder.use_scalar_index`` for more information.
        with_row_id: bool, optional, default False
            Return row ID.
        with_row_address: bool, optional, default False
            Return row address
        use_stats: bool, optional, default True
            Use stats pushdown during filters.
        fast_search: bool, optional, default False
        full_text_query: str or dict, optional
            query string to search for, the results will be ranked by BM25.
            e.g. "hello world", would match documents contains "hello" or "world".
            or a dictionary with the following keys:

            - columns: list[str]
                The columns to search,
                currently only supports a single column in the columns list.
            - query: str
                The query string to search for.
        include_deleted_rows: bool, optional, default False
            If True, then rows that have been deleted, but are still present in the
            fragment, will be returned.  These rows will have the _rowid column set
            to null.  All other columns will reflect the value stored on disk and may
            not be null.

            Note: if this is a search operation, or a take operation (including scalar
            indexed scans) then deleted rows cannot be returned.
        order_by: list of ColumnOrdering, default None
            If not specified, the rows will be returned as the file order
            if scan_in_order is true. Otherwise it will fellow as a random order.
            If specified, the return rows will follow the orderings. If a string is
            specified, it will assume ascending and nulls last ordering.
        disable_scoring_autoprojection: bool, default False
            Currently, when a search (vector or full text) is performed, the scoring
            column (_distance, _score) is added to the end of the output even when a
            projection is specified.  In the future, this will change.  The columns will
            only be present if there is no projection or if they are explicitly
            specified in the projection.

            This parameter allows you to opt-in to the new behavior early, to avoid
            being subject to breaking changes in the future.

        Notes
        -----
        If BOTH filter and nearest is specified, then:

        1. nearest is executed first.
        2. The results are filtered afterward, unless pre-filter sets to True.
        """
        return self.scanner(
            columns=columns,
            filter=filter,
            limit=limit,
            offset=offset,
            nearest=nearest,
            batch_size=batch_size,
            batch_size_bytes=batch_size_bytes,
            io_buffer_size=io_buffer_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            late_materialization=late_materialization,
            blob_handling=blob_handling,
            use_scalar_index=use_scalar_index,
            scan_in_order=scan_in_order,
            prefilter=prefilter,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            use_stats=use_stats,
            fast_search=fast_search,
            full_text_query=full_text_query,
            include_deleted_rows=include_deleted_rows,
            order_by=order_by,
            disable_scoring_autoprojection=disable_scoring_autoprojection,
        ).to_table()

    def to_pandas(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
        batch_size: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: Optional[bool] = None,
        *,
        prefilter: Optional[bool] = None,
        with_row_id: Optional[bool] = None,
        with_row_address: Optional[bool] = None,
        use_stats: Optional[bool] = None,
        fast_search: Optional[bool] = None,
        full_text_query: Optional[Union[str, dict, FullTextQuery]] = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        blob_mode: str = _BLOB_PANDAS_MODE_LAZY,
        use_scalar_index: Optional[bool] = None,
        include_deleted_rows: Optional[bool] = None,
        order_by: Optional[List[ColumnOrdering]] = None,
        disable_scoring_autoprojection: Optional[bool] = None,
        **kwargs,
    ) -> "pd.DataFrame":
        """Read the data into a :py:class:`pandas.DataFrame`.

        Parameters are the same as :meth:`to_table`, except pandas export uses
        ``blob_mode`` instead of Arrow-facing ``blob_handling``.

        Parameters
        ----------
        blob_mode: str, default "lazy"
            Controls how blob columns are returned.

            - ``"lazy"``: return :class:`lance.BlobFile` objects
            - ``"bytes"``: return Python ``bytes``
            - ``"descriptions"``: preserve ``to_table().to_pandas()`` behavior
        **kwargs
            Forwarded to :meth:`pyarrow.Table.to_pandas` for non-blob columns.
        """
        return self.scanner(
            columns=columns,
            filter=filter,
            limit=limit,
            offset=offset,
            nearest=nearest,
            batch_size=batch_size,
            io_buffer_size=io_buffer_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            late_materialization=late_materialization,
            use_scalar_index=use_scalar_index,
            scan_in_order=scan_in_order,
            prefilter=prefilter,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            use_stats=use_stats,
            fast_search=fast_search,
            full_text_query=full_text_query,
            include_deleted_rows=include_deleted_rows,
            order_by=order_by,
            disable_scoring_autoprojection=disable_scoring_autoprojection,
        ).to_pandas(blob_mode=blob_mode, **kwargs)

    @property
    def partition_expression(self):
        """
        Not implemented (just override pyarrow dataset to prevent segfault)
        """
        raise NotImplementedError("partitioning not yet supported")

    def replace_schema(self, schema: Schema):
        """
        Not implemented (just override pyarrow dataset to prevent segfault)

        See :py:method:`replace_schema_metadata` or :py:method:`replace_field_metadata`
        """
        raise NotImplementedError(
            "Cannot replace the schema of a dataset.  This method exists for backwards"
            " compatibility with pyarrow.  Use replace_schema_metadata or "
            "replace_field_metadata to change the metadata"
        )

    def replace_schema_metadata(self, new_metadata: Dict[str, str]):
        """
        Replace the schema metadata of the dataset

        .. deprecated:: 0.32.1
            Use :func:`update_schema_metadata` with ``replace=True`` instead.
            This method will be removed in a future version.

        Parameters
        ----------
        new_metadata: dict
            The new metadata to set
        """
        warnings.warn(
            "replace_schema_metadata is deprecated. "
            "Use update_schema_metadata(metadata, replace=True) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.update_schema_metadata(new_metadata, replace=True)

    def replace_field_metadata(self, field_name: str, new_metadata: Dict[str, str]):
        """
        Replace the metadata of a field in the schema

        .. deprecated:: 0.32.1
            Use :func:`update_field_metadata` with ``replace=True`` instead.
            This method will be removed in a future version.

        Parameters
        ----------
        field_name: str
            The name of the field to replace the metadata for
        new_metadata: dict
            The new metadata to set
        """
        warnings.warn(
            "replace_field_metadata is deprecated. "
            "Use update_field_metadata with field paths and replace=True instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.update_field_metadata({field_name: new_metadata}, replace=True)

    # Unified metadata APIs

    @property
    def metadata(self) -> Dict[str, str]:
        """
        Get the table metadata of the dataset.

        Returns
        -------
        Dict[str, str]
            The table metadata as a dictionary of key-value pairs.
        """
        return self._ds.metadata()

    @property
    def schema_metadata(self) -> Dict[str, str]:
        """
        Get the schema metadata of the dataset.

        Returns
        -------
        Dict[str, str]
            The schema metadata as a dictionary of key-value pairs.
        """
        return self._ds.schema_metadata()

    def update_metadata(
        self, values: Dict[str, Optional[str]], *, replace: bool = False
    ) -> Dict[str, str]:
        """
        Update the table metadata of the dataset.

        This method supports both incremental updates (default) and full replacement.

        Parameters
        ----------
        values : Dict[str, Optional[str]]
            Metadata updates where keys are metadata keys and values are:
            - str: Set the metadata key to this value
            - None: Remove the metadata key (ignored in replace mode)
        replace : bool, default False
            If True, completely replace all table metadata with the provided values.
            If False, incrementally update only the specified keys.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table([pa.array([1, 2, 3])], names=['x'])
        >>> dataset = lance.write_dataset(data, "memory://test_metadata")
        >>> # Incremental update (add/update specific keys)
        >>> dataset.update_metadata(
        ...     {"author": "John", "version": "1.2"}) # doctest: +ELLIPSIS
        {...}
        >>> # Remove a key (incremental mode only)
        >>> dataset.update_metadata({"old_key": None}) # doctest: +ELLIPSIS
        {...}
        >>> # Full replacement (replaces all metadata)
        >>> dataset.update_metadata(
        ...     {"author": "John"}, replace=True) # doctest: +ELLIPSIS
        {...}
        """
        return self._ds.update_metadata(values, replace=replace)

    def update_config(
        self, values: Dict[str, Optional[str]], *, replace: bool = False
    ) -> Dict[str, str]:
        """
        Update the configuration of the dataset using unified API.

        This method supports both incremental updates (default) and full replacement.

        Parameters
        ----------
        values : Dict[str, Optional[str]]
            Configuration updates where keys are config keys and values are:
            - str: Set the config key to this value
            - None: Remove the config key (ignored in replace mode)
        replace : bool, default False
            If True, completely replace all configuration with the provided values.
            If False, incrementally update only the specified keys.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table([pa.array([1, 2, 3])], names=['x'])
        >>> dataset = lance.write_dataset(data, "memory://test_config")
        >>> # Incremental update (add/update specific keys)
        >>> dataset.update_config(
        ...     {"batch_size": "1000", "compression": "zstd"}) # doctest: +ELLIPSIS
        {...}
        >>> # Remove a key (incremental mode only)
        >>> dataset.update_config({"batch_size": None}) # doctest: +ELLIPSIS
        {...}
        >>> # Full replacement (replaces all config)
        >>> dataset.update_config(
        ...     {"batch_size": "1000"}, replace=True) # doctest: +ELLIPSIS
        {...}
        """
        return self._ds.update_config(values, replace=replace)

    def update_schema_metadata(
        self, values: Dict[str, Optional[str]], *, replace: bool = False
    ) -> Dict[str, str]:
        """
        Update the schema metadata of the dataset.

        This method supports both incremental updates (default) and full replacement.

        Parameters
        ----------
        values : Dict[str, Optional[str]]
            Schema metadata updates where keys are metadata keys and values are:
            - str: Set the metadata key to this value
            - None: Remove the metadata key (ignored in replace mode)
        replace : bool, default False
            If True, completely replace all schema metadata with the provided values.
            If False, incrementally update only the specified keys.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table([pa.array([1, 2, 3])], names=['x'])
        >>> dataset = lance.write_dataset(data, "memory://test_schema")
        >>> # Incremental update (add/update specific keys)
        >>> dataset.update_schema_metadata(
        ...     {"encoding": "utf-8", "created_by": "lance"}) # doctest: +ELLIPSIS
        {...}
        >>> # Remove a key (incremental mode only)
        >>> dataset.update_schema_metadata({"encoding": None}) # doctest: +ELLIPSIS
        {...}
        >>> # Full replacement (replaces all schema metadata)
        >>> dataset.update_schema_metadata(
        ...     {"encoding": "utf-8"}, replace=True) # doctest: +ELLIPSIS
        {...}
        """
        return self._ds.update_schema_metadata(values, replace=replace)

    def update_field_metadata(
        self,
        field_updates: Dict[str, Dict[str, Optional[str]]],
        *,
        replace: bool = False,
    ) -> None:
        """
        Update metadata for multiple fields in the dataset.

        This method supports both incremental updates (default) and full replacement.

        Parameters
        ----------
        field_updates : Dict[str, Dict[str, Optional[str]]]
            Field metadata updates where keys are field paths and values are
            metadata dictionaries.
            For each field's metadata dictionary:
            - str values: Set the metadata key to this value
            - None values: Remove the metadata key (ignored in replace mode)
        replace : bool, default False
            If True, completely replace all metadata for the specified fields.
            If False, incrementally update only the specified keys for each field.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table([pa.array([1, 2, 3]), pa.array(['a', 'b', 'c'])],
        ...                names=['user_id', 'user_name'])
        >>> dataset = lance.write_dataset(data, "memory://test_field")
        >>> # Incremental update for multiple fields
        >>> dataset.update_field_metadata({
        ...     "user_id": {"description": "User ID", "nullable": "false"},
        ...     "user_name": {"description": "User name", "max_length": "100"}
        ... })
        >>>
        >>> # Remove keys (incremental mode only)
        >>> dataset.update_field_metadata({"user_id": {"old_key": None}})
        >>>
        >>> # Full replacement for specific fields
        >>> dataset.update_field_metadata({
        ...     "user_id": {"description": "User ID"}
        ... }, replace=True)
        """
        # Use the new path-based Rust function directly
        self._ds.update_field_metadata_by_path(field_updates, replace=replace)

    def get_fragments(self, filter: Optional[Expression] = None) -> List[LanceFragment]:
        """Get all fragments from the dataset.

        Note: filter is not supported yet.
        """
        if filter is not None:
            raise ValueError("get_fragments() does not support filter yet")
        return [
            LanceFragment(self, fragment_id=None, fragment=f)
            for f in self._ds.get_fragments()
        ]

    def get_fragment(self, fragment_id: int) -> Optional[LanceFragment]:
        """Get the fragment with fragment id."""
        raw_fragment = self._ds.get_fragment(fragment_id)
        if raw_fragment is None:
            return None
        return LanceFragment(self, fragment_id=None, fragment=raw_fragment)

    def io_stats_snapshot(self) -> IOStats:
        """
        Get a snapshot of current IO statistics without resetting counters.

        Returns the current IO statistics without modifying the internal state.
        Use this when you need to check stats without resetting them. Multiple
        calls will return the same values until IO operations are performed.

        Returns
        -------
        IOStats
            Object containing IO statistics with the following attributes:
            - read_iops: Number of read operations
            - read_bytes: Total bytes read
            - write_iops: Number of write operations
            - write_bytes: Total bytes written
            - num_hops: Number of network hops (for remote storage)

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table({"x": [1, 2, 3]})
        >>> dataset = lance.write_dataset(data, "memory://test_stats")
        >>> result = dataset.to_table()
        >>> # Check stats without resetting
        >>> stats = dataset.io_stats_snapshot()
        >>> print(f"Read {stats.read_bytes} bytes in {stats.read_iops} operations")
        Read ... bytes in ... operations
        >>> # Can check again and see the same values
        >>> stats2 = dataset.io_stats_snapshot()
        >>> assert stats.read_bytes == stats2.read_bytes

        See Also
        --------
        io_stats_incremental : Get stats and reset counters for incremental tracking
        """
        return self._ds.io_stats_snapshot()

    def io_stats_incremental(self) -> IOStats:
        """
        Get incremental IO statistics and reset the counters.

        Returns IO statistics (number of operations and bytes) since the last
        time this method was called, then resets the internal counters to zero.
        This is useful for tracking IO operations between different stages of
        processing.

        Returns
        -------
        IOStats
            Object containing IO statistics with the following attributes:
            - read_iops: Number of read operations
            - read_bytes: Total bytes read
            - write_iops: Number of write operations
            - write_bytes: Total bytes written
            - num_hops: Number of network hops (for remote storage)

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table({"x": [1, 2, 3]})
        >>> dataset = lance.write_dataset(data, "memory://test_stats")
        >>> result = dataset.to_table()
        >>> # Get incremental stats (and reset)
        >>> stats = dataset.io_stats_incremental()
        >>> print(f"Read {stats.read_bytes} bytes in {stats.read_iops} operations")
        Read ... bytes in ... operations
        >>> # Next call returns only new stats since last call
        >>> more_data = dataset.to_table()
        >>> stats2 = dataset.io_stats_incremental()
        >>> print(f"Read {stats2.read_bytes} more bytes")
        Read ... more bytes

        See Also
        --------
        io_stats_snapshot : Get stats without resetting counters
        """
        return self._ds.io_stats_incremental()

    def to_batches(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[Union[str, pa.compute.Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[dict] = None,
        batch_size: Optional[int] = None,
        batch_size_bytes: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: Optional[bool] = None,
        *,
        prefilter: Optional[bool] = None,
        with_row_id: Optional[bool] = None,
        with_row_address: Optional[bool] = None,
        use_stats: Optional[bool] = None,
        full_text_query: Optional[Union[str, dict]] = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        blob_handling: Optional[str] = None,
        use_scalar_index: Optional[bool] = None,
        strict_batch_size: Optional[bool] = None,
        order_by: Optional[List[ColumnOrdering]] = None,
        disable_scoring_autoprojection: Optional[bool] = None,
        **kwargs,
    ) -> Iterator[pa.RecordBatch]:
        """Read the dataset as materialized record batches.

        Parameters
        ----------
        **kwargs : dict, optional
            Arguments for :py:meth:`~LanceDataset.scanner`.

        Returns
        -------
        record_batches : Iterator of :py:class:`~pyarrow.RecordBatch`
        """
        return self.scanner(
            columns=columns,
            filter=filter,
            limit=limit,
            offset=offset,
            nearest=nearest,
            batch_size=batch_size,
            batch_size_bytes=batch_size_bytes,
            io_buffer_size=io_buffer_size,
            batch_readahead=batch_readahead,
            fragment_readahead=fragment_readahead,
            late_materialization=late_materialization,
            blob_handling=blob_handling,
            use_scalar_index=use_scalar_index,
            scan_in_order=scan_in_order,
            prefilter=prefilter,
            with_row_id=with_row_id,
            with_row_address=with_row_address,
            use_stats=use_stats,
            full_text_query=full_text_query,
            strict_batch_size=strict_batch_size,
            order_by=order_by,
            disable_scoring_autoprojection=disable_scoring_autoprojection,
        ).to_batches()

    def sample(
        self,
        num_rows: int,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        randomize_order: bool = True,
        **kwargs,
    ) -> pa.Table:
        """Select a random sample of data

        Parameters
        ----------
        num_rows: int
            number of rows to retrieve
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
        **kwargs : dict, optional
            see scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        total_num_rows = self.count_rows()
        indices = random.sample(range(total_num_rows), num_rows)
        if not randomize_order:
            # Sort the indices in order to increase the locality and thus reduce
            # the number of random reads.
            indices = sorted(indices)
        return self.take(indices, columns, **kwargs)

    def take(
        self,
        indices: Union[List[int], pa.Array],
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
    ) -> pa.Table:
        """Select rows of data by index.

        Parameters
        ----------
        indices : Array or array-like
            indices of rows to select in the dataset.
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.

        Returns
        -------
        table : pyarrow.Table
        """
        columns_with_transform = None
        if isinstance(columns, dict):
            columns_with_transform = list(columns.items())
            columns = None
        return pa.Table.from_batches(
            [self._ds.take(indices, columns, columns_with_transform)]
        )

    def _take_rows(
        self,
        row_ids: Union[List[int], pa.Array],
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        **kargs,
    ) -> pa.Table:
        """Select rows by row_ids.

        **Unstable API**. Internal use only

        Parameters
        ----------
        row_ids : List Array or array-like
            row IDs to select in the dataset.
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        columns_with_transform = None
        if isinstance(columns, dict):
            columns_with_transform = list(columns.items())
            columns = None
        return pa.Table.from_batches(
            [self._ds.take_rows(row_ids, columns, columns_with_transform)]
        )

    def take_blobs(
        self,
        blob_column: str,
        ids: Optional[Union[List[int], pa.Array]] = None,
        addresses: Optional[Union[List[int], pa.Array]] = None,
        indices: Optional[Union[List[int], pa.Array]] = None,
    ) -> List[BlobFile]:
        """
        Select blobs by row IDs.

        Instead of loading large binary blob data into memory before processing it,
        this API allows you to open binary blob data as a regular Python file-like
        object. For more details, see :py:class:`lance.BlobFile`.

        Exactly one of ids, addresses, or indices must be specified.

        Parameters
        ----------
        blob_column : str
            The name of the blob column to select.
        ids : Integer Array or array-like
            row IDs to select in the dataset.
        addresses: Integer Array or array-like
            The (unstable) row addresses to select in the dataset.
        indices : Integer Array or array-like
            The offset / indices of the row in the dataset.

        Returns
        -------
        blob_files : List[BlobFile]
        """
        if sum([bool(v is not None) for v in [ids, addresses, indices]]) != 1:
            raise ValueError(
                "Exactly one of ids, indices, or addresses must be specified"
            )

        if ids is not None:
            lance_blob_files = self._ds.take_blobs(ids, blob_column)
        elif addresses is not None:
            lance_blob_files = self._ds.take_blobs_by_addresses(addresses, blob_column)
        elif indices is not None:
            lance_blob_files = self._ds.take_blobs_by_indices(indices, blob_column)
        else:
            raise ValueError("Either ids, addresses, or indices must be specified")
        return [BlobFile(lance_blob_file) for lance_blob_file in lance_blob_files]

    def head(self, num_rows, **kwargs):
        """
        Load the first N rows of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.
        **kwargs : dict, optional
            See scanner() method for full parameter description.

        Returns
        -------
        table : Table
        """
        kwargs["limit"] = num_rows
        return self.scanner(**kwargs).to_table()

    def count_rows(
        self, filter: Optional[Union[str, pa.compute.Expression]] = None, **kwargs
    ) -> int:
        """Count rows matching the scanner filter.

        Parameters
        ----------
        **kwargs : dict, optional
            See py:method:`scanner` method for full parameter description.

        Returns
        -------
        count : int
            The total number of rows in the dataset.

        """
        if isinstance(filter, pa.compute.Expression):
            # TODO: consolidate all to use scanner
            return self.scanner(
                columns=[], with_row_id=True, filter=filter
            ).count_rows()

        return self._ds.count_rows(filter)

    def join(
        self,
        right_dataset,
        keys,
        right_keys=None,
        join_type="left outer",
        left_suffix=None,
        right_suffix=None,
        coalesce_keys=True,
        use_threads=True,
    ):
        """
        Not implemented (just override pyarrow dataset to prevent segfault)
        """
        raise NotImplementedError("Versioning not yet supported in Rust")

    def alter_columns(self, *alterations: Iterable[AlterColumn]):
        """Alter column name, data type, and nullability.

        Columns that are renamed can keep any indices that are on them. If a
        column has an IVF_PQ index, it can be kept if the column is casted to
        another type. However, other index types don't support casting at this
        time.

        Column types can be upcasted (such as int32 to int64) or downcasted
        (such as int64 to int32). However, downcasting will fail if there are
        any values that cannot be represented in the new type. In general,
        columns can be casted to same general type: integers to integers,
        floats to floats, and strings to strings. However, strings, binary, and
        list columns can be casted between their size variants. For example,
        string to large string, binary to large binary, and list to large list.

        Columns that are renamed can keep any indices that are on them. However, if
        the column is casted to a different type, its indices will be dropped.

        Parameters
        ----------
        alterations : Iterable[Dict[str, Any]]
            A sequence of dictionaries, each with the following keys:

            - "path": str
                The column path to alter. For a top-level column, this is the name.
                For a nested column, this is the dot-separated path, e.g. "a.b.c".
            - "name": str, optional
                The new name of the column. If not specified, the column name is
                not changed.
            - "nullable": bool, optional
                Whether the column should be nullable. If not specified, the column
                nullability is not changed. Only non-nullable columns can be changed
                to nullable. Currently, you cannot change a nullable column to
                non-nullable.
            - "data_type": pyarrow.DataType, optional
                The new data type to cast the column to. If not specified, the column
                data type is not changed.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> schema = pa.schema([pa.field('a', pa.int64()),
        ...                     pa.field('b', pa.string(), nullable=False)])
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.alter_columns({"path": "a", "name": "x"},
        ...                       {"path": "b", "nullable": True})
        >>> dataset.to_table().to_pandas()
           x  b
        0  1  a
        1  2  b
        2  3  c
        >>> dataset.alter_columns({"path": "x", "data_type": pa.int32()})
        >>> dataset.schema
        x: int32
        b: string
        """
        self._ds.alter_columns(list(alterations))

    def merge(
        self,
        data_obj: ReaderLike,
        left_on: str,
        right_on: Optional[str] = None,
        schema=None,
    ):
        """
        Merge another dataset into this one.

        Performs a left join, where the dataset is the left side and data_obj
        is the right side. Rows existing in the dataset but not on the left will
        be filled with null values, unless Lance doesn't support null values for
        some types, in which case an error will be raised.

        Parameters
        ----------
        data_obj: Reader-like
            The data to be merged. Acceptable types are:
            - Pandas DataFrame, Pyarrow Table, Dataset, Scanner,
            Iterator[RecordBatch], or RecordBatchReader
        left_on: str
            The name of the column in the dataset to join on.
        right_on: str or None
            The name of the column in data_obj to join on. If None, defaults to
            left_on.

        Examples
        --------

        >>> import lance
        >>> import pyarrow as pa
        >>> df = pa.table({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        >>> dataset = lance.write_dataset(df, "dataset")
        >>> dataset.to_table().to_pandas()
           x  y
        0  1  a
        1  2  b
        2  3  c
        >>> new_df = pa.table({'x': [1, 2, 3], 'z': ['d', 'e', 'f']})
        >>> dataset.merge(new_df, 'x')
        >>> dataset.to_table().to_pandas()
           x  y  z
        0  1  a  d
        1  2  b  e
        2  3  c  f

        See Also
        --------
        LanceDataset.add_columns :
            Add new columns by computing batch-by-batch.
        """
        if right_on is None:
            right_on = left_on

        reader = _coerce_reader(data_obj, schema)

        self._ds.merge(reader, left_on, right_on)

    def add_columns(
        self,
        transforms: (
            Dict[str, str]
            | BatchUDF
            | ReaderLike
            | pyarrow.Field
            | List[pyarrow.Field]
            | pyarrow.Schema
        ),
        read_columns: List[str] | None = None,
        reader_schema: Optional[pa.Schema] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Add new columns with defined values.

        There are several ways to specify the new columns. First, you can provide
        SQL expressions for each new column. Second you can provide a UDF that
        takes a batch of existing data and returns a new batch with the new
        columns. These new columns will be appended to the dataset.

        You can also provide a RecordBatchReader which will read the new column
        values from some external source.  This is often useful when the new column
        values have already been staged to files (often by some distributed process)

        See the :func:`lance.add_columns_udf` decorator for more information on
        writing UDFs.

        Parameters
        ----------
        transforms : dict or AddColumnsUDF or ReaderLike
            If this is a dictionary, then the keys are the names of the new
            columns and the values are SQL expression strings. These strings can
            reference existing columns in the dataset.
            If this is a AddColumnsUDF, then it is a UDF that takes a batch of
            existing data and returns a new batch with the new columns.
            If this is :class:`pyarrow.Field` or :class:`pyarrow.Schema`, it adds
            all NULL columns with the given schema, in a metadata-only operation.
        read_columns : list of str, optional
            The names of the columns that the UDF will read. If None, then the
            UDF will read all columns. This is only used when transforms is a
            UDF. Otherwise, the read columns are inferred from the SQL expressions.

            This can include _rowid or _rowaddr to read the row id or row address
            from the dataset.
        reader_schema: pa.Schema, optional
            Only valid if transforms is a `ReaderLike` object.  This will be used to
            determine the schema of the reader.
        batch_size: int, optional
            The number of rows to read at a time from the source dataset when applying
            the transform.  This is ignored if the dataset is a v1 dataset.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3]})
        >>> dataset = lance.write_dataset(table, "my_dataset")
        >>> @lance.batch_udf()
        ... def double_a(batch):
        ...     df = batch.to_pandas()
        ...     return pd.DataFrame({'double_a': 2 * df['a']})
        >>> dataset.add_columns(double_a)
        >>> dataset.to_table().to_pandas()
           a  double_a
        0  1         2
        1  2         4
        2  3         6
        >>> dataset.add_columns({"triple_a": "a * 3"})
        >>> dataset.to_table().to_pandas()
           a  double_a  triple_a
        0  1         2         3
        1  2         4         6
        2  3         6         9

        See Also
        --------
        LanceDataset.merge :
            Merge a pre-computed set of columns into the dataset.
        """
        if isinstance(transforms, pa.Field):
            transforms = [transforms]
        if (
            isinstance(transforms, list)
            and len(transforms) > 0
            and isinstance(transforms[0], pa.Field)
        ):
            transforms = pa.schema(transforms)

        if isinstance(transforms, pa.Schema):
            self._ds.add_columns_with_schema(transforms)
            return

        transforms = normalize_transform(transforms, self, read_columns, reader_schema)
        if isinstance(transforms, pa.RecordBatchReader):
            self._ds.add_columns_from_reader(transforms, batch_size)
            return
        else:
            self._ds.add_columns(transforms, read_columns, batch_size)

            if isinstance(transforms, BatchUDF):
                if transforms.cache is not None:
                    transforms.cache.cleanup()

    def drop_columns(self, columns: List[str]):
        """Drop one or more columns from the dataset

        This is a metadata-only operation and does not remove the data from the
        underlying storage. In order to remove the data, you must subsequently
        call ``compact_files`` to rewrite the data without the removed columns and
        then call ``cleanup_old_versions`` to remove the old files.

        Parameters
        ----------
        columns : list of str
            The names of the columns to drop. These can be nested column references
            (e.g. "a.b.c") or top-level column names (e.g. "a").

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.drop_columns(["a"])
        >>> dataset.to_table().to_pandas()
           b
        0  a
        1  b
        2  c
        """
        self._ds.drop_columns(columns)
        # Indices might have changed
        self._list_indices_res = None

    def delete(
        self,
        predicate: Union[str, pa.compute.Expression],
        *,
        conflict_retries: int = 10,
        retry_timeout: timedelta = timedelta(seconds=30),
    ) -> DeleteResult:
        """
        Delete rows from the dataset.

        This marks rows as deleted, but does not physically remove them from the
        files. This keeps the existing indexes still valid.

        Parameters
        ----------
        predicate : str or pa.compute.Expression
            The predicate to use to select rows to delete. May either be a SQL
            string or a pyarrow Expression.
        conflict_retries : int, optional
            Number of times to retry the operation if there is contention.
            Default is 10.
        retry_timeout : timedelta, optional
            The timeout used to limit retries. This is the maximum time to spend on
            the operation before giving up. At least one attempt will be made,
            regardless of how long it takes to complete. Subsequent attempts will be
            cancelled once this timeout is reached. Default is 30 seconds.

        Returns
        -------
        dict
            A dictionary containing the number of rows deleted, with the key
            ``num_deleted_rows``.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.delete("a = 1 or b in ('a', 'b')")
        {'num_deleted_rows': 2}
        """
        if isinstance(predicate, pa.compute.Expression):
            predicate = str(predicate)
        return self._ds.delete(predicate, conflict_retries, retry_timeout)

    def truncate_table(self) -> None:
        """
        Truncate the dataset by deleting all rows.
        The schema is preserved and a new version is created.
        """
        self._ds.truncate_table()
        self._list_indices_res = None

    def insert(
        self,
        data: ReaderLike,
        *,
        mode="append",
        **kwargs,
    ):
        """
        Insert data into the dataset.

        Parameters
        ----------
        data_obj: Reader-like
            The data to be written. Acceptable types are:
            - Pandas DataFrame, Pyarrow Table, Dataset, Scanner, or RecordBatchReader
            - Huggingface dataset
        mode: str, default 'append'
            The mode to use when writing the data. Options are:
                **create** - create a new dataset (raises if uri already exists).
                **overwrite** - create a new snapshot version
                **append** - create a new version that is the concat of the input the
                latest version (raises if uri does not exist)
        **kwargs : dict, optional
            Additional keyword arguments to pass to :func:`write_dataset`.
        """
        new_ds = write_dataset(data, self, mode=mode, **kwargs)
        self._ds = new_ds._ds

    def merge_insert(
        self,
        on: Optional[Union[str, Iterable[str]]] = None,
    ) -> MergeInsertBuilder:
        """
        Returns a builder that can be used to create a "merge insert" operation

        This operation can add rows, update rows, and remove rows in a single
        transaction. It is a very generic tool that can be used to create
        behaviors like "insert if not exists", "update or insert (i.e. upsert)",
        or even replace a portion of existing data with new data (e.g. replace
        all data where month="january")

        The merge insert operation works by combining new data from a
        **source table** with existing data in a **target table** by using a
        join.  There are three categories of records.

        "Matched" records are records that exist in both the source table and
        the target table. "Not matched" records exist only in the source table
        (e.g. these are new data). "Not matched by source" records exist only
        in the target table (this is old data).

        The builder returned by this method can be used to customize what
        should happen for each category of data.

        Please note that the data will be reordered as part of this
        operation.  This is because updated rows will be deleted from the
        dataset and then reinserted at the end with the new values.  The
        order of the newly inserted rows may fluctuate randomly because a
        hash-join operation is used internally.

        Parameters
        ----------

        on: Optional[Union[str, Iterable[str]]], default None
            A column (or columns) to join on.  This is how records from the
            source table and target table are matched.  Typically this is some
            kind of key or id column.

            If ``on`` is not provided (or is ``None``), the merge insert
            operation will use the dataset's unenforced primary key as defined
            in the schema metadata. If no primary key is configured and
            ``on`` is None, a :class:`ValueError` will be raised.

        Examples
        --------

        Use `when_matched_update_all()` and `when_not_matched_insert_all()` to
        perform an "upsert" operation.  This will update rows that already exist
        in the dataset and insert rows that do not exist.

        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [2, 1, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> new_table = pa.table({"a": [2, 3, 4], "b": ["x", "y", "z"]})
        >>> # Perform a "upsert" operation
        >>> dataset.merge_insert("a")     \\
        ...             .when_matched_update_all()     \\
        ...             .when_not_matched_insert_all() \\
        ...             .execute(new_table)
        {'num_inserted_rows': 1, 'num_updated_rows': 2, 'num_deleted_rows': 0}
        >>> dataset.to_table().sort_by("a").to_pandas()
           a  b
        0  1  b
        1  2  x
        2  3  y
        3  4  z

        Use `when_not_matched_insert_all()` to perform an "insert if not exists"
        operation.  This will only insert rows that do not already exist in the
        dataset.

        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example2")
        >>> new_table = pa.table({"a": [2, 3, 4], "b": ["x", "y", "z"]})
        >>> # Perform an "insert if not exists" operation
        >>> dataset.merge_insert("a")     \\
        ...             .when_not_matched_insert_all() \\
        ...             .execute(new_table)
        {'num_inserted_rows': 1, 'num_updated_rows': 0, 'num_deleted_rows': 0}
        >>> dataset.to_table().sort_by("a").to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  z

        You are not required to provide all the columns. If you only want to
        update a subset of columns, you can omit columns you don't want to
        update. Omitted columns will keep their existing values if they are
        updated, or will be null if they are inserted.

        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"], \\
        ...                   "c": ["x", "y", "z"]})
        >>> dataset = lance.write_dataset(table, "example3")
        >>> new_table = pa.table({"a": [2, 3, 4], "b": ["x", "y", "z"]})
        >>> # Perform an "upsert" operation, only updating column "a"
        >>> dataset.merge_insert("a")     \\
        ...             .when_matched_update_all()     \\
        ...             .when_not_matched_insert_all() \\
        ...             .execute(new_table)
        {'num_inserted_rows': 1, 'num_updated_rows': 2, 'num_deleted_rows': 0}
        >>> dataset.to_table().sort_by("a").to_pandas()
           a  b    c
        0  1  a    x
        1  2  x    y
        2  3  y    z
        3  4  z  NaN
        """
        return MergeInsertBuilder(self._ds, on)

    def update(
        self,
        updates: Dict[str, str],
        where: Optional[str] = None,
        conflict_retries: int = 10,
        retry_timeout: timedelta = timedelta(seconds=30),
    ) -> UpdateResult:
        """
        Update column values for rows matching where.

        Parameters
        ----------
        updates : dict of str to str
            A mapping of column names to a SQL expression.
        where : str, optional
            A SQL predicate indicating which rows should be updated.
        conflict_retries : int, optional
            Number of times to retry the operation if there is contention.
            Default is 10.
        retry_timeout : timedelta, optional
            The timeout used to limit retries. This is the maximum time to spend on
            the operation before giving up. At least one attempt will be made,
            regardless of how long it takes to complete. Subsequent attempts will be
            cancelled once this timeout is reached. Default is 30 seconds.

        Returns
        -------
        updates : dict
            A dictionary containing the number of rows updated.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2, 3], "b": ["a", "b", "c"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> update_stats = dataset.update(dict(a = 'a + 2'), where="b != 'a'")
        >>> update_stats["num_updated_rows"] = 2
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  4  b
        2  5  c
        """
        if isinstance(where, pa.compute.Expression):
            where = str(where)
        return self._ds.update(updates, where, conflict_retries, retry_timeout)

    def versions(self):
        """
        Return all versions in this dataset.
        """
        versions = self._ds.versions()
        for v in versions:
            # TODO: python datetime supports only microsecond precision. When a
            # separate Version object is implemented, expose the precise timestamp
            # (ns) to python.
            ts_nanos = v["timestamp"]
            v["timestamp"] = datetime.fromtimestamp(ts_nanos // 1e9) + timedelta(
                microseconds=(ts_nanos % 1e9) // 1e3
            )
        return versions

    @property
    def version(self) -> int:
        """
        Returns the currently checked out version of the dataset
        """
        return self._ds.version()

    @property
    def latest_version(self) -> int:
        """
        Returns the latest version of the dataset.
        """
        return self._ds.latest_version()

    @property
    def initial_storage_options(self) -> Optional[Dict[str, str]]:
        """
        Get the initial storage options used to open this dataset.

        This returns the options that were provided when the dataset was opened,
        without any refresh from the provider. Returns None if no storage options
        were provided.
        """
        return self._ds.initial_storage_options()

    def latest_storage_options(self) -> Optional[Dict[str, str]]:
        """
        Get the latest storage options, potentially refreshed from the provider.

        If a storage options provider was configured and credentials are expiring,
        this will refresh them.

        Returns
        -------
        Optional[Dict[str, str]]
            - Storage options dict if configured (static or refreshed from provider)
            - None if no storage options were configured for this dataset

        Raises
        ------
        IOError
            If an error occurs while fetching/refreshing options from the provider
        """
        return self._ds.latest_storage_options()

    @property
    def storage_options_accessor(self):
        """
        Get the storage options accessor for this dataset.

        The accessor bundles static storage options and optional dynamic provider,
        handling caching and refresh logic internally.

        Returns None if neither storage options nor a provider were configured.
        """
        return self._ds.storage_options_accessor()

    def new_file_session(self):
        """
        Create a new file session for reading and writing files in this dataset.

        The file session will use the dataset's storage options and provider
        for credential management, enabling automatic credential refresh for
        long-running operations.

        Returns
        -------
        LanceFileSession
            A file session configured for this dataset's storage location.
        """
        from lance.file import LanceFileSession

        return LanceFileSession(
            base_path=self._uri,
            storage_options=self.latest_storage_options(),
            namespace_client=self._namespace_client,
            table_id=self._table_id,
        )

    def checkout_version(
        self, version: int | str | Tuple[Optional[str], Optional[int]]
    ) -> "LanceDataset":
        """
        Load the given version of the dataset.

        Unlike the :func:`dataset` constructor, this will re-use the
        current cache.
        This is a no-op if the dataset is already at the given version.

        Parameters
        ----------
        version: int | str | Tuple[Optional[str], Optional[int]],
            An integer specifies a version number in the current branch; a string
            specifies a tag name; a Tuple[Optional[str], Optional[int]] specifies
            a version number in a specified branch. (None, None) means the latest
            version_number on the main branch.

        Returns
        -------
        LanceDataset
        """
        ds = copy.copy(self)
        if version != ds.version:
            ds._ds = self._ds.checkout_version(version)
        return ds

    def restore(self):
        """
        Restore the currently checked out version as the latest version of the dataset.

        This creates a new commit.
        """
        self._ds.restore()

    def add_bases(
        self, new_bases: list, transaction_properties: Optional[Dict[str, str]] = None
    ):
        """
        Add new base paths to the dataset for multi-base storage.

        This method allows you to register additional storage locations (bases)
        that can be used for future data writes. The base paths are added to the
        dataset's manifest and can be referenced by name in subsequent write operations.

        Parameters
        ----------
        new_bases : list[DatasetBasePath]
            A list of DatasetBasePath objects representing the new storage locations
            to add. Each base path should have a unique name and path.
        transaction_properties : Optional[Dict[str, str]]
            Optional key-value pairs for audit metadata.

        Returns
        -------
        LanceDataset
            Returns self to allow method chaining.
        """
        self._ds.add_bases(new_bases, transaction_properties)
        return self

    def cleanup_old_versions(
        self,
        older_than: Optional[timedelta] = None,
        retain_versions: Optional[int] = None,
        *,
        delete_unverified: bool = False,
        error_if_tagged_old_versions: bool = True,
        delete_rate_limit: Optional[int] = None,
    ) -> CleanupStats:
        """
        Cleans up old versions of the dataset.

        Some dataset changes, such as overwriting, leave behind data that is not
        referenced by the latest dataset version.  The old data is left in place
        to allow the dataset to be restored back to an older version.

        This method will remove older versions and any data files they reference.
        Once this cleanup task has run you will not be able to checkout or restore
        these older versions.

        Parameters
        ----------

        older_than: timedelta, optional
            Only versions older than this will be removed.  If ``older_than`` and
            ``retain_versions`` are not specified, this will default to two weeks.

        retain_versions: int, optional
            Retain the last N versions of the dataset.

        delete_unverified: bool, default False
            Files leftover from a failed transaction may appear to be part of an
            in-progress operation (e.g. appending new data) and these files will
            not be deleted unless they are at least 7 days old.  If delete_unverified
            is True then these files will be deleted regardless of their age.

            This should only be set to True if you can guarantee that no other process
            is currently working on this dataset.  Otherwise the dataset could be put
            into a corrupted state.

        error_if_tagged_old_versions: bool, default True
            Some versions may have tags associated with them. Tagged versions will
            not be cleaned up, regardless of how old they are. If this argument
            is set to `True` (the default), an exception will be raised if any
            tagged versions match the parameters. Otherwise, tagged versions will
            be ignored without any error and only untagged versions will be
            cleaned up.

        delete_rate_limit: int, optional
            Maximum number of delete operations per second. When not set (default),
            deletions run at full speed. Set this to a positive integer to avoid
            hitting object store request rate limits (e.g. S3 HTTP 503 SlowDown).
            For example, ``delete_rate_limit=100`` limits to 100 operations/second.
        """
        if older_than is None and retain_versions is None:
            older_than = timedelta(days=14)

        return self._ds.cleanup_old_versions(
            td_to_micros(older_than) if older_than else None,
            retain_versions,
            delete_unverified,
            error_if_tagged_old_versions,
            delete_rate_limit,
        )

    def create_scalar_index(
        self,
        column: str,
        index_type: Union[
            Literal["BTREE"],
            Literal["BITMAP"],
            Literal["LABEL_LIST"],
            Literal["INVERTED"],
            Literal["FTS"],
            Literal["NGRAM"],
            Literal["ZONEMAP"],
            Literal["BLOOMFILTER"],
            Literal["RTREE"],
            IndexConfig,
        ],
        name: Optional[str] = None,
        *,
        replace: bool = True,
        train: bool = True,
        fragment_ids: Optional[List[int]] = None,
        index_uuid: Optional[str] = None,
        progress_callback: Optional[Callable[[IndexProgress], None]] = None,
        **kwargs,
    ):
        """Create a scalar index on a column.

        Scalar indices, like vector indices, can be used to speed up scans.  A scalar
        index can speed up scans that contain filter expressions on the indexed column.
        For example, the following scan will be faster if the column ``my_col`` has
        a scalar index:

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/images.lance")
            my_table = dataset.scanner(filter="my_col != 7").to_table()

        Vector search with pre-filers can also benefit from scalar indices. For example,

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/images.lance")
            my_table = dataset.scanner(
                nearest=dict(
                   column="vector",
                   q=[1, 2, 3, 4],
                   k=10,
                )
                filter="my_col != 7",
                prefilter=True
            )


        There are 5 types of scalar indices available today.

        * ``BTREE``. The most common type is ``BTREE``. This index is inspired
          by the btree data structure although only the first few layers of the btree
          are cached in memory.  It will
          perform well on columns with a large number of unique values and few rows per
          value.
        * ``BITMAP``. This index stores a bitmap for each unique value in the column.
          This index is useful for columns with a small number of unique values and
          many rows per value.
        * ``LABEL_LIST``. A special index that is used to index list
          columns whose values have small cardinality.  For example, a column that
          contains lists of tags (e.g. ``["tag1", "tag2", "tag3"]``) can be indexed
          with a ``LABEL_LIST`` index. This index can speed up list membership
          filters such as ``array_has_any``, ``array_has_all``, and
          ``array_has`` / ``array_contains``.
        * ``NGRAM``. A special index that is used to index string columns. This index
          creates a bitmap for each ngram in the string.  By default we use trigrams.
          This index can currently speed up queries using the ``contains`` function
          in filters.
        * ``ZONEMAP``. This inexact index breaks the column into fixed-size chunks
          called zones and stores summary statistics for each zone (min, max,
          null_count, nan_count, fragment_id, local_row_offset). It's very small but
          only effective if the column is at least approximately in sorted order.
        * ``INVERTED`` (alias: ``FTS``). It is used to index document columns. This
          index can conduct full-text searches. For example, a column that contains any
          word
          of query string "hello world". The results will be ranked by BM25.
        * ``BLOOMFILTER``. This inexact index uses a bloom filter.  It is small
             but can only handle filters with equals and not equals and may require
             more I/O than a btree or bitmap index```

        Note that the ``LANCE_BYPASS_SPILLING`` environment variable can be used to
        bypass spilling to disk. Setting this to true can avoid memory exhaustion
        issues (see https://github.com/apache/datafusion/issues/10073 for more info).

        **Experimental API**

        Parameters
        ----------
        column : str
            The column to be indexed.  Must be a boolean, integer, float,
            or string column.
        index_type : str
            The type of the index.  One of ``"BTREE"``, ``"BITMAP"``,
            ``"LABEL_LIST"``, ``"NGRAM"``, ``"ZONEMAP"``, ``"INVERTED"``,
            ``"FTS"``, ``"BLOOMFILTER"``, ``"RTREE"``.
        name : str, optional
            The index name. If not provided, it will be generated from the
            column name.
        replace : bool, default True
            Replace the existing index if it exists.
        train : bool, default True
            If True, the index will be trained on the data to determine optimal
            structure. If False, an empty index will be created that can be
            populated later.
        fragment_ids : List[int], optional
            If provided, the index will be created only on the specified fragments.
            This enables distributed/fragment-level indexing. When provided, the
            method returns metadata for one segment but does not commit
            the index to the dataset. The segment can be planned, merged, and
            committed later using the segment builder and commit APIs.
            This parameter is passed via kwargs internally.
        index_uuid : str, optional
            A UUID to use for the segment written by this call.
            If not provided, a new UUID will be generated. This parameter is
            passed via kwargs internally.
        progress_callback : callable, optional
            A callback that receives :class:`lance.progress.IndexProgress` events while
            the index is being built.

        with_position: bool, default False
            This is for the ``INVERTED`` index. If True, the index will store the
            positions of the words in the document, so that you can conduct phrase
            query. This will significantly increase the index size.
            It won't impact the performance of non-phrase queries even if it is set to
            True.
        memory_limit: int, optional
            This is for the ``INVERTED`` index. Total build-time memory limit in MiB.
            If set, Lance divides this budget evenly across the workers. If unset,
            the default will be 2 GiB per worker. This parameter is only used for the
            current build and is not persisted with the index.

            A larger memory limit will create an index with fewer shards which will
            be easier to search so this is a trade-off between build resources and
            search cost.
        num_workers: int, optional
            This is for the ``INVERTED`` index. Number of workers to use for
            the current build. The effective worker count is clamped to
            ``[1, num_compute_cpus]``. If unset, Lance uses ``num_compute_cpus``
            workers unless ``LANCE_FTS_NUM_SHARDS`` is set. This parameter is
            only used for the current build and is not persisted with the index.
        base_tokenizer: str, default "simple"
            This is for the ``INVERTED`` index. The base tokenizer to use. The
            value can be:
            * "simple": splits tokens on whitespace and punctuation.
            * "whitespace": splits tokens on whitespace.
            * "raw": no tokenization.
        language: str, default "English"
            This is for the ``INVERTED`` index. The language for stemming
            and stop words. This is only used when `stem` or `remove_stop_words` is true
        max_token_length: Optional[int], default 40
            This is for the ``INVERTED`` index. The maximum token length.
            Any token longer than this will be removed.
        lower_case: bool, default True
            This is for the ``INVERTED`` index. If True, the index will convert all
            text to lowercase.
        stem: bool, default True
            This is for the ``INVERTED`` index. If True, the index will stem the
            tokens.
        remove_stop_words: bool, default True
            This is for the ``INVERTED`` index. If True, the index will remove
            stop words.
        custom_stop_words: Optional[List[str]], default None
            This is for the ``INVERTED`` index, only used when `remove_stop_words` is
            true.
            Users can specify their customised stop words, if none, the built-in stop
            words for the specified language will be used.
        ascii_folding: bool, default True
            This is for the ``INVERTED`` index. If True, the index will convert
            non-ascii characters to ascii characters if possible.
            This would remove accents like "é" -> "e".

        Examples
        --------

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/images.lance")
            dataset.create_index(
                "category",
                "BTREE",
            )

        Scalar indices can only speed up scans for basic filters using
        equality, comparison, range (e.g. ``my_col BETWEEN 0 AND 100``), and set
        membership (e.g. `my_col IN (0, 1, 2)`)

        Scalar indices can be used if the filter contains multiple indexed columns and
        the filter criteria are AND'd or OR'd together
        (e.g. ``my_col < 0 AND other_col> 100``)

        Scalar indices may be used if the filter contains non-indexed columns but,
        depending on the structure of the filter, they may not be usable.  For example,
        if the column ``not_indexed`` does not have a scalar index then the filter
        ``my_col = 0 OR not_indexed = 1`` will not be able to use any scalar index on
        ``my_col``.

        To determine if a scan is making use of a scalar index you can use
        ``explain_plan`` to look at the query plan that lance has created.  Queries
        that use scalar indices will either have a ``ScalarIndexQuery`` relation or a
        ``MaterializeIndex`` operator.

        """
        if isinstance(column, str):
            column = [column]

        if len(column) > 1:
            raise NotImplementedError(
                "Scalar indices currently only support a single column"
            )

        column = column[0]
        lance_field = self._ds.lance_schema.field_case_insensitive(column)
        if lance_field is None:
            raise KeyError(f"{column} not found in schema")

        # TODO: Add documentation of IndexConfig approach for creating
        # indexes that need parameterization
        if isinstance(index_type, str):
            index_type = index_type.upper()
            if index_type not in [
                "BTREE",
                "BITMAP",
                "NGRAM",
                "ZONEMAP",
                "LABEL_LIST",
                "INVERTED",
                "FTS",
                "BLOOMFILTER",
                "RTREE",
            ]:
                raise NotImplementedError(
                    (
                        'Only "BTREE", "BITMAP", "NGRAM", "ZONEMAP", "LABEL_LIST", '
                        '"INVERTED", "BLOOMFILTER" or "RTREE" are supported for '
                        f"scalar columns.  Received {index_type}",
                    )
                )

            field = lance_field.to_arrow()

            field_type = field.type
            field_meta = field.metadata
            if hasattr(field_type, "storage_type"):
                field_type = field_type.storage_type

            if index_type in ["BTREE", "BITMAP", "ZONEMAP"]:
                if (
                    not pa.types.is_integer(field_type)
                    and not pa.types.is_floating(field_type)
                    and not pa.types.is_boolean(field_type)
                    and not pa.types.is_string(field_type)
                    and not pa.types.is_temporal(field_type)
                    and not pa.types.is_fixed_size_binary(field_type)
                ):
                    raise TypeError(
                        f"BTREE/BITMAP index column {column} must be int",
                        ", float, bool, str, fixed-size-binary, or temporal ",
                    )
            elif index_type == "LABEL_LIST":
                if not pa.types.is_list(field_type):
                    raise TypeError(f"LABEL_LIST index column {column} must be a list")
            elif index_type == "NGRAM":
                if not pa.types.is_string(field_type) and not pa.types.is_large_string(
                    field_type
                ):
                    raise TypeError(f"NGRAM index column {column} must be a string")
            elif index_type in ["INVERTED", "FTS"]:
                value_type = field_type
                if pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
                    value_type = field_type.value_type
                if (
                    not pa.types.is_string(value_type)
                    and not pa.types.is_large_string(value_type)
                    and not (
                        pa.types.is_large_binary(value_type)
                        and field_meta[b"ARROW:extension:name"] == b"lance.json"
                    )
                ):
                    raise TypeError(
                        f"INVERTED index column {column} must be string, large string"
                        f" or list of strings, or json, but got {value_type}"
                    )

            if pa.types.is_duration(field_type):
                raise TypeError(
                    f"Scalar index column {column} cannot currently be a duration"
                )
        elif isinstance(index_type, IndexConfig):
            config = json.dumps(index_type.parameters)
            kwargs["config"] = indices.IndexConfig(index_type.index_type, config)
            index_type = "scalar"
        else:
            raise Exception("index_type must be str or IndexConfig")

        # Add fragment_ids and index_uuid to kwargs if provided
        if fragment_ids is not None:
            kwargs["fragment_ids"] = fragment_ids
        if index_uuid is not None:
            kwargs["index_uuid"] = index_uuid
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback

        self._ds.create_index([column], index_type, name, replace, train, None, kwargs)

    def _create_index_impl(
        self,
        column: Union[str, List[str]],
        index_type: str,
        name: Optional[str] = None,
        metric: str = "L2",
        replace: bool = False,
        num_partitions: Optional[int] = None,
        ivf_centroids: Optional[
            Union[np.ndarray, pa.FixedSizeListArray, pa.FixedShapeTensorArray]
        ] = None,
        pq_codebook: Optional[
            Union[np.ndarray, pa.FixedSizeListArray, pa.FixedShapeTensorArray]
        ] = None,
        num_sub_vectors: Optional[int] = None,
        accelerator: Optional[Union[str, "torch.Device"]] = None,
        index_cache_size: Optional[int] = None,
        shuffle_partition_batches: Optional[int] = None,
        shuffle_partition_concurrency: Optional[int] = None,
        ivf_centroids_file: Optional[str] = None,
        precomputed_partition_dataset: Optional[str] = None,
        storage_options: Optional[Dict[str, str]] = None,
        filter_nan: bool = True,
        train: bool = True,
        fragment_ids: Optional[List[int]] = None,
        index_uuid: Optional[str] = None,
        *,
        target_partition_size: Optional[int] = None,
        skip_transpose: bool = False,
        require_commit: bool = True,
        **kwargs,
    ) -> Index:
        if not require_commit and fragment_ids is None:
            raise ValueError(
                "create_index_uncommitted requires fragment_ids "
                "for distributed index build"
            )

        # Only support building index for 1 column from the API aspect, however
        # the internal implementation might support building multi-column index later.
        if isinstance(column, str):
            column = [column]

        # validate args
        for c in column:
            lance_field = self._ds.lance_schema.field_case_insensitive(c)
            if lance_field is None:
                raise KeyError(f"{c} not found in schema")
            field = lance_field.to_arrow()
            is_multivec = False
            if pa.types.is_fixed_size_list(field.type):
                dimension = field.type.list_size
            elif pa.types.is_list(field.type) and pa.types.is_fixed_size_list(
                field.type.value_type
            ):
                dimension = field.type.value_type.list_size
                is_multivec = True
            elif (
                isinstance(field.type, pa.FixedShapeTensorType)
                and len(field.type.shape) == 1
            ):
                dimension = field.type.shape[0]
            else:
                raise TypeError(
                    f"Vector column {c} must be FixedSizeListArray "
                    f"1-dimensional FixedShapeTensorArray, got {field.type}"
                )

            if num_sub_vectors is not None and dimension % num_sub_vectors != 0:
                raise ValueError(
                    f"dimension ({dimension}) must be divisible by num_sub_vectors"
                    f" ({num_sub_vectors})"
                )

            element_type = field.type.value_type
            if is_multivec:
                element_type = field.type.value_type.value_type
            if not (
                pa.types.is_floating(element_type) or pa.types.is_uint8(element_type)
            ):
                raise TypeError(
                    f"Vector column {c} must have floating value type, "
                    f"got {field.type.value_type}"
                )

        if not isinstance(metric, str) or metric.lower() not in [
            "l2",
            "cosine",
            "euclidean",
            "dot",
            "hamming",
        ]:
            raise ValueError(f"Metric {metric} not supported.")

        kwargs["metric_type"] = metric

        index_type = index_type.upper()
        valid_index_types = [
            "IVF_FLAT",
            "IVF_PQ",
            "IVF_SQ",
            "IVF_HNSW_FLAT",
            "IVF_HNSW_PQ",
            "IVF_HNSW_SQ",
            "IVF_RQ",
        ]
        if index_type not in valid_index_types:
            raise NotImplementedError(
                f"Only {valid_index_types} index types supported. Got {index_type}"
            )

        # Handle timing for various parts of accelerated builds
        timers = {}
        use_cuvs = is_cuvs_accelerator(accelerator)
        if accelerator is not None and index_type != "IVF_PQ":
            LOGGER.warning(
                "Index type %s does not support GPU acceleration; falling back to CPU",
                index_type,
            )
            accelerator = None
            use_cuvs = False

        # IMPORTANT: Distributed indexing is CPU-only. Enforce single-node when
        # accelerator or torch-related paths are detected.
        torch_detected = False
        try:
            if accelerator is not None:
                torch_detected = True
            else:
                impl = kwargs.get("implementation")
                use_torch_flag = kwargs.get("use_torch") is True
                one_pass_flag = kwargs.get("one_pass_ivfpq") is True
                torch_centroids = _check_for_torch(ivf_centroids)
                torch_codebook = _check_for_torch(pq_codebook)
                if (
                    (isinstance(impl, str) and impl.lower() == "torch")
                    or use_torch_flag
                    or one_pass_flag
                    or torch_centroids
                    or torch_codebook
                ):
                    torch_detected = True
        except Exception:
            # Be conservative: if detection fails, do not modify behavior
            pass

        if torch_detected:
            if require_commit:
                if fragment_ids is not None or index_uuid is not None:
                    LOGGER.info(
                        "Torch detected; "
                        "enforce single-node indexing (distributed is CPU-only)."
                    )
                fragment_ids = None
                index_uuid = None
            else:
                if index_uuid is not None:
                    LOGGER.info(
                        "Torch detected; "
                        "enforce single-node indexing (distributed is CPU-only)."
                    )
                index_uuid = None

        if accelerator is not None:
            if num_partitions is None:
                num_rows = self.count_rows()
                num_partitions = _target_partition_size_to_num_partitions(
                    num_rows, target_partition_size
                )
            if use_cuvs:
                from .cuvs import build_vector_index_on_cuvs

                LOGGER.info("Doing cuVS vector backend build")
                timers["ivf+pq_build:start"] = time.time()
                artifact_root, _, ivf_centroids, pq_codebook = (
                    build_vector_index_on_cuvs(
                        self,
                        column[0],
                        metric,
                        accelerator,
                        num_partitions,
                        num_sub_vectors,
                        storage_options=storage_options,
                        sample_rate=kwargs.get("sample_rate", 256),
                        max_iters=kwargs.get("max_iters", 50),
                        num_bits=kwargs.get("num_bits", 8),
                        batch_size=1024 * 128,
                        filter_nan=filter_nan,
                    )
                )
                kwargs["precomputed_partition_artifact_uri"] = artifact_root
                timers["ivf+pq_build:end"] = time.time()
                ivfpq_build_time = (
                    timers["ivf+pq_build:end"] - timers["ivf+pq_build:start"]
                )
                LOGGER.info("cuVS ivf+pq build time: %ss", ivfpq_build_time)
            else:
                from .vector import (
                    one_pass_assign_ivf_pq_on_accelerator,
                    one_pass_train_ivf_pq_on_accelerator,
                )

                LOGGER.info("Doing one-pass ivfpq accelerated computations")
                timers["ivf+pq_train:start"] = time.time()
                (
                    ivf_centroids,
                    ivf_kmeans,
                    pq_codebook,
                    pq_kmeans_list,
                ) = one_pass_train_ivf_pq_on_accelerator(
                    self,
                    column[0],
                    num_partitions,
                    metric,
                    accelerator,
                    num_sub_vectors=num_sub_vectors,
                    batch_size=20480,
                    filter_nan=filter_nan,
                )
                timers["ivf+pq_train:end"] = time.time()
                ivfpq_train_time = (
                    timers["ivf+pq_train:end"] - timers["ivf+pq_train:start"]
                )
                LOGGER.info("ivf+pq training time: %ss", ivfpq_train_time)
                timers["ivf+pq_assign:start"] = time.time()
                shuffle_output_dir, shuffle_buffers = (
                    one_pass_assign_ivf_pq_on_accelerator(
                        self,
                        column[0],
                        metric,
                        accelerator,
                        ivf_kmeans,
                        pq_kmeans_list,
                        batch_size=20480,
                        filter_nan=filter_nan,
                    )
                )
                timers["ivf+pq_assign:end"] = time.time()
                ivfpq_assign_time = (
                    timers["ivf+pq_assign:end"] - timers["ivf+pq_assign:start"]
                )
                LOGGER.info("ivf+pq transform time: %ss", ivfpq_assign_time)

                kwargs["precomputed_shuffle_buffers"] = shuffle_buffers
                kwargs["precomputed_shuffle_buffers_path"] = os.path.join(
                    shuffle_output_dir, "data"
                )
        if index_type.startswith("IVF"):
            if (ivf_centroids is not None) and (ivf_centroids_file is not None):
                raise ValueError(
                    "ivf_centroids and ivf_centroids_file"
                    " cannot be provided at the same time"
                )

            if ivf_centroids_file is not None:
                from pyarrow.fs import FileSystem

                fs, path = FileSystem.from_uri(ivf_centroids_file)
                with fs.open_input_file(path) as f:
                    ivf_centroids = np.load(f)
                num_partitions = ivf_centroids.shape[0]

            if isinstance(num_partitions, float):
                warnings.warn("num_partitions is float, converting to int")
                num_partitions = int(num_partitions)
            elif num_partitions is not None and not isinstance(num_partitions, int):
                raise TypeError(
                    f"num_partitions must be int, got {type(num_partitions)}"
                )
            if num_partitions is not None:
                kwargs["num_partitions"] = num_partitions
            if target_partition_size is not None:
                kwargs["target_partition_size"] = target_partition_size

            if (precomputed_partition_dataset is not None) and (ivf_centroids is None):
                raise ValueError(
                    "ivf_centroids must be provided when"
                    " precomputed_partition_dataset is provided"
                )
            if precomputed_partition_dataset is not None:
                LOGGER.info("Using provided precomputed partition dataset")
                precomputed_ds = LanceDataset(
                    precomputed_partition_dataset, storage_options=storage_options
                )
                if not (
                    "PQ" in index_type
                    and pq_codebook is None
                    and accelerator is not None
                    and "precomputed_partitions_file" in kwargs
                ):
                    # In this case, the precomputed partitions file would be used
                    # without being turned into a set of precomputed buffers, so it
                    # needs to have a very specific format
                    if len(precomputed_ds.get_fragments()) != 1:
                        raise ValueError(
                            "precomputed_partition_dataset must have only one fragment"
                        )
                    files = precomputed_ds.get_fragments()[0].data_files()
                    if len(files) != 1:
                        raise ValueError(
                            "precomputed_partition_dataset must have only one files"
                        )
                kwargs["precomputed_partitions_file"] = precomputed_partition_dataset

            if (ivf_centroids is None) and (pq_codebook is not None):
                raise ValueError(
                    "ivf_centroids must be specified when pq_codebook is provided"
                )

            if ivf_centroids is not None:
                # User provided IVF centroids
                if _check_for_numpy(ivf_centroids) and isinstance(
                    ivf_centroids, np.ndarray
                ):
                    if (
                        len(ivf_centroids.shape) != 2
                        or ivf_centroids.shape[0] != num_partitions
                    ):
                        raise ValueError(
                            f"Ivf centroids must be 2D array: (clusters, dim), "
                            f"got {ivf_centroids.shape}"
                        )
                    if ivf_centroids.dtype not in [np.float16, np.float32, np.float64]:
                        raise TypeError(
                            "IVF centroids must be floating number"
                            + f"got {ivf_centroids.dtype}"
                        )
                    dim = ivf_centroids.shape[1]
                    values = pa.array(ivf_centroids.reshape(-1))
                    ivf_centroids = pa.FixedSizeListArray.from_arrays(values, dim)
                kwargs["ivf_centroids"] = pa.RecordBatch.from_arrays(
                    [ivf_centroids], ["_ivf_centroids"]
                )

        if "PQ" in index_type:
            if num_sub_vectors is None:
                raise ValueError(
                    "num_partitions and num_sub_vectors are required for IVF_PQ"
                )
            kwargs["num_sub_vectors"] = num_sub_vectors

            # Always attach PQ codebook if provided (global training invariant)
            if pq_codebook is not None:
                # User provided PQ codebook
                if _check_for_numpy(pq_codebook) and isinstance(
                    pq_codebook, np.ndarray
                ):
                    if (
                        len(pq_codebook.shape) != 3
                        or pq_codebook.shape[0] != num_sub_vectors
                        or pq_codebook.shape[1] != 256
                    ):
                        raise ValueError(
                            f"PQ codebook must be 3D array: (sub_vectors, 256, dim), "
                            f"got {pq_codebook.shape}"
                        )
                    if pq_codebook.dtype not in [np.float16, np.float32, np.float64]:
                        raise TypeError(
                            "PQ codebook must be floating number"
                            + f"got {pq_codebook.dtype}"
                        )
                    values = pa.array(pq_codebook.reshape(-1))
                    pq_codebook = pa.FixedSizeListArray.from_arrays(
                        values, pq_codebook.shape[2]
                    )
                pq_codebook_batch = pa.RecordBatch.from_arrays(
                    [pq_codebook], ["_pq_codebook"]
                )
                kwargs["pq_codebook"] = pq_codebook_batch

        if shuffle_partition_batches is not None:
            kwargs["shuffle_partition_batches"] = shuffle_partition_batches
        if shuffle_partition_concurrency is not None:
            kwargs["shuffle_partition_concurrency"] = shuffle_partition_concurrency

        if skip_transpose:
            kwargs["skip_transpose"] = True

        # Add fragment_ids and index_uuid to kwargs if provided for
        # distributed indexing
        if fragment_ids is not None:
            kwargs["fragment_ids"] = fragment_ids
        if index_uuid is not None:
            kwargs["index_uuid"] = index_uuid

        timers["final_create_index:start"] = time.time()
        index = self._ds.create_index(
            column, index_type, name, replace, train, storage_options, kwargs
        )
        timers["final_create_index:end"] = time.time()
        final_create_index_time = (
            timers["final_create_index:end"] - timers["final_create_index:start"]
        )
        LOGGER.info("Final create_index rust time: %ss", final_create_index_time)
        # Save disk space
        if "precomputed_shuffle_buffers_path" in kwargs.keys() and os.path.exists(
            kwargs["precomputed_shuffle_buffers_path"]
        ):
            LOGGER.info(
                "Temporary shuffle buffers stored at %s, you may want to delete it.",
                kwargs["precomputed_shuffle_buffers_path"],
            )
        if "precomputed_partition_artifact_uri" in kwargs.keys() and os.path.exists(
            kwargs["precomputed_partition_artifact_uri"]
        ):
            LOGGER.info(
                "Temporary precomputed partition artifact stored at %s; "
                "you may want to delete it.",
                kwargs["precomputed_partition_artifact_uri"],
            )
        return index

    def create_index(
        self,
        column: Union[str, List[str]],
        index_type: str,
        name: Optional[str] = None,
        metric: str = "L2",
        replace: bool = False,
        num_partitions: Optional[int] = None,
        ivf_centroids: Optional[
            Union[np.ndarray, pa.FixedSizeListArray, pa.FixedShapeTensorArray]
        ] = None,
        pq_codebook: Optional[
            Union[np.ndarray, pa.FixedSizeListArray, pa.FixedShapeTensorArray]
        ] = None,
        num_sub_vectors: Optional[int] = None,
        accelerator: Optional[Union[str, "torch.Device"]] = None,
        index_cache_size: Optional[int] = None,
        shuffle_partition_batches: Optional[int] = None,
        shuffle_partition_concurrency: Optional[int] = None,
        # experimental parameters
        ivf_centroids_file: Optional[str] = None,
        precomputed_partition_dataset: Optional[str] = None,
        storage_options: Optional[Dict[str, str]] = None,
        filter_nan: bool = True,
        train: bool = True,
        # distributed indexing parameters
        fragment_ids: Optional[List[int]] = None,
        index_uuid: Optional[str] = None,
        *,
        target_partition_size: Optional[int] = None,
        skip_transpose: bool = False,
        progress_callback: Optional[Callable[[IndexProgress], None]] = None,
        **kwargs,
    ) -> LanceDataset:
        """Create index on column.

        **Experimental API**

        Parameters
        ----------
        column : str
            The column to be indexed.
        index_type : str
            The type of the index.
            ``"IVF_PQ, IVF_HNSW_PQ and IVF_HNSW_SQ"`` are supported now.
        name : str, optional
            The index name. If not provided, it will be generated from the
            column name.
        metric : str
            The distance metric type, i.e., "L2" (alias to "euclidean"), "cosine"
            or "dot" (dot product). Default is "L2".
        replace : bool
            Replace the existing index if it exists.
        num_partitions : int, optional
            The number of partitions of IVF (Inverted File Index).
            Deprecated. Use target_partition_size instead.
        ivf_centroids : optional
            It can be either :py:class:`np.ndarray`,
            :py:class:`pyarrow.FixedSizeListArray` or
            :py:class:`pyarrow.FixedShapeTensorArray`.
            A ``num_partitions x dimension`` array of existing K-mean centroids
            for IVF clustering. If not provided, a new KMeans model will be trained.
        pq_codebook : optional,
            It can be :py:class:`np.ndarray`, :py:class:`pyarrow.FixedSizeListArray`,
            or :py:class:`pyarrow.FixedShapeTensorArray`.
            A ``num_sub_vectors x (2 ^ nbits * dimensions // num_sub_vectors)``
            array of K-mean centroids for PQ codebook.

            Note: ``nbits`` is always 8 for now.
            If not provided, a new PQ model will be trained.
        num_sub_vectors : int, optional
            The number of sub-vectors for PQ (Product Quantization).
        accelerator : str or ``torch.Device``, optional
            If set, use an accelerator to speed up the training process.
            Accepted accelerator:

            - "cuda" (Nvidia GPU)
            - "mps" (Apple Silicon GPU)
            - "cuvs" for the optional `pylance-cuvs` backend

            If not set, use the CPU.
        index_cache_size : int, optional
            The size of the index cache in number of entries. Default value is 256.
        shuffle_partition_batches : int, optional
            The number of batches, using the row group size of the dataset, to include
            in each shuffle partition. Default value is 10240.

            Assuming the row group size is 1024, each shuffle partition will hold
            10240 * 1024 = 10,485,760 rows. By making this value smaller, this shuffle
            will consume less memory but will take longer to complete, and vice versa.
        shuffle_partition_concurrency : int, optional
            The number of shuffle partitions to process concurrently. Default value is 2

            By making this value smaller, this shuffle will consume less memory but will
            take longer to complete, and vice versa.
        storage_options : optional, dict
            Extra options that make sense for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.
        filter_nan: bool
            Defaults to True. False is UNSAFE, and will cause a crash if any null/nan
            values are present (and otherwise will not). Disables the null filter used
            for nullable columns. Obtains a small speed boost.
        train : bool, default True
            If True, the index will be trained on the data (e.g., compute IVF
            centroids, PQ codebooks). If False, an empty index structure will be
            created without training, which can be populated later.
        fragment_ids : List[int], optional
            If provided, the index will be created only on the specified fragments.
            This enables distributed/fragment-level indexing. When provided, the
            method creates one segment but does not commit the index
            to the dataset. The returned metadata can be passed to
            optionally merged with ``merge_existing_index_segments(...)``
            and then committed with ``commit_existing_index_segments(...)``.
        index_uuid : str, optional
            A UUID to use for the segment written by this call.
            If not provided, a new UUID will be generated.
        progress_callback : callable, optional
            A callback that receives :class:`lance.progress.IndexProgress` events while
            the index is being built.
        target_partition_size: int, optional
            The target partition size. If set, the number of partitions will be computed
            based on the target partition size.
            Otherwise, the target partition size will be set by index type.
        kwargs :
            Parameters passed to the index building process.



        The SQ (Scalar Quantization) is available for only ``IVF_HNSW_SQ`` index type,
        this quantization method is used to reduce the memory usage of the index,
        it maps the float vectors to integer vectors, each integer is of ``num_bits``,
        now only 8 bits are supported.

        If ``index_type`` is "IVF_*", then the following parameters are required:
            num_partitions

        If ``index_type`` is with "PQ", then the following parameters are required:
            num_sub_vectors

        Optional parameters for `IVF_PQ`:

            - ivf_centroids
                Existing K-mean centroids for IVF clustering.
            - num_bits
                The number of bits for PQ (Product Quantization). Default is 8.
                Only 4, 8 are supported.
            - index_file_version
                The version of the index file. Default is "V3".
            - precomputed_partition_artifact_uri
                An advanced input produced by an external backend such as
                `pylance-cuvs`. When set, Lance skips its own partition
                assignment and consumes the precomputed partition-local artifact
                during finalization. Requires `ivf_centroids` and
                `pq_codebook`.

        Optional parameters for `IVF_RQ`:

            - num_bits
                The number of bits for RQ (Rabit Quantization). Default is 1.

        Optional parameters for `IVF_HNSW_*`:
            max_level
                Int, the maximum number of levels in the graph.
            m
                Int, the number of edges per node in the graph.
            ef_construction
                Int, the number of nodes to examine during the construction.

        Examples
        --------

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/sift.lance")
            dataset.create_index(
                "vector",
                "IVF_PQ",
                num_partitions=256,
                num_sub_vectors=16
            )

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/sift.lance")
            dataset.create_index(
                "vector",
                "IVF_HNSW_SQ",
                num_partitions=256,
            )

        Experimental Accelerator (GPU) support:

        - *accelerate*: use GPU to train IVF partitions.
            Supports CUDA (Nvidia) and MPS (Apple) via the built-in torch path.
            `accelerator="cuvs"` delegates IVF_PQ build preparation to the
            optional `pylance-cuvs` backend.

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/sift.lance")
            dataset.create_index(
                "vector",
                "IVF_PQ",
                num_partitions=256,
                num_sub_vectors=16,
                accelerator="cuda"
            )

        Note: GPU acceleration is currently supported only for the ``IVF_PQ`` index
        type. Providing an accelerator for other index types will fall back to CPU
        index building.

        References
        ----------
        * `Faiss Index <https://github.com/facebookresearch/faiss/wiki/Faiss-indexes>`_
        * IVF introduced in `Video Google: a text retrieval approach to object matching
          in videos <https://ieeexplore.ieee.org/abstract/document/1238663>`_
        * `Product quantization for nearest neighbor search
          <https://hal.inria.fr/inria-00514462v2/document>`_

        """
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        self._create_index_impl(
            column,
            index_type,
            name=name,
            metric=metric,
            replace=replace,
            num_partitions=num_partitions,
            ivf_centroids=ivf_centroids,
            pq_codebook=pq_codebook,
            num_sub_vectors=num_sub_vectors,
            accelerator=accelerator,
            index_cache_size=index_cache_size,
            shuffle_partition_batches=shuffle_partition_batches,
            shuffle_partition_concurrency=shuffle_partition_concurrency,
            ivf_centroids_file=ivf_centroids_file,
            precomputed_partition_dataset=precomputed_partition_dataset,
            storage_options=storage_options,
            filter_nan=filter_nan,
            train=train,
            fragment_ids=fragment_ids,
            index_uuid=index_uuid,
            target_partition_size=target_partition_size,
            skip_transpose=skip_transpose,
            require_commit=True,
            **kwargs,
        )
        return self

    def create_index_uncommitted(
        self,
        column: Union[str, List[str]],
        index_type: str,
        name: Optional[str] = None,
        metric: str = "L2",
        replace: bool = False,
        num_partitions: Optional[int] = None,
        ivf_centroids: Optional[
            Union[np.ndarray, pa.FixedSizeListArray, pa.FixedShapeTensorArray]
        ] = None,
        pq_codebook: Optional[
            Union[np.ndarray, pa.FixedSizeListArray, pa.FixedShapeTensorArray]
        ] = None,
        num_sub_vectors: Optional[int] = None,
        accelerator: Optional[Union[str, "torch.Device"]] = None,
        index_cache_size: Optional[int] = None,
        shuffle_partition_batches: Optional[int] = None,
        shuffle_partition_concurrency: Optional[int] = None,
        ivf_centroids_file: Optional[str] = None,
        precomputed_partition_dataset: Optional[str] = None,
        storage_options: Optional[Dict[str, str]] = None,
        filter_nan: bool = True,
        train: bool = True,
        fragment_ids: Optional[List[int]] = None,
        index_uuid: Optional[str] = None,
        *,
        target_partition_size: Optional[int] = None,
        skip_transpose: bool = False,
        **kwargs,
    ) -> Index:
        """
        Create one segment without publishing it and return its metadata.

        This is the public distributed-build API for vector index
        construction. Unlike :meth:`create_index`, this method does not publish
        the index into the dataset manifest. Instead, it writes one segment
        under ``_indices/<segment_uuid>/`` and returns the resulting
        :class:`Index` metadata.

        Callers should:

        1. run :meth:`create_index_uncommitted` on each worker with that worker's
           assigned ``fragment_ids``
        2. collect the returned :class:`Index` objects
        3. optionally merge one or more caller-defined groups with
           :meth:`merge_existing_index_segments`
        4. commit the final segment list with
           :meth:`commit_existing_index_segments`

        Parameters are the same as :meth:`create_index`, with one additional
        requirement:

        - ``fragment_ids`` must be provided

        Returns
        -------
        Index
            Metadata for the segment that was written by this call.
        """
        return self._create_index_impl(
            column,
            index_type,
            name=name,
            metric=metric,
            replace=replace,
            num_partitions=num_partitions,
            ivf_centroids=ivf_centroids,
            pq_codebook=pq_codebook,
            num_sub_vectors=num_sub_vectors,
            accelerator=accelerator,
            index_cache_size=index_cache_size,
            shuffle_partition_batches=shuffle_partition_batches,
            shuffle_partition_concurrency=shuffle_partition_concurrency,
            ivf_centroids_file=ivf_centroids_file,
            precomputed_partition_dataset=precomputed_partition_dataset,
            storage_options=storage_options,
            filter_nan=filter_nan,
            train=train,
            fragment_ids=fragment_ids,
            index_uuid=index_uuid,
            target_partition_size=target_partition_size,
            skip_transpose=skip_transpose,
            require_commit=False,
            **kwargs,
        )

    def drop_index(self, name: str):
        """
        Drops an index from the dataset

        Note: Indices are dropped by "index name".  This is not the same as the field
        name. If you did not specify a name when you created the index then a name was
        generated for you.  You can use the `describe_indices` method to get the names
        of the indices.
        """
        return self._ds.drop_index(name)

    def prewarm_index(self, name: str, *, with_position: bool = False):
        """
        Prewarm an index

        This will load the entire index into memory.  This can help avoid cold start
        issues with index queries.  If the index does not fit in the index cache, then
        this will result in wasted I/O.

        Parameters
        ----------
        name: str
            The name of the index to prewarm.
        with_position: bool, default False
            This is only supported for ``INVERTED`` indices. If True, positions are
            also loaded into the cache during prewarm so phrase queries do not need a
            separate lazy positions read.
        """
        return self._ds.prewarm_index(name, with_position=with_position)

    def merge_index_metadata(
        self,
        index_uuid: str,
        index_type: str,
        batch_readhead: Optional[int] = None,
        progress_callback: Optional[Callable[[IndexProgress], None]] = None,
    ):
        """
        Merge distributed scalar index metadata.

        Vector distributed indexing no longer uses this API. For vector indices,
        build segments with :meth:`create_index_uncommitted`, optionally merge
        caller-defined groups with :meth:`merge_existing_index_segments`, and
        publish them with :meth:`commit_existing_index_segments`.

        This method does NOT commit changes.

        This API merges temporary scalar index files (for example per-fragment
        BTree or inverted index outputs).
        After this method returns, callers MUST explicitly commit
        the index manifest using lance.LanceDataset.commit(...)
        with a LanceOperation.CreateIndex.

        Parameters
        ----------
        index_uuid: str
            The shared UUID used when building fragment-level scalar indices.
        index_type: str
            Index type name. Must be one of the enum values in
            :class:`lance.indices.SupportedDistributedIndices`
            supported by scalar distributed merge.
        batch_readhead: int, optional
            Prefetch concurrency used by BTREE merge reader. Default: 1.
        progress_callback: callable, optional
            A callback that receives :class:`lance.progress.IndexProgress` events while
            metadata is being merged.
        """
        # Normalize type
        t = index_type.upper()

        valid = {member.name for member in SupportedDistributedIndices}
        if t not in valid:
            raise NotImplementedError(
                f"Only {', '.join(sorted(valid))} are supported, received {index_type}"
            )

        # Merge physical index files at the index directory
        self._ds.merge_index_metadata(index_uuid, t, batch_readhead, progress_callback)
        return None

    def merge_existing_index_segments(self, segments: List[Index]) -> Index:
        """
        Merge one caller-defined group of existing uncommitted segments.
        """
        return self._ds.merge_existing_index_segments(segments)

    def commit_existing_index_segments(
        self, index_name: str, column: str, segments: List[Index]
    ) -> LanceDataset:
        """
        Commit built index segments as one logical index.
        """
        self._ds.commit_existing_index_segments(index_name, column, segments)
        return self

    def session(self) -> Session:
        """
        Return the dataset session, which holds the dataset's state.
        """
        return self._ds.session()

    @staticmethod
    def _commit(
        base_uri: Union[str, Path],
        operation: LanceOperation.BaseOperation,
        read_version: Optional[int] = None,
        commit_lock: Optional[CommitLock] = None,
    ) -> LanceDataset:
        warnings.warn(
            "LanceDataset._commit() is deprecated, use LanceDataset.commit() instead",
            DeprecationWarning,
        )
        return LanceDataset.commit(base_uri, operation, read_version, commit_lock)

    @staticmethod
    def commit(
        base_uri: Union[str, Path, LanceDataset],
        operation: Union[LanceOperation.BaseOperation, Transaction],
        read_version: Optional[int] = None,
        commit_lock: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_v2_manifest_paths: Optional[bool] = None,
        detached: Optional[bool] = False,
        max_retries: int = 20,
        *,
        commit_message: Optional[str] = None,
        enable_stable_row_ids: Optional[bool] = None,
        namespace_client: Optional["LanceNamespace"] = None,
        table_id: Optional[List[str]] = None,
        namespace_client_managed_versioning: bool = False,
    ) -> LanceDataset:
        """Create a new version of dataset

        This method is an advanced method which allows users to describe a change
        that has been made to the data files.  This method is not needed when using
        Lance to apply changes (e.g. when using :py:class:`LanceDataset` or
        :py:func:`write_dataset`.)

        It's current purpose is to allow for changes being made in a distributed
        environment where no single process is doing all of the work.  For example,
        a distributed bulk update or a distributed bulk modify operation.

        Once all of the changes have been made, this method can be called to make
        the changes visible by updating the dataset manifest.

        Warnings
        --------
        This is an advanced API and doesn't provide the same level of validation
        as the other APIs. For example, it's the responsibility of the caller to
        ensure that the fragments are valid for the schema.

        Parameters
        ----------
        base_uri: str, Path, or LanceDataset
            The base uri of the dataset, or the dataset object itself. Using
            the dataset object can be more efficient because it can re-use the
            file metadata cache.
        operation: BaseOperation
            The operation to apply to the dataset.  This describes what changes
            have been made. See available operations under :class:`LanceOperation`.
        read_version: int, optional
            The version of the dataset that was used as the base for the changes.
            This is not needed for overwrite or restore operations.
        commit_lock : CommitLock, optional
            A custom commit lock.  Only needed if your object store does not support
            atomic commits.  See the user guide for more details.
        storage_options : optional, dict
            Extra options that make sense for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.
        enable_v2_manifest_paths : bool, optional
            If True, and this is a new dataset, uses the new V2 manifest paths.
            These paths provide more efficient opening of datasets with many
            versions on object stores. This parameter has no effect if the dataset
            already exists. To migrate an existing dataset, instead use the
            :meth:`migrate_manifest_paths_v2` method. Default is True. WARNING:
            turning this on will make the dataset unreadable for older versions
            of Lance (prior to 0.17.0).
        detached : bool, optional
            If True, then the commit will not be part of the dataset lineage.  It will
            never show up as the latest dataset and the only way to check it out in the
            future will be to specifically check it out by version.  The version will be
            a random version that is only unique amongst detached commits.  The caller
            should store this somewhere as there will be no other way to obtain it in
            the future.
        max_retries : int
            The maximum number of retries to perform when committing the dataset.
        commit_message: str, optional
            A message to associate with this commit. This message will be stored in the
            dataset's metadata and can be retrieved using read_transaction().
        enable_stable_row_ids: bool, optional
            If True, enables stable row IDs when creating a new dataset.  Stable
            row IDs assign each row a monotonically increasing id that persists
            across compaction and other maintenance operations.  This option is
            ignored for existing datasets.
        namespace_client : LanceNamespace, optional
            A namespace client. Must be provided together with table_id.
            Use lance.namespace.connect() to create a namespace.
        table_id : List[str], optional
            The table identifier within the namespace (e.g., ["workspace", "table"]).
            Must be provided together with namespace_client.

        Returns
        -------
        LanceDataset
            A new version of Lance Dataset.

        Examples
        --------

        Creating a new dataset with the :class:`LanceOperation.Overwrite` operation:

        >>> import lance
        >>> import pyarrow as pa
        >>> tab1 = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> tab2 = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> fragment1 = lance.fragment.LanceFragment.create("example", tab1)
        >>> fragment2 = lance.fragment.LanceFragment.create("example", tab2)
        >>> fragments = [fragment1, fragment2]
        >>> operation = lance.LanceOperation.Overwrite(tab1.schema, fragments)
        >>> dataset = lance.LanceDataset.commit("example", operation)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        """
        if isinstance(base_uri, Path):
            base_uri = str(base_uri)
        elif isinstance(base_uri, LanceDataset):
            base_uri = base_uri._ds
        elif not isinstance(base_uri, str):
            raise TypeError(
                f"base_uri must be str, Path, or LanceDataset, got {type(base_uri)}"
            )

        if commit_lock:
            if not callable(commit_lock):
                raise TypeError(
                    f"commit_lock must be a function, got {type(commit_lock)}"
                )

        if (
            isinstance(operation, LanceOperation.BaseOperation)
            and read_version is None
            and not isinstance(
                operation, (LanceOperation.Overwrite, LanceOperation.Restore)
            )
        ):
            raise ValueError(
                "read_version is required for all operations except "
                "Overwrite and Restore"
            )

        # Storage options provider is automatically created in Rust when
        # namespace_client and table_id are provided
        if isinstance(operation, Transaction):
            if commit_message is not None:
                raise ValueError(
                    "commit_message is not supported when calling commit with "
                    "a Transaction.  Set the message on the transaction properties "
                    "instead."
                )
            new_ds = _Dataset.commit_transaction(
                base_uri,
                operation,
                commit_lock,
                storage_options=storage_options,
                enable_v2_manifest_paths=enable_v2_manifest_paths,
                detached=detached,
                max_retries=max_retries,
                enable_stable_row_ids=enable_stable_row_ids,
                namespace_client=namespace_client,
                table_id=table_id,
                namespace_client_managed_versioning=namespace_client_managed_versioning,
            )
        elif isinstance(operation, LanceOperation.BaseOperation):
            new_ds = _Dataset.commit(
                base_uri,
                operation,
                read_version,
                commit_lock,
                storage_options=storage_options,
                enable_v2_manifest_paths=enable_v2_manifest_paths,
                detached=detached,
                max_retries=max_retries,
                commit_message=commit_message,
                enable_stable_row_ids=enable_stable_row_ids,
                namespace_client=namespace_client,
                table_id=table_id,
                namespace_client_managed_versioning=namespace_client_managed_versioning,
            )
        else:
            raise TypeError(
                "operation must be a LanceOperation.BaseOperation or Transaction, "
                f"got {type(operation)}"
            )

        ds = LanceDataset.__new__(LanceDataset)
        ds._storage_options = storage_options
        ds._namespace_client = namespace_client
        ds._table_id = table_id
        ds._namespace_client_managed_versioning = namespace_client_managed_versioning
        ds._ds = new_ds
        ds._uri = new_ds.uri
        ds._default_scan_options = None
        ds._read_params = None
        return ds

    @staticmethod
    def commit_batch(
        dest: Union[str, Path, LanceDataset],
        transactions: Sequence[Transaction],
        commit_lock: Optional[CommitLock] = None,
        storage_options: Optional[Dict[str, str]] = None,
        enable_v2_manifest_paths: Optional[bool] = None,
        detached: Optional[bool] = False,
        max_retries: int = 20,
    ) -> BulkCommitResult:
        """Create a new version of dataset with multiple transactions.

        This method is an advanced method which allows users to describe a change
        that has been made to the data files.  This method is not needed when using
        Lance to apply changes (e.g. when using :py:class:`LanceDataset` or
        :py:func:`write_dataset`.)

        Parameters
        ----------
        dest: str, Path, or LanceDataset
            The base uri of the dataset, or the dataset object itself. Using
            the dataset object can be more efficient because it can re-use the
            file metadata cache.
        transactions: Iterable[Transaction]
            The transactions to apply to the dataset. These will be merged into
            a single transaction and applied to the dataset. Note: Only append
            transactions are currently supported. Other transaction types will be
            supported in the future.
        commit_lock : CommitLock, optional
            A custom commit lock.  Only needed if your object store does not support
            atomic commits.  See the user guide for more details.
        storage_options : optional, dict
            Extra options that make sense for a particular storage connection. This is
            used to store connection parameters like credentials, endpoint, etc.
        enable_v2_manifest_paths : bool, optional
            If True, and this is a new dataset, uses the new V2 manifest paths.
            These paths provide more efficient opening of datasets with many
            versions on object stores. This parameter has no effect if the dataset
            already exists. To migrate an existing dataset, instead use the
            :meth:`migrate_manifest_paths_v2` method. Default is False. WARNING:
            turning this on will make the dataset unreadable for older versions
            of Lance (prior to 0.17.0).
        detached : bool, optional
            If True, then the commit will not be part of the dataset lineage.  It will
            never show up as the latest dataset and the only way to check it out in the
            future will be to specifically check it out by version.  The version will be
            a random version that is only unique amongst detached commits.  The caller
            should store this somewhere as there will be no other way to obtain it in
            the future.
        max_retries : int
            The maximum number of retries to perform when committing the dataset.

        Returns
        -------
        dict with keys:
            dataset: LanceDataset
                A new version of Lance Dataset.
            merged: Transaction
                The merged transaction that was applied to the dataset.
        """
        if isinstance(dest, Path):
            dest = str(dest)
        elif isinstance(dest, LanceDataset):
            dest = dest._ds
        elif not isinstance(dest, str):
            raise TypeError(
                f"base_uri must be str, Path, or LanceDataset, got {type(dest)}"
            )

        if commit_lock:
            if not callable(commit_lock):
                raise TypeError(
                    f"commit_lock must be a function, got {type(commit_lock)}"
                )

        new_ds, merged = _Dataset.commit_batch(
            dest,
            transactions,
            commit_lock,
            storage_options=storage_options,
            enable_v2_manifest_paths=enable_v2_manifest_paths,
            detached=detached,
            max_retries=max_retries,
        )
        ds = LanceDataset.__new__(LanceDataset)
        ds._ds = new_ds
        ds._uri = new_ds.uri
        ds._storage_options = storage_options
        ds._namespace_client = None
        ds._table_id = None
        ds._default_scan_options = None
        ds._read_params = None
        return BulkCommitResult(
            dataset=ds,
            merged=merged,
        )

    def validate(self):
        """
        Validate the dataset.

        This checks the integrity of the dataset and will raise an exception if
        the dataset is corrupted.
        """
        self._ds.validate()

    def shallow_clone(
        self,
        target_path: str | Path,
        reference: int | str | Tuple[Optional[str], Optional[int]],
        storage_options: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "LanceDataset":
        """
        Shallow clone the specified version into a new dataset at target_path.

        This operation copies only the dataset metadata and references, without
        rewriting data files.

        Parameters
        ----------
        target_path : str or Path
            The URI or filesystem path to clone the dataset into.
        reference : int, str or Tuple[Optional[str], Optional[int]]
            An integer specifies a version number in the current branch; a string
            specifies a tag name; a Tuple[Optional[str], Optional[int]] specifies
            a version number in a specified branch. (None, None) means the latest
            version_number on the main branch.
        storage_options : dict, optional
            Object store configuration for the new dataset (e.g., credentials,
            endpoints). If not specified, the storage options of the source dataset
            will be used.

        Returns
        -------
        LanceDataset
            A new LanceDataset representing the shallow-cloned dataset.
        """
        if isinstance(target_path, Path):
            target_uri = os.fspath(target_path)
        else:
            target_uri = target_path

        if storage_options is None:
            storage_options = self._storage_options
        self._ds.shallow_clone(target_uri, reference, storage_options)

        # Open and return a fresh dataset at the target URI to avoid manual overrides
        return LanceDataset(target_uri, storage_options=storage_options, **kwargs)

    def migrate_manifest_paths_v2(self):
        """
        Migrate the manifest paths to the new format.

        This will update the manifest to use the new v2 format for paths.

        This function is idempotent, and can be run multiple times without
        changing the state of the object store.

        DANGER: this should not be run while other concurrent operations are happening.
        And it should also run until completion before resuming other operations.
        """
        self._ds.migrate_manifest_paths_v2()

    def delete_config_keys(self, keys: list[str]) -> None:
        """Delete specified configuration keys from the dataset.

        Parameters
        ----------
        keys : list of str
            A list of configuration keys to remove from the dataset.
            Non-existent keys will be silently ignored.
        """
        self._ds.delete_config_keys(keys)

    def config(self) -> dict[str, str]:
        """Get configs of the dataset.

        Parameters
        ----------
        Returns
        -------
        dict[str, str]
            A list of configuration items.
        """
        return self._ds.config()

    def read_transaction(self, version: int) -> Optional[Transaction]:
        """Get Transaction at a specific version.

        Parameters
        ----------
        version: int
            The version number of the dataset to retrieve properties from

        Returns
        -------
        Dict[str, str] or None
            A dictionary containing the properties of the Transaction
            , or None if the transaction doesn't exist.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> from lance.dataset import LANCE_COMMIT_MESSAGE_KEY
        >>> table1 = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> ds = lance.write_dataset(   \\
        ... table1, \\
        ... "properties_example",   \\
        ... commit_message="initial")
        >>> table2 = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> properties2 = {LANCE_COMMIT_MESSAGE_KEY: "Second commit", \\
        ... "another_prop": "another_value"}
        >>> ds = lance.write_dataset( \\
        ... table2, \\
        ... "properties_example",   \\
        ... mode="append",  \\
        ... transaction_properties=properties2)
        >>> print(ds.read_transaction(1).transaction_properties)
        {'__lance_commit_message': 'initial'}
        >>> print(ds.read_transaction(2).transaction_properties.get("another_prop"))
        another_value
        """
        return self._ds.read_transaction(version)

    def get_transactions(self, recent_transactions=10) -> List[Optional[Transaction]]:
        return self._ds.get_transactions(recent_transactions)

    def sql(self, sql: str) -> "SqlQueryBuilder":
        """Execute SQL query on the dataset.

        Parameters
        ----------
        sql : str
            SQL SELECT statement to execute. The query must reference the dataset
            as the FROM clause or omit the FROM clause to query the current dataset.

        Returns
        -------
        SqlQueryBuilder
            A builder that can be used to configure and build the query.

        Examples
        --------
         .. code-block:: python

            import lance
            dataset = lance.dataset("/tmp/data.lance")
            query = dataset.sql("SELECT id, name FROM dataset WHERE age > 30").build()
            query.to_batch_records()

        """
        return SqlQueryBuilder(self._ds.sql(sql))

    def delta(
        self,
        compared_against: Optional[int] = None,
        *,
        begin_version: Optional[int] = None,
        end_version: Optional[int] = None,
    ) -> "DatasetDelta":
        """
        Compare changes between two versions of this dataset.

        You must specify either ``compared_against`` (shorthand for comparing the
        current version against a specific older version) or both ``begin_version``
        and ``end_version`` for an explicit range.

        Parameters
        ----------
        compared_against : int, optional
            The version to compare the current dataset version against.
            This is a shorthand for setting ``begin_version=compared_against``
            and ``end_version=self.version``.
        begin_version : int, optional
            The start version (exclusive) for the comparison range.
            Must be used together with ``end_version``.
        end_version : int, optional
            The end version (inclusive) for the comparison range.
            Must be used together with ``begin_version``.

        Returns
        -------
        DatasetDelta
            An object that can list transactions or stream inserted/updated rows.

        Raises
        ------
        ValueError
            If both ``compared_against`` and version range are specified,
            or if neither is specified.

        Examples
        --------
        .. code-block:: python

            import lance
            import pyarrow as pa

            # Write initial data (v1)
            ds = lance.write_dataset(
                pa.table({"id": [1, 2], "val": ["a", "b"]}),
                "memory://delta_demo"
            )

            # Append some data to create v2
            ds_append = lance.write_dataset(
                pa.table({"id": [3], "val": ["c"]}),
                "memory://delta_demo",
                mode="append"
            )

            # Compute inserted rows from v1 -> v2 (shorthand)
            delta = ds_append.delta(compared_against=1)
            reader = delta.get_inserted_rows()
            for batch in reader:
                print(batch)

            # Or using explicit version range
            delta = ds_append.delta(begin_version=1, end_version=2)
        """
        has_compared_against = compared_against is not None

        builder = _DatasetDeltaBuilder(self._ds.delta())

        if has_compared_against:
            builder = builder.compared_against_version(compared_against)
        else:
            if begin_version:
                builder = builder.with_begin_version(begin_version)

            if end_version:
                builder = builder.with_end_version(end_version)

        return builder.build()

    @property
    def optimize(self) -> "DatasetOptimizer":
        return DatasetOptimizer(self)

    @property
    def stats(self) -> "LanceStats":
        """
        **Experimental API**
        """
        return LanceStats(self._ds)

    @staticmethod
    def drop(
        base_uri: Union[str, Path],
        storage_options: Optional[Dict[str, str]] = None,
        ignore_not_found: Optional[bool] = None,
    ) -> None:
        _Dataset.drop(str(base_uri), storage_options, ignore_not_found=ignore_not_found)

    def get_ivf_model(self, index_name: str):
        """Return the IVF model for a vector index.

        This directly forwards to the underlying Rust Dataset implementation
        and returns a ``PyIvfModel`` instance (see ``lance.indices``).

        Parameters
        ----------
        index_name : str
            The name of the vector index (e.g. ``"vector_idx"``).
        """
        return self._ds.get_ivf_model(index_name)

    def _default_vector_index_for_column(self, column: str) -> str:
        """Return the first index name that covers *column* and is IVF-based.

        Raises KeyError if no such index exists.
        """
        # Resolve column path to field id for describe_indices matching.
        lance_field = self._ds.lance_schema.field_case_insensitive(column)
        if lance_field is None:
            raise KeyError(f"No IVF index for column '{column}'")
        field_id = lance_field.id()

        indices = self.describe_indices()
        for idx in indices:
            if field_id in idx.fields:
                # Use index_stats to get the concrete IVF subtype.
                index_type = self.stats.index_stats(idx.name).get("index_type", "")
                if index_type.startswith("IVF"):
                    return idx.name
        raise KeyError(f"No IVF index for column '{column}'")

    def centroids(
        self,
        *,
        index_name: str | None = None,
        column: str | None = None,
    ):
        """Return IVF centroids for a given index/column.

        Parameters
        ----------
        index_name : str, optional
            Explicit index name.  If omitted, *column* must be provided.
        column : str, optional
            Column whose default IVF index should be inspected.
        """

        if index_name is None:
            if column is None:
                raise ValueError("Must provide 'index_name' or 'column'.")
            index_name = self._default_vector_index_for_column(column)

        ivf = self.get_ivf_model(index_name)
        if ivf is None:
            return None

        return ivf.centroids


class SqlQuery:
    """
    An executable SQL query.

    This is created by calling :meth:`SqlQueryBuilder.build`.
    """

    def __init__(self, query):
        self._query = query

    def to_batch_records(self) -> list[pa.RecordBatch]:
        """
        Execute the query and return a list of RecordBatches.

        This is an eager operation that will load all results into memory.

        Returns
        -------
        list[pyarrow.RecordBatch]
        """
        return self._query.to_batch_records()

    def to_stream_reader(self) -> pa.RecordBatchReader:
        """
        Execute the query and return a RecordBatchReader.

        This is a lazy operation that will stream results.

        Returns
        -------
        pyarrow.RecordBatchReader
        """
        return self._query.to_stream_reader()


class SqlQueryBuilder:
    """
    A builder for SQL queries on a Lance dataset.

    This builder allows for chaining of options to configure the SQL query.
    It is returned by :meth:`LanceDataset.sql`.
    """

    def __init__(self, builder):
        self._builder = builder

    def table_name(self, table_name: str) -> "SqlQueryBuilder":
        """
        Set the table name for the query.

        Parameters
        ----------
        table_name: str
            The name of the table to query.
        """
        self._builder = self._builder.table_name(table_name)
        return self

    def with_row_id(self, with_row_id: bool = True) -> "SqlQueryBuilder":
        """
        Include the row ID in the query result.

        Parameters
        ----------
        with_row_id: bool, default True
            Whether to include the row ID column (`_rowid`).
        """
        self._builder = self._builder.with_row_id(with_row_id)
        return self

    def with_row_addr(self, with_row_addr: bool = True) -> "SqlQueryBuilder":
        """
        Include the row address in the query result.

        Parameters
        ----------
        with_row_addr: bool, default True
            Whether to include the row address column (`_rowaddr`).
        """
        self._builder = self._builder.with_row_addr(with_row_addr)
        return self

    def build(self) -> SqlQuery:
        """
        Build the query.

        Returns
        -------
        SqlQuery
            An executable query object.
        """
        return SqlQuery(self._builder.build())


class DatasetDelta:
    """
    A view of differences between two versions.

    Created by :meth:`LanceDataset.delta`.
    Provides convenient methods to stream inserted/updated rows or list transactions.
    """

    def __init__(self, delta):
        self._delta = delta

    def list_transactions(self) -> List[Transaction]:
        """
        List transactions in the range from begin_version + 1 to end_version.
        """
        return self._delta.list_transactions()

    def get_inserted_rows(self) -> pa.RecordBatchReader:
        """
        Return a streaming RecordBatchReader for inserted rows.
        """
        return self._delta.get_inserted_rows()

    def get_updated_rows(self) -> pa.RecordBatchReader:
        """
        Return a streaming RecordBatchReader for updated rows.
        """
        return self._delta.get_updated_rows()


class _DatasetDeltaBuilder:
    """Internal builder for :class:`DatasetDelta`.

    This class is not part of the public API. Use :meth:`LanceDataset.delta` instead.
    """

    def __init__(self, builder):
        self._builder = builder

    def compared_against_version(self, version: int) -> "_DatasetDeltaBuilder":
        self._builder = self._builder.compared_against_version(version)
        return self

    def with_begin_version(self, version: int) -> "_DatasetDeltaBuilder":
        self._builder = self._builder.with_begin_version(version)
        return self

    def with_end_version(self, version: int) -> "_DatasetDeltaBuilder":
        self._builder = self._builder.with_end_version(version)
        return self

    def build(self) -> DatasetDelta:
        return DatasetDelta(self._builder.build())


class BulkCommitResult(TypedDict):
    dataset: LanceDataset
    merged: Transaction


@dataclass
class Transaction:
    read_version: int
    operation: LanceOperation.BaseOperation
    uuid: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    transaction_properties: Optional[Dict[str, str]] = dataclasses.field(
        default_factory=dict
    )


class Tag(TypedDict):
    branch: Optional[str]
    version: int
    manifest_size: int


class Branch(TypedDict):
    parent_branch: Optional[str]
    branch_identifier: List[Tuple[int, str]]
    parent_version: int
    create_at: int
    manifest_size: int


class Version(TypedDict):
    version: int
    timestamp: int | datetime
    metadata: Dict[str, str]


class UpdateResult(TypedDict):
    num_rows_updated: int


class DeleteResult(TypedDict):
    num_deleted_rows: int


class AlterColumn(TypedDict):
    path: str
    name: Optional[str]
    nullable: Optional[bool]
    data_type: Optional[pa.DataType]


class ExecuteResult(TypedDict):
    num_inserted_rows: int
    num_updated_rows: int
    num_deleted_rows: int


@dataclass
class IndexFile:
    """Metadata about a file in an index segment."""

    path: str
    size_bytes: int


@dataclass
class Index:
    """Represents an index in the dataset."""

    uuid: str
    name: str
    fields: List[int]
    dataset_version: int
    fragment_ids: Set[int]
    index_version: int
    created_at: Optional[datetime] = None
    base_id: Optional[int] = None
    files: Optional[List["IndexFile"]] = None
    index_details: Optional[tuple[str, bytes]] = None


class AutoCleanupConfig(TypedDict):
    interval: int
    older_than_seconds: int


# LanceOperation is a namespace for operations that can be applied to a dataset.
class LanceOperation:
    @staticmethod
    def _validate_fragments(fragments):
        if not isinstance(fragments, list):
            raise TypeError(
                f"fragments must be list[FragmentMetadata], got {type(fragments)}"
            )
        if len(fragments) > 0 and not all(
            isinstance(f, FragmentMetadata) for f in fragments
        ):
            raise TypeError(
                f"fragments must be list[FragmentMetadata], got {type(fragments[0])}"
            )

    class BaseOperation(ABC):
        """
        Base class for operations that can be applied to a dataset.

        See available operations under :class:`LanceOperation`.
        """

    @dataclass
    class Overwrite(BaseOperation):
        """
        Overwrite or create a new dataset.

        Attributes
        ----------
        new_schema: pyarrow.Schema
            The schema of the new dataset.
        fragments: list[FragmentMetadata]
            The fragments that make up the new dataset.
        initial_bases: list[DatasetBasePath], optional
            Base paths to register when creating a new dataset (CREATE mode only).
            **Only valid in CREATE mode**. Will raise an error if used with
            OVERWRITE on existing dataset.

        Warning
        -------
        This is an advanced API for distributed operations. To overwrite or
        create new dataset on a single machine, use :func:`lance.write_dataset`.

        Examples
        --------

        To create or overwrite a dataset, first use
        :meth:`lance.fragment.LanceFragment.create` to create fragments. Then
        collect the fragment metadata into a list and pass it along with the
        schema to this class. Finally, pass the operation to the
        :meth:`LanceDataset.commit` method to create the new dataset.

        >>> import lance
        >>> import pyarrow as pa
        >>> tab1 = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> tab2 = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> fragment1 = lance.fragment.LanceFragment.create("example", tab1)
        >>> fragment2 = lance.fragment.LanceFragment.create("example", tab2)
        >>> fragments = [fragment1, fragment2]
        >>> operation = lance.LanceOperation.Overwrite(tab1.schema, fragments)
        >>> dataset = lance.LanceDataset.commit("example", operation)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        """

        new_schema: LanceSchema | pa.Schema
        fragments: Iterable[FragmentMetadata]
        initial_bases: Optional[List[DatasetBasePath]] = None

        def __post_init__(self):
            if isinstance(self.new_schema, pa.Schema):
                self.new_schema = LanceSchema.from_pyarrow(self.new_schema)
            LanceOperation._validate_fragments(self.fragments)

    @dataclass
    class Append(BaseOperation):
        """
        Append new rows to the dataset.

        Attributes
        ----------
        fragments: list[FragmentMetadata]
            The fragments that contain the new rows.

        Warning
        -------
        This is an advanced API for distributed operations. To append to a
        dataset on a single machine, use :func:`lance.write_dataset`.

        Examples
        --------

        To append new rows to a dataset, first use
        :meth:`lance.fragment.LanceFragment.create` to create fragments. Then
        collect the fragment metadata into a list and pass it to this class.
        Finally, pass the operation to the :meth:`LanceDataset.commit`
        method to create the new dataset.

        >>> import lance
        >>> import pyarrow as pa
        >>> tab1 = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> dataset = lance.write_dataset(tab1, "example")
        >>> tab2 = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> fragment = lance.fragment.LanceFragment.create("example", tab2)
        >>> operation = lance.LanceOperation.Append([fragment])
        >>> dataset = lance.LanceDataset.commit("example", operation,
        ...                                     read_version=dataset.version)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        """

        fragments: Iterable[FragmentMetadata]

        def __post_init__(self):
            LanceOperation._validate_fragments(self.fragments)

    @dataclass
    class Delete(BaseOperation):
        """
        Remove fragments or rows from the dataset.

        Attributes
        ----------
        updated_fragments: list[FragmentMetadata]
            The fragments that have been updated with new deletion vectors.
        deleted_fragment_ids: list[int]
            The ids of the fragments that have been deleted entirely. These are
            the fragments where :meth:`LanceFragment.delete()` returned None.
        predicate: str
            The original SQL predicate used to select the rows to delete.

        Warning
        -------
        This is an advanced API for distributed operations. To delete rows from
        dataset on a single machine, use :meth:`lance.LanceDataset.delete`.

        Examples
        --------

        To delete rows from a dataset, call :meth:`lance.fragment.LanceFragment.delete`
        on each of the fragments. If that returns a new fragment, add that to
        the ``updated_fragments`` list. If it returns None, that means the whole
        fragment was deleted, so add the fragment id to the ``deleted_fragment_ids``.
        Finally, pass the operation to the :meth:`LanceDataset.commit` method to
        complete the deletion operation.

        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"a": [1, 2], "b": ["a", "b"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> table = pa.table({"a": [3, 4], "b": ["c", "d"]})
        >>> dataset = lance.write_dataset(table, "example", mode="append")
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        >>> predicate = "a >= 2"
        >>> updated_fragments = []
        >>> deleted_fragment_ids = []
        >>> for fragment in dataset.get_fragments():
        ...     new_fragment = fragment.delete(predicate)
        ...     if new_fragment is not None:
        ...         updated_fragments.append(new_fragment)
        ...     else:
        ...         deleted_fragment_ids.append(fragment.fragment_id)
        >>> operation = lance.LanceOperation.Delete(updated_fragments,
        ...                                         deleted_fragment_ids,
        ...                                         predicate)
        >>> dataset = lance.LanceDataset.commit("example", operation,
        ...                                     read_version=dataset.version)
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        """

        updated_fragments: Iterable[FragmentMetadata]
        deleted_fragment_ids: Iterable[int]
        predicate: str

        def __post_init__(self):
            LanceOperation._validate_fragments(self.updated_fragments)

    @dataclass
    class Update(BaseOperation):
        """
        Operation that updates rows in the dataset.

        Attributes
        ----------
        removed_fragment_ids: list[int]
            The ids of the fragments that have been removed entirely.
        updated_fragments: list[FragmentMetadata]
            The fragments that have been updated with new deletion vectors.
        new_fragments: list[FragmentMetadata]
            The fragments that contain the new rows.
        fields_modified: list[int]
            If any fields are modified in updated_fragments, then they must be
            listed here so those fragments can be removed from indices covering
            those fields.
        fields_for_preserving_frag_bitmap: list[int]
            The fields that used to judge whether to preserve the new frag's id into
            the frag bitmap of the specified indices.
        """

        removed_fragment_ids: List[int] = dataclasses.field(default_factory=list)
        updated_fragments: List[FragmentMetadata] = dataclasses.field(
            default_factory=list
        )
        new_fragments: List[FragmentMetadata] = dataclasses.field(default_factory=list)
        fields_modified: List[int] = dataclasses.field(default_factory=list)
        fields_for_preserving_frag_bitmap: List[int] = dataclasses.field(
            default_factory=list
        )
        update_mode: str = ""

        def __post_init__(self):
            LanceOperation._validate_fragments(self.updated_fragments)
            LanceOperation._validate_fragments(self.new_fragments)

    @dataclass
    class Merge(BaseOperation):
        """
        Operation that adds columns. Unlike Overwrite, this should not change
        the structure of the fragments, allowing existing indices to be kept.

        Attributes
        ----------
        fragments: iterable of FragmentMetadata
            The fragments that make up the new dataset.
        schema: LanceSchema or pyarrow.Schema
            The schema of the new dataset. Passing a LanceSchema is preferred,
            and passing a pyarrow.Schema is deprecated.

        Warning
        -------
        This is an advanced API for distributed operations. To overwrite or
        create new dataset on a single machine, use :func:`lance.write_dataset`.

        Examples
        --------

        To add new columns to a dataset, first define a method that will create
        the new columns based on the existing columns. Then use
        :meth:`lance.fragment.LanceFragment.add_columns`

        >>> import lance
        >>> import pyarrow as pa
        >>> import pyarrow.compute as pc
        >>> table = pa.table({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.to_table().to_pandas()
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  4  d
        >>> def double_a(batch: pa.RecordBatch) -> pa.RecordBatch:
        ...     doubled = pc.multiply(batch["a"], 2)
        ...     return pa.record_batch([doubled], ["a_doubled"])
        >>> fragments = []
        >>> for fragment in dataset.get_fragments():
        ...     new_fragment, new_schema = fragment.merge_columns(double_a,
        ...                                                       columns=['a'])
        ...     fragments.append(new_fragment)
        >>> operation = lance.LanceOperation.Merge(fragments, new_schema)
        >>> dataset = lance.LanceDataset.commit("example", operation,
        ...                                     read_version=dataset.version)
        >>> dataset.to_table().to_pandas()
           a  b  a_doubled
        0  1  a          2
        1  2  b          4
        2  3  c          6
        3  4  d          8
        """

        fragments: Iterable[FragmentMetadata]
        schema: LanceSchema | pa.Schema

        def __post_init__(self):
            if isinstance(self.schema, pa.Schema):
                warnings.warn(
                    "Passing a pyarrow.Schema to Merge is deprecated. "
                    "Please use a LanceSchema instead.",
                    DeprecationWarning,
                )
                self.schema = LanceSchema.from_pyarrow(self.schema)
            LanceOperation._validate_fragments(self.fragments)

    @dataclass
    class Restore(BaseOperation):
        """
        Operation that restores a previous version of the dataset.
        """

        version: int

    @dataclass
    class RewriteGroup:
        """
        Collection of rewritten files
        """

        old_fragments: Iterable[FragmentMetadata]
        new_fragments: Iterable[FragmentMetadata]

    @dataclass
    class RewrittenIndex:
        """
        An index that has been rewritten
        """

        old_id: str
        new_id: str
        new_details_type_url: str
        new_details_value: bytes
        new_index_version: int

    @dataclass
    class Rewrite(BaseOperation):
        """
        Operation that rewrites one or more files and indices into one
        or more files and indices.

        Attributes
        ----------
        groups: list[RewriteGroup]
            Groups of files that have been rewritten.
        rewritten_indices: list[RewrittenIndex]
            Indices that have been rewritten.

        Warning
        -------
        This is an advanced API not intended for general use.
        """

        groups: Iterable[LanceOperation.RewriteGroup]
        rewritten_indices: Iterable[LanceOperation.RewrfittenIndex]

        def __post_init__(self):
            all_frags = [old for group in self.groups for old in group.old_fragments]
            all_frags += [new for group in self.groups for new in group.new_fragments]
            LanceOperation._validate_fragments(all_frags)

    @dataclass
    class CreateIndex(BaseOperation):
        """
        Operation that creates an index on the dataset.
        """

        new_indices: List[Index]
        removed_indices: List[Index]

    @dataclass
    class DataReplacementGroup:
        """
        Group of data replacements
        """

        fragment_id: int
        new_file: DataFile

    @dataclass
    class DataReplacement(BaseOperation):
        """
        Operation that replaces existing datafiles in the dataset.
        """

        replacements: List[LanceOperation.DataReplacementGroup]

    @dataclass
    class Project(BaseOperation):
        """
        Operation that project columns.
        Use this operator for drop column or rename/swap column.

        Attributes
        ----------
        schema: LanceSchema
            The lance schema of the new dataset.

        Examples
        --------
        Use the projece operator to swap column:

        >>> import lance
        >>> import pyarrow as pa
        >>> import pyarrow.compute as pc
        >>> from lance.schema import LanceSchema
        >>> table = pa.table({"a": [1, 2], "b": ["a", "b"], "b1": ["c", "d"]})
        >>> dataset = lance.write_dataset(table, "example")
        >>> dataset.to_table().to_pandas()
           a  b b1
        0  1  a  c
        1  2  b  d
        >>>
        >>> ## rename column `b` into `b0` and rename b1 into `b`
        >>> table = pa.table({"a": [3, 4], "b0": ["a", "b"], "b": ["c", "d"]})
        >>> lance_schema = LanceSchema.from_pyarrow(table.schema)
        >>> operation = lance.LanceOperation.Project(lance_schema)
        >>> dataset = lance.LanceDataset.commit("example", operation, read_version=1)
        >>> dataset.to_table().to_pandas()
           a b0  b
        0  1  a  c
        1  2  b  d
        """

        schema: LanceSchema

    @dataclass
    class UpdateMap:
        """
        Represents updates to a metadata map.

        Attributes
        ----------
        updates : Dict[str, Optional[str]]
            Dictionary of key-value pairs to update. None values mean delete the key.
        replace : bool
            If True, replace the entire map with the new entries.
            If False, merge the new entries with the existing map.
        """

        updates: Dict[str, Optional[str]]
        replace: bool = False

    @dataclass
    class UpdateConfig(BaseOperation):
        """
        Operation that updates dataset metadata.

        This operation can update various metadata levels:
        - Dataset configuration
        - Table metadata
        - Schema metadata
        - Field-specific metadata

        Attributes
        ----------
        config_updates : Optional[UpdateMap]
            Updates to the dataset configuration.
        table_metadata_updates : Optional[UpdateMap]
            Updates to the table metadata.
        schema_metadata_updates : Optional[UpdateMap]
            Updates to the schema metadata.
        field_metadata_updates : Optional[Dict[int, UpdateMap]]
            Updates to field metadata, keyed by field ID.
        """

        config_updates: Optional[LanceOperation.UpdateMap] = None
        table_metadata_updates: Optional[LanceOperation.UpdateMap] = None
        schema_metadata_updates: Optional[LanceOperation.UpdateMap] = None
        field_metadata_updates: Optional[Dict[int, LanceOperation.UpdateMap]] = None


@dataclass
class ColumnOrdering:
    """
    This class is used to define the column ordering rules for the `sort` operator.
    It allows users to specify the sorting order (ascending or descending)
    and the position of null values (first or last).
    """

    column_name: str
    ascending: bool = True
    nulls_first: bool = False


def _needs_substrait_placeholder(t: pa.DataType) -> bool:
    """Return True if *t* contains a type that PyArrow's substrait serializer
    cannot handle at any nesting depth.

    Three cases require a placeholder:

    * ``fixed_size_list`` — substrait has no equivalent type.
    * Arrow extension types (e.g. ``fixed_shape_tensor``) — substrait cannot
      represent them.
    * A struct whose fields carry non-``None`` metadata — substrait rejects
      such structs.  Extension types leave ``metadata={}`` on struct fields
      after a lance round-trip; ``{}`` is non-``None``.
    """
    if pa.types.is_fixed_size_list(t):
        return True
    if isinstance(t, pa.lib.BaseExtensionType):
        return True
    if pa.types.is_struct(t):
        for i in range(t.num_fields):
            f = t.field(i)
            if f.metadata is not None:
                return True
            if _needs_substrait_placeholder(f.type):
                return True
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return _needs_substrait_placeholder(t.value_type)
    return False


class ScannerBuilder:
    def __init__(self, ds: LanceDataset):
        self.ds = ds
        self._limit = None
        self._filter = None
        self._search_filter = None
        self._substrait_filter = None
        self._prefilter = False
        self._late_materialization = None
        self._blob_handling = None
        self._offset = None
        self._columns = None
        self._columns_with_transform = None
        self._nearest = None
        self._batch_size: Optional[int] = None
        self._batch_size_bytes: Optional[int] = None
        self._io_buffer_size: Optional[int] = None
        self._batch_readahead: Optional[int] = None
        self._fragment_readahead: Optional[int] = None
        self._scan_in_order = True
        self._fragments = None
        self._with_row_id = False
        self._with_row_address = False
        self._use_stats = True
        self._fast_search = False
        self._full_text_query = None
        self._use_scalar_index = None
        self._include_deleted_rows = None
        self._scan_stats_callback: Optional[Callable[[ScanStatistics], None]] = None
        self._strict_batch_size = False
        self._orderings = None
        self._disable_scoring_autoprojection = False
        self._substrait_aggregate = None

    def apply_defaults(self, default_opts: Dict[str, Any]) -> ScannerBuilder:
        for key, value in default_opts.items():
            setter = getattr(self, key, None)
            if setter is None:
                raise ValueError(f"Unknown option {key}")
            if isinstance(value, dict):
                setter(**value)
            else:
                setter(value)
        return self

    def batch_size(self, batch_size: int) -> ScannerBuilder:
        """Set the maximum number of rows per batch.

        Note: this can be overridden by ``batch_size_bytes`` or by a
        dataset-level ``batch_size_bytes`` configured via
        ``FileReaderOptions``.
        """
        self._batch_size = batch_size
        return self

    def batch_size_bytes(self, batch_size_bytes: int) -> ScannerBuilder:
        """Set the target batch size in bytes.

        When set, the scanner will produce batches whose total size in bytes
        is approximately this value, overriding the row-based ``batch_size``.

        This can also be configured at the dataset level via
        ``FileReaderOptions``.  A scanner-level setting takes precedence
        over the dataset-level default.
        """
        self._batch_size_bytes = batch_size_bytes
        return self

    def io_buffer_size(self, io_buffer_size: int) -> ScannerBuilder:
        """
        Set the I/O buffer size for the Scanner

        This is the amount of RAM that will be reserved for holding I/O received from
        storage before it is processed.  This is used to control the amount of memory
        used by the scanner.  If the buffer is full then the scanner will block until
        the buffer is processed.

        Generally this should scale with the number of concurrent I/O threads.  The
        default is 2GiB which comfortably provides enough space for somewhere between
        32 and 256 concurrent I/O threads.

        This value is not a hard cap on the amount of RAM the scanner will use.  Some
        space is used for the compute (which can be controlled by the batch size) and
        Lance does not keep track of memory after it is returned to the user.

        Currently, if there is a single batch of data which is larger than the io buffer
        size then the scanner will deadlock.  This is a known issue and will be fixed in
        a future release.

        This parameter is only used when reading v2 files
        """
        self._io_buffer_size = io_buffer_size
        return self

    def batch_readahead(self, nbatches: Optional[int] = None) -> ScannerBuilder:
        """
        This parameter is ignored when reading v2 files
        """
        if nbatches is not None and int(nbatches) < 0:
            raise ValueError("batch_readahead must be non-negative")
        self._batch_readahead = nbatches
        return self

    def fragment_readahead(self, nfragments: Optional[int] = None) -> ScannerBuilder:
        if nfragments is not None and int(nfragments) < 0:
            raise ValueError("fragment_readahead must be non-negative")
        self._fragment_readahead = nfragments
        return self

    def scan_in_order(self, scan_in_order: bool = True) -> ScannerBuilder:
        """
        Whether to scan the dataset in order of fragments and batches.

        If set to False, the scanner may read fragments concurrently and yield
        batches out of order. This may improve performance since it allows more
        concurrency in the scan, but can also use more memory.

        This parameter is ignored when using v2 files.  In the v2 file format
        there is no penalty to scanning in order and so all scans will scan in
        order.
        """
        self._scan_in_order = scan_in_order
        return self

    def limit(self, n: Optional[int] = None) -> ScannerBuilder:
        if n is not None and int(n) < 0:
            raise ValueError("Limit must be non-negative")
        self._limit = n
        return self

    def offset(self, n: Optional[int] = None) -> ScannerBuilder:
        if n is not None and int(n) < 0:
            raise ValueError("Offset must be non-negative")
        self._offset = n
        return self

    def columns(
        self, cols: Optional[Union[List[str], Dict[str, str]]] = None
    ) -> ScannerBuilder:
        if cols is None:
            self._columns = None
        elif isinstance(cols, dict):
            self._columns_with_transform = list(cols.items())
        elif isinstance(cols, list):
            self._columns = cols
        else:
            raise TypeError(
                f"columns must be a list or dict[name, expression], got {type(cols)}"
            )
        return self

    def filter(
        self,
        filter: Union[
            str, pa.compute.Expression, FullTextQuery, VectorSearchQuery, dict
        ],
    ) -> ScannerBuilder:
        """
        Add a filter to the scanner.

        :param filter: The filter to apply.  This can be a string, a pyarrow compute
            expression, a FullTextQuery, a VectorSearchQuery, or a dictionary.

        :return: The scanner builder.
        """
        if isinstance(filter, FullTextQuery):
            self._search_filter = PySearchFilter.from_full_text_query(filter.inner)
        elif isinstance(filter, VectorSearchQuery):
            self._search_filter = PySearchFilter.from_vector_search_query(filter.inner)
        elif isinstance(filter, str):
            self._filter = filter
        elif isinstance(filter, pa.compute.Expression):
            try:
                from pyarrow.substrait import serialize_expressions

                fields_without_lists = []
                counter = 0
                # Pyarrow cannot handle certain types when converting to
                # Substrait (e.g. fixed_size_list at any nesting depth, or
                # struct fields with non-None metadata left by extension types
                # after a lance round-trip).  We replace any top-level field
                # whose type tree contains such a type with an int8 placeholder
                # so that ordinal field references in the filter remain correct.
                for field in self.ds.schema:
                    if _needs_substrait_placeholder(field.type):
                        pos = counter
                        counter += 1
                        fields_without_lists.append(
                            pa.field(f"__unlikely_name_placeholder_{pos}", pa.int8())
                        )
                    else:
                        fields_without_lists.append(field)
                        # Serialize the pyarrow compute expression toSubstrait and use
                        # that as a filter.
                        counter += 1
                scalar_schema = pa.schema(fields_without_lists)
                substrait_filter = serialize_expressions(
                    [filter], ["my_filter"], scalar_schema
                )
                if isinstance(substrait_filter, memoryview):
                    self._substrait_filter = substrait_filter.tobytes()
                else:
                    try:
                        self._substrait_filter = substrait_filter.to_pybytes()
                    except AttributeError:
                        raise TypeError(
                            "serialize_expressions returned unexpected"
                            f"type {type(substrait_filter)}"
                        )
            except ImportError:
                # serialize_expressions was introduced in pyarrow 14.  Fallback to
                # stringifying the expression if pyarrow is too old
                self._filter = str(filter)
        else:
            expr_filter = filter.get("expr_filter")
            if expr_filter is not None:
                self.filter(expr_filter)

            search_filter = filter.get("search_filter")
            if search_filter is not None:
                self.filter(search_filter)

        return self

    def prefilter(self, prefilter: bool) -> ScannerBuilder:
        self._prefilter = prefilter
        return self

    def with_row_id(self, with_row_id: bool = True) -> ScannerBuilder:
        """Enable returns with row IDs."""
        self._with_row_id = with_row_id
        return self

    def with_row_address(self, with_row_address: bool = True) -> ScannerBuilder:
        """
        Enables returns with row addresses.

        Row addresses are a unique but unstable identifier for each row in the
        dataset that consists of the fragment id (upper 32 bits) and the row
        offset in the fragment (lower 32 bits).  Row IDs are generally preferred
        since they do not change when a row is modified or compacted.  However,
        row addresses may be useful in some advanced use cases.
        """
        self._with_row_address = with_row_address
        return self

    def late_materialization(
        self, late_materialization: bool | List[str]
    ) -> ScannerBuilder:
        self._late_materialization = late_materialization
        return self

    def blob_handling(self, blob_handling: Optional[str]) -> ScannerBuilder:
        if blob_handling is None:
            self._blob_handling = None
            return self

        allowed = {"all_binary", "blobs_descriptions", "all_descriptions"}
        if blob_handling not in allowed:
            raise ValueError(
                f"Invalid blob_handling: {blob_handling}. Expected one of: "
                + ", ".join(sorted(allowed))
            )
        self._blob_handling = blob_handling
        return self

    def use_stats(self, use_stats: bool = True) -> ScannerBuilder:
        """
        Enable use of statistics for query planning.

        Disabling statistics is used for debugging and benchmarking purposes.
        This should be left on for normal use.
        """
        self._use_stats = use_stats
        return self

    def use_scalar_index(self, use_scalar_index: bool = True) -> ScannerBuilder:
        """
        Set whether scalar indices should be used in a query

        Scans will use scalar indices, when available, to optimize queries with filters.
        However, in some corner cases, scalar indices may make performance worse.  This
        parameter allows users to disable scalar indices in these cases.
        """
        self._use_scalar_index = use_scalar_index
        return self

    def with_fragments(
        self, fragments: Optional[Iterable[LanceFragment]]
    ) -> ScannerBuilder:
        if fragments is not None:
            inner_fragments = []
            for f in fragments:
                if isinstance(f, LanceFragment):
                    inner_fragments.append(f._fragment)
                else:
                    raise TypeError(
                        f"fragments must be an iterable of LanceFragment. "
                        f"Got {type(f)} instead."
                    )
            fragments = inner_fragments

        self._fragments = fragments
        return self

    def nearest(
        self,
        column: str,
        q: QueryVectorLike,
        k: Optional[int] = None,
        metric: Optional[str] = None,
        nprobes: Optional[int] = None,
        minimum_nprobes: Optional[int] = None,
        maximum_nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        use_index: bool = True,
        ef: Optional[int] = None,
        distance_range: Optional[tuple[Optional[float], Optional[float]]] = None,
    ) -> ScannerBuilder:
        self._nearest = _build_vector_search_query(
            column,
            q,
            dataset=self.ds,
            k=k,
            metric=metric,
            nprobes=nprobes,
            minimum_nprobes=minimum_nprobes,
            maximum_nprobes=maximum_nprobes,
            refine_factor=refine_factor,
            use_index=use_index,
            ef=ef,
            distance_range=distance_range,
        )
        return self

    def fast_search(self, flag: bool) -> ScannerBuilder:
        """Enable fast search, which only perform search on the indexed data.

        Users can use `Table::optimize()` or `create_index()` to include the new data
        into index, thus make new data searchable.
        """
        self._fast_search = flag
        return self

    def include_deleted_rows(self, flag: bool) -> ScannerBuilder:
        """Include deleted rows

        Rows which have been deleted, but are still present in the fragment, will be
        returned.  These rows will have all columns (except _rowaddr) set to null
        """
        self._include_deleted_rows = flag
        return self

    def full_text_search(
        self,
        query: str | FullTextQuery,
        columns: Optional[List[str]] = None,
    ) -> ScannerBuilder:
        """
        Filter rows by full text searching. *Experimental API*,
        may remove it after we support to do this within `filter` SQL-like expression

        Must create inverted index on the given column before searching,

        Parameters
        ----------
        query : str | Query
            If str, the query string to search for, a match query would be performed.
            If Query, the query object to search for,
            and the `columns` parameter will be ignored.
        columns : list of str, optional
            The columns to search in. If None, search in all indexed columns.
        """
        if isinstance(query, FullTextQuery):
            self._full_text_query = query.inner
        else:
            self._full_text_query = {
                "query": query,
                "columns": columns,
            }
        return self

    def scan_stats_callback(
        self, callback: Callable[[ScanStatistics], None]
    ) -> ScannerBuilder:
        """
        Set a callback function that will be called with the scan statistics after the
        scan is complete.  Errors raised by the callback will be logged but not
        re-raised.
        """
        self._scan_stats_callback = callback
        return self

    def strict_batch_size(self, strict_batch_size: bool = False) -> ScannerBuilder:
        """
        If True, then all batches except the last batch will have exactly
        `batch_size` rows.
        By default, it is false.
        If this is true then small batches will need to be merged together
        which will require a data copy and incur a (typically very small)
        performance penalty.
        """
        self._strict_batch_size = strict_batch_size
        return self

    def order_by(self, orderings: Optional[list[ColumnOrdering]]) -> ScannerBuilder:
        if orderings is not None:
            inner_orderings = []
            for order in orderings:
                if isinstance(order, ColumnOrdering):
                    inner_orderings.append(order)
                else:
                    raise TypeError(
                        f"orderings must be a list of ColumnOrdering. "
                        f"Got {type(order)} instead."
                    )
            orderings = inner_orderings
        self._orderings = orderings
        return self

    def disable_scoring_autoprojection(self, disable: bool = True) -> ScannerBuilder:
        self._disable_scoring_autoprojection = disable
        return self

    def substrait_aggregate(self, aggregate: bytes) -> ScannerBuilder:
        """
        Set a Substrait aggregate expression for the scanner.

        Parameters
        ----------
        aggregate : bytes
            The serialized Substrait Aggregate plan bytes.

        Returns
        -------
        ScannerBuilder
            This builder for method chaining.
        """
        self._substrait_aggregate = aggregate
        return self

    def to_scanner(self) -> LanceScanner:
        scanner = self.ds._ds.scanner(
            self._columns,
            self._columns_with_transform,
            self._filter,
            self._search_filter,
            self._prefilter,
            self._limit,
            self._offset,
            self._nearest,
            self._batch_size,
            self._batch_size_bytes,
            self._io_buffer_size,
            self._batch_readahead,
            self._fragment_readahead,
            self._scan_in_order,
            self._fragments,
            self._with_row_id,
            self._with_row_address,
            self._use_stats,
            self._substrait_filter,
            self._fast_search,
            self._full_text_query,
            self._late_materialization,
            self._blob_handling,
            self._use_scalar_index,
            self._include_deleted_rows,
            self._scan_stats_callback,
            self._strict_batch_size,
            self._orderings,
            self._disable_scoring_autoprojection,
            self._substrait_aggregate,
        )
        return LanceScanner(scanner, self.ds, _snapshot_scanner_builder(self))


class LanceScanner(pa.dataset.Scanner):
    def __init__(
        self,
        scanner: _Scanner,
        dataset: LanceDataset,
        snapshot: Optional[Dict[str, Any]] = None,
    ):
        self._scanner = scanner
        self._ds = dataset
        self._snapshot = snapshot

    def to_table(self) -> pa.Table:
        """
        Read the data into memory and return a pyarrow Table.
        """
        return self.to_reader().read_all()

    def to_reader(self) -> pa.RecordBatchReader:
        return self._scanner.to_pyarrow()

    def to_batches(self) -> Iterator[RecordBatch]:
        yield from self.to_reader()

    def to_pandas(
        self, *, blob_mode: str = _BLOB_PANDAS_MODE_LAZY, **kwargs: Any
    ) -> "pd.DataFrame":
        """Read the scan results into a :py:class:`pandas.DataFrame`.

        ``blob_mode`` is pandas-specific and does not replace Arrow's
        ``blob_handling`` setting used by :meth:`to_table`.
        """
        blob_mode = _normalize_blob_pandas_mode(blob_mode)
        schema = self.projected_schema
        blob_columns = _blob_columns_in_schema(schema)
        if not blob_columns or blob_mode == _BLOB_PANDAS_MODE_DESCRIPTIONS:
            return self.to_table().to_pandas(**kwargs)

        if self._snapshot is None:
            raise NotImplementedError(
                "blob-aware to_pandas requires a scanner created from the Python API"
            )

        snapshot = dict(self._snapshot)
        if blob_mode == _BLOB_PANDAS_MODE_BYTES:
            snapshot["_blob_handling"] = "all_binary"
            return (
                _scanner_from_snapshot(self._ds, snapshot)
                .to_table()
                .to_pandas(**kwargs)
            )

        blob_sources = _blob_column_sources(schema, self._snapshot, self._ds.schema)
        snapshot["_with_row_address"] = True
        snapshot["_blob_handling"] = "blobs_descriptions"
        table = _scanner_from_snapshot(self._ds, snapshot).to_table()

        requested_rowaddr = bool(self._snapshot.get("_with_row_address", False))
        if _BLOB_ROW_ADDR_COLUMN not in table.schema.names:
            raise RuntimeError("blob-aware to_pandas expected _rowaddr in scan results")

        row_addrs = table.column(_BLOB_ROW_ADDR_COLUMN).to_pylist()
        columns_to_drop = [name for name in blob_columns if name in table.schema.names]
        if not requested_rowaddr:
            columns_to_drop.append(_BLOB_ROW_ADDR_COLUMN)
        non_blob_table = (
            table.drop_columns(columns_to_drop) if columns_to_drop else table
        )
        if non_blob_table.num_columns == 0:
            dataframe = pd.DataFrame(index=range(table.num_rows))
        else:
            dataframe = non_blob_table.to_pandas(**kwargs)

        output_names = [field.name for field in schema]
        for index, name in enumerate(output_names):
            if name not in blob_columns:
                continue

            descriptions = table.column(name).to_pylist()
            non_null_positions = [
                pos
                for pos, description in enumerate(descriptions)
                if not _is_null_blob_description(description)
            ]
            non_null_addrs = [row_addrs[pos] for pos in non_null_positions]
            blob_files = (
                self._ds.take_blobs(blob_sources[name], addresses=non_null_addrs)
                if non_null_addrs
                else []
            )
            blob_iter = iter(blob_files)
            values = []
            for description in descriptions:
                if _is_null_blob_description(description):
                    values.append(None)
                    continue
                values.append(next(blob_iter))
            dataframe.insert(index, name, values)

        return dataframe

    @property
    def projected_schema(self) -> Schema:
        return self._scanner.schema

    @staticmethod
    def from_dataset(*args, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("from dataset")

    @staticmethod
    def from_fragment(*args, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("from fragment")

    @staticmethod
    def from_batches(*args, **kwargs):
        """
        Not implemented
        """
        raise NotImplementedError("from batches")

    @property
    def dataset_schema(self) -> Schema:
        """The schema with which batches will be read from fragments."""
        return self._ds.schema

    def scan_batches(self):
        """
        Consume a Scanner in record batches with corresponding fragments.

        Returns
        -------
        record_batches : iterator of TaggedRecordBatch
        """
        lst = []
        reader = self.to_reader()
        while True:
            batch = reader.read_next_batch()
            if batch is None:
                reader.close()
                break
            lst.append(batch)
        return lst

    def take(self, indices):
        """
        Not implemented
        """
        raise NotImplementedError("take")

    def head(self, num_rows):
        """
        Load the first N rows of the dataset.

        Parameters
        ----------
        num_rows : int
            The number of rows to load.

        Returns
        -------
        Table
        """
        return self.to_table()[:num_rows]

    def count_rows(self):
        """Count rows matching the scanner filter.

        Returns
        -------
        count : int

        """
        return self._scanner.count_rows()

    def explain_plan(self, verbose=False) -> str:
        """Return the execution plan for this scanner.

        Parameters
        ----------
        verbose : bool, default False
            Use a verbose output format.

        Returns
        -------
        plan : str
        """

        return self._scanner.explain_plan(verbose=verbose)

    def analyze_plan(self) -> str:
        """Execute the plan for this scanner and display with runtime metrics.

        Parameters
        ----------
        verbose : bool, default False
            Use a verbose output format.

        Returns
        -------
        plan : str
        """

        return self._scanner.analyze_plan()


class DatasetOptimizer:
    def __init__(self, dataset: LanceDataset):
        self._dataset = dataset

    def compact_files(
        self,
        *,
        target_rows_per_fragment: Optional[int] = None,
        max_rows_per_group: Optional[int] = None,
        max_bytes_per_file: Optional[int] = None,
        materialize_deletions: Optional[bool] = None,
        materialize_deletions_threshold: Optional[float] = None,
        defer_index_remap: Optional[bool] = None,
        num_threads: Optional[int] = None,
        batch_size: Optional[int] = None,
        compaction_mode: Optional[
            Literal["reencode", "try_binary_copy", "force_binary_copy"]
        ] = None,
        binary_copy_read_batch_bytes: Optional[int] = None,
    ) -> CompactionMetrics:
        """Compacts small files in the dataset, reducing total number of files.

        This does a few things:
         * Removes deleted rows from fragments
         * Removes dropped columns from fragments
         * Merges small fragments into larger ones

        This method preserves the insertion order of the dataset. This may mean
        it leaves small fragments in the dataset if they are not adjacent to
        other fragments that need compaction. For example, if you have fragments
        with row counts 5 million, 100, and 5 million, the middle fragment will
        not be compacted because the fragments it is adjacent to do not need
        compaction.

        Default values for these options can be stored in the dataset manifest
        config using keys prefixed with ``lance.compaction.``. For example,
        setting the config key ``lance.compaction.target_rows_per_fragment`` to
        ``"500000"`` will use 500,000 as the default target rows per fragment.
        Explicitly provided parameters take precedence over manifest config
        values, which in turn take precedence over hardcoded defaults.

        Supported config keys: ``lance.compaction.target_rows_per_fragment``,
        ``lance.compaction.max_rows_per_group``,
        ``lance.compaction.max_bytes_per_file``,
        ``lance.compaction.materialize_deletions``,
        ``lance.compaction.materialize_deletions_threshold``,
        ``lance.compaction.defer_index_remap``,
        ``lance.compaction.batch_size``,
        ``lance.compaction.compaction_mode``,
        ``lance.compaction.binary_copy_read_batch_bytes``.

        Parameters
        ----------
        target_rows_per_fragment: int, optional
            The target number of rows per fragment. This is the number of rows
            that will be in each fragment after compaction. If not specified,
            uses the manifest config value, or 1024*1024.
        max_rows_per_group: int, optional
            Max number of rows per group. This does not affect which fragments
            need compaction, but does affect how they are re-written if selected.
            If not specified, uses the manifest config value, or 1024.

            This setting only affects datasets using the legacy storage format.
            The newer format does not require row groups.
        max_bytes_per_file: Optional[int], default None
            Max number of bytes in a single file.  This does not affect which
            fragments need compaction, but does affect how they are re-written if
            selected.  If this value is too small you may end up with fragments
            that are smaller than `target_rows_per_fragment`.

            The default will use the default from ``write_dataset``.
        materialize_deletions: bool, optional
            Whether to compact fragments with soft deleted rows so they are no
            longer present in the file. If not specified, uses the manifest
            config value, or True.
        materialize_deletions_threshold: float, optional
            The fraction of original rows that are soft deleted in a fragment
            before the fragment is a candidate for compaction. If not specified,
            uses the manifest config value, or 0.1.
        defer_index_remap: bool, optional
            Whether to defer index remapping during compaction. If not specified,
            uses the manifest config value, or False.
        num_threads: int, optional
            The number of threads to use when performing compaction. If not
            specified, defaults to the number of cores on the machine.
        batch_size: int, optional
            The batch size to use when scanning input fragments.  You may want
            to reduce this if you are running out of memory during compaction.

            The default will use the same default from ``scanner``.
        compaction_mode: str, optional
            The compaction mode. Valid values:

            - ``"reencode"``: Decode and re-encode data (default).
            - ``"try_binary_copy"``: Try binary copy if fragments are
              compatible, fall back to reencode otherwise.
            - ``"force_binary_copy"``: Use binary copy or fail if fragments
              are not compatible.
        binary_copy_read_batch_bytes: int, optional
            The batch size in bytes for reading during binary copy operations.
            Controls how much data is read at once when performing binary copy.
            Defaults to 16MB.

        Returns
        -------
        CompactionMetrics
            Metrics about the compaction process

        See Also
        --------
        lance.optimize.Compaction
        """
        opts = {
            k: v
            for k, v in dict(
                target_rows_per_fragment=target_rows_per_fragment,
                max_rows_per_group=max_rows_per_group,
                max_bytes_per_file=max_bytes_per_file,
                materialize_deletions=materialize_deletions,
                materialize_deletions_threshold=materialize_deletions_threshold,
                defer_index_remap=defer_index_remap,
                num_threads=num_threads,
                batch_size=batch_size,
                compaction_mode=compaction_mode,
                binary_copy_read_batch_bytes=binary_copy_read_batch_bytes,
            ).items()
            if v is not None
        }
        return Compaction.execute(self._dataset, opts)

    def optimize_indices(self, **kwargs):
        """Optimizes index performance.

        As new data arrives it is not added to existing indexes automatically.
        When searching we need to perform an indexed search of the old data plus
        an expensive unindexed search on the new data.  As the amount of new
        unindexed data grows this can have an impact on search latency.
        This function will add the new data to existing indexes, restoring the
        performance.  This function does not retrain the index, it only assigns
        the new data to existing partitions.  This means an update is much quicker
        than retraining the entire index but may have less accuracy (especially
        if the new data exhibits new patterns, concepts, or trends)

        Parameters
        ----------
        num_indices_to_merge: optional, int, default None
            The number of indices to merge.
            If set to 0, new delta index will be created.
        index_names: List[str], default None
            The names of the indices to optimize.
            If None, all indices will be optimized.
        retrain: bool, default False, deprecated
            Whether to retrain the whole index.
            If true, the index will be retrained based on the current data,
            `num_indices_to_merge` will be ignored,
            and all indices will be merged into one.

            This is useful when the data distribution has changed significantly,
            and we want to retrain the index to improve the search quality.
            This would be faster than re-create the index from scratch.
        """
        self._dataset._ds.optimize_indices(**kwargs)

    def enable_auto_cleanup(self, auto_cleanup_config: AutoCleanupConfig, **kwargs):
        """Enable autocleaning for an existing dataset.

        Parameters
        ----------
        auto_cleanup_config: AutoCleanupConfig
            Config options for automatic cleanup of the dataset.
            If set, dataset's old versions will be automatically
            cleaned up according to this parameter.
        """
        self._dataset._ds.update_config(
            {
                "lance.auto_cleanup.interval": str(auto_cleanup_config["interval"]),
                "lance.auto_cleanup.older_than": f"{auto_cleanup_config['older_than_seconds']}s",  # noqa E501
            }
        )

    def disable_auto_cleanup(self, **kwargs):
        """Disable autocleaning via delete related keys."""
        self._dataset._ds.delete_config_keys(
            ["lance.auto_cleanup.interval", "lance.auto_cleanup.older_than"]
        )


class Tags:
    """
    Dataset tag manager.
    """

    def __init__(self, dataset: _Dataset):
        self._ds = dataset

    def list(self) -> dict[str, Tag]:
        """
        List all dataset tags.

        Returns
        -------
        dict[str, Tag]
            A dictionary mapping tag names to version numbers.
        """
        return self._ds.tags()

    def get_version(self, tag: str) -> Optional[int]:
        """
        Get the version of a specific tag by name.

        Parameters
        ----------
        tag: str
            The name of the tag to retrieve.

        Returns
        -------
        int or None
            The version number of the tag if it exists, otherwise None.
        """
        return self._ds.get_version(tag)

    def list_ordered(self, order: Optional[str] = None) -> list[str, Tag]:
        """
        List all dataset tags.

        Parameters
        ----------
        order: str, optional
            The order in which to return the tags.
            "asc" or "desc" can be used to specify the order explicitly.
            default 'desc'.

        Returns
        -------
        list[str, Tag]
            An ordered list of tuples mapping tag names to its `Tag` metadata.
        """
        return self._ds.tags_ordered(order)

    def create(
        self,
        tag: str,
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]] = None,
    ) -> None:
        """
        Create a tag for a given dataset version.

        Parameters
        ----------
        tag: str,
            The name of the tag to create. This name must be unique among all tag
            names for the dataset.
        reference : int, str or Tuple[Optional[str], Optional[int]]
            An integer specifies a version number in the current branch; a string
            specifies a tag name; a Tuple[Optional[str], Optional[int]] specifies
            a version number in a specified branch. (None, None) means the latest
            version_number on the main branch.
        """
        self._ds.create_tag(tag, reference)

    def delete(self, tag: str) -> None:
        """
        Delete tag from the dataset.

        Parameters
        ----------
        tag: str,
            The name of the tag to delete.

        """
        self._ds.delete_tag(tag)

    def update(
        self,
        tag: str,
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]] = None,
    ) -> None:
        """
        Update tag to a new version.

        Parameters
        ----------
        tag: str,
            The name of the tag to update.
        reference : int, str or Tuple[Optional[str], Optional[int]]
            An integer specifies a version number in the current branch; a string
            specifies a tag name; a Tuple[Optional[str], Optional[int]] specifies
            a version number in a specified branch. (None, None) means the latest
            version_number on the main branch.
        """
        self._ds.update_tag(tag, reference)


class Branches:
    """
    Dataset branch manager.
    """

    def __init__(self, dataset: _Dataset):
        self._ds = dataset

    def list(self) -> dict[str, Branch]:
        """
        List all dataset branches.

        Returns
        -------
        dict[str, Branch]
            A dictionary mapping branch names to branch metadata.
        """
        return self._ds.branches()

    def list_ordered(self, order: Optional[str] = None) -> List[Tuple[str, Branch]]:
        """
        List all dataset branches ordered by parent version.
        """
        return self._ds.branches_ordered(order)

    def delete(self, branch: str) -> None:
        """
        Delete a branch.
        """
        self._ds.delete_branch(branch)


@dataclass
class FieldStatistics:
    """Statistics about a field in the dataset"""

    id: int  #: id of the field
    bytes_on_disk: int  #: (possibly compressed) bytes on disk used to store the field


@dataclass
class DataStatistics:
    """Statistics about the data in the dataset"""

    fields: FieldStatistics  #: Statistics about the fields in the dataset


class DatasetStats(TypedDict):
    num_deleted_rows: int
    num_fragments: int
    num_small_files: int


class LanceStats:
    """
    Statistics about a LanceDataset.
    """

    def __init__(self, dataset: _Dataset):
        self._ds = dataset

    def dataset_stats(self, max_rows_per_group: int = 1024) -> DatasetStats:
        """
        Statistics about the dataset.
        """
        return {
            "num_deleted_rows": self._ds.count_deleted_rows(),
            "num_fragments": self._ds.count_fragments(),
            "num_small_files": self._ds.num_small_files(max_rows_per_group),
        }

    def index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Statistics about an index.

        Parameters
        ----------
        index_name: str
            The name of the index to get statistics for.
        """
        index_stats = json.loads(self._ds.index_statistics(index_name))
        return index_stats

    def data_stats(self) -> DataStatistics:
        """
        Statistics about the data in the dataset.
        """
        return self._ds.data_stats()


def write_dataset(
    data_obj: ReaderLike,
    uri: Optional[Union[str, Path, LanceDataset]] = None,
    schema: Optional[pa.Schema] = None,
    mode: str = "create",
    *,
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
    max_bytes_per_file: int = 90 * 1024 * 1024 * 1024,
    commit_lock: Optional[CommitLock] = None,
    progress: Optional[FragmentWriteProgress] = None,
    storage_options: Optional[Dict[str, str]] = None,
    data_storage_version: Optional[
        Literal["stable", "2.0", "2.1", "2.2", "2.3", "next", "legacy", "0.1"]
    ] = None,
    use_legacy_format: Optional[bool] = None,
    enable_v2_manifest_paths: bool = True,
    enable_stable_row_ids: bool = False,
    auto_cleanup_options: Optional[AutoCleanupConfig] = None,
    commit_message: Optional[str] = None,
    transaction_properties: Optional[Dict[str, str]] = None,
    initial_bases: Optional[List[DatasetBasePath]] = None,
    target_bases: Optional[List[str]] = None,
    external_blob_mode: Literal["reference", "ingest"] = "reference",
    allow_external_blob_outside_bases: bool = False,
    blob_pack_file_size_threshold: Optional[int] = None,
    namespace_client: Optional[LanceNamespace] = None,
    table_id: Optional[List[str]] = None,
) -> LanceDataset:
    """Write a given data_obj to the given uri

    Parameters
    ----------
    data_obj: Reader-like
        The data to be written. Acceptable types are:
        - Pandas DataFrame, Pyarrow Table, Dataset, Scanner, or RecordBatchReader
        - Huggingface dataset
    uri: str, Path, LanceDataset, or None
        Where to write the dataset to (directory). If a LanceDataset is passed,
        the session will be reused.
        Either `uri` or (`namespace_client` + `table_id`) must be provided.
    schema: Schema, optional
        If specified and the input is a pandas DataFrame, use this schema
        instead of the default pandas to arrow table conversion.
    mode: str
        **create** - create a new dataset (raises if uri already exists).
        **overwrite** - create a new snapshot version
        **append** - create a new version that is the concat of the input and the
        latest version, or a new dataset if uri doesn't exist.
    max_rows_per_file: int, default 1024 * 1024
        The max number of rows to write before starting a new file
    max_rows_per_group: int, default 1024
        The max number of rows before starting a new group (in the same file)
    max_bytes_per_file: int, default 90 * 1024 * 1024 * 1024
        The max number of bytes to write before starting a new file. This is a
        soft limit. This limit is checked after each group is written, which
        means larger groups may cause this to be overshot meaningfully. This
        defaults to 90 GB, since we have a hard limit of 100 GB per file on
        object stores.
    commit_lock : CommitLock, optional
        A custom commit lock.  Only needed if your object store does not support
        atomic commits.  See the user guide for more details.
    progress: FragmentWriteProgress, optional
        *Experimental API*. Progress tracking for writing the fragment. Pass
        a custom class that defines hooks to be called when each fragment is
        starting to write and finishing writing.
    storage_options : optional, dict
        Extra options that make sense for a particular storage connection. This is
        used to store connection parameters like credentials, endpoint, etc.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read.  The default (None)
        will use the latest stable version.  See the user guide for more details.
    use_legacy_format : optional, bool, default None
        Deprecated method for setting the data storage version. Use the
        `data_storage_version` parameter instead.
    enable_v2_manifest_paths : bool, optional
        If True, and this is a new dataset, uses the new V2 manifest paths.
        These paths provide more efficient opening of datasets with many
        versions on object stores. This parameter has no effect if the dataset
        already exists. To migrate an existing dataset, instead use the
        :meth:`LanceDataset.migrate_manifest_paths_v2` method. Default is True.
    enable_stable_row_ids : bool, optional
        Experimental parameter: if set to true, the writer will use stable row ids.
        These row ids are stable after compaction operations, but not after updates.
        This makes compaction more efficient, since with stable row ids no
        secondary indices need to be updated to point to new row ids.
    auto_cleanup_options: optional, AutoCleanupConfig
        Config options for automatic cleanup of the dataset.
        If set, and this is a new dataset, old dataset versions will be automatically
        cleaned up according to this parameter.
        To add autocleaning to an existing dataset, use Dataset::update_config to set
        lance.auto_cleanup.interval and lance.auto_cleanup.older_than.
        Both parameters must be set to invoke autocleaning.
        If you do not set this parameter(default behavior),
        then no autocleaning will be performed.
        Note: this option only takes effect when creating a new dataset,
        it has no effect on existing datasets.
    commit_message: str, optional
        A message to associate with this commit. This message will be stored in the
        dataset's metadata and can be retrieved using read_transaction().
    transaction_properties: dict of str to str, optional
        Custom properties to associate with this transaction.
        Properties will be stored in the dataset's transaction
        and can be retrieved using read_transaction().
        If both `commit_message` and `properties` are provided, `commit_message` will
        override any "lance.commit.message" key in `properties`.
    initial_bases: list of DatasetBasePath, optional
        New base paths to register in the manifest. Only used in **CREATE mode**.
        Cannot be specified in APPEND or OVERWRITE modes.
    target_bases: list of str, optional
        References to base paths where data should be written. Can be
        specified in all modes.

        Each string is resolved by trying to match:
        1. Base name (e.g., "primary", "archive") from registered bases
        2. Base path URI (e.g., "s3://bucket1/data")

        **CREATE mode**: References must match bases in `initial_bases`
        **APPEND/OVERWRITE modes**: References must match bases in the existing manifest
    external_blob_mode: {"reference", "ingest"}, default "reference"
        How external blob URIs are handled on write.

        - ``"reference"`` stores the URI as an external blob reference.
        - ``"ingest"`` reads the external bytes during write and stores them in
          Lance-managed storage using the normal inline / packed / dedicated
          thresholds.
    allow_external_blob_outside_bases: bool, default False
        If False, external blob URIs must map to the dataset root or a registered
        base path. If True, external blob URIs outside registered bases are allowed.
        This option only applies when ``external_blob_mode="reference"``.
    blob_pack_file_size_threshold: optional, int, default None
        Maximum size in bytes for blob v2 pack (.blob) sidecar files. When a pack
        file reaches this size, a new one is started. If not set, defaults to 1 GiB.
    namespace_client : optional, LanceNamespace
        A namespace client from which to fetch table location and storage options.
        Must be provided together with `table_id`. Cannot be used with `uri`.
        When provided, the table location will be fetched automatically from the
        namespace via describe_table(). Storage options will be automatically refreshed
        before they expire.
    table_id : optional, List[str]
        The table identifier when using a namespace (e.g., ["my_table"]).
        Must be provided together with `namespace_client`. Cannot be used with `uri`.

    Notes
    -----
    When using `namespace_client` and `table_id`:
    - The `uri` parameter is optional and will be fetched from the namespace
    - Storage options from describe_table() will be used automatically
    - Storage options provider will be created automatically for credential refresh
    - Initial storage options from describe_table() will be merged with
      any provided `storage_options`
    """
    # Validate that user provides either uri OR (namespace_client + table_id), not both
    has_uri = uri is not None
    has_namespace = namespace_client is not None or table_id is not None

    if has_uri and has_namespace:
        raise ValueError(
            "Cannot specify both 'uri' and 'namespace_client/table_id'. "
            "Please provide either 'uri' or both 'namespace_client' and 'table_id'."
        )
    elif not has_uri and not has_namespace:
        raise ValueError(
            "Must specify either 'uri' or both 'namespace_client' and 'table_id'."
        )

    # Handle namespace-based dataset writing
    if namespace_client is not None:
        if table_id is None:
            raise ValueError(
                "Both 'namespace_client' and 'table_id' must be provided together."
            )

        # Implement write_into_namespace logic in Python
        # This follows the same pattern as the Rust implementation:
        # - CREATE mode: calls namespace.declare_table()
        # - APPEND/OVERWRITE mode: calls namespace.describe_table()
        # - Both modes: create storage options provider and merge storage options

        from .namespace import (
            DeclareTableRequest,
            DescribeTableRequest,
        )

        # Determine which namespace method to call based on mode
        if mode == "create":
            declare_request = DeclareTableRequest(id=table_id, location=None)
            response = namespace_client.declare_table(declare_request)
        elif mode in ("append", "overwrite"):
            request = DescribeTableRequest(id=table_id, version=None)
            response = namespace_client.describe_table(request)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Get table location from response
        uri = response.location
        if not uri:
            raise ValueError(
                f"Namespace did not return a table location in {mode} response"
            )

        # Check if namespace manages versioning (commits go through namespace API)
        namespace_client_managed_versioning = (
            getattr(response, "managed_versioning", None) is True
        )

        # Use namespace storage options
        namespace_storage_options = response.storage_options

        # Merge namespace storage options with any existing options
        # Namespace options take precedence (same as Rust implementation)
        # Storage options provider will be created automatically in Rust
        if namespace_storage_options is not None:
            if storage_options is None:
                storage_options = dict(namespace_storage_options)
            else:
                merged_options = dict(storage_options)
                merged_options.update(namespace_storage_options)
                storage_options = merged_options
    elif table_id is not None:
        raise ValueError(
            "Both 'namespace_client' and 'table_id' must be provided together."
        )
    else:
        namespace_client_managed_versioning = False

    if use_legacy_format is not None:
        warnings.warn(
            "use_legacy_format is deprecated, use data_storage_version instead",
            DeprecationWarning,
        )
        if use_legacy_format:
            data_storage_version = "legacy"
        else:
            data_storage_version = "stable"

    reader = _coerce_reader(data_obj, schema)
    _validate_schema(reader.schema)
    # TODO add support for passing in LanceDataset and LanceScanner here

    # Merge properties and commit_message with priority to commit_message
    merged_properties = _merge_message_to_properties(
        commit_message, transaction_properties
    )

    params = {
        "mode": mode,
        "max_rows_per_file": max_rows_per_file,
        "max_rows_per_group": max_rows_per_group,
        "max_bytes_per_file": max_bytes_per_file,
        "progress": progress,
        "storage_options": storage_options,
        "data_storage_version": data_storage_version,
        "enable_v2_manifest_paths": enable_v2_manifest_paths,
        "enable_stable_row_ids": enable_stable_row_ids,
        "auto_cleanup_options": auto_cleanup_options,
        "transaction_properties": merged_properties,
        "initial_bases": initial_bases,
        "target_bases": target_bases,
        "external_blob_mode": external_blob_mode,
        "allow_external_blob_outside_bases": allow_external_blob_outside_bases,
        "blob_pack_file_size_threshold": blob_pack_file_size_threshold,
    }

    # Add namespace_client and table_id for storage options provider and managed
    # versioning. The storage options provider will be created automatically in Rust.
    if namespace_client is not None and table_id is not None:
        params["namespace_client"] = namespace_client
        params["table_id"] = table_id
        params["namespace_client_managed_versioning"] = (
            namespace_client_managed_versioning
        )

    if commit_lock:
        if not callable(commit_lock):
            raise TypeError(f"commit_lock must be a function, got {type(commit_lock)}")
        params["commit_handler"] = commit_lock

    if isinstance(uri, Path):
        uri = os.fspath(uri)
    elif isinstance(uri, LanceDataset):
        uri = uri._ds
    elif not isinstance(uri, str):
        raise TypeError(f"dest must be a str, Path, or LanceDataset. Got {type(uri)}")

    inner_ds = _write_dataset(reader, uri, params)

    ds = LanceDataset.__new__(LanceDataset)
    ds._storage_options = storage_options
    ds._namespace_client = namespace_client
    ds._table_id = table_id
    ds._namespace_client_managed_versioning = namespace_client_managed_versioning
    ds._ds = inner_ds
    ds._uri = inner_ds.uri
    ds._default_scan_options = None
    ds._read_params = None
    return ds


def _coerce_query_vector(query: QueryVectorLike) -> tuple[pa.Array, int]:
    # if the query is a multivector, convert it to pa.ListArray
    if hasattr(query, "__getitem__") and isinstance(
        query[0], (list, tuple, np.ndarray, pa.Array)
    ):
        dim = len(query[0])
        multivector_query = []
        for q in query:
            if len(q) != dim:
                raise ValueError(
                    "All query vectors must have the same length, "
                    f"but got {dim} and {len(q)}"
                )
            multivector_query.append(_coerce_query_vector(q)[0])
        query = pa.array(multivector_query, type=pa.list_(pa.float32()))
        return (query, dim)

    if isinstance(query, pa.Scalar):
        if isinstance(query, pa.ExtensionScalar):
            # If it's an extension scalar then convert to storage
            query = query.value
        if isinstance(query.type, pa.FixedSizeListType):
            query = query.values
    elif isinstance(query, (list, tuple)) or (
        _check_for_numpy(query),
        isinstance(query, np.ndarray),
    ):
        query = np.array(query).astype("float64")  # workaround for GH-608
        query = pa.FloatingPointArray.from_pandas(query, type=pa.float32())
    elif not isinstance(query, pa.Array):
        try:
            query = pa.array(query)
        except:  # noqa: E722
            raise TypeError(
                "Query vectors should be an array of floats, "
                f"got {type(query)} which we cannot coerce to a "
                "float array"
            )

    # At this point `query` should be an arrow array
    if not isinstance(query, pa.FloatingPointArray):
        if pa.types.is_integer(query.type):
            query = query.cast(pa.float32())
        else:
            raise TypeError(
                "query vector must be list-like or pa.FloatingPointArray "
                f"but received {query.type}"
            )

    return (query, len(query))


def _build_vector_search_query(
    column: str,
    q,
    *,
    dataset: Optional["LanceDataset"] = None,
    k: Optional[int] = None,
    metric: Optional[str] = None,
    nprobes: Optional[int] = None,
    minimum_nprobes: Optional[int] = None,
    maximum_nprobes: Optional[int] = None,
    refine_factor: Optional[int] = None,
    use_index: bool = True,
    ef: Optional[int] = None,
    distance_range: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> dict:
    """Configure nearest neighbor search.

    Parameters
    ----------
    column: str
        The name of the vector column to search.
    q: QueryVectorLike
        The query vector.
    k: int, optional
        The number of nearest neighbors to return.
    metric: str, optional
        The distance metric to use (e.g., "L2", "cosine", "dot", "hamming").
    nprobes: int, optional
        The number of partitions to search. Sets both minimum_nprobes and
        maximum_nprobes to the same value.
    minimum_nprobes: int, optional
        The minimum number of partitions to search.
    maximum_nprobes: int, optional
        The maximum number of partitions to search.
    refine_factor: int, optional
        The refine factor for the search.
    use_index: bool, default True
        Whether to use the index for the search.
    ef: int, optional
        The ef parameter for HNSW search.
    distance_range: tuple[Optional[float], Optional[float]], optional
        A tuple of (lower_bound, upper_bound) to filter results by distance.
        Both bounds are optional. The lower bound is inclusive and the upper
        bound is exclusive, so (0.0, 1.0) keeps distances d where
        0.0 <= d < 1.0, (None, 0.5) keeps d < 0.5, and (0.5, None) keeps d >= 0.5.

    Returns
    -------
    ScannerBuilder
        The scanner builder for method chaining.
    """
    q, q_dim = _coerce_query_vector(q)

    lance_field = dataset._ds.lance_schema.field_case_insensitive(column)
    if lance_field is None:
        raise ValueError(f"Embedding column {column} is not in the dataset")

    column_field = lance_field.to_arrow()
    column_type = column_field.type
    if hasattr(column_type, "storage_type"):
        column_type = column_type.storage_type
    if pa.types.is_fixed_size_list(column_type):
        dim = column_type.list_size
    elif pa.types.is_list(column_type) and pa.types.is_fixed_size_list(
        column_type.value_type
    ):
        dim = column_type.value_type.list_size
    else:
        raise TypeError(
            f"Query column {column} must be a vector. Got {column_field.type}."
        )

    if q_dim != dim:
        raise ValueError(
            f"Query vector size {len(q)} does not match index column size {dim}"
        )

    if k is not None and int(k) <= 0:
        raise ValueError(f"Nearest-K must be > 0 but got {k}")
    if nprobes is not None and int(nprobes) <= 0:
        raise ValueError(f"Nprobes must be > 0 but got {nprobes}")
    if minimum_nprobes is not None and int(minimum_nprobes) < 0:
        raise ValueError(f"Minimum nprobes must be >= 0 but got {minimum_nprobes}")
    if maximum_nprobes is not None and int(maximum_nprobes) < 0:
        raise ValueError(f"Maximum nprobes must be >= 0 but got {maximum_nprobes}")

    if nprobes is not None:
        if minimum_nprobes is not None or maximum_nprobes is not None:
            raise ValueError(
                "nprobes cannot be set in combination with minimum_nprobes or "
                "maximum_nprobes"
            )
        else:
            minimum_nprobes = nprobes
            maximum_nprobes = nprobes
    if (
        minimum_nprobes is not None
        and maximum_nprobes is not None
        and minimum_nprobes > maximum_nprobes
    ):
        raise ValueError("minimum_nprobes must be <= maximum_nprobes")
    if refine_factor is not None and int(refine_factor) < 1:
        raise ValueError(f"Refine factor must be 1 or more got {refine_factor}")
    if ef is not None and int(ef) <= 0:
        # `ef` should be >= `k`, but `k` could be None so we can't check it here
        # the rust code will check it
        raise ValueError(f"ef must be > 0 but got {ef}")

    if distance_range is not None:
        if len(distance_range) != 2:
            raise ValueError(
                "distance_range must be a tuple of (lower_bound, upper_bound)"
            )

    return {
        "column": column,
        "q": q,
        "k": k,
        "metric": metric,
        "minimum_nprobes": minimum_nprobes,
        "maximum_nprobes": maximum_nprobes,
        "refine_factor": refine_factor,
        "use_index": use_index,
        "ef": ef,
        "distance_range": distance_range,
    }


def _validate_schema(schema: pa.Schema):
    """
    Make sure the metadata is valid utf8
    """
    if schema.metadata is not None:
        _validate_metadata(schema.metadata)


def _validate_metadata(metadata: dict):
    """
    Make sure the metadata values are valid utf8 (can be nested)

    Raises ValueError if not valid utf8
    """
    for k, v in metadata.items():
        if isinstance(v, bytes):
            try:
                v.decode("utf8")
            except UnicodeDecodeError:
                raise ValueError(
                    f"Metadata key {k} is not valid utf8. "
                    "Consider base64 encode for generic binary metadata."
                )
        elif isinstance(v, dict):
            _validate_metadata(v)


def _merge_message_to_properties(
    commit_message: Optional[str],
    properties: Optional[Dict[str, str]],
) -> Optional[Dict[str, str]]:
    """
    handles merging commit_message into properties,
    prioritizing the commit_message if properties also has the same key
    """
    if properties is None and commit_message is None:
        return None

    merged_properties = properties.copy() if properties else {}
    if commit_message is not None:
        merged_properties[LANCE_COMMIT_MESSAGE_KEY] = commit_message
    return merged_properties


class VectorIndexReader:
    """
    This class allows you to initialize a reader for a specific vector index,
    retrieve the number of partitions,
    access the centroids of the index,
    and read specific partitions of the index.

    Parameters
    ----------
    dataset: LanceDataset
        The dataset containing the index.
    index_name: str
        The name of the vector index to read.

    Examples
    --------
    .. code-block:: python

        import lance
        from lance.dataset import VectorIndexReader
        import numpy as np
        import pyarrow as pa
        vectors = np.random.rand(256, 2)
        data = pa.table({"vector": pa.array(vectors.tolist(),
            type=pa.list_(pa.float32(), 2))})
        dataset = lance.write_dataset(data, "/tmp/index_reader_demo")
        dataset.create_index("vector", index_type="IVF_PQ",
            num_partitions=4, num_sub_vectors=2)
        reader = VectorIndexReader(dataset, "vector_idx")
        assert reader.num_partitions() == 4
        partition = reader.read_partition(0)
        assert "_rowid" in partition.column_names

    Exceptions
    ----------
    ValueError
        If the specified index is not a vector index.
    """

    def __init__(self, dataset: LanceDataset, index_name: str):
        stats = dataset.stats.index_stats(index_name)
        self.dataset = dataset
        self.index_name = index_name
        self.stats = stats
        try:
            self.num_partitions()
        except KeyError:
            raise ValueError(f"Index {index_name} is not vector index")

    def num_partitions(self) -> int:
        """
        Returns the number of partitions in the dataset.

        Returns
        -------
        int
            The number of partitions.
        """

        return self.stats["indices"][0]["num_partitions"]

    def centroids(self) -> np.ndarray:
        """
        Returns the centroids of the index

        Returns
        -------
        np.ndarray
            The centroids of IVF
            with shape (num_partitions, dim)
        """
        # when we have more delta indices,
        # they are with the same centroids
        return np.array(
            self.dataset._ds.get_index_centroids(self.stats["indices"][0]["centroids"])
        )

    def read_partition(
        self, partition_id: int, *, with_vector: bool = False
    ) -> pa.Table:
        """
        Returns a pyarrow table for the given IVF partition

        Parameters
        ----------
        partition_id: int
            The id of the partition to read
        with_vector: bool, default False
            Whether to include the vector column in the reader,
            for IVF_PQ, the vector column is PQ codes

        Returns
        -------
        pa.Table
            A pyarrow table for the given partition,
            containing the row IDs, and quantized vectors (if with_vector is True).
        """

        if partition_id < 0 or partition_id >= self.num_partitions():
            raise IndexError(
                f"Partition id {partition_id} is out of range, "
                f"expected 0 <= partition_id < {self.num_partitions()}"
            )

        return self.dataset._ds.read_index_partition(
            self.index_name, partition_id, with_vector
        ).read_all()


class VectorSearchQuery:
    _inner: dict

    def __init__(
        self,
        column: str,
        q: QueryVectorLike,
        k: Optional[int] = None,
        metric: Optional[str] = None,
        nprobes: Optional[int] = None,
        minimum_nprobes: Optional[int] = None,
        maximum_nprobes: Optional[int] = None,
        refine_factor: Optional[int] = None,
        use_index: bool = True,
        ef: Optional[int] = None,
    ):
        self._inner = _build_vector_search_query(
            column,
            q,
            k=k,
            metric=metric,
            nprobes=nprobes,
            minimum_nprobes=minimum_nprobes,
            maximum_nprobes=maximum_nprobes,
            refine_factor=refine_factor,
            use_index=use_index,
            ef=ef,
        )

    def inner(self):
        return self._inner
