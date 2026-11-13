# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import copy
import json
import os
import random
import re
import time
import warnings
from datetime import timedelta
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
    Tuple,
    Union,
)

import pyarrow as pa
import pyarrow.dataset
from pyarrow import Schema

from lance.log import LOGGER

from ._dataset.cleanup import CleanupPolicy, KeepOne, _cleanup_policy_to_json
from ._dataset.delta import DatasetDelta, _DatasetDeltaBuilder
from ._dataset.merge_insert import MergeInsertBuilder
from ._dataset.optimize import AutoCleanupConfig, DatasetOptimizer
from ._dataset.refs import (
    Branches,
    Tags,
    Version,
    VersionRef,
    _convert_version_timestamp,
)
from ._dataset.scanner import (
    _BLOB_PANDAS_MODE_LAZY,
    ColumnOrdering,
    LanceScanner,
    ScannerBuilder,
    VectorSearchQuery,
    _serialize_expression,
)
from ._dataset.sql import SqlQueryBuilder
from ._dataset.stats import LanceStats
from ._dataset.transaction import (
    AlterColumn,
    BulkCommitResult,
    DeleteResult,
    LanceOperation,
    Transaction,
    UpdateResult,
)

# Imported at runtime, not only for the annotations below: importing it here
# is what registers `Bitmap` as a `collections.abc.MutableSet`.
from .bitmap import Bitmap  # noqa: TC001
from .blob import DEFAULT_BLOB_BUFFER_SIZE, BlobFile, _validate_buffer_size
from .dependencies import (
    _check_for_numpy,
    _check_for_torch,
    _is_pydantic_base_model_class,
    _validate_pydantic_list,
    model_to_dict,
    torch,
)
from .dependencies import numpy as np
from .dependencies import pandas as pd
from .fragment import LanceFragment
from .indices import IndexConfig, IndexSegment, SupportedDistributedIndices
from .lance import (
    CleanupExplanation,
    CleanupStats,
    DatasetBasePath,
    ExpireVersionsPlan,
    ExpireVersionsStats,
    IOStats,
    LanceSchema,
    ScanStatistics,
    _Dataset,
    _write_dataset,
    indices,
)
from .lance import __version__ as __version__
from .lance import _Session as Session
from .query import DocumentGranularity, FullTextQuery, MinHashQuery
from .types import _coerce_reader, _coerce_source, _WriteSource
from .udf import BatchUDF, normalize_transform
from .udf import BatchUDFCheckpoint as BatchUDFCheckpoint
from .udf import batch_udf as batch_udf
from .util import (
    _normalize_index_segment_ids,
    _target_partition_size_to_num_partitions,
    td_to_micros,
)

if TYPE_CHECKING:
    import uuid

    from pyarrow._compute import Expression

    from lance.namespace import LanceNamespace

    from . import mem_wal
    from ._dataset.index import Index, IndexInformation
    from .commit import CommitLock
    from .lance.indices import IndexDescription
    from .progress import FragmentWriteProgress, IndexProgress
    from .types import ReaderLike

LANCE_COMMIT_MESSAGE_KEY = "__lance_commit_message"
# Mirrors Rust's `lance::dataset::DEFAULT_COMMIT_TIMEOUT`; keep the two in sync.
_DEFAULT_COMMIT_TIMEOUT = timedelta(minutes=30)


def _resolve_blob_selection(
    ids: Optional[Union[List[int], pa.Array]],
    addresses: Optional[Union[List[int], pa.Array]],
    indices: Optional[Union[List[int], pa.Array]],
) -> Tuple[str, Union[List[int], pa.Array]]:
    if sum([bool(v is not None) for v in [ids, addresses, indices]]) != 1:
        raise ValueError("Exactly one of ids, indices, or addresses must be specified")

    if ids is not None:
        return "ids", ids
    if addresses is not None:
        return "addresses", addresses
    if indices is not None:
        return "indices", indices
    raise ValueError("Either ids, addresses, or indices must be specified")


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
        base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        uri = os.fspath(uri) if isinstance(uri, Path) else uri
        self._uri = uri
        self._storage_options = storage_options
        self._base_store_params = base_store_params

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
            base_store_params=base_store_params,
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
        base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        return cls(
            uri,
            version,
            storage_options=storage_options,
            serialized_manifest=manifest,
            default_scan_options=default_scan_options,
            read_params=read_params,
            base_store_params=base_store_params,
        )

    @classmethod
    def from_pydantic_model(
        cls,
        model_class,
        data,
        uri: Optional[Union[str, Path]] = None,
        mode: str = "create",
        **kwargs,
    ) -> "LanceDataset":
        """Create a LanceDataset from a Pydantic model class and a list of instances.

        The table name is inferred from the model class name converted to snake_case.
        The schema is inferred from the model class's field annotations, not from
        the data, so optional fields are typed correctly even if every value in a
        given batch happens to be None.

        Parameters
        ----------
        model_class : type
            A Pydantic BaseModel subclass.
        data : list
            A list of Pydantic model instances.
        uri : str or Path, optional
            The URI to write the dataset to. If not provided, the model class name
            converted to snake_case is used as the path.
        mode : str, optional
            The write mode. One of "create", "overwrite", or "append".
        **kwargs
            Additional arguments passed to write_dataset().
        """
        if not _is_pydantic_base_model_class(model_class):
            raise TypeError(
                f"`model_class` must be a Pydantic BaseModel subclass, "
                f"got {model_class!r}"
            )
        _validate_pydantic_list(data, model_class)
        if not data:
            raise ValueError(
                "`data` must be a non-empty list of Pydantic model instances."
            )
        if uri is None:
            uri = re.sub(r"(?<!^)(?=[A-Z])", "_", model_class.__name__).lower()
        from .pydantic import pydantic_to_schema

        dicts = [model_to_dict(item) for item in data]
        schema = pydantic_to_schema(model_class)
        table = pa.Table.from_pylist(dicts, schema=schema)
        return write_dataset(table, uri, mode=mode, **kwargs)

    def __reduce__(self):
        return type(self).__deserialize__, (
            self.uri,
            self._storage_options,
            self._ds.version(),
            self._ds.serialized_manifest(),
            self._default_scan_options,
            self._read_params,
            self._base_store_params,
        )

    def __getstate__(self):
        return (
            self.uri,
            self._storage_options,
            self._ds.version(),
            self._ds.serialized_manifest(),
            self._default_scan_options,
            self._read_params,
            self._base_store_params,
        )

    def __setstate__(self, state):
        # Handle backwards compatibility - state may not have read_params or
        # base_store_params.
        (
            self._uri,
            self._storage_options,
            version,
            manifest,
            default_scan_options,
            *rest,  # Capture optional read_params and base_store_params.
        ) = state
        read_params = rest[0] if rest else None
        base_store_params = rest[1] if len(rest) > 1 else None
        self._ds = _Dataset(
            self._uri,
            version,
            storage_options=self._storage_options,
            manifest=manifest,
            default_scan_options=default_scan_options,
            read_params=read_params,
            base_store_params=base_store_params,
        )
        self._default_scan_options = default_scan_options
        self._read_params = read_params
        self._base_store_params = base_store_params
        self._namespace_client = None
        self._table_id = None
        self._namespace_client_managed_versioning = False

    def __copy__(self):
        ds = LanceDataset.__new__(LanceDataset)
        ds._uri = self._uri
        ds._storage_options = self._storage_options
        ds._base_store_params = self._base_store_params
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
            Name of the branch to create. ``"main"`` is reserved for the
            default branch and cannot be used as a new branch name. A literal
            ``"."`` path segment is rejected. New names are limited to a
            portable budget of 211 bytes measured after the name is storage
            path encoded twice; longer names that already exist remain
            available for lookup, checkout, metadata updates, and deletion.
        reference: Optional[int | str | Tuple[Optional[str], Optional[int]]
            An integer specifies a version number in the current branch; a string
            specifies a tag name; a Tuple[Optional[str], Optional[int]] specifies
            a version number in a specified branch. (None, None) means the latest
            version_number on the default branch. ``("main", version)`` is an
            explicit alias for the default branch in this reference context.
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
        ds._base_store_params = self._base_store_params
        ds._namespace_client = self._namespace_client
        ds._table_id = self._table_id
        ds._namespace_client_managed_versioning = (
            self._namespace_client_managed_versioning
        )
        ds._default_scan_options = self._default_scan_options
        ds._read_params = self._read_params
        return ds

    def checkout_latest(self):
        """Check out the latest version of the current branch."""
        self._ds.checkout_latest()

    def list_indices(self) -> List[IndexInformation]:
        """
        Returns index information for all indices in the dataset.

        This method is deprecated.  Use describe_indices() instead, which returns
        richer per-index information.

        Each returned :class:`IndexInformation` describes one index segment.
        """
        warnings.warn(
            "The 'list_indices' method is deprecated. It may be removed in a future "
            "version. Use describe_indices() instead.",
            DeprecationWarning,
        )

        return [
            {
                "name": desc.name,
                "type": desc.index_type,
                "uuid": segment.uuid,
                "fields": desc.field_names,
                "version": segment.dataset_version_at_last_update,
                "fragment_ids": segment.fragment_ids,
                "base_id": segment.base_id,
            }
            for desc in self.describe_indices()
            for segment in desc.segments
        ]

    def describe_indices(self) -> List[IndexDescription]:
        """Returns index information for all indices in the dataset."""
        return self._ds.describe_indices()

    def remap_row_addrs(self, addrs: "pa.Array") -> "Optional[pa.Array]":
        """Remap row addresses across compactions still recorded in the
        fragment-reuse index. Rows a compaction dropped become null. The index
        retains only recent rounds (older ones are pruned as index remap catches
        up), so remap promptly: an address whose round was pruned is returned
        unchanged, not remapped. Returns ``None`` when there is no fragment-reuse
        index.
        """
        return self._ds.remap_row_addrs(addrs)

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
            Union[str, Expression, FullTextQuery, VectorSearchQuery, Dict[str, Any]]
        ] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[Union[Dict[str, Any], MinHashQuery]] = None,
        batch_size: Optional[int] = None,
        batch_size_bytes: Optional[int] = None,
        batch_readahead: Optional[int] = None,
        fragment_readahead: Optional[int] = None,
        scan_in_order: Optional[bool] = None,
        fragments: Optional[Iterable[LanceFragment]] = None,
        index_segments: Optional[Iterable[Union[str, uuid.UUID]]] = None,
        full_text_query: Optional[Union[str, Dict[str, Any], FullTextQuery]] = None,
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
        ignored_scalar_indices: Optional[Iterable[str]] = None,
        include_deleted_rows: Optional[bool] = None,
        scan_stats_callback: Optional[Callable[[ScanStatistics], None]] = None,
        strict_batch_size: Optional[bool] = None,
        order_by: Optional[List[Union[ColumnOrdering, str]]] = None,
        disable_scoring_autoprojection: Optional[bool] = None,
        row_addr_allowlist: Optional[bytes] = None,
        row_addr_blocklist: Optional[bytes] = None,
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
        nearest: dict or MinHashQuery, default None
            Get the rows corresponding to the K most similar vectors, or, when a
            :class:`~lance.query.MinHashQuery` is given, the ``limit`` rows most
            similar to its text under the column's MinHash LSH index (adds a
            ``_distance`` column equal to ``1 - estimated Jaccard similarity``;
            unindexed rows are scored on the fly unless ``fast_search`` is set).
            Vector search example:

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

            ``q`` may also be a 2-D array-like value, or a list of vectors, for
            fixed-size vector columns. In that case Lance runs a batch nearest-neighbor
            query, returns up to ``k`` rows for each query vector, and adds
            an Int32 non-null ``query_index`` as the first output column to identify
            the source query for each result row.
            Flattened 1-D arrays whose length is a multiple of the vector dimension are
            rejected. Datasets that already contain a ``query_index`` column cannot be
            used for batch nearest-neighbor search. When ``use_index`` is true and a
            vector index is available, each query vector is searched through the index
            path; otherwise the flat batch path is used.

        batch_size: int, default None
            The maximum number of rows per batch.  In some cases batches can be
            smaller than this size. If a byte limit is also configured, both
            limits apply and the one reached first determines the batch size.
        batch_size_bytes: int, default None
            If set, the scanner will produce batches whose total size in bytes
            is approximately this value. If ``batch_size`` is also set, both
            limits apply and the one reached first determines the batch size.
            This cannot be combined with ``strict_batch_size=True`` because
            strict row batching can merge batches beyond the byte limit.
            This can also be configured at the dataset level via
            ``FileReaderOptions``.  A scanner-level setting takes precedence
            over the dataset-level default.
        io_buffer_size: int, default None
            The maximum number of bytes to buffer from storage before applying
            backpressure. See ``ScannerBuilder.io_buffer_size`` for more information.
        batch_readahead: int, optional
            The number of batches to decode concurrently.
        fragment_readahead: int, optional
            The number of fragments whose reads may be scheduled concurrently.
            This applies even when ``scan_in_order`` is true. Set this to ``1``
            to avoid overlapping I/O from multiple fragments.
        scan_in_order: bool, default True
            Whether to return fragments and batches in order. This does not make
            storage reads sequential; use ``fragment_readahead=1`` for that. If
            false, throughput may be higher, but batches will be returned out of
            order and memory use might increase.
        fragments: iterable of LanceFragment, default None
            Restrict the search scope to these fragments; an empty iterable returns
            no rows. Vector search enforces this scope before candidate selection,
            even with ``prefilter=False``. Index segments cannot return rows outside
            it. Ordinary scans preserve the given order when ``scan_in_order=True``;
            vector search ranks results by distance.
        index_segments: iterable of str or uuid.UUID, default None
            Restrict vector index search to these segment UUIDs. When omitted, Lance
            selects a suitable logical index and its relevant segments automatically.
            With explicit ``fragments`` and ``fast_search=False``, fragments not
            covered by the selected segments are searched with flat KNN, including
            fragments covered only by an unselected segment. Without ``fragments``,
            indexed search searches only the selected segments and does not add
            appended unindexed fragments.

            The list must be nonempty. With index use enabled, UUIDs must identify
            existing segments of one logical index on the query's vector column.
            Query metric compatibility is checked when selected segments overlap
            the fragment scope. With no overlap, ordinary search uses flat KNN with
            the query metric, or the selected index's metric when unset, while fast
            search returns no rows.
            Only supported for vector search.
            With ``nearest["use_index"]=False``, a flat scan uses the fragment scope
            rather than segment selection.
        prefilter: bool, default False
            If True, select vector top-k from rows matching the SQL filter. This can
            be more expensive and does not guarantee exact results from an
            approximate index.

            If False, apply the SQL filter after the combined ANN and flat top-k.
            Rejected rows are not replaced, so fewer than ``k`` rows (or none) may
            be returned. This also applies to searches without a vector index.
            The fragment scope, deletions, and external row masks restrict candidates
            regardless of this setting.
        use_scalar_index: bool, default True
            Lance will automatically use scalar indices to optimize a query.  In some
            corner cases this can make query performance worse and this parameter can
            be used to disable scalar indices in these cases.
        ignored_scalar_indices: iterable of str, default None
            Logical scalar index names to ignore during expression-filter planning.
            Other scalar indices remain eligible, and the original filter is still
            evaluated. Inverted indices used by ``contains_tokens`` filters are
            included; vector index segments and explicit full-text search queries
            are unaffected. If no index query remains, ``fast_search`` falls back
            to scanning all selected fragments and may return more matching rows.
            See ``ScannerBuilder.ignored_scalar_indices`` for more information.
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
            If True, then vector search, full text search, and scalar-indexed
            filters will only search indexed fragments, which yields faster
            search time but may skip recently appended unindexed data. Vector search
            skips flat fallback outside the selected segments' coverage and stale
            overlay rows needing replay. The fragment scope and SQL filter placement
            still apply. With no usable vector index coverage, the result is empty.
        scan_stats_callback: Callable[[ScanStatistics], None], default None
            A callback function that will be called with the scan statistics after the
            scan is complete.  Errors raised by the callback will be logged but not
            re-raised.
        strict_batch_size: bool, default False
            If True, all batches except the last batch will have exactly
            ``batch_size`` rows. This cannot be combined with a byte limit,
            including one configured through ``FileReaderOptions``.
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
        row_addr_allowlist: bytes, default None
            Restrict the scan to these row addresses. A serialized roaring treemap
            over ``_rowid`` (``RowAddrTreeMap::serialize_into`` output). Applied
            before KNN / BM25 ranking, so top-k is computed over the surviving rows
            rather than filtered afterwards.
        row_addr_blocklist: bytes, default None
            Exclude these row addresses, same encoding as ``row_addr_allowlist``.
            Combined with it when both are given.


        .. note::

            With both ``filter`` and ``nearest``, SQL filtering runs after vector
            top-k by default. Set ``prefilter=True`` to filter before top-k instead.
            Both strategies support explicit fragments and mixed ANN and flat search.


        For debugging ANN results, you can choose to not use the index
        even if present by specifying ``use_index=False``. For example,
        the following will always return exact KNN results:


        .. code-block:: python

            dataset.to_table(nearest={
                "column": "vector",
                "k": 10,
                "q": <query vector>,
                "use_index": False
            })

        Examples
        --------
        A segment covers F0/F1 and F2 is appended after indexing. Searching F1/F2
        combines indexed and unindexed rows; fast search keeps only indexed F1.

        >>> from tempfile import TemporaryDirectory
        >>> import lance
        >>> import numpy as np
        >>> import pyarrow as pa
        >>> table = pa.table({
        ...     "id": range(12),
        ...     "vector": pa.array(
        ...         [[float(i), 0, 0, 0] for i in range(12)],
        ...         type=pa.list_(pa.float32(), 4),
        ...     ),
        ... })
        >>> with TemporaryDirectory() as uri:
        ...     dataset = lance.write_dataset(
        ...         table.slice(0, 8), uri, max_rows_per_file=4,
        ...     )
        ...     dataset = dataset.create_index(
        ...         "vector", "IVF_FLAT", num_partitions=1,
        ...         ivf_centroids=np.zeros((1, 4), dtype=np.float32),
        ...     )
        ...     dataset = lance.write_dataset(table.slice(8), uri, mode="append")
        ...     segment_uuid = dataset.describe_indices()[0].segments[0].uuid
        ...     options = dict(
        ...         columns=["id"], fragments=dataset.get_fragments()[1:],
        ...         index_segments=[segment_uuid], prefilter=False,
        ...         nearest={
        ...             "column": "vector", "q": [0.0] * 4, "k": 6, "metric": "l2",
        ...         },
        ...     )
        ...     mixed = dataset.scanner(**options, fast_search=False).to_table()
        ...     fast = dataset.scanner(**options, fast_search=True).to_table()
        ...     mixed_ids, fast_ids = mixed["id"].to_pylist(), fast["id"].to_pylist()
        >>> mixed_ids
        [4, 5, 6, 7, 8, 9]
        >>> fast_ids
        [4, 5, 6, 7]

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
        if row_addr_allowlist is not None or row_addr_blocklist is not None:
            builder.row_addr_prefilter(row_addr_allowlist, row_addr_blocklist)
        setopt(builder.limit, limit)
        setopt(builder.offset, offset)
        setopt(builder.batch_size, batch_size)
        setopt(builder.batch_size_bytes, batch_size_bytes)
        setopt(builder.io_buffer_size, io_buffer_size)
        setopt(builder.batch_readahead, batch_readahead)
        setopt(builder.fragment_readahead, fragment_readahead)
        setopt(builder.scan_in_order, scan_in_order)
        setopt(builder.with_fragments, fragments)
        setopt(builder.with_index_segments, index_segments)
        setopt(builder.late_materialization, late_materialization)
        setopt(builder.blob_handling, blob_handling)
        setopt(builder.with_row_id, with_row_id)
        setopt(builder.with_row_address, with_row_address)
        setopt(builder.use_stats, use_stats)
        setopt(builder.use_scalar_index, use_scalar_index)
        setopt(builder.ignored_scalar_indices, ignored_scalar_indices)
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
            if isinstance(nearest, MinHashQuery):
                builder = builder.minhash_search(nearest)
            else:
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
        The default data file version for writes that omit ``data_storage_version``.
        Existing files may use other V2 versions; this is not a snapshot version
        summary.
        """
        return self._ds.data_storage_version

    @property
    def has_stable_row_ids(self) -> bool:
        """
        Whether this dataset has stable row IDs enabled.

        This is based on the dataset manifest feature flag and does not depend on
        whether the current version has any fragments.
        """
        return self._ds.has_stable_row_ids

    def get_row_addresses(
        self, row_ids: Union[List[int], pa.Array, pa.ChunkedArray]
    ) -> pa.UInt64Array:
        """
        Resolve row IDs to their current row addresses, in input order.

        A row address is ``(fragment_id << 32) | offset``. With stable row IDs,
        null, unknown and deleted IDs come back as null. Without stable row IDs,
        row IDs are row addresses, so each non-null ID is returned unchanged.

        Parameters
        ----------
        row_ids : list of int, pyarrow.Array or pyarrow.ChunkedArray
            Row IDs from ``_rowid``. Integer arrays are cast to uint64, so
            negative values raise an error.

        Returns
        -------
        pyarrow.UInt64Array

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> table = pa.table({"x": [1, 2, 3]})
        >>> ds = lance.write_dataset(
        ...     table, "memory://get_row_addresses", enable_stable_row_ids=True
        ... )
        >>> _ = ds.delete("x = 2")
        >>> ds.get_row_addresses([0, 1, 2, 100]).to_pylist()
        [0, None, 2, None]
        """
        if isinstance(row_ids, pa.ChunkedArray):
            row_ids = row_ids.combine_chunks()
        if isinstance(row_ids, pa.Array):
            row_ids = row_ids.cast(pa.uint64())
        else:
            row_ids = pa.array(row_ids, type=pa.uint64())
        return pa.array(self._ds.row_ids_to_row_addrs(row_ids))

    @property
    def max_field_id(self) -> int:
        """
        The max_field_id in manifest
        """
        return self._ds.max_field_id

    def to_table(
        self,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[Union[str, Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[Union[Dict[str, Any], MinHashQuery]] = None,
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
        full_text_query: Optional[Union[str, Dict[str, Any], FullTextQuery]] = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        blob_handling: Optional[str] = None,
        use_scalar_index: Optional[bool] = None,
        ignored_scalar_indices: Optional[Iterable[str]] = None,
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
        nearest: dict or MinHashQuery, default None
            Get the rows corresponding to the K most similar vectors, or, when a
            :class:`~lance.query.MinHashQuery` is given, the ``limit`` rows most
            similar to its text under the column's MinHash LSH index (adds a
            ``_distance`` column equal to ``1 - estimated Jaccard similarity``;
            unindexed rows are scored on the fly unless ``fast_search`` is set).
            Vector search example:

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
            If True, select vector top-k from rows matching the SQL filter. If False,
            filter the top-k result without replacing rejected rows. See
            :meth:`scanner` for details; approximate indices remain approximate.
        late_materialization: bool or List[str], default None
            Allows custom control over late materialization.  See
            ``ScannerBuilder.late_materialization`` for more information.
        blob_handling: str, default None
            Controls how blob columns are returned. See ``LanceDataset.scanner`` for
            details.
        use_scalar_index: bool, default True
            Allows custom control over scalar index usage.  See
            ``ScannerBuilder.use_scalar_index`` for more information.
        ignored_scalar_indices: iterable of str, default None
            Scalar index names to ignore during expression-filter planning. See
            ``ScannerBuilder.ignored_scalar_indices`` for more information.
        with_row_id: bool, optional, default False
            Return row ID.
        with_row_address: bool, optional, default False
            Return row address
        use_stats: bool, optional, default True
            Use stats pushdown during filters.
        fast_search: bool, optional, default False
            Only search indexed fragments for vector, full text, and scalar-indexed
            filter queries. This may skip recently appended unindexed data.
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
        With both ``filter`` and ``nearest``, SQL filtering runs after vector top-k
        by default. Set ``prefilter=True`` to filter before top-k instead. Postfilter
        may return fewer than ``k`` rows. For fragment and segment selection, use
        :meth:`scanner` followed by :meth:`LanceScanner.to_table`.
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
            ignored_scalar_indices=ignored_scalar_indices,
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
        filter: Optional[Union[str, Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[Union[Dict[str, Any], MinHashQuery]] = None,
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
        full_text_query: Optional[Union[str, Dict[str, Any], FullTextQuery]] = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        blob_mode: str = _BLOB_PANDAS_MODE_LAZY,
        use_scalar_index: Optional[bool] = None,
        ignored_scalar_indices: Optional[Iterable[str]] = None,
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
            ignored_scalar_indices=ignored_scalar_indices,
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

    def fragment_ids(self) -> Bitmap:
        """Get the ids of all fragments in the currently checked out version.

        Unlike :meth:`get_fragments`, this does not create an object for every
        fragment.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table({"x": range(4)})
        >>> dataset = lance.write_dataset(
        ...     data, "memory://fragment_ids", max_rows_per_file=2
        ... )
        >>> list(dataset.fragment_ids())
        [0, 1]
        """
        return self._ds.fragment_ids()

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
        filter: Optional[Union[str, Expression]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        nearest: Optional[Union[Dict[str, Any], MinHashQuery]] = None,
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
        full_text_query: Optional[Union[str, Dict[str, Any]]] = None,
        io_buffer_size: Optional[int] = None,
        late_materialization: Optional[bool | List[str]] = None,
        blob_handling: Optional[str] = None,
        use_scalar_index: Optional[bool] = None,
        ignored_scalar_indices: Optional[Iterable[str]] = None,
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
            ignored_scalar_indices=ignored_scalar_indices,
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
        *,
        ignore_deletions: bool = False,
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
        ignore_deletions: bool, default False
            Take the rows as if the dataset had no deletions. Deletion files
            are read only where a row id could name more than one physical
            copy (fragments whose id spans overlap, as after an update), which
            on a table with many fragments saves one request per fragment. A
            deleted row may come back with its stored values or be dropped, so
            the result is only complete for live ids: use it only for ids known
            to be live, for example ids read from an index built after the
            last deletion.
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
            [
                self._ds.take_rows(
                    row_ids,
                    columns,
                    columns_with_transform,
                    ignore_deletions=ignore_deletions,
                )
            ]
        )

    def take_blobs(
        self,
        blob_column: str,
        ids: Optional[Union[List[int], pa.Array]] = None,
        addresses: Optional[Union[List[int], pa.Array]] = None,
        indices: Optional[Union[List[int], pa.Array]] = None,
        *,
        buffer_size: int = DEFAULT_BLOB_BUFFER_SIZE,
    ) -> List[Optional[BlobFile]]:
        """
        Select blobs by row IDs.

        Instead of loading large binary blob data into memory before processing it,
        this API allows you to open binary blob data as a regular Python file-like
        object. For more details, see :py:class:`lance.BlobFile`.

        If you plan to read each selected blob completely with ``read()`` or
        ``readall()``, use :py:meth:`read_blobs` instead. It materializes blob
        payloads with Lance's planned batched reader.

        ``read_range`` and ``read_ranges`` do not use the sequential buffer and
        do not change the sequential cursor.

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
        buffer_size : int, default 512 KiB
            Sequential read-ahead size in bytes. ``0`` disables read-ahead.

        Returns
        -------
        blob_files : List[Optional[BlobFile]]
            One element per selected row. Null blob values return ``None``;
            valid empty blobs return a ``BlobFile`` with size zero.
        """
        buffer_size = _validate_buffer_size(buffer_size)
        selection_kind, selection_values = _resolve_blob_selection(
            ids, addresses, indices
        )

        if selection_kind == "ids":
            lance_blob_files = self._ds.take_blobs(selection_values, blob_column)
        elif selection_kind == "addresses":
            lance_blob_files = self._ds.take_blobs_by_addresses(
                selection_values, blob_column
            )
        else:
            lance_blob_files = self._ds.take_blobs_by_indices(
                selection_values, blob_column
            )
        return [
            BlobFile(lance_blob_file, buffer_size=buffer_size)
            if lance_blob_file is not None
            else None
            for lance_blob_file in lance_blob_files
        ]

    def read_blobs(
        self,
        blob_column: str,
        ids: Optional[Union[List[int], pa.Array]] = None,
        addresses: Optional[Union[List[int], pa.Array]] = None,
        indices: Optional[Union[List[int], pa.Array]] = None,
        *,
        io_buffer_size: Optional[int] = None,
        preserve_order: Optional[bool] = None,
    ) -> List[Tuple[int, Optional[bytes]]]:
        """
        Read blobs directly into memory using Lance's planned blob reader.

        Unlike :py:meth:`take_blobs`, which returns file-like :py:class:`lance.BlobFile`
        handles for random access, this API plans and executes batched reads and
        returns materialized blob payloads. Use this API for training loaders,
        batch preprocessing, and other workflows that need complete blob bytes.

        Exactly one of ids, addresses, or indices must be specified.

        Parameters
        ----------
        blob_column : str
            The name of the blob column to read.
        ids : Integer Array or array-like
            Row IDs to read in the dataset.
        addresses : Integer Array or array-like
            The (unstable) row addresses to read in the dataset.
        indices : Integer Array or array-like
            The offset / indices of the row in the dataset.
        io_buffer_size : int, optional
            Override the scheduler I/O buffer size used while materializing blobs.
        preserve_order : bool, optional
            If True, returned rows follow the requested selection order.

        Returns
        -------
        blobs : List[Tuple[int, Optional[bytes]]]
            One ``(row_address, blob_bytes)`` pair per selected row. Null blob
            values return ``None``; valid empty blobs return ``b""``.
        """
        selection_kind, selection_values = _resolve_blob_selection(
            ids, addresses, indices
        )

        kwargs = {
            "io_buffer_size": io_buffer_size,
            "preserve_order": preserve_order,
        }
        if selection_kind == "ids":
            return self._ds.read_blobs(selection_values, blob_column, **kwargs)
        if selection_kind == "addresses":
            return self._ds.read_blobs_by_addresses(
                selection_values, blob_column, **kwargs
            )
        return self._ds.read_blobs_by_indices(selection_values, blob_column, **kwargs)

    def read_blob_ranges(
        self,
        blob_column: str,
        requests: Sequence[Tuple[int, int, int]],
        *,
        selector: Literal["ids", "addresses", "indices"],
        io_buffer_size: Optional[int] = None,
        preserve_order: Optional[bool] = None,
    ) -> List[Tuple[int, int, Optional[bytes]]]:
        """
        Read row-specific blob-local byte ranges with one planned API call.

        Each request is a ``(row, offset, length)`` tuple. ``selector`` defines
        whether every ``row`` value is interpreted as a stable row ID, physical
        row address, or dataset index. Repeat a row value to read multiple ranges
        from the same blob. Planning, range validation, physical-source grouping,
        request coalescing, and bounded I/O scheduling all happen in Rust; this
        method does not use a Python thread pool.

        Every request produces one result. Requests on null blobs return ``None``,
        including when the requested range is empty. Empty ranges on non-null
        blobs return empty bytes without issuing payload I/O. For every request,
        offset plus length must fit in an unsigned 64-bit integer. Ranges on
        non-null blobs must not extend beyond the logical blob size. Blob-local
        bounds are not evaluated for null values because they have no logical
        payload length. By default results preserve the input request order.

        Parameters
        ----------
        blob_column : str
            The name of the blob column to read.
        requests : Sequence[Tuple[int, int, int]]
            Complete ``(row, offset, length)`` requests.
        selector : {"ids", "addresses", "indices"}
            Interpretation of every request's ``row`` value.
        io_buffer_size : int, optional
            Override the scheduler I/O buffer size used while materializing ranges.
        preserve_order : bool, optional
            If True, returned results follow request order. If False,
            ``request_index`` still identifies every result.

        Examples
        --------
        Read two disjoint ranges from the same row index:

        .. code-block:: python

            dataset.read_blob_ranges(
                "images",
                requests=[(7, 0, 1024), (7, 4096, 1024)],
                selector="indices",
            )

        Returns
        -------
        results : List[Tuple[int, int, Optional[bytes]]]
            One ``(request_index, row_address, data)`` tuple per request.
            The Python list retains all returned payload bytes in memory; peak
            result memory is therefore proportional to their total logical size.
        """
        return self._ds.read_blob_ranges(
            list(requests),
            blob_column,
            selector,
            io_buffer_size=io_buffer_size,
            preserve_order=preserve_order,
        )

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

    def slice(
        self,
        start: int,
        end: int,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
    ) -> pa.Table:
        """Select a contiguous range of rows by position.

        Equivalent to ``dataset.take(list(range(start, end)))``, but pushed
        down as an offset/limit scan instead of a materialized index list.

        Parameters
        ----------
        start : int
            The index of the first row to include (inclusive). Must be
            non-negative.
        end : int
            The index to stop before (exclusive). Must be greater than or
            equal to ``start``.
        columns: list of str, or dict of str to str default None
            List of column names to be fetched.
            Or a dictionary of column names to SQL expressions.
            All columns are fetched if None or unspecified.

        Returns
        -------
        table : pyarrow.Table

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> tbl = pa.table({"id": range(100)})
        >>> dataset = lance.write_dataset(tbl, "memory://slice_dataset")
        >>> dataset.slice(10, 20)
        pyarrow.Table
        id: int64
        ----
        id: [[10,11,12,13,14,15,16,17,18,19]]
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")
        return self.scanner(offset=start, limit=end - start, columns=columns).to_table()

    def count_rows(
        self, filter: Optional[Union[str, Expression]] = None, **kwargs: Any
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
                nullability is not changed. A non-nullable column can always be made
                nullable. A nullable column can be made non-nullable only if it
                contains no NULL values; otherwise an error is raised.
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
        The right-side join column must contain unique keys; null counts as a
        key, so at most one null value is allowed. Duplicate right-side keys
        raise an error.

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
        predicate: Union[str, Expression],
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
        encoded_predicate = (
            _serialize_expression(predicate, self._ds.schema)
            if isinstance(predicate, pa.compute.Expression)
            else predicate
        )
        return self._ds.delete(encoded_predicate, conflict_retries, retry_timeout)

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
        where: Optional[Union[str, Expression]] = None,
        conflict_retries: int = 10,
        retry_timeout: timedelta = timedelta(seconds=30),
        data_storage_version: Optional[str] = None,
    ) -> UpdateResult:
        """
        Update column values for rows matching where.

        Parameters
        ----------
        updates : dict of str to str
            A mapping of column names to a SQL expression.
        where : str or pa.compute.Expression, optional
            A SQL predicate or pyarrow Expression indicating which rows should
            be updated.
        conflict_retries : int, optional
            Number of times to retry the operation if there is contention.
            Default is 10.
        retry_timeout : timedelta, optional
            The timeout used to limit retries. This is the maximum time to spend on
            the operation before giving up. At least one attempt will be made,
            regardless of how long it takes to complete. Subsequent attempts will be
            cancelled once this timeout is reached. Default is 30 seconds.
        data_storage_version : str, optional
            Output data file version, such as "2.2", "stable", or "next". If
            omitted, use the dataset's default write version without changing it.
            V1/V2 cross-family targets are rejected.

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
        encoded_where = (
            _serialize_expression(where, self._ds.schema)
            if isinstance(where, pa.compute.Expression)
            else where
        )
        return self._ds.update(
            updates,
            encoded_where,
            conflict_retries,
            retry_timeout,
            data_storage_version,
        )

    def versions(self) -> List[Version]:
        """
        Return all versions in this dataset.

        Each entry is a :class:`Version` with ``version``, ``timestamp``,
        ``metadata``, and ``manifest_size``. ``manifest_size`` is the size in
        bytes of that version's manifest file, or ``None`` if the commit
        handler could not report it.
        """
        versions = self._ds.versions()
        for v in versions:
            _convert_version_timestamp(v)
        return versions

    def get_version(self) -> Version:
        """
        Return the currently checked out version, with its timestamp and the
        summary of its manifest.

        The summary is in ``metadata``. It counts the fragments, data files,
        deletion files, rows and bytes of the version, and is computed from the
        manifest that is already loaded, so nothing is read from storage.

        Use :attr:`version` instead when only the version number is needed.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> data = pa.table({"x": [1, 2, 3]})
        >>> dataset = lance.write_dataset(data, "memory://get_version")
        >>> version = dataset.get_version()
        >>> version["version"]
        1
        >>> version["metadata"]["total_rows"]
        '3'
        """
        return _convert_version_timestamp(self._ds.current_version())

    def version_refs(self) -> List[VersionRef]:
        """
        Return lightweight references to all attached versions in the current branch.

        Unlike :meth:`versions`, this does not read or deserialize every manifest.
        Use :attr:`latest_version` instead when only the latest version is needed.
        """
        return self._ds.version_refs()

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
            version_number on the default branch. ``("main", version)`` is an
            explicit alias for the default branch in this reference context.

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

    def base_paths(self) -> Dict[int, DatasetBasePath]:
        """Return the base paths registered in the current dataset snapshot.

        The returned dictionary maps each base path ID to an independent
        :class:`DatasetBasePath` object. It includes registered bases that are not
        referenced by any data files. The primary dataset storage is not added to
        the result unless it was explicitly registered as a base path.

        This method does not refresh the dataset to the latest version. Modifying
        the returned dictionary does not modify the dataset, and previously
        returned values do not change when the dataset is updated or checked out
        at another version. The dictionary iteration order is unspecified.

        Returns
        -------
        Dict[int, DatasetBasePath]
            Registered base paths keyed by base path ID. Each value exposes
            ``id``, ``name``, ``path``, and ``is_dataset_root`` as read-only
            attributes. ``is_dataset_root`` describes the base's path layout; it
            does not identify the dataset's current primary storage. Runtime
            storage options are not included.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> dataset = lance.write_dataset(
        ...     pa.table({"x": [1]}),
        ...     "memory://base-paths-example",
        ...     initial_bases=[
        ...         lance.DatasetBasePath(
        ...             "memory://base-paths-data", name="data"
        ...         )
        ...     ],
        ... )
        >>> base_paths = dataset.base_paths()
        >>> all(base_id == base.id for base_id, base in base_paths.items())
        True
        """
        return self._ds.base_paths()

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
        versions: Optional[List[int]] = None,
        delete_concurrency: Optional[int] = None,
        keep_one: Optional[KeepOne] = None,
        policy: Optional[CleanupPolicy] = None,
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
            Remove all versions older than this wall-clock age, including sampled
            commits from ``keep_one``. Without ``keep_one``, ``retain_versions``
            or ``versions``, defaults to two weeks. With ``keep_one`` alone,
            sampled history has no expiration limit.

        retain_versions: int, optional
            Retain the last N versions of the dataset. Must be positive.

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

        delete_concurrency: int, optional
            Maximum concurrent file deletions across this cleanup and its cascaded
            branches. Defaults to the object store's I/O parallelism.
            Must be positive (at most 2**61 - 1 on 64-bit platforms).

        delete_rate_limit: int, optional
            Maximum number of delete operations per second. When not set (default),
            deletions run at full speed. Set this to a positive integer to avoid
            hitting object store request rate limits (e.g. S3 HTTP 503 SlowDown).
            For example, ``delete_rate_limit=100`` limits to 100 operations/second.

        versions: list[int], optional
            Clean up only the specified dataset versions. The current version is
            never removed, and tagged versions are still protected by
            ``error_if_tagged_old_versions``.

        keep_one: KeepOne, optional
            Keep the newest eligible commit per UTC epoch-aligned interval, breaking
            timestamp ties by version. Its ``older_than`` age controls when thinning
            begins and is independent of this method's expiration ``older_than``.
            The hinted version and all later versions are protected from thinning
            even with backwards clocks. Tags, current versions and branch
            dependencies remain protected. For daily history between two and seven
            days, use ``older_than=timedelta(days=7),
            keep_one=KeepOne(timedelta(days=1), timedelta(days=2))``.
            The two-day delay bounds exposure to delayed hint writes; it does not
            make hint updates monotonic.

        policy: CleanupPolicy, optional
            Retention rules ordered by strictly increasing ``older_than`` ages.
            Each version uses the oldest matching KeepOne rule. RemoveAll is
            optional and must be last; without it, sampled history never expires.
            KeepOne ages are hint-relative and at least two days; RemoveAll uses
            wall-clock age. All comparisons are strictly older-than, so a version
            exactly at a threshold remains in the younger band. Version filters,
            tags and branch protections still apply. Cannot be combined with
            ``older_than`` or ``keep_one``. For example::

                dataset.cleanup_old_versions(policy=CleanupPolicy(rules=[
                    KeepOne(timedelta(hours=1), timedelta(days=2)),
                    KeepOne(timedelta(days=1), timedelta(days=7)),
                    RemoveAll(timedelta(days=30)),
                ]))
        """
        if policy is not None and (older_than is not None or keep_one is not None):
            raise ValueError("policy cannot be combined with older_than or keep_one")
        if (
            older_than is None
            and retain_versions is None
            and versions is None
            and keep_one is None
            and policy is None
        ):
            policy = CleanupPolicy()

        return self._ds.cleanup_old_versions(
            td_to_micros(older_than) if older_than is not None else None,
            retain_versions,
            delete_unverified,
            error_if_tagged_old_versions,
            delete_rate_limit,
            versions,
            delete_concurrency,
            (
                td_to_micros(keep_one.per_interval),
                td_to_micros(keep_one.older_than),
            )
            if keep_one is not None
            else None,
            _cleanup_policy_to_json(policy) if policy is not None else None,
        )

    def expire_versions(
        self,
        before: Optional[timedelta] = None,
        before_version: Optional[int] = None,
        *,
        keep_one_per: Optional[timedelta] = None,
        error_if_tagged_old_versions: bool = True,
        delete_rate_limit: Optional[int] = None,
    ) -> "ExpireVersionsStats":
        """Remove version history without deleting any data files.

        Unlike :meth:`cleanup_old_versions`, this never computes the set of live data
        files, so it does not read the manifests it is expiring. On a table with a very
        large number of versions that difference is hours against minutes.

        The data files that only an expired version referenced are **not** removed. They
        become unreferenced, and the next :meth:`cleanup_old_versions` reclaims them --
        and is much cheaper for having fewer manifests left to read. The intended order
        is expire, then clean up.

        The newest version, tags on the current branch, and versions a branch is rooted
        at are never removed.

        Parameters
        ----------
        before: timedelta, optional
            Expire versions written more than this long ago. Age comes from the manifest
            object's write time, not the commit timestamp recorded inside it; use
            ``before_version`` when an exact boundary matters.
        before_version: int, optional
            Expire versions numbered below this. Exact, and needs no timestamps.
        keep_one_per: timedelta, optional
            Instead of expiring every version past the cutoff, keep the newest one in
            each bucket of this width. ``timedelta(hours=1)`` keeps one per hour.
            Applies only beyond the cutoff, so ``before=timedelta(days=7)`` with
            ``keep_one_per=timedelta(hours=1)`` reads as "full history for a week,
            hourly before that". Must be at least one hour.

            .. warning::
                Thinning carries a rollback risk on an actively committing table.
                Resolving the latest version starts at the version hint and probes
                upward, stopping at the first version that is missing. Expiry refuses to
                remove anything at or above the hint, but that floor is read once and
                hint writes are unconditional, so a commit that started earlier can
                publish a *lower* hint afterwards. A gap above that lowered hint hides
                every version above it, and the table reads as an older state while the
                newer manifests are still there. The one-hour minimum bounds the
                exposure rather than removing it. Use thinning to repair a table that
                has accumulated far more versions than it can carry, not as a default.
        error_if_tagged_old_versions: bool, default True
            Raise instead of silently keeping a tagged version the policy would expire.
        delete_rate_limit: int, optional
            Maximum delete requests per second. One request is one manifest.

        Returns
        -------
        ExpireVersionsStats
            Counts of what was removed and retained. ``failed_deletes`` is non-zero
            when the run completed but could not remove everything it selected; those
            versions remain expirable and a later run retries them.

        Notes
        -----
        With no ``before`` and no ``before_version`` this expires nothing, rather than
        everything.
        """
        return self._ds.expire_versions(
            td_to_micros(before) if before is not None else None,
            before_version,
            td_to_micros(keep_one_per) if keep_one_per is not None else None,
            error_if_tagged_old_versions,
            delete_rate_limit,
        )

    def explain_expire_versions(
        self,
        before: Optional[timedelta] = None,
        before_version: Optional[int] = None,
        *,
        keep_one_per: Optional[timedelta] = None,
        error_if_tagged_old_versions: bool = True,
        delete_rate_limit: Optional[int] = None,
    ) -> "ExpireVersionsPlan":
        """Report what :meth:`expire_versions` would remove, without removing it.

        Takes the same arguments as :meth:`expire_versions`.
        """
        return self._ds.explain_expire_versions(
            td_to_micros(before) if before is not None else None,
            before_version,
            td_to_micros(keep_one_per) if keep_one_per is not None else None,
            error_if_tagged_old_versions,
            delete_rate_limit,
        )

    def explain_cleanup_old_versions(
        self,
        older_than: Optional[timedelta] = None,
        retain_versions: Optional[int] = None,
        *,
        delete_unverified: bool = False,
        error_if_tagged_old_versions: bool = True,
        delete_rate_limit: Optional[int] = None,
        versions: Optional[List[int]] = None,
        include_files: bool = False,
        max_files: int = 1000,
        keep_one: Optional[KeepOne] = None,
        policy: Optional[CleanupPolicy] = None,
    ) -> CleanupExplanation:
        """
        Explain what :meth:`cleanup_old_versions` would remove without deleting files.

        Parameters
        ----------

        older_than: timedelta, optional
            Remove all versions older than this wall-clock age, including sampled
            commits. Defaults to two weeks only when no version filters or
            ``keep_one`` are specified, as in :meth:`cleanup_old_versions`.

        retain_versions: int, optional
            Retain the last N versions of the dataset. Must be positive.

        delete_unverified: bool, default False
            Include unverified files that cleanup would remove when this is set.

        error_if_tagged_old_versions: bool, default True
            If set to `True`, an exception will be raised if any tagged versions
            match the parameters. Otherwise, tagged versions will be ignored.

        delete_rate_limit: int, optional
            Retained for compatibility; ignored because explain does not delete files.

        versions: list[int], optional
            Explain cleanup only for the specified dataset versions.

        include_files: bool, default False
            If `True`, include candidate files in the explanation up to
            ``max_files`` entries. Aggregate stats always include all candidates.

        max_files: int, default 1000
            Maximum number of candidate files to include when ``include_files``
            is `True`.

        keep_one: KeepOne, optional
            Thinning interval and minimum age, as in :meth:`cleanup_old_versions`.
            Its age is independent of the expiration ``older_than`` argument.

        policy: CleanupPolicy, optional
            Ordered retention rules, as in :meth:`cleanup_old_versions`.
            Cannot be combined with ``older_than`` or ``keep_one``.
        """
        if policy is not None and (older_than is not None or keep_one is not None):
            raise ValueError("policy cannot be combined with older_than or keep_one")
        if (
            older_than is None
            and retain_versions is None
            and versions is None
            and keep_one is None
            and policy is None
        ):
            policy = CleanupPolicy()
        if max_files <= 0:
            raise ValueError("max_files must be positive")

        return self._ds.explain_cleanup_old_versions(
            td_to_micros(older_than) if older_than is not None else None,
            retain_versions,
            delete_unverified,
            error_if_tagged_old_versions,
            delete_rate_limit,
            versions,
            include_files,
            max_files,
            (
                td_to_micros(keep_one.per_interval),
                td_to_micros(keep_one.older_than),
            )
            if keep_one is not None
            else None,
            _cleanup_policy_to_json(policy) if policy is not None else None,
        )

    def _prepare_scalar_index_request(
        self,
        column: Union[str, List[str]],
        index_type: Union[str, IndexConfig],
        kwargs: dict,
    ) -> tuple[str, str, str]:
        """Validate and normalize a scalar-index request for the native
        ``create_index`` call.
        """
        if isinstance(column, str):
            column = [column]

        if len(column) > 1:
            raise NotImplementedError(
                "Scalar indices currently only support a single column"
            )

        column = column[0]
        lance_field = self._ds.lance_schema.field_case_insensitive(column)

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
                "MINHASHLSH",
                "FM",
            ]:
                raise NotImplementedError(
                    (
                        'Only "BTREE", "BITMAP", "NGRAM", "ZONEMAP", "LABEL_LIST", '
                        '"INVERTED", "BLOOMFILTER", "RTREE", "MINHASHLSH" or "FM" are '
                        f"supported for scalar columns.  Received {index_type}",
                    )
                )

            if lance_field is None:
                if index_type in ["INVERTED", "FTS"]:
                    # Rust resolves public FTS paths through intervening list layers.
                    return column, index_type, index_type
                raise KeyError(f"{column} not found in schema")

            field = lance_field.to_arrow()

            field_type = field.type
            field_meta = field.metadata
            if hasattr(field_type, "storage_type"):
                field_type = field_type.storage_type

            if index_type in ["BTREE", "BITMAP"]:
                if (
                    not pa.types.is_integer(field_type)
                    and not pa.types.is_floating(field_type)
                    and not pa.types.is_boolean(field_type)
                    and not pa.types.is_string(field_type)
                    and not pa.types.is_large_string(field_type)
                    and not pa.types.is_binary(field_type)
                    and not pa.types.is_large_binary(field_type)
                    and not pa.types.is_temporal(field_type)
                    and not pa.types.is_decimal128(field_type)
                    and not pa.types.is_decimal256(field_type)
                    and not pa.types.is_fixed_size_binary(field_type)
                ):
                    raise TypeError(
                        f"BTREE/BITMAP index column {column} must be int",
                        ", float, bool, str, large_str, binary, large_binary, "
                        "decimal, fixed-size-binary, or temporal",
                    )
            elif index_type == "LABEL_LIST":
                if not (
                    pa.types.is_list(field_type) or pa.types.is_large_list(field_type)
                ):
                    raise TypeError(
                        f"LABEL_LIST index column {column} must be a list or large list"
                    )
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

            return column, index_type, index_type
        elif isinstance(index_type, IndexConfig):
            logical_index_type = index_type.index_type.upper()
            if lance_field is None and logical_index_type not in ["INVERTED", "FTS"]:
                raise KeyError(f"{column} not found in schema")
            parameters = dict(index_type.parameters)
            if logical_index_type in ["INVERTED", "FTS"]:
                parameters["document_granularity"] = kwargs["document_granularity"]
            config = json.dumps(parameters)
            kwargs["config"] = indices.IndexConfig(index_type.index_type, config)
            return column, "scalar", logical_index_type
        else:
            raise Exception("index_type must be str or IndexConfig")

    @staticmethod
    def _normalized_index_type(
        index_type: Union[str, IndexConfig],
    ) -> str:
        if isinstance(index_type, IndexConfig):
            index_type = index_type.index_type
        return index_type.upper()

    @classmethod
    def _is_segment_native_scalar_index_type(
        cls,
        index_type: Union[str, IndexConfig],
    ) -> bool:
        return cls._normalized_index_type(index_type) in {
            "BTREE",
            "BITMAP",
            "INVERTED",
            "FTS",
            "NGRAM",
            "RTREE",
            "ZONEMAP",
            "BLOOMFILTER",
            "LABEL_LIST",
            "MINHASHLSH",
            "FM",
        }

    @classmethod
    def _requires_uncommitted_scalar_index(
        cls,
        index_type: Union[str, IndexConfig],
    ) -> bool:
        return cls._normalized_index_type(index_type) in {
            "BTREE",
            "BITMAP",
            "NGRAM",
            "RTREE",
            "ZONEMAP",
            "BLOOMFILTER",
            "LABEL_LIST",
            "MINHASHLSH",
            "FM",
        }

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
            Literal["MINHASHLSH"],
            Literal["FM"],
            IndexConfig,
        ],
        name: Optional[str] = None,
        *,
        replace: bool = True,
        train: bool = True,
        fragment_ids: Optional[List[int]] = None,
        index_uuid: Optional[str] = None,
        progress_callback: Optional[Callable[[IndexProgress], None]] = None,
        format_version: Optional[Union[int, str]] = None,
        document_granularity: DocumentGranularity = DocumentGranularity.ROW,
        **kwargs: Any,
    ) -> None:
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


        Lance supports the following scalar index types:

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
        * ``FM``. An FM-index for substring search. It answers ``contains``
          filters exactly, including single-character patterns.

        Index training sorts can spill to disk under memory pressure.
        ``LANCE_MEM_POOL_SIZE`` configures this pool in bytes; it is not a limit
        on the process's total memory usage. Spilling still requires memory for
        individual input batches and for sorting and merging them, so large
        batches or a small pool can cause memory reservation failures.

        Setting ``LANCE_BYPASS_SPILLING`` to any value (including ``0`` or
        ``false``) bypasses the bounded memory pool for these sorts. Sorting then
        stays in memory, ignoring ``LANCE_MEM_POOL_SIZE``. This can avoid pool
        reservation failures but increases memory use and can exhaust system
        memory; use it only when the sort fits in available memory. The historical
        SortExec allocation issues in
        https://github.com/apache/datafusion/issues/10073 were fixed upstream by
        https://github.com/apache/datafusion/pull/14644; that fix does not remove
        the batch-size and memory-pool constraints above.

        **Experimental API**

        Parameters
        ----------
        column : str
            The column to be indexed.  Must be a boolean, integer, float,
            string, binary, decimal, fixed-size-binary, or
            supported temporal column.
        index_type : str
            The type of the index.  One of ``"BTREE"``, ``"BITMAP"``,
            ``"LABEL_LIST"``, ``"NGRAM"``, ``"ZONEMAP"``, ``"INVERTED"``,
            ``"FTS"``, ``"BLOOMFILTER"``, ``"RTREE"``, ``"MINHASHLSH"``, ``"FM"``.
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
            the index to the dataset. The segment can be optionally merged with
            other segments and committed later with
            ``commit_existing_index_segments(...)``.
            This parameter is passed via kwargs internally.
        index_uuid : str, optional
            A UUID to use for the segment written by this call.
            If not provided, a new UUID will be generated. This parameter is
            passed via kwargs internally.
        progress_callback : callable, optional
            A callback that receives :class:`lance.progress.IndexProgress` events while
            the index is being built.
        format_version: int or str, optional
            This is for the ``INVERTED`` / ``FTS`` index. Explicit on-disk FTS
            format version to write when creating a new index. Accepts ``1``,
            ``2``, ``3``, ``"v1"``, ``"v2"``, or ``"v3"``.
            If unset, Lance uses ``LANCE_FTS_FORMAT_VERSION`` when present and
            otherwise writes v2 for text analysis with ``block_size=128`` and
            v3 for code analysis or ``block_size=256``.

        num_segments: int, optional
            This is for ``index_type="FM"``. Split the fragments, in order, into
            at most this many segments, each an independent FM-index. There are
            never more segments than fragments. ``create_index_uncommitted``
            ignores it and always writes one segment.

        document_granularity: DocumentGranularity, default ROW
            This is for the ``INVERTED`` / ``FTS`` index. ``ROW`` treats all
            selected text in one dataset row as one document. ``LIST_ELEMENT``
            treats each element of the deepest list on ``column`` as one document.

        with_position: bool, default False
            This is for the ``INVERTED`` index. If True, the index will store the
            positions of the words in the document, so that you can conduct phrase
            query. This will significantly increase the index size.
            It won't impact the performance of non-phrase queries even if it is set to
            True.
            This option is not supported with the ``ngram`` tokenizer when creating
            a new index.
        block_size: int, default 128
            This is for the ``INVERTED`` index. Number of documents per compressed
            posting block. Must be one of ``128`` or ``256``.
            ``block_size=256`` is experimental and may introduce breaking changes.
            Use ``128`` when stable compatibility with the legacy posting layout is
            required.
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
            * "ngram": produces character N-grams for substring search.
            * "icu": ICU dictionary-based Unicode word segmentation.
            * "icu/split": ICU segmentation with simple-style delimiter splitting.
            * "code": splits code-like text. Requires FTS format v3 and enables
              the code analyzer options below.
        language: str, default "English"
            This is for the ``INVERTED`` index. The language for stemming
            and stop words. This is only used when `stem` or `remove_stop_words` is true
        max_token_length: Optional[int], default 40
            This is for the ``INVERTED`` index. The exclusive token length limit
            in UTF-8 bytes, not characters. Tokens of this length or longer are
            removed. For example, the default removes both 40 ASCII characters
            and 20 repetitions of ``é`` (40 UTF-8 bytes). Set to ``None`` to
            disable the limit.
        lower_case: bool, default True
            This is for the ``INVERTED`` index. If True, the index will convert all
            text to lowercase.
        stem: bool, default True (False for the "ngram" tokenizer)
            This is for the ``INVERTED`` index. If True, the index will stem the
            tokens.
        remove_stop_words: bool, default True (False for the "ngram" tokenizer)
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
        analyzer: str, optional
            This is for the ``INVERTED`` index. Selects an analyzer profile:
            ``"text"`` (the default) or ``"code"``. The ``"code"`` profile sets
            ``base_tokenizer="code"`` and applies code-oriented defaults:
            ``split_identifiers=False``, ``split_on_numerics=True``,
            ``preserve_original=True``, ``stem=False``, and
            ``remove_stop_words=False``. If unset, the profile is inferred from
            ``base_tokenizer="code"``. Explicit arguments override the profile
            defaults. Code analysis requires FTS format v3.
        split_identifiers: bool, default False
            This is for the ``INVERTED`` index with the code analyzer. If True,
            identifiers such as ``getUserName`` are split into subwords
            (``get``, ``User``, ``Name``) so each subword is searchable. When
            False, an identifier is only indexed as a whole, so ``getUserName``
            is not matched by ``user``.
        split_on_numerics: bool, default False (True for the "code" analyzer)
            This is for the ``INVERTED`` index with the code analyzer. If True,
            identifier subwords are split at letter/number boundaries, for
            example ``value42`` into ``value`` and ``42``. Only takes effect
            when ``split_identifiers=True``.
        preserve_original: bool, default False (True for the "code" analyzer)
            This is for the ``INVERTED`` index with the code analyzer. If True,
            the complete identifier is indexed in addition to its subwords.
            Only takes effect when ``split_identifiers=True``.
        index_operators: bool, default False
            This is for the ``INVERTED`` index with the code analyzer. If True,
            code operators such as ``::``, ``->``, and ``!=`` are indexed.
            Other punctuation is still dropped.

        Examples
        --------

        .. code-block:: python

            import lance

            dataset = lance.dataset("/tmp/images.lance")
            dataset.create_scalar_index(
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
        if not isinstance(document_granularity, DocumentGranularity):
            raise TypeError(
                "document_granularity must be a lance.query.DocumentGranularity"
            )
        kwargs["document_granularity"] = document_granularity.value
        column, index_type, logical_index_type = self._prepare_scalar_index_request(
            column, index_type, kwargs
        )

        if fragment_ids is not None and self._requires_uncommitted_scalar_index(
            logical_index_type
        ):
            raise ValueError(
                f"{logical_index_type} distributed indexing uses "
                "create_index_uncommitted(..., "
                f'index_type="{logical_index_type}", fragment_ids=...)'
            )
        # Add fragment_ids and index_uuid to kwargs if provided
        if fragment_ids is not None:
            kwargs["fragment_ids"] = fragment_ids
        if index_uuid is not None:
            kwargs["index_uuid"] = index_uuid
        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback
        if format_version is not None:
            kwargs["format_version"] = format_version
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
        streaming_sample_rate: Optional[int] = None,
        streaming_coreset_rate: Optional[int] = None,
        streaming_refine_passes: Optional[int] = None,
        skip_transpose: bool = False,
        rabitq_model: Optional[str] = None,
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

        if index_cache_size is not None:
            # The parameter has never reached Rust: index building does not use
            # the index cache, and the cache is sized on the dataset or session.
            warnings.warn(
                "The 'index_cache_size' parameter of create_index is ignored. "
                "The index cache is sized on the dataset, via "
                "lance.dataset(..., index_cache_size_bytes=...) or a Session.",
                DeprecationWarning,
                stacklevel=3,
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
            "IVF_HNSW_RQ",
            "IVF_RQ",
        ]
        if index_type not in valid_index_types:
            raise NotImplementedError(
                f"Only {valid_index_types} index types supported. Got {index_type}"
            )

        # Handle timing for various parts of accelerated builds
        timers = {}
        if accelerator is not None and index_type != "IVF_PQ":
            LOGGER.warning(
                "Index type %s does not support GPU acceleration; falling back to CPU",
                index_type,
            )
            accelerator = None

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
            from .vector import (
                one_pass_assign_ivf_pq_on_accelerator,
                one_pass_train_ivf_pq_on_accelerator,
            )

            LOGGER.info("Doing one-pass ivfpq accelerated computations")
            if num_partitions is None:
                num_rows = self.count_rows()
                num_partitions = _target_partition_size_to_num_partitions(
                    num_rows, target_partition_size
                )
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
            ivfpq_train_time = timers["ivf+pq_train:end"] - timers["ivf+pq_train:start"]
            LOGGER.info("ivf+pq training time: %ss", ivfpq_train_time)
            timers["ivf+pq_assign:start"] = time.time()
            shuffle_output_dir, shuffle_buffers = one_pass_assign_ivf_pq_on_accelerator(
                self,
                column[0],
                metric,
                accelerator,
                ivf_kmeans,
                pq_kmeans_list,
                batch_size=20480,
                filter_nan=filter_nan,
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
            if streaming_sample_rate is not None:
                kwargs["streaming_sample_rate"] = streaming_sample_rate
            if streaming_coreset_rate is not None:
                kwargs["streaming_coreset_rate"] = streaming_coreset_rate
            if streaming_refine_passes is not None:
                kwargs["streaming_refine_passes"] = streaming_refine_passes

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
                    if len(ivf_centroids.shape) != 2:
                        raise ValueError(
                            f"Ivf centroids must be 2D array: (clusters, dim), "
                            f"got {ivf_centroids.shape}"
                        )
                    if ivf_centroids.shape[0] == 0:
                        # Zero partitions panics in the Rust residual step.
                        raise ValueError(
                            "Ivf centroids must have at least one cluster, "
                            f"got {ivf_centroids.shape}"
                        )
                    if (
                        num_partitions is not None
                        and ivf_centroids.shape[0] != num_partitions
                    ):
                        raise ValueError(
                            f"Ivf centroids has {ivf_centroids.shape[0]} clusters, "
                            f"but num_partitions={num_partitions}"
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
                    num_bits = kwargs.get("num_bits", 8)
                    expected_centroids = 2**num_bits
                    if (
                        len(pq_codebook.shape) != 3
                        or pq_codebook.shape[0] != num_sub_vectors
                        or pq_codebook.shape[1] != expected_centroids
                    ):
                        raise ValueError(
                            "PQ codebook must be 3D array: "
                            f"(sub_vectors, {expected_centroids}, dim) "
                            f"for num_bits={num_bits}, "
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

        if rabitq_model is not None:
            kwargs["rabitq_model"] = rabitq_model

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
        streaming_sample_rate: Optional[int] = None,
        streaming_coreset_rate: Optional[int] = None,
        streaming_refine_passes: Optional[int] = None,
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
            A ``num_clusters x dimension`` array of existing K-mean centroids
            for IVF clustering. The row count determines the number of IVF
            partitions. If not provided, a new KMeans model will be trained.
        ivf_centroids_file : str, optional
            Path to a ``.npy`` file holding the same array as ``ivf_centroids``.
            Mutually exclusive with it. The file's row count sets the number of
            IVF partitions, so passing a ``num_partitions`` that disagrees with
            it is an error.
        pq_codebook : optional,
            It can be :py:class:`np.ndarray`, :py:class:`pyarrow.FixedSizeListArray`,
            or :py:class:`pyarrow.FixedShapeTensorArray`.
            A ``num_sub_vectors x (2 ^ num_bits) x
            (dimensions // num_sub_vectors)`` array of K-mean centroids for PQ
            codebook. ``num_bits`` defaults to 8.
            If not provided, a new PQ model will be trained.
        num_sub_vectors : int, optional
            The number of sub-vectors for PQ (Product Quantization).
        accelerator : str or ``torch.Device``, optional
            If set, use an accelerator to speed up the training process.
            Accepted accelerator: "cuda" (Nvidia GPU) and "mps" (Apple Silicon GPU).
            If not set, use the CPU.
        index_cache_size : int, optional
            Deprecated and ignored. Index building does not read the index cache;
            size it on the dataset with
            ``lance.dataset(..., index_cache_size_bytes=...)`` or on a ``Session``.
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
            ``merge_existing_index_segments(...)`` if grouping is needed and then
            committed with ``commit_existing_index_segments(...)``.

            Vector segments support both shared and independent model scopes. If
            the caller provides the same IVF centroids, and for IVF_PQ the same
            PQ codebook, to each worker, the resulting segments share model
            semantics and are suitable for workflows that physically merge
            compatible segments. If those artifacts are omitted, each segment can
            train its own IVF/PQ model for its assigned fragments. Such segments
            can be committed together and are queried independently by segment
            UUID; partition ids are interpreted within each segment's own model.
            Keep independently trained segments as separate physical segments
            unless the merge workflow can preserve or reconcile the model state.
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
        streaming_sample_rate : int, optional
            If set below ``sample_rate``, IVF kmeans trains incrementally and samples
            at most ``num_partitions * streaming_sample_rate`` vectors per step. For
            ``num_partitions > 256``, chunks are compressed into a weighted coreset
            and final centroids are trained with weighted hierarchical kmeans.
        streaming_coreset_rate : int, optional
            If set, controls the final weighted coreset budget independently from
            ``streaming_sample_rate``. The budget is
            ``num_partitions * streaming_coreset_rate``.
        streaming_refine_passes : int, optional
            Number of extra streaming Lloyd refinement passes to run after streaming
            coreset training. Each pass loads at most
            ``num_partitions * streaming_sample_rate`` raw vectors at a time.
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

        Optional parameters for `IVF_RQ` and `IVF_HNSW_RQ`:

            - num_bits
                The number of bits for RQ (Rabit Quantization). Default is 5.

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
            Only supports CUDA (Nvidia) or MPS (Apple) currently.
            Requires PyTorch being installed.

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
            streaming_sample_rate=streaming_sample_rate,
            streaming_coreset_rate=streaming_coreset_rate,
            streaming_refine_passes=streaming_refine_passes,
            skip_transpose=skip_transpose,
            require_commit=True,
            **kwargs,
        )
        return self

    def create_index_uncommitted(
        self,
        column: Union[str, List[str]],
        index_type: Union[str, IndexConfig],
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
        streaming_sample_rate: Optional[int] = None,
        streaming_coreset_rate: Optional[int] = None,
        streaming_refine_passes: Optional[int] = None,
        skip_transpose: bool = False,
        rabitq_model: Optional[str] = None,
        **kwargs,
    ) -> Index:
        """
        Create one segment without publishing it and return its metadata.

        This is the public distributed-build API for vector, BTREE scalar,
        canonical bitmap scalar, INVERTED scalar, NGRAM scalar, RTREE scalar,
        ZONEMAP scalar, BLOOMFILTER scalar, LABEL_LIST scalar, MINHASHLSH
        scalar, and FM scalar index construction.
        Unlike :meth:`create_index`, this method does not publish the index into
        the dataset manifest. Instead, it writes one segment under
        ``_indices/<segment_uuid>/`` and returns the resulting
        :class:`Index` metadata.

        Callers should:

        1. run :meth:`create_index_uncommitted` on each worker with that worker's
           assigned ``fragment_ids``
        2. collect the returned :class:`Index` objects
        3. optionally merge caller-defined groups with
           :meth:`merge_existing_index_segments`
        4. commit the final segment list with
           :meth:`commit_existing_index_segments`

        BTREE, BITMAP, INVERTED, NGRAM, RTREE, ZONEMAP, BLOOMFILTER,
        LABEL_LIST, MINHASHLSH, and FM segments may be merged with
        :meth:`merge_existing_index_segments` before commit. MINHASHLSH
        segments of one index must share parameters. NGRAM segments
        built before a deferred compaction must be merged before commit so
        their postings can be rebuilt against current row addresses.
        Parameters are the same as :meth:`create_index`, with one additional
        requirement:

        - ``fragment_ids`` must be provided
        - Vector segments support both shared and independent model scopes. Pass
          the same IVF centroids, and for IVF_PQ the same PQ codebook, to each
          worker when segments need shared model semantics or physical merge
          compatibility. If these artifacts are omitted, each segment may train
          its own IVF/PQ model and can be committed with other segments as one
          logical index; query execution searches each segment by UUID and
          interprets partition ids within that segment. Keep independently
          trained segments as separate physical segments unless the merge
          workflow can preserve or reconcile the model state.
        - ``rabitq_model`` (``IVF_RQ`` and ``IVF_HNSW_RQ``): a JSON string produced by
          ``lance.lance.indices.build_rq_model``. It must be identical across all
          workers for their segments to be mergeable, since it pins the RaBitQ
          rotation so every segment rotates vectors the same way. If omitted, each
          call generates its own random rotation, which is only safe for a single,
          non-merged segment.

        Returns
        -------
        Index
            Metadata for the segment that was written by this call.
        """
        is_scalar_segment_request = self._is_segment_native_scalar_index_type(
            index_type
        )
        if is_scalar_segment_request:
            if fragment_ids is None:
                raise ValueError(
                    "create_index_uncommitted requires fragment_ids "
                    "for distributed index build"
                )

            kwargs = dict(kwargs)
            column, rust_index_type, _ = self._prepare_scalar_index_request(
                column, index_type, kwargs
            )
            kwargs["fragment_ids"] = fragment_ids
            if index_uuid is not None:
                kwargs["index_uuid"] = index_uuid

            return self._ds.create_index(
                [column],
                rust_index_type,
                name,
                replace,
                train,
                storage_options,
                kwargs,
            )

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
            streaming_sample_rate=streaming_sample_rate,
            streaming_coreset_rate=streaming_coreset_rate,
            streaming_refine_passes=streaming_refine_passes,
            skip_transpose=skip_transpose,
            rabitq_model=rabitq_model,
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

    def prewarm_index(
        self,
        name: str,
        *,
        with_position: bool = False,
        index_segments: Optional[Iterable[Union[str, uuid.UUID]]] = None,
    ):
        """
        Prewarm an index

        By default, this will load the entire index into memory. This can help
        avoid cold start issues with index queries. If the index does not fit in
        the index cache, then this will result in wasted I/O.

        Use ``session().index_cache_size_bytes()`` before and after prewarm to
        inspect how much the index cache grew.

        Parameters
        ----------
        name: str
            The name of the index to prewarm.
        with_position: bool, default False
            This is only supported for ``INVERTED`` indices. If True, positions are
            also loaded into the cache during prewarm so phrase queries do not need a
            separate lazy positions read.
        index_segments: iterable of str or uuid.UUID, default None
            If specified, prewarm only these physical index segment UUIDs from the
            named logical index. Use :meth:`describe_indices` to inspect logical
            indices and obtain segment UUIDs from ``IndexDescription.segments``.
        """
        return self._ds.prewarm_index(
            name,
            with_position=with_position,
            index_segments=_normalize_index_segment_ids(index_segments),
        )

    def merge_index_metadata(
        self,
        index_uuid: str,
        index_type: str,
        batch_readhead: Optional[int] = None,
        progress_callback: Optional[Callable[[IndexProgress], None]] = None,
    ):
        """
        Merge distributed scalar index metadata.

        Vector and Bitmap distributed indexing no longer use this API. For
        those index families, build segments with
        :meth:`create_index_uncommitted`, optionally merge caller-defined
        groups with :meth:`merge_existing_index_segments`, and publish them with
        :meth:`commit_existing_index_segments`.

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
        self, index_name: str, column: str, segments: List[Union[IndexSegment, Index]]
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
    def _inherit_base_store_params(
        dataset_or_uri: Union[str, Path, LanceDataset, None],
        base_store_params: Optional[Dict[str, Dict[str, str]]],
    ) -> Optional[Dict[str, Dict[str, str]]]:
        if base_store_params is None and isinstance(dataset_or_uri, LanceDataset):
            return dataset_or_uri._base_store_params
        return base_store_params

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
        base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
        commit_timeout: Optional[timedelta] = _DEFAULT_COMMIT_TIMEOUT,
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
            If True, enables stable row IDs when creating a new dataset.  Each new
            row is assigned the next id in increasing order and keeps that id for
            its lifetime: compaction, update and merge insert relocate or rewrite
            the row without changing its id.  This option is ignored for existing
            datasets.
        namespace_client : LanceNamespace, optional
            A namespace client. Must be provided together with table_id.
            Use lance.namespace.connect() to create a namespace.
        table_id : List[str], optional
            The table identifier within the namespace (e.g., ["workspace", "table"]).
            Must be provided together with namespace_client.
        base_store_params : dict of str to dict, optional
            Runtime-only object store parameters keyed by base path URI.
        commit_timeout : timedelta, optional
            Maximum time to wait for the commit operation (including retries on
            conflict) to complete. Defaults to 30 minutes. Pass ``None`` to
            disable the timeout entirely. Must be a positive duration.

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
        base_store_params = LanceDataset._inherit_base_store_params(
            base_uri, base_store_params
        )

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
                commit_timeout=commit_timeout,
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
                commit_timeout=commit_timeout,
            )
        else:
            raise TypeError(
                "operation must be a LanceOperation.BaseOperation or Transaction, "
                f"got {type(operation)}"
            )

        ds = LanceDataset.__new__(LanceDataset)
        ds._storage_options = storage_options
        ds._base_store_params = base_store_params
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
        base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
        commit_timeout: Optional[timedelta] = _DEFAULT_COMMIT_TIMEOUT,
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
        base_store_params : dict of str to dict, optional
            Runtime-only object store parameters keyed by base path URI.
        commit_timeout : timedelta, optional
            Maximum time to wait for the commit operation (including retries on
            conflict) to complete. Defaults to 30 minutes. Pass ``None`` to
            disable the timeout entirely. Must be a positive duration.

        Returns
        -------
        dict with keys:
            dataset: LanceDataset
                A new version of Lance Dataset.
            merged: Transaction
                The merged transaction that was applied to the dataset.
        """
        base_store_params = LanceDataset._inherit_base_store_params(
            dest, base_store_params
        )

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
            commit_timeout=commit_timeout,
        )
        ds = LanceDataset.__new__(LanceDataset)
        ds._ds = new_ds
        ds._uri = new_ds.uri
        ds._storage_options = storage_options
        ds._base_store_params = base_store_params
        ds._namespace_client = None
        ds._table_id = None
        ds._namespace_client_managed_versioning = False
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
            version_number on the default branch. ``("main", version)`` is an
            explicit alias for the default branch in this reference context.
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

    def deep_clone(
        self,
        target_path: str | Path,
        reference: int | str | Tuple[Optional[str], Optional[int]],
        storage_options: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> "LanceDataset":
        """
        Deep clone the specified version into a new dataset at target_path.

        This operation copies the data, deletion, and index files referenced by
        the selected version before committing the destination dataset.

        Parameters
        ----------
        target_path : str or Path
            The URI or filesystem path to clone the dataset into.
        reference : int, str or Tuple[Optional[str], Optional[int]]
            An integer specifies a version number in the current branch; a string
            specifies a tag name; a Tuple[Optional[str], Optional[int]] specifies
            a version number in a specified branch.
        storage_options : dict, optional
            Object store configuration for the destination dataset. If omitted,
            the source dataset's storage options are used.

        Returns
        -------
        LanceDataset
            The deep-cloned dataset.
        """
        if isinstance(target_path, Path):
            target_uri = os.fspath(target_path)
        else:
            target_uri = target_path

        if storage_options is None:
            storage_options = self._storage_options
        self._ds.deep_clone(target_uri, reference, storage_options)

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

    def sql(
        self,
        sql: str,
        *,
        parameters: Optional[
            Union[Sequence[Any], Dict[str, Any], pa.RecordBatch]
        ] = None,
    ) -> "SqlQueryBuilder":
        """Execute SQL query on the dataset.

        Parameters
        ----------
        sql : str
            SQL SELECT statement to execute. The query must reference the dataset
            as the FROM clause or omit the FROM clause to query the current dataset.
        parameters : sequence, dict, or pyarrow.RecordBatch, optional
            Values to bind to SQL placeholders. See
            :meth:`SqlQueryBuilder.parameters` for supported forms.

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
        builder = SqlQueryBuilder(self._ds.sql(sql))
        if parameters is not None:
            builder.parameters(parameters)
        return builder

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

        Reading row changes requires stable row IDs at both nonzero endpoints.
        Enable them when creating the dataset with ``enable_stable_row_ids=True``.
        Transaction listing is available without stable row IDs.

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
                "memory://delta_demo",
                enable_stable_row_ids=True,
            )

            # Append some data to create v2
            ds.insert(pa.table({"id": [3], "val": ["c"]}))

            # Compute inserted rows from v1 -> v2 (shorthand)
            delta = ds.delta(compared_against=1)
            reader = delta.get_inserted_rows()
            for batch in reader:
                print(batch)

            # Or using explicit version range
            delta = ds.delta(begin_version=1, end_version=2)
        """
        builder = _DatasetDeltaBuilder(self._ds.delta())

        # Forward every argument as given: the Rust builder rejects conflicting
        # or half-set selections, which filtering here would hide.
        if compared_against is not None:
            builder = builder.compared_against_version(compared_against)
        if begin_version is not None:
            builder = builder.with_begin_version(begin_version)
        if end_version is not None:
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
        """Delete a dataset and everything under ``base_uri``.

        To limit the damage a mistyped or misconfigured path can do, ``base_uri``
        must be a dataset root, meaning it holds a manifest that can be read, or a
        namespace declare/deregister marker. Anything else raises
        :class:`ValueError`, including a path that holds only data files or only
        unreadable manifests: such leftovers need an explicit storage-level delete.

        Note that a path which passes this check is deleted in full, including any
        unmanaged files kept next to the dataset.

        Parameters
        ----------
        base_uri : str or Path
            Root of the dataset to delete.
        storage_options : optional, dict
            Extra options for the storage backend.
        ignore_not_found : optional, bool
            If True, return successfully when ``base_uri`` does not exist.

        Raises
        ------
        ValueError
            If ``base_uri`` is not a Lance dataset root.
        """
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

    def initialize_mem_wal(
        self,
        *,
        maintained_indexes: Optional[List[str]] = None,
        bucket_column: Optional[str] = None,
        num_buckets: Optional[int] = None,
        identity_column: Optional[str] = None,
        unsharded: bool = False,
        durable_write: Optional[bool] = None,
        max_wal_buffer_size: Optional[int] = None,
        max_wal_flush_interval_ms: Optional[int] = None,
        max_memtable_size: Optional[int] = None,
        max_memtable_rows: Optional[int] = None,
        max_memtable_batches: Optional[int] = None,
        max_unflushed_memtable_bytes: Optional[int] = None,
        manifest_scan_batch_size: Optional[int] = None,
        backpressure_log_interval_ms: Optional[int] = None,
        stats_log_interval_ms: Optional[int] = None,
        hnsw_params: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        """Initialize MemWAL on this dataset.

        Must be called once before any calls to `mem_wal_writer`. Append-only
        tables may omit primary-key metadata; primary keys are only required
        for primary-key lookup and last-write-wins deduplication workflows.

        At most one sharding mode may be selected: bucket sharding
        (``bucket_column`` + ``num_buckets``), identity sharding
        (``identity_column``), or ``unsharded``. With none selected, shards are
        managed manually by passing shard IDs to `mem_wal_writer`.

        Any writer-configuration keyword arguments (``durable_write``,
        ``max_memtable_size``, ``max_wal_flush_interval_ms``, etc. — the same
        knobs accepted by `mem_wal_writer`) are recorded as the default
        `~lance.mem_wal.ShardWriter` configuration in the MemWAL index, so
        every writer starts from the same defaults.

        Parameters
        ----------
        maintained_indexes : list of str, optional
            Names of existing indexes to keep updated as data is written
            through the MemWAL. Must reference indexes that already exist.
            Omitted (the default) keeps every index on the table updated,
            including ones created later; an empty list keeps none.
        hnsw_params : dict, optional
            Per-index HNSW build-parameter overrides recorded as writer-config
            defaults, keyed by maintained vector index name. Each value is a dict
            with any of ``num_edges`` (graph degree; level 0 keeps
            ``2 * num_edges``, equivalent to FAISS's ``M``), ``ef_construction``,
            and ``max_level``. Without an override an index uses the default
            parameters. These are writer config — they affect the MemTable a
            writer builds and may be overridden per-writer in
            :meth:`mem_wal_writer`.
        bucket_column : str, optional
            With ``num_buckets``, hash-bucket writes by this scalar column.
        num_buckets : int, optional
            Number of hash buckets (shards). Required with ``bucket_column``.
        identity_column : str, optional
            Shard by the raw value of this scalar column.
        unsharded : bool, default False
            Route every write to a single shard.

        Raises
        ------
        IOError
            - An entry in *maintained_indexes* does not exist on the dataset.
            - MemWAL has already been initialized on this dataset.
        ValueError
            More than one sharding mode was selected, or bucket sharding was
            given only one of ``bucket_column`` / ``num_buckets``.
        """
        self._ds.initialize_mem_wal(
            maintained_indexes=maintained_indexes,
            bucket_column=bucket_column,
            num_buckets=num_buckets,
            identity_column=identity_column,
            unsharded=unsharded,
            durable_write=durable_write,
            max_wal_buffer_size=max_wal_buffer_size,
            max_wal_flush_interval_ms=max_wal_flush_interval_ms,
            max_memtable_size=max_memtable_size,
            max_memtable_rows=max_memtable_rows,
            max_memtable_batches=max_memtable_batches,
            max_unflushed_memtable_bytes=max_unflushed_memtable_bytes,
            manifest_scan_batch_size=manifest_scan_batch_size,
            backpressure_log_interval_ms=backpressure_log_interval_ms,
            stats_log_interval_ms=stats_log_interval_ms,
            hnsw_params=hnsw_params,
        )

    def mem_wal_index_details(self) -> Optional[Dict[str, Any]]:
        """Return the MemWAL index details, or ``None`` if not initialized.

        Returns
        -------
        dict or None
            A dict with ``num_shards``, ``maintained_indexes``,
            ``writer_config_defaults``, ``sharding_specs``, and ``dir_id``, or
            ``None`` when MemWAL has not been initialized on this dataset.
            ``dir_id`` names the MemWAL that commits recording compaction
            progress must carry.
        """
        return self._ds.mem_wal_index_details()

    def mem_wal_writer(
        self,
        shard_id: str,
        *,
        durable_write: Optional[bool] = None,
        max_wal_buffer_size: Optional[int] = None,
        max_wal_flush_interval_ms: Optional[int] = None,
        max_memtable_size: Optional[int] = None,
        max_memtable_rows: Optional[int] = None,
        max_memtable_batches: Optional[int] = None,
        max_unflushed_memtable_bytes: Optional[int] = None,
        manifest_scan_batch_size: Optional[int] = None,
        backpressure_log_interval_ms: Optional[int] = None,
        stats_log_interval_ms: Optional[int] = None,
        hnsw_params: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> "mem_wal.ShardWriter":
        """Get a ShardWriter for the specified shard.

        `initialize_mem_wal` must be called before using this method.
        Each shard is an independent write path; use different shard IDs
        to achieve parallel ingestion without writer contention.

        Parameters
        ----------
        shard_id : str
            UUID string identifying the write shard (e.g.
            ``str(uuid.uuid4())``).
        durable_write : bool, optional
            Whether to fsync WAL writes (default: ``True``).
        max_wal_buffer_size : int, optional
            Maximum WAL buffer size in bytes (default: 10 MB).
        max_wal_flush_interval_ms : int, optional
            Maximum WAL flush interval in milliseconds (default: 100).
        max_memtable_size : int, optional
            Maximum MemTable size in bytes (default: 256 MB).
        max_memtable_rows : int, optional
            Maximum rows per MemTable (default: 100 000).
        max_memtable_batches : int, optional
            Maximum batches per MemTable (default: 8 000).
        max_unflushed_memtable_bytes : int, optional
            Maximum unflushed bytes before backpressure (default: 1 GB).
        manifest_scan_batch_size : int, optional
            Batch size for manifest scans (default: 2).
        backpressure_log_interval_ms : int, optional
            Interval for backpressure log messages in milliseconds
            (default: 30 000).
        stats_log_interval_ms : int, optional
            Interval for statistics log messages in milliseconds
            (default: 60 000).  Pass ``0`` to disable.
        hnsw_params : dict, optional
            Per-index HNSW build-parameter overrides for the MemTable this
            writer builds, keyed by maintained vector index name. Each value is
            a dict with any of ``num_edges`` (graph degree; level 0 keeps
            ``2 * num_edges``, equivalent to FAISS's ``M``), ``ef_construction``,
            and ``max_level``. Without an override an index uses the default
            parameters.

        Returns
        -------
        ShardWriter
            A context-manager-compatible writer for the specified shard.

        Examples
        --------
        >>> import lance
        >>> import pyarrow as pa
        >>> import tempfile
        >>> import uuid
        >>> schema = pa.schema([
        ...     pa.field("id", pa.int64(), nullable=False,
        ...              metadata={"lance-schema:unenforced-primary-key": "true"}),
        ...     pa.field("val", pa.float32()),
        ... ])
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     ds = lance.write_dataset(
        ...         pa.table({"id": [1], "val": [0.1]}, schema=schema),
        ...         tmpdir,
        ...     )
        ...     ds.initialize_mem_wal()
        ...     shard_id = str(uuid.uuid4())
        ...     new_data = pa.table({"id": [2], "val": [0.2]}, schema=schema)
        ...     with ds.mem_wal_writer(shard_id) as writer:
        ...         writer.put(new_data)
        """
        import lance.mem_wal as _mw

        kwargs = {
            name: val
            for name, val in [
                ("durable_write", durable_write),
                ("max_wal_buffer_size", max_wal_buffer_size),
                ("max_wal_flush_interval_ms", max_wal_flush_interval_ms),
                ("max_memtable_size", max_memtable_size),
                ("max_memtable_rows", max_memtable_rows),
                ("max_memtable_batches", max_memtable_batches),
                ("max_unflushed_memtable_bytes", max_unflushed_memtable_bytes),
                ("manifest_scan_batch_size", manifest_scan_batch_size),
                ("backpressure_log_interval_ms", backpressure_log_interval_ms),
                ("stats_log_interval_ms", stats_log_interval_ms),
                ("hnsw_params", hnsw_params),
            ]
            if val is not None
        }
        raw = self._ds.mem_wal_writer(shard_id, **kwargs)
        return _mw.ShardWriter(raw)

    def tracked_files(
        self,
        *,
        min_version: Optional[int] = None,
        progress: Optional[Callable] = None,
    ) -> pa.RecordBatchReader:
        """Stream all files referenced by any manifest version of this dataset.

        Parameters
        ----------
        min_version : int, optional
            If set, only include manifests with version >= min_version.
        progress : callable, optional
            Called after each manifest is processed with two arguments:
            ``(manifests_processed: int, manifests_total: Optional[int])``.
            ``manifests_total`` is ``None`` until all manifest locations
            have been listed. Works well with ``tqdm``::

                from tqdm import tqdm
                pbar = tqdm(unit="manifest")
                def on_progress(processed, total):
                    if total is not None:
                        pbar.total = total
                    pbar.update(1)
                reader = ds.tracked_files(progress=on_progress)
                table = reader.read_all()
                pbar.close()

        Returns
        -------
        pyarrow.RecordBatchReader
            Schema:

            - **version** (int64): manifest version number
            - **base_uri** (dictionary<int32, utf8>): storage root URI
            - **path** (utf8): file path relative to ``base_uri``
            - **type** (dictionary<int8, utf8>): one of ``manifest``,
              ``data file``, ``deletion file``, ``transaction file``,
              ``index file``

            Output order is non-deterministic.
        """
        return self._ds.tracked_files(min_version=min_version, progress=progress)

    def all_files(self) -> pa.RecordBatchReader:
        """Stream all files physically present at this dataset's base URI.

        Returns a :class:`pyarrow.RecordBatchReader` with schema:

        - **base_uri** (dictionary<int32, utf8>): storage root URI
        - **path** (utf8): file path relative to ``base_uri``
        - **size_bytes** (int64): file size in bytes
        - **last_modified** (timestamp[us, UTC]): last modification time

        Only the primary object store is scanned; alternate ``base_paths``
        entries are not included.
        """
        return self._ds.all_files()


# LanceOperation is a namespace for operations that can be applied to a dataset.


def write_dataset(
    data_obj: ReaderLike,
    uri: Optional[Union[str, Path, LanceDataset]] = None,
    schema: Optional[pa.Schema] = None,
    mode: str = "create",
    *,
    max_rows_per_file: int = 1024 * 1024,
    max_rows_per_group: int = 1024,
    max_bytes_per_file: int = 90 * 1024 * 1024 * 1024,
    data_cache_bytes: Optional[int] = None,
    max_page_bytes: Optional[int] = None,
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
    target_all_bases: Optional[bool] = None,
    base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
    external_blob_mode: Literal["reference", "ingest"] = "reference",
    allow_external_blob_outside_bases: bool = False,
    blob_pack_file_size_threshold: Optional[int] = None,
    namespace_client: Optional[LanceNamespace] = None,
    table_id: Optional[List[str]] = None,
    session: Optional[Session] = None,
) -> LanceDataset:
    """Write a given data_obj to the given uri

    Parameters
    ----------
    data_obj: Reader-like
        The data to be written. Acceptable types are:
        - Pandas DataFrame, Polars DataFrame
        - Pyarrow Table, RecordBatch, Dataset, Scanner, or RecordBatchReader
        - An iterable of Pyarrow RecordBatch (requires ``schema``)
        - A dict of columns, or a non-empty list of row dicts
        - A non-empty list of instances of the same Pydantic model class
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
    data_cache_bytes : int, optional
        Total bytes to buffer for column data before writing pages. The budget
        is divided evenly across top-level columns. If not set, the current
        file writer uses 8 MiB per column. Ignored for legacy V1 files.
    max_page_bytes : int, optional
        Best-effort maximum page size in bytes. If not set, the current file
        writer uses its configured default. Ignored for legacy V1 files.
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

        For writes involving additional base paths, a key of the form
        ``base_<id>.<key>`` applies ``<key>`` only to the base path with that
        id, overriding the unscoped options that every base inherits. For
        example ``{"account_key": "shared", "base_1.account_key": "abc"}``
        makes base 1 use ``account_key = abc`` while all other options are
        shared.
    data_storage_version: optional, str, default None
        The version of the data storage format to use. Newer versions are more
        efficient but require newer versions of lance to read. For create and
        overwrite, None uses the latest stable version. For append, None uses
        the dataset's default write version; an explicit V2 version applies
        only to new files and does not change that default.
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
        A row then keeps the same id for its lifetime: compaction, update and merge
        insert relocate or rewrite the row without changing its id. This only
        applies to new datasets; a write to an existing dataset keeps that
        dataset's own setting.
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
    target_all_bases: bool, optional
        Write new data files round-robin across every base registered in the
        manifest. When True (include primary), the dataset's primary storage
        participates in the rotation as the first slot; when False, only the
        registered bases are used. Cannot be combined with ``target_bases``.
    base_store_params : dict of str to dict, optional
        Runtime-only object store parameters keyed by base path URI. Each key
        is a base path URI (e.g., "s3://bucket/path") and each value is a dict
        of storage options (credentials, endpoint, etc.) for that base. These
        are not persisted to the manifest. These take precedence over
        ``base_<id>.<key>`` entries in ``storage_options``. When a base has no
        explicit entry here, the top-level ``storage_options`` is used as a
        fallback.
    external_blob_mode: {"reference", "ingest"}, default "reference"
        How external blob URIs are handled on write.

        - ``"reference"`` stores the URI as an external blob reference.
        - ``"ingest"`` reads the external bytes during write and stores them in
          Lance-managed storage using the normal inline / packed / dedicated
          thresholds.
    allow_external_blob_outside_bases: bool, default False
        If False, external blob URIs must map to a registered non-dataset-root
        base path. If True, external blob URIs outside registered bases are
        allowed. Only valid when ``external_blob_mode="reference"``. Setting
        this to True with ``"ingest"`` mode raises an error.
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
    base_store_params = LanceDataset._inherit_base_store_params(uri, base_store_params)

    # Merge properties and commit_message with priority to commit_message
    merged_properties = _merge_message_to_properties(
        commit_message, transaction_properties
    )

    params = {
        "mode": mode,
        "max_rows_per_file": max_rows_per_file,
        "max_rows_per_group": max_rows_per_group,
        "max_bytes_per_file": max_bytes_per_file,
        "data_cache_bytes": data_cache_bytes,
        "max_page_bytes": max_page_bytes,
        "progress": progress,
        "storage_options": storage_options,
        "data_storage_version": data_storage_version,
        "enable_v2_manifest_paths": enable_v2_manifest_paths,
        "enable_stable_row_ids": enable_stable_row_ids,
        "auto_cleanup_options": auto_cleanup_options,
        "transaction_properties": merged_properties,
        "initial_bases": initial_bases,
        "target_bases": target_bases,
        "target_all_bases": target_all_bases,
        "base_store_params": base_store_params,
        "external_blob_mode": external_blob_mode,
        "allow_external_blob_outside_bases": allow_external_blob_outside_bases,
        "blob_pack_file_size_threshold": blob_pack_file_size_threshold,
        "session": session,
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
    ds._base_store_params = base_store_params
    ds._namespace_client = namespace_client
    ds._table_id = table_id
    ds._namespace_client_managed_versioning = namespace_client_managed_versioning
    ds._ds = inner_ds
    ds._uri = inner_ds.uri
    ds._default_scan_options = None
    ds._read_params = None
    return ds


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


@_coerce_source.register(LanceDataset)
def _lance_dataset_source(data_obj: LanceDataset, schema=None) -> _WriteSource:
    # Capture the native dataset version and all default scan options once.
    return _coerce_source(data_obj.scanner(), schema)
