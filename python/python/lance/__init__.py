# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from . import io, log
from .blob import (
    Blob,
    BlobArray,
    BlobColumn,
    BlobDescriptor,
    BlobDescriptorArrayBuilder,
    BlobFile,
    DedicatedBlobWriter,
    PackedBlobWriter,
    blob_array,
    blob_field,
)
from .dataset import (
    DataStatistics,
    FieldStatistics,
    Index,
    IndexFile,
    LanceDataset,
    LanceOperation,
    LanceScanner,
    MergeInsertBuilder,
    Session,
    Transaction,
    __version__,
    batch_udf,
    write_dataset,
)
from .fragment import FragmentMetadata, LanceFragment
from .lance import (
    CleanupCandidateFile,
    CleanupExplanation,
    CleanupReferencedBranch,
    CleanupStats,
    DatasetBasePath,
    FFILanceTableProvider,
    FtsToken,
    ScanStatistics,
    bytes_read_counter,
    iops_counter,
    simd_info,
    tokenize,
)
from .mem_wal import (
    CompactedSsTable,
    ExecutionPlan,
    LsmPointLookupPlanner,
    LsmScanner,
    LsmVectorSearchPlanner,
    ShardingField,
    ShardingSpec,
    ShardSnapshot,
    ShardWriter,
    evaluate_sharding_spec,
)
from .namespace import (
    DescribeTableRequest,
    LanceNamespace,
)
from .progress import IndexProgress
from .schema import json_to_schema, schema_to_json
from .util import sanitize_ts

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from lance.commit import CommitLock
    from lance.dependencies import pandas as pd

    ts_types = Union[datetime, pd.Timestamp, str]


__all__ = [
    "Blob",
    "BlobArray",
    "BlobColumn",
    "BlobFile",
    "DedicatedBlobWriter",
    "BlobDescriptorArrayBuilder",
    "PackedBlobWriter",
    "BlobDescriptor",
    "blob_array",
    "blob_field",
    "CleanupCandidateFile",
    "CleanupExplanation",
    "CleanupReferencedBranch",
    "CleanupStats",
    "DatasetBasePath",
    "DataStatistics",
    "FieldStatistics",
    "FragmentMetadata",
    "FtsToken",
    "Index",
    "IndexFile",
    "LanceDataset",
    "LanceFragment",
    "LanceOperation",
    "LanceScanner",
    "MergeInsertBuilder",
    "ScanStatistics",
    "Transaction",
    "__version__",
    "batch_udf",
    "bytes_read_counter",
    "dataset",
    "io",
    "iops_counter",
    "json_to_schema",
    "schema_to_json",
    "set_logger",
    "simd_info",
    "tokenize",
    "write_dataset",
    "FFILanceTableProvider",
    "IndexProgress",
    "ExecutionPlan",
    "LsmPointLookupPlanner",
    "LsmScanner",
    "LsmVectorSearchPlanner",
    "CompactedSsTable",
    "ShardSnapshot",
    "ShardWriter",
    "ShardingField",
    "ShardingSpec",
    "evaluate_sharding_spec",
]


def dataset(
    uri: Optional[Union[str, Path]] = None,
    version: Optional[int | str] = None,
    asof: Optional[ts_types] = None,
    block_size: Optional[int] = None,
    commit_lock: Optional[CommitLock] = None,
    index_cache_size: Optional[int] = None,
    storage_options: Optional[Dict[str, str]] = None,
    default_scan_options: Optional[Dict[str, str]] = None,
    metadata_cache_size_bytes: Optional[int] = None,
    index_cache_size_bytes: Optional[int] = None,
    read_params: Optional[Dict[str, any]] = None,
    session: Optional[Session] = None,
    namespace_client: Optional[LanceNamespace] = None,
    table_id: Optional[List[str]] = None,
    base_store_params: Optional[Dict[str, Dict[str, str]]] = None,
) -> LanceDataset:
    """
    Opens the Lance dataset from the address specified.

    Parameters
    ----------
    uri : str, optional
        Address to the Lance dataset. It can be a local file path `/tmp/data.lance`,
        or a cloud object store URI, i.e., `s3://bucket/data.lance`.
        Either `uri` or (`namespace_client` + `table_id`) must be provided.
    version : optional, int | str
        If specified, load a specific version of the Lance dataset. Else, loads the
        latest version. A version number (`int`) or a tag (`str`) can be provided.
    asof : optional, datetime or str
        If specified, find the latest version created on or earlier than the given
        argument value. If a version is already specified, this arg is ignored.
    block_size : optional, int
        Block size in bytes. Provide a hint for the size of the minimal I/O request.
    commit_lock : optional, lance.commit.CommitLock
        A custom commit lock.  Only needed if your object store does not support
        atomic commits.  See the user guide for more details.
    index_cache_size : optional, int
        Index cache size. Index cache is a LRU cache with TTL. This number specifies the
        number of index pages, for example, IVF partitions, to be cached in
        the host memory. Default value is ``256``.

        Roughly, for an ``IVF_PQ`` partition with ``n`` rows, the size of each index
        page equals the combination of the pq code (``nd.array([n,pq], dtype=uint8))``
        and the row ids (``nd.array([n], dtype=uint64)``).
        Approximately, ``n = Total Rows / number of IVF partitions``.
        ``pq = number of PQ sub-vectors``.
    storage_options : optional, dict
        Extra options that make sense for a particular storage connection. This is
        used to store connection parameters like credentials, endpoint, etc.
        Set ``scheduler_error_mode`` to ``"fail_fast"`` to cancel the remaining
        physical reads in a logical scheduler request after one read fails. The
        default is ``"best_effort"``.

        For datasets with additional registered base paths, a key of the form
        ``base_<id>.<key>`` applies ``<key>`` only to the base path with that
        manifest id, overriding the unscoped options that every base inherits.
        For example ``{"account_key": "shared", "base_1.account_key": "abc"}``
        makes base 1 use ``account_key = abc`` while all other options are
        shared.
    default_scan_options : optional, dict
        Default scan options that are used when scanning the dataset.  This accepts
        the same arguments described in :py:meth:`lance.LanceDataset.scanner`.  The
        arguments will be applied to any scan operation.

        This can be useful to supply defaults for common parameters such as
        ``batch_size``.

        It can also be used to create a view of the dataset that includes meta
        fields such as ``_rowid`` or ``_rowaddr``.  If ``default_scan_options`` is
        provided then the schema returned by :py:meth:`lance.LanceDataset.schema` will
        include these fields if the appropriate scan options are set.
    metadata_cache_size_bytes : optional, int
        Size of the metadata cache in bytes. This cache is used to store metadata
        information about the dataset, such as schema and statistics. If not specified,
        a default size will be used.
    read_params : optional, dict
        Dictionary of read parameters. Currently supports:
        - cache_repetition_index (bool): Whether to cache repetition indices for
          large string/binary columns. This is enabled by default. You can disable
          it globally by setting LANCE_READ_CACHE_REPETITION_INDEX=false.
        - validate_on_decode (bool): Whether to validate data during decoding
    session : optional, lance.Session
        A session to use for this dataset. This contains the caches used by the
        across multiple datasets.
    namespace_client : optional, LanceNamespace
        A namespace client from which to fetch table location and storage options.
        Use lance.namespace.connect() to create a namespace instance.
        Must be provided together with `table_id`. Cannot be used with `uri`.
        When provided, the table location will be fetched automatically from the
        namespace via describe_table().
    table_id : optional, List[str]
        The table identifier when using a namespace (e.g., ["my_table"]).
        Must be provided together with `namespace_client`. Cannot be used with `uri`.
    base_store_params : dict of str to dict, optional
        Runtime-only object store parameters keyed by base path URI. Each key
        is a base path URI (e.g., "s3://bucket/path") and each value is a dict
        of storage options (credentials, endpoint, etc.) for that base.  These
        take precedence over ``base_<id>.<key>`` entries in ``storage_options``.
        When a base has no explicit entry here, the top-level
        ``storage_options`` is used as a fallback.

    Notes
    -----
    When using `namespace_client` and `table_id`:
    - The `uri` parameter is optional and will be fetched from the namespace
    - Storage options from describe_table() will be used automatically
    - A dynamic storage options provider will be created to refresh credentials
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

    # Handle namespace resolution in Python
    namespace_client_managed_versioning = False
    if namespace_client is not None:
        if table_id is None:
            raise ValueError(
                "Both 'namespace_client' and 'table_id' must be provided together."
            )

        # Resolve the latest table metadata here. The requested dataset version is
        # applied by the lower-level dataset open path after namespace resolution.
        request = DescribeTableRequest(id=table_id, version=None)
        response = namespace_client.describe_table(request)

        uri = response.location
        if uri is None:
            raise ValueError("Namespace did not return a 'location' for the table")

        # Check if namespace manages versioning (commits go through namespace API)
        namespace_client_managed_versioning = (
            getattr(response, "managed_versioning", None) is True
        )

        namespace_storage_options = response.storage_options

        # Merge namespace storage options with user-provided options
        # Namespace options take precedence
        if namespace_storage_options is not None:
            if storage_options is None:
                storage_options = namespace_storage_options
            else:
                merged_options = dict(storage_options)
                merged_options.update(namespace_storage_options)
                storage_options = merged_options
    elif table_id is not None:
        raise ValueError(
            "Both 'namespace_client' and 'table_id' must be provided together."
        )

    ds = LanceDataset(
        uri,
        version,
        block_size,
        commit_lock=commit_lock,
        index_cache_size=index_cache_size,
        storage_options=storage_options,
        default_scan_options=default_scan_options,
        metadata_cache_size_bytes=metadata_cache_size_bytes,
        index_cache_size_bytes=index_cache_size_bytes,
        read_params=read_params,
        session=session,
        namespace_client=namespace_client,
        table_id=table_id,
        namespace_client_managed_versioning=namespace_client_managed_versioning,
        base_store_params=base_store_params,
    )
    if version is None and asof is not None:
        ts_cutoff = sanitize_ts(asof)
        ver_cutoff = max(
            [v["version"] for v in ds.versions() if v["timestamp"] < ts_cutoff],
            default=None,
        )
        if ver_cutoff is None:
            raise ValueError(
                f"{ts_cutoff} is earlier than the first version of this dataset"
            )
        else:
            return LanceDataset(
                uri,
                ver_cutoff,
                block_size,
                commit_lock=commit_lock,
                index_cache_size=index_cache_size,
                storage_options=storage_options,
                metadata_cache_size_bytes=metadata_cache_size_bytes,
                index_cache_size_bytes=index_cache_size_bytes,
                read_params=read_params,
                session=session,
                namespace_client=namespace_client,
                table_id=table_id,
                namespace_client_managed_versioning=namespace_client_managed_versioning,
                base_store_params=base_store_params,
            )
    else:
        return ds


def set_logger(
    file_path="pylance.log",
    name="pylance",
    level=logging.INFO,
    format_string=None,
    log_handler=None,
):
    log.set_logger(file_path, name, level, format_string, log_handler)


def __warn_on_fork():
    warnings.warn(
        "lance is not fork-safe. If you are using multiprocessing, use spawn or \
forkserver instead."
    )


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=__warn_on_fork)
