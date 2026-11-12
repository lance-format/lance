# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Python scanner configuration, execution, and blob conversion."""

from __future__ import annotations

import copy
import operator
from dataclasses import dataclass
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
    Union,
    cast,
)

import numpy as np
import pyarrow as pa
from pyarrow import RecordBatch, Schema

from ..dependencies import _check_for_numpy
from ..dependencies import pandas as pd
from ..fragment import LanceFragment
from ..lance import (
    PySearchFilter,
    ScanStatistics,
    _format_field_path,
    _parse_field_path,
    _Scanner,
    _serialize_row_addrs,
)
from ..query import FullTextQuery, MinHashQuery
from ..types import _coerce_source, _WriteSource
from ..util import _normalize_index_segment_ids

if TYPE_CHECKING:
    import uuid

    from ..blob import BlobFile
    from ..dataset import LanceDataset

    QueryVectorLike = Union[
        pd.Series,
        pa.Array,
        pa.Scalar,
        np.ndarray,
        Iterable[Union[float, Iterable[float]]],
    ]

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


def _field_blob_paths(field: pa.Field, parent: str = "") -> Iterator[str]:
    segment = _format_field_path([field.name])
    field_path = f"{parent}.{segment}" if parent else segment
    if _is_blob_field(field):
        yield field_path
    elif pa.types.is_struct(field.type):
        for i in range(field.type.num_fields):
            yield from _field_blob_paths(field.type.field(i), field_path)


def _blob_paths_in_schema(schema: pa.Schema) -> list[str]:
    return [path for field in schema for path in _field_blob_paths(field)]


def _normalize_blob_pandas_mode(
    blob_mode: str,
) -> Literal["lazy", "bytes", "descriptions"]:
    if blob_mode not in _BLOB_PANDAS_MODES:
        raise ValueError("blob_mode must be one of: 'lazy', 'bytes', 'descriptions'")
    return cast("Literal['lazy', 'bytes', 'descriptions']", blob_mode)


def _sources_from_transforms(
    output_paths: list[str],
    transforms: dict[str, str],
) -> dict[str, str]:
    result = {}
    for path in output_paths:
        segments = _parse_field_path(path)
        source_path = transforms[segments[0]]
        result[path] = (
            source_path
            if not segments[1:]
            else _format_field_path(_parse_field_path(source_path) + segments[1:])
        )
    return result


def _sources_from_direct_projection(
    output_paths: list[str],
    dataset_schema: pa.Schema,
) -> dict[str, str]:
    source_paths = set(_blob_paths_in_schema(dataset_schema))
    result = {}
    for path in output_paths:
        segments = _parse_field_path(path)
        result[path] = (
            segments[0] if len(segments) == 1 and segments[0] in source_paths else path
        )
    return result


def _blob_column_sources(
    schema: pa.Schema,
    snapshot: Dict[str, Any],
    dataset_schema: pa.Schema,
) -> dict[str, str]:
    output_paths = _blob_paths_in_schema(schema)
    transforms = dict(snapshot.get("_columns_with_transform") or ())
    if transforms:
        return _sources_from_transforms(output_paths, transforms)
    return _sources_from_direct_projection(output_paths, dataset_schema)


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
    return False


def _descriptors_at_path(table: pa.Table, path: str) -> list[Optional[Dict[str, Any]]]:
    segments = _parse_field_path(path)
    values = table.column(segments[0]).to_pylist()

    for segment in segments[1:]:
        values = [value.get(segment) if value is not None else None for value in values]

    return values


def _replace_value_at_path(
    parent: Optional[Dict[str, Any]],
    segments: list[str],
    value: Any,
) -> Optional[Dict[str, Any]]:
    if parent is None:
        return None

    segment = segments[0]
    updated = dict(parent)

    if len(segments) == 1:
        updated[segment] = value
    else:
        updated[segment] = _replace_value_at_path(
            parent[segment],
            segments[1:],
            value,
        )

    return updated


def _replace_in_struct_column(
    dataframe: "pd.DataFrame",
    path: str,
    values: list[Optional[BlobFile]],
) -> None:
    segments = _parse_field_path(path)
    parent_name = segments[0]
    nested_segments = segments[1:]

    parents = dataframe[parent_name].tolist()

    dataframe[parent_name] = [
        _replace_value_at_path(parent, nested_segments, value)
        for parent, value in zip(parents, values)
    ]


def _fetch_blob_files_for_paths(
    dataset: "LanceDataset",
    table: pa.Table,
    blob_paths: list[str],
    blob_sources: dict[str, str],
    row_addrs: list[int],
) -> dict[str, list[Optional[BlobFile]]]:
    blob_files: dict[str, list[Optional[BlobFile]]] = {}
    for path in blob_paths:
        descriptors = _descriptors_at_path(table, path)

        null_mask = [
            _is_null_blob_description(descriptor) for descriptor in descriptors
        ]

        non_null_addresses = [
            row_addrs[index] for index, is_null in enumerate(null_mask) if not is_null
        ]

        fetched_blobs = (
            iter(dataset.take_blobs(blob_sources[path], addresses=non_null_addresses))
            if non_null_addresses
            else iter([])
        )

        blobs_for_path = []
        for is_null in null_mask:
            if is_null:
                blobs_for_path.append(None)
            else:
                blobs_for_path.append(next(fetched_blobs))

        blob_files[path] = blobs_for_path

    return blob_files


def _place_blob_files(
    dataframe: "pd.DataFrame",
    blob_files: dict[str, list[Optional[BlobFile]]],
    schema: pa.Schema,
) -> None:
    top_level: dict[str, list[Optional[BlobFile]]] = {}
    nested: dict[str, list[Optional[BlobFile]]] = {}
    for path, values in blob_files.items():
        segments = _parse_field_path(path)
        if len(segments) == 1:
            top_level[segments[0]] = values
        else:
            nested[path] = values

    for index, field in enumerate(schema):
        if field.name in top_level:
            dataframe.insert(index, field.name, top_level[field.name])

    for path, values in nested.items():
        _replace_in_struct_column(dataframe, path, values)


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


def _needs_substrait_placeholder(t: pa.DataType, nested: bool = False) -> bool:
    """Return True if *t* contains a type that PyArrow's substrait serializer
    or Lance's substrait decoder cannot handle.

    These require a placeholder at any nesting depth:

    * ``fixed_size_list`` — substrait has no equivalent type.
    * ``large_list``, ``duration`` and ``decimal256`` — PyArrow has no
      substrait conversion for them.
    * Arrow extension types (e.g. ``fixed_shape_tensor``) — substrait cannot
      represent them.
    * A struct whose fields carry non-``None`` metadata — substrait rejects
      such structs.  Extension types leave ``metadata={}`` on struct fields
      after a lance round-trip; ``{}`` is non-``None``.
    * ``time64("us")`` — PyArrow writes it as substrait ``time``, which Lance
      cannot decode.

    Unsigned integers, ``large_string``, ``large_binary``, ``date64``,
    ``time32`` and ``time64`` need one only inside a list, struct or map.
    PyArrow 16 and later write them as user-defined types; Lance drops those
    from the top level of the schema but cannot decode them when nested.  It
    writes a dictionary as its value type, so the value type decides.
    """
    if (
        pa.types.is_fixed_size_list(t)
        or pa.types.is_large_list(t)
        or pa.types.is_duration(t)
        or pa.types.is_decimal256(t)
        or (pa.types.is_time64(t) and t.unit == "us")
    ):
        return True
    if isinstance(t, pa.lib.BaseExtensionType):
        return True
    if nested and (
        pa.types.is_unsigned_integer(t)
        or pa.types.is_large_string(t)
        or pa.types.is_large_binary(t)
        or pa.types.is_date64(t)
        or pa.types.is_time32(t)
        or pa.types.is_time64(t)
    ):
        return True
    if pa.types.is_dictionary(t):
        return _needs_substrait_placeholder(t.value_type, nested)
    if pa.types.is_struct(t):
        for i in range(t.num_fields):
            f = t.field(i)
            if f.metadata is not None:
                return True
            if _needs_substrait_placeholder(f.type, nested=True):
                return True
    if pa.types.is_list(t):
        return _needs_substrait_placeholder(t.value_type, nested=True)
    if pa.types.is_map(t):
        return any(
            _needs_substrait_placeholder(child, nested=True)
            for child in (t.key_type, t.item_type)
        )
    return False


def _serialize_expression(
    expression: pa.compute.Expression, schema: pa.Schema
) -> bytes:
    from pyarrow.substrait import serialize_expressions

    # Keep each field's position so Substrait references still resolve against
    # the stored schema when PyArrow cannot serialize an unrelated field's type.
    scalar_schema = pa.schema(
        [
            (
                pa.field(f"__unlikely_name_placeholder_{i}", pa.int8())
                if _needs_substrait_placeholder(field.type)
                else field
            )
            for i, field in enumerate(schema)
        ]
    )
    serialized = serialize_expressions([expression], ["my_filter"], scalar_schema)
    if isinstance(serialized, memoryview):
        return serialized.tobytes()
    try:
        return serialized.to_pybytes()
    except AttributeError:
        raise TypeError(
            f"serialize_expressions returned unexpected type {type(serialized)}"
        )


def serialize_row_addrs(addrs: Iterable[int]) -> bytes:
    """Encode row addresses for ``row_addr_allowlist`` / ``row_addr_blocklist``.

    Those parameters take a serialized roaring treemap over ``_rowid``; this is
    the way to produce one from Python.

    >>> blob = serialize_row_addrs([0, 2, 4])                    # doctest: +SKIP
    >>> ds.scanner(row_addr_allowlist=blob).to_table()           # doctest: +SKIP
    """
    return _serialize_row_addrs(list(addrs))


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
        _check_for_numpy(query) and isinstance(query, np.ndarray)
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
    quantized_refine_factor: Optional[int] = None,
    use_index: bool = True,
    ef: Optional[int] = None,
    query_parallelism: Optional[int] = None,
    approx_mode: Literal["fast", "normal", "accurate"] = "normal",
    distance_range: Optional[tuple[Optional[float], Optional[float]]] = None,
    search_effort: Optional[float] = None,
    refine_column: Optional[str] = None,
    refine_q: Optional[QueryVectorLike] = None,
    refine_metric: Optional[str] = None,
) -> dict:
    """Configure nearest neighbor search.

    Parameters
    ----------
    column: str
        The name of the vector column to search.
    q: QueryVectorLike
        The query vector. For fixed-size vector columns, this may be a 2-D
        array-like or list-shaped batch of query vectors. Batch queries return up to
        ``k`` rows per query vector and include Int32 non-null ``query_index`` as
        the first output column.
        Flattened 1-D inputs whose length is a multiple of the vector dimension are
        rejected. Datasets with an existing ``query_index`` column cannot be used for
        batch search. When ``use_index`` is true and a vector index is available,
        each query vector is searched through the index path; otherwise the flat batch
        path is used.
    k: int, optional
        The number of nearest neighbors to return.
    metric: str, optional
        The distance metric to use (e.g., "L2", "cosine", "dot", "hamming").
    nprobes: int, optional
        The number of partitions to search, setting both the minimum and maximum.
    minimum_nprobes: int, optional
        The minimum number of partitions to search.
    maximum_nprobes: int, optional
        The maximum number of partitions to search per index segment. This bounds
        both initial and later probing without disabling Auto probing.
    refine_factor: int, optional
        The refine factor for the search.
    quantized_refine_factor: int, optional
        For an IVF_RQ flat index, overfetch this many one-bit candidates per
        requested result and rerank them using the most accurate stored RQ data.
        Indexed candidates do not read their original vectors. Unindexed fragments
        and stale overlay rows still use flat fallback and may read original vectors.
        With ``distance_range``, selected partitions use Accurate RQ search directly
        so the range applies before top-k truncation.
    use_index: bool, default True
        Whether to use the index for the search.
    ef: int, optional
        Beam width for HNSW search. Defaults to ``1.5 * k * refine_factor``,
        floored at 256.
    query_parallelism: int, optional
        Maximum partition-search concurrency for a single vector query.
        The default is 0. Value 0 uses the automatic policy, which currently
        maps to the single-worker sequential path. Value -1 uses the CPU pool
        size. Value 1 uses the single-worker sequential path. Values >= 2 use
        the partition-parallel path and are clamped to the CPU pool size.
    search_effort: float, optional
        Initial IVF search budget, a finite value in [0, 1]. None resolves
        to 0.5, preserving Auto. Zero starts at the caller minimum (at least one
        available partition); one starts at all available partitions, subject
        to maximum_nprobes. For L2, cosine and dot with k <= 100000, efforts
        0.1, 0.25 and 0.5 target 80%, 90% and 95% mean partition-routing recall,
        respectively. Intermediate budgets interpolate geometrically between
        these anchors. These empirical targets do not guarantee end-to-end
        Recall@k. Hamming and larger k retain uncalibrated interpolation.
        Later expansion to find enough candidates is unchanged. This is
        independent of approx_mode, ef and refinement, and is not a recall
        guarantee. When supplied, effort ignores nprobes, including an explicit
        0.5. With None, an explicitly supplied nprobes remains effective.
    approx_mode: {"fast", "normal", "accurate"}, default "normal"
        Controls the speed / accuracy tradeoff for approximate vector search
        when supported by the selected index. This currently affects
        RQ-quantized indexes (such as IVF_RQ) and HNSW sub-index searches
        that exclude rows (for example through a prefilter or deletions),
        where ``fast`` enables the ACORN traversal. Other index types ignore
        this setting. ``fast`` favors lower latency and may reduce recall,
        ``normal`` uses the default balance, and ``accurate`` favors higher
        recall and may increase latency.
    distance_range: tuple[Optional[float], Optional[float]], optional
        A tuple of (lower_bound, upper_bound) to filter results by distance.
        Both bounds are optional. The lower bound is inclusive and the upper
        bound is exclusive, so (0.0, 1.0) keeps distances d where
        0.0 <= d < 1.0, (None, 0.5) keeps d < 0.5, and (0.5, None) keeps d >= 0.5.
    refine_column, refine_q, refine_metric: optional
        Supply all three to rerank ``k * refine_factor`` coarse candidates
        using another Float32 fixed-size vector column. ``refine_q`` must
        match that column's dimension and the number of coarse queries.
        Requires an explicit positive ``refine_factor``. Final ``_distance``
        and ``distance_range`` use ``refine_metric``. With ``use_index=False``,
        coarse TopM still precedes refinement. Query filters are unsupported;
        scalar filters retain their usual prefilter/postfilter behavior.

    Returns
    -------
    ScannerBuilder
        The scanner builder for method chaining.
    """
    q, q_dim = _coerce_query_vector(q)
    refine_args = (refine_column, refine_q, refine_metric)
    if any(value is not None for value in refine_args):
        if not all(value is not None for value in refine_args):
            raise ValueError(
                "refine_column, refine_q, and refine_metric must be provided together"
            )
        refine_q, _ = _coerce_query_vector(cast("QueryVectorLike", refine_q))

    if dataset is not None:
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
    if search_effort is None and nprobes is not None and int(nprobes) <= 0:
        raise ValueError(f"Nprobes must be > 0 but got {nprobes}")
    if minimum_nprobes is not None and int(minimum_nprobes) < 0:
        raise ValueError(f"Minimum nprobes must be >= 0 but got {minimum_nprobes}")
    if maximum_nprobes is not None and int(maximum_nprobes) < 0:
        raise ValueError(f"Maximum nprobes must be >= 0 but got {maximum_nprobes}")

    if (
        minimum_nprobes is not None
        and maximum_nprobes is not None
        and minimum_nprobes > maximum_nprobes
    ):
        raise ValueError("minimum_nprobes must be <= maximum_nprobes")
    if refine_factor is not None and int(refine_factor) < 1:
        raise ValueError(f"Refine factor must be 1 or more got {refine_factor}")
    if quantized_refine_factor is not None and int(quantized_refine_factor) < 1:
        raise ValueError(
            f"Quantized refine factor must be 1 or more got {quantized_refine_factor}"
        )
    if ef is not None and int(ef) <= 0:
        # `ef` should be >= `k`, but `k` could be None so we can't check it here
        # the rust code will check it
        raise ValueError(f"ef must be > 0 but got {ef}")
    if query_parallelism is not None:
        query_parallelism = operator.index(query_parallelism)

    if query_parallelism is not None and query_parallelism < -1:
        raise ValueError("query_parallelism must be >= -1")

    if approx_mode not in {"fast", "normal", "accurate"}:
        raise ValueError(
            "approx_mode must be one of 'fast', 'normal', or 'accurate', "
            f"got {approx_mode!r}"
        )

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
        "nprobes": nprobes,
        "minimum_nprobes": minimum_nprobes,
        "maximum_nprobes": maximum_nprobes,
        "refine_factor": refine_factor,
        "quantized_refine_factor": quantized_refine_factor,
        "use_index": use_index,
        "ef": ef,
        "query_parallelism": query_parallelism,
        "approx_mode": approx_mode,
        "search_effort": search_effort,
        "distance_range": distance_range,
        "refine_column": refine_column,
        "refine_q": refine_q,
        "refine_metric": refine_metric,
    }


class ScannerBuilder:
    def __init__(self, ds: LanceDataset):
        self.ds = ds
        self._limit = None
        self._filter = None
        self._search_filter = None
        self._substrait_filter = None
        self._prefilter = False
        self._row_addr_allowlist: Optional[bytes] = None
        self._row_addr_blocklist: Optional[bytes] = None
        self._late_materialization = None
        self._blob_handling = None
        self._offset = None
        self._columns = None
        self._columns_with_transform = None
        self._nearest = None
        self._minhash_query: Optional[Dict[str, str]] = None
        self._batch_size: Optional[int] = None
        self._batch_size_bytes: Optional[int] = None
        self._io_buffer_size: Optional[int] = None
        self._batch_readahead: Optional[int] = None
        self._fragment_readahead: Optional[int] = None
        self._scan_in_order = True
        self._fragments = None
        self._index_segments = None
        self._with_row_id = False
        self._with_row_address = False
        self._use_stats = True
        self._fast_search = False
        self._full_text_query = None
        self._use_scalar_index = None
        self._ignored_scalar_indices = None
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

        If a byte limit is also configured, both limits apply and the one
        reached first determines the batch size.
        """
        self._batch_size = batch_size
        return self

    def batch_size_bytes(self, batch_size_bytes: int) -> ScannerBuilder:
        """Set the target batch size in bytes.

        When set, the scanner will produce batches whose total size in bytes
        is approximately this value. If ``batch_size`` is also set, both
        limits apply and the one reached first determines the batch size.

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

        Generally this should scale with the number of concurrent I/O threads.  If
        unset, v2 scans choose a default based on the object store and
        ``LANCE_DEFAULT_IO_BUFFER_SIZE`` can override that default.

        This value is not a hard cap on the amount of RAM the scanner will use.  Some
        space is used for the compute (which can be controlled by the batch size) and
        Lance does not keep track of memory after it is returned to the user.

        This parameter is only used when reading v2 files
        """
        self._io_buffer_size = io_buffer_size
        return self

    def batch_readahead(self, nbatches: Optional[int] = None) -> ScannerBuilder:
        """
        Set the maximum number of batches to decode concurrently.

        This parameter must be greater than zero.
        """
        if nbatches is not None and int(nbatches) <= 0:
            raise ValueError("batch_readahead must be greater than 0")
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
            self._search_filter = PySearchFilter.from_vector_search_query(
                filter.inner()
            )
        elif isinstance(filter, str):
            self._filter = filter
        elif isinstance(filter, pa.compute.Expression):
            self._substrait_filter = _serialize_expression(filter, self.ds._ds.schema)
        else:
            expr_filter = filter.get("expr_filter")
            if expr_filter is not None:
                self.filter(expr_filter)

            search_filter = filter.get("search_filter")
            if search_filter is not None:
                self.filter(search_filter)

        return self

    def row_addr_prefilter(
        self,
        allowlist: Optional[bytes] = None,
        blocklist: Optional[bytes] = None,
    ) -> ScannerBuilder:
        """Restrict the scan to an externally supplied set of row addresses.

        allowlist / blocklist are serialized roaring treemaps over ``_rowid``
        (``RowAddrTreeMap::serialize_into`` output); passing neither clears the
        mask. Applied before KNN / BM25 ranking, so top-k is computed over the
        surviving rows rather than filtered afterwards.

        Bytes rather than an object so the mask can be produced by a different
        extension module -- nothing Rust-typed crosses the boundary.
        """
        self._row_addr_allowlist = allowlist
        self._row_addr_blocklist = blocklist
        return self

    def prefilter(self, prefilter: bool) -> ScannerBuilder:
        """Choose SQL filtering before vector top-k (True) or after it (False).

        Defaults to False. Postfilter does not replace rejected neighbors; prefilter
        does not guarantee exact ANN results. Fragment scope, deletions, and external
        row masks restrict candidates in either mode. See :meth:`LanceDataset.scanner`.
        """
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

    def ignored_scalar_indices(self, index_names: Iterable[str]) -> ScannerBuilder:
        """Ignore named scalar indices from expression-filter planning.

        Other scalar indices remain eligible, and the original filter is still
        evaluated. Names that do not identify an available scalar index have no
        effect. This option is ignored when scalar indices are disabled with
        :meth:`use_scalar_index`.

        Inverted index names are also honored for expression filters such as
        ``contains_tokens``, which fall back to evaluating the function without
        that index. This does not control vector index segments or explicit
        full-text search queries.

        With :meth:`fast_search`, only a filter plan that still uses an index
        skips unindexed fragments. If ignoring indices removes every index query,
        the scan falls back to evaluating the filter over all selected fragments,
        potentially returning more matching rows. Explicit fragment and row
        selections still apply; use them to preserve a caller-defined scan scope.
        """
        index_names = list(index_names)
        if any(not isinstance(name, str) or not name for name in index_names):
            raise ValueError("ignored_scalar_indices must contain non-empty strings")
        self._ignored_scalar_indices = index_names
        return self

    def with_fragments(
        self, fragments: Optional[Iterable[LanceFragment]]
    ) -> ScannerBuilder:
        """Set the search scope, or clear it with None.

        An empty iterable yields no rows.
        Vector search enforces this scope before candidate selection regardless of
        SQL prefilter placement. Fragments outside selected index coverage use flat
        KNN unless fast search is enabled. This can be called before or after
        :meth:`nearest` and :meth:`with_index_segments`.
        See :meth:`LanceDataset.scanner` for a mixed-search example.
        """
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

    def with_index_segments(
        self, index_segments: Optional[Iterable[Union[str, uuid.UUID]]]
    ) -> ScannerBuilder:
        """Select vector index segments, or restore automatic selection with None.

        With index use enabled, UUIDs must identify existing segments of one logical
        index on the query's vector column. Metric compatibility is checked when
        selected segments overlap the fragment scope. With no overlap, ordinary
        search uses flat KNN with the query metric, or the selected index's metric
        when unset, while fast search returns no rows.
        An empty iterable is invalid, and segment selection is only supported for
        vector search.
        With explicit fragments, uncovered fragments use flat KNN unless fast search
        is enabled. Without fragments, indexed search searches only these segments.
        This controls index use rather than flat scans with ``use_index=False``.
        It can be called before or after :meth:`nearest` and :meth:`with_fragments`.
        See :meth:`LanceDataset.scanner` for a mixed-search example.
        """
        self._index_segments = _normalize_index_segment_ids(index_segments)
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
        query_parallelism: Optional[int] = None,
        approx_mode: Literal["fast", "normal", "accurate"] = "normal",
        distance_range: Optional[tuple[Optional[float], Optional[float]]] = None,
        search_effort: Optional[float] = None,
        quantized_refine_factor: Optional[int] = None,
        refine_column: Optional[str] = None,
        refine_q: Optional[QueryVectorLike] = None,
        refine_metric: Optional[str] = None,
    ) -> ScannerBuilder:
        """Configure nearest neighbor search.

        Fragment and segment selection can be configured before or after this method.
        The fragment scope applies with either SQL prefilter or postfilter. With a
        suitable index, uncovered fragments use flat KNN unless fast search is enabled;
        without an index, ordinary search uses flat KNN. See
        :meth:`LanceDataset.scanner` for a mixed-search example and filtering semantics.

        Parameters
        ----------
        q: QueryVectorLike
            A single query vector or, for fixed-size vector columns, a 2-D array-like
            or list-shaped batch of query vectors. Batch queries return up to ``k`` rows
            per query and include Int32 non-null ``query_index`` as the first output
            column. Flattened 1-D inputs whose length is a multiple of the vector
            dimension are rejected. Datasets with an existing ``query_index`` column
            cannot be used for batch search.
            When ``use_index`` is true and a vector index is available, each query
            vector is searched through the index path; otherwise the flat batch path
            is used.
        ef: int, optional
            Beam width for HNSW search. Defaults to ``1.5 * k * refine_factor``,
            floored at 256.
        query_parallelism: int, optional
            Maximum partition-search concurrency for a single vector query.
            The default is 0. Value 0 uses the automatic policy, which
            currently maps to the single-worker sequential path. Value -1 uses
            the CPU pool size. Value 1 uses the single-worker sequential path.
            Values >= 2 use the partition-parallel path and are clamped to the
            CPU pool size.
        search_effort: float, optional
            Initial IVF search budget, a finite value in [0, 1]. None resolves
            to 0.5, preserving Auto. Zero starts at the caller minimum (at least one
            available partition); one starts at all available partitions, subject
            to maximum_nprobes. For L2, cosine and dot with k <= 100000, efforts
            0.1, 0.25 and 0.5 target 80%, 90% and 95% mean partition-routing recall,
            respectively. Intermediate budgets interpolate geometrically between
            these anchors. These empirical targets do not guarantee end-to-end
            Recall@k. Hamming and larger k retain uncalibrated interpolation.
            Later expansion to find enough candidates is unchanged. This is
            independent of approx_mode, ef and refinement, and is not a recall
            guarantee. When supplied, effort ignores nprobes, including an explicit
            0.5. With None, an explicitly supplied nprobes remains effective.
        approx_mode: {"fast", "normal", "accurate"}, default "normal"
            Controls the speed / accuracy tradeoff for approximate vector search
            when supported by the selected index. This currently affects
            RQ-quantized indexes (such as IVF_RQ) and HNSW sub-index searches
            that exclude rows (for example through a prefilter or deletions),
            where ``fast`` enables the ACORN traversal. Other index types ignore
            this setting. ``fast`` favors lower latency and may reduce recall,
            ``normal`` uses the default balance, and ``accurate`` favors higher
            recall and may increase latency.
        quantized_refine_factor: int, optional
            For an IVF_RQ flat index, overfetch this many one-bit candidates per
            requested result and rerank them with the index's most accurate stored
            RQ representation. Unlike ``refine_factor``, indexed candidates do not
            read their original vectors. Unindexed fragments and stale overlay rows
            still use flat fallback and may read original vectors. The two stages can
            be combined; exact refinement remains the final stage and reads original
            vectors for its candidates. With ``distance_range``, selected partitions
            use Accurate RQ search directly so the range applies before top-k
            truncation.
        refine_column, refine_q, refine_metric: optional
            Supply all three to rerank ``k * refine_factor`` coarse candidates
            using another Float32 fixed-size vector column. ``refine_q`` must
            match that column's dimension and the number of coarse queries.
            Requires an explicit positive ``refine_factor``. Final ``_distance``
            and ``distance_range`` use ``refine_metric``. With ``use_index=False``,
            coarse TopM still precedes refinement. Query filters are unsupported;
            scalar filters retain their usual prefilter/postfilter behavior.

        Examples
        --------
        .. code-block:: python

            builder.nearest(
                "pca128",
                q128,
                k=10,
                refine_factor=2,
                refine_column="embedding",
                refine_q=q1024,
                refine_metric="cosine",
            )
        """
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
            quantized_refine_factor=quantized_refine_factor,
            use_index=use_index,
            ef=ef,
            query_parallelism=query_parallelism,
            approx_mode=approx_mode,
            search_effort=search_effort,
            distance_range=distance_range,
            refine_column=refine_column,
            refine_q=refine_q,
            refine_metric=refine_metric,
        )
        return self

    def fast_search(self, flag: bool) -> ScannerBuilder:
        """Set whether searches use only indexed data (default: False).

        Vector search skips flat fallback outside selected segment coverage and stale
        overlay rows requiring replay. The fragment scope and SQL prefilter placement
        are unchanged. No usable vector index coverage produces an empty result.
        Use :meth:`LanceDataset.create_index` or
        :meth:`DatasetOptimizer.optimize_indices` to index newly appended data.
        See :meth:`LanceDataset.scanner` for a mixed-search example.

        For scalar expression filters, this applies only when the filter plan
        uses an index. If no index query remains, including after
        :meth:`ignored_scalar_indices`, the scan falls back to all selected
        fragments while retaining the original filter and explicit row selections.

        Users can use `Table::optimize()` or `create_index()` to include new data
        in an index, thus making new data searchable.
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

    def minhash_search(self, query: MinHashQuery) -> ScannerBuilder:
        """
        Find the rows most similar to ``query.text`` using the MinHash LSH index
        on ``query.column``.

        The number of rows comes from :meth:`limit` (default 10). Results carry
        a ``_distance`` column equal to ``1 - estimated Jaccard similarity`` and
        are ordered by ascending distance. Cannot be combined with
        :meth:`nearest` or :meth:`full_text_search`.

        Rows the index does not cover (fragments appended after the index was
        built, or rows changed by a data overlay) are scored on the fly, which
        costs a scan of those rows per query; run ``optimize_indices`` to index
        them, or enable :meth:`fast_search` to search only the indexed rows.

        Parameters
        ----------
        query : MinHashQuery
            The query text and the column to search.
        """
        if not isinstance(query, MinHashQuery):
            raise TypeError(
                f"minhash_search expects a MinHashQuery, got {type(query).__name__}"
            )
        self._minhash_query = {"column": query.column, "text": query.text}
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

        This cannot be combined with ``batch_size_bytes`` because merging
        batches to the strict row count can exceed the byte limit.
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
            self._index_segments,
            self._with_row_id,
            self._with_row_address,
            self._use_stats,
            self._substrait_filter,
            self._fast_search,
            self._full_text_query,
            self._late_materialization,
            self._blob_handling,
            self._use_scalar_index,
            self._ignored_scalar_indices,
            self._include_deleted_rows,
            self._scan_stats_callback,
            self._strict_batch_size,
            self._orderings,
            self._disable_scoring_autoprojection,
            self._substrait_aggregate,
            self._row_addr_allowlist,
            self._row_addr_blocklist,
            self._minhash_query,
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
        blob_paths = _blob_paths_in_schema(schema)
        if not blob_paths or blob_mode == _BLOB_PANDAS_MODE_DESCRIPTIONS:
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
        blob_files = _fetch_blob_files_for_paths(
            self._ds, table, blob_paths, blob_sources, row_addrs
        )

        columns_to_drop = []
        for path in blob_files:
            segments = _parse_field_path(path)
            if len(segments) == 1 and segments[0] in table.schema.names:
                columns_to_drop.append(segments[0])
        if not requested_rowaddr:
            columns_to_drop.append(_BLOB_ROW_ADDR_COLUMN)
        non_blob_table = (
            table.drop_columns(columns_to_drop) if columns_to_drop else table
        )
        if non_blob_table.num_columns == 0:
            dataframe = pd.DataFrame(index=range(table.num_rows))
        else:
            dataframe = non_blob_table.to_pandas(**kwargs)

        _place_blob_files(dataframe, blob_files, schema)
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
        """Select rows of data by their position in the scan results.

        ``indices`` are 0-based positions within this scanner's output — the
        rows that survive the scanner's filter, in scan order. This matches
        ``pyarrow.dataset.Scanner.take()`` and is different from
        :meth:`LanceDataset.take`, which ignores the scanner filter and takes
        rows by absolute dataset row number.

        Parameters
        ----------
        indices : int or array-like
            Positions of the matching rows to select. A large index requires
            scanning through the matching rows up to it, so the cost is
            proportional to the largest index, not to the number of indices.

        Returns
        -------
        Table
        """
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, pa.Array):
            # pyarrow Scanner.take() also accepts pa.Array; list() on one
            # yields pyarrow Scalars, so normalize to python ints.
            indices = indices.to_pylist()
        indices = list(indices)
        if not indices:
            raise ValueError("take requires at least one index")
        if any(not isinstance(i, (int, np.integer)) or i < 0 for i in indices):
            raise ValueError(f"indices must be non-negative integers, got {indices}")

        # Collect the rows at the requested filtered-relative positions. Rows
        # are gathered in scan order first and re-ordered to the caller's
        # index order at the end, so repeated and unsorted indices work.
        max_index = max(indices)
        sorted_unique = sorted(set(indices))
        batches = []
        matched = 0  # number of matching rows consumed so far
        exhausted = False
        for batch in self.to_batches():
            batch_end = matched + batch.num_rows
            positions = [i - matched for i in sorted_unique if matched <= i < batch_end]
            if positions:
                batches.append(batch.take(positions))
            matched = batch_end
            if matched > max_index:
                exhausted = True
                break
        if not exhausted:
            # The stream ended before covering max_index, so matched is the
            # full row count of the scan output. (count_rows is unavailable
            # here anyway when the scanner has a limit/offset.)
            raise IndexError(
                f"index {max_index} is out of bounds for a scan with {matched} rows"
            )

        # Reaching here means max_index was covered by the scan, so at least
        # one row was collected.
        table = pa.Table.from_batches(batches)
        # rows are in sorted-unique index order; map back to the caller's order.
        row_lookup = {idx: i for i, idx in enumerate(sorted_unique)}
        take_order = [row_lookup[i] for i in indices]
        return table.take(take_order)

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

    def count_rows(self) -> int:
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

    def analyze_plan(self, count_rows: bool = False) -> str:
        """Execute the plan for this scanner and display with runtime metrics.

        Full-text-search nodes include the execution-time ``tokenized_query``
        text and positions.

        Parameters
        ----------
        count_rows : bool, default False
            If True, auto-apply a ``COUNT(*)`` aggregate before analyzing so
            the returned plan reflects what :py:meth:`count_rows` would
            execute (including the optimizer's count-pushdown decisions).

        Returns
        -------
        plan : str
        """

        return self._scanner.analyze_plan(count_rows=count_rows)


class VectorSearchQuery:
    """Vector search parameters, including continuous ``search_effort`` in [0, 1].

    If metric is omitted and use_index is True, use the column's index metric
    when an index exists. Otherwise, use the default for the vector's element type.

    Effort defaults to None, which resolves to 0.5 (Auto). Explicit effort
    ignores nprobes. See :meth:`ScannerBuilder.nearest` for probe bounds and
    other search parameters.
    """

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
        query_parallelism: Optional[int] = None,
        approx_mode: Literal["fast", "normal", "accurate"] = "normal",
        search_effort: Optional[float] = None,
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
            query_parallelism=query_parallelism,
            approx_mode=approx_mode,
            search_effort=search_effort,
        )

    def inner(self):
        return self._inner


@_coerce_source.register(LanceScanner)
def _lance_scanner_source(data_obj: LanceScanner, schema=None) -> _WriteSource:
    if type(data_obj) is not LanceScanner:
        return _WriteSource.from_reader(data_obj.to_reader())
    snapshot = data_obj._snapshot
    replayable = False
    if snapshot is not None:
        defaults = vars(ScannerBuilder(data_obj._ds)).copy()
        defaults.pop("ds")
        # These settings preserve source values and order. Expressions, searches,
        # callbacks, and sorting can change results across otherwise valid scans.
        stable_options = {
            "_columns",
            "_limit",
            "_offset",
            "_batch_size",
            "_batch_size_bytes",
            "_io_buffer_size",
            "_batch_readahead",
            "_fragment_readahead",
            "_fragments",
            "_with_row_id",
            "_with_row_address",
            "_use_stats",
            "_use_scalar_index",
            "_include_deleted_rows",
            "_late_materialization",
            "_blob_handling",
            "_strict_batch_size",
            "_row_addr_allowlist",
            "_row_addr_blocklist",
        }
        replayable = snapshot.keys() == defaults.keys() and all(
            key in stable_options or value == defaults[key]
            for key, value in snapshot.items()
        )
    if replayable:
        return _WriteSource(data_obj.projected_schema, data_obj.to_reader)
    return _WriteSource.from_reader(data_obj.to_reader())
