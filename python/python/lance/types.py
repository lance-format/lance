# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional, Union

import pyarrow as pa
from pyarrow import RecordBatch

from .dependencies import (
    _check_for_hugging_face,
    _check_for_pandas,
    _check_for_polars,
    _is_pydantic_base_model,
    _validate_pydantic_list,
    model_to_dict,
)
from .dependencies import pandas as pd
from .dependencies import polars as pl

if TYPE_CHECKING:
    from pyarrow.dataset import Dataset as ArrowDataset
    from pyarrow.dataset import Scanner as ArrowScanner
    from pydantic import BaseModel

    from .dependencies import datasets

    ReaderLike = Union[
        pd.DataFrame,
        pl.DataFrame,
        pa.Table,
        pa.RecordBatch,
        pa.RecordBatchReader,
        ArrowDataset,
        ArrowScanner,
        datasets.Dataset,
        datasets.IterableDataset,
        dict[str, Any],
        list[dict[str, Any]],
        list[BaseModel],
        Iterable[RecordBatch],
    ]


def _casting_recordbatch_iter(
    input_iter: Iterable[pa.RecordBatch], schema: pa.Schema
) -> Iterable[pa.RecordBatch]:
    """
    Wrapper around an iterator of record batches. If the batches don't match the
    schema, try to cast them to the schema. If that fails, raise an error.

    This is helpful for users who might have written the iterator with default
    data types in PyArrow, but specified more specific types in the schema. For
    example, PyArrow defaults to float64 for floating point types, but Lance
    uses float32 for vectors.
    """
    for batch in input_iter:
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError(f"Expected RecordBatch, got {type(batch)}")
        if batch.schema != schema:
            try:
                # RecordBatch doesn't have a cast method, but table does.
                batch = pa.Table.from_batches([batch]).cast(schema).to_batches()[0]
            except pa.lib.ArrowInvalid:
                raise ValueError(
                    f"Input RecordBatch iterator yielded a batch with schema that "
                    f"does not match the expected schema.\nExpected:\n{schema}\n"
                    f"Got:\n{batch.schema}"
                )
        yield batch


@dataclass(frozen=True)
class _WriteSource:
    """Internal adapter for Python write inputs.

    ``open_reader`` is a zero-argument callable that returns a RecordBatchReader.
    Construction does not call it. A write opens a reader when execution needs
    the input. Each reader must have ``schema``, including its metadata.

    ``mode="replayable"`` promises a fresh reader with the same values and row
    order on every call, including retries. Keep the source unchanged, or pin
    its version, until the operation finishes. The factory can run on a worker
    thread. Factory calls can overlap, so the factory must support concurrent
    calls. Lance owns each returned reader and releases it after execution.
    Cancellation cannot interrupt a factory before it returns. Do not return a
    shared or previously consumed reader.

    ``mode="materialized"`` permits Lance to collect an already resident input
    into an in-memory table. ``mode="streaming"`` supplies one reader and uses
    the existing replay buffer when a merge enables retries. Other write APIs
    consume one reader, regardless of the mode.

    """

    schema: pa.Schema
    open_reader: Callable[[], pa.RecordBatchReader]
    mode: Literal["materialized", "replayable", "streaming"] = "replayable"

    def __post_init__(self):
        if not isinstance(self.schema, pa.Schema):
            raise TypeError(f"schema must be a pyarrow.Schema, got {type(self.schema)}")
        if not callable(self.open_reader):
            raise TypeError("open_reader must be callable")
        if self.mode not in ("materialized", "replayable", "streaming"):
            raise ValueError(f"Unknown write source mode: {self.mode!r}")
        factory = self.open_reader
        schema = self.schema

        def checked_open_reader() -> pa.RecordBatchReader:
            reader = factory()
            if not isinstance(reader, pa.RecordBatchReader):
                raise TypeError(
                    "open_reader must return a pyarrow.RecordBatchReader, "
                    f"got {type(reader)}"
                )
            if not reader.schema.equals(schema, check_metadata=True):
                actual = reader.schema
                reader.close()
                raise ValueError(
                    "Write source reader schema does not match the declared schema."
                    f"\nExpected:\n{schema}\nGot:\n{actual}"
                )
            return reader

        object.__setattr__(self, "open_reader", checked_open_reader)

    @classmethod
    def from_reader(
        cls, reader: pa.RecordBatchReader, *, materialized: bool = False
    ) -> _WriteSource:
        """Adapt one reader, optionally declaring its input already resident."""
        if not isinstance(reader, pa.RecordBatchReader):
            raise TypeError(f"Expected RecordBatchReader, got {type(reader)}")
        return cls(
            reader.schema,
            lambda: reader,
            "materialized" if materialized else "streaming",
        )


@singledispatch
def _coerce_source(data_obj, schema: Optional[pa.Schema] = None) -> _WriteSource:
    """Adapt a write input with the internal type-specific registry.

    Native ordered Lance scans can be rescanned when their settings preserve
    values and row order. Arrow support covers unfiltered built-in
    ``InMemoryDataset``, ``FileSystemDataset``, and unions of these datasets.
    Arrow dataset files must remain unchanged until execution finishes. Filters,
    computed projections, searches, custom Arrow datasets, and opaque scanners
    retain the streaming path when Lance cannot establish stable results.

    """
    if _check_for_pandas(data_obj) and isinstance(data_obj, pd.DataFrame):
        _coerce_source.register(pd.DataFrame, _pandas_source)
        return _pandas_source(data_obj, schema)
    if _check_for_polars(data_obj) and isinstance(data_obj, pl.DataFrame):
        _coerce_source.register(pl.DataFrame, _polars_source)
        return _polars_source(data_obj, schema)
    if _check_for_hugging_face(data_obj):
        _coerce_source.register(type(data_obj), _hugging_face_source)
        return _hugging_face_source(data_obj, schema)
    return _iterable_source(data_obj, schema)


@_coerce_source.register(_WriteSource)
def _write_source(data_obj: _WriteSource, schema=None) -> _WriteSource:
    return data_obj


@_coerce_source.register(pa.Table)
def _table_source(data_obj: pa.Table, schema=None) -> _WriteSource:
    return _WriteSource(data_obj.schema, data_obj.to_reader, "materialized")


@_coerce_source.register(pa.RecordBatch)
def _batch_source(data_obj: pa.RecordBatch, schema=None) -> _WriteSource:
    return _coerce_source(pa.Table.from_batches([data_obj]))


@_coerce_source.register(pa.RecordBatchReader)
def _reader_source(data_obj: pa.RecordBatchReader, schema=None) -> _WriteSource:
    return _WriteSource.from_reader(data_obj)


def _arrow_dataset_replayable(data_obj: pa.dataset.Dataset) -> bool:
    if type(data_obj) not in (
        pa.dataset.InMemoryDataset,
        pa.dataset.FileSystemDataset,
        pa.dataset.UnionDataset,
    ):
        return False
    try:
        # This public API rejects hidden Dataset filters before yielding fragments.
        # Do not consume the iterator or inspect source data to test repeatability.
        data_obj.get_fragments()
    except ValueError:
        return False
    if isinstance(data_obj, pa.dataset.UnionDataset):
        return all(_arrow_dataset_replayable(child) for child in data_obj.children)
    return True


@_coerce_source.register(pa.dataset.Dataset)
def _arrow_dataset_source(data_obj: pa.dataset.Dataset, schema=None) -> _WriteSource:
    scanner = pa.dataset.Scanner.from_dataset(data_obj)
    return _WriteSource(
        scanner.projected_schema,
        scanner.to_reader,
        "replayable" if _arrow_dataset_replayable(data_obj) else "streaming",
    )


@_coerce_source.register(pa.dataset.Scanner)
def _arrow_scanner_source(data_obj: pa.dataset.Scanner, schema=None) -> _WriteSource:
    # Scanner.from_batches is one-shot. Arrow exposes no public capability query.
    return _WriteSource.from_reader(data_obj.to_reader())


@_coerce_source.register(dict)
def _dict_source(data_obj: dict[str, object], schema=None) -> _WriteSource:
    # DatasetDict subclasses dict, so singledispatch reaches this handler first.
    if _check_for_hugging_face(data_obj):
        _coerce_source.register(type(data_obj), _hugging_face_source)
        return _hugging_face_source(data_obj, schema)
    batch = pa.RecordBatch.from_pydict(data_obj, schema=schema)
    return _coerce_source(batch)


@_coerce_source.register(list)
def _list_source(data_obj: list[object], schema=None) -> _WriteSource:
    if data_obj and isinstance(data_obj[0], dict):
        batch = pa.RecordBatch.from_pylist(data_obj, schema=schema)
        return _coerce_source(batch)
    if data_obj and _is_pydantic_base_model(data_obj[0]):
        model_class = type(data_obj[0])
        _validate_pydantic_list(data_obj, model_class)
        if schema is None:
            from .pydantic import pydantic_to_schema

            schema = pydantic_to_schema(model_class)
        dicts = [model_to_dict(item) for item in data_obj]
        batch = pa.RecordBatch.from_pylist(dicts, schema=schema)
        return _WriteSource.from_reader(
            pa.RecordBatchReader.from_batches(batch.schema, [batch])
        )
    return _iterable_source(data_obj, schema)


def _iterable_source(data_obj, schema) -> _WriteSource:
    if isinstance(data_obj, Iterable):
        if schema is not None:
            data = _casting_recordbatch_iter(data_obj, schema)
            return _WriteSource.from_reader(
                pa.RecordBatchReader.from_batches(schema, data)
            )
        else:
            raise ValueError(
                "Must provide schema to write dataset from RecordBatch iterable"
            )
    else:
        raise TypeError(
            f"Unknown data type {type(data_obj)}. "
            "Please check "
            "https://lance.org/guide/read_and_write/ "
            "to see supported types."
        )


def _pandas_source(data_obj, schema=None) -> _WriteSource:
    return _coerce_source(pa.Table.from_pandas(data_obj, schema=schema))


def _polars_source(data_obj, schema=None) -> _WriteSource:
    return _coerce_source(data_obj.to_arrow())


def _hugging_face_source(data_obj, schema=None) -> _WriteSource:
    from .dependencies import datasets as hf_datasets

    if isinstance(data_obj, hf_datasets.Dataset):
        return _WriteSource.from_reader(data_obj.data.to_reader())
    if isinstance(data_obj, hf_datasets.DatasetDict):
        raise ValueError(
            "DatasetDict is not yet supported. For now please "
            "iterate through the DatasetDict and pass in single "
            "Dataset instances (e.g., from dataset_dict.data) to "
            "`write_dataset`. "
        )
    if isinstance(data_obj, hf_datasets.IterableDataset):
        if schema is None:
            schema = data_obj.features.arrow_schema

        def batch_iter():
            # Callers can supply their own reader to control the batch size.
            for dict_batch in data_obj.iter(batch_size=1000):
                yield pa.RecordBatch.from_pydict(dict_batch, schema=schema)

        return _WriteSource.from_reader(
            pa.RecordBatchReader.from_batches(schema, batch_iter())
        )
    raise TypeError(
        f"Unknown HuggingFace dataset type: {type(data_obj)}. "
        "Please provide a single Dataset or DatasetDict."
    )


def _coerce_reader(
    data_obj: ReaderLike, schema: Optional[pa.Schema] = None
) -> pa.RecordBatchReader:
    return _coerce_source(data_obj, schema).open_reader()
