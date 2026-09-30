# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import singledispatch
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

import pyarrow as pa
from pyarrow import RecordBatch

from .dependencies import (
    _check_for_hugging_face,
    _check_for_pandas,
    _is_pydantic_base_model,
    _validate_pydantic_list,
    model_to_dict,
)
from .dependencies import pandas as pd

if TYPE_CHECKING:
    from . import dataset

    ReaderLike = Union[
        pd.Timestamp,
        pa.Table,
        pa.dataset.Dataset,
        pa.dataset.Scanner,
        pa.RecordBatch,
        Iterable[RecordBatch],
        pa.RecordBatchReader,
    ]


class SourceStrategy(Enum):
    """How a write source can be consumed and replayed after a conflict."""

    MATERIALIZED = "materialized"
    """Already in memory; expose batches and statistics without spilling."""
    RESCANNABLE = "rescannable"
    """Each reader observes the same source snapshot without materialization."""
    ONE_SHOT = "one_shot"
    """Can be consumed once; retrying merges may buffer this input."""


@dataclass(frozen=True)
class WriteSource:
    """A registered source's schema, reader factory, and replay strategy.

    For ``RESCANNABLE``, ``reader_factory`` must return a fresh
    :class:`pyarrow.RecordBatchReader` on every call, with the declared schema
    and the same data. It must preserve scan options such as filters and
    projections. For the other strategies, the factory is called once.

    Register a source with :func:`coerce_source` rather than passing this object
    to a write API directly.
    """

    strategy: SourceStrategy
    schema: pa.Schema
    reader_factory: Callable[[], pa.RecordBatchReader]


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


@singledispatch
def coerce_source(
    data_obj: ReaderLike, schema: Optional[pa.Schema] = None
) -> WriteSource:
    """Convert a write input to its reader and replay strategy.

    A registration handles conversion and classification together. Optional
    dependencies are registered on first use, without importing them for Arrow
    inputs. Unregistered iterables retain the one-shot RecordBatch contract.

    Third-party sources can register their own conversion, for example::

        @coerce_source.register(MySnapshot)
        def my_snapshot_source(snapshot, schema=None):
            return WriteSource(
                SourceStrategy.RESCANNABLE,
                snapshot.schema,
                snapshot.to_reader,
            )

    A re-scannable factory must return the same data and schema on each call;
    see :class:`WriteSource`. Register only types that can honor that contract.
    """
    if _check_for_pandas(data_obj):
        if pd.DataFrame not in coerce_source.registry:
            coerce_source.register(pd.DataFrame)(_coerce_pandas)
        handler = coerce_source.dispatch(type(data_obj))
        if handler is not coerce_source.__wrapped__:
            return handler(data_obj, schema)
    if (
        type(data_obj).__module__.startswith("polars")
        and data_obj.__class__.__name__ == "DataFrame"
    ):
        coerce_source.register(type(data_obj))(_coerce_polars)
        return _coerce_polars(data_obj, schema)
    if _check_for_hugging_face(data_obj):
        return _coerce_hugging_face(data_obj, schema)
    if isinstance(data_obj, Iterable):
        return _coerce_batch_iterable(data_obj, schema)
    raise TypeError(
        f"Unknown data type {type(data_obj)}. "
        "Please check "
        "https://lance.org/guide/read_and_write/ "
        "to see supported types."
    )


@coerce_source.register(pa.Table)
def _coerce_table(data_obj, schema=None) -> WriteSource:
    return WriteSource(SourceStrategy.MATERIALIZED, data_obj.schema, data_obj.to_reader)


@coerce_source.register(pa.RecordBatch)
def _coerce_batch(data_obj, schema=None) -> WriteSource:
    return coerce_source(pa.Table.from_batches([data_obj]))


@coerce_source.register(pa.dataset.Dataset)
def _coerce_dataset(data_obj, schema=None) -> WriteSource:
    return coerce_source(pa.dataset.Scanner.from_dataset(data_obj))


@coerce_source.register(pa.dataset.Scanner)
def _coerce_scanner(data_obj, schema=None) -> WriteSource:
    # Scanner.from_batches wraps a one-shot input, and PyArrow exposes no
    # non-consuming replayability check for an arbitrary Scanner.
    return WriteSource(
        SourceStrategy.ONE_SHOT, data_obj.projected_schema, data_obj.to_reader
    )


@coerce_source.register(pa.RecordBatchReader)
def _coerce_recordbatch_reader(data_obj, schema=None) -> WriteSource:
    return WriteSource(SourceStrategy.ONE_SHOT, data_obj.schema, lambda: data_obj)


@coerce_source.register(dict)
def _coerce_dict(data_obj, schema=None) -> WriteSource:
    # HuggingFace DatasetDict inherits dict; register it before Arrow attempts
    # to interpret its datasets as column values.
    if _check_for_hugging_face(data_obj):
        return _coerce_hugging_face(data_obj, schema)
    return coerce_source(pa.RecordBatch.from_pydict(data_obj, schema=schema))


@coerce_source.register(list)
def _coerce_list(data_obj, schema=None) -> WriteSource:
    if data_obj and isinstance(data_obj[0], dict):
        return coerce_source(pa.RecordBatch.from_pylist(data_obj, schema=schema))
    if data_obj and _is_pydantic_base_model(data_obj[0]):
        model_class = type(data_obj[0])
        _validate_pydantic_list(data_obj, model_class)
        if schema is None:
            from .pydantic import pydantic_to_schema

            schema = pydantic_to_schema(model_class)
        dicts = [model_to_dict(item) for item in data_obj]
        batch = pa.RecordBatch.from_pylist(dicts, schema=schema)
        return coerce_source(pa.RecordBatchReader.from_batches(batch.schema, [batch]))
    return _coerce_batch_iterable(data_obj, schema)


def _coerce_batch_iterable(data_obj, schema) -> WriteSource:
    if schema is None:
        raise ValueError(
            "Must provide schema to write dataset from RecordBatch iterable"
        )
    data = _casting_recordbatch_iter(data_obj, schema)
    return coerce_source(pa.RecordBatchReader.from_batches(schema, data))


def _coerce_pandas(data_obj, schema=None) -> WriteSource:
    return coerce_source(pa.Table.from_pandas(data_obj, schema=schema))


def _coerce_polars(data_obj, schema=None) -> WriteSource:
    return coerce_source(data_obj.to_arrow())


def _coerce_hugging_face(data_obj, schema) -> WriteSource:
    from .dependencies import datasets as hf_datasets

    for source_type, adapter in (
        (hf_datasets.Dataset, _coerce_hf_dataset),
        (hf_datasets.DatasetDict, _coerce_hf_dataset_dict),
        (hf_datasets.IterableDataset, _coerce_hf_iterable),
    ):
        if source_type not in coerce_source.registry:
            coerce_source.register(source_type)(adapter)
    handler = coerce_source.dispatch(type(data_obj))
    if handler in (coerce_source.__wrapped__, _coerce_dict):
        raise TypeError(
            f"Unknown HuggingFace dataset type: {type(data_obj)}. "
            "Please provide a single Dataset or DatasetDict."
        )
    return handler(data_obj, schema)


def _coerce_hf_dataset(data_obj, schema=None) -> WriteSource:
    return coerce_source(data_obj.data.to_reader())


def _coerce_hf_dataset_dict(data_obj, schema=None) -> WriteSource:
    raise ValueError(
        "DatasetDict is not yet supported. For now please "
        "iterate through the DatasetDict and pass in single "
        "Dataset instances (e.g., from dataset_dict.data) to "
        "`write_dataset`. "
    )


def _coerce_hf_iterable(data_obj, schema=None) -> WriteSource:
    if schema is None:
        schema = data_obj.features.arrow_schema

    def batch_iter():
        # Keep the existing default chunk size; callers can construct a reader
        # themselves when they need a different size.
        for dict_batch in data_obj.iter(batch_size=1000):
            yield pa.RecordBatch.from_pydict(dict_batch, schema=schema)

    return coerce_source(pa.RecordBatchReader.from_batches(schema, batch_iter()))


def _coerce_lance_dataset(data_obj: dataset.LanceDataset, schema=None) -> WriteSource:
    # Capture one scanner now: re-reading the mutable Python Dataset object on
    # each retry could observe a newer version after another operation.
    return _coerce_lance_scanner(data_obj.scanner())


def _coerce_lance_scanner(data_obj: dataset.LanceScanner, schema=None) -> WriteSource:
    strategy = (
        SourceStrategy.RESCANNABLE
        if data_obj._scanner.is_repeatable()
        else SourceStrategy.ONE_SHOT
    )
    return WriteSource(strategy, data_obj.projected_schema, data_obj.to_reader)


def _coerce_reader(
    data_obj: ReaderLike, schema: Optional[pa.Schema] = None
) -> pa.RecordBatchReader:
    return coerce_source(data_obj, schema).reader_factory()
