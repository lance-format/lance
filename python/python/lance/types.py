# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Optional, Union

import pyarrow as pa
from pyarrow import RecordBatch

from . import dataset
from .dependencies import (
    _check_for_hugging_face,
    _check_for_pandas,
    _is_pydantic_base_model,
    _validate_pydantic_list,
    model_to_dict,
)
from .dependencies import pandas as pd

if TYPE_CHECKING:
    ReaderLike = Union[
        pd.Timestamp,
        pa.Table,
        pa.dataset.Dataset,
        pa.dataset.Scanner,
        pa.RecordBatch,
        Iterable[RecordBatch],
        pa.RecordBatchReader,
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


def _is_materialized(data_obj: ReaderLike) -> bool:
    """Whether ``data_obj`` is fully materialized in memory.

    Materialized sources (tables, in-memory frames) can be wrapped in an
    in-memory table for replay without spilling and to expose exact statistics.
    Streaming or re-readable sources (readers, scanners, datasets, generators)
    are not considered materialized.
    """
    if _check_for_pandas(data_obj) and isinstance(data_obj, pd.DataFrame):
        return True
    if isinstance(data_obj, (pa.Table, pa.RecordBatch)):
        return True
    if (
        type(data_obj).__module__.startswith("polars")
        and data_obj.__class__.__name__ == "DataFrame"
    ):
        return True
    if isinstance(data_obj, dict):
        return True
    if (
        isinstance(data_obj, list)
        and len(data_obj) > 0
        and isinstance(data_obj[0], dict)
    ):
        return True
    return False


def _coerce_reader(
    data_obj: ReaderLike, schema: Optional[pa.Schema] = None
) -> pa.RecordBatchReader:
    if _check_for_pandas(data_obj) and isinstance(data_obj, pd.DataFrame):
        return pa.Table.from_pandas(data_obj, schema=schema).to_reader()
    elif isinstance(data_obj, pa.Table):
        return data_obj.to_reader()
    elif isinstance(data_obj, pa.RecordBatch):
        return pa.Table.from_batches([data_obj]).to_reader()
    elif isinstance(data_obj, dataset.LanceDataset):
        return data_obj.scanner().to_reader()
    elif isinstance(data_obj, pa.dataset.Dataset):
        return pa.dataset.Scanner.from_dataset(data_obj).to_reader()
    elif isinstance(data_obj, pa.dataset.Scanner):
        return data_obj.to_reader()
    elif isinstance(data_obj, pa.RecordBatchReader):
        return data_obj
    elif (
        type(data_obj).__module__.startswith("polars")
        and data_obj.__class__.__name__ == "DataFrame"
    ):
        return data_obj.to_arrow().to_reader()
    elif _check_for_hugging_face(data_obj):
        from .dependencies import datasets as hf_datasets

        if isinstance(data_obj, hf_datasets.Dataset):
            if schema is None:
                schema = data_obj.features.arrow_schema
            return data_obj.data.to_reader()
        elif isinstance(data_obj, hf_datasets.DatasetDict):
            raise ValueError(
                "DatasetDict is not yet supported. For now please "
                "iterate through the DatasetDict and pass in single "
                "Dataset instances (e.g., from dataset_dict.data) to "
                "`write_dataset`. "
            )
        elif isinstance(data_obj, hf_datasets.IterableDataset):
            if schema is None:
                schema = data_obj.features.arrow_schema

            def batch_iter():
                # Try to provide a reasonable batch size. If the user needs to
                # override this, they can do the conversion to a reader themselves.
                for dict_batch in data_obj.iter(batch_size=1000):
                    yield pa.RecordBatch.from_pydict(dict_batch, schema=schema)

            return pa.RecordBatchReader.from_batches(schema, batch_iter())
        else:
            raise TypeError(
                f"Unknown HuggingFace dataset type: {type(data_obj)}. "
                "Please provide a single Dataset or DatasetDict."
            )

    elif isinstance(data_obj, dict):
        batch = pa.RecordBatch.from_pydict(data_obj, schema=schema)
        return pa.RecordBatchReader.from_batches(batch.schema, [batch])
    elif (
        isinstance(data_obj, list)
        and len(data_obj) > 0
        and isinstance(data_obj[0], dict)
    ):
        # List of dictionaries
        batch = pa.RecordBatch.from_pylist(data_obj, schema=schema)
        return pa.RecordBatchReader.from_batches(batch.schema, [batch])
    elif (
        isinstance(data_obj, list)
        and len(data_obj) > 0
        and _is_pydantic_base_model(data_obj[0])
    ):
        model_class = type(data_obj[0])
        _validate_pydantic_list(data_obj, model_class)
        if schema is None:
            from .pydantic import pydantic_to_schema

            schema = pydantic_to_schema(model_class)
        dicts = [model_to_dict(item) for item in data_obj]
        batch = pa.RecordBatch.from_pylist(dicts, schema=schema)
        return pa.RecordBatchReader.from_batches(batch.schema, [batch])
    # for other iterables, assume they are of type Iterable[RecordBatch]
    elif isinstance(data_obj, Iterable):
        if schema is not None:
            data = _casting_recordbatch_iter(data_obj, schema)
            return pa.RecordBatchReader.from_batches(schema, data)
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
