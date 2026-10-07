# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import os
import pickle
import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING, Any, Callable, Dict, List, NamedTuple, Optional, Union

import pyarrow as pa

from .dependencies import (
    _check_for_pandas,
)
from .dependencies import pandas as pd
from .types import _coerce_reader

if TYPE_CHECKING:
    from pathlib import Path

    from .dataset import LanceDataset, LanceFragment
    from .types import ReaderLike


class BatchUDF:
    """A user-defined function that can be passed to :meth:`LanceDataset.add_columns`.

    Use :func:`lance.add_columns_udf` decorator to wrap a function with this class.
    """

    def __init__(
        self,
        func: Callable[[pa.RecordBatch], Any],
        output_schema: Optional[pa.Schema] = None,
        checkpoint_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.func = func
        self.output_schema = output_schema
        self.cache: Optional[BatchUDFCheckpoint]
        if checkpoint_file is not None:
            self.cache = BatchUDFCheckpoint(checkpoint_file)
        else:
            self.cache = None

    def __call__(self, batch: pa.RecordBatch) -> Any:
        # Directly call inner function. This is to allow the user to test the
        # function and have it behave exactly as it was written.
        return self.func(batch)

    def _call(self, batch: pa.RecordBatch) -> pa.RecordBatch:
        if self.output_schema is None:
            raise ValueError(
                "output_schema must be provided when using a function that "
                "returns a RecordBatch"
            )
        result = self.func(batch)

        if _check_for_pandas(result):
            if isinstance(result, pd.DataFrame):
                result = pa.RecordBatch.from_pandas(result)
        assert result.schema == self.output_schema, (
            f"Output schema of function does not match the expected schema. "
            f"Expected:\n{self.output_schema}\nGot:\n{result.schema}"
        )
        return result


def batch_udf(
    output_schema: Optional[pa.Schema] = None,
    checkpoint_file: Optional[Union[str, Path]] = None,
) -> Callable[[Callable[[pa.RecordBatch], Any]], BatchUDF]:
    """
    Create a user defined function (UDF) that adds columns to a dataset.

    This function is used to add columns to a dataset. It takes a function that
    takes a single argument, a RecordBatch, and returns a RecordBatch. The
    function is called once for each batch in the dataset. The function should
    not modify the input batch, but instead create a new batch with the new
    columns added.

    Parameters
    ----------
    output_schema : Schema, optional
        The schema of the output RecordBatch. This is used to validate the
        output of the function. If not provided, the schema of the first output
        RecordBatch will be used.
    checkpoint_file : str or Path, optional
        If specified, this file will be used as a cache for unsaved results of
        this UDF. If the process fails, and you call add_columns again with this
        same file, it will resume from the last saved state. This is useful for
        long running processes that may fail and need to be resumed. This file
        may get very large. It will hold up to an entire data files' worth of
        results on disk, which can be multiple gigabytes of data.

    Returns
    -------
    AddColumnsUDF
    """

    def inner(func: Callable[[pa.RecordBatch], Any]) -> BatchUDF:
        return BatchUDF(func, output_schema, checkpoint_file)

    return inner


class BatchUDFCheckpoint:
    """A cache for BatchUDF results to avoid recomputation.

    This is backed by a SQLite database.
    """

    class BatchInfo(NamedTuple):
        fragment_id: int
        batch_index: int

    def __init__(self, path):
        self.path = path
        # We don't re-use the connection because it's not thread safe.
        # Each method creates and closes its own connection.
        # Note: sqlite3's context manager only handles transactions, not connection
        # closing. We use closing() to ensure connections are closed, which is
        # required on Windows to avoid file locking issues.
        with closing(sqlite3.connect(path)) as conn:
            # One table to store the results for each batch.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batches
                (fragment_id INT, batch_index INT, result BLOB)
                """
            )
            # One table to store fully written (but not committed) fragments.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS fragments (fragment_id INT, data BLOB)"
            )
            conn.commit()

    def cleanup(self):
        os.remove(self.path)

    def get_batch(self, info: BatchInfo) -> Optional[pa.RecordBatch]:
        with closing(sqlite3.connect(self.path)) as conn:
            cursor = conn.execute(
                "SELECT result FROM batches WHERE fragment_id = ? AND batch_index = ?",
                (info.fragment_id, info.batch_index),
            )
            row = cursor.fetchone()
            if row is not None:
                return pickle.loads(row[0])
            return None

    def insert_batch(self, info: BatchInfo, batch: pa.RecordBatch):
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "INSERT INTO batches (fragment_id, batch_index, result) "
                "VALUES (?, ?, ?)",
                (info.fragment_id, info.batch_index, pickle.dumps(batch)),
            )
            conn.commit()

    def get_fragment(self, fragment_id: int) -> Optional[str]:
        """Retrieves a fragment as a JSON string."""
        with closing(sqlite3.connect(self.path)) as conn:
            cursor = conn.execute(
                "SELECT data FROM fragments WHERE fragment_id = ?", (fragment_id,)
            )
            row = cursor.fetchone()
            if row is not None:
                return row[0]
            return None

    def insert_fragment(self, fragment_id: int, fragment: str):
        """Save a JSON string of a fragment to the cache."""
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute(
                "INSERT INTO fragments (fragment_id, data) VALUES (?, ?)",
                (fragment_id, fragment),
            )
            # Clear all batches for the fragment
            conn.execute("DELETE FROM batches WHERE fragment_id = ?", (fragment_id,))
            conn.commit()


def normalize_transform(
    udf_like: Dict[str, str] | BatchUDF | ReaderLike,
    data_source: LanceDataset | LanceFragment,
    read_columns: Optional[List[str]] = None,
    reader_schema: Optional[pa.Schema] = None,
):
    filtered_read_columns = None
    with_row_id = False
    with_row_address = False
    if read_columns is not None:
        filtered_read_columns = []
        for col in read_columns:
            if col == "_rowid":
                with_row_id = True
            elif col == "_rowaddr":
                with_row_address = True
            else:
                filtered_read_columns.append(col)

    if isinstance(udf_like, BatchUDF):
        if udf_like.output_schema is None:
            # Infer the schema based on the first batch
            sample_batch = udf_like(
                next(
                    iter(
                        data_source.to_batches(
                            limit=1,
                            columns=filtered_read_columns,
                            with_row_id=with_row_id,
                            with_row_address=with_row_address,
                        )
                    )
                )
            )
            if _check_for_pandas(sample_batch) and isinstance(
                sample_batch, pd.DataFrame
            ):
                sample_batch = pa.RecordBatch.from_pandas(sample_batch)
            udf_like.output_schema = sample_batch.schema

        return udf_like
    elif isinstance(udf_like, dict):
        for k, v in udf_like.items():
            if not isinstance(k, str):
                raise TypeError(f"Column names must be a string. Got {type(k)}")
            if not isinstance(v, str):
                raise TypeError(f"Column expressions must be a string. Got {type(k)}")

        return udf_like
    # Is this a callable/function that is not a BatchUDF?  If so, wrap in a BatchUDF
    elif callable(udf_like):
        try:
            sample_batch = udf_like(
                next(
                    iter(
                        data_source.to_batches(
                            limit=1,
                            columns=filtered_read_columns,
                            with_row_id=with_row_id,
                            with_row_address=with_row_address,
                        )
                    )
                )
            )
            if _check_for_pandas(sample_batch) and isinstance(
                sample_batch, pd.DataFrame
            ):
                sample_batch = pa.RecordBatch.from_pandas(sample_batch)
            udf_like = BatchUDF(udf_like, output_schema=sample_batch.schema)

            return udf_like
        except Exception as inner_err:
            raise TypeError(
                "transforms must be a BatchUDF, dict, map function, or ReaderLike "
                f"value.  Received {type(udf_like)}, which is callable, but gave "
                f"an error when called with a batch of data: {inner_err}"
            )
    # Last thing we check is to see if we can coerce into a RecordBatchReader
    else:
        try:
            reader = _coerce_reader(udf_like, reader_schema)
            return reader

        except TypeError as inner_err:
            raise TypeError(
                "transforms must be a BatchUDF, dict, map function, or ReaderLike  "
                f"value.  Received {type(udf_like)}.  Could not coerce to a "
                f"reader: {inner_err}"
            )
