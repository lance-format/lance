# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Tests for the ``BatchUDF`` public contract.

This module is in the pyright target configured in ``pyproject.toml``, so it
doubles as a strict client for the signatures: the annotated locals below fail
the repository type check if a parameter or return type is narrowed or widened
incorrectly -- narrowing ``checkpoint_file`` back to ``str``, for example,
reports

    error: Argument of type "Path" cannot be assigned to parameter
           "checkpoint_file" of type "str | None"

It cannot catch the annotations being *deleted*, because pyright infers the
same types from the function bodies. That failure mode is mypy-specific
(``no-untyped-call``); policing it would take
``reportMissingParameterType`` on ``lance/udf.py``, which needs one more
parameter annotated and a pre-existing narrowing diagnostic resolved.
"""

from pathlib import Path

import pyarrow as pa
from lance.udf import BatchUDF, batch_udf


def _add_doubled(batch: pa.RecordBatch) -> pa.RecordBatch:
    doubled = [value * 2 for value in batch.column("a").to_pylist()]
    return pa.RecordBatch.from_pydict({"doubled": doubled})


def test_batch_udf_constructor(tmp_path: Path) -> None:
    output_schema = pa.schema([pa.field("doubled", pa.int64())])

    udf: BatchUDF = BatchUDF(_add_doubled, output_schema=output_schema)
    assert udf.output_schema == output_schema
    assert udf.cache is None

    # checkpoint_file accepts a Path as well as a str.
    checkpointed: BatchUDF = BatchUDF(
        _add_doubled,
        output_schema=output_schema,
        checkpoint_file=tmp_path / "checkpoint.sqlite",
    )
    assert checkpointed.cache is not None


def test_batch_udf_decorator() -> None:
    output_schema = pa.schema([pa.field("doubled", pa.int64())])

    @batch_udf(output_schema=output_schema)
    def doubled(batch: pa.RecordBatch) -> pa.RecordBatch:
        return _add_doubled(batch)

    # The decorator returns a BatchUDF, not the original function.
    udf: BatchUDF = doubled
    assert udf.output_schema == output_schema

    # Calling it delegates straight to the wrapped function, so a UDF stays
    # testable on its own.
    result = udf(pa.RecordBatch.from_pydict({"a": [1, 2, 3]}))
    assert result.column("doubled").to_pylist() == [2, 4, 6]
