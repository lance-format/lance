# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Type-checking regression tests for optional dependency input types.

This module is part of the pyright target configured in ``pyproject.toml``, so
a regression in the annotations below fails the repository type check and not
only the runtime suite.

The rejected cases are pinned with ``pyright: ignore`` comments and
``reportUnnecessaryTypeIgnoreComment``, so the file also fails if an annotation
becomes *too* permissive and one of them stops being an error. They sit in a
``TYPE_CHECKING`` block because they are invalid at runtime; the matching
runtime assertions live in ``test_dataset.py``.

Like ``test_fragment_typing.py``, this module does not import ``pytest``: the
lint workflow installs pyright without the test dependencies.
"""

# pyright: reportUnnecessaryTypeIgnoreComment=true

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
from lance.util import sanitize_ts

if TYPE_CHECKING:
    import lance
    import polars as pl
    import pyarrow as pa
    from lance.types import ReaderLike
    from pyarrow.dataset import Dataset as ArrowDataset
    from pyarrow.dataset import Scanner as ArrowScanner
    from pydantic import BaseModel

    def _accept_reader(reader: ReaderLike) -> None:
        pass

    def _check_reader_types(
        pandas_dataframe: pd.DataFrame,
        polars_dataframe: pl.DataFrame,
        arrow_dataset: ArrowDataset,
        arrow_scanner: ArrowScanner,
        lance_dataset: lance.LanceDataset,
        table: pa.Table,
        batch: pa.RecordBatch,
        reader: pa.RecordBatchReader,
        batches: list[pa.RecordBatch],
        models: list[BaseModel],
    ) -> None:
        # One case per branch of `lance.types._coerce_reader`.
        _accept_reader(pandas_dataframe)
        _accept_reader(polars_dataframe)
        _accept_reader(arrow_dataset)
        _accept_reader(arrow_scanner)
        _accept_reader(lance_dataset)
        _accept_reader(table)
        _accept_reader(batch)
        _accept_reader(reader)
        _accept_reader(batches)
        _accept_reader(models)
        _accept_reader({"a": [1.0, 2.0]})
        _accept_reader([{"a": 1.0}, {"a": 2.0}])
        # No rejected case here: `ReaderLike` names pyarrow types, which are
        # unresolved without `pyarrow-stubs`, and a union with an unresolved
        # member accepts anything. The `asof` pins below have no such member.

    def _check_accepted_asof_types() -> None:
        # ``sanitize_ts`` is exercised at runtime below; this pins the public
        # entry point that forwards to it.
        lance.dataset("memory://unused", asof=pd.Timestamp("2026-01-01"))
        lance.dataset("memory://unused", asof=pd.NaT)
        lance.dataset("memory://unused", asof=datetime(2026, 1, 1))
        lance.dataset("memory://unused", asof="2026-01-01")

    def _check_rejected_asof_types() -> None:
        # `DatetimeIndex` has a `to_pydatetime` method, so it satisfies a
        # structural timestamp annotation without being a valid instant.
        index = pd.DatetimeIndex(["2026-01-01"])
        lance.dataset(
            "memory://unused",
            asof=index,  # pyright: ignore[reportArgumentType]
        )
        sanitize_ts(object())  # pyright: ignore[reportArgumentType]


def test_sanitize_ts_accepts_pandas_timestamp() -> None:
    # The stubs bundled with pandas type this constructor as
    # ``Timestamp | NaTType``, so both halves have to satisfy ``ts_types``.
    result: datetime = sanitize_ts(pd.Timestamp("2026-01-01"))

    assert result == datetime(2026, 1, 1)


def test_sanitize_ts_accepts_datetime_and_str() -> None:
    assert sanitize_ts(datetime(2026, 1, 1)) == datetime(2026, 1, 1)
    assert sanitize_ts("2026-01-01 00:00:00") == datetime(2026, 1, 1)
