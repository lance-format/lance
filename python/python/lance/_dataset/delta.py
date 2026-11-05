# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset version delta bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import pyarrow as pa

    from .transaction import Transaction


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

        Raises ``ValueError`` if either nonzero endpoint lacks stable row IDs.
        """
        return self._delta.get_inserted_rows()

    def get_updated_rows(self) -> pa.RecordBatchReader:
        """
        Return a streaming RecordBatchReader for updated rows.

        Raises ``ValueError`` if either nonzero endpoint lacks stable row IDs.
        """
        return self._delta.get_updated_rows()

    def get_deleted_row_ids(self) -> pa.RecordBatchReader:
        """
        Return a streaming RecordBatchReader of the row ids deleted in the range.

        The batches carry a single ``_rowid`` column. Requires stable row ids.
        """
        return self._delta.get_deleted_row_ids()


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
