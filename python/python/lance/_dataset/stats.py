# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset and index statistics exposed to Python."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, TypedDict

if TYPE_CHECKING:
    from ..lance import _Dataset


@dataclass
class FieldStatistics:
    """Statistics about a field in the dataset"""

    id: int  #: id of the field
    bytes_on_disk: int  #: (possibly compressed) bytes on disk used to store the field


@dataclass
class DataStatistics:
    """Statistics about the data in the dataset"""

    fields: FieldStatistics  #: Statistics about the fields in the dataset


class DatasetStats(TypedDict):
    num_deleted_rows: int
    num_fragments: int
    num_small_files: int


class LanceStats:
    """
    Statistics about a LanceDataset.
    """

    def __init__(self, dataset: _Dataset):
        self._ds = dataset

    def dataset_stats(self, max_rows_per_group: int = 1024) -> DatasetStats:
        """
        Statistics about the dataset.
        """
        return {
            "num_deleted_rows": self._ds.count_deleted_rows(),
            "num_fragments": self._ds.count_fragments(),
            "num_small_files": self._ds.num_small_files(max_rows_per_group),
        }

    def index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Statistics about an index.

        Parameters
        ----------
        index_name: str
            The name of the index to get statistics for.
        """
        index_stats = json.loads(self._ds.index_statistics(index_name))
        return index_stats

    def data_stats(self) -> DataStatistics:
        """
        Statistics about the data in the dataset.
        """
        return self._ds.data_stats()
