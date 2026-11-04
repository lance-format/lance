# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset and index statistics exposed to Python."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict

import pyarrow as pa

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


class FragmentColumnLayoutStats(TypedDict):
    fragment_id: int
    #: Data files holding at least one column of the schema. A file holding
    #: only tombstones, dropped columns, or spilled row lineage is not counted.
    live_file_count: int
    #: Recorded size in bytes of each data file, in the fragment's file order;
    #: ``None`` when the manifest has no size for the file.
    file_sizes: List[Optional[int]]
    #: Number of schema fields each data file holds, in the fragment's file order.
    fields_per_file: List[int]
    #: Share of the fragment's field slots holding no live data (tombstoned by a
    #: column update, or left by a dropped column). Spilled row lineage is not
    #: counted. A fragment rewrite reclaims these slots.
    dead_slot_ratio: float
    #: Data overlay files attached to the fragment.
    overlay_count: int


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

    def zone_statistics(self, index_name: str) -> pa.Table:
        """
        Per-zone statistics for a ZoneMap index.

        The table contains the persisted index entries and does not scan data
        files. It is not a snapshot of the current dataset: ``fragment_id`` and
        ``zone_start`` use the original index address domain, ``zone_length``
        is the recorded offset span rather than a post-deletion row count, and
        fragment reuse, data overlays, and later appends are not remapped or
        represented.

        Parameters
        ----------
        index_name : str
            Name of the ZoneMap index.

        Returns
        -------
        pyarrow.Table
            One row per zone. Columns are ``min``, ``max``, ``null_count``,
            ``nan_count``, ``negative_nan_count``, ``fragment_id``,
            ``zone_start``, and ``zone_length``. ``zone_length`` is an offset
            span. ``negative_nan_count`` is null for a legacy zone with NaNs
            whose sign split was not persisted; it must not be read as zero.

            The ``rows_per_zone`` schema metadata is present only when every
            returned segment uses the same value. It is omitted when segments
            of one index use different values.

        Raises
        ------
        ValueError
            If ``index_name`` does not exist or is not a ZoneMap index.

        Examples
        --------
        .. code-block:: python

            zones = dataset.stats.zone_statistics("x_zonemap")
            print(
                zones.select(
                    [
                        "fragment_id",
                        "zone_start",
                        "zone_length",
                        "negative_nan_count",
                    ]
                )
            )
        """
        return pa.Table.from_batches([self._ds.zone_statistics(index_name)])

    def column_layout_stats(self) -> List[FragmentColumnLayoutStats]:
        """
        How each fragment's columns are laid out across data files, in
        manifest fragment order. Only manifest metadata is read.

        This is an experimental API. It may change at any time.
        """
        return self._ds.column_layout_stats()
