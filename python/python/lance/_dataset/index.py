# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Dataset index metadata and readers."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, TypedDict

if TYPE_CHECKING:
    from datetime import datetime

    import pyarrow as pa

    from ..bitmap import Bitmap
    from ..dataset import LanceDataset


@dataclass
class IndexFile:
    """Metadata about a file in an index segment."""

    path: str
    size_bytes: int


@dataclass
class Index:
    """Represents an index in the dataset."""

    uuid: str
    name: str
    fields: List[int]
    dataset_version: int
    fragment_ids: Bitmap
    """The fragments covered by this index. A ``Set[int]``/``List[int]`` is
    also accepted when constructing an ``Index``."""
    index_version: int
    created_at: Optional[datetime] = None
    base_id: Optional[int] = None
    files: Optional[List["IndexFile"]] = None
    index_details: Optional[Tuple[str, bytes]] = None
    covering_fields: List[int] = dataclasses.field(default_factory=list)
    clustering_id: Optional[int] = None
    """The clustering definition this segment was built under, if any."""
    metadata: Dict[str, str] = dataclasses.field(default_factory=dict)
    """Generic metadata attached to this segment. Preserved verbatim, including
    keys this library does not understand."""


class IndexInformation(TypedDict):
    """Information about a single index segment, as returned by
    :meth:`LanceDataset.list_indices`."""

    name: str
    type: str
    uuid: str
    fields: List[str]
    version: int
    fragment_ids: Bitmap
    base_id: Optional[int]


class VectorIndexReader:
    """
    This class allows you to initialize a reader for a specific vector index,
    retrieve the number of partitions,
    and read specific partitions of the index.

    Parameters
    ----------
    dataset: LanceDataset
        The dataset containing the index.
    index_name: str
        The name of the vector index to read.

    Examples
    --------
    .. code-block:: python

        import lance
        from lance._dataset.index import VectorIndexReader
        import numpy as np
        import pyarrow as pa
        vectors = np.random.rand(256, 2)
        data = pa.table({"vector": pa.array(vectors.tolist(),
            type=pa.list_(pa.float32(), 2))})
        dataset = lance.write_dataset(data, "/tmp/index_reader_demo")
        dataset.create_index("vector", index_type="IVF_PQ",
            num_partitions=4, num_sub_vectors=2)
        reader = VectorIndexReader(dataset, "vector_idx")
        assert reader.num_partitions() == 4
        partition = reader.read_partition(0)
        assert "_rowid" in partition.column_names

    Exceptions
    ----------
    ValueError
        If the specified index is not a vector index.
    """

    def __init__(self, dataset: LanceDataset, index_name: str):
        stats = dataset.stats.index_stats(index_name)
        self.dataset = dataset
        self.index_name = index_name
        self.stats = stats
        try:
            self.num_partitions()
        except KeyError:
            raise ValueError(f"Index {index_name} is not vector index")

    def num_partitions(self) -> int:
        """
        Returns the number of partitions in the dataset.

        Returns
        -------
        int
            The number of partitions.
        """

        return self.stats["indices"][0]["num_partitions"]

    def read_partition(
        self, partition_id: int, *, with_vector: bool = False
    ) -> pa.Table:
        """
        Returns a pyarrow table for the given IVF partition

        Parameters
        ----------
        partition_id: int
            The id of the partition to read
        with_vector: bool, default False
            Whether to include the vector column in the reader,
            for IVF_PQ, the vector column is PQ codes

        Returns
        -------
        pa.Table
            A pyarrow table for the given partition,
            containing the row IDs, and quantized vectors (if with_vector is True).
        """

        if partition_id < 0 or partition_id >= self.num_partitions():
            raise IndexError(
                f"Partition id {partition_id} is out of range, "
                f"expected 0 <= partition_id < {self.num_partitions()}"
            )

        return self.dataset._ds.read_index_partition(
            self.index_name, partition_id, with_vector
        ).read_all()
