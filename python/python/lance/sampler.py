# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# PEP-585. Can be removed after deprecating python 3.8 support.
from __future__ import annotations

import gc
import math
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from heapq import heappush, heappushpop
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

import pyarrow as pa
import pyarrow.compute as pc

import lance
from lance.dependencies import numpy as np
from lance.log import LOGGER

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    "maybe_sample",
    "Sampler",
    "FragmentSampler",
    "FullScanSampler",
    "ShardedFragmentSampler",
    "ShardedBatchSampler",
]


def _efficient_sample(
    dataset: lance.LanceDataset,
    n: int,
    columns: Optional[Union[List[str], Dict[str, str]]],
    batch_size: int,
    max_takes: int,
) -> Generator[pa.RecordBatch, None, None]:
    """Sample n records from the dataset.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset to sample from.
    n : int
        The number of records to sample.
    columns : list[str]
        The columns to load.
    batch_size : int
        The batch size to use when loading the data.
    max_takes : int
        The maximum number of takes to perform. This is used to limit the number of
        random reads. Large enough value can give a good random sample without
        having to issue too many random reads.

    Returns
    -------
    Generator of a RecordBatch.
    """
    buf: list[pa.RecordBatch] = []
    total_records = len(dataset)
    assert total_records > n
    chunk_size = total_records // max_takes
    chunk_sample_size = n // max_takes

    num_sampled = 0

    for idx, i in enumerate(range(0, total_records, chunk_size)):
        # If we have already sampled enough, break. This can happen if there
        # is a remainder in the division.
        if num_sampled >= n:
            break
        num_sampled += chunk_sample_size

        # If we are at the last chunk, we may not have enough records to sample.
        local_size = min(chunk_size, total_records - i)
        local_sample_size = min(chunk_sample_size, local_size)

        if local_sample_size < local_size:
            # Add more randomness within each chunk, if there is room.
            offset = i + np.random.randint(0, local_size - local_sample_size)
        else:
            offset = i

        buf.extend(
            dataset.take(
                list(range(offset, offset + local_sample_size)),
                columns=columns,
            ).to_batches()
        )
        if idx % 50 == 0:
            LOGGER.info("Sampled at offset=%s, len=%s", offset, chunk_sample_size)
        if sum(len(b) for b in buf) >= batch_size:
            tbl = pa.Table.from_batches(buf)
            buf.clear()
            tbl = tbl.combine_chunks()
            yield tbl.to_batches()[0]
            del tbl
    if buf:
        tbl = pa.Table.from_batches(buf).combine_chunks()
        yield tbl.to_batches()[0]
        del tbl


def _filtered_efficient_sample(
    dataset: lance.LanceDataset,
    n: int,
    columns: List[str],
    batch_size: int,
    target_takes: int,
    filter: str,
) -> Generator[pa.RecordBatch, None, None]:
    total_records = len(dataset)
    shard_size = math.ceil(n / target_takes)
    num_shards = math.ceil(total_records / shard_size)

    shards = list(range(num_shards))
    random.shuffle(shards)

    tables = []
    remaining_rows = n
    remaining_in_batch = min(batch_size, n)
    for shard in shards:
        start = shard * shard_size
        end = min(start + shard_size, total_records)
        table = dataset.to_table(
            columns=columns,
            offset=start,
            limit=(end - start),
            batch_size=shard_size,
        )
        if len(columns) == 1 and filter.lower() == f"{columns[0]} is not null".lower():
            table = pc.drop_null(table)
        elif filter is not None:
            raise NotImplementedError(f"Can't yet run filter <{filter}> in-memory")
        if table.num_rows > 0:
            if table.num_rows > remaining_rows:
                table = table.slice(0, remaining_rows)
            tables.append(table)
            remaining_rows -= table.num_rows
            remaining_in_batch = remaining_in_batch - table.num_rows
            if remaining_in_batch <= 0:
                combined = pa.concat_tables(tables).combine_chunks()
                batch = combined.slice(0, batch_size).to_batches()[0]
                yield batch
                remaining_in_batch = min(batch_size, remaining_rows)
                if len(combined) > batch_size:
                    leftover = combined.slice(batch_size)
                    tables = [leftover]
                    remaining_in_batch -= len(leftover)
                else:
                    tables = []
            if remaining_rows <= 0:
                break


def maybe_sample(
    dataset: Union[str, Path, lance.LanceDataset],
    n: int,
    columns: Union[list[str], str],
    batch_size: int = 10240,
    max_takes: int = 2048,
    filt: Optional[str] = None,
) -> Generator[pa.RecordBatch, None, None]:
    """Sample n records from the dataset.

    Parameters
    ----------
    dataset : Union[str, Path, lance.LanceDataset]
        The dataset to sample from.
    n : int
        The number of records to sample.
    columns : Union[list[str], dict[str, str], str]
        The columns to load.
    batch_size : int, optional
        The batch size to use when loading the data, by default 10240.
    max_takes : int, optional
        The maximum number of takes to perform, by default 2048.
        This is employed to minimize the number of random reads necessary for sampling.
        A sufficiently large value can provide an effective random sample without
        the need for excessive random reads.
    filt : str, optional
        The filter to apply to the dataset, by default None.  If a filter is provided,
        then we will first load all row ids in memory and then batch through the ids
        in random order until enough matches have been found.

    Returns
    -------
    Generator[pa.RecordBatch]
        A generator that yields [RecordBatch] of data.
    """
    if isinstance(dataset, (str, Path)):
        dataset = lance.dataset(dataset)

    if isinstance(columns, str):
        columns = [columns]

    if n >= len(dataset):
        # Dont have enough data in the dataset. Just do a full scan
        yield from dataset.to_batches(
            columns=columns, batch_size=batch_size, filter=filt
        )
    elif filt is not None:
        yield from _filtered_efficient_sample(
            dataset, n, columns, batch_size, max_takes, filt
        )
    elif n > max_takes:
        yield from _efficient_sample(dataset, n, columns, batch_size, max_takes)
    else:
        choices = np.random.choice(len(dataset), n, replace=False)
        idx = 0
        while idx < len(choices):
            end = min(idx + batch_size, len(choices))
            tbl = dataset.take(choices[idx:end], columns=columns).combine_chunks()
            yield tbl.to_batches()[0]
            idx += batch_size


T = TypeVar("T")


@dataclass(order=True)
class PrioritizedItem(Generic[T]):
    priority: int
    item: T = field(compare=False)


def reservoir_sampling(stream: Iterable[T], k: int) -> list[T]:
    rng = np.random.default_rng()
    heap = []
    for idx, item in enumerate(stream):
        entry = PrioritizedItem(rng.integers(0, k * 2), item)
        if len(heap) < k:
            heappush(heap, entry)
        else:
            vic = heappushpop(heap, entry)
            del vic
        if idx % 10240 == 0:
            LOGGER.info("Force Python GC")
            gc.collect()
    samples = [i.item for i in heap]
    del heap
    return samples


class Sampler(ABC):
    """Sampler over LanceDataset.

    To implement a new `Sampler`, you can implement the `__call__` method to yield
    a `pyarrow.RecordBatch`.
    """

    @abstractmethod
    def __call__(
        self,
        ds: lance.LanceDataset,
        *args,
        batch_size: int = 128,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[str] = None,
        batch_readahead: int = 16,
        with_row_id: bool = False,
        **kwargs,
    ) -> Generator[pa.RecordBatch, None, None]:
        """A generator to yield `pyarrow.RecordBatch` from the dataset."""
        pass


class FragmentSampler(Sampler):
    """Sampling over Fragments.

    To implement a new `FragmentSampler`, you can implement the `iter_fragments` method
    to yield fragments in desired order.
    """

    def __call__(
        self,
        dataset: lance.LanceDataset,
        *args,
        batch_size: int = 128,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[str] = None,
        batch_readahead: int = 16,
        with_row_id: bool = False,
        **kwargs,
    ) -> Generator[pa.RecordBatch, None, None]:
        fragments = self.iter_fragments(dataset, *args, **kwargs)
        scanner = dataset.scanner(
            batch_size=batch_size,
            columns=columns,
            filter=filter,
            with_row_id=with_row_id,
            batch_readahead=batch_readahead,
            fragments=list(fragments),
        )
        yield from scanner.to_batches()

    @abstractmethod
    def iter_fragments(
        self, ds: lance.LanceDataset, *args, **kwargs
    ) -> Generator[lance.LanceFragment, None, None]:
        """Iterate over data fragments."""
        pass


class FullScanSampler(FragmentSampler):
    """Default Sampler, which scan the entire dataset sequentially."""

    def iter_fragments(
        self, dataset: lance.LanceDataset, **kwargs
    ) -> Generator[lance.LanceFragment, None, None]:
        for fragment in dataset.get_fragments():
            yield fragment


class ShardedFragmentSampler(FragmentSampler):
    """Sharded fragments by rank and world_size.

    Each rank / process will process a subset of the fragments.
    It yields batches from ``ds.fragments[rank::world_size]``.

    This sampler is more efficient than `ShardedBatchSampler` when the dataset is large.

    Parameters
    ----------
    rank : int
        The rank of the process in the distributed cluster.
    world_size : int
        The total number of processes in the distributed cluster.
    randomize : bool
        If set true, randomize
    seed : int
        The random seed to use when randomize is set true.
    """

    def __init__(
        self, rank: int, world_size: int, randomize: bool = False, seed: int = 0
    ):
        super().__init__()

        self._rank = rank
        self._world_size = world_size
        self._randomize = randomize
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    @staticmethod
    def from_torch(randomize: bool = False, seed: int = 0) -> ShardedFragmentSampler:
        """Use from a PyTorch distributed environment.

        Automatically infer `rank` and `world_size` from :mod:`torch.distributed`.
        """
        import torch

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return ShardedFragmentSampler(rank, world_size, randomize=randomize, seed=seed)

    def iter_fragments(
        self, dataset: lance.LanceDataset, **kwargs
    ) -> Generator[lance.LanceFragment, None, None]:
        fragments = dataset.get_fragments()
        if self._randomize:
            random.seed(self._seed)
            random.shuffle(fragments)
        for idx in range(self._rank, len(fragments), self._world_size):
            yield fragments[idx]


class ShardedBatchSampler(Sampler):
    """Sharded batch sampler.

    Each rank / process will process a subset of the batches.

    The input is subdivided into batches (of size `batch_size`).  Each rank / process
    takes every Nth batch (where N is the world size).  The order in which batches
    are loaded is randomized.

    When there is no filter then each process only needs to load the rows assigned to
    it but this process is still slightly less efficient than ShardedFragmentSampler
    since it requires loading rows by range instead of loading all rows for a
    given fragment.

    If there is a filter then we cannot divide the row ids ahead of time.  Instead,
    each process will load the entire filtered dataset and discard the rows that are
    not assigned to it.  The resulting stream is then randomized via a reservoir
    sampler.  This does not perfectly randomize the stream but it should generate
    a stream that is random enough for many use cases.

    """

    def __init__(
        self, rank: int, world_size: int, randomize: bool = False, seed: int = 0
    ):
        self._rank = rank
        self._world_size = world_size
        self._randomize = randomize
        self._seed = seed
        self._epoch = 0

    def __len__(self):
        return self._len

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    @staticmethod
    def from_torch(randomize: bool = False, seed: int = 0) -> ShardedBatchSampler:
        """Use it from a PyTorch distributed environment.

        Automatically infer `rank` and `world_size` from `torch.distributed`.
        """
        import torch

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return ShardedBatchSampler(rank, world_size, randomize=randomize, seed=seed)

    # Performs a filtered scan of the dataset and then throws away all but the Nth
    # rows (where N is the world size)
    def _shard_scan(
        self,
        dataset: lance.LanceDataset,
        batch_size: int,
        columns: Optional[Union[List[str], Dict[str, str]]],
        batch_readahead: int,
        filter: str,
    ) -> Generator[pa.RecordBatch, None, None]:
        accumulated_batches = []
        rows_accumulated = 0
        rows_to_skip = self._rank
        for batch in dataset.scanner(
            columns=columns,
            batch_readahead=batch_readahead,
            filter=filter,
            scan_in_order=True,
        ).to_batches():
            batch = batch.slice(rows_to_skip, batch.num_rows - rows_to_skip)
            # Take every Nth row
            indices = list(range(0, batch.num_rows, self._world_size))
            rows_to_skip = (
                self._world_size - (batch.num_rows % self._world_size)
            ) % self._world_size
            batch = batch.take(indices)

            # Add to our collection
            rows_accumulated += batch.num_rows
            accumulated_batches.append(batch)

            # If we have enough to generate 1 or more batches then do so
            if rows_accumulated > batch_size:
                big_batch = (
                    pa.Table.from_batches(accumulated_batches)
                    .combine_chunks()
                    .to_batches()[0]
                )
                accumulated_batches = []
                while big_batch.num_rows > batch_size:
                    next_batch = big_batch.slice(0, batch_size)
                    big_batch = big_batch.slice(batch_size)
                    yield next_batch
                rows_accumulated = big_batch.num_rows
                if big_batch.num_rows > 0:
                    accumulated_batches.append(big_batch)
        # deliver any batches left over, they will be <= batch
        # size but that is ok because we are done
        last_batch = (
            pa.Table.from_batches(accumulated_batches).combine_chunks().to_batches()[0]
        )
        yield last_batch

    def _sample_filtered(
        self,
        dataset: lance.LanceDataset,
        batch_size: int,
        columns: Optional[Union[List[str], Dict[str, str]]],
        batch_readahead: int,
        filter: str,
    ) -> Generator[pa.RecordBatch, None, None]:
        shard_scan = self._shard_scan(
            dataset, batch_size, columns, batch_readahead, filter
        )
        if not self._randomize:
            yield from shard_scan

        random.seed(self._seed + self._epoch)
        heap = []
        # We want to randomize the incoming sequence.  The normal approach
        # is to pull the whole thing in memory and run fisher-yates.  We
        # want to avoid buffering the entire input.  So, as an approximation,
        # we are using a heap + random number in a style similar to reservoir
        # sampling.
        #
        # We will keep up to k batches in the reservoir.  The higher
        # k the more randomness we will get from the reservoir shuffle
        # but the more memory we need.
        #
        # Picking 256 as a heuristic which should be 32Ki rows with
        # the default batch size
        k = 256
        for batch in shard_scan:
            priority = random.randint(0, k * 2 - 1)
            entry = PrioritizedItem(priority, batch)
            if len(heap) < k:
                heappush(heap, entry)
            else:
                next_batch = heappushpop(heap, entry)
                yield next_batch.item
        for batch in heap:
            yield batch.item

    def _sample_all(
        self,
        dataset: lance.LanceDataset,
        batch_size: int,
        columns: Optional[List[str]],
        batch_readahead: int,
    ) -> Generator[pa.RecordBatch, None, None]:
        total = dataset.count_rows()

        def _gen_ranges():
            for start in range(
                self._rank * batch_size,
                total,
                self._world_size * batch_size,
            ):
                yield start, min(start + batch_size, total)

        ranges = list(_gen_ranges())
        if self._randomize:
            random.seed(self._seed)
            random.shuffle(ranges)

        return dataset._ds.take_scan(
            ranges,
            columns=columns,
            batch_readahead=batch_readahead,
        )

    def __call__(
        self,
        dataset: lance.LanceDataset,
        *args,
        batch_size: int = 128,
        columns: Optional[List[str]] = None,
        filter: Optional[str] = None,
        batch_readahead: int = 16,
        with_row_id: Optional[bool] = None,
        **kwargs,
    ) -> Generator[pa.RecordBatch, None, None]:
        if filter is None:
            if with_row_id is not None:
                warnings.warn(
                    "with_row_id is not supported for ShardedBatchSampler",
                )
            return self._sample_all(dataset, batch_size, columns, batch_readahead)
        else:
            return self._sample_filtered(
                dataset, batch_size, columns, batch_readahead, filter
            )


class ShardedFixedBatchSampler(ShardedBatchSampler):
    """
    Sharded fixed batch sampler for distributed index-based batching.

    This sampler is designed for static datasets with a known total number of rows.
    It divides the dataset into consecutive index ranges (batches) and assigns each
    process (rank) a unique subset of these batches for efficient distributed loading.

    Features:
    - Requires `total_num_rows` and `batch_size` to be specified.
    - Each rank receives consecutive, non-overlapping index ranges.
    - Optionally randomizes the order of batches per epoch if `randomize=True`.
    - Suitable for integration with PyTorch DataLoader or similar frameworks.

    Example (total_num_rows=1000, world_size=4, batch_size=100):
    - Rank 0: [0-99], [100-199], [200-299]
    - Rank 1: [250-349], [350-449], [450-549]
    - Rank 2: [500-599], [600-699], [700-799]
    - Rank 3: [750-849], [850-949], [950-999]

    Parameters
    ----------
    rank : int
        The rank (process index) in the distributed cluster.
    world_size : int
        The total number of processes in the distributed cluster.
    randomize : bool, default False
        Whether to randomize the order of batches for each epoch.
    seed : int, default 0
        Random seed for reproducibility when randomize is enabled.
    batch_size : int, default 0
        The number of rows per batch.
    total_num_rows : int, default 0
        The total number of rows in the dataset.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        randomize: bool = False,
        seed: int = 0,
        batch_size: int = 0,
        total_num_rows: int = 0,
    ):
        super().__init__(rank, world_size, randomize, seed)
        self._total_num_rows = total_num_rows
        self._batch_size = batch_size
        self._len = self._compute_length()

    # The sampler here is mainly implemented with the hope that
    # the data of batch_size are all adjacent, so we don't want
    # to use filter to break this adjacent feature.
    def _compute_length(self):
        if self._batch_size == 0 and self._total_num_rows == 0:
            return 0
        per_rank = math.ceil(self._total_num_rows / self._world_size)
        return math.ceil(per_rank / self._batch_size)

    def __len__(self):
        return self._len

    def __iter__(self) -> Generator[List[int], None, None]:
        per_rank = math.ceil(self._total_num_rows / self._world_size)
        start = self._rank * per_rank
        end = min(start + per_rank, self._total_num_rows)

        batches = []
        current = start
        while current < end:
            batch_end = min(current + self._batch_size, end)
            batches.append(list(range(current, batch_end)))
            current = batch_end

        if self._randomize:
            random.seed(self._seed + self._epoch)
            random.shuffle(batches)

        yield from batches

    @staticmethod
    def from_torch(
        batch_size: int, total_num_rows: int, randomize: bool = False, seed: int = 0
    ) -> ShardedFixedBatchSampler:
        import torch

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        return ShardedFixedBatchSampler(
            rank, world_size, randomize, seed, batch_size, total_num_rows
        )
