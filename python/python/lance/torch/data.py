# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Read Lance dataset as torch DataPipe."""

# PEP-585. Can be removed after deprecating python 3.8 support.
from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
)

import pyarrow as pa

import lance
from lance._dataset.cache import CachedDataset
from lance.dependencies import _check_for_numpy, torch
from lance.dependencies import numpy as np

from ..sampler import (
    FullScanSampler,
    Sampler,
    ShardedBatchSampler,
    ShardedFragmentSampler,
    maybe_sample,
)
from .dist import get_global_rank, get_global_world_size

__all__ = ["LanceDataset", "SafeLanceDataset", "get_safe_loader"]


class ToTensorFn(Protocol):
    def __call__(
        self,
        batch: Union[pa.RecordBatch, Dict[str, Any]],
        *,
        hf_converter: Optional[dict] = None,
        use_blob_api: bool = False,
        **kwargs: Any,
    ) -> Union[dict[str, torch.Tensor], torch.Tensor]: ...


def _is_bfloat16_type(t: pa.DataType) -> bool:
    """Check if a PyArrow type is the lance bfloat16 extension type."""
    return isinstance(t, pa.ExtensionType) and t.extension_name == "lance.bfloat16"


def _bf16_to_tensor(arr: pa.Array) -> torch.Tensor:
    """Convert a bfloat16 extension array to a torch.bfloat16 tensor.

    Reinterprets the raw bytes as uint16 and views as bfloat16,
    since they share the same 2-byte memory layout.
    Null values are replaced with NaN.
    """
    storage = arr.storage if isinstance(arr.type, pa.ExtensionType) else arr
    buf = storage.buffers()[1]
    offset = storage.offset * 2  # 2 bytes per bf16 value
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            tensor = torch.frombuffer(
                memoryview(buf),
                dtype=torch.uint16,
                count=len(storage),
                offset=offset,
            ).view(torch.bfloat16)
    except (AttributeError, RuntimeError, TypeError):
        np_uint16 = np.frombuffer(
            buf, dtype=np.uint16, count=len(storage), offset=offset
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable",
                category=UserWarning,
            )
            tensor = torch.from_numpy(np_uint16).view(torch.bfloat16)
    if arr.null_count > 0:
        tensor = tensor.clone()
        null_mask = torch.from_numpy(arr.is_null().to_numpy(zero_copy_only=False))
        tensor[null_mask] = float("nan")
    return tensor


# Convert an Arrow FSL array into a 2D torch tensor
def _fsl_to_tensor(arr: pa.FixedSizeListArray, dimension: int) -> torch.Tensor:
    # Note: FixedSizeListArray.values does not take offset/len into account and
    # so may we need to slice here
    values = arr.values
    start = arr.offset * dimension
    num_vals = len(arr) * dimension
    values = values.slice(start, num_vals)
    # Convert to numpy
    nparr = values.to_numpy(zero_copy_only=False).reshape(-1, dimension)
    return torch.from_numpy(nparr)


def _to_tensor(
    batch: Union[pa.RecordBatch, Dict[str, pa.Array]],
    *,
    uint64_as_int64: bool = True,
    hf_converter: Optional[dict] = None,
    use_blob_api: bool = False,
    **kwargs,
) -> Union[dict[str, torch.Tensor], torch.Tensor]:
    """Convert a pyarrow RecordBatch to torch Tensor."""
    ret = {}

    cols = (
        batch.column_names if isinstance(batch, pa.RecordBatch) else list(batch.keys())
    )
    for col in cols:
        arr: pa.Array = batch[col]

        if (
            use_blob_api
            and isinstance(arr, list)
            and arr
            and isinstance(arr[0], lance.BlobFile)
        ):
            raise NotImplementedError(
                'Need user-provided "to_tensor_fn" for Blob files'
            )

        tensor: torch.Tensor = None
        if (isinstance(arr.type, pa.FixedShapeTensorType)) and (
            pa.types.is_floating(arr.type.value_type)
            or pa.types.is_integer(arr.type.value_type)
        ):
            arr = arr.storage

        if (pa.types.is_fixed_size_list(arr.type)) and (
            pa.types.is_floating(arr.type.value_type)
            or pa.types.is_integer(arr.type.value_type)
        ):
            tensor = _fsl_to_tensor(arr, arr.type.list_size)
        elif pa.types.is_fixed_size_list(arr.type) and _is_bfloat16_type(
            arr.type.value_type
        ):
            values = arr.values
            start = arr.offset * arr.type.list_size
            num_vals = len(arr) * arr.type.list_size
            values = values.slice(start, num_vals)
            tensor = _bf16_to_tensor(values).view(-1, arr.type.list_size)
        elif (
            pa.types.is_integer(arr.type)
            or pa.types.is_floating(arr.type)
            or pa.types.is_boolean(arr.type)
        ):
            tensor = torch.from_numpy(arr.to_numpy(zero_copy_only=False))

            if uint64_as_int64 and tensor.dtype == torch.uint64:
                tensor = tensor.to(torch.int64)
        elif _is_bfloat16_type(arr.type):
            tensor = _bf16_to_tensor(arr)
        elif hf_converter is not None:
            tensor = hf_converter.to_pytorch(col, arr)

        if tensor is None:
            raise ValueError(
                "Only support FixedSizeList<f16/bf16/f32/f64> or "
                + f"numeric/bfloat16 values, got: {arr.type}"
            )

        del arr
        ret[col] = tensor
    if len(ret) == 1:
        t = next(iter(ret.values()))
        del ret
        return t
    return ret


class TensorDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset that wraps over a tensor, returns in batches.

    Unlike `torch.utils.data.TensorDataset`, this has the same behavior as LanceDataset
    that it yields tensor in batches.
    """

    def __init__(
        self, data: Union[torch.Tensor, np.ndarray], batch_size: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if _check_for_numpy(data) and isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        self._data: torch.Tensor = data
        self._batch_size = batch_size

    def __repr__(self):
        return "LanceTensorDataset"

    def __len__(self) -> int:
        return math.ceil(self._data.shape[0] / self._batch_size)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx >= len(self):
            raise StopIteration
        start = idx * self._batch_size
        end = min((idx + 1) * self._batch_size, self._data.shape[0])
        return self._data[start:end, :]


def concat_batches(bs):
    return pa.RecordBatch.from_arrays(
        [
            pa.concat_arrays([b.columns[i] for b in bs])
            for i in range(bs[0].num_columns)
        ],
        schema=bs[0].schema,
    )


def _buffer_arrow_batches(
    it: Iterable[pa.RecordBatch],
    buffer_size: int = 10240,
) -> Iterable[pa.RecordBatch]:
    buffer = []
    cur_size = 0
    for item in it:
        if cur_size > 0 and cur_size + item.num_rows > buffer_size:
            if len(buffer) == 1:
                # Most of the time, we are in the happy situation where we have a single
                # batch to yield.
                yield buffer[0]
            else:
                yield concat_batches(buffer)
            buffer = []
            cur_size = 0

        buffer.append(item)
        cur_size += item.num_rows
    if buffer:
        yield concat_batches(buffer)


class LanceDataset(torch.utils.data.IterableDataset):
    """PyTorch :class:`torch.utils.data.IterableDataset` over lance dataset."""

    def __init__(
        self,
        dataset: Union[torch.utils.data.Dataset, str, Path],
        batch_size: int,
        *args,
        dataset_options: Optional[Dict[str, Any]] = None,
        columns: Optional[Union[List[str], Dict[str, str]]] = None,
        filter: Optional[str] = None,
        samples: Optional[int] = 0,
        cache: Optional[Union[str, bool]] = None,
        with_row_id: bool = False,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shard_granularity: Optional[Literal["fragment", "batch"]] = None,
        batch_readahead: int = 16,
        to_tensor_fn: Optional[ToTensorFn] = _to_tensor,
        sampler: Optional[Sampler] = None,
        auto_detect_rank: bool = True,
        **kwargs,
    ):
        """Use PyTorch Dataset API to read Lance dataset.

        Parameters
        ----------
        dataset : Union[torch.utils.data.Dataset, str, Path]
            Lance dataset to read. Can be URI, path, or an initialized Lance Dataset.
        batch_size : int
            Batch size to yield for each iteration.
        columns : list of str, optional
            The names of the column to read, by default None, which means reading all
            columns.
        filter : str, optional
            If set, only rows that match the filter will be read.  Currently, this
            can only be used when doing a full scan (`sampler` is None and
            shard_granularity is None or "fragment" and `samples` is None)
        cache : str or bool, optional
            If set true, the dataset will be cached on disk from the first iteration.
            The following iterations will read from the cache.
        with_row_id : bool, optional
            If set true, the returned batch will have an additional column named
            `_rowid` that contains the row id of the batch.
        rank: int, optional (deprecated)
            If set, the rank (idx) of this process in distributed training / inference.
        world_size: int, optional (deprecated)
            If set, the total number of processes in distributed training / inference.
        shard_granularity: str, optional
            The basic unit of sharding data. If set to "fragment", each worker will get
            the a subset of fragments.
            If set to "batch", it will read the "batch" interleave with the
            same fragments.
        batch_readahead: int, optional
            The number of batches to read ahead in different (Rust) threads for each
            fragment.
        sampler: callable, optional
            A function that samples the dataset.
        to_tensor_fn : callable, optional
            A function that converts a pyarrow RecordBatch to torch.Tensor.
            Should accept a batch (RecordBatch or Dict[str, pa.Array]) as the first
            argument, plus optional keyword arguments ``hf_converter`` and
            ``use_blob_api``.
        auto_detect_rank: bool = True, optional
            If set true, the rank and world_size will be detected automatically.
        """
        super().__init__()
        if isinstance(dataset, (str, Path)):
            dataset_options = dataset_options or {}
            dataset = lance.dataset(dataset, **dataset_options)
        self.dataset = dataset
        self.columns = columns
        self.batch_size = batch_size
        self.samples: Optional[int] = samples
        self.filter = filter
        self.with_row_id = with_row_id
        self.batch_readahead = batch_readahead
        self._to_tensor_fn = to_tensor_fn
        self._hf_converter = None

        self._blob_columns = self._blob_columns()
        if self._blob_columns:
            self.with_row_id = True

        # As Shared Dataset
        self.shard_granularity = shard_granularity
        self.rank = rank
        self.world_size = world_size
        if rank is not None and world_size is not None:
            warnings.warn("rank and world_size are deprecated", DeprecationWarning)
        self.sampler: Optional[Sampler] = sampler

        # Dataset with huggingface metadata
        if (
            dataset.schema.metadata is not None
            and (hf_meta := dataset.schema.metadata.get(b"huggingface")) is not None
        ):
            from ..hf import HuggingFaceConverter

            hf_ds_info = json.loads(hf_meta)
            self._hf_converter = HuggingFaceConverter(hf_ds_info)

        self.cache = cache
        self.cached_ds: Optional[CachedDataset] = None
        self._auto_detect_rank = auto_detect_rank

    def __repr__(self) -> str:
        return f"LanceTorchDataset({self.dataset.uri}, size={self.samples})"

    @property
    def schema(self) -> pa.Schema:
        if not self.columns:
            return self.dataset.schema
        fields = [self.dataset.schema.field(col) for col in self.columns]
        return pa.schema(fields, metadata=self.dataset.schema.metadata)

    def __iter__(self):
        if self.sampler is None:
            if self.rank is not None:
                rank = self.rank
            elif self._auto_detect_rank:
                rank = get_global_rank()
            else:
                rank = None

            if self.world_size is not None:
                world_size = self.world_size
            elif self._auto_detect_rank:
                world_size = get_global_world_size()
            else:
                world_size = None
            if self.shard_granularity is None:
                if rank is not None and world_size is not None:
                    sampler = ShardedFragmentSampler(rank=rank, world_size=world_size)
                else:
                    sampler = FullScanSampler()
            elif self.shard_granularity == "batch":
                sampler = ShardedBatchSampler(rank, world_size)
            elif self.shard_granularity == "fragment":
                sampler = ShardedFragmentSampler(rank, world_size)
            else:
                raise ValueError(f"Invalid shard_granularity: {self.shard_granularity}")
        else:
            sampler = self.sampler

        projected_columns = self.columns or self.dataset.schema.names
        if self._blob_columns:
            projected_columns = [
                c for c in projected_columns if c not in self._blob_columns
            ]

        stream: Iterable[pa.RecordBatch]
        if self.cached_ds:
            stream = self.cached_ds
        else:
            if self.samples:
                raw_stream = maybe_sample(
                    self.dataset,
                    n=self.samples,
                    columns=projected_columns,
                    batch_size=self.batch_size,
                    filt=self.filter,
                )
            else:
                raw_stream = sampler(
                    self.dataset,
                    columns=projected_columns,
                    filter=self.filter,
                    batch_size=self.batch_size,
                    with_row_id=self.with_row_id,
                    batch_readahead=self.batch_readahead,
                )

            stream = _buffer_arrow_batches(raw_stream, buffer_size=self.batch_size)

            if self.cache:
                self.cached_ds = CachedDataset(stream, cache=self.cache)
                stream = self.cached_ds

        use_blob_api = bool(self._blob_columns)
        for batch in stream:
            if use_blob_api:
                dict_batch = {}
                assert "_rowid" in batch.column_names
                row_ids = batch["_rowid"]
                for col in batch.column_names:
                    dict_batch[col] = batch[col]
                for col in self._blob_columns:
                    dict_batch[col] = self.dataset.take_blobs(
                        ids=row_ids.to_pylist(), blob_column=col
                    )
                batch = dict_batch
            if self._to_tensor_fn is not None:
                batch = self._to_tensor_fn(
                    batch, hf_converter=self._hf_converter, use_blob_api=use_blob_api
                )
            yield batch
            del batch

    def _blob_columns(self) -> List[str]:
        """Returns True if one of the projected column is Large Blob encoded."""
        cols = self.columns
        if not cols:
            cols = self.dataset.schema.names
        blob_cols = []
        for col in cols:
            field = self.dataset.schema.field(col)
            if (
                field.type == pa.large_binary()
                and field.metadata is not None
                and field.metadata.get(b"lance-encoding:blob") == b"true"
            ):
                logging.debug("Column %s is a Large Blob column", col)
                blob_cols.append(col)
        return blob_cols


class SafeLanceDataset(torch.utils.data.Dataset):
    def __init__(self, uri, *, dataset_options=None, **kwargs):
        super().__init__(**kwargs)
        self.uri = uri
        self.dataset_options = dataset_options or {}
        self._len = self._safe_preload()
        self._ds = None

    def _safe_preload(self):
        """Main-process safe metadata loading"""
        ds = lance.dataset(self.uri, **self.dataset_options)
        length = ds.count_rows()
        del ds
        return length

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        """Batch data fetching with worker-safe initialization

        Args:
            indices: List[int] - batch indices to retrieve

        Returns:
            List[dict] - samples in original data format
        """
        if self._ds is None:
            # Worker-process initialization
            self._ds = lance.dataset(self.uri, **self.dataset_options)

        # Leverage native batch reading
        batch = self._ds.take(indices)

        # Convert to python-native format
        return batch.to_pylist()


def get_safe_loader(dataset, batch_size=32, num_workers=4, **kwargs):
    """Create a DataLoader with safe multiprocessing defaults

    Args:
        dataset: Input dataset object
        batch_size: Number of samples per batch (default=32)
        num_workers: Number of parallel data workers (default=4)
        **kwargs: Additional DataLoader arguments. Note:
                 - Forces 'spawn' context for Windows compatibility
                 - Sets persistent_workers=True by default
                 - Both defaults apply only when num_workers > 0
                 - User-provided args override defaults

    Returns:
        Configured DataLoader instance with process-safe settings
    """

    loader_args = {"batch_size": batch_size, "num_workers": num_workers}
    # DataLoader rejects persistent_workers=True and multiprocessing_context
    # when num_workers == 0, so only set them for multi-process loading.
    if num_workers > 0:
        loader_args["persistent_workers"] = True
        # Force spawn context for Windows/multiprocessing compatibility
        loader_args["multiprocessing_context"] = torch.multiprocessing.get_context(
            "spawn"
        )
    # User-provided arguments take priority
    loader_args.update(kwargs)

    return torch.utils.data.DataLoader(dataset, **loader_args)
