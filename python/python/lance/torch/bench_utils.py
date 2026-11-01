# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Benchmark Utilities built on PyTorch"""

from typing import Optional, Tuple, Union

import pyarrow as pa

from lance.dependencies import _check_for_numpy, torch
from lance.dependencies import numpy as np

from .. import LanceDataset
from . import preferred_device
from .data import LanceDataset as PytorchLanceDataset
from .distance import pairwise_cosine, pairwise_l2

__all__ = ["ground_truth"]


def sort_tensors(
    source: torch.Tensor, other: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort multiple tensors based on the order of the source.

    Sort both tensor based on the `source` tensor (a M x N tensor),
    and keep k values for each row.

    Parameters
    ----------
    source: torch.Tensor
        The source tensor to sort by.
    other: torch.Tensor
        The other tensor to sort, the order of the other tensor will be
        determined by the `source` tensor.
    k: int
        The number of values to keep.

    Returns
    -------
    The sorted source tensor and the sorted other tensor.

    """
    sorted_values, indices = torch.sort(source, dim=1)
    indices = indices[:, :k]
    sorted_others = torch.gather(other, 1, indices)
    return sorted_values[:, :k], sorted_others


def ground_truth(
    ds: LanceDataset,
    column: str,
    query: Union[torch.Tensor, np.ndarray],
    metric_type: str = "L2",
    k: int = 100,
    batch_size: int = 10240,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Find ground truth from dataset.

    Parameters
    ----------
    ds: LanceDataset
        The dataset to test.
    column: str
        The name of the vector column.
    query: 2-D vectors
        A 2-D query vectors, with the shape of [N, dimension].
    k: int
        The number of the nearest vectors to collect for each query vector.
    metric_type: str
        Metric type. How to compute distance, accepts L2 or cosine.
    batch_size: int
        Batch size to read from the input dataset.

    Returns
    -------
    a 2-D array of row_ids for the nearest vectors from each query vector.
    """
    device = preferred_device(device=device)

    if _check_for_numpy(query) and isinstance(query, np.ndarray):
        query = torch.from_numpy(query)
    query = query.to(device)
    metric_type = metric_type.lower()

    all_ids = None
    all_dists = None

    tds = PytorchLanceDataset(
        ds, batch_size=batch_size, columns=[column], with_row_id=True
    )
    from tqdm import tqdm

    total = ds.count_rows()
    with tqdm(total=total) as progress_bar:
        for batch in tds:
            vectors = batch[column].to(device)
            row_ids = batch["_rowid"].to(device).broadcast_to((query.shape[0], -1))
            if metric_type == "l2":
                dists = pairwise_l2(query, vectors)
            elif metric_type == "cosine":
                dists = pairwise_cosine(query, vectors)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")

            dists, row_ids = sort_tensors(dists, row_ids, k)

            if all_ids is None:
                all_ids = row_ids
            else:
                all_ids = torch.cat([all_ids, row_ids], 1)

            if all_dists is None:
                all_dists = dists
            else:
                all_dists = torch.cat([all_dists, dists], 1)
            all_dists, all_ids = sort_tensors(all_dists, all_ids, k)
            progress_bar.update(vectors.shape[0])

    return all_ids


def recall(expected: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Recalls

    Parameters
    ----------
    expected: ndarray
        The ground truth
    actual: ndarray
        The ANN results
    """
    assert expected.shape == actual.shape
    recalls = np.array(
        [np.isin(exp, act).sum() / exp.shape[0] for exp, act in zip(expected, actual)]
    )
    return recalls


def infer_vector_column(ds: LanceDataset) -> Optional[str]:
    """Infer vector column from dataset."""
    for field in ds.schema:
        if pa.types.is_fixed_size_list(field.type):
            return field.name
        elif (
            isinstance(field.type, pa.FixedShapeTensorType)
            and len(field.type.shape) == 1
        ):
            return field.name

    return None
