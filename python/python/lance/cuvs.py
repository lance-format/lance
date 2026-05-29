# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from __future__ import annotations

import os
import tempfile
from importlib import import_module
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pathlib import Path


def is_cuvs_accelerator(accelerator: object) -> bool:
    return isinstance(accelerator, str) and accelerator.lower() == "cuvs"


def _dataset_storage_options(dataset) -> Optional[dict[str, str]]:
    latest_storage_options = getattr(dataset, "latest_storage_options", None)
    if callable(latest_storage_options):
        storage_options = latest_storage_options()
        if storage_options is not None:
            return storage_options
    return getattr(dataset, "_storage_options", None)


def _require_lance_cuvs():
    try:
        backend = import_module("lance_cuvs")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "accelerator='cuvs' requires the optional 'pylance-cuvs' loader "
            "package and a matching backend package such as "
            "'pylance-cuvs-cu12'."
        ) from exc
    missing = [
        attr
        for attr in ("train_ivf_pq", "build_ivf_pq_artifact")
        if not hasattr(backend, attr)
    ]
    if missing:
        raise ImportError(
            "The installed 'pylance-cuvs' package is incompatible with this "
            f"Lance build; missing backend APIs: {', '.join(missing)}"
        )
    return backend


def build_vector_index_on_cuvs(
    dataset,
    column: str,
    metric_type: str,
    accelerator: str,
    num_partitions: int,
    num_sub_vectors: int,
    dst_dataset_uri: str | Path | None = None,
    storage_options: Optional[dict[str, str]] = None,
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
    batch_size: int = 1024 * 128,
    filter_nan: bool = True,
):
    if not is_cuvs_accelerator(accelerator):
        raise ValueError("build_vector_index_on_cuvs requires accelerator='cuvs'")

    backend = _require_lance_cuvs()
    if storage_options is None:
        storage_options = _dataset_storage_options(dataset)
    artifact_uri = (
        os.fspath(dst_dataset_uri)
        if dst_dataset_uri is not None
        else tempfile.mkdtemp(prefix="lance-cuvs-artifact-")
    )
    training = backend.train_ivf_pq(
        dataset.uri,
        column,
        metric_type=metric_type,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
        num_bits=num_bits,
        filter_nan=filter_nan,
        storage_options=storage_options,
    )
    artifact = backend.build_ivf_pq_artifact(
        dataset.uri,
        column,
        training=training,
        artifact_uri=artifact_uri,
        batch_size=batch_size,
        filter_nan=filter_nan,
        storage_options=storage_options,
    )
    return (
        artifact.artifact_uri,
        artifact.files,
        training.ivf_centroids(),
        training.pq_codebook(),
    )


def prepare_global_ivf_pq_on_cuvs(
    dataset,
    column: str,
    num_partitions: int,
    num_sub_vectors: int,
    *,
    distance_type: str = "l2",
    accelerator: str = "cuvs",
    sample_rate: int = 256,
    max_iters: int = 50,
    num_bits: int = 8,
    filter_nan: bool = True,
    storage_options: Optional[dict[str, str]] = None,
):
    if not is_cuvs_accelerator(accelerator):
        raise ValueError("prepare_global_ivf_pq_on_cuvs requires accelerator='cuvs'")

    backend = _require_lance_cuvs()
    if storage_options is None:
        storage_options = _dataset_storage_options(dataset)
    training = backend.train_ivf_pq(
        dataset.uri,
        column,
        metric_type=distance_type,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        sample_rate=sample_rate,
        max_iters=max_iters,
        num_bits=num_bits,
        filter_nan=filter_nan,
        storage_options=storage_options,
    )
    return {
        "ivf_centroids": training.ivf_centroids(),
        "pq_codebook": training.pq_codebook(),
    }
