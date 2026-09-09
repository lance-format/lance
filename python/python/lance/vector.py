# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Embedding vector utilities"""

from __future__ import annotations

import re
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

import pyarrow as pa
from tqdm.auto import tqdm

from . import write_dataset
from .dependencies import (
    _check_for_numpy,
    torch,
)
from .dependencies import numpy as np
from .log import LOGGER
from .util import MetricType, _normalize_index_segment_ids, _normalize_metric_type

if TYPE_CHECKING:
    import uuid
    from pathlib import Path

    from . import LanceDataset


def _normalize_vectors(vectors, ndim):
    if ndim is None:
        ndim = len(next(iter(vectors)))
    values = np.array(vectors, dtype="float32").ravel()
    return pa.FixedSizeListArray.from_arrays(values, list_size=ndim)


def _validate_ndim(values, ndim):
    for v in values:
        if ndim is None:
            ndim = len(v)
        else:
            if ndim != len(v):
                raise ValueError(f"Expected {ndim} dimensions but got {len(v)} for {v}")
    return ndim


def vec_to_table(
    data: Union[dict, list, np.ndarray],
    names: Optional[Union[str, list]] = None,
    ndim: Optional[int] = None,
    check_ndim: bool = True,
) -> pa.Table:
    """
    Create a pyarrow Table containing vectors.
    Vectors are created as FixedSizeListArray's in pyarrow with Float32 values.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> from lance.vector import vec_to_table
    >>> dd = {"vector0": np.random.randn(10), "vector1": np.random.randn(10)}
    >>> vec_to_table(dd)
    pyarrow.Table
    id: string
    vector: fixed_size_list<item: float>[10]
      child 0, item: float
    ----
    id: [["vector0","vector1"]]
    vector: [[[1.7640524,0.4001572,0.978738,2.2408931,1.867558,-0.9772779,0.95008844,\
-0.1513572,-0.10321885,0.41059852],[0.14404356,1.4542735,0.7610377,\
0.121675014,0.44386324,0.33367434,1.4940791,-0.20515826,0.3130677,-0.85409576]]]
    >>> vec_to_table(dd).to_pandas()
            id                                             vector
    0  vector0  [1.7640524, 0.4001572, 0.978738, 2.2408931, 1....
    1  vector1  [0.14404356, 1.4542735, 0.7610377, 0.121675014...

    Parameters
    ----------
    data: dict, list, or np.ndarray
        If dict, the keys are added as "id" column
        If list, then each element is assumed to be a vector
        If ndarray, then each row is assumed to be a vector
    names: str or list, optional
        If data is dict, then names should be a list of 2 str; default ["id", "vector"]
        If data is list or ndarray, then names should be str; default "vector"
    ndim: int, optional
        Number of dimensions of the vectors. Inferred if omitted.
    check_ndim: bool, default True
        Whether to verify that all vectors have the same length

    Returns
    -------
    tbl: pa.Table
        A pyarrow Table with vectors converted to appropriate types
    """
    if isinstance(data, dict):
        if names is None:
            names = ["id", "vector"]
        elif not isinstance(names, (list, tuple)) and len(names) == 2:
            raise ValueError(
                "If data is a dict, names must be a list or tuple of 2 strings"
            )
        values = list(data.values())
        if check_ndim:
            ndim = _validate_ndim(values, ndim)
        vectors = _normalize_vectors(values, ndim)
        ids = pa.array(data.keys())
        arrays = [ids, vectors]
    elif isinstance(data, list) or (
        _check_for_numpy(data) and isinstance(data, np.ndarray)
    ):
        if names is None:
            names = ["vector"]
        elif isinstance(names, str):
            names = [names]
        elif not isinstance(names, (list, tuple)) and len(names) == 1:
            raise ValueError(f"names cannot be more than 1 got {len(names)}")
        if check_ndim:
            ndim = _validate_ndim(data, ndim)
        vectors = _normalize_vectors(data, ndim)
        arrays = [vectors]
    else:
        raise NotImplementedError(
            f"data must be dict, list, or ndarray (require numpy installed), \
            got {type(data)} instead"
        )
    return pa.Table.from_arrays(arrays, names=names)


CUDA_REGEX = re.compile(r"^cuda(:\d+)?$")


def train_pq_codebook_on_accelerator(
    dataset: LanceDataset | Path | str,
    metric_type: MetricType,
    accelerator: Union[str, "torch.Device"],
    num_sub_vectors: int,
    batch_size: int = 1024 * 10 * 4,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, List[Any]]:
    """Use accelerator (GPU or MPS) to train pq codebook."""

    from .torch.data import LanceDataset as TorchDataset
    from .torch.kmeans import KMeans

    metric_type = _normalize_metric_type(metric_type)

    centroids_list = []
    kmeans_list = []

    field_names = [f"__residual_subvec_{i + 1}" for i in range(num_sub_vectors)]

    sample_size = 256 * 256

    ds_init = TorchDataset(
        dataset,
        batch_size=256,
        columns=field_names,
        samples=256,
    )

    init_centroids = next(iter(ds_init))

    ds_fit = TorchDataset(
        dataset,
        batch_size=20480,
        columns=field_names,
        samples=sample_size,
        cache=True,
    )

    for sub_vector in range(num_sub_vectors):
        LOGGER.info("Training IVF partitions using GPU(%s)", accelerator)
        if num_sub_vectors == 1:
            # sampler has different behaviour with one column
            init_centroids_slice = init_centroids
        else:
            init_centroids_slice = init_centroids[field_names[sub_vector]]
        kmeans_local = KMeans(
            256,
            max_iters=50,
            metric=metric_type,
            device=accelerator,
            centroids=init_centroids_slice,
        )
        if num_sub_vectors == 1:
            kmeans_local.fit(ds_fit)
        else:
            kmeans_local.fit(ds_fit, column=field_names[sub_vector])

        ivf_centroids_local = kmeans_local.centroids.cpu().numpy()
        centroids_list.append(ivf_centroids_local)
        kmeans_list.append(kmeans_local)

    pq_codebook = np.stack(centroids_list).astype(dtype)
    return pq_codebook, kmeans_list


def _sample_init_centroids(
    ds: Iterable["torch.Tensor"], k: int, filter_nan: bool
) -> "torch.Tensor":
    """Take up to k vectors from ds to seed kmeans, skipping non-finite ones."""
    # `column is not null` does not exclude NaN vectors, so they can still be
    # sampled here.  Training drops them (distance returns id -1), but a NaN
    # centroid never recovers and leaves every partition NaN.
    sampled = []
    num_sampled = 0
    for batch in ds:
        if filter_nan:
            batch = batch[batch.isfinite().flatten(1).all(dim=1)]
        if batch.shape[0] == 0:
            continue
        sampled.append(batch)
        num_sampled += batch.shape[0]
        if num_sampled >= k:
            break

    if num_sampled == 0:
        raise ValueError(
            "Cannot initialize centroids: the sampled vectors are all null or "
            "non-finite"
        )
    return torch.cat(sampled)[:k]


def train_ivf_centroids_on_accelerator(
    dataset: LanceDataset,
    column: str,
    k: int,
    metric_type: MetricType,
    accelerator: Union[str, "torch.Device"],
    batch_size: int = 1024 * 10 * 4,
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
    filter_nan: bool = True,
) -> Tuple[np.ndarray, Any]:
    """Use accelerator (GPU or MPS) to train kmeans."""

    from .torch.data import LanceDataset as TorchDataset
    from .torch.kmeans import KMeans

    metric_type = _normalize_metric_type(metric_type)
    vector_value_type = dataset.schema.field(column).type.value_type

    if isinstance(accelerator, str) and (
        not (CUDA_REGEX.match(accelerator) or accelerator == "mps")
    ):
        raise ValueError(
            "Train ivf centroids on accelerator: "
            + f"only support 'cuda' or 'mps' as accelerator, got '{accelerator}'."
        )

    sample_size = k * sample_rate

    k = int(k)

    if dataset.schema.field(column).nullable and filter_nan:
        filt = f"{column} is not null"
    else:
        filt = None

    LOGGER.info("Randomly select %s centroids from %s (filt=%s)", k, dataset, filt)

    ds = TorchDataset(
        dataset,
        batch_size=k,
        columns=[column],
        samples=sample_size,
        filter=filt,
    )

    init_centroids = _sample_init_centroids(ds, k, filter_nan)
    LOGGER.info("Done sampling: centroids shape: %s", init_centroids.shape)

    ds = TorchDataset(
        dataset,
        batch_size=20480,
        columns=[column],
        samples=sample_size,
        filter=filt,
        cache=True,
    )

    LOGGER.info("Training IVF partitions using GPU(%s)", accelerator)
    kmeans = KMeans(
        k,
        max_iters=max_iters,
        metric=metric_type,
        device=accelerator,
        centroids=init_centroids,
    )
    kmeans.fit(ds)

    centroids = (
        kmeans.centroids.cpu().numpy().astype(vector_value_type.to_pandas_dtype())
    )

    return centroids, kmeans


def compute_pq_codes(
    dataset: LanceDataset,
    kmeans_list: List[Any],  # KMeans
    batch_size: int = 1024 * 10 * 4,
    dst_dataset_uri: Optional[Union[str, Path]] = None,
    allow_cuda_tf32: bool = True,
) -> Tuple[Union[str, Path], List[str]]:
    """Compute pq codes for each row using GPU kmeans and spill to disk.

    Parameters
    ----------
    dataset: LanceDataset
        Dataset to compute pq codes for.
    kmeans_list: List[lance.torch.kmeans.KMeans]
        KMeans models to use to compute pq (one per subspace)
    batch_size: int, default 40960
        The batch size used to read the dataset.
    dst_dataset_uri: Union[str, Path], optional
        The path to store the partitions.  If not specified a random
        directory is used instead
    allow_cuda_tf32: bool, default True
        Whether to allow tf32 for matmul on CUDA.

    Returns
    -------
    Tuple[Union[str, Path], List[str]]
        The absolute path of the pq codes dataset and shuffle buffers
    """
    from .torch.data import LanceDataset as TorchDataset

    torch.backends.cuda.matmul.allow_tf32 = allow_cuda_tf32

    num_rows = dataset.count_rows()

    num_sub_vectors = len(kmeans_list)

    field_names = [f"__residual_subvec_{i + 1}" for i in range(num_sub_vectors)]

    torch_ds = TorchDataset(
        dataset,
        batch_size=batch_size,
        with_row_id=False,
        columns=["row_id", "partition"] + field_names,
    )
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=1,
        pin_memory=True,
        collate_fn=_collate_fn,
    )
    output_schema = pa.schema(
        [
            pa.field("row_id", pa.uint64()),
            pa.field("__ivf_part_id", pa.uint32()),
            pa.field("__pq_code", pa.list_(pa.uint8(), list_size=num_sub_vectors)),
        ]
    )

    progress = tqdm(total=num_rows)
    progress.set_description("Assigning PQ codes")

    device = kmeans_list[0].device

    def _pq_codes_assignment() -> Iterable[pa.RecordBatch]:
        with torch.no_grad():
            for batch in loader:
                vecs_lists = [
                    batch[field_names[i]]
                    .to(device)
                    .reshape(-1, kmeans_list[i].centroids.shape[1])
                    for i in range(num_sub_vectors)
                ]

                pq_codes = torch.stack(
                    [
                        kmeans_list[i].transform(vecs_lists[i])
                        for i in range(num_sub_vectors)
                    ],
                    dim=1,
                )
                pq_codes = pq_codes.to(torch.uint8)

                ids = batch["row_id"].reshape(-1)
                partitions = batch["partition"].reshape(-1)

                ids = ids.cpu()
                partitions = partitions.cpu()
                pq_codes = pq_codes.cpu()

                pq_values = pa.array(pq_codes.numpy().reshape(-1))
                pq_codes = pa.FixedSizeListArray.from_arrays(pq_values, num_sub_vectors)
                part_batch = pa.RecordBatch.from_arrays(
                    [ids, partitions, pq_codes],
                    schema=output_schema,
                )

                progress.update(part_batch.num_rows)
                yield part_batch

    rbr = pa.RecordBatchReader.from_batches(output_schema, _pq_codes_assignment())
    if dst_dataset_uri is None:
        dst_dataset_uri = tempfile.mkdtemp()
    ds = write_dataset(
        rbr,
        dst_dataset_uri,
        schema=output_schema,
        data_storage_version="legacy",
    )

    progress.close()

    LOGGER.info("Saved precomputed pq_codes to %s", dst_dataset_uri)

    shuffle_buffers = [
        data_file.path for frag in ds.get_fragments() for data_file in frag.data_files()
    ]
    return dst_dataset_uri, shuffle_buffers


def _collate_fn(batch):
    return batch[0]


def compute_partitions(
    dataset: LanceDataset,
    column: str,
    kmeans: Any,  # KMeans
    batch_size: int = 1024 * 10 * 4,
    dst_dataset_uri: Optional[Union[str, Path]] = None,
    allow_cuda_tf32: bool = True,
    num_sub_vectors: Optional[int] = None,
    filter_nan: bool = True,
    sample_size: Optional[int] = None,
) -> str:
    """Compute partitions for each row using GPU kmeans and spill to disk.

    Parameters
    ----------
    dataset: LanceDataset
        Dataset to compute partitions for.
    column: str
        Column name of the vector column.
    kmeans: lance.torch.kmeans.KMeans
        KMeans model to use to compute partitions.
    batch_size: int, default 40960
        The batch size used to read the dataset.
    dst_dataset_uri: Union[str, Path], optional
        The path to store the partitions.  If not specified a random
        directory is used instead
    allow_cuda_tf32: bool, default True
        Whether to allow tf32 for matmul on CUDA.

    Returns
    -------
    str
        The absolute path of the partition dataset.
    """
    from .torch.data import LanceDataset as TorchDataset

    torch.backends.cuda.matmul.allow_tf32 = allow_cuda_tf32

    num_rows = dataset.count_rows()

    if dataset.schema.field(column).nullable and filter_nan:
        filt = f"{column} is not null"
    else:
        filt = None

    torch_ds = TorchDataset(
        dataset,
        batch_size=batch_size,
        with_row_id=True,
        columns=[column],
        samples=sample_size,
        filter=filt,
    )
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=1,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    dim = kmeans.centroids.shape[1]

    fields = []
    if num_sub_vectors is not None:
        field_names = [f"__residual_subvec_{i + 1}" for i in range(num_sub_vectors)]
        subvector_size = dim // num_sub_vectors
        fields = [
            pa.field(name, pa.list_(pa.float32(), list_size=subvector_size))
            for name in field_names
        ]

    output_schema = pa.schema(
        [
            pa.field("row_id", pa.uint64()),
            pa.field("partition", pa.uint32()),
        ]
        + fields
    )

    progress = tqdm(total=num_rows)

    if num_sub_vectors is not None:
        progress.set_description("Assigning partitions and computing residuals")
    else:
        progress.set_description("Assigning partitions")

    def _partition_assignment() -> Iterable[pa.RecordBatch]:
        id_offset = 0
        with torch.no_grad():
            for batch in loader:
                if sample_size is None:
                    vecs = batch[column]
                    ids = batch["_rowid"].reshape(-1)
                else:
                    # No row ids with sampling
                    vecs = batch
                    ids = torch.arange(id_offset, id_offset + vecs.size(0))
                    id_offset += vecs.size(0)

                vecs = vecs.to(kmeans.device).reshape(-1, kmeans.centroids.shape[1])

                partitions = kmeans.transform(vecs)

                # this is expected to be true, so just assert
                assert vecs.shape[0] == ids.shape[0]

                # Ignore any invalid vectors.
                mask_gpu = partitions.isfinite() & (partitions >= 0)
                mask = mask_gpu.cpu()
                ids = ids[mask]
                partitions = partitions[mask_gpu]

                partitions = partitions.cpu()

                split_columns = []
                if num_sub_vectors is not None:
                    residual_vecs = vecs[mask_gpu] - kmeans.centroids[partitions]
                    for i in range(num_sub_vectors):
                        subvector_tensor = residual_vecs[
                            :, i * subvector_size : (i + 1) * subvector_size
                        ]
                        subvector_arr = pa.array(
                            subvector_tensor.cpu().detach().numpy().reshape(-1)
                        )
                        subvector_fsl = pa.FixedSizeListArray.from_arrays(
                            subvector_arr, subvector_size
                        )
                        split_columns.append(subvector_fsl)

                part_batch = pa.RecordBatch.from_arrays(
                    [
                        ids.numpy(),
                        partitions.numpy(),
                    ]
                    + split_columns,
                    schema=output_schema,
                )
                if len(part_batch) < len(ids):
                    LOGGER.warning(
                        "%s vectors are ignored during partition assignment",
                        len(part_batch) - len(ids),
                    )

                progress.update(part_batch.num_rows)
                yield part_batch

    rbr = pa.RecordBatchReader.from_batches(output_schema, _partition_assignment())
    if dst_dataset_uri is None:
        dst_dataset_uri = tempfile.mkdtemp()
    write_dataset(
        rbr,
        dst_dataset_uri,
        schema=output_schema,
        max_rows_per_file=dataset.count_rows(),
        data_storage_version="stable",
    )

    progress.close()

    LOGGER.info("Saved precomputed partitions to %s", dst_dataset_uri)
    return str(dst_dataset_uri)


def one_pass_train_ivf_pq_on_accelerator(
    dataset: LanceDataset,
    column: str,
    k: int,
    metric_type: MetricType,
    accelerator: Union[str, "torch.Device"],
    num_sub_vectors: int,
    batch_size: int = 1024 * 10 * 4,
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
    filter_nan: bool = True,
):
    metric_type = _normalize_metric_type(metric_type)
    centroids, kmeans = train_ivf_centroids_on_accelerator(
        dataset,
        column,
        k,
        metric_type,
        accelerator,
        batch_size,
        sample_rate=sample_rate,
        max_iters=max_iters,
        filter_nan=filter_nan,
    )
    dataset_residuals = compute_partitions(
        dataset,
        column,
        kmeans,
        batch_size,
        num_sub_vectors=num_sub_vectors,
        filter_nan=filter_nan,
        sample_size=256 * 256,
    )
    pq_codebook, kmeans_list = train_pq_codebook_on_accelerator(
        dataset_residuals, metric_type, accelerator, num_sub_vectors, batch_size
    )
    pq_codebook = pq_codebook.astype(dtype=centroids.dtype)
    return centroids, kmeans, pq_codebook, kmeans_list


def one_pass_assign_ivf_pq_on_accelerator(
    dataset: LanceDataset,
    column: str,
    metric_type: MetricType,
    accelerator: Union[str, "torch.Device"],
    ivf_kmeans: Any,  # KMeans
    pq_kmeans_list: List[Any],  # List[KMeans]
    dst_dataset_uri: Optional[Union[str, Path]] = None,
    batch_size: int = 1024 * 10 * 4,
    *,
    filter_nan: bool = True,
    allow_cuda_tf32: bool = True,
):
    """Compute partitions for each row using GPU kmeans and spill to disk.

    Parameters
    ----------

    Returns
    -------
    str
        The absolute path of the ivfpq codes dataset, as precomputed partition buffers.
    """
    from .torch.data import LanceDataset as TorchDataset

    torch.backends.cuda.matmul.allow_tf32 = allow_cuda_tf32

    num_rows = dataset.count_rows()

    if dataset.schema.field(column).nullable and filter_nan:
        filt = f"{column} is not null"
    else:
        filt = None

    torch_ds = TorchDataset(
        dataset,
        batch_size=batch_size,
        with_row_id=True,
        columns=[column],
        filter=filt,
    )
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=1,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    num_sub_vectors = len(pq_kmeans_list)
    dim = ivf_kmeans.centroids.shape[1]
    subvector_size = dim // num_sub_vectors

    output_schema = pa.schema(
        [
            pa.field("row_id", pa.uint64()),
            pa.field("__ivf_part_id", pa.uint32()),
            pa.field("__pq_code", pa.list_(pa.uint8(), list_size=num_sub_vectors)),
        ]
    )

    progress = tqdm(total=num_rows)

    progress.set_description("Assigning partitions and computing pq codes")

    def _partition_and_pq_codes_assignment() -> Iterable[pa.RecordBatch]:
        with torch.no_grad():
            first_iter = True
            for batch in loader:
                vecs = (
                    batch[column]
                    .to(ivf_kmeans.device)
                    .reshape(-1, ivf_kmeans.centroids.shape[1])
                )

                partitions = ivf_kmeans.transform(vecs)
                ids = batch["_rowid"].reshape(-1)

                # this is expected to be true, so just assert
                assert vecs.shape[0] == ids.shape[0]

                # Ignore any invalid vectors.
                mask_gpu = partitions.isfinite() & (partitions >= 0)
                ids = ids.to(ivf_kmeans.device)[mask_gpu].cpu().reshape(-1)
                partitions = partitions[mask_gpu].cpu()
                vecs = vecs[mask_gpu]

                residual_vecs = vecs - ivf_kmeans.centroids[partitions]
                # cast centroids to the same dtype as vecs
                if first_iter:
                    first_iter = False
                    LOGGER.info("Residual shape: %s", residual_vecs.shape)
                    for kmeans in pq_kmeans_list:
                        cents: torch.Tensor = kmeans.centroids
                        kmeans.centroids = cents.to(
                            dtype=vecs.dtype, device=ivf_kmeans.device
                        )
                pq_codes = torch.stack(
                    [
                        pq_kmeans_list[i].transform(
                            residual_vecs[
                                :, i * subvector_size : (i + 1) * subvector_size
                            ]
                        )
                        for i in range(num_sub_vectors)
                    ],
                    dim=1,
                )
                pq_codes = pq_codes.to(torch.uint8)

                pq_values = pa.array(pq_codes.cpu().numpy().reshape(-1))
                pq_codes = pa.FixedSizeListArray.from_arrays(pq_values, num_sub_vectors)
                part_batch = pa.RecordBatch.from_arrays(
                    [ids, partitions, pq_codes],
                    schema=output_schema,
                )

                if len(part_batch) < len(ids):
                    LOGGER.warning(
                        "%s vectors are ignored during partition assignment",
                        len(part_batch) - len(ids),
                    )

                progress.update(part_batch.num_rows)
                yield part_batch

    rbr = pa.RecordBatchReader.from_batches(
        output_schema, _partition_and_pq_codes_assignment()
    )
    if dst_dataset_uri is None:
        dst_dataset_uri = tempfile.mkdtemp()
        if re.search(r".:\\", dst_dataset_uri) is not None:
            # Hack for Windows due to
            # https://github.com/apache/arrow-rs-object-store/issues/499
            dst_dataset_uri = dst_dataset_uri.replace("\\", "/", 1)
    ds = write_dataset(
        rbr,
        dst_dataset_uri,
        schema=output_schema,
        data_storage_version="legacy",
    )

    progress.close()

    LOGGER.info("Saved precomputed pq_codes to %s", dst_dataset_uri)

    shuffle_buffers = [
        data_file.path for frag in ds.get_fragments() for data_file in frag.data_files()
    ]
    return dst_dataset_uri, shuffle_buffers


# =============================================================================
# Hamming Distance Clustering
# =============================================================================


def hamming_clustering_for_ivf_partition(
    dataset: "LanceDataset",
    index_name: str,
    partition_id: int,
    hamming_threshold: int,
    *,
    index_segments: Optional[Iterable[Union[str, uuid.UUID]]] = None,
) -> pa.RecordBatchReader:
    """
    Perform hamming clustering on a partition of an IVF_FLAT index.

    Loads a partition from every segment of an IVF_FLAT index on a hash
    column, computes pairwise hamming distances between all hashes in the
    combined partition, filters by threshold, and clusters the results using
    union-find. All segments of the logical index must share the same global
    IVF centroids; an error is raised if they do not.

    Parameters
    ----------
    dataset : LanceDataset
        The Lance dataset containing the hash column with an IVF_FLAT index.
    index_name : str
        Name of the IVF_FLAT index on the hash column
    partition_id : int
        The partition ID within the IVF_FLAT index
    hamming_threshold : int
        Maximum hamming distance to consider as similar
    index_segments : iterable of str or uuid.UUID, optional
        If specified, only these physical index segment UUIDs of the named
        logical index contribute rows. Use
        :meth:`LanceDataset.describe_indices` to obtain segment UUIDs from
        ``IndexDescription.segments``. Defaults to all segments.

    Returns
    -------
    pa.RecordBatchReader
        A reader yielding batches with columns:

        - 'representative': uint64 - The representative row ID for each cluster
        - 'duplicates': list<uint64> - List of duplicate row IDs in each cluster
    """
    return dataset._ds.hamming_clustering_for_ivf_partition(
        index_name,
        partition_id,
        hamming_threshold,
        _normalize_index_segment_ids(index_segments),
    )


def get_ivf_partition_info(
    dataset: "LanceDataset",
    index_name: str,
    *,
    index_segments: Optional[Iterable[Union[str, uuid.UUID]]] = None,
) -> List[dict]:
    """
    Get partition information for an IVF_FLAT index.

    Partition sizes are aggregated across all segments of the logical index
    unless a subset is selected via ``index_segments``.

    Parameters
    ----------
    dataset : LanceDataset
        The Lance dataset containing the hash column with an IVF_FLAT index.
    index_name : str
        Name of the IVF_FLAT index
    index_segments : iterable of str or uuid.UUID, optional
        If specified, only these physical index segment UUIDs of the named
        logical index contribute to the sizes. Defaults to all segments.

    Returns
    -------
    list[dict]
        List of partition info dicts with 'partition_id' and 'size'
    """
    return dataset._ds.get_ivf_partition_info(
        index_name, _normalize_index_segment_ids(index_segments)
    )


def hamming_clustering_for_sample(
    dataset: "LanceDataset",
    column: str,
    sample_size: Optional[int] = None,
    hamming_threshold: int = 10,
) -> pa.RecordBatchReader:
    """
    Perform pairwise hamming distance clustering on a sample of the dataset.

    Randomly samples rows from the dataset, computes pairwise hamming distances
    between all hashes in the sample, filters by threshold, and clusters the
    results using union-find.

    Parameters
    ----------
    dataset : LanceDataset
        The Lance dataset containing the hash column.
    column : str
        Name of the hash column (must be FixedSizeList<UInt8, N> where N is a
        positive multiple of 8 bytes)
    sample_size : int, optional
        Number of rows to sample. If None, uses all rows.
    hamming_threshold : int, default 10
        Maximum hamming distance to consider as similar

    Returns
    -------
    pa.RecordBatchReader
        A reader yielding batches with columns:

        - 'representative': uint64 - The representative row ID for each cluster
        - 'duplicates': list<uint64> - List of duplicate row IDs in each cluster
    """
    return dataset._ds.hamming_clustering_for_sample(
        column, sample_size, hamming_threshold
    )


def hamming_clustering_for_range(
    dataset: "LanceDataset",
    column: str,
    fragment_id: int,
    start_row: int,
    num_rows: int,
    hamming_threshold: int = 10,
) -> pa.RecordBatchReader:
    """
    Perform pairwise hamming distance clustering on a contiguous range of rows.

    Reads a contiguous range of rows from a specific fragment, computes pairwise
    hamming distances between all hashes in the range, filters by threshold,
    and clusters the results using union-find.

    Unlike sampling, this reads sequential rows which is useful for distributed
    processing where each worker handles a specific range of a fragment.

    Parameters
    ----------
    dataset : LanceDataset
        The Lance dataset containing the hash column.
    column : str
        Name of the hash column (must be FixedSizeList<UInt8, N> where N is a
        positive multiple of 8 bytes)
    fragment_id : int
        The fragment ID to read from
    start_row : int
        The starting row offset within the fragment
    num_rows : int
        Number of rows to read from the start position
    hamming_threshold : int, default 10
        Maximum hamming distance to consider as similar

    Returns
    -------
    pa.RecordBatchReader
        A reader yielding batches with columns:

        - 'representative': uint64 - The representative row ID for each cluster
        - 'duplicates': list<uint64> - List of duplicate row IDs in each cluster
    """
    return dataset._ds.hamming_clustering_for_range(
        column, fragment_id, start_row, num_rows, hamming_threshold
    )
