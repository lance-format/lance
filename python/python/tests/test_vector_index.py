# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import json
import logging
import os
import platform
import random
import shutil
import string
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from conftest import ProgressRecorder, progress_event_tags, stage_progress_values
from lance import LanceDataset, LanceFragment
from lance.dataset import VectorIndexReader
from lance.indices import IndexFileVersion, IndicesBuilder
from lance.query import MatchQuery, PhraseQuery
from lance.util import (  # noqa: E402
    _target_partition_size_to_num_partitions,
    validate_vector_index,
)
from lance.vector import vec_to_table  # noqa: E402


def create_table(nvec=1000, ndim=128, nans=0, nullify=False, dtype=np.float32):
    mat = np.random.randn(nvec, ndim)
    if nans > 0:
        nans_mat = np.empty((nans, ndim))
        nans_mat[:] = np.nan
        mat = np.concatenate((mat, nans_mat), axis=0)
    mat = mat.astype(dtype)
    price = np.random.rand(nvec + nans) * 100

    def gen_str(n):
        return "".join(random.choices(string.ascii_letters + string.digits, k=n))

    meta = np.array([gen_str(100) for _ in range(nvec + nans)])
    tbl = (
        vec_to_table(data=mat)
        .append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
        .append_column("id", pa.array(range(nvec + nans)))
    )
    if nullify:
        idx = tbl.schema.get_field_index("vector")
        vecs = tbl[idx].to_pylist()
        nullified = [vec if i % 2 == 0 else None for i, vec in enumerate(vecs)]
        field = tbl.schema.field(idx)
        vecs = pa.array(nullified, field.type)
        tbl = tbl.set_column(idx, field, vecs)
    return tbl


def create_multivec_table(
    nvec=1000,
    nvec_per_row=5,
    ndim=128,
    nans=0,
    nullify=False,
    dtype=np.float32,
    seed=None,
):
    rng = np.random.default_rng(seed)
    text_rng = random.Random(seed)
    mat = rng.standard_normal((nvec, nvec_per_row, ndim))
    if nans > 0:
        nans_mat = np.empty((nans, ndim))
        nans_mat[:] = np.nan
        mat = np.concatenate((mat, nans_mat), axis=0)
    mat = mat.astype(dtype)
    price = rng.random(nvec + nans) * 100

    def gen_str(n):
        return "".join(text_rng.choices(string.ascii_letters + string.digits, k=n))

    meta = np.array([gen_str(100) for _ in range(nvec + nans)])

    multi_vec_type = pa.list_(pa.list_(pa.float32(), ndim))
    tbl = pa.Table.from_arrays(
        [
            pa.array((mat[i].tolist() for i in range(nvec)), type=multi_vec_type),
        ],
        schema=pa.schema(
            [
                pa.field("vector", pa.list_(pa.list_(pa.float32(), ndim))),
            ]
        ),
    )
    tbl = (
        tbl.append_column("price", pa.array(price))
        .append_column("meta", pa.array(meta))
        .append_column("id", pa.array(range(nvec + nans)))
    )
    if nullify:
        idx = tbl.schema.get_field_index("vector")
        vecs = tbl[idx].to_pylist()
        nullified = [vec if i % 2 == 0 else None for i, vec in enumerate(vecs)]
        field = tbl.schema.field(idx)
        vecs = pa.array(nullified, field.type)
        tbl = tbl.set_column(idx, field, vecs)
    return tbl


@pytest.fixture()
def dataset(tmp_path):
    tbl = create_table()
    yield lance.write_dataset(tbl, tmp_path)


@pytest.fixture()
def indexed_dataset(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    yield dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        max_iters=2,
        sample_rate=2,
    )


@pytest.fixture()
def multivec_dataset():
    # Keep at least 100 logical rows for the top-k assertions below. Five
    # vectors per row still exercises multivector deduplication and fanout.
    tbl = create_multivec_table(nvec=128, seed=42)
    yield lance.write_dataset(tbl, "memory://")


@pytest.fixture()
def indexed_multivec_dataset(multivec_dataset):
    yield multivec_dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        num_bits=4,
        max_iters=2,
        sample_rate=2,
        metric="cosine",
    )


def run(ds, q=None, assert_func=None):
    if q is None:
        q = np.random.randn(128)
    project = [None, ["price"], ["vector", "price"], ["vector", "meta", "price"]]
    refine = [None, 1, 2]
    filters = [None, pc.field("price") > 50.0]
    times = []

    for columns in project:
        expected_columns = []
        if columns is None:
            expected_columns.extend(ds.schema.names)
        else:
            expected_columns.extend(columns)
        # TODO: _distance shouldn't be returned by default either
        if "_distance" not in expected_columns:
            expected_columns.append("_distance")

        for filter_ in filters:
            for rf in refine:
                start = time.time()
                rs = ds.to_table(
                    columns=columns,
                    nearest={
                        "column": "vector",
                        "q": q,
                        "k": 15,
                        "nprobes": 1,
                        "refine_factor": rf,
                    },
                    filter=filter_,
                )
                end = time.time()
                times.append(end - start)
                assert rs.column_names == expected_columns
                if filter_ is not None:
                    inmem = pa.dataset.dataset(rs)
                    assert len(inmem.to_table(filter=filter_)) == len(rs)
                else:
                    assert len(rs) == 15
                    distances = rs["_distance"].to_numpy()
                    assert (distances.max() - distances.min()) > 1e-6
                    if assert_func is not None:
                        assert_func(rs)
    return times


def test_flat(dataset):
    run(dataset)


@pytest.mark.parametrize(
    "queries",
    [
        np.random.randn(2, 128).astype(np.float32),
        np.random.randn(1, 128).astype(np.float32),
    ],
    ids=["two_queries", "single_query"],
)
def test_batch_flat_query_matches_repeated_single_queries(dataset, queries):
    k = 5
    query_count = queries.shape[0]

    batch = dataset.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": queries,
            "k": k,
            "use_index": False,
        },
    )

    assert batch.num_rows == query_count * k
    assert batch.column_names == ["query_index", "id", "_distance"]
    query_index_field = batch.schema.field("query_index")
    assert query_index_field.type == pa.int32()
    assert not query_index_field.nullable
    expected_query_index = sum([[i] * k for i in range(query_count)], [])
    assert batch["query_index"].to_pylist() == expected_query_index

    _assert_batch_matches_single_queries(
        dataset,
        queries,
        k=k,
        nearest_kwargs={"use_index": False},
    )


@pytest.mark.parametrize("metric", ["l2", "cosine"])
@pytest.mark.parametrize("query_count", [3, 1], ids=["three_queries", "single_query"])
def test_batch_indexed_query_matches_repeated_single_queries(
    dataset, metric, query_count
):
    indexed = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        metric=metric,
    )
    # Give the query vectors deliberately different magnitudes: a cosine batch
    # that normalized the whole concatenated key by one global norm would scale
    # them unequally and diverge from per-query single search.
    scales = np.linspace(0.1, 10.0, query_count).reshape(-1, 1)
    queries = (np.random.randn(query_count, 128) * scales).astype(np.float32)
    k = 5

    # nprobes covers every partition so the shared-scan batch path and the
    # repeated single-query path search the same partitions deterministically.
    nearest_kwargs = {"use_index": True, "nprobes": 4}
    batch = indexed.to_table(
        columns=["id"],
        nearest={"column": "vector", "q": queries, "k": k, **nearest_kwargs},
    )

    assert batch.column_names == ["query_index", "id", "_distance"]
    assert batch["query_index"].to_pylist() == sum(
        [[i] * k for i in range(query_count)], []
    )

    _assert_batch_matches_single_queries(
        indexed,
        queries,
        k=k,
        nearest_kwargs=nearest_kwargs,
    )


def _assert_batch_matches_single_queries(ds, queries, k, nearest_kwargs):
    batch = ds.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": queries,
            "k": k,
            **nearest_kwargs,
        },
    )
    if "distance_range" in nearest_kwargs:
        lo, hi = nearest_kwargs["distance_range"]
        assert all(lo <= d < hi for d in batch["_distance"].to_pylist())

    for query_index, query in enumerate(queries):
        single = ds.to_table(
            columns=["id"],
            nearest={
                "column": "vector",
                "q": query,
                "k": k,
                **nearest_kwargs,
            },
        )
        batch_slice = batch.filter(pc.field("query_index") == query_index)
        assert batch_slice["id"].to_pylist() == single["id"].to_pylist()
        np.testing.assert_allclose(
            batch_slice["_distance"].to_numpy(),
            single["_distance"].to_numpy(),
        )


def test_batch_vector_search_rejects_dataset_query_index_column(tmp_path):
    dim = 128
    table = create_table(nvec=80, ndim=dim)
    table = table.append_column(
        "query_index",
        pa.array(range(80), type=pa.uint32()),
    )
    ds = lance.write_dataset(table, tmp_path / "with_query_index")

    queries = np.random.randn(2, dim).astype(np.float32)
    with pytest.raises(Exception, match="query_index"):
        ds.to_table(
            columns=["id", "query_index"],
            nearest={
                "column": "vector",
                "q": queries,
                "k": 5,
                "use_index": False,
            },
        )


def test_flat_1d_query_length_multiple_of_dim_is_rejected(dataset):
    q = np.random.randn(256).astype(np.float32)
    with pytest.raises(ValueError, match=r"256.*128"):
        dataset.to_table(
            columns=["id"],
            nearest={
                "column": "vector",
                "q": q,
                "k": 5,
                "use_index": False,
            },
        )


def test_batch_fast_search_without_index_returns_empty_with_query_index(dataset):
    queries = np.random.randn(2, 128).astype(np.float32)
    batch = dataset.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": queries,
            "k": 5,
        },
        fast_search=True,
    )
    assert batch.num_rows == 0
    assert "query_index" in batch.column_names


def test_ann(indexed_dataset):
    run(indexed_dataset)


def test_create_index_progress_callback_vector(tmp_path):
    ds = _make_sample_dataset_base(tmp_path, "vector_progress", 1500, 128)
    recorder = ProgressRecorder()

    ds.create_index(
        column="vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=4,
        progress_callback=recorder,
    )

    tags = progress_event_tags(recorder.events)
    expected_order = [
        "start:train_ivf",
        "complete:train_ivf",
        "start:train_quantizer",
        "complete:train_quantizer",
        "start:shuffle",
        "complete:shuffle",
        "start:merge_partitions",
        "complete:merge_partitions",
    ]
    positions = [tags.index(tag) for tag in expected_order]
    assert positions == sorted(positions)

    shuffle_progress = stage_progress_values(recorder.events, "shuffle")
    assert shuffle_progress
    assert shuffle_progress[-1] == ds.count_rows()

    merge_progress = stage_progress_values(recorder.events, "merge_partitions")
    assert merge_progress
    assert merge_progress[-1] == 4


def test_create_index_progress_callback_error_before_completion_propagates(tmp_path):
    ds = _make_sample_dataset_base(
        tmp_path, "vector_progress_post_commit_error", 1500, 128
    )
    recorder = ProgressRecorder(fail_on_tag="start:train_ivf")

    with pytest.raises(RuntimeError, match="progress callback failure"):
        ds.create_index(
            column="vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=4,
            progress_callback=recorder,
        )

    tags = progress_event_tags(recorder.events)
    assert tags == ["start:train_ivf"]
    assert not ds.has_index
    assert ds.describe_indices() == []


def test_distributed_ivf_pq_partition_window_env_override(tmp_path, monkeypatch):
    # Keep this before other distributed vector merge tests so the process-level
    # lazy window size initialization reads this override.
    monkeypatch.setenv("LANCE_IVF_PQ_MERGE_PARTITION_WINDOW_SIZE", "4")
    monkeypatch.setenv("LANCE_IVF_PQ_MERGE_PARTITION_PREFETCH_WINDOW_COUNT", "2")

    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((640, 32), dtype=np.float32)
    data = vec_to_table(data=matrix).append_column("id", pa.array(range(640)))
    q = rng.standard_normal(32).astype(np.float32)
    assert_distributed_vector_consistency(
        data,
        "vector",
        index_type="IVF_PQ",
        index_params={
            "num_partitions": 10,
            "num_sub_vectors": 4,
            "max_iters": 2,
        },
        queries=[q],
        topk=10,
        world=2,
        tmp_path=tmp_path,
        similarity_metric="recall",
        similarity_threshold=0.80,
    )


@pytest.mark.parametrize(
    "fixture_name,index_type,index_params,similarity_threshold",
    [
        ("dataset", "IVF_FLAT", {"num_partitions": 4}, 0.80),
        (
            "indexed_dataset",
            "IVF_PQ",
            {"num_partitions": 4, "num_sub_vectors": 16},
            0.80,
        ),
        ("dataset", "IVF_SQ", {"num_partitions": 4}, 0.80),
    ],
)
def test_distributed_vector(
    request, fixture_name, index_type, index_params, similarity_threshold
):
    ds = request.getfixturevalue(fixture_name)
    q = np.random.default_rng(42).standard_normal(128).astype(np.float32)
    assert_distributed_vector_consistency(
        ds.to_table(),
        "vector",
        index_type=index_type,
        index_params=index_params,
        queries=[q],
        topk=10,
        world=2,
        similarity_metric="recall",
        similarity_threshold=similarity_threshold,
    )


def test_rowid_order(indexed_dataset):
    rs = indexed_dataset.to_table(
        columns=["meta"],
        with_row_id=True,
        nearest={
            "column": "vector",
            "q": np.random.randn(128),
            "k": 10,
            "use_index": False,
        },
        limit=10,
    )

    assert rs.schema[0].name == "meta"
    assert rs.schema[1].name == "_distance"
    assert rs.schema[2].name == "_rowid"


def test_ann_append(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )
    new_data = create_table(nvec=10)
    dataset = lance.write_dataset(new_data, dataset.uri, mode="append")
    q = new_data["vector"][0].as_py()

    def func(rs: pa.Table):
        if "vector" not in rs:
            return
        assert rs["vector"][0].as_py() == q

    run(dataset, q=np.array(q), assert_func=func)


def test_invalid_subvectors(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    with pytest.raises(
        ValueError,
        match="dimension .* must be divisible by num_sub_vectors",
    ):
        dataset.create_index(
            "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=15
        )


@pytest.mark.cuda
def test_invalid_subvectors_cuda(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    with pytest.raises(
        ValueError,
        match="dimension .* must be divisible by num_sub_vectors",
    ):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=15,
            accelerator="cuda",
        )


@pytest.mark.cuda
def test_f16_cuda(tmp_path):
    tbl = create_table(dtype=np.float16)
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        accelerator="cuda",
        one_pass_ivfpq=True,
    )
    validate_vector_index(dataset, "vector")


@pytest.mark.parametrize(
    "index_file_version", [IndexFileVersion.V3, IndexFileVersion.LEGACY]
)
def test_index_with_nans(tmp_path, index_file_version):
    tbl = create_table(nvec=256, ndim=32, nans=8)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        max_iters=2,
        index_file_version=index_file_version,
    )
    idx_stats = dataset.stats.index_stats("vector_idx")
    assert idx_stats["indices"][0]["index_file_version"] == index_file_version
    validate_vector_index(dataset, "vector", sample_size=16)


@pytest.mark.parametrize(
    "index_file_version", [IndexFileVersion.V3, IndexFileVersion.LEGACY]
)
def test_torch_index_with_nans(tmp_path, index_file_version):
    torch = pytest.importorskip("torch")

    # Torch PQ initialization samples 256 valid residuals. Keep a small margin
    # after NaN filtering so every platform can produce a complete sample batch.
    tbl = create_table(nvec=320, ndim=32, nans=8)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        max_iters=2,
        accelerator=torch.device("cpu"),
        one_pass_ivfpq=True,
        index_file_version=index_file_version,
    )
    idx_stats = dataset.stats.index_stats("vector_idx")
    assert idx_stats["indices"][0]["index_file_version"] == index_file_version
    validate_vector_index(dataset, "vector", sample_size=16)


def test_torch_index_nan_init_centroid(tmp_path):
    """A NaN vector must never seed a centroid.

    `vector is not null` does not exclude NaN, so sampling could pick one; with
    a single partition that left every residual NaN and the index build failed.
    """
    torch = pytest.importorskip("torch")
    from lance.torch.data import LanceDataset as TorchDataset
    from lance.vector import _sample_init_centroids

    # Only the last 8 rows are finite, so any sample that keeps NaN rows seeds
    # the centroid with one.
    mat = np.full((32, 8), np.nan, dtype=np.float32)
    mat[24:] = np.random.randn(8, 8).astype(np.float32)
    dataset = lance.write_dataset(vec_to_table(data=mat), tmp_path)
    assert dataset.to_table()["vector"].null_count == 0

    ds = TorchDataset(dataset, batch_size=1, columns=["vector"], samples=32)
    centroids = _sample_init_centroids(ds, 4, filter_nan=True)
    assert centroids.shape[0] == 4
    assert torch.isfinite(centroids).all()


def test_torch_index_all_nan_rejected(tmp_path):
    pytest.importorskip("torch")
    from lance.torch.data import LanceDataset as TorchDataset
    from lance.vector import _sample_init_centroids

    mat = np.full((16, 8), np.nan, dtype=np.float32)
    dataset = lance.write_dataset(vec_to_table(data=mat), tmp_path)

    ds = TorchDataset(dataset, batch_size=1, columns=["vector"], samples=16)
    with pytest.raises(ValueError, match="all null or non-finite"):
        _sample_init_centroids(ds, 1, filter_nan=True)


def test_index_with_no_centroid_movement(tmp_path):
    torch = pytest.importorskip("torch")

    # this test makes the centroids essentially [1..]
    # this makes sure the early stop condition in the index building code
    # doesn't do divide by zero
    # Torch one-pass PQ emits an 8-bit codebook, which requires 256 rows.
    mat = np.ones((256, 16), dtype=np.float32)

    tbl = vec_to_table(data=mat)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        max_iters=2,
        accelerator=torch.device("cpu"),
    )
    validate_vector_index(dataset, "vector", sample_size=8)


def test_index_with_pq_codebook(tmp_path):
    dim = 16
    rng = np.random.default_rng(42)
    # Eight-bit PQ still requires its 256 centroid training rows even when the
    # initial codebook is supplied; reducing the dimension keeps this fixture small.
    vectors = rng.standard_normal((256, dim), dtype=np.float32)
    tbl = vec_to_table(data=vectors)
    dataset = lance.write_dataset(tbl, tmp_path)
    pq_codebook = rng.standard_normal((4, 256, dim // 4), dtype=np.float32)
    ivf_centroids = rng.standard_normal((1, dim), dtype=np.float32)

    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        max_iters=2,
        ivf_centroids=ivf_centroids,
        pq_codebook=pq_codebook,
    )
    index = dataset.stats.index_stats("vector_idx")
    assert index["indices"][0]["sub_index"]["nbits"] == 8
    validate_vector_index(
        dataset, "vector", refine_factor=256, sample_size=8, pass_threshold=0.99
    )

    pq_codebook = pa.FixedShapeTensorArray.from_numpy_ndarray(pq_codebook)

    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        max_iters=2,
        ivf_centroids=ivf_centroids,
        pq_codebook=pq_codebook,
        replace=True,
    )
    validate_vector_index(
        dataset, "vector", refine_factor=256, sample_size=8, pass_threshold=0.99
    )


def test_index_with_4bit_numpy_pq_codebook(tmp_path):
    dim = 32
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((32, dim), dtype=np.float32)
    tbl = vec_to_table(data=vectors)
    dataset = lance.write_dataset(tbl, tmp_path)
    pq_codebook = rng.standard_normal((4, 16, dim // 4), dtype=np.float32)

    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        num_bits=4,
        max_iters=2,
        ivf_centroids=rng.standard_normal((1, dim), dtype=np.float32),
        pq_codebook=pq_codebook,
    )

    index = dataset.stats.index_stats("vector_idx")
    assert index["indices"][0]["sub_index"]["nbits"] == 4

    result = dataset.to_table(
        nearest={
            "column": "vector",
            "q": vectors[0],
            "k": 10,
        }
    )
    assert result.num_rows == 10


def test_index_with_pq_codebook_rejects_wrong_num_bits_shape(tmp_path):
    dim = 16
    rng = np.random.default_rng(42)
    tbl = vec_to_table(data=rng.standard_normal((8, dim), dtype=np.float32))
    dataset = lance.write_dataset(tbl, tmp_path)
    pq_codebook = rng.standard_normal((4, 256, dim // 4), dtype=np.float32)

    with pytest.raises(
        ValueError,
        match=r"\(sub_vectors, 16, dim\) for num_bits=4, got \(4, 256, 4\)",
    ):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=1,
            num_sub_vectors=4,
            num_bits=4,
            ivf_centroids=rng.standard_normal((1, dim), dtype=np.float32),
            pq_codebook=pq_codebook,
        )


@pytest.mark.cuda
@pytest.mark.parametrize("nullify", [False, True])
def test_create_index_using_cuda(tmp_path, nullify):
    tbl = create_table(nullify=nullify)
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        accelerator="cuda",
    )
    q = np.random.randn(128)
    expected = dataset.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": q,
            "k": 10,  # Use non-default k
        },
    )["id"].to_numpy()
    assert len(expected) == 10

    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        metric="cosine",
        num_partitions=4,
        num_sub_vectors=16,
        accelerator="cuda",
    )
    q = np.random.randn(128)
    expected = dataset.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": q,
            "k": 10,  # Use non-default k
        },
    )["id"].to_numpy()
    assert len(expected) == 10


def test_create_index_unsupported_accelerator(tmp_path):
    # Even attempting to use an accelerator will trigger torch import
    # so make sure it's available
    pytest.importorskip("torch")

    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="no-supported",
        )

    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="0cuda",
        )

    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="cuda-0",
        )

    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="cuda:",
        )

    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="cuda:abc",
        )


def test_create_index_accelerator_fallback(tmp_path, caplog):
    tbl = create_table(nvec=64, ndim=32)
    dataset = lance.write_dataset(tbl, tmp_path)

    with caplog.at_level(logging.WARNING):
        dataset = dataset.create_index(
            "vector",
            index_type="IVF_HNSW_SQ",
            num_partitions=1,
            max_iters=2,
            max_level=2,
            m=4,
            ef_construction=16,
            accelerator="cuda",
        )

    stats = dataset.stats.index_stats("vector_idx")
    assert stats["index_type"] == "IVF_HNSW_SQ"
    assert any(
        "does not support GPU acceleration; falling back to CPU" in record.message
        for record in caplog.records
    )


def test_use_index(dataset, tmp_path):
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )
    q = np.random.randn(128)
    expected = dataset.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": q,
            "k": 12,  # Use non-default k
        },
    )["id"].to_numpy()

    actual = ann_ds.to_table(
        columns=["id"],
        nearest={"column": "vector", "q": q, "k": 12, "use_index": False},
    )["id"].to_numpy()

    assert np.all(expected == actual)

    # Can omit k but provide limit
    actual = ann_ds.to_table(
        columns=["id"],
        nearest={"column": "vector", "q": q, "use_index": False},
        limit=12,
    )["id"].to_numpy()
    assert np.all(expected == actual)


def test_nearest_errors(dataset, tmp_path):
    import pandas as pd

    with pytest.raises(ValueError, match="does not match index column size"):
        dataset.to_table(
            columns=["id"],
            nearest={"column": "vector", "q": np.random.randn(127), "k": 10},
        )

    df = pd.DataFrame({"a": [5], "b": [10]})
    ds = lance.write_dataset(pa.Table.from_pandas(df), tmp_path / "dataset.lance")

    with pytest.raises(TypeError, match="must be a vector"):
        ds.to_table(nearest={"column": "a", "q": np.random.randn(128), "k": 10})


def test_has_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )
    assert ann_ds.has_index

    assert ann_ds.describe_indices()[0].field_names == ["vector"]


def test_index_type(tmp_path):
    index_cases = [
        ("IVF_PQ", {"num_sub_vectors": 4, "num_bits": 4}),
        (
            "IVF_HNSW_SQ",
            {"max_level": 2, "m": 4, "ef_construction": 16},
        ),
        (
            "IVF_HNSW_PQ",
            {
                "num_sub_vectors": 4,
                "num_bits": 4,
                "max_level": 2,
                "m": 4,
                "ef_construction": 16,
            },
        ),
        (
            "IVF_HNSW_FLAT",
            {"max_level": 2, "m": 4, "ef_construction": 16},
        ),
    ]
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((64, 32), dtype=np.float32)
    table = vec_to_table(data=vectors).append_column("id", pa.array(range(64)))
    ann_ds = lance.write_dataset(table, tmp_path / "replace_index_type")
    assert not ann_ds.has_index

    for case_index, (index_type, index_options) in enumerate(index_cases):
        ann_ds = ann_ds.create_index(
            "vector",
            index_type=index_type,
            num_partitions=1,
            max_iters=2,
            sample_rate=2,
            replace=case_index > 0,
            **index_options,
        )
        stats = ann_ds.stats.index_stats("vector_idx")
        assert stats["index_type"] == index_type
        assert stats["num_indices"] == 1
        indices = ann_ds.describe_indices()
        assert len(indices) == 1
        assert indices[0].field_names == ["vector"]

        nearest = {
            "column": "vector",
            "q": vectors[0],
            "k": 10,
            "nprobes": 1,
            "refine_factor": 4,
        }
        if "HNSW" in index_type:
            nearest["ef"] = 64
        actual = ann_ds.to_table(columns=["id"], nearest=nearest)
        expected = ann_ds.to_table(
            columns=["id"],
            nearest={
                "column": "vector",
                "q": vectors[0],
                "k": 10,
                "use_index": False,
            },
        )
        actual_ids = set(actual["id"].to_pylist())
        expected_ids = set(expected["id"].to_pylist())
        assert actual.num_rows == 10
        assert len(actual_ids) == 10
        assert len(actual_ids & expected_ids) / len(expected_ids) >= 0.5


def test_create_dot_index(tmp_path):
    rng = np.random.default_rng(42)
    table = vec_to_table(data=rng.standard_normal((64, 32), dtype=np.float32))
    ann_ds = lance.write_dataset(table, tmp_path / "indexed.lance")
    assert not ann_ds.has_index
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        num_bits=4,
        max_iters=2,
        metric="dot",
    )
    assert ann_ds.has_index


def test_create_4bit_ivf_pq_index(tmp_path):
    rng = np.random.default_rng(42)
    table = vec_to_table(data=rng.standard_normal((32, 32), dtype=np.float32))
    ann_ds = lance.write_dataset(table, tmp_path / "indexed.lance")
    assert not ann_ds.has_index
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        num_bits=4,
        max_iters=2,
        metric="l2",
    )
    index = ann_ds.stats.index_stats("vector_idx")
    assert index["indices"][0]["sub_index"]["nbits"] == 4


def test_create_ivf_pq_with_target_partition_size(dataset, tmp_path):
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_sub_vectors=16,
        target_partition_size=1000,
    )
    assert ann_ds.stats.index_stats("vector_idx")["indices"][0]["num_partitions"] == 1

    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_sub_vectors=16,
        target_partition_size=500,
        replace=True,
    )
    assert ann_ds.stats.index_stats("vector_idx")["indices"][0]["num_partitions"] == 2

    # setting both num_partitions and target_partition_size will use num_partitions
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_sub_vectors=16,
        num_partitions=2,
        target_partition_size=1000,
        replace=True,
    )
    assert ann_ds.stats.index_stats("vector_idx")["indices"][0]["num_partitions"] == 2


def test_target_partition_size_to_num_partitions_clamps():
    assert _target_partition_size_to_num_partitions(1000, 1000) == 1
    assert _target_partition_size_to_num_partitions(1000, 500) == 2
    assert _target_partition_size_to_num_partitions(8192 * 5000, 8192) == 4096


def test_index_size_stats(tmp_path: Path):
    num_rows = 512
    dims = 32
    schema = pa.schema([pa.field("a", pa.list_(pa.float32(), dims), False)])
    values = pc.random(num_rows * dims).cast("float32")
    table = pa.Table.from_pydict(
        {"a": pa.FixedSizeListArray.from_arrays(values, dims)}, schema=schema
    )

    base_dir = tmp_path / "test"

    dataset = lance.write_dataset(table, base_dir)

    index_name = "vec_idx"
    dataset.create_index(
        "a", "IVF_PQ", name=index_name, num_partitions=2, num_sub_vectors=1
    )

    # Expect to see non-zero sizes here but all sizes are zero
    stats = dataset.stats.index_stats(index_name)
    stats = stats["indices"][0]
    assert stats["partitions"][0]["size"] + stats["partitions"][1]["size"] == num_rows


def test_ivf_flat_over_binary_vector(tmp_path):
    dim = 128
    nvec = 1000
    data = np.random.randint(0, 256, (nvec, dim // 8)).tolist()
    array = pa.array(data, type=pa.list_(pa.uint8(), dim // 8))
    tbl = pa.Table.from_pydict({"vector": array})
    ds = lance.write_dataset(tbl, tmp_path)
    ds.create_index("vector", index_type="IVF_FLAT", num_partitions=4, metric="hamming")
    stats = ds.stats.index_stats("vector_idx")
    assert stats["indices"][0]["metric_type"] == "hamming"
    assert stats["index_type"] == "IVF_FLAT"

    query = np.random.randint(0, 256, dim // 8).astype(np.uint8)
    ds.to_table(
        nearest={
            "column": "vector",
            "q": query,
            "k": 10,
            "metric": "hamming",
        }
    )


def test_ivf_flat_respects_index_metric_binary(tmp_path):
    # Searching with binary vectors should default to hamming distance
    table = pa.Table.from_pydict(
        {
            "vector": pa.array([[0], [128], [255]], type=pa.list_(pa.uint8(), 1)),
            "id": pa.array([0, 1, 2], type=pa.int32()),
        }
    )

    ds = lance.write_dataset(table, tmp_path)
    ds = ds.create_index(
        "vector",
        index_type="IVF_FLAT",
        num_partitions=1,
        metric="hamming",
    )

    query = np.array([128], dtype=np.uint8)

    # Search should succeed and use the index's Hamming metric.
    indexed = ds.scanner(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": query,
            "k": 3,
        },
    )
    plan = indexed.explain_plan()
    indexed = indexed.to_table()

    # Should succeed even though user asked for L2 (index metric is used).
    assert indexed["id"].to_pylist() == [1, 0, 2]
    assert "metric=Hamming" in plan
    assert "metric=L2" not in plan


def test_bruteforce_uses_user_metric(tmp_path):
    # Even if an index exists, a brute-force scan (use_index=False) should
    # respect the user-specified metric instead of the index metric.
    vectors = np.array(
        [
            [10.0, 10.0],  # Large magnitude, best under dot product
            [-1.0, -1.0],
            [1.0, 1.0],  # Closest under L2
        ],
        dtype=np.float32,
    )
    table = pa.Table.from_pydict(
        {
            "vector": pa.array(vectors.tolist(), type=pa.list_(pa.float32(), 2)),
            "id": pa.array([0, 1, 2], type=pa.int32()),
        }
    )

    ds = lance.write_dataset(table, tmp_path)
    # Build an index with L2 metric.
    ds = ds.create_index(
        "vector",
        index_type="IVF_FLAT",
        num_partitions=1,
        metric="l2",
    )

    query = np.array([1.0, 1.0], dtype=np.float32)

    # Brute-force search should honor the requested dot metric (not the index's L2).
    brute_force = ds.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": query,
            "k": 3,
            "metric": "dot",
            "use_index": False,
        },
    )

    # Under dot product the largest magnitude vector ranks first; under L2 it is last.
    assert brute_force["id"].to_pylist() == [0, 2, 1]


def test_create_ivf_sq_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_SQ",
        num_partitions=4,
    )
    assert ann_ds.describe_indices()[0].field_names == ["vector"]


def test_create_ivf_rq_index():
    ds = lance.write_dataset(create_table(), "memory://")
    ds = ds.create_index(
        "vector",
        index_type="IVF_RQ",
        num_partitions=4,
    )
    assert ds.describe_indices()[0].field_names == ["vector"]
    stats = ds.stats.index_stats("vector_idx")
    assert stats["indices"][0]["sub_index"]["num_bits"] == 5
    assert stats["indices"][0]["sub_index"]["packed"] is True

    with pytest.raises(
        NotImplementedError,
        match="Creating empty vector indices with train=False is not yet implemented",
    ):
        ds.delete("id>=0")
        ds = ds.create_index(
            "vector",
            index_type="IVF_RQ",
            num_partitions=4,
            num_bits=1,
            replace=True,
        )

    zero_vectors = np.zeros((1000, 128)).astype(np.float32).tolist()
    tbl = pa.Table.from_pydict(
        {"vector": pa.array(zero_vectors, type=pa.list_(pa.float32(), 128))}
    )
    ds = lance.write_dataset(tbl, "memory://", mode="overwrite")
    ds = ds.create_index(
        "vector",
        index_type="IVF_RQ",
        num_partitions=4,
        num_bits=1,
    )

    res = ds.to_table(
        nearest={
            "column": "vector",
            "q": np.zeros(128),
            "k": 10,
        }
    )
    assert res.num_rows == 10
    assert res["_distance"].to_numpy().min() == 0.0
    assert res["_distance"].to_numpy().max() == 0.0


def test_build_rq_model_default_num_bits():
    from lance.lance import indices

    model = json.loads(indices.build_rq_model(dimension=8))
    assert model["num_bits"] == 5


def test_create_ivf_rq_skip_transpose():
    ds = lance.write_dataset(create_table(), "memory://")
    ds = ds.create_index(
        "vector",
        index_type="IVF_RQ",
        num_partitions=4,
        num_bits=1,
        skip_transpose=True,
    )
    stats = ds.stats.index_stats("vector_idx")
    assert stats["indices"][0]["sub_index"]["packed"] is False


def _assert_recall_at_least(ds, query, metric=None, k=10, recall_requirement=0.5):
    nearest = {"column": "vector", "q": query, "k": k}
    if metric is not None:
        nearest["metric"] = metric

    gt_ids = ds.to_table(nearest=nearest, columns=["id"])["id"].to_numpy()
    create_index_kwargs = {
        "index_type": "IVF_RQ",
        "num_partitions": 4,
        "num_bits": 9,
    }
    if metric is not None:
        create_index_kwargs["metric"] = metric
    indexed = ds.create_index("vector", **create_index_kwargs)
    result_ids = indexed.to_table(nearest=nearest, columns=["id"])["id"].to_numpy()

    assert result_ids.shape[0] == k
    recall = len(set(gt_ids) & set(result_ids)) / k
    assert recall >= recall_requirement, (
        f"recall={recall}, gt={gt_ids}, result={result_ids}"
    )
    return indexed


def test_create_ivf_rq_multi_bit_searches_l2_and_cosine():
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((1000, 128)).astype(np.float32)
    tbl = vec_to_table(data=mat).append_column("id", pa.array(range(len(mat))))

    ds = lance.write_dataset(tbl, "memory://")
    ds = _assert_recall_at_least(ds, mat[0])
    stats = ds.stats.index_stats("vector_idx")
    assert stats["indices"][0]["sub_index"]["num_bits"] == 9
    assert stats["indices"][0]["sub_index"]["query_estimator"] == "raw_query"
    for approx_mode in ["fast", "normal", "accurate"]:
        result = ds.to_table(
            nearest={
                "column": "vector",
                "q": mat[0],
                "k": 10,
                "approx_mode": approx_mode,
            },
            columns=["id"],
        )
        assert result.num_rows == 10

    cosine_ds = lance.write_dataset(tbl, "memory://")
    cosine_ds = _assert_recall_at_least(cosine_ds, mat[1], metric="cosine")
    cosine_stats = cosine_ds.stats.index_stats("vector_idx")
    assert cosine_stats["indices"][0]["sub_index"]["num_bits"] == 9
    assert cosine_stats["indices"][0]["sub_index"]["query_estimator"] == "raw_query"


def test_create_ivf_rq_requires_dim_divisible_by_8():
    vectors = np.zeros((1000, 30), dtype=np.float32).tolist()
    tbl = pa.Table.from_pydict(
        {"vector": pa.array(vectors, type=pa.list_(pa.float32(), 30))}
    )
    ds = lance.write_dataset(tbl, "memory://", mode="overwrite")

    with pytest.raises(
        ValueError, match="vector dimension must be divisible by 8 for IVF_RQ"
    ):
        ds.create_index(
            "vector",
            index_type="IVF_RQ",
            num_partitions=4,
            num_bits=1,
        )


def test_create_ivf_rq_mostly_null():
    ndim = 128
    nvec = 100
    nnull = 9900
    vectors = np.random.randn(nvec, ndim).astype(np.float32).tolist()
    vectors += [None] * nnull
    tbl = pa.table(
        {
            "vector": pa.array(vectors, type=pa.list_(pa.float32(), ndim)),
            "id": pa.array(range(nvec + nnull), type=pa.int32()),
        }
    )
    ds = lance.write_dataset(tbl, "memory://")
    ds = ds.create_index(
        "vector",
        index_type="IVF_RQ",
        num_partitions=4,
        num_bits=1,
    )

    q = np.random.randn(ndim).astype(np.float32)
    result = ds.to_table(
        nearest={"column": "vector", "q": q, "k": 10},
    )
    assert result.num_rows == 10


def test_multivec_ann(indexed_multivec_dataset: lance.LanceDataset):
    rng = np.random.default_rng(42)
    query = rng.random((5, 128))
    results = indexed_multivec_dataset.scanner(
        nearest={
            "column": "vector",
            "q": query,
            "k": 100,
            "nprobes": 1,
            "refine_factor": 2,
        }
    ).to_table()
    assert results.num_rows == 100
    assert results["vector"].type == pa.list_(pa.list_(pa.float32(), 128))
    assert len(results["vector"][0]) == 5
    ground_truth = indexed_multivec_dataset.to_table(
        columns=["id"],
        nearest={"column": "vector", "q": query, "k": 100, "use_index": False},
    )
    actual_ids = set(results["id"].to_pylist())
    expected_ids = set(ground_truth["id"].to_pylist())
    assert len(actual_ids & expected_ids) / len(expected_ids) >= 0.5

    # query with single vector also works
    query = rng.random(128)
    results = indexed_multivec_dataset.to_table(
        nearest={"column": "vector", "q": query, "k": 100}
    )
    # we don't verify the number of results here,
    # because for multivector, it's not guaranteed to return k results
    assert results["vector"].type == pa.list_(pa.list_(pa.float32(), 128))
    assert len(results["vector"][0]) == 5

    query = [query, query]
    doubled_results = indexed_multivec_dataset.to_table(
        nearest={"column": "vector", "q": query, "k": 100}
    )
    assert len(results) == len(doubled_results)
    for i in range(len(results)):
        assert (
            results["_distance"][i].as_py() * 2
            == doubled_results["_distance"][i].as_py()
        )

    # query with a vector that dim not match
    query = rng.random(256)
    with pytest.raises(ValueError, match="does not match index column size"):
        indexed_multivec_dataset.to_table(
            nearest={"column": "vector", "q": query, "k": 100}
        )

    # query with a list of vectors that some dim not match
    query = [rng.random(128)] * 5 + [rng.random(256)]
    with pytest.raises(ValueError, match="All query vectors must have the same length"):
        indexed_multivec_dataset.to_table(
            nearest={"column": "vector", "q": query, "k": 100}
        )


def test_multivec_search_paths(tmp_path: Path):
    vector_type = pa.list_(pa.list_(pa.float32(), 2))
    query = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    uri = tmp_path / "multivec_distance.lance"

    indexed_rows = pa.table(
        {
            "id": pa.array([0, 1], type=pa.int32()),
            "vector": pa.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [1.0, 0.0]],
                ],
                type=vector_type,
            ),
        }
    )
    dataset = lance.write_dataset(indexed_rows, uri)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_FLAT",
        metric="cosine",
        num_partitions=1,
    )

    unindexed_rows = pa.table(
        {
            "id": pa.array([2, 3], type=pa.int32()),
            "vector": pa.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[-1.0, 0.0], [0.0, -1.0]],
                ],
                type=vector_type,
            ),
        }
    )
    dataset = lance.write_dataset(unindexed_rows, uri, mode="append")

    nearest = {"column": "vector", "q": query, "k": 4, "metric": "cosine"}
    flat = dataset.to_table(columns=["id"], nearest={**nearest, "use_index": False})
    mixed = dataset.to_table(columns=["id"], nearest=nearest)

    dataset.optimize.optimize_indices()
    fully_indexed = dataset.to_table(columns=["id"], nearest=nearest, fast_search=True)

    for result in [flat, mixed, fully_indexed]:
        assert result["id"].to_pylist() == [0, 2, 1, 3]
        np.testing.assert_allclose(result["_distance"].to_numpy(), [0.0, 0.0, 1.0, 2.0])


def test_pre_populated_ivf_centroids(dataset, tmp_path: Path):
    centroids = np.random.randn(5, 128).astype(np.float32)  # IVF5
    dataset_with_index = dataset.create_index(
        ["vector"],
        index_type="IVF_PQ",
        metric="cosine",
        ivf_centroids=centroids,
        num_partitions=5,
        num_sub_vectors=8,
    )

    q = np.random.randn(128)
    actual = dataset_with_index.to_table(
        columns=["id"],
        nearest={"column": "vector", "q": q, "k": 10, "use_index": False},
    )["id"].to_numpy()
    assert len(actual) == 10

    index_meta = dataset_with_index.describe_indices()[0]
    index_uuid = index_meta.segments[0].uuid
    assert len(index_uuid) == 36
    assert index_meta.segments[0].fragment_ids == {0}

    expected_filepath = str(tmp_path / "_indices" / index_uuid / "index.idx")
    if platform.system() == "Windows":
        expected_filepath = expected_filepath.replace("\\", "/")
    expected_statistics = {
        "index_type": "IVF_PQ",
        "uuid": index_uuid,
        "uri": expected_filepath,
        "metric_type": "cosine",
        "num_partitions": 5,
        "sub_index": {
            "dimension": 128,
            "index_type": "PQ",
            "metric_type": "l2",
            "nbits": 8,
            "num_sub_vectors": 8,
            "transposed": True,
        },
        "index_file_version": IndexFileVersion.V3,
    }

    with pytest.raises(KeyError, match='Index "non-existent_idx" not found'):
        # increase 1 miss of index_cache.metadata_cache
        assert dataset_with_index.stats.index_stats("non-existent_idx")
    with pytest.raises(KeyError, match='Index "" not found'):
        # increase 1 miss of index_cache.metadata_cache
        assert dataset_with_index.stats.index_stats("")
    with pytest.raises(TypeError):
        dataset_with_index.stats.index_stats()

    # increase 1 hit of index_cache.metadata_cache
    actual_statistics = dataset_with_index.stats.index_stats("vector_idx")
    assert actual_statistics["num_indexed_rows"] == 1000
    assert actual_statistics["num_unindexed_rows"] == 0

    idx_stats = actual_statistics["indices"][0]
    partitions = idx_stats.pop("partitions")
    idx_stats.pop("centroids")
    idx_stats.pop("loss")
    assert idx_stats == expected_statistics
    assert len(partitions) == 5
    partition_keys = {"size"}
    assert all([partition_keys == set(p.keys()) for p in partitions])


def test_create_ivf_pq_skip_transpose(dataset, tmp_path: Path):
    ds = lance.write_dataset(
        dataset.to_table(), tmp_path / "indexed_skip_transpose.lance"
    )
    ds = ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        skip_transpose=True,
    )

    stats = ds.stats.index_stats("vector_idx")
    assert stats["indices"][0]["sub_index"]["transposed"] is False


def test_optimize_index(dataset, tmp_path):
    dataset_uri = tmp_path / "dataset.lance"
    assert not dataset.has_index
    ds = lance.write_dataset(dataset.to_table(), dataset_uri)
    ds = ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=2,
    )

    assert ds.has_index

    # New data
    tbl = create_table(nvec=200)
    ds = lance.write_dataset(tbl, dataset_uri, mode="append")

    assert len(ds) == 1200
    assert ds.has_index

    indices_dir = dataset_uri / "_indices"
    assert len(list(indices_dir.iterdir())) == 1

    ds = ds.optimize.optimize_indices()
    assert len(list(indices_dir.iterdir())) == 2


def test_optimize_index_cosine(dataset, tmp_path):
    dataset_uri = tmp_path / "dataset.lance"
    assert not dataset.has_index
    ds = lance.write_dataset(dataset.to_table(), dataset_uri)
    ds = ds.create_index(
        "vector",
        metric="cosine",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=2,
    )

    assert len(ds) == 1000
    assert ds.has_index

    n_results_before_append = ds.to_table(
        nearest={
            "q": [0.1 for _ in range(128)],
            "column": "vector",
            "k": len(ds),
            "nprobes": 1,
        },
        fast_search=True,
    ).num_rows

    # New data
    tbl = create_table(nvec=200)
    ds = lance.write_dataset(tbl, dataset_uri, mode="append")

    assert len(ds) == 1200
    assert ds.has_index

    indices_dir = dataset_uri / "_indices"
    assert len(list(indices_dir.iterdir())) == 1

    # with fast search the index doesn't contain new data yet
    assert (
        ds.to_table(
            nearest={
                "q": [0.1 for _ in range(128)],
                "column": "vector",
                "k": len(ds),
                "nprobes": 1,
            },
            fast_search=True,
        ).num_rows
        == n_results_before_append
    )

    ds.optimize.optimize_indices()
    assert len(list(indices_dir.iterdir())) == 2

    ds = lance.dataset(dataset_uri)

    assert (
        ds.to_table(
            nearest={
                "q": [0.1 for _ in range(128)],
                "column": "vector",
                "k": len(ds),
                "nprobes": 1,
            },
            fast_search=True,
        ).num_rows
        > n_results_before_append
    )


def test_create_index_dot(dataset, tmp_path):
    dataset_uri = tmp_path / "dataset.lance"
    assert not dataset.has_index
    ds = lance.write_dataset(dataset.to_table(), dataset_uri)
    ds = ds.create_index(
        "vector",
        index_type="IVF_PQ",
        metric="dot",
        num_partitions=4,
        num_sub_vectors=2,
    )

    assert ds.has_index
    assert "dot" == ds.stats.index_stats("vector_idx")["indices"][0]["metric_type"]


def create_uniform_table(min, max, nvec, offset, ndim=8):
    mat = np.random.uniform(min, max, (nvec, ndim))

    tbl = vec_to_table(data=mat)
    tbl = pa.Table.from_pydict(
        {
            "vector": tbl.column(0).chunk(0),
            "filterable": np.arange(offset, offset + nvec),
        }
    )
    return tbl


def test_optimize_index_recall(tmp_path: Path):
    base_dir = tmp_path / "dataset"
    data = create_uniform_table(min=0, max=1, nvec=300, offset=0)

    dataset = lance.write_dataset(data, base_dir, max_rows_per_file=150)
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=2, num_sub_vectors=2
    )
    assert len(dataset.get_fragments()) == 2

    sample_indices = random.sample(range(300), 50)
    sample_query_indices = sample_indices[0:40]
    sample_delete_indices = sample_indices[40:]
    vecs = data.column("vector").chunk(0)
    sample_queries = [
        {"column": "vector", "q": vecs[i].values, "k": 5} for i in sample_query_indices
    ]
    sample_delete_queries = [
        {"column": "vector", "q": vecs[i].values, "k": 5} for i in sample_delete_indices
    ]

    def has_target(target, results):
        for item in results:
            if item.values == target:
                return True
        return False

    def check_index(has_knn_combined, delete_has_happened):
        for query in sample_queries:
            results = dataset.to_table(nearest=query)
            assert has_target(query["q"], results["vector"])
            plan = dataset.scanner(nearest=query).explain_plan(verbose=True)
            assert ("KNNVectorDistance" in plan) == has_knn_combined
        for query in sample_delete_queries:
            results = dataset.to_table(nearest=query).column("vector")
            assert delete_has_happened != has_target(query["q"], results)

    # Original state is 2 indexed fragments of size 150.  This should not require
    # a combined scan
    check_index(has_knn_combined=False, delete_has_happened=False)

    # Add a new fragment, now a combined scan is required
    extra_data = create_uniform_table(min=1000, max=1001, nvec=100, offset=300)
    dataset = lance.write_dataset(
        extra_data, base_dir, mode="append", max_rows_per_file=100
    )
    check_index(has_knn_combined=True, delete_has_happened=False)

    for row_id in sample_delete_indices:
        dataset.delete(f"filterable == {row_id}")

    # Delete some rows, combined KNN still needed
    check_index(has_knn_combined=True, delete_has_happened=True)

    # Optimize the index, combined KNN should no longer be needed
    dataset.optimize.optimize_indices()
    check_index(has_knn_combined=False, delete_has_happened=True)


def test_knn_with_deletions(tmp_path):
    dims = 5
    values = pa.array(
        [x for val in range(50) for x in [float(val)] * 5], type=pa.float32()
    )
    tbl = pa.Table.from_pydict(
        {
            "vector": pa.FixedSizeListArray.from_arrays(values, dims),
            "filterable": pa.array(range(50)),
        }
    )
    dataset = lance.write_dataset(tbl, tmp_path, max_rows_per_group=10)

    dataset.delete("not (filterable % 5 == 0)")

    # Do KNN with k=100, should return 10 vectors
    expected = [
        [0.0] * 5,
        [5.0] * 5,
        [10.0] * 5,
        [15.0] * 5,
        [20.0] * 5,
        [25.0] * 5,
        [30.0] * 5,
        [35.0] * 5,
        [40.0] * 5,
        [45.0] * 5,
    ]

    results = dataset.to_table(
        nearest={"column": "vector", "q": [0.0] * 5, "k": 100}
    ).column("vector")
    assert len(results) == 10

    assert expected == [r.as_py() for r in results]


def test_index_cache_size(tmp_path):
    rng = np.random.default_rng(seed=42)

    def query_index(ds, ntimes, q=None):
        ndim = ds.schema[0].type.list_size
        for _ in range(ntimes):
            ds.to_table(
                nearest={
                    "column": "vector",
                    "q": q if q is not None else rng.standard_normal(ndim),
                    "nprobes": 20,
                },
            )

    tbl = create_table(nvec=1024, ndim=16)
    dataset = lance.write_dataset(tbl, tmp_path / "test")

    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=128, num_sub_vectors=2
    )

    indexed_dataset = lance.dataset(tmp_path / "test", index_cache_size_bytes=0)
    # Zero size index cache means all queries should miss the cache
    assert np.isclose(indexed_dataset._ds.index_cache_hit_rate(), 0.0)
    query_index(indexed_dataset, 1)
    # index cache is size=0, there should be no hit
    assert np.isclose(indexed_dataset._ds.index_cache_hit_rate(), 0.0)

    indexed_dataset = lance.dataset(tmp_path / "test")
    # query using the same vector, we should get a very high hit rate
    # it isn't always exactly 199/200 perhaps because the stats counter
    # is a relaxed atomic counter and may lag behind the true value or perhaps
    # because the cache takes some time to get populated by background threads
    query_index(indexed_dataset, 200, q=rng.standard_normal(16))
    assert indexed_dataset._ds.index_cache_hit_rate() > 0.95

    last_hit_rate = indexed_dataset._ds.index_cache_hit_rate()

    # send a few queries with different vectors, the hit rate should drop
    query_index(indexed_dataset, 128)

    assert last_hit_rate > indexed_dataset._ds.index_cache_hit_rate()


def test_index_cache_size_bytes(tmp_path):
    """Test the new index_cache_size_bytes parameter."""
    rng = np.random.default_rng(seed=42)

    def query_index(ds, ntimes, q=None):
        ndim = ds.schema[0].type.list_size
        for _ in range(ntimes):
            ds.to_table(
                nearest={
                    "column": "vector",
                    "q": q if q is not None else rng.standard_normal(ndim),
                    "minimum_nprobes": 1,
                },
            )

    tbl = create_table(nvec=1024, ndim=16)
    dataset = lance.write_dataset(tbl, tmp_path / "test")

    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=128, num_sub_vectors=2
    )

    # Test with index_cache_size_bytes=0 (no cache)
    indexed_dataset = lance.dataset(tmp_path / "test", index_cache_size_bytes=0)
    assert np.isclose(indexed_dataset._ds.index_cache_hit_rate(), 0.0)
    query_index(indexed_dataset, 1)
    # No cache, so hit rate should be 0
    assert np.isclose(indexed_dataset._ds.index_cache_hit_rate(), 0.0)

    # Test with index_cache_size_bytes=20MB (1 entry equivalent)
    indexed_dataset = lance.dataset(
        tmp_path / "test", index_cache_size_bytes=20 * 1024 * 1024
    )
    # Query using the same vector, we should get a good hit rate
    query_index(indexed_dataset, 200, q=rng.standard_normal(16))
    assert indexed_dataset._ds.index_cache_hit_rate() > 0.8


def test_index_cache_size_deprecation(tmp_path):
    """Test that index_cache_size shows deprecation warning."""
    import warnings

    tbl = create_table(nvec=100, ndim=16)
    lance.write_dataset(tbl, tmp_path / "test")

    # Test deprecation warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # This should trigger the deprecation warning
        lance.dataset(tmp_path / "test", index_cache_size=256)

        # Check that a deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "index_cache_size" in str(w[0].message)
        assert "index_cache_size_bytes" in str(w[0].message)


def test_f16_index(tmp_path: Path):
    DIM = 32
    total = 256
    uri = tmp_path / "f16data.lance"
    rng = np.random.default_rng(42)
    f16_data = rng.uniform(0, 1, total * DIM).astype(np.float16)
    fsl = pa.FixedSizeListArray.from_arrays(f16_data, DIM)
    tbl = pa.Table.from_pydict({"vector": fsl})
    dataset = lance.write_dataset(tbl, uri)
    dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        max_iters=2,
    )

    q = rng.uniform(0, 1, DIM).astype(np.float16)
    rst = dataset.to_table(
        nearest={
            "column": "vector",
            "q": q,
            "k": 10,
        }
    )

    assert rst.schema.field("vector").type.value_type == pa.float16()
    assert len(rst) == 10


def test_vector_with_nans(tmp_path: Path):
    DIM = 32
    TOTAL = 320
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, TOTAL * DIM).astype(np.float32)

    # Put the 1st vector as NaN.
    np.put(data, range(DIM, 2 * DIM), np.nan)
    fsl = pa.FixedSizeListArray.from_arrays(data, DIM)
    tbl = pa.Table.from_pydict({"vector": fsl})

    dataset = lance.write_dataset(tbl, tmp_path)
    row = dataset._take_rows([1])
    assert row["vector"]

    ds = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        max_iters=2,
        replace=True,
    )
    tbl = ds.to_table(
        nearest={"column": "vector", "q": data[0:DIM], "k": TOTAL, "nprobes": 1},
        with_row_id=True,
    )
    assert len(tbl) == TOTAL - 1
    assert 1 not in tbl["_rowid"].to_numpy(), "Row with ID 1 is not in the index"


def test_validate_vector_index(tmp_path: Path):
    # make sure the sanity check is correctly catchting issues
    ds = lance.write_dataset(create_table(), tmp_path)
    validate_vector_index(ds, "vector", sample_size=100)

    called = False

    def direct_first_call_to_new_table(*args, **kwargs):
        nonlocal called
        if called:
            return ds.to_table(*args, **kwargs)
        called = True
        return create_table()

    # return a new random table so things fail
    ds.sample = direct_first_call_to_new_table
    with pytest.raises(ValueError, match="Vector index failed sanity check"):
        validate_vector_index(ds, "vector", sample_size=100)


def test_dynamic_projection_with_vectors_index(tmp_path: Path):
    ds = lance.write_dataset(create_table(), tmp_path)
    ds = ds.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )

    res = ds.to_table(
        nearest={
            "column": "vector",
            "q": np.random.randn(128),
        },
        columns={
            "vec": "vector",
            "vec_f16": "_cast_list_f16(vector)",
        },
    )

    # TODO: _distance shouldn't be returned by default
    assert res.column_names == ["vec", "vec_f16", "_distance"]

    original = np.stack(res["vec"].to_numpy())
    casted = np.stack(res["vec_f16"].to_numpy())

    assert (original.astype(np.float16) == casted).all()


def test_index_cast_centroids(tmp_path):
    torch = pytest.importorskip("torch")

    dim = 16
    rng = np.random.default_rng(42)
    # Torch one-pass PQ emits an 8-bit codebook, which requires 256 rows.
    tbl = vec_to_table(data=rng.standard_normal((256, dim), dtype=np.float32))

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=2,
        num_sub_vectors=4,
        max_iters=2,
        accelerator=torch.device("cpu"),
    )

    # Get the centroids
    index_name = dataset.describe_indices()[0].name
    index_stats = dataset.stats.index_stats(index_name)
    centroids = index_stats["indices"][0]["centroids"]
    values = pa.array([x for arr in centroids for x in arr], pa.float32())
    centroids = pa.FixedSizeListArray.from_arrays(values, dim)

    # Cast invalidates the attached index; drop it first per the new contract.
    dataset.drop_index(index_name)
    dataset.alter_columns(dict(path="vector", data_type=pa.list_(pa.float16(), dim)))

    # centroids are f32, but the column is now f16
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=2,
        num_sub_vectors=4,
        max_iters=2,
        accelerator=torch.device("cpu"),
        ivf_centroids=centroids,
    )


def test_fragment_scan_disallowed_on_ann(dataset):
    q = np.random.randn(128)
    with pytest.raises(
        ValueError, match="This operation is not supported for fragment scan"
    ):
        scanner = dataset.scanner(
            columns=["id"],
            nearest={
                "column": "vector",
                "q": q,
            },
            fragments=[LanceFragment(dataset, 0)],
        )
        scanner.explain_plan(True)


def test_fragment_scan_allowed_on_ann_with_file_scan_prefilter(dataset):
    q = np.random.randn(128)
    scanner = dataset.scanner(
        prefilter=True,
        filter="id>0",
        columns=["id"],
        nearest={
            "column": "vector",
            "q": q,
        },
        fragments=[LanceFragment(dataset, 0)],
    )
    scanner.explain_plan(True)


def test_fragment_scan_disallowed_on_ann_with_index_scan_prefilter(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path, max_rows_per_file=250)
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )
    dataset.create_scalar_index("id", index_type="BTREE")

    assert len(dataset.get_fragments()) == 4

    q = np.random.randn(128)
    results = dataset.scanner(
        prefilter=True,
        filter="id > 50",
        columns=["id"],
        nearest={"column": "vector", "q": q, "use_index": True},
        fragments=[dataset.get_fragment(1)],
    ).to_table()

    results_no_scalar_index = dataset.scanner(
        prefilter=True,
        filter="id > 50",
        columns=["id"],
        nearest={"column": "vector", "q": q, "use_index": True},
        fragments=[dataset.get_fragment(1)],
        use_scalar_index=False,
    ).to_table()

    assert results == results_no_scalar_index


def test_describe_indices(dataset):
    indices = dataset.describe_indices()
    assert len(indices) == 0

    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )
    indices = dataset.describe_indices()
    assert len(indices) == 1


def test_describe_vector_index(indexed_dataset: LanceDataset):
    info = indexed_dataset.describe_indices()[0]

    assert info.name == "vector_idx"
    assert info.type_url == "/lance.index.pb.VectorIndexDetails"
    assert info.index_type == "IVF_PQ"
    assert info.num_rows_indexed == 1000
    assert info.fields == [0]
    assert info.field_names == ["vector"]
    assert len(info.segments) == 1
    assert info.segments[0].fragment_ids == {0}
    assert info.segments[0].dataset_version_at_last_update == 1
    assert info.segments[0].index_version == 1
    assert info.segments[0].created_at is not None

    details = info.details
    assert details["metric_type"] == "L2"
    assert details["compression"]["type"] == "pq"
    assert details["compression"]["num_bits"] == 8
    assert details["compression"]["num_sub_vectors"] == 16


def test_describe_index_runtime_hints_stored(tmp_path):
    tbl = create_table(nvec=300, ndim=16)
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=4,
        max_iters=100,
        sample_rate=512,
    )
    details = dataset.describe_indices()[0].details
    hints = details.get("runtime_hints", {})
    assert hints.get("lance.ivf.max_iters") == "100"
    assert hints.get("lance.ivf.sample_rate") == "512"
    assert hints.get("lance.pq.max_iters") == "100"
    assert hints.get("lance.pq.sample_rate") == "512"


def test_optimize_indices(indexed_dataset):
    data = create_table()
    indexed_dataset = lance.write_dataset(data, indexed_dataset.uri, mode="append")
    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_indices"] == 1
    indexed_dataset.optimize.optimize_indices(num_indices_to_merge=0)
    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_indices"] == 2


@pytest.mark.parametrize("enable_stable_row_ids", [False, True])
def test_segment_ownership_filter_precedes_partition_topk(
    tmp_path, enable_stable_row_ids
):
    ndim = 4

    def table(ids, value):
        vectors = np.full((len(ids), ndim), value, dtype=np.float32)
        return pa.table(
            {
                "id": pa.array(ids, type=pa.int64()),
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(vectors.reshape(-1), type=pa.float32()), ndim
                ),
            }
        )

    dataset = lance.write_dataset(
        table(range(20), 1.0),
        tmp_path,
        mode="create",
        enable_stable_row_ids=enable_stable_row_ids,
    )
    dataset = lance.write_dataset(
        table(range(100, 120), 0.0), dataset.uri, mode="append"
    )
    dataset = dataset.create_index(
        "vector", index_type="IVF_FLAT", metric="l2", num_partitions=1
    )

    fragment = dataset.get_fragment(1)
    row_ids = fragment.to_table(columns=["id"], with_row_id=True)["_rowid"].to_pylist()
    update_data = pa.table(
        {
            "_rowid": pa.array(row_ids, type=pa.uint64()),
            "vector": pa.array(
                [[10.0] * ndim] * len(row_ids), type=pa.list_(pa.float32(), ndim)
            ),
        }
    )
    updated_fragment, fields_modified = fragment.update_columns(update_data)
    dataset = lance.LanceDataset.commit(
        dataset.uri,
        lance.LanceOperation.Update(
            updated_fragments=[updated_fragment], fields_modified=fields_modified
        ),
        read_version=dataset.version,
    )
    dataset.optimize.optimize_indices(num_indices_to_merge=0)
    dataset = lance.dataset(dataset.uri)

    def assert_current_nearest_rows():
        result = dataset.to_table(
            columns=["id"],
            nearest={
                "column": "vector",
                "q": np.zeros(ndim, dtype=np.float32),
                "k": 5,
            },
        )

        assert all(row_id < 20 for row_id in result["id"].to_pylist())
        assert result["_distance"].to_pylist() == pytest.approx([4.0] * 5)

    assert_current_nearest_rows()
    dataset.optimize.optimize_indices(num_indices_to_merge=2)
    dataset = lance.dataset(dataset.uri)
    assert_current_nearest_rows()


def test_no_stale_duplicate_after_partial_column_update(tmp_path):
    # Regression test: updating an indexed vector column in place (via the
    # low-level fragment.update_columns API + LanceOperation.Update) and then
    # delta-optimizing the index must not leave a stale copy of the row in the
    # original index segment.
    #
    # Mechanism: update_columns rewrites only the column data file, keeping the
    # fragment id and row address. Committing the Update prunes the fragment
    # from the old index segment's fragment_bitmap, but that segment's index
    # file still physically holds the row's OLD vector. optimize_indices then
    # builds a new delta segment with the NEW vector. Before the fix a KNN query
    # searched both segments and returned the updated row TWICE - once with the
    # stale vector (old segment) and once with the new value (delta segment).
    np.random.seed(42)
    ndim = 16

    # Fragment 0: a "far" cluster bounded to [-1, 1]. No bulk vector is close to
    # the query (all-10.8), so the bulk cannot crowd the stale copy out of top-k.
    n_bulk = 1000
    bulk = np.random.uniform(-1, 1, (n_bulk, ndim)).astype(np.float32)
    table0 = pa.table(
        {
            "id": pa.array(range(n_bulk), type=pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(bulk.reshape(-1), type=pa.float32()), list_size=ndim
            ),
        }
    )
    ds = lance.write_dataset(table0, tmp_path, mode="create")

    # Fragment 1: a single row whose ORIGINAL vector (all 2.0) is closer to the
    # query than any bulk vector, so its stale copy ranks well inside top-k.
    orig = np.full((1, ndim), 2.0, dtype=np.float32)
    table1 = pa.table(
        {
            "id": pa.array([10_000], type=pa.int64()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(orig.reshape(-1), type=pa.float32()), list_size=ndim
            ),
        }
    )
    ds = lance.write_dataset(table1, tmp_path, mode="append")
    assert len(ds.get_fragments()) == 2

    # One index segment covering BOTH fragments {0, 1}.
    ds = ds.create_index(
        "vector",
        index_type="IVF_PQ",
        metric="l2",
        num_partitions=1,
        num_sub_vectors=ndim,
    )

    # Overwrite fragment 1's vector in place and commit Update(fields_modified).
    new_vec = [10.8] * ndim
    frag = ds.get_fragment(1)
    rowids = frag.to_table(columns=["id"], with_row_id=True)["_rowid"].to_pylist()
    update_data = pa.table(
        {
            "_rowid": pa.array(rowids, type=pa.uint64()),
            "vector": pa.array(
                [new_vec] * len(rowids), type=pa.list_(pa.float32(), ndim)
            ),
        }
    )
    updated_fragment, fields_modified = frag.update_columns(update_data)
    op = lance.LanceOperation.Update(
        updated_fragments=[updated_fragment],
        fields_modified=fields_modified,
    )
    ds = lance.LanceDataset.commit(ds.uri, op, read_version=ds.version)

    # Delta-optimize: appends a new segment for the updated fragment; the old
    # segment is left intact, still physically holding the stale vector.
    ds.optimize.optimize_indices(num_indices_to_merge=0)
    ds = lance.dataset(ds.uri)
    assert ds.stats.index_stats("vector_idx")["num_indices"] == 2

    # KNN near the NEW value via the default vector search (searches all
    # segments). The updated row must appear EXACTLY ONCE.
    #
    # This pins the filtering only. With a single partition the late search
    # returns before the shared budget is consulted, so the accounting half of
    # the fix is pinned by the Rust unit test
    # `test_unowned_row_does_not_fill_the_shared_budget` instead.
    q = np.array(new_vec, dtype=np.float32)
    res = ds.to_table(
        columns=["id"],
        nearest={"column": "vector", "q": q, "k": 10},
        with_row_id=True,
    ).to_pandas()
    dupes = res[res["id"] == 10_000]
    assert len(dupes) == 1, (
        f"updated row id=10000 returned {len(dupes)} times "
        f"(stale index segment not masked); rowids={res['_rowid'].tolist()}"
    )
    # A mask that over-restricts would drop the old segment wholesale and still
    # satisfy the assertion above, so pin the full result set too.
    assert len(res) == 10, f"expected a full top-10, got {len(res)} rows"
    assert res["id"].is_unique, f"duplicate ids in result: {res['id'].tolist()}"


@pytest.mark.parametrize("retrain", [None, False, True])
def test_retrain_indices(tmp_path, retrain):
    rng = np.random.default_rng(42)
    ndim = 16
    initial_vectors = rng.standard_normal((64, ndim), dtype=np.float32)
    appended_vectors = rng.standard_normal((64, ndim), dtype=np.float32) + 100
    old_centroid = np.full((1, ndim), -1000, dtype=np.float32)

    indexed_dataset = lance.write_dataset(vec_to_table(initial_vectors), tmp_path)
    indexed_dataset = indexed_dataset.create_index(
        "vector",
        index_type="IVF_FLAT",
        num_partitions=1,
        ivf_centroids=old_centroid,
        index_file_version=IndexFileVersion.V3,
    )
    indexed_dataset = lance.write_dataset(
        vec_to_table(appended_vectors), indexed_dataset.uri, mode="append"
    )

    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_indices"] == 1

    indexed_dataset.optimize.optimize_indices(num_indices_to_merge=0)
    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_indices"] == 2
    assert all(
        index["centroids"] == old_centroid.tolist() for index in stats["indices"]
    )

    kwargs = {} if retrain is None else {"retrain": retrain}
    indexed_dataset.optimize.optimize_indices(**kwargs)
    stats = indexed_dataset.stats.index_stats("vector_idx")
    centroids = [index["centroids"] for index in stats["indices"]]
    if retrain:
        expected_centroid = np.concatenate([initial_vectors, appended_vectors]).mean(
            axis=0
        )
        assert stats["num_indices"] == 1
        assert np.allclose(centroids[0][0], expected_centroid)
    else:
        assert all(centroid == old_centroid.tolist() for centroid in centroids)


def test_no_include_deleted_rows(indexed_dataset):
    with pytest.raises(ValueError, match="Cannot include deleted rows"):
        indexed_dataset.to_table(
            nearest={
                "column": "vector",
                "q": np.random.randn(128),
                "k": 10,
            },
            with_row_id=True,
            include_deleted_rows=True,
        )


def test_drop_indices(indexed_dataset):
    idx_name = indexed_dataset.describe_indices()[0].name

    indexed_dataset.drop_index(idx_name)
    indices = indexed_dataset.describe_indices()
    assert len(indices) == 0

    test_vec = (
        indexed_dataset.take([0], columns=["vector"]).column("vector").to_pylist()[0]
    )

    # make sure we can still search the column (will do flat search)
    results = indexed_dataset.to_table(
        nearest={
            "column": "vector",
            "q": test_vec,
            "k": 15,
            "nprobes": 1,
        },
    )

    assert len(results) == 15


def test_read_partition(indexed_dataset):
    idx_name = indexed_dataset.describe_indices()[0].name
    reader = VectorIndexReader(indexed_dataset, idx_name)

    num_rows = indexed_dataset.count_rows()
    row_sum = 0
    for part_id in range(reader.num_partitions()):
        res = reader.read_partition(part_id)
        row_sum += res.num_rows
        assert "_rowid" in res.column_names
    assert row_sum == num_rows

    row_sum = 0
    for part_id in range(reader.num_partitions()):
        res = reader.read_partition(part_id, with_vector=True)
        row_sum += res.num_rows
        pq_column = res["__pq_code"]
        assert "_rowid" in res.column_names
        assert pq_column.type == pa.list_(pa.uint8(), 16)
    assert row_sum == num_rows

    # error tests
    with pytest.raises(IndexError, match="out of range"):
        reader.read_partition(reader.num_partitions() + 1)

    with pytest.raises(ValueError, match="not vector index"):
        indexed_dataset.create_scalar_index("id", index_type="BTREE")
        VectorIndexReader(indexed_dataset, "id_idx")


def test_read_partition_nested_vector_quoted_field(tmp_path):
    num_rows = 1024
    dimensions = 8
    rng = np.random.default_rng(42)
    values = rng.integers(0, 256, size=num_rows * dimensions, dtype=np.uint8)
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(values), dimensions)
    nested = pa.StructArray.from_arrays([vectors], names=["embedding.v1"])
    dataset = lance.write_dataset(pa.table({"data": nested}), tmp_path)
    # Match nested uint8 pHash indexes without introducing PQ training setup.
    dataset = dataset.create_index(
        "data.`embedding.v1`",
        index_type="IVF_FLAT",
        name="vector_idx",
        metric="hamming",
        num_partitions=4,
    )

    reader = VectorIndexReader(dataset, "vector_idx")
    for with_vector in (False, True):
        partitions = [
            reader.read_partition(partition_id, with_vector=with_vector)
            for partition_id in range(reader.num_partitions())
        ]

        assert all("_rowid" in partition.column_names for partition in partitions)
        assert sum(partition.num_rows for partition in partitions) == num_rows


def test_vector_index_with_prefilter_and_scalar_index(indexed_dataset):
    uri = indexed_dataset.uri
    new_table = create_table()
    ds = lance.write_dataset(new_table, uri, mode="append")
    ds.optimize.optimize_indices(num_indices_to_merge=0)
    ds.create_scalar_index("id", index_type="BTREE")

    raw_table = create_table()
    ds = lance.write_dataset(raw_table, uri, mode="append")
    ds.optimize.optimize_indices(num_indices_to_merge=0, index_names=["vector_idx"])

    res = ds.to_table(
        nearest={
            "column": "vector",
            "q": np.random.randn(128),
            "k": 10,
        },
        filter="id > 0",
        with_row_id=True,
        prefilter=True,
    )
    assert len(res) == 10


def test_vector_index_with_nprobes(indexed_dataset):
    res = indexed_dataset.scanner(
        nearest={
            "column": "vector",
            "q": np.random.randn(128),
            "k": 10,
            "nprobes": 7,
        }
    ).explain_plan()

    assert "minimum_nprobes=7" in res
    assert "maximum_nprobes=Some(7)" in res

    res = indexed_dataset.scanner(
        nearest={
            "column": "vector",
            "q": np.random.randn(128),
            "k": 10,
            "minimum_nprobes": 7,
        }
    ).explain_plan()

    assert "minimum_nprobes=7" in res
    assert "maximum_nprobes=None" in res

    res = indexed_dataset.scanner(
        nearest={
            "column": "vector",
            "q": np.random.randn(128),
            "k": 10,
            "minimum_nprobes": 7,
            "maximum_nprobes": 10,
        }
    ).explain_plan()

    assert "minimum_nprobes=7" in res
    assert "maximum_nprobes=Some(10)" in res

    res = indexed_dataset.scanner(
        nearest={
            "column": "vector",
            "q": np.random.randn(128),
            "k": 10,
            "maximum_nprobes": 30,
        }
    ).analyze_plan()


def test_vector_index_with_query_parallelism(indexed_dataset):
    q = np.random.randn(128)

    sequential = indexed_dataset.to_table(
        nearest={
            "column": "vector",
            "q": q,
            "k": 10,
            "query_parallelism": 0,
        }
    )
    parallel = indexed_dataset.to_table(
        nearest={
            "column": "vector",
            "q": q,
            "k": 10,
            "query_parallelism": -1,
        }
    )

    assert sequential == parallel


def test_vector_index_invalid_query_parallelism(indexed_dataset):
    with pytest.raises(ValueError, match="query_parallelism"):
        indexed_dataset.scanner(
            nearest={
                "column": "vector",
                "q": np.random.randn(128),
                "k": 10,
                "query_parallelism": -2,
            }
        )


def test_vector_index_with_approx_mode(indexed_dataset):
    q = np.random.randn(128)

    for approx_mode in ["fast", "normal", "accurate"]:
        result = indexed_dataset.to_table(
            nearest={
                "column": "vector",
                "q": q,
                "k": 10,
                "approx_mode": approx_mode,
            }
        )
        assert len(result) == 10


def test_vector_index_invalid_approx_mode(indexed_dataset):
    with pytest.raises(ValueError, match="approx_mode"):
        indexed_dataset.scanner(
            nearest={
                "column": "vector",
                "q": np.random.randn(128),
                "k": 10,
                "approx_mode": "hacc",
            }
        )


def test_knn_deleted_rows(tmp_path):
    data = create_table()
    ds = lance.write_dataset(data, tmp_path)
    ds.create_index(
        "vector",
        index_type="IVF_PQ",
        metric="cosine",
        num_partitions=4,
        num_sub_vectors=4,
    )
    ds.insert(create_table())

    ds.delete("id = 0")
    assert ds.count_rows() == data.num_rows * 2 - 2
    results = ds.to_table(
        nearest={"column": "vector", "q": data["vector"][0], "k": ds.count_rows()}
    )
    assert 0 not in results["id"]
    assert results.num_rows == ds.count_rows()


def test_nested_field_vector_index(tmp_path):
    """Test vector index creation and querying on nested fields

    Note: While scalar indices work on nested fields, vector indices currently
    have a limitation in the DataFusion integration layer that prevents them
    from working with nested field paths. The Python validation layer now
    correctly handles nested paths, but the Rust planner needs additional work.
    """
    # Create a dataset with nested vector field
    dimensions = 128
    num_rows = 256

    # Generate random vectors
    vectors = np.random.randn(num_rows, dimensions).astype(np.float32)
    vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.flatten()), dimensions
    )

    # Create nested structure with vector field
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field(
                "data",
                pa.struct(
                    [
                        pa.field("embedding", pa.list_(pa.float32(), dimensions)),
                        pa.field("label", pa.string()),
                    ]
                ),
            ),
        ]
    )

    # Create struct array
    struct_array = pa.StructArray.from_arrays(
        [vector_array, pa.array([f"label_{i}" for i in range(num_rows)])],
        names=["embedding", "label"],
    )

    data = pa.table({"id": list(range(num_rows)), "data": struct_array}, schema=schema)

    # Create dataset
    uri = tmp_path / "test_nested_vector"
    dataset = lance.write_dataset(data, uri)

    # Verify the schema
    assert "data" in dataset.schema.names
    field = dataset.schema.field("data")
    assert pa.types.is_struct(field.type)

    # Create vector index on nested column
    dataset = dataset.create_index(
        column="data.embedding",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
    )

    # Verify index was created
    indices = dataset.describe_indices()
    assert len(indices) == 1
    assert indices[0].field_names == ["data.embedding"]

    reader = VectorIndexReader(dataset, indices[0].name)
    for with_vector in (False, True):
        partition_rows = sum(
            reader.read_partition(partition_id, with_vector=with_vector).num_rows
            for partition_id in range(reader.num_partitions())
        )
        assert partition_rows == num_rows

    # Test querying with the index
    query_vec = vectors[0]
    result = dataset.to_table(
        nearest={"column": "data.embedding", "q": query_vec, "k": 10, "nprobes": 2}
    )

    # Verify results
    assert len(result) == 10
    assert "data" in result.column_names
    assert "_distance" in result.column_names

    # The first result should be the query vector itself (or very close)
    assert result["id"][0].as_py() == 0
    assert result["_distance"][0].as_py() < 0.01  # Should be nearly zero

    # Write additional data to the dataset
    new_vectors = np.random.randn(50, dimensions).astype(np.float32)
    new_vector_array = pa.FixedSizeListArray.from_arrays(
        pa.array(new_vectors.flatten()), dimensions
    )

    new_struct_array = pa.StructArray.from_arrays(
        [new_vector_array, pa.array([f"new_label_{i}" for i in range(50)])],
        names=["embedding", "label"],
    )

    new_data = pa.table(
        {"id": list(range(num_rows, num_rows + 50)), "data": new_struct_array},
        schema=schema,
    )

    dataset = lance.write_dataset(new_data, uri, mode="append")

    # Verify query still works after appending data
    result = dataset.to_table(
        nearest={"column": "data.embedding", "q": query_vec, "k": 15, "nprobes": 2}
    )

    assert len(result) == 15
    assert "data" in result.column_names

    # Optimize the index to include new data
    dataset.optimize.optimize_indices()

    # Verify query works after optimization
    result = dataset.to_table(
        nearest={"column": "data.embedding", "q": query_vec, "k": 20, "nprobes": 2}
    )

    assert len(result) == 20

    # Test with cosine metric
    dataset = dataset.create_index(
        column="data.embedding",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        metric="cosine",
        replace=True,
    )

    result = dataset.to_table(
        nearest={"column": "data.embedding", "q": query_vec, "k": 10, "nprobes": 2}
    )

    assert len(result) == 10

    # Verify total row count
    assert dataset.count_rows() == num_rows + 50


def test_prewarm_index(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path, data_storage_version="2.1")
    dataset = dataset.create_index(
        "vector",
        name="vector_index",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
    )
    # Prewarm the index
    dataset.prewarm_index("vector_index")

    new_data = create_table(nvec=10)
    dataset = lance.write_dataset(new_data, dataset.uri, mode="append")
    q = new_data["vector"][0].as_py()

    def func(rs: pa.Table):
        if "vector" not in rs:
            return
        assert rs["vector"][0].as_py() == q

    run(dataset, q=np.array(q), assert_func=func)


def test_scanner_rejects_unknown_index_segments(tmp_path):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index("vector", index_type="IVF_FLAT", num_partitions=4)

    with pytest.raises(
        ValueError, match="with_index_segments referenced unknown index segments"
    ):
        dataset.scanner(
            nearest={
                "column": "vector",
                "q": np.random.randn(128).astype(np.float32),
                "k": 10,
            },
            index_segments=[uuid.uuid4()],
        ).to_table()


def test_vector_index_distance_range(tmp_path):
    """Ensure vector index honors distance_range."""
    ndim = 128
    rng = np.random.default_rng(seed=42)
    base = rng.standard_normal((509, ndim)).astype(np.float32)
    zero_vec = np.zeros((1, ndim), dtype=np.float32)
    near_vec = np.full((1, ndim), 0.01, dtype=np.float32)
    far_vec = np.full((1, ndim), 500.0, dtype=np.float32)
    matrix = np.concatenate([zero_vec, near_vec, far_vec, base], axis=0)
    tbl = vec_to_table(data=matrix).append_column(
        "id", pa.array(np.arange(matrix.shape[0], dtype=np.int64))
    )
    dataset = lance.write_dataset(tbl, tmp_path / "vrange")
    indexed = dataset.create_index("vector", index_type="IVF_FLAT", num_partitions=4)

    q = zero_vec[0]
    distance_range = (0.0, 0.5)
    nprobes_all = 4

    # Brute force baseline (exact):
    # get full distance distribution and build expected in-range ids.
    all_results = indexed.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": q,
            "k": matrix.shape[0],
            "use_index": False,
        },
    )
    all_distances = all_results["_distance"].to_numpy()
    assert len(all_distances) == matrix.shape[0]
    assert all_distances.min() == 0.0
    assert (
        all_distances.max() > distance_range[1]
    )  # ensure some values are out of range

    in_range_mask = (all_distances >= distance_range[0]) & (
        all_distances < distance_range[1]
    )
    expected_ids = set(all_results["id"].to_numpy()[in_range_mask].tolist())
    assert len(expected_ids) > 0

    # Compare distance_range results:
    # brute-force vs index path should match exactly for IVF_FLAT
    brute_results = indexed.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": q,
            "k": matrix.shape[0],
            "distance_range": distance_range,
            "use_index": False,
        },
    )

    index_results = indexed.to_table(
        columns=["id"],
        nearest={
            "column": "vector",
            "q": q,
            "k": matrix.shape[0],
            "distance_range": distance_range,
            "nprobes": nprobes_all,
        },
    )

    brute_ids = brute_results["id"].to_numpy()
    index_ids = index_results["id"].to_numpy()
    brute_distances = brute_results["_distance"].to_numpy()
    index_distances = index_results["_distance"].to_numpy()

    assert set(brute_ids.tolist()).issubset(expected_ids)
    assert set(index_ids.tolist()).issubset(expected_ids)
    assert len(brute_ids) == len(index_ids)
    assert np.array_equal(brute_ids, index_ids)
    assert np.all(brute_distances >= distance_range[0]) and np.all(
        brute_distances < distance_range[1]
    )
    assert np.all(index_distances >= distance_range[0]) and np.all(
        index_distances < distance_range[1]
    )
    assert np.allclose(brute_distances, index_distances, rtol=1e-5, atol=0.0)


# =============================================================================
# Distributed vector index consistency helper
# =============================================================================


def _split_fragments_evenly(fragment_ids, world):
    """Split fragment_ids into `world` contiguous groups for distributed build.

    This keeps groups balanced and deterministic.
    """
    if world <= 0:
        raise ValueError(f"world must be >= 1, got {world}")
    n = len(fragment_ids)
    if n == 0:
        return [[] for _ in range(world)]
    world = min(world, n)
    group_size = n // world
    remainder = n % world
    groups = []
    start = 0
    for rank in range(world):
        extra = 1 if rank < remainder else 0
        end = start + group_size + extra
        groups.append(fragment_ids[start:end])
        start = end
    return groups


def build_distributed_vector_index(
    dataset,
    column,
    *,
    index_type="IVF_PQ",
    num_partitions=None,
    num_sub_vectors=None,
    world=2,
    **index_params,
):
    """Build a distributed vector index over fragment groups and commit."""

    frags = dataset.get_fragments()
    frag_ids = [f.fragment_id for f in frags]
    groups = _split_fragments_evenly(frag_ids, world)
    segments = []

    for g in groups:
        if not g:
            continue
        segments.append(
            dataset.create_index_uncommitted(
                column=column,
                index_type=index_type,
                fragment_ids=g,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
                **index_params,
            )
        )

    return dataset.commit_existing_index_segments(f"{column}_idx", column, segments)


def _commit_segments_helper(
    ds, segments, column: str, index_name: Optional[str] = None
):
    if index_name is None:
        index_name = f"{column}_idx"
    return ds.commit_existing_index_segments(index_name, column, segments)


def _build_segments(
    ds,
    column: str,
    index_type: str,
    fragment_groups,
    *,
    index_name: Optional[str] = None,
    **index_kwargs,
):
    if index_name is None:
        index_name = f"{column}_idx"

    segments = []
    for group in fragment_groups:
        if not group:
            continue
        segments.append(
            ds.create_index_uncommitted(
                column=column,
                index_type=index_type,
                name=index_name,
                fragment_ids=group,
                **index_kwargs,
            )
        )
    return segments


def assert_distributed_vector_consistency(
    data,
    column,
    *,
    index_type="IVF_PQ",
    index_params=None,
    queries=None,
    topk=10,
    world=2,
    tmp_path=None,
    similarity_metric="strict",
    similarity_threshold=1.0,
):
    """Recall-only consistency check between single-machine and distributed indices.

    This helper keeps the original signature for compatibility but ignores
    similarity_metric. It compares recall@K against a ground truth computed via
    exact search (use_index=False), requires both indices to reach at least 0.5
    recall, and bounds their recall difference with similarity_threshold.

    Steps
    -----
    1) Write `data` to two URIs (single, distributed); ensure distributed has >=2
       fragments (rewrite with max_rows_per_file if needed)
    2) Build a single-machine index via `create_index`
    3) Global training (IVF/PQ) using `IndicesBuilder.prepare_global_ivf_pq` when
       appropriate; for IVF_FLAT/SQ variants, train IVF centroids via
       `IndicesBuilder.train_ivf`
    4) Build the distributed index via
       `lance.indices.builder.build_distributed_vector_index`, passing the
       preprocessed artifacts
    5) For each query, compute ground-truth TopK IDs using exact search
       (use_index=False), then compute TopK using single index and the distributed
       index with consistent nearest settings (refine_factor=100; IVF probes all
       fixture partitions)
    6) Compute recall for single and distributed, require each to be >= 0.5,
       and bound their absolute difference with similarity_threshold.
    """
    # Keep signature compatibility but ignore the superseded metric selector.
    _ = similarity_metric

    index_params = index_params or {}

    # Create two datasets: single-machine and distributed builds
    tmp_dir = None
    if tmp_path is not None:
        base = str(tmp_path)
        single_uri = os.path.join(base, "vector_single")
        dist_uri = os.path.join(base, "vector_distributed")
    else:
        tmp_dir = tempfile.mkdtemp(prefix="lance_vec_consistency_")
        base = tmp_dir
        single_uri = os.path.join(base, "vector_single")
        dist_uri = os.path.join(base, "vector_distributed")

    single_ds = lance.write_dataset(data, single_uri)
    dist_ds = lance.write_dataset(data, dist_uri)

    # Ensure distributed dataset has ≥2 fragments by rewriting with small files
    if len(dist_ds.get_fragments()) < 2:
        dist_ds = lance.write_dataset(
            data, dist_uri, mode="overwrite", max_rows_per_file=500
        )

    num_rows = single_ds.count_rows()
    nparts = index_params.get("num_partitions", None)
    is_pq = index_type in {"IVF_PQ", "IVF_HNSW_PQ"}
    # Eight-bit PQ needs at least 256 centroids and sample_rate >= 2.
    sample_rate = 2 if is_pq else min(8, num_rows // max(1, nparts or 1))
    max_iters = index_params.get("max_iters", 5)
    build_params = dict(index_params)
    build_params.setdefault("sample_rate", sample_rate)
    build_params.setdefault("max_iters", max_iters)

    # Build single-machine index
    single_ds = single_ds.create_index(
        column=column,
        index_type=index_type,
        **build_params,
    )

    # Global training / preparation for distributed build
    preprocessed = None
    builder = IndicesBuilder(single_ds, column)
    nsub = index_params.get("num_sub_vectors", None)
    dist_type = index_params.get("metric", "l2")

    if is_pq:
        assert num_rows >= 512, "8-bit PQ training requires at least 512 rows"
        preprocessed = builder.prepare_global_ivf_pq(
            nparts,
            nsub,
            distance_type=dist_type,
            sample_rate=sample_rate,
            max_iters=max_iters,
        )
    elif (
        ("IVF_FLAT" in index_type)
        or ("IVF_SQ" in index_type)
        or ("IVF_HNSW_FLAT" in index_type)
    ):
        ivf_model = builder.train_ivf(
            nparts,
            distance_type=dist_type,
            sample_rate=sample_rate,
            max_iters=max_iters,
        )
        preprocessed = {"ivf_centroids": ivf_model.centroids}

    # Distributed build + merge
    extra = {
        k: v
        for k, v in index_params.items()
        if k not in {"num_partitions", "num_sub_vectors"}
    }
    if preprocessed is not None:
        if (
            "ivf_centroids" in preprocessed
            and preprocessed["ivf_centroids"] is not None
        ):
            extra["ivf_centroids"] = preprocessed["ivf_centroids"]
        if "pq_codebook" in preprocessed and preprocessed["pq_codebook"] is not None:
            extra["pq_codebook"] = preprocessed["pq_codebook"]

    dist_ds = build_distributed_vector_index(
        dist_ds,
        column,
        index_type=index_type,
        num_partitions=index_params.get("num_partitions", None),
        num_sub_vectors=index_params.get("num_sub_vectors", None),
        world=world,
        **extra,
    )

    # Normalize queries into a list of np.ndarray
    dim = single_ds.schema.field(column).type.list_size
    if queries is None:
        queries = [np.random.randn(dim).astype(np.float32)]
    elif isinstance(queries, np.ndarray) and queries.ndim == 1:
        queries = [queries.astype(np.float32)]
    else:
        queries = [np.asarray(q, dtype=np.float32) for q in queries]

    # Collect TopK id lists for ground truth, single, and distributed
    gt_ids = []
    single_ids = []
    dist_ids = []

    for q in queries:
        # Ground truth via exact search
        gt_tbl = single_ds.to_table(
            nearest={"column": column, "q": q, "k": topk, "use_index": False},
            columns=["id"],
        )
        gt_ids.append(np.array(gt_tbl["id"].to_pylist(), dtype=np.int64))

        # Consistent nearest settings for index-based search
        nearest = {"column": column, "q": q, "k": topk, "refine_factor": 100}
        if "IVF" in index_type:
            nearest["nprobes"] = int(index_params.get("num_partitions", 4))
        if "HNSW" in index_type:
            # Ensure ef is large enough even when refine_factor multiplies k for HNSW
            effective_k = topk * int(
                nearest["refine_factor"]
            )  # HNSW uses k * refine_factor
            nearest["ef"] = max(effective_k, 256)

        s_tbl = single_ds.to_table(nearest=nearest, columns=["id"])  # single index
        d_tbl = dist_ds.to_table(nearest=nearest, columns=["id"])  # distributed index
        single_ids.append(np.array(s_tbl["id"].to_pylist(), dtype=np.int64))
        dist_ids.append(np.array(d_tbl["id"].to_pylist(), dtype=np.int64))

    gt_ids = np.array(gt_ids, dtype=object)
    single_ids = np.array(single_ids, dtype=object)
    dist_ids = np.array(dist_ids, dtype=object)

    # User-specified recall computation
    def compute_recall(gt: np.ndarray, result: np.ndarray) -> float:
        recalls = [
            np.isin(rst, gt_vector).sum() / rst.shape[0]
            for (rst, gt_vector) in zip(result, gt)
        ]
        return np.mean(recalls)

    rs = compute_recall(gt_ids, single_ids)
    rd = compute_recall(gt_ids, dist_ids)

    assert rs >= 0.5, (
        f"Single-machine {index_type} recall below 0.5: recall={rs:.3f}, "
        f"num_partitions={nparts}, topk={topk}, queries={len(queries)}"
    )
    assert rd >= 0.5, (
        f"Distributed {index_type} recall below 0.5: recall={rd:.3f}, "
        f"num_partitions={nparts}, topk={topk}, queries={len(queries)}"
    )
    max_recall_difference = 1 - similarity_threshold
    assert abs(rs - rd) <= max_recall_difference, (
        f"Recall difference too large: single={rs:.3f}, distributed={rd:.3f}, "
        f"diff={abs(rs - rd):.3f} (> {max_recall_difference:.3f})"
    )

    # Cleanup temporary directory if used
    if tmp_dir is not None:
        try:
            shutil.rmtree(tmp_dir)
        except Exception as e:
            logging.exception("Failed to remove temporary directory %s: %s", tmp_dir, e)


def _make_sample_dataset_base(
    tmp_path: Path,
    name: str,
    n_rows: int = 1000,
    dim: int = 128,
    max_rows_per_file: int = 500,
):
    """Common helper to construct sample datasets for distributed index tests."""
    mat = np.random.default_rng(42).random((n_rows, dim), dtype=np.float32)
    ids = np.arange(n_rows)
    arr = pa.FixedSizeListArray.from_arrays(pa.array(mat.reshape(-1)), dim)
    tbl = pa.table({"id": ids, "vector": arr})
    return lance.write_dataset(
        tbl, tmp_path / name, max_rows_per_file=max_rows_per_file
    )


@pytest.mark.parametrize(
    "index_type",
    [
        "IVF_FLAT",
        "IVF_PQ",
        "IVF_SQ",
    ],
)
def test_distributed_ivf_two_shard_build_merge_and_search(tmp_path, index_type):
    dim = 32
    num_partitions = 2
    ds = _make_sample_dataset_base(
        tmp_path,
        f"dist_{index_type.lower()}",
        n_rows=640,
        dim=dim,
        max_rows_per_file=320,
    )
    frags = ds.get_fragments()
    assert len(frags) == 2
    fragment_groups = [[fragment.fragment_id] for fragment in frags]
    builder = IndicesBuilder(ds, "vector")
    build_kwargs = {"num_partitions": num_partitions}
    if index_type == "IVF_PQ":
        preprocessed = builder.prepare_global_ivf_pq(
            num_partitions=num_partitions,
            num_subvectors=4,
            distance_type="l2",
            sample_rate=2,
            max_iters=2,
        )
        assert set(preprocessed) == {"ivf_centroids", "pq_codebook"}
        assert len(preprocessed["ivf_centroids"]) == num_partitions
        assert preprocessed["ivf_centroids"].type.list_size == dim
        assert len(preprocessed["pq_codebook"]) > 0
        assert preprocessed["pq_codebook"].type.list_size == dim
        build_kwargs.update(
            num_sub_vectors=4,
            ivf_centroids=preprocessed["ivf_centroids"],
            pq_codebook=preprocessed["pq_codebook"],
        )
    else:
        ivf_model = builder.train_ivf(
            num_partitions=num_partitions,
            distance_type="l2",
            sample_rate=8,
            max_iters=2,
        )
        build_kwargs["ivf_centroids"] = ivf_model.centroids

    segments = _build_segments(
        ds,
        "vector",
        index_type,
        fragment_groups,
        index_name="vector_idx",
        **build_kwargs,
    )
    assert len(segments) == 2
    ds = _commit_segments_helper(ds, segments, column="vector")

    stats = ds.stats.index_stats("vector_idx")
    assert stats["index_type"] == index_type
    q = np.random.default_rng(43).random(dim, dtype=np.float32)
    results = ds.to_table(
        nearest={
            "column": "vector",
            "q": q,
            "k": 5,
            "nprobes": num_partitions,
            "refine_factor": 10,
        }
    )
    assert 0 < len(results) <= 5


def test_commit_existing_index_segments_accepts_index_metadata(tmp_path):
    ds = _make_sample_dataset_base(
        tmp_path, "legacy_metadata_commit", n_rows=512, dim=32, max_rows_per_file=256
    )
    frags = ds.get_fragments()
    assert len(frags) == 2

    ivf_model = IndicesBuilder(ds, "vector").train_ivf(
        num_partitions=2,
        distance_type="l2",
        sample_rate=8,
        max_iters=2,
    )
    base_kwargs = {
        "column": "vector",
        "index_type": "IVF_FLAT",
        "num_partitions": 2,
        "ivf_centroids": ivf_model.centroids,
    }
    first = ds.create_index_uncommitted(
        **base_kwargs,
        fragment_ids=[frags[0].fragment_id],
    )
    second = ds.create_index_uncommitted(
        **base_kwargs,
        fragment_ids=[frags[1].fragment_id],
    )

    merged = ds.merge_existing_index_segments([first, second])
    ds = ds.commit_existing_index_segments("vector_idx", "vector", [merged])

    q = np.random.rand(32).astype(np.float32)
    results = ds.to_table(nearest={"column": "vector", "q": q, "k": 5})
    assert 0 < len(results) <= 5


def test_distributed_ivf_rq_shared_rotation(tmp_path):
    """Two IVF_RQ segments built on separate fragments with one shared RaBitQ rotation
    merge into a single committed, queryable index. The shared ``rabitq_model`` (from
    ``lance.lance.indices.build_rq_model``) is what makes the independently built
    segments mergeable."""
    from lance.lance import indices

    dim = 32
    ds = _make_sample_dataset_base(
        tmp_path, "dist_rq_merge", n_rows=512, dim=dim, max_rows_per_file=256
    )
    frags = ds.get_fragments()
    assert len(frags) == 2

    ivf_model = IndicesBuilder(ds, "vector").train_ivf(
        num_partitions=2,
        distance_type="l2",
        sample_rate=8,
        max_iters=2,
    )
    rabitq_model = indices.build_rq_model(dimension=dim, num_bits=1)
    base_kwargs = {
        "column": "vector",
        "index_type": "IVF_RQ",
        "num_partitions": 2,
        "num_bits": 1,
        "ivf_centroids": ivf_model.centroids,
        "rabitq_model": rabitq_model,
    }
    first = ds.create_index_uncommitted(
        **base_kwargs,
        fragment_ids=[frags[0].fragment_id],
    )
    second = ds.create_index_uncommitted(
        **base_kwargs,
        fragment_ids=[frags[1].fragment_id],
    )

    merged = ds.merge_existing_index_segments([first, second])
    ds = ds.commit_existing_index_segments("vector_idx", "vector", [merged])

    q = np.random.rand(dim).astype(np.float32)
    results = ds.to_table(nearest={"column": "vector", "q": q, "k": 5})
    assert 0 < len(results) <= 5


def test_commit_existing_index_segments_accepts_uncommitted_vector_segments(tmp_path):
    dim = 32
    ds = _make_sample_dataset_base(
        tmp_path,
        "segment_commit_ds",
        n_rows=512,
        dim=dim,
        max_rows_per_file=256,
    )
    frags = ds.get_fragments()
    assert len(frags) == 2
    ivf_model = IndicesBuilder(ds, "vector").train_ivf(
        num_partitions=2,
        distance_type="l2",
        sample_rate=8,
        max_iters=2,
    )

    segments = [
        ds.create_index_uncommitted(
            "vector",
            "IVF_FLAT",
            name="vector_idx",
            train=True,
            fragment_ids=[fragment.fragment_id],
            num_partitions=2,
            ivf_centroids=ivf_model.centroids,
        )
        for fragment in frags
    ]

    assert len(segments) == 2
    ds = ds.commit_existing_index_segments("vector_idx", "vector", segments)

    q = np.random.rand(dim).astype(np.float32)
    results = ds.to_table(nearest={"column": "vector", "q": q, "k": 5})
    assert 0 < len(results) <= 5


def test_distributed_ivf_pq_order_invariance(tmp_path: Path):
    """Ensure distributed IVF_PQ build is invariant to shard build order."""
    dim = 32
    ds = _make_sample_dataset_base(
        tmp_path, "dist_ds", n_rows=640, dim=dim, max_rows_per_file=320
    )

    # Global IVF+PQ training once; artifacts are reused across shard orders.
    builder = IndicesBuilder(ds, "vector")
    pre = builder.prepare_global_ivf_pq(
        num_partitions=2,
        num_subvectors=4,
        distance_type="l2",
        sample_rate=2,
        max_iters=2,
    )

    # Copy the dataset twice so index manifests do not clash and we can vary
    # the shard build order independently on identical data.
    ds_order_12 = lance.write_dataset(
        ds.to_table(), tmp_path / "pq_order_node1_node2", max_rows_per_file=320
    )
    ds_order_21 = lance.write_dataset(
        ds.to_table(), tmp_path / "pq_order_node2_node1", max_rows_per_file=320
    )

    # For each copy, derive two shard groups from its own fragments.
    frags_12 = ds_order_12.get_fragments()
    assert len(frags_12) == 2
    mid_12 = len(frags_12) // 2
    node1_12 = [f.fragment_id for f in frags_12[:mid_12]]
    node2_12 = [f.fragment_id for f in frags_12[mid_12:]]
    assert node1_12 and node2_12

    frags_21 = ds_order_21.get_fragments()
    assert len(frags_21) == 2
    mid_21 = len(frags_21) // 2
    node1_21 = [f.fragment_id for f in frags_21[:mid_21]]
    node2_21 = [f.fragment_id for f in frags_21[mid_21:]]
    assert node1_21 and node2_21

    def build_distributed_ivf_pq(ds_copy, shard_order):
        try:
            segments = _build_segments(
                ds_copy,
                "vector",
                "IVF_PQ",
                shard_order,
                index_name="vector_idx",
                num_partitions=2,
                num_sub_vectors=4,
                ivf_centroids=pre["ivf_centroids"],
                pq_codebook=pre["pq_codebook"],
            )
            return _commit_segments_helper(ds_copy, segments, column="vector")
        except ValueError as e:
            raise e

    ds_12 = build_distributed_ivf_pq(ds_order_12, [node1_12, node2_12])
    ds_21 = build_distributed_ivf_pq(ds_order_21, [node2_21, node1_21])

    # Sample queries once from the original dataset and reuse for both index builds
    # to check order invariance under distributed PQ training and merging.
    k = 5
    queries = np.random.default_rng(43).random((3, dim), dtype=np.float32)

    def collect_ids_and_distances(ds_with_index):
        ids_per_query = []
        dists_per_query = []
        for q in queries:
            tbl = ds_with_index.to_table(
                columns=["id", "_distance"],
                nearest={
                    "column": "vector",
                    "q": q,
                    "k": k,
                    "nprobes": 2,
                    "refine_factor": 10,
                },
            )
            ids_per_query.append([int(x) for x in tbl["id"].to_pylist()])
            dists_per_query.append(tbl["_distance"].to_numpy())
        return ids_per_query, dists_per_query

    ids_12, dists_12 = collect_ids_and_distances(ds_12)
    ids_21, dists_21 = collect_ids_and_distances(ds_21)

    # TopK ids must match exactly and distances must be numerically stable across
    # different shard build orders (allow tiny floating error).
    assert ids_12 == ids_21
    for a, b in zip(dists_12, dists_21):
        assert np.allclose(a, b, atol=1e-6)


def test_fts_filter_vector_search(tmp_path):
    # Create dataset with vector and text columns
    ids = list(range(1, 301))
    vectors = [[float(i)] * 4 for i in ids]

    # Create text data:
    #   "text <i>" for ids 1-255, 299, 300,
    #   "noop <i>" for 256-298,
    texts = []
    for i in ids:
        if i <= 255:
            texts.append(f"text {i}")
        elif i <= 298:
            texts.append(f"noop {i}")
        else:
            texts.append(f"text {i}")

    categories = []
    for i in ids:
        if i % 3 == 1:
            categories.append("literature")
        elif i % 3 == 2:
            categories.append("science")
        else:
            categories.append("geography")

    table = pa.table(
        {
            "id": ids,
            "vector": pa.array(vectors, type=pa.list_(pa.float32(), 4)),
            "text": texts,
            "category": categories,
        }
    )

    # Write dataset and create indices
    dataset = lance.write_dataset(table, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=2,
        num_sub_vectors=4,
    )
    dataset.create_scalar_index("text", index_type="INVERTED", with_position=True)

    query_vector = [300.0, 300.0, 300.0, 300.0]

    # Case 1: search with prefilter=true, query_filter=match("text")
    scanner = dataset.scanner(
        filter=MatchQuery("text", "text"),
        nearest={"column": "vector", "q": query_vector, "k": 5},
        prefilter=True,
    )

    result = scanner.to_table()
    ids_result = result["id"].to_pylist()
    assert [300, 299, 255, 254, 253] == ids_result

    # Case 2: search with prefilter=true, search_filter=match("text"),
    #         filter="category='geography'"
    scanner = dataset.scanner(
        nearest={"column": "vector", "q": query_vector, "k": 5},
        prefilter=True,
        filter={
            "expr_filter": "category='geography'",
            "search_filter": MatchQuery("text", "text"),
        },
    )

    result = scanner.to_table()
    ids_result = result["id"].to_pylist()
    assert [300, 255, 252, 249, 246] == ids_result

    # Case 3: search with prefilter=false, search_filter=match("text")
    scanner = dataset.scanner(
        filter=MatchQuery("text", "text"),
        nearest={"column": "vector", "q": query_vector, "k": 5},
        prefilter=False,
    )

    result = scanner.to_table()
    ids_result = result["id"].to_pylist()
    assert [300, 299] == ids_result

    # Case 4: search with prefilter=false, search_filter=match("text"),
    #         filter="category='geography'"
    scanner = dataset.scanner(
        nearest={"column": "vector", "q": query_vector, "k": 5},
        prefilter=False,
        filter={
            "expr_filter": "category='geography'",
            "search_filter": MatchQuery("text", "text"),
        },
    )

    result = scanner.to_table()
    ids_result = result["id"].to_pylist()
    assert [300] == ids_result

    # Case 5: search with prefilter=false, search_filter=phrase("text")
    scanner = dataset.scanner(
        nearest={"column": "vector", "q": query_vector, "k": 5},
        prefilter=False,
        filter=PhraseQuery("text", "text"),
    )

    with pytest.raises(ValueError):
        scanner.to_table()

    # Case 6: search with prefilter=false, search_filter=phrase("text")
    scanner = dataset.scanner(
        nearest={"column": "vector", "q": query_vector, "k": 5},
        prefilter=False,
        filter={
            "expr_filter": "category='geography'",
            "search_filter": PhraseQuery("text", "text"),
        },
    )

    with pytest.raises(ValueError):
        scanner.to_table()
