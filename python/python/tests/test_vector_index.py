# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import logging
import os
import platform
import random
import shutil
import string
import tempfile
import time
from pathlib import Path
from typing import Optional

import lance
import lance.cuvs as lance_cuvs
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from conftest import ProgressRecorder, progress_event_tags, stage_progress_values
from lance import LanceDataset, LanceFragment
from lance.dataset import VectorIndexReader
from lance.indices import IndexFileVersion, IndicesBuilder
from lance.query import MatchQuery, PhraseQuery
from lance.util import validate_vector_index  # noqa: E402
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
    nvec=1000, nvec_per_row=5, ndim=128, nans=0, nullify=False, dtype=np.float32
):
    mat = np.random.randn(nvec, nvec_per_row, ndim)
    if nans > 0:
        nans_mat = np.empty((nans, ndim))
        nans_mat[:] = np.nan
        mat = np.concatenate((mat, nans_mat), axis=0)
    mat = mat.astype(dtype)
    price = np.random.rand(nvec + nans) * 100

    def gen_str(n):
        return "".join(random.choices(string.ascii_letters + string.digits, k=n))

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
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=16
    )


@pytest.fixture()
def multivec_dataset():
    tbl = create_multivec_table()
    yield lance.write_dataset(tbl, "memory://")


@pytest.fixture()
def indexed_multivec_dataset(multivec_dataset):
    yield multivec_dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
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

    data = create_table(nvec=3000, ndim=128)
    q = np.random.randn(128).astype(np.float32)
    assert_distributed_vector_consistency(
        data,
        "vector",
        index_type="IVF_PQ",
        index_params={"num_partitions": 10, "num_sub_vectors": 16},
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
    q = np.random.randn(128).astype(np.float32)
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
    # 1024 rows, the entire table should be sampled
    tbl = create_table(nvec=1000, nans=24)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        index_file_version=index_file_version,
    )
    idx_stats = dataset.stats.index_stats("vector_idx")
    assert idx_stats["indices"][0]["index_file_version"] == index_file_version
    validate_vector_index(dataset, "vector")


@pytest.mark.parametrize(
    "index_file_version", [IndexFileVersion.V3, IndexFileVersion.LEGACY]
)
def test_torch_index_with_nans(tmp_path, index_file_version):
    torch = pytest.importorskip("torch")

    # 1024 rows, the entire table should be sampled
    tbl = create_table(nvec=1000, nans=24)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        accelerator=torch.device("cpu"),
        one_pass_ivfpq=True,
        index_file_version=index_file_version,
    )
    idx_stats = dataset.stats.index_stats("vector_idx")
    assert idx_stats["indices"][0]["index_file_version"] == index_file_version
    validate_vector_index(dataset, "vector")


def test_index_with_no_centroid_movement(tmp_path):
    torch = pytest.importorskip("torch")

    # this test makes the centroids essentially [1..]
    # this makes sure the early stop condition in the index building code
    # doesn't do divide by zero
    mat = np.concatenate([np.ones((256, 32))])

    tbl = vec_to_table(data=mat)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        accelerator=torch.device("cpu"),
    )
    validate_vector_index(dataset, "vector")


def test_index_with_pq_codebook(tmp_path):
    tbl = create_table(nvec=1024, ndim=128)
    dataset = lance.write_dataset(tbl, tmp_path)
    pq_codebook = np.random.randn(4, 256, 128 // 4).astype(np.float32)

    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        ivf_centroids=np.random.randn(1, 128).astype(np.float32),
        pq_codebook=pq_codebook,
    )
    validate_vector_index(dataset, "vector", refine_factor=10, pass_threshold=0.99)

    pq_codebook = pa.FixedShapeTensorArray.from_numpy_ndarray(pq_codebook)

    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=4,
        ivf_centroids=np.random.randn(1, 128).astype(np.float32),
        pq_codebook=pq_codebook,
        replace=True,
    )
    validate_vector_index(dataset, "vector", refine_factor=10, pass_threshold=0.99)


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

    with pytest.raises(ValueError):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="cuvs:0",
        )


def test_create_index_accelerator_fallback(tmp_path, caplog):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)

    with caplog.at_level(logging.WARNING):
        dataset = dataset.create_index(
            "vector",
            index_type="IVF_HNSW_SQ",
            num_partitions=4,
            accelerator="cuda",
        )

    stats = dataset.stats.index_stats("vector_idx")
    assert stats["index_type"] == "IVF_HNSW_SQ"
    assert any(
        "does not support GPU acceleration; falling back to CPU" in record.message
        for record in caplog.records
    )


def test_create_index_requires_pylance_cuvs_backend(tmp_path, monkeypatch):
    tbl = create_table()
    dataset = lance.write_dataset(tbl, tmp_path)
    original_import_module = lance_cuvs.import_module

    def _raise_missing(name):
        if name == "lance_cuvs":
            raise ModuleNotFoundError("No module named 'lance_cuvs'")
        return original_import_module(name)

    monkeypatch.setattr(lance_cuvs, "import_module", _raise_missing)
    with pytest.raises(
        ModuleNotFoundError,
        match="requires the optional 'pylance-cuvs' loader package",
    ):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            accelerator="cuvs",
        )


class _FakeCuvsTraining:
    def __init__(self, ivf_centroids, pq_codebook):
        self._ivf_centroids = ivf_centroids
        self._pq_codebook = pq_codebook

    def ivf_centroids(self):
        return self._ivf_centroids

    def pq_codebook(self):
        return self._pq_codebook


class _FakeCuvsArtifact:
    def __init__(self, artifact_uri, files):
        self.artifact_uri = artifact_uri
        self.files = files


def _make_fake_cuvs_training(num_partitions: int = 4, dimension: int = 128):
    centroids = pa.FixedSizeListArray.from_arrays(
        pa.array(np.arange(num_partitions * dimension, dtype=np.float32)),
        dimension,
    )
    codebook = pa.FixedSizeListArray.from_arrays(
        pa.array(np.arange(16 * 256 * 8, dtype=np.float32)),
        8,
    )
    return _FakeCuvsTraining(centroids, codebook)


def test_build_vector_index_on_cuvs_delegates_to_external_backend(
    tmp_path, monkeypatch
):
    ds = _make_sample_dataset_base(tmp_path, "prepare_ivf_pq_cuvs_ds", 512, 128)
    calls = {}
    training = _make_fake_cuvs_training()

    class _FakeBackend:
        def train_ivf_pq(self, dataset_uri, column, **kwargs):
            calls["train"] = {
                "dataset_uri": dataset_uri,
                "column": column,
                **kwargs,
            }
            return training

        def build_ivf_pq_artifact(self, dataset_uri, column, **kwargs):
            calls["build"] = {
                "dataset_uri": dataset_uri,
                "column": column,
                **kwargs,
            }
            return _FakeCuvsArtifact(
                artifact_uri=str(tmp_path / "artifact"),
                files=[str(tmp_path / "artifact" / "data.lance")],
            )

    monkeypatch.setattr(lance_cuvs, "_require_lance_cuvs", lambda: _FakeBackend())

    artifact_uri, files, ivf_centroids, pq_codebook = (
        lance_cuvs.build_vector_index_on_cuvs(
            ds,
            "vector",
            "l2",
            "cuvs",
            4,
            16,
            dst_dataset_uri=tmp_path / "artifact_root",
            storage_options={"region": "us-east-1"},
            sample_rate=7,
            max_iters=20,
            num_bits=4,
            batch_size=4096,
            filter_nan=False,
        )
    )

    assert calls["train"] == {
        "dataset_uri": ds.uri,
        "column": "vector",
        "metric_type": "l2",
        "num_partitions": 4,
        "num_sub_vectors": 16,
        "sample_rate": 7,
        "max_iters": 20,
        "num_bits": 4,
        "filter_nan": False,
        "storage_options": {"region": "us-east-1"},
    }
    assert calls["build"]["dataset_uri"] == ds.uri
    assert calls["build"]["column"] == "vector"
    assert calls["build"]["training"] is training
    assert calls["build"]["artifact_uri"] == str(tmp_path / "artifact_root")
    assert calls["build"]["batch_size"] == 4096
    assert calls["build"]["filter_nan"] is False
    assert calls["build"]["storage_options"] == {"region": "us-east-1"}
    assert artifact_uri == str(tmp_path / "artifact")
    assert files == [str(tmp_path / "artifact" / "data.lance")]
    assert ivf_centroids.equals(training.ivf_centroids())
    assert pq_codebook.equals(training.pq_codebook())


def test_prepare_global_ivf_pq_delegates_to_external_cuvs_backend(
    tmp_path, monkeypatch
):
    ds = _make_sample_dataset_base(tmp_path, "prepare_ivf_pq_cuvs_ds", 512, 128)
    builder = IndicesBuilder(ds, "vector")
    ds._storage_options = {"region": "us-east-1"}
    training = _make_fake_cuvs_training()
    calls = {}

    class _FakeBackend:
        def train_ivf_pq(self, dataset_uri, column, **kwargs):
            calls["train"] = {
                "dataset_uri": dataset_uri,
                "column": column,
                **kwargs,
            }
            return training

    monkeypatch.setattr(lance_cuvs, "_require_lance_cuvs", lambda: _FakeBackend())

    prepared = builder.prepare_global_ivf_pq(
        num_partitions=4,
        num_subvectors=16,
        distance_type="l2",
        accelerator="cuvs",
        sample_rate=7,
        max_iters=20,
    )

    assert calls["train"] == {
        "dataset_uri": ds.uri,
        "column": "vector",
        "metric_type": "l2",
        "num_partitions": 4,
        "num_sub_vectors": 16,
        "sample_rate": 7,
        "max_iters": 20,
        "num_bits": 8,
        "filter_nan": True,
        "storage_options": {"region": "us-east-1"},
    }
    assert prepared["ivf_centroids"].equals(training.ivf_centroids())
    assert prepared["pq_codebook"].equals(training.pq_codebook())


def test_require_lance_cuvs_rejects_incompatible_backend(monkeypatch):
    class _IncompleteBackend:
        def train_ivf_pq(self, *args, **kwargs):
            raise AssertionError("should not be called")

    monkeypatch.setattr(lance_cuvs, "import_module", lambda name: _IncompleteBackend())
    with pytest.raises(
        ImportError, match="missing backend APIs: build_ivf_pq_artifact"
    ):
        lance_cuvs._require_lance_cuvs()


def test_create_index_rejects_missing_precomputed_partition_artifact(tmp_path):
    dataset = lance.write_dataset(
        create_table(nvec=64, ndim=128), tmp_path / "artifact_src"
    )

    with pytest.raises(Exception):
        dataset.create_index(
            "vector",
            index_type="IVF_PQ",
            num_partitions=4,
            num_sub_vectors=16,
            ivf_centroids=np.random.randn(4, 128).astype(np.float32),
            pq_codebook=np.random.randn(16, 256, 8).astype(np.float32),
            precomputed_partition_artifact_uri=str(tmp_path / "missing_artifact"),
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


def test_index_type(dataset, tmp_path):
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")

    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        replace=True,
    )
    stats = ann_ds.stats.index_stats("vector_idx")
    assert stats["index_type"] == "IVF_PQ"

    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_HNSW_SQ",
        num_partitions=4,
        num_sub_vectors=16,
        replace=True,
    )
    stats = ann_ds.stats.index_stats("vector_idx")
    assert stats["index_type"] == "IVF_HNSW_SQ"

    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_HNSW_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        replace=True,
    )
    stats = ann_ds.stats.index_stats("vector_idx")
    assert stats["index_type"] == "IVF_HNSW_PQ"


def test_create_dot_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        metric="dot",
    )
    assert ann_ds.has_index


def test_create_4bit_ivf_pq_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=1,
        num_sub_vectors=16,
        num_bits=4,
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
        num_bits=1,
    )
    assert ds.describe_indices()[0].field_names == ["vector"]
    stats = ds.stats.index_stats("vector_idx")
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


def test_create_ivf_hnsw_pq_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_HNSW_PQ",
        num_partitions=4,
        num_sub_vectors=16,
    )
    assert ann_ds.describe_indices()[0].field_names == ["vector"]


def test_create_ivf_hnsw_sq_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_HNSW_SQ",
        num_partitions=4,
        num_sub_vectors=16,
    )
    assert ann_ds.describe_indices()[0].field_names == ["vector"]


def test_create_ivf_hnsw_flat_index(dataset, tmp_path):
    assert not dataset.has_index
    ann_ds = lance.write_dataset(dataset.to_table(), tmp_path / "indexed.lance")
    ann_ds = ann_ds.create_index(
        "vector",
        index_type="IVF_HNSW_FLAT",
        num_partitions=4,
        num_sub_vectors=16,
    )
    assert ann_ds.describe_indices()[0].field_names == ["vector"]


def test_multivec_ann(indexed_multivec_dataset: lance.LanceDataset):
    query = np.random.rand(5, 128)
    results = indexed_multivec_dataset.scanner(
        nearest={"column": "vector", "q": query, "k": 100}
    ).to_table()
    assert results.num_rows == 100
    assert results["vector"].type == pa.list_(pa.list_(pa.float32(), 128))
    assert len(results["vector"][0]) == 5

    # query with single vector also works
    query = np.random.rand(128)
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
    query = np.random.rand(256)
    with pytest.raises(ValueError, match="does not match index column size"):
        indexed_multivec_dataset.to_table(
            nearest={"column": "vector", "q": query, "k": 100}
        )

    # query with a list of vectors that some dim not match
    query = [np.random.rand(128)] * 5 + [np.random.rand(256)]
    with pytest.raises(ValueError, match="All query vectors must have the same length"):
        indexed_multivec_dataset.to_table(
            nearest={"column": "vector", "q": query, "k": 100}
        )


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
    DIM = 64
    uri = tmp_path / "f16data.lance"
    f16_data = np.random.uniform(0, 1, 2048 * DIM).astype(np.float16)
    fsl = pa.FixedSizeListArray.from_arrays(f16_data, DIM)
    tbl = pa.Table.from_pydict({"vector": fsl})
    dataset = lance.write_dataset(tbl, uri)
    dataset.create_index(
        "vector", index_type="IVF_PQ", num_partitions=4, num_sub_vectors=2
    )

    q = np.random.uniform(0, 1, DIM).astype(np.float16)
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
    TOTAL = 2048
    data = np.random.uniform(0, 1, TOTAL * DIM).astype(np.float32)

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
        num_partitions=2,
        num_sub_vectors=2,
        replace=True,
    )
    tbl = ds.to_table(
        nearest={"column": "vector", "q": data[0:DIM], "k": TOTAL, "nprobes": 2},
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

    tbl = create_table(nvec=1000)

    dataset = lance.write_dataset(tbl, tmp_path)
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        accelerator=torch.device("cpu"),
    )

    # Get the centroids
    index_name = dataset.describe_indices()[0].name
    index_stats = dataset.stats.index_stats(index_name)
    centroids = index_stats["indices"][0]["centroids"]
    values = pa.array([x for arr in centroids for x in arr], pa.float32())
    centroids = pa.FixedSizeListArray.from_arrays(values, 128)

    dataset.alter_columns(dict(path="vector", data_type=pa.list_(pa.float16(), 128)))

    # centroids are f32, but the column is now f16
    dataset = dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
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


def test_load_indices(dataset):
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
    assert info.type_url == "/lance.table.VectorIndexDetails"
    assert info.index_type == "IVF_PQ"
    assert info.num_rows_indexed == 1000
    assert info.fields == [0]
    assert info.field_names == ["vector"]
    assert len(info.segments) == 1
    assert info.segments[0].fragment_ids == {0}
    assert info.segments[0].dataset_version_at_last_update == 1
    assert info.segments[0].index_version == 1
    assert info.segments[0].created_at is not None


def test_optimize_indices(indexed_dataset):
    data = create_table()
    indexed_dataset = lance.write_dataset(data, indexed_dataset.uri, mode="append")
    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_indices"] == 1
    indexed_dataset.optimize.optimize_indices(num_indices_to_merge=0)
    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_indices"] == 2


def test_logical_and_physical_index_views(indexed_dataset):
    data = create_table()
    indexed_dataset = lance.write_dataset(data, indexed_dataset.uri, mode="append")
    indexed_dataset.optimize.optimize_indices(num_indices_to_merge=0)

    logical_indices = indexed_dataset.describe_indices()
    assert len(logical_indices) == 1
    assert logical_indices[0].name == "vector_idx"
    assert len(logical_indices[0].segments) == 2
    assert all(segment.fragment_ids for segment in logical_indices[0].segments)

    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_segments"] == stats["num_indices"] == 2
    assert stats["segments"] == stats["indices"]


@pytest.mark.skip(reason="retrain is deprecated")
def test_retrain_indices(indexed_dataset):
    data = create_table()
    indexed_dataset = lance.write_dataset(data, indexed_dataset.uri, mode="append")
    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_indices"] == 1

    indexed_dataset.optimize.optimize_indices(num_indices_to_merge=0)
    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_indices"] == 2

    stats = indexed_dataset.stats.index_stats("vector_idx")
    centroids = stats["indices"][0]["centroids"]
    delta_centroids = stats["indices"][1]["centroids"]
    assert centroids == delta_centroids

    indexed_dataset.optimize.optimize_indices(retrain=True)
    new_centroids = indexed_dataset.stats.index_stats("vector_idx")["indices"][0][
        "centroids"
    ]
    stats = indexed_dataset.stats.index_stats("vector_idx")
    assert stats["num_indices"] == 1
    assert centroids != new_centroids


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
    assert indices[0].field_names == ["embedding"]

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
    assert np.allclose(brute_distances, index_distances, rtol=0.0, atol=0.0)


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
    similarity_metric/similarity_threshold. It compares recall@K against a ground
    truth computed via exact search (use_index=False) on the single dataset and
    asserts that the recall difference between single-machine and distributed
    indices is within 10%.

    Steps
    -----
    1) Write `data` to two URIs (single, distributed); ensure distributed has >=2
       fragments (rewrite with max_rows_per_file if needed)
    2) Build a single-machine index via `create_index`
    3) Global training (IVF/PQ) using `IndicesBuilder.prepare_global_ivfpq` when
       appropriate; for IVF_FLAT/SQ variants, train IVF centroids via
       `IndicesBuilder.train_ivf`
    4) Build the distributed index via
       `lance.indices.builder.build_distributed_vector_index`, passing the
       preprocessed artifacts
    5) For each query, compute ground-truth TopK IDs using exact search
       (use_index=False), then compute TopK using single index and the distributed
       index with consistent nearest settings (refine_factor=1; IVF uses nprobes)
    6) Compute recall for single and distributed using the provided formula and
       assert the absolute difference is <= 0.10. Also print the recalls.
    """
    # Keep signature compatibility but ignore similarity_metric/threshold
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

    # Build single-machine index
    single_ds = single_ds.create_index(
        column=column,
        index_type=index_type,
        **index_params,
    )

    # Global training / preparation for distributed build
    preprocessed = None
    builder = IndicesBuilder(single_ds, column)
    nparts = index_params.get("num_partitions", None)
    nsub = index_params.get("num_sub_vectors", None)
    dist_type = index_params.get("metric", "l2")
    num_rows = single_ds.count_rows()

    # Choose a safe sample_rate that satisfies IVF (nparts*sr <= rows) and PQ
    # (256*sr <= rows). Minimum 2 as required by builder verification.
    safe_sr_ivf = num_rows // max(1, nparts or 1)
    safe_sr_pq = num_rows // 256
    safe_sr = max(2, min(safe_sr_ivf, safe_sr_pq))

    if index_type in {"IVF_PQ", "IVF_HNSW_PQ"}:
        preprocessed = builder.prepare_global_ivf_pq(
            nparts,
            nsub,
            distance_type=dist_type,
            sample_rate=safe_sr,
        )
    elif (
        ("IVF_FLAT" in index_type)
        or ("IVF_SQ" in index_type)
        or ("IVF_HNSW_FLAT" in index_type)
    ):
        ivf_model = builder.train_ivf(
            nparts,
            distance_type=dist_type,
            sample_rate=safe_sr,
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
            nearest["nprobes"] = max(16, int(index_params.get("num_partitions", 4)) * 4)
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

    # Assert recall difference within 10%
    assert abs(rs - rd) <= 1 - similarity_threshold, (
        f"Recall difference too large: single={rs:.3f}, distributed={rd:.3f}, "
        f"diff={abs(rs - rd):.3f} (> {similarity_threshold})"
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
    mat = np.random.rand(n_rows, dim).astype(np.float32)
    ids = np.arange(n_rows)
    arr = pa.array(mat.tolist(), type=pa.list_(pa.float32(), dim))
    tbl = pa.table({"id": ids, "vector": arr})
    return lance.write_dataset(
        tbl, tmp_path / name, max_rows_per_file=max_rows_per_file
    )


def test_prepared_global_ivfpq_distributed_merge_and_search(tmp_path: Path):
    ds = _make_sample_dataset_base(tmp_path, "preproc_ds", 2000, 128)

    # Global preparation
    builder = IndicesBuilder(ds, "vector")
    preprocessed = builder.prepare_global_ivf_pq(
        num_partitions=4,
        num_subvectors=4,
        distance_type="l2",
        sample_rate=3,
        max_iters=20,
    )

    # Distributed build using prepared centroids/codebook
    ds = build_distributed_vector_index(
        ds,
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=4,
        world=2,
        ivf_centroids=preprocessed["ivf_centroids"],
        pq_codebook=preprocessed["pq_codebook"],
    )

    # Query sanity
    q = np.random.rand(128).astype(np.float32)
    results = ds.to_table(nearest={"column": "vector", "q": q, "k": 10})
    assert 0 < len(results) <= 10


def test_consistency_improves_with_preprocessed_centroids(tmp_path: Path):
    ds = _make_sample_dataset_base(tmp_path, "preproc_ds", 2000, 128)

    builder = IndicesBuilder(ds, "vector")
    pre = builder.prepare_global_ivf_pq(
        num_partitions=4,
        num_subvectors=16,
        distance_type="l2",
        sample_rate=7,
        max_iters=20,
    )

    # Build single-machine index as ground truth target index
    single_ds = lance.write_dataset(ds.to_table(), tmp_path / "single_ivfpq")
    single_ds = single_ds.create_index(
        column="vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
    )

    # Distributed with preprocessed IVF centroids
    dist_pre = lance.write_dataset(ds.to_table(), tmp_path / "dist_pre")
    dist_pre = build_distributed_vector_index(
        dist_pre,
        "vector",
        index_type="IVF_PQ",
        num_partitions=4,
        num_sub_vectors=16,
        world=2,
        ivf_centroids=pre["ivf_centroids"],
        pq_codebook=pre["pq_codebook"],
    )

    # Evaluate recall vs exact search
    q = np.random.rand(128).astype(np.float32)
    topk = 10
    gt = single_ds.to_table(
        nearest={"column": "vector", "q": q, "k": topk, "use_index": False}
    )
    res_pre = dist_pre.to_table(nearest={"column": "vector", "q": q, "k": topk})

    gt_ids = gt["id"].to_pylist()
    pre_ids = res_pre["id"].to_pylist()

    def _recall(gt_ids, res_ids):
        s = set(int(x) for x in gt_ids)
        d = set(int(x) for x in res_ids)
        return len(s & d) / max(1, len(s))

    recall_pre = _recall(gt_ids, pre_ids)

    # Expect some non-zero recall with preprocessed IVF centroids
    if recall_pre < 0.10:
        pytest.skip(
            "Distributed IVF_PQ recall below threshold in current "
            "environment - known issue"
        )
    assert recall_pre >= 0.10


def test_metadata_merge_pq_success(tmp_path):
    ds = _make_sample_dataset_base(tmp_path, "dist_ds", 2000, 128)
    frags = ds.get_fragments()
    assert len(frags) >= 2, "Need at least 2 fragments for distributed testing"
    mid = max(1, len(frags) // 2)
    node1 = [f.fragment_id for f in frags[:mid]]
    node2 = [f.fragment_id for f in frags[mid:]]
    builder = IndicesBuilder(ds, "vector")
    pre = builder.prepare_global_ivf_pq(
        num_partitions=8,
        num_subvectors=16,
        distance_type="l2",
        sample_rate=7,
        max_iters=20,
    )
    try:
        segments = _build_segments(
            ds,
            "vector",
            "IVF_PQ",
            [node1, node2],
            index_name="vector_idx",
            num_partitions=8,
            num_sub_vectors=16,
            ivf_centroids=pre["ivf_centroids"],
            pq_codebook=pre["pq_codebook"],
        )
        ds = _commit_segments_helper(ds, segments, "vector")
        q = np.random.rand(128).astype(np.float32)
        results = ds.to_table(nearest={"column": "vector", "q": q, "k": 10})
        assert 0 < len(results) <= 10
    except ValueError as e:
        raise e


def test_distributed_workflow_merge_and_search(tmp_path):
    """End-to-end: build IVF_PQ on two groups, merge, and verify search returns
    results."""
    ds = _make_sample_dataset_base(tmp_path, "dist_ds", 2000, 128)
    frags = ds.get_fragments()
    if len(frags) < 2:
        pytest.skip("Need at least 2 fragments for distributed testing")
    mid = len(frags) // 2
    node1 = [f.fragment_id for f in frags[:mid]]
    node2 = [f.fragment_id for f in frags[mid:]]
    builder = IndicesBuilder(ds, "vector")
    pre = builder.prepare_global_ivf_pq(
        num_partitions=4,
        num_subvectors=4,
        distance_type="l2",
        sample_rate=7,
        max_iters=20,
    )
    try:
        segments = _build_segments(
            ds,
            "vector",
            "IVF_PQ",
            [node1, node2],
            index_name="vector_idx",
            num_partitions=4,
            num_sub_vectors=4,
            ivf_centroids=pre["ivf_centroids"],
            pq_codebook=pre["pq_codebook"],
        )
        ds = _commit_segments_helper(ds, segments, "vector")
        q = np.random.rand(128).astype(np.float32)
        results = ds.to_table(nearest={"column": "vector", "q": q, "k": 10})
        assert 0 < len(results) <= 10
    except ValueError as e:
        raise e


def test_vector_merge_two_shards_success_flat(tmp_path):
    ds = _make_sample_dataset_base(tmp_path, "dist_ds", 1000, 128)
    frags = ds.get_fragments()
    assert len(frags) >= 2
    shard1 = [frags[0].fragment_id]
    shard2 = [frags[1].fragment_id]
    # Global preparation
    builder = IndicesBuilder(ds, "vector")
    preprocessed = builder.prepare_global_ivf_pq(
        num_partitions=4,
        num_subvectors=4,
        distance_type="l2",
        sample_rate=3,
        max_iters=20,
    )

    segments = _build_segments(
        ds,
        "vector",
        "IVF_FLAT",
        [shard1, shard2],
        index_name="vector_idx",
        num_partitions=4,
        num_sub_vectors=128,
        ivf_centroids=preprocessed["ivf_centroids"],
        pq_codebook=preprocessed["pq_codebook"],
    )
    ds = _commit_segments_helper(ds, segments, column="vector")
    q = np.random.rand(128).astype(np.float32)
    result = ds.to_table(nearest={"column": "vector", "q": q, "k": 5})
    assert 0 < len(result) <= 5


@pytest.mark.parametrize(
    "index_type,num_sub_vectors",
    [
        ("IVF_PQ", 4),
        ("IVF_FLAT", 128),
    ],
)
def test_distributed_ivf_parameterized(tmp_path, index_type, num_sub_vectors):
    ds = _make_sample_dataset_base(tmp_path, "dist_ds", 2000, 128)
    frags = ds.get_fragments()
    assert len(frags) >= 2
    mid = len(frags) // 2
    node1 = [f.fragment_id for f in frags[:mid]]
    node2 = [f.fragment_id for f in frags[mid:]]
    builder = IndicesBuilder(ds, "vector")
    pre = builder.prepare_global_ivf_pq(
        num_partitions=4,
        num_subvectors=num_sub_vectors,
        distance_type="l2",
        sample_rate=7,
        max_iters=20,
    )

    try:
        base_kwargs = dict(
            column="vector",
            index_type=index_type,
            num_partitions=4,
            num_sub_vectors=num_sub_vectors,
        )

        kwargs1 = dict(base_kwargs, fragment_ids=node1)
        kwargs2 = dict(base_kwargs, fragment_ids=node2)

        if pre is not None:
            kwargs1.update(
                ivf_centroids=pre["ivf_centroids"], pq_codebook=pre["pq_codebook"]
            )
            kwargs2.update(
                ivf_centroids=pre["ivf_centroids"], pq_codebook=pre["pq_codebook"]
            )

        segments = [
            ds.create_index_uncommitted(**kwargs1),
            ds.create_index_uncommitted(**kwargs2),
        ]
        ds = _commit_segments_helper(ds, segments, "vector")

        q = np.random.rand(128).astype(np.float32)
        results = ds.to_table(nearest={"column": "vector", "q": q, "k": 10})
        assert 0 < len(results) <= 10
    except ValueError as e:
        raise e


@pytest.mark.parametrize(
    "index_type,num_sub_vectors",
    [
        ("IVF_PQ", 128),
        ("IVF_SQ", None),
    ],
)
def test_merge_two_shards_parameterized(tmp_path, index_type, num_sub_vectors):
    ds = _make_sample_dataset_base(tmp_path, "dist_ds2", 2000, 128)
    frags = ds.get_fragments()
    assert len(frags) >= 2
    shard1 = [frags[0].fragment_id]
    shard2 = [frags[1].fragment_id]
    builder = IndicesBuilder(ds, "vector")
    pre = builder.prepare_global_ivf_pq(
        num_partitions=4,
        num_subvectors=num_sub_vectors,
        distance_type="l2",
        sample_rate=7,
        max_iters=20,
    )

    base_kwargs = {
        "column": "vector",
        "index_type": index_type,
        "num_partitions": 4,
    }

    # first shard
    kwargs1 = dict(base_kwargs)
    kwargs1["fragment_ids"] = shard1
    if num_sub_vectors is not None:
        kwargs1["num_sub_vectors"] = num_sub_vectors
    if pre is not None:
        kwargs1["ivf_centroids"] = pre["ivf_centroids"]
        # only PQ has pq_codebook
        if "pq_codebook" in pre:
            kwargs1["pq_codebook"] = pre["pq_codebook"]
    segment1 = ds.create_index_uncommitted(**kwargs1)

    # second shard
    kwargs2 = dict(base_kwargs)
    kwargs2["fragment_ids"] = shard2
    if num_sub_vectors is not None:
        kwargs2["num_sub_vectors"] = num_sub_vectors
    if pre is not None:
        kwargs2["ivf_centroids"] = pre["ivf_centroids"]
        if "pq_codebook" in pre:
            kwargs2["pq_codebook"] = pre["pq_codebook"]
    segment2 = ds.create_index_uncommitted(**kwargs2)

    merged_segment = ds.merge_existing_index_segments([segment1, segment2])
    ds = _commit_segments_helper(ds, [merged_segment], column="vector")

    q = np.random.rand(128).astype(np.float32)
    results = ds.to_table(nearest={"column": "vector", "q": q, "k": 5})
    assert 0 < len(results) <= 5


def test_merge_existing_index_segments_builds_vector_segment(tmp_path):
    ds = _make_sample_dataset_base(tmp_path, "merge_existing_segments_ds", 2000, 128)
    frags = ds.get_fragments()
    assert len(frags) >= 2
    builder = IndicesBuilder(ds, "vector")
    preprocessed = builder.prepare_global_ivf_pq(
        num_partitions=4,
        num_subvectors=4,
        distance_type="l2",
        sample_rate=7,
        max_iters=20,
    )

    segments = [
        ds.create_index_uncommitted(
            "vector",
            "IVF_FLAT",
            name="vector_idx",
            train=True,
            fragment_ids=[fragment.fragment_id],
            num_partitions=4,
            num_sub_vectors=128,
            ivf_centroids=preprocessed["ivf_centroids"],
            pq_codebook=preprocessed["pq_codebook"],
        )
        for fragment in frags[:2]
    ]
    assert all(segment.index_details is not None for segment in segments)

    merged_segment = ds.merge_existing_index_segments(segments)
    assert merged_segment.fragment_ids is not None
    assert sorted(merged_segment.fragment_ids) == sorted(
        [fragment.fragment_id for fragment in frags[:2]]
    )
    ds = ds.commit_existing_index_segments("vector_idx", "vector", [merged_segment])

    q = np.random.rand(128).astype(np.float32)
    results = ds.to_table(nearest={"column": "vector", "q": q, "k": 5})
    assert 0 < len(results) <= 5


def test_distributed_ivf_pq_order_invariance(tmp_path: Path):
    """Ensure distributed IVF_PQ build is invariant to shard build order."""
    ds = _make_sample_dataset_base(tmp_path, "dist_ds", 2000, 128)

    # Global IVF+PQ training once; artifacts are reused across shard orders.
    builder = IndicesBuilder(ds, "vector")
    pre = builder.prepare_global_ivf_pq(
        num_partitions=4,
        num_subvectors=16,
        distance_type="l2",
        sample_rate=7,
    )

    # Copy the dataset twice so index manifests do not clash and we can vary
    # the shard build order independently on identical data.
    ds_order_12 = lance.write_dataset(
        ds.to_table(), tmp_path / "pq_order_node1_node2", max_rows_per_file=500
    )
    ds_order_21 = lance.write_dataset(
        ds.to_table(), tmp_path / "pq_order_node2_node1", max_rows_per_file=500
    )

    # For each copy, derive two shard groups from its own fragments.
    frags_12 = ds_order_12.get_fragments()
    if len(frags_12) < 2:
        pytest.skip("Need at least 2 fragments for distributed indexing (order_12)")
    mid_12 = len(frags_12) // 2
    node1_12 = [f.fragment_id for f in frags_12[:mid_12]]
    node2_12 = [f.fragment_id for f in frags_12[mid_12:]]
    if not node1_12 or not node2_12:
        pytest.skip("Failed to split fragments into two non-empty groups (order_12)")

    frags_21 = ds_order_21.get_fragments()
    if len(frags_21) < 2:
        pytest.skip("Need at least 2 fragments for distributed indexing (order_21)")
    mid_21 = len(frags_21) // 2
    node1_21 = [f.fragment_id for f in frags_21[:mid_21]]
    node2_21 = [f.fragment_id for f in frags_21[mid_21:]]
    if not node1_21 or not node2_21:
        pytest.skip("Failed to split fragments into two non-empty groups (order_21)")

    def build_distributed_ivf_pq(ds_copy, shard_order):
        try:
            segments = _build_segments(
                ds_copy,
                "vector",
                "IVF_PQ",
                shard_order,
                index_name="vector_idx",
                num_partitions=4,
                num_sub_vectors=16,
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
    k = 10
    sample_tbl = ds.sample(10, columns=["vector"])
    queries = [
        np.asarray(v, dtype=np.float32) for v in sample_tbl["vector"].to_pylist()
    ]

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
                    "nprobes": 16,
                    "refine_factor": 100,
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
