# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
import math
import os
import pathlib

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.file import LanceFileReader, LanceFileWriter
from lance.indices import IndicesBuilder, IvfModel, PqModel

DIMENSION = 128
NUM_SUBVECTORS = 8
NUM_FRAGMENTS = 3
SMALL_ROWS_PER_FRAGMENT = 100
SMALL_NUM_ROWS = SMALL_ROWS_PER_FRAGMENT * NUM_FRAGMENTS
SMALL_NUM_PARTITIONS = round(np.sqrt(SMALL_NUM_ROWS))
PQ_ROWS_PER_FRAGMENT = 512
PQ_NUM_ROWS = PQ_ROWS_PER_FRAGMENT * NUM_FRAGMENTS
MOSTLY_NULL_ROWS_PER_FRAGMENT = 2000
MOSTLY_NULL_NUM_ROWS = MOSTLY_NULL_ROWS_PER_FRAGMENT * NUM_FRAGMENTS
MOSTLY_NULL_NUM_PARTITIONS = round(np.sqrt(MOSTLY_NULL_NUM_ROWS))
TRAINING_SAMPLE_RATE = 2
TRAINING_MAX_ITERS = 2


def make_ds(
    num_rows: int,
    rows_per_frag: int,
    tmpdir: pathlib.Path,
    dtype: str,
    name: str = "dataset",
):
    vectors = np.random.default_rng(42).standard_normal((num_rows, DIMENSION))
    vectors = vectors.astype(dtype)
    vectors = vectors.reshape(-1)
    vectors = pa.FixedSizeListArray.from_arrays(vectors, DIMENSION)
    table = pa.Table.from_arrays([vectors], names=["vectors"])
    uri = str(tmpdir / name)

    ds = lance.write_dataset(table, uri, max_rows_per_file=rows_per_frag)
    return ds


@pytest.fixture(
    params=[np.float16, np.float32, np.float64],
    ids=["f16", "f32", "f64"],
)
def small_rand_dataset(tmpdir, request):
    return make_ds(SMALL_NUM_ROWS, SMALL_ROWS_PER_FRAGMENT, tmpdir, request.param)


@pytest.fixture
def small_float32_dataset(tmpdir):
    return make_ds(SMALL_NUM_ROWS, SMALL_ROWS_PER_FRAGMENT, tmpdir, np.float32)


@pytest.fixture(
    params=[np.float16, np.float32, np.float64],
    ids=["f16", "f32", "f64"],
)
def pq_rand_dataset(tmpdir, request):
    return make_ds(
        PQ_NUM_ROWS,
        PQ_ROWS_PER_FRAGMENT,
        tmpdir,
        request.param,
        name="pq_dataset",
    )


@pytest.fixture
def pq_float32_dataset(tmpdir):
    return make_ds(
        PQ_NUM_ROWS,
        PQ_ROWS_PER_FRAGMENT,
        tmpdir,
        np.float32,
        name="pq_dataset",
    )


@pytest.fixture
def mostly_null_dataset(tmpdir):
    values = np.random.default_rng(42).standard_normal(MOSTLY_NULL_NUM_ROWS * DIMENSION)
    values = pa.array(values.astype(np.float32))
    null_mask = pa.array(np.arange(MOSTLY_NULL_NUM_ROWS) % 10 != 0)
    vectors = pa.FixedSizeListArray.from_arrays(
        values,
        DIMENSION,
        mask=null_mask,
    )
    table = pa.Table.from_arrays([vectors], names=["vectors"])

    uri = str(tmpdir / "nulls_dataset")
    ds = lance.write_dataset(
        table,
        uri,
        max_rows_per_file=MOSTLY_NULL_ROWS_PER_FRAGMENT,
    )
    return ds


def make_multivector_dataset(tmpdir):
    dimension = 4
    vector_type = pa.list_(pa.list_(pa.float32(), dimension))
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("embeddings", vector_type),
        ]
    )
    table = pa.Table.from_pylist(
        [
            {"id": 1, "embeddings": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]},
            {"id": 2, "embeddings": [[0.8, 0.7, 0.6, 0.5]]},
            {"id": 3, "embeddings": [[0.3, 0.2, 0.1, 0.0], [0.9, 0.8, 0.7, 0.6]]},
            {"id": 4, "embeddings": [[0.4, 0.1, 0.2, 0.3]]},
        ],
        schema=schema,
    )
    ds = lance.write_dataset(
        table,
        pathlib.Path(tmpdir) / "multivector_fragment_ivf",
        max_rows_per_file=2,
    )
    return ds, dimension


def test_ivf_centroids(tmpdir, small_rand_dataset):
    ivf = IndicesBuilder(small_rand_dataset, "vectors").train_ivf(
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
    )

    assert ivf.distance_type == "l2"
    assert len(ivf.centroids) == SMALL_NUM_PARTITIONS

    ivf.save(str(tmpdir / "ivf"))
    reloaded = IvfModel.load(str(tmpdir / "ivf"))
    assert reloaded.distance_type == "l2"
    assert ivf.centroids == reloaded.centroids


def test_ivf_centroids_hamming(tmpdir):
    num_rows = SMALL_NUM_ROWS
    vectors = np.random.default_rng(42).integers(
        0,
        256,
        size=(num_rows, DIMENSION),
        dtype=np.uint8,
    )
    vectors_flat = vectors.reshape(-1)
    vectors_arr = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors_flat, type=pa.uint8()), DIMENSION
    )
    table = pa.Table.from_arrays([vectors_arr], names=["vectors"])
    uri = str(tmpdir / "hamming_dataset")
    ds = lance.write_dataset(table, uri, max_rows_per_file=SMALL_ROWS_PER_FRAGMENT)

    ivf = IndicesBuilder(ds, "vectors").train_ivf(
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        distance_type="hamming",
    )

    assert ivf.distance_type == "hamming"
    expected_partitions = round(math.sqrt(num_rows))
    assert len(ivf.centroids) == expected_partitions

    ivf.save(str(tmpdir / "ivf_hamming"))
    reloaded = IvfModel.load(str(tmpdir / "ivf_hamming"))
    assert reloaded.distance_type == "hamming"
    assert ivf.centroids == reloaded.centroids


@pytest.mark.parametrize("distance_type", ["l2", "cosine", "dot"])
def test_ivf_centroids_mostly_null(mostly_null_dataset, distance_type):
    ivf = IndicesBuilder(mostly_null_dataset, "vectors").train_ivf(
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        distance_type=distance_type,
    )

    assert ivf.distance_type == distance_type
    assert len(ivf.centroids) == MOSTLY_NULL_NUM_PARTITIONS


@pytest.mark.cuda
def test_ivf_centroids_cuda(small_rand_dataset):
    ivf = IndicesBuilder(small_rand_dataset, "vectors").train_ivf(
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        accelerator="cuda",
    )

    assert ivf.distance_type == "l2"
    # Can't use SMALL_NUM_PARTITIONS here because
    # CUDA uses math.ceil and CPU uses round to calc. num_partitions
    assert len(ivf.centroids) == math.ceil(np.sqrt(SMALL_NUM_ROWS))


@pytest.mark.cuda
@pytest.mark.parametrize("distance_type", ["l2", "cosine", "dot"])
def test_ivf_centroids_mostly_null_cuda(mostly_null_dataset, distance_type):
    ivf = IndicesBuilder(mostly_null_dataset, "vectors").train_ivf(
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        accelerator="cuda",
        distance_type=distance_type,
    )

    assert ivf.distance_type == distance_type
    assert len(ivf.centroids) == MOSTLY_NULL_NUM_PARTITIONS


def test_ivf_centroids_distance_type(tmpdir, small_float32_dataset):
    def check(distance_type):
        ivf = IndicesBuilder(small_float32_dataset, "vectors").train_ivf(
            sample_rate=TRAINING_SAMPLE_RATE,
            max_iters=TRAINING_MAX_ITERS,
            distance_type=distance_type,
        )
        assert ivf.distance_type == distance_type
        ivf.save(str(tmpdir / "ivf"))
        reloaded = IvfModel.load(str(tmpdir / "ivf"))
        assert reloaded.distance_type == distance_type

    check("l2")
    check("cosine")
    check("dot")


def test_num_partitions(small_float32_dataset):
    ivf = IndicesBuilder(small_float32_dataset, "vectors").train_ivf(
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        num_partitions=10,
    )
    assert ivf.num_partitions == 10


@pytest.fixture
def small_rand_ivf(small_rand_dataset):
    dtype = small_rand_dataset.schema.field("vectors").type.value_type.to_pandas_dtype()
    centroids = np.random.default_rng(42).random(DIMENSION * 100).astype(dtype)
    centroids = pa.FixedSizeListArray.from_arrays(centroids, DIMENSION)
    return IvfModel(centroids, "l2")


@pytest.fixture
def pq_rand_ivf(pq_rand_dataset):
    dtype = pq_rand_dataset.schema.field("vectors").type.value_type.to_pandas_dtype()
    centroids = np.random.default_rng(42).random(DIMENSION * 100).astype(dtype)
    centroids = pa.FixedSizeListArray.from_arrays(centroids, DIMENSION)
    return IvfModel(centroids, "l2")


@pytest.fixture
def small_float32_ivf(small_float32_dataset):
    centroids = np.random.default_rng(42).random(DIMENSION * 100).astype(np.float32)
    centroids = pa.FixedSizeListArray.from_arrays(centroids, DIMENSION)
    return IvfModel(centroids, "l2")


def test_gen_pq(tmpdir, pq_rand_dataset, pq_rand_ivf):
    pq = IndicesBuilder(pq_rand_dataset, "vectors").train_pq(
        pq_rand_ivf,
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
    )
    assert pq.dimension == DIMENSION
    assert pq.num_subvectors == NUM_SUBVECTORS

    pq.save(str(tmpdir / "pq"))
    reloaded = PqModel.load(str(tmpdir / "pq"))
    assert pq.dimension == reloaded.dimension
    assert pq.codebook == reloaded.codebook

    pq_4bit = IndicesBuilder(pq_rand_dataset, "vectors").train_pq(
        pq_rand_ivf,
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        num_bits=4,
    )
    assert pq_4bit.num_bits == 4
    assert len(pq_4bit.codebook) == 16

    pq_4bit.save(str(tmpdir / "pq_4bit"))
    reloaded = PqModel.load(str(tmpdir / "pq_4bit"))
    assert reloaded.num_bits == 4

    legacy_pq_uri = str(tmpdir / "legacy_pq")
    with LanceFileWriter(
        legacy_pq_uri,
        pa.schema(
            [pa.field("codebook", pq.codebook.type)],
            metadata={b"num_subvectors": str(pq.num_subvectors).encode()},
        ),
    ) as writer:
        writer.write_batch(pa.table([pq.codebook], names=["codebook"]))
    assert PqModel.load(legacy_pq_uri).num_bits == 8


def test_ivf_centroids_fragment_ids(tmpdir):
    rows_per_fragment = 32
    vectors = np.concatenate(
        [
            np.zeros((rows_per_fragment, DIMENSION), dtype=np.float32),
            np.full((rows_per_fragment, DIMENSION), 10.0, dtype=np.float32),
        ],
        axis=0,
    )
    vectors = vectors.reshape(-1)
    table = pa.Table.from_arrays(
        [pa.FixedSizeListArray.from_arrays(vectors, DIMENSION)], names=["vectors"]
    )
    ds = lance.write_dataset(
        table,
        pathlib.Path(tmpdir) / "fragment_ivf",
        max_rows_per_file=rows_per_fragment,
    )
    fragment_ids = [fragment.fragment_id for fragment in ds.get_fragments()]

    first_ivf = IndicesBuilder(ds, "vectors").train_ivf(
        num_partitions=1,
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        fragment_ids=[fragment_ids[0]],
    )
    second_ivf = IndicesBuilder(ds, "vectors").train_ivf(
        num_partitions=1,
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        fragment_ids=[fragment_ids[1]],
    )

    first_centroid = first_ivf.centroids.values.to_numpy().reshape(-1, DIMENSION)[0]
    second_centroid = second_ivf.centroids.values.to_numpy().reshape(-1, DIMENSION)[0]

    assert np.allclose(first_centroid, 0.0, atol=1e-4)
    assert np.allclose(second_centroid, 10.0, atol=1e-4)


def test_ivf_centroids_multivector_fragment_ids(tmpdir):
    ds, dimension = make_multivector_dataset(tmpdir)
    builder = IndicesBuilder(ds, "embeddings")
    assert builder.dimension == dimension

    centroids = pa.FixedSizeListArray.from_arrays(
        pa.array(
            [
                0.1,
                0.2,
                0.3,
                0.4,
                0.8,
                0.7,
                0.6,
                0.5,
            ],
            type=pa.float32(),
        ),
        dimension,
    )
    fragment_ids = [fragment.fragment_id for fragment in ds.get_fragments()]

    index = ds.create_index_uncommitted(
        "embeddings",
        index_type="IVF_HNSW_SQ",
        metric="cosine",
        num_partitions=2,
        fragment_ids=fragment_ids,
        index_uuid="00000000-0000-4000-8000-000000000001",
        ivf_centroids=centroids,
    )

    assert index.uuid == "00000000-0000-4000-8000-000000000001"
    assert index.fragment_ids == set(fragment_ids)
    assert index.name == "embeddings_idx"


def test_indices_builder_multivector_distributed_dimensions(tmpdir, monkeypatch):
    ds, dimension = make_multivector_dataset(tmpdir)
    builder = IndicesBuilder(ds, "embeddings")
    centroids = pa.FixedSizeListArray.from_arrays(
        pa.array([0.1, 0.2, 0.3, 0.4], type=pa.float32()),
        dimension,
    )
    codebook = pa.FixedSizeListArray.from_arrays(
        pa.array([0.1, 0.2, 0.3, 0.4], type=pa.float32()),
        dimension,
    )
    ivf = IvfModel(centroids, "l2")
    pq = PqModel(2, codebook)

    from lance.lance import indices

    captured_dimensions = {}

    def train_pq_model(*args, **kwargs):
        captured_dimensions["train_pq"] = args[2]
        return codebook

    def transform_vectors(*args):
        captured_dimensions["transform_vectors"] = args[2]

    def load_shuffled_vectors(*args):
        captured_dimensions["load_shuffled_vectors"] = args[6]

    monkeypatch.setattr(indices, "train_pq_model", train_pq_model)
    monkeypatch.setattr(indices, "transform_vectors", transform_vectors)
    monkeypatch.setattr(indices, "load_shuffled_vectors", load_shuffled_vectors)
    monkeypatch.setattr(builder, "_verify_pq_sample_rate", lambda *args: None)

    builder.train_pq(ivf, num_subvectors=2, sample_rate=2)
    builder.transform_vectors(ivf, pq, str(pathlib.Path(tmpdir) / "transformed"))
    builder.load_shuffled_vectors(["sorted"], str(tmpdir), ivf, pq)

    assert captured_dimensions == {
        "train_pq": dimension,
        "transform_vectors": dimension,
        "load_shuffled_vectors": dimension,
    }


def test_pq_fragment_ids(pq_float32_dataset):
    fragment_id = pq_float32_dataset.get_fragments()[0].fragment_id
    ivf = IndicesBuilder(pq_float32_dataset, "vectors").train_ivf(
        num_partitions=4,
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        fragment_ids=[fragment_id],
    )

    pq = IndicesBuilder(pq_float32_dataset, "vectors").train_pq(
        ivf,
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
        fragment_ids=[fragment_id],
    )

    assert pq.dimension == DIMENSION
    assert pq.num_subvectors == NUM_SUBVECTORS


def test_pq_invalid_sub_vectors(
    small_float32_dataset,
    small_float32_ivf,
):
    with pytest.raises(
        ValueError,
        match="must be divisible by num_subvectors .* without remainder",
    ):
        IndicesBuilder(small_float32_dataset, "vectors").train_pq(
            small_float32_ivf,
            sample_rate=TRAINING_SAMPLE_RATE,
            max_iters=TRAINING_MAX_ITERS,
            num_subvectors=5,
        )


def test_gen_pq_mostly_null(mostly_null_dataset):
    centroids = np.random.default_rng(42).random(DIMENSION * 100).astype(np.float32)
    centroids = pa.FixedSizeListArray.from_arrays(centroids, DIMENSION)
    ivf = IvfModel(centroids, "l2")

    pq = IndicesBuilder(mostly_null_dataset, "vectors").train_pq(
        ivf,
        sample_rate=TRAINING_SAMPLE_RATE,
        max_iters=TRAINING_MAX_ITERS,
    )
    assert pq.dimension == DIMENSION
    assert pq.num_subvectors == NUM_SUBVECTORS


@pytest.mark.cuda
def test_assign_partitions(small_rand_dataset, small_rand_ivf):
    builder = IndicesBuilder(small_rand_dataset, "vectors")

    partitions_uri = builder.assign_ivf_partitions(small_rand_ivf, accelerator="cuda")

    partitions = lance.dataset(partitions_uri)
    found_row_ids = set()
    for batch in partitions.to_batches():
        row_ids = batch["row_id"]
        for row_id in row_ids:
            found_row_ids.add(row_id)
        part_ids = batch["partition"]
        for part_id in part_ids:
            assert part_id.as_py() < 100
    assert len(found_row_ids) == small_rand_dataset.count_rows()


@pytest.mark.cuda
@pytest.mark.parametrize("distance_type", ["l2", "cosine", "dot"])
def test_assign_partitions_mostly_null(mostly_null_dataset, distance_type):
    centroids = np.random.default_rng(42).random(DIMENSION * 100).astype(np.float32)
    centroids = pa.FixedSizeListArray.from_arrays(centroids, DIMENSION)
    ivf = IvfModel(centroids, distance_type)

    builder = IndicesBuilder(mostly_null_dataset, "vectors")

    partitions_uri = builder.assign_ivf_partitions(ivf, accelerator="cuda")

    partitions = lance.dataset(partitions_uri)
    found_row_ids = set()
    for batch in partitions.to_batches():
        row_ids = batch["row_id"]
        for row_id in row_ids:
            found_row_ids.add(row_id)
        part_ids = batch["partition"]
        for part_id in part_ids:
            assert part_id.as_py() < 100
    assert len(found_row_ids) == (mostly_null_dataset.count_rows() / 10)


@pytest.fixture
def small_rand_pq(small_rand_dataset, small_rand_ivf):
    dtype = small_rand_dataset.schema.field("vectors").type.value_type.to_pandas_dtype()
    codebook = np.random.default_rng(42).random(DIMENSION * 256).astype(dtype)
    codebook = pa.FixedSizeListArray.from_arrays(codebook, DIMENSION)
    pq = PqModel(NUM_SUBVECTORS, codebook)
    return pq


def test_vector_transform(tmpdir, small_rand_dataset, small_rand_ivf, small_rand_pq):
    fragments = list(small_rand_dataset.get_fragments())

    builder = IndicesBuilder(small_rand_dataset, "vectors")
    uri = str(tmpdir / "transformed")
    builder.transform_vectors(small_rand_ivf, small_rand_pq, uri, fragments=fragments)

    reader = LanceFileReader(uri)
    assert reader.metadata().num_rows == (SMALL_ROWS_PER_FRAGMENT * len(fragments))
    data = next(reader.read_all(batch_size=10000).to_batches())

    row_id = data.column("_rowid")
    assert row_id.type == pa.uint64()

    pq_code = data.column("__pq_code")
    assert pq_code.type == pa.list_(pa.uint8(), 8)

    part_id = data.column("__ivf_part_id")
    assert part_id.type == pa.uint32()

    # We need to close the file to be able to overwrite it on Windows.
    del reader

    # test when fragments = None
    builder.transform_vectors(small_rand_ivf, small_rand_pq, uri, fragments=None)
    reader = LanceFileReader(uri)

    assert reader.metadata().num_rows == SMALL_NUM_ROWS


@pytest.mark.cuda
def test_vector_transform_with_precomputed_partitions(
    tmpdir, small_rand_dataset, small_rand_ivf, small_rand_pq
):
    fragments = list(small_rand_dataset.get_fragments())
    builder = IndicesBuilder(small_rand_dataset, "vectors")

    partitions = builder.assign_ivf_partitions(small_rand_ivf, accelerator="cuda")

    uri = str(tmpdir / "transformed")
    builder.transform_vectors(
        small_rand_ivf,
        small_rand_pq,
        uri,
        fragments=fragments,
        partition_ds_uri=partitions,
    )

    reader = LanceFileReader(uri)
    assert reader.metadata().num_rows == (SMALL_ROWS_PER_FRAGMENT * len(fragments))
    data = next(reader.read_all(batch_size=10000).to_batches())

    row_id = data.column("_rowid")
    assert row_id.type == pa.uint64()

    pq_code = data.column("__pq_code")
    assert pq_code.type == pa.list_(pa.uint8(), 8)

    part_id = data.column("__ivf_part_id")
    assert part_id.type == pa.uint32()

    # We need to close the file to be able to overwrite it on Windows.
    del reader

    # test when fragments = None
    builder.transform_vectors(small_rand_ivf, small_rand_pq, uri, fragments=None)
    reader = LanceFileReader(uri)

    assert reader.metadata().num_rows == SMALL_NUM_ROWS


def test_shuffle_vectors(tmpdir, small_rand_dataset, small_rand_ivf, small_rand_pq):
    builder = IndicesBuilder(small_rand_dataset, "vectors")
    uri = str(tmpdir / "transformed_shuffle")
    builder.transform_vectors(small_rand_ivf, small_rand_pq, uri, fragments=None)

    # test shuffle for transformed vectors
    filenames = builder.shuffle_transformed_vectors(
        ["transformed_shuffle"], str(tmpdir), small_rand_ivf, "sorted"
    )

    for fname in filenames:
        full_path = str(tmpdir / fname)
        assert os.path.getsize(full_path) > 0


def test_load_shuffled_vectors(
    tmpdir, small_rand_dataset, small_rand_ivf, small_rand_pq
):
    fragments = list(small_rand_dataset.get_fragments())

    fragments1 = fragments[:1]
    fragments2 = fragments[1:]

    builder = IndicesBuilder(small_rand_dataset, "vectors")

    uri_1 = str(tmpdir / "transformed1")
    builder.transform_vectors(
        small_rand_ivf, small_rand_pq, uri_1, fragments=fragments1
    )
    filenames1 = builder.shuffle_transformed_vectors(
        ["transformed1"], str(tmpdir), small_rand_ivf, "frags1_sorted"
    )

    uri_2 = str(tmpdir / "transformed2")
    builder.transform_vectors(
        small_rand_ivf, small_rand_pq, uri_2, fragments=fragments2
    )
    filenames2 = builder.shuffle_transformed_vectors(
        ["transformed2"], str(tmpdir), small_rand_ivf, "frags2_sorted"
    )

    sorted_filenames = filenames1 + filenames2
    builder.load_shuffled_vectors(
        sorted_filenames, str(tmpdir), small_rand_ivf, small_rand_pq
    )

    final_ds = lance.dataset(str(tmpdir / "dataset"))
    stats = final_ds.stats.index_stats("vectors_idx")
    assert stats["name"] == "vectors_idx"
    assert stats["num_indexed_fragments"] == NUM_FRAGMENTS
