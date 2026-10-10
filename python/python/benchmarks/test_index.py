# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from lance.indices import IndicesBuilder, IvfModel, PqModel
from lance.vector import find_duplicate_pairs

N_DIMS = 512


@pytest.mark.benchmark(group="duplicate_pairs")
@pytest.mark.parametrize("index_type", ["IVF_FLAT", "IVF_PQ"])
@pytest.mark.parametrize("num_rows", [1024, 1025, 8192])
def test_duplicate_pairs_exhaust_partition(tmp_path, benchmark, index_type, num_rows):
    rng = np.random.default_rng(9493)
    vectors = rng.normal(size=(num_rows, 64)).astype(np.float32)
    dataset = lance.write_dataset(
        pa.table(
            {
                "vector": pa.array(vectors.tolist(), pa.list_(pa.float32(), 64)),
            }
        ),
        tmp_path,
    )
    params = {}
    if index_type == "IVF_PQ":
        params = dict(
            num_bits=8,
            num_sub_vectors=16,
            pq_codebook=rng.normal(size=(16, 256, 4)).astype(np.float32),
        )
    dataset.create_index(
        "vector",
        index_type,
        num_partitions=1,
        ivf_centroids=np.zeros((1, 64), dtype=np.float32),
        **params,
    )

    def exhaust():
        with find_duplicate_pairs(dataset, "vector", 0.0) as reader:
            return sum(batch.num_rows for batch in reader)

    assert benchmark(exhaust) == 0


def gen_table(num_rows):
    values = pc.random(num_rows * N_DIMS).cast(pa.float32())
    vectors = pa.FixedSizeListArray.from_arrays(values, N_DIMS)
    table = pa.table({"vector": vectors})

    return table


def gen_dataset(tmpdir_factory, num_rows):
    tmp_path = Path(tmpdir_factory.mktemp("index_dataset"))
    table = gen_table(num_rows)
    dataset = lance.write_dataset(table, tmp_path)

    return dataset


@pytest.fixture(scope="module")
def test_dataset(tmpdir_factory):
    # We are writing to this, so it's not beneficial to cache it in the data_dir.
    return gen_dataset(tmpdir_factory, 1_000)


@pytest.fixture(scope="module")
def test_large_dataset(tmpdir_factory):
    # We are writing to this, so it's not beneficial to cache it in the data_dir.
    return gen_dataset(tmpdir_factory, 1_000_000)


@pytest.mark.benchmark(group="create_index")
def test_create_ivf_pq(test_dataset, benchmark):
    benchmark(
        test_dataset.create_index,
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=8,
        num_sub_vectors=2,
        num_bits=8,
        replace=True,
    )


@pytest.mark.benchmark(group="create_index")
def test_create_ivf_pq_torch_cpu(test_dataset, benchmark):
    from lance.dependencies import torch

    benchmark(
        test_dataset.create_index,
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=8,
        num_sub_vectors=2,
        num_bits=8,
        replace=True,
        accelerator=torch.device("cpu"),
    )


@pytest.mark.benchmark(group="create_index")
def test_create_ivf_pq_torch_cpu_one_pass(test_dataset, benchmark):
    from lance.dependencies import torch

    benchmark(
        test_dataset.create_index,
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=8,
        num_sub_vectors=2,
        num_bits=8,
        replace=True,
        accelerator=torch.device("cpu"),
        one_pass_ivfpq=True,
    )


@pytest.mark.benchmark(group="create_index")
@pytest.mark.cuda
def test_create_ivf_pq_cuda(test_dataset, benchmark):
    benchmark(
        test_dataset.create_index,
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=8,
        num_sub_vectors=2,
        num_bits=8,
        accelerator="cuda",
        replace=True,
    )


@pytest.mark.benchmark(group="create_index")
@pytest.mark.cuda
def test_create_ivf_pq_cuda_one_pass(test_dataset, benchmark):
    benchmark(
        test_dataset.create_index,
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=8,
        num_sub_vectors=2,
        num_bits=8,
        accelerator="cuda",
        replace=True,
        one_pass_ivfpq=True,
    )


@pytest.mark.benchmark(group="optimize_index")
@pytest.mark.parametrize("num_partitions", [256, 512])
@pytest.mark.parametrize("num_small_indexes", [5])
@pytest.mark.parametrize("num_new_rows", [12_000])
def test_optimize_index(
    test_large_dataset,
    benchmark,
    num_partitions,
    num_small_indexes,
    num_new_rows,
):
    # insert smaller batch(es) into the large dataset,
    # then benchmark the optimize_index method
    test_large_dataset = test_large_dataset.create_index(
        column="vector",
        index_type="IVF_PQ",
        metric_type="L2",
        num_partitions=num_partitions,
        num_bits=8,
        replace=True,
    )

    for _ in range(num_small_indexes):
        small_table = gen_table(num_new_rows // num_small_indexes)
        lance.write_dataset(small_table, test_large_dataset.uri, mode="append")

    benchmark(test_large_dataset.optimize.optimize_indices)


@pytest.mark.benchmark(group="optimize_index")
@pytest.mark.parametrize("num_partitions", [100, 300])
def test_train_ivf(test_large_dataset, benchmark, num_partitions):
    builder = IndicesBuilder(test_large_dataset, "vector")
    benchmark.pedantic(
        builder.train_ivf,
        kwargs={"num_partitions": num_partitions},
        iterations=1,
        rounds=1,
    )


# Pre-computing partition assignment only makes sense on CUDA and so this benchmark runs
# only on CUDA.
@pytest.mark.benchmark(group="assign_partitions")
@pytest.mark.parametrize("num_partitions", [100, 300])
def test_partition_assignment(test_large_dataset, benchmark, num_partitions):
    from lance.dependencies import torch

    try:
        if not torch.cuda.is_available():
            return
    except:  # noqa: E722
        return
    builder = IndicesBuilder(test_large_dataset, "vector")
    ivf = builder.train_ivf(num_partitions=num_partitions)
    benchmark.pedantic(
        builder.assign_ivf_partitions, args=[ivf, None, "cuda"], iterations=1, rounds=1
    )


def rand_ivf(rand_dataset):
    dtype = rand_dataset.schema.field("vector").type.value_type.to_pandas_dtype()
    centroids = np.random.rand(N_DIMS * 35000).astype(dtype)
    centroids = pa.FixedSizeListArray.from_arrays(centroids, N_DIMS)
    return IvfModel(centroids, "l2")


def rand_pq(rand_dataset, rand_ivf):
    dtype = rand_dataset.schema.field("vector").type.value_type.to_pandas_dtype()
    codebook = np.random.rand(N_DIMS * 256).astype(dtype)
    codebook = pa.FixedSizeListArray.from_arrays(codebook, N_DIMS)
    pq = PqModel(512 // 16, codebook)
    return pq


def gen_rand_part_ids(dataset, dest_uri):
    row_ids = dataset.to_table(with_row_address=True, columns=[])
    part_ids = np.random.randint(0, 35000, size=row_ids.num_rows, dtype=np.uint32)
    table = pa.table({"row_id": row_ids.column(0), "partition": part_ids})
    lance.write_dataset(table, dest_uri, max_rows_per_file=row_ids.num_rows)


@pytest.mark.benchmark(group="transform_vectors")
def test_transform_vectors_no_precomputed_parts(test, tmpdir, benchmark):
    ivf = rand_ivf(test_dataset)
    pq = rand_pq(test_dataset, ivf)
    builder = IndicesBuilder(test_dataset, "vector")
    dst_uri = str(tmpdir / "output.lance")

    benchmark.pedantic(
        builder.transform_vectors,
        args=[ivf, pq, dst_uri],
        iterations=1,
        rounds=1,
    )


@pytest.mark.benchmark(group="transform_vectors")
def test_transform_vectors_with_precomputed_parts(
    test_large_dataset, tmpdir, benchmark
):
    ivf = rand_ivf(test_large_dataset)
    pq = rand_pq(test_large_dataset, ivf)
    builder = IndicesBuilder(test_large_dataset, "vector")
    dst_uri = str(tmpdir / "output.lance")
    part_ids_path = str(tmpdir / "part_ids")
    gen_rand_part_ids(test_large_dataset, part_ids_path)
    benchmark.pedantic(
        builder.transform_vectors,
        args=[ivf, pq, dst_uri, None, part_ids_path],
        iterations=1,
        rounds=1,
    )


@pytest.mark.benchmark(group="shuffle_vectors")
def test_shuffle_vectors(test_large_dataset, tmpdir, benchmark):
    ivf = rand_ivf(test_large_dataset)
    pq = rand_pq(test_large_dataset, ivf)
    builder = IndicesBuilder(test_large_dataset, "vector")
    transformed_uri = str(tmpdir / "output.lance")
    part_ids_path = str(tmpdir / "part_ids")
    gen_rand_part_ids(test_large_dataset, part_ids_path)
    builder.transform_vectors(ivf, pq, transformed_uri, None, part_ids_path)
    shuffle_out = str(tmpdir)
    benchmark.pedantic(
        builder.shuffle_transformed_vectors,
        args=[["output.lance"], shuffle_out, ivf],
        iterations=1,
        rounds=1,
    )
