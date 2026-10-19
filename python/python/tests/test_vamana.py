# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

# fork(vamana): IVF_VAMANA_* indices through the Python API. The graph itself
# is tested in Rust; these tests check that Python reaches it: the index types
# and their keyword arguments, the errors Lance raises for what it refuses,
# search through the scanner, optimize, and segments built apart.

import re

import lance
import numpy as np
import pyarrow as pa
import pytest

ROWS = 600
DIM = 32
# A graph over product codes draws several times as slowly as one over scalar
# codes in a debug build, so those cases index fewer dimensions, in four
# sub-vectors of two.
PQ_DIM = 8
K = 10
EF = 60
# Each unlike its default, so that the test sees each one reach the build.
GRAPH = {"max_degree": 16, "search_list_size": 32, "alpha": 1.1}
INDEX = "vector_idx"


def dimension_of(index_type):
    return PQ_DIM if index_type == "IVF_VAMANA_PQ" else DIM


def table_of(vectors, first_id, nulls=None):
    values = pa.array(vectors.reshape(-1), type=pa.float32())
    mask = None if nulls is None else pa.array(nulls)
    column = pa.FixedSizeListArray.from_arrays(values, vectors.shape[1], mask=mask)
    ids = pa.array(np.arange(first_id, first_id + len(vectors)), type=pa.int64())
    return pa.table({"id": ids, "vector": column})


def write(path, vectors, nulls=None):
    """`vectors` as a dataset of two fragments."""
    dataset = lance.write_dataset(
        table_of(vectors, 0, nulls), path, max_rows_per_file=len(vectors) // 2
    )
    assert len(dataset.get_fragments()) == 2
    return dataset


def centroids(metric, dimension):
    """Two IVF centroids that split the unit cube under `metric`: two points
    of its diagonal under l2; under cosine, where the diagonal is one
    direction, unit vectors along the even axes and along the odd ones."""
    if metric == "cosine":
        axes = np.arange(dimension) % 2
        unit = np.sqrt(dimension / 2)
        return np.stack([(axes == parity) / unit for parity in (0, 1)]).astype(
            np.float32
        )
    return np.stack([np.full(dimension, value) for value in (0.25, 0.75)]).astype(
        np.float32
    )


def codebook(metric, dimension):
    """Product codewords no build trains, the same on every run: the residuals
    of 256 seeded vectors of the cube against the nearer centroid, as k-means
    would start from, normalized under cosine as the codes measure them."""
    vectors = np.random.default_rng(5).random((256, dimension), dtype=np.float32)
    if metric == "cosine":
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    centers = centroids(metric, dimension)
    nearer = np.argmin(((vectors[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
    residuals = vectors - centers[nearer]
    return residuals.reshape(256, 4, dimension // 4).transpose(1, 0, 2).copy()


def index_options(index_type, metric):
    """Centroids given, and for product codes a codebook, so that the codes
    are the same on every run."""
    dimension = dimension_of(index_type)
    options = dict(
        GRAPH,
        metric=metric,
        num_partitions=2,
        ivf_centroids=centroids(metric, dimension),
    )
    if index_type == "IVF_VAMANA_PQ":
        options.update(num_sub_vectors=4, pq_codebook=codebook(metric, dimension))
    return options


def exact_distances(vectors, query, metric):
    vectors = vectors.astype(np.float64)
    query = query.astype(np.float64)
    if metric == "cosine":
        norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
        return 1 - vectors @ query / norms
    return ((vectors - query) ** 2).sum(axis=1)


def search(dataset, query, metric, **options):
    nearest = {
        "column": "vector",
        "q": query,
        "k": K,
        "nprobes": 2,
        "ef": EF,
        "metric": metric,
        **options,
    }
    return dataset.to_table(columns=["id", "_distance"], nearest=nearest)


def recall(dataset, vectors, metric, nulls=None):
    """recall@10 of ten seeded queries. Every distance returned must be the
    exact one: over scalar and product codes the walk's shortlist is scored
    again from the dataset."""
    found = 0
    queries = np.random.default_rng(7).random((10, vectors.shape[1]))
    for query in queries.astype(np.float32):
        result = search(dataset, query, metric)
        ids = result["id"].to_numpy()
        exact = exact_distances(vectors, query, metric)
        np.testing.assert_allclose(
            result["_distance"].to_numpy(), exact[ids], rtol=1e-4, atol=1e-5
        )
        if nulls is not None:
            assert not nulls[ids].any()
            exact[nulls] = np.inf
        found += len(set(ids.tolist()) & set(np.argsort(exact)[:K].tolist()))
    return found / (10 * K)


@pytest.mark.parametrize(
    "index_type", ["IVF_VAMANA_FLAT", "IVF_VAMANA_SQ", "IVF_VAMANA_PQ"]
)
@pytest.mark.parametrize("metric", ["l2", "cosine"])
def test_an_index_is_built_described_and_searched(tmp_path, index_type, metric):
    dimension = dimension_of(index_type)
    vectors = np.random.default_rng(3).random((ROWS, dimension), dtype=np.float32)
    nulls = np.arange(ROWS) % 25 == 0
    dataset = write(tmp_path / "ds.lance", vectors, nulls)
    dataset.create_index("vector", index_type, **index_options(index_type, metric))

    [description] = dataset.describe_indices()
    assert description.index_type == index_type
    assert description.details["vamana"] == pytest.approx(GRAPH)
    stats = dataset.stats.index_stats(INDEX)
    assert stats["index_type"] == index_type
    for segment in stats["indices"]:
        assert segment["metric_type"] == metric
        assert segment["num_partitions"] == 2
        assert segment["sub_index"]["index_type"] == "VAMANA"
        assert segment["sub_index"]["params"] == pytest.approx(GRAPH)

    assert recall(dataset, vectors, metric, nulls) >= 0.95

    # As over HNSW, a search list shorter than the results is refused.
    with pytest.raises(pa.ArrowInvalid, match="ef must be greater than or equal to k"):
        search(dataset, vectors[1], metric, ef=K - 1)


@pytest.mark.parametrize(
    "options,error,message",
    [
        ({"max_degree": 0}, ValueError, "max_degree must be between 1 and 1024"),
        ({"search_list_size": 0}, ValueError, "search_list_size must be greater"),
        ({"alpha": 0.9}, ValueError, "alpha must be a finite value of at least 1.0"),
        ({"metric": "dot"}, NotImplementedError, "the l2 and cosine distances"),
    ],
)
def test_lance_refuses_what_it_cannot_build(tmp_path, options, error, message):
    vectors = np.random.default_rng(3).random((ROWS, DIM), dtype=np.float32)
    dataset = write(tmp_path / "ds.lance", vectors)
    with pytest.raises(error, match=message):
        dataset.create_index(
            "vector",
            "IVF_VAMANA_SQ",
            **{**index_options("IVF_VAMANA_SQ", "l2"), **options},
        )
    assert dataset.describe_indices() == []


@pytest.mark.parametrize("num_indices_to_merge", [0, 1])
def test_appended_rows_are_found_before_and_after_optimize(
    tmp_path, num_indices_to_merge
):
    vectors = np.random.default_rng(3).random((ROWS + 100, DIM), dtype=np.float32)
    dataset = write(tmp_path / "ds.lance", vectors[:ROWS])
    dataset.create_index(
        "vector", "IVF_VAMANA_SQ", **index_options("IVF_VAMANA_SQ", "l2")
    )
    dataset = lance.write_dataset(
        table_of(vectors[ROWS:], ROWS), dataset.uri, mode="append"
    )
    assert dataset.stats.index_stats(INDEX)["num_unindexed_rows"] == 100

    for optimized in [False, True]:
        if optimized:
            dataset.optimize.optimize_indices(num_indices_to_merge=num_indices_to_merge)
        result = search(dataset, vectors[ROWS + 17], "l2")
        assert result["id"][0].as_py() == ROWS + 17
        assert recall(dataset, vectors, "l2") >= 0.95

    stats = dataset.stats.index_stats(INDEX)
    assert stats["num_unindexed_rows"] == 0
    assert stats["num_indexed_rows"] == ROWS + 100
    assert len(stats["indices"]) == 2 - num_indices_to_merge


def test_segments_built_apart_are_merged_and_searched(tmp_path):
    vectors = np.random.default_rng(3).random((ROWS, DIM), dtype=np.float32)
    dataset = write(tmp_path / "ds.lance", vectors)
    segments = [
        dataset.create_index_uncommitted(
            "vector",
            "IVF_VAMANA_SQ",
            name=INDEX,
            fragment_ids=[fragment.fragment_id],
            **index_options("IVF_VAMANA_SQ", "l2"),
        )
        for fragment in dataset.get_fragments()
    ]
    merged = dataset.merge_existing_index_segments(segments)
    dataset = dataset.commit_existing_index_segments(INDEX, "vector", [merged])

    stats = dataset.stats.index_stats(INDEX)
    assert stats["index_type"] == "IVF_VAMANA_SQ"
    assert stats["num_segments"] == 1
    assert stats["num_indexed_rows"] == ROWS
    assert recall(dataset, vectors, "l2") >= 0.95


def test_a_stop_margin_reaches_the_walk(tmp_path):
    """`stop_margin` reaches the scanner, which stops each walk by it: with
    `ef` far above `k`, a margin of zero measures fewer distances than the
    same cap without one, and the answer still holds `k` rows at their exact
    distances. A margin no walk can stop by is refused by Lance, which alone
    checks it."""
    vectors = np.random.default_rng(3).random((ROWS, DIM), dtype=np.float32)
    dataset = write(tmp_path / "ds.lance", vectors)
    dataset.create_index(
        "vector", "IVF_VAMANA_SQ", **index_options("IVF_VAMANA_SQ", "l2")
    )
    query = vectors[17]

    def comparisons(**options):
        nearest = {"column": "vector", "q": query, "k": K, "nprobes": 2, "ef": 200}
        plan = dataset.scanner(nearest={**nearest, **options}).analyze_plan()
        return sum(int(n) for n in re.findall(r"index_comparisons=(\d+)", plan))

    unruled = comparisons()
    assert 0 < comparisons(stop_margin=0.0) < unruled

    result = search(dataset, query, "l2", ef=200, stop_margin=0.0)
    ids = result["id"].to_numpy()
    assert len(ids) == K and ids[0] == 17
    np.testing.assert_allclose(
        result["_distance"].to_numpy(),
        exact_distances(vectors, query, "l2")[ids],
        rtol=1e-4,
        atol=1e-5,
    )

    for margin in [-0.1, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="finite fraction"):
            search(dataset, query, "l2", stop_margin=margin)
