# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.vector import (
    find_duplicate_pairs,
    find_duplicate_pairs_in_partition,
    get_ivf_partition_info,
    hamming_clustering_for_ivf_partition,
    hamming_clustering_for_sample,
    vec_to_table,
)


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
@pytest.mark.parametrize("index_type", ["IVF_FLAT", "IVF_HNSW_FLAT"])
def test_duplicate_pairs_flat(tmp_path, metric, index_type):
    vectors = np.array([[1, 0], [1, 0], [0, 1], [2, 0]], dtype=np.float32)
    table = pa.table({"id": range(4), "vector": pa.array(vectors.tolist())})
    table = table.set_column(
        1, "vector", pa.array(vectors.tolist(), pa.list_(pa.float32(), 2))
    )
    ds = lance.write_dataset(table, tmp_path, max_rows_per_file=2)
    ds = ds.create_index(
        "vector",
        index_type,
        metric=metric,
        num_partitions=1,
        ivf_centroids=np.array([[1, 0]], dtype=np.float32),
    )
    ids = ds.to_table(columns=["id"], with_row_id=True).to_pylist()
    positions = {r["_rowid"]: r["id"] for r in ids}
    segment_id = ds.describe_indices()[0].segments[0].uuid
    threshold = -0.5 if metric == "dot" else 0.5
    expected = {}
    for i in range(4):
        for j in range(i + 1, 4):
            if metric == "l2":
                d = np.sum((vectors[i] - vectors[j]) ** 2)
            elif metric == "cosine":
                d = 1 - np.dot(vectors[i], vectors[j]) / (
                    np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j])
                )
            else:
                d = 1 - np.dot(vectors[i], vectors[j])
            if d <= threshold:
                expected[(i, j)] = float(d)
    with find_duplicate_pairs(ds, "vector", threshold) as reader:
        pairs = reader.read_all()
    with find_duplicate_pairs_in_partition(
        ds, "vector", segment_id, 0, threshold
    ) as reader:
        assert reader.read_all().equals(pairs)
    with find_duplicate_pairs(
        ds, "vector", threshold, max_concurrency=1, memory_limit=0
    ) as reader:
        assert reader.read_all().equals(pairs)
    actual = {}
    for row in pairs.to_pylist():
        a, b = row["row_id_a"], row["row_id_b"]
        key = tuple(sorted((positions[a], positions[b])))
        assert key not in actual
        actual[key] = row["distance"]
    assert actual == pytest.approx(expected)
    # Exhaustive candidate recall, including exact threshold matches.
    assert len(actual.keys() & expected.keys()) / len(expected) == 1.0
    ds.delete("id = 1")
    with find_duplicate_pairs(ds, "vector", 100) as reader:
        result = reader.read_all().to_pylist()
    assert all(positions[r[k]] != 1 for r in result for k in ["row_id_a", "row_id_b"])
    assert len(result) == 3


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
@pytest.mark.parametrize(
    "index_type,bits",
    [
        ("IVF_PQ", 4),
        ("IVF_PQ", 8),
        ("IVF_SQ", 8),
        ("IVF_HNSW_PQ", 4),
        ("IVF_HNSW_SQ", 8),
        ("IVF_RQ", 1),
        ("IVF_RQ", 5),
        ("IVF_RQ", 9),
    ],
)
def test_duplicate_pairs_quantized(tmp_path, metric, index_type, bits):
    rng = np.random.default_rng(7)
    unique = rng.normal(size=(17, 8)).astype(np.float32)
    vectors = np.repeat(unique, 2, axis=0)
    table = pa.table({"vector": pa.array(vectors.tolist(), pa.list_(pa.float32(), 8))})
    ds = lance.write_dataset(table, tmp_path, max_rows_per_file=10)
    params = {"num_bits": bits}
    if "PQ" in index_type:
        params.update(
            num_sub_vectors=4,
            pq_codebook=rng.normal(size=(4, 2**bits, 2)).astype(np.float32),
        )
    ds = ds.create_index(
        "vector",
        index_type,
        metric=metric,
        num_partitions=1,
        ivf_centroids=np.ones((1, 8), dtype=np.float32),
        **params,
    )
    with find_duplicate_pairs(ds, "vector", 1e6) as reader:
        expected_table = reader.read_all()
    # Distances and order do not depend on concurrency, spilling or the
    # scoped API.
    with find_duplicate_pairs(
        ds, "vector", 1e6, max_concurrency=1, memory_limit=0
    ) as reader:
        assert reader.read_all().equals(expected_table)
    segment = ds.describe_indices()[0].segments[0].uuid
    with find_duplicate_pairs_in_partition(ds, "vector", segment, 0, 1e6) as reader:
        assert reader.read_all().equals(expected_table)
    all_pairs = expected_table.to_pylist()
    assert len(all_pairs) == len(vectors) * (len(vectors) - 1) // 2
    keys = {tuple(sorted((r["row_id_a"], r["row_id_b"]))) for r in all_pairs}
    assert len(keys) == len(all_pairs)
    assert all(np.isfinite(r["distance"]) for r in all_pairs)
    if metric == "cosine":
        assert all(0 <= r["distance"] <= 2 for r in all_pairs)
    if metric != "dot":
        row_ids = ds.to_table(columns=[], with_row_id=True)["_rowid"].to_pylist()
        expected = {(row_ids[i], row_ids[i + 1]) for i in range(0, len(vectors), 2)}
        with find_duplicate_pairs(ds, "vector", 1e-5) as reader:
            result = reader.read_all().to_pylist()
        found = {tuple(sorted((r["row_id_a"], r["row_id_b"]))) for r in result}
        assert len(expected & found) / len(expected) >= 0.5


def test_duplicate_pairs_validation_and_snapshot(tmp_path):
    table = pa.table(
        {"vector": pa.array([[1.0, 0.0], None, [1.0, 0.0]], pa.list_(pa.float32(), 2))}
    )
    ds = lance.write_dataset(table, tmp_path)
    with pytest.raises(ValueError, match="exactly one vector index"):
        find_duplicate_pairs(ds, "vector", 0)
    ds = ds.create_index(
        "vector",
        "IVF_FLAT",
        num_partitions=1,
        ivf_centroids=np.ones((1, 2), dtype=np.float32),
    )
    segment = ds.describe_indices()[0].segments[0].uuid
    for threshold in [float("nan"), float("inf"), -float("inf")]:
        with pytest.raises(ValueError, match="must be finite"):
            find_duplicate_pairs(ds, "vector", threshold)
    with pytest.raises(ValueError, match="partition_id"):
        find_duplicate_pairs_in_partition(ds, "vector", segment, 1, 0)
    with pytest.raises(ValueError, match="segment_id"):
        find_duplicate_pairs_in_partition(ds, "vector", "not-a-uuid", 0, 0)
    with pytest.raises(ValueError, match="not an active segment"):
        find_duplicate_pairs_in_partition(
            ds, "vector", "00000000-0000-0000-0000-000000000000", 0, 0
        )
    reader = find_duplicate_pairs(ds, "vector", 0)
    ds.delete("true")
    assert reader.read_all().num_rows == 1
    reader.close()
    with find_duplicate_pairs(ds, "vector", 0) as empty:
        assert empty.schema.names == ["row_id_a", "row_id_b", "distance"]
        assert empty.read_all().num_rows == 0
    current = lance.write_dataset(table, tmp_path, mode="append")
    with pytest.raises(ValueError, match="does not cover all fragments"):
        find_duplicate_pairs(current, "vector", 0)


@pytest.mark.parametrize("memory_limit", [0, 256 * 1024 * 1024])
@pytest.mark.parametrize("max_concurrency", [1, 4])
def test_duplicate_pairs_cross_batch_order(tmp_path, memory_limit, max_concurrency):
    # One anchor has matches on both sides of the vector batch boundary.
    n = 8193
    vectors = np.column_stack((np.arange(n), np.ones(n))).astype(np.float32)
    vectors[-1] = vectors[0]
    vectors[1] = vectors[0]
    vectors[2] = vectors[0]
    ds = lance.write_dataset(
        pa.table(
            {
                "id": np.arange(n),
                "vector": pa.array(vectors.tolist(), pa.list_(pa.float32(), 2)),
            }
        ),
        tmp_path,
    )
    ds = ds.create_index(
        "vector",
        "IVF_FLAT",
        num_partitions=1,
        ivf_centroids=np.ones((1, 2), dtype=np.float32),
    )
    # Keep the physical index boundary while limiting the number of anchors
    # scored by this order/spill test. Removed IDs must not appear in pairs.
    ds.delete(f"id >= 3 AND id < {n - 1}")
    with find_duplicate_pairs(
        ds,
        "vector",
        0,
        memory_limit=memory_limit,
        max_concurrency=max_concurrency,
    ) as reader:
        # Exhaust all surviving anchors, including the physical partition tail.
        batches = list(reader)
    # Tile order: the (batch 0, batch 0) tile, then (batch 0, batch 1), each
    # row-major and emitted as one batch per job of up to 32 anchor rows.
    assert [batch.num_rows for batch in batches] == [3, 3]
    pairs = pa.Table.from_batches(batches).to_pylist()
    assert [(p["row_id_a"], p["row_id_b"]) for p in pairs] == [
        (0, 1),
        (0, 2),
        (1, 2),
        (0, n - 1),
        (1, n - 1),
        (2, n - 1),
    ]
    with find_duplicate_pairs(ds, "vector", 1e10) as reader:
        assert next(reader).num_rows <= 32 * 8192
        # Closing a partially consumed reader cancels further enumeration.
    assert ds.count_rows() == 4


@pytest.mark.parametrize("memory_limit", [0, 256 * 1024 * 1024])
def test_duplicate_pairs_parallel_scoring_preserves_order(tmp_path, memory_limit):
    # 4096-d vectors cut the 8K batch maximum to 512 rows, so 1027 rows give
    # three vector batches and every 32-row job exceeds the inline threshold.
    # Keeping three rows per batch makes each of the six tiles a single cheap
    # CPU-pool job, and with forced spill the next batch load has to wait for
    # earlier jobs to release their decoded batches.
    n, batch_rows, dim = 1027, 512, 4096
    positions = np.arange(n)
    vectors = np.repeat((positions % 2)[:, None], dim, axis=1).astype(np.float32)
    table = pa.table(
        {
            "id": positions,
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1)), dim
            ),
        }
    )
    ds = lance.write_dataset(table, tmp_path, max_rows_per_file=300)
    ds = ds.create_index(
        "vector",
        "IVF_FLAT",
        num_partitions=1,
        ivf_centroids=np.ones((1, dim), dtype=np.float32),
    )
    kept = [b * batch_rows + r for b in range(n // batch_rows + 1) for r in range(3)]
    ds.delete(f"id NOT IN ({', '.join(map(str, kept))})")
    row_ids = {
        r["id"]: r["_rowid"]
        for r in ds.to_table(columns=["id"], with_row_id=True).to_pylist()
    }
    # Tile order: anchor batch, candidate batch, then row-major positions.
    expected = sorted(
        ((a, b) for i, a in enumerate(kept) for b in kept[i + 1 :] if a % 2 == b % 2),
        key=lambda p: (p[0] // batch_rows, p[1] // batch_rows, p[0], p[1]),
    )
    with find_duplicate_pairs(
        ds, "vector", 0, max_concurrency=1, memory_limit=memory_limit
    ) as reader:
        batches = list(reader)
    # One output batch per tile: 1 pair on the diagonal, 5 off it.
    assert [batch.num_rows for batch in batches] == [1, 5, 5, 1, 5, 1]
    serial = pa.Table.from_batches(batches)
    assert [(p["row_id_a"], p["row_id_b"]) for p in serial.to_pylist()] == [
        (row_ids[a], row_ids[b]) for a, b in expected
    ]
    with find_duplicate_pairs(
        ds, "vector", 0, max_concurrency=8, memory_limit=memory_limit
    ) as reader:
        assert reader.read_all().equals(serial)
    segment = ds.describe_indices()[0].segments[0].uuid
    with find_duplicate_pairs_in_partition(
        ds,
        "vector",
        segment,
        0,
        0,
        max_concurrency=4,
        memory_limit=memory_limit,
    ) as reader:
        assert reader.read_all().equals(serial)
    with pytest.raises(ValueError, match="max_concurrency must be positive"):
        find_duplicate_pairs(ds, "vector", 0, max_concurrency=0)


@pytest.mark.parametrize("bits", [4, 8])
def test_duplicate_pairs_exhaust_pq_partition(tmp_path, bits):
    # Fully consume the zero-output case that exposed per-anchor source reads.
    rng = np.random.default_rng(9493)
    vectors = rng.normal(size=(1025, 64)).astype(np.float32)
    ds = lance.write_dataset(
        pa.table(
            {
                "vector": pa.array(vectors.tolist(), pa.list_(pa.float32(), 64)),
            }
        ),
        tmp_path,
    )
    ds = ds.create_index(
        "vector",
        "IVF_PQ",
        num_partitions=1,
        ivf_centroids=np.zeros((1, 64), dtype=np.float32),
        num_bits=bits,
        num_sub_vectors=16,
        pq_codebook=rng.normal(size=(16, 2**bits, 4)).astype(np.float32),
    )
    with find_duplicate_pairs(ds, "vector", 0.0) as reader:
        assert reader.read_all().num_rows == 0


def test_duplicate_pairs_independent_segments(tmp_path):
    vectors = [[1.0, 0.0]] * 4
    ds = lance.write_dataset(
        pa.table({"vector": pa.array(vectors, pa.list_(pa.float32(), 2))}),
        tmp_path,
        max_rows_per_file=2,
    )
    segments = [
        ds.create_index_uncommitted(
            "vector",
            "IVF_FLAT",
            name="vector_idx",
            train=True,
            fragment_ids=[fragment.fragment_id],
            num_partitions=1,
            ivf_centroids=np.array([[i + 1, 0]], dtype=np.float32),
        )
        for i, fragment in enumerate(ds.get_fragments())
    ]
    ds = ds.commit_existing_index_segments("vector_idx", "vector", segments)
    pairs = find_duplicate_pairs(ds, "vector", 0).read_all()
    assert pairs.num_rows == 2  # Cross-segment pairs are intentionally omitted.
    expected = []
    for segment in ds.describe_indices()[0].segments:
        expected.extend(
            find_duplicate_pairs_in_partition(ds, "vector", segment.uuid, 0, 0)
            .read_all()
            .to_pylist()
        )
    assert pairs.to_pylist() == expected


@pytest.mark.parametrize("stable_ids", [False, True])
def test_duplicate_pairs_segment_ownership(tmp_path, stable_ids):
    ds = lance.write_dataset(
        pa.table({"vector": pa.array([[1.0, 0.0]] * 4, pa.list_(pa.float32(), 2))}),
        tmp_path,
        max_rows_per_file=2,
        enable_stable_row_ids=stable_ids,
    )
    ds = ds.create_index(
        "vector",
        "IVF_FLAT",
        num_partitions=1,
        ivf_centroids=np.ones((1, 2), dtype=np.float32),
    )
    fragment = ds.get_fragment(1)
    row_ids = fragment.to_table(with_row_id=True)["_rowid"]
    updated, fields = fragment.update_columns(
        pa.table(
            {
                "_rowid": row_ids,
                "vector": pa.array([[10.0, 0.0]] * 2, pa.list_(pa.float32(), 2)),
            }
        )
    )
    ds = lance.LanceDataset.commit(
        ds.uri,
        lance.LanceOperation.Update(
            updated_fragments=[updated],
            fields_modified=fields,
        ),
        read_version=ds.version,
    )
    ds.optimize.optimize_indices(num_indices_to_merge=0)
    ds = lance.dataset(ds.uri)
    pairs = find_duplicate_pairs(ds, "vector", 0).read_all().to_pylist()
    assert len(pairs) == 2
    current = {
        r["_rowid"]: r["vector"] for r in ds.to_table(with_row_id=True).to_pylist()
    }
    assert all(current[p["row_id_a"]] == current[p["row_id_b"]] for p in pairs)
    assert len({p[k] for p in pairs for k in ["row_id_a", "row_id_b"]}) == 4


def test_duplicate_pairs_hamming(tmp_path):
    ds = lance.write_dataset(_binary_vectors_table(), tmp_path)
    ds = ds.create_index(
        "vector",
        "IVF_FLAT",
        metric="hamming",
        num_partitions=1,
        ivf_centroids=pa.array([[0] * 4], pa.list_(pa.uint8(), 4)),
    )
    pairs = find_duplicate_pairs(ds, "vector", 2).read_all().to_pylist()
    assert {(p["row_id_a"], p["row_id_b"]): p["distance"] for p in pairs} == {
        (0, 1): 2.0,
        (1, 2): 2.0,
    }


@pytest.mark.parametrize("index_type", ["IVF_PQ", "IVF_SQ"])
def test_duplicate_pairs_exactly_representable_codes(tmp_path, index_type):
    vectors = np.array([[0, 0, 2, 2], [3, 3, 4, 4], [8, 8, 6, 6]], dtype=np.float32)
    params = {}
    if index_type == "IVF_PQ":
        # Every source subvector is an exact codebook entry.
        codebook = np.repeat(np.arange(16, dtype=np.float32), 2).reshape(16, 2)
        params = dict(
            num_bits=4, num_sub_vectors=2, pq_codebook=np.stack([codebook] * 2)
        )
    else:
        # Endpoints of the scalar quantizer are exactly representable.
        vectors = np.array([[0, 255], [255, 0], [0, 0]], dtype=np.float32)
    dim = vectors.shape[1]
    ds = lance.write_dataset(
        pa.table(
            {
                "vector": pa.array(vectors.tolist(), pa.list_(pa.float32(), dim)),
            }
        ),
        tmp_path,
    )
    ds = ds.create_index(
        "vector",
        index_type,
        num_partitions=1,
        ivf_centroids=np.zeros((1, dim), dtype=np.float32),
        **params,
    )
    result = find_duplicate_pairs(ds, "vector", 1e6).read_all().to_pylist()
    assert len(result) == 3
    for pair in result:
        expected = np.sum((vectors[pair["row_id_a"]] - vectors[pair["row_id_b"]]) ** 2)
        assert pair["distance"] == pytest.approx(expected)


@pytest.mark.parametrize("stable_ids", [False, True])
def test_duplicate_pairs_after_compaction(tmp_path, stable_ids):
    ds = lance.write_dataset(
        pa.table(
            {
                "id": range(4),
                "vector": pa.array([[1.0, 0.0]] * 4, pa.list_(pa.float32(), 2)),
            }
        ),
        tmp_path,
        max_rows_per_file=2,
        enable_stable_row_ids=stable_ids,
    )
    ds = ds.create_index(
        "vector",
        "IVF_FLAT",
        num_partitions=1,
        ivf_centroids=np.ones((1, 2), dtype=np.float32),
    )
    ds.delete("id = 1")
    ds.optimize.compact_files(
        target_rows_per_fragment=10,
        defer_index_remap=not stable_ids,
        materialize_deletions=True,
    )
    current = ds.to_table(with_row_id=True)["_rowid"].to_pylist()
    result = find_duplicate_pairs(ds, "vector", 0).read_all().to_pylist()
    assert len(result) == 3
    assert {p[k] for p in result for k in ["row_id_a", "row_id_b"]} == set(current)


@pytest.mark.parametrize(
    "index_type,bits",
    [
        ("IVF_PQ", 4),
        ("IVF_PQ", 8),
        ("IVF_RQ", 1),
        ("IVF_RQ", 5),
        ("IVF_RQ", 9),
    ],
)
def test_duplicate_pairs_quantized_batch_boundary(tmp_path, index_type, bits):
    # A 8192-row batch followed by one row exercises partial transposed PQ
    # reads and the packed RQ tail. Deleting the rows in between keeps the
    # scored anchors (and the output) small while staging every row.
    rng = np.random.default_rng(83)
    vectors = rng.normal(size=(8193, 8)).astype(np.float32)
    vectors[-1] = vectors[0]
    ds = lance.write_dataset(
        pa.table(
            {
                "id": np.arange(len(vectors)),
                "vector": pa.array(vectors.tolist(), pa.list_(pa.float32(), 8)),
            }
        ),
        tmp_path,
    )
    params = {"num_bits": bits}
    if index_type == "IVF_PQ":
        params.update(
            num_sub_vectors=4,
            pq_codebook=rng.normal(size=(4, 2**bits, 2)).astype(np.float32),
        )
    ds = ds.create_index(
        "vector",
        index_type,
        num_partitions=1,
        ivf_centroids=np.ones((1, 8), dtype=np.float32),
        **params,
    )
    ds.delete("id >= 1 AND id < 8192")
    results = []
    for memory_limit in [0, 256 * 1024 * 1024]:
        with find_duplicate_pairs(
            ds, "vector", 1e6, memory_limit=memory_limit
        ) as reader:
            results.append(reader.read_all())
    # Spilled and in-memory staging give bit-identical distances.
    assert results[0].equals(results[1])
    assert results[0].to_pylist() == [
        {"row_id_a": 0, "row_id_b": 8192, "distance": 0.0}
    ]


def test_duplicate_pairs_partition_scope(tmp_path):
    ds = lance.write_dataset(
        pa.table(
            {
                "vector": pa.array(
                    [[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]],
                    pa.list_(pa.float32(), 2),
                ),
            }
        ),
        tmp_path,
    )
    ds = ds.create_index(
        "vector",
        "IVF_FLAT",
        num_partitions=3,
        ivf_centroids=np.array([[0, 0], [10, 0], [100, 0]], dtype=np.float32),
    )
    pairs = find_duplicate_pairs(ds, "vector", 1e6).read_all().to_pylist()
    assert {(p["row_id_a"], p["row_id_b"]) for p in pairs} == {(0, 1), (2, 3)}
    segment = ds.describe_indices()[0].segments[0].uuid
    assert (
        find_duplicate_pairs_in_partition(ds, "vector", segment, 2, 1e6)
        .read_all()
        .num_rows
        == 0
    )


def test_dict():
    ids, vectors = _create_data()
    dd = dict(zip(ids, vectors))
    tbl = vec_to_table(dd)
    expected = [pa.array(ids), _to_vec(vectors)]
    assert_table(tbl, expected)

    new_tbl = vec_to_table(dd, names=["foo", "bar"])
    assert new_tbl.column_names == ["foo", "bar"]

    with pytest.raises(ValueError):
        ids, vectors = _create_bad_dims()
        dd = dict(zip(ids, vectors))
        vec_to_table(dd)


def test_list():
    _, vectors = _create_data()
    tbl = vec_to_table(vectors)
    expected = [_to_vec(vectors)]
    assert_table(tbl, expected)

    with pytest.raises(ValueError):
        _, vectors = _create_bad_dims()
        vec_to_table(vectors)


def test_ndarray():
    _, vectors = _create_data()
    tbl = vec_to_table(np.array(vectors))
    expected = [_to_vec(vectors)]
    assert_table(tbl, expected)

    with pytest.raises(ValueError):
        _, vectors = _create_bad_dims()
        vec_to_table(np.array(vectors))


def assert_table(tbl, expected_arrays, names=None):
    if names is None:
        if len(expected_arrays) == 1:
            names = ["vector"]
        else:
            names = ["id", "vector"]

    for i, n in enumerate(names):
        assert_array_eq(tbl[n], expected_arrays[i])


def assert_array_eq(left: pa.Array, right: pa.Array):
    if isinstance(left, pa.ChunkedArray):
        left = left.combine_chunks()
    if isinstance(right, pa.ChunkedArray):
        right = right.combine_chunks()
    if pa.types.is_float32(left.type):
        assert np.all(
            np.abs(
                left.to_numpy(zero_copy_only=False)
                - right.to_numpy(zero_copy_only=False)
            )
            < 1e-6
        )
    if pa.types.is_fixed_size_list(left.type):
        assert_array_eq(left.values, right.values)
    else:
        assert np.all(left.to_numpy(False) == right.to_numpy(False))


def _create_data():
    ids = list(range(10))
    vectors = np.random.randn(10, 8)
    return ids, vectors


def _create_bad_dims():
    ids = list(range(10))
    vectors = [np.random.randn(8) for _ in ids]
    vectors[5] = np.random.randn(5)
    return ids, vectors


def _to_vec(lst):
    return pa.FixedSizeListArray.from_arrays(
        pa.array(np.array(lst).ravel(), type=pa.float32()), list_size=8
    )


def _binary_vectors_table():
    vectors = pa.FixedSizeListArray.from_arrays(
        pa.array(
            [
                0x0F,
                0,
                0,
                0,
                0x03,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            type=pa.uint8(),
        ),
        list_size=4,
    )
    ids = pa.array([0, 1, 2], type=pa.int32())
    return pa.Table.from_arrays([ids, vectors], names=["id", "vector"])


def test_binary_vectors_default_hamming(tmp_path):
    dataset = lance.write_dataset(_binary_vectors_table(), tmp_path / "bin")
    scanner = dataset.scanner(
        nearest={"column": "vector", "q": [0x0F, 0, 0, 0], "k": 3}
    )

    plan = scanner.analyze_plan()
    assert "metric=hamming" in plan

    tbl = scanner.to_table()
    assert tbl["id"].to_pylist() == [0, 1, 2]
    assert tbl["_distance"].to_pylist() == [0.0, 2.0, 4.0]


def test_binary_vectors_invalid_metric(tmp_path):
    dataset = lance.write_dataset(_binary_vectors_table(), tmp_path / "bin")
    with pytest.raises(
        ValueError, match="Distance type l2 does not support .*UInt8 vectors"
    ):
        dataset.scanner(
            nearest={
                "column": "vector",
                "q": [0x0F, 0, 0, 0],
                "k": 1,
                "metric": "l2",
            }
        ).to_table()


def _hash_table(hashes):
    """Build a table with a ``hash`` column of FixedSizeList<UInt8, N>.

    ``hashes`` is a list of byte sequences, one per row. The byte width must
    be a positive multiple of 8.
    """
    byte_width = len(hashes[0])
    assert byte_width > 0 and byte_width % 8 == 0
    assert all(len(row) == byte_width for row in hashes)
    flat = [byte for row in hashes for byte in row]
    values = pa.FixedSizeListArray.from_arrays(
        pa.array(flat, type=pa.uint8()), list_size=byte_width
    )
    return pa.Table.from_arrays([values], names=["hash"])


@pytest.mark.parametrize("byte_width", [8, 16])
def test_hamming_clustering_for_sample(tmp_path, byte_width):
    hash_a = [0] * byte_width
    hash_b = [0] * (byte_width - 8) + [255] + [0] * 7  # 8 bits from hash_a
    hash_c = list(range(1, byte_width + 1))  # far from both
    # Rows 0,1,2 share hash_a; rows 3,4 share hash_b; row 5 is unique.
    table = _hash_table([hash_a, hash_a, hash_a, hash_b, hash_b, hash_c])
    dataset = lance.write_dataset(table, tmp_path / "hashes")

    # threshold 0 => only exact-match hashes cluster together. Full scan
    # (sample_size=None) yields deterministic row ids 0..5.
    result = hamming_clustering_for_sample(dataset, "hash", None, 0).read_all()

    clusters = {
        rep: sorted(dups)
        for rep, dups in zip(
            result["representative"].to_pylist(),
            result["duplicates"].to_pylist(),
        )
    }
    # Singleton row 5 is not emitted as a cluster.
    assert clusters == {0: [1, 2], 3: [4]}


@pytest.mark.parametrize("byte_width", [8, 16])
def test_hamming_clustering_multi_segment(tmp_path, byte_width):
    mask = (1 << 64) - 1

    def hash_bytes(value):
        if byte_width == 8:
            lanes = [(value * 0x9E3779B97F4A7C15) & mask]
        else:
            # Adjacent logical values share the first 64-bit lane and differ in
            # later lanes, so threshold-0 clustering must compare every lane.
            lanes = [
                ((value // 2) * 0x9E3779B97F4A7C15) & mask,
                ((value * 0xD6E8FEB86659FD93) ^ 0xA5A5A5A5A5A5A5A5) & mask,
            ]
        return [
            byte for lane_value in lanes for byte in lane_value.to_bytes(8, "little")
        ]

    # 25 distinct hash values, two copies each; the same table is written to
    # fragment 0 and appended as fragment 1.
    values = [i // 2 for i in range(50)]
    table = _hash_table([hash_bytes(value) for value in values])
    dataset = lance.write_dataset(table, tmp_path / "hashes")
    dataset.create_index(
        "hash", index_type="IVF_FLAT", num_partitions=4, metric="hamming"
    )
    dataset = lance.write_dataset(table, tmp_path / "hashes", mode="append")
    # Optimizing with merge disabled creates a delta segment for fragment 1.
    dataset.optimize.optimize_indices(num_indices_to_merge=0)

    index = dataset.describe_indices()[0]
    assert len(index.segments) == 2

    infos = get_ivf_partition_info(dataset, index.name)
    assert sum(info["size"] for info in infos) == 100

    # All four copies of each value cluster together across both fragments.
    frag1_start = 1 << 32
    clusters = []
    for info in infos:
        result = hamming_clustering_for_ivf_partition(
            dataset, index.name, info["partition_id"], 0
        ).read_all()
        clusters.extend(
            zip(
                result["representative"].to_pylist(),
                result["duplicates"].to_pylist(),
            )
        )
    assert len(clusters) == 25
    for representative, duplicates in clusters:
        assert representative < frag1_start
        assert len(duplicates) == 3
        assert any(dup >= frag1_start for dup in duplicates)

    # Selecting the fragment-0 segment reproduces the single-segment scope.
    first_segment = next(
        segment for segment in index.segments if segment.fragment_ids == {0}
    )
    infos = get_ivf_partition_info(
        dataset, index.name, index_segments=[first_segment.uuid]
    )
    assert sum(info["size"] for info in infos) == 50
    num_selected_clusters = 0
    for info in infos:
        result = hamming_clustering_for_ivf_partition(
            dataset,
            index.name,
            info["partition_id"],
            0,
            index_segments=[first_segment.uuid],
        ).read_all()
        for duplicates in result["duplicates"].to_pylist():
            num_selected_clusters += 1
            assert duplicates == [dup for dup in duplicates if dup < frag1_start]
            assert len(duplicates) == 1
    assert num_selected_clusters == 25

    with pytest.raises(ValueError, match="invalid index segment uuid"):
        get_ivf_partition_info(dataset, index.name, index_segments=["not-a-uuid"])
    with pytest.raises(TypeError, match="str or uuid.UUID"):
        get_ivf_partition_info(dataset, index.name, index_segments=[123])
    with pytest.raises(TypeError, match="not a single"):
        get_ivf_partition_info(dataset, index.name, index_segments=first_segment.uuid)


@pytest.mark.parametrize("index_type", ["IVF_PQ", "IVF_SQ"])
def test_duplicate_pairs_quantized_cosine_renormalizes(tmp_path, index_type):
    # A deliberately coarse quantizer distinguishes 1 - cos of the
    # renormalized reconstructions from their (unbounded) squared L2.
    vectors = np.array([[1, 0], [0, 1], [-1, 0], [1, 0]], dtype=np.float32)
    params = {}
    if index_type == "IVF_PQ":
        codebook = np.tile(np.array([[2, 0], [0, 3], [-2, 0]], np.float32), (86, 1))[
            :256
        ]
        params = dict(num_bits=8, num_sub_vectors=1, pq_codebook=codebook[None])
        # Codewords [2, 0] and [0, 3] are orthogonal although 13 apart.
        reconstructed = np.array([[2, 0], [0, 3]], np.float64)
    else:
        # Bounds [-1, 1]: 0 truncates to code 127, reconstructed as -1/255.
        reconstructed = np.array([[1, -1 / 255], [-1 / 255, 1]], np.float64)
    a, b = reconstructed
    expected = 1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    ds = lance.write_dataset(
        pa.table({"vector": pa.array(vectors.tolist(), pa.list_(pa.float32(), 2))}),
        tmp_path,
    )
    ds = ds.create_index(
        "vector",
        index_type,
        metric="cosine",
        num_partitions=1,
        ivf_centroids=np.zeros((1, 2), np.float32),
        **params,
    )
    pairs = find_duplicate_pairs(ds, "vector", 100).read_all().to_pylist()
    distances = {
        tuple(sorted((p["row_id_a"], p["row_id_b"]))): p["distance"] for p in pairs
    }
    assert len(distances) == 6
    assert distances[(0, 1)] == pytest.approx(expected, abs=1e-6)
    # Identical codes score exactly zero; opposite vectors reach the bound 2.
    assert distances[(0, 3)] == 0
    assert distances[(0, 2)] == pytest.approx(2.0, abs=1e-4)
    assert all(0 <= d <= 2 for d in distances.values())
    # Threshold filtering must use the same native score as enumeration.
    selected = (
        find_duplicate_pairs(ds, "vector", expected - 1e-3).read_all().to_pylist()
    )
    assert (0, 1) not in {
        tuple(sorted((p["row_id_a"], p["row_id_b"]))) for p in selected
    }
