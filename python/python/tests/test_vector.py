# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import lance
import numpy as np
import pyarrow as pa
import pytest
from lance.vector import (
    get_ivf_partition_info,
    hamming_clustering_for_ivf_partition,
    hamming_clustering_for_sample,
    vec_to_table,
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
