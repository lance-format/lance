# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Check the calibration contract independently of native probe implementation."""

import io
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest
from calibrate import K_BUCKETS, calibrate, evaluate, scale_at


def test_k_buckets_and_heldout_isolation(tmp_path):
    groups = []
    for k in K_BUCKETS:
        # With Auto=10, the small-k group needs three partitions, while the
        # other groups need seven for >=81% and nine for >=91% coverage.
        ranks = (
            np.array([[2], [3], [2], [3]])
            if k == 1
            else np.tile([1] * 8 + [7, 9], (4, k // 10))
        )
        path = tmp_path / f"rows-{k}.npz"
        np.savez(
            path,
            auto=np.full(4, 10),
            ranks=ranks,
            query_ids=np.arange(4),
            split=[0, 0, 1, 1],
        )
        groups.append({"corpus": "rows", "metric": "l2", "k": k, "path": path.name})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"groups": groups}))
    frozen = tmp_path / "frozen.json"
    calibrate(manifest, frozen)
    profiles = json.loads(frozen.read_text())["profiles"]["l2"]
    assert [p["scale"] for p in profiles["1"]] == [0.201, 0.201]
    for k in K_BUCKETS[1:]:
        assert [p["scale"] for p in profiles[str(k)]] == [0.601, 0.801]

    output = tmp_path / "evaluation.json"
    evaluate(manifest, frozen, output)
    rows = json.loads(output.read_text())["results"]
    assert len(rows) == 3 * len(K_BUCKETS)
    assert all(row["queries"] == 2 for row in rows)
    assert all(row["mean_routing_recall"] >= row["target"] for row in rows)

    # Changing only evaluation neighbors cannot influence fitted parameters.
    for group in groups:
        path = tmp_path / group["path"]
        with np.load(path) as archive:
            arrays = dict(archive)
        arrays["ranks"][2:] = 10
        np.savez(path, **arrays)
    second = tmp_path / "second.json"
    calibrate(manifest, second)
    assert json.loads(second.read_text())["profiles"] == {"l2": profiles}


def test_k_interpolation_preserves_anchors():
    profiles = {
        "10": [{"scale": 0.1}, {"scale": 0.4}],
        "1000": [{"scale": 0.4}, {"scale": 1.0}],
    }
    assert scale_at(profiles, 10, 0) == 0.1
    assert scale_at(profiles, 1000, 1) == 1.0
    assert scale_at(profiles, 100, 0) == pytest.approx(0.2)
    assert scale_at(profiles, 100, 1) == pytest.approx(math.sqrt(0.4))
    with pytest.raises(ValueError, match="outside calibrated anchors"):
        scale_at(profiles, 9, 0)


def test_anchor_selection_uses_calibration_quality_and_cost(tmp_path):
    groups = []
    # k=20 requires an extra knot to reach the target. k=50 already reaches it
    # by interpolation, but needs a knot to avoid twice the necessary probes.
    for k, rank in [(11, 2), (20, 9), (50, 2), (100, 6)]:
        path = tmp_path / f"k-{k}.npz"
        np.savez(
            path,
            auto=np.full(4, 10),
            query_ids=np.arange(4),
            split=[0, 0, 1, 1],
            ranks=np.full((4, k), rank),
        )
        groups.append({"corpus": "rows", "metric": "l2", "k": k, "path": path.name})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"groups": groups, "mandatory_k_anchors": [11, 100]})
    )
    first = tmp_path / "first.json"
    calibrate(manifest, first)
    expected = json.loads(first.read_text())
    assert expected["k_anchors"] == [11, 20, 50, 100]
    for group in groups:
        path = tmp_path / group["path"]
        with np.load(path) as values:
            arrays = dict(values)
        arrays["ranks"][2:] = 10
        np.savez(path, **arrays)
    second = tmp_path / "second.json"
    calibrate(manifest, second)
    actual = json.loads(second.read_text())
    assert actual["k_anchors"] == expected["k_anchors"]
    assert actual["profiles"] == expected["profiles"]


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
@pytest.mark.parametrize("fragment_size", [2, 4])
def test_exact_truth_merges_source_batches(tmp_path, metric, fragment_size):
    import lance
    from prepare_large import direct_truth, exact_truth

    vectors = [[2, 1], [1, 4], [-3, 2], [4, -1], [1, 1], [-2, -1]]
    queries = np.array([[1, 2], [-1, 3]], dtype=np.float32)
    dataset = lance.write_dataset(
        pa.table({"vector": pa.array(vectors, type=pa.list_(pa.float32(), 2))}),
        tmp_path / "rows.lance",
        max_rows_per_file=fragment_size,
        max_rows_per_group=fragment_size,
    )

    def distance(query, vector):
        if metric == "l2":
            return math.fsum((float(a) - b) ** 2 for a, b in zip(query, vector))
        product = math.fsum(float(a) * b for a, b in zip(query, vector))
        if metric == "cosine":
            product /= math.hypot(*query) * math.hypot(*vector)
        return -product

    expected = np.array(
        [
            sorted(range(len(vectors)), key=lambda i: (distance(query, vectors[i]), i))[
                :3
            ]
            for query in queries
        ]
    )
    ids, scores, previous_scores = exact_truth(
        dataset,
        metric,
        queries,
        3,
        batch_size=2,
        query_block=1,
        previous_truth=expected[:, :2],
    )
    np.testing.assert_array_equal(ids, expected)
    np.testing.assert_allclose(scores[:, :2], previous_scores, rtol=0, atol=1e-14)
    for position, query in enumerate(queries):
        direct_ids, direct_scores = direct_truth(dataset, metric, query, 3)
        np.testing.assert_array_equal(direct_ids, expected[position])
        np.testing.assert_allclose(direct_scores, scores[position], rtol=0, atol=1e-14)


def test_float32_reference_does_not_override_float64_truth(tmp_path):
    import lance
    from prepare_large import exact_truth, verify_prefix

    # Float32 normalization rounds both vector norms to one. Float64 ranking
    # distinguishes them, so a Float32 tie can legitimately choose the other ID.
    vectors = [[1, 0.0002], [1, 0.0001]]
    dataset = lance.write_dataset(
        pa.table({"vector": pa.array(vectors, type=pa.list_(pa.float32(), 2))}),
        tmp_path / "rows.lance",
    )
    queries = np.array([[1, 0]], dtype=np.float32)
    old_positions = np.array([[0]])
    ids, scores, old_scores = exact_truth(
        dataset, "cosine", queries, 1, previous_truth=old_positions
    )
    audit = verify_prefix(
        dataset, "cosine", queries, ids, scores, old_positions, old_scores
    )
    assert audit["max_score_error"] > 1e-10
    assert audit["independent_check"]["max_score_error"] < 1e-14
    assert audit["independent_check"]["strict_id_overlap"] == 1.0
    np.testing.assert_array_equal(ids, [[1]])


def test_native_worker_records_binary_ids(tmp_path, monkeypatch):
    import lance
    from native import worker

    dataset = lance.write_dataset(
        pa.table(
            {
                "vector": pa.array(
                    [[1, 0], [0, 1], [2, 0], [0, 2]], type=pa.list_(pa.float32(), 2)
                )
            }
        ),
        tmp_path / "rows.lance",
    )
    dataset.create_index(
        "vector",
        "IVF_FLAT",
        name="flat",
        metric="l2",
        num_partitions=1,
        ivf_centroids=np.array([[0, 0]], dtype=np.float32),
    )
    dataset = lance.dataset(tmp_path / "rows.lance")
    queries = tmp_path / "queries.npz"
    np.savez(
        queries,
        vectors=np.array([[1, 0], [0, 1]], dtype=np.float32),
        query_ids=[0, 1],
        split=[1, 1],
        truth=[[0, 2], [1, 3]],
    )
    datasets = tmp_path / "datasets.json"
    datasets.write_text(
        json.dumps(
            {
                "flat": {
                    "uri": str(tmp_path / "rows.lance"),
                    "version": dataset.version,
                    "rows": 4,
                    "metric": "l2",
                    "index_name": "flat",
                    "index_uuid": dataset.list_indices()[0]["uuid"],
                }
            }
        )
    )
    requests = [
        {"op": "setup", "index": "flat", "cache": "cold"},
        {
            "op": "query",
            "position": 0,
            "k": 2,
            "filtered": False,
            "refine": 0,
            "effort": 1.0,
            "cache": "cold",
        },
        {
            "op": "query",
            "position": 1,
            "k": 2,
            "filtered": False,
            "refine": 0,
            "effort": 1.0,
            "cache": "cold",
        },
    ]
    output = io.StringIO()
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(map(json.dumps, requests))))
    monkeypatch.setattr("sys.stdout", output)
    ids_path = tmp_path / "returned.ids"
    worker(
        SimpleNamespace(
            runtime=Path(lance.__file__).parent.parent,
            queries=queries,
            truth=None,
            datasets=datasets,
            ids=ids_path,
            index_cache_gib=1,
        )
    )
    ready, first, second = map(json.loads, output.getvalue().splitlines())
    assert ready["ready"]
    assert first["recall"] == second["recall"] == 1.0
    assert first["partitions"] == second["partitions"] == 1
    assert first["ids_offset"] == 0 and second["ids_offset"] == 16
    np.testing.assert_array_equal(np.fromfile(ids_path, dtype="<u8"), [0, 2, 1, 3])
