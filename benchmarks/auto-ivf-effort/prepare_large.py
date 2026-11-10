# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Extend frozen query sets to exact top-100000 truth and k-bucket routing inputs.

The input JSON supplies immutable dataset versions and the existing query/split
arrays. Each corpus is prepared independently. Native comparisons must use the
recorded index, row IDs and query vectors from this output.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
from calibrate import K_BUCKETS, digest, k_bucket
from lance._dataset.index import VectorIndexReader

MANDATORY_K_ANCHORS = (
    1,
    2,
    10,
    11,
    100,
    101,
    200,
    201,
    500,
    501,
    1000,
    1001,
    2000,
    5000,
    10000,
    10001,
    20000,
    50000,
    100000,
)
K_VALUES = tuple(
    sorted(
        MANDATORY_K_ANCHORS
        + (5, 20, 50, 150, 300, 750, 1500, 3000, 7500, 15000, 30000, 75000)
    )
)
# The unchanged Auto profiles at the direct-parent source aff6f1d1.
PROFILES = {
    "l2": list(
        zip(
            [0.2175, 0.265, 0.33, 0.375, 0.3725, 0.4175, 0.57, 0.75],
            [5, 6, 11, 15, 22, 29, 98, 463],
            [19, 24, 38, 56, 81, 93, 238, 704],
        )
    ),
    "cosine": list(
        zip(
            [0.235, 0.2875, 0.38, 0.415, 0.45, 0.47, 0.6475, 0.58],
            [3, 8, 7, 18, 28, 34, 104, 738],
            [50, 77, 106, 156, 192, 248, 447, 958],
        )
    ),
    "dot": list(
        zip(
            [0.14, 0.055, 0.0625, 0.0675, 0.07, 0.0725, 0.0825, 0.21],
            [56, 144, 200, 211, 244, 279, 517, 1009],
            [112, 432, 768, 833, 971, 1092, 1824, 2528],
        )
    ),
}


def emit(event, **values):
    print(json.dumps({"time": time.time(), "event": event, **values}), flush=True)


def matrix(column):
    if hasattr(column, "combine_chunks"):
        column = column.combine_chunks()
    return column.values.to_numpy().reshape(len(column), column.type.list_size)


def exact_truth(
    dataset,
    metric,
    queries,
    k,
    batch_size=262144,
    query_block=1024,
    previous_truth=None,
):
    """Exhaustive Float64 ranking; retained equal scores are ordered by row position.

    Cutoff ties follow NumPy argpartition. Alternative tied IDs still count as
    strict-ID misses. Query norms are omitted from L2 scores since they cannot
    change the ranking within a query.
    """
    if metric not in PROFILES or not 1 <= k <= dataset.count_rows():
        raise ValueError(f"Invalid oracle metric={metric}, k={k}")
    queries = queries.astype(np.float64)
    if not np.isfinite(queries).all():
        raise ValueError("Oracle queries must be finite")
    if metric == "cosine":
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Cosine oracle queries must have positive norms")
        queries /= norms
    best_scores = np.full((len(queries), k), np.inf)
    best_ids = np.full((len(queries), k), -1, dtype=np.int64)
    previous_scores = (
        None if previous_truth is None else np.full(previous_truth.shape, np.nan)
    )
    start = 0

    def merge(rows, scores, ids):
        scores = np.concatenate([best_scores[rows], scores], axis=1)
        ids = np.concatenate(
            [best_ids[rows], np.broadcast_to(ids, (len(rows), len(ids)))], axis=1
        )
        keep = np.argpartition(scores, k - 1, axis=1)[:, :k]
        best_scores[rows] = np.take_along_axis(scores, keep, axis=1)
        best_ids[rows] = np.take_along_axis(ids, keep, axis=1)

    with ThreadPoolExecutor(16) as pool:
        for batch in dataset.to_batches(
            columns=["vector"], batch_size=batch_size, scan_in_order=True
        ):
            vectors = matrix(batch["vector"]).astype(np.float64)
            if not np.isfinite(vectors).all():
                raise ValueError(f"Non-finite source vectors at row offset {start}")
            ids = np.arange(start, start + len(vectors), dtype=np.int64)
            if metric == "cosine":
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                if np.any(norms == 0):
                    raise ValueError(f"Zero source vector norm at row offset {start}")
                vectors /= norms
            norms = np.sum(vectors**2, axis=1) if metric == "l2" else None
            for block in range(0, len(queries), query_block):
                rows = np.arange(block, min(block + query_block, len(queries)))
                products = queries[rows] @ vectors.T
                scores = norms[None, :] - 2 * products if metric == "l2" else -products
                if previous_truth is not None:
                    offsets = previous_truth[rows] - start
                    row, column = np.nonzero((offsets >= 0) & (offsets < len(vectors)))
                    previous_scores[rows[row], column] = scores[
                        row, offsets[row, column]
                    ]
                list(
                    pool.map(
                        lambda part: merge(part, scores[part - block], ids),
                        [part for part in np.array_split(rows, 16) if len(part)],
                    )
                )
            start += len(vectors)
            emit("truth_progress", rows=start, total=dataset.count_rows(), k=k)
    assert np.all(best_ids >= 0)
    assert previous_scores is None or np.isfinite(previous_scores).all()
    order = np.lexsort((best_ids, best_scores), axis=1)
    return (
        np.take_along_axis(best_ids, order, axis=1),
        np.take_along_axis(best_scores, order, axis=1),
        previous_scores,
    )


def routing_distances(queries, centroids, metric):
    queries = queries.astype(np.float64)
    centroids = centroids.astype(np.float64)
    if metric == "cosine":
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    products = queries @ centroids.T
    if metric == "dot":
        distances = 1 - products
    else:
        distances = (
            np.sum(queries**2, axis=1)[:, None]
            - 2 * products
            + np.sum(centroids**2, axis=1)[None, :]
        )
        np.maximum(distances, 0, out=distances)
    distances = distances.astype(np.float32)
    order = np.argsort(distances, axis=1, kind="stable")
    return np.take_along_axis(distances, order, axis=1), order


def direct_truth(dataset, metric, query, k):
    """Independent Float64 full sort using direct per-vector formulas."""
    if metric not in PROFILES or not 1 <= k <= dataset.count_rows():
        raise ValueError(f"Invalid oracle metric={metric}, k={k}")
    query = query.astype(np.float64)
    query_norm = np.sqrt(np.einsum("i,i->", query, query))
    scores = np.empty(dataset.count_rows(), dtype=np.float64)
    start = 0
    for batch in dataset.to_batches(
        columns=["vector"], batch_size=65536, scan_in_order=True
    ):
        vectors = matrix(batch["vector"]).astype(np.float64)
        if metric == "l2":
            values = np.sum((vectors - query) ** 2, axis=1) - query_norm**2
        else:
            values = -np.einsum("ij,j->i", vectors, query, optimize=False)
            if metric == "cosine":
                norms = np.sqrt(np.einsum("ij,ij->i", vectors, vectors, optimize=False))
                values /= norms * query_norm
        scores[start : start + len(vectors)] = values
        start += len(vectors)
        if start % 1000000 < len(vectors):
            emit("independent_truth_progress", rows=start, total=len(scores))
    assert start == len(scores) and np.isfinite(scores).all()
    positions = np.argsort(scores, kind="stable")[:k]
    return positions, scores[positions]


def verify_prefix(
    dataset, metric, queries, truth, scores, previous_positions, previous_scores
):
    """Retain old-reference differences and check the worst case independently.

    Archived native Float32 oracles can select different cutoff neighbors from
    Float64 source-vector ranking. Score equality to that older oracle is not a
    correctness criterion for the new one. Strict-ID recall always uses the
    original new oracle IDs, including any tie differences from native results.
    """
    error = np.abs(
        scores[:, : previous_positions.shape[1]] - np.sort(previous_scores, axis=1)
    )
    audit = {
        "previous_k": previous_positions.shape[1],
        "max_score_error": float(np.max(error)),
        "queries_with_score_error_over_1e_10": int(
            np.sum(np.any(error > 1e-10, axis=1))
        ),
        "queries_with_different_ids": int(
            np.sum(
                np.any(
                    truth[:, : previous_positions.shape[1]] != previous_positions,
                    axis=1,
                )
            )
        ),
    }
    if audit["max_score_error"] > 1e-10:
        position, _ = np.unravel_index(np.argmax(error), error.shape)
        ids, reference = direct_truth(
            dataset, metric, queries[position], truth.shape[1]
        )
        np.testing.assert_allclose(scores[position], reference, rtol=1e-12, atol=1e-12)
        audit["independent_check"] = {
            "query_position": int(position),
            "source_rows": dataset.count_rows(),
            "k": truth.shape[1],
            "max_score_error": float(np.max(np.abs(scores[position] - reference))),
            "strict_id_overlap": float(
                len(np.intersect1d(ids, truth[position])) / len(ids)
            ),
        }
    return audit


def prepare(spec, out):
    out.mkdir(parents=True, exist_ok=True)
    frozen_input = {
        "spec": spec,
        "queries_sha256": digest(Path(spec["queries"])),
        "generator_sha256": digest(Path(__file__)),
    }
    input_path = out / "input.json"
    if input_path.exists():
        assert json.loads(input_path.read_text()) == frozen_input, (
            "Resume input changed"
        )
    else:
        input_path.write_text(json.dumps(frozen_input, indent=2) + "\n")
    dataset = lance.dataset(spec["uri"], version=spec["version"])
    assert dataset.count_rows() == spec["rows"]
    with np.load(spec["queries"]) as archive:
        queries, query_ids, split = (
            archive[key] for key in ["vectors", "query_ids", "split"]
        )
        previous_truth = archive["truth"]
    assert set(split) == {0, 1} and len(np.unique(query_ids)) == len(query_ids)
    assert queries.shape == (len(split), spec["dimension"])
    # A vector must not occur on both sides of the frozen split.
    keys = (
        np.ascontiguousarray(queries)
        .view(np.dtype((np.void, queries.dtype.itemsize * queries.shape[1])))
        .ravel()
    )
    assert not np.intersect1d(keys[split == 0], keys[split == 1]).size
    np.savez(out / "queries.npz", vectors=queries, query_ids=query_ids, split=split)

    indices = [
        index for index in dataset.list_indices() if index["name"] == spec["index_name"]
    ]
    assert len(indices) == 1 and indices[0]["uuid"] == spec["index_uuid"], indices
    row_ids = dataset.to_table(columns=[], with_row_id=True, scan_in_order=True)[
        "_rowid"
    ].to_numpy()
    assert np.all(row_ids[1:] > row_ids[:-1])
    if spec["previous_truth_kind"] == "row_id":
        previous_positions = np.searchsorted(row_ids, previous_truth)
        assert np.array_equal(row_ids[previous_positions], previous_truth)
    elif spec["previous_truth_kind"] == "position":
        previous_positions = previous_truth.astype(np.int64)
        assert np.all((previous_positions >= 0) & (previous_positions < len(row_ids)))
    else:
        raise ValueError(f"Unknown previous_truth_kind: {spec['previous_truth_kind']}")
    np.save(out / "row-ids.npy", row_ids)
    centroids = matrix(dataset.centroids(index_name=spec["index_name"]))
    np.save(out / "centroids.npy", centroids)
    membership_path = out / "membership.npy"
    if not membership_path.exists():
        membership = np.full(len(row_ids), -1, dtype=np.int32)
        reader = VectorIndexReader(dataset, spec["index_name"])
        for partition in range(reader.num_partitions()):
            ids = reader.read_partition(partition)["_rowid"].to_numpy()
            positions = np.searchsorted(row_ids, ids)
            assert np.array_equal(row_ids[positions], ids)
            assert np.all(membership[positions] == -1)
            membership[positions] = partition
            if partition % 512 == 0:
                emit("membership", corpus=spec["corpus"], partition=partition)
        assert np.all(membership >= 0)
        np.save(out / "membership.tmp.npy", membership)
        (out / "membership.tmp.npy").replace(membership_path)
    membership = np.load(membership_path, mmap_mode="r")
    truth_path = out / "truth-positions.npy"
    if not truth_path.exists():
        truth, scores, previous_scores = exact_truth(
            dataset, spec["metric"], queries, 100000, previous_truth=previous_positions
        )
        np.save(out / "truth-scores.npy", scores)
        np.save(out / "previous-truth-scores.npy", previous_scores)
        np.save(out / "truth-positions.tmp.npy", truth)
        (out / "truth-positions.tmp.npy").replace(truth_path)
        del truth, scores, previous_scores
    truth = np.load(truth_path, mmap_mode="r")
    assert truth.shape == (len(queries), 100000)
    scores = np.load(out / "truth-scores.npy", mmap_mode="r")
    previous_scores = np.load(out / "previous-truth-scores.npy")
    prefix_audit = verify_prefix(
        dataset,
        spec["metric"],
        queries,
        truth,
        scores,
        previous_positions,
        previous_scores,
    )
    if "independent_check" in prefix_audit:
        position = prefix_audit["independent_check"]["query_position"]
        prefix_audit["independent_check"]["query_id"] = int(query_ids[position])
    (out / "oracle-prefix.json").write_text(json.dumps(prefix_audit, indent=2) + "\n")
    del scores, previous_scores
    truth_ids = np.lib.format.open_memmap(
        out / "truth-rowids.npy", mode="w+", dtype=np.uint64, shape=truth.shape
    )
    for start in range(0, len(queries), 64):
        truth_ids[start : start + 64] = row_ids[truth[start : start + 64]]
    truth_ids.flush()
    del truth_ids
    distances, order = routing_distances(queries, centroids, spec["metric"])
    inverse = np.empty(order.shape, dtype=np.uint32)
    np.put_along_axis(inverse, order, np.arange(1, len(centroids) + 1)[None, :], axis=1)
    ranks = np.lib.format.open_memmap(
        out / "ranks.npy", mode="w+", dtype=np.uint32, shape=truth.shape
    )
    for start in range(0, len(queries), 64):
        ranks[start : start + 64] = np.take_along_axis(
            inverse[start : start + 64], membership[truth[start : start + 64]], axis=1
        )
    ranks.flush()
    groups = []
    for k in K_VALUES:
        margin, floor, cap = PROFILES[spec["metric"]][K_BUCKETS.index(k_bucket(k))]
        widened = distances.astype(np.float64)
        nearest = widened[:, :1]
        scale = np.abs(1 - nearest) if spec["metric"] == "dot" else nearest
        auto = np.clip(
            (widened - nearest <= float(np.float32(margin)) * scale).sum(axis=1),
            floor,
            min(cap, len(centroids)),
        )
        path = out / f"k-{k}.npz"
        np.savez_compressed(
            path, query_ids=query_ids, split=split, auto=auto, ranks=ranks[:, :k]
        )
        groups.append(
            {
                "corpus": spec["corpus"],
                "metric": spec["metric"],
                "k": k,
                "path": path.name,
            }
        )
        emit("prepared_group", corpus=spec["corpus"], k=k)
    (out / "manifest.json").write_text(
        json.dumps(
            {"groups": groups, "mandatory_k_anchors": MANDATORY_K_ANCHORS}, indent=2
        )
        + "\n"
    )
    (out / "identity.json").write_text(
        json.dumps(
            {
                "input": spec,
                "queries_sha256": digest(Path(spec["queries"])),
                "files": {
                    path.name: digest(path) for path in sorted(out.glob("*.npy"))
                },
                "lance_version": lance.__version__,
                "queries": len(queries),
                "calibration_queries": int(np.sum(split == 0)),
                "evaluation_queries": int(np.sum(split == 1)),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(json.loads(args.spec.read_text()), args.output)
