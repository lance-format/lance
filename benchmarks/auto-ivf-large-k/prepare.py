# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Freeze an IVF_FLAT index, exact top-k ground truth and partition routing.

The root must contain data/<name>/{base,queries}.lance and VERIFIED. An optional
data/<name>/ground_truth.lance with published neighbor_ids is only cross-checked.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
from common import (
    DATASETS,
    FILTER_STRIDE,
    FILTERED_K,
    MAX_K,
    SPLIT_SEED,
    emit,
    matrix,
    positions,
    routing_distances,
    row_ids_by_position,
    save,
)
from lance.dataset import VectorIndexReader

INDEX_NAME = "auto_flat"
QUERY_BLOCK = 1024
BASE_BATCH = 262_144


def build_index(dataset, name, metric, out):
    assert dataset.list_indices() == []
    count = dataset.count_rows()
    partitions = math.ceil(count / 4096)
    started = time.monotonic()
    last = [0.0]

    def progress(event):
        now = time.monotonic()
        if event.event != "progress" or now - last[0] >= 30:
            emit("build_progress", dataset=name, progress=dataclasses.asdict(event))
            last[0] = now

    dataset = dataset.create_index(
        "vector",
        "IVF_FLAT",
        name=INDEX_NAME,
        metric=metric,
        num_partitions=partitions,
        progress_callback=progress,
    )
    stats = dataset.stats.index_stats(INDEX_NAME)
    assert stats["num_indexed_rows"] == count
    assert stats["num_unindexed_rows"] == 0
    save(
        out / "build.json",
        {
            "rows": count,
            "partitions": partitions,
            "index_name": INDEX_NAME,
            "version": dataset.version,
            "seconds": time.monotonic() - started,
            "indices": dataset.list_indices(),
            "lance_version": lance.__version__,
        },
    )
    emit("build_complete", dataset=name, seconds=time.monotonic() - started)


def adopt_index(dataset, index_name, out):
    """Reuse a frozen single-segment IVF_FLAT index built by an earlier study."""
    stats = dataset.stats.index_stats(index_name)
    assert stats["num_segments"] == 1
    count = dataset.count_rows()
    assert stats["num_indexed_rows"] == count
    assert stats["num_unindexed_rows"] == 0
    assert stats["index_type"] == "IVF_FLAT", stats["index_type"]
    save(
        out / "build.json",
        {
            "rows": count,
            "partitions": stats["indices"][0]["num_partitions"],
            "index_name": index_name,
            "version": dataset.version,
            "adopted": True,
            "indices": dataset.list_indices(),
            "lance_version": lance.__version__,
        },
    )


def exact_truth(dataset, metric, queries, k):
    """Exact float64 top-k; cutoff ties follow NumPy's argpartition selection.

    Equal-score neighbors retained in the top-k set are ordered by position.
    Strict-ID recall counts a different selection at the cutoff as a miss.
    """
    queries = queries.astype(np.float64)
    if metric == "cosine":
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    best_scores = np.full((len(queries), k), np.inf)
    best_ids = np.full((len(queries), k), -1, dtype=np.int64)
    start = 0
    pool = ThreadPoolExecutor(16)

    def merge(rows, scores, ids):
        scores = np.concatenate([best_scores[rows], scores], axis=1)
        ids = np.concatenate(
            [best_ids[rows], np.broadcast_to(ids, (len(rows), len(ids)))], axis=1
        )
        keep = np.argpartition(scores, k - 1, axis=1)[:, :k]
        best_scores[rows] = np.take_along_axis(scores, keep, axis=1)
        best_ids[rows] = np.take_along_axis(ids, keep, axis=1)

    for batch in dataset.to_batches(columns=["vector"], batch_size=BASE_BATCH):
        vectors = matrix(batch["vector"]).astype(np.float64)
        ids = np.arange(start, start + len(vectors), dtype=np.int64)
        if metric == "cosine":
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.sum(vectors**2, axis=1) if metric == "l2" else None
        for block in range(0, len(queries), QUERY_BLOCK):
            rows = np.arange(block, min(block + QUERY_BLOCK, len(queries)))
            products = queries[rows] @ vectors.T
            # Query norms are constant per row and do not affect L2 ranking.
            scores = norms[None, :] - 2.0 * products if metric == "l2" else -products
            slices = np.array_split(rows, 16)
            list(
                pool.map(
                    lambda part: merge(part, scores[part - block], ids),
                    [part for part in slices if len(part)],
                )
            )
        start += len(vectors)
        if start % (BASE_BATCH * 16) < BASE_BATCH:
            emit("truth_progress", rows=start)
    pool.shutdown()
    order = np.lexsort((best_ids, best_scores), axis=1)
    return (
        np.take_along_axis(best_ids, order, axis=1),
        np.take_along_axis(best_scores, order, axis=1),
    )


def filtered_truth(dataset, lookup, metric, queries, k):
    """Exact top-k among rows matching the measurement prefilter."""
    matching = np.flatnonzero(lookup % np.uint64(FILTER_STRIDE) == 0)
    vectors = matrix(dataset.take(matching.tolist(), columns=["vector"])["vector"])
    vectors = vectors.astype(np.float64)
    queries = queries.astype(np.float64)
    if metric == "cosine":
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    products = queries @ vectors.T
    if metric == "l2":
        scores = np.sum(vectors**2, axis=1)[None, :] - 2.0 * products
    else:
        scores = -products
    order = np.lexsort((np.broadcast_to(matching, scores.shape), scores), axis=1)[:, :k]
    return matching[order], np.take_along_axis(scores, order, axis=1), matching


def save_query_split(queries, out):
    """Keep calibration unchanged and exclude duplicate vectors from held-out data."""
    permutation = np.random.default_rng(SPLIT_SEED).permutation(len(queries))
    calibration = permutation[: len(permutation) // 2]
    candidate_evaluation = permutation[len(permutation) // 2 :]
    keys = (
        np.ascontiguousarray(queries)
        .view(np.dtype((np.void, queries.dtype.itemsize * queries.shape[1])))
        .ravel()
    )
    _, groups = np.unique(keys, return_inverse=True)
    seen = set(groups[calibration].tolist())
    evaluation, excluded = [], []
    for query_id in candidate_evaluation:
        group = int(groups[query_id])
        if group in seen:
            excluded.append(int(query_id))
        else:
            seen.add(group)
            evaluation.append(int(query_id))
    np.savez(
        out / "split.npz",
        calibration=calibration,
        evaluation=np.asarray(evaluation, dtype=np.int64),
    )
    save(
        out / "split-audit.json",
        {
            "seed": SPLIT_SEED,
            "calibration_queries": len(calibration),
            "evaluation_queries": len(evaluation),
            "excluded_duplicate_evaluation_queries": excluded,
        },
    )


def prepare(root, name):
    metric = DATASETS[name]
    directory = root / "data" / name
    assert (directory / "VERIFIED").exists()
    out = root / name
    out.mkdir(exist_ok=True)
    queries = matrix(lance.dataset(directory / "queries.lance").to_table()["vector"])
    assert queries.dtype == np.float32 and np.isfinite(queries).all()
    save_query_split(queries, out)
    if not (out / "build.json").exists():
        imported = directory / "import.json"
        if imported.exists():
            spec = json.loads(imported.read_text())
            dataset = lance.dataset(directory / "base.lance", version=spec["version"])
            adopt_index(dataset, spec["index"]["name"], out)
        else:
            build_index(lance.dataset(directory / "base.lance"), name, metric, out)
    build = json.loads((out / "build.json").read_text())
    dataset = lance.dataset(directory / "base.lance", version=build["version"])
    if not (out / "row-ids.npy").exists():
        np.save(out / "row-ids.npy", row_ids_by_position(dataset))
    lookup = np.load(out / "row-ids.npy")
    assert len(lookup) == build["rows"]

    if not (out / "truth.npz").exists():
        started = time.monotonic()
        truth, scores = exact_truth(dataset, metric, queries, MAX_K)
        np.savez(out / "truth.npz", ids=truth, scores=scores)
        emit("truth_complete", dataset=name, seconds=time.monotonic() - started)
    truth = np.load(out / "truth.npz")["ids"]
    if not (out / "filtered-truth.npz").exists():
        ids, scores, matching = filtered_truth(
            dataset, lookup, metric, queries, FILTERED_K
        )
        np.savez(out / "filtered-truth.npz", ids=ids, scores=scores, positions=matching)
    check = {}
    if (directory / "ground_truth.lance").exists():
        published = lance.dataset(directory / "ground_truth.lance").to_table()
        published = matrix(published["neighbor_ids"]).astype(np.int64)
        width = published.shape[1]
        # Published neighbors are ordered; compare as sets to tolerate exact ties.
        overlap = [
            len(np.intersect1d(published[i], truth[i, :width])) / width
            for i in range(len(truth))
        ]
        check = {
            "published_width": width,
            "mean_overlap": float(np.mean(overlap)),
            "min_overlap": float(np.min(overlap)),
            "queries_below_1": int(np.sum(np.asarray(overlap) < 1)),
        }
        emit("published_truth_check", dataset=name, **check)

    if not (out / "mapping.npz").exists():
        membership = np.full(build["rows"], -1, dtype=np.int32)
        reader = VectorIndexReader(dataset, build["index_name"])
        sizes = np.empty(reader.num_partitions(), dtype=np.int64)
        for pid in range(reader.num_partitions()):
            row_ids = reader.read_partition(pid)["_rowid"].to_numpy()
            row_positions = positions(row_ids, lookup)
            assert np.all(membership[row_positions] == -1)
            membership[row_positions] = pid
            sizes[pid] = len(row_ids)
            if pid % 1024 == 0:
                emit("mapping", dataset=name, partition=pid)
        assert np.all(membership >= 0)
        assert sizes.sum() == build["rows"]
        np.savez_compressed(
            out / "mapping.npz", truth_partitions=membership[truth], sizes=sizes
        )
        del membership, reader
    mapping = np.load(out / "mapping.npz")

    centroids = matrix(dataset.centroids(index_name=build["index_name"]))
    np.save(out / "centroids.npy", centroids)
    distances, order = routing_distances(metric, queries, centroids)
    ranks = np.empty(order.shape, dtype=np.int32)
    np.put_along_axis(ranks, order, np.arange(1, order.shape[1] + 1)[None, :], axis=1)
    np.savez_compressed(
        out / "routes.npz",
        distances=distances,
        order=order.astype(np.int32),
        ranks=np.take_along_axis(ranks, mapping["truth_partitions"], axis=1),
        scanned_rows=np.cumsum(mapping["sizes"][order], axis=1),
    )
    save(
        out / "prepared.json",
        {
            "metric": metric,
            "rows": build["rows"],
            "partitions": int(len(centroids)),
            "queries": len(queries),
            "truth_k": MAX_K,
            "published_truth": check,
            "centroid_sha256": hashlib.sha256(centroids.tobytes()).hexdigest(),
            "truth_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
            "partition_size": {
                "mean": float(mapping["sizes"].mean()),
                "min": int(mapping["sizes"].min()),
                "max": int(mapping["sizes"].max()),
            },
        },
    )
    emit("prepared", dataset=name, partitions=len(centroids), queries=len(queries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    args = parser.parse_args()
    prepare(args.root, args.dataset)
