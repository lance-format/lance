# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Build frozen IVF_FLAT indices and collect actual partition routing evidence.

Run from the repository Python environment with ``uv run``. The root directory
must contain data/<HF dataset name>/{base,queries,ground_truth}.lance and VERIFIED.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path

import lance
import numpy as np
from lance.dataset import VectorIndexReader


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def emit(event, **values):
    print(json.dumps({"time": time.time(), "event": event, **values}), flush=True)


def matrix(column):
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    return array.values.to_numpy().reshape(len(array), array.type.list_size)


def offsets(dataset):
    fragments = dataset.get_fragments()
    result = np.zeros(max(f.fragment_id for f in fragments) + 1, dtype=np.int64)
    start = 0
    for fragment in fragments:
        result[fragment.fragment_id] = start
        start += fragment.count_rows()
    return result


def positions(row_ids, fragment_offsets):
    return fragment_offsets[(row_ids >> np.uint64(32)).astype(np.int64)] + (
        row_ids & np.uint64(0xFFFFFFFF)
    ).astype(np.int64)


def prepare(root, name):
    directory = root / "data" / name
    assert (directory / "VERIFIED").exists()
    out = root / name
    out.mkdir(exist_ok=True)
    queries_table = lance.dataset(directory / "queries.lance").to_table()
    truth_table = lance.dataset(directory / "ground_truth.lance").to_table()
    assert np.array_equal(queries_table["query_id"], truth_table["query_id"])
    queries = matrix(queries_table["vector"])
    truth = matrix(truth_table["neighbor_ids"])
    split = np.random.default_rng(2249).permutation(len(queries))
    np.savez(
        out / "split.npz",
        calibration=split[: len(split) // 2],
        evaluation=split[len(split) // 2 :],
    )
    dataset = lance.dataset(directory / "base.lance")
    if not (out / "build.json").exists():
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
            name="dot_flat",
            metric="dot",
            num_partitions=partitions,
            progress_callback=progress,
        )
        stats = dataset.stats.index_stats("dot_flat")
        assert stats["num_indexed_rows"] == count
        assert stats["num_unindexed_rows"] == 0
        save(
            out / "build.json",
            {
                "rows": count,
                "partitions": partitions,
                "version": dataset.version,
                "seconds": time.monotonic() - started,
                "indices": dataset.list_indices(),
                "lance_version": lance.__version__,
            },
        )
        emit("build_complete", dataset=name, seconds=time.monotonic() - started)
    build = json.loads((out / "build.json").read_text())
    dataset = lance.dataset(
        directory / "base.lance",
        version=build["version"],
        index_cache_size_bytes=128 * 1024**3,
    )
    centroids = matrix(dataset.centroids(index_name="dot_flat"))
    np.save(out / "centroids.npy", centroids)
    fragment_offsets = offsets(dataset)
    np.save(out / "fragment-offsets.npy", fragment_offsets)
    if not (out / "mapping.npz").exists():
        membership = np.full(build["rows"], -1, dtype=np.int32)
        sizes = np.empty(len(centroids), dtype=np.int64)
        rng = np.random.default_rng(2249)
        sample_partitions = set(rng.choice(len(centroids), 16, replace=False).tolist())
        sample_positions = []
        sample_membership = []
        reader = VectorIndexReader(dataset, "dot_flat")
        for pid in range(reader.num_partitions()):
            row_ids = reader.read_partition(pid)["_rowid"].to_numpy()
            row_positions = positions(row_ids, fragment_offsets)
            assert np.all(membership[row_positions] == -1)
            membership[row_positions] = pid
            sizes[pid] = len(row_ids)
            if pid in sample_partitions and len(row_positions):
                sampled = rng.choice(
                    row_positions, min(64, len(row_positions)), replace=False
                )
                sample_positions.extend(sampled.tolist())
                sample_membership.extend([pid] * len(sampled))
            if pid % 512 == 0:
                emit("mapping", dataset=name, partition=pid)
        assert np.all(membership >= 0)
        assert sizes.sum() == build["rows"]
        np.savez_compressed(
            out / "mapping.npz",
            truth_partitions=membership[truth],
            sizes=sizes,
            sample_positions=sample_positions,
            sample_partitions=sample_membership,
        )
        del membership, reader
    mapping = np.load(out / "mapping.npz")
    scores = queries @ centroids.T
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty(order.shape, dtype=np.int32)
    np.put_along_axis(ranks, order, np.arange(1, len(centroids) + 1)[None, :], axis=1)
    truth_ranks = np.take_along_axis(ranks, mapping["truth_partitions"], axis=1)
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    scanned_rows = np.cumsum(mapping["sizes"][order], axis=1)
    np.savez_compressed(
        out / "routes.npz",
        scores=sorted_scores,
        ranks=truth_ranks,
        scanned_rows=scanned_rows,
    )
    # A fixed calibration-only subset diagnoses residuals without reading the
    # evaluation queries during model selection. Truth vectors retain their norms.
    diagnostic_ids = split[: min(256, len(split) // 2)]
    ids, inverse = np.unique(truth[diagnostic_ids], return_inverse=True)
    vectors = matrix(dataset.take(ids.tolist(), columns=["vector"])["vector"])
    selected_vectors = vectors[inverse].reshape(
        len(diagnostic_ids), truth.shape[1], queries.shape[1]
    )
    projected = np.einsum("qd,qkd->qk", queries[diagnostic_ids], selected_vectors)
    projected_centroids = np.take_along_axis(
        scores[diagnostic_ids], mapping["truth_partitions"][diagnostic_ids], axis=1
    )
    residuals = projected - projected_centroids
    sample_vectors = matrix(
        dataset.take(mapping["sample_positions"].tolist(), columns=["vector"])["vector"]
    )
    sample_residuals = (
        queries[diagnostic_ids]
        @ (sample_vectors - centroids[mapping["sample_partitions"]]).T
    )
    percentiles = [0, 10, 50, 90, 95, 99, 100]
    save(
        out / "diagnostics.json",
        {
            "query_ids": diagnostic_ids.tolist(),
            "percentiles": percentiles,
            "query_norm": np.percentile(
                np.linalg.norm(queries[diagnostic_ids], axis=1), percentiles
            ).tolist(),
            "neighbor_norm": np.percentile(
                np.linalg.norm(selected_vectors, axis=2), percentiles
            ).tolist(),
            "centroid_norm": np.percentile(
                np.linalg.norm(centroids, axis=1), percentiles
            ).tolist(),
            "nearest_centroid_score": np.percentile(
                sorted_scores[diagnostic_ids, 0], percentiles
            ).tolist(),
            "neighbor_score": np.percentile(projected, percentiles).tolist(),
            "projected_residual": np.percentile(residuals, percentiles).tolist(),
            "random_partition_projected_residual": np.percentile(
                sample_residuals, percentiles
            ).tolist(),
            "random_partition_vector_norm": np.percentile(
                np.linalg.norm(sample_vectors, axis=1), percentiles
            ).tolist(),
            "centroid_sha256": hashlib.sha256(centroids.tobytes()).hexdigest(),
        },
    )
    emit(
        "prepared",
        dataset=name,
        rows=build["rows"],
        queries=len(queries),
        partitions=len(centroids),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "dataset", choices=["wiki-cohere-35m", "dpr-wikipedia-single-nq"]
    )
    args = parser.parse_args()
    prepare(args.root, args.dataset)
