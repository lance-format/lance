# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Prepare top-100,000 truth without modifying the completed k<=1000 study."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import lance
import numpy as np
from common import DATASETS, emit, matrix, positions, save
from lance.dataset import VectorIndexReader
from prepare import exact_truth

K_VALUES = [10_000, 100_000]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(previous, root, name):
    source = previous / name
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    directory = previous / "data" / name
    assert (directory / "VERIFIED").exists()
    for filename in ["build.json", "split.npz", "centroids.npy", "row-ids.npy"]:
        destination = out / filename
        if not destination.exists():
            shutil.copy2(source / filename, destination)
        assert sha256(destination) == sha256(source / filename), filename
    if not (root / "data").exists():
        (root / "data").symlink_to((previous / "data").resolve())
    build = json.loads((out / "build.json").read_text())
    dataset = lance.dataset(directory / "base.lance", version=build["version"])
    queries = matrix(lance.dataset(directory / "queries.lance").to_table()["vector"])
    metric = DATASETS[name]
    max_k = max(K_VALUES)
    assert build["rows"] >= max_k
    truth_path, scores_path = out / "truth-ids.npy", out / "truth-scores.npy"
    if not truth_path.exists() or not scores_path.exists():
        started = time.monotonic()
        truth, scores = exact_truth(dataset, metric, queries, max_k)
        for path, values in [(truth_path, truth), (scores_path, scores)]:
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, values)
            temporary.replace(path)
        del truth, scores
        emit("truth_complete", dataset=name, seconds=time.monotonic() - started)
    truth = np.load(truth_path, mmap_mode="r")
    scores = np.load(scores_path, mmap_mode="r")
    assert truth.shape == scores.shape == (len(queries), max_k)
    assert np.isfinite(scores).all()
    assert np.all(scores[:, 1:] >= scores[:, :-1])
    previous_scores = np.load(source / "truth.npz")["scores"]
    score_error = float(
        np.max(np.abs(scores[:, : previous_scores.shape[1]] - previous_scores))
    )
    assert np.allclose(
        scores[:, : previous_scores.shape[1]], previous_scores, rtol=0, atol=1e-10
    ), score_error
    emit("oracle_prefix_checked", dataset=name, max_score_error=score_error)
    del previous_scores, scores

    membership_path = out / "membership.npy"
    if not membership_path.exists():
        membership = np.full(build["rows"], -1, dtype=np.int32)
        lookup = np.load(out / "row-ids.npy", mmap_mode="r")
        reader = VectorIndexReader(dataset, build["index_name"])
        sizes = np.empty(reader.num_partitions(), dtype=np.int64)
        for pid in range(reader.num_partitions()):
            ids = reader.read_partition(pid)["_rowid"].to_numpy()
            row_positions = positions(ids, lookup)
            assert np.all(membership[row_positions] == -1)
            membership[row_positions] = pid
            sizes[pid] = len(ids)
            if pid % 1024 == 0:
                emit("mapping", dataset=name, partition=pid)
        assert np.all(membership >= 0) and sizes.sum() == build["rows"]
        original_sizes = np.load(source / "mapping.npz")["sizes"]
        np.testing.assert_array_equal(sizes, original_sizes)
        np.save(out / "sizes.npy", sizes)
        np.save(membership_path, membership)
        del membership, reader, lookup
    membership = np.load(membership_path, mmap_mode="r")
    routes = np.load(source / "routes.npz")
    for field in ["distances", "order", "scanned_rows"]:
        np.save(out / f"{field}.npy", routes[field])
    order = routes["order"]
    inverse = np.empty(order.shape, dtype=np.int32)
    np.put_along_axis(inverse, order, np.arange(1, order.shape[1] + 1)[None, :], axis=1)
    ranks_path = out / "ranks.npy"
    if not ranks_path.exists():
        temporary = ranks_path.with_suffix(".tmp.npy")
        ranks = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.int32, shape=truth.shape
        )
        for start in range(0, len(truth), 64):
            stop = min(start + 64, len(truth))
            ranks[start:stop] = np.take_along_axis(
                inverse[start:stop], membership[truth[start:stop]], axis=1
            )
        ranks.flush()
        del ranks
        temporary.replace(ranks_path)
    save(
        out / "prepared.json",
        {
            "dataset": name,
            "metric": metric,
            "rows": build["rows"],
            "partitions": build["partitions"],
            "queries": len(queries),
            "truth_k": max_k,
            "previous_study": str(previous.resolve()),
            "previous_oracle_max_score_error": score_error,
            "sha256": {path.name: sha256(path) for path in sorted(out.glob("*.npy"))},
            "split_sha256": sha256(out / "split.npz"),
            "build_sha256": sha256(out / "build.json"),
        },
    )
    emit("prepared", dataset=name, queries=len(queries), truth_k=max_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    args = parser.parse_args()
    assert args.root.resolve() != args.previous.resolve()
    prepare(args.previous, args.root, args.dataset)
