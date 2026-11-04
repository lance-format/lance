# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Create a task-owned S3 RQ5 clone of the frozen DINO corpus."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import lance
import numpy as np

SOURCE = (
    "s3://mmlb-us-east-1/rq-benchmark/results/"
    "20260909-fixed20-proportional-77824c43/datasets/dino/data"
)


def prepare(target, cache, output):
    source = lance.dataset(SOURCE, version=3)
    assert source.count_rows() == 10_000_000
    flat = next(
        index for index in source.list_indices() if index["name"] == "vector_ivf_flat"
    )
    assert flat["uuid"] == "3bb9ea84-e76b-401b-981a-42e4f538dfe1"
    centroids = np.load(cache / "centroids.npy")
    actual = np.asarray(
        source.centroids(index_name=flat["name"]).to_pylist(), dtype=np.float32
    )
    assert np.array_equal(centroids, actual)
    dataset = source.shallow_clone(target, reference=3)
    last = [0.0]

    def progress(event):
        now = time.monotonic()
        if event.event != "progress" or now - last[0] >= 20:
            print(json.dumps(dataclasses.asdict(event)), flush=True)
            last[0] = now

    dataset.create_index(
        "vector",
        "IVF_RQ",
        name=flat["name"],
        replace=True,
        metric="l2",
        num_bits=5,
        ivf_centroids=centroids,
        num_partitions=len(centroids),
        progress_callback=progress,
    )
    dataset = lance.dataset(target)
    rq = next(
        index for index in dataset.list_indices() if index["name"] == flat["name"]
    )
    stats = dataset.stats.index_stats(rq["name"])
    assert stats["num_indexed_rows"] == 10_000_000 and stats["num_unindexed_rows"] == 0
    result = {
        "flat": {
            "uri": SOURCE,
            "metric": "l2",
            "version": 3,
            "rows": 10_000_000,
            "index_name": flat["name"],
            "index_uuid": flat["uuid"],
        },
        "rq": {
            "uri": target,
            "metric": "l2",
            "version": dataset.version,
            "rows": 10_000_000,
            "index_name": rq["name"],
            "index_uuid": rq["uuid"],
        },
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target")
    parser.add_argument("cache", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.target, args.cache, args.output)
