# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Reuse frozen OSS-2221 and OSS-2249 query splits, truth and IVF membership.

Run from the repository's Python environment. The first input is the extracted
recall95 campaign, the second the extracted dot campaign's ``evidence`` folder.
The historical oracle loader verifies its original manifests and query hashes;
restore its reproduction-inputs at their original /home/ubuntu paths first.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from calibrate import digest

PROFILES = {
    "l2": {
        1: (0.2175, 5, 19),
        10: (0.265, 6, 24),
        100: (0.33, 11, 38),
        1000: (0.4175, 29, 93),
    },
    "cosine": {
        1: (0.235, 3, 50),
        10: (0.2875, 8, 77),
        100: (0.38, 7, 106),
        1000: (0.47, 34, 248),
    },
    "dot": {
        1: (0.14, 56, 112),
        10: (0.055, 144, 432),
        100: (0.0625, 200, 768),
    },
}


def budgets(distances, metric, k):
    margin, floor, cap = PROFILES[metric][k]
    distances = distances.astype(np.float64)
    nearest = distances[:, :1]
    scale = np.abs(1 - nearest) if metric == "dot" else nearest
    selected = (distances - nearest <= float(np.float32(margin)) * scale).sum(axis=1)
    return np.clip(selected, floor, min(cap, distances.shape[1]))


def prepare(recall95, dot, output):
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(
        0, str(recall95 / "reproduction-inputs/global-auto-parameters-20260909")
    )
    from matrix_gt import load_ground_truth

    sys.path.insert(
        0, str(recall95 / "reproduction-inputs/unified-auto-metrics-20260908")
    )
    from offline_calibration import routing_scores

    groups = []
    for corpus, metric in [("dino", "l2"), ("laion", "cosine"), ("fineweb", "cosine")]:
        cache = recall95 / "calibration" / corpus
        origin = json.loads((cache / "original-input.json").read_text())
        cache_identity = json.loads((cache / "cache.json").read_text())
        for name in ["rowids", "partitions", "centroids"]:
            assert (
                digest(cache / f"{name}.npy") == cache_identity["files"][name]["sha256"]
            )
        queries_path = Path(origin["queries"]["path"])
        assert digest(queries_path) == origin["queries"]["sha256"]
        queries = np.load(queries_path)
        labels = queries["split"]
        assert set(labels) <= {"calibration", "test", "reserve"}
        admitted = labels != "reserve"
        vectors = queries["vectors"][admitted]
        ids = queries["source_query_ids"][admitted]
        split = np.where(labels[admitted] == "calibration", 0, 1).astype(np.uint8)
        # Drop repeated vectors before selection, retaining calibration precedence.
        # Otherwise a different source ID could leak the same query into held-out.
        hashes = np.array([hashlib.sha256(q.tobytes()).hexdigest() for q in vectors])
        seen, retained = set(), []
        for i in np.argsort(split, kind="stable"):
            if hashes[i] not in seen:
                seen.add(hashes[i])
                retained.append(i)
        retained = np.array(sorted(retained))
        vectors, ids, split = vectors[retained], ids[retained], split[retained]
        manifest = json.loads(Path(origin["index_manifest"]["path"]).read_text())
        truth, _, oracle = load_ground_truth(
            origin["gt"], manifest, origin["queries"], ids, vectors
        )
        rowids = np.load(cache / "rowids.npy", mmap_mode="r")
        membership = np.load(cache / "partitions.npy", mmap_mode="r")
        positions = np.searchsorted(rowids, truth[:, :1000])
        assert np.array_equal(rowids[positions], truth[:, :1000])
        centroids = np.load(cache / "centroids.npy")
        scores = routing_scores(vectors, centroids, metric)
        order = np.argsort(scores, axis=1, kind="stable")
        ranks = np.empty(order.shape, dtype=np.int32)
        np.put_along_axis(
            ranks, order, np.arange(1, len(centroids) + 1)[None, :], axis=1
        )
        truth_ranks = np.take_along_axis(ranks, membership[positions], axis=1)
        distances = np.take_along_axis(scores, order, axis=1)
        np.savez_compressed(
            output / f"{corpus}-queries.npz",
            vectors=vectors,
            query_ids=ids,
            split=split,
            truth=truth,
        )
        for k in PROFILES[metric]:
            path = f"{corpus}-{k}.npz"
            np.savez_compressed(
                output / path,
                query_ids=ids,
                split=split,
                auto=budgets(distances, metric, k),
                ranks=truth_ranks[:, :k],
            )
            groups.append({"corpus": corpus, "metric": metric, "k": k, "path": path})
        print(
            json.dumps({"prepared": corpus, "queries": len(ids), "oracle": oracle}),
            flush=True,
        )

    for corpus in ["wiki-cohere-35m", "dpr-wikipedia-single-nq"]:
        directory = dot / corpus
        routes = np.load(directory / "routes.npz")
        splits = np.load(directory / "split.npz")
        ids = np.arange(len(routes["scores"]))
        split = np.ones(len(ids), dtype=np.uint8)
        split[splits["calibration"]] = 0
        assert np.array_equal(np.sort(splits["evaluation"]), np.flatnonzero(split == 1))
        distances = np.float32(1) - routes["scores"]
        for k in PROFILES["dot"]:
            path = f"{corpus}-{k}.npz"
            np.savez_compressed(
                output / path,
                query_ids=ids,
                split=split,
                auto=budgets(distances, "dot", k),
                ranks=routes["ranks"][:, :k],
            )
            groups.append({"corpus": corpus, "metric": "dot", "k": k, "path": path})
        print(json.dumps({"prepared": corpus, "queries": len(ids)}), flush=True)
    (output / "manifest.json").write_text(
        json.dumps({"groups": groups}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recall95", type=Path)
    parser.add_argument("dot", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.recall95, args.dot, args.output)
