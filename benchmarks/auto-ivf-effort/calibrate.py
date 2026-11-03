# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Calibrate fractions of Auto's budget without reading evaluation queries.

Each manifest group describes one corpus and k. Its NPZ contains one-dimensional
``auto`` budgets, ``query_ids`` and ``split`` (0=calibration, 1=evaluation), plus
one-based ground-truth partition ``ranks`` of shape (queries, k). All arrays must
come from the same frozen index and centroid ranking. A separate evaluate command
consumes the frozen output; evaluation never selects or changes a scale.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load(manifest, split):
    groups = []
    for spec in json.loads(manifest.read_text())["groups"]:
        path = manifest.parent / spec["path"]
        with np.load(path) as arrays:
            ids = arrays["query_ids"]
            assert len(np.unique(ids)) == len(ids), spec
            selected = arrays["split"] == split
            auto = arrays["auto"][selected]
            ranks = arrays["ranks"][selected]
            assert len(auto) > 0 and ranks.shape == (len(auto), spec["k"]), spec
            assert np.all(auto >= 1) and np.all(ranks >= 1), spec
            groups.append({**spec, "auto": auto, "ranks": ranks})
    return groups


def measure(group, scale):
    budgets = np.maximum(1, np.ceil(group["auto"] * scale)).astype(np.int64)
    recalls = (group["ranks"] <= budgets[:, None]).mean(axis=1)
    return {
        "corpus": group["corpus"],
        "metric": group["metric"],
        "k": group["k"],
        "queries": len(budgets),
        "mean_routing_recall": float(recalls.mean()),
        "mean_partitions": float(budgets.mean()),
        "p95_partitions": float(np.percentile(budgets, 95)),
    }


def calibrate(manifest, output):
    groups = load(manifest, 0)
    result = {
        "manifest_sha256": digest(manifest),
        "input_sha256": {
            group["path"]: digest(manifest.parent / group["path"]) for group in groups
        },
        "evaluation_used": False,
        "scale_step": 0.001,
        "profiles": {},
    }
    # One scalar per metric/target keeps the existing k-dependent Auto profile.
    # Satisfy every calibration corpus/k separately, rather than hiding a miss
    # behind the average of unrelated workloads. The 1% headroom follows the
    # original 96% calibration / 95% evaluation protocol.
    for metric in sorted({group["metric"] for group in groups}):
        selected = [group for group in groups if group["metric"] == metric]
        profiles = []
        for effort, target in [(0.1, 0.8), (0.25, 0.9)]:
            low, high = 1, 1000
            required = target + 0.01
            assert all(
                measure(group, 1)["mean_routing_recall"] >= required
                for group in selected
            ), f"{metric}: Auto cannot meet calibration target {required}"
            while low < high:
                middle = (low + high) // 2
                if all(
                    measure(group, middle / 1000)["mean_routing_recall"] >= required
                    for group in selected
                ):
                    high = middle
                else:
                    low = middle + 1
            scale = low / 1000
            profiles.append(
                {
                    "effort": effort,
                    "target": target,
                    "calibration_target": required,
                    "scale": scale,
                    "calibration": [measure(group, scale) for group in selected],
                }
            )
        result["profiles"][metric] = profiles
    # Exclusive creation makes an accidental post-evaluation overwrite visible.
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


def evaluate(manifest, frozen, output):
    profiles = json.loads(frozen.read_text())
    assert digest(manifest) == profiles["manifest_sha256"]
    for path, expected in profiles["input_sha256"].items():
        assert digest(manifest.parent / path) == expected, path
    results = []
    for group in load(manifest, 1):
        results.extend(
            {
                "effort": profile["effort"],
                "target": profile["target"],
                **measure(group, profile["scale"]),
            }
            for profile in profiles["profiles"][group["metric"]]
            + [{"effort": 0.5, "target": 0.95, "scale": 1.0}]
        )
    with output.open("x") as stream:
        json.dump(
            {"frozen_sha256": digest(frozen), "results": results}, stream, indent=2
        )
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("manifest", type=Path)
    calibration.add_argument("output", type=Path)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("manifest", type=Path)
    evaluation.add_argument("frozen", type=Path)
    evaluation.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.manifest, args.output)
    else:
        evaluate(args.manifest, args.frozen, args.output)
