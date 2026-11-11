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
import bisect
import hashlib
import json
import math
from pathlib import Path

import numpy as np

K_BUCKETS = (1, 10, 100, 200, 500, 1000, 10_000, 100_000)


def k_bucket(k):
    if not 1 <= k <= K_BUCKETS[-1]:
        raise ValueError(f"k must be in [1, {K_BUCKETS[-1]}], got {k}")
    return next(upper for upper in K_BUCKETS if k <= upper)


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


def scale_at(profiles, k, tier):
    anchors = sorted(map(int, profiles))
    upper = bisect.bisect_left(anchors, k)
    if upper < len(anchors) and anchors[upper] == k:
        return profiles[str(k)][tier]["scale"]
    if upper == 0 or upper == len(anchors):
        raise ValueError(f"k={k} lies outside calibrated anchors {anchors}")
    lower_k, upper_k = anchors[upper - 1], anchors[upper]
    lower = profiles[str(lower_k)][tier]["scale"]
    higher = profiles[str(upper_k)][tier]["scale"]
    weight = math.log(k / lower_k) / math.log(upper_k / lower_k)
    return lower * (higher / lower) ** weight


def calibrate(manifest, output):
    groups = load(manifest, 0)
    if not groups:
        raise ValueError("Calibration requires nonempty input groups")
    ks = sorted({group["k"] for group in groups})
    mandatory = json.loads(manifest.read_text()).get("mandatory_k_anchors", ks)
    if not set(mandatory) <= set(ks) or not {ks[0], ks[-1]} <= set(mandatory):
        raise ValueError("Mandatory anchors must include the input range endpoints")
    result = {
        "manifest_sha256": digest(manifest),
        "input_sha256": {
            group["path"]: digest(manifest.parent / group["path"]) for group in groups
        },
        "evaluation_used": False,
        "scale_step": 0.001,
        "mandatory_k_anchors": mandatory,
        "interpolation_max_probe_overhead": 1.05,
        "profiles": {},
    }
    # Fit each k independently using the original 1% calibration headroom.
    # A common fraction across a wide Auto bucket can overscan at its lower end.
    pointwise = {}
    for metric in sorted({group["metric"] for group in groups}):
        pointwise[metric] = {}
        for k in ks:
            selected = [
                group
                for group in groups
                if group["metric"] == metric and group["k"] == k
            ]
            if not selected:
                raise ValueError(f"{metric}: no calibration data for k={k}")
            profiles = []
            for effort, target in [(0.1, 0.8), (0.25, 0.9)]:
                low, high = 1, 1000
                required = target + 0.01
                assert all(
                    measure(group, 1)["mean_routing_recall"] >= required
                    for group in selected
                ), f"{metric}/{k}: Auto cannot meet calibration target {required}"
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
            pointwise[metric][str(k)] = profiles

    # Keep every Auto boundary, then add a check point only when interpolation
    # misses a calibration constraint or costs >5% more probes than fitting that
    # point directly. Selection and the cost comparison use calibration only.
    anchors = set(mandatory)
    while True:
        profiles = {
            metric: {str(k): values[str(k)] for k in sorted(anchors)}
            for metric, values in pointwise.items()
        }
        added, checks = set(), []
        for group in groups:
            k, metric = group["k"], group["metric"]
            if k in anchors:
                continue
            for tier in range(2):
                optimum = pointwise[metric][str(k)][tier]
                scale = scale_at(profiles[metric], k, tier)
                observed = measure(group, scale)
                best = measure(group, optimum["scale"])
                overhead = observed["mean_partitions"] / best["mean_partitions"]
                if (
                    observed["mean_routing_recall"] < optimum["calibration_target"]
                    or overhead > 1.05
                ):
                    added.add(k)
                checks.append(
                    {
                        **observed,
                        "effort": optimum["effort"],
                        "scale": scale,
                        "pointwise_scale": optimum["scale"],
                        "probe_overhead": overhead,
                    }
                )
        if not added:
            result["k_anchors"] = sorted(anchors)
            result["profiles"] = profiles
            result["interpolation_checks"] = checks
            break
        anchors.update(added)
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
        for tier, (effort, target) in enumerate([(0.1, 0.8), (0.25, 0.9), (0.5, 0.95)]):
            scale = (
                1.0
                if effort == 0.5
                else scale_at(profiles["profiles"][group["metric"]], group["k"], tier)
            )
            results.append(
                {
                    "effort": effort,
                    "target": target,
                    "interpolated": effort != 0.5
                    and group["k"] not in profiles["k_anchors"],
                    **measure(group, scale),
                }
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
