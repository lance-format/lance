# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Select k > 100 Auto profiles from calibration queries only.

For every metric group and anchor k, minimize the corpus-averaged mean initial
partition count subject to CALIBRATION_TARGET mean recall on every corpus of
the group. Recall is the routing coverage of exact top-k neighbors, which equals
IVF_FLAT recall up to exact distance ties.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from common import (
    ANCHORS,
    CALIBRATION_TARGET,
    EXISTING_K100,
    EXTRAPOLATION_K,
    GROUPS,
    MEASURED_K,
    emit,
    gap_counts,
    predict,
    save,
)

COARSE_MARGINS = np.r_[
    np.arange(0, 0.1, 0.005),
    np.arange(0.1, 1.0, 0.02),
    np.arange(1.0, 4.01, 0.1),
]
FIXED = [1, 2, 4, 8, 16, 20, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]
FIXED += [1024, 1536, 2048, 3072, 4096, 6144, 8192]


def geometric(limit, ratio=1.08):
    values = {1, limit}
    value = 1.0
    while value < limit:
        values.add(int(round(value)))
        value *= ratio
    return np.array(sorted(v for v in values if 1 <= v <= limit))


def load(root, name, ids_key="calibration"):
    routes = np.load(root / name / "routes.npz")
    ids = np.load(root / name / "split.npz")[ids_key]
    return {
        "distances": routes["distances"][ids],
        "ranks": routes["ranks"][ids],
        "rows": routes["scanned_rows"][ids],
        "ids": ids,
    }


def coverage(ranks, k, partitions):
    """coverage[q, b] = number of exact top-k neighbors in the first b partitions."""
    counts = np.zeros((len(ranks), partitions + 1), dtype=np.int32)
    np.add.at(counts, (np.arange(len(ranks))[:, None], ranks[:, :k]), 1)
    return np.cumsum(counts, axis=1, dtype=np.int32)


def evaluate(metric, data, k, profile):
    """Mean recall and initial-budget distribution of one profile."""
    budgets = predict(metric, data["distances"], profile)
    cover = data["coverage"][k]
    recall = cover[np.arange(len(budgets)), budgets] / k
    rows = data["rows"][np.arange(len(budgets)), budgets - 1]
    return {
        "recall": float(recall.mean()),
        "partitions": distribution(budgets),
        "rows": distribution(rows),
    }


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        **{f"p{p}": float(np.percentile(values, p)) for p in [50, 90, 95, 99]},
        "max": float(values.max()),
    }


def search(metric, corpora, k, margins, floors, caps):
    """Best (margin, floor, cap) per corpus and in common over a grid."""
    best = {name: None for name in corpora} | {"common": None}
    partitions = {name: data["distances"].shape[1] for name, data in corpora.items()}
    for margin in margins:
        selected = {
            name: gap_counts(metric, data["distances"], margin)
            for name, data in corpora.items()
        }
        for floor in floors:
            valid_caps = caps[caps >= floor]
            if not len(valid_caps):
                continue
            metrics = {}
            for name, data in corpora.items():
                budgets = np.minimum(
                    np.maximum(selected[name], floor)[:, None], valid_caps[None, :]
                )
                budgets = np.minimum(budgets, partitions[name])
                cover = data["coverage"][k]
                recall = np.take_along_axis(cover, budgets, axis=1).mean(axis=0) / k
                cost = budgets.mean(axis=0)
                metrics[name] = (recall, cost)
                feasible = np.flatnonzero(recall >= CALIBRATION_TARGET)
                if len(feasible):
                    index = feasible[np.argmin(cost[feasible])]
                    candidate = {
                        "margin": float(margin),
                        "floor": int(floor),
                        "cap": int(valid_caps[index]),
                        "recall": float(recall[index]),
                        "mean_partitions": float(cost[index]),
                    }
                    if (
                        best[name] is None
                        or candidate["mean_partitions"] < best[name]["mean_partitions"]
                    ):
                        best[name] = candidate
            feasible = np.flatnonzero(
                np.logical_and.reduce(
                    [metrics[name][0] >= CALIBRATION_TARGET for name in corpora]
                )
            )
            if not len(feasible):
                continue
            # Corpora have different partition counts; weight them equally.
            cost = np.mean([metrics[name][1] for name in corpora], axis=0)
            index = feasible[np.argmin(cost[feasible])]
            candidate = {
                "margin": float(margin),
                "floor": int(floor),
                "cap": int(valid_caps[index]),
                "mean_partitions": float(cost[index]),
                "datasets": {
                    name: {
                        "recall": float(metrics[name][0][index]),
                        "mean_partitions": float(metrics[name][1][index]),
                    }
                    for name in corpora
                },
            }
            if (
                best["common"] is None
                or candidate["mean_partitions"] < best["common"]["mean_partitions"]
            ):
                best["common"] = candidate
    return best


def refine(metric, corpora, k, coarse, limit):
    """Search integer floors/caps and fine margins around a coarse optimum."""

    def around(value, low, high):
        # At most 41 integer values within 15% (and at least +-4) of the optimum.
        span = max(4, int(value * 0.15))
        values = np.linspace(max(low, value - span), min(high, value + span), 41)
        return np.unique(np.round(values).astype(np.int64))

    margin = coarse["margin"]
    margins = np.unique(
        np.round(np.clip(margin + np.arange(-0.05, 0.0501, 0.0025), 0, None), 6)
    )
    floors = around(coarse["floor"], 1, limit)
    caps = around(coarse["cap"], 1, limit)
    return search(metric, corpora, k, margins, floors, caps)


def calibrate(root, k_values, metrics):
    path = root / "calibration.json"
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "target_recall": CALIBRATION_TARGET,
            "evaluation_used": False,
            "anchors": ANCHORS,
            "grid": {
                "coarse_margins": COARSE_MARGINS.tolist(),
                "floor_cap_ratio": 1.08,
                "refinement": (
                    "up to 41 integer floors/caps within 15% (at least 4), "
                    "margin +-0.05 by 0.0025"
                ),
            },
            "fixed": {},
            "profiles": {},
            "per_corpus": {},
        }
    )
    # Metric groups are independent, so each can be calibrated once its corpora
    # are prepared.
    for metric in metrics:
        names = GROUPS[metric]
        corpora = {name: load(root, name) for name in names}
        partitions = {
            name: data["distances"].shape[1] for name, data in corpora.items()
        }
        for name, data in corpora.items():
            data["coverage"] = {
                k: coverage(data["ranks"], k, partitions[name]) for k in k_values
            }
            result["fixed"][name] = {
                str(k): [
                    {
                        "nprobes": budget,
                        "recall": float(data["coverage"][k][:, budget].mean() / k),
                        "rows": distribution(data["rows"][:, budget - 1]),
                    }
                    for budget in sorted(
                        {b for b in FIXED if b < partitions[name]} | {partitions[name]}
                    )
                ]
                for k in k_values
            }
        limit = max(partitions.values())
        grid = geometric(limit)
        result["profiles"][metric] = {}
        result["per_corpus"].update({name: {} for name in names})
        for k in [anchor for anchor in k_values if anchor in ANCHORS + EXTRAPOLATION_K]:
            coarse = search(metric, corpora, k, COARSE_MARGINS, grid, grid)
            emit("coarse", metric=metric, k=k, best=coarse)
            fine = refine(metric, corpora, k, coarse["common"], limit)
            result["profiles"][metric][str(k)] = fine["common"]
            for name in names:
                tuned = refine(metric, {name: corpora[name]}, k, coarse[name], limit)
                result["per_corpus"][name][str(k)] = tuned[name]
            emit("refined", metric=metric, k=k, best=fine["common"])
            save(root / "calibration.json", result)
    save(root / "calibration.json", result)
    return result


def simulate(root, result, ids_key, metrics):
    """Route-level budgets and recall of candidate bucket structures for every k."""
    structures = {
        # Pre-registered primary design: buckets end at each anchor.
        "anchor_buckets": lambda k: next(a for a in ANCHORS if k <= a),
        # Simpler alternative: one bucket calibrated at the largest anchor.
        "single_bucket": lambda k: ANCHORS[-1],
    }
    k_values = sorted(set(MEASURED_K + EXTRAPOLATION_K + [150, 300, 400, 750]))
    path = root / f"simulation-{ids_key}.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for metric in metrics:
        for name in GROUPS[metric]:
            data = load(root, name, ids_key)
            partitions = data["distances"].shape[1]
            data["coverage"] = {
                k: coverage(data["ranks"], k, partitions) for k in k_values
            }
            report[name] = {}
            for k in k_values:
                rows = {}
                if k <= 100:
                    rows["existing"] = evaluate(metric, data, k, EXISTING_K100[metric])
                else:
                    for label, bucket in structures.items():
                        anchor = bucket(min(k, ANCHORS[-1]))
                        profile = result["profiles"][metric][str(anchor)]
                        rows[label] = evaluate(
                            metric,
                            data,
                            k,
                            (profile["margin"], profile["floor"], profile["cap"]),
                        )
                    if str(k) in result["profiles"][metric]:
                        profile = result["profiles"][metric][str(k)]
                        rows["anchor_profile"] = evaluate(
                            metric,
                            data,
                            k,
                            (profile["margin"], profile["floor"], profile["cap"]),
                        )
                    rows["legacy"] = legacy(metric, data, k)
                report[name][str(k)] = rows
    save(path, report)
    return report


def legacy(metric, data, k):
    """Main's k > 100 heuristic: signed distances below 81x the nearest."""
    distances = data["distances"]
    threshold = distances[:, :1] * np.float32(81.0)
    budgets = np.maximum((distances <= threshold).sum(axis=1), 1)
    cover = data["coverage"][k]
    recall = cover[np.arange(len(budgets)), budgets] / k
    return {
        "recall": float(recall.mean()),
        "partitions": distribution(budgets),
        "rows": distribution(data["rows"][np.arange(len(budgets)), budgets - 1]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--metrics", nargs="+", default=list(GROUPS))
    parser.add_argument(
        "--simulate-only",
        action="store_true",
        help="reuse calibration.json and only simulate structures on calibration",
    )
    args = parser.parse_args()
    k_values = sorted(set(ANCHORS + EXTRAPOLATION_K + MEASURED_K))
    if args.simulate_only:
        result = json.loads((args.root / "calibration.json").read_text())
    else:
        result = calibrate(args.root, k_values, args.metrics)
    simulate(args.root, result, "calibration", args.metrics)
    emit("frozen", profiles={m: result["profiles"][m] for m in args.metrics})
