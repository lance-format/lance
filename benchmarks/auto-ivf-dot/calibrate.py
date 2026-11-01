# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Select dot probing parameters using calibration queries only."""

import argparse
from pathlib import Path

import numpy as np
from prepare import emit, save

NAMES = ["wiki-cohere-35m", "dpr-wikipedia-single-nq"]
FLOORS = sorted(set([1, 2, 4, 6, 12, 384, 512, *range(8, 257, 8)]))
CAPS = [
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    16384,
]
CAPS = sorted(set(CAPS + list(range(16, 769, 16))))
MARGINS = {
    "relative_score": np.r_[
        np.arange(0, 0.1005, 0.0005),
        np.arange(0.105, 1.005, 0.005),
        np.arange(1.05, 4.05, 0.05),
    ],
    "standard_deviation": np.arange(0, 8.025, 0.025),
}


def load(root, name):
    routes = np.load(root / name / "routes.npz")
    ids = np.load(root / name / "split.npz")["calibration"]
    scores = routes["scores"][ids].astype(np.float64)
    ranks = routes["ranks"][ids]
    rows = routes["scanned_rows"][ids]
    # The denominator is homogeneous in the query, unlike abs(1 - dot).
    scales = {
        "relative_score": np.abs(scores[:, :1]),
        "standard_deviation": np.std(scores, axis=1, keepdims=True),
    }
    gaps = {}
    for mode, scale in scales.items():
        gaps[mode] = np.divide(
            scores[:, :1] - scores,
            scale,
            out=np.full_like(scores, np.inf),
            where=scale != 0,
        )
        gaps[mode][scores == scores[:, :1]] = 0
    coverage = {}
    for k in [1, 10, 100]:
        counts = np.zeros((len(ids), scores.shape[1] + 1), dtype=np.int16)
        np.add.at(counts, (np.arange(len(ids))[:, None], ranks[:, :k]), 1)
        coverage[k] = np.cumsum(counts, axis=1, dtype=np.int16)
    return {
        "scores": scores,
        "gaps": gaps,
        "coverage": coverage,
        "rows": rows,
        "ids": ids,
    }


def summary(values):
    return {
        "mean": float(np.mean(values)),
        **{f"p{p}": float(np.percentile(values, p)) for p in [90, 95, 99]},
        "max": float(np.max(values)),
    }


def calibration(root):
    data = {name: load(root, name) for name in NAMES}
    result = {
        "target_recall": 0.96,
        "evaluation_used": False,
        "grid": {
            "floors": FLOORS,
            "caps": CAPS,
            "margins": {key: value.tolist() for key, value in MARGINS.items()},
        },
        "fixed": {},
        "profiles": {},
    }
    for name, values in data.items():
        result["fixed"][name] = {}
        for k in [1, 10, 100]:
            budgets = [
                p
                for p in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
                if p < values["scores"].shape[1]
            ] + [values["scores"].shape[1]]
            rows = [
                {
                    "nprobes": budget,
                    "recall": float(values["coverage"][k][:, budget].mean() / k),
                    "scanned_rows": summary(values["rows"][:, budget - 1]),
                }
                for budget in budgets
            ]
            result["fixed"][name][str(k)] = rows
    # Keep separate optima for each corpus and a common profile satisfying both.
    # 96% calibration provides headroom for the independent 95% target.
    for k in [1, 10, 100]:
        best = {}
        for mode, margins in MARGINS.items():
            best[mode] = {key: None for key in NAMES + ["common"]}
            for margin in margins:
                selected = {
                    name: np.array(
                        [
                            np.searchsorted(row, margin, side="right")
                            for row in values["gaps"][mode]
                        ]
                    )
                    for name, values in data.items()
                }
                for floor in FLOORS:
                    caps = np.asarray([cap for cap in CAPS if cap >= floor])
                    metrics = {}
                    for name, values in data.items():
                        budgets = np.minimum(
                            np.maximum(selected[name][:, None], floor), caps[None, :]
                        )
                        budgets = np.minimum(budgets, values["scores"].shape[1])
                        recall = (
                            values["coverage"][k][
                                np.arange(len(budgets))[:, None], budgets
                            ].mean(axis=0)
                            / k
                        )
                        cost = budgets.mean(axis=0)
                        metrics[name] = (recall, cost)
                        valid = np.flatnonzero(recall >= result["target_recall"])
                        if len(valid):
                            index = valid[np.argmin(cost[valid])]
                            candidate = {
                                "margin": float(margin),
                                "floor": floor,
                                "cap": int(caps[index]),
                                "recall": float(recall[index]),
                                "mean_partitions": float(cost[index]),
                            }
                            old = best[mode][name]
                            if (
                                old is None
                                or candidate["mean_partitions"] < old["mean_partitions"]
                            ):
                                best[mode][name] = candidate
                    valid = np.flatnonzero(
                        np.logical_and.reduce(
                            [
                                metrics[name][0] >= result["target_recall"]
                                for name in NAMES
                            ]
                        )
                    )
                    cost = np.mean([metrics[name][1] for name in NAMES], axis=0)
                    if len(valid):
                        index = valid[np.argmin(cost[valid])]
                        candidate = {
                            "mode": mode,
                            "margin": float(margin),
                            "floor": floor,
                            "cap": int(caps[index]),
                            "mean_partitions": float(cost[index]),
                            "datasets": {
                                name: {
                                    "recall": float(metrics[name][0][index]),
                                    "mean_partitions": float(metrics[name][1][index]),
                                }
                                for name in NAMES
                            },
                        }
                        old = best[mode]["common"]
                        if (
                            old is None
                            or candidate["mean_partitions"] < old["mean_partitions"]
                        ):
                            best[mode]["common"] = candidate
            emit("calibrated", k=k, mode=mode, best=best[mode])
            result["profiles"][str(k)] = best
            save(root / "calibration.json", result)
    # Freeze both common families before ever examining evaluation outcomes.
    result["selected_mode"] = min(
        MARGINS,
        key=lambda mode: sum(
            result["profiles"][str(k)][mode]["common"]["mean_partitions"]
            for k in [1, 10, 100]
        ),
    )
    result["selected"] = {
        str(k): result["profiles"][str(k)][result["selected_mode"]]["common"]
        for k in [1, 10, 100]
    }
    save(root / "calibration.json", result)
    emit("frozen", mode=result["selected_mode"], profiles=result["selected"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    calibration(args.root)
