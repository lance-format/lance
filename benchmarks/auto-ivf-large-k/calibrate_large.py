# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Fit the two large-k anchors using only the original calibration queries."""

import argparse
import json
from pathlib import Path

import numpy as np
from calibrate import (
    COARSE_MARGINS,
    FIXED,
    coverage,
    distribution,
    evaluate,
    geometric,
    refine,
    search,
)
from common import CALIBRATION_TARGET, EVALUATION_TARGET, GROUPS, emit, predict, save
from prepare_large import K_VALUES, sha256


def load(root, name, split):
    out = root / name
    ids = np.load(out / "split.npz")[split]
    data = {
        "ids": ids,
        "distances": np.load(out / "distances.npy", mmap_mode="r")[ids],
        "rows": np.load(out / "scanned_rows.npy", mmap_mode="r")[ids],
    }
    ranks = np.load(out / "ranks.npy", mmap_mode="r")[ids]
    data["coverage"] = {
        k: coverage(ranks, k, data["distances"].shape[1]) for k in K_VALUES
    }
    return data


def calibrate(root, metrics):
    path = root / "calibration.json"
    result = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "target_recall": CALIBRATION_TARGET,
            "evaluation_used": False,
            "anchors": K_VALUES,
            "grid": {
                "coarse_margins": COARSE_MARGINS.tolist(),
                "floor_cap_ratio": 1.08,
                "refinement": (
                    "up to 41 integer floors/caps within 15% (at least 4), "
                    "margin +-0.05 by 0.0025"
                ),
            },
            "profiles": {},
            "per_corpus": {},
            "fixed": {},
            "prepared_sha256": {},
        }
    )
    for metric in metrics:
        names = GROUPS[metric]
        corpora = {name: load(root, name, "calibration") for name in names}
        for name, data in corpora.items():
            result["prepared_sha256"][name] = sha256(root / name / "prepared.json")
            partitions = data["distances"].shape[1]
            result["fixed"][name] = {
                str(k): [
                    {
                        "nprobes": budget,
                        "recall": float(data["coverage"][k][:, budget].mean() / k),
                        "rows": distribution(data["rows"][:, budget - 1]),
                    }
                    for budget in sorted(
                        {b for b in FIXED if b < partitions} | {partitions}
                    )
                ]
                for k in K_VALUES
            }
            result["per_corpus"][name] = {}
        result["profiles"][metric] = {}
        limit = max(data["distances"].shape[1] for data in corpora.values())
        grid = geometric(limit)
        for k in K_VALUES:
            coarse = search(metric, corpora, k, COARSE_MARGINS, grid, grid)
            emit("coarse", metric=metric, k=k, best=coarse)
            fine = refine(metric, corpora, k, coarse["common"], limit)
            result["profiles"][metric][str(k)] = fine["common"]
            for name in names:
                tuned = refine(metric, {name: corpora[name]}, k, coarse[name], limit)
                result["per_corpus"][name][str(k)] = tuned[name]
            save(path, result)
            emit("refined", metric=metric, k=k, profile=fine["common"])
        del corpora
    return result


def simulate(root, result, split, metrics):
    path = root / f"simulation-{split}.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    for metric in metrics:
        for name in GROUPS[metric]:
            data = load(root, name, split)
            report[name] = {}
            for k in K_VALUES:
                shared = result["profiles"][metric][str(k)]
                tuned = result["per_corpus"][name][str(k)]
                profiles = {"shared": shared, "tuned": tuned}
                rows = {}
                for label, profile in profiles.items():
                    values = (profile["margin"], profile["floor"], profile["cap"])
                    rows[label] = evaluate(metric, data, k, values)
                    budgets = predict(metric, data["distances"], values)
                    available = data["rows"][np.arange(len(budgets)), budgets - 1]
                    rows[label]["queries_with_initial_rows_below_k"] = int(
                        np.sum(available < k)
                    )
                report[name][str(k)] = rows
            del data
            emit("simulated", dataset=name, split=split, result=report[name])
    save(path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--metrics", nargs="+", choices=list(GROUPS), default=list(GROUPS)
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="use already frozen profiles on held-out queries",
    )
    args = parser.parse_args()
    if args.evaluate:
        frozen = json.loads((args.root / "calibration-frozen.json").read_text())
        assert sha256(args.root / "calibration.json") == sha256(
            args.root / "calibration-frozen.json"
        )
        report = simulate(args.root, frozen, "evaluation", args.metrics)
        emit(
            "quality",
            target=EVALUATION_TARGET,
            passed={
                f"{name}/{k}": row["shared"]["recall"] >= EVALUATION_TARGET
                for name, rows in report.items()
                for k, row in rows.items()
            },
        )
    else:
        result = calibrate(args.root, args.metrics)
        simulate(args.root, result, "calibration", args.metrics)
