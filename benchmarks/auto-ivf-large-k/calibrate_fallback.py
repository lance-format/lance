# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Calibrate one fallback threshold per metric without using held-out queries."""

import argparse
import json
from pathlib import Path

import numpy as np
from build_large_patch import BASE_PROFILES
from common import GROUPS, emit, gap_counts, predict, save, summary
from prepare_large import sha256

K_VALUES = [
    1,
    2,
    10,
    11,
    100,
    101,
    200,
    500,
    1000,
    1001,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
]
TARGET_PERCENT = 96
SIGNED_FACTORS = [
    -256.0,
    -81.0,
    -7.0,
    -1.0,
    0.0,
    0.1,
    0.5,
    0.6,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    20.0,
    40.0,
    81.0,
    160.0,
    256.0,
]


def load(source, name, split):
    directory = source / name
    ids = np.load(directory / "split.npz")[split]
    data = {
        "ids": ids,
        "distances": np.load(directory / "distances.npy", mmap_mode="r")[ids],
        "ranks": np.load(directory / "ranks.npy", mmap_mode="r")[ids],
        "rows": np.load(directory / "scanned_rows.npy", mmap_mode="r")[ids],
    }
    assert data["ranks"].shape == (len(ids), max(K_VALUES))
    assert data["rows"].shape == data["distances"].shape
    return data


def signed_counts(distances, factor):
    with np.errstate(over="ignore", invalid="ignore"):
        threshold = distances[:, :1] * np.float32(factor)
    return np.maximum((distances <= threshold).sum(axis=1), 1)


def counts(metric, data, family, threshold):
    if family == "signed":
        return signed_counts(data["distances"], threshold)
    assert family == "gap"
    return np.maximum(gap_counts(metric, data["distances"], threshold), 1)


def required_hits(total):
    return (TARGET_PERCENT * total + 99) // 100


def passes(data, budgets, ks):
    for k in ks:
        hits = int(np.count_nonzero(data["ranks"][:, :k] <= budgets[:, None]))
        if hits < required_hits(len(budgets) * k):
            return False
    return True


def critical_values(metric, data, family, ks):
    """The exact mean-recall constraint is an order statistic over true neighbors."""
    distances = data["distances"].astype(np.float64)
    nearest = distances[:, :1]
    if family == "gap":
        numerator = distances - nearest
        scale = np.abs(1.0 - nearest) if metric == "dot" else nearest
    else:
        assert family == "signed" and metric != "dot"
        assert np.all(nearest >= 0)
        numerator, scale = distances, nearest
    required = np.full(distances.shape, np.inf)
    np.divide(numerator, scale, out=required, where=scale != 0)
    required[(scale == 0) & (numerator == 0)] = 0
    # The caller minimum always searches the first partition, even when a
    # multiplier below one selects no positive distances.
    required[:, 0] = 0
    result = {}
    for k in ks:
        values = np.take_along_axis(required, data["ranks"][:, :k] - 1, axis=1)
        flat = values.reshape(-1)
        ordinal = required_hits(len(flat)) - 1
        flat.partition(ordinal)
        result[str(k)] = float(flat[ordinal])
    return result


def fit(metric, data, family, ks):
    critical = critical_values(metric, data, family, ks)
    seed = max(critical.values())
    if not np.isfinite(seed) or seed > np.finfo(np.float32).max:
        return {
            "feasible": False,
            "critical_values": {
                k: value if np.isfinite(value) else None
                for k, value in critical.items()
            },
        }
    threshold = np.float32(seed)
    ordered_ks = sorted(ks, key=lambda k: critical[str(k)], reverse=True)
    for _ in range(64):
        budgets = counts(metric, data, family, threshold)
        if passes(data, budgets, ordered_ks):
            break
        threshold = np.nextafter(threshold, np.float32(np.inf))
    else:
        raise AssertionError("Order-statistic seed did not approach feasibility")
    assert np.isfinite(threshold)
    for _ in range(64):
        if threshold == 0:
            break
        previous = np.nextafter(threshold, np.float32(0))
        if not passes(data, counts(metric, data, family, previous), ordered_ks):
            break
        threshold = previous
    else:
        raise AssertionError("Order-statistic seed did not approach minimality")
    return {
        "feasible": True,
        "threshold": float(threshold),
        "f32_bits": int(threshold.view(np.uint32)),
        "preceding_f32_fails": bool(threshold != 0),
        "critical_values": critical,
    }


def recall_curve(data, budgets):
    covered_by_rank = np.count_nonzero(data["ranks"] <= budgets[:, None], axis=0)
    covered = np.cumsum(covered_by_rank, dtype=np.int64)
    totals = len(budgets) * np.arange(1, len(covered) + 1, dtype=np.int64)
    return covered / totals


def curve_summary(curve):
    return {
        "k_range": [1, len(curve)],
        "minimum_initial_recall": float(curve.min()),
        "k_at_minimum": int(curve.argmin()) + 1,
        "k_below_96_percent": int(np.count_nonzero(curve < 0.96)),
        "k_below_95_percent": int(np.count_nonzero(curve < 0.95)),
    }


def fit_all_k(metric, data, family, anchors=None):
    maximum = data["ranks"].shape[1]
    constraints = sorted(
        {k for k in (K_VALUES if anchors is None else anchors) if k <= maximum}
        | {maximum}
    )
    for _ in range(32):
        fitted = fit(metric, data, family, constraints)
        if not fitted["feasible"]:
            return fitted
        curve = recall_curve(data, counts(metric, data, family, fitted["threshold"]))
        failed = np.flatnonzero(curve < TARGET_PERCENT / 100)
        if not len(failed):
            return {**fitted, "all_k": curve_summary(curve)}
        # A failing prefix adds an exact constraint. At convergence the complete
        # range passes, while a preceding f32 still fails an included constraint.
        worst = int(failed[np.argmin(curve[failed])]) + 1
        assert worst not in constraints
        constraints = sorted([*constraints, worst])
    raise AssertionError("All-k calibration did not converge in 32 refinements")


def evaluate(data, budgets, ks):
    ordinals = np.arange(len(budgets))
    initial_rows = data["rows"][ordinals, budgets - 1]
    report = {}
    for k in ks:
        fill = np.argmax(data["rows"] >= k, axis=1) + 1
        assert np.all(data["rows"][ordinals, fill - 1] >= k)
        final = np.maximum(budgets, fill)
        initial_recall = float(np.mean(data["ranks"][:, :k] <= budgets[:, None]))
        final_recall = (
            initial_recall
            if np.array_equal(final, budgets)
            else float(np.mean(data["ranks"][:, :k] <= final[:, None]))
        )
        report[str(k)] = {
            "queries": len(budgets),
            "initial_recall": initial_recall,
            "initial_partitions": summary(budgets),
            "initial_scanned_rows": summary(initial_rows),
            "initial_queries_below_k": int(np.count_nonzero(initial_rows < k)),
            "count_expanded_recall": final_recall,
            "count_expanded_partitions": summary(final),
            "count_expanded_scanned_rows": summary(data["rows"][ordinals, final - 1]),
        }
    return report


def signed_dot_bounds(data, ks):
    """Optimistic bounds over all finite factors, including count-based expansion."""
    distances = data["distances"]
    nearest = distances[:, 0]
    first = signed_counts(distances, 1.0)
    full = np.full(len(nearest), distances.shape[1], dtype=np.int64)
    bounds = {}
    for label, constrained in [
        ("factor_lt_one", nearest > 0),
        ("factor_ge_one", nearest < 0),
    ]:
        # Constrained signs can search at most the nearest-distance ties.
        # Treat the other sign as exhaustive, making the bound optimistic.
        budgets = np.where(constrained | (nearest == 0), first, full)
        bounds[label] = evaluate(data, budgets, ks)
    return bounds


def source_identity(source, names):
    return {
        "source": str(source.resolve()),
        "large_k_calibration_sha256": sha256(source / "calibration-frozen.json"),
        "prepared_sha256": {
            name: sha256(source / name / "prepared.json") for name in names
        },
        "split_sha256": {name: sha256(source / name / "split.npz") for name in names},
    }


def calibrate(source, root, metrics):
    destination = root / "calibration.json"
    assert not destination.exists(), "Preserve earlier fits; choose another directory"
    root.mkdir(parents=True, exist_ok=True)
    names = [name for metric in metrics for name in GROUPS[metric]]
    result = {
        "target_recall": TARGET_PERCENT / 100,
        "evaluation_used": False,
        "k": K_VALUES,
        "calibration_k_range": [1, max(K_VALUES)],
        "calibration_all_k": True,
        "profiles": {},
        "per_corpus": {},
        "calibration_results": {},
        "harness_sha256": {
            filename: sha256(Path(__file__).with_name(filename))
            for filename in [
                "calibrate_fallback.py",
                "common.py",
                "prepare_large.py",
                "prepare.py",
                "build_large_patch.py",
                "FALLBACK_PROTOCOL.md",
            ]
        },
        **source_identity(source, names),
    }
    for metric in metrics:
        corpora = {name: load(source, name, "calibration") for name in GROUPS[metric]}
        result["profiles"][metric] = {}
        for name, data in corpora.items():
            nearest = data["distances"][:, 0]
            result["per_corpus"][name] = {
                "nearest_signs": {
                    "negative": int(np.count_nonzero(nearest < 0)),
                    "zero": int(np.count_nonzero(nearest == 0)),
                    "positive": int(np.count_nonzero(nearest > 0)),
                }
            }
            for family in ["gap"] if metric == "dot" else ["gap", "signed"]:
                fitted = fit_all_k(metric, data, family)
                result["per_corpus"][name][family] = fitted
                emit("fallback_fitted", dataset=name, family=family, fit=fitted)
        for family in ["gap"] if metric == "dot" else ["gap", "signed"]:
            fitted = [result["per_corpus"][name][family] for name in corpora]
            feasible = all(item["feasible"] for item in fitted)
            profile = {"feasible": feasible}
            if feasible:
                threshold = max(item["threshold"] for item in fitted)
                profile.update(
                    threshold=threshold,
                    f32_bits=int(np.float32(threshold).view(np.uint32)),
                    preceding_f32_fails=threshold != 0,
                    datasets={},
                    all_k={},
                )
                for name, data in corpora.items():
                    budgets = counts(metric, data, family, threshold)
                    assert passes(data, budgets, K_VALUES)
                    profile["datasets"][name] = evaluate(data, budgets, K_VALUES)
                    curve = recall_curve(data, budgets)
                    assert np.all(curve >= TARGET_PERCENT / 100)
                    profile["all_k"][name] = curve_summary(curve)
            result["profiles"][metric][family] = profile
        for name, data in corpora.items():
            rows = {}
            for k in K_VALUES:
                factor = 0.6 if k == 1 else 7.0 if k <= 10 else 81.0
                rows[str(k)] = evaluate(
                    data, signed_counts(data["distances"], factor), [k]
                )[str(k)]
            result["calibration_results"][name] = {"current": rows}
            if metric == "dot":
                result["calibration_results"][name]["signed_upper_bounds"] = (
                    signed_dot_bounds(data, K_VALUES)
                )
                result["calibration_results"][name]["signed_sweep"] = {
                    str(factor): evaluate(
                        data, signed_counts(data["distances"], factor), K_VALUES
                    )
                    for factor in SIGNED_FACTORS
                }
        if metric == "dot":
            result["profiles"][metric]["signed"] = {
                "infeasible_proven": all(
                    any(
                        row["count_expanded_recall"] < TARGET_PERCENT / 100
                        for name in corpora
                        for row in result["calibration_results"][name][
                            "signed_upper_bounds"
                        ][domain].values()
                    )
                    for domain in ["factor_lt_one", "factor_ge_one"]
                ),
                "proof": "Optimistic upper bounds for factor<1 and factor>=1",
            }
        save(destination, result)
        emit("fallback_metric_complete", metric=metric)
        del corpora
    return result


def reference_profile(metric, k, large):
    endpoints = [1, 10, 100, 200, 500, 1000]
    if k <= endpoints[-1]:
        index = next(index for index, maximum in enumerate(endpoints) if k <= maximum)
        return tuple(values[index] for values in BASE_PROFILES[metric])
    if k in [10000, 100000]:
        profile = large["profiles"][metric][str(k)]
        return profile["margin"], profile["floor"], profile["cap"]
    return None


def simulate(source, root, split):
    frozen = root / "calibration-frozen.json"
    assert sha256(frozen) == sha256(root / "calibration.json")
    calibration = json.loads(frozen.read_text())
    assert calibration["evaluation_used"] is False and calibration["k"] == K_VALUES
    for filename, expected in calibration["harness_sha256"].items():
        assert sha256(Path(__file__).with_name(filename)) == expected, filename
    names = list(calibration["prepared_sha256"])
    for key, value in source_identity(source, names).items():
        assert calibration[key] == value, key
    large = json.loads((source / "calibration-frozen.json").read_text())
    report = {
        "split": split,
        "calibration_sha256": sha256(frozen),
        "datasets": {},
        "all_k": {},
        "curve_sha256": {},
    }
    for metric, profiles in calibration["profiles"].items():
        for name in GROUPS[metric]:
            data = load(source, name, split)
            policies = {}
            report["all_k"][name] = {}
            curve_directory = root / name
            curve_directory.mkdir(exist_ok=True)
            for family in ["gap", "signed"]:
                profile = profiles[family]
                if profile.get("feasible"):
                    budgets = counts(metric, data, family, profile["threshold"])
                    policies[family] = evaluate(data, budgets, K_VALUES)
                    curve = recall_curve(data, budgets)
                    curve_path = curve_directory / f"routing-{split}-{family}.npy"
                    np.save(curve_path, curve)
                    report["curve_sha256"][str(curve_path.relative_to(root))] = sha256(
                        curve_path
                    )
                    report["all_k"][name][family] = curve_summary(curve)
            policies["current"] = {}
            policies["reference_profiles"] = {}
            for k in K_VALUES:
                factor = 0.6 if k == 1 else 7.0 if k <= 10 else 81.0
                policies["current"][str(k)] = evaluate(
                    data, signed_counts(data["distances"], factor), [k]
                )[str(k)]
                profile = reference_profile(metric, k, large)
                if profile is not None:
                    policies["reference_profiles"][str(k)] = evaluate(
                        data, predict(metric, data["distances"], profile), [k]
                    )[str(k)]
            report["datasets"][name] = policies
            save(root / f"simulation-{split}.json", report)
            emit("fallback_simulated", dataset=name, split=split)
            del data
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--metrics", nargs="+", choices=list(GROUPS), default=list(GROUPS)
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--calibration-simulation", action="store_true")
    args = parser.parse_args()
    if args.evaluate or args.calibration_simulation:
        simulate(
            args.source,
            args.root,
            "evaluation" if args.evaluate else "calibration",
        )
    else:
        calibrate(args.source, args.root, args.metrics)
