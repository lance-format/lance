# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Small independent checks for fallback threshold calibration."""

import math

import numpy as np
import pytest
from calibrate_fallback import (
    SIGNED_FACTORS,
    counts,
    evaluate,
    fit,
    fit_all_k,
    recall_curve,
    signed_counts,
    signed_dot_bounds,
)


def scalar_passes(data, metric, family, threshold, ks):
    hits = {k: 0 for k in ks}
    for distances, ranks in zip(data["distances"], data["ranks"]):
        nearest = float(distances[0])
        if family == "signed":
            boundary = float(np.float32(distances[0] * np.float32(threshold)))
            budget = sum(float(value) <= boundary for value in distances)
        else:
            scale = abs(1 - nearest) if metric == "dot" else nearest
            budget = sum(
                float(value) - nearest <= float(np.float32(threshold)) * scale
                for value in distances
            )
        for k in ks:
            hits[k] += sum(int(rank) <= max(budget, 1) for rank in ranks[:k])
    return all(hits[k] >= math.ceil(0.96 * len(data["ranks"]) * k) for k in ks)


@pytest.mark.parametrize(
    "metric,family",
    [
        ("l2", "gap"),
        ("cosine", "gap"),
        ("dot", "gap"),
        ("l2", "signed"),
        ("cosine", "signed"),
    ],
)
def test_minimal_f32_threshold_matches_scalar_comparisons(metric, family):
    rng = np.random.default_rng(2254)
    distances = np.sort(rng.uniform(0.25, 3.0, size=(17, 11)), axis=1).astype(
        np.float32
    )
    if metric == "dot":
        distances -= np.float32(1.5)
    data = {
        "distances": distances,
        "ranks": rng.integers(1, 12, size=(17, 13), dtype=np.int32),
    }
    ks = [1, 2, 7, 13]
    result = fit(metric, data, family, ks)
    assert result["feasible"]
    threshold = np.float32(result["threshold"])
    assert scalar_passes(data, metric, family, threshold, ks)
    if threshold > 0:
        previous = np.nextafter(threshold, np.float32(0))
        assert not scalar_passes(data, metric, family, previous, ks)


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
def test_zero_scale_can_make_gap_target_infeasible(metric):
    nearest = 1.0 if metric == "dot" else 0.0
    data = {
        "distances": np.array([[nearest, nearest + 1]], dtype=np.float32),
        "ranks": np.array([[2]], dtype=np.int32),
    }
    result = fit(metric, data, "gap", [1])
    assert not result["feasible"]
    assert result["critical_values"] == {"1": None}


@pytest.mark.parametrize("metric", ["l2", "cosine", "dot"])
def test_zero_scale_includes_all_nearest_ties(metric):
    nearest = 1.0 if metric == "dot" else 0.0
    data = {
        "distances": np.array([[nearest, nearest, nearest + 1]], dtype=np.float32),
        "ranks": np.array([[1, 2]], dtype=np.int32),
    }
    result = fit(metric, data, "gap", [1, 2])
    assert result["feasible"] and result["threshold"] == 0
    np.testing.assert_array_equal(counts(metric, data, "gap", 0), [2])


def test_zero_multiplier_preserves_caller_minimum():
    data = {
        "distances": np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
        "ranks": np.array([[1, 1]], dtype=np.int32),
    }
    result = fit("l2", data, "signed", [1, 2])
    assert result["threshold"] == 0
    np.testing.assert_array_equal(signed_counts(data["distances"], 0), [1])


def test_all_k_fit_checks_prefixes_between_reporting_anchors():
    # Missing one neighbor at rank two is tolerable at k=100 but not at k=2.
    ranks = np.ones((2, 100), dtype=np.int32)
    ranks[:, 1] = 2
    data = {
        "distances": np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32),
        "ranks": ranks,
    }
    anchors_only = fit("l2", data, "gap", [1, 100])
    assert anchors_only["threshold"] == 0
    fitted = fit_all_k("l2", data, "gap", anchors=[1, 100])
    assert fitted["threshold"] == 1
    assert fitted["all_k"]["k_below_96_percent"] == 0
    curve = recall_curve(data, counts("l2", data, "gap", fitted["threshold"]))
    np.testing.assert_array_equal(curve, np.ones(100))


@pytest.mark.parametrize("nearest", [-3.0, 0.0, 0.25])
def test_signed_dot_bounds_are_optimistic(nearest):
    data = {
        "distances": np.array(
            [[nearest, nearest + 0.5, nearest + 1]], dtype=np.float32
        ),
        "ranks": np.array([[3, 2, 3, 1]], dtype=np.int32),
        "rows": np.array([[1, 3, 6]], dtype=np.int64),
    }
    ks = [1, 2, 4]
    bounds = signed_dot_bounds(data, ks)
    for factor in SIGNED_FACTORS:
        actual = evaluate(data, counts("dot", data, "signed", factor), ks)
        domain = "factor_lt_one" if factor < 1 else "factor_ge_one"
        for k in ks:
            assert (
                actual[str(k)]["count_expanded_recall"]
                <= bounds[domain][str(k)]["count_expanded_recall"]
            )


def test_count_expansion_does_not_imply_recall():
    data = {
        "ranks": np.array([[3, 3]], dtype=np.int32),
        "rows": np.array([[1, 3, 6]], dtype=np.int64),
    }
    row = evaluate(data, np.array([1]), [2])["2"]
    assert row["initial_queries_below_k"] == 1
    assert row["count_expanded_partitions"]["mean"] == 2
    assert row["count_expanded_recall"] == 0
