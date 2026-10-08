# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Shared definitions for the k > 100 Auto IVF probing study."""

import json
import time

import numpy as np

# Each corpus is evaluated with the metric its embeddings were trained for.
DATASETS = {
    "dino-10m": "l2",
    "laion-10m": "cosine",
    "fineweb-10m": "cosine",
    "wiki-cohere-35m": "dot",
    "dpr-wikipedia-single-nq": "dot",
}
GROUPS = {
    metric: [name for name, value in DATASETS.items() if value == metric]
    for metric in ["l2", "cosine", "dot"]
}
# Calibrated k > 100 buckets end at these values. k=100 is the existing
# 11-100 profile and only serves as the boundary control.
ANCHORS = [200, 500, 1000]
MEASURED_K = [100, 101, 200, 500, 1000]
# Routing-only evidence for k beyond the largest calibrated anchor.
EXTRAPOLATION_K = [2000, 5000]
MAX_K = max(EXTRAPOLATION_K)
SPLIT_SEED = 2254
# A selective prefilter on row ids leaves fewer than k matches in the initial
# budget for large k, exercising late probing.
FILTER_STRIDE = 1000
FILTER = f"_rowid % {FILTER_STRIDE} = 0"
FILTERED_K = 1000
CALIBRATION_TARGET = 0.96
EVALUATION_TARGET = 0.95
# Existing k <= 100 profiles, (margin, floor, cap), used as the k=100 control.
EXISTING_K100 = {
    "l2": (0.33, 11, 38),
    "cosine": (0.38, 7, 106),
    "dot": (0.0625, 200, 768),
}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def emit(event, **values):
    print(json.dumps({"time": time.time(), "event": event, **values}), flush=True)


def matrix(column):
    array = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    return array.values.to_numpy(zero_copy_only=False).reshape(
        len(array), array.type.list_size
    )


def row_ids_by_position(dataset):
    """`_rowid` of every source row in position order.

    Works for both row-address and stable row id datasets, which these corpora mix.
    """
    table = dataset.to_table(columns=[], with_row_id=True)
    row_ids = table["_rowid"].to_numpy()
    assert np.all(np.diff(row_ids.astype(np.int64)) > 0)
    return row_ids


def positions(row_ids, lookup):
    """Map `_rowid` values back to source row positions."""
    result = np.searchsorted(lookup, row_ids)
    assert np.all(lookup[np.minimum(result, len(lookup) - 1)] == row_ids)
    return result.astype(np.int64)


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        **{f"p{p}": float(np.percentile(values, p)) for p in [50, 90, 95, 99]},
        "max": float(np.max(values)),
    }


def routing_distances(metric, queries, centroids):
    """Sorted centroid distances as native Auto sees them, in float32.

    Cosine indices route normalized queries with squared L2. Dot uses the signed
    `1 - dot` distance. Distances are computed in float64 and rounded once.
    """
    queries = queries.astype(np.float64)
    centroids = centroids.astype(np.float64)
    if metric == "cosine":
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    products = queries @ centroids.T
    if metric == "dot":
        distances = 1.0 - products
    else:
        distances = (
            np.sum(queries**2, axis=1)[:, None]
            - 2.0 * products
            + np.sum(centroids**2, axis=1)[None, :]
        )
        np.maximum(distances, 0.0, out=distances)
    distances = distances.astype(np.float32)
    order = np.argsort(distances, axis=1, kind="stable")
    return np.take_along_axis(distances, order, axis=1), order


def gap_counts(metric, distances, margin):
    """Number of leading partitions within the relative gap, as in Rust.

    Rust stores the margin as f32 and widens every operand to f64.
    """
    distances = distances.astype(np.float64)
    nearest = distances[:, :1]
    scale = np.abs(1.0 - nearest) if metric == "dot" else nearest
    margin = float(np.float32(margin))
    return ((distances - nearest) <= margin * scale).sum(axis=1)


def predict(metric, distances, profile):
    """Initial budget of a (margin, floor, cap) profile with default caller bounds."""
    margin, floor, cap = profile
    counts = gap_counts(metric, distances, margin)
    partitions = distances.shape[1]
    return np.minimum(np.minimum(np.maximum(counts, floor), cap), partitions)
