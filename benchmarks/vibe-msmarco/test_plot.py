# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Unit tests for the VIBE IVF_RQ latency chart helpers."""

from __future__ import annotations

import json
from pathlib import Path

from plot import load_results, plot_results


def _sample_result(label: str, num_bits: int, warm_mean: float, cold_mean: float) -> dict:
    return {
        "label": label,
        "index": f"IVF_RQ{num_bits}",
        "num_bits": num_bits,
        "warm": {
            "mode": "warm",
            "summary": {
                "count": 3,
                "mean_ms": warm_mean,
                "median_ms": warm_mean - 0.2,
                "p95_ms": warm_mean + 0.4,
                "min_ms": warm_mean - 0.5,
                "max_ms": warm_mean + 0.6,
                "qps": 1000.0 / warm_mean,
            },
        },
        "cold": {
            "mode": "cold",
            "summary": {
                "count": 3,
                "mean_ms": cold_mean,
                "median_ms": cold_mean - 0.3,
                "p95_ms": cold_mean + 0.8,
                "min_ms": cold_mean - 0.7,
                "max_ms": cold_mean + 1.0,
                "qps": 1000.0 / cold_mean,
            },
        },
    }


def test_load_and_plot(tmp_path: Path) -> None:
    rows = [
        _sample_result("v11.0.0", 1, 4.0, 12.0),
        _sample_result("v12.0.0", 1, 3.2, 10.5),
        _sample_result("c8f182179 (main)", 1, 2.8, 9.4),
        _sample_result("v11.0.0", 5, 6.5, 18.0),
        _sample_result("v12.0.0", 5, 5.1, 15.2),
        _sample_result("c8f182179 (main)", 5, 4.4, 13.0),
    ]
    for row in rows:
        name = f"{row['label'].replace(' ', '_')}-rq{row['num_bits']}.json"
        (tmp_path / name).write_text(json.dumps(row) + "\n")
    (tmp_path / "_manifest.json").write_text("{}\n")

    loaded = load_results(tmp_path)
    assert [row["label"] for row in loaded if row["num_bits"] == 1] == [
        "v11.0.0",
        "v12.0.0",
        "c8f182179 (main)",
    ]

    out = tmp_path / "chart.png"
    plot_results(loaded, out, "unit test")
    assert out.is_file()
    assert out.stat().st_size > 1000
