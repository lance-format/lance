#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Render IVF_RQ latency as four charts: {RQ1, RQ5} × {cold, warm}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from bench import nearest_rank


VERSION_ORDER = ("v9.0.1", "v10.0.0", "v11.0.0", "v12.0.0")
MODES = ("cold", "warm")
# Same three encodings in every panel; the panel title is the cache mode.
METRICS = (
    {
        "key": "mean_ms",
        "label": "mean",
        "color": "#e67e22",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.4,
        "markersize": 8,
        "annotate": True,
        "offset": (0, 8),
    },
    {
        "key": "median_ms",
        "label": "median",
        "color": "#7f8c8d",
        "marker": "D",
        "linestyle": "--",
        "linewidth": 1.8,
        "markersize": 7,
        "annotate": False,
        "offset": (0, -11),
    },
    {
        "key": "p99_ms",
        "label": "p99",
        "color": "#c0392b",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.4,
        "markersize": 8,
        "annotate": True,
        "offset": (0, 11),
    },
)


def _sort_key(label: str) -> tuple[int, str]:
    if label in VERSION_ORDER:
        return (VERSION_ORDER.index(label), label)
    return (len(VERSION_ORDER), label)


def axis_label(label: str) -> str:
    """Keep the last tick readable; a SHA plus '(main)' wraps off the axis."""
    if label.endswith(" (main)") or " (main)" in label:
        return "main"
    return label


def panel_filename(out: Path, index_name: str, mode: str) -> Path:
    slug = index_name.lower().replace("ivf_rq", "rq")
    return out.parent / f"ivf_{slug}_{mode}.png"


def _result_paths(results_dir: Path) -> list[Path]:
    manifest = results_dir / "_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        paths = [
            path if path.is_absolute() else results_dir / path
            for path in (Path(item) for item in payload.get("results", []))
        ]
        if paths:
            return paths
    return [
        path
        for path in sorted(results_dir.glob("*.json"))
        if not path.name.startswith("_")
    ]


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in _result_paths(results_dir):
        payload = json.loads(path.read_text())
        if "index" in payload and "warm" in payload and "cold" in payload:
            rows.append(payload)
    rows.sort(key=lambda row: (_sort_key(row["label"]), row["num_bits"]))
    return rows


def series_ms(row: dict, mode: str, key: str) -> float:
    """Read a latency stat, computing p99 from samples when the summary omits it."""
    summary = row[mode]["summary"]
    if key in summary:
        return summary[key]
    if key == "p99_ms":
        samples = row[mode].get("latencies_ms")
        if samples:
            return nearest_rank(sorted(samples), 0.99)
    raise KeyError(f"{mode} summary is missing {key}")


def _style_axis(ax, labels: list[str], title: str) -> None:
    xs = list(range(len(labels)))
    for x in xs:
        ax.axvline(x, color="#bbbbbb", linestyle=":", linewidth=0.8)
    ax.set_xticks(xs, labels)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title(title)
    ax.set_xlabel("Lance / pylance version")
    ax.set_ylabel("Query latency (ms)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1.0)
    ax.set_ylim(ymin - span * 0.08, ymax + span * 0.16)


def _draw_panel(ax, rows: list[dict], mode: str, title: str) -> None:
    labels = [axis_label(row["label"]) for row in rows]
    xs = list(range(len(labels)))
    for spec in METRICS:
        values = [series_ms(row, mode, spec["key"]) for row in rows]
        ax.plot(
            xs,
            values,
            color=spec["color"],
            marker=spec["marker"],
            markersize=spec["markersize"],
            linewidth=spec["linewidth"],
            linestyle=spec["linestyle"],
            label=spec["label"],
        )
        if spec["annotate"]:
            for x, value in zip(xs, values):
                ax.annotate(
                    f"{value:.1f}",
                    (x, value),
                    textcoords="offset points",
                    xytext=spec["offset"],
                    ha="center",
                    fontsize=8,
                    color=spec["color"],
                )
    _style_axis(ax, labels, title)


def _write_single_panel(
    rows: list[dict], mode: str, title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    _draw_panel(ax, rows, mode, title)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_results(rows: list[dict], out: Path, subtitle: str) -> None:
    if not rows:
        raise SystemExit("no result JSON files found")

    indexes = sorted({row["index"] for row in rows})
    n_idx = len(indexes)
    fig, axes = plt.subplots(n_idx, 2, figsize=(13.6, 4.8 * n_idx), sharey=False)
    if n_idx == 1:
        axis_rows = [axes]
    else:
        axis_rows = axes

    for i, index_name in enumerate(indexes):
        series = [row for row in rows if row["index"] == index_name]
        for j, mode in enumerate(MODES):
            title = f"{index_name} {mode}"
            _draw_panel(axis_rows[i][j], series, mode, title)
            _write_single_panel(series, mode, title, panel_filename(out, index_name, mode))

    handles, legend_labels = axis_rows[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle("Lance IVF_RQ search latency over recent versions", fontsize=13, y=1.01)
    fig.text(
        0.5,
        0.01,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--subtitle",
        default=(
            "vibe-msmarco-qwen-1024 · 8.84M × 1024-d · IVF 1024 partitions · "
            "k=10 nprobes=20 · select _rowid only · each version builds its own index · "
            "OS cache dropped then this index primed · first query discarded · "
            "warm = prewarm_index · p99 = nearest-rank of the same 100 queries"
        ),
    )
    args = parser.parse_args()
    plot_results(load_results(args.results_dir), args.out, args.subtitle)


if __name__ == "__main__":
    main()
