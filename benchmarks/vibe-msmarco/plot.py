#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Render a ClickBench-style IVF_RQ warm/cold latency chart."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


VERSION_ORDER = ("v11.0.0", "v12.0.0")


def _sort_key(label: str) -> tuple[int, str]:
    if label in VERSION_ORDER:
        return (VERSION_ORDER.index(label), label)
    return (len(VERSION_ORDER), label)


def _result_paths(results_dir: Path) -> list[Path]:
    manifest = results_dir / "_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        paths = [Path(item) for item in payload.get("results", [])]
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


def plot_results(rows: list[dict], out: Path, subtitle: str) -> None:
    if not rows:
        raise SystemExit("no result JSON files found")

    indexes = sorted({row["index"] for row in rows})
    fig, axes = plt.subplots(1, len(indexes), figsize=(12.5, 5.2), sharey=True)
    if len(indexes) == 1:
        axes = [axes]

    colors = {"warm": "#2ca02c", "cold": "#ff7f0e"}
    markers = {"warm": "o", "cold": "s"}

    for ax, index_name in zip(axes, indexes):
        series = [row for row in rows if row["index"] == index_name]
        labels = [
            row["label"].replace(" (main)", "\n(main)") for row in series
        ]
        xs = list(range(len(labels)))
        for mode in ("cold", "warm"):
            means = [row[mode]["summary"]["mean_ms"] for row in series]
            medians = [row[mode]["summary"]["median_ms"] for row in series]
            ax.plot(
                xs,
                means,
                color=colors[mode],
                marker=markers[mode],
                linewidth=2,
                label=f"{mode} mean",
            )
            ax.plot(
                xs,
                medians,
                color=colors[mode],
                marker=markers[mode],
                linewidth=1.2,
                linestyle="--",
                alpha=0.85,
                label=f"{mode} median",
            )
        for x in xs:
            ax.axvline(x, color="#bbbbbb", linestyle=":", linewidth=0.8)
        ax.set_xticks(xs, labels)
        ax.set_title(index_name)
        ax.set_xlabel("Lance / pylance version")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    axes[0].set_ylabel("Query latency (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.suptitle("Lance IVF_RQ search latency over recent versions", fontsize=13)
    fig.text(
        0.5,
        0.01,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
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
            "warm = prewarm_index"
        ),
    )
    args = parser.parse_args()
    plot_results(load_results(args.results_dir), args.out, args.subtitle)


if __name__ == "__main__":
    main()
