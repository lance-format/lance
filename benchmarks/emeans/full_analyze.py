# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Summarize the centroid-training and full IVF_PQ build benchmarks.

Reads `train.jsonl` (from `emeans_bench`, run by the training script) and
`ivf.jsonl` (from `ivf_build_bench`) in a results folder and writes
`summary.md` and figures next to them. Every point is the median over seeds
or repetitions, with the range as error bars.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["main-lloyd", "head-lloyd", "head-emeans"]
NAMES = {
    "main-lloyd": "main: hierarchical Lloyd",
    "head-lloyd": "branch: hierarchical Lloyd",
    "head-emeans": "branch: hierarchical e-means",
}
COLORS = {"main-lloyd": "#2a78d6", "head-lloyd": "#eb6834", "head-emeans": "#1baf7a"}
IVF_VARIANT = {"main": "main-lloyd", "lloyd": "head-lloyd", "emeans": "head-emeans"}
STAGES = [
    ("ivf_sample", "IVF sample read"),
    ("ivf_kmeans", "IVF centroid training"),
    ("pq_prep", "PQ sample read + residuals"),
    ("pq_train", "PQ codebook training"),
    ("shuffle", "shuffle: assign + encode"),
    ("merge", "merge partitions"),
    ("other", "other"),
]
STAGE_COLORS = [
    "#6250d6",
    "#eb6834",
    "#e87ba4",
    "#eda100",
    "#2a78d6",
    "#1baf7a",
    "#b5b4ad",
]
INK = "#0b0b0b"
MUTED = "#52514e"

plt.rcParams.update(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#b5b4ad",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 5,
    }
)


def stats(values):
    values = np.asarray(values, dtype=float)
    return np.median(values), values.min(), values.max()


def load(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- training


def train_variant(row):
    return f"{row['label'].split('-')[1]}-{row['node_algorithm']}"


def train_index(rows):
    """(variant, dim, k, sample_rate, threads) -> list of rows."""
    index = defaultdict(list)
    for row in rows:
        key = (
            train_variant(row),
            row["dim"],
            row["k"],
            row["sample_rate"],
            row["threads"],
        )
        index[key].append(row)
    return index


def errorbar(ax, xs, groups, variant, metric="train_secs"):
    med, lo, hi = zip(*(stats([r[metric] for r in g]) for g in groups))
    med, lo, hi = map(np.array, (med, lo, hi))
    ax.errorbar(
        xs,
        med,
        yerr=[med - lo, hi - med],
        fmt="o-",
        color=COLORS[variant],
        label=NAMES[variant],
        capsize=2,
    )
    return med


def plot_train_grid(index, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, dim in zip(axes[:2], (128, 1024)):
        for variant in VARIANTS:
            ks = sorted(
                k
                for (v, d, k, sr, t) in index
                if (v, d, sr, t) == (variant, dim, 256, 96)
            )
            if not ks:
                continue
            errorbar(ax, ks, [index[(variant, dim, k, 256, 96)] for k in ks], variant)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("centroids k (training rows = 256·k)")
        ax.set_ylabel("training time (s)")
        ax.set_title(f"d = {dim}, 96 vCPUs", color=INK)
    # Speedup over main at every grid point.
    ax = axes[2]
    labels, speedups = [], {v: [] for v in VARIANTS[1:]}
    for dim in (128, 1024):
        for k in sorted(
            {k for (v, d, k, sr, t) in index if d == dim and sr == 256 and t == 96}
        ):
            base = index.get(("main-lloyd", dim, k, 256, 96))
            if not base:
                continue
            labels.append(f"d{dim}\nk{k // 1024}K")
            for variant in VARIANTS[1:]:
                group = index.get((variant, dim, k, 256, 96))
                speedups[variant].append(
                    stats([r["train_secs"] for r in base])[0]
                    / stats([r["train_secs"] for r in group])[0]
                    if group
                    else np.nan
                )
    x = np.arange(len(labels))
    for i, variant in enumerate(VARIANTS[1:]):
        bars = ax.bar(
            x + (i - 0.5) * 0.4,
            speedups[variant],
            0.38,
            color=COLORS[variant],
            label=NAMES[variant],
        )
        for bar, value in zip(bars, speedups[variant]):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.1f}×",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=INK,
                )
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_yscale("log")
    ax.axhline(1, color=MUTED, linestyle=":")
    ax.set_ylabel("speedup over main (×)")
    ax.set_title("speedup over main's Lloyd, 96 vCPUs", color=INK)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "train_grid.png", dpi=150)
    plt.close(fig)


def plot_train_sweeps(index, out):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    # Dimension at k = 65,536.
    ax = axes[0]
    for variant in VARIANTS:
        dims = sorted(
            {
                d
                for (v, d, k, sr, t) in index
                if (v, k, sr, t) == (variant, 65536, 256, 96)
            }
        )
        if dims:
            errorbar(
                ax, dims, [index[(variant, d, 65536, 256, 96)] for d in dims], variant
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("dimension")
    ax.set_ylabel("training time (s)")
    ax.set_title("k = 65,536, 96 vCPUs", color=INK)
    # vCPUs for two configurations.
    for ax, (dim, k) in zip(axes[1:3], ((128, 262144), (1024, 65536))):
        for variant in VARIANTS:
            threads = sorted(
                {
                    t
                    for (v, d, kk, sr, t) in index
                    if (v, d, kk, sr) == (variant, dim, k, 256)
                }
            )
            if threads:
                errorbar(
                    ax,
                    threads,
                    [index[(variant, dim, k, 256, t)] for t in threads],
                    variant,
                )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([8, 16, 32, 48, 64, 96], ["8", "16", "32", "48", "64", "96"])
        ax.set_xlabel("vCPUs (half as many cores, with hyperthreads)")
        ax.set_ylabel("training time (s)")
        ax.set_title(f"d = {dim}, k = {k:,}", color=INK)
    # Sample rate at k = 65,536, both dimensions.
    ax = axes[3]
    for variant in VARIANTS:
        for dim, style in ((128, "o-"), (1024, "s--")):
            rates = sorted(
                {
                    sr
                    for (v, d, k, sr, t) in index
                    if (v, d, k, t) == (variant, dim, 65536, 96)
                }
            )
            if not rates:
                continue
            groups = [index[(variant, dim, 65536, sr, 96)] for sr in rates]
            med = [stats([r["train_secs"] for r in g])[0] for g in groups]
            ax.plot(
                rates,
                med,
                style,
                color=COLORS[variant],
                label=f"{NAMES[variant]}, d={dim}" if variant else None,
            )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("training rows per centroid (sample rate)")
    ax.set_ylabel("training time (s)")
    ax.set_title(
        "k = 65,536, 96 vCPUs (solid d=128, dashed d=1024)", color=INK, fontsize=8
    )
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "train_sweeps.png", dpi=150)
    plt.close(fig)


def train_tables(index):
    lines = []
    configs = sorted({(d, k, sr, t) for (v, d, k, sr, t) in index})
    lines.append(
        "| d | k | rows/centroid | vCPUs | main Lloyd (s) | branch Lloyd (s) | branch e-means (s) | Lloyd speedup | e-means vs main | e-means vs branch Lloyd | val MSE e-means / main |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for d, k, sr, t in configs:
        groups = {v: index.get((v, d, k, sr, t)) for v in VARIANTS}
        if not all(groups.values()):
            continue
        secs = {v: stats([r["train_secs"] for r in g])[0] for v, g in groups.items()}
        mse = {v: stats([r["val_mse"] for r in g])[0] for v, g in groups.items()}
        n = min(len(g) for g in groups.values())
        lines.append(
            f"| {d} | {k:,} | {sr} | {t} | {secs['main-lloyd']:.1f} | {secs['head-lloyd']:.1f} | "
            f"{secs['head-emeans']:.1f} | {secs['main-lloyd'] / secs['head-lloyd']:.1f}× | "
            f"{secs['main-lloyd'] / secs['head-emeans']:.1f}× | {secs['head-lloyd'] / secs['head-emeans']:.1f}× | "
            f"{mse['head-emeans'] / mse['main-lloyd']:.4f} (n={n}) |"
        )
    return lines


# ---------------------------------------------------------------- IVF builds


def message_time(row, pattern, nth=0):
    hits = [t for t, _, message in row["log"] if re.search(pattern, message)]
    return hits[nth] if len(hits) > nth else None


def stage_times(row):
    events = {(stage, event): t for stage, event, t in row["stages"]}
    ivf_msg = message_time(row, r"^Start to train IVF model")
    pq_msg = message_time(row, r"^Start to train quantizer")
    # Without the marker line, the whole stage counts as training.
    ivf_msg = events[("train_ivf", "start")] if ivf_msg is None else ivf_msg
    pq_msg = events[("train_quantizer", "start")] if pq_msg is None else pq_msg
    out = {
        "ivf_sample": ivf_msg - events[("train_ivf", "start")],
        "ivf_kmeans": events[("train_ivf", "complete")] - ivf_msg,
        "pq_prep": pq_msg - events[("train_quantizer", "start")],
        "pq_train": events[("train_quantizer", "complete")] - pq_msg,
        "shuffle": events[("shuffle", "complete")] - events[("shuffle", "start")],
        "merge": events[("merge_partitions", "complete")]
        - events[("merge_partitions", "start")],
    }
    out["other"] = row["build_secs"] - sum(out.values())
    return out


def ivf_config(row):
    return (row["rows"], row["dim"], row["k"], row["num_sub_vectors"], row["threads"])


def ivf_config_name(config):
    rows, dim, k, m, threads = config
    return f"{rows / 2**20:.0f}M × {dim}d, IVF{k}, PQ{m}, {threads} vCPUs"


def plot_ivf(rows, out):
    by_config = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_config[ivf_config(row)][IVF_VARIANT[row["variant"]]].append(row)
    configs = [c for c in by_config if all(by_config[c][v] for v in VARIANTS)]
    configs.sort(key=lambda c: (c[4] != 96, c[1], c[0], c[2], -c[3]))
    if not configs:
        return [], by_config
    height = 1.55 * len(configs) + 1.2
    fig, axes = plt.subplots(len(configs), 1, figsize=(11, height), squeeze=False)
    for ax, config in zip(axes[:, 0], configs):
        for i, variant in enumerate(VARIANTS):
            group = by_config[config][variant]
            parts = {
                name: np.median([stage_times(r)[name] for r in group])
                for name, _ in STAGES
            }
            total = np.median([r["build_secs"] for r in group])
            left = 0.0
            for (name, label), color in zip(STAGES, STAGE_COLORS):
                width = max(parts[name], 0.0)
                ax.barh(
                    i,
                    width,
                    left=left,
                    color=color,
                    height=0.72,
                    edgecolor="#fcfcfb",
                    linewidth=1,
                    label=label,
                )
                left += width
            ax.text(left, i, f"  {total:.0f}s", va="center", fontsize=8, color=INK)
        ax.set_yticks(
            range(3), ["main", "branch, Lloyd IVF", "branch, e-means IVF"], fontsize=8
        )
        ax.invert_yaxis()
        ax.set_title(ivf_config_name(config), loc="left", fontsize=9, color=INK)
        ax.grid(axis="y", visible=False)
    axes[-1, 0].set_xlabel("build time (s), median over repetitions")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[: len(STAGES)],
        names[: len(STAGES)],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.tight_layout(rect=(0, 0.75 / height, 0.97, 1))
    fig.savefig(out / "ivf_stages.png", dpi=150)
    plt.close(fig)

    # Total speedup and the IVF-training share of each build.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    labels = [
        f"{rows / 2**20:.0f}M×{dim}d\nIVF{k // 1024}K PQ{m}\n{threads} vCPUs"
        for rows, dim, k, m, threads in configs
    ]
    x = np.arange(len(configs))
    for i, variant in enumerate(VARIANTS[1:]):
        speedup = [
            np.median([r["build_secs"] for r in by_config[c]["main-lloyd"]])
            / np.median([r["build_secs"] for r in by_config[c][variant]])
            for c in configs
        ]
        bars = axes[0].bar(
            x + (i - 0.5) * 0.4,
            speedup,
            0.38,
            color=COLORS[variant],
            label=NAMES[variant],
        )
        for bar, value in zip(bars, speedup):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.2f}×",
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK,
            )
    axes[0].axhline(1, color=MUTED, linestyle=":")
    axes[0].set_xticks(x, labels, fontsize=7)
    axes[0].set_ylim(0, 1.7)
    axes[0].set_ylabel("full build speedup over main (×)")
    axes[0].set_title("end-to-end IVF_PQ build speedup", color=INK)
    axes[0].legend(fontsize=7, loc="upper right")
    for i, variant in enumerate(VARIANTS):
        share = [
            100
            * np.median(
                [
                    stage_times(r)["ivf_kmeans"] / r["build_secs"]
                    for r in by_config[c][variant]
                ]
            )
            for c in configs
        ]
        bars = axes[1].bar(
            x + (i - 1) * 0.28, share, 0.26, color=COLORS[variant], label=NAMES[variant]
        )
        for bar, value in zip(bars, share):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.0f}%",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=INK,
            )
    axes[1].set_xticks(x, labels, fontsize=7)
    axes[1].set_ylim(0, 40)
    axes[1].set_ylabel("IVF centroid training, % of build")
    axes[1].set_title("share of the build spent training IVF centroids", color=INK)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ivf_speedup.png", dpi=150)
    plt.close(fig)
    return configs, by_config


def ivf_tables(configs, by_config):
    lines = [
        "| configuration | variant | build (s) | speedup | IVF sample | IVF train | PQ prep | PQ train | shuffle | merge | peak RSS (GB) | recall@10 (nprobes, refine) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for config in configs:
        base = np.median([r["build_secs"] for r in by_config[config]["main-lloyd"]])
        for variant in VARIANTS:
            group = by_config[config][variant]
            total = np.median([r["build_secs"] for r in group])
            parts = {
                name: np.median([stage_times(r)[name] for r in group])
                for name, _ in STAGES
            }
            recall = defaultdict(list)
            for r in group:
                for entry in r["recall"]:
                    recall[(entry["nprobes"], entry["refine"])].append(entry["recall"])
            recall_text = ", ".join(
                f"{np.mean(v):.3f} ({p},{f})" for (p, f), v in sorted(recall.items())
            )
            rss = np.median([r["peak_rss_bytes"] for r in group]) / 1e9
            lines.append(
                f"| {ivf_config_name(config)} | {NAMES[variant]} (n={len(group)}) | {total:.1f} | {base / total:.2f}× | "
                + " | ".join(
                    f"{parts[name]:.1f}"
                    for name in (
                        "ivf_sample",
                        "ivf_kmeans",
                        "pq_prep",
                        "pq_train",
                        "shuffle",
                        "merge",
                    )
                )
                + f" | {rss:.0f} | {recall_text} |"
            )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    out = args.results
    summary = ["# Centroid training and full IVF_PQ builds", ""]

    train = load(out / "train.jsonl")
    if train:
        index = train_index(train)
        plot_train_grid(index, out)
        plot_train_sweeps(index, out)
        summary += [
            "## Centroid training",
            "",
            *train_tables(index),
            "",
            "![grid](train_grid.png)",
            "",
            "![sweeps](train_sweeps.png)",
            "",
        ]

    ivf = load(out / "ivf.jsonl")
    if ivf:
        configs, by_config = plot_ivf(ivf, out)
        summary += [
            "## Full IVF_PQ builds",
            "",
            *ivf_tables(configs, by_config),
            "",
            "![stages](ivf_stages.png)",
            "",
            "![speedup](ivf_speedup.png)",
            "",
        ]

    (out / "summary.md").write_text("\n".join(summary))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
