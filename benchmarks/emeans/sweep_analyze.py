# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Summarize and plot `sweep.sh`: hierarchical Lloyd vs hierarchical e-means
across vector dimensions, e-means batch sizes and epochs, thread counts and
scale.

Writes `summary.md` and the figures `dims.png`, `dims_breakdown.png`,
`root.png`, `batch.png`, `epochs.png`, `threads.png` and, with the large
runs, `large.png` into the sweep folder.

    uv run benchmarks/emeans/sweep_analyze.py --sweep sweep
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ALGORITHMS = ["lloyd", "emeans"]
NAMES = {"lloyd": "Hierarchical Lloyd (default)", "emeans": "Hierarchical e-means"}
COLORS = {"lloyd": "#2f5d8a", "emeans": "#e8913a"}
DIM_COLORS = {
    32: "#6b4c9a",
    64: "#2f8a6b",
    128: "#7a9a2f",
    256: "#c2752f",
    512: "#a33d3d",
    1024: "#2f5d8a",
}
STEPS = [
    "node training: assign",
    "node training: update",
    "membership: assign all node rows",
    "leaf means",
    "sampling and gather",
    "other",
]
WORK = ["assign", "update", "gather", "other"]


def load(sweep):
    with open(sweep / "sweep.jsonl") as f:
        runs = [json.loads(line) for line in f if line.strip()]
    for run in runs:
        run["build"] = "profile" if run["profiled"] else "timing"
    return runs


def key(run, **override):
    fields = {
        "alg": run["node_algorithm"],
        "dim": run["dim"],
        "rows": run["rows"],
        "k": run["k"],
        "scale": run["batch_scale"],
        "epochs": run["epochs"],
        "threads": run["threads"],
        "seed": run["seed"],
        "build": run["build"],
    }
    fields.update(override)
    return tuple(sorted(fields.items()))


def busy(run):
    return sum(total["secs"] for total in run["work"].values())


def step_split(run):
    """Seconds per step; exact for one-thread runs."""
    alg = run["node_algorithm"]

    def phase(name):
        return run["phases"][name]["secs"]

    assign = phase(f"{alg}_assign")
    update = phase(f"{alg}_update") + (
        phase("emeans_finalize") if alg == "emeans" else 0.0
    )
    steps = {
        "node training: assign": assign,
        "node training: update": update,
        "membership: assign all node rows": phase("membership"),
        "leaf means": phase("leaves"),
        "sampling and gather": phase("node_sample"),
    }
    steps["other"] = run["train_secs"] - sum(steps.values())
    return steps


def capacity_split(run):
    """Thread-seconds of a run split into busy work by kind and idle."""
    split = {f"busy: {name}": run["work"][name]["secs"] for name in WORK}
    split["idle"] = run["threads"] * run["train_secs"] - busy(run)
    return split


def root_secs(run):
    """Wall time of the root node: it samples, trains and assigns with the pool
    to itself before any other node starts."""
    return sum(
        run["phases"][p]["first_secs"]
        for p in ("node_sample", "node_train", "membership")
    )


class Results:
    def __init__(self, sweep, runs):
        self.sweep = sweep
        self.runs = runs
        self.by_key = {key(r): r for r in runs}
        self.dists = {}
        self.max_threads = max(r["threads"] for r in runs)
        self.rng = np.random.default_rng(0)

    def get(self, run, **override):
        return self.by_key.get(key(run, **override))

    def dist(self, run):
        if run["label"] not in self.dists:
            path = self.sweep / f"{run['label']}.f32"
            self.dists[run["label"]] = np.fromfile(path, dtype="<f4").astype(np.float64)
        return self.dists[run["label"]]

    def select(self, **fields):
        return [
            r for r in self.runs if all(dict(key(r))[f] == v for f, v in fields.items())
        ]

    def clean_secs(self, alg, dim, n, k, threads):
        """Median training time of the unprofiled runs."""
        runs = self.select(
            alg=alg,
            dim=dim,
            rows=n,
            k=k,
            scale=1.0,
            epochs=3,
            threads=threads,
            build="timing",
        )
        return float(np.median([r["train_secs"] for r in runs]))

    def mse_ratio(self, runs, resamples=1000):
        """Mean over seeds of MSE(run) / MSE(Lloyd, same data and seed), with a
        95% bootstrap interval over queries drawn the same for both runs."""
        pairs = []
        for run in runs:
            baseline = self.get(run, alg="lloyd", scale=1.0, epochs=3)
            if baseline is not None:
                pairs.append((self.dist(run), self.dist(baseline)))
        if not pairs:
            return None
        num_queries = len(pairs[0][0])
        draws = self.rng.integers(0, num_queries, size=(resamples, num_queries))
        point = np.mean([a.mean() / b.mean() for a, b in pairs])
        boot = np.mean(
            [a[draws].mean(axis=1) / b[draws].mean(axis=1) for a, b in pairs], axis=0
        )
        low, high = np.percentile(boot, [2.5, 97.5])
        return point, low, high


def median_range(values):
    values = np.array(values)
    return np.median(values), values.min(), values.max()


def plot_dims(res, plt, n, k, dims):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    P = res.max_threads
    for alg in ALGORITHMS:
        med, low, high, eff, idle, inflation, mse = [], [], [], [], [], [], []
        for dim in dims:
            timing = res.select(
                alg=alg,
                dim=dim,
                rows=n,
                k=k,
                scale=1.0,
                epochs=3,
                threads=P,
                build="timing",
            )
            m, lo, hi = median_range([r["train_secs"] for r in timing])
            med.append(m), low.append(lo), high.append(hi)
            full = res.select(
                alg=alg,
                dim=dim,
                rows=n,
                k=k,
                scale=1.0,
                epochs=3,
                threads=P,
                build="profile",
            )[0]
            single = res.select(
                alg=alg,
                dim=dim,
                rows=n,
                k=k,
                scale=1.0,
                epochs=3,
                threads=1,
                build="profile",
            )[0]
            eff.append(
                res.clean_secs(alg, dim, n, k, 1)
                / (P * res.clean_secs(alg, dim, n, k, P))
            )
            idle.append(capacity_split(full)["idle"] / (P * full["train_secs"]))
            inflation.append(busy(full) / busy(single) - 1)
            mse.append(res.mse_ratio(timing))
        med = np.array(med)
        axes[0, 0].errorbar(
            dims,
            med,
            yerr=[med - low, np.array(high) - med],
            fmt="o-",
            capsize=4,
            color=COLORS[alg],
            label=NAMES[alg],
        )
        axes[1, 0].plot(dims, eff, "o-", color=COLORS[alg], label=NAMES[alg])
        axes[1, 1].plot(dims, idle, "o-", color=COLORS[alg], label=NAMES[alg])
        axes[1, 2].plot(dims, inflation, "o-", color=COLORS[alg], label=NAMES[alg])
        if alg == "emeans":
            ratio = np.array([m[0] for m in mse])
            axes[0, 2].errorbar(
                dims,
                ratio,
                yerr=[
                    ratio - [m[1] for m in mse],
                    np.array([m[2] for m in mse]) - ratio,
                ],
                fmt="o-",
                capsize=4,
                color=COLORS[alg],
            )
    for threads, style, label in [
        (P, "o-", f"{P} threads (median of seeds)"),
        (1, "s--", "1 thread"),
    ]:
        speedups = []
        for dim in dims:

            def pick(alg, dim=dim, threads=threads):
                return res.select(
                    alg=alg,
                    dim=dim,
                    rows=n,
                    k=k,
                    scale=1.0,
                    epochs=3,
                    threads=threads,
                    build="timing",
                )

            lloyd = {r["seed"]: r["train_secs"] for r in pick("lloyd")}
            speedups.append(
                np.median(
                    [
                        lloyd[r["seed"]] / r["train_secs"]
                        for r in pick("emeans")
                        if r["seed"] in lloyd
                    ]
                )
            )
        axes[0, 1].plot(dims, speedups, style, color=COLORS["emeans"], label=label)
    ceiling = [
        res.select(
            alg="lloyd",
            dim=d,
            rows=n,
            k=k,
            scale=1.0,
            epochs=3,
            threads=P,
            build="profile",
        )[0]["ceiling_speedup"]
        / P
        for d in dims
    ]
    axes[1, 0].plot(dims, ceiling, "--", color="gray", label="hardware ceiling")
    axes[0, 1].axhline(1, color="gray", linestyle=":")
    axes[0, 2].axhline(1, color="gray", linestyle=":")
    titles = [
        ("training time (s)", True),
        ("speedup of e-means over Lloyd", False),
        ("validation MSE: e-means / Lloyd (95% CI)", False),
        (f"parallel efficiency at {P} threads: T1 / (P x TP)", False),
        (f"idle share of thread-seconds at {P} threads", False),
        (f"busy-work growth at {P} threads vs 1 (contention)", False),
    ]
    for axis, (title, logy) in zip(axes.flat, titles):
        axis.set_xscale("log", base=2)
        axis.set_xticks(dims, [str(d) for d in dims])
        axis.set_xlabel("vector dimension (PCA of DINO10B)")
        axis.set_title(title, fontsize=10)
        if logy:
            axis.set_yscale("log")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.suptitle(f"{n:,} vectors | k={k:,} | {P} threads | dimension sweep")
    fig.tight_layout()
    fig.savefig(res.sweep / "dims.png", dpi=140)


def plot_dims_breakdown(res, plt, n, k, dims):
    P = res.max_threads
    fig, axes = plt.subplots(2, 2, figsize=(19, 8.5))
    cmap = plt.get_cmap("tab10")
    for row, alg in enumerate(ALGORITHMS):
        for col, threads in enumerate([1, P]):
            axis = axes[row, col]
            parts = STEPS if threads == 1 else [f"busy: {w}" for w in WORK] + ["idle"]
            bottoms = np.zeros(len(dims))
            totals = []
            for dim_index, dim in enumerate(dims):
                run = res.select(
                    alg=alg,
                    dim=dim,
                    rows=n,
                    k=k,
                    scale=1.0,
                    epochs=3,
                    threads=threads,
                    build="profile",
                )[0]
                split = step_split(run) if threads == 1 else capacity_split(run)
                total = run["train_secs"] * threads
                totals.append(run["train_secs"])
                for i, part in enumerate(parts):
                    share = split[part] / total
                    axis.bar(
                        dim_index,
                        share,
                        bottom=bottoms[dim_index],
                        color=cmap(i),
                        label=part if dim_index == 0 else None,
                    )
                    if share > 0.05:
                        axis.text(
                            dim_index,
                            bottoms[dim_index] + share / 2,
                            f"{share:.0%}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white",
                        )
                    bottoms[dim_index] += share
            for dim_index, secs in enumerate(totals):
                axis.text(dim_index, 1.01, f"{secs:.3g}s", ha="center", fontsize=8)
            axis.set_xticks(range(len(dims)), [str(d) for d in dims])
            axis.set_xlabel("vector dimension")
            axis.set_ylim(0, 1.08)
            title = (
                "one thread: share of training time"
                if threads == 1
                else f"{P} threads: share of thread-seconds"
            )
            axis.set_title(f"{NAMES[alg]}, {title}", fontsize=10)
            axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle(
        f"{n:,} vectors | k={k:,} | seed 1, profiled build; wall seconds on top"
    )
    fig.tight_layout()
    fig.savefig(res.sweep / "dims_breakdown.png", dpi=140)


def plot_root(res, plt, n, k, dims):
    P = res.max_threads
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for alg in ALGORITHMS:
        share, train_eff, member_eff = [], [], []
        for dim in dims:
            full, single = (
                res.select(
                    alg=alg,
                    dim=dim,
                    rows=n,
                    k=k,
                    scale=1.0,
                    epochs=3,
                    threads=t,
                    build="profile",
                )[0]
                for t in (P, 1)
            )
            share.append(root_secs(full) / full["train_secs"])

            def first(run, p):
                return run["phases"][p]["first_secs"]

            train_eff.append(
                first(single, "node_train") / (P * first(full, "node_train"))
            )
            member_eff.append(
                first(single, "membership") / (P * first(full, "membership"))
            )
        axes[0].plot(dims, share, "o-", color=COLORS[alg], label=NAMES[alg])
        axes[1].plot(dims, train_eff, "o-", color=COLORS[alg], label=NAMES[alg])
        axes[2].plot(dims, member_eff, "o-", color=COLORS[alg], label=NAMES[alg])
    titles = [
        f"root node's share of wall time at {P} threads",
        "root training: parallel efficiency",
        "root membership: parallel efficiency",
    ]
    for axis, title in zip(axes, titles):
        axis.set_xscale("log", base=2)
        axis.set_xticks(dims, [str(d) for d in dims])
        axis.set_xlabel("vector dimension")
        axis.set_title(title, fontsize=10)
        axis.set_ylim(0, None)
        axis.legend(fontsize=8)
    fig.suptitle(
        f"{n:,} vectors | k={k:,} | the root trains alone, so its efficiency is the "
        "within-node parallelism"
    )
    fig.tight_layout()
    fig.savefig(res.sweep / "root.png", dpi=140)


def plot_sweep_param(res, plt, n, k, param, values, base_value, path, title):
    P = res.max_threads
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    dims = sorted(
        {r["dim"] for r in res.runs if r[param] != base_value and r["rows"] == n}
    )
    for dim in dims:
        times, mses, idles, xs = [], [], [], []
        lloyd = np.median(
            [
                r["train_secs"]
                for r in res.select(
                    alg="lloyd", dim=dim, rows=n, k=k, threads=P, build="timing"
                )
            ]
        )
        for value in values:
            sel = (
                {"scale": value, "epochs": 3}
                if param == "batch_scale"
                else {"scale": 1.0, "epochs": value}
            )
            timing = res.select(
                alg="emeans", dim=dim, rows=n, k=k, threads=P, build="timing", **sel
            )
            if not timing:
                continue
            xs.append(value)
            times.append(np.median([r["train_secs"] for r in timing]) / lloyd)
            mses.append(res.mse_ratio(timing))
            profile = res.select(
                alg="emeans", dim=dim, rows=n, k=k, threads=P, build="profile", **sel
            )
            idles.append(
                capacity_split(profile[0])["idle"] / (P * profile[0]["train_secs"])
                if profile
                else np.nan
            )
        color = DIM_COLORS.get(dim)
        axes[0].plot(xs, times, "o-", color=color, label=f"d={dim}")
        ratio = np.array([m[0] for m in mses])
        axes[1].errorbar(
            xs,
            ratio,
            yerr=[ratio - [m[1] for m in mses], np.array([m[2] for m in mses]) - ratio],
            fmt="o-",
            capsize=3,
            color=color,
            label=f"d={dim}",
        )
        axes[2].plot(xs, idles, "o-", color=color, label=f"d={dim}")
    for axis, ylabel in zip(
        axes,
        [
            "training time / Lloyd's",
            "validation MSE / Lloyd's",
            f"idle share of thread-seconds ({P} threads)",
        ],
    ):
        if param == "batch_scale":
            axis.set_xscale("log", base=2)
            axis.set_xlabel("batch size relative to the paper's batch law")
        else:
            axis.set_xlabel("epochs")
        axis.axvline(base_value, color="gray", linestyle=":")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    axes[0].axhline(1, color="gray", linestyle="--")
    axes[1].axhline(1, color="gray", linestyle="--")
    fig.suptitle(
        f"{n:,} vectors | k={k:,} | {P} threads | {title}; dotted: default; "
        "dashed: Lloyd"
    )
    fig.tight_layout()
    fig.savefig(res.sweep / path, dpi=140)


def plot_threads(res, plt, n, k):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    dims = sorted(
        {
            r["dim"]
            for r in res.runs
            if r["threads"] not in (1, res.max_threads) and r["rows"] == n
        }
    )
    for axis, dim in zip(axes, dims):
        for alg in ALGORITHMS:
            runs = sorted(
                res.select(
                    alg=alg,
                    dim=dim,
                    rows=n,
                    k=k,
                    scale=1.0,
                    epochs=3,
                    build="profile",
                    seed=1,
                ),
                key=lambda r: r["threads"],
            )
            t1 = runs[0]["train_secs"]
            axis.plot(
                [r["threads"] for r in runs],
                [t1 / r["train_secs"] for r in runs],
                "o-",
                color=COLORS[alg],
                label=NAMES[alg],
            )
        ceiling = sorted(
            {
                r["threads"]: r["ceiling_speedup"]
                for r in res.select(dim=dim, rows=n, k=k, build="profile")
            }.items()
        )
        axis.plot(*zip(*ceiling), "--", color="gray", label="hardware ceiling")
        axis.plot(
            [1, res.max_threads],
            [1, res.max_threads],
            ":",
            color="gray",
            label="perfect",
        )
        axis.set(xlabel="threads", ylabel="speedup over one thread", title=f"d={dim}")
        axis.legend(fontsize=8)
    fig.suptitle(f"{n:,} vectors | k={k:,} | thread scaling, profiled build")
    fig.tight_layout()
    fig.savefig(res.sweep / "threads.png", dpi=140)


def plot_large(res, plt, n, k):
    P = res.max_threads
    dims = sorted({r["dim"] for r in res.select(rows=n)})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.38
    for i, alg in enumerate(ALGORITHMS):
        meds = [
            np.median(
                [
                    r["train_secs"]
                    for r in res.select(
                        alg=alg, dim=d, rows=n, k=k, threads=P, build="timing"
                    )
                ]
            )
            for d in dims
        ]
        bars = axes[0].bar(
            np.arange(len(dims)) + (i - 0.5) * width,
            meds,
            width,
            color=COLORS[alg],
            label=NAMES[alg],
        )
        axes[0].bar_label(bars, labels=[f"{m:.0f}s" for m in meds], fontsize=8)
    mses = [
        res.mse_ratio(
            res.select(alg="emeans", dim=d, rows=n, k=k, threads=P, build="timing")
        )
        for d in dims
    ]
    ratio = np.array([m[0] for m in mses])
    axes[1].errorbar(
        range(len(dims)),
        ratio,
        yerr=[ratio - [m[1] for m in mses], np.array([m[2] for m in mses]) - ratio],
        fmt="o",
        capsize=6,
        color=COLORS["emeans"],
    )
    axes[1].axhline(1, color="gray", linestyle="--")
    for axis in axes:
        axis.set_xticks(range(len(dims)), [f"d={d}" for d in dims])
    axes[0].set_ylabel("training time (s, median of seeds)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("validation MSE: e-means / Lloyd (95% CI)")
    fig.suptitle(f"{n:,} vectors | k={k:,} | {P} threads | larger scale")
    fig.tight_layout()
    fig.savefig(res.sweep / "large.png", dpi=140)


def summary(res, n, k, dims, large):
    P = res.max_threads
    lines = [
        "# Hierarchical Lloyd vs hierarchical e-means: sweep\n",
        (
            f"Base scale: {n:,} DINO10B vectors, k={k:,}, {P} threads. Dimensions "
            "below 1,024 project onto the leading principal components.\n"
        ),
    ]
    lines.append("## By dimension\n")
    lines.append(
        "| d | Lloyd s | e-means s | speedup | 1-thread speedup | MSE e-means / Lloyd | "
        "efficiency Lloyd | efficiency e-means | idle Lloyd | idle e-means | "
        "root share Lloyd | root share e-means |"
    )
    lines.append("|" + "---|" * 12)
    for dim in dims:
        row = [str(dim)]
        secs = {}
        for alg in ALGORITHMS:
            secs[alg] = np.median(
                [
                    r["train_secs"]
                    for r in res.select(
                        alg=alg,
                        dim=dim,
                        rows=n,
                        k=k,
                        scale=1.0,
                        epochs=3,
                        threads=P,
                        build="timing",
                    )
                ]
            )
            row.append(f"{secs[alg]:.1f}")
        row.append(f"{secs['lloyd'] / secs['emeans']:.2f}x")
        prof = {
            (alg, t): res.select(
                alg=alg,
                dim=dim,
                rows=n,
                k=k,
                scale=1.0,
                epochs=3,
                threads=t,
                build="profile",
            )[0]
            for alg in ALGORITHMS
            for t in (1, P)
        }
        clean = {
            (alg, t): res.clean_secs(alg, dim, n, k, t)
            for alg in ALGORITHMS
            for t in (1, P)
        }
        row.append(f"{clean[('lloyd', 1)] / clean[('emeans', 1)]:.2f}x")
        mse = res.mse_ratio(
            res.select(
                alg="emeans",
                dim=dim,
                rows=n,
                k=k,
                scale=1.0,
                epochs=3,
                threads=P,
                build="timing",
            )
        )
        row.append(f"{mse[0]:.4f} [{mse[1]:.4f}, {mse[2]:.4f}]")
        for alg in ALGORITHMS:
            row.append(f"{clean[(alg, 1)] / (P * clean[(alg, P)]):.0%}")
        for alg in ALGORITHMS:
            full = prof[(alg, P)]
            row.append(f"{capacity_split(full)['idle'] / (P * full['train_secs']):.0%}")
        for alg in ALGORITHMS:
            full = prof[(alg, P)]
            row.append(f"{root_secs(full) / full['train_secs']:.0%}")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## One-thread time split (seconds)\n")
    lines.append(
        "| d | algorithm | " + " | ".join(STEPS) + " | assign share | update share |"
    )
    lines.append("|" + "---|" * (len(STEPS) + 4))
    for dim in dims:
        for alg in ALGORITHMS:
            run = res.select(
                alg=alg,
                dim=dim,
                rows=n,
                k=k,
                scale=1.0,
                epochs=3,
                threads=1,
                build="profile",
            )[0]
            split = step_split(run)
            assign = (
                split["node training: assign"]
                + split["membership: assign all node rows"]
            )
            update = split["node training: update"] + split["leaf means"]
            lines.append(
                f"| {dim} | {alg} | "
                + " | ".join(f"{split[s]:.1f}" for s in STEPS)
                + f" | {assign / (assign + update):.1%} | "
                f"{update / (assign + update):.1%} |"
            )

    for param, values, title in [
        (
            "batch_scale",
            [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0],
            "e-means batch size (x the batch law)",
        ),
        ("epochs", [1, 2, 3, 5, 8], "e-means epochs"),
    ]:
        lines.append(f"\n## {title}\n")
        lines.append("| d | value | time / Lloyd | MSE / Lloyd | idle share |")
        lines.append("|---|---|---|---|---|")
        for dim in sorted({r["dim"] for r in res.runs if r[param] not in (1.0, 3)}):
            lloyd = np.median(
                [
                    r["train_secs"]
                    for r in res.select(
                        alg="lloyd", dim=dim, rows=n, k=k, threads=P, build="timing"
                    )
                ]
            )
            for value in values:
                sel = (
                    {"scale": value, "epochs": 3}
                    if param == "batch_scale"
                    else {"scale": 1.0, "epochs": value}
                )
                timing = res.select(
                    alg="emeans", dim=dim, rows=n, k=k, threads=P, build="timing", **sel
                )
                if not timing:
                    continue
                mse = res.mse_ratio(timing)
                profile = res.select(
                    alg="emeans",
                    dim=dim,
                    rows=n,
                    k=k,
                    threads=P,
                    build="profile",
                    **sel,
                )
                idle = (
                    f"{capacity_split(profile[0])['idle'] / (P * profile[0]['train_secs']):.0%}"
                    if profile
                    else ""
                )
                lines.append(
                    f"| {dim} | {value} | "
                    f"{np.median([r['train_secs'] for r in timing]) / lloyd:.2f} | "
                    f"{mse[0]:.4f} [{mse[1]:.4f}, {mse[2]:.4f}] | {idle} |"
                )

    if large:
        ln, lk = large
        lines.append(f"\n## Larger scale: {ln:,} vectors, k={lk:,}\n")
        lines.append(
            "| d | Lloyd s | e-means s | speedup | MSE e-means / Lloyd | idle Lloyd | "
            "idle e-means |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for dim in sorted({r["dim"] for r in res.select(rows=ln)}):
            secs = {
                alg: np.median(
                    [
                        r["train_secs"]
                        for r in res.select(
                            alg=alg, dim=dim, rows=ln, k=lk, threads=P, build="timing"
                        )
                    ]
                )
                for alg in ALGORITHMS
            }
            mse = res.mse_ratio(
                res.select(
                    alg="emeans", dim=dim, rows=ln, k=lk, threads=P, build="timing"
                )
            )
            idle = []
            for alg in ALGORITHMS:
                full = res.select(
                    alg=alg, dim=dim, rows=ln, k=lk, threads=P, build="profile"
                )
                idle.append(
                    f"{capacity_split(full[0])['idle'] / (P * full[0]['train_secs']):.0%}"
                    if full
                    else ""
                )
            lines.append(
                f"| {dim} | {secs['lloyd']:.1f} | {secs['emeans']:.1f} | "
                f"{secs['lloyd'] / secs['emeans']:.2f}x | "
                f"{mse[0]:.4f} [{mse[1]:.4f}, {mse[2]:.4f}] | {idle[0]} | {idle[1]} |"
            )
        singles = {
            alg: res.select(alg=alg, rows=ln, k=lk, threads=1, build="profile")
            for alg in ALGORITHMS
        }
        for alg in ALGORITHMS:
            for single in singles[alg]:
                full = res.select(
                    alg=alg,
                    dim=single["dim"],
                    rows=ln,
                    k=lk,
                    threads=P,
                    build="profile",
                )[0]
                lines.append(
                    f"\n{NAMES[alg]}, d={single['dim']}: one thread "
                    f"{single['train_secs']:.0f} s (profiled), {P} threads "
                    f"{full['train_secs']:.1f} s, efficiency "
                    f"{single['train_secs'] / (P * full['train_secs']):.0%}."
                )
    (res.sweep / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", required=True, type=Path)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = Results(args.sweep, load(args.sweep))
    scales = defaultdict(int)
    for run in res.runs:
        scales[(run["rows"], run["k"])] += 1
    (n, k), *rest = sorted(scales, key=lambda s: -scales[s])
    large = max(rest) if rest else None
    dims = sorted(
        {
            r["dim"]
            for r in res.select(rows=n, k=k, alg="lloyd", build="profile", threads=1)
        }
    )
    plot_dims(res, plt, n, k, dims)
    plot_dims_breakdown(res, plt, n, k, dims)
    plot_root(res, plt, n, k, dims)
    plot_sweep_param(
        res,
        plt,
        n,
        k,
        "batch_scale",
        [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0],
        1.0,
        "batch.png",
        "e-means batch size",
    )
    plot_sweep_param(
        res, plt, n, k, "epochs", [1, 2, 3, 5, 8], 3, "epochs.png", "e-means epochs"
    )
    plot_threads(res, plt, n, k)
    if large:
        plot_large(res, plt, *large)
    summary(res, n, k, dims, large)


if __name__ == "__main__":
    main()
