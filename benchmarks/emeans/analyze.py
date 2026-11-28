# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Summarize and plot the hierarchical Lloyd vs hierarchical e-means benchmark.

Reads `timing.jsonl` (speed and quality over several seeds), `profile.jsonl`
(phase and busy timers over thread counts) and the per-query distance dumps
written by `run.sh`, then writes `summary.md` and four figures:

- `headline.png`: training time and validation MSE relative to Lloyd,
- `breakdown.png`: where single-thread time goes, and how the thread capacity
  of the full-thread run splits into busy work and idle time,
- `scaling.png`: wall time, parallel efficiency and idle share by thread count,
- `utilization.png`: busy threads over the course of the full-thread runs.

    uv run benchmarks/emeans/analyze.py --results results
"""

import argparse
import json
from pathlib import Path

import numpy as np

ALGORITHMS = ["lloyd", "emeans"]
NAMES = {"lloyd": "Hierarchical Lloyd (default)", "emeans": "Hierarchical e-means"}
COLORS = {"lloyd": "#2f5d8a", "emeans": "#e8913a"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def phase(run, name):
    return run["phases"][name]["secs"]


def work(run, name):
    return run["work"][name]["secs"]


def busy(run):
    return sum(total["secs"] for total in run["work"].values())


def single_thread_split(run):
    """Seconds per step of a one-thread run, where step timers are exact."""
    alg = run["node_algorithm"]
    train_assign = phase(run, f"{alg}_assign")
    train_update = phase(run, f"{alg}_update")
    if alg == "emeans":
        train_update += phase(run, "emeans_finalize")
    steps = {
        "node training: assign": train_assign,
        "node training: update": train_update,
        "node training: other": phase(run, "node_train") - train_assign - train_update,
        "membership (assign all node rows)": phase(run, "membership"),
        "leaf means": phase(run, "leaves"),
        "sampling and gather": phase(run, "node_sample"),
    }
    steps["untimed"] = run["train_secs"] - sum(steps.values())
    return steps


def parallel_regions(run):
    """Fork-joins the run issued, counted from the step timers."""
    alg = run["node_algorithm"]

    def calls(name):
        return run["phases"][name]["calls"]

    regions = calls("membership") + 2 * calls("leaves")
    if alg == "lloyd":
        regions += calls("lloyd_assign") + 2 * calls("lloyd_update")
    else:
        regions += calls("emeans_assign")
    return regions


def capacity_split(run, single):
    """Thread-seconds of a parallel run: busy work by kind, then idle (handing
    out and collecting tasks, waiting at joins, and starved of work)."""
    capacity = run["threads"] * run["train_secs"]
    split = {f"busy: {name}": total["secs"] for name, total in run["work"].items()}
    split["idle"] = capacity - busy(run)
    return capacity, split, busy(run) - busy(single)


def starved_share(run, at_most):
    """Share of the run's wall time with at most `at_most` threads busy."""
    histogram = np.array(run["busy_histogram"], dtype=float)
    return histogram[: at_most + 1].sum() / histogram.sum()


def paired_ratio(dists, alg, seeds, rng, resamples=2000):
    """Mean over seeds of MSE(alg) / MSE(lloyd) with a bootstrap over queries
    that draws the same queries for both runs of a seed."""
    num_queries = len(dists[("lloyd", seeds[0])])
    draws = rng.integers(0, num_queries, size=(resamples, num_queries))
    per_seed, boot = [], np.zeros(resamples)
    for seed in seeds:
        a, b = dists[(alg, seed)], dists[("lloyd", seed)]
        per_seed.append(a.mean() / b.mean())
        boot += a[draws].mean(axis=1) / b[draws].mean(axis=1)
    boot /= len(seeds)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(per_seed)), float(low), float(high), per_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timing = load_jsonl(args.results / "timing.jsonl")
    profile = load_jsonl(args.results / "profile.jsonl")
    seeds = sorted({run["seed"] for run in timing})
    by_alg = {
        alg: [r for r in timing if r["node_algorithm"] == alg] for alg in ALGORITHMS
    }
    dists = {
        (run["node_algorithm"], run["seed"]): np.fromfile(
            args.results / f"{run['label']}.f32", dtype="<f4"
        ).astype(np.float64)
        for run in timing
    }
    rng = np.random.default_rng(args.seed)
    first = timing[0]
    setup = (
        f"{first['rows']:,} DINO10B vectors | k={first['k']:,} | d={first['dim']:,} | "
        f"{first['threads']} threads"
    )

    # Speed and quality.
    secs = {alg: np.array([r["train_secs"] for r in by_alg[alg]]) for alg in ALGORITHMS}
    quality = {alg: paired_ratio(dists, alg, seeds, rng) for alg in ALGORITHMS}
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    xs = np.arange(len(ALGORITHMS))
    medians = [np.median(secs[alg]) for alg in ALGORITHMS]
    left.bar(xs, medians, color=[COLORS[a] for a in ALGORITHMS], width=0.6)
    for x, alg in zip(xs, ALGORITHMS):
        left.scatter(
            np.full(len(secs[alg]), x), secs[alg], color="black", s=12, zorder=3
        )
    for x, alg, m in zip(xs, ALGORITHMS, medians):
        left.text(x, secs[alg].max() * 1.03, f"{m:.1f} s", ha="center", va="bottom")
    left.set_xticks(xs, [NAMES[a] for a in ALGORITHMS])
    left.set_ylabel("training time (s, lower is better)")
    left.set_ylim(0, max(medians) * 1.25)
    for x, alg in zip(xs, ALGORITHMS):
        ratio, low, high, per_seed = quality[alg]
        right.errorbar(
            x,
            ratio,
            yerr=[[ratio - low], [high - ratio]],
            fmt="o",
            capsize=8,
            color=COLORS[alg],
            markersize=9,
        )
        right.scatter(np.full(len(per_seed), x + 0.12), per_seed, color="gray", s=10)
    right.axhline(1.0, linestyle="--", color="gray")
    right.set_xticks(xs, [NAMES[a] for a in ALGORITHMS])
    right.set_xlim(-0.5, len(ALGORITHMS) - 0.5)
    right.set_ylabel("validation MSE / Lloyd (lower is better)")
    fig.suptitle(setup)
    fig.text(
        0.5,
        0.01,
        f"Loading and scoring excluded; median of {len(seeds)} seeds (dots: each seed). "
        f"MSE: mean of per-seed ratios over {len(dists[('lloyd', seeds[0])]):,} held-out "
        "queries, 95% paired bootstrap interval; gray: each seed",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.results / "headline.png", dpi=150)

    # Time breakdown.
    runs = {(r["node_algorithm"], r["threads"]): r for r in profile}
    max_threads = max(r["threads"] for r in profile)
    singles = {alg: runs[(alg, 1)] for alg in ALGORITHMS}
    fulls = {alg: runs[(alg, max_threads)] for alg in ALGORITHMS}
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2))
    cmap = plt.get_cmap("tab10")
    for axis, title, splits, totals in [
        (
            left,
            "One thread: where training time goes",
            {alg: single_thread_split(singles[alg]) for alg in ALGORITHMS},
            {alg: singles[alg]["train_secs"] for alg in ALGORITHMS},
        ),
        (
            right,
            f"{max_threads} threads: thread-seconds (threads x wall)",
            {alg: capacity_split(fulls[alg], singles[alg])[1] for alg in ALGORITHMS},
            {alg: max_threads * fulls[alg]["train_secs"] for alg in ALGORITHMS},
        ),
    ]:
        keys = list(next(iter(splits.values())).keys())
        for y, alg in enumerate(ALGORITHMS):
            start = 0.0
            for i, key in enumerate(keys):
                share = splits[alg][key] / totals[alg]
                axis.barh(
                    y, share, left=start, color=cmap(i), label=key if y == 0 else None
                )
                if share > 0.04:
                    axis.text(
                        start + share / 2,
                        y,
                        f"{share:.0%}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white",
                    )
                start += share
            axis.text(1.01, y, f"{totals[alg]:.0f} s", va="center", fontsize=9)
        axis.set_yticks(range(len(ALGORITHMS)), [NAMES[a] for a in ALGORITHMS])
        axis.set_xlim(0, 1.12)
        axis.set_title(title)
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
    fig.suptitle(setup.rsplit("|", 1)[0] + "| seed 1, profiled build")
    fig.tight_layout()
    fig.savefig(args.results / "breakdown.png", dpi=150)

    # Scaling.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ceiling = sorted({r["threads"]: r["ceiling_speedup"] for r in profile}.items())
    for alg in ALGORITHMS:
        series = sorted(
            (r for r in profile if r["node_algorithm"] == alg),
            key=lambda r: r["threads"],
        )
        threads = np.array([r["threads"] for r in series])
        wall = np.array([r["train_secs"] for r in series])
        t1 = singles[alg]["train_secs"]
        idle = [1 - busy(r) / (r["threads"] * r["train_secs"]) for r in series]
        axes[0].plot(threads, wall, "o-", color=COLORS[alg], label=NAMES[alg])
        axes[0].plot(threads, t1 / threads, ":", color=COLORS[alg])
        axes[1].plot(
            threads, t1 / (threads * wall), "o-", color=COLORS[alg], label=NAMES[alg]
        )
        axes[2].plot(threads, idle, "o-", color=COLORS[alg], label=NAMES[alg])
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="threads",
        ylabel="training time (s); dotted: perfect scaling",
    )
    axes[1].plot(
        [t for t, _ in ceiling],
        [c / t for t, c in ceiling],
        "--",
        color="gray",
        label="hardware ceiling: independent distance work",
    )
    axes[1].set(
        xscale="log",
        xlabel="threads",
        ylabel="parallel efficiency T1 / (P x TP)",
        ylim=(0, 1.05),
    )
    axes[2].set(
        xscale="log",
        xlabel="threads",
        ylabel="idle share of thread-seconds",
        ylim=(0, 1),
    )
    for axis in axes:
        axis.legend(fontsize=8)
    fig.suptitle(setup.rsplit("|", 1)[0] + "| seed 1, profiled build")
    fig.tight_layout()
    fig.savefig(args.results / "scaling.png", dpi=150)

    # Busy threads over time.
    fig, axes = plt.subplots(len(ALGORITHMS), 1, figsize=(12, 5.5), sharex=True)
    for axis, alg in zip(axes, ALGORITHMS):
        run = fulls[alg]
        timeline = np.array(run["busy_timeline"])
        times = (np.arange(len(timeline)) + 0.5) * run["busy_timeline_window_secs"]
        axis.fill_between(times, timeline, color=COLORS[alg], step="mid")
        axis.axhline(max_threads, linestyle="--", color="gray")
        axis.set_ylim(0, max_threads * 1.1)
        axis.set_ylabel("busy threads")
        axis.set_title(
            f"{NAMES[alg]}: {run['train_secs']:.1f} s, mean "
            f"{np.mean(timeline):.1f} of {max_threads} threads busy",
            fontsize=10,
        )
    axes[-1].set_xlabel("seconds into training")
    fig.suptitle(setup.rsplit("|", 1)[0] + "| seed 1, profiled build, sampled every ms")
    fig.tight_layout()
    fig.savefig(args.results / "utilization.png", dpi=150)

    # Summary.
    lines = [f"# Hierarchical Lloyd vs hierarchical e-means\n\n{setup}\n"]
    lines.append("| | Lloyd | e-means |\n|---|---|---|")
    lines.append(
        "| training time, median (s) | "
        + " | ".join(
            f"{np.median(secs[a]):.1f} ({secs[a].min():.1f}-{secs[a].max():.1f})"
            for a in ALGORITHMS
        )
        + " |"
    )
    lines.append(
        "| validation MSE / Lloyd | 1 | "
        + "{:.4f} [{:.4f}, {:.4f}]".format(*quality["emeans"][:3])
        + " |"
    )
    lines.append(
        "| validation MSE, mean | "
        + " | ".join(
            f"{np.mean([r['val_mse'] for r in by_alg[a]]):.1f}" for a in ALGORITHMS
        )
        + " |"
    )
    lines.append(
        "| training MSE, mean | "
        + " | ".join(
            f"{np.mean([r['train_mse'] for r in by_alg[a]]):.1f}" for a in ALGORITHMS
        )
        + " |"
    )
    for alg in ALGORITHMS:
        single, full = singles[alg], fulls[alg]
        split = single_thread_split(single)
        assign = (
            split["node training: assign"] + split["membership (assign all node rows)"]
        )
        update = split["node training: update"] + split["leaf means"]
        capacity, cap_split, inflation = capacity_split(full, single)
        idle = cap_split["idle"]
        lines.append(f"\n## {NAMES[alg]}\n")
        lines.append(
            f"One thread: {single['train_secs']:.1f} s; timers cover "
            f"{busy(single) / single['train_secs']:.1%} of it as busy work.\n"
        )
        lines.append("| step (one thread) | seconds | share |\n|---|---|---|")
        for key, value in split.items():
            lines.append(
                f"| {key} | {value:.1f} | {value / single['train_secs']:.1%} |"
            )
        lines.append(
            f"\nAssign vs update, one thread: assign {assign / (assign + update):.1%}, "
            f"update {update / (assign + update):.1%} (node training plus membership "
            "and leaf means).\n"
        )
        lines.append(
            f"{max_threads} threads: {full['train_secs']:.1f} s wall, "
            f"{capacity:.0f} thread-seconds, speedup "
            f"{single['train_secs'] / full['train_secs']:.1f}x, efficiency "
            f"{single['train_secs'] / capacity:.1%} (an embarrassingly parallel "
            f"distance computation reaches {full['ceiling_speedup']:.1f}x here).\n"
        )
        lines.append("| thread-seconds | seconds | share |\n|---|---|---|")
        for key, value in cap_split.items():
            lines.append(f"| {key} | {value:.1f} | {value / capacity:.1%} |")
        lines.append(
            f"\nIdle {idle / capacity:.1%} of thread-seconds. Busy work grew by "
            f"{inflation:.1f} s ({inflation / capacity:.1%} of thread-seconds) over the "
            "one-thread run: the same work runs slower when every core is loaded, "
            "including on the slower efficiency cores. "
            f"{parallel_regions(full):,} fork-joins; an empty 64-task one costs "
            f"{full['empty_region_secs']['tasks_64'] * 1e6:.0f} µs here. "
            f"At most a quarter of the threads were busy for "
            f"{starved_share(full, max_threads // 4):.1%} of the wall time.\n"
        )
    (args.results / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
