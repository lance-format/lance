#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Compare two arms of the sparse structural benchmarks.

Two subcommands, because the two measurements need different statistics:

  footprint <before.json> <after.json>
      Exact byte counts, so a plain per-case diff is the whole story.

  timing <criterion-root>
      Criterion means collected over several interleaved rounds. Reported as a paired
      comparison: each round contributes one before/after pair, and the p-value comes from
      a permutation test over the per-round deltas. That is the right test here because the
      rounds are matched in time, so it controls for the drift that makes a naive
      before-then-after comparison unreliable.
"""

from __future__ import annotations

import itertools
import json
import statistics
import sys
from pathlib import Path

# Below this, a timing difference is not distinguishable from run-to-run noise on a typical
# developer machine even with pinning, so it is reported as "noise" regardless of p-value.
NOISE_FLOOR_PCT = 3.0


def load_footprint(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            rows[row["case"]] = row
    return rows


def fmt_bytes(n: float) -> str:
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if abs(n) >= scale:
            return f"{n / scale:,.2f} {unit}"
    return f"{n:,.0f} B"


def pct(before: float, after: float) -> str:
    if before == 0:
        return "n/a" if after == 0 else "+inf"
    return f"{100.0 * (after - before) / before:+.1f}%"


def footprint(before_path: Path, after_path: Path) -> None:
    before = load_footprint(before_path)
    after = load_footprint(after_path)

    shared = [c for c in before if c in after]
    if not shared:
        sys.exit("no cases in common between the two arms")

    missing = sorted(set(before) ^ set(after))
    if missing:
        print(f"warning: cases present in only one arm, skipped: {', '.join(missing)}\n")

    for metric, label in (
        ("cache_bytes", "resident scheduler state (LanceCache)"),
        ("init_bytes", "bytes allocated during initialize"),
        ("init_allocs", "allocation count during initialize"),
    ):
        print(f"\n{label}")
        print(
            f"  {'case':<30} {'layout':>9} {'pages':>6} {'before':>14} {'after':>14} {'delta':>10}"
        )
        print("  " + "-" * 88)
        total_before = total_after = 0
        for case in shared:
            b, a = before[case][metric], after[case][metric]
            total_before += b
            total_after += a
            show = fmt_bytes if metric != "init_allocs" else lambda n: f"{n:,.0f}"
            print(
                f"  {case:<30} {before[case]['layout']:>9} {before[case]['pages']:>6} "
                f"{show(b):>14} {show(a):>14} {pct(b, a):>10}"
            )
        show = fmt_bytes if metric != "init_allocs" else lambda n: f"{n:,.0f}"
        print("  " + "-" * 88)
        print(
            f"  {'total':<30} {'':>9} {'':>6} {show(total_before):>14} "
            f"{show(total_after):>14} {pct(total_before, total_after):>10}"
        )

    print(
        "\nA negative delta on resident scheduler state is the headline: that state is held"
        "\nfor as long as the dataset is open, once per page per column."
    )


def load_timing(root: Path) -> dict[str, dict[str, dict[int, float]]]:
    """-> {group/bench: {arm: {round: mean_seconds}}}"""
    out: dict[str, dict[str, dict[int, float]]] = {}
    for estimates in root.glob("r*/*/**/new/estimates.json"):
        parts = estimates.relative_to(root).parts
        rnd = int(parts[0][1:])
        arm = parts[1]
        bench = "/".join(parts[2:-2])
        try:
            mean = json.loads(estimates.read_text())["mean"]["point_estimate"]
        except (json.JSONDecodeError, KeyError):
            continue
        out.setdefault(bench, {}).setdefault(arm, {})[rnd] = mean / 1e9
    return out


def permutation_p(deltas: list[float]) -> float:
    """Exact two-sided sign-flip permutation test on paired deltas.

    With n rounds there are only 2**n sign assignments, so for the round counts used here
    the test enumerates them all rather than sampling. Note the consequence: the smallest
    p-value attainable is 2/2**n, so fewer than 6 rounds can never clear p<0.05.
    """
    n = len(deltas)
    if n == 0:
        return 1.0
    observed = abs(statistics.fmean(deltas))
    if n > 20:  # keep it exact but bounded
        deltas = deltas[:20]
        n = 20
    extreme = 0
    total = 0
    for signs in itertools.product((1, -1), repeat=n):
        total += 1
        flipped = statistics.fmean(s * d for s, d in zip(signs, deltas))
        if abs(flipped) >= observed:
            extreme += 1
    return extreme / total


def timing(root: Path) -> None:
    data = load_timing(root)
    if not data:
        sys.exit(f"no criterion estimates found under {root}")

    print(
        f"  {'benchmark':<40} {'before':>10} {'after':>10} {'delta':>9} {'rounds':>7} {'p':>7}  verdict"
    )
    print("  " + "-" * 100)

    for bench in sorted(data):
        arms = data[bench]
        if "before" not in arms or "after" not in arms:
            continue
        rounds = sorted(set(arms["before"]) & set(arms["after"]))
        if not rounds:
            continue
        pairs = [(arms["before"][r], arms["after"][r]) for r in rounds]
        deltas = [a - b for b, a in pairs]
        mean_before = statistics.fmean(b for b, _ in pairs)
        mean_after = statistics.fmean(a for _, a in pairs)
        change = 100.0 * statistics.fmean(deltas) / mean_before if mean_before else 0.0
        p = permutation_p(deltas)

        if abs(change) < NOISE_FLOOR_PCT:
            verdict = "noise"
        elif p >= 0.05:
            verdict = "not significant"
        else:
            verdict = "SLOWER" if change > 0 else "faster"

        print(
            f"  {bench:<40} {mean_before * 1e3:>9.3f}ms {mean_after * 1e3:>9.3f}ms "
            f"{change:>+8.1f}% {len(rounds):>7} {p:>7.3f}  {verdict}"
        )

    print(
        f"\n  Deltas below {NOISE_FLOOR_PCT}% are reported as noise regardless of p-value."
        "\n  p is an exact sign-flip permutation test over per-round paired deltas."
    )


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    mode = sys.argv[1]
    if mode == "footprint":
        footprint(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "timing":
        timing(Path(sys.argv[2]))
    else:
        sys.exit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
