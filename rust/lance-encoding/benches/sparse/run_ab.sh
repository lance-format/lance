#!/usr/bin/env bash
# Interleaved A/B comparison of the sparse structural read path across two git refs.
#
#   ./run_ab.sh <before-ref> <after-ref> [rounds]
#   ./run_ab.sh d8030a364 HEAD 8
#
# Use at least 6 rounds. The paired test in compare.py is an exact two-sided sign-flip
# permutation test, whose smallest attainable p-value is 2/2**rounds; at 5 rounds that
# floor is 0.0625, so no result could ever clear p<0.05 no matter how large the effect.
#
# What it does, and why each step matters:
#
#  1. Checks out both refs into throwaway worktrees.
#  2. Copies *this* tree's bench sources into both worktrees, so identical benchmark code
#     runs against two different library versions. Without this the comparison silently
#     becomes "old bench vs new bench".
#  3. Builds each worktree into its OWN target dir. A shared CARGO_TARGET_DIR is unusable:
#     cargo's unit hash covers package name, version, profile and features but not the
#     source directory, so two worktrees at the same crate version collide and cargo
#     reports the stale artifact as fresh.
#  4. Runs the footprint report on both and diffs it. These numbers are exact, so one run
#     is enough.
#  5. Runs the Criterion benches with the arms interleaved and rotated per round, so
#     thermal drift and background load bias both arms equally instead of whichever arm
#     happens to run first.
#
# Timing results are only meaningful pinned to physical cores on an otherwise idle
# machine. Set CPUS to a list of physical core ids for your host; the default is a
# reasonable guess and will be skipped if taskset is unavailable.
set -euo pipefail

BEFORE=${1:?usage: run_ab.sh <before-ref> <after-ref> [rounds]}
AFTER=${2:?usage: run_ab.sh <before-ref> <after-ref> [rounds]}
ROUNDS=${3:-8}
if ((ROUNDS < 6)); then
    echo "warning: $ROUNDS rounds cannot reach p<0.05 (floor is 2/2**$ROUNDS); use 6 or more" >&2
fi

REPO=$(git rev-parse --show-toplevel)
BENCH_DIR=rust/lance-encoding/benches
WORK=${SPARSE_AB_WORKDIR:-/tmp/sparse-ab}
CPUS=${CPUS:-}

mkdir -p "$WORK"
echo "workdir: $WORK"

# --- 1. worktrees -----------------------------------------------------------------
for arm in before after; do
    ref=$([[ $arm == before ]] && echo "$BEFORE" || echo "$AFTER")
    dir=$WORK/$arm
    if [[ ! -d $dir ]]; then
        git -C "$REPO" worktree add --detach "$dir" "$ref" >/dev/null
    fi
    echo "$arm -> $ref ($(git -C "$dir" rev-parse --short HEAD))"
done

# --- 2. pin the benchmark sources ------------------------------------------------
# Both arms must run byte-identical bench code for the comparison to mean anything.
# A no-op when an arm's worktree is the tree this script was invoked from.
sync_file() {
    [[ $(realpath "$1") == $(realpath -m "$2") ]] || cp "$1" "$2"
}

for arm in before after; do
    dir=$WORK/$arm
    mkdir -p "$dir/$BENCH_DIR/sparse" "$dir/rust/lance-encoding/tests"
    sync_file "$REPO/$BENCH_DIR/sparse/cases.rs" "$dir/$BENCH_DIR/sparse/cases.rs"
    sync_file "$REPO/$BENCH_DIR/sparse_footprint.rs" "$dir/$BENCH_DIR/sparse_footprint.rs"
    sync_file "$REPO/$BENCH_DIR/sparse_decode.rs" "$dir/$BENCH_DIR/sparse_decode.rs"
    sync_file "$REPO/rust/lance-encoding/tests/sparse_bench_layouts.rs" \
        "$dir/rust/lance-encoding/tests/sparse_bench_layouts.rs"

    # Register the bench targets if this ref predates them.
    python3 - "$dir/rust/lance-encoding/Cargo.toml" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
for name in ("sparse_decode", "sparse_footprint"):
    entry = f'[[bench]]\nname = "{name}"\nharness = false\n'
    if entry not in src:
        src = src.replace("[lints]", entry + "\n[lints]", 1)
path.write_text(src)
PY
done

# --- 3. build --------------------------------------------------------------------
for arm in before after; do
    echo "=== building $arm ==="
    CARGO_TARGET_DIR=$WORK/target-$arm cargo build --manifest-path "$WORK/$arm/Cargo.toml" \
        --profile bench -p lance-encoding --benches >"$WORK/$arm.build.log" 2>&1 || {
        echo "build failed for $arm; see $WORK/$arm.build.log" >&2
        tail -30 "$WORK/$arm.build.log" >&2
        exit 1
    }
done

# bin <bench-name> <arm> -> path to the most recently built bench executable.
# The `bench` profile emits into release/deps alongside stale hashes from earlier builds,
# so pick the newest rather than the first match.
bin() {
    find "$WORK/target-$2/release/deps" -maxdepth 1 -name "$1-*" -type f -perm -u+x \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2
}

# --- 4. exact memory comparison --------------------------------------------------
echo
echo "=== layout guard (both arms must agree the inputs are still sparse) ==="
for arm in before after; do
    CARGO_TARGET_DIR=$WORK/target-$arm cargo test --manifest-path "$WORK/$arm/Cargo.toml" \
        -p lance-encoding --test sparse_bench_layouts >"$WORK/$arm.layout.log" 2>&1 &&
        echo "  $arm: ok" ||
        {
            echo "  $arm: FAILED - inputs no longer hit the sparse layout" >&2
            tail -20 "$WORK/$arm.layout.log" >&2
            exit 1
        }
done

echo
echo "=== footprint ==="
for arm in before after; do
    exe=$(bin sparse_footprint "$arm")
    "$exe" --json >"$WORK/$arm.footprint.json"
done
python3 "$REPO/$BENCH_DIR/sparse/compare.py" footprint \
    "$WORK/before.footprint.json" "$WORK/after.footprint.json"

# --- 5. interleaved timing -------------------------------------------------------
echo
echo "=== timing ($ROUNDS rounds, arms rotated) ==="
pin=()
if [[ -n $CPUS ]] && command -v taskset >/dev/null; then
    pin=(taskset -c "$CPUS")
    echo "pinned to cpus $CPUS"
else
    echo "not pinned; set CPUS=<physical core ids> for lower variance"
fi

for ((r = 1; r <= ROUNDS; r++)); do
    # Rotate which arm runs first so neither is systematically favoured.
    order=(before after)
    if ((r % 2 == 0)); then order=(after before); fi
    for arm in "${order[@]}"; do
        home=$WORK/crit/r$r/$arm
        mkdir -p "$home"
        exe=$(bin sparse_decode "$arm")
        CRITERION_HOME=$home "${pin[@]}" "$exe" --bench >"$home/stdout.txt" 2>&1 ||
            echo "  round $r $arm: bench exited nonzero, see $home/stdout.txt" >&2
        echo "  round $r $arm: done"
    done
done

python3 "$REPO/$BENCH_DIR/sparse/compare.py" timing "$WORK/crit"
