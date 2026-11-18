# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Fit PCA to a prefix of a `.bvecs` file, for `emeans_bench --pca`.

Writes the mean, then every principal component in decreasing order of
variance, as little-endian float32 (`dim + 1` rows of `dim` values), and
prints the share of variance the leading components keep.

    uv run benchmarks/emeans/pca.py --base base.bvecs --rows 1000000 --out pca.f32
"""

import argparse
import json

import numpy as np

CHUNK_ROWS = 100_000


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    raw = np.memmap(args.base, dtype=np.uint8, mode="r")
    dim = int(raw[:4].view("<u4")[0])
    rows = raw[: args.rows * (dim + 4)].reshape(args.rows, dim + 4)
    if not (rows[:, :4].view("<u4")[:, 0] == dim).all():
        raise ValueError(f"{args.base} is not a {dim}-dimensional bvecs file")

    total = np.zeros(dim)
    gram = np.zeros((dim, dim))
    for start in range(0, args.rows, CHUNK_ROWS):
        chunk = rows[start : start + CHUNK_ROWS, 4:].astype(np.float64)
        total += chunk.sum(axis=0)
        gram += chunk.T @ chunk
    mean = total / args.rows
    covariance = gram / args.rows - np.outer(mean, mean)
    variances, components = np.linalg.eigh(covariance)
    order = np.argsort(variances)[::-1]
    variances, components = variances[order], components[:, order].T

    np.concatenate([mean[None, :], components]).astype("<f4").tofile(args.out)
    kept = np.cumsum(variances) / variances.sum()
    print(
        json.dumps(
            {d: round(float(kept[d - 1]), 4) for d in (32, 64, 128, 256, 512, dim)}
        )
    )


if __name__ == "__main__":
    main()
