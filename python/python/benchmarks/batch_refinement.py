# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Compare native batch refinement with serial and concurrent single queries.

Build with release-with-debug, then run from python/:
    uv run --no-sync python python/benchmarks/batch_refinement.py \
        --directory /tmp/batch-refinement --output /tmp/refinement.json

Reuse the directory across revisions to preserve the dataset, index and queries.
The index is warmed; the OS page cache is not flushed. This is not a cold-storage
benchmark. Dataset creation and validation are excluded from measured latency.
"""

import argparse
import json
import platform
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa


def prepare(args):
    args.directory.mkdir(parents=True, exist_ok=True)
    dataset_path = args.directory / "vectors.lance"
    query_path = args.directory / "queries.npz"
    config_path = args.directory / "config.json"
    config = {
        "rows": args.rows,
        "dimension": args.dimension,
        "partitions": args.partitions,
        "seed": args.seed,
    }
    if dataset_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Existing dataset configuration differs from arguments")
    else:
        rng = np.random.default_rng(args.seed)
        scale = np.exp(-np.arange(args.dimension) / 128).astype(np.float32)
        vectors = rng.standard_normal((args.rows, args.dimension), dtype=np.float32)
        vectors *= scale
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        table = pa.table(
            {
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(vectors.reshape(-1)), args.dimension
                ),
                "id": pa.array(np.arange(args.rows)),
            }
        )
        ds = lance.write_dataset(table, dataset_path, max_rows_per_file=4096)
        ds.create_index(
            "vector",
            index_type="IVF_SQ",
            metric="l2",
            num_partitions=args.partitions,
        )
        independent = rng.standard_normal((64, args.dimension), dtype=np.float32)
        independent *= scale
        independent /= np.linalg.norm(independent, axis=1, keepdims=True)
        anchor = rng.standard_normal(args.dimension, dtype=np.float32) * scale
        anchor /= np.linalg.norm(anchor)
        nearby = (
            anchor
            + 0.005
            * rng.standard_normal((64, args.dimension), dtype=np.float32)
            * scale
        )
        nearby /= np.linalg.norm(nearby, axis=1, keepdims=True)
        np.savez(query_path, independent=independent, nearby=nearby)
        config_path.write_text(json.dumps(config))
    with np.load(query_path) as data:
        queries = {name: data[name] for name in data.files}
    return dataset_path, queries


def unpack(table, count):
    ids = table["id"].to_numpy()
    distances = table["_distance"].to_numpy()
    if "query_index" not in table.column_names:
        return [(ids, distances)]
    indices = table["query_index"].to_numpy()
    return [(ids[indices == i], distances[indices == i]) for i in range(count)]


def validate(actual, expected):
    assert len(actual) == len(expected)
    for (ids, distances), (expected_ids, expected_distances) in zip(actual, expected):
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(distances, expected_distances, atol=2e-6, rtol=2e-5)


def run(args):
    dataset_path, query_sets = prepare(args)
    ds = lance.dataset(dataset_path, index_cache_size_bytes=512 * 1024**2)
    parameters = {
        "column": "vector",
        "k": args.k,
        "nprobes": args.nprobes,
        "refine_factor": args.refine_factor,
        "query_parallelism": 1,
    }

    def single(query):
        return unpack(
            ds.to_table(
                columns=["id", "_distance"], nearest={**parameters, "q": query}
            ),
            1,
        )[0]

    records = []
    plans = {}
    checksums = {}
    recall = {}
    rng = random.Random(args.seed)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for distribution, all_queries in query_sets.items():
            hits = 0
            for query in all_queries[:8]:
                actual_ids, _ = single(query)
                exact = ds.to_table(
                    columns=["id", "_distance"],
                    nearest={
                        "column": "vector",
                        "q": query,
                        "k": args.k,
                        "use_index": False,
                    },
                )["id"].to_numpy()
                hits += len(np.intersect1d(actual_ids, exact))
            recall[distribution] = hits / (8 * args.k)
            for width in args.batch_sizes:
                queries = all_queries[:width]
                expected = [single(query) for query in queries]
                checksums[f"{distribution}/{width}"] = [
                    ids.tolist() for ids, _ in expected
                ]
                nearest = {**parameters, "q": queries}
                scanner = ds.scanner(columns=["id", "_distance"], nearest=nearest)
                plans[f"{distribution}/{width}"] = scanner.analyze_plan()

                def serial():
                    return [single(query) for query in queries]

                def parallel():
                    return list(executor.map(single, queries))

                def batch():
                    return unpack(
                        ds.to_table(columns=["id", "_distance"], nearest=nearest), width
                    )

                methods = {"serial": serial, "parallel": parallel, "batch": batch}
                for method in methods.values():
                    validate(method(), expected)
                for repeat in range(args.repeats):
                    order = list(methods)
                    rng.shuffle(order)
                    for name in order:
                        start = time.perf_counter()
                        result = methods[name]()
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        validate(result, expected)
                        records.append(
                            {
                                "distribution": distribution,
                                "batch_size": width,
                                "method": name,
                                "repeat": repeat,
                                "elapsed_ms": elapsed_ms,
                            }
                        )
    medians = [
        {
            "distribution": distribution,
            "batch_size": width,
            **{
                name: float(
                    np.median(
                        [
                            record["elapsed_ms"]
                            for record in records
                            if record["distribution"] == distribution
                            and record["batch_size"] == width
                            and record["method"] == name
                        ]
                    )
                )
                for name in ["serial", "parallel", "batch"]
            },
        }
        for distribution in query_sets
        for width in args.batch_sizes
    ]
    output = {
        "label": args.label,
        "platform": platform.platform(),
        "lance_version": lance.__version__,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset_version": ds.version,
        "recall_at_k_over_eight_queries": recall,
        "plans": plans,
        "result_ids": checksums,
        "medians_ms": medians,
        "samples": records,
    }
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="release-with-debug")
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--dimension", type=int, default=1024)
    parser.add_argument("--partitions", type=int, default=64)
    parser.add_argument("--nprobes", type=int, default=16)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--refine-factor", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    if any(width < 1 or width > 64 for width in arguments.batch_sizes):
        parser.error("batch sizes must be between 1 and 64")
    run(arguments)
