# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Non-production paired IVF_RQ POC. Run through the checkout's uv environment."""

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
from lance.indices import IndicesBuilder
from lance.lance import indices

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--queries", type=int, default=32)
parser.add_argument("--partitions", type=int, default=4096)
parser.add_argument("--nprobes", type=int, default=64)
parser.add_argument("--bits", default="3,5,7,9")
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--prepare-only", action="store_true")
parser.add_argument("--build-only", action="store_true")
parser.add_argument(
    "--cascade-factors", default="", help="Optional code-only cascade diagnostics"
)
args = parser.parse_args()
root, out = Path(args.dataset), Path(args.output)
out.mkdir(parents=True, exist_ok=True)
base = lance.dataset(root / "base.lance")
query_ds = lance.dataset(root / "queries.lance")
n, dim = base.count_rows(), base.schema.field("vector").type.list_size
query_ids = np.linspace(0, query_ds.count_rows() - 1, args.queries, dtype=np.int64)
queries = (
    query_ds.take(query_ids, columns=["vector"])["vector"]
    .combine_chunks()
    .flatten()
    .to_numpy()
    .reshape(-1, dim)
)
np.save(out / "queries.npy", queries)
if not (out / "queries.lance").exists():
    lance.write_dataset(
        pa.table(
            {
                "vector": pa.FixedSizeListArray.from_arrays(
                    pa.array(queries.ravel()), dim
                )
            }
        ),
        out / "queries.lance",
    )
assert np.allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-3), (
    "POC expects normalized queries"
)
records = out / "results.jsonl"


def emit(value):
    with records.open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


emit(
    {
        "event": "environment",
        "dataset": str(root),
        "rows": n,
        "dimensions": dim,
        "query_ids": query_ids.tolist(),
        "nprobes": args.nprobes,
        "partitions": args.partitions,
        "version": lance.__version__,
        "source_id": os.environ.get("LAYERED_POC_SOURCE_ID"),
        "threads": os.cpu_count(),
        "provenance": json.loads((root / "provenance.json").read_text())["revision"],
    }
)

# Ground truth at deep k is recomputed for this exact query subset and corpus.
gt_path = out / "gt10000.npy"
if gt_path.exists():
    gt = np.load(gt_path)
    assert gt.shape == (len(queries), 10000)
else:
    best_scores = np.empty((0, len(queries)), np.float32)
    best_ids = np.empty((0, len(queries)), np.int64)
    offset = 0
    started = time.perf_counter()
    for batch in base.to_batches(columns=["vector"], batch_size=32768):
        vec = batch["vector"].flatten().to_numpy().reshape(-1, dim)
        assert len(vec) == batch.num_rows
        assert np.allclose(np.linalg.norm(vec, axis=1), 1, atol=1e-3), (
            "POC expects normalized corpus"
        )
        scores = vec @ queries.T
        ids = np.broadcast_to(
            np.arange(offset, offset + len(vec), dtype=np.int64)[:, None], scores.shape
        )
        scores, ids = (
            np.concatenate([best_scores, scores]),
            np.concatenate([best_ids, ids]),
        )
        keep = min(10000, len(scores))
        selected = np.argpartition(-scores, keep - 1, axis=0)[:keep]
        best_scores, best_ids = (
            np.take_along_axis(scores, selected, axis=0),
            np.take_along_axis(ids, selected, axis=0),
        )
        offset += len(vec)
    assert offset == n
    order = np.argsort(-best_scores, axis=0)
    gt = np.take_along_axis(best_ids, order, axis=0).T
    np.save(gt_path, gt)
    emit(
        {
            "event": "exact_gt",
            "seconds": time.perf_counter() - started,
            "rows": offset,
            "k": gt.shape[1],
        }
    )

model_path, centroids_path = out / "rotation.json", out / "centroids.arrow"
if not model_path.exists():
    model_path.write_text(
        indices.build_rq_model(dimension=dim, num_bits=9, dtype="float32")
    )
if centroids_path.exists():
    with pa.ipc.open_file(centroids_path) as reader:
        centroids = reader.read_all()["centroids"].combine_chunks()
else:
    started = time.perf_counter()
    centroids = (
        IndicesBuilder(base, "vector")
        .train_ivf(args.partitions, distance_type="cosine")
        .centroids
    )
    table = pa.table({"centroids": centroids})
    with pa.ipc.new_file(centroids_path, table.schema) as writer:
        writer.write_table(table)
    emit({"event": "train_ivf", "seconds": time.perf_counter() - started})
emit(
    {
        "event": "models",
        "rotation_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "centroids_sha256": hashlib.sha256(centroids_path.read_bytes()).hexdigest(),
    }
)

# Freeze exact assignments as well as centroids: independent approximate HNSW
# assignment runs must not become a recall/performance confound between layouts.
assignment_uri = out / "assignments.lance"
assignment_meta = out / "assignments.json"
centroid_hash = hashlib.sha256(centroids_path.read_bytes()).hexdigest()
if not assignment_meta.exists():
    centers = centroids.flatten().to_numpy().reshape(args.partitions, dim).copy()
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assignment_digest = hashlib.sha256()
    assignment_rows = 0
    started = time.perf_counter()

    def partition_batches():
        global assignment_rows
        for batch in base.to_batches(
            columns=["vector"], with_row_id=True, batch_size=4096
        ):
            vectors = batch["vector"].flatten().to_numpy().reshape(-1, dim)
            parts = np.argmax(vectors @ centers.T, axis=1).astype(np.uint32)
            row_ids = batch["_rowid"]
            assignment_digest.update(row_ids.to_numpy().tobytes())
            assignment_digest.update(parts.tobytes())
            assignment_rows += len(parts)
            yield pa.record_batch(
                [row_ids, pa.array(parts)], names=["row_id", "partition"]
            )

    lance.write_dataset(
        partition_batches(),
        assignment_uri,
        mode="overwrite",
        schema=pa.schema(
            [pa.field("row_id", pa.uint64()), pa.field("partition", pa.uint32())]
        ),
        max_rows_per_file=n,
        max_rows_per_group=32768,
    )
    assert assignment_rows == n
    assignment_meta.write_text(
        json.dumps(
            {
                "rows": n,
                "centroids_sha256": centroid_hash,
                "assignment_sha256": assignment_digest.hexdigest(),
                "seconds": time.perf_counter() - started,
            }
        )
    )
assignment_info = json.loads(assignment_meta.read_text())
assert (
    assignment_info["rows"] == n
    and assignment_info["centroids_sha256"] == centroid_hash
)
emit({"event": "assignments", **assignment_info})

starts = {}
offset = 0
for fragment in base.get_fragments():
    starts[fragment.fragment_id] = offset
    offset += fragment.count_rows()


def search(ds, query, k, precision="full", cascade=None, refine=None):
    nearest = {
        "column": "vector",
        "q": query,
        "k": k,
        "nprobes": args.nprobes,
        "rq_precision": precision,
        "rq_cascade_factor": cascade,
        "refine_factor": refine,
    }
    result = ds.to_table(columns=[], with_row_id=True, nearest=nearest)
    rowids = result["_rowid"].combine_chunks().to_numpy()
    ids = np.fromiter(
        (starts[int(v) >> 32] + (int(v) & 0xFFFFFFFF) for v in rowids), dtype=np.int64
    )
    return ids


if args.prepare_only:
    raise SystemExit(0)

for bits in map(int, args.bits.split(",")):
    model = json.loads(model_path.read_text())
    model["num_bits"] = bits
    for layered in [False] if bits == 3 else [False, True]:
        name = f"{'layered' if layered else 'native'}{bits}"
        uri = out / (name + ".lance")
        if not uri.exists():
            shutil.copytree(
                root / "base.lance",
                uri,
                copy_function=lambda source, target: os.link(source, target)
                if "/data/" in source
                else shutil.copy2(source, target),
            )
        dataset = lance.dataset(uri)
        if not dataset.list_indices():
            started = time.perf_counter()
            dataset.create_index(
                "vector",
                index_type="IVF_RQ",
                num_bits=bits,
                layered=layered,
                num_partitions=args.partitions,
                metric="cosine",
                ivf_centroids=centroids,
                precomputed_partition_dataset=str(assignment_uri),
                rabitq_model=json.dumps(model),
            )
            emit(
                {
                    "event": "build",
                    "name": name,
                    "seconds": time.perf_counter() - started,
                }
            )
        modes = [("full", None, None)]
        if args.build_only:
            continue
        if layered:
            modes += [
                ("sign", None, None),
                ("high", None, None),
            ]
            modes += [
                ("full", int(factor), None)
                for factor in args.cascade_factors.split(",")
                if factor
            ]
        for precision, cascade, refine in modes:
            ds = lance.dataset(uri, index_cache_size_bytes=128 * 1024**3)
            for index in ds.list_indices():
                ds.prewarm_index(index["name"])
            resident_bytes = ds.session().index_cache_size_bytes()
            assert resident_bytes < 0.95 * 128 * 1024**3, (
                "warm-memory run requires a larger cache"
            )
            emit(
                {
                    "event": "prewarm",
                    "name": name,
                    "precision": precision,
                    "cascade_factor": cascade,
                    "refine_factor": refine,
                    "resident_bytes": resident_bytes,
                    "cache_capacity_bytes": 128 * 1024**3,
                }
            )
            for k in [100, 1000, 10000]:
                # Untimed warm-up for every measured mode/k pair.
                for q in queries:
                    search(ds, q, k, precision, cascade, refine)
                times, recalls, first_results = [], [], []
                for repeat in range(args.repeats):
                    for qi, q in enumerate(queries):
                        started = time.perf_counter()
                        ids = search(ds, q, k, precision, cascade, refine)
                        times.append((time.perf_counter() - started) * 1000)
                        recalls.append(len(np.intersect1d(ids, gt[qi, :k])) / k)
                        if repeat == 0:
                            first_results.append(ids)
                artifact = (
                    out / f"{name}_{precision}_cascade{cascade}_refine{refine}_k{k}.npz"
                )
                predicted = np.full((len(queries), k), -1, dtype=np.int64)
                for qi, ids in enumerate(first_results):
                    predicted[qi, : len(ids)] = ids
                np.savez_compressed(
                    artifact,
                    predicted=predicted,
                    returned_rows=np.asarray([len(ids) for ids in first_results]),
                    truth=gt[:, :k],
                    latencies_ms=np.asarray(times).reshape(args.repeats, len(queries)),
                    recalls=np.asarray(recalls),
                )
                emit(
                    {
                        "event": "query",
                        "name": name,
                        "precision": precision,
                        "cascade_factor": cascade,
                        "refine_factor": refine,
                        "k": k,
                        "cache": "warm_memory",
                        "recall": float(np.mean(recalls)),
                        "p50_ms": float(np.percentile(times, 50)),
                        "p95_ms": float(np.percentile(times, 95)),
                        "p99_ms": float(np.percentile(times, 99)),
                        "samples": len(times),
                        "min_returned_rows": min(map(len, first_results)),
                    }
                )
            if precision == "full" and refine is None:
                for workers in [1, 8, 16, 32]:
                    work = list(queries) * 4
                    started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        list(
                            pool.map(
                                lambda q: search(
                                    ds, q, 1000, precision, cascade, refine
                                ),
                                work,
                            )
                        )
                    emit(
                        {
                            "event": "throughput",
                            "name": name,
                            "cascade_factor": cascade,
                            "k": 1000,
                            "workers": workers,
                            "queries": len(work),
                            "qps": len(work) / (time.perf_counter() - started),
                        }
                    )
