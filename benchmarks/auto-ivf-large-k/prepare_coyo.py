# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Freeze the full COYO RQ5 study; see COYO_SEARCH_EFFORT_PROTOCOL.md."""

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import lance
import numpy as np
import prepare
import pyarrow as pa
from common import emit, matrix, positions, row_ids_by_position, save, summary
from lance.dataset import VectorIndexReader
from lance.lance import indices
from prepare_large import sha256

NAME = "coyo-ve-qwen3vl-2048"
REVISION = "95efb82f4914f320e59cf286a06f23b7cfdb2112"
ROWS = 15_380_795
DIMENSION = 2048
MAX_K = 100_000
INDEX_NAME = "coyo_rq5"


def check_oracle(out):
    """Exercise blockwise merging against an independent exhaustive reference."""
    rng = np.random.default_rng(2254)
    vectors = rng.normal(size=(37, 9)).astype(np.float32)
    queries = rng.normal(size=(5, 9)).astype(np.float32)
    dataset = lance.write_dataset(
        pa.table({"vector": pa.FixedSizeListArray.from_arrays(vectors.ravel(), 9)}),
        out / "oracle-fixture.lance",
    )
    old_batch, old_block = prepare.BASE_BATCH, prepare.QUERY_BLOCK
    checks = []
    try:
        prepare.BASE_BATCH, prepare.QUERY_BLOCK = 7, 2
        for metric in ["cosine", "dot", "l2"]:
            expected = np.empty((len(queries), len(vectors)), dtype=np.float64)
            for query_id, query in enumerate(queries.astype(np.float64)):
                for row, vector in enumerate(vectors.astype(np.float64)):
                    if metric == "cosine":
                        expected[query_id, row] = -np.dot(query, vector) / (
                            np.linalg.norm(query) * np.linalg.norm(vector)
                        )
                    elif metric == "dot":
                        expected[query_id, row] = -np.dot(query, vector)
                    else:
                        expected[query_id, row] = np.dot(vector, vector) - 2 * np.dot(
                            query, vector
                        )
            for k in [1, 13, len(vectors)]:
                ids, scores = prepare.exact_truth(dataset, metric, queries, k)
                order = np.argsort(expected, axis=1)[:, :k]
                assert np.array_equal(ids, order)
                np.testing.assert_allclose(
                    scores, np.take_along_axis(expected, order, axis=1), atol=1e-12
                )
                checks.append({"metric": metric, "k": k})
    finally:
        prepare.BASE_BATCH, prepare.QUERY_BLOCK = old_batch, old_block
    save(out / "oracle-check.json", {"passed": True, "checks": checks})


def freeze_queries(source, root):
    out = root / "source" / NAME
    out.mkdir(parents=True)
    check_oracle(out)
    public = lance.dataset(source / "queries.lance", version=1)
    all_queries = matrix(public.to_table(columns=["vector"])["vector"])
    assert all_queries.shape == (25_000, DIMENSION)
    assert all_queries.dtype == np.float32 and np.isfinite(all_queries).all()
    full_split = out / "full-split"
    full_split.mkdir()
    prepare.save_query_split(all_queries, full_split)
    split = np.load(full_split / "split.npz")
    selected = np.concatenate([split["calibration"][:2], split["evaluation"][:128]])
    table = public.take(selected.tolist()).append_column(
        "source_position", pa.array(selected)
    )
    assert table["query_id"].to_pylist() == [f"q{row:06d}" for row in selected]
    lance.write_dataset(table, root / "source" / "data" / NAME / "queries.lance")
    np.savez(out / "split.npz", calibration=np.arange(2), evaluation=np.arange(2, 130))
    save(
        out / "query-selection.json",
        {
            "source_revision": REVISION,
            "full_split_sha256": sha256(full_split / "split.npz"),
            "selected": [
                {
                    "local_position": i,
                    "source_position": int(row),
                    "query_id": f"q{row:06d}",
                    "split": "calibration" if i < 2 else "evaluation",
                }
                for i, row in enumerate(selected)
            ],
        },
    )
    emit("queries_frozen", count=len(selected), evaluation=128)
    return out, table


def crosscheck_truth(dataset, table, truth, scores, out):
    checks = []
    for i, query in enumerate(matrix(table["vector"]).astype(np.float64)):
        published = np.asarray(table["neighbors"][i].as_py(), dtype=np.int64)
        published_scores = np.asarray(table["scores"][i].as_py(), dtype=np.float64)
        assert len(published) == len(set(published.tolist())) == 1000
        assert np.all((published >= 0) & (published < ROWS))
        rows = dataset.take(published.tolist(), columns=["id", "vector"])
        assert rows["id"].to_pylist() == table["neighbor_ids"][i].as_py()
        vectors = matrix(rows["vector"]).astype(np.float64)
        exact = (
            vectors @ query / np.linalg.norm(vectors, axis=1) / np.linalg.norm(query)
        )
        missing = np.setdiff1d(published, truth[i, :1000])
        # Compare every published score using its own ID; order can differ at ties.
        checks.append(
            {
                "local_position": i,
                "source_query_id": table["query_id"][i].as_py(),
                "top1000_overlap": len(np.intersect1d(published, truth[i, :1000]))
                / 1000,
                "max_published_score_error": float(
                    np.max(np.abs(exact - published_scores))
                ),
                "missing_published_ids": missing.tolist(),
                "largest_missing_similarity_above_cutoff": (
                    float(np.max(exact[np.isin(published, missing)] + scores[i, 999]))
                    if len(missing)
                    else None
                ),
            }
        )
    save(out / "public-truth-crosscheck.json", checks)
    emit(
        "public_truth_crosschecked",
        overlap=summary([row["top1000_overlap"] for row in checks]),
        max_score_error=max(row["max_published_score_error"] for row in checks),
    )


def run(source, root):
    verified = json.loads((source.parent / "coyo-source-verified.json").read_text())
    assert verified["revision"] == REVISION
    assert verified["repo"] == f"lance-format/{NAME}"
    assert not root.exists(), "Preserve existing preparations and measurements"
    out, table = freeze_queries(source, root)
    save(root / "source-verification.json", verified)
    original = lance.dataset(source / "base.lance", version=1)
    assert original.count_rows() == ROWS and original.list_indices() == []
    assert np.array_equal(
        original.to_table(columns=["row"])["row"].to_numpy(), np.arange(ROWS)
    )
    lookup = row_ids_by_position(original)
    np.save(out / "row-ids.npy", lookup)
    queries = matrix(table["vector"])
    started = time.monotonic()
    truth, scores = prepare.exact_truth(original, "cosine", queries, MAX_K)
    assert np.all((truth >= 0) & (truth < ROWS))
    assert all(len(np.unique(row)) == MAX_K for row in truth)
    np.save(out / "truth-ids.npy", truth)
    np.save(out / "truth-scores.npy", scores)
    save(out / "truth-timing.json", {"seconds": time.monotonic() - started})
    emit("truth_complete", seconds=time.monotonic() - started)
    crosscheck_truth(original, table, truth, scores, out)

    indexed = root / "indices" / NAME
    indexed.mkdir(parents=True)
    clone = original.shallow_clone(indexed / "rq5.lance", original.version)
    assert np.array_equal(row_ids_by_position(clone), lookup)
    model = indices.build_rq_model(dimension=DIMENSION, num_bits=5)
    (indexed / "rabitq-model.json").write_text(model + "\n")
    partitions = math.ceil(ROWS / 4096)
    started = time.monotonic()
    last = [0.0]

    def progress(event):
        now = time.monotonic()
        if event.event != "progress" or now - last[0] >= 30:
            emit("rq5_build_progress", progress=dataclasses.asdict(event))
            last[0] = now

    clone.create_index(
        "vector",
        "IVF_RQ",
        name=INDEX_NAME,
        metric="cosine",
        num_bits=5,
        rabitq_model=model,
        num_partitions=partitions,
        progress_callback=progress,
    )
    stats = clone.stats.index_stats(INDEX_NAME)
    assert stats["index_type"] == "IVF_RQ" and stats["num_segments"] == 1
    assert stats["num_indexed_rows"] == ROWS and stats["num_unindexed_rows"] == 0
    assert stats["indices"][0]["sub_index"]["num_bits"] == 5
    assert stats["indices"][0]["sub_index"]["packed"] is True
    for segment in stats["indices"]:
        segment.pop("centroids", None)
    centroids = matrix(clone.centroids(index_name=INDEX_NAME))
    np.save(out / "centroids.npy", centroids)
    membership = np.full(ROWS, -1, dtype=np.int32)
    reader = VectorIndexReader(clone, INDEX_NAME)
    assert reader.num_partitions() == partitions
    for partition in range(partitions):
        rows = positions(reader.read_partition(partition)["_rowid"].to_numpy(), lookup)
        assert len(np.unique(rows)) == len(rows) and np.all(membership[rows] == -1)
        membership[rows] = partition
        if partition % 512 == 0:
            emit("rq5_membership_audit", partition=partition)
    assert np.all(membership >= 0)
    np.save(out / "membership.npy", membership)
    build = {
        "rows": ROWS,
        "partitions": partitions,
        "index_name": INDEX_NAME,
        "version": original.version,
        "rq_version": clone.version,
    }
    save(out / "build.json", build)
    source_hashes = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    save(out / "prepared.json", {"sha256": source_hashes, "source_revision": REVISION})
    rq_indices = clone.list_indices()
    assert len(rq_indices) == 1
    clone_path = indexed / "rq5.lance"
    index_dir = clone_path / "_indices" / rq_indices[0]["uuid"]
    index_hashes = {
        str(p.relative_to(clone_path)): sha256(p)
        for p in sorted(index_dir.rglob("*"))
        if p.is_file()
    }
    assert index_hashes
    save(
        indexed / "prepared.json",
        {
            "source": str((root / "source").resolve()),
            "corpus": NAME,
            "metric": "cosine",
            "source_build": build,
            "rq_version": clone.version,
            "rq_indices": rq_indices,
            "rq_stats": stats,
            "rabitq_model_sha256": sha256(indexed / "rabitq-model.json"),
            "index_sha256": index_hashes,
            "input_sha256": {
                **source_hashes,
                "prepared.json": sha256(out / "prepared.json"),
            },
            "membership_complete": True,
            "build_seconds": time.monotonic() - started,
            "binary_sha256": sha256(Path(lance.__file__).parent / "lance.abi3.so"),
        },
    )
    emit("coyo_prepared", seconds=time.monotonic() - started)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    run(args.source, args.root)
