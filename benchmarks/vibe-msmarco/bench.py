#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Download, index, and time IVF_RQ search on vibe-msmarco-qwen-1024.

Queries project only ``_rowid``. Each (version, index) cell drops the OS page
cache first so later cells do not inherit another index's pages. The first
query of each mode is discarded (process / file-open / JIT). Cold then times
fresh dataset handles with metadata already opened; warm calls
``prewarm_index`` and reuses one handle.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

HF_DATASET = "lance-format/vibe-msmarco-qwen-1024"
VECTOR_COLUMN = "vector"
NUM_PARTITIONS = 1024
NPROBES = 20
TOP_K = 10
METRIC = "cosine"
DEFAULT_QUERY_COUNT = 100
DEFAULT_DISCARD_FIRST = 1
# Large enough to hold IVF_RQ5 codes (~6 GiB) plus IVF metadata.
INDEX_CACHE_BYTES = 8 * 1024 * 1024 * 1024


def _import_lance():
    import lance

    return lance


def _dataset(lance, uri: str, cache_bytes: int = INDEX_CACHE_BYTES):
    try:
        return lance.dataset(uri, index_cache_size_bytes=cache_bytes)
    except TypeError:
        # Older bindings only expose the deprecated entry-count cache.
        return lance.dataset(uri, index_cache_size=max(cache_bytes // (20 * 1024 * 1024), 1))


def _index_entries(ds) -> list[Any]:
    if hasattr(ds, "describe_indices"):
        try:
            return list(ds.describe_indices())
        except Exception:
            pass
    return list(ds.list_indices())


def _index_name(entry: Any) -> str:
    if hasattr(entry, "name"):
        return entry.name
    if isinstance(entry, dict):
        return entry["name"]
    raise TypeError(f"unknown index entry type: {type(entry)!r}")


def _find_index_name(ds, preferred: str | None = None) -> str:
    names = [_index_name(entry) for entry in _index_entries(ds)]
    if not names:
        raise RuntimeError("dataset has no indexes")
    if preferred is not None:
        if preferred in names:
            return preferred
        raise RuntimeError(f"index {preferred!r} not found; available: {names}")
    return names[0]


def _open_index_metadata(ds, index_name: str) -> None:
    """Load IVF metadata without prewarming partitions."""
    _index_entries(ds)
    stats = ds.stats
    if hasattr(stats, "index_stats"):
        stats.index_stats(index_name)
    elif hasattr(ds, "index_statistics"):
        ds.index_statistics(index_name)


def _prewarm(ds, index_name: str) -> None:
    ds.prewarm_index(index_name)


def _search(ds, query, k: int = TOP_K, nprobes: int = NPROBES):
    nearest = {
        "column": VECTOR_COLUMN,
        "q": query,
        "k": k,
        "nprobes": nprobes,
        "metric": METRIC,
    }
    # Prefer an empty projection + system row id so search does not read
    # payload columns. Fall back to selecting ``_rowid`` by name.
    try:
        return ds.to_table(nearest=nearest, columns=[], with_row_id=True)
    except TypeError:
        return ds.to_table(nearest=nearest, columns=["_rowid"])
    except Exception:
        return ds.to_table(nearest=nearest, columns=["_rowid"])


def _query_vectors(lance, queries_uri: str, limit: int):
    import numpy as np

    qds = lance.dataset(queries_uri)
    table = qds.to_table(columns=["vector"], limit=limit)
    vectors = table.column("vector")
    out = []
    for i in range(len(vectors)):
        values = vectors[i].values
        out.append(np.asarray(values.to_numpy(), dtype="float32"))
    return out


def download(data_dir: Path) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download

    data_dir.mkdir(parents=True, exist_ok=True)
    root = Path(
        snapshot_download(
            HF_DATASET,
            repo_type="dataset",
            local_dir=str(data_dir / "hf"),
        )
    )
    corpus = root / "base.lance"
    queries = root / "queries.lance"
    if not corpus.exists() or not queries.exists():
        raise FileNotFoundError(f"expected base.lance and queries.lance under {root}")
    return corpus, queries


def work_slug(label: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)
    return slug.strip("_") or "shared"


def work_corpus_path(data_dir: Path, label: str) -> Path:
    return data_dir / f"work-{work_slug(label)}" / "base.lance"


def _clone_file(src_file: Path, dst_file: Path, *, hardlink: bool) -> None:
    if hardlink:
        try:
            os.link(src_file, dst_file)
            return
        except OSError:
            pass
    shutil.copy2(src_file, dst_file)


def clone_corpus(src: Path, dst: Path) -> Path:
    """Clone the snapshot corpus so indexing does not mutate the download.

    Fragment files under ``data/`` are hardlinked when possible. Manifests,
    versions, and other metadata are copied so a later commit cannot change
    the Hugging Face snapshot in place.
    """
    if dst.exists():
        try:
            ds = _import_lance().dataset(str(dst))
            if ds.count_rows() > 0:
                return dst
        except Exception:
            shutil.rmtree(dst, ignore_errors=True)
    print(f"cloning corpus {src} -> {dst}", flush=True)

    def copy_file(src_file: str, dst_file: str) -> None:
        rel = Path(src_file).relative_to(src)
        _clone_file(
            Path(src_file),
            Path(dst_file),
            hardlink=bool(rel.parts) and rel.parts[0] == "data",
        )

    shutil.copytree(src, dst, copy_function=copy_file)
    return dst


def prepare_work_corpus(src: Path, dst: Path) -> Path:
    return clone_corpus(src, dst)


def build_index(corpus_uri: str, num_bits: int, index_name: str) -> dict[str, Any]:
    lance = _import_lance()
    ds = lance.dataset(corpus_uri)
    print(
        f"building IVF_RQ{num_bits} name={index_name} partitions={NUM_PARTITIONS}",
        flush=True,
    )
    started = time.perf_counter()
    created = ds.create_index(
        VECTOR_COLUMN,
        index_type="IVF_RQ",
        metric=METRIC,
        num_partitions=NUM_PARTITIONS,
        num_bits=num_bits,
        name=index_name,
        replace=True,
    )
    if created is not None:
        ds = created
    elapsed = time.perf_counter() - started
    name = _find_index_name(ds, index_name)
    print(f"built {name} in {elapsed:.1f}s", flush=True)
    return {
        "index_name": name,
        "num_bits": num_bits,
        "build_seconds": elapsed,
    }


def advise_dontneed(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def index_storage_files(corpus_uri: str, index_name: str) -> list[Path]:
    lance = _import_lance()
    ds = lance.dataset(corpus_uri)
    files: list[Path] = []
    for entry in _index_entries(ds):
        if _index_name(entry) != index_name:
            continue
        uuid = entry["uuid"] if isinstance(entry, dict) else getattr(entry, "uuid", None)
        if uuid is None:
            continue
        root = Path(corpus_uri) / "_indices" / str(uuid)
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return files


def drop_os_page_cache(paths: list[Path] | None = None) -> None:
    """Evict process caches and the kernel page cache.

    Requires passwordless ``sudo`` for ``/proc/sys/vm/drop_caches``. Also
    ``posix_fadvise(DONTNEED)`` the given files so a specific index is dropped
    even if another process refaults pages immediately.
    """
    gc.collect()
    if paths:
        for path in paths:
            if path.is_file():
                advise_dontneed(path)
    subprocess.check_call(["sync"])
    subprocess.check_call(
        ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
    )


def split_discard(
    latencies: list[float], discard_first: int
) -> tuple[list[float], list[float]]:
    if discard_first < 0:
        raise ValueError(f"discard_first must be >= 0, got {discard_first}")
    if discard_first >= len(latencies):
        raise ValueError(
            f"discard_first={discard_first} leaves no timed queries "
            f"(count={len(latencies)})"
        )
    return latencies[:discard_first], latencies[discard_first:]


def _summarize(latencies: list[float]) -> dict[str, float]:
    ordered = sorted(latencies)
    return {
        "count": len(ordered),
        "mean_ms": statistics.fmean(ordered) * 1000.0,
        "median_ms": statistics.median(ordered) * 1000.0,
        "p95_ms": ordered[max(int(len(ordered) * 0.95) - 1, 0)] * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
        "qps": len(ordered) / sum(ordered),
    }


def _time_queries(search: Callable[[Any], Any], queries) -> list[float]:
    latencies = []
    for i, query in enumerate(queries):
        started = time.perf_counter()
        table = search(query)
        elapsed = time.perf_counter() - started
        if table.num_rows <= 0:
            raise RuntimeError(f"query {i} returned no rows")
        latencies.append(elapsed)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  query {i + 1}/{len(queries)}  {elapsed * 1000:.2f} ms  "
                f"rows={table.num_rows}",
                flush=True,
            )
    return latencies


def _mode_payload(
    mode: str,
    latencies: list[float],
    discard_first: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    discarded, kept = split_discard(latencies, discard_first)
    payload = {
        "mode": mode,
        "discard_first": discard_first,
        "discarded_ms": [latency * 1000.0 for latency in discarded],
        "summary": _summarize(kept),
        "latencies_ms": [latency * 1000.0 for latency in kept],
    }
    if extra:
        payload.update(extra)
    return payload


def bench_warm(
    corpus_uri: str, index_name: str, queries, discard_first: int
) -> dict[str, Any]:
    lance = _import_lance()
    ds = _dataset(lance, corpus_uri)
    _open_index_metadata(ds, index_name)
    print(f"prewarming {index_name}", flush=True)
    started = time.perf_counter()
    _prewarm(ds, index_name)
    prewarm_seconds = time.perf_counter() - started
    print(f"prewarm finished in {prewarm_seconds:.1f}s", flush=True)

    def search(query):
        return _search(ds, query)

    latencies = _time_queries(search, queries)
    return _mode_payload(
        "warm",
        latencies,
        discard_first,
        extra={"prewarm_seconds": prewarm_seconds},
    )


def bench_cold(
    corpus_uri: str, index_name: str, queries, discard_first: int
) -> dict[str, Any]:
    lance = _import_lance()
    latencies: list[float] = []
    for i, query in enumerate(queries):
        # Fresh handle so previously loaded partitions are not reused. Open
        # index metadata before the timer so "cold" means partitions, not
        # first-time index open.
        ds = _dataset(lance, corpus_uri)
        _open_index_metadata(ds, index_name)
        started = time.perf_counter()
        table = _search(ds, query)
        elapsed = time.perf_counter() - started
        if table.num_rows <= 0:
            raise RuntimeError(f"query {i} returned no rows")
        latencies.append(elapsed)
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  query {i + 1}/{len(queries)}  {elapsed * 1000:.2f} ms  "
                f"rows={table.num_rows}",
                flush=True,
            )
    return _mode_payload("cold", latencies, discard_first)


def _pylance_info() -> dict[str, Any]:
    import lance

    return {
        "pylance_version": getattr(lance, "__version__", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


def run_one(
    *,
    data_dir: Path,
    label: str,
    num_bits: int,
    query_count: int,
    skip_index: bool,
    drop_caches: bool,
    discard_first: int,
) -> dict[str, Any]:
    lance = _import_lance()
    hf_root = data_dir / "hf"
    src = hf_root / "base.lance"
    queries_uri = str(hf_root / "queries.lance")
    if not src.exists():
        download(data_dir)
    corpus = prepare_work_corpus(src, work_corpus_path(data_dir, label))
    corpus_uri = str(corpus)
    index_name = f"ivf_rq{num_bits}"

    ds = _dataset(lance, corpus_uri)
    have_index = False
    try:
        have_index = _find_index_name(ds, index_name) == index_name
    except Exception:
        have_index = False

    build = None
    if not skip_index or not have_index:
        build = build_index(corpus_uri, num_bits, index_name)
        index_name = build["index_name"]
    else:
        index_name = _find_index_name(ds, index_name)

    queries = _query_vectors(lance, queries_uri, query_count + discard_first)
    protocol = {
        "drop_os_page_cache": drop_caches,
        "discard_first": discard_first,
        "timed_queries": query_count,
        "cold": (
            "drop caches, then fresh dataset per query; metadata opened "
            "untimed; first query discarded so the timed set is lance-cold "
            "with this index already in the OS page cache"
        ),
        "warm": "prewarm_index on one handle; first query discarded",
    }
    if drop_caches:
        files = index_storage_files(corpus_uri, index_name)
        print(f"dropping OS page cache for {len(files)} index files", flush=True)
        drop_os_page_cache(files)
    print(
        f"timing {label} IVF_RQ{num_bits}  timed={query_count}  "
        f"discard_first={discard_first}  k={TOP_K} nprobes={NPROBES}",
        flush=True,
    )
    cold = bench_cold(corpus_uri, index_name, queries, discard_first)
    warm = bench_warm(corpus_uri, index_name, queries, discard_first)
    runtime = _pylance_info()
    return {
        "label": label,
        "index": f"IVF_RQ{num_bits}",
        "num_bits": num_bits,
        "num_partitions": NUM_PARTITIONS,
        "nprobes": NPROBES,
        "k": TOP_K,
        "metric": METRIC,
        "query_count": query_count,
        "dataset": HF_DATASET,
        "corpus_uri": corpus_uri,
        "index_writer": runtime["pylance_version"],
        "protocol": protocol,
        "build": build,
        "cold": cold,
        "warm": warm,
        "runtime": runtime,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download", help="Download the Hugging Face snapshot")
    p_dl.add_argument("--data-dir", type=Path, required=True)

    p_run = sub.add_parser("run", help="Build one IVF_RQ index and time warm/cold search")
    p_run.add_argument("--data-dir", type=Path, required=True)
    p_run.add_argument("--label", required=True)
    p_run.add_argument("--bits", type=int, required=True, choices=(1, 5))
    p_run.add_argument("--out", type=Path, required=True)
    p_run.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    p_run.add_argument("--skip-index", action="store_true")
    p_run.add_argument(
        "--drop-caches",
        action="store_true",
        help="sync + drop OS page cache before timing this cell",
    )
    p_run.add_argument(
        "--discard-first",
        type=int,
        default=DEFAULT_DISCARD_FIRST,
        help="drop the first N queries of each mode from the summary",
    )

    args = parser.parse_args()
    if args.cmd == "download":
        corpus, queries = download(args.data_dir)
        print(f"corpus={corpus}")
        print(f"queries={queries}")
        return

    result = run_one(
        data_dir=args.data_dir,
        label=args.label,
        num_bits=args.bits,
        query_count=args.query_count,
        skip_index=args.skip_index,
        drop_caches=args.drop_caches,
        discard_first=args.discard_first,
    )
    _write_json(args.out, result)
    print(json.dumps({
        "label": result["label"],
        "index": result["index"],
        "cold": result["cold"]["summary"],
        "warm": result["warm"]["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
