# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Unit tests for the VIBE IVF_RQ latency chart helpers."""

from __future__ import annotations

import json
from pathlib import Path

from bench import (
    advise_dontneed,
    clone_corpus,
    prime_os_page_cache,
    split_discard,
    work_corpus_path,
)
from plot import axis_label, load_results, plot_results
from run_versions import bench_run_cmd, latest_label


def _sample_result(label: str, num_bits: int, warm_mean: float, cold_mean: float) -> dict:
    return {
        "label": label,
        "index": f"IVF_RQ{num_bits}",
        "num_bits": num_bits,
        "warm": {
            "mode": "warm",
            "summary": {
                "count": 3,
                "mean_ms": warm_mean,
                "median_ms": warm_mean - 0.2,
                "p95_ms": warm_mean + 0.4,
                "min_ms": warm_mean - 0.5,
                "max_ms": warm_mean + 0.6,
                "qps": 1000.0 / warm_mean,
            },
        },
        "cold": {
            "mode": "cold",
            "summary": {
                "count": 3,
                "mean_ms": cold_mean,
                "median_ms": cold_mean - 0.3,
                "p95_ms": cold_mean + 0.8,
                "min_ms": cold_mean - 0.7,
                "max_ms": cold_mean + 1.0,
                "qps": 1000.0 / cold_mean,
            },
        },
    }


def test_load_and_plot(tmp_path: Path) -> None:
    rows = [
        _sample_result("v11.0.0", 1, 4.0, 12.0),
        _sample_result("v12.0.0", 1, 3.2, 10.5),
        _sample_result("c8f182179 (main)", 1, 2.8, 9.4),
        _sample_result("v11.0.0", 5, 6.5, 18.0),
        _sample_result("v12.0.0", 5, 5.1, 15.2),
        _sample_result("c8f182179 (main)", 5, 4.4, 13.0),
    ]
    for row in rows:
        name = f"{row['label'].replace(' ', '_')}-rq{row['num_bits']}.json"
        (tmp_path / name).write_text(json.dumps(row) + "\n")
    (tmp_path / "_manifest.json").write_text(
        json.dumps(
            {
                "results": [
                    str(tmp_path / f"{row['label'].replace(' ', '_')}-rq{row['num_bits']}.json")
                    for row in rows
                ]
            }
        )
        + "\n"
    )
    leftover = _sample_result("v11.0.0-shared-index", 1, 99.0, 99.0)
    (tmp_path / "leftover-shared-index.json").write_text(json.dumps(leftover) + "\n")

    loaded = load_results(tmp_path)
    assert [row["label"] for row in loaded if row["num_bits"] == 1] == [
        "v11.0.0",
        "v12.0.0",
        "c8f182179 (main)",
    ]
    assert all(row["label"] != "v11.0.0-shared-index" for row in loaded)

    out = tmp_path / "chart.png"
    plot_results(loaded, out, "unit test")
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_each_version_gets_its_own_work_corpus(tmp_path: Path) -> None:
    v11 = work_corpus_path(tmp_path, "v11.0.0")
    v12 = work_corpus_path(tmp_path, "v12.0.0")
    main = work_corpus_path(tmp_path, "c8f182179 (main)")
    assert v11 != v12 != main
    assert v11.parent.name == "work-v11.0.0"
    assert v12.parent.name == "work-v12.0.0"
    assert "c8f182179" in main.parent.name


def test_clone_hardlinks_data_and_copies_metadata(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "data").mkdir(parents=True)
    (src / "data" / "frag.lance").write_bytes(b"vector-bytes")
    (src / "_latest").write_text("v1\n")
    dst = tmp_path / "dst"
    clone_corpus(src, dst)

    src_data = src / "data" / "frag.lance"
    dst_data = dst / "data" / "frag.lance"
    assert dst_data.read_bytes() == b"vector-bytes"
    assert src_data.stat().st_ino == dst_data.stat().st_ino
    assert (src / "_latest").stat().st_ino != (dst / "_latest").stat().st_ino
    (dst / "_latest").write_text("v2\n")
    assert (src / "_latest").read_text() == "v1\n"


def test_version_matrix_always_builds_index(tmp_path: Path) -> None:
    cmd = bench_run_cmd(
        python=tmp_path / "python",
        data_dir=tmp_path / "data",
        label="v12.0.0",
        bits=5,
        query_count=100,
        out=tmp_path / "out.json",
    )
    assert "--skip-index" not in cmd
    assert "--drop-caches" in cmd
    assert cmd[cmd.index("--discard-first") + 1] == "1"
    assert cmd[cmd.index("--label") + 1] == "v12.0.0"
    assert cmd[cmd.index("--bits") + 1] == "5"


def test_split_discard_keeps_only_stable_queries() -> None:
    discarded, kept = split_discard([0.04, 0.01, 0.012, 0.011], 1)
    assert discarded == [0.04]
    assert kept == [0.01, 0.012, 0.011]


def test_advise_dontneed_on_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"x" * 4096)
    advise_dontneed(path)
    assert path.read_bytes()[:4] == b"xxxx"


def test_prime_os_page_cache_reads_whole_file(tmp_path: Path) -> None:
    path = tmp_path / "idx.bin"
    path.write_bytes(b"q" * (1024 * 64))
    prime_os_page_cache([path])
    assert path.stat().st_size == 1024 * 64


def test_latest_label_names_engine_revision_not_bench_commit() -> None:
    assert latest_label() == "c8f182179 (main)"


def test_axis_label_keeps_main_on_one_line() -> None:
    assert axis_label("c8f182179 (main)") == "main"
    assert axis_label("v12.0.0") == "v12.0.0"
