# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Independently audit returned IDs from the constant-fallback experiment."""

import argparse
import csv
import json
from pathlib import Path

import lance
import numpy as np
from common import DATASETS, matrix, save, summary
from measure_fallback import NATIVE_K
from prepare_large import sha256


def audit(root, names, label="", smoke=False):
    frozen = root / "calibration-frozen.json"
    assert sha256(frozen) == sha256(root / "calibration.json")
    calibration = json.loads(frozen.read_text())
    assert calibration["calibration_all_k"] is True
    assert calibration["calibration_k_range"] == [1, 100000]
    source = Path(calibration["source"])
    candidate_sha = (root / "candidate-binary.sha256").read_text().split()[0]
    baseline_sha = (root / "baseline-binary.sha256").read_text().split()[0]
    assert candidate_sha != baseline_sha
    report = {
        "smoke": smoke,
        "calibration_sha256": sha256(frozen),
        "candidate_binary_sha256": candidate_sha,
        "baseline_binary_sha256": baseline_sha,
        "datasets": {},
        "paired_baseline": {},
        "meets_target": {},
        "prediction_discrepancies": [],
    }
    filename = "audited-results" + (f"-{label}" if label else "") + ".json"
    for name in names:
        directory = source / name
        native = root / name / ("native" + (f"-{label}" if label else ""))
        identity = json.loads((native / "identity.json").read_text())
        assert identity["dataset"] == name
        assert identity["candidate_binary_sha256"] == candidate_sha
        assert identity["baseline_binary_sha256"] == baseline_sha
        assert identity["calibration_sha256"] == sha256(frozen)
        assert identity["prepared_sha256"] == sha256(directory / "prepared.json")
        assert identity["prepared_sha256"] == calibration["prepared_sha256"][name]
        assert identity["split_sha256"] == sha256(directory / "split.npz")
        assert identity["split_sha256"] == calibration["split_sha256"][name]
        assert identity["build_sha256"] == sha256(directory / "build.json")
        assert identity["split"] == ("calibration" if smoke else "evaluation")
        if not smoke:
            assert identity["k"] == NATIVE_K
            assert identity["query_count"] == 512
            assert identity["timing_queries"] == 128
            assert identity["baseline_queries"] == 32
        for path, expected in identity["harness_sha256"].items():
            assert sha256(Path(__file__).with_name(path)) == expected, path
        for path, expected in identity["files_sha256"].items():
            assert sha256(native / path) == expected, path
        prepared = json.loads((directory / "prepared.json").read_text())
        assert prepared["build_sha256"] == sha256(directory / "build.json")
        assert prepared["split_sha256"] == sha256(directory / "split.npz")
        for path, expected in prepared["sha256"].items():
            assert sha256(directory / path) == expected, path
        split = np.load(directory / "split.npz")
        assert not np.intersect1d(split["calibration"], split["evaluation"]).size
        ids = split[identity["split"]][: identity["query_count"]]
        queries = matrix(
            lance.dataset(source / "data" / name / "queries.lance").to_table()["vector"]
        )
        keys = (
            np.ascontiguousarray(queries)
            .view(np.dtype((np.void, queries.dtype.itemsize * queries.shape[1])))
            .ravel()
        )
        _, groups = np.unique(keys, return_inverse=True)
        assert not np.intersect1d(
            groups[split["calibration"]], groups[split["evaluation"]]
        ).size
        assert len(np.unique(groups[ids])) == len(ids)
        truth = np.load(directory / "truth-ids.npy", mmap_mode="r")
        build = json.loads((directory / "build.json").read_text())
        report["datasets"][name] = {}
        report["paired_baseline"][name] = {}
        for k in identity["k"]:
            with (native / f"records-{k}.csv").open() as stream:
                records = list(csv.DictReader(stream))
            assert {row["policy"] for row in records} == {"candidate", "baseline"}
            report["datasets"][name][str(k)] = {}
            for policy in ["candidate", "baseline"]:
                rows = [row for row in records if row["policy"] == policy]
                count = (
                    len(ids)
                    if policy == "candidate"
                    else min(len(ids), identity["baseline_queries"])
                )
                assert [int(row["ordinal"]) for row in rows] == list(range(count))
                assert [int(row["query_id"]) for row in rows] == ids[:count].tolist()
                neighbors = np.load(native / f"ids-{k}-{policy}.npy", mmap_mode="r")
                assert neighbors.shape == (count, k)
                recalls = []
                for ordinal, row in enumerate(rows):
                    found = neighbors[ordinal]
                    assert int(row["k"]) == k
                    assert int(row["returned"]) == k
                    assert len(np.unique(found)) == k
                    assert np.all((found >= 0) & (found < build["rows"]))
                    exact = truth[int(row["query_id"]), :k]
                    recall = (
                        np.count_nonzero(
                            np.isin(found, exact, assume_unique=True, kind="sort")
                        )
                        / k
                    )
                    assert abs(recall - float(row["recall"])) < 1e-12
                    recalls.append(float(recall))
                    assert int(row["bytes_read"]) == 0
                    assert 1 <= int(row["partitions"]) <= build["partitions"]
                    assert k <= int(row["comparisons"]) <= build["rows"]
                    assert (
                        np.isfinite(float(row["latency_ms"]))
                        and float(row["latency_ms"]) > 0
                    )
                    if int(row["partitions"]) != int(
                        row["predicted_final_partitions"]
                    ) or int(row["comparisons"]) != int(row["predicted_final_rows"]):
                        report["prediction_discrepancies"].append(
                            {"dataset": name, **row}
                        )
                timed = [row for row in rows if row["phase"] == "timing"]
                assert [int(row["query_id"]) for row in timed] == ids[
                    : min(count, identity["timing_queries"])
                ].tolist()
                assert all(row["phase"] == "recall" for row in rows[len(timed) :])
                mean_recall = float(np.mean(recalls))
                report["datasets"][name][str(k)][policy] = {
                    "queries": count,
                    "timing_queries": len(timed),
                    "recall": mean_recall,
                    "recall_distribution": {"min": min(recalls), **summary(recalls)},
                    "latency_ms": summary([float(row["latency_ms"]) for row in timed]),
                    "partitions": summary([int(row["partitions"]) for row in rows]),
                    "scanned_rows": summary([int(row["comparisons"]) for row in rows]),
                    "returned": summary([int(row["returned"]) for row in rows]),
                    "max_absolute_routing_recall_difference": max(
                        abs(float(row["recall"]) - float(row["route_recall"]))
                        for row in rows
                    ),
                }
                if policy == "candidate":
                    report["meets_target"][f"{name}/{k}"] = mean_recall >= 0.95
                del neighbors
            report["paired_baseline"][name][str(k)] = {
                policy: {
                    "queries": len(rows),
                    "recall": float(np.mean([float(row["recall"]) for row in rows])),
                    "latency_ms": summary([float(row["latency_ms"]) for row in rows]),
                    "partitions": summary([int(row["partitions"]) for row in rows]),
                    "scanned_rows": summary([int(row["comparisons"]) for row in rows]),
                }
                for policy in ["candidate", "baseline"]
                for rows in [
                    [
                        row
                        for row in records
                        if row["policy"] == policy
                        and int(row["ordinal"]) < identity["baseline_queries"]
                    ]
                ]
            }
            save(root / filename, report)
    save(root / filename, report)
    if not smoke:
        assert all(report["meets_target"].values()), report["meets_target"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS)
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert not args.smoke or args.label
    audit(args.root, args.datasets, args.label, args.smoke)
