# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Independently audit binary neighbors, exact recall, coverage and large-k timings."""

import argparse
import csv
import json
from pathlib import Path

import lance
import numpy as np
from common import DATASETS, EVALUATION_TARGET, matrix, save, summary
from measure import fixed_budgets
from prepare_large import K_VALUES, sha256


def audit(root, names):
    calibration = json.loads((root / "calibration-frozen.json").read_text())
    calibration_sha = sha256(root / "calibration-frozen.json")
    assert sha256(root / "calibration.json") == calibration_sha
    candidate_sha = (root / "candidate-binary.sha256").read_text().split()[0]
    baseline_sha = (root / "baseline-binary.sha256").read_text().split()[0]
    assert candidate_sha != baseline_sha
    report = {
        "datasets": {},
        "paired_baseline": {},
        "auto_meets_target": {},
        "prediction_discrepancies": [],
    }
    for name in names:
        out = root / name
        result = out / "native"
        identity = json.loads((result / "identity.json").read_text())
        assert identity["candidate_binary_sha256"] == candidate_sha
        assert identity["baseline_binary_sha256"] == baseline_sha
        assert identity["calibration_sha256"] == calibration_sha
        assert identity["prepared_sha256"] == sha256(out / "prepared.json")
        assert identity["split_sha256"] == sha256(out / "split.npz")
        assert identity["split"] == "evaluation" and identity["k"] == K_VALUES
        for filename, expected in identity["harness_sha256"].items():
            assert sha256(Path(__file__).with_name(filename)) == expected, filename
        for filename, expected in identity["files_sha256"].items():
            assert sha256(result / filename) == expected, filename
        prepared = json.loads((out / "prepared.json").read_text())
        for filename, expected in prepared["sha256"].items():
            assert sha256(out / filename) == expected, filename
        assert calibration["prepared_sha256"][name] == sha256(out / "prepared.json")
        split = np.load(out / "split.npz")
        ids = split["evaluation"]
        assert identity["query_count"] == len(ids)
        assert not np.intersect1d(split["calibration"], ids).size
        queries = matrix(
            lance.dataset(root / "data" / name / "queries.lance").to_table()["vector"]
        )
        keys = (
            np.ascontiguousarray(queries)
            .view(np.dtype((np.void, queries.dtype.itemsize * queries.shape[1])))
            .ravel()
        )
        _, groups = np.unique(keys, return_inverse=True)
        assert not np.intersect1d(groups[split["calibration"]], groups[ids]).size
        assert len(np.unique(groups[ids])) == len(ids)
        truth = np.load(out / "truth-ids.npy", mmap_mode="r")
        build = json.loads((out / "build.json").read_text())
        report["datasets"][name] = {}
        report["paired_baseline"][name] = {}
        for k in K_VALUES:
            with (result / f"records-{k}.csv").open() as stream:
                records = list(csv.DictReader(stream))
            policies = [
                "auto",
                *[f"fixed{b}" for b in fixed_budgets(calibration, name, k)],
                "tuned",
                "legacy",
            ]
            assert {r["policy"] for r in records} == set(policies)
            report["datasets"][name][str(k)] = {}
            for policy in policies:
                rows = [r for r in records if r["policy"] == policy]
                count = (
                    min(len(ids), identity["legacy_queries"])
                    if policy == "legacy"
                    else len(ids)
                )
                assert [int(r["ordinal"]) for r in rows] == list(range(count))
                assert [int(r["query_id"]) for r in rows] == ids[:count].tolist()
                neighbors = np.load(result / f"ids-{k}-{policy}.npy", mmap_mode="r")
                assert neighbors.shape == (count, k)
                recall = []
                for ordinal, row in enumerate(rows):
                    returned = int(row["returned"])
                    found = neighbors[ordinal, :returned]
                    assert 0 <= returned <= k
                    assert np.all(neighbors[ordinal, returned:] == -1)
                    assert np.all((found >= 0) & (found < build["rows"]))
                    assert len(np.unique(found)) == returned
                    if not policy.startswith("fixed"):
                        assert returned == k
                    exact = truth[int(row["query_id"]), :k]
                    actual = (
                        np.count_nonzero(
                            np.isin(found, exact, assume_unique=True, kind="sort")
                        )
                        / k
                    )
                    assert abs(actual - float(row["recall"])) < 1e-12
                    recall.append(float(actual))
                    assert int(row["k"]) == k
                    assert int(row["bytes_read"]) == 0, (name, k, policy, ordinal)
                    assert 1 <= int(row["partitions"]) <= build["partitions"]
                    assert returned <= int(row["comparisons"]) <= build["rows"]
                    assert (
                        np.isfinite(float(row["latency_ms"]))
                        and float(row["latency_ms"]) > 0
                    )
                    if int(row["partitions"]) != int(
                        row["predicted_initial_partitions"]
                    ) or int(row["comparisons"]) != int(row["predicted_initial_rows"]):
                        report["prediction_discrepancies"].append(
                            {"dataset": name, **row}
                        )
                timed = [r for r in rows if r["phase"] == "timing"]
                assert [int(r["query_id"]) for r in timed] == ids[
                    : min(count, identity["timing_queries"])
                ].tolist()
                assert all(r["phase"] == "recall" for r in rows[len(timed) :])
                mean_recall = float(np.mean(recall))
                report["datasets"][name][str(k)][policy] = {
                    "queries": count,
                    "timing_queries": len(timed),
                    "recall": mean_recall,
                    "returned": summary([int(r["returned"]) for r in rows]),
                    "latency_ms": summary([float(r["latency_ms"]) for r in timed]),
                    "partitions": summary([int(r["partitions"]) for r in rows]),
                    "scanned_rows": summary([int(r["comparisons"]) for r in rows]),
                }
                if policy == "auto":
                    report["auto_meets_target"][f"{name}/{k}"] = (
                        mean_recall >= EVALUATION_TARGET
                    )
                del neighbors
            report["paired_baseline"][name][str(k)] = {
                policy: {
                    "queries": len(rows),
                    "recall": float(np.mean([float(r["recall"]) for r in rows])),
                    "latency_ms": summary([float(r["latency_ms"]) for r in rows]),
                    "partitions": summary([int(r["partitions"]) for r in rows]),
                    "scanned_rows": summary([int(r["comparisons"]) for r in rows]),
                }
                for policy in ["auto", "legacy"]
                for rows in [
                    [
                        r
                        for r in records
                        if r["policy"] == policy
                        and int(r["ordinal"]) < identity["legacy_queries"]
                    ]
                ]
            }
            save(root / "audited-results.json", report)
    save(root / "audited-results.json", report)
    assert all(report["auto_meets_target"].values()), report["auto_meets_target"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASETS), default=list(DATASETS)
    )
    args = parser.parse_args()
    audit(args.root, args.datasets)
