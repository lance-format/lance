# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Independently audit native per-query output against the published ground truth."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import lance
import numpy as np
from calibrate import NAMES, summary
from prepare import matrix, save


def audit(root):
    calibration = json.loads((root / "calibration.json").read_text())
    report = {
        "profiles": calibration["selected"],
        "datasets": {},
        "query_integrity": {},
    }
    discrepancies = []
    for name in NAMES:
        directory = root / name
        identity = json.loads((directory / "identity-native.json").read_text())
        assert (
            identity["binary_sha256"]
            == (root / "candidate-binary.sha256").read_text().split()[0]
        )
        assert (
            identity["calibration_sha256"]
            == hashlib.sha256((root / "calibration.json").read_bytes()).hexdigest()
        )
        split = np.load(directory / "split.npz")
        ids = split["evaluation"]
        assert not set(ids) & set(split["calibration"])
        queries = matrix(
            lance.dataset(root / "data" / name / "queries.lance").to_table()["vector"]
        )
        hashes = [hashlib.sha256(row.tobytes()).digest() for row in queries]
        report["query_integrity"][name] = {
            "disjoint_query_ids": True,
            "shared_vector_hashes": len(
                {hashes[i] for i in ids} & {hashes[i] for i in split["calibration"]}
            ),
        }
        truth = matrix(
            lance.dataset(root / "data" / name / "ground_truth.lance").to_table()[
                "neighbor_ids"
            ]
        )
        build = json.loads((directory / "build.json").read_text())
        assert truth.shape[1] == 100
        assert int(truth.max()) < build["rows"]
        assert all(len(set(row.tolist())) == 100 for row in truth)
        with (directory / "measure-native.csv").open() as stream:
            records = list(csv.DictReader(stream))
        report["datasets"][name] = {}
        for k in [1, 10, 100]:
            result = {}
            groups = sorted({row["policy"] for row in records if int(row["k"]) == k})
            fixed = calibration["fixed"][name][str(k)]
            meeting = next(i for i, row in enumerate(fixed) if row["recall"] >= 0.95)
            expected_policies = {"auto", "tuned", "legacy"} | {
                f"fixed{row['nprobes']}"
                for row in fixed[max(0, meeting - 1) : meeting + 2]
            }
            assert set(groups) == expected_policies
            for policy in groups:
                rows = [
                    row
                    for row in records
                    if int(row["k"]) == k and row["policy"] == policy
                ]
                query_ids = [int(row["query_id"]) for row in rows]
                expected_ids = (
                    ids[: identity["legacy_queries"]] if policy == "legacy" else ids
                )
                assert len(query_ids) == len(set(query_ids)) == len(expected_ids)
                assert set(query_ids) == set(expected_ids)
                recalls = []
                for row, query_id in zip(rows, query_ids):
                    neighbors = json.loads(row["neighbor_ids"])
                    assert len(neighbors) == len(set(neighbors)) == k
                    recall = len(set(neighbors) & set(truth[query_id, :k].tolist())) / k
                    assert abs(recall - float(row["recall"])) < 1e-12
                    assert int(row["bytes_read"]) == 0, (
                        name,
                        k,
                        policy,
                        query_id,
                        row["bytes_read"],
                    )
                    recalls.append(recall)
                    if (
                        int(row["partitions"]) != int(row["predicted_partitions"])
                        or int(row["comparisons"]) != int(row["predicted_rows"])
                        or abs(recall - float(row["route_recall"])) > 1e-12
                    ):
                        discrepancies.append({"dataset": name, **row})
                timed = [row for row in rows if row["phase"] == "timing"]
                expected_timed = min(identity["timing_queries"], len(expected_ids))
                assert len(timed) == expected_timed
                mean = float(np.mean(recalls))
                error = 1.96 * float(np.std(recalls, ddof=1)) / np.sqrt(len(recalls))
                result[policy] = {
                    "queries": len(rows),
                    "timing_queries": len(timed),
                    "recall": mean,
                    "recall_ci95_normal": [
                        max(0.0, mean - error),
                        min(1.0, mean + error),
                    ],
                    "meets_recall95": mean >= 0.95,
                    "latency_ms": summary([float(row["latency_ms"]) for row in timed]),
                    "partitions": summary([int(row["partitions"]) for row in rows]),
                    "scanned_rows": summary([int(row["comparisons"]) for row in rows]),
                    "bytes_read": 0,
                }
            report["datasets"][name][str(k)] = result
    report["prediction_discrepancies"] = len(discrepancies)
    report["common_profile_passed_all_recall_targets"] = all(
        result["auto"]["meets_recall95"]
        for dataset in report["datasets"].values()
        for result in dataset.values()
    )
    save(root / "prediction-discrepancies.json", discrepancies)
    save(root / "audited-results.json", report)
    print(json.dumps(report, indent=2))
    assert report["common_profile_passed_all_recall_targets"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    audit(args.root)
