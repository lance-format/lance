# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Independently recompute held-out results from raw per-query records."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import lance
import numpy as np
from common import DATASETS, EVALUATION_TARGET, MEASURED_K, matrix, save, summary
from measure import fixed_budgets


def audit(root, names):
    candidate = (root / "candidate-binary.sha256").read_text().split()[0]
    baseline = (root / "baseline-binary.sha256").read_text().split()[0]
    assert candidate != baseline, "The two arms must use their own compiled binaries"
    calibration = json.loads((root / "calibration.json").read_text())
    calibration_sha = hashlib.sha256(
        (root / "calibration.json").read_bytes()
    ).hexdigest()
    report = {"datasets": {}, "auto_meets_target": {}, "baseline_comparisons": {}}
    discrepancies = []
    for name in names:
        out = root / name
        identity = json.loads((out / "identity-native.json").read_text())
        assert identity["binary_sha256"] == candidate, name
        assert identity["baseline_binary_sha256"] == baseline, name
        assert identity["split"] == "evaluation", name
        assert identity["harness_sha256"] == {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__).with_name("measure.py"),
                Path(__file__).with_name("common.py"),
            ]
        }, name
        assert identity["calibration_sha256"] == calibration_sha, name
        assert identity["k"] == MEASURED_K, (name, identity["k"])
        assert (
            identity["prepared_sha256"]
            == hashlib.sha256((out / "prepared.json").read_bytes()).hexdigest()
        ), name
        assert identity["filtered_queries"] > 0, name
        split = np.load(out / "split.npz")
        assert (
            identity["split_sha256"]
            == hashlib.sha256((out / "split.npz").read_bytes()).hexdigest()
        ), name
        vectors = matrix(
            lance.dataset(root / "data" / name / "queries.lance").to_table()["vector"]
        )
        vector_keys = (
            np.ascontiguousarray(vectors)
            .view(np.dtype((np.void, vectors.dtype.itemsize * vectors.shape[1])))
            .ravel()
        )
        _, vector_groups = np.unique(vector_keys, return_inverse=True)
        assert not np.intersect1d(
            vector_groups[split["calibration"]], vector_groups[split["evaluation"]]
        ).size, name
        assert len(np.unique(vector_groups[split["evaluation"]])) == len(
            split["evaluation"]
        ), name
        ids = split["evaluation"][: identity["query_count"]]
        assert not set(split["evaluation"].tolist()) & set(
            split["calibration"].tolist()
        )
        truth = np.load(out / "truth.npz")["ids"]
        filtered = np.load(out / "filtered-truth.npz")
        with (out / "measure-native.csv").open() as stream:
            records = list(csv.DictReader(stream))
        baseline_identity = json.loads(
            (out / "identity-baseline-audit.json").read_text()
        )
        assert baseline_identity["binary_sha256"] == baseline, name
        with (out / "measure-baseline-audit.csv").open() as stream:
            baseline_rows = list(csv.DictReader(stream))
        for k in MEASURED_K:
            baseline_auto = {
                int(r["query_id"]): r
                for r in baseline_rows
                if int(r["k"]) == k and r["policy"] == "auto"
            }
            assert set(baseline_auto) == set(
                ids[: baseline_identity["query_count"]].tolist()
            ), (name, k, "baseline query coverage")
            comparison = "auto" if k <= 100 else "legacy"
            for row in records:
                query_id = int(row["query_id"])
                if (
                    int(row["k"]) == k
                    and row["policy"] == comparison
                    and row["filtered"] == "False"
                    and query_id in baseline_auto
                ):
                    reference = baseline_auto[query_id]
                    for field in [
                        "neighbor_ids",
                        "returned",
                        "partitions",
                        "comparisons",
                    ]:
                        assert row[field] == reference[field], (
                            name,
                            k,
                            query_id,
                            field,
                        )
        report["datasets"][name] = {}
        report["baseline_comparisons"][name] = {}
        groups = sorted(
            {(int(r["k"]), r["policy"], r["filtered"] == "True") for r in records}
        )
        expected_groups = {
            (k, policy, False)
            for k in MEASURED_K
            for policy in [
                "auto",
                *[f"fixed{b}" for b in fixed_budgets(calibration, name, k)],
                *(["tuned", "legacy"] if k > 100 else []),
            ]
        } | {
            (k, policy, True)
            for k in [101, 1000]
            for policy in ["auto", "legacy", "fixed20"]
        }
        assert set(groups) == expected_groups, (name, set(groups) ^ expected_groups)
        for k, policy, is_filtered in groups:
            rows = [
                r
                for r in records
                if int(r["k"]) == k
                and r["policy"] == policy
                and (r["filtered"] == "True") == is_filtered
                and r["phase"] != "warmup"
            ]
            query_ids = [int(r["query_id"]) for r in rows]
            assert len(query_ids) == len(set(query_ids)), (name, k, policy)
            if not is_filtered:
                expected = (
                    ids[: identity["legacy_queries"]] if policy == "legacy" else ids
                )
                assert set(query_ids) == set(expected.tolist()), (name, k, policy)
            else:
                assert set(query_ids) == set(
                    ids[: identity["filtered_queries"]].tolist()
                ), (name, k, policy, "filtered query coverage")
            reference = filtered["ids"] if is_filtered else truth
            recalls = []
            for row, query_id in zip(rows, query_ids):
                neighbors = json.loads(row["neighbor_ids"])
                assert len(neighbors) == len(set(neighbors)) == int(row["returned"])
                assert np.isfinite(float(row["latency_ms"]))
                assert float(row["latency_ms"]) > 0
                if not is_filtered:
                    assert len(neighbors) <= k
                    if not policy.startswith("fixed"):
                        assert len(neighbors) == k
                    assert int(row["bytes_read"]) == 0, (name, k, policy, query_id)
                else:
                    assert set(neighbors) <= set(filtered["positions"].tolist())
                    if policy in ("auto", "legacy"):
                        assert len(neighbors) == k, (name, k, policy, query_id)
                recall = len(set(neighbors) & set(reference[query_id, :k].tolist())) / k
                assert abs(recall - float(row["recall"])) < 1e-12
                recalls.append(recall)
                if not is_filtered and (
                    int(row["partitions"]) != int(row["predicted_partitions"])
                    or int(row["comparisons"]) != int(row["predicted_rows"])
                ):
                    discrepancies.append({"dataset": name, **row})
            timed = [r for r in rows if r["phase"] == "timing"]
            if not is_filtered:
                timing_count = min(identity["timing_queries"], len(rows))
                assert [int(r["query_id"]) for r in timed] == ids[
                    :timing_count
                ].tolist()
            mean = float(np.mean(recalls))
            error = 1.96 * float(np.std(recalls, ddof=1)) / np.sqrt(len(recalls))
            key = f"{k}{'-filtered' if is_filtered else ''}"
            report["datasets"][name].setdefault(key, {})[policy] = {
                "queries": len(rows),
                "timing_queries": len(timed),
                "recall": mean,
                "recall_ci95_normal": [max(0.0, mean - error), min(1.0, mean + error)],
                "returned": summary([int(r["returned"]) for r in rows]),
                "latency_ms": summary([float(r["latency_ms"]) for r in timed])
                if timed
                else None,
                "partitions": summary([int(r["partitions"]) for r in rows]),
                "scanned_rows": summary([int(r["comparisons"]) for r in rows]),
            }
            if policy == "auto" and not is_filtered:
                report["auto_meets_target"][f"{name}/{k}"] = mean >= EVALUATION_TARGET
        # Compare actual binaries on exactly the interleaved query subset. The
        # full Auto timing population must not be compared to 32 baseline rows.
        for k in MEASURED_K:
            if k <= 100:
                continue
            paired_ids = set(ids[: identity["legacy_queries"]].tolist())
            paired = {
                policy: [
                    row
                    for row in records
                    if row["filtered"] == "False"
                    and row["phase"] == "timing"
                    and int(row["k"]) == k
                    and row["policy"] == policy
                    and int(row["query_id"]) in paired_ids
                ]
                for policy in ["auto", "legacy"]
            }
            assert all(len(rows) == len(paired_ids) for rows in paired.values())
            report["baseline_comparisons"][name][str(k)] = {
                policy: {
                    "queries": len(rows),
                    "recall": float(np.mean([float(r["recall"]) for r in rows])),
                    "latency_ms": summary([float(r["latency_ms"]) for r in rows]),
                    "partitions": summary([int(r["partitions"]) for r in rows]),
                    "scanned_rows": summary([int(r["comparisons"]) for r in rows]),
                }
                for policy, rows in paired.items()
            }
    report["prediction_discrepancies"] = len(discrepancies)
    save(root / "prediction-discrepancies.json", discrepancies)
    save(root / "audited-results.json", report)
    print(json.dumps(report["auto_meets_target"], indent=2))
    assert all(report["auto_meets_target"].values()), report["auto_meets_target"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("datasets", nargs="*", default=sorted(DATASETS))
    args = parser.parse_args()
    audit(args.root, args.datasets)
