# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Independently recompute native recall, coverage and paired summaries."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS = ("binary", "cache", "index", "k", "refine", "filtered", "effort")


def audit(directory, query_file):
    identity = json.loads((directory / "identity.json").read_text())
    with query_file.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == identity["queries_sha256"]
        )
    with np.load(query_file) as arrays:
        queries = {name: arrays[name] for name in ("truth", "query_ids", "split")}
    groups = collections.defaultdict(list)
    defaults = collections.defaultdict(dict)
    seen = set()
    for line in (directory / "queries.jsonl").read_text().splitlines():
        row = json.loads(line)
        position, k = row["position"], row["k"]
        assert queries["split"][position] == 1
        assert row["query_id"] == int(queries["query_ids"][position])
        ids = np.asarray(row["ids"], dtype=np.uint64)
        assert len(ids) == k and len(np.unique(ids)) == k
        truth = queries["truth"][position]
        if row["filtered"]:
            assert np.all(ids % 10 == 0)
            truth = truth[truth % 10 == 0]
        assert len(truth) >= k
        recall = len(np.intersect1d(ids, truth[:k])) / k
        assert abs(row["recall"] - recall) < 1e-12
        key = tuple(row[field] for field in FIELDS)
        unique = (*key, row["query_id"], row["repeat"])
        assert unique not in seen
        seen.add(unique)
        groups[key].append(row)
        if row["effort"] == 0.5:
            defaults[(*key[1:], row["query_id"], row["repeat"])][row["binary"]] = row[
                "ids"
            ]
    assert len(seen) == 7872, len(seen)
    assert len(groups) == 42, len(groups)
    for pair in defaults.values():
        assert pair["baseline"] == pair["candidate"]

    summaries = []
    for key, rows in sorted(groups.items()):
        by_query = collections.defaultdict(list)
        for row in rows:
            by_query[row["query_id"]].append(row)
        cold = key[1] == "cold"
        expected = set(
            identity["evaluation_ids"][:16] if cold else identity["evaluation_ids"]
        )
        assert set(by_query) == expected
        timed_ids = set(identity["evaluation_ids"][: 16 if cold else 64])
        for query_id, repeated in by_query.items():
            expected_repeats = 1 if cold or query_id not in timed_ids else 3
            assert len(repeated) == expected_repeats
            assert all(row["timed"] == (query_id in timed_ids) for row in repeated)
        summary = dict(zip(FIELDS, key))
        summary["recall_queries"] = len(by_query)
        summary["timing_queries"] = len(timed_ids)
        summary["recall"] = float(
            np.mean(
                [
                    np.mean([row["recall"] for row in repeated])
                    for repeated in by_query.values()
                ]
            )
        )
        for metric in [
            "latency_ms",
            "cpu_ms",
            "partitions",
            "comparisons",
            "bytes_read",
            "iops",
            "scan_requests",
        ]:
            # One median per timed query gives every query equal weight.
            values = [
                float(np.median([row[metric] for row in repeated]))
                for query_id, repeated in by_query.items()
                if query_id in timed_ids
            ]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "p50": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
        timed = [row for row in rows if row["timed"]]
        summary["sequential_qps"] = (
            len(timed) * 1000 / sum(row["latency_ms"] for row in timed)
        )
        summaries.append(summary)
    output = {
        "audit_passed": True,
        "records": len(seen),
        "default_ids_identical": True,
        "method": (
            "Strict-ID recall averaged per query; timing is the median of three "
            "passes per query, then mean/p50/p95 across 64 queries. Cold samples "
            "have 16 queries and one pass. QPS uses all measured query wall times "
            "and excludes driver IPC."
        ),
        "identity": identity,
        "groups": summaries,
    }
    (directory / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    lines = [
        "| Scenario | Effort | Recall@k | Mean latency | P95 latency | Mean CPU | "
        "Mean probes | Mean read bytes | Mean IOPS | Sequential QPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        if row["binary"] != "candidate":
            continue
        scenario = f"{row['cache']} {row['index']} k={row['k']} refine={row['refine']}"
        if row["filtered"]:
            scenario += " filter=10%"
        lines.append(
            f"| {scenario} | {row['effort']} | {row['recall']:.3%} | "
            f"{row['latency_ms']['mean']:.3f} ms | {row['latency_ms']['p95']:.3f} ms | "
            f"{row['cpu_ms']['mean']:.3f} ms | {row['partitions']['mean']:.2f} | "
            f"{row['bytes_read']['mean']:.0f} B | {row['iops']['mean']:.2f} | "
            f"{row['sequential_qps']:.2f} queries/s |"
        )
    (directory / "results.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({"audit_passed": True, "records": len(seen), "groups": len(groups)})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("queries", type=Path)
    args = parser.parse_args()
    audit(args.directory, args.queries)
