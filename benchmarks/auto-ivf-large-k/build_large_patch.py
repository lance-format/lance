# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Apply frozen large-k profiles to an isolated experimental source checkout."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

from prepare_large import K_VALUES

RELATIVE_SOURCE = Path("rust/lance/src/io/exec/knn/adaptive_probe.rs")
BASE_SHA256 = "c2161dca2e078c88a6ed88cfae703311f10ac86e8eb62793ce6e5382b156c0b1"
BASE_KNN_SHA256 = "7b8fa385a69625dadcd803da42bbff4ab3d1447267776bb0c50459bf3ac137d7"
BASE_PROFILES = {
    "l2": [
        [0.2175, 0.265, 0.33, 0.375, 0.3725, 0.4175],
        [5, 6, 11, 15, 22, 29],
        [19, 24, 38, 56, 81, 93],
    ],
    "cosine": [
        [0.235, 0.2875, 0.38, 0.415, 0.45, 0.47],
        [3, 8, 7, 18, 28, 34],
        [50, 77, 106, 156, 192, 248],
    ],
    "dot": [
        [0.14, 0.055, 0.0625, 0.0675, 0.07, 0.0725],
        [56, 144, 200, 211, 244, 279],
        [112, 432, 768, 833, 971, 1092],
    ],
}


def patch(source, calibration, destination):
    path = source / RELATIVE_SOURCE
    before = path.read_text()
    assert hashlib.sha256(before.encode()).hexdigest() == BASE_SHA256
    assert source.resolve() != Path(__file__).resolve().parents[2], (
        "Use an isolated experiment checkout"
    )
    after = before.replace(
        "|| query.k > 1000",
        "|| (query.k > 1000 && !matches!(query.k, 10_000 | 100_000))",
    )
    after = after.replace(
        "//! No extra index statistics or file-format changes are needed.",
        "//! No extra index statistics or file-format changes are needed.\n"
        "//! This isolated experiment additionally enables exactly k=10_000 and\n"
        "//! k=100_000 with frozen profiles; intermediate k values remain legacy.",
    )
    old_buckets = "            201..=500 => 4,\n            _ => 5,"
    new_buckets = (
        "            201..=500 => 4,\n            501..=1000 => 5,\n"
        "            10_000 => 6,\n            _ => 7,"
    )
    assert after.count(old_buckets) == 1
    after = after.replace(old_buckets, new_buckets)
    for metric, arrays in BASE_PROFILES.items():
        for field, old in zip(["margin", "floor", "cap"], arrays):
            values = old + [
                calibration["profiles"][metric][str(k)][field] for k in K_VALUES
            ]
            old_text, new_text = str(old) + "[bucket]", str(values) + "[bucket]"
            assert after.count(old_text) == 1, old_text
            after = after.replace(old_text, new_text)
    cases = []
    metric_types = {"l2": "L2", "cosine": "Cosine", "dot": "Dot"}
    for metric, metric_type in metric_types.items():
        for k in K_VALUES:
            profile = calibration["profiles"][metric][str(k)]
            cases.append(
                f"    #[case::{metric}_top{k}(DistanceType::{metric_type}, {k}, "
                f"{profile['margin']}, {profile['floor']}, {profile['cap']})]\n"
            )
    test_marker = "    fn test_auto_probe_metric_profiles("
    assert after.count(test_marker) == 1
    after = after.replace(test_marker, "".join(cases) + test_marker)
    patch_text = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{RELATIVE_SOURCE}",
            tofile=f"b/{RELATIVE_SOURCE}",
        )
    )
    gate_path = source / "rust/lance/src/io/exec/knn.rs"
    gate_before = gate_path.read_text()
    assert hashlib.sha256(gate_before.encode()).hexdigest() == BASE_KNN_SHA256
    old_values = "#[values(1, 100, 101, 200, 500, 1000)] k: usize,"
    assert gate_before.count(old_values) == 1
    gate_after = gate_before.replace(
        old_values,
        "#[values(1, 100, 101, 200, 500, 1000, 10_000, 100_000)] k: usize,",
    )
    patch_text += "".join(
        difflib.unified_diff(
            gate_before.splitlines(keepends=True),
            gate_after.splitlines(keepends=True),
            fromfile="a/rust/lance/src/io/exec/knn.rs",
            tofile="b/rust/lance/src/io/exec/knn.rs",
        )
    )
    path.write_text(after)
    gate_path.write_text(gate_after)
    destination.write_text(patch_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("patch", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.calibration.read_text())
    assert frozen["evaluation_used"] is False and frozen["anchors"] == K_VALUES
    patch(args.source, frozen, args.patch)
