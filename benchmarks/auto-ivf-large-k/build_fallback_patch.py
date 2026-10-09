# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors
"""Build an isolated, k-independent fallback experiment from frozen constants."""

import argparse
import difflib
import hashlib
import json
import struct
from pathlib import Path

from build_large_patch import BASE_KNN_SHA256, BASE_SHA256, RELATIVE_SOURCE


def patch(source, calibration, destination):
    path = source / RELATIVE_SOURCE
    before = path.read_text()
    assert hashlib.sha256(before.encode()).hexdigest() == BASE_SHA256
    assert source.resolve() != Path(__file__).resolve().parents[2], (
        "Use an isolated experiment checkout"
    )
    profiles = {metric: item["gap"] for metric, item in calibration["profiles"].items()}
    assert set(profiles) == {"l2", "cosine", "dot"}
    for profile in profiles.values():
        assert profile["feasible"]
        bits = struct.unpack("<I", struct.pack("<f", profile["threshold"]))[0]
        assert bits == profile["f32_bits"]
    after = before.replace(
        "    Legacy,\n    Adaptive(AutoProbeConfig),",
        "    Legacy,\n    Fallback,\n    Adaptive(AutoProbeConfig),",
    )
    marker = "            return Ok(Self::Legacy);\n        }\n        Ok(read_config"
    assert after.count(marker) == 1
    after = after.replace(
        marker,
        "            return Ok(match index.metric_type() {\n"
        "                DistanceType::L2 | DistanceType::Cosine "
        "| DistanceType::Dot => {\n"
        "                    Self::Fallback\n"
        "                }\n"
        "                _ => Self::Legacy,\n"
        "            });\n        }\n        Ok(read_config",
    )
    marker = "            Self::Legacy => apply_legacy_probes(query, distances),"
    assert after.count(marker) == 1
    after = after.replace(
        marker,
        marker + "\n            Self::Fallback => "
        "apply_fallback_probes(query, distances, metric),",
    )
    helper = [
        "/// Experimental constant margins for current-format fallback routing.",
        "/// Historical indices and uncalibrated metrics retain their original policy.",
        "fn apply_fallback_probes(query: &mut Query, "
        "distances: &[f32], metric: DistanceType) {",
        "    let margin = match metric {",
    ]
    for metric, rust_type in [("l2", "L2"), ("cosine", "Cosine"), ("dot", "Dot")]:
        bits = profiles[metric]["f32_bits"]
        helper.append(
            f"        DistanceType::{rust_type} => f32::from_bits(0x{bits:08x}),"
        )
    helper += [
        "        _ => return apply_legacy_probes(query, distances),",
        "    };",
        "    AutoProbeConfig {",
        "        min_initial_nprobes: 1,",
        "        margin,",
        "        max_initial_nprobes: None,",
        "    }",
        "    .apply(query, distances, metric);",
        "}",
        "",
    ]
    marker = (
        "#[derive(Clone, Copy, Debug, PartialEq)]\npub(super) struct AutoProbeConfig"
    )
    assert after.count(marker) == 1
    after = after.replace(marker, "\n".join(helper) + "\n" + marker)
    after = after.replace(
        "//! No extra index statistics or file-format changes are needed.",
        "//! No extra index statistics or file-format changes are needed.\n"
        "//! This isolated experiment uses one constant gap margin per metric for\n"
        "//! current-format fallback queries, independent of k and learned caps.",
    )
    after = after.replace(
        "//! use these profiles. Larger k, Hamming, other index types, "
        "and explicitly bounded\n"
        "//! Auto queries retain their existing heuristic and ignore these overrides.",
        "//! use these profiles. Other current-format L2/cosine/dot Auto queries use\n"
        "//! experimental constant fallback margins and ignore these overrides.\n"
        "//! Historical indices and Hamming retain their existing heuristic.",
    )

    tests = ["", "    #[rstest]"]
    examples = {
        "l2": ("L2", [1.0, 1.25, 1.5, 20.0]),
        "cosine": ("Cosine", [0.5, 0.625, 0.75, 10.0]),
        "dot": ("Dot", [-3.0, -2.0, -1.0, 77.0]),
    }
    for metric, (rust_type, distances) in examples.items():
        nearest = distances[0]
        scale = abs(1 - nearest) if metric == "dot" else nearest
        expected = max(
            sum(
                value - nearest <= profiles[metric]["threshold"] * scale
                for value in distances
            ),
            1,
        )
        tests.append(
            f"    #[case::{metric}(DistanceType::{rust_type}, "
            f"&{distances}, {expected})]"
        )
    tests += [
        "    fn test_fallback_budget_is_independent_of_k(",
        "        #[case] metric: DistanceType,",
        "        #[case] distances: &[f32],",
        "        #[case] expected: usize,",
        "        #[values(1, 2, 10, 11, 100, 1000, 1001, 10000, 100000)] k: usize,",
        "    ) {",
        "        let mut query = query();",
        "        query.k = k;",
        "        AutoProbePolicy::Fallback.apply(&mut query, distances, metric);",
        "        assert_eq!(query.minimum_nprobes, expected);",
        "        assert_eq!(query.maximum_nprobes, None);",
        "    }",
        "",
        "    #[rstest]",
        "    fn test_fallback_dot_preserves_positive_scaling(",
        "        #[values(0.125, 1.0, 8.0)] scale: f32,",
        "    ) {",
        "        let original = [4.0, 3.0, 2.0, -76.0];",
        "        let distances = original.map(|value| 1.0 - scale * value);",
        "        let mut actual = query();",
        "        actual.k = 100000;",
        "        AutoProbePolicy::Fallback.apply("
        "&mut actual, &distances, DistanceType::Dot);",
        "        let mut reference = query();",
        "        reference.k = 100000;",
        "        let reference_distances = original.map(|value| 1.0 - value);",
        "        AutoProbePolicy::Fallback.apply("
        "&mut reference, &reference_distances, DistanceType::Dot);",
        "        assert_eq!(actual.minimum_nprobes, reference.minimum_nprobes);",
        "    }",
        "",
        "    #[rstest]",
        "    #[case::minimum(4, None, 4)]",
        "    #[case::maximum(1, Some(2), 2)]",
        "    #[case::fixed(2, Some(2), 2)]",
        "    fn test_fallback_preserves_caller_bounds(",
        "        #[case] minimum: usize,",
        "        #[case] maximum: Option<usize>,",
        "        #[case] expected: usize,",
        "        #[values(DistanceType::L2, DistanceType::Cosine, DistanceType::Dot)]",
        "        metric: DistanceType,",
        "    ) {",
        "        let mut query = query();",
        "        query.minimum_nprobes = minimum;",
        "        query.maximum_nprobes = maximum;",
        "        AutoProbePolicy::Fallback.apply("
        "&mut query, &[1.0, 1.0, 1.0, 1.0], metric);",
        "        assert_eq!(query.minimum_nprobes, expected);",
        "        assert_eq!(query.maximum_nprobes, maximum);",
        "    }",
        "",
    ]
    end = after.rfind("\n}")
    assert end >= 0 and not after[end + 2 :].strip()
    after = after[:end] + "\n".join(tests) + after[end:]

    gate_path = source / "rust/lance/src/io/exec/knn.rs"
    gate_before = gate_path.read_text()
    assert hashlib.sha256(gate_before.encode()).hexdigest() == BASE_KNN_SHA256
    expected = (
        "                    AutoProbePolicy::Fixed\n"
        "                } else {\n"
        "                    AutoProbePolicy::Legacy\n"
        "                }"
    )
    assert gate_before.count(expected) == 1
    gate_after = gate_before.replace(
        expected,
        "                    AutoProbePolicy::Fixed\n"
        '                } else if matches!(scenario, "legacy" | "hamming") {\n'
        "                    AutoProbePolicy::Legacy\n"
        "                } else {\n"
        "                    AutoProbePolicy::Fallback\n"
        "                }",
    )
    patch_text = ""
    for old, new, relative in [
        (before, after, RELATIVE_SOURCE),
        (gate_before, gate_after, "rust/lance/src/io/exec/knn.rs"),
    ]:
        patch_text += "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
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
    assert frozen["evaluation_used"] is False
    assert frozen["calibration_all_k"] is True
    assert frozen["calibration_k_range"] == [1, 100000]
    patch(args.source, frozen, args.patch)
