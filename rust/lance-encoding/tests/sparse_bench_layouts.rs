// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
// The reporting helper below prints its findings for a human to read.
#![allow(clippy::print_stdout)]
//! Guards the sparse benchmark inputs against writer heuristic drift.
//!
//! The benches in `benches/sparse_decode.rs` and `benches/sparse_footprint.rs` are only
//! meaningful if their inputs actually land on the sparse structural layout. Layout
//! selection is a writer-side heuristic, so a change to the rep/def budget or to automatic
//! sparse selection could quietly move a case onto the dense mini-block path, leaving the
//! benches green while measuring nothing. This test fails instead.

#[path = "../benches/sparse/cases.rs"]
mod cases;

use cases::{cases, encode_unchecked, layouts};

#[test]
fn bench_cases_produce_declared_layout() {
    for case in cases() {
        let encoded = encode_unchecked(&case);
        let observed = layouts(&encoded);
        assert!(!observed.is_empty(), "case {} produced no pages", case.name);
        for layout in &observed {
            assert_eq!(
                *layout, case.expect,
                "case {} expected {} but observed {:?}",
                case.name, case.expect, observed
            );
        }
    }
}

/// Prints the observed layout for every case, so the matrix can be re-derived after a
/// writer change without editing code:
/// `cargo test -p lance-encoding --test sparse_bench_layouts -- --ignored --nocapture`
#[test]
#[ignore = "reporting helper, not an assertion"]
fn report_bench_case_layouts() {
    for case in cases() {
        let encoded = encode_unchecked(&case);
        let observed = layouts(&encoded);
        println!(
            "{:<30} declared={:<10} observed={:<10} pages={} rows={}",
            case.name,
            case.expect.to_string(),
            observed
                .first()
                .map(|l| l.to_string())
                .unwrap_or_else(|| "none".into()),
            observed.len(),
            encoded.num_rows,
        );
    }
}
