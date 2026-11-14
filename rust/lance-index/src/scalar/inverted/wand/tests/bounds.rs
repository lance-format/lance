// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::*;

/// `exclusive_partial_score_limit` must agree with `score_sum_cannot_compete`
/// for every f32 partial score: scores at or below the limit are rejected,
/// scores above it compete. Sweeps ULP neighbourhoods of the boundary plus
/// coarse samples, over floors, remaining bounds and clause counts.
#[test]
fn exclusive_partial_score_limit_matches_predicate() {
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut checked = 0usize;
    for case in 0..2_000 {
        let floor = if case % 7 == 0 {
            f32::from_bits((next() as u32) & 0x7F7F_FFFF)
        } else {
            (next() % 10_000) as f32 / 37.0 + 1e-6
        };
        let remaining = if case % 5 == 0 {
            f64::from(f32::from_bits((next() as u32) & 0x7F7F_FFFF))
        } else {
            (next() % 10_000) as f64 / 41.0
        };
        let num_lists = 2 + (next() % 6) as usize;
        let factor = score_sum_upper_bound_factor(num_lists);
        let rejects = |score: f32| {
            score_sum_cannot_compete(
                score,
                remaining,
                floor,
                factor,
                CompetitiveFloorMode::Exclusive,
            )
        };
        let limit = exclusive_partial_score_limit(remaining, floor, factor);
        let mut probes = vec![
            0.0_f32,
            f32::MIN_POSITIVE,
            floor,
            f32::MAX,
            f32::MIN,
            ((f64::from(floor) / factor) - remaining) as f32,
        ];
        if let Some(limit) = limit {
            let mut up = limit;
            let mut down = limit;
            for _ in 0..4 {
                probes.push(up);
                probes.push(down);
                up = next_up_f32(up);
                down = next_down_f32(down);
            }
        }
        for _ in 0..16 {
            probes.push(f32::from_bits((next() as u32) & 0x7F7F_FFFF));
        }
        for score in probes {
            let expected = rejects(score);
            let actual = limit.is_some_and(|limit| score <= limit);
            assert_eq!(
                expected, actual,
                "score={score:?} limit={limit:?} floor={floor:?} remaining={remaining:?} num_lists={num_lists}"
            );
            checked += 1;
        }
    }
    assert!(checked > 50_000);
}

#[test]
fn exclusive_partial_score_limit_handles_non_finite_inputs() {
    let factor = score_sum_upper_bound_factor(3);
    assert_eq!(
        exclusive_partial_score_limit(f64::INFINITY, 1.0, factor),
        None
    );
    assert_eq!(exclusive_partial_score_limit(1.0, f32::NAN, factor), None);
    assert_eq!(
        exclusive_partial_score_limit(1.0, f32::INFINITY, factor),
        None
    );
    // A floor no partial score can fail to beat rejects nothing.
    let limit = exclusive_partial_score_limit(1.0e30, 1.0, factor);
    assert!(limit.is_none_or(|limit| !score_sum_cannot_compete(
        next_up_f32(limit),
        1.0e30,
        1.0,
        factor,
        CompetitiveFloorMode::Exclusive
    )));
}

#[test]
fn conservative_score_sum_covers_query_order_f32_rounding() {
    let values = [
        f32::from_bits(0x3e65_15bd),
        f32::from_bits(0x34b4_3b11),
        f32::from_bits(0x35e9_48ed),
        f32::from_bits(0x3203_3773),
    ];
    let exact_score = values
        .into_iter()
        .fold(0.0_f32, |score, value| score + value);

    assert_eq!(exact_score.to_bits(), 0x3e65_164a);
    assert!(conservative_score_sum(values.into_iter()) >= exact_score);

    let mut reordered = [
        f32::from_bits(0x3c87_b63e),
        f32::from_bits(0x3d28_d10b),
        f32::from_bits(0x3cc4_29c0),
    ];
    let bound = conservative_score_sum(reordered.into_iter());
    reordered.sort_by(|left, right| right.total_cmp(left));
    let reordered_score = reordered
        .into_iter()
        .fold(0.0_f32, |score, value| score + value);
    assert_eq!(reordered_score.to_bits(), 0x3da7_6086);
    assert!(bound >= reordered_score);
}

#[test]
fn test_maxscore_prefix_bound_covers_f32_summation_rounding() {
    let remaining_bounds = [6.286_838_4e-7_f32, 0.015_441_144_f32];
    let essential_score = 2.762_496_2_f32;
    let threshold = f32::from_bits(0x4031_c9bc);

    // Raw f32 prefix accumulation rounds down to an apparent tie, while
    // scoring the clauses individually produces a competitive document.
    let raw_prefix = remaining_bounds.into_iter().sum::<f32>();
    assert_eq!(essential_score + raw_prefix, threshold);
    let actual_score = remaining_bounds
        .into_iter()
        .rev()
        .fold(essential_score, |score, bound| score + bound);
    assert!(actual_score > threshold);

    let prefix_bound = remaining_bounds.into_iter().map(f64::from).sum::<f64>();
    assert!(!score_sum_cannot_compete(
        essential_score,
        prefix_bound,
        threshold,
        score_sum_upper_bound_factor(3),
        CompetitiveFloorMode::Exclusive,
    ));
}

#[test]
fn two_clause_inclusive_bound_rounds_outward_at_f32_boundary() {
    let partial_score = 1.0_f32;
    let remaining_upper_bound = f32::from_bits(0x3380_0001);
    let floor = partial_score + remaining_upper_bound;

    assert_eq!(floor.to_bits(), 0x3f80_0001);
    assert!(!score_sum_cannot_compete(
        partial_score,
        f64::from(remaining_upper_bound),
        floor,
        score_sum_upper_bound_factor(2),
        CompetitiveFloorMode::Inclusive,
    ));
}
