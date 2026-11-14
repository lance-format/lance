// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Score arithmetic: competitive-floor bounds, outward `f32` rounding and query-order summation.

use std::fmt::Debug;

use smallvec::SmallVec;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::scalar::inverted) enum CompetitiveFloorMode {
    #[default]
    Exclusive,
    Inclusive,
}

impl CompetitiveFloorMode {
    #[inline]
    pub(super) fn rejects_upper_bound(self, upper_bound: f64, floor: f32) -> bool {
        match self {
            Self::Exclusive => upper_bound <= f64::from(floor),
            Self::Inclusive => upper_bound < f64::from(floor),
        }
    }

    #[inline]
    pub(super) fn accepts_score(self, score: f32, floor: f32) -> bool {
        match self {
            Self::Exclusive => score > floor,
            Self::Inclusive => score >= floor,
        }
    }
}

/// Lucene's `MathUtil.sumUpperBound` factor, adapted to Lance's `f32` score
/// accumulation. Prefix bounds are summed in `f64`, then widened enough to
/// cover any recursive `f32` summation order of the same non-negative values.
#[inline]
pub(in crate::scalar::inverted) fn score_sum_upper_bound_factor(num_values: usize) -> f64 {
    if num_values <= 2 {
        1.0
    } else {
        let relative_error_bound = (num_values - 1) as f64 * f64::from(f32::EPSILON);
        1.0 + 2.0 * relative_error_bound
    }
}

#[inline]
pub(in crate::scalar::inverted) fn score_sum_cannot_compete(
    partial_score: f32,
    remaining_upper_bound: f64,
    floor: f32,
    upper_bound_factor: f64,
    floor_mode: CompetitiveFloorMode,
) -> bool {
    let upper_bound = (f64::from(partial_score) + remaining_upper_bound) * upper_bound_factor;
    floor_mode.rejects_upper_bound(f64::from(outward_f32_upper_bound(upper_bound)), floor)
}

pub(super) type ScoreContribution = ((u32, u32), f32);

/// Largest `f32` partial score that `score_sum_cannot_compete` still rejects
/// under the exclusive floor, given the other clauses' upper bound. The
/// rejection predicate is monotone in the partial score, so every score at or
/// below the returned limit is rejected and every score above it competes.
/// Returns `None` when no finite score is rejected.
pub(super) fn exclusive_partial_score_limit(
    remaining_upper_bound: f64,
    floor: f32,
    upper_bound_factor: f64,
) -> Option<f32> {
    let rejects = |score: f32| {
        score_sum_cannot_compete(
            score,
            remaining_upper_bound,
            floor,
            upper_bound_factor,
            CompetitiveFloorMode::Exclusive,
        )
    };
    if !remaining_upper_bound.is_finite() || !floor.is_finite() {
        return None;
    }
    // Start from the real-valued boundary and walk at most a few ULPs to the
    // exact f32 boundary of the (monotone) predicate.
    let mut limit = ((f64::from(floor) / upper_bound_factor) - remaining_upper_bound) as f32;
    if !limit.is_finite() {
        return None;
    }
    if rejects(limit) {
        for _ in 0..8 {
            let up = next_up_f32(limit);
            if up == limit || !rejects(up) {
                return Some(limit);
            }
            limit = up;
        }
    } else {
        for _ in 0..8 {
            let down = next_down_f32(limit);
            if down == limit {
                return None;
            }
            if rejects(down) {
                return Some(down);
            }
            limit = down;
        }
    }
    // The walk did not converge (pathological inputs); fall back to an exact
    // binary search over the ordered f32 bit patterns.
    let mut lo = f32::MIN;
    let mut hi = f32::MAX;
    if !rejects(lo) {
        return None;
    }
    if rejects(hi) {
        return Some(hi);
    }
    for _ in 0..64 {
        let mid = f32::from_bits(((lo.to_bits() as i64 + hi.to_bits() as i64) / 2) as u32);
        if mid == lo || mid == hi {
            break;
        }
        if rejects(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(lo)
}

pub(super) fn next_down_f32(value: f32) -> f32 {
    -next_up_f32(-value)
}

#[inline]
pub(super) fn score_contributions_in_query_order(
    mut contributions: SmallVec<[ScoreContribution; 8]>,
) -> f32 {
    if !contributions
        .windows(2)
        .all(|window| window[0].0 <= window[1].0)
    {
        contributions.sort_unstable_by_key(|contribution| contribution.0);
    }
    contributions
        .into_iter()
        .fold(0.0_f32, |score, (_, contribution)| score + contribution)
}

pub(super) fn conservative_score_sum(scores: impl Iterator<Item = f32>) -> f32 {
    let (num_scores, exact) = scores.fold((0, 0.0_f64), |(count, sum), score| {
        (count + 1, sum + f64::from(score))
    });
    let widened = exact * score_sum_upper_bound_factor(num_scores);
    outward_f32_upper_bound(widened)
}

/// Round a wide score bound to the least `f32` that still covers it.
#[inline]
pub(in crate::scalar::inverted) fn outward_f32_upper_bound(value: f64) -> f32 {
    let rounded = value as f32;
    if f64::from(rounded) < value {
        next_up_f32(rounded)
    } else {
        rounded
    }
}

pub(super) fn next_up_f32(value: f32) -> f32 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

#[inline]
pub(super) fn next_up_f64(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}
