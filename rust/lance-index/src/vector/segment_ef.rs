// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Beam width (`ef`) for each segment of a partition split into a base graph
//! and delta graphs.
//!
//! A query searches every segment and merges their top-k. A delta holding a
//! share `f` of the partition's live rows holds about `f * k` of the true top-k,
//! so it needs a narrower beam than the base to find them; searching it with
//! the full `ef` costs about as much as searching the base.

use lance_core::{Error, Result};

/// How a delta segment's `ef` is derived from the query's `ef`.
///
/// The base segment (the one with the most rows) is always searched with the
/// query's `ef`, and no segment is searched with less than the floor of
/// [`SegmentEfParams`] or more than the query's `ef`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SegmentEfPolicy {
    /// Every segment is searched with the query's `ef`.
    Uniform,
    /// Deltas are searched with `ef / divisor`.
    Capped { divisor: usize },
    /// A delta holding a share `f` of the live rows is searched with
    /// `ef * min(1, multiplier * f)`.
    ///
    /// Measured on 384-d embeddings (m = 20, 1M and 4M row bases, 1 to 32
    /// deltas holding 1.6% to 100% of the base, 0 to 50% deleted rows, ef 64 to
    /// 256), multiplier 4 with floor 20 halves the distance computations of a
    /// query (2.3x faster on average) while recall@10 stays within 0.009 of
    /// [`Self::Uniform`]. The recall loss is largest when the deltas together
    /// hold as many rows as the base; a delta holding a quarter of the live rows
    /// or more keeps the full `ef`.
    Proportional { multiplier: f64 },
}

/// A [`SegmentEfPolicy`] with the floor that bounds it from below.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentEfParams {
    pub policy: SegmentEfPolicy,
    /// Smallest `ef` a delta is searched with, raised to `k * refine_factor`
    /// when that is larger, since a search must keep that many candidates.
    pub floor: usize,
    /// Whether the floor grows with a segment's deleted rows, by
    /// `1 / (1 - dead_fraction)`: deleted nodes are still traversed but never
    /// returned, so they thin the beam.
    pub scale_floor_by_dead: bool,
}

/// Uniform, so narrowing delta segments is opt-in, with floor 20 for a policy
/// that narrows.
impl Default for SegmentEfParams {
    fn default() -> Self {
        Self {
            policy: SegmentEfPolicy::Uniform,
            floor: 20,
            scale_floor_by_dead: false,
        }
    }
}

impl SegmentEfParams {
    pub fn validate(&self) -> Result<()> {
        match self.policy {
            SegmentEfPolicy::Capped { divisor: 0 } => Err(Error::invalid_input(
                "SegmentEfPolicy::Capped divisor must be at least 1, got 0",
            )),
            SegmentEfPolicy::Proportional { multiplier }
                if !multiplier.is_finite() || multiplier <= 0.0 =>
            {
                Err(Error::invalid_input(format!(
                    "SegmentEfPolicy::Proportional multiplier must be finite and positive, got {multiplier}"
                )))
            }
            _ => Ok(()),
        }
    }
}

/// Rows of one segment of a partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentRows {
    /// Rows the segment holds, deleted or not.
    pub rows: usize,
    /// Rows of `rows` that are deleted.
    pub dead_rows: usize,
}

/// The `ef` to search each segment with, in the order of `segments`.
///
/// `ef` is the query's beam width and `k` the number of results each segment
/// must return (`k * refine_factor`). The segment with the most rows, the
/// first of them on a tie, is the base and keeps `ef`.
///
/// # Examples
///
/// ```
/// use lance_index::vector::segment_ef::{
///     SegmentEfParams, SegmentEfPolicy, SegmentRows, segment_efs,
/// };
///
/// let params = SegmentEfParams {
///     policy: SegmentEfPolicy::Proportional { multiplier: 4.0 },
///     floor: 32,
///     scale_floor_by_dead: false,
/// };
/// let segments = [
///     SegmentRows { rows: 1_000_000, dead_rows: 0 },
///     SegmentRows { rows: 125_000, dead_rows: 0 },
/// ];
/// assert_eq!(segment_efs(128, 10, &segments, &params)?, vec![128, 57]);
/// # Ok::<(), lance_core::Error>(())
/// ```
pub fn segment_efs(
    ef: usize,
    k: usize,
    segments: &[SegmentRows],
    params: &SegmentEfParams,
) -> Result<Vec<usize>> {
    params.validate()?;
    if ef < k {
        return Err(Error::invalid_input(format!(
            "ef must be greater than or equal to k, got ef {ef} and k {k}"
        )));
    }
    if let Some((i, segment)) = segments
        .iter()
        .enumerate()
        .find(|(_, segment)| segment.dead_rows > segment.rows)
    {
        return Err(Error::invalid_input(format!(
            "segment {i} has {} dead rows but only {} rows",
            segment.dead_rows, segment.rows
        )));
    }

    let live = |segment: &SegmentRows| segment.rows - segment.dead_rows;
    let live_total: usize = segments.iter().map(live).sum();
    let base = segments
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.rows.cmp(&b.rows).then(j.cmp(i)))
        .map(|(i, _)| i);
    if params.policy == SegmentEfPolicy::Uniform || live_total == 0 {
        return Ok(vec![ef; segments.len()]);
    }

    Ok(segments
        .iter()
        .enumerate()
        .map(|(i, segment)| {
            if Some(i) == base {
                return ef;
            }
            let mut floor = params.floor.max(k) as f64;
            if params.scale_floor_by_dead && segment.rows > 0 {
                floor /= live(segment) as f64 / segment.rows as f64;
            }
            let wanted = match params.policy {
                SegmentEfPolicy::Uniform => ef as f64,
                SegmentEfPolicy::Capped { divisor } => ef as f64 / divisor as f64,
                SegmentEfPolicy::Proportional { multiplier } => {
                    let share = live(segment) as f64 / live_total as f64;
                    ef as f64 * (multiplier * share).min(1.0)
                }
            };
            (wanted.max(floor).ceil() as usize).min(ef)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn rows(rows: usize, dead_rows: usize) -> SegmentRows {
        SegmentRows { rows, dead_rows }
    }

    fn params(policy: SegmentEfPolicy, floor: usize) -> SegmentEfParams {
        SegmentEfParams {
            policy,
            floor,
            scale_floor_by_dead: false,
        }
    }

    const CAPPED: SegmentEfPolicy = SegmentEfPolicy::Capped { divisor: 4 };
    const PROPORTIONAL: SegmentEfPolicy = SegmentEfPolicy::Proportional { multiplier: 4.0 };

    #[rstest]
    // Small deltas drop to the floor; the base keeps ef.
    #[case::proportional_small(PROPORTIONAL, 32, &[rows(1000, 0), rows(16, 0), rows(16, 0)], &[128, 32, 32])]
    // A delta holding a third of the rows needs the full beam.
    #[case::proportional_large(PROPORTIONAL, 32, &[rows(1000, 0), rows(500, 0)], &[128, 128])]
    #[case::proportional_mid(PROPORTIONAL, 20, &[rows(900, 0), rows(100, 0)], &[128, 52])]
    #[case::capped(CAPPED, 20, &[rows(1000, 0), rows(500, 0)], &[128, 32])]
    #[case::capped_floor(CAPPED, 48, &[rows(1000, 0), rows(500, 0)], &[128, 48])]
    #[case::uniform(SegmentEfPolicy::Uniform, 20, &[rows(1000, 0), rows(10, 0)], &[128, 128])]
    // The base is the largest segment wherever it sits; ties go to the first.
    #[case::base_not_first(CAPPED, 20, &[rows(10, 0), rows(1000, 0)], &[32, 128])]
    #[case::tie(CAPPED, 20, &[rows(10, 0), rows(10, 0)], &[128, 32])]
    #[case::single_segment(CAPPED, 20, &[rows(10, 0)], &[128])]
    #[case::no_segments(CAPPED, 20, &[], &[])]
    // Nothing live: no share to scale by, so every segment keeps ef.
    #[case::all_dead(PROPORTIONAL, 20, &[rows(10, 10), rows(5, 5)], &[128, 128])]
    // Shares count live rows only.
    #[case::dead_shrink_share(PROPORTIONAL, 20, &[rows(1000, 0), rows(400, 300)], &[128, 47])]
    fn test_segment_efs(
        #[case] policy: SegmentEfPolicy,
        #[case] floor: usize,
        #[case] segments: &[SegmentRows],
        #[case] expected: &[usize],
    ) {
        assert_eq!(
            segment_efs(128, 10, segments, &params(policy, floor)).unwrap(),
            expected
        );
    }

    #[test]
    fn test_segment_efs_never_below_k_or_above_ef() {
        let segments = [rows(1000, 0), rows(1, 0)];
        // k above the floor raises it.
        assert_eq!(
            segment_efs(128, 50, &segments, &params(PROPORTIONAL, 20)).unwrap(),
            vec![128, 50]
        );
        // A floor above ef is capped at ef.
        assert_eq!(
            segment_efs(40, 10, &segments, &params(PROPORTIONAL, 64)).unwrap(),
            vec![40, 40]
        );
    }

    #[test]
    fn test_segment_efs_scales_floor_by_dead() {
        let segments = [rows(1000, 0), rows(100, 50)];
        let scaled = SegmentEfParams {
            scale_floor_by_dead: true,
            ..params(CAPPED, 20)
        };
        // floor 20 / (1 - 0.5) = 40, above ef / 4 = 32.
        assert_eq!(
            segment_efs(128, 10, &segments, &scaled).unwrap(),
            vec![128, 40]
        );
        assert_eq!(
            segment_efs(128, 10, &segments, &params(CAPPED, 20)).unwrap(),
            vec![128, 32]
        );
    }

    #[rstest]
    #[case::ef_below_k(8, &[rows(10, 0)], SegmentEfParams::default(), "ef must be greater")]
    #[case::too_many_dead(64, &[rows(10, 11)], SegmentEfParams::default(), "11 dead rows")]
    #[case::zero_divisor(64, &[rows(10, 0)], params(SegmentEfPolicy::Capped { divisor: 0 }, 20), "divisor")]
    #[case::bad_multiplier(64, &[rows(10, 0)], params(SegmentEfPolicy::Proportional { multiplier: f64::NAN }, 20), "multiplier")]
    fn test_segment_efs_rejects_invalid_input(
        #[case] ef: usize,
        #[case] segments: &[SegmentRows],
        #[case] params: SegmentEfParams,
        #[case] message: &str,
    ) {
        let error = segment_efs(ef, 10, segments, &params).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains(message), "{error}");
    }
}
