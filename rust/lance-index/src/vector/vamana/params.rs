// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance_core::{Error, Result};
use serde::{Deserialize, Serialize};

use super::build::validate_alpha;
use super::graph::check_max_degree;
use crate::vector::Query;

/// The version of the per-partition [`VamanaMetadata`] this code writes.
pub const VAMANA_FORMAT_VERSION: u32 = 1;

/// How a partition's graph is built.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VamanaBuildParams {
    /// `R`: the most out-edges a vertex keeps.
    pub max_degree: u32,
    /// `L`: the candidates each build-time search keeps. Unrelated to the list
    /// a query keeps.
    pub search_list_size: usize,
    /// The pruning slack of the second build pass; the first pass prunes at 1.
    pub alpha: f32,
}

impl Default for VamanaBuildParams {
    /// The best of the `(R, L)` pairs measured on SIFT1M against Lance's HNSW
    /// at matched memory.
    fn default() -> Self {
        Self {
            max_degree: 64,
            search_list_size: 100,
            alpha: 1.2,
        }
    }
}

impl VamanaBuildParams {
    pub fn with_max_degree(mut self, max_degree: u32) -> Self {
        self.max_degree = max_degree;
        self
    }

    pub fn with_search_list_size(mut self, search_list_size: usize) -> Self {
        self.search_list_size = search_list_size;
        self
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn validate(&self) -> Result<()> {
        check_max_degree(self.max_degree)?;
        if self.search_list_size == 0 {
            return Err(Error::invalid_input(
                "Vamana search_list_size must be greater than zero".to_string(),
            ));
        }
        validate_alpha(self.alpha)
    }
}

/// How far past its `rank`-th nearest candidate a query's walk goes on
/// expanding, as a fraction of that candidate's length: the `gamma` of
/// adaptive beam search (Al-Jazzazi et al., NeurIPS 2025).
///
/// A walk stops once none of its unexpanded candidates is among its `rank`
/// nearest or nearer than `1 + margin` times the length to the `rank`-th. The
/// distances a walk measures are squared lengths, so the bar is
/// `(1 + margin)^2` times the `rank`-th's distance. A query whose neighbors
/// stand clear of everything else stops soon after finding them; one crowded
/// by near-equals walks further. At zero a walk expands its `rank` nearest and
/// stops.
///
/// Each walk still keeps every candidate its partition answers with, whatever
/// their distance. Its list length `ef` becomes a cap: a walk meant to follow
/// the margin wants one it never reaches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveStop {
    /// `gamma`: at least zero, with a finite `(1 + margin)^2`.
    pub margin: f32,
    /// The `k` the query asked for. Carried here because a partition is asked
    /// for more candidates than that whenever they are scored again.
    pub rank: usize,
}

impl AdaptiveStop {
    /// Refuse a margin no walk can stop by, and a rank other than `k`, the
    /// query's own, which must be at least one.
    pub fn validate(&self, k: usize) -> Result<()> {
        // A square that overflows would make the bar over a zero distance not
        // a number.
        let widened = 1.0 + self.margin;
        if !(self.margin >= 0.0 && (widened * widened).is_finite()) {
            return Err(Error::invalid_input(format!(
                "stop_margin {} is not a finite fraction of at least zero with a finite \
                 (1 + stop_margin)^2, so it cannot say how far past its k-th candidate a walk \
                 goes on",
                self.margin
            )));
        }
        if k == 0 {
            return Err(Error::invalid_input(
                "stop_margin hangs its bar off the query's k-th candidate, but k is zero"
                    .to_string(),
            ));
        }
        if self.rank != k {
            return Err(Error::invalid_input(format!(
                "stop_margin hangs its bar off the query's k-th candidate, but carries rank {} \
                 for a query of k {k}",
                self.rank
            )));
        }
        Ok(())
    }
}

/// How a query searches one partition's graph.
#[derive(Debug, Clone, Copy)]
pub struct VamanaQueryParams {
    /// `L`: the candidates the walk keeps.
    pub ef: usize,
    pub lower_bound: Option<f32>,
    pub upper_bound: Option<f32>,
    pub dist_q_c: f32,
    /// When the walk stops with candidates still unexpanded; `None` walks
    /// until nothing in its list is left to expand.
    pub stop: Option<AdaptiveStop>,
}

impl From<&Query> for VamanaQueryParams {
    fn from(query: &Query) -> Self {
        let k = query.k * query.refine_factor.unwrap_or(1) as usize;
        Self {
            ef: query.ef.unwrap_or(k + k / 2),
            lower_bound: query.lower_bound,
            upper_bound: query.upper_bound,
            dist_q_c: query.dist_q_c,
            stop: query.adaptive_stop,
        }
    }
}

/// What one partition's sub-index records beside its graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VamanaMetadata {
    pub format_version: u32,
    pub params: VamanaBuildParams,
    /// Where a walk starts when the partition keeps no entry points.
    pub medoid: u32,
    /// Ascending; a walk starts at the one nearest its query.
    pub entry_points: Vec<u32>,
    pub num_vertices: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_build_params_are_rejected() {
        assert!(VamanaBuildParams::default().validate().is_ok());
        for params in [
            VamanaBuildParams::default().with_max_degree(0),
            VamanaBuildParams::default().with_max_degree(1025),
            VamanaBuildParams::default().with_search_list_size(0),
            VamanaBuildParams::default().with_alpha(0.9),
            VamanaBuildParams::default().with_alpha(f32::NAN),
            VamanaBuildParams::default().with_alpha(f32::INFINITY),
        ] {
            assert!(params.validate().is_err(), "{params:?}");
        }
    }

    #[test]
    fn metadata_is_a_json_object_with_its_params() {
        let metadata = VamanaMetadata {
            format_version: VAMANA_FORMAT_VERSION,
            params: VamanaBuildParams::default(),
            medoid: 3,
            entry_points: vec![1, 5],
            num_vertices: 9,
        };
        let json = serde_json::to_string(&metadata).unwrap();
        assert_eq!(
            json,
            r#"{"format_version":1,"params":{"max_degree":64,"search_list_size":100,"alpha":1.2},"medoid":3,"entry_points":[1,5],"num_vertices":9}"#
        );
        assert_eq!(
            serde_json::from_str::<VamanaMetadata>(&json).unwrap(),
            metadata
        );
    }
}
