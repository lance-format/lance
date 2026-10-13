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

/// How a query searches one partition's graph.
#[derive(Debug, Clone, Copy)]
pub struct VamanaQueryParams {
    /// `L`: the candidates the walk keeps.
    pub ef: usize,
    pub lower_bound: Option<f32>,
    pub upper_bound: Option<f32>,
    pub dist_q_c: f32,
}

impl From<&Query> for VamanaQueryParams {
    fn from(query: &Query) -> Self {
        let k = query.k * query.refine_factor.unwrap_or(1) as usize;
        Self {
            ef: query.ef.unwrap_or(k + k / 2),
            lower_bound: query.lower_bound,
            upper_bound: query.upper_bound,
            dist_q_c: query.dist_q_c,
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
