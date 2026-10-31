// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Constructors of IVF_VAMANA_* build parameters.

use std::collections::HashMap;

use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::vamana::VamanaBuildParams;
use lance_linalg::distance::DistanceType;

use super::super::{IndexFileVersion, StageParams, VectorIndexParams};

impl VectorIndexParams {
    /// Parameters of an `IVF_VAMANA_FLAT` index: IVF partitions that each hold
    /// a Vamana graph over the partition's vectors as they are.
    ///
    /// Only the l2 and cosine distances are supported; building with any
    /// other is refused.
    ///
    /// ```
    /// # use lance::index::vector::VectorIndexParams;
    /// # use lance_index::vector::ivf::IvfBuildParams;
    /// # use lance_index::vector::vamana::VamanaBuildParams;
    /// # use lance_linalg::distance::DistanceType;
    /// let params = VectorIndexParams::ivf_vamana(
    ///     DistanceType::L2,
    ///     IvfBuildParams::new(4),
    ///     VamanaBuildParams::default().with_max_degree(32),
    /// );
    /// assert_eq!(params.index_type().to_string(), "IVF_VAMANA_FLAT");
    /// ```
    pub fn ivf_vamana(
        distance_type: DistanceType,
        ivf: IvfBuildParams,
        vamana: VamanaBuildParams,
    ) -> Self {
        Self::vamana(
            distance_type,
            vec![StageParams::Ivf(ivf), StageParams::Vamana(vamana)],
        )
    }

    /// Parameters of an `IVF_VAMANA_SQ` index: the graph of each partition is
    /// built over, and walked with, scalar-quantized codes.
    ///
    /// ```
    /// # use lance::index::vector::VectorIndexParams;
    /// # use lance_index::vector::ivf::IvfBuildParams;
    /// # use lance_index::vector::sq::builder::SQBuildParams;
    /// # use lance_index::vector::vamana::VamanaBuildParams;
    /// # use lance_linalg::distance::DistanceType;
    /// let params = VectorIndexParams::with_ivf_vamana_sq_params(
    ///     DistanceType::Cosine,
    ///     IvfBuildParams::new(4),
    ///     VamanaBuildParams::default(),
    ///     SQBuildParams::default(),
    /// );
    /// assert_eq!(params.index_type().to_string(), "IVF_VAMANA_SQ");
    /// ```
    pub fn with_ivf_vamana_sq_params(
        distance_type: DistanceType,
        ivf: IvfBuildParams,
        vamana: VamanaBuildParams,
        sq: SQBuildParams,
    ) -> Self {
        Self::vamana(
            distance_type,
            vec![
                StageParams::Ivf(ivf),
                StageParams::Vamana(vamana),
                StageParams::SQ(sq),
            ],
        )
    }

    /// Parameters of an `IVF_VAMANA_PQ` index: the graph of each partition is
    /// built over, and walked with, product-quantized codes.
    ///
    /// ```
    /// # use lance::index::vector::VectorIndexParams;
    /// # use lance_index::vector::ivf::IvfBuildParams;
    /// # use lance_index::vector::pq::PQBuildParams;
    /// # use lance_index::vector::vamana::VamanaBuildParams;
    /// # use lance_linalg::distance::DistanceType;
    /// let params = VectorIndexParams::with_ivf_vamana_pq_params(
    ///     DistanceType::L2,
    ///     IvfBuildParams::new(4),
    ///     VamanaBuildParams::default(),
    ///     PQBuildParams::new(16, 8),
    /// );
    /// assert_eq!(params.index_type().to_string(), "IVF_VAMANA_PQ");
    /// ```
    pub fn with_ivf_vamana_pq_params(
        distance_type: DistanceType,
        ivf: IvfBuildParams,
        vamana: VamanaBuildParams,
        pq: PQBuildParams,
    ) -> Self {
        Self::vamana(
            distance_type,
            vec![
                StageParams::Ivf(ivf),
                StageParams::Vamana(vamana),
                StageParams::PQ(pq),
            ],
        )
    }

    fn vamana(distance_type: DistanceType, stages: Vec<StageParams>) -> Self {
        Self {
            stages,
            metric_type: distance_type,
            version: IndexFileVersion::V3,
            skip_transpose: false,
            runtime_hints: HashMap::new(),
        }
    }
}
