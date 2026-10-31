// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! `create_index` parameters of the IVF_VAMANA_* index types.

use lance::index::vector::VectorIndexParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::vamana::VamanaBuildParams;
use lance_linalg::distance::MetricType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The parameters of `index_type`, one of the IVF_VAMANA_* types, whose graph
/// takes the keyword arguments `max_degree`, `search_list_size` and `alpha`.
/// Lance validates them when the build starts.
pub fn vector_index_params(
    index_type: &str,
    metric_type: MetricType,
    ivf: IvfBuildParams,
    pq: PQBuildParams,
    sq: SQBuildParams,
    kwargs: Option<&Bound<PyDict>>,
) -> PyResult<VectorIndexParams> {
    let mut vamana = VamanaBuildParams::default();
    if let Some(kwargs) = kwargs {
        if let Some(max_degree) = kwargs.get_item("max_degree")? {
            vamana.max_degree = max_degree.extract()?;
        }
        if let Some(search_list_size) = kwargs.get_item("search_list_size")? {
            vamana.search_list_size = search_list_size.extract()?;
        }
        if let Some(alpha) = kwargs.get_item("alpha")? {
            vamana.alpha = alpha.extract()?;
        }
    }
    match index_type {
        "IVF_VAMANA_FLAT" => Ok(VectorIndexParams::ivf_vamana(metric_type, ivf, vamana)),
        "IVF_VAMANA_PQ" => Ok(VectorIndexParams::with_ivf_vamana_pq_params(
            metric_type,
            ivf,
            vamana,
            pq,
        )),
        "IVF_VAMANA_SQ" => Ok(VectorIndexParams::with_ivf_vamana_sq_params(
            metric_type,
            ivf,
            vamana,
            sq,
        )),
        _ => Err(PyValueError::new_err(format!(
            "Index type '{index_type}' is not supported."
        ))),
    }
}
