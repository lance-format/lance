// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The IVF_VAMANA_* part of [`VectorIndexDetails`](lance_index::pb::VectorIndexDetails):
//! the `vamana_index_config` field written at creation, read back into build
//! parameters, shown by `describe_indices`, and inferred from index files.

use std::collections::HashMap;

use lance_index::pb::vector_index_details::Compression;
use lance_index::pb::{VamanaParameters, VectorIndexDetails};
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::shared::partition_merger::SupportedIvfIndexType;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::vector::vamana::{VAMANA_METADATA_KEY, VamanaBuildParams, VamanaMetadata};
use lance_linalg::distance::DistanceType;
use lance_table::format::IndexMetadata;
use serde::Serialize;

use super::super::VectorIndexParams;
use super::super::details::apply_runtime_hints;
use crate::{Error, Result};

/// The graph parameters an IVF_VAMANA_* index records in its details, or
/// `None` for the parameters of any other index.
pub fn config(params: &VectorIndexParams) -> Option<VamanaParameters> {
    super::vamana_stage(params).map(config_of)
}

/// The build parameters `config` records.
pub fn build_params(config: &VamanaParameters) -> VamanaBuildParams {
    VamanaBuildParams {
        max_degree: config.max_degree,
        search_list_size: config.search_list_size as usize,
        alpha: config.alpha,
    }
}

/// The graph parameters `segment`'s details record, if any: those its graphs
/// were built with, which an index whose graphs are all empty records only
/// here.
pub fn recorded_params(segment: &IndexMetadata) -> Option<VamanaBuildParams> {
    let details = segment.index_details.as_deref()?;
    if details.value.is_empty() {
        return None;
    }
    details
        .to_msg::<VectorIndexDetails>()
        .ok()?
        .vamana_index_config
        .as_ref()
        .map(build_params)
}

fn config_of(vamana: &VamanaBuildParams) -> VamanaParameters {
    debug_assert!(
        u32::try_from(vamana.search_list_size).is_ok(),
        "check_params refuses a search list of {} rows, wider than 32 bits",
        vamana.search_list_size
    );
    VamanaParameters {
        max_degree: vamana.max_degree,
        search_list_size: u32::try_from(vamana.search_list_size).unwrap_or(u32::MAX),
        alpha: vamana.alpha,
    }
}

/// The parameters that build an index like the one `config` and
/// `compression` describe, or `None` for a combination no IVF_VAMANA_* index
/// has.
pub fn vector_params(
    metric_type: DistanceType,
    ivf: IvfBuildParams,
    config: VamanaParameters,
    compression: Option<Compression>,
    runtime_hints: &HashMap<String, String>,
) -> Option<VectorIndexParams> {
    let vamana = build_params(&config);
    let mut params = match compression {
        None | Some(Compression::Flat(_)) => {
            VectorIndexParams::ivf_vamana(metric_type, ivf, vamana)
        }
        Some(Compression::Sq(sq)) => VectorIndexParams::with_ivf_vamana_sq_params(
            metric_type,
            ivf,
            vamana,
            SQBuildParams {
                num_bits: sq.num_bits as u16,
                ..Default::default()
            },
        ),
        Some(Compression::Pq(pq)) => VectorIndexParams::with_ivf_vamana_pq_params(
            metric_type,
            ivf,
            vamana,
            PQBuildParams {
                num_bits: pq.num_bits as usize,
                num_sub_vectors: pq.num_sub_vectors as usize,
                ..Default::default()
            },
        ),
        Some(Compression::Rq(_)) => return None,
    };
    apply_runtime_hints(runtime_hints, &mut params);
    Some(params)
}

/// What `describe_indices` shows of an IVF_VAMANA_* index's graph.
#[derive(Serialize)]
pub struct VamanaDetailsJson {
    max_degree: u32,
    search_list_size: u32,
    alpha: f32,
}

impl From<VamanaParameters> for VamanaDetailsJson {
    fn from(config: VamanaParameters) -> Self {
        Self {
            max_degree: config.max_degree,
            search_list_size: config.search_list_size,
            alpha: config.alpha,
        }
    }
}

/// The graph parameters an index file of `index_type` records, for details
/// inferred from disk: `None` for any type but IVF_VAMANA_*.
///
/// Every partition records the same parameters; an empty partition records
/// an empty string. An index of empty partitions only reports the defaults.
pub fn infer(
    main_metadata: &HashMap<String, String>,
    index_type: SupportedIvfIndexType,
) -> Result<Option<VamanaParameters>> {
    if !matches!(
        index_type,
        SupportedIvfIndexType::IvfVamanaFlat
            | SupportedIvfIndexType::IvfVamanaPq
            | SupportedIvfIndexType::IvfVamanaSq
    ) {
        return Ok(None);
    }
    let partitions = main_metadata.get(VAMANA_METADATA_KEY).ok_or_else(|| {
        Error::index(format!(
            "{} index file is missing its {VAMANA_METADATA_KEY} metadata",
            index_type.as_str()
        ))
    })?;
    let partitions: Vec<String> = serde_json::from_str(partitions)?;
    let params = match partitions.iter().find(|partition| !partition.is_empty()) {
        Some(partition) => serde_json::from_str::<VamanaMetadata>(partition)?.params,
        None => VamanaBuildParams::default(),
    };
    if u32::try_from(params.search_list_size).is_err() {
        return Err(Error::index(format!(
            "{} index file records a search list of {} rows, wider than 32 bits",
            index_type.as_str(),
            params.search_list_size
        )));
    }
    Ok(Some(config_of(&params)))
}
