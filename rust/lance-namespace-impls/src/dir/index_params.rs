// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use lance::index::IndexParams;
use lance::index::vector::VectorIndexParams;
use lance_core::Result;
use lance_index::IndexType;
use lance_index::scalar::{BuiltinIndexType, InvertedIndexParams, ScalarIndexParams};
use lance_index::vector::bq::{
    RABIT_MAX_NUM_BITS, RABIT_MIN_NUM_BITS, RQBuildParams, validate_supported_rq_num_bits,
};
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_linalg::distance::MetricType;
use lance_namespace::error::NamespaceError;
use lance_namespace::models::{
    CreateTableIndexRequest, DescribeTableIndexStatsResponse, IndexContent,
};

use super::DirectoryNamespace;

pub(super) enum DirectoryIndexParams {
    Scalar {
        index_type: IndexType,
        params: ScalarIndexParams,
    },
    Inverted(InvertedIndexParams),
    Vector {
        index_type: IndexType,
        params: VectorIndexParams,
    },
}

impl DirectoryIndexParams {
    pub(super) fn index_type(&self) -> IndexType {
        match self {
            Self::Scalar { index_type, .. } | Self::Vector { index_type, .. } => *index_type,
            Self::Inverted(_) => IndexType::Inverted,
        }
    }

    pub(super) fn params(&self) -> &dyn IndexParams {
        match self {
            Self::Scalar { params, .. } => params,
            Self::Inverted(params) => params,
            Self::Vector { params, .. } => params,
        }
    }
}

impl DirectoryNamespace {
    pub(super) fn parse_index_type(index_type: &str) -> Result<IndexType> {
        match index_type.trim().to_ascii_uppercase().as_str() {
            "SCALAR" | "BTREE" => Ok(IndexType::BTree),
            "BITMAP" => Ok(IndexType::Bitmap),
            "LABEL_LIST" | "LABELLIST" => Ok(IndexType::LabelList),
            "INVERTED" | "FTS" => Ok(IndexType::Inverted),
            "NGRAM" => Ok(IndexType::NGram),
            "ZONEMAP" | "ZONE_MAP" => Ok(IndexType::ZoneMap),
            "BLOOMFILTER" | "BLOOM_FILTER" => Ok(IndexType::BloomFilter),
            "RTREE" | "R_TREE" => Ok(IndexType::RTree),
            "VECTOR" | "IVF_PQ" => Ok(IndexType::IvfPq),
            "IVF_FLAT" => Ok(IndexType::IvfFlat),
            "IVF_SQ" => Ok(IndexType::IvfSq),
            "IVF_RQ" => Ok(IndexType::IvfRq),
            "IVF_HNSW_FLAT" => Ok(IndexType::IvfHnswFlat),
            "IVF_HNSW_SQ" => Ok(IndexType::IvfHnswSq),
            "IVF_HNSW_PQ" => Ok(IndexType::IvfHnswPq),
            other => Err(NamespaceError::InvalidInput {
                message: format!("Unsupported index_type '{}'", other),
            }
            .into()),
        }
    }

    pub(super) fn parse_metric_type(distance_type: Option<&str>) -> Result<MetricType> {
        let distance_type = distance_type.unwrap_or("l2");
        MetricType::try_from(distance_type).map_err(|e| {
            lance_core::Error::from(NamespaceError::InvalidInput {
                message: format!(
                    "Unsupported distance_type '{}' for vector index: {}",
                    distance_type, e
                ),
            })
        })
    }

    pub(super) fn build_index_params(
        request: &CreateTableIndexRequest,
    ) -> Result<DirectoryIndexParams> {
        let index_type = Self::parse_index_type(&request.index_type)?;
        Ok(match index_type {
            IndexType::BTree => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::BTree),
            },
            IndexType::Bitmap => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::Bitmap),
            },
            IndexType::LabelList => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::LabelList),
            },
            IndexType::NGram => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::NGram),
            },
            IndexType::ZoneMap => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
            },
            IndexType::BloomFilter => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::BloomFilter),
            },
            IndexType::RTree => DirectoryIndexParams::Scalar {
                index_type,
                params: ScalarIndexParams::for_builtin(BuiltinIndexType::RTree),
            },
            IndexType::Inverted => {
                let mut params = InvertedIndexParams::default();
                if let Some(with_position) = request.with_position {
                    params = params.with_position(with_position);
                }
                if let Some(base_tokenizer) = &request.base_tokenizer {
                    params = params.base_tokenizer(base_tokenizer.clone());
                }
                if let Some(language) = &request.language {
                    params = params.language(language)?;
                }
                if let Some(max_token_length) = request.max_token_length {
                    if max_token_length < 0 {
                        return Err(NamespaceError::InvalidInput {
                            message: format!(
                                "FTS max_token_length must be non-negative, got {}",
                                max_token_length
                            ),
                        }
                        .into());
                    }
                    params = params.max_token_length(Some(max_token_length as usize));
                }
                if let Some(lower_case) = request.lower_case {
                    params = params.lower_case(lower_case);
                }
                if let Some(stem) = request.stem {
                    params = params.stem(stem);
                }
                if let Some(remove_stop_words) = request.remove_stop_words {
                    params = params.remove_stop_words(remove_stop_words);
                }
                if let Some(ascii_folding) = request.ascii_folding {
                    params = params.ascii_folding(ascii_folding);
                }
                DirectoryIndexParams::Inverted(params)
            }
            IndexType::IvfFlat => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_flat_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                ),
            },
            IndexType::IvfPq => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_pq_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    PQBuildParams::default(),
                ),
            },
            IndexType::IvfSq => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_sq_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    SQBuildParams::default(),
                ),
            },
            IndexType::IvfRq => {
                let rq_params = if let Some(requested_num_bits) = request.num_bits {
                    let invalid_num_bits = || NamespaceError::InvalidInput {
                        message: format!(
                            "IVF_RQ num_bits must be in {}..={}, got {}",
                            RABIT_MIN_NUM_BITS, RABIT_MAX_NUM_BITS, requested_num_bits
                        ),
                    };
                    let num_bits =
                        u8::try_from(requested_num_bits).map_err(|_| invalid_num_bits())?;
                    validate_supported_rq_num_bits(num_bits).map_err(|_| invalid_num_bits())?;
                    RQBuildParams::new(num_bits)
                } else {
                    RQBuildParams::default()
                };
                DirectoryIndexParams::Vector {
                    index_type,
                    params: VectorIndexParams::with_ivf_rq_params(
                        Self::parse_metric_type(request.distance_type.as_deref())?,
                        IvfBuildParams::default(),
                        rq_params,
                    ),
                }
            }
            IndexType::IvfHnswFlat => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::ivf_hnsw(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    HnswBuildParams::default(),
                ),
            },
            IndexType::IvfHnswSq => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_hnsw_sq_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    HnswBuildParams::default(),
                    SQBuildParams::default(),
                ),
            },
            IndexType::IvfHnswPq => DirectoryIndexParams::Vector {
                index_type,
                params: VectorIndexParams::with_ivf_hnsw_pq_params(
                    Self::parse_metric_type(request.distance_type.as_deref())?,
                    IvfBuildParams::default(),
                    HnswBuildParams::default(),
                    PQBuildParams::default(),
                ),
            },
            other => {
                return Err(NamespaceError::InvalidInput {
                    message: format!("Unsupported index type for namespace API: {}", other),
                }
                .into());
            }
        })
    }

    pub(super) fn paginate_indices(
        indices: &mut Vec<IndexContent>,
        page_token: Option<String>,
        limit: Option<i32>,
    ) -> Option<String> {
        indices.sort_by(|a, b| a.index_name.cmp(&b.index_name));

        if let Some(start_after) = page_token {
            if let Some(index) = indices
                .iter()
                .position(|index| index.index_name.as_str() > start_after.as_str())
            {
                indices.drain(0..index);
            } else {
                indices.clear();
            }
        }

        let mut next_page_token = None;
        if let Some(limit) = limit
            && limit >= 0
        {
            let limit = limit as usize;
            if limit > 0 && indices.len() > limit {
                next_page_token = Some(indices[limit - 1].index_name.clone());
            }
            indices.truncate(limit);
        }
        if indices.is_empty() {
            None
        } else {
            next_page_token
        }
    }

    pub(super) fn describe_table_index_stats_response(
        stats: &serde_json::Value,
    ) -> DescribeTableIndexStatsResponse {
        let get_i64 = |key: &str| {
            stats.get(key).and_then(|value| {
                value
                    .as_i64()
                    .or_else(|| value.as_u64().and_then(|v| i64::try_from(v).ok()))
            })
        };

        DescribeTableIndexStatsResponse {
            distance_type: stats
                .get("distance_type")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            index_type: stats
                .get("index_type")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            num_indexed_rows: get_i64("num_indexed_rows"),
            num_unindexed_rows: get_i64("num_unindexed_rows"),
            num_indices: get_i64("num_indices").and_then(|value| i32::try_from(value).ok()),
            ..Default::default()
        }
    }
}
