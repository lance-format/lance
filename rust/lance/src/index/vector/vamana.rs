// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! IVF_VAMANA_* indices: IVF partitions that each hold a Vamana graph.
//!
//! This module exists only in the Vamana fork of Lance. The graph and its
//! upkeep live in [`lance_index::vector::vamana`]; this module ties them to
//! datasets. Upstream files reach it through hooks marked `// fork(vamana)`,
//! and `git grep 'fork(vamana)'` lists every one of them.
//!
//! Every IVF_VAMANA_PQ segment keeps its product codes row-major, as the
//! shards of an index built apart must for their merge
//! (`IvfIndexBuilder::with_transpose(false)`), and a partition's codes are
//! transposed when it loads. So a segment of any origin, a merge's included,
//! can be merged again.

pub mod details;
mod index;
mod layout;
mod local;
pub mod merge;
mod params;
mod rescore;
pub mod segments;
#[cfg(test)]
mod tests;
mod vectors;

use std::collections::HashMap;
use std::ops::RangeInclusive;
use std::sync::Arc;

use arrow_schema::DataType;
use lance_core::cache::LanceCache;
use lance_index::IndexType;
use lance_index::frag_reuse::CompactFragReuseIndex;
use lance_index::optimize::OptimizeOptions;
use lance_index::pb::VectorIndexDetails;
use lance_index::progress::IndexBuildProgress;
use lance_index::scalar::RowAddrTranslator;
use lance_index::vector::VectorIndex;
use lance_index::vector::flat::index::FlatQuantizer;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::ivf::storage::IvfModel;
use lance_index::vector::pq::{PQBuildParams, ProductQuantizer};
use lance_index::vector::quantizer::{QuantizationType, Quantizer};
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::v3::shuffler::Shuffler;
use lance_index::vector::vamana::{Vamana, VamanaBuildParams};
use lance_io::object_store::ObjectStore;
use lance_io::stream::RecordBatchStream;
use lance_linalg::distance::DistanceType;
use lance_table::format::{IndexFile, IndexMetadata};
use object_store::path::Path;
use uuid::Uuid;

use super::builder::{ExistingIndex, IvfIndexBuilder, VectorIndexBuildSummary};
use super::ivf::v2::{IVFIndex, IvfOpenContext, IvfStateEntryBox};
use super::utils::get_vector_type;
use super::{IndexFileVersion, StageParams, VectorIndexParams};
use crate::dataset::Dataset;
use crate::index::frag_reuse::ResolvedRemapping;
use crate::{Error, Result};

pub use index::wrap_for_search;

/// The `index_version`s of the IVF_VAMANA_* indices this build reads.
///
/// They sit far above upstream's vector versions and outside
/// `IndexType::max_vector_version()`: an upstream build hides these indices
/// and keeps them through its commits, and this build still hides every
/// upstream vector version it does not know.
pub const VAMANA_INDEX_VERSIONS: RangeInclusive<i32> = 100..=100;

/// The type of an IVF_VAMANA_* index whose partitions hold `quantization`
/// codes. RaBitQ codes are never built under a graph.
pub fn index_type(quantization: QuantizationType) -> IndexType {
    match quantization {
        QuantizationType::Flat | QuantizationType::FlatBin => IndexType::IvfVamanaFlat,
        QuantizationType::Scalar => IndexType::IvfVamanaSq,
        QuantizationType::Product => IndexType::IvfVamanaPq,
        QuantizationType::Rabit => IndexType::Vector,
    }
}

/// Whether this build reads `index` as an IVF_VAMANA_* index: one of its
/// versions, with Vamana's details or with empty vector details, as upstream
/// reads its own versions whatever their details. Details stay empty only
/// where inferring them from the index files failed.
pub fn is_supported_segment(index: &IndexMetadata) -> bool {
    VAMANA_INDEX_VERSIONS.contains(&index.index_version)
        && index.index_details.as_deref().is_some_and(|details| {
            details.type_url.ends_with("VectorIndexDetails")
                && (details.value.is_empty()
                    || details
                        .to_msg::<VectorIndexDetails>()
                        .is_ok_and(|details| details.vamana_index_config.is_some()))
        })
}

/// The Vamana stage of `params`, when they build an IVF_VAMANA_* index.
pub fn vamana_stage(params: &VectorIndexParams) -> Option<&VamanaBuildParams> {
    params.stages.iter().find_map(|stage| match stage {
        StageParams::Vamana(vamana) => Some(vamana),
        _ => None,
    })
}

/// Refuse IVF_VAMANA_* parameters no index can be built from, before a row
/// is read or a definition committed. Parameters without a Vamana stage pass.
///
/// Dot is refused because Lance measures it as `1 - dot`, which goes below
/// zero, where an `alpha` above 1 tightens the prune instead of relaxing it.
pub fn check_params(dataset: &Dataset, column: &str, params: &VectorIndexParams) -> Result<()> {
    let Some(vamana) = vamana_stage(params) else {
        return Ok(());
    };
    if !matches!(
        params.stages.as_slice(),
        [StageParams::Ivf(_), StageParams::Vamana(_)]
            | [
                StageParams::Ivf(_),
                StageParams::Vamana(_),
                StageParams::PQ(_) | StageParams::SQ(_)
            ]
    ) {
        return Err(Error::not_supported(format!(
            "IVF_VAMANA takes an IVF stage, a Vamana stage and at most one PQ or SQ stage, got \
             {:?}",
            params.stages
        )));
    }
    if let Some(StageParams::SQ(sq)) = params.stages.last()
        && sq.num_bits != 8
    {
        return Err(Error::not_supported(format!(
            "IVF_VAMANA_SQ encodes 8-bit scalar codes only, not {}-bit ones",
            sq.num_bits
        )));
    }
    if !matches!(params.metric_type, DistanceType::L2 | DistanceType::Cosine) {
        return Err(Error::not_supported(format!(
            "IVF_VAMANA supports the l2 and cosine distances, not {}",
            params.metric_type
        )));
    }
    if matches!(params.version, IndexFileVersion::Legacy) {
        return Err(Error::not_supported(
            "IVF_VAMANA is written in the V3 index file format only, not the legacy one",
        ));
    }
    vamana.validate()?;
    if u32::try_from(vamana.search_list_size).is_err() {
        return Err(Error::invalid_input(format!(
            "Vamana search_list_size {} does not fit in 32 bits",
            vamana.search_list_size
        )));
    }
    let (vector_type, element_type) = get_vector_type(dataset.schema(), column)?;
    let DataType::FixedSizeList(_, dimension) = &vector_type else {
        return Err(Error::not_supported(format!(
            "IVF_VAMANA indexes one vector per row, but column {column} is {vector_type}"
        )));
    };
    if !matches!(
        element_type,
        DataType::Float16 | DataType::Float32 | DataType::Float64
    ) {
        return Err(Error::not_supported(format!(
            "IVF_VAMANA indexes float vectors, but column {column} holds {element_type}"
        )));
    }
    // A shard keeps its given centroids without the checks training runs, and
    // a codebook of another type fails deep in the shuffle: both builds take
    // the codebook as one flat array of the vectors' values.
    if let Some(StageParams::Ivf(ivf)) = params.stages.first()
        && let Some(centroids) = ivf.centroids.as_deref()
        && (centroids.value_length() != *dimension || centroids.value_type() != element_type)
    {
        return Err(Error::invalid_input(format!(
            "IVF_VAMANA centroids of {} {} values do not fit column {column} of {dimension} \
             {element_type} values",
            centroids.value_length(),
            centroids.value_type()
        )));
    }
    if let Some(StageParams::PQ(pq)) = params.stages.last()
        && let Some(codebook) = pq.codebook.as_ref()
        && *codebook.data_type() != element_type
    {
        return Err(Error::invalid_input(format!(
            "An IVF_VAMANA codebook of {} does not fit column {column} of {element_type} \
             values: it takes a flat array of {element_type} values",
            codebook.data_type()
        )));
    }
    Ok(())
}

/// The fragments a build over the `requested` ones indexes, given `effective`,
/// the reading the build makes of them: a request naming every fragment builds
/// the whole index. A request for an IVF_VAMANA_* index with given centroids
/// builds a shard, which merges with the shards built after it, even over
/// every fragment.
pub fn shard_fragments(
    params: Option<&VectorIndexParams>,
    requested: Option<&[u32]>,
    effective: Option<Vec<u32>>,
) -> Option<Vec<u32>> {
    let given_centroids = |params: &VectorIndexParams| matches!(params.stages.first(), Some(StageParams::Ivf(ivf)) if ivf.centroids.is_some());
    match (params, requested) {
        (Some(params), Some(requested))
            if vamana_stage(params).is_some() && given_centroids(params) =>
        {
            Some(Dataset::normalize_fragment_ids(requested))
        }
        _ => effective,
    }
}

fn invalid_stages(params: &VectorIndexParams) -> Error {
    Error::index(format!(
        "Build Vector Index: invalid IVF_VAMANA stages: {:?}",
        params.stages
    ))
}

/// Build an IVF_VAMANA_* index over `dataset`'s `column`, or over the
/// fragments in `fragment_ids` only.
#[allow(clippy::too_many_arguments)]
pub async fn build(
    dataset: &Dataset,
    column: &str,
    uuid: Uuid,
    params: &VectorIndexParams,
    shuffler: Box<dyn Shuffler>,
    ivf_params: IvfBuildParams,
    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    fragment_ids: Option<&[u32]>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<VectorIndexBuildSummary> {
    build_with(
        dataset,
        column,
        uuid,
        params,
        shuffler,
        ivf_params,
        frag_reuse_index,
        fragment_ids,
        None,
        progress,
    )
    .await
}

/// Build one shard of an IVF_VAMANA_* index built apart: a segment over the
/// fragments in `fragment_ids` that keeps the IVF model `ivf`, even under
/// `retrain`, and, over product codes, the codebook `global_pq` makes of the
/// PQ stage, so that the shards merge into one segment. Scalar codes are
/// trained by each shard, as IVF_SQ's are, and a merge re-encodes them to
/// bounds the shards share.
#[allow(clippy::too_many_arguments)]
pub async fn build_shard(
    dataset: &Dataset,
    column: &str,
    uuid: Uuid,
    params: &VectorIndexParams,
    shuffler: Box<dyn Shuffler>,
    ivf_params: IvfBuildParams,
    ivf: IvfModel,
    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    fragment_ids: &[u32],
    global_pq: impl Fn(&PQBuildParams) -> Result<ProductQuantizer>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<Vec<IndexFile>> {
    let shard = Shard {
        ivf,
        codebook: match params.stages.last() {
            Some(StageParams::PQ(pq)) => Some(global_pq(pq)?),
            _ => None,
        },
    };
    let summary = build_with(
        dataset,
        column,
        uuid,
        params,
        shuffler,
        ivf_params,
        frag_reuse_index,
        Some(fragment_ids),
        Some(shard),
        progress,
    )
    .await?;
    Ok(summary.files)
}

/// What every shard of an index built apart shares.
struct Shard {
    ivf: IvfModel,
    /// The product quantizer, for product codes.
    codebook: Option<ProductQuantizer>,
}

#[allow(clippy::too_many_arguments)]
async fn build_with(
    dataset: &Dataset,
    column: &str,
    uuid: Uuid,
    params: &VectorIndexParams,
    shuffler: Box<dyn Shuffler>,
    ivf_params: IvfBuildParams,
    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    fragment_ids: Option<&[u32]>,
    shard: Option<Shard>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<VectorIndexBuildSummary> {
    let vamana = vamana_stage(params)
        .ok_or_else(|| invalid_stages(params))?
        .clone();
    let index_dir = dataset.indices_dir().join(uuid.to_string());
    let (shard_ivf, codebook) = match shard {
        Some(Shard { ivf, codebook }) => (Some(ivf), codebook),
        None => (None, None),
    };
    macro_rules! build {
        ($quantizer:ty, $quantizer_params:expr, $quantizer_value:expr) => {{
            let mut builder = IvfIndexBuilder::<Vamana, $quantizer>::new(
                dataset.clone(),
                column.to_owned(),
                index_dir,
                params.metric_type,
                shuffler,
                Some(ivf_params),
                Some($quantizer_params),
                vamana,
                frag_reuse_index,
            )?;
            builder
                .with_optional_fragment_filter(fragment_ids)
                .with_progress(progress)
                .with_transpose(false);
            if let Some(ivf) = shard_ivf {
                builder.with_ivf(ivf);
            }
            if let Some(quantizer) = $quantizer_value {
                builder.with_quantizer(quantizer);
            }
            builder.build().await?
        }};
    }
    let summary = match params.stages.last() {
        Some(StageParams::Vamana(_)) => build!(FlatQuantizer, (), None),
        Some(StageParams::SQ(sq)) => build!(ScalarQuantizer, sq.clone(), None),
        Some(StageParams::PQ(pq)) => build!(ProductQuantizer, pq.clone(), codebook),
        _ => return Err(invalid_stages(params)),
    };
    Ok(summary)
}

/// Build a delta IVF_VAMANA_* index that shares an existing index's IVF model
/// and quantizer.
#[allow(clippy::too_many_arguments)]
pub async fn build_incremental(
    dataset: &Dataset,
    column: &str,
    index_dir: Path,
    params: &VectorIndexParams,
    shuffler: Box<dyn Shuffler>,
    ivf_model: IvfModel,
    quantizer: Quantizer,
    quantization_type: QuantizationType,
    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<VectorIndexBuildSummary> {
    let vamana = vamana_stage(params)
        .ok_or_else(|| invalid_stages(params))?
        .clone();
    macro_rules! incremental {
        ($quantizer:ty) => {{
            let mut builder = IvfIndexBuilder::<Vamana, $quantizer>::new_incremental(
                dataset.clone(),
                column.to_owned(),
                index_dir,
                params.metric_type,
                shuffler,
                vamana,
                frag_reuse_index,
                OptimizeOptions::append(),
            )?;
            builder
                .with_ivf(ivf_model)
                .with_quantizer(quantizer.try_into()?)
                .with_transpose(false)
                .with_progress(progress);
            builder.build().await
        }};
    }
    match quantization_type {
        QuantizationType::Flat => incremental!(FlatQuantizer),
        QuantizationType::Scalar => incremental!(ScalarQuantizer),
        QuantizationType::Product => incremental!(ProductQuantizer),
        QuantizationType::FlatBin | QuantizationType::Rabit => {
            Err(unsupported_codes(quantization_type))
        }
    }
}

fn unsupported_codes(quantization_type: QuantizationType) -> Error {
    Error::not_supported(format!(
        "IVF_VAMANA does not support {quantization_type} codes"
    ))
}

/// Optimize an IVF_VAMANA_* index: fold `unindexed` rows and the segments of
/// `existing_indices` into a new segment, as `optimize_indices` asks.
#[allow(clippy::too_many_arguments)]
pub async fn optimize(
    dataset: &Dataset,
    unindexed: Option<impl RecordBatchStream + Unpin + 'static>,
    column: &str,
    index_dir: Path,
    distance_type: DistanceType,
    shuffler: Box<dyn Shuffler>,
    ivf_model: &IvfModel,
    quantizer: Quantizer,
    quantization_type: QuantizationType,
    reference_index: &dyn VectorIndex,
    existing_indices: &[ExistingIndex],
    frag_reuse_index: Option<Arc<CompactFragReuseIndex>>,
    options: &OptimizeOptions,
    target_partition_size: Option<usize>,
) -> Result<VectorIndexBuildSummary> {
    let vamana = derive_params(reference_index)?;
    macro_rules! optimize {
        ($quantizer:ty) => {{
            let mut builder = IvfIndexBuilder::<Vamana, $quantizer>::new_incremental(
                dataset.clone(),
                column.to_owned(),
                index_dir,
                distance_type,
                shuffler,
                vamana,
                frag_reuse_index,
                options.clone(),
            )?;
            builder
                .with_ivf(ivf_model.clone())
                .with_quantizer(quantizer.try_into()?)
                .with_transpose(false)
                .with_existing_index_sources(existing_indices.to_vec())
                .with_progress(options.progress.clone())
                .with_target_partition_size(target_partition_size)
                .shuffle_data_input(unindexed);
            builder.build().await
        }};
    }
    match quantization_type {
        QuantizationType::Flat => optimize!(FlatQuantizer),
        QuantizationType::Scalar => optimize!(ScalarQuantizer),
        QuantizationType::Product => optimize!(ProductQuantizer),
        QuantizationType::FlatBin | QuantizationType::Rabit => {
            Err(unsupported_codes(quantization_type))
        }
    }
}

/// Write a remapped copy of an IVF_VAMANA_* index under `index_dir`.
///
/// Each partition's graph follows the remap through consolidation (see
/// [`lance_index::vector::vamana::remap`]) rather than being built again.
pub async fn remap(
    dataset: Dataset,
    column: String,
    index_dir: Path,
    index: Arc<dyn VectorIndex>,
    mapping: &RowAddrTranslator,
    quantization_type: QuantizationType,
) -> Result<Vec<IndexFile>> {
    // Boxed like `remap_index_file_v3`'s own arms: the remap is polled on the
    // calling thread under Python's `block_on`.
    macro_rules! remap {
        ($quantizer:ty) => {{
            let mut builder = IvfIndexBuilder::<Vamana, $quantizer>::new_remapper(
                dataset, column, index_dir, index,
            )?;
            builder.with_transpose(false);
            Box::pin(builder.remap_streaming(mapping)).await
        }};
    }
    match quantization_type {
        QuantizationType::Flat => remap!(FlatQuantizer),
        QuantizationType::Scalar => remap!(ScalarQuantizer),
        QuantizationType::Product => remap!(ProductQuantizer),
        QuantizationType::FlatBin | QuantizationType::Rabit => {
            Err(unsupported_codes(quantization_type))
        }
    }
}

/// Open an IVF_VAMANA_* index file of `dataset`'s `column` as queries search
/// it, with the state entry the index cache keeps for it. The entry is always
/// there: the cache keeps it, never the index this returns.
#[allow(clippy::too_many_arguments)]
pub async fn open_ivf(
    dataset: &Dataset,
    index_meta: &IndexMetadata,
    column: &str,
    index_type: &str,
    object_store: Arc<ObjectStore>,
    index_dir: Path,
    uuid: Uuid,
    remapping: Option<ResolvedRemapping>,
    file_metadata_cache: &LanceCache,
    index_cache: LanceCache,
    file_sizes: HashMap<String, u64>,
    open_context: IvfOpenContext,
) -> Result<(Arc<dyn VectorIndex>, Option<IvfStateEntryBox>)> {
    macro_rules! open {
        ($quantizer:ty) => {{
            let ivf = IVFIndex::<Vamana, $quantizer>::try_new(
                object_store,
                index_dir,
                uuid,
                remapping,
                file_metadata_cache,
                index_cache,
                file_sizes,
                open_context,
            )
            .await?;
            let entry = ivf.to_state_entry();
            let index = wrap_for_search(dataset, index_meta, column, Arc::new(ivf))?;
            Ok((index, Some(entry)))
        }};
    }
    match index_type {
        "IVF_VAMANA_FLAT" => open!(FlatQuantizer),
        "IVF_VAMANA_SQ" => open!(ScalarQuantizer),
        "IVF_VAMANA_PQ" => open!(ProductQuantizer),
        other => Err(Error::index(format!("Unsupported index type: {other}"))),
    }
}

/// The parameters that build another index like `index`: a delta sharing its
/// codebook when `keep_codebook`, otherwise a segment that trains its own.
pub fn params_for_index(
    index: &dyn VectorIndex,
    metric_type: DistanceType,
    ivf_params: IvfBuildParams,
    quantizer: Quantizer,
    quantization_type: QuantizationType,
    keep_codebook: bool,
) -> Result<VectorIndexParams> {
    let vamana = derive_params(index)?;
    Ok(match quantization_type {
        QuantizationType::Flat => VectorIndexParams::ivf_vamana(metric_type, ivf_params, vamana),
        QuantizationType::Scalar => {
            let quantizer: ScalarQuantizer = quantizer.try_into()?;
            VectorIndexParams::with_ivf_vamana_sq_params(
                metric_type,
                ivf_params,
                vamana,
                super::derive_sq_params(&quantizer),
            )
        }
        QuantizationType::Product => {
            let quantizer: ProductQuantizer = quantizer.try_into()?;
            let mut pq_params = super::derive_pq_params(&quantizer);
            if !keep_codebook {
                pq_params.codebook = None;
            }
            VectorIndexParams::with_ivf_vamana_pq_params(metric_type, ivf_params, vamana, pq_params)
        }
        QuantizationType::FlatBin | QuantizationType::Rabit => {
            return Err(unsupported_codes(quantization_type));
        }
    })
}

/// The build parameters `index`'s partitions record, or the defaults when
/// every partition is empty and records none.
pub fn derive_params(index: &dyn VectorIndex) -> Result<VamanaBuildParams> {
    let statistics = index.statistics()?;
    match statistics
        .get("sub_index")
        .and_then(|sub_index| sub_index.get("params"))
    {
        Some(params) => Ok(serde_json::from_value(params.clone())?),
        None => Ok(VamanaBuildParams::default()),
    }
}
