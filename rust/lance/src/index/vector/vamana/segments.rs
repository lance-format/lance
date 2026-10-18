// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Merges of IVF_VAMANA_* segments: the checks a merge passes before it
//! writes anything, and the root file of the merged segment.
//!
//! A merge of segments built apart unites their storage in one
//! `auxiliary.idx` and leaves the root file to the index type. As IVF_HNSW_*
//! merges do, each partition's graph is drawn anew over the partition's
//! merged rows, with the build parameters the segments share.

use std::sync::Arc;

use lance_core::cache::LanceCache;
use lance_core::datatypes::Schema;
use lance_core::utils::tokio::spawn_cpu;
use lance_file::reader::{FileReader, FileReaderOptions};
use lance_file::versions as file_versions;
use lance_file::writer::FileWriterOptions;
use lance_index::pb;
use lance_index::progress::IndexBuildProgress;
use lance_index::vector::flat::index::FlatQuantizer;
use lance_index::vector::ivf::storage::{IVF_METADATA_KEY, IvfModel};
use lance_index::vector::pq::ProductQuantizer;
use lance_index::vector::quantizer::Quantization;
use lance_index::vector::shared::{SupportedIvfIndexType, write_unified_ivf_and_index_metadata};
use lance_index::vector::sq::ScalarQuantizer;
use lance_index::vector::storage::IvfQuantizationStorage;
use lance_index::vector::v3::subindex::IvfSubIndex;
use lance_index::vector::vamana::{VAMANA_METADATA_KEY, Vamana, VamanaBuildParams, VamanaMetadata};
use lance_index::{
    INDEX_AUXILIARY_FILE_NAME, INDEX_FILE_NAME, INDEX_METADATA_SCHEMA_KEY, IndexMetadata,
};
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use lance_linalg::distance::DistanceType;
use lance_table::format::{IndexFile, IndexMetadata as TableIndexMetadata};
use object_store::path::Path;
use prost::Message;

use crate::{Error, Result};

/// Refuse a merge of IVF_VAMANA_* `segments` in `indices_dir` that cannot
/// become one segment, before the merge writes anything: a group of
/// IVF_VAMANA_* segments and others, segments over different codes or
/// metrics, graphs built under different parameters, as the segments' details
/// or their root files record them, or a segment whose files are gone. A
/// group with no IVF_VAMANA_* segment passes: its own rules apply.
pub async fn check_mergeable(
    object_store: &ObjectStore,
    indices_dir: &Path,
    segments: &[TableIndexMetadata],
) -> Result<()> {
    let Some(vamana) = segments
        .iter()
        .find(|segment| super::is_supported_segment(segment))
    else {
        return Ok(());
    };
    if let Some(other) = segments
        .iter()
        .find(|segment| !super::is_supported_segment(segment))
    {
        return Err(Error::invalid_input(format!(
            "Cannot merge IVF_VAMANA segment {} with segment {} of index version {}, which is \
             not an IVF_VAMANA segment",
            vamana.uuid, other.uuid, other.index_version
        )));
    }
    // An index whose graphs are all empty records its parameters only here.
    let mut recorded = None;
    for segment in segments {
        if let Some(params) = super::details::recorded_params(segment) {
            agree(&mut recorded, params, || {
                format!("segment {}", segment.uuid)
            })?;
        }
    }
    let mut roots = Vec::with_capacity(segments.len());
    for segment in segments {
        let dir = indices_dir.clone().join(segment.uuid.to_string());
        // `read_sources` opens the roots; nothing opens the storage before the merge writes.
        let storage = dir.clone().join(INDEX_AUXILIARY_FILE_NAME);
        if !object_store.exists(&storage).await? {
            return Err(Error::not_found(storage.to_string()));
        }
        roots.push(dir.join(INDEX_FILE_NAME));
    }
    if let Some(params) = read_sources(object_store, &roots).await?.params {
        agree(&mut recorded, params, || {
            "the graphs their root files record".to_string()
        })?;
    }
    Ok(())
}

/// Hold `params` to the parameters `agreed` so far, found in `place`: one
/// merged graph keeps one set.
fn agree(
    agreed: &mut Option<VamanaBuildParams>,
    params: VamanaBuildParams,
    place: impl FnOnce() -> String,
) -> Result<()> {
    match agreed {
        Some(expected) if *expected != params => Err(Error::invalid_input(format!(
            "Vamana build parameters mismatch while merging index segments: expected \
             {expected:?}, got {params:?} in {}",
            place()
        ))),
        Some(_) => Ok(()),
        None => {
            *agreed = Some(params);
            Ok(())
        }
    }
}

/// Write the root file of the segment merged into `index_dir`: the graph of
/// each partition of `ivf`, drawn over the partition's rows in the merged
/// storage `aux_reader` reads, with the parameters the root files `sources`
/// of the merged segments record.
pub async fn write_merged_root(
    object_store: &ObjectStore,
    index_dir: &Path,
    aux_reader: FileReader,
    ivf: &IvfModel,
    index_metadata: &IndexMetadata,
    sources: &[Path],
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<IndexFile> {
    let sources = read_sources(object_store, sources).await?;
    let loss = sources.loss().await?;
    // Without parameters every partition is empty, and no graph is drawn.
    let params = sources.params.unwrap_or_default();
    macro_rules! write_root {
        ($storage:expr) => {
            write_root(
                $storage,
                object_store,
                index_dir,
                ivf,
                index_metadata,
                params,
                loss,
                progress,
            )
            .await
        };
    }
    match index_metadata.index_type.as_str() {
        "IVF_VAMANA_FLAT" => {
            write_root!(IvfQuantizationStorage::<FlatQuantizer>::try_new(aux_reader, None).await?)
        }
        "IVF_VAMANA_SQ" => {
            write_root!(IvfQuantizationStorage::<ScalarQuantizer>::try_new(aux_reader, None).await?)
        }
        "IVF_VAMANA_PQ" => {
            write_root!(
                IvfQuantizationStorage::<ProductQuantizer>::try_new(aux_reader, None).await?
            )
        }
        other => Err(Error::index(format!(
            "Cannot write the root of merged IVF_VAMANA segments for index type {other}"
        ))),
    }
}

/// What the root files of the segments a merge unites record together.
struct Sources {
    /// The parameters every graph of the roots was built under, or `None`
    /// when no partition records any.
    params: Option<VamanaBuildParams>,
    /// Each root, opened.
    roots: Vec<(Path, FileReader)>,
}

impl Sources {
    /// The sum of the roots' k-means losses, `None` when a root records none.
    /// The segments share their centroids, so that their losses add up; rows
    /// a merge leaves out stay counted.
    async fn loss(&self) -> Result<Option<f64>> {
        let mut loss = Some(0.0);
        for (source, reader) in &self.roots {
            let corrupt = |message: String| Error::corrupt_file(source.clone(), message);
            let position = reader
                .metadata()
                .file_schema
                .metadata
                .get(IVF_METADATA_KEY)
                .ok_or_else(|| {
                    corrupt(format!(
                        "{IVF_METADATA_KEY} not found in an IVF_VAMANA segment's root"
                    ))
                })?
                .parse::<u32>()
                .map_err(|error| {
                    corrupt(format!("{IVF_METADATA_KEY} is not a position: {error}"))
                })?;
            let ivf = pb::Ivf::decode(reader.read_global_buffer(position).await?)
                .map_err(|error| corrupt(format!("the IVF buffer does not decode: {error}")))?;
            loss = loss.zip(ivf.loss).map(|(sum, loss)| sum + loss);
        }
        Ok(loss)
    }
}

/// Open the root files `sources` of the segments a merge unites, refusing
/// roots of different index types or metrics and graphs built under
/// different parameters.
async fn read_sources(object_store: &ObjectStore, sources: &[Path]) -> Result<Sources> {
    let scheduler = ScanScheduler::new(
        Arc::new(object_store.clone()),
        SchedulerConfig::max_bandwidth(object_store),
    );
    let mut index_metadata: Option<IndexMetadata> = None;
    let mut params = None;
    let mut roots = Vec::with_capacity(sources.len());
    for source in sources {
        let file = scheduler
            .open_file(source, &CachedFileSize::unknown())
            .await?;
        let reader = FileReader::try_open(
            file,
            None,
            Arc::default(),
            &LanceCache::no_cache(),
            FileReaderOptions::default(),
        )
        .await?;
        let corrupt = |message: String| Error::corrupt_file(source.clone(), message);
        let schema_metadata = &reader.metadata().file_schema.metadata;
        let get = |key: &str| {
            schema_metadata
                .get(key)
                .ok_or_else(|| corrupt(format!("{key} not found in an IVF_VAMANA segment's root")))
        };

        let metadata: IndexMetadata = serde_json::from_str(get(INDEX_METADATA_SCHEMA_KEY)?)
            .map_err(|error| {
                corrupt(format!(
                    "{INDEX_METADATA_SCHEMA_KEY} does not decode: {error}"
                ))
            })?;
        match &index_metadata {
            Some(expected)
                if expected.index_type != metadata.index_type
                    || expected.distance_type != metadata.distance_type =>
            {
                return Err(Error::invalid_input(format!(
                    "Cannot merge an {} segment under {} with an {} segment under {} in {source}",
                    expected.index_type,
                    expected.distance_type,
                    metadata.index_type,
                    metadata.distance_type
                )));
            }
            Some(_) => {}
            None => index_metadata = Some(metadata),
        }

        let partitions: Vec<String> =
            serde_json::from_str(get(VAMANA_METADATA_KEY)?).map_err(|error| {
                corrupt(format!(
                    "{VAMANA_METADATA_KEY} is not a list of strings: {error}"
                ))
            })?;
        for partition in partitions.iter().filter(|partition| !partition.is_empty()) {
            let metadata = serde_json::from_str::<VamanaMetadata>(partition).map_err(|error| {
                corrupt(format!(
                    "a partition's {VAMANA_METADATA_KEY} entry does not decode: {error}"
                ))
            })?;
            agree(&mut params, metadata.params, || source.to_string())?;
        }
        roots.push((source.clone(), reader));
    }
    Ok(Sources { params, roots })
}

/// Draw the graph of each partition of `ivf` over its rows in `storage`, one
/// partition at a time, and write the graphs as the root file in `index_dir`
/// with the IVF model and the metadata an index build writes.
#[allow(clippy::too_many_arguments)]
async fn write_root<Q: Quantization + 'static>(
    storage: IvfQuantizationStorage<Q>,
    object_store: &ObjectStore,
    index_dir: &Path,
    ivf: &IvfModel,
    index_metadata: &IndexMetadata,
    params: VamanaBuildParams,
    loss: Option<f64>,
    progress: Arc<dyn IndexBuildProgress>,
) -> Result<IndexFile> {
    let centroids = ivf.centroids.clone().ok_or_else(|| {
        Error::corrupt_file(
            index_dir.clone().join(INDEX_AUXILIARY_FILE_NAME),
            "the merged IVF_VAMANA storage carries no IVF centroids",
        )
    })?;
    let index_type = SupportedIvfIndexType::from_index_type_str(&index_metadata.index_type)
        .ok_or_else(|| {
            Error::index(format!(
                "Unsupported merged index type {}",
                index_metadata.index_type
            ))
        })?;
    let distance_type = DistanceType::try_from(index_metadata.distance_type.as_str())?;
    let num_partitions = ivf.num_partitions();

    let path = index_dir.clone().join(INDEX_FILE_NAME);
    let mut writer = file_versions::create_writer(
        storage.reader().metadata().version(),
        object_store.create(&path).await?,
        Schema::try_from(Vamana::schema().as_ref())?,
        FileWriterOptions::default(),
    )?;
    progress
        .stage_start("write_root_index", Some(1), "files")
        .await?;
    progress
        .stage_start(
            "rebuild_vamana_graph",
            Some(num_partitions as u64),
            "partitions",
        )
        .await?;

    let mut root_ivf = IvfModel::new(centroids, loss);
    let mut partition_metadata = Vec::with_capacity(num_partitions);
    for partition in 0..num_partitions {
        if ivf.partition_size(partition) == 0 {
            root_ivf.add_partition(0);
            partition_metadata.push(String::new());
        } else {
            let rows = storage.load_partition(partition, None).await?;
            let params = params.clone();
            let graph = spawn_cpu(move || Vamana::index_vectors(&rows, params)).await?;
            let batch = graph.to_batch()?;
            writer.write_batch(&batch).await?;
            root_ivf.add_partition(batch.num_rows() as u32);
            partition_metadata.push(serde_json::to_string(graph.metadata())?);
        }
        progress
            .stage_progress("rebuild_vamana_graph", partition as u64 + 1)
            .await?;
    }
    progress.stage_complete("rebuild_vamana_graph").await?;

    write_unified_ivf_and_index_metadata(&mut writer, &root_ivf, distance_type, index_type).await?;
    writer.add_schema_metadata(
        VAMANA_METADATA_KEY,
        serde_json::to_string(&partition_metadata)?,
    );
    let summary = writer.finish().await?;
    progress.stage_progress("write_root_index", 1).await?;
    progress.stage_complete("write_root_index").await?;
    Ok(IndexFile {
        path: INDEX_FILE_NAME.to_string(),
        size_bytes: summary.size_bytes,
        file_metadata_size_bytes: None,
    })
}
