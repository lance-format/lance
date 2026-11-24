// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector segment query contracts and validation before transaction publication.

use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use futures::{FutureExt, StreamExt, TryStreamExt, stream};
use lance_core::{Error, Result, cache::CacheKey};
use lance_index::{
    INDEX_FILE_NAME, INDEX_METADATA_SCHEMA_KEY, IndexMetadata, IndexType,
    metrics::NoOpMetricsCollector, vector::VectorIndex,
};
use lance_io::{
    scheduler::{ScanScheduler, SchedulerConfig},
    utils::{CachedFileSize, read_last_block, read_version},
};
use lance_linalg::distance::DistanceType;
use lance_table::{
    format::{IndexMetadata as TableIndexMetadata, Manifest},
    transaction::action::Action,
};
use uuid::Uuid;

use crate::{
    dataset::{Dataset, transaction::Operation},
    index::{
        DatasetIndexExt,
        frag_reuse::{OpenPurpose, SegmentRemappingPlan},
        vector::{
            open_index_file,
            utils::{get_vector_type, infer_vector_dim},
        },
    },
};

/// What a search ranks a vector segment by: its metric and its index type.
/// Merging the distances of segments that differ in either ranks part of the
/// data by the wrong one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorQueryContract {
    metric: DistanceType,
    /// `IndexType::Vector` for an index extension, which names no type this
    /// build knows; it matches only another such segment.
    index_type: IndexType,
}

impl VectorQueryContract {
    /// The contract of an opened segment, from the two accessors every
    /// vector index implements, legacy ones included.
    pub(crate) fn of_index(index: &dyn VectorIndex) -> Self {
        Self {
            metric: index.metric_type(),
            index_type: index.index_type(),
        }
    }

    /// The contract an index file of the current format records in its
    /// schema: the type the opener dispatches on, and the metric the opened
    /// index ranks by.
    fn from_file_metadata(metadata: &IndexMetadata) -> Result<Self> {
        let index_type = IndexType::try_from(metadata.index_type.as_str())?;
        // The opener opens these files only as one of the IVF types.
        if !index_type.is_vector() || index_type == IndexType::Vector {
            return Err(Error::index(format!(
                "Unsupported index type: {}",
                metadata.index_type
            )));
        }
        Ok(Self {
            metric: DistanceType::try_from(metadata.distance_type.as_str())?,
            index_type,
        })
    }
}

impl std::fmt::Display for VectorQueryContract {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.index_type {
            // Its `Display` says IVF_PQ.
            IndexType::Vector => write!(f, "metric {:?}, an index extension's type", self.metric),
            index_type => write!(f, "metric {:?}, type {index_type}", self.metric),
        }
    }
}

/// Refuses an index addition that would leave the segments of one index name
/// keyed on different columns, or its vector segments under different query
/// contracts. The index APIs compare the contracts of opened segments with
/// [`super::validate_vector_query_compatibility`]; a raw transaction reaches the
/// commit without them, so each commit attempt compares every segment the new
/// manifest keeps under each name the transaction adds to, retained ones
/// included, whatever their details say: a search merges all of them once it
/// picks the name, and one this build would leave out as too new, a newer
/// build merges. The caller of a raw transaction sets every field of the
/// segment metadata, so each contract is read from the segment's own index
/// file, and a file that cannot be read refuses the commit.
///
/// A search opens every segment of the name it picks as a vector segment, so
/// when some segments of a name have an index file, all of them need one: a
/// scalar segment on the same column, or a vector segment whose file is gone,
/// would fail every search. A name none of whose segments has one, such as a
/// zone map on a vector-typed column, is not a vector index.
pub async fn validate_committed_vector_segments(
    dataset: &Dataset,
    operation: &Operation,
    proposed_manifest: &Manifest,
    proposed_indices: &[TableIndexMetadata],
) -> Result<()> {
    let mut segments_by_name: BTreeMap<&str, Vec<&TableIndexMetadata>> = match operation {
        Operation::CreateIndex { new_indices, .. } => new_indices
            .iter()
            .map(|index| (index.name.as_str(), Vec::new()))
            .collect(),
        Operation::CompositeOperation(composite) => composite
            .iter_actions()
            .filter_map(|action| match action {
                Action::AddIndexSegment(segment) => Some((segment.name.as_str(), Vec::new())),
                _ => None,
            })
            .collect(),
        _ => return Ok(()),
    };
    if segments_by_name.is_empty() {
        return Ok(());
    }
    // Actions can introduce fields and base paths in the same commit, so
    // resolve their final segments through the manifest being published.
    let proposed_dataset = matches!(operation, Operation::CompositeOperation(_))
        .then(|| dataset.with_manifest_view(Arc::new(proposed_manifest.clone())));
    let dataset = proposed_dataset.as_ref().unwrap_or(dataset);
    let operation = operation.name();
    for segment in proposed_indices {
        if let Some(segments) = segments_by_name.get_mut(segment.name.as_str()) {
            segments.push(segment);
        }
    }

    for (name, segments) in segments_by_name {
        // A search picks a name by a segment keyed on the queried vector
        // column, then merges the other segments of that name with it.
        let keyed_on_a_vector_column = segments.iter().any(|segment| {
            segment
                .fields
                .first()
                .and_then(|&field_id| dataset.schema().field_by_id(field_id))
                .is_some_and(|field| infer_vector_dim(&field.data_type()).is_ok())
        });
        if segments.len() < 2 || !keyed_on_a_vector_column {
            continue;
        }
        let first = segments[0];
        if let Some(other) = segments
            .iter()
            .find(|segment| segment.keyed_fields() != first.keyed_fields())
        {
            let columns = |segment: &TableIndexMetadata| {
                segment
                    .keyed_fields()
                    .iter()
                    .map(|&field_id| {
                        dataset
                            .schema()
                            .field_path(field_id)
                            .unwrap_or_else(|_| format!("field {field_id}"))
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(Error::invalid_input(format!(
                "{operation}: segments {} (on {}) and {} (on {}) would both remain in index \
                 '{name}'; a search reads every segment of an index as keyed on the queried \
                 column, so all of them must be keyed on the same columns",
                first.uuid,
                columns(first),
                other.uuid,
                columns(other),
            )));
        }
        // Boxed and collected first: reading a segment is a large future, and
        // mapping these references inside the stream makes rustc report the
        // commit future's `Send` as "not general enough".
        let contracts = segments
            .into_iter()
            .map(|segment| {
                async move {
                    let contract =
                        vector_segment_contract(dataset, segment)
                            .await
                            .map_err(|error| {
                                Error::io(format!(
                                    "{operation}: cannot read vector segment {} to check that \
                                     index '{}' keeps one metric and index type: {error}",
                                    segment.uuid, segment.name
                                ))
                            })?;
                    Ok::<_, Error>((segment, contract))
                }
                .boxed()
            })
            .collect::<Vec<_>>();
        let mut contracts = stream::iter(contracts).buffered(dataset.object_store.io_parallelism());
        let mut reference: Option<(&TableIndexMetadata, VectorQueryContract)> = None;
        let mut without_index_file = None;
        while let Some((segment, contract)) = contracts.try_next().await? {
            match contract {
                None => {
                    without_index_file.get_or_insert(segment);
                }
                Some(contract) => match &reference {
                    None => reference = Some((segment, contract)),
                    Some((first, expected)) if *expected != contract => {
                        return Err(Error::invalid_input(format!(
                            "{operation}: vector segments {} ({expected}) and {} ({contract}) \
                             would both remain in index '{name}'; a search ranks every segment \
                             of an index by one metric and index type, so replace all of its \
                             segments in one commit",
                            first.uuid, segment.uuid,
                        )));
                    }
                    Some(_) => {}
                },
            }
            if let (Some((first, expected)), Some(other)) = (&reference, without_index_file) {
                return Err(Error::invalid_input(format!(
                    "{operation}: vector segment {} ({expected}) and segment {}, which has no \
                     vector index file, would both remain in index '{name}'; a search opens \
                     every segment of an index as a vector segment",
                    first.uuid, other.uuid,
                )));
            }
        }
    }
    Ok(())
}

// The enclosing cache namespace identifies the physical index file. A legacy
// marker routes the lookup through the active session's extension registry and
// snapshot-specific remapping plan.
struct VectorFileTypeKey;

impl CacheKey for VectorFileTypeKey {
    type ValueType = Option<IndexType>;

    fn key(&self) -> Cow<'_, str> {
        "type".into()
    }

    fn type_name() -> &'static str {
        "VectorIndexFileType"
    }
}

impl Dataset {
    /// Returns the physical index types of the query-visible segments on `column`,
    /// in the order of the supplied UUIDs. A UUID outside this snapshot's query
    /// listing or belonging to another column is an error.
    ///
    /// For current formats the type is read from the index file schema and cached
    /// without loading IVF centroids, partitions, or quantizers. Older formats and
    /// index extensions report their type through the ordinary query opener.
    pub async fn vector_index_types(&self, column: &str, uuids: &[Uuid]) -> Result<Vec<IndexType>> {
        if uuids.is_empty() {
            return Ok(Vec::new());
        }
        let field_id = self.schema().field_id(column)?;
        get_vector_type(self.schema(), column)?;
        // The listing enforces version and fragment reuse visibility before any
        // physical type cache hit. Row address translation cannot change a type.
        let indices = self.load_indices().await?;
        let mut by_uuid = HashMap::with_capacity(indices.len());
        for index in indices.iter() {
            by_uuid.entry(index.uuid).or_insert(index);
        }
        let read = |uuid: Uuid| {
            let segment = by_uuid.get(&uuid).copied();
            async move {
                let segment =
                    segment.ok_or_else(|| Error::index_not_found(format!("uuid={uuid}")))?;
                if segment.fields.first() != Some(&field_id) {
                    return Err(Error::invalid_input(format!(
                        "Vector index segment {uuid} does not belong to column '{column}'"
                    )));
                }
                self.query_vector_segment_type(column, segment).await
            }
        };
        // A legacy segment can require a full open. Limit the number of these a
        // single query can construct concurrently even on a high-parallelism store.
        stream::iter(uuids.iter().copied())
            .map(read)
            .buffered(self.object_store.io_parallelism().min(8))
            .try_collect()
            .await
    }

    async fn query_vector_segment_type(
        &self,
        column: &str,
        segment: &TableIndexMetadata,
    ) -> Result<IndexType> {
        let store = self.object_store_for_index(segment).await?;
        let path = self
            .indice_files_dir(segment)?
            .join(segment.uuid.to_string())
            .join(INDEX_FILE_NAME);
        let cache = self
            .metadata_cache
            .file_metadata_cache(&path)
            .with_key_prefix(&store.store_prefix);
        let file_type = cache
            .get_or_insert_with_key(VectorFileTypeKey, || async {
                let scheduler =
                    ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
                let size = segment
                    .files
                    .iter()
                    .flatten()
                    .find(|file| file.path == INDEX_FILE_NAME)
                    .map(|file| CachedFileSize::new(file.size_bytes))
                    .unwrap_or_else(CachedFileSize::unknown);
                let file = scheduler.open_file(&path, &size).await?;
                let schema = match lance_file::reader::FileReader::read_schema(&file).await {
                    Ok(schema) => schema,
                    Err(Error::VersionConflict {
                        major_version: 0,
                        minor_version: 0..=2,
                        ..
                    }) => return Ok(None),
                    Err(err) => return Err(err),
                };
                let raw = schema
                    .metadata
                    .get(INDEX_METADATA_SCHEMA_KEY)
                    .ok_or_else(|| Error::index("Index Metadata not found".to_owned()))?;
                let metadata = serde_json::from_str(raw)?;
                Ok(Some(
                    VectorQueryContract::from_file_metadata(&metadata)?.index_type,
                ))
            })
            .await?;
        match *file_type {
            Some(index_type) => Ok(index_type),
            None => Ok(self
                .open_vector_index_from_metadata_with_plan(
                    column,
                    segment,
                    None,
                    OpenPurpose::Query,
                    &NoOpMetricsCollector,
                )
                .await?
                .index_type()),
        }
    }
}

/// A vector segment's contract, read from its own index file. A file of the
/// current format records it in its schema, read the way the opener reads it
/// before loading any centroid; a file of an older format, or an index
/// extension's, is opened on the identity plan, since a staged segment is in
/// no listing a query plan covers, and a plan translates row addresses, never
/// a metric or a type.
///
/// `None` when the segment has no index file: the files it records name
/// none, or it records no files and the store has none.
pub(super) async fn vector_segment_contract(
    dataset: &Dataset,
    segment: &TableIndexMetadata,
) -> Result<Option<VectorQueryContract>> {
    // Not asked of the store: a missing key costs a request and its retries
    // per segment, and a store may answer it with a denial instead.
    if segment
        .files
        .as_ref()
        .is_some_and(|files| !files.iter().any(|file| file.path == INDEX_FILE_NAME))
    {
        return Ok(None);
    }
    let object_store = dataset.object_store_for_index(segment).await?;
    let index_file = dataset
        .indice_files_dir(segment)?
        .join(segment.uuid.to_string())
        .join(INDEX_FILE_NAME);
    let file_sizes = segment.file_size_map();
    let version = async {
        let reader = open_index_file(
            object_store.as_ref(),
            &index_file,
            INDEX_FILE_NAME,
            &file_sizes,
        )
        .await?;
        read_version(&read_last_block(reader.as_ref()).await?)
    };
    let version = match version.await {
        Ok(version) => version,
        Err(error) if error.is_not_found() => return Ok(None),
        Err(error) => return Err(error),
    };
    if matches!(version, (0, 3) | (2, _)) {
        let metadata = dataset
            .read_vector_index_metadata(&object_store, &index_file, &file_sizes)
            .await?;
        return VectorQueryContract::from_file_metadata(&metadata).map(Some);
    }
    let field_id = segment
        .fields
        .first()
        .ok_or_else(|| Error::invalid_input("the segment names no indexed column".to_string()))?;
    let column = dataset.schema().field_path(*field_id)?;
    let index = dataset
        .open_vector_index_from_metadata_with_plan(
            &column,
            segment,
            Some(&SegmentRemappingPlan::Identity),
            OpenPurpose::Maintenance,
            &NoOpMetricsCollector,
        )
        .await?;
    Ok(Some(VectorQueryContract::of_index(index.as_ref())))
}
