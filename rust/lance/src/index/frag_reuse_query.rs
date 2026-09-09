// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Query-time translation for tagged FRI histories. Mapping files are opened on demand.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use async_trait::async_trait;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, WeakLanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_index::frag_reuse::row_map::RowMapReader;
use lance_index::scalar::IndexStore;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::IndexMetadata;
use lance_table::system_index::frag_reuse::ledger::{FragReuseLedger, Mapping};
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;

use crate::Dataset;

const MAPPING_FILE: &str = "stable_partition.lance";

struct PartitionReader {
    store: LanceIndexStore,
    reader: OnceCell<RowMapReader>,
}

pub struct QueryFragReuseIndex {
    cache: WeakLanceCache,
    key: QueryKey,
    ledger: FragReuseLedger,
    live_fragments: RoaringBitmap,
    partitions: HashMap<usize, PartitionReader>,
}

impl DeepSizeOf for QueryFragReuseIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.live_fragments.serialized_size()
            + self.ledger.deep_size_of_children(context)
            + self
                .partitions
                .values()
                .map(|p| {
                    p.store.deep_size_of_children(context)
                        + p.reader.get().map_or(0, |reader| {
                            reader.counts().num_blocks()
                                * reader.counts().num_destinations() as usize
                                * 4
                        })
                })
                .sum::<usize>()
    }
}

#[derive(Clone)]
struct QueryKey(String);
impl CacheKey for QueryKey {
    type ValueType = QueryFragReuseIndex;
    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.0.as_str().into()
    }
    fn type_name() -> &'static str {
        "QueryFragReuseIndex"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.fri-query", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_str(&self.0);
    }
}

#[derive(Clone)]
struct VectorFormatKey(uuid::Uuid);

impl CacheKey for VectorFormatKey {
    type ValueType = bool;
    fn key(&self) -> std::borrow::Cow<'_, str> {
        self.0.to_string().into()
    }
    fn type_name() -> &'static str {
        "VectorBatchRemapping"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.vector-batch-remapping", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_fixed_bytes(self.0.as_bytes());
    }
}

/// Legacy vector file readers keep their existing V1 behavior and scan fallback.
pub async fn vector_supports_batch_remapping(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> Result<bool> {
    let supported = dataset
        .index_cache
        .get_or_insert_with_key(VectorFormatKey(index.uuid), || async {
            let path = dataset
                .indice_files_dir(index)?
                .join(index.uuid.to_string())
                .join(super::INDEX_FILE_NAME);
            let store = dataset.object_store_for_index(index).await?;
            let reader = super::vector::open_index_file(
                store.as_ref(),
                &path,
                super::INDEX_FILE_NAME,
                &index.file_size_map(),
            )
            .await?;
            let tail = lance_io::utils::read_last_block(reader.as_ref()).await?;
            Ok(matches!(
                lance_io::utils::read_version(&tail)?,
                (0, 3) | (2, _)
            ))
        })
        .await?;
    Ok(*supported)
}

impl QueryFragReuseIndex {
    pub(crate) async fn open(dataset: &Dataset, index: &IndexMetadata) -> Result<Arc<Self>> {
        let key = QueryKey(format!("{}:{}", dataset.manifest_location.path, index.uuid));
        dataset
            .index_cache
            .get_or_insert_with_key(key.clone(), || async {
                let ledger = load_ledger(dataset, index).await?;
                let mut partitions = HashMap::new();
                for (position, transition) in ledger.transitions().iter().enumerate() {
                    if let Mapping::StablePartition(reference) = transition.mapping() {
                        let base = match reference.base_id {
                            None => dataset.base.clone(),
                            Some(id) => dataset
                                .manifest
                                .base_paths
                                .get(&id)
                                .ok_or_else(|| {
                                    corrupt(format!(
                                        "mapping {} references missing base {id}",
                                        reference.map_id
                                    ))
                                })?
                                .extract_path(dataset.session.store_registry())?,
                        };
                        let directory = base.join("_fri").join(reference.map_id.as_str());
                        let cache = dataset.metadata_cache.file_metadata_cache(&directory);
                        let store = LanceIndexStore::new(
                            dataset.object_store(reference.base_id).await?,
                            directory,
                            Arc::new(cache),
                        )
                        .with_file_sizes(HashMap::from([(
                            MAPPING_FILE.to_string(),
                            reference.map_size_bytes,
                        )]));
                        partitions.insert(
                            position,
                            PartitionReader {
                                store,
                                reader: OnceCell::new(),
                            },
                        );
                    }
                }
                Ok(Self {
                    live_fragments: dataset.fragment_bitmap.as_ref().clone(),
                    ledger,
                    partitions,
                    key,
                    cache: WeakLanceCache::from(&dataset.index_cache),
                })
            })
            .await
    }

    /// Whether segment metadata could require translation. V1 may have projected
    /// coverage to destinations without changing addresses stored in the index.
    /// Only metadata disjoint from the whole lineage proves independence.
    pub(crate) fn may_need_translation(&self, provenance: Option<&RoaringBitmap>) -> bool {
        provenance.is_none_or(|bitmap| {
            bitmap
                .iter()
                .any(|fragment| self.ledger.contains_fragment(fragment))
        })
    }

    /// Coverage belongs to the logical index's union, but each contributing
    /// segment must be probed. Direct destination coverage takes precedence.
    pub(crate) fn segment_coverage(&self, provenance: &[RoaringBitmap]) -> Vec<RoaringBitmap> {
        let mut coverage = provenance.to_vec();
        for transition in self.ledger.transitions() {
            let sources: RoaringBitmap = transition.sources().iter().map(|f| f.id as u32).collect();
            let destinations: RoaringBitmap = transition
                .destinations()
                .iter()
                .map(|f| f.id as u32)
                .collect();
            let union = coverage
                .iter()
                .fold(RoaringBitmap::new(), |mut union, bitmap| {
                    union |= bitmap;
                    union
                });
            let complete = sources.is_subset(&union);
            let direct = &union & &destinations;
            for bitmap in &mut coverage {
                let contributes = !bitmap.is_disjoint(&sources);
                *bitmap -= &sources;
                if complete && contributes {
                    *bitmap |= &destinations - &direct;
                }
            }
        }
        coverage
    }

    async fn translate(
        self: &Arc<Self>,
        addresses: &[RowAddress],
    ) -> Result<Vec<Option<RowAddress>>> {
        let mut result = Vec::with_capacity(addresses.len());
        for batch in addresses.chunks(64 * 1024) {
            let mut output: Vec<_> = batch.iter().copied().map(Some).collect();
            let mut pending: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (position, address) in output.iter_mut().enumerate() {
                let current = batch[position];
                if self.live_fragments.contains(current.fragment_id()) {
                    continue;
                }
                if let Some(consumer) = self.ledger.consumer(current.fragment_id()) {
                    pending.entry(consumer).or_default().push(position);
                } else {
                    *address = None;
                }
            }
            while let Some((index, positions)) = pending.pop_first() {
                let transition = &self.ledger.transitions()[index];
                let mut rows: Vec<_> = positions.iter().map(|&position| output[position]).collect();
                match transition.mapping() {
                    Mapping::OrderedCompaction(remap) => {
                        for address in &mut rows {
                            *address = address.and_then(|current| {
                                remap.get(current.into()).flatten().map(RowAddress::from)
                            });
                        }
                    }
                    Mapping::StablePartition(_) => {
                        self.translate_partition(index, transition, &mut rows)
                            .await?
                    }
                }
                for (position, address) in positions.into_iter().zip(rows) {
                    output[position] = address;
                    if let Some(current) = address {
                        if self.live_fragments.contains(current.fragment_id()) {
                            continue;
                        }
                        if let Some(consumer) = self.ledger.consumer(current.fragment_id()) {
                            pending.entry(consumer).or_default().push(position);
                        } else {
                            output[position] = None;
                        }
                    }
                }
            }
            result.extend(output);
        }
        Ok(result)
    }
    async fn translate_partition(
        self: &Arc<Self>,
        index: usize,
        transition: &lance_table::system_index::frag_reuse::ledger::Transition,
        output: &mut [Option<RowAddress>],
    ) -> Result<()> {
        let mut start = 0_u64;
        let mut sources = HashMap::with_capacity(transition.sources().len());
        for source in transition.sources() {
            sources.insert(source.id as u32, (start, source.physical_rows));
            start = start
                .checked_add(source.physical_rows)
                .ok_or_else(|| corrupt("source physical row count overflow"))?;
        }
        let mut requests = Vec::new();
        for (position, address) in output.iter().enumerate() {
            let Some(address) = address else { continue };
            if let Some(&(base, rows)) = sources.get(&address.fragment_id()) {
                if u64::from(address.row_offset()) >= rows {
                    return Err(corrupt(format!(
                        "address {address} exceeds source length {rows}"
                    )));
                }
                requests.push((base + u64::from(address.row_offset()), position));
            }
        }
        if requests.is_empty() {
            return Ok(());
        }
        let partition = self
            .partitions
            .get(&index)
            .ok_or_else(|| corrupt("missing partition reader"))?;
        let was_open = partition.reader.get().is_some();
        let reader = partition
            .reader
            .get_or_try_init(|| async {
                let reader =
                    RowMapReader::open(partition.store.open_index_file(MAPPING_FILE).await?)
                        .await?;
                let counts = reader.counts();
                if counts.total_rows() != start
                    || counts.num_destinations() as usize != transition.destinations().len()
                {
                    return Err(corrupt("row-map dimensions differ from transition digests"));
                }
                for (label, destination) in transition.destinations().iter().enumerate() {
                    if u64::from(counts.total(label as u16)) != destination.physical_rows {
                        return Err(corrupt(format!(
                            "row-map total differs for destination {}",
                            destination.id
                        )));
                    }
                }
                Ok(reader)
            })
            .await?;
        if !was_open {
            // The cached metadata gains a counts matrix on first use. Reweigh
            // the entry so lazy loading cannot bypass the cache's byte budget.
            self.cache.insert_with_key(&self.key, self.clone()).await;
        }
        requests.sort_unstable_by_key(|&(row, _)| row);
        let mut remaining = requests.as_slice();
        while let Some(&(first, _)) = remaining.first() {
            let counts = reader.counts();
            let block = counts.block_of(first);
            let range = counts.block_range(block);
            let end = remaining.partition_point(|&(row, _)| row < range.end);
            let (batch, rest) = remaining.split_at(end);
            let labels = reader.block_labels(block).await?;
            if labels.len() as u64 != range.end - range.start {
                return Err(corrupt(format!(
                    "row-map block {block} has an unexpected label count"
                )));
            }
            // One sweep per touched block: bounded label memory and no
            // repeated prefix scans for dense index pages or duplicates.
            let mut counters = counts.counters_at_block(block);
            let mut requested = batch.iter().peekable();
            for (offset, label) in labels.iter().enumerate() {
                let translated = if let Some(label) = label {
                    let counter = counters
                        .get_mut(label as usize)
                        .ok_or_else(|| corrupt(format!("invalid row-map label {label}")))?;
                    let destination_offset = *counter;
                    *counter = counter
                        .checked_add(1)
                        .ok_or_else(|| corrupt("row-map count overflow"))?;
                    Some(RowAddress::new_from_parts(
                        transition.destinations()[label as usize].id as u32,
                        destination_offset,
                    ))
                } else {
                    None
                };
                let row = range.start + offset as u64;
                while let Some(&&(requested_row, position)) = requested.peek() {
                    if requested_row != row {
                        break;
                    }
                    output[position] = translated;
                    requested.next();
                }
            }
            if counters != counts.counters_at_block(block + 1) {
                return Err(corrupt(format!(
                    "row-map labels disagree with counts in block {block}"
                )));
            }
            remaining = rest;
        }

        Ok(())
    }
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::corrupt_file_named("FRI query", message)
}

async fn load_ledger(dataset: &Dataset, index: &IndexMetadata) -> Result<FragReuseLedger> {
    let details = index
        .index_details
        .as_ref()
        .ok_or_else(|| corrupt("missing FRI details"))?;
    FragReuseLedger::decode(index.index_version, details, |file| async move {
        let end = file
            .offset
            .checked_add(file.size)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| corrupt("external FRI range overflow"))?;
        let path = dataset
            .indice_files_dir(index)?
            .join(index.uuid.to_string())
            .join(file.path.as_str());
        dataset
            .object_store_for_index(index)
            .await?
            .open(&path)
            .await?
            .get_range(file.offset as usize..end)
            .await
            .map_err(Error::from)
    })
    .await
}

/// Applies the shared FRI history and limits translated rows to this segment's coverage.
pub struct QueryRowIdRemapper {
    mapping: Arc<QueryFragReuseIndex>,
    coverage: RoaringBitmap,
}

impl std::fmt::Debug for QueryRowIdRemapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryRowIdRemapper")
            .field("coverage", &self.coverage)
            .finish_non_exhaustive()
    }
}

impl QueryRowIdRemapper {
    pub(crate) fn new(mapping: Arc<QueryFragReuseIndex>, coverage: RoaringBitmap) -> Self {
        Self { mapping, coverage }
    }
}

#[async_trait]
impl lance_index::scalar::BatchRowIdRemapper for QueryRowIdRemapper {
    async fn remap_row_ids(&self, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        let addresses = row_ids
            .iter()
            .copied()
            .map(RowAddress::from)
            .collect::<Vec<_>>();
        Ok(self
            .mapping
            .translate(&addresses)
            .await?
            .into_iter()
            .map(|address| {
                address
                    .filter(|address| self.coverage.contains(address.fragment_id()))
                    .map(u64::from)
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
    use crate::session::index_caches::IndexMetadataKey;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Int32Type;
    use arrow_array::{RecordBatch, RecordBatchIterator, UInt32Array, cast::AsArray};
    #[cfg(feature = "geo")]
    use geo_types::line_string;
    #[cfg(feature = "geo")]
    use geoarrow_array::{GeoArrowArray, builder::LineStringBuilder};
    #[cfg(feature = "geo")]
    use geoarrow_schema::{Dimension, LineStringType};
    use lance_core::cache::LanceCache;
    use lance_index::IndexType;
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use lance_index::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use lance_index::metrics::NoOpMetricsCollector;
    use lance_index::scalar::ScalarIndexParams;
    use lance_table::format::pb::fragment_reuse_index_details::{
        FragmentDigest, InlineContent, StablePartition, Transition, transition,
    };
    use lance_table::format::{Fragment, pb};
    use lance_table::transaction::{Operation, Transaction};
    use prost::Message;
    use prost::encoding::WireType;
    use tokio::io::AsyncWriteExt;
    use uuid::Uuid;

    async fn fixture() -> Dataset {
        fixture_with_index(IndexType::BTree).await
    }

    async fn fixture_with_index(index_type: IndexType) -> Dataset {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(4))
            .await
            .unwrap();
        if index_type != IndexType::BTree {
            dataset
                .create_index(
                    &["i"],
                    index_type,
                    Some("i_idx".into()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
            return dataset;
        }
        let batch = dataset
            .scan()
            .with_row_id()
            .project_with_transform(&[("value", "i")])
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let reader = arrow_array::RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let params = ScalarIndexParams::default();
        let index = crate::index::create::CreateIndexBuilder::new(
            &mut dataset,
            &["i"],
            IndexType::BTree,
            &params,
        )
        .name("i_idx".into())
        .preprocessed_data(Box::new(reader))
        .execute_uncommitted()
        .await
        .unwrap();
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![index],
                        removed_indices: vec![],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        dataset
    }

    async fn prepare(dataset: &Dataset) -> (Transition, Vec<Fragment>) {
        let batch = dataset.scan().try_into_batch().await.unwrap();
        let values = batch["i"].as_primitive::<Int32Type>();
        let labels: Vec<_> = values.iter().map(|v| (v.unwrap() % 2) as u16).collect();
        let mut destinations = Vec::new();
        for label in 0..2 {
            let positions = UInt32Array::from(
                labels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &l)| (l == label).then_some(i as u32))
                    .collect::<Vec<_>>(),
            );
            let batch = RecordBatch::try_new(
                batch.schema(),
                batch
                    .columns()
                    .iter()
                    .map(|column| arrow::compute::take(column, &positions, None).unwrap())
                    .collect(),
            )
            .unwrap();
            let transaction = InsertBuilder::new(Arc::new(dataset.clone()))
                .with_params(&WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                })
                .execute_uncommitted(vec![batch])
                .await
                .unwrap();
            let Operation::Append { fragments } = transaction.operation else {
                unreachable!()
            };
            destinations.extend(fragments);
        }
        for (i, fragment) in destinations.iter_mut().enumerate() {
            fragment.id = 10 + i as u64;
        }
        let mut source_rows = Vec::new();
        let mut sources = Vec::new();
        for fragment in dataset.fragments().iter() {
            let deleted: Option<RoaringBitmap> = dataset
                .get_fragment(fragment.id as usize)
                .unwrap()
                .get_deletion_vector()
                .await
                .unwrap()
                .map(|v| v.iter().collect());
            let rows = fragment.physical_rows.unwrap() as u64;
            sources.push(FragmentDigest {
                id: fragment.id,
                physical_rows: rows,
                num_deleted_rows: deleted.as_ref().map_or(0, |d| d.len()),
            });
            source_rows.push(SourceRows {
                physical_rows: rows,
                deleted,
            });
        }
        let id = Uuid::new_v4();
        let store = LanceIndexStore::with_format_version(
            dataset.object_store.clone(),
            dataset.base.clone().join("_fri").join(id.to_string()),
            Arc::new(LanceCache::with_capacity(1024 * 1024)),
            lance_file::version::ConcreteFileVersion::V2_1,
        );
        let writer = store
            .new_index_file(MAPPING_FILE, RowMapWriter::schema())
            .await
            .unwrap();
        let mut writer = RowMapWriter::try_new_with_block_rows(writer, source_rows, 2, 3).unwrap();
        writer.append_labels(&labels).await.unwrap();
        let (file, _) = writer.finish().await.unwrap();
        let transition = Transition {
            sources,
            destinations: destinations
                .iter()
                .map(|f| FragmentDigest {
                    id: f.id,
                    physical_rows: f.physical_rows.unwrap() as u64,
                    num_deleted_rows: 0,
                })
                .collect(),
            mapping: Some(transition::Mapping::StablePartition(StablePartition {
                map_id: id.to_string(),
                map_size_bytes: file.size_bytes,
                base_id: None,
            })),
        };
        (transition, destinations)
    }

    fn field(tag: u32, bytes: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        prost::encoding::encode_key(tag, WireType::LengthDelimited, &mut output);
        prost::encoding::encode_varint(bytes.len() as u64, &mut output);
        output.extend_from_slice(bytes);
        output
    }

    // Assemble a reader snapshot directly. Publishing rewrites and their FRI
    // deltas atomically belongs to the writer PR, not this test helper.
    async fn install(
        dataset: &mut Dataset,
        content: Vec<u8>,
        destinations: Vec<Fragment>,
        external: bool,
    ) -> IndexMetadata {
        let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
        let uuid = Uuid::new_v4();
        let details = if external {
            let path = dataset
                .indices_dir()
                .join(uuid.to_string())
                .join("details.binpb");
            let mut writer = dataset.object_store.create(&path).await.unwrap();
            writer.write_all(&content).await.unwrap();
            writer.shutdown().await.unwrap();
            field(
                2,
                &pb::ExternalFile {
                    path: "details.binpb".into(),
                    offset: 0,
                    size: content.len() as u64,
                }
                .encode_to_vec(),
            )
        } else {
            field(1, &content)
        };
        let fri = IndexMetadata {
            uuid,
            fields: vec![],
            covering_fields: vec![],
            name: FRAG_REUSE_INDEX_NAME.into(),
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(destinations.iter().map(|f| f.id as u32).collect()),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "/lance.table.FragmentReuseIndexDetails".into(),
                value: details,
            })),
            index_version: 1,
            created_at: None,
            base_id: None,
            files: None,
        };
        indices.push(fri.clone());
        let manifest = Arc::make_mut(&mut dataset.manifest);
        manifest.reader_feature_flags |= lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
        manifest.writer_feature_flags |= lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
        Arc::make_mut(&mut dataset.manifest).fragments = destinations.into();
        dataset.fragment_bitmap = Arc::new(
            dataset
                .manifest
                .fragments
                .iter()
                .map(|f| f.id as u32)
                .collect(),
        );
        let key = IndexMetadataKey {
            version: dataset.manifest.version,
            store_identity: &dataset.object_store.store_prefix,
            e_tag: dataset.manifest_location.e_tag.as_deref(),
        };
        dataset
            .index_cache
            .insert_with_key(&key, Arc::new(indices))
            .await;
        fri
    }

    // Maintenance and clone reopen the manifest instead of using the query cache.
    // Persist the assembled fixture without requiring the future rewrite writer.
    async fn persist_fixture(dataset: &mut Dataset, indices: Vec<IndexMetadata>) {
        let mut manifest = dataset.manifest.as_ref().clone();
        manifest.version += 1;
        manifest.update_max_fragment_id();
        manifest.transaction_file = None;
        manifest.transaction_section = None;
        let location = crate::dataset::write_manifest_file(
            &dataset.object_store,
            dataset.commit_handler.as_ref(),
            &dataset.base,
            &mut manifest,
            Some(indices),
            &crate::dataset::ManifestWriteConfig::default(),
            dataset.manifest_location.naming_scheme,
            None,
            false,
        )
        .await
        .unwrap();
        *dataset = dataset.checkout_version(location.version).await.unwrap();
    }

    #[rstest::rstest]
    #[case::inline(false)]
    #[case::external(true)]
    #[tokio::test]
    async fn unknown_mapping_falls_back_to_scan(#[case] external: bool) {
        let mut dataset = fixture().await;
        let (mut transition, destinations) = prepare(&dataset).await;
        transition.mapping = None;
        let mut raw = transition.encode_to_vec();
        raw.extend(field(17, b"future mapping"));
        let fri = install(&mut dataset, field(2, &raw), destinations, external).await;
        let mapping = QueryFragReuseIndex::open(&dataset, &fri).await.unwrap();
        assert!(mapping.ledger.has_unsupported_transitions());
        assert!(mapping.ledger.transitions().is_empty());
        assert!(mapping.partitions.is_empty());

        let mut scan = dataset.scan();
        scan.filter("i = 2").unwrap();
        assert!(
            !scan
                .explain_plan(false)
                .await
                .unwrap()
                .contains("ScalarIndexQuery")
        );
        assert_eq!(scan.try_into_batch().await.unwrap().num_rows(), 1);
    }

    #[tokio::test]
    async fn projected_coverage_does_not_skip_address_translation() {
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        install(&mut dataset, content, destinations, false).await;
        let mut indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        let index = indices
            .iter_mut()
            .find(|index| index.name == "i_idx")
            .unwrap();
        index.fragment_bitmap = Some(dataset.fragment_bitmap.as_ref().clone());
        let key = IndexMetadataKey {
            version: dataset.manifest.version,
            store_identity: &dataset.object_store.store_prefix,
            e_tag: dataset.manifest_location.e_tag.as_deref(),
        };
        dataset
            .index_cache
            .insert_with_key(&key, Arc::new(indices))
            .await;
        for value in 0..8 {
            assert_eq!(
                dataset
                    .count_rows(Some(format!("i = {value}")))
                    .await
                    .unwrap(),
                1
            );
        }
    }

    #[tokio::test]
    async fn unrelated_fm_index_loads_with_tagged_history() {
        let batch = arrow_array::record_batch!(("text", Utf8, ["alpha", "beta"])).unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            "memory://",
            None,
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["text"],
                IndexType::Fm,
                Some("text_idx".into()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        let transition = Transition {
            sources: vec![FragmentDigest {
                id: 900,
                physical_rows: 1,
                num_deleted_rows: 0,
            }],
            destinations: vec![FragmentDigest {
                id: 901,
                physical_rows: 1,
                num_deleted_rows: 0,
            }],
            mapping: Some(transition::Mapping::StablePartition(StablePartition {
                map_id: Uuid::new_v4().to_string(),
                map_size_bytes: 100,
                base_id: None,
            })),
        };
        let fragments = dataset.manifest.fragments.as_ref().clone();
        let fri = install(
            &mut dataset,
            InlineContent {
                legacy_versions: vec![],
                transitions: vec![transition],
            }
            .encode_to_vec(),
            fragments,
            false,
        )
        .await;
        let index = dataset
            .load_index_by_name("text_idx")
            .await
            .unwrap()
            .unwrap();
        crate::index::scalar::open_scalar_index(
            &dataset,
            "text",
            &index,
            &lance_index::metrics::NoOpMetricsCollector,
        )
        .await
        .unwrap();
        assert_eq!(
            dataset
                .count_rows(Some("contains(text, 'alpha')".into()))
                .await
                .unwrap(),
            1
        );
        let mapping = QueryFragReuseIndex::open(&dataset, &fri).await.unwrap();
        assert!(
            mapping
                .partitions
                .values()
                .all(|partition| partition.reader.get().is_none())
        );
        let current = RowAddress::new_from_parts(0, 0);
        assert_eq!(
            mapping
                .translate(&[current, RowAddress::new_from_parts(999, 0)])
                .await
                .unwrap(),
            vec![Some(current), None]
        );
        let error = dataset
            .index_statistics(FRAG_REUSE_INDEX_NAME)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
    }

    #[rstest::rstest]
    #[case::inline(false)]
    #[case::external(true)]
    #[tokio::test]
    async fn scalar_queries_translate_deleted_sources_and_lazy_blocks(
        #[case] external: bool,
        #[values(
            IndexType::BTree,
            IndexType::Bitmap,
            IndexType::ZoneMap,
            IndexType::BloomFilter
        )]
        index_type: IndexType,
    ) {
        let mut dataset = fixture_with_index(index_type).await;
        dataset.delete("i = 3").await.unwrap();
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, external).await;
        let mapping = QueryFragReuseIndex::open(&dataset, &fri).await.unwrap();
        assert!(
            mapping
                .partitions
                .values()
                .all(|p| p.reader.get().is_none())
        );
        let index = dataset.load_index_by_name("i_idx").await.unwrap().unwrap();
        assert_eq!(
            index.fragment_bitmap.as_ref().unwrap(),
            dataset.fragment_bitmap.as_ref()
        );
        assert!(
            mapping
                .partitions
                .values()
                .all(|p| p.reader.get().is_none())
        );
        let plan = dataset
            .scan()
            .filter("i = 2")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(plan.contains("ScalarIndexQuery"), "{plan}");
        for value in 0..8 {
            assert_eq!(
                dataset
                    .count_rows(Some(format!("i = {value}")))
                    .await
                    .unwrap(),
                usize::from(value != 3)
            );
        }
        assert!(
            mapping
                .partitions
                .values()
                .all(|p| p.reader.get().is_some())
        );
        let metrics = lance_index::metrics::LocalMetricsCollector::default();
        crate::index::scalar::open_scalar_index(&dataset, "i", &index, &metrics)
            .await
            .unwrap();
        assert_eq!(
            metrics
                .index_loads
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        let mut other_snapshot = dataset.clone();
        other_snapshot.manifest_location.path =
            dataset.base.clone().join("_versions").join("999.manifest");
        crate::index::scalar::open_scalar_index(&other_snapshot, "i", &index, &metrics)
            .await
            .unwrap();
        assert_eq!(
            metrics
                .index_loads
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        let inputs = [
            RowAddress::new_from_parts(1, 3),
            RowAddress::new_from_parts(0, 2),
            RowAddress::new_from_parts(0, 2),
            RowAddress::new_from_parts(0, 3),
        ];
        assert_eq!(
            mapping.translate(&inputs).await.unwrap(),
            vec![
                Some(RowAddress::new_from_parts(11, 2)),
                Some(RowAddress::new_from_parts(10, 1)),
                Some(RowAddress::new_from_parts(10, 1)),
                None
            ]
        );
    }

    #[rstest::rstest]
    #[case::ngram(IndexType::NGram)]
    #[case::inverted(IndexType::Inverted)]
    #[tokio::test]
    async fn text_indices_translate_through_the_shared_remapper(#[case] index_type: IndexType) {
        let batch = arrow_array::record_batch!(
            ("i", Int32, [0, 1, 2, 3, 4, 5, 6, 7]),
            (
                "text",
                Utf8,
                [
                    Some("even"),
                    Some("odd"),
                    Some("even"),
                    Some("odd"),
                    None,
                    Some("odd"),
                    Some("even"),
                    Some("odd")
                ]
            )
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        if index_type == IndexType::Inverted {
            dataset
                .create_index(
                    &["text"],
                    index_type,
                    Some("text_idx".into()),
                    &lance_index::scalar::InvertedIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
        } else {
            dataset
                .create_index(
                    &["text"],
                    index_type,
                    Some("text_idx".into()),
                    &ScalarIndexParams::default(),
                    true,
                )
                .await
                .unwrap();
        }
        dataset.delete("i = 3").await.unwrap();
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        install(&mut dataset, content, destinations, false).await;
        let index = dataset
            .load_index_by_name("text_idx")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            index.fragment_bitmap.as_ref().unwrap(),
            dataset.fragment_bitmap.as_ref()
        );
        for _ in 0..2 {
            let mut scan = dataset.scan();
            if index_type == IndexType::Inverted {
                scan.full_text_search(lance_index::scalar::FullTextSearchQuery::new("even".into()))
                    .unwrap();
            } else {
                scan.filter("contains(text, 'even')").unwrap();
                let plan = scan.explain_plan(false).await.unwrap();
                assert!(plan.contains("ScalarIndexQuery"), "{plan}");
            }
            let result = scan.try_into_batch().await.unwrap();
            let actual = result
                .column_by_name("i")
                .unwrap()
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(actual, std::collections::BTreeSet::from([0, 2, 6]));
        }
    }

    #[tokio::test]
    async fn label_list_translates_values_and_null_rows() {
        let labels =
            arrow_array::ListArray::from_iter_primitive::<arrow_array::types::Int64Type, _, _>(
                (0..8).map(|i| {
                    if i == 4 {
                        None
                    } else {
                        Some(vec![Some(i % 2)])
                    }
                }),
            );
        let batch = RecordBatch::try_from_iter([
            (
                "i",
                Arc::new(arrow_array::Int32Array::from_iter_values(0..8)) as arrow_array::ArrayRef,
            ),
            ("labels", Arc::new(labels) as arrow_array::ArrayRef),
        ])
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema());
        let mut dataset = Dataset::write(
            reader,
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["labels"],
                IndexType::LabelList,
                Some("labels_idx".into()),
                &ScalarIndexParams::default(),
                true,
            )
            .await
            .unwrap();
        dataset.delete("i = 3").await.unwrap();
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        install(&mut dataset, content, destinations, false).await;
        for (predicate, expected) in [
            ("array_has_any(labels, [0])", vec![0, 2, 6]),
            ("NOT array_has_any(labels, [0])", vec![1, 5, 7]),
        ] {
            let mut scan = dataset.scan();
            scan.filter(predicate).unwrap();
            let plan = scan.explain_plan(false).await.unwrap();
            assert!(plan.contains("ScalarIndexQuery"), "{plan}");
            let result = scan.try_into_batch().await.unwrap();
            let actual = result
                .column_by_name("i")
                .unwrap()
                .as_primitive::<Int32Type>()
                .values()
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(actual, expected.into_iter().collect());
        }
    }

    #[cfg(feature = "geo")]
    #[tokio::test]
    async fn rtree_queries_translate_partition_addresses() {
        let geometry_type = LineStringType::new(Dimension::XY, Default::default());
        let mut geometry = LineStringBuilder::new(geometry_type.clone());
        for i in 0..8 {
            let line = line_string![(x: i as f64, y: 0.0), (x: i as f64, y: 1.0)];
            geometry
                .push_line_string((i != 7).then_some(&line))
                .unwrap();
        }
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![
                arrow_schema::Field::new("i", arrow_schema::DataType::Int32, false),
                geometry_type.to_field("geometry", true),
            ])),
            vec![
                Arc::new(arrow_array::Int32Array::from_iter_values(0..8)),
                geometry.finish().to_array_ref(),
            ],
        )
        .unwrap();
        let mut dataset = Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
            "memory://",
            Some(WriteParams {
                max_rows_per_file: 4,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        dataset
            .create_index(
                &["geometry"],
                IndexType::RTree,
                Some("geometry_idx".into()),
                &ScalarIndexParams::new("RTree".into()),
                true,
            )
            .await
            .unwrap();
        dataset.delete("i = 1").await.unwrap();
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        install(&mut dataset, content, destinations, false).await;
        let sql = "SELECT i FROM dataset WHERE ST_Intersects(geometry, ST_GeomFromText('LINESTRING (0 0.5, 6 0.5)')) ORDER BY i";
        let batches = dataset
            .sql(sql)
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let ids: Vec<_> = batches
            .iter()
            .flat_map(|batch| {
                batch["i"]
                    .as_primitive::<Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(ids, vec![0, 2, 3, 4, 5, 6]);
        let plan = dataset
            .sql(&format!("EXPLAIN {sql}"))
            .build()
            .await
            .unwrap()
            .into_batch_records()
            .await
            .unwrap();
        let plan = arrow::util::pretty::pretty_format_batches(&plan)
            .unwrap()
            .to_string();
        assert!(plan.contains("ScalarIndexQuery"), "{plan}");
        assert_eq!(
            dataset
                .count_rows(Some("geometry IS NULL".into()))
                .await
                .unwrap(),
            1
        );
    }

    #[tokio::test]
    async fn compaction_only_writer_stays_on_version_zero() {
        let mut dataset = fixture().await;
        for round in 0..2 {
            if round == 1 {
                let batch = dataset
                    .scan()
                    .limit(Some(1), None)
                    .unwrap()
                    .try_into_batch()
                    .await
                    .unwrap();
                dataset
                    .append(
                        RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                        None,
                    )
                    .await
                    .unwrap();
                dataset.delete("i = 1").await.unwrap();
            }
            crate::dataset::optimize::compact_files(
                &mut dataset,
                crate::dataset::optimize::CompactionOptions {
                    target_rows_per_fragment: 100,
                    defer_index_remap: true,
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();
            let indices = crate::index::load_all_indices(&dataset).await.unwrap();
            let fri = indices
                .iter()
                .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
                .unwrap();
            assert_eq!(fri.index_version, 0);
            let flag = lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
            assert_eq!(dataset.manifest.reader_feature_flags & flag, 0);
            assert_eq!(dataset.manifest.writer_feature_flags & flag, 0);
            let ledger = load_ledger(&dataset, fri).await.unwrap();
            assert!(
                ledger.transitions().iter().all(|transition| matches!(
                    transition.mapping(),
                    Mapping::OrderedCompaction(_)
                ))
            );
            let user_index = indices.iter().find(|index| index.name == "i_idx").unwrap();
            let remapper = super::super::frag_reuse::open_row_id_remapping(
                &dataset,
                user_index,
                &NoOpMetricsCollector,
            )
            .await
            .unwrap()
            .unwrap();
            assert!(matches!(
                remapper.1,
                lance_index::scalar::RowIdRemapping::InMemory(_)
            ));
            assert_eq!(dataset.count_rows(Some("i = 2".into())).await.unwrap(), 1);
        }
    }

    #[tokio::test]
    async fn released_legacy_fri_keeps_version_zero_on_append() {
        let dir =
            crate::utils::test::copy_test_data_to_tmp("fri_straddle_pre_6610/fri_straddle_dataset")
                .unwrap();
        let uri = dir.std_path().to_str().unwrap();
        let mut dataset = Dataset::open(uri).await.unwrap();
        let indices = crate::index::load_all_indices(&dataset).await.unwrap();
        let fri = indices
            .iter()
            .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
            .unwrap();
        assert_eq!(fri.index_version, 0);
        // Decode the released writer's actual Any, not a reserialized modern protobuf.
        let ledger = load_ledger(&dataset, fri).await.unwrap();
        assert!(!ledger.transitions().is_empty());
        let legacy = dataset
            .open_frag_reuse_index(&NoOpMetricsCollector)
            .await
            .unwrap()
            .unwrap();
        for transition in ledger.transitions() {
            for source in transition.sources() {
                for offset in 0..source.physical_rows {
                    let original = RowAddress::new_from_parts(source.id as u32, offset as u32);
                    let mut translated = Some(u64::from(original));
                    while let Some(current) = translated {
                        let Some(index) = ledger.consumer(RowAddress::from(current).fragment_id())
                        else {
                            break;
                        };
                        let Mapping::OrderedCompaction(remap) =
                            ledger.transitions()[index].mapping()
                        else {
                            unreachable!()
                        };
                        translated = remap.get(current).unwrap();
                    }
                    assert_eq!(translated, legacy.remap_row_id(original.into()));
                }
            }
        }
        let original_rows = dataset.count_rows(None).await.unwrap();
        let batch = dataset
            .scan()
            .limit(Some(1), None)
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                None,
            )
            .await
            .unwrap();
        let reopened = Dataset::open(uri).await.unwrap();
        assert_eq!(reopened.count_rows(None).await.unwrap(), original_rows + 1);
        let indices = crate::index::load_all_indices(&reopened).await.unwrap();
        let fri = indices
            .iter()
            .find(|index| index.name == FRAG_REUSE_INDEX_NAME)
            .unwrap();
        assert_eq!(fri.index_version, 0);
        let flag = lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
        assert_eq!(reopened.manifest.reader_feature_flags & flag, 0);
        assert_eq!(reopened.manifest.writer_feature_flags & flag, 0);
    }

    #[tokio::test]
    async fn legacy_vector_format_is_excluded_from_rewritten_coverage() {
        let dir = crate::utils::test::copy_test_data_to_tmp("v0.10.15/non_divisible_pq").unwrap();
        let mut dataset = Dataset::open(dir.std_path().to_str().unwrap())
            .await
            .unwrap();
        let indices = crate::index::load_all_indices(&dataset).await.unwrap();
        let vector = indices
            .iter()
            .find(|index| !index.fields.is_empty())
            .unwrap()
            .clone();
        assert!(
            !vector_supports_batch_remapping(&dataset, &vector)
                .await
                .unwrap()
        );
        let mut destinations = dataset.fragments().to_vec();
        let sources = destinations
            .iter()
            .map(|fragment| FragmentDigest {
                id: fragment.id,
                physical_rows: 1,
                num_deleted_rows: 0,
            })
            .collect();
        for fragment in &mut destinations {
            fragment.id += 10;
        }
        let transition = Transition {
            sources,
            destinations: destinations
                .iter()
                .map(|fragment| FragmentDigest {
                    id: fragment.id,
                    physical_rows: 1,
                    num_deleted_rows: 0,
                })
                .collect(),
            mapping: Some(transition::Mapping::StablePartition(StablePartition {
                map_id: Uuid::new_v4().to_string(),
                map_size_bytes: 1,
                base_id: None,
            })),
        };
        // No row-map file is written: the unsupported vector segment must fall back.
        install(
            &mut dataset,
            InlineContent {
                legacy_versions: vec![],
                transitions: vec![transition],
            }
            .encode_to_vec(),
            destinations,
            false,
        )
        .await;
        assert!(
            dataset
                .load_index_by_name(&vector.name)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(dataset.count_rows(Some("id = 0".into())).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn vector_partition_uses_shared_remapping_and_cached_reconstruction() {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .col(
                "vector",
                lance_datagen::array::rand_vec::<arrow_array::types::Float32Type>(4.into()),
            )
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(4))
            .await
            .unwrap();
        let params = crate::index::vector::VectorIndexParams::ivf_flat(
            1,
            lance_linalg::distance::DistanceType::L2,
        );
        dataset
            .create_index(
                &["vector"],
                IndexType::Vector,
                Some("vector_idx".into()),
                &params,
                true,
            )
            .await
            .unwrap();
        let original = dataset
            .scan()
            .filter("i = 2")
            .unwrap()
            .try_into_batch()
            .await
            .unwrap();
        let query = original
            .column_by_name("vector")
            .unwrap()
            .as_fixed_size_list()
            .value(0);
        let query = query.as_primitive::<arrow_array::types::Float32Type>();
        dataset.delete("i = 3").await.unwrap();
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;
        let mapping = QueryFragReuseIndex::open(&dataset, &fri).await.unwrap();
        assert!(
            mapping
                .partitions
                .values()
                .all(|partition| partition.reader.get().is_none())
        );
        for _ in 0..2 {
            let mut scan = dataset.scan();
            scan.nearest("vector", query, 1).unwrap();
            let plan = scan.explain_plan(false).await.unwrap();
            assert!(plan.contains("ANN"), "{plan}");
            let result = scan.try_into_batch().await.unwrap();
            assert_eq!(result.num_rows(), 1);
            assert_eq!(
                result
                    .column_by_name("i")
                    .unwrap()
                    .as_primitive::<Int32Type>()
                    .value(0),
                2
            );
        }
        assert!(
            mapping
                .partitions
                .values()
                .all(|partition| partition.reader.get().is_some())
        );
    }

    #[tokio::test]
    async fn future_index_version_rejects_filtered_reads_with_upgrade() {
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let mut fri = install(&mut dataset, content, destinations, false).await;
        fri.index_version = 2;
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .iter()
            .map(|index| {
                if index.uuid == fri.uuid {
                    fri.clone()
                } else {
                    index.clone()
                }
            })
            .collect::<Vec<_>>();
        persist_fixture(&mut dataset, indices).await;
        let error = dataset.count_rows(Some("i = 2".into())).await.unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("Please upgrade"));
    }

    #[rstest::rstest]
    #[case::eager_compaction("eager")]
    #[case::deferred_compaction("deferred")]
    #[case::statistics("statistics")]
    #[case::cleanup("cleanup")]
    #[case::shallow_clone("shallow")]
    #[case::deep_clone("deep")]
    #[tokio::test]
    async fn unsupported_maintenance_preserves_snapshot(#[case] operation: &str) {
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        persist_fixture(&mut dataset, indices).await;
        let version = dataset.manifest.version;
        let error = match operation {
            "eager" | "deferred" => crate::dataset::optimize::compact_files(
                &mut dataset,
                crate::dataset::optimize::CompactionOptions {
                    target_rows_per_fragment: 100,
                    defer_index_remap: operation == "deferred",
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap_err(),
            "statistics" => dataset
                .index_statistics(FRAG_REUSE_INDEX_NAME)
                .await
                .unwrap_err(),
            "cleanup" => crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
                .await
                .unwrap_err(),
            "shallow" => dataset
                .shallow_clone("memory://fri-shallow", version, None)
                .await
                .unwrap_err(),
            "deep" => dataset
                .deep_clone("memory://fri-deep", version, None)
                .await
                .unwrap_err(),
            _ => unreachable!(),
        };
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(
            error.to_string().to_lowercase().contains("upgrade"),
            "{error}"
        );
        assert_eq!(dataset.manifest.version, version);
        let indices = crate::index::load_all_indices(&dataset).await.unwrap();
        assert_eq!(
            indices.iter().find(|index| index.uuid == fri.uuid),
            Some(&fri)
        );
        assert_eq!(dataset.count_rows(Some("i = 2".into())).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn split_provenance_requires_all_sources_and_excludes_direct_coverage() {
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;
        let mapping = QueryFragReuseIndex::open(&dataset, &fri).await.unwrap();
        let source_a = RoaringBitmap::from_iter([0]);
        let source_b = RoaringBitmap::from_iter([1]);
        assert_eq!(
            mapping.segment_coverage(std::slice::from_ref(&source_a)),
            vec![RoaringBitmap::new()]
        );
        assert_eq!(
            mapping.segment_coverage(&[source_a.clone(), source_b.clone()]),
            vec![RoaringBitmap::from_iter([10, 11]); 2]
        );
        assert_eq!(
            mapping.segment_coverage(&[source_a, source_b, RoaringBitmap::from_iter([10])]),
            vec![
                RoaringBitmap::from_iter([11]),
                RoaringBitmap::from_iter([11]),
                RoaringBitmap::from_iter([10])
            ]
        );
    }
    #[tokio::test]
    async fn corrupt_counts_are_errors_instead_of_scan_fallback() {
        let mut dataset = fixture().await;
        let (mut transition, destinations) = prepare(&dataset).await;
        transition.destinations[0].physical_rows -= 1;
        transition.destinations[1].physical_rows += 1;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;
        let mapping = QueryFragReuseIndex::open(&dataset, &fri).await.unwrap();
        let error = mapping
            .translate(&[RowAddress::new_from_parts(0, 0)])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("row-map total differs"));
    }

    #[tokio::test]
    async fn unsupported_scalar_type_is_not_advertised_for_rewritten_fragments() {
        let mut dataset = fixture().await;
        // Only metadata is needed: the unsupported reader must never open it.
        let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
        indices[0].index_version = 0;
        indices[0].index_details = Some(Arc::new(
            prost_types::Any::from_msg(&lance_index::pb::FmIndexDetails::default()).unwrap(),
        ));
        let key = IndexMetadataKey {
            version: dataset.manifest.version,
            store_identity: &dataset.object_store.store_prefix,
            e_tag: dataset.manifest_location.e_tag.as_deref(),
        };
        dataset
            .index_cache
            .insert_with_key(&key, Arc::new(indices))
            .await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        install(&mut dataset, content, destinations, false).await;
        assert!(dataset.load_index_by_name("i_idx").await.unwrap().is_none());
        assert_eq!(dataset.count_rows(Some("i = 2".into())).await.unwrap(), 1);
    }
    #[rstest::rstest]
    #[case::complete(false)]
    #[case::unknown_middle_mapping(true)]
    #[tokio::test]
    async fn mixed_chain_keeps_independent_direct_coverage(#[case] unknown_middle: bool) {
        let digest = |id| FragmentDigest {
            id,
            physical_rows: 2,
            num_deleted_rows: 0,
        };
        let ordered = |source, destination| {
            let bitmap = roaring::RoaringTreemap::from_iter([
                u64::from(RowAddress::new_from_parts(source, 0)),
                u64::from(RowAddress::new_from_parts(source, 1)),
            ]);
            let mut changed_row_addrs = Vec::new();
            bitmap.serialize_into(&mut changed_row_addrs).unwrap();
            Transition {
                sources: vec![digest(u64::from(source))],
                destinations: vec![digest(destination)],
                mapping: Some(transition::Mapping::OrderedCompaction(
                    pb::fragment_reuse_index_details::OrderedCompaction { changed_row_addrs },
                )),
            }
        };
        let mut content = Vec::new();
        for mut transition in [ordered(2, 3), ordered(0, 1), ordered(1, 2)] {
            let is_unknown = unknown_middle && transition.sources[0].id == 1;
            if is_unknown {
                transition.mapping = None;
            }
            let mut raw = transition.encode_to_vec();
            if is_unknown {
                raw.extend(field(17, b"future mapping"));
            }
            content.extend(field(2, &raw));
        }
        let details = prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: field(1, &content),
        };
        let ledger = FragReuseLedger::decode(1, &details, |_| async { panic!("inline content") })
            .await
            .unwrap();
        let mapping = Arc::new(QueryFragReuseIndex {
            live_fragments: RoaringBitmap::from_iter([3, 9]),
            ledger,
            partitions: HashMap::new(),
            cache: WeakLanceCache::from(&LanceCache::with_capacity(0)),
            key: QueryKey("test".into()),
        });
        let inputs = [0, 1, 2, 3, 9].map(|f| RowAddress::new_from_parts(f, 1));
        assert_eq!(
            mapping.translate(&inputs).await.unwrap(),
            vec![
                (!unknown_middle).then_some(RowAddress::new_from_parts(3, 1)),
                (!unknown_middle).then_some(RowAddress::new_from_parts(3, 1)),
                Some(RowAddress::new_from_parts(3, 1)),
                Some(inputs[3]),
                Some(inputs[4])
            ]
        );
        assert_eq!(
            mapping
                .segment_coverage(&[RoaringBitmap::from_iter([0]), RoaringBitmap::from_iter([2])])
                .into_iter()
                .map(|coverage| coverage & &mapping.live_fragments)
                .collect::<Vec<_>>(),
            vec![RoaringBitmap::new(), RoaringBitmap::from_iter([3])]
        );
    }
    #[rstest::rstest]
    #[case::inline(false)]
    #[case::external(true)]
    #[tokio::test]
    async fn append_carries_future_fri_without_interpreting_it(#[case] external: bool) {
        let mut dataset = fixture().await;
        let batch = dataset.scan().try_into_batch().await.unwrap();
        let mut transition = Transition {
            sources: vec![FragmentDigest {
                id: 900,
                physical_rows: 1,
                num_deleted_rows: 0,
            }],
            destinations: vec![FragmentDigest {
                id: 901,
                physical_rows: 1,
                num_deleted_rows: 0,
            }],
            mapping: None,
        }
        .encode_to_vec();
        transition.extend(field(17, b"opaque future mapping reference"));
        let content = field(2, &transition);
        let uuid = Uuid::new_v4();
        let details_path = dataset
            .indices_dir()
            .join(uuid.to_string())
            .join("details.binpb");
        let details = if external {
            let mut writer = dataset.object_store.create(&details_path).await.unwrap();
            writer.write_all(&content).await.unwrap();
            writer.shutdown().await.unwrap();
            field(
                2,
                &pb::ExternalFile {
                    path: "details.binpb".into(),
                    offset: 0,
                    size: content.len() as u64,
                }
                .encode_to_vec(),
            )
        } else {
            field(1, &content)
        };
        let fri = IndexMetadata {
            uuid,
            fields: vec![],
            covering_fields: vec![],
            name: FRAG_REUSE_INDEX_NAME.into(),
            dataset_version: dataset.manifest.version,
            fragment_bitmap: Some(RoaringBitmap::from_iter([901])),
            index_details: Some(Arc::new(prost_types::Any {
                type_url: "/lance.table.FragmentReuseIndexDetails".into(),
                value: details,
            })),
            index_version: 2,
            created_at: None,
            base_id: None,
            files: None,
        };
        dataset
            .apply_commit(
                Transaction::new(
                    dataset.manifest.version,
                    Operation::CreateIndex {
                        new_indices: vec![fri.clone()],
                        removed_indices: vec![],
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap();
        let version = dataset.manifest.version;
        let flag = lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
        assert_eq!(dataset.manifest.reader_feature_flags & flag, flag);
        assert_eq!(dataset.manifest.writer_feature_flags & flag, flag);
        let error = dataset
            .apply_commit(
                Transaction::new(
                    version,
                    Operation::Rewrite {
                        groups: vec![],
                        rewritten_indices: vec![],
                        frag_reuse_index: None,
                    },
                    None,
                ),
                &Default::default(),
                &Default::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("Tagged FRI"));
        let error = dataset
            .shallow_clone("memory://fri-shallow", version, None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("relocation"));
        let error = dataset
            .deep_clone("memory://fri-deep", version, None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("relocation"));
        let error = crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("Upgrade"));
        assert_eq!(dataset.manifest.version, version);
        let key = QueryKey(format!("{}:{}", dataset.manifest_location.path, uuid));
        assert!(dataset.index_cache.get_with_key(&key).await.is_none());
        dataset
            .append(
                RecordBatchIterator::new(vec![Ok(batch.clone())], batch.schema()),
                Some(WriteParams {
                    mode: WriteMode::Append,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        assert_eq!(dataset.manifest.version, version + 1);
        assert_eq!(dataset.manifest.reader_feature_flags & flag, flag);
        assert_eq!(dataset.manifest.writer_feature_flags & flag, flag);
        assert!(dataset.index_cache.get_with_key(&key).await.is_none());
        let indices = lance_table::io::manifest::read_manifest_indexes(
            &dataset.object_store,
            &dataset.manifest_location,
            &dataset.manifest,
        )
        .await
        .unwrap();
        assert_eq!(indices.iter().find(|index| index.uuid == uuid), Some(&fri));
        if external {
            let bytes = dataset
                .object_store
                .open(&details_path)
                .await
                .unwrap()
                .get_range(0..content.len())
                .await
                .unwrap();
            assert_eq!(bytes.as_ref(), content);
        }
    }
}
