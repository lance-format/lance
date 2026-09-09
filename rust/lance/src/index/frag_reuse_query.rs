// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Query-time translation for tagged FRI histories. Mapping files are opened on demand.

use std::collections::HashMap;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch, UInt32Array, UInt64Array};
use async_trait::async_trait;
use bytes::{Buf, Bytes};
use futures::TryStreamExt;
use lance_core::cache::{CacheKey, CacheKeySchema, KeyBuilder, WeakLanceCache};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_index::frag_reuse::row_map::RowMapReader;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::{IndexFile, IndexReader, IndexStore, IndexWriter};
use lance_io::stream::{RecordBatchStream, RecordBatchStreamAdapter};
use lance_table::format::{IndexMetadata, pb};
use lance_table::system_index::frag_reuse::ledger::{FragReuseLedger, Mapping};
use prost::Message;
use prost::encoding::{DecodeContext, WireType, decode_key, decode_varint, skip_field};
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
    partitions: HashMap<usize, PartitionReader>,
}

impl DeepSizeOf for QueryFragReuseIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.ledger.raw_content().len()
            + self
                .ledger
                .transitions()
                .iter()
                .map(|t| {
                    std::mem::size_of_val(t)
                        + std::mem::size_of_val(t.sources())
                        + std::mem::size_of_val(t.destinations())
                        + match t.mapping() {
                            Mapping::OrderedCompaction(remap) => {
                                remap.deep_size_of_children(context)
                            }
                            Mapping::StablePartition(reference) => reference.map_id.capacity(),
                            Mapping::Unknown { .. } => 0,
                        }
                })
                .sum::<usize>()
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
                    ledger,
                    partitions,
                    key,
                    cache: WeakLanceCache::from(&dataset.index_cache),
                })
            })
            .await
    }

    /// Whether stored addresses cross any retained transition. Unsupported index
    /// readers can still serve directly covered fragments without translation.
    pub(crate) fn needs_translation(&self, provenance: Option<&RoaringBitmap>) -> bool {
        provenance.is_none_or(|bitmap| {
            self.ledger
                .transitions()
                .iter()
                .any(|t| t.sources().iter().any(|s| bitmap.contains(s.id as u32)))
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
            let complete = sources.is_subset(&union)
                && !matches!(transition.mapping(), Mapping::Unknown { .. });
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
        let mut output: Vec<_> = addresses.iter().copied().map(Some).collect();
        for (index, transition) in self.ledger.transitions().iter().enumerate() {
            match transition.mapping() {
                Mapping::OrderedCompaction(remap) => {
                    for address in &mut output {
                        if let Some(current) = address {
                            *address = remap
                                .get((*current).into())
                                .unwrap_or(Some((*current).into()))
                                .map(RowAddress::from);
                        }
                    }
                }
                Mapping::Unknown { .. } => {
                    let sources: RoaringBitmap =
                        transition.sources().iter().map(|s| s.id as u32).collect();
                    for address in &mut output {
                        if address.is_some_and(|a| sources.contains(a.fragment_id())) {
                            *address = None;
                        }
                    }
                }
                Mapping::StablePartition(_) => {
                    self.translate_partition(index, transition, &mut output)
                        .await?;
                }
            }
        }
        Ok(output)
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
    if index.index_version != 1 {
        return Err(Error::not_supported(format!(
            "unsupported tagged FRI index_version {}",
            index.index_version
        )));
    }
    let details = index
        .index_details
        .as_ref()
        .filter(|d| {
            d.type_url
                .ends_with("/lance.table.FragmentReuseIndexDetails")
        })
        .ok_or_else(|| corrupt("unexpected FRI details type"))?;
    // Extract InlineContent before prost drops future transition alternatives.
    let mut wire = Bytes::copy_from_slice(&details.value);
    let mut content = None;
    while wire.has_remaining() {
        let (tag, kind) = decode_key(&mut wire).map_err(|e| corrupt(e.to_string()))?;
        if tag == 1 || tag == 2 {
            if kind != WireType::LengthDelimited || content.is_some() {
                return Err(corrupt("invalid or repeated FRI content"));
            }
            let size = decode_varint(&mut wire).map_err(|e| corrupt(e.to_string()))?;
            if size > wire.remaining() as u64 {
                return Err(corrupt("truncated FRI content"));
            }
            content = Some((tag, wire.split_to(size as usize)));
        } else {
            skip_field(kind, tag, &mut wire, DecodeContext::default())
                .map_err(|e| corrupt(e.to_string()))?;
        }
    }
    let (tag, content) = content.ok_or_else(|| corrupt("missing FRI content"))?;
    let content = if tag == 1 {
        content
    } else {
        let file = pb::ExternalFile::decode(content).map_err(|e| corrupt(e.to_string()))?;
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
            .await?
    };
    FragReuseLedger::decode(index.index_version, content)
}

/// B-tree decode adapter. Translates bounded batches before the scalar index
/// evaluates predicates, preserving lazy row-map I/O and current coverage.
#[derive(Clone)]
pub struct TranslatedIndexStore {
    inner: Arc<dyn IndexStore>,
    mapping: Arc<QueryFragReuseIndex>,
    coverage: RoaringBitmap,
}

impl std::fmt::Debug for TranslatedIndexStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranslatedIndexStore")
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl DeepSizeOf for TranslatedIndexStore {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.inner.deep_size_of_children(context)
            + self.mapping.deep_size_of_children(context)
            + self.coverage.serialized_size()
    }
}

impl TranslatedIndexStore {
    pub(crate) fn new(
        inner: Arc<dyn IndexStore>,
        mapping: Arc<QueryFragReuseIndex>,
        coverage: RoaringBitmap,
    ) -> Self {
        Self {
            inner,
            mapping,
            coverage,
        }
    }
}

#[async_trait]
impl IndexStore for TranslatedIndexStore {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_arc(&self) -> Arc<dyn IndexStore> {
        Arc::new(self.clone())
    }
    fn io_parallelism(&self) -> usize {
        self.inner.io_parallelism()
    }
    async fn new_index_file(
        &self,
        name: &str,
        _schema: Arc<arrow_schema::Schema>,
    ) -> Result<Box<dyn IndexWriter>> {
        Err(Error::not_supported(format!(
            "cannot create {name} through a translated query store"
        )))
    }
    async fn open_index_file(&self, name: &str) -> Result<Arc<dyn IndexReader>> {
        Ok(Arc::new(TranslatedIndexReader {
            inner: self.inner.open_index_file(name).await?,
            mapping: self.mapping.clone(),
            coverage: self.coverage.clone(),
        }))
    }
    fn with_io_priority(&self, priority: u64) -> Arc<dyn IndexStore> {
        Arc::new(Self {
            inner: self.inner.with_io_priority(priority),
            ..self.clone()
        })
    }
    async fn copy_index_file(
        &self,
        name: &str,
        _destination: &dyn IndexStore,
    ) -> Result<IndexFile> {
        Err(Error::not_supported(format!(
            "copying translated B-tree file {name} requires rebuilding its page lookup"
        )))
    }

    async fn rename_index_file(&self, name: &str, new_name: &str) -> Result<IndexFile> {
        Err(Error::not_supported(format!(
            "cannot rename {name} to {new_name} through a translated query store"
        )))
    }
    async fn delete_index_file(&self, name: &str) -> Result<()> {
        Err(Error::not_supported(format!(
            "cannot delete {name} through a translated query store"
        )))
    }
    async fn list_files_with_sizes(&self) -> Result<Vec<IndexFile>> {
        self.inner.list_files_with_sizes().await
    }
}

#[derive(Clone)]
struct TranslatedIndexReader {
    inner: Arc<dyn IndexReader>,
    mapping: Arc<QueryFragReuseIndex>,
    coverage: RoaringBitmap,
}

impl TranslatedIndexReader {
    async fn translate(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let Ok(column) = batch.schema().index_of("ids") else {
            return Ok(batch);
        };
        // B-tree page files store physical row addresses in the `ids` column.
        let ids = batch
            .column_by_name("ids")
            .ok_or_else(|| Error::invalid_input("B-tree page is missing its ids column"))?
            .as_primitive_opt::<arrow_array::types::UInt64Type>()
            .ok_or_else(|| Error::invalid_input("B-tree row addresses must be UInt64"))?;
        let (valid_positions, addresses): (Vec<_>, Vec<_>) = ids
            .iter()
            .enumerate()
            .filter_map(|(position, address)| {
                address.map(|address| (position, RowAddress::from(address)))
            })
            .unzip();
        let translated = self.mapping.translate(&addresses).await?;
        let mut positions = Vec::with_capacity(batch.num_rows());
        let mut addresses = Vec::with_capacity(batch.num_rows());
        for (position, address) in valid_positions.into_iter().zip(translated) {
            if ids.is_valid(position)
                && let Some(address) = address
                && self.coverage.contains(address.fragment_id())
            {
                positions.push(position as u32);
                addresses.push(u64::from(address));
            }
        }
        let positions = UInt32Array::from(positions);
        let mut columns = batch
            .columns()
            .iter()
            .map(|array| arrow::compute::take(array, &positions, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        columns[column] = Arc::new(UInt64Array::from(addresses));
        Ok(RecordBatch::try_new(batch.schema(), columns)?)
    }
}

#[async_trait]
impl IndexReader for TranslatedIndexReader {
    async fn read_record_batch(&self, n: u64, batch_size: u64) -> Result<RecordBatch> {
        self.translate(self.inner.read_record_batch(n, batch_size).await?)
            .await
    }
    async fn read_range(
        &self,
        range: Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<RecordBatch> {
        self.translate(self.inner.read_range(range, projection).await?)
            .await
    }
    async fn read_range_stream(
        &self,
        range: Range<usize>,
        projection: Option<&[&str]>,
    ) -> Result<Pin<Box<dyn RecordBatchStream>>> {
        let stream = self.inner.read_range_stream(range, projection).await?;
        let schema = stream.schema();
        let reader = self.clone();
        let stream = stream.and_then(move |batch| {
            let reader = reader.clone();
            async move { reader.translate(batch).await }
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
    async fn read_global_buffer(&self, index: u32) -> Result<bytes::Bytes> {
        self.inner.read_global_buffer(index).await
    }
    async fn num_batches(&self, batch_size: u64) -> u32 {
        self.inner.num_batches(batch_size).await
    }
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }
    fn schema(&self) -> &lance_core::datatypes::Schema {
        self.inner.schema()
    }
    fn file_size_bytes(&self) -> Option<u64> {
        self.inner.file_size_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use crate::index::DatasetIndexExt;
    use crate::session::index_caches::IndexMetadataKey;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Int32Type;
    use lance_core::cache::LanceCache;
    use lance_index::IndexType;
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use lance_index::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use lance_index::scalar::ScalarIndexParams;
    use lance_table::format::Fragment;
    use lance_table::format::pb::fragment_reuse_index_details::{
        FragmentDigest, InlineContent, StablePartition, Transition, transition,
    };
    use lance_table::transaction::{Operation, Transaction};
    use tokio::io::AsyncWriteExt;
    use uuid::Uuid;

    async fn fixture() -> Dataset {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset(FragmentCount::from(2), FragmentRowCount::from(4))
            .await
            .unwrap();
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
                vec![arrow::compute::take(values, &positions, None).unwrap()],
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
            encoding: Some(transition::Encoding::StablePartition(StablePartition {
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

    #[rstest::rstest]
    #[case::inline(false)]
    #[case::external(true)]
    #[tokio::test]
    async fn btree_queries_translate_deleted_sources_and_lazy_blocks(#[case] external: bool) {
        let mut dataset = fixture().await;
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

    #[tokio::test]
    async fn unknown_encoding_falls_back_to_scan_without_opening_payload() {
        let mut dataset = fixture().await;
        let (mut transition, destinations) = prepare(&dataset).await;
        transition.encoding = None;
        let mut raw = transition.encode_to_vec();
        raw.extend(field(17, b"unknown-file-reference"));
        let fri = install(&mut dataset, field(2, &raw), destinations, false).await;
        let mapping = QueryFragReuseIndex::open(&dataset, &fri).await.unwrap();
        assert!(mapping.partitions.is_empty());
        let index = dataset.load_index_by_name("i_idx").await.unwrap().unwrap();
        assert!(index.fragment_bitmap.as_ref().unwrap().is_empty());
        let plan = dataset
            .scan()
            .filter("i = 2")
            .unwrap()
            .explain_plan(false)
            .await
            .unwrap();
        assert!(!plan.contains("ScalarIndexQuery"), "{plan}");
        assert_eq!(dataset.count_rows(Some("i = 2".into())).await.unwrap(), 1);
        assert_eq!(
            mapping.segment_coverage(&[
                RoaringBitmap::from_iter([0, 1]),
                RoaringBitmap::from_iter([10])
            ]),
            vec![RoaringBitmap::new(), RoaringBitmap::from_iter([10])]
        );
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
            prost_types::Any::from_msg(&lance_index::pbold::BitmapIndexDetails::default()).unwrap(),
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
    #[tokio::test]
    async fn mixed_chain_keeps_independent_direct_coverage() {
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
                encoding: Some(transition::Encoding::OrderedCompaction(
                    pb::fragment_reuse_index_details::OrderedCompaction { changed_row_addrs },
                )),
            }
        };
        let mut opaque = Transition {
            sources: vec![digest(1)],
            destinations: vec![digest(2)],
            encoding: None,
        }
        .encode_to_vec();
        opaque.extend(field(17, b"future encoding"));
        let mut content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![ordered(2, 3), ordered(0, 1)],
        }
        .encode_to_vec();
        content.extend(field(2, &opaque));
        let mapping = Arc::new(QueryFragReuseIndex {
            ledger: FragReuseLedger::decode(1, content.into()).unwrap(),
            partitions: HashMap::new(),
            cache: WeakLanceCache::from(&LanceCache::with_capacity(0)),
            key: QueryKey("test".into()),
        });
        let inputs = [0, 1, 2, 3, 9].map(|f| RowAddress::new_from_parts(f, 1));
        assert_eq!(
            mapping.translate(&inputs).await.unwrap(),
            vec![
                None,
                None,
                Some(RowAddress::new_from_parts(3, 1)),
                Some(inputs[3]),
                Some(inputs[4])
            ]
        );
        assert_eq!(
            mapping
                .segment_coverage(&[RoaringBitmap::from_iter([0]), RoaringBitmap::from_iter([2])]),
            vec![RoaringBitmap::new(), RoaringBitmap::from_iter([3])]
        );
    }
}
