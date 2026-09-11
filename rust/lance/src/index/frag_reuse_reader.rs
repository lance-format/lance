// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compose mapping readers along fragment lineage to reach live fragments.

use crate::Dataset;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, Result};
use lance_table::format::IndexMetadata;
use lance_table::system_index::frag_reuse::ledger::FragReuseLedger;
use roaring::RoaringBitmap;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

mod cache;
use cache::CachedMapping;

#[cfg(test)]
tokio::task_local! {
    static LEGACY_READER_ONLY: ();
}

#[cfg(test)]
fn check_reader_path() -> Result<()> {
    if LEGACY_READER_ONLY.try_with(|_| ()).is_ok() {
        return Err(Error::internal("V1 operation entered the new FRI reader"));
    }
    Ok(())
}

/// Filter and rewrite the index listing for querying under a tagged history.
///
/// The returned coverage bitmaps are snapshot-derived query inputs and must
/// never be persisted back to a manifest: stored segment metadata keeps its
/// original provenance.
pub(super) async fn load_indices(
    dataset: &Dataset,
    fri: &IndexMetadata,
    indices: &[IndexMetadata],
) -> Result<Arc<Vec<IndexMetadata>>> {
    let mapping = FragmentReuseIndex::open(dataset, fri).await?;
    let mut groups: HashMap<&str, Vec<(usize, &IndexMetadata)>> = HashMap::new();
    let mut result = vec![None; indices.len()];
    for (position, index) in indices.iter().enumerate() {
        if !super::index_is_usable(index) {
            continue;
        }
        if index.name == lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME {
            result[position] = Some(index.clone());
        } else {
            groups
                .entry(&index.name)
                .or_default()
                .push((position, index));
        }
    }
    for members in groups.into_values() {
        let mut supported = Vec::with_capacity(members.len());
        for (position, index) in members {
            if mapping.may_need_translation(index.fragment_bitmap.as_ref()) {
                let can_remap = if super::segment_has_vector_details(index) {
                    super::frag_reuse_remapping::vector_supports_batch_remapping(dataset, index)
                        .await?
                } else {
                    index
                        .index_details
                        .as_ref()
                        .and_then(|details| {
                            super::scalar::SCALAR_INDEX_PLUGIN_REGISTRY
                                .get_plugin_by_details(details)
                                .ok()
                        })
                        .is_some_and(|plugin| plugin.supports_batch_row_id_remapping())
                };
                if index.fragment_bitmap.is_none() || !can_remap {
                    continue;
                }
            }
            supported.push((position, index));
        }
        // Only usable segments may establish coverage. A destination needs all
        // contributing sources; every retained contributing segment must be queried.
        let provenance: Vec<_> = supported
            .iter()
            .map(|(_, index)| {
                index.fragment_bitmap.clone().ok_or_else(|| {
                    Error::internal(format!(
                        "query segment {} has no fragment coverage",
                        index.uuid
                    ))
                })
            })
            .collect::<Result<_>>()?;
        for ((position, index), coverage) in supported
            .into_iter()
            .zip(mapping.segment_coverage(&provenance))
        {
            if coverage.is_empty() {
                continue;
            }
            let mut index = index.clone();
            index.fragment_bitmap = Some(coverage);
            result[position] = Some(index);
        }
    }
    Ok(Arc::new(result.into_iter().flatten().collect()))
}

/// One segment's derived query-time inputs from the coverage backtrack.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentPlanParts {
    /// Live fragments this segment answers for in the current snapshot.
    pub coverage: RoaringBitmap,
    /// Fragments directly covered by other selected group members; translated
    /// paths entering them belong to those siblings.
    pub excluded: RoaringBitmap,
}

/// A validated FRI graph whose mapping payloads are opened only when needed.
pub struct FragmentReuseIndex {
    ledger: Arc<FragReuseLedger>,
    live_fragments: Arc<RoaringBitmap>,
    readers: Vec<Arc<CachedMapping>>,
}

impl FragmentReuseIndex {
    /// Open the history for this dataset snapshot without reading mapping labels.
    pub async fn open(dataset: &Dataset, index: &IndexMetadata) -> Result<Arc<Self>> {
        #[cfg(test)]
        check_reader_path()?;
        let ledger = cache::open_ledger(dataset, index).await?;
        if ledger.has_unsupported_transitions() {
            log::debug!(
                "FRI {} has unsupported mappings; only paths reaching live fragments can provide query coverage",
                index.uuid
            );
        }
        let mut readers = Vec::with_capacity(ledger.transitions().len());
        for transition in ledger.transitions() {
            readers.push(cache::open_mapping(dataset, transition).await?);
        }
        Ok(Arc::new(Self {
            ledger,
            live_fragments: dataset.fragment_bitmap.clone(),
            readers,
        }))
    }

    /// Whether segment metadata could require translation. V1 may have projected
    /// coverage to destinations without changing addresses stored in the index.
    /// Identity requires live-only coverage disjoint from supported lineage.
    /// Dead fragments may belong to an omitted unknown mapping; those must not
    /// get an identity remapper merely because they are absent from the graph.
    pub fn may_need_translation(&self, provenance: Option<&RoaringBitmap>) -> bool {
        provenance.is_none_or(|bitmap| {
            !bitmap.is_subset(self.live_fragments.as_ref())
                || bitmap
                    .iter()
                    .any(|fragment| self.ledger.contains_fragment(fragment))
        })
    }

    /// Resolve each live fragment backwards to the nearest available index coverage.
    /// A mapping requires complete source coverage; failure for one destination does
    /// not discard another destination's coverage. Every contributing segment is retained.
    pub fn segment_coverage(&self, provenance: &[RoaringBitmap]) -> Vec<RoaringBitmap> {
        self.segment_plans(provenance)
            .into_iter()
            .map(|plan| plan.coverage)
            .collect()
    }

    /// Resolve each segment's query coverage and sibling exclusions in one pass,
    /// so "direct coverage wins" is owned by one algorithm.
    ///
    /// Coverage: each live fragment backtracks to the nearest available index
    /// coverage. A mapping requires complete source coverage; failure for one
    /// destination does not discard another destination's coverage. Every
    /// contributing segment is retained.
    ///
    /// Exclusions: the fragments directly covered by other group members,
    /// derived from the same `direct` map the backtrack builds. This is
    /// deliberately an over-approximation of each segment's true sibling
    /// contention: fragments that are not on a path this segment translates are
    /// never checked per-hop, so extra members are inert. Do not "optimize"
    /// this to exact per-path sets.
    pub fn segment_plans(&self, provenance: &[RoaringBitmap]) -> Vec<SegmentPlanParts> {
        let mut direct: HashMap<u32, Vec<usize>> = HashMap::new();
        for (segment, bitmap) in provenance.iter().enumerate() {
            for fragment in bitmap {
                direct.entry(fragment).or_default().push(segment);
            }
        }
        // Direct coverage takes precedence: a fragment resolves to its direct
        // segments when present and to its producing transition otherwise. The
        // transition resolution is shared by all of that transition's
        // destinations; None records incomplete source coverage.
        let mut transitions: HashMap<usize, Option<Vec<usize>>> = HashMap::new();
        let mut coverage = vec![RoaringBitmap::new(); provenance.len()];
        let mut pending = Vec::new();
        for destination in self.live_fragments.iter() {
            let producer = match (direct.get(&destination), self.ledger.producer(destination)) {
                (Some(segments), _) => {
                    for &segment in segments {
                        coverage[segment].insert(destination);
                    }
                    continue;
                }
                (None, Some(producer)) => producer,
                (None, None) => continue,
            };
            pending.push((producer, false));
            while let Some((index, expanded)) = pending.pop() {
                if transitions.contains_key(&index) {
                    continue;
                }
                let sources = self.ledger.transitions()[index].sources();
                if !expanded {
                    pending.push((index, true));
                    pending.extend(sources.iter().filter_map(|source| {
                        let source = source.id as u32;
                        (!direct.contains_key(&source))
                            .then(|| self.ledger.producer(source))
                            .flatten()
                            .map(|producer| (producer, false))
                    }));
                    continue;
                }
                let mut segments = Vec::new();
                let complete = sources.iter().all(|source| {
                    let source = source.id as u32;
                    let contributors = direct.get(&source).or_else(|| {
                        self.ledger
                            .producer(source)
                            .and_then(|producer| transitions.get(&producer))
                            .and_then(|resolution| resolution.as_ref())
                    });
                    match contributors {
                        Some(contributors) => {
                            segments.extend_from_slice(contributors);
                            true
                        }
                        None => false,
                    }
                });
                let resolution = complete.then(|| {
                    segments.sort_unstable();
                    segments.dedup();
                    segments
                });
                transitions.insert(index, resolution);
            }
            if let Some(Some(segments)) = transitions.get(&producer) {
                for &segment in segments {
                    coverage[segment].insert(destination);
                }
            }
        }
        let group_direct: RoaringBitmap = direct.keys().copied().collect();
        provenance
            .iter()
            .zip(coverage)
            .map(|(own, coverage)| SegmentPlanParts {
                coverage,
                excluded: &group_direct - own,
            })
            .collect()
    }

    /// Remap physical row IDs through supported lineage, stopping at live fragments.
    /// Missing or deleted paths produce `None`. Input order and duplicates are preserved.
    pub async fn remap_row_ids(self: &Arc<Self>, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        self.remap_row_ids_excluding(row_ids, &RoaringBitmap::new())
            .await
    }

    /// Drop paths that enter fragments supplied by other selected index segments.
    /// Check after each mapping, not only at live destinations: branches may reconverge.
    /// Starting addresses are not excluded because legacy metadata can already describe
    /// projected coverage rather than the original addresses stored in an index.
    pub(super) async fn remap_row_ids_excluding(
        self: &Arc<Self>,
        row_ids: &[u64],
        excluded_fragments: &RoaringBitmap,
    ) -> Result<Vec<Option<u64>>> {
        let mut result = Vec::with_capacity(row_ids.len());
        for batch in row_ids.chunks(64 * 1024) {
            let mut output: Vec<_> = batch.iter().copied().map(Some).collect();
            let mut pending: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (position, address) in output.iter_mut().enumerate() {
                let current = RowAddress::from(batch[position]);
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
                let rows: Vec<_> = positions
                    .iter()
                    .map(|&position| {
                        output[position].ok_or_else(|| {
                            Error::internal("deleted address queued for FRI translation")
                        })
                    })
                    .collect::<Result<_>>()?;
                let rows = self.readers[index].remap_row_ids(&rows).await?;
                if rows.len() != positions.len() {
                    return Err(Error::internal(
                        "mapping reader changed translation batch length",
                    ));
                }
                for (position, address) in positions.into_iter().zip(rows) {
                    let address = address.filter(|row_id| {
                        !excluded_fragments.contains(RowAddress::from(*row_id).fragment_id())
                    });
                    output[position] = address;
                    if let Some(current) = address {
                        let current = RowAddress::from(current);
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
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::corrupt_file_named("FRI query", message)
}

async fn load_ledger(dataset: &Dataset, index: &IndexMetadata) -> Result<FragReuseLedger> {
    #[cfg(test)]
    check_reader_path()?;
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

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
    use crate::session::index_caches::IndexMetadataKey;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Int32Type;
    use arrow_array::{RecordBatch, RecordBatchIterator, UInt32Array, cast::AsArray};
    use lance_core::cache::LanceCache;
    use lance_core::utils::fragment_reuse::OrderedCompactionMapping;
    use lance_index::IndexType;
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use lance_index::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use lance_index::frag_reuse::stable_partition::MAPPING_FILE;
    use lance_index::scalar::IndexStore;
    use lance_index::scalar::ScalarIndexParams;
    use lance_index::scalar::lance_format::LanceIndexStore;
    use lance_table::format::pb::fragment_reuse_index_details::{
        FragmentDigest, InlineContent, StablePartition, Transition, transition,
    };
    use lance_table::format::{Fragment, pb};
    use lance_table::system_index::frag_reuse::ledger::Mapping;
    use lance_table::transaction::{Operation, Transaction};
    use prost::Message;
    use prost::encoding::WireType;
    use tokio::io::AsyncWriteExt;
    use uuid::Uuid;
    pub async fn fixture() -> Dataset {
        fixture_with_index(IndexType::BTree).await
    }

    pub async fn fixture_with_index(index_type: IndexType) -> Dataset {
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

    pub async fn prepare(dataset: &Dataset) -> (Transition, Vec<Fragment>) {
        let source_ids: Vec<u64> = dataset.fragments().iter().map(|f| f.id).collect();
        prepare_partition(dataset, &source_ids, 10).await
    }

    /// [`prepare`] over a subset of fragments: scans `source_ids` in order,
    /// alternates their rows across two uncommitted destinations numbered
    /// from `dest_base_id`, and writes the row map for that partition.
    pub async fn prepare_partition(
        dataset: &Dataset,
        source_ids: &[u64],
        dest_base_id: u64,
    ) -> (Transition, Vec<Fragment>) {
        let source_fragments: Vec<Fragment> = source_ids
            .iter()
            .map(|id| {
                dataset
                    .fragments()
                    .iter()
                    .find(|f| f.id == *id)
                    .unwrap()
                    .clone()
            })
            .collect();
        let batch = {
            let mut scan = dataset.scan();
            scan.with_fragments(source_fragments.clone());
            scan.try_into_batch().await.unwrap()
        };
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
            fragment.id = dest_base_id + i as u64;
        }
        let mut source_rows = Vec::new();
        let mut sources = Vec::new();
        for fragment in source_fragments.iter() {
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

    pub fn field(tag: u32, bytes: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        prost::encoding::encode_key(tag, WireType::LengthDelimited, &mut output);
        prost::encoding::encode_varint(bytes.len() as u64, &mut output);
        output.extend_from_slice(bytes);
        output
    }

    // Assemble a reader snapshot directly. Publishing rewrites and their FRI
    // deltas atomically belongs to the writer PR, not this test helper.
    pub async fn install(
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
    pub async fn persist_fixture(dataset: &mut Dataset, indices: Vec<IndexMetadata>) {
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

    // Deferred compaction is no longer in this list: on a tagged table it
    // appends an ordered-compaction transition to the tagged entry (see the
    // chained end-to-end test in `crate::index::frag_reuse`).
    #[rstest::rstest]
    #[case::eager_compaction("eager")]
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
            "eager" => crate::dataset::optimize::compact_files(
                &mut dataset,
                crate::dataset::optimize::CompactionOptions {
                    target_rows_per_fragment: 100,
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

    #[rstest::rstest]
    #[case::inline(false)]
    #[case::external(true)]
    #[tokio::test]
    async fn restore_preserves_snapshot_and_mapping_references(#[case] external: bool) {
        let mut dataset = fixture().await;
        let legacy = dataset.clone();
        let (transition, destinations) = prepare(&dataset).await;
        let source = u64::from(RowAddress::new_from_parts(
            transition.sources[0].id as u32,
            0,
        ));
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, external).await;
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        persist_fixture(&mut dataset, indices).await;
        let tagged = dataset.clone();
        let expected_mapping = FragmentReuseIndex::open(&tagged, &fri)
            .await
            .unwrap()
            .remap_row_ids(&[source])
            .await
            .unwrap();
        let mut latest = tagged.manifest.version;
        // Restore old data, then restore tagged data from the now-legacy latest snapshot.
        for mut target in [legacy, tagged] {
            let fragments = target.manifest.fragments.clone();
            let expected_indices = lance_table::io::manifest::read_manifest_indexes(
                &target.object_store,
                &target.manifest_location,
                &target.manifest,
            )
            .await
            .unwrap();
            target.restore().await.unwrap();
            latest += 1;
            assert_eq!(target.manifest.version, latest);
            assert_eq!(target.manifest.fragments, fragments);
            let restored_indices = lance_table::io::manifest::read_manifest_indexes(
                &target.object_store,
                &target.manifest_location,
                &target.manifest,
            )
            .await
            .unwrap();
            assert_eq!(restored_indices, expected_indices);
            let flag = lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
            assert_ne!(target.manifest.reader_feature_flags & flag, 0);
            assert_ne!(target.manifest.writer_feature_flags & flag, 0);
            assert_eq!(target.count_rows(None).await.unwrap(), 8);
            for value in 0..8 {
                assert_eq!(
                    target
                        .count_rows(Some(format!("i = {value}")))
                        .await
                        .unwrap(),
                    1
                );
            }
            if let Some(restored_fri) = restored_indices.iter().find(|index| index.uuid == fri.uuid)
            {
                assert_eq!(
                    FragmentReuseIndex::open(&target, restored_fri)
                        .await
                        .unwrap()
                        .remap_row_ids(&[source])
                        .await
                        .unwrap(),
                    expected_mapping
                );
            }
        }
    }

    #[tokio::test]
    async fn manifest_publication_rejects_tagged_history_without_flags() {
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        install(&mut dataset, content, destinations, false).await;
        let indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        let mut manifest = dataset.manifest.as_ref().clone();
        let flag = lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
        manifest.reader_feature_flags &= !flag;
        manifest.writer_feature_flags &= !flag;
        let error = crate::dataset::write_manifest_file(
            &dataset.object_store,
            dataset.commit_handler.as_ref(),
            &dataset.base,
            &mut manifest,
            Some(indices),
            &Default::default(),
            dataset.manifest_location.naming_scheme,
            None,
            false,
        )
        .await
        .unwrap_err();
        let error = Error::from(error);
        assert!(matches!(error, Error::CorruptFile { .. }), "{error}");
        assert!(
            error
                .to_string()
                .contains("tagged FRI metadata requires both"),
            "{error}"
        );
    }

    #[tokio::test]
    async fn clone_without_fri_flag_does_not_read_index_metadata() {
        let dataset = fixture().await;
        let mut location = dataset.manifest_location.clone();
        location.path = dataset.base.clone().join("missing.manifest");
        lance_table::system_index::frag_reuse::metadata::ensure_clone_supported(
            &dataset.object_store,
            &location,
            &dataset.manifest,
        )
        .await
        .unwrap();
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
        let mapping = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
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
    async fn public_reader_translates_while_unintegrated_indices_scan() {
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let sources: RoaringBitmap = transition.sources.iter().map(|f| f.id as u32).collect();
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;
        let reader = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
        assert_eq!(
            reader.segment_coverage(&[sources]),
            vec![[10, 11].into_iter().collect()]
        );
        let result = reader
            .remap_row_ids(&[
                u64::from(RowAddress::new_from_parts(0, 0)),
                u64::from(RowAddress::new_from_parts(10, 0)),
            ])
            .await
            .unwrap();
        assert!(result[0].is_some_and(|a| {
            dataset
                .fragment_bitmap
                .contains(RowAddress::from(a).fragment_id())
        }));
        assert_eq!(
            result[1],
            Some(u64::from(RowAddress::new_from_parts(10, 0)))
        );
        assert!(
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .any(|i| i.name == "i_idx")
        );
        assert_eq!(dataset.count_rows(Some("i = 2".into())).await.unwrap(), 1);
    }

    #[tokio::test]
    async fn legacy_compaction_does_not_open_the_new_reader() {
        LEGACY_READER_ONLY
            .scope((), async {
                let mut dataset = fixture().await;
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
                    .find(|i| i.name == FRAG_REUSE_INDEX_NAME)
                    .unwrap();
                assert_eq!(fri.index_version, 0);
                assert_eq!(dataset.manifest.reader_feature_flags & 512, 0);
                assert_eq!(dataset.manifest.writer_feature_flags & 512, 0);
                assert_eq!(dataset.count_rows(Some("i = 2".into())).await.unwrap(), 1);
                assert!(
                    dataset
                        .open_frag_reuse_index(&lance_index::metrics::NoOpMetricsCollector)
                        .await
                        .unwrap()
                        .is_some()
                );
            })
            .await;
    }

    #[tokio::test]
    async fn snapshots_share_history_without_sharing_live_fragments() {
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
        let (first, concurrent) = futures::try_join!(
            FragmentReuseIndex::open(&dataset, &fri),
            FragmentReuseIndex::open(&dataset, &fri),
        )
        .unwrap();
        assert!(Arc::ptr_eq(&first.ledger, &concurrent.ledger));
        assert!(Arc::ptr_eq(&first.readers[0], &concurrent.readers[0]));
        let before = dataset.index_cache.size_bytes().await;
        let source = u64::from(RowAddress::new_from_parts(0, 0));
        let expected = first.remap_row_ids(&[source]).await.unwrap();
        assert!(
            dataset.index_cache.size_bytes().await > before,
            "lazy counts must be charged to the cache"
        );
        let previous = dataset.clone();
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
        let current = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
        assert!(Arc::ptr_eq(&first.ledger, &current.ledger));
        assert!(Arc::ptr_eq(&first.readers[0], &current.readers[0]));
        assert!(Arc::ptr_eq(
            &first.live_fragments,
            &previous.fragment_bitmap
        ));
        assert!(Arc::ptr_eq(
            &current.live_fragments,
            &dataset.fragment_bitmap
        ));
        let added = (&*dataset.fragment_bitmap - &*previous.fragment_bitmap)
            .min()
            .unwrap();
        let added = u64::from(RowAddress::new_from_parts(added, 0));
        assert_eq!(first.remap_row_ids(&[added]).await.unwrap(), vec![None]);
        assert_eq!(
            current.remap_row_ids(&[added]).await.unwrap(),
            vec![Some(added)]
        );
        assert_eq!(current.remap_row_ids(&[source]).await.unwrap(), expected);
    }

    #[rstest::rstest]
    #[case::same_mappings(false)]
    #[case::pruned_history(true)]
    #[tokio::test]
    async fn new_ledger_reuses_retained_mappings(#[case] prune: bool) {
        let mut dataset = fixture().await;
        let (partition, destinations) = prepare(&dataset).await;
        let mut changed_row_addrs = Vec::new();
        roaring::RoaringTreemap::from_iter(
            (0..4).map(|row| u64::from(RowAddress::new_from_parts(900, row))),
        )
        .serialize_into(&mut changed_row_addrs)
        .unwrap();
        let ordered = Transition {
            sources: vec![FragmentDigest {
                id: 900,
                physical_rows: 4,
                num_deleted_rows: 0,
            }],
            destinations: vec![FragmentDigest {
                id: 0,
                physical_rows: 4,
                num_deleted_rows: 0,
            }],
            mapping: Some(transition::Mapping::OrderedCompaction(
                pb::fragment_reuse_index_details::OrderedCompaction { changed_row_addrs },
            )),
        };
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![ordered.clone(), partition.clone()],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;
        let first = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
        let old_address = u64::from(RowAddress::new_from_parts(900, 0));
        let expected = first.remap_row_ids(&[old_address]).await.unwrap();
        assert!(expected[0].is_some());

        let mut replacement = fri.clone();
        replacement.uuid = Uuid::new_v4();
        let retained = if prune {
            vec![partition]
        } else {
            vec![ordered, partition]
        };
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: retained,
        }
        .encode_to_vec();
        replacement.index_details = Some(Arc::new(prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: field(1, &content),
        }));
        let mut indices = crate::index::load_all_indices(&dataset)
            .await
            .unwrap()
            .as_ref()
            .clone();
        *indices.iter_mut().find(|i| i.uuid == fri.uuid).unwrap() = replacement.clone();
        // Model a committed pruned ledger; implementing cleanup is the writer PR's job.
        persist_fixture(&mut dataset, indices).await;
        let current = FragmentReuseIndex::open(&dataset, &replacement)
            .await
            .unwrap();
        assert!(!Arc::ptr_eq(&first.ledger, &current.ledger));
        let retained_partition = usize::from(!prune);
        assert!(Arc::ptr_eq(
            &first.readers[1],
            &current.readers[retained_partition]
        ));
        if !prune {
            assert!(Arc::ptr_eq(&first.readers[0], &current.readers[0]));
        }
        assert_eq!(
            current.remap_row_ids(&[old_address]).await.unwrap(),
            if prune { vec![None] } else { expected.clone() }
        );
        assert_eq!(first.remap_row_ids(&[old_address]).await.unwrap(), expected);
    }

    #[tokio::test]
    async fn mapping_cache_isolates_layout_and_storage_binding() {
        let mut dataset = fixture().await;
        let (mut partition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![partition.clone()],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;
        let first = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
        let mut rebound = dataset.clone();
        rebound.object_store = Arc::new(dataset.object_store.as_ref().clone());
        let second = FragmentReuseIndex::open(&rebound, &fri).await.unwrap();
        assert!(Arc::ptr_eq(&first.ledger, &second.ledger));
        assert!(!Arc::ptr_eq(&first.readers[0], &second.readers[0]));
        let source = u64::from(RowAddress::new_from_parts(0, 0));
        let original = first.remap_row_ids(&[source]).await.unwrap();
        assert_eq!(second.remap_row_ids(&[source]).await.unwrap(), original);

        partition.sources.reverse();
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![partition],
        }
        .encode_to_vec();
        let mut changed = fri.clone();
        changed.uuid = Uuid::new_v4();
        changed.index_details = Some(Arc::new(prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: field(1, &content),
        }));
        let changed = FragmentReuseIndex::open(&dataset, &changed).await.unwrap();
        assert!(!Arc::ptr_eq(&first.readers[0], &changed.readers[0]));
        assert_ne!(changed.remap_row_ids(&[source]).await.unwrap(), original);
        assert_eq!(first.remap_row_ids(&[source]).await.unwrap(), original);
    }

    #[tokio::test]
    async fn historical_version_zero_reuses_legacy_reader_after_append() {
        LEGACY_READER_ONLY
            .scope((), async {
                let dir = crate::utils::test::copy_test_data_to_tmp(
                    "fri_straddle_pre_6610/fri_straddle_dataset",
                )
                .unwrap();
                let uri = dir.std_path().to_str().unwrap();
                let mut dataset = Dataset::open(uri).await.unwrap();
                let metrics = lance_index::metrics::NoOpMetricsCollector;
                let original = dataset
                    .open_frag_reuse_index(&metrics)
                    .await
                    .unwrap()
                    .unwrap();
                let cached = dataset
                    .open_frag_reuse_index(&metrics)
                    .await
                    .unwrap()
                    .unwrap();
                assert!(Arc::ptr_eq(&original, &cached));
                let old_snapshot = dataset.clone();
                let old_rows = dataset.count_rows(None).await.unwrap();
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
                let current = dataset
                    .open_frag_reuse_index(&metrics)
                    .await
                    .unwrap()
                    .unwrap();
                assert!(Arc::ptr_eq(&original, &current));
                let reopened = Dataset::open(uri).await.unwrap();
                for snapshot in [&old_snapshot, &dataset, &reopened] {
                    let indices = snapshot.load_indices().await.unwrap();
                    let fri = indices
                        .iter()
                        .find(|i| i.name == FRAG_REUSE_INDEX_NAME)
                        .unwrap();
                    assert_eq!(fri.index_version, 0);
                    assert_eq!(snapshot.manifest.reader_feature_flags & 512, 0);
                    assert_eq!(snapshot.manifest.writer_feature_flags & 512, 0);
                    assert!(
                        snapshot
                            .open_frag_reuse_index(&metrics)
                            .await
                            .unwrap()
                            .is_some()
                    );
                }
                assert_eq!(old_snapshot.count_rows(None).await.unwrap(), old_rows);
                assert_eq!(reopened.count_rows(None).await.unwrap(), old_rows + 1);
            })
            .await;
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
        let mapping = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
        assert!(mapping.ledger.has_unsupported_transitions());
        assert!(mapping.may_need_translation(Some(&RoaringBitmap::from_iter([0, 1]))));
        assert!(mapping.ledger.transitions().is_empty());

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
        let mapping = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();

        let current = u64::from(RowAddress::new_from_parts(0, 0));
        assert_eq!(
            mapping
                .remap_row_ids(&[current, u64::from(RowAddress::new_from_parts(999, 0))])
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

    // A,B -> C,D -> E,F. Each destination contains one row. Coverage is
    // conservative for each mapping and requires both source fragments.
    async fn chain_reader() -> FragmentReuseIndex {
        let transitions = [(vec![0, 1], vec![2, 3]), (vec![2, 3], vec![4, 5])]
            .into_iter()
            .map(|(sources, destinations)| {
                let mut changed_row_addrs = Vec::new();
                roaring::RoaringTreemap::from_iter(
                    sources
                        .iter()
                        .map(|&source| u64::from(RowAddress::new_from_parts(source, 0))),
                )
                .serialize_into(&mut changed_row_addrs)
                .unwrap();
                let digest = |id: u32| FragmentDigest {
                    id: u64::from(id),
                    physical_rows: 1,
                    num_deleted_rows: 0,
                };
                Transition {
                    sources: sources.into_iter().map(digest).collect(),
                    destinations: destinations.into_iter().map(digest).collect(),
                    mapping: Some(transition::Mapping::OrderedCompaction(
                        pb::fragment_reuse_index_details::OrderedCompaction { changed_row_addrs },
                    )),
                }
            })
            .collect();
        let details = prost_types::Any {
            type_url: "/lance.table.FragmentReuseIndexDetails".into(),
            value: field(
                1,
                &InlineContent {
                    legacy_versions: vec![],
                    transitions,
                }
                .encode_to_vec(),
            ),
        };
        let ledger = FragReuseLedger::decode(1, &details, |_| async { panic!("inline history") })
            .await
            .unwrap();
        FragmentReuseIndex {
            ledger: Arc::new(ledger),
            live_fragments: Arc::new(RoaringBitmap::from_iter([4, 5])),
            // Coverage must never load mapping payloads.
            readers: vec![],
        }
    }

    #[rstest::rstest]
    #[case::nearest_coverage(vec![vec![0], vec![1], vec![2, 3]], vec![vec![], vec![], vec![4, 5]])]
    #[case::mixed_depths(vec![vec![0], vec![1], vec![2]], vec![vec![4, 5], vec![4, 5], vec![4, 5]])]
    #[case::independent_destination(vec![vec![0], vec![4]], vec![vec![], vec![4]])]
    #[case::direct_and_derived(vec![vec![0], vec![1], vec![4]], vec![vec![5], vec![5], vec![4]])]
    #[tokio::test]
    async fn coverage_resolves_each_destination_independently(
        #[case] provenance: Vec<Vec<u32>>,
        #[case] expected: Vec<Vec<u32>>,
    ) {
        let reader = chain_reader().await;
        let bitmaps = |values: Vec<Vec<u32>>| {
            values
                .into_iter()
                .map(RoaringBitmap::from_iter)
                .collect::<Vec<_>>()
        };
        assert_eq!(
            reader.segment_coverage(&bitmaps(provenance)),
            bitmaps(expected)
        );
    }

    #[tokio::test]
    async fn segment_plans_excludes_the_sibling_direct_union() {
        let reader = chain_reader().await;
        // Segments X, Y, Z with mixed direct and derived coverage.
        let provenance = vec![
            RoaringBitmap::from_iter([0]),
            RoaringBitmap::from_iter([1]),
            RoaringBitmap::from_iter([4]),
        ];
        let plans = reader.segment_plans(&provenance);
        // The thin wrapper and the combined pass agree on coverage.
        assert_eq!(
            reader.segment_coverage(&provenance),
            plans
                .iter()
                .map(|plan| plan.coverage.clone())
                .collect::<Vec<_>>()
        );
        // Exclusions equal the old sibling-union formula: everything any group
        // member covers directly, minus the segment's own provenance.
        let group: RoaringBitmap = provenance.iter().flatten().collect();
        for (plan, own) in plans.iter().zip(&provenance) {
            assert_eq!(plan.excluded, &group - own);
        }
        assert_eq!(plans[0].excluded, RoaringBitmap::from_iter([1, 4]));
        assert_eq!(plans[1].excluded, RoaringBitmap::from_iter([0, 4]));
        assert_eq!(plans[2].excluded, RoaringBitmap::from_iter([0, 1]));
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
        let mapping = Arc::new(FragmentReuseIndex {
            live_fragments: Arc::new(RoaringBitmap::from_iter([3, 9])),
            readers: ledger
                .transitions()
                .iter()
                .map(|t| {
                    let Mapping::OrderedCompaction(remap) = t.mapping() else {
                        unreachable!()
                    };
                    CachedMapping::uncached(Arc::new(OrderedCompactionMapping::new(remap.clone())))
                })
                .collect(),
            ledger: Arc::new(ledger),
        });
        let inputs = [0, 1, 2, 3, 9].map(|f| u64::from(RowAddress::new_from_parts(f, 1)));
        assert_eq!(
            mapping
                .remap_row_ids_excluding(&inputs, &RoaringBitmap::from_iter([2]))
                .await
                .unwrap(),
            vec![
                None,
                None,
                Some(u64::from(RowAddress::new_from_parts(3, 1))),
                Some(inputs[3]),
                Some(inputs[4])
            ],
        );
        assert_eq!(
            mapping.remap_row_ids(&inputs).await.unwrap(),
            vec![
                (!unknown_middle).then_some(u64::from(RowAddress::new_from_parts(3, 1))),
                (!unknown_middle).then_some(u64::from(RowAddress::new_from_parts(3, 1))),
                Some(u64::from(RowAddress::new_from_parts(3, 1))),
                Some(inputs[3]),
                Some(inputs[4])
            ]
        );
        assert_eq!(
            mapping
                .segment_coverage(&[RoaringBitmap::from_iter([0]), RoaringBitmap::from_iter([2])])
                .into_iter()
                .map(|coverage| coverage & mapping.live_fragments.as_ref())
                .collect::<Vec<_>>(),
            vec![RoaringBitmap::new(), RoaringBitmap::from_iter([3])]
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
        let mapping = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
        let error = mapping
            .remap_row_ids(&[u64::from(RowAddress::new_from_parts(0, 0))])
            .await
            .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("row-map total differs"));
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
                        stable_partition: None,
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
    #[tokio::test]
    async fn v0_writer_metadata_is_never_tagged() {
        use lance_table::system_index::frag_reuse::FragReuseIndexDetails;
        use lance_table::system_index::frag_reuse::metadata::is_tagged;

        let dataset = fixture().await;
        let details = FragReuseIndexDetails { versions: vec![] };
        let meta = crate::index::frag_reuse::build_frag_reuse_index_metadata(
            &dataset,
            None,
            details,
            RoaringBitmap::new(),
        )
        .await
        .unwrap();
        assert_eq!(meta.index_version, 0);
        assert!(!is_tagged(&meta));

        // Carrying an existing v0 entry forward preserves version 0.
        let details = FragReuseIndexDetails { versions: vec![] };
        let carried = crate::index::frag_reuse::build_frag_reuse_index_metadata(
            &dataset,
            Some(&meta),
            details,
            RoaringBitmap::new(),
        )
        .await
        .unwrap();
        assert_eq!(carried.index_version, 0);
        assert!(!is_tagged(&carried));
    }
}

#[cfg(test)]
#[path = "frag_reuse_reader_tests.rs"]
mod consumer_tests;
