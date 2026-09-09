// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compose mapping readers along fragment lineage to reach live fragments.

use crate::Dataset;
use lance_core::utils::address::RowAddress;
use lance_core::utils::fragment_reuse::{MappingReader, OrderedCompactionMapping};
use lance_core::{Error, Result};
use lance_index::frag_reuse::stable_partition::{
    FragmentLayout, MAPPING_FILE, StablePartitionMapping,
};
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_table::format::IndexMetadata;
use lance_table::system_index::frag_reuse::ledger::{FragReuseLedger, Mapping};
use roaring::RoaringBitmap;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

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

/// A validated FRI graph whose mapping payloads are opened only when needed.
pub struct FragmentReuseIndex {
    ledger: FragReuseLedger,
    live_fragments: RoaringBitmap,
    readers: Vec<Arc<dyn MappingReader>>,
}

impl FragmentReuseIndex {
    /// Open the history for this dataset snapshot without reading mapping labels.
    pub async fn open(dataset: &Dataset, index: &IndexMetadata) -> Result<Arc<Self>> {
        #[cfg(test)]
        check_reader_path()?;
        let ledger = load_ledger(dataset, index).await?;
        let mut readers: Vec<Arc<dyn MappingReader>> =
            Vec::with_capacity(ledger.transitions().len());
        for transition in ledger.transitions() {
            let reader: Arc<dyn MappingReader> = match transition.mapping() {
                Mapping::OrderedCompaction(remap) => Arc::new(OrderedCompactionMapping::new(
                    remap.clone(),
                    transition.sources().iter().map(|f| f.id as u32).collect(),
                    transition
                        .destinations()
                        .iter()
                        .map(|f| f.id as u32)
                        .collect(),
                )),
                Mapping::StablePartition(reference) => {
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
                    let layout = |fragments: &[lance_table::format::pb::fragment_reuse_index_details::FragmentDigest]| {
                        fragments.iter().map(|f| FragmentLayout { id: f.id as u32, physical_rows: f.physical_rows }).collect()
                    };
                    Arc::new(StablePartitionMapping::try_new(
                        Arc::new(store),
                        layout(transition.sources()),
                        layout(transition.destinations()),
                    )?)
                }
            };
            readers.push(reader);
        }
        Ok(Arc::new(Self {
            ledger,
            live_fragments: dataset.fragment_bitmap.as_ref().clone(),
            readers,
        }))
    }

    /// Whether segment metadata could require translation. V1 may have projected
    /// coverage to destinations without changing addresses stored in the index.
    /// Only metadata disjoint from the whole lineage proves independence.
    pub fn may_need_translation(&self, provenance: Option<&RoaringBitmap>) -> bool {
        provenance.is_none_or(|bitmap| {
            bitmap
                .iter()
                .any(|fragment| self.ledger.contains_fragment(fragment))
        })
    }

    /// Coverage belongs to the logical index's union, but each contributing
    /// segment must be probed. Direct destination coverage takes precedence.
    pub fn segment_coverage(&self, provenance: &[RoaringBitmap]) -> Vec<RoaringBitmap> {
        let mut coverage = provenance.to_vec();
        for (transition, reader) in self.ledger.transitions().iter().zip(&self.readers) {
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
            let mapped = reader.coverage(&union);
            let direct = &union & &destinations;
            for bitmap in &mut coverage {
                let contributes = !bitmap.is_disjoint(&sources);
                *bitmap -= &sources;
                if contributes {
                    *bitmap |= &mapped - &direct;
                }
            }
        }
        for bitmap in &mut coverage {
            *bitmap &= &self.live_fragments;
        }
        coverage
    }

    /// Translate a batch through supported lineage, stopping at live fragments.
    /// Missing or deleted paths produce `None`. Input order and duplicates are preserved.
    pub async fn translate(
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
                let rows: Vec<_> = positions
                    .iter()
                    .map(|&position| {
                        output[position].ok_or_else(|| {
                            Error::internal("deleted address queued for FRI translation")
                        })
                    })
                    .collect::<Result<_>>()?;
                let rows = self.readers[index].translate(&rows).await?;
                if rows.len() != positions.len() {
                    return Err(Error::internal(
                        "mapping reader changed translation batch length",
                    ));
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
mod tests {
    use super::*;
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use crate::index::{DatasetIndexExt, DatasetIndexInternalExt};
    use crate::session::index_caches::IndexMetadataKey;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Int32Type;
    use arrow_array::{RecordBatch, RecordBatchIterator, UInt32Array, cast::AsArray};
    use lance_core::cache::LanceCache;
    use lance_index::IndexType;
    use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
    use lance_index::frag_reuse::row_map::{RowMapWriter, SourceRows};
    use lance_index::scalar::IndexStore;
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
            .translate(&[
                RowAddress::new_from_parts(0, 0),
                RowAddress::new_from_parts(10, 0),
            ])
            .await
            .unwrap();
        assert!(result[0].is_some_and(|a| dataset.fragment_bitmap.contains(a.fragment_id())));
        assert_eq!(result[1], Some(RowAddress::new_from_parts(10, 0)));
        assert!(
            dataset
                .load_indices()
                .await
                .unwrap()
                .iter()
                .all(|i| i.name == FRAG_REUSE_INDEX_NAME)
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
            live_fragments: RoaringBitmap::from_iter([3, 9]),
            readers: ledger
                .transitions()
                .iter()
                .map(|t| {
                    let Mapping::OrderedCompaction(remap) = t.mapping() else {
                        unreachable!()
                    };
                    Arc::new(OrderedCompactionMapping::new(
                        remap.clone(),
                        t.sources().iter().map(|f| f.id as u32).collect(),
                        t.destinations().iter().map(|f| f.id as u32).collect(),
                    )) as Arc<dyn MappingReader>
                })
                .collect(),
            ledger,
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
            .translate(&[RowAddress::new_from_parts(0, 0)])
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
}
