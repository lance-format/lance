// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Compose mapping readers along fragment lineage to reach live fragments.

use crate::Dataset;
use lance_core::Result;
use lance_select::Remapper;
use lance_table::format::IndexMetadata;
use lance_table::system_index::frag_reuse::history::FragReuseHistory;
use lance_table::system_index::frag_reuse::ledger::FragReuseLedger;
use lance_table::system_index::frag_reuse::mapping::MappingSource;
use lance_table::system_index::frag_reuse::plan::{
    ReadContext, ReadPlan, RowDomain, SegmentSummary, TranslationPlan, TranslationSupport,
    segment_plans, translation_fingerprint,
};
use roaring::RoaringBitmap;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(test)]
tokio::task_local! {
    static LEGACY_READER_ONLY: ();
}

#[cfg(test)]
fn check_reader_path() -> Result<()> {
    if LEGACY_READER_ONLY.try_with(|_| ()).is_ok() {
        return Err(lance_core::Error::internal(
            "V1 operation entered the new FRI reader",
        ));
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
        if !super::segment::index_is_usable(index) {
            continue;
        }
        // System indices are table-level metadata; the user-index coverage
        // filter below does not apply to them. MemWAL in particular carries
        // `fragment_bitmap: None`, so passing it through the coverage grouping
        // would silently drop it and every `load_index_by_name` caller would
        // then see an initialized dataset as uninitialized. Pass every system
        // index through untouched, not just FRI by name. (`is_system_index`
        // matches a fixed name set, so a new system index type must be added to
        // that classifier to be recognized here.)
        if lance_table::system_index::is_system_index(index) {
            result[position] = Some(index.clone());
        } else {
            groups
                .entry(&index.name)
                .or_default()
                .push((position, index));
        }
    }
    let context = mapping.context();
    for members in groups.into_values() {
        let segments: Vec<&IndexMetadata> = members.iter().map(|(_, index)| *index).collect();
        let summaries = mapping.segment_summaries(dataset, &segments).await?;
        let plans = mapping.history.plan(&context, &summaries);
        let coverage = mapping.history.effective_coverage(&context, &summaries);
        for (((position, index), plan), coverage) in members.into_iter().zip(plans).zip(coverage) {
            // A segment that derives no coverage stays listed with empty
            // coverage: its fragments are scanned, and consumers that need the
            // index itself (full text search takes its tokenizer from it)
            // still find it. Only an excluded segment is left out.
            if plan == ReadPlan::Excluded {
                continue;
            }
            let mut index = index.clone();
            index.fragment_bitmap = Some(coverage);
            result[position] = Some(index);
        }
    }
    Ok(Arc::new(result.into_iter().flatten().collect()))
}

/// Whether this build can translate `index`'s stored addresses in batches:
/// `Some(true)` when its type has a batch remapper, `Some(false)` when it
/// does not, `None` when its details are missing or cannot be interpreted.
async fn segment_supports_batch_remapping(
    dataset: &Dataset,
    index: &IndexMetadata,
) -> Result<Option<bool>> {
    let Some(details) = index.index_details.as_ref() else {
        return Ok(None);
    };
    if super::segment::segment_has_vector_details(index) {
        return Ok(Some(
            super::frag_reuse_remapping::vector_supports_batch_remapping(dataset, index).await?,
        ));
    }
    Ok(super::frag_reuse_remapping::scalar_plugin_translates(
        details,
    ))
}

/// One segment's derived query-time inputs from the coverage backtrack.
#[deprecated(since = "15.0.0", note = "use FragReuseHistory::plan from lance-table")]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SegmentPlanParts {
    /// Live fragments this segment answers for in the current snapshot.
    pub coverage: RoaringBitmap,
    /// Fragments directly covered by other selected group members.
    pub excluded: RoaringBitmap,
    /// Positions (in ledger order) of the transitions on this segment's path.
    pub path: Vec<usize>,
}

/// A validated FRI graph whose mapping payloads are opened only when needed.
pub struct FragmentReuseIndex {
    history: Arc<FragReuseHistory>,
    entry_version: u64,
    live_fragments: Arc<RoaringBitmap>,
    mappings: Arc<dyn MappingSource>,
    /// Every address followed to a live fragment, for [`Self::remap_row_ids`].
    to_live: Remapper,
}

impl FragmentReuseIndex {
    /// Open the history for this dataset snapshot without reading mapping labels.
    pub async fn open(dataset: &Dataset, index: &IndexMetadata) -> Result<Arc<Self>> {
        #[cfg(test)]
        check_reader_path()?;
        let history = crate::index::frag_reuse::cached_frag_reuse_history(dataset, index).await?;
        let ledger = history.ledger();
        if ledger.has_unsupported_transitions() {
            log::debug!(
                "FRI {} has unsupported mappings; only paths reaching live fragments can provide query coverage",
                index.uuid
            );
        }
        let mappings =
            super::frag_reuse_remapping::frag_reuse_mapping_source(dataset, &history).await?;
        Ok(Arc::new(Self::new(
            history,
            index.dataset_version,
            dataset.fragment_bitmap.clone(),
            mappings,
        )))
    }

    fn new(
        history: Arc<FragReuseHistory>,
        entry_version: u64,
        live_fragments: Arc<RoaringBitmap>,
        mappings: Arc<dyn MappingSource>,
    ) -> Self {
        let to_live = TranslationPlan::following(&history, &live_fragments).bind(mappings.clone());
        Self {
            history,
            entry_version,
            live_fragments,
            mappings,
            to_live,
        }
    }

    pub(crate) fn history(&self) -> &FragReuseHistory {
        &self.history
    }

    /// The source the history's translation plans are bound to.
    pub(crate) fn mapping_source(&self) -> &Arc<dyn MappingSource> {
        &self.mappings
    }

    fn ledger(&self) -> &FragReuseLedger {
        self.history.ledger()
    }

    /// Whether the history carries transitions this build cannot interpret.
    pub(crate) fn has_unsupported_transitions(&self) -> bool {
        self.ledger().has_unsupported_transitions()
    }

    pub(crate) fn context(&self) -> ReadContext<'_> {
        ReadContext {
            entry_version: self.entry_version,
            live_fragments: &self.live_fragments,
        }
    }

    /// The summaries the history plans `segments` from. Whether a kind can
    /// translate is decided only for segments whose plan depends on it, since
    /// deciding can read the segment's files.
    pub(crate) async fn segment_summaries(
        &self,
        dataset: &Dataset,
        segments: &[&IndexMetadata],
    ) -> Result<Vec<SegmentSummary>> {
        let context = self.context();
        let mut summaries = Vec::with_capacity(segments.len());
        for segment in segments {
            let mut summary = SegmentSummary {
                uuid: segment.uuid,
                fragment_bitmap: segment.fragment_bitmap.clone(),
                dataset_version: segment.dataset_version,
                domain: if segment.results_are_row_addrs() {
                    RowDomain::Address
                } else {
                    RowDomain::StableRowId
                },
                // Never read unless the check below says the plan consults it.
                translation: TranslationSupport::Unsupported,
            };
            if self.history.needs_translation_support(&context, &summary) {
                summary.translation =
                    match segment_supports_batch_remapping(dataset, segment).await? {
                        Some(true) => TranslationSupport::Supported,
                        Some(false) => TranslationSupport::Unsupported,
                        None => TranslationSupport::Uninterpretable,
                    };
            }
            summaries.push(summary);
        }
        Ok(summaries)
    }

    /// Whether segment metadata could require translation: the ledger has
    /// unsupported transitions, or the segment's coverage is unknown, not all
    /// live, or touches the supported lineage.
    #[deprecated(
        since = "15.0.0",
        note = "use FragReuseHistory::plan from lance-table, which also applies the caught-up rule"
    )]
    pub fn may_need_translation(&self, provenance: Option<&RoaringBitmap>) -> bool {
        let ledger = self.ledger();
        if ledger.has_unsupported_transitions() {
            return true;
        }
        provenance.is_none_or(|bitmap| {
            !bitmap.is_subset(self.live_fragments.as_ref())
                || bitmap
                    .iter()
                    .any(|fragment| ledger.contains_fragment(fragment))
        })
    }

    /// Resolve each live fragment backwards to the nearest available index coverage.
    #[deprecated(
        since = "15.0.0",
        note = "use FragReuseHistory::effective_coverage from lance-table"
    )]
    pub fn segment_coverage(&self, provenance: &[RoaringBitmap]) -> Vec<RoaringBitmap> {
        segment_plans(self.ledger(), &self.live_fragments, provenance)
            .into_iter()
            .map(|plan| plan.coverage)
            .collect()
    }

    /// Resolve each segment's query coverage, sibling exclusions and the
    /// transitions on its path in one pass.
    #[deprecated(since = "15.0.0", note = "use FragReuseHistory::plan from lance-table")]
    #[allow(deprecated)]
    pub fn segment_plans(&self, provenance: &[RoaringBitmap]) -> Vec<SegmentPlanParts> {
        segment_plans(self.ledger(), &self.live_fragments, provenance)
            .into_iter()
            .map(|plan| SegmentPlanParts {
                coverage: plan.coverage,
                excluded: plan.excluded,
                path: plan.path,
            })
            .collect()
    }

    /// Fingerprint of everything that can change a segment's translated output.
    #[deprecated(
        since = "15.0.0",
        note = "use TranslationPlan::fingerprint from lance-table"
    )]
    pub fn translation_fingerprint(
        &self,
        coverage: &RoaringBitmap,
        excluded: &RoaringBitmap,
        path: &[usize],
    ) -> [u8; 32] {
        translation_fingerprint(self.ledger(), coverage, excluded, path)
    }

    /// Remap physical row IDs through supported lineage, stopping at live fragments.
    /// Missing or deleted paths produce `None`. Input order and duplicates are preserved.
    pub async fn remap_row_ids(self: &Arc<Self>, row_ids: &[u64]) -> Result<Vec<Option<u64>>> {
        let mut translated: Vec<Option<u64>> = row_ids.iter().copied().map(Some).collect();
        self.to_live.translate_in_place(&mut translated).await?;
        Ok(translated)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dataset::{InsertBuilder, WriteMode, WriteParams};
    use crate::index::{DatasetIndexExt, loading::DatasetIndexInternalExt};
    use crate::session::index_caches::IndexMetadataKey;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::types::Int32Type;
    use arrow_array::{RecordBatch, RecordBatchIterator, UInt32Array, cast::AsArray};
    use lance_core::Error;
    use lance_core::cache::LanceCache;
    use lance_core::utils::address::RowAddress;
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
            .with_row_address()
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
        prepare_partition_with_block_rows(dataset, source_ids, dest_base_id, 3).await
    }

    /// [`prepare_partition`] with the row map's block size chosen by the
    /// caller, for tests that need several blocks per cache chunk.
    pub async fn prepare_partition_with_block_rows(
        dataset: &Dataset,
        source_ids: &[u64],
        dest_base_id: u64,
        block_rows: u32,
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
        let mut writer =
            RowMapWriter::try_new_with_block_rows(writer, source_rows, 2, block_rows).unwrap();
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
            clustering_id: None,
            metadata: Default::default(),
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
        let indices = crate::index::loading::load_all_indices(&dataset)
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
    // chained end-to-end test in `crate::index::frag_reuse`). Cleanup is no
    // longer in this list either: tagged trim is implemented (see
    // `crate::dataset::index::frag_reuse::tests::tagged_trim`), and on this
    // fixture it is a retaining no-op. Shallow clone is no longer in this
    // list: it relocates the entry's row-map references, and deep clone is
    // not either: it copies the row maps and localizes their references (see
    // the clone tests in `crate::index::frag_reuse`).
    #[rstest::rstest]
    #[case::eager_compaction("eager")]
    #[case::statistics("statistics")]
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
        let indices = crate::index::loading::load_all_indices(&dataset)
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
            _ => unreachable!(),
        };
        assert!(matches!(error, Error::NotSupported { .. }), "{error}");
        assert!(
            error.to_string().to_lowercase().contains("upgrade"),
            "{error}"
        );
        assert_eq!(dataset.manifest.version, version);
        let indices = crate::index::loading::load_all_indices(&dataset)
            .await
            .unwrap();
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
        let indices = crate::index::loading::load_all_indices(&dataset)
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
        let indices = crate::index::loading::load_all_indices(&dataset)
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

    /// A segment built before every transition, whose kind translates.
    fn translating_segment(fragment_bitmap: RoaringBitmap) -> SegmentSummary {
        SegmentSummary {
            uuid: uuid::Uuid::new_v4(),
            fragment_bitmap: Some(fragment_bitmap),
            dataset_version: 0,
            domain: RowDomain::Address,
            translation: TranslationSupport::Supported,
        }
    }

    /// The coverage each segment of one index answers for, the segments
    /// built before every transition.
    fn translated_coverage(
        mapping: &FragmentReuseIndex,
        provenance: &[RoaringBitmap],
    ) -> Vec<RoaringBitmap> {
        let group: Vec<_> = provenance
            .iter()
            .cloned()
            .map(translating_segment)
            .collect();
        mapping
            .history()
            .effective_coverage(&mapping.context(), &group)
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
            translated_coverage(&mapping, std::slice::from_ref(&source_a)),
            vec![RoaringBitmap::new()]
        );
        assert_eq!(
            translated_coverage(&mapping, &[source_a.clone(), source_b.clone()]),
            vec![RoaringBitmap::from_iter([10, 11]); 2]
        );
        assert_eq!(
            translated_coverage(
                &mapping,
                &[source_a, source_b, RoaringBitmap::from_iter([10])]
            ),
            vec![
                RoaringBitmap::from_iter([11]),
                RoaringBitmap::from_iter([11]),
                RoaringBitmap::from_iter([10])
            ]
        );
    }

    /// The released reader methods keep answering as the plan does.
    #[tokio::test]
    #[allow(deprecated)]
    async fn deprecated_reader_methods_agree_with_the_plan() {
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;
        let mapping = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
        let provenance = [
            RoaringBitmap::from_iter([0]),
            RoaringBitmap::from_iter([1]),
            RoaringBitmap::from_iter([10]),
        ];

        let coverage = translated_coverage(&mapping, &provenance);
        assert_eq!(mapping.segment_coverage(&provenance), coverage);
        let plans = mapping.segment_plans(&provenance);
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.coverage.clone())
                .collect::<Vec<_>>(),
            coverage
        );
        assert_eq!(plans[2].excluded, RoaringBitmap::from_iter([0, 1]));
        assert_eq!(
            mapping.translation_fingerprint(&plans[0].coverage, &plans[0].excluded, &plans[0].path),
            mapping.translation_fingerprint(&plans[1].coverage, &plans[1].excluded, &plans[1].path),
        );
        assert!(mapping.may_need_translation(Some(&provenance[0])));
        assert!(!mapping.may_need_translation(Some(&RoaringBitmap::new())));
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
            translated_coverage(&reader, &[sources]),
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
                let indices = crate::index::loading::load_all_indices(&dataset)
                    .await
                    .unwrap();
                let fri = indices
                    .iter()
                    .find(|i| i.name == FRAG_REUSE_INDEX_NAME)
                    .unwrap();
                assert_eq!(fri.index_version, 0);
                assert_eq!(
                    dataset.manifest.reader_feature_flags
                        & lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX,
                    0
                );
                assert_eq!(
                    dataset.manifest.writer_feature_flags
                        & lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX,
                    0
                );
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
        let indices = crate::index::loading::load_all_indices(&dataset)
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
        assert!(Arc::ptr_eq(&first.history, &concurrent.history));
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
        assert!(Arc::ptr_eq(&first.history, &current.history));
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
        let mut indices = crate::index::loading::load_all_indices(&dataset)
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
        assert!(!Arc::ptr_eq(&first.history, &current.history));
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
        assert!(Arc::ptr_eq(&first.history, &second.history));
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
                    assert_eq!(
                        snapshot.manifest.reader_feature_flags
                            & lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX,
                        0
                    );
                    assert_eq!(
                        snapshot.manifest.writer_feature_flags
                            & lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX,
                        0
                    );
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
        assert!(mapping.ledger().has_unsupported_transitions());
        assert_eq!(
            mapping.history().plan(
                &mapping.context(),
                &[translating_segment(RoaringBitmap::from_iter([0, 1]))]
            ),
            vec![ReadPlan::Excluded]
        );
        assert!(mapping.ledger().transitions().is_empty());

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

    // A dropped unknown transition leaves no record of which fragments it
    // touched, so a segment whose bitmap is entirely live and absent from the
    // (now empty) retained graph would otherwise be granted identity and return
    // its pre-transition addresses. Deny identity to every segment while any
    // unsupported transition is present. Without that guard the
    // live-destination bitmap below is a subset of live fragments and disjoint
    // from the retained ledger, so it would be planned direct.
    #[tokio::test]
    async fn dropped_unknown_transition_denies_identity_to_live_segment() {
        let mut dataset = fixture().await;
        let (mut transition, destinations) = prepare(&dataset).await;
        transition.mapping = None;
        let mut raw = transition.encode_to_vec();
        raw.extend(field(17, b"future mapping"));
        // Destinations become the live fragments (ids 10, 11) after install.
        let live_destinations: RoaringBitmap = destinations.iter().map(|f| f.id as u32).collect();
        let fri = install(&mut dataset, field(2, &raw), destinations, false).await;
        let mapping = FragmentReuseIndex::open(&dataset, &fri).await.unwrap();
        assert!(mapping.ledger().has_unsupported_transitions());
        assert!(mapping.ledger().transitions().is_empty());
        // The bitmap IS a subset of live fragments and disjoint from the empty
        // retained ledger: identity would be granted but for the guard.
        assert!(live_destinations.is_subset(mapping.live_fragments.as_ref()));
        assert!(
            live_destinations
                .iter()
                .all(|f| !mapping.ledger().contains_fragment(f))
        );
        let segment = translating_segment(live_destinations);
        assert!(
            mapping
                .history()
                .needs_translation_support(&mapping.context(), &segment),
            "a dropped unknown transition must deny identity to every segment"
        );
        assert_eq!(
            mapping.history().plan(&mapping.context(), &[segment]),
            vec![ReadPlan::Excluded]
        );
    }

    // MemWAL and other system indices carry `fragment_bitmap: None` by design.
    // The coverage rewrite must pass every system index through untouched, not
    // just FRI by name: otherwise MemWAL falls into the coverage groups,
    // a segment without a bitmap is excluded, and every `load_index_by_name`
    // caller sees an initialized dataset as uninitialized. A non-system user
    // index with no coverage must still be dropped, proving the gate is
    // system-vs-user, not "None passes through".
    #[tokio::test]
    async fn tagged_history_keeps_mem_wal_system_index_visible() {
        use lance_table::system_index::mem_wal::{MEM_WAL_INDEX_NAME, new_mem_wal_index_meta};
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;

        let mem_wal = new_mem_wal_index_meta(dataset.manifest.version, Default::default()).unwrap();
        let mut orphan_user_index = mem_wal.clone();
        orphan_user_index.name = "user_index_without_coverage".into();
        orphan_user_index.fragment_bitmap = None;

        let indices = vec![fri.clone(), mem_wal.clone(), orphan_user_index.clone()];
        let listing = load_indices(&dataset, &fri, &indices).await.unwrap();

        assert!(
            listing.iter().any(|i| i.name == MEM_WAL_INDEX_NAME),
            "tagged FRI must not hide the MemWAL system index"
        );
        assert!(listing.iter().any(|i| i.name == FRAG_REUSE_INDEX_NAME));
        assert!(
            !listing
                .iter()
                .any(|i| i.name == "user_index_without_coverage"),
            "a non-system index with no coverage must still be dropped"
        );
    }

    // The coverage-rewritten listing is cached per snapshot identity, so a second
    // load_indices at the same snapshot returns the identical Arc instead of
    // re-running segment_coverage and the per-index bitmap rewrites.
    #[tokio::test]
    async fn derived_listing_is_cached_per_snapshot() {
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        install(&mut dataset, content, destinations, false).await;

        let first = dataset.load_indices().await.unwrap();
        let second = dataset.load_indices().await.unwrap();
        assert!(
            Arc::ptr_eq(&first, &second),
            "derived listing must be served from cache on the second call"
        );
    }

    // Bookkeeping lookups read the FRI entry off the raw manifest listing, so
    // they never drive the coverage rewrite. `frag_reuse_index_uuid` returns the
    // installed FRI uuid without opening the mapping.
    #[tokio::test]
    async fn frag_reuse_index_uuid_uses_bookkeeping_path() {
        let mut dataset = fixture().await;
        let (transition, destinations) = prepare(&dataset).await;
        let content = InlineContent {
            legacy_versions: vec![],
            transitions: vec![transition],
        }
        .encode_to_vec();
        let fri = install(&mut dataset, content, destinations, false).await;
        assert_eq!(dataset.frag_reuse_index_uuid().await, Some(fri.uuid));
    }

    // Boundary: a metadata-only bookkeeping lookup must not parse the mapping, so
    // it survives a corrupt ledger; but a caller that actually uses the mapping
    // (the query reader) still surfaces the corruption rather than swallowing it.
    #[tokio::test]
    async fn corrupt_ledger_spares_bookkeeping_but_not_the_reader() {
        let mut dataset = fixture().await;
        let (_transition, destinations) = prepare(&dataset).await;
        // Undecodable ledger bytes: the raw listing still names the FRI, but the
        // query reader cannot decode the transitions.
        let content = field(2, b"not a valid transition stream");
        let fri = install(&mut dataset, content, destinations, false).await;

        // Bookkeeping path (raw manifest listing) still resolves the uuid.
        assert_eq!(dataset.frag_reuse_index_uuid().await, Some(fri.uuid));

        // Query path decodes the ledger and must report the corruption.
        assert!(FragmentReuseIndex::open(&dataset, &fri).await.is_err());
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
        let history = crate::utils::test::frag_reuse_history_from_content(
            lance_table::system_index::frag_reuse::history::HistoryFormat::Tagged,
            &content,
        )
        .unwrap();
        let mapping = Arc::new(FragmentReuseIndex::new(
            Arc::new(history),
            0,
            Arc::new(RoaringBitmap::from_iter([3, 9])),
            // Ordered compactions only: no row map is read.
            Arc::new(
                lance_index::frag_reuse::mapping_source::RowMapMappingSource::new(
                    Arc::new(|base| {
                        Err(Error::internal(format!("unexpected row map base {base:?}")))
                    }),
                    lance_core::cache::WeakLanceCache::from(
                        &lance_core::cache::LanceCache::no_cache(),
                    ),
                ),
            ),
        ));
        let inputs = [0, 1, 2, 3, 9].map(|f| u64::from(RowAddress::new_from_parts(f, 1)));
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
        // An unknown transition anywhere denies every segment coverage.
        assert_eq!(
            translated_coverage(
                &mapping,
                &[RoaringBitmap::from_iter([0]), RoaringBitmap::from_iter([2])]
            ),
            if unknown_middle {
                vec![RoaringBitmap::new(); 2]
            } else {
                vec![RoaringBitmap::new(), RoaringBitmap::from_iter([3])]
            }
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
            clustering_id: None,
            metadata: Default::default(),
        };
        // Installed by writing the manifest directly: the commit chokepoint
        // (correctly) refuses a hand-built tagged entry, and no writer of this
        // version can produce an index_version-2 entry.
        let flag = lance_table::feature_flags::FLAG_FRAGMENT_REUSE_INDEX;
        let mut indices = dataset.load_indices().await.unwrap().as_ref().clone();
        indices.push(fri.clone());
        {
            let manifest = Arc::make_mut(&mut dataset.manifest);
            manifest.reader_feature_flags |= flag;
            manifest.writer_feature_flags |= flag;
        }
        persist_fixture(&mut dataset, indices).await;
        let version = dataset.manifest.version;
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
        // Shallow clone would have to relocate references inside content it
        // cannot interpret; the version gate rejects it before any decode.
        let error = dataset
            .shallow_clone("memory://fri-shallow", version, None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("index_version 2"), "{error}");
        // Deep clone would have to enumerate row maps referenced by content
        // it cannot interpret; the same version gate rejects it too.
        let error = dataset
            .deep_clone("memory://fri-deep", version, None)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().contains("index_version 2"), "{error}");
        let error = crate::dataset::index::frag_reuse::cleanup_frag_reuse_index(&mut dataset)
            .await
            .unwrap_err();
        assert!(matches!(error, Error::NotSupported { .. }));
        assert!(error.to_string().to_lowercase().contains("upgrade"));
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

#[cfg(test)]
#[path = "frag_reuse_reader_tests.rs"]
mod consumer_tests;
