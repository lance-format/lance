// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Integration-level compaction tests and their shared fixtures.

mod binary_copy;
mod blob;
mod commit;
mod concurrent_reindex;
mod execution;
mod frag_reuse;
mod frag_reuse_contract;
mod index_remap;
mod lineage;
mod overlays;
mod planning;
mod undeferrable_withdrawal;
mod vector;
use self::remapping::RemappedIndex;
use super::*;
use crate::dataset::UpdateBuilder;
use crate::dataset::WriteDestination;
use crate::dataset::index::DatasetIndexRemapperOptions;
use crate::dataset::index::frag_reuse::cleanup_frag_reuse_index;
use crate::dataset::optimize::remapping::{transpose_row_addrs, transpose_row_ids_from_digest};
use crate::dataset::rowids::read_spilled_row_ids;
use crate::dataset::rowids::{
    INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, SPILL_ROW_LINEAGE_CONFIG_KEY, get_row_id_index,
};
use crate::dataset::scanner::ColumnOrdering;
use crate::dataset::transaction::{Operation, RewriteGroup, Transaction};
use crate::dataset::{WriteMode, WriteParams};
use crate::index::CreateIndexBuilder;
use crate::index::DatasetIndexExt;
use crate::index::frag_reuse::{load_frag_reuse_index_details, open_frag_reuse_index};
use crate::index::loading::{DatasetIndexInternalExt, load_all_indices};
use crate::index::vector::{StageParams, VectorIndexParams};
use crate::session::caches::{ManifestKey, TransactionKey};
use crate::session::index_caches::RowIdIndexKey;
use crate::utils::test::{
    AmbiguousCommitHandler, AmbiguousFailure, DatagenExt, FragmentCount, FragmentRowCount,
};
use arrow::datatypes::{UInt8Type, UInt64Type};
use arrow_array::cast::AsArray;
use arrow_array::types::{Float32Type, Float64Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, LargeBinaryArray,
    LargeStringArray, PrimitiveArray, RecordBatch, RecordBatchIterator, StructArray,
};
use arrow_schema::{DataType, Field, Field as ArrowField, Schema, Schema as ArrowSchema};
use arrow_select::concat::concat_batches;
use async_trait::async_trait;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use lance_arrow::{BLOB_META_KEY, FixedSizeListArrayExt};
use lance_core::Error;
use lance_core::ROW_ID;
use lance_core::ROW_ID_FIELD_ID;
use lance_core::utils::address::RowAddress;
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_core::utils::tempfile::TempStrDir;
use lance_datagen::{ArrayGeneratorExt, BatchCount, Dimension, RowCount};
use lance_file::reader::FileReader as LFReader;
use lance_file::version::ConcreteFileVersion;
use lance_file::version::LanceFileVersion;
use lance_index::frag_reuse::CompactFragReuseIndexHandle;
use lance_index::frag_reuse::FRAG_REUSE_INDEX_NAME;
use lance_index::frag_reuse::FragReuseGroup;
use lance_index::is_system_index;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::{
    BuiltinIndexType, FullTextSearchQuery, InvertedIndexParams, ScalarIndexParams,
};
use lance_index::vector::bq::RQBuildParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::sq::builder::SQBuildParams;
use lance_index::{Index, IndexType};
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_linalg::distance::{DistanceType, MetricType};
use lance_table::feature_flags::{FLAG_FRAG_REUSE_WITH_STABLE_ROW_IDS, FLAG_SPILLED_ROW_LINEAGE};
use lance_table::format::Manifest;
use lance_table::format::RowIdMeta;
use lance_table::format::pb::transaction::Operation as PbOperation;
use lance_table::format::{Fragment, IndexMetadata, RowDatasetVersionSequence};
use lance_table::io::commit::{
    CommitError, CommitHandler, ConditionalPutCommitHandler, ManifestLocation,
    ManifestNamingScheme, ManifestWriter, RenameCommitHandler,
};
use lance_table::io::manifest::read_manifest_indexes;
use lance_table::rowids::RowIdSequence;
use lance_table::rowids::read_row_ids;
use lance_testing::datagen::{BatchGenerator, IncrementingInt32, RandomVector};
use roaring::{RoaringBitmap, RoaringTreemap};
use rstest::rstest;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashSet;
use std::collections::{BTreeMap, HashMap};
use std::io::Cursor;
use std::ops::Range;
use std::sync::Arc;
use uuid::Uuid;

fn sample_data() -> RecordBatch {
    let schema = Schema::new(vec![Field::new("a", DataType::Int64, false)]);

    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Int64Array::from_iter_values(0..10_000))],
    )
    .unwrap()
}

/// Build (or, with `replace`, rebuild) a scalar index named "scalar" on `col`.
async fn create_scalar_index(dataset: &mut Dataset, col: &str, replace: bool) {
    dataset
        .create_index(
            &[col],
            IndexType::Scalar,
            Some("scalar".into()),
            &ScalarIndexParams::default(),
            replace,
        )
        .await
        .unwrap();
}

#[derive(Debug, Default, Clone, PartialEq)]
struct MockIndexRemapperExpectation {
    expected: HashMap<u64, Option<u64>>,
    answer: Vec<RemappedIndex>,
}

#[derive(Debug, Default, Clone, PartialEq)]
struct MockIndexRemapper {
    expectations: Vec<MockIndexRemapperExpectation>,
}

impl MockIndexRemapper {
    fn stringify_map(map: &HashMap<u64, Option<u64>>) -> String {
        let mut sorted_keys = map.keys().collect::<Vec<_>>();
        sorted_keys.sort();
        let mut first_keys = sorted_keys
            .into_iter()
            .take(10)
            .map(|key| {
                format!(
                    "{}:{:?}",
                    RowAddress::from(*key),
                    map[key].map(RowAddress::from)
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        if map.len() > 10 {
            first_keys.push_str(", ...");
        }
        let mut result_str = format!("(len={})", map.len());
        result_str.push_str(&first_keys);
        result_str
    }

    fn in_any_order(expectations: &[Self]) -> Self {
        let expectations = expectations
            .iter()
            .flat_map(|item| item.expectations.clone())
            .collect::<Vec<_>>();
        Self { expectations }
    }
}

#[async_trait]
impl IndexRemapper for MockIndexRemapper {
    async fn remap_indices(
        &self,
        index_map: RowAddrRemap,
        _: &[u64],
    ) -> Result<Vec<RemappedIndex>> {
        for expectation in &self.expectations {
            let matches = match &index_map {
                RowAddrRemap::Direct(map) => map == &expectation.expected,
                RowAddrRemap::Compact(_) => {
                    let expected_frags: RoaringBitmap = expectation
                        .expected
                        .keys()
                        .map(|addr| (addr >> 32) as u32)
                        .collect();
                    index_map.affected_fragments() == expected_frags
                        && expectation
                            .expected
                            .iter()
                            .all(|(k, v)| index_map.get(*k) == Some(*v))
                }
            };
            if matches {
                return Ok(expectation.answer.clone());
            }
        }
        panic!(
            "Unexpected index map; expected one of:\n  {}",
            self.expectations
                .iter()
                .map(|expectation| Self::stringify_map(&expectation.expected))
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }
}

#[async_trait]
impl IndexRemapperOptions for MockIndexRemapper {
    async fn create_remapper(&self, _: &Dataset) -> Result<Option<Box<dyn IndexRemapper>>> {
        Ok(Some(Box::new(self.clone())))
    }
}

fn list_data_files(uri: &str) -> std::collections::BTreeSet<String> {
    std::fs::read_dir(std::path::Path::new(uri).join("data"))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default()
}

async fn execute_compaction_plan(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> Vec<RewriteResult> {
    let plan = DefaultCompactionPlanner::new(options.clone())
        .unwrap()
        .with_index_remapper(&DatasetIndexRemapperOptions::default())
        .plan(dataset)
        .await
        .unwrap();
    assert!(!plan.tasks.is_empty());
    let snapshot = dataset.clone();
    futures::stream::iter(plan.tasks)
        .map(|task| rewrite_files(Cow::Borrowed(&snapshot), task, options))
        .buffer_unordered(1)
        .try_collect()
        .await
        .unwrap()
}

fn row_addrs(frag_idx: u32, offsets: Range<u32>) -> Range<u64> {
    let start = RowAddress::new_from_parts(frag_idx, offsets.start);
    let end = RowAddress::new_from_parts(frag_idx, offsets.end);
    start.into()..end.into()
}

// The outer list has one item per new fragment
// The inner list has ranges of old row ids that map to the new fragment, in order
fn expect_remap(
    ranges: &[Vec<(Range<u64>, bool)>],
    starting_new_frag_idx: u32,
) -> MockIndexRemapper {
    let mut expected_remap: HashMap<u64, Option<u64>> = HashMap::default();
    expected_remap.reserve(ranges.iter().map(|r| r.len()).sum());
    for (new_frag_offset, new_frag_ranges) in ranges.iter().enumerate() {
        let new_frag_idx = starting_new_frag_idx + new_frag_offset as u32;
        let mut row_offset = 0;
        for (old_id_range, is_found) in new_frag_ranges.iter() {
            for old_id in old_id_range.clone() {
                if *is_found {
                    let new_id = RowAddress::new_from_parts(new_frag_idx, row_offset);
                    expected_remap.insert(old_id, Some(new_id.into()));
                    row_offset += 1;
                } else {
                    expected_remap.insert(old_id, None);
                }
            }
        }
    }
    MockIndexRemapper {
        expectations: vec![MockIndexRemapperExpectation {
            expected: expected_remap,
            answer: vec![],
        }],
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
struct IgnoreRemap {}

#[async_trait]
impl IndexRemapper for IgnoreRemap {
    async fn remap_indices(&self, _: RowAddrRemap, _: &[u64]) -> Result<Vec<RemappedIndex>> {
        Ok(Vec::new())
    }
}

#[async_trait]
impl IndexRemapperOptions for IgnoreRemap {
    async fn create_remapper(&self, _: &Dataset) -> Result<Option<Box<dyn IndexRemapper>>> {
        Ok(None)
    }
}

#[derive(Debug, Clone, Copy)]
enum IndexPresence {
    None,
    /// BTree: address-domain and its type supports an eager remap.
    RemapCapable,
    /// ZoneMap: address-domain, but its type always reports
    /// `RemapResult::Drop`, so it can never be an eager-remap consumer.
    AddressDomainOnly,
}

/// `cat` is a Utf8 column alongside the numeric `i` one so a test can build
/// a row-id-domain index (e.g. NGram) over it -- every remaining
/// row-id-domain scalar index type requires a text column, unlike BTree,
/// Bitmap, and LabelList, which moved to row-address domain and no longer
/// serve as that fixture.
async fn stable_row_id_dataset(uri: &str, num_fragments: u32) -> Dataset {
    lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .col(
            "cat",
            lance_datagen::array::cycle_utf8_literals(&["needle", "haystack"]),
        )
        .into_dataset_with_params(
            uri,
            FragmentCount::from(num_fragments),
            FragmentRowCount::from(10),
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 10,
                ..Default::default()
            }),
        )
        .await
        .unwrap()
}

fn written_addr(i: u32) -> u64 {
    u64::from(RowAddress::new_from_parts(i / 10, i % 10))
}

async fn rows_by_addr(dataset: &Dataset) -> HashMap<u64, (u64, i32)> {
    let batch = dataset
        .scan()
        .project(&["i"])
        .unwrap()
        .with_row_id()
        .with_row_address()
        .try_into_batch()
        .await
        .unwrap();
    let addrs = batch[lance_core::ROW_ADDR]
        .as_primitive::<UInt64Type>()
        .values();
    let row_ids = batch[ROW_ID].as_primitive::<UInt64Type>().values();
    let values = batch["i"].as_primitive::<Int32Type>().values();
    addrs
        .iter()
        .zip(row_ids.iter().zip(values.iter()))
        .map(|(addr, (row_id, value))| (*addr, (*row_id, *value)))
        .collect()
}

async fn assert_moves_keep_row_identity(before: &HashMap<u64, (u64, i32)>, dataset: &Dataset) {
    let after = rows_by_addr(dataset).await;
    assert_eq!(after.len(), before.len());
    let frag_reuse = dataset.frag_reuse_index().await.unwrap().unwrap();
    for (old_addr, row) in before {
        let new_addr = frag_reuse
            .remap_row_id(*old_addr)
            .unwrap_or_else(|| panic!("live row {row:?} at {old_addr} maps to a deletion"));
        assert_eq!(
            after.get(&new_addr),
            Some(row),
            "row moved from {old_addr} to {new_addr}"
        );
    }
}

fn flag_words(dataset: &Dataset, flag: u64) -> (bool, bool) {
    (
        dataset.manifest.reader_feature_flags & flag != 0,
        dataset.manifest.writer_feature_flags & flag != 0,
    )
}

#[derive(Clone, Copy, Debug)]
enum RecordingMismatch {
    SomeTasksRecorded,
    NoTaskRecorded,
    IncompatibleIndexCreated,
    /// The sticky flag outlives a tagged entry, for example through a
    /// restore to an untagged version and a migration to stable row ids.
    TaggedHistoryRequired,
}

#[derive(Clone, Copy, Debug)]
enum CatchUp {
    Remap,
    MergeSegments,
}

async fn catch_up_index(dataset: &mut Dataset, name: &str, catch_up: CatchUp) {
    match catch_up {
        CatchUp::Remap => {
            remapping::remap_column_index(dataset, &["i"], Some(name.into()))
                .await
                .unwrap();
        }
        CatchUp::MergeSegments => {
            let merged = dataset
                .merge_existing_index_segments(dataset.load_indices_by_name(name).await.unwrap())
                .await
                .unwrap();
            dataset
                .commit_existing_index_segments(name, "i", vec![merged])
                .await
                .unwrap();
        }
    }
}

async fn frag_reuse_watermarks(dataset: &Dataset) -> Vec<u64> {
    dataset
        .frag_reuse_index()
        .await
        .unwrap()
        .map(|index| {
            index
                .details
                .versions
                .iter()
                .map(|version| version.dataset_version)
                .collect()
        })
        .unwrap_or_default()
}

async fn stored_index(dataset: &Dataset, name: &str) -> IndexMetadata {
    read_manifest_indexes(
        &dataset.object_store,
        &dataset.manifest_location,
        &dataset.manifest,
    )
    .await
    .unwrap()
    .into_iter()
    .find(|index| index.name == name)
    .unwrap()
}

async fn rows_found_through_index(dataset: &Dataset) -> usize {
    dataset
        .count_rows(Some("i < 10 OR i = 70".to_owned()))
        .await
        .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum Race {
    Append,
    BuildIndex,
    /// From another session, so the rebase reads its rewrite from the transaction file.
    CompactElsewhere {
        excluded: &'static [u32],
    },
    /// Like `CompactElsewhere`, without `defer_index_remap`.
    CompactEagerlyElsewhere {
        excluded: &'static [u32],
    },
}

#[derive(Debug, Default)]
struct RaceBeforeRewrite {
    races: std::sync::Mutex<Vec<Race>>,
    handle: std::sync::Mutex<Option<Dataset>>,
    built: std::sync::Mutex<Option<(Dataset, IndexMetadata)>>,
}

impl RaceBeforeRewrite {
    fn arm(&self, handle: &Dataset, races: Vec<Race>) {
        *self.handle.lock().unwrap() = Some(handle.clone());
        *self.races.lock().unwrap() = races.into_iter().rev().collect();
    }
}

#[async_trait]
impl CommitHandler for RaceBeforeRewrite {
    async fn commit(
        &self,
        manifest: &mut Manifest,
        indices: Option<Vec<IndexMetadata>>,
        base_path: &object_store::path::Path,
        object_store: &ObjectStore,
        manifest_writer: ManifestWriter,
        naming_scheme: ManifestNamingScheme,
        transaction: Option<lance_table::format::Transaction>,
    ) -> std::result::Result<ManifestLocation, CommitError> {
        let is_rewrite = transaction
            .as_ref()
            .and_then(|transaction| transaction.as_pb().operation.as_ref())
            .is_some_and(|operation| matches!(operation, PbOperation::Rewrite(_)));
        let race = if is_rewrite {
            self.races.lock().unwrap().pop()
        } else {
            None
        };
        if let Some(race) = race {
            let mut handle = self.handle.lock().unwrap().clone().unwrap();
            handle.checkout_latest().await.unwrap();
            match race {
                Race::Append => {
                    let batch = arrow_array::record_batch!(("i", Int32, [9000, 9001])).unwrap();
                    handle
                        .append(
                            RecordBatchIterator::new([Ok(batch.clone())], batch.schema()),
                            None,
                        )
                        .await
                        .unwrap();
                }
                Race::BuildIndex => {
                    let index = handle
                        .create_index_builder(
                            &["i"],
                            IndexType::BTree,
                            &ScalarIndexParams::default(),
                        )
                        .name("second".into())
                        .execute_uncommitted()
                        .await
                        .unwrap();
                    *self.built.lock().unwrap() = Some((handle, index));
                }
                Race::CompactElsewhere { excluded } => {
                    let mut elsewhere = Dataset::open(handle.uri()).await.unwrap();
                    compact_files(
                        &mut elsewhere,
                        deferred_compaction_excluding(excluded),
                        None,
                    )
                    .await
                    .unwrap();
                }
                Race::CompactEagerlyElsewhere { excluded } => {
                    let mut elsewhere = Dataset::open(handle.uri()).await.unwrap();
                    compact_files(&mut elsewhere, eager_compaction_excluding(excluded), None)
                        .await
                        .unwrap();
                }
            }
        }
        ConditionalPutCommitHandler
            .commit(
                manifest,
                indices,
                base_path,
                object_store,
                manifest_writer,
                naming_scheme,
                transaction,
            )
            .await
    }
}

async fn racing_dataset(uri: &str, handler: Arc<RaceBeforeRewrite>) -> Dataset {
    let mut data_gen =
        BatchGenerator::new().col(Box::new(IncrementingInt32::new().named("i".to_owned())));
    let mut dataset = Dataset::write(
        data_gen.batch(500),
        uri,
        Some(WriteParams {
            max_rows_per_file: 100,
            commit_handler: Some(handler),
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    dataset.delete("i >= 10 AND i < 60").await.unwrap();
    create_scalar_index(&mut dataset, "i", false).await;
    dataset
}

#[derive(Clone, Copy, Debug)]
enum Boundary {
    AtWatermark,
    AfterWatermark,
}

#[derive(Clone, Copy, Debug)]
enum Publication {
    Direct,
    AfterRetries,
    OverAConcurrentCleanup,
}

fn stores_reuse_details_externally(entry: &IndexMetadata) -> bool {
    let proto = entry
        .index_details
        .as_ref()
        .unwrap()
        .to_msg::<lance_table::format::pb::FragmentReuseIndexDetails>()
        .unwrap();
    matches!(
        proto.content,
        Some(lance_table::format::pb::fragment_reuse_index_details::Content::External(_))
    )
}

/// The index directories holding a reuse details file.
fn reuse_details_dirs(uri: &str) -> HashSet<String> {
    std::fs::read_dir(format!("{uri}/_indices"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|dir| {
            dir.join(lance_index::frag_reuse::FRAG_REUSE_DETAILS_FILE_NAME)
                .exists()
        })
        .map(|dir| dir.file_name().unwrap().to_string_lossy().into_owned())
        .collect()
}

fn reuse_details_requests(stats: &lance_io::utils::tracking_store::IoStats, method: &str) -> usize {
    stats
        .requests
        .iter()
        .filter(|request| {
            request.method.starts_with(method)
                && request
                    .path
                    .as_ref()
                    .ends_with(lance_index::frag_reuse::FRAG_REUSE_DETAILS_FILE_NAME)
        })
        .count()
}

/// The deletions sit at non-zero offsets, so a mistranslated address loses rows.
async fn indexed_dataset(
    uri: &str,
    index: BuiltinIndexType,
    enable_stable_row_ids: bool,
) -> Dataset {
    let mut dataset = lance_datagen::gen_batch()
        .col("i", lance_datagen::array::step::<Int32Type>())
        .into_dataset_with_params(
            uri,
            FragmentCount::from(6),
            FragmentRowCount::from(10),
            Some(WriteParams {
                max_rows_per_file: 10,
                enable_stable_row_ids,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
    dataset
        .create_index_builder(
            &["i"],
            IndexType::Scalar,
            &ScalarIndexParams::for_builtin(index),
        )
        .name("i_idx".into())
        .await
        .unwrap();
    dataset.delete("i IN (3, 24, 45)").await.unwrap();
    dataset
}

fn deferred_compaction_excluding(excluded_fragment_ids: &[u32]) -> CompactionOptions {
    CompactionOptions {
        target_rows_per_fragment: 20,
        defer_index_remap: true,
        excluded_fragment_ids: excluded_fragment_ids.to_vec(),
        ..Default::default()
    }
}

fn fragments_except(dataset: &Dataset, kept: &[u32]) -> Vec<u32> {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .filter(|id| !kept.contains(id))
        .collect()
}

async fn plan_one_task(dataset: &Dataset, options: &CompactionOptions) -> RewriteResult {
    let plan = plan_compaction(dataset, options).await.unwrap();
    assert_eq!(plan.tasks().len(), 1);
    plan.compaction_tasks()
        .next()
        .unwrap()
        .execute(dataset)
        .await
        .unwrap()
}

// Keep racing tasks' source ranges disjoint even when the planner groups
// three small fragments.
async fn execute_first_fragment_pair(
    dataset: &Dataset,
    options: &CompactionOptions,
) -> RewriteResult {
    let plan = plan_compaction(dataset, options).await.unwrap();
    let mut task = plan.compaction_tasks().next().unwrap();
    assert!(task.task.fragments.len() >= 2);
    task.task.fragments.truncate(2);
    task.execute(dataset).await.unwrap()
}

async fn commit_one(
    dataset: &mut Dataset,
    result: RewriteResult,
    options: &CompactionOptions,
) -> Result<()> {
    commit_compaction(
        dataset,
        vec![result],
        Arc::new(DatasetIndexRemapperOptions::default()),
        options,
    )
    .await
    .map(|_| ())
}

async fn values_by_address(dataset: &Dataset) -> HashMap<u64, i32> {
    let batch = dataset
        .scan()
        .project(&["i"])
        .unwrap()
        .with_row_address()
        .try_into_batch()
        .await
        .unwrap();
    let addrs = batch[lance_core::ROW_ADDR]
        .as_primitive::<UInt64Type>()
        .values();
    let values = batch["i"].as_primitive::<Int32Type>().values();
    addrs.iter().copied().zip(values.iter().copied()).collect()
}

async fn values_matching(
    dataset: &Dataset,
    column: &str,
    filter: &str,
    use_scalar_index: bool,
) -> Vec<i32> {
    let mut scanner = dataset.scan();
    scanner.filter(filter).unwrap();
    scanner.project(&[column]).unwrap();
    scanner.use_scalar_index(use_scalar_index);
    let mut values = scanner.try_into_batch().await.unwrap()[column]
        .as_primitive::<Int32Type>()
        .values()
        .to_vec();
    values.sort_unstable();
    values
}

async fn assert_index_is_used(dataset: &Dataset, filter: &str, index_name: &str) {
    let mut scanner = dataset.scan();
    scanner.filter(filter).unwrap();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(
        plan.contains(&format!("@{index_name}(")),
        "{filter}: {plan}"
    );
}

async fn assert_index_answers_like_a_scan(dataset: &Dataset) {
    for filter in ["i < 15", "i >= 20 AND i < 35", "i > 40"] {
        assert_eq!(
            values_matching(dataset, "i", filter, true).await,
            values_matching(dataset, "i", filter, false).await,
            "{filter}"
        );
        assert_index_is_used(dataset, filter, "i_idx").await;
    }
}

async fn assert_mappings_intact(uri: &str, before: &HashMap<u64, i32>) {
    let dataset = Dataset::open(uri).await.unwrap();
    let frag_reuse = dataset.frag_reuse_index().await.unwrap();
    let after = values_by_address(&dataset).await;
    assert_eq!(after.len(), before.len());
    for (old_addr, value) in before {
        let new_addr = frag_reuse
            .as_ref()
            .map_or(Some(*old_addr), |index| index.remap_row_id(*old_addr))
            .unwrap_or_else(|| panic!("live row {value} at {old_addr} maps to a deletion"));
        assert_eq!(after.get(&new_addr), Some(value), "moved from {old_addr}");
    }
    assert_index_answers_like_a_scan(&dataset).await;
}

#[derive(Clone, Copy, Debug)]
enum ReuseHistoryChange {
    Created,
    Replaced,
    Removed,
}

fn eager_compaction_excluding(excluded_fragment_ids: &[u32]) -> CompactionOptions {
    CompactionOptions {
        defer_index_remap: false,
        ..deferred_compaction_excluding(excluded_fragment_ids)
    }
}

async fn commit_empty_frag_reuse_index(dataset: &mut Dataset) {
    let entry = crate::index::frag_reuse::build_frag_reuse_index_metadata(
        dataset,
        None,
        lance_table::system_index::frag_reuse::FragReuseIndexDetails { versions: vec![] },
        RoaringBitmap::new(),
    )
    .await
    .unwrap();
    let transaction = Transaction::new(
        dataset.manifest.version,
        Operation::CreateIndex {
            new_indices: vec![entry],
            removed_indices: vec![],
        },
        None,
    );
    dataset
        .apply_commit(transaction, &Default::default(), &Default::default())
        .await
        .unwrap();
}

/// Every row of `before` keeps its stable row id and value in `dataset`,
/// wherever it now lives.
async fn assert_rows_keep_their_ids(before: &HashMap<u64, (u64, i32)>, dataset: &Dataset) {
    let by_id = |rows: &HashMap<u64, (u64, i32)>| {
        rows.values()
            .map(|(row_id, value)| (*row_id, *value))
            .collect::<HashMap<_, _>>()
    };
    assert_eq!(by_id(&rows_by_addr(dataset).await), by_id(before));
}

#[derive(Clone, Copy, Debug)]
enum History {
    Absent,
    /// The entry a cleanup leaves once no index needs any version.
    Empty,
    Recorded,
}

fn newest_fragment(dataset: &Dataset) -> u32 {
    dataset
        .get_fragments()
        .iter()
        .map(|fragment| fragment.id() as u32)
        .max()
        .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum SecondRewrite {
    /// Copies the sources' data files as they are.
    BinaryCopy,
    /// A deletion in the sources makes binary copy fall back to a scan.
    Scan,
    /// Binary copy, with row ids spilled into a lineage file.
    SpilledBinaryCopy,
    /// A scan, with row ids spilled into the data file.
    SpilledScan,
}

/// The zone map serves every fragment, and queries go through it and answer
/// as a forced scan does, read from a cold handle.
async fn assert_index_serves_every_fragment(uri: &str) {
    let dataset = Dataset::open(uri).await.unwrap();
    let coverage = dataset
        .load_indices_by_name("i_idx")
        .await
        .unwrap()
        .iter()
        .filter_map(|index| index.fragment_bitmap.clone())
        .fold(RoaringBitmap::new(), |coverage, bitmap| coverage | bitmap);
    for fragment in dataset.get_fragments() {
        assert!(
            coverage.contains(fragment.id() as u32),
            "fragment {} is scanned, not served by the index: {coverage:?}",
            fragment.id()
        );
    }
    assert_index_answers_like_a_scan(&dataset).await;
}

#[derive(Clone, Copy, Debug)]
enum ZoneMapMaintenance {
    None,
    OptimizeIndices,
    MergeSegments,
    Cleanup,
    AppendAndOptimize,
}

/// The table's one recorded version holds the requested source fragments,
/// and every row of `before` keeps its stable row id.
async fn assert_recorded_fragments(
    before: &HashMap<u64, (u64, i32)>,
    dataset: &Dataset,
    expected: &[u64],
) {
    let frag_reuse = dataset.frag_reuse_index().await.unwrap().unwrap();
    assert_eq!(frag_reuse.details.versions.len(), 1);
    assert_eq!(frag_reuse.details.versions[0].old_frag_ids(), expected);
    assert_rows_keep_their_ids(before, dataset).await;
}

#[derive(Clone, Copy, Debug)]
enum AddressIndex {
    ZoneMap,
    BloomFilter,
}

async fn row_id_domain_indices(dataset: &Dataset) -> Result<Vec<String>> {
    Ok(load_all_indices(dataset)
        .await?
        .iter()
        .filter(|index| !is_system_index(index) && !index.results_are_row_addrs())
        .map(|index| index.name.clone())
        .collect())
}

/// Every row must be reachable through the named scalar index, whose row
/// ids no FRI may have rewritten.
async fn assert_id_idx_answers_every_row(dataset: &Dataset, index_name: &str) {
    if !dataset
        .load_indices_by_name(index_name)
        .await
        .unwrap()
        .is_empty()
    {
        for id in 0..12i32 {
            let mut scanner = dataset.scan();
            scanner.filter(&format!("id = {id}")).unwrap();
            scanner.project(&["id"]).unwrap();
            scanner.use_scalar_index(true);
            let found = scanner.try_into_batch().await.unwrap()["id"]
                .as_primitive::<Int32Type>()
                .values()
                .to_vec();
            assert_eq!(found, vec![id], "btree lost row {id}");
        }
    }
}

/// 12 rows over 3 fragments with a zone map on `value`, stable row ids on.
///
/// `cat` is a Utf8 column alongside the numeric `id`/`value` ones so a test
/// can build a row-id-domain index (e.g. NGram) over it -- every remaining
/// row-id-domain scalar index type requires a text column, unlike BTree and
/// Bitmap, which moved to row-address domain and no longer serve as that
/// fixture.
async fn zonemap_stable_row_id_dataset(uri: &str) -> Dataset {
    let batch = arrow_array::record_batch!(
        ("id", Int32, (0..12).collect::<Vec<_>>()),
        (
            "value",
            Int64,
            [
                Some(0),
                None,
                Some(20),
                Some(30),
                Some(40),
                None,
                Some(60),
                Some(70),
                Some(80),
                None,
                Some(100),
                Some(110)
            ]
        ),
        (
            "cat",
            Utf8,
            (0..12).map(|i| format!("cat-{i}")).collect::<Vec<_>>()
        )
    )
    .unwrap();
    let reader = RecordBatchIterator::new([Ok(batch.clone())], batch.schema());
    let mut dataset = Dataset::write(
        reader,
        uri,
        Some(WriteParams {
            max_rows_per_file: 4,
            max_rows_per_group: 4,
            enable_stable_row_ids: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
    assert!(dataset.manifest.uses_stable_row_ids());
    assert_eq!(dataset.get_fragments().len(), 3);

    dataset
        .create_index(
            &["value"],
            IndexType::ZoneMap,
            Some("value_idx".into()),
            &ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap),
            false,
        )
        .await
        .unwrap();
    assert_zonemap_answers_match_scan(&dataset).await;
    dataset
}

async fn zonemap_coverage(dataset: &Dataset) -> RoaringBitmap {
    dataset
        .load_indices_by_name("value_idx")
        .await
        .unwrap()
        .iter()
        .filter_map(|idx| idx.fragment_bitmap.clone())
        .fold(RoaringBitmap::new(), |mut acc, bm| {
            acc |= bm;
            acc
        })
}

/// The index path and a full scan must agree, for a value predicate, a range
/// predicate and the null bitmap.
async fn assert_zonemap_answers_match_scan(dataset: &Dataset) {
    async fn scan_ids(dataset: &Dataset, filter: &str, use_scalar_index: bool) -> Vec<i32> {
        let mut scanner = dataset.scan();
        scanner.filter(filter).unwrap();
        scanner.project(&["id"]).unwrap();
        scanner.use_scalar_index(use_scalar_index);
        scanner.try_into_batch().await.unwrap()["id"]
            .as_primitive::<Int32Type>()
            .values()
            .to_vec()
    }

    for (filter, expected) in [
        ("value IS NULL", vec![1, 5, 9]),
        ("value = 20", vec![2]),
        ("value > 90", vec![10, 11]),
    ] {
        assert_eq!(scan_ids(dataset, filter, false).await, expected, "{filter}");
        assert_eq!(scan_ids(dataset, filter, true).await, expected, "{filter}");
    }
}

async fn assert_zonemap_is_used(dataset: &Dataset) {
    let mut scanner = dataset.scan();
    scanner.filter("value IS NULL").unwrap();
    let plan = scanner.explain_plan(false).await.unwrap();
    assert!(
        plan.contains("ScalarIndexQuery: query=[value IS NULL]@value_idx(ZoneMap)"),
        "Expected ZoneMap index query in plan: {plan}"
    );
}

impl AddressIndex {
    fn column(self) -> &'static str {
        match self {
            Self::ZoneMap => "value",
            Self::BloomFilter => "id",
        }
    }

    async fn create(self, dataset: &mut Dataset, replace: bool) {
        let (index_type, builtin) = match self {
            Self::ZoneMap => (IndexType::ZoneMap, BuiltinIndexType::ZoneMap),
            Self::BloomFilter => (IndexType::BloomFilter, BuiltinIndexType::BloomFilter),
        };
        dataset
            .create_index(
                &[self.column()],
                index_type,
                Some("addr_idx".into()),
                &ScalarIndexParams::for_builtin(builtin),
                replace,
            )
            .await
            .unwrap();
    }

    fn cases(self, is_appended: bool) -> Vec<(&'static str, Vec<i32>)> {
        match (self, is_appended) {
            (Self::ZoneMap, false) => vec![
                ("value IS NULL", vec![5, 9]),
                ("value = 20", vec![2]),
                ("value > 90", vec![10, 11]),
            ],
            (Self::ZoneMap, true) => vec![
                ("value IS NULL", vec![5, 9, 12, 14]),
                ("value = 20", vec![2]),
                ("value > 90", vec![10, 11, 13, 15]),
            ],
            (Self::BloomFilter, false) => vec![
                ("id = 1", vec![]),
                ("id = 2", vec![2]),
                ("id = 10", vec![10]),
            ],
            (Self::BloomFilter, true) => vec![
                ("id = 2", vec![2]),
                ("id = 10", vec![10]),
                ("id = 13", vec![13]),
            ],
        }
    }
}

async fn assert_index_answers(dataset: &Dataset, cases: &[(&str, Vec<i32>)]) {
    for (filter, expected) in cases {
        for use_scalar_index in [false, true] {
            assert_eq!(
                &values_matching(dataset, "id", filter, use_scalar_index).await,
                expected,
                "{filter}"
            );
        }
        assert_index_is_used(dataset, filter, "addr_idx").await;
    }
}

#[derive(Clone, Copy, Debug)]
enum RaceOutcome {
    Committed,
    Rejected,
    ConflictThenRejected,
}

/// How the second commit's rebase sees the first commit's transaction. An
/// uncached manifest is decoded again and its inline transaction replaces the
/// cached one; a decoded rewrite lacks the fragment reuse index.
#[derive(Clone, Copy, Debug, PartialEq)]
enum FirstCommitSeenAs {
    /// The shared session caches the manifest under the key the rebase lists.
    InMemory,
    /// The shared session caches the manifest only under the e_tag-less key
    /// a `RenameCommitHandler` commit leaves.
    InlineTransaction,
    SeparateSession,
}

async fn uncached_listed_manifest(rebasing: &Dataset, version: u64) -> ManifestLocation {
    let listed = rebasing
        .commit_handler
        .list_manifest_locations_since(
            &rebasing.base,
            rebasing.object_store.as_ref(),
            rebasing.manifest.version,
        )
        .try_filter(|location| std::future::ready(location.version == version))
        .try_next()
        .await
        .unwrap()
        .unwrap();
    let key = ManifestKey {
        version,
        e_tag: listed.e_tag.as_deref(),
    };
    assert!(key.e_tag.is_some());
    assert!(rebasing.metadata_cache.get_with_key(&key).await.is_none());
    listed
}

struct IndependentRemapperOptions;

#[async_trait]
impl IndexRemapperOptions for IndependentRemapperOptions {
    async fn create_remapper(&self, dataset: &Dataset) -> Result<Option<Box<dyn IndexRemapper>>> {
        Ok(Some(Box::new(IndependentRemapper {
            inner: DatasetIndexRemapperOptions::default()
                .create_remapper(dataset)
                .await?
                .unwrap(),
        })))
    }
}

struct IndependentRemapper {
    inner: Box<dyn IndexRemapper>,
}

#[async_trait]
impl IndexRemapper for IndependentRemapper {
    async fn remap_indices(
        &self,
        mapping: RowAddrRemap,
        affected: &[u64],
    ) -> Result<Vec<RemappedIndex>> {
        self.inner.remap_indices(mapping, affected).await
    }
}

async fn assert_vector_index_coverage(dataset: &Dataset) {
    let expected_batch = dataset
        .scan()
        .filter("vec IS NOT NULL")
        .unwrap()
        .project::<String>(&[])
        .unwrap()
        .with_row_id()
        .try_into_batch()
        .await
        .unwrap();
    let expected = expected_batch[ROW_ID]
        .as_primitive::<arrow_array::types::UInt64Type>()
        .values()
        .iter()
        .copied()
        .collect::<roaring::RoaringTreemap>();
    let mut stored = roaring::RoaringTreemap::new();
    let indices = dataset.load_indices_by_name("vec_idx").await.unwrap();
    for metadata in indices {
        let index = dataset
            .open_vector_index("vec", &metadata.uuid, &NoOpMetricsCollector)
            .await
            .unwrap();
        for partition in 0..index.ivf_model().num_partitions() {
            let mut stream = index
                .partition_reader(partition, false, &NoOpMetricsCollector)
                .await
                .unwrap();
            while let Some(batch) = stream.try_next().await.unwrap() {
                for address in batch[ROW_ID]
                    .as_primitive::<arrow_array::types::UInt64Type>()
                    .values()
                {
                    assert!(
                        stored.insert(*address),
                        "duplicate stored address {address}"
                    );
                }
            }
        }
    }
    assert_eq!(stored, expected);
}

/// Build an `id` + `vec` dataset, create the given IVF vector index,
/// optionally delete rows, then run deferred compaction (which materializes
/// the deletions into the fragment-reuse index) and assert that KNN over
/// surviving vectors during the FRI window (a) never returns a deleted row
/// and (b) stays consistent with the pre-compaction answer.
///
/// The deletion path is the interesting one: materialized deletions drop
/// rows from the quantization storage at load time, which shifts storage
/// positions. Flat storage (FLAT/PQ/SQ/RQ) is scanned linearly so this is
/// fine, but the HNSW graph addresses storage positionally and is not
/// frag-reuse aware, so a desync would surface here as recall collapse or a
/// resurrected/again-deleted row.
/// Top-k `id`s for a KNN query against the `vec` column.
async fn vector_knn_ids(dataset: &Dataset, query: &[f32], k: usize) -> Vec<i32> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type};
    let qa = PrimitiveArray::<Float32Type>::from_iter_values(query.iter().copied());
    let mut scanner = dataset.scan();
    scanner.nearest("vec", &qa, k).unwrap();
    scanner.nprobes(64);
    scanner.project(&["id"]).unwrap();
    let batches = scanner
        .try_into_stream()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await
        .unwrap();
    let mut ids = Vec::new();
    for b in &batches {
        ids.extend(b["id"].as_primitive::<Int32Type>().values().iter().copied());
    }
    ids
}

async fn check_vector_defer_compaction(
    params: VectorIndexParams,
    delete_predicate: Option<&str>,
    k: usize,
    min_overlap: usize,
) {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type};
    use lance_datagen::Dimension;

    const DIM: u32 = 32;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(1000))
        .await
        .unwrap();

    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".into()),
            &params,
            false,
        )
        .await
        .unwrap();
    let original_uuid = dataset
        .load_index_by_name("vec_idx")
        .await
        .unwrap()
        .unwrap()
        .uuid;

    if let Some(pred) = delete_predicate {
        dataset.delete(pred).await.unwrap();
    }

    // Collect surviving (id, vec) pairs and the set of surviving ids.
    let mut survivors: Vec<(i32, Vec<f32>)> = Vec::new();
    {
        let mut scanner = dataset.scan();
        scanner.project(&["id", "vec"]).unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        for batch in &batches {
            let ids = batch["id"].as_primitive::<Int32Type>();
            let vecs = batch["vec"].as_fixed_size_list();
            for i in 0..batch.num_rows() {
                let v = vecs.value(i);
                let v = v.as_primitive::<Float32Type>().values().to_vec();
                survivors.push((ids.value(i), v));
            }
        }
    }
    assert!(!survivors.is_empty());
    let surviving_ids: std::collections::HashSet<i32> =
        survivors.iter().map(|(id, _)| *id).collect();

    // Sample queries from survivors and capture the pre-compaction answer.
    let step = (survivors.len() / 16).max(1);
    let queries: Vec<(i32, Vec<f32>)> = survivors.iter().step_by(step).cloned().collect();
    let mut baseline: Vec<Vec<i32>> = Vec::new();
    for (_, q) in &queries {
        baseline.push(vector_knn_ids(&dataset, q, k).await);
    }

    // Deferred compaction materializes the deletions into the frag-reuse index.
    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 2_000,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(metrics.fragments_removed > 0);
    assert!(
        dataset
            .load_indices()
            .await
            .unwrap()
            .iter()
            .any(|idx| idx.name == FRAG_REUSE_INDEX_NAME),
        "deferred compaction must record a frag-reuse index"
    );
    assert_eq!(
        dataset
            .load_index_by_name("vec_idx")
            .await
            .unwrap()
            .unwrap()
            .uuid,
        original_uuid,
        "index must not be physically remapped yet (FRI window)"
    );

    // During the FRI window: no deleted rows, and stable vs the baseline.
    for (i, (_, q)) in queries.iter().enumerate() {
        let after = vector_knn_ids(&dataset, q, k).await;
        for id in &after {
            assert!(
                surviving_ids.contains(id),
                "KNN returned id {id} that is not a surviving row (query #{i})"
            );
        }
        let overlap = after.iter().filter(|id| baseline[i].contains(id)).count();
        assert!(
            overlap >= min_overlap,
            "KNN top-{k} diverged after deferred compaction: overlap {overlap} < {min_overlap} (query #{i})"
        );
    }
}

fn small_ivf() -> lance_index::vector::ivf::IvfBuildParams {
    lance_index::vector::ivf::IvfBuildParams {
        max_iters: 2,
        num_partitions: Some(2),
        sample_rate: 2,
        ..Default::default()
    }
}

// NOTE: IVF_HNSW_* under materialized deletions is a known gap (lance#3993,
// HNSW auto-remap not implemented) — the HNSW graph isn't realigned after the
// frag-reuse drop. Deferred remap is gated off for HNSW tables, so there is
// no lance-level reproducer here; the gate is tested in the data plane.
// Merge-only HNSW is covered (see the *_remap_and_trim tests).

/// Merge-only deferred compaction, then a PHYSICAL remap + FRI trim. Asserts
/// the index is rebuilt with its details kept, the fragment-reuse index
/// trims to zero versions, and KNN stays consistent with the
/// pre-compaction answer through both the FRI window and the physical
/// remap. (HNSW rebuilds its graph on physical remap, so the overlap is
/// recall-tolerant.)
async fn check_vector_remap_and_trim(
    params: VectorIndexParams,
    k: usize,
    window_overlap: usize,
    post_remap_overlap: Option<usize>,
) {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float32Type, Int32Type};
    use lance_datagen::Dimension;

    const DIM: u32 = 8;
    let mut dataset = lance_datagen::gen_batch()
        .col("id", lance_datagen::array::step::<Int32Type>())
        .col(
            "vec",
            lance_datagen::array::rand_vec::<Float32Type>(Dimension::from(DIM)),
        )
        .into_ram_dataset(FragmentCount::from(6), FragmentRowCount::from(64))
        .await
        .unwrap();
    dataset
        .create_index(
            &["vec"],
            IndexType::Vector,
            Some("vec_idx".into()),
            &params,
            false,
        )
        .await
        .unwrap();
    let original = dataset
        .load_index_by_name("vec_idx")
        .await
        .unwrap()
        .unwrap();
    let original_uuid = original.uuid;

    // Sample queries from stored vectors + capture the pre-compaction answer.
    let mut rows: Vec<Vec<f32>> = Vec::new();
    {
        let mut scanner = dataset.scan();
        scanner.project(&["vec"]).unwrap();
        let batches = scanner
            .try_into_stream()
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        for batch in &batches {
            let vecs = batch["vec"].as_fixed_size_list();
            for i in 0..batch.num_rows() {
                let v = vecs.value(i);
                rows.push(v.as_primitive::<Float32Type>().values().to_vec());
            }
        }
    }
    let step = (rows.len() / 4).max(1);
    let queries: Vec<Vec<f32>> = rows.iter().step_by(step).cloned().collect();
    let mut baseline: Vec<Vec<i32>> = Vec::new();
    for q in &queries {
        baseline.push(vector_knn_ids(&dataset, q, k).await);
    }

    // Merge-only deferred compaction.
    let metrics = compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 128,
            defer_index_remap: true,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();
    assert!(metrics.fragments_removed > 0);
    assert_eq!(
        dataset
            .load_index_by_name("vec_idx")
            .await
            .unwrap()
            .unwrap()
            .uuid,
        original_uuid,
        "index must not be physically remapped yet (FRI window)"
    );
    for (i, q) in queries.iter().enumerate() {
        let window = vector_knn_ids(&dataset, q, k).await;
        let overlap = window.iter().filter(|id| baseline[i].contains(id)).count();
        assert!(
            overlap >= window_overlap,
            "FRI-window KNN diverged: overlap {overlap} < {window_overlap} (query #{i})"
        );
    }

    // Physical remap + trim the fragment-reuse index.
    remapping::remap_column_index(&mut dataset, &["vec"], Some("vec_idx".into()))
        .await
        .unwrap();
    cleanup_frag_reuse_index(&mut dataset).await.unwrap();

    let remapped = dataset
        .load_index_by_name("vec_idx")
        .await
        .unwrap()
        .unwrap();
    assert_ne!(
        remapped.uuid, original_uuid,
        "index should have been physically remapped"
    );
    assert_eq!(
        remapped.index_details, original.index_details,
        "a remap keeps the index's details"
    );
    if let Some(meta) = dataset
        .load_index_by_name(FRAG_REUSE_INDEX_NAME)
        .await
        .unwrap()
    {
        let versions = load_frag_reuse_index_details(&dataset, &meta)
            .await
            .unwrap()
            .versions
            .len();
        assert_eq!(versions, 0, "frag-reuse index must trim to zero versions");
    }

    for (i, q) in queries.iter().enumerate() {
        let after = vector_knn_ids(&dataset, q, k).await;
        // No stale/desynced addresses (a bad address fails the take above).
        assert!(
            !after.is_empty(),
            "post-remap KNN returned no rows (query #{i})"
        );
        // Physical remap rebuilds the HNSW graph, so recall is only compared
        // for the exact (non-HNSW) types.
        if let Some(min_overlap) = post_remap_overlap {
            let overlap = after.iter().filter(|id| baseline[i].contains(id)).count();
            assert!(
                overlap >= min_overlap,
                "post-remap KNN diverged: overlap {overlap} < {min_overlap} (query #{i})"
            );
        }
    }
}

/// The ids of `query`'s `k` nearest neighbours, through every partition of
/// the index or, without it, by exact search.
async fn exact_knn_ids(dataset: &Dataset, query: &[f32], k: usize, use_index: bool) -> Vec<i32> {
    let query = PrimitiveArray::<Float32Type>::from_iter_values(query.iter().copied());
    let mut scanner = dataset.scan();
    scanner
        .nearest("vec", &query, k)
        .unwrap()
        .minimum_nprobes(2)
        .use_index(use_index);
    scanner.project(&["id"]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    let mut ids = batch["id"].as_primitive::<Int32Type>().values().to_vec();
    ids.sort_unstable();
    ids
}

// Scalar index correctness across deferred compaction WITH materialized
// deletions. The existing test_read_*_index_with_defer_index_remap tests are
// merge-only and project no columns (count-only), so they never take and
// never exercise the deletion drop path. These add an `id` column, delete a
// prefix, defer-compact, then run the indexed query *projecting id* (a take)
// and assert no deleted row is returned. Bitmap/BTree have no positional
// internal structure so the drop path is exact; the Inverted (FTS) index
// does (see its test below), and currently desyncs under deletions.

// NOTE: Inverted/FTS under materialized deletions is broken (BM25 scores
// via positional num_tokens[doc_id]; the frag-reuse drop shifts doc_id
// positions -> out-of-bounds). It is gated off defer in the data plane
// until fixed, so there is no lance-level reproducer here. Merge-only FTS
// is covered by test_read_inverted_index_with_defer_index_remap.

/// Writes `sample_data` as 10 fragments of 100 rows each, deletes half of
/// fragment 0's rows, and puts a one-cell data overlay on it, so every
/// `max_source_*` planning budget is exercised against a dataset that
/// carries deleted rows and overlay files.
async fn dataset_with_ten_small_fragments(test_uri: &str) -> Dataset {
    let data = sample_data();
    let schema = data.schema();
    let write_params = WriteParams {
        max_rows_per_file: 100,
        ..Default::default()
    };
    Dataset::write(
        RecordBatchIterator::new(vec![Ok(data.slice(0, 100))], schema.clone()),
        test_uri,
        Some(write_params.clone()),
    )
    .await
    .unwrap();
    for i in 1..10 {
        let mut append_params = write_params.clone();
        append_params.mode = WriteMode::Append;
        Dataset::write(
            RecordBatchIterator::new(vec![Ok(data.slice(i * 100, 100))], schema.clone()),
            test_uri,
            Some(append_params),
        )
        .await
        .unwrap();
    }
    let mut dataset = Dataset::open(test_uri).await.unwrap();
    assert_eq!(dataset.get_fragments().len(), 10);
    // Fragment 0 keeps 100 physical rows but only 50 live rows.
    dataset.delete("a < 50").await.unwrap();
    // The overlay shadows one surviving base value (offset 60, kept clear
    // of the delete predicate) without adding any rows.
    let dataset = commit_overlay(
        dataset,
        0,
        &[0],
        OverlayCoverage::dense(bitmap([60])),
        vec![Arc::new(Int64Array::from(vec![60_000_i64]))],
    )
    .await;
    assert_eq!(
        dataset.get_fragment(0).unwrap().metadata().overlays.len(),
        1
    );
    dataset
}

fn count_all_files_in(dir: &std::path::Path) -> std::io::Result<usize> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut count = 0;
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            count += count_all_files_in(&path)?;
        } else if path.is_file() {
            // Ignore macOS system files if any
            if path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|file_name| !file_name.starts_with('.'))
            {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn count_data_files_in(base_dir: &str) -> usize {
    let data_dir = std::path::Path::new(base_dir).join("data");
    count_all_files_in(&data_dir).unwrap_or(0)
}

async fn read_blob_bytes_by_index(
    dataset: &Arc<Dataset>,
    column: &str,
) -> Vec<(i32, Option<Vec<u8>>)> {
    let mut scanner = dataset.scan();
    scanner.with_row_id();
    let batch = scanner
        .project(&["id", column])
        .unwrap()
        .try_into_batch()
        .await
        .unwrap();
    let ids = batch
        .column_by_name("id")
        .unwrap()
        .as_primitive::<Int32Type>();
    let row_ids = batch
        .column_by_name(ROW_ID)
        .unwrap()
        .as_primitive::<UInt64Type>();

    let mut result = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let row_id = row_ids.value(i);
        let id = ids.value(i);
        let blobs = dataset.take_blobs(&[row_id], column).await.unwrap();
        match blobs.into_iter().next().flatten() {
            Some(blob) => {
                let data = blob.read().await.unwrap();
                result.push((id, Some(data.to_vec())));
            }
            None => result.push((id, None)),
        }
    }
    result
}

fn mixed_blob_values() -> Vec<(i32, Option<Vec<u8>>)> {
    vec![
        (0, Some(vec![b'0'; 80])),
        (1, None),
        (2, Some(Vec::new())),
        (3, Some(vec![b'3'; 80])),
        (4, Some(vec![b'4'; 80])),
        (5, Some(vec![b'5'; 80])),
    ]
}

async fn assert_compaction_preserves_blob_values(
    mut dataset: Dataset,
    expected: &[(i32, Option<Vec<u8>>)],
) {
    assert_eq!(dataset.get_fragments().len(), 3);

    let mut before = read_blob_bytes_by_index(&Arc::new(dataset.clone()), "blob").await;
    before.sort_by_key(|(id, _)| *id);
    assert_eq!(before, expected);

    compact_files(
        &mut dataset,
        CompactionOptions {
            target_rows_per_fragment: 1024 * 1024,
            ..Default::default()
        },
        None,
    )
    .await
    .unwrap();

    assert_eq!(dataset.get_fragments().len(), 1);

    let mut after = read_blob_bytes_by_index(&Arc::new(dataset), "blob").await;
    after.sort_by_key(|(id, _)| *id);
    assert_eq!(after, expected);
}

// ---- `max_overlays_per_fragment` compaction trigger ----
//
// Tests for the trigger that fully compacts a fragment carrying too many data
// overlay files into a fresh fragment with the overlays (and deletions)
// materialized into the base data.
use arrow_array::record_batch;
use lance_file::writer::FileWriterOptions;
use lance_io::utils::CachedFileSize;
use lance_table::format::DataFile;
use lance_table::format::overlay::{DataOverlayFile, OverlayCoverage};

use crate::dataset::DATA_DIR;
use crate::dataset::transaction::DataOverlayGroup;

/// Two-fragment Int32 dataset: `id` (field 0) = 0..12 and `val` (field 1) =
/// id * 10, six rows per fragment (fragments 0 and 1).
async fn create_base_dataset(uri: &str) -> Dataset {
    let batch = record_batch!(
        ("id", Int32, (0..12).collect::<Vec<_>>()),
        ("val", Int32, (0..12).map(|v| v * 10).collect::<Vec<_>>())
    )
    .unwrap();
    let schema = batch.schema();
    let write_params = WriteParams {
        max_rows_per_file: 6,
        max_rows_per_group: 6,
        data_storage_version: Some(LanceFileVersion::Stable),
        ..Default::default()
    };
    let reader = RecordBatchIterator::new(vec![Ok(batch)], schema.clone());
    Dataset::write(reader, uri, Some(write_params))
        .await
        .unwrap()
}

fn i32_array(values: impl IntoIterator<Item = Option<i32>>) -> ArrayRef {
    Arc::new(Int32Array::from_iter(values))
}

fn bitmap(offsets: impl IntoIterator<Item = u32>) -> RoaringBitmap {
    RoaringBitmap::from_iter(offsets)
}

/// Write a dense overlay covering `fields` of `fragment_id` with `columns`
/// as the per-field value columns, then commit it as a `DataOverlay`.
async fn commit_overlay(
    dataset: Dataset,
    fragment_id: u64,
    fields: &[i32],
    coverage: OverlayCoverage,
    columns: Vec<ArrayRef>,
) -> Dataset {
    let read_version = dataset.version().version;
    let overlay_schema = dataset.schema().project_by_ids(fields, true);
    let filename = format!("{}.lance", Uuid::new_v4());
    let path = dataset.base.clone().join(DATA_DIR).join(filename.as_str());
    let obj_writer = dataset.object_store.create(&path).await.unwrap();
    let file_version = LanceFileVersion::Stable.resolve();
    let mut writer = lance_file::versions::create_writer(
        file_version,
        obj_writer,
        overlay_schema,
        FileWriterOptions::default(),
    )
    .unwrap();
    for (column_index, array) in columns.into_iter().enumerate() {
        writer.write_column(column_index, array).await.unwrap();
    }
    let summary = writer.finish().await.unwrap();

    let mut data_file = DataFile::new_unstarted(filename, file_version);
    data_file.fields = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(f, _)| *f as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.column_indices = writer
        .field_id_to_column_indices()
        .iter()
        .map(|(_, c)| *c as i32)
        .collect::<Vec<_>>()
        .into();
    data_file.file_size_bytes = CachedFileSize::new(summary.size_bytes);

    Dataset::commit(
        WriteDestination::Dataset(Arc::new(dataset)),
        Operation::DataOverlay {
            groups: vec![DataOverlayGroup {
                fragment_id,
                overlays: vec![DataOverlayFile {
                    data_file,
                    coverage,
                    committed_version: 0,
                }],
            }],
        },
        Some(read_version),
        None,
        None,
        Arc::new(Default::default()),
        false,
    )
    .await
    .unwrap()
}

/// Commit `n` distinct single-cell overlays to fragment 0 (offset `i`, val
/// column set to `1000 + i`), so the fragment ends up with `n` overlays. The
/// `1000 +` offset keeps overlaid values clear of the base `id * 10` values.
async fn commit_n_overlays(mut dataset: Dataset, n: u32) -> Dataset {
    for i in 0..n {
        dataset = commit_overlay(
            dataset,
            0,
            &[1],
            OverlayCoverage::dense(bitmap([i])),
            vec![i32_array([Some(1000 + i as i32)])],
        )
        .await;
    }
    dataset
}

/// Options whose only compaction trigger is the overlay limit: base
/// fragments here are far below the default 1M-row target, which would
/// otherwise make them size-based compaction candidates on their own.
fn overlay_only_options(max_overlays_per_fragment: usize) -> CompactionOptions {
    CompactionOptions {
        max_overlays_per_fragment: Some(max_overlays_per_fragment),
        target_rows_per_fragment: 6,
        ..Default::default()
    }
}

/// Scan `id` and `val` and return an `id -> val` map (order-independent).
async fn id_val_map(dataset: &Dataset) -> BTreeMap<i32, Option<i32>> {
    let mut scanner = dataset.scan();
    scanner.project(&["id", "val"]).unwrap();
    let batch = scanner.try_into_batch().await.unwrap();
    let mut out = BTreeMap::new();
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let vals = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    for i in 0..batch.num_rows() {
        let v = if vals.is_null(i) {
            None
        } else {
            Some(vals.value(i))
        };
        out.insert(ids.value(i), v);
    }
    out
}
