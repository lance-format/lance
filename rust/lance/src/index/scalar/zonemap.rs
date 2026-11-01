// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_schema::DataType;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{FutureExt, StreamExt, TryStreamExt};
use itertools::Itertools;
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::lance_format::LanceIndexStore;
use lance_index::scalar::zonemap::{
    HarvestedZoneSeed, NULL_BITMAP_POLICY_META_KEY, ReusedZones, ScannedZones, SeedOutcome,
    ZONEMAP_INDEX_FILENAME, ZoneMapIndex, ZoneMapIndexBuilderParams, ZoneMapSegmentBase,
    train_zonemap_scan, write_zonemap_segment,
};
use lance_index::scalar::{
    CreatedIndex, IndexStore, ScalarIndex, TrainingCriteria, index_files_to_table,
};
use lance_select::RowAddrTreeMap;
use lance_table::format::{Fragment, IndexMetadata};
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::index::append::SeedHarvester;
use crate::index::scalar::{TrainingRows, scan_training_data};
use crate::{Dataset, Error, Result, dataset::index::LanceIndexStoreExt};

/// Merge one caller-defined group of source ZoneMap segments into a single segment.
pub(in crate::index) async fn merge_segments(
    dataset: &Dataset,
    segments: Vec<IndexMetadata>,
    staged: Option<&crate::index::frag_reuse::StagedRemappingPlans>,
) -> Result<IndexMetadata> {
    if segments.is_empty() {
        return Err(Error::index("No segment metadata was provided".to_string()));
    }

    let field_id = *segments[0].fields.first().ok_or_else(|| {
        Error::invalid_input(format!(
            "CreateIndex: segment {} is missing field ids",
            segments[0].uuid
        ))
    })?;
    let field_path = dataset.schema().field_path(field_id)?;

    let mut scalar_indices = Vec::with_capacity(segments.len());
    let mut fragment_bitmap = RoaringBitmap::new();
    let dataset_fragments = dataset.fragment_bitmap.as_ref();
    for segment in &segments {
        let effective = segment
            .effective_fragment_bitmap(dataset_fragments)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "CreateIndex: segment {} is missing fragment coverage",
                    segment.uuid
                ))
            })?;
        fragment_bitmap |= effective;
        let scalar_index = super::open_scalar_index_with_plan(
            dataset,
            &field_path,
            segment,
            staged.and_then(|plans| plans.get(&segment.uuid)),
            crate::index::frag_reuse::OpenPurpose::Maintenance,
            &NoOpMetricsCollector,
        )
        .await?;
        scalar_indices.push((segment.uuid, scalar_index));
    }

    let mut source_indices = Vec::with_capacity(scalar_indices.len());
    for (segment_uuid, scalar_index) in &scalar_indices {
        let zonemap_index = scalar_index
            .as_any()
            .downcast_ref::<ZoneMapIndex>()
            .ok_or_else(|| {
                Error::index(format!(
                    "merge_existing_index_segments: expected zonemap segment {}, got {:?}",
                    segment_uuid,
                    scalar_index.index_type()
                ))
            })?;
        source_indices.push(zonemap_index);
    }

    let new_uuid = Uuid::new_v4();
    let new_store = LanceIndexStore::from_dataset_for_new(dataset, &new_uuid)?;
    let created_index = lance_index::scalar::zonemap::merge_zonemap_indices(
        &source_indices,
        &new_store,
        &fragment_bitmap,
    )
    .await?;

    Ok(IndexMetadata {
        uuid: new_uuid,
        dataset_version: dataset.manifest.version,
        fragment_bitmap: Some(fragment_bitmap),
        index_details: Some(Arc::new(created_index.index_details)),
        index_version: created_index.index_version as i32,
        created_at: Some(chrono::Utc::now()),
        base_id: None,
        files: Some(index_files_to_table(created_index.files)),
        clustering_id: None,
        metadata: Default::default(),
        ..segments[0].clone()
    })
}

/// What a seed-aware segment build reused and scanned, recorded for tests.
#[cfg(test)]
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SeedReuseReport {
    /// Fragments whose every zone came from whole seed records.
    pub reused_fragments: Vec<u64>,
    /// Fragments with some zones from seeds and scanned intervals.
    pub partial_fragments: Vec<u64>,
    pub reused_records: usize,
    pub zones_from_seeds: usize,
    /// Fragments scanned whole.
    pub scanned_fragments: Vec<u64>,
    /// Physical rows selected for the scan (whole fragments plus intervals).
    pub scanned_rows: u64,
    /// Scans issued: batches of decided fragments with rows left to read.
    pub scans: usize,
    /// Most raw seed bytes owned by the build at any one moment (live
    /// ownership, decremented when a fragment's buffer is dropped).
    pub max_live_seed_bytes: usize,
    /// Most fragment harvests in flight at once (the task bound; one
    /// harvest may issue several storage reads).
    pub max_concurrent_harvests: usize,
    /// The harvest concurrency the build ran with.
    pub harvest_concurrency: usize,
    /// Raw seed bytes harvested over the whole build.
    pub total_seed_bytes: usize,
    /// Whether a physical row mask was given to the scan.
    pub masked: bool,
}

#[cfg(test)]
tokio::task_local! {
    /// Collects one [`SeedReuseReport`] per seed-aware segment build.
    pub static SEED_REUSE_REPORTS: std::sync::Mutex<Vec<SeedReuseReport>>;
}

/// The physical row count of `fragment`, established before any seed
/// decision so every zone's length is known. Resolved through
/// [`FileFragment::physical_rows`], which serves the manifest's count
/// without I/O only when the manifest's writer version makes it
/// trustworthy and otherwise reads a data file's footer.
async fn physical_rows(dataset: &Dataset, fragment: &Fragment) -> Result<u64> {
    let file_fragment = dataset.get_fragment(fragment.id as usize).ok_or_else(|| {
        Error::invalid_input(format!("fragment {} is not in the dataset", fragment.id))
    })?;
    Ok(file_fragment.physical_rows().await? as u64)
}

/// Most seed footers read at once by [`assemble_segment`]. The store's
/// parallelism bounds it below this; the cap keeps the window of raw seed
/// buffers small on stores that report hundreds of parallel slots.
const ZONEMAP_HARVEST_CONCURRENCY_CAP: usize = 16;

pub fn harvest_concurrency(dataset: &Dataset) -> usize {
    dataset
        .object_store
        .io_parallelism()
        .clamp(1, ZONEMAP_HARVEST_CONCURRENCY_CAP)
}

/// Live counts of a bounded harvest: fragments in flight and raw seed bytes
/// held (read and not yet folded), each with its peak.
#[derive(Default)]
struct HarvestCounters {
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    live_seed_bytes: AtomicUsize,
    max_live_seed_bytes: AtomicUsize,
}

impl HarvestCounters {
    fn enter(&self) -> InFlightHarvest<'_> {
        let live = self.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
        self.max_in_flight.fetch_max(live, Ordering::Relaxed);
        InFlightHarvest(self)
    }

    fn hold(&self, bytes: usize) {
        let live = self.live_seed_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.max_live_seed_bytes.fetch_max(live, Ordering::Relaxed);
    }

    fn release(&self, bytes: usize) {
        self.live_seed_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }
}

struct InFlightHarvest<'a>(&'a HarvestCounters);

impl Drop for InFlightHarvest<'_> {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Decided fragments per scan in [`assemble_segment`]. One scanner
/// pipelines its fragments' reads and is planned once, so fragments are
/// scanned in batches of this many rather than one scanner per fragment.
const ZONEMAP_SCAN_BATCH_FRAGMENTS: usize = 16;

/// Most batches scanned at once. Each scan already reads at the store's
/// parallelism; a second one in flight only covers the gap between batches.
const ZONEMAP_SCAN_CONCURRENCY_CAP: usize = 2;

fn scan_concurrency(dataset: &Dataset) -> usize {
    (dataset.object_store.io_parallelism() / 16).clamp(1, ZONEMAP_SCAN_CONCURRENCY_CAP)
}

/// How a fragment's rows are read once its seed is decided.
enum FragmentScan {
    /// Every target zone came from the seed.
    None,
    /// The fragment is read whole.
    Whole,
    /// Only these physical rows are read: the seed records that cross a
    /// target boundary.
    Intervals(RowAddrTreeMap),
}

/// One fragment's seed decision, made before any of its rows are read.
struct FragmentPlan<'a> {
    fragment: &'a Fragment,
    physical_rows: u64,
    /// Zones this fragment's seed supplied, if any.
    reused: ReusedZones,
    scan: FragmentScan,
    scanned_rows: u64,
    seed_bytes: usize,
}

/// A batch of decided fragments with the zone pieces their scan produced.
struct BatchResult<'a> {
    plans: Vec<FragmentPlan<'a>>,
    scanned: Option<ScannedZones>,
}

/// Decide one fragment: read its seed, fold the records that fit the
/// target zones, and leave the rest to a scan of this fragment alone. The
/// raw seed buffer is released before the plan is returned.
#[allow(clippy::too_many_arguments)]
fn plan_fragment<'a>(
    dataset: &'a Dataset,
    harvester: Option<&'a SeedHarvester<'a>>,
    fragment: &'a Fragment,
    counters: &'a HarvestCounters,
    rows_per_zone: u64,
    data_type: &'a DataType,
    keep_positions: bool,
    legacy_storage: bool,
) -> BoxFuture<'a, Result<FragmentPlan<'a>>> {
    async move {
        let _in_flight = counters.enter();
        let physical_rows = physical_rows(dataset, fragment).await?;
        let mut reused = ReusedZones::new(keep_positions);
        let mut seed_bytes = 0;
        let seed = match harvester {
            Some(harvester) => harvester.harvest(fragment).await?,
            None => None,
        };
        let outcome = match seed {
            Some(seed) => {
                seed_bytes = seed.bytes.len();
                counters.hold(seed_bytes);
                // A deletion file would mix physical and live populations in one
                // zone, and V1 storage cannot take a physical mask: those
                // fragments are reused whole or scanned whole.
                let allow_partial = !legacy_storage && fragment.deletion_file.is_none();
                let harvested = HarvestedZoneSeed {
                    seed,
                    physical_rows,
                };
                let outcome = reused.reuse(&harvested, rows_per_zone, data_type, allow_partial);
                drop(harvested);
                counters.release(seed_bytes);
                outcome?
            }
            None => SeedOutcome::Rejected,
        };
        let base_addr = fragment.id << 32;
        let (scan, scanned_rows) = match outcome {
            SeedOutcome::Complete => (FragmentScan::None, 0),
            SeedOutcome::Partial(intervals) => {
                let mut mask = RowAddrTreeMap::new();
                let mut rows = 0;
                for interval in intervals {
                    rows += interval.end - interval.start;
                    mask.insert_range(base_addr + interval.start..base_addr + interval.end);
                }
                (FragmentScan::Intervals(mask), rows)
            }
            SeedOutcome::Rejected => (FragmentScan::Whole, physical_rows),
        };
        Ok(FragmentPlan {
            fragment,
            physical_rows,
            reused,
            scan,
            scanned_rows,
            seed_bytes,
        })
    }
    .boxed()
}

/// Read, in one scan, the rows the `plans` of a batch left to it and cut
/// them into their fragments' target zones. Whole fragments and interval
/// fragments share the scan through one physical mask; a batch with
/// nothing to scan reads nothing.
fn scan_batch<'a>(
    dataset: &'a Dataset,
    column: &'a str,
    criteria: &'a TrainingCriteria,
    data_type: &'a DataType,
    rows_per_zone: u64,
    keep_positions: bool,
    plans: Result<Vec<FragmentPlan<'a>>>,
) -> BoxFuture<'a, Result<BatchResult<'a>>> {
    async move {
        let plans = plans?;
        let mut fragments = Vec::new();
        let mut physical_rows = HashMap::new();
        let mut mask = RowAddrTreeMap::new();
        let mut masked = false;
        let mut batch_rows = 0u64;
        for plan in &plans {
            match &plan.scan {
                FragmentScan::None => continue,
                FragmentScan::Whole => {
                    let base_addr = plan.fragment.id << 32;
                    mask.insert_range(base_addr..base_addr + plan.physical_rows);
                }
                FragmentScan::Intervals(intervals) => {
                    mask |= intervals;
                    masked = true;
                }
            }
            fragments.push(plan.fragment.clone());
            physical_rows.insert(plan.fragment.id, plan.physical_rows);
            batch_rows += plan.scanned_rows;
        }
        if fragments.is_empty() {
            return Ok(BatchResult {
                plans,
                scanned: None,
            });
        }
        // Only interval fragments need the mask; without one the whole
        // fragments are read as such, which V1 storage also accepts. The
        // batch's own row count is the progress total: nothing counts the
        // dataset per batch.
        let scan = scan_training_data(
            dataset,
            column,
            criteria,
            Some(TrainingRows::Fragments(fragments)),
            masked.then_some(mask),
            Some(batch_rows),
        )
        .await?;
        let scanned = train_zonemap_scan(
            data_type,
            rows_per_zone,
            keep_positions,
            physical_rows,
            scan,
        )
        .await?;
        Ok(BatchResult {
            plans,
            scanned: Some(scanned),
        })
    }
    .boxed()
}

/// Assemble one segment over `fragments`, fragment by fragment: each reads
/// its seed, folds the records that fit the target zones, and leaves only
/// what the seed did not cover to a scan. Decided fragments are scanned in
/// batches of [`ZONEMAP_SCAN_BATCH_FRAGMENTS`], each batch as soon as it is
/// decided, so scanning overlaps the seed reads of later fragments. Up to
/// [`harvest_concurrency`] seeds are read at once and [`scan_concurrency`]
/// batches scanned at once; results fold in fragment order, and a raw seed
/// buffer is held only until its fragment is decided.
#[allow(clippy::too_many_arguments)]
async fn assemble_segment(
    dataset: &Dataset,
    column: &str,
    base: ZoneMapSegmentBase<'_>,
    data_type: &DataType,
    fragments: &[Fragment],
    harvester: Option<SeedHarvester<'_>>,
    criteria: &TrainingCriteria,
    store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let rows_per_zone = base.rows_per_zone();
    let keep_positions = base.keeps_null_positions();
    let legacy_storage = dataset.manifest.data_storage_format.lance_file_format()
        == lance_file::version::ConcreteFileVersion::V1;
    let counters = HarvestCounters::default();
    let harvester = harvester.as_ref();
    let concurrency = harvest_concurrency(dataset);
    let scans = scan_concurrency(dataset);
    // Type-erased so the enclosing future's `Send` bound does not depend
    // on the stages' borrowed lifetimes.
    let mut results: BoxStream<'_, Result<BatchResult<'_>>> =
        futures::stream::iter(fragments.iter())
            .map(|fragment| {
                plan_fragment(
                    dataset,
                    harvester,
                    fragment,
                    &counters,
                    rows_per_zone,
                    data_type,
                    keep_positions,
                    legacy_storage,
                )
            })
            .buffered(concurrency)
            .try_chunks(ZONEMAP_SCAN_BATCH_FRAGMENTS)
            .map(|plans| {
                scan_batch(
                    dataset,
                    column,
                    criteria,
                    data_type,
                    rows_per_zone,
                    keep_positions,
                    plans.map_err(|err| err.1),
                )
            })
            .buffered(scans)
            .boxed();

    let mut reused = ReusedZones::new(keep_positions);
    let mut scanned = ScannedZones::default();
    let mut scanned_fragments: Vec<u64> = Vec::new();
    let mut scanned_rows = 0u64;
    let mut total_seed_bytes = 0usize;
    let mut scan_count = 0usize;
    while let Some(BatchResult {
        plans,
        scanned: pieces,
    }) = results.try_next().await?
    {
        for plan in plans {
            total_seed_bytes += plan.seed_bytes;
            scanned_rows += plan.scanned_rows;
            if !matches!(plan.scan, FragmentScan::None) {
                scanned_fragments.push(plan.fragment.id);
            }
            reused.merge(plan.reused);
        }
        if let Some(pieces) = pieces {
            scan_count += 1;
            scanned.merge(pieces);
        }
    }
    // The pipeline is drained; release its borrow of the counters.
    drop(results);
    #[cfg(test)]
    let masked = !legacy_storage && !scanned_fragments.is_empty();
    let max_live_seed_bytes = counters.max_live_seed_bytes.into_inner();
    let max_concurrent_harvests = counters.max_in_flight.into_inner();
    log::info!(
        "zone map segment for {column}: {} fragments from seeds, {} partly, {} scanned in {scan_count} scans ({scanned_rows} rows, {} seed records, {total_seed_bytes} seed bytes, at most {max_live_seed_bytes} held at once over {max_concurrent_harvests} concurrent harvests)",
        reused.reused_fragments.len(),
        reused.partial_fragments.len(),
        scanned_fragments.len(),
        reused.reused_records
    );
    #[cfg(test)]
    let _ = SEED_REUSE_REPORTS.try_with(|reports| {
        reports.lock().unwrap().push(SeedReuseReport {
            reused_fragments: reused.reused_fragments.clone(),
            partial_fragments: reused.partial_fragments.clone(),
            reused_records: reused.reused_records,
            zones_from_seeds: reused.num_zones(),
            scanned_fragments: scanned_fragments.clone(),
            scanned_rows,
            scans: scan_count,
            max_live_seed_bytes,
            max_concurrent_harvests,
            harvest_concurrency: concurrency,
            total_seed_bytes,
            masked,
        })
    });
    let refreshed: Vec<u64> = fragments.iter().map(|fragment| fragment.id).collect();
    write_zonemap_segment(base, data_type, reused, scanned, &refreshed, store).await
}

/// The fragments with these ids, in id order, erroring on an unknown id.
fn fragments_by_id(dataset: &Dataset, ids: Vec<u32>) -> Result<Vec<Fragment>> {
    let ids: Vec<u32> = ids.into_iter().sorted().dedup().collect();
    ids.iter()
        .zip(dataset.get_frags_from_ordered_ids(&ids))
        .map(|(id, fragment)| {
            fragment.map(|f| f.metadata().clone()).ok_or_else(|| {
                Error::invalid_input_source(format!("No fragment with id {}", id).into())
            })
        })
        .collect()
}

/// Train a new ZoneMap segment over `fragment_ids` (all fragments when
/// `None`), reusing the write seeds those fragments already carry and
/// scanning only the rows they do not cover.
pub(in crate::index) async fn build_segment_with_seeds(
    dataset: &Dataset,
    column: &str,
    params: ZoneMapIndexBuilderParams,
    data_type: &DataType,
    fragment_ids: Option<Vec<u32>>,
    criteria: &TrainingCriteria,
    store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let fragments = match fragment_ids {
        Some(ids) => fragments_by_id(dataset, ids)?,
        None => dataset.fragments().as_ref().clone(),
    };
    // An index that does not request seeds reads none, as on update.
    let harvester = params
        .use_seeds_for(data_type)
        .then(|| SeedHarvester::for_column(dataset, column))
        .flatten();
    assemble_segment(
        dataset,
        column,
        ZoneMapSegmentBase::New(params),
        data_type,
        &fragments,
        harvester,
        criteria,
        store,
    )
    .await
}

/// Append `unindexed` fragments to the ZoneMap segment `reference_index`,
/// reusing their write seeds where usable when `harvest` is set. `None`
/// when the reference is not a ZoneMap segment, in which case the generic
/// update path applies.
#[allow(clippy::too_many_arguments)]
pub(in crate::index) async fn update_segment(
    dataset: &Dataset,
    reference_index: &Arc<dyn ScalarIndex>,
    reference: &IndexMetadata,
    column: &str,
    unindexed: &[Fragment],
    criteria: &TrainingCriteria,
    harvest: bool,
    store: &dyn IndexStore,
) -> Result<Option<CreatedIndex>> {
    let Some(zone_map) = reference_index.as_any().downcast_ref::<ZoneMapIndex>() else {
        return Ok(None);
    };
    let harvester = if harvest {
        reference
            .keyed_field()
            .and_then(|field_id| SeedHarvester::for_field_id(dataset, field_id))
    } else {
        None
    };
    let data_type = zone_map.value_type().clone();
    assemble_segment(
        dataset,
        column,
        ZoneMapSegmentBase::Update(zone_map),
        &data_type,
        unindexed,
        harvester,
        criteria,
        store,
    )
    .await
    .map(Some)
}

/// The persisted null-position policy of a ZoneMap segment, read from its
/// index file's schema metadata (one footer read, served from the file
/// metadata cache afterwards). `true` unless the index was created with
/// `store_null_bitmap: false`; an index without the key (legacy or default)
/// keeps the default.
pub async fn stored_null_bitmap_policy(dataset: &Dataset, index: &IndexMetadata) -> Result<bool> {
    let store = LanceIndexStore::from_dataset_for_existing(dataset, index).await?;
    let file = store.open_index_file(ZONEMAP_INDEX_FILENAME).await?;
    Ok(file
        .schema()
        .metadata
        .get(NULL_BITMAP_POLICY_META_KEY)
        .is_none_or(|value| value != "false"))
}
