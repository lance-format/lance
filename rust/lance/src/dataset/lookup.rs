// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Ordered, deletion-aware lookup of unique keys through ordinary scalar indices.

mod capture;
mod uint64;

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::{Arc, Mutex, Weak};

use arrow_array::{
    Array,
    cast::AsArray,
    make_array,
    types::{UInt32Type, UInt64Type},
};
use arrow_row::{RowConverter, SortField};
use arrow_schema::DataType;
use futures::TryStreamExt;
use lance_core::utils::address::RowAddress;
use lance_core::{Error, ROW_ADDR, ROW_ID, Result};
use lance_index::metrics::NoOpMetricsCollector;
use lance_index::scalar::{LOOKUP_KEY_INDEX, ScalarIndex, registry::VALUE_COLUMN_NAME};
use lance_io::object_store::ObjectStore;
use lance_table::format::overlay::staleness::overlay_affects_index;
use lance_table::format::{Fragment, IndexMetadata, Manifest, ROW_ID_FIELD_ID, RowIdMeta};
use lance_table::rowids::{read_row_ids, segment::U64Segment};
use object_store::path::Path;
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;

use self::uint64::UInt64KeyPositions;
use super::BaseObjectStores;
use crate::Dataset;
use crate::index::{DatasetIndexExt, scalar::open_scalar_index};
use crate::session::{caches::DSMetadataCache, index_caches::DSIndexCache};

/// One prepared column per dataset handle, shared by clones. Weak snapshot
/// identity also makes `Arc::make_mut` invalidate the entry before mutation.
/// Keeping this on the handle avoids sharing store-bound indices between two
/// independently opened datasets that happen to have the same version number.
#[derive(Default)]
pub struct LookupCache(Mutex<Option<LookupCacheEntry>>);

struct LookupCacheEntry {
    manifest: Weak<Manifest>,
    store: Weak<ObjectStore>,
    index_cache: Weak<DSIndexCache>,
    metadata_cache: Weak<DSMetadataCache>,
    base_stores: BaseObjectStores,
    manifest_path: Path,
    column: String,
    plan: Arc<OnceCell<PreparedLookup>>,
}

impl LookupCache {
    fn plan(&self, dataset: &Dataset, column: &str) -> Result<Arc<OnceCell<PreparedLookup>>> {
        let mut entry = self
            .0
            .lock()
            .map_err(|_| Error::internal("Lookup cache lock poisoned"))?;
        if let Some(entry) = entry.as_ref()
            && entry.column == column
            && std::ptr::eq(entry.manifest.as_ptr(), Arc::as_ptr(&dataset.manifest))
            && std::ptr::eq(entry.store.as_ptr(), Arc::as_ptr(&dataset.object_store))
            && std::ptr::eq(
                entry.index_cache.as_ptr(),
                Arc::as_ptr(&dataset.index_cache),
            )
            && std::ptr::eq(
                entry.metadata_cache.as_ptr(),
                Arc::as_ptr(&dataset.metadata_cache),
            )
            && Arc::ptr_eq(&entry.base_stores, &dataset.base_object_stores)
            && entry.manifest_path == dataset.manifest_location.path
        {
            return Ok(entry.plan.clone());
        }
        let plan = Arc::new(OnceCell::new());
        *entry = Some(LookupCacheEntry {
            manifest: Arc::downgrade(&dataset.manifest),
            store: Arc::downgrade(&dataset.object_store),
            index_cache: Arc::downgrade(&dataset.index_cache),
            metadata_cache: Arc::downgrade(&dataset.metadata_cache),
            base_stores: dataset.base_object_stores.clone(),
            manifest_path: dataset.manifest_location.path.clone(),
            column: column.to_owned(),
            plan: plan.clone(),
        });
        Ok(plan)
    }
}

struct LookupSegment {
    metadata: IndexMetadata,
    coverage: RoaringBitmap,
    index: OnceCell<Arc<dyn ScalarIndex>>,
}

#[derive(Default)]
struct LookupGroup {
    segments: Vec<LookupSegment>,
    coverage: RoaringBitmap,
}

struct InlineRange {
    ids: Range<u64>,
    fragment_id: u32,
    offset: u32,
}

struct PreparedLookup {
    segments: Vec<LookupSegment>,
    ranges: Vec<InlineRange>,
    /// Prefix maxima let point queries find all overlapping ranges, preserving
    /// duplicate-key detection even for invalid overlapping row-ID metadata.
    range_max_ends: Vec<u64>,
    uncovered: Vec<Fragment>,
}

impl PreparedLookup {
    async fn new(dataset: &Dataset, column: &str, field_id: i32) -> Result<Self> {
        let mut ranges = Vec::new();
        let mut inline_coverage = RoaringBitmap::new();
        if column == ROW_ID && dataset.manifest.uses_stable_row_ids() {
            for fragment in dataset.fragments().iter() {
                if let Some(fragment_ranges) = compact_inline_ranges(fragment)? {
                    inline_coverage.insert(fragment.id as u32);
                    ranges.extend(fragment_ranges);
                }
            }
        }
        ranges.sort_unstable_by_key(|range| range.ids.start);
        let mut max_end = 0;
        let range_max_ends = ranges
            .iter()
            .map(|range| {
                max_end = max_end.max(range.ids.end);
                max_end
            })
            .collect();
        let mut groups: BTreeMap<String, LookupGroup> = BTreeMap::new();
        if inline_coverage != *dataset.fragment_bitmap {
            let indices = dataset.load_indices().await?;
            for index in indices.iter().filter(|index| {
                index.keyed_field() == Some(field_id)
                    && (!dataset.manifest.uses_stable_row_ids() || index.results_are_row_addrs())
                    && index
                        .index_details
                        .as_ref()
                        .is_some_and(|details| details.type_url.ends_with("BTreeIndexDetails"))
            }) {
                let Some(coverage) = &index.fragment_bitmap else {
                    continue;
                };
                let mut coverage = (coverage & dataset.fragment_bitmap.as_ref()) - &inline_coverage;
                for fragment in dataset
                    .fragments()
                    .iter()
                    .filter(|fragment| !fragment.overlays.is_empty())
                {
                    if fragment.overlays.iter().any(|overlay| {
                        overlay_affects_index(
                            overlay,
                            index.keyed_fields(),
                            index.dataset_version,
                            dataset.schema(),
                        )
                    }) {
                        coverage.remove(fragment.id as u32);
                    }
                }
                if coverage.is_empty() {
                    continue;
                }
                let group = groups.entry(index.name.clone()).or_default();
                group.coverage |= &coverage;
                group.segments.push(LookupSegment {
                    metadata: index.clone(),
                    coverage,
                    index: OnceCell::new(),
                });
            }
        }
        let group = groups
            .into_values()
            .max_by_key(|group| group.coverage.len())
            .unwrap_or_default();
        let covered = group.coverage | inline_coverage;
        let uncovered = dataset
            .fragments()
            .iter()
            .filter(|fragment| !covered.contains(fragment.id as u32))
            .cloned()
            .collect();
        Ok(Self {
            segments: group.segments,
            ranges,
            range_max_ends,
            uncovered,
        })
    }

    fn lookup_ranges(&self, keys: &dyn Array) -> Vec<(usize, RowAddress)> {
        if self.ranges.is_empty() {
            return Vec::new();
        }
        let mut matches = Vec::with_capacity(keys.len());
        for (position, key) in keys.as_primitive::<UInt64Type>().iter().enumerate() {
            let Some(key) = key else { continue };
            let end = self.ranges.partition_point(|range| range.ids.start <= key);
            let start = self.range_max_ends[..end].partition_point(|&end| end <= key);
            for range in &self.ranges[start..end] {
                if range.ids.contains(&key) {
                    matches.push((
                        position,
                        RowAddress::new_from_parts(
                            range.fragment_id,
                            range.offset + (key - range.ids.start) as u32,
                        ),
                    ));
                }
            }
        }
        matches
    }

    /// Resolve just the indexed and inline fragments. Bulk captures call this
    /// with bounded batches and scan the disjoint uncovered fragments once.
    async fn lookup_indexed(
        &self,
        dataset: &Dataset,
        column: &str,
        keys: &dyn Array,
    ) -> Result<Vec<Option<RowAddress>>> {
        let mut addresses = vec![None; keys.len()];
        dataset
            .assign_lookup_matches(&mut addresses, self.lookup_ranges(keys), column)
            .await?;
        let lookup_keys = make_array(keys.to_data());
        for segment in &self.segments {
            let index = segment
                .index
                .get_or_try_init(|| {
                    open_scalar_index(dataset, column, &segment.metadata, &NoOpMetricsCollector)
                })
                .await?;
            let mut matches = index.lookup(lookup_keys.clone(), &NoOpMetricsCollector)?;
            // A legacy segment on a dataset without stable IDs reports _rowid,
            // whose values are physical addresses on that dataset.
            let row_column = if index.results_are_row_addresses() {
                ROW_ADDR
            } else {
                ROW_ID
            };
            while let Some(batch) = matches.try_next().await? {
                let key_indices = batch
                    .column_by_name(LOOKUP_KEY_INDEX)
                    .and_then(|column| column.as_primitive_opt::<UInt32Type>())
                    .ok_or_else(|| Error::index("Scalar lookup must return UInt32 key_index"))?;
                let row_addresses = batch
                    .column_by_name(row_column)
                    .and_then(|column| column.as_primitive_opt::<UInt64Type>())
                    .ok_or_else(|| {
                        Error::index(format!("Scalar lookup must return UInt64 {row_column}"))
                    })?;
                let mut live = Vec::with_capacity(batch.num_rows());
                for (key, address) in key_indices.iter().zip(row_addresses.iter()) {
                    let (Some(key), Some(address)) = (key, address) else {
                        return Err(Error::index(
                            "Scalar lookup returned a null key position or address",
                        ));
                    };
                    let address = RowAddress::from(address);
                    if segment.coverage.contains(address.fragment_id()) {
                        live.push((key as usize, address));
                    }
                }
                dataset
                    .assign_lookup_matches(&mut addresses, live, column)
                    .await?;
            }
        }
        Ok(addresses)
    }
}

/// Only decode a bounded amount of inline metadata. Large arrays, sparse
/// encodings and spilled lineage keep using the ordinary index/scan path.
fn compact_inline_ranges(fragment: &Fragment) -> Result<Option<Vec<InlineRange>>> {
    const MAX_INLINE_BYTES: usize = 4096;
    let Some(RowIdMeta::Inline(bytes)) = &fragment.row_id_meta else {
        return Ok(None);
    };
    if bytes.len() > MAX_INLINE_BYTES {
        return Ok(None);
    }
    let sequence = read_row_ids(bytes)?;
    if sequence.len() > u32::MAX as u64 {
        return Ok(None);
    }
    let mut ranges = Vec::with_capacity(sequence.segments().len());
    let mut offset = 0;
    for segment in sequence.segments() {
        let U64Segment::Range(ids) = segment else {
            return Ok(None);
        };
        if !ids.is_empty() {
            ranges.push(InlineRange {
                ids: ids.clone(),
                fragment_id: fragment.id as u32,
                offset,
            });
        }
        offset += segment.len() as u32;
    }
    Ok(Some(ranges))
}

impl Dataset {
    /// Resolve unique keys to live physical row addresses, preserving input order.
    ///
    /// Repeated keys repeat their address; null, missing, or deleted keys yield
    /// `None`. Multiple live matches for a requested key are an error. Compact
    /// inline row-ID ranges use arithmetic, ordinary address-domain BTree indices
    /// supply covered fragments, and remaining fragments scan the column in batches.
    pub(crate) async fn lookup_row_addresses(
        &self,
        column: &str,
        keys: &dyn Array,
    ) -> Result<Vec<Option<RowAddress>>> {
        let (field_id, data_type) = if column == ROW_ID && self.manifest.uses_stable_row_ids() {
            (ROW_ID_FIELD_ID, DataType::UInt64)
        } else {
            let field = self.schema().field(column).ok_or_else(|| {
                Error::invalid_input(format!("Lookup column '{column}' does not exist"))
            })?;
            (field.id, field.data_type())
        };
        if keys.data_type() != &data_type {
            return Err(Error::invalid_input(format!(
                "Lookup keys for '{column}' have type {}, expected {data_type}",
                keys.data_type()
            )));
        }
        if u32::try_from(keys.len()).is_err() {
            return Err(Error::invalid_input(format!(
                "Lookup accepts at most {} keys, got {}",
                u32::MAX,
                keys.len()
            )));
        }
        if keys.len() == keys.null_count() {
            return Ok(vec![None; keys.len()]);
        }
        let cell = self.lookup_cache.plan(self, column)?;
        let plan = cell
            .get_or_try_init(|| PreparedLookup::new(self, column, field_id))
            .await?;
        let mut addresses = plan.lookup_indexed(self, column, keys).await?;
        if plan.uncovered.is_empty() {
            return Ok(addresses);
        }
        // Retain only the requested keys and consume scan batches incrementally;
        // an uncovered tail never constructs another dataset-wide reverse index.
        let keys = make_array(keys.to_data());
        let uint64_positions = keys
            .as_primitive_opt::<UInt64Type>()
            .map(UInt64KeyPositions::new);
        let converter = RowConverter::new(vec![SortField::new(keys.data_type().clone())])?;
        let key_rows = if uint64_positions.is_none() {
            Some(converter.convert_columns(std::slice::from_ref(&keys))?)
        } else {
            None
        };
        let mut positions = HashMap::new();
        if let Some(key_rows) = &key_rows {
            positions.reserve(keys.len() - keys.null_count());
            for position in 0..keys.len() {
                if keys.is_valid(position) {
                    positions
                        .entry(key_rows.row(position))
                        .or_insert_with(Vec::new)
                        .push(position);
                }
            }
        }
        let mut scan = self.scan();
        scan.with_fragments(plan.uncovered.clone())
            .with_row_address()
            .use_scalar_index(false)
            .project_with_transform(&[(VALUE_COLUMN_NAME, column)])?;
        // Scans can themselves contain row-ID takes. Box this boundary so the
        // two async execution paths do not recursively expand their future types.
        let mut stream = Box::pin(scan.try_into_stream()).await?;
        while let Some(batch) = stream.try_next().await? {
            let values = batch
                .column_by_name(VALUE_COLUMN_NAME)
                .ok_or_else(|| Error::internal("Lookup scan omitted the key column"))?;
            let row_addresses = batch
                .column_by_name(ROW_ADDR)
                .ok_or_else(|| Error::internal("Lookup scan omitted the row address column"))?
                .as_primitive::<UInt64Type>();
            // Stable IDs need exact integer equality. Avoid encoding and hashing
            // Arrow rows for every scanned value, including nonmatching values.
            if let Some(positions) = &uint64_positions {
                positions.assign_matches(
                    values.as_primitive::<UInt64Type>(),
                    row_addresses,
                    &mut addresses,
                    column,
                )?;
                continue;
            }
            let rows = converter.convert_columns(std::slice::from_ref(values))?;
            for offset in 0..batch.num_rows() {
                if values.is_null(offset) || row_addresses.is_null(offset) {
                    continue;
                }
                if let Some(keys) = positions.get(&rows.row(offset)) {
                    let address = RowAddress::from(row_addresses.value(offset));
                    for key in keys {
                        assign_unique_address(&mut addresses, *key, address, column)?;
                    }
                }
            }
        }
        Ok(addresses)
    }

    async fn assign_lookup_matches(
        &self,
        addresses: &mut [Option<RowAddress>],
        matches: Vec<(usize, RowAddress)>,
        column: &str,
    ) -> Result<()> {
        let mut needs_io: BTreeMap<u32, Vec<(usize, RowAddress)>> = BTreeMap::new();
        for (key, address) in matches {
            let Some(fragment) = self.find_fragment(address.fragment_id() as u64) else {
                continue;
            };
            if fragment.files.is_empty() {
                return Err(Error::not_found(format!(
                    "Fragment {} does not contain any data",
                    fragment.id
                )));
            }
            if self.manifest.writer_version.is_some()
                && fragment.deletion_file.is_none()
                && let Some(rows) = fragment.physical_rows
            {
                validate_lookup_offset(address, rows)?;
                assign_unique_address(addresses, key, address, column)?;
            } else {
                needs_io
                    .entry(address.fragment_id())
                    .or_default()
                    .push((key, address));
            }
        }
        for (fragment_id, matches) in needs_io {
            let Some(fragment) = self.get_fragment(fragment_id as usize) else {
                continue;
            };
            let deletions = fragment.get_deletion_vector().await?;
            let rows = fragment.physical_rows().await?;
            for (key, address) in matches {
                validate_lookup_offset(address, rows)?;
                if deletions
                    .as_ref()
                    .is_some_and(|deleted| deleted.contains(address.row_offset()))
                {
                    continue;
                }
                assign_unique_address(addresses, key, address, column)?;
            }
        }
        Ok(())
    }
}

fn validate_lookup_offset(address: RowAddress, physical_rows: usize) -> Result<()> {
    if address.row_offset() as usize >= physical_rows {
        return Err(Error::index(format!(
            "Lookup returned address {address} outside fragment {} with {physical_rows} rows",
            address.fragment_id()
        )));
    }
    Ok(())
}

fn assign_unique_address(
    addresses: &mut [Option<RowAddress>],
    key: usize,
    address: RowAddress,
    column: &str,
) -> Result<()> {
    let num_keys = addresses.len();
    let entry = addresses.get_mut(key).ok_or_else(|| {
        Error::index(format!(
            "Lookup returned key position {key} for {num_keys} keys"
        ))
    })?;
    if let Some(previous) = *entry
        && previous != address
    {
        return Err(Error::invalid_input(format!(
            "Lookup column '{column}' is not unique for key at position {key}: live addresses {previous} and {address}"
        )));
    }
    *entry = Some(address);
    Ok(())
}
