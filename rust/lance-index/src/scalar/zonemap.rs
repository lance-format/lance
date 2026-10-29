// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Zone Map Index
//!
//! Zone maps are a columnar database technique for predicate pushdown and scan pruning.
//! They break data into fixed-size chunks called "zones" and maintain summary statistics
//! (min, max, null count) for each zone. This enables efficient filtering by eliminating
//! zones that cannot contain matching values.
//!
//! Zone maps are "inexact" filters - they can definitively exclude zones but may include
//! false positives that require rechecking.
//!
//!
use crate::pbold;
use crate::scalar::expression::{SargableQueryParser, ScalarQueryParser};
use crate::scalar::registry::{
    BasicTrainer, ScalarIndexPlugin, TrainingCriteria, TrainingOrdering, TrainingRequest,
};
use crate::scalar::seed::IndexSeedWriter;
use crate::scalar::{
    BuiltinIndexType, CreatedIndex, IndexFile, SargableQuery, ScalarIndexParams, UpdateCriteria,
    compute_next_prefix,
};
use lance_arrow_stats::StatisticsAccumulator;
use lance_core::cache::{LanceCache, WeakLanceCache};
use lance_core::utils::row_addr_remap::RowAddrRemap;
use lance_index_core::remapping::RowAddrTranslator;
use lance_index_core::remapping::{BatchRowIdRemapper, remap_row_addrs_tree_map_async};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::sync::LazyLock;

use arrow_array::{
    ArrayRef, RecordBatch, UInt32Array, UInt64Array, new_empty_array, new_null_array,
};
use arrow_schema::{DataType, Field};
use datafusion_common::ScalarValue;
use datafusion_execution::SendableRecordBatchStream;
use lance_select::{RowAddrTreeMap, RowSetOps};
use std::{collections::HashMap, sync::Arc};

use super::{AnyQuery, IndexStore, MetricsCollector, ScalarIndex, SearchResult};
use crate::scalar::RowIdRemapper;
use crate::{Index, IndexType};
use async_trait::async_trait;
use lance_core::Error;
use lance_core::Result;
use lance_core::deepsize::DeepSizeOf;
use roaring::RoaringBitmap;

use super::zoned::{ZoneBound, ZoneProcessor, ZoneTrainer, search_zones};
use std::collections::HashSet;

/// `rows` without the positions of `fragments`.
fn without_fragments(
    rows: &RowAddrTreeMap,
    fragments: impl IntoIterator<Item = u64>,
) -> RowAddrTreeMap {
    let removed: HashSet<u32> = fragments
        .into_iter()
        .filter_map(|fragment| u32::try_from(fragment).ok())
        .collect();
    let mut rows = rows.clone();
    let kept: Vec<u32> = rows
        .iter()
        .map(|(fragment, _)| *fragment)
        .filter(|fragment| !removed.contains(fragment))
        .collect();
    rows.retain_fragments(kept);
    rows
}
pub const ROWS_PER_ZONE_DEFAULT: u64 = 8192; // 1 zone every two batches

const ZONEMAP_FILENAME: &str = "zonemap.lance";
const ZONEMAP_SIZE_META_KEY: &str = "rows_per_zone";
const NULL_BITMAP_META_KEY: &str = "null_bitmap";
/// Index-file metadata recording that the index was configured without a
/// null bitmap. Absent on every index built before the option existed and on
/// indexes that keep one, so a missing bitmap on an index without this key
/// still means "legacy, positions unknown" rather than "not wanted".
pub const NULL_BITMAP_POLICY_META_KEY: &str = "store_null_bitmap";
/// The index file of a ZoneMap segment, whose schema metadata carries
/// `rows_per_zone` and the null bitmap policy.
pub const ZONEMAP_INDEX_FILENAME: &str = ZONEMAP_FILENAME;
const SEED_NULL_BITMAP_META_KEY: &str = "seed_null_bitmap";
/// Seed schema-metadata key carrying the seed format version. Absent on
/// version 0 seeds (see `docs/src/format/index/scalar/zone_write_seeds.md`).
const SEED_VERSION_META_KEY: &str = "version";
/// The seed format version this writer emits.
const SEED_FORMAT_VERSION: &str = "1";
/// The ZoneMap index file versions this build reads and writes
/// (`IndexMetadata.index_version`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneMapIndexVersion {
    /// Seven columns. `max` may hold the `+NaN` marker for a zone with NaNs
    /// and the NaN sign split is not stored. Never written for a new segment.
    V0 = 0,
    /// Adds the non-null `negative_nan_count` column; `min`/`max` are
    /// ordinary bounds that exclude NaNs.
    V1 = 1,
}

impl ZoneMapIndexVersion {
    /// The version new segments are written with.
    pub const CURRENT: Self = Self::V1;

    /// The version a segment's metadata declares; `NotSupported` for one
    /// this build does not know.
    pub fn from_metadata(index_version: u32) -> Result<Self> {
        match index_version {
            0 => Ok(Self::V0),
            1 => Ok(Self::V1),
            other => Err(Error::not_supported_source(
                format!(
                    "zone map index version {other} is not supported by this build \
                     (supported: 0 and 1); please upgrade Lance"
                )
                .into(),
            )),
        }
    }
}

const ZONEMAP_INDEX_VERSION: u32 = ZoneMapIndexVersion::CURRENT as u32;

/// The invariants a version 1 zone must satisfy, checked when a file is
/// read and before one is written: the sign split fits the NaN count, the
/// counts fit the zone, and no bound is a NaN (nested types store
/// placeholder bounds that are never consulted). Returns the violation.
fn v1_zone_violation(
    min: &ScalarValue,
    max: &ScalarValue,
    null_count: u32,
    nan_count: u32,
    negative_nan_count: u32,
    length: u64,
    nested: bool,
) -> Option<String> {
    if negative_nan_count > nan_count {
        return Some(format!(
            "negative_nan_count {negative_nan_count} above nan_count {nan_count}"
        ));
    }
    if u64::from(null_count) + u64::from(nan_count) > length {
        return Some(format!(
            "null_count {null_count} plus nan_count {nan_count} above its length {length}"
        ));
    }
    if !nested && (ZoneMapIndex::scalar_is_nan(min) || ZoneMapIndex::scalar_is_nan(max)) {
        return Some("a NaN bound".to_string());
    }
    None
}

/// One zone of a loaded index, as exposed by [`ZoneMapIndex::zone_summaries`].
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct ZoneSummary {
    pub bound: ZoneBound,
    pub min: ScalarValue,
    pub max: ScalarValue,
    pub null_count: u32,
    pub nan_count: u32,
}

/// Basic stats about zonemap index
#[derive(Debug, PartialEq, Clone)]
pub(crate) struct ZoneMapStatistics {
    min: ScalarValue,
    max: ScalarValue,
    null_count: u32,
    // only apply to float type
    nan_count: u32,
    /// NaNs with the sign bit set (`<= nan_count`). `None` when the source
    /// cannot split the sign: zones loaded from ZoneMap index files (the file
    /// format has no column for it) and version 0 seeds. Version 1 seeds and
    /// freshly computed zones carry `Some`.
    negative_nan_count: Option<u32>,
    // Bound of this zone within the fragment. Persisted as three separate columns
    // (fragment_id, zone_start, zone_length) in the index file.
    bound: ZoneBound,
}

impl DeepSizeOf for ZoneMapStatistics {
    fn deep_size_of_children(&self, _context: &mut lance_core::deepsize::Context) -> usize {
        // Estimate sizes for ScalarValue
        let min_size = self.min.size() - std::mem::size_of::<ScalarValue>();
        let max_size = self.max.size() - std::mem::size_of::<ScalarValue>();

        min_size + max_size
    }
}

impl AsRef<ZoneBound> for ZoneMapStatistics {
    fn as_ref(&self) -> &ZoneBound {
        &self.bound
    }
}

/// ZoneMap index
/// At high level it's a columnar database technique for predicate push down and scan pruning.
/// It breaks data into fixed-size chunks called `zones` and store summary statistics(min, max, null_count,
/// nan_count, fragment_id, local_row_offset) for each zone. It enables efficient filtering by skipping zones that do not contain matching values
///
/// This is an inexact filter, similar to a bloom filter. It can return false positives that require rechecking.
///
/// Note that it cannot return false negatives.
/// Input:
/// * Fragment 1: - 10 rows   -> 0  -> 9
/// * Fragment 2: - 7 rows    -> 10 -> 16
/// * Fragment 3: - 4 rows    -> 20 -> 23
/// * Zone size AKA “rows_per_zone” (from user) - 5
///
/// Output:
/// fragment id | min | max | zone_length
/// 1           | 0   |  4  | 5
/// 1           | 5   |  9  | 5
/// 2           | 10  | 14  | 5
/// 2           | 15  | 16  | 2
/// 3           | 20  | 23  | 4
pub struct ZoneMapIndex {
    zones: Vec<ZoneMapStatistics>,
    data_type: DataType,
    // The maximum rows per zone provided by user
    rows_per_zone: u64,
    use_seeds: bool,
    /// The file version this segment was loaded from. A `V0` segment's
    /// zones are never written into a `V1` file.
    version: ZoneMapIndexVersion,
    /// Whether this index is configured to keep exact null positions. False
    /// never regains a bitmap on update or rebuild; a legacy index without
    /// positions keeps `true` and regains one on a full retrain.
    store_null_bitmap: bool,
    store: Arc<dyn IndexStore>,
    /// Legacy synchronous remapper (index_version 0). Mutually exclusive with
    /// `batch_remapper`; both `None` means no translation is needed.
    fri: Option<Arc<dyn RowIdRemapper>>,
    /// Asynchronous batch remapper (tagged histories).
    batch_remapper: Option<Arc<dyn BatchRowIdRemapper>>,
    index_cache: WeakLanceCache,
    // Exact set of null row addresses across all zones; None when loaded from an
    // older index that did not persist this bitmap.
    null_rows: Option<RowAddrTreeMap>,
}

impl std::fmt::Debug for ZoneMapIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZoneMapIndex")
            .field("zones", &self.zones)
            .field("data_type", &self.data_type)
            .field("rows_per_zone", &self.rows_per_zone)
            .field("use_seeds", &self.use_seeds)
            .field("version", &self.version)
            .field("store_null_bitmap", &self.store_null_bitmap)
            .field("store", &self.store)
            .field("fri", &self.fri)
            .field("index_cache", &self.index_cache)
            .finish()
    }
}

impl DeepSizeOf for ZoneMapIndex {
    fn deep_size_of_children(&self, context: &mut lance_core::deepsize::Context) -> usize {
        self.zones.deep_size_of_children(context) + self.null_rows.deep_size_of_children(context)
    }
}

impl ZoneMapIndex {
    /// Returns the rows-per-zone parameter for this index.
    pub fn rows_per_zone(&self) -> u64 {
        self.rows_per_zone
    }

    /// The file version this segment was loaded from.
    pub fn version(&self) -> ZoneMapIndexVersion {
        self.version
    }

    /// Whether the index is configured to keep exact null positions.
    pub fn store_null_bitmap(&self) -> bool {
        self.store_null_bitmap
    }

    /// Whether this loaded index actually carries exact null positions.
    pub fn has_null_bitmap(&self) -> bool {
        self.null_rows.is_some()
    }

    /// Number of zones in this index.
    pub fn num_zones(&self) -> usize {
        self.zones.len()
    }

    /// The indexed column's value type.
    pub fn value_type(&self) -> &DataType {
        &self.data_type
    }

    /// Every zone's bound and statistics, in index order (for tests and
    /// diagnostics).
    #[doc(hidden)]
    pub fn zone_summaries(&self) -> Vec<ZoneSummary> {
        self.zones
            .iter()
            .map(|zone| ZoneSummary {
                bound: zone.bound.clone(),
                min: zone.min.clone(),
                max: zone.max.clone(),
                null_count: zone.null_count,
                nan_count: zone.nan_count,
            })
            .collect()
    }

    /// The builder parameters that reproduce this index's configuration.
    fn builder_params(&self) -> ZoneMapIndexBuilderParams {
        ZoneMapIndexBuilderParams {
            rows_per_zone: self.rows_per_zone,
            use_seeds: Some(self.use_seeds),
            store_null_bitmap: self.store_null_bitmap,
        }
    }

    fn scalar_is_nan(value: &ScalarValue) -> bool {
        match value {
            ScalarValue::Float16(Some(value)) => value.is_nan(),
            ScalarValue::Float32(Some(value)) => value.is_nan(),
            ScalarValue::Float64(Some(value)) => value.is_nan(),
            _ => false,
        }
    }

    fn scalar_is_negative_nan(value: &ScalarValue) -> bool {
        match value {
            ScalarValue::Float16(Some(value)) => value.is_nan() && value.is_sign_negative(),
            ScalarValue::Float32(Some(value)) => value.is_nan() && value.is_sign_negative(),
            ScalarValue::Float64(Some(value)) => value.is_nan() && value.is_sign_negative(),
            _ => false,
        }
    }

    /// Whether the zone may contain a positive NaN (sorts above positive
    /// infinity under Arrow Rust ordering). Conservative `true` when the NaN
    /// sign split is unknown.
    fn zone_positive_nan_possible(zone: &ZoneMapStatistics) -> bool {
        match zone.negative_nan_count {
            Some(negative) => zone.nan_count > negative,
            None => zone.nan_count > 0,
        }
    }

    /// Whether the zone may contain a negative NaN (sorts below negative
    /// infinity under Arrow Rust ordering). Conservative `true` when the NaN
    /// sign split is unknown.
    fn zone_negative_nan_possible(zone: &ZoneMapStatistics) -> bool {
        match zone.negative_nan_count {
            Some(negative) => negative > 0,
            None => zone.nan_count > 0,
        }
    }

    /// Whether the zone may contain any non-null value (finite or NaN).
    /// Recorded bounds win over counts: a remapped legacy zone inherits its
    /// source's counts, which are not exact for the shorter run.
    fn zone_non_null_possible(zone: &ZoneMapStatistics) -> bool {
        !Self::zone_has_missing_extrema(zone)
            || zone.bound.length as u64 > u64::from(zone.null_count)
    }

    /// Returns true if the zone has a non-null, non-NaN min value.
    fn zone_has_finite_min(zone: &ZoneMapStatistics) -> bool {
        !(zone.min.is_null() || Self::scalar_is_nan(&zone.min))
    }

    /// Returns true if both min and max are non-null / non-NaN.
    fn zone_has_finite_extrema(zone: &ZoneMapStatistics) -> bool {
        Self::zone_has_finite_min(zone) && !(zone.max.is_null() || Self::scalar_is_nan(&zone.max))
    }

    fn zone_has_missing_extrema(zone: &ZoneMapStatistics) -> bool {
        zone.min.is_null() || zone.max.is_null()
    }

    /// Counts prove whether missing extrema mean "no comparable values" or
    /// "bounds unknown". Only the latter must conservatively retain the zone.
    fn zone_has_comparable_values(zone: &ZoneMapStatistics) -> bool {
        zone.bound.length as u128 > u128::from(zone.null_count) + u128::from(zone.nan_count)
    }

    /// Global `[min, max]` folded across one or more ZoneMap segments (the
    /// disjoint per-column segments of a multi-segment index), without a scan.
    ///
    /// `None` when no zone has a finite bound, or when any zone's `max` is NaN:
    /// `ScalarValue`'s total order ranks NaN above every finite value, so a
    /// NaN-bearing zone hides its true finite max and no sound upper bound exists
    /// without a scan — folding only the finite maxes would yield a *subset* that
    /// prunes live rows. Folding raw zones (not each segment's `value_range`)
    /// keeps this exact across segments.
    ///
    /// Otherwise the range is a superset of the segments' live values,
    /// conservative under deletion vectors: safe to prune with, not guaranteed
    /// tight. The caller must ensure the segments jointly cover every live
    /// fragment.
    pub fn value_range_over<'a>(
        segments: impl IntoIterator<Item = &'a Self>,
    ) -> Option<(ScalarValue, ScalarValue)> {
        let mut min: Option<&ScalarValue> = None;
        let mut max: Option<&ScalarValue> = None;
        for seg in segments.into_iter() {
            // Nested types have no meaningful ordering
            if seg.data_type.is_nested() {
                return None;
            }
            for zone in seg.zones.iter() {
                // Legacy Decimal zones can contain comparable values even though their
                // extrema were never written, so skipping them would produce a subset.
                if Self::zone_has_missing_extrema(zone) && Self::zone_has_comparable_values(zone) {
                    return None;
                }
                // Any NaN breaks the fold: positive NaNs sort above every
                // finite value and negative NaNs below, so no finite global
                // range bounds a NaN-bearing zone. Legacy zones mark this with
                // a NaN max; fresh zones keep finite bounds plus the counts.
                if zone.nan_count > 0 || Self::scalar_is_nan(&zone.max) {
                    return None;
                }
                if Self::scalar_is_finite_bound(&zone.min)
                    && min.is_none_or(|cur| zone.min.partial_cmp(cur).is_some_and(|o| o.is_lt()))
                {
                    min = Some(&zone.min);
                }
                if Self::scalar_is_finite_bound(&zone.max)
                    && max.is_none_or(|cur| zone.max.partial_cmp(cur).is_some_and(|o| o.is_gt()))
                {
                    max = Some(&zone.max);
                }
            }
        }
        Some((min?.clone(), max?.clone()))
    }

    /// A scalar usable as a global-range bound: non-null and, for floats, non-NaN.
    fn scalar_is_finite_bound(v: &ScalarValue) -> bool {
        !v.is_null() && !Self::scalar_is_nan(v)
    }

    /// Evaluates whether a zone could potentially contain values matching the query.
    ///
    /// Matching follows Arrow Rust total ordering, where NaNs rank by sign and
    /// payload around the finite values. The counts know only how many NaNs of
    /// each sign a zone holds, never their payloads, so a zone is pruned only
    /// when the statistics prove no value can match; anything short of proof
    /// keeps the zone for the predicate recheck.
    fn evaluate_zone_against_query(
        &self,
        zone: &ZoneMapStatistics,
        query: &SargableQuery,
    ) -> Result<bool> {
        use std::ops::Bound;

        // For nested types we only track null_count; prune only when certain.
        if self.data_type.is_nested() {
            let all_null = zone.null_count as usize == zone.bound.length;
            return match query {
                SargableQuery::IsNull() => Ok(zone.null_count > 0),
                SargableQuery::Equals(target) => {
                    if target.is_null() {
                        Ok(zone.null_count > 0)
                    } else {
                        Ok(!all_null)
                    }
                }
                SargableQuery::IsIn(values) => {
                    if values.iter().any(|v| !v.is_null()) {
                        Ok(!all_null)
                    } else {
                        Ok(zone.null_count > 0)
                    }
                }
                _ => Ok(!all_null),
            };
        }

        match query {
            SargableQuery::IsNull() => {
                // Zone contains matching values if it has any null values
                Ok(zone.null_count > 0)
            }
            SargableQuery::Equals(target) => {
                // Zone contains matching values if target falls within [min, max] range
                // Handle null values - if target is null, check null_count
                if target.is_null() {
                    return Ok(zone.null_count > 0);
                }

                // Handle NaN values - if target is NaN, check nan_count
                let is_nan = match target {
                    ScalarValue::Float16(Some(f)) => f.is_nan(),
                    ScalarValue::Float32(Some(f)) => f.is_nan(),
                    ScalarValue::Float64(Some(f)) => f.is_nan(),
                    _ => false,
                };

                if is_nan {
                    return Ok(if Self::scalar_is_negative_nan(target) {
                        Self::zone_negative_nan_possible(zone)
                    } else {
                        Self::zone_positive_nan_possible(zone)
                    });
                }

                if Self::zone_has_missing_extrema(zone) {
                    return Ok(Self::zone_has_comparable_values(zone));
                }

                Ok(target >= &zone.min && target <= &zone.max)
            }
            SargableQuery::Range(start, end) => {
                // NaN range bounds resolve from the counts alone, so handle
                // them before the missing-extrema fallback: an all-NaN zone
                // has null bounds yet still answers `x <= -NaN` exactly.
                // A NaN on one side makes that side's answer independent of
                // the other bound (conservatively over-inclusive).
                //
                // Total ordering ranks NaNs of the same sign by payload, and
                // the counts say nothing about payloads. So a count can prove
                // "no NaN of this sign", never "no NaN of this sign on this
                // side of the bound NaN": whenever a same-sign NaN is
                // possible, the zone stays and the predicate recheck decides.
                match start {
                    Bound::Included(s) if Self::scalar_is_nan(s) => {
                        return Ok(if Self::scalar_is_negative_nan(s) {
                            // x >= -NaN: every finite value and positive NaN
                            // sorts above any -NaN; other -NaNs depend on
                            // payload, which the counts cannot rule out.
                            Self::zone_non_null_possible(zone)
                        } else {
                            // x >= +NaN: only positive NaNs can sort there.
                            Self::zone_positive_nan_possible(zone)
                        });
                    }
                    Bound::Excluded(s) if Self::scalar_is_nan(s) => {
                        return Ok(if Self::scalar_is_negative_nan(s) {
                            // x > -NaN: every finite value and positive NaN;
                            // other -NaNs depend on payload.
                            Self::zone_non_null_possible(zone)
                        } else {
                            // x > +NaN: a positive NaN with a larger payload
                            // still sorts above the bound, so a positive NaN
                            // count keeps the zone.
                            Self::zone_positive_nan_possible(zone)
                        });
                    }
                    _ => {}
                }
                match end {
                    Bound::Included(e) if Self::scalar_is_nan(e) => {
                        return Ok(if Self::scalar_is_negative_nan(e) {
                            // x <= -NaN: only negative NaNs can sort there.
                            Self::zone_negative_nan_possible(zone)
                        } else {
                            // x <= +NaN: every finite value and negative NaN;
                            // other +NaNs depend on payload.
                            Self::zone_non_null_possible(zone)
                        });
                    }
                    Bound::Excluded(e) if Self::scalar_is_nan(e) => {
                        return Ok(if Self::scalar_is_negative_nan(e) {
                            // x < -NaN: a negative NaN with a larger payload
                            // still sorts below the bound, so a negative NaN
                            // count keeps the zone.
                            Self::zone_negative_nan_possible(zone)
                        } else {
                            // x < +NaN: every finite value and negative NaN;
                            // other +NaNs depend on payload.
                            Self::zone_non_null_possible(zone)
                        });
                    }
                    _ => {}
                }

                // Both bounds are now finite or unbounded. Each value class is
                // judged against the WHOLE range, then the results are OR'd: a
                // positive NaN satisfying the lower bound and a negative NaN
                // satisfying the upper bound are different values, so neither
                // matches a two-sided finite range. Positive NaNs sort above
                // every finite value (they match only when the upper side is
                // unbounded), negative NaNs below (only when the lower side is
                // unbounded).
                //
                // Ordinary values are judged by the extrema; a non-finite
                // bound (legacy NaN marker) means unknown and keeps the zone.
                // Counts decide only when no bound was recorded: remapped
                // legacy zones inherit inexact counts.
                let ordinary_possible = if Self::zone_has_missing_extrema(zone) {
                    Self::zone_has_comparable_values(zone)
                } else {
                    let above_start = match start {
                        Bound::Unbounded => true,
                        Bound::Included(s) => {
                            !Self::scalar_is_finite_bound(&zone.max) || &zone.max >= s
                        }
                        Bound::Excluded(s) => {
                            !Self::scalar_is_finite_bound(&zone.max) || &zone.max > s
                        }
                    };
                    let below_end = match end {
                        Bound::Unbounded => true,
                        Bound::Included(e) => {
                            !Self::scalar_is_finite_bound(&zone.min) || &zone.min <= e
                        }
                        Bound::Excluded(e) => {
                            !Self::scalar_is_finite_bound(&zone.min) || &zone.min < e
                        }
                    };
                    above_start && below_end
                };
                let positive_nan_matches =
                    Self::zone_positive_nan_possible(zone) && matches!(end, Bound::Unbounded);
                let negative_nan_matches =
                    Self::zone_negative_nan_possible(zone) && matches!(start, Bound::Unbounded);

                Ok(ordinary_possible || positive_nan_matches || negative_nan_matches)
            }
            SargableQuery::IsIn(values) => {
                // Zone contains matching values if any value in the set falls within [min, max]
                Ok(values.iter().any(|value| {
                    if value.is_null() {
                        zone.null_count > 0
                    } else {
                        match value {
                            ScalarValue::Float16(Some(f)) => {
                                if f.is_nan() {
                                    if f.is_sign_negative() {
                                        Self::zone_negative_nan_possible(zone)
                                    } else {
                                        Self::zone_positive_nan_possible(zone)
                                    }
                                } else if Self::zone_has_missing_extrema(zone) {
                                    Self::zone_has_comparable_values(zone)
                                } else if !Self::zone_has_finite_min(zone) {
                                    false
                                } else {
                                    value >= &zone.min && value <= &zone.max
                                }
                            }
                            ScalarValue::Float32(Some(f)) => {
                                if f.is_nan() {
                                    if f.is_sign_negative() {
                                        Self::zone_negative_nan_possible(zone)
                                    } else {
                                        Self::zone_positive_nan_possible(zone)
                                    }
                                } else if Self::zone_has_missing_extrema(zone) {
                                    Self::zone_has_comparable_values(zone)
                                } else if !Self::zone_has_finite_min(zone) {
                                    false
                                } else {
                                    value >= &zone.min && value <= &zone.max
                                }
                            }
                            ScalarValue::Float64(Some(f)) => {
                                if f.is_nan() {
                                    if f.is_sign_negative() {
                                        Self::zone_negative_nan_possible(zone)
                                    } else {
                                        Self::zone_positive_nan_possible(zone)
                                    }
                                } else if Self::zone_has_missing_extrema(zone) {
                                    Self::zone_has_comparable_values(zone)
                                } else if !Self::zone_has_finite_min(zone) {
                                    false
                                } else {
                                    value >= &zone.min && value <= &zone.max
                                }
                            }
                            _ => {
                                if Self::zone_has_missing_extrema(zone) {
                                    Self::zone_has_comparable_values(zone)
                                } else {
                                    Self::zone_has_finite_extrema(zone)
                                        && value >= &zone.min
                                        && value <= &zone.max
                                }
                            }
                        }
                    }
                }))
            }
            SargableQuery::FullTextSearch(_) => Err(Error::not_supported_source(
                "full text search is not supported for zonemap indexes".into(),
            )),
            SargableQuery::LikePrefix(prefix) => {
                // For prefix matching, a zone can match if:
                // - zone.max >= prefix (there could be values >= prefix)
                // - zone.min < next_prefix (there could be values < next_prefix)
                //
                // For example, prefix "foo":
                // - Zone [aaa, azz]: max="azz" < "foo", so no match
                // - Zone [fa, foz]: min="fa" < "fop", max="foz" >= "foo", so potential match
                // - Zone [fop, fzz]: min="fop" >= "fop", so no match

                let prefix_str = match prefix {
                    ScalarValue::Utf8(Some(s)) => s.as_str(),
                    ScalarValue::LargeUtf8(Some(s)) => s.as_str(),
                    _ => return Ok(true), // Conservative: include zone if not a string prefix
                };

                // Empty prefix matches everything
                if prefix_str.is_empty() {
                    return Ok(true);
                }

                // Check zone.max >= prefix
                let max_check = &zone.max >= prefix;
                if !max_check {
                    return Ok(false);
                }

                // Compute next_prefix by incrementing the last byte
                // If the prefix ends with 0xFF bytes, we need to handle overflow
                let next_prefix = compute_next_prefix(prefix_str);

                match next_prefix {
                    Some(next) => {
                        // Check zone.min < next_prefix
                        let next_scalar = match prefix {
                            ScalarValue::Utf8(_) => ScalarValue::Utf8(Some(next)),
                            ScalarValue::LargeUtf8(_) => ScalarValue::LargeUtf8(Some(next)),
                            _ => return Ok(true),
                        };
                        Ok(zone.min < next_scalar)
                    }
                    None => {
                        // No upper bound (prefix is all 0xFF), so any zone with max >= prefix matches
                        Ok(true)
                    }
                }
            }
        }
    }

    /// Additive sibling of [`Self::load`] for mappings that require
    /// asynchronous batch row-ID translation.
    async fn load_with_remapping(
        store: Arc<dyn IndexStore>,
        remapping: Option<Arc<dyn BatchRowIdRemapper>>,
        index_cache: &LanceCache,
        use_seeds: bool,
        version: ZoneMapIndexVersion,
    ) -> Result<Arc<Self>> {
        lance_index_core::remapping::check_batch_remapping_entry()?;
        let index = Self::load(store, None, index_cache, use_seeds, version).await?;
        let mut index = Arc::into_inner(index)
            .ok_or_else(|| Error::internal("freshly loaded zone map index must be unshared"))?;
        index.batch_remapper = remapping;
        debug_assert!(index.fri.is_none() || index.batch_remapper.is_none());
        Ok(Arc::new(index))
    }

    /// Load the scalar index from storage, decoding the file under the
    /// contract of `version` (the segment's `IndexMetadata.index_version`).
    async fn load(
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
        use_seeds: bool,
        version: ZoneMapIndexVersion,
    ) -> Result<Arc<Self>>
    where
        Self: Sized,
    {
        let index_file = store.open_index_file(ZONEMAP_FILENAME).await?;
        let zone_maps = index_file
            .read_range(0..index_file.num_rows(), None)
            .await?;
        let file_schema = index_file.schema();

        let rows_per_zone: u64 = file_schema
            .metadata
            .get(ZONEMAP_SIZE_META_KEY)
            .and_then(|bs| bs.parse().ok())
            .unwrap_or(ROWS_PER_ZONE_DEFAULT);

        let null_rows = if let Some(idx_str) = file_schema.metadata.get(NULL_BITMAP_META_KEY) {
            let idx = idx_str.parse::<u32>().map_err(|e| {
                Error::invalid_input(format!("invalid null bitmap buffer index: {e}"))
            })?;
            let bytes = index_file.read_global_buffer(idx).await?;
            Some(RowAddrTreeMap::deserialize_from(bytes.as_ref())?)
        } else {
            None
        };
        let store_null_bitmap = file_schema
            .metadata
            .get(NULL_BITMAP_POLICY_META_KEY)
            .is_none_or(|value| value != "false");

        Ok(Arc::new(Self::try_from_serialized(
            zone_maps,
            store,
            fri,
            index_cache,
            rows_per_zone,
            null_rows,
            use_seeds,
            store_null_bitmap,
            version,
        )?))
    }

    #[allow(clippy::too_many_arguments)]
    fn try_from_serialized(
        data: RecordBatch,
        store: Arc<dyn IndexStore>,
        fri: Option<Arc<dyn RowIdRemapper>>,
        index_cache: &LanceCache,
        rows_per_zone: u64,
        null_rows: Option<RowAddrTreeMap>,
        use_seeds: bool,
        store_null_bitmap: bool,
        version: ZoneMapIndexVersion,
    ) -> Result<Self> {
        // The RecordBatch should have columns: min, max, null_count
        let min_col = data
            .column_by_name("min")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'min' column"))?;
        let max_col = data
            .column_by_name("max")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'max' column"))?;
        let null_count_col = data
            .column_by_name("null_count")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'null_count' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'null_count' column is not UInt32")
            })?;
        let nan_count_col = data
            .column_by_name("nan_count")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'nan_count' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'nan_count' column is not UInt32")
            })?;
        let zone_length = data
            .column_by_name("zone_length")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'zone_length' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'zone_length' column is not UInt64")
            })?;

        let fragment_id_col = data
            .column_by_name("fragment_id")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'fragment_id' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'fragment_id' column is not UInt64")
            })?;

        let zone_start_col = data
            .column_by_name("zone_start")
            .ok_or_else(|| Error::invalid_input("ZoneMapIndex: missing 'zone_start' column"))?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| {
                Error::invalid_input("ZoneMapIndex: 'zone_start' column is not UInt64")
            })?;

        // Only version 1 stores the NaN sign split.
        let negative_nan_count_col = match version {
            ZoneMapIndexVersion::V1 => Some(
                data.column_by_name("negative_nan_count")
                    .ok_or_else(|| {
                        Error::corrupt_file_named(
                            ZONEMAP_FILENAME,
                            "ZoneMapIndex: version 1 file has no 'negative_nan_count' column",
                        )
                    })?
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| {
                        Error::corrupt_file_named(
                            ZONEMAP_FILENAME,
                            "ZoneMapIndex: 'negative_nan_count' column is not UInt32",
                        )
                    })?,
            ),
            ZoneMapIndexVersion::V0 => None,
        };

        let data_type = min_col.data_type().clone();

        if data.num_rows() == 0 {
            return Ok(Self {
                zones: Vec::new(),
                data_type,
                rows_per_zone,
                use_seeds,
                version,
                store_null_bitmap,
                store,
                fri,
                batch_remapper: None,
                index_cache: WeakLanceCache::from(index_cache),
                null_rows,
            });
        }

        let num_zones = data.num_rows();
        let mut zones = Vec::with_capacity(num_zones);

        for i in 0..num_zones {
            let (min, max) = if data_type.is_nested() {
                (
                    ScalarValue::try_new_null(&data_type)?,
                    ScalarValue::try_new_null(&data_type)?,
                )
            } else {
                (
                    ScalarValue::try_from_array(min_col, i)?,
                    ScalarValue::try_from_array(max_col, i)?,
                )
            };
            let null_count = null_count_col.value(i);
            let nan_count = nan_count_col.value(i);
            let length = zone_length.value(i);
            let negative_nan_count = match negative_nan_count_col {
                Some(col) => {
                    let negative = col.value(i);
                    if let Some(violation) = v1_zone_violation(
                        &min,
                        &max,
                        null_count,
                        nan_count,
                        negative,
                        length,
                        data_type.is_nested(),
                    ) {
                        return Err(Error::corrupt_file_named(
                            ZONEMAP_FILENAME,
                            format!("ZoneMapIndex: version 1 zone {i} has {violation}"),
                        ));
                    }
                    Some(negative)
                }
                // Version 0 stores no split: unknown, unless the zone has no
                // NaNs at all.
                None if nan_count == 0 => Some(0),
                None => None,
            };
            zones.push(ZoneMapStatistics {
                min,
                max,
                null_count,
                nan_count,
                negative_nan_count,
                bound: ZoneBound {
                    fragment_id: fragment_id_col.value(i),
                    start: zone_start_col.value(i),
                    length: length as usize,
                },
            });
        }

        Ok(Self {
            zones,
            data_type,
            rows_per_zone,
            use_seeds,
            version,
            store_null_bitmap,
            store,
            fri,
            batch_remapper: None,
            index_cache: WeakLanceCache::from(index_cache),
            null_rows,
        })
    }
}

#[async_trait]
impl Index for ZoneMapIndex {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_index(self: Arc<Self>) -> Arc<dyn Index> {
        self
    }

    async fn prewarm(&self) -> Result<()> {
        // Not much to prewarm
        Ok(())
    }

    fn statistics(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "num_zones": self.zones.len(),
            "rows_per_zone": self.rows_per_zone,
        }))
    }

    fn index_type(&self) -> IndexType {
        IndexType::ZoneMap
    }

    async fn calculate_included_frags(&self) -> Result<RoaringBitmap> {
        let mut frag_ids = RoaringBitmap::new();

        // Loop through zones and add unique fragment IDs to the bitmap
        for zone in &self.zones {
            frag_ids.insert(zone.bound.fragment_id as u32);
        }

        Ok(frag_ids)
    }
}

#[async_trait]
impl ScalarIndex for ZoneMapIndex {
    async fn search(
        &self,
        query: &dyn AnyQuery,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        let query = query.as_any().downcast_ref::<SargableQuery>().unwrap();
        let result = if let SargableQuery::IsNull() = query
            && let Some(null_rows) = &self.null_rows
        {
            SearchResult::exact(null_rows.clone())
        } else {
            search_zones(&self.zones, metrics, |zone| {
                self.evaluate_zone_against_query(zone, query)
            })?
        };

        let (selected, nulls) = if let Some(remapper) = &self.fri {
            // Legacy synchronous remapping path.
            (
                remapper.remap_row_addrs_tree_map(result.row_addrs().selected_rows()),
                remapper.remap_row_addrs_tree_map(result.row_addrs().null_rows()),
            )
        } else if let Some(remapper) = &self.batch_remapper {
            // Tagged asynchronous path.
            (
                remap_row_addrs_tree_map_async(
                    remapper.as_ref(),
                    result.row_addrs().selected_rows(),
                )
                .await?,
                remap_row_addrs_tree_map_async(remapper.as_ref(), result.row_addrs().null_rows())
                    .await?,
            )
        } else {
            return Ok(result);
        };

        Ok(match result {
            SearchResult::Exact(_) => SearchResult::exact(selected).with_nulls(nulls),
            SearchResult::AtMost(_) => SearchResult::at_most(selected).with_nulls(nulls),
            SearchResult::AtLeast(_) => SearchResult::at_least(selected).with_nulls(nulls),
        })
    }

    fn results_are_row_addresses(&self) -> bool {
        true
    }

    fn can_remap(&self) -> bool {
        false
    }

    /// Remap the row ids, creating a new remapped version of this index in `dest_store`
    async fn remap(
        &self,
        _mapping: &RowAddrRemap,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        Err(Error::invalid_input_source(
            "ZoneMapIndex does not support remap".into(),
        ))
    }

    async fn remap_streaming(
        &self,
        _translator: &RowAddrTranslator,
        _dest_store: &dyn IndexStore,
    ) -> Result<CreatedIndex> {
        // No mapping to materialize for an index that cannot be remapped.
        Err(Error::invalid_input_source(
            "ZoneMapIndex does not support remap".into(),
        ))
    }

    /// Add the new data , creating an updated version of the index in `dest_store`
    async fn update(
        &self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
        _old_data_filter: Option<super::OldIndexDataFilter>,
    ) -> Result<CreatedIndex> {
        // Train new zones for the incoming data stream
        let schema = new_data.schema();
        let value_type = schema.field(0).data_type().clone();

        let options = self.builder_params();
        let processor = ZoneMapProcessor::new(value_type.clone())?;
        // Positions are only worth collecting when the index keeps them.
        let trainer = ZoneTrainer::new(processor, self.rows_per_zone)?
            .with_null_tracking(self.null_rows.is_some());
        let (new_zones, new_null_rows) = trainer.train(new_data).await?;

        // Additive: the stream is "the new data", with no claim about the
        // rest of its fragments, so every existing zone stays. Superseding a
        // re-summarized fragment is the dataset builder's job
        // (`build_zonemap_segment`), which knows the refreshed fragment set
        // and the physical row counts.
        let mut updated_zones = self.zones.clone();
        updated_zones.extend(new_zones);
        updated_zones.sort_by_key(|z| (z.bound.fragment_id, z.bound.start));

        // Merge existing and new null rows.  If the existing index had no null bitmap
        // (legacy format — null positions unknown), preserve that None: updating cannot
        // recover the missing information, and claiming the result has zero nulls would
        // be a false negative.  Only a full retrain produces a fresh, complete bitmap.
        let merged_null_rows = self.null_rows.as_ref().map(|existing| {
            let mut merged = existing.clone();
            merged |= &new_null_rows;
            merged
        });

        // This entry has no source data to recover version 1 statistics
        // from, so the file keeps the segment's version.
        let mut builder = ZoneMapIndexBuilder::try_new(options, self.data_type.clone())?
            .with_version(self.version);
        builder.options.rows_per_zone = self.rows_per_zone;
        builder.maps = updated_zones;
        builder.set_null_rows(merged_null_rows);
        let has_null_bitmap = builder.null_rows.is_some();
        let files = builder.write_index(dest_store).await?;

        Ok(CreatedIndex {
            index_details: make_zone_map_index_details(
                self.rows_per_zone,
                self.use_seeds,
                has_null_bitmap,
            ),
            index_version: self.version as u32,
            files,
        })
    }

    fn update_criteria(&self) -> UpdateCriteria {
        UpdateCriteria::only_new_data(
            TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        )
    }

    fn derive_index_params(&self) -> Result<ScalarIndexParams> {
        let params = serde_json::to_value(self.builder_params())?;
        Ok(ScalarIndexParams::for_builtin(BuiltinIndexType::ZoneMap).with_params(&params))
    }

    /// Single-segment `[min, max]` folded from this index's zones; see
    /// [`value_range_over`](Self::value_range_over) for the full contract.
    fn value_range(&self) -> Option<(ScalarValue, ScalarValue)> {
        // We don't record min/max for nested types
        if self.data_type.is_nested() {
            return None;
        }
        Self::value_range_over([self])
    }
}

impl ZoneMapIndex {
    /// Append `seeds` through the same validated reuse core as the dataset
    /// path, without a scan: every record of every seed must lie inside a
    /// target zone of this index, otherwise `None` (the caller scans). The
    /// file's physical row count is unknown here, so the seed's own coverage
    /// stands in for it.
    async fn try_update_with_seeds(
        &self,
        seeds: &[crate::scalar::seed::FragmentSeed],
        dest_store: &dyn IndexStore,
    ) -> Result<Option<CreatedIndex>> {
        // Version 0 zones cannot be copied into the version 1 output; the
        // caller rebuilds the segment from source data.
        if self.version != ZoneMapIndexVersion::V1 {
            return Ok(None);
        }
        let need_positions = self.null_rows.is_some();
        let mut reused = ReusedZones::new(need_positions);
        for seed in seeds {
            let harvested = HarvestedZoneSeed {
                seed: seed.clone(),
                physical_rows: seed.physical_rows,
            };
            if reused.reuse(&harvested, self.rows_per_zone, &self.data_type, false)?
                != SeedOutcome::Complete
            {
                return Ok(None);
            }
        }
        let refreshed: Vec<u64> = seeds.iter().map(|seed| seed.fragment_id).collect();
        build_zonemap_segment(
            ZoneMapSegmentBase::Update(self),
            &self.data_type,
            reused,
            None,
            &HashMap::new(),
            &refreshed,
            dest_store,
        )
        .await
        .map(Some)
    }
}

/// The destination runs of `zone` under `remapper`. With `fragment_filter`,
/// rows that land outside those fragments are dropped before the runs are
/// built, so a run the caller would not keep can neither fail here nor be
/// checked later.
fn remap_zone(
    zone: &ZoneMapStatistics,
    remapper: &dyn RowIdRemapper,
    remapped_null_rows: Option<&RowAddrTreeMap>,
    is_nested: bool,
    fragment_filter: Option<&RoaringBitmap>,
) -> Result<Vec<ZoneMapStatistics>> {
    let zone_start = (zone.bound.fragment_id << 32).saturating_add(zone.bound.start);
    let mut remapped = (0..zone.bound.length as u64)
        .filter_map(|offset| remapper.remap_row_id(zone_start.saturating_add(offset)))
        .filter(|row_id| {
            fragment_filter.is_none_or(|filter| {
                u32::try_from(row_id >> 32).is_ok_and(|fragment| filter.contains(fragment))
            })
        })
        .collect::<Vec<_>>();
    remapped.sort_unstable();
    remapped.dedup();

    let make_zone = |start: u64, end: u64| -> Result<ZoneMapStatistics> {
        let length = (end - start + 1) as usize;
        let null_count = if let Some(null_rows) = remapped_null_rows {
            u32::try_from(
                (start..=end)
                    .filter(|row_id| null_rows.contains(*row_id))
                    .count(),
            )
            .map_err(|_| {
                Error::invalid_input(format!(
                    "remapped ZoneMap zone has more null rows than can be represented: \
                     fragment_id={}, start={}, length={}",
                    start >> 32,
                    start & u64::from(u32::MAX),
                    length
                ))
            })?
        } else if length == zone.bound.length {
            zone.null_count
        } else if zone.null_count == 0 {
            0
        } else if zone.null_count as usize == zone.bound.length {
            u32::try_from(length).map_err(|_| {
                Error::invalid_input(format!(
                    "remapped all-null ZoneMap zone length cannot be represented: \
                     fragment_id={}, start={}, length={}",
                    start >> 32,
                    start & u64::from(u32::MAX),
                    length
                ))
            })?
        } else if length > 1 || (!is_nested && !ZoneMapIndex::zone_has_missing_extrema(zone)) {
            // Without exact null positions, one null conservatively preserves both
            // null and non-null candidates when the run has multiple rows. A scalar
            // singleton is also safe when its extrema independently prove that it
            // may contain a comparable value. Otherwise null_count == length would
            // fabricate an all-null zone and could prune live non-null rows.
            1
        } else {
            return Err(Error::not_supported(format!(
                "cannot safely remap a mixed-null ZoneMap zone without an exact null bitmap: \
                 fragment_id={}, start={}, original_length={}, null_count={}, remapped_length={}, \
                 nested={}, missing_extrema={}",
                zone.bound.fragment_id,
                zone.bound.start,
                zone.bound.length,
                zone.null_count,
                length,
                is_nested,
                ZoneMapIndex::zone_has_missing_extrema(zone)
            )));
        };

        Ok(ZoneMapStatistics {
            min: zone.min.clone(),
            max: zone.max.clone(),
            null_count,
            nan_count: zone.nan_count,
            negative_nan_count: zone.negative_nan_count,
            bound: ZoneBound {
                fragment_id: start >> 32,
                start: start & u64::from(u32::MAX),
                length,
            },
        })
    };
    let mut zones = Vec::new();
    let mut run_start = None;
    let mut previous = 0u64;
    for row_id in remapped {
        if run_start.is_none() {
            run_start = Some(row_id);
        } else if row_id != previous.saturating_add(1) || row_id >> 32 != previous >> 32 {
            if let Some(start) = run_start {
                zones.push(make_zone(start, previous)?);
            }
            run_start = Some(row_id);
        }
        previous = row_id;
    }
    if let Some(start) = run_start {
        zones.push(make_zone(start, previous)?);
    }
    Ok(zones)
}

/// Merge caller-selected ZoneMap segments into one self-contained segment.
pub async fn merge_zonemap_indices(
    source_indices: &[&ZoneMapIndex],
    dest_store: &dyn IndexStore,
    fragment_filter: &RoaringBitmap,
) -> Result<CreatedIndex> {
    let first = source_indices.first().ok_or_else(|| {
        Error::invalid_input("merge_zonemap_indices requires at least one source index")
    })?;
    let rows_per_zone = first.rows_per_zone;
    let use_seeds = first.use_seeds;
    let store_null_bitmap = first.store_null_bitmap;
    let data_type = first.data_type.clone();
    // The output keeps the sources' version; this entry has no source data
    // to upgrade a version 0 segment with, so mixed inputs are refused
    // before anything is written.
    let version = first.version;
    if let Some(other) = source_indices
        .iter()
        .find(|source| source.version != version)
    {
        return Err(Error::not_supported_source(
            format!(
                "cannot merge ZoneMap segments of different file versions ({} and {}); \
                 rebuild them from their data (optimize_indices) instead",
                version as u32, other.version as u32
            )
            .into(),
        ));
    }

    let mut zones = Vec::new();
    let mut merged_null_rows = RowAddrTreeMap::new();
    let mut any_missing_bitmap = false;
    for source in source_indices {
        if source.batch_remapper.is_some() {
            return Err(Error::not_supported(
                "this index maintenance operation does not support asynchronous row-ID remapping",
            ));
        }
        if source.rows_per_zone != rows_per_zone {
            return Err(Error::invalid_input(format!(
                "cannot merge ZoneMap segments with different rows_per_zone values: {} and {}",
                rows_per_zone, source.rows_per_zone
            )));
        }
        if source.data_type != data_type {
            return Err(Error::invalid_input(format!(
                "cannot merge ZoneMap segments with different value types: {:?} and {:?}",
                data_type, source.data_type
            )));
        }
        if source.store_null_bitmap != store_null_bitmap {
            return Err(Error::invalid_input(format!(
                "cannot merge ZoneMap segments with different null bitmap settings: {} and {}",
                store_null_bitmap, source.store_null_bitmap
            )));
        }
        let remapped_null_rows = source.null_rows.as_ref().map(|null_rows| {
            source.fri.as_deref().map_or_else(
                || null_rows.clone(),
                |remapper| remapper.remap_row_addrs_tree_map(null_rows),
            )
        });
        for zone in &source.zones {
            let source_zones = source.fri.as_deref().map_or_else(
                || Ok(vec![zone.clone()]),
                |remapper| {
                    remap_zone(
                        zone,
                        remapper,
                        remapped_null_rows.as_ref(),
                        source.data_type.is_nested(),
                        Some(fragment_filter),
                    )
                },
            )?;
            // Only the runs that are kept are held to the version 1 contract.
            let retained = source_zones.into_iter().filter(|zone| {
                u32::try_from(zone.bound.fragment_id)
                    .is_ok_and(|fragment_id| fragment_filter.contains(fragment_id))
            });
            for remapped in retained {
                if version == ZoneMapIndexVersion::V1 {
                    reject_inexact_remapped_zone(zone, &remapped, remapped_null_rows.is_some())?;
                }
                zones.push(remapped);
            }
        }
        match remapped_null_rows {
            Some(mut filtered) => {
                filtered.retain_fragments(fragment_filter.iter());
                merged_null_rows |= &filtered;
            }
            None => any_missing_bitmap = true,
        }
    }
    zones.sort_by_key(|zone| (zone.bound.fragment_id, zone.bound.start));

    let mut builder =
        ZoneMapIndexBuilder::try_new(first.builder_params(), data_type)?.with_version(version);
    builder.maps = zones;
    builder.set_null_rows((!any_missing_bitmap).then_some(merged_null_rows));
    let has_null_bitmap = builder.null_rows.is_some();
    let files = builder.write_index(dest_store).await?;

    Ok(CreatedIndex {
        index_details: make_zone_map_index_details(rows_per_zone, use_seeds, has_null_bitmap),
        index_version: version as u32,
        files,
    })
}

/// `remap_zone` copies a zone's counts into each destination run, which is
/// exact only when the run keeps every row of the zone. A version 1 file
/// must hold exact counts, so a narrower run is accepted only when the zone
/// has no NaNs and its null count is exact: recomputed from a bitmap, zero,
/// or a zone whose every row is null, since every surviving row of it is a
/// null too (`remap_zone` then counts the run's own length). Otherwise the
/// merge is refused before writing.
fn reject_inexact_remapped_zone(
    source: &ZoneMapStatistics,
    remapped: &ZoneMapStatistics,
    null_positions_exact: bool,
) -> Result<()> {
    if remapped.bound.length == source.bound.length {
        return Ok(());
    }
    let all_null = source.null_count as usize == source.bound.length;
    let nulls_exact = null_positions_exact || source.null_count == 0 || all_null;
    if source.nan_count == 0 && nulls_exact {
        return Ok(());
    }
    Err(Error::not_supported_source(
        format!(
            "cannot merge a remapped ZoneMap zone into a version 1 segment: the run \
             (fragment {}, start {}, length {}) keeps only part of its source zone \
             (fragment {}, start {}, length {}, null_count {}, nan_count {}), so the \
             inherited counts are no longer exact; rebuild it from its data \
             (optimize_indices) instead",
            remapped.bound.fragment_id,
            remapped.bound.start,
            remapped.bound.length,
            source.bound.fragment_id,
            source.bound.start,
            source.bound.length,
            source.null_count,
            source.nan_count
        )
        .into(),
    ))
}

fn default_rows_per_zone() -> u64 {
    *DEFAULT_ROWS_PER_ZONE
}

fn default_store_null_bitmap() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneMapIndexBuilderParams {
    #[serde(default = "default_rows_per_zone")]
    rows_per_zone: u64,
    /// Whether to embed per-fragment seed buffers in data files for use during
    /// incremental index updates. `None` means auto-detect based on column type
    /// (see [`default_use_seeds`]). Resolved to a concrete `bool` during
    /// training in [`ZoneMapIndexPlugin::new_training_request`].
    #[serde(default)]
    use_seeds: Option<bool>,
    /// Whether the index keeps the exact row address of every null, which
    /// makes `IS NULL` and `IS NOT NULL` exact. Without it the index keeps
    /// only per-zone null counts and those predicates are rechecked.
    #[serde(default = "default_store_null_bitmap")]
    store_null_bitmap: bool,
}

static DEFAULT_ROWS_PER_ZONE: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_ZONEMAP_DEFAULT_ROWS_PER_ZONE")
        .unwrap_or_else(|_| (ROWS_PER_ZONE_DEFAULT).to_string())
        .parse()
        .expect("failed to parse LANCE_ZONEMAP_DEFAULT_ROWS_PER_ZONE")
});

impl Default for ZoneMapIndexBuilderParams {
    fn default() -> Self {
        Self {
            rows_per_zone: *DEFAULT_ROWS_PER_ZONE,
            use_seeds: None,
            store_null_bitmap: true,
        }
    }
}

impl ZoneMapIndexBuilderParams {
    pub fn new(rows_per_zone: u64) -> Self {
        Self {
            rows_per_zone,
            use_seeds: None,
            store_null_bitmap: true,
        }
    }

    pub fn rows_per_zone(&self) -> u64 {
        self.rows_per_zone
    }

    pub fn with_null_bitmap(mut self, store_null_bitmap: bool) -> Self {
        self.store_null_bitmap = store_null_bitmap;
        self
    }

    pub fn store_null_bitmap(&self) -> bool {
        self.store_null_bitmap
    }

    /// Whether new data files should carry write seeds for this index, with
    /// `None` resolved from the column type (see `default_use_seeds`).
    pub fn use_seeds_for(&self, data_type: &DataType) -> bool {
        self.use_seeds
            .unwrap_or_else(|| default_use_seeds(data_type))
    }
}

// A builder for zonemap index
pub struct ZoneMapIndexBuilder {
    options: ZoneMapIndexBuilderParams,
    /// The file version to write; see [`Self::with_version`].
    version: ZoneMapIndexVersion,

    items_type: DataType,
    maps: Vec<ZoneMapStatistics>,
    // None means "legacy index — null positions unknown"; Some means a complete bitmap.
    // write_index omits the null-bitmap global buffer when this is None, preserving the
    // legacy format so that downstream searches remain conservative.
    null_rows: Option<RowAddrTreeMap>,
}

impl ZoneMapIndexBuilder {
    pub fn try_new(options: ZoneMapIndexBuilderParams, items_type: DataType) -> Result<Self> {
        // Checked here, before any segment is written: a build that scans
        // nothing (no fragments, or every zone from seeds) never reaches the
        // trainer's own check.
        if options.rows_per_zone == 0 {
            return Err(Error::invalid_input(
                "rows_per_zone must be greater than zero",
            ));
        }
        Ok(Self {
            options,
            version: ZoneMapIndexVersion::CURRENT,
            items_type,
            maps: Vec::new(),
            null_rows: None,
        })
    }

    /// Write the file in `version` instead of the current one. Only the
    /// low-level update of an existing segment uses this, to keep that
    /// segment's version.
    #[doc(hidden)]
    pub fn with_version(mut self, version: ZoneMapIndexVersion) -> Self {
        self.version = version;
        self
    }

    /// Train the builder using the shared zone trainer.  The input stream must contain
    /// the value column followed by `_rowaddr`, matching the dataset scan order enforced
    /// by the scalar index registry.
    pub async fn train(&mut self, batches_source: SendableRecordBatchStream) -> Result<()> {
        let processor = ZoneMapProcessor::new(self.items_type.clone())?;
        let trainer = ZoneTrainer::new(processor, self.options.rows_per_zone)?
            .with_null_tracking(self.options.store_null_bitmap);
        let (maps, null_rows) = trainer.train(batches_source).await?;
        self.maps = maps;
        self.set_null_rows(Some(null_rows));
        Ok(())
    }

    /// Record exact null positions, unless the index is configured without
    /// a bitmap, in which case they are dropped here so no path can write one.
    fn set_null_rows(&mut self, null_rows: Option<RowAddrTreeMap>) {
        self.null_rows = if self.options.store_null_bitmap {
            null_rows
        } else {
            None
        };
    }

    fn zonemap_stats_as_batch(&self) -> Result<RecordBatch> {
        // Flush self.maps as a RecordBatch
        let (mins, maxs) = if self.maps.is_empty() {
            let bounds = new_empty_array(&self.items_type);
            (bounds.clone(), bounds)
        } else if matches!(self.items_type, DataType::Struct(_)) {
            // Scalar index files use v2.0, which cannot encode struct validity. Nested
            // bounds are never consulted, so persist a valid placeholder and restore the
            // typed-null sentinel when loading the index.
            let placeholder = ScalarValue::new_default(&self.items_type)?;
            let bounds =
                ScalarValue::iter_to_array(std::iter::repeat_n(placeholder, self.maps.len()))?;
            (bounds.clone(), bounds)
        } else if self.version == ZoneMapIndexVersion::V0 {
            // Version 0 readers take a NaN max as the marker that the zone
            // holds NaNs.
            let legacy_max = |stat: &ZoneMapStatistics| {
                if stat.nan_count > 0
                    && let Some(nan) = nan_scalar(&self.items_type)
                {
                    nan
                } else {
                    stat.max.clone()
                }
            };
            (
                ScalarValue::iter_to_array(self.maps.iter().map(|stat| stat.min.clone()))?,
                ScalarValue::iter_to_array(self.maps.iter().map(legacy_max))?,
            )
        } else {
            // Version 1 bounds are ordinary: NaNs are counted, not marked.
            (
                ScalarValue::iter_to_array(self.maps.iter().map(|stat| stat.min.clone()))?,
                ScalarValue::iter_to_array(self.maps.iter().map(|stat| stat.max.clone()))?,
            )
        };
        let null_counts =
            UInt32Array::from_iter_values(self.maps.iter().map(|stat| stat.null_count));

        let nan_counts = UInt32Array::from_iter_values(self.maps.iter().map(|stat| stat.nan_count));

        // A version 1 zone must know its sign split and satisfy the loader's
        // invariants; a bad zone fails the write instead of producing a file
        // its own reader refuses.
        let negative_nan_counts = match self.version {
            ZoneMapIndexVersion::V1 => Some(UInt32Array::from_iter_values(
                self.maps
                    .iter()
                    .map(|stat| {
                        let negative = stat.negative_nan_count.ok_or_else(|| {
                            Error::invalid_input(format!(
                                "cannot write a version 1 ZoneMap zone with an unknown NaN sign \
                                 split (fragment {}, start {})",
                                stat.bound.fragment_id, stat.bound.start
                            ))
                        })?;
                        if let Some(violation) = v1_zone_violation(
                            &stat.min,
                            &stat.max,
                            stat.null_count,
                            stat.nan_count,
                            negative,
                            stat.bound.length as u64,
                            self.items_type.is_nested(),
                        ) {
                            return Err(Error::invalid_input(format!(
                                "cannot write a version 1 ZoneMap zone with {violation} \
                                 (fragment {}, start {})",
                                stat.bound.fragment_id, stat.bound.start
                            )));
                        }
                        Ok(negative)
                    })
                    .collect::<Result<Vec<_>>>()?,
            )),
            ZoneMapIndexVersion::V0 => None,
        };

        let fragment_ids =
            UInt64Array::from_iter_values(self.maps.iter().map(|stat| stat.bound.fragment_id));

        let zone_lengths =
            UInt64Array::from_iter_values(self.maps.iter().map(|stat| stat.bound.length as u64));

        let zone_starts =
            UInt64Array::from_iter_values(self.maps.iter().map(|stat| stat.bound.start));

        let mut fields = vec![
            // min and max can be null if the entire batch is null values
            Field::new("min", self.items_type.clone(), true),
            Field::new("max", self.items_type.clone(), true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("nan_count", DataType::UInt32, false),
        ];
        let mut columns: Vec<ArrayRef> = vec![
            mins,
            maxs,
            Arc::new(null_counts) as ArrayRef,
            Arc::new(nan_counts) as ArrayRef,
        ];
        if let Some(negative_nan_counts) = negative_nan_counts {
            fields.push(Field::new("negative_nan_count", DataType::UInt32, false));
            columns.push(Arc::new(negative_nan_counts) as ArrayRef);
        }
        fields.extend([
            Field::new("fragment_id", DataType::UInt64, false),
            Field::new("zone_start", DataType::UInt64, false),
            Field::new("zone_length", DataType::UInt64, false),
        ]);
        columns.extend([
            Arc::new(fragment_ids) as ArrayRef,
            Arc::new(zone_starts) as ArrayRef,
            Arc::new(zone_lengths) as ArrayRef,
        ]);
        let schema = Arc::new(arrow_schema::Schema::new(fields));
        Ok(RecordBatch::try_new(schema, columns)?)
    }

    pub async fn write_index(self, index_store: &dyn IndexStore) -> Result<Vec<IndexFile>> {
        let record_batch = self.zonemap_stats_as_batch()?;

        let mut file_schema = record_batch.schema().as_ref().clone();
        file_schema.metadata.insert(
            ZONEMAP_SIZE_META_KEY.to_string(),
            self.options.rows_per_zone.to_string(),
        );
        if !self.options.store_null_bitmap {
            file_schema
                .metadata
                .insert(NULL_BITMAP_POLICY_META_KEY.to_string(), "false".to_string());
        }

        let mut index_file = index_store
            .new_index_file(ZONEMAP_FILENAME, Arc::new(file_schema))
            .await?;
        index_file.write_record_batch(record_batch).await?;

        let zonemap_file = if let Some(null_rows) = self.null_rows {
            let mut null_bitmap_bytes = Vec::with_capacity(null_rows.serialized_size());
            null_rows.serialize_into(&mut null_bitmap_bytes)?;
            let null_bitmap_idx = index_file
                .add_global_buffer(bytes::Bytes::from(null_bitmap_bytes))
                .await?;
            index_file
                .finish_with_metadata(HashMap::from([(
                    NULL_BITMAP_META_KEY.to_string(),
                    null_bitmap_idx.to_string(),
                )]))
                .await?
        } else {
            index_file.finish_with_metadata(HashMap::new()).await?
        };

        Ok(vec![zonemap_file])
    }
}

/// Per-zone accumulator behind [`ZoneMapProcessor`].
#[derive(Debug)]
enum ZoneStatsAccumulator {
    /// min, max, null_count and nan_count for scalar types.
    Full(StatisticsAccumulator),
    /// Parent-level null count only. Nested bounds are never consulted, so the
    /// child values are not read at all.
    NullOnly { null_count: u64 },
}

/// Index-specific processor that computes zone statistics while the trainer
/// handles chunking and fragment boundaries.
///
/// For non-nested types, tracks min, max, null_count, and nan_count.
/// For nested types (List, FixedSizeList, Struct, Map, etc.), tracks only the
/// parent-level null_count without inspecting child values; min and max are
/// stored as typed null values.
#[derive(Debug)]
struct ZoneMapProcessor {
    data_type: DataType,
    statistics: ZoneStatsAccumulator,
}

impl ZoneMapProcessor {
    fn new(data_type: DataType) -> Result<Self> {
        let statistics = if data_type.is_nested() {
            ZoneStatsAccumulator::NullOnly { null_count: 0 }
        } else {
            ZoneStatsAccumulator::Full(StatisticsAccumulator::new(&data_type))
        };
        Ok(Self {
            data_type,
            statistics,
        })
    }

    fn scalar_value_from_stat(
        value: Option<&ArrayRef>,
        data_type: &DataType,
    ) -> Result<ScalarValue> {
        let array = value
            .cloned()
            .unwrap_or_else(|| new_null_array(data_type, 1));
        Ok(ScalarValue::try_from_array(&array, 0)?)
    }

    fn stat_count_to_u32(name: &str, value: u64) -> Result<u32> {
        u32::try_from(value).map_err(|_| {
            Error::invalid_input(format!(
                "{} value {} exceeds the supported UInt32 range",
                name, value
            ))
        })
    }
}

/// The positive-NaN scalar for a float type; `None` for everything else.
fn nan_scalar(data_type: &DataType) -> Option<ScalarValue> {
    match data_type {
        DataType::Float16 => Some(ScalarValue::Float16(Some(half::f16::NAN))),
        DataType::Float32 => Some(ScalarValue::Float32(Some(f32::NAN))),
        DataType::Float64 => Some(ScalarValue::Float64(Some(f64::NAN))),
        _ => None,
    }
}

impl ZoneProcessor for ZoneMapProcessor {
    type ZoneStatistics = ZoneMapStatistics;

    fn process_chunk(&mut self, array: &ArrayRef) -> Result<()> {
        match &mut self.statistics {
            ZoneStatsAccumulator::Full(statistics) => statistics.update(array)?,
            ZoneStatsAccumulator::NullOnly { null_count } => {
                if array.data_type() != &self.data_type {
                    return Err(Error::invalid_input(format!(
                        "Type mismatch: expected {:?}, got {:?}",
                        self.data_type,
                        array.data_type()
                    )));
                }
                *null_count += array.null_count() as u64;
            }
        }
        Ok(())
    }

    fn finish_zone(&mut self, bound: ZoneBound) -> Result<Self::ZoneStatistics> {
        let statistics = match &self.statistics {
            ZoneStatsAccumulator::Full(statistics) => statistics.statistics(),
            ZoneStatsAccumulator::NullOnly { null_count } => {
                return Ok(ZoneMapStatistics {
                    min: ScalarValue::try_new_null(&self.data_type)?,
                    max: ScalarValue::try_new_null(&self.data_type)?,
                    null_count: Self::stat_count_to_u32("null_count", *null_count)?,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound,
                });
            }
        };
        let null_count = Self::stat_count_to_u32("null_count", statistics.null_count)?;

        let nan_count = Self::stat_count_to_u32("nan_count", statistics.nan_count.unwrap_or(0))?;
        let negative_nan_count = Self::stat_count_to_u32(
            "negative_nan_count",
            statistics.negative_nan_count.unwrap_or(0),
        )?;
        // Bounds exclude NaNs of either sign (the accumulator skips them);
        // NaN presence lives in the exact counts. The legacy "NaN max" marker
        // is applied only when serializing the ZoneMap INDEX file, whose
        // format predates the sign split.
        Ok(ZoneMapStatistics {
            min: Self::scalar_value_from_stat(
                statistics.min.as_ref().map(|scalar| scalar.as_array()),
                &self.data_type,
            )?,
            max: Self::scalar_value_from_stat(
                statistics.max.as_ref().map(|scalar| scalar.as_array()),
                &self.data_type,
            )?,
            null_count,
            nan_count,
            negative_nan_count: Some(negative_nan_count),
            bound,
        })
    }

    fn reset(&mut self) -> Result<()> {
        match &mut self.statistics {
            ZoneStatsAccumulator::Full(statistics) => statistics.reset(),
            ZoneStatsAccumulator::NullOnly { null_count } => *null_count = 0,
        }
        Ok(())
    }
}

fn make_zone_map_index_details(
    rows_per_zone: u64,
    use_seeds: bool,
    has_null_bitmap: bool,
) -> prost_types::Any {
    prost_types::Any::from_msg(&pbold::ZoneMapIndexDetails {
        rows_per_zone: Some(rows_per_zone),
        use_seeds: Some(use_seeds),
        has_null_bitmap: Some(has_null_bitmap),
    })
    .unwrap()
}

/// Returns true when seed-based incremental updates should be enabled by
/// default for the given column type.
///
/// Seeds pay off for variable-length types (strings, binary) — which can be
/// arbitrarily wide — and fixed-width types wider than 8 bytes (e.g.
/// Decimal128, FixedSizeBinary tensors). Fixed-width types ≤ 8 bytes (Int64,
/// Float64, …) scan fast enough that the seed overhead is not worth it.
/// Partial reuse of a fragment whose seed records cross target zone
/// boundaries leaves one scattered interval per crossing record to the
/// scan. On object stores each interval is at least one request, and for a
/// narrow column those requests are small: measured against a sequential
/// scan of the whole fragment, reusing the seed stops paying for itself
/// once about half the rows are rescanned anyway. Wide columns save enough
/// bytes per interval to keep partial reuse at any fraction.
const NARROW_PARTIAL_REUSE_MAX_RESCAN_PERCENT: u64 = 50;

/// Whether scanning only `uncovered_rows` of a fragment of `physical_rows`
/// in scattered intervals beats scanning the fragment whole.
fn partial_reuse_pays_off(data_type: &DataType, uncovered_rows: u64, physical_rows: u64) -> bool {
    if !is_narrow_column(data_type) {
        return true;
    }
    uncovered_rows.saturating_mul(100)
        <= physical_rows.saturating_mul(NARROW_PARTIAL_REUSE_MAX_RESCAN_PERCENT)
}

/// Fixed-width values of at most 8 bytes: the column types whose scan is
/// cheap enough that seeds are not requested by default and whose partial
/// reuse is bounded by [`NARROW_PARTIAL_REUSE_MAX_RESCAN_PERCENT`].
fn is_narrow_column(data_type: &DataType) -> bool {
    !default_use_seeds(data_type)
}

fn default_use_seeds(data_type: &DataType) -> bool {
    match data_type {
        // Variable-length: width is unbounded, skipping scans is always valuable.
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => true,
        // Fixed-width types wider than 8 bytes.
        DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => true,
        DataType::FixedSizeBinary(n) => *n > 8,
        // Nested types (FSL, List, Struct, Map, …): typically wide.
        _ if data_type.is_nested() => true,
        _ => false,
    }
}

#[derive(Debug, Default)]
pub struct ZoneMapIndexPlugin;

impl ZoneMapIndexPlugin {
    async fn train_zonemap_index(
        batches_source: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        options: Option<ZoneMapIndexBuilderParams>,
    ) -> Result<Vec<IndexFile>> {
        let value_type = batches_source.schema().field(0).data_type().clone();

        let mut builder = ZoneMapIndexBuilder::try_new(options.unwrap_or_default(), value_type)?;

        builder.train(batches_source).await?;

        builder.write_index(index_store).await
    }
}

pub struct ZoneMapIndexTrainingRequest {
    pub params: ZoneMapIndexBuilderParams,
    pub criteria: TrainingCriteria,
}

impl ZoneMapIndexTrainingRequest {
    pub fn new(params: ZoneMapIndexBuilderParams) -> Self {
        Self {
            params,
            criteria: TrainingCriteria::new(TrainingOrdering::Addresses).with_row_addr(),
        }
    }
}

impl TrainingRequest for ZoneMapIndexTrainingRequest {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn criteria(&self) -> &TrainingCriteria {
        &self.criteria
    }
}

#[async_trait]
impl BasicTrainer for ZoneMapIndexPlugin {
    fn new_training_request(
        &self,
        params: &str,
        field: &Field,
    ) -> Result<Box<dyn TrainingRequest>> {
        let mut params = serde_json::from_str::<ZoneMapIndexBuilderParams>(params)?;
        // Resolve None → type-based default so train_index always sees Some(bool).
        params.use_seeds = Some(params.use_seeds_for(field.data_type()));

        Ok(Box::new(ZoneMapIndexTrainingRequest::new(params)))
    }

    async fn train_index(
        &self,
        data: SendableRecordBatchStream,
        index_store: &dyn IndexStore,
        request: Box<dyn TrainingRequest>,
        _fragment_ids: Option<Vec<u32>>,
        _progress: Arc<dyn crate::progress::IndexBuildProgress>,
    ) -> Result<CreatedIndex> {
        let request = (request as Box<dyn std::any::Any>)
            .downcast::<ZoneMapIndexTrainingRequest>()
            .map_err(|_| {
                Error::invalid_input_source(
                    "must provide training request created by new_training_request".into(),
                )
            })?;
        let rows_per_zone = request.params.rows_per_zone;
        let use_seeds = request.params.use_seeds.unwrap_or(false);
        // Training a new index populates the null bitmap unless the index is
        // configured without one.
        let has_null_bitmap = request.params.store_null_bitmap;
        let files = Self::train_zonemap_index(data, index_store, Some(request.params)).await?;
        Ok(CreatedIndex {
            index_details: make_zone_map_index_details(rows_per_zone, use_seeds, has_null_bitmap),
            index_version: ZONEMAP_INDEX_VERSION,
            files,
        })
    }
}

#[async_trait]
impl ScalarIndexPlugin for ZoneMapIndexPlugin {
    fn basic_trainer(&self) -> Option<&dyn BasicTrainer> {
        Some(self)
    }

    fn name(&self) -> &str {
        "ZoneMap"
    }

    fn provides_exact_answer(&self) -> bool {
        false
    }

    fn version(&self) -> u32 {
        ZONEMAP_INDEX_VERSION
    }

    fn new_query_parser(
        &self,
        index_name: String,
        index_details: &prost_types::Any,
    ) -> Option<Box<dyn ScalarQueryParser>> {
        let has_null_bitmap = index_details
            .to_msg::<pbold::ZoneMapIndexDetails>()
            .ok()
            .and_then(|d| d.has_null_bitmap)
            .unwrap_or(false);
        let parser = SargableQueryParser::new(index_name, self.name().to_string(), true);
        let parser = if has_null_bitmap {
            parser.with_exact_null_tracking()
        } else {
            parser
        };
        Some(Box::new(parser))
    }

    async fn load_index(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        index_version: u32,
        frag_reuse_index: Option<Arc<dyn RowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let use_seeds = index_details
            .to_msg::<pbold::ZoneMapIndexDetails>()
            .ok()
            .and_then(|d| d.use_seeds)
            .unwrap_or(false);
        let version = ZoneMapIndexVersion::from_metadata(index_version)?;
        Ok(
            ZoneMapIndex::load(index_store, frag_reuse_index, cache, use_seeds, version).await?
                as Arc<dyn ScalarIndex>,
        )
    }

    fn supports_batch_row_id_remapping(&self) -> bool {
        true
    }

    async fn load_index_with_remapping(
        &self,
        index_store: Arc<dyn IndexStore>,
        index_details: &prost_types::Any,
        index_version: u32,
        frag_reuse_index: Option<Arc<dyn BatchRowIdRemapper>>,
        cache: &LanceCache,
    ) -> Result<Arc<dyn ScalarIndex>> {
        let use_seeds = index_details
            .to_msg::<pbold::ZoneMapIndexDetails>()
            .ok()
            .and_then(|d| d.use_seeds)
            .unwrap_or(false);
        let version = ZoneMapIndexVersion::from_metadata(index_version)?;
        Ok(ZoneMapIndex::load_with_remapping(
            index_store,
            frag_reuse_index,
            cache,
            use_seeds,
            version,
        )
        .await? as Arc<dyn ScalarIndex>)
    }

    fn might_use_seeds(&self, index_details: &prost_types::Any) -> bool {
        index_details
            .to_msg::<pbold::ZoneMapIndexDetails>()
            .ok()
            .and_then(|d| d.use_seeds)
            .unwrap_or(false)
    }

    async fn create_seed_writer(
        &self,
        field_path: &str,
        data_type: &DataType,
        index_details: &prost_types::Any,
    ) -> Result<Option<Box<dyn crate::scalar::seed::IndexSeedWriter>>> {
        let details = index_details.to_msg::<pbold::ZoneMapIndexDetails>().ok();
        let Some(rows_per_zone) = details.as_ref().and_then(|d| d.rows_per_zone) else {
            return Ok(None);
        };
        if !details.as_ref().and_then(|d| d.use_seeds).unwrap_or(false) {
            return Ok(None);
        }
        // Seeds follow the index's zone size. The null-position policy is
        // not in the details; this entry collects positions (the default),
        // and lance's own collector builds the writer from the persisted
        // index configuration instead (see `dataset::write::seeds`).
        Ok(Some(Box::new(ZoneMapSeedWriter::new(
            field_path,
            rows_per_zone,
            data_type.clone(),
        )?)))
    }

    async fn update_from_seeds(
        &self,
        seeds: Vec<crate::scalar::seed::FragmentSeed>,
        reference_index: Arc<dyn ScalarIndex>,
        index_details: &prost_types::Any,
        dest_store: &dyn IndexStore,
    ) -> Result<Option<CreatedIndex>> {
        // Only an index that records its zone size can take seeds.
        if index_details
            .to_msg::<pbold::ZoneMapIndexDetails>()
            .ok()
            .and_then(|d| d.rows_per_zone)
            .is_none()
        {
            return Ok(None);
        }

        // A seed record is reused only when it fits whole inside one target
        // zone; rows without a reusable record are scanned (see
        // `ReusedZones::reuse`).
        let Some(zone_map) = reference_index.as_any().downcast_ref::<ZoneMapIndex>() else {
            return Ok(None);
        };
        zone_map.try_update_with_seeds(&seeds, dest_store).await
    }
}

/// A seed writer that observes column values during data file writes and
/// accumulates zone map statistics for later harvest during index updates.
///
/// Zone statistics are serialized as Arrow IPC bytes and embedded in the data
/// file footer as a global buffer, keyed by `"lance.seed.<column_name>"`.
/// It has one output contract, seed format version 1; a target version, if
/// one is ever added, must be refused when the writer is created, before any
/// data file is written. Existing files' seeds are never read on the write
/// path, so files carrying other seed versions neither block nor slow a write.
#[derive(Debug)]
pub struct ZoneMapSeedWriter {
    column_name: String,
    rows_per_zone: u64,
    data_type: DataType,
    completed_zones: Vec<ZoneMapStatistics>,
    processor: ZoneMapProcessor,
    rows_in_current_zone: u64,
    next_zone_start: u64,
    /// Null row offsets within this fragment (sequential, 0-indexed).
    null_offsets: RoaringBitmap,
    /// Whether the seed records null positions at all. Counts are always
    /// exact; an index that keeps no bitmap has no use for positions.
    collect_null_positions: bool,
}

impl ZoneMapSeedWriter {
    /// Create a new `ZoneMapSeedWriter` for the given column.
    pub fn new(
        column_name: impl Into<String>,
        rows_per_zone: u64,
        data_type: DataType,
    ) -> Result<Self> {
        if rows_per_zone == 0 {
            return Err(lance_core::Error::invalid_input(
                "rows_per_zone must be greater than zero",
            ));
        }
        let processor = ZoneMapProcessor::new(data_type.clone())?;
        Ok(Self {
            column_name: column_name.into(),
            rows_per_zone,
            data_type,
            completed_zones: Vec::new(),
            processor,
            rows_in_current_zone: 0,
            next_zone_start: 0,
            null_offsets: RoaringBitmap::new(),
            collect_null_positions: true,
        })
    }

    /// Record exact null positions in the seed (the default) or only counts.
    pub fn with_null_positions(mut self, collect: bool) -> Self {
        self.collect_null_positions = collect;
        self
    }

    fn seed_batch_from_zones(
        zones: &[ZoneMapStatistics],
        data_type: &DataType,
    ) -> Result<arrow_array::RecordBatch> {
        let mins = if zones.is_empty() {
            arrow_array::new_empty_array(data_type)
        } else {
            datafusion_common::ScalarValue::iter_to_array(zones.iter().map(|s| s.min.clone()))?
        };
        let maxs = if zones.is_empty() {
            arrow_array::new_empty_array(data_type)
        } else {
            datafusion_common::ScalarValue::iter_to_array(zones.iter().map(|s| s.max.clone()))?
        };
        let null_counts =
            arrow_array::UInt32Array::from_iter_values(zones.iter().map(|s| s.null_count));
        let nan_counts =
            arrow_array::UInt32Array::from_iter_values(zones.iter().map(|s| s.nan_count));
        let negative_nan_counts = arrow_array::UInt32Array::from_iter_values(
            zones
                .iter()
                .map(|s| {
                    s.negative_nan_count.ok_or_else(|| {
                        lance_core::Error::invalid_input(
                            "version 1 seeds require the exact NaN sign split; a zone \
                             without one cannot be serialized as a seed",
                        )
                    })
                })
                .collect::<Result<Vec<u32>>>()?,
        );
        let zone_lengths =
            arrow_array::UInt64Array::from_iter_values(zones.iter().map(|s| s.bound.length as u64));

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new("min", data_type.clone(), true),
            Field::new("max", data_type.clone(), true),
            Field::new("null_count", DataType::UInt32, false),
            Field::new("nan_count", DataType::UInt32, false),
            Field::new("negative_nan_count", DataType::UInt32, false),
            Field::new("zone_length", DataType::UInt64, false),
        ]));

        let columns: Vec<ArrayRef> = vec![
            mins,
            maxs,
            Arc::new(null_counts) as ArrayRef,
            Arc::new(nan_counts) as ArrayRef,
            Arc::new(negative_nan_counts) as ArrayRef,
            Arc::new(zone_lengths) as ArrayRef,
        ];
        Ok(arrow_array::RecordBatch::try_new(schema, columns)?)
    }

    /// Deserialize zone map seed bytes (Arrow IPC) into zone statistics and a null bitmap.
    ///
    /// Returns zone statistics (zones located by their cumulative lengths) and an
    /// optional `RoaringBitmap` of null row offsets within the fragment. The bitmap is absent
    /// for seeds written by older code that did not track null positions.
    #[cfg(test)]
    pub(crate) fn deserialize_seed(
        fragment_id: u64,
        bytes: &bytes::Bytes,
    ) -> Result<(Vec<ZoneMapStatistics>, Option<RoaringBitmap>)> {
        Self::decode_seed(fragment_id, bytes, true)
    }

    /// [`Self::deserialize_seed`], with the null positions left undecoded
    /// (`None`) when the caller does not need them: the hex metadata is still
    /// part of the bytes read, but it is neither decoded nor allocated.
    fn decode_seed(
        fragment_id: u64,
        bytes: &bytes::Bytes,
        decode_null_positions: bool,
    ) -> Result<(Vec<ZoneMapStatistics>, Option<RoaringBitmap>)> {
        use arrow_ipc::reader::FileReader;
        use std::io::Cursor;

        let cursor = Cursor::new(bytes.as_ref());
        let mut reader = FileReader::try_new(cursor, None).map_err(|e| {
            lance_core::Error::invalid_input(format!("failed to read zone map seed IPC: {}", e))
        })?;

        // Version dispatch: absence means version 0 (the original schema,
        // whose max may carry the legacy +NaN marker and whose NaN sign split
        // is unknown). Unknown future versions must not be misread as either,
        // and are refused before any other metadata is interpreted.
        let version = reader.schema().metadata.get(SEED_VERSION_META_KEY).cloned();
        let is_v1 = match version.as_deref() {
            None => false,
            Some(SEED_FORMAT_VERSION) => true,
            Some(other) => {
                return Err(lance_core::Error::not_supported_source(
                    format!(
                        "zone map seed format version '{}' is not supported by this \
                         build; please upgrade Lance",
                        other
                    )
                    .into(),
                ));
            }
        };

        let null_bitmap = if let Some(hex) = reader
            .schema()
            .metadata
            .get(SEED_NULL_BITMAP_META_KEY)
            .filter(|_| decode_null_positions)
        {
            let bitmap_bytes = hex_decode(hex).map_err(|e| {
                lance_core::Error::invalid_input(format!(
                    "failed to decode seed null bitmap hex: {}",
                    e
                ))
            })?;
            let bitmap = if bitmap_bytes.is_empty() {
                RoaringBitmap::default()
            } else {
                RoaringBitmap::deserialize_from(bitmap_bytes.as_slice()).map_err(|e| {
                    lance_core::Error::invalid_input(format!(
                        "failed to deserialize seed null bitmap: {}",
                        e
                    ))
                })?
            };
            Some(bitmap)
        } else {
            None
        };

        let batch = match reader.next() {
            Some(Ok(batch)) => batch,
            Some(Err(e)) => {
                return Err(lance_core::Error::invalid_input(format!(
                    "failed to read zone map seed batch: {}",
                    e
                )));
            }
            None => return Ok((Vec::new(), null_bitmap)),
        };

        let min_col = batch
            .column_by_name("min")
            .ok_or_else(|| lance_core::Error::invalid_input("seed batch missing 'min' column"))?;
        let max_col = batch
            .column_by_name("max")
            .ok_or_else(|| lance_core::Error::invalid_input("seed batch missing 'max' column"))?;
        let null_count_col = batch
            .column_by_name("null_count")
            .ok_or_else(|| {
                lance_core::Error::invalid_input("seed batch missing 'null_count' column")
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| lance_core::Error::invalid_input("seed 'null_count' is not UInt32"))?;
        let nan_count_col = batch
            .column_by_name("nan_count")
            .ok_or_else(|| {
                lance_core::Error::invalid_input("seed batch missing 'nan_count' column")
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt32Array>()
            .ok_or_else(|| lance_core::Error::invalid_input("seed 'nan_count' is not UInt32"))?;
        let zone_length_col = batch
            .column_by_name("zone_length")
            .ok_or_else(|| {
                lance_core::Error::invalid_input("seed batch missing 'zone_length' column")
            })?
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| lance_core::Error::invalid_input("seed 'zone_length' is not UInt64"))?;
        let negative_nan_count_col = if is_v1 {
            Some(
                batch
                    .column_by_name("negative_nan_count")
                    .ok_or_else(|| {
                        lance_core::Error::invalid_input(
                            "version 1 seed batch missing 'negative_nan_count' column",
                        )
                    })?
                    .as_any()
                    .downcast_ref::<arrow_array::UInt32Array>()
                    .ok_or_else(|| {
                        lance_core::Error::invalid_input("seed 'negative_nan_count' is not UInt32")
                    })?,
            )
        } else {
            None
        };

        let num_zones = batch.num_rows();
        let mut zones = Vec::with_capacity(num_zones);
        // Zones cover the file's rows in order without gaps, so zone `i`
        // starts at the checked cumulative sum of the earlier lengths; short
        // zones may appear anywhere (binary-copy carry concatenates inputs).
        let mut next_zone_start: u64 = 0;
        for i in 0..num_zones {
            let zone_start = next_zone_start;
            let zone_length = zone_length_col.value(i);
            let null_count = null_count_col.value(i);
            let nan_count = nan_count_col.value(i);
            let negative_nan_count = negative_nan_count_col.map(|col| col.value(i));
            if zone_length == 0 {
                return Err(lance_core::Error::invalid_input(format!(
                    "seed zone {} has zero length",
                    i
                )));
            }
            next_zone_start = next_zone_start.checked_add(zone_length).ok_or_else(|| {
                lance_core::Error::invalid_input("seed zone lengths overflow u64")
            })?;
            if is_v1 {
                // Validate per the format: exact counts that fit the zone,
                // checked without overflow.
                if u64::from(null_count) + u64::from(nan_count) > zone_length {
                    return Err(lance_core::Error::invalid_input(format!(
                        "seed zone {} counts exceed zone_length: null_count={} nan_count={} \
                         zone_length={}",
                        i, null_count, nan_count, zone_length
                    )));
                }
                if let Some(negative) = negative_nan_count
                    && negative > nan_count
                {
                    return Err(lance_core::Error::invalid_input(format!(
                        "seed zone {} negative_nan_count {} exceeds nan_count {}",
                        i, negative, nan_count
                    )));
                }
            }
            zones.push(ZoneMapStatistics {
                min: datafusion_common::ScalarValue::try_from_array(min_col, i)?,
                max: datafusion_common::ScalarValue::try_from_array(max_col, i)?,
                null_count,
                nan_count,
                negative_nan_count,
                bound: ZoneBound {
                    fragment_id,
                    start: zone_start,
                    length: zone_length as usize,
                },
            });
        }
        Ok((zones, null_bitmap))
    }
}

impl ZoneMapSeedWriter {
    /// Embed the null positions in the batch's schema metadata so
    /// `deserialize_seed` can reconstruct them.
    fn with_null_positions_metadata(
        batch: RecordBatch,
        mut null_offsets: RoaringBitmap,
    ) -> lance_core::Result<RecordBatch> {
        // Run containers make contiguous null regions a few bytes each.
        null_offsets.optimize();
        let bitmap_bytes = if null_offsets.is_empty() {
            Vec::default()
        } else {
            let mut bitmap_bytes = Vec::with_capacity(null_offsets.serialized_size());
            null_offsets
                .serialize_into(&mut bitmap_bytes)
                .map_err(|e| {
                    lance_core::Error::invalid_input(format!(
                        "failed to serialize seed null bitmap: {}",
                        e
                    ))
                })?;
            bitmap_bytes
        };
        let mut schema = batch.schema().as_ref().clone();
        schema.metadata.insert(
            SEED_NULL_BITMAP_META_KEY.to_string(),
            hex_encode(&bitmap_bytes),
        );
        Ok(RecordBatch::try_new(
            Arc::new(schema),
            batch.columns().to_vec(),
        )?)
    }

    /// Record the null positions of `chunk` as runs: one bitmap insert per
    /// contiguous null region instead of one per null row, so all-null and
    /// clustered-null data cost almost nothing to track. `base_offset` is the
    /// fragment-local offset of the chunk's first row.
    fn record_null_runs(&mut self, chunk: &ArrayRef, base_offset: u32) {
        let len = chunk.len() as u32;
        let Some(nulls) = chunk.nulls() else {
            return;
        };
        // The validity buffer marks valid rows; the gaps between its set runs
        // are the null runs.
        let mut cursor = 0u32;
        for (start, end) in nulls.inner().set_slices() {
            let start = start as u32;
            if start > cursor {
                self.null_offsets
                    .insert_range(base_offset + cursor..base_offset + start);
            }
            cursor = end as u32;
        }
        if cursor < len {
            self.null_offsets
                .insert_range(base_offset + cursor..base_offset + len);
        }
    }
}

impl IndexSeedWriter for ZoneMapSeedWriter {
    fn column_name(&self) -> &str {
        &self.column_name
    }

    fn observe_batch(&mut self, values: &ArrayRef) -> lance_core::Result<()> {
        let mut offset = 0usize;

        while offset < values.len() {
            let remaining_in_zone = self.rows_per_zone - self.rows_in_current_zone;
            let chunk_len = ((values.len() as u64 - offset as u64).min(remaining_in_zone)) as usize;
            let chunk = values.slice(offset, chunk_len);
            self.processor.process_chunk(&chunk)?;

            if self.collect_null_positions && chunk.null_count() > 0 {
                let base_offset = (self.next_zone_start + self.rows_in_current_zone) as u32;
                self.record_null_runs(&chunk, base_offset);
            }

            self.rows_in_current_zone += chunk_len as u64;
            offset += chunk_len;

            if self.rows_in_current_zone >= self.rows_per_zone {
                let bound = ZoneBound {
                    fragment_id: 0,
                    start: self.next_zone_start,
                    length: self.rows_per_zone as usize,
                };
                let stats = self.processor.finish_zone(bound)?;
                self.processor.reset()?;
                self.completed_zones.push(stats);
                self.next_zone_start += self.rows_per_zone;
                self.rows_in_current_zone = 0;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> lance_core::Result<Option<bytes::Bytes>> {
        use arrow_ipc::writer::FileWriter;
        use std::io::Cursor;

        // Flush partial final zone
        if self.rows_in_current_zone > 0 {
            let bound = ZoneBound {
                fragment_id: 0,
                start: self.next_zone_start,
                length: self.rows_in_current_zone as usize,
            };
            let stats = self.processor.finish_zone(bound)?;
            self.processor.reset()?;
            self.completed_zones.push(stats);
            self.next_zone_start += self.rows_in_current_zone;
            self.rows_in_current_zone = 0;
        }

        if self.completed_zones.is_empty() {
            return Ok(None);
        }

        let batch = Self::seed_batch_from_zones(&self.completed_zones, &self.data_type)?;

        let batch = if self.collect_null_positions {
            Self::with_null_positions_metadata(batch, std::mem::take(&mut self.null_offsets))?
        } else {
            batch
        };
        let mut schema = batch.schema().as_ref().clone();
        schema.metadata.insert(
            SEED_VERSION_META_KEY.to_string(),
            SEED_FORMAT_VERSION.to_string(),
        );
        let batch = RecordBatch::try_new(Arc::new(schema), batch.columns().to_vec())?;

        // Serialize to Arrow IPC
        let mut buf = Cursor::new(Vec::new());
        {
            let mut writer = FileWriter::try_new(&mut buf, batch.schema_ref()).map_err(|e| {
                lance_core::Error::invalid_input(format!("failed to create IPC writer: {}", e))
            })?;
            writer.write(&batch).map_err(|e| {
                lance_core::Error::invalid_input(format!("failed to write IPC batch: {}", e))
            })?;
            writer.finish().map_err(|e| {
                lance_core::Error::invalid_input(format!("failed to finish IPC writer: {}", e))
            })?;
        }

        // Reset state for next fragment
        self.completed_zones.clear();
        self.next_zone_start = 0;
        self.processor = ZoneMapProcessor::new(self.data_type.clone())?;

        Ok(Some(bytes::Bytes::from(buf.into_inner())))
    }

    fn schema_metadata_key(&self) -> String {
        format!(
            "{}{}",
            crate::scalar::seed::SEED_META_KEY_PREFIX,
            self.column_name
        )
    }

    fn schema_metadata_value(&self, buf_index: u32) -> String {
        format!("{}:{}", buf_index, self.rows_per_zone)
    }

    /// Version 1 seeds change how `max` and zone positions are read, so a
    /// reader must declare support for them.
    fn required_feature_flags(&self) -> u64 {
        lance_table::feature_flags::FLAG_ZONE_WRITE_SEEDS
    }
}

/// A write seed read from the data file currently serving a fragment, with
/// the number of physical rows that file holds.
#[derive(Debug, Clone)]
pub struct HarvestedZoneSeed {
    pub seed: crate::scalar::seed::FragmentSeed,
    pub physical_rows: u64,
}

/// How one fragment's seed was used by [`ReusedZones::reuse`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedOutcome {
    /// Every target zone of the fragment was summarized from whole records.
    Complete,
    /// These physical intervals of the fragment must be scanned; every other
    /// row was summarized from whole records.
    Partial(Vec<std::ops::Range<u64>>),
    /// Nothing was reused; the whole fragment must be scanned.
    Rejected,
}

/// Target zones assembled from valid write seeds, ready for
/// [`build_zonemap_segment`]. Zones are keyed by `(fragment, zone index)`;
/// a zone fed only partly from seeds is completed by the scan.
#[derive(Debug, Default)]
pub struct ReusedZones {
    zones: HashMap<(u64, u64), ZoneMapStatistics>,
    null_rows: RowAddrTreeMap,
    with_null_positions: bool,
    /// Fragments whose every zone came from whole records.
    pub reused_fragments: Vec<u64>,
    /// Fragments with whole records in some zones and scanned intervals.
    pub partial_fragments: Vec<u64>,
    /// Seed records folded into zones.
    pub reused_records: usize,
    /// Fragments that had a seed which could not be used; they are scanned.
    pub rejected_fragments: Vec<u64>,
}

impl ReusedZones {
    /// An empty accumulator; `need_null_positions` must match the index
    /// the zones are built into (see [`build_zonemap_segment`]).
    pub fn new(need_null_positions: bool) -> Self {
        Self {
            with_null_positions: need_null_positions,
            ..Default::default()
        }
    }

    /// Fold another assembly into this one. The two cover disjoint
    /// fragments, so zone keys never collide.
    pub fn merge(&mut self, other: Self) {
        self.zones.extend(other.zones);
        self.null_rows |= &other.null_rows;
        self.reused_fragments.extend(other.reused_fragments);
        self.partial_fragments.extend(other.partial_fragments);
        self.rejected_fragments.extend(other.rejected_fragments);
        self.reused_records += other.reused_records;
    }

    pub fn num_zones(&self) -> usize {
        self.zones.len()
    }

    /// Validate one fragment's seed and fold every record that lies inside
    /// a single target zone `[k * rows_per_zone, (k + 1) * rows_per_zone)`
    /// into that zone. A record crossing a target boundary cannot be split
    /// into exact counts, so its rows are left to the scan: with
    /// `allow_partial` the fragment becomes [`SeedOutcome::Partial`] with
    /// those intervals, otherwise (and when no record fits at all) the
    /// fragment is [`SeedOutcome::Rejected`] and nothing of it is kept.
    /// The borrowed raw bytes are the caller's to release.
    pub fn reuse(
        &mut self,
        harvested: &HarvestedZoneSeed,
        rows_per_zone: u64,
        data_type: &DataType,
        allow_partial: bool,
    ) -> Result<SeedOutcome> {
        if rows_per_zone == 0 {
            return Err(Error::invalid_input(
                "rows_per_zone must be greater than zero",
            ));
        }
        let fragment_id = harvested.seed.fragment_id;
        let Some(validated) = validate_seed(
            &harvested.seed,
            Some(harvested.physical_rows),
            data_type,
            self.with_null_positions,
        )?
        else {
            self.rejected_fragments.push(fragment_id);
            return Ok(SeedOutcome::Rejected);
        };
        let placement = place_records(&validated.records, rows_per_zone, harvested.physical_rows);
        if placement.fitted.is_empty() {
            self.rejected_fragments.push(fragment_id);
            return Ok(SeedOutcome::Rejected);
        }
        if !placement.uncovered.is_empty() {
            let uncovered_rows: u64 = placement
                .uncovered
                .iter()
                .map(|interval| interval.end - interval.start)
                .sum();
            if !allow_partial
                || !partial_reuse_pays_off(data_type, uncovered_rows, harvested.physical_rows)
            {
                self.rejected_fragments.push(fragment_id);
                return Ok(SeedOutcome::Rejected);
            }
        }
        for (zone_index, record) in placement.fitted {
            let target = self
                .zones
                .entry((fragment_id, zone_index))
                .or_insert_with(|| ZoneMapStatistics {
                    min: ScalarValue::try_new_null(data_type).unwrap_or(record.min.clone()),
                    max: ScalarValue::try_new_null(data_type).unwrap_or(record.max.clone()),
                    null_count: 0,
                    nan_count: 0,
                    negative_nan_count: Some(0),
                    bound: target_zone(
                        fragment_id,
                        zone_index,
                        rows_per_zone,
                        harvested.physical_rows,
                    ),
                });
            fold_zone_stats(target, &record)?;
            self.reused_records += 1;
        }
        if let Some(mut bitmap) = validated.null_positions {
            // Checked by the validator.
            let fragment = u32::try_from(fragment_id)
                .map_err(|_| Error::internal(format!("fragment id {fragment_id} exceeds u32")))?;
            // Positions inside intervals left to the scan come from the scan.
            for interval in &placement.uncovered {
                bitmap.remove_range(interval.start as u32..interval.end as u32);
            }
            self.null_rows.insert_bitmap(fragment, bitmap);
        }
        if placement.uncovered.is_empty() {
            self.reused_fragments.push(fragment_id);
            Ok(SeedOutcome::Complete)
        } else {
            self.partial_fragments.push(fragment_id);
            Ok(SeedOutcome::Partial(placement.uncovered))
        }
    }
}

/// A zone write seed parsed and shape-validated for binary-copy carrying.
///
/// Binary-copy compaction moves whole data files' pages verbatim, so an
/// output file's correct seed is the concatenation of its inputs' zone
/// records in copy order. Parsing proves the buffer really is a zone map
/// seed of a supported version whose statistics satisfy that version's
/// layout contract; merging refuses anything it cannot reproduce exactly.
#[derive(Debug)]
pub struct MergeableSeed {
    /// `None` is version 0; `"1"` is version 1. Other versions classify as
    /// [`CarriedSeed::UnsupportedVersion`] before any statistic is read.
    version: Option<String>,
    /// The zone records under a metadata-free schema, so sources concatenate.
    batch: RecordBatch,
    /// Exact null offsets within the covered rows; `None` when the source
    /// predates the bitmap.
    null_bitmap: Option<RoaringBitmap>,
    /// Checked sum of `zone_length`: the rows this seed covers.
    total_rows: u64,
    /// The collection granularity the locator declares (write-time
    /// configuration, not a layout rule).
    rows_per_zone: u64,
}

impl MergeableSeed {
    /// The rows this seed covers; the caller must check it against the file's
    /// physical row count before treating the seed as complete coverage.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Whether this seed declares format version 1 (finite bounds plus exact
    /// NaN sign counts, locatable by zone length).
    pub fn is_version_1(&self) -> bool {
        self.version.as_deref() == Some(SEED_FORMAT_VERSION)
    }

    /// The collection granularity the source's locator declared.
    pub fn rows_per_zone(&self) -> u64 {
        self.rows_per_zone
    }

    /// Exact null offsets within the covered rows, when the source recorded them.
    pub fn null_bitmap(&self) -> Option<&RoaringBitmap> {
        self.null_bitmap.as_ref()
    }
}

/// What a seed buffer turned out to be when parsed for binary-copy carrying.
#[derive(Debug)]
pub enum CarriedSeed {
    /// A seed of a supported version whose statistics passed validation.
    Supported(MergeableSeed),
    /// A seed declaring a format version this build cannot interpret. It is
    /// a recognized seed (so it does not disqualify the file), but its column
    /// is copied without a seed: unknown statistics are never relabeled.
    UnsupportedVersion(String),
}

fn seed_parse_error(message: impl std::fmt::Display) -> lance_core::Error {
    lance_core::Error::invalid_input(format!("zone map seed: {}", message))
}

/// Parse and validate one seed buffer for binary-copy carrying.
///
/// `rows_per_zone` is the collection granularity the locator declares
/// (`<buffer_index>:<rows_per_zone>`); zones are located by their lengths,
/// never by it. A buffer whose `version` metadata this build does not
/// support classifies as [`CarriedSeed::UnsupportedVersion`] without its
/// statistics being read. An error means the buffer must be neither carried
/// nor silently dropped: an unrecognized shape, or statistics that
/// contradict the layout (zero-length zones, counts beyond the zone, a
/// bitmap outside the covered rows). The caller then refuses binary copy for
/// the file, preserving the existing protection against unknown global
/// buffers.
pub fn parse_seed_for_binary_copy(bytes: &bytes::Bytes, rows_per_zone: u64) -> Result<CarriedSeed> {
    if rows_per_zone == 0 {
        return Err(seed_parse_error("locator declares rows_per_zone 0"));
    }
    use arrow_ipc::reader::FileReader;
    use std::io::Cursor;

    let mut reader = FileReader::try_new(Cursor::new(bytes.as_ref()), None)
        .map_err(|e| seed_parse_error(format!("not a readable IPC buffer: {}", e)))?;
    let schema = reader.schema();
    let version = schema.metadata.get(SEED_VERSION_META_KEY).cloned();
    let is_v1 = match version.as_deref() {
        None => false,
        Some(SEED_FORMAT_VERSION) => true,
        Some(other) => return Ok(CarriedSeed::UnsupportedVersion(other.to_string())),
    };
    let null_bitmap_hex = schema.metadata.get(SEED_NULL_BITMAP_META_KEY).cloned();

    let batch = match reader.next() {
        Some(Ok(batch)) => batch,
        Some(Err(e)) => {
            return Err(seed_parse_error(format!("unreadable zone batch: {}", e)));
        }
        None => return Err(seed_parse_error("no zone batch")),
    };
    if reader.next().is_some() {
        return Err(seed_parse_error("more than one zone batch"));
    }

    let expected_columns = if is_v1 { 6 } else { 5 };
    if batch.num_columns() != expected_columns {
        return Err(seed_parse_error(format!(
            "expected {} columns, found {}",
            expected_columns,
            batch.num_columns()
        )));
    }
    let min_col = batch
        .column_by_name("min")
        .ok_or_else(|| seed_parse_error("missing 'min' column"))?;
    let max_col = batch
        .column_by_name("max")
        .ok_or_else(|| seed_parse_error("missing 'max' column"))?;
    if min_col.data_type() != max_col.data_type() {
        return Err(seed_parse_error("'min' and 'max' column types differ"));
    }
    let null_count_col = batch
        .column_by_name("null_count")
        .and_then(|col| col.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| seed_parse_error("missing or mistyped 'null_count' column"))?
        .clone();
    let nan_count_col = batch
        .column_by_name("nan_count")
        .and_then(|col| col.as_any().downcast_ref::<UInt32Array>())
        .ok_or_else(|| seed_parse_error("missing or mistyped 'nan_count' column"))?
        .clone();
    let zone_length_col = batch
        .column_by_name("zone_length")
        .and_then(|col| col.as_any().downcast_ref::<UInt64Array>())
        .ok_or_else(|| seed_parse_error("missing or mistyped 'zone_length' column"))?
        .clone();
    let negative_nan_count_col = if is_v1 {
        Some(
            batch
                .column_by_name("negative_nan_count")
                .and_then(|col| col.as_any().downcast_ref::<UInt32Array>())
                .ok_or_else(|| seed_parse_error("missing or mistyped 'negative_nan_count' column"))?
                .clone(),
        )
    } else {
        None
    };
    // The count columns are non-null by contract; `min`/`max` are nullable
    // (a null bound means unknown or no ordinary values).
    for name in [
        "null_count",
        "nan_count",
        "negative_nan_count",
        "zone_length",
    ] {
        if let Some(col) = batch.column_by_name(name)
            && col.null_count() > 0
        {
            return Err(seed_parse_error(format!(
                "'{}' column must be non-null",
                name
            )));
        }
    }

    let num_zones = batch.num_rows();
    if num_zones == 0 {
        return Err(seed_parse_error("zero zones"));
    }
    let mut total_rows: u64 = 0;
    for i in 0..num_zones {
        let zone_length = zone_length_col.value(i);
        if zone_length == 0 {
            return Err(seed_parse_error(format!("zone {} has zero length", i)));
        }
        let null_count = null_count_col.value(i);
        let nan_count = nan_count_col.value(i);
        if u64::from(null_count) + u64::from(nan_count) > zone_length {
            return Err(seed_parse_error(format!(
                "zone {} counts exceed zone_length: null_count={} nan_count={} zone_length={}",
                i, null_count, nan_count, zone_length
            )));
        }
        if let Some(negative) = negative_nan_count_col.as_ref().map(|col| col.value(i))
            && negative > nan_count
        {
            return Err(seed_parse_error(format!(
                "zone {} negative_nan_count {} exceeds nan_count {}",
                i, negative, nan_count
            )));
        }
        total_rows = total_rows
            .checked_add(zone_length)
            .ok_or_else(|| seed_parse_error("zone_length sum overflows u64"))?;
    }

    let null_bitmap = match null_bitmap_hex {
        None => None,
        Some(hex) => {
            let bitmap_bytes = hex_decode(&hex)
                .map_err(|e| seed_parse_error(format!("bad null bitmap hex: {}", e)))?;
            let bitmap = if bitmap_bytes.is_empty() {
                RoaringBitmap::default()
            } else {
                RoaringBitmap::deserialize_from(bitmap_bytes.as_slice())
                    .map_err(|e| seed_parse_error(format!("bad null bitmap: {}", e)))?
            };
            if let Some(max) = bitmap.max()
                && u64::from(max) >= total_rows
            {
                return Err(seed_parse_error(format!(
                    "null bitmap offset {} outside the {} covered rows",
                    max, total_rows
                )));
            }
            Some(bitmap)
        }
    };

    // A metadata-free schema, so batches from different sources concatenate.
    let canonical_schema = Arc::new(arrow_schema::Schema::new(batch.schema().fields().clone()));
    let batch = RecordBatch::try_new(canonical_schema, batch.columns().to_vec())?;

    Ok(CarriedSeed::Supported(MergeableSeed {
        version,
        batch,
        null_bitmap,
        total_rows,
        rows_per_zone,
    }))
}

/// Merge the seeds of the files an output binary-copies, in copy order, into
/// the output file's seed.
///
/// Every record keeps its original statistics; nothing is recomputed or
/// re-chunked. Zones are located by their lengths, so sources collected at
/// different granularities concatenate freely and short zones may land
/// anywhere in the output (the caller picks the output locator's
/// `rows_per_zone` from the current configuration). `Ok(None)` means the
/// sources cannot be carried: any version 0 source (no implicit format
/// conversion) or mismatched value types.
///
/// The null bitmap is the union of the sources' bitmaps shifted by each
/// source's cumulative row offset; any source without a bitmap leaves the
/// output without one (absence is "unknown", never zero).
pub fn merge_seeds_for_binary_copy(sources: &[&MergeableSeed]) -> Result<Option<bytes::Bytes>> {
    use arrow_ipc::writer::FileWriter;
    use std::io::Cursor;

    let Some(first) = sources.first() else {
        return Ok(None);
    };
    if sources
        .iter()
        .any(|source| !source.is_version_1() || source.batch.schema() != first.batch.schema())
    {
        return Ok(None);
    }

    let mut null_bitmap = sources
        .iter()
        .all(|source| source.null_bitmap.is_some())
        .then(RoaringBitmap::new);
    let mut offset: u64 = 0;
    for source in sources {
        if let Some(merged) = null_bitmap.as_mut() {
            for value in source.null_bitmap.as_ref().expect("checked above") {
                let shifted = offset
                    .checked_add(u64::from(value))
                    .and_then(|shifted| u32::try_from(shifted).ok());
                let Some(shifted) = shifted else {
                    // The offsets no longer fit the bitmap's 32-bit space.
                    return Ok(None);
                };
                merged.insert(shifted);
            }
        }
        offset = offset
            .checked_add(source.total_rows)
            .ok_or_else(|| seed_parse_error("merged row count overflows u64"))?;
    }

    let batches: Vec<&RecordBatch> = sources.iter().map(|source| &source.batch).collect();
    let merged = arrow_select::concat::concat_batches(&first.batch.schema(), batches)
        .map_err(|e| seed_parse_error(format!("failed to concatenate zone records: {}", e)))?;

    let mut schema = merged.schema().as_ref().clone();
    schema.metadata.insert(
        SEED_VERSION_META_KEY.to_string(),
        SEED_FORMAT_VERSION.to_string(),
    );
    if let Some(mut bitmap) = null_bitmap {
        bitmap.optimize();
        let bitmap_bytes = if bitmap.is_empty() {
            Vec::default()
        } else {
            let mut bitmap_bytes = Vec::with_capacity(bitmap.serialized_size());
            bitmap
                .serialize_into(&mut bitmap_bytes)
                .map_err(|e| seed_parse_error(format!("failed to serialize null bitmap: {}", e)))?;
            bitmap_bytes
        };
        schema.metadata.insert(
            SEED_NULL_BITMAP_META_KEY.to_string(),
            hex_encode(&bitmap_bytes),
        );
    }
    let batch = RecordBatch::try_new(Arc::new(schema), merged.columns().to_vec())?;

    let mut buf = Cursor::new(Vec::new());
    {
        let mut writer = FileWriter::try_new(&mut buf, batch.schema_ref())
            .map_err(|e| seed_parse_error(format!("failed to create IPC writer: {}", e)))?;
        writer
            .write(&batch)
            .map_err(|e| seed_parse_error(format!("failed to write IPC batch: {}", e)))?;
        writer
            .finish()
            .map_err(|e| seed_parse_error(format!("failed to finish IPC writer: {}", e)))?;
    }
    Ok(Some(bytes::Bytes::from(buf.into_inner())))
}

/// The record size a seed was written with, from its locator.
pub fn seed_rows_per_zone(seed: &crate::scalar::seed::FragmentSeed) -> Option<u64> {
    seed.metadata_value
        .split_once(':')
        .and_then(|(_, rows)| rows.parse::<u64>().ok())
        .filter(|rows| *rows > 0)
}

/// Fold the seeds of each fragment into their target zones without reading
/// the column (see [`ReusedZones::reuse`] with `allow_partial = false`).
pub fn reuse_zone_seeds(
    seeds: &[HarvestedZoneSeed],
    rows_per_zone: u64,
    data_type: &DataType,
    need_null_positions: bool,
) -> Result<ReusedZones> {
    let mut reused = ReusedZones::new(need_null_positions);
    for harvested in seeds {
        reused.reuse(harvested, rows_per_zone, data_type, false)?;
    }
    Ok(reused)
}

/// The target zone `k` of a fragment of `physical_rows` rows.
fn target_zone(
    fragment_id: u64,
    zone_index: u64,
    rows_per_zone: u64,
    physical_rows: u64,
) -> ZoneBound {
    let start = zone_index * rows_per_zone;
    ZoneBound {
        fragment_id,
        start,
        length: ((start + rows_per_zone).min(physical_rows) - start) as usize,
    }
}

/// Records placed in the target zone they lie in, and the intervals of the
/// records that cross a target boundary.
struct Placement {
    fitted: Vec<(u64, ZoneMapStatistics)>,
    uncovered: Vec<std::ops::Range<u64>>,
}

/// Interval intersection: a record `[start, end)` lies in target zone `k`
/// iff `start / T == (end - 1) / T`. Records tile `[0, physical_rows)`, so
/// the rows not covered by fitted records are exactly the crossing records
/// plus the records that cannot satisfy the version 1 contract: a version 0
/// record with NaNs hides its ordinary maximum behind the marker and has an
/// unknown sign split, so its rows are scanned. A version 0 record without
/// NaNs provably has a zero split and is fitted with `Some(0)`.
fn place_records(
    records: &[ZoneMapStatistics],
    rows_per_zone: u64,
    _physical_rows: u64,
) -> Placement {
    let mut placement = Placement {
        fitted: Vec::with_capacity(records.len()),
        uncovered: Vec::new(),
    };
    for record in records {
        let start = record.bound.start;
        let end = start + record.bound.length as u64;
        let first = start / rows_per_zone;
        let last = (end - 1) / rows_per_zone;
        let split_known = record.negative_nan_count.is_some() || record.nan_count == 0;
        if first == last && split_known {
            let mut record = record.clone();
            record.negative_nan_count.get_or_insert(0);
            placement.fitted.push((first, record));
        } else {
            placement.uncovered.push(start..end);
        }
    }
    placement
}

struct ValidatedSeed {
    records: Vec<ZoneMapStatistics>,
    null_positions: Option<RoaringBitmap>,
}

/// Decode and check one fragment's seed; `None` means the fragment is scanned.
///
/// Records tile the file's rows in order; each holds at most the locator's
/// row count (a binary-copy carry concatenates inputs, so a short record may
/// appear anywhere, not only last). `expected_rows` is `None` when the
/// caller cannot know the file's physical row count; the layout, count and
/// bound checks still apply, coverage does not. A version 1 record carries
/// ordinary bounds and an exact sign split; a version 0 record carries the
/// legacy NaN marker and no split, which is accepted here and decided per
/// record by [`place_records`]. The decoder stays strict (a seed format
/// version this build does not know is `NotSupported` there); at this
/// optional reuse boundary that seed is simply unavailable and the
/// fragment's column is scanned, which yields valid statistics from
/// readable data. Table and data-file feature checks are unaffected.
fn validate_seed(
    seed: &crate::scalar::seed::FragmentSeed,
    expected_rows: Option<u64>,
    data_type: &DataType,
    need_null_positions: bool,
) -> Result<Option<ValidatedSeed>> {
    let fragment_id = seed.fragment_id;
    let reject = |reason: String| {
        log::debug!("not reusing the zone map seed of fragment {fragment_id}: {reason}");
        Ok(None)
    };
    if u32::try_from(fragment_id).is_err() {
        return reject("fragment id exceeds u32".into());
    }
    // The locator's record size is the writer's configuration, not a cap on
    // carried records (a binary copy keeps older, larger records under a
    // newer locator); a record's fit is decided against the target zones.
    if seed_rows_per_zone(seed).is_none() {
        return reject("locator carries no record size".into());
    }
    let (records, null_positions) =
        match ZoneMapSeedWriter::decode_seed(fragment_id, &seed.bytes, need_null_positions) {
            Ok(decoded) => decoded,
            Err(err @ Error::NotSupported { .. }) => {
                return reject(format!("unsupported seed version, scanning instead: {err}"));
            }
            Err(err) => return reject(format!("undecodable: {err}")),
        };
    if records.is_empty() {
        return reject("no records".into());
    }
    let nested = data_type.is_nested();
    let floating = nan_scalar(data_type).is_some();
    let mut covered: u64 = 0;
    let mut total_nulls: u64 = 0;
    for (i, record) in records.iter().enumerate() {
        let rows = record.bound.length as u64;
        if rows == 0 {
            return reject(format!("record {i} is empty"));
        }
        if record.bound.start != covered {
            return reject(format!(
                "record {i} starts at {} after {covered} rows",
                record.bound.start
            ));
        }
        covered = match covered.checked_add(rows) {
            Some(covered) => covered,
            None => return reject("row count overflow".into()),
        };
        let counted = u64::from(record.null_count) + u64::from(record.nan_count);
        if counted > rows {
            return reject(format!(
                "record {i} counts {counted} nulls and NaNs in {rows} rows"
            ));
        }
        total_nulls += u64::from(record.null_count);
        if record.nan_count > 0 && !floating {
            return reject(format!("record {i} counts NaNs in a non-floating column"));
        }
        if nested {
            // Nested bounds are never consulted; only the null count matters.
            continue;
        }
        if record.min.data_type() != *data_type || record.max.data_type() != *data_type {
            return reject(format!(
                "record {i} bounds are {:?}, the column is {data_type:?}",
                record.min.data_type()
            ));
        }
        let min_is_nan = ZoneMapIndex::scalar_is_nan(&record.min);
        let max_is_nan = ZoneMapIndex::scalar_is_nan(&record.max);
        match record.negative_nan_count {
            Some(negative) => {
                if negative > record.nan_count {
                    return reject(format!(
                        "record {i} counts {negative} negative NaNs of {}",
                        record.nan_count
                    ));
                }
                if min_is_nan || max_is_nan {
                    return reject(format!("record {i} carries a NaN bound"));
                }
            }
            None => {
                if min_is_nan || (max_is_nan != (record.nan_count > 0)) {
                    return reject(format!(
                        "record {i} NaN marker disagrees with its NaN count"
                    ));
                }
            }
        }
        let comparable = rows > counted;
        if comparable && (record.min.is_null() || record.max.is_null()) {
            // The evaluators do not all treat missing bounds conservatively;
            // a scan recovers real bounds.
            return reject(format!("record {i} holds comparable values without bounds"));
        }
        if !comparable && (!record.min.is_null() || (!record.max.is_null() && !max_is_nan)) {
            return reject(format!("record {i} has bounds without comparable values"));
        }
    }
    if let Some(expected) = expected_rows
        && covered != expected
    {
        return reject(format!(
            "seed covers {covered} rows, the file holds {expected}"
        ));
    }
    let null_positions = if need_null_positions {
        let Some(bitmap) = null_positions else {
            return reject("seed carries no null positions".into());
        };
        if bitmap.len() != total_nulls {
            return reject(format!(
                "{} null positions for {total_nulls} counted nulls",
                bitmap.len()
            ));
        }
        if bitmap
            .max()
            .is_some_and(|offset| u64::from(offset) >= covered)
        {
            return reject("null position beyond the file".into());
        }
        Some(bitmap)
    } else {
        None
    };
    Ok(Some(ValidatedSeed {
        records,
        null_positions,
    }))
}

/// Fold `piece` (a whole seed record or a scanned part of the zone) into the
/// target zone `zone` under the version 1 contract: ordinary bounds widen,
/// counts add, and the NaN sign split adds when both sides know it (every
/// piece admitted by [`place_records`] or produced by a scan does). The
/// zone's bound is the target zone and is not touched.
fn fold_zone_stats(zone: &mut ZoneMapStatistics, piece: &ZoneMapStatistics) -> Result<()> {
    zone.null_count = zone
        .null_count
        .checked_add(piece.null_count)
        .ok_or_else(|| Error::invalid_input("null_count exceeds the supported UInt32 range"))?;
    zone.nan_count = zone
        .nan_count
        .checked_add(piece.nan_count)
        .ok_or_else(|| Error::invalid_input("nan_count exceeds the supported UInt32 range"))?;
    zone.negative_nan_count = match (zone.negative_nan_count, piece.negative_nan_count) {
        (Some(a), Some(b)) => Some(a.checked_add(b).ok_or_else(|| {
            Error::invalid_input("negative_nan_count exceeds the supported UInt32 range")
        })?),
        _ => None,
    };
    if !piece.min.is_null()
        && (zone.min.is_null() || piece.min.partial_cmp(&zone.min).is_some_and(|o| o.is_lt()))
    {
        zone.min = piece.min.clone();
    }
    if !piece.max.is_null()
        && (zone.max.is_null() || piece.max.partial_cmp(&zone.max).is_some_and(|o| o.is_gt()))
    {
        zone.max = piece.max.clone();
    }
    Ok(())
}

/// What a segment built by [`build_zonemap_segment`] starts from.
pub enum ZoneMapSegmentBase<'a> {
    /// A new index with these parameters.
    New(ZoneMapIndexBuilderParams),
    /// An update carrying this index's zones and configuration.
    Update(&'a ZoneMapIndex),
}

impl ZoneMapSegmentBase<'_> {
    pub fn rows_per_zone(&self) -> u64 {
        match self {
            Self::New(params) => params.rows_per_zone,
            Self::Update(index) => index.rows_per_zone,
        }
    }

    /// Whether the built segment keeps exact null positions: a new index
    /// configured with a bitmap, or an existing index that already has one.
    pub fn keeps_null_positions(&self) -> bool {
        match self {
            Self::New(params) => params.store_null_bitmap,
            Self::Update(index) => index.null_rows.is_some(),
        }
    }
}

/// Zone statistics produced by scanning rows, ready to complete the zones
/// of [`ReusedZones`]: a zone fed partly from seeds receives the scanned
/// rows' piece, a zone with no seed is the scanned piece.
#[derive(Debug, Default)]
pub struct ScannedZones {
    zones: Vec<ZoneMapStatistics>,
    null_rows: RowAddrTreeMap,
}

impl ScannedZones {
    /// Append another scan's pieces; the two cover disjoint rows.
    pub fn merge(&mut self, other: Self) {
        self.zones.extend(other.zones);
        self.null_rows |= &other.null_rows;
    }
}

/// Train zone pieces from `scan`, whose rows carry their addresses and are
/// cut into target zones of `rows_per_zone` aligned to each fragment's
/// `physical_rows`. One call may cover a single fragment or many.
pub async fn train_zonemap_scan(
    data_type: &DataType,
    rows_per_zone: u64,
    keep_positions: bool,
    physical_rows: HashMap<u64, u64>,
    scan: SendableRecordBatchStream,
) -> Result<ScannedZones> {
    let processor = ZoneMapProcessor::new(data_type.clone())?;
    let trainer = ZoneTrainer::new(processor, rows_per_zone)?
        .with_null_tracking(keep_positions)
        .aligned_to(physical_rows);
    let (zones, null_rows) = trainer.train(scan).await?;
    Ok(ScannedZones { zones, null_rows })
}

/// [`write_zonemap_segment`] after training `scan`, when there is one.
pub async fn build_zonemap_segment(
    base: ZoneMapSegmentBase<'_>,
    data_type: &DataType,
    reused: ReusedZones,
    scan: Option<SendableRecordBatchStream>,
    physical_rows: &HashMap<u64, u64>,
    refreshed_fragments: &[u64],
    dest_store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let scanned = match scan {
        Some(scan) => {
            let data_type = match &base {
                ZoneMapSegmentBase::Update(index) => &index.data_type,
                ZoneMapSegmentBase::New(_) => data_type,
            };
            train_zonemap_scan(
                data_type,
                base.rows_per_zone(),
                base.keeps_null_positions(),
                physical_rows.clone(),
                scan,
            )
            .await?
        }
        None => ScannedZones::default(),
    };
    write_zonemap_segment(
        base,
        data_type,
        reused,
        scanned,
        refreshed_fragments,
        dest_store,
    )
    .await
}

/// Write one ZoneMap segment from the target zones seeds supplied plus,
/// optionally, a scan of the rows they did not: whole fragments and the
/// uncovered intervals of partial ones. The scan is cut at the same physical
/// boundaries (`physical_rows` gives each scanned fragment's row count, so
/// the last zone's length is exact even with deleted trailing rows), and a
/// zone fed by both seeds and the scan is folded into one record.
///
/// `refreshed_fragments` are the fragments this build summarizes; on an
/// update, the base index's previous zones and null positions for them are
/// superseded (a column replacement retires the values they described).
/// When the base keeps exact null positions, `reused` must have been built
/// with `need_null_positions` and the scan collects positions too.
pub async fn write_zonemap_segment(
    base: ZoneMapSegmentBase<'_>,
    data_type: &DataType,
    reused: ReusedZones,
    scanned: ScannedZones,
    refreshed_fragments: &[u64],
    dest_store: &dyn IndexStore,
) -> Result<CreatedIndex> {
    let keep_positions = base.keeps_null_positions();
    let (existing, mut params) = match base {
        ZoneMapSegmentBase::New(params) => (None, params),
        ZoneMapSegmentBase::Update(index) => (Some(index), index.builder_params()),
    };
    // The output is a version 1 file; an older base cannot contribute its
    // zones to it (the dataset path rebuilds such a segment from its data).
    if let Some(index) = existing
        && index.version != ZoneMapIndexVersion::V1
    {
        return Err(Error::not_supported_source(
            format!(
                "cannot update a version {} ZoneMap segment in place; rebuild it from its \
                 data (optimize_indices) instead",
                index.version as u32
            )
            .into(),
        ));
    }
    let data_type = existing.map_or(data_type, |index| &index.data_type);
    params.use_seeds = Some(params.use_seeds_for(data_type));
    if keep_positions && !reused.with_null_positions && !reused.zones.is_empty() {
        return Err(Error::internal(
            "zone map seeds were reused without null positions for an index that keeps them",
        ));
    }

    let refreshed: HashSet<u64> = refreshed_fragments.iter().copied().collect();
    let mut zones: Vec<ZoneMapStatistics> = existing
        .map(|index| {
            index
                .zones
                .iter()
                .filter(|zone| !refreshed.contains(&zone.bound.fragment_id))
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    let mut null_rows = keep_positions.then(|| {
        let mut rows = existing
            .and_then(|index| index.null_rows.as_ref())
            .map(|rows| without_fragments(rows, refreshed.iter().copied()))
            .unwrap_or_default();
        rows |= &reused.null_rows;
        rows
    });
    let mut targets = reused.zones;
    for piece in scanned.zones {
        let key = (
            piece.bound.fragment_id,
            piece.bound.start / params.rows_per_zone,
        );
        match targets.get_mut(&key) {
            Some(zone) => fold_zone_stats(zone, &piece)?,
            None => {
                targets.insert(key, piece);
            }
        }
    }
    if let Some(rows) = null_rows.as_mut() {
        *rows |= &scanned.null_rows;
    }
    zones.extend(targets.into_values());
    zones.sort_by_key(|zone| (zone.bound.fragment_id, zone.bound.start));

    let rows_per_zone = params.rows_per_zone;
    let use_seeds = params.use_seeds.unwrap_or(false);
    let mut builder = ZoneMapIndexBuilder::try_new(params, data_type.clone())?;
    builder.maps = zones;
    builder.set_null_rows(null_rows);
    let has_null_bitmap = builder.null_rows.is_some();
    let files = builder.write_index(dest_store).await?;
    Ok(CreatedIndex {
        index_details: make_zone_map_index_details(rows_per_zone, use_seeds, has_null_bitmap),
        index_version: ZONEMAP_INDEX_VERSION,
        files,
    })
}

fn hex_encode(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    if s.is_empty() {
        return Ok(Vec::default());
    }
    if !s.len().is_multiple_of(2) {
        return Err(format!("odd hex length: {}", s.len()));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("invalid hex at {}: {}", i, e))
        })
        .collect()
}

#[cfg(test)]
mod tests;
