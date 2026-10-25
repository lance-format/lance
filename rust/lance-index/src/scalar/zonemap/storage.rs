// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Resident zone statistics. Common extrema borrow Arrow buffers and queries borrow
//! individual values; owned scalars are needed only by index maintenance and callers
//! returning a global value range.

use std::cmp::Ordering;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::{
    Array, ArrayRef, BooleanArray, GenericByteArray, PrimitiveArray, RecordBatch, UInt32Array,
    downcast_primitive_array,
};
use arrow_schema::DataType;
use datafusion_common::ScalarValue;
use itertools::Either;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};
use lance_select::RowAddrTreeMap;

use super::query::{
    BooleanBound, ByteBound, ByteKind, PrimitiveBound, PrimitiveKind, ZoneMapPredicate,
    ZoneMapValue, scalar_is_nan,
};
use super::{ValueRange, ZONEMAP_FILENAME, ZoneBound, ZoneMapIndexVersion, ZoneMapStatistics};
use crate::scalar::zoned::{collect_zone_matches, search_zones};
use crate::scalar::{MetricsCollector, SearchResult};

// These extrema/count arrays are flat. Slices retain their owner's full allocation,
// so different slice pointers must not charge the same owner again in another batch.
fn array_buffer_size(array: &dyn Array, context: &mut Context) -> usize {
    let data = array.to_data();
    let mut buffer_size = |buffer: &arrow::buffer::Buffer| {
        if context.mark_seen(buffer.data_ptr().as_ptr() as usize) {
            buffer.capacity()
        } else {
            0
        }
    };
    let mut size = data.buffers().iter().map(&mut buffer_size).sum();
    if let Some(nulls) = data.nulls() {
        size += buffer_size(nulls.inner().inner());
    }
    size
}

type OwnedValue = fn(&dyn Array, usize) -> ScalarValue;

#[derive(Debug)]
pub(super) struct Extrema {
    array: ArrayRef,
    // Selected with a checked Arrow type dispatch when the resident column is made.
    // This makes final range bounds infallible without losing errors from a decoder.
    owned: OwnedValue,
}

fn primitive_owned<T: PrimitiveKind>(array: &dyn Array, index: usize) -> ScalarValue {
    let array = array.as_primitive::<T>();
    T::owned(array, array.is_valid(index).then(|| array.value(index)))
}

fn primitive_converter<T: PrimitiveKind>(_array: &PrimitiveArray<T>) -> OwnedValue {
    primitive_owned::<T>
}

fn byte_owned<T: ByteKind>(array: &dyn Array, index: usize) -> ScalarValue {
    let array = array.as_bytes::<T>();
    T::owned(ByteBound::new(array, index).value)
}

fn boolean_owned(array: &dyn Array, index: usize) -> ScalarValue {
    let array = array.as_boolean();
    ScalarValue::Boolean(array.is_valid(index).then(|| array.value(index)))
}

impl Extrema {
    fn from_array(array: &ArrayRef) -> Option<Self> {
        // These types have no extrema in current ZoneMap writers. Keep their
        // original scalar representation along with nested/other legacy types.
        if matches!(
            array.data_type(),
            DataType::Interval(_) | DataType::Decimal32(_, _) | DataType::Decimal64(_, _)
        ) {
            return None;
        }
        let owned = downcast_primitive_array!(array => { Some(primitive_converter(array)) }
            DataType::Boolean => Some(boolean_owned as OwnedValue),
            DataType::Utf8 => Some(byte_owned::<GenericStringType<i32>> as OwnedValue),
            DataType::LargeUtf8 => Some(byte_owned::<GenericStringType<i64>> as OwnedValue),
            DataType::Binary => Some(byte_owned::<GenericBinaryType<i32>> as OwnedValue),
            DataType::LargeBinary => Some(byte_owned::<GenericBinaryType<i64>> as OwnedValue),
            _ => None,
        )?;
        Some(Self {
            array: array.clone(),
            owned,
        })
    }

    #[inline]
    fn value(&self, index: usize) -> ZoneValue<'_> {
        ZoneValue::Array(self, index)
    }
}

impl DeepSizeOf for Extrema {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        if !context.mark_seen(Arc::as_ptr(&self.array) as *const () as usize) {
            return 0;
        }
        let timezone = match self.array.data_type() {
            DataType::Timestamp(_, timezone) => timezone.deep_size_of_children(context),
            _ => 0,
        };
        std::mem::size_of_val(self.array.as_ref())
            + array_buffer_size(self.array.as_ref(), context)
            + timezone
    }
}

#[derive(Debug)]
pub(super) enum ZoneMaps {
    Generic(Vec<ZoneMapStatistics>),
    Columnar(Vec<ColumnarZones>),
}

#[derive(Debug)]
pub(super) struct ColumnarZones {
    min: Extrema,
    max: Extrema,
    null_counts: UInt32Array,
    nan_counts: UInt32Array,
    /// The NaN sign split, stored by version 1 files only. A version 0 zone's
    /// split is unknown unless it has no NaNs (see [`Self::negative_nan_count`]).
    negative_nan_counts: Option<UInt32Array>,
    bounds: Vec<ZoneBound>,
}

impl From<Vec<ZoneMapStatistics>> for ZoneMaps {
    fn from(zones: Vec<ZoneMapStatistics>) -> Self {
        Self::Generic(zones)
    }
}

impl ZoneMaps {
    /// Decode one batch of the index file under the contract of `version`.
    /// A version 1 batch must carry the non-null `negative_nan_count` column
    /// and every zone must satisfy the version 1 invariants.
    pub(super) fn from_batch(
        data: &RecordBatch,
        version: ZoneMapIndexVersion,
    ) -> Result<(Self, DataType)> {
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
                    .downcast_ref::<UInt32Array>()
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
            let zones = if Extrema::from_array(min_col).is_some()
                && Extrema::from_array(max_col).is_some()
            {
                Self::Columnar(Vec::new())
            } else {
                Self::Generic(Vec::new())
            };
            return Ok((zones, data_type));
        }

        let num_zones = data.num_rows();
        let bounds = || {
            (0..num_zones)
                .map(|i| ZoneBound {
                    fragment_id: fragment_id_col.value(i),
                    start: zone_start_col.value(i),
                    length: zone_length.value(i) as usize,
                })
                .collect()
        };
        let zones = match Self::from_arrays(
            min_col,
            max_col,
            null_count_col,
            nan_count_col,
            negative_nan_count_col,
            bounds,
        ) {
            Some(zones) => zones,
            None => Self::Generic(Self::generic_zones(
                data,
                min_col,
                max_col,
                null_count_col,
                nan_count_col,
                negative_nan_count_col,
                zone_length,
                fragment_id_col,
                zone_start_col,
            )?),
        };
        if version == ZoneMapIndexVersion::V1 {
            zones.check_v1_invariants(data_type.is_nested())?;
        }
        Ok((zones, data_type))
    }

    /// Every zone of a declared version 1 batch must satisfy the version 1
    /// contract; a violation is a corrupt file, not a conservative zone.
    fn check_v1_invariants(&self, nested: bool) -> Result<()> {
        for (i, zone) in self.iter().enumerate() {
            let negative = zone.negative_nan_count.ok_or_else(|| {
                Error::internal("a version 1 zone was decoded without its sign split")
            })?;
            if let Some(violation) = super::v1_zone_violation(
                &zone.min,
                &zone.max,
                zone.null_count,
                zone.nan_count,
                negative,
                zone.bound.length as u64,
                nested,
            ) {
                return Err(Error::corrupt_file_named(
                    ZONEMAP_FILENAME,
                    format!("ZoneMapIndex: version 1 zone {i} has {violation}"),
                ));
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn generic_zones(
        data: &RecordBatch,
        min_col: &ArrayRef,
        max_col: &ArrayRef,
        null_count_col: &UInt32Array,
        nan_count_col: &UInt32Array,
        negative_nan_count_col: Option<&UInt32Array>,
        zone_length: &arrow_array::UInt64Array,
        fragment_id_col: &arrow_array::UInt64Array,
        zone_start_col: &arrow_array::UInt64Array,
    ) -> Result<Vec<ZoneMapStatistics>> {
        let data_type = min_col.data_type().clone();
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
            zones.push(ZoneMapStatistics {
                min,
                max,
                null_count,
                nan_count,
                negative_nan_count: negative_nan_count(negative_nan_count_col, nan_count_col, i),
                bound: ZoneBound {
                    fragment_id: fragment_id_col.value(i),
                    start: zone_start_col.value(i),
                    length: zone_length.value(i) as usize,
                },
            });
        }

        Ok(zones)
    }

    pub(super) fn from_arrays(
        min: &ArrayRef,
        max: &ArrayRef,
        null_counts: &UInt32Array,
        nan_counts: &UInt32Array,
        negative_nan_counts: Option<&UInt32Array>,
        bounds: impl FnOnce() -> Vec<ZoneBound>,
    ) -> Option<Self> {
        Some(Self::Columnar(vec![ColumnarZones {
            min: Extrema::from_array(min)?,
            max: Extrema::from_array(max)?,
            null_counts: null_counts.clone(),
            nan_counts: nan_counts.clone(),
            negative_nan_counts: negative_nan_counts.cloned(),
            bounds: bounds(),
        }]))
    }

    pub(super) fn append(&mut self, other: Self) -> Result<()> {
        if matches!(&other, Self::Generic(zones) if zones.is_empty()) {
            return Ok(());
        }
        match (self, other) {
            (Self::Generic(zones), Self::Generic(mut batch)) => zones.append(&mut batch),
            (Self::Columnar(chunks), Self::Columnar(mut batch)) => chunks.append(&mut batch),
            _ => {
                return Err(Error::internal(
                    "ZoneMapIndex: inconsistent batch representation for one file schema",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn reserve(&mut self, summaries: usize, batch_size: usize) {
        match self {
            Self::Generic(zones) => zones.reserve(summaries),
            Self::Columnar(chunks) => chunks.reserve(summaries.div_ceil(batch_size)),
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Generic(zones) => zones.len(),
            Self::Columnar(chunks) => chunks.iter().map(|chunk| chunk.bounds.len()).sum(),
        }
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = ZoneMapStatistics<ZoneValue<'_>>> {
        match self {
            Self::Generic(zones) => Either::Left(zones.iter().map(ZoneMapStatistics::as_view)),
            Self::Columnar(chunks) => Either::Right(chunks.iter().flat_map(ColumnarZones::iter)),
        }
    }

    pub(super) fn extend_value_range<'a>(
        &'a self,
        range: &mut ValueRange<ZoneValue<'a>>,
    ) -> Option<()> {
        match self {
            Self::Generic(zones) => {
                // A legacy Vec can mix scalar variants. Its zones must be folded
                // against the current global bounds, not a separate local range.
                for zone in zones {
                    range.extend_zone(zone.as_view())?;
                }
            }
            Self::Columnar(chunks) => {
                for chunk in chunks {
                    let local = chunk.value_range()?;
                    range.extend_bounds(local.min, local.max);
                }
            }
        }
        Some(())
    }

    pub(super) fn search(
        &self,
        predicate: &impl ZoneMapPredicate,
        metrics: &dyn MetricsCollector,
    ) -> Result<SearchResult> {
        match self {
            Self::Generic(zones) => search_zones(zones, metrics, |zone| predicate.evaluate(zone)),
            Self::Columnar(chunks) => {
                metrics.record_comparisons(self.len());
                let mut row_addrs = RowAddrTreeMap::new();
                for chunk in chunks {
                    chunk.search(predicate, &mut row_addrs)?;
                }
                Ok(SearchResult::at_most(row_addrs))
            }
        }
    }

    pub(super) fn to_vec(&self) -> Vec<ZoneMapStatistics> {
        let mut zones = Vec::with_capacity(self.len());
        zones.extend(self.iter().map(|zone| zone.into_owned()));
        zones
    }

    #[cfg(test)]
    pub(super) fn batch_lengths(&self) -> Vec<usize> {
        match self {
            Self::Generic(zones) => vec![zones.len()],
            Self::Columnar(chunks) => chunks.iter().map(|chunk| chunk.bounds.len()).collect(),
        }
    }

    #[cfg(test)]
    pub(super) fn get(&self, index: usize) -> ZoneMapStatistics {
        self.iter().nth(index).unwrap().into_owned()
    }

    #[cfg(test)]
    pub(super) fn as_generic_mut(&mut self) -> &mut Vec<ZoneMapStatistics> {
        if !matches!(self, Self::Generic(_)) {
            *self = Self::Generic(self.to_vec());
        }
        match self {
            Self::Generic(zones) => zones,
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
impl PartialEq<Vec<ZoneMapStatistics>> for ZoneMaps {
    fn eq(&self, other: &Vec<ZoneMapStatistics>) -> bool {
        self.to_vec() == *other
    }
}

impl DeepSizeOf for ZoneMaps {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            Self::Generic(zones) => zones.deep_size_of_children(context),
            Self::Columnar(zones) => zones.deep_size_of_children(context),
        }
    }
}

/// The NaN sign split of zone `index`: the stored version 1 value, else
/// provably zero for a zone without NaNs, else unknown.
fn negative_nan_count(
    negative_nan_counts: Option<&UInt32Array>,
    nan_counts: &UInt32Array,
    index: usize,
) -> Option<u32> {
    match negative_nan_counts {
        Some(counts) => Some(counts.value(index)),
        None if nan_counts.value(index) == 0 => Some(0),
        None => None,
    }
}

impl ColumnarZones {
    #[inline]
    fn negative_nan_count(&self, index: usize) -> Option<u32> {
        negative_nan_count(self.negative_nan_counts.as_ref(), &self.nan_counts, index)
    }

    fn iter(&self) -> impl ExactSizeIterator<Item = ZoneMapStatistics<ZoneValue<'_>>> {
        (0..self.bounds.len()).map(|index| ZoneMapStatistics {
            min: self.min.value(index),
            max: self.max.value(index),
            null_count: self.null_counts.value(index),
            nan_count: self.nan_counts.value(index),
            negative_nan_count: self.negative_nan_count(index),
            bound: self.bounds[index].clone(),
        })
    }

    fn search(
        &self,
        predicate: &impl ZoneMapPredicate,
        row_addrs: &mut RowAddrTreeMap,
    ) -> Result<()> {
        let min = self.min.array.as_ref();
        let max = self.max.array.as_ref();
        downcast_primitive_array!(min, max => {
            self.search_primitive(min, max, predicate, row_addrs)
        }
            (DataType::Boolean, DataType::Boolean) => self.search_booleans(min.as_boolean(), max.as_boolean(), predicate, row_addrs),
            (DataType::Utf8, DataType::Utf8) => self.search_bytes(min.as_bytes::<GenericStringType<i32>>(), max.as_bytes::<GenericStringType<i32>>(), predicate, row_addrs),
            (DataType::LargeUtf8, DataType::LargeUtf8) => self.search_bytes(min.as_bytes::<GenericStringType<i64>>(), max.as_bytes::<GenericStringType<i64>>(), predicate, row_addrs),
            (DataType::Binary, DataType::Binary) => self.search_bytes(min.as_bytes::<GenericBinaryType<i32>>(), max.as_bytes::<GenericBinaryType<i32>>(), predicate, row_addrs),
            (DataType::LargeBinary, DataType::LargeBinary) => self.search_bytes(min.as_bytes::<GenericBinaryType<i64>>(), max.as_bytes::<GenericBinaryType<i64>>(), predicate, row_addrs),
            // Older files need not have the same min/max types. Preserve their
            // ScalarValue comparison behavior instead of coercing either bound.
            _ => collect_zone_matches(self.iter(), row_addrs, |zone| predicate.evaluate(zone)),
        )
    }

    fn value_range(&self) -> Option<ValueRange<ZoneValue<'_>>> {
        let min = self.min.array.as_ref();
        let max = self.max.array.as_ref();
        downcast_primitive_array!(min, max => {
            self.range_values(
                move |index| PrimitiveBound::new(min, index),
                move |index| PrimitiveBound::new(max, index),
            )
        }
            (DataType::Boolean, DataType::Boolean) => self.range_values(
                move |index| BooleanBound(nullable_boolean(min.as_boolean(), index)),
                move |index| BooleanBound(nullable_boolean(max.as_boolean(), index)),
            ),
            (DataType::Utf8, DataType::Utf8) => self.range_bytes(min.as_bytes::<GenericStringType<i32>>(), max.as_bytes::<GenericStringType<i32>>()),
            (DataType::LargeUtf8, DataType::LargeUtf8) => self.range_bytes(min.as_bytes::<GenericStringType<i64>>(), max.as_bytes::<GenericStringType<i64>>()),
            (DataType::Binary, DataType::Binary) => self.range_bytes(min.as_bytes::<GenericBinaryType<i32>>(), max.as_bytes::<GenericBinaryType<i32>>()),
            (DataType::LargeBinary, DataType::LargeBinary) => self.range_bytes(min.as_bytes::<GenericBinaryType<i64>>(), max.as_bytes::<GenericBinaryType<i64>>()),
            _ => self.range_indices(self.iter()),
        )
    }

    fn range_bytes<T: ByteKind>(
        &self,
        min: &GenericByteArray<T>,
        max: &GenericByteArray<T>,
    ) -> Option<ValueRange<ZoneValue<'_>>> {
        self.range_values(
            move |index| ByteBound::new(min, index),
            move |index| ByteBound::new(max, index),
        )
    }

    fn range_values<V: ZoneMapValue + PartialOrd>(
        &self,
        min: impl Fn(usize) -> V,
        max: impl Fn(usize) -> V,
    ) -> Option<ValueRange<ZoneValue<'_>>> {
        self.range_indices(self.iter_values(min, max))
    }

    fn range_indices<V: ZoneMapValue + PartialOrd>(
        &self,
        zones: impl Iterator<Item = ZoneMapStatistics<V>>,
    ) -> Option<ValueRange<ZoneValue<'_>>> {
        let mut range = ValueRange::default();
        let mut min_index = None;
        let mut max_index = None;
        for (index, zone) in zones.enumerate() {
            let (min_changed, max_changed) = range.extend_zone(zone)?;
            if min_changed {
                min_index = Some(index);
            }
            if max_changed {
                max_index = Some(index);
            }
        }
        // Each Arrow column has one scalar kind and comparison metadata, so
        // only its winners need the general cross-column comparison path.
        Some(ValueRange {
            min: min_index.map(|index| self.min.value(index)),
            max: max_index.map(|index| self.max.value(index)),
        })
    }

    fn search_primitive<T: PrimitiveKind>(
        &self,
        min: &PrimitiveArray<T>,
        max: &PrimitiveArray<T>,
        predicate: &impl ZoneMapPredicate,
        row_addrs: &mut RowAddrTreeMap,
    ) -> Result<()> {
        self.search_values(
            move |index| PrimitiveBound::new(min, index),
            move |index| PrimitiveBound::new(max, index),
            predicate,
            row_addrs,
        )
    }

    fn search_bytes<T: ByteKind>(
        &self,
        min: &GenericByteArray<T>,
        max: &GenericByteArray<T>,
        predicate: &impl ZoneMapPredicate,
        row_addrs: &mut RowAddrTreeMap,
    ) -> Result<()> {
        self.search_values(
            move |index| ByteBound::new(min, index),
            move |index| ByteBound::new(max, index),
            predicate,
            row_addrs,
        )
    }

    fn search_booleans(
        &self,
        min: &BooleanArray,
        max: &BooleanArray,
        predicate: &impl ZoneMapPredicate,
        row_addrs: &mut RowAddrTreeMap,
    ) -> Result<()> {
        self.search_values(
            move |index| BooleanBound(min.is_valid(index).then(|| min.value(index))),
            move |index| BooleanBound(max.is_valid(index).then(|| max.value(index))),
            predicate,
            row_addrs,
        )
    }

    fn search_values<V: ZoneMapValue>(
        &self,
        min: impl Fn(usize) -> V,
        max: impl Fn(usize) -> V,
        predicate: &impl ZoneMapPredicate,
        row_addrs: &mut RowAddrTreeMap,
    ) -> Result<()>
    where
        ScalarValue: PartialOrd<V>,
    {
        collect_zone_matches(self.iter_values(min, max), row_addrs, |zone| {
            predicate.evaluate(zone)
        })
    }

    fn iter_values<V>(
        &self,
        min: impl Fn(usize) -> V,
        max: impl Fn(usize) -> V,
    ) -> impl ExactSizeIterator<Item = ZoneMapStatistics<V>> {
        (0..self.bounds.len()).map(move |index| ZoneMapStatistics {
            min: min(index),
            max: max(index),
            null_count: self.null_counts.value(index),
            nan_count: self.nan_counts.value(index),
            negative_nan_count: self.negative_nan_count(index),
            bound: self.bounds[index].clone(),
        })
    }
}

impl DeepSizeOf for ColumnarZones {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.min.deep_size_of_children(context)
            + self.max.deep_size_of_children(context)
            + array_buffer_size(&self.null_counts, context)
            + array_buffer_size(&self.nan_counts, context)
            + self
                .negative_nan_counts
                .as_ref()
                .map_or(0, |counts| array_buffer_size(counts, context))
            + self.bounds.capacity() * std::mem::size_of::<ZoneBound>()
    }
}

impl ZoneMapStatistics {
    #[inline]
    pub(super) fn as_view(&self) -> ZoneMapStatistics<ZoneValue<'_>> {
        ZoneMapStatistics {
            min: ZoneValue::Scalar(&self.min),
            max: ZoneValue::Scalar(&self.max),
            null_count: self.null_count,
            nan_count: self.nan_count,
            negative_nan_count: self.negative_nan_count,
            bound: self.bound.clone(),
        }
    }
}

impl ZoneMapStatistics<ZoneValue<'_>> {
    pub(super) fn into_owned(self) -> ZoneMapStatistics {
        ZoneMapStatistics {
            min: self.min.into_owned(),
            max: self.max.into_owned(),
            null_count: self.null_count,
            nan_count: self.nan_count,
            negative_nan_count: self.negative_nan_count,
            bound: self.bound,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ZoneValue<'a> {
    Scalar(&'a ScalarValue),
    Array(&'a Extrema, usize),
}

impl ZoneMapValue for ZoneValue<'_> {
    #[inline]
    fn is_null(&self) -> bool {
        match self {
            Self::Scalar(value) => value.is_null(),
            Self::Array(values, index) => values.array.is_null(*index),
        }
    }

    #[inline]
    fn is_nan(&self) -> bool {
        match self {
            Self::Scalar(value) => scalar_is_nan(value),
            Self::Array(values, index) => {
                let array = values.array.as_ref();
                match array.data_type() {
                    DataType::Float16 => {
                        PrimitiveBound::new(array.as_primitive::<Float16Type>(), *index).is_nan()
                    }
                    DataType::Float32 => {
                        PrimitiveBound::new(array.as_primitive::<Float32Type>(), *index).is_nan()
                    }
                    DataType::Float64 => {
                        PrimitiveBound::new(array.as_primitive::<Float64Type>(), *index).is_nan()
                    }
                    _ => false,
                }
            }
        }
    }
}

impl ZoneValue<'_> {
    pub(super) fn into_owned(self) -> ScalarValue {
        match self {
            Self::Scalar(value) => value.clone(),
            Self::Array(values, index) => (values.owned)(values.array.as_ref(), index),
        }
    }
}

impl PartialEq<ScalarValue> for ZoneValue<'_> {
    fn eq(&self, other: &ScalarValue) -> bool {
        match self {
            Self::Scalar(value) => (*value).eq(other),
            Self::Array(values, index) => {
                let array = values.array.as_ref();
                downcast_primitive_array!(array => { PrimitiveBound::new(array, *index).eq(other) }
                    DataType::Boolean => BooleanBound(array.as_boolean().is_valid(*index).then(|| array.as_boolean().value(*index))).eq(other),
                    DataType::Utf8 => ByteBound::new(array.as_bytes::<GenericStringType<i32>>(), *index).eq(other),
                    DataType::LargeUtf8 => ByteBound::new(array.as_bytes::<GenericStringType<i64>>(), *index).eq(other),
                    DataType::Binary => ByteBound::new(array.as_bytes::<GenericBinaryType<i32>>(), *index).eq(other),
                    DataType::LargeBinary => ByteBound::new(array.as_bytes::<GenericBinaryType<i64>>(), *index).eq(other),
                    _ => unreachable!("resident extrema were validated at construction"),
                )
            }
        }
    }
}

impl PartialOrd<ScalarValue> for ZoneValue<'_> {
    #[inline]
    fn partial_cmp(&self, other: &ScalarValue) -> Option<Ordering> {
        match self {
            Self::Scalar(value) => (*value).partial_cmp(other),
            Self::Array(values, index) => {
                let array = values.array.as_ref();
                downcast_primitive_array!(array => { PrimitiveBound::new(array, *index).partial_cmp(other) }
                    DataType::Boolean => BooleanBound(array.as_boolean().is_valid(*index).then(|| array.as_boolean().value(*index))).partial_cmp(other),
                    DataType::Utf8 => ByteBound::new(array.as_bytes::<GenericStringType<i32>>(), *index).partial_cmp(other),
                    DataType::LargeUtf8 => ByteBound::new(array.as_bytes::<GenericStringType<i64>>(), *index).partial_cmp(other),
                    DataType::Binary => ByteBound::new(array.as_bytes::<GenericBinaryType<i32>>(), *index).partial_cmp(other),
                    DataType::LargeBinary => ByteBound::new(array.as_bytes::<GenericBinaryType<i64>>(), *index).partial_cmp(other),
                    _ => unreachable!("resident extrema were validated at construction"),
                )
            }
        }
    }
}

impl PartialEq<ZoneValue<'_>> for ScalarValue {
    fn eq(&self, other: &ZoneValue<'_>) -> bool {
        other.eq(self)
    }
}

impl PartialOrd<ZoneValue<'_>> for ScalarValue {
    #[inline]
    fn partial_cmp(&self, other: &ZoneValue<'_>) -> Option<Ordering> {
        other.partial_cmp(self).map(Ordering::reverse)
    }
}

impl PartialEq for ZoneValue<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (_, Self::Scalar(value)) => self.eq(*value),
            (Self::Scalar(value), other) => other.eq(*value),
            (Self::Array(left, index), Self::Array(right, other_index)) => {
                let left = left.array.as_ref();
                let right = right.array.as_ref();
                downcast_primitive_array!(left, right => { PrimitiveBound::new(left, *index).equals(&PrimitiveBound::new(right, *other_index)) }
                    (DataType::Boolean, DataType::Boolean) => nullable_boolean(left.as_boolean(), *index) == nullable_boolean(right.as_boolean(), *other_index),
                    (DataType::Utf8, DataType::Utf8) => byte_values_equal::<GenericStringType<i32>>(left, *index, right, *other_index),
                    (DataType::LargeUtf8, DataType::LargeUtf8) => byte_values_equal::<GenericStringType<i64>>(left, *index, right, *other_index),
                    (DataType::Binary, DataType::Binary) => byte_values_equal::<GenericBinaryType<i32>>(left, *index, right, *other_index),
                    (DataType::LargeBinary, DataType::LargeBinary) => byte_values_equal::<GenericBinaryType<i64>>(left, *index, right, *other_index),
                    _ => false,
                )
            }
        }
    }
}

impl PartialOrd for ZoneValue<'_> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (_, Self::Scalar(other)) => self.partial_cmp(*other),
            (Self::Scalar(value), other) => other.partial_cmp(*value).map(Ordering::reverse),
            (Self::Array(left, index), Self::Array(right, other_index)) => {
                let left = left.array.as_ref();
                let right = right.array.as_ref();
                downcast_primitive_array!(left, right => { PrimitiveBound::new(left, *index).compare(&PrimitiveBound::new(right, *other_index)) }
                    (DataType::Boolean, DataType::Boolean) => nullable_boolean(left.as_boolean(), *index).partial_cmp(&nullable_boolean(right.as_boolean(), *other_index)),
                    (DataType::Utf8, DataType::Utf8) => compare_byte_values::<GenericStringType<i32>>(left, *index, right, *other_index),
                    (DataType::LargeUtf8, DataType::LargeUtf8) => compare_byte_values::<GenericStringType<i64>>(left, *index, right, *other_index),
                    (DataType::Binary, DataType::Binary) => compare_byte_values::<GenericBinaryType<i32>>(left, *index, right, *other_index),
                    (DataType::LargeBinary, DataType::LargeBinary) => compare_byte_values::<GenericBinaryType<i64>>(left, *index, right, *other_index),
                    _ => None,
                )
            }
        }
    }
}

fn nullable_boolean(array: &BooleanArray, index: usize) -> Option<bool> {
    array.is_valid(index).then(|| array.value(index))
}

fn byte_values_equal<T: ByteKind>(
    left: &dyn Array,
    index: usize,
    right: &dyn Array,
    other_index: usize,
) -> bool {
    ByteBound::new(left.as_bytes::<T>(), index).value
        == ByteBound::new(right.as_bytes::<T>(), other_index).value
}

fn compare_byte_values<T: ByteKind>(
    left: &dyn Array,
    index: usize,
    right: &dyn Array,
    other_index: usize,
) -> Option<Ordering> {
    ByteBound::new(left.as_bytes::<T>(), index)
        .value
        .partial_cmp(&ByteBound::new(right.as_bytes::<T>(), other_index).value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int64Array, StringArray};

    fn range_over_groups<'a>(
        groups: impl IntoIterator<Item = &'a ZoneMaps>,
    ) -> Option<(ScalarValue, ScalarValue)> {
        let mut range = ValueRange::default();
        for group in groups {
            group.extend_value_range(&mut range)?;
        }
        Some((range.min?.into_owned(), range.max?.into_owned()))
    }

    fn scalar_zone(min: ScalarValue, max: ScalarValue, index: usize) -> ZoneMapStatistics {
        ZoneMapStatistics {
            null_count: if min.is_null() && max.is_null() {
                16
            } else {
                0
            },
            nan_count: if scalar_is_nan(&min) || scalar_is_nan(&max) {
                16
            } else {
                0
            },
            negative_nan_count: Some(0),
            min,
            max,
            bound: ZoneBound {
                fragment_id: index as u64,
                start: 0,
                length: 16,
            },
        }
    }

    #[test]
    fn legacy_mixed_ranges_keep_global_comparison_context() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![5]));
        let max: ArrayRef = Arc::new(Int64Array::from(vec![15]));
        let counts = UInt32Array::from(vec![0]);
        let numeric = ZoneMaps::from_arrays(&values, &max, &counts, &counts, None, || {
            vec![ZoneBound {
                fragment_id: 0,
                start: 0,
                length: 16,
            }]
        })
        .unwrap();
        let mixed: ZoneMaps = vec![
            scalar_zone(
                ScalarValue::Utf8(Some("z".into())),
                ScalarValue::Utf8(Some("zz".into())),
                1,
            ),
            scalar_zone(ScalarValue::Int64(Some(0)), ScalarValue::Int64(Some(20)), 2),
        ]
        .into();
        assert_eq!(
            range_over_groups([&numeric, &mixed]),
            Some((ScalarValue::Int64(Some(0)), ScalarValue::Int64(Some(20))))
        );
        // Preserve the existing first-compatible-kind behavior, including its
        // asymmetric result for incomparable legacy scalar variants.
        assert_eq!(
            range_over_groups([&mixed, &numeric]),
            Some((
                ScalarValue::Utf8(Some("z".into())),
                ScalarValue::Utf8(Some("zz".into()))
            ))
        );
    }

    fn float_values(data_type: &DataType) -> Vec<ScalarValue> {
        match data_type {
            DataType::Float16 => [
                0, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e01, 0x7e02, 0xfe01,
            ]
            .into_iter()
            .map(|bits| ScalarValue::Float16(Some(half::f16::from_bits(bits))))
            .chain([ScalarValue::Float16(None)])
            .collect(),
            DataType::Float32 => [
                0,
                0x8000_0000,
                0x3f80_0000,
                0xbf80_0000,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0001,
                0x7fc0_0002,
                0xffc0_0001,
            ]
            .into_iter()
            .map(|bits| ScalarValue::Float32(Some(f32::from_bits(bits))))
            .chain([ScalarValue::Float32(None)])
            .collect(),
            DataType::Float64 => [
                0,
                0x8000_0000_0000_0000,
                0x3ff0_0000_0000_0000,
                0xbff0_0000_0000_0000,
                0x7ff0_0000_0000_0000,
                0xfff0_0000_0000_0000,
                0x7ff8_0000_0000_0001,
                0x7ff8_0000_0000_0002,
                0xfff8_0000_0000_0001,
            ]
            .into_iter()
            .map(|bits| ScalarValue::Float64(Some(f64::from_bits(bits))))
            .chain([ScalarValue::Float64(None)])
            .collect(),
            _ => unreachable!(),
        }
    }

    #[rstest::rstest]
    #[case::float16(float_values(&DataType::Float16))]
    #[case::float32(float_values(&DataType::Float32))]
    #[case::float64(float_values(&DataType::Float64))]
    #[case::uint64(vec![
        ScalarValue::UInt64(Some(0)), ScalarValue::UInt64(Some(i64::MAX as u64)),
        ScalarValue::UInt64(Some(i64::MAX as u64 + 1)), ScalarValue::UInt64(Some(u64::MAX)),
        ScalarValue::UInt64(None),
    ])]
    #[case::decimal128(vec![
        ScalarValue::Decimal128(Some(-1), 38, 2), ScalarValue::Decimal128(Some(0), 38, 2),
        ScalarValue::Decimal128(Some(1), 38, 2), ScalarValue::Decimal128(Some(1), 37, 2),
        ScalarValue::Decimal128(Some(1), 38, 3), ScalarValue::Decimal128(Some(1), 38, -2),
        ScalarValue::Decimal128(None, 38, 2), ScalarValue::Decimal128(None, 37, 2),
        ScalarValue::Decimal128(None, 38, 3),
    ])]
    #[case::decimal256(vec![
        ScalarValue::Decimal256(Some(arrow::datatypes::i256::from_i128(-1)), 76, 2),
        ScalarValue::Decimal256(Some(arrow::datatypes::i256::from_i128(1)), 76, 2),
        ScalarValue::Decimal256(Some(arrow::datatypes::i256::from_i128(1)), 75, 2),
        ScalarValue::Decimal256(Some(arrow::datatypes::i256::from_i128(1)), 76, 3),
        ScalarValue::Decimal256(Some(arrow::datatypes::i256::from_parts(7, 1)), 76, 2),
        ScalarValue::Decimal256(None, 76, 2), ScalarValue::Decimal256(None, 75, 2),
        ScalarValue::Decimal256(None, 76, 3),
    ])]
    #[case::timestamp(vec![
        ScalarValue::TimestampMicrosecond(Some(1), Some("UTC".into())),
        ScalarValue::TimestampMicrosecond(Some(1), Some("America/New_York".into())),
        ScalarValue::TimestampMicrosecond(Some(1), None),
        ScalarValue::TimestampMicrosecond(None, Some("UTC".into())),
        ScalarValue::TimestampMicrosecond(None, Some("America/New_York".into())),
        ScalarValue::TimestampMillisecond(Some(1), Some("UTC".into())),
        ScalarValue::TimestampSecond(None, Some("UTC".into())),
    ])]
    fn native_bounds_preserve_scalar_equality_and_partial_order(#[case] values: Vec<ScalarValue>) {
        let extrema = values
            .iter()
            .map(|value| Extrema::from_array(&value.to_array_of_size(1).unwrap()).unwrap())
            .collect::<Vec<_>>();
        let mut targets = values.clone();
        targets.extend([
            ScalarValue::Null,
            ScalarValue::Int64(Some(1)),
            ScalarValue::Int64(None),
            ScalarValue::Utf8(Some("1".into())),
        ]);
        for (index, value) in values.iter().enumerate() {
            let bound = extrema[index].value(0);
            assert_eq!(bound.into_owned(), *value);
            assert_eq!(bound.into_owned().data_type(), value.data_type());
            for target in &targets {
                assert_eq!(
                    bound.eq(target),
                    value.eq(target),
                    "{value:?} == {target:?}"
                );
                assert_eq!(target.eq(&bound), target.eq(value));
                assert_eq!(
                    bound.partial_cmp(target),
                    value.partial_cmp(target),
                    "{value:?} <=> {target:?}"
                );
                assert_eq!(target.partial_cmp(&bound), target.partial_cmp(value));
            }
            for (other_index, other) in values.iter().enumerate() {
                let other_bound = extrema[other_index].value(0);
                assert_eq!(bound.eq(&other_bound), value.eq(other));
                assert_eq!(bound.partial_cmp(&other_bound), value.partial_cmp(other));
            }
        }

        for exclude_nans in [false, true] {
            let mut compact = ZoneMaps::Columnar(Vec::new());
            let mut generic = Vec::new();
            for (index, value) in values
                .iter()
                .enumerate()
                .filter(|(_, value)| !exclude_nans || !scalar_is_nan(value))
            {
                let zone = scalar_zone(value.clone(), value.clone(), index);
                let array = value.to_array_of_size(1).unwrap();
                compact
                    .append(
                        ZoneMaps::from_arrays(
                            &array,
                            &array,
                            &UInt32Array::from(vec![zone.null_count]),
                            &UInt32Array::from(vec![zone.nan_count]),
                            zone.negative_nan_count
                                .map(|negative| UInt32Array::from(vec![negative]))
                                .as_ref(),
                            || vec![zone.bound.clone()],
                        )
                        .unwrap(),
                    )
                    .unwrap();
                generic.push(zone);
            }
            let generic: ZoneMaps = generic.into();
            let actual = range_over_groups([&compact]);
            let expected = range_over_groups([&generic]);
            assert_eq!(actual, expected);
            if let (Some(actual), Some(expected)) = (actual, expected) {
                assert_eq!(actual.0.data_type(), expected.0.data_type());
                assert_eq!(actual.1.data_type(), expected.1.data_type());
            }
        }
    }

    #[test]
    fn sliced_shared_buffers_are_counted_once_at_full_capacity() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(7); 4096]).slice(10, 128));
        let counts = UInt32Array::from(vec![0; 4096]).slice(10, 128);
        let bounds = (0..128)
            .map(|i| ZoneBound {
                fragment_id: i,
                start: 0,
                length: 8192,
            })
            .collect::<Vec<_>>();
        let bound_capacity = bounds.capacity() * std::mem::size_of::<ZoneBound>();
        let zones =
            ZoneMaps::from_arrays(&values, &values, &counts, &counts, None, || bounds).unwrap();
        let mut context = Context::new();
        let expected = std::mem::size_of::<ColumnarZones>()
            + bound_capacity
            + std::mem::size_of_val(values.as_ref())
            + values.as_ref().deep_size_of_children(&mut context)
            + (&counts as &dyn Array).deep_size_of_children(&mut context);
        assert_eq!(zones.deep_size_of_children(&mut Context::new()), expected);
        assert!(expected >= 4096 * (8 + 4) + bound_capacity);
    }

    #[test]
    fn shared_buffers_across_batches_are_counted_once() {
        let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(7); 4096]));
        let counts = UInt32Array::from(vec![0; 4096]);
        let mut zones = ZoneMaps::from_arrays(
            &values.slice(10, 128),
            &values.slice(10, 128),
            &counts.slice(10, 128),
            &counts.slice(10, 128),
            None,
            || {
                vec![
                    ZoneBound {
                        fragment_id: 0,
                        start: 0,
                        length: 1
                    };
                    128
                ]
            },
        )
        .unwrap();
        zones
            .append(
                ZoneMaps::from_arrays(
                    &values.slice(200, 7),
                    &values.slice(200, 7),
                    &counts.slice(200, 7),
                    &counts.slice(200, 7),
                    None,
                    || {
                        vec![
                            ZoneBound {
                                fragment_id: 1,
                                start: 0,
                                length: 1
                            };
                            7
                        ]
                    },
                )
                .unwrap(),
            )
            .unwrap();
        let ZoneMaps::Columnar(chunks) = &zones else {
            panic!("expected columnar batches");
        };
        let metadata = chunks.capacity() * std::mem::size_of::<ColumnarZones>();
        let bounds = chunks
            .iter()
            .map(|chunk| chunk.bounds.capacity() * std::mem::size_of::<ZoneBound>())
            .sum::<usize>();
        let mut context = Context::new();
        let buffers = values.as_ref().deep_size_of_children(&mut context)
            + (&counts as &dyn Array).deep_size_of_children(&mut context);
        assert_eq!(
            zones.deep_size_of_children(&mut Context::new()),
            metadata + bounds + buffers + 4 * std::mem::size_of::<Int64Array>()
        );
        assert_eq!(zones.batch_lengths(), vec![128, 7]);
    }

    #[test]
    fn string_views_borrow_complete_values() {
        let long = "世界".repeat(16_384);
        let array = StringArray::from(vec![Some(long.as_str()), None, Some("")]);
        let extrema = Extrema::from_array(&(Arc::new(array.clone()) as ArrayRef)).unwrap();
        let value = ByteBound::new(extrema.array.as_string::<i32>(), 0)
            .value
            .unwrap();
        assert_eq!(value.as_ptr(), array.value(0).as_ptr());
        assert_eq!(value, long);
        assert!(extrema.value(1).is_null());
        assert_eq!(
            extrema.value(2).into_owned(),
            ScalarValue::Utf8(Some(String::new()))
        );
    }
}
