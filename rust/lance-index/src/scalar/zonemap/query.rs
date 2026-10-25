// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The predicate evaluator accepts either the original scalars or typed borrowed
//! bounds. Selecting the bound type once per search keeps storage dispatch out
//! of the per-zone comparison path.

use std::cmp::Ordering;
use std::marker::PhantomData;

use arrow_array::types::*;
use arrow_array::{Array, ArrowNativeTypeOp, GenericByteArray, OffsetSizeTrait, PrimitiveArray};
use arrow_schema::DataType;
use datafusion_common::ScalarValue;
use lance_core::Result;

use super::ZoneMapStatistics;

pub(super) trait ZoneMapValue: PartialOrd<ScalarValue> {
    fn is_null(&self) -> bool;
    fn is_nan(&self) -> bool;
}

pub(super) trait ZoneMapPredicate {
    fn evaluate<V>(&self, zone: &ZoneMapStatistics<V>) -> Result<bool>
    where
        V: ZoneMapValue,
        ScalarValue: PartialOrd<V>;
}

pub(super) fn scalar_is_nan(value: &ScalarValue) -> bool {
    match value {
        ScalarValue::Float16(Some(value)) => value.is_nan(),
        ScalarValue::Float32(Some(value)) => value.is_nan(),
        ScalarValue::Float64(Some(value)) => value.is_nan(),
        _ => false,
    }
}

impl ZoneMapValue for ScalarValue {
    #[inline]
    fn is_null(&self) -> bool {
        self.is_null()
    }

    #[inline]
    fn is_nan(&self) -> bool {
        scalar_is_nan(self)
    }
}

pub(super) trait PrimitiveKind: ArrowPrimitiveType + Sized {
    // Only decimal comparisons need runtime metadata; other bounds use `()`.
    type Metadata: Copy + Default;

    fn metadata(_array: &PrimitiveArray<Self>) -> Self::Metadata {
        Self::Metadata::default()
    }

    /// Distinguishes another scalar variant (`None`) from a typed null (`Some(None)`).
    fn scalar_value(
        metadata: Self::Metadata,
        value: &ScalarValue,
        equality: bool,
    ) -> Option<Option<Self::Native>>;

    fn matching_metadata(_left: Self::Metadata, _right: Self::Metadata, _equality: bool) -> bool {
        true
    }

    fn owned(array: &PrimitiveArray<Self>, value: Option<Self::Native>) -> ScalarValue;

    fn is_nan(_value: Self::Native) -> bool {
        false
    }
}

macro_rules! primitive_kinds {
    ($($arrow:ty => $scalar:ident),+ $(,)?) => {
        $(impl PrimitiveKind for $arrow {
            type Metadata = ();
            #[inline]
            fn scalar_value(_metadata: (), value: &ScalarValue, _equality: bool) -> Option<Option<Self::Native>> {
                match value {
                    ScalarValue::$scalar(value, ..) => Some(*value),
                    _ => None,
                }
            }
            fn owned(_array: &PrimitiveArray<Self>, value: Option<Self::Native>) -> ScalarValue { ScalarValue::$scalar(value) }
        })+
    };
}

// Arrow's shared dispatch instantiates interval and Decimal32/64 branches too;
// storage keeps those unsupported writer types in the generic representation.
primitive_kinds! {
    Int8Type => Int8,
    Int16Type => Int16,
    Int32Type => Int32,
    Int64Type => Int64,
    UInt8Type => UInt8,
    UInt16Type => UInt16,
    UInt32Type => UInt32,
    UInt64Type => UInt64,
    Date32Type => Date32,
    Date64Type => Date64,
    Time32SecondType => Time32Second,
    Time32MillisecondType => Time32Millisecond,
    Time64MicrosecondType => Time64Microsecond,
    Time64NanosecondType => Time64Nanosecond,
    DurationSecondType => DurationSecond,
    DurationMillisecondType => DurationMillisecond,
    DurationMicrosecondType => DurationMicrosecond,
    DurationNanosecondType => DurationNanosecond,
    IntervalYearMonthType => IntervalYearMonth,
    IntervalDayTimeType => IntervalDayTime,
    IntervalMonthDayNanoType => IntervalMonthDayNano,
}

macro_rules! float_kinds {
    ($($arrow:ty => $scalar:ident),+ $(,)?) => {$(
        impl PrimitiveKind for $arrow {
            type Metadata = ();
            fn scalar_value(_metadata: (), value: &ScalarValue, _equality: bool) -> Option<Option<Self::Native>> {
                match value { ScalarValue::$scalar(value) => Some(*value), _ => None }
            }
            fn owned(_array: &PrimitiveArray<Self>, value: Option<Self::Native>) -> ScalarValue { ScalarValue::$scalar(value) }
            fn is_nan(value: Self::Native) -> bool { value.is_nan() }
        }
    )+};
}

float_kinds! { Float16Type => Float16, Float32Type => Float32, Float64Type => Float64 }

// ScalarValue ignores timezone for comparisons, but owned bounds retain it.
macro_rules! timestamp_kinds {
    ($($arrow:ty => $scalar:ident),+ $(,)?) => {$(
        impl PrimitiveKind for $arrow {
            type Metadata = ();
            fn scalar_value(_metadata: (), value: &ScalarValue, _equality: bool) -> Option<Option<Self::Native>> {
                match value { ScalarValue::$scalar(value, _) => Some(*value), _ => None }
            }
            fn owned(array: &PrimitiveArray<Self>, value: Option<Self::Native>) -> ScalarValue {
                let DataType::Timestamp(_, timezone) = array.data_type() else {
                    unreachable!("a validated timestamp array has timestamp metadata")
                };
                ScalarValue::$scalar(value, timezone.clone())
            }
        }
    )+};
}

timestamp_kinds! {
    TimestampSecondType => TimestampSecond,
    TimestampMillisecondType => TimestampMillisecond,
    TimestampMicrosecondType => TimestampMicrosecond,
    TimestampNanosecondType => TimestampNanosecond,
}

macro_rules! decimal_kinds {
    ($($arrow:ty => $scalar:ident),+ $(,)?) => {$(
        impl PrimitiveKind for $arrow {
            type Metadata = (u8, i8);
            fn metadata(array: &PrimitiveArray<Self>) -> Self::Metadata { (array.precision(), array.scale()) }
            fn scalar_value((precision, scale): Self::Metadata, value: &ScalarValue, equality: bool) -> Option<Option<Self::Native>> {
                match value {
                    ScalarValue::$scalar(value, other_precision, other_scale)
                        if scale == *other_scale && (!equality || precision == *other_precision) => Some(*value),
                    _ => None,
                }
            }
            // ScalarValue equality includes precision, but ordering only checks scale.
            fn matching_metadata(left: Self::Metadata, right: Self::Metadata, equality: bool) -> bool {
                left.1 == right.1 && (!equality || left.0 == right.0)
            }
            fn owned(array: &PrimitiveArray<Self>, value: Option<Self::Native>) -> ScalarValue {
                ScalarValue::$scalar(value, array.precision(), array.scale())
            }
        }
    )+};
}

decimal_kinds! { Decimal32Type => Decimal32, Decimal64Type => Decimal64, Decimal128Type => Decimal128, Decimal256Type => Decimal256 }

pub(super) struct PrimitiveBound<T: PrimitiveKind> {
    pub(super) value: Option<T::Native>,
    metadata: T::Metadata,
}

impl<T: PrimitiveKind> PrimitiveBound<T> {
    pub(super) fn new(array: &PrimitiveArray<T>, index: usize) -> Self {
        Self {
            value: array.is_valid(index).then(|| array.value(index)),
            metadata: T::metadata(array),
        }
    }

    pub(super) fn compare(&self, other: &Self) -> Option<Ordering> {
        T::matching_metadata(self.metadata, other.metadata, false)
            .then(|| native_cmp(self.value, other.value))
    }

    pub(super) fn equals(&self, other: &Self) -> bool {
        T::matching_metadata(self.metadata, other.metadata, true)
            && native_eq(self.value, other.value)
    }
}

// Arrow's float operations match ScalarValue's bitwise equality and total order.
fn native_cmp<T: ArrowNativeTypeOp>(left: Option<T>, right: Option<T>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.compare(right),
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
    }
}

fn native_eq<T: ArrowNativeTypeOp>(left: Option<T>, right: Option<T>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left.is_eq(right),
        (None, None) => true,
        _ => false,
    }
}

impl<T: PrimitiveKind> ZoneMapValue for PrimitiveBound<T> {
    #[inline]
    fn is_null(&self) -> bool {
        self.value.is_none()
    }

    #[inline]
    fn is_nan(&self) -> bool {
        self.value.is_some_and(T::is_nan)
    }
}

impl<T: PrimitiveKind> PartialEq<ScalarValue> for PrimitiveBound<T> {
    fn eq(&self, other: &ScalarValue) -> bool {
        T::scalar_value(self.metadata, other, true)
            .is_some_and(|value| native_eq(self.value, value))
    }
}

impl<T: PrimitiveKind> PartialEq for PrimitiveBound<T> {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl<T: PrimitiveKind> PartialOrd for PrimitiveBound<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.compare(other)
    }
}

impl<T: PrimitiveKind> PartialOrd<ScalarValue> for PrimitiveBound<T> {
    #[inline]
    fn partial_cmp(&self, other: &ScalarValue) -> Option<Ordering> {
        Some(native_cmp(
            self.value,
            T::scalar_value(self.metadata, other, false)?,
        ))
    }
}

impl<T: PrimitiveKind> PartialEq<PrimitiveBound<T>> for ScalarValue {
    fn eq(&self, other: &PrimitiveBound<T>) -> bool {
        other.eq(self)
    }
}

impl<T: PrimitiveKind> PartialOrd<PrimitiveBound<T>> for ScalarValue {
    #[inline]
    fn partial_cmp(&self, other: &PrimitiveBound<T>) -> Option<Ordering> {
        other.partial_cmp(self).map(Ordering::reverse)
    }
}

pub(super) trait ByteKind: ByteArrayType<Native: Ord> {
    fn scalar_value(value: &ScalarValue) -> Option<Option<&Self::Native>>;
    fn owned(value: Option<&Self::Native>) -> ScalarValue;
}

impl<O: OffsetSizeTrait> ByteKind for GenericStringType<O> {
    fn scalar_value(value: &ScalarValue) -> Option<Option<&str>> {
        match (O::IS_LARGE, value) {
            (false, ScalarValue::Utf8(value)) | (true, ScalarValue::LargeUtf8(value)) => {
                Some(value.as_deref())
            }
            _ => None,
        }
    }

    fn owned(value: Option<&str>) -> ScalarValue {
        if O::IS_LARGE {
            ScalarValue::LargeUtf8(value.map(str::to_owned))
        } else {
            ScalarValue::Utf8(value.map(str::to_owned))
        }
    }
}

impl<O: OffsetSizeTrait> ByteKind for GenericBinaryType<O> {
    fn scalar_value(value: &ScalarValue) -> Option<Option<&[u8]>> {
        match (O::IS_LARGE, value) {
            (false, ScalarValue::Binary(value)) | (true, ScalarValue::LargeBinary(value)) => {
                Some(value.as_deref())
            }
            _ => None,
        }
    }

    fn owned(value: Option<&[u8]>) -> ScalarValue {
        if O::IS_LARGE {
            ScalarValue::LargeBinary(value.map(<[u8]>::to_vec))
        } else {
            ScalarValue::Binary(value.map(<[u8]>::to_vec))
        }
    }
}

pub(super) struct ByteBound<'a, T: ByteKind> {
    pub(super) value: Option<&'a T::Native>,
    kind: PhantomData<T>,
}

impl<'a, T: ByteKind> ByteBound<'a, T> {
    pub(super) fn new(array: &'a GenericByteArray<T>, index: usize) -> Self {
        Self {
            value: array.is_valid(index).then(|| array.value(index)),
            kind: PhantomData,
        }
    }
}

impl<T: ByteKind> ZoneMapValue for ByteBound<'_, T> {
    #[inline]
    fn is_null(&self) -> bool {
        self.value.is_none()
    }

    #[inline]
    fn is_nan(&self) -> bool {
        false
    }
}

impl<T: ByteKind> PartialEq<ScalarValue> for ByteBound<'_, T> {
    fn eq(&self, other: &ScalarValue) -> bool {
        T::scalar_value(other).is_some_and(|value| self.value == value)
    }
}

impl<T: ByteKind> PartialEq for ByteBound<'_, T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: ByteKind> PartialOrd for ByteBound<'_, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: ByteKind> PartialOrd<ScalarValue> for ByteBound<'_, T> {
    #[inline]
    fn partial_cmp(&self, other: &ScalarValue) -> Option<Ordering> {
        self.value.partial_cmp(&T::scalar_value(other)?)
    }
}

impl<T: ByteKind> PartialEq<ByteBound<'_, T>> for ScalarValue {
    fn eq(&self, other: &ByteBound<'_, T>) -> bool {
        other.eq(self)
    }
}

impl<T: ByteKind> PartialOrd<ByteBound<'_, T>> for ScalarValue {
    #[inline]
    fn partial_cmp(&self, other: &ByteBound<'_, T>) -> Option<Ordering> {
        other.partial_cmp(self).map(Ordering::reverse)
    }
}

#[derive(PartialEq, PartialOrd)]
pub(super) struct BooleanBound(pub(super) Option<bool>);

impl ZoneMapValue for BooleanBound {
    fn is_null(&self) -> bool {
        self.0.is_none()
    }
    fn is_nan(&self) -> bool {
        false
    }
}

impl PartialEq<ScalarValue> for BooleanBound {
    fn eq(&self, other: &ScalarValue) -> bool {
        matches!(other, ScalarValue::Boolean(value) if self.0 == *value)
    }
}

impl PartialOrd<ScalarValue> for BooleanBound {
    fn partial_cmp(&self, other: &ScalarValue) -> Option<Ordering> {
        match other {
            ScalarValue::Boolean(value) => self.0.partial_cmp(value),
            _ => None,
        }
    }
}

impl PartialEq<BooleanBound> for ScalarValue {
    fn eq(&self, other: &BooleanBound) -> bool {
        other.eq(self)
    }
}

impl PartialOrd<BooleanBound> for ScalarValue {
    fn partial_cmp(&self, other: &BooleanBound) -> Option<Ordering> {
        other.partial_cmp(self).map(Ordering::reverse)
    }
}
