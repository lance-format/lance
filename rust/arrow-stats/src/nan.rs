// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::cast::AsArray;
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef};

/// Number of non-null NaN values in an array, and how many of them carry the
/// sign bit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NanCounts {
    pub total: u64,
    pub negative: u64,
}

macro_rules! count_nans_typed {
    ($array:expr, $arrow_type:ty) => {{
        let typed = $array.as_primitive::<$arrow_type>();
        let mut counts = NanCounts::default();
        for i in 0..typed.len() {
            if typed.is_null(i) {
                continue;
            }
            let value = typed.value(i);
            if value.is_nan() {
                counts.total += 1;
                if value.is_sign_negative() {
                    counts.negative += 1;
                }
            }
        }
        counts
    }};
}

/// Count the non-null NaN values in an array, split by sign.
///
/// Returns zero counts for non-float types.
pub fn count_nans(array: &ArrayRef) -> NanCounts {
    use arrow_schema::DataType::*;
    match array.data_type() {
        Float16 => count_nans_typed!(array, Float16Type),
        Float32 => count_nans_typed!(array, Float32Type),
        Float64 => count_nans_typed!(array, Float64Type),
        _ => NanCounts::default(),
    }
}
