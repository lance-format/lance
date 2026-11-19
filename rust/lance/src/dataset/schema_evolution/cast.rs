// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::compute::{CastOptions, can_cast_types};
use arrow_array::cast::AsArray;
use arrow_array::types::Float64Type;
use arrow_array::{Array, ArrayRef};
use arrow_cast::display::{ArrayFormatter, FormatOptions};
use arrow_cast::{cast, cast_with_options};
use arrow_schema::DataType;

use crate::{Error, Result};

const STRICT: CastOptions<'static> = CastOptions {
    safe: false,
    format_options: FormatOptions::new(),
};

/// Casts the values of column `name` for `alter_columns`, failing if a value
/// cannot be represented in `to_type`.
///
/// `safe: false` only makes arrow fail on values it cannot convert at all,
/// such as an integer overflow. It still truncates timestamps to a coarser
/// unit, rounds decimals to a smaller scale and turns out-of-range floats into
/// infinity, so those are checked after the cast. Floats may still lose
/// precision: only a finite value that becomes infinite is rejected.
pub(super) fn cast_column(name: &str, column: &ArrayRef, to_type: &DataType) -> Result<ArrayRef> {
    let casted = cast_with_options(column, to_type, &STRICT)?;
    if let Some(value) = first_unrepresentable(column, &casted)? {
        return Err(Error::invalid_input(format!(
            "Cannot cast column \"{}\" from {:?} to {:?}: value {} cannot be represented in the new type",
            name,
            column.data_type(),
            to_type,
            value
        )));
    }
    Ok(casted)
}

/// Returns the first value of `original` that `casted` does not represent.
fn first_unrepresentable(original: &ArrayRef, casted: &ArrayRef) -> Result<Option<String>> {
    use DataType::*;
    let index = match (original.data_type(), casted.data_type()) {
        (Dictionary(_, value_type), _) => {
            return first_unrepresentable(&cast(original, value_type)?, casted);
        }
        (_, Dictionary(_, value_type)) => {
            return first_unrepresentable(original, &cast(casted, value_type)?);
        }
        (Struct(_), Struct(_)) => {
            for (original, casted) in original
                .as_struct()
                .columns()
                .iter()
                .zip(casted.as_struct().columns())
            {
                if let Some(value) = first_unrepresentable(original, casted)? {
                    return Ok(Some(value));
                }
            }
            return Ok(None);
        }
        (
            List(item) | LargeList(item) | FixedSizeList(item, _),
            List(_) | LargeList(_) | FixedSizeList(_, _),
        ) => {
            // Lay the original items out like the casted ones (a cast to a fixed
            // size list pads null lists) so the two can be compared item by item.
            let layout = match casted.data_type() {
                LargeList(_) => LargeList(item.clone()),
                FixedSizeList(_, size) => FixedSizeList(item.clone(), *size),
                _ => List(item.clone()),
            };
            let original = cast_with_options(original, &layout, &STRICT)?;
            return first_unrepresentable(list_values(&original)?, list_values(casted)?);
        }
        (from, to) if from.is_floating() && to.is_floating() => {
            let original = cast(original, &Float64)?;
            let casted = cast(casted, &Float64)?;
            original
                .as_primitive::<Float64Type>()
                .iter()
                .zip(casted.as_primitive::<Float64Type>().iter())
                .position(|(original, casted)| {
                    original.is_some_and(f64::is_finite) && casted.is_some_and(f64::is_infinite)
                })
        }
        (from, to) if (from.is_temporal() || from.is_decimal()) && can_cast_types(to, from) => {
            // Adding a time zone to a timestamp shifts it from local time to UTC,
            // but removing one keeps the UTC value, so a round trip between them
            // does not return the original. Only the unit conversion can lose
            // values, so check that on its own.
            let casted = match (from, to) {
                (Timestamp(_, from_tz), Timestamp(unit, to_tz)) if from_tz != to_tz => {
                    cast_with_options(original, &Timestamp(*unit, from_tz.clone()), &STRICT)?
                }
                _ => casted.clone(),
            };
            let round_trip = cast(&casted, from)?;
            arrow_ord::cmp::distinct(original, &round_trip)?
                .values()
                .set_indices()
                .next()
        }
        _ => None,
    };
    index
        .map(|index| {
            Ok(
                ArrayFormatter::try_new(original.as_ref(), &FormatOptions::default())?
                    .value(index)
                    .to_string(),
            )
        })
        .transpose()
}

fn list_values(array: &ArrayRef) -> Result<&ArrayRef> {
    match array.data_type() {
        DataType::List(_) => Ok(array.as_list::<i32>().values()),
        DataType::LargeList(_) => Ok(array.as_list::<i64>().values()),
        DataType::FixedSizeList(_, _) => Ok(array.as_fixed_size_list().values()),
        other => Err(Error::internal(format!(
            "Expected a list array while checking a cast, got {other:?}"
        ))),
    }
}
