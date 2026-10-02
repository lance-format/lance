// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Choosing and materializing the placeholder ("blank") values that restore deleted
//! rows when a fragment's columns are rewritten. See [`super::DeletionRestorer`].

use std::{collections::HashMap, sync::Arc};

use arrow_array::cast::AsArray;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, FixedSizeListArray, GenericListArray,
    LargeBinaryArray, LargeStringArray, MapArray, OffsetSizeTrait, RecordBatch, StringArray,
    StringViewArray, StructArray, UInt32Array, new_null_array,
};
use arrow_buffer::{NullBuffer, NullBufferBuilder, OffsetBuffer};
use arrow_schema::{ArrowError, DataType, Field as ArrowField};
use arrow_select::dictionary::garbage_collect_any_dictionary;
use arrow_select::interleave::interleave;
use lance_arrow::FieldExt;
use lance_core::datatypes::Field as LanceField;
use lance_core::{Error, Result, datatypes::Schema};
use lance_file::version::ConcreteFileVersion;

/// Materialize the one-row source that later blank-only batches copy from.
///
/// Two things keep this row from pinning the payload of the batch it came from:
///
/// * a top-level column planned as [`BlankPlan::Interleave`] is replaced with its
///   filler. Nothing ever reads the source row of such a column -- the filler is what a
///   blank gets -- so a multi-megabyte first live value would be retained for nothing.
/// * dictionary values are garbage collected. `take` shares the values array, so
///   without this the whole dictionary would stay alive behind a single key.
///
/// Every other column keeps the taken row. For `List` and `Map` that is required: their
/// blanks reuse the child array as-is. For `Struct` and `FixedSizeList` it is merely
/// unrefined -- their children are planned independently, so a variable-width child's
/// bytes are retained even though its blanks come from a filler. `take` has already
/// narrowed that to one logical row, which is the same bound the container arms carry,
/// so it is not worth a second recursion to shave.
pub(super) fn compact_blank_source(
    source: &RecordBatch,
    storage_version: ConcreteFileVersion,
    write_schema: Option<&Schema>,
) -> Result<RecordBatch> {
    debug_assert_eq!(source.num_rows(), 1);
    let source = arrow_select::take::take_record_batch(source, &UInt32Array::from(vec![0]))?;
    let plans = blank_plans(&source, storage_version, write_schema)?;
    let columns = source
        .columns()
        .iter()
        .zip(&plans)
        .map(|(array, plan)| match plan {
            BlankPlan::Interleave(filler) => Ok(filler.clone()),
            _ => compact_nested_dictionaries(array.clone()),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RecordBatch::try_new(source.schema(), columns)?)
}

/// Rebuild `array` with every dictionary it contains narrowed to the values its keys
/// actually reference.
///
/// The container arms exist only to reach nested dictionaries; they rebuild the
/// wrapper unchanged otherwise. Note that [`AnyDictionaryArray::with_values`] drops
/// the `is_ordered` flag, which is immaterial for a blank source.
fn compact_nested_dictionaries(array: ArrayRef) -> Result<ArrayRef> {
    match array.data_type() {
        DataType::Dictionary(_, _) => {
            let dictionary = garbage_collect_any_dictionary(array.as_any_dictionary())?;
            let values =
                compact_nested_dictionaries(dictionary.as_any_dictionary().values().clone())?;
            Ok(dictionary.as_any_dictionary().with_values(values))
        }
        DataType::Struct(_) => {
            let array = array.as_struct();
            let columns = array
                .columns()
                .iter()
                .cloned()
                .map(compact_nested_dictionaries)
                .collect::<Result<Vec<_>>>()?;
            // Supply the length: `try_new` cannot infer it for a struct with no fields.
            Ok(Arc::new(StructArray::try_new_with_length(
                array.fields().clone(),
                columns,
                array.nulls().cloned(),
                array.len(),
            )?))
        }
        DataType::List(field) => {
            let array = array.as_list::<i32>();
            let values = compact_nested_dictionaries(array.values().clone())?;
            Ok(Arc::new(GenericListArray::<i32>::try_new(
                field.clone(),
                array.offsets().clone(),
                values,
                array.nulls().cloned(),
            )?))
        }
        DataType::LargeList(field) => {
            let array = array.as_list::<i64>();
            let values = compact_nested_dictionaries(array.values().clone())?;
            Ok(Arc::new(GenericListArray::<i64>::try_new(
                field.clone(),
                array.offsets().clone(),
                values,
                array.nulls().cloned(),
            )?))
        }
        DataType::FixedSizeList(field, size) => {
            let array = array.as_fixed_size_list();
            let values = compact_nested_dictionaries(array.values().clone())?;
            Ok(Arc::new(FixedSizeListArray::try_new_with_length(
                field.clone(),
                *size,
                values,
                array.nulls().cloned(),
                array.len(),
            )?))
        }
        DataType::Map(entries, ordered) => {
            let entries = entries.clone();
            let ordered = *ordered;
            // `MapArray::try_new` rejects a nullable entries field (arrow 58), which the
            // Arrow spec forbids but Lance tolerates. Leave such a map untouched rather
            // than error: a blank source is a single row, so skipping dictionary GC for
            // it is immaterial, and `blank_plan` already routes it to `Take`.
            if entries.is_nullable() {
                return Ok(array);
            }
            let map = array.as_map();
            let compacted_entries = compact_nested_dictionaries(Arc::new(map.entries().clone()))?;
            Ok(Arc::new(MapArray::try_new(
                entries,
                map.offsets().clone(),
                compacted_entries.as_struct().clone(),
                map.nulls().cloned(),
                ordered,
            )?))
        }
        _ => Ok(array),
    }
}

/// Add blank rows where there are deleted rows
///
/// `batch_offsets` must be strictly increasing, and no offset may require more
/// live rows before it than the batch has left: an offset is the position a blank
/// takes in the output, so either kind of violation asks for an impossible number
/// of live rows in between.
///
/// A blank holds the cheapest value its column can represent, chosen per column by
/// [`blank_plan`]: a null where the column is nullable, an empty value for the
/// variable-width types whose byte cost depends on the value, and a copy of the batch's
/// first row for the fixed-width ones where every value costs the same (so the batch
/// must have at least one row). Legacy (v1) files opt out entirely and copy row zero.
///
/// Every live row in `batch` appears exactly once and in its original order. Live
/// rows not consumed before the last blank offset are appended to the output. In
/// particular, offsets `0..n` produce `n` blanks followed by the complete input;
/// [`super::DeletionRestorer::take_pending_blanks_with_plans`] relies on this when it
/// removes that trailing source row.
///
/// [`super::DeletionRestorer::restore`] defers blanks from an empty batch until a live
/// row is available as a source. Only legacy files, which cannot defer without changing
/// their row-group layout, can still reach the error below.
pub(super) fn add_blanks(
    batch: RecordBatch,
    batch_offsets: &[u32],
    storage_version: ConcreteFileVersion,
    write_schema: Option<&Schema>,
) -> Result<RecordBatch> {
    // Fast early return
    if batch_offsets.is_empty() {
        return Ok(batch);
    }

    if batch.num_rows() == 0 {
        return Err(Error::not_supported(
            "Fragment Updater: missing too many rows in merge, run compaction to materialize \
             deletions first",
        ));
    }

    let plans = blank_plans(&batch, storage_version, write_schema)?;
    add_blanks_with_plans(batch, batch_offsets, &plans)
}

pub(super) fn blank_plans(
    batch: &RecordBatch,
    storage_version: ConcreteFileVersion,
    write_schema: Option<&Schema>,
) -> Result<Vec<BlankPlan>> {
    // Exhaustive on purpose. `matches!` would let a future format variant fall through
    // to the `!cheap_blanks` path below and silently return to copying row zero, which
    // reintroduces the offset overflow this fixes on the newest format. A `match` forces
    // a deliberate choice for every version, the same way `validate_nulls` and
    // `check_field_conflict` do.
    let cheap_blanks = match storage_version {
        // Legacy (v1) is largely deprecated and its writer supports physical nulls for a
        // narrower set of types, so it keeps the original behavior of copying row zero
        // into every blank.
        ConcreteFileVersion::V1 => false,
        // Everything below the planner assumes the format can store a null for any type
        // it will actually null: v2.0 only forbids struct-level nulls, and the planner
        // never nulls a struct (it mirrors row zero's validity), so v2.0+ is uniform.
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => true,
    };
    if !cheap_blanks {
        return Ok((0..batch.num_columns()).map(|_| BlankPlan::Take).collect());
    }

    // Index the write schema's top-level fields by name instead of calling
    // `Schema::field`, which parses its argument as a dotted field path. A field name
    // is not a path: a backtick is legal in one but is the path syntax's quote
    // character, so `Schema::field` would either fail to parse the name or resolve it
    // to a different field.
    let write_fields = write_schema.map(|schema| {
        schema
            .fields
            .iter()
            .map(|field| (field.name.as_str(), field))
            .collect::<HashMap<_, _>>()
    });
    batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, array)| {
            let write_field = match &write_fields {
                Some(fields) => Some(*fields.get(field.name().as_str()).ok_or_else(|| {
                    Error::internal(format!(
                        "Fragment Updater: field {} is missing from the write schema",
                        field.name()
                    ))
                })?),
                None => None,
            };
            blank_plan(field, write_field, array)
        })
        .collect()
}

/// Apply precomputed blank plans to a batch.
///
/// `plans` must have been produced by [`blank_plans`] for `batch`; plan variants
/// and their nested fillers are coupled to the batch's column order and types.
pub(super) fn add_blanks_with_plans(
    batch: RecordBatch,
    batch_offsets: &[u32],
    plans: &[BlankPlan],
) -> Result<RecordBatch> {
    debug_assert!(!batch_offsets.is_empty());
    debug_assert_eq!(batch.num_columns(), plans.len());
    // A blank can need a live row to copy, and the Arrow kernels below run unchecked,
    // so refuse an empty batch here rather than in debug only. `add_blanks` reports the
    // same condition with the wording callers already match on.
    if batch.num_rows() == 0 {
        return Err(Error::internal(
            "Fragment Updater: cannot place blanks in a batch with no live rows",
        ));
    }

    let needs_take = plans.iter().any(BlankPlan::needs_take);
    let needs_interleave = plans.iter().any(BlankPlan::needs_interleave);
    let output_len = batch
        .num_rows()
        .checked_add(batch_offsets.len())
        .ok_or_else(|| Error::internal("Fragment Updater: blank output row count overflow"))?;
    let mut take_indices = needs_take.then(|| Vec::with_capacity(output_len));
    // Indices for `interleave`: `(0, pos)` picks live row `pos` out of the column
    // itself, `(1, 0)` picks the column's one-row blank filler.
    let mut interleave_indices = needs_interleave.then(|| Vec::with_capacity(output_len));

    let num_live_rows = u32::try_from(batch.num_rows()).map_err(|_| {
        Error::internal(format!(
            "Fragment Updater: blank source has {} rows, exceeding the u32 row-address limit",
            batch.num_rows()
        ))
    })?;
    let mut batch_pos = 0;
    let mut next_id = 0;
    for (idx, batch_offset) in batch_offsets.iter().enumerate() {
        // A non-increasing offset panics in debug and wraps in release; reject it
        // up front so the error names the real problem.
        let num_rows = batch_offset.checked_sub(next_id).ok_or_else(|| {
            Error::internal(format!(
                "Fragment Updater: blank offsets must be strictly increasing, but offset \
                 {batch_offset} (entry {idx} of {}) is below the expected minimum {next_id}",
                batch_offsets.len()
            ))
        })?;
        // An offset needing more live rows than remain would index past the batch.
        // The Arrow kernels run unchecked below, so catch this here rather than
        // letting them panic or, worse, read the wrong rows.
        if num_rows > num_live_rows - batch_pos {
            return Err(Error::internal(format!(
                "Fragment Updater: blank offset {batch_offset} (entry {idx} of \
                 {}) needs {num_rows} more live rows before it, but {} of the batch's \
                 {num_live_rows} are still unused",
                batch_offsets.len(),
                num_live_rows - batch_pos
            )));
        }
        if let Some(indices) = take_indices.as_mut() {
            indices.extend(batch_pos..batch_pos + num_rows);
            indices.push(0);
        }
        if let Some(indices) = interleave_indices.as_mut() {
            indices.extend((batch_pos..batch_pos + num_rows).map(|pos| (0, pos as usize)));
            indices.push((1, 0));
        }
        next_id = batch_offset.checked_add(1).ok_or_else(|| {
            Error::internal(format!(
                "Fragment Updater: blank offset {batch_offset} cannot be followed by another row"
            ))
        })?;
        batch_pos = batch_pos.checked_add(num_rows).ok_or_else(|| {
            Error::internal("Fragment Updater: live-row selection position overflow")
        })?;
    }
    if let Some(indices) = take_indices.as_mut() {
        indices.extend(batch_pos..num_live_rows);
    }
    if let Some(indices) = interleave_indices.as_mut() {
        indices.extend((batch_pos..num_live_rows).map(|pos| (0, pos as usize)));
    }
    let take_indices = take_indices.map(UInt32Array::from);

    let arrays = batch
        .columns()
        .iter()
        .zip(plans)
        .map(|(array, plan)| {
            apply_blank_plan(
                array,
                plan,
                take_indices.as_ref(),
                interleave_indices.as_deref(),
            )
            .map_err(|error| Error::arrow(format!("Failed to add blanks: {error}")))
        })
        .collect::<Result<Vec<_>>>()?;

    let batch = RecordBatch::try_new(batch.schema(), arrays)?;

    Ok(batch)
}

pub(super) enum BlankPlan {
    /// Copy row zero for each blank. This is the cheapest option when every value
    /// has the same physical cost, and it preserves dictionary values buffers.
    Take,
    /// Interleave a one-row empty or null filler with the live rows.
    Interleave(ArrayRef),
    /// Rebuild list offsets while preserving the child array unchanged.
    List { blank_is_null: bool },
    /// Rebuild map offsets while preserving the entries array unchanged.
    Map { blank_is_null: bool },
    /// Expand each selected parent row into child selections and apply the child plan.
    FixedSizeList {
        child: Box<Self>,
        blank_is_null: bool,
    },
    /// Apply the two strategies independently to the children and rebuild validity.
    Struct(Vec<Self>),
}

impl BlankPlan {
    fn needs_take(&self) -> bool {
        match self {
            Self::Take => true,
            Self::Interleave(_) | Self::List { .. } | Self::Map { .. } => false,
            // Struct and FixedSizeList validity follows the same source rows as
            // their `Take` children, so both require the take selection.
            Self::FixedSizeList { .. } | Self::Struct(_) => true,
        }
    }

    fn needs_interleave(&self) -> bool {
        match self {
            Self::Take => false,
            Self::Interleave(_)
            | Self::List { .. }
            | Self::Map { .. }
            | Self::FixedSizeList { .. }
            | Self::Struct(_) => true,
        }
    }
}

fn interleaved_nulls(
    array: &dyn Array,
    blank_is_null: bool,
    indices: &[(usize, usize)],
) -> std::result::Result<Option<NullBuffer>, ArrowError> {
    let mut nulls = NullBufferBuilder::new(indices.len());
    for (source, row) in indices {
        let is_valid = match source {
            0 => array.is_valid(*row),
            1 => !blank_is_null,
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "blank plan has invalid source {source}"
                )));
            }
        };
        nulls.append(is_valid);
    }
    Ok(nulls.finish())
}

/// Build the output offsets for a list-like column whose blanks are empty.
///
/// Reusing absolute input offsets is valid because `add_blanks` keeps every live row
/// exactly once and in identity order, inserting only zero-length blanks. `kind` names
/// the layout in error messages.
fn blank_offsets<O: OffsetSizeTrait>(
    kind: &str,
    value_offsets: &[O],
    num_live_rows: usize,
    indices: &[(usize, usize)],
) -> std::result::Result<OffsetBuffer<O>, ArrowError> {
    let first = *value_offsets.first().ok_or_else(|| {
        ArrowError::ComputeError(format!("{kind} blank plan has no initial offset"))
    })?;
    let mut offsets = Vec::with_capacity(indices.len() + 1);
    offsets.push(first);
    let mut next_live_row = 0;
    for (source, row) in indices {
        let last = *offsets.last().ok_or_else(|| {
            ArrowError::ComputeError(format!("{kind} blank plan has no preceding offset"))
        })?;
        let next = match source {
            0 if *row == next_live_row && next_live_row < num_live_rows => {
                next_live_row += 1;
                value_offsets[row + 1]
            }
            0 => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "{kind} blank plan requires live row {next_live_row}, got {row}"
                )));
            }
            // A blank is an empty value, so it does not advance the child offset.
            1 if *row == 0 => last,
            1 => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "{kind} blank plan requires filler row 0, got {row}"
                )));
            }
            source => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "{kind} blank plan has invalid source {source}"
                )));
            }
        };
        // Monotonicity holds for any well-formed input, but check it rather than let
        // `OffsetBuffer::new` assert: a panic here would take down the writer.
        if next < last {
            return Err(ArrowError::ComputeError(format!(
                "{kind} blank plan produced a non-monotonic offset: {next:?} after {last:?}"
            )));
        }
        offsets.push(next);
    }
    if next_live_row != num_live_rows {
        return Err(ArrowError::InvalidArgumentError(format!(
            "{kind} blank plan selected {next_live_row} of {num_live_rows} live rows"
        )));
    }
    Ok(OffsetBuffer::new(offsets.into()))
}

fn apply_list_blank_plan<O: OffsetSizeTrait>(
    array: &GenericListArray<O>,
    blank_is_null: bool,
    indices: &[(usize, usize)],
) -> std::result::Result<ArrayRef, ArrowError> {
    let field = match array.data_type() {
        DataType::List(field) | DataType::LargeList(field) => field.clone(),
        data_type => {
            return Err(ArrowError::InvalidArgumentError(format!(
                "list blank plan requires a list array, got {data_type}"
            )));
        }
    };
    Ok(Arc::new(GenericListArray::<O>::try_new(
        field,
        blank_offsets("list", array.value_offsets(), array.len(), indices)?,
        array.values().clone(),
        interleaved_nulls(array, blank_is_null, indices)?,
    )?))
}

fn apply_map_blank_plan(
    array: &MapArray,
    blank_is_null: bool,
    indices: &[(usize, usize)],
) -> std::result::Result<ArrayRef, ArrowError> {
    let (entries, ordered) = match array.data_type() {
        DataType::Map(entries, ordered) => (entries.clone(), *ordered),
        data_type => {
            return Err(ArrowError::InvalidArgumentError(format!(
                "map blank plan requires a map array, got {data_type}"
            )));
        }
    };
    Ok(Arc::new(MapArray::try_new(
        entries,
        blank_offsets("map", array.value_offsets(), array.len(), indices)?,
        array.entries().clone(),
        interleaved_nulls(array, blank_is_null, indices)?,
        ordered,
    )?))
}

fn apply_fixed_size_list_blank_plan(
    array: &FixedSizeListArray,
    child_plan: &BlankPlan,
    blank_is_null: bool,
    take_indices: &UInt32Array,
    interleave_indices: &[(usize, usize)],
) -> std::result::Result<ArrayRef, ArrowError> {
    let (field, list_size) = match array.data_type() {
        DataType::FixedSizeList(field, size) => (field.clone(), *size),
        data_type => {
            return Err(ArrowError::InvalidArgumentError(format!(
                "fixed-size-list blank plan requires a FixedSizeList array, got {data_type}"
            )));
        }
    };
    let size = usize::try_from(list_size).map_err(|_| {
        ArrowError::InvalidArgumentError(format!("FixedSizeList has a negative size {list_size}"))
    })?;
    let output_len = interleave_indices.len();
    // Both index vectors are built in one pass by `add_blanks_with_plans`, so they
    // describe the same output rows. Check it rather than only assert in debug:
    // `UInt32Array::value` panics out of range, and this is the writer's path.
    if take_indices.len() != output_len {
        return Err(ArrowError::InvalidArgumentError(format!(
            "fixed-size-list blank plan got {} take indices for {output_len} output rows",
            take_indices.len()
        )));
    }

    let values = if size == 0 {
        array.values().clone()
    } else {
        let child_len = output_len.checked_mul(size).ok_or_else(|| {
            ArrowError::ComputeError("FixedSizeList child selection length overflow".to_string())
        })?;
        // One entry per child row is `size` times the parent's, so only build the
        // selection the child plan actually reads.
        let mut child_take_indices = child_plan
            .needs_take()
            .then(|| Vec::with_capacity(child_len));
        let mut child_interleave_indices = child_plan
            .needs_interleave()
            .then(|| Vec::with_capacity(child_len));

        for (output_row, (source, _)) in interleave_indices.iter().enumerate() {
            let input_row = take_indices.value(output_row) as usize;
            // `FixedSizeListArray::value_offset` truncates this product to `i32`, and
            // slicing already rebases the child array, so compute it directly.
            let start = input_row.checked_mul(size).ok_or_else(|| {
                ArrowError::ComputeError("FixedSizeList child offset overflow".to_string())
            })?;
            let end = start.checked_add(size).ok_or_else(|| {
                ArrowError::ComputeError("FixedSizeList child range overflow".to_string())
            })?;
            for child_row in start..end {
                if let Some(indices) = child_take_indices.as_mut() {
                    indices.push(u32::try_from(child_row).map_err(|_| {
                        ArrowError::ComputeError(
                            "FixedSizeList child index exceeds UInt32 capacity".to_string(),
                        )
                    })?);
                }
                if let Some(indices) = child_interleave_indices.as_mut() {
                    indices.push(match source {
                        0 => (0, child_row),
                        1 => (1, 0),
                        _ => {
                            return Err(ArrowError::InvalidArgumentError(format!(
                                "fixed-size-list blank plan has invalid source {source}"
                            )));
                        }
                    });
                }
            }
        }

        let child_take_indices = child_take_indices.map(UInt32Array::from);
        apply_blank_plan(
            array.values(),
            child_plan,
            child_take_indices.as_ref(),
            child_interleave_indices.as_deref(),
        )?
    };

    let mut nulls = NullBufferBuilder::new(output_len);
    for (output_row, (source, _)) in interleave_indices.iter().enumerate() {
        let input_row = take_indices.value(output_row) as usize;
        nulls.append(match source {
            0 => array.is_valid(input_row),
            // When a null blank is unavailable, preserve row zero's validity so
            // any nullable children copied by `Take` remain masked.
            1 => !blank_is_null && array.is_valid(input_row),
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "fixed-size-list blank plan has invalid source {source}"
                )));
            }
        });
    }

    Ok(Arc::new(FixedSizeListArray::try_new_with_length(
        field,
        list_size,
        values,
        nulls.finish(),
        output_len,
    )?))
}

fn apply_blank_plan(
    array: &ArrayRef,
    plan: &BlankPlan,
    take_indices: Option<&UInt32Array>,
    interleave_indices: Option<&[(usize, usize)]>,
) -> std::result::Result<ArrayRef, ArrowError> {
    match plan {
        BlankPlan::Take => arrow::compute::take(
            array.as_ref(),
            take_indices.ok_or_else(|| {
                ArrowError::InvalidArgumentError("take indices are missing".to_string())
            })?,
            None,
        ),
        BlankPlan::Interleave(filler) => interleave(
            &[array.as_ref(), filler.as_ref()],
            interleave_indices.ok_or_else(|| {
                ArrowError::InvalidArgumentError("interleave indices are missing".to_string())
            })?,
        ),
        BlankPlan::List { blank_is_null } => match array.data_type() {
            DataType::List(_) => apply_list_blank_plan(
                array.as_list::<i32>(),
                *blank_is_null,
                interleave_indices.ok_or_else(|| {
                    ArrowError::InvalidArgumentError("interleave indices are missing".to_string())
                })?,
            ),
            DataType::LargeList(_) => apply_list_blank_plan(
                array.as_list::<i64>(),
                *blank_is_null,
                interleave_indices.ok_or_else(|| {
                    ArrowError::InvalidArgumentError("interleave indices are missing".to_string())
                })?,
            ),
            data_type => Err(ArrowError::InvalidArgumentError(format!(
                "list blank plan requires a list array, got {data_type}"
            ))),
        },
        BlankPlan::Map { blank_is_null } => {
            let array = array.as_map_opt().ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "map blank plan requires a map array, got {}",
                    array.data_type()
                ))
            })?;
            apply_map_blank_plan(
                array,
                *blank_is_null,
                interleave_indices.ok_or_else(|| {
                    ArrowError::InvalidArgumentError("interleave indices are missing".to_string())
                })?,
            )
        }
        BlankPlan::FixedSizeList {
            child,
            blank_is_null,
        } => {
            let array = array.as_fixed_size_list_opt().ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "fixed-size-list blank plan requires a FixedSizeList array, got {}",
                    array.data_type()
                ))
            })?;
            apply_fixed_size_list_blank_plan(
                array,
                child,
                *blank_is_null,
                take_indices.ok_or_else(|| {
                    ArrowError::InvalidArgumentError("take indices are missing".to_string())
                })?,
                interleave_indices.ok_or_else(|| {
                    ArrowError::InvalidArgumentError("interleave indices are missing".to_string())
                })?,
            )
        }
        BlankPlan::Struct(plans) => {
            let struct_array = array.as_struct_opt().ok_or_else(|| {
                ArrowError::InvalidArgumentError(format!(
                    "struct blank plan requires a struct array, got {}",
                    array.data_type()
                ))
            })?;
            let children = struct_array
                .columns()
                .iter()
                .zip(plans)
                .map(|(array, plan)| {
                    apply_blank_plan(array, plan, take_indices, interleave_indices)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;

            let indices = take_indices.ok_or_else(|| {
                ArrowError::InvalidArgumentError("take indices are missing".to_string())
            })?;
            let mut nulls = NullBufferBuilder::new(indices.len());
            for row in indices.values() {
                // A blank copies row zero for every child on the `Take` path, so
                // its parent validity must match that same row. This keeps child
                // nulls masked exactly as they were in the input struct.
                nulls.append(struct_array.is_valid(*row as usize));
            }
            // Supply the length: a struct with no fields has no child to take it from,
            // and `NullBufferBuilder::finish` yields `None` when every row is valid.
            Ok(Arc::new(StructArray::try_new_with_length(
                struct_array.fields().clone(),
                children,
                nulls.finish(),
                indices.len(),
            )?))
        }
    }
}

/// Choose how to add a blank to a column, recursing into nested children.
///
/// Fixed-width and dictionary arrays use `take`: copying a key or fixed-width value
/// costs no more than any synthetic replacement, and dictionary values remain shared.
/// Variable-width arrays take a physical null where the column allows one, and an empty
/// value where it does not. Variable-size containers only rebuild their offsets and
/// preserve their children. Structs and fixed-size lists plan children independently, so
/// dictionary values stay shared while variable-width siblings are still shrunk.
///
/// Null beats empty for a nullable column even though both cost nothing to store: an
/// empty value is a real value, and a reader is entitled to interpret it. A blob v2
/// column, for instance, is a struct of `data` and `uri`, and an empty `uri` reads as an
/// external reference to nowhere rather than as "no blob".
///
/// `arrow_field` supplies the input nullability and logical-type metadata.
/// `write_field` carries the dataset's nullability contract when one is known. Both
/// must allow nulls, recursively, so a permissive update stream cannot introduce a
/// null into a non-nullable nested dataset field. Logical extension types are checked
/// against the physical type the writer will receive.
///
/// A struct never takes a null itself, even when it could: its children already hold the
/// cheapest value each of them can, so nulling the parent would save nothing and would
/// discard row zero's validity, which `apply_blank_plan` preserves.
/// Choose the blank for one column. Never reached for legacy (v1) files (see
/// [`blank_plans`]); v2.0+ can store a null for every type this plans a null for.
fn blank_plan(
    arrow_field: &ArrowField,
    write_field: Option<&LanceField>,
    array: &ArrayRef,
) -> Result<BlankPlan> {
    const EMPTY: &[u8] = &[];
    let is_nullable = arrow_field.is_nullable() && write_field.is_none_or(|field| field.nullable);
    // A blob v2 column cannot take the generic struct blank. Its `data` and `uri` children
    // are both declared nullable, so that plan would null both -- a *valid* struct with null
    // children, which the preprocessor rejects ("must set exactly one of `data` and `uri`").
    // A non-nullable column additionally cannot hold the null blob that "no data and no uri"
    // would otherwise mean. The empty-inline descriptor sidesteps both, on every version and
    // for either nullability, so blob-ness alone selects it. Ask it of both schemas: the
    // planner sees the caller's batch field, while the preprocessor decides from the write
    // schema, and an untagged batch field survives `Field::project_by_field`, which
    // short-circuits on blob fields without comparing metadata.
    if arrow_field.is_blob_v2() || write_field.is_some_and(|field| field.is_blob_v2()) {
        return Ok(blob_blank_plan(array));
    }
    // A nullable column takes a null blank, a non-nullable one takes the empty value. No
    // per-version null check is needed: v1 is already excluded, and the only null v2.0
    // forbids is a struct-level one, which the `Struct` arm never produces (it mirrors
    // row zero's validity). Every null blank below is for a non-struct type, which v2.0+
    // stores. The arrow.json physical type (LargeBinary) also stores nulls on these
    // versions, so there is no carve-out to make for it either.
    let can_be_null = is_nullable;
    let filler = |empty: ArrayRef| {
        BlankPlan::Interleave(if can_be_null {
            new_null_array(array.data_type(), 1)
        } else {
            empty
        })
    };
    let plan = match array.data_type() {
        DataType::Utf8 => filler(Arc::new(StringArray::from(vec![""]))),
        DataType::LargeUtf8 => filler(Arc::new(LargeStringArray::from(vec![""]))),
        DataType::Utf8View => filler(Arc::new(StringViewArray::from(vec![""]))),
        DataType::Binary => filler(Arc::new(BinaryArray::from(vec![EMPTY]))),
        DataType::LargeBinary => filler(Arc::new(LargeBinaryArray::from(vec![EMPTY]))),
        DataType::BinaryView => filler(Arc::new(BinaryViewArray::from(vec![EMPTY]))),
        DataType::List(_) | DataType::LargeList(_) => BlankPlan::List {
            blank_is_null: can_be_null,
        },
        DataType::Map(entries_field, _) => {
            // The blank path rebuilds the map with `MapArray::try_new`, which rejects a
            // nullable entries field (arrow 58: `field.is_nullable() || ...`). The Arrow
            // spec forbids such a field, but Lance's `Field` validation lets it through,
            // and the old `take` path never checked. Fall back to copying row zero so
            // those maps keep working instead of newly failing on a deleted fragment.
            if entries_field.is_nullable() {
                BlankPlan::Take
            } else {
                BlankPlan::Map {
                    blank_is_null: can_be_null,
                }
            }
        }
        DataType::FixedSizeList(child_field, _) => {
            let child = blank_plan(
                child_field,
                write_child_field(write_field, child_field.name()),
                array.as_fixed_size_list().values(),
            )?;
            if matches!(child, BlankPlan::Take) {
                BlankPlan::Take
            } else {
                BlankPlan::FixedSizeList {
                    child: Box::new(child),
                    blank_is_null: can_be_null,
                }
            }
        }
        DataType::Struct(fields) => {
            let plans = fields
                .iter()
                .zip(array.as_struct().columns())
                .map(|(field, child)| {
                    let write_child = required_write_child_field(write_field, field.name())?;
                    blank_plan(field, write_child, child)
                })
                .collect::<Result<Vec<_>>>()?;
            if plans.iter().all(|plan| matches!(plan, BlankPlan::Take)) {
                BlankPlan::Take
            } else {
                BlankPlan::Struct(plans)
            }
        }
        // These layouts spend the same bytes on every logical value. In
        // particular, dictionary arrays only need their fixed-width keys taken;
        // Arrow's take kernel shares the values array.
        _ => BlankPlan::Take,
    };
    Ok(plan)
}

/// Plan the blank for a blob v2 column, nullable or not.
///
/// The cheapest descriptor a blob column can hold is the *empty inline blob*: `data`
/// present and zero length, `uri` absent. The preprocessor routes that to
/// `push_inline(b"")` -- below both the inline and dedicated thresholds -- so it consumes
/// no blob id and writes no sidecar bytes, which is the whole point of not copying row
/// zero. `BlobArrayBuilder::push_empty` builds exactly this shape. It sets exactly one of
/// `data`/`uri` and never nulls the struct, so it satisfies the preprocessor's contract on
/// every file version and regardless of the column's nullability.
///
/// Three layouts must not take it. A *prepared* or stored *descriptor* struct carries a
/// `kind` discriminant, and `validate_prepared_blob_array` rejects a row whose `kind`
/// says inline while `data` is absent -- and the generic plan would `Take` `kind` from
/// row zero while nulling `data`. A struct with a non-nullable non-`data` child (a
/// `position`/`size` the schema pins as required) cannot be nulled down to `data` alone.
/// Anything that is not the logical `{data, uri, ..}` shape falls back to copying row
/// zero, which is what every blank did before this optimization and is always a valid
/// descriptor.
fn blob_blank_plan(array: &ArrayRef) -> BlankPlan {
    let Some(fields) = array.as_struct_opt().map(|array| array.fields().clone()) else {
        return BlankPlan::Take;
    };
    // Require a `data` child of `LargeBinary` and no `kind` discriminant. Anything else
    // copies row zero.
    let Some(data_index) = fields
        .iter()
        .position(|field| field.name() == "data" && field.data_type() == &DataType::LargeBinary)
    else {
        return BlankPlan::Take;
    };
    if fields.iter().any(|field| field.name() == "kind") {
        return BlankPlan::Take;
    }
    // The empty-inline descriptor is a valid blob row only when every non-`data` child is
    // null, so the row sets `data` alone. A non-nullable non-`data` child cannot be nulled,
    // and taking it from row zero would leave, say, `position`/`size` set while `uri` is
    // null -- a descriptor the blob writer rejects. Fall back to copying row zero for that
    // shape: the pre-optimization blank, which is always a valid descriptor.
    if fields
        .iter()
        .enumerate()
        .any(|(index, field)| index != data_index && !field.is_nullable())
    {
        return BlankPlan::Take;
    }
    let plans = fields
        .iter()
        .enumerate()
        .map(|(index, field)| {
            if index == data_index {
                // Present and empty. Absent `data` together with absent `uri` is how the
                // preprocessor spells a null blob, which this column cannot hold.
                BlankPlan::Interleave(Arc::new(LargeBinaryArray::from(vec![&[] as &[u8]])))
            } else {
                // Every non-`data` child is nullable here, so the blank nulls it and sets
                // `data` alone -- exactly one of `data`/`uri`, with no `position`/`size`.
                BlankPlan::Interleave(new_null_array(field.data_type(), 1))
            }
        })
        .collect();
    BlankPlan::Struct(plans)
}

/// Find a nested write-schema field by its literal name, if the schema models it.
///
/// Field names are not parsed as paths here: nested names may legally contain dots or
/// backticks, just like the top-level names handled by [`blank_plans`].
///
/// `None` means "the write schema places no constraint on this child", which is not an
/// error. A Lance `Field` only gets children for a fixed-size list when the item is a
/// struct, so a `FixedSizeList<Float32>` -- an embedding column -- has none at all, and
/// its item nullability is not persisted; it is rebuilt as nullable from the logical type
/// string. Falling back to the Arrow field is the only declaration there is. For the
/// shapes that are modelled, the writer's own recursive null check is the backstop.
///
/// `Schema::validate` rejects duplicate names only among top-level fields, so a struct
/// could in principle declare two children with the same name and the first would win
/// here. No public API builds such a schema, and the consequence would be bounded to a
/// blank at a deleted row's slot, so this resolves by first match rather than erroring.
fn write_child_field<'a>(
    write_parent: Option<&'a LanceField>,
    child_name: &str,
) -> Option<&'a LanceField> {
    write_parent?
        .children
        .iter()
        .find(|child| child.name == child_name)
}

/// Resolve a child for a container whose children are always represented in a Lance
/// schema. A missing child here is a schema mismatch, not an absent constraint.
fn required_write_child_field<'a>(
    write_parent: Option<&'a LanceField>,
    child_name: &str,
) -> Result<Option<&'a LanceField>> {
    let Some(write_parent) = write_parent else {
        return Ok(None);
    };
    write_child_field(Some(write_parent), child_name)
        .map(Some)
        .ok_or_else(|| {
            Error::internal(format!(
                "Fragment Updater: child field {child_name} is missing from write-schema field {}",
                write_parent.name
            ))
        })
}
