// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sparse structural planning and serialization.
//!
//! Value storage is selected independently by the planar encoder.
//! Normalization is shared with dense planning; sparse plans are built only after selection.
//! Validity polarity compares inline descriptors first, then bounds codec analysis
//! to the minority side for strongly imbalanced non-symbolic sets. Otherwise it
//! compares estimated wire cost; ties use null positions. These are writer policies.

use crate::encodings::physical::{
    constant::ConstantEncoder,
    range::RangeEncoder,
    sequence::{BlockCost, select_block_codec},
};
use prost::Message;

use arrow_buffer::BooleanBuffer;
use lance_core::{Error, Result};

use crate::{
    buffer::LanceBuffer,
    data::{BlockInfo, FixedWidthDataBlock},
    format::{metadata::CompressiveEncoding, pb23},
    repdef::{NormalizedStructuralLayer, NormalizedStructuralPlan},
};

use super::{
    SparseCountPlan, SparsePositionPlan, SparseStructuralLayerPlan, SparseStructuralPlan,
    SparseValidityMeaning, SparseValiditySet,
};

#[derive(Clone, Copy, Default)]
struct PositionSetStats {
    count: u64,
    first: u64,
    last: u64,
    step: Option<u64>,
}

impl PositionSetStats {
    fn observe(&mut self, position: u64) {
        match self.count {
            0 => self.first = position,
            1 => self.step = Some(position - self.last),
            _ if self.step != Some(position - self.last) => self.step = None,
            _ => {}
        }
        self.last = position;
        self.count += 1;
    }

    fn observe_range(&mut self, start: u64, end: u64) {
        if start == end {
            return;
        }
        self.observe(start);
        if end - start > 1 {
            self.observe(start + 1);
            if end - start > 2 {
                if self.step != Some(1) {
                    self.step = None;
                }
                self.count += end - start - 2;
                self.last = end - 1;
            }
        }
    }

    fn inline_plan(self, domain_len: u64) -> Option<SparsePositionPlan> {
        if self.count == 0 {
            Some(SparsePositionPlan::empty())
        } else if self.count == domain_len {
            Some(SparsePositionPlan::all(domain_len))
        } else if self.count == 1 || self.step == Some(1) {
            Some(SparsePositionPlan::range(self.first, self.count))
        } else {
            self.step.map(|step| SparsePositionPlan::Arithmetic {
                start: self.first,
                step,
                len: self.count,
            })
        }
    }

    fn wire_cost(self, validity: &BooleanBuffer, want_valid: bool, domain_len: u64) -> Result<u64> {
        let (encoding, payload_cost) = if let Some(plan) = self.inline_plan(domain_len) {
            (encode_position_set(&plan)?.1, 0)
        } else {
            let inverse;
            let selected = if want_valid {
                validity
            } else {
                inverse = !validity;
                &inverse
            };
            let values = selected.set_indices().map(|i| i as u64);
            let codec = crate::encodings::physical::sequence::select_u64_sequence_codec(
                values,
                &Default::default(),
                STRUCTURAL_BLOCK_COST,
            )?;
            let encoding = codec.expected_encoding().clone();
            let payload_cost = codec.estimated_wire_bytes() - encoding.encoded_len() as u64;
            (encoding, payload_cost)
        };
        let descriptor = pb23::SparseValiditySet {
            meaning: pb23::sparse_validity_set::Meaning::SparseValidityNullPositions as i32,
            positions: Some(encoding),
            num_positions: self.count,
        };
        Ok((descriptor.encoded_len() as u64).saturating_add(payload_cost))
    }

    fn to_plan(
        self,
        validity: &BooleanBuffer,
        want_valid: bool,
        domain_len: u64,
        label: &str,
    ) -> Result<SparsePositionPlan> {
        if let Some(plan) = self.inline_plan(domain_len) {
            return Ok(plan);
        }

        let capacity = usize::try_from(self.count).map_err(|_| {
            Error::invalid_input_source(
                format!("Sparse structural {label} positions exceed usize::MAX").into(),
            )
        })?;
        let inverse;
        let selected = if want_valid {
            validity
        } else {
            inverse = !validity;
            &inverse
        };
        let mut positions = Vec::with_capacity(capacity);
        positions.extend(selected.set_indices().map(|index| index as u64));
        SparsePositionPlan::from_positions(positions, domain_len, label)
    }
}

fn usize_to_u64(value: usize, label: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| {
        Error::invalid_input_source(
            format!("Sparse structural {label} {value} exceeds u64::MAX").into(),
        )
    })
}

fn validity_set(
    validity: Option<&BooleanBuffer>,
    num_slots: usize,
    label: &str,
) -> Result<SparseValiditySet> {
    let Some(validity) = validity else {
        return Ok(SparseValiditySet {
            meaning: SparseValidityMeaning::NullPositions,
            positions: SparsePositionPlan::empty(),
        });
    };
    if validity.len() != num_slots {
        return Err(Error::invalid_input_source(
            format!(
                "Sparse structural {label} validity length {} does not match {} slots",
                validity.len(),
                num_slots
            )
            .into(),
        ));
    }

    let domain_len = usize_to_u64(num_slots, "validity domain")?;
    let mut valid_stats = PositionSetStats::default();
    let mut null_stats = PositionSetStats::default();
    let mut previous_end = 0;
    for (start, end) in validity.set_slices() {
        valid_stats.observe_range(start as u64, end as u64);
        null_stats.observe_range(previous_end, start as u64);
        previous_end = end as u64;
    }
    null_stats.observe_range(previous_end, domain_len);

    if null_stats.count == 0 {
        return Ok(SparseValiditySet {
            meaning: SparseValidityMeaning::NullPositions,
            positions: SparsePositionPlan::empty(),
        });
    }
    if valid_stats.count == 0 {
        return Ok(SparseValiditySet {
            meaning: SparseValidityMeaning::ValidPositions,
            positions: SparsePositionPlan::empty(),
        });
    }

    // A non-symbolic sequence needs at least one aligned payload buffer.
    // An inline descriptor below that bound wins without scanning its complement.
    let payload_lower_bound = 16 + crate::encoder::MIN_PAGE_BUFFER_ALIGNMENT;
    let valid_inline_cost = valid_stats
        .inline_plan(domain_len)
        .map(|_| valid_stats.wire_cost(validity, true, domain_len))
        .transpose()?;
    let null_inline_cost = null_stats
        .inline_plan(domain_len)
        .map(|_| null_stats.wire_cost(validity, false, domain_len))
        .transpose()?;
    let prefer_valid = match (valid_inline_cost, null_inline_cost) {
        (Some(valid), None) if valid < payload_lower_bound => true,
        (None, Some(null)) if null <= payload_lower_bound => false,
        (Some(valid), Some(null)) => valid < null,
        // Walking a nearly dense position sequence defeats sparse planning.
        // Symbolic dense sequences were already considered above.
        (None, None) if valid_stats.count < null_stats.count / 8 => true,
        (None, None) if null_stats.count < valid_stats.count / 8 => false,
        _ => {
            valid_stats.wire_cost(validity, true, domain_len)?
                < null_stats.wire_cost(validity, false, domain_len)?
        }
    };
    if prefer_valid {
        Ok(SparseValiditySet {
            meaning: SparseValidityMeaning::ValidPositions,
            positions: valid_stats.to_plan(validity, true, domain_len, label)?,
        })
    } else {
        Ok(SparseValiditySet {
            meaning: SparseValidityMeaning::NullPositions,
            positions: null_stats.to_plan(validity, false, domain_len, label)?,
        })
    }
}

/// Builds the semantic sparse plan directly from the once-normalized Arrow layers.
pub(in crate::encodings::logical::primitive) fn plan(
    normalized: &NormalizedStructuralPlan,
    num_visible_items: u64,
) -> Result<SparseStructuralPlan> {
    let mut layers = Vec::with_capacity(normalized.layers().len());

    for layer in normalized.layers() {
        match layer {
            NormalizedStructuralLayer::Validity {
                validity,
                num_slots,
            } => {
                layers.push(SparseStructuralLayerPlan::Validity {
                    num_slots: usize_to_u64(num_slots, "validity slot count")?,
                    validity: validity_set(validity, num_slots, "validity")?,
                });
            }
            NormalizedStructuralLayer::FixedSizeList {
                validity,
                dimension,
                num_slots,
            } => {
                if dimension == 0 {
                    return Err(Error::invalid_input_source(
                        "Sparse structural fixed-size-list dimension is zero".into(),
                    ));
                }
                layers.push(SparseStructuralLayerPlan::FixedSizeList {
                    num_slots: usize_to_u64(num_slots, "fixed-size-list slot count")?,
                    dimension: usize_to_u64(dimension, "fixed-size-list dimension")?,
                    validity: validity_set(validity, num_slots, "fixed-size-list validity")?,
                });
            }
            NormalizedStructuralLayer::List {
                offsets,
                validity,
                num_slots,
            } => {
                let expected_offsets = num_slots.checked_add(1).ok_or_else(|| {
                    Error::invalid_input_source(
                        "Sparse structural list offset count overflows".into(),
                    )
                })?;
                if offsets.len() != expected_offsets {
                    return Err(Error::invalid_input_source(
                        format!(
                            "Sparse structural list has {} offsets for {} slots",
                            offsets.len(),
                            num_slots
                        )
                        .into(),
                    ));
                }
                if offsets.first().copied() != Some(0) {
                    return Err(Error::invalid_input_source(
                        "Sparse structural list offsets must start at zero".into(),
                    ));
                }

                let mut non_empty_positions = Vec::new();
                let mut counts = Vec::new();
                for slot in 0..num_slots {
                    let start = offsets[slot];
                    let end = offsets[slot + 1];
                    let count = end.checked_sub(start).ok_or_else(|| {
                        Error::invalid_input_source(
                            format!(
                                "Sparse structural list offsets decrease at slot {slot}: {start}..{end}"
                            )
                            .into(),
                        )
                    })?;
                    let is_valid = validity.is_none_or(|validity| validity.value(slot));
                    if !is_valid && count != 0 {
                        return Err(Error::invalid_input_source(
                            format!(
                                "Sparse structural null list slot {slot} has {count} child slots"
                            )
                            .into(),
                        ));
                    }
                    if is_valid && count > 0 {
                        non_empty_positions.push(usize_to_u64(slot, "list position")?);
                        counts.push(u64::try_from(count).map_err(|_| {
                            Error::invalid_input_source(
                                format!("Sparse structural list count {count} exceeds u64::MAX")
                                    .into(),
                            )
                        })?);
                    }
                }

                let num_slots_u64 = usize_to_u64(num_slots, "list slot count")?;
                let num_child_slots = offsets.last().copied().ok_or_else(|| {
                    Error::invalid_input_source("Sparse structural list has no offsets".into())
                })?;
                let num_child_slots = u64::try_from(num_child_slots).map_err(|_| {
                    Error::invalid_input_source(
                        format!(
                            "Sparse structural list child slot count {num_child_slots} is negative"
                        )
                        .into(),
                    )
                })?;
                layers.push(SparseStructuralLayerPlan::List {
                    num_slots: num_slots_u64,
                    num_child_slots,
                    non_empty_positions: SparsePositionPlan::from_positions(
                        non_empty_positions,
                        num_slots_u64,
                        "list non-empty",
                    )?,
                    counts: SparseCountPlan::from_counts(counts)?,
                    validity: validity_set(validity, num_slots, "list validity")?,
                });
            }
        }
    }

    let row_domain = match layers.first() {
        Some(SparseStructuralLayerPlan::Validity { num_slots, .. })
        | Some(SparseStructuralLayerPlan::List { num_slots, .. })
        | Some(SparseStructuralLayerPlan::FixedSizeList { num_slots, .. }) => *num_slots,
        None => {
            return Err(Error::invalid_input_source(
                "Sparse structural encoding requires at least one Arrow structural layer".into(),
            ));
        }
    };
    let plan = SparseStructuralPlan {
        layers,
        num_visible_items,
    };
    plan.validate(row_domain)?;
    Ok(plan)
}

/// Encoded structure, independent of the page value plane.
pub(super) struct EncodedStructuralPlan {
    pub(super) layers: Vec<pb23::SparseStructuralLayer>,
    pub(super) buffers: Vec<LanceBuffer>,
}

const STRUCTURAL_BLOCK_COST: BlockCost =
    BlockCost::new(16, crate::encoder::MIN_PAGE_BUFFER_ALIGNMENT);

fn encode_u64_values(values: Vec<u64>) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
    let len = values.len();
    let block = FixedWidthDataBlock {
        data: LanceBuffer::reinterpret_vec(values),
        bits_per_value: 64,
        num_values: usize_to_u64(len, "u64 value count")?,
        block_info: BlockInfo::new(),
    };
    let codec = select_block_codec(
        &block,
        &[0..len],
        &Default::default(),
        STRUCTURAL_BLOCK_COST,
        false,
    )?;
    codec.compress(block)
}

fn encode_position_set(
    positions: &SparsePositionPlan,
) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
    Ok(match positions {
        SparsePositionPlan::Empty => (None, ConstantEncoder::new(64, 0).encoding(0)?),
        SparsePositionPlan::All { len } => (None, RangeEncoder::new(64, 0, 1).encoding(*len)?),
        SparsePositionPlan::Range { start, len } => {
            (None, RangeEncoder::new(64, *start, 1).encoding(*len)?)
        }
        SparsePositionPlan::Arithmetic { start, step, len } => {
            (None, RangeEncoder::new(64, *start, *step).encoding(*len)?)
        }
        SparsePositionPlan::Explicit(values) => return encode_u64_values(values.clone()),
    })
}

fn encode_count_set(
    counts: &SparseCountPlan,
) -> Result<(Option<LanceBuffer>, CompressiveEncoding)> {
    Ok(match counts {
        SparseCountPlan::Empty => (None, ConstantEncoder::new(64, 0).encoding(0)?),
        SparseCountPlan::Constant { value, len } => {
            (None, ConstantEncoder::new(64, *value).encoding(*len)?)
        }
        SparseCountPlan::Explicit { counts, .. } => return encode_u64_values(counts.to_vec()),
    })
}

fn encode_validity_set(
    validity: &SparseValiditySet,
) -> Result<(Option<LanceBuffer>, pb23::SparseValiditySet)> {
    let (buffer, positions) = encode_position_set(&validity.positions)?;
    let meaning = match validity.meaning {
        SparseValidityMeaning::NullPositions => {
            pb23::sparse_validity_set::Meaning::SparseValidityNullPositions
        }
        SparseValidityMeaning::ValidPositions => {
            pb23::sparse_validity_set::Meaning::SparseValidityValidPositions
        }
    };
    Ok((
        buffer,
        pb23::SparseValiditySet {
            meaning: meaning as i32,
            positions: Some(positions),
            num_positions: validity.positions.len(),
        },
    ))
}

// Structural plans are validated by plan() before entering the writer.
pub(super) fn encode_structural_plan(plan: &SparseStructuralPlan) -> Result<EncodedStructuralPlan> {
    let mut layers = Vec::with_capacity(plan.layers.len());
    let mut buffers = Vec::new();

    for layer in &plan.layers {
        match layer {
            SparseStructuralLayerPlan::Validity { validity, .. } => {
                let (validity_buffer, validity) = encode_validity_set(validity)?;
                buffers.extend(validity_buffer);
                layers.push(pb23::SparseStructuralLayer {
                    layer: Some(pb23::sparse_structural_layer::Layer::Validity(validity)),
                });
            }
            SparseStructuralLayerPlan::List {
                num_child_slots,
                non_empty_positions,
                counts,
                validity,
                ..
            } => {
                let num_non_empty_positions = non_empty_positions.len();
                let (position_buffer, non_empty_positions) =
                    encode_position_set(non_empty_positions)?;
                buffers.extend(position_buffer);
                let (count_buffer, counts) = encode_count_set(counts)?;
                buffers.extend(count_buffer);
                let (validity_buffer, validity) = encode_validity_set(validity)?;
                buffers.extend(validity_buffer);
                layers.push(pb23::SparseStructuralLayer {
                    layer: Some(pb23::sparse_structural_layer::Layer::List(
                        pb23::SparseListLayer {
                            num_child_slots: *num_child_slots,
                            non_empty_positions: Some(non_empty_positions),
                            num_non_empty_positions,
                            counts: Some(counts),
                            validity: Some(validity),
                        },
                    )),
                });
            }
            SparseStructuralLayerPlan::FixedSizeList {
                dimension,
                validity,
                ..
            } => {
                let (validity_buffer, validity) = encode_validity_set(validity)?;
                buffers.extend(validity_buffer);
                layers.push(pb23::SparseStructuralLayer {
                    layer: Some(pb23::sparse_structural_layer::Layer::FixedSizeList(
                        pb23::SparseFixedSizeListLayer {
                            dimension: *dimension,
                            validity: Some(validity),
                        },
                    )),
                });
            }
        }
    }

    Ok(EncodedStructuralPlan { layers, buffers })
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use arrow_array::{
        Array, ArrayRef, DictionaryArray, FixedSizeBinaryArray, FixedSizeListArray, Int8Array,
        Int32Array, LargeListArray, ListArray, StringArray, StructArray,
        builder::{Int32Builder, MapBuilder, StringBuilder},
        types::Int8Type,
    };
    use arrow_buffer::{BooleanBuffer, NullBuffer, OffsetBuffer, ScalarBuffer};
    use arrow_schema::{DataType, Field as ArrowField, Fields};

    use crate::{
        constants::{
            PACKED_STRUCT_META_KEY, STRUCTURAL_ENCODING_FULLZIP, STRUCTURAL_ENCODING_META_KEY,
            STRUCTURAL_ENCODING_MINIBLOCK, STRUCTURAL_ENCODING_SPARSE,
        },
        encoder::{
            ColumnIndexSequence, EncodingOptions, FieldEncoder, MIN_PAGE_BUFFER_ALIGNMENT,
            OutOfLineBuffers,
        },
        testing::{
            TestCases, TestEncoding, check_round_trip_encoding_of_data, test_encoding_strategy,
        },
    };

    use super::*;
    use crate::{decoder::PageEncoding, encoder::EncodedPage, format::metadata};
    use lance_core::datatypes::Field;

    fn sparse_metadata() -> HashMap<String, String> {
        HashMap::from([(
            STRUCTURAL_ENCODING_META_KEY.to_string(),
            STRUCTURAL_ENCODING_SPARSE.to_string(),
        )])
    }

    #[test]
    fn validity_cost_includes_inline_metadata_and_regular_stride() {
        // Regular stride needs no payload.
        let bitmap = BooleanBuffer::from_iter((0..1024).map(|i| i % 3 == 0));
        let selected = validity_set(Some(&bitmap), bitmap.len(), "test").unwrap();
        assert_eq!(selected.meaning, SparseValidityMeaning::ValidPositions);
        assert!(matches!(
            selected.positions,
            SparsePositionPlan::Arithmetic {
                start: 0,
                step: 3,
                ..
            }
        ));
        let (payload, descriptor) = encode_validity_set(&selected).unwrap();
        assert!(payload.is_none());
        assert_eq!(descriptor.num_positions, 342);
        // Both sides are ranges, but start=0 costs less metadata despite its larger cardinality.
        let bitmap = BooleanBuffer::from_iter((0..1024).map(|i| i < 900));
        let selected = validity_set(Some(&bitmap), bitmap.len(), "test").unwrap();
        assert_eq!(selected.meaning, SparseValidityMeaning::ValidPositions);
    }

    fn structural_metadata(value: &str) -> HashMap<String, String> {
        HashMap::from([(STRUCTURAL_ENCODING_META_KEY.to_string(), value.to_string())])
    }

    fn null_buffer(validity: impl IntoIterator<Item = bool>) -> NullBuffer {
        NullBuffer::new(BooleanBuffer::from_iter(validity))
    }

    fn list_i32(offsets: Vec<i32>, validity: Option<Vec<bool>>) -> ArrayRef {
        let num_values = offsets.last().copied().unwrap_or_default();
        let values = Arc::new(Int32Array::from_iter_values(0..num_values)) as ArrayRef;
        Arc::new(
            ListArray::try_new(
                Arc::new(ArrowField::new("item", DataType::Int32, true)),
                OffsetBuffer::new(ScalarBuffer::from(offsets)),
                values,
                validity.map(null_buffer),
            )
            .unwrap(),
        )
    }

    fn sparse_list_values(
        num_rows: usize,
        stride: usize,
        values: ArrayRef,
        item_field: Arc<ArrowField>,
    ) -> ArrayRef {
        let mut offsets = Vec::with_capacity(num_rows + 1);
        let mut num_values = 0_i32;
        offsets.push(num_values);
        for row in 0..num_rows {
            if (row + 1).is_multiple_of(stride) || row + 1 == num_rows {
                num_values += 1;
            }
            offsets.push(num_values);
        }
        assert_eq!(values.len(), num_values as usize);
        Arc::new(
            ListArray::try_new(
                item_field,
                OffsetBuffer::new(ScalarBuffer::from(offsets)),
                values,
                None,
            )
            .unwrap(),
        )
    }

    fn sparse_i32_list(num_rows: usize, stride: usize) -> ArrayRef {
        let num_values = num_rows.div_ceil(stride);
        sparse_list_values(
            num_rows,
            stride,
            Arc::new(Int32Array::from_iter_values(0..num_values as i32)),
            Arc::new(ArrowField::new("item", DataType::Int32, true)),
        )
    }

    fn unsplittable_nested_list(values: ArrayRef, item_field: Arc<ArrowField>) -> ArrayRef {
        const NUM_INNER_LISTS: usize = 70_000;

        let mut inner_offsets = vec![0_i32; NUM_INNER_LISTS + 1];
        assert!(!values.is_empty());
        assert!(values.len() <= NUM_INNER_LISTS);
        for value_index in 1..=values.len() {
            inner_offsets[NUM_INNER_LISTS - values.len() + value_index] = value_index as i32;
        }
        let inner = Arc::new(
            ListArray::try_new(
                item_field,
                OffsetBuffer::new(ScalarBuffer::from(inner_offsets)),
                values,
                None,
            )
            .unwrap(),
        ) as ArrayRef;
        Arc::new(
            ListArray::try_new(
                Arc::new(ArrowField::new("item", inner.data_type().clone(), true)),
                OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, NUM_INNER_LISTS as i32])),
                inner,
                None,
            )
            .unwrap(),
        )
    }

    fn variable_packed_struct_values(num_values: usize) -> (ArrayRef, Arc<ArrowField>) {
        let fields = Fields::from(vec![ArrowField::new("value", DataType::Utf8, false)]);
        let values = Arc::new(StructArray::new(
            fields.clone(),
            vec![Arc::new(StringArray::from_iter_values(
                (0..num_values).map(|index| format!("value-{index}")),
            ))],
            None,
        )) as ArrayRef;
        let item_field = Arc::new(
            ArrowField::new("item", DataType::Struct(fields), true).with_metadata(HashMap::from([
                (PACKED_STRUCT_META_KEY.to_string(), "true".to_string()),
            ])),
        );
        (values, item_field)
    }

    fn dictionary_values(num_values: usize) -> (ArrayRef, Arc<ArrowField>) {
        let keys = Int8Array::from_iter_values((0..num_values).map(|index| (index % 2) as i8));
        let values = Arc::new(StringArray::from(vec!["value-0", "value-1"]));
        let dictionary = Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap());
        let item_field = Arc::new(ArrowField::new(
            "item",
            dictionary.data_type().clone(),
            true,
        ));
        (dictionary, item_field)
    }

    fn large_list_i32(offsets: Vec<i64>, validity: Option<Vec<bool>>) -> ArrayRef {
        let num_values = offsets.last().copied().unwrap_or_default();
        let values = Arc::new(Int32Array::from_iter_values(0..num_values as i32)) as ArrayRef;
        Arc::new(
            LargeListArray::try_new(
                Arc::new(ArrowField::new("item", DataType::Int32, true)),
                OffsetBuffer::new(ScalarBuffer::from(offsets)),
                values,
                validity.map(null_buffer),
            )
            .unwrap(),
        )
    }

    fn map_i32() -> ArrayRef {
        let mut builder = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        builder.keys().append_value("a");
        builder.values().append_value(1);
        builder.append(true).unwrap();
        builder.append(false).unwrap();
        builder.append(true).unwrap();
        builder.keys().append_value("b");
        builder.values().append_null();
        builder.keys().append_value("c");
        builder.values().append_value(3);
        builder.append(true).unwrap();
        Arc::new(builder.finish())
    }

    fn fixed_size_list_struct() -> ArrayRef {
        let fields = Fields::from(vec![ArrowField::new("value", DataType::Int32, true)]);
        let child = Arc::new(StructArray::new(
            fields.clone(),
            vec![Arc::new(Int32Array::from(vec![
                Some(0),
                Some(1),
                None,
                Some(3),
                Some(4),
                Some(5),
                None,
                Some(7),
                Some(8),
                Some(9),
                Some(10),
                None,
            ]))],
            Some(null_buffer([
                true, true, false, true, true, true, true, false, true, true, true, true,
            ])),
        )) as ArrayRef;
        Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(ArrowField::new("item", DataType::Struct(fields), true)),
                2,
                child,
                Some(null_buffer([true, false, true, true, false, true])),
            )
            .unwrap(),
        )
    }

    fn list_fixed_size_list_struct() -> ArrayRef {
        let struct_fields = Fields::from(vec![ArrowField::new("value", DataType::Int32, true)]);
        let structs = Arc::new(StructArray::new(
            struct_fields.clone(),
            vec![Arc::new(Int32Array::from(vec![
                Some(0),
                None,
                Some(2),
                Some(3),
                None,
                Some(5),
            ]))],
            Some(null_buffer([true, true, false, true, true, true])),
        )) as ArrayRef;
        let fixed_size_list = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(ArrowField::new(
                    "item",
                    DataType::Struct(struct_fields),
                    true,
                )),
                2,
                structs,
                Some(null_buffer([true, false, true])),
            )
            .unwrap(),
        ) as ArrayRef;
        Arc::new(
            ListArray::try_new(
                Arc::new(ArrowField::new(
                    "item",
                    fixed_size_list.data_type().clone(),
                    true,
                )),
                OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 1, 1, 3, 3])),
                fixed_size_list,
                Some(null_buffer([true, false, true, true])),
            )
            .unwrap(),
        )
    }

    fn nullable_struct() -> ArrayRef {
        let fields = Fields::from(vec![ArrowField::new("value", DataType::Int32, true)]);
        Arc::new(StructArray::new(
            fields,
            vec![Arc::new(Int32Array::from(vec![
                Some(10),
                Some(20),
                None,
                Some(40),
                Some(50),
            ]))],
            Some(null_buffer([true, false, true, true, false])),
        ))
    }

    fn deeply_nested() -> ArrayRef {
        let leaf = Arc::new(Int32Array::from(vec![
            Some(0),
            None,
            Some(2),
            Some(3),
            None,
            Some(5),
            Some(6),
            Some(7),
        ])) as ArrayRef;
        let inner = Arc::new(
            LargeListArray::try_new(
                Arc::new(ArrowField::new("item", DataType::Int32, true)),
                OffsetBuffer::new(ScalarBuffer::from(vec![0_i64, 2, 2, 3, 5, 5, 8])),
                leaf,
                Some(null_buffer([true, false, true, true, true, true])),
            )
            .unwrap(),
        ) as ArrayRef;
        let struct_fields = Fields::from(vec![ArrowField::new(
            "inner",
            inner.data_type().clone(),
            true,
        )]);
        let structs = Arc::new(StructArray::new(
            struct_fields.clone(),
            vec![inner],
            Some(null_buffer([true, true, false, true, true, true])),
        )) as ArrayRef;
        Arc::new(
            ListArray::try_new(
                Arc::new(ArrowField::new(
                    "item",
                    DataType::Struct(struct_fields),
                    true,
                )),
                OffsetBuffer::new(ScalarBuffer::from(vec![0_i32, 2, 2, 2, 4, 6])),
                structs,
                Some(null_buffer([true, false, true, true, true])),
            )
            .unwrap(),
        )
    }

    #[derive(Debug, PartialEq)]
    enum TestLayout {
        Chunked(metadata::MiniBlockLayout),
        Zipped(metadata::FullZipLayout),
        Constant(metadata::ConstantLayout),
        Sparse(pb23::SparseStructure),
    }

    fn page_layout(page: &EncodedPage) -> TestLayout {
        let dense = match &page.description {
            PageEncoding::Planar(layout) => {
                if let Some(sparse) = layout.sparse() {
                    return TestLayout::Sparse(sparse.clone());
                }
                layout.dense_execution_layout().unwrap()
            }
            PageEncoding::Structural(layout) => layout.clone(),
            _ => panic!("expected structural page"),
        };
        match dense.layout.unwrap() {
            metadata::page_layout::Layout::MiniBlockLayout(v) => TestLayout::Chunked(v),
            metadata::page_layout::Layout::FullZipLayout(v) => TestLayout::Zipped(v),
            metadata::page_layout::Layout::ConstantLayout(v) => TestLayout::Constant(v),
            _ => panic!("unexpected layout"),
        }
    }

    fn create_encoder(
        array: &ArrayRef,
        version: TestEncoding,
        metadata: HashMap<String, String>,
    ) -> Result<Box<dyn FieldEncoder>> {
        let arrow_field =
            ArrowField::new("values", array.data_type().clone(), true).with_metadata(metadata);
        let field = Field::try_from(&arrow_field)?;
        let strategy = test_encoding_strategy(version);
        let options = EncodingOptions {
            cache_bytes_per_column: 1,
            ..Default::default()
        };
        crate::testing::create_test_field_encoder(
            strategy.as_ref(),
            &field,
            &mut ColumnIndexSequence::default(),
            &options,
        )
    }

    async fn encode_pages(
        array: ArrayRef,
        version: TestEncoding,
        metadata: HashMap<String, String>,
    ) -> Result<Vec<EncodedPage>> {
        encode_chunks(vec![array], version, metadata).await
    }

    async fn encode_chunks(
        arrays: Vec<ArrayRef>,
        version: TestEncoding,
        metadata: HashMap<String, String>,
    ) -> Result<Vec<EncodedPage>> {
        let first = arrays
            .first()
            .ok_or_else(|| Error::internal("test input has no arrays".to_string()))?;
        let mut encoder = create_encoder(first, version, metadata)?;
        let mut external_buffers = OutOfLineBuffers::new(0, MIN_PAGE_BUFFER_ALIGNMENT);
        let mut pages = Vec::new();
        let mut row_number = 0_u64;
        for array in arrays {
            let num_rows = array.len() as u64;
            for task in encoder.maybe_encode(
                array,
                &mut external_buffers,
                crate::repdef::RepDefBuilder::default(),
                row_number,
                num_rows,
            )? {
                pages.push(task.await?);
            }
            row_number += num_rows;
        }
        for task in encoder.flush(&mut external_buffers)? {
            pages.push(task.await?);
        }
        for column in encoder.finish(&mut external_buffers).await? {
            pages.extend(column.final_pages);
        }
        Ok(pages)
    }

    fn sparse_layout(page: &EncodedPage) -> &pb23::SparseStructure {
        let PageEncoding::Planar(layout) = &page.description else {
            panic!("expected planar page");
        };
        layout.sparse().expect("expected sparse structure")
    }

    fn list_layer(sparse: &pb23::SparseStructure) -> &pb23::SparseListLayer {
        sparse
            .layers
            .iter()
            .find_map(|layer| match layer.layer.as_ref() {
                Some(pb23::sparse_structural_layer::Layer::List(layer)) => Some(layer),
                _ => None,
            })
            .expect("expected sparse list layer")
    }

    fn validity_layer(layer: &pb23::SparseStructuralLayer) -> Option<&pb23::SparseValiditySet> {
        match layer.layer.as_ref() {
            Some(pb23::sparse_structural_layer::Layer::Validity(layer)) => Some(layer),
            _ => None,
        }
    }

    fn fixed_size_list_dimension(layer: &pb23::SparseStructuralLayer) -> Option<u64> {
        match layer.layer.as_ref() {
            Some(pb23::sparse_structural_layer::Layer::FixedSizeList(layer)) => {
                Some(layer.dimension)
            }
            _ => None,
        }
    }

    fn planned_list(
        offsets: Vec<i32>,
        list_validity: Option<Vec<bool>>,
        leaf_validity: Option<Vec<bool>>,
    ) -> SparseStructuralPlan {
        let num_values = u64::try_from(*offsets.last().unwrap()).unwrap();
        let mut builder = crate::repdef::RepDefBuilder::default();
        assert!(!builder.add_offsets(
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            list_validity.map(null_buffer),
        ));
        if let Some(leaf_validity) = leaf_validity {
            builder.add_validity_bitmap(null_buffer(leaf_validity));
        } else {
            builder.add_no_null(num_values as usize);
        }
        let normalized = crate::repdef::RepDefBuilder::normalize(vec![builder]);
        plan(&normalized, num_values).unwrap()
    }

    fn planned_list_layer(plan: &SparseStructuralPlan) -> &SparseStructuralLayerPlan {
        plan.layers
            .iter()
            .find(|layer| matches!(layer, SparseStructuralLayerPlan::List { .. }))
            .expect("expected planned list layer")
    }

    #[test]
    fn test_semantic_position_and_count_forms() {
        let empty = planned_list(vec![0, 0, 0, 0], None, None);
        assert_eq!(empty.num_visible_items, 0);
        assert!(matches!(
            planned_list_layer(&empty),
            SparseStructuralLayerPlan::List {
                non_empty_positions: SparsePositionPlan::Empty,
                counts: SparseCountPlan::Empty,
                ..
            }
        ));

        let all = planned_list(vec![0, 2, 4, 6], None, None);
        assert!(matches!(
            planned_list_layer(&all),
            SparseStructuralLayerPlan::List {
                non_empty_positions: SparsePositionPlan::All { len: 3 },
                counts: SparseCountPlan::Constant { value: 2, len: 3 },
                ..
            }
        ));

        let range = planned_list(vec![0, 2, 4, 4, 4], None, None);
        assert!(matches!(
            planned_list_layer(&range),
            SparseStructuralLayerPlan::List {
                non_empty_positions: SparsePositionPlan::Range { start: 0, len: 2 },
                counts: SparseCountPlan::Constant { value: 2, len: 2 },
                ..
            }
        ));

        let explicit = planned_list(vec![0, 1, 1, 4, 4, 6], None, None);
        assert!(matches!(
            planned_list_layer(&explicit),
            SparseStructuralLayerPlan::List {
                non_empty_positions: SparsePositionPlan::Arithmetic { start: 0, step: 2, len: 3 },
                counts: SparseCountPlan::Explicit { counts, .. },
                ..
            } if counts.as_ref() == [1, 3, 2]
        ));
    }

    #[test]
    fn test_validity_polarity_uses_semantic_encoded_cost() {
        let mostly_valid = BooleanBuffer::from_iter([true, false, true, true, false, true]);
        let validity = validity_set(Some(&mostly_valid), mostly_valid.len(), "test").unwrap();
        assert_eq!(validity.meaning, SparseValidityMeaning::NullPositions);
        assert_eq!(validity.positions.materialize().unwrap(), vec![1, 4]);

        let mostly_null = BooleanBuffer::from_iter([false, true, false, false, true, false]);
        let validity = validity_set(Some(&mostly_null), mostly_null.len(), "test").unwrap();
        assert_eq!(validity.meaning, SparseValidityMeaning::ValidPositions);
        assert_eq!(validity.positions.materialize().unwrap(), vec![1, 4]);

        let valid_island = BooleanBuffer::from_iter([false, false, true, true, false]);
        let validity = validity_set(Some(&valid_island), valid_island.len(), "test").unwrap();
        assert_eq!(validity.meaning, SparseValidityMeaning::ValidPositions);
        assert!(matches!(
            validity.positions,
            SparsePositionPlan::Range { start: 2, len: 2 }
        ));

        let all_valid = BooleanBuffer::from_iter([true, true, true]);
        let validity = validity_set(Some(&all_valid), all_valid.len(), "test").unwrap();
        assert_eq!(validity.meaning, SparseValidityMeaning::NullPositions);
        assert!(matches!(validity.positions, SparsePositionPlan::Empty));

        let all_null = BooleanBuffer::from_iter([false, false, false]);
        let validity = validity_set(Some(&all_null), all_null.len(), "test").unwrap();
        assert_eq!(validity.meaning, SparseValidityMeaning::ValidPositions);
        assert!(matches!(validity.positions, SparsePositionPlan::Empty));
    }

    #[tokio::test]
    async fn test_explicit_sparse_nullable_primitive_roundtrip() {
        let array = Arc::new(Int32Array::from(vec![
            Some(10),
            None,
            Some(20),
            Some(30),
            None,
            Some(40),
        ])) as ArrayRef;
        let pages = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert_eq!(pages.len(), 1);
        let sparse = sparse_layout(&pages[0]);
        assert_eq!(pages[0].num_rows, 6);
        assert_eq!(sparse.layers.len(), 1);
        let validity = validity_layer(&sparse.layers[0]).unwrap();
        assert_eq!(
            validity.meaning,
            pb23::sparse_validity_set::Meaning::SparseValidityNullPositions as i32
        );
        assert!(matches!(
            validity.positions.as_ref().unwrap().compression,
            Some(metadata::compressive_encoding::Compression::Range(_))
        ));

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_range(1..5)
            .with_indices(vec![0, 2, 5]);
        check_round_trip_encoding_of_data(vec![array], &cases, sparse_metadata()).await;
    }

    #[tokio::test]
    async fn test_explicit_sparse_nullable_struct_roundtrip() {
        let array = nullable_struct();
        let pages = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert_eq!(pages.len(), 1);
        let sparse = sparse_layout(&pages[0]);
        assert_eq!(sparse.layers.len(), 2);
        assert!(
            sparse
                .layers
                .iter()
                .all(|layer| validity_layer(layer).is_some())
        );
        let struct_validity = validity_layer(&sparse.layers[0]).unwrap();
        assert_eq!(
            struct_validity.meaning,
            pb23::sparse_validity_set::Meaning::SparseValidityNullPositions as i32
        );
        assert!(matches!(
            struct_validity.positions.as_ref().unwrap().compression,
            Some(metadata::compressive_encoding::Compression::Range(_))
        ));

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_range(1..5)
            .with_indices(vec![0, 2, 4]);
        check_round_trip_encoding_of_data(vec![array], &cases, sparse_metadata()).await;
    }

    #[tokio::test]
    async fn test_explicit_sparse_struct_with_constant_and_sparse_children() {
        let fields = Fields::from(vec![
            ArrowField::new("constant", DataType::Int32, true),
            ArrowField::new("sparse", DataType::Int32, true),
        ]);
        let array = Arc::new(StructArray::new(
            fields,
            vec![
                Arc::new(Int32Array::from(vec![None::<i32>; 5])),
                Arc::new(Int32Array::from(vec![
                    Some(10),
                    Some(20),
                    None,
                    Some(40),
                    Some(50),
                ])),
            ],
            Some(null_buffer([true, false, true, true, false])),
        )) as ArrayRef;
        let pages = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert!(pages.iter().any(|page| matches!(&page.description,
            PageEncoding::Planar(layout) if matches!(layout.values.as_ref().unwrap().codec,
                Some(pb23::value_plane::Codec::None(_))))));
        assert!(
            pages
                .iter()
                .any(|page| matches!(page_layout(page), TestLayout::Sparse(_)))
        );

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_range(1..5)
            .with_indices(vec![0, 2, 4]);
        check_round_trip_encoding_of_data(vec![array], &cases, sparse_metadata()).await;
    }

    #[tokio::test]
    async fn test_explicit_sparse_emits_both_validity_polarities() {
        let mostly_valid = Arc::new(Int32Array::from(vec![
            Some(0),
            None,
            Some(2),
            Some(3),
            None,
            Some(5),
        ])) as ArrayRef;
        let mostly_null = Arc::new(Int32Array::from(vec![
            None,
            Some(1),
            None,
            None,
            Some(4),
            None,
        ])) as ArrayRef;

        let null_positions = encode_pages(
            mostly_valid.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        let validity = validity_layer(&sparse_layout(&null_positions[0]).layers[0]).unwrap();
        assert_eq!(
            validity.meaning,
            pb23::sparse_validity_set::Meaning::SparseValidityNullPositions as i32
        );

        let valid_positions = encode_pages(
            mostly_null.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        let validity = validity_layer(&sparse_layout(&valid_positions[0]).layers[0]).unwrap();
        assert_eq!(
            validity.meaning,
            pb23::sparse_validity_set::Meaning::SparseValidityValidPositions as i32
        );

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_range(1..5)
            .with_indices(vec![0, 2, 5]);
        for array in [mostly_valid, mostly_null] {
            check_round_trip_encoding_of_data(vec![array], &cases, sparse_metadata()).await;
        }
    }

    #[tokio::test]
    async fn test_explicit_sparse_nested_page_boundaries_range_and_take() {
        let nested = deeply_nested();
        let chunks = vec![nested.slice(0, 2), nested.slice(2, 3)];
        let pages = encode_chunks(
            chunks.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert!(pages.len() >= 2, "expected multiple sparse pages");
        let sparse = pages
            .iter()
            .map(sparse_layout)
            .collect::<Vec<&pb23::SparseStructure>>();
        assert!(sparse.iter().any(|layout| {
            layout
                .layers
                .iter()
                .filter(|layer| {
                    matches!(
                        layer.layer.as_ref(),
                        Some(pb23::sparse_structural_layer::Layer::List(_))
                    )
                })
                .count()
                >= 2
        }));
        assert!(sparse.iter().any(|layout| {
            layout
                .layers
                .iter()
                .any(|layer| validity_layer(layer).is_some())
        }));

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_batch_size(2)
            .with_range(1..5)
            .with_range(2..4)
            .with_indices(vec![0, 2, 4])
            .with_indices(vec![1, 3]);
        check_round_trip_encoding_of_data(chunks, &cases, sparse_metadata()).await;
    }

    #[tokio::test]
    async fn test_explicit_sparse_list_and_large_list_null_empty_roundtrip() {
        let list = list_i32(
            vec![0, 0, 0, 2, 3, 3],
            Some(vec![false, true, true, true, true]),
        );
        let large_list = large_list_i32(
            vec![0, 0, 0, 2, 3, 3],
            Some(vec![false, true, true, true, true]),
        );
        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_range(0..5)
            .with_range(1..4)
            .with_indices(vec![0, 1, 4])
            .with_indices(vec![2, 3]);

        for array in [list, large_list] {
            let pages = encode_pages(
                array.clone(),
                TestEncoding::StructuralSparse,
                sparse_metadata(),
            )
            .await
            .unwrap();
            assert!(pages.iter().map(sparse_layout).all(|layout| {
                layout.layers.iter().any(|layer| {
                    matches!(
                        layer.layer.as_ref(),
                        Some(pb23::sparse_structural_layer::Layer::List(_))
                    )
                })
            }));
            check_round_trip_encoding_of_data(vec![array], &cases, sparse_metadata()).await;
        }
    }

    #[tokio::test]
    async fn test_explicit_sparse_map_and_fixed_size_list_roundtrip() {
        let map = map_i32();
        let map_pages = encode_pages(
            map.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert!(map_pages.iter().map(sparse_layout).any(|layout| {
            layout.layers.iter().any(|layer| {
                matches!(
                    layer.layer.as_ref(),
                    Some(pb23::sparse_structural_layer::Layer::List(_))
                )
            })
        }));
        let map_cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_range(1..4)
            .with_indices(vec![0, 2, 3]);
        check_round_trip_encoding_of_data(vec![map], &map_cases, sparse_metadata()).await;

        let fsl = fixed_size_list_struct();
        let fsl_pages = encode_pages(
            fsl.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert_eq!(
            fsl_pages.iter().map(|page| page.num_rows).sum::<u64>(),
            fsl.len() as u64 * 2
        );
        assert!(fsl_pages.iter().map(sparse_layout).any(|layout| {
            layout
                .layers
                .iter()
                .any(|layer| fixed_size_list_dimension(layer) == Some(2))
        }));
        let fsl_cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_range(1..6)
            .with_indices(vec![0, 3, 5]);
        check_round_trip_encoding_of_data(vec![fsl], &fsl_cases, sparse_metadata()).await;
    }

    #[tokio::test]
    async fn test_explicit_sparse_list_fixed_size_list_struct_roundtrip() {
        let array = list_fixed_size_list_struct();
        let pages = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert!(pages.iter().map(sparse_layout).any(|layout| {
            let kinds = layout
                .layers
                .iter()
                .map(|layer| match layer.layer.as_ref().unwrap() {
                    pb23::sparse_structural_layer::Layer::Validity(_) => "validity",
                    pb23::sparse_structural_layer::Layer::List(_) => "list",
                    pb23::sparse_structural_layer::Layer::FixedSizeList(_) => "fixed-size-list",
                })
                .collect::<Vec<_>>();
            kinds.starts_with(&["list", "fixed-size-list"])
        }));
        assert_eq!(
            pages.iter().map(|page| page.num_rows).sum::<u64>(),
            array.len() as u64 * 2
        );

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_range(1..4)
            .with_indices(vec![0, 2, 3])
            .with_indices(vec![1, 3]);
        check_round_trip_encoding_of_data(vec![array], &cases, sparse_metadata()).await;
    }

    #[tokio::test]
    async fn test_explicit_sparse_serializes_semantic_list_forms() {
        let all_array = list_i32(vec![0, 2, 4, 6], None);
        let all = encode_pages(
            all_array.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        let all_layer = list_layer(sparse_layout(&all[0]));
        assert!(matches!(
            all_layer.non_empty_positions.as_ref().unwrap().compression,
            Some(metadata::compressive_encoding::Compression::Range(_))
        ));
        assert!(matches!(
            all_layer.counts.as_ref().unwrap().compression,
            Some(metadata::compressive_encoding::Compression::Constant(_))
        ));
        assert_eq!(all[0].data.len(), 2);

        let range_array = list_i32(vec![0, 2, 4, 4, 4], None);
        let range = encode_pages(
            range_array.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        let range_layer = list_layer(sparse_layout(&range[0]));
        assert!(matches!(
            range_layer
                .non_empty_positions
                .as_ref()
                .unwrap()
                .compression,
            Some(metadata::compressive_encoding::Compression::Range(_))
        ));
        assert!(matches!(
            range_layer.counts.as_ref().unwrap().compression,
            Some(metadata::compressive_encoding::Compression::Constant(_))
        ));
        assert_eq!(range[0].data.len(), 2);

        let explicit_array = list_i32(vec![0, 1, 1, 4, 4, 6], None);
        let explicit = encode_pages(
            explicit_array.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        let explicit_layer = list_layer(sparse_layout(&explicit[0]));
        assert!(matches!(
            explicit_layer
                .non_empty_positions
                .as_ref()
                .unwrap()
                .compression,
            Some(metadata::compressive_encoding::Compression::Range(_))
        ));
        assert!(
            explicit_layer
                .counts
                .as_ref()
                .unwrap()
                .compression
                .is_some()
        );
        assert_eq!(explicit[0].data.len(), 3);

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1])
            .with_range(1..3)
            .with_indices(vec![0, 2]);
        for array in [all_array, range_array, explicit_array] {
            check_round_trip_encoding_of_data(vec![array], &cases, sparse_metadata()).await;
        }
    }

    #[tokio::test]
    async fn test_constant_layout_boundary_is_explicit() {
        let structural_only = encode_pages(
            list_i32(vec![0, 0, 0, 0], None),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert!(
            matches!(&structural_only[0].description, PageEncoding::Planar(p) if matches!(p.values.as_ref().unwrap().codec, Some(pb23::value_plane::Codec::None(_))))
        );

        let empty_struct = Arc::new(StructArray::new_empty_fields(3, None)) as ArrayRef;
        let empty_struct_pages = encode_pages(
            empty_struct,
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert!(
            matches!(&empty_struct_pages[0].description, PageEncoding::Planar(p) if matches!(p.values.as_ref().unwrap().codec, Some(pb23::value_plane::Codec::None(_))))
        );

        let all_null = Arc::new(Int32Array::from(vec![None, None, None])) as ArrayRef;
        let all_null_pages =
            encode_pages(all_null, TestEncoding::StructuralSparse, sparse_metadata())
                .await
                .unwrap();
        assert!(
            matches!(&all_null_pages[0].description, PageEncoding::Planar(p) if matches!(p.values.as_ref().unwrap().codec, Some(pb23::value_plane::Codec::None(_))))
        );

        let constant = Arc::new(Int32Array::from(vec![7, 7, 7])) as ArrayRef;
        let constant_pages =
            encode_pages(constant, TestEncoding::StructuralSparse, sparse_metadata())
                .await
                .unwrap();
        assert!(matches!(
            page_layout(&constant_pages[0]),
            TestLayout::Constant(_)
        ));
    }

    #[tokio::test]
    async fn test_default_and_explicit_dense_layouts_are_unchanged() {
        let array = Arc::new(Int32Array::from_iter_values(0..16)) as ArrayRef;
        let v2_2_default = encode_pages(array.clone(), TestEncoding::StructuralU32, HashMap::new())
            .await
            .unwrap();
        let v2_2_miniblock = encode_pages(
            array.clone(),
            TestEncoding::StructuralU32,
            structural_metadata(STRUCTURAL_ENCODING_MINIBLOCK),
        )
        .await
        .unwrap();
        assert_eq!(v2_2_default.len(), v2_2_miniblock.len());
        for (default, explicit) in v2_2_default.iter().zip(v2_2_miniblock.iter()) {
            assert!(matches!(page_layout(default), TestLayout::Chunked(_)));
            assert_eq!(page_layout(default), page_layout(explicit));
            assert_eq!(default.data, explicit.data);
        }

        let v2_3_default = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            HashMap::new(),
        )
        .await
        .unwrap();
        assert!(matches!(
            page_layout(&v2_3_default[0]),
            TestLayout::Chunked(_)
        ));

        let v2_3_miniblock = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            structural_metadata(STRUCTURAL_ENCODING_MINIBLOCK),
        )
        .await
        .unwrap();
        assert!(matches!(
            page_layout(&v2_3_miniblock[0]),
            TestLayout::Chunked(_)
        ));

        let v2_3_fullzip = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            structural_metadata(STRUCTURAL_ENCODING_FULLZIP),
        )
        .await
        .unwrap();
        assert!(matches!(
            page_layout(&v2_3_fullzip[0]),
            TestLayout::Zipped(_)
        ));

        let v2_3_sparse = encode_pages(array, TestEncoding::StructuralSparse, sparse_metadata())
            .await
            .unwrap();
        assert!(matches!(
            page_layout(&v2_3_sparse[0]),
            TestLayout::Chunked(_)
        ));
    }

    #[tokio::test]
    async fn test_auto_sparse_for_split_required_page() {
        const NUM_ROWS: usize = 70_000;
        let array = sparse_i32_list(NUM_ROWS, 2_000);

        let within_budget = encode_pages(
            sparse_i32_list(4_096, 1_024),
            TestEncoding::StructuralSparse,
            HashMap::new(),
        )
        .await
        .unwrap();
        assert_eq!(within_budget.len(), 1);
        assert!(matches!(
            page_layout(&within_budget[0]),
            TestLayout::Chunked(_)
        ));

        let automatic = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            HashMap::new(),
        )
        .await
        .unwrap();
        assert_eq!(automatic.len(), 1);
        assert!(matches!(page_layout(&automatic[0]), TestLayout::Sparse(_)));

        let miniblock = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            structural_metadata(STRUCTURAL_ENCODING_MINIBLOCK),
        )
        .await
        .unwrap();
        assert!(miniblock.len() > 1);
        assert!(
            miniblock
                .iter()
                .all(|page| matches!(page_layout(page), TestLayout::Chunked(_)))
        );

        let fullzip = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            structural_metadata(STRUCTURAL_ENCODING_FULLZIP),
        )
        .await
        .unwrap();
        assert_eq!(fullzip.len(), miniblock.len());
        assert!(
            fullzip
                .iter()
                .all(|page| matches!(page_layout(page), TestLayout::Zipped(_)))
        );

        let sparse = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert_eq!(sparse.len(), 1);
        assert!(matches!(page_layout(&sparse[0]), TestLayout::Sparse(_)));

        let v2_2_default = encode_pages(array.clone(), TestEncoding::StructuralU32, HashMap::new())
            .await
            .unwrap();
        let v2_2_miniblock = encode_pages(
            array.clone(),
            TestEncoding::StructuralU32,
            structural_metadata(STRUCTURAL_ENCODING_MINIBLOCK),
        )
        .await
        .unwrap();
        assert_eq!(v2_2_default.len(), v2_2_miniblock.len());
        for (default, explicit) in v2_2_default.iter().zip(v2_2_miniblock.iter()) {
            assert_eq!(page_layout(default), page_layout(explicit));
            assert_eq!(default.data, explicit.data);
        }

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1024 * 1024])
            .with_batch_size(NUM_ROWS as u32)
            .with_range(1_999..2_002)
            .with_indices(vec![0, 1_999, 2_000, NUM_ROWS as u64 - 1]);
        check_round_trip_encoding_of_data(vec![array], &cases, HashMap::new()).await;
    }

    #[tokio::test]
    async fn test_auto_sparse_for_single_row_over_budget() {
        let array = unsplittable_nested_list(
            Arc::new(Int32Array::from(vec![42, 43])),
            Arc::new(ArrowField::new("item", DataType::Int32, true)),
        );

        let automatic = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            HashMap::new(),
        )
        .await
        .unwrap();
        assert_eq!(automatic.len(), 1);
        assert!(matches!(page_layout(&automatic[0]), TestLayout::Sparse(_)));

        let v2_2 = encode_pages(array.clone(), TestEncoding::StructuralU32, HashMap::new())
            .await
            .unwrap();
        assert_eq!(v2_2.len(), 1);
        assert!(matches!(page_layout(&v2_2[0]), TestLayout::Zipped(_)));

        let miniblock_error = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            structural_metadata(STRUCTURAL_ENCODING_MINIBLOCK),
        )
        .await
        .unwrap_err();
        assert!(
            miniblock_error
                .to_string()
                .contains("Mini-block cannot encode 70000 rep/def levels")
        );

        let fullzip = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            structural_metadata(STRUCTURAL_ENCODING_FULLZIP),
        )
        .await
        .unwrap();
        assert!(matches!(page_layout(&fullzip[0]), TestLayout::Zipped(_)));

        let explicit_sparse = encode_pages(
            array.clone(),
            TestEncoding::StructuralSparse,
            sparse_metadata(),
        )
        .await
        .unwrap();
        assert!(matches!(
            page_layout(&explicit_sparse[0]),
            TestLayout::Sparse(_)
        ));

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_range(0..1)
            .with_indices(vec![0]);
        check_round_trip_encoding_of_data(vec![array], &cases, HashMap::new()).await;
    }

    #[tokio::test]
    async fn test_auto_sparse_composes_dictionary_and_packed_values() {
        const NUM_ROWS: usize = 70_000;
        let num_values = NUM_ROWS.div_ceil(2_000);

        let (dictionary, dictionary_field) = dictionary_values(num_values);
        let dictionary_array = sparse_list_values(NUM_ROWS, 2_000, dictionary, dictionary_field);
        let dictionary_pages = encode_pages(
            dictionary_array.clone(),
            TestEncoding::StructuralSparse,
            HashMap::new(),
        )
        .await
        .unwrap();
        assert_eq!(dictionary_pages.len(), 1);
        assert!(
            dictionary_pages
                .iter()
                .all(|page| matches!(page_layout(page), TestLayout::Sparse(_)))
        );

        let (packed_values, packed_field) = variable_packed_struct_values(num_values);
        let packed_array = sparse_list_values(NUM_ROWS, 2_000, packed_values, packed_field);
        let packed_pages = encode_pages(
            packed_array.clone(),
            TestEncoding::StructuralSparse,
            HashMap::new(),
        )
        .await
        .unwrap();
        assert_eq!(packed_pages.len(), 1);
        assert!(
            packed_pages
                .iter()
                .all(|page| matches!(page_layout(page), TestLayout::Sparse(_)))
        );

        let (packed_values, packed_field) = variable_packed_struct_values(2);
        let unsplittable_packed = unsplittable_nested_list(packed_values, packed_field);
        let packed_fallback = encode_pages(
            unsplittable_packed.clone(),
            TestEncoding::StructuralSparse,
            HashMap::new(),
        )
        .await
        .unwrap();
        assert_eq!(packed_fallback.len(), 1);
        assert!(matches!(
            page_layout(&packed_fallback[0]),
            TestLayout::Sparse(_)
        ));

        let (dictionary, dictionary_field) = dictionary_values(2);
        let unsplittable_dictionary = unsplittable_nested_list(dictionary, dictionary_field);
        let v2_2_error = encode_pages(
            unsplittable_dictionary.clone(),
            TestEncoding::StructuralU32,
            HashMap::new(),
        )
        .await
        .unwrap_err();
        assert!(
            v2_2_error
                .to_string()
                .contains("Mini-block cannot encode 70000 rep/def levels")
        );
        let v2_3 = encode_pages(
            unsplittable_dictionary.clone(),
            TestEncoding::StructuralSparse,
            HashMap::new(),
        )
        .await
        .unwrap();
        assert_eq!(v2_3.len(), 1);
        assert!(matches!(page_layout(&v2_3[0]), TestLayout::Sparse(_)));
        let PageEncoding::Planar(layout) = &v2_3[0].description else {
            unreachable!()
        };
        assert!(
            matches!(layout.values.as_ref().unwrap().codec.as_ref().unwrap(), pb23::value_plane::Codec::Chunked(v) if v.dictionary.is_some())
        );
        check_round_trip_encoding_of_data(
            vec![unsplittable_dictionary],
            &TestCases::default()
                .with_encoding(TestEncoding::StructuralSparse)
                .with_page_sizes(vec![1024 * 1024])
                .with_ingest_batch_counts([1]),
            HashMap::new(),
        )
        .await;

        let cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1024 * 1024])
            .with_batch_size(NUM_ROWS as u32)
            .with_range(1_999..2_002)
            .with_indices(vec![0, 2_000, NUM_ROWS as u64 - 1]);
        check_round_trip_encoding_of_data(vec![dictionary_array], &cases, HashMap::new()).await;
        let packed_cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1024 * 1024])
            .with_batch_size(NUM_ROWS as u32);
        check_round_trip_encoding_of_data(vec![packed_array], &packed_cases, HashMap::new()).await;

        let single_row_cases = TestCases::default()
            .with_encoding(TestEncoding::StructuralSparse)
            .with_page_sizes(vec![1024 * 1024])
            .with_range(0..1)
            .with_indices(vec![0]);
        check_round_trip_encoding_of_data(
            vec![unsplittable_packed],
            &single_row_cases,
            HashMap::new(),
        )
        .await;
    }

    #[tokio::test]
    async fn test_auto_sparse_wide_values_use_zipped_storage() {
        let first = vec![0xAB_u8; 5_000];
        let second = vec![0xCD_u8; 5_000];
        let fixed_size_binary = Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                [Some(first.as_slice()), Some(second.as_slice())].into_iter(),
                5_000,
            )
            .unwrap(),
        ) as ArrayRef;

        const FSL_DIMENSION: i32 = 2_048;
        let fixed_size_list = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(ArrowField::new("item", DataType::Int32, true)),
                FSL_DIMENSION,
                Arc::new(Int32Array::from_iter_values(0..(FSL_DIMENSION * 2))),
                None,
            )
            .unwrap(),
        ) as ArrayRef;

        for (label, values) in [
            ("fixed-size binary", fixed_size_binary),
            ("fixed-size list", fixed_size_list),
        ] {
            let item_field = Arc::new(ArrowField::new("item", values.data_type().clone(), true));
            let array = unsplittable_nested_list(values, item_field);

            let v2_2 = encode_pages(array.clone(), TestEncoding::StructuralU32, HashMap::new())
                .await
                .unwrap();
            let v2_3 = encode_pages(
                array.clone(),
                TestEncoding::StructuralSparse,
                HashMap::new(),
            )
            .await
            .unwrap();
            assert_eq!(v2_2.len(), 1, "unexpected {label} page count");
            assert!(matches!(page_layout(&v2_2[0]), TestLayout::Zipped(_)));
            assert_eq!(v2_3.len(), 1, "unexpected {label} page count");
            let PageEncoding::Planar(layout) = &v2_3[0].description else {
                unreachable!()
            };
            assert!(layout.sparse().is_some());
            assert!(matches!(
                layout.values.as_ref().unwrap().codec,
                Some(pb23::value_plane::Codec::Zipped(_))
            ));
            let explicit = encode_pages(
                array.clone(),
                TestEncoding::StructuralSparse,
                sparse_metadata(),
            )
            .await
            .unwrap();
            assert_eq!(explicit.len(), 1);

            let cases = TestCases::default()
                .with_u32_structural_encodings()
                .with_range(0..1)
                .with_indices(vec![0]);
            check_round_trip_encoding_of_data(vec![array], &cases, HashMap::new()).await;
        }
    }

    #[test]
    fn test_explicit_sparse_rejects_lance_2_2() {
        let array = Arc::new(Int32Array::from(vec![Some(1), None])) as ArrayRef;
        let Err(error) = create_encoder(&array, TestEncoding::StructuralU32, sparse_metadata())
        else {
            panic!("expected Lance 2.2 to reject explicit sparse encoding");
        };
        assert!(
            error
                .to_string()
                .contains("not enabled by the selected file format")
        );

        let structural_only = list_i32(vec![0, 0, 0], None);
        let Err(error) = create_encoder(
            &structural_only,
            TestEncoding::StructuralU32,
            sparse_metadata(),
        ) else {
            panic!("expected Lance 2.2 structural-only input to reject explicit sparse encoding");
        };
        assert!(
            error
                .to_string()
                .contains("not enabled by the selected file format")
        );
    }
}
