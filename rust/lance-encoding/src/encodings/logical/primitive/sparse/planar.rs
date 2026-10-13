// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sparse structure composed with independently scheduled value storage.

use super::*;
use crate::format::pb23;
use arrow_array::new_null_array;
use arrow_data::transform::MutableArrayData;

fn gather_materialized_values(
    arrays: &[ArrayRef],
    positions: &SparsePositionPlan,
) -> Result<ArrayRef> {
    let data = arrays
        .iter()
        .map(|array| array.to_data())
        .collect::<Vec<_>>();
    let mut selected = MutableArrayData::new(
        data.iter().collect(),
        false,
        usize_from_u64(positions.len(), "materialized values")?,
    );
    let mut array_index = 0;
    let mut array_start = 0;
    for range in positions_as_ranges(positions)? {
        let mut start = range.start as usize;
        let end = range.end as usize;
        while start < end {
            let array = data.get(array_index).ok_or_else(|| {
                Error::invalid_input("Materialized range exceeds the value arrays")
            })?;
            let array_end = array_start + array.len();
            if start >= array_end {
                array_index += 1;
                array_start = array_end;
                continue;
            }
            let take_end = end.min(array_end);
            selected.extend(array_index, start - array_start, take_end - array_start);
            start = take_end;
        }
    }
    Ok(make_array(selected.freeze()))
}

/// Reuse a binary payload when omitted null slots contain no bytes.
fn compact_variable_offsets(
    mut data: crate::data::VariableWidthBlock,
    positions: &SparsePositionPlan,
) -> Result<Option<DataBlock>> {
    let width = usize::from(data.bits_per_offset / 8);
    if ![4, 8].contains(&width) {
        return Ok(None);
    }
    let offsets = data.offsets.as_ref();
    let offset = |index: u64| -> Result<&[u8]> {
        let start = usize_from_u64(index, "variable value index")?
            .checked_mul(width)
            .ok_or_else(|| Error::invalid_input("Variable offset index overflows"))?;
        let end = start
            .checked_add(width)
            .ok_or_else(|| Error::invalid_input("Variable offset end overflows"))?;
        offsets
            .get(start..end)
            .ok_or_else(|| Error::invalid_input("Variable offset is missing"))
    };
    let capacity = usize_from_u64(positions.len(), "selected variable values")?
        .checked_add(1)
        .and_then(|len| len.checked_mul(width))
        .ok_or_else(|| Error::invalid_input("Selected variable offset byte count overflows"))?;
    let mut selected = Vec::with_capacity(capacity);
    selected.extend_from_slice(offset(0)?);
    let mut previous = 0;
    for range in positions_as_ranges(positions)? {
        if offset(previous)? != offset(range.start)? {
            return Ok(None);
        }
        for index in range.clone() {
            selected.extend_from_slice(offset(index + 1)?);
        }
        previous = range.end;
    }
    if offset(previous)? != offset(data.num_values)? {
        return Ok(None);
    }
    data.offsets = LanceBuffer::from(selected);
    data.num_values = positions.len();
    data.block_info = BlockInfo::new();
    let mut data = DataBlock::VariableWidth(data);
    data.compute_stat();
    Ok(Some(data))
}

fn restore_leaf_slots(
    array: &dyn Array,
    positions: &SparsePositionPlan,
    slots: usize,
) -> Result<ArrayRef> {
    let data = array.to_data();
    let mut restored = MutableArrayData::new(vec![&data], true, slots);
    let mut output_offset = 0usize;
    let mut value_offset = 0usize;
    for range in positions_as_ranges(positions)? {
        if range.end > slots as u64 || range.start < output_offset as u64 {
            return Err(Error::invalid_input(
                "Materialized ranges exceed or overlap leaf slots",
            ));
        }
        let start = range.start as usize;
        let end = range.end as usize;
        let count = end - start;
        if count > array.len() - value_offset {
            return Err(Error::invalid_input(
                "Materialized range exceeds decoded values",
            ));
        }
        restored.extend_nulls(start - output_offset);
        restored.extend(0, value_offset, value_offset + count);
        output_offset = end;
        value_offset += count;
    }
    if value_offset != array.len() {
        return Err(Error::invalid_input(
            "Decoded values exceed materialized ranges",
        ));
    }
    restored.extend_nulls(slots - output_offset);
    Ok(make_array(restored.freeze()))
}

fn positions_as_ranges(positions: &SparsePositionPlan) -> Result<Vec<Range<u64>>> {
    match positions {
        SparsePositionPlan::Empty => Ok(vec![]),
        SparsePositionPlan::All { len } => Ok(vec![0..*len]),
        SparsePositionPlan::Range { start, len } => Ok(vec![*start..*start + *len]),
        SparsePositionPlan::Arithmetic {
            start,
            step: 1,
            len,
        } => Ok(vec![*start..*start + *len]),
        _ => Ok(coalesce_ranges(
            positions
                .materialize()?
                .into_iter()
                .map(|p| p..p + 1)
                .collect(),
        )),
    }
}

fn valid_ranges(validity: &SparseValiditySet, len: u64) -> Result<Vec<Range<u64>>> {
    let ranges = positions_as_ranges(&validity.positions)?;
    if validity.meaning == SparseValidityMeaning::ValidPositions {
        return Ok(ranges);
    }
    let mut result = Vec::with_capacity(ranges.len() + 1);
    let mut start = 0;
    for range in ranges {
        push_coalesced_range(&mut result, start..range.start);
        start = range.end;
    }
    push_coalesced_range(&mut result, start..len);
    Ok(result)
}

fn intersect_ranges(left: &[Range<u64>], right: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        let start = left[i].start.max(right[j].start);
        let end = left[i].end.min(right[j].end);
        if start < end {
            push_coalesced_range(&mut out, start..end);
        }
        if left[i].end < right[j].end {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

/// Derive the stored leaf positions without introducing redundant wire metadata.
/// Null ancestors exclude their entire child ranges; Arrow slots are restored
/// only after the requested values have been decoded.
fn materialized_positions(plan: &SparseStructuralPlan, rows: u64) -> Result<SparsePositionPlan> {
    let mut ranges = vec![0..rows];
    for (layer_index, layer) in plan.layers.iter().enumerate() {
        let (slots, validity) = match layer {
            SparseStructuralLayerPlan::Validity {
                num_slots,
                validity,
            }
            | SparseStructuralLayerPlan::List {
                num_slots,
                validity,
                ..
            }
            | SparseStructuralLayerPlan::FixedSizeList {
                num_slots,
                validity,
                ..
            } => (*num_slots, validity),
        };
        // Preserve a symbolic leaf sequence when no ancestor removes slots.
        // A range or arithmetic descriptor may represent far more positions
        // than a page reader should ever allocate during initialization.
        if layer_index + 1 == plan.layers.len()
            && matches!(layer, SparseStructuralLayerPlan::Validity { .. })
            && ranges.as_slice() == [0..slots]
            && validity.meaning == SparseValidityMeaning::ValidPositions
        {
            return Ok(validity.positions.clone());
        }
        ranges = intersect_ranges(&ranges, &valid_ranges(validity, slots)?);
        match layer {
            SparseStructuralLayerPlan::Validity { .. } => {}
            SparseStructuralLayerPlan::List {
                non_empty_positions,
                counts,
                num_child_slots,
                ..
            } => {
                let selection =
                    select_position_set(non_empty_positions, &ranges, slots, "materialized list")?;
                ranges =
                    child_ranges_from_counts(counts, *num_child_slots, &selection.ordinal_ranges)?;
            }
            SparseStructuralLayerPlan::FixedSizeList { dimension, .. } => {
                for range in &mut ranges {
                    range.start = range
                        .start
                        .checked_mul(*dimension)
                        .ok_or_else(|| Error::invalid_input("Sparse child start overflows"))?;
                    range.end = range
                        .end
                        .checked_mul(*dimension)
                        .ok_or_else(|| Error::invalid_input("Sparse child end overflows"))?;
                }
            }
        }
    }
    position_segments_to_set(ranges, plan.num_visible_items, "materialized leaf")
}

#[allow(clippy::too_many_arguments)]
pub(in crate::encodings::logical::primitive) fn encode(
    ctx: &PrimitiveEncodeContext,
    arrays: Vec<ArrayRef>,
    plan: SparseStructuralPlan,
    constant: Option<Vec<u8>>,
    row_number: u64,
    num_rows: u64,
    compression: Arc<dyn CompressionStrategy>,
) -> Result<EncodedPage> {
    let rows = match plan.layers.first() {
        Some(SparseStructuralLayerPlan::Validity { num_slots, .. })
        | Some(SparseStructuralLayerPlan::List { num_slots, .. })
        | Some(SparseStructuralLayerPlan::FixedSizeList { num_slots, .. }) => *num_slots,
        None => plan.num_visible_items,
    };
    let positions = materialized_positions(&plan, rows)?;
    let scalar = if constant.is_some() {
        constant
    } else if positions.is_empty() || !super::super::planar::inline_type(&ctx.field.data_type()) {
        None
    } else {
        let validity =
            BooleanBuffer::from_iter((0..plan.num_visible_items).map(|i| positions.contains(i)));
        PrimitiveStructuralEncoder::find_constant_scalar(&arrays, Some(&validity))?
            .and_then(|scalar| lance_arrow::scalar::try_inline_value(&scalar))
    };
    let wide_fixed = super::super::planar::fixed_width_bytes(&ctx.field.data_type())
        .is_some_and(|width| width >= 256);
    let data = if wide_fixed || scalar.is_some() || positions.is_empty() {
        None
    } else {
        let data = DataBlock::from_arrays(&arrays, plan.num_visible_items);
        Some(match data {
            DataBlock::AllNull(_) => data,
            other => other.remove_outer_validity(),
        })
    };
    let dictionary = data.as_ref().and_then(|data| {
        PrimitiveStructuralEncoder::should_dictionary_encode(
            data,
            &ctx.field,
            FixedWidthDictionaryEncoding::Include64Bit,
        )
        .and_then(|budget| {
            dict::dictionary_encode(data, budget.max_dict_entries, budget.max_encoded_size)
        })
    });
    let (values, mut buffers) = if positions.is_empty()
        || matches!(ctx.field.data_type(), DataType::Struct(fields) if fields.is_empty())
    {
        (pb23::value_plane::Codec::None(pb23::NoValues {}), vec![])
    } else if let Some(scalar) = scalar {
        (
            pb23::value_plane::Codec::Constant(pb23::ConstantValue {
                scalar: scalar.into(),
            }),
            vec![],
        )
    } else if dictionary.is_some()
        || data.as_ref().is_some_and(|data| {
            (PrimitiveStructuralEncoder::is_narrow(data)
                && !matches!(data, DataBlock::Struct(s) if s.has_variable_width_child()))
                || matches!(data, DataBlock::Dictionary(_))
        })
    {
        let data = data.ok_or_else(|| Error::internal("Chunked planar values are missing"))?;
        let (data, dictionary) = if let Some((indices, dictionary)) = dictionary {
            (indices, Some(dictionary))
        } else if let DataBlock::Dictionary(dictionary) = data {
            let (mut indices, values) = dictionary.into_parts();
            indices.compute_stat();
            (indices, Some(values))
        } else {
            (data, None)
        };
        let encoded = PrimitiveStructuralEncoder::encode_miniblock(
            ctx.column_idx,
            &ctx.field,
            compression.as_ref(),
            data,
            SerializedRepDefs::new(None, None, vec![DefinitionInterpretation::AllValidItem]),
            row_number,
            dictionary,
            plan.num_visible_items,
            MiniblockChunkSize::U32,
        )?;
        let PageEncoding::Structural(layout) = encoded.description else {
            return Err(Error::internal(
                "Chunked value encoder returned unexpected metadata",
            ));
        };
        let planar = crate::format::planar::from_dense_execution(layout)?;
        let Some(pb23::ValuePlane {
            codec: Some(pb23::value_plane::Codec::Chunked(mut values)),
        }) = planar.values
        else {
            return Err(Error::internal(
                "Chunked value encoder returned a different value plane",
            ));
        };
        values.levels = None;
        values.num_items = 0;
        (pb23::value_plane::Codec::Chunked(values), encoded.data)
    } else {
        let compact = match data {
            Some(DataBlock::VariableWidth(data)) => compact_variable_offsets(data, &positions)?,
            _ => None,
        };
        let data = if let Some(data) = compact {
            data
        } else {
            let array = gather_materialized_values(&arrays, &positions)?;
            DataBlock::from(array).remove_outer_validity()
        };
        let encoded = PrimitiveStructuralEncoder::encode_full_zip(
            ctx.column_idx,
            &ctx.field,
            compression.as_ref(),
            data,
            SerializedRepDefs::new(None, None, vec![DefinitionInterpretation::AllValidItem]),
            row_number,
            positions.len(),
        )?;
        let PageEncoding::Structural(layout) = encoded.description else {
            return Err(Error::internal(
                "Zipped encoder returned unexpected metadata",
            ));
        };
        let planar = crate::format::planar::from_dense_execution(layout)?;
        let Some(pb23::ValuePlane {
            codec: Some(pb23::value_plane::Codec::Zipped(mut values)),
        }) = planar.values
        else {
            return Err(Error::internal(
                "Zipped encoder returned a different value plane",
            ));
        };
        values.levels = None;
        (pb23::value_plane::Codec::Zipped(values), encoded.data)
    };
    let structural = writer::encode_structural_plan(&plan)?;
    buffers.extend(structural.buffers);
    Ok(EncodedPage {
        description: PageEncoding::Planar(pb23::PlanarLayout {
            structure: Some(pb23::StructuralPlane {
                codec: Some(pb23::structural_plane::Codec::Sparse(
                    pb23::SparseStructure {
                        layers: structural.layers,
                    },
                )),
            }),
            values: Some(pb23::ValuePlane {
                codec: Some(values),
            }),
        }),
        data: buffers,
        row_number,
        num_rows,
        column_idx: ctx.column_idx,
    })
}

#[derive(Debug)]
enum ValueStorage {
    Chunked(MiniBlockScheduler),
    Zipped {
        scheduler: FullZipScheduler,
        num_values: u64,
    },
    Constant(ArrayRef),
    None,
}

struct CachedPlanarSparse {
    plan: SparseStructuralPlan,
    positions: SparsePositionPlan,
    values: Option<Arc<dyn CachedPageData>>,
}

impl DeepSizeOf for CachedPlanarSparse {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.positions.deep_size()
            + self
                .plan
                .layers
                .iter()
                .map(|layer| match layer {
                    SparseStructuralLayerPlan::Validity { validity, .. }
                    | SparseStructuralLayerPlan::FixedSizeList { validity, .. } => {
                        validity.deep_size()
                    }
                    SparseStructuralLayerPlan::List {
                        validity,
                        non_empty_positions,
                        counts,
                        ..
                    } => {
                        validity.deep_size() + non_empty_positions.deep_size() + counts.deep_size()
                    }
                })
                .sum::<usize>()
            + self
                .values
                .as_ref()
                .map_or(0, |v| v.deep_size_of_children(context))
            + self.plan.layers.capacity() * std::mem::size_of::<SparseStructuralLayerPlan>()
    }
}
impl CachedPageData for CachedPlanarSparse {
    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }
}

pub(in crate::encodings::logical::primitive) struct PlanarSparseScheduler {
    layers: Vec<SparseLayerDecompressors>,
    structural_buffers: Vec<Range<u64>>,
    values: ValueStorage,
    rows: u64,
    row_scale: u64,
    leaf_slots: u64,
    data_type: DataType,
    state: Option<Arc<CachedPlanarSparse>>,
}
impl std::fmt::Debug for PlanarSparseScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanarSparseScheduler")
            .field("rows", &self.rows)
            .field("values", &self.values)
            .finish_non_exhaustive()
    }
}

impl PlanarSparseScheduler {
    pub(in crate::encodings::logical::primitive) fn try_new(
        page: &PageInfo,
        layout: &pb23::PlanarLayout,
        field: &Field,
        decompressors: &dyn DecompressionStrategy,
    ) -> Result<Self> {
        super::super::planar::validate_page(page, layout, field, decompressors)?;
        let structure = layout
            .sparse()
            .ok_or_else(|| Error::invalid_input("Planar sparse structure is missing"))?;
        let (rows, row_scale) = structure.row_count_and_scale(page.num_rows)?;
        let mut domain = rows;
        let mut layers = Vec::with_capacity(structure.layers.len());
        for layer in &structure.layers {
            let child = child_domain(layer, domain)?;
            layers.push(layer_decompressors(layer, domain, child)?);
            domain = child;
        }
        let leaf_slots = domain;
        let codec = layout
            .values
            .as_ref()
            .and_then(|v| v.codec.as_ref())
            .ok_or_else(|| Error::invalid_input("Planar value plane is missing"))?;
        let value_buffers = match codec {
            pb23::value_plane::Codec::Indexed(_) => 3,
            pb23::value_plane::Codec::Chunked(v) => 2 + usize::from(v.dictionary.is_some()),
            pb23::value_plane::Codec::Zipped(v) => {
                if matches!(
                    v.addressing,
                    Some(pb23::zipped_values::Addressing::BitsPerOffset(_))
                ) {
                    2
                } else {
                    1
                }
            }
            _ => 0,
        };
        let expected = value_buffers
            + layers
                .iter()
                .map(SparseLayerDecompressors::num_buffers)
                .sum::<usize>();
        if page.buffer_offsets_and_sizes.len() != expected {
            return Err(Error::invalid_input(format!(
                "Planar sparse page has {} buffers, expected {expected}",
                page.buffer_offsets_and_sizes.len()
            )));
        }
        let buffers = &page.buffer_offsets_and_sizes[..value_buffers];
        let values = match codec {
            pb23::value_plane::Codec::Indexed(_) => {
                return Err(Error::not_supported(
                    "Indexed planar values are not implemented",
                ));
            }
            pb23::value_plane::Codec::Chunked(v) => {
                if v.levels.is_some() || v.num_items != 0 {
                    return Err(Error::invalid_input(
                        "Sparse chunked values contain dense level metadata",
                    ));
                }
                if v.compression.is_none() || v.dictionary.is_some() != (v.num_dictionary_items > 0)
                {
                    return Err(Error::invalid_input(
                        "Sparse chunked value compression or dictionary is invalid",
                    ));
                }
                ValueStorage::Chunked(MiniBlockScheduler::try_new(
                    buffers,
                    page.priority,
                    leaf_slots,
                    &metadata::MiniBlockLayout {
                        value_compression: v.compression.clone(),
                        dictionary: v.dictionary.clone(),
                        num_dictionary_items: v.num_dictionary_items,
                        num_buffers: v.num_buffers,
                        num_items: leaf_slots,
                        layers: vec![metadata::RepDefLayer::RepdefAllValidItem as i32],
                        has_large_chunk: true,
                        ..Default::default()
                    },
                    decompressors,
                )?)
            }
            pb23::value_plane::Codec::Zipped(v) => {
                if v.levels.is_some()
                    || v.num_visible_items != v.num_items
                    || v.compression.is_none()
                {
                    return Err(Error::invalid_input(
                        "Sparse zipped values require one value per item and no dense controls",
                    ));
                }
                match v.addressing {
                    Some(pb23::zipped_values::Addressing::BitsPerValue(bits))
                        if bits > 0 && bits % 8 == 0 =>
                    {
                        let expected_bytes = v
                            .num_items
                            .checked_mul(u64::from(bits / 8))
                            .ok_or_else(|| Error::invalid_input("Zipped byte count overflows"))?;
                        if buffers[0].1 != expected_bytes {
                            return Err(Error::invalid_input(
                                "Sparse zipped fixed byte count mismatch",
                            ));
                        }
                    }
                    Some(pb23::zipped_values::Addressing::BitsPerOffset(bits))
                        if [8, 16, 32, 64].contains(&bits) =>
                    {
                        let n = v
                            .num_items
                            .checked_add(1)
                            .ok_or_else(|| Error::invalid_input("Zipped offset count overflows"))?;
                        if !buffers[1].1.is_multiple_of(n)
                            || ![1, 2, 4, 8].contains(&(buffers[1].1 / n))
                        {
                            return Err(Error::invalid_input(
                                "Sparse zipped offset buffer has invalid length",
                            ));
                        }
                    }
                    _ => return Err(Error::invalid_input("Sparse zipped addressing is invalid")),
                }
                let mut scheduler = FullZipScheduler::try_new(
                    buffers,
                    page.priority,
                    v.num_items,
                    &v.full_zip(vec![metadata::RepDefLayer::RepdefAllValidItem as i32], None)?,
                    decompressors,
                )?;
                scheduler.enable_cache = true;
                ValueStorage::Zipped {
                    scheduler,
                    num_values: v.num_items,
                }
            }
            pb23::value_plane::Codec::Constant(v) => {
                let data_type = field.data_type();
                let width = if data_type == DataType::Boolean {
                    Some(1)
                } else {
                    data_type.byte_width_opt()
                };
                if v.scalar.is_empty() || v.scalar.len() > 32 || width != Some(v.scalar.len()) {
                    return Err(Error::invalid_input(
                        "Planar scalar length does not match its leaf type",
                    ));
                }
                ValueStorage::Constant(lance_arrow::scalar::decode_scalar_from_inline_value(
                    &data_type, &v.scalar,
                )?)
            }
            pb23::value_plane::Codec::None(_) => ValueStorage::None,
        };
        let structural_buffers = page.buffer_offsets_and_sizes[value_buffers..]
            .iter()
            .map(|&(p, s)| checked_buffer_range(p, s, "planar structure"))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            structural_buffers,
            values,
            rows,
            row_scale,
            leaf_slots,
            data_type: field.data_type(),
            state: None,
        })
    }

    fn value_scheduler(&self) -> Option<&dyn StructuralPageScheduler> {
        match &self.values {
            ValueStorage::Chunked(s) => Some(s),
            ValueStorage::Zipped { scheduler, .. } => Some(scheduler),
            _ => None,
        }
    }
}

impl StructuralPageScheduler for PlanarSparseScheduler {
    fn init_layout(&self) -> Result<PageInitialization> {
        let mut ranges = self
            .value_scheduler()
            .map(|s| s.init_layout())
            .transpose()?
            .map_or(vec![], |l| l.buffer_ranges);
        ranges.extend(self.structural_buffers.iter().cloned().map(Some));
        Ok(PageInitialization::new(ranges))
    }
    fn init_from_buffers<'a>(
        &'a mut self,
        buffers: PageInitializationBuffers,
        io: &Arc<dyn EncodingsIo>,
    ) -> BoxFuture<'a, Result<Arc<dyn CachedPageData>>> {
        let io = io.clone();
        async move {
            let count = self
                .value_scheduler()
                .map(|s| s.init_layout())
                .transpose()?
                .map_or(0, |l| l.buffer_ranges.len());
            let mut value_buffers = buffers.buffer_slots;
            if value_buffers.len() != count + self.structural_buffers.len() {
                return Err(Error::invalid_input(
                    "Planar initialization buffer count mismatch",
                ));
            }
            let structural = value_buffers.split_off(count);
            if let ValueStorage::Chunked(scheduler) = &self.values {
                let metadata = value_buffers
                    .first()
                    .and_then(Option::as_ref)
                    .ok_or_else(|| Error::invalid_input("Planar chunk metadata is missing"))?;
                if !metadata.len().is_multiple_of(4) {
                    return Err(Error::invalid_input(
                        "Planar chunk metadata length is not divisible by four",
                    ));
                }
                let bytes = metadata
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .try_fold(0_u64, |total, word| {
                        let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
                        total
                            .checked_add((u64::from(word >> 4) + 1) * MINIBLOCK_ALIGNMENT as u64)
                            .ok_or_else(|| Error::invalid_input("Planar chunk byte sum overflows"))
                    })?;
                if bytes != scheduler.buffer_offsets_and_sizes[1].1 {
                    return Err(Error::invalid_input(
                        "Planar chunk byte sum does not match the value buffer",
                    ));
                }
            }
            if let ValueStorage::Zipped {
                scheduler,
                num_values,
            } = &self.values
                && let Some(index) = value_buffers.first().and_then(Option::as_ref)
            {
                let count = usize_from_u64(num_values + 1, "zipped index count")?;
                let width = index.len() / count;
                if ![1, 2, 4, 8].contains(&width) || !index.len().is_multiple_of(count) {
                    return Err(Error::invalid_input(
                        "Planar zipped index has an invalid length",
                    ));
                }
                let mut previous = 0;
                for (i, bytes) in index.chunks_exact(width).enumerate() {
                    let mut value = [0; 8];
                    value[..width].copy_from_slice(bytes);
                    let offset = u64::from_le_bytes(value);
                    if (i == 0 && offset != 0)
                        || offset < previous
                        || offset > scheduler.data_buf_size
                    {
                        return Err(Error::invalid_input(
                            "Planar zipped byte offsets are unordered or out of bounds",
                        ));
                    }
                    previous = offset;
                }
                if previous != scheduler.data_buf_size {
                    return Err(Error::invalid_input(
                        "Planar zipped final byte offset does not match the value buffer",
                    ));
                }
            }
            let values = match &mut self.values {
                ValueStorage::Chunked(s) => Some(
                    s.init_from_buffers(
                        PageInitializationBuffers {
                            buffer_slots: value_buffers,
                        },
                        &io,
                    )
                    .await?,
                ),
                ValueStorage::Zipped { scheduler, .. } if scheduler.needs_initialization() => Some(
                    scheduler
                        .init_from_buffers(
                            PageInitializationBuffers {
                                buffer_slots: value_buffers,
                            },
                            &io,
                        )
                        .await?,
                ),
                _ => None,
            };
            let mut structural = PageInitializationBuffers {
                buffer_slots: structural,
            }
            .into_required("planar structure")?
            .into_iter();
            let layers = self
                .layers
                .iter()
                .map(|l| decode_layer(l, &mut structural))
                .collect::<Result<Vec<_>>>()?;
            let plan = SparseStructuralPlan {
                layers,
                num_visible_items: self.leaf_slots,
            };
            plan.validate(self.rows)?;
            let positions = match self.values {
                ValueStorage::Chunked(_) => SparsePositionPlan::all(self.leaf_slots),
                _ => materialized_positions(&plan, self.rows)?,
            };
            if let ValueStorage::Zipped { num_values, .. } = self.values
                && positions.len() != num_values
            {
                return Err(Error::invalid_input(
                    "Sparse zipped value count disagrees with structure",
                ));
            }
            if matches!(self.values, ValueStorage::None)
                && !positions.is_empty()
                && !matches!(&self.data_type, DataType::Struct(fields) if fields.is_empty())
            {
                return Err(Error::invalid_input(
                    "No-values plane has materialized leaf values",
                ));
            }
            Ok(Arc::new(CachedPlanarSparse {
                plan,
                positions,
                values,
            }) as Arc<dyn CachedPageData>)
        }
        .boxed()
    }
    fn try_load(&mut self, state: &Arc<dyn CachedPageData>) -> Result<()> {
        let state = state
            .clone()
            .as_arc_any()
            .downcast::<CachedPlanarSparse>()
            .map_err(|_| Error::invalid_input("Unexpected planar sparse cache type"))?;
        if let Some(values) = &state.values {
            match &mut self.values {
                ValueStorage::Chunked(s) => s.try_load(values)?,
                ValueStorage::Zipped { scheduler, .. } => scheduler.try_load(values)?,
                _ => return Err(Error::invalid_input("Unexpected cached planar value state")),
            }
        }
        self.state = Some(state);
        Ok(())
    }
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Vec<PageLoadTask>> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| Error::internal("Planar sparse page is not initialized"))?
            .clone();
        let num_rows = validate_slice_ranges(
            ranges,
            self.rows
                .checked_mul(self.row_scale)
                .ok_or_else(|| Error::invalid_input("Planar row domain overflows"))?,
            "planar row",
        )?;
        let ranges = ranges
            .iter()
            .map(|r| {
                if !r.start.is_multiple_of(self.row_scale) || !r.end.is_multiple_of(self.row_scale)
                {
                    return Err(Error::invalid_input(
                        "Planar row range is not aligned to fixed-size-list scale",
                    ));
                }
                Ok(r.start / self.row_scale..r.end / self.row_scale)
            })
            .collect::<Result<Vec<_>>>()?;
        let selection = slice_sparse_plan(&state.plan, &ranges, self.rows)?;
        let selected = select_position_set(
            &state.positions,
            &selection.leaf_ranges,
            self.leaf_slots,
            "planar value",
        )?;
        let tasks = if selected.positions.is_empty() {
            vec![]
        } else {
            self.value_scheduler()
                .map(|s| s.schedule_ranges(&selected.ordinal_ranges, io))
                .transpose()?
                .unwrap_or_default()
        };
        let scalar = match &self.values {
            ValueStorage::Constant(s) => Some(s.clone()),
            _ => None,
        };
        let data_type = self.data_type.clone();
        let row_scale = self.row_scale;
        let rows = self.rows;
        Ok(vec![PageLoadTask {
            num_rows,
            decoder_fut: async move {
                let mut decoders = VecDeque::new();
                for task in tasks {
                    decoders.push_back((task.decoder_fut.await?, task.num_rows));
                }
                Ok(Box::new(PlanarSparseDecoder {
                    ranges: ranges.into(),
                    range_offset: 0,
                    rows,
                    row_scale,
                    num_rows,
                    state,
                    decoders,
                    scalar,
                    data_type,
                }) as Box<dyn StructuralPageDecoder>)
            }
            .boxed(),
        }])
    }
}

struct PlanarSparseDecoder {
    ranges: VecDeque<Range<u64>>,
    range_offset: u64,
    rows: u64,
    row_scale: u64,
    num_rows: u64,
    state: Arc<CachedPlanarSparse>,
    decoders: VecDeque<(Box<dyn StructuralPageDecoder>, u64)>,
    scalar: Option<ArrayRef>,
    data_type: DataType,
}
impl std::fmt::Debug for PlanarSparseDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanarSparseDecoder")
            .field("rows", &self.num_rows)
            .finish_non_exhaustive()
    }
}
impl StructuralPageDecoder for PlanarSparseDecoder {
    fn num_rows(&self) -> u64 {
        self.num_rows
    }
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        if !num_rows.is_multiple_of(self.row_scale) {
            return Err(Error::invalid_input(
                "Planar drain is not aligned to fixed-size-list scale",
            ));
        }
        let mut remaining = num_rows / self.row_scale;
        let mut ranges = Vec::new();
        while remaining > 0 {
            let r = self
                .ranges
                .front()
                .ok_or_else(|| Error::invalid_input("Planar drain exceeds scheduled rows"))?;
            let start = r.start + self.range_offset;
            let take = remaining.min(r.end - start);
            ranges.push(start..start + take);
            self.range_offset += take;
            remaining -= take;
            if start + take == r.end {
                self.ranges.pop_front();
                self.range_offset = 0;
            }
        }
        let selection = slice_sparse_plan(&self.state.plan, &ranges, self.rows)?;
        let selected = select_position_set(
            &self.state.positions,
            &selection.leaf_ranges,
            self.state.plan.num_visible_items,
            "planar drain",
        )?;
        let mut tasks = Vec::new();
        let mut remaining = selected.positions.len();
        while remaining > 0 && !self.decoders.is_empty() {
            let (decoder, available) = self
                .decoders
                .front_mut()
                .ok_or_else(|| Error::internal("Missing planar value decoder"))?;
            let take = remaining.min(*available);
            tasks.push(decoder.drain(take)?);
            *available -= take;
            remaining -= take;
            if *available == 0 {
                self.decoders.pop_front();
            }
        }
        if remaining > 0
            && self.scalar.is_none()
            && !matches!(&self.data_type, DataType::Struct(fields) if fields.is_empty())
        {
            return Err(Error::invalid_input(
                "Planar value decoders ended before the selected leaf values",
            ));
        }
        Ok(Box::new(PlanarSparseTask {
            plan: selection.plan,
            positions: selected.positions,
            tasks,
            scalar: self.scalar.clone(),
            data_type: self.data_type.clone(),
        }))
    }
}

#[derive(Debug)]
struct PlanarSparseTask {
    plan: SparseStructuralPlan,
    positions: SparsePositionPlan,
    tasks: Vec<Box<dyn DecodePageTask>>,
    scalar: Option<ArrayRef>,
    data_type: DataType,
}
impl PlanarSparseTask {
    fn decode_checked(self) -> Result<DecodedPage> {
        let slots = usize_from_u64(self.plan.num_visible_items, "decoded leaf slots")?;
        let array = if let Some(scalar) = self.scalar {
            // Sparse structure carries validity. Materializing nullable take indices
            // duplicates that work and allocates eight bytes per output slot.
            let bytes = lance_arrow::scalar::try_inline_value(&scalar)
                .ok_or_else(|| Error::invalid_input("Planar constant is not an inline scalar"))?;
            let (data, bits_per_value) = if self.data_type == DataType::Boolean {
                (
                    LanceBuffer::from(vec![
                        if bytes[0] == 0 { 0 } else { u8::MAX };
                        slots.div_ceil(8)
                    ]),
                    1,
                )
            } else {
                let size = slots
                    .checked_mul(bytes.len())
                    .ok_or_else(|| Error::invalid_input("Planar constant output size overflows"))?;
                let mut data = arrow_buffer::MutableBuffer::from_len_zeroed(size);
                if slots > 0 {
                    data.as_slice_mut()[..bytes.len()].copy_from_slice(&bytes);
                    let mut filled = bytes.len();
                    while filled < size {
                        let count = filled.min(size - filled);
                        data.as_slice_mut().copy_within(..count, filled);
                        filled += count;
                    }
                }
                (
                    LanceBuffer::from(arrow_buffer::Buffer::from(data)),
                    bytes.len() as u64 * 8,
                )
            };
            return Ok(DecodedPage {
                data: DataBlock::FixedWidth(FixedWidthDataBlock {
                    data,
                    bits_per_value,
                    num_values: slots as u64,
                    block_info: BlockInfo::new(),
                }),
                repdef: RepDefUnraveler::new_sparse(self.plan),
            });
        } else if self.tasks.is_empty() {
            new_null_array(&self.data_type, slots)
        } else {
            let mut tasks = self.tasks.into_iter();
            let first = tasks
                .next()
                .ok_or_else(|| Error::internal("Missing planar decode task"))?
                .decode()?
                .data;
            let data = if tasks.len() == 0 {
                first
            } else {
                let mut builder = DataBlockBuilder::with_capacity_estimate(first.data_size());
                builder.append(&first, 0..first.num_values())?;
                for task in tasks {
                    let data = task.decode()?.data;
                    builder.append(&data, 0..data.num_values())?;
                }
                builder.finish()
            };
            if data.num_values() != self.positions.len() {
                return Err(Error::invalid_input("Decoded planar value count mismatch"));
            }
            if self.positions.len() == slots as u64 {
                return Ok(DecodedPage {
                    data,
                    repdef: RepDefUnraveler::new_sparse(self.plan),
                });
            }
            if !data.is_nullable()
                && super::super::planar::fixed_width_bytes(&self.data_type).is_some()
            {
                let flat = match data {
                    DataBlock::FixedWidth(flat) => flat,
                    DataBlock::FixedSizeList(fsl) => fsl.try_into_flat().ok_or_else(|| {
                        Error::invalid_input("Planar fixed-size-list values cannot be flattened")
                    })?,
                    _ => {
                        return Err(Error::invalid_input(
                            "Planar fixed-width leaf has non-fixed values",
                        ));
                    }
                };
                let width = usize_from_u64(flat.bits_per_value / 8, "decoded value width")?;
                let size = slots.checked_mul(width).ok_or_else(|| {
                    Error::invalid_input("Planar restored fixed-width byte count overflows")
                })?;
                let source = flat.data.as_ref();
                if !flat.bits_per_value.is_multiple_of(8)
                    || source.len() != self.positions.len() as usize * width
                {
                    return Err(Error::invalid_input(
                        "Planar decoded fixed-width byte count mismatch",
                    ));
                }
                let mut restored = vec![0; size];
                let mut source_offset = 0;
                for range in positions_as_ranges(&self.positions)? {
                    let start = usize_from_u64(range.start, "restored slot start")? * width;
                    let end = usize_from_u64(range.end, "restored slot end")? * width;
                    let length = end - start;
                    restored
                        .get_mut(start..end)
                        .ok_or_else(|| {
                            Error::invalid_input("Planar restored range exceeds leaf slots")
                        })?
                        .copy_from_slice(&source[source_offset..source_offset + length]);
                    source_offset += length;
                }
                let data = crate::data::FixedSizeListBlock::from_flat(
                    FixedWidthDataBlock {
                        data: LanceBuffer::from(restored),
                        bits_per_value: flat.bits_per_value,
                        num_values: slots as u64,
                        block_info: BlockInfo::new(),
                    },
                    &self.data_type,
                );
                return Ok(DecodedPage {
                    data,
                    repdef: RepDefUnraveler::new_sparse(self.plan),
                });
            }
            let array = make_array(data.into_arrow(self.data_type.clone(), true)?);
            restore_leaf_slots(array.as_ref(), &self.positions, slots)?
        };
        let data = DataBlock::from_decoded_array(array);
        let data = match data {
            DataBlock::AllNull(_) => data,
            other => other.remove_outer_validity(),
        };
        Ok(DecodedPage {
            data,
            repdef: RepDefUnraveler::new_sparse(self.plan),
        })
    }
}

impl DecodePageTask for PlanarSparseTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (*self).decode_checked()))
            .map_err(|_| Error::invalid_input("Malformed planar sparse value payload"))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{TestEncoding, test_compression_strategy};
    use arrow_array::{FixedSizeBinaryArray, Int32Array, StringArray};
    use std::sync::Mutex;

    #[derive(Debug)]
    struct RecordingIo {
        data: Bytes,
        requests: Mutex<Vec<Range<u64>>>,
    }
    impl EncodingsIo for RecordingIo {
        fn submit_request(
            &self,
            ranges: Vec<Range<u64>>,
            _priority: u64,
        ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
            self.requests.lock().unwrap().extend(ranges.iter().cloned());
            let result = ranges
                .into_iter()
                .map(|r| self.data.slice(r.start as usize..r.end as usize))
                .collect();
            std::future::ready(Ok(result)).boxed()
        }
    }

    fn flat_plan(array: &ArrayRef) -> SparseStructuralPlan {
        let len = array.len() as u64;
        SparseStructuralPlan {
            layers: vec![SparseStructuralLayerPlan::Validity {
                num_slots: len,
                validity: SparseValiditySet {
                    meaning: SparseValidityMeaning::ValidPositions,
                    positions: SparsePositionPlan::from_positions(
                        (0..len).filter(|&i| array.is_valid(i as usize)).collect(),
                        len,
                        "test nulls",
                    )
                    .unwrap(),
                },
            }],
            num_visible_items: len,
        }
    }

    fn page(
        array: ArrayRef,
        plan: SparseStructuralPlan,
        rows: u64,
    ) -> (PageInfo, Field, Arc<RecordingIo>) {
        let field =
            Field::try_from(&ArrowField::new("values", array.data_type().clone(), true)).unwrap();
        let ctx = PrimitiveEncodeContext {
            column_idx: 0,
            field: field.clone(),
            encoding_metadata: Arc::new(HashMap::new()),
            is_simple_validity: true,
            has_repdef_info: true,
        };
        // Primitive leaf extraction already moved outer validity into the plan.
        let array = make_array(array.to_data().into_builder().nulls(None).build().unwrap());
        let encoded = encode(
            &ctx,
            vec![array],
            plan,
            None,
            0,
            rows,
            test_compression_strategy(TestEncoding::StructuralSparse, Default::default()),
        )
        .unwrap();
        let mut data = Vec::new();
        let buffers = encoded
            .data
            .iter()
            .map(|b| {
                let start = data.len() as u64;
                data.extend_from_slice(b);
                (start, b.len() as u64)
            })
            .collect::<Vec<_>>();
        (
            PageInfo {
                num_rows: rows,
                priority: 0,
                encoding: encoded.description,
                buffer_offsets_and_sizes: buffers.into(),
            },
            field,
            Arc::new(RecordingIo {
                data: Bytes::from(data),
                requests: Mutex::new(vec![]),
            }),
        )
    }

    fn scheduler(page: &PageInfo, field: &Field) -> Result<PlanarSparseScheduler> {
        let PageEncoding::Planar(layout) = &page.encoding else {
            unreachable!()
        };
        PlanarSparseScheduler::try_new(
            page,
            layout,
            field,
            &crate::compression::DefaultDecompressionStrategy::default(),
        )
    }

    async fn initialize(
        scheduler: &mut PlanarSparseScheduler,
        io: &Arc<dyn EncodingsIo>,
    ) -> Result<Arc<dyn CachedPageData>> {
        let layout = scheduler.init_layout()?;
        let buffers = io
            .submit_request(layout.required_ranges().cloned().collect(), 0)
            .await?;
        let state = scheduler
            .init_from_buffers(layout.with_buffers(buffers)?, io)
            .await?;
        scheduler.try_load(&state)?;
        Ok(state)
    }

    async fn read(
        scheduler: &PlanarSparseScheduler,
        io: &Arc<dyn EncodingsIo>,
        range: Range<u64>,
    ) -> Result<DecodedPage> {
        let count = range.end - range.start;
        let mut tasks = scheduler.schedule_ranges(&[range], io)?;
        assert_eq!(tasks.len(), 1);
        let mut decoder = tasks.remove(0).decoder_fut.await?;
        decoder.drain(count)?.decode()
    }

    #[rstest::rstest]
    #[case::fixed(false)]
    #[case::variable(true)]
    #[tokio::test]
    async fn zipped_null_selection_reads_no_values_and_live_selection_reads_one_value(
        #[case] variable: bool,
    ) {
        let raw = (0..16)
            .map(|i| (i % 3 == 0).then(|| vec![i as u8 + b'a'; 1024]))
            .collect::<Vec<_>>();
        let array: ArrayRef = if variable {
            Arc::new(StringArray::from_iter(
                raw.iter()
                    .map(|v| v.as_ref().map(|v| std::str::from_utf8(v).unwrap())),
            ))
        } else {
            Arc::new(
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                    raw.iter().map(|v| v.as_deref()),
                    1024,
                )
                .unwrap(),
            )
        };
        let (page, field, recorder) = page(array.clone(), flat_plan(&array), 16);
        let io: Arc<dyn EncodingsIo> = recorder.clone();
        let mut cold = scheduler(&page, &field).unwrap();
        let cache = initialize(&mut cold, &io).await.unwrap();
        // Initialization may read structure and the byte index, never value bytes.
        let (value_start, value_size) = page.buffer_offsets_and_sizes[0];
        assert!(
            recorder
                .requests
                .lock()
                .unwrap()
                .iter()
                .all(|r| r.end <= value_start || r.start >= value_start + value_size)
        );
        let mut warm = scheduler(&page, &field).unwrap();
        warm.try_load(&cache).unwrap();
        recorder.requests.lock().unwrap().clear();
        let nulls = read(&warm, &io, 1..3).await.unwrap();
        assert_eq!(nulls.data.num_values(), 2);
        assert!(recorder.requests.lock().unwrap().is_empty());
        read(&warm, &io, 3..4).await.unwrap();
        let requested = recorder.requests.lock().unwrap();
        assert_eq!(requested.len(), 1);
        assert!(requested[0].end - requested[0].start <= 1040);
        assert!(requested[0].end - requested[0].start < value_size);
    }

    #[rstest::rstest]
    #[case::constant(false)]
    #[case::none(true)]
    #[tokio::test]
    async fn zero_payload_values_and_cached_structure(#[case] all_null: bool) {
        let array: ArrayRef = Arc::new(Int32Array::from_iter(
            (0..16).map(|i| (!all_null && i % 3 == 0).then_some(42)),
        ));
        let (page, field, recorder) = page(array.clone(), flat_plan(&array), 16);
        assert!(page.buffer_offsets_and_sizes.is_empty());
        let io: Arc<dyn EncodingsIo> = recorder.clone();
        let mut s = scheduler(&page, &field).unwrap();
        initialize(&mut s, &io).await.unwrap();
        recorder.requests.lock().unwrap().clear();
        read(&s, &io, 0..16).await.unwrap();
        assert!(recorder.requests.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn empty_list_selection_does_not_fetch_chunked_values() {
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..64));
        let mut plan = flat_plan(&array);
        plan.layers.insert(
            0,
            SparseStructuralLayerPlan::List {
                num_slots: 128,
                num_child_slots: 64,
                non_empty_positions: SparsePositionPlan::Range { start: 127, len: 1 },
                counts: SparseCountPlan::Constant { value: 64, len: 1 },
                validity: SparseValiditySet {
                    meaning: SparseValidityMeaning::NullPositions,
                    positions: SparsePositionPlan::Empty,
                },
            },
        );
        let (page, field, recorder) = page(array, plan, 128);
        let io: Arc<dyn EncodingsIo> = recorder.clone();
        let mut s = scheduler(&page, &field).unwrap();
        initialize(&mut s, &io).await.unwrap();
        recorder.requests.lock().unwrap().clear();
        let decoded = read(&s, &io, 0..127).await.unwrap();
        assert_eq!(decoded.data.num_values(), 0);
        assert!(recorder.requests.lock().unwrap().is_empty());
        read(&s, &io, 127..128).await.unwrap();
        assert!(!recorder.requests.lock().unwrap().is_empty());
    }

    #[test]
    fn malformed_envelopes_fail_before_io() {
        let array: ArrayRef = Arc::new(Int32Array::from_iter(
            (0..16).map(|i| (i % 3 == 0).then_some(i)),
        ));
        let (mut page, field, _) = page(array.clone(), flat_plan(&array), 16);
        let PageEncoding::Planar(original) = page.encoding.clone() else {
            unreachable!()
        };
        for mutation in 0..7 {
            let mut layout = original.clone();
            match mutation {
                0 => layout.structure = None,
                1 => layout.values = None,
                2 => {
                    let Some(pb23::value_plane::Codec::Chunked(v)) =
                        layout.values.as_mut().unwrap().codec.as_mut()
                    else {
                        unreachable!()
                    };
                    v.compression = None;
                }
                3 => {
                    let Some(pb23::value_plane::Codec::Chunked(v)) =
                        layout.values.as_mut().unwrap().codec.as_mut()
                    else {
                        unreachable!()
                    };
                    v.num_buffers += 1;
                }
                4 => {
                    let Some(pb23::structural_plane::Codec::Sparse(s)) =
                        layout.structure.as_mut().unwrap().codec.as_mut()
                    else {
                        unreachable!()
                    };
                    s.layers[0].layer = None;
                }
                5 => {
                    layout.values.as_mut().unwrap().codec =
                        Some(pb23::value_plane::Codec::Constant(pb23::ConstantValue {
                            scalar: vec![0].into(),
                        }));
                }
                6 => {
                    let Some(pb23::structural_plane::Codec::Sparse(s)) =
                        layout.structure.as_mut().unwrap().codec.as_mut()
                    else {
                        unreachable!()
                    };
                    s.layers[0].layer = Some(pb23::sparse_structural_layer::Layer::FixedSizeList(
                        pb23::SparseFixedSizeListLayer {
                            dimension: 0,
                            validity: None,
                        },
                    ));
                }
                _ => unreachable!(),
            }
            page.encoding = PageEncoding::Planar(layout);
            assert!(
                matches!(scheduler(&page, &field), Err(Error::InvalidInput { .. })),
                "mutation {mutation}"
            );
        }
        page.encoding = PageEncoding::Planar(original);
        page.buffer_offsets_and_sizes = vec![(u64::MAX, 1), (0, 0)].into();
        assert!(scheduler(&page, &field).is_err());
    }
    #[test]
    fn materialized_leaf_ranges_remain_symbolic() {
        let len = 1_u64 << 40;
        let positions = SparsePositionPlan::Arithmetic {
            start: 2,
            step: 3,
            len,
        };
        let plan = SparseStructuralPlan {
            layers: vec![SparseStructuralLayerPlan::Validity {
                num_slots: 3 * len,
                validity: SparseValiditySet {
                    meaning: SparseValidityMeaning::ValidPositions,
                    positions: positions.clone(),
                },
            }],
            num_visible_items: 3 * len,
        };
        assert_eq!(materialized_positions(&plan, 3 * len).unwrap(), positions);
    }

    #[tokio::test]
    async fn corrupt_chunk_metadata_is_rejected_during_initialization() {
        let array: ArrayRef = Arc::new(Int32Array::from_iter_values(0..64));
        let (page, field, mut recorder) = page(array.clone(), flat_plan(&array), 64);
        let mut data = recorder.data.to_vec();
        data[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        Arc::get_mut(&mut recorder).unwrap().data = Bytes::from(data);
        let io: Arc<dyn EncodingsIo> = recorder;
        let mut s = scheduler(&page, &field).unwrap();
        let error = initialize(&mut s, &io).await.err().unwrap();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("byte sum"));
    }
}
