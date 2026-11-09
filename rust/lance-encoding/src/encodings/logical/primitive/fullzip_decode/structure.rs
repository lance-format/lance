// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Read FullZip repetition/definition data while skipping every leaf value.

use futures::{StreamExt, TryStreamExt, stream};

use super::*;

#[derive(Debug, Clone, Copy)]
enum ValueSize {
    Fixed(u64),
    ScalarFixed(u64),
    Variable(u8),
}

#[derive(Debug, Default)]
struct RowStructure {
    rep: Vec<u16>,
    def: Vec<u16>,
    visible: u64,
}

fn corrupt(message: impl Into<String>) -> Error {
    Error::corrupt_file_named("full_zip_list_structure", message.into())
}

impl FullZipScheduler {
    pub(super) fn schedule_list_structure(
        &self,
        ranges: &[Range<u64>],
        io: &Arc<dyn EncodingsIo>,
        projection: StructuralProjection,
    ) -> Result<Vec<PageLoadTask>> {
        let index = self.rep_index;
        if ranges
            .iter()
            .any(|range| range.start > range.end || range.end > self.rows_in_page)
        {
            return Err(corrupt("List row selection exceeds page bounds"));
        }
        let ranges = ranges.to_vec();
        let rows = ranges.iter().try_fold(0_u64, |sum, range| {
            sum.checked_add(range.end - range.start)
                .ok_or_else(|| corrupt("Selected row count overflow"))
        })?;
        let details = self.details.clone();
        let priority = self.priority;
        let data_start = self.data_buf_position;
        let data_size = self.data_buf_size;
        let value_size = match &details.value_decompressor {
            PerValueDecompressor::Fixed(decoder) => {
                let bits = decoder.bits_per_value();
                if !bits.is_multiple_of(8) {
                    return Err(corrupt("FullZip value width is not byte aligned"));
                }
                if projection.data_type.is_primitive()
                    || matches!(
                        projection.data_type,
                        arrow_schema::DataType::Boolean
                            | arrow_schema::DataType::Null
                            | arrow_schema::DataType::FixedSizeBinary(_)
                            | arrow_schema::DataType::Binary
                            | arrow_schema::DataType::LargeBinary
                            | arrow_schema::DataType::Utf8
                            | arrow_schema::DataType::LargeUtf8
                    )
                {
                    ValueSize::ScalarFixed(bits / 8)
                } else {
                    ValueSize::Fixed(bits / 8)
                }
            }
            PerValueDecompressor::Variable(_) => {
                let width = self.bits_per_offset / 8;
                if !matches!(width, 4 | 8) {
                    return Err(corrupt("Unsupported value length width"));
                }
                ValueSize::Variable(width)
            }
        };
        // Coalescing headers across a gap would fetch precisely the image bytes
        // this projection omits. File-backed I/O supports exact adjacent-only reads.
        let io = io.with_coalesce_gap(0).unwrap_or_else(|| io.clone());
        let header_io = StructuralMetadataIo::wrap(io.clone(), self.structural_cache.clone());
        let cached = self.cached_state.clone();
        let decoder_fut = async move {
            let mut row_ranges = Vec::new();
            for range in ranges.iter().filter(|r| !r.is_empty()) {
                let Some(index) = index else {
                    let (ValueSize::Fixed(bytes) | ValueSize::ScalarFixed(bytes)) = value_size
                    else {
                        return Err(corrupt("Variable values have no row index"));
                    };
                    let row_bytes = bytes
                        .checked_add(details.ctrl_word_parser.bytes_per_word() as u64)
                        .ok_or_else(|| corrupt("Fixed row size overflow"))?;
                    for row in range.clone() {
                        let start = row
                            .checked_mul(row_bytes)
                            .ok_or_else(|| corrupt("Fixed row offset overflow"))?;
                        let end = start
                            .checked_add(row_bytes)
                            .filter(|end| *end <= data_size)
                            .ok_or_else(|| corrupt("Fixed row exceeds data buffer"))?;
                        row_ranges.push(
                            data_start
                                .checked_add(start)
                                .ok_or_else(|| corrupt("File offset overflow"))?
                                ..data_start
                                    .checked_add(end)
                                    .ok_or_else(|| corrupt("File offset overflow"))?,
                        );
                    }
                    continue;
                };
                let start = range
                    .start
                    .checked_mul(index.bytes_per_value)
                    .ok_or_else(|| corrupt("Row index offset overflow"))?;
                let end = range
                    .end
                    .checked_add(1)
                    .and_then(|n| n.checked_mul(index.bytes_per_value))
                    .ok_or_else(|| corrupt("Row index end overflow"))?;
                let bytes = if let Some(cached) = &cached {
                    if end > cached.rep_index_buffer.len() as u64 {
                        return Err(corrupt("Cached row index is truncated"));
                    }
                    cached
                        .rep_index_buffer
                        .slice_with_length(start as usize, (end - start) as usize)
                } else {
                    LanceBuffer::from_bytes(
                        io.submit_single(
                            index
                                .buf_position
                                .checked_add(start)
                                .ok_or_else(|| corrupt("Index file offset overflow"))?
                                ..index
                                    .buf_position
                                    .checked_add(end)
                                    .ok_or_else(|| corrupt("Index file offset overflow"))?,
                            priority,
                        )
                        .await?,
                        1,
                    )
                };
                if bytes.len() as u64 != end - start {
                    return Err(corrupt("Short row index read"));
                }
                let offsets =
                    ByteUnpacker::new(bytes, index.bytes_per_value as usize).collect::<Vec<_>>();
                if offsets.len() != (range.end - range.start + 1) as usize {
                    return Err(corrupt("Truncated list row index"));
                }
                for pair in offsets.windows(2) {
                    if pair[0] >= pair[1] || pair[1] > data_size {
                        return Err(corrupt("Invalid list row byte range"));
                    }
                    row_ranges.push(
                        data_start
                            .checked_add(pair[0])
                            .ok_or_else(|| corrupt("File offset overflow"))?
                            ..data_start
                                .checked_add(pair[1])
                                .ok_or_else(|| corrupt("File offset overflow"))?,
                    );
                }
            }
            let row_structure = stream::iter(row_ranges)
                .map(|range| {
                    read_row(
                        range,
                        header_io.clone(),
                        details.clone(),
                        value_size,
                        projection.mode,
                        priority,
                    )
                })
                .buffered(super::super::structure::METADATA_READ_CONCURRENCY)
                .try_collect::<Vec<_>>()
                .await?;
            Ok(Box::new(ListStructureDecoder {
                rows: row_structure.into(),
                position: 0,
                details,
                projection,
            }) as Box<dyn StructuralPageDecoder>)
        }
        .boxed();
        Ok(vec![PageLoadTask {
            decoder_fut,
            num_rows: rows,
        }])
    }
}

async fn read_row(
    range: Range<u64>,
    io: Arc<dyn EncodingsIo>,
    details: Arc<FullZipDecodeDetails>,
    value_size: ValueSize,
    mode: ListReadMode,
    priority: u64,
) -> Result<RowStructure> {
    let control_width = details.ctrl_word_parser.bytes_per_word() as u64;
    if control_width == 0 {
        return Ok(RowStructure {
            visible: 1,
            ..Default::default()
        });
    }
    if range.end - range.start < control_width {
        return Err(corrupt("List row has no complete control word"));
    }
    if mode == ListReadMode::NonEmpty && details.def_meaning.len() == 2 && details.max_rep == 1 {
        if range.end - range.start > control_width {
            // A null/empty outer list occupies just one control. More bytes
            // imply a nonempty list, including multiple zero-width null items.
            return Ok(RowStructure {
                rep: vec![1],
                def: vec![0],
                visible: 1,
            });
        }
        if details.def_meaning[0] == DefinitionInterpretation::AllValidItem
            && !matches!(value_size, ValueSize::Fixed(0) | ValueSize::ScalarFixed(0))
            && matches!(
                details.def_meaning[1],
                DefinitionInterpretation::NullableList | DefinitionInterpretation::EmptyableList
            )
        {
            return Ok(RowStructure {
                rep: vec![1],
                def: vec![1],
                visible: 0,
            });
        }
    }
    if let ValueSize::ScalarFixed(bytes) = value_size
        && details.def_meaning.len() == 2
        && details.max_rep == 1
        && range.end - range.start > control_width
    {
        // Scalar item validity cannot affect array_length at any dimension.
        // Fixed-size-list leaves are deliberately excluded: their validity can
        // affect a request for an inner dimension.
        let stride = bytes
            .checked_add(control_width)
            .ok_or_else(|| corrupt("Item stride overflow"))?;
        let row_bytes = range.end - range.start;
        if !row_bytes.is_multiple_of(stride) {
            return Err(corrupt("Fixed scalar row is not a whole number of items"));
        }
        let count = usize::try_from(row_bytes / stride)
            .map_err(|_| corrupt("List length exceeds addressable memory"))?;
        let mut rep = vec![0; count];
        rep[0] = 1;
        return Ok(RowStructure {
            rep,
            def: vec![0; count],
            visible: count as u64,
        });
    }
    let mut row = RowStructure::default();
    let mut cursor = range.start;
    let mut prefix = Bytes::new();
    let mut prefix_start = cursor;
    while cursor < range.end {
        let end = cursor
            .checked_add(control_width)
            .ok_or_else(|| corrupt("Control offset overflow"))?;
        if end > range.end {
            return Err(corrupt("Truncated list control word"));
        }
        if cursor < prefix_start || end > prefix_start + prefix.len() as u64 {
            let extra = match value_size {
                ValueSize::Variable(width) => u64::from(width),
                _ => 0,
            };
            let prefix_end = end + extra.min(range.end - end);
            // The first valid variable item needs a length header before its
            // payload. Null/invisible controls can only move that payload later,
            // so this prefix never crosses into a value, even with leading nulls.
            prefix = io.submit_single(cursor..prefix_end, priority).await?;
            if prefix.len() as u64 != prefix_end - cursor {
                return Err(corrupt("Short list header read"));
            }
            prefix_start = cursor;
        }
        let control_start = (cursor - prefix_start) as usize;
        let control = &prefix[control_start..control_start + control_width as usize];
        let description =
            details
                .ctrl_word_parser
                .parse_desc(control, details.max_rep, details.max_visible_def);
        if description.is_new_row != (cursor == range.start) {
            return Err(corrupt("Unexpected list row boundary"));
        }
        details
            .ctrl_word_parser
            .parse(control, &mut row.rep, &mut row.def);
        row.visible += u64::from(description.is_visible);
        cursor = end;
        if mode == ListReadMode::NonEmpty && details.def_meaning.len() == 2 && details.max_rep == 1
        {
            return Ok(row);
        }
        if let ValueSize::Fixed(bytes) | ValueSize::ScalarFixed(bytes) = value_size {
            if description.is_visible {
                cursor = cursor
                    .checked_add(bytes)
                    .ok_or_else(|| corrupt("Fixed payload offset overflow"))?;
                if cursor > range.end {
                    return Err(corrupt("Fixed payload crosses row boundary"));
                }
            }
        } else if description.is_valid_item
            && let ValueSize::Variable(length_width) = value_size
        {
            let end = cursor
                .checked_add(u64::from(length_width))
                .ok_or_else(|| corrupt("Length offset overflow"))?;
            if end > range.end {
                return Err(corrupt("Truncated binary length"));
            }
            let mut length = [0_u8; 8];
            let available =
                (prefix_start + prefix.len() as u64 - cursor).min(u64::from(length_width)) as usize;
            let offset = (cursor - prefix_start) as usize;
            length[..available].copy_from_slice(&prefix[offset..offset + available]);
            if available < usize::from(length_width) {
                let bytes = io
                    .submit_single(cursor + available as u64..end, priority)
                    .await?;
                if bytes.len() != usize::from(length_width) - available {
                    return Err(corrupt("Short binary length read"));
                }
                length[available..usize::from(length_width)].copy_from_slice(&bytes);
            }
            cursor = end
                .checked_add(u64::from_le_bytes(length))
                .ok_or_else(|| corrupt("Binary payload offset overflow"))?;
            if cursor > range.end {
                return Err(corrupt("Binary payload crosses list row boundary"));
            }
        }
    }
    Ok(row)
}

#[derive(Debug)]
struct ListStructureDecoder {
    rows: Arc<[RowStructure]>,
    position: usize,
    details: Arc<FullZipDecodeDetails>,
    projection: StructuralProjection,
}

impl StructuralPageDecoder for ListStructureDecoder {
    fn drain(&mut self, num_rows: u64) -> Result<Box<dyn DecodePageTask>> {
        let end = self
            .position
            .checked_add(
                usize::try_from(num_rows)
                    .map_err(|_| corrupt("Row count exceeds addressable memory"))?,
            )
            .ok_or_else(|| corrupt("Row count overflow"))?;
        if end > self.rows.len() {
            return Err(corrupt("Drain exceeds available list rows"));
        }
        let task = ListStructureTask {
            rows: self.rows.clone(),
            range: self.position..end,
            details: self.details.clone(),
            projection: self.projection.clone(),
        };
        self.position = end;
        Ok(Box::new(task))
    }

    fn num_rows(&self) -> u64 {
        self.rows.len() as u64
    }
}

#[derive(Debug)]
struct ListStructureTask {
    rows: Arc<[RowStructure]>,
    range: Range<usize>,
    details: Arc<FullZipDecodeDetails>,
    projection: StructuralProjection,
}

impl DecodePageTask for ListStructureTask {
    fn decode(self: Box<Self>) -> Result<DecodedPage> {
        let rows = &self.rows[self.range];
        let visible = rows.iter().map(|r| r.visible).sum::<u64>();
        let rep = rows
            .iter()
            .flat_map(|r| r.rep.iter().copied())
            .collect::<Vec<_>>();
        let has_defs = self
            .details
            .def_meaning
            .iter()
            .any(|d| d.num_def_levels() > 0);
        let def = has_defs.then(|| {
            rows.iter()
                .flat_map(|r| {
                    if r.def.is_empty() {
                        vec![0; r.rep.len()]
                    } else {
                        r.def.clone()
                    }
                })
                .collect()
        });
        Ok(DecodedPage {
            data: self.projection.placeholder_values(visible)?,
            repdef: RepDefUnraveler::new(
                (!rep.is_empty()).then_some(rep),
                def,
                self.details.def_meaning.clone(),
                visible,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encodings::logical::primitive::structure::tests::GuardIo;
    use std::sync::Mutex;

    #[derive(Debug)]
    struct NoValueDecode;
    impl FixedPerValueDecompressor for NoValueDecode {
        fn bits_per_value(&self) -> u64 {
            32
        }
        fn decompress(&self, _: FixedWidthDataBlock, _: u64) -> Result<crate::data::DataBlock> {
            panic!("structure reads must not decode values")
        }
    }

    fn details(layers: Vec<DefinitionInterpretation>, def_bits: u8) -> Arc<FullZipDecodeDetails> {
        Arc::new(FullZipDecodeDetails {
            value_decompressor: PerValueDecompressor::Fixed(Arc::new(NoValueDecode)),
            max_visible_def: max_visible_level(&layers),
            def_meaning: layers.into(),
            ctrl_word_parser: ControlWordParser::new(1, def_bits),
            max_rep: 1,
        })
    }

    #[tokio::test]
    async fn length_reads_headers_and_skips_payload() {
        let mut bytes = vec![2];
        bytes.extend(4_u64.to_le_bytes());
        bytes.extend([0xDE, 0xAD, 0xBE, 0xEF]);
        bytes.extend([1, 0]);
        bytes.extend(0_u64.to_le_bytes());
        let io = Arc::new(GuardIo {
            bytes: bytes.into(),
            payloads: std::iter::once(9..13).collect(),
            reads: Mutex::default(),
        });
        let row = read_row(
            0..23,
            io.clone(),
            details(
                vec![
                    DefinitionInterpretation::NullableItem,
                    DefinitionInterpretation::AllValidList,
                ],
                1,
            ),
            ValueSize::Variable(8),
            ListReadMode::Length,
            0,
        )
        .await
        .unwrap();
        assert_eq!(row.visible, 3);
        assert_eq!(row.rep, vec![1, 0, 0]);
        assert_eq!(row.def, vec![0, 1, 0]);
        assert_eq!(*io.reads.lock().unwrap(), vec![0..9, 13..22, 22..23]);
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn variable_prefixes_skip_payload_after_null_controls(
        #[values(0, 1, 8, 9, 17)] leading_nulls: usize,
        #[values(4, 8)] length_width: u8,
    ) {
        let mut bytes = (0..leading_nulls)
            .map(|i| if i == 0 { 3 } else { 1 })
            .collect::<Vec<_>>();
        bytes.push(if leading_nulls == 0 { 2 } else { 0 });
        bytes.extend_from_slice(&7_u64.to_le_bytes()[..length_width as usize]);
        let payload_start = bytes.len() as u64;
        bytes.extend([0xA5; 7]);
        bytes.push(0);
        bytes.extend_from_slice(&0_u64.to_le_bytes()[..length_width as usize]);
        bytes.push(1);
        let end = bytes.len() as u64;
        let io = Arc::new(GuardIo {
            bytes: bytes.into(),
            payloads: std::iter::once(payload_start..payload_start + 7).collect(),
            reads: Mutex::default(),
        });
        let row = read_row(
            0..end,
            io,
            details(
                vec![
                    DefinitionInterpretation::NullableItem,
                    DefinitionInterpretation::AllValidList,
                ],
                1,
            ),
            ValueSize::Variable(length_width),
            ListReadMode::Length,
            0,
        )
        .await
        .unwrap();
        assert_eq!(row.visible, leading_nulls as u64 + 3);
        assert_eq!(row.rep[0], 1);
        assert!(row.rep[1..].iter().all(|rep| *rep == 0));
        let mut expected = vec![1; leading_nulls];
        expected.extend([0, 0, 1]);
        assert_eq!(row.def, expected);
    }

    #[rstest::rstest]
    #[case::length(ListReadMode::Length, 2)]
    #[case::presence(ListReadMode::NonEmpty, 1)]
    #[tokio::test]
    async fn fixed_scalar_length_uses_row_index_span(
        #[case] mode: ListReadMode,
        #[case] expected: u64,
    ) {
        let io = Arc::new(GuardIo {
            bytes: Bytes::from_static(&[4, 1, 2, 3, 4, 1, 0, 0, 0, 0]),
            payloads: vec![1..5, 6..10],
            reads: Mutex::default(),
        });
        let row = read_row(
            0..10,
            io.clone(),
            details(
                vec![
                    DefinitionInterpretation::NullableItem,
                    DefinitionInterpretation::NullableAndEmptyableList,
                ],
                2,
            ),
            ValueSize::ScalarFixed(4),
            mode,
            0,
        )
        .await
        .unwrap();
        assert_eq!(row.visible, expected);
        assert_eq!(row.rep.len(), expected as usize);
        assert!(io.reads.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn zero_width_item_is_still_nonempty() {
        let io = Arc::new(GuardIo {
            bytes: Bytes::from_static(&[2]),
            payloads: vec![],
            reads: Mutex::default(),
        });
        let row = read_row(
            0..1,
            io,
            details(
                vec![
                    DefinitionInterpretation::AllValidItem,
                    DefinitionInterpretation::NullableList,
                ],
                1,
            ),
            ValueSize::Fixed(0),
            ListReadMode::NonEmpty,
            0,
        )
        .await
        .unwrap();
        assert_eq!(row.visible, 1);
        assert_eq!(row.def, vec![0]);
    }

    #[rstest::rstest]
    #[case::null_list(6, 0, 2)]
    #[case::empty_list(7, 0, 3)]
    #[case::null_element(5, 1, 1)]
    #[tokio::test]
    async fn presence_distinguishes_null_and_empty(
        #[case] control: u8,
        #[case] visible: u64,
        #[case] definition: u16,
    ) {
        let io = Arc::new(GuardIo {
            bytes: Bytes::from(vec![control]),
            payloads: vec![],
            reads: Mutex::default(),
        });
        let row = read_row(
            0..1,
            io,
            details(
                vec![
                    DefinitionInterpretation::NullableItem,
                    DefinitionInterpretation::NullableAndEmptyableList,
                ],
                2,
            ),
            ValueSize::Variable(8),
            ListReadMode::NonEmpty,
            0,
        )
        .await
        .unwrap();
        assert_eq!(row.visible, visible);
        assert_eq!(row.def, vec![definition]);
    }

    #[tokio::test]
    async fn corrupt_length_does_not_read_beyond_row() {
        let mut bytes = vec![1];
        bytes.extend(u64::MAX.to_le_bytes());
        let io = Arc::new(GuardIo {
            bytes: bytes.into(),
            payloads: vec![],
            reads: Mutex::default(),
        });
        let error = read_row(
            0..9,
            io,
            details(
                vec![
                    DefinitionInterpretation::AllValidItem,
                    DefinitionInterpretation::AllValidList,
                ],
                0,
            ),
            ValueSize::Variable(8),
            ListReadMode::Length,
            0,
        )
        .await
        .unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("payload offset overflow"));
    }
}
