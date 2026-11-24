// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{ops::Range, sync::Arc};

use arrow_schema::DataType;
use futures::{FutureExt, future::BoxFuture};

use lance_core::Result;

use crate::{
    EncodingsIo,
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, NullableDataBlock, VariableWidthBlock},
    decoder::{PageScheduler, PrimitivePageDecoder},
    encoder::{ArrayEncoder, EncodedArray},
    encodings::physical::fsst::{
        ParsedFsstDecompressor, get_fsst_decompressor, parse_fsst_symbol_table,
    },
    format::ProtobufUtils,
};

#[derive(Debug)]
pub struct FsstPageScheduler {
    inner_scheduler: Box<dyn PageScheduler>,
    decompressor: Arc<ParsedFsstDecompressor>,
}

impl FsstPageScheduler {
    pub fn new(inner_scheduler: Box<dyn PageScheduler>, symbol_table: LanceBuffer) -> Self {
        Self {
            inner_scheduler,
            decompressor: Arc::new(parse_fsst_symbol_table(symbol_table.as_ref())),
        }
    }
}

impl PageScheduler for FsstPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let inner_decoder = self
            .inner_scheduler
            .schedule_ranges(ranges, scheduler, top_level_row);
        let decompressor = self.decompressor.clone();

        async move {
            let inner_decoder = inner_decoder.await?;
            Ok(Box::new(FsstPageDecoder {
                inner_decoder,
                decompressor,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

struct FsstPageDecoder {
    inner_decoder: Box<dyn PrimitivePageDecoder>,
    decompressor: Arc<ParsedFsstDecompressor>,
}

impl PrimitivePageDecoder for FsstPageDecoder {
    fn variable_width_bytes(&self, rows_to_skip: u64, num_rows: u64) -> Result<Option<u64>> {
        // FSST decompression expands by at most 8x, so the inner (compressed)
        // byte count times 8 is a safe upper bound on the decoded size.
        Ok(self
            .inner_decoder
            .variable_width_bytes(rows_to_skip, num_rows)?
            .map(|compressed_bytes| compressed_bytes.saturating_mul(8)))
    }

    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        let decompressor = get_fsst_decompressor(&self.decompressor)?;
        let compressed_data = self.inner_decoder.decode(rows_to_skip, num_rows)?;
        let (string_data, nulls) = match compressed_data {
            DataBlock::Nullable(nullable) => {
                let data = nullable.data.as_variable_width().unwrap();
                Result::Ok((data, Some(nullable.nulls)))
            }
            DataBlock::VariableWidth(variable) => Ok((variable, None)),
            _ => panic!("Received non-variable width data from inner decoder"),
        }?;

        let offsets = string_data.offsets.borrow_to_typed_slice::<i32>();
        let bytes = string_data.data.as_ref();

        let (mut decompressed_bytes, decompressed_offsets) = decompressor
            .decompress_into(bytes, offsets.as_ref())
            .map_err(crate::encodings::physical::fsst::map_fsst_error)?;
        decompressed_bytes.shrink_to_fit();

        let new_string_data = DataBlock::VariableWidth(VariableWidthBlock {
            bits_per_offset: 32,
            data: LanceBuffer::from(decompressed_bytes),
            num_values: num_rows,
            offsets: LanceBuffer::reinterpret_vec(decompressed_offsets),
            block_info: BlockInfo::new(),
        });

        if let Some(nulls) = nulls {
            Ok(DataBlock::Nullable(NullableDataBlock {
                data: Box::new(new_string_data),
                nulls,
                block_info: BlockInfo::new(),
            }))
        } else {
            Ok(new_string_data)
        }
    }
}

#[derive(Debug)]
pub struct FsstArrayEncoder {
    inner_encoder: Box<dyn ArrayEncoder>,
}

impl FsstArrayEncoder {
    pub fn new(inner_encoder: Box<dyn ArrayEncoder>) -> Self {
        Self { inner_encoder }
    }
}

impl ArrayEncoder for FsstArrayEncoder {
    fn encode(
        &self,
        data: DataBlock,
        data_type: &DataType,
        buffer_index: &mut u32,
    ) -> lance_core::Result<EncodedArray> {
        let (data, nulls) = match data {
            DataBlock::Nullable(nullable) => {
                let data = nullable.data.as_variable_width().unwrap();
                (data, Some(nullable.nulls))
            }
            DataBlock::VariableWidth(variable) => (variable, None),
            _ => panic!("Expected variable width data block"),
        };
        assert_eq!(data.bits_per_offset, 32);
        let num_values = data.num_values;
        let offsets = data.offsets.borrow_to_typed_slice::<i32>();
        let bytes_data = data.data.as_ref();

        let mut symbol_table = vec![0_u8; fsst::fsst::FSST_SYMBOL_TABLE_SIZE];
        let (dest_values, dest_offsets) =
            fsst::fsst::compress_into(&mut symbol_table, bytes_data, offsets.as_ref())?;

        let dest_offset = LanceBuffer::reinterpret_vec(dest_offsets);
        let dest_values = LanceBuffer::from(dest_values);
        let dest_data = DataBlock::VariableWidth(VariableWidthBlock {
            bits_per_offset: 32,
            data: dest_values,
            num_values,
            offsets: dest_offset,
            block_info: BlockInfo::new(),
        });

        let data_block = if let Some(nulls) = nulls {
            DataBlock::Nullable(NullableDataBlock {
                data: Box::new(dest_data),
                nulls,
                block_info: BlockInfo::new(),
            })
        } else {
            dest_data
        };

        let inner_encoded = self
            .inner_encoder
            .encode(data_block, data_type, buffer_index)?;

        let encoding = ProtobufUtils::fsst(inner_encoded.encoding, symbol_table);

        Ok(EncodedArray {
            data: inner_encoded.data,
            encoding,
        })
    }
}
