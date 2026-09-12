// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::{Error, Result, datatypes::Field};
use lance_encoding::{
    compression::{
        BlockCompressor, CompressionStrategy, field_metadata_params, finalize_miniblock_compressor,
        try_bitpacking_block, try_bitpacking_miniblock, try_byte_stream_split_miniblock,
        try_fixed_packed_struct_miniblock, try_fixed_u8_rle_block, try_fixed_u8_rle_miniblock,
        try_general_block, try_raw_block, try_raw_fixed_size_list_miniblock,
        try_raw_fixed_width_miniblock, try_raw_per_value, try_uncompressed_fixed_width_miniblock,
        try_variable_packed_struct_per_value, try_variable_width_miniblock,
        try_variable_width_per_value,
    },
    compression_config::{CompressionFieldParams, CompressionParams},
    data::DataBlock,
    encodings::logical::primitive::{fullzip::PerValueCompressor, miniblock::MiniBlockCompressor},
};

#[derive(Debug, Clone)]
pub(super) struct Strategy {
    params: CompressionParams,
}

impl Strategy {
    pub(super) fn new(params: CompressionParams) -> Self {
        Self { params }
    }

    fn field_params(&self, field: &Field) -> CompressionFieldParams {
        let mut params = self
            .params
            .get_field_params(&field.name, &field.data_type());
        params.merge(&field_metadata_params(field));
        params
    }
}

impl CompressionStrategy for Strategy {
    fn create_miniblock_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        let params = self.field_params(field);
        let compressor =
            if let Some(compressor) = try_uncompressed_fixed_width_miniblock(data, &params) {
                compressor
            } else if let Some(compressor) = try_byte_stream_split_miniblock(data, &params) {
                compressor
            } else if let Some(compressor) = try_fixed_u8_rle_miniblock(data, &params) {
                compressor
            } else if let Some(compressor) = try_bitpacking_miniblock(data) {
                compressor
            } else if let Some(compressor) = try_raw_fixed_width_miniblock(data) {
                compressor
            } else if let Some(compressor) = try_variable_width_miniblock(field, data, &params)? {
                compressor
            } else if let Some(compressor) = try_fixed_packed_struct_miniblock(data)? {
                compressor
            } else if let Some(compressor) = try_raw_fixed_size_list_miniblock(data) {
                compressor
            } else {
                return Err(Error::not_supported_source(
                    format!(
                        "Mini-block compression not yet supported for block type {}",
                        data.name()
                    )
                    .into(),
                ));
            };
        finalize_miniblock_compressor(data, compressor, &params)
    }

    fn create_per_value(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn PerValueCompressor>> {
        let params = self.field_params(field);
        if let Some(compressor) = try_raw_per_value(data) {
            return Ok(compressor);
        }
        if let Some(compressor) =
            try_variable_packed_struct_per_value(Arc::new(self.clone()), field, data)?
        {
            return Ok(compressor);
        }
        if let Some(compressor) = try_variable_width_per_value(field, data, &params)? {
            return Ok(compressor);
        }
        Err(Error::not_supported_source(
            format!(
                "Per-value compression not yet supported for block type {}",
                data.name()
            )
            .into(),
        ))
    }

    fn create_block_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn BlockCompressor>> {
        let params = self.field_params(field);
        if let Some(compressor) = try_fixed_u8_rle_block(data, &params)? {
            return Ok(compressor);
        }
        if let Some(compressor) = try_bitpacking_block(data) {
            return Ok(compressor);
        }
        if let Some(compressor) = try_general_block(data, &params)? {
            return Ok(compressor);
        }
        if let Some(compressor) = try_raw_block(data) {
            return Ok(compressor);
        }
        Err(Error::not_supported_source(
            format!(
                "Block compression not yet supported for block type {}",
                data.name()
            )
            .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_schema::{DataType, Field as ArrowField};
    use lance_encoding::buffer::LanceBuffer;
    use lance_encoding::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
    use lance_encoding::statistics::ComputeStat;

    /// A 128-bit block of low-magnitude values, the shape a `Decimal128` column produces.
    fn decimal128_block() -> DataBlock {
        let num_values = 2048u64;
        let data: Vec<u128> = (0..num_values).map(|i| (i % 1000) as u128).collect();
        let mut block = FixedWidthDataBlock {
            bits_per_value: 128,
            data: LanceBuffer::reinterpret_vec(data),
            num_values,
            block_info: BlockInfo::default(),
        };
        block.compute_stat();
        DataBlock::FixedWidth(block)
    }

    fn decimal_field() -> Field {
        let arrow_field = ArrowField::new("decimal", DataType::Decimal128(38, 0), true);
        let mut field = Field::try_from(&arrow_field).unwrap();
        field.id = -1;
        field
    }

    /// 128-bit bitpacking arrived in 2.3. A 2.2 reader decodes 8/16/32/64-bit bitpacking
    /// alone, so this strategy must leave a 128-bit block to another codec.
    #[test]
    fn u128_values_are_not_bitpacked() {
        let strategy = Strategy::new(CompressionParams::default());

        let compressor = strategy
            .create_block_compressor(&decimal_field(), &decimal128_block())
            .unwrap();

        let debug_str = format!("{compressor:?}");
        assert!(
            !debug_str.contains("Bitpacking"),
            "2.2 must not bitpack 128-bit values, got: {debug_str}"
        );
    }
}
