// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use lance_core::{Error, Result, datatypes::Field};
use lance_encoding::compression::PerValueGeneralCompression;
use lance_encoding::compression::try_packed_struct_per_value;
use lance_encoding::constants::PCODEC_COMPRESSION;
use lance_encoding::{
    compression::{
        BlockCompressor, CompressionStrategy, field_metadata_params, finalize_miniblock_compressor,
        try_alp_miniblock, try_bitpacking_miniblock, try_byte_stream_split_miniblock,
        try_child_rle_miniblock, try_fixed_packed_struct_miniblock, try_fixed_size_list_block,
        try_fixed_u8_rle_miniblock, try_general_block, try_pcodec_miniblock, try_raw_block,
        try_raw_fixed_size_list_miniblock, try_raw_fixed_width_miniblock, try_raw_per_value,
        try_uncompressed_fixed_width_miniblock, try_variable_rle_block,
        try_variable_width_miniblock_with_generic_offsets, try_variable_width_per_value,
        try_wide_bitpacking_block, try_wide_bitpacking_miniblock_with_target,
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

    /// Parameters for buffers other than value mini-blocks: dictionary indices,
    /// dictionary values and block-compressed sequences. pcodec only compresses
    /// value mini-blocks, so these buffers keep their default compression.
    fn auxiliary_params(&self, field: &Field) -> CompressionFieldParams {
        let mut params = self.field_params(field);
        if params.compression.as_deref() == Some(PCODEC_COMPRESSION) {
            params.compression = None;
            params.compression_level = None;
        }
        params
    }
}

impl CompressionStrategy for Strategy {
    fn minichunk_size(&self, field: &Field) -> Result<Option<u64>> {
        self.params.minichunk_size(field, None)
    }

    fn allows_byte_wide_fsl_booleans(&self) -> bool {
        true
    }

    fn create_dictionary_index_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        let params = self.auxiliary_params(field);
        // Dictionary lookup already adds an indirection. Keep index decoding
        // simple instead of stacking child decompression and run expansion.
        let compressor = try_uncompressed_fixed_width_miniblock(data, &params)
            .or_else(|| try_byte_stream_split_miniblock(data, &params))
            .or_else(|| try_fixed_u8_rle_miniblock(data, &params))
            .or_else(|| try_bitpacking_miniblock(data))
            .or_else(|| try_raw_fixed_width_miniblock(data))
            .ok_or_else(|| Error::internal("Dictionary indices must be fixed-width values"))?;
        finalize_miniblock_compressor(data, compressor, &params)
    }

    fn create_miniblock_compressor(
        &self,
        field: &Field,
        data: &DataBlock,
    ) -> Result<Box<dyn MiniBlockCompressor>> {
        let params = self.field_params(field);
        if let Some(compressor) = try_pcodec_miniblock(field, data, &params)? {
            return Ok(compressor);
        }
        let compressor = if let Some(compressor) =
            try_uncompressed_fixed_width_miniblock(data, &params)
        {
            compressor
        } else if let Some(compressor) = try_alp_miniblock(field, data, &params) {
            compressor
        } else if let Some(compressor) = try_byte_stream_split_miniblock(data, &params) {
            compressor
        } else if let Some(compressor) = try_child_rle_miniblock(data, &params) {
            compressor
        } else if let Some(compressor) = try_wide_bitpacking_miniblock_with_target(data, &params) {
            compressor
        } else if let Some(compressor) = try_raw_fixed_width_miniblock(data) {
            compressor
        } else if let Some(compressor) =
            try_variable_width_miniblock_with_generic_offsets(field, data, &params)?
        {
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
        if let Some(compressor) =
            lance_encoding::compression::try_tensor_per_value(field, data, &params)?
        {
            return Ok(compressor);
        }
        if let Some(compressor) = try_raw_per_value(data) {
            return Ok(compressor);
        }
        if let Some(compressor) = try_packed_struct_per_value(Arc::new(self.clone()), field, data)?
        {
            return Ok(compressor);
        }
        if let Some(compressor) =
            try_variable_width_per_value(field, data, &params, PerValueGeneralCompression::Any)?
        {
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
        let params = self.auxiliary_params(field);
        if let Some(compressor) = try_fixed_size_list_block(data, &params)? {
            return Ok(compressor);
        }
        if let Some(compressor) = try_variable_rle_block(data, &params)? {
            return Ok(compressor);
        }
        if let Some(compressor) = try_wide_bitpacking_block(data) {
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

    /// 128-bit bitpacking is an encoding 2.3 introduced, so this is the one strategy that
    /// composes the selector offering it.
    #[test]
    fn u128_values_are_bitpacked() {
        let strategy = Strategy::new(CompressionParams::default());

        let miniblock = strategy
            .create_miniblock_compressor(&decimal_field(), &decimal128_block())
            .unwrap();
        let debug_str = format!("{miniblock:?}");
        assert!(
            debug_str.contains("InlineBitpacking { uncompressed_bit_width: 128 }"),
            "2.3 must bitpack 128-bit mini-block values, got: {debug_str}"
        );

        let block = strategy
            .create_block_compressor(&decimal_field(), &decimal128_block())
            .unwrap();
        let debug_str = format!("{block:?}");
        assert!(
            debug_str.contains("OutOfLineBitpacking")
                && debug_str.contains("uncompressed_bit_width: 128"),
            "2.3 must bitpack 128-bit blocks, got: {debug_str}"
        );
    }

    /// A 128-bit inline chunk is 1024 values (16 KiB raw), so it yields to an explicit
    /// General chunk target of any other size, as narrower values do.
    #[rstest::rstest]
    #[case::conflicting_target(4096, false)]
    #[case::matching_target(16384, true)]
    fn u128_bitpacking_honors_explicit_chunk_target(
        #[case] minichunk_size: i64,
        #[case] is_bitpacked: bool,
    ) {
        let mut params = CompressionParams::default();
        params.columns.insert(
            "decimal".into(),
            CompressionFieldParams {
                compression: Some("lz4".into()),
                minichunk_size: Some(minichunk_size),
                ..Default::default()
            },
        );
        let miniblock = Strategy::new(params)
            .create_miniblock_compressor(&decimal_field(), &decimal128_block())
            .unwrap();
        let debug_str = format!("{miniblock:?}");
        assert_eq!(
            debug_str.contains("InlineBitpacking { uncompressed_bit_width: 128 }"),
            is_bitpacked,
            "{debug_str}"
        );
    }
}
