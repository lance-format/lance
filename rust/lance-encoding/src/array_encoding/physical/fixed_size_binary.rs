// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_buffer::ScalarBuffer;
use futures::{FutureExt, future::BoxFuture};
use lance_core::Result;

use crate::{
    EncodingsIo,
    buffer::LanceBuffer,
    data::{BlockInfo, DataBlock, VariableWidthBlock},
    decoder::{PageScheduler, PrimitivePageDecoder},
};

/// A scheduler for fixed size binary data
#[derive(Debug)]
pub struct FixedSizeBinaryPageScheduler {
    bytes_scheduler: Box<dyn PageScheduler>,
    byte_width: u32,
    bytes_per_offset: u32,
}

impl FixedSizeBinaryPageScheduler {
    pub fn new(
        bytes_scheduler: Box<dyn PageScheduler>,
        byte_width: u32,
        bytes_per_offset: u32,
    ) -> Self {
        Self {
            bytes_scheduler,
            byte_width,
            bytes_per_offset,
        }
    }
}

impl PageScheduler for FixedSizeBinaryPageScheduler {
    fn schedule_ranges(
        &self,
        ranges: &[std::ops::Range<u64>],
        scheduler: &Arc<dyn EncodingsIo>,
        top_level_row: u64,
    ) -> BoxFuture<'static, Result<Box<dyn PrimitivePageDecoder>>> {
        let expanded_ranges = ranges
            .iter()
            .map(|range| {
                (range.start * self.byte_width as u64)..(range.end * self.byte_width as u64)
            })
            .collect::<Vec<_>>();

        let bytes_page_decoder =
            self.bytes_scheduler
                .schedule_ranges(&expanded_ranges, scheduler, top_level_row);

        let byte_width = self.byte_width as u64;
        let bytes_per_offset = self.bytes_per_offset;

        async move {
            let bytes_decoder = bytes_page_decoder.await?;
            Ok(Box::new(FixedSizeBinaryDecoder {
                bytes_decoder,
                byte_width,
                bytes_per_offset,
            }) as Box<dyn PrimitivePageDecoder>)
        }
        .boxed()
    }
}

pub struct FixedSizeBinaryDecoder {
    bytes_decoder: Box<dyn PrimitivePageDecoder>,
    byte_width: u64,
    bytes_per_offset: u32,
}

impl PrimitivePageDecoder for FixedSizeBinaryDecoder {
    fn variable_width_bytes(&self, _rows_to_skip: u64, num_rows: u64) -> Result<Option<u64>> {
        Ok(Some(num_rows * self.byte_width))
    }

    fn decode(&self, rows_to_skip: u64, num_rows: u64) -> Result<DataBlock> {
        let rows_to_skip = rows_to_skip * self.byte_width;
        let num_bytes = num_rows * self.byte_width;
        let bytes = self.bytes_decoder.decode(rows_to_skip, num_bytes)?;
        let bytes = bytes.as_fixed_width().unwrap();
        debug_assert_eq!(bytes.bits_per_value, self.byte_width * 8);

        let offsets_buffer = match self.bytes_per_offset {
            8 => {
                let offsets_vec = (0..(num_rows + 1))
                    .map(|i| i * self.byte_width)
                    .collect::<Vec<_>>();

                ScalarBuffer::from(offsets_vec).into_inner()
            }
            4 => {
                let offsets_vec = (0..(num_rows as u32 + 1))
                    .map(|i| i * self.byte_width as u32)
                    .collect::<Vec<_>>();

                ScalarBuffer::from(offsets_vec).into_inner()
            }
            _ => panic!("Unsupported offsets type"),
        };

        let string_data = DataBlock::VariableWidth(VariableWidthBlock {
            bits_per_offset: (self.bytes_per_offset * 8) as u8,
            data: bytes.data,
            num_values: num_rows,
            offsets: LanceBuffer::from(offsets_buffer),
            block_info: BlockInfo::new(),
        });

        Ok(string_data)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, FixedSizeBinaryArray, StringArray};
    use arrow_buffer::Buffer;
    use arrow_data::ArrayData;
    use arrow_schema::DataType;

    use crate::array_encoding::physical::fixed_size_binary::FixedSizeBinaryDecoder;
    use crate::data::{DataBlock, FixedWidthDataBlock};
    use crate::decoder::PrimitivePageDecoder;
    struct FixedWidthCloningPageDecoder {
        data_block: FixedWidthDataBlock,
    }

    impl PrimitivePageDecoder for FixedWidthCloningPageDecoder {
        // clone the given data block as decoded data block
        fn decode(
            &self,
            _rows_to_skip: u64,
            _num_rows: u64,
        ) -> lance_core::error::Result<DataBlock> {
            Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
                data: self.data_block.data.deep_copy(),
                bits_per_value: self.data_block.bits_per_value,
                num_values: self.data_block.num_values,
                block_info: self.data_block.block_info.clone(),
            }))
        }
    }

    #[test]
    fn test_fixed_size_binary_decoder() {
        let values: [u8; 6] = *b"aaabbb";
        let num_values = 2u64;
        let byte_width = 3;
        let array_data = ArrayData::builder(DataType::FixedSizeBinary(byte_width))
            .len(num_values as usize)
            .add_buffer(Buffer::from(&values[..]))
            .build()
            .unwrap();
        let fixed_size_binary_array = FixedSizeBinaryArray::from(array_data);
        let arrays = vec![Arc::new(fixed_size_binary_array) as ArrayRef];
        let fixed_width_data_block = DataBlock::from_arrays(&arrays, num_values);
        assert_eq!(fixed_width_data_block.name(), "FixedWidth");

        let bytes_decoder = FixedWidthCloningPageDecoder {
            data_block: fixed_width_data_block.as_fixed_width().unwrap(),
        };
        let decoder = FixedSizeBinaryDecoder {
            bytes_decoder: Box::new(bytes_decoder),
            byte_width: byte_width as u64,
            bytes_per_offset: 4, // 32-bits offset binary
        };

        let decoded_binary = decoder.decode(0, num_values).unwrap();
        let maybe_data = decoded_binary.into_arrow(DataType::Utf8, true);
        assert!(maybe_data.is_ok());
        let data = maybe_data.unwrap();
        let string_array = StringArray::from(data);
        assert_eq!(string_array.len(), num_values as usize);
        assert_eq!(string_array.value(0), "aaa");
        assert_eq!(string_array.value(1), "bbb");
    }

    #[test]
    fn test_fixed_size_binary_pages_split_by_byte_budget() {
        use crate::array_encoding::logical::primitive::PrimitiveFieldDecoder;
        use crate::array_encoding::logical::r#struct::SimpleStructDecoder;
        use crate::decoder::{DecoderReady, DrainLimit, LogicalPageDecoder};
        use arrow_schema::{Field as ArrowField, Fields};
        use lance_core::Result;
        use std::collections::VecDeque;

        #[derive(Debug)]
        struct NeverDecodedStub;

        impl PrimitivePageDecoder for NeverDecodedStub {
            fn decode(&self, _rows_to_skip: u64, _num_rows: u64) -> Result<DataBlock> {
                unreachable!("byte accounting must not decode any values")
            }
        }

        // Two pages of 3 rows x 10 bytes each; each page fits the budget alone
        // but the pair does not.
        let fields = Fields::from(vec![ArrowField::new("value", DataType::Utf8, false)]);
        let mut root = SimpleStructDecoder::new(fields, 6);
        for _ in 0..2 {
            root.accept_child(DecoderReady {
                decoder: Box::new(PrimitiveFieldDecoder::new_from_data(
                    Arc::new(FixedSizeBinaryDecoder {
                        bytes_decoder: Box::new(NeverDecodedStub),
                        byte_width: 10,
                        bytes_per_offset: 4,
                    }),
                    DataType::Utf8,
                    3,
                    false,
                )),
                path: VecDeque::from([0]),
            })
            .unwrap();
        }

        // Budget for page 1 plus two rows of page 2.
        let limit = root.max_rows_to_drain(6, 50).unwrap();
        assert_eq!(
            limit,
            DrainLimit {
                rows: 5,
                bytes: 50,
                dictionary_rows: None,
            }
        );

        // Budget for exactly one page: the batch stops at the page boundary.
        let limit = root.max_rows_to_drain(6, 30).unwrap();
        assert_eq!(
            limit,
            DrainLimit {
                rows: 3,
                bytes: 30,
                dictionary_rows: None,
            }
        );

        // Both pages fit a large enough budget.
        let limit = root.max_rows_to_drain(6, 60).unwrap();
        assert_eq!(
            limit,
            DrainLimit {
                rows: 6,
                bytes: 60,
                dictionary_rows: None,
            }
        );
    }
}
