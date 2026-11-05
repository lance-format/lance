// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::make_array;
use arrow_data::ArrayData;
use arrow_schema::DataType;
use lance_core::{Error, Result};

use crate::{data::DataBlock, decoder::ListReadMode};

/// Bound independent header reads even when a scan selects many list rows/chunks.
pub(super) const METADATA_READ_CONCURRENCY: usize = 32;

/// The planner guarantees that these leaf values cannot be observed. Keep the
/// Arrow type valid while parent decoders reconstruct offsets and validity from
/// structural data. In particular, fixed-size lists and non-nullable map keys
/// must not acquire null children when their parent validity is restored.
#[derive(Debug, Clone)]
pub(super) struct StructuralProjection {
    pub mode: ListReadMode,
    pub data_type: DataType,
}

impl StructuralProjection {
    pub fn placeholder_values(&self, len: u64) -> Result<DataBlock> {
        if matches!(&self.data_type, DataType::Struct(fields) if fields.is_empty()) {
            return Ok(DataBlock::AllNull(crate::data::AllNullDataBlock {
                num_values: len,
            }));
        }
        fn valid_placeholder(data: ArrayData) -> Result<ArrayData> {
            if data.data_type() == &DataType::Null {
                return Ok(data);
            }
            let children = if let DataType::Dictionary(_, value) = data.data_type() {
                vec![valid_placeholder(ArrayData::new_null(
                    value,
                    usize::from(!data.is_empty()),
                ))?]
            } else {
                data.child_data()
                    .iter()
                    .cloned()
                    .map(valid_placeholder)
                    .collect::<Result<Vec<_>>>()?
            };
            Ok(data
                .into_builder()
                .nulls(None)
                .child_data(children)
                .build()?)
        }
        let len = usize::try_from(len).map_err(|_| {
            Error::invalid_input("Structural leaf length exceeds addressable memory")
        })?;
        let array = make_array(valid_placeholder(ArrayData::new_null(
            &self.data_type,
            len,
        ))?);
        Ok(DataBlock::from_decoded_array(array))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use crate::EncodingsIo;
    use bytes::Bytes;
    use futures::{FutureExt, future::BoxFuture};
    use lance_core::Result;
    use std::{ops::Range, sync::Mutex};

    #[derive(Debug)]
    pub struct GuardIo {
        pub bytes: Bytes,
        pub payloads: Vec<Range<u64>>,
        pub reads: Mutex<Vec<Range<u64>>>,
    }

    impl EncodingsIo for GuardIo {
        fn submit_request(
            &self,
            ranges: Vec<Range<u64>>,
            _priority: u64,
        ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
            for range in &ranges {
                assert!(
                    !self
                        .payloads
                        .iter()
                        .any(|p| p.start < range.end && range.start < p.end),
                    "read payload: {range:?}"
                );
            }
            self.reads.lock().unwrap().extend(ranges.iter().cloned());
            futures::future::ready(Ok(ranges
                .into_iter()
                .map(|r| self.bytes.slice(r.start as usize..r.end as usize))
                .collect()))
            .boxed()
        }
    }
}
