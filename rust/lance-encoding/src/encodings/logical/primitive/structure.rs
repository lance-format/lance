// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::make_array;
use arrow_data::ArrayData;
use arrow_schema::DataType;
use bytes::Bytes;
use futures::{FutureExt, StreamExt, TryStreamExt, future::BoxFuture, stream};
use lance_core::cache::{CacheKey, CacheKeySchema, Context, DeepSizeOf, KeyBuilder, LanceCache};
use lance_core::{Error, Result};
use std::{borrow::Cow, ops::Range, sync::Arc};

use crate::EncodingsIo;
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

struct MetadataBytes(Bytes);

impl DeepSizeOf for MetadataBytes {
    fn deep_size_of_children(&self, _: &mut Context) -> usize {
        self.0.len()
    }
}

struct MetadataRange(Range<u64>);

impl CacheKey for MetadataRange {
    type ValueType = MetadataBytes;
    fn key(&self) -> Cow<'_, str> {
        format!("list-structure/{}/{}", self.0.start, self.0.end).into()
    }
    fn type_name() -> &'static str {
        "ListStructureMetadata"
    }
    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.encoding.list-structure-metadata", 1)
    }
    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.0.start);
        builder.write_u64(self.0.end);
    }
}

/// Only wrap requests already proven to contain list metadata. The supplied
/// cache is file-scoped, and its usual capacity/admission policy bounds these
/// entries. Ordinary value reads never consult this cache namespace.
#[derive(Debug)]
pub(super) struct StructuralMetadataIo {
    inner: Arc<dyn EncodingsIo>,
    cache: Arc<LanceCache>,
}

impl StructuralMetadataIo {
    pub fn wrap(
        inner: Arc<dyn EncodingsIo>,
        cache: Option<Arc<LanceCache>>,
    ) -> Arc<dyn EncodingsIo> {
        match cache.filter(|cache| cache.capacity_bytes() != Some(0)) {
            Some(cache) => Arc::new(Self { inner, cache }),
            None => inner,
        }
    }
}

impl EncodingsIo for StructuralMetadataIo {
    fn submit_request(
        &self,
        ranges: Vec<Range<u64>>,
        priority: u64,
    ) -> BoxFuture<'static, Result<Vec<Bytes>>> {
        let io = self.inner.clone();
        let cache = self.cache.clone();
        async move {
            stream::iter(ranges)
                .map(|range| {
                    let io = io.clone();
                    let cache = cache.clone();
                    async move {
                        let value = cache
                            .get_or_insert_with_key(MetadataRange(range.clone()), || async move {
                                let bytes = io.submit_single(range, priority).await?;
                                // A slice may retain a much larger allocation. Own only
                                // the requested bytes so cache accounting stays accurate.
                                Ok(MetadataBytes(Bytes::copy_from_slice(&bytes)))
                            })
                            .await?;
                        Ok(value.0.clone())
                    }
                })
                .buffered(METADATA_READ_CONCURRENCY)
                .try_collect()
                .await
        }
        .boxed()
    }

    fn with_coalesce_gap(&self, gap: u64) -> Option<Arc<dyn EncodingsIo>> {
        Some(Arc::new(Self {
            inner: self.inner.with_coalesce_gap(gap)?,
            cache: self.cache.clone(),
        }))
    }

    fn with_bypass_backpressure(&self) -> Option<Arc<dyn EncodingsIo>> {
        Some(Arc::new(Self {
            inner: self.inner.with_bypass_backpressure()?,
            cache: self.cache.clone(),
        }))
    }

    fn with_io_stats(
        &self,
        stats: Arc<dyn lance_core::utils::io_stats::IoStatsRecorder>,
    ) -> Option<Arc<dyn EncodingsIo>> {
        Some(Arc::new(Self {
            inner: self.inner.with_io_stats(stats)?,
            cache: self.cache.clone(),
        }))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::StructuralMetadataIo;
    use crate::EncodingsIo;
    use bytes::Bytes;
    use futures::{FutureExt, future::BoxFuture};
    use lance_core::Result;
    use lance_core::cache::LanceCache;
    use std::{
        ops::Range,
        sync::{Arc, Mutex},
    };

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

    #[tokio::test]
    async fn metadata_cache_preserves_range_order_and_file_scope() {
        let io = Arc::new(GuardIo {
            bytes: Bytes::from((0..48).collect::<Vec<u8>>()),
            payloads: std::iter::once(8..40).collect(),
            reads: Mutex::default(),
        });
        let cache = LanceCache::with_capacity(1024 * 1024);
        let first =
            StructuralMetadataIo::wrap(io.clone(), Some(Arc::new(cache.with_key_prefix("file-a"))));
        let ranges = vec![0..8, 40..48];
        let before = first.submit_request(ranges.clone(), 0).await.unwrap();
        let after = first
            .submit_request(ranges.into_iter().rev().collect(), 0)
            .await
            .unwrap();
        assert_eq!(before[0], after[1]);
        assert_eq!(before[1], after[0]);
        assert_eq!(io.reads.lock().unwrap().len(), 2);
        let second =
            StructuralMetadataIo::wrap(io.clone(), Some(Arc::new(cache.with_key_prefix("file-b"))));
        second.submit_single(0..8, 0).await.unwrap();
        assert_eq!(io.reads.lock().unwrap().len(), 3);
    }
}
