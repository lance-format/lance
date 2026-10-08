// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Code-only cache entries of native IVF_RQ partitions.
//!
//! A native IVF_RQ index with a flat sub-index whose small columns are
//! resident caches a partition as its code columns alone
//! (`EntryColumns::Codes`): every read attaches views of the resident rows
//! (copies under `LANCE_RQ_RESIDENT_ATTACH=copy`) and builds the storage for
//! that read alone, so the cache holds no second copy of the row ids and
//! factors the store already keeps. The codes are kept packed and
//! blocked (see `normalize_entry_codes`), and persisted as a raw fixed-width
//! body (see [`super::raw_body`]):
//!
//! ```text
//! v1: [raw body]
//! ```

use std::borrow::Cow;

use arrow_array::RecordBatch;
use lance_core::cache::{
    CacheCodec, CacheCodecImpl, CacheEntryReader, CacheEntryWriter, CacheKey, CacheKeySchema,
    KeyBuilder,
};
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::{Error, Result};

use super::raw_body::{read_raw_batch, write_raw_batch};

/// Body version of the entries this build writes: a raw body.
const RAW_BODY_VERSION: u32 = 1;

/// The code columns of a native IVF_RQ partition: the sign codes, packed,
/// and the ex codes, blocked, at the partition's rows.
#[derive(Debug)]
pub struct PartitionCodes(pub RecordBatch);

impl DeepSizeOf for PartitionCodes {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.0.deep_size_of_children(context)
    }
}

impl CacheCodecImpl for PartitionCodes {
    const TYPE_ID: &'static str = "lance.vector.rq.partition-codes";
    const CURRENT_VERSION: u32 = RAW_BODY_VERSION;

    fn serialize(&self, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        write_raw_batch(writer, &self.0)
    }

    fn deserialize(reader: &mut CacheEntryReader<'_>) -> Result<Self> {
        match reader.version() {
            RAW_BODY_VERSION => Ok(Self(read_raw_batch(reader)?)),
            version => Err(Error::invalid_input(format!(
                "unsupported partition codes cache version {version}"
            ))),
        }
    }
}

/// Key of a native IVF_RQ partition's code-only entry, under the index's
/// namespace of the index cache. Its entries are a type of their own, so an
/// index that caches whole partitions never finds one, nor the reverse.
pub struct PartitionCodesKey {
    pub partition: usize,
}

impl CacheKey for PartitionCodesKey {
    type ValueType = PartitionCodes;

    fn key(&self) -> Cow<'_, str> {
        format!("ivf-codes-{}", self.partition).into()
    }

    fn type_name() -> &'static str {
        "IVFPartitionCodes"
    }

    fn schema() -> CacheKeySchema {
        CacheKeySchema::new("lance.index.ivf-partition-codes-key", 1)
    }

    fn write_key(&self, builder: &mut KeyBuilder) {
        builder.write_u64(self.partition as u64);
    }

    /// No plane tag and the lowest memory priority, as a whole native
    /// partition's entry.
    fn codec() -> Option<CacheCodec> {
        Some(CacheCodec::from_impl::<PartitionCodes>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, UInt8Array};
    use bytes::Bytes;
    use lance_arrow::FixedSizeListArrayExt;
    use lance_core::cache::{CacheDecode, CacheMissReason};

    use crate::vector::bq::raw_body::RAW_BODY_KIND;
    use crate::vector::bq::storage::{RABIT_BLOCKED_EX_CODE_COLUMN, RABIT_CODE_COLUMN};

    /// Envelope bytes ahead of an entry's body.
    fn body_offset() -> usize {
        4 + 1 + 2 + PartitionCodes::TYPE_ID.len() + 4
    }

    fn codes(rows: usize, width: usize) -> ArrayRef {
        Arc::new(
            FixedSizeListArray::try_new_from_values(
                UInt8Array::from(
                    (0..rows * width)
                        .map(|v| (v % 251) as u8)
                        .collect::<Vec<_>>(),
                ),
                width as i32,
            )
            .unwrap(),
        )
    }

    fn partition_codes(rows: usize) -> RecordBatch {
        RecordBatch::try_from_iter([
            (RABIT_CODE_COLUMN, codes(rows, 16)),
            (RABIT_BLOCKED_EX_CODE_COLUMN, codes(rows, 48)),
        ])
        .unwrap()
    }

    fn encode(batch: &RecordBatch) -> Bytes {
        let codec = PartitionCodesKey { partition: 0 }.codec_for_key().unwrap();
        let mut encoded = Vec::new();
        codec
            .serialize(
                &(Arc::new(PartitionCodes(batch.clone())) as Arc<dyn std::any::Any + Send + Sync>),
                &mut encoded,
            )
            .unwrap();
        Bytes::from(encoded)
    }

    fn decode(encoded: &Bytes) -> CacheDecode<RecordBatch> {
        let codec = PartitionCodesKey { partition: 0 }.codec_for_key().unwrap();
        match codec.deserialize(encoded) {
            CacheDecode::Hit(value) => {
                CacheDecode::Hit(value.downcast::<PartitionCodes>().unwrap().0.clone())
            }
            CacheDecode::Miss(reason) => CacheDecode::Miss(reason),
        }
    }

    /// Codes round trip through a raw body of exactly their values, beyond
    /// the schema and padding: no validity bitmap. Decoded entries are
    /// charged their values.
    #[test]
    fn partition_codes_round_trip_through_raw_bodies() {
        for rows in [0, 1, 33, 300] {
            let original = partition_codes(rows);
            let encoded = encode(&original);
            let body = &encoded[body_offset()..];
            assert_eq!(body[0], RAW_BODY_KIND, "rows={rows}");
            let schema_len = u32::from_le_bytes(body[1..5].try_into().unwrap()) as usize;
            let values = [16 * rows, 48 * rows]
                .map(|bytes| bytes.next_multiple_of(8))
                .iter()
                .sum::<usize>();
            assert_eq!(body.len(), 1 + 4 + schema_len + 8 + values, "rows={rows}");
            let CacheDecode::Hit(decoded) = decode(&encoded) else {
                panic!("rows={rows}: decode missed")
            };
            assert_eq!(decoded, original, "rows={rows}");
            assert_eq!(
                PartitionCodes(decoded).deep_size_of(),
                PartitionCodes(original).deep_size_of(),
                "rows={rows}"
            );
        }
    }

    /// A body this build cannot read is a miss, never a panic.
    #[test]
    fn newer_or_corrupt_partition_codes_miss() {
        let encoded = encode(&partition_codes(40)).to_vec();
        let version_at = body_offset() - 4;
        let mut newer = encoded.clone();
        newer[version_at..body_offset()]
            .copy_from_slice(&(PartitionCodes::CURRENT_VERSION + 1).to_le_bytes());
        assert!(matches!(
            decode(&Bytes::from(newer)),
            CacheDecode::Miss(CacheMissReason::VersionTooNew)
        ));
        for len in [body_offset(), body_offset() + 1, encoded.len() - 1] {
            assert!(
                matches!(
                    decode(&Bytes::from(encoded[..len].to_vec())),
                    CacheDecode::Miss(_)
                ),
                "truncated to {len} bytes"
            );
        }
    }

    /// Entries of other partitions, and whole-partition entries, have other
    /// keys; the codec tags no plane and keeps the lowest priority.
    #[test]
    fn partition_codes_key_and_codec() {
        let key = |partition| PartitionCodesKey { partition };
        assert_ne!(key(1).key(), key(2).key());
        assert_eq!(PartitionCodesKey::type_name(), "IVFPartitionCodes");
        let codec = key(1).codec_for_key().unwrap();
        assert_eq!(codec.type_id(), PartitionCodes::TYPE_ID);
        assert_eq!(codec.plane_tag(), None);
        assert_eq!(codec.memory_priority(), 0);
        assert!(!codec.supports_row_selection());
        let factor: ArrayRef = Arc::new(Float32Array::from(vec![0.5f32; 3]));
        let other = RecordBatch::try_from_iter([("factor", factor)]).unwrap();
        let CacheDecode::Hit(decoded) = decode(&encode(&other)) else {
            panic!("decode missed")
        };
        assert_eq!(decoded, other);
    }
}
