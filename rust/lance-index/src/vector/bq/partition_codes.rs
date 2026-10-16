// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Code-only cache entries of native IVF_RQ partitions.
//!
//! A native IVF_RQ index with a flat sub-index whose small columns are
//! resident caches a partition as its code columns alone
//! (`EntryColumns::Codes`): every read attaches views of the resident rows
//! and builds the storage for that read alone, so the cache holds no second
//! copy of the row ids and
//! factors the store already keeps. The codes are kept packed and
//! blocked and serialized as one Arrow IPC batch.

use std::borrow::Cow;

use arrow_array::RecordBatch;
use lance_core::Result;
use lance_core::cache::{
    CacheCodec, CacheCodecImpl, CacheEntryReader, CacheEntryWriter, CacheKey, CacheKeySchema,
    KeyBuilder,
};
use lance_core::deepsize::{Context, DeepSizeOf};

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
    const CURRENT_VERSION: u32 = 1;

    fn serialize(&self, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        writer.write_ipc(&self.0)
    }

    fn deserialize(reader: &mut CacheEntryReader<'_>) -> Result<Self> {
        Ok(Self(reader.read_ipc()?))
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
