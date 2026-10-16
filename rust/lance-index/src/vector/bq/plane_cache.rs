// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Persistent whole-plane cache entries for native RaBitQ projections.

use lance_core::Result;
use lance_core::cache::{CacheCodecImpl, CacheEntryReader, CacheEntryWriter};

use super::planes::PlaneBatch;

impl CacheCodecImpl for PlaneBatch {
    const TYPE_ID: &'static str = "lance.vector.rabit.plane";
    const CURRENT_VERSION: u32 = 1;

    fn serialize(&self, writer: &mut CacheEntryWriter<'_>) -> Result<()> {
        writer.write_ipc(&self.0)
    }

    fn deserialize(reader: &mut CacheEntryReader<'_>) -> Result<Self> {
        Ok(Self(reader.read_ipc()?))
    }
}
