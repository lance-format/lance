// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::{FutureExt, future::BoxFuture};
use lance_core::{
    Error, Result,
    utils::{memory::MemoryReserver, tokio::MemoryOwned},
};

use crate::resources::{flat_batch_allowance, flat_batch_retained_bytes, retain_batch_owner};

pub(super) struct StrictBatchResources {
    pub(super) pool: Arc<dyn MemoryReserver>,
    pub(super) directory: Option<Box<dyn Send + Sync>>,
    pub(super) concat: Option<BoxFuture<'static, Result<RecordBatch>>>,
    pub(super) _stream_control: Box<dyn Send + Sync>,
}

impl StrictBatchResources {
    pub(super) fn start_concat(
        &mut self,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        directory: Option<Box<dyn Send + Sync>>,
        num_rows: usize,
    ) -> Result<Option<RecordBatch>> {
        let backing = flat_batch_allowance(&schema, num_rows)?;
        // concat_batches collects batch references and a column's array
        // references. The primitive builders allocate values and validity once.
        let temporary = batches
            .len()
            .checked_mul(4 * size_of::<usize>())
            .and_then(|n| n.checked_add(256))
            .ok_or_else(|| Error::invalid_input("strict batch concat workspace overflow"))?;
        // Small copies stay on the caller; dispatch larger ones to the CPU pool.
        if backing < 128 * 1024 {
            let allowance = backing
                .checked_add(temporary)
                .ok_or_else(|| Error::invalid_input("strict batch concat allowance overflow"))?;
            let output = MemoryOwned::try_new(
                self.pool.as_ref(),
                "strict batch concat backing",
                allowance,
                || Ok(arrow::compute::concat_batches(&schema, &batches)?),
            )?;
            drop(batches);
            drop(directory);
            return finish_concat(output).map(Some);
        }
        let output = MemoryOwned::<Option<RecordBatch>>::try_new(
            self.pool.as_ref(),
            "strict batch concat backing",
            backing,
            || Ok(None),
        )?;
        let inputs = Arc::new((batches, directory));
        let cpu_inputs = inputs.clone();
        let task = output.into_cpu_task(
            self.pool.clone(),
            "strict batch concat",
            temporary,
            inputs,
            move |_| Ok(arrow::compute::concat_batches(&schema, &cpu_inputs.0)?),
        )?;
        self.concat = Some(async move { finish_concat(task.await?) }.boxed());
        Ok(None)
    }
}

fn finish_concat(mut output: MemoryOwned<RecordBatch>) -> Result<RecordBatch> {
    output.try_shrink_workspace(flat_batch_retained_bytes(output.value())?)?;
    let (batch, owner) = output.into_parts();
    retain_batch_owner(batch, Arc::new(owner))
}
