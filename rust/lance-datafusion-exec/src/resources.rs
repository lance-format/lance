// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Experimental bridge to one existing DataFusion pool, without another budget.

use arrow::array::{ArrayData, make_array};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_schema::{DataType, Schema};
use std::{panic::AssertUnwindSafe, sync::Arc};

use datafusion_execution::{
    TaskContext,
    memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation},
};
use lance_core::{
    Error, Result,
    utils::{
        memory::{MemoryLease, MemoryReserver},
        tasks::ResourceTaskGroup,
    },
};

/// Experimental adapter. Every returned owner uses the supplied pool directly.
#[derive(Debug)]
#[doc(hidden)]
pub struct DataFusionMemoryReserver {
    pool: Arc<dyn MemoryPool>,
    tasks: Arc<ResourceTaskGroup>,
    _control: MemoryReservation,
}

impl DataFusionMemoryReserver {
    /// Attach an operation resource owner to the supplied context. The
    /// adapter only draws from this context's memory pool.
    pub fn configure(context: TaskContext) -> Result<TaskContext> {
        let pool = context.memory_pool().clone();
        let control = Self::reservation(&pool, "scan resource adapter", 1024)?;
        let resources = Arc::new(Self {
            pool,
            tasks: Arc::default(),
            _control: control,
        });
        let config = context.session_config().clone().with_extension(resources);
        Ok(context.with_session_config(config))
    }

    /// Resolve the single operation adapter and verify that it corresponds to
    /// the context's current pool, including any execution tracking wrapper.
    pub fn from_context(context: &TaskContext) -> Result<Option<Arc<Self>>> {
        let resources = context.session_config().get_extension::<Self>();
        if resources
            .as_ref()
            .is_some_and(|resources| !Arc::ptr_eq(&resources.pool, context.memory_pool()))
        {
            return Err(Error::invalid_input(
                "scan resource adapter must share the TaskContext memory pool",
            ));
        }
        Ok(resources)
    }

    fn reservation(
        pool: &Arc<dyn MemoryPool>,
        stage: &str,
        bytes: usize,
    ) -> Result<MemoryReservation> {
        let reservation = MemoryConsumer::new(stage).register(pool);
        reservation
            .try_grow(bytes)
            .map_err(|source| Error::External {
                source: Box::new(source),
            })?;
        Ok(reservation)
    }
}

struct DataFusionLease(MemoryReservation);
impl MemoryLease for DataFusionLease {
    fn try_shrink(&mut self, bytes: usize) -> Result<bool> {
        let reserved = self.0.size();
        if bytes > reserved {
            return Err(Error::invalid_input(format!(
                "memory lease cannot grow: requested_bytes={bytes}, reserved_bytes={reserved}"
            )));
        }
        self.0.shrink(reserved - bytes);
        Ok(true)
    }
}
impl MemoryReserver for DataFusionMemoryReserver {
    fn task_group(&self) -> Option<Arc<ResourceTaskGroup>> {
        Some(self.tasks.clone())
    }
    fn reserve_lease(&self, stage: &str, bytes: usize) -> Result<Box<dyn MemoryLease>> {
        Ok(Box::new(DataFusionLease(Self::reservation(
            &self.pool, stage, bytes,
        )?)))
    }
    fn reserve(&self, stage: &str, bytes: usize) -> Result<Box<dyn Send + Sync>> {
        Ok(Box::new(Self::reservation(&self.pool, stage, bytes)?))
    }
}

/// Maximum backing and owner storage for the supported flat layouts.
pub fn flat_batch_allowance(schema: &Schema, rows: usize) -> Result<usize> {
    if schema.fields().is_empty() {
        return Err(Error::not_supported(
            "accounted scan batches require at least one column",
        ));
    }
    schema.fields().iter().try_fold(0usize, |total, field| {
        let width = match field.data_type() {
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            data_type => {
                return Err(Error::not_supported(format!(
                    "accounted scan batches require Int32 or Int64, got {data_type} for {}",
                    field.name()
                )));
            }
        };
        // Arrow's primitive builder sizes values and validity for the entire
        // output up front. Include two alignment paddings and the owner/array
        // descriptors (including the temporary ArrayData during attachment).
        rows.checked_mul(width)
            .and_then(|n| n.checked_add(rows.div_ceil(8)))
            .and_then(|n| n.checked_add(128 + 1024))
            .and_then(|n| n.checked_add(total))
            .ok_or_else(|| Error::invalid_input("accounted batch allowance overflow"))
    })
}

/// Arrow backing excludes sidecar storage, which is admitted separately for
/// each column before it is associated with the allocation.
pub fn flat_batch_retained_bytes(batch: &arrow_array::RecordBatch) -> Result<usize> {
    batch.columns().iter().try_fold(0usize, |total, column| {
        total
            .checked_add(column.get_buffer_memory_size())
            .and_then(|n| n.checked_add(1024))
            .ok_or_else(|| Error::invalid_input("retained batch allowance overflow"))
    })
}

/// Retain one existing backing owner through every Arrow buffer and slice.
/// The caller admits sidecar storage before calling this function. Int32 and
/// Int64 arrays, including validity buffers, preserve their original backing.
#[doc(hidden)]
pub fn retain_batch_owner(
    batch: arrow_array::RecordBatch,
    owner: Arc<dyn std::any::Any + Send + Sync>,
) -> Result<arrow_array::RecordBatch> {
    struct OwnedBuffer {
        _buffer: Buffer,
        _owner: AssertUnwindSafe<Arc<dyn std::any::Any + Send + Sync>>,
    }
    fn wrap(buffer: &Buffer, owner: &Arc<dyn std::any::Any + Send + Sync>) -> Result<Buffer> {
        let ptr = std::ptr::NonNull::new(buffer.as_ptr().cast_mut())
            .ok_or_else(|| Error::internal("Arrow backing pointer is null"))?;
        // SAFETY: the owner retains the immutable original buffer, including its
        // allocation and offset, for the entire lifetime of this view.
        Ok(unsafe {
            Buffer::from_custom_allocation(
                ptr,
                buffer.len(),
                Arc::new(OwnedBuffer {
                    _buffer: buffer.clone(),
                    _owner: AssertUnwindSafe(owner.clone()),
                }),
            )
        })
    }
    fn data(value: ArrayData, owner: &Arc<dyn std::any::Any + Send + Sync>) -> Result<ArrayData> {
        let buffers = value
            .buffers()
            .iter()
            .map(|b| wrap(b, owner))
            .collect::<Result<Vec<_>>>()?;
        let nulls = value
            .nulls()
            .map(|nulls| {
                wrap(nulls.inner().inner(), owner).map(|buffer| {
                    let bitmap = BooleanBuffer::new(buffer, nulls.offset(), nulls.len());
                    // SAFETY: this is the original bitmap and original slice.
                    unsafe { NullBuffer::new_unchecked(bitmap, nulls.null_count()) }
                })
            })
            .transpose()?;
        Ok(value.into_builder().buffers(buffers).nulls(nulls).build()?)
    }
    flat_batch_allowance(batch.schema_ref(), 0)?;
    let columns = batch
        .columns()
        .iter()
        .map(|array| data(array.to_data(), &owner).map(make_array))
        .collect::<Result<Vec<_>>>()?;
    Ok(arrow_array::RecordBatch::try_new(batch.schema(), columns)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    #[test]
    fn test_reserver_shares_the_task_context_pool() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1104));
        let context = TaskContext::default().with_runtime(
            datafusion_execution::runtime_env::RuntimeEnvBuilder::new()
                .with_memory_pool(pool.clone())
                .build_arc()
                .unwrap(),
        );
        let context = DataFusionMemoryReserver::configure(context).unwrap();
        let adapter = DataFusionMemoryReserver::from_context(&context)
            .unwrap()
            .unwrap();
        let baseline = pool.reserved();
        let external = MemoryConsumer::new("DataFusion operator").register(&pool);
        external.try_grow(32).unwrap();
        let error = adapter.reserve("scratch serialization", 64).err().unwrap();
        assert!(matches!(error, Error::External { .. }));
        assert!(error.to_string().contains("scratch serialization"));
        assert_eq!(pool.reserved(), baseline + 32);
        drop(external);
        let mut lease = adapter.reserve_lease("scratch serialization", 64).unwrap();
        assert_eq!(pool.reserved(), baseline + 64);
        assert!(lease.try_shrink(8).unwrap());
        assert_eq!(pool.reserved(), baseline + 8);
        drop(lease);
        assert_eq!(pool.reserved(), baseline);
        let other_runtime = datafusion_execution::runtime_env::RuntimeEnvBuilder::new()
            .build_arc()
            .unwrap();
        let context = context.with_runtime(other_runtime);
        let error = DataFusionMemoryReserver::from_context(&context).unwrap_err();
        assert!(matches!(error, Error::InvalidInput { .. }));
        assert!(error.to_string().contains("TaskContext memory pool"));
        drop((context, adapter));
        assert_eq!(pool.reserved(), 0);
    }
}
