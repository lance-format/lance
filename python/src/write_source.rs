// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::FromPyArrow;
use arrow_array::RecordBatchReader;
use arrow_schema::SchemaRef;
use datafusion::{
    catalog::{TableProvider, streaming::StreamingTable},
    error::{DataFusionError, Result},
    execution::{SendableRecordBatchStream, TaskContext},
    physical_plan::{stream::RecordBatchStreamAdapter, streaming::PartitionStream},
};
use futures::TryStreamExt;
use lance_datafusion::utils::reader_to_stream;
use pyo3::prelude::*;

pub(crate) fn reader_factory_provider(
    schema: SchemaRef,
    reader_factory: Py<PyAny>,
) -> Result<Arc<dyn TableProvider>> {
    let partition = Arc::new(ReaderFactoryPartition {
        schema: schema.clone(),
        reader_factory: Arc::new(reader_factory),
    });
    Ok(Arc::new(StreamingTable::try_new(schema, vec![partition])?))
}

struct ReaderFactoryPartition {
    schema: SchemaRef,
    reader_factory: Arc<Py<PyAny>>,
}

impl std::fmt::Debug for ReaderFactoryPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReaderFactoryPartition")
            .field("schema", &self.schema)
            .finish()
    }
}

impl PartitionStream for ReaderFactoryPartition {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
        let reader_factory = self.reader_factory.clone();
        let schema = self.schema.clone();
        let stream = futures::stream::once(async move {
            // A factory can call back into Lance and synchronously wait for its
            // async scanner. Keep that wait off the runtime's worker threads.
            let reader = tokio::task::spawn_blocking(move || {
                Python::attach(|py| {
                    let reader = reader_factory.bind(py).call0()?;
                    ArrowArrayStreamReader::from_pyarrow_bound(&reader)
                })
            })
            .await
            .map_err(|error| DataFusionError::External(Box::new(error)))?
            .map_err(|error| DataFusionError::External(Box::new(error)))?;

            let actual_schema = reader.schema();
            if actual_schema != schema {
                return Err(DataFusionError::Execution(format!(
                    "Re-scannable source reader schema does not match the registered schema. \
                     Expected: {schema:?}. Actual: {actual_schema:?}"
                )));
            }

            Ok(reader_to_stream(Box::new(reader)))
        })
        .try_flatten();

        Box::pin(RecordBatchStreamAdapter::new(self.schema.clone(), stream))
    }
}
