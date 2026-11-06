// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Blob v2 values read as logical bytes instead of the descriptors the
//! sources hold, for
//! [`LsmPointLookupPlanner::with_blobs_as_logical`](super::LsmPointLookupPlanner::with_blobs_as_logical).

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use arrow_array::{Array, ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::TryStreamExt;
use lance_arrow::FieldExt;
use lance_core::datatypes::{BLOB_V2_LOGICAL_FIELDS, BlobV2Layout};
use lance_core::utils::blob::blob_path;
use lance_core::{Error, ROW_ID, Result};
use lance_datafusion::exec::OneShotExec;
use lance_io::object_store::{ObjectStore, ObjectStoreParams};
use object_store::path::Path;

use super::collector::LsmDataSourceCollector;
use super::data_source::LsmDataSource;
use crate::Dataset;
use crate::blob::blob_v2_layout;
use crate::dataset::DATA_DIR;
use crate::dataset::blob::{
    BlobV2BatchRewritePlan, SidecarRead, prepared_blob_batch_to_logical, prepared_sidecar_reads,
};
use crate::dataset::mem_wal::memtable::batch_store::{BatchStore, StoredBatch};
use crate::dataset::mem_wal::reconcile::relabel_to;
use crate::session::Session;

/// Whether `schema` holds a blob v2 field at any depth.
pub(super) fn has_blobs(schema: &Schema) -> bool {
    has_blob_matching(schema, &|field| field.is_blob_v2())
}

/// Whether `schema` holds a blob v2 descriptor at any depth.
pub(super) fn has_blob_descriptors(schema: &Schema) -> bool {
    has_blob_matching(schema, &|field| {
        blob_v2_layout(field) == Some(BlobV2Layout::Descriptor)
    })
}

fn has_blob_matching(schema: &Schema, blob: &dyn Fn(&Field) -> bool) -> bool {
    fn holds(field: &Field, blob: &dyn Fn(&Field) -> bool) -> bool {
        blob(field)
            || match field.data_type() {
                DataType::Struct(children) => children.iter().any(|c| holds(c, blob)),
                DataType::List(item) | DataType::LargeList(item) => holds(item, blob),
                _ => false,
            }
    }
    schema.fields().iter().any(|f| holds(f, blob))
}

/// `plan`, a scan of `dataset` carrying `_rowaddr`, with each blob v2 value
/// read as logical bytes.
pub(super) fn dataset_blobs_as_logical(
    plan: Arc<dyn ExecutionPlan>,
    dataset: Arc<Dataset>,
    keep_row_addr: bool,
    ctx: Arc<TaskContext>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let rewrite = Arc::new(BlobV2BatchRewritePlan::try_new(
        dataset.schema(),
        plan.schema().as_ref(),
        keep_row_addr,
        false,
    )?);
    let schema = rewrite.output_schema().clone();
    let batches = plan.execute(0, ctx)?.and_then(move |batch| {
        let (rewrite, dataset) = (rewrite.clone(), dataset.clone());
        async move {
            rewrite
                .transform_batch(&dataset, batch)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))
        }
    });
    Ok(Arc::new(OneShotExec::new(Box::pin(
        RecordBatchStreamAdapter::new(schema, batches),
    ))))
}

/// Where a memtable's packed and dedicated payloads are: the sidecars of the
/// data file its generation will flush to.
pub(super) struct MemTableSidecars {
    store: Arc<ObjectStore>,
    data_dir: Path,
    data_file_key: String,
}

impl MemTableSidecars {
    /// The sidecars of the memtable in `batch_store`, one of `source`'s.
    pub(super) async fn open(
        collector: &LsmDataSourceCollector,
        source: &LsmDataSource,
        batch_store: &BatchStore,
        session: Option<&Arc<Session>>,
        store_params: Option<&ObjectStoreParams>,
    ) -> Result<Self> {
        let (Some(shard_id), Some(target)) = (source.shard_id(), batch_store.target()) else {
            return Err(Error::internal(format!(
                "point lookup: memtable generation {} holds blob descriptors but has no data target",
                source.generation()
            )));
        };
        let uri = collector.generation_uri(&shard_id, &target.generation_dir);
        let registry = session
            .map(|session| session.store_registry())
            .unwrap_or_default();
        let params = store_params.cloned().unwrap_or_default();
        let (store, generation) = ObjectStore::from_uri_and_params(registry, &uri, &params).await?;
        Ok(Self {
            store,
            data_dir: generation.join(DATA_DIR),
            data_file_key: target.data_file_key().to_string(),
        })
    }
}

/// `plan`, a scan of the memtable in `batch_store` carrying `_rowid`, with
/// each blob v2 value read as logical bytes: inline from the value the
/// memtable holds, otherwise from its sidecar.
pub(super) fn memtable_blobs_as_logical(
    plan: Arc<dyn ExecutionPlan>,
    batch_store: Arc<BatchStore>,
    sidecars: Arc<MemTableSidecars>,
    ctx: Arc<TaskContext>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let schema: SchemaRef = Arc::new(Schema::new_with_metadata(
        plan.schema()
            .fields()
            .iter()
            .map(|f| Arc::new(logical_blob_field(f)))
            .collect::<Vec<_>>(),
        plan.schema().metadata().clone(),
    ));
    let output = schema.clone();
    let batches = plan.execute(0, ctx)?.and_then(move |batch| {
        let (batch_store, sidecars, schema) =
            (batch_store.clone(), sidecars.clone(), output.clone());
        async move {
            memtable_batch_as_logical(&batch, &batch_store, &sidecars, &schema)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))
        }
    });
    Ok(Arc::new(OneShotExec::new(Box::pin(
        RecordBatchStreamAdapter::new(schema, batches),
    ))))
}

/// `field` with each blob v2 descriptor at any depth in the complete logical
/// layout.
fn logical_blob_field(field: &FieldRef) -> Field {
    if blob_v2_layout(field) == Some(BlobV2Layout::Descriptor) {
        return field
            .as_ref()
            .clone()
            .with_data_type(DataType::Struct(BLOB_V2_LOGICAL_FIELDS.clone()));
    }
    let data_type = match field.data_type() {
        DataType::Struct(children) => DataType::Struct(
            children
                .iter()
                .map(|c| Arc::new(logical_blob_field(c)))
                .collect(),
        ),
        DataType::List(item) => DataType::List(Arc::new(logical_blob_field(item))),
        DataType::LargeList(item) => DataType::LargeList(Arc::new(logical_blob_field(item))),
        other => other.clone(),
    };
    field.as_ref().clone().with_data_type(data_type)
}

/// `batch`'s blob columns rebuilt from the prepared values the memtable holds
/// at each row's `_rowid`.
async fn memtable_batch_as_logical(
    batch: &RecordBatch,
    batch_store: &BatchStore,
    sidecars: &MemTableSidecars,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    if batch.num_rows() == 0 {
        return Ok(RecordBatch::new_empty(schema.clone()));
    }
    let row_ids = batch
        .column_by_name(ROW_ID)
        .ok_or_else(|| Error::internal("memtable blob read needs `_rowid`"))?
        .as_primitive::<UInt64Type>();
    // The stored batches the rows sit in, and each row's place among them.
    let mut held: Vec<&StoredBatch> = Vec::new();
    let mut held_at: HashMap<usize, usize> = HashMap::new();
    let mut rows = Vec::with_capacity(row_ids.len());
    for &row_id in row_ids.values() {
        let (stored, row) = batch_store
            .locate_row(row_id)
            .ok_or_else(|| Error::internal(format!("memtable holds no row {row_id}")))?;
        let at = *held_at.entry(stored.batch_position).or_insert_with(|| {
            held.push(stored);
            held.len() - 1
        });
        rows.push((at, row));
    }

    let blob_columns = (0..batch.num_columns())
        .filter(|&i| schema.field(i).data_type() != batch.schema().field(i).data_type())
        .collect::<Vec<_>>();
    let mut prepared_fields = Vec::with_capacity(blob_columns.len());
    let mut prepared_columns = Vec::with_capacity(blob_columns.len());
    for &i in &blob_columns {
        let name = batch.schema().field(i).name().clone();
        let values = held
            .iter()
            .map(|s| {
                s.data
                    .column_by_name(&name)
                    .map(|c| c.as_ref())
                    .ok_or_else(|| Error::internal(format!("memtable batch lacks column `{name}`")))
            })
            .collect::<Result<Vec<&dyn Array>>>()?;
        let field = held[0].data.schema().field_with_name(&name)?.clone();
        prepared_columns.push(arrow_select::interleave::interleave(&values, &rows)?);
        prepared_fields.push(field);
    }
    let prepared = RecordBatch::try_new(Arc::new(Schema::new(prepared_fields)), prepared_columns)?;

    let reads = prepared_sidecar_reads(&prepared)?;
    if !reads.is_empty() {
        // A durable put leaves its payload in the generation's open pack
        // until the WAL flush finishes it; finish it now so it can be read.
        batch_store.finish_blob_pack().await?;
    }
    let payloads = futures::future::try_join_all(reads.into_iter().map(|read| async move {
        let path = blob_path(&sidecars.data_dir, &sidecars.data_file_key, read.blob_id);
        let start = read.position as usize;
        let bytes = sidecars
            .store
            .open(&path)
            .await?
            .get_range(start..start + read.size as usize)
            .await?;
        Ok::<_, Error>((read, bytes))
    }))
    .await?
    .into_iter()
    .collect::<HashMap<SidecarRead, _>>();
    let logical = prepared_blob_batch_to_logical(&prepared, &payloads)?;

    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    for (converted, &i) in logical.columns().iter().zip(&blob_columns) {
        // The memtable labels nested fields with its own field ids.
        columns[i] = relabel_to(converted, schema.field(i).data_type())?;
    }
    Ok(RecordBatch::try_new(schema.clone(), columns)?)
}
