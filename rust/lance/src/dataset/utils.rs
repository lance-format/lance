// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{Error, Result};
use arrow_array::{
    Array, ArrayRef, RecordBatch, RecordBatchIterator, RecordBatchReader, UInt64Array,
};
use arrow_schema::{
    DataType, Field as ArrowField, Schema as ArrowSchema, SchemaRef as ArrowSchemaRef,
};
use datafusion::error::Result as DFResult;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use lance_arrow::json::{
    arrow_json_to_lance_json, convert_json_columns, convert_lance_json_to_arrow,
    has_arrow_json_fields, has_json_fields, lance_json_to_arrow_json,
};
use lance_core::{ROW_ADDR, ROW_ID};
use lance_table::rowids::{RowIdIndex, RowIdSequence};
use roaring::RoaringTreemap;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::mpsc::Receiver;

/// Which column a capture stream consumes, and how its values accumulate.
#[derive(Debug, Clone, Copy)]
pub enum RowCapture {
    /// Capture `_rowid`. `stable` follows the dataset's row id feature: with
    /// stable row ids the values accumulate as a `RowIdSequence`, otherwise as
    /// addresses.
    RowId { stable: bool },
    /// Capture `_rowaddr`, always accumulated as addresses. Deletion flows need
    /// only the addresses of the removed rows for their deletion vectors, so
    /// capturing those directly keeps them out of the row id domain: the capture
    /// needs no row id index to translate ids back to addresses.
    RowAddr,
}

impl RowCapture {
    fn column(&self) -> &'static str {
        match self {
            Self::RowId { .. } => ROW_ID,
            Self::RowAddr => ROW_ADDR,
        }
    }

    fn accumulator(&self) -> CapturedRowIds {
        match self {
            Self::RowId { stable } => CapturedRowIds::new(*stable),
            Self::RowAddr => CapturedRowIds::AddressSet(RoaringTreemap::new()),
        }
    }
}

fn capture_and_project(
    captured: &mut CapturedRowIds,
    batch: RecordBatch,
    column: &str,
    value_idx: usize,
    output_projection: &[usize],
) -> DFResult<RecordBatch> {
    let values_arr = batch.column(value_idx);
    let values = values_arr
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(format!(
                "{column} had an unexpected type: {}",
                values_arr.data_type()
            ))
        })?;
    if values.null_count() > 0 {
        // Both capture columns are nullable, and `values()` reads the buffer
        // without consulting validity, so a null would be captured as some
        // unrelated address. Deletion vectors are built straight from these,
        // so refuse the batch instead of deleting whatever that address is.
        return Err(datafusion::error::DataFusionError::Execution(format!(
            "{column} had {} null values in a capture stream",
            values.null_count()
        )));
    }
    captured.capture(values.values())?;
    Ok(batch.project(output_projection)?)
}

/// Given a stream carrying the column named by `capture`, return a stream that
/// captures that column's values and drops it from the output. At completion of
/// the stream the captured values can be received from the returned receiver.
///
/// A `RowId` capture without stable row ids accumulates with
/// `RoaringTreemap::append`, which rejects values that do not arrive ascending;
/// see `CapturedRowIds::AddressStyle` for why compaction wants that. A `RowAddr`
/// capture only needs the set, so any order will do, and a stable-row-id capture
/// keeps arrival order, which the new fragments' row id sequences are built from.
pub fn make_row_capture_stream(
    mut target: SendableRecordBatchStream,
    capture: RowCapture,
) -> Result<(SendableRecordBatchStream, Receiver<CapturedRowIds>)> {
    let mut captured = capture.accumulator();
    let column = capture.column();

    let (tx, rx) = std::sync::mpsc::channel();

    let schema = target.schema();
    let (value_idx, _) = schema.column_with_name(column).ok_or_else(|| {
        Error::internal(format!(
            "A capture stream needs a `{column}` column, but none of the stream's {} columns is it",
            schema.fields().len()
        ))
    })?;
    let output_cols = (0..schema.fields.len())
        .filter(|col| *col != value_idx)
        .collect::<Vec<_>>();
    let output_schema = Arc::new(schema.project(&output_cols)?);

    let stream = futures::stream::poll_fn(move |cx| match target.poll_next_unpin(cx) {
        std::task::Poll::Ready(Some(Ok(batch))) => {
            let res = capture_and_project(&mut captured, batch, column, value_idx, &output_cols);
            std::task::Poll::Ready(Some(res))
        }
        std::task::Poll::Ready(Some(Err(err))) => std::task::Poll::Ready(Some(Err(err))),
        std::task::Poll::Ready(None) => {
            let captured_out = std::mem::replace(&mut captured, capture.accumulator());
            tx.send(captured_out).unwrap();
            std::task::Poll::Ready(None)
        }
        std::task::Poll::Pending => std::task::Poll::Pending,
    });

    let stream = RecordBatchStreamAdapter::new(output_schema, stream);

    Ok((Box::pin(stream), rx))
}

#[derive(Debug)]
pub enum CapturedRowIds {
    /// Addresses that arrive ascending, enforced with `append`. Compaction is
    /// the caller that needs the check: it writes its rows in capture order and
    /// then recovers that order by iterating the treemap, which is ascending —
    /// so for the pairing to hold, arrival order has to be ascending too.
    /// Update and merge insert accumulate here as well, and only have to stay
    /// within the check rather than depending on it.
    AddressStyle(RoaringTreemap),
    /// Addresses collected as a set, with no ordering requirement. Deletion
    /// flows want only the set of removed rows, and a plan that reads rows the
    /// index's way rather than the fragments' hands them over unordered.
    AddressSet(RoaringTreemap),
    SequenceStyle(RowIdSequence),
}

impl CapturedRowIds {
    pub fn new(stable_row_ids: bool) -> Self {
        if stable_row_ids {
            Self::SequenceStyle(RowIdSequence::new())
        } else {
            Self::AddressStyle(RoaringTreemap::new())
        }
    }

    pub fn capture(&mut self, row_ids: &[u64]) -> DFResult<()> {
        match self {
            Self::AddressStyle(ids) => {
                // Not just the cheap path: compaction writes rows in capture
                // order and recovers the pairing by iterating the treemap
                // ascending, so accepting a reordered batch here would remap
                // index entries onto the wrong rows. Let `append` catch it.
                ids.append(row_ids.iter().cloned())
                    .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
            }
            Self::AddressSet(ids) => {
                ids.extend(row_ids.iter().cloned());
            }
            Self::SequenceStyle(sequence) => {
                sequence.extend(row_ids.into());
            }
        }
        Ok(())
    }

    pub fn row_id_sequence(&self) -> Option<&RowIdSequence> {
        match self {
            Self::SequenceStyle(sequence) => Some(sequence),
            _ => None,
        }
    }

    pub fn row_addrs(&self, index: Option<&RowIdIndex>) -> Result<Cow<'_, RoaringTreemap>> {
        match self {
            Self::AddressStyle(addrs) | Self::AddressSet(addrs) => Ok(Cow::Borrowed(addrs)),
            Self::SequenceStyle(sequence) => {
                let mut treemap = RoaringTreemap::new();
                let Some(index) = index else {
                    panic!("RowIdIndex required for sequence style row ids")
                };
                for row_id in sequence.iter() {
                    treemap.insert(
                        index
                            .get(row_id)?
                            .expect("row id missing from index")
                            .into(),
                    );
                }
                Ok(Cow::Owned(treemap))
            }
        }
    }
}

/// Returns the physical field for a view type, or `None` if no conversion is needed.
fn physical_field(field: &ArrowField) -> Option<ArrowField> {
    match field.data_type() {
        DataType::Utf8View => Some(
            ArrowField::new(field.name(), DataType::Utf8, field.is_nullable())
                .with_metadata(field.metadata().clone()),
        ),
        DataType::BinaryView => Some(
            ArrowField::new(field.name(), DataType::Binary, field.is_nullable())
                .with_metadata(field.metadata().clone()),
        ),
        _ => None,
    }
}

/// Cast `Utf8View`/`BinaryView` columns in a batch to their classic offset equivalents.
fn downcast_view_columns(
    batch: &RecordBatch,
) -> std::result::Result<RecordBatch, arrow_schema::ArrowError> {
    let schema = batch.schema();
    let mut new_fields: Vec<ArrowField> = Vec::with_capacity(schema.fields().len());
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    let mut changed = false;

    for (i, field) in schema.fields().iter().enumerate() {
        if let Some(phys) = physical_field(field) {
            changed = true;
            new_columns.push(arrow_cast::cast(
                batch.column(i).as_ref(),
                phys.data_type(),
            )?);
            new_fields.push(phys);
        } else {
            new_columns.push(batch.column(i).clone());
            new_fields.push(field.as_ref().clone());
        }
    }

    if !changed {
        return Ok(batch.clone());
    }

    RecordBatch::try_new(
        Arc::new(ArrowSchema::new_with_metadata(
            new_fields,
            schema.metadata().clone(),
        )),
        new_columns,
    )
}

/// Adapter around the existing JSON and view-type conversion utilities.
#[derive(Debug, Clone)]
pub struct SchemaAdapter {
    logical_schema: ArrowSchemaRef,
}

impl SchemaAdapter {
    /// Create a new adapter given the logical Arrow schema.
    pub fn new(logical_schema: ArrowSchemaRef) -> Self {
        Self { logical_schema }
    }

    /// Determine if the logical schema includes fields that require physical conversion.
    pub fn requires_physical_conversion(&self) -> bool {
        self.logical_schema
            .fields()
            .iter()
            .any(|field| has_arrow_json_fields(field) || physical_field(field).is_some())
    }

    /// Determine if the physical schema includes Lance JSON fields that must be converted back.
    pub fn requires_logical_conversion(schema: &ArrowSchemaRef) -> bool {
        schema.fields().iter().any(|field| has_json_fields(field))
    }

    pub fn to_physical_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        if self.requires_physical_conversion() {
            let batch = convert_json_columns(&batch)?;
            Ok(downcast_view_columns(&batch)?)
        } else {
            Ok(batch)
        }
    }

    /// Build the physical Arrow schema for `logical_schema`: Arrow JSON fields
    /// become Lance JSON fields and view types are downcast to their classic
    /// offset equivalents. Fields needing no conversion are left unchanged.
    fn physical_schema(logical_schema: &ArrowSchemaRef) -> ArrowSchemaRef {
        let mut new_fields = Vec::with_capacity(logical_schema.fields().len());
        for field in logical_schema.fields() {
            if has_arrow_json_fields(field) {
                new_fields.push(Arc::new(arrow_json_to_lance_json(field)));
            } else if let Some(phys) = physical_field(field) {
                new_fields.push(Arc::new(phys));
            } else {
                new_fields.push(Arc::clone(field));
            }
        }
        Arc::new(ArrowSchema::new_with_metadata(
            new_fields,
            logical_schema.metadata().clone(),
        ))
    }

    /// Wrap a synchronous [`RecordBatchReader`] so each batch is converted from
    /// its logical form to the physical form Lance stores on disk (Arrow JSON →
    /// Lance JSON, view types → offset types). Returns the reader unchanged when
    /// no field needs conversion.
    pub fn to_physical_reader(
        &self,
        reader: Box<dyn RecordBatchReader + Send>,
    ) -> Box<dyn RecordBatchReader + Send> {
        if !self.requires_physical_conversion() {
            return reader;
        }
        let schema = Self::physical_schema(&reader.schema());
        let converted = reader.map(|batch| {
            let batch = batch?;
            let batch = convert_json_columns(&batch)?;
            downcast_view_columns(&batch)
        });
        Box::new(RecordBatchIterator::new(converted, schema))
    }

    /// Convert a logical stream into a physical stream.
    pub fn to_physical_stream(
        &self,
        stream: SendableRecordBatchStream,
    ) -> SendableRecordBatchStream {
        if !self.requires_physical_conversion() {
            return stream;
        }

        let converted_schema = Self::physical_schema(&stream.schema());

        let converted_stream = stream.map(move |batch_result| {
            batch_result.and_then(|batch| {
                let batch = convert_json_columns(&batch).map_err(|e| {
                    datafusion::error::DataFusionError::ArrowError(Box::new(e), None)
                })?;
                downcast_view_columns(&batch)
                    .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))
            })
        });

        Box::pin(RecordBatchStreamAdapter::new(
            converted_schema,
            converted_stream,
        ))
    }

    /// Convert a physical stream into a logical stream.
    pub fn to_logical_stream(
        &self,
        stream: SendableRecordBatchStream,
    ) -> SendableRecordBatchStream {
        if !Self::requires_logical_conversion(&stream.schema()) {
            return stream;
        }

        let arrow_schema = stream.schema();
        let mut new_fields = Vec::with_capacity(arrow_schema.fields().len());
        for field in arrow_schema.fields() {
            if has_json_fields(field) {
                new_fields.push(lance_json_to_arrow_json(field));
            } else {
                new_fields.push(field.as_ref().clone());
            }
        }
        let converted_schema = Arc::new(ArrowSchema::new_with_metadata(
            new_fields,
            arrow_schema.metadata().clone(),
        ));

        let converted_stream = stream.map(move |batch_result| {
            batch_result.and_then(|batch| {
                convert_lance_json_to_arrow(&batch).map_err(|e| {
                    datafusion::error::DataFusionError::ArrowError(
                        Box::new(arrow_schema::ArrowError::InvalidArgumentError(
                            e.to_string(),
                        )),
                        None,
                    )
                })
            })
        });

        Box::pin(RecordBatchStreamAdapter::new(
            converted_schema,
            converted_stream,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_accumulators_differ_on_order() {
        // Each capture picks its accumulator, and that pairing is the contract
        // the two tests below rest on.
        assert!(matches!(
            RowCapture::RowAddr.accumulator(),
            CapturedRowIds::AddressSet(_)
        ));
        assert!(matches!(
            RowCapture::RowId { stable: false }.accumulator(),
            CapturedRowIds::AddressStyle(_)
        ));
        // A stable-row-id capture has to reach the sequence: every consumer of
        // one reads it through `row_id_sequence()`, which returns `None` for the
        // address variants, so getting this wrong is a silent skip rather than
        // an error.
        assert!(matches!(
            RowCapture::RowId { stable: true }.accumulator(),
            CapturedRowIds::SequenceStyle(_)
        ));

        // The delete flow's accumulator takes addresses in any order...
        let mut set = CapturedRowIds::AddressSet(RoaringTreemap::new());
        set.capture(&[5, 3]).unwrap();
        assert_eq!(
            set.row_addrs(None).unwrap().iter().collect::<Vec<_>>(),
            vec![3, 5]
        );

        // ...while compaction's insists on ascending input, because it pairs
        // the captured addresses positionally with the rows it wrote.
        let mut ordered = CapturedRowIds::AddressStyle(RoaringTreemap::new());
        assert!(ordered.capture(&[5, 3]).is_err());
    }

    #[test]
    fn capture_refuses_null_addresses() {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new(ROW_ADDR, DataType::UInt64, true),
            ArrowField::new("x", DataType::UInt64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(UInt64Array::from(vec![Some(0), None])),
                Arc::new(UInt64Array::from(vec![Some(1), Some(2)])),
            ],
        )
        .unwrap();

        // The null here sits over a 0 in the values buffer, so an unguarded
        // capture would delete fragment 0 row 0.
        let mut captured = CapturedRowIds::AddressSet(RoaringTreemap::new());
        let err = capture_and_project(&mut captured, batch, ROW_ADDR, 0, &[1]).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(ROW_ADDR) && msg.contains("null"),
            "error should name the column and the nulls, got: {err}"
        );
        // The check runs before the capture, so nothing reached the accumulator.
        assert!(captured.row_addrs(None).unwrap().is_empty());
    }
}
