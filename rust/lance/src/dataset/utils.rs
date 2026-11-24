// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::Result;
use arrow_array::{RecordBatch, UInt64Array};
use arrow_schema::{Schema as ArrowSchema, SchemaRef as ArrowSchemaRef};
use datafusion::physical_plan::SendableRecordBatchStream as DataFusionRecordBatchStream;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter as DataFusionStreamAdapter;
use futures::StreamExt;
use lance_arrow::json::{convert_lance_json_to_arrow, has_json_fields, lance_json_to_arrow_json};
use lance_core::{Error, ROW_CREATED_AT_VERSION, ROW_ID};
use lance_io::stream::{RecordBatchStreamAdapter, SendableRecordBatchStream};
use lance_table::format::{RowDatasetVersionRun, RowDatasetVersionSequence};
use lance_table::rowids::segment::U64Segment;
use lance_table::rowids::{RowIdIndex, RowIdSequence};
use roaring::RoaringTreemap;
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::mpsc::Receiver;

fn u64_values<'a>(batch: &'a RecordBatch, column: usize, what: &str) -> &'a [u64] {
    let array = batch.column(column);
    array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap_or_else(|| panic!("{what} had an unexpected type: {}", array.data_type()))
        .values()
}

fn extract_row_ids(
    row_ids: &mut CapturedRowIds,
    batch: RecordBatch,
    row_id_idx: usize,
    created_at_idx: Option<usize>,
    data_projection: &[usize],
) -> Result<RecordBatch> {
    row_ids.capture(u64_values(&batch, row_id_idx, "Row ids"))?;
    if let Some(created_at_idx) = created_at_idx {
        row_ids.capture_created_at(u64_values(&batch, created_at_idx, "Created-at versions"));
    }
    Ok(batch.project(data_projection)?)
}

/// Given a stream that includes a row id column, return a stream that will
/// capture the row id. At completion of the stream, the captured row ids can
/// be received from the returned receiver.
///
/// A `_row_created_at_version` column, if the stream carries one, is captured
/// alongside the row ids and removed from the output the same way.
pub fn make_rowid_capture_stream(
    mut target: SendableRecordBatchStream,
    stable_row_ids: bool,
) -> Result<(SendableRecordBatchStream, Receiver<CapturedRowIds>)> {
    let mut row_ids = CapturedRowIds::new(stable_row_ids);

    let (tx, rx) = std::sync::mpsc::channel();

    let schema = target.schema();
    let (row_id_idx, _) = schema
        .column_with_name(ROW_ID)
        .expect("Received a batch without row ids");
    let created_at_idx = schema
        .column_with_name(ROW_CREATED_AT_VERSION)
        .map(|(idx, _)| idx);
    // Started here, rather than on the first batch, so a stream that carries
    // the column but no rows still reports an empty capture.
    if created_at_idx.is_some()
        && let CapturedRowIds::SequenceStyle { created_at, .. } = &mut row_ids
    {
        *created_at = Some(RowDatasetVersionSequence::new());
    }
    let data_cols = (0..schema.fields.len())
        .filter(|col| *col != row_id_idx && Some(*col) != created_at_idx)
        .collect::<Vec<_>>();
    let output_schema = Arc::new(schema.project(&data_cols)?);

    let stream = futures::stream::poll_fn(move |cx| match target.poll_next_unpin(cx) {
        std::task::Poll::Ready(Some(Ok(batch))) => {
            let res = extract_row_ids(&mut row_ids, batch, row_id_idx, created_at_idx, &data_cols);
            std::task::Poll::Ready(Some(res))
        }
        std::task::Poll::Ready(Some(Err(err))) => std::task::Poll::Ready(Some(Err(err))),
        std::task::Poll::Ready(None) => {
            let row_ids_out = std::mem::take(&mut row_ids);
            tx.send(row_ids_out).unwrap();
            std::task::Poll::Ready(None)
        }
        std::task::Poll::Pending => std::task::Poll::Pending,
    });

    let stream = RecordBatchStreamAdapter::new(output_schema, stream);

    Ok((Box::pin(stream), rx))
}

#[derive(Debug)]
pub enum CapturedRowIds {
    AddressStyle(RoaringTreemap),
    SequenceStyle {
        row_ids: RowIdSequence,
        /// The created-at versions of the captured rows, in capture order,
        /// when the stream carried them; `None` when it did not.
        created_at: Option<RowDatasetVersionSequence>,
    },
}

impl CapturedRowIds {
    pub fn new(stable_row_ids: bool) -> Self {
        if stable_row_ids {
            Self::SequenceStyle {
                row_ids: RowIdSequence::new(),
                created_at: None,
            }
        } else {
            Self::AddressStyle(RoaringTreemap::new())
        }
    }

    pub fn capture(&mut self, row_ids: &[u64]) -> Result<()> {
        match self {
            Self::AddressStyle(ids) => {
                // Assume they are sorted
                ids.append(row_ids.iter().cloned())
                    .map_err(|e| Error::execution(e.to_string()))?;
            }
            Self::SequenceStyle {
                row_ids: sequence, ..
            } => {
                sequence.extend(row_ids.into());
            }
        }
        Ok(())
    }

    /// Record the created-at versions of the rows just passed to [`Self::capture`].
    pub fn capture_created_at(&mut self, versions: &[u64]) {
        let Self::SequenceStyle {
            created_at: Some(sequence),
            ..
        } = self
        else {
            return;
        };
        // Run-length encoded as they arrive, and a run that carries on from
        // the previous batch is extended rather than restarted: a run per
        // batch would bloat the inline metadata of a large rewrite.
        for run in versions.chunk_by(|a, b| a == b) {
            let (version, rows) = (run[0], run.len() as u64);
            let start = match sequence.runs.last_mut() {
                Some(RowDatasetVersionRun {
                    span: U64Segment::Range(span),
                    version: last_version,
                }) => {
                    if *last_version == version {
                        span.end += rows;
                        continue;
                    }
                    span.end
                }
                // Every run pushed below spans a range, so this is the first.
                _ => 0,
            };
            sequence.runs.push(RowDatasetVersionRun {
                span: U64Segment::Range(start..start + rows),
                version,
            });
        }
    }

    pub fn row_id_sequence(&self) -> Option<&RowIdSequence> {
        match self {
            Self::SequenceStyle { row_ids, .. } => Some(row_ids),
            _ => None,
        }
    }

    /// The captured rows' created-at versions, in capture order, when the
    /// stream carried them.
    pub fn created_at_sequence(&self) -> Option<&RowDatasetVersionSequence> {
        match self {
            Self::SequenceStyle { created_at, .. } => created_at.as_ref(),
            Self::AddressStyle(_) => None,
        }
    }

    pub fn row_addrs(&self, index: Option<&RowIdIndex>) -> Result<Cow<'_, RoaringTreemap>> {
        match self {
            Self::AddressStyle(addrs) => Ok(Cow::Borrowed(addrs)),
            Self::SequenceStyle {
                row_ids: sequence, ..
            } => {
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

impl Default for CapturedRowIds {
    fn default() -> Self {
        Self::AddressStyle(RoaringTreemap::new())
    }
}

pub use lance_table_runtime::write::SchemaAdapter;

fn requires_logical_conversion(schema: &ArrowSchemaRef) -> bool {
    schema.fields().iter().any(|field| has_json_fields(field))
}

/// Convert a physical stream into a logical stream.
pub fn to_logical_stream(stream: DataFusionRecordBatchStream) -> DataFusionRecordBatchStream {
    if !requires_logical_conversion(&stream.schema()) {
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

    Box::pin(DataFusionStreamAdapter::new(
        converted_schema,
        converted_stream,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_created_at_extends_a_run_across_batches() {
        let mut captured = CapturedRowIds::SequenceStyle {
            row_ids: RowIdSequence::new(),
            created_at: Some(RowDatasetVersionSequence::new()),
        };
        captured.capture_created_at(&[1, 1, 2]);
        captured.capture_created_at(&[]);
        captured.capture_created_at(&[2, 3]);
        // One run per version, the one split by the batch boundary included,
        // with spans absolute over the whole capture.
        assert_eq!(
            captured.created_at_sequence(),
            Some(&RowDatasetVersionSequence::from_versions(&[1, 1, 2, 2, 3]))
        );

        // A stream that did not carry the column captures nothing.
        let mut not_carried = CapturedRowIds::new(true);
        not_carried.capture_created_at(&[1, 2]);
        assert_eq!(not_carried.created_at_sequence(), None);
    }
}
