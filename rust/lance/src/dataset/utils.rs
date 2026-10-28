// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::{Dataset, Result};
use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, RecordBatchReader, UInt64Array};
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
use lance_core::{ROW_CREATED_AT_VERSION, ROW_ID};
use lance_table::format::{RowDatasetVersionRun, RowDatasetVersionSequence};
use lance_table::rowids::RowIdSequence;
use lance_table::rowids::segment::U64Segment;
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
) -> DFResult<RecordBatch> {
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

    pub fn capture(&mut self, row_ids: &[u64]) -> DFResult<()> {
        match self {
            Self::AddressStyle(ids) => {
                // Assume they are sorted
                ids.append(row_ids.iter().cloned())
                    .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;
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

    /// Resolve captured IDs as a set of live addresses. Positional lookups use
    /// bounded batches, and uncovered fragments are scanned once for the set.
    pub async fn row_addrs(&self, dataset: &Dataset) -> Result<Cow<'_, RoaringTreemap>> {
        match self {
            Self::AddressStyle(addrs) => Ok(Cow::Borrowed(addrs)),
            Self::SequenceStyle {
                row_ids: sequence, ..
            } => Ok(Cow::Owned(
                dataset.resolve_captured_row_ids(sequence).await?,
            )),
        }
    }
}

impl Default for CapturedRowIds {
    fn default() -> Self {
        Self::AddressStyle(RoaringTreemap::new())
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

/// Converts between the Arrow representations callers use and the ones Lance
/// stores: Arrow JSON text ↔ Lance JSONB, and top-level view arrays → offset
/// arrays.
///
/// Dataset writes convert once, in the data file writer
/// ([`V2WriterAdapter`](super::write::V2WriterAdapter)). The physical methods
/// are otherwise only for combining caller data with values read back from
/// storage before that point.
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
    use crate::dataset::WriteParams;
    use crate::dataset::builder::DatasetBuilder;
    use crate::dataset::optimize::{CompactionOptions, compact_files};
    use crate::dataset::rowids::{
        INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, SPILL_ROW_LINEAGE_CONFIG_KEY,
    };
    use crate::index::DatasetIndexExt;
    use crate::utils::test::{DatagenExt, FragmentCount, FragmentRowCount};
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Int32Type, UInt64Type};
    use lance_core::utils::address::RowAddress;
    use lance_core::utils::tempfile::TempStrDir;
    use lance_core::{Error, ROW_ADDR};
    use lance_index::{IndexType, scalar::ScalarIndexParams};
    use lance_table::format::RowIdMeta;
    use lance_table::rowids::write_row_ids;

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

    #[rstest::rstest]
    #[case::unindexed(0)]
    #[case::partial(1)]
    #[case::full(2)]
    #[tokio::test]
    async fn captured_row_ids_scan_uncovered_fragments_once(#[case] indexed_fragments: usize) {
        let directory = TempStrDir::default();
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_dataset_with_params(
                &directory,
                FragmentCount::from(4),
                FragmentRowCount::from(5000),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 5000,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        dataset
            .update_config([
                (SPILL_ROW_LINEAGE_CONFIG_KEY, "true"),
                (INLINE_ROW_LINEAGE_MAX_BYTES_CONFIG_KEY, "0"),
            ])
            .await
            .unwrap();
        compact_files(
            &mut dataset,
            CompactionOptions {
                target_rows_per_fragment: 10000,
                defer_index_remap: false,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(dataset.fragments().len(), 2);
        assert!(
            dataset
                .fragments()
                .iter()
                .all(|f| f.row_id_meta == Some(RowIdMeta::Column))
        );
        if indexed_fragments != 0 {
            let fragments = dataset.fragments()[..indexed_fragments]
                .iter()
                .map(|fragment| fragment.id as u32)
                .collect();
            dataset
                .create_index_builder(&[ROW_ID], IndexType::BTree, &ScalarIndexParams::default())
                .fragments(fragments)
                .await
                .unwrap();
        }
        dataset.delete("i = 7 OR i = 10007").await.unwrap();
        let oracle = dataset
            .scan()
            .project::<&str>(&[])
            .unwrap()
            .with_row_id()
            .with_row_address()
            .try_into_batch()
            .await
            .unwrap();
        let ids = oracle[ROW_ID].as_primitive::<UInt64Type>();
        let addresses = oracle[ROW_ADDR].as_primitive::<UInt64Type>();
        let expected: RoaringTreemap = addresses.values().iter().copied().collect();
        let single_id = ids.value(ids.len() - 1);
        let single_address = addresses.value(addresses.len() - 1);
        let mut all_ids: Vec<_> = ids.values().iter().rev().copied().collect();
        all_ids.extend([single_id, single_id]);
        let mut data_reads = Vec::new();
        for requested in [vec![single_id], all_ids] {
            // Fresh sessions make metadata caching independent of request size.
            let reader = DatasetBuilder::from_uri(&directory).load().await.unwrap();
            reader.object_store.io_stats_incremental();
            let is_single = requested.len() == 1;
            let captured = CapturedRowIds::SequenceStyle {
                row_ids: RowIdSequence::from(requested.as_slice()),
                created_at: None,
            };
            let actual = captured.row_addrs(&reader).await.unwrap().into_owned();
            assert_eq!(
                actual,
                if is_single {
                    RoaringTreemap::from_iter([single_address])
                } else {
                    expected.clone()
                }
            );
            let mut reads = reader
                .object_store
                .io_stats_incremental()
                .requests
                .into_iter()
                .filter(|request| request.path.as_ref().contains("/data/"))
                .map(|request| {
                    (
                        request.path.to_string(),
                        request.range.map(|r| (r.start, r.end)),
                    )
                })
                .collect::<Vec<_>>();
            reads.sort_unstable();
            data_reads.push(reads);
        }
        assert_eq!(
            data_reads[0], data_reads[1],
            "larger captures must not rescan uncovered data"
        );
        assert_eq!(data_reads[0].is_empty(), indexed_fragments == 2);
        for missing_id in [7, 20000] {
            let captured = CapturedRowIds::SequenceStyle {
                row_ids: RowIdSequence::from(&[missing_id][..]),
                created_at: None,
            };
            let error = captured.row_addrs(&dataset).await.unwrap_err();
            assert!(matches!(error, Error::Internal { .. }));
            assert!(
                error
                    .to_string()
                    .contains(&format!("Captured row ID {missing_id} has no live address"))
            );
        }
    }

    #[rstest::rstest]
    #[case::ambiguous(false)]
    #[case::deleted_original(true)]
    #[tokio::test]
    async fn captured_row_ids_detect_duplicates_across_coverage(
        #[case] has_deleted_original: bool,
    ) {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(3),
                FragmentRowCount::from(4),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 4,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        if has_deleted_original {
            dataset.delete("i = 2").await.unwrap();
        }
        // The first fragment has an inline range; the nonmonotonic last
        // fragment requires a scan. Row ID 2 crosses the two resolution paths.
        let manifest = Arc::make_mut(&mut dataset.manifest);
        Arc::make_mut(&mut manifest.fragments)[2].row_id_meta = Some(RowIdMeta::Inline(
            write_row_ids(&RowIdSequence::from(&[2, 9, 8, 11][..])).into(),
        ));
        let captured = CapturedRowIds::SequenceStyle {
            row_ids: RowIdSequence::from(&[2, 8, 2, 11][..]),
            created_at: None,
        };
        let result = captured.row_addrs(&dataset).await;
        if has_deleted_original {
            assert_eq!(
                result.unwrap().into_owned(),
                RoaringTreemap::from_iter(
                    [0, 2, 3].map(|offset| u64::from(RowAddress::new_from_parts(2, offset)))
                )
            );
        } else {
            let error = result.unwrap_err();
            assert!(matches!(error, Error::InvalidInput { .. }));
            assert!(error.to_string().contains("not unique"));
        }
    }

    #[tokio::test]
    async fn captured_row_ids_empty_and_address_domain() {
        let mut dataset = lance_datagen::gen_batch()
            .col("i", lance_datagen::array::step::<Int32Type>())
            .into_ram_dataset_with_params(
                FragmentCount::from(1),
                FragmentRowCount::from(1),
                Some(WriteParams {
                    enable_stable_row_ids: true,
                    max_rows_per_file: 1,
                    ..Default::default()
                }),
            )
            .await
            .unwrap();
        dataset.delete("true").await.unwrap();
        let mut captured = CapturedRowIds::new(true);
        assert!(captured.row_addrs(&dataset).await.unwrap().is_empty());
        captured.capture(&[0]).unwrap();
        let error = captured.row_addrs(&dataset).await.unwrap_err();
        assert!(matches!(error, Error::Internal { .. }));
        assert!(
            error
                .to_string()
                .contains("Captured row ID 0 has no live address")
        );

        let physical = CapturedRowIds::AddressStyle(RoaringTreemap::from_iter([7, 9]));
        assert!(matches!(
            physical.row_addrs(&dataset).await.unwrap(),
            Cow::Borrowed(_)
        ));
    }
}
